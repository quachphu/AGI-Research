    {}

    >>> class Something(dict, CantSympify):
    ...     pass
    ...
    >>> sympify(Something())
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: {}

    """

    __slots__ = ()


def _is_numpy_instance(a):
    """
    Checks if an object is an instance of a type from the numpy module.
    """
    # This check avoids unnecessarily importing NumPy.  We check the whole
    # __mro__ in case any base type is a numpy type.
    return any(type_.__module__ == 'numpy'
               for type_ in type(a).__mro__)


def _convert_numpy_types(a, **sympify_args):
    """
    Converts a numpy datatype input to an appropriate SymPy type.
    """
    import numpy as np
    if not isinstance(a, np.floating):
        if np.iscomplex(a):
            return _sympy_converter[complex](a.item())
        else:
            return sympify(a.item(), **sympify_args)
    else:
        from .numbers import Float
        prec = np.finfo(a).nmant + 1
        # E.g. double precision means prec=53 but nmant=52
        # Leading bit of mantissa is always 1, so is not stored
        if np.isposinf(a):
            return Float('inf')
        elif np.isneginf(a):
            return Float('-inf')
        else:
            p, q = a.as_integer_ratio()
            a = mlib.from_rational(p, q, prec)
            return Float(a, precision=prec)


@overload
def sympify(a: int, *, strict: bool = False) -> Integer: ... # type: ignore
@overload
def sympify(a: float, *, strict: bool = False) -> Float: ...
@overload
def sympify(a: Expr | complex, *, strict: bool = False) -> Expr: ...
@overload
def sympify(a: Tbasic, *, strict: bool = False) -> Tbasic: ...
@overload
def sympify(a: Any, *, strict: bool = False) -> Basic: ...

def sympify(a, locals=None, convert_xor=True, strict=False, rational=False,
        evaluate=None):
    """
    Converts an arbitrary expression to a type that can be used inside SymPy.

    Explanation
    ===========

    It will convert Python ints into instances of :class:`~.Integer`, floats
    into instances of :class:`~.Float`, etc. It is also able to coerce
    symbolic expressions which inherit from :class:`~.Basic`. This can be
    useful in cooperation with SAGE.

    .. warning::
        Note that this function uses ``eval``, and thus shouldn't be used on
        unsanitized input.

    If the argument is already a type that SymPy understands, it will do
    nothing but return that value. This can be used at the beginning of a
    function to ensure you are working with the correct type.

    Examples
    ========

    >>> from sympy import sympify

    >>> sympify(2).is_integer
    True
    >>> sympify(2).is_real
    True

    >>> sympify(2.0).is_real
    True
    >>> sympify("2.0").is_real
    True
    >>> sympify("2e-45").is_real
    True

    If the expression could not be converted, a SympifyError is raised.

    >>> sympify("x***2")
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: "could not parse 'x***2'"

    When attempting to parse non-Python syntax using ``sympify``, it raises a
    ``SympifyError``:

    >>> sympify("2x+1")
    Traceback (most recent call last):
    ...
    SympifyError: Sympify of expression 'could not parse '2x+1'' failed

    To parse non-Python syntax, use ``parse_expr`` from ``sympy.parsing.sympy_parser``.

    >>> from sympy.parsing.sympy_parser import parse_expr
    >>> parse_expr("2x+1", transformations="all")
    2*x + 1

    For more details about ``transformations``: see :func:`~sympy.parsing.sympy_parser.parse_expr`

    Locals
    ------

    The sympification happens with access to everything that is loaded
    by ``from sympy import *``; anything used in a string that is not
    defined by that import will be converted to a symbol. In the following,
    the ``bitcount`` function is treated as a symbol and the ``O`` is
    interpreted as the :class:`~.Order` object (used with series) and it raises
    an error when used improperly:

    >>> s = 'bitcount(42)'
    >>> sympify(s)
    bitcount(42)
    >>> sympify("O(x)")
    O(x)
    >>> sympify("O + 1")
    Traceback (most recent call last):
    ...
    TypeError: unbound method...

    In order to have ``bitcount`` be recognized it can be imported into a
    namespace dictionary and passed as locals:

    >>> ns = {}
    >>> exec('from sympy.core.evalf import bitcount', ns)
    >>> sympify(s, locals=ns)
    6

    In order to have the ``O`` interpreted as a Symbol, identify it as such
    in the namespace dictionary. This can be done in a variety of ways; all
    three of the following are possibilities:

    >>> from sympy import Symbol
    >>> ns["O"] = Symbol("O")  # method 1
    >>> exec('from sympy.abc import O', ns)  # method 2
    >>> ns.update(dict(O=Symbol("O")))  # method 3
    >>> sympify("O + 1", locals=ns)
    O + 1

    If you want *all* single-letter and Greek-letter variables to be symbols
    then you can use the clashing-symbols dictionaries that have been defined
    there as private variables: ``_clash1`` (single-letter variables),
    ``_clash2`` (the multi-letter Greek names) or ``_clash`` (both single and
    multi-letter names that are defined in ``abc``).

    >>> from sympy.abc import _clash1
    >>> set(_clash1)  # if this fails, see issue #23903
    {'E', 'I', 'N', 'O', 'Q', 'S'}
    >>> sympify('I & Q', _clash1)
    I & Q

    Strict
    ------

    If the option ``strict`` is set to ``True``, only the types for which an
    explicit conversion has been defined are converted. In the other
    cases, a SympifyError is raised.

    >>> print(sympify(None))
    None
    >>> sympify(None, strict=True)
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: None

    .. deprecated:: 1.6

       ``sympify(obj)`` automatically falls back to ``str(obj)`` when all
       other conversion methods fail, but this is deprecated. ``strict=True``
       will disable this deprecated behavior. See
       :ref:`deprecated-sympify-string-fallback`.

    Evaluation
    ----------

    If the option ``evaluate`` is set to ``False``, then arithmetic and
    operators will be converted into their SymPy equivalents and the
    ``evaluate=False`` option will be added. Nested ``Add`` or ``Mul`` will
    be denested first. This is done via an AST transformation that replaces
    operators with their SymPy equivalents, so if an operand redefines any
    of those operations, the redefined operators will not be used. If
    argument a is not a string, the mathematical expression is evaluated
    before being passed to sympify, so adding ``evaluate=False`` will still
    return the evaluated result of expression.

    >>> sympify('2**2 / 3 + 5')
    19/3
    >>> sympify('2**2 / 3 + 5', evaluate=False)
    2**2/3 + 5
    >>> sympify('4/2+7', evaluate=True)
    9
    >>> sympify('4/2+7', evaluate=False)
    4/2 + 7
    >>> sympify(4/2+7, evaluate=False)
    9.00000000000000

    Extending
    ---------

    To extend ``sympify`` to convert custom objects (not derived from ``Basic``),
    just define a ``_sympy_`` method to your class. You can do that even to
    classes that you do not own by subclassing or adding the method at runtime.

    >>> from sympy import Matrix
    >>> class MyList1(object):
    ...     def __iter__(self):
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    ...     def _sympy_(self): return Matrix(self)
    >>> sympify(MyList1())
    Matrix([
    [1],
    [2]])

    If you do not have control over the class definition you could also use the
    ``converter`` global dictionary. The key is the class and the value is a
    function that takes a single argument and returns the desired SymPy
    object, e.g. ``converter[MyList] = lambda x: Matrix(x)``.

    >>> class MyList2(object):   # XXX Do not do this if you control the class!
    ...     def __iter__(self):  #     Use _sympy_!
    ...         yield 1
    ...         yield 2
    ...         return
    ...     def __getitem__(self, i): return list(self)[i]
    >>> from sympy.core.sympify import converter
    >>> converter[MyList2] = lambda x: Matrix(x)
    >>> sympify(MyList2())
    Matrix([
    [1],
    [2]])

    Notes
    =====

    The keywords ``rational`` and ``convert_xor`` are only used
    when the input is a string.

    convert_xor
    -----------

    >>> sympify('x^y',convert_xor=True)
    x**y
    >>> sympify('x^y',convert_xor=False)
    x ^ y

    rational
    --------

    >>> sympify('0.1',rational=False)
    0.1
    >>> sympify('0.1',rational=True)
    1/10

    Sometimes autosimplification during sympification results in expressions
    that are very different in structure than what was entered. Until such
    autosimplification is no longer done, the ``kernS`` function might be of
    some use. In the example below you can see how an expression reduces to
    $-1$ by autosimplification, but does not do so when ``kernS`` is used.

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x
    >>> -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1
    -1
    >>> s = '-2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1'
    >>> sympify(s)
    -1
    >>> kernS(s)
    -2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x))) - 1

    Parameters
    ==========

    a :
        - any object defined in SymPy
        - standard numeric Python types: ``int``, ``long``, ``float``, ``Decimal``
        - strings (like ``"0.09"``, ``"2e-19"`` or ``'sin(x)'``)
        - booleans, including ``None`` (will leave ``None`` unchanged)
        - dicts, lists, sets or tuples containing any of the above

    convert_xor : bool, optional
        If true, treats ``^`` as exponentiation.
        If False, treats ``^`` as XOR itself.
        Used only when input is a string.

    locals : any object defined in SymPy, optional
        In order to have strings be recognized it can be imported
        into a namespace dictionary and passed as locals.

    strict : bool, optional
        If the option strict is set to ``True``, only the types for which
        an explicit conversion has been defined are converted. In the
        other cases, a SympifyError is raised.

    rational : bool, optional
        If ``True``, converts floats into :class:`~.Rational`.
        If ``False``, it lets floats remain as it is.
        Used only when input is a string.

    evaluate : bool, optional
        If False, then arithmetic and operators will be converted into
        their SymPy equivalents. If True the expression will be evaluated
        and the result will be returned.

    """
    # XXX: If a is a Basic subclass rather than instance (e.g. sin rather than
    # sin(x)) then a.__sympy__ will be the property. Only on the instance will
    # a.__sympy__ give the *value* of the property (True). Since sympify(sin)
    # was used for a long time we allow it to pass. However if strict=True as
    # is the case in internal calls to _sympify then we only allow
    # is_sympy=True.
    #
    # https://github.com/sympy/sympy/issues/20124
    is_sympy = getattr(a, '__sympy__', None)
    if is_sympy is True:
        return a
    elif is_sympy is not None:
        if not strict:
            return a
        else:
            raise SympifyError(a)

    if isinstance(a, CantSympify):
        raise SympifyError(a)

    cls = getattr(a, "__class__", None)

    #Check if there exists a converter for any of the types in the mro
    for superclass in getmro(cls):
        #First check for user defined converters
        conv = _external_converter.get(superclass)
        if conv is None:
            #if none exists, check for SymPy defined converters
            conv = _sympy_converter.get(superclass)
        if conv is not None:
            return conv(a)

    if cls is type(None):
        if strict:
            raise SympifyError(a)
        else:
            return a

    if evaluate is None:
        evaluate = global_parameters.evaluate

    # Support for basic numpy datatypes
    if _is_numpy_instance(a):
        import numpy as np
        if np.isscalar(a):
            return _convert_numpy_types(a, locals=locals,
                convert_xor=convert_xor, strict=strict, rational=rational,
                evaluate=evaluate)

    _sympy_ = getattr(a, "_sympy_", None)
    if _sympy_ is not None:
        return a._sympy_()

    if not strict:
        # Put numpy array conversion _before_ float/int, see
        # <https://github.com/sympy/sympy/issues/13924>.
        flat = getattr(a, "flat", None)
        if flat is not None:
            shape = getattr(a, "shape", None)
            if shape is not None:
                from sympy.tensor.array import Array
                return Array(a.flat, a.shape)  # works with e.g. NumPy arrays

    if not isinstance(a, str):
        if _is_numpy_instance(a):
            import numpy as np
            assert not isinstance(a, np.number)
            if isinstance(a, np.ndarray):
                # Scalar arrays (those with zero dimensions) have sympify
                # called on the scalar element.
                if a.ndim == 0:
                    try:
                        return sympify(a.item(),
                                       locals=locals,
                                       convert_xor=convert_xor,
                                       strict=strict,
                                       rational=rational,
                                       evaluate=evaluate)
                    except SympifyError:
                        pass
        elif hasattr(a, '__float__'):
            # float and int can coerce size-one numpy arrays to their lone
            # element.  See issue https://github.com/numpy/numpy/issues/10404.
            return sympify(float(a))
        elif hasattr(a, '__int__'):
            return sympify(int(a))

    if strict:
        raise SympifyError(a)

    if iterable(a):
        try:
            return type(a)([sympify(x, locals=locals, convert_xor=convert_xor,
                rational=rational, evaluate=evaluate) for x in a])
        except TypeError:
            # Not all iterables are rebuildable with their type.
            pass

    if not isinstance(a, str):
        raise SympifyError('cannot sympify object of type %r' % type(a))

    from sympy.parsing.sympy_parser import (parse_expr, TokenError,
                                            standard_transformations)
    from sympy.parsing.sympy_parser import convert_xor as t_convert_xor
    from sympy.parsing.sympy_parser import rationalize as t_rationalize

    transformations = standard_transformations

    if rational:
        transformations += (t_rationalize,)
    if convert_xor:
        transformations += (t_convert_xor,)

    try:
        a = a.replace('\n', '')
        expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
    except (TokenError, SyntaxError) as exc:
        raise SympifyError('could not parse %r' % a, exc)

    return expr


def _sympify(a):
    """
    Short version of :func:`~.sympify` for internal usage for ``__add__`` and
    ``__eq__`` methods where it is ok to allow some things (like Python
    integers and floats) in the expression. This excludes things (like strings)
    that are unwise to allow into such an expression.

    >>> from sympy import Integer
    >>> Integer(1) == 1
    True

    >>> Integer(1) == '1'
    False

    >>> from sympy.abc import x
    >>> x + 1
    x + 1

    >>> x + '1'
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for +: 'Symbol' and 'str'

    see: sympify

    """
    return sympify(a, strict=True)


def kernS(s):
    """Use a hack to try keep autosimplification from distributing a
    a number into an Add; this modification does not
    prevent the 2-arg Mul from becoming an Add, however.

    Examples
    ========

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x, y

    The 2-arg Mul distributes a number (or minus sign) across the terms
    of an expression, but kernS will prevent that:

    >>> 2*(x + y), -(x + 1)
    (2*x + 2*y, -x - 1)
    >>> kernS('2*(x + y)')
    2*(x + y)
    >>> kernS('-(x + 1)')
    -(x + 1)

    If use of the hack fails, the un-hacked string will be passed to sympify...
    and you get what you get.

    XXX This hack should not be necessary once issue 4596 has been resolved.
    """
    hit = False
    quoted = '"' in s or "'" in s
    if '(' in s and not quoted:
        if s.count('(') != s.count(")"):
            raise SympifyError('unmatched left parenthesis')

        # strip all space from s
        s = ''.join(s.split())
        olds = s
        # now use space to represent a symbol that
        # will
        # step 1. turn potential 2-arg Muls into 3-arg versions
        # 1a. *( -> * *(
        s = s.replace('*(', '* *(')
        # 1b. close up exponentials
        s = s.replace('** *', '**')
        # 2. handle the implied multiplication of a negated
        # parenthesized expression in two steps
        # 2a:  -(...)  -->  -( *(...)
        target = '-( *('
        s = s.replace('-(', target)
        # 2b: double the matching closing parenthesis
        # -( *(...)  -->  -( *(...))
        i = nest = 0
        assert target.endswith('(')  # assumption below
        while True:
            j = s.find(target, i)
            if j == -1:
                break
            j += len(target) - 1
            for j in range(j, len(s)):
                if s[j] == "(":
                    nest += 1
                elif s[j] == ")":
                    nest -= 1
                if nest == 0:
                    break
            s = s[:j] + ")" + s[j:]
            i = j + 2  # the first char after 2nd )
        if ' ' in s:
            # get a unique kern
            kern = '_'
            while kern in s:
                kern += choice(string.ascii_letters + string.digits)
            s = s.replace(' ', kern)
            hit = kern in s
        else:
            hit = False

    for i in range(2):
        try:
            expr = sympify(s)
            break
        except TypeError:  # the kern might cause unknown errors...
            if hit:
                s = olds  # maybe it didn't like the kern; use un-kerned s
                hit = False
                continue
            expr = sympify(s)  # let original error raise

    if not hit:
        return expr

    from .symbol import Symbol
    rep = {Symbol(kern): 1}
    def _clear(expr):
        if isinstance(expr, (list, tuple, set)):
            return type(expr)([_clear(e) for e in expr])
        if hasattr(expr, 'subs'):
            return expr.subs(rep, hack2=True)
        return expr
    expr = _clear(expr)
    # hope that kern is not there anymore
    return expr


# Avoid circular import
from .basic import Basic
<|endoftext|>"""
When you need to use random numbers in SymPy library code, import from here
so there is only one generator working for SymPy. Imports from here should
behave the same as if they were being imported from Python's random module.
But only the routines currently used in SymPy are included here. To use others
import ``rng`` and access the method directly. For example, to capture the
current state of the generator use ``rng.getstate()``.

There is intentionally no Random to import from here. If you want
to control the state of the generator, import ``seed`` and call it
with or without an argument to set the state.

Examples
========

>>> from sympy.core.random import random, seed
>>> assert random() < 1
>>> seed(1); a = random()
>>> b = random()
>>> seed(1); c = random()
>>> assert a == c
>>> assert a != b  # remote possibility this will fail

"""
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int

import random as _random
rng = _random.Random()

choice = rng.choice
random = rng.random
randint = rng.randint
randrange = rng.randrange
sample = rng.sample
# seed = rng.seed
shuffle = rng.shuffle
uniform = rng.uniform

_assumptions_rng = _random.Random()
_assumptions_shuffle = _assumptions_rng.shuffle


def seed(a=None, version=2):
    rng.seed(a=a, version=version)
    _assumptions_rng.seed(a=a, version=version)


def random_complex_number(a=2, b=-1, c=3, d=1, rational=False, tolerance=None):
    """
    Return a random complex number.

    To reduce chance of hitting branch cuts or anything, we guarantee
    b <= Im z <= d, a <= Re z <= c

    When rational is True, a rational approximation to a random number
    is obtained within specified tolerance, if any.
    """
    from sympy.core.numbers import I
    from sympy.simplify.simplify import nsimplify
    A, B = uniform(a, c), uniform(b, d)
    if not rational:
        return A + I*B
    return (nsimplify(A, rational=True, tolerance=tolerance) +
        I*nsimplify(B, rational=True, tolerance=tolerance))


def verify_numerically(f, g, z=None, tol=1.0e-6, a=2, b=-1, c=3, d=1):
    """
    Test numerically that f and g agree when evaluated in the argument z.

    If z is None, all symbols will be tested. This routine does not test
    whether there are Floats present with precision higher than 15 digits
    so if there are, your results may not be what you expect due to round-
    off errors.

    Examples
    ========

    >>> from sympy import sin, cos
    >>> from sympy.abc import x
    >>> from sympy.core.random import verify_numerically as tn
    >>> tn(sin(x)**2 + cos(x)**2, 1, x)
    True
    """
    from sympy.core.symbol import Symbol
    from sympy.core.sympify import sympify
    from sympy.core.numbers import comp
    f, g = (sympify(i) for i in (f, g))
    if z is None:
        z = f.free_symbols | g.free_symbols
    elif isinstance(z, Symbol):
        z = [z]
    reps = list(zip(z, [random_complex_number(a, b, c, d) for _ in z]))
    z1 = f.subs(reps).n()
    z2 = g.subs(reps).n()
    return comp(z1, z2, tol)


def test_derivative_numerically(f, z, tol=1.0e-6, a=2, b=-1, c=3, d=1):
    """
    Test numerically that the symbolically computed derivative of f
    with respect to z is correct.

    This routine does not test whether there are Floats present with
    precision higher than 15 digits so if there are, your results may
    not be what you expect due to round-off errors.

    Examples
    ========

    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> from sympy.core.random import test_derivative_numerically as td
    >>> td(sin(x), x)
    True
    """
    from sympy.core.numbers import comp
    from sympy.core.function import Derivative
    z0 = random_complex_number(a, b, c, d)
    f1 = f.diff(z).subs(z, z0)
    f2 = Derivative(f, z).doit_numerically(z0)
    return comp(f1.n(), f2.n(), tol)


def _randrange(seed=None):
    """Return a randrange generator.

    ``seed`` can be

    * None - return randomly seeded generator
    * int - return a generator seeded with the int
    * list - the values to be returned will be taken from the list
      in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.core.random import _randrange
    >>> rr = _randrange()
    >>> rr(1000) # doctest: +SKIP
    999
    >>> rr = _randrange(3)
    >>> rr(1000) # doctest: +SKIP
    238
    >>> rr = _randrange([0, 5, 1, 3, 4])
    >>> rr(3), rr(3)
    (0, 1)
    """
    if seed is None:
        return randrange
    elif isinstance(seed, int):
        rng.seed(seed)
        return randrange
    elif is_sequence(seed):
        seed = list(seed)  # make a copy
        seed.reverse()

        def give(a, b=None, seq=seed):
            if b is None:
                a, b = 0, a
            a, b = as_int(a), as_int(b)
            w = b - a
            if w < 1:
                raise ValueError('_randrange got empty range')
            try:
                x = seq.pop()
            except IndexError:
                raise ValueError('_randrange sequence was too short')
            if a <= x < b:
                return x
            else:
                return give(a, b, seq)
        return give
    else:
        raise ValueError('_randrange got an unexpected seed')


def _randint(seed=None):
    """Return a randint generator.

    ``seed`` can be

    * None - return randomly seeded generator
    * int - return a generator seeded with the int
    * list - the values to be returned will be taken from the list
      in the order given; the provided list is not modified.

    Examples
    ========

    >>> from sympy.core.random import _randint
    >>> ri = _randint()
    >>> ri(1, 1000) # doctest: +SKIP
    999
    >>> ri = _randint(3)
    >>> ri(1, 1000) # doctest: +SKIP
    238
    >>> ri = _randint([0, 5, 1, 2, 4])
    >>> ri(1, 3), ri(1, 3)
    (1, 2)
    """
    if seed is None:
        return randint
    elif isinstance(seed, int):
        rng.seed(seed)
        return randint
    elif is_sequence(seed):
        seed = list(seed)  # make a copy
        seed.reverse()

        def give(a, b, seq=seed):
            a, b = as_int(a), as_int(b)
            w = b - a
            if w < 0:
                raise ValueError('_randint got empty range')
            try:
                x = seq.pop()
            except IndexError:
                raise ValueError('_randint sequence was too short')
            if a <= x <= b:
                return x
            else:
                return give(a, b, seq)
        return give
    else:
        raise ValueError('_randint got an unexpected seed')
<|endoftext|>"""Module for SymPy containers

    (SymPy objects that store other SymPy objects)

    The containers implemented in this module are subclassed to Basic.
    They are supposed to work seamlessly within the SymPy framework.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableSet
from typing import Any, Callable

from .basic import Basic
from .sorting import default_sort_key, ordered
from .sympify import _sympify, sympify, _sympy_converter, SympifyError
from sympy.core.kind import Kind
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int


class Tuple(Basic):
    """
    Wrapper around the builtin tuple object.

    Explanation
    ===========

    The Tuple is a subclass of Basic, so that it works well in the
    SymPy framework.  The wrapped tuple is available as self.args, but
    you can also access elements or slices with [:] syntax.

    Parameters
    ==========

    sympify : bool
        If ``False``, ``sympify`` is not called on ``args``. This
        can be used for speedups for very large tuples where the
        elements are known to already be SymPy objects.

    Examples
    ========

    >>> from sympy import Tuple, symbols
    >>> a, b, c, d = symbols('a b c d')
    >>> Tuple(a, b, c)[1:]
    (b, c)
    >>> Tuple(a, b, c).subs(a, d)
    (d, b, c)

    """

    def __new__(cls, *args, **kwargs):
        if kwargs.get('sympify', True):
            args = (sympify(arg) for arg in args)
        obj = Basic.__new__(cls, *args)
        return obj

    def __getitem__(self, i):
        if isinstance(i, slice):
            indices = i.indices(len(self))
            return Tuple(*(self.args[j] for j in range(*indices)))
        return self.args[i]

    def __len__(self):
        return len(self.args)

    def __contains__(self, item):
        return item in self.args

    def __iter__(self):
        return iter(self.args)

    def __add__(self, other):
        if isinstance(other, Tuple):
            return Tuple(*(self.args + other.args))
        elif isinstance(other, tuple):
            return Tuple(*(self.args + other))
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Tuple):
            return Tuple(*(other.args + self.args))
        elif isinstance(other, tuple):
            return Tuple(*(other + self.args))
        else:
            return NotImplemented

    def __mul__(self, other):
        try:
            n = as_int(other)
        except ValueError:
            raise TypeError("Can't multiply sequence by non-integer of type '%s'" % type(other))
        return self.func(*(self.args*n))

    __rmul__ = __mul__

    def __eq__(self, other):
        if isinstance(other, Basic):
            return super().__eq__(other)
        return self.args == other

    def __ne__(self, other):
        if isinstance(other, Basic):
            return super().__ne__(other)
        return self.args != other

    def __hash__(self):
        return hash(self.args)

    def _to_mpmath(self, prec):
        return tuple(a._to_mpmath(prec) for a in self.args)

    def __lt__(self, other):
        return _sympify(self.args < other.args)

    def __le__(self, other):
        return _sympify(self.args <= other.args)

    # XXX: Basic defines count() as something different, so we can't
    # redefine it here. Originally this lead to cse() test failure.
    def tuple_count(self, value) -> int:
        """Return number of occurrences of value."""
        return self.args.count(value)

    def index(self, value, start=None, stop=None):
        """Searches and returns the first index of the value."""
        # XXX: One would expect:
        #
        # return self.args.index(value, start, stop)
        #
        # here. Any trouble with that? Yes:
        #
        # >>> (1,).index(1, None, None)
        # Traceback (most recent call last):
        #   File "<stdin>", line 1, in <module>
        # TypeError: slice indices must be integers or None or have an __index__ method
        #
        # See: http://bugs.python.org/issue13340

        if start is None and stop is None:
            return self.args.index(value)
        elif stop is None:
            return self.args.index(value, start)
        else:
            return self.args.index(value, start, stop)

    @property
    def kind(self):
        """
        The kind of a Tuple instance.

        The kind of a Tuple is always of :class:`TupleKind` but
        parametrised by the number of elements and the kind of each element.

        Examples
        ========

        >>> from sympy import Tuple, Matrix
        >>> Tuple(1, 2).kind
        TupleKind(NumberKind, NumberKind)
        >>> Tuple(Matrix([1, 2]), 1).kind
        TupleKind(MatrixKind(NumberKind), NumberKind)
        >>> Tuple(1, 2).kind.element_kind
        (NumberKind, NumberKind)

        See Also
        ========

        sympy.matrices.kind.MatrixKind
        sympy.core.kind.NumberKind
        """
        return TupleKind(*(i.kind for i in self.args))

_sympy_converter[tuple] = lambda tup: Tuple(*tup)





def tuple_wrapper(method):
    """
    Decorator that converts any tuple in the function arguments into a Tuple.

    Explanation
    ===========

    The motivation for this is to provide simple user interfaces.  The user can
    call a function with regular tuples in the argument, and the wrapper will
    convert them to Tuples before handing them to the function.

    Explanation
    ===========

    >>> from sympy.core.containers import tuple_wrapper
    >>> def f(*args):
    ...    return args
    >>> g = tuple_wrapper(f)

    The decorated function g sees only the Tuple argument:

    >>> g(0, (1, 2), 3)
    (0, (1, 2), 3)

    """
    def wrap_tuples(*args, **kw_args):
        newargs = []
        for arg in args:
            if isinstance(arg, tuple):
                newargs.append(Tuple(*arg))
            else:
                newargs.append(arg)
        return method(*newargs, **kw_args)
    return wrap_tuples


class Dict(Basic):
    """
    Wrapper around the builtin dict object.

    Explanation
    ===========

    The Dict is a subclass of Basic, so that it works well in the
    SymPy framework.  Because it is immutable, it may be included
    in sets, but its values must all be given at instantiation and
    cannot be changed afterwards.  Otherwise it behaves identically
    to the Python dict.

    Examples
    ========

    >>> from sympy import Dict, Symbol

    >>> D = Dict({1: 'one', 2: 'two'})
    >>> for key in D:
    ...    if key == 1:
    ...        print('%s %s' % (key, D[key]))
    1 one

    The args are sympified so the 1 and 2 are Integers and the values
    are Symbols. Queries automatically sympify args so the following work:

    >>> 1 in D
    True
    >>> D.has(Symbol('one')) # searches keys and values
    True
    >>> 'one' in D # not in the keys
    False
    >>> D[1]
    one

    """

    elements: frozenset[Tuple]
    _dict: dict[Basic, Basic]

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (dict, Dict)):
            items = [Tuple(k, v) for k, v in args[0].items()]
        elif iterable(args) and all(len(arg) == 2 for arg in args):
            items = [Tuple(k, v) for k, v in args]
        else:
            raise TypeError('Pass Dict args as Dict((k1, v1), ...) or Dict({k1: v1, ...})')
        elements = frozenset(items)
        obj = Basic.__new__(cls, *ordered(items))
        obj.elements = elements
        obj._dict = dict(items)  # In case Tuple decides it wants to sympify
        return obj

    def __getitem__(self, key):
        """x.__getitem__(y) <==> x[y]"""
        try:
            key = _sympify(key)
        except SympifyError:
            raise KeyError(key)

        return self._dict[key]

    def __setitem__(self, key, value):
        raise NotImplementedError("SymPy Dicts are Immutable")

    def items(self):
        '''Returns a set-like object providing a view on dict's items.
        '''
        return self._dict.items()

    def keys(self):
        '''Returns the list of the dict's keys.'''
        return self._dict.keys()

    def values(self):
        '''Returns the list of the dict's values.'''
        return self._dict.values()

    def __iter__(self):
        '''x.__iter__() <==> iter(x)'''
        return iter(self._dict)

    def __len__(self):
        '''x.__len__() <==> len(x)'''
        return self._dict.__len__()

    def get(self, key, default=None):
        '''Returns the value for key if the key is in the dictionary.'''
        try:
            key = _sympify(key)
        except SympifyError:
            return default
        return self._dict.get(key, default)

    def __contains__(self, key):
        '''D.__contains__(k) -> True if D has a key k, else False'''
        try:
            key = _sympify(key)
        except SympifyError:
            return False
        return key in self._dict

    def __lt__(self, other):
        return _sympify(self.args < other.args)

    @property
    def _sorted_args(self):
        return tuple(sorted(self.args, key=default_sort_key))

    def __eq__(self, other):
        if isinstance(other, dict):
            return self == Dict(other)
        return super().__eq__(other)

    __hash__ : Callable[[Basic], Any] = Basic.__hash__

# this handles dict, defaultdict, OrderedDict
_sympy_converter[dict] = lambda d: Dict(*d.items())

class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        if iterable:
            self.map = OrderedDict((item, None) for item in iterable)
        else:
            self.map = OrderedDict()

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        self.map[key] = None

    def discard(self, key):
        self.map.pop(key)

    def pop(self, last=True):
        return self.map.popitem(last=last)[0]

    def __iter__(self):
        yield from self.map.keys()

    def __repr__(self):
        if not self.map:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.map.keys()))

    def intersection(self, other):
        return self.__class__([val for val in self if val in other])

    def difference(self, other):
        return self.__class__([val for val in self if val not in other])

    def update(self, iterable):
        for val in iterable:
            self.add(val)

class TupleKind(Kind):
    """
    TupleKind is a subclass of Kind, which is used to define Kind of ``Tuple``.

    Parameters of TupleKind will be kinds of all the arguments in Tuples, for
    example

    Parameters
    ==========

    args : tuple(element_kind)
       element_kind is kind of element.
       args is tuple of kinds of element

    Examples
    ========

    >>> from sympy import Tuple
    >>> Tuple(1, 2).kind
    TupleKind(NumberKind, NumberKind)
    >>> Tuple(1, 2).kind.element_kind
    (NumberKind, NumberKind)

    See Also
    ========

    sympy.core.kind.NumberKind
    MatrixKind
    sympy.sets.sets.SetKind
    """
    def __new__(cls, *args):
        obj = super().__new__(cls, *args)
        obj.element_kind = args
        return obj

    def __repr__(self):
        return "TupleKind{}".format(self.element_kind)
<|endoftext|>"""
Base class to provide str and repr hooks that `init_printing` can overwrite.

This is exposed publicly in the `printing.defaults` module,
but cannot be defined there without causing circular imports.
"""

class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way. Natively Python uses ``repr()``
    even if ``str()`` was explicitly requested. Mix in this trait into
    a class to get proper default printing.

    This also adds support for LaTeX printing in jupyter notebooks.
    """

    # Since this class is used as a mixin we set empty slots. That means that
    # instances of any subclasses that use slots will not need to have a
    # __dict__.
    __slots__ = ()

    # Note, we always use the default ordering (lex) in __str__ and __repr__,
    # regardless of the global setting. See issue 5487.
    def __str__(self):
        from sympy.printing.str import sstr
        return sstr(self, order=None)

    __repr__ = __str__

    def _repr_disabled(self):
        """
        No-op repr function used to disable jupyter display hooks.

        When :func:`sympy.init_printing` is used to disable certain display
        formats, this function is copied into the appropriate ``_repr_*_``
        attributes.

        While we could just set the attributes to `None``, doing it this way
        allows derived classes to call `super()`.
        """
        return None

    # We don't implement _repr_png_ here because it would add a large amount of
    # data to any notebook containing SymPy expressions, without adding
    # anything useful to the notebook. It can still enabled manually, e.g.,
    # for the qtconsole, with init_printing().
    _repr_png_ = _repr_disabled

    _repr_svg_ = _repr_disabled

    def _repr_latex_(self):
        """
        IPython/Jupyter LaTeX printing

        To change the behavior of this (e.g., pass in some settings to LaTeX),
        use init_printing(). init_printing() will also enable LaTeX printing
        for built in numeric types like ints and container types that contain
        SymPy objects, like lists and dictionaries of expressions.
        """
        from sympy.printing.latex import latex
        s = latex(self, mode='plain')
        return "$\\displaystyle %s$" % s
<|endoftext|>"""
Adaptive numerical evaluation of SymPy expressions, using mpmath
for mathematical functions.
"""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Any, overload, Type

import math

import mpmath.libmp as libmp
from mpmath import (
    make_mpc, make_mpf, mp, mpc, mpf, nsum, quadts, quadosc, workprec)
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
                          fnan, finf, fninf, fnone, fone, fzero, mpf_abs, mpf_add,
                          mpf_atan, mpf_atan2, mpf_cmp, mpf_cos, mpf_e, mpf_exp, mpf_log, mpf_lt,
                          mpf_mul, mpf_neg, mpf_pi, mpf_pow, mpf_pow_int, mpf_shift, mpf_sin,
                          mpf_sqrt, normalize, round_nearest, to_int, to_str, mpf_tan)
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps

from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int

if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    from sympy.core.power import Pow
    from sympy.core.symbol import Symbol
    from sympy.integrals.integrals import Integral
    from sympy.concrete.summations import Sum
    from sympy.concrete.products import Product
    from sympy.functions.elementary.exponential import exp, log
    from sympy.functions.elementary.complexes import Abs, re, im
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.trigonometric import atan
    from .numbers import Float, Rational, Integer, AlgebraicNumber, Number

LG10 = math.log2(10)
rnd = round_nearest


def bitcount(n):
    """Return smallest integer, b, such that |n|/2**b < 1.
    """
    return mpmath_bitcount(abs(int(n)))

# Used in a few places as placeholder values to denote exponents and
# precision levels, e.g. of exact numbers. Must be careful to avoid
# passing these to mpmath functions or returning them in final results.
INF = float(mpmath_inf)
MINUS_INF = float(-mpmath_inf)

# ~= 100 digits. Real men set this to INF.
DEFAULT_MAXPREC = 333


class PrecisionExhausted(ArithmeticError):
    pass

#----------------------------------------------------------------------------#
#                                                                            #
#              Helper functions for arithmetic and complex parts             #
#                                                                            #
#----------------------------------------------------------------------------#

"""
An mpf value tuple is a tuple of integers (sign, man, exp, bc)
representing a floating-point number: [1, -1][sign]*man*2**exp where
sign is 0 or 1 and bc should correspond to the number of bits used to
represent the mantissa (man) in binary notation, e.g.
"""

MPF_TUP = tuple[int, int, int, int]  # mpf value tuple

"""
Explanation
===========

>>> from sympy.core.evalf import bitcount
>>> sign, man, exp, bc = 0, 5, 1, 3
>>> n = [1, -1][sign]*man*2**exp
>>> n, bitcount(man)
(10, 3)

A temporary result is a tuple (re, im, re_acc, im_acc) where
re and im are nonzero mpf value tuples representing approximate
numbers, or None to denote exact zeros.

re_acc, im_acc are integers denoting log2(e) where e is the estimated
relative accuracy of the respective complex part, but may be anything
if the corresponding complex part is None.

"""
TMP_RES = Any  # temporary result, should be some variant of
# tUnion[tTuple[Optional[MPF_TUP], Optional[MPF_TUP],
#               Optional[int], Optional[int]],
#        'ComplexInfinity']
# but mypy reports error because it doesn't know as we know
# 1. re and re_acc are either both None or both MPF_TUP
# 2. sometimes the result can't be zoo

# type of the "options" parameter in internal evalf functions
OPT_DICT = dict[str, Any]


def fastlog(x: MPF_TUP | None) -> int | Any:
    """Fast approximation of log2(x) for an mpf value tuple x.

    Explanation
    ===========

    Calculated as exponent + width of mantissa. This is an
    approximation for two reasons: 1) it gives the ceil(log2(abs(x)))
    value and 2) it is too high by 1 in the case that x is an exact
    power of 2. Although this is easy to remedy by testing to see if
    the odd mpf mantissa is 1 (indicating that one was dealing with
    an exact power of 2) that would decrease the speed and is not
    necessary as this is only being used as an approximation for the
    number of bits in x. The correct return value could be written as
    "x[2] + (x[3] if x[1] != 1 else 0)".
        Since mpf tuples always have an odd mantissa, no check is done
    to see if the mantissa is a multiple of 2 (in which case the
    result would be too large by 1).

    Examples
    ========

    >>> from sympy import log
    >>> from sympy.core.evalf import fastlog, bitcount
    >>> s, m, e = 0, 5, 1
    >>> bc = bitcount(m)
    >>> n = [1, -1][s]*m*2**e
    >>> n, (log(n)/log(2)).evalf(2), fastlog((s, m, e, bc))
    (10, 3.3, 4)
    """

    if not x or x == fzero:
        return MINUS_INF
    return x[2] + x[3]


def pure_complex(v: Expr, or_real=False) -> tuple[Number, Number] | None:
    """Return a and b if v matches a + I*b where b is not zero and
    a and b are Numbers, else None. If `or_real` is True then 0 will
    be returned for `b` if `v` is a real number.

    Examples
    ========

    >>> from sympy.core.evalf import pure_complex
    >>> from sympy import sqrt, I, S
    >>> a, b, surd = S(2), S(3), sqrt(2)
    >>> pure_complex(a)
    >>> pure_complex(a, or_real=True)
    (2, 0)
    >>> pure_complex(surd)
    >>> pure_complex(a + b*I)
    (2, 3)
    >>> pure_complex(I)
    (0, 1)
    """
    h, t = v.as_coeff_Add()
    if t:
        c, i = t.as_coeff_Mul()
        if i is S.ImaginaryUnit:
            return h, c
    elif or_real:
        return h, S.Zero
    return None


# I don't know what this is, see function scaled_zero below
SCALED_ZERO_TUP = tuple[list[int], int, int, int]



@overload
def scaled_zero(mag: SCALED_ZERO_TUP, sign=1) -> MPF_TUP:
    ...
@overload
def scaled_zero(mag: int, sign=1) -> tuple[SCALED_ZERO_TUP, int]:
    ...
def scaled_zero(mag: SCALED_ZERO_TUP | int, sign=1) -> \
        MPF_TUP | tuple[SCALED_ZERO_TUP, int]:
    """Return an mpf representing a power of two with magnitude ``mag``
    and -1 for precision. Or, if ``mag`` is a scaled_zero tuple, then just
    remove the sign from within the list that it was initially wrapped
    in.

    Examples
    ========

    >>> from sympy.core.evalf import scaled_zero
    >>> from sympy import Float
    >>> z, p = scaled_zero(100)
    >>> z, p
    (([0], 1, 100, 1), -1)
    >>> ok = scaled_zero(z)
    >>> ok
    (0, 1, 100, 1)
    >>> Float(ok)
    1.26765060022823e+30
    >>> Float(ok, p)
    0.e+30
    >>> ok, p = scaled_zero(100, -1)
    >>> Float(scaled_zero(ok), p)
    -0.e+30
    """
    if isinstance(mag, tuple) and len(mag) == 4 and iszero(mag, scaled=True):
        return (mag[0][0],) + mag[1:]
    elif isinstance(mag, SYMPY_INTS):
        if sign not in [-1, 1]:
            raise ValueError('sign must be +/-1')
        rv, p = mpf_shift(fone, mag), -1
        s = 0 if sign == 1 else 1
        rv = ([s],) + rv[1:]
        return rv, p
    else:
        raise ValueError('scaled zero expects int or scaled_zero tuple.')


def iszero(mpf: MPF_TUP | SCALED_ZERO_TUP | None, scaled=False) -> bool | None:
    if not scaled:
        return not mpf or not mpf[1] and not mpf[-1]
    return mpf and isinstance(mpf[0], list) and mpf[1] == mpf[-1] == 1


def complex_accuracy(result: TMP_RES) -> int | Any:
    """
    Returns relative accuracy of a complex number with given accuracies
    for the real and imaginary parts. The relative accuracy is defined
    in the complex norm sense as ||z|+|error|| / |z| where error
    is equal to (real absolute error) + (imag absolute error)*i.

    The full expression for the (logarithmic) error can be approximated
    easily by using the max norm to approximate the complex norm.

    In the worst case (re and im equal), this is wrong by a factor
    sqrt(2), or by log2(sqrt(2)) = 0.5 bit.
    """
    if result is S.ComplexInfinity:
        return INF
    re, im, re_acc, im_acc = result
    if not im:
        if not re:
            return INF
        return re_acc
    if not re:
        return im_acc
    re_size = fastlog(re)
    im_size = fastlog(im)
    absolute_error = max(re_size - re_acc, im_size - im_acc)
    relative_error = absolute_error - max(re_size, im_size)
    return -relative_error


def get_abs(expr: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    result = evalf(expr, prec + 2, options)
    if result is S.ComplexInfinity:
        return finf, None, prec, None
    re, im, re_acc, im_acc = result
    if not re:
        re, re_acc, im, im_acc = im, im_acc, re, re_acc
    if im:
        if expr.is_number:
            abs_expr, _, acc, _ = evalf(abs(N(expr, prec + 2)),
                                        prec + 2, options)
            return abs_expr, None, acc, None
        else:
            if 'subs' in options:
                return libmp.mpc_abs((re, im), prec), None, re_acc, None
            return abs(expr), None, prec, None
    elif re:
        return mpf_abs(re), None, re_acc, None
    else:
        return None, None, None, None


def get_complex_part(expr: Expr, no: int, prec: int, options: OPT_DICT) -> TMP_RES:
    """no = 0 for real part, no = 1 for imaginary part"""
    workprec = prec
    i = 0
    while 1:
        res = evalf(expr, workprec, options)
        if res is S.ComplexInfinity:
            return fnan, None, prec, None
        value, accuracy = res[no::2]
        # XXX is the last one correct? Consider re((1+I)**2).n()
        if (not value) or accuracy >= prec or -value[2] > prec:
            return value, None, accuracy, None
        workprec += max(30, 2**i)
        i += 1


def evalf_abs(expr: 'Abs', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_abs(expr.args[0], prec, options)


def evalf_re(expr: 're', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_complex_part(expr.args[0], 0, prec, options)


def evalf_im(expr: 'im', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_complex_part(expr.args[0], 1, prec, options)


def finalize_complex(re: MPF_TUP, im: MPF_TUP, prec: int) -> TMP_RES:
    if re == fzero and im == fzero:
        raise ValueError("got complex zero with unknown accuracy")
    elif re == fzero:
        return None, im, None, prec
    elif im == fzero:
        return re, None, prec, None

    size_re = fastlog(re)
    size_im = fastlog(im)
    if size_re > size_im:
        re_acc = prec
        im_acc = prec + min(-(size_re - size_im), 0)
    else:
        im_acc = prec
        re_acc = prec + min(-(size_im - size_re), 0)
    return re, im, re_acc, im_acc


def chop_parts(value: TMP_RES, prec: int) -> TMP_RES:
    """
    Chop off tiny real or complex parts.
    """
    if value is S.ComplexInfinity:
        return value
    re, im, re_acc, im_acc = value
    # Method 1: chop based on absolute value
    if re and re not in _infs_nan and (fastlog(re) < -prec + 4):
        re, re_acc = None, None
    if im and im not in _infs_nan and (fastlog(im) < -prec + 4):
        im, im_acc = None, None
    # Method 2: chop if inaccurate and relatively small
    if re and im:
        delta = fastlog(re) - fastlog(im)
        if re_acc < 2 and (delta - re_acc <= -prec + 4):
            re, re_acc = None, None
        if im_acc < 2 and (delta - im_acc >= prec - 4):
            im, im_acc = None, None
    return re, im, re_acc, im_acc


def check_target(expr: Expr, result: TMP_RES, prec: int):
    a = complex_accuracy(result)
    if a < prec:
        raise PrecisionExhausted("Failed to distinguish the expression: \n\n%s\n\n"
            "from zero. Try simplifying the input, using chop=True, or providing "
            "a higher maxn for evalf" % (expr))


def get_integer_part(expr: Expr, no: int, options: OPT_DICT, return_ints=False) -> \
        TMP_RES | tuple[int, int]:
    """
    With no = 1, computes ceiling(expr)
    With no = -1, computes floor(expr)

    Note: this function either gives the exact result or signals failure.
    """
    from sympy.functions.elementary.complexes import re, im
    # The expression is likely less than 2^30 or so
    assumed_size = 30
    result = evalf(expr, assumed_size, options)
    if result is S.ComplexInfinity:
        raise ValueError("Cannot get integer part of Complex Infinity")
    ire, iim, ire_acc, iim_acc = result

    # We now know the size, so we can calculate how much extra precision
    # (if any) is needed to get within the nearest integer
    if ire and iim:
        gap = max(fastlog(ire) - ire_acc, fastlog(iim) - iim_acc)
    elif ire:
        gap = fastlog(ire) - ire_acc
    elif iim:
        gap = fastlog(iim) - iim_acc
    else:
        # ... or maybe the expression was exactly zero
        if return_ints:
            return 0, 0
        else:
            return None, None, None, None

    margin = 10

    if gap >= -margin:
        prec = margin + assumed_size + gap
        ire, iim, ire_acc, iim_acc = evalf(
            expr, prec, options)
    else:
        prec = assumed_size

    # We can now easily find the nearest integer, but to find floor/ceil, we
    # must also calculate whether the difference to the nearest integer is
    # positive or negative (which may fail if very close).
    def calc_part(re_im: Expr, nexpr: MPF_TUP):
        from .add import Add
        _, _, exponent, _ = nexpr
        is_int = exponent == 0
        nint = int(to_int(nexpr, rnd))
        if is_int:
            # make sure that we had enough precision to distinguish
            # between nint and the re or im part (re_im) of expr that
            # was passed to calc_part
            ire, iim, ire_acc, iim_acc = evalf(
                re_im - nint, 10, options)  # don't need much precision
            assert not iim
            size = -fastlog(ire) + 2  # -ve b/c ire is less than 1
            if size > prec:
                ire, iim, ire_acc, iim_acc = evalf(
                    re_im, size, options)
                assert not iim
                nexpr = ire
            nint = int(to_int(nexpr, rnd))
            _, _, new_exp, _ = ire
            is_int = new_exp == 0
        if not is_int:
            # if there are subs and they all contain integer re/im parts
            # then we can (hopefully) safely substitute them into the
            # expression
            s = options.get('subs', False)
            if s:
                # use strict=False with as_int because we take
                # 2.0 == 2
                def is_int_reim(x):
                    """Check for integer or integer + I*integer."""
                    try:
                        as_int(x, strict=False)
                        return True
                    except ValueError:
                        try:
                            [as_int(i, strict=False) for i in x.as_real_imag()]
                            return True
                        except ValueError:
                            return False

                if all(is_int_reim(v) for v in s.values()):
                    re_im = re_im.subs(s)

            re_im = Add(re_im, -nint, evaluate=False)
            x, _, x_acc, _ = evalf(re_im, 10, options)
            try:
                check_target(re_im, (x, None, x_acc, None), 3)
            except PrecisionExhausted:
                if not re_im.equals(0):
                    raise PrecisionExhausted
                x = fzero
            nint += int(no*(mpf_cmp(x or fzero, fzero) == no))
        nint = from_int(nint)
        return nint, INF

    re_, im_, re_acc, im_acc = None, None, None, None

    if ire is not None and ire != fzero:
        re_, re_acc = calc_part(re(expr, evaluate=False), ire)
    if iim is not None and iim != fzero:
        im_, im_acc = calc_part(im(expr, evaluate=False), iim)

    if return_ints:
        return int(to_int(re_ or fzero)), int(to_int(im_ or fzero))
    return re_, im_, re_acc, im_acc


def evalf_ceiling(expr: 'ceiling', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_integer_part(expr.args[0], 1, options)


def evalf_floor(expr: 'floor', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_integer_part(expr.args[0], -1, options)


def evalf_float(expr: 'Float', prec: int, options: OPT_DICT) -> TMP_RES:
    return expr._mpf_, None, prec, None


def evalf_rational(expr: 'Rational', prec: int, options: OPT_DICT) -> TMP_RES:
    return from_rational(expr.p, expr.q, prec), None, prec, None


def evalf_integer(expr: 'Integer', prec: int, options: OPT_DICT) -> TMP_RES:
    return from_int(expr.p, prec), None, prec, None

#----------------------------------------------------------------------------#
#                                                                            #
#                            Arithmetic operations                           #
#                                                                            #
#----------------------------------------------------------------------------#


def add_terms(terms: list, prec: int, target_prec: int) -> \
        tuple[MPF_TUP | SCALED_ZERO_TUP | None, int | None]:
    """
    Helper for evalf_add. Adds a list of (mpfval, accuracy) terms.

    Returns
    =======

    - None, None if there are no non-zero terms;
    - terms[0] if there is only 1 term;
    - scaled_zero if the sum of the terms produces a zero by cancellation
      e.g. mpfs representing 1 and -1 would produce a scaled zero which need
      special handling since they are not actually zero and they are purposely
      malformed to ensure that they cannot be used in anything but accuracy
      calculations;
    - a tuple that is scaled to target_prec that corresponds to the
      sum of the terms.

    The returned mpf tuple will be normalized to target_prec; the input
    prec is used to define the working precision.

    XXX explain why this is needed and why one cannot just loop using mpf_add
    """

    terms = [t for t in terms if not iszero(t[0])]
    if not terms:
        return None, None
    elif len(terms) == 1:
        return terms[0]

    # see if any argument is NaN or oo and thus warrants a special return
    special = []
    from .numbers import Float
    for t in terms:
        arg = Float._new(t[0], 1)
        if arg is S.NaN or arg.is_infinite:
            special.append(arg)
    if special:
        from .add import Add
        rv = evalf(Add(*special), prec + 4, {})
        return rv[0], rv[2]

    working_prec = 2*prec
    sum_man, sum_exp = 0, 0
    absolute_err: list[int] = []

    for x, accuracy in terms:
        sign, man, exp, bc = x
        if sign:
            man = -man
        absolute_err.append(bc + exp - accuracy)
        delta = exp - sum_exp
        if exp >= sum_exp:
            # x much larger than existing sum?
            # first: quick test
            if ((delta > working_prec) and
                ((not sum_man) or
                 delta - bitcount(abs(sum_man)) > working_prec)):
                sum_man = man
                sum_exp = exp
            else:
                sum_man += (man << delta)
        else:
            delta = -delta
            # x much smaller than existing sum?
            if delta - bc > working_prec:
                if not sum_man:
                    sum_man, sum_exp = man, exp
            else:
                sum_man = (sum_man << delta) + man
                sum_exp = exp
    absolute_error = max(absolute_err)
    if not sum_man:
        return scaled_zero(absolute_error)
    if sum_man < 0:
        sum_sign = 1
        sum_man = -sum_man
    else:
        sum_sign = 0
    sum_bc = bitcount(sum_man)
    sum_accuracy = sum_exp + sum_bc - absolute_error
    r = normalize(sum_sign, sum_man, sum_exp, sum_bc, target_prec,
        rnd), sum_accuracy
    return r


def evalf_add(v: 'Add', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        h, c = res
        re, _, re_acc, _ = evalf(h, prec, options)
        im, _, im_acc, _ = evalf(c, prec, options)
        return re, im, re_acc, im_acc

    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)

    i = 0
    target_prec = prec
    while 1:
        options['maxprec'] = min(oldmaxprec, 2*prec)

        terms = [evalf(arg, prec + 10, options) for arg in v.args]
        n = terms.count(S.ComplexInfinity)
        if n >= 2:
            return fnan, None, prec, None
        re, re_acc = add_terms(
            [a[0::2] for a in terms if isinstance(a, tuple) and a[0]], prec, target_prec)
        im, im_acc = add_terms(
            [a[1::2] for a in terms if isinstance(a, tuple) and a[1]], prec, target_prec)
        if n == 1:
            if re in (finf, fninf, fnan) or im in (finf, fninf, fnan):
                return fnan, None, prec, None
            return S.ComplexInfinity
        acc = complex_accuracy((re, im, re_acc, im_acc))
        if acc >= target_prec:
            if options.get('verbose'):
                print("ADD: wanted", target_prec, "accurate bits, got", re_acc, im_acc)
            break
        else:
            if (prec - target_prec) > options['maxprec']:
                break

            prec = prec + max(10 + 2**i, target_prec - acc)
            i += 1
            if options.get('verbose'):
                print("ADD: restarting with prec", prec)

    options['maxprec'] = oldmaxprec
    if iszero(re, scaled=True):
        re = scaled_zero(re)
    if iszero(im, scaled=True):
        im = scaled_zero(im)
    return re, im, re_acc, im_acc


def evalf_mul(v: 'Mul', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        # the only pure complex that is a mul is h*I
        _, h = res
        im, _, im_acc, _ = evalf(h, prec, options)
        return None, im, None, im_acc
    args = list(v.args)

    # see if any argument is NaN or oo and thus warrants a special return
    has_zero = False
    special = []
    from .numbers import Float
    for arg in args:
        result = evalf(arg, prec, options)
        if result is S.ComplexInfinity:
            special.append(result)
            continue
        if result[0] is None:
            if result[1] is None:
                has_zero = True
            continue
        num = Float._new(result[0], 1)
        if num is S.NaN:
            return fnan, None, prec, None
        if num.is_infinite:
            special.append(num)
    if special:
        if has_zero:
            return fnan, None, prec, None
        from .mul import Mul
        return evalf(Mul(*special), prec + 4, {})
    if has_zero:
        return None, None, None, None

    # With guard digits, multiplication in the real case does not destroy
    # accuracy. This is also true in the complex case when considering the
    # total accuracy; however accuracy for the real or imaginary parts
    # separately may be lower.
    acc = prec

    # XXX: big overestimate
    working_prec = prec + len(args) + 5

    # Empty product is 1
    start = man, exp, bc = MPZ(1), 0, 1

    # First, we multiply all pure real or pure imaginary numbers.
    # direction tells us that the result should be multiplied by
    # I**direction; all other numbers get put into complex_factors
    # to be multiplied out after the first phase.
    last = len(args)
    direction = 0
    args.append(S.One)
    complex_factors = []

    for i, arg in enumerate(args):
        if i != last and pure_complex(arg):
            args[-1] = (args[-1]*arg).expand()
            continue
        elif i == last and arg is S.One:
            continue
        re, im, re_acc, im_acc = evalf(arg, working_prec, options)
        if re and im:
            complex_factors.append((re, im, re_acc, im_acc))
            continue
        elif re:
            (s, m, e, b), w_acc = re, re_acc
        elif im:
            (s, m, e, b), w_acc = im, im_acc
            direction += 1
        else:
            return None, None, None, None
        direction += 2*s
        man *= m
        exp += e
        bc += b
        while bc > 3*working_prec:
            man >>= working_prec
            exp += working_prec
            bc -= working_prec
        acc = min(acc, w_acc)
    sign = (direction & 2) >> 1
    if not complex_factors:
        v = normalize(sign, man, exp, bitcount(man), prec, rnd)
        # multiply by i
        if direction & 1:
            return None, v, None, acc
        else:
            return v, None, acc, None
    else:
        # initialize with the first term
        if (man, exp, bc) != start:
            # there was a real part; give it an imaginary part
            re, im = (sign, man, exp, bitcount(man)), (0, MPZ(0), 0, 0)
            i0 = 0
        else:
            # there is no real part to start (other than the starting 1)
            wre, wim, wre_acc, wim_acc = complex_factors[0]
            acc = min(acc,
                      complex_accuracy((wre, wim, wre_acc, wim_acc)))
            re = wre
            im = wim
            i0 = 1

        for wre, wim, wre_acc, wim_acc in complex_factors[i0:]:
            # acc is the overall accuracy of the product; we aren't
            # computing exact accuracies of the product.
            acc = min(acc,
                      complex_accuracy((wre, wim, wre_acc, wim_acc)))

            use_prec = working_prec
            A = mpf_mul(re, wre, use_prec)
            B = mpf_mul(mpf_neg(im), wim, use_prec)
            C = mpf_mul(re, wim, use_prec)
            D = mpf_mul(im, wre, use_prec)
            re = mpf_add(A, B, use_prec)
            im = mpf_add(C, D, use_prec)
        if options.get('verbose'):
            print("MUL: wanted", prec, "accurate bits, got", acc)
        # multiply by I
        if direction & 1:
            re, im = mpf_neg(im), re
        return re, im, acc, acc


def evalf_pow(v: 'Pow', prec: int, options) -> TMP_RES:

    target_prec = prec
    base, exp = v.args

    # We handle x**n separately. This has two purposes: 1) it is much
    # faster, because we avoid calling evalf on the exponent, and 2) it
    # allows better handling of real/imaginary parts that are exactly zero
    if exp.is_Integer:
        p: int = exp.p  # type: ignore
        # Exact
        if not p:
            return fone, None, prec, None
        # Exponentiation by p magnifies relative error by |p|, so the
        # base must be evaluated with increased precision if p is large
        prec += int(math.log2(abs(p)))
        result = evalf(base, prec + 5, options)
        if result is S.ComplexInfinity:
            if p < 0:
                return None, None, None, None
            return result
        re, im, re_acc, im_acc = result
        # Real to integer power
        if re and not im:
            return mpf_pow_int(re, p, target_prec), None, target_prec, None
        # (x*I)**n = I**n * x**n
        if im and not re:
            z = mpf_pow_int(im, p, target_prec)
            case = p % 4
            if case == 0:
                return z, None, target_prec, None
            if case == 1:
                return None, z, None, target_prec
            if case == 2:
                return mpf_neg(z), None, target_prec, None
            if case == 3:
                return None, mpf_neg(z), None, target_prec
        # Zero raised to an integer power
        if not re:
            if p < 0:
                return S.ComplexInfinity
            return None, None, None, None
        # General complex number to arbitrary integer power
        re, im = libmp.mpc_pow_int((re, im), p, prec)
        # Assumes full accuracy in input
        return finalize_complex(re, im, target_prec)

    result = evalf(base, prec + 5, options)
    if result is S.ComplexInfinity:
        if exp.is_Rational:
            if exp < 0:
                return None, None, None, None
            return result
        raise NotImplementedError

    # Pure square root
    if exp is S.Half:
        xre, xim, _, _ = result
        # General complex square root
        if xim:
            re, im = libmp.mpc_sqrt((xre or fzero, xim), prec)
            return finalize_complex(re, im, prec)
        if not xre:
            return None, None, None, None
        # Square root of a negative real number
        if mpf_lt(xre, fzero):
            return None, mpf_sqrt(mpf_neg(xre), prec), None, prec
        # Positive square root
        return mpf_sqrt(xre, prec), None, prec, None

    # We first evaluate the exponent to find its magnitude
    # This determines the working precision that must be used
    prec += 10
    result = evalf(exp, prec, options)
    if result is S.ComplexInfinity:
        return fnan, None, prec, None
    yre, yim, _, _ = result
    # Special cases: x**0
    if not (yre or yim):
        return fone, None, prec, None

    ysize = fastlog(yre)
    # Restart if too big
    # XXX: prec + ysize might exceed maxprec
    if ysize > 5:
        prec += ysize
        yre, yim, _, _ = evalf(exp, prec, options)

    # Pure exponential function; no need to evalf the base
    if base is S.Exp1:
        if yim:
            re, im = libmp.mpc_exp((yre or fzero, yim), prec)
            return finalize_complex(re, im, target_prec)
        return mpf_exp(yre, target_prec), None, target_prec, None

    xre, xim, _, _ = evalf(base, prec + 5, options)
    # 0**y
    if not (xre or xim):
        if yim:
            return fnan, None, prec, None
        if yre[0] == 1:  # y < 0
            return S.ComplexInfinity
        return None, None, None, None

    # (real ** complex) or (complex ** complex)
    if yim:
        re, im = libmp.mpc_pow(
            (xre or fzero, xim or fzero), (yre or fzero, yim),
            target_prec)
        return finalize_complex(re, im, target_prec)
    # complex ** real
    if xim:
        re, im = libmp.mpc_pow_mpf((xre or fzero, xim), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    # negative ** real
    elif mpf_lt(xre, fzero):
        re, im = libmp.mpc_pow_mpf((xre, fzero), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    # positive ** real
    else:
        return mpf_pow(xre, yre, target_prec), None, target_prec, None


#----------------------------------------------------------------------------#
#                                                                            #
#                            Special functions                               #
#                                                                            #
#----------------------------------------------------------------------------#


def evalf_exp(expr: 'exp', prec: int, options: OPT_DICT) -> TMP_RES:
    from .power import Pow
    return evalf_pow(Pow(S.Exp1, expr.exp, evaluate=False), prec, options)


def evalf_trig(v: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    """
    This function handles sin , cos and tan of complex arguments.

    """
    from sympy.functions.elementary.trigonometric import cos, sin, tan
    if isinstance(v, cos):
        func = mpf_cos
    elif isinstance(v, sin):
        func = mpf_sin
    elif isinstance(v,tan):
        func = mpf_tan
    else:
        raise NotImplementedError
    arg = v.args[0]
    # 20 extra bits is possibly overkill. It does make the need
    # to restart very unlikely
    xprec = prec + 20
    re, im, re_acc, im_acc = evalf(arg, xprec, options)
    if im:
        if 'subs' in options:
            v = v.subs(options['subs'])
        return evalf(v._eval_evalf(prec), prec, options)
    if not re:
        if isinstance(v, cos):
            return fone, None, prec, None
        elif isinstance(v, sin):
            return None, None, None, None
        elif isinstance(v,tan):
            return None, None, None, None
        else:
            raise NotImplementedError
    # For trigonometric functions, we are interested in the
    # fixed-point (absolute) accuracy of the argument.
    xsize = fastlog(re)
    # Magnitude <= 1.0. OK to compute directly, because there is no
    # danger of hitting the first root of cos (with sin, magnitude
    # <= 2.0 would actually be ok)
    if xsize < 1:
        return func(re, prec, rnd), None, prec, None
    # Very large
    if xsize >= 10:
        xprec = prec + xsize
        re, im, re_acc, im_acc = evalf(arg, xprec, options)
    # Need to repeat in case the argument is very close to a
    # multiple of pi (or pi/2), hitting close to a root
    while 1:
        y = func(re, prec, rnd)
        ysize = fastlog(y)
        gap = -ysize
        accuracy = (xprec - xsize) - gap
        if accuracy < prec:
            if options.get('verbose'):
                print("SIN/COS/TAN", accuracy, "wanted", prec, "gap", gap)
                print(to_str(y, 10))
            if xprec > options.get('maxprec', DEFAULT_MAXPREC):
                return y, None, accuracy, None
            xprec += gap
            re, im, re_acc, im_acc = evalf(arg, xprec, options)
            continue
        else:
            return y, None, prec, None


def evalf_log(expr: 'log', prec: int, options: OPT_DICT) -> TMP_RES:
    if len(expr.args)>1:
        expr = expr.doit()
        return evalf(expr, prec, options)
    arg = expr.args[0]
    workprec = prec + 10
    result = evalf(arg, workprec, options)
    if result is S.ComplexInfinity:
        return result
    xre, xim, xacc, _ = result

    # evalf can return NoneTypes if chop=True
    # issue 18516, 19623
    if xre is xim is None:
        # Dear reviewer, I do not know what -inf is;
        # it looks to be (1, 0, -789, -3)
        # but I'm not sure in general,
        # so we just let mpmath figure
        # it out by taking log of 0 directly.
        # It would be better to return -inf instead.
        xre = fzero

    if xim:
        from sympy.functions.elementary.complexes import Abs
        from sympy.functions.elementary.exponential import log

        # XXX: use get_abs etc instead
        re = evalf_log(
            log(Abs(arg, evaluate=False), evaluate=False), prec, options)
        im = mpf_atan2(xim, xre or fzero, prec)
        return re[0], im, re[2], prec

    imaginary_term = (mpf_cmp(xre, fzero) < 0)

    re = mpf_log(mpf_abs(xre), prec, rnd)
    size = fastlog(re)
    if prec - size > workprec and re != fzero:
        from .add import Add
        # We actually need to compute 1+x accurately, not x
        add = Add(S.NegativeOne, arg, evaluate=False)
        xre, xim, _, _ = evalf_add(add, prec, options)
        prec2 = workprec - fastlog(xre)
        # xre is now x - 1 so we add 1 back here to calculate x
        re = mpf_log(mpf_abs(mpf_add(xre, fone, prec2)), prec, rnd)

    re_acc = prec

    if imaginary_term:
        return re, mpf_pi(prec), re_acc, prec
    else:
        return re, None, re_acc, None


def evalf_atan(v: 'atan', prec: int, options: OPT_DICT) -> TMP_RES:
    arg = v.args[0]
    xre, xim, reacc, imacc = evalf(arg, prec + 5, options)
    if xre is xim is None:
        return (None,)*4
    if xim:
        raise NotImplementedError
    return mpf_atan(xre, prec, rnd), None, prec, None


def evalf_subs(prec: int, subs: dict) -> dict:
    """ Change all Float entries in `subs` to have precision prec. """
    newsubs = {}
    for a, b in subs.items():
        b = S(b)
        if b.is_Float:
            b = b._eval_evalf(prec)
        newsubs[a] = b
    return newsubs


def evalf_piecewise(expr: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    from .numbers import Float, Integer
    if 'subs' in options:
        expr = expr.subs(evalf_subs(prec, options['subs']))
        newopts = options.copy()
        del newopts['subs']
        if hasattr(expr, 'func'):
            return evalf(expr, prec, newopts)
        if isinstance(expr, float):
            return evalf(Float(expr), prec, newopts)
        if isinstance(expr, int):
            return evalf(Integer(expr), prec, newopts)

    # We still have undefined symbols
    raise NotImplementedError


def evalf_alg_num(a: 'AlgebraicNumber', prec: int, options: OPT_DICT) -> TMP_RES:
    return evalf(a.to_root(), prec, options)

#----------------------------------------------------------------------------#
#                                                                            #
#                            High-level operations                           #
#                                                                            #
#----------------------------------------------------------------------------#


def as_mpmath(x: Any, prec: int, options: OPT_DICT) -> mpc | mpf:
    from .numbers import Infinity, NegativeInfinity, Zero
    x = sympify(x)
    if isinstance(x, Zero) or x == 0.0:
        return mpf(0)
    if isinstance(x, Infinity):
        return mpf('inf')
    if isinstance(x, NegativeInfinity):
        return mpf('-inf')
    # XXX
    result = evalf(x, prec, options)
    return quad_to_mpmath(result)


def do_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    func = expr.args[0]
    x, xlow, xhigh = expr.args[1]
    if xlow == xhigh:
        xlow = xhigh = 0
    elif x not in func.free_symbols:
        # only the difference in limits matters in this case
        # so if there is a symbol in common that will cancel
        # out when taking the difference, then use that
        # difference
        if xhigh.free_symbols & xlow.free_symbols:
            diff = xhigh - xlow
            if diff.is_number:
                xlow, xhigh = 0, diff

    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)
    options['maxprec'] = min(oldmaxprec, 2*prec)

    with workprec(prec + 5):
        xlow = as_mpmath(xlow, prec + 15, options)
        xhigh = as_mpmath(xhigh, prec + 15, options)

        # Integration is like summation, and we can phone home from
        # the integrand function to update accuracy summation style
        # Note that this accuracy is inaccurate, since it fails
        # to account for the variable quadrature weights,
        # but it is better than nothing

        from sympy.functions.elementary.trigonometric import cos, sin
        from .symbol import Wild

        have_part = [False, False]
        max_real_term: float | int = MINUS_INF
        max_imag_term: float | int = MINUS_INF

        def f(t: Expr) -> mpc | mpf:
            nonlocal max_real_term, max_imag_term
            re, im, re_acc, im_acc = evalf(func, mp.prec, {'subs': {x: t}})

            have_part[0] = re or have_part[0]
            have_part[1] = im or have_part[1]

            max_real_term = max(max_real_term, fastlog(re))
            max_imag_term = max(max_imag_term, fastlog(im))

            if im:
                return mpc(re or fzero, im)
            return mpf(re or fzero)

        if options.get('quad') == 'osc':
            A = Wild('A', exclude=[x])
            B = Wild('B', exclude=[x])
            D = Wild('D')
            m = func.match(cos(A*x + B)*D)
            if not m:
                m = func.match(sin(A*x + B)*D)
            if not m:
                raise ValueError("An integrand of the form sin(A*x+B)*f(x) "
                  "or cos(A*x+B)*f(x) is required for oscillatory quadrature")
            period = as_mpmath(2*S.Pi/m[A], prec + 15, options)
            result = quadosc(f, [xlow, xhigh], period=period)
            # XXX: quadosc does not do error detection yet
            quadrature_error = MINUS_INF
        else:
            result, quadrature_err = quadts(f, [xlow, xhigh], error=1)
            quadrature_error = fastlog(quadrature_err._mpf_)

    options['maxprec'] = oldmaxprec

    if have_part[0]:
        re: MPF_TUP | None = result.real._mpf_
        re_acc: int | None
        if re == fzero:
            re_s, re_acc = scaled_zero(int(-max(prec, max_real_term, quadrature_error)))
            re = scaled_zero(re_s)  # handled ok in evalf_integral
        else:
            re_acc = int(-max(max_real_term - fastlog(re) - prec, quadrature_error))
    else:
        re, re_acc = None, None

    if have_part[1]:
        im: MPF_TUP | None = result.imag._mpf_
        im_acc: int | None
        if im == fzero:
            im_s, im_acc = scaled_zero(int(-max(prec, max_imag_term, quadrature_error)))
            im = scaled_zero(im_s)  # handled ok in evalf_integral
        else:
            im_acc = int(-max(max_imag_term - fastlog(im) - prec, quadrature_error))
    else:
        im, im_acc = None, None

    result = re, im, re_acc, im_acc
    return result


def evalf_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    limits = expr.limits
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    workprec = prec
    i = 0
    maxprec = options.get('maxprec', INF)
    while 1:
        result = do_integral(expr, workprec, options)
        accuracy = complex_accuracy(result)
        if accuracy >= prec:  # achieved desired precision
            break
        if workprec >= maxprec:  # can't increase accuracy any more
            break
        if accuracy == -1:
            # maybe the answer really is zero and maybe we just haven't increased
            # the precision enough. So increase by doubling to not take too long
            # to get to maxprec.
            workprec *= 2
        else:
            workprec += max(prec, 2**i)
        workprec = min(workprec, maxprec)
        i += 1
    return result


def check_convergence(numer: Expr, denom: Expr, n: Symbol) -> tuple[int, Any, Any]:
    """
    Returns
    =======

    (h, g, p) where
    -- h is:
        > 0 for convergence of rate 1/factorial(n)**h
        < 0 for divergence of rate factorial(n)**(-h)
        = 0 for geometric or polynomial convergence or divergence

    -- abs(g) is:
        > 1 for geometric convergence of rate 1/h**n
        < 1 for geometric divergence of rate h**n
        = 1 for polynomial convergence or divergence

        (g < 0 indicates an alternating series)

    -- p is:
        > 1 for polynomial convergence of rate 1/n**h
        <= 1 for polynomial divergence of rate n**(-h)

    """
    from sympy.polys.polytools import Poly
    npol = Poly(numer, n)
    dpol = Poly(denom, n)
    p = npol.degree()
    q = dpol.degree()
    rate = q - p
    if rate:
        return rate, None, None
    constant = dpol.LC() / npol.LC()
    from .numbers import equal_valued
    if not equal_valued(abs(constant), 1):
        return rate, constant, None
    if npol.degree() == dpol.degree() == 0:
        return rate, constant, 0
    pc = npol.all_coeffs()[1]
    qc = dpol.all_coeffs()[1]
    return rate, constant, (qc - pc)/dpol.LC()


def hypsum(expr: Expr, n: Symbol, start: int, prec: int) -> mpf:
    """
    Sum a rapidly convergent infinite hypergeometric series with
    given general term, e.g. e = hypsum(1/factorial(n), n). The
    quotient between successive terms must be a quotient of integer
    polynomials.
    """
    from .numbers import Float, equal_valued
    from sympy.simplify.simplify import hypersimp

    if prec == float('inf'):
        raise NotImplementedError('does not support inf prec')

    if start:
        expr = expr.subs(n, n + start)
    hs = hypersimp(expr, n)
    if hs is None:
        raise NotImplementedError("a hypergeometric series is required")
    num, den = hs.as_numer_denom()

    func1 = lambdify(n, num)
    func2 = lambdify(n, den)

    h, g, p = check_convergence(num, den, n)

    if h < 0:
        raise ValueError("Sum diverges like (n!)^%i" % (-h))

    eterm = expr.subs(n, 0)
    if not eterm.is_Rational:
        raise NotImplementedError("Non rational term functionality is not implemented.")

    term: Rational = eterm # type: ignore

    # Direct summation if geometric or faster
    if h > 0 or (h == 0 and abs(g) > 1):
        term = (MPZ(term.p) << prec) // term.q
        s = term
        k = 1
        while abs(term) > 5:
            term *= MPZ(func1(k - 1))
            term //= MPZ(func2(k - 1))
            s += term
            k += 1
        return from_man_exp(s, -prec)
    else:
        alt = g < 0
        if abs(g) < 1:
            raise ValueError("Sum diverges like (%i)^n" % abs(1/g))
        if p < 1 or (equal_valued(p, 1) and not alt):
            raise ValueError("Sum diverges like n^%i" % (-p))
        # We have polynomial convergence: use Richardson extrapolation
        vold = None
        ndig = prec_to_dps(prec)
        while True:
            # Need to use at least quad precision because a lot of cancellation
            # might occur in the extrapolation process; we check the answer to
            # make sure that the desired precision has been reached, too.
            prec2 = 4*prec
            term0 = (MPZ(term.p) << prec2) // term.q

            def summand(k, _term=[term0]):
                if k:
                    k = int(k)
                    _term[0] *= MPZ(func1(k - 1))
                    _term[0] //= MPZ(func2(k - 1))
                return make_mpf(from_man_exp(_term[0], -prec2))

            with workprec(prec):
                v = nsum(summand, [0, mpmath_inf], method='richardson')
            vf = Float(v, ndig)
            if vold is not None and vold == vf:
                break
            prec += prec  # double precision each time
            vold = vf

        return v._mpf_


def evalf_prod(expr: 'Product', prec: int, options: OPT_DICT) -> TMP_RES:
    if all((l[1] - l[2]).is_Integer for l in expr.limits):
        result = evalf(expr.doit(), prec=prec, options=options)
    else:
        from sympy.concrete.summations import Sum
        result = evalf(expr.rewrite(Sum), prec=prec, options=options)
    return result


def evalf_sum(expr: 'Sum', prec: int, options: OPT_DICT) -> TMP_RES:
    from .numbers import Float
    if 'subs' in options:
        expr = expr.subs(options['subs']) # type: ignore
    func = expr.function
    limits = expr.limits
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    if func.is_zero:
        return None, None, prec, None
    prec2 = prec + 10
    try:
        n, a, b = limits[0]
        if b is not S.Infinity or a is S.NegativeInfinity or a != int(a):
            raise NotImplementedError
        # Use fast hypergeometric summation if possible
        v = hypsum(func, n, int(a), prec2)
        delta = prec - fastlog(v)
        if fastlog(v) < -10:
            v = hypsum(func, n, int(a), delta)
        return v, None, min(prec, delta), None
    except NotImplementedError:
        # Euler-Maclaurin summation for general series
        eps = Float(2.0)**(-prec)
        for i in range(1, 5):
            m = n = 2**i * prec
            s, err = expr.euler_maclaurin(m=m, n=n, eps=eps,
                eval_integral=False)
            err = err.evalf()
            if err is S.NaN:
                raise NotImplementedError
            if err <= eps:
                break
        err = fastlog(evalf(abs(err), 20, options)[0])
        re, im, re_acc, im_acc = evalf(s, prec2, options)
        if re_acc is None:
            re_acc = -err
        if im_acc is None:
            im_acc = -err
        return re, im, re_acc, im_acc


#----------------------------------------------------------------------------#
#                                                                            #
#                            Symbolic interface                              #
#                                                                            #
#----------------------------------------------------------------------------#

def evalf_symbol(x: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    val = options['subs'][x]
    if isinstance(val, mpf):
        if not val:
            return None, None, None, None
        return val._mpf_, None, prec, None
    else:
        if '_cache' not in options:
            options['_cache'] = {}
        cache = options['_cache']
        cached, cached_prec = cache.get(x, (None, MINUS_INF))
        if cached_prec >= prec:
            return cached
        v = evalf(sympify(val), prec, options)
        cache[x] = (v, prec)
        return v

evalf_table: dict[Type[Expr], Callable[[Expr, int, OPT_DICT], TMP_RES]] = {}


def _create_evalf_table():
    global evalf_table
    from sympy.concrete.products import Product
    from sympy.concrete.summations import Sum
    from .add import Add
    from .mul import Mul
    from .numbers import Exp1, Float, Half, ImaginaryUnit, Integer, NaN, NegativeOne, One, Pi, Rational, \
        Zero, ComplexInfinity, AlgebraicNumber
    from .power import Pow
    from .symbol import Dummy, Symbol
    from sympy.functions.elementary.complexes import Abs, im, re
    from sympy.functions.elementary.exponential import exp, log
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.functions.elementary.trigonometric import atan, cos, sin, tan
    from sympy.integrals.integrals import Integral
    evalf_table = {
        Symbol: evalf_symbol,
        Dummy: evalf_symbol,
        Float: evalf_float,
        Rational: evalf_rational,
        Integer: evalf_integer,
        Zero: lambda x, prec, options: (None, None, prec, None),
        One: lambda x, prec, options: (fone, None, prec, None),
        Half: lambda x, prec, options: (fhalf, None, prec, None),
        Pi: lambda x, prec, options: (mpf_pi(prec), None, prec, None),
        Exp1: lambda x, prec, options: (mpf_e(prec), None, prec, None),
        ImaginaryUnit: lambda x, prec, options: (None, fone, None, prec),
        NegativeOne: lambda x, prec, options: (fnone, None, prec, None),
        ComplexInfinity: lambda x, prec, options: S.ComplexInfinity,
        NaN: lambda x, prec, options: (fnan, None, prec, None),

        exp: evalf_exp,

        cos: evalf_trig,
        sin: evalf_trig,
        tan: evalf_trig,

        Add: evalf_add,
        Mul: evalf_mul,
        Pow: evalf_pow,

        log: evalf_log,
        atan: evalf_atan,
        Abs: evalf_abs,

        re: evalf_re,
        im: evalf_im,
        floor: evalf_floor,
        ceiling: evalf_ceiling,

        Integral: evalf_integral,
        Sum: evalf_sum,
        Product: evalf_prod,
        Piecewise: evalf_piecewise,

        AlgebraicNumber: evalf_alg_num,
    }


def evalf(x: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    """
    Evaluate the ``Expr`` instance, ``x``
    to a binary precision of ``prec``. This
    function is supposed to be used internally.

    Parameters
    ==========

    x : Expr
        The formula to evaluate to a float.
    prec : int
        The binary precision that the output should have.
    options : dict
        A dictionary with the same entries as
        ``EvalfMixin.evalf`` and in addition,
        ``maxprec`` which is the maximum working precision.

    Returns
    =======

    An optional tuple, ``(re, im, re_acc, im_acc)``
    which are the real, imaginary, real accuracy
    and imaginary accuracy respectively. ``re`` is
    an mpf value tuple and so is ``im``. ``re_acc``
    and ``im_acc`` are ints.

    NB: all these return values can be ``None``.
    If all values are ``None``, then that represents 0.
    Note that 0 is also represented as ``fzero = (0, 0, 0, 0)``.
    """
    from sympy.functions.elementary.complexes import re as re_, im as im_
    try:
        rf = evalf_table[type(x)]
        r = rf(x, prec, options)
    except KeyError:
        # Fall back to ordinary evalf if possible
        if 'subs' in options:
            x = x.subs(evalf_subs(prec, options['subs']))
        xe = x._eval_evalf(prec)
        if xe is None:
            raise NotImplementedError
        as_real_imag = getattr(xe, "as_real_imag", None)
        if as_real_imag is None:
            raise NotImplementedError # e.g. FiniteSet(-1.0, 1.0).evalf()
        re, im = as_real_imag()
        if re.has(re_) or im.has(im_):
            raise NotImplementedError
        if not re:
            re = None
            reprec = None
        elif re.is_number:
            re = re._to_mpmath(prec, allow_ints=False)._mpf_
            reprec = prec
        else:
            raise NotImplementedError
        if not im:
            im = None
            imprec = None
        elif im.is_number:
            im = im._to_mpmath(prec, allow_ints=False)._mpf_
            imprec = prec
        else:
            raise NotImplementedError
        r = re, im, reprec, imprec

    if options.get("verbose"):
        print("### input", x)
        print("### output", to_str(r[0] or fzero, 50) if isinstance(r, tuple) else r)
        print("### raw", r) # r[0], r[2]
        print()
    chop = options.get('chop', False)
    if chop:
        if chop is True:
            chop_prec = prec
        else:
            # convert (approximately) from given tolerance;
            # the formula here will will make 1e-i rounds to 0 for
            # i in the range +/-27 while 2e-i will not be chopped
            chop_prec = int(round(-3.321*math.log10(chop) + 2.5))
            if chop_prec == 3:
                chop_prec -= 1
        r = chop_parts(r, chop_prec)
    if options.get("strict"):
        check_target(x, r, prec)
    return r


def quad_to_mpmath(q, ctx=None):
    """Turn the quad returned by ``evalf`` into an ``mpf`` or ``mpc``. """
    mpc = make_mpc if ctx is None else ctx.make_mpc
    mpf = make_mpf if ctx is None else ctx.make_mpf
    if q is S.ComplexInfinity:
        raise NotImplementedError
    re, im, _, _ = q
    if im:
        if not re:
            re = fzero
        return mpc((re, im))
    elif re:
        return mpf(re)
    else:
        return mpf(fzero)


class EvalfMixin:
    """Mixin class adding evalf capability."""

    __slots__: tuple[str, ...] = ()

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """
        Evaluate the given formula to an accuracy of *n* digits.

        Parameters
        ==========

        subs : dict, optional
            Substitute numerical values for symbols, e.g.
            ``subs={x:3, y:1+pi}``. The substitutions must be given as a
            dictionary.

        maxn : int, optional
            Allow a maximum temporary working precision of maxn digits.

        chop : bool or number, optional
            Specifies how to replace tiny real or imaginary parts in
            subresults by exact zeros.

            When ``True`` the chop value defaults to standard precision.

            Otherwise the chop value is used to determine the
            magnitude of "small" for purposes of chopping.

            >>> from sympy import N
            >>> x = 1e-4
            >>> N(x, chop=True)
            0.000100000000000000
            >>> N(x, chop=1e-5)
            0.000100000000000000
            >>> N(x, chop=1e-4)
            0

        strict : bool, optional
            Raise ``PrecisionExhausted`` if any subresult fails to
            evaluate to full accuracy, given the available maxprec.

        quad : str, optional
            Choose algorithm for numerical quadrature. By default,
            tanh-sinh quadrature is used. For oscillatory
            integrals on an infinite interval, try ``quad='osc'``.

        verbose : bool, optional
            Print debug information.

        Notes
        =====

        When Floats are naively substituted into an expression,
        precision errors may adversely affect the result. For example,
        adding 1e16 (a Float) to 1 will truncate to 1e16; if 1e16 is
        then subtracted, the result will be 0.
        That is exactly what happens in the following:

        >>> from sympy.abc import x, y, z
        >>> values = {x: 1e16, y: 1, z: 1e16}
        >>> (x + y - z).subs(values)
        0

        Using the subs argument for evalf is the accurate way to
        evaluate such an expression:

        >>> (x + y - z).evalf(subs=values)
        1.00000000000000
        """
        from .numbers import Float, Number
        n = n if n is not None else 15

        if subs and is_sequence(subs):
            raise TypeError('subs must be given as a dictionary')

        # for sake of sage that doesn't like evalf(1)
        if n == 1 and isinstance(self, Number):
            from .expr import _mag
            rv = self.evalf(2, subs, maxn, chop, strict, quad, verbose)
            m = _mag(rv)
            rv = rv.round(1 - m)
            return rv

        if not evalf_table:
            _create_evalf_table()
        prec = dps_to_prec(n)
        options = {'maxprec': max(prec, int(maxn*LG10)), 'chop': chop,
               'strict': strict, 'verbose': verbose}
        if subs is not None:
            options['subs'] = subs
        if quad is not None:
            options['quad'] = quad
        try:
            result = evalf(self, prec + 4, options)
        except NotImplementedError:
            # Fall back to the ordinary evalf
            if hasattr(self, 'subs') and subs is not None:  # issue 20291
                v = self.subs(subs)._eval_evalf(prec)
            else:
                v = self._eval_evalf(prec)
            if v is None:
                return self
            elif not v.is_number:
                return v
            try:
                # If the result is numerical, normalize it
                result = evalf(v, prec, options)
            except NotImplementedError:
                # Probably contains symbols or unknown functions
                return v
        if result is S.ComplexInfinity:
            return result
        re, im, re_acc, im_acc = result
        if re is S.NaN or im is S.NaN:
            return S.NaN
        if re:
            p = max(min(prec, re_acc), 1)
            re = Float._new(re, p)
        else:
            re = S.Zero
        if im:
            p = max(min(prec, im_acc), 1)
            im = Float._new(im, p)
            return re + im*S.ImaginaryUnit
        else:
            return re

    n = evalf

    def _evalf(self, prec: int) -> Expr:
        """Helper for evalf. Does the same thing but takes binary precision"""
        r = self._eval_evalf(prec)
        if r is None:
            r = self # type: ignore
        return r # type: ignore

    def _eval_evalf(self, prec: int) -> Expr | None:
        return None

    def _to_mpmath(self, prec, allow_ints=True):
        # mpmath functions accept ints as input
        errmsg = "cannot convert to mpmath number"
        if allow_ints and self.is_Integer:
            return self.p
        if hasattr(self, '_as_mpf_val'):
            return make_mpf(self._as_mpf_val(prec))
        try:
            result = evalf(self, prec, {})
            return quad_to_mpmath(result)
        except NotImplementedError:
            v = self._eval_evalf(prec)
            if v is None:
                raise ValueError(errmsg)
            if v.is_Float:
                return make_mpf(v._mpf_)
            # Number + Number*I is also fine
            re, im = v.as_real_imag()
            if allow_ints and re.is_Integer:
                re = from_int(re.p)
            elif re.is_Float:
                re = re._mpf_
            else:
                raise ValueError(errmsg)
            if allow_ints and im.is_Integer:
                im = from_int(im.p)
            elif im.is_Float:
                im = im._mpf_
            else:
                raise ValueError(errmsg)
            return make_mpc((re, im))


def N(x, n=15, **options):
    r"""
    Calls x.evalf(n, \*\*options).

    Explanations
    ============

    Both .n() and N() are equivalent to .evalf(); use the one that you like better.
    See also the docstring of .evalf() for information on the options.

    Examples
    ========

    >>> from sympy import Sum, oo, N
    >>> from sympy.abc import k
    >>> Sum(1/k**k, (k, 1, oo))
    Sum(k**(-k), (k, 1, oo))
    >>> N(_, 4)
    1.291

    """
    # by using rational=True, any evaluation of a string
    # will be done using exact values for the Floats
    return sympify(x, rational=True).evalf(n, **options)


def _evalf_with_bounded_error(x: Expr, eps: Expr | None = None,
                              m: int = 0,
                              options: OPT_DICT | None = None) -> TMP_RES:
    """
    Evaluate *x* to within a bounded absolute error.

    Parameters
    ==========

    x : Expr
        The quantity to be evaluated.
    eps : Expr, None, optional (default=None)
        Positive real upper bound on the acceptable error.
    m : int, optional (default=0)
        If *eps* is None, then use 2**(-m) as the upper bound on the error.
    options: OPT_DICT
        As in the ``evalf`` function.

    Returns
    =======

    A tuple ``(re, im, re_acc, im_acc)``, as returned by ``evalf``.

    See Also
    ========

    evalf

    """
    if eps is not None:
        if not (eps.is_Rational or eps.is_Float) or not eps > 0:
            raise ValueError("eps must be positive")
        r, _, _, _ = evalf(1/eps, 1, {})
        m = fastlog(r)

    c, d, _, _ = evalf(x, 1, {})
    # Note: If x = a + b*I, then |a| <= 2|c| and |b| <= 2|d|, with equality
    # only in the zero case.
    # If a is non-zero, then |c| = 2**nc for some integer nc, and c has
    # bitcount 1. Therefore 2**fastlog(c) = 2**(nc+1) = 2|c| is an upper bound
    # on |a|. Likewise for b and d.
    nr, ni = fastlog(c), fastlog(d)
    n = max(nr, ni) + 1
    # If x is 0, then n is MINUS_INF, and p will be 1. Otherwise,
    # n - 1 bits get us past the integer parts of a and b, and +1 accounts for
    # the factor of <= sqrt(2) that is |x|/max(|a|, |b|).
    p = max(1, m + n + 1)

    options = options or {}
    return evalf(x, p, options)
<|endoftext|>"""
Module to efficiently partition SymPy objects.

This system is introduced because class of SymPy object does not always
represent the mathematical classification of the entity. For example,
``Integral(1, x)`` and ``Integral(Matrix([1,2]), x)`` are both instance
of ``Integral`` class. However the former is number and the latter is
matrix.

One way to resolve this is defining subclass for each mathematical type,
such as ``MatAdd`` for the addition between matrices. Basic algebraic
operation such as addition or multiplication take this approach, but
defining every class for every mathematical object is not scalable.

Therefore, we define the "kind" of the object and let the expression
infer the kind of itself from its arguments. Function and class can
filter the arguments by their kind, and behave differently according to
the type of itself.

This module defines basic kinds for core objects. Other kinds such as
``ArrayKind`` or ``MatrixKind`` can be found in corresponding modules.

.. notes::
       This approach is experimental, and can be replaced or deleted in the future.
       See https://github.com/sympy/sympy/pull/20549.
"""

from collections import defaultdict

from .cache import cacheit
from sympy.multipledispatch.dispatcher import (Dispatcher,
    ambiguity_warn, ambiguity_register_error_ignore_dup,
    str_signature, RaiseNotImplementedError)


class KindMeta(type):
    """
    Metaclass for ``Kind``.

    Assigns empty ``dict`` as class attribute ``_inst`` for every class,
    in order to endow singleton-like behavior.
    """
    def __new__(cls, clsname, bases, dct):
        dct['_inst'] = {}
        return super().__new__(cls, clsname, bases, dct)


class Kind(object, metaclass=KindMeta):
    """
    Base class for kinds.

    Kind of the object represents the mathematical classification that
    the entity falls into. It is expected that functions and classes
    recognize and filter the argument by its kind.

    Kind of every object must be carefully selected so that it shows the
    intention of design. Expressions may have different kind according
    to the kind of its arguments. For example, arguments of ``Add``
    must have common kind since addition is group operator, and the
    resulting ``Add()`` has the same kind.

    For the performance, each kind is as broad as possible and is not
    based on set theory. For example, ``NumberKind`` includes not only
    complex number but expression containing ``S.Infinity`` or ``S.NaN``
    which are not strictly number.

    Kind may have arguments as parameter. For example, ``MatrixKind()``
    may be constructed with one element which represents the kind of its
    elements.

    ``Kind`` behaves in singleton-like fashion. Same signature will
    return the same object.

    """
    def __new__(cls, *args):
        if args in cls._inst:
            inst = cls._inst[args]
        else:
            inst = super().__new__(cls)
            cls._inst[args] = inst
        return inst


class _UndefinedKind(Kind):
    """
    Default kind for all SymPy object. If the kind is not defined for
    the object, or if the object cannot infer the kind from its
    arguments, this will be returned.

    Examples
    ========

    >>> from sympy import Expr
    >>> Expr().kind
    UndefinedKind
    """
    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return "UndefinedKind"

UndefinedKind = _UndefinedKind()


class _NumberKind(Kind):
    """
    Kind for all numeric object.

    This kind represents every number, including complex numbers,
    infinity and ``S.NaN``. Other objects such as quaternions do not
    have this kind.

    Most ``Expr`` are initially designed to represent the number, so
    this will be the most common kind in SymPy core. For example
    ``Symbol()``, which represents a scalar, has this kind as long as it
    is commutative.

    Numbers form a field. Any operation between number-kind objects will
    result this kind as well.

    Examples
    ========

    >>> from sympy import S, oo, Symbol
    >>> S.One.kind
    NumberKind
    >>> (-oo).kind
    NumberKind
    >>> S.NaN.kind
    NumberKind

    Commutative symbol are treated as number.

    >>> x = Symbol('x')
    >>> x.kind
    NumberKind
    >>> Symbol('y', commutative=False).kind
    UndefinedKind

    Operation between numbers results number.

    >>> (x+1).kind
    NumberKind

    See Also
    ========

    sympy.core.expr.Expr.is_Number : check if the object is strictly
    subclass of ``Number`` class.

    sympy.core.expr.Expr.is_number : check if the object is number
    without any free symbol.

    """
    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return "NumberKind"

NumberKind = _NumberKind()


class _BooleanKind(Kind):
    """
    Kind for boolean objects.

    SymPy's ``S.true``, ``S.false``, and built-in ``True`` and ``False``
    have this kind. Boolean number ``1`` and ``0`` are not relevant.

    Examples
    ========

    >>> from sympy import S, Q
    >>> S.true.kind
    BooleanKind
    >>> Q.even(3).kind
    BooleanKind
    """
    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return "BooleanKind"

BooleanKind = _BooleanKind()


class KindDispatcher:
    """
    Dispatcher to select a kind from multiple kinds by binary dispatching.

    .. notes::
       This approach is experimental, and can be replaced or deleted in
       the future.

    Explanation
    ===========

    SymPy object's :obj:`sympy.core.kind.Kind()` vaguely represents the
    algebraic structure where the object belongs to. Therefore, with
    given operation, we can always find a dominating kind among the
    different kinds. This class selects the kind by recursive binary
    dispatching. If the result cannot be determined, ``UndefinedKind``
    is returned.

    Examples
    ========

    Multiplication between numbers return number.

    >>> from sympy import NumberKind, Mul
    >>> Mul._kind_dispatcher(NumberKind, NumberKind)
    NumberKind

    Multiplication between number and unknown-kind object returns unknown kind.

    >>> from sympy import UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind)
    UndefinedKind

    Any number and order of kinds is allowed.

    >>> Mul._kind_dispatcher(UndefinedKind, NumberKind)
    UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind, NumberKind)
    UndefinedKind

    Since matrix forms a vector space over scalar field, multiplication
    between matrix with numeric element and number returns matrix with
    numeric element.

    >>> from sympy.matrices import MatrixKind
    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), NumberKind)
    MatrixKind(NumberKind)

    If a matrix with number element and another matrix with unknown-kind
    element are multiplied, we know that the result is matrix but the
    kind of its elements is unknown.

    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), MatrixKind(UndefinedKind))
    MatrixKind(UndefinedKind)

    Parameters
    ==========

    name : str

    commutative : bool, optional
        If True, binary dispatch will be automatically registered in
        reversed order as well.

    doc : str, optional

    """
    def __init__(self, name, commutative=False, doc=None):
        self.name = name
        self.doc = doc
        self.commutative = commutative
        self._dispatcher = Dispatcher(name)

    def __repr__(self):
        return "<dispatched %s>" % self.name

    def register(self, *types, **kwargs):
        """
        Register the binary dispatcher for two kind classes.

        If *self.commutative* is ``True``, signature in reversed order is
        automatically registered as well.
        """
        on_ambiguity = kwargs.pop("on_ambiguity", None)
        if not on_ambiguity:
            if self.commutative:
                on_ambiguity = ambiguity_register_error_ignore_dup
            else:
                on_ambiguity = ambiguity_warn
        kwargs.update(on_ambiguity=on_ambiguity)

        if not len(types) == 2:
            raise RuntimeError(
                "Only binary dispatch is supported, but got %s types: <%s>." % (
                len(types), str_signature(types)
            ))

        def _(func):
            self._dispatcher.add(types, func, **kwargs)
            if self.commutative:
                self._dispatcher.add(tuple(reversed(types)), func, **kwargs)
        return _

    def __call__(self, *args, **kwargs):
        if self.commutative:
            kinds = frozenset(args)
        else:
            kinds = []
            prev = None
            for a in args:
                if prev is not a:
                    kinds.append(a)
                    prev = a
        return self.dispatch_kinds(kinds, **kwargs)

    @cacheit
    def dispatch_kinds(self, kinds, **kwargs):
        # Quick exit for the case where all kinds are same
        if len(kinds) == 1:
            result, = kinds
            if not isinstance(result, Kind):
                raise RuntimeError("%s is not a kind." % result)
            return result

        for i,kind in enumerate(kinds):
            if not isinstance(kind, Kind):
                raise RuntimeError("%s is not a kind." % kind)

            if i == 0:
                result = kind
            else:
                prev_kind = result

                t1, t2 = type(prev_kind), type(kind)
                k1, k2 = prev_kind, kind
                func = self._dispatcher.dispatch(t1, t2)
                if func is None and self.commutative:
                    # try reversed order
                    func = self._dispatcher.dispatch(t2, t1)
                    k1, k2 = k2, k1
                if func is None:
                    # unregistered kind relation
                    result = UndefinedKind
                else:
                    result = func(k1, k2)
                if not isinstance(result, Kind):
                    raise RuntimeError(
                        "Dispatcher for {!r} and {!r} must return a Kind, but got {!r}".format(
                        prev_kind, kind, result
                    ))

        return result

    @property
    def __doc__(self):
        docs = [
            "Kind dispatcher : %s" % self.name,
            "Note that support for this is experimental. See the docs for :class:`KindDispatcher` for details"
        ]

        if self.doc:
            docs.append(self.doc)

        s = "Registered kind classes\n"
        s += '=' * len(s)
        docs.append(s)

        amb_sigs = []

        typ_sigs = defaultdict(list)
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)

        for func, sigs in typ_sigs.items():

            sigs_str = ', '.join('<%s>' % str_signature(sig) for sig in sigs)

            if isinstance(func, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue

            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            if func.__doc__:
                s += func.__doc__.strip()
            else:
                s += func.__name__
            docs.append(s)

        if amb_sigs:
            s = "Ambiguous kind classes\n"
            s += '=' * len(s)
            docs.append(s)

            s = '\n'.join(amb_sigs)
            docs.append(s)

        return '\n\n'.join(docs)
<|endoftext|>"""
Replacement rules.
"""

class Transform:
    """
    Immutable mapping that can be used as a generic transformation rule.

    Parameters
    ==========

    transform : callable
        Computes the value corresponding to any key.

    filter : callable, optional
        If supplied, specifies which objects are in the mapping.

    Examples
    ========

    >>> from sympy.core.rules import Transform
    >>> from sympy.abc import x

    This Transform will return, as a value, one more than the key:

    >>> add1 = Transform(lambda x: x + 1)
    >>> add1[1]
    2
    >>> add1[x]
    x + 1

    By default, all values are considered to be in the dictionary. If a filter
    is supplied, only the objects for which it returns True are considered as
    being in the dictionary:

    >>> add1_odd = Transform(lambda x: x + 1, lambda x: x%2 == 1)
    >>> 2 in add1_odd
    False
    >>> add1_odd.get(2, 0)
    0
    >>> 3 in add1_odd
    True
    >>> add1_odd[3]
    4
    >>> add1_odd.get(3, 0)
    4
    """

    def __init__(self, transform, filter=lambda x: True):
        self._transform = transform
        self._filter = filter

    def __contains__(self, item):
        return self._filter(item)

    def __getitem__(self, key):
        if self._filter(key):
            return self._transform(key)
        else:
            raise KeyError(key)

    def get(self, item, default=None):
        if item in self:
            return self[item]
        else:
            return default
<|endoftext|>"""
This module contains the machinery handling assumptions.
Do also consider the guide :ref:`assumptions-guide`.

All symbolic objects have assumption attributes that can be accessed via
``.is_<assumption name>`` attribute.

Assumptions determine certain properties of symbolic objects and can
have 3 possible values: ``True``, ``False``, ``None``.  ``True`` is returned if the
object has the property and ``False`` is returned if it does not or cannot
(i.e. does not make sense):

    >>> from sympy import I
    >>> I.is_algebraic
    True
    >>> I.is_real
    False
    >>> I.is_prime
    False

When the property cannot be determined (or when a method is not
implemented) ``None`` will be returned. For example,  a generic symbol, ``x``,
may or may not be positive so a value of ``None`` is returned for ``x.is_positive``.

By default, all symbolic values are in the largest set in the given context
without specifying the property. For example, a symbol that has a property
being integer, is also real, complex, etc.

Here follows a list of possible assumption names:

.. glossary::

    commutative
        object commutes with any other object with
        respect to multiplication operation. See [12]_.

    complex
        object can have only values from the set
        of complex numbers. See [13]_.

    imaginary
        object value is a number that can be written as a real
        number multiplied by the imaginary unit ``I``.  See
        [3]_.  Please note that ``0`` is not considered to be an
        imaginary number, see
        `issue #7649 <https://github.com/sympy/sympy/issues/7649>`_.

    real
        object can have only values from the set
        of real numbers.

    extended_real
        object can have only values from the set
        of real numbers, ``oo`` and ``-oo``.

    integer
        object can have only values from the set
        of integers.

    odd
    even
        object can have only values from the set of
        odd (even) integers [2]_.

    prime
        object is a natural number greater than 1 that has
        no positive divisors other than 1 and itself.  See [6]_.

    composite
        object is a positive integer that has at least one positive
        divisor other than 1 or the number itself.  See [4]_.

    zero
        object has the value of 0.

    nonzero
        object is a real number that is not zero.

    rational
        object can have only values from the set
        of rationals.

    algebraic
        object can have only values from the set
        of algebraic numbers [11]_.

    transcendental
        object can have only values from the set
        of transcendental numbers [10]_.

    irrational
        object value cannot be represented exactly by :class:`~.Rational`, see [5]_.

    finite
    infinite
        object absolute value is bounded (arbitrarily large).
        See [7]_, [8]_, [9]_.

    negative
    nonnegative
        object can have only negative (nonnegative)
        values [1]_.

    positive
    nonpositive
        object can have only positive (nonpositive) values.

    extended_negative
    extended_nonnegative
    extended_positive
    extended_nonpositive
    extended_nonzero
        as without the extended part, but also including infinity with
        corresponding sign, e.g., extended_positive includes ``oo``

    hermitian
    antihermitian
        object belongs to the field of Hermitian
        (antihermitian) operators.

Examples
========

    >>> from sympy import Symbol
    >>> x = Symbol('x', real=True); x
    x
    >>> x.is_real
    True
    >>> x.is_complex
    True

See Also
========

.. seealso::

    :py:class:`sympy.core.numbers.ImaginaryUnit`
    :py:class:`sympy.core.numbers.Zero`
    :py:class:`sympy.core.numbers.One`
    :py:class:`sympy.core.numbers.Infinity`
    :py:class:`sympy.core.numbers.NegativeInfinity`
    :py:class:`sympy.core.numbers.ComplexInfinity`

Notes
=====

The fully-resolved assumptions for any SymPy expression
can be obtained as follows:

    >>> from sympy.core.assumptions import assumptions
    >>> x = Symbol('x',positive=True)
    >>> assumptions(x + I)
    {'commutative': True, 'complex': True, 'composite': False, 'even':
    False, 'extended_negative': False, 'extended_nonnegative': False,
    'extended_nonpositive': False, 'extended_nonzero': False,
    'extended_positive': False, 'extended_real': False, 'finite': True,
    'imaginary': False, 'infinite': False, 'integer': False, 'irrational':
    False, 'negative': False, 'noninteger': False, 'nonnegative': False,
    'nonpositive': False, 'nonzero': False, 'odd': False, 'positive':
    False, 'prime': False, 'rational': False, 'real': False, 'zero':
    False}

Developers Notes
================

The current (and possibly incomplete) values are stored
in the ``obj._assumptions dictionary``; queries to getter methods
(with property decorators) or attributes of objects/classes
will return values and update the dictionary.

    >>> eq = x**2 + I
    >>> eq._assumptions
    {}
    >>> eq.is_finite
    True
    >>> eq._assumptions
    {'finite': True, 'infinite': False}

For a :class:`~.Symbol`, there are two locations for assumptions that may
be of interest. The ``assumptions0`` attribute gives the full set of
assumptions derived from a given set of initial assumptions. The
latter assumptions are stored as ``Symbol._assumptions_orig``

    >>> Symbol('x', prime=True, even=True)._assumptions_orig
    {'even': True, 'prime': True}

The ``_assumptions_orig`` are not necessarily canonical nor are they filtered
in any way: they records the assumptions used to instantiate a Symbol and (for
storage purposes) represent a more compact representation of the assumptions
needed to recreate the full set in ``Symbol.assumptions0``.


References
==========

.. [1] https://en.wikipedia.org/wiki/Negative_number
.. [2] https://en.wikipedia.org/wiki/Parity_%28mathematics%29
.. [3] https://en.wikipedia.org/wiki/Imaginary_number
.. [4] https://en.wikipedia.org/wiki/Composite_number
.. [5] https://en.wikipedia.org/wiki/Irrational_number
.. [6] https://en.wikipedia.org/wiki/Prime_number
.. [7] https://en.wikipedia.org/wiki/Finite
.. [8] https://docs.python.org/3/library/math.html#math.isfinite
.. [9] https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
.. [10] https://en.wikipedia.org/wiki/Transcendental_number
.. [11] https://en.wikipedia.org/wiki/Algebraic_number
.. [12] https://en.wikipedia.org/wiki/Commutative_property
.. [13] https://en.wikipedia.org/wiki/Complex_number

"""

from sympy.utilities.exceptions import sympy_deprecation_warning

from .facts import FactRules, FactKB
from .sympify import sympify

from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions

def _load_pre_generated_assumption_rules() -> FactRules:
    """ Load the assumption rules from pre-generated data

    To update the pre-generated data, see :method::`_generate_assumption_rules`
    """
    _assume_rules=FactRules._from_python(_assumptions)
    return _assume_rules

def _generate_assumption_rules():
    """ Generate the default assumption rules

    This method should only be called to update the pre-generated
    assumption rules.

    To update the pre-generated assumptions run: bin/ask_update.py

    """
    _assume_rules = FactRules([

    'integer        ->  rational',
    'rational       ->  real',
    'rational       ->  algebraic',
    'algebraic      ->  complex',
    'transcendental ==  complex & !algebraic',
    'real           ->  hermitian',
    'imaginary      ->  complex',
    'imaginary      ->  antihermitian',
    'extended_real  ->  commutative',
    'complex        ->  commutative',
    'complex        ->  finite',

    'odd            ==  integer & !even',
    'even           ==  integer & !odd',

    'real           ->  complex',
    'extended_real  ->  real | infinite',
    'real           ==  extended_real & finite',

    'extended_real        ==  extended_negative | zero | extended_positive',
    'extended_negative    ==  extended_nonpositive & extended_nonzero',
    'extended_positive    ==  extended_nonnegative & extended_nonzero',

    'extended_nonpositive ==  extended_real & !extended_positive',
    'extended_nonnegative ==  extended_real & !extended_negative',

    'real           ==  negative | zero | positive',
    'negative       ==  nonpositive & nonzero',
    'positive       ==  nonnegative & nonzero',

    'nonpositive    ==  real & !positive',
    'nonnegative    ==  real & !negative',

    'positive       ==  extended_positive & finite',
    'negative       ==  extended_negative & finite',
    'nonpositive    ==  extended_nonpositive & finite',
    'nonnegative    ==  extended_nonnegative & finite',
    'nonzero        ==  extended_nonzero & finite',

    'zero           ->  even & finite',
    'zero           ==  extended_nonnegative & extended_nonpositive',
    'zero           ==  nonnegative & nonpositive',
    'nonzero        ->  real',

    'prime          ->  integer & positive',
    'composite      ->  integer & positive & !prime',
    '!composite     ->  !positive | !even | prime',

    'irrational     ==  real & !rational',

    'imaginary      ->  !extended_real',

    'infinite       ==  !finite',
    'noninteger     ==  extended_real & !integer',
    'extended_nonzero == extended_real & !zero',
    ])
    return _assume_rules


_assume_rules = _load_pre_generated_assumption_rules()
_assume_defined = _assume_rules.defined_facts.copy()
_assume_defined.add('polar')
_assume_defined = frozenset(_assume_defined)


def assumptions(expr, _check=None):
    """return the T/F assumptions of ``expr``"""
    n = sympify(expr)
    if n.is_Symbol:
        rv = n.assumptions0  # are any important ones missing?
        if _check is not None:
            rv = {k: rv[k] for k in set(rv) & set(_check)}
        return rv
    rv = {}
    for k in _assume_defined if _check is None else _check:
        v = getattr(n, 'is_{}'.format(k))
        if v is not None:
            rv[k] = v
    return rv


def common_assumptions(exprs, check=None):
    """return those assumptions which have the same True or False
    value for all the given expressions.

    Examples
    ========

    >>> from sympy.core import common_assumptions
    >>> from sympy import oo, pi, sqrt
    >>> common_assumptions([-4, 0, sqrt(2), 2, pi, oo])
    {'commutative': True, 'composite': False,
    'extended_real': True, 'imaginary': False, 'odd': False}

    By default, all assumptions are tested; pass an iterable of the
    assumptions to limit those that are reported:

    >>> common_assumptions([0, 1, 2], ['positive', 'integer'])
    {'integer': True}
    """
    check = _assume_defined if check is None else set(check)
    if not check or not exprs:
        return {}

    # get all assumptions for each
    assume = [assumptions(i, _check=check) for i in sympify(exprs)]
    # focus on those of interest that are True
    for i, e in enumerate(assume):
        assume[i] = {k: e[k] for k in set(e) & check}
    # what assumptions are in common?
    common = set.intersection(*[set(i) for i in assume])
    # which ones hold the same value
    a = assume[0]
    return {k: a[k] for k in common if all(a[k] == b[k]
        for b in assume)}


def failing_assumptions(expr, **assumptions):
    """
    Return a dictionary containing assumptions with values not
    matching those of the passed assumptions.

    Examples
    ========

    >>> from sympy import failing_assumptions, Symbol

    >>> x = Symbol('x', positive=True)
    >>> y = Symbol('y')
    >>> failing_assumptions(6*x + y, positive=True)
    {'positive': None}

    >>> failing_assumptions(x**2 - 1, positive=True)
    {'positive': None}

    If *expr* satisfies all of the assumptions, an empty dictionary is returned.

    >>> failing_assumptions(x**2, positive=True)
    {}

    """
    expr = sympify(expr)
    failed = {}
    for k in assumptions:
        test = getattr(expr, 'is_%s' % k, None)
        if test is not assumptions[k]:
            failed[k] = test
    return failed  # {} or {assumption: value != desired}


def check_assumptions(expr, against=None, **assume):
    """
    Checks whether assumptions of ``expr`` match the T/F assumptions
    given (or possessed by ``against``). True is returned if all
    assumptions match; False is returned if there is a mismatch and
    the assumption in ``expr`` is not None; else None is returned.

    Explanation
    ===========

    *assume* is a dict of assumptions with True or False values

    Examples
    ========

    >>> from sympy import Symbol, pi, I, exp, check_assumptions
    >>> check_assumptions(-5, integer=True)
    True
    >>> check_assumptions(pi, real=True, integer=False)
    True
    >>> check_assumptions(pi, negative=True)
    False
    >>> check_assumptions(exp(I*pi/7), real=False)
    True
    >>> x = Symbol('x', positive=True)
    >>> check_assumptions(2*x + 1, positive=True)
    True
    >>> check_assumptions(-2*x - 5, positive=True)
    False

    To check assumptions of *expr* against another variable or expression,
    pass the expression or variable as ``against``.

    >>> check_assumptions(2*x + 1, x)
    True

    To see if a number matches the assumptions of an expression, pass
    the number as the first argument, else its specific assumptions
    may not have a non-None value in the expression:

    >>> check_assumptions(x, 3)
    >>> check_assumptions(3, x)
    True

    ``None`` is returned if ``check_assumptions()`` could not conclude.

    >>> check_assumptions(2*x - 1, x)

    >>> z = Symbol('z')
    >>> check_assumptions(z, real=True)

    See Also
    ========

    failing_assumptions

    """
    expr = sympify(expr)
    if against is not None:
        if assume:
            raise ValueError(
                'Expecting `against` or `assume`, not both.')
        assume = assumptions(against)
    known = True
    for k, v in assume.items():
        if v is None:
            continue
        e = getattr(expr, 'is_' + k, None)
        if e is None:
            known = None
        elif v != e:
            return False
    return known


class StdFactKB(FactKB):
    """A FactKB specialized for the built-in rules

    This is the only kind of FactKB that Basic objects should use.
    """
    def __init__(self, facts=None):
        super().__init__(_assume_rules)
        # save a copy of the facts dict
        if not facts:
            self._generator = {}
        elif not isinstance(facts, FactKB):
            self._generator = facts.copy()
        else:
            self._generator = facts.generator
        if facts:
            self.deduce_all_facts(facts)

    def copy(self):
        return self.__class__(self)

    @property
    def generator(self):
        return self._generator.copy()


def as_property(fact):
    """Convert a fact name to the name of the corresponding property"""
    return 'is_%s' % fact


def make_property(fact):
    """Create the automagic property corresponding to a fact."""

    def getit(self):
        try:
            return self._assumptions[fact]
        except KeyError:
            if self._assumptions is self.default_assumptions:
                self._assumptions = self.default_assumptions.copy()
            return _ask(fact, self)

    getit.func_name = as_property(fact)
    return property(getit)


def _ask(fact, obj):
    """
    Find the truth value for a property of an object.

    This function is called when a request is made to see what a fact
    value is.

    For this we use several techniques:

    First, the fact-evaluation function is tried, if it exists (for
    example _eval_is_integer). Then we try related facts. For example

        rational   -->   integer

    another example is joined rule:

        integer & !odd  --> even

    so in the latter case if we are looking at what 'even' value is,
    'integer' and 'odd' facts will be asked.

    In all cases, when we settle on some fact value, its implications are
    deduced, and the result is cached in ._assumptions.
    """
    # FactKB which is dict-like and maps facts to their known values:
    assumptions = obj._assumptions

    # A dict that maps facts to their handlers:
    handler_map = obj._prop_handler

    # This is our queue of facts to check:
    facts_to_check = [fact]
    facts_queued = {fact}

    # Loop over the queue as it extends
    for fact_i in facts_to_check:

        # If fact_i has already been determined then we don't need to rerun the
        # handler. There is a potential race condition for multithreaded code
        # though because it's possible that fact_i was checked in another
        # thread. The main logic of the loop below would potentially skip
        # checking assumptions[fact] in this case so we check it once after the
        # loop to be sure.
        if fact_i in assumptions:
            continue

        # Now we call the associated handler for fact_i if it exists.
        fact_i_value = None
        handler_i = handler_map.get(fact_i)
        if handler_i is not None:
            fact_i_value = handler_i(obj)

        # If we get a new value for fact_i then we should update our knowledge
        # of fact_i as well as any related facts that can be inferred using the
        # inference rules connecting the fact_i and any other fact values that
        # are already known.
        if fact_i_value is not None:
            assumptions.deduce_all_facts(((fact_i, fact_i_value),))

        # Usually if assumptions[fact] is now not None then that is because of
        # the call to deduce_all_facts above. The handler for fact_i returned
        # True or False and knowing fact_i (which is equal to fact in the first
        # iteration) implies knowing a value for fact. It is also possible
        # though that independent code e.g. called indirectly by the handler or
        # called in another thread in a multithreaded context might have
        # resulted in assumptions[fact] being set. Either way we return it.
        fact_value = assumptions.get(fact)
        if fact_value is not None:
            return fact_value

        # Extend the queue with other facts that might determine fact_i. Here
        # we randomise the order of the facts that are checked. This should not
        # lead to any non-determinism if all handlers are logically consistent
        # with the inference rules for the facts. Non-deterministic assumptions
        # queries can result from bugs in the handlers that are exposed by this
        # call to shuffle. These are pushed to the back of the queue meaning
        # that the inference graph is traversed in breadth-first order.
        new_facts_to_check = list(_assume_rules.prereq[fact_i] - facts_queued)
        shuffle(new_facts_to_check)
        facts_to_check.extend(new_facts_to_check)
        facts_queued.update(new_facts_to_check)

    # The above loop should be able to handle everything fine in a
    # single-threaded context but in multithreaded code it is possible that
    # this thread skipped computing a particular fact that was computed in
    # another thread (due to the continue). In that case it is possible that
    # fact was inferred and is now stored in the assumptions dict but it wasn't
    # checked for in the body of the loop. This is an obscure case but to make
    # sure we catch it we check once here at the end of the loop.
    if fact in assumptions:
        return assumptions[fact]

    # This query can not be answered. It's possible that e.g. another thread
    # has already stored None for fact but assumptions._tell does not mind if
    # we call _tell twice setting the same value. If this raises
    # InconsistentAssumptions then it probably means that another thread
    # attempted to compute this and got a value of True or False rather than
    # None. In that case there must be a bug in at least one of the handlers.
    # If the handlers are all deterministic and are consistent with the
    # inference rules then the same value should be computed for fact in all
    # threads.
    assumptions._tell(fact, None)
    return None


def _prepare_class_assumptions(cls):
    """Precompute class level assumptions and generate handlers.

    This is called by Basic.__init_subclass__ each time a Basic subclass is
    defined.
    """

    local_defs = {}
    for k in _assume_defined:
        attrname = as_property(k)
        v = cls.__dict__.get(attrname, '')
        if isinstance(v, (bool, int, type(None))):
            if v is not None:
                v = bool(v)
            local_defs[k] = v

    defs = {}
    for base in reversed(cls.__bases__):
        assumptions = getattr(base, '_explicit_class_assumptions', None)
        if assumptions is not None:
            defs.update(assumptions)
    defs.update(local_defs)

    cls._explicit_class_assumptions = defs
    cls.default_assumptions = StdFactKB(defs)

    cls._prop_handler = {}
    for k in _assume_defined:
        eval_is_meth = getattr(cls, '_eval_is_%s' % k, None)
        if eval_is_meth is not None:
            cls._prop_handler[k] = eval_is_meth

    # Put definite results directly into the class dict, for speed
    for k, v in cls.default_assumptions.items():
        setattr(cls, as_property(k), v)

    # protection e.g. for Integer.is_even=F <- (Rational.is_integer=F)
    derived_from_bases = set()
    for base in cls.__bases__:
        default_assumptions = getattr(base, 'default_assumptions', None)
        # is an assumption-aware class
        if default_assumptions is not None:
            derived_from_bases.update(default_assumptions)

    for fact in derived_from_bases - set(cls.default_assumptions):
        pname = as_property(fact)
        if pname not in cls.__dict__:
            setattr(cls, pname, make_property(fact))

    # Finally, add any missing automagic property (e.g. for Basic)
    for fact in _assume_defined:
        pname = as_property(fact)
        if not hasattr(cls, pname):
            setattr(cls, pname, make_property(fact))


# XXX: ManagedProperties used to be the metaclass for Basic but now Basic does
# not use a metaclass. We leave this here for backwards compatibility for now
# in case someone has been using the ManagedProperties class in downstream
# code. The reason that it might have been used is that when subclassing a
# class and wanting to use a metaclass the metaclass must be a subclass of the
# metaclass for the class that is being subclassed. Anyone wanting to subclass
# Basic and use a metaclass in their subclass would have needed to subclass
# ManagedProperties. Here ManagedProperties is not the metaclass for Basic any
# more but it should still be usable as a metaclass for Basic subclasses since
# it is a subclass of type which is now the metaclass for Basic.
class ManagedProperties(type):
    def __init__(cls, *args, **kwargs):
        msg = ("The ManagedProperties metaclass. "
               "Basic does not use metaclasses any more")
        sympy_deprecation_warning(msg,
            deprecated_since_version="1.12",
            active_deprecations_target='managedproperties')

        # Here we still call this function in case someone is using
        # ManagedProperties for something that is not a Basic subclass. For
        # Basic subclasses this function is now called by __init_subclass__ and
        # so this metaclass is not needed any more.
        _prepare_class_assumptions(cls)
<|endoftext|>from __future__ import annotations

from typing import overload, TYPE_CHECKING

from operator import attrgetter
from collections import defaultdict

from sympy.utilities.exceptions import sympy_deprecation_warning

from .sympify import _sympify as _sympify_, sympify
from .basic import Basic
from .cache import cacheit
from .sorting import ordered
from .logic import fuzzy_and
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from sympy.multipledispatch.dispatcher import (Dispatcher,
    ambiguity_register_error_ignore_dup,
    str_signature, RaiseNotImplementedError)


if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    from sympy.logic.boolalg import Boolean, And, Or


class AssocOp(Basic):
    """ Associative operations, can separate noncommutative and
    commutative parts.

    (a op b) op c == a op (b op c) == a op b op c.

    Base class for Add and Mul.

    This is an abstract base class, concrete derived classes must define
    the attribute `identity`.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Parameters
    ==========

    *args :
        Arguments which are operated

    evaluate : bool, optional
        Evaluate the operation. If not passed, refer to ``global_parameters.evaluate``.
    """

    # for performance reason, we don't let is_commutative go to assumptions,
    # and keep it right here
    __slots__: tuple[str, ...] = ('is_commutative',)

    _args_type: type[Basic] | None = None

    @cacheit
    def __new__(cls, *args, evaluate=None, _sympify=True):
        # Allow faster processing by passing ``_sympify=False``, if all arguments
        # are already sympified.
        if _sympify:
            args = list(map(_sympify_, args))

        # Disallow non-Expr args in Add/Mul
        typ = cls._args_type
        if typ is not None:
            from .relational import Relational
            if any(isinstance(arg, Relational) for arg in args):
                raise TypeError("Relational cannot be used in %s" % cls.__name__)

            # This should raise TypeError once deprecation period is over:
            for arg in args:
                if not isinstance(arg, typ):
                    sympy_deprecation_warning(
                        f"""

Using non-Expr arguments in {cls.__name__} is deprecated (in this case, one of
the arguments has type {type(arg).__name__!r}).

If you really did intend to use a multiplication or addition operation with
this object, use the * or + operator instead.

                        """,
                        deprecated_since_version="1.7",
                        active_deprecations_target="non-expr-args-deprecated",
                        stacklevel=4,
                    )

        if evaluate is None:
            evaluate = global_parameters.evaluate
        if not evaluate:
            obj = cls._from_args(args)
            obj = cls._exec_constructor_postprocessors(obj)
            return obj

        args = [a for a in args if a is not cls.identity]

        if len(args) == 0:
            return cls.identity
        if len(args) == 1:
            return args[0]

        c_part, nc_part, order_symbols = cls.flatten(args)
        is_commutative = not nc_part
        obj = cls._from_args(c_part + nc_part, is_commutative)
        obj = cls._exec_constructor_postprocessors(obj)

        if order_symbols is not None:
            from sympy.series.order import Order
            return Order(obj, *order_symbols)
        return obj

    @classmethod
    def _from_args(cls, args, is_commutative=None):
        """Create new instance with already-processed args.
        If the args are not in canonical order, then a non-canonical
        result will be returned, so use with caution. The order of
        args may change if the sign of the args is changed."""
        if len(args) == 0:
            return cls.identity
        elif len(args) == 1:
            return args[0]

        obj = super().__new__(cls, *args)
        if is_commutative is None:
            is_commutative = fuzzy_and(a.is_commutative for a in args)
        obj.is_commutative = is_commutative
        return obj

    def _new_rawargs(self, *args, reeval=True, **kwargs):
        """Create new instance of own class with args exactly as provided by
        caller but returning the self class identity if args is empty.

        Examples
        ========

           This is handy when we want to optimize things, e.g.

               >>> from sympy import Mul, S
               >>> from sympy.abc import x, y
               >>> e = Mul(3, x, y)
               >>> e.args
               (3, x, y)
               >>> Mul(*e.args[1:])
               x*y
               >>> e._new_rawargs(*e.args[1:])  # the same as above, but faster
               x*y

           Note: use this with caution. There is no checking of arguments at
           all. This is best used when you are rebuilding an Add or Mul after
           simply removing one or more args. If, for example, modifications,
           result in extra 1s being inserted they will show up in the result:

               >>> m = (x*y)._new_rawargs(S.One, x); m
               1*x
               >>> m == x
               False
               >>> m.is_Mul
               True

           Another issue to be aware of is that the commutativity of the result
           is based on the commutativity of self. If you are rebuilding the
           terms that came from a commutative object then there will be no
           problem, but if self was non-commutative then what you are
           rebuilding may now be commutative.

           Although this routine tries to do as little as possible with the
           input, getting the commutativity right is important, so this level
           of safety is enforced: commutativity will always be recomputed if
           self is non-commutative and kwarg `reeval=False` has not been
           passed.
        """
        if reeval and self.is_commutative is False:
            is_commutative = None
        else:
            is_commutative = self.is_commutative
        return self._from_args(args, is_commutative)

    @classmethod
    def flatten(cls, seq):
        """Return seq so that none of the elements are of type `cls`. This is
        the vanilla routine that will be used if a class derived from AssocOp
        does not define its own flatten routine."""
        # apply associativity, no commutativity property is used
        new_seq = []
        while seq:
            o = seq.pop()
            if o.__class__ is cls:  # classes must match exactly
                seq.extend(o.args)
            else:
                new_seq.append(o)
        new_seq.reverse()

        # c_part, nc_part, order_symbols
        return [], new_seq, None

    def _matches_commutative(self, expr, repl_dict=None, old=False):
        """
        Matches Add/Mul "pattern" to an expression "expr".

        repl_dict ... a dictionary of (wild: expression) pairs, that get
                      returned with the results

        This function is the main workhorse for Add/Mul.

        Examples
        ========

        >>> from sympy import symbols, Wild, sin
        >>> a = Wild("a")
        >>> b = Wild("b")
        >>> c = Wild("c")
        >>> x, y, z = symbols("x y z")
        >>> (a+sin(b)*c)._matches_commutative(x+sin(y)*z)
        {a_: x, b_: y, c_: z}

        In the example above, "a+sin(b)*c" is the pattern, and "x+sin(y)*z" is
        the expression.

        The repl_dict contains parts that were already matched. For example
        here:

        >>> (x+sin(b)*c)._matches_commutative(x+sin(y)*z, repl_dict={a: x})
        {a_: x, b_: y, c_: z}

        the only function of the repl_dict is to return it in the
        result, e.g. if you omit it:

        >>> (x+sin(b)*c)._matches_commutative(x+sin(y)*z)
        {b_: y, c_: z}

        the "a: x" is not returned in the result, but otherwise it is
        equivalent.

        """
        from .function import _coeff_isneg
        # make sure expr is Expr if pattern is Expr
        from .expr import Expr
        if isinstance(self, Expr) and not isinstance(expr, Expr):
            return None

        if repl_dict is None:
            repl_dict = {}

        # handle simple patterns
        if self == expr:
            return repl_dict

        d = self._matches_simple(expr, repl_dict)
        if d is not None:
            return d

        # eliminate exact part from pattern: (2+a+w1+w2).matches(expr) -> (w1+w2).matches(expr-a-2)
        from .function import WildFunction
        from .symbol import Wild
        wild_part, exact_part = sift(self.args, lambda p:
            p.has(Wild, WildFunction) and not expr.has(p),
            binary=True)
        if not exact_part:
            wild_part = list(ordered(wild_part))
            if self.is_Add:
                # in addition to normal ordered keys, impose
                # sorting on Muls with leading Number to put
                # them in order
                wild_part = sorted(wild_part, key=lambda x:
                    x.args[0] if x.is_Mul and x.args[0].is_Number else
                    0)
        else:
            exact = self._new_rawargs(*exact_part)
            free = expr.free_symbols
            if free and (exact.free_symbols - free):
                # there are symbols in the exact part that are not
                # in the expr; but if there are no free symbols, let
                # the matching continue
                return None
            newexpr = self._combine_inverse(expr, exact)
            if not old and (expr.is_Add or expr.is_Mul):
                check = newexpr
                if _coeff_isneg(check):
                    check = -check
                if check.count_ops() > expr.count_ops():
                    return None
            newpattern = self._new_rawargs(*wild_part)
            return newpattern.matches(newexpr, repl_dict)

        # now to real work ;)
        i = 0
        saw = set()
        while expr not in saw:
            saw.add(expr)
            args = tuple(ordered(self.make_args(expr)))
            if self.is_Add and expr.is_Add:
                # in addition to normal ordered keys, impose
                # sorting on Muls with leading Number to put
                # them in order
                args = tuple(sorted(args, key=lambda x:
                    x.args[0] if x.is_Mul and x.args[0].is_Number else
                    0))
            expr_list = (self.identity,) + args
            for last_op in reversed(expr_list):
                for w in reversed(wild_part):
                    d1 = w.matches(last_op, repl_dict)
                    if d1 is not None:
                        d2 = self.xreplace(d1).matches(expr, d1)
                        if d2 is not None:
                            return d2

            if i == 0:
                if self.is_Mul:
                    # make e**i look like Mul
                    if expr.is_Pow and expr.exp.is_Integer:
                        from .mul import Mul
                        if expr.exp > 0:
                            expr = Mul(*[expr.base, expr.base**(expr.exp - 1)], evaluate=False)
                        else:
                            expr = Mul(*[1/expr.base, expr.base**(expr.exp + 1)], evaluate=False)
                        i += 1
                        continue

                elif self.is_Add:
                    # make i*e look like Add
                    c, e = expr.as_coeff_Mul()
                    if abs(c) > 1:
                        from .add import Add
                        if c > 0:
                            expr = Add(*[e, (c - 1)*e], evaluate=False)
                        else:
                            expr = Add(*[-e, (c + 1)*e], evaluate=False)
                        i += 1
                        continue

                    # try collection on non-Wild symbols
                    from sympy.simplify.radsimp import collect
                    was = expr
                    did = set()
                    for w in reversed(wild_part):
                        c, w = w.as_coeff_mul(Wild)
                        free = c.free_symbols - did
                        if free:
                            did.update(free)
                            expr = collect(expr, free)
                    if expr != was:
                        i += 0
                        continue

                break  # if we didn't continue, there is nothing more to do

        return

    def _has_matcher(self):
        """Helper for .has() that checks for containment of
        subexpressions within an expr by using sets of args
        of similar nodes, e.g. x + 1 in x + y + 1 checks
        to see that {x, 1} & {x, y, 1} == {x, 1}
        """
        def _ncsplit(expr):
            # this is not the same as args_cnc because here
            # we don't assume expr is a Mul -- hence deal with args --
            # and always return a set.
            cpart, ncpart = sift(expr.args,
                lambda arg: arg.is_commutative is True, binary=True)
            return set(cpart), ncpart

        c, nc = _ncsplit(self)
        cls = self.__class__

        def is_in(expr):
            if isinstance(expr, cls):
                if expr == self:
                    return True
                _c, _nc = _ncsplit(expr)
                if (c & _c) == c:
                    if not nc:
                        return True
                    elif len(nc) <= len(_nc):
                        for i in range(len(_nc) - len(nc) + 1):
                            if _nc[i:i + len(nc)] == nc:
                                return True
            return False
        return is_in

    def _eval_evalf(self, prec):
        """
        Evaluate the parts of self that are numbers; if the whole thing
        was a number with no functions it would have been evaluated, but
        it wasn't so we must judiciously extract the numbers and reconstruct
        the object. This is *not* simply replacing numbers with evaluated
        numbers. Numbers should be handled in the largest pure-number
        expression as possible. So the code below separates ``self`` into
        number and non-number parts and evaluates the number parts and
        walks the args of the non-number part recursively (doing the same
        thing).
        """
        from .add import Add
        from .mul import Mul
        from .symbol import Symbol
        from .function import AppliedUndef
        if isinstance(self, (Mul, Add)):
            x, tail = self.as_independent(Symbol, AppliedUndef)
            # if x is an AssocOp Function then the _evalf below will
            # call _eval_evalf (here) so we must break the recursion
            if not (tail is self.identity or
                    isinstance(x, AssocOp) and x.is_Function or
                    x is self.identity and isinstance(tail, AssocOp)):
                # here, we have a number so we just call to _evalf with prec;
                # prec is not the same as n, it is the binary precision so
                # that's why we don't call to evalf.
                x = x._evalf(prec) if x is not self.identity else self.identity
                args = []
                tail_args = tuple(self.func.make_args(tail))
                for a in tail_args:
                    # here we call to _eval_evalf since we don't know what we
                    # are dealing with and all other _eval_evalf routines should
                    # be doing the same thing (i.e. taking binary prec and
                    # finding the evalf-able args)
                    newa = a._eval_evalf(prec)
                    if newa is None:
                        args.append(a)
                    else:
                        args.append(newa)
                return self.func(x, *args)

        # this is the same as above, but there were no pure-number args to
        # deal with
        args = []
        for a in self.args:
            newa = a._eval_evalf(prec)
            if newa is None:
                args.append(a)
            else:
                args.append(newa)
        return self.func(*args)

    @overload
    @classmethod
    def make_args(cls: type[Add], expr: Expr) -> tuple[Expr, ...]: ... # type: ignore
    @overload
    @classmethod
    def make_args(cls: type[Mul], expr: Expr) -> tuple[Expr, ...]: ... # type: ignore
    @overload
    @classmethod
    def make_args(cls: type[And], expr: Boolean) -> tuple[Boolean, ...]: ... # type: ignore
    @overload
    @classmethod
    def make_args(cls: type[Or], expr: Boolean) -> tuple[Boolean, ...]: ... # type: ignore

    @classmethod
    def make_args(cls: type[Basic], expr: Basic) -> tuple[Basic, ...]:
        """
        Return a sequence of elements `args` such that cls(*args) == expr

        Examples
        ========

        >>> from sympy import Symbol, Mul, Add
        >>> x, y = map(Symbol, 'xy')

        >>> Mul.make_args(x*y)
        (x, y)
        >>> Add.make_args(x*y)
        (x*y,)
        >>> set(Add.make_args(x*y + y)) == set([y, x*y])
        True

        """
        if isinstance(expr, cls):
            return expr.args
        else:
            return (sympify(expr),)

    def doit(self, **hints):
        if hints.get('deep', True):
            terms = [term.doit(**hints) for term in self.args]
        else:
            terms = self.args
        return self.func(*terms, evaluate=True)

class ShortCircuit(Exception):
    pass


class LatticeOp(AssocOp):
    """
    Join/meet operations of an algebraic lattice[1].

    Explanation
    ===========

    These binary operations are associative (op(op(a, b), c) = op(a, op(b, c))),
    commutative (op(a, b) = op(b, a)) and idempotent (op(a, a) = op(a) = a).
    Common examples are AND, OR, Union, Intersection, max or min. They have an
    identity element (op(identity, a) = a) and an absorbing element
    conventionally called zero (op(zero, a) = zero).

    This is an abstract base class, concrete derived classes must declare
    attributes zero and identity. All defining properties are then respected.

    Examples
    ========

    >>> from sympy import Integer
    >>> from sympy.core.operations import LatticeOp
    >>> class my_join(LatticeOp):
    ...     zero = Integer(0)
    ...     identity = Integer(1)
    >>> my_join(2, 3) == my_join(3, 2)
    True
    >>> my_join(2, my_join(3, 4)) == my_join(2, 3, 4)
    True
    >>> my_join(0, 1, 4, 2, 3, 4)
    0
    >>> my_join(1, 2)
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lattice_%28order%29
    """

    is_commutative = True

    def __new__(cls, *args, **options):
        args = (_sympify_(arg) for arg in args)

        try:
            # /!\ args is a generator and _new_args_filter
            # must be careful to handle as such; this
            # is done so short-circuiting can be done
            # without having to sympify all values
            _args = frozenset(cls._new_args_filter(args))
        except ShortCircuit:
            return sympify(cls.zero)
        if not _args:
            return sympify(cls.identity)
        elif len(_args) == 1:
            return set(_args).pop()
        else:
            # XXX in almost every other case for __new__, *_args is
            # passed along, but the expectation here is for _args
            obj = super(AssocOp, cls).__new__(cls, *ordered(_args))
            obj._argset = _args
            return obj

    @classmethod
    def _new_args_filter(cls, arg_sequence, call_cls=None):
        """Generator filtering args"""
        ncls = call_cls or cls
        for arg in arg_sequence:
            if arg == ncls.zero:
                raise ShortCircuit(arg)
            elif arg == ncls.identity:
                continue
            elif arg.func == ncls:
                yield from arg.args
            else:
                yield arg

    @classmethod
    def make_args(cls, expr):
        """
        Return a set of args such that cls(*arg_set) == expr.
        """
        if isinstance(expr, cls):
            return expr._argset
        else:
            return frozenset([sympify(expr)])


class AssocOpDispatcher:
    """
    Handler dispatcher for associative operators

    .. notes::
       This approach is experimental, and can be replaced or deleted in the future.
       See https://github.com/sympy/sympy/pull/19463.

    Explanation
    ===========

    If arguments of different types are passed, the classes which handle the operation for each type
    are collected. Then, a class which performs the operation is selected by recursive binary dispatching.
    Dispatching relation can be registered by ``register_handlerclass`` method.

    Priority registration is unordered. You cannot make ``A*B`` and ``B*A`` refer to
    different handler classes. All logic dealing with the order of arguments must be implemented
    in the handler class.

    Examples
    ========

    >>> from sympy import Add, Expr, Symbol
    >>> from sympy.core.add import add

    >>> class NewExpr(Expr):
    ...     @property
    ...     def _add_handler(self):
    ...         return NewAdd
    >>> class NewAdd(NewExpr, Add):
    ...     pass
    >>> add.register_handlerclass((Add, NewAdd), NewAdd)

    >>> a, b = Symbol('a'), NewExpr()
    >>> add(a, b) == NewAdd(a, b)
    True

    """
    def __init__(self, name, doc=None):
        self.name = name
        self.doc = doc
        self.handlerattr = "_%s_handler" % name
        self._handlergetter = attrgetter(self.handlerattr)
        self._dispatcher = Dispatcher(name)

    def __repr__(self):
        return "<dispatched %s>" % self.name

    def register_handlerclass(self, classes, typ, on_ambiguity=ambiguity_register_error_ignore_dup):
        """
        Register the handler class for two classes, in both straight and reversed order.

        Paramteters
        ===========

        classes : tuple of two types
            Classes who are compared with each other.

        typ:
            Class which is registered to represent *cls1* and *cls2*.
            Handler method of *self* must be implemented in this class.
        """
        if not len(classes) == 2:
            raise RuntimeError(
                "Only binary dispatch is supported, but got %s types: <%s>." % (
                len(classes), str_signature(classes)
            ))
        if len(set(classes)) == 1:
            raise RuntimeError(
                "Duplicate types <%s> cannot be dispatched." % str_signature(classes)
            )
        self._dispatcher.add(tuple(classes), typ, on_ambiguity=on_ambiguity)
        self._dispatcher.add(tuple(reversed(classes)), typ, on_ambiguity=on_ambiguity)

    @cacheit
    def __call__(self, *args, _sympify=True, **kwargs):
        """
        Parameters
        ==========

        *args :
            Arguments which are operated
        """
        if _sympify:
            args = tuple(map(_sympify_, args))
        handlers = frozenset(map(self._handlergetter, args))

        # no need to sympify again
        return self.dispatch(handlers)(*args, _sympify=False, **kwargs)

    @cacheit
    def dispatch(self, handlers):
        """
        Select the handler class, and return its handler method.
        """

        # Quick exit for the case where all handlers are same
        if len(handlers) == 1:
            h, = handlers
            if not isinstance(h, type):
                raise RuntimeError("Handler {!r} is not a type.".format(h))
            return h

        # Recursively select with registered binary priority
        for i, typ in enumerate(handlers):

            if not isinstance(typ, type):
                raise RuntimeError("Handler {!r} is not a type.".format(typ))

            if i == 0:
                handler = typ
            else:
                prev_handler = handler
                handler = self._dispatcher.dispatch(prev_handler, typ)

                if not isinstance(handler, type):
                    raise RuntimeError(
                        "Dispatcher for {!r} and {!r} must return a type, but got {!r}".format(
                        prev_handler, typ, handler
                    ))

        # return handler class
        return handler

    @property
    def __doc__(self):
        docs = [
            "Multiply dispatched associative operator: %s" % self.name,
            "Note that support for this is experimental, see the docs for :class:`AssocOpDispatcher` for details"
        ]

        if self.doc:
            docs.append(self.doc)

        s = "Registered handler classes\n"
        s += '=' * len(s)
        docs.append(s)

        amb_sigs = []

        typ_sigs = defaultdict(list)
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)

        for typ, sigs in typ_sigs.items():

            sigs_str = ', '.join('<%s>' % str_signature(sig) for sig in sigs)

            if isinstance(typ, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue

            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            s += typ.__name__
            docs.append(s)

        if amb_sigs:
            s = "Ambiguous handler classes\n"
            s += '=' * len(s)
            docs.append(s)

            s = '\n'.join(amb_sigs)
            docs.append(s)

        return '\n\n'.join(docs)
<|endoftext|>from __future__ import annotations

from typing import Iterator

from .basic import Basic
from .sorting import ordered
from .sympify import sympify
from sympy.utilities.iterables import iterable



def iterargs(expr):
    """Yield the args of a Basic object in a breadth-first traversal.
    Depth-traversal stops if `arg.args` is either empty or is not
    an iterable.

    Examples
    ========

    >>> from sympy import Integral, Function
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> from sympy.core.traversal import iterargs
    >>> list(iterargs(Integral(f(x), (f(x), 1))))
    [Integral(f(x), (f(x), 1)), f(x), (f(x), 1), x, f(x), 1, x]

    See Also
    ========
    iterfreeargs, preorder_traversal
    """
    args = [expr]
    for i in args:
        yield i
        args.extend(i.args)


def iterfreeargs(expr, _first=True):
    """Yield the args of a Basic object in a breadth-first traversal.
    Depth-traversal stops if `arg.args` is either empty or is not
    an iterable. The bound objects of an expression will be returned
    as canonical variables.

    Examples
    ========

    >>> from sympy import Integral, Function
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> from sympy.core.traversal import iterfreeargs
    >>> list(iterfreeargs(Integral(f(x), (f(x), 1))))
    [Integral(f(x), (f(x), 1)), 1]

    See Also
    ========
    iterargs, preorder_traversal
    """
    args = [expr]
    for i in args:
        yield i
        if _first and hasattr(i, 'bound_symbols'):
            void = i.canonical_variables.values()
            for i in iterfreeargs(i.as_dummy(), _first=False):
                if not i.has(*void):
                    yield i
        args.extend(i.args)


class preorder_traversal:
    """
    Do a pre-order traversal of a tree.

    This iterator recursively yields nodes that it has visited in a pre-order
    fashion. That is, it yields the current node then descends through the
    tree breadth-first to yield all of a node's children's pre-order
    traversal.


    For an expression, the order of the traversal depends on the order of
    .args, which in many cases can be arbitrary.

    Parameters
    ==========
    node : SymPy expression
        The expression to traverse.
    keys : (default None) sort key(s)
        The key(s) used to sort args of Basic objects. When None, args of Basic
        objects are processed in arbitrary order. If key is defined, it will
        be passed along to ordered() as the only key(s) to use to sort the
        arguments; if ``key`` is simply True then the default keys of ordered
        will be used.

    Yields
    ======
    subtree : SymPy expression
        All of the subtrees in the tree.

    Examples
    ========

    >>> from sympy import preorder_traversal, symbols
    >>> x, y, z = symbols('x y z')

    The nodes are returned in the order that they are encountered unless key
    is given; simply passing key=True will guarantee that the traversal is
    unique.

    >>> list(preorder_traversal((x + y)*z, keys=None)) # doctest: +SKIP
    [z*(x + y), z, x + y, y, x]
    >>> list(preorder_traversal((x + y)*z, keys=True))
    [z*(x + y), z, x + y, x, y]

    """
    def __init__(self, node, keys=None):
        self._skip_flag = False
        self._pt = self._preorder_traversal(node, keys)

    def _preorder_traversal(self, node, keys):
        yield node
        if self._skip_flag:
            self._skip_flag = False
            return
        if isinstance(node, Basic):
            if not keys and hasattr(node, '_argset'):
                # LatticeOp keeps args as a set. We should use this if we
                # don't care about the order, to prevent unnecessary sorting.
                args = node._argset
            else:
                args = node.args
            if keys:
                if keys != True:
                    args = ordered(args, keys, default=False)
                else:
                    args = ordered(args)
            for arg in args:
                yield from self._preorder_traversal(arg, keys)
        elif iterable(node):
            for item in node:
                yield from self._preorder_traversal(item, keys)

    def skip(self):
        """
        Skip yielding current node's (last yielded node's) subtrees.

        Examples
        ========

        >>> from sympy import preorder_traversal, symbols
        >>> x, y, z = symbols('x y z')
        >>> pt = preorder_traversal((x + y*z)*z)
        >>> for i in pt:
        ...     print(i)
        ...     if i == x + y*z:
        ...             pt.skip()
        z*(x + y*z)
        z
        x + y*z
        """
        self._skip_flag = True

    def __next__(self):
        return next(self._pt)

    def __iter__(self) -> Iterator[Basic]:
        return self


def use(expr, func, level=0, args=(), kwargs={}):
    """
    Use ``func`` to transform ``expr`` at the given level.

    Examples
    ========

    >>> from sympy import use, expand
    >>> from sympy.abc import x, y

    >>> f = (x + y)**2*x + 1

    >>> use(f, expand, level=2)
    x*(x**2 + 2*x*y + y**2) + 1
    >>> expand(f)
    x**3 + 2*x**2*y + x*y**2 + 1

    """
    def _use(expr, level):
        if not level:
            return func(expr, *args, **kwargs)
        else:
            if expr.is_Atom:
                return expr
            else:
                level -= 1
                _args = [_use(arg, level) for arg in expr.args]
                return expr.__class__(*_args)

    return _use(sympify(expr), level)


def walk(e, *target):
    """Iterate through the args that are the given types (target) and
    return a list of the args that were traversed; arguments
    that are not of the specified types are not traversed.

    Examples
    ========

    >>> from sympy.core.traversal import walk
    >>> from sympy import Min, Max
    >>> from sympy.abc import x, y, z
    >>> list(walk(Min(x, Max(y, Min(1, z))), Min))
    [Min(x, Max(y, Min(1, z)))]
    >>> list(walk(Min(x, Max(y, Min(1, z))), Min, Max))
    [Min(x, Max(y, Min(1, z))), Max(y, Min(1, z)), Min(1, z)]

    See Also
    ========

    bottom_up
    """
    if isinstance(e, target):
        yield e
        for i in e.args:
            yield from walk(i, *target)


def bottom_up(rv, F, atoms=False, nonbasic=False):
    """Apply ``F`` to all expressions in an expression tree from the
    bottom up. If ``atoms`` is True, apply ``F`` even if there are no args;
    if ``nonbasic`` is True, try to apply ``F`` to non-Basic objects.
    """
    args = getattr(rv, 'args', None)
    if args is not None:
        if args:
            args = tuple([bottom_up(a, F, atoms, nonbasic) for a in args])
            if args != rv.args:
                rv = rv.func(*args)
            rv = F(rv)
        elif atoms:
            rv = F(rv)
    else:
        if nonbasic:
            try:
                rv = F(rv)
            except TypeError:
                pass

    return rv


def postorder_traversal(node, keys=None):
    """
    Do a postorder traversal of a tree.

    This generator recursively yields nodes that it has visited in a postorder
    fashion. That is, it descends through the tree depth-first to yield all of
    a node's children's postorder traversal before yielding the node itself.

    Parameters
    ==========

    node : SymPy expression
        The expression to traverse.
    keys : (default None) sort key(s)
        The key(s) used to sort args of Basic objects. When None, args of Basic
        objects are processed in arbitrary order. If key is defined, it will
        be passed along to ordered() as the only key(s) to use to sort the
        arguments; if ``key`` is simply True then the default keys of
        ``ordered`` will be used (node count and default_sort_key).

    Yields
    ======
    subtree : SymPy expression
        All of the subtrees in the tree.

    Examples
    ========

    >>> from sympy import postorder_traversal
    >>> from sympy.abc import w, x, y, z

    The nodes are returned in the order that they are encountered unless key
    is given; simply passing key=True will guarantee that the traversal is
    unique.

    >>> list(postorder_traversal(w + (x + y)*z)) # doctest: +SKIP
    [z, y, x, x + y, z*(x + y), w, w + z*(x + y)]
    >>> list(postorder_traversal(w + (x + y)*z, keys=True))
    [w, z, x, y, x + y, z*(x + y), w + z*(x + y)]


    """
    if isinstance(node, Basic):
        args = node.args
        if keys:
            if keys != True:
                args = ordered(args, keys, default=False)
            else:
                args = ordered(args)
        for arg in args:
            yield from postorder_traversal(arg, keys)
    elif iterable(node):
        for item in node:
            yield from postorder_traversal(item, keys)
    yield node
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, overload
from collections.abc import Iterable, Mapping
from functools import reduce
import re

from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .cache import cacheit
from .logic import fuzzy_or, fuzzy_not
from .intfunc import mod_inverse
from .sorting import default_sort_key
from .kind import NumberKind
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int, func_name, filldedent
from sympy.utilities.iterables import has_variety, sift
from mpmath.libmp import mpf_log, prec_to_dps
from mpmath.libmp.libintmath import giant_steps


if TYPE_CHECKING:
    from typing import Any
    from typing_extensions import Self
    from .numbers import Number

from collections import defaultdict


def _corem(eq, c):  # helper for extract_additively
    # return co, diff from co*c + diff
    co = []
    non = []
    for i in Add.make_args(eq):
        ci = i.coeff(c)
        if not ci:
            non.append(i)
        else:
            co.append(ci)
    return Add(*co), Add(*non)


@sympify_method_args
class Expr(Basic, EvalfMixin):
    """
    Base class for algebraic expressions.

    Explanation
    ===========

    Everything that requires arithmetic operations to be defined
    should subclass this class, instead of Basic (which should be
    used only for argument storage and expression manipulation, i.e.
    pattern matching, substitutions, etc).

    If you want to override the comparisons of expressions:
    Should use _eval_is_ge for inequality, or _eval_is_eq, with multiple dispatch.
    _eval_is_ge return true if x >= y, false if x < y, and None if the two types
    are not comparable or the comparison is indeterminate

    See Also
    ========

    sympy.core.basic.Basic
    """

    __slots__: tuple[str, ...] = ()

    if TYPE_CHECKING:

        def __new__(cls, *args: Basic) -> Self:
            ...

        @overload # type: ignore
        def subs(self, arg1: Mapping[Basic | complex, Expr | complex], arg2: None=None) -> Expr: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Expr | complex]], arg2: None=None, **kwargs: Any) -> Expr: ...
        @overload
        def subs(self, arg1: Expr | complex, arg2: Expr | complex) -> Expr: ...
        @overload
        def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
        @overload
        def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

        def subs(self, arg1: Mapping[Basic | complex, Basic | complex] | Basic | complex, # type: ignore
                 arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
            ...

        def simplify(self, **kwargs) -> Expr:
            ...

        def evalf(self, n: int = 15, subs: dict[Basic, Basic | float] | None = None,
                  maxn: int = 100, chop: bool = False, strict: bool  = False,
                  quad: str | None = None, verbose: bool = False) -> Expr:
            ...

        n = evalf

    is_scalar = True  # self derivative is 1

    @property
    def _diff_wrt(self):
        """Return True if one can differentiate with respect to this
        object, else False.

        Explanation
        ===========

        Subclasses such as Symbol, Function and Derivative return True
        to enable derivatives wrt them. The implementation in Derivative
        separates the Symbol and non-Symbol (_diff_wrt=True) variables and
        temporarily converts the non-Symbols into Symbols when performing
        the differentiation. By default, any object deriving from Expr
        will behave like a scalar with self.diff(self) == 1. If this is
        not desired then the object must also set `is_scalar = False` or
        else define an _eval_derivative routine.

        Note, see the docstring of Derivative for how this should work
        mathematically. In particular, note that expr.subs(yourclass, Symbol)
        should be well-defined on a structural level, or this will lead to
        inconsistent results.

        Examples
        ========

        >>> from sympy import Expr
        >>> e = Expr()
        >>> e._diff_wrt
        False
        >>> class MyScalar(Expr):
        ...     _diff_wrt = True
        ...
        >>> MyScalar().diff(MyScalar())
        1
        >>> class MySymbol(Expr):
        ...     _diff_wrt = True
        ...     is_scalar = False
        ...
        >>> MySymbol().diff(MySymbol())
        Derivative(MySymbol(), MySymbol())
        """
        return False

    @cacheit
    def sort_key(self, order=None):

        coeff, expr = self.as_coeff_Mul()

        if expr.is_Pow:
            base, exp = expr.as_base_exp()
            if base is S.Exp1:
                # If we remove this, many doctests will go crazy:
                # (keeps E**x sorted like the exp(x) function,
                #  part of exp(x) to E**x transition)
                base, exp = Function("exp")(exp), S.One
            expr = base
        else:
            exp = S.One

        if expr.is_Dummy:
            args = (expr.sort_key(),)
        elif expr.is_Atom:
            args = (str(expr),)
        else:
            if expr.is_Add:
                args = expr.as_ordered_terms(order=order)
            elif expr.is_Mul:
                args = expr.as_ordered_factors(order=order)
            else:
                args = expr.args

            args = tuple(
                [ default_sort_key(arg, order=order) for arg in args ])

        args = (len(args), tuple(args))
        exp = exp.sort_key(order=order)

        return expr.class_key(), args, exp, coeff

    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.
        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args

    # ***************
    # * Arithmetics *
    # ***************
    # Expr and its subclasses use _op_priority to determine which object
    # passed to a binary special method (__mul__, etc.) will handle the
    # operation. In general, the 'call_highest_priority' decorator will choose
    # the object with the highest _op_priority to handle the call.
    # Custom subclasses that want to define their own binary special methods
    # should set an _op_priority value that is higher than the default.
    #
    # **NOTE**:
    # This is a temporary fix, and will eventually be replaced with
    # something better and more powerful.  See issue 5510.
    _op_priority = 10.0

    @property
    def _add_handler(self):
        return Add

    @property
    def _mul_handler(self):
        return Mul

    def __pos__(self) -> Expr:
        return self

    def __neg__(self) -> Expr:
        # Mul has its own __neg__ routine, so we just
        # create a 2-args Mul with the -1 in the canonical
        # slot 0.
        c = self.is_commutative
        return Mul._from_args((S.NegativeOne, self), c)

    def __abs__(self) -> Expr:
        from sympy.functions.elementary.complexes import Abs
        return Abs(self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other) -> Expr:
        return Add(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other) -> Expr:
        return Add(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other) -> Expr:
        return Add(self, -other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other) -> Expr:
        return Add(other, -self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other) -> Expr:
        return Mul(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other) -> Expr:
        return Mul(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rpow__')
    def _pow(self, other):
        return Pow(self, other)

    def __pow__(self, other, mod=None) -> Expr:
        if mod is None:
            return self._pow(other)
        try:
            _self, other, mod = as_int(self), as_int(other), as_int(mod)
            if other >= 0:
                return _sympify(pow(_self, other, mod))
            else:
                return _sympify(mod_inverse(pow(_self, -other, mod), mod))
        except ValueError:
            power = self._pow(other)
            try:
                return power%mod
            except TypeError:
                return NotImplemented

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other) -> Expr:
        return Pow(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other) -> Expr:
        denom = Pow(other, S.NegativeOne)
        if self is S.One:
            return denom
        else:
            return Mul(self, denom)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other) -> Expr:
        denom = Pow(self, S.NegativeOne)
        if other is S.One:
            return denom
        else:
            return Mul(other, denom)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rmod__')
    def __mod__(self, other) -> Expr:
        return Mod(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__mod__')
    def __rmod__(self, other) -> Expr:
        return Mod(other, self)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rfloordiv__')
    def __floordiv__(self, other) -> Expr:
        from sympy.functions.elementary.integers import floor
        return floor(self / other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__floordiv__')
    def __rfloordiv__(self, other) -> Expr:
        from sympy.functions.elementary.integers import floor
        return floor(other / self)


    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__rdivmod__')
    def __divmod__(self, other) -> tuple[Expr, Expr]:
        from sympy.functions.elementary.integers import floor
        return floor(self / other), Mod(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    @call_highest_priority('__divmod__')
    def __rdivmod__(self, other) -> tuple[Expr, Expr]:
        from sympy.functions.elementary.integers import floor
        return floor(other / self), Mod(other, self)

    def __int__(self) -> int:
        if not self.is_number:
            raise TypeError("Cannot convert symbols to int")
        r = self.round(2)
        if not r.is_Number:
            raise TypeError("Cannot convert complex to int")
        if r in (S.NaN, S.Infinity, S.NegativeInfinity):
            raise TypeError("Cannot convert %s to int" % r)
        i = int(r)
        if not i:
            return i
        if int_valued(r):
            # non-integer self should pass one of these tests
            if (self > i) is S.true:
                return i
            if (self < i) is S.true:
                return i - 1
            ok = self.equals(i)
            if ok is None:
                raise TypeError('cannot compute int value accurately')
            if ok:
                return i
            # off by one
            return i - (1 if i > 0 else -1)
        return i

    def __float__(self) -> float:
        # Don't bother testing if it's a number; if it's not this is going
        # to fail, and if it is we still need to check that it evalf'ed to
        # a number.
        result = self.evalf()
        if result.is_Number:
            return float(result)
        if result.is_number and result.as_real_imag()[1]:
            raise TypeError("Cannot convert complex to float")
        raise TypeError("Cannot convert expression to float")

    def __complex__(self) -> complex:
        result = self.evalf()
        re, im = result.as_real_imag()
        return complex(float(re), float(im))

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __ge__(self, other):
        from .relational import GreaterThan
        return GreaterThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __le__(self, other):
        from .relational import LessThan
        return LessThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __gt__(self, other):
        from .relational import StrictGreaterThan
        return StrictGreaterThan(self, other)

    @sympify_return([('other', 'Expr')], NotImplemented)
    def __lt__(self, other):
        from .relational import StrictLessThan
        return StrictLessThan(self, other)

    def __trunc__(self):
        if not self.is_number:
            raise TypeError("Cannot truncate symbols and expressions")
        else:
            return Integer(self)

    def __format__(self, format_spec: str):
        if self.is_number:
            mt = re.match(r'\+?\d*\.(\d+)f', format_spec)
            if mt:
                prec = int(mt.group(1))
                rounded = self.round(prec)
                if rounded.is_Integer:
                    return format(int(rounded), format_spec)
                if rounded.is_Float:
                    return format(rounded, format_spec)
        return super().__format__(format_spec)

    @staticmethod
    def _from_mpmath(x, prec):
        if hasattr(x, "_mpf_"):
            return Float._new(x._mpf_, prec)
        elif hasattr(x, "_mpc_"):
            re, im = x._mpc_
            re = Float._new(re, prec)
            im = Float._new(im, prec)*S.ImaginaryUnit
            return re + im
        else:
            raise TypeError("expected mpmath number (mpf or mpc)")

    @property
    def is_number(self):
        """Returns True if ``self`` has no free symbols and no
        undefined functions (AppliedUndef, to be precise). It will be
        faster than ``if not self.free_symbols``, however, since
        ``is_number`` will fail as soon as it hits a free symbol
        or undefined function.

        Examples
        ========

        >>> from sympy import Function, Integral, cos, sin, pi
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> x.is_number
        False
        >>> f(1).is_number
        False
        >>> (2*x).is_number
        False
        >>> (2 + Integral(2, x)).is_number
        False
        >>> (2 + Integral(2, (x, 1, 2))).is_number
        True

        Not all numbers are Numbers in the SymPy sense:

        >>> pi.is_number, pi.is_Number
        (True, False)

        If something is a number it should evaluate to a number with
        real and imaginary parts that are Numbers; the result may not
        be comparable, however, since the real and/or imaginary part
        of the result may not have precision.

        >>> cos(1).is_number and cos(1).is_comparable
        True

        >>> z = cos(1)**2 + sin(1)**2 - 1
        >>> z.is_number
        True
        >>> z.is_comparable
        False

        See Also
        ========

        sympy.core.basic.Basic.is_comparable
        """
        return all(obj.is_number for obj in self.args)

    def _eval_is_comparable(self):
        # Basic._eval_is_comparable always returns False, so we override it
        # here
        is_extended_real = self.is_extended_real
        if is_extended_real is False:
            return False
        if not self.is_number:
            return False

        # XXX: as_real_imag() can be a very expensive operation. It should not
        # be used here because is_comparable is used implicitly in many places.
        # Probably this method should just return self.evalf(2).is_Number.

        n, i = self.as_real_imag()

        if not n.is_Number:
            n = n.evalf(2)
            if not n.is_Number:
                return False

        if not i.is_Number:
            i = i.evalf(2)
            if not i.is_Number:
                return False

        if i:
            # if _prec = 1 we can't decide and if not,
            # the answer is False because numbers with
            # imaginary parts can't be compared
            # so return False
            return False
        else:
            return n._prec != 1

    def _random(self, n=None, re_min=-1, im_min=-1, re_max=1, im_max=1):
        """Return self evaluated, if possible, replacing free symbols with
        random complex values, if necessary.

        Explanation
        ===========

        The random complex value for each free symbol is generated
        by the random_complex_number routine giving real and imaginary
        parts in the range given by the re_min, re_max, im_min, and im_max
        values. The returned value is evaluated to a precision of n
        (if given) else the maximum of 15 and the precision needed
        to get more than 1 digit of precision. If the expression
        could not be evaluated to a number, or could not be evaluated
        to more than 1 digit of precision, then None is returned.

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.abc import x, y
        >>> x._random()                         # doctest: +SKIP
        0.0392918155679172 + 0.916050214307199*I
        >>> x._random(2)                        # doctest: +SKIP
        -0.77 - 0.87*I
        >>> (x + y/2)._random(2)                # doctest: +SKIP
        -0.57 + 0.16*I
        >>> sqrt(2)._random(2)
        1.4

        See Also
        ========

        sympy.core.random.random_complex_number
        """

        free = self.free_symbols
        prec = 1
        if free:
            from sympy.core.random import random_complex_number
            a, c, b, d = re_min, re_max, im_min, im_max
            reps = dict(list(zip(free, [random_complex_number(a, b, c, d, rational=True)
                           for zi in free])))
            try:
                nmag = abs(self.evalf(2, subs=reps))
            except (ValueError, TypeError):
                # if an out of range value resulted in evalf problems
                # then return None -- XXX is there a way to know how to
                # select a good random number for a given expression?
                # e.g. when calculating n! negative values for n should not
                # be used
                return None
        else:
            reps = {}
            nmag = abs(self.evalf(2))

        if not hasattr(nmag, '_prec'):
            # e.g. exp_polar(2*I*pi) doesn't evaluate but is_number is True
            return None

        if nmag._prec == 1:
            # increase the precision up to the default maximum
            # precision to see if we can get any significance

            # evaluate
            for prec in giant_steps(2, DEFAULT_MAXPREC):
                nmag = abs(self.evalf(prec, subs=reps))
                if nmag._prec != 1:
                    break

        if nmag._prec != 1:
            if n is None:
                n = max(prec, 15)
            return self.evalf(n, subs=reps)

        # never got any significance
        return None

    def is_constant(self, *wrt, **flags):
        """Return True if self is constant, False if not, or None if
        the constancy could not be determined conclusively.

        Explanation
        ===========

        If an expression has no free symbols then it is a constant. If
        there are free symbols it is possible that the expression is a
        constant, perhaps (but not necessarily) zero. To test such
        expressions, a few strategies are tried:

        1) numerical evaluation at two random points. If two such evaluations
        give two different values and the values have a precision greater than
        1 then self is not constant. If the evaluations agree or could not be
        obtained with any precision, no decision is made. The numerical testing
        is done only if ``wrt`` is different than the free symbols.

        2) differentiation with respect to variables in 'wrt' (or all free
        symbols if omitted) to see if the expression is constant or not. This
        will not always lead to an expression that is zero even though an
        expression is constant (see added test in test_expr.py). If
        all derivatives are zero then self is constant with respect to the
        given symbols.

        3) finding out zeros of denominator expression with free_symbols.
        It will not be constant if there are zeros. It gives more negative
        answers for expression that are not constant.

        If neither evaluation nor differentiation can prove the expression is
        constant, None is returned unless two numerical values happened to be
        the same and the flag ``failing_number`` is True -- in that case the
        numerical value will be returned.

        If flag simplify=False is passed, self will not be simplified;
        the default is True since self should be simplified before testing.

        Examples
        ========

        >>> from sympy import cos, sin, Sum, S, pi
        >>> from sympy.abc import a, n, x, y
        >>> x.is_constant()
        False
        >>> S(2).is_constant()
        True
        >>> Sum(x, (x, 1, 10)).is_constant()
        True
        >>> Sum(x, (x, 1, n)).is_constant()
        False
        >>> Sum(x, (x, 1, n)).is_constant(y)
        True
        >>> Sum(x, (x, 1, n)).is_constant(n)
        False
        >>> Sum(x, (x, 1, n)).is_constant(x)
        True
        >>> eq = a*cos(x)**2 + a*sin(x)**2 - a
        >>> eq.is_constant()
        True
        >>> eq.subs({x: pi, a: 2}) == eq.subs({x: pi, a: 3}) == 0
        True

        >>> (0**x).is_constant()
        False
        >>> x.is_constant()
        False
        >>> (x**x).is_constant()
        False
        >>> one = cos(x)**2 + sin(x)**2
        >>> one.is_constant()
        True
        >>> ((one - 1)**(x + 1)).is_constant() in (True, False) # could be 0 or 1
        True
        """

        simplify = flags.get('simplify', True)

        if self.is_number:
            return True
        free = self.free_symbols
        if not free:
            return True  # assume f(1) is some constant

        # if we are only interested in some symbols and they are not in the
        # free symbols then this expression is constant wrt those symbols
        wrt = set(wrt)
        if wrt and not wrt & free:
            return True
        wrt = wrt or free

        # simplify unless this has already been done
        expr = self
        if simplify:
            expr = expr.simplify()

        # is_zero should be a quick assumptions check; it can be wrong for
        # numbers (see test_is_not_constant test), giving False when it
        # shouldn't, but hopefully it will never give True unless it is sure.
        if expr.is_zero:
            return True

        # Don't attempt substitution or differentiation with non-number symbols
        wrt_number = {sym for sym in wrt if sym.kind is NumberKind}

        # try numerical evaluation to see if we get two different values
        failing_number = None
        if wrt_number == free:
            # try 0 (for a) and 1 (for b)
            try:
                a = expr.subs(list(zip(free, [0]*len(free))),
                    simultaneous=True)
                if a is S.NaN:
                    # evaluation may succeed when substitution fails
                    a = expr._random(None, 0, 0, 0, 0)
            except ZeroDivisionError:
                a = None
            if a is not None and a is not S.NaN:
                try:
                    b = expr.subs(list(zip(free, [1]*len(free))),
                        simultaneous=True)
                    if b is S.NaN:
                        # evaluation may succeed when substitution fails
                        b = expr._random(None, 1, 0, 1, 0)
                except ZeroDivisionError:
                    b = None
                if b is not None and b is not S.NaN and b.equals(a) is False:
                    return False
                # try random real
                b = expr._random(None, -1, 0, 1, 0)
                if b is not None and b is not S.NaN and b.equals(a) is False:
                    return False
                # try random complex
                b = expr._random()
                if b is not None and b is not S.NaN:
                    if b.equals(a) is False:
                        return False
                    failing_number = a if a.is_number else b

        # now we will test each wrt symbol (or all free symbols) to see if the
        # expression depends on them or not using differentiation. This is
        # not sufficient for all expressions, however, so we don't return
        # False if we get a derivative other than 0 with free symbols.
        for w in wrt_number:
            deriv = expr.diff(w)
            if simplify:
                deriv = deriv.simplify()
            if deriv != 0:
                if not (pure_complex(deriv, or_real=True)):
                    if flags.get('failing_number', False):
                        return failing_number
                return False
        from sympy.solvers.solvers import denoms
        return fuzzy_not(fuzzy_or(den.is_zero for den in denoms(self)))

    def equals(self, other, failing_expression=False):
        """Return True if self == other, False if it does not, or None. If
        failing_expression is True then the expression which did not simplify
        to a 0 will be returned instead of None.

        Explanation
        ===========

        If ``self`` is a Number (or complex number) that is not zero, then
        the result is False.

        If ``self`` is a number and has not evaluated to zero, evalf will be
        used to test whether the expression evaluates to zero. If it does so
        and the result has significance (i.e. the precision is either -1, for
