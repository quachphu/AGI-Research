    >>> f = implemented_function('f', lambda x: x+1)
    >>> lam_f = lambdify(x, f(x))
    >>> lam_f(4)
    5
    """
    # Delayed import to avoid circular imports
    from sympy.core.function import UndefinedFunction
    # if name, create function to hold implementation
    kwargs = {}
    if isinstance(symfunc, UndefinedFunction):
        kwargs = symfunc._kwargs
        symfunc = symfunc.__name__
    if isinstance(symfunc, str):
        # Keyword arguments to UndefinedFunction are added as attributes to
        # the created class.
        symfunc = UndefinedFunction(
            symfunc, _imp_=staticmethod(implementation), **kwargs)
    elif not isinstance(symfunc, UndefinedFunction):
        raise ValueError(filldedent('''
            symfunc should be either a string or
            an UndefinedFunction instance.'''))
    return symfunc


def _too_large_for_docstring(expr, limit):
    """Decide whether an ``Expr`` is too large to be fully rendered in a
    ``lambdify`` docstring.

    This is a fast alternative to ``count_ops``, which can become prohibitively
    slow for large expressions, because in this instance we only care whether
    ``limit`` is exceeded rather than counting the exact number of nodes in the
    expression.

    Parameters
    ==========
    expr : ``Expr``, (nested) ``list`` of ``Expr``, or ``Matrix``
        The same objects that can be passed to the ``expr`` argument of
        ``lambdify``.
    limit : ``int`` or ``None``
        The threshold above which an expression contains too many nodes to be
        usefully rendered in the docstring. If ``None`` then there is no limit.

    Returns
    =======
    bool
        ``True`` if the number of nodes in the expression exceeds the limit,
        ``False`` otherwise.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import _too_large_for_docstring
    >>> expr = x
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    False
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    Does this split it?

    >>> expr = [x, y, z]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = [x, [y], z, [[x+y], [x*y*z, [x+y+z]]]]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = ((x + y + z)**5).expand()
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> from sympy import Matrix
    >>> expr = Matrix([[(x + y + z), ((x + y + z)**2).expand(),
    ...                 ((x + y + z)**3).expand(), ((x + y + z)**4).expand()]])
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 1000)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    """
    # Must be imported here to avoid a circular import error
    from sympy.core.traversal import postorder_traversal

    if limit is None:
        return False

    i = 0
    for _ in postorder_traversal(expr):
        i += 1
        if i > limit:
            return True
    return False
<|endoftext|>"""Simple tools for timing functions' execution, when IPython is not available. """


import timeit
import math


_scales = [1e0, 1e3, 1e6, 1e9]
_units = ['s', 'ms', '\N{GREEK SMALL LETTER MU}s', 'ns']


def timed(func, setup="pass", limit=None):
    """Adaptively measure execution time of a function. """
    timer = timeit.Timer(func, setup=setup)
    repeat, number = 3, 1

    for i in range(1, 10):
        if timer.timeit(number) >= 0.2:
            break
        elif limit is not None and number >= limit:
            break
        else:
            number *= 10

    time = min(timer.repeat(repeat, number)) / number

    if time > 0.0:
        order = min(-int(math.floor(math.log10(time)) // 3), 3)
    else:
        order = 3

    return (number, time, time*_scales[order], _units[order])


# Code for doing inline timings of recursive algorithms.

def __do_timings():
    import os
    res = os.getenv('SYMPY_TIMINGS', '')
    res = [x.strip() for x in res.split(',')]
    return set(res)

_do_timings = __do_timings()
_timestack = None


def _print_timestack(stack, level=1):
    print('-'*level, '%.2f %s%s' % (stack[2], stack[0], stack[3]))
    for s in stack[1]:
        _print_timestack(s, level + 1)


def timethis(name):
    def decorator(func):
        if name not in _do_timings:
            return func

        def wrapper(*args, **kwargs):
            from time import time
            global _timestack
            oldtimestack = _timestack
            _timestack = [func.func_name, [], 0, args]
            t1 = time()
            r = func(*args, **kwargs)
            t2 = time()
            _timestack[2] = t2 - t1
            if oldtimestack is not None:
                oldtimestack[1].append(_timestack)
                _timestack = oldtimestack
            else:
                _print_timestack(_timestack)
                _timestack = None
            return r
        return wrapper
    return decorator
<|endoftext|>"""
The objects in this module allow the usage of the MatchPy pattern matching
library on SymPy expressions.
"""
import re
from typing import List, Callable, NamedTuple, Any, Dict

from sympy.core.sympify import _sympify
from sympy.external import import_module
from sympy.functions import (log, sin, cos, tan, cot, csc, sec, erf, gamma, uppergamma)
from sympy.functions.elementary.hyperbolic import acosh, asinh, atanh, acoth, acsch, asech, cosh, sinh, tanh, coth, sech, csch
from sympy.functions.elementary.trigonometric import atan, acsc, asin, acot, acos, asec
from sympy.functions.special.error_functions import fresnelc, fresnels, erfc, erfi, Ei
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.integrals.integrals import Integral
from sympy.printing.repr import srepr
from sympy.utilities.decorator import doctest_depends_on


matchpy = import_module("matchpy")


__doctest_requires__ = {('*',): ['matchpy']}


if matchpy:
    from matchpy import Operation, CommutativeOperation, AssociativeOperation, OneIdentityOperation
    from matchpy.expressions.functions import op_iter, create_operation_expression, op_len

    Operation.register(Integral)
    Operation.register(Pow)
    OneIdentityOperation.register(Pow)

    Operation.register(Add)
    OneIdentityOperation.register(Add)
    CommutativeOperation.register(Add)
    AssociativeOperation.register(Add)

    Operation.register(Mul)
    OneIdentityOperation.register(Mul)
    CommutativeOperation.register(Mul)
    AssociativeOperation.register(Mul)

    Operation.register(Equality)
    CommutativeOperation.register(Equality)
    Operation.register(Unequality)
    CommutativeOperation.register(Unequality)

    Operation.register(exp)
    Operation.register(log)
    Operation.register(gamma)
    Operation.register(uppergamma)
    Operation.register(fresnels)
    Operation.register(fresnelc)
    Operation.register(erf)
    Operation.register(Ei)
    Operation.register(erfc)
    Operation.register(erfi)
    Operation.register(sin)
    Operation.register(cos)
    Operation.register(tan)
    Operation.register(cot)
    Operation.register(csc)
    Operation.register(sec)
    Operation.register(sinh)
    Operation.register(cosh)
    Operation.register(tanh)
    Operation.register(coth)
    Operation.register(csch)
    Operation.register(sech)
    Operation.register(asin)
    Operation.register(acos)
    Operation.register(atan)
    Operation.register(acot)
    Operation.register(acsc)
    Operation.register(asec)
    Operation.register(asinh)
    Operation.register(acosh)
    Operation.register(atanh)
    Operation.register(acoth)
    Operation.register(acsch)
    Operation.register(asech)

    @op_iter.register(Integral)  # type: ignore
    def _(operation):
        return iter((operation._args[0],) + operation._args[1])

    @op_iter.register(Basic)  # type: ignore
    def _(operation):
        return iter(operation._args)

    @op_len.register(Integral)  # type: ignore
    def _(operation):
        return 1 + len(operation._args[1])

    @op_len.register(Basic)  # type: ignore
    def _(operation):
        return len(operation._args)

    @create_operation_expression.register(Basic)
    def sympy_op_factory(old_operation, new_operands, variable_name=True):
        return type(old_operation)(*new_operands)


if matchpy:
    from matchpy import Wildcard
else:
    class Wildcard: # type: ignore
        def __init__(self, min_length, fixed_size, variable_name, optional):
            self.min_count = min_length
            self.fixed_size = fixed_size
            self.variable_name = variable_name
            self.optional = optional


@doctest_depends_on(modules=('matchpy',))
class _WildAbstract(Wildcard, Symbol):
    min_length: int  # abstract field required in subclasses
    fixed_size: bool  # abstract field required in subclasses

    def __init__(self, variable_name=None, optional=None, **assumptions):
        min_length = self.min_length
        fixed_size = self.fixed_size
        if optional is not None:
            optional = _sympify(optional)
        Wildcard.__init__(self, min_length, fixed_size, str(variable_name), optional)

    def __getstate__(self):
        return {
            "min_length": self.min_length,
            "fixed_size": self.fixed_size,
            "min_count": self.min_count,
            "variable_name": self.variable_name,
            "optional": self.optional,
        }

    def __new__(cls, variable_name=None, optional=None, **assumptions):
        cls._sanitize(assumptions, cls)
        return _WildAbstract.__xnew__(cls, variable_name, optional, **assumptions)

    def __getnewargs__(self):
        return self.variable_name, self.optional

    @staticmethod
    def __xnew__(cls, variable_name=None, optional=None, **assumptions):
        obj = Symbol.__xnew__(cls, variable_name, **assumptions)
        return obj

    def _hashable_content(self):
        if self.optional:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name, self.optional)
        else:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name)

    def __copy__(self) -> '_WildAbstract':
        return type(self)(variable_name=self.variable_name, optional=self.optional)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name


@doctest_depends_on(modules=('matchpy',))
class WildDot(_WildAbstract):
    min_length = 1
    fixed_size = True


@doctest_depends_on(modules=('matchpy',))
class WildPlus(_WildAbstract):
    min_length = 1
    fixed_size = False


@doctest_depends_on(modules=('matchpy',))
class WildStar(_WildAbstract):
    min_length = 0
    fixed_size = False


def _get_srepr(expr):
    s = srepr(expr)
    s = re.sub(r"WildDot\('(\w+)'\)", r"\1", s)
    s = re.sub(r"WildPlus\('(\w+)'\)", r"*\1", s)
    s = re.sub(r"WildStar\('(\w+)'\)", r"*\1", s)
    return s


class ReplacementInfo(NamedTuple):
    replacement: Any
    info: Any


@doctest_depends_on(modules=('matchpy',))
class Replacer:
    """
    Replacer object to perform multiple pattern matching and subexpression
    replacements in SymPy expressions.

    Examples
    ========

    Example to construct a simple first degree equation solver:

    >>> from sympy.utilities.matchpy_connector import WildDot, Replacer
    >>> from sympy import Equality, Symbol
    >>> x = Symbol("x")
    >>> a_ = WildDot("a_", optional=1)
    >>> b_ = WildDot("b_", optional=0)

    The lines above have defined two wildcards, ``a_`` and ``b_``, the
    coefficients of the equation `a x + b = 0`. The optional values specified
    indicate which expression to return in case no match is found, they are
    necessary in equations like `a x = 0` and `x + b = 0`.

    Create two constraints to make sure that ``a_`` and ``b_`` will not match
    any expression containing ``x``:

    >>> from matchpy import CustomConstraint
    >>> free_x_a = CustomConstraint(lambda a_: not a_.has(x))
    >>> free_x_b = CustomConstraint(lambda b_: not b_.has(x))

    Now create the rule replacer with the constraints:

    >>> replacer = Replacer(common_constraints=[free_x_a, free_x_b])

    Add the matching rule:

    >>> replacer.add(Equality(a_*x + b_, 0), -b_/a_)

    Let's try it:

    >>> replacer.replace(Equality(3*x + 4, 0))
    -4/3

    Notice that it will not match equations expressed with other patterns:

    >>> eq = Equality(3*x, 4)
    >>> replacer.replace(eq)
    Eq(3*x, 4)

    In order to extend the matching patterns, define another one (we also need
    to clear the cache, because the previous result has already been memorized
    and the pattern matcher will not iterate again if given the same expression)

    >>> replacer.add(Equality(a_*x, b_), b_/a_)
    >>> replacer._matcher.clear()
    >>> replacer.replace(eq)
    4/3
    """

    def __init__(self, common_constraints: list = [], lambdify: bool = False, info: bool = False):
        self._matcher = matchpy.ManyToOneMatcher()
        self._common_constraint = common_constraints
        self._lambdify = lambdify
        self._info = info
        self._wildcards: Dict[str, Wildcard] = {}

    def _get_lambda(self, lambda_str: str) -> Callable[..., Expr]:
        exec("from sympy import *")
        return eval(lambda_str, locals())

    def _get_custom_constraint(self, constraint_expr: Expr, condition_template: str) -> Callable[..., Expr]:
        wilds = [x.name for x in constraint_expr.atoms(_WildAbstract)]
        lambdaargs = ', '.join(wilds)
        fullexpr = _get_srepr(constraint_expr)
        condition = condition_template.format(fullexpr)
        return matchpy.CustomConstraint(
            self._get_lambda(f"lambda {lambdaargs}: ({condition})"))

    def _get_custom_constraint_nonfalse(self, constraint_expr: Expr) -> Callable[..., Expr]:
        return self._get_custom_constraint(constraint_expr, "({}) != False")

    def _get_custom_constraint_true(self, constraint_expr: Expr) -> Callable[..., Expr]:
        return self._get_custom_constraint(constraint_expr, "({}) == True")

    def add(self, expr: Expr, replacement, conditions_true: List[Expr] = [],
            conditions_nonfalse: List[Expr] = [], info: Any = None) -> None:
        expr = _sympify(expr)
        replacement = _sympify(replacement)
        constraints = self._common_constraint[:]
        constraint_conditions_true = [
            self._get_custom_constraint_true(cond) for cond in conditions_true]
        constraint_conditions_nonfalse = [
            self._get_custom_constraint_nonfalse(cond) for cond in conditions_nonfalse]
        constraints.extend(constraint_conditions_true)
        constraints.extend(constraint_conditions_nonfalse)
        pattern = matchpy.Pattern(expr, *constraints)
        if self._lambdify:
            lambda_str = f"lambda {', '.join((x.name for x in expr.atoms(_WildAbstract)))}: {_get_srepr(replacement)}"
            lambda_expr = self._get_lambda(lambda_str)
            replacement = lambda_expr
        else:
            self._wildcards.update({str(i): i for i in expr.atoms(Wildcard)})
        if self._info:
            replacement = ReplacementInfo(replacement, info)
        self._matcher.add(pattern, replacement)

    def replace(self, expression, max_count: int = -1):
        # This method partly rewrites the .replace method of ManyToOneReplacer
        # in MatchPy.
        # License: https://github.com/HPAC/matchpy/blob/master/LICENSE
        infos = []
        replaced = True
        replace_count = 0
        while replaced and (max_count < 0 or replace_count < max_count):
            replaced = False
            for subexpr, pos in matchpy.preorder_iter_with_position(expression):
                try:
                    replacement_data, subst = next(iter(self._matcher.match(subexpr)))
                    if self._info:
                        replacement = replacement_data.replacement
                        infos.append(replacement_data.info)
                    else:
                        replacement = replacement_data

                    if self._lambdify:
                        result = replacement(**subst)
                    else:
                        result = replacement.xreplace({self._wildcards[k]: v for k, v in subst.items()})

                    expression = matchpy.functions.replace(expression, pos, result)
                    replaced = True
                    break
                except StopIteration:
                    pass
            replace_count += 1
        if self._info:
            return expression, infos
        else:
            return expression
<|endoftext|>from collections import Counter, defaultdict, OrderedDict
from itertools import (
    chain, combinations, combinations_with_replacement, cycle, islice,
    permutations, product, groupby
)
# For backwards compatibility
from itertools import product as cartes # noqa: F401
from operator import gt



# this is the logical location of these functions
from sympy.utilities.enumerative import (
    multiset_partitions_taocp, list_visitor, MultisetPartitionTraverser)

from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated


def is_palindromic(s, i=0, j=None):
    """
    Return True if the sequence is the same from left to right as it
    is from right to left in the whole sequence (default) or in the
    Python slice ``s[i: j]``; else False.

    Examples
    ========

    >>> from sympy.utilities.iterables import is_palindromic
    >>> is_palindromic([1, 0, 1])
    True
    >>> is_palindromic('abcbb')
    False
    >>> is_palindromic('abcbb', 1)
    False

    Normal Python slicing is performed in place so there is no need to
    create a slice of the sequence for testing:

    >>> is_palindromic('abcbb', 1, -1)
    True
    >>> is_palindromic('abcbb', -4, -1)
    True

    See Also
    ========

    sympy.ntheory.digits.is_palindromic: tests integers

    """
    i, j, _ = slice(i, j).indices(len(s))
    m = (j - i)//2
    # if length is odd, middle element will be ignored
    return all(s[i + k] == s[j - 1 - k] for k in range(m))


def flatten(iterable, levels=None, cls=None):  # noqa: F811
    """
    Recursively denest iterable containers.

    >>> from sympy import flatten

    >>> flatten([1, 2, 3])
    [1, 2, 3]
    >>> flatten([1, 2, [3]])
    [1, 2, 3]
    >>> flatten([1, [2, 3], [4, 5]])
    [1, 2, 3, 4, 5]
    >>> flatten([1.0, 2, (1, None)])
    [1.0, 2, 1, None]

    If you want to denest only a specified number of levels of
    nested containers, then set ``levels`` flag to the desired
    number of levels::

    >>> ls = [[(-2, -1), (1, 2)], [(0, 0)]]

    >>> flatten(ls, levels=1)
    [(-2, -1), (1, 2), (0, 0)]

    If cls argument is specified, it will only flatten instances of that
    class, for example:

    >>> from sympy import Basic, S
    >>> class MyOp(Basic):
    ...     pass
    ...
    >>> flatten([MyOp(S(1), MyOp(S(2), S(3)))], cls=MyOp)
    [1, 2, 3]

    adapted from https://kogs-www.informatik.uni-hamburg.de/~meine/python_tricks
    """
    from sympy.tensor.array import NDimArray
    if levels is not None:
        if not levels:
            return iterable
        elif levels > 0:
            levels -= 1
        else:
            raise ValueError(
                "expected non-negative number of levels, got %s" % levels)

    if cls is None:
        def reducible(x):
            return is_sequence(x, set)
    else:
        def reducible(x):
            return isinstance(x, cls)

    result = []

    for el in iterable:
        if reducible(el):
            if hasattr(el, 'args') and not isinstance(el, NDimArray):
                el = el.args
            result.extend(flatten(el, levels=levels, cls=cls))
        else:
            result.append(el)

    return result


def unflatten(iter, n=2):
    """Group ``iter`` into tuples of length ``n``. Raise an error if
    the length of ``iter`` is not a multiple of ``n``.
    """
    if n < 1 or len(iter) % n:
        raise ValueError('iter length is not a multiple of %i' % n)
    return list(zip(*(iter[i::n] for i in range(n))))


def reshape(seq, how):
    """Reshape the sequence according to the template in ``how``.

    Examples
    ========

    >>> from sympy.utilities import reshape
    >>> seq = list(range(1, 9))

    >>> reshape(seq, [4]) # lists of 4
    [[1, 2, 3, 4], [5, 6, 7, 8]]

    >>> reshape(seq, (4,)) # tuples of 4
    [(1, 2, 3, 4), (5, 6, 7, 8)]

    >>> reshape(seq, (2, 2)) # tuples of 4
    [(1, 2, 3, 4), (5, 6, 7, 8)]

    >>> reshape(seq, (2, [2])) # (i, i, [i, i])
    [(1, 2, [3, 4]), (5, 6, [7, 8])]

    >>> reshape(seq, ((2,), [2])) # etc....
    [((1, 2), [3, 4]), ((5, 6), [7, 8])]

    >>> reshape(seq, (1, [2], 1))
    [(1, [2, 3], 4), (5, [6, 7], 8)]

    >>> reshape(tuple(seq), ([[1], 1, (2,)],))
    (([[1], 2, (3, 4)],), ([[5], 6, (7, 8)],))

    >>> reshape(tuple(seq), ([1], 1, (2,)))
    (([1], 2, (3, 4)), ([5], 6, (7, 8)))

    >>> reshape(list(range(12)), [2, [3], {2}, (1, (3,), 1)])
    [[0, 1, [2, 3, 4], {5, 6}, (7, (8, 9, 10), 11)]]

    """
    m = sum(flatten(how))
    n, rem = divmod(len(seq), m)
    if m < 0 or rem:
        raise ValueError('template must sum to positive number '
        'that divides the length of the sequence')
    i = 0
    container = type(how)
    rv = [None]*n
    for k in range(len(rv)):
        _rv = []
        for hi in how:
            if isinstance(hi, int):
                _rv.extend(seq[i: i + hi])
                i += hi
            else:
                n = sum(flatten(hi))
                hi_type = type(hi)
                _rv.append(hi_type(reshape(seq[i: i + n], hi)[0]))
                i += n
        rv[k] = container(_rv)
    return type(seq)(rv)


def group(seq, multiple=True):
    """
    Splits a sequence into a list of lists of equal, adjacent elements.

    Examples
    ========

    >>> from sympy import group

    >>> group([1, 1, 1, 2, 2, 3])
    [[1, 1, 1], [2, 2], [3]]
    >>> group([1, 1, 1, 2, 2, 3], multiple=False)
    [(1, 3), (2, 2), (3, 1)]
    >>> group([1, 1, 3, 2, 2, 1], multiple=False)
    [(1, 2), (3, 1), (2, 2), (1, 1)]

    See Also
    ========

    multiset

    """
    if multiple:
        return [(list(g)) for _, g in groupby(seq)]
    return [(k, len(list(g))) for k, g in groupby(seq)]


def _iproduct2(iterable1, iterable2):
    '''Cartesian product of two possibly infinite iterables'''

    it1 = iter(iterable1)
    it2 = iter(iterable2)

    elems1 = []
    elems2 = []

    sentinel = object()
    def append(it, elems):
        e = next(it, sentinel)
        if e is not sentinel:
            elems.append(e)

    n = 0
    append(it1, elems1)
    append(it2, elems2)

    while n <= len(elems1) + len(elems2):
        for m in range(n-len(elems1)+1, len(elems2)):
            yield (elems1[n-m], elems2[m])
        n += 1
        append(it1, elems1)
        append(it2, elems2)


def iproduct(*iterables):
    '''
    Cartesian product of iterables.

    Generator of the Cartesian product of iterables. This is analogous to
    itertools.product except that it works with infinite iterables and will
    yield any item from the infinite product eventually.

    Examples
    ========

    >>> from sympy.utilities.iterables import iproduct
    >>> sorted(iproduct([1,2], [3,4]))
    [(1, 3), (1, 4), (2, 3), (2, 4)]

    With an infinite iterator:

    >>> from sympy import S
    >>> (3,) in iproduct(S.Integers)
    True
    >>> (3, 4) in iproduct(S.Integers, S.Integers)
    True

    .. seealso::

       `itertools.product
       <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    '''
    if len(iterables) == 0:
        yield ()
        return
    elif len(iterables) == 1:
        for e in iterables[0]:
            yield (e,)
    elif len(iterables) == 2:
        yield from _iproduct2(*iterables)
    else:
        first, others = iterables[0], iterables[1:]
        for ef, eo in _iproduct2(first, iproduct(*others)):
            yield (ef,) + eo


def multiset(seq):
    """Return the hashable sequence in multiset form with values being the
    multiplicity of the item in the sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset
    >>> multiset('mississippi')
    {'i': 4, 'm': 1, 'p': 2, 's': 4}

    See Also
    ========

    group

    """
    return dict(Counter(seq).items())




def ibin(n, bits=None, str=False):
    """Return a list of length ``bits`` corresponding to the binary value
    of ``n`` with small bits to the right (last). If bits is omitted, the
    length will be the number required to represent ``n``. If the bits are
    desired in reversed order, use the ``[::-1]`` slice of the returned list.

    If a sequence of all bits-length lists starting from ``[0, 0,..., 0]``
    through ``[1, 1, ..., 1]`` are desired, pass a non-integer for bits, e.g.
    ``'all'``.

    If the bit *string* is desired pass ``str=True``.

    Examples
    ========

    >>> from sympy.utilities.iterables import ibin
    >>> ibin(2)
    [1, 0]
    >>> ibin(2, 4)
    [0, 0, 1, 0]

    If all lists corresponding to 0 to 2**n - 1, pass a non-integer
    for bits:

    >>> bits = 2
    >>> for i in ibin(2, 'all'):
    ...     print(i)
    (0, 0)
    (0, 1)
    (1, 0)
    (1, 1)

    If a bit string is desired of a given length, use str=True:

    >>> n = 123
    >>> bits = 10
    >>> ibin(n, bits, str=True)
    '0001111011'
    >>> ibin(n, bits, str=True)[::-1]  # small bits left
    '1101111000'
    >>> list(ibin(3, 'all', str=True))
    ['000', '001', '010', '011', '100', '101', '110', '111']

    """
    if n < 0:
        raise ValueError("negative numbers are not allowed")
    n = as_int(n)

    if bits is None:
        bits = 0
    else:
        try:
            bits = as_int(bits)
        except ValueError:
            bits = -1
        else:
            if n.bit_length() > bits:
                raise ValueError(
                    "`bits` must be >= {}".format(n.bit_length()))

    if not str:
        if bits >= 0:
            return [1 if i == "1" else 0 for i in bin(n)[2:].rjust(bits, "0")]
        else:
            return variations(range(2), n, repetition=True)
    else:
        if bits >= 0:
            return bin(n)[2:].rjust(bits, "0")
        else:
            return (bin(i)[2:].rjust(n, "0") for i in range(2**n))


def variations(seq, n, repetition=False):
    r"""Returns an iterator over the n-sized variations of ``seq`` (size N).
    ``repetition`` controls whether items in ``seq`` can appear more than once;

    Examples
    ========

    ``variations(seq, n)`` will return `\frac{N!}{(N - n)!}` permutations without
    repetition of ``seq``'s elements:

        >>> from sympy import variations
        >>> list(variations([1, 2], 2))
        [(1, 2), (2, 1)]

    ``variations(seq, n, True)`` will return the `N^n` permutations obtained
    by allowing repetition of elements:

        >>> list(variations([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 1), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(variations([0, 1], 3, repetition=False))
        []
        >>> list(variations([0, 1], 3, repetition=True))[:4]
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]

    .. seealso::

       `itertools.permutations
       <https://docs.python.org/3/library/itertools.html#itertools.permutations>`_,
       `itertools.product
       <https://docs.python.org/3/library/itertools.html#itertools.product>`_
    """
    if not repetition:
        seq = tuple(seq)
        if len(seq) < n:
            return iter(())  # 0 length iterator
        return permutations(seq, n)
    else:
        if n == 0:
            return iter(((),))  # yields 1 empty tuple
        else:
            return product(seq, repeat=n)


def subsets(seq, k=None, repetition=False):
    r"""Generates all `k`-subsets (combinations) from an `n`-element set, ``seq``.

    A `k`-subset of an `n`-element set is any subset of length exactly `k`. The
    number of `k`-subsets of an `n`-element set is given by ``binomial(n, k)``,
    whereas there are `2^n` subsets all together. If `k` is ``None`` then all
    `2^n` subsets will be returned from shortest to longest.

    Examples
    ========

    >>> from sympy import subsets

    ``subsets(seq, k)`` will return the
    `\frac{n!}{k!(n - k)!}` `k`-subsets (combinations)
    without repetition, i.e. once an item has been removed, it can no
    longer be "taken":

        >>> list(subsets([1, 2], 2))
        [(1, 2)]
        >>> list(subsets([1, 2]))
        [(), (1,), (2,), (1, 2)]
        >>> list(subsets([1, 2, 3], 2))
        [(1, 2), (1, 3), (2, 3)]


    ``subsets(seq, k, repetition=True)`` will return the
    `\frac{(n - 1 + k)!}{k!(n - 1)!}`
    combinations *with* repetition:

        >>> list(subsets([1, 2], 2, repetition=True))
        [(1, 1), (1, 2), (2, 2)]

    If you ask for more items than are in the set you get the empty set unless
    you allow repetitions:

        >>> list(subsets([0, 1], 3, repetition=False))
        []
        >>> list(subsets([0, 1], 3, repetition=True))
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]

    """
    if k is None:
        if not repetition:
            return chain.from_iterable((combinations(seq, k)
                                        for k in range(len(seq) + 1)))
        else:
            return chain.from_iterable((combinations_with_replacement(seq, k)
                                        for k in range(len(seq) + 1)))
    else:
        if not repetition:
            return combinations(seq, k)
        else:
            return combinations_with_replacement(seq, k)


def filter_symbols(iterator, exclude):
    """
    Only yield elements from `iterator` that do not occur in `exclude`.

    Parameters
    ==========

    iterator : iterable
        iterator to take elements from

    exclude : iterable
        elements to exclude

    Returns
    =======

    iterator : iterator
        filtered iterator
    """
    exclude = set(exclude)
    for s in iterator:
        if s not in exclude:
            yield s

def numbered_symbols(prefix='x', cls=None, start=0, exclude=(), *args, **assumptions):
    """
    Generate an infinite stream of Symbols consisting of a prefix and
    increasing subscripts provided that they do not occur in ``exclude``.

    Parameters
    ==========

    prefix : str, optional
        The prefix to use. By default, this function will generate symbols of
        the form "x0", "x1", etc.

    cls : class, optional
        The class to use. By default, it uses ``Symbol``, but you can also use ``Wild``
        or ``Dummy``.

    start : int, optional
        The start number.  By default, it is 0.

    exclude : list, tuple, set of cls, optional
        Symbols to be excluded.

    *args, **kwargs
        Additional positional and keyword arguments are passed to the *cls* class.

    Returns
    =======

    sym : Symbol
        The subscripted symbols.
    """
    exclude = set(exclude or [])
    if cls is None:
        # We can't just make the default cls=Symbol because it isn't
        # imported yet.
        from sympy.core import Symbol
        cls = Symbol

    while True:
        name = '%s%s' % (prefix, start)
        s = cls(name, *args, **assumptions)
        if s not in exclude:
            yield s
        start += 1


def capture(func):
    """Return the printed output of func().

    ``func`` should be a function without arguments that produces output with
    print statements.

    >>> from sympy.utilities.iterables import capture
    >>> from sympy import pprint
    >>> from sympy.abc import x
    >>> def foo():
    ...     print('hello world!')
    ...
    >>> 'hello' in capture(foo) # foo, not foo()
    True
    >>> capture(lambda: pprint(2/x))
    '2\\n-\\nx\\n'

    """
    from io import StringIO
    import sys

    stdout = sys.stdout
    sys.stdout = file = StringIO()
    try:
        func()
    finally:
        sys.stdout = stdout
    return file.getvalue()


def sift(seq, keyfunc, binary=False):
    """
    Sift the sequence, ``seq`` according to ``keyfunc``.

    Returns
    =======

    When ``binary`` is ``False`` (default), the output is a dictionary
    where elements of ``seq`` are stored in a list keyed to the value
    of keyfunc for that element. If ``binary`` is True then a tuple
    with lists ``T`` and ``F`` are returned where ``T`` is a list
    containing elements of seq for which ``keyfunc`` was ``True`` and
    ``F`` containing those elements for which ``keyfunc`` was ``False``;
    a ValueError is raised if the ``keyfunc`` is not binary.

    Examples
    ========

    >>> from sympy.utilities import sift
    >>> from sympy.abc import x, y
    >>> from sympy import sqrt, exp, pi, Tuple

    >>> sift(range(5), lambda x: x % 2)
    {0: [0, 2, 4], 1: [1, 3]}

    sift() returns a defaultdict() object, so any key that has no matches will
    give [].

    >>> sift([x], lambda x: x.is_commutative)
    {True: [x]}
    >>> _[False]
    []

    Sometimes you will not know how many keys you will get:

    >>> sift([sqrt(x), exp(x), (y**x)**2],
    ...      lambda x: x.as_base_exp()[0])
    {E: [exp(x)], x: [sqrt(x)], y: [y**(2*x)]}

    Sometimes you expect the results to be binary; the
    results can be unpacked by setting ``binary`` to True:

    >>> sift(range(4), lambda x: x % 2, binary=True)
    ([1, 3], [0, 2])
    >>> sift(Tuple(1, pi), lambda x: x.is_rational, binary=True)
    ([1], [pi])

    A ValueError is raised if the predicate was not actually binary
    (which is a good test for the logic where sifting is used and
    binary results were expected):

    >>> unknown = exp(1) - pi  # the rationality of this is unknown
    >>> args = Tuple(1, pi, unknown)
    >>> sift(args, lambda x: x.is_rational, binary=True)
    Traceback (most recent call last):
    ...
    ValueError: keyfunc gave non-binary output

    The non-binary sifting shows that there were 3 keys generated:

    >>> set(sift(args, lambda x: x.is_rational).keys())
    {None, False, True}

    If you need to sort the sifted items it might be better to use
    ``ordered`` which can economically apply multiple sort keys
    to a sequence while sorting.

    See Also
    ========

    ordered

    """
    if not binary:
        m = defaultdict(list)
        for i in seq:
            m[keyfunc(i)].append(i)
        return m
    sift = F, T = [], []
    for i in seq:
        try:
            sift[keyfunc(i)].append(i)
        except (IndexError, TypeError):
            raise ValueError('keyfunc gave non-binary output')
    return T, F


def take(iter, n):
    """Return ``n`` items from ``iter`` iterator. """
    return [ value for _, value in zip(range(n), iter) ]


def dict_merge(*dicts):
    """Merge dictionaries into a single dictionary. """
    merged = {}

    for dict in dicts:
        merged.update(dict)

    return merged


def common_prefix(*seqs):
    """Return the subsequence that is a common start of sequences in ``seqs``.

    >>> from sympy.utilities.iterables import common_prefix
    >>> common_prefix(list(range(3)))
    [0, 1, 2]
    >>> common_prefix(list(range(3)), list(range(4)))
    [0, 1, 2]
    >>> common_prefix([1, 2, 3], [1, 2, 5])
    [1, 2]
    >>> common_prefix([1, 2, 3], [1, 3, 5])
    [1]
    """
    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    i = 0
    for i in range(min(len(s) for s in seqs)):
        if not all(seqs[j][i] == seqs[0][i] for j in range(len(seqs))):
            break
    else:
        i += 1
    return seqs[0][:i]


def common_suffix(*seqs):
    """Return the subsequence that is a common ending of sequences in ``seqs``.

    >>> from sympy.utilities.iterables import common_suffix
    >>> common_suffix(list(range(3)))
    [0, 1, 2]
    >>> common_suffix(list(range(3)), list(range(4)))
    []
    >>> common_suffix([1, 2, 3], [9, 2, 3])
    [2, 3]
    >>> common_suffix([1, 2, 3], [9, 7, 3])
    [3]
    """

    if not all(seqs):
        return []
    elif len(seqs) == 1:
        return seqs[0]
    i = 0
    for i in range(-1, -min(len(s) for s in seqs) - 1, -1):
        if not all(seqs[j][i] == seqs[0][i] for j in range(len(seqs))):
            break
    else:
        i -= 1
    if i == -1:
        return []
    else:
        return seqs[0][i + 1:]


def prefixes(seq):
    """
    Generate all prefixes of a sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import prefixes

    >>> list(prefixes([1,2,3,4]))
    [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]

    """
    n = len(seq)

    for i in range(n):
        yield seq[:i + 1]


def postfixes(seq):
    """
    Generate all postfixes of a sequence.

    Examples
    ========

    >>> from sympy.utilities.iterables import postfixes

    >>> list(postfixes([1,2,3,4]))
    [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]]

    """
    n = len(seq)

    for i in range(n):
        yield seq[n - i - 1:]


def topological_sort(graph, key=None):
    r"""
    Topological sort of graph's vertices.

    Parameters
    ==========

    graph : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph to be sorted topologically.

    key : callable[T] (optional)
        Ordering key for vertices on the same level. By default the natural
        (e.g. lexicographic) ordering is used (in this case the base type
        must implement ordering relations).

    Examples
    ========

    Consider a graph::

        +---+     +---+     +---+
        | 7 |\    | 5 |     | 3 |
        +---+ \   +---+     +---+
          |   _\___/ ____   _/ |
          |  /  \___/    \ /   |
          V  V           V V   |
         +----+         +---+  |
         | 11 |         | 8 |  |
         +----+         +---+  |
          | | \____   ___/ _   |
          | \      \ /    / \  |
          V  \     V V   /  V  V
        +---+ \   +---+ |  +----+
        | 2 |  |  | 9 | |  | 10 |
        +---+  |  +---+ |  +----+
               \________/

    where vertices are integers. This graph can be encoded using
    elementary Python's data structures as follows::

        >>> V = [2, 3, 5, 7, 8, 9, 10, 11]
        >>> E = [(7, 11), (7, 8), (5, 11), (3, 8), (3, 10),
        ...      (11, 2), (11, 9), (11, 10), (8, 9)]

    To compute a topological sort for graph ``(V, E)`` issue::

        >>> from sympy.utilities.iterables import topological_sort

        >>> topological_sort((V, E))
        [3, 5, 7, 8, 11, 2, 9, 10]

    If specific tie breaking approach is needed, use ``key`` parameter::

        >>> topological_sort((V, E), key=lambda v: -v)
        [7, 5, 11, 3, 10, 8, 9, 2]

    Only acyclic graphs can be sorted. If the input graph has a cycle,
    then ``ValueError`` will be raised::

        >>> topological_sort((V, E + [(10, 7)]))
        Traceback (most recent call last):
        ...
        ValueError: cycle detected

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Topological_sorting

    """
    V, E = graph

    L = []
    S = set(V)
    E = list(E)

    S.difference_update(u for v, u in E)

    if key is None:
        def key(value):
            return value

    S = sorted(S, key=key, reverse=True)

    while S:
        node = S.pop()
        L.append(node)

        for u, v in list(E):
            if u == node:
                E.remove((u, v))

                for _u, _v in E:
                    if v == _v:
                        break
                else:
                    kv = key(v)

                    for i, s in enumerate(S):
                        ks = key(s)

                        if kv > ks:
                            S.insert(i, v)
                            break
                    else:
                        S.append(v)

    if E:
        raise ValueError("cycle detected")
    else:
        return L


def strongly_connected_components(G):
    r"""
    Strongly connected components of a directed graph in reverse topological
    order.


    Parameters
    ==========

    G : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph whose strongly connected components are to be found.


    Examples
    ========

    Consider a directed graph (in dot notation)::

        digraph {
            A -> B
            A -> C
            B -> C
            C -> B
            B -> D
        }

    .. graphviz::

        digraph {
            A -> B
            A -> C
            B -> C
            C -> B
            B -> D
        }

    where vertices are the letters A, B, C and D. This graph can be encoded
    using Python's elementary data structures as follows::

        >>> V = ['A', 'B', 'C', 'D']
        >>> E = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B'), ('B', 'D')]

    The strongly connected components of this graph can be computed as

        >>> from sympy.utilities.iterables import strongly_connected_components

        >>> strongly_connected_components((V, E))
        [['D'], ['B', 'C'], ['A']]

    This also gives the components in reverse topological order.

    Since the subgraph containing B and C has a cycle they must be together in
    a strongly connected component. A and D are connected to the rest of the
    graph but not in a cyclic manner so they appear as their own strongly
    connected components.


    Notes
    =====

    The vertices of the graph must be hashable for the data structures used.
    If the vertices are unhashable replace them with integer indices.

    This function uses Tarjan's algorithm to compute the strongly connected
    components in `O(|V|+|E|)` (linear) time.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Strongly_connected_component
    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm


    See Also
    ========

    sympy.utilities.iterables.connected_components

    """
    # Map from a vertex to its neighbours
    V, E = G
    Gmap = {vi: [] for vi in V}
    for v1, v2 in E:
        Gmap[v1].append(v2)
    return _strongly_connected_components(V, Gmap)


def _strongly_connected_components(V, Gmap):
    """More efficient internal routine for strongly_connected_components"""
    #
    # Here V is an iterable of vertices and Gmap is a dict mapping each vertex
    # to a list of neighbours e.g.:
    #
    #   V = [0, 1, 2, 3]
    #   Gmap = {0: [2, 3], 1: [0]}
    #
    # For a large graph these data structures can often be created more
    # efficiently then those expected by strongly_connected_components() which
    # in this case would be
    #
    #   V = [0, 1, 2, 3]
    #   Gmap = [(0, 2), (0, 3), (1, 0)]
    #
    # XXX: Maybe this should be the recommended function to use instead...
    #

    # Non-recursive Tarjan's algorithm:
    lowlink = {}
    indices = {}
    stack = OrderedDict()
    callstack = []
    components = []
    nomore = object()

    def start(v):
        index = len(stack)
        indices[v] = lowlink[v] = index
        stack[v] = None
        callstack.append((v, iter(Gmap[v])))

    def finish(v1):
        # Finished a component?
        if lowlink[v1] == indices[v1]:
            component = [stack.popitem()[0]]
            while component[-1] is not v1:
                component.append(stack.popitem()[0])
            components.append(component[::-1])
        v2, _ = callstack.pop()
        if callstack:
            v1, _ = callstack[-1]
            lowlink[v1] = min(lowlink[v1], lowlink[v2])

    for v in V:
        if v in indices:
            continue
        start(v)
        while callstack:
            v1, it1 = callstack[-1]
            v2 = next(it1, nomore)
            # Finished children of v1?
            if v2 is nomore:
                finish(v1)
            # Recurse on v2
            elif v2 not in indices:
                start(v2)
            elif v2 in stack:
                lowlink[v1] = min(lowlink[v1], indices[v2])

    # Reverse topological sort order:
    return components


def connected_components(G):
    r"""
    Connected components of an undirected graph or weakly connected components
    of a directed graph.


    Parameters
    ==========

    G : tuple[list, list[tuple[T, T]]
        A tuple consisting of a list of vertices and a list of edges of
        a graph whose connected components are to be found.


    Examples
    ========


    Given an undirected graph::

        graph {
            A -- B
            C -- D
        }

    .. graphviz::

        graph {
            A -- B
            C -- D
        }

    We can find the connected components using this function if we include
    each edge in both directions::

        >>> from sympy.utilities.iterables import connected_components

        >>> V = ['A', 'B', 'C', 'D']
        >>> E = [('A', 'B'), ('B', 'A'), ('C', 'D'), ('D', 'C')]
        >>> connected_components((V, E))
        [['A', 'B'], ['C', 'D']]

    The weakly connected components of a directed graph can found the same
    way.


    Notes
    =====

    The vertices of the graph must be hashable for the data structures used.
    If the vertices are unhashable replace them with integer indices.

    This function uses Tarjan's algorithm to compute the connected components
    in `O(|V|+|E|)` (linear) time.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Component_%28graph_theory%29
    .. [2] https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm


    See Also
    ========

    sympy.utilities.iterables.strongly_connected_components

    """
    # Duplicate edges both ways so that the graph is effectively undirected
    # and return the strongly connected components:
    V, E = G
    E_undirected = []
    for v1, v2 in E:
        E_undirected.extend([(v1, v2), (v2, v1)])
    return strongly_connected_components((V, E_undirected))


def rotate_left(x, y):
    """
    Left rotates a list x by the number of steps specified
    in y.

    Examples
    ========

    >>> from sympy.utilities.iterables import rotate_left
    >>> a = [0, 1, 2]
    >>> rotate_left(a, 1)
    [1, 2, 0]
    """
    if len(x) == 0:
        return []
    y = y % len(x)
    return x[y:] + x[:y]


def rotate_right(x, y):
    """
    Right rotates a list x by the number of steps specified
    in y.

    Examples
    ========

    >>> from sympy.utilities.iterables import rotate_right
    >>> a = [0, 1, 2]
    >>> rotate_right(a, 1)
    [2, 0, 1]
    """
    if len(x) == 0:
        return []
    y = len(x) - y % len(x)
    return x[y:] + x[:y]


def least_rotation(x, key=None):
    '''
    Returns the number of steps of left rotation required to
    obtain lexicographically minimal string/list/tuple, etc.

    Examples
    ========

    >>> from sympy.utilities.iterables import least_rotation, rotate_left
    >>> a = [3, 1, 5, 1, 2]
    >>> least_rotation(a)
    3
    >>> rotate_left(a, _)
    [1, 2, 3, 1, 5]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lexicographically_minimal_string_rotation

    '''
    from sympy.functions.elementary.miscellaneous import Id
    if key is None: key = Id
    S = x + x      # Concatenate string to it self to avoid modular arithmetic
    f = [-1] * len(S)     # Failure function
    k = 0       # Least rotation of string found so far
    for j in range(1,len(S)):
        sj = S[j]
        i = f[j-k-1]
        while i != -1 and sj != S[k+i+1]:
            if key(sj) < key(S[k+i+1]):
                k = j-i-1
            i = f[i]
        if sj != S[k+i+1]:
            if key(sj) < key(S[k]):
                k = j
            f[j-k] = -1
        else:
            f[j-k] = i+1
    return k


def multiset_combinations(m, n, g=None):
    """
    Return the unique combinations of size ``n`` from multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_combinations
    >>> from itertools import combinations
    >>> [''.join(i) for i in  multiset_combinations('baby', 3)]
    ['abb', 'aby', 'bby']

    >>> def count(f, s): return len(list(f(s, 3)))

    The number of combinations depends on the number of letters; the
    number of unique combinations depends on how the letters are
    repeated.

    >>> s1 = 'abracadabra'
    >>> s2 = 'banana tree'
    >>> count(combinations, s1), count(multiset_combinations, s1)
    (165, 23)
    >>> count(combinations, s2), count(multiset_combinations, s2)
    (165, 54)

    """
    from sympy.core.sorting import ordered
    if g is None:
        if isinstance(m, dict):
            if any(as_int(v) < 0 for v in m.values()):
                raise ValueError('counts cannot be negative')
            N = sum(m.values())
            if n > N:
                return
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(m)
            N = len(m)
            if n > N:
                return
            try:
                m = multiset(m)
                g = [(k, m[k]) for k in ordered(m)]
            except TypeError:
                m = list(ordered(m))
                g = [list(i) for i in group(m, multiple=False)]
        del m
    else:
        # not checking counts since g is intended for internal use
        N = sum(v for k, v in g)
    if n > N or not n:
        yield []
    else:
        for i, (k, v) in enumerate(g):
            if v >= n:
                yield [k]*n
                v = n - 1
            for v in range(min(n, v), 0, -1):
                for j in multiset_combinations(None, n - v, g[i + 1:]):
                    rv = [k]*v + j
                    if len(rv) == n:
                        yield rv

def multiset_permutations(m, size=None, g=None):
    """
    Return the unique permutations of multiset ``m``.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_permutations
    >>> from sympy import factorial
    >>> [''.join(i) for i in multiset_permutations('aab')]
    ['aab', 'aba', 'baa']
    >>> factorial(len('banana'))
    720
    >>> len(list(multiset_permutations('banana')))
    60
    """
    from sympy.core.sorting import ordered
    if g is None:
        if isinstance(m, dict):
            if any(as_int(v) < 0 for v in m.values()):
                raise ValueError('counts cannot be negative')
            g = [[k, m[k]] for k in ordered(m)]
        else:
            m = list(ordered(m))
            g = [list(i) for i in group(m, multiple=False)]
        del m
    do = [gi for gi in g if gi[1] > 0]
    SUM = sum(gi[1] for gi in do)
    if not do or size is not None and (size > SUM or size < 1):
        if not do and size is None or size == 0:
            yield []
        return
    elif size == 1:
        for k, v in do:
            yield [k]
    elif len(do) == 1:
        k, v = do[0]
        v = v if size is None else (size if size <= v else 0)
        yield [k for i in range(v)]
    elif all(v == 1 for k, v in do):
        for p in permutations([k for k, v in do], size):
            yield list(p)
    else:
        size = size if size is not None else SUM
        for i, (k, v) in enumerate(do):
            do[i][1] -= 1
            for j in multiset_permutations(None, size - 1, do):
                if j:
                    yield [k] + j
            do[i][1] += 1


def _partition(seq, vector, m=None):
    """
    Return the partition of seq as specified by the partition vector.

    Examples
    ========

    >>> from sympy.utilities.iterables import _partition
    >>> _partition('abcde', [1, 0, 1, 2, 0])
    [['b', 'e'], ['a', 'c'], ['d']]

    Specifying the number of bins in the partition is optional:

    >>> _partition('abcde', [1, 0, 1, 2, 0], 3)
    [['b', 'e'], ['a', 'c'], ['d']]

    The output of _set_partitions can be passed as follows:

    >>> output = (3, [1, 0, 1, 2, 0])
    >>> _partition('abcde', *output)
    [['b', 'e'], ['a', 'c'], ['d']]

    See Also
    ========

    combinatorics.partitions.Partition.from_rgs

    """
    if m is None:
        m = max(vector) + 1
    elif isinstance(vector, int):  # entered as m, vector
        vector, m = m, vector
    p = [[] for i in range(m)]
    for i, v in enumerate(vector):
        p[v].append(seq[i])
    return p


def _set_partitions(n):
    """Cycle through all partitions of n elements, yielding the
    current number of partitions, ``m``, and a mutable list, ``q``
    such that ``element[i]`` is in part ``q[i]`` of the partition.

    NOTE: ``q`` is modified in place and generally should not be changed
    between function calls.

    Examples
    ========

    >>> from sympy.utilities.iterables import _set_partitions, _partition
    >>> for m, q in _set_partitions(3):
    ...     print('%s %s %s' % (m, q, _partition('abc', q, m)))
    1 [0, 0, 0] [['a', 'b', 'c']]
    2 [0, 0, 1] [['a', 'b'], ['c']]
    2 [0, 1, 0] [['a', 'c'], ['b']]
    2 [0, 1, 1] [['a'], ['b', 'c']]
    3 [0, 1, 2] [['a'], ['b'], ['c']]

    Notes
    =====

    This algorithm is similar to, and solves the same problem as,
    Algorithm 7.2.1.5H, from volume 4A of Knuth's The Art of Computer
    Programming.  Knuth uses the term "restricted growth string" where
    this code refers to a "partition vector". In each case, the meaning is
    the same: the value in the ith element of the vector specifies to
    which part the ith set element is to be assigned.

    At the lowest level, this code implements an n-digit big-endian
    counter (stored in the array q) which is incremented (with carries) to
    get the next partition in the sequence.  A special twist is that a
    digit is constrained to be at most one greater than the maximum of all
    the digits to the left of it.  The array p maintains this maximum, so
    that the code can efficiently decide when a digit can be incremented
    in place or whether it needs to be reset to 0 and trigger a carry to
    the next digit.  The enumeration starts with all the digits 0 (which
    corresponds to all the set elements being assigned to the same 0th
    part), and ends with 0123...n, which corresponds to each set element
    being assigned to a different, singleton, part.

    This routine was rewritten to use 0-based lists while trying to
    preserve the beauty and efficiency of the original algorithm.

    References
    ==========

    .. [1] Nijenhuis, Albert and Wilf, Herbert. (1978) Combinatorial Algorithms,
        2nd Ed, p 91, algorithm "nexequ". Available online from
        https://www.math.upenn.edu/~wilf/website/CombAlgDownld.html (viewed
        November 17, 2012).

    """
    p = [0]*n
    q = [0]*n
    nc = 1
    yield nc, q
    while nc != n:
        m = n
        while 1:
            m -= 1
            i = q[m]
            if p[i] != 1:
                break
            q[m] = 0
        i += 1
        q[m] = i
        m += 1
        nc += m - n
        p[0] += n - m
        if i == nc:
            p[nc] = 0
            nc += 1
        p[i - 1] -= 1
        p[i] += 1
        yield nc, q


def multiset_partitions(multiset, m=None):
    """
    Return unique partitions of the given multiset (in list form).
    If ``m`` is None, all multisets will be returned, otherwise only
    partitions with ``m`` parts will be returned.

    If ``multiset`` is an integer, a range [0, 1, ..., multiset - 1]
    will be supplied.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_partitions
    >>> list(multiset_partitions([1, 2, 3, 4], 2))
    [[[1, 2, 3], [4]], [[1, 2, 4], [3]], [[1, 2], [3, 4]],
    [[1, 3, 4], [2]], [[1, 3], [2, 4]], [[1, 4], [2, 3]],
    [[1], [2, 3, 4]]]
    >>> list(multiset_partitions([1, 2, 3, 4], 1))
    [[[1, 2, 3, 4]]]

    Only unique partitions are returned and these will be returned in a
    canonical order regardless of the order of the input:

    >>> a = [1, 2, 2, 1]
    >>> ans = list(multiset_partitions(a, 2))
    >>> a.sort()
    >>> list(multiset_partitions(a, 2)) == ans
    True
    >>> a = range(3, 1, -1)
    >>> (list(multiset_partitions(a)) ==
    ...  list(multiset_partitions(sorted(a))))
    True

    If m is omitted then all partitions will be returned:

    >>> list(multiset_partitions([1, 1, 2]))
    [[[1, 1, 2]], [[1, 1], [2]], [[1, 2], [1]], [[1], [1], [2]]]
    >>> list(multiset_partitions([1]*3))
    [[[1, 1, 1]], [[1], [1, 1]], [[1], [1], [1]]]

    Counting
    ========

    The number of partitions of a set is given by the bell number:

    >>> from sympy import bell
    >>> len(list(multiset_partitions(5))) == bell(5) == 52
    True

    The number of partitions of length k from a set of size n is given by the
    Stirling Number of the 2nd kind:

    >>> from sympy.functions.combinatorial.numbers import stirling
    >>> stirling(5, 2) == len(list(multiset_partitions(5, 2))) == 15
    True

    These comments on counting apply to *sets*, not multisets.

    Notes
    =====

    When all the elements are the same in the multiset, the order
    of the returned partitions is determined by the ``partitions``
    routine. If one is counting partitions then it is better to use
    the ``nT`` function.

    See Also
    ========

    partitions
    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition
    sympy.functions.combinatorial.numbers.nT

    """
    # This function looks at the supplied input and dispatches to
    # several special-case routines as they apply.
    if isinstance(multiset, int):
        n = multiset
        if m and m > n:
            return
        multiset = list(range(n))
        if m == 1:
            yield [multiset[:]]
            return

        # If m is not None, it can sometimes be faster to use
        # MultisetPartitionTraverser.enum_range() even for inputs
        # which are sets.  Since the _set_partitions code is quite
        # fast, this is only advantageous when the overall set
        # partitions outnumber those with the desired number of parts
        # by a large factor.  (At least 60.)  Such a switch is not
        # currently implemented.
        for nc, q in _set_partitions(n):
            if m is None or nc == m:
                rv = [[] for i in range(nc)]
                for i in range(n):
                    rv[q[i]].append(multiset[i])
                yield rv
        return

    if len(multiset) == 1 and isinstance(multiset, str):
        multiset = [multiset]

    if not has_variety(multiset):
        # Only one component, repeated n times.  The resulting
        # partitions correspond to partitions of integer n.
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return
        x = multiset[:1]
        for size, p in partitions(n, m, size=True):
            if m is None or size == m:
                rv = []
                for k in sorted(p):
                    rv.extend([x*k]*p[k])
                yield rv
    else:
        from sympy.core.sorting import ordered
        multiset = list(ordered(multiset))
        n = len(multiset)
        if m and m > n:
            return
        if m == 1:
            yield [multiset[:]]
            return

        # Split the information of the multiset into two lists -
        # one of the elements themselves, and one (of the same length)
        # giving the number of repeats for the corresponding element.
        elements, multiplicities = zip(*group(multiset, False))

        if len(elements) < len(multiset):
            # General case - multiset with more than one distinct element
            # and at least one element repeated more than once.
            if m:
                mpt = MultisetPartitionTraverser()
                for state in mpt.enum_range(multiplicities, m-1, m):
                    yield list_visitor(state, elements)
            else:
                for state in multiset_partitions_taocp(multiplicities):
                    yield list_visitor(state, elements)
        else:
            # Set partitions case - no repeated elements. Pretty much
            # same as int argument case above, with same possible, but
            # currently unimplemented optimization for some cases when
            # m is not None
            for nc, q in _set_partitions(n):
                if m is None or nc == m:
                    rv = [[] for i in range(nc)]
                    for i in range(n):
                        rv[q[i]].append(i)
                    yield [[multiset[j] for j in i] for i in rv]


def partitions(n, m=None, k=None, size=False):
    """Generate all partitions of positive integer, n.

    Each partition is represented as a dictionary, mapping an integer
    to the number of copies of that integer in the partition.  For example,
    the first partition of 4 returned is {4: 1}, "4: one of them".

    Parameters
    ==========
    n : int
    m : int, optional
        limits number of parts in partition (mnemonic: m, maximum parts)
    k : int, optional
        limits the numbers that are kept in the partition (mnemonic: k, keys)
    size : bool, default: False
        If ``True``, (M, P) is returned where M is the sum of the
        multiplicities and P is the generated partition.
        If ``False``, only the generated partition is returned.

    Examples
    ========

    >>> from sympy.utilities.iterables import partitions

    The numbers appearing in the partition (the key of the returned dict)
    are limited with k:

    >>> for p in partitions(6, k=2):  # doctest: +SKIP
    ...     print(p)
    {2: 3}
    {1: 2, 2: 2}
    {1: 4, 2: 1}
    {1: 6}

    The maximum number of parts in the partition (the sum of the values in
    the returned dict) are limited with m (default value, None, gives
    partitions from 1 through n):

    >>> for p in partitions(6, m=2):  # doctest: +SKIP
    ...     print(p)
    ...
    {6: 1}
    {1: 1, 5: 1}
    {2: 1, 4: 1}
    {3: 2}

    References
    ==========

    .. [1] modified from Tim Peter's version to allow for k and m values:
           https://code.activestate.com/recipes/218332-generator-for-integer-partitions/

    See Also
    ========

    sympy.combinatorics.partitions.Partition
    sympy.combinatorics.partitions.IntegerPartition

    """
    if (n <= 0 or
        m is not None and m < 1 or
        k is not None and k < 1 or
        m and k and m*k < n):
        # the empty set is the only way to handle these inputs
        # and returning {} to represent it is consistent with
        # the counting convention, e.g. nT(0) == 1.
        if size:
            yield 0, {}
        else:
            yield {}
        return

    if m is None:
        m = n
    else:
        m = min(m, n)
    k = min(k or n, n)

    n, m, k = as_int(n), as_int(m), as_int(k)
    q, r = divmod(n, k)
    ms = {k: q}
    keys = [k]  # ms.keys(), from largest to smallest
    if r:
        ms[r] = 1
        keys.append(r)
    room = m - q - bool(r)
    if size:
        yield sum(ms.values()), ms.copy()
    else:
        yield ms.copy()

    while keys != [1]:
        # Reuse any 1's.
        if keys[-1] == 1:
            del keys[-1]
            reuse = ms.pop(1)
            room += reuse
        else:
            reuse = 0

        while 1:
            # Let i be the smallest key larger than 1.  Reuse one
            # instance of i.
            i = keys[-1]
            newcount = ms[i] = ms[i] - 1
            reuse += i
            if newcount == 0:
                del keys[-1], ms[i]
            room += 1

            # Break the remainder into pieces of size i-1.
            i -= 1
            q, r = divmod(reuse, i)
            need = q + bool(r)
            if need > room:
                if not keys:
                    return
                continue

            ms[i] = q
            keys.append(i)
            if r:
                ms[r] = 1
                keys.append(r)
            break
        room -= need
        if size:
            yield sum(ms.values()), ms.copy()
        else:
            yield ms.copy()


def ordered_partitions(n, m=None, sort=True):
    """Generates ordered partitions of integer *n*.

    Parameters
    ==========
    n : int
    m : int, optional
        The default value gives partitions of all sizes else only
        those with size m. In addition, if *m* is not None then
        partitions are generated *in place* (see examples).
    sort : bool, default: True
        Controls whether partitions are
        returned in sorted order when *m* is not None; when False,
        the partitions are returned as fast as possible with elements
        sorted, but when m|n the partitions will not be in
        ascending lexicographical order.

    Examples
    ========

    >>> from sympy.utilities.iterables import ordered_partitions

    All partitions of 5 in ascending lexicographical:

    >>> for p in ordered_partitions(5):
    ...     print(p)
    [1, 1, 1, 1, 1]
    [1, 1, 1, 2]
    [1, 1, 3]
    [1, 2, 2]
    [1, 4]
    [2, 3]
    [5]

    Only partitions of 5 with two parts:

    >>> for p in ordered_partitions(5, 2):
    ...     print(p)
    [1, 4]
    [2, 3]

    When ``m`` is given, a given list objects will be used more than
    once for speed reasons so you will not see the correct partitions
    unless you make a copy of each as it is generated:

    >>> [p for p in ordered_partitions(7, 3)]
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2]]
    >>> [list(p) for p in ordered_partitions(7, 3)]
    [[1, 1, 5], [1, 2, 4], [1, 3, 3], [2, 2, 3]]

    When ``n`` is a multiple of ``m``, the elements are still sorted
    but the partitions themselves will be *unordered* if sort is False;
    the default is to return them in ascending lexicographical order.

    >>> for p in ordered_partitions(6, 2):
    ...     print(p)
    [1, 5]
    [2, 4]
    [3, 3]

    But if speed is more important than ordering, sort can be set to
    False:

    >>> for p in ordered_partitions(6, 2, sort=False):
    ...     print(p)
    [1, 5]
    [3, 3]
    [2, 4]

    References
    ==========

    .. [1] Generating Integer Partitions, [online],
        Available: https://jeromekelleher.net/generating-integer-partitions.html
    .. [2] Jerome Kelleher and Barry O'Sullivan, "Generating All
        Partitions: A Comparison Of Two Encodings", [online],
        Available: https://arxiv.org/pdf/0909.2331v2.pdf
    """
    if n < 1 or m is not None and m < 1:
        # the empty set is the only way to handle these inputs
        # and returning {} to represent it is consistent with
        # the counting convention, e.g. nT(0) == 1.
        yield []
        return

    if m is None:
        # The list `a`'s leading elements contain the partition in which
        # y is the biggest element and x is either the same as y or the
        # 2nd largest element; v and w are adjacent element indices
        # to which x and y are being assigned, respectively.
        a = [1]*n
        y = -1
        v = n
        while v > 0:
            v -= 1
            x = a[v] + 1
            while y >= 2 * x:
                a[v] = x
                y -= x
                v += 1
            w = v + 1
            while x <= y:
                a[v] = x
                a[w] = y
                yield a[:w + 1]
                x += 1
                y -= 1
            a[v] = x + y
            y = a[v] - 1
            yield a[:w]
    elif m == 1:
        yield [n]
    elif n == m:
        yield [1]*n
    else:
        # recursively generate partitions of size m
        for b in range(1, n//m + 1):
            a = [b]*m
            x = n - b*m
            if not x:
                if sort:
                    yield a
            elif not sort and x <= m:
                for ax in ordered_partitions(x, sort=False):
                    mi = len(ax)
                    a[-mi:] = [i + b for i in ax]
                    yield a
                    a[-mi:] = [b]*mi
            else:
                for mi in range(1, m):
                    for ax in ordered_partitions(x, mi, sort=True):
                        a[-mi:] = [i + b for i in ax]
                        yield a
                        a[-mi:] = [b]*mi


def binary_partitions(n):
    """
    Generates the binary partition of *n*.

    A binary partition consists only of numbers that are
    powers of two. Each step reduces a `2^{k+1}` to `2^k` and
    `2^k`. Thus 16 is converted to 8 and 8.

    Examples
    ========

    >>> from sympy.utilities.iterables import binary_partitions
    >>> for i in binary_partitions(5):
    ...     print(i)
    ...
    [4, 1]
    [2, 2, 1]
    [2, 1, 1, 1]
    [1, 1, 1, 1, 1]

    References
    ==========

    .. [1] TAOCP 4, section 7.2.1.5, problem 64

    """
    from math import ceil, log2
    power = int(2**(ceil(log2(n))))
    acc = 0
    partition = []
    while power:
        if acc + power <= n:
            partition.append(power)
            acc += power
        power >>= 1

    last_num = len(partition) - 1 - (n & 1)
    while last_num >= 0:
        yield partition
        if partition[last_num] == 2:
            partition[last_num] = 1
            partition.append(1)
            last_num -= 1
            continue
        partition.append(1)
        partition[last_num] >>= 1
        x = partition[last_num + 1] = partition[last_num]
        last_num += 1
        while x > 1:
            if x <= len(partition) - last_num - 1:
                del partition[-x + 1:]
                last_num += 1
                partition[last_num] = x
            else:
                x >>= 1
    yield [1]*n


def has_dups(seq):
    """Return True if there are any duplicate elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_dups, Dict, Set
    >>> has_dups((1, 2, 1))
    True
    >>> has_dups(range(3))
    False
    >>> all(has_dups(c) is False for c in (set(), Set(), dict(), Dict()))
    True
    """
    from sympy.core.containers import Dict
    from sympy.sets.sets import Set
    if isinstance(seq, (dict, set, Dict, Set)):
        return False
    unique = set()
    try:
        return any(True for s in seq if s in unique or unique.add(s))
    except TypeError:
        return len(seq) != len(list(uniq(seq)))


def has_variety(seq):
    """Return True if there are any different elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_variety

    >>> has_variety((1, 2, 1))
    True
    >>> has_variety((1, 1, 1))
    False
    """
    for i, s in enumerate(seq):
        if i == 0:
            sentinel = s
        else:
            if s != sentinel:
                return True
    return False


def uniq(seq, result=None):
    """
    Yield unique elements from ``seq`` as an iterator. The second
    parameter ``result``  is used internally; it is not necessary
    to pass anything for this.

    Note: changing the sequence during iteration will raise a
    RuntimeError if the size of the sequence is known; if you pass
    an iterator and advance the iterator you will change the
    output of this routine but there will be no warning.

    Examples
    ========

    >>> from sympy.utilities.iterables import uniq
    >>> dat = [1, 4, 1, 5, 4, 2, 1, 2]
    >>> type(uniq(dat)) in (list, tuple)
    False

    >>> list(uniq(dat))
    [1, 4, 5, 2]
    >>> list(uniq(x for x in dat))
    [1, 4, 5, 2]
    >>> list(uniq([[1], [2, 1], [1]]))
    [[1], [2, 1]]
    """
    try:
        n = len(seq)
    except TypeError:
        n = None
    def check():
        # check that size of seq did not change during iteration;
        # if n == None the object won't support size changing, e.g.
        # an iterator can't be changed
        if n is not None and len(seq) != n:
            raise RuntimeError('sequence changed size during iteration')
    try:
        seen = set()
        result = result or []
        for i, s in enumerate(seq):
            if not (s in seen or seen.add(s)):
                yield s
                check()
    except TypeError:
        if s not in result:
            yield s
            check()
            result.append(s)
        if hasattr(seq, '__getitem__'):
            yield from uniq(seq[i + 1:], result)
        else:
            yield from uniq(seq, result)


def generate_bell(n):
    """Return permutations of [0, 1, ..., n - 1] such that each permutation
    differs from the last by the exchange of a single pair of neighbors.
    The ``n!`` permutations are returned as an iterator. In order to obtain
    the next permutation from a random starting permutation, use the
    ``next_trotterjohnson`` method of the Permutation class (which generates
    the same sequence in a different manner).

    Examples
    ========

    >>> from itertools import permutations
    >>> from sympy.utilities.iterables import generate_bell
    >>> from sympy import zeros, Matrix

    This is the sort of permutation used in the ringing of physical bells,
    and does not produce permutations in lexicographical order. Rather, the
    permutations differ from each other by exactly one inversion, and the
    position at which the swapping occurs varies periodically in a simple
    fashion. Consider the first few permutations of 4 elements generated
    by ``permutations`` and ``generate_bell``:

    >>> list(permutations(range(4)))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
    >>> list(generate_bell(4))[:5]
    [(0, 1, 2, 3), (0, 1, 3, 2), (0, 3, 1, 2), (3, 0, 1, 2), (3, 0, 2, 1)]

    Notice how the 2nd and 3rd lexicographical permutations have 3 elements
    out of place whereas each "bell" permutation always has only two
    elements out of place relative to the previous permutation (and so the
    signature (+/-1) of a permutation is opposite of the signature of the
    previous permutation).

    How the position of inversion varies across the elements can be seen
    by tracing out where the largest number appears in the permutations:

    >>> m = zeros(4, 24)
    >>> for i, p in enumerate(generate_bell(4)):
    ...     m[:, i] = Matrix([j - 3 for j in list(p)])  # make largest zero
    >>> m.print_nonzero('X')
    [XXX  XXXXXX  XXXXXX  XXX]
    [XX XX XXXX XX XXXX XX XX]
    [X XXXX XX XXXX XX XXXX X]
    [ XXXXXX  XXXXXX  XXXXXX ]

    See Also
    ========

    sympy.combinatorics.permutations.Permutation.next_trotterjohnson

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Method_ringing

    .. [2] https://stackoverflow.com/questions/4856615/recursive-permutation/4857018

    .. [3] https://web.archive.org/web/20160313023044/http://programminggeeks.com/bell-algorithm-for-permutation/

    .. [4] https://en.wikipedia.org/wiki/Steinhaus%E2%80%93Johnson%E2%80%93Trotter_algorithm

    .. [5] Generating involutions, derangements, and relatives by ECO
           Vincent Vajnovszki, DMTCS vol 1 issue 12, 2010

    """
    n = as_int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')
    if n == 1:
        yield (0,)
    elif n == 2:
        yield (0, 1)
        yield (1, 0)
    elif n == 3:
        yield from [(0, 1, 2), (0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)]
    else:
        m = n - 1
        op = [0] + [-1]*m
        l = list(range(n))
        while True:
            yield tuple(l)
            # find biggest element with op
            big = None, -1  # idx, value
            for i in range(n):
                if op[i] and l[i] > big[1]:
                    big = i, l[i]
            i, _ = big
            if i is None:
                break  # there are no ops left
            # swap it with neighbor in the indicated direction
            j = i + op[i]
            l[i], l[j] = l[j], l[i]
            op[i], op[j] = op[j], op[i]
            # if it landed at the end or if the neighbor in the same
            # direction is bigger then turn off op
            if j == 0 or j == m or l[j + op[j]] > l[j]:
                op[j] = 0
            # any element bigger to the left gets +1 op
            for i in range(j):
                if l[i] > l[j]:
                    op[i] = 1
            # any element bigger to the right gets -1 op
            for i in range(j + 1, n):
                if l[i] > l[j]:
                    op[i] = -1


def generate_involutions(n):
    """
    Generates involutions.

    An involution is a permutation that when multiplied
    by itself equals the identity permutation. In this
    implementation the involutions are generated using
    Fixed Points.

    Alternatively, an involution can be considered as
    a permutation that does not contain any cycles with
    a length that is greater than two.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_involutions
    >>> list(generate_involutions(3))
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    >>> len(list(generate_involutions(4)))
    10

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PermutationInvolution.html

    """
    idx = list(range(n))
    for p in permutations(idx):
        for i in idx:
            if p[p[i]] != i:
                break
        else:
            yield p


def multiset_derangements(s):
    """Generate derangements of the elements of s *in place*.

    Examples
    ========

    >>> from sympy.utilities.iterables import multiset_derangements, uniq

    Because the derangements of multisets (not sets) are generated
    in place, copies of the return value must be made if a collection
    of derangements is desired or else all values will be the same:

    >>> list(uniq([i for i in multiset_derangements('1233')]))
    [[None, None, None, None]]
    >>> [i.copy() for i in multiset_derangements('1233')]
    [['3', '3', '1', '2'], ['3', '3', '2', '1']]
    >>> [''.join(i) for i in multiset_derangements('1233')]
    ['3312', '3321']
    """
    from sympy.core.sorting import ordered
    # create multiset dictionary of hashable elements or else
    # remap elements to integers
    try:
        ms = multiset(s)
    except TypeError:
        # give each element a canonical integer value
        key = dict(enumerate(ordered(uniq(s))))
        h = []
        for si in s:
            for k in key:
                if key[k] == si:
                    h.append(k)
                    break
        for i in multiset_derangements(h):
            yield [key[j] for j in i]
        return

    mx = max(ms.values())  # max repetition of any element
    n = len(s)  # the number of elements

    ## special cases

    # 1) one element has more than half the total cardinality of s: no
    # derangements are possible.
    if mx*2 > n:
        return

    # 2) all elements appear once: singletons
    if len(ms) == n:
        yield from _set_derangements(s)
        return

    # find the first element that is repeated the most to place
    # in the following two special cases where the selection
    # is unambiguous: either there are two elements with multiplicity
    # of mx or else there is only one with multiplicity mx
    for M in ms:
        if ms[M] == mx:
            break

    inonM = [i for i in range(n) if s[i] != M]  # location of non-M
    iM = [i for i in range(n) if s[i] == M]  # locations of M
    rv = [None]*n

    # 3) half are the same
    if 2*mx == n:
        # M goes into non-M locations
        for i in inonM:
            rv[i] = M
        # permutations of non-M go to M locations
        for p in multiset_permutations([s[i] for i in inonM]):
            for i, pi in zip(iM, p):
                rv[i] = pi
            yield rv
        # clean-up (and encourages proper use of routine)
        rv[:] = [None]*n
        return

    # 4) single repeat covers all but 1 of the non-repeats:
    # if there is one repeat then the multiset of the values
    # of ms would be {mx: 1, 1: n - mx}, i.e. there would
    # be n - mx + 1 values with the condition that n - 2*mx = 1
    if n - 2*mx == 1 and len(ms.values()) == n - mx + 1:
        for i, i1 in enumerate(inonM):
            ifill = inonM[:i] + inonM[i+1:]
            for j in ifill:
                rv[j] = M
            for p in permutations([s[j] for j in ifill]):
                rv[i1] = s[i1]
                for j, pi in zip(iM, p):
                    rv[j] = pi
                k = i1
                for j in iM:
                    rv[j], rv[k] = rv[k], rv[j]
                    yield rv
                    k = j
        # clean-up (and encourages proper use of routine)
        rv[:] = [None]*n
        return

    ## general case is handled with 3 helpers:
    #    1) `finish_derangements` will place the last two elements
    #       which have arbitrary multiplicities, e.g. for multiset
    #       {c: 3, a: 2, b: 2}, the last two elements are a and b
    #    2) `iopen` will tell where a given element can be placed
    #    3) `do` will recursively place elements into subsets of
    #        valid locations

    def finish_derangements():
        """Place the last two elements into the partially completed
        derangement, and yield the results.
        """

        a = take[1][0]  # penultimate element
        a_ct = take[1][1]
        b = take[0][0]  # last element to be placed
        b_ct = take[0][1]

        # split the indexes of the not-already-assigned elements of rv into
        # three categories
        forced_a = []  # positions which must have an a
        forced_b = []  # positions which must have a b
        open_free = []  # positions which could take either
        for i in range(len(s)):
            if rv[i] is None:
                if s[i] == a:
                    forced_b.append(i)
                elif s[i] == b:
                    forced_a.append(i)
                else:
                    open_free.append(i)

        if len(forced_a) > a_ct or len(forced_b) > b_ct:
            # No derangement possible
            return

        for i in forced_a:
            rv[i] = a
        for i in forced_b:
            rv[i] = b
        for a_place in combinations(open_free, a_ct - len(forced_a)):
            for a_pos in a_place:
                rv[a_pos] = a
            for i in open_free:
                if rv[i] is None:  # anything not in the subset is set to b
                    rv[i] = b
            yield rv
            # Clean up/undo the final placements
            for i in open_free:
                rv[i] = None

        # additional cleanup - clear forced_a, forced_b
        for i in forced_a:
            rv[i] = None
        for i in forced_b:
            rv[i] = None

    def iopen(v):
        # return indices at which element v can be placed in rv:
        # locations which are not already occupied if that location
        # does not already contain v in the same location of s
        return [i for i in range(n) if rv[i] is None and s[i] != v]

    def do(j):
        if j == 1:
            # handle the last two elements (regardless of multiplicity)
            # with a special method
            yield from finish_derangements()
        else:
            # place the mx elements of M into a subset of places
            # into which it can be replaced
            M, mx = take[j]
            for i in combinations(iopen(M), mx):
                # place M
                for ii in i:
                    rv[ii] = M
                # recursively place the next element
                yield from do(j - 1)
                # mark positions where M was placed as once again
                # open for placement of other elements
                for ii in i:
                    rv[ii] = None

    # process elements in order of canonically decreasing multiplicity
    take = sorted(ms.items(), key=lambda x:(x[1], x[0]))
    yield from do(len(take) - 1)
    rv[:] = [None]*n


def random_derangement(t, choice=None, strict=True):
    """Return a list of elements in which none are in the same positions
    as they were originally. If an element fills more than half of the positions
    then an error will be raised since no derangement is possible. To obtain
    a derangement of as many items as possible--with some of the most numerous
    remaining in their original positions--pass `strict=False`. To produce a
    pseudorandom derangment, pass a pseudorandom selector like `choice` (see
    below).

    Examples
    ========

    >>> from sympy.utilities.iterables import random_derangement
    >>> t = 'SymPy: a CAS in pure Python'
    >>> d = random_derangement(t)
    >>> all(i != j for i, j in zip(d, t))
    True

    A predictable result can be obtained by using a pseudorandom
    generator for the choice:

    >>> from sympy.core.random import seed, choice as c
    >>> seed(1)
    >>> d = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert len(set(d)) != 1  # we got different values

    By reseeding, the same sequence can be obtained:

    >>> seed(1)
    >>> d2 = [''.join(random_derangement(t, c)) for i in range(5)]
    >>> assert d == d2
    """
    if choice is None:
        import secrets
        choice = secrets.choice
    def shuffle(rv):
        '''Knuth shuffle'''
        for i in range(len(rv) - 1, 0, -1):
            x = choice(rv[:i + 1])
            j = rv.index(x)
            rv[i], rv[j] = rv[j], rv[i]
    def pick(rv, n):
        '''shuffle rv and return the first n values
        '''
        shuffle(rv)
        return rv[:n]
    ms = multiset(t)
    tot = len(t)
    ms = sorted(ms.items(), key=lambda x: x[1])
  # if there are not enough spaces for the most
  # plentiful element to move to then some of them
  # will have to stay in place
    M, mx = ms[-1]
    n = len(t)
    xs = 2*mx - tot
    if xs > 0:
        if strict:
            raise ValueError('no derangement possible')
        opts = [i for (i, c) in enumerate(t) if c == ms[-1][0]]
        pick(opts, xs)
        stay = sorted(opts[:xs])
        rv = list(t)
        for i in reversed(stay):
            rv.pop(i)
        rv = random_derangement(rv, choice)
        for i in stay:
            rv.insert(i, ms[-1][0])
        return ''.join(rv) if type(t) is str else rv
  # the normal derangement calculated from here
    if n == len(ms):
      # approx 1/3 will succeed
        rv = list(t)
        while True:
            shuffle(rv)
            if all(i != j for i,j in zip(rv, t)):
                break
    else:
      # general case
        rv = [None]*n
        while True:
            j = 0
            while j > -len(ms):  # do most numerous first
                j -= 1
                e, c = ms[j]
                opts = [i for i in range(n) if rv[i] is None and t[i] != e]
                if len(opts) < c:
                    for i in range(n):
                        rv[i] = None
                    break # try again
                pick(opts, c)
                for i in range(c):
                    rv[opts[i]] = e
            else:
                return rv
    return rv


def _set_derangements(s):
    """
    yield derangements of items in ``s`` which are assumed to contain
    no repeated elements
    """
    if len(s) < 2:
        return
    if len(s) == 2:
        yield [s[1], s[0]]
        return
    if len(s) == 3:
        yield [s[1], s[2], s[0]]
        yield [s[2], s[0], s[1]]
        return
    for p in permutations(s):
        if not any(i == j for i, j in zip(p, s)):
            yield list(p)


def generate_derangements(s):
    """
    Return unique derangements of the elements of iterable ``s``.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements
    >>> list(generate_derangements([0, 1, 2]))
    [[1, 2, 0], [2, 0, 1]]
    >>> list(generate_derangements([0, 1, 2, 2]))
    [[2, 2, 0, 1], [2, 2, 1, 0]]
    >>> list(generate_derangements([0, 1, 1]))
    []

    See Also
    ========

    sympy.functions.combinatorial.factorials.subfactorial

    """
    if not has_dups(s):
        yield from _set_derangements(s)
    else:
        for p in multiset_derangements(s):
            yield list(p)


def necklaces(n, k, free=False):
    """
    A routine to generate necklaces that may (free=True) or may not
    (free=False) be turned over to be viewed. The "necklaces" returned
    are comprised of ``n`` integers (beads) with ``k`` different
    values (colors). Only unique necklaces are returned.

    Examples
    ========

    >>> from sympy.utilities.iterables import necklaces, bracelets
    >>> def show(s, i):
    ...     return ''.join(s[j] for j in i)

    The "unrestricted necklace" is sometimes also referred to as a
    "bracelet" (an object that can be turned over, a sequence that can
    be reversed) and the term "necklace" is used to imply a sequence
    that cannot be reversed. So ACB == ABC for a bracelet (rotate and
    reverse) while the two are different for a necklace since rotation
    alone cannot make the two sequences the same.

    (mnemonic: Bracelets can be viewed Backwards, but Not Necklaces.)

    >>> B = [show('ABC', i) for i in bracelets(3, 3)]
    >>> N = [show('ABC', i) for i in necklaces(3, 3)]
    >>> set(N) - set(B)
    {'ACB'}

    >>> list(necklaces(4, 2))
    [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1),
     (0, 1, 0, 1), (0, 1, 1, 1), (1, 1, 1, 1)]

    >>> [show('.o', i) for i in bracelets(4, 2)]
    ['....', '...o', '..oo', '.o.o', '.ooo', 'oooo']

    References
    ==========

    .. [1] https://mathworld.wolfram.com/Necklace.html

    .. [2] Frank Ruskey, Carla Savage, and Terry Min Yih Wang,
        Generating necklaces, Journal of Algorithms 13 (1992), 414-430;
        https://doi.org/10.1016/0196-6774(92)90047-G

    """
    # The FKM algorithm
    if k == 0 and n > 0:
        return
    a = [0]*n
    yield tuple(a)
    if n == 0:
        return
    while True:
        i = n - 1
        while a[i] == k - 1:
            i -= 1
            if i == -1:
                return
        a[i] += 1
        for j in range(n - i - 1):
            a[j + i + 1] = a[j]
        if n % (i + 1) == 0 and (not free or all(a <= a[j::-1] + a[-1:j:-1] for j in range(n - 1))):
            # No need to test j = n - 1.
            yield tuple(a)


def bracelets(n, k):
    """Wrapper to necklaces to return a free (unrestricted) necklace."""
    return necklaces(n, k, free=True)


def generate_oriented_forest(n):
    """
    This algorithm generates oriented forests.

    An oriented graph is a directed graph having no symmetric pair of directed
    edges. A forest is an acyclic graph, i.e., it has no cycles. A forest can
    also be described as a disjoint union of trees, which are graphs in which
    any two vertices are connected by exactly one simple path.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_oriented_forest
    >>> list(generate_oriented_forest(4))
    [[0, 1, 2, 3], [0, 1, 2, 2], [0, 1, 2, 1], [0, 1, 2, 0], \
    [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]]

    References
    ==========

    .. [1] T. Beyer and S.M. Hedetniemi: constant time generation of
           rooted trees, SIAM J. Computing Vol. 9, No. 4, November 1980

    .. [2] https://stackoverflow.com/questions/1633833/oriented-forest-taocp-algorithm-in-python

    """
    P = list(range(-1, n))
    while True:
        yield P[1:]
        if P[n] > 0:
            P[n] = P[P[n]]
        else:
            for p in range(n - 1, 0, -1):
                if P[p] != 0:
                    target = P[p] - 1
                    for q in range(p - 1, 0, -1):
                        if P[q] == target:
                            break
                    offset = p - q
                    for i in range(p, n + 1):
                        P[i] = P[i - offset]
                    break
            else:
                break


def minlex(seq, directed=True, key=None):
    r"""
    Return the rotation of the sequence in which the lexically smallest
    elements appear first, e.g. `cba \rightarrow acb`.

    The sequence returned is a tuple, unless the input sequence is a string
    in which case a string is returned.

    If ``directed`` is False then the smaller of the sequence and the
    reversed sequence is returned, e.g. `cba \rightarrow abc`.

    If ``key`` is not None then it is used to extract a comparison key from each element in iterable.

    Examples
    ========

    >>> from sympy.combinatorics.polyhedron import minlex
    >>> minlex((1, 2, 0))
    (0, 1, 2)
    >>> minlex((1, 0, 2))
    (0, 2, 1)
    >>> minlex((1, 0, 2), directed=False)
    (0, 1, 2)

    >>> minlex('11010011000', directed=True)
    '00011010011'
    >>> minlex('11010011000', directed=False)
    '00011001011'

    >>> minlex(('bb', 'aaa', 'c', 'a'))
    ('a', 'bb', 'aaa', 'c')
    >>> minlex(('bb', 'aaa', 'c', 'a'), key=len)
    ('c', 'a', 'bb', 'aaa')

    """
    from sympy.functions.elementary.miscellaneous import Id
    if key is None: key = Id
    best = rotate_left(seq, least_rotation(seq, key=key))
    if not directed:
        rseq = seq[::-1]
        rbest = rotate_left(rseq, least_rotation(rseq, key=key))
        best = min(best, rbest, key=key)

    # Convert to tuple, unless we started with a string.
    return tuple(best) if not isinstance(seq, str) else best


def runs(seq, op=gt):
    """Group the sequence into lists in which successive elements
    all compare the same with the comparison operator, ``op``:
    op(seq[i + 1], seq[i]) is True from all elements in a run.

    Examples
    ========

    >>> from sympy.utilities.iterables import runs
    >>> from operator import ge
    >>> runs([0, 1, 2, 2, 1, 4, 3, 2, 2])
    [[0, 1, 2], [2], [1, 4], [3], [2], [2]]
    >>> runs([0, 1, 2, 2, 1, 4, 3, 2, 2], op=ge)
    [[0, 1, 2, 2], [1, 4], [3], [2, 2]]
    """
    cycles = []
    seq = iter(seq)
    try:
        run = [next(seq)]
    except StopIteration:
        return []
    while True:
        try:
            ei = next(seq)
        except StopIteration:
            break
        if op(ei, run[-1]):
            run.append(ei)
            continue
        else:
            cycles.append(run)
            run = [ei]
    if run:
        cycles.append(run)
    return cycles


def sequence_partitions(l, n, /):
    r"""Returns the partition of sequence $l$ into $n$ bins

    Explanation
    ===========

    Given the sequence $l_1 \cdots l_m \in V^+$ where
    $V^+$ is the Kleene plus of $V$

    The set of $n$ partitions of $l$ is defined as:

    .. math::
        \{(s_1, \cdots, s_n) | s_1 \in V^+, \cdots, s_n \in V^+,
        s_1 \cdots s_n = l_1 \cdots l_m\}

    Parameters
    ==========

    l : Sequence[T]
        A nonempty sequence of any Python objects

    n : int
        A positive integer

    Yields
    ======

    out : list[Sequence[T]]
        A list of sequences with concatenation equals $l$.
        This should conform with the type of $l$.

    Examples
    ========

    >>> from sympy.utilities.iterables import sequence_partitions
    >>> for out in sequence_partitions([1, 2, 3, 4], 2):
    ...     print(out)
    [[1], [2, 3, 4]]
    [[1, 2], [3, 4]]
    [[1, 2, 3], [4]]

    Notes
    =====

    This is modified version of EnricoGiampieri's partition generator
    from https://stackoverflow.com/questions/13131491/partition-n-items-into-k-bins-in-python-lazily

    See Also
    ========

    sequence_partitions_empty
    """
    # Asserting l is nonempty is done only for sanity check
    if n == 1 and l:
        yield [l]
        return
    for i in range(1, len(l)):
        for part in sequence_partitions(l[i:], n - 1):
            yield [l[:i]] + part


def sequence_partitions_empty(l, n, /):
    r"""Returns the partition of sequence $l$ into $n$ bins with
    empty sequence

    Explanation
    ===========

    Given the sequence $l_1 \cdots l_m \in V^*$ where
    $V^*$ is the Kleene star of $V$

    The set of $n$ partitions of $l$ is defined as:

    .. math::
        \{(s_1, \cdots, s_n) | s_1 \in V^*, \cdots, s_n \in V^*,
        s_1 \cdots s_n = l_1 \cdots l_m\}

    There are more combinations than :func:`sequence_partitions` because
    empty sequence can fill everywhere, so we try to provide different
    utility for this.

    Parameters
    ==========

    l : Sequence[T]
        A sequence of any Python objects (can be possibly empty)

    n : int
        A positive integer

    Yields
    ======

    out : list[Sequence[T]]
        A list of sequences with concatenation equals $l$.
        This should conform with the type of $l$.

    Examples
    ========

    >>> from sympy.utilities.iterables import sequence_partitions_empty
    >>> for out in sequence_partitions_empty([1, 2, 3, 4], 2):
    ...     print(out)
    [[], [1, 2, 3, 4]]
    [[1], [2, 3, 4]]
    [[1, 2], [3, 4]]
    [[1, 2, 3], [4]]
    [[1, 2, 3, 4], []]

    See Also
    ========

    sequence_partitions
    """
    if n < 1:
        return
    if n == 1:
        yield [l]
        return
    for i in range(0, len(l) + 1):
        for part in sequence_partitions_empty(l[i:], n - 1):
            yield [l[:i]] + part


def kbins(l, k, ordered=None):
    """
    Return sequence ``l`` partitioned into ``k`` bins.

    Examples
    ========

    The default is to give the items in the same order, but grouped
    into k partitions without any reordering:

    >>> from sympy.utilities.iterables import kbins
    >>> for p in kbins(list(range(5)), 2):
    ...     print(p)
    ...
    [[0], [1, 2, 3, 4]]
    [[0, 1], [2, 3, 4]]
    [[0, 1, 2], [3, 4]]
    [[0, 1, 2, 3], [4]]

    The ``ordered`` flag is either None (to give the simple partition
    of the elements) or is a 2 digit integer indicating whether the order of
    the bins and the order of the items in the bins matters. Given::

        A = [[0], [1, 2]]
        B = [[1, 2], [0]]
        C = [[2, 1], [0]]
        D = [[0], [2, 1]]

    the following values for ``ordered`` have the shown meanings::

        00 means A == B == C == D
        01 means A == B
        10 means A == D
        11 means A == A

    >>> for ordered_flag in [None, 0, 1, 10, 11]:
    ...     print('ordered = %s' % ordered_flag)
    ...     for p in kbins(list(range(3)), 2, ordered=ordered_flag):
    ...         print('     %s' % p)
    ...
    ordered = None
         [[0], [1, 2]]
         [[0, 1], [2]]
    ordered = 0
         [[0, 1], [2]]
         [[0, 2], [1]]
         [[0], [1, 2]]
    ordered = 1
         [[0], [1, 2]]
         [[0], [2, 1]]
         [[1], [0, 2]]
         [[1], [2, 0]]
         [[2], [0, 1]]
         [[2], [1, 0]]
    ordered = 10
         [[0, 1], [2]]
         [[2], [0, 1]]
         [[0, 2], [1]]
         [[1], [0, 2]]
         [[0], [1, 2]]
         [[1, 2], [0]]
    ordered = 11
         [[0], [1, 2]]
         [[0, 1], [2]]
         [[0], [2, 1]]
         [[0, 2], [1]]
         [[1], [0, 2]]
         [[1, 0], [2]]
         [[1], [2, 0]]
         [[1, 2], [0]]
         [[2], [0, 1]]
         [[2, 0], [1]]
         [[2], [1, 0]]
         [[2, 1], [0]]

    See Also
    ========

    partitions, multiset_partitions

    """
    if ordered is None:
        yield from sequence_partitions(l, k)
    elif ordered == 11:
        for pl in multiset_permutations(l):
            pl = list(pl)
            yield from sequence_partitions(pl, k)
    elif ordered == 00:
        yield from multiset_partitions(l, k)
    elif ordered == 10:
        for p in multiset_partitions(l, k):
            for perm in permutations(p):
                yield list(perm)
    elif ordered == 1:
        for kgot, p in partitions(len(l), k, size=True):
            if kgot != k:
                continue
            for li in multiset_permutations(l):
                rv = []
                i = j = 0
                li = list(li)
                for size, multiplicity in sorted(p.items()):
                    for m in range(multiplicity):
                        j = i + size
                        rv.append(li[i: j])
                        i = j
                yield rv
    else:
        raise ValueError(
            'ordered must be one of 00, 01, 10 or 11, not %s' % ordered)


def permute_signs(t):
    """Return iterator in which the signs of non-zero elements
    of t are permuted.

    Examples
    ========

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2)]
    """
    for signs in product(*[(1, -1)]*(len(t) - t.count(0))):
        signs = list(signs)
        yield type(t)([i*signs.pop() if i else i for i in t])


def signed_permutations(t):
    """Return iterator in which the signs of non-zero elements
    of t and the order of the elements are permuted and all
    returned values are unique.

    Examples
    ========

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((0, 1, 2)))
    [(0, 1, 2), (0, -1, 2), (0, 1, -2), (0, -1, -2), (0, 2, 1),
    (0, -2, 1), (0, 2, -1), (0, -2, -1), (1, 0, 2), (-1, 0, 2),
    (1, 0, -2), (-1, 0, -2), (1, 2, 0), (-1, 2, 0), (1, -2, 0),
    (-1, -2, 0), (2, 0, 1), (-2, 0, 1), (2, 0, -1), (-2, 0, -1),
    (2, 1, 0), (-2, 1, 0), (2, -1, 0), (-2, -1, 0)]
    """
    return (type(t)(i) for j in multiset_permutations(t)
        for i in permute_signs(j))


def rotations(s, dir=1):
    """Return a generator giving the items in s as list where
    each subsequent list has the items rotated to the left (default)
    or right (``dir=-1``) relative to the previous list.

    Examples
    ========

    >>> from sympy import rotations
    >>> list(rotations([1,2,3]))
    [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    >>> list(rotations([1,2,3], -1))
    [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    """
    seq = list(s)
    for i in range(len(seq)):
        yield seq
        seq = rotate_left(seq, dir)


def roundrobin(*iterables):
    """roundrobin recipe taken from itertools documentation:
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    roundrobin('ABC', 'D', 'EF') --> A D E B F C

    Recipe credited to George Sakkis
    """
    nexts = cycle(iter(it).__next__ for it in iterables)

    pending = len(iterables)
    while pending:
        try:
            for nxt in nexts:
                yield nxt()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))



class NotIterable:
    """
    Use this as mixin when creating a class which is not supposed to
    return true when iterable() is called on its instances because
    calling list() on the instance, for example, would result in
    an infinite loop.
    """
    pass


def iterable(i, exclude=(str, dict, NotIterable)):
    """
    Return a boolean indicating whether ``i`` is SymPy iterable.
    True also indicates that the iterator is finite, e.g. you can
    call list(...) on the instance.

    When SymPy is working with iterables, it is almost always assuming
    that the iterable is not a string or a mapping, so those are excluded
    by default. If you want a pure Python definition, make exclude=None. To
    exclude multiple items, pass them as a tuple.

    You can also set the _iterable attribute to True or False on your class,
    which will override the checks here, including the exclude test.

    As a rule of thumb, some SymPy functions use this to check if they should
    recursively map over an object. If an object is technically iterable in
    the Python sense but does not desire this behavior (e.g., because its
    iteration is not finite, or because iteration might induce an unwanted
    computation), it should disable it by setting the _iterable attribute to False.

    See also: is_sequence

    Examples
    ========

    >>> from sympy.utilities.iterables import iterable
    >>> from sympy import Tuple
    >>> things = [[1], (1,), set([1]), Tuple(1), (j for j in [1, 2]), {1:2}, '1', 1]
    >>> for i in things:
    ...     print('%s %s' % (iterable(i), type(i)))
    True <... 'list'>
    True <... 'tuple'>
    True <... 'set'>
    True <class 'sympy.core.containers.Tuple'>
    True <... 'generator'>
    False <... 'dict'>
    False <... 'str'>
    False <... 'int'>

    >>> iterable({}, exclude=None)
    True
    >>> iterable({}, exclude=str)
    True
    >>> iterable("no", exclude=str)
    False

    """
    if hasattr(i, '_iterable'):
        return i._iterable
    try:
        iter(i)
    except TypeError:
        return False
    if exclude:
        return not isinstance(i, exclude)
    return True


def is_sequence(i, include=None):
    """
    Return a boolean indicating whether ``i`` is a sequence in the SymPy
    sense. If anything that fails the test below should be included as
    being a sequence for your application, set 'include' to that object's
    type; multiple types should be passed as a tuple of types.

    Note: although generators can generate a sequence, they often need special
    handling to make sure their elements are captured before the generator is
    exhausted, so these are not included by default in the definition of a
    sequence.

    See also: iterable

    Examples
    ========

    >>> from sympy.utilities.iterables import is_sequence
    >>> from types import GeneratorType
    >>> is_sequence([])
    True
    >>> is_sequence(set())
    False
    >>> is_sequence('abc')
    False
    >>> is_sequence('abc', include=str)
    True
    >>> generator = (c for c in 'abc')
    >>> is_sequence(generator)
    False
    >>> is_sequence(generator, include=(str, GeneratorType))
    True

    """
    return (hasattr(i, '__getitem__') and
            iterable(i) or
            bool(include) and
            isinstance(i, include))


@deprecated(
    """
    Using postorder_traversal from the sympy.utilities.iterables submodule is
    deprecated.

    Instead, use postorder_traversal from the top-level sympy namespace, like

        sympy.postorder_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved")
def postorder_traversal(node, keys=None):
    from sympy.core.traversal import postorder_traversal as _postorder_traversal
    return _postorder_traversal(node, keys=keys)


@deprecated(
    """
    Using interactive_traversal from the sympy.utilities.iterables submodule
    is deprecated.

    Instead, use interactive_traversal from the top-level sympy namespace,
    like

        sympy.interactive_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved")
def interactive_traversal(expr):
    from sympy.interactive.traversal import interactive_traversal as _interactive_traversal
    return _interactive_traversal(expr)


@deprecated(
    """
    Importing default_sort_key from sympy.utilities.iterables is deprecated.
    Use from sympy import default_sort_key instead.
    """,
    deprecated_since_version="1.10",
active_deprecations_target="deprecated-sympy-core-compatibility",
)
def default_sort_key(*args, **kwargs):
    from sympy import default_sort_key as _default_sort_key
    return _default_sort_key(*args, **kwargs)


@deprecated(
    """
    Importing default_sort_key from sympy.utilities.iterables is deprecated.
    Use from sympy import default_sort_key instead.
    """,
    deprecated_since_version="1.10",
active_deprecations_target="deprecated-sympy-core-compatibility",
)
def ordered(*args, **kwargs):
    from sympy import ordered as _ordered
    return _ordered(*args, **kwargs)
<|endoftext|>"""Miscellaneous stuff that does not really fit anywhere else."""

from __future__ import annotations

import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent


class Undecidable(ValueError):
    # an error to be raised when a decision cannot be made definitively
    # where a definitive answer is needed
    pass


def filldedent(s, w=70, **kwargs):
    """
    Strips leading and trailing empty lines from a copy of ``s``, then dedents,
    fills and returns it.

    Empty line stripping serves to deal with docstrings like this one that
    start with a newline after the initial triple quote, inserting an empty
    line at the beginning of the string.

    Additional keyword arguments will be passed to ``textwrap.fill()``.

    See Also
    ========
    strlines, rawlines

    """
    return '\n' + fill(dedent(str(s)).strip('\n'), width=w, **kwargs)


def strlines(s, c=64, short=False):
    """Return a cut-and-pastable string that, when printed, is
    equivalent to the input.  The lines will be surrounded by
    parentheses and no line will be longer than c (default 64)
    characters. If the line contains newlines characters, the
    `rawlines` result will be returned.  If ``short`` is True
    (default is False) then if there is one line it will be
    returned without bounding parentheses.

    Examples
    ========

    >>> from sympy.utilities.misc import strlines
    >>> q = 'this is a long string that should be broken into shorter lines'
    >>> print(strlines(q, 40))
    (
    'this is a long string that should be b'
    'roken into shorter lines'
    )
    >>> q == (
    ... 'this is a long string that should be b'
    ... 'roken into shorter lines'
    ... )
    True

    See Also
    ========
    filldedent, rawlines
    """
    if not isinstance(s, str):
        raise ValueError('expecting string input')
    if '\n' in s:
        return rawlines(s)
    q = '"' if repr(s).startswith('"') else "'"
    q = (q,)*2
    if '\\' in s:  # use r-string
        m = '(\nr%s%%s%s\n)' % q
        j = '%s\nr%s' % q
        c -= 3
    else:
        m = '(\n%s%%s%s\n)' % q
        j = '%s\n%s' % q
        c -= 2
    out = []
    while s:
        out.append(s[:c])
        s=s[c:]
    if short and len(out) == 1:
        return (m % out[0]).splitlines()[1]  # strip bounding (\n...\n)
    return m % j.join(out)


def rawlines(s):
    """Return a cut-and-pastable string that, when printed, is equivalent
    to the input. Use this when there is more than one line in the
    string. The string returned is formatted so it can be indented
    nicely within tests; in some cases it is wrapped in the dedent
    function which has to be imported from textwrap.

    Examples
    ========

    Note: because there are characters in the examples below that need
    to be escaped because they are themselves within a triple quoted
    docstring, expressions below look more complicated than they would
    be if they were printed in an interpreter window.

    >>> from sympy.utilities.misc import rawlines
    >>> from sympy import TableForm
    >>> s = str(TableForm([[1, 10]], headings=(None, ['a', 'bee'])))
    >>> print(rawlines(s))
    (
        'a bee\\n'
        '-----\\n'
        '1 10 '
    )
    >>> print(rawlines('''this
    ... that'''))
    dedent('''\\
        this
        that''')

    >>> print(rawlines('''this
    ... that
    ... '''))
    dedent('''\\
        this
        that
        ''')

    >>> s = \"\"\"this
    ... is a triple '''
    ... \"\"\"
    >>> print(rawlines(s))
    dedent(\"\"\"\\
        this
        is a triple '''
        \"\"\")

    >>> print(rawlines('''this
    ... that
    ...     '''))
    (
        'this\\n'
        'that\\n'
        '    '
    )

    See Also
    ========
    filldedent, strlines
    """
    lines = s.split('\n')
    if len(lines) == 1:
        return repr(lines[0])
    triple = ["'''" in s, '"""' in s]
    if any(li.endswith(' ') for li in lines) or '\\' in s or all(triple):
        rv = []
        # add on the newlines
        trailing = s.endswith('\n')
        last = len(lines) - 1
        for i, li in enumerate(lines):
            if i != last or trailing:
                rv.append(repr(li + '\n'))
            else:
                rv.append(repr(li))
        return '(\n    %s\n)' % '\n    '.join(rv)
    else:
        rv = '\n    '.join(lines)
        if triple[0]:
            return 'dedent("""\\\n    %s""")' % rv
        else:
            return "dedent('''\\\n    %s''')" % rv

ARCH = str(struct.calcsize('P') * 8) + "-bit"


# XXX: PyPy does not support hash randomization
HASH_RANDOMIZATION = getattr(sys.flags, 'hash_randomization', False)

_debug_tmp: list[str] = []
_debug_iter = 0

def debug_decorator(func):
    """If SYMPY_DEBUG is True, it will print a nice execution tree with
    arguments and results of all decorated functions, else do nothing.
    """
    from sympy import SYMPY_DEBUG

    if not SYMPY_DEBUG:
        return func

    def maketree(f, *args, **kw):
        global _debug_tmp, _debug_iter
        oldtmp = _debug_tmp
        _debug_tmp = []
        _debug_iter += 1

        def tree(subtrees):
            def indent(s, variant=1):
                x = s.split("\n")
                r = "+-%s\n" % x[0]
                for a in x[1:]:
                    if a == "":
                        continue
                    if variant == 1:
                        r += "| %s\n" % a
                    else:
                        r += "  %s\n" % a
                return r
            if len(subtrees) == 0:
                return ""
            f = []
            for a in subtrees[:-1]:
                f.append(indent(a))
            f.append(indent(subtrees[-1], 2))
            return ''.join(f)

        # If there is a bug and the algorithm enters an infinite loop, enable the
        # following lines. It will print the names and parameters of all major functions
        # that are called, *before* they are called
        #from functools import reduce
        #print("%s%s %s%s" % (_debug_iter, reduce(lambda x, y: x + y, \
        #    map(lambda x: '-', range(1, 2 + _debug_iter))), f.__name__, args))

        r = f(*args, **kw)

        _debug_iter -= 1
        s = "%s%s = %s\n" % (f.__name__, args, r)
        if _debug_tmp != []:
            s += tree(_debug_tmp)
        _debug_tmp = oldtmp
        _debug_tmp.append(s)
        if _debug_iter == 0:
            print(_debug_tmp[0])
            _debug_tmp = []
        return r

    def decorated(*args, **kwargs):
        return maketree(func, *args, **kwargs)

    return decorated


def debug(*args):
    """
    Print ``*args`` if SYMPY_DEBUG is True, else do nothing.
    """
    from sympy import SYMPY_DEBUG
    if SYMPY_DEBUG:
        print(*args, file=sys.stderr)


def debugf(string, args):
    """
    Print ``string%args`` if SYMPY_DEBUG is True, else do nothing. This is
    intended for debug messages using formatted strings.
    """
    from sympy import SYMPY_DEBUG
    if SYMPY_DEBUG:
        print(string%args, file=sys.stderr)


def find_executable(executable, path=None):
    """Try to find 'executable' in the directories listed in 'path' (a
    string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH']).  Returns the complete filename or None if not
    found
    """
    from .exceptions import sympy_deprecation_warning
    sympy_deprecation_warning(
        """
        sympy.utilities.misc.find_executable() is deprecated. Use the standard
        library shutil.which() function instead.
        """,
        deprecated_since_version="1.7",
        active_deprecations_target="deprecated-find-executable",
    )
    if path is None:
        path = os.environ['PATH']
    paths = path.split(os.pathsep)
    extlist = ['']
    if os.name == 'os2':
        (base, ext) = os.path.splitext(executable)
        # executable files on OS/2 can have an arbitrary extension, but
        # .exe is automatically appended if no dot is present in the name
        if not ext:
            executable = executable + ".exe"
    elif sys.platform == 'win32':
        pathext = os.environ['PATHEXT'].lower().split(os.pathsep)
        (base, ext) = os.path.splitext(executable)
        if ext.lower() not in pathext:
            extlist = pathext
    for ext in extlist:
        execname = executable + ext
        if os.path.isfile(execname):
            return execname
        else:
            for p in paths:
                f = os.path.join(p, execname)
                if os.path.isfile(f):
                    return f

    return None


def func_name(x, short=False):
    """Return function name of `x` (if defined) else the `type(x)`.
    If short is True and there is a shorter alias for the result,
    return the alias.

    Examples
    ========

    >>> from sympy.utilities.misc import func_name
    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> func_name(Matrix.eye(3))
    'MutableDenseMatrix'
    >>> func_name(x < 1)
    'StrictLessThan'
    >>> func_name(x < 1, short=True)
    'Lt'
    """
    alias = {
    'GreaterThan': 'Ge',
    'StrictGreaterThan': 'Gt',
    'LessThan': 'Le',
    'StrictLessThan': 'Lt',
    'Equality': 'Eq',
    'Unequality': 'Ne',
    }
    typ = type(x)
    if str(typ).startswith("<type '"):
        typ = str(typ).split("'")[1].split("'")[0]
    elif str(typ).startswith("<class '"):
        typ = str(typ).split("'")[1].split("'")[0]
    rv = getattr(getattr(x, 'func', x), '__name__', typ)
    if '.' in rv:
        rv = rv.split('.')[-1]
    if short:
        rv = alias.get(rv, rv)
    return rv


def _replace(reps):
    """Return a function that can make the replacements, given in
    ``reps``, on a string. The replacements should be given as mapping.

    Examples
    ========

    >>> from sympy.utilities.misc import _replace
    >>> f = _replace(dict(foo='bar', d='t'))
    >>> f('food')
    'bart'
    >>> f = _replace({})
    >>> f('food')
    'food'
    """
    if not reps:
        return lambda x: x
    D = lambda match: reps[match.group(0)]
    pattern = _re.compile("|".join(
        [_re.escape(k) for k, v in reps.items()]), _re.MULTILINE)
    return lambda string: pattern.sub(D, string)


def replace(string, *reps):
    """Return ``string`` with all keys in ``reps`` replaced with
    their corresponding values, longer strings first, irrespective
    of the order they are given.  ``reps`` may be passed as tuples
    or a single mapping.

    Examples
    ========

    >>> from sympy.utilities.misc import replace
    >>> replace('foo', {'oo': 'ar', 'f': 'b'})
    'bar'
    >>> replace("spamham sha", ("spam", "eggs"), ("sha","md5"))
    'eggsham md5'

    There is no guarantee that a unique answer will be
    obtained if keys in a mapping overlap (i.e. are the same
    length and have some identical sequence at the
    beginning/end):

    >>> reps = [
    ...     ('ab', 'x'),
    ...     ('bc', 'y')]
    >>> replace('abc', *reps) in ('xc', 'ay')
    True

    References
    ==========

    .. [1] https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
    """
    if len(reps) == 1:
        kv = reps[0]
        if isinstance(kv, dict):
            reps = kv
        else:
            return string.replace(*kv)
    else:
        reps = dict(reps)
    return _replace(reps)(string)


def translate(s, a, b=None, c=None):
    """Return ``s`` where characters have been replaced or deleted.

    SYNTAX
    ======

    translate(s, None, deletechars):
        all characters in ``deletechars`` are deleted
    translate(s, map [,deletechars]):
        all characters in ``deletechars`` (if provided) are deleted
        then the replacements defined by map are made; if the keys
        of map are strings then the longer ones are handled first.
        Multicharacter deletions should have a value of ''.
    translate(s, oldchars, newchars, deletechars)
        all characters in ``deletechars`` are deleted
        then each character in ``oldchars`` is replaced with the
        corresponding character in ``newchars``

    Examples
    ========

    >>> from sympy.utilities.misc import translate
    >>> abc = 'abc'
    >>> translate(abc, None, 'a')
    'bc'
    >>> translate(abc, {'a': 'x'}, 'c')
    'xb'
    >>> translate(abc, {'abc': 'x', 'a': 'y'})
    'x'

    >>> translate('abcd', 'ac', 'AC', 'd')
    'AbC'

    There is no guarantee that a unique answer will be
    obtained if keys in a mapping overlap are the same
    length and have some identical sequences at the
    beginning/end:

    >>> translate(abc, {'ab': 'x', 'bc': 'y'}) in ('xc', 'ay')
    True
    """

    mr = {}
    if a is None:
        if c is not None:
            raise ValueError('c should be None when a=None is passed, instead got %s' % c)
        if b is None:
            return s
        c = b
        a = b = ''
    else:
        if isinstance(a, dict):
            short = {}
            for k in list(a.keys()):
                if len(k) == 1 and len(a[k]) == 1:
                    short[k] = a.pop(k)
            mr = a
            c = b
            if short:
                a, b = [''.join(i) for i in list(zip(*short.items()))]
            else:
                a = b = ''
        elif len(a) != len(b):
            raise ValueError('oldchars and newchars have different lengths')

    if c:
        val = str.maketrans('', '', c)
        s = s.translate(val)
    s = replace(s, mr)
    n = str.maketrans(a, b)
    return s.translate(n)


def ordinal(num):
    """Return ordinal number string of num, e.g. 1 becomes 1st.
    """
    # modified from https://codereview.stackexchange.com/questions/41298/producing-ordinal-numbers
    n = as_int(num)
    k = abs(n) % 100
    if 11 <= k <= 13:
        suffix = 'th'
    elif k % 10 == 1:
        suffix = 'st'
    elif k % 10 == 2:
        suffix = 'nd'
    elif k % 10 == 3:
        suffix = 'rd'
    else:
        suffix = 'th'
    return str(n) + suffix


def as_int(n, strict=True):
    """
    Convert the argument to a builtin integer.

    The return value is guaranteed to be equal to the input. ValueError is
    raised if the input has a non-integral value. When ``strict`` is True, this
    uses `__index__ <https://docs.python.org/3/reference/datamodel.html#object.__index__>`_
    and when it is False it uses ``int``.


    Examples
    ========

    >>> from sympy.utilities.misc import as_int
    >>> from sympy import sqrt, S

    The function is primarily concerned with sanitizing input for
    functions that need to work with builtin integers, so anything that
    is unambiguously an integer should be returned as an int:

    >>> as_int(S(3))
    3

    Floats, being of limited precision, are not assumed to be exact and
    will raise an error unless the ``strict`` flag is False. This
    precision issue becomes apparent for large floating point numbers:

    >>> big = 1e23
    >>> type(big) is float
    True
    >>> big == int(big)
    True
    >>> as_int(big)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    >>> as_int(big, strict=False)
    99999999999999991611392

    Input that might be a complex representation of an integer value is
    also rejected by default:

    >>> one = sqrt(3 + 2*sqrt(2)) - sqrt(2)
    >>> int(one) == 1
    True
    >>> as_int(one)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    """
    if strict:
        try:
            if isinstance(n, bool):
                raise TypeError
            return operator.index(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
    else:
        try:
            result = int(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
        if n - result:
            raise ValueError('%s is not an integer' % (n,))
        return result
<|endoftext|>"""This module contains some general purpose utilities that are used across
SymPy.
"""
from .iterables import (flatten, group, take, subsets,
    variations, numbered_symbols, cartes, capture, dict_merge,
    prefixes, postfixes, sift, topological_sort, unflatten,
    has_dups, has_variety, reshape, rotations)

from .misc import filldedent

from .lambdify import lambdify

from .decorator import threaded, xthreaded, public, memoize_property

from .timeutils import timed

__all__ = [
    'flatten', 'group', 'take', 'subsets', 'variations', 'numbered_symbols',
    'cartes', 'capture', 'dict_merge', 'prefixes', 'postfixes', 'sift',
    'topological_sort', 'unflatten', 'has_dups', 'has_variety', 'reshape',
    'rotations',

    'filldedent',

    'lambdify',

    'threaded', 'xthreaded', 'public', 'memoize_property',

    'timed',
]
<|endoftext|>"""
Algorithms and classes to support enumerative combinatorics.

Currently just multiset partitions, but more could be added.

Terminology (following Knuth, algorithm 7.1.2.5M TAOCP)
*multiset* aaabbcccc has a *partition* aaabc | bccc

The submultisets, aaabc and bccc of the partition are called
*parts*, or sometimes *vectors*.  (Knuth notes that multiset
partitions can be thought of as partitions of vectors of integers,
where the ith element of the vector gives the multiplicity of
element i.)

The values a, b and c are *components* of the multiset.  These
correspond to elements of a set, but in a multiset can be present
with a multiplicity greater than 1.

The algorithm deserves some explanation.

Think of the part aaabc from the multiset above.  If we impose an
ordering on the components of the multiset, we can represent a part
with a vector, in which the value of the first element of the vector
corresponds to the multiplicity of the first component in that
part. Thus, aaabc can be represented by the vector [3, 1, 1].  We
can also define an ordering on parts, based on the lexicographic
ordering of the vector (leftmost vector element, i.e., the element
with the smallest component number, is the most significant), so
that [3, 1, 1] > [3, 1, 0] and [3, 1, 1] > [2, 1, 4].  The ordering
on parts can be extended to an ordering on partitions: First, sort
the parts in each partition, left-to-right in decreasing order. Then
partition A is greater than partition B if A's leftmost/greatest
part is greater than B's leftmost part.  If the leftmost parts are
equal, compare the second parts, and so on.

In this ordering, the greatest partition of a given multiset has only
one part.  The least partition is the one in which the components
are spread out, one per part.

The enumeration algorithms in this file yield the partitions of the
argument multiset in decreasing order.  The main data structure is a
stack of parts, corresponding to the current partition.  An
important invariant is that the parts on the stack are themselves in
decreasing order.  This data structure is decremented to find the
next smaller partition.  Most often, decrementing the partition will
only involve adjustments to the smallest parts at the top of the
stack, much as adjacent integers *usually* differ only in their last
few digits.

Knuth's algorithm uses two main operations on parts:

Decrement - change the part so that it is smaller in the
  (vector) lexicographic order, but reduced by the smallest amount possible.
  For example, if the multiset has vector [5,
  3, 1], and the bottom/greatest part is [4, 2, 1], this part would
  decrement to [4, 2, 0], while [4, 0, 0] would decrement to [3, 3,
  1].  A singleton part is never decremented -- [1, 0, 0] is not
  decremented to [0, 3, 1].  Instead, the decrement operator needs
  to fail for this case.  In Knuth's pseudocode, the decrement
  operator is step m5.

Spread unallocated multiplicity - Once a part has been decremented,
  it cannot be the rightmost part in the partition.  There is some
  multiplicity that has not been allocated, and new parts must be
  created above it in the stack to use up this multiplicity.  To
  maintain the invariant that the parts on the stack are in
  decreasing order, these new parts must be less than or equal to
  the decremented part.
  For example, if the multiset is [5, 3, 1], and its most
  significant part has just been decremented to [5, 3, 0], the
  spread operation will add a new part so that the stack becomes
  [[5, 3, 0], [0, 0, 1]].  If the most significant part (for the
  same multiset) has been decremented to [2, 0, 0] the stack becomes
  [[2, 0, 0], [2, 0, 0], [1, 3, 1]].  In the pseudocode, the spread
  operation for one part is step m2.  The complete spread operation
  is a loop of steps m2 and m3.

In order to facilitate the spread operation, Knuth stores, for each
component of each part, not just the multiplicity of that component
in the part, but also the total multiplicity available for this
component in this part or any lesser part above it on the stack.

One added twist is that Knuth does not represent the part vectors as
arrays. Instead, he uses a sparse representation, in which a
component of a part is represented as a component number (c), plus
the multiplicity of the component in that part (v) as well as the
total multiplicity available for that component (u).  This saves
time that would be spent skipping over zeros.

"""

class PartComponent:
    """Internal class used in support of the multiset partitions
    enumerators and the associated visitor functions.

    Represents one component of one part of the current partition.

    A stack of these, plus an auxiliary frame array, f, represents a
    partition of the multiset.

    Knuth's pseudocode makes c, u, and v separate arrays.
    """

    __slots__ = ('c', 'u', 'v')

    def __init__(self):
        self.c = 0   # Component number
        self.u = 0   # The as yet unpartitioned amount in component c
                     # *before* it is allocated by this triple
        self.v = 0   # Amount of c component in the current part
                     # (v<=u).  An invariant of the representation is
                     # that the next higher triple for this component
                     # (if there is one) will have a value of u-v in
                     # its u attribute.

    def __repr__(self):
        "for debug/algorithm animation purposes"
        return 'c:%d u:%d v:%d' % (self.c, self.u, self.v)

    def __eq__(self, other):
        """Define  value oriented equality, which is useful for testers"""
        return (isinstance(other, self.__class__) and
                self.c == other.c and
                self.u == other.u and
                self.v == other.v)

    def __ne__(self, other):
        """Defined for consistency with __eq__"""
        return not self == other


# This function tries to be a faithful implementation of algorithm
# 7.1.2.5M in Volume 4A, Combinatoral Algorithms, Part 1, of The Art
# of Computer Programming, by Donald Knuth.  This includes using
# (mostly) the same variable names, etc.  This makes for rather
# low-level Python.

# Changes from Knuth's pseudocode include
# - use PartComponent struct/object instead of 3 arrays
# - make the function a generator
# - map (with some difficulty) the GOTOs to Python control structures.
# - Knuth uses 1-based numbering for components, this code is 0-based
# - renamed variable l to lpart.
# - flag variable x takes on values True/False instead of 1/0
#
def multiset_partitions_taocp(multiplicities):
    """Enumerates partitions of a multiset.

    Parameters
    ==========

    multiplicities
         list of integer multiplicities of the components of the multiset.

    Yields
    ======

    state
        Internal data structure which encodes a particular partition.
        This output is then usually processed by a visitor function
        which combines the information from this data structure with
        the components themselves to produce an actual partition.

        Unless they wish to create their own visitor function, users will
        have little need to look inside this data structure.  But, for
        reference, it is a 3-element list with components:

        f
            is a frame array, which is used to divide pstack into parts.

        lpart
            points to the base of the topmost part.

        pstack
            is an array of PartComponent objects.

        The ``state`` output offers a peek into the internal data
        structures of the enumeration function.  The client should
        treat this as read-only; any modification of the data
        structure will cause unpredictable (and almost certainly
        incorrect) results.  Also, the components of ``state`` are
        modified in place at each iteration.  Hence, the visitor must
        be called at each loop iteration.  Accumulating the ``state``
        instances and processing them later will not work.

    Examples
    ========

    >>> from sympy.utilities.enumerative import list_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> # variables components and multiplicities represent the multiset 'abb'
    >>> components = 'ab'
    >>> multiplicities = [1, 2]
    >>> states = multiset_partitions_taocp(multiplicities)
    >>> list(list_visitor(state, components) for state in states)
    [[['a', 'b', 'b']],
    [['a', 'b'], ['b']],
    [['a'], ['b', 'b']],
    [['a'], ['b'], ['b']]]

    See Also
    ========

    sympy.utilities.iterables.multiset_partitions: Takes a multiset
        as input and directly yields multiset partitions.  It
        dispatches to a number of functions, including this one, for
        implementation.  Most users will find it more convenient to
        use than multiset_partitions_taocp.

    """

    # Important variables.
    # m is the number of components, i.e., number of distinct elements
    m = len(multiplicities)
    # n is the cardinality, total number of elements whether or not distinct
    n = sum(multiplicities)

    # The main data structure, f segments pstack into parts.  See
    # list_visitor() for example code indicating how this internal
    # state corresponds to a partition.

    # Note: allocation of space for stack is conservative.  Knuth's
    # exercise 7.2.1.5.68 gives some indication of how to tighten this
    # bound, but this is not implemented.
    pstack = [PartComponent() for i in range(n * m + 1)]
    f = [0] * (n + 1)

    # Step M1 in Knuth (Initialize)
    # Initial state - entire multiset in one part.
    for j in range(m):
        ps = pstack[j]
        ps.c = j
        ps.u = multiplicities[j]
        ps.v = multiplicities[j]

    # Other variables
    f[0] = 0
    a = 0
    lpart = 0
    f[1] = m
    b = m  # in general, current stack frame is from a to b - 1

    while True:
        while True:
            # Step M2 (Subtract v from u)
            k = b
            x = False
            for j in range(a, b):
                pstack[k].u = pstack[j].u - pstack[j].v
                if pstack[k].u == 0:
                    x = True
                elif not x:
                    pstack[k].c = pstack[j].c
                    pstack[k].v = min(pstack[j].v, pstack[k].u)
                    x = pstack[k].u < pstack[j].v
                    k = k + 1
                else:  # x is True
                    pstack[k].c = pstack[j].c
                    pstack[k].v = pstack[k].u
                    k = k + 1
                # Note: x is True iff v has changed

            # Step M3 (Push if nonzero.)
            if k > b:
                a = b
                b = k
                lpart = lpart + 1
                f[lpart + 1] = b
                # Return to M2
            else:
                break  # Continue to M4

        # M4  Visit a partition
        state = [f, lpart, pstack]
        yield state

        # M5 (Decrease v)
        while True:
            j = b-1
            while (pstack[j].v == 0):
                j = j - 1
            if j == a and pstack[j].v == 1:
                # M6 (Backtrack)
                if lpart == 0:
                    return
                lpart = lpart - 1
                b = a
                a = f[lpart]
                # Return to M5
            else:
                pstack[j].v = pstack[j].v - 1
                for k in range(j + 1, b):
                    pstack[k].v = pstack[k].u
                break  # GOTO M2

# --------------- Visitor functions for multiset partitions ---------------
# A visitor takes the partition state generated by
# multiset_partitions_taocp or other enumerator, and produces useful
# output (such as the actual partition).


def factoring_visitor(state, primes):
    """Use with multiset_partitions_taocp to enumerate the ways a
    number can be expressed as a product of factors.  For this usage,
    the exponents of the prime factors of a number are arguments to
    the partition enumerator, while the corresponding prime factors
    are input here.

    Examples
    ========

    To enumerate the factorings of a number we can think of the elements of the
    partition as being the prime factors and the multiplicities as being their
    exponents.

    >>> from sympy.utilities.enumerative import factoring_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> from sympy import factorint
    >>> primes, multiplicities = zip(*factorint(24).items())
    >>> primes
    (2, 3)
    >>> multiplicities
    (3, 1)
    >>> states = multiset_partitions_taocp(multiplicities)
    >>> list(factoring_visitor(state, primes) for state in states)
    [[24], [8, 3], [12, 2], [4, 6], [4, 2, 3], [6, 2, 2], [2, 2, 2, 3]]
    """
    f, lpart, pstack = state
    factoring = []
    for i in range(lpart + 1):
        factor = 1
        for ps in pstack[f[i]: f[i + 1]]:
            if ps.v > 0:
                factor *= primes[ps.c] ** ps.v
        factoring.append(factor)
    return factoring


def list_visitor(state, components):
    """Return a list of lists to represent the partition.

    Examples
    ========

    >>> from sympy.utilities.enumerative import list_visitor
    >>> from sympy.utilities.enumerative import multiset_partitions_taocp
    >>> states = multiset_partitions_taocp([1, 2, 1])
    >>> s = next(states)
    >>> list_visitor(s, 'abc')  # for multiset 'a b b c'
    [['a', 'b', 'b', 'c']]
    >>> s = next(states)
    >>> list_visitor(s, [1, 2, 3])  # for multiset '1 2 2 3
    [[1, 2, 2], [3]]
    """
    f, lpart, pstack = state

    partition = []
    for i in range(lpart+1):
        part = []
        for ps in pstack[f[i]:f[i+1]]:
            if ps.v > 0:
                part.extend([components[ps.c]] * ps.v)
        partition.append(part)

    return partition


class MultisetPartitionTraverser():
    """
    Has methods to ``enumerate`` and ``count`` the partitions of a multiset.

    This implements a refactored and extended version of Knuth's algorithm
    7.1.2.5M [AOCP]_."

    The enumeration methods of this class are generators and return
    data structures which can be interpreted by the same visitor
    functions used for the output of ``multiset_partitions_taocp``.

    Examples
    ========

    >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
    >>> m = MultisetPartitionTraverser()
    >>> m.count_partitions([4,4,4,2])
    127750
    >>> m.count_partitions([3,3,3])
    686

    See Also
    ========

    multiset_partitions_taocp
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [AOCP] Algorithm 7.1.2.5M in Volume 4A, Combinatoral Algorithms,
           Part 1, of The Art of Computer Programming, by Donald Knuth.

    .. [Factorisatio] On a Problem of Oppenheim concerning
           "Factorisatio Numerorum" E. R. Canfield, Paul Erdos, Carl
           Pomerance, JOURNAL OF NUMBER THEORY, Vol. 17, No. 1. August
           1983.  See section 7 for a description of an algorithm
           similar to Knuth's.

    .. [Yorgey] Generating Multiset Partitions, Brent Yorgey, The
           Monad.Reader, Issue 8, September 2007.

    """

    def __init__(self):
        self.debug = False
        # TRACING variables.  These are useful for gathering
        # statistics on the algorithm itself, but have no particular
        # benefit to a user of the code.
        self.k1 = 0
        self.k2 = 0
        self.p1 = 0
        self.pstack = None
        self.f = None
        self.lpart = 0
        self.discarded = 0
        # dp_stack is list of lists of (part_key, start_count) pairs
        self.dp_stack = []

        # dp_map is map part_key-> count, where count represents the
        # number of multiset which are descendants of a part with this
        # key, **or any of its decrements**

        # Thus, when we find a part in the map, we add its count
        # value to the running total, cut off the enumeration, and
        # backtrack

        if not hasattr(self, 'dp_map'):
            self.dp_map = {}

    def db_trace(self, msg):
        """Useful for understanding/debugging the algorithms.  Not
        generally activated in end-user code."""
        if self.debug:
            # XXX: animation_visitor is undefined... Clearly this does not
            # work and was not tested. Previous code in comments below.
            raise RuntimeError
            #letters = 'abcdefghijklmnopqrstuvwxyz'
            #state = [self.f, self.lpart, self.pstack]
            #print("DBG:", msg,
            #      ["".join(part) for part in list_visitor(state, letters)],
            #      animation_visitor(state))

    #
    # Helper methods for enumeration
    #
    def _initialize_enumeration(self, multiplicities):
        """Allocates and initializes the partition stack.

        This is called from the enumeration/counting routines, so
        there is no need to call it separately."""

        num_components = len(multiplicities)
        # cardinality is the total number of elements, whether or not distinct
        cardinality = sum(multiplicities)

        # pstack is the partition stack, which is segmented by
        # f into parts.
        self.pstack = [PartComponent() for i in
                       range(num_components * cardinality + 1)]
        self.f = [0] * (cardinality + 1)

        # Initial state - entire multiset in one part.
        for j in range(num_components):
            ps = self.pstack[j]
            ps.c = j
            ps.u = multiplicities[j]
            ps.v = multiplicities[j]

        self.f[0] = 0
        self.f[1] = num_components
        self.lpart = 0

    # The decrement_part() method corresponds to step M5 in Knuth's
    # algorithm.  This is the base version for enum_all().  Modified
    # versions of this method are needed if we want to restrict
    # sizes of the partitions produced.
    def decrement_part(self, part):
        """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        If you think of the v values in the part as a multi-digit
        integer (least significant digit on the right) this is
        basically decrementing that integer, but with the extra
        constraint that the leftmost digit cannot be decremented to 0.

        Parameters
        ==========

        part
           The part, represented as a list of PartComponent objects,
           which is to be decremented.

        """
        plen = len(part)
        for j in range(plen - 1, -1, -1):
            if j == 0 and part[j].v > 1 or j > 0 and part[j].v > 0:
                # found val to decrement
                part[j].v -= 1
                # Reset trailing parts back to maximum
                for k in range(j + 1, plen):
                    part[k].v = part[k].u
                return True
        return False

    # Version to allow number of parts to be bounded from above.
    # Corresponds to (a modified) step M5.
    def decrement_part_small(self, part, ub):
        """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        ub
            the maximum number of parts allowed in a partition
            returned by the calling traversal.

        Notes
        =====

        The goal of this modification of the ordinary decrement method
        is to fail (meaning that the subtree rooted at this part is to
        be skipped) when it can be proved that this part can only have
        child partitions which are larger than allowed by ``ub``. If a
        decision is made to fail, it must be accurate, otherwise the
        enumeration will miss some partitions.  But, it is OK not to
        capture all the possible failures -- if a part is passed that
        should not be, the resulting too-large partitions are filtered
        by the enumeration one level up.  However, as is usual in
        constrained enumerations, failing early is advantageous.

        The tests used by this method catch the most common cases,
        although this implementation is by no means the last word on
        this problem.  The tests include:

        1) ``lpart`` must be less than ``ub`` by at least 2.  This is because
           once a part has been decremented, the partition
           will gain at least one child in the spread step.

        2) If the leading component of the part is about to be
           decremented, check for how many parts will be added in
           order to use up the unallocated multiplicity in that
           leading component, and fail if this number is greater than
           allowed by ``ub``.  (See code for the exact expression.)  This
           test is given in the answer to Knuth's problem 7.2.1.5.69.

        3) If there is *exactly* enough room to expand the leading
           component by the above test, check the next component (if
           it exists) once decrementing has finished.  If this has
           ``v == 0``, this next component will push the expansion over the
           limit by 1, so fail.
        """
        if self.lpart >= ub - 1:
            self.p1 += 1  # increment to keep track of usefulness of tests
            return False
        plen = len(part)
        for j in range(plen - 1, -1, -1):
            # Knuth's mod, (answer to problem 7.2.1.5.69)
            if j == 0 and (part[0].v - 1)*(ub - self.lpart) < part[0].u:
                self.k1 += 1
                return False

            if j == 0 and part[j].v > 1 or j > 0 and part[j].v > 0:
                # found val to decrement
                part[j].v -= 1
                # Reset trailing parts back to maximum
                for k in range(j + 1, plen):
                    part[k].v = part[k].u

                # Have now decremented part, but are we doomed to
                # failure when it is expanded?  Check one oddball case
                # that turns out to be surprisingly common - exactly
                # enough room to expand the leading component, but no
                # room for the second component, which has v=0.
                if (plen > 1 and part[1].v == 0 and
                    (part[0].u - part[0].v) ==
                        ((ub - self.lpart - 1) * part[0].v)):
                    self.k2 += 1
                    self.db_trace("Decrement fails test 3")
                    return False
                return True
        return False

    def decrement_part_large(self, part, amt, lb):
        """Decrements part, while respecting size constraint.

        A part can have no children which are of sufficient size (as
        indicated by ``lb``) unless that part has sufficient
        unallocated multiplicity.  When enforcing the size constraint,
        this method will decrement the part (if necessary) by an
        amount needed to ensure sufficient unallocated multiplicity.

        Returns True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        amt
            Can only take values 0 or 1.  A value of 1 means that the
            part must be decremented, and then the size constraint is
            enforced.  A value of 0 means just to enforce the ``lb``
            size constraint.

        lb
            The partitions produced by the calling enumeration must
            have more parts than this value.

        """

        if amt == 1:
            # In this case we always need to decrement, *before*
            # enforcing the "sufficient unallocated multiplicity"
            # constraint.  Easiest for this is just to call the
            # regular decrement method.
            if not self.decrement_part(part):
                return False

        # Next, perform any needed additional decrementing to respect
        # "sufficient unallocated multiplicity" (or fail if this is
        # not possible).
        min_unalloc = lb - self.lpart
        if min_unalloc <= 0:
            return True
        total_mult = sum(pc.u for pc in part)
        total_alloc = sum(pc.v for pc in part)
        if total_mult <= min_unalloc:
            return False

        deficit = min_unalloc - (total_mult - total_alloc)
        if deficit <= 0:
            return True

        for i in range(len(part) - 1, -1, -1):
            if i == 0:
                if part[0].v > deficit:
                    part[0].v -= deficit
                    return True
                else:
                    return False  # This shouldn't happen, due to above check
            else:
                if part[i].v >= deficit:
                    part[i].v -= deficit
                    return True
                else:
                    deficit -= part[i].v
                    part[i].v = 0

    def decrement_part_range(self, part, lb, ub):
        """Decrements part (a subrange of pstack), if possible, returning
        True iff the part was successfully decremented.

        Parameters
        ==========

        part
            part to be decremented (topmost part on the stack)

        ub
            the maximum number of parts allowed in a partition
            returned by the calling traversal.

        lb
            The partitions produced by the calling enumeration must
            have more parts than this value.

        Notes
        =====

        Combines the constraints of _small and _large decrement
        methods.  If returns success, part has been decremented at
        least once, but perhaps by quite a bit more if needed to meet
        the lb constraint.
        """

        # Constraint in the range case is just enforcing both the
        # constraints from _small and _large cases.  Note the 0 as the
        # second argument to the _large call -- this is the signal to
        # decrement only as needed to for constraint enforcement.  The
        # short circuiting and left-to-right order of the 'and'
        # operator is important for this to work correctly.
        return self.decrement_part_small(part, ub) and \
            self.decrement_part_large(part, 0, lb)

    def spread_part_multiplicity(self):
        """Returns True if a new part has been created, and
        adjusts pstack, f and lpart as needed.

        Notes
        =====

        Spreads unallocated multiplicity from the current top part
        into a new part created above the current on the stack.  This
        new part is constrained to be less than or equal to the old in
        terms of the part ordering.

        This call does nothing (and returns False) if the current top
        part has no unallocated multiplicity.

        """
        j = self.f[self.lpart]  # base of current top part
        k = self.f[self.lpart + 1]  # ub of current; potential base of next
        base = k  # save for later comparison

        changed = False  # Set to true when the new part (so far) is
                         # strictly less than (as opposed to less than
                         # or equal) to the old.
        for j in range(self.f[self.lpart], self.f[self.lpart + 1]):
            self.pstack[k].u = self.pstack[j].u - self.pstack[j].v
            if self.pstack[k].u == 0:
                changed = True
            else:
                self.pstack[k].c = self.pstack[j].c
                if changed:  # Put all available multiplicity in this part
                    self.pstack[k].v = self.pstack[k].u
                else:  # Still maintaining ordering constraint
                    if self.pstack[k].u < self.pstack[j].v:
                        self.pstack[k].v = self.pstack[k].u
                        changed = True
                    else:
                        self.pstack[k].v = self.pstack[j].v
                k = k + 1
        if k > base:
            # Adjust for the new part on stack
            self.lpart = self.lpart + 1
            self.f[self.lpart + 1] = k
            return True
        return False

    def top_part(self):
        """Return current top part on the stack, as a slice of pstack.

        """
        return self.pstack[self.f[self.lpart]:self.f[self.lpart + 1]]

    # Same interface and functionality as multiset_partitions_taocp(),
    # but some might find this refactored version easier to follow.
    def enum_all(self, multiplicities):
        """Enumerate the partitions of a multiset.

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_all([2,2])
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b', 'b']],
        [['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'a'], ['b'], ['b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']],
        [['a', 'b'], ['a'], ['b']],
        [['a'], ['a'], ['b', 'b']],
        [['a'], ['a'], ['b'], ['b']]]

        See Also
        ========

        multiset_partitions_taocp:
            which provides the same result as this method, but is
            about twice as fast.  Hence, enum_all is primarily useful
            for testing.  Also see the function for a discussion of
            states and visitors.

        """
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                pass

            # M4  Visit a partition
            state = [self.f, self.lpart, self.pstack]
            yield state

            # M5 (Decrease v)
            while not self.decrement_part(self.top_part()):
                # M6 (Backtrack)
                if self.lpart == 0:
                    return
                self.lpart -= 1

    def enum_small(self, multiplicities, ub):
        """Enumerate multiset partitions with no more than ``ub`` parts.

        Equivalent to enum_range(multiplicities, 0, ub)

        Parameters
        ==========

        multiplicities
             list of multiplicities of the components of the multiset.

        ub
            Maximum number of parts

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_small([2,2], 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b', 'b']],
        [['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']]]

        The implementation is based, in part, on the answer given to
        exercise 69, in Knuth [AOCP]_.

        See Also
        ========

        enum_all, enum_large, enum_range

        """

        # Keep track of iterations which do not yield a partition.
        # Clearly, we would like to keep this number small.
        self.discarded = 0
        if ub <= 0:
            return
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                self.db_trace('spread 1')
                if self.lpart >= ub:
                    self.discarded += 1
                    self.db_trace('  Discarding')
                    self.lpart = ub - 2
                    break
            else:
                # M4  Visit a partition
                state = [self.f, self.lpart, self.pstack]
                yield state

            # M5 (Decrease v)
            while not self.decrement_part_small(self.top_part(), ub):
                self.db_trace("Failed decrement, going to backtrack")
                # M6 (Backtrack)
                if self.lpart == 0:
                    return
                self.lpart -= 1
                self.db_trace("Backtracked to")
            self.db_trace("decrement ok, about to expand")

    def enum_large(self, multiplicities, lb):
        """Enumerate the partitions of a multiset with lb < num(parts)

        Equivalent to enum_range(multiplicities, lb, sum(multiplicities))

        Parameters
        ==========

        multiplicities
            list of multiplicities of the components of the multiset.

        lb
            Number of parts in the partition must be greater than
            this lower bound.


        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_large([2,2], 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a'], ['b'], ['b']],
        [['a', 'b'], ['a'], ['b']],
        [['a'], ['a'], ['b', 'b']],
        [['a'], ['a'], ['b'], ['b']]]

        See Also
        ========

        enum_all, enum_small, enum_range

        """
        self.discarded = 0
        if lb >= sum(multiplicities):
            return
        self._initialize_enumeration(multiplicities)
        self.decrement_part_large(self.top_part(), 0, lb)
        while True:
            good_partition = True
            while self.spread_part_multiplicity():
                if not self.decrement_part_large(self.top_part(), 0, lb):
                    # Failure here should be rare/impossible
                    self.discarded += 1
                    good_partition = False
                    break

            # M4  Visit a partition
            if good_partition:
                state = [self.f, self.lpart, self.pstack]
                yield state

            # M5 (Decrease v)
            while not self.decrement_part_large(self.top_part(), 1, lb):
                # M6 (Backtrack)
                if self.lpart == 0:
                    return
                self.lpart -= 1

    def enum_range(self, multiplicities, lb, ub):

        """Enumerate the partitions of a multiset with
        ``lb < num(parts) <= ub``.

        In particular, if partitions with exactly ``k`` parts are
        desired, call with ``(multiplicities, k - 1, k)``.  This
        method generalizes enum_all, enum_small, and enum_large.

        Examples
        ========

        >>> from sympy.utilities.enumerative import list_visitor
        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> states = m.enum_range([2,2], 1, 2)
        >>> list(list_visitor(state, 'ab') for state in states)
        [[['a', 'a', 'b'], ['b']],
        [['a', 'a'], ['b', 'b']],
        [['a', 'b', 'b'], ['a']],
        [['a', 'b'], ['a', 'b']]]

        """
        # combine the constraints of the _large and _small
        # enumerations.
        self.discarded = 0
        if ub <= 0 or lb >= sum(multiplicities):
            return
        self._initialize_enumeration(multiplicities)
        self.decrement_part_large(self.top_part(), 0, lb)
        while True:
            good_partition = True
            while self.spread_part_multiplicity():
                self.db_trace("spread 1")
                if not self.decrement_part_large(self.top_part(), 0, lb):
                    # Failure here - possible in range case?
                    self.db_trace("  Discarding (large cons)")
                    self.discarded += 1
                    good_partition = False
                    break
                elif self.lpart >= ub:
                    self.discarded += 1
                    good_partition = False
                    self.db_trace("  Discarding small cons")
                    self.lpart = ub - 2
                    break

            # M4  Visit a partition
            if good_partition:
                state = [self.f, self.lpart, self.pstack]
                yield state

            # M5 (Decrease v)
            while not self.decrement_part_range(self.top_part(), lb, ub):
                self.db_trace("Failed decrement, going to backtrack")
                # M6 (Backtrack)
                if self.lpart == 0:
                    return
                self.lpart -= 1
                self.db_trace("Backtracked to")
            self.db_trace("decrement ok, about to expand")

    def count_partitions_slow(self, multiplicities):
        """Returns the number of partitions of a multiset whose elements
        have the multiplicities given in ``multiplicities``.

        Primarily for comparison purposes.  It follows the same path as
        enumerate, and counts, rather than generates, the partitions.

        See Also
        ========

        count_partitions
            Has the same calling interface, but is much faster.

        """
        # number of partitions so far in the enumeration
        self.pcount = 0
        self._initialize_enumeration(multiplicities)
        while True:
            while self.spread_part_multiplicity():
                pass

            # M4  Visit (count) a partition
            self.pcount += 1

            # M5 (Decrease v)
            while not self.decrement_part(self.top_part()):
                # M6 (Backtrack)
                if self.lpart == 0:
                    return self.pcount
                self.lpart -= 1

    def count_partitions(self, multiplicities):
        """Returns the number of partitions of a multiset whose components
        have the multiplicities given in ``multiplicities``.

        For larger counts, this method is much faster than calling one
        of the enumerators and counting the result.  Uses dynamic
        programming to cut down on the number of nodes actually
        explored.  The dictionary used in order to accelerate the
        counting process is stored in the ``MultisetPartitionTraverser``
        object and persists across calls.  If the user does not
        expect to call ``count_partitions`` for any additional
        multisets, the object should be cleared to save memory.  On
        the other hand, the cache built up from one count run can
        significantly speed up subsequent calls to ``count_partitions``,
        so it may be advantageous not to clear the object.

        Examples
        ========

        >>> from sympy.utilities.enumerative import MultisetPartitionTraverser
        >>> m = MultisetPartitionTraverser()
        >>> m.count_partitions([9,8,2])
        288716
        >>> m.count_partitions([2,2])
        9
        >>> del m

        Notes
        =====

        If one looks at the workings of Knuth's algorithm M [AOCP]_, it
        can be viewed as a traversal of a binary tree of parts.  A
        part has (up to) two children, the left child resulting from
        the spread operation, and the right child from the decrement
        operation.  The ordinary enumeration of multiset partitions is
        an in-order traversal of this tree, and with the partitions
        corresponding to paths from the root to the leaves. The
        mapping from paths to partitions is a little complicated,
        since the partition would contain only those parts which are
        leaves or the parents of a spread link, not those which are
        parents of a decrement link.

        For counting purposes, it is sufficient to count leaves, and
        this can be done with a recursive in-order traversal.  The
        number of leaves of a subtree rooted at a particular part is a
        function only of that part itself, so memoizing has the
        potential to speed up the counting dramatically.

        This method follows a computational approach which is similar
        to the hypothetical memoized recursive function, but with two
        differences:

        1) This method is iterative, borrowing its structure from the
           other enumerations and maintaining an explicit stack of
           parts which are in the process of being counted.  (There
           may be multisets which can be counted reasonably quickly by
           this implementation, but which would overflow the default
           Python recursion limit with a recursive implementation.)

        2) Instead of using the part data structure directly, a more
           compact key is constructed.  This saves space, but more
           importantly coalesces some parts which would remain
           separate with physical keys.

        Unlike the enumeration functions, there is currently no _range
        version of count_partitions.  If someone wants to stretch
        their brain, it should be possible to construct one by
        memoizing with a histogram of counts rather than a single
        count, and combining the histograms.
        """
        # number of partitions so far in the enumeration
        self.pcount = 0

        # dp_stack is list of lists of (part_key, start_count) pairs
        self.dp_stack = []

        self._initialize_enumeration(multiplicities)
        pkey = part_key(self.top_part())
        self.dp_stack.append([(pkey, 0), ])
        while True:
            while self.spread_part_multiplicity():
                pkey = part_key(self.top_part())
                if pkey in self.dp_map:
                    # Already have a cached value for the count of the
                    # subtree rooted at this part.  Add it to the
                    # running counter, and break out of the spread
                    # loop.  The -1 below is to compensate for the
                    # leaf that this code path would otherwise find,
                    # and which gets incremented for below.

                    self.pcount += (self.dp_map[pkey] - 1)
                    self.lpart -= 1
                    break
                else:
                    self.dp_stack.append([(pkey, self.pcount), ])

            # M4  count a leaf partition
            self.pcount += 1

            # M5 (Decrease v)
            while not self.decrement_part(self.top_part()):
                # M6 (Backtrack)
                for key, oldcount in self.dp_stack.pop():
                    self.dp_map[key] = self.pcount - oldcount
                if self.lpart == 0:
                    return self.pcount
                self.lpart -= 1

            # At this point have successfully decremented the part on
            # the stack and it does not appear in the cache.  It needs
            # to be added to the list at the top of dp_stack
            pkey = part_key(self.top_part())
            self.dp_stack[-1].append((pkey, self.pcount),)


def part_key(part):
    """Helper for MultisetPartitionTraverser.count_partitions that
    creates a key for ``part``, that only includes information which can
    affect the count for that part.  (Any irrelevant information just
    reduces the effectiveness of dynamic programming.)

    Notes
    =====

    This member function is a candidate for future exploration. There
    are likely symmetries that can be exploited to coalesce some
    ``part_key`` values, and thereby save space and improve
    performance.

    """
    # The component number is irrelevant for counting partitions, so
    # leave it out of the memo key.
    rval = []
    for ps in part:
        rval.append(ps.u)
        rval.append(ps.v)
    return tuple(rval)
<|endoftext|># This module is deprecated and will be removed.

import sys
import os
from io import StringIO

from sympy.utilities.decorator import deprecated


@deprecated(
    """
    The sympy.utilities.pkgdata module and its get_resource function are
    deprecated. Use the stdlib importlib.resources module instead.
    """,
    deprecated_since_version="1.12",
    active_deprecations_target="pkgdata",
)
def get_resource(identifier, pkgname=__name__):

    mod = sys.modules[pkgname]
    fn = getattr(mod, '__file__', None)
    if fn is None:
        raise OSError("%r has no __file__!")
    path = os.path.join(os.path.dirname(fn), identifier)
    loader = getattr(mod, '__loader__', None)
    if loader is not None:
        try:
            data = loader.get_data(path)
        except (OSError, AttributeError):
            pass
        else:
            return StringIO(data.decode('utf-8'))
    return open(os.path.normpath(path), 'rb')
<|endoftext|>from functools import wraps


def recurrence_memo(initial):
    """
    Memo decorator for sequences defined by recurrence

    Examples
    ========

    >>> from sympy.utilities.memoization import recurrence_memo
    >>> @recurrence_memo([1]) # 0! = 1
    ... def factorial(n, prev):
    ...     return n * prev[-1]
    >>> factorial(4)
    24
    >>> factorial(3) # use cache values
    6
    >>> factorial.cache_length() # cache length can be obtained
    5
    >>> factorial.fetch_item(slice(2, 4))
    [2, 6]

    """
    cache = initial

    def decorator(f):
        @wraps(f)
        def g(n):
            L = len(cache)
            if n < L:
                return cache[n]
            for i in range(L, n + 1):
                cache.append(f(i, cache))
            return cache[-1]
        g.cache_length = lambda: len(cache)
        g.fetch_item = lambda x: cache[x]
        return g
    return decorator


def assoc_recurrence_memo(base_seq):
    """
    Memo decorator for associated sequences defined by recurrence starting from base

    base_seq(n) -- callable to get base sequence elements

    XXX works only for Pn0 = base_seq(0) cases
    XXX works only for m <= n cases
    """

    cache = []

    def decorator(f):
        @wraps(f)
        def g(n, m):
            L = len(cache)
            if n < L:
                return cache[n][m]

            for i in range(L, n + 1):
                # get base sequence
                F_i0 = base_seq(i)
                F_i_cache = [F_i0]
                cache.append(F_i_cache)

                # XXX only works for m <= n cases
                # generate assoc sequence
                for j in range(1, i + 1):
                    F_ij = f(i, j, cache)
                    F_i_cache.append(F_ij)

            return cache[n][m]

        return g
    return decorator
<|endoftext|>"""
General SymPy exceptions and warnings.
"""

import warnings
import contextlib

from textwrap import dedent


class SymPyDeprecationWarning(DeprecationWarning):
    r"""
    A warning for deprecated features of SymPy.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    Note that simply constructing this class will not cause a warning to be
    issued. To do that, you must call the :func`sympy_deprecation_warning`
    function. For this reason, it is not recommended to ever construct this
    class directly.

    Explanation
    ===========

    The ``SymPyDeprecationWarning`` class is a subclass of
    ``DeprecationWarning`` that is used for all deprecations in SymPy. A
    special subclass is used so that we can automatically augment the warning
    message with additional metadata about the version the deprecation was
    introduced in and a link to the documentation. This also allows users to
    explicitly filter deprecation warnings from SymPy using ``warnings``
    filters (see :ref:`silencing-sympy-deprecation-warnings`).

    Additionally, ``SymPyDeprecationWarning`` is enabled to be shown by
    default, unlike normal ``DeprecationWarning``\s, which are only shown by
    default in interactive sessions. This ensures that deprecation warnings in
    SymPy will actually be seen by users.

    See the documentation of :func:`sympy_deprecation_warning` for a
    description of the parameters to this function.

    To mark a function as deprecated, you can use the :func:`@deprecated
    <sympy.utilities.decorator.deprecated>` decorator.

    See Also
    ========
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.exceptions.ignore_warnings
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    """
    def __init__(self, message, *, deprecated_since_version, active_deprecations_target):

        super().__init__(message, deprecated_since_version,
                     active_deprecations_target)
        self.message = message
        if not isinstance(deprecated_since_version, str):
            raise TypeError(f"'deprecated_since_version' should be a string, got {deprecated_since_version!r}")
        self.deprecated_since_version = deprecated_since_version
        self.active_deprecations_target = active_deprecations_target
        if any(i in active_deprecations_target for i in '()='):
            raise ValueError("active_deprecations_target be the part inside of the '(...)='")

        self.full_message = f"""

{dedent(message).strip()}

See https://docs.sympy.org/latest/explanation/active-deprecations.html#{active_deprecations_target}
for details.

This has been deprecated since SymPy version {deprecated_since_version}. It
will be removed in a future version of SymPy.
"""

    def __str__(self):
        return self.full_message

    def __repr__(self):
        return f"{self.__class__.__name__}({self.message!r}, deprecated_since_version={self.deprecated_since_version!r}, active_deprecations_target={self.active_deprecations_target!r})"

    def __eq__(self, other):
        return isinstance(other, SymPyDeprecationWarning) and self.args == other.args

    # Make pickling work. The by default, it tries to recreate the expression
    # from its args, but this doesn't work because of our keyword-only
    # arguments.
    @classmethod
    def _new(cls, message, deprecated_since_version,
              active_deprecations_target):
        return cls(message, deprecated_since_version=deprecated_since_version, active_deprecations_target=active_deprecations_target)

    def __reduce__(self):
        return (self._new, (self.message, self.deprecated_since_version, self.active_deprecations_target))

# Python by default hides DeprecationWarnings, which we do not want.
warnings.simplefilter("once", SymPyDeprecationWarning)

def sympy_deprecation_warning(message, *, deprecated_since_version,
                              active_deprecations_target, stacklevel=3):
    r'''
    Warn that a feature is deprecated in SymPy.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    To mark an entire function or class as deprecated, you can use the
    :func:`@deprecated <sympy.utilities.decorator.deprecated>` decorator.

    Parameters
    ==========

    message : str
         The deprecation message. This may span multiple lines and contain
         code examples. Messages should be wrapped to 80 characters. The
         message is automatically dedented and leading and trailing whitespace
         stripped. Messages may include dynamic content based on the user
         input, but avoid using ``str(expression)`` if an expression can be
         arbitrary, as it might be huge and make the warning message
         unreadable.

    deprecated_since_version : str
         The version of SymPy the feature has been deprecated since. For new
         deprecations, this should be the version in `sympy/release.py
         <https://github.com/sympy/sympy/blob/master/sympy/release.py>`_
         without the ``.dev``. If the next SymPy version ends up being
         different from this, the release manager will need to update any
         ``SymPyDeprecationWarning``\s using the incorrect version. This
         argument is required and must be passed as a keyword argument.
         (example:  ``deprecated_since_version="1.10"``).

    active_deprecations_target : str
        The Sphinx target corresponding to the section for the deprecation in
        the :ref:`active-deprecations` document (see
        ``doc/src/explanation/active-deprecations.md``). This is used to
        automatically generate a URL to the page in the warning message. This
        argument is required and must be passed as a keyword argument.
        (example: ``active_deprecations_target="deprecated-feature-abc"``)

    stacklevel : int, default: 3
        The ``stacklevel`` parameter that is passed to ``warnings.warn``. If
        you create a wrapper that calls this function, this should be
        increased so that the warning message shows the user line of code that
        produced the warning. Note that in some cases there will be multiple
        possible different user code paths that could result in the warning.
        In that case, just choose the smallest common stacklevel.

    Examples
    ========

    >>> from sympy.utilities.exceptions import sympy_deprecation_warning
    >>> def is_this_zero(x, y=0):
    ...     """
    ...     Determine if x = 0.
    ...
    ...     Parameters
    ...     ==========
    ...
    ...     x : Expr
    ...       The expression to check.
    ...
    ...     y : Expr, optional
    ...       If provided, check if x = y.
    ...
    ...       .. deprecated:: 1.1
    ...
    ...          The ``y`` argument to ``is_this_zero`` is deprecated. Use
    ...          ``is_this_zero(x - y)`` instead.
    ...
    ...     """
    ...     from sympy import simplify
    ...
    ...     if y != 0:
    ...         sympy_deprecation_warning("""
    ...     The y argument to is_zero() is deprecated. Use is_zero(x - y) instead.""",
    ...             deprecated_since_version="1.1",
    ...             active_deprecations_target='is-this-zero-y-deprecation')
    ...     return simplify(x - y) == 0
    >>> is_this_zero(0)
    True
    >>> is_this_zero(1, 1) # doctest: +SKIP
    <stdin>:1: SymPyDeprecationWarning:
    <BLANKLINE>
    The y argument to is_zero() is deprecated. Use is_zero(x - y) instead.
    <BLANKLINE>
    See https://docs.sympy.org/latest/explanation/active-deprecations.html#is-this-zero-y-deprecation
    for details.
    <BLANKLINE>
    This has been deprecated since SymPy version 1.1. It
    will be removed in a future version of SymPy.
    <BLANKLINE>
      is_this_zero(1, 1)
    True

    See Also
    ========

    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.ignore_warnings
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    w = SymPyDeprecationWarning(message,
                            deprecated_since_version=deprecated_since_version,
                                active_deprecations_target=active_deprecations_target)
    warnings.warn(w, stacklevel=stacklevel)


@contextlib.contextmanager
def ignore_warnings(warningcls):
    '''
    Context manager to suppress warnings during tests.

    .. note::

       Do not use this with SymPyDeprecationWarning in the tests.
       warns_deprecated_sympy() should be used instead.

    This function is useful for suppressing warnings during tests. The warns
    function should be used to assert that a warning is raised. The
    ignore_warnings function is useful in situation when the warning is not
    guaranteed to be raised (e.g. on importing a module) or if the warning
    comes from third-party code.

    This function is also useful to prevent the same or similar warnings from
    being issue twice due to recursive calls.

    When the warning is coming (reliably) from SymPy the warns function should
    be preferred to ignore_warnings.

    >>> from sympy.utilities.exceptions import ignore_warnings
    >>> import warnings

    Here's a warning:

    >>> with warnings.catch_warnings():  # reset warnings in doctest
    ...     warnings.simplefilter('error')
    ...     warnings.warn('deprecated', UserWarning)
    Traceback (most recent call last):
      ...
    UserWarning: deprecated

    Let's suppress it with ignore_warnings:

    >>> with warnings.catch_warnings():  # reset warnings in doctest
    ...     warnings.simplefilter('error')
    ...     with ignore_warnings(UserWarning):
    ...         warnings.warn('deprecated', UserWarning)

    (No warning emitted)

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    # Absorbs all warnings in warnrec
    with warnings.catch_warnings(record=True) as warnrec:
        # Make sure our warning doesn't get filtered
        warnings.simplefilter("always", warningcls)
        # Now run the test
        yield

    # Reissue any warnings that we aren't testing for
    for w in warnrec:
        if not issubclass(w.category, warningcls):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
<|endoftext|>"""Module with some functions for MathML, like transforming MathML
content in MathML presentation.

To use this module, you will need lxml.
"""

from pathlib import Path

from sympy.utilities.decorator import doctest_depends_on


__doctest_requires__ = {('apply_xsl', 'c2p'): ['lxml']}


def add_mathml_headers(s):
    return """<math xmlns:mml="http://www.w3.org/1998/Math/MathML"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.w3.org/1998/Math/MathML
        http://www.w3.org/Math/XMLSchema/mathml2/mathml2.xsd">""" + s + "</math>"


def _read_binary(pkgname, filename):
    import sys

    if sys.version_info >= (3, 10):
        # files was added in Python 3.9 but only seems to work here in 3.10+
        from importlib.resources import files
        return files(pkgname).joinpath(filename).read_bytes()
    else:
        # read_binary was deprecated in Python 3.11
        from importlib.resources import read_binary
        return read_binary(pkgname, filename)


def _read_xsl(xsl):
    # Previously these values were allowed:
    if xsl == 'mathml/data/simple_mmlctop.xsl':
        xsl = 'simple_mmlctop.xsl'
    elif xsl == 'mathml/data/mmlctop.xsl':
        xsl = 'mmlctop.xsl'
    elif xsl == 'mathml/data/mmltex.xsl':
        xsl = 'mmltex.xsl'

    if xsl in ['simple_mmlctop.xsl', 'mmlctop.xsl', 'mmltex.xsl']:
        xslbytes = _read_binary('sympy.utilities.mathml.data', xsl)
    else:
        xslbytes = Path(xsl).read_bytes()

    return xslbytes


@doctest_depends_on(modules=('lxml',))
def apply_xsl(mml, xsl):
    """Apply a xsl to a MathML string.

    Parameters
    ==========

    mml
        A string with MathML code.
    xsl
        A string giving the name of an xsl (xml stylesheet) file which can be
        found in sympy/utilities/mathml/data. The following files are supplied
        with SymPy:

        - mmlctop.xsl
        - mmltex.xsl
        - simple_mmlctop.xsl

        Alternatively, a full path to an xsl file can be given.

    Examples
    ========

    >>> from sympy.utilities.mathml import apply_xsl
    >>> xsl = 'simple_mmlctop.xsl'
    >>> mml = '<apply> <plus/> <ci>a</ci> <ci>b</ci> </apply>'
    >>> res = apply_xsl(mml,xsl)
    >>> print(res)
    <?xml version="1.0"?>
    <mrow xmlns="http://www.w3.org/1998/Math/MathML">
      <mi>a</mi>
      <mo> + </mo>
      <mi>b</mi>
    </mrow>
    """
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False)
    ac = etree.XSLTAccessControl.DENY_ALL

    s = etree.XML(_read_xsl(xsl), parser=parser)
    transform = etree.XSLT(s, access_control=ac)
    doc = etree.XML(mml, parser=parser)
    result = transform(doc)
    s = str(result)
    return s


@doctest_depends_on(modules=('lxml',))
def c2p(mml, simple=False):
    """Transforms a document in MathML content (like the one that sympy produces)
    in one document in MathML presentation, more suitable for printing, and more
    widely accepted

    Examples
    ========

    >>> from sympy.utilities.mathml import c2p
    >>> mml = '<apply> <exp/> <cn>2</cn> </apply>'
    >>> c2p(mml,simple=True) != c2p(mml,simple=False)
    True

    """

    if not mml.startswith('<math'):
        mml = add_mathml_headers(mml)

    if simple:
        return apply_xsl(mml, 'mathml/data/simple_mmlctop.xsl')

    return apply_xsl(mml, 'mathml/data/mmlctop.xsl')
<|endoftext|>from __future__ import annotations
from typing import Callable, Optional

from collections import OrderedDict
import os
import re
import subprocess
import warnings

from .util import (
    find_binary_of_command, unique_list, CompileError
)


class CompilerRunner:
    """ CompilerRunner base class.

    Parameters
    ==========

    sources : list of str
        Paths to sources.
    out : str
    flags : iterable of str
        Compiler flags.
    run_linker : bool
    compiler_name_exe : (str, str) tuple
        Tuple of compiler name &  command to call.
    cwd : str
        Path of root of relative paths.
    include_dirs : list of str
        Include directories.
    libraries : list of str
        Libraries to link against.
    library_dirs : list of str
        Paths to search for shared libraries.
    std : str
        Standard string, e.g. ``'c++11'``, ``'c99'``, ``'f2003'``.
    define: iterable of strings
        macros to define
    undef : iterable of strings
        macros to undefine
    preferred_vendor : string
        name of preferred vendor e.g. 'gnu' or 'intel'

    Methods
    =======

    run():
        Invoke compilation as a subprocess.

    """

    environ_key_compiler: str  # e.g. 'CC', 'CXX', ...
    environ_key_flags: str  # e.g. 'CFLAGS', 'CXXFLAGS', ...
    environ_key_ldflags: str = "LDFLAGS"  # typically 'LDFLAGS'

    # Subclass to vendor/binary dict
    compiler_dict: dict[str, str]

    # Standards should be a tuple of supported standards
    # (first one will be the default)
    standards: tuple[None | str, ...]

    # Subclass to dict of binary/formater-callback
    std_formater: dict[str, Callable[[Optional[str]], str]]

    # subclass to be e.g. {'gcc': 'gnu', ...}
    compiler_name_vendor_mapping: dict[str, str]

    def __init__(self, sources, out, flags=None, run_linker=True, compiler=None, cwd='.',
                 include_dirs=None, libraries=None, library_dirs=None, std=None, define=None,
                 undef=None, strict_aliasing=None, preferred_vendor=None, linkline=None, **kwargs):
        if isinstance(sources, str):
            raise ValueError("Expected argument sources to be a list of strings.")
        self.sources = list(sources)
        self.out = out
        self.flags = flags or []
        if os.environ.get(self.environ_key_flags):
            self.flags += os.environ[self.environ_key_flags].split()
        self.cwd = cwd
        if compiler:
            self.compiler_name, self.compiler_binary = compiler
        elif os.environ.get(self.environ_key_compiler):
            self.compiler_binary = os.environ[self.environ_key_compiler]
            for k, v in self.compiler_dict.items():
                if k in self.compiler_binary:
                    self.compiler_vendor = k
                    self.compiler_name = v
                    break
            else:
                self.compiler_vendor, self.compiler_name = list(self.compiler_dict.items())[0]
                warnings.warn("failed to determine what kind of compiler %s is, assuming %s" %
                              (self.compiler_binary, self.compiler_name))
        else:
            # Find a compiler
            if preferred_vendor is None:
                preferred_vendor = os.environ.get('SYMPY_COMPILER_VENDOR', None)
            self.compiler_name, self.compiler_binary, self.compiler_vendor = self.find_compiler(preferred_vendor)
            if self.compiler_binary is None:
                raise ValueError("No compiler found (searched: {})".format(', '.join(self.compiler_dict.values())))
        self.define = define or []
        self.undef = undef or []
        self.include_dirs = include_dirs or []
        self.libraries = libraries or []
        self.library_dirs = library_dirs or []
        self.std = std or self.standards[0]
        self.run_linker = run_linker
        if self.run_linker:
            # both gnu and intel compilers use '-c' for disabling linker
            self.flags = list(filter(lambda x: x != '-c', self.flags))
        else:
            if '-c' not in self.flags:
                self.flags.append('-c')

        if self.std:
            self.flags.append(self.std_formater[
                self.compiler_name](self.std))

        self.linkline = (linkline or []) + [lf for lf in map(
            str.strip, os.environ.get(self.environ_key_ldflags, "").split()
        ) if lf != ""]

        if strict_aliasing is not None:
            nsa_re = re.compile("no-strict-aliasing$")
            sa_re = re.compile("strict-aliasing$")
            if strict_aliasing is True:
                if any(map(nsa_re.match, flags)):
                    raise CompileError("Strict aliasing cannot be both enforced and disabled")
                elif any(map(sa_re.match, flags)):
                    pass  # already enforced
                else:
                    flags.append('-fstrict-aliasing')
            elif strict_aliasing is False:
                if any(map(nsa_re.match, flags)):
                    pass  # already disabled
                else:
                    if any(map(sa_re.match, flags)):
                        raise CompileError("Strict aliasing cannot be both enforced and disabled")
                    else:
                        flags.append('-fno-strict-aliasing')
            else:
                msg = "Expected argument strict_aliasing to be True/False, got {}"
                raise ValueError(msg.format(strict_aliasing))

    @classmethod
    def find_compiler(cls, preferred_vendor=None):
        """ Identify a suitable C/fortran/other compiler. """
        candidates = list(cls.compiler_dict.keys())
        if preferred_vendor:
            if preferred_vendor in candidates:
                candidates = [preferred_vendor]+candidates
            else:
                raise ValueError("Unknown vendor {}".format(preferred_vendor))
        name, path = find_binary_of_command([cls.compiler_dict[x] for x in candidates])
        return name, path, cls.compiler_name_vendor_mapping[name]

    def cmd(self):
        """ List of arguments (str) to be passed to e.g. ``subprocess.Popen``. """
        cmd = (
            [self.compiler_binary] +
            self.flags +
            ['-U'+x for x in self.undef] +
            ['-D'+x for x in self.define] +
            ['-I'+x for x in self.include_dirs] +
            self.sources
        )
        if self.run_linker:
            cmd += (['-L'+x for x in self.library_dirs] +
                    ['-l'+x for x in self.libraries] +
                    self.linkline)
        counted = []
        for envvar in re.findall(r'\$\{(\w+)\}', ' '.join(cmd)):
            if os.getenv(envvar) is None:
                if envvar not in counted:
                    counted.append(envvar)
                    msg = "Environment variable '{}' undefined.".format(envvar)
                    raise CompileError(msg)
        return cmd

    def run(self):
        self.flags = unique_list(self.flags)

        # Append output flag and name to tail of flags
        self.flags.extend(['-o', self.out])
        env = os.environ.copy()
        env['PWD'] = self.cwd

        # NOTE: intel compilers seems to need shell=True
        p = subprocess.Popen(' '.join(self.cmd()),
                             shell=True,
                             cwd=self.cwd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             env=env)
        comm = p.communicate()
        try:
            self.cmd_outerr = comm[0].decode('utf-8')
        except UnicodeDecodeError:
            self.cmd_outerr = comm[0].decode('iso-8859-1')  # win32
        self.cmd_returncode = p.returncode

        # Error handling
        if self.cmd_returncode != 0:
            msg = "Error executing '{}' in {} (exited status {}):\n {}\n".format(
                ' '.join(self.cmd()), self.cwd, str(self.cmd_returncode), self.cmd_outerr
            )
            raise CompileError(msg)

        return self.cmd_outerr, self.cmd_returncode


class CCompilerRunner(CompilerRunner):

    environ_key_compiler = 'CC'
    environ_key_flags = 'CFLAGS'

    compiler_dict = OrderedDict([
        ('gnu', 'gcc'),
        ('intel', 'icc'),
        ('llvm', 'clang'),
    ])

    standards = ('c89', 'c90', 'c99', 'c11')  # First is default

    std_formater = {
        'gcc': '-std={}'.format,
        'icc': '-std={}'.format,
        'clang': '-std={}'.format,
    }

    compiler_name_vendor_mapping = {
        'gcc': 'gnu',
        'icc': 'intel',
        'clang': 'llvm'
    }


def _mk_flag_filter(cmplr_name):  # helper for class initialization
    not_welcome = {'g++': ("Wimplicit-interface",)}  # "Wstrict-prototypes",)}
    if cmplr_name in not_welcome:
        def fltr(x):
            for nw in not_welcome[cmplr_name]:
                if nw in x:
                    return False
            return True
    else:
        def fltr(x):
            return True
    return fltr


class CppCompilerRunner(CompilerRunner):

    environ_key_compiler = 'CXX'
    environ_key_flags = 'CXXFLAGS'

    compiler_dict = OrderedDict([
        ('gnu', 'g++'),
        ('intel', 'icpc'),
        ('llvm', 'clang++'),
    ])

    # First is the default, c++0x == c++11
    standards = ('c++98', 'c++0x')

    std_formater = {
        'g++': '-std={}'.format,
        'icpc': '-std={}'.format,
        'clang++': '-std={}'.format,
    }

    compiler_name_vendor_mapping = {
        'g++': 'gnu',
        'icpc': 'intel',
        'clang++': 'llvm'
    }


class FortranCompilerRunner(CompilerRunner):

    environ_key_compiler = 'FC'
    environ_key_flags = 'FFLAGS'

    standards = (None, 'f77', 'f95', 'f2003', 'f2008')

    std_formater = {
        'gfortran': lambda x: '-std=gnu' if x is None else '-std=legacy' if x == 'f77' else '-std={}'.format(x),
        'ifort': lambda x: '-stand f08' if x is None else '-stand f{}'.format(x[-2:]),  # f2008 => f08
    }

    compiler_dict = OrderedDict([
        ('gnu', 'gfortran'),
        ('intel', 'ifort'),
    ])

    compiler_name_vendor_mapping = {
        'gfortran': 'gnu',
        'ifort': 'intel',
    }
<|endoftext|>from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch

from sympy.testing.pytest import XFAIL


def may_xfail(func):
    if sys.platform.lower() == 'darwin' or os.name == 'nt':
        # sympy.utilities._compilation needs more testing on Windows and macOS
        # once those two platforms are reliably supported this xfail decorator
        # may be removed.
        return XFAIL(func)
    else:
        return func


class CompilerNotFoundError(FileNotFoundError):
    pass


class CompileError (Exception):
    """Failure to compile one or more C/C++ source files."""


def get_abspath(path, cwd='.'):
    """ Returns the absolute path.

    Parameters
    ==========

    path : str
        (relative) path.
    cwd : str
        Path to root of relative path.
    """
    if os.path.isabs(path):
        return path
    else:
        if not os.path.isabs(cwd):
            cwd = os.path.abspath(cwd)
        return os.path.abspath(
            os.path.join(cwd, path)
        )


def make_dirs(path):
    """ Create directories (equivalent of ``mkdir -p``). """
    if path[-1] == '/':
        parent = os.path.dirname(path[:-1])
    else:
        parent = os.path.dirname(path)

    if len(parent) > 0:
        if not os.path.exists(parent):
            make_dirs(parent)

    if not os.path.exists(path):
        os.mkdir(path, 0o777)
    else:
        assert os.path.isdir(path)

def missing_or_other_newer(path, other_path, cwd=None):
    """
    Investigate if path is non-existent or older than provided reference
    path.

    Parameters
    ==========
    path: string
        path to path which might be missing or too old
    other_path: string
        reference path
    cwd: string
        working directory (root of relative paths)

    Returns
    =======
    True if path is older or missing.
    """
    cwd = cwd or '.'
    path = get_abspath(path, cwd=cwd)
    other_path = get_abspath(other_path, cwd=cwd)
    if not os.path.exists(path):
        return True
    if os.path.getmtime(other_path) - 1e-6 >= os.path.getmtime(path):
        # 1e-6 is needed because http://stackoverflow.com/questions/17086426/
        return True
    return False

def copy(src, dst, only_update=False, copystat=True, cwd=None,
         dest_is_dir=False, create_dest_dirs=False):
    """ Variation of ``shutil.copy`` with extra options.

    Parameters
    ==========

    src : str
        Path to source file.
    dst : str
        Path to destination.
    only_update : bool
        Only copy if source is newer than destination
        (returns None if it was newer), default: ``False``.
    copystat : bool
        See ``shutil.copystat``. default: ``True``.
    cwd : str
        Path to working directory (root of relative paths).
    dest_is_dir : bool
        Ensures that dst is treated as a directory. default: ``False``
    create_dest_dirs : bool
        Creates directories if needed.

    Returns
    =======

    Path to the copied file.

    """
    if cwd:  # Handle working directory
        if not os.path.isabs(src):
            src = os.path.join(cwd, src)
        if not os.path.isabs(dst):
            dst = os.path.join(cwd, dst)

    if not os.path.exists(src):  # Make sure source file exists
        raise FileNotFoundError("Source: `{}` does not exist".format(src))

    # We accept both (re)naming destination file _or_
    # passing a (possible non-existent) destination directory
    if dest_is_dir:
        if not dst[-1] == '/':
