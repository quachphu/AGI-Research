    print("x", file=sys.stderr)

    """
    __slots__ = _fields = ('name',)
    _construct_name = String

stdout = Stream('stdout')
stderr = Stream('stderr')


class Print(Token):
    r""" Represents print command in the code.

    Parameters
    ==========

    formatstring : str
    *args : Basic instances (or convertible to such through sympify)

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy import pycode
    >>> print(pycode(Print('x y'.split(), "coordinate: %12.5g %12.5g\\n")))
    print("coordinate: %12.5g %12.5g\n" % (x, y), end="")

    """

    __slots__ = _fields = ('print_args', 'format_string', 'file')
    defaults = {'format_string': none, 'file': none}

    _construct_print_args = staticmethod(_mk_Tuple)
    _construct_format_string = QuotedString
    _construct_file = Stream


class FunctionPrototype(Node):
    """ Represents a function prototype

    Allows the user to generate forward declaration in e.g. C/C++.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'

    """

    __slots__ = ('return_type', 'name', 'parameters')
    _fields: tuple[str, ...] = __slots__ + Node._fields

    _construct_return_type = Type
    _construct_name = String

    @staticmethod
    def _construct_parameters(args):
        def _var(arg):
            if isinstance(arg, Declaration):
                return arg.variable
            elif isinstance(arg, Variable):
                return arg
            else:
                return Variable.deduced(arg)
        return Tuple(*map(_var, args))

    @classmethod
    def from_FunctionDefinition(cls, func_def):
        if not isinstance(func_def, FunctionDefinition):
            raise TypeError("func_def is not an instance of FunctionDefinition")
        return cls(**func_def.kwargs(exclude=('body',)))


class FunctionDefinition(FunctionPrototype):
    """ Represents a function definition in the code.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    body : CodeBlock or iterable
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'
    >>> from sympy.codegen.ast import FunctionDefinition, Return
    >>> body = [Return(x*y)]
    >>> fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    >>> print(ccode(fd))
    double foo(double x, double y){
        return x*y;
    }
    """

    __slots__ = ('body', )
    _fields = FunctionPrototype._fields[:-1] + __slots__ + Node._fields

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def from_FunctionPrototype(cls, func_proto, body):
        if not isinstance(func_proto, FunctionPrototype):
            raise TypeError("func_proto is not an instance of FunctionPrototype")
        return cls(body=body, **func_proto.kwargs())


class Return(Token):
    """ Represents a return command in the code.

    Parameters
    ==========

    return : Basic

    Examples
    ========

    >>> from sympy.codegen.ast import Return
    >>> from sympy.printing.pycode import pycode
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> print(pycode(Return(x)))
    return x

    """
    __slots__ = _fields = ('return',)
    _construct_return=staticmethod(_sympify)


class FunctionCall(Token, Expr):
    """ Represents a call to a function in the code.

    Parameters
    ==========

    name : str
    function_args : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import FunctionCall
    >>> from sympy import pycode
    >>> fcall = FunctionCall('foo', 'bar baz'.split())
    >>> print(pycode(fcall))
    foo(bar, baz)

    """
    __slots__ = _fields = ('name', 'function_args')

    _construct_name = String
    _construct_function_args = staticmethod(lambda args: Tuple(*args))


class Raise(Token):
    """ Prints as 'raise ...' in Python, 'throw ...' in C++"""
    __slots__ = _fields = ('exception',)


class RuntimeError_(Token):
    """ Represents 'std::runtime_error' in C++ and 'RuntimeError' in Python.

    Note that the latter is uncommon, and you might want to use e.g. ValueError.
    """
    __slots__ = _fields = ('message',)
    _construct_message = String
<|endoftext|>from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import cos, sin


def _cosm1(x, *, evaluate=True):
    return Add(cos(x, evaluate=evaluate), -S.One, evaluate=evaluate)


class cosm1(Function):
    """ Minus one plus cosine of x, i.e. cos(x) - 1. For use when x is close to zero.

    Helper class for use with e.g. scipy.special.cosm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cosm1.html
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return -sin(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_cos(self, x, **kwargs):
        return _cosm1(x)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(cos).evalf(*args, **kwargs)

    def _eval_simplify(self, **kwargs):
        x, = self.args
        candidate = _cosm1(x.simplify(**kwargs))
        if candidate != _cosm1(x, evaluate=False):
            return candidate
        else:
            return cosm1(x)


def _powm1(x, y, *, evaluate=True):
    return Add(Pow(x, y, evaluate=evaluate), -S.One, evaluate=evaluate)


class powm1(Function):
    """ Minus one plus x to the power of y, i.e. x**y - 1. For use when x is close to one or y is close to zero.

    Helper class for use with e.g. scipy.special.powm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.powm1.html
    """
    nargs = 2

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return Pow(self.args[0], self.args[1])*self.args[1]/self.args[0]
        elif argindex == 2:
            return log(self.args[0])*Pow(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Pow(self, x, y, **kwargs):
        return _powm1(x, y)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(Pow).evalf(*args, **kwargs)

    def _eval_simplify(self, **kwargs):
        x, y = self.args
        candidate = _powm1(x.simplify(**kwargs), y.simplify(**kwargs))
        if candidate != _powm1(x, y, evaluate=False):
            return candidate
        else:
            return powm1(x, y)
<|endoftext|>"""
AST nodes specific to the C family of languages
"""

from sympy.codegen.ast import (
    Attribute, Declaration, Node, String, Token, Type, none,
    FunctionCall, CodeBlock
    )
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify

void = Type('void')

restrict = Attribute('restrict')  # guarantees no pointer aliasing
volatile = Attribute('volatile')
static = Attribute('static')


def alignof(arg):
    """ Generate of FunctionCall instance for calling 'alignof' """
    return FunctionCall('alignof', [String(arg) if isinstance(arg, str) else arg])


def sizeof(arg):
    """ Generate of FunctionCall instance for calling 'sizeof'

    Examples
    ========

    >>> from sympy.codegen.ast import real
    >>> from sympy.codegen.cnodes import sizeof
    >>> from sympy import ccode
    >>> ccode(sizeof(real))
    'sizeof(double)'
    """
    return FunctionCall('sizeof', [String(arg) if isinstance(arg, str) else arg])


class CommaOperator(Basic):
    """ Represents the comma operator in C """
    def __new__(cls, *args):
        return Basic.__new__(cls, *[sympify(arg) for arg in args])


class Label(Node):
    """ Label for use with e.g. goto statement.

    Examples
    ========

    >>> from sympy import ccode, Symbol
    >>> from sympy.codegen.cnodes import Label, PreIncrement
    >>> print(ccode(Label('foo')))
    foo:
    >>> print(ccode(Label('bar', [PreIncrement(Symbol('a'))])))
    bar:
    ++(a);

    """
    __slots__ = _fields = ('name', 'body')
    defaults = {'body': none}
    _construct_name = String

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class goto(Token):
    """ Represents goto in C """
    __slots__ = _fields = ('label',)
    _construct_label = Label


class PreDecrement(Basic):
    """ Represents the pre-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreDecrement
    >>> from sympy import ccode
    >>> ccode(PreDecrement(x))
    '--(x)'

    """
    nargs = 1


class PostDecrement(Basic):
    """ Represents the post-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PostDecrement
    >>> from sympy import ccode
    >>> ccode(PostDecrement(x))
    '(x)--'

    """
    nargs = 1


class PreIncrement(Basic):
    """ Represents the pre-increment operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreIncrement
    >>> from sympy import ccode
    >>> ccode(PreIncrement(x))
    '++(x)'

    """
    nargs = 1


class PostIncrement(Basic):
    """ Represents the post-increment operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PostIncrement
    >>> from sympy import ccode
    >>> ccode(PostIncrement(x))
    '(x)++'

    """
    nargs = 1


class struct(Node):
    """ Represents a struct in C """
    __slots__ = _fields = ('name', 'declarations')
    defaults = {'name': none}
    _construct_name = String

    @classmethod
    def _construct_declarations(cls, args):
        return Tuple(*[Declaration(arg) for arg in args])


class union(struct):
    """ Represents a union in C """
    __slots__ = ()
<|endoftext|>from itertools import chain
from sympy.codegen.fnodes import Module
from sympy.core.symbol import Dummy
from sympy.printing.fortran import FCodePrinter

""" This module collects utilities for rendering Fortran code. """


def render_as_module(definitions, name, declarations=(), printer_settings=None):
    """ Creates a ``Module`` instance and renders it as a string.

    This generates Fortran source code for a module with the correct ``use`` statements.

    Parameters
    ==========

    definitions : iterable
        Passed to :class:`sympy.codegen.fnodes.Module`.
    name : str
        Passed to :class:`sympy.codegen.fnodes.Module`.
    declarations : iterable
        Passed to :class:`sympy.codegen.fnodes.Module`. It will be extended with
        use statements, 'implicit none' and public list generated from ``definitions``.
    printer_settings : dict
        Passed to ``FCodePrinter`` (default: ``{'standard': 2003, 'source_format': 'free'}``).

    """
    printer_settings = printer_settings or {'standard': 2003, 'source_format': 'free'}
    printer = FCodePrinter(printer_settings)
    dummy = Dummy()
    if isinstance(definitions, Module):
        raise ValueError("This function expects to construct a module on its own.")
    mod = Module(name, chain(declarations, [dummy]), definitions)
    fstr = printer.doprint(mod)
    module_use_str = '   %s\n' % '   \n'.join(['use %s, only: %s' % (k, ', '.join(v)) for
                                                k, v in printer.module_uses.items()])
    module_use_str += '   implicit none\n'
    module_use_str += '   private\n'
    module_use_str += '   public %s\n' % ', '.join([str(node.name) for node in definitions if getattr(node, 'name', None)])
    return fstr.replace(printer.doprint(dummy), module_use_str)
<|endoftext|>import math
from sympy.sets.sets import Interval
from sympy.calculus.singularities import is_increasing, is_decreasing
from sympy.codegen.rewriting import Optimization
from sympy.core.function import UndefinedFunction

"""
This module collects classes useful for approximate rewriting of expressions.
This can be beneficial when generating numeric code for which performance is
of greater importance than precision (e.g. for preconditioners used in iterative
methods).
"""

class SumApprox(Optimization):
    """
    Approximates sum by neglecting small terms.

    Explanation
    ===========

    If terms are expressions which can be determined to be monotonic, then
    bounds for those expressions are added.

    Parameters
    ==========

    bounds : dict
        Mapping expressions to length 2 tuple of bounds (low, high).
    reltol : number
        Threshold for when to ignore a term. Taken relative to the largest
        lower bound among bounds.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x, y, z
    >>> from sympy.codegen.rewriting import optimize
    >>> from sympy.codegen.approximations import SumApprox
    >>> bounds = {x: (-1, 1), y: (1000, 2000), z: (-10, 3)}
    >>> sum_approx3 = SumApprox(bounds, reltol=1e-3)
    >>> sum_approx2 = SumApprox(bounds, reltol=1e-2)
    >>> sum_approx1 = SumApprox(bounds, reltol=1e-1)
    >>> expr = 3*(x + y + exp(z))
    >>> optimize(expr, [sum_approx3])
    3*(x + y + exp(z))
    >>> optimize(expr, [sum_approx2])
    3*y + 3*exp(z)
    >>> optimize(expr, [sum_approx1])
    3*y

    """

    def __init__(self, bounds, reltol, **kwargs):
        super().__init__(**kwargs)
        self.bounds = bounds
        self.reltol = reltol

    def __call__(self, expr):
        return expr.factor().replace(self.query, lambda arg: self.value(arg))

    def query(self, expr):
        return expr.is_Add

    def value(self, add):
        for term in add.args:
            if term.is_number or term in self.bounds or len(term.free_symbols) != 1:
                continue
            fs, = term.free_symbols
            if fs not in self.bounds:
                continue
            intrvl = Interval(*self.bounds[fs])
            if is_increasing(term, intrvl, fs):
                self.bounds[term] = (
                    term.subs({fs: self.bounds[fs][0]}),
                    term.subs({fs: self.bounds[fs][1]})
                )
            elif is_decreasing(term, intrvl, fs):
                self.bounds[term] = (
                    term.subs({fs: self.bounds[fs][1]}),
                    term.subs({fs: self.bounds[fs][0]})
                )
            else:
                return add

        if all(term.is_number or term in self.bounds for term in add.args):
            bounds = [(term, term) if term.is_number else self.bounds[term] for term in add.args]
            largest_abs_guarantee = 0
            for lo, hi in bounds:
                if lo <= 0 <= hi:
                    continue
                largest_abs_guarantee = max(largest_abs_guarantee,
                                            min(abs(lo), abs(hi)))
            new_terms = []
            for term, (lo, hi) in zip(add.args, bounds):
                if max(abs(lo), abs(hi)) >= largest_abs_guarantee*self.reltol:
                    new_terms.append(term)
            return add.func(*new_terms)
        else:
            return add


class SeriesApprox(Optimization):
    """ Approximates functions by expanding them as a series.

    Parameters
    ==========

    bounds : dict
        Mapping expressions to length 2 tuple of bounds (low, high).
    reltol : number
        Threshold for when to ignore a term. Taken relative to the largest
        lower bound among bounds.
    max_order : int
        Largest order to include in series expansion
    n_point_checks : int (even)
        The validity of an expansion (with respect to reltol) is checked at
        discrete points (linearly spaced over the bounds of the variable). The
        number of points used in this numerical check is given by this number.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x, y
    >>> from sympy.codegen.rewriting import optimize
    >>> from sympy.codegen.approximations import SeriesApprox
    >>> bounds = {x: (-.1, .1), y: (pi-1, pi+1)}
    >>> series_approx2 = SeriesApprox(bounds, reltol=1e-2)
    >>> series_approx3 = SeriesApprox(bounds, reltol=1e-3)
    >>> series_approx8 = SeriesApprox(bounds, reltol=1e-8)
    >>> expr = sin(x)*sin(y)
    >>> optimize(expr, [series_approx2])
    x*(-y + (y - pi)**3/6 + pi)
    >>> optimize(expr, [series_approx3])
    (-x**3/6 + x)*sin(y)
    >>> optimize(expr, [series_approx8])
    sin(x)*sin(y)

    """
    def __init__(self, bounds, reltol, max_order=4, n_point_checks=4, **kwargs):
        super().__init__(**kwargs)
        self.bounds = bounds
        self.reltol = reltol
        self.max_order = max_order
        if n_point_checks % 2 == 1:
            raise ValueError("Checking the solution at expansion point is not helpful")
        self.n_point_checks = n_point_checks
        self._prec = math.ceil(-math.log10(self.reltol))

    def __call__(self, expr):
        return expr.factor().replace(self.query, lambda arg: self.value(arg))

    def query(self, expr):
        return (expr.is_Function and not isinstance(expr, UndefinedFunction)
                and len(expr.args) == 1)

    def value(self, fexpr):
        free_symbols = fexpr.free_symbols
        if len(free_symbols) != 1:
            return fexpr
        symb, = free_symbols
        if symb not in self.bounds:
            return fexpr
        lo, hi = self.bounds[symb]
        x0 = (lo + hi)/2
        cheapest = None
        for n in range(self.max_order+1, 0, -1):
            fseri = fexpr.series(symb, x0=x0, n=n).removeO()
            n_ok = True
            for idx in range(self.n_point_checks):
                x = lo + idx*(hi - lo)/(self.n_point_checks - 1)
                val = fseri.xreplace({symb: x})
                ref = fexpr.xreplace({symb: x})
                if abs((1 - val/ref).evalf(self._prec)) > self.reltol:
                    n_ok = False
                    break

            if n_ok:
                cheapest = fseri
            else:
                break

        if cheapest is None:
            return fexpr
        else:
            return cheapest
<|endoftext|>from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from .ast import Token, none


def _logaddexp(x1, x2, *, evaluate=True):
    return log(Add(exp(x1, evaluate=evaluate), exp(x2, evaluate=evaluate), evaluate=evaluate))


_two = S.One*2
_ln2 = log(_two)


def _lb(x, *, evaluate=True):
    return log(x, evaluate=evaluate)/_ln2


def _exp2(x, *, evaluate=True):
    return Pow(_two, x, evaluate=evaluate)


def _logaddexp2(x1, x2, *, evaluate=True):
    return _lb(Add(_exp2(x1, evaluate=evaluate),
                   _exp2(x2, evaluate=evaluate), evaluate=evaluate))


class logaddexp(Function):
    """ Logarithm of the sum of exponentiations of the inputs.

    Helper class for use with e.g. numpy.logaddexp

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    """
    nargs = 2

    def __new__(cls, *args):
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            wrt, other = self.args
        elif argindex == 2:
            other, wrt = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One/(S.One + exp(other-wrt))

    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        return _logaddexp(x1, x2)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        a, b = (x.simplify(**kwargs) for x in self.args)
        candidate = _logaddexp(a, b)
        if candidate != _logaddexp(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp(a, b)


class logaddexp2(Function):
    """ Logarithm of the sum of exponentiations of the inputs in base-2.

    Helper class for use with e.g. numpy.logaddexp2

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html
    """
    nargs = 2

    def __new__(cls, *args):
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            wrt, other = self.args
        elif argindex == 2:
            other, wrt = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One/(S.One + _exp2(other-wrt))

    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        return _logaddexp2(x1, x2)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        a, b = (x.simplify(**kwargs).factor() for x in self.args)
        candidate = _logaddexp2(a, b)
        if candidate != _logaddexp2(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp2(a, b)


class amin(Token):
    """ Minimum value along an axis.

    Helper class for use with e.g. numpy.amin


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.amin.html
    """
    __slots__ = _fields = ('array', 'axis')
    defaults = {'axis': none}
    _construct_axis = staticmethod(sympify)


class amax(Token):
    """ Maximum value along an axis.

    Helper class for use with e.g. numpy.amax


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.amax.html
    """
    __slots__ = _fields = ('array', 'axis')
    defaults = {'axis': none}
    _construct_axis = staticmethod(sympify)


class maximum(Function):
    """ Element-wise maximum of array elements.

    Helper class for use with e.g. numpy.maximum


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    """

    def _eval_rewrite_as_Max(self, *args):
        return Max(*self.args)


class minimum(Function):
    """ Element-wise minimum of array elements.

    Helper class for use with e.g. numpy.minimum


    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
    """

    def _eval_rewrite_as_Min(self, *args):
        return Min(*self.args)
<|endoftext|>from sympy.printing.pycode import PythonCodePrinter

""" This module collects utilities for rendering Python code. """


def render_as_module(content, standard='python3'):
    """Renders Python code as a module (with the required imports).

    Parameters
    ==========

    standard :
        See the parameter ``standard`` in
        :meth:`sympy.printing.pycode.pycode`
    """

    printer = PythonCodePrinter({'standard':standard})
    pystr = printer.doprint(content)
    if printer._settings['fully_qualified_modules']:
        module_imports_str = '\n'.join('import %s' % k for k in printer.module_imports)
    else:
        module_imports_str = '\n'.join(['from %s import %s' % (k, ', '.join(v)) for
                                        k, v in printer.module_imports.items()])
    return module_imports_str + '\n\n' + pystr
<|endoftext|>from sympy.core.containers import Tuple
from sympy.core.numbers import oo
from sympy.core.relational import (Gt, Lt)
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And
from sympy.codegen.ast import (
    Assignment, AddAugmentedAssignment, break_, CodeBlock, Declaration, FunctionDefinition,
    Print, Return, Scope, While, Variable, Pointer, real
)
from sympy.codegen.cfunctions import isnan

""" This module collects functions for constructing ASTs representing algorithms. """

def newtons_method(expr, wrt, atol=1e-12, delta=None, *, rtol=4e-16, debug=False,
                   itermax=None, counter=None, delta_fn=lambda e, x: -e/e.diff(x),
                   cse=False, handle_nan=None,
                   bounds=None):
    """ Generates an AST for Newton-Raphson method (a root-finding algorithm).

    Explanation
    ===========

    Returns an abstract syntax tree (AST) based on ``sympy.codegen.ast`` for Netwon's
    method of root-finding.

    Parameters
    ==========

    expr : expression
    wrt : Symbol
        With respect to, i.e. what is the variable.
    atol : number or expression
        Absolute tolerance (stopping criterion)
    rtol : number or expression
        Relative tolerance (stopping criterion)
    delta : Symbol
        Will be a ``Dummy`` if ``None``.
    debug : bool
        Whether to print convergence information during iterations
    itermax : number or expr
        Maximum number of iterations.
    counter : Symbol
        Will be a ``Dummy`` if ``None``.
    delta_fn: Callable[[Expr, Symbol], Expr]
        computes the step, default is newtons method. For e.g. Halley's method
        use delta_fn=lambda e, x: -2*e*e.diff(x)/(2*e.diff(x)**2 - e*e.diff(x, 2))
    cse: bool
        Perform common sub-expression elimination on delta expression
    handle_nan: Token
        How to handle occurrence of not-a-number (NaN).
    bounds: Optional[tuple[Expr, Expr]]
        Perform optimization within bounds

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.ast import Assignment
    >>> from sympy.codegen.algorithms import newtons_method
    >>> x, dx, atol = symbols('x dx atol')
    >>> expr = cos(x) - x**3
    >>> algo = newtons_method(expr, x, atol=atol, delta=dx)
    >>> algo.has(Assignment(dx, -expr/expr.diff(x)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Newton%27s_method

    """

    if delta is None:
        delta = Dummy()
        Wrapper = Scope
        name_d = 'delta'
    else:
        Wrapper = lambda x: x
        name_d = delta.name

    delta_expr = delta_fn(expr, wrt)
    if cse:
        from sympy.simplify.cse_main import cse
        cses, (red,) = cse([delta_expr.factor()])
        whl_bdy = [Assignment(dum, sub_e) for dum, sub_e in cses]
        whl_bdy += [Assignment(delta, red)]
    else:
        whl_bdy = [Assignment(delta, delta_expr)]
    if handle_nan is not None:
        whl_bdy += [While(isnan(delta), CodeBlock(handle_nan, break_))]
    whl_bdy += [AddAugmentedAssignment(wrt, delta)]
    if bounds is not None:
        whl_bdy += [Assignment(wrt, Min(Max(wrt, bounds[0]), bounds[1]))]
    if debug:
        prnt = Print([wrt, delta], r"{}=%12.5g {}=%12.5g\n".format(wrt.name, name_d))
        whl_bdy += [prnt]
    req = Gt(Abs(delta), atol + rtol*Abs(wrt))
    declars = [Declaration(Variable(delta, type=real, value=oo))]
    if itermax is not None:
        counter = counter or Dummy(integer=True)
        v_counter = Variable.deduced(counter, 0)
        declars.append(Declaration(v_counter))
        whl_bdy.append(AddAugmentedAssignment(counter, 1))
        req = And(req, Lt(counter, itermax))
    whl = While(req, CodeBlock(*whl_bdy))
    blck = declars
    if debug:
        blck.append(Print([wrt], r"{}=%12.5g\n".format(wrt.name)))
    blck += [whl]
    return Wrapper(CodeBlock(*blck))


def _symbol_of(arg):
    if isinstance(arg, Declaration):
        arg = arg.variable.symbol
    elif isinstance(arg, Variable):
        arg = arg.symbol
    return arg


def newtons_method_function(expr, wrt, params=None, func_name="newton", attrs=Tuple(), *, delta=None, **kwargs):
    """ Generates an AST for a function implementing the Newton-Raphson method.

    Parameters
    ==========

    expr : expression
    wrt : Symbol
        With respect to, i.e. what is the variable
    params : iterable of symbols
        Symbols appearing in expr that are taken as constants during the iterations
        (these will be accepted as parameters to the generated function).
    func_name : str
        Name of the generated function.
    attrs : Tuple
        Attribute instances passed as ``attrs`` to ``FunctionDefinition``.
    \\*\\*kwargs :
        Keyword arguments passed to :func:`sympy.codegen.algorithms.newtons_method`.

    Examples
    ========

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.algorithms import newtons_method_function
    >>> from sympy.codegen.pyutils import render_as_module
    >>> x = symbols('x')
    >>> expr = cos(x) - x**3
    >>> func = newtons_method_function(expr, x)
    >>> py_mod = render_as_module(func)  # source code as string
    >>> namespace = {}
    >>> exec(py_mod, namespace, namespace)
    >>> res = eval('newton(0.5)', namespace)
    >>> abs(res - 0.865474033102) < 1e-12
    True

    See Also
    ========

    sympy.codegen.algorithms.newtons_method

    """
    if params is None:
        params = (wrt,)
    pointer_subs = {p.symbol: Symbol('(*%s)' % p.symbol.name)
                    for p in params if isinstance(p, Pointer)}
    if delta is None:
        delta = Symbol('d_' + wrt.name)
        if expr.has(delta):
            delta = None  # will use Dummy
    algo = newtons_method(expr, wrt, delta=delta, **kwargs).xreplace(pointer_subs)
    if isinstance(algo, Scope):
        algo = algo.body
    not_in_params = expr.free_symbols.difference({_symbol_of(p) for p in params})
    if not_in_params:
        raise ValueError("Missing symbols in params: %s" % ', '.join(map(str, not_in_params)))
    declars = tuple(Variable(p, real) for p in params)
    body = CodeBlock(algo, Return(wrt))
    return FunctionDefinition(real, func_name, declars, body, attrs=attrs)
<|endoftext|>"""
AST nodes specific to Fortran.

The functions defined in this module allows the user to express functions such as ``dsign``
as a SymPy function for symbolic manipulation.
"""

from __future__ import annotations
from sympy.codegen.ast import (
    Attribute, CodeBlock, FunctionCall, Node, none, String,
    Token, _mk_Tuple, Variable
)
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable



pure = Attribute('pure')
elemental = Attribute('elemental')  # (all elemental procedures are also pure)

intent_in = Attribute('intent_in')
intent_out = Attribute('intent_out')
intent_inout = Attribute('intent_inout')

allocatable = Attribute('allocatable')

class Program(Token):
    """ Represents a 'program' block in Fortran.

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy.codegen.fnodes import Program
    >>> prog = Program('myprogram', [Print([42])])
    >>> from sympy import fcode
    >>> print(fcode(prog, source_format='free'))
    program myprogram
        print *, 42
    end program

    """
    __slots__ = _fields = ('name', 'body')
    _construct_name = String
    _construct_body = staticmethod(lambda body: CodeBlock(*body))


class use_rename(Token):
    """ Represents a renaming in a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use_rename, use
    >>> from sympy import fcode
    >>> ren = use_rename("thingy", "convolution2d")
    >>> print(fcode(ren, source_format='free'))
    thingy => convolution2d
    >>> full = use('signallib', only=['snr', ren])
    >>> print(fcode(full, source_format='free'))
    use signallib, only: snr, thingy => convolution2d

    """
    __slots__ = _fields = ('local', 'original')
    _construct_local = String
    _construct_original = String

def _name(arg):
    if hasattr(arg, 'name'):
        return arg.name
    else:
        return String(arg)

class use(Token):
    """ Represents a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use
    >>> from sympy import fcode
    >>> fcode(use('signallib'), source_format='free')
    'use signallib'
    >>> fcode(use('signallib', [('metric', 'snr')]), source_format='free')
    'use signallib, metric => snr'
    >>> fcode(use('signallib', only=['snr', 'convolution2d']), source_format='free')
    'use signallib, only: snr, convolution2d'

    """
    __slots__ = _fields = ('namespace', 'rename', 'only')
    defaults = {'rename': none, 'only': none}
    _construct_namespace = staticmethod(_name)
    _construct_rename = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else use_rename(*arg) for arg in args]))
    _construct_only = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else _name(arg) for arg in args]))


class Module(Token):
    """ Represents a module in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Module
    >>> from sympy import fcode
    >>> print(fcode(Module('signallib', ['implicit none'], []), source_format='free'))
    module signallib
    implicit none
    <BLANKLINE>
    contains
    <BLANKLINE>
    <BLANKLINE>
    end module

    """
    __slots__ = _fields = ('name', 'declarations', 'definitions')
    defaults = {'declarations': Tuple()}
    _construct_name = String

    @classmethod
    def _construct_declarations(cls, args):
        args = [Str(arg) if isinstance(arg, str) else arg for arg in args]
        return CodeBlock(*args)

    _construct_definitions = staticmethod(lambda arg: CodeBlock(*arg))


class Subroutine(Node):
    """ Represents a subroutine in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import Print
    >>> from sympy.codegen.fnodes import Subroutine
    >>> x, y = symbols('x y', real=True)
    >>> sub = Subroutine('mysub', [x, y], [Print([x**2 + y**2, x*y])])
    >>> print(fcode(sub, source_format='free', standard=2003))
    subroutine mysub(x, y)
    real*8 :: x
    real*8 :: y
    print *, x**2 + y**2, x*y
    end subroutine

    """
    __slots__ = ('name', 'parameters', 'body')
    _fields = __slots__ + Node._fields
    _construct_name = String
    _construct_parameters = staticmethod(lambda params: Tuple(*map(Variable.deduced, params)))

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

class SubroutineCall(Token):
    """ Represents a call to a subroutine in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import SubroutineCall
    >>> from sympy import fcode
    >>> fcode(SubroutineCall('mysub', 'x y'.split()))
    '       call mysub(x, y)'

    """
    __slots__ = _fields = ('name', 'subroutine_args')
    _construct_name = staticmethod(_name)
    _construct_subroutine_args = staticmethod(_mk_Tuple)


class Do(Token):
    """ Represents a Do loop in in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import aug_assign, Print
    >>> from sympy.codegen.fnodes import Do
    >>> i, n = symbols('i n', integer=True)
    >>> r = symbols('r', real=True)
    >>> body = [aug_assign(r, '+', 1/i), Print([i, r])]
    >>> do1 = Do(body, i, 1, n)
    >>> print(fcode(do1, source_format='free'))
    do i = 1, n
        r = r + 1d0/i
        print *, i, r
    end do
    >>> do2 = Do(body, i, 1, n, 2)
    >>> print(fcode(do2, source_format='free'))
    do i = 1, n, 2
        r = r + 1d0/i
        print *, i, r
    end do

    """

    __slots__ = _fields = ('body', 'counter', 'first', 'last', 'step', 'concurrent')
    defaults = {'step': Integer(1), 'concurrent': false}
    _construct_body = staticmethod(lambda body: CodeBlock(*body))
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)
    _construct_concurrent = staticmethod(lambda arg: true if arg else false)


class ArrayConstructor(Token):
    """ Represents an array constructor.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import ArrayConstructor
    >>> ac = ArrayConstructor([1, 2, 3])
    >>> fcode(ac, standard=95, source_format='free')
    '(/1, 2, 3/)'
    >>> fcode(ac, standard=2003, source_format='free')
    '[1, 2, 3]'

    """
    __slots__ = _fields = ('elements',)
    _construct_elements = staticmethod(_mk_Tuple)


class ImpliedDoLoop(Token):
    """ Represents an implied do loop in Fortran.

    Examples
    ========

    >>> from sympy import Symbol, fcode
    >>> from sympy.codegen.fnodes import ImpliedDoLoop, ArrayConstructor
    >>> i = Symbol('i', integer=True)
    >>> idl = ImpliedDoLoop(i**3, i, -3, 3, 2)  # -27, -1, 1, 27
    >>> ac = ArrayConstructor([-28, idl, 28]) # -28, -27, -1, 1, 27, 28
    >>> fcode(ac, standard=2003, source_format='free')
    '[-28, (i**3, i = -3, 3, 2), 28]'

    """
    __slots__ = _fields = ('expr', 'counter', 'first', 'last', 'step')
    defaults = {'step': Integer(1)}
    _construct_expr = staticmethod(sympify)
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)


class Extent(Basic):
    """ Represents a dimension extent.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Extent
    >>> e = Extent(-3, 3)  # -3, -2, -1, 0, 1, 2, 3
    >>> from sympy import fcode
    >>> fcode(e, source_format='free')
    '-3:3'
    >>> from sympy.codegen.ast import Variable, real
    >>> from sympy.codegen.fnodes import dimension, intent_out
    >>> dim = dimension(e, e)
    >>> arr = Variable('x', real, attrs=[dim, intent_out])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'real*8, dimension(-3:3, -3:3), intent(out) :: x'

    """
    def __new__(cls, *args):
        if len(args) == 2:
            low, high = args
            return Basic.__new__(cls, sympify(low), sympify(high))
        elif len(args) == 0 or (len(args) == 1 and args[0] in (':', None)):
            return Basic.__new__(cls)  # assumed shape
        else:
            raise ValueError("Expected 0 or 2 args (or one argument == None or ':')")

    def _sympystr(self, printer):
        if len(self.args) == 0:
            return ':'
        return ":".join(str(arg) for arg in self.args)

assumed_extent = Extent() # or Extent(':'), Extent(None)


def dimension(*args):
    """ Creates a 'dimension' Attribute with (up to 7) extents.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import dimension, intent_in
    >>> dim = dimension('2', ':')  # 2 rows, runtime determined number of columns
    >>> from sympy.codegen.ast import Variable, integer
    >>> arr = Variable('a', integer, attrs=[dim, intent_in])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'integer*4, dimension(2, :), intent(in) :: a'

    """
    if len(args) > 7:
        raise ValueError("Fortran only supports up to 7 dimensional arrays")
    parameters = []
    for arg in args:
        if isinstance(arg, Extent):
            parameters.append(arg)
        elif isinstance(arg, str):
            if arg == ':':
                parameters.append(Extent())
            else:
                parameters.append(String(arg))
        elif iterable(arg):
            parameters.append(Extent(*arg))
        else:
            parameters.append(sympify(arg))
    if len(args) == 0:
        raise ValueError("Need at least one dimension")
    return Attribute('dimension', parameters)


assumed_size = dimension('*')

def array(symbol, dim, intent=None, *, attrs=(), value=None, type=None):
    """ Convenience function for creating a Variable instance for a Fortran array.

    Parameters
    ==========

    symbol : symbol
    dim : Attribute or iterable
        If dim is an ``Attribute`` it need to have the name 'dimension'. If it is
        not an ``Attribute``, then it is passed to :func:`dimension` as ``*dim``
    intent : str
        One of: 'in', 'out', 'inout' or None
    \\*\\*kwargs:
        Keyword arguments for ``Variable`` ('type' & 'value')

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.ast import integer, real
    >>> from sympy.codegen.fnodes import array
    >>> arr = array('a', '*', 'in', type=integer)
    >>> print(fcode(arr.as_Declaration(), source_format='free', standard=2003))
    integer*4, dimension(*), intent(in) :: a
    >>> x = array('x', [3, ':', ':'], intent='out', type=real)
    >>> print(fcode(x.as_Declaration(value=1), source_format='free', standard=2003))
    real*8, dimension(3, :, :), intent(out) :: x = 1

    """
    if isinstance(dim, Attribute):
        if str(dim.name) != 'dimension':
            raise ValueError("Got an unexpected Attribute argument as dim: %s" % str(dim))
    else:
        dim = dimension(*dim)

    attrs = list(attrs) + [dim]
    if intent is not None:
        if intent not in (intent_in, intent_out, intent_inout):
            intent = {'in': intent_in, 'out': intent_out, 'inout': intent_inout}[intent]
        attrs.append(intent)
    if type is None:
        return Variable.deduced(symbol, value=value, attrs=attrs)
    else:
        return Variable(symbol, type, value=value, attrs=attrs)

def _printable(arg):
    return String(arg) if isinstance(arg, str) else sympify(arg)


def allocated(array):
    """ Creates an AST node for a function call to Fortran's "allocated(...)"

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import allocated
    >>> alloc = allocated('x')
    >>> fcode(alloc, source_format='free')
    'allocated(x)'

    """
    return FunctionCall('allocated', [_printable(array)])


def lbound(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "lbound(...)"

    Parameters
    ==========

    array : Symbol or String
    dim : expr
    kind : expr

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import lbound
    >>> lb = lbound('arr', dim=2)
    >>> fcode(lb, source_format='free')
    'lbound(arr, 2)'

    """
    return FunctionCall(
        'lbound',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def ubound(array, dim=None, kind=None):
    return FunctionCall(
        'ubound',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def shape(source, kind=None):
    """ Creates an AST node for a function call to Fortran's "shape(...)"

    Parameters
    ==========

    source : Symbol or String
    kind : expr

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import shape
    >>> shp = shape('x')
    >>> fcode(shp, source_format='free')
    'shape(x)'

    """
    return FunctionCall(
        'shape',
        [_printable(source)] +
        ([_printable(kind)] if kind else [])
    )


def size(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "size(...)"

    Examples
    ========

    >>> from sympy import fcode, Symbol
    >>> from sympy.codegen.ast import FunctionDefinition, real, Return
    >>> from sympy.codegen.fnodes import array, sum_, size
    >>> a = Symbol('a', real=True)
    >>> body = [Return((sum_(a**2)/size(a))**.5)]
    >>> arr = array(a, dim=[':'], intent='in')
    >>> fd = FunctionDefinition(real, 'rms', [arr], body)
    >>> print(fcode(fd, source_format='free', standard=2003))
    real*8 function rms(a)
    real*8, dimension(:), intent(in) :: a
    rms = sqrt(sum(a**2)*1d0/size(a))
    end function

    """
    return FunctionCall(
        'size',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def reshape(source, shape, pad=None, order=None):
    """ Creates an AST node for a function call to Fortran's "reshape(...)"

    Parameters
    ==========

    source : Symbol or String
    shape : ArrayExpr

    """
    return FunctionCall(
        'reshape',
        [_printable(source), _printable(shape)] +
        ([_printable(pad)] if pad else []) +
        ([_printable(order)] if pad else [])
    )


def bind_C(name=None):
    """ Creates an Attribute ``bind_C`` with a name.

    Parameters
    ==========

    name : str

    Examples
    ========

    >>> from sympy import fcode, Symbol
    >>> from sympy.codegen.ast import FunctionDefinition, real, Return
    >>> from sympy.codegen.fnodes import array, sum_, bind_C
    >>> a = Symbol('a', real=True)
    >>> s = Symbol('s', integer=True)
    >>> arr = array(a, dim=[s], intent='in')
    >>> body = [Return((sum_(a**2)/s)**.5)]
    >>> fd = FunctionDefinition(real, 'rms', [arr, s], body, attrs=[bind_C('rms')])
    >>> print(fcode(fd, source_format='free', standard=2003))
    real*8 function rms(a, s) bind(C, name="rms")
    real*8, dimension(s), intent(in) :: a
    integer*4 :: s
    rms = sqrt(sum(a**2)/s)
    end function

    """
    return Attribute('bind_C', [String(name)] if name else [])

class GoTo(Token):
    """ Represents a goto statement in Fortran

    Examples
    ========

    >>> from sympy.codegen.fnodes import GoTo
    >>> go = GoTo([10, 20, 30], 'i')
    >>> from sympy import fcode
    >>> fcode(go, source_format='free')
    'go to (10, 20, 30), i'

    """
    __slots__ = _fields = ('labels', 'expr')
    defaults = {'expr': none}
    _construct_labels = staticmethod(_mk_Tuple)
    _construct_expr = staticmethod(sympify)


class FortranReturn(Token):
    """ AST node explicitly mapped to a fortran "return".

    Explanation
    ===========

    Because a return statement in fortran is different from C, and
    in order to aid reuse of our codegen ASTs the ordinary
    ``.codegen.ast.Return`` is interpreted as assignment to
    the result variable of the function. If one for some reason needs
    to generate a fortran RETURN statement, this node should be used.

    Examples
    ========

    >>> from sympy.codegen.fnodes import FortranReturn
    >>> from sympy import fcode
    >>> fcode(FortranReturn('x'))
    '       return x'

    """
    __slots__ = _fields = ('return_value',)
    defaults = {'return_value': none}
    _construct_return_value = staticmethod(sympify)


class FFunction(Function):
    _required_standard = 77

    def _fcode(self, printer):
        name = self.__class__.__name__
        if printer._settings['standard'] < self._required_standard:
            raise NotImplementedError("%s requires Fortran %d or newer" %
                                      (name, self._required_standard))
        return '{}({})'.format(name, ', '.join(map(printer._print, self.args)))


class F95Function(FFunction):
    _required_standard = 95


class isign(FFunction):
    """ Fortran sign intrinsic for integer arguments. """
    nargs = 2


class dsign(FFunction):
    """ Fortran sign intrinsic for double precision arguments. """
    nargs = 2


class cmplx(FFunction):
    """ Fortran complex conversion function. """
    nargs = 2  # may be extended to (2, 3) at a later point


class kind(FFunction):
    """ Fortran kind function. """
    nargs = 1


class merge(F95Function):
    """ Fortran merge function """
    nargs = 3


class _literal(Float):
    _token: str
    _decimals: int

    def _fcode(self, printer, *args, **kwargs):
        mantissa, sgnd_ex = ('%.{}e'.format(self._decimals) % self).split('e')
        mantissa = mantissa.strip('0').rstrip('.')
        ex_sgn, ex_num = sgnd_ex[0], sgnd_ex[1:].lstrip('0')
        ex_sgn = '' if ex_sgn == '+' else ex_sgn
        return (mantissa or '0') + self._token + ex_sgn + (ex_num or '0')


class literal_sp(_literal):
    """ Fortran single precision real literal """
    _token = 'e'
    _decimals = 9


class literal_dp(_literal):
    """ Fortran double precision real literal """
    _token = 'd'
    _decimals = 17


class sum_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)


class product_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)
<|endoftext|>"""
This module contains SymPy functions mathcin corresponding to special math functions in the
C standard library (since C99, also available in C++11).

The functions defined in this module allows the user to express functions such as ``expm1``
as a SymPy function for symbolic manipulation.

"""
from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.logic.boolalg import BooleanFunction, true, false

def _expm1(x):
    return exp(x) - S.One


class expm1(Function):
    """
    Represents the exponential function minus one.

    Explanation
    ===========

    The benefit of using ``expm1(x)`` over ``exp(x) - 1``
    is that the latter is prone to cancellation under finite precision
    arithmetic when x is close to zero.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import expm1
    >>> '%.0e' % expm1(1e-99).evalf()
    '1e-99'
    >>> from math import exp
    >>> exp(1e-99) - 1
    0.0
    >>> expm1(x).diff(x)
    exp(x)

    See Also
    ========

    log1p
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return exp(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_expand_func(self, **hints):
        return _expm1(*self.args)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return exp(arg) - S.One

    _eval_rewrite_as_tractable = _eval_rewrite_as_exp

    @classmethod
    def eval(cls, arg):
        exp_arg = exp.eval(arg)
        if exp_arg is not None:
            return exp_arg - S.One

    def _eval_is_real(self):
        return self.args[0].is_real

    def _eval_is_finite(self):
        return self.args[0].is_finite


def _log1p(x):
    return log(x + S.One)


class log1p(Function):
    """
    Represents the natural logarithm of a number plus one.

    Explanation
    ===========

    The benefit of using ``log1p(x)`` over ``log(x + 1)``
    is that the latter is prone to cancellation under finite precision
    arithmetic when x is close to zero.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log1p
    >>> from sympy import expand_log
    >>> '%.0e' % expand_log(log1p(1e-99)).evalf()
    '1e-99'
    >>> from math import log
    >>> log(1 + 1e-99)
    0.0
    >>> log1p(x).diff(x)
    1/(x + 1)

    See Also
    ========

    expm1
    """
    nargs = 1


    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return S.One/(self.args[0] + S.One)
        else:
            raise ArgumentIndexError(self, argindex)


    def _eval_expand_func(self, **hints):
        return _log1p(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log1p(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    @classmethod
    def eval(cls, arg):
        if arg.is_Rational:
            return log(arg + S.One)
        elif not arg.is_Float:  # not safe to add 1 to Float
            return log.eval(arg + S.One)
        elif arg.is_number:
            return log(Rational(arg) + S.One)

    def _eval_is_real(self):
        return (self.args[0] + S.One).is_nonnegative

    def _eval_is_finite(self):
        if (self.args[0] + S.One).is_zero:
            return False
        return self.args[0].is_finite

    def _eval_is_positive(self):
        return self.args[0].is_positive

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_nonnegative(self):
        return self.args[0].is_nonnegative

_Two = S(2)

def _exp2(x):
    return Pow(_Two, x)

class exp2(Function):
    """
    Represents the exponential function with base two.

    Explanation
    ===========

    The benefit of using ``exp2(x)`` over ``2**x``
    is that the latter is not as efficient under finite precision
    arithmetic.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import exp2
    >>> exp2(2).evalf() == 4.0
    True
    >>> exp2(x).diff(x)
    log(2)*exp2(x)

    See Also
    ========

    log2
    """
    nargs = 1


    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return self*log(_Two)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _exp2(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow

    def _eval_expand_func(self, **hints):
        return _exp2(*self.args)

    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            return _exp2(arg)


def _log2(x):
    return log(x)/log(_Two)


class log2(Function):
    """
    Represents the logarithm function with base two.

    Explanation
    ===========

    The benefit of using ``log2(x)`` over ``log(x)/log(2)``
    is that the latter is not as efficient under finite precision
    arithmetic.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log2
    >>> log2(4).evalf() == 2.0
    True
    >>> log2(x).diff(x)
    1/(x*log(2))

    See Also
    ========

    exp2
    log10
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return S.One/(log(_Two)*self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            result = log.eval(arg, base=_Two)
            if result.is_Atom:
                return result
        elif arg.is_Pow and arg.base == _Two:
            return arg.exp

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_expand_func(self, **hints):
        return _log2(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log2(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_log


def _fma(x, y, z):
    return x*y + z


class fma(Function):
    """
    Represents "fused multiply add".

    Explanation
    ===========

    The benefit of using ``fma(x, y, z)`` over ``x*y + z``
    is that, under finite precision arithmetic, the former is
    supported by special instructions on some CPUs.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.codegen.cfunctions import fma
    >>> fma(x, y, z).diff(x)
    y

    """
    nargs = 3

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex in (1, 2):
            return self.args[2 - argindex]
        elif argindex == 3:
            return S.One
        else:
            raise ArgumentIndexError(self, argindex)


    def _eval_expand_func(self, **hints):
        return _fma(*self.args)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return _fma(arg)


_Ten = S(10)


def _log10(x):
    return log(x)/log(_Ten)


class log10(Function):
    """
    Represents the logarithm function with base ten.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log10
    >>> log10(100).evalf() == 2.0
    True
    >>> log10(x).diff(x)
    1/(x*log(10))

    See Also
    ========

    log2
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return S.One/(log(_Ten)*self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            result = log.eval(arg, base=_Ten)
            if result.is_Atom:
                return result
        elif arg.is_Pow and arg.base == _Ten:
            return arg.exp

    def _eval_expand_func(self, **hints):
        return _log10(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log10(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_log


def _Sqrt(x):
    return Pow(x, S.Half)


class Sqrt(Function):  # 'sqrt' already defined in sympy.functions.elementary.miscellaneous
    """
    Represents the square root function.

    Explanation
    ===========

    The reason why one would use ``Sqrt(x)`` over ``sqrt(x)``
    is that the latter is internally represented as ``Pow(x, S.Half)`` which
    may not be what one wants when doing code-generation.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import Sqrt
    >>> Sqrt(x)
    Sqrt(x)
    >>> Sqrt(x).diff(x)
    1/(2*sqrt(x))

    See Also
    ========

    Cbrt
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return Pow(self.args[0], Rational(-1, 2))/_Two
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_expand_func(self, **hints):
        return _Sqrt(*self.args)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _Sqrt(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow


def _Cbrt(x):
    return Pow(x, Rational(1, 3))


class Cbrt(Function):  # 'cbrt' already defined in sympy.functions.elementary.miscellaneous
    """
    Represents the cube root function.

    Explanation
    ===========

    The reason why one would use ``Cbrt(x)`` over ``cbrt(x)``
    is that the latter is internally represented as ``Pow(x, Rational(1, 3))`` which
    may not be what one wants when doing code-generation.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import Cbrt
    >>> Cbrt(x)
    Cbrt(x)
    >>> Cbrt(x).diff(x)
    1/(3*x**(2/3))

    See Also
    ========

    Sqrt
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return Pow(self.args[0], Rational(-_Two/3))/3
        else:
            raise ArgumentIndexError(self, argindex)


    def _eval_expand_func(self, **hints):
        return _Cbrt(*self.args)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _Cbrt(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow


def _hypot(x, y):
    return sqrt(Pow(x, 2) + Pow(y, 2))


class hypot(Function):
    """
    Represents the hypotenuse function.

    Explanation
    ===========

    The hypotenuse function is provided by e.g. the math library
    in the C99 standard, hence one may want to represent the function
    symbolically when doing code-generation.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.codegen.cfunctions import hypot
    >>> hypot(3, 4).evalf() == 5.0
    True
    >>> hypot(x, y)
    hypot(x, y)
    >>> hypot(x, y).diff(x)
    x/hypot(x, y)

    """
    nargs = 2

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex in (1, 2):
            return 2*self.args[argindex-1]/(_Two*self.func(*self.args))
        else:
            raise ArgumentIndexError(self, argindex)


    def _eval_expand_func(self, **hints):
        return _hypot(*self.args)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _hypot(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow


class isnan(BooleanFunction):
    nargs = 1

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return true
        elif arg.is_number:
            return false
        else:
            return None


class isinf(BooleanFunction):
    nargs = 1

    @classmethod
    def eval(cls, arg):
        if arg.is_infinite:
            return true
        elif arg.is_finite:
            return false
        else:
            return None
<|endoftext|>from sympy.core.containers import Tuple
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
import sympy.polys

from math import gcd


def egyptian_fraction(r, algorithm="Greedy"):
    """
    Return the list of denominators of an Egyptian fraction
    expansion [1]_ of the said rational `r`.

    Parameters
    ==========

    r : Rational or (p, q)
        a positive rational number, ``p/q``.
    algorithm : { "Greedy", "Graham Jewett", "Takenouchi", "Golomb" }, optional
        Denotes the algorithm to be used (the default is "Greedy").

    Examples
    ========

    >>> from sympy import Rational
    >>> from sympy.ntheory.egyptian_fraction import egyptian_fraction
    >>> egyptian_fraction(Rational(3, 7))
    [3, 11, 231]
    >>> egyptian_fraction((3, 7), "Graham Jewett")
    [7, 8, 9, 56, 57, 72, 3192]
    >>> egyptian_fraction((3, 7), "Takenouchi")
    [4, 7, 28]
    >>> egyptian_fraction((3, 7), "Golomb")
    [3, 15, 35]
    >>> egyptian_fraction((11, 5), "Golomb")
    [1, 2, 3, 4, 9, 234, 1118, 2580]

    See Also
    ========

    sympy.core.numbers.Rational

    Notes
    =====

    Currently the following algorithms are supported:

    1) Greedy Algorithm

       Also called the Fibonacci-Sylvester algorithm [2]_.
       At each step, extract the largest unit fraction less
       than the target and replace the target with the remainder.

       It has some distinct properties:

       a) Given `p/q` in lowest terms, generates an expansion of maximum
          length `p`. Even as the numerators get large, the number of
          terms is seldom more than a handful.

       b) Uses minimal memory.

       c) The terms can blow up (standard examples of this are 5/121 and
          31/311).  The denominator is at most squared at each step
          (doubly-exponential growth) and typically exhibits
          singly-exponential growth.

    2) Graham Jewett Algorithm

       The algorithm suggested by the result of Graham and Jewett.
       Note that this has a tendency to blow up: the length of the
       resulting expansion is always ``2**(x/gcd(x, y)) - 1``.  See [3]_.

    3) Takenouchi Algorithm

       The algorithm suggested by Takenouchi (1921).
       Differs from the Graham-Jewett algorithm only in the handling
       of duplicates.  See [3]_.

    4) Golomb's Algorithm

       A method given by Golumb (1962), using modular arithmetic and
       inverses.  It yields the same results as a method using continued
       fractions proposed by Bleicher (1972).  See [4]_.

    If the given rational is greater than or equal to 1, a greedy algorithm
    of summing the harmonic sequence 1/1 + 1/2 + 1/3 + ... is used, taking
    all the unit fractions of this sequence until adding one more would be
    greater than the given number.  This list of denominators is prefixed
    to the result from the requested algorithm used on the remainder.  For
    example, if r is 8/3, using the Greedy algorithm, we get [1, 2, 3, 4,
    5, 6, 7, 14, 420], where the beginning of the sequence, [1, 2, 3, 4, 5,
    6, 7] is part of the harmonic sequence summing to 363/140, leaving a
    remainder of 31/420, which yields [14, 420] by the Greedy algorithm.
    The result of egyptian_fraction(Rational(8, 3), "Golomb") is [1, 2, 3,
    4, 5, 6, 7, 14, 574, 2788, 6460, 11590, 33062, 113820], and so on.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Egyptian_fraction
    .. [2] https://en.wikipedia.org/wiki/Greedy_algorithm_for_Egyptian_fractions
    .. [3] https://www.ics.uci.edu/~eppstein/numth/egypt/conflict.html
    .. [4] https://web.archive.org/web/20180413004012/https://ami.ektf.hu/uploads/papers/finalpdf/AMI_42_from129to134.pdf

    """

    if not isinstance(r, Rational):
        if isinstance(r, (Tuple, tuple)) and len(r) == 2:
            r = Rational(*r)
        else:
            raise ValueError("Value must be a Rational or tuple of ints")
    if r <= 0:
        raise ValueError("Value must be positive")

    # common cases that all methods agree on
    x, y = r.as_numer_denom()
    if y == 1 and x == 2:
        return [Integer(i) for i in [1, 2, 3, 6]]
    if x == y + 1:
        return [S.One, y]

    prefix, rem = egypt_harmonic(r)
    if rem == 0:
        return prefix
    # work in Python ints
    x, y = rem.p, rem.q
    # assert x < y and gcd(x, y) = 1

    if algorithm == "Greedy":
        postfix = egypt_greedy(x, y)
    elif algorithm == "Graham Jewett":
        postfix = egypt_graham_jewett(x, y)
    elif algorithm == "Takenouchi":
        postfix = egypt_takenouchi(x, y)
    elif algorithm == "Golomb":
        postfix = egypt_golomb(x, y)
    else:
        raise ValueError("Entered invalid algorithm")
    return prefix + [Integer(i) for i in postfix]


def egypt_greedy(x, y):
    # assumes gcd(x, y) == 1
    if x == 1:
        return [y]
    else:
        a = (-y) % x
        b = y*(y//x + 1)
        c = gcd(a, b)
        if c > 1:
            num, denom = a//c, b//c
        else:
            num, denom = a, b
        return [y//x + 1] + egypt_greedy(num, denom)


def egypt_graham_jewett(x, y):
    # assumes gcd(x, y) == 1
    l = [y] * x

    # l is now a list of integers whose reciprocals sum to x/y.
    # we shall now proceed to manipulate the elements of l without
    # changing the reciprocated sum until all elements are unique.

    while len(l) != len(set(l)):
        l.sort()  # so the list has duplicates. find a smallest pair
        for i in range(len(l) - 1):
            if l[i] == l[i + 1]:
                break
        # we have now identified a pair of identical
        # elements: l[i] and l[i + 1].
        # now comes the application of the result of graham and jewett:
        l[i + 1] = l[i] + 1
        # and we just iterate that until the list has no duplicates.
        l.append(l[i]*(l[i] + 1))
    return sorted(l)


def egypt_takenouchi(x, y):
    # assumes gcd(x, y) == 1
    # special cases for 3/y
    if x == 3:
        if y % 2 == 0:
            return [y//2, y]
        i = (y - 1)//2
        j = i + 1
        k = j + i
        return [j, k, j*k]
    l = [y] * x
    while len(l) != len(set(l)):
        l.sort()
        for i in range(len(l) - 1):
            if l[i] == l[i + 1]:
                break
        k = l[i]
        if k % 2 == 0:
            l[i] = l[i] // 2
            del l[i + 1]
        else:
            l[i], l[i + 1] = (k + 1)//2, k*(k + 1)//2
    return sorted(l)


def egypt_golomb(x, y):
    # assumes x < y and gcd(x, y) == 1
    if x == 1:
        return [y]
    xp = sympy.polys.ZZ.invert(int(x), int(y))
    rv = [xp*y]
    rv.extend(egypt_golomb((x*xp - 1)//y, xp))
    return sorted(rv)


def egypt_harmonic(r):
    # assumes r is Rational
    rv = []
    d = S.One
    acc = S.Zero
    while acc + 1/d <= r:
        acc += 1/d
        rv.append(d)
        d += 1
    return (rv, r - acc)
<|endoftext|>"""
Integer factorization
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, OrderedDict
from collections.abc import MutableMapping
import math

from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, Integer
from sympy.core.intfunc import num_digits
from sympy.core.power import Pow
from sympy.core.random import _randint
from sympy.core.singleton import S
from sympy.external.gmpy import (SYMPY_INTS, gcd, sqrt as isqrt,
                                 sqrtrem, iroot, bit_scan1, remove)
from .primetest import isprime, MERSENNE_PRIME_EXPONENTS, is_mersenne_prime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor


def smoothness(n):
    """
    Return the B-smooth and B-power smooth values of n.

    The smoothness of n is the largest prime factor of n; the power-
    smoothness is the largest divisor raised to its multiplicity.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import smoothness
    >>> smoothness(2**7*3**2)
    (3, 128)
    >>> smoothness(2**4*13)
    (13, 16)
    >>> smoothness(2)
    (2, 2)

    See Also
    ========

    factorint, smoothness_p
    """

    if n == 1:
        return (1, 1)  # not prime, but otherwise this causes headaches
    facs = factorint(n)
    return max(facs), max(m**facs[m] for m in facs)


def smoothness_p(n, m=-1, power=0, visual=None):
    """
    Return a list of [m, (p, (M, sm(p + m), psm(p + m)))...]
    where:

    1. p**M is the base-p divisor of n
    2. sm(p + m) is the smoothness of p + m (m = -1 by default)
    3. psm(p + m) is the power smoothness of p + m

    The list is sorted according to smoothness (default) or by power smoothness
    if power=1.

    The smoothness of the numbers to the left (m = -1) or right (m = 1) of a
    factor govern the results that are obtained from the p +/- 1 type factoring
    methods.

        >>> from sympy.ntheory.factor_ import smoothness_p, factorint
        >>> smoothness_p(10431, m=1)
        (1, [(3, (2, 2, 4)), (19, (1, 5, 5)), (61, (1, 31, 31))])
        >>> smoothness_p(10431)
        (-1, [(3, (2, 2, 2)), (19, (1, 3, 9)), (61, (1, 5, 5))])
        >>> smoothness_p(10431, power=1)
        (-1, [(3, (2, 2, 2)), (61, (1, 5, 5)), (19, (1, 3, 9))])

    If visual=True then an annotated string will be returned:

        >>> print(smoothness_p(21477639576571, visual=1))
        p**i=4410317**1 has p-1 B=1787, B-pow=1787
        p**i=4869863**1 has p-1 B=2434931, B-pow=2434931

    This string can also be generated directly from a factorization dictionary
    and vice versa:

        >>> factorint(17*9)
        {3: 2, 17: 1}
        >>> smoothness_p(_)
        'p**i=3**2 has p-1 B=2, B-pow=2\\np**i=17**1 has p-1 B=2, B-pow=16'
        >>> smoothness_p(_)
        {3: 2, 17: 1}

    The table of the output logic is:

        ====== ====== ======= =======
        |              Visual
        ------ ----------------------
        Input  True   False   other
        ====== ====== ======= =======
        dict    str    tuple   str
        str     str    tuple   dict
        tuple   str    tuple   str
        n       str    tuple   tuple
        mul     str    tuple   tuple
        ====== ====== ======= =======

    See Also
    ========

    factorint, smoothness
    """

    # visual must be True, False or other (stored as None)
    if visual in (1, 0):
        visual = bool(visual)
    elif visual not in (True, False):
        visual = None

    if isinstance(n, str):
        if visual:
            return n
        d = {}
        for li in n.splitlines():
            k, v = [int(i) for i in
                    li.split('has')[0].split('=')[1].split('**')]
            d[k] = v
        if visual is not True and visual is not False:
            return d
        return smoothness_p(d, visual=False)
    elif not isinstance(n, tuple):
        facs = factorint(n, visual=False)

    if power:
        k = -1
    else:
        k = 1
    if isinstance(n, tuple):
        rv = n
    else:
        rv = (m, sorted([(f,
                          tuple([M] + list(smoothness(f + m))))
                         for f, M in list(facs.items())],
                        key=lambda x: (x[1][k], x[0])))

    if visual is False or (visual is not True) and (type(n) in [int, Mul]):
        return rv
    lines = []
    for dat in rv[1]:
        dat = flatten(dat)
        dat.insert(2, m)
        lines.append('p**i=%i**%i has p%+i B=%i, B-pow=%i' % tuple(dat))
    return '\n'.join(lines)


def multiplicity(p, n):
    """
    Find the greatest integer m such that p**m divides n.

    Examples
    ========

    >>> from sympy import multiplicity, Rational
    >>> [multiplicity(5, n) for n in [8, 5, 25, 125, 250]]
    [0, 1, 2, 3, 3]
    >>> multiplicity(3, Rational(1, 9))
    -2

    Note: when checking for the multiplicity of a number in a
    large factorial it is most efficient to send it as an unevaluated
    factorial or to call ``multiplicity_in_factorial`` directly:

    >>> from sympy.ntheory import multiplicity_in_factorial
    >>> from sympy import factorial
    >>> p = factorial(25)
    >>> n = 2**100
    >>> nfac = factorial(n, evaluate=False)
    >>> multiplicity(p, nfac)
    52818775009509558395695966887
    >>> _ == multiplicity_in_factorial(p, n)
    True

    See Also
    ========

    trailing

    """
    try:
        p, n = as_int(p), as_int(n)
    except ValueError:
        from sympy.functions.combinatorial.factorials import factorial
        if all(isinstance(i, (SYMPY_INTS, Rational)) for i in (p, n)):
            p = Rational(p)
            n = Rational(n)
            if p.q == 1:
                if n.p == 1:
                    return -multiplicity(p.p, n.q)
                return multiplicity(p.p, n.p) - multiplicity(p.p, n.q)
            elif p.p == 1:
                return multiplicity(p.q, n.q)
            else:
                like = min(
                    multiplicity(p.p, n.p),
                    multiplicity(p.q, n.q))
                cross = min(
                    multiplicity(p.q, n.p),
                    multiplicity(p.p, n.q))
                return like - cross
        elif (isinstance(p, (SYMPY_INTS, Integer)) and
                isinstance(n, factorial) and
                isinstance(n.args[0], Integer) and
                n.args[0] >= 0):
            return multiplicity_in_factorial(p, n.args[0])
        raise ValueError('expecting ints or fractions, got %s and %s' % (p, n))

    if n == 0:
        raise ValueError('no such integer exists: multiplicity of %s is not-defined' %(n))
    return remove(n, p)[1]


def multiplicity_in_factorial(p, n):
    """return the largest integer ``m`` such that ``p**m`` divides ``n!``
    without calculating the factorial of ``n``.

    Parameters
    ==========

    p : Integer
        positive integer
    n : Integer
        non-negative integer

    Examples
    ========

    >>> from sympy.ntheory import multiplicity_in_factorial
    >>> from sympy import factorial

    >>> multiplicity_in_factorial(2, 3)
    1

    An instructive use of this is to tell how many trailing zeros
    a given factorial has. For example, there are 6 in 25!:

    >>> factorial(25)
    15511210043330985984000000
    >>> multiplicity_in_factorial(10, 25)
    6

    For large factorials, it is much faster/feasible to use
    this function rather than computing the actual factorial:

    >>> multiplicity_in_factorial(factorial(25), 2**100)
    52818775009509558395695966887

    See Also
    ========

    multiplicity

    """

    p, n = as_int(p), as_int(n)

    if p <= 0:
        raise ValueError('expecting positive integer got %s' % p )

    if n < 0:
        raise ValueError('expecting non-negative integer got %s' % n )

    # keep only the largest of a given multiplicity since those
    # of a given multiplicity will be goverened by the behavior
    # of the largest factor
    f = defaultdict(int)
    for k, v in factorint(p).items():
        f[v] = max(k, f[v])
    # multiplicity of p in n! depends on multiplicity
    # of prime `k` in p, so we floor divide by `v`
    # and keep it if smaller than the multiplicity of p
    # seen so far
    return min((n + k - sum(digits(n, k)))//(k - 1)//v for v, k in f.items())


def _perfect_power(n, next_p=2):
    """ Return integers ``(b, e)`` such that ``n == b**e`` if ``n`` is a unique
    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a perfect power).

    Explanation
    ===========

    This is a low-level helper for ``perfect_power``, for internal use.

    Parameters
    ==========

    n : int
        assume that n is a nonnegative integer
    next_p : int
        Assume that n has no factor less than next_p.
        i.e., all(n % p for p in range(2, next_p)) is True

    Examples
    ========
    >>> from sympy.ntheory.factor_ import _perfect_power
    >>> _perfect_power(16)
    (2, 4)
    >>> _perfect_power(17)
    False

    """
    if n <= 3:
        return False

    factors = {}
    g = 0
    multi = 1

    def done(n, factors, g, multi):
        g = gcd(g, multi)
        if g == 1:
            return False
        factors[n] = multi
        return math.prod(p**(e//g) for p, e in factors.items()), g

    # If n is small, only trial factoring is faster
    if n <= 1_000_000:
        n = _factorint_small(factors, n, 1_000, 1_000, next_p)[0]
        if n > 1:
            return False
        g = gcd(*factors.values())
        if g == 1:
            return False
        return math.prod(p**(e//g) for p, e in factors.items()), g

    # divide by 2
    if next_p < 3:
        g = bit_scan1(n)
        if g:
            if g == 1:
                return False
            n >>= g
            factors[2] = g
            if n == 1:
                return 2, g
            else:
                # If `m**g`, then we have found perfect power.
                # Otherwise, there is no possibility of perfect power, especially if `g` is prime.
                m, _exact = iroot(n, g)
                if _exact:
                    return 2*m, g
                elif isprime(g):
                    return False
        next_p = 3

    # square number?
    while n & 7 == 1: # n % 8 == 1:
        m, _exact = iroot(n, 2)
        if _exact:
            n = m
            multi <<= 1
        else:
            break
    if n < next_p**3:
        return done(n, factors, g, multi)

    # trial factoring
    # Since the maximum value an exponent can take is `log_{next_p}(n)`,
    # the number of exponents to be checked can be reduced by performing a trial factoring.
    # The value of `tf_max` needs more consideration.
    tf_max = n.bit_length()//27 + 24
    if next_p < tf_max:
        for p in primerange(next_p, tf_max):
            m, t = remove(n, p)
            if t:
                n = m
                t *= multi
                _g = gcd(g, t)
                if _g == 1:
                    return False
                factors[p] = t
                if n == 1:
                    return math.prod(p**(e//_g)
                                        for p, e in factors.items()), _g
                elif g == 0 or _g < g: # If g is updated
                    g = _g
                    m, _exact = iroot(n**multi, g)
                    if _exact:
                        return m * math.prod(p**(e//g)
                                            for p, e in factors.items()), g
                    elif isprime(g):
                        return False
        next_p = tf_max
    if n < next_p**3:
        return done(n, factors, g, multi)

    # check iroot
    if g:
        # If g is non-zero, the exponent is a divisor of g.
        # 2 can be omitted since it has already been checked.
        prime_iter = sorted(factorint(g >> bit_scan1(g)).keys())
    else:
        # The maximum possible value of the exponent is `log_{next_p}(n)`.
        # To compensate for the presence of computational error, 2 is added.
        prime_iter = primerange(3, int(math.log(n, next_p)) + 2)
    logn = math.log2(n)
    threshold = logn / 40 # Threshold for direct calculation
    for p in prime_iter:
        if threshold < p:
            # If p is large, find the power root p directly without `iroot`.
            while True:
                b = pow(2, logn / p)
                rb = int(b + 0.5)
                if abs(rb - b) < 0.01 and rb**p == n:
                    n = rb
                    multi *= p
                    logn = math.log2(n)
                else:
                    break
        else:
            while True:
                m, _exact = iroot(n, p)
                if _exact:
                    n = m
                    multi *= p
                    logn = math.log2(n)
                else:
                    break
        if n < next_p**(p + 2):
            break
    return done(n, factors, g, multi)


def perfect_power(n, candidates=None, big=True, factor=True):
    """
    Return ``(b, e)`` such that ``n`` == ``b**e`` if ``n`` is a unique
    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a
    perfect power). A ValueError is raised if ``n`` is not Rational.

    By default, the base is recursively decomposed and the exponents
    collected so the largest possible ``e`` is sought. If ``big=False``
    then the smallest possible ``e`` (thus prime) will be chosen.

    If ``factor=True`` then simultaneous factorization of ``n`` is
    attempted since finding a factor indicates the only possible root
    for ``n``. This is True by default since only a few small factors will
    be tested in the course of searching for the perfect power.

    The use of ``candidates`` is primarily for internal use; if provided,
    False will be returned if ``n`` cannot be written as a power with one
    of the candidates as an exponent and factoring (beyond testing for
    a factor of 2) will not be attempted.

    Examples
    ========

    >>> from sympy import perfect_power, Rational
    >>> perfect_power(16)
    (2, 4)
    >>> perfect_power(16, big=False)
    (4, 2)

    Negative numbers can only have odd perfect powers:

    >>> perfect_power(-4)
    False
    >>> perfect_power(-8)
    (-2, 3)

    Rationals are also recognized:

    >>> perfect_power(Rational(1, 2)**3)
    (1/2, 3)
    >>> perfect_power(Rational(-3, 2)**3)
    (-3/2, 3)

    Notes
    =====

    To know whether an integer is a perfect power of 2 use

        >>> is2pow = lambda n: bool(n and not n & (n - 1))
        >>> [(i, is2pow(i)) for i in range(5)]
        [(0, False), (1, True), (2, True), (3, False), (4, True)]

    It is not necessary to provide ``candidates``. When provided
    it will be assumed that they are ints. The first one that is
    larger than the computed maximum possible exponent will signal
    failure for the routine.

        >>> perfect_power(3**8, [9])
        False
        >>> perfect_power(3**8, [2, 4, 8])
        (3, 8)
        >>> perfect_power(3**8, [4, 8], big=False)
        (9, 4)

    See Also
    ========
    sympy.core.intfunc.integer_nthroot
    sympy.ntheory.primetest.is_square
    """
    # negative handling
    if n < 0:
        if candidates is None:
            pp = perfect_power(-n, big=True, factor=factor)
            if not pp:
                return False

            b, e = pp
            e2 = e & (-e)
            b, e = b ** e2, e // e2

            if e <= 1:
                return False

            if big or isprime(e):
                return -b, e

            for p in primerange(3, e + 1):
                if e % p == 0:
                    return - b ** (e // p), p

        odd_candidates = {i for i in candidates if i % 2}
        if not odd_candidates:
            return False

        pp = perfect_power(-n, odd_candidates, big, factor)
        if pp:
            return -pp[0], pp[1]

        return False

    # non-integer handling
    if isinstance(n, Rational) and not isinstance(n, Integer):
        p, q = n.p, n.q

        if p == 1:
            qq = perfect_power(q, candidates, big, factor)
            return (S.One / qq[0], qq[1]) if qq is not False else False

        if not (pp:=perfect_power(p, factor=factor)):
            return False
        if not (qq:=perfect_power(q, factor=factor)):
            return False
        (num_base, num_exp), (den_base, den_exp) = pp, qq

        def compute_tuple(exponent):
            """Helper to compute final result given an exponent"""
            new_num = num_base ** (num_exp // exponent)
            new_den = den_base ** (den_exp // exponent)
            return n.func(new_num, new_den), exponent

        if candidates:
            valid_candidates = [i for i in candidates
                                if num_exp % i == 0 and den_exp % i == 0]
            if not valid_candidates:
                return False

            e = max(valid_candidates) if big else min(valid_candidates)
            return compute_tuple(e)

        g = math.gcd(num_exp, den_exp)
        if g == 1:
            return False

        if big:
            return compute_tuple(g)

        e = next(p for p in primerange(2, g + 1) if g % p == 0)
        return compute_tuple(e)

    if candidates is not None:
        candidates = set(candidates)

    # positive integer handling
    n = as_int(n)

    if candidates is None and big:
        return _perfect_power(n)

    if n <= 3:
        # no unique exponent for 0, 1
        # 2 and 3 have exponents of 1
        return False
    logn = math.log2(n)
    max_possible = int(logn) + 2  # only check values less than this
    not_square = n % 10 in [2, 3, 7, 8]  # squares cannot end in 2, 3, 7, 8
    min_possible = 2 + not_square
    if not candidates:
        candidates = primerange(min_possible, max_possible)
    else:
        candidates = sorted([i for i in candidates
            if min_possible <= i < max_possible])
        if n%2 == 0:
            e = bit_scan1(n)
            candidates = [i for i in candidates if e%i == 0]
        if big:
            candidates = reversed(candidates)
        for e in candidates:
            r, ok = iroot(n, e)
            if ok:
                return int(r), e
        return False

    def _factors():
        rv = 2 + n % 2
        while True:
            yield rv
            rv = nextprime(rv)

    for fac, e in zip(_factors(), candidates):
        # see if there is a factor present
        if factor and n % fac == 0:
            # find what the potential power is
            e = remove(n, fac)[1]
            # if it's a trivial power we are done
            if e == 1:
                return False

            # maybe the e-th root of n is exact
            r, exact = iroot(n, e)
            if not exact:
                # Having a factor, we know that e is the maximal
                # possible value for a root of n.
                # If n = fac**e*m can be written as a perfect
                # power then see if m can be written as r**E where
                # gcd(e, E) != 1 so n = (fac**(e//E)*r)**E
                m = n//fac**e
                rE = perfect_power(m, candidates=divisors(e, generator=True))
                if not rE:
                    return False
                else:
                    r, E = rE
                    r, e = fac**(e//E)*r, E
            if not big:
                e0 = primefactors(e)
                if e0[0] != e:
                    r, e = r**(e//e0[0]), e0[0]
            return int(r), e

        # Weed out downright impossible candidates
        if logn/e < 40:
            b = 2.0**(logn/e)
            if abs(int(b + 0.5) - b) > 0.01:
                continue

        # now see if the plausible e makes a perfect power
        r, exact = iroot(n, e)
        if exact:
            if big:
                m = perfect_power(r, big=big, factor=factor)
                if m:
                    r, e = m[0], e*m[1]
            return int(r), e

    return False


class FactorCache(MutableMapping):
    """ Provides a cache for prime factors.
    ``factor_cache`` is pre-prepared as an instance of ``FactorCache``,
    and ``factorint`` internally references it to speed up
    the factorization of prime factors.

    While cache is automatically added during the execution of ``factorint``,
    users can also manually add prime factors independently.

    >>> from sympy import factor_cache
    >>> factor_cache[15] = 5

    Furthermore, by customizing ``get_external``,
    it is also possible to use external databases.
    The following is an example using http://factordb.com .

    .. code-block:: python

        import requests
        from sympy import factor_cache

        def get_external(self, n: int) -> list[int] | None:
            res = requests.get("http://factordb.com/api", params={"query": str(n)})
            if res.status_code != requests.codes.ok:
                return None
            j = res.json()
            if j.get("status") in ["FF", "P"]:
                return list(int(p) for p, _ in j.get("factors"))

        factor_cache.get_external = get_external

    Be aware that writing this code will trigger internet access
    to factordb.com when calling ``factorint``.

    """
    def __init__(self, maxsize: int | None = None):
        self._cache: OrderedDict[int, int] = OrderedDict()
        self.maxsize = maxsize

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, n) -> bool:
        return n in self._cache

    def __getitem__(self, n: int) -> int:
        factor = self.get(n)
        if factor is None:
            raise KeyError(f"{n} does not exist.")
        return factor

    def __setitem__(self, n: int, factor: int):
        if not (1 < factor <= n and n % factor == 0 and isprime(factor)):
            raise ValueError(f"{factor} is not a prime factor of {n}")
        self._cache[n] = max(self._cache.get(n, 0), factor)
        if self.maxsize is not None and len(self._cache) > self.maxsize:
            self._cache.popitem(False)

    def __delitem__(self, n: int):
        if n not in self._cache:
            raise KeyError(f"{n} does not exist.")
        del self._cache[n]

    def __iter__(self):
        return self._cache.__iter__()

    def cache_clear(self) -> None:
        """ Clear the cache """
        self._cache = OrderedDict()

    @property
    def maxsize(self) -> int | None:
        """ Returns the maximum cache size; if ``None``, it is unlimited. """
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError("maxsize must be None or a non-negative integer.")
        self._maxsize = value
        if value is not None:
            while len(self._cache) > value:
                self._cache.popitem(False)

    def get(self, n: int, default=None):
        """ Return the prime factor of ``n``.
        If it does not exist in the cache, return the value of ``default``.
        """
        if n <= sieve._list[-1]:
            if sieve._list[bisect_left(sieve._list, n)] == n:
                return n
        if n in self._cache:
            self._cache.move_to_end(n)
            return self._cache[n]
        if factors := self.get_external(n):
            self.add(n, factors)
            return self._cache[n]
        return default

    def add(self, n: int, factors: list[int]) -> None:
        for p in sorted(factors, reverse=True):
            self[n] = p
            n, _ = remove(n, p)

    def get_external(self, n: int) -> list[int] | None:
        return None


factor_cache = FactorCache(maxsize=1000)


def pollard_rho(n, s=2, a=1, retries=5, seed=1234, max_steps=None, F=None):
    r"""
    Use Pollard's rho method to try to extract a nontrivial factor
    of ``n``. The returned factor may be a composite number. If no
    factor is found, ``None`` is returned.

    The algorithm generates pseudo-random values of x with a generator
    function, replacing x with F(x). If F is not supplied then the
    function x**2 + ``a`` is used. The first value supplied to F(x) is ``s``.
    Upon failure (if ``retries`` is > 0) a new ``a`` and ``s`` will be
    supplied; the ``a`` will be ignored if F was supplied.

    The sequence of numbers generated by such functions generally have a
    a lead-up to some number and then loop around back to that number and
    begin to repeat the sequence, e.g. 1, 2, 3, 4, 5, 3, 4, 5 -- this leader
    and loop look a bit like the Greek letter rho, and thus the name, 'rho'.

    For a given function, very different leader-loop values can be obtained
    so it is a good idea to allow for retries:

    >>> from sympy.ntheory.generate import cycle_length
    >>> n = 16843009
    >>> F = lambda x:(2048*pow(x, 2, n) + 32767) % n
    >>> for s in range(5):
    ...     print('loop length = %4i; leader length = %3i' % next(cycle_length(F, s)))
    ...
    loop length = 2489; leader length =  43
    loop length =   78; leader length = 121
    loop length = 1482; leader length = 100
    loop length = 1482; leader length = 286
    loop length = 1482; leader length = 101

    Here is an explicit example where there is a three element leadup to
    a sequence of 3 numbers (11, 14, 4) that then repeat:

    >>> x=2
    >>> for i in range(9):
    ...     print(x)
    ...     x=(x**2+12)%17
    ...
    2
    16
    13
    11
    14
    4
    11
    14
    4
    >>> next(cycle_length(lambda x: (x**2+12)%17, 2))
    (3, 3)
    >>> list(cycle_length(lambda x: (x**2+12)%17, 2, values=True))
    [2, 16, 13, 11, 14, 4]

    Instead of checking the differences of all generated values for a gcd
    with n, only the kth and 2*kth numbers are checked, e.g. 1st and 2nd,
    2nd and 4th, 3rd and 6th until it has been detected that the loop has been
    traversed. Loops may be many thousands of steps long before rho finds a
    factor or reports failure. If ``max_steps`` is specified, the iteration
    is cancelled with a failure after the specified number of steps.

    Examples
    ========

    >>> from sympy import pollard_rho
    >>> n=16843009
    >>> F=lambda x:(2048*pow(x,2,n) + 32767) % n
    >>> pollard_rho(n, F=F)
    257

    Use the default setting with a bad value of ``a`` and no retries:

    >>> pollard_rho(n, a=n-2, retries=0)

    If retries is > 0 then perhaps the problem will correct itself when
    new values are generated for a:

    >>> pollard_rho(n, a=n-2, retries=1)
    257

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 229-231

    """
    n = int(n)
    if n < 5:
        raise ValueError('pollard_rho should receive n > 4')
    randint = _randint(seed + retries)
    V = s
    for i in range(retries + 1):
        U = V
        if not F:
            F = lambda x: (pow(x, 2, n) + a) % n
        j = 0
        while 1:
            if max_steps and (j > max_steps):
                break
            j += 1
            U = F(U)
            V = F(F(V))  # V is 2x further along than U
            g = gcd(U - V, n)
            if g == 1:
                continue
            if g == n:
                break
            return int(g)
        V = randint(0, n - 1)
        a = randint(1, n - 3)  # for x**2 + a, a%n should not be 0 or -2
        F = None
    return None


def pollard_pm1(n, B=10, a=2, retries=0, seed=1234):
    """
    Use Pollard's p-1 method to try to extract a nontrivial factor
    of ``n``. Either a divisor (perhaps composite) or ``None`` is returned.

    The value of ``a`` is the base that is used in the test gcd(a**M - 1, n).
    The default is 2.  If ``retries`` > 0 then if no factor is found after the
    first attempt, a new ``a`` will be generated randomly (using the ``seed``)
    and the process repeated.

    Note: the value of M is lcm(1..B) = reduce(ilcm, range(2, B + 1)).

    A search is made for factors next to even numbers having a power smoothness
    less than ``B``. Choosing a larger B increases the likelihood of finding a
    larger factor but takes longer. Whether a factor of n is found or not
    depends on ``a`` and the power smoothness of the even number just less than
    the factor p (hence the name p - 1).

    Although some discussion of what constitutes a good ``a`` some
    descriptions are hard to interpret. At the modular.math site referenced
    below it is stated that if gcd(a**M - 1, n) = N then a**M % q**r is 1
    for every prime power divisor of N. But consider the following:

        >>> from sympy.ntheory.factor_ import smoothness_p, pollard_pm1
        >>> n=257*1009
        >>> smoothness_p(n)
        (-1, [(257, (1, 2, 256)), (1009, (1, 7, 16))])

    So we should (and can) find a root with B=16:

        >>> pollard_pm1(n, B=16, a=3)
        1009

    If we attempt to increase B to 256 we find that it does not work:

        >>> pollard_pm1(n, B=256)
        >>>

    But if the value of ``a`` is changed we find that only multiples of
    257 work, e.g.:

        >>> pollard_pm1(n, B=256, a=257)
        1009

    Checking different ``a`` values shows that all the ones that did not
    work had a gcd value not equal to ``n`` but equal to one of the
    factors:

        >>> from sympy import ilcm, igcd, factorint, Pow
        >>> M = 1
        >>> for i in range(2, 256):
        ...     M = ilcm(M, i)
        ...
        >>> set([igcd(pow(a, M, n) - 1, n) for a in range(2, 256) if
        ...      igcd(pow(a, M, n) - 1, n) != n])
        {1009}

    But does aM % d for every divisor of n give 1?

        >>> aM = pow(255, M, n)
        >>> [(d, aM%Pow(*d.args)) for d in factorint(n, visual=True).args]
        [(257**1, 1), (1009**1, 1)]

    No, only one of them. So perhaps the principle is that a root will
    be found for a given value of B provided that:

    1) the power smoothness of the p - 1 value next to the root
       does not exceed B
    2) a**M % p != 1 for any of the divisors of n.

    By trying more than one ``a`` it is possible that one of them
    will yield a factor.

    Examples
    ========

    With the default smoothness bound, this number cannot be cracked:

        >>> from sympy.ntheory import pollard_pm1
        >>> pollard_pm1(21477639576571)

    Increasing the smoothness bound helps:

        >>> pollard_pm1(21477639576571, B=2000)
        4410317

    Looking at the smoothness of the factors of this number we find:

        >>> from sympy.ntheory.factor_ import smoothness_p, factorint
        >>> print(smoothness_p(21477639576571, visual=1))
        p**i=4410317**1 has p-1 B=1787, B-pow=1787
        p**i=4869863**1 has p-1 B=2434931, B-pow=2434931

    The B and B-pow are the same for the p - 1 factorizations of the divisors
    because those factorizations had a very large prime factor:

        >>> factorint(4410317 - 1)
        {2: 2, 617: 1, 1787: 1}
        >>> factorint(4869863-1)
        {2: 1, 2434931: 1}

    Note that until B reaches the B-pow value of 1787, the number is not cracked;

        >>> pollard_pm1(21477639576571, B=1786)
        >>> pollard_pm1(21477639576571, B=1787)
        4410317

    The B value has to do with the factors of the number next to the divisor,
    not the divisors themselves. A worst case scenario is that the number next
    to the factor p has a large prime divisisor or is a perfect power. If these
    conditions apply then the power-smoothness will be about p/2 or p. The more
    realistic is that there will be a large prime factor next to p requiring
    a B value on the order of p/2. Although primes may have been searched for
    up to this level, the p/2 is a factor of p - 1, something that we do not
    know. The modular.math reference below states that 15% of numbers in the
    range of 10**15 to 15**15 + 10**4 are 10**6 power smooth so a B of 10**6
    will fail 85% of the time in that range. From 10**8 to 10**8 + 10**3 the
    percentages are nearly reversed...but in that range the simple trial
    division is quite fast.

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 236-238
    .. [2] https://web.archive.org/web/20150716201437/http://modular.math.washington.edu/edu/2007/spring/ent/ent-html/node81.html
    .. [3] https://www.cs.toronto.edu/~yuvalf/Factorization.pdf
    """

    n = int(n)
    if n < 4 or B < 3:
        raise ValueError('pollard_pm1 should receive n > 3 and B > 2')
    randint = _randint(seed + B)

    # computing a**lcm(1,2,3,..B) % n for B > 2
    # it looks weird, but it's right: primes run [2, B]
    # and the answer's not right until the loop is done.
    for i in range(retries + 1):
        aM = a
        for p in sieve.primerange(2, B + 1):
            e = int(math.log(B, p))
            aM = pow(aM, pow(p, e), n)
        g = gcd(aM - 1, n)
        if 1 < g < n:
            return int(g)

        # get a new a:
        # since the exponent, lcm(1..B), is even, if we allow 'a' to be 'n-1'
        # then (n - 1)**even % n will be 1 which will give a g of 0 and 1 will
        # give a zero, too, so we set the range as [2, n-2]. Some references
        # say 'a' should be coprime to n, but either will detect factors.
        a = randint(2, n - 2)


def _trial(factors, n, candidates, verbose=False):
    """
    Helper function for integer factorization. Trial factors ``n`
    against all integers given in the sequence ``candidates``
    and updates the dict ``factors`` in-place. Returns the reduced
    value of ``n`` and a flag indicating whether any factors were found.
    """
    if verbose:
        factors0 = list(factors.keys())
    nfactors = len(factors)
    for d in candidates:
        if n % d == 0:
            if n != d:
                factor_cache[n] = d
            n, m = remove(n // d, d)
            factors[d] = m + 1
    if verbose:
        for k in sorted(set(factors).difference(set(factors0))):
            print(factor_msg % (k, factors[k]))
    return int(n), len(factors) != nfactors


def _check_termination(factors, n, limit, use_trial, use_rho, use_pm1,
                       verbose, next_p):
    """
    Helper function for integer factorization. Checks if ``n``
    is a prime or a perfect power, and in those cases updates the factorization.
    """
    if verbose:
        print('Check for termination')
    if n == 1:
        if verbose:
            print(complete_msg)
        return True
    if n < next_p**2 or isprime(n):
        factor_cache[n] = n
        factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return True

    # since we've already been factoring there is no need to do
    # simultaneous factoring with the power check
    p = _perfect_power(n, next_p)
    if not p:
        return False
    base, exp = p
    if base < next_p**2 or isprime(base):
        factor_cache[n] = base
        factors[base] = exp
    else:
        facs = factorint(base, limit, use_trial, use_rho, use_pm1,
                         verbose=False)
        for b, e in facs.items():
            if verbose:
                print(factor_msg % (b, e))
            factors[b] = exp*e
    if verbose:
        print(complete_msg)
    return True


trial_int_msg = "Trial division with ints [%i ... %i] and fail_max=%i"
trial_msg = "Trial division with primes [%i ... %i]"
rho_msg = "Pollard's rho with retries %i, max_steps %i and seed %i"
pm1_msg = "Pollard's p-1 with smoothness bound %i and seed %i"
ecm_msg = "Elliptic Curve with B1 bound %i, B2 bound %i, num_curves %i"
factor_msg = '\t%i ** %i'
fermat_msg = 'Close factors satisfying Fermat condition found.'
complete_msg = 'Factorization is complete.'


def _factorint_small(factors, n, limit, fail_max, next_p=2):
    """
    Return the value of n and either a 0 (indicating that factorization up
    to the limit was complete) or else the next near-prime that would have
    been tested.

    Factoring stops if there are fail_max unsuccessful tests in a row.

    If factors of n were found they will be in the factors dictionary as
    {factor: multiplicity} and the returned value of n will have had those
    factors removed. The factors dictionary is modified in-place.

    """

    def done(n, d):
        """return n, d if the sqrt(n) was not reached yet, else
           n, 0 indicating that factoring is done.
        """
        if d*d <= n:
            return n, d
        return n, 0

    limit2 = limit**2
    threshold2 = min(n, limit2)

    if next_p < 3:
        if not n & 1:
            m = bit_scan1(n)
            factors[2] = m
            n >>= m
            threshold2 = min(n, limit2)
        next_p = 3
        if threshold2 < 9: # next_p**2 = 9
            return done(n, next_p)

    if next_p < 5:
        if not n % 3:
            n //= 3
            m = 1
            while not n % 3:
                n //= 3
                m += 1
                if m == 20:
                    n, mm = remove(n, 3)
                    m += mm
                    break
            factors[3] = m
            threshold2 = min(n, limit2)
        next_p = 5
        if threshold2 < 25: # next_p**2 = 25
            return done(n, next_p)

    # Because of the order of checks, starting from `min_p = 6k+5`,
    # useless checks are caused.
    # We want to calculate
    # next_p += [-1, -2, 3, 2, 1, 0][next_p % 6]
    p6 = next_p % 6
    next_p += (-1 if p6 < 2 else 5) - p6

    fails = 0
    while fails < fail_max:
        # next_p % 6 == 5
        if n % next_p:
            fails += 1
        else:
            n //= next_p
            m = 1
            while not n % next_p:
                n //= next_p
                m += 1
                if m == 20:
                    n, mm = remove(n, next_p)
                    m += mm
                    break
            factors[next_p] = m
            fails = 0
            threshold2 = min(n, limit2)
        next_p += 2
        if threshold2 < next_p**2:
            return done(n, next_p)

        # next_p % 6 == 1
        if n % next_p:
            fails += 1
        else:
            n //= next_p
            m = 1
            while not n % next_p:
                n //= next_p
                m += 1
                if m == 20:
                    n, mm = remove(n, next_p)
                    m += mm
                    break
            factors[next_p] = m
            fails = 0
            threshold2 = min(n, limit2)
        next_p += 4
        if threshold2 < next_p**2:
            return done(n, next_p)
    return done(n, next_p)


def factorint(n, limit=None, use_trial=True, use_rho=True, use_pm1=True,
              use_ecm=True, verbose=False, visual=None, multiple=False):
    r"""
    Given a positive integer ``n``, ``factorint(n)`` returns a dict containing
    the prime factors of ``n`` as keys and their respective multiplicities
    as values. For example:

    >>> from sympy.ntheory import factorint
    >>> factorint(2000)    # 2000 = (2**4) * (5**3)
    {2: 4, 5: 3}
    >>> factorint(65537)   # This number is prime
    {65537: 1}

    For input less than 2, factorint behaves as follows:

        - ``factorint(1)`` returns the empty factorization, ``{}``
        - ``factorint(0)`` returns ``{0:1}``
        - ``factorint(-n)`` adds ``-1:1`` to the factors and then factors ``n``

    Partial Factorization:

    If ``limit`` (> 3) is specified, the search is stopped after performing
    trial division up to (and including) the limit (or taking a
    corresponding number of rho/p-1 steps). This is useful if one has
    a large number and only is interested in finding small factors (if
    any). Note that setting a limit does not prevent larger factors
    from being found early; it simply means that the largest factor may
    be composite. Since checking for perfect power is relatively cheap, it is
    done regardless of the limit setting.

    This number, for example, has two small factors and a huge
    semi-prime factor that cannot be reduced easily:

    >>> from sympy.ntheory import isprime
    >>> a = 1407633717262338957430697921446883
    >>> f = factorint(a, limit=10000)
    >>> f == {991: 1, int(202916782076162456022877024859): 1, 7: 1}
    True
    >>> isprime(max(f))
    False

    This number has a small factor and a residual perfect power whose
    base is greater than the limit:

    >>> factorint(3*101**7, limit=5)
    {3: 1, 101: 7}

    List of Factors:

    If ``multiple`` is set to ``True`` then a list containing the
    prime factors including multiplicities is returned.

    >>> factorint(24, multiple=True)
    [2, 2, 2, 3]

    Visual Factorization:

    If ``visual`` is set to ``True``, then it will return a visual
    factorization of the integer.  For example:

    >>> from sympy import pprint
    >>> pprint(factorint(4200, visual=True))
     3  1  2  1
    2 *3 *5 *7

    Note that this is achieved by using the evaluate=False flag in Mul
    and Pow. If you do other manipulations with an expression where
    evaluate=False, it may evaluate.  Therefore, you should use the
    visual option only for visualization, and use the normal dictionary
    returned by visual=False if you want to perform operations on the
    factors.

    You can easily switch between the two forms by sending them back to
    factorint:

    >>> from sympy import Mul
    >>> regular = factorint(1764); regular
    {2: 2, 3: 2, 7: 2}
    >>> pprint(factorint(regular))
     2  2  2
    2 *3 *7

    >>> visual = factorint(1764, visual=True); pprint(visual)
     2  2  2
    2 *3 *7
    >>> print(factorint(visual))
    {2: 2, 3: 2, 7: 2}

    If you want to send a number to be factored in a partially factored form
    you can do so with a dictionary or unevaluated expression:

    >>> factorint(factorint({4: 2, 12: 3})) # twice to toggle to dict form
    {2: 10, 3: 3}
    >>> factorint(Mul(4, 12, evaluate=False))
    {2: 4, 3: 1}

    The table of the output logic is:

        ====== ====== ======= =======
                       Visual
        ------ ----------------------
        Input  True   False   other
        ====== ====== ======= =======
        dict    mul    dict    mul
        n       mul    dict    dict
        mul     mul    dict    dict
        ====== ====== ======= =======

    Notes
    =====

    Algorithm:

    The function switches between multiple algorithms. Trial division
    quickly finds small factors (of the order 1-5 digits), and finds
    all large factors if given enough time. The Pollard rho and p-1
    algorithms are used to find large factors ahead of time; they
    will often find factors of the order of 10 digits within a few
    seconds:

    >>> factors = factorint(12345678910111213141516)
    >>> for base, exp in sorted(factors.items()):
    ...     print('%s %s' % (base, exp))
    ...
    2 2
    2507191691 1
    1231026625769 1

    Any of these methods can optionally be disabled with the following
    boolean parameters:

        - ``use_trial``: Toggle use of trial division
        - ``use_rho``: Toggle use of Pollard's rho method
        - ``use_pm1``: Toggle use of Pollard's p-1 method

    ``factorint`` also periodically checks if the remaining part is
    a prime number or a perfect power, and in those cases stops.

    For unevaluated factorial, it uses Legendre's formula(theorem).


    If ``verbose`` is set to ``True``, detailed progress is printed.

    See Also
    ========

    smoothness, smoothness_p, divisors

    """
    if isinstance(n, Dict):
        n = dict(n)
    if multiple:
        fac = factorint(n, limit=limit, use_trial=use_trial,
                           use_rho=use_rho, use_pm1=use_pm1,
                           verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One/p]*(-fac[p])
                               for p in sorted(fac)), [])
        return factorlist

    factordict = {}
    if visual and not isinstance(n, (Mul, dict)):
        factordict = factorint(n, limit=limit, use_trial=use_trial,
                               use_rho=use_rho, use_pm1=use_pm1,
                               verbose=verbose, visual=False)
    elif isinstance(n, Mul):
        factordict = {int(k): int(v) for k, v in
            n.as_powers_dict().items()}
    elif isinstance(n, dict):
        factordict = n
    if factordict and isinstance(n, (Mul, dict)):
        # check it
        for key in list(factordict.keys()):
            if isprime(key):
                continue
            e = factordict.pop(key)
            d = factorint(key, limit=limit, use_trial=use_trial, use_rho=use_rho,
                          use_pm1=use_pm1, verbose=verbose, visual=False)
            for k, v in d.items():
                if k in factordict:
                    factordict[k] += v*e
                else:
                    factordict[k] = v*e
    if visual or (type(n) is dict and
                  visual is not True and
                  visual is not False):
        if factordict == {}:
            return S.One
        if -1 in factordict:
            factordict.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False)
                     for i in sorted(factordict.items())])
        return Mul(*args, evaluate=False)
    elif isinstance(n, (dict, Mul)):
        return factordict

    assert use_trial or use_rho or use_pm1 or use_ecm

    from sympy.functions.combinatorial.factorials import factorial
    if isinstance(n, factorial):
        x = as_int(n.args[0])
        if x >= 20:
            factors = {}
            m = 2 # to initialize the if condition below
            for p in sieve.primerange(2, x + 1):
                if m > 1:
                    m, q = 0, x // p
                    while q != 0:
                        m += q
                        q //= p
                factors[p] = m
            if factors and verbose:
                for k in sorted(factors):
                    print(factor_msg % (k, factors[k]))
            if verbose:
                print(complete_msg)
            return factors
        else:
            # if n < 20!, direct computation is faster
            # since it uses a lookup table
            n = n.func(x)

    n = as_int(n)
    if limit:
        limit = int(limit)
        use_ecm = False

    # special cases
    if n < 0:
        factors = factorint(
            -n, limit=limit, use_trial=use_trial, use_rho=use_rho,
            use_pm1=use_pm1, verbose=verbose, visual=False)
        factors[-1] = 1
        return factors

    if limit and limit < 2:
        if n == 1:
            return {}
        return {n: 1}
    elif n < 10:
        # doing this we are assured of getting a limit > 2
        # when we have to compute it later
        return [{0: 1}, {}, {2: 1}, {3: 1}, {2: 2}, {5: 1},
                {2: 1, 3: 1}, {7: 1}, {2: 3}, {3: 2}][n]

    factors = {}

    # do simplistic factorization
    if verbose:
        sn = str(n)
        if len(sn) > 50:
            print('Factoring %s' % sn[:5] + \
                  '..(%i other digits)..' % (len(sn) - 10) + sn[-5:])
        else:
            print('Factoring', n)

    # this is the preliminary factorization for small factors
    # We want to guarantee that there are no small prime factors,
    # so we run even if `use_trial` is False.
    small = 2**15
    fail_max = 600
    small = min(small, limit or small)
    if verbose:
        print(trial_int_msg % (2, small, fail_max))
    n, next_p = _factorint_small(factors, n, small, fail_max)
    if factors and verbose:
        for k in sorted(factors):
            print(factor_msg % (k, factors[k]))
    if next_p == 0:
        if n > 1:
            factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return factors
    # Check if it exists in the cache
    while p := factor_cache.get(n):
        n, e = remove(n, p)
        factors[int(p)] = int(e)
    # first check if the simplistic run didn't finish
    # because of the limit and check for a perfect
    # power before exiting
    if limit and next_p > limit:
        if verbose:
            print('Exceeded limit:', limit)
        if _check_termination(factors, n, limit, use_trial,
                              use_rho, use_pm1, verbose, next_p):
            return factors
        if n > 1:
            factors[int(n)] = 1
        return factors
    if _check_termination(factors, n, limit, use_trial,
                          use_rho, use_pm1, verbose, next_p):
        return factors

    # continue with more advanced factorization methods
    # ...do a Fermat test since it's so easy and we need the
    # square root anyway. Finding 2 factors is easy if they are
    # "close enough." This is the big root equivalent of dividing by
    # 2, 3, 5.
    sqrt_n = isqrt(n)
    a = sqrt_n + 1
    # If `n % 4 == 1`, `a` must be odd for `a**2 - n` to be a square number.
    if (n % 4 == 1) ^ (a & 1):
        a += 1
    a2 = a**2
    b2 = a2 - n
    for _ in range(3):
        b, fermat = sqrtrem(b2)
        if not fermat:
            if verbose:
                print(fermat_msg)
            for r in [a - b, a + b]:
                facs = factorint(r, limit=limit, use_trial=use_trial,
                                 use_rho=use_rho, use_pm1=use_pm1,
                                 verbose=verbose)
                for k, v in facs.items():
                    factors[k] = factors.get(k, 0) + v
                factor_cache.add(n, facs)
            if verbose:
                print(complete_msg)
            return factors
        b2 += (a + 1) << 2  # equiv to (a + 2)**2 - n
        a += 2

    # these are the limits for trial division which will
    # be attempted in parallel with pollard methods
    low, high = next_p, 2*next_p

    # add 1 to make sure limit is reached in primerange calls
    _limit = (limit or sqrt_n) + 1
    iteration = 0
    while 1:
        high_ = min(high, _limit)

        # Trial division
        if use_trial:
            if verbose:
                print(trial_msg % (low, high_))
            ps = sieve.primerange(low, high_)
            n, found_trial = _trial(factors, n, ps, verbose)
            next_p = high_
            if found_trial and _check_termination(factors, n, limit, use_trial,
                                                  use_rho, use_pm1, verbose, next_p):
                return factors
        else:
            found_trial = False

        if high > _limit:
            if verbose:
                print('Exceeded limit:', _limit)
            if n > 1:
                factors[int(n)] = 1
            if verbose:
                print(complete_msg)
            return factors

        # Only used advanced methods when no small factors were found
        if not found_trial:
            # Pollard p-1
            if use_pm1:
                if verbose:
                    print(pm1_msg % (low, high_))
                c = pollard_pm1(n, B=low, seed=high_)
                if c:
                    if c < next_p**2 or isprime(c):
                        ps = [c]
                    else:
                        ps = factorint(c, limit=limit,
                                       use_trial=use_trial,
                                       use_rho=use_rho,
                                       use_pm1=use_pm1,
                                       use_ecm=use_ecm,
                                       verbose=verbose)
                    n, _ = _trial(factors, n, ps, verbose=False)
                    if _check_termination(factors, n, limit, use_trial,
                                          use_rho, use_pm1, verbose, next_p):
                        return factors

            # Pollard rho
            if use_rho:
                if verbose:
                    print(rho_msg % (1, low, high_))
                c = pollard_rho(n, retries=1, max_steps=low, seed=high_)
                if c:
                    if c < next_p**2 or isprime(c):
                        ps = [c]
                    else:
                        ps = factorint(c, limit=limit,
                                       use_trial=use_trial,
                                       use_rho=use_rho,
                                       use_pm1=use_pm1,
                                       use_ecm=use_ecm,
                                       verbose=verbose)
                    n, _ = _trial(factors, n, ps, verbose=False)
                    if _check_termination(factors, n, limit, use_trial,
                                          use_rho, use_pm1, verbose, next_p):
                        return factors
        # Use subexponential algorithms if use_ecm
        # Use pollard algorithms for finding small factors for 3 iterations
        # if after small factors the number of digits of n >= 25 then use ecm
        iteration += 1
        if use_ecm and iteration >= 3 and num_digits(n) >= 24:
            break
        low, high = high, high*2

    B1 = 10000
    B2 = 100*B1
    num_curves = 50
    while(1):
        if verbose:
            print(ecm_msg % (B1, B2, num_curves))
        factor = _ecm_one_factor(n, B1, B2, num_curves, seed=B1)
        if factor:
            if factor < next_p**2 or isprime(factor):
                ps = [factor]
            else:
                ps = factorint(factor, limit=limit,
                           use_trial=use_trial,
                           use_rho=use_rho,
                           use_pm1=use_pm1,
                           use_ecm=use_ecm,
                           verbose=verbose)
            n, _ = _trial(factors, n, ps, verbose=False)
            if _check_termination(factors, n, limit, use_trial,
                                  use_rho, use_pm1, verbose, next_p):
                return factors
        B1 *= 5
        B2 = 100*B1
        num_curves *= 4


def factorrat(rat, limit=None, use_trial=True, use_rho=True, use_pm1=True,
              verbose=False, visual=None, multiple=False):
    r"""
    Given a Rational ``r``, ``factorrat(r)`` returns a dict containing
    the prime factors of ``r`` as keys and their respective multiplicities
    as values. For example:

    >>> from sympy import factorrat, S
    >>> factorrat(S(8)/9)    # 8/9 = (2**3) * (3**-2)
    {2: 3, 3: -2}
    >>> factorrat(S(-1)/987)    # -1/789 = -1 * (3**-1) * (7**-1) * (47**-1)
    {-1: 1, 3: -1, 7: -1, 47: -1}

    Please see the docstring for ``factorint`` for detailed explanations
    and examples of the following keywords:

        - ``limit``: Integer limit up to which trial division is done
        - ``use_trial``: Toggle use of trial division
        - ``use_rho``: Toggle use of Pollard's rho method
        - ``use_pm1``: Toggle use of Pollard's p-1 method
        - ``verbose``: Toggle detailed printing of progress
        - ``multiple``: Toggle returning a list of factors or dict
        - ``visual``: Toggle product form of output
    """
    if multiple:
        fac = factorrat(rat, limit=limit, use_trial=use_trial,
                  use_rho=use_rho, use_pm1=use_pm1,
                  verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One/p]*(-fac[p])
                               for p, _ in sorted(fac.items(),
                                                        key=lambda elem: elem[0]
                                                        if elem[1] > 0
                                                        else 1/elem[0])), [])
        return factorlist

    f = factorint(rat.p, limit=limit, use_trial=use_trial,
                  use_rho=use_rho, use_pm1=use_pm1,
                  verbose=verbose).copy()
    f = defaultdict(int, f)
    for p, e in factorint(rat.q, limit=limit,
                          use_trial=use_trial,
                          use_rho=use_rho,
                          use_pm1=use_pm1,
                          verbose=verbose).items():
        f[p] += -e

    if len(f) > 1 and 1 in f:
        del f[1]
    if not visual:
        return dict(f)
    else:
        if -1 in f:
            f.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False)
                     for i in sorted(f.items())])
        return Mul(*args, evaluate=False)


def primefactors(n, limit=None, verbose=False, **kwargs):
    """Return a sorted list of n's prime factors, ignoring multiplicity
    and any composite factor that remains if the limit was set too low
    for complete factorization. Unlike factorint(), primefactors() does
    not return -1 or 0.

    Parameters
    ==========

    n : integer
    limit, verbose, **kwargs :
        Additional keyword arguments to be passed to ``factorint``.
        Since ``kwargs`` is new in version 1.13,
        ``limit`` and ``verbose`` are retained for compatibility purposes.

    Returns
    =======

    list(int) : List of prime numbers dividing ``n``

    Examples
    ========

    >>> from sympy.ntheory import primefactors, factorint, isprime
    >>> primefactors(6)
    [2, 3]
    >>> primefactors(-5)
    [5]

    >>> sorted(factorint(123456).items())
    [(2, 6), (3, 1), (643, 1)]
    >>> primefactors(123456)
    [2, 3, 643]

    >>> sorted(factorint(10000000001, limit=200).items())
    [(101, 1), (99009901, 1)]
    >>> isprime(99009901)
    False
    >>> primefactors(10000000001, limit=300)
    [101]

    See Also
    ========

    factorint, divisors

    """
    n = int(n)
    kwargs.update({"visual": None, "multiple": False,
                   "limit": limit, "verbose": verbose})
    factors = sorted(factorint(n=n, **kwargs).keys())
    # We want to calculate
    # s = [f for f in factors if isprime(f)]
    s = [f for f in factors[:-1:] if f not in [-1, 0, 1]]
    if factors and isprime(factors[-1]):
        s += [factors[-1]]
    return s


def _divisors(n, proper=False):
    """Helper function for divisors which generates the divisors.

    Parameters
    ==========

    n : int
        a nonnegative integer
    proper: bool
        If `True`, returns the generator that outputs only the proper divisor (i.e., excluding n).

    """
    if n <= 1:
        if not proper and n:
            yield 1
        return

    factordict = factorint(n)
    ps = sorted(factordict.keys())

    def rec_gen(n=0):
        if n == len(ps):
            yield 1
        else:
            pows = [1]
            for _ in range(factordict[ps[n]]):
                pows.append(pows[-1] * ps[n])
            yield from (p * q for q in rec_gen(n + 1) for p in pows)

    if proper:
        yield from (p for p in rec_gen() if p != n)
    else:
        yield from rec_gen()


def divisors(n, generator=False, proper=False):
    r"""
    Return all divisors of n sorted from 1..n by default.
    If generator is ``True`` an unordered generator is returned.

    The number of divisors of n can be quite large if there are many
    prime factors (counting repeated factors). If only the number of
    factors is desired use divisor_count(n).

    Examples
    ========

    >>> from sympy import divisors, divisor_count
    >>> divisors(24)
    [1, 2, 3, 4, 6, 8, 12, 24]
    >>> divisor_count(24)
    8

    >>> list(divisors(120, generator=True))
    [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 15, 30, 60, 120]

    Notes
    =====

    This is a slightly modified version of Tim Peters referenced at:
    https://stackoverflow.com/questions/1010381/python-factorization

    See Also
    ========

    primefactors, factorint, divisor_count
    """
    rv = _divisors(as_int(abs(n)), proper)
    return rv if generator else sorted(rv)


def divisor_count(n, modulus=1, proper=False):
    """
    Return the number of divisors of ``n``. If ``modulus`` is not 1 then only
    those that are divisible by ``modulus`` are counted. If ``proper`` is True
    then the divisor of ``n`` will not be counted.

    Examples
    ========

    >>> from sympy import divisor_count
    >>> divisor_count(6)
    4
    >>> divisor_count(6, 2)
    2
    >>> divisor_count(6, proper=True)
    3

    See Also
    ========

    factorint, divisors, totient, proper_divisor_count

    """

    if not modulus:
        return 0
    elif modulus != 1:
        n, r = divmod(n, modulus)
        if r:
            return 0
    if n == 0:
        return 0
    n = Mul(*[v + 1 for k, v in factorint(n).items() if k > 1])
    if n and proper:
        n -= 1
    return n


def proper_divisors(n, generator=False):
    """
    Return all divisors of n except n, sorted by default.
    If generator is ``True`` an unordered generator is returned.

    Examples
    ========

    >>> from sympy import proper_divisors, proper_divisor_count
    >>> proper_divisors(24)
    [1, 2, 3, 4, 6, 8, 12]
    >>> proper_divisor_count(24)
    7
    >>> list(proper_divisors(120, generator=True))
    [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 15, 30, 60]

    See Also
    ========

    factorint, divisors, proper_divisor_count

    """
    return divisors(n, generator=generator, proper=True)


def proper_divisor_count(n, modulus=1):
    """
    Return the number of proper divisors of ``n``.

    Examples
    ========

    >>> from sympy import proper_divisor_count
    >>> proper_divisor_count(6)
    3
    >>> proper_divisor_count(6, modulus=2)
    1

    See Also
    ========

    divisors, proper_divisors, divisor_count

    """
    return divisor_count(n, modulus=modulus, proper=True)


def _udivisors(n):
    """Helper function for udivisors which generates the unitary divisors.

    Parameters
    ==========

    n : int
        a nonnegative integer

    """
    if n <= 1:
        if n == 1:
            yield 1
        return

    factorpows = [p**e for p, e in factorint(n).items()]
    # We want to calculate
    # yield from (math.prod(s) for s in powersets(factorpows))
    for i in range(2**len(factorpows)):
        d = 1
        for k in range(i.bit_length()):
            if i & 1:
                d *= factorpows[k]
            i >>= 1
        yield d


def udivisors(n, generator=False):
    r"""
    Return all unitary divisors of n sorted from 1..n by default.
    If generator is ``True`` an unordered generator is returned.

    The number of unitary divisors of n can be quite large if there are many
    prime factors. If only the number of unitary divisors is desired use
    udivisor_count(n).

    Examples
    ========

    >>> from sympy.ntheory.factor_ import udivisors, udivisor_count
    >>> udivisors(15)
    [1, 3, 5, 15]
    >>> udivisor_count(15)
    4

    >>> sorted(udivisors(120, generator=True))
    [1, 3, 5, 8, 15, 24, 40, 120]

    See Also
    ========

    primefactors, factorint, divisors, divisor_count, udivisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Unitary_divisor
    .. [2] https://mathworld.wolfram.com/UnitaryDivisor.html

    """
    rv = _udivisors(as_int(abs(n)))
    return rv if generator else sorted(rv)


def udivisor_count(n):
    """
    Return the number of unitary divisors of ``n``.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.ntheory.factor_ import udivisor_count
    >>> udivisor_count(120)
    8

    See Also
    ========

    factorint, divisors, udivisors, divisor_count, totient

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """

    if n == 0:
        return 0
    return 2**len([p for p in factorint(n) if p > 1])


def _antidivisors(n):
    """Helper function for antidivisors which generates the antidivisors.

    Parameters
    ==========

    n : int
        a nonnegative integer

    """
    if n <= 2:
        return
    for d in _divisors(n):
        y = 2*d
        if n > y and n % y:
            yield y
    for d in _divisors(2*n-1):
        if n > d >= 2 and n % d:
            yield d
    for d in _divisors(2*n+1):
        if n > d >= 2 and n % d:
            yield d


def antidivisors(n, generator=False):
    r"""
    Return all antidivisors of n sorted from 1..n by default.

    Antidivisors [1]_ of n are numbers that do not divide n by the largest
    possible margin.  If generator is True an unordered generator is returned.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import antidivisors
    >>> antidivisors(24)
    [7, 16]

    >>> sorted(antidivisors(128, generator=True))
    [3, 5, 15, 17, 51, 85]

    See Also
    ========

    primefactors, factorint, divisors, divisor_count, antidivisor_count

    References
    ==========

    .. [1] definition is described in https://oeis.org/A066272/a066272a.html

    """
    rv = _antidivisors(as_int(abs(n)))
    return rv if generator else sorted(rv)


def antidivisor_count(n):
    """
    Return the number of antidivisors [1]_ of ``n``.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.ntheory.factor_ import antidivisor_count
    >>> antidivisor_count(13)
    4
    >>> antidivisor_count(27)
    5

    See Also
    ========

    factorint, divisors, antidivisors, divisor_count, totient

    References
    ==========

    .. [1] formula from https://oeis.org/A066272

    """

    n = as_int(abs(n))
    if n <= 2:
        return 0
    return divisor_count(2*n - 1) + divisor_count(2*n + 1) + \
        divisor_count(n) - divisor_count(n, 2) - 5

@deprecated("""\
The `sympy.ntheory.factor_.totient` has been moved to `sympy.functions.combinatorial.numbers.totient`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def totient(n):
    r"""
    Calculate the Euler totient function phi(n)

    .. deprecated:: 1.13

        The ``totient`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.totient`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``totient(n)`` or `\phi(n)` is the number of positive integers `\leq` n
    that are relatively prime to n.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> totient(1)
    1
    >>> totient(25)
    20
    >>> totient(45) == totient(5)*totient(9)
    True

    See Also
    ========

    divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html

    """
    from sympy.functions.combinatorial.numbers import totient as _totient
    return _totient(n)


@deprecated("""\
The `sympy.ntheory.factor_.reduced_totient` has been moved to `sympy.functions.combinatorial.numbers.reduced_totient`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def reduced_totient(n):
    r"""
    Calculate the Carmichael reduced totient function lambda(n)

    .. deprecated:: 1.13

        The ``reduced_totient`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.reduced_totient`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``reduced_totient(n)`` or `\lambda(n)` is the smallest m > 0 such that
    `k^m \equiv 1 \mod n` for all k relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import reduced_totient
    >>> reduced_totient(1)
    1
    >>> reduced_totient(8)
    2
    >>> reduced_totient(30)
    4

    See Also
    ========

    totient

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_function
    .. [2] https://mathworld.wolfram.com/CarmichaelFunction.html

    """
    from sympy.functions.combinatorial.numbers import reduced_totient as _reduced_totient
    return _reduced_totient(n)


@deprecated("""\
The `sympy.ntheory.factor_.divisor_sigma` has been moved to `sympy.functions.combinatorial.numbers.divisor_sigma`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def divisor_sigma(n, k=1):
    r"""
    Calculate the divisor function `\sigma_k(n)` for positive integer n

    .. deprecated:: 1.13

        The ``divisor_sigma`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.divisor_sigma`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k(n) = \prod_{i=1}^\omega (1+p_i^k+p_i^{2k}+\cdots
        + p_i^{m_ik}).

    Parameters
    ==========

    n : integer

    k : integer, optional
        power of divisors in the sum

        for k = 0, 1:
        ``divisor_sigma(n, 0)`` is equal to ``divisor_count(n)``
        ``divisor_sigma(n, 1)`` is equal to ``sum(divisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> divisor_sigma(18, 0)
    6
    >>> divisor_sigma(39, 1)
    56
    >>> divisor_sigma(12, 2)
    210
    >>> divisor_sigma(37)
    38

    See Also
    ========

    divisor_count, totient, divisors, factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """
    from sympy.functions.combinatorial.numbers import divisor_sigma as func_divisor_sigma
    return func_divisor_sigma(n, k)


def _divisor_sigma(n:int, k:int=1) -> int:
    r""" Calculate the divisor function `\sigma_k(n)` for positive integer n

    Parameters
    ==========

    n : int
        positive integer
    k : int
        nonnegative integer

    See Also
    ========

    sympy.functions.combinatorial.numbers.divisor_sigma

    """
    if k == 0:
        return math.prod(e + 1 for e in factorint(n).values())
    return math.prod((p**(k*(e + 1)) - 1)//(p**k - 1) for p, e in factorint(n).items())


def core(n, t=2):
    r"""
    Calculate core(n, t) = `core_t(n)` of a positive integer n

    ``core_2(n)`` is equal to the squarefree part of n

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        core_t(n) = \prod_{i=1}^\omega p_i^{m_i \mod t}.

    Parameters
    ==========

    n : integer

    t : integer
        core(n, t) calculates the t-th power free part of n

        ``core(n, 2)`` is the squarefree part of ``n``
        ``core(n, 3)`` is the cubefree part of ``n``

        Default for t is 2.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import core
    >>> core(24, 2)
    6
    >>> core(9424, 3)
    1178
    >>> core(379238)
    379238
    >>> core(15**11, 10)
    15

    See Also
    ========

    factorint, sympy.solvers.diophantine.diophantine.square_factor

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square-free_integer#Squarefree_core

    """

    n = as_int(n)
    t = as_int(t)
    if n <= 0:
        raise ValueError("n must be a positive integer")
    elif t <= 1:
        raise ValueError("t must be >= 2")
    else:
        y = 1
        for p, e in factorint(n).items():
            y *= p**(e % t)
        return y


@deprecated("""\
The `sympy.ntheory.factor_.udivisor_sigma` has been moved to `sympy.functions.combinatorial.numbers.udivisor_sigma`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def udivisor_sigma(n, k=1):
    r"""
    Calculate the unitary divisor function `\sigma_k^*(n)` for positive integer n

    .. deprecated:: 1.13

        The ``udivisor_sigma`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.udivisor_sigma`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    ``udivisor_sigma(n, k)`` is equal to ``sum([x**k for x in udivisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k^*(n) = \prod_{i=1}^\omega (1+ p_i^{m_ik}).

    Parameters
    ==========

    k : power of divisors in the sum

        for k = 0, 1:
        ``udivisor_sigma(n, 0)`` is equal to ``udivisor_count(n)``
        ``udivisor_sigma(n, 1)`` is equal to ``sum(udivisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import udivisor_sigma
    >>> udivisor_sigma(18, 0)
    4
    >>> udivisor_sigma(74, 1)
    114
    >>> udivisor_sigma(36, 3)
    47450
    >>> udivisor_sigma(111)
    152

    See Also
    ========

    divisor_count, totient, divisors, udivisors, udivisor_count, divisor_sigma,
    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """
    from sympy.functions.combinatorial.numbers import udivisor_sigma as _udivisor_sigma
    return _udivisor_sigma(n, k)


@deprecated("""\
The `sympy.ntheory.factor_.primenu` has been moved to `sympy.functions.combinatorial.numbers.primenu`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def primenu(n):
    r"""
    Calculate the number of distinct prime factors for a positive integer n.

    .. deprecated:: 1.13

        The ``primenu`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.primenu`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primenu(n)`` or `\nu(n)` is:

    .. math ::
        \nu(n) = k.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primenu
    >>> primenu(1)
    0
    >>> primenu(30)
    3

    See Also
    ========

    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html

    """
    from sympy.functions.combinatorial.numbers import primenu as _primenu
    return _primenu(n)


@deprecated("""\
The `sympy.ntheory.factor_.primeomega` has been moved to `sympy.functions.combinatorial.numbers.primeomega`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def primeomega(n):
    r"""
    Calculate the number of prime factors counting multiplicities for a
    positive integer n.

    .. deprecated:: 1.13

        The ``primeomega`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.primeomega`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primeomega(n)``  or `\Omega(n)` is:

    .. math ::
        \Omega(n) = \sum_{i=1}^k m_i.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primeomega
    >>> primeomega(1)
    0
    >>> primeomega(20)
    3

    See Also
    ========

    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html

    """
    from sympy.functions.combinatorial.numbers import primeomega as _primeomega
    return _primeomega(n)


def mersenne_prime_exponent(nth):
    """Returns the exponent ``i`` for the nth Mersenne prime (which
    has the form `2^i - 1`).

    Examples
    ========

    >>> from sympy.ntheory.factor_ import mersenne_prime_exponent
    >>> mersenne_prime_exponent(1)
    2
    >>> mersenne_prime_exponent(20)
    4423
    """
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; mersenne_prime_exponent(1) == 2")
    if n > 51:
        raise ValueError("There are only 51 perfect numbers; nth must be less than or equal to 51")
    return MERSENNE_PRIME_EXPONENTS[n - 1]


def is_perfect(n):
    """Returns True if ``n`` is a perfect number, else False.

    A perfect number is equal to the sum of its positive, proper divisors.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> from sympy.ntheory.factor_ import is_perfect, divisors
    >>> is_perfect(20)
    False
    >>> is_perfect(6)
    True
    >>> 6 == divisor_sigma(6) - 6 == sum(divisors(6)[:-1])
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PerfectNumber.html
    .. [2] https://en.wikipedia.org/wiki/Perfect_number

    """
    n = as_int(n)
    if n < 1:
        return False
    if n % 2 == 0:
        m = (n.bit_length() + 1) >> 1
        if (1 << (m - 1)) * ((1 << m) - 1) != n:
            # Even perfect numbers must be of the form `2^{m-1}(2^m-1)`
            return False
        return m in MERSENNE_PRIME_EXPONENTS or is_mersenne_prime(2**m - 1)

    # n is an odd integer
    if n < 10**2000:  # https://www.lirmm.fr/~ochem/opn/
        return False
    if n % 105 == 0:  # not divis by 105
        return False
    if all(n % m != r for m, r in [(12, 1), (468, 117), (324, 81)]):
        return False
    # there are many criteria that the factor structure of n
    # must meet; since we will have to factor it to test the
    # structure we will have the factors and can then check
    # to see whether it is a perfect number or not. So we
    # skip the structure checks and go straight to the final
    # test below.
    result = abundance(n) == 0
    if result:
        raise ValueError(filldedent('''In 1888, Sylvester stated: "
            ...a prolonged meditation on the subject has satisfied
            me that the existence of any one such [odd perfect number]
            -- its escape, so to say, from the complex web of conditions
            which hem it in on all sides -- would be little short of a
            miracle." I guess SymPy just found that miracle and it
            factors like this: %s''' % factorint(n)))
    return result


def abundance(n):
    """Returns the difference between the sum of the positive
    proper divisors of a number and the number.

    Examples
    ========

    >>> from sympy.ntheory import abundance, is_perfect, is_abundant
    >>> abundance(6)
    0
    >>> is_perfect(6)
    True
    >>> abundance(10)
    -2
    >>> is_abundant(10)
    False
    """
    return _divisor_sigma(n) - 2 * n


def is_abundant(n):
    """Returns True if ``n`` is an abundant number, else False.

    A abundant number is smaller than the sum of its positive proper divisors.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_abundant
    >>> is_abundant(20)
    True
    >>> is_abundant(15)
    False

    References
    ==========

    .. [1] https://mathworld.wolfram.com/AbundantNumber.html

    """
    n = as_int(n)
    if is_perfect(n):
        return False
    return n % 6 == 0 or bool(abundance(n) > 0)


def is_deficient(n):
    """Returns True if ``n`` is a deficient number, else False.

    A deficient number is greater than the sum of its positive proper divisors.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_deficient
    >>> is_deficient(20)
    False
    >>> is_deficient(15)
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DeficientNumber.html

    """
    n = as_int(n)
    if is_perfect(n):
        return False
    return bool(abundance(n) < 0)


def is_amicable(m, n):
    """Returns True if the numbers `m` and `n` are "amicable", else False.

    Amicable numbers are two different numbers so related that the sum
    of the proper divisors of each is equal to that of the other.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> from sympy.ntheory.factor_ import is_amicable
    >>> is_amicable(220, 284)
    True
    >>> divisor_sigma(220) == divisor_sigma(284)
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Amicable_numbers

    """
    return m != n and m + n == _divisor_sigma(m) == _divisor_sigma(n)


def is_carmichael(n):
    """ Returns True if the numbers `n` is Carmichael number, else False.

    Parameters
    ==========

    n : Integer

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_number
    .. [2] https://oeis.org/A002997

    """
    if n < 561:
        return False
    return n % 2 and not isprime(n) and \
           all(e == 1 and (n - 1) % (p - 1) == 0 for p, e in factorint(n).items())


def find_carmichael_numbers_in_range(x, y):
    """ Returns a list of the number of Carmichael in the range

    See Also
    ========

    is_carmichael

    """
    if 0 <= x <= y:
        if x % 2 == 0:
            return [i for i in range(x + 1, y, 2) if is_carmichael(i)]
        else:
            return [i for i in range(x, y, 2) if is_carmichael(i)]
    else:
        raise ValueError('The provided range is not valid. x and y must be non-negative integers and x <= y')


def find_first_n_carmichaels(n):
    """ Returns the first n Carmichael numbers.

    Parameters
    ==========

    n : Integer

    See Also
    ========

    is_carmichael

    """
    i = 561
    carmichaels = []

    while len(carmichaels) < n:
        if is_carmichael(i):
            carmichaels.append(i)
        i += 2

    return carmichaels


def dra(n, b):
    """
    Returns the additive digital root of a natural number ``n`` in base ``b``
    which is a single digit value obtained by an iterative process of summing
    digits, on each iteration using the result from the previous iteration to
    compute a digit sum.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import dra
    >>> dra(3110, 12)
    8

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Digital_root

    """

    num = abs(as_int(n))
    b = as_int(b)
    if b <= 1:
        raise ValueError("Base should be an integer greater than 1")

    if num == 0:
        return 0

    return (1 + (num - 1) % (b - 1))


def drm(n, b):
    """
    Returns the multiplicative digital root of a natural number ``n`` in a given
    base ``b`` which is a single digit value obtained by an iterative process of
    multiplying digits, on each iteration using the result from the previous
    iteration to compute the digit multiplication.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import drm
    >>> drm(9876, 10)
    0

    >>> drm(49, 10)
    8

    References
    ==========

    .. [1] https://mathworld.wolfram.com/MultiplicativeDigitalRoot.html

    """

    n = abs(as_int(n))
    b = as_int(b)
    if b <= 1:
        raise ValueError("Base should be an integer greater than 1")
    while n > b:
        mul = 1
        while n > 1:
            n, r = divmod(n, b)
            if r == 0:
                return 0
            mul *= r
        n = mul
    return n
<|endoftext|>from math import log

from sympy.core.random import _randint
from sympy.external.gmpy import gcd, invert, sqrt
from sympy.utilities.misc import as_int
from .generate import sieve, primerange
from .primetest import isprime


#----------------------------------------------------------------------------#
#                                                                            #
#                   Lenstra's Elliptic Curve Factorization                   #
#                                                                            #
#----------------------------------------------------------------------------#


class Point:
    """Montgomery form of Points in an elliptic curve.
    In this form, the addition and doubling of points
    does not need any y-coordinate information thus
    decreasing the number of operations.
    Using Montgomery form we try to perform point addition
    and doubling in least amount of multiplications.

    The elliptic curve used here is of the form
    (E : b*y**2*z = x**3 + a*x**2*z + x*z**2).
    The a_24 parameter is equal to (a + 2)/4.

    References
    ==========

    .. [1] Kris Gaj, Soonhak Kwon, Patrick Baier, Paul Kohlbrenner, Hoang Le, Mohammed Khaleeluddin, Ramakrishna Bachimanchi,
           Implementing the Elliptic Curve Method of Factoring in Reconfigurable Hardware,
           Cryptographic Hardware and Embedded Systems - CHES 2006 (2006), pp. 119-133,
           https://doi.org/10.1007/11894063_10
           https://www.hyperelliptic.org/tanja/SHARCS/talks06/Gaj.pdf

    """

    def __init__(self, x_cord, z_cord, a_24, mod):
        """
        Initial parameters for the Point class.

        Parameters
        ==========

        x_cord : X coordinate of the Point
        z_cord : Z coordinate of the Point
        a_24 : Parameter of the elliptic curve in Montgomery form
        mod : modulus
        """
        self.x_cord = x_cord
        self.z_cord = z_cord
        self.a_24 = a_24
        self.mod = mod

    def __eq__(self, other):
        """Two points are equal if X/Z of both points are equal
        """
        if self.a_24 != other.a_24 or self.mod != other.mod:
            return False
        return self.x_cord * other.z_cord % self.mod ==\
            other.x_cord * self.z_cord % self.mod

    def add(self, Q, diff):
        """
        Add two points self and Q where diff = self - Q. Moreover the assumption
        is self.x_cord*Q.x_cord*(self.x_cord - Q.x_cord) != 0. This algorithm
        requires 6 multiplications. Here the difference between the points
        is already known and using this algorithm speeds up the addition
        by reducing the number of multiplication required. Also in the
        mont_ladder algorithm is constructed in a way so that the difference
        between intermediate points is always equal to the initial point.
        So, we always know what the difference between the point is.


        Parameters
        ==========

        Q : point on the curve in Montgomery form
        diff : self - Q

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p2 = Point(13, 10, 7, 29)
        >>> p3 = p2.add(p1, p1)
        >>> p3.x_cord
        23
        >>> p3.z_cord
        17
        """
        u = (self.x_cord - self.z_cord)*(Q.x_cord + Q.z_cord)
        v = (self.x_cord + self.z_cord)*(Q.x_cord - Q.z_cord)
        add, subt = u + v, u - v
        x_cord = diff.z_cord * add * add % self.mod
        z_cord = diff.x_cord * subt * subt % self.mod
        return Point(x_cord, z_cord, self.a_24, self.mod)

    def double(self):
        """
        Doubles a point in an elliptic curve in Montgomery form.
        This algorithm requires 5 multiplications.

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p2 = p1.double()
        >>> p2.x_cord
        13
        >>> p2.z_cord
        10
        """
        u = pow(self.x_cord + self.z_cord, 2, self.mod)
        v = pow(self.x_cord - self.z_cord, 2, self.mod)
        diff = u - v
        x_cord = u*v % self.mod
        z_cord = diff*(v + self.a_24*diff) % self.mod
        return Point(x_cord, z_cord, self.a_24, self.mod)

    def mont_ladder(self, k):
        """
        Scalar multiplication of a point in Montgomery form
        using Montgomery Ladder Algorithm.
        A total of 11 multiplications are required in each step of this
        algorithm.

        Parameters
        ==========

        k : The positive integer multiplier

        Examples
        ========

        >>> from sympy.ntheory.ecm import Point
        >>> p1 = Point(11, 16, 7, 29)
        >>> p3 = p1.mont_ladder(3)
        >>> p3.x_cord
        23
        >>> p3.z_cord
        17
        """
        Q = self
        R = self.double()
        for i in bin(k)[3:]:
            if  i  == '1':
                Q = R.add(Q, self)
                R = R.double()
            else:
                R = Q.add(R, self)
                Q = Q.double()
        return Q


def _ecm_one_factor(n, B1=10000, B2=100000, max_curve=200, seed=None):
    """Returns one factor of n using
    Lenstra's 2 Stage Elliptic curve Factorization
    with Suyama's Parameterization. Here Montgomery
    arithmetic is used for fast computation of addition
    and doubling of points in elliptic curve.

    Explanation
    ===========

    This ECM method considers elliptic curves in Montgomery
    form (E : b*y**2*z = x**3 + a*x**2*z + x*z**2) and involves
    elliptic curve operations (mod N), where the elements in
    Z are reduced (mod N). Since N is not a prime, E over FF(N)
    is not really an elliptic curve but we can still do point additions
    and doubling as if FF(N) was a field.

    Stage 1 : The basic algorithm involves taking a random point (P) on an
    elliptic curve in FF(N). The compute k*P using Montgomery ladder algorithm.
    Let q be an unknown factor of N. Then the order of the curve E, |E(FF(q))|,
    might be a smooth number that divides k. Then we have k = l * |E(FF(q))|
    for some l. For any point belonging to the curve E, |E(FF(q))|*P = O,
    hence k*P = l*|E(FF(q))|*P. Thus kP.z_cord = 0 (mod q), and the unknownn
    factor of N (q) can be recovered by taking gcd(kP.z_cord, N).

    Stage 2 : This is a continuation of Stage 1 if k*P != O. The idea utilize
    the fact that even if kP != 0, the value of k might miss just one large
    prime divisor of |E(FF(q))|. In this case we only need to compute the
    scalar multiplication by p to get p*k*P = O. Here a second bound B2
    restrict the size of possible values of p.

    Parameters
    ==========

    n : Number to be Factored. Assume that it is a composite number.
    B1 : Stage 1 Bound. Must be an even number.
    B2 : Stage 2 Bound. Must be an even number.
    max_curve : Maximum number of curves generated

    Returns
    =======

    integer | None : a non-trivial divisor of ``n``. ``None`` if not found

    References
    ==========

    .. [1] Carl Pomerance, Richard Crandall, Prime Numbers: A Computational Perspective,
           2nd Edition (2005), page 344, ISBN:978-0387252827
    """
    randint = _randint(seed)

    # When calculating T, if (B1 - 2*D) is negative, it cannot be calculated.
    D = min(sqrt(B2), B1 // 2 - 1)
    sieve.extend(D)
    beta = [0] * D
    S = [0] * D
    k = 1
    for p in primerange(2, B1 + 1):
        k *= pow(p, int(log(B1, p)))

    # Pre-calculate the prime numbers to be used in stage 2.
    # Using the fact that the x-coordinates of point P and its
    # inverse -P coincide, the number of primes to be checked
    # in stage 2 can be reduced.
    deltas_list = []
    for r in range(B1 + 2*D, B2 + 2*D, 4*D):
        # d in deltas iff r+(2d+1) and/or r-(2d+1) is prime
        deltas = {abs(q - r) >> 1 for q in primerange(r - 2*D, r + 2*D)}
        deltas_list.append(list(deltas))

    for _ in range(max_curve):
        #Suyama's Parametrization
        sigma = randint(6, n - 1)
        u = (sigma**2 - 5) % n
        v = (4*sigma) % n
        u_3 = pow(u, 3, n)

        try:
            # We use the elliptic curve y**2 = x**3 + a*x**2 + x
            # where a = pow(v - u, 3, n)*(3*u + v)*invert(4*u_3*v, n) - 2
            # However, we do not declare a because it is more convenient
            # to use a24 = (a + 2)*invert(4, n) in the calculation.
            a24 = pow(v - u, 3, n)*(3*u + v)*invert(16*u_3*v, n) % n
        except ZeroDivisionError:
            #If the invert(16*u_3*v, n) doesn't exist (i.e., g != 1)
            g = gcd(2*u_3*v, n)
            #If g = n, try another curve
            if g == n:
                continue
            return g

        Q = Point(u_3, pow(v, 3, n), a24, n)
        Q = Q.mont_ladder(k)
        g = gcd(Q.z_cord, n)

        #Stage 1 factor
        if g != 1 and g != n:
            return g
        #Stage 1 failure. Q.z = 0, Try another curve
        elif g == n:
            continue

        #Stage 2 - Improved Standard Continuation
        S[0] = Q
        Q2 = Q.double()
        S[1] = Q2.add(Q, Q)
        beta[0] = (S[0].x_cord*S[0].z_cord) % n
        beta[1] = (S[1].x_cord*S[1].z_cord) % n
        for d in range(2, D):
            S[d] = S[d - 1].add(Q2, S[d - 2])
            beta[d] = (S[d].x_cord*S[d].z_cord) % n
        # i.e., S[i] = Q.mont_ladder(2*i + 1)

        g = 1
        W = Q.mont_ladder(4*D)
        T = Q.mont_ladder(B1 - 2*D)
        R = Q.mont_ladder(B1 + 2*D)
        for deltas in deltas_list:
            # R = Q.mont_ladder(r) where r in range(B1 + 2*D, B2 + 2*D, 4*D)
            alpha = (R.x_cord*R.z_cord) % n
            for delta in deltas:
                # We want to calculate
                # f = R.x_cord * S[delta].z_cord - S[delta].x_cord * R.z_cord
                f = (R.x_cord - S[delta].x_cord)*\
                    (R.z_cord + S[delta].z_cord) - alpha + beta[delta]
                g = (g*f) % n
            T, R = R, R.add(W, T)
        g = gcd(n, g)

        #Stage 2 Factor found
        if g != 1 and g != n:
            return g


def ecm(n, B1=10000, B2=100000, max_curve=200, seed=1234):
    """Performs factorization using Lenstra's Elliptic curve method.

    This function repeatedly calls ``_ecm_one_factor`` to compute the factors
    of n. First all the small factors are taken out using trial division.
    Then ``_ecm_one_factor`` is used to compute one factor at a time.

    Parameters
    ==========

    n : Number to be Factored
    B1 : Stage 1 Bound. Must be an even number.
    B2 : Stage 2 Bound. Must be an even number.
    max_curve : Maximum number of curves generated
    seed : Initialize pseudorandom generator

    Examples
    ========

    >>> from sympy.ntheory import ecm
    >>> ecm(25645121643901801)
    {5394769, 4753701529}
    >>> ecm(9804659461513846513)
    {4641991, 2112166839943}
    """
    from .factor_ import _perfect_power
    n = as_int(n)
    if B1 % 2 != 0 or B2 % 2 != 0:
        raise ValueError("both bounds must be even")
    TF_LIMIT = 100000
    factors = set()
    for prime in sieve.primerange(2, TF_LIMIT):
        if n % prime == 0:
            factors.add(prime)
            while(n % prime == 0):
                n //= prime

    queue = []
    def check(m):
        if isprime(m):
            factors.add(m)
            return
        if result := _perfect_power(m, TF_LIMIT):
            return check(result[0])
        queue.append(m)
    check(n)
    while queue:
        n = queue.pop()
        factor = _ecm_one_factor(n, B1, B2, max_curve, seed)
        if factor is None:
            raise ValueError("Increase the bounds")
        check(factor)
        check(n // factor)
    return factors
<|endoftext|>from __future__ import annotations

from sympy.external.gmpy import (gcd, lcm, invert, sqrt, jacobi,
                                 bit_scan1, remove)
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence, gf_csolve
from .primetest import isprime
from .generate import primerange
from .factor_ import factorint, _perfect_power
from .modular import crt
from sympy.utilities.decorator import deprecated
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int
from sympy.utilities.iterables import iproduct
from sympy.core.random import _randint, randint

from itertools import product


def n_order(a, n):
    r""" Returns the order of ``a`` modulo ``n``.

    Explanation
    ===========

    The order of ``a`` modulo ``n`` is the smallest integer
    ``k`` such that `a^k` leaves a remainder of 1 with ``n``.

    Parameters
    ==========

    a : integer
    n : integer, n > 1. a and n should be relatively prime

    Returns
    =======

    int : the order of ``a`` modulo ``n``

    Raises
    ======

    ValueError
        If `n \le 1` or `\gcd(a, n) \neq 1`.
        If ``a`` or ``n`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory import n_order
    >>> n_order(3, 7)
    6
    >>> n_order(4, 7)
    3

    See Also
    ========

    is_primitive_root
        We say that ``a`` is a primitive root of ``n``
        when the order of ``a`` modulo ``n`` equals ``totient(n)``

    """
    a, n = as_int(a), as_int(n)
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    a = a % n
    # Trivial
    if a == 1:
        return 1
    if gcd(a, n) != 1:
        raise ValueError("The two numbers should be relatively prime")
    a_order = 1
    for p, e in factorint(n).items():
        pe = p**e
        pe_order = (p - 1) * p**(e - 1)
        factors = factorint(p - 1)
        if e > 1:
            factors[p] = e - 1
        order = 1
        for px, ex in factors.items():
            x = pow(a, pe_order // px**ex, pe)
            while x != 1:
                x = pow(x, px, pe)
                order *= px
        a_order = lcm(a_order, order)
    return int(a_order)


def _primitive_root_prime_iter(p):
    r""" Generates the primitive roots for a prime ``p``.

    Explanation
    ===========

    The primitive roots generated are not necessarily sorted.
    However, the first one is the smallest primitive root.

    Find the element whose order is ``p-1`` from the smaller one.
    If we can find the first primitive root ``g``, we can use the following theorem.

    .. math ::
        \operatorname{ord}(g^k) = \frac{\operatorname{ord}(g)}{\gcd(\operatorname{ord}(g), k)}

    From the assumption that `\operatorname{ord}(g)=p-1`,
    it is a necessary and sufficient condition for
    `\operatorname{ord}(g^k)=p-1` that `\gcd(p-1, k)=1`.

    Parameters
    ==========

    p : odd prime

    Yields
    ======

    int
        the primitive roots of ``p``

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_iter
    >>> sorted(_primitive_root_prime_iter(19))
    [2, 3, 10, 13, 14, 15]

    References
    ==========

    .. [1] W. Stein "Elementary Number Theory" (2011), page 44

    """
    if p == 3:
        yield 2
        return
    # Let p = +-1 (mod 4a). Legendre symbol (a/p) = 1, so `a` is not the primitive root.
    # Corollary : If p = +-1 (mod 8), then 2 is not the primitive root of p.
    g_min = 3 if p % 8 in [1, 7] else 2
    if p < 41:
        # small case
        g = 5 if p == 23 else g_min
    else:
        v = [(p - 1) // i for i in factorint(p - 1).keys()]
        for g in range(g_min, p):
            if all(pow(g, pw, p) != 1 for pw in v):
                break
    yield g
    # g**k is the primitive root of p iff gcd(p - 1, k) = 1
    for k in range(3, p, 2):
        if gcd(p - 1, k) == 1:
            yield pow(g, k, p)


def _primitive_root_prime_power_iter(p, e):
    r""" Generates the primitive roots of `p^e`.

    Explanation
    ===========

    Let ``g`` be the primitive root of ``p``.
    If `g^{p-1} \not\equiv 1 \pmod{p^2}`, then ``g`` is primitive root of `p^e`.
    Thus, if we find a primitive root ``g`` of ``p``,
    then `g, g+p, g+2p, \ldots, g+(p-1)p` are primitive roots of `p^2` except one.
    That one satisfies `\hat{g}^{p-1} \equiv 1 \pmod{p^2}`.
    If ``h`` is the primitive root of `p^2`,
    then `h, h+p^2, h+2p^2, \ldots, h+(p^{e-2}-1)p^e` are primitive roots of `p^e`.

    Parameters
    ==========

    p : odd prime
    e : positive integer

    Yields
    ======

    int
        the primitive roots of `p^e`

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power_iter
    >>> sorted(_primitive_root_prime_power_iter(5, 2))
    [2, 3, 8, 12, 13, 17, 22, 23]

    """
    if e == 1:
        yield from _primitive_root_prime_iter(p)
    else:
        p2 = p**2
        for g in _primitive_root_prime_iter(p):
            t = (g - pow(g, 2 - p, p2)) % p2
            for k in range(0, p2, p):
                if k != t:
                    yield from (g + k + m for m in range(0, p**e, p2))


def _primitive_root_prime_power2_iter(p, e):
    r""" Generates the primitive roots of `2p^e`.

    Explanation
    ===========

    If ``g`` is the primitive root of ``p**e``,
    then the odd one of ``g`` and ``g+p**e`` is the primitive root of ``2*p**e``.

    Parameters
    ==========

    p : odd prime
    e : positive integer

    Yields
    ======

    int
        the primitive roots of `2p^e`

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power2_iter
    >>> sorted(_primitive_root_prime_power2_iter(5, 2))
    [3, 13, 17, 23, 27, 33, 37, 47]

    """
    for g in _primitive_root_prime_power_iter(p, e):
        if g % 2 == 1:
            yield g
        else:
            yield g + p**e


def primitive_root(p, smallest=True):
    r""" Returns a primitive root of ``p`` or None.

    Explanation
    ===========

    For the definition of primitive root,
    see the explanation of ``is_primitive_root``.

    The primitive root of ``p`` exist only for
    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).
    Now, if we know the primitive root of ``q``,
    we can calculate the primitive root of `q^e`,
    and if we know the primitive root of `q^e`,
    we can calculate the primitive root of `2q^e`.
    When there is no need to find the smallest primitive root,
    this property can be used to obtain a fast primitive root.
    On the other hand, when we want the smallest primitive root,
    we naively determine whether it is a primitive root or not.

    Parameters
    ==========

    p : integer, p > 1
    smallest : if True the smallest primitive root is returned or None

    Returns
    =======

    int | None :
        If the primitive root exists, return the primitive root of ``p``.
        If not, return None.

    Raises
    ======

    ValueError
        If `p \le 1` or ``p`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import primitive_root
    >>> primitive_root(19)
    2
    >>> primitive_root(21) is None
    True
    >>> primitive_root(50, smallest=False)
    27

    See Also
    ========

    is_primitive_root

    References
    ==========

    .. [1] W. Stein "Elementary Number Theory" (2011), page 44
    .. [2] P. Hackman "Elementary Number Theory" (2009), Chapter C

    """
    p = as_int(p)
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    if p <= 4:
        return p - 1
    p_even = p % 2 == 0
    if not p_even:
        q = p  # p is odd
    elif p % 4:
        q = p//2  # p had 1 factor of 2
    else:
        return None  # p had more than one factor of 2
    if isprime(q):
        e = 1
    else:
        m = _perfect_power(q, 3)
        if not m:
            return None
        q, e = m
        if not isprime(q):
            return None
    if not smallest:
        if p_even:
            return next(_primitive_root_prime_power2_iter(q, e))
        return next(_primitive_root_prime_power_iter(q, e))
    if p_even:
        for i in range(3, p, 2):
            if i % q and is_primitive_root(i, p):
                return i
    g = next(_primitive_root_prime_iter(q))
    if e == 1 or pow(g, q - 1, q**2) != 1:
        return g
    for i in range(g + 1, p):
        if i % q and is_primitive_root(i, p):
            return i


def is_primitive_root(a, p):
    r""" Returns True if ``a`` is a primitive root of ``p``.

    Explanation
    ===========

    ``a`` is said to be the primitive root of ``p`` if `\gcd(a, p) = 1` and
    `\phi(p)` is the smallest positive number s.t.

        `a^{\phi(p)} \equiv 1 \pmod{p}`.

    where `\phi(p)` is Euler's totient function.

    The primitive root of ``p`` exist only for
    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).
    Hence, if it is not such a ``p``, it returns False.
    To determine the primitive root, we need to know
    the prime factorization of ``q-1``.
    The hardness of the determination depends on this complexity.

    Parameters
    ==========

    a : integer
    p : integer, ``p`` > 1. ``a`` and ``p`` should be relatively prime

    Returns
    =======

    bool : If True, ``a`` is the primitive root of ``p``.

    Raises
    ======

    ValueError
        If `p \le 1` or `\gcd(a, p) \neq 1`.
        If ``a`` or ``p`` is not an integer.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> from sympy.ntheory import is_primitive_root, n_order
    >>> is_primitive_root(3, 10)
    True
    >>> is_primitive_root(9, 10)
    False
    >>> n_order(3, 10) == totient(10)
    True
    >>> n_order(9, 10) == totient(10)
    False

    See Also
    ========

    primitive_root

    """
    a, p = as_int(a), as_int(p)
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    a = a % p
    if gcd(a, p) != 1:
        raise ValueError("The two numbers should be relatively prime")
    # Primitive root of p exist only for
    # p = 2, 4, q**e, 2*q**e (q is odd prime)
    if p <= 4:
        # The primitive root is only p-1.
        return a == p - 1
    if p % 2:
        q = p  # p is odd
    elif p % 4:
        q = p//2  # p had 1 factor of 2
    else:
        return False  # p had more than one factor of 2
    if isprime(q):
        group_order = q - 1
        factors = factorint(q - 1).keys()
    else:
        m = _perfect_power(q, 3)
        if not m:
            return False
        q, e = m
        if not isprime(q):
            return False
        group_order = q**(e - 1)*(q - 1)
        factors = set(factorint(q - 1).keys())
        factors.add(q)
    return all(pow(a, group_order // prime, p) != 1 for prime in factors)


def _sqrt_mod_tonelli_shanks(a, p):
    """
    Returns the square root in the case of ``p`` prime with ``p == 1 (mod 8)``

    Assume that the root exists.

    Parameters
    ==========

    a : int
    p : int
        prime number. should be ``p % 8 == 1``

    Returns
    =======

    int : Generally, there are two roots, but only one is returned.
          Which one is returned is random.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_tonelli_shanks
    >>> _sqrt_mod_tonelli_shanks(2, 17) in [6, 11]
    True

    References
    ==========

    .. [1] Carl Pomerance, Richard Crandall, Prime Numbers: A Computational Perspective,
           2nd Edition (2005), page 101, ISBN:978-0387252827

    """
    s = bit_scan1(p - 1)
    t = p >> s
    # find a non-quadratic residue
    if p % 12 == 5:
        # Legendre symbol (3/p) == -1 if p % 12 in [5, 7]
        d = 3
    elif p % 5 in [2, 3]:
        # Legendre symbol (5/p) == -1 if p % 5 in [2, 3]
        d = 5
    else:
        while 1:
            d = randint(6, p - 1)
            if jacobi(d, p) == -1:
                break
    #assert legendre_symbol(d, p) == -1
    A = pow(a, t, p)
    D = pow(d, t, p)
    m = 0
    for i in range(s):
        adm = A*pow(D, m, p) % p
        adm = pow(adm, 2**(s - 1 - i), p)
        if adm % p == p - 1:
            m += 2**i
    #assert A*pow(D, m, p) % p == 1
    x = pow(a, (t + 1)//2, p)*pow(D, m//2, p) % p
    return x


def sqrt_mod(a, p, all_roots=False):
    """
    Find a root of ``x**2 = a mod p``.

    Parameters
    ==========

    a : integer
    p : positive integer
    all_roots : if True the list of roots is returned or None

    Notes
    =====

    If there is no root it is returned None; else the returned root
    is less or equal to ``p // 2``; in general is not the smallest one.
    It is returned ``p // 2`` only if it is the only root.

    Use ``all_roots`` only when it is expected that all the roots fit
    in memory; otherwise use ``sqrt_mod_iter``.

    Examples
    ========

    >>> from sympy.ntheory import sqrt_mod
    >>> sqrt_mod(11, 43)
    21
    >>> sqrt_mod(17, 32, True)
    [7, 9, 23, 25]
    """
    if all_roots:
        return sorted(sqrt_mod_iter(a, p))
    p = abs(as_int(p))
    halfp = p // 2
    x = None
    for r in sqrt_mod_iter(a, p):
        if r < halfp:
            return r
        elif r > halfp:
            return p - r
        else:
            x = r
    return x


def sqrt_mod_iter(a, p, domain=int):
    """
    Iterate over solutions to ``x**2 = a mod p``.

    Parameters
    ==========

    a : integer
    p : positive integer
    domain : integer domain, ``int``, ``ZZ`` or ``Integer``

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import sqrt_mod_iter
    >>> list(sqrt_mod_iter(11, 43))
    [21, 22]

    See Also
    ========

    sqrt_mod : Same functionality, but you want a sorted list or only one solution.

    """
    a, p = as_int(a), abs(as_int(p))
    v = []
    pv = []
    _product = product
    for px, ex in factorint(p).items():
        if a % px:
            # `len(rx)` is at most 4
            rx = _sqrt_mod_prime_power(a, px, ex)
        else:
            # `len(list(rx))` can be assumed to be large.
            # The `itertools.product` is disadvantageous in terms of memory usage.
            # It is also inferior to iproduct in speed if not all Cartesian products are needed.
            rx = _sqrt_mod1(a, px, ex)
            _product = iproduct
        if not rx:
            return
        v.append(rx)
        pv.append(px**ex)
    if len(v) == 1:
        yield from map(domain, v[0])
    else:
        mm, e, s = gf_crt1(pv, ZZ)
        for vx in _product(*v):
            yield domain(gf_crt2(vx, pv, mm, e, s, ZZ))


def _sqrt_mod_prime_power(a, p, k):
    """
    Find the solutions to ``x**2 = a mod p**k`` when ``a % p != 0``.
    If no solution exists, return ``None``.
    Solutions are returned in an ascending list.

    Parameters
    ==========

    a : integer
    p : prime number
    k : positive integer

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
    >>> _sqrt_mod_prime_power(11, 43, 1)
    [21, 22]

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 160
    .. [2] http://www.numbertheory.org/php/squareroot.html
    .. [3] [Gathen99]_
    """
    pk = p**k
    a = a % pk

    if p == 2:
        # see Ref.[2]
        if a % 8 != 1:
            return None
        # Trivial
        if k <= 3:
            return list(range(1, pk, 2))
        r = 1
        # r is one of the solutions to x**2 - a = 0 (mod 2**3).
        # Hensel lift them to solutions of x**2 - a = 0 (mod 2**k)
        # if r**2 - a = 0 mod 2**nx but not mod 2**(nx+1)
        # then r + 2**(nx - 1) is a root mod 2**(nx+1)
        for nx in range(3, k):
            if ((r**2 - a) >> nx) % 2:
                r += 1 << (nx - 1)
        # r is a solution of x**2 - a = 0 (mod 2**k), and
        # there exist other solutions -r, r+h, -(r+h), and these are all solutions.
        h = 1 << (k - 1)
        return sorted([r, pk - r, (r + h) % pk, -(r + h) % pk])

    # If the Legendre symbol (a/p) is not 1, no solution exists.
    if jacobi(a, p) != 1:
        return None
    if p % 4 == 3:
        res = pow(a, (p + 1) // 4, p)
    elif p % 8 == 5:
        res = pow(a, (p + 3) // 8, p)
        if pow(res, 2, p) != a % p:
            res = res * pow(2, (p - 1) // 4, p) % p
    else:
        res = _sqrt_mod_tonelli_shanks(a, p)
    if k > 1:
        # Hensel lifting with Newton iteration, see Ref.[3] chapter 9
        # with f(x) = x**2 - a; one has f'(a) != 0 (mod p) for p != 2
        px = p
        for _ in range(k.bit_length() - 1):
            px = px**2
            frinv = invert(2*res, px)
            res = (res - (res**2 - a)*frinv) % px
        if k & (k - 1): # If k is not a power of 2
            frinv = invert(2*res, pk)
            res = (res - (res**2 - a)*frinv) % pk
    return sorted([res, pk - res])


def _sqrt_mod1(a, p, n):
    """
    Find solution to ``x**2 == a mod p**n`` when ``a % p == 0``.
    If no solution exists, return ``None``.

    Parameters
    ==========

    a : integer
    p : prime number, p must divide a
    n : positive integer

    References
    ==========

    .. [1] http://www.numbertheory.org/php/squareroot.html
    """
    pn = p**n
    a = a % pn
    if a == 0:
        # case gcd(a, p**k) = p**n
        return range(0, pn, p**((n + 1) // 2))
    # case gcd(a, p**k) = p**r, r < n
    a, r = remove(a, p)
    if r % 2 == 1:
        return None
    res = _sqrt_mod_prime_power(a, p, n - r)
    if res is None:
        return None
    m = r // 2
    return (x for rx in res for x in range(rx*p**m, pn, p**(n - m)))


def is_quad_residue(a, p):
    """
    Returns True if ``a`` (mod ``p``) is in the set of squares mod ``p``,
    i.e a % p in set([i**2 % p for i in range(p)]).

    Parameters
    ==========

    a : integer
    p : positive integer

    Returns
    =======

    bool : If True, ``x**2 == a (mod p)`` has solution.

    Raises
    ======

    ValueError
        If ``a``, ``p`` is not integer.
        If ``p`` is not positive.

    Examples
    ========

    >>> from sympy.ntheory import is_quad_residue
    >>> is_quad_residue(21, 100)
    True

    Indeed, ``pow(39, 2, 100)`` would be 21.

    >>> is_quad_residue(21, 120)
    False

    That is, for any integer ``x``, ``pow(x, 2, 120)`` is not 21.

    If ``p`` is an odd
    prime, an iterative method is used to make the determination:

    >>> from sympy.ntheory import is_quad_residue
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]
    >>> [j for j in range(7) if is_quad_residue(j, 7)]
    [0, 1, 2, 4]

    See Also
    ========

    legendre_symbol, jacobi_symbol, sqrt_mod
    """
    a, p = as_int(a), as_int(p)
    if p < 1:
        raise ValueError('p must be > 0')
    a %= p
    if a < 2 or p < 3:
        return True
    # Since we want to compute the Jacobi symbol,
    # we separate p into the odd part and the rest.
    t = bit_scan1(p)
    if t:
        # The existence of a solution to a power of 2 is determined
        # using the logic of `p==2` in `_sqrt_mod_prime_power` and `_sqrt_mod1`.
        a_ = a % (1 << t)
        if a_:
            r = bit_scan1(a_)
            if r % 2 or (a_ >> r) & 6:
                return False
        p >>= t
        a %= p
        if a < 2 or p < 3:
            return True
    # If Jacobi symbol is -1 or p is prime, can be determined by Jacobi symbol only
    j = jacobi(a, p)
    if j == -1 or isprime(p):
        return j == 1
    # Checks if `x**2 = a (mod p)` has a solution
    for px, ex in factorint(p).items():
        if a % px:
            if jacobi(a, px) != 1:
                return False
        else:
            a_ = a % px**ex
            if a_ == 0:
                continue
            a_, r = remove(a_, px)
            if r % 2 or jacobi(a_, px) != 1:
                return False
    return True


def is_nthpow_residue(a, n, m):
    """
    Returns True if ``x**n == a (mod m)`` has solutions.

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76

    """
    a = a % m
    a, n, m = as_int(a), as_int(n), as_int(m)
    if m <= 0:
        raise ValueError('m must be > 0')
    if n < 0:
        raise ValueError('n must be >= 0')
    if n == 0:
        if m == 1:
            return False
        return a == 1
    if a == 0:
        return True
    if n == 1:
        return True
    if n == 2:
        return is_quad_residue(a, m)
    return all(_is_nthpow_residue_bign_prime_power(a, n, p, e)
               for p, e in factorint(m).items())


def _is_nthpow_residue_bign_prime_power(a, n, p, k):
    r"""
    Returns True if `x^n = a \pmod{p^k}` has solutions for `n > 2`.

    Parameters
    ==========

    a : positive integer
    n : integer, n > 2
    p : prime number
    k : positive integer

    """
    while a % p == 0:
        a %= pow(p, k)
        if not a:
            return True
        a, mu = remove(a, p)
        if mu % n:
            return False
        k -= mu
    if p != 2:
        f = p**(k - 1)*(p - 1) # f = totient(p**k)
        return pow(a, f // gcd(f, n), pow(p, k)) == 1
    if n & 1:
        return True
    c = min(bit_scan1(n) + 2, k)
    return a % pow(2, c) == 1


def _nthroot_mod1(s, q, p, all_roots):
    """
    Root of ``x**q = s mod p``, ``p`` prime and ``q`` divides ``p - 1``.
    Assume that the root exists.

    Parameters
    ==========

    s : integer
    q : integer, n > 2. ``q`` divides ``p - 1``.
    p : prime number
    all_roots : if False returns the smallest root, else the list of roots

    Returns
    =======

    list[int] | int :
        Root of ``x**q = s mod p``. If ``all_roots == True``,
        returned ascending list. otherwise, returned an int.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _nthroot_mod1
    >>> _nthroot_mod1(5, 3, 13, False)
    7
    >>> _nthroot_mod1(13, 4, 17, True)
    [3, 5, 12, 14]

    References
    ==========

    .. [1] A. M. Johnston, A Generalized qth Root Algorithm,
           ACM-SIAM Symposium on Discrete Algorithms (1999), pp. 929-930

    """
    g = next(_primitive_root_prime_iter(p))
    r = s
    for qx, ex in factorint(q).items():
        f = (p - 1) // qx**ex
        while f % qx == 0:
            f //= qx
        z = f*invert(-f, qx)
        x = (1 + z) // qx
        t = discrete_log(p, pow(r, f, p), pow(g, f*qx, p))
        for _ in range(ex):
            # assert t == discrete_log(p, pow(r, f, p), pow(g, f*qx, p))
            r = pow(r, x, p)*pow(g, -z*t % (p - 1), p) % p
            t //= qx
    res = [r]
    h = pow(g, (p - 1) // q, p)
    #assert pow(h, q, p) == 1
    hx = r
    for _ in range(q - 1):
        hx = (hx*h) % p
        res.append(hx)
    if all_roots:
        res.sort()
        return res
    return min(res)


def _nthroot_mod_prime_power(a, n, p, k):
    """ Root of ``x**n = a mod p**k``.

    Parameters
    ==========

    a : integer
    n : integer, n > 2
    p : prime number
    k : positive integer

    Returns
    =======

    list[int] :
        Ascending list of roots of ``x**n = a mod p**k``.
        If no solution exists, return ``[]``.

    """
    if not _is_nthpow_residue_bign_prime_power(a, n, p, k):
        return []
    a_mod_p = a % p
    if a_mod_p == 0:
        base_roots = [0]
    elif (p - 1) % n == 0:
        base_roots = _nthroot_mod1(a_mod_p, n, p, all_roots=True)
    else:
        # The roots of ``x**n - a = 0 (mod p)`` are roots of
        # ``gcd(x**n - a, x**(p - 1) - 1) = 0 (mod p)``
        pa = n
        pb = p - 1
        b = 1
        if pa < pb:
            a_mod_p, pa, b, pb = b, pb, a_mod_p, pa
        # gcd(x**pa - a, x**pb - b) = gcd(x**pb - b, x**pc - c)
        # where pc = pa % pb; c = b**-q * a mod p
        while pb:
            q, pc = divmod(pa, pb)
            c = pow(b, -q, p) * a_mod_p % p
            pa, pb = pb, pc
            a_mod_p, b = b, c
        if pa == 1:
            base_roots = [a_mod_p]
        elif pa == 2:
            base_roots = sqrt_mod(a_mod_p, p, all_roots=True)
        else:
            base_roots = _nthroot_mod1(a_mod_p, pa, p, all_roots=True)
    if k == 1:
        return base_roots
    a %= p**k
    tot_roots = set()
    for root in base_roots:
        diff = pow(root, n - 1, p)*n % p
        new_base = p
        if diff != 0:
            m_inv = invert(diff, p)
            for _ in range(k - 1):
                new_base *= p
                tmp = pow(root, n, new_base) - a
                tmp *= m_inv
                root = (root - tmp) % new_base
            tot_roots.add(root)
        else:
            roots_in_base = {root}
            for _ in range(k - 1):
                new_base *= p
                new_roots = set()
                for k_ in roots_in_base:
                    if pow(k_, n, new_base) != a % new_base:
                        continue
                    while k_ not in new_roots:
                        new_roots.add(k_)
                        k_ = (k_ + (new_base // p)) % new_base
                roots_in_base = new_roots
            tot_roots = tot_roots | roots_in_base
    return sorted(tot_roots)


def nthroot_mod(a, n, p, all_roots=False):
    """
    Find the solutions to ``x**n = a mod p``.

    Parameters
    ==========

    a : integer
    n : positive integer
    p : positive integer
    all_roots : if False returns the smallest root, else the list of roots

    Returns
    =======

        list[int] | int | None :
            solutions to ``x**n = a mod p``.
            The table of the output type is:

            ========== ========== ==========
            all_roots  has roots  Returns
            ========== ========== ==========
            True       Yes        list[int]
            True       No         []
            False      Yes        int
            False      No         None
            ========== ========== ==========

    Raises
    ======

        ValueError
            If ``a``, ``n`` or ``p`` is not integer.
            If ``n`` or ``p`` is not positive.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import nthroot_mod
    >>> nthroot_mod(11, 4, 19)
    8
    >>> nthroot_mod(11, 4, 19, True)
    [8, 11]
    >>> nthroot_mod(68, 3, 109)
    23

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76

    """
    a = a % p
    a, n, p = as_int(a), as_int(n), as_int(p)

    if n < 1:
        raise ValueError("n should be positive")
    if p < 1:
        raise ValueError("p should be positive")
    if n == 1:
        return [a] if all_roots else a
    if n == 2:
        return sqrt_mod(a, p, all_roots)
    base = []
    prime_power = []
    for q, e in factorint(p).items():
        tot_roots = _nthroot_mod_prime_power(a, n, q, e)
        if not tot_roots:
            return [] if all_roots else None
        prime_power.append(q**e)
        base.append(sorted(tot_roots))
    P, E, S = gf_crt1(prime_power, ZZ)
    ret = sorted(map(int, {gf_crt2(c, prime_power, P, E, S, ZZ)
                           for c in product(*base)}))
    if all_roots:
        return ret
    if ret:
        return ret[0]


def quadratic_residues(p) -> list[int]:
    """
    Returns the list of quadratic residues.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import quadratic_residues
    >>> quadratic_residues(7)
    [0, 1, 2, 4]
    """
    p = as_int(p)
    r = {pow(i, 2, p) for i in range(p // 2 + 1)}
    return sorted(r)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.legendre_symbol` has been moved to `sympy.functions.combinatorial.numbers.legendre_symbol`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def legendre_symbol(a, p):
    r"""
    Returns the Legendre symbol `(a / p)`.

    .. deprecated:: 1.13

        The ``legendre_symbol`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.legendre_symbol`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is
    defined as

    .. math ::
        \genfrac(){}{}{a}{p} = \begin{cases}
             0 & \text{if } p \text{ divides } a\\
             1 & \text{if } a \text{ is a quadratic residue modulo } p\\
            -1 & \text{if } a \text{ is a quadratic nonresidue modulo } p
        \end{cases}

    Parameters
    ==========

    a : integer
    p : odd prime

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import legendre_symbol
    >>> [legendre_symbol(i, 7) for i in range(7)]
    [0, 1, 1, -1, 1, -1, -1]
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]

    See Also
    ========

    is_quad_residue, jacobi_symbol

    """
    from sympy.functions.combinatorial.numbers import legendre_symbol as _legendre_symbol
    return _legendre_symbol(a, p)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.jacobi_symbol` has been moved to `sympy.functions.combinatorial.numbers.jacobi_symbol`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def jacobi_symbol(m, n):
    r"""
    Returns the Jacobi symbol `(m / n)`.

    .. deprecated:: 1.13

        The ``jacobi_symbol`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.jacobi_symbol`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol
    is defined as the product of the Legendre symbols corresponding to the
    prime factors of ``n``:

    .. math ::
        \genfrac(){}{}{m}{n} =
            \genfrac(){}{}{m}{p^{1}}^{\alpha_1}
            \genfrac(){}{}{m}{p^{2}}^{\alpha_2}
            ...
            \genfrac(){}{}{m}{p^{k}}^{\alpha_k}
            \text{ where } n =
                p_1^{\alpha_1}
                p_2^{\alpha_2}
                ...
                p_k^{\alpha_k}

    Like the Legendre symbol, if the Jacobi symbol `\genfrac(){}{}{m}{n} = -1`
    then ``m`` is a quadratic nonresidue modulo ``n``.

    But, unlike the Legendre symbol, if the Jacobi symbol
    `\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue
    modulo ``n``.

    Parameters
    ==========

    m : integer
    n : odd positive integer

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import jacobi_symbol, legendre_symbol
    >>> from sympy import S
    >>> jacobi_symbol(45, 77)
    -1
    >>> jacobi_symbol(60, 121)
    1

    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can
    be demonstrated as follows:

    >>> L = legendre_symbol
    >>> S(45).factors()
    {3: 2, 5: 1}
    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1
    True

    See Also
    ========

    is_quad_residue, legendre_symbol
    """
    from sympy.functions.combinatorial.numbers import jacobi_symbol as _jacobi_symbol
    return _jacobi_symbol(m, n)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.mobius` has been moved to `sympy.functions.combinatorial.numbers.mobius`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def mobius(n):
    """
    Mobius function maps natural number to {-1, 0, 1}

    .. deprecated:: 1.13

        The ``mobius`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.mobius`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.

    It is an important multiplicative function in number theory
    and combinatorics.  It has applications in mathematical series,
    algebraic number theory and also physics (Fermion operator has very
    concrete realization with Mobius Function model).

    Parameters
    ==========

    n : positive integer

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import mobius
    >>> mobius(13*7)
    1
    >>> mobius(1)
    1
    >>> mobius(13*7*5)
    -1
    >>> mobius(13**2)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_function
    .. [2] Thomas Koshy "Elementary Number Theory with Applications"

    """
    from sympy.functions.combinatorial.numbers import mobius as _mobius
    return _mobius(n)


def _discrete_log_trial_mul(n, a, b, order=None):
    """
    Trial multiplication algorithm for computing the discrete logarithm of
    ``a`` to the base ``b`` modulo ``n``.

    The algorithm finds the discrete logarithm using exhaustive search. This
    naive method is used as fallback algorithm of ``discrete_log`` when the
    group order is very small. The value ``n`` must be greater than 1.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_trial_mul
    >>> _discrete_log_trial_mul(41, 15, 7)
    3

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n
    if order is None:
        order = n
    x = 1
    for i in range(order):
        if x == a:
            return i
        x = x * b % n
    raise ValueError("Log does not exist")


def _discrete_log_shanks_steps(n, a, b, order=None):
    """
    Baby-step giant-step algorithm for computing the discrete logarithm of
    ``a`` to the base ``b`` modulo ``n``.

    The algorithm is a time-memory trade-off of the method of exhaustive
    search. It uses `O(sqrt(m))` memory, where `m` is the group order.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_shanks_steps
    >>> _discrete_log_shanks_steps(41, 15, 7)
    3

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n
    if order is None:
        order = n_order(b, n)
    m = sqrt(order) + 1
    T = {}
    x = 1
    for i in range(m):
        T[x] = i
        x = x * b % n
    z = pow(b, -m, n)
    x = a
    for i in range(m):
        if x in T:
            return i * m + T[x]
        x = x * z % n
    raise ValueError("Log does not exist")


def _discrete_log_pollard_rho(n, a, b, order=None, retries=10, rseed=None):
    """
    Pollard's Rho algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    It is a randomized algorithm with the same expected running time as
    ``_discrete_log_shanks_steps``, but requires a negligible amount of memory.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pollard_rho
    >>> _discrete_log_pollard_rho(227, 3**7, 3)
    7

    See Also
    ========
