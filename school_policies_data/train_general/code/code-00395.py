    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for minimum value needs to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the minimum have to be checked.
        If unspecified, then the global minimum is returned.

    Returns
    =======

    number
        Minimum value of the function in the given domain.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, sin, cos, minimum
    >>> x = Symbol('x')

    >>> f = x**2 + 2*x + 5
    >>> minimum(f, x, S.Reals)
    4

    >>> minimum(sin(x), x, Interval(2, 3))
    sin(3)

    >>> minimum(sin(x)*cos(x), x)
    -1/2

    """
    if isinstance(symbol, Symbol):
        if domain is S.EmptySet:
            raise ValueError("Minimum value not defined for empty domain.")

        return function_range(f, symbol, domain).inf
    else:
        raise ValueError("%s is not a valid symbol." % symbol)
<|endoftext|>"""Calculus-related methods."""

from .euler import euler_equations
from .singularities import (singularities, is_increasing,
                            is_strictly_increasing, is_decreasing,
                            is_strictly_decreasing, is_monotonic)
from .finite_diff import finite_diff_weights, apply_finite_diff, differentiate_finite
from .util import (periodicity, not_empty_in, is_convex,
                   stationary_points, minimum, maximum)
from .accumulationbounds import AccumBounds

__all__ = [
'euler_equations',

'singularities', 'is_increasing',
'is_strictly_increasing', 'is_decreasing',
'is_strictly_decreasing', 'is_monotonic',

'finite_diff_weights', 'apply_finite_diff', 'differentiate_finite',

'periodicity', 'not_empty_in', 'is_convex', 'stationary_points',
'minimum', 'maximum',

'AccumBounds'
]
<|endoftext|>"""
Finite difference weights
=========================

This module implements an algorithm for efficient generation of finite
difference weights for ordinary differentials of functions for
derivatives from 0 (interpolation) up to arbitrary order.

The core algorithm is provided in the finite difference weight generating
function (``finite_diff_weights``), and two convenience functions are provided
for:

- estimating a derivative (or interpolate) directly from a series of points
    is also provided (``apply_finite_diff``).
- differentiating by using finite difference approximations
    (``differentiate_finite``).

"""

from sympy.core.function import Derivative
from sympy.core.singleton import S
from sympy.core.function import Subs
from sympy.core.traversal import preorder_traversal
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable



def finite_diff_weights(order, x_list, x0=S.One):
    """
    Calculates the finite difference weights for an arbitrarily spaced
    one-dimensional grid (``x_list``) for derivatives at ``x0`` of order
    0, 1, ..., up to ``order`` using a recursive formula. Order of accuracy
    is at least ``len(x_list) - order``, if ``x_list`` is defined correctly.

    Parameters
    ==========

    order: int
        Up to what derivative order weights should be calculated.
        0 corresponds to interpolation.
    x_list: sequence
        Sequence of (unique) values for the independent variable.
        It is useful (but not necessary) to order ``x_list`` from
        nearest to furthest from ``x0``; see examples below.
    x0: Number or Symbol
        Root or value of the independent variable for which the finite
        difference weights should be generated. Default is ``S.One``.

    Returns
    =======

    list
        A list of sublists, each corresponding to coefficients for
        increasing derivative order, and each containing lists of
        coefficients for increasing subsets of x_list.

    Examples
    ========

    >>> from sympy import finite_diff_weights, S
    >>> res = finite_diff_weights(1, [-S(1)/2, S(1)/2, S(3)/2, S(5)/2], 0)
    >>> res
    [[[1, 0, 0, 0],
      [1/2, 1/2, 0, 0],
      [3/8, 3/4, -1/8, 0],
      [5/16, 15/16, -5/16, 1/16]],
     [[0, 0, 0, 0],
      [-1, 1, 0, 0],
      [-1, 1, 0, 0],
      [-23/24, 7/8, 1/8, -1/24]]]
    >>> res[0][-1]  # FD weights for 0th derivative, using full x_list
    [5/16, 15/16, -5/16, 1/16]
    >>> res[1][-1]  # FD weights for 1st derivative
    [-23/24, 7/8, 1/8, -1/24]
    >>> res[1][-2]  # FD weights for 1st derivative, using x_list[:-1]
    [-1, 1, 0, 0]
    >>> res[1][-1][0]  # FD weight for 1st deriv. for x_list[0]
    -23/24
    >>> res[1][-1][1]  # FD weight for 1st deriv. for x_list[1], etc.
    7/8

    Each sublist contains the most accurate formula at the end.
    Note, that in the above example ``res[1][1]`` is the same as ``res[1][2]``.
    Since res[1][2] has an order of accuracy of
    ``len(x_list[:3]) - order = 3 - 1 = 2``, the same is true for ``res[1][1]``!

    >>> res = finite_diff_weights(1, [S(0), S(1), -S(1), S(2), -S(2)], 0)[1]
    >>> res
    [[0, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [0, 1/2, -1/2, 0, 0],
     [-1/2, 1, -1/3, -1/6, 0],
     [0, 2/3, -2/3, -1/12, 1/12]]
    >>> res[0]  # no approximation possible, using x_list[0] only
    [0, 0, 0, 0, 0]
    >>> res[1]  # classic forward step approximation
    [-1, 1, 0, 0, 0]
    >>> res[2]  # classic centered approximation
    [0, 1/2, -1/2, 0, 0]
    >>> res[3:]  # higher order approximations
    [[-1/2, 1, -1/3, -1/6, 0], [0, 2/3, -2/3, -1/12, 1/12]]

    Let us compare this to a differently defined ``x_list``. Pay attention to
    ``foo[i][k]`` corresponding to the gridpoint defined by ``x_list[k]``.

    >>> foo = finite_diff_weights(1, [-S(2), -S(1), S(0), S(1), S(2)], 0)[1]
    >>> foo
    [[0, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [1/2, -2, 3/2, 0, 0],
     [1/6, -1, 1/2, 1/3, 0],
     [1/12, -2/3, 0, 2/3, -1/12]]
    >>> foo[1]  # not the same and of lower accuracy as res[1]!
    [-1, 1, 0, 0, 0]
    >>> foo[2]  # classic double backward step approximation
    [1/2, -2, 3/2, 0, 0]
    >>> foo[4]  # the same as res[4]
    [1/12, -2/3, 0, 2/3, -1/12]

    Note that, unless you plan on using approximations based on subsets of
    ``x_list``, the order of gridpoints does not matter.

    The capability to generate weights at arbitrary points can be
    used e.g. to minimize Runge's phenomenon by using Chebyshev nodes:

    >>> from sympy import cos, symbols, pi, simplify
    >>> N, (h, x) = 4, symbols('h x')
    >>> x_list = [x+h*cos(i*pi/(N)) for i in range(N,-1,-1)] # chebyshev nodes
    >>> print(x_list)
    [-h + x, -sqrt(2)*h/2 + x, x, sqrt(2)*h/2 + x, h + x]
    >>> mycoeffs = finite_diff_weights(1, x_list, 0)[1][4]
    >>> [simplify(c) for c in  mycoeffs] #doctest: +NORMALIZE_WHITESPACE
    [(h**3/2 + h**2*x - 3*h*x**2 - 4*x**3)/h**4,
    (-sqrt(2)*h**3 - 4*h**2*x + 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,
    (6*h**2*x - 8*x**3)/h**4,
    (sqrt(2)*h**3 - 4*h**2*x - 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,
    (-h**3/2 + h**2*x + 3*h*x**2 - 4*x**3)/h**4]

    Notes
    =====

    If weights for a finite difference approximation of 3rd order
    derivative is wanted, weights for 0th, 1st and 2nd order are
    calculated "for free", so are formulae using subsets of ``x_list``.
    This is something one can take advantage of to save computational cost.
    Be aware that one should define ``x_list`` from nearest to furthest from
    ``x0``. If not, subsets of ``x_list`` will yield poorer approximations,
    which might not grand an order of accuracy of ``len(x_list) - order``.

    See also
    ========

    sympy.calculus.finite_diff.apply_finite_diff

    References
    ==========

    .. [1] Generation of Finite Difference Formulas on Arbitrarily Spaced
            Grids, Bengt Fornberg; Mathematics of computation; 51; 184;
            (1988); 699-706; doi:10.1090/S0025-5718-1988-0935077-0

    """
    # The notation below closely corresponds to the one used in the paper.
    order = S(order)
    if not order.is_number:
        raise ValueError("Cannot handle symbolic order.")
    if order < 0:
        raise ValueError("Negative derivative order illegal.")
    if int(order) != order:
        raise ValueError("Non-integer order illegal")
    M = order
    N = len(x_list) - 1
    delta = [[[0 for nu in range(N+1)] for n in range(N+1)] for
             m in range(M+1)]
    delta[0][0][0] = S.One
    c1 = S.One
    for n in range(1, N+1):
        c2 = S.One
        for nu in range(n):
            c3 = x_list[n] - x_list[nu]
            c2 = c2 * c3
            if n <= M:
                delta[n][n-1][nu] = 0
            for m in range(min(n, M)+1):
                delta[m][n][nu] = (x_list[n]-x0)*delta[m][n-1][nu] -\
                    m*delta[m-1][n-1][nu]
                delta[m][n][nu] /= c3
        for m in range(min(n, M)+1):
            delta[m][n][n] = c1/c2*(m*delta[m-1][n-1][n-1] -
                                    (x_list[n-1]-x0)*delta[m][n-1][n-1])
        c1 = c2
    return delta


def apply_finite_diff(order, x_list, y_list, x0=S.Zero):
    """
    Calculates the finite difference approximation of
    the derivative of requested order at ``x0`` from points
    provided in ``x_list`` and ``y_list``.

    Parameters
    ==========

    order: int
        order of derivative to approximate. 0 corresponds to interpolation.
    x_list: sequence
        Sequence of (unique) values for the independent variable.
    y_list: sequence
        The function value at corresponding values for the independent
        variable in x_list.
    x0: Number or Symbol
        At what value of the independent variable the derivative should be
        evaluated. Defaults to 0.

    Returns
    =======

    sympy.core.add.Add or sympy.core.numbers.Number
        The finite difference expression approximating the requested
        derivative order at ``x0``.

    Examples
    ========

    >>> from sympy import apply_finite_diff
    >>> cube = lambda arg: (1.0*arg)**3
    >>> xlist = range(-3,3+1)
    >>> apply_finite_diff(2, xlist, map(cube, xlist), 2) - 12 # doctest: +SKIP
    -3.55271367880050e-15

    we see that the example above only contain rounding errors.
    apply_finite_diff can also be used on more abstract objects:

    >>> from sympy import IndexedBase, Idx
    >>> x, y = map(IndexedBase, 'xy')
    >>> i = Idx('i')
    >>> x_list, y_list = zip(*[(x[i+j], y[i+j]) for j in range(-1,2)])
    >>> apply_finite_diff(1, x_list, y_list, x[i])
    ((x[i + 1] - x[i])/(-x[i - 1] + x[i]) - 1)*y[i]/(x[i + 1] - x[i]) -
    (x[i + 1] - x[i])*y[i - 1]/((x[i + 1] - x[i - 1])*(-x[i - 1] + x[i])) +
    (-x[i - 1] + x[i])*y[i + 1]/((x[i + 1] - x[i - 1])*(x[i + 1] - x[i]))

    Notes
    =====

    Order = 0 corresponds to interpolation.
    Only supply so many points you think makes sense
    to around x0 when extracting the derivative (the function
    need to be well behaved within that region). Also beware
    of Runge's phenomenon.

    See also
    ========

    sympy.calculus.finite_diff.finite_diff_weights

    References
    ==========

    Fortran 90 implementation with Python interface for numerics: finitediff_

    .. _finitediff: https://github.com/bjodah/finitediff

    """

    # In the original paper the following holds for the notation:
    # M = order
    # N = len(x_list) - 1

    N = len(x_list) - 1
    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list not equal in length.")

    delta = finite_diff_weights(order, x_list, x0)

    derivative = 0
    for nu in range(len(x_list)):
        derivative += delta[order][N][nu]*y_list[nu]
    return derivative


def _as_finite_diff(derivative, points=1, x0=None, wrt=None):
    """
    Returns an approximation of a derivative of a function in
    the form of a finite difference formula. The expression is a
    weighted sum of the function at a number of discrete values of
    (one of) the independent variable(s).

    Parameters
    ==========

    derivative: a Derivative instance

    points: sequence or coefficient, optional
        If sequence: discrete values (length >= order+1) of the
        independent variable used for generating the finite
        difference weights.
        If it is a coefficient, it will be used as the step-size
        for generating an equidistant sequence of length order+1
        centered around ``x0``. default: 1 (step-size 1)

    x0: number or Symbol, optional
        the value of the independent variable (``wrt``) at which the
        derivative is to be approximated. Default: same as ``wrt``.

    wrt: Symbol, optional
        "with respect to" the variable for which the (partial)
        derivative is to be approximated for. If not provided it
        is required that the Derivative is ordinary. Default: ``None``.

    Examples
    ========

    >>> from sympy import symbols, Function, exp, sqrt, Symbol
    >>> from sympy.calculus.finite_diff import _as_finite_diff
    >>> x, h = symbols('x h')
    >>> f = Function('f')
    >>> _as_finite_diff(f(x).diff(x))
    -f(x - 1/2) + f(x + 1/2)

    The default step size and number of points are 1 and ``order + 1``
    respectively. We can change the step size by passing a symbol
    as a parameter:

    >>> _as_finite_diff(f(x).diff(x), h)
    -f(-h/2 + x)/h + f(h/2 + x)/h

    We can also specify the discretized values to be used in a sequence:

    >>> _as_finite_diff(f(x).diff(x), [x, x+h, x+2*h])
    -3*f(x)/(2*h) + 2*f(h + x)/h - f(2*h + x)/(2*h)

    The algorithm is not restricted to use equidistant spacing, nor
    do we need to make the approximation around ``x0``, but we can get
    an expression estimating the derivative at an offset:

    >>> e, sq2 = exp(1), sqrt(2)
    >>> xl = [x-h, x+h, x+e*h]
    >>> _as_finite_diff(f(x).diff(x, 1), xl, x+h*sq2)
    2*h*((h + sqrt(2)*h)/(2*h) - (-sqrt(2)*h + h)/(2*h))*f(E*h + x)/((-h + E*h)*(h + E*h)) +
    (-(-sqrt(2)*h + h)/(2*h) - (-sqrt(2)*h + E*h)/(2*h))*f(-h + x)/(h + E*h) +
    (-(h + sqrt(2)*h)/(2*h) + (-sqrt(2)*h + E*h)/(2*h))*f(h + x)/(-h + E*h)

    Partial derivatives are also supported:

    >>> y = Symbol('y')
    >>> d2fdxdy=f(x,y).diff(x,y)
    >>> _as_finite_diff(d2fdxdy, wrt=x)
    -Derivative(f(x - 1/2, y), y) + Derivative(f(x + 1/2, y), y)

    See also
    ========

    sympy.calculus.finite_diff.apply_finite_diff
    sympy.calculus.finite_diff.finite_diff_weights

    """
    if derivative.is_Derivative:
        pass
    elif derivative.is_Atom:
        return derivative
    else:
        return derivative.fromiter(
            [_as_finite_diff(ar, points, x0, wrt) for ar
             in derivative.args], **derivative.assumptions0)

    if wrt is None:
        old = None
        for v in derivative.variables:
            if old is v:
                continue
            derivative = _as_finite_diff(derivative, points, x0, v)
            old = v
        return derivative

    order = derivative.variables.count(wrt)

    if x0 is None:
        x0 = wrt

    if not iterable(points):
        if getattr(points, 'is_Function', False) and wrt in points.args:
            points = points.subs(wrt, x0)
        # points is simply the step-size, let's make it a
        # equidistant sequence centered around x0
        if order % 2 == 0:
            # even order => odd number of points, grid point included
            points = [x0 + points*i for i
                      in range(-order//2, order//2 + 1)]
        else:
            # odd order => even number of points, half-way wrt grid point
            points = [x0 + points*S(i)/2 for i
                      in range(-order, order + 1, 2)]
    others = [wrt, 0]
    for v in set(derivative.variables):
        if v == wrt:
            continue
        others += [v, derivative.variables.count(v)]
    if len(points) < order+1:
        raise ValueError("Too few points for order %d" % order)
    return apply_finite_diff(order, points, [
        Derivative(derivative.expr.subs({wrt: x}), *others) for
        x in points], x0)


def differentiate_finite(expr, *symbols,
                         points=1, x0=None, wrt=None, evaluate=False):
    r""" Differentiate expr and replace Derivatives with finite differences.

    Parameters
    ==========

    expr : expression
    \*symbols : differentiate with respect to symbols
    points: sequence, coefficient or undefined function, optional
        see ``Derivative.as_finite_difference``
    x0: number or Symbol, optional
        see ``Derivative.as_finite_difference``
    wrt: Symbol, optional
        see ``Derivative.as_finite_difference``

    Examples
    ========

    >>> from sympy import sin, Function, differentiate_finite
    >>> from sympy.abc import x, y, h
    >>> f, g = Function('f'), Function('g')
    >>> differentiate_finite(f(x)*g(x), x, points=[x-h, x+h])
    -f(-h + x)*g(-h + x)/(2*h) + f(h + x)*g(h + x)/(2*h)

    ``differentiate_finite`` works on any expression, including the expressions
    with embedded derivatives:

    >>> differentiate_finite(f(x) + sin(x), x, 2)
    -2*f(x) + f(x - 1) + f(x + 1) - 2*sin(x) + sin(x - 1) + sin(x + 1)
    >>> differentiate_finite(f(x, y), x, y)
    f(x - 1/2, y - 1/2) - f(x - 1/2, y + 1/2) - f(x + 1/2, y - 1/2) + f(x + 1/2, y + 1/2)
    >>> differentiate_finite(f(x)*g(x).diff(x), x)
    (-g(x) + g(x + 1))*f(x + 1/2) - (g(x) - g(x - 1))*f(x - 1/2)

    To make finite difference with non-constant discretization step use
    undefined functions:

    >>> dx = Function('dx')
    >>> differentiate_finite(f(x)*g(x).diff(x), points=dx(x))
    -(-g(x - dx(x)/2 - dx(x - dx(x)/2)/2)/dx(x - dx(x)/2) +
    g(x - dx(x)/2 + dx(x - dx(x)/2)/2)/dx(x - dx(x)/2))*f(x - dx(x)/2)/dx(x) +
    (-g(x + dx(x)/2 - dx(x + dx(x)/2)/2)/dx(x + dx(x)/2) +
    g(x + dx(x)/2 + dx(x + dx(x)/2)/2)/dx(x + dx(x)/2))*f(x + dx(x)/2)/dx(x)

    """
    if any(term.is_Derivative for term in list(preorder_traversal(expr))):
        evaluate = False

    Dexpr = expr.diff(*symbols, evaluate=evaluate)
    if evaluate:
        sympy_deprecation_warning("""
        The evaluate flag to differentiate_finite() is deprecated.

        evaluate=True expands the intermediate derivatives before computing
        differences, but this usually not what you want, as it does not
        satisfy the product rule.
        """,
            deprecated_since_version="1.5",
             active_deprecations_target="deprecated-differentiate_finite-evaluate",
        )
        return Dexpr.replace(
            lambda arg: arg.is_Derivative,
            lambda arg: arg.as_finite_difference(points=points, x0=x0, wrt=wrt))
    else:
        DFexpr = Dexpr.as_finite_difference(points=points, x0=x0, wrt=wrt)
        return DFexpr.replace(
            lambda arg: isinstance(arg, Subs),
            lambda arg: arg.expr.as_finite_difference(
                    points=points, x0=arg.point[0], wrt=arg.variables[0]))
<|endoftext|>"""
Singularities
=============

This module implements algorithms for finding singularities for a function
and identifying types of functions.

The differential calculus methods in this module include methods to identify
the following function types in the given ``Interval``:
- Increasing
- Strictly Increasing
- Decreasing
- Strictly Decreasing
- Monotonic

"""

from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import sec, csc, cot, tan, cos
from sympy.functions.elementary.hyperbolic import (
    sech, csch, coth, tanh, cosh, asech, acsch, atanh, acoth)
from sympy.utilities.misc import filldedent


def singularities(expression, symbol, domain=None):
    """
    Find singularities of a given function.

    Parameters
    ==========

    expression : Expr
        The target function in which singularities need to be found.
    symbol : Symbol
        The symbol over the values of which the singularity in
        expression in being searched for.

    Returns
    =======

    Set
        A set of values for ``symbol`` for which ``expression`` has a
        singularity. An ``EmptySet`` is returned if ``expression`` has no
        singularities for any given value of ``Symbol``.

    Raises
    ======

    NotImplementedError
        Methods for determining the singularities of this function have
        not been developed.

    Notes
    =====

    This function does not find non-isolated singularities
    nor does it find branch points of the expression.

    Currently supported functions are:
        - univariate continuous (real or complex) functions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathematical_singularity

    Examples
    ========

    >>> from sympy import singularities, Symbol, log
    >>> x = Symbol('x', real=True)
    >>> y = Symbol('y', real=False)
    >>> singularities(x**2 + x + 1, x)
    EmptySet
    >>> singularities(1/(x + 1), x)
    {-1}
    >>> singularities(1/(y**2 + 1), y)
    {-I, I}
    >>> singularities(1/(y**3 + 1), y)
    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}
    >>> singularities(log(x), x)
    {0}

    """
    from sympy.solvers.solveset import solveset

    if domain is None:
        domain = S.Reals if symbol.is_real else S.Complexes
    try:
        sings = S.EmptySet
        e = expression.rewrite([sec, csc, cot, tan], cos)
        e = e.rewrite([sech, csch, coth, tanh], cosh)
        for i in e.atoms(Pow):
            if i.exp.is_infinite:
                raise NotImplementedError
            if i.exp.is_negative:
                # XXX: exponent of varying sign not handled
                sings += solveset(i.base, symbol, domain)
        for i in expression.atoms(log, asech, acsch):
            sings += solveset(i.args[0], symbol, domain)
        for i in expression.atoms(atanh, acoth):
            sings += solveset(i.args[0] - 1, symbol, domain)
            sings += solveset(i.args[0] + 1, symbol, domain)
        return sings
    except NotImplementedError:
        raise NotImplementedError(filldedent('''
            Methods for determining the singularities
            of this function have not been developed.'''))


###########################################################################
#                      DIFFERENTIAL CALCULUS METHODS                      #
###########################################################################


def monotonicity_helper(expression, predicate, interval=S.Reals, symbol=None):
    """
    Helper function for functions checking function monotonicity.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked
    predicate : function
        The property being tested for. The function takes in an integer
        and returns a boolean. The integer input is the derivative and
        the boolean result should be true if the property is being held,
        and false otherwise.
    interval : Set, optional
        The range of values in which we are testing, defaults to all reals.
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    It returns a boolean indicating whether the interval in which
    the function's derivative satisfies given predicate is a superset
    of the given interval.

    Returns
    =======

    Boolean
        True if ``predicate`` is true for all the derivatives when ``symbol``
        is varied in ``range``, False otherwise.

    """
    from sympy.solvers.solveset import solveset

    expression = sympify(expression)
    free = expression.free_symbols

    if symbol is None:
        if len(free) > 1:
            raise NotImplementedError(
                'The function has not yet been implemented'
                ' for all multivariate expressions.'
            )

    variable = symbol or (free.pop() if free else Symbol('x'))
    derivative = expression.diff(variable)
    predicate_interval = solveset(predicate(derivative), variable, S.Reals)
    return interval.is_subset(predicate_interval)


def is_increasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is increasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is increasing (either strictly increasing or
        constant) in the given ``interval``, False otherwise.

    Examples
    ========

    >>> from sympy import is_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_increasing(-x**2, Interval(-oo, 0))
    True
    >>> is_increasing(-x**2, Interval(0, oo))
    False
    >>> is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))
    False
    >>> is_increasing(x**2 + y, Interval(1, 2), x)
    True

    """
    return monotonicity_helper(expression, lambda x: x >= 0, interval, symbol)


def is_strictly_increasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is strictly increasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is strictly increasing in the given ``interval``,
        False otherwise.

    Examples
    ========

    >>> from sympy import is_strictly_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import Interval, oo
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3))
    False
    >>> is_strictly_increasing(-x**2, Interval(0, oo))
    False
    >>> is_strictly_increasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x > 0, interval, symbol)


def is_decreasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is decreasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is decreasing (either strictly decreasing or
        constant) in the given ``interval``, False otherwise.

    Examples
    ========

    >>> from sympy import is_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))
    False
    >>> is_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x <= 0, interval, symbol)


def is_strictly_decreasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is strictly decreasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is strictly decreasing in the given ``interval``,
        False otherwise.

    Examples
    ========

    >>> from sympy import is_strictly_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))
    False
    >>> is_strictly_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_strictly_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x < 0, interval, symbol)


def is_monotonic(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is monotonic in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is monotonic in the given ``interval``,
        False otherwise.

    Raises
    ======

    NotImplementedError
        Monotonicity check has not been implemented for the queried function.

    Examples
    ========

    >>> from sympy import is_monotonic
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_monotonic(-x**2, S.Reals)
    False
    >>> is_monotonic(x**2 + y + 1, Interval(1, 2), x)
    True

    """
    from sympy.solvers.solveset import solveset

    expression = sympify(expression)

    free = expression.free_symbols
    if symbol is None and len(free) > 1:
        raise NotImplementedError(
            'is_monotonic has not yet been implemented'
            ' for all multivariate expressions.'
        )

    variable = symbol or (free.pop() if free else Symbol('x'))
    turning_points = solveset(expression.diff(variable), variable, interval)
    return interval.intersection(turning_points) is S.EmptySet
<|endoftext|>from sympy.core import Add, Mul, Pow, S
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import _sympifyit, oo, zoo
from sympy.core.relational import is_le, is_lt, is_ge, is_gt
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And
from sympy.multipledispatch import dispatch
from sympy.series.order import Order
from sympy.sets.sets import FiniteSet


class AccumulationBounds(Expr):
    r"""An accumulation bounds.

    # Note AccumulationBounds has an alias: AccumBounds

    AccumulationBounds represent an interval `[a, b]`, which is always closed
    at the ends. Here `a` and `b` can be any value from extended real numbers.

    The intended meaning of AccummulationBounds is to give an approximate
    location of the accumulation points of a real function at a limit point.

    Let `a` and `b` be reals such that `a \le b`.

    `\left\langle a, b\right\rangle = \{x \in \mathbb{R} \mid a \le x \le b\}`

    `\left\langle -\infty, b\right\rangle = \{x \in \mathbb{R} \mid x \le b\} \cup \{-\infty, \infty\}`

    `\left\langle a, \infty \right\rangle = \{x \in \mathbb{R} \mid a \le x\} \cup \{-\infty, \infty\}`

    `\left\langle -\infty, \infty \right\rangle = \mathbb{R} \cup \{-\infty, \infty\}`

    ``oo`` and ``-oo`` are added to the second and third definition respectively,
    since if either ``-oo`` or ``oo`` is an argument, then the other one should
    be included (though not as an end point). This is forced, since we have,
    for example, ``1/AccumBounds(0, 1) = AccumBounds(1, oo)``, and the limit at
    `0` is not one-sided. As `x` tends to `0-`, then `1/x \rightarrow -\infty`, so `-\infty`
    should be interpreted as belonging to ``AccumBounds(1, oo)`` though it need
    not appear explicitly.

    In many cases it suffices to know that the limit set is bounded.
    However, in some other cases more exact information could be useful.
    For example, all accumulation values of `\cos(x) + 1` are non-negative.
    (``AccumBounds(-1, 1) + 1 = AccumBounds(0, 2)``)

    A AccumulationBounds object is defined to be real AccumulationBounds,
    if its end points are finite reals.

    Let `X`, `Y` be real AccumulationBounds, then their sum, difference,
    product are defined to be the following sets:

    `X + Y = \{ x+y \mid x \in X \cap y \in Y\}`

    `X - Y = \{ x-y \mid x \in X \cap y \in Y\}`

    `X \times Y = \{ x \times y \mid x \in X \cap y \in Y\}`

    When an AccumBounds is raised to a negative power, if 0 is contained
    between the bounds then an infinite range is returned, otherwise if an
    endpoint is 0 then a semi-infinite range with consistent sign will be returned.

    AccumBounds in expressions behave a lot like Intervals but the
    semantics are not necessarily the same. Division (or exponentiation
    to a negative integer power) could be handled with *intervals* by
    returning a union of the results obtained after splitting the
    bounds between negatives and positives, but that is not done with
    AccumBounds. In addition, bounds are assumed to be independent of
    each other; if the same bound is used in more than one place in an
    expression, the result may not be the supremum or infimum of the
    expression (see below). Finally, when a boundary is ``1``,
    exponentiation to the power of ``oo`` yields ``oo``, neither
    ``1`` nor ``nan``.

    Examples
    ========

    >>> from sympy import AccumBounds, sin, exp, log, pi, E, S, oo
    >>> from sympy.abc import x

    >>> AccumBounds(0, 1) + AccumBounds(1, 2)
    AccumBounds(1, 3)

    >>> AccumBounds(0, 1) - AccumBounds(0, 2)
    AccumBounds(-2, 1)

    >>> AccumBounds(-2, 3)*AccumBounds(-1, 1)
    AccumBounds(-3, 3)

    >>> AccumBounds(1, 2)*AccumBounds(3, 5)
    AccumBounds(3, 10)

    The exponentiation of AccumulationBounds is defined
    as follows:

    If 0 does not belong to `X` or `n > 0` then

    `X^n = \{ x^n \mid x \in X\}`

    >>> AccumBounds(1, 4)**(S(1)/2)
    AccumBounds(1, 2)

    otherwise, an infinite or semi-infinite result is obtained:

    >>> 1/AccumBounds(-1, 1)
    AccumBounds(-oo, oo)
    >>> 1/AccumBounds(0, 2)
    AccumBounds(1/2, oo)
    >>> 1/AccumBounds(-oo, 0)
    AccumBounds(-oo, 0)

    A boundary of 1 will always generate all nonnegatives:

    >>> AccumBounds(1, 2)**oo
    AccumBounds(0, oo)
    >>> AccumBounds(0, 1)**oo
    AccumBounds(0, oo)

    If the exponent is itself an AccumulationBounds or is not an
    integer then unevaluated results will be returned unless the base
    values are positive:

    >>> AccumBounds(2, 3)**AccumBounds(-1, 2)
    AccumBounds(1/3, 9)
    >>> AccumBounds(-2, 3)**AccumBounds(-1, 2)
    AccumBounds(-2, 3)**AccumBounds(-1, 2)

    >>> AccumBounds(-2, -1)**(S(1)/2)
    sqrt(AccumBounds(-2, -1))

    Note: `\left\langle a, b\right\rangle^2` is not same as `\left\langle a, b\right\rangle \times \left\langle a, b\right\rangle`

    >>> AccumBounds(-1, 1)**2
    AccumBounds(0, 1)

    >>> AccumBounds(1, 3) < 4
    True

    >>> AccumBounds(1, 3) < -1
    False

    Some elementary functions can also take AccumulationBounds as input.
    A function `f` evaluated for some real AccumulationBounds `\left\langle a, b \right\rangle`
    is defined as `f(\left\langle a, b\right\rangle) = \{ f(x) \mid a \le x \le b \}`

    >>> sin(AccumBounds(pi/6, pi/3))
    AccumBounds(1/2, sqrt(3)/2)

    >>> exp(AccumBounds(0, 1))
    AccumBounds(1, E)

    >>> log(AccumBounds(1, E))
    AccumBounds(0, 1)

    Some symbol in an expression can be substituted for a AccumulationBounds
    object. But it does not necessarily evaluate the AccumulationBounds for
    that expression.

    The same expression can be evaluated to different values depending upon
    the form it is used for substitution since each instance of an
    AccumulationBounds is considered independent. For example:

    >>> (x**2 + 2*x + 1).subs(x, AccumBounds(-1, 1))
    AccumBounds(-1, 4)

    >>> ((x + 1)**2).subs(x, AccumBounds(-1, 1))
    AccumBounds(0, 4)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_arithmetic

    .. [2] https://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf

    Notes
    =====

    Do not use ``AccumulationBounds`` for floating point interval arithmetic
    calculations, use ``mpmath.iv`` instead.
    """

    is_extended_real = True
    is_number = False

    def __new__(cls, min, max) -> Expr: # type: ignore

        min = _sympify(min)
        max = _sympify(max)

        # Only allow real intervals (use symbols with 'is_extended_real=True').
        if not min.is_extended_real or not max.is_extended_real:
            raise ValueError("Only real AccumulationBounds are supported")

        if max == min:
            return max

        # Make sure that the created AccumBounds object will be valid.
        if max.is_number and min.is_number:
            bad = max.is_comparable and min.is_comparable and max < min
        else:
            bad = (max - min).is_extended_negative
        if bad:
            raise ValueError(
                "Lower limit should be smaller than upper limit")

        return Basic.__new__(cls, min, max)

    # setting the operation priority
    _op_priority = 11.0

    def _eval_is_real(self):
        if self.min.is_real and self.max.is_real:
            return True

    @property
    def min(self):
        """
        Returns the minimum possible value attained by AccumulationBounds
        object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).min
        1

        """
        return self.args[0]

    @property
    def max(self):
        """
        Returns the maximum possible value attained by AccumulationBounds
        object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).max
        3

        """
        return self.args[1]

    @property
    def delta(self):
        """
        Returns the difference of maximum possible value attained by
        AccumulationBounds object and minimum possible value attained
        by AccumulationBounds object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).delta
        2

        """
        return self.max - self.min

    @property
    def mid(self):
        """
        Returns the mean of maximum possible value attained by
        AccumulationBounds object and minimum possible value
        attained by AccumulationBounds object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).mid
        2

        """
        return (self.min + self.max) / 2

    @_sympifyit('other', NotImplemented)
    def _eval_power(self, other):
        return self.__pow__(other)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                return AccumBounds(
                    Add(self.min, other.min),
                    Add(self.max, other.max))
            if other is S.Infinity and self.min is S.NegativeInfinity or \
                    other is S.NegativeInfinity and self.max is S.Infinity:
                return AccumBounds(-oo, oo)
            elif other.is_extended_real:
                if self.min is S.NegativeInfinity and self.max is S.Infinity:
                    return AccumBounds(-oo, oo)
                elif self.min is S.NegativeInfinity:
                    return AccumBounds(-oo, self.max + other)
                elif self.max is S.Infinity:
                    return AccumBounds(self.min + other, oo)
                else:
                    return AccumBounds(Add(self.min, other), Add(self.max, other))
            return Add(self, other, evaluate=False)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return AccumBounds(-self.max, -self.min)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                return AccumBounds(
                    Add(self.min, -other.max),
                    Add(self.max, -other.min))
            if other is S.NegativeInfinity and self.min is S.NegativeInfinity or \
                    other is S.Infinity and self.max is S.Infinity:
                return AccumBounds(-oo, oo)
            elif other.is_extended_real:
                if self.min is S.NegativeInfinity and self.max is S.Infinity:
                    return AccumBounds(-oo, oo)
                elif self.min is S.NegativeInfinity:
                    return AccumBounds(-oo, self.max - other)
                elif self.max is S.Infinity:
                    return AccumBounds(self.min - other, oo)
                else:
                    return AccumBounds(
                        Add(self.min, -other),
                        Add(self.max, -other))
            return Add(self, -other, evaluate=False)
        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return self.__neg__() + other

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if self.args == (-oo, oo):
            return self
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                if other.args == (-oo, oo):
                    return other
                v = set()
                for a in self.args:
                    vi = other*a
                    v.update(vi.args or (vi,))
                return AccumBounds(Min(*v), Max(*v))
            if other is S.Infinity:
                if self.min.is_zero:
                    return AccumBounds(0, oo)
                if self.max.is_zero:
                    return AccumBounds(-oo, 0)
            if other is S.NegativeInfinity:
                if self.min.is_zero:
                    return AccumBounds(-oo, 0)
                if self.max.is_zero:
                    return AccumBounds(0, oo)
            if other.is_extended_real:
                if other.is_zero:
                    if self.max is S.Infinity:
                        return AccumBounds(0, oo)
                    if self.min is S.NegativeInfinity:
                        return AccumBounds(-oo, 0)
                    return S.Zero
                if other.is_extended_positive:
                    return AccumBounds(
                        Mul(self.min, other),
                        Mul(self.max, other))
                elif other.is_extended_negative:
                    return AccumBounds(
                        Mul(self.max, other),
                        Mul(self.min, other))
            if isinstance(other, Order):
                return other
            return Mul(self, other, evaluate=False)
        return NotImplemented

    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                if other.min.is_positive or other.max.is_negative:
                    return self * AccumBounds(1/other.max, 1/other.min)

                if (self.min.is_extended_nonpositive and self.max.is_extended_nonnegative and
                    other.min.is_extended_nonpositive and other.max.is_extended_nonnegative):
                    if self.min.is_zero and other.min.is_zero:
                        return AccumBounds(0, oo)
                    if self.max.is_zero and other.min.is_zero:
                        return AccumBounds(-oo, 0)
                    return AccumBounds(-oo, oo)

                if self.max.is_extended_negative:
                    if other.min.is_extended_negative:
                        if other.max.is_zero:
                            return AccumBounds(self.max / other.min, oo)
                        if other.max.is_extended_positive:
                            # if we were dealing with intervals we would return
                            # Union(Interval(-oo, self.max/other.max),
                            #       Interval(self.max/other.min, oo))
                            return AccumBounds(-oo, oo)

                    if other.min.is_zero and other.max.is_extended_positive:
                        return AccumBounds(-oo, self.max / other.max)

                if self.min.is_extended_positive:
                    if other.min.is_extended_negative:
                        if other.max.is_zero:
                            return AccumBounds(-oo, self.min / other.min)
                        if other.max.is_extended_positive:
                            # if we were dealing with intervals we would return
                            # Union(Interval(-oo, self.min/other.min),
                            #       Interval(self.min/other.max, oo))
                            return AccumBounds(-oo, oo)

                    if other.min.is_zero and other.max.is_extended_positive:
                        return AccumBounds(self.min / other.max, oo)

            elif other.is_extended_real:
                if other in (S.Infinity, S.NegativeInfinity):
                    if self == AccumBounds(-oo, oo):
                        return AccumBounds(-oo, oo)
                    if self.max is S.Infinity:
                        return AccumBounds(Min(0, other), Max(0, other))
                    if self.min is S.NegativeInfinity:
                        return AccumBounds(Min(0, -other), Max(0, -other))
                if other.is_extended_positive:
                    return AccumBounds(self.min / other, self.max / other)
                elif other.is_extended_negative:
                    return AccumBounds(self.max / other, self.min / other)
            if (1 / other) is S.ComplexInfinity:
                return Mul(self, 1 / other, evaluate=False)
            else:
                return Mul(self, 1 / other)

        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rtruediv__(self, other):
        if isinstance(other, Expr):
            if other.is_extended_real:
                if other.is_zero:
                    return S.Zero
                if (self.min.is_extended_nonpositive and self.max.is_extended_nonnegative):
                    if self.min.is_zero:
                        if other.is_extended_positive:
                            return AccumBounds(Mul(other, 1 / self.max), oo)
                        if other.is_extended_negative:
                            return AccumBounds(-oo, Mul(other, 1 / self.max))
                    if self.max.is_zero:
                        if other.is_extended_positive:
                            return AccumBounds(-oo, Mul(other, 1 / self.min))
                        if other.is_extended_negative:
                            return AccumBounds(Mul(other, 1 / self.min), oo)
                    return AccumBounds(-oo, oo)
                else:
                    return AccumBounds(Min(other / self.min, other / self.max),
                                       Max(other / self.min, other / self.max))
            return Mul(other, 1 / self, evaluate=False)
        else:
            return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __pow__(self, other):
        if isinstance(other, Expr):
            if other is S.Infinity:
                if self.min.is_extended_nonnegative:
                    if self.max < 1:
                        return S.Zero
                    if self.min > 1:
                        return S.Infinity
                    return AccumBounds(0, oo)
                elif self.max.is_extended_negative:
                    if self.min > -1:
                        return S.Zero
                    if self.max < -1:
                        return zoo
                    return S.NaN
                else:
                    if self.min > -1:
                        if self.max < 1:
                            return S.Zero
                        return AccumBounds(0, oo)
                    return AccumBounds(-oo, oo)

            if other is S.NegativeInfinity:
                return (1/self)**oo

            # generically true
            if (self.max - self.min).is_nonnegative:
                # well defined
                if self.min.is_nonnegative:
                    # no 0 to worry about
                    if other.is_nonnegative:
                        # no infinity to worry about
                        return self.func(self.min**other, self.max**other)

            if other.is_zero:
                return S.One  # x**0 = 1

            if other.is_Integer or other.is_integer:
                if self.min.is_extended_positive:
                    return AccumBounds(
                        Min(self.min**other, self.max**other),
                        Max(self.min**other, self.max**other))
                elif self.max.is_extended_negative:
                    return AccumBounds(
                        Min(self.max**other, self.min**other),
                        Max(self.max**other, self.min**other))

                if other % 2 == 0:
                    if other.is_extended_negative:
                        if self.min.is_zero:
                            return AccumBounds(self.max**other, oo)
                        if self.max.is_zero:
                            return AccumBounds(self.min**other, oo)
                        return (1/self)**(-other)
                    return AccumBounds(
                        S.Zero, Max(self.min**other, self.max**other))
                elif other % 2 == 1:
                    if other.is_extended_negative:
                        if self.min.is_zero:
                            return AccumBounds(self.max**other, oo)
                        if self.max.is_zero:
                            return AccumBounds(-oo, self.min**other)
                        return (1/self)**(-other)
                    return AccumBounds(self.min**other, self.max**other)

            # non-integer exponent
            # 0**neg or neg**frac yields complex
            if (other.is_number or other.is_rational) and (
                    self.min.is_extended_nonnegative or (
                    other.is_extended_nonnegative and
                    self.min.is_extended_nonnegative)):
                num, den = other.as_numer_denom()
                if num is S.One:
                    return AccumBounds(*[i**(1/den) for i in self.args])

                elif den is not S.One:  # e.g. if other is not Float
                    return (self**num)**(1/den)  # ok for non-negative base

            if isinstance(other, AccumBounds):
                if (self.min.is_extended_positive or
                        self.min.is_extended_nonnegative and
                        other.min.is_extended_nonnegative):
                    p = [self**i for i in other.args]
                    if not any(i.is_Pow for i in p):
                        a = [j for i in p for j in i.args or (i,)]
                        try:
                            return self.func(min(a), max(a))
                        except TypeError:  # can't sort
                            pass

            return Pow(self, other, evaluate=False)

        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rpow__(self, other):
        if other.is_real and other.is_extended_nonnegative and (
                self.max - self.min).is_extended_positive:
            if other is S.One:
                return S.One
            if other.is_extended_positive:
                a, b = [other**i for i in self.args]
                if min(a, b) != a:
                    a, b = b, a
                return self.func(a, b)
            if other.is_zero:
                if self.min.is_zero:
                    return self.func(0, 1)
                if self.min.is_extended_positive:
                    return S.Zero

        return Pow(other, self, evaluate=False)

    def __abs__(self):
        if self.max.is_extended_negative:
            return self.__neg__()
        elif self.min.is_extended_negative:
            return AccumBounds(S.Zero, Max(abs(self.min), self.max))
        else:
            return self


    def __contains__(self, other):
        """
        Returns ``True`` if other is contained in self, where other
        belongs to extended real numbers, ``False`` if not contained,
        otherwise TypeError is raised.

        Examples
        ========

        >>> from sympy import AccumBounds, oo
        >>> 1 in AccumBounds(-1, 3)
        True

        -oo and oo go together as limits (in AccumulationBounds).

        >>> -oo in AccumBounds(1, oo)
        True

        >>> oo in AccumBounds(-oo, 0)
        True

        """
        other = _sympify(other)

        if other in (S.Infinity, S.NegativeInfinity):
            if self.min is S.NegativeInfinity or self.max is S.Infinity:
                return True
            return False

        rv = And(self.min <= other, self.max >= other)
        if rv not in (True, False):
            raise TypeError("input failed to evaluate")
        return rv

    def intersection(self, other):
        """
        Returns the intersection of 'self' and 'other'.
        Here other can be an instance of :py:class:`~.FiniteSet` or AccumulationBounds.

        Parameters
        ==========

        other : AccumulationBounds
            Another AccumulationBounds object with which the intersection
            has to be computed.

        Returns
        =======

        AccumulationBounds
            Intersection of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy import AccumBounds, FiniteSet
        >>> AccumBounds(1, 3).intersection(AccumBounds(2, 4))
        AccumBounds(2, 3)

        >>> AccumBounds(1, 3).intersection(AccumBounds(4, 6))
        EmptySet

        >>> AccumBounds(1, 4).intersection(FiniteSet(1, 2, 5))
        {1, 2}

        """
        if not isinstance(other, (AccumBounds, FiniteSet)):
            raise TypeError(
                "Input must be AccumulationBounds or FiniteSet object")

        if isinstance(other, FiniteSet):
            fin_set = S.EmptySet
            for i in other:
                if i in self:
                    fin_set = fin_set + FiniteSet(i)
            return fin_set

        if self.max < other.min or self.min > other.max:
            return S.EmptySet

        if self.min <= other.min:
            if self.max <= other.max:
                return AccumBounds(other.min, self.max)
            if self.max > other.max:
                return other

        if other.min <= self.min:
            if other.max < self.max:
                return AccumBounds(self.min, other.max)
            if other.max > self.max:
                return self

    def union(self, other):
        # TODO : Devise a better method for Union of AccumBounds
        # this method is not actually correct and
        # can be made better
        if not isinstance(other, AccumBounds):
            raise TypeError(
                "Input must be AccumulationBounds or FiniteSet object")

        if self.min <= other.min and self.max >= other.min:
            return AccumBounds(self.min, Max(self.max, other.max))

        if other.min <= self.min and other.max >= self.min:
            return AccumBounds(other.min, Max(self.max, other.max))


@dispatch(AccumulationBounds, AccumulationBounds) # type: ignore # noqa:F811
def _eval_is_le(lhs, rhs): # noqa:F811
    if is_le(lhs.max, rhs.min):
        return True
    if is_gt(lhs.min, rhs.max):
        return False


@dispatch(AccumulationBounds, Basic) # type: ignore # noqa:F811
def _eval_is_le(lhs, rhs): # noqa: F811

    """
    Returns ``True `` if range of values attained by ``lhs`` AccumulationBounds
    object is greater than the range of values attained by ``rhs``,
    where ``rhs`` may be any value of type AccumulationBounds object or
    extended real number value, ``False`` if ``rhs`` satisfies
    the same property, else an unevaluated :py:class:`~.Relational`.

    Examples
    ========

    >>> from sympy import AccumBounds, oo
    >>> AccumBounds(1, 3) > AccumBounds(4, oo)
    False
    >>> AccumBounds(1, 4) > AccumBounds(3, 4)
    AccumBounds(1, 4) > AccumBounds(3, 4)
    >>> AccumBounds(1, oo) > -1
    True

    """
    if not rhs.is_extended_real:
            raise TypeError(
                "Invalid comparison of %s %s" %
                (type(rhs), rhs))
    elif rhs.is_comparable:
        if is_le(lhs.max, rhs):
            return True
        if is_gt(lhs.min, rhs):
            return False


@dispatch(AccumulationBounds, AccumulationBounds)
def _eval_is_ge(lhs, rhs): # noqa:F811
    if is_ge(lhs.min, rhs.max):
        return True
    if is_lt(lhs.max, rhs.min):
        return False


@dispatch(AccumulationBounds, Expr)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa: F811
    """
    Returns ``True`` if range of values attained by ``lhs`` AccumulationBounds
    object is less that the range of values attained by ``rhs``, where
    other may be any value of type AccumulationBounds object or extended
    real number value, ``False`` if ``rhs`` satisfies the same
    property, else an unevaluated :py:class:`~.Relational`.

    Examples
    ========

    >>> from sympy import AccumBounds, oo
    >>> AccumBounds(1, 3) >= AccumBounds(4, oo)
    False
    >>> AccumBounds(1, 4) >= AccumBounds(3, 4)
    AccumBounds(1, 4) >= AccumBounds(3, 4)
    >>> AccumBounds(1, oo) >= 1
    True
    """

    if not rhs.is_extended_real:
        raise TypeError(
            "Invalid comparison of %s %s" %
            (type(rhs), rhs))
    elif rhs.is_comparable:
        if is_ge(lhs.min, rhs):
            return True
        if is_lt(lhs.max, rhs):
            return False


@dispatch(Expr, AccumulationBounds)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811
    if not lhs.is_extended_real:
        raise TypeError(
            "Invalid comparison of %s %s" %
            (type(lhs), lhs))
    elif lhs.is_comparable:
        if is_le(rhs.max, lhs):
            return True
        if is_gt(rhs.min, lhs):
            return False


@dispatch(AccumulationBounds, AccumulationBounds)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811
    if is_ge(lhs.min, rhs.max):
        return True
    if is_lt(lhs.max, rhs.min):
        return False

# setting an alias for AccumulationBounds
AccumBounds = AccumulationBounds
<|endoftext|>from sympy.core import Atom, Basic


class CartanType_generator():
    """
    Constructor for actually creating things
    """
    def __call__(self, *args):
        c = args[0]
        if isinstance(c, list):
            letter, n = c[0], int(c[1])
        elif isinstance(c, str):
            letter, n = c[0], int(c[1:])
        else:
            raise TypeError("Argument must be a string (e.g. 'A3') or a list (e.g. ['A', 3])")

        if n < 0:
            raise ValueError("Lie algebra rank cannot be negative")
        if letter == "A":
            from . import type_a
            return type_a.TypeA(n)
        if letter == "B":
            from . import type_b
            return type_b.TypeB(n)

        if letter == "C":
            from . import type_c
            return type_c.TypeC(n)

        if letter == "D":
            from . import type_d
            return type_d.TypeD(n)

        if letter == "E":
            if n >= 6 and n <= 8:
                from . import type_e
                return type_e.TypeE(n)

        if letter == "F":
            if n == 4:
                from . import type_f
                return type_f.TypeF(n)

        if letter == "G":
            if n == 2:
                from . import type_g
                return type_g.TypeG(n)

CartanType = CartanType_generator()


class Standard_Cartan(Atom):
    """
    Concrete base class for Cartan types such as A4, etc
    """

    def __new__(cls, series, n):
        obj = Basic.__new__(cls)
        obj.n = n
        obj.series = series
        return obj

    def rank(self):
        """
        Returns the rank of the Lie algebra
        """
        return self.n

    def series(self):
        """
        Returns the type of the Lie algebra
        """
        return self.series
<|endoftext|>from .cartan_type import Standard_Cartan
from sympy.core.backend import eye

class TypeC(Standard_Cartan):

    def __new__(cls, n):
        if n < 3:
            raise ValueError("n cannot be less than 3")
        return Standard_Cartan.__new__(cls, "C", n)


    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("C3")
        >>> c.dimension()
        3
        """
        n = self.n
        return n

    def basic_root(self, i, j):
        """Generate roots with 1 in ith position and a -1 in jth position
        """
        n = self.n
        root = [0]*n
        root[i] = 1
        root[j] = -1
        return root

    def simple_root(self, i):
        """The ith simple root for the C series

        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        In C_n, the first n-1 simple roots are the same as
        the roots in A_(n-1) (a 1 in the ith position, a -1
        in the (i+1)th position, and zeroes elsewhere).  The
        nth simple root is the root in which there is a 2 in
        the nth position and zeroes elsewhere.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("C3")
        >>> c.simple_root(2)
        [0, 1, -1]

        """

        n = self.n
        if i < n:
            return self.basic_root(i-1,i)
        else:
            root = [0]*self.n
            root[n-1] = 2
            return root


    def positive_roots(self):
        """Generates all the positive roots of A_n

        This is half of all of the roots of C_n; by multiplying all the
        positive roots by -1 we get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}

        """

        n = self.n
        posroots = {}
        k = 0
        for i in range(0, n-1):
            for j in range(i+1, n):
                k += 1
                posroots[k] = self.basic_root(i, j)
                k += 1
                root = self.basic_root(i, j)
                root[j] = 1
                posroots[k] = root

        for i in range(0, n):
            k += 1
            root = [0]*n
            root[i] = 2
            posroots[k] = root

        return posroots

    def roots(self):
        """
        Returns the total number of roots for C_n"
        """

        n = self.n
        return 2*(n**2)

    def cartan_matrix(self):
        """The Cartan matrix for C_n

        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('C4')
        >>> c.cartan_matrix()
        Matrix([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -2,  2]])

        """

        n = self.n
        m = 2 * eye(n)
        for i in range(1, n - 1):
            m[i, i+1] = -1
            m[i, i-1] = -1
        m[0,1] = -1
        m[n-1, n-2] = -2
        return m


    def basis(self):
        """
        Returns the number of independent generators of C_n
        """

        n = self.n
        return n*(2*n + 1)

    def lie_algebra(self):
        """
        Returns the Lie algebra associated with C_n"
        """

        n = self.n
        return "sp(" + str(2*n) + ")"

    def dynkin_diagram(self):
        n = self.n
        diag = "---".join("0" for i in range(1, n)) + "=<=0\n"
        diag += "   ".join(str(i) for i in range(1, n+1))
        return diag
<|endoftext|>from .cartan_type import CartanType

def CartanMatrix(ct):
    """Access the Cartan matrix of a specific Lie algebra

    Examples
    ========

    >>> from sympy.liealgebras.cartan_matrix import CartanMatrix
    >>> CartanMatrix("A2")
    Matrix([
    [ 2, -1],
    [-1,  2]])

    >>> CartanMatrix(['C', 3])
    Matrix([
    [ 2, -1,  0],
    [-1,  2, -1],
    [ 0, -2,  2]])

    This method works by returning the Cartan matrix
    which corresponds to Cartan type t.
    """

    return CartanType(ct).cartan_matrix()
<|endoftext|>from sympy.liealgebras.cartan_type import Standard_Cartan
from sympy.core.backend import eye


class TypeA(Standard_Cartan):
    """
    This class contains the information about
    the A series of simple Lie algebras.
    ====
    """

    def __new__(cls, n):
        if n < 1:
            raise ValueError("n cannot be less than 1")
        return Standard_Cartan.__new__(cls, "A", n)


    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A4")
        >>> c.dimension()
        5
        """
        return self.n+1


    def basic_root(self, i, j):
        """
        This is a method just to generate roots
        with a 1 iin the ith position and a -1
        in the jth position.

        """

        n = self.n
        root = [0]*(n+1)
        root[i] = 1
        root[j] = -1
        return root

    def simple_root(self, i):
        """
        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        In A_n the ith simple root is the root which has a 1
        in the ith position, a -1 in the (i+1)th position,
        and zeroes elsewhere.

        This method returns the ith simple root for the A series.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A4")
        >>> c.simple_root(1)
        [1, -1, 0, 0, 0]

        """

        return self.basic_root(i-1, i)

    def positive_roots(self):
        """
        This method generates all the positive roots of
        A_n.  This is half of all of the roots of A_n;
        by multiplying all the positive roots by -1 we
        get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
        """

        n = self.n
        posroots = {}
        k = 0
        for i in range(0, n):
            for j in range(i+1, n+1):
                k += 1
                posroots[k] = self.basic_root(i, j)
        return posroots

    def highest_root(self):
        """
        Returns the highest weight root for A_n
        """

        return self.basic_root(0, self.n)

    def roots(self):
        """
        Returns the total number of roots for A_n
        """
        n = self.n
        return n*(n+1)

    def cartan_matrix(self):
        """
        Returns the Cartan matrix for A_n.
        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('A4')
        >>> c.cartan_matrix()
        Matrix([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]])

        """

        n = self.n
        m = 2 * eye(n)
        for i in range(1, n - 1):
            m[i, i+1] = -1
            m[i, i-1] = -1
        m[0,1] = -1
        m[n-1, n-2] = -1
        return m

    def basis(self):
        """
        Returns the number of independent generators of A_n
        """
        n = self.n
        return n**2 - 1

    def lie_algebra(self):
        """
        Returns the Lie algebra associated with A_n
        """
        n = self.n
        return "su(" + str(n + 1) + ")"

    def dynkin_diagram(self):
        n = self.n
        diag = "---".join("0" for i in range(1, n+1)) + "\n"
        diag += "   ".join(str(i) for i in range(1, n+1))
        return diag
<|endoftext|>from .cartan_type import Standard_Cartan
from sympy.core.backend import eye

class TypeD(Standard_Cartan):

    def __new__(cls, n):
        if n < 3:
            raise ValueError("n cannot be less than 3")
        return Standard_Cartan.__new__(cls, "D", n)


    def dimension(self):
        """Dmension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("D4")
        >>> c.dimension()
        4
        """

        return self.n

    def basic_root(self, i, j):
        """
        This is a method just to generate roots
        with a 1 iin the ith position and a -1
        in the jth position.

        """

        n = self.n
        root = [0]*n
        root[i] = 1
        root[j] = -1
        return root

    def simple_root(self, i):
        """
        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        In D_n, the first n-1 simple roots are the same as
        the roots in A_(n-1) (a 1 in the ith position, a -1
        in the (i+1)th position, and zeroes elsewhere).
        The nth simple root is the root in which there 1s in
        the nth and (n-1)th positions, and zeroes elsewhere.

        This method returns the ith simple root for the D series.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("D4")
        >>> c.simple_root(2)
        [0, 1, -1, 0]

        """

        n = self.n
        if i < n:
            return self.basic_root(i-1, i)
        else:
            root = [0]*n
            root[n-2] = 1
            root[n-1] = 1
            return root


    def positive_roots(self):
        """
        This method generates all the positive roots of
        A_n.  This is half of all of the roots of D_n
        by multiplying all the positive roots by -1 we
        get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
        """

        n = self.n
        posroots = {}
        k = 0
        for i in range(0, n-1):
            for j in range(i+1, n):
                k += 1
                posroots[k] = self.basic_root(i, j)
                k += 1
                root = self.basic_root(i, j)
                root[j] = 1
                posroots[k] = root
        return posroots

    def roots(self):
        """
        Returns the total number of roots for D_n"
        """

        n = self.n
        return 2*n*(n-1)

    def cartan_matrix(self):
        """
        Returns the Cartan matrix for D_n.
        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('D4')
        >>> c.cartan_matrix()
            Matrix([
            [ 2, -1,  0,  0],
            [-1,  2, -1, -1],
            [ 0, -1,  2,  0],
            [ 0, -1,  0,  2]])

        """

        n = self.n
        m = 2*eye(n)
        for i in range(1, n - 2):
            m[i,i+1] = -1
            m[i,i-1] = -1
        m[n-2, n-3] = -1
        m[n-3, n-1] = -1
        m[n-1, n-3] = -1
        m[0, 1] = -1
        return m

    def basis(self):
        """
        Returns the number of independent generators of D_n
        """
        n = self.n
        return n*(n-1)/2

    def lie_algebra(self):
        """
        Returns the Lie algebra associated with D_n"
        """

        n = self.n
        return "so(" + str(2*n) + ")"

    def dynkin_diagram(self):
        n = self.n
        diag = " "*4*(n-3) + str(n-1) + "\n"
        diag += " "*4*(n-3) + "0\n"
        diag += " "*4*(n-3) +"|\n"
        diag += " "*4*(n-3) + "|\n"
        diag += "---".join("0" for i in range(1,n)) + "\n"
        diag += "   ".join(str(i) for i in range(1, n-1)) + "   "+str(n)
        return diag
<|endoftext|># -*- coding: utf-8 -*-

from .cartan_type import Standard_Cartan
from sympy.core.backend import Matrix

class TypeG(Standard_Cartan):

    def __new__(cls, n):
        if n != 2:
            raise ValueError("n should be 2")
        return Standard_Cartan.__new__(cls, "G", 2)


    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("G2")
        >>> c.dimension()
        3
        """
        return 3

    def simple_root(self, i):
        """The ith simple root of G_2

        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("G2")
        >>> c.simple_root(1)
        [0, 1, -1]

        """
        if i == 1:
            return [0, 1, -1]
        else:
            return [1, -2, 1]

    def positive_roots(self):
        """Generate all the positive roots of A_n

        This is half of all of the roots of A_n; by multiplying all the
        positive roots by -1 we get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}

        """

        roots = {1: [0, 1, -1], 2: [1, -2, 1], 3: [1, -1, 0], 4: [1, 0, 1],
                5: [1, 1, -2], 6: [2, -1, -1]}
        return roots

    def roots(self):
        """
        Returns the total number of roots of G_2"
        """
        return 12

    def cartan_matrix(self):
        """The Cartan matrix for G_2

        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("G2")
        >>> c.cartan_matrix()
        Matrix([
            [ 2, -1],
            [-3,  2]])

        """

        m = Matrix( 2, 2, [2, -1, -3, 2])
        return m

    def basis(self):
        """
        Returns the number of independent generators of G_2
        """
        return 14

    def dynkin_diagram(self):
        diag = "0≡<≡0\n1   2"
        return diag
<|endoftext|>from .cartan_type import Standard_Cartan
from sympy.core.backend import Matrix, Rational


class TypeF(Standard_Cartan):

    def __new__(cls, n):
        if n != 4:
            raise ValueError("n should be 4")
        return Standard_Cartan.__new__(cls, "F", 4)

    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("F4")
        >>> c.dimension()
        4
        """

        return 4


    def basic_root(self, i, j):
        """Generate roots with 1 in ith position and -1 in jth position

        """

        n = self.n
        root = [0]*n
        root[i] = 1
        root[j] = -1
        return root

    def simple_root(self, i):
        """The ith simple root of F_4

        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("F4")
        >>> c.simple_root(3)
        [0, 0, 0, 1]

        """

        if i < 3:
            return self.basic_root(i-1, i)
        if i == 3:
            root = [0]*4
            root[3] = 1
            return root
        if i == 4:
            root = [Rational(-1, 2)]*4
            return root

    def positive_roots(self):
        """Generate all the positive roots of A_n

        This is half of all of the roots of F_4; by multiplying all the
        positive roots by -1 we get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}

        """

        n = self.n
        posroots = {}
        k = 0
        for i in range(0, n-1):
            for j in range(i+1, n):
                k += 1
                posroots[k] = self.basic_root(i, j)
                k += 1
                root = self.basic_root(i, j)
                root[j] = 1
                posroots[k] = root

        for i in range(0, n):
            k += 1
            root = [0]*n
            root[i] = 1
            posroots[k] = root

        k += 1
        root = [Rational(1, 2)]*n
        posroots[k] = root
        for i in range(1, 4):
            k += 1
            root = [Rational(1, 2)]*n
            root[i] = Rational(-1, 2)
            posroots[k] = root

        posroots[k+1] = [Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2)]
        posroots[k+2] = [Rational(1, 2), Rational(-1, 2), Rational(1, 2), Rational(-1, 2)]
        posroots[k+3] = [Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]
        posroots[k+4] = [Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(-1, 2)]

        return posroots


    def roots(self):
        """
        Returns the total number of roots for F_4
        """
        return 48

    def cartan_matrix(self):
        """The Cartan matrix for F_4

        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('A4')
        >>> c.cartan_matrix()
        Matrix([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]])
        """

        m = Matrix( 4, 4, [2, -1, 0, 0, -1, 2, -2, 0, 0,
            -1, 2, -1, 0, 0, -1, 2])
        return m

    def basis(self):
        """
        Returns the number of independent generators of F_4
        """
        return 52

    def dynkin_diagram(self):
        diag = "0---0=>=0---0\n"
        diag += "   ".join(str(i) for i in range(1, 5))
        return diag
<|endoftext|>from .cartan_type import CartanType


def DynkinDiagram(t):
    """Display the Dynkin diagram of a given Lie algebra

    Works by generating the CartanType for the input, t, and then returning the
    Dynkin diagram method from the individual classes.

    Examples
    ========

    >>> from sympy.liealgebras.dynkin_diagram import DynkinDiagram
    >>> print(DynkinDiagram("A3"))
    0---0---0
    1   2   3

    >>> print(DynkinDiagram("B4"))
    0---0---0=>=0
    1   2   3   4

    """

    return CartanType(t).dynkin_diagram()
<|endoftext|># -*- coding: utf-8 -*-

from .cartan_type import CartanType
from mpmath import fac
from sympy.core.backend import Matrix, eye, Rational, igcd
from sympy.core.basic import Atom

class WeylGroup(Atom):

    """
    For each semisimple Lie group, we have a Weyl group.  It is a subgroup of
    the isometry group of the root system.  Specifically, it's the subgroup
    that is generated by reflections through the hyperplanes orthogonal to
    the roots.  Therefore, Weyl groups are reflection groups, and so a Weyl
    group is a finite Coxeter group.

    """

    def __new__(cls, cartantype):
        obj = Atom.__new__(cls)
        obj.cartan_type = CartanType(cartantype)
        return obj

    def generators(self):
        """
        This method creates the generating reflections of the Weyl group for
        a given Lie algebra.  For a Lie algebra of rank n, there are n
        different generating reflections.  This function returns them as
        a list.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("F4")
        >>> c.generators()
        ['r1', 'r2', 'r3', 'r4']
        """
        n = self.cartan_type.rank()
        generators = []
        for i in range(1, n+1):
            reflection = "r"+str(i)
            generators.append(reflection)
        return generators

    def group_order(self):
        """
        This method returns the order of the Weyl group.
        For types A, B, C, D, and E the order depends on
        the rank of the Lie algebra.  For types F and G,
        the order is fixed.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("D4")
        >>> c.group_order()
        192.0
        """
        n = self.cartan_type.rank()
        if self.cartan_type.series == "A":
            return fac(n+1)

        if self.cartan_type.series in ("B", "C"):
            return fac(n)*(2**n)

        if self.cartan_type.series == "D":
            return fac(n)*(2**(n-1))

        if self.cartan_type.series == "E":
            if n == 6:
                return 51840
            if n == 7:
                return 2903040
            if n == 8:
                return 696729600
        if self.cartan_type.series == "F":
            return 1152

        if self.cartan_type.series == "G":
            return 12

    def group_name(self):
        """
        This method returns some general information about the Weyl group for
        a given Lie algebra.  It returns the name of the group and the elements
        it acts on, if relevant.
        """
        n = self.cartan_type.rank()
        if self.cartan_type.series == "A":
            return "S"+str(n+1) + ": the symmetric group acting on " + str(n+1) + " elements."

        if self.cartan_type.series in ("B", "C"):
            return "The hyperoctahedral group acting on " + str(2*n) + " elements."

        if self.cartan_type.series == "D":
            return "The symmetry group of the " + str(n) + "-dimensional demihypercube."

        if self.cartan_type.series == "E":
            if n == 6:
                return "The symmetry group of the 6-polytope."

            if n == 7:
                return "The symmetry group of the 7-polytope."

            if n == 8:
                return "The symmetry group of the 8-polytope."

        if self.cartan_type.series == "F":
            return "The symmetry group of the 24-cell, or icositetrachoron."

        if self.cartan_type.series == "G":
            return "D6, the dihedral group of order 12, and symmetry group of the hexagon."

    def element_order(self, weylelt):
        """
        This method returns the order of a given Weyl group element, which should
        be specified by the user in the form of products of the generating
        reflections, i.e. of the form r1*r2 etc.

        For types A-F, this method current works by taking the matrix form of
        the specified element, and then finding what power of the matrix is the
        identity.  It then returns this power.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> b = WeylGroup("B4")
        >>> b.element_order('r1*r4*r2')
        4
        """
        n = self.cartan_type.rank()
        if self.cartan_type.series == "A":
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(n+1):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        if self.cartan_type.series == "D":
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(n):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        if self.cartan_type.series == "E":
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(8):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        if self.cartan_type.series == "G":
            elts = list(weylelt)
            reflections = elts[1::3]
            m = self.delete_doubles(reflections)
            while self.delete_doubles(m) != m:
                m = self.delete_doubles(m)
                reflections = m
            if len(reflections) % 2 == 1:
                return 2

            elif len(reflections) == 0:
                return 1

            else:
                if len(reflections) == 1:
                    return 2
                else:
                    m = len(reflections) // 2
                    lcm = (6 * m)/ igcd(m, 6)
                order = lcm / m
                return order


        if self.cartan_type.series == 'F':
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(4):
                a *= self.matrix_form(weylelt)
                order += 1
            return order


        if self.cartan_type.series in ("B", "C"):
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(n):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

    def delete_doubles(self, reflections):
        """
        This is a helper method for determining the order of an element in the
        Weyl group of G2.  It takes a Weyl element and if repeated simple reflections
        in it, it deletes them.
        """
        counter = 0
        copy = list(reflections)
        for elt in copy:
            if counter < len(copy)-1:
                if copy[counter + 1] == elt:
                    del copy[counter]
                    del copy[counter]
            counter += 1


        return copy


    def matrix_form(self, weylelt):
        """
        This method takes input from the user in the form of products of the
        generating reflections, and returns the matrix corresponding to the
        element of the Weyl group.  Since each element of the Weyl group is
        a reflection of some type, there is a corresponding matrix representation.
        This method uses the standard representation for all the generating
        reflections.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> f = WeylGroup("F4")
        >>> f.matrix_form('r2*r3')
        Matrix([
        [1, 0, 0,  0],
        [0, 1, 0,  0],
        [0, 0, 0, -1],
        [0, 0, 1,  0]])

        """
        elts = list(weylelt)
        reflections = elts[1::3]
        n = self.cartan_type.rank()
        if self.cartan_type.series == 'A':
            matrixform = eye(n+1)
            for elt in reflections:
                a = int(elt)
                mat = eye(n+1)
                mat[a-1, a-1] = 0
                mat[a-1, a] = 1
                mat[a, a-1] = 1
                mat[a, a] = 0
                matrixform *= mat
            return matrixform

        if self.cartan_type.series == 'D':
            matrixform = eye(n)
            for elt in reflections:
                a = int(elt)
                mat = eye(n)
                if a < n:
                    mat[a-1, a-1] = 0
                    mat[a-1, a] = 1
                    mat[a, a-1] = 1
                    mat[a, a] = 0
                    matrixform *= mat
                else:
                    mat[n-2, n-1] = -1
                    mat[n-2, n-2] = 0
                    mat[n-1, n-2] = -1
                    mat[n-1, n-1] = 0
                    matrixform *= mat
            return matrixform

        if self.cartan_type.series == 'G':
            matrixform = eye(3)
            for elt in reflections:
                a = int(elt)
                if a == 1:
                    gen1 = Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                    matrixform *= gen1
                else:
                    gen2 = Matrix([[Rational(2, 3), Rational(2, 3), Rational(-1, 3)],
                                   [Rational(2, 3), Rational(-1, 3), Rational(2, 3)],
                                   [Rational(-1, 3), Rational(2, 3), Rational(2, 3)]])
                    matrixform *= gen2
            return matrixform

        if self.cartan_type.series == 'F':
            matrixform = eye(4)
            for elt in reflections:
                a = int(elt)
                if a == 1:
                    mat = Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                    matrixform *= mat
                elif a == 2:
                    mat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
                    matrixform *= mat
                elif a == 3:
                    mat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
                    matrixform *= mat
                else:

                    mat = Matrix([[Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2)],
                        [Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2)],
                        [Rational(1, 2), Rational(-1, 2), Rational(1, 2), Rational(-1, 2)],
                        [Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]])
                    matrixform *= mat
            return matrixform

        if self.cartan_type.series == 'E':
            matrixform = eye(8)
            for elt in reflections:
                a = int(elt)
                if a == 1:
                    mat = Matrix([[Rational(3, 4), Rational(1, 4), Rational(1, 4), Rational(1, 4),
                        Rational(1, 4), Rational(1, 4), Rational(1, 4), Rational(-1, 4)],
                        [Rational(1, 4), Rational(3, 4), Rational(-1, 4), Rational(-1, 4),
                            Rational(-1, 4), Rational(-1, 4), Rational(1, 4), Rational(-1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(3, 4), Rational(-1, 4),
                        Rational(-1, 4), Rational(-1, 4), Rational(-1, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(3, 4),
                        Rational(-1, 4), Rational(-1, 4), Rational(-1, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4),
                        Rational(3, 4), Rational(-1, 4), Rational(-1, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4),
                        Rational(-1, 4), Rational(3, 4), Rational(-1, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4),
                        Rational(-1, 4), Rational(-1, 4), Rational(-3, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4),
                        Rational(-1, 4), Rational(-1, 4), Rational(-1, 4), Rational(3, 4)]])
                    matrixform *= mat
                elif a == 2:
                    mat = eye(8)
                    mat[0, 0] = 0
                    mat[0, 1] = -1
                    mat[1, 0] = -1
                    mat[1, 1] = 0
                    matrixform *= mat
                else:
                    mat = eye(8)
                    mat[a-3, a-3] = 0
                    mat[a-3, a-2] = 1
                    mat[a-2, a-3] = 1
                    mat[a-2, a-2] = 0
                    matrixform *= mat
            return matrixform


        if self.cartan_type.series in ("B", "C"):
            matrixform = eye(n)
            for elt in reflections:
                a = int(elt)
                mat = eye(n)
                if a == 1:
                    mat[0, 0] = -1
                    matrixform *= mat
                else:
                    mat[a - 2, a - 2] = 0
                    mat[a-2, a-1] = 1
                    mat[a - 1, a - 2] = 1
                    mat[a -1, a - 1] = 0
                    matrixform *= mat
            return matrixform



    def coxeter_diagram(self):
        """
        This method returns the Coxeter diagram corresponding to a Weyl group.
        The Coxeter diagram can be obtained from a Lie algebra's Dynkin diagram
        by deleting all arrows; the Coxeter diagram is the undirected graph.
        The vertices of the Coxeter diagram represent the generating reflections
        of the Weyl group, $s_i$.  An edge is drawn between $s_i$ and $s_j$ if the order
        $m(i, j)$ of $s_is_j$ is greater than two.  If there is one edge, the order
        $m(i, j)$ is 3.  If there are two edges, the order $m(i, j)$ is 4, and if there
        are three edges, the order $m(i, j)$ is 6.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("B3")
        >>> print(c.coxeter_diagram())
        0---0===0
        1   2   3
        """
        n = self.cartan_type.rank()
        if self.cartan_type.series in ("A", "D", "E"):
            return self.cartan_type.dynkin_diagram()

        if self.cartan_type.series in ("B", "C"):
            diag = "---".join("0" for i in range(1, n)) + "===0\n"
            diag += "   ".join(str(i) for i in range(1, n+1))
            return diag

        if self.cartan_type.series == "F":
            diag = "0---0===0---0\n"
            diag += "   ".join(str(i) for i in range(1, 5))
            return diag

        if self.cartan_type.series == "G":
            diag = "0≡≡≡0\n1   2"
            return diag
<|endoftext|>import itertools

from .cartan_type import Standard_Cartan
from sympy.core.backend import eye, Rational
from sympy.core.singleton import S

class TypeE(Standard_Cartan):

    def __new__(cls, n):
        if n < 6 or n > 8:
            raise ValueError("Invalid value of n")
        return Standard_Cartan.__new__(cls, "E", n)

    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("E6")
        >>> c.dimension()
        8
        """

        return 8

    def basic_root(self, i, j):
        """
        This is a method just to generate roots
        with a -1 in the ith position and a 1
        in the jth position.

        """

        root = [0]*8
        root[i] = -1
        root[j] = 1
        return root

    def simple_root(self, i):
        """
        Every Lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        This method returns the ith simple root for E_n.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("E6")
        >>> c.simple_root(2)
        [1, 1, 0, 0, 0, 0, 0, 0]
        """
        n = self.n
        if i == 1:
            root = [-0.5]*8
            root[0] = 0.5
            root[7] = 0.5
            return root
        elif i == 2:
            root = [0]*8
            root[1] = 1
            root[0] = 1
            return root
        else:
            if i in (7, 8) and n == 6:
                raise ValueError("E6 only has six simple roots!")
            if i == 8 and n == 7:
                raise ValueError("E7 only has seven simple roots!")

            return self.basic_root(i - 3, i - 2)

    def positive_roots(self):
        """
        This method generates all the positive roots of
        A_n.  This is half of all of the roots of E_n;
        by multiplying all the positive roots by -1 we
        get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
        """
        n = self.n
        neghalf = Rational(-1, 2)
        poshalf = S.Half
        if n == 6:
            posroots = {}
            k = 0
            for i in range(n-1):
                for j in range(i+1, n-1):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root

            root = [poshalf, poshalf, poshalf, poshalf, poshalf,
                    neghalf, neghalf, poshalf]
            for a, b, c, d, e in itertools.product(
                    range(2), range(2), range(2), range(2), range(2)):
                if (a + b + c + d + e)%2 == 0:
                    k += 1
                    if a == 1:
                        root[0] = neghalf
                    if b == 1:
                        root[1] = neghalf
                    if c == 1:
                        root[2] = neghalf
                    if d == 1:
                        root[3] = neghalf
                    if e == 1:
                        root[4] = neghalf
                    posroots[k] = root[:]
            return posroots
        if n == 7:
            posroots = {}
            k = 0
            for i in range(n-1):
                for j in range(i+1, n-1):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root

            k += 1
            posroots[k] = [0, 0, 0, 0, 0, 1, 1, 0]
            root = [poshalf, poshalf, poshalf, poshalf, poshalf,
                    neghalf, neghalf, poshalf]
            for a, b, c, d, e, f in itertools.product(
                    range(2), range(2), range(2), range(2), range(2), range(2)):
                if (a + b + c + d + e + f)%2 == 0:
                    k += 1
                    if a == 1:
                        root[0] = neghalf
                    if b == 1:
                        root[1] = neghalf
                    if c == 1:
                        root[2] = neghalf
                    if d == 1:
                        root[3] = neghalf
                    if e == 1:
                        root[4] = neghalf
                    if f == 1:
                        root[5] = poshalf
                    posroots[k] = root[:]
            return posroots
        if n == 8:
            posroots = {}
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root

            root = [poshalf, poshalf, poshalf, poshalf, poshalf,
                    neghalf, neghalf, poshalf]
            for a, b, c, d, e, f, g in itertools.product(
                    range(2), range(2), range(2), range(2), range(2),
                    range(2), range(2)):
                if (a + b + c + d + e + f + g)%2 == 0:
                    k += 1
                    if a == 1:
                        root[0] = neghalf
                    if b == 1:
                        root[1] = neghalf
                    if c == 1:
                        root[2] = neghalf
                    if d == 1:
                        root[3] = neghalf
                    if e == 1:
                        root[4] = neghalf
                    if f == 1:
                        root[5] = poshalf
                    if g == 1:
                        root[6] = poshalf
                    posroots[k] = root[:]
            return posroots



    def roots(self):
        """
        Returns the total number of roots of E_n
        """

        n = self.n
        if n == 6:
            return 72
        if n == 7:
            return 126
        if n == 8:
            return 240


    def cartan_matrix(self):
        """
        Returns the Cartan matrix for G_2
        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('A4')
        >>> c.cartan_matrix()
        Matrix([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]])


        """

        n = self.n
        m = 2*eye(n)
        for i in range(3, n - 1):
            m[i, i+1] = -1
            m[i, i-1] = -1
        m[0, 2] = m[2, 0] = -1
        m[1, 3] = m[3, 1] = -1
        m[2, 3] = -1
        m[n-1, n-2] = -1
        return m


    def basis(self):
        """
        Returns the number of independent generators of E_n
        """

        n = self.n
        if n == 6:
            return 78
        if n == 7:
            return 133
        if n == 8:
            return 248

    def dynkin_diagram(self):
        n = self.n
        diag = " "*8 + str(2) + "\n"
        diag += " "*8 + "0\n"
        diag += " "*8 + "|\n"
        diag += " "*8 + "|\n"
        diag += "---".join("0" for i in range(1, n)) + "\n"
        diag += "1   " + "   ".join(str(i) for i in range(3, n+1))
        return diag
<|endoftext|>from .cartan_type import CartanType
from sympy.core.basic import Atom

class RootSystem(Atom):
    """Represent the root system of a simple Lie algebra

    Every simple Lie algebra has a unique root system.  To find the root
    system, we first consider the Cartan subalgebra of g, which is the maximal
    abelian subalgebra, and consider the adjoint action of g on this
    subalgebra.  There is a root system associated with this action. Now, a
    root system over a vector space V is a set of finite vectors Phi (called
    roots), which satisfy:

    1.  The roots span V
    2.  The only scalar multiples of x in Phi are x and -x
    3.  For every x in Phi, the set Phi is closed under reflection
        through the hyperplane perpendicular to x.
    4.  If x and y are roots in Phi, then the projection of y onto
        the line through x is a half-integral multiple of x.

    Now, there is a subset of Phi, which we will call Delta, such that:
    1.  Delta is a basis of V
    2.  Each root x in Phi can be written x = sum k_y y for y in Delta

    The elements of Delta are called the simple roots.
    Therefore, we see that the simple roots span the root space of a given
    simple Lie algebra.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Root_system
    .. [2] Lie Algebras and Representation Theory - Humphreys

    """

    def __new__(cls, cartantype):
        """Create a new RootSystem object

        This method assigns an attribute called cartan_type to each instance of
        a RootSystem object.  When an instance of RootSystem is called, it
        needs an argument, which should be an instance of a simple Lie algebra.
        We then take the CartanType of this argument and set it as the
        cartan_type attribute of the RootSystem instance.

        """
        obj = Atom.__new__(cls)
        obj.cartan_type = CartanType(cartantype)
        return obj

    def simple_roots(self):
        """Generate the simple roots of the Lie algebra

        The rank of the Lie algebra determines the number of simple roots that
        it has.  This method obtains the rank of the Lie algebra, and then uses
        the simple_root method from the Lie algebra classes to generate all the
        simple roots.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> roots = c.simple_roots()
        >>> roots
        {1: [1, -1, 0, 0], 2: [0, 1, -1, 0], 3: [0, 0, 1, -1]}

        """
        n = self.cartan_type.rank()
        roots = {i: self.cartan_type.simple_root(i) for i in range(1, n+1)}
        return roots


    def all_roots(self):
        """Generate all the roots of a given root system

        The result is a dictionary where the keys are integer numbers.  It
        generates the roots by getting the dictionary of all positive roots
        from the bases classes, and then taking each root, and multiplying it
        by -1 and adding it to the dictionary.  In this way all the negative
        roots are generated.

        """
        alpha = self.cartan_type.positive_roots()
        keys = list(alpha.keys())
        k = max(keys)
        for val in keys:
            k += 1
            root = alpha[val]
            newroot = [-x for x in root]
            alpha[k] = newroot
        return alpha

    def root_space(self):
        """Return the span of the simple roots

        The root space is the vector space spanned by the simple roots, i.e. it
        is a vector space with a distinguished basis, the simple roots.  This
        method returns a string that represents the root space as the span of
        the simple roots, alpha[1],...., alpha[n].

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.root_space()
        'alpha[1] + alpha[2] + alpha[3]'

        """
        n = self.cartan_type.rank()
        rs = " + ".join("alpha["+str(i) +"]" for i in range(1, n+1))
        return rs

    def add_simple_roots(self, root1, root2):
        """Add two simple roots together

        The function takes as input two integers, root1 and root2.  It then
        uses these integers as keys in the dictionary of simple roots, and gets
        the corresponding simple roots, and then adds them together.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> newroot = c.add_simple_roots(1, 2)
        >>> newroot
        [1, 0, -1, 0]

        """

        alpha = self.simple_roots()
        if root1 > len(alpha) or root2 > len(alpha):
            raise ValueError("You've used a root that doesn't exist!")
        a1 = alpha[root1]
        a2 = alpha[root2]
        newroot = [_a1 + _a2 for _a1, _a2 in zip(a1, a2)]
        return newroot

    def add_as_roots(self, root1, root2):
        """Add two roots together if and only if their sum is also a root

        It takes as input two vectors which should be roots.  It then computes
        their sum and checks if it is in the list of all possible roots.  If it
        is, it returns the sum.  Otherwise it returns a string saying that the
        sum is not a root.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.add_as_roots([1, 0, -1, 0], [0, 0, 1, -1])
        [1, 0, 0, -1]
        >>> c.add_as_roots([1, -1, 0, 0], [0, 0, -1, 1])
        'The sum of these two roots is not a root'

        """
        alpha = self.all_roots()
        newroot = [r1 + r2 for r1, r2 in zip(root1, root2)]
        if newroot in alpha.values():
            return newroot
        else:
            return "The sum of these two roots is not a root"


    def cartan_matrix(self):
        """Cartan matrix of Lie algebra associated with this root system

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.cartan_matrix()
        Matrix([
            [ 2, -1,  0],
            [-1,  2, -1],
            [ 0, -1,  2]])
        """
        return self.cartan_type.cartan_matrix()

    def dynkin_diagram(self):
        """Dynkin diagram of the Lie algebra associated with this root system

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> print(c.dynkin_diagram())
        0---0---0
        1   2   3
        """
        return self.cartan_type.dynkin_diagram()
<|endoftext|>from .cartan_type import Standard_Cartan
from sympy.core.backend import eye

class TypeB(Standard_Cartan):

    def __new__(cls, n):
        if n < 2:
            raise ValueError("n cannot be less than 2")
        return Standard_Cartan.__new__(cls, "B", n)

    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("B3")
        >>> c.dimension()
        3
        """

        return self.n

    def basic_root(self, i, j):
        """
        This is a method just to generate roots
        with a 1 iin the ith position and a -1
        in the jth position.

        """
        root = [0]*self.n
        root[i] = 1
        root[j] = -1
        return root

    def simple_root(self, i):
        """
        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        In B_n the first n-1 simple roots are the same as the
        roots in A_(n-1) (a 1 in the ith position, a -1 in
        the (i+1)th position, and zeroes elsewhere).  The n-th
        simple root is the root with a 1 in the nth position
        and zeroes elsewhere.

        This method returns the ith simple root for the B series.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("B3")
        >>> c.simple_root(2)
        [0, 1, -1]

        """
        n = self.n
        if i < n:
            return self.basic_root(i-1, i)
        else:
            root = [0]*self.n
            root[n-1] = 1
            return root

    def positive_roots(self):
        """
        This method generates all the positive roots of
        A_n.  This is half of all of the roots of B_n;
        by multiplying all the positive roots by -1 we
        get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
        """

        n = self.n
        posroots = {}
        k = 0
        for i in range(0, n-1):
            for j in range(i+1, n):
                k += 1
                posroots[k] = self.basic_root(i, j)
                k += 1
                root = self.basic_root(i, j)
                root[j] = 1
                posroots[k] = root

        for i in range(0, n):
            k += 1
            root = [0]*n
            root[i] = 1
            posroots[k] = root

        return posroots

    def roots(self):
        """
        Returns the total number of roots for B_n"
        """

        n = self.n
        return 2*(n**2)

    def cartan_matrix(self):
        """
        Returns the Cartan matrix for B_n.
        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('B4')
        >>> c.cartan_matrix()
        Matrix([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -2],
        [ 0,  0, -1,  2]])

        """

        n = self.n
        m = 2* eye(n)
        for i in range(1, n - 1):
            m[i, i+1] = -1
            m[i, i-1] = -1
        m[0, 1] = -1
        m[n-2, n-1] = -2
        m[n-1, n-2] = -1
        return m

    def basis(self):
        """
        Returns the number of independent generators of B_n
        """

        n = self.n
        return (n**2 - n)/2

    def lie_algebra(self):
        """
        Returns the Lie algebra associated with B_n
        """

        n = self.n
        return "so(" + str(2*n) + ")"

    def dynkin_diagram(self):
        n = self.n
        diag = "---".join("0" for i in range(1, n)) + "=>=0\n"
        diag += "   ".join(str(i) for i in range(1, n+1))
        return diag
<|endoftext|>from __future__ import annotations
from typing import Any

from functools import reduce
from itertools import permutations

from sympy.combinatorics import Permutation
from sympy.core import (
    Basic, Expr, Function, diff,
    Pow, Mul, Add, Lambda, S, Tuple, Dict
)
from sympy.core.cache import cacheit

from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve

from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning,
                                        ignore_warnings)


# TODO you are a bit excessive in the use of Dummies
# TODO dummy point, literal field
# TODO too often one needs to call doit or simplify on the output, check the
# tests and find out why
from sympy.tensor.array import ImmutableDenseNDimArray


class Manifold(Basic):
    """
    A mathematical manifold.

    Explanation
    ===========

    A manifold is a topological space that locally resembles
    Euclidean space near each point [1].
    This class does not provide any means to study the topological
    characteristics of the manifold that it represents, though.

    Parameters
    ==========

    name : str
        The name of the manifold.

    dim : int
        The dimension of the manifold.

    Examples
    ========

    >>> from sympy.diffgeom import Manifold
    >>> m = Manifold('M', 2)
    >>> m
    M
    >>> m.dim
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Manifold
    """

    def __new__(cls, name, dim, **kwargs):
        if not isinstance(name, Str):
            name = Str(name)
        dim = _sympify(dim)
        obj = super().__new__(cls, name, dim)

        obj.patches = _deprecated_list(
            """
            Manifold.patches is deprecated. The Manifold object is now
            immutable. Instead use a separate list to keep track of the
            patches.
            """, [])
        return obj

    @property
    def name(self):
        return self.args[0]

    @property
    def dim(self):
        return self.args[1]


class Patch(Basic):
    """
    A patch on a manifold.

    Explanation
    ===========

    Coordinate patch, or patch in short, is a simply-connected open set around
    a point in the manifold [1]. On a manifold one can have many patches that
    do not always include the whole manifold. On these patches coordinate
    charts can be defined that permit the parameterization of any point on the
    patch in terms of a tuple of real numbers (the coordinates).

    This class does not provide any means to study the topological
    characteristics of the patch that it represents.

    Parameters
    ==========

    name : str
        The name of the patch.

    manifold : Manifold
        The manifold on which the patch is defined.

    Examples
    ========

    >>> from sympy.diffgeom import Manifold, Patch
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> p
    P
    >>> p.dim
    2

    References
    ==========

    .. [1] G. Sussman, J. Wisdom, W. Farr, Functional Differential Geometry
           (2013)

    """
    def __new__(cls, name, manifold, **kwargs):
        if not isinstance(name, Str):
            name = Str(name)
        obj = super().__new__(cls, name, manifold)

        obj.manifold.patches.append(obj) # deprecated
        obj.coord_systems = _deprecated_list(
            """
            Patch.coord_systms is deprecated. The Patch class is now
            immutable. Instead use a separate list to keep track of coordinate
            systems.
            """, [])
        return obj

    @property
    def name(self):
        return self.args[0]

    @property
    def manifold(self):
        return self.args[1]

    @property
    def dim(self):
        return self.manifold.dim


class CoordSystem(Basic):
    """
    A coordinate system defined on the patch.

    Explanation
    ===========

    Coordinate system is a system that uses one or more coordinates to uniquely
    determine the position of the points or other geometric elements on a
    manifold [1].

    By passing ``Symbols`` to *symbols* parameter, user can define the name and
    assumptions of coordinate symbols of the coordinate system. If not passed,
    these symbols are generated automatically and are assumed to be real valued.

    By passing *relations* parameter, user can define the transform relations of
    coordinate systems. Inverse transformation and indirect transformation can
    be found automatically. If this parameter is not passed, coordinate
    transformation cannot be done.

    Parameters
    ==========

    name : str
        The name of the coordinate system.

    patch : Patch
        The patch where the coordinate system is defined.

    symbols : list of Symbols, optional
        Defines the names and assumptions of coordinate symbols.

    relations : dict, optional
        Key is a tuple of two strings, who are the names of the systems where
        the coordinates transform from and transform to.
        Value is a tuple of the symbols before transformation and a tuple of
        the expressions after transformation.

    Examples
    ========

    We define two-dimensional Cartesian coordinate system and polar coordinate
    system.

    >>> from sympy import symbols, pi, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): [(x, y), (sqrt(x**2 + y**2), atan2(y, x))],
    ... ('Pol', 'Car2D'): [(r, theta), (r*cos(theta), r*sin(theta))]
    ... }
    >>> Car2D = CoordSystem('Car2D', p, (x, y), relation_dict)
    >>> Pol = CoordSystem('Pol', p, (r, theta), relation_dict)

    ``symbols`` property returns ``CoordinateSymbol`` instances. These symbols
    are not same with the symbols used to construct the coordinate system.

    >>> Car2D
    Car2D
    >>> Car2D.dim
    2
    >>> Car2D.symbols
    (x, y)
    >>> _[0].func
    <class 'sympy.diffgeom.diffgeom.CoordinateSymbol'>

    ``transformation()`` method returns the transformation function from
    one coordinate system to another. ``transform()`` method returns the
    transformed coordinates.

    >>> Car2D.transformation(Pol)
    Lambda((x, y), Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]]))
    >>> Car2D.transform(Pol)
    Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]])
    >>> Car2D.transform(Pol, [1, 2])
    Matrix([
    [sqrt(5)],
    [atan(2)]])

    ``jacobian()`` method returns the Jacobian matrix of coordinate
    transformation between two systems. ``jacobian_determinant()`` method
    returns the Jacobian determinant of coordinate transformation between two
    systems.

    >>> Pol.jacobian(Car2D)
    Matrix([
    [cos(theta), -r*sin(theta)],
    [sin(theta),  r*cos(theta)]])
    >>> Pol.jacobian(Car2D, [1, pi/2])
    Matrix([
    [0, -1],
    [1,  0]])
    >>> Car2D.jacobian_determinant(Pol)
    1/sqrt(x**2 + y**2)
    >>> Car2D.jacobian_determinant(Pol, [1,0])
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coordinate_system

    """
    def __new__(cls, name, patch, symbols=None, relations={}, **kwargs):
        if not isinstance(name, Str):
            name = Str(name)

        # canonicallize the symbols
        if symbols is None:
            names = kwargs.get('names', None)
            if names is None:
                symbols = Tuple(
                    *[Symbol('%s_%s' % (name.name, i), real=True)
                      for i in range(patch.dim)]
                )
            else:
                sympy_deprecation_warning(
                    f"""
The 'names' argument to CoordSystem is deprecated. Use 'symbols' instead. That
is, replace

    CoordSystem(..., names={names})

with

    CoordSystem(..., symbols=[{', '.join(["Symbol(" + repr(n) + ", real=True)" for n in names])}])
                    """,
                    deprecated_since_version="1.7",
                    active_deprecations_target="deprecated-diffgeom-mutable",
                )
                symbols = Tuple(
                    *[Symbol(n, real=True) for n in names]
                )
        else:
            syms = []
            for s in symbols:
                if isinstance(s, Symbol):
                    syms.append(Symbol(s.name, **s._assumptions.generator))
                elif isinstance(s, str):
                    sympy_deprecation_warning(
                        f"""

Passing a string as the coordinate symbol name to CoordSystem is deprecated.
Pass a Symbol with the appropriate name and assumptions instead.

That is, replace {s} with Symbol({s!r}, real=True).
                        """,

                        deprecated_since_version="1.7",
                        active_deprecations_target="deprecated-diffgeom-mutable",
                    )
                    syms.append(Symbol(s, real=True))
            symbols = Tuple(*syms)

        # canonicallize the relations
        rel_temp = {}
        for k,v in relations.items():
            s1, s2 = k
            if not isinstance(s1, Str):
                s1 = Str(s1)
            if not isinstance(s2, Str):
                s2 = Str(s2)
            key = Tuple(s1, s2)

            # Old version used Lambda as a value.
            if isinstance(v, Lambda):
                v = (tuple(v.signature), tuple(v.expr))
            else:
                v = (tuple(v[0]), tuple(v[1]))
            rel_temp[key] = v
        relations = Dict(rel_temp)

        # construct the object
        obj = super().__new__(cls, name, patch, symbols, relations)

        # Add deprecated attributes
        obj.transforms = _deprecated_dict(
            """
            CoordSystem.transforms is deprecated. The CoordSystem class is now
            immutable. Use the 'relations' keyword argument to the
            CoordSystems() constructor to specify relations.
            """, {})
        obj._names = [str(n) for n in symbols]
        obj.patch.coord_systems.append(obj) # deprecated
        obj._dummies = [Dummy(str(n)) for n in symbols] # deprecated
        obj._dummy = Dummy()

        return obj

    @property
    def name(self):
        return self.args[0]

    @property
    def patch(self):
        return self.args[1]

    @property
    def manifold(self):
        return self.patch.manifold

    @property
    def symbols(self):
        return tuple(CoordinateSymbol(self, i, **s._assumptions.generator)
            for i,s in enumerate(self.args[2]))

    @property
    def relations(self):
        return self.args[3]

    @property
    def dim(self):
        return self.patch.dim

    ##########################################################################
    # Finding transformation relation
    ##########################################################################

    def transformation(self, sys):
        """
        Return coordinate transformation function from *self* to *sys*.

        Parameters
        ==========

        sys : CoordSystem

        Returns
        =======

        sympy.Lambda

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transformation(R2_p)
        Lambda((x, y), Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]]))

        """
        signature = self.args[2]

        key = Tuple(self.name, sys.name)
        if self == sys:
            expr = Matrix(self.symbols)
        elif key in self.relations:
            expr = Matrix(self.relations[key][1])
        elif key[::-1] in self.relations:
            expr = Matrix(self._inverse_transformation(sys, self))
        else:
            expr = Matrix(self._indirect_transformation(self, sys))
        return Lambda(signature, expr)

    @staticmethod
    def _solve_inverse(sym1, sym2, exprs, sys1_name, sys2_name):
        ret = solve(
            [t[0] - t[1] for t in zip(sym2, exprs)],
            list(sym1), dict=True)

        if len(ret) == 0:
            temp = "Cannot solve inverse relation from {} to {}."
            raise NotImplementedError(temp.format(sys1_name, sys2_name))
        elif len(ret) > 1:
            temp = "Obtained multiple inverse relation from {} to {}."
            raise ValueError(temp.format(sys1_name, sys2_name))

        return ret[0]

    @classmethod
    def _inverse_transformation(cls, sys1, sys2):
        # Find the transformation relation from sys2 to sys1
        forward = sys1.transform(sys2)
        inv_results = cls._solve_inverse(sys1.symbols, sys2.symbols, forward,
                                         sys1.name, sys2.name)
        signature = tuple(sys1.symbols)
        return [inv_results[s] for s in signature]

    @classmethod
    @cacheit
    def _indirect_transformation(cls, sys1, sys2):
        # Find the transformation relation between two indirectly connected
        # coordinate systems
        rel = sys1.relations
        path = cls._dijkstra(sys1, sys2)

        transforms = []
        for s1, s2 in zip(path, path[1:]):
            if (s1, s2) in rel:
                transforms.append(rel[(s1, s2)])
            else:
                sym2, inv_exprs = rel[(s2, s1)]
                sym1 = tuple(Dummy() for i in sym2)
                ret = cls._solve_inverse(sym2, sym1, inv_exprs, s2, s1)
                ret = tuple(ret[s] for s in sym2)
                transforms.append((sym1, ret))
        syms = sys1.args[2]
        exprs = syms
        for newsyms, newexprs in transforms:
            exprs = tuple(e.subs(zip(newsyms, exprs)) for e in newexprs)
        return exprs

    @staticmethod
    def _dijkstra(sys1, sys2):
        # Use Dijkstra algorithm to find the shortest path between two indirectly-connected
        # coordinate systems
        # return value is the list of the names of the systems.
        relations = sys1.relations
        graph = {}
        for s1, s2 in relations.keys():
            if s1 not in graph:
                graph[s1] = {s2}
            else:
                graph[s1].add(s2)
            if s2 not in graph:
                graph[s2] = {s1}
            else:
                graph[s2].add(s1)

        path_dict = {sys:[0, [], 0] for sys in graph} # minimum distance, path, times of visited

        def visit(sys):
            path_dict[sys][2] = 1
            for newsys in graph[sys]:
                distance = path_dict[sys][0] + 1
                if path_dict[newsys][0] >= distance or not path_dict[newsys][1]:
                    path_dict[newsys][0] = distance
                    path_dict[newsys][1] = list(path_dict[sys][1])
                    path_dict[newsys][1].append(sys)

        visit(sys1.name)

        while True:
            min_distance = max(path_dict.values(), key=lambda x:x[0])[0]
            newsys = None
            for sys, lst in path_dict.items():
                if 0 < lst[0] <= min_distance and not lst[2]:
                    min_distance = lst[0]
                    newsys = sys
            if newsys is None:
                break
            visit(newsys)

        result = path_dict[sys2.name][1]
        result.append(sys2.name)

        if result == [sys2.name]:
            raise KeyError("Two coordinate systems are not connected.")
        return result

    def connect_to(self, to_sys, from_coords, to_exprs, inverse=True, fill_in_gaps=False):
        sympy_deprecation_warning(
            """
            The CoordSystem.connect_to() method is deprecated. Instead,
            generate a new instance of CoordSystem with the 'relations'
            keyword argument (CoordSystem classes are now immutable).
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
        )

        from_coords, to_exprs = dummyfy(from_coords, to_exprs)
        self.transforms[to_sys] = Matrix(from_coords), Matrix(to_exprs)

        if inverse:
            to_sys.transforms[self] = self._inv_transf(from_coords, to_exprs)

        if fill_in_gaps:
            self._fill_gaps_in_transformations()

    @staticmethod
    def _inv_transf(from_coords, to_exprs):
        # Will be removed when connect_to is removed
        inv_from = [i.as_dummy() for i in from_coords]
        inv_to = solve(
            [t[0] - t[1] for t in zip(inv_from, to_exprs)],
            list(from_coords), dict=True)[0]
        inv_to = [inv_to[fc] for fc in from_coords]
        return Matrix(inv_from), Matrix(inv_to)

    @staticmethod
    def _fill_gaps_in_transformations():
        # Will be removed when connect_to is removed
        raise NotImplementedError

    ##########################################################################
    # Coordinate transformations
    ##########################################################################

    def transform(self, sys, coordinates=None):
        """
        Return the result of coordinate transformation from *self* to *sys*.
        If coordinates are not given, coordinate symbols of *self* are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.ImmutableDenseMatrix containing CoordinateSymbol

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transform(R2_p)
        Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]])
        >>> R2_r.transform(R2_p, [0, 1])
        Matrix([
        [   1],
        [pi/2]])

        """
        if coordinates is None:
            coordinates = self.symbols
        if self != sys:
            transf = self.transformation(sys)
            coordinates = transf(*coordinates)
        else:
            coordinates = Matrix(coordinates)
        return coordinates

    def coord_tuple_transform_to(self, to_sys, coords):
        """Transform ``coords`` to coord system ``to_sys``."""
        sympy_deprecation_warning(
            """
            The CoordSystem.coord_tuple_transform_to() method is deprecated.
            Use the CoordSystem.transform() method instead.
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
        )

        coords = Matrix(coords)
        if self != to_sys:
            with ignore_warnings(SymPyDeprecationWarning):
                transf = self.transforms[to_sys]
            coords = transf[1].subs(list(zip(transf[0], coords)))
        return coords

    def jacobian(self, sys, coordinates=None):
        """
        Return the jacobian matrix of a transformation on given coordinates.
        If coordinates are not given, coordinate symbols of *self* are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.ImmutableDenseMatrix

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_p.jacobian(R2_r)
        Matrix([
        [cos(theta), -rho*sin(theta)],
        [sin(theta),  rho*cos(theta)]])
        >>> R2_p.jacobian(R2_r, [1, 0])
        Matrix([
        [1, 0],
        [0, 1]])

        """
        result = self.transform(sys).jacobian(self.symbols)
        if coordinates is not None:
            result = result.subs(list(zip(self.symbols, coordinates)))
        return result
    jacobian_matrix = jacobian

    def jacobian_determinant(self, sys, coordinates=None):
        """
        Return the jacobian determinant of a transformation on given
        coordinates. If coordinates are not given, coordinate symbols of *self*
        are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.Expr

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.jacobian_determinant(R2_p)
        1/sqrt(x**2 + y**2)
        >>> R2_r.jacobian_determinant(R2_p, [1, 0])
        1

        """
        return self.jacobian(sys, coordinates).det()


    ##########################################################################
    # Points
    ##########################################################################

    def point(self, coords):
        """Create a ``Point`` with coordinates given in this coord system."""
        return Point(self, coords)

    def point_to_coords(self, point):
        """Calculate the coordinates of a point in this coord system."""
        return point.coords(self)

    ##########################################################################
    # Base fields.
    ##########################################################################

    def base_scalar(self, coord_index):
        """Return ``BaseScalarField`` that takes a point and returns one of the coordinates."""
        return BaseScalarField(self, coord_index)
    coord_function = base_scalar

    def base_scalars(self):
        """Returns a list of all coordinate functions.
        For more details see the ``base_scalar`` method of this class."""
        return [self.base_scalar(i) for i in range(self.dim)]
    coord_functions = base_scalars

    def base_vector(self, coord_index):
        """Return a basis vector field.
        The basis vector field for this coordinate system. It is also an
        operator on scalar fields."""
        return BaseVectorField(self, coord_index)

    def base_vectors(self):
        """Returns a list of all base vectors.
        For more details see the ``base_vector`` method of this class."""
        return [self.base_vector(i) for i in range(self.dim)]

    def base_oneform(self, coord_index):
        """Return a basis 1-form field.
        The basis one-form field for this coordinate system. It is also an
        operator on vector fields."""
        return Differential(self.coord_function(coord_index))

    def base_oneforms(self):
        """Returns a list of all base oneforms.
        For more details see the ``base_oneform`` method of this class."""
        return [self.base_oneform(i) for i in range(self.dim)]


class CoordinateSymbol(Symbol):
    """A symbol which denotes an abstract value of i-th coordinate of
    the coordinate system with given context.

    Explanation
    ===========

    Each coordinates in coordinate system are represented by unique symbol,
    such as x, y, z in Cartesian coordinate system.

    You may not construct this class directly. Instead, use `symbols` method
    of CoordSystem.

    Parameters
    ==========

    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import symbols, Lambda, Matrix, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): Lambda((x, y), Matrix([sqrt(x**2 + y**2), atan2(y, x)])),
    ... ('Pol', 'Car2D'): Lambda((r, theta), Matrix([r*cos(theta), r*sin(theta)]))
    ... }
    >>> Car2D = CoordSystem('Car2D', p, [x, y], relation_dict)
    >>> Pol = CoordSystem('Pol', p, [r, theta], relation_dict)
    >>> x, y = Car2D.symbols

    ``CoordinateSymbol`` contains its coordinate symbol and index.

    >>> x.name
    'x'
    >>> x.coord_sys == Car2D
    True
    >>> x.index
    0
    >>> x.is_real
    True

    You can transform ``CoordinateSymbol`` into other coordinate system using
    ``rewrite()`` method.

    >>> x.rewrite(Pol)
    r*cos(theta)
    >>> sqrt(x**2 + y**2).rewrite(Pol).simplify()
    r

    """
    def __new__(cls, coord_sys, index, **assumptions):
        name = coord_sys.args[2][index].name
        obj = super().__new__(cls, name, **assumptions)
        obj.coord_sys = coord_sys
        obj.index = index
        return obj

    def __getnewargs__(self):
        return (self.coord_sys, self.index)

    def _hashable_content(self):
        return (
            self.coord_sys, self.index
        ) + tuple(sorted(self.assumptions0.items()))

    def _eval_rewrite(self, rule, args, **hints):
        if isinstance(rule, CoordSystem):
            return rule.transform(self.coord_sys)[self.index]
        return super()._eval_rewrite(rule, args, **hints)


class Point(Basic):
    """Point defined in a coordinate system.

    Explanation
    ===========

    Mathematically, point is defined in the manifold and does not have any coordinates
    by itself. Coordinate system is what imbues the coordinates to the point by coordinate
    chart. However, due to the difficulty of realizing such logic, you must supply
    a coordinate system and coordinates to define a Point here.

    The usage of this object after its definition is independent of the
    coordinate system that was used in order to define it, however due to
    limitations in the simplification routines you can arrive at complicated
    expressions if you use inappropriate coordinate systems.

    Parameters
    ==========

    coord_sys : CoordSystem

    coords : list
        The coordinates of the point.

    Examples
    ========

    >>> from sympy import pi
    >>> from sympy.diffgeom import Point
    >>> from sympy.diffgeom.rn import R2, R2_r, R2_p
    >>> rho, theta = R2_p.symbols

    >>> p = Point(R2_p, [rho, 3*pi/4])

    >>> p.manifold == R2
    True

    >>> p.coords()
    Matrix([
    [   rho],
    [3*pi/4]])
    >>> p.coords(R2_r)
    Matrix([
    [-sqrt(2)*rho/2],
    [ sqrt(2)*rho/2]])

    """

    def __new__(cls, coord_sys, coords, **kwargs):
        coords = Matrix(coords)
        obj = super().__new__(cls, coord_sys, coords)
        obj._coord_sys = coord_sys
        obj._coords = coords
        return obj

    @property
    def patch(self):
        return self._coord_sys.patch

    @property
    def manifold(self):
        return self._coord_sys.manifold

    @property
    def dim(self):
        return self.manifold.dim

    def coords(self, sys=None):
        """
        Coordinates of the point in given coordinate system. If coordinate system
        is not passed, it returns the coordinates in the coordinate system in which
        the point was defined.
        """
        if sys is None:
            return self._coords
        else:
            return self._coord_sys.transform(sys, self._coords)

    @property
    def free_symbols(self):
        return self._coords.free_symbols


class BaseScalarField(Expr):
    """Base scalar field over a manifold for a given coordinate system.

    Explanation
    ===========

    A scalar field takes a point as an argument and returns a scalar.
    A base scalar field of a coordinate system takes a point and returns one of
    the coordinates of that point in the coordinate system in question.

    To define a scalar field you need to choose the coordinate system and the
    index of the coordinate.

    The use of the scalar field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in
    the simplification routines you may arrive at more complicated
    expression if you use unappropriate coordinate systems.
    You can build complicated scalar fields by just building up SymPy
    expressions containing ``BaseScalarField`` instances.

    Parameters
    ==========

    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import Function, pi
    >>> from sympy.diffgeom import BaseScalarField
    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> rho, _ = R2_p.symbols
    >>> point = R2_p.point([rho, 0])
    >>> fx, fy = R2_r.base_scalars()
    >>> ftheta = BaseScalarField(R2_r, 1)

    >>> fx(point)
    rho
    >>> fy(point)
    0

    >>> (fx**2+fy**2).rcall(point)
    rho**2

    >>> g = Function('g')
    >>> fg = g(ftheta-pi)
    >>> fg.rcall(point)
    g(-pi)

    """

    is_commutative = True

    def __new__(cls, coord_sys, index, **kwargs):
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def patch(self):
        return self.coord_sys.patch

    @property
    def manifold(self):
        return self.coord_sys.manifold

    @property
    def dim(self):
        return self.manifold.dim

    def __call__(self, *args):
        """Evaluating the field at a point or doing nothing.
        If the argument is a ``Point`` instance, the field is evaluated at that
        point. The field is returned itself if the argument is any other
        object. It is so in order to have working recursive calling mechanics
        for all fields (check the ``__call__`` method of ``Expr``).
        """
        point = args[0]
        if len(args) != 1 or not isinstance(point, Point):
            return self
        coords = point.coords(self._coord_sys)
        # XXX Calling doit  is necessary with all the Subs expressions
        # XXX Calling simplify is necessary with all the trig expressions
        return simplify(coords[self._index]).doit()

    # XXX Workaround for limitations on the content of args
    free_symbols: set[Any] = set()


class BaseVectorField(Expr):
    r"""Base vector field over a manifold for a given coordinate system.

    Explanation
    ===========

    A vector field is an operator taking a scalar field and returning a
    directional derivative (which is also a scalar field).
    A base vector field is the same type of operator, however the derivation is
    specifically done with respect to a chosen coordinate.

    To define a base vector field you need to choose the coordinate system and
    the index of the coordinate.

    The use of the vector field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in the
    simplification routines you may arrive at more complicated expression if you
    use unappropriate coordinate systems.

    Parameters
    ==========
    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import BaseVectorField
    >>> from sympy import pprint

    >>> x, y = R2_r.symbols
    >>> rho, theta = R2_p.symbols
    >>> fx, fy = R2_r.base_scalars()
    >>> point_p = R2_p.point([rho, theta])
    >>> point_r = R2_r.point([x, y])

    >>> g = Function('g')
    >>> s_field = g(fx, fy)

    >>> v = BaseVectorField(R2_r, 1)
    >>> pprint(v(s_field))
    / d           \|
    |---(g(x, xi))||
    \dxi          /|xi=y
    >>> pprint(v(s_field).rcall(point_r).doit())
    d
    --(g(x, y))
    dy
    >>> pprint(v(s_field).rcall(point_p))
    / d                        \|
    |---(g(rho*cos(theta), xi))||
    \dxi                       /|xi=rho*sin(theta)

    """

    is_commutative = False

    def __new__(cls, coord_sys, index, **kwargs):
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def patch(self):
        return self.coord_sys.patch

    @property
    def manifold(self):
        return self.coord_sys.manifold

    @property
    def dim(self):
        return self.manifold.dim

    def __call__(self, scalar_field):
        """Apply on a scalar field.
        The action of a vector field on a scalar field is a directional
        differentiation.
        If the argument is not a scalar field an error is raised.
        """
        if covariant_order(scalar_field) or contravariant_order(scalar_field):
            raise ValueError('Only scalar fields can be supplied as arguments to vector fields.')

        if scalar_field is None:
            return self

        base_scalars = list(scalar_field.atoms(BaseScalarField))

        # First step: e_x(x+r**2) -> e_x(x) + 2*r*e_x(r)
        d_var = self._coord_sys._dummy
        # TODO: you need a real dummy function for the next line
        d_funcs = [Function('_#_%s' % i)(d_var) for i,
                   b in enumerate(base_scalars)]
        d_result = scalar_field.subs(list(zip(base_scalars, d_funcs)))
        d_result = d_result.diff(d_var)

        # Second step: e_x(x) -> 1 and e_x(r) -> cos(atan2(x, y))
        coords = self._coord_sys.symbols
        d_funcs_deriv = [f.diff(d_var) for f in d_funcs]
        d_funcs_deriv_sub = []
        for b in base_scalars:
            jac = self._coord_sys.jacobian(b._coord_sys, coords)
            d_funcs_deriv_sub.append(jac[b._index, self._index])
        d_result = d_result.subs(list(zip(d_funcs_deriv, d_funcs_deriv_sub)))

        # Remove the dummies
        result = d_result.subs(list(zip(d_funcs, base_scalars)))
        result = result.subs(list(zip(coords, self._coord_sys.coord_functions())))
        return result.doit()


def _find_coords(expr):
    # Finds CoordinateSystems existing in expr
    fields = expr.atoms(BaseScalarField, BaseVectorField)
    return {f._coord_sys for f in fields}


class Commutator(Expr):
    r"""Commutator of two vector fields.

    Explanation
    ===========

    The commutator of two vector fields `v_1` and `v_2` is defined as the
    vector field `[v_1, v_2]` that evaluated on each scalar field `f` is equal
    to `v_1(v_2(f)) - v_2(v_1(f))`.

    Examples
    ========


    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import Commutator
    >>> from sympy import simplify

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_r = R2_p.base_vector(0)

    >>> c_xy = Commutator(e_x, e_y)
    >>> c_xr = Commutator(e_x, e_r)
    >>> c_xy
    0

    Unfortunately, the current code is not able to compute everything:

    >>> c_xr
    Commutator(e_x, e_rho)
    >>> simplify(c_xr(fy**2))
    -2*cos(theta)*y**2/(x**2 + y**2)

    """
    def __new__(cls, v1, v2):
        if (covariant_order(v1) or contravariant_order(v1) != 1
                or covariant_order(v2) or contravariant_order(v2) != 1):
            raise ValueError(
                'Only commutators of vector fields are supported.')
        if v1 == v2:
            return S.Zero
        coord_sys = set().union(*[_find_coords(v) for v in (v1, v2)])
        if len(coord_sys) == 1:
            # Only one coordinate systems is used, hence it is easy enough to
            # actually evaluate the commutator.
            if all(isinstance(v, BaseVectorField) for v in (v1, v2)):
                return S.Zero
            bases_1, bases_2 = [list(v.atoms(BaseVectorField))
                                for v in (v1, v2)]
            coeffs_1 = [v1.expand().coeff(b) for b in bases_1]
            coeffs_2 = [v2.expand().coeff(b) for b in bases_2]
            res = 0
            for c1, b1 in zip(coeffs_1, bases_1):
                for c2, b2 in zip(coeffs_2, bases_2):
                    res += c1*b1(c2)*b2 - c2*b2(c1)*b1
            return res
        else:
            obj = super().__new__(cls, v1, v2)
            obj._v1 = v1 # deprecated assignment
            obj._v2 = v2 # deprecated assignment
            return obj

    @property
    def v1(self):
        return self.args[0]

    @property
    def v2(self):
        return self.args[1]

    def __call__(self, scalar_field):
        """Apply on a scalar field.
        If the argument is not a scalar field an error is raised.
        """
        return self.v1(self.v2(scalar_field)) - self.v2(self.v1(scalar_field))


class Differential(Expr):
    r"""Return the differential (exterior derivative) of a form field.

    Explanation
    ===========

    The differential of a form (i.e. the exterior derivative) has a complicated
    definition in the general case.
    The differential `df` of the 0-form `f` is defined for any vector field `v`
    as `df(v) = v(f)`.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import Differential
    >>> from sympy import pprint

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> g = Function('g')
    >>> s_field = g(fx, fy)
    >>> dg = Differential(s_field)

    >>> dg
    d(g(x, y))
    >>> pprint(dg(e_x))
    / d           \|
    |---(g(xi, y))||
    \dxi          /|xi=x
    >>> pprint(dg(e_y))
    / d           \|
    |---(g(x, xi))||
    \dxi          /|xi=y

    Applying the exterior derivative operator twice always results in:

    >>> Differential(dg)
    0
    """

    is_commutative = False

    def __new__(cls, form_field):
        if contravariant_order(form_field):
            raise ValueError(
                'A vector field was supplied as an argument to Differential.')
        if isinstance(form_field, Differential):
            return S.Zero
        else:
            obj = super().__new__(cls, form_field)
            obj._form_field = form_field # deprecated assignment
            return obj

    @property
    def form_field(self):
        return self.args[0]

    def __call__(self, *vector_fields):
        """Apply on a list of vector_fields.

        Explanation
        ===========

        If the number of vector fields supplied is not equal to 1 + the order of
        the form field inside the differential the result is undefined.

        For 1-forms (i.e. differentials of scalar fields) the evaluation is
        done as `df(v)=v(f)`. However if `v` is ``None`` instead of a vector
        field, the differential is returned unchanged. This is done in order to
        permit partial contractions for higher forms.

        In the general case the evaluation is done by applying the form field
        inside the differential on a list with one less elements than the number
        of elements in the original list. Lowering the number of vector fields
        is achieved through replacing each pair of fields by their
        commutator.

        If the arguments are not vectors or ``None``s an error is raised.
        """
        if any((contravariant_order(a) != 1 or covariant_order(a)) and a is not None
                for a in vector_fields):
            raise ValueError('The arguments supplied to Differential should be vector fields or Nones.')
        k = len(vector_fields)
        if k == 1:
            if vector_fields[0]:
                return vector_fields[0].rcall(self._form_field)
            return self
        else:
            # For higher form it is more complicated:
            # Invariant formula:
            # https://en.wikipedia.org/wiki/Exterior_derivative#Invariant_formula
            # df(v1, ... vn) = +/- vi(f(v1..no i..vn))
            #                  +/- f([vi,vj],v1..no i, no j..vn)
            f = self._form_field
            v = vector_fields
            ret = 0
            for i in range(k):
                t = v[i].rcall(f.rcall(*v[:i] + v[i + 1:]))
                ret += (-1)**i*t
                for j in range(i + 1, k):
                    c = Commutator(v[i], v[j])
                    if c:  # TODO this is ugly - the Commutator can be Zero and
                        # this causes the next line to fail
                        t = f.rcall(*(c,) + v[:i] + v[i + 1:j] + v[j + 1:])
                        ret += (-1)**(i + j)*t
            return ret


class TensorProduct(Expr):
    """Tensor product of forms.

    Explanation
    ===========

    The tensor product permits the creation of multilinear functionals (i.e.
    higher order tensors) out of lower order fields (e.g. 1-forms and vector
    fields). However, the higher tensors thus created lack the interesting
    features provided by the other type of product, the wedge product, namely
    they are not antisymmetric and hence are not form fields.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import TensorProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> TensorProduct(dx, dy)(e_x, e_y)
    1
    >>> TensorProduct(dx, dy)(e_y, e_x)
    0
    >>> TensorProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> TensorProduct(e_x, e_y)(fx**2, fy**2)
    4*x*y
    >>> TensorProduct(e_y, dx)(fy)
    dx

    You can nest tensor products.

    >>> tp1 = TensorProduct(dx, dy)
    >>> TensorProduct(tp1, dx)(e_x, e_y, e_x)
    1

    You can make partial contraction for instance when 'raising an index'.
    Putting ``None`` in the second argument of ``rcall`` means that the
    respective position in the tensor product is left as it is.

    >>> TP = TensorProduct
    >>> metric = TP(dx, dx) + 3*TP(dy, dy)
    >>> metric.rcall(e_y, None)
    3*dy

    Or automatically pad the args with ``None`` without specifying them.

    >>> metric.rcall(e_y)
    3*dy

    """
    def __new__(cls, *args):
        scalar = Mul(*[m for m in args if covariant_order(m) + contravariant_order(m) == 0])
        multifields = [m for m in args if covariant_order(m) + contravariant_order(m)]
        if multifields:
            if len(multifields) == 1:
                return scalar*multifields[0]
            return scalar*super().__new__(cls, *multifields)
        else:
            return scalar

    def __call__(self, *fields):
        """Apply on a list of fields.

        If the number of input fields supplied is not equal to the order of
        the tensor product field, the list of arguments is padded with ``None``'s.

        The list of arguments is divided in sublists depending on the order of
        the forms inside the tensor product. The sublists are provided as
        arguments to these forms and the resulting expressions are given to the
        constructor of ``TensorProduct``.

        """
        tot_order = covariant_order(self) + contravariant_order(self)
        tot_args = len(fields)
        if tot_args != tot_order:
            fields = list(fields) + [None]*(tot_order - tot_args)
        orders = [covariant_order(f) + contravariant_order(f) for f in self._args]
        indices = [sum(orders[:i + 1]) for i in range(len(orders) - 1)]
        fields = [fields[i:j] for i, j in zip([0] + indices, indices + [None])]
        multipliers = [t[0].rcall(*t[1]) for t in zip(self._args, fields)]
        return TensorProduct(*multipliers)


class WedgeProduct(TensorProduct):
    """Wedge product of forms.

    Explanation
    ===========

    In the context of integration only completely antisymmetric forms make
    sense. The wedge product permits the creation of such forms.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import WedgeProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> WedgeProduct(dx, dy)(e_x, e_y)
    1
    >>> WedgeProduct(dx, dy)(e_y, e_x)
    -1
    >>> WedgeProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> WedgeProduct(e_x, e_y)(fy, None)
    -e_x

    You can nest wedge products.

    >>> wp1 = WedgeProduct(dx, dy)
    >>> WedgeProduct(wp1, dx)(e_x, e_y, e_x)
    0

    """
    # TODO the calculation of signatures is slow
    # TODO you do not need all these permutations (neither the prefactor)
    def __call__(self, *fields):
        """Apply on a list of vector_fields.
        The expression is rewritten internally in terms of tensor products and evaluated."""
        orders = (covariant_order(e) + contravariant_order(e) for e in self.args)
        mul = 1/Mul(*(factorial(o) for o in orders))
        perms = permutations(fields)
        perms_par = (Permutation(
            p).signature() for p in permutations(range(len(fields))))
        tensor_prod = TensorProduct(*self.args)
        return mul*Add(*[tensor_prod(*p[0])*p[1] for p in zip(perms, perms_par)])


class LieDerivative(Expr):
    """Lie derivative with respect to a vector field.

    Explanation
    ===========

    The transport operator that defines the Lie derivative is the pushforward of
    the field to be derived along the integral curve of the field with respect
    to which one derives.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> from sympy.diffgeom import (LieDerivative, TensorProduct)

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_rho, e_theta = R2_p.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> LieDerivative(e_x, fy)
    0
    >>> LieDerivative(e_x, fx)
    1
    >>> LieDerivative(e_x, e_x)
    0

    The Lie derivative of a tensor field by another tensor field is equal to
    their commutator:

    >>> LieDerivative(e_x, e_rho)
    Commutator(e_x, e_rho)
    >>> LieDerivative(e_x + e_y, fx)
    1

    >>> tp = TensorProduct(dx, dy)
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))

    """
    def __new__(cls, v_field, expr):
        expr_form_ord = covariant_order(expr)
        if contravariant_order(v_field) != 1 or covariant_order(v_field):
            raise ValueError('Lie derivatives are defined only with respect to'
                             ' vector fields. The supplied argument was not a '
                             'vector field.')
        if expr_form_ord > 0:
            obj = super().__new__(cls, v_field, expr)
            # deprecated assignments
            obj._v_field = v_field
            obj._expr = expr
            return obj
        if expr.atoms(BaseVectorField):
            return Commutator(v_field, expr)
        else:
            return v_field.rcall(expr)

    @property
    def v_field(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    def __call__(self, *args):
        v = self.v_field
        expr = self.expr
        lead_term = v(expr(*args))
        rest = Add(*[Mul(*args[:i] + (Commutator(v, args[i]),) + args[i + 1:])
                     for i in range(len(args))])
        return lead_term - rest


class BaseCovarDerivativeOp(Expr):
    """Covariant derivative operator with respect to a base vector.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import BaseCovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct

    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))
    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    >>> cvd(fx)
    1
    >>> cvd(fx*e_x)
    e_x
    """

    def __new__(cls, coord_sys, index, christoffel):
        index = _sympify(index)
        christoffel = ImmutableDenseNDimArray(christoffel)
        obj = super().__new__(cls, coord_sys, index, christoffel)
        # deprecated assignments
        obj._coord_sys = coord_sys
        obj._index = index
        obj._christoffel = christoffel
        return obj

    @property
    def coord_sys(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def christoffel(self):
        return self.args[2]

    def __call__(self, field):
        """Apply on a scalar field.

        The action of a vector field on a scalar field is a directional
        differentiation.
        If the argument is not a scalar field the behaviour is undefined.
        """
        if covariant_order(field) != 0:
            raise NotImplementedError()

        field = vectors_in_basis(field, self._coord_sys)

        wrt_vector = self._coord_sys.base_vector(self._index)
        wrt_scalar = self._coord_sys.coord_function(self._index)
        vectors = list(field.atoms(BaseVectorField))

        # First step: replace all vectors with something susceptible to
        # derivation and do the derivation
        # TODO: you need a real dummy function for the next line
        d_funcs = [Function('_#_%s' % i)(wrt_scalar) for i,
                   b in enumerate(vectors)]
        d_result = field.subs(list(zip(vectors, d_funcs)))
        d_result = wrt_vector(d_result)

        # Second step: backsubstitute the vectors in
        d_result = d_result.subs(list(zip(d_funcs, vectors)))

        # Third step: evaluate the derivatives of the vectors
        derivs = []
        for v in vectors:
            d = Add(*[(self._christoffel[k, wrt_vector._index, v._index]
                       *v._coord_sys.base_vector(k))
                      for k in range(v._coord_sys.dim)])
            derivs.append(d)
        to_subs = [wrt_vector(d) for d in d_funcs]
        # XXX: This substitution can fail when there are Dummy symbols and the
        # cache is disabled: https://github.com/sympy/sympy/issues/17794
        result = d_result.subs(list(zip(to_subs, derivs)))

        # Remove the dummies
        result = result.subs(list(zip(d_funcs, vectors)))
        return result.doit()


class CovarDerivativeOp(Expr):
    """Covariant derivative operator.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import CovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct
    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()
    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))

    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = CovarDerivativeOp(fx*e_x, ch)
    >>> cvd(fx)
    x
    >>> cvd(fx*e_x)
    x*e_x

    """

    def __new__(cls, wrt, christoffel):
        if len({v._coord_sys for v in wrt.atoms(BaseVectorField)}) > 1:
            raise NotImplementedError()
        if contravariant_order(wrt) != 1 or covariant_order(wrt):
            raise ValueError('Covariant derivatives are defined only with '
                             'respect to vector fields. The supplied argument '
                             'was not a vector field.')
        christoffel = ImmutableDenseNDimArray(christoffel)
        obj = super().__new__(cls, wrt, christoffel)
        # deprecated assignments
        obj._wrt = wrt
        obj._christoffel = christoffel
        return obj

    @property
    def wrt(self):
        return self.args[0]

    @property
    def christoffel(self):
        return self.args[1]

    def __call__(self, field):
        vectors = list(self._wrt.atoms(BaseVectorField))
        base_ops = [BaseCovarDerivativeOp(v._coord_sys, v._index, self._christoffel)
                    for v in vectors]
        return self._wrt.subs(list(zip(vectors, base_ops))).rcall(field)


###############################################################################
# Integral curves on vector fields
###############################################################################
def intcurve_series(vector_field, param, start_point, n=6, coord_sys=None, coeffs=False):
    r"""Return the series expansion for an integral curve of the field.

    Explanation
    ===========

    Integral curve is a function `\gamma` taking a parameter in `R` to a point
    in the manifold. It verifies the equation:

    `V(f)\big(\gamma(t)\big) = \frac{d}{dt}f\big(\gamma(t)\big)`

    where the given ``vector_field`` is denoted as `V`. This holds for any
    value `t` for the parameter and any scalar field `f`.

    This equation can also be decomposed of a basis of coordinate functions
    `V(f_i)\big(\gamma(t)\big) = \frac{d}{dt}f_i\big(\gamma(t)\big) \quad \forall i`

    This function returns a series expansion of `\gamma(t)` in terms of the
    coordinate system ``coord_sys``. The equations and expansions are necessarily
    done in coordinate-system-dependent way as there is no other way to
    represent movement between points on the manifold (i.e. there is no such
    thing as a difference of points for a general manifold).

    Parameters
    ==========
    vector_field
        the vector field for which an integral curve will be given

    param
        the argument of the function `\gamma` from R to the curve

    start_point
        the point which corresponds to `\gamma(0)`

    n
        the order to which to expand

    coord_sys
        the coordinate system in which to expand
        coeffs (default False) - if True return a list of elements of the expansion

    Examples
    ========

    Use the predefined R2 manifold:

    >>> from sympy.abc import t, x, y
    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import intcurve_series

    Specify a starting point and a vector field:

    >>> start_point = R2_r.point([x, y])
    >>> vector_field = R2_r.e_x

    Calculate the series:

    >>> intcurve_series(vector_field, t, start_point, n=3)
    Matrix([
    [t + x],
    [    y]])

    Or get the elements of the expansion in a list:

    >>> series = intcurve_series(vector_field, t, start_point, n=3, coeffs=True)
    >>> series[0]
    Matrix([
    [x],
    [y]])
    >>> series[1]
    Matrix([
    [t],
    [0]])
    >>> series[2]
    Matrix([
    [0],
    [0]])

    The series in the polar coordinate system:

    >>> series = intcurve_series(vector_field, t, start_point,
    ...             n=3, coord_sys=R2_p, coeffs=True)
    >>> series[0]
    Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]])
    >>> series[1]
    Matrix([
    [t*x/sqrt(x**2 + y**2)],
    [   -t*y/(x**2 + y**2)]])
    >>> series[2]
    Matrix([
    [t**2*(-x**2/(x**2 + y**2)**(3/2) + 1/sqrt(x**2 + y**2))/2],
    [                                t**2*x*y/(x**2 + y**2)**2]])

    See Also
    ========

    intcurve_diffequ

    """
    if contravariant_order(vector_field) != 1 or covariant_order(vector_field):
        raise ValueError('The supplied field was not a vector field.')

    def iter_vfield(scalar_field, i):
        """Return ``vector_field`` called `i` times on ``scalar_field``."""
        return reduce(lambda s, v: v.rcall(s), [vector_field, ]*i, scalar_field)

    def taylor_terms_per_coord(coord_function):
        """Return the series for one of the coordinates."""
        return [param**i*iter_vfield(coord_function, i).rcall(start_point)/factorial(i)
                for i in range(n)]
    coord_sys = coord_sys if coord_sys else start_point._coord_sys
    coord_functions = coord_sys.coord_functions()
    taylor_terms = [taylor_terms_per_coord(f) for f in coord_functions]
    if coeffs:
        return [Matrix(t) for t in zip(*taylor_terms)]
    else:
        return Matrix([sum(c) for c in taylor_terms])


def intcurve_diffequ(vector_field, param, start_point, coord_sys=None):
    r"""Return the differential equation for an integral curve of the field.

    Explanation
    ===========

    Integral curve is a function `\gamma` taking a parameter in `R` to a point
    in the manifold. It verifies the equation:

    `V(f)\big(\gamma(t)\big) = \frac{d}{dt}f\big(\gamma(t)\big)`

    where the given ``vector_field`` is denoted as `V`. This holds for any
    value `t` for the parameter and any scalar field `f`.

    This function returns the differential equation of `\gamma(t)` in terms of the
    coordinate system ``coord_sys``. The equations and expansions are necessarily
    done in coordinate-system-dependent way as there is no other way to
    represent movement between points on the manifold (i.e. there is no such
    thing as a difference of points for a general manifold).

    Parameters
    ==========

    vector_field
        the vector field for which an integral curve will be given

    param
        the argument of the function `\gamma` from R to the curve

    start_point
        the point which corresponds to `\gamma(0)`

    coord_sys
        the coordinate system in which to give the equations

    Returns
    =======

    a tuple of (equations, initial conditions)

    Examples
    ========

    Use the predefined R2 manifold:

    >>> from sympy.abc import t
    >>> from sympy.diffgeom.rn import R2, R2_p, R2_r
    >>> from sympy.diffgeom import intcurve_diffequ

    Specify a starting point and a vector field:

    >>> start_point = R2_r.point([0, 1])
    >>> vector_field = -R2.y*R2.e_x + R2.x*R2.e_y

    Get the equation:

    >>> equations, init_cond = intcurve_diffequ(vector_field, t, start_point)
    >>> equations
    [f_1(t) + Derivative(f_0(t), t), -f_0(t) + Derivative(f_1(t), t)]
    >>> init_cond
    [f_0(0), f_1(0) - 1]

    The series in the polar coordinate system:

    >>> equations, init_cond = intcurve_diffequ(vector_field, t, start_point, R2_p)
    >>> equations
    [Derivative(f_0(t), t), Derivative(f_1(t), t) - 1]
    >>> init_cond
    [f_0(0) - 1, f_1(0) - pi/2]

    See Also
    ========

    intcurve_series

    """
    if contravariant_order(vector_field) != 1 or covariant_order(vector_field):
        raise ValueError('The supplied field was not a vector field.')
    coord_sys = coord_sys if coord_sys else start_point._coord_sys
    gammas = [Function('f_%d' % i)(param) for i in range(
        start_point._coord_sys.dim)]
    arbitrary_p = Point(coord_sys, gammas)
    coord_functions = coord_sys.coord_functions()
    equations = [simplify(diff(cf.rcall(arbitrary_p), param) - vector_field.rcall(cf).rcall(arbitrary_p))
                 for cf in coord_functions]
    init_cond = [simplify(cf.rcall(arbitrary_p).subs(param, 0) - cf.rcall(start_point))
                 for cf in coord_functions]
    return equations, init_cond


###############################################################################
# Helpers
###############################################################################
def dummyfy(args, exprs):
    # TODO Is this a good idea?
    d_args = Matrix([s.as_dummy() for s in args])
    reps = dict(zip(args, d_args))
    d_exprs = Matrix([_sympify(expr).subs(reps) for expr in exprs])
    return d_args, d_exprs

###############################################################################
# Helpers
###############################################################################
def contravariant_order(expr, _strict=False):
    """Return the contravariant order of an expression.

    Examples
    ========

    >>> from sympy.diffgeom import contravariant_order
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.abc import a

    >>> contravariant_order(a)
    0
    >>> contravariant_order(a*R2.x + 2)
    0
    >>> contravariant_order(a*R2.x*R2.e_y + R2.e_x)
    1

    """
    # TODO move some of this to class methods.
    # TODO rewrite using the .as_blah_blah methods
    if isinstance(expr, Add):
        orders = [contravariant_order(e) for e in expr.args]
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing contravariant fields of varying order.')
        return orders[0]
    elif isinstance(expr, Mul):
        orders = [contravariant_order(e) for e in expr.args]
        not_zero = [o for o in orders if o != 0]
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between vectors.')
        return 0 if not not_zero else not_zero[0]
    elif isinstance(expr, Pow):
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError(
                'Misformed expression containing a power of a vector.')
        return 0
    elif isinstance(expr, BaseVectorField):
        return 1
    elif isinstance(expr, TensorProduct):
        return sum(contravariant_order(a) for a in expr.args)
    elif not _strict or expr.atoms(BaseScalarField):
        return 0
    else:  # If it does not contain anything related to the diffgeom module and it is _strict
        return -1


def covariant_order(expr, _strict=False):
    """Return the covariant order of an expression.

    Examples
    ========

    >>> from sympy.diffgeom import covariant_order
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.abc import a

    >>> covariant_order(a)
    0
    >>> covariant_order(a*R2.x + 2)
    0
    >>> covariant_order(a*R2.x*R2.dy + R2.dx)
    1

    """
    # TODO move some of this to class methods.
    # TODO rewrite using the .as_blah_blah methods
    if isinstance(expr, Add):
        orders = [covariant_order(e) for e in expr.args]
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing form fields of varying order.')
        return orders[0]
    elif isinstance(expr, Mul):
        orders = [covariant_order(e) for e in expr.args]
        not_zero = [o for o in orders if o != 0]
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between forms.')
        return 0 if not not_zero else not_zero[0]
    elif isinstance(expr, Pow):
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError(
                'Misformed expression containing a power of a form.')
        return 0
    elif isinstance(expr, Differential):
        return covariant_order(*expr.args) + 1
    elif isinstance(expr, TensorProduct):
        return sum(covariant_order(a) for a in expr.args)
    elif not _strict or expr.atoms(BaseScalarField):
        return 0
    else:  # If it does not contain anything related to the diffgeom module and it is _strict
        return -1


###############################################################################
# Coordinate transformation functions
###############################################################################
def vectors_in_basis(expr, to_sys):
    """Transform all base vectors in base vectors of a specified coord basis.
    While the new base vectors are in the new coordinate system basis, any
    coefficients are kept in the old system.

    Examples
    ========

    >>> from sympy.diffgeom import vectors_in_basis
    >>> from sympy.diffgeom.rn import R2_r, R2_p

    >>> vectors_in_basis(R2_r.e_x, R2_p)
    -y*e_theta/(x**2 + y**2) + x*e_rho/sqrt(x**2 + y**2)
    >>> vectors_in_basis(R2_p.e_r, R2_r)
    sin(theta)*e_y + cos(theta)*e_x

    """
    vectors = list(expr.atoms(BaseVectorField))
    new_vectors = []
    for v in vectors:
        cs = v._coord_sys
        jac = cs.jacobian(to_sys, cs.coord_functions())
        new = (jac.T*Matrix(to_sys.base_vectors()))[v._index]
        new_vectors.append(new)
    return expr.subs(list(zip(vectors, new_vectors)))


###############################################################################
# Coordinate-dependent functions
###############################################################################
def twoform_to_matrix(expr):
    """Return the matrix representing the twoform.

    For the twoform `w` return the matrix `M` such that `M[i,j]=w(e_i, e_j)`,
    where `e_i` is the i-th base vector field for the coordinate system in
    which the expression of `w` is given.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import twoform_to_matrix, TensorProduct
    >>> TP = TensorProduct

    >>> twoform_to_matrix(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    Matrix([
    [1, 0],
    [0, 1]])
    >>> twoform_to_matrix(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    Matrix([
    [x, 0],
    [0, 1]])
    >>> twoform_to_matrix(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy) - TP(R2.dx, R2.dy)/2)
    Matrix([
    [   1, 0],
    [-1/2, 1]])

    """
    if covariant_order(expr) != 2 or contravariant_order(expr):
        raise ValueError('The input expression is not a two-form.')
    coord_sys = _find_coords(expr)
    if len(coord_sys) != 1:
        raise ValueError('The input expression concerns more than one '
                         'coordinate systems, hence there is no unambiguous '
                         'way to choose a coordinate system for the matrix.')
    coord_sys = coord_sys.pop()
    vectors = coord_sys.base_vectors()
    expr = expr.expand()
    matrix_content = [[expr.rcall(v1, v2) for v1 in vectors]
                      for v2 in vectors]
    return Matrix(matrix_content)


def metric_to_Christoffel_1st(expr):
    """Return the nested list of Christoffel symbols for the given metric.
    This returns the Christoffel symbol of first kind that represents the
    Levi-Civita connection for the given metric.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Christoffel_1st, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Christoffel_1st(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> metric_to_Christoffel_1st(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[1/2, 0], [0, 0]], [[0, 0], [0, 0]]]

    """
    matrix = twoform_to_matrix(expr)
    if not matrix.is_symmetric():
        raise ValueError(
            'The two-form representing the metric is not symmetric.')
    coord_sys = _find_coords(expr).pop()
    deriv_matrices = [matrix.applyfunc(d) for d in coord_sys.base_vectors()]
    indices = list(range(coord_sys.dim))
    christoffel = [[[(deriv_matrices[k][i, j] + deriv_matrices[j][i, k] - deriv_matrices[i][j, k])/2
                     for k in indices]
                    for j in indices]
                   for i in indices]
    return ImmutableDenseNDimArray(christoffel)


def metric_to_Christoffel_2nd(expr):
    """Return the nested list of Christoffel symbols for the given metric.
    This returns the Christoffel symbol of second kind that represents the
    Levi-Civita connection for the given metric.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Christoffel_2nd(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> metric_to_Christoffel_2nd(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[1/(2*x), 0], [0, 0]], [[0, 0], [0, 0]]]

    """
    ch_1st = metric_to_Christoffel_1st(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    # XXX workaround, inverting a matrix does not work if it contains non
    # symbols
    #matrix = twoform_to_matrix(expr).inv()
    matrix = twoform_to_matrix(expr)
    s_fields = set()
    for e in matrix:
        s_fields.update(e.atoms(BaseScalarField))
    s_fields = list(s_fields)
    dums = coord_sys.symbols
    matrix = matrix.subs(list(zip(s_fields, dums))).inv().subs(list(zip(dums, s_fields)))
    # XXX end of workaround
    christoffel = [[[Add(*[matrix[i, l]*ch_1st[l, j, k] for l in indices])
                     for k in indices]
                    for j in indices]
                   for i in indices]
    return ImmutableDenseNDimArray(christoffel)


def metric_to_Riemann_components(expr):
    """Return the components of the Riemann tensor expressed in a given basis.

    Given a metric it calculates the components of the Riemann tensor in the
    canonical basis of the coordinate system in which the metric expression is
    given.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Riemann_components, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Riemann_components(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]
    >>> non_trivial_metric = exp(2*R2.r)*TP(R2.dr, R2.dr) + \
        R2.r**2*TP(R2.dtheta, R2.dtheta)
    >>> non_trivial_metric
    exp(2*rho)*TensorProduct(drho, drho) + rho**2*TensorProduct(dtheta, dtheta)
    >>> riemann = metric_to_Riemann_components(non_trivial_metric)
    >>> riemann[0, :, :, :]
    [[[0, 0], [0, 0]], [[0, exp(-2*rho)*rho], [-exp(-2*rho)*rho, 0]]]
    >>> riemann[1, :, :, :]
    [[[0, -1/rho], [1/rho, 0]], [[0, 0], [0, 0]]]

    """
    ch_2nd = metric_to_Christoffel_2nd(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    deriv_ch = [[[[d(ch_2nd[i, j, k])
                   for d in coord_sys.base_vectors()]
                  for k in indices]
                 for j in indices]
                for i in indices]
    riemann_a = [[[[deriv_ch[rho][sig][nu][mu] - deriv_ch[rho][sig][mu][nu]
                    for nu in indices]
                   for mu in indices]
                  for sig in indices]
                     for rho in indices]
    riemann_b = [[[[Add(*[ch_2nd[rho, l, mu]*ch_2nd[l, sig, nu] - ch_2nd[rho, l, nu]*ch_2nd[l, sig, mu] for l in indices])
                    for nu in indices]
                   for mu in indices]
                  for sig in indices]
                 for rho in indices]
    riemann = [[[[riemann_a[rho][sig][mu][nu] + riemann_b[rho][sig][mu][nu]
                  for nu in indices]
                     for mu in indices]
                for sig in indices]
               for rho in indices]
    return ImmutableDenseNDimArray(riemann)


def metric_to_Ricci_components(expr):

    """Return the components of the Ricci tensor expressed in a given basis.

    Given a metric it calculates the components of the Ricci tensor in the
    canonical basis of the coordinate system in which the metric expression is
    given.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Ricci_components, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Ricci_components(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[0, 0], [0, 0]]
    >>> non_trivial_metric = exp(2*R2.r)*TP(R2.dr, R2.dr) + \
                             R2.r**2*TP(R2.dtheta, R2.dtheta)
    >>> non_trivial_metric
    exp(2*rho)*TensorProduct(drho, drho) + rho**2*TensorProduct(dtheta, dtheta)
    >>> metric_to_Ricci_components(non_trivial_metric)
    [[1/rho, 0], [0, exp(-2*rho)*rho]]

    """
    riemann = metric_to_Riemann_components(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    ricci = [[Add(*[riemann[k, i, k, j] for k in indices])
              for j in indices]
             for i in indices]
    return ImmutableDenseNDimArray(ricci)

###############################################################################
# Classes for deprecation
###############################################################################

class _deprecated_container:
    # This class gives deprecation warning.
    # When deprecated features are completely deleted, this should be removed as well.
    # See https://github.com/sympy/sympy/pull/19368
    def __init__(self, message, data):
        super().__init__(data)
        self.message = message

    def warn(self):
        sympy_deprecation_warning(
            self.message,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
            stacklevel=4
        )

    def __iter__(self):
        self.warn()
        return super().__iter__()

    def __getitem__(self, key):
        self.warn()
        return super().__getitem__(key)

    def __contains__(self, key):
        self.warn()
        return super().__contains__(key)


class _deprecated_list(_deprecated_container, list):
    pass


class _deprecated_dict(_deprecated_container, dict):
    pass


# Import at end to avoid cyclic imports
from sympy.simplify.simplify import simplify
<|endoftext|>"""Predefined R^n manifolds together with common coord. systems.

Coordinate systems are predefined as well as the transformation laws between
them.

Coordinate functions can be accessed as attributes of the manifold (eg `R2.x`),
as attributes of the coordinate systems (eg `R2_r.x` and `R2_p.theta`), or by
using the usual `coord_sys.coord_function(index, name)` interface.
"""

from typing import Any
import warnings

from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from .diffgeom import Manifold, Patch, CoordSystem

__all__ = [
    'R2', 'R2_origin', 'relations_2d', 'R2_r', 'R2_p',
    'R3', 'R3_origin', 'relations_3d', 'R3_r', 'R3_c', 'R3_s'
]

###############################################################################
# R2
###############################################################################
R2: Any = Manifold('R^2', 2)

R2_origin: Any = Patch('origin', R2)

x, y = symbols('x y', real=True)
r, theta = symbols('rho theta', nonnegative=True)

relations_2d = {
    ('rectangular', 'polar'): [(x, y), (sqrt(x**2 + y**2), atan2(y, x))],
    ('polar', 'rectangular'): [(r, theta), (r*cos(theta), r*sin(theta))],
}

R2_r: Any = CoordSystem('rectangular', R2_origin, (x, y), relations_2d)
R2_p: Any = CoordSystem('polar', R2_origin, (r, theta), relations_2d)

# support deprecated feature
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x, y, r, theta = symbols('x y r theta', cls=Dummy)
    R2_r.connect_to(R2_p, [x, y],
                        [sqrt(x**2 + y**2), atan2(y, x)],
                    inverse=False, fill_in_gaps=False)
    R2_p.connect_to(R2_r, [r, theta],
                        [r*cos(theta), r*sin(theta)],
                    inverse=False, fill_in_gaps=False)

# Defining the basis coordinate functions and adding shortcuts for them to the
# manifold and the patch.
R2.x, R2.y = R2_origin.x, R2_origin.y = R2_r.x, R2_r.y = R2_r.coord_functions()
R2.r, R2.theta = R2_origin.r, R2_origin.theta = R2_p.r, R2_p.theta = R2_p.coord_functions()

# Defining the basis vector fields and adding shortcuts for them to the
# manifold and the patch.
R2.e_x, R2.e_y = R2_origin.e_x, R2_origin.e_y = R2_r.e_x, R2_r.e_y = R2_r.base_vectors()
R2.e_r, R2.e_theta = R2_origin.e_r, R2_origin.e_theta = R2_p.e_r, R2_p.e_theta = R2_p.base_vectors()

# Defining the basis oneform fields and adding shortcuts for them to the
# manifold and the patch.
R2.dx, R2.dy = R2_origin.dx, R2_origin.dy = R2_r.dx, R2_r.dy = R2_r.base_oneforms()
R2.dr, R2.dtheta = R2_origin.dr, R2_origin.dtheta = R2_p.dr, R2_p.dtheta = R2_p.base_oneforms()

###############################################################################
# R3
###############################################################################
R3: Any = Manifold('R^3', 3)

R3_origin: Any = Patch('origin', R3)

x, y, z = symbols('x y z', real=True)
rho, psi, r, theta, phi = symbols('rho psi r theta phi', nonnegative=True)

relations_3d = {
    ('rectangular', 'cylindrical'): [(x, y, z),
                                     (sqrt(x**2 + y**2), atan2(y, x), z)],
    ('cylindrical', 'rectangular'): [(rho, psi, z),
                                     (rho*cos(psi), rho*sin(psi), z)],
    ('rectangular', 'spherical'): [(x, y, z),
                                   (sqrt(x**2 + y**2 + z**2),
                                    acos(z/sqrt(x**2 + y**2 + z**2)),
                                    atan2(y, x))],
    ('spherical', 'rectangular'): [(r, theta, phi),
                                   (r*sin(theta)*cos(phi),
                                    r*sin(theta)*sin(phi),
                                    r*cos(theta))],
    ('cylindrical', 'spherical'): [(rho, psi, z),
                                   (sqrt(rho**2 + z**2),
                                    acos(z/sqrt(rho**2 + z**2)),
                                    psi)],
    ('spherical', 'cylindrical'): [(r, theta, phi),
                                   (r*sin(theta), phi, r*cos(theta))],
}

R3_r: Any = CoordSystem('rectangular', R3_origin, (x, y, z), relations_3d)
R3_c: Any = CoordSystem('cylindrical', R3_origin, (rho, psi, z), relations_3d)
R3_s: Any = CoordSystem('spherical', R3_origin, (r, theta, phi), relations_3d)

# support deprecated feature
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    x, y, z, rho, psi, r, theta, phi = symbols('x y z rho psi r theta phi', cls=Dummy)
    R3_r.connect_to(R3_c, [x, y, z],
                        [sqrt(x**2 + y**2), atan2(y, x), z],
                    inverse=False, fill_in_gaps=False)
    R3_c.connect_to(R3_r, [rho, psi, z],
                        [rho*cos(psi), rho*sin(psi), z],
                    inverse=False, fill_in_gaps=False)
    ## rectangular <-> spherical
    R3_r.connect_to(R3_s, [x, y, z],
                        [sqrt(x**2 + y**2 + z**2), acos(z/
                                sqrt(x**2 + y**2 + z**2)), atan2(y, x)],
                    inverse=False, fill_in_gaps=False)
    R3_s.connect_to(R3_r, [r, theta, phi],
                        [r*sin(theta)*cos(phi), r*sin(
                            theta)*sin(phi), r*cos(theta)],
                    inverse=False, fill_in_gaps=False)
    ## cylindrical <-> spherical
    R3_c.connect_to(R3_s, [rho, psi, z],
                        [sqrt(rho**2 + z**2), acos(z/sqrt(rho**2 + z**2)), psi],
                    inverse=False, fill_in_gaps=False)
    R3_s.connect_to(R3_c, [r, theta, phi],
                        [r*sin(theta), phi, r*cos(theta)],
                    inverse=False, fill_in_gaps=False)

# Defining the basis coordinate functions.
R3_r.x, R3_r.y, R3_r.z = R3_r.coord_functions()
R3_c.rho, R3_c.psi, R3_c.z = R3_c.coord_functions()
R3_s.r, R3_s.theta, R3_s.phi = R3_s.coord_functions()

# Defining the basis vector fields.
R3_r.e_x, R3_r.e_y, R3_r.e_z = R3_r.base_vectors()
R3_c.e_rho, R3_c.e_psi, R3_c.e_z = R3_c.base_vectors()
R3_s.e_r, R3_s.e_theta, R3_s.e_phi = R3_s.base_vectors()

# Defining the basis oneform fields.
R3_r.dx, R3_r.dy, R3_r.dz = R3_r.base_oneforms()
R3_c.drho, R3_c.dpsi, R3_c.dz = R3_c.base_oneforms()
R3_s.dr, R3_s.dtheta, R3_s.dphi = R3_s.base_oneforms()
<|endoftext|>"""Tools for solving inequalities and systems of inequalities. """
import itertools

from sympy.calculus.util import (continuous_domain, periodicity,
    function_range)
from sympy.core import sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Relational, Lt, Ge, Eq
from sympy.core.symbol import Symbol, Dummy
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection
from sympy.core.singleton import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.complexes import Abs
from sympy.logic import And
from sympy.polys import Poly, PolynomialError, parallel_poly_from_expr
from sympy.polys.polyutils import _nsort
from sympy.solvers.solveset import solvify, solveset
from sympy.utilities.iterables import sift, iterable
from sympy.utilities.misc import filldedent


def solve_poly_inequality(poly, rel):
    """Solve a polynomial inequality with rational coefficients.

    Examples
    ========

    >>> from sympy import solve_poly_inequality, Poly
    >>> from sympy.abc import x

    >>> solve_poly_inequality(Poly(x, x, domain='ZZ'), '==')
    [{0}]

    >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '!=')
    [Interval.open(-oo, -1), Interval.open(-1, 1), Interval.open(1, oo)]

    >>> solve_poly_inequality(Poly(x**2 - 1, x, domain='ZZ'), '==')
    [{-1}, {1}]

    See Also
    ========
    solve_poly_inequalities
    """
    if not isinstance(poly, Poly):
        raise ValueError(
            'For efficiency reasons, `poly` should be a Poly instance')
    if poly.as_expr().is_number:
        t = Relational(poly.as_expr(), 0, rel)
        if t is S.true:
            return [S.Reals]
        elif t is S.false:
            return [S.EmptySet]
        else:
            raise NotImplementedError(
                "could not determine truth value of %s" % t)

    reals, intervals = poly.real_roots(multiple=False), []

    if rel == '==':
        for root, _ in reals:
            interval = Interval(root, root)
            intervals.append(interval)
    elif rel == '!=':
        left = S.NegativeInfinity

        for right, _ in reals + [(S.Infinity, 1)]:
            interval = Interval(left, right, True, True)
            intervals.append(interval)
            left = right
    else:
        if poly.LC() > 0:
            sign = +1
        else:
            sign = -1

        eq_sign, equal = None, False

        if rel == '>':
            eq_sign = +1
        elif rel == '<':
            eq_sign = -1
        elif rel == '>=':
            eq_sign, equal = +1, True
        elif rel == '<=':
            eq_sign, equal = -1, True
        else:
            raise ValueError("'%s' is not a valid relation" % rel)

        right, right_open = S.Infinity, True

        for left, multiplicity in reversed(reals):
            if multiplicity % 2:
                if sign == eq_sign:
                    intervals.insert(
                        0, Interval(left, right, not equal, right_open))

                sign, right, right_open = -sign, left, not equal
            else:
                if sign == eq_sign and not equal:
                    intervals.insert(
                        0, Interval(left, right, True, right_open))
                    right, right_open = left, True
                elif sign != eq_sign and equal:
                    intervals.insert(0, Interval(left, left))

        if sign == eq_sign:
            intervals.insert(
                0, Interval(S.NegativeInfinity, right, True, right_open))

    return intervals


def solve_poly_inequalities(polys):
    """Solve polynomial inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy import Poly
