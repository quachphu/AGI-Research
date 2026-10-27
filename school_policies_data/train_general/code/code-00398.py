     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}

    3. If system is non-linear polynomial and zero-dimensional then it
    returns both solution (real and complex solutions, if present) using
    :func:`~.solve_poly_system`:

    >>> from sympy import sqrt
    >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}

    4. ``nonlinsolve`` can solve some linear (zero or positive dimensional)
    system (because it uses the :func:`sympy.polys.polytools.groebner` function to get the
    groebner basis and then uses the ``substitution`` function basis as the
    new `system`). But it is not recommended to solve linear system using
    ``nonlinsolve``, because :func:`~.linsolve` is better for general linear systems.

    >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9, y + z - 4], [x, y, z])
    {(3*z - 5, 4 - z, z)}

    5. System having polynomial equations and only real solution is
    solved using :func:`~.solve_poly_system`:

    >>> e1 = sqrt(x**2 + y**2) - 10
    >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
    >>> nonlinsolve((e1, e2), (x, y))
    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}

    6. It is better to use symbols instead of trigonometric functions or
    :class:`~.Function`. For example, replace $\sin(x)$ with a symbol, replace
    $f(x)$ with a symbol and so on. Get a solution from ``nonlinsolve`` and then
    use :func:`~.solveset` to get the value of $x$.

    How nonlinsolve is better than old solver ``_solve_system`` :
    =============================================================

    1. A positive dimensional system solver: nonlinsolve can return
    solution for positive dimensional system. It finds the
    Groebner Basis of the positive dimensional system(calling it as
    basis) then we can start solving equation(having least number of
    variable first in the basis) using solveset and substituting that
    solved solutions into other equation(of basis) to get solution in
    terms of minimum variables. Here the important thing is how we
    are substituting the known values and in which equations.

    2. Real and complex solutions: nonlinsolve returns both real
    and complex solution. If all the equations in the system are polynomial
    then using :func:`~.solve_poly_system` both real and complex solution is returned.
    If all the equations in the system are not polynomial equation then goes to
    ``substitution`` method with this polynomial and non polynomial equation(s),
    to solve for unsolved variables. Here to solve for particular variable
    solveset_real and solveset_complex is used. For both real and complex
    solution ``_solve_using_known_values`` is used inside ``substitution``
    (``substitution`` will be called when any non-polynomial equation is present).
    If a solution is valid its general solution is added to the final result.

    3. :class:`~.Complement` and :class:`~.Intersection` will be added:
    nonlinsolve maintains dict for complements and intersections. If solveset
    find complements or/and intersections with any interval or set during the
    execution of ``substitution`` function, then complement or/and
    intersection for that variable is added before returning final solution.

    """
    if not system:
        return S.EmptySet

    if not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise ValueError(filldedent(msg))

    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    if not is_sequence(symbols) or not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise IndexError(filldedent(msg))

    symbols = list(map(_sympify, symbols))
    system, symbols, swap = recast_to_symbols(system, symbols)
    if swap:
        soln = nonlinsolve(system, symbols)
        return FiniteSet(*[tuple(i.xreplace(swap) for i in s) for s in soln])

    if len(system) == 1 and len(symbols) == 1:
        return _solveset_work(system, symbols)

    # main code of def nonlinsolve() starts from here

    polys, polys_expr, nonpolys, denominators, unrad_changed = \
        _separate_poly_nonpoly(system, symbols)

    poly_eqs = []
    poly_sol = [{}]

    if polys:
        poly_sol, poly_eqs = _handle_poly(polys, symbols)
        if poly_sol and poly_sol[0]:
            poly_syms = set().union(*(eq.free_symbols for eq in polys))
            unrad_syms = set().union(*(eq.free_symbols for eq in unrad_changed))
            if unrad_syms == poly_syms and unrad_changed:
                # if all the symbols have been solved by _handle_poly
                # and unrad has been used then check solutions
                poly_sol = [sol for sol in poly_sol if checksol(unrad_changed, sol)]

    # Collect together the unsolved polynomials with the non-polynomial
    # equations.
    remaining = poly_eqs + nonpolys

    # to_tuple converts a solution dictionary to a tuple containing the
    # value for each symbol
    to_tuple = lambda sol: tuple(sol[s] for s in symbols)

    if not remaining:
        # If there is nothing left to solve then return the solution from
        # solve_poly_system directly.
        return FiniteSet(*map(to_tuple, poly_sol))
    else:
        # Here we handle:
        #
        #  1. The Groebner basis if solve_poly_system failed.
        #  2. The Groebner basis in the positive-dimensional case.
        #  3. Any non-polynomial equations
        #
        # If solve_poly_system did succeed then we pass those solutions in as
        # preliminary results.
        subs_res = substitution(remaining, symbols, result=poly_sol, exclude=denominators)

        if not isinstance(subs_res, FiniteSet):
            return subs_res

        # check solutions produced by substitution. Currently, checking is done for
        # only those solutions which have non-Set variable values.
        if unrad_changed:
            result = [dict(zip(symbols, sol)) for sol in subs_res.args]
            correct_sols = [sol for sol in result if any(isinstance(v, Set) for v in sol)
                            or checksol(unrad_changed, sol) != False]
            return FiniteSet(*map(to_tuple, correct_sols))
        else:
            return subs_res
<|endoftext|>"""
This module contains pdsolve() and different helper functions that it
uses. It is heavily inspired by the ode module and hence the basic
infrastructure remains the same.

**Functions in this module**

    These are the user functions in this module:

    - pdsolve()     - Solves PDE's
    - classify_pde() - Classifies PDEs into possible hints for dsolve().
    - pde_separate() - Separate variables in partial differential equation either by
                       additive or multiplicative separation approach.

    These are the helper functions in this module:

    - pde_separate_add() - Helper function for searching additive separable solutions.
    - pde_separate_mul() - Helper function for searching multiplicative
                           separable solutions.

**Currently implemented solver methods**

The following methods are implemented for solving partial differential
equations.  See the docstrings of the various pde_hint() functions for
more information on each (run help(pde)):

  - 1st order linear homogeneous partial differential equations
    with constant coefficients.
  - 1st order linear general partial differential equations
    with constant coefficients.
  - 1st order linear partial differential equations with
    variable coefficients.

"""
from functools import reduce

from itertools import combinations_with_replacement
from sympy.simplify import simplify  # type: ignore
from sympy.core import Add, S
from sympy.core.function import Function, expand, AppliedUndef, Subs
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, symbols
from sympy.functions import exp
from sympy.integrals.integrals import Integral, integrate
from sympy.utilities.iterables import has_dups, is_sequence
from sympy.utilities.misc import filldedent

from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.solvers import solve
from sympy.simplify.radsimp import collect

import operator


allhints = (
    "1st_linear_constant_coeff_homogeneous",
    "1st_linear_constant_coeff",
    "1st_linear_constant_coeff_Integral",
    "1st_linear_variable_coeff"
    )


def pdsolve(eq, func=None, hint='default', dict=False, solvefun=None, **kwargs):
    """
    Solves any (supported) kind of partial differential equation.

    **Usage**

        pdsolve(eq, f(x,y), hint) -> Solve partial differential equation
        eq for function f(x,y), using method hint.

    **Details**

        ``eq`` can be any supported partial differential equation (see
            the pde docstring for supported methods).  This can either
            be an Equality, or an expression, which is assumed to be
            equal to 0.

        ``f(x,y)`` is a function of two variables whose derivatives in that
            variable make up the partial differential equation. In many
            cases it is not necessary to provide this; it will be autodetected
            (and an error raised if it could not be detected).

        ``hint`` is the solving method that you want pdsolve to use.  Use
            classify_pde(eq, f(x,y)) to get all of the possible hints for
            a PDE.  The default hint, 'default', will use whatever hint
            is returned first by classify_pde().  See Hints below for
            more options that you can use for hint.

        ``solvefun`` is the convention used for arbitrary functions returned
            by the PDE solver. If not set by the user, it is set by default
            to be F.

    **Hints**

        Aside from the various solving methods, there are also some
        meta-hints that you can pass to pdsolve():

        "default":
                This uses whatever hint is returned first by
                classify_pde(). This is the default argument to
                pdsolve().

        "all":
                To make pdsolve apply all relevant classification hints,
                use pdsolve(PDE, func, hint="all").  This will return a
                dictionary of hint:solution terms.  If a hint causes
                pdsolve to raise the NotImplementedError, value of that
                hint's key will be the exception object raised.  The
                dictionary will also include some special keys:

                - order: The order of the PDE.  See also ode_order() in
                  deutils.py
                - default: The solution that would be returned by
                  default.  This is the one produced by the hint that
                  appears first in the tuple returned by classify_pde().

        "all_Integral":
                This is the same as "all", except if a hint also has a
                corresponding "_Integral" hint, it only returns the
                "_Integral" hint.  This is useful if "all" causes
                pdsolve() to hang because of a difficult or impossible
                integral.  This meta-hint will also be much faster than
                "all", because integrate() is an expensive routine.

        See also the classify_pde() docstring for more info on hints,
        and the pde docstring for a list of all supported hints.

    **Tips**
        - You can declare the derivative of an unknown function this way:

            >>> from sympy import Function, Derivative
            >>> from sympy.abc import x, y # x and y are the independent variables
            >>> f = Function("f")(x, y) # f is a function of x and y
            >>> # fx will be the partial derivative of f with respect to x
            >>> fx = Derivative(f, x)
            >>> # fy will be the partial derivative of f with respect to y
            >>> fy = Derivative(f, y)

        - See test_pde.py for many tests, which serves also as a set of
          examples for how to use pdsolve().
        - pdsolve always returns an Equality class (except for the case
          when the hint is "all" or "all_Integral"). Note that it is not possible
          to get an explicit solution for f(x, y) as in the case of ODE's
        - Do help(pde.pde_hintname) to get help more information on a
          specific hint


    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, Eq
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> u = f(x, y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> eq = Eq(1 + (2*(ux/u)) + (3*(uy/u)), 0)
    >>> pdsolve(eq)
    Eq(f(x, y), F(3*x - 2*y)*exp(-2*x/13 - 3*y/13))

    """

    if not solvefun:
        solvefun = Function('F')

    # See the docstring of _desolve for more details.
    hints = _desolve(eq, func=func, hint=hint, simplify=True,
                     type='pde', **kwargs)
    eq = hints.pop('eq', False)
    all_ = hints.pop('all', False)

    if all_:
        # TODO : 'best' hint should be implemented when adequate
        # number of hints are added.
        pdedict = {}
        failed_hints = {}
        gethints = classify_pde(eq, dict=True)
        pdedict.update({'order': gethints['order'],
                        'default': gethints['default']})
        for hint in hints:
            try:
                rv = _helper_simplify(eq, hint, hints[hint]['func'],
                    hints[hint]['order'], hints[hint][hint], solvefun)
            except NotImplementedError as detail:
                failed_hints[hint] = detail
            else:
                pdedict[hint] = rv
        pdedict.update(failed_hints)
        return pdedict

    else:
        return _helper_simplify(eq, hints['hint'], hints['func'],
                                hints['order'], hints[hints['hint']], solvefun)


def _helper_simplify(eq, hint, func, order, match, solvefun):
    """Helper function of pdsolve that calls the respective
    pde functions to solve for the partial differential
    equations. This minimizes the computation in
    calling _desolve multiple times.
    """
    solvefunc = globals()["pde_" + hint.removesuffix("_Integral")]
    return _handle_Integral(solvefunc(eq, func, order,
        match, solvefun), func, order, hint)


def _handle_Integral(expr, func, order, hint):
    r"""
    Converts a solution with integrals in it into an actual solution.

    Simplifies the integral mainly using doit()
    """
    if hint.endswith("_Integral"):
        return expr

    elif hint == "1st_linear_constant_coeff":
        return simplify(expr.doit())

    else:
        return expr


def classify_pde(eq, func=None, dict=False, *, prep=True, **kwargs):
    """
    Returns a tuple of possible pdsolve() classifications for a PDE.

    The tuple is ordered so that first item is the classification that
    pdsolve() uses to solve the PDE by default.  In general,
    classifications near the beginning of the list will produce
    better solutions faster than those near the end, though there are
    always exceptions.  To make pdsolve use a different classification,
    use pdsolve(PDE, func, hint=<classification>).  See also the pdsolve()
    docstring for different meta-hints you can use.

    If ``dict`` is true, classify_pde() will return a dictionary of
    hint:match expression terms. This is intended for internal use by
    pdsolve().  Note that because dictionaries are ordered arbitrarily,
    this will most likely not be in the same order as the tuple.

    You can get help on different hints by doing help(pde.pde_hintname),
    where hintname is the name of the hint without "_Integral".

    See sympy.pde.allhints or the sympy.pde docstring for a list of all
    supported hints that can be returned from classify_pde.


    Examples
    ========

    >>> from sympy.solvers.pde import classify_pde
    >>> from sympy import Function, Eq
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> u = f(x, y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> eq = Eq(1 + (2*(ux/u)) + (3*(uy/u)), 0)
    >>> classify_pde(eq)
    ('1st_linear_constant_coeff_homogeneous',)
    """

    if func and len(func.args) != 2:
        raise NotImplementedError("Right now only partial "
            "differential equations of two variables are supported")

    if prep or func is None:
        prep, func_ = _preprocess(eq, func)
        if func is None:
            func = func_

    if isinstance(eq, Equality):
        if eq.rhs != 0:
            return classify_pde(eq.lhs - eq.rhs, func)
        eq = eq.lhs

    f = func.func
    x = func.args[0]
    y = func.args[1]
    fx = f(x,y).diff(x)
    fy = f(x,y).diff(y)

    # TODO : For now pde.py uses support offered by the ode_order function
    # to find the order with respect to a multi-variable function. An
    # improvement could be to classify the order of the PDE on the basis of
    # individual variables.
    order = ode_order(eq, f(x,y))

    # hint:matchdict or hint:(tuple of matchdicts)
    # Also will contain "default":<default hint> and "order":order items.
    matching_hints = {'order': order}

    if not order:
        if dict:
            matching_hints["default"] = None
            return matching_hints
        return ()

    eq = expand(eq)

    a = Wild('a', exclude = [f(x,y)])
    b = Wild('b', exclude = [f(x,y), fx, fy, x, y])
    c = Wild('c', exclude = [f(x,y), fx, fy, x, y])
    d = Wild('d', exclude = [f(x,y), fx, fy, x, y])
    e = Wild('e', exclude = [f(x,y), fx, fy])
    n = Wild('n', exclude = [x, y])
    # Try removing the smallest power of f(x,y)
    # from the highest partial derivatives of f(x,y)
    reduced_eq = eq
    if eq.is_Add:
        power = None
        for i in set(combinations_with_replacement((x,y), order)):
            coeff = eq.coeff(f(x,y).diff(*i))
            if coeff == 1:
                continue
            match = coeff.match(a*f(x,y)**n)
            if match and match[a]:
                if power is None or match[n] < power:
                    power = match[n]
        if power:
            den = f(x,y)**power
            reduced_eq = Add(*[arg/den for arg in eq.args])

    if order == 1:
        reduced_eq = collect(reduced_eq, f(x, y))
        r = reduced_eq.match(b*fx + c*fy + d*f(x,y) + e)
        if r:
            if not r[e]:
                ## Linear first-order homogeneous partial-differential
                ## equation with constant coefficients
                r.update({'b': b, 'c': c, 'd': d})
                matching_hints["1st_linear_constant_coeff_homogeneous"] = r
            elif r[b]**2 + r[c]**2 != 0:
                ## Linear first-order general partial-differential
                ## equation with constant coefficients
                r.update({'b': b, 'c': c, 'd': d, 'e': e})
                matching_hints["1st_linear_constant_coeff"] = r
                matching_hints["1st_linear_constant_coeff_Integral"] = r

        else:
            b = Wild('b', exclude=[f(x, y), fx, fy])
            c = Wild('c', exclude=[f(x, y), fx, fy])
            d = Wild('d', exclude=[f(x, y), fx, fy])
            r = reduced_eq.match(b*fx + c*fy + d*f(x,y) + e)
            if r:
                r.update({'b': b, 'c': c, 'd': d, 'e': e})
                matching_hints["1st_linear_variable_coeff"] = r

    # Order keys based on allhints.
    rettuple = tuple(i for i in allhints if i in matching_hints)

    if dict:
        # Dictionaries are ordered arbitrarily, so make note of which
        # hint would come first for pdsolve().  Use an ordered dict in Py 3.
        matching_hints["default"] = None
        matching_hints["ordered_hints"] = rettuple
        for i in allhints:
            if i in matching_hints:
                matching_hints["default"] = i
                break
        return matching_hints
    return rettuple


def checkpdesol(pde, sol, func=None, solve_for_func=True):
    """
    Checks if the given solution satisfies the partial differential
    equation.

    pde is the partial differential equation which can be given in the
    form of an equation or an expression. sol is the solution for which
    the pde is to be checked. This can also be given in an equation or
    an expression form. If the function is not provided, the helper
    function _preprocess from deutils is used to identify the function.

    If a sequence of solutions is passed, the same sort of container will be
    used to return the result for each solution.

    The following methods are currently being implemented to check if the
    solution satisfies the PDE:

        1. Directly substitute the solution in the PDE and check. If the
           solution has not been solved for f, then it will solve for f
           provided solve_for_func has not been set to False.

    If the solution satisfies the PDE, then a tuple (True, 0) is returned.
    Otherwise a tuple (False, expr) where expr is the value obtained
    after substituting the solution in the PDE. However if a known solution
    returns False, it may be due to the inability of doit() to simplify it to zero.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy.solvers.pde import checkpdesol, pdsolve
    >>> x, y = symbols('x y')
    >>> f = Function('f')
    >>> eq = 2*f(x,y) + 3*f(x,y).diff(x) + 4*f(x,y).diff(y)
    >>> sol = pdsolve(eq)
    >>> assert checkpdesol(eq, sol)[0]
    >>> eq = x*f(x,y) + f(x,y).diff(x)
    >>> checkpdesol(eq, sol)
    (False, (x*F(4*x - 3*y) - 6*F(4*x - 3*y)/25 + 4*Subs(Derivative(F(_xi_1), _xi_1), _xi_1, 4*x - 3*y))*exp(-6*x/25 - 8*y/25))
    """

    # Converting the pde into an equation
    if not isinstance(pde, Equality):
        pde = Eq(pde, 0)

    # If no function is given, try finding the function present.
    if func is None:
        try:
            _, func = _preprocess(pde.lhs)
        except ValueError:
            funcs = [s.atoms(AppliedUndef) for s in (
                sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(funcs)
            if len(funcs) != 1:
                raise ValueError(
                    'must pass func arg to checkpdesol for this case.')
            func = funcs.pop()

    # If the given solution is in the form of a list or a set
    # then return a list or set of tuples.
    if is_sequence(sol, set):
        return type(sol)([checkpdesol(
            pde, i, func=func,
            solve_for_func=solve_for_func) for i in sol])

    # Convert solution into an equation
    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed

    # Try solving for the function
    solved = sol.lhs == func and not sol.rhs.has(func)
    if solve_for_func and not solved:
        solved = solve(sol, func)
        if solved:
            if len(solved) == 1:
                return checkpdesol(pde, Eq(func, solved[0]),
                    func=func, solve_for_func=False)
            else:
                return checkpdesol(pde, [Eq(func, t) for t in solved],
                    func=func, solve_for_func=False)

    # try direct substitution of the solution into the PDE and simplify
    if sol.lhs == func:
        pde = pde.lhs - pde.rhs
        s = simplify(pde.subs(func, sol.rhs).doit())
        return s is S.Zero, s

    raise NotImplementedError(filldedent('''
        Unable to test if %s is a solution to %s.''' % (sol, pde)))



def pde_1st_linear_constant_coeff_homogeneous(eq, func, order, match, solvefun):
    r"""
    Solves a first order linear homogeneous
    partial differential equation with constant coefficients.

    The general form of this partial differential equation is

    .. math:: a \frac{\partial f(x,y)}{\partial x}
              + b \frac{\partial f(x,y)}{\partial y} + c f(x,y) = 0

    where `a`, `b` and `c` are constants.

    The general solution is of the form:

    .. math::
        f(x, y) = F(- a y + b x ) e^{- \frac{c (a x + b y)}{a^2 + b^2}}

    and can be found in SymPy with ``pdsolve``::

        >>> from sympy.solvers import pdsolve
        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Function, pprint
        >>> f = Function('f')
        >>> u = f(x,y)
        >>> ux = u.diff(x)
        >>> uy = u.diff(y)
        >>> genform = a*ux + b*uy + c*u
        >>> pprint(genform)
          d               d
        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y)
          dx              dy

        >>> pprint(pdsolve(genform))
                                 -c*(a*x + b*y)
                                 ---------------
                                      2    2
                                     a  + b
        f(x, y) = F(-a*y + b*x)*e

    Examples
    ========

    >>> from sympy import pdsolve
    >>> from sympy import Function, pprint
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> pdsolve(f(x,y) + f(x,y).diff(x) + f(x,y).diff(y))
    Eq(f(x, y), F(x - y)*exp(-x/2 - y/2))
    >>> pprint(pdsolve(f(x,y) + f(x,y).diff(x) + f(x,y).diff(y)))
                          x   y
                        - - - -
                          2   2
    f(x, y) = F(x - y)*e

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """
    # TODO : For now homogeneous first order linear PDE's having
    # two variables are implemented. Once there is support for
    # solving systems of ODE's, this can be extended to n variables.

    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    return Eq(f(x,y), exp(-S(d)/(b**2 + c**2)*(b*x + c*y))*solvefun(c*x - b*y))


def pde_1st_linear_constant_coeff(eq, func, order, match, solvefun):
    r"""
    Solves a first order linear partial differential equation
    with constant coefficients.

    The general form of this partial differential equation is

    .. math:: a \frac{\partial f(x,y)}{\partial x}
              + b \frac{\partial f(x,y)}{\partial y}
              + c f(x,y) = G(x,y)

    where `a`, `b` and `c` are constants and `G(x, y)` can be an arbitrary
    function in `x` and `y`.

    The general solution of the PDE is:

    .. math::
        f(x, y) = \left. \left[F(\eta) + \frac{1}{a^2 + b^2}
        \int\limits^{a x + b y} G\left(\frac{a \xi + b \eta}{a^2 + b^2},
        \frac{- a \eta + b \xi}{a^2 + b^2} \right)
        e^{\frac{c \xi}{a^2 + b^2}}\, d\xi\right]
        e^{- \frac{c \xi}{a^2 + b^2}}
        \right|_{\substack{\eta=- a y + b x\\ \xi=a x + b y }}\, ,

    where `F(\eta)` is an arbitrary single-valued function. The solution
    can be found in SymPy with ``pdsolve``::

        >>> from sympy.solvers import pdsolve
        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Function, pprint
        >>> f = Function('f')
        >>> G = Function('G')
        >>> u = f(x, y)
        >>> ux = u.diff(x)
        >>> uy = u.diff(y)
        >>> genform = a*ux + b*uy + c*u - G(x,y)
        >>> pprint(genform)
          d               d
        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y) - G(x, y)
          dx              dy
        >>> pprint(pdsolve(genform, hint='1st_linear_constant_coeff_Integral'))
                  //          a*x + b*y                                             \         \|
                  ||              /                                                 |         ||
                  ||             |                                                  |         ||
                  ||             |                                      c*xi        |         ||
                  ||             |                                     -------      |         ||
                  ||             |                                      2    2      |         ||
                  ||             |      /a*xi + b*eta  -a*eta + b*xi\  a  + b       |         ||
                  ||             |     G|------------, -------------|*e        d(xi)|         ||
                  ||             |      |   2    2         2    2   |               |         ||
                  ||             |      \  a  + b         a  + b    /               |  -c*xi  ||
                  ||             |                                                  |  -------||
                  ||            /                                                   |   2    2||
                  ||                                                                |  a  + b ||
        f(x, y) = ||F(eta) + -------------------------------------------------------|*e       ||
                  ||                                  2    2                        |         ||
                  \\                                 a  + b                         /         /|eta=-a*y + b*x, xi=a*x + b*y

    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, pprint, exp
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> eq = -2*f(x,y).diff(x) + 4*f(x,y).diff(y) + 5*f(x,y) - exp(x + 3*y)
    >>> pdsolve(eq)
    Eq(f(x, y), (F(4*x + 2*y)*exp(x/2) + exp(x + 4*y)/15)*exp(-y))

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """

    # TODO : For now homogeneous first order linear PDE's having
    # two variables are implemented. Once there is support for
    # solving systems of ODE's, this can be extended to n variables.
    xi, eta = symbols("xi eta")
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]
    expterm = exp(-S(d)/(b**2 + c**2)*xi)
    functerm = solvefun(eta)
    solvedict = solve((b*x + c*y - xi, c*x - b*y - eta), x, y)
    # Integral should remain as it is in terms of xi,
    # doit() should be done in _handle_Integral.
    genterm = (1/S(b**2 + c**2))*Integral(
        (1/expterm*e).subs(solvedict), (xi, b*x + c*y))
    return Eq(f(x,y), Subs(expterm*(functerm + genterm),
        (eta, xi), (c*x - b*y, b*x + c*y)))


def pde_1st_linear_variable_coeff(eq, func, order, match, solvefun):
    r"""
    Solves a first order linear partial differential equation
    with variable coefficients. The general form of this partial
    differential equation is

    .. math:: a(x, y) \frac{\partial f(x, y)}{\partial x}
                + b(x, y) \frac{\partial f(x, y)}{\partial y}
                + c(x, y) f(x, y) = G(x, y)

    where `a(x, y)`, `b(x, y)`, `c(x, y)` and `G(x, y)` are arbitrary
    functions in `x` and `y`. This PDE is converted into an ODE by
    making the following transformation:

    1. `\xi` as `x`

    2. `\eta` as the constant in the solution to the differential
       equation `\frac{dy}{dx} = -\frac{b}{a}`

    Making the previous substitutions reduces it to the linear ODE

    .. math:: a(\xi, \eta)\frac{du}{d\xi} + c(\xi, \eta)u - G(\xi, \eta) = 0

    which can be solved using ``dsolve``.

    >>> from sympy.abc import x, y
    >>> from sympy import Function, pprint
    >>> a, b, c, G, f= [Function(i) for i in ['a', 'b', 'c', 'G', 'f']]
    >>> u = f(x,y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> genform = a(x, y)*u + b(x, y)*ux + c(x, y)*uy - G(x,y)
    >>> pprint(genform)
                                         d                     d
    -G(x, y) + a(x, y)*f(x, y) + b(x, y)*--(f(x, y)) + c(x, y)*--(f(x, y))
                                         dx                    dy


    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, pprint
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> eq =  x*(u.diff(x)) - y*(u.diff(y)) + y**2*u - y**2
    >>> pdsolve(eq)
    Eq(f(x, y), F(x*y)*exp(y**2/2) + 1)

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """
    from sympy.solvers.ode import dsolve

    eta = symbols("eta")
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]


    if not d:
         # To deal with cases like b*ux = e or c*uy = e
        if not (b and c):
            if c:
                try:
                    tsol = integrate(e/c, y)
                except NotImplementedError:
                    raise NotImplementedError("Unable to find a solution"
                        " due to inability of integrate")
                else:
                    return Eq(f(x,y), solvefun(x) + tsol)
            if b:
                try:
                    tsol = integrate(e/b, x)
                except NotImplementedError:
                    raise NotImplementedError("Unable to find a solution"
                        " due to inability of integrate")
                else:
                    return Eq(f(x,y), solvefun(y) + tsol)

    if not c:
        # To deal with cases when c is 0, a simpler method is used.
        # The PDE reduces to b*(u.diff(x)) + d*u = e, which is a linear ODE in x
        plode = f(x).diff(x)*b + d*f(x) - e
        sol = dsolve(plode, f(x))
        syms = sol.free_symbols - plode.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, y)
        return Eq(f(x, y), rhs)

    if not b:
        # To deal with cases when b is 0, a simpler method is used.
        # The PDE reduces to c*(u.diff(y)) + d*u = e, which is a linear ODE in y
        plode = f(y).diff(y)*c + d*f(y) - e
        sol = dsolve(plode, f(y))
        syms = sol.free_symbols - plode.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, x)
        return Eq(f(x, y), rhs)

    dummy = Function('d')
    h = (c/b).subs(y, dummy(x))
    sol = dsolve(dummy(x).diff(x) - h, dummy(x))
    if isinstance(sol, list):
        sol = sol[0]
    solsym = sol.free_symbols - h.free_symbols - {x, y}
    if len(solsym) == 1:
        solsym = solsym.pop()
        etat = (solve(sol, solsym)[0]).subs(dummy(x), y)
        ysub = solve(eta - etat, y)[0]
        deq = (b*(f(x).diff(x)) + d*f(x) - e).subs(y, ysub)
        final = (dsolve(deq, f(x), hint='1st_linear')).rhs
        if isinstance(final, list):
            final = final[0]
        finsyms = final.free_symbols - deq.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(final, finsyms, solvefun, etat)
        return Eq(f(x, y), rhs)

    else:
        raise NotImplementedError("Cannot solve the partial differential equation due"
            " to inability of constantsimp")


def _simplify_variable_coeff(sol, syms, func, funcarg):
    r"""
    Helper function to replace constants by functions in 1st_linear_variable_coeff
    """
    eta = Symbol("eta")
    if len(syms) == 1:
        sym = syms.pop()
        final = sol.subs(sym, func(funcarg))

    else:
        for sym in syms:
            final = sol.subs(sym, func(funcarg))

    return simplify(final.subs(eta, funcarg))


def pde_separate(eq, fun, sep, strategy='mul'):
    """Separate variables in partial differential equation either by additive
    or multiplicative separation approach. It tries to rewrite an equation so
    that one of the specified variables occurs on a different side of the
    equation than the others.

    :param eq: Partial differential equation

    :param fun: Original function F(x, y, z)

    :param sep: List of separated functions [X(x), u(y, z)]

    :param strategy: Separation strategy. You can choose between additive
        separation ('add') and multiplicative separation ('mul') which is
        default.

    Examples
    ========

    >>> from sympy import E, Eq, Function, pde_separate, Derivative as D
    >>> from sympy.abc import x, t
    >>> u, X, T = map(Function, 'uXT')

    >>> eq = Eq(D(u(x, t), x), E**(u(x, t))*D(u(x, t), t))
    >>> pde_separate(eq, u(x, t), [X(x), T(t)], strategy='add')
    [exp(-X(x))*Derivative(X(x), x), exp(T(t))*Derivative(T(t), t)]

    >>> eq = Eq(D(u(x, t), x, 2), D(u(x, t), t, 2))
    >>> pde_separate(eq, u(x, t), [X(x), T(t)], strategy='mul')
    [Derivative(X(x), (x, 2))/X(x), Derivative(T(t), (t, 2))/T(t)]

    See Also
    ========
    pde_separate_add, pde_separate_mul
    """

    do_add = False
    if strategy == 'add':
        do_add = True
    elif strategy == 'mul':
        do_add = False
    else:
        raise ValueError('Unknown strategy: %s' % strategy)

    if isinstance(eq, Equality):
        if eq.rhs != 0:
            return pde_separate(Eq(eq.lhs - eq.rhs, 0), fun, sep, strategy)
    else:
        return pde_separate(Eq(eq, 0), fun, sep, strategy)

    if eq.rhs != 0:
        raise ValueError("Value should be 0")

    # Handle arguments
    orig_args = list(fun.args)
    subs_args = [arg for s in sep for arg in s.args]

    if do_add:
        functions = reduce(operator.add, sep)
    else:
        functions = reduce(operator.mul, sep)

    # Check whether variables match
    if len(subs_args) != len(orig_args):
        raise ValueError("Variable counts do not match")
    # Check for duplicate arguments like  [X(x), u(x, y)]
    if has_dups(subs_args):
        raise ValueError("Duplicate substitution arguments detected")
    # Check whether the variables match
    if set(orig_args) != set(subs_args):
        raise ValueError("Arguments do not match")

    # Substitute original function with separated...
    result = eq.lhs.subs(fun, functions).doit()

    # Divide by terms when doing multiplicative separation
    if not do_add:
        eq = 0
        for i in result.args:
            eq += i/functions
        result = eq

    svar = subs_args[0]
    dvar = subs_args[1:]
    return _separate(result, svar, dvar)


def pde_separate_add(eq, fun, sep):
    """
    Helper function for searching additive separable solutions.

    Consider an equation of two independent variables x, y and a dependent
    variable w, we look for the product of two functions depending on different
    arguments:

    `w(x, y, z) = X(x) + y(y, z)`

    Examples
    ========

    >>> from sympy import E, Eq, Function, pde_separate_add, Derivative as D
    >>> from sympy.abc import x, t
    >>> u, X, T = map(Function, 'uXT')

    >>> eq = Eq(D(u(x, t), x), E**(u(x, t))*D(u(x, t), t))
    >>> pde_separate_add(eq, u(x, t), [X(x), T(t)])
    [exp(-X(x))*Derivative(X(x), x), exp(T(t))*Derivative(T(t), t)]

    """
    return pde_separate(eq, fun, sep, strategy='add')


def pde_separate_mul(eq, fun, sep):
    """
    Helper function for searching multiplicative separable solutions.

    Consider an equation of two independent variables x, y and a dependent
    variable w, we look for the product of two functions depending on different
    arguments:

    `w(x, y, z) = X(x)*u(y, z)`

    Examples
    ========

    >>> from sympy import Function, Eq, pde_separate_mul, Derivative as D
    >>> from sympy.abc import x, y
    >>> u, X, Y = map(Function, 'uXY')

    >>> eq = Eq(D(u(x, y), x, 2), D(u(x, y), y, 2))
    >>> pde_separate_mul(eq, u(x, y), [X(x), Y(y)])
    [Derivative(X(x), (x, 2))/X(x), Derivative(Y(y), (y, 2))/Y(y)]

    """
    return pde_separate(eq, fun, sep, strategy='mul')


def _separate(eq, dep, others):
    """Separate expression into two parts based on dependencies of variables."""

    # FIRST PASS
    # Extract derivatives depending our separable variable...
    terms = set()
    for term in eq.args:
        if term.is_Mul:
            for i in term.args:
                if i.is_Derivative and not i.has(*others):
                    terms.add(term)
                    continue
        elif term.is_Derivative and not term.has(*others):
            terms.add(term)
    # Find the factor that we need to divide by
    div = set()
    for term in terms:
        ext, sep = term.expand().as_independent(dep)
        # Failed?
        if sep.has(*others):
            return None
        div.add(ext)
    # FIXME: Find lcm() of all the divisors and divide with it, instead of
    # current hack :(
    # https://github.com/sympy/sympy/issues/4597
    if len(div) > 0:
        # double sum required or some tests will fail
        eq = Add(*[simplify(Add(*[term/i for i in div])) for term in eq.args])
    # SECOND PASS - separate the derivatives
    div = set()
    lhs = rhs = 0
    for term in eq.args:
        # Check, whether we have already term with independent variable...
        if not term.has(*others):
            lhs += term
            continue
        # ...otherwise, try to separate
        temp, sep = term.expand().as_independent(dep)
        # Failed?
        if sep.has(*others):
            return None
        # Extract the divisors
        div.add(sep)
        rhs -= term.expand()
    # Do the division
    fulldiv = reduce(operator.add, div)
    lhs = simplify(lhs/fulldiv).expand()
    rhs = simplify(rhs/fulldiv).expand()
    # ...and check whether we were successful :)
    if lhs.has(*others) or rhs.has(dep):
        return None
    return [lhs, rhs]
<|endoftext|>"""Tools for optimizing a linear function for a given simplex.

For the linear objective function ``f`` with linear constraints
expressed using `Le`, `Ge` or `Eq` can be found with ``lpmin`` or
``lpmax``. The symbols are **unbounded** unless specifically
constrained.

As an alternative, the matrices describing the objective and the
constraints, and an optional list of bounds can be passed to
``linprog`` which will solve for the minimization of ``C*x``
under constraints ``A*x <= b`` and/or ``Aeq*x = beq``, and
individual bounds for variables given as ``(lo, hi)``. The values
returned are **nonnegative** unless bounds are provided that
indicate otherwise.

Errors that might be raised are UnboundedLPError when there is no
finite solution for the system or InfeasibleLPError when the
constraints represent impossible conditions (i.e. a non-existent
 simplex).

Here is a simple 1-D system: minimize `x` given that ``x >= 1``.

    >>> from sympy.solvers.simplex import lpmin, linprog
    >>> from sympy.abc import x

    The function and a list with the constraint is passed directly
    to `lpmin`:

    >>> lpmin(x, [x >= 1])
    (1, {x: 1})

    For `linprog` the matrix for the objective is `[1]` and the
    uivariate constraint can be passed as a bound with None acting
    as infinity:

    >>> linprog([1], bounds=(1, None))
    (1, [1])

    Or the matrices, corresponding to ``x >= 1`` expressed as
    ``-x <= -1`` as required by the routine, can be passed:

    >>> linprog([1], [-1], [-1])
    (1, [1])

    If there is no limit for the objective, an error is raised.
    In this case there is a valid region of interest (simplex)
    but no limit to how small ``x`` can be:

    >>> lpmin(x, [])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    An error is raised if there is no possible solution:

    >>> lpmin(x,[x<=1,x>=2])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError:
    Inconsistent/False constraint
"""

from sympy.core import sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Le, Ge, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sorting import ordered
from sympy.functions.elementary.complexes import sign
from sympy.matrices.dense import Matrix, zeros
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.utilities.iterables import numbered_symbols
from sympy.utilities.misc import filldedent


class UnboundedLPError(Exception):
    """
    A linear programming problem is said to be unbounded if its objective
    function can assume arbitrarily large values.

    Example
    =======

    Suppose you want to maximize
        2x
    subject to
        x >= 0

    There's no upper limit that 2x can take.
    """

    pass


class InfeasibleLPError(Exception):
    """
    A linear programming problem is considered infeasible if its
    constraint set is empty. That is, if the set of all vectors
    satisfying the constraints is empty, then the problem is infeasible.

    Example
    =======

    Suppose you want to maximize
        x
    subject to
        x >= 10
        x <= 9

    No x can satisfy those constraints.
    """

    pass


def _pivot(M, i, j):
    """
    The pivot element `M[i, j]` is inverted and the rest of the matrix
    modified and returned as a new matrix; original is left unmodified.

    Example
    =======

    >>> from sympy.matrices.dense import Matrix
    >>> from sympy.solvers.simplex import _pivot
    >>> from sympy import var
    >>> Matrix(3, 3, var('a:i'))
    Matrix([
    [a, b, c],
    [d, e, f],
    [g, h, i]])
    >>> _pivot(_, 1, 0)
    Matrix([
    [-a/d, -a*e/d + b, -a*f/d + c],
    [ 1/d,        e/d,        f/d],
    [-g/d,  h - e*g/d,  i - f*g/d]])
    """
    Mi, Mj, Mij = M[i, :], M[:, j], M[i, j]
    if Mij == 0:
        raise ZeroDivisionError(
            "Tried to pivot about zero-valued entry.")
    A = M - Mj * (Mi / Mij)
    A[i, :] = Mi / Mij
    A[:, j] = -Mj / Mij
    A[i, j] = 1 / Mij
    return A


def _choose_pivot_row(A, B, candidate_rows, pivot_col, Y):
    # Choose row with smallest ratio
    # If there are ties, pick using Bland's rule
    return min(candidate_rows, key=lambda i: (B[i] / A[i, pivot_col], Y[i]))


def _simplex(A, B, C, D=None, dual=False):
    """Return ``(o, x, y)`` obtained from the two-phase simplex method
    using Bland's rule: ``o`` is the minimum value of primal,
    ``Cx - D``, under constraints ``Ax <= B`` (with ``x >= 0``) and
    the maximum of the dual, ``y^{T}B - D``, under constraints
    ``A^{T}*y >= C^{T}`` (with ``y >= 0``). To compute the dual of
    the system, pass `dual=True` and ``(o, y, x)`` will be returned.

    Note: the nonnegative constraints for ``x`` and ``y`` supercede
    any values of ``A`` and ``B`` that are inconsistent with that
    assumption, so if a constraint of ``x >= -1`` is represented
    in ``A`` and ``B``, no value will be obtained that is negative; if
    a constraint of ``x <= -1`` is represented, an error will be
    raised since no solution is possible.

    This routine relies on the ability of determining whether an
    expression is 0 or not. This is guaranteed if the input contains
    only Float or Rational entries. It will raise a TypeError if
    a relationship does not evaluate to True or False.

    Examples
    ========

    >>> from sympy.solvers.simplex import _simplex
    >>> from sympy import Matrix

    Consider the simple minimization of ``f = x + y + 1`` under the
    constraint that ``y + 2*x >= 4``. This is the "standard form" of
    a minimization.

    In the nonnegative quadrant, this inequality describes a area above
    a triangle with vertices at (0, 4), (0, 0) and (2, 0). The minimum
    of ``f`` occurs at (2, 0). Define A, B, C, D for the standard
    minimization:

    >>> A = Matrix([[2, 1]])
    >>> B = Matrix([4])
    >>> C = Matrix([[1, 1]])
    >>> D = Matrix([-1])

    Confirm that this is the system of interest:

    >>> from sympy.abc import x, y
    >>> X = Matrix([x, y])
    >>> (C*X - D)[0]
    x + y + 1
    >>> [i >= j for i, j in zip(A*X, B)]
    [2*x + y >= 4]

    Since `_simplex` will do a minimization for constraints given as
    ``A*x <= B``, the signs of ``A`` and ``B`` must be negated since
    the currently correspond to a greater-than inequality:

    >>> _simplex(-A, -B, C, D)
    (3, [2, 0], [1/2])

    The dual of minimizing ``f`` is maximizing ``F = c*y - d`` for
    ``a*y <= b`` where ``a``, ``b``, ``c``, ``d`` are derived from the
    transpose of the matrix representation of the standard minimization:

    >>> tr = lambda a, b, c, d: [i.T for i in (a, c, b, d)]
    >>> a, b, c, d = tr(A, B, C, D)

    This time ``a*x <= b`` is the expected inequality for the `_simplex`
    method, but to maximize ``F``, the sign of ``c`` and ``d`` must be
    changed (so that minimizing the negative will give the negative of
    the maximum of ``F``):

    >>> _simplex(a, b, -c, -d)
    (-3, [1/2], [2, 0])

    The negative of ``F`` and the min of ``f`` are the same. The dual
    point `[1/2]` is the value of ``y`` that minimized ``F = c*y - d``
    under constraints a*x <= b``:

    >>> y = Matrix(['y'])
    >>> (c*y - d)[0]
    4*y + 1
    >>> [i <= j for i, j in zip(a*y,b)]
    [2*y <= 1, y <= 1]

    In this 1-dimensional dual system, the more restrictive constraint is
    the first which limits ``y`` between 0 and 1/2 and the maximum of
    ``F`` is attained at the nonzero value, hence is ``4*(1/2) + 1 = 3``.

    In this case the values for ``x`` and ``y`` were the same when the
    dual representation was solved. This is not always the case (though
    the value of the function will be the same).

    >>> l = [[1, 1], [-1, 1], [0, 1], [-1, 0]], [5, 1, 2, -1], [[1, 1]], [-1]
    >>> A, B, C, D = [Matrix(i) for i in l]
    >>> _simplex(A, B, -C, -D)
    (-6, [3, 2], [1, 0, 0, 0])
    >>> _simplex(A, B, -C, -D, dual=True)  # [5, 0] != [3, 2]
    (-6, [1, 0, 0, 0], [5, 0])

    In both cases the function has the same value:

    >>> Matrix(C)*Matrix([3, 2]) == Matrix(C)*Matrix([5, 0])
    True

    See Also
    ========
    _lp - poses min/max problem in form compatible with _simplex
    lpmin - minimization which calls _lp
    lpmax - maximimzation which calls _lp

    References
    ==========

    .. [1] Thomas S. Ferguson, LINEAR PROGRAMMING: A Concise Introduction
           web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_lp.pdf

    """
    A, B, C, D = [Matrix(i) for i in (A, B, C, D or [0])]
    if dual:
        _o, d, p = _simplex(-A.T, C.T, B.T, -D)
        return -_o, d, p

    if A and B:
        M = Matrix([[A, B], [C, D]])
    else:
        if A or B:
            raise ValueError("must give A and B")
        # no constraints given
        M = Matrix([[C, D]])
    n = M.cols - 1
    m = M.rows - 1

    if not all(i.is_Float or i.is_Rational for i in M):
        # with literal Float and Rational we are guaranteed the
        # ability of determining whether an expression is 0 or not
        raise TypeError(filldedent("""
            Only rationals and floats are allowed.
            """
            )
        )

    # x variables have priority over y variables during Bland's rule
    # since False < True
    X = [(False, j) for j in range(n)]
    Y = [(True, i) for i in range(m)]

    # Phase 1: find a feasible solution or determine none exist

    ## keep track of last pivot row and column
    last = None

    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        if all(B[i] >= 0 for i in range(B.rows)):
            # We have found a feasible solution
            break

        # Find k: first row with a negative rightmost entry
        for k in range(B.rows):
            if B[k] < 0:
                break  # use current value of k below
        else:
            pass  # error will raise below

        # Choose pivot column, c
        piv_cols = [_ for _ in range(A.cols) if A[k, _] < 0]
        if not piv_cols:
            raise InfeasibleLPError(filldedent("""
                The constraint set is empty!"""))
        _, c = min((X[i], i) for i in piv_cols) # Bland's rule

        # Choose pivot row, r
        piv_rows = [_ for _ in range(A.rows) if A[_, c] > 0 and B[_] > 0]
        piv_rows.append(k)
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        # check for oscillation
        if (r, c) == last:
            # Not sure what to do here; it looks like there will be
            # oscillations; see o1 test added at this commit to
            # see a system with no solution and the o2 for one
            # with a solution. In the case of o2, the solution
            # from linprog is the same as the one from lpmin, but
            # the matrices created in the lpmin case are different
            # than those created without replacements in linprog and
            # the matrices in the linprog case lead to oscillations.
            # If the matrices could be re-written in linprog like
            # lpmin does, this behavior could be avoided and then
            # perhaps the oscillating case would only occur when
            # there is no solution. For now, the output is checked
            # before exit if oscillations were detected and an
            # error is raised there if the solution was invalid.
            #
            # cf section 6 of Ferguson for a non-cycling modification
            last = True
            break
        last = r, c

        M = _pivot(M, r, c)
        X[c], Y[r] = Y[r], X[c]

    # Phase 2: from a feasible solution, pivot to optimal
    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        C = M[-1, :-1]

        # Choose a pivot column, c
        piv_cols = [_ for _ in range(n) if C[_] < 0]
        if not piv_cols:
            break
        _, c = min((X[i], i) for i in piv_cols)  # Bland's rule

        # Choose a pivot row, r
        piv_rows = [_ for _ in range(m) if A[_, c] > 0]
        if not piv_rows:
            raise UnboundedLPError(filldedent("""
                Objective function can assume
                arbitrarily large values!"""))
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        M = _pivot(M, r, c)
        X[c], Y[r] = Y[r], X[c]

    argmax = [None] * n
    argmin_dual = [None] * m

    for i, (v, n) in enumerate(X):
        if v == False:
            argmax[n] = 0
        else:
            argmin_dual[n] = M[-1, i]

    for i, (v, n) in enumerate(Y):
        if v == True:
            argmin_dual[n] = 0
        else:
            argmax[n] = M[i, -1]

    if last and not all(i >= 0 for i in argmax + argmin_dual):
        raise InfeasibleLPError(filldedent("""
            Oscillating system led to invalid solution.
            If you believe there was a valid solution, please
            report this as a bug."""))
    return -M[-1, -1], argmax, argmin_dual


## routines that use _simplex or support those that do


def _abcd(M, list=False):
    """return parts of M as matrices or lists

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd

    >>> m = Matrix(3, 3, range(9)); m
    Matrix([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]])
    >>> a, b, c, d = _abcd(m)
    >>> a
    Matrix([
    [0, 1],
    [3, 4]])
    >>> b
    Matrix([
    [2],
    [5]])
    >>> c
    Matrix([[6, 7]])
    >>> d
    Matrix([[8]])

    The matrices can be returned as compact lists, too:

    >>> L = a, b, c, d = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    """

    def aslist(i):
        l = i.tolist()
        if len(l[0]) == 1:  # col vector
            return [i[0] for i in l]
        return l

    m = M[:-1, :-1], M[:-1, -1], M[-1, :-1], M[-1:, -1:]
    if not list:
        return m
    return tuple([aslist(i) for i in m])


def _m(a, b, c, d=None):
    """return Matrix([[a, b], [c, d]]) from matrices
    in Matrix or list form.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd, _m
    >>> m = Matrix(3, 3, range(9))
    >>> L = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    >>> _abcd(m)
    (Matrix([
    [0, 1],
    [3, 4]]), Matrix([
    [2],
    [5]]), Matrix([[6, 7]]), Matrix([[8]]))
    >>> assert m == _m(*L) == _m(*_)
    """
    a, b, c, d = [Matrix(i) for i in (a, b, c, d or [0])]
    return Matrix([[a, b], [c, d]])


def _primal_dual(M, factor=True):
    """return primal and dual function and constraints
    assuming that ``M = Matrix([[A, b], [c, d]])`` and the
    function ``c*x - d`` is being minimized with ``Ax >= b``
    for nonnegative values of ``x``. The dual and its
    constraints will be for maximizing `b.T*y - d` subject
    to ``A.T*y <= c.T``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _primal_dual, lpmin, lpmax
    >>> from sympy import Matrix

    The following matrix represents the primal task of
    minimizing x + y + 7 for y >= x + 1 and y >= -2*x + 3.
    The dual task seeks to maximize x + 3*y + 7 with
    2*y - x <= 1 and and x + y <= 1:

    >>> M = Matrix([
    ...     [-1, 1,  1],
    ...     [ 2, 1,  3],
    ...     [ 1, 1, -7]])
    >>> p, d = _primal_dual(M)

    The minimum of the primal and maximum of the dual are the same
    (though they occur at different points):

    >>> lpmin(*p)
    (28/3, {x1: 2/3, x2: 5/3})
    >>> lpmax(*d)
    (28/3, {y1: 1/3, y2: 2/3})

    If the equivalent (but canonical) inequalities are
    desired, leave `factor=True`, otherwise the unmodified
    inequalities for M will be returned.

    >>> m = Matrix([
    ... [-3, -2,  4, -2],
    ... [ 2,  0,  0, -2],
    ... [ 0,  1, -3,  0]])

    >>> _primal_dual(m, False)  # last condition is 2*x1 >= -2
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, 2*x1 >= -2]),
    (-2*y1 - 2*y2,
        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))

    >>> _primal_dual(m)  # condition now x1 >= -1
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, x1 >= -1]),
    (-2*y1 - 2*y2,
        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))

    If you pass the transpose of the matrix, the primal will be
    identified as the standard minimization problem and the
    dual as the standard maximization:

    >>> _primal_dual(m.T)
    ((-2*x1 - 2*x2,
        [-3*x1 + 2*x2 >= 0, -2*x1 >= 1, 4*x1 >= -3]),
    (y2 - 3*y3,
        [-3*y1 - 2*y2 + 4*y3 <= -2, y1 <= -1]))

    A matrix must have some size or else None will be returned for
    the functions:

    >>> _primal_dual(Matrix([[1, 2]]))
    ((x1 - 2, []), (-2, []))

    >>> _primal_dual(Matrix([]))
    ((None, []), (None, []))

    References
    ==========

    .. [1] David Galvin, Relations between Primal and Dual
           www3.nd.edu/~dgalvin1/30210/30210_F07/presentations/dual_opt.pdf
    """
    if not M:
        return (None, []), (None, [])
    if not hasattr(M, "shape"):
        if len(M) not in (3, 4):
            raise ValueError("expecting Matrix or 3 or 4 lists")
        M = _m(*M)
    m, n = [i - 1 for i in M.shape]
    A, b, c, d = _abcd(M)
    d = d[0]
    _ = lambda x: numbered_symbols(x, start=1)
    x = Matrix([i for i, j in zip(_("x"), range(n))])
    yT = Matrix([i for i, j in zip(_("y"), range(m))]).T

    def ineq(L, r, op):
        rv = []
        for r in (op(i, j) for i, j in zip(L, r)):
            if r == True:
                continue
            elif r == False:
                return [False]
            if factor:
                f = factor_terms(r)
                if f.lhs.is_Mul and f.rhs % f.lhs.args[0] == 0:
                    assert len(f.lhs.args) == 2, f.lhs
                    k = f.lhs.args[0]
                    r = r.func(sign(k) * f.lhs.args[1], f.rhs // abs(k))
            rv.append(r)
        return rv

    eq = lambda x, d: x[0] - d if x else -d
    F = eq(c * x, d)
    f = eq(yT * b, d)
    return (F, ineq(A * x, b, Ge)), (f, ineq(yT * A, c, Le))


def _rel_as_nonpos(constr, syms):
    """return `(np, d, aux)` where `np` is a list of nonpositive
    expressions that represent the given constraints (possibly
    rewritten in terms of auxilliary variables) expressible with
    nonnegative symbols, and `d` is a dictionary mapping a given
    symbols to an expression with an auxilliary variable. In some
    cases a symbol will be used as part of the change of variables,
    e.g. x: x - z1 instead of x: z1 - z2.

    If any constraint is False/empty, return None. All variables in
    ``constr`` are assumed to be unbounded unless explicitly indicated
    otherwise with a univariate constraint, e.g. ``x >= 0`` will
    restrict ``x`` to nonnegative values.

    The ``syms`` must be included so all symbols can be given an
    unbounded assumption if they are not otherwise bound with
    univariate conditions like ``x <= 3``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _rel_as_nonpos
    >>> from sympy.abc import x, y
    >>> _rel_as_nonpos([x >= y, x >= 0, y >= 0], (x, y))
    ([-x + y], {}, [])
    >>> _rel_as_nonpos([x >= 3, x <= 5], [x])
    ([_z1 - 2], {x: _z1 + 3}, [_z1])
    >>> _rel_as_nonpos([x <= 5], [x])
    ([], {x: 5 - _z1}, [_z1])
    >>> _rel_as_nonpos([x >= 1], [x])
    ([], {x: _z1 + 1}, [_z1])
    """
    r = {}  # replacements to handle change of variables
    np = []  # nonpositive expressions
    aux = []  # auxilliary symbols added
    ui = numbered_symbols("z", start=1, cls=Dummy)  # auxilliary symbols
    univariate = {}  # {x: interval} for univariate constraints
    unbound = []  # symbols designated as unbound
    syms = set(syms)  # the expected syms of the system

    # separate out univariates
    for i in constr:
        if i == True:
            continue  # ignore
        if i == False:
            return  # no solution
        if i.has(S.Infinity, S.NegativeInfinity):
            raise ValueError("only finite bounds are permitted")
        if isinstance(i, (Le, Ge)):
            i = i.lts - i.gts
            freei = i.free_symbols
            if freei - syms:
                raise ValueError(
                    "unexpected symbol(s) in constraint: %s" % (freei - syms)
                )
            if len(freei) > 1:
                np.append(i)
            elif freei:
                x = freei.pop()
                if x in unbound:
                    continue  # will handle later
                ivl = Le(i, 0, evaluate=False).as_set()
                if x not in univariate:
                    univariate[x] = ivl
                else:
                    univariate[x] &= ivl
            elif i:
                return False
        else:
            raise TypeError(filldedent("""
                only equalities like Eq(x, y) or non-strict
                inequalities like x >= y are allowed in lp, not %s""" % i))

    # introduce auxilliary variables as needed for univariate
    # inequalities
    for x in syms:
        i = univariate.get(x, True)
        if not i:
            return None  # no solution possible
        if i == True:
            unbound.append(x)
            continue
        a, b = i.inf, i.sup
        if a.is_infinite:
            u = next(ui)
            r[x] = b - u
            aux.append(u)
        elif b.is_infinite:
            if a:
                u = next(ui)
                r[x] = a + u
                aux.append(u)
            else:
                # standard nonnegative relationship
                pass
        else:
            u = next(ui)
            aux.append(u)
            # shift so u = x - a => x = u + a
            r[x] = u + a
            # add constraint for u <= b - a
            # since when u = b-a then x = u + a = b - a + a = b:
            # the upper limit for x
            np.append(u - (b - a))

    # make change of variables for unbound variables
    for x in unbound:
        u = next(ui)
        r[x] = u - x  # reusing x
        aux.append(u)

    return np, r, aux


def _lp_matrices(objective, constraints):
    """return A, B, C, D, r, x+X, X for maximizing
    objective = Cx - D with constraints Ax <= B, introducing
    introducing auxilliary variables, X, as necessary to make
    replacements of symbols as given in r, {xi: expression with Xj},
    so all variables in x+X will take on nonnegative values.

    Every univariate condition creates a semi-infinite
    condition, e.g. a single ``x <= 3`` creates the
    interval ``[-oo, 3]`` while ``x <= 3`` and ``x >= 2``
    create an interval ``[2, 3]``. Variables not in a univariate
    expression will take on nonnegative values.
    """

    # sympify input and collect free symbols
    F = sympify(objective)
    np = [sympify(i) for i in constraints]
    syms = set.union(*[i.free_symbols for i in [F] + np], set())

    # change Eq(x, y) to x - y <= 0 and y - x <= 0
    for i in range(len(np)):
        if isinstance(np[i], Eq):
            np[i] = np[i].lhs - np[i].rhs <= 0
            np.append(-np[i].lhs <= 0)

    # convert constraints to nonpositive expressions
    _ = _rel_as_nonpos(np, syms)
    if _ is None:
        raise InfeasibleLPError(filldedent("""
            Inconsistent/False constraint"""))
    np, r, aux = _

    # do change of variables
    F = F.xreplace(r)
    np = [i.xreplace(r) for i in np]

    # convert to matrices
    xx = list(ordered(syms)) + aux
    A, B = linear_eq_to_matrix(np, xx)
    C, D = linear_eq_to_matrix([F], xx)
    return A, B, C, D, r, xx, aux


def _lp(min_max, f, constr):
    """Return the optimization (min or max) of ``f`` with the given
    constraints. All variables are unbounded unless constrained.

    If `min_max` is 'max' then the results corresponding to the
    maximization of ``f`` will be returned, else the minimization.
    The constraints can be given as Le, Ge or Eq expressions.

    Examples
    ========

    >>> from sympy.solvers.simplex import _lp as lp
    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z
    >>> f = x + y - 2*z
    >>> c = [7*x + 4*y - 7*z <= 3, 3*x - y + 10*z <= 6]
    >>> c += [i >= 0 for i in (x, y, z)]
    >>> lp(min, f, c)
    (-6/5, {x: 0, y: 0, z: 3/5})

    By passing max, the maximum value for f under the constraints
    is returned (if possible):

    >>> lp(max, f, c)
    (3/4, {x: 0, y: 3/4, z: 0})

    Constraints that are equalities will require that the solution
    also satisfy them:

    >>> lp(max, f, c + [Eq(y - 9*x, 1)])
    (5/7, {x: 0, y: 1, z: 1/7})

    All symbols are reported, even if they are not in the objective
    function:

    >>> lp(min, x, [y + x >= 3, x >= 0])
    (0, {x: 0, y: 3})
    """
    # get the matrix components for the system expressed
    # in terms of only nonnegative variables
    A, B, C, D, r, xx, aux = _lp_matrices(f, constr)

    how = str(min_max).lower()
    if "max" in how:
        # _simplex minimizes for Ax <= B so we
        # have to change the sign of the function
        # and negate the optimal value returned
        _o, p, d = _simplex(A, B, -C, -D)
        o = -_o
    elif "min" in how:
        o, p, d = _simplex(A, B, C, D)
    else:
        raise ValueError("expecting min or max")

    # restore original variables and remove aux from p
    p = dict(zip(xx, p))
    if r:  # p has original symbols and auxilliary symbols
        # if r has x: x - z1 use values from p to update
        r = {k: v.xreplace(p) for k, v in r.items()}
        # then use the actual value of x (= x - z1) in p
        p.update(r)
        # don't show aux
        p = {k: p[k] for k in ordered(p) if k not in aux}

    # not returning dual since there may be extra constraints
    # when a variable has finite bounds
    return o, p


def lpmin(f, constr):
    """return minimum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmin
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmin(x, [2*x - 3*y >= -1, Eq(x + 3*y, 2), x <= 2*y])
    (1/3, {x: 1/3, y: 5/9})

    Negative values for variables are permitted unless explicitly
    excluding, so minimizing ``x`` for ``x <= 3`` is an
    unbounded problem while the following has a bounded solution:

    >>> lpmin(x, [x >= 0, x <= 3])
    (0, {x: 0})

    Without indicating that ``x`` is nonnegative, there
    is no minimum for this objective:

    >>> lpmin(x, [x <= 3])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    See Also
    ========
    linprog, lpmax
    """
    return _lp(min, f, constr)


def lpmax(f, constr):
    """return maximum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmax
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmax(x, [2*x - 3*y >= -1, Eq(x+ 3*y,2), x <= 2*y])
    (4/5, {x: 4/5, y: 2/5})

    Negative values for variables are permitted unless explicitly
    excluding:

    >>> lpmax(x, [x <= -1])
    (-1, {x: -1})

    If a non-negative constraint is added for x, there is no
    possible solution:

    >>> lpmax(x, [x <= -1, x >= 0])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError: inconsistent/False constraint

    See Also
    ========
    linprog, lpmin
    """
    return _lp(max, f, constr)


def _handle_bounds(bounds):
    # introduce auxiliary variables as needed for univariate
    # inequalities

    def _make_list(length: int, index_value_pairs):
        li = [0] * length
        for idx, val in index_value_pairs:
            li[idx] = val
        return li

    unbound = []
    row = []
    row2 = []
    b_len = len(bounds)
    for x, (a, b) in enumerate(bounds):
        if a is None and b is None:
            unbound.append(x)
        elif a is None:
            # r[x] = b - u
            b_len += 1
            row.append(_make_list(b_len, [(x, 1), (-1, 1)]))
            row.append(_make_list(b_len, [(x, -1), (-1, -1)]))
            row2.extend([[b], [-b]])
        elif b is None:
            if a:
                # r[x] = a + u
                b_len += 1
                row.append(_make_list(b_len, [(x, 1), (-1, -1)]))
                row.append(_make_list(b_len, [(x, -1), (-1, 1)]))
                row2.extend([[a], [-a]])
            else:
                # standard nonnegative relationship
                pass
        else:
            # r[x] = u + a
            b_len += 1
            row.append(_make_list(b_len, [(x, 1), (-1, -1)]))
            row.append(_make_list(b_len, [(x, -1), (-1, 1)]))
            # u <= b - a
            row.append(_make_list(b_len, [(-1, 1)]))
            row2.extend([[a], [-a], [b - a]])

    # make change of variables for unbound variables
    for x in unbound:
        # r[x] = u - v
        b_len += 2
        row.append(_make_list(b_len, [(x, 1), (-1, 1), (-2, -1)]))
        row.append(_make_list(b_len, [(x, -1), (-1, -1), (-2, 1)]))
        row2.extend([[0], [0]])

    return Matrix([r + [0]*(b_len - len(r)) for r in row]), Matrix(row2)


def linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    """Return the minimization of ``c*x`` with the given
    constraints ``A*x <= b`` and ``A_eq*x = b_eq``. Unless bounds
    are given, variables will have nonnegative values in the solution.

    If ``A`` is not given, then the dimension of the system will
    be determined by the length of ``C``.

    By default, all variables will be nonnegative. If ``bounds``
    is given as a single tuple, ``(lo, hi)``, then all variables
    will be constrained to be between ``lo`` and ``hi``. Use
    None for a ``lo`` or ``hi`` if it is unconstrained in the
    negative or positive direction, respectively, e.g.
    ``(None, 0)`` indicates nonpositive values. To set
    individual ranges, pass a list with length equal to the
    number of columns in ``A``, each element being a tuple; if
    only a few variables take on non-default values they can be
    passed as a dictionary with keys giving the corresponding
    column to which the variable is assigned, e.g. ``bounds={2:
    (1, 4)}`` would limit the 3rd variable to have a value in
    range ``[1, 4]``.

    Examples
    ========

    >>> from sympy.solvers.simplex import linprog
    >>> from sympy import symbols, Eq, linear_eq_to_matrix as M, Matrix
    >>> x = x1, x2, x3, x4 = symbols('x1:5')
    >>> X = Matrix(x)
    >>> c, d = M(5*x2 + x3 + 4*x4 - x1, x)
    >>> a, b = M([5*x2 + 2*x3 + 5*x4 - (x1 + 5)], x)
    >>> aeq, beq = M([Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)], x)
    >>> constr = [i <= j for i,j in zip(a*X, b)]
    >>> constr += [Eq(i, j) for i,j in zip(aeq*X, beq)]
    >>> linprog(c, a, b, aeq, beq)
    (9/2, [0, 1/2, 0, 1/2])
    >>> assert all(i.subs(dict(zip(x, _[1]))) for i in constr)

    See Also
    ========
    lpmin, lpmax
    """

    ## the objective
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError("C must be a single row.")

    ## the inequalities
    if not A:
        if b:
            raise ValueError("A and b must both be given")
        # the governing equations will be simple constraints
        # on variables
        A, b = zeros(0, C.cols), zeros(C.cols, 1)
    else:
        A, b = [Matrix(i) for i in (A, b)]

    if A.cols != C.cols:
        raise ValueError("number of columns in A and C must match")

    ## the equalities
    if A_eq is None:
        if not b_eq is None:
            raise ValueError("A_eq and b_eq must both be given")
    else:
        A_eq, b_eq = [Matrix(i) for i in (A_eq, b_eq)]
        # if x == y then x <= y and x >= y (-x <= -y)
        A = A.col_join(A_eq)
        A = A.col_join(-A_eq)
        b = b.col_join(b_eq)
        b = b.col_join(-b_eq)

    if not (bounds is None or bounds == {} or bounds == (0, None)):
        ## the bounds are interpreted
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all(
                type(i) is tuple and len(i) == 2 for i in bounds):
            pass # individual bounds
        elif type(bounds) is dict and all(
                type(i) is tuple and len(i) == 2
                for i in bounds.values()):
            # sparse bounds
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                i, j = db.popitem()
                bounds[i] = j  # IndexError if out-of-bounds indices
        else:
            raise ValueError("unexpected bounds %s" % bounds)
        A_, b_ = _handle_bounds(bounds)
        aux = A_.cols - A.cols
        if A:
            A = Matrix([[A, zeros(A.rows, aux)], [A_]])
            b = b.col_join(b_)
        else:
            A = A_
            b = b_
        C = C.row_join(zeros(1, aux))
    else:
        aux = -A.cols  # set so -aux will give all cols below

    o, p, d = _simplex(A, b, C)
    return o, p[:-aux]  # don't include aux values

def show_linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    from sympy import symbols
    ## the objective
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError("C must be a single row.")

    ## the inequalities
    if not A:
        if b:
            raise ValueError("A and b must both be given")
        # the governing equations will be simple constraints
        # on variables
        A, b = zeros(0, C.cols), zeros(C.cols, 1)
    else:
        A, b = [Matrix(i) for i in (A, b)]

    if A.cols != C.cols:
        raise ValueError("number of columns in A and C must match")

    ## the equalities
    if A_eq is None:
        if not b_eq is None:
            raise ValueError("A_eq and b_eq must both be given")
    else:
        A_eq, b_eq = [Matrix(i) for i in (A_eq, b_eq)]

    if not (bounds is None or bounds == {} or bounds == (0, None)):
        ## the bounds are interpreted
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all(
                type(i) is tuple and len(i) == 2 for i in bounds):
            pass # individual bounds
        elif type(bounds) is dict and all(
                type(i) is tuple and len(i) == 2
                for i in bounds.values()):
            # sparse bounds
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                i, j = db.popitem()
                bounds[i] = j  # IndexError if out-of-bounds indices
        else:
            raise ValueError("unexpected bounds %s" % bounds)

    x = Matrix(symbols('x1:%s' % (A.cols+1)))
    f,c = (C*x)[0], [i<=j for i,j in zip(A*x, b)] + [Eq(i,j) for i,j in zip(A_eq*x,b_eq)]
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            c.append(x[i]>=lo)
        if hi is not None:
            c.append(x[i]<=hi)
    return f,c
<|endoftext|>r"""
This module contains the implementation of the internal helper functions for the lie_group hint for
dsolve. These helper functions apply different heuristics on the given equation
and return the solution. These functions are used by :py:meth:`sympy.solvers.ode.single.LieGroup`

References
=========

- `abaco1_simple`, `function_sum` and `chi`  are referenced from E.S Cheb-Terrab, L.G.S Duarte
and L.A,C.P da Mota, Computer Algebra Solving of First Order ODEs Using
Symmetry Methods, pp. 7 - pp. 8

- `abaco1_product`, `abaco2_similar`, `abaco2_unique_unknown`, `linear`  and `abaco2_unique_general`
are referenced from E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
ODE Patterns, pp. 7 - pp. 12

- `bivariate` from Lie Groups and Differential Equations pp. 327 - pp. 329

"""
from itertools import islice

from sympy.core import Add, S, Mul, Pow
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, AppliedUndef, expand
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.functions import exp, log
from sympy.integrals.integrals import integrate
from sympy.polys import Poly
from sympy.polys.polytools import cancel, div
from sympy.simplify import (collect, powsimp,  # type: ignore
    separatevars, simplify)
from sympy.solvers import solve
from sympy.solvers.pde import pdsolve

from sympy.utilities import numbered_symbols
from sympy.solvers.deutils import _preprocess, ode_order
from .ode import checkinfsol


lie_heuristics = (
    "abaco1_simple",
    "abaco1_product",
    "abaco2_similar",
    "abaco2_unique_unknown",
    "abaco2_unique_general",
    "linear",
    "function_sum",
    "bivariate",
    "chi"
    )


def _ode_lie_group_try_heuristic(eq, heuristic, func, match, inf):

    xi = Function("xi")
    eta = Function("eta")
    f = func.func
    x = func.args[0]
    y = match['y']
    h = match['h']
    tempsol = []
    if not inf:
        try:
            inf = infinitesimals(eq, hint=heuristic, func=func, order=1, match=match)
        except ValueError:
            return None
    for infsim in inf:
        xiinf = (infsim[xi(x, func)]).subs(func, y)
        etainf = (infsim[eta(x, func)]).subs(func, y)
        # This condition creates recursion while using pdsolve.
        # Since the first step while solving a PDE of form
        # a*(f(x, y).diff(x)) + b*(f(x, y).diff(y)) + c = 0
        # is to solve the ODE dy/dx = b/a
        if simplify(etainf/xiinf) == h:
            continue
        rpde = f(x, y).diff(x)*xiinf + f(x, y).diff(y)*etainf
        r = pdsolve(rpde, func=f(x, y)).rhs
        s = pdsolve(rpde - 1, func=f(x, y)).rhs
        newcoord = [_lie_group_remove(coord) for coord in [r, s]]
        r = Dummy("r")
        s = Dummy("s")
        C1 = Symbol("C1")
        rcoord = newcoord[0]
        scoord = newcoord[-1]
        try:
            sol = solve([r - rcoord, s - scoord], x, y, dict=True)
            if sol == []:
                continue
        except NotImplementedError:
            continue
        else:
            sol = sol[0]
            xsub = sol[x]
            ysub = sol[y]
            num = simplify(scoord.diff(x) + scoord.diff(y)*h)
            denom = simplify(rcoord.diff(x) + rcoord.diff(y)*h)
            if num and denom:
                diffeq = simplify((num/denom).subs([(x, xsub), (y, ysub)]))
                sep = separatevars(diffeq, symbols=[r, s], dict=True)
                if sep:
                    # Trying to separate, r and s coordinates
                    deq = integrate((1/sep[s]), s) + C1 - integrate(sep['coeff']*sep[r], r)
                    # Substituting and reverting back to original coordinates
                    deq = deq.subs([(r, rcoord), (s, scoord)])
                    try:
                        sdeq = solve(deq, y)
                    except NotImplementedError:
                        tempsol.append(deq)
                    else:
                        return [Eq(f(x), sol) for sol in sdeq]


            elif denom: # (ds/dr) is zero which means s is constant
                return [Eq(f(x), solve(scoord - C1, y)[0])]

            elif num: # (dr/ds) is zero which means r is constant
                return [Eq(f(x), solve(rcoord - C1, y)[0])]

    # If nothing works, return solution as it is, without solving for y
    if tempsol:
        return [Eq(sol.subs(y, f(x)), 0) for sol in tempsol]
    return None


def _ode_lie_group( s, func, order, match):

    heuristics = lie_heuristics
    inf = {}
    f = func.func
    x = func.args[0]
    df = func.diff(x)
    xi = Function("xi")
    eta = Function("eta")
    xis = match['xi']
    etas = match['eta']
    y = match.pop('y', None)
    if y:
        h = -simplify(match[match['d']]/match[match['e']])
    else:
        y = Dummy("y")
        h = s.subs(func, y)

    if xis is not None and etas is not None:
        inf = [{xi(x, f(x)): S(xis), eta(x, f(x)): S(etas)}]

        if checkinfsol(Eq(df, s), inf, func=f(x), order=1)[0][0]:
            heuristics = ["user_defined"] + list(heuristics)

    match = {'h': h, 'y': y}

    # This is done so that if any heuristic raises a ValueError
    # another heuristic can be used.
    sol = None
    for heuristic in heuristics:
        sol = _ode_lie_group_try_heuristic(Eq(df, s), heuristic, func, match, inf)
        if sol:
            return sol
    return sol


def infinitesimals(eq, func=None, order=None, hint='default', match=None):
    r"""
    The infinitesimal functions of an ordinary differential equation, `\xi(x,y)`
    and `\eta(x,y)`, are the infinitesimals of the Lie group of point transformations
    for which the differential equation is invariant. So, the ODE `y'=f(x,y)`
    would admit a Lie group `x^*=X(x,y;\varepsilon)=x+\varepsilon\xi(x,y)`,
    `y^*=Y(x,y;\varepsilon)=y+\varepsilon\eta(x,y)` such that `(y^*)'=f(x^*, y^*)`.
    A change of coordinates, to `r(x,y)` and `s(x,y)`, can be performed so this Lie group
    becomes the translation group, `r^*=r` and `s^*=s+\varepsilon`.
    They are tangents to the coordinate curves of the new system.

    Consider the transformation `(x, y) \to (X, Y)` such that the
    differential equation remains invariant. `\xi` and `\eta` are the tangents to
    the transformed coordinates `X` and `Y`, at `\varepsilon=0`.

    .. math:: \left(\frac{\partial X(x,y;\varepsilon)}{\partial\varepsilon
                }\right)|_{\varepsilon=0} = \xi,
              \left(\frac{\partial Y(x,y;\varepsilon)}{\partial\varepsilon
                }\right)|_{\varepsilon=0} = \eta,

    The infinitesimals can be found by solving the following PDE:

        >>> from sympy import Function, Eq, pprint
        >>> from sympy.abc import x, y
        >>> xi, eta, h = map(Function, ['xi', 'eta', 'h'])
        >>> h = h(x, y)  # dy/dx = h
        >>> eta = eta(x, y)
        >>> xi = xi(x, y)
        >>> genform = Eq(eta.diff(x) + (eta.diff(y) - xi.diff(x))*h
        ... - (xi.diff(y))*h**2 - xi*(h.diff(x)) - eta*(h.diff(y)), 0)
        >>> pprint(genform)
        /d               d           \                     d              2       d                       d             d
        |--(eta(x, y)) - --(xi(x, y))|*h(x, y) - eta(x, y)*--(h(x, y)) - h (x, y)*--(xi(x, y)) - xi(x, y)*--(h(x, y)) + --(eta(x, y)) = 0
        \dy              dx          /                     dy                     dy                      dx            dx

    Solving the above mentioned PDE is not trivial, and can be solved only by
    making intelligent assumptions for `\xi` and `\eta` (heuristics). Once an
    infinitesimal is found, the attempt to find more heuristics stops. This is done to
    optimise the speed of solving the differential equation. If a list of all the
    infinitesimals is needed, ``hint`` should be flagged as ``all``, which gives
    the complete list of infinitesimals. If the infinitesimals for a particular
    heuristic needs to be found, it can be passed as a flag to ``hint``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.solvers.ode.lie_group import infinitesimals
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = f(x).diff(x) - x**2*f(x)
    >>> infinitesimals(eq)
    [{eta(x, f(x)): exp(x**3/3), xi(x, f(x)): 0}]

    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14

    """

    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if not func:
        eq, func = _preprocess(eq)
    variables = func.args
    if len(variables) != 1:
        raise ValueError("ODE's have only one independent variable")
    else:
        x = variables[0]
        if not order:
            order = ode_order(eq, func)
        if order != 1:
            raise NotImplementedError("Infinitesimals for only "
                "first order ODE's have been implemented")
        else:
            df = func.diff(x)
            # Matching differential equation of the form a*df + b
            a = Wild('a', exclude = [df])
            b = Wild('b', exclude = [df])
            if match:  # Used by lie_group hint
                h = match['h']
                y = match['y']
            else:
                match = collect(expand(eq), df).match(a*df + b)
                if match:
                    h = -simplify(match[b]/match[a])
                else:
                    try:
                        sol = solve(eq, df)
                    except NotImplementedError:
                        raise NotImplementedError("Infinitesimals for the "
                            "first order ODE could not be found")
                    else:
                        h = sol[0]  # Find infinitesimals for one solution
                y = Dummy("y")
                h = h.subs(func, y)

            u = Dummy("u")
            hx = h.diff(x)
            hy = h.diff(y)
            hinv = ((1/h).subs([(x, u), (y, x)])).subs(u, y)  # Inverse ODE
            match = {'h': h, 'func': func, 'hx': hx, 'hy': hy, 'y': y, 'hinv': hinv}
            if hint == 'all':
                xieta = []
                for heuristic in lie_heuristics:
                    function = globals()['lie_heuristic_' + heuristic]
                    inflist = function(match, comp=True)
                    if inflist:
                        xieta.extend([inf for inf in inflist if inf not in xieta])
                if xieta:
                    return xieta
                else:
                    raise NotImplementedError("Infinitesimals could not be found for "
                        "the given ODE")

            elif hint == 'default':
                for heuristic in lie_heuristics:
                    function = globals()['lie_heuristic_' + heuristic]
                    xieta = function(match, comp=False)
                    if xieta:
                        return xieta

                raise NotImplementedError("Infinitesimals could not be found for"
                    " the given ODE")

            elif hint not in lie_heuristics:
                raise ValueError("Heuristic not recognized: " + hint)

            else:
                function = globals()['lie_heuristic_' + hint]
                xieta = function(match, comp=True)
                if xieta:
                    return xieta
                else:
                    raise ValueError("Infinitesimals could not be found using the"
                         " given heuristic")


def lie_heuristic_abaco1_simple(match, comp=False):
    r"""
    The first heuristic uses the following four sets of
    assumptions on `\xi` and `\eta`

    .. math:: \xi = 0, \eta = f(x)

    .. math:: \xi = 0, \eta = f(y)

    .. math:: \xi = f(x), \eta = 0

    .. math:: \xi = f(y), \eta = 0

    The success of this heuristic is determined by algebraic factorisation.
    For the first assumption `\xi = 0` and `\eta` to be a function of `x`, the PDE

    .. math:: \frac{\partial \eta}{\partial x} + (\frac{\partial \eta}{\partial y}
                - \frac{\partial \xi}{\partial x})*h
                - \frac{\partial \xi}{\partial y}*h^{2}
                - \xi*\frac{\partial h}{\partial x} - \eta*\frac{\partial h}{\partial y} = 0

    reduces to `f'(x) - f\frac{\partial h}{\partial y} = 0`
    If `\frac{\partial h}{\partial y}` is a function of `x`, then this can usually
    be integrated easily. A similar idea is applied to the other 3 assumptions as well.


    References
    ==========

    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra
      Solving of First Order ODEs Using Symmetry Methods, pp. 8


    """

    xieta = []
    y = match['y']
    h = match['h']
    func = match['func']
    x = func.args[0]
    hx = match['hx']
    hy = match['hy']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    hysym = hy.free_symbols
    if y not in hysym:
        try:
            fx = exp(integrate(hy, x))
        except NotImplementedError:
            pass
        else:
            inf = {xi: S.Zero, eta: fx}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    factor = hy/h
    facsym = factor.free_symbols
    if x not in facsym:
        try:
            fy = exp(integrate(factor, y))
        except NotImplementedError:
            pass
        else:
            inf = {xi: S.Zero, eta: fy.subs(y, func)}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    factor = -hx/h
    facsym = factor.free_symbols
    if y not in facsym:
        try:
            fx = exp(integrate(factor, x))
        except NotImplementedError:
            pass
        else:
            inf = {xi: fx, eta: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    factor = -hx/(h**2)
    facsym = factor.free_symbols
    if x not in facsym:
        try:
            fy = exp(integrate(factor, y))
        except NotImplementedError:
            pass
        else:
            inf = {xi: fy.subs(y, func), eta: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    if xieta:
        return xieta

def lie_heuristic_abaco1_product(match, comp=False):
    r"""
    The second heuristic uses the following two assumptions on `\xi` and `\eta`

    .. math:: \eta = 0, \xi = f(x)*g(y)

    .. math:: \eta = f(x)*g(y), \xi = 0

    The first assumption of this heuristic holds good if
    `\frac{1}{h^{2}}\frac{\partial^2}{\partial x \partial y}\log(h)` is
    separable in `x` and `y`, then the separated factors containing `x`
    is `f(x)`, and `g(y)` is obtained by

    .. math:: e^{\int f\frac{\partial}{\partial x}\left(\frac{1}{f*h}\right)\,dy}

    provided `f\frac{\partial}{\partial x}\left(\frac{1}{f*h}\right)` is a function
    of `y` only.

    The second assumption holds good if `\frac{dy}{dx} = h(x, y)` is rewritten as
    `\frac{dy}{dx} = \frac{1}{h(y, x)}` and the same properties of the first assumption
    satisfies. After obtaining `f(x)` and `g(y)`, the coordinates are again
    interchanged, to get `\eta` as `f(x)*g(y)`


    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 7 - pp. 8

    """

    xieta = []
    y = match['y']
    h = match['h']
    hinv = match['hinv']
    func = match['func']
    x = func.args[0]
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)


    inf = separatevars(((log(h).diff(y)).diff(x))/h**2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx*((1/(fx*h)).diff(x)))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            inf = {eta: S.Zero, xi: (fx*gy).subs(y, func)}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    u1 = Dummy("u1")
    inf = separatevars(((log(hinv).diff(y)).diff(x))/hinv**2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx*((1/(fx*hinv)).diff(x)))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            etaval = fx*gy
            etaval = (etaval.subs([(x, u1), (y, x)])).subs(u1, y)
            inf = {eta: etaval.subs(y, func), xi: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)

    if xieta:
        return xieta

def lie_heuristic_bivariate(match, comp=False):
    r"""
    The third heuristic assumes the infinitesimals `\xi` and `\eta`
    to be bi-variate polynomials in `x` and `y`. The assumption made here
    for the logic below is that `h` is a rational function in `x` and `y`
    though that may not be necessary for the infinitesimals to be
    bivariate polynomials. The coefficients of the infinitesimals
    are found out by substituting them in the PDE and grouping similar terms
    that are polynomials and since they form a linear system, solve and check
    for non trivial solutions. The degree of the assumed bivariates
    are increased till a certain maximum value.

    References
    ==========
    - Lie Groups and Differential Equations
      pp. 327 - pp. 329

    """

    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    if h.is_rational_function():
        # The maximum degree that the infinitesimals can take is
        # calculated by this technique.
        etax, etay, etad, xix, xiy, xid = symbols("etax etay etad xix xiy xid")
        ipde = etax + (etay - xix)*h - xiy*h**2 - xid*hx - etad*hy
        num, denom = cancel(ipde).as_numer_denom()
        deg = Poly(num, x, y).total_degree()
        deta = Function('deta')(x, y)
        dxi = Function('dxi')(x, y)
        ipde = (deta.diff(x) + (deta.diff(y) - dxi.diff(x))*h - (dxi.diff(y))*h**2
            - dxi*hx - deta*hy)
        xieq = Symbol("xi0")
        etaeq = Symbol("eta0")

        for i in range(deg + 1):
            if i:
                xieq += Add(*[
                    Symbol("xi_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                    for power in range(i + 1)])
                etaeq += Add(*[
                    Symbol("eta_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                    for power in range(i + 1)])
            pden, denom = (ipde.subs({dxi: xieq, deta: etaeq}).doit()).as_numer_denom()
            pden = expand(pden)

            # If the individual terms are monomials, the coefficients
            # are grouped
            if pden.is_polynomial(x, y) and pden.is_Add:
                polyy = Poly(pden, x, y).as_dict()
            if polyy:
                symset = xieq.free_symbols.union(etaeq.free_symbols) - {x, y}
                soldict = solve(polyy.values(), *symset)
                if isinstance(soldict, list):
                    soldict = soldict[0]
                if any(soldict.values()):
                    xired = xieq.subs(soldict)
                    etared = etaeq.subs(soldict)
                    # Scaling is done by substituting one for the parameters
                    # This can be any number except zero.
                    dict_ = dict.fromkeys(symset, 1)
                    inf = {eta: etared.subs(dict_).subs(y, func),
                        xi: xired.subs(dict_).subs(y, func)}
                    return [inf]

def lie_heuristic_chi(match, comp=False):
    r"""
    The aim of the fourth heuristic is to find the function `\chi(x, y)`
    that satisfies the PDE `\frac{d\chi}{dx} + h\frac{d\chi}{dx}
    - \frac{\partial h}{\partial y}\chi = 0`.

    This assumes `\chi` to be a bivariate polynomial in `x` and `y`. By intuition,
    `h` should be a rational function in `x` and `y`. The method used here is
    to substitute a general binomial for `\chi` up to a certain maximum degree
    is reached. The coefficients of the polynomials, are calculated by by collecting
    terms of the same order in `x` and `y`.

    After finding `\chi`, the next step is to use `\eta = \xi*h + \chi`, to
    determine `\xi` and `\eta`. This can be done by dividing `\chi` by `h`
    which would give `-\xi` as the quotient and `\eta` as the remainder.


    References
    ==========
    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra
      Solving of First Order ODEs Using Symmetry Methods, pp. 8

    """

    h = match['h']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    if h.is_rational_function():
        schi, schix, schiy = symbols("schi, schix, schiy")
        cpde = schix + h*schiy - hy*schi
        num, denom = cancel(cpde).as_numer_denom()
        deg = Poly(num, x, y).total_degree()

        chi = Function('chi')(x, y)
        chix = chi.diff(x)
        chiy = chi.diff(y)
        cpde = chix + h*chiy - hy*chi
        chieq = Symbol("chi")
        for i in range(1, deg + 1):
            chieq += Add(*[
                Symbol("chi_" + str(power) + "_" + str(i - power))*x**power*y**(i - power)
                for power in range(i + 1)])
            cnum, cden = cancel(cpde.subs({chi : chieq}).doit()).as_numer_denom()
            cnum = expand(cnum)
            if cnum.is_polynomial(x, y) and cnum.is_Add:
                cpoly = Poly(cnum, x, y).as_dict()
                if cpoly:
                    solsyms = chieq.free_symbols - {x, y}
                    soldict = solve(cpoly.values(), *solsyms)
                    if isinstance(soldict, list):
                        soldict = soldict[0]
                    if any(soldict.values()):
                        chieq = chieq.subs(soldict)
                        dict_ = dict.fromkeys(solsyms, 1)
                        chieq = chieq.subs(dict_)
                        # After finding chi, the main aim is to find out
                        # eta, xi by the equation eta = xi*h + chi
                        # One method to set xi, would be rearranging it to
                        # (eta/h) - xi = (chi/h). This would mean dividing
                        # chi by h would give -xi as the quotient and eta
                        # as the remainder. Thanks to Sean Vig for suggesting
                        # this method.
                        xic, etac = div(chieq, h)
                        inf = {eta: etac.subs(y, func), xi: -xic.subs(y, func)}
                        return [inf]

def lie_heuristic_function_sum(match, comp=False):
    r"""
    This heuristic uses the following two assumptions on `\xi` and `\eta`

    .. math:: \eta = 0, \xi = f(x) + g(y)

    .. math:: \eta = f(x) + g(y), \xi = 0

    The first assumption of this heuristic holds good if

    .. math:: \frac{\partial}{\partial y}[(h\frac{\partial^{2}}{
                \partial x^{2}}(h^{-1}))^{-1}]

    is separable in `x` and `y`,

    1. The separated factors containing `y` is `\frac{\partial g}{\partial y}`.
       From this `g(y)` can be determined.
    2. The separated factors containing `x` is `f''(x)`.
    3. `h\frac{\partial^{2}}{\partial x^{2}}(h^{-1})` equals
       `\frac{f''(x)}{f(x) + g(y)}`. From this `f(x)` can be determined.

    The second assumption holds good if `\frac{dy}{dx} = h(x, y)` is rewritten as
    `\frac{dy}{dx} = \frac{1}{h(y, x)}` and the same properties of the first
    assumption satisfies. After obtaining `f(x)` and `g(y)`, the coordinates
    are again interchanged, to get `\eta` as `f(x) + g(y)`.

    For both assumptions, the constant factors are separated among `g(y)`
    and `f''(x)`, such that `f''(x)` obtained from 3] is the same as that
    obtained from 2]. If not possible, then this heuristic fails.


    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 7 - pp. 8

    """

    xieta = []
    h = match['h']
    func = match['func']
    hinv = match['hinv']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    for odefac in [h, hinv]:
        factor = odefac*((1/odefac).diff(x, 2))
        sep = separatevars((1/factor).diff(y), dict=True, symbols=[x, y])
        if sep and sep['coeff'] and sep[x].has(x) and sep[y].has(y):
            k = Dummy("k")
            try:
                gy = k*integrate(sep[y], y)
            except NotImplementedError:
                pass
            else:
                fdd = 1/(k*sep[x]*sep['coeff'])
                fx = simplify(fdd/factor - gy)
                check = simplify(fx.diff(x, 2) - fdd)
                if fx:
                    if not check:
                        fx = fx.subs(k, 1)
                        gy = (gy/k)
                    else:
                        sol = solve(check, k)
                        if sol:
                            sol = sol[0]
                            fx = fx.subs(k, sol)
                            gy = (gy/k)*sol
                        else:
                            continue
                    if odefac == hinv:  # Inverse ODE
                        fx = fx.subs(x, y)
                        gy = gy.subs(y, x)
                    etaval = factor_terms(fx + gy)
                    if etaval.is_Mul:
                        etaval = Mul(*[arg for arg in etaval.args if arg.has(x, y)])
                    if odefac == hinv:  # Inverse ODE
                        inf = {eta: etaval.subs(y, func), xi : S.Zero}
                    else:
                        inf = {xi: etaval.subs(y, func), eta : S.Zero}
                    if not comp:
                        return [inf]
                    else:
                        xieta.append(inf)

        if xieta:
            return xieta

def lie_heuristic_abaco2_similar(match, comp=False):
    r"""
    This heuristic uses the following two assumptions on `\xi` and `\eta`

    .. math:: \eta = g(x), \xi = f(x)

    .. math:: \eta = f(y), \xi = g(y)

    For the first assumption,

    1. First `\frac{\frac{\partial h}{\partial y}}{\frac{\partial^{2} h}{
       \partial yy}}` is calculated. Let us say this value is A

    2. If this is constant, then `h` is matched to the form `A(x) + B(x)e^{
       \frac{y}{C}}` then, `\frac{e^{\int \frac{A(x)}{C} \,dx}}{B(x)}` gives `f(x)`
       and `A(x)*f(x)` gives `g(x)`

    3. Otherwise `\frac{\frac{\partial A}{\partial X}}{\frac{\partial A}{
       \partial Y}} = \gamma` is calculated. If

       a] `\gamma` is a function of `x` alone

       b] `\frac{\gamma\frac{\partial h}{\partial y} - \gamma'(x) - \frac{
       \partial h}{\partial x}}{h + \gamma} = G` is a function of `x` alone.
       then, `e^{\int G \,dx}` gives `f(x)` and `-\gamma*f(x)` gives `g(x)`

    The second assumption holds good if `\frac{dy}{dx} = h(x, y)` is rewritten as
    `\frac{dy}{dx} = \frac{1}{h(y, x)}` and the same properties of the first assumption
    satisfies. After obtaining `f(x)` and `g(x)`, the coordinates are again
    interchanged, to get `\xi` as `f(x^*)` and `\eta` as `g(y^*)`

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """

    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    hinv = match['hinv']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    factor = cancel(h.diff(y)/h.diff(y, 2))
    factorx = factor.diff(x)
    factory = factor.diff(y)
    if not factor.has(x) and not factor.has(y):
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        match = h.match(A + B*exp(y/C))
        try:
            tau = exp(-integrate(match[A]/match[C]), x)/match[B]
        except NotImplementedError:
            pass
        else:
            gx = match[A]*tau
            return [{xi: tau, eta: gx}]

    else:
        gamma = cancel(factorx/factory)
        if not gamma.has(y):
            tauint = cancel((gamma*hy - gamma.diff(x) - hx)/(h + gamma))
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    pass
                else:
                    gx = -tau*gamma
                    return [{xi: tau, eta: gx}]

    factor = cancel(hinv.diff(y)/hinv.diff(y, 2))
    factorx = factor.diff(x)
    factory = factor.diff(y)
    if not factor.has(x) and not factor.has(y):
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        match = h.match(A + B*exp(y/C))
        try:
            tau = exp(-integrate(match[A]/match[C]), x)/match[B]
        except NotImplementedError:
            pass
        else:
            gx = match[A]*tau
            return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]

    else:
        gamma = cancel(factorx/factory)
        if not gamma.has(y):
            tauint = cancel((gamma*hinv.diff(y) - gamma.diff(x) - hinv.diff(x))/(
                hinv + gamma))
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    pass
                else:
                    gx = -tau*gamma
                    return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]


def lie_heuristic_abaco2_unique_unknown(match, comp=False):
    r"""
    This heuristic assumes the presence of unknown functions or known functions
    with non-integer powers.

    1. A list of all functions and non-integer powers containing x and y
    2. Loop over each element `f` in the list, find `\frac{\frac{\partial f}{\partial x}}{
       \frac{\partial f}{\partial x}} = R`

       If it is separable in `x` and `y`, let `X` be the factors containing `x`. Then

       a] Check if `\xi = X` and `\eta = -\frac{X}{R}` satisfy the PDE. If yes, then return
          `\xi` and `\eta`
       b] Check if `\xi = \frac{-R}{X}` and `\eta = -\frac{1}{X}` satisfy the PDE.
           If yes, then return `\xi` and `\eta`

       If not, then check if

       a] :math:`\xi = -R,\eta = 1`

       b] :math:`\xi = 1, \eta = -\frac{1}{R}`

       are solutions.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """

    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    funclist = []
    for atom in h.atoms(Pow):
        base, exp = atom.as_base_exp()
        if base.has(x) and base.has(y):
            if not exp.is_Integer:
                funclist.append(atom)

    for function in h.atoms(AppliedUndef):
        syms = function.free_symbols
        if x in syms and y in syms:
            funclist.append(function)

    for f in funclist:
        frac = cancel(f.diff(y)/f.diff(x))
        sep = separatevars(frac, dict=True, symbols=[x, y])
        if sep and sep['coeff']:
            xitry1 = sep[x]
            etatry1 = -1/(sep[y]*sep['coeff'])
            pde1 = etatry1.diff(y)*h - xitry1.diff(x)*h - xitry1*hx - etatry1*hy
            if not simplify(pde1):
                return [{xi: xitry1, eta: etatry1.subs(y, func)}]
            xitry2 = 1/etatry1
            etatry2 = 1/xitry1
            pde2 = etatry2.diff(x) - (xitry2.diff(y))*h**2 - xitry2*hx - etatry2*hy
            if not simplify(expand(pde2)):
                return [{xi: xitry2.subs(y, func), eta: etatry2}]

        else:
            etatry = -1/frac
            pde = etatry.diff(x) + etatry.diff(y)*h - hx - etatry*hy
            if not simplify(pde):
                return [{xi: S.One, eta: etatry.subs(y, func)}]
            xitry = -frac
            pde = -xitry.diff(x)*h -xitry.diff(y)*h**2 - xitry*hx -hy
            if not simplify(expand(pde)):
                return [{xi: xitry.subs(y, func), eta: S.One}]


def lie_heuristic_abaco2_unique_general(match, comp=False):
    r"""
    This heuristic finds if infinitesimals of the form `\eta = f(x)`, `\xi = g(y)`
    without making any assumptions on `h`.

    The complete sequence of steps is given in the paper mentioned below.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    A = hx.diff(y)
    B = hy.diff(y) + hy**2
    C = hx.diff(x) - hx**2

    if not (A and B and C):
        return

    Ax = A.diff(x)
    Ay = A.diff(y)
    Axy = Ax.diff(y)
    Axx = Ax.diff(x)
    Ayy = Ay.diff(y)
    D = simplify(2*Axy + hx*Ay - Ax*hy + (hx*hy + 2*A)*A)*A - 3*Ax*Ay
    if not D:
        E1 = simplify(3*Ax**2 + ((hx**2 + 2*C)*A - 2*Axx)*A)
        if E1:
            E2 = simplify((2*Ayy + (2*B - hy**2)*A)*A - 3*Ay**2)
            if not E2:
                E3 = simplify(
                    E1*((28*Ax + 4*hx*A)*A**3 - E1*(hy*A + Ay)) - E1.diff(x)*8*A**4)
                if not E3:
                    etaval = cancel((4*A**3*(Ax - hx*A) + E1*(hy*A - Ay))/(S(2)*A*E1))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            xival = -4*A**3*etaval/E1
                            if y not in xival:
                                return [{xi: xival, eta: etaval.subs(y, func)}]

    else:
        E1 = simplify((2*Ayy + (2*B - hy**2)*A)*A - 3*Ay**2)
        if E1:
            E2 = simplify(
                4*A**3*D - D**2 + E1*((2*Axx - (hx**2 + 2*C)*A)*A - 3*Ax**2))
            if not E2:
                E3 = simplify(
                   -(A*D)*E1.diff(y) + ((E1.diff(x) - hy*D)*A + 3*Ay*D +
                    (A*hx - 3*Ax)*E1)*E1)
                if not E3:
                    etaval = cancel(((A*hx - Ax)*E1 - (Ay + A*hy)*D)/(S(2)*A*D))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            xival = -E1*etaval/D
                            if y not in xival:
                                return [{xi: xival, eta: etaval.subs(y, func)}]


def lie_heuristic_linear(match, comp=False):
    r"""
    This heuristic assumes

    1. `\xi = ax + by + c` and
    2. `\eta = fx + gy + h`

    After substituting the following assumptions in the determining PDE, it
    reduces to

    .. math:: f + (g - a)h - bh^{2} - (ax + by + c)\frac{\partial h}{\partial x}
                 - (fx + gy + c)\frac{\partial h}{\partial y}

    Solving the reduced PDE obtained, using the method of characteristics, becomes
    impractical. The method followed is grouping similar terms and solving the system
    of linear equations obtained. The difference between the bivariate heuristic is that
    `h` need not be a rational function in this case.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)

    coeffdict = {}
    symbols = numbered_symbols("c", cls=Dummy)
    symlist = [next(symbols) for _ in islice(symbols, 6)]
    C0, C1, C2, C3, C4, C5 = symlist
    pde = C3 + (C4 - C0)*h - (C0*x + C1*y + C2)*hx - (C3*x + C4*y + C5)*hy - C1*h**2
    pde, denom = pde.as_numer_denom()
    pde = powsimp(expand(pde))
    if pde.is_Add:
        terms = pde.args
        for term in terms:
            if term.is_Mul:
                rem = Mul(*[m for m in term.args if not m.has(x, y)])
                xypart = term/rem
                if xypart not in coeffdict:
                    coeffdict[xypart] = rem
                else:
                    coeffdict[xypart] += rem
            else:
                if term not in coeffdict:
                    coeffdict[term] = S.One
                else:
                    coeffdict[term] += S.One

    sollist = coeffdict.values()
    soldict = solve(sollist, symlist)
    if soldict:
        if isinstance(soldict, list):
            soldict = soldict[0]
        subval = soldict.values()
        if any(t for t in subval):
            onedict = dict(zip(symlist, [1]*6))
            xival = C0*x + C1*func + C2
            etaval = C3*x + C4*func + C5
            xival = xival.subs(soldict)
            etaval = etaval.subs(soldict)
            xival = xival.subs(onedict)
            etaval = etaval.subs(onedict)
            return [{xi: xival, eta: etaval}]


def _lie_group_remove(coords):
    r"""
    This function is strictly meant for internal use by the Lie group ODE solving
    method. It replaces arbitrary functions returned by pdsolve as follows:

    1] If coords is an arbitrary function, then its argument is returned.
    2] An arbitrary function in an Add object is replaced by zero.
    3] An arbitrary function in a Mul object is replaced by one.
    4] If there is no arbitrary function coords is returned unchanged.

    Examples
    ========

    >>> from sympy.solvers.ode.lie_group import _lie_group_remove
    >>> from sympy import Function
    >>> from sympy.abc import x, y
    >>> F = Function("F")
    >>> eq = x**2*y
    >>> _lie_group_remove(eq)
    x**2*y
    >>> eq = F(x**2*y)
    >>> _lie_group_remove(eq)
    x**2*y
    >>> eq = x*y**2 + F(x**3)
    >>> _lie_group_remove(eq)
    x*y**2
    >>> eq = (F(x**3) + y)*x**4
    >>> _lie_group_remove(eq)
    x**4*y

    """
    if isinstance(coords, AppliedUndef):
        return coords.args[0]
    elif coords.is_Add:
        subfunc = coords.atoms(AppliedUndef)
        if subfunc:
            for func in subfunc:
                coords = coords.subs(func, 0)
        return coords
    elif coords.is_Pow:
        base, expr = coords.as_base_exp()
        base = _lie_group_remove(base)
        expr = _lie_group_remove(expr)
        return base**expr
    elif coords.is_Mul:
        mulargs = []
        coordargs = coords.args
        for arg in coordargs:
            if not isinstance(coords, AppliedUndef):
                mulargs.append(_lie_group_remove(arg))
        return Mul(*mulargs)
    return coords
<|endoftext|>from sympy.core import S, Pow
from sympy.core.function import (Derivative, AppliedUndef, diff)
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify

from sympy.logic.boolalg import BooleanAtom
from sympy.functions import exp
from sympy.series import Order
from sympy.simplify.simplify import simplify, posify, besselsimp
from sympy.simplify.trigsimp import trigsimp
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.solvers import solve
from sympy.solvers.deutils import _preprocess, ode_order
from sympy.utilities.iterables import iterable, is_sequence


def sub_func_doit(eq, func, new):
    r"""
    When replacing the func with something else, we usually want the
    derivative evaluated, so this function helps in making that happen.

    Examples
    ========

    >>> from sympy import Derivative, symbols, Function
    >>> from sympy.solvers.ode.subscheck import sub_func_doit
    >>> x, z = symbols('x, z')
    >>> y = Function('y')

    >>> sub_func_doit(3*Derivative(y(x), x) - 1, y(x), x)
    2

    >>> sub_func_doit(x*Derivative(y(x), x) - y(x)**2 + y(x), y(x),
    ... 1/(x*(z + 1/x)))
    x*(-1/(x**2*(z + 1/x)) + 1/(x**3*(z + 1/x)**2)) + 1/(x*(z + 1/x))
    ...- 1/(x**2*(z + 1/x)**2)
    """
    reps= {func: new}
    for d in eq.atoms(Derivative):
        if d.expr == func:
            reps[d] = new.diff(*d.variable_count)
        else:
            reps[d] = d.xreplace({func: new}).doit(deep=False)
    return eq.xreplace(reps)


def checkodesol(ode, sol, func=None, order='auto', solve_for_func=True):
    r"""
    Substitutes ``sol`` into ``ode`` and checks that the result is ``0``.

    This works when ``func`` is one function, like `f(x)` or a list of
    functions like `[f(x), g(x)]` when `ode` is a system of ODEs.  ``sol`` can
    be a single solution or a list of solutions.  Each solution may be an
    :py:class:`~sympy.core.relational.Equality` that the solution satisfies,
    e.g. ``Eq(f(x), C1), Eq(f(x) + C1, 0)``; or simply an
    :py:class:`~sympy.core.expr.Expr`, e.g. ``f(x) - C1``. In most cases it
    will not be necessary to explicitly identify the function, but if the
    function cannot be inferred from the original equation it can be supplied
    through the ``func`` argument.

    If a sequence of solutions is passed, the same sort of container will be
    used to return the result for each solution.

    It tries the following methods, in order, until it finds zero equivalence:

    1. Substitute the solution for `f` in the original equation.  This only
       works if ``ode`` is solved for `f`.  It will attempt to solve it first
       unless ``solve_for_func == False``.
    2. Take `n` derivatives of the solution, where `n` is the order of
       ``ode``, and check to see if that is equal to the solution.  This only
       works on exact ODEs.
    3. Take the 1st, 2nd, ..., `n`\th derivatives of the solution, each time
       solving for the derivative of `f` of that order (this will always be
       possible because `f` is a linear operator). Then back substitute each
       derivative into ``ode`` in reverse order.

    This function returns a tuple.  The first item in the tuple is ``True`` if
    the substitution results in ``0``, and ``False`` otherwise. The second
    item in the tuple is what the substitution results in.  It should always
    be ``0`` if the first item is ``True``. Sometimes this function will
    return ``False`` even when an expression is identically equal to ``0``.
    This happens when :py:meth:`~sympy.simplify.simplify.simplify` does not
    reduce the expression to ``0``.  If an expression returned by this
    function vanishes identically, then ``sol`` really is a solution to
    the ``ode``.

    If this function seems to hang, it is probably because of a hard
    simplification.

    To use this function to test, test the first item of the tuple.

    Examples
    ========

    >>> from sympy import (Eq, Function, checkodesol, symbols,
    ...     Derivative, exp)
    >>> x, C1, C2 = symbols('x,C1,C2')
    >>> f, g = symbols('f g', cls=Function)
    >>> checkodesol(f(x).diff(x), Eq(f(x), C1))
    (True, 0)
    >>> assert checkodesol(f(x).diff(x), C1)[0]
    >>> assert not checkodesol(f(x).diff(x), x)[0]
    >>> checkodesol(f(x).diff(x, 2), x**2)
    (False, 2)

    >>> eqs = [Eq(Derivative(f(x), x), f(x)), Eq(Derivative(g(x), x), g(x))]
    >>> sol = [Eq(f(x), C1*exp(x)), Eq(g(x), C2*exp(x))]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    """
    if iterable(ode):
        return checksysodesol(ode, sol, func=func)

    if not isinstance(ode, Equality):
        ode = Eq(ode, 0)
    if func is None:
        try:
            _, func = _preprocess(ode.lhs)
        except ValueError:
            funcs = [s.atoms(AppliedUndef) for s in (
                sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(*funcs)
            if len(funcs) != 1:
                raise ValueError(
                    'must pass func arg to checkodesol for this case.')
            func = funcs.pop()
    if not isinstance(func, AppliedUndef) or len(func.args) != 1:
        raise ValueError(
            "func must be a function of one variable, not %s" % func)
    if is_sequence(sol, set):
        return type(sol)([checkodesol(ode, i, order=order, solve_for_func=solve_for_func) for i in sol])

    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed

    if order == 'auto':
        order = ode_order(ode, func)
    solved = sol.lhs == func and not sol.rhs.has(func)
    if solve_for_func and not solved:
        rhs = solve(sol, func)
        if rhs:
            eqs = [Eq(func, t) for t in rhs]
            if len(rhs) == 1:
                eqs = eqs[0]
            return checkodesol(ode, eqs, order=order,
                solve_for_func=False)

    x = func.args[0]

    # Handle series solutions here
    if sol.has(Order):
        assert sol.lhs == func
        Oterm = sol.rhs.getO()
        solrhs = sol.rhs.removeO()

        Oexpr = Oterm.expr
        assert isinstance(Oexpr, Pow)
        sorder = Oexpr.exp
        assert Oterm == Order(x**sorder)

        odesubs = (ode.lhs-ode.rhs).subs(func, solrhs).doit().expand()

        neworder = Order(x**(sorder - order))
        odesubs = odesubs + neworder
        assert odesubs.getO() == neworder
        residual = odesubs.removeO()

        return (residual == 0, residual)

    s = True
    testnum = 0
    while s:
        if testnum == 0:
            # First pass, try substituting a solved solution directly into the
            # ODE. This has the highest chance of succeeding.
            ode_diff = ode.lhs - ode.rhs

            if sol.lhs == func:
                s = sub_func_doit(ode_diff, func, sol.rhs)
                s = besselsimp(s)
            else:
                testnum += 1
                continue
            ss = simplify(s.rewrite(exp))
            if ss:
                # with the new numer_denom in power.py, if we do a simple
                # expansion then testnum == 0 verifies all solutions.
                s = ss.expand(force=True)
            else:
                s = 0
            testnum += 1
        elif testnum == 1:
            # Second pass. If we cannot substitute f, try seeing if the nth
            # derivative is equal, this will only work for odes that are exact,
            # by definition.
            s = simplify(
                trigsimp(diff(sol.lhs, x, order) - diff(sol.rhs, x, order)) -
                trigsimp(ode.lhs) + trigsimp(ode.rhs))
            # s2 = simplify(
            #     diff(sol.lhs, x, order) - diff(sol.rhs, x, order) - \
            #     ode.lhs + ode.rhs)
            testnum += 1
        elif testnum == 2:
            # Third pass. Try solving for df/dx and substituting that into the
            # ODE. Thanks to Chris Smith for suggesting this method.  Many of
            # the comments below are his, too.
            # The method:
            # - Take each of 1..n derivatives of the solution.
            # - Solve each nth derivative for d^(n)f/dx^(n)
            #   (the differential of that order)
            # - Back substitute into the ODE in decreasing order
            #   (i.e., n, n-1, ...)
            # - Check the result for zero equivalence
            if sol.lhs == func and not sol.rhs.has(func):
                diffsols = {0: sol.rhs}
            elif sol.rhs == func and not sol.lhs.has(func):
                diffsols = {0: sol.lhs}
            else:
                diffsols = {}
            sol = sol.lhs - sol.rhs
            for i in range(1, order + 1):
                # Differentiation is a linear operator, so there should always
                # be 1 solution. Nonetheless, we test just to make sure.
                # We only need to solve once.  After that, we automatically
                # have the solution to the differential in the order we want.
                if i == 1:
                    ds = sol.diff(x)
                    try:
                        sdf = solve(ds, func.diff(x, i))
                        if not sdf:
                            raise NotImplementedError
                    except NotImplementedError:
                        testnum += 1
                        break
                    else:
                        diffsols[i] = sdf[0]
                else:
                    # This is what the solution says df/dx should be.
                    diffsols[i] = diffsols[i - 1].diff(x)

            # Make sure the above didn't fail.
            if testnum > 2:
                continue
            else:
                # Substitute it into ODE to check for self consistency.
                lhs, rhs = ode.lhs, ode.rhs
                for i in range(order, -1, -1):
                    if i == 0 and 0 not in diffsols:
                        # We can only substitute f(x) if the solution was
                        # solved for f(x).
                        break
                    lhs = sub_func_doit(lhs, func.diff(x, i), diffsols[i])
                    rhs = sub_func_doit(rhs, func.diff(x, i), diffsols[i])
                    ode_or_bool = Eq(lhs, rhs)
                    ode_or_bool = simplify(ode_or_bool)

                    if isinstance(ode_or_bool, (bool, BooleanAtom)):
                        if ode_or_bool:
                            lhs = rhs = S.Zero
                    else:
                        lhs = ode_or_bool.lhs
                        rhs = ode_or_bool.rhs
                # No sense in overworking simplify -- just prove that the
                # numerator goes to zero
                num = trigsimp((lhs - rhs).as_numer_denom()[0])
                # since solutions are obtained using force=True we test
                # using the same level of assumptions
                ## replace function with dummy so assumptions will work
                _func = Dummy('func')
                num = num.subs(func, _func)
                ## posify the expression
                num, reps = posify(num)
                s = simplify(num).xreplace(reps).xreplace({_func: func})
                testnum += 1
        else:
            break

    if not s:
        return (True, s)
    elif s is True:  # The code above never was able to change s
        raise NotImplementedError("Unable to test if " + str(sol) +
            " is a solution to " + str(ode) + ".")
    else:
        return (False, s)


def checksysodesol(eqs, sols, func=None):
    r"""
    Substitutes corresponding ``sols`` for each functions into each ``eqs`` and
    checks that the result of substitutions for each equation is ``0``. The
    equations and solutions passed can be any iterable.

    This only works when each ``sols`` have one function only, like `x(t)` or `y(t)`.
    For each function, ``sols`` can have a single solution or a list of solutions.
    In most cases it will not be necessary to explicitly identify the function,
    but if the function cannot be inferred from the original equation it
    can be supplied through the ``func`` argument.

    When a sequence of equations is passed, the same sequence is used to return
    the result for each equation with each function substituted with corresponding
    solutions.

    It tries the following method to find zero equivalence for each equation:

    Substitute the solutions for functions, like `x(t)` and `y(t)` into the
    original equations containing those functions.
    This function returns a tuple.  The first item in the tuple is ``True`` if
    the substitution results for each equation is ``0``, and ``False`` otherwise.
    The second item in the tuple is what the substitution results in.  Each element
    of the ``list`` should always be ``0`` corresponding to each equation if the
    first item is ``True``. Note that sometimes this function may return ``False``,
    but with an expression that is identically equal to ``0``, instead of returning
    ``True``.  This is because :py:meth:`~sympy.simplify.simplify.simplify` cannot
    reduce the expression to ``0``.  If an expression returned by each function
    vanishes identically, then ``sols`` really is a solution to ``eqs``.

    If this function seems to hang, it is probably because of a difficult simplification.

    Examples
    ========

    >>> from sympy import Eq, diff, symbols, sin, cos, exp, sqrt, S, Function
    >>> from sympy.solvers.ode.subscheck import checksysodesol
    >>> C1, C2 = symbols('C1:3')
    >>> t = symbols('t')
    >>> x, y = symbols('x, y', cls=Function)
    >>> eq = (Eq(diff(x(t),t), x(t) + y(t) + 17), Eq(diff(y(t),t), -2*x(t) + y(t) + 12))
    >>> sol = [Eq(x(t), (C1*sin(sqrt(2)*t) + C2*cos(sqrt(2)*t))*exp(t) - S(5)/3),
    ... Eq(y(t), (sqrt(2)*C1*cos(sqrt(2)*t) - sqrt(2)*C2*sin(sqrt(2)*t))*exp(t) - S(46)/3)]
    >>> checksysodesol(eq, sol)
    (True, [0, 0])
    >>> eq = (Eq(diff(x(t),t),x(t)*y(t)**4), Eq(diff(y(t),t),y(t)**3))
    >>> sol = [Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), -sqrt(2)*sqrt(-1/(C2 + t))/2),
    ... Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), sqrt(2)*sqrt(-1/(C2 + t))/2)]
    >>> checksysodesol(eq, sol)
    (True, [0, 0])

    """
    def _sympify(eq):
        return list(map(sympify, eq if iterable(eq) else [eq]))
    eqs = _sympify(eqs)
    for i in range(len(eqs)):
        if isinstance(eqs[i], Equality):
            eqs[i] = eqs[i].lhs - eqs[i].rhs
    if func is None:
        funcs = []
        for eq in eqs:
            derivs = eq.atoms(Derivative)
            func = set().union(*[d.atoms(AppliedUndef) for d in derivs])
            funcs.extend(func)
        funcs = list(set(funcs))
    if not all(isinstance(func, AppliedUndef) and len(func.args) == 1 for func in funcs)\
    and len({func.args for func in funcs})!=1:
        raise ValueError("func must be a function of one variable, not %s" % func)
    for sol in sols:
        if len(sol.atoms(AppliedUndef)) != 1:
            raise ValueError("solutions should have one function only")
    if len(funcs) != len({sol.lhs for sol in sols}):
        raise ValueError("number of solutions provided does not match the number of equations")
    dictsol = {}
    for sol in sols:
        func = list(sol.atoms(AppliedUndef))[0]
        if sol.rhs == func:
            sol = sol.reversed
        solved = sol.lhs == func and not sol.rhs.has(func)
        if not solved:
            rhs = solve(sol, func)
            if not rhs:
                raise NotImplementedError
        else:
            rhs = sol.rhs
        dictsol[func] = rhs
    checkeq = []
    for eq in eqs:
        for func in funcs:
            eq = sub_func_doit(eq, func, dictsol[func])
        ss = simplify(eq)
        if ss != 0:
            eq = ss.expand(force=True)
            if eq != 0:
                eq = sqrtdenest(eq).simplify()
        else:
            eq = 0
        checkeq.append(eq)
    if len(set(checkeq)) == 1 and list(set(checkeq))[0] == 0:
        return (True, checkeq)
    else:
        return (False, checkeq)
<|endoftext|>from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
                                 AppliedUndef, Function, Subs)
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
                             piecewise_fold, sqrt, log)
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
                                       strongly_connected_components)
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate


def _get_func_order(eqs, funcs):
    return {func: max(ode_order(eq, func) for eq in eqs) for func in funcs}


class ODEOrderError(ValueError):
    """Raised by linear_ode_to_matrix if the system has the wrong order"""
    pass


class ODENonlinearError(NonlinearError):
    """Raised by linear_ode_to_matrix if the system is nonlinear"""
    pass


def _simpsol(soleq):
    lhs = soleq.lhs
    sol = soleq.rhs
    sol = powsimp(sol)
    gens = list(sol.atoms(exp))
    p = Poly(sol, *gens, expand=False)
    gens = [factor_terms(g) for g in gens]
    if not gens:
        gens = p.gens
    syms = [Symbol('C1'), Symbol('C2')]
    terms = []
    for coeff, monom in zip(p.coeffs(), p.monoms()):
        coeff = piecewise_fold(coeff)
        if isinstance(coeff, Piecewise):
            coeff = Piecewise(*((ratsimp(coef).collect(syms), cond) for coef, cond in coeff.args))
        else:
            coeff = ratsimp(coeff).collect(syms)
        monom = Mul(*(g ** i for g, i in zip(gens, monom)))
        terms.append(coeff * monom)
    return Eq(lhs, Add(*terms))


def _solsimp(e, t):
    no_t, has_t = powsimp(expand_mul(e)).as_independent(t)

    no_t = ratsimp(no_t)
    has_t = has_t.replace(exp, lambda a: exp(factor_terms(a)))

    return no_t + has_t


def simpsol(sol, wrt1, wrt2, doit=True):
    """Simplify solutions from dsolve_system."""

    # The parameter sol is the solution as returned by dsolve (list of Eq).
    #
    # The parameters wrt1 and wrt2 are lists of symbols to be collected for
    # with those in wrt1 being collected for first. This allows for collecting
    # on any factors involving the independent variable before collecting on
    # the integration constants or vice versa using e.g.:
    #
    #     sol = simpsol(sol, [t], [C1, C2])  # t first, constants after
    #     sol = simpsol(sol, [C1, C2], [t])  # constants first, t after
    #
    # If doit=True (default) then simpsol will begin by evaluating any
    # unevaluated integrals. Since many integrals will appear multiple times
    # in the solutions this is done intelligently by computing each integral
    # only once.
    #
    # The strategy is to first perform simple cancellation with factor_terms
    # and then multiply out all brackets with expand_mul. This gives an Add
    # with many terms.
    #
    # We split each term into two multiplicative factors dep and coeff where
    # all factors that involve wrt1 are in dep and any constant factors are in
    # coeff e.g.
    #         sqrt(2)*C1*exp(t) -> ( exp(t), sqrt(2)*C1 )
    #
    # The dep factors are simplified using powsimp to combine expanded
    # exponential factors e.g.
    #              exp(a*t)*exp(b*t) -> exp(t*(a+b))
    #
    # We then collect coefficients for all terms having the same (simplified)
    # dep. The coefficients are then simplified using together and ratsimp and
    # lastly by recursively applying the same transformation to the
    # coefficients to collect on wrt2.
    #
    # Finally the result is recombined into an Add and signsimp is used to
    # normalise any minus signs.

    def simprhs(rhs, rep, wrt1, wrt2):
        """Simplify the rhs of an ODE solution"""
        if rep:
            rhs = rhs.subs(rep)
        rhs = factor_terms(rhs)
        rhs = simp_coeff_dep(rhs, wrt1, wrt2)
        rhs = signsimp(rhs)
        return rhs

    def simp_coeff_dep(expr, wrt1, wrt2=None):
        """Split rhs into terms, split terms into dep and coeff and collect on dep"""
        add_dep_terms = lambda e: e.is_Add and e.has(*wrt1)
        expandable = lambda e: e.is_Mul and any(map(add_dep_terms, e.args))
        expand_func = lambda e: expand_mul(e, deep=False)
        expand_mul_mod = lambda e: e.replace(expandable, expand_func)
        terms = Add.make_args(expand_mul_mod(expr))
        dc = {}
        for term in terms:
            coeff, dep = term.as_independent(*wrt1, as_Add=False)
            # Collect together the coefficients for terms that have the same
            # dependence on wrt1 (after dep is normalised using simpdep).
            dep = simpdep(dep, wrt1)

            # See if the dependence on t cancels out...
            if dep is not S.One:
                dep2 = factor_terms(dep)
                if not dep2.has(*wrt1):
                    coeff *= dep2
                    dep = S.One

            if dep not in dc:
                dc[dep] = coeff
            else:
                dc[dep] += coeff
        # Apply the method recursively to the coefficients but this time
        # collecting on wrt2 rather than wrt2.
        termpairs = ((simpcoeff(c, wrt2), d) for d, c in dc.items())
        if wrt2 is not None:
            termpairs = ((simp_coeff_dep(c, wrt2), d) for c, d in termpairs)
        return Add(*(c * d for c, d in termpairs))

    def simpdep(term, wrt1):
        """Normalise factors involving t with powsimp and recombine exp"""
        def canonicalise(a):
            # Using factor_terms here isn't quite right because it leads to things
            # like exp(t*(1+t)) that we don't want. We do want to cancel factors
            # and pull out a common denominator but ideally the numerator would be
            # expressed as a standard form polynomial in t so we expand_mul
            # and collect afterwards.
            a = factor_terms(a)
            num, den = a.as_numer_denom()
            num = expand_mul(num)
            num = collect(num, wrt1)
            return num / den

        term = powsimp(term)
        rep = {e: exp(canonicalise(e.args[0])) for e in term.atoms(exp)}
        term = term.subs(rep)
        return term

    def simpcoeff(coeff, wrt2):
        """Bring to a common fraction and cancel with ratsimp"""
        coeff = together(coeff)
        if coeff.is_polynomial():
            # Calling ratsimp can be expensive. The main reason is to simplify
            # sums of terms with irrational denominators so we limit ourselves
            # to the case where the expression is polynomial in any symbols.
            # Maybe there's a better approach...
            coeff = ratsimp(radsimp(coeff))
        # collect on secondary variables first and any remaining symbols after
        if wrt2 is not None:
            syms = list(wrt2) + list(ordered(coeff.free_symbols - set(wrt2)))
        else:
            syms = list(ordered(coeff.free_symbols))
        coeff = collect(coeff, syms)
        coeff = together(coeff)
        return coeff

    # There are often repeated integrals. Collect unique integrals and
    # evaluate each once and then substitute into the final result to replace
    # all occurrences in each of the solution equations.
    if doit:
        integrals = set().union(*(s.atoms(Integral) for s in sol))
        rep = {i: factor_terms(i).doit() for i in integrals}
    else:
        rep = {}

    sol = [Eq(s.lhs, simprhs(s.rhs, rep, wrt1, wrt2)) for s in sol]
    return sol


def linodesolve_type(A, t, b=None):
    r"""
    Helper function that determines the type of the system of ODEs for solving with :obj:`sympy.solvers.ode.systems.linodesolve()`

    Explanation
    ===========

    This function takes in the coefficient matrix and/or the non-homogeneous term
    and returns the type of the equation that can be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`.

    If the system is constant coefficient homogeneous, then "type1" is returned

    If the system is constant coefficient non-homogeneous, then "type2" is returned

    If the system is non-constant coefficient homogeneous, then "type3" is returned

    If the system is non-constant coefficient non-homogeneous, then "type4" is returned

    If the system has a non-constant coefficient matrix which can be factorized into constant
    coefficient matrix, then "type5" or "type6" is returned for when the system is homogeneous or
    non-homogeneous respectively.

    Note that, if the system of ODEs is of "type3" or "type4", then along with the type,
    the commutative antiderivative of the coefficient matrix is also returned.

    If the system cannot be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`, then
    NotImplementedError is raised.

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of ODEs
    b : Matrix or None
        Non-homogeneous term of the system. The default value is None.
        If this argument is None, then the system is assumed to be homogeneous.

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import linodesolve_type
    >>> t = symbols("t")
    >>> A = Matrix([[1, 1], [2, 3]])
    >>> b = Matrix([t, 1])

    >>> linodesolve_type(A, t)
    {'antiderivative': None, 'type_of_equation': 'type1'}

    >>> linodesolve_type(A, t, b=b)
    {'antiderivative': None, 'type_of_equation': 'type2'}

    >>> A_t = Matrix([[1, t], [-t, 1]])

    >>> linodesolve_type(A_t, t)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type3'}

    >>> linodesolve_type(A_t, t, b=b)
    {'antiderivative': Matrix([
    [      t, t**2/2],
    [-t**2/2,      t]]), 'type_of_equation': 'type4'}

    >>> A_non_commutative = Matrix([[1, t], [t, -1]])
    >>> linodesolve_type(A_non_commutative, t)
    Traceback (most recent call last):
    ...
    NotImplementedError:
    The system does not have a commutative antiderivative, it cannot be
    solved by linodesolve.

    Returns
    =======

    Dict

    Raises
    ======

    NotImplementedError
        When the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linodesolve: Function for which linodesolve_type gets the information

    """

    match = {}
    is_non_constant = not _matrix_is_constant(A, t)
    is_non_homogeneous = not (b is None or b.is_zero_matrix)
    type = "type{}".format(int("{}{}".format(int(is_non_constant), int(is_non_homogeneous)), 2) + 1)

    B = None
    match.update({"type_of_equation": type, "antiderivative": B})

    if is_non_constant:
        B, is_commuting = _is_commutative_anti_derivative(A, t)
        if not is_commuting:
            raise NotImplementedError(filldedent('''
                The system does not have a commutative antiderivative, it cannot be solved
                by linodesolve.
            '''))

        match['antiderivative'] = B
        match.update(_first_order_type5_6_subs(A, t, b=b))

    return match


def _first_order_type5_6_subs(A, t, b=None):
    match = {}

    factor_terms = _factor_matrix(A, t)
    is_homogeneous = b is None or b.is_zero_matrix

    if factor_terms is not None:
        t_ = Symbol("{}_".format(t))
        F_t = integrate(factor_terms[0], t)
        inverse = solveset(Eq(t_, F_t), t)

        # Note: A simple way to check if a function is invertible
        # or not.
        if isinstance(inverse, FiniteSet) and not inverse.has(Piecewise)\
            and len(inverse) == 1:

            A = factor_terms[1]
            if not is_homogeneous:
                b = b / factor_terms[0]
                b = b.subs(t, list(inverse)[0])
            type = "type{}".format(5 + (not is_homogeneous))
            match.update({'func_coeff': A, 'tau': F_t,
                          't_': t_, 'type_of_equation': type, 'rhs': b})

    return match


def linear_ode_to_matrix(eqs, funcs, t, order):
    r"""
    Convert a linear system of ODEs to matrix form

    Explanation
    ===========

    Express a system of linear ordinary differential equations as a single
    matrix differential equation [1]. For example the system $x' = x + y + 1$
    and $y' = x - y$ can be represented as

    .. math:: A_1 X' = A_0 X + b

    where $A_1$ and $A_0$ are $2 \times 2$ matrices and $b$, $X$ and $X'$ are
    $2 \times 1$ matrices with $X = [x, y]^T$.

    Higher-order systems are represented with additional matrices e.g. a
    second-order system would look like

    .. math:: A_2 X'' =  A_1 X' + A_0 X  + b

    Examples
    ========

    >>> from sympy import Function, Symbol, Matrix, Eq
    >>> from sympy.solvers.ode.systems import linear_ode_to_matrix
    >>> t = Symbol('t')
    >>> x = Function('x')
    >>> y = Function('y')

    We can create a system of linear ODEs like

    >>> eqs = [
    ...     Eq(x(t).diff(t), x(t) + y(t) + 1),
    ...     Eq(y(t).diff(t), x(t) - y(t)),
    ... ]
    >>> funcs = [x(t), y(t)]
    >>> order = 1 # 1st order system

    Now ``linear_ode_to_matrix`` can represent this as a matrix
    differential equation.

    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, order)
    >>> A1
    Matrix([
    [1, 0],
    [0, 1]])
    >>> A0
    Matrix([
    [1, 1],
    [1,  -1]])
    >>> b
    Matrix([
    [1],
    [0]])

    The original equations can be recovered from these matrices:

    >>> eqs_mat = Matrix([eq.lhs - eq.rhs for eq in eqs])
    >>> X = Matrix(funcs)
    >>> A1 * X.diff(t) - A0 * X - b == eqs_mat
    True

    If the system of equations has a maximum order greater than the
    order of the system specified, a ODEOrderError exception is raised.

    >>> eqs = [Eq(x(t).diff(t, 2), x(t).diff(t) + x(t)), Eq(y(t).diff(t), y(t) + x(t))]
    >>> linear_ode_to_matrix(eqs, funcs, t, 1)
    Traceback (most recent call last):
    ...
    ODEOrderError: Cannot represent system in 1-order form

    If the system of equations is nonlinear, then ODENonlinearError is
    raised.

    >>> eqs = [Eq(x(t).diff(t), x(t) + y(t)), Eq(y(t).diff(t), y(t)**2 + x(t))]
    >>> linear_ode_to_matrix(eqs, funcs, t, 1)
    Traceback (most recent call last):
    ...
    ODENonlinearError: The system of ODEs is nonlinear.

    Parameters
    ==========

    eqs : list of SymPy expressions or equalities
        The equations as expressions (assumed equal to zero).
    funcs : list of applied functions
        The dependent variables of the system of ODEs.
    t : symbol
        The independent variable.
    order : int
        The order of the system of ODEs.

    Returns
    =======

    The tuple ``(As, b)`` where ``As`` is a tuple of matrices and ``b`` is the
    the matrix representing the rhs of the matrix equation.

    Raises
    ======

    ODEOrderError
        When the system of ODEs have an order greater than what was specified
    ODENonlinearError
        When the system of ODEs is nonlinear

    See Also
    ========

    linear_eq_to_matrix: for systems of linear algebraic equations.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_differential_equation

    """
    from sympy.solvers.solveset import linear_eq_to_matrix

    if any(ode_order(eq, func) > order for eq in eqs for func in funcs):
        msg = "Cannot represent system in {}-order form"
        raise ODEOrderError(msg.format(order))

    As = []

    for o in range(order, -1, -1):
        # Work from the highest derivative down
        syms = [func.diff(t, o) for func in funcs]

        # Ai is the matrix for X(t).diff(t, o)
        # eqs is minus the remainder of the equations.
        try:
            Ai, b = linear_eq_to_matrix(eqs, syms)
        except NonlinearError:
            raise ODENonlinearError("The system of ODEs is nonlinear.")

        Ai = Ai.applyfunc(expand_mul)

        As.append(Ai if o == order else -Ai)

        if o:
            eqs = [-eq for eq in b]
        else:
            rhs = b

    return As, rhs


def matrix_exp(A, t):
    r"""
    Matrix exponential $\exp(A*t)$ for the matrix ``A`` and scalar ``t``.

    Explanation
    ===========

    This functions returns the $\exp(A*t)$ by doing a simple
    matrix multiplication:

    .. math:: \exp(A*t) = P * expJ * P^{-1}

    where $expJ$ is $\exp(J*t)$. $J$ is the Jordan normal
    form of $A$ and $P$ is matrix such that:

    .. math:: A = P * J * P^{-1}

    The matrix exponential $\exp(A*t)$ appears in the solution of linear
    differential equations. For example if $x$ is a vector and $A$ is a matrix
    then the initial value problem

    .. math:: \frac{dx(t)}{dt} = A \times x(t),   x(0) = x0

    has the unique solution

    .. math:: x(t) = \exp(A t) x0

    Examples
    ========

    >>> from sympy import Symbol, Matrix, pprint
    >>> from sympy.solvers.ode.systems import matrix_exp
    >>> t = Symbol('t')

    We will consider a 2x2 matrix for comupting the exponential

    >>> A = Matrix([[2, -5], [2, -4]])
    >>> pprint(A)
    [2  -5]
    [     ]
    [2  -4]

    Now, exp(A*t) is given as follows:

    >>> pprint(matrix_exp(A, t))
    [   -t           -t                    -t              ]
    [3*e  *sin(t) + e  *cos(t)         -5*e  *sin(t)       ]
    [                                                      ]
    [         -t                     -t           -t       ]
    [      2*e  *sin(t)         - 3*e  *sin(t) + e  *cos(t)]

    Parameters
    ==========

    A : Matrix
        The matrix $A$ in the expression $\exp(A*t)$
    t : Symbol
        The independent variable

    See Also
    ========

    matrix_exp_jordan_form: For exponential of Jordan normal form

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jordan_normal_form
    .. [2] https://en.wikipedia.org/wiki/Matrix_exponential

    """
    P, expJ = matrix_exp_jordan_form(A, t)
    return P * expJ * P.inv()


def matrix_exp_jordan_form(A, t):
    r"""
    Matrix exponential $\exp(A*t)$ for the matrix *A* and scalar *t*.

    Explanation
    ===========

    Returns the Jordan form of the $\exp(A*t)$ along with the matrix $P$ such that:

    .. math::
        \exp(A*t) = P * expJ * P^{-1}

    Examples
    ========

    >>> from sympy import Matrix, Symbol
    >>> from sympy.solvers.ode.systems import matrix_exp, matrix_exp_jordan_form
    >>> t = Symbol('t')

    We will consider a 2x2 defective matrix. This shows that our method
    works even for defective matrices.

    >>> A = Matrix([[1, 1], [0, 1]])

    It can be observed that this function gives us the Jordan normal form
    and the required invertible matrix P.

    >>> P, expJ = matrix_exp_jordan_form(A, t)

    Here, it is shown that P and expJ returned by this function is correct
    as they satisfy the formula: P * expJ * P_inverse = exp(A*t).

    >>> P * expJ * P.inv() == matrix_exp(A, t)
    True

    Parameters
    ==========

    A : Matrix
        The matrix $A$ in the expression $\exp(A*t)$
    t : Symbol
        The independent variable

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Defective_matrix
    .. [2] https://en.wikipedia.org/wiki/Jordan_matrix
    .. [3] https://en.wikipedia.org/wiki/Jordan_normal_form

    """

    N, M = A.shape
    if N != M:
        raise ValueError('Needed square matrix but got shape (%s, %s)' % (N, M))
    elif A.has(t):
        raise ValueError('Matrix A should not depend on t')

    def jordan_chains(A):
        '''Chains from Jordan normal form analogous to M.eigenvects().
        Returns a dict with eignevalues as keys like:
            {e1: [[v111,v112,...], [v121, v122,...]], e2:...}
        where vijk is the kth vector in the jth chain for eigenvalue i.
        '''
        P, blocks = A.jordan_cells()
        basis = [P[:,i] for i in range(P.shape[1])]
        n = 0
        chains = {}
        for b in blocks:
            eigval = b[0, 0]
            size = b.shape[0]
            if eigval not in chains:
                chains[eigval] = []
            chains[eigval].append(basis[n:n+size])
            n += size
        return chains

    eigenchains = jordan_chains(A)

    # Needed for consistency across Python versions
    eigenchains_iter = sorted(eigenchains.items(), key=default_sort_key)
    isreal = not A.has(I)

    blocks = []
    vectors = []
    seen_conjugate = set()
    for e, chains in eigenchains_iter:
        for chain in chains:
            n = len(chain)
            if isreal and e != e.conjugate() and e.conjugate() in eigenchains:
                if e in seen_conjugate:
                    continue
                seen_conjugate.add(e.conjugate())
                exprt = exp(re(e) * t)
                imrt = im(e) * t
                imblock = Matrix([[cos(imrt), sin(imrt)],
                                  [-sin(imrt), cos(imrt)]])
                expJblock2 = Matrix(n, n, lambda i,j:
                        imblock * t**(j-i) / factorial(j-i) if j >= i
                        else zeros(2, 2))
                expJblock = Matrix(2*n, 2*n, lambda i,j: expJblock2[i//2,j//2][i%2,j%2])

                blocks.append(exprt * expJblock)
                for i in range(n):
                    vectors.append(re(chain[i]))
                    vectors.append(im(chain[i]))
            else:
                vectors.extend(chain)
                fun = lambda i,j: t**(j-i)/factorial(j-i) if j >= i else 0
                expJblock = Matrix(n, n, fun)
                blocks.append(exp(e * t) * expJblock)

    expJ = Matrix.diag(*blocks)
    P = Matrix(N, N, lambda i,j: vectors[j][i])

    return P, expJ


# Note: To add a docstring example with tau
def linodesolve(A, t, b=None, B=None, type="auto", doit=False,
                tau=None):
    r"""
    System of n equations linear first-order differential equations

    Explanation
    ===========

    This solver solves the system of ODEs of the following form:

    .. math::
        X'(t) = A(t) X(t) +  b(t)

    Here, $A(t)$ is the coefficient matrix, $X(t)$ is the vector of n independent variables,
    $b(t)$ is the non-homogeneous term and $X'(t)$ is the derivative of $X(t)$

    Depending on the properties of $A(t)$ and $b(t)$, this solver evaluates the solution
    differently.

    When $A(t)$ is constant coefficient matrix and $b(t)$ is zero vector i.e. system is homogeneous,
    the system is "type1". The solution is:

    .. math::
        X(t) = \exp(A t) C

    Here, $C$ is a vector of constants and $A$ is the constant coefficient matrix.

    When $A(t)$ is constant coefficient matrix and $b(t)$ is non-zero i.e. system is non-homogeneous,
    the system is "type2". The solution is:

    .. math::
        X(t) = e^{A t} ( \int e^{- A t} b \,dt + C)

    When $A(t)$ is coefficient matrix such that its commutative with its antiderivative $B(t)$ and
    $b(t)$ is a zero vector i.e. system is homogeneous, the system is "type3". The solution is:

    .. math::
        X(t) = \exp(B(t)) C

    When $A(t)$ is commutative with its antiderivative $B(t)$ and $b(t)$ is non-zero i.e. system is
    non-homogeneous, the system is "type4". The solution is:

    .. math::
        X(t) =  e^{B(t)} ( \int e^{-B(t)} b(t) \,dt + C)

    When $A(t)$ is a coefficient matrix such that it can be factorized into a scalar and a constant
    coefficient matrix:

    .. math::
        A(t) = f(t) * A

    Where $f(t)$ is a scalar expression in the independent variable $t$ and $A$ is a constant matrix,
    then we can do the following substitutions:

    .. math::
        tau = \int f(t) dt, X(t) = Y(tau), b(t) = b(f^{-1}(tau))

    Here, the substitution for the non-homogeneous term is done only when its non-zero.
    Using these substitutions, our original system becomes:

    .. math::
        Y'(tau) = A * Y(tau) + b(tau)/f(tau)

    The above system can be easily solved using the solution for "type1" or "type2" depending
    on the homogeneity of the system. After we get the solution for $Y(tau)$, we substitute the
    solution for $tau$ as $t$ to get back $X(t)$

    .. math::
        X(t) = Y(tau)

    Systems of "type5" and "type6" have a commutative antiderivative but we use this solution
    because its faster to compute.

    The final solution is the general solution for all the four equations since a constant coefficient
    matrix is always commutative with its antidervative.

    An additional feature of this function is, if someone wants to substitute for value of the independent
    variable, they can pass the substitution `tau` and the solution will have the independent variable
    substituted with the passed expression(`tau`).

    Parameters
    ==========

    A : Matrix
        Coefficient matrix of the system of linear first order ODEs.
    t : Symbol
        Independent variable in the system of ODEs.
    b : Matrix or None
        Non-homogeneous term in the system of ODEs. If None is passed,
        a homogeneous system of ODEs is assumed.
    B : Matrix or None
        Antiderivative of the coefficient matrix. If the antiderivative
        is not passed and the solution requires the term, then the solver
        would compute it internally.
    type : String
        Type of the system of ODEs passed. Depending on the type, the
        solution is evaluated. The type values allowed and the corresponding
        system it solves are: "type1" for constant coefficient homogeneous
        "type2" for constant coefficient non-homogeneous, "type3" for non-constant
        coefficient homogeneous, "type4" for non-constant coefficient non-homogeneous,
        "type5" and "type6" for non-constant coefficient homogeneous and non-homogeneous
        systems respectively where the coefficient matrix can be factorized to a constant
        coefficient matrix.
        The default value is "auto" which will let the solver decide the correct type of
        the system passed.
    doit : Boolean
        Evaluate the solution if True, default value is False
    tau: Expression
        Used to substitute for the value of `t` after we get the solution of the system.

    Examples
    ========

    To solve the system of ODEs using this function directly, several things must be
    done in the right order. Wrong inputs to the function will lead to incorrect results.

    >>> from sympy import symbols, Function, Eq
    >>> from sympy.solvers.ode.systems import canonical_odes, linear_ode_to_matrix, linodesolve, linodesolve_type
    >>> from sympy.solvers.ode.subscheck import checkodesol
    >>> f, g = symbols("f, g", cls=Function)
    >>> x, a = symbols("x, a")
    >>> funcs = [f(x), g(x)]
    >>> eqs = [Eq(f(x).diff(x) - f(x), a*g(x) + 1), Eq(g(x).diff(x) + g(x), a*f(x))]

    Here, it is important to note that before we derive the coefficient matrix, it is
    important to get the system of ODEs into the desired form. For that we will use
    :obj:`sympy.solvers.ode.systems.canonical_odes()`.

    >>> eqs = canonical_odes(eqs, funcs, x)
    >>> eqs
    [[Eq(Derivative(f(x), x), a*g(x) + f(x) + 1), Eq(Derivative(g(x), x), a*f(x) - g(x))]]

    Now, we will use :obj:`sympy.solvers.ode.systems.linear_ode_to_matrix()` to get the coefficient matrix and the
    non-homogeneous term if it is there.

    >>> eqs = eqs[0]
    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)
    >>> A = A0

    We have the coefficient matrices and the non-homogeneous term ready. Now, we can use
    :obj:`sympy.solvers.ode.systems.linodesolve_type()` to get the information for the system of ODEs
    to finally pass it to the solver.

    >>> system_info = linodesolve_type(A, x, b=b)
    >>> sol_vector = linodesolve(A, x, b=b, B=system_info['antiderivative'], type=system_info['type_of_equation'])

    Now, we can prove if the solution is correct or not by using :obj:`sympy.solvers.ode.checkodesol()`

    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    We can also use the doit method to evaluate the solutions passed by the function.

    >>> sol_vector_evaluated = linodesolve(A, x, b=b, type="type2", doit=True)

    Now, we will look at a system of ODEs which is non-constant.

    >>> eqs = [Eq(f(x).diff(x), f(x) + x*g(x)), Eq(g(x).diff(x), -x*f(x) + g(x))]

    The system defined above is already in the desired form, so we do not have to convert it.

    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)
    >>> A = A0

    A user can also pass the commutative antiderivative required for type3 and type4 system of ODEs.
    Passing an incorrect one will lead to incorrect results. If the coefficient matrix is not commutative
    with its antiderivative, then :obj:`sympy.solvers.ode.systems.linodesolve_type()` raises a NotImplementedError.
    If it does have a commutative antiderivative, then the function just returns the information about the system.

    >>> system_info = linodesolve_type(A, x, b=b)

    Now, we can pass the antiderivative as an argument to get the solution. If the system information is not
    passed, then the solver will compute the required arguments internally.

    >>> sol_vector = linodesolve(A, x, b=b)

    Once again, we can verify the solution obtained.

    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    Returns
    =======

    List

    Raises
    ======

    ValueError
        This error is raised when the coefficient matrix, non-homogeneous term
        or the antiderivative, if passed, are not a matrix or
        do not have correct dimensions
    NonSquareMatrixError
        When the coefficient matrix or its antiderivative, if passed is not a
        square matrix
    NotImplementedError
        If the coefficient matrix does not have a commutative antiderivative

    See Also
    ========

    linear_ode_to_matrix: Coefficient matrix computation function
    canonical_odes: System of ODEs representation change
    linodesolve_type: Getting information about systems of ODEs to pass in this solver

    """

    if not isinstance(A, MatrixBase):
        raise ValueError(filldedent('''\
            The coefficients of the system of ODEs should be of type Matrix
        '''))

    if not A.is_square:
        raise NonSquareMatrixError(filldedent('''\
            The coefficient matrix must be a square
        '''))

    if b is not None:
        if not isinstance(b, MatrixBase):
            raise ValueError(filldedent('''\
                The non-homogeneous terms of the system of ODEs should be of type Matrix
            '''))

        if A.rows != b.rows:
            raise ValueError(filldedent('''\
                The system of ODEs should have the same number of non-homogeneous terms and the number of
                equations
            '''))

    if B is not None:
        if not isinstance(B, MatrixBase):
            raise ValueError(filldedent('''\
                The antiderivative of coefficients of the system of ODEs should be of type Matrix
            '''))

        if not B.is_square:
            raise NonSquareMatrixError(filldedent('''\
                The antiderivative of the coefficient matrix must be a square
            '''))

        if A.rows != B.rows:
            raise ValueError(filldedent('''\
                        The coefficient matrix and its antiderivative should have same dimensions
                    '''))

    if not any(type == "type{}".format(i) for i in range(1, 7)) and not type == "auto":
        raise ValueError(filldedent('''\
                    The input type should be a valid one
                '''))

    n = A.rows

    # constants = numbered_symbols(prefix='C', cls=Dummy, start=const_idx+1)
    Cvect = Matrix([Dummy() for _ in range(n)])

    if b is None and any(type == typ for typ in ["type2", "type4", "type6"]):
        b = zeros(n, 1)

    is_transformed = tau is not None
    passed_type = type

    if type == "auto":
        system_info = linodesolve_type(A, t, b=b)
        type = system_info["type_of_equation"]
        B = system_info["antiderivative"]

    if type in ("type5", "type6"):
        is_transformed = True
        if passed_type != "auto":
            if tau is None:
                system_info = _first_order_type5_6_subs(A, t, b=b)
                if not system_info:
                    raise ValueError(filldedent('''
                        The system passed isn't {}.
                    '''.format(type)))

                tau = system_info['tau']
                t = system_info['t_']
                A = system_info['A']
                b = system_info['b']

    intx_wrtt = lambda x: Integral(x, t) if x else 0
    if type in ("type1", "type2", "type5", "type6"):
        P, J = matrix_exp_jordan_form(A, t)
        P = simplify(P)

        if type in ("type1", "type5"):
            sol_vector = P * (J * Cvect)
        else:
            Jinv = J.subs(t, -t)
            sol_vector = P * J * ((Jinv * P.inv() * b).applyfunc(intx_wrtt) + Cvect)
    else:
        if B is None:
            B, _ = _is_commutative_anti_derivative(A, t)

        if type == "type3":
            sol_vector = B.exp() * Cvect
        else:
            sol_vector = B.exp() * (((-B).exp() * b).applyfunc(intx_wrtt) + Cvect)

    if is_transformed:
        sol_vector = sol_vector.subs(t, tau)

    gens = sol_vector.atoms(exp)

    if type != "type1":
        sol_vector = [expand_mul(s) for s in sol_vector]

    sol_vector = [collect(s, ordered(gens), exact=True) for s in sol_vector]

    if doit:
        sol_vector = [s.doit() for s in sol_vector]

    return sol_vector


def _matrix_is_constant(M, t):
    """Checks if the matrix M is independent of t or not."""
    return all(coef.as_independent(t, as_Add=True)[1] == 0 for coef in M)


def canonical_odes(eqs, funcs, t):
    r"""
    Function that solves for highest order derivatives in a system

    Explanation
    ===========

    This function inputs a system of ODEs and based on the system,
    the dependent variables and their highest order, returns the system
    in the following form:

    .. math::
        X'(t) = A(t) X(t) + b(t)

    Here, $X(t)$ is the vector of dependent variables of lower order, $A(t)$ is
    the coefficient matrix, $b(t)$ is the non-homogeneous term and $X'(t)$ is the
    vector of dependent variables in their respective highest order. We use the term
    canonical form to imply the system of ODEs which is of the above form.

    If the system passed has a non-linear term with multiple solutions, then a list of
    systems is returned in its canonical form.

    Parameters
    ==========

    eqs : List
        List of the ODEs
    funcs : List
        List of dependent variables
    t : Symbol
        Independent variable

    Examples
    ========

    >>> from sympy import symbols, Function, Eq, Derivative
    >>> from sympy.solvers.ode.systems import canonical_odes
    >>> f, g = symbols("f g", cls=Function)
    >>> x, y = symbols("x y")
    >>> funcs = [f(x), g(x)]
    >>> eqs = [Eq(f(x).diff(x) - 7*f(x), 12*g(x)), Eq(g(x).diff(x) + g(x), 20*f(x))]

    >>> canonical_eqs = canonical_odes(eqs, funcs, x)
    >>> canonical_eqs
    [[Eq(Derivative(f(x), x), 7*f(x) + 12*g(x)), Eq(Derivative(g(x), x), 20*f(x) - g(x))]]

    >>> system = [Eq(Derivative(f(x), x)**2 - 2*Derivative(f(x), x) + 1, 4), Eq(-y*f(x) + Derivative(g(x), x), 0)]

    >>> canonical_system = canonical_odes(system, funcs, x)
    >>> canonical_system
    [[Eq(Derivative(f(x), x), -1), Eq(Derivative(g(x), x), y*f(x))], [Eq(Derivative(f(x), x), 3), Eq(Derivative(g(x), x), y*f(x))]]

    Returns
    =======

    List

    """
    from sympy.solvers.solvers import solve

    order = _get_func_order(eqs, funcs)

    canon_eqs = solve(eqs, *[func.diff(t, order[func]) for func in funcs], dict=True)

    systems = []
    for eq in canon_eqs:
        system = [Eq(func.diff(t, order[func]), eq[func.diff(t, order[func])]) for func in funcs]
        systems.append(system)

    return systems


def _is_commutative_anti_derivative(A, t):
    r"""
    Helper function for determining if the Matrix passed is commutative with its antiderivative

    Explanation
    ===========

    This function checks if the Matrix $A$ passed is commutative with its antiderivative with respect
    to the independent variable $t$.

    .. math::
        B(t) = \int A(t) dt

    The function outputs two values, first one being the antiderivative $B(t)$, second one being a
    boolean value, if True, then the matrix $A(t)$ passed is commutative with $B(t)$, else the matrix
    passed isn't commutative with $B(t)$.

    Parameters
    ==========

    A : Matrix
        The matrix which has to be checked
    t : Symbol
        Independent variable

    Examples
    ========

    >>> from sympy import symbols, Matrix
    >>> from sympy.solvers.ode.systems import _is_commutative_anti_derivative
    >>> t = symbols("t")
    >>> A = Matrix([[1, t], [-t, 1]])

    >>> B, is_commuting = _is_commutative_anti_derivative(A, t)
    >>> is_commuting
    True

    Returns
    =======

    Matrix, Boolean

    """
    B = integrate(A, t)
    is_commuting = (B*A - A*B).applyfunc(expand).applyfunc(factor_terms).is_zero_matrix

    is_commuting = False if is_commuting is None else is_commuting

    return B, is_commuting


def _factor_matrix(A, t):
    term = None
    for element in A:
        temp_term = element.as_independent(t)[1]
        if temp_term.has(t):
            term = temp_term
            break

    if term is not None:
        A_factored = (A/term).applyfunc(ratsimp)
        can_factor = _matrix_is_constant(A_factored, t)
        term = (term, A_factored) if can_factor else None

    return term


def _is_second_order_type2(A, t):
    term = _factor_matrix(A, t)
    is_type2 = False

    if term is not None:
        term = 1/term[0]
        is_type2 = term.is_polynomial()

    if is_type2:
        poly = Poly(term.expand(), t)
        monoms = poly.monoms()

        if monoms[0][0] in (2, 4):
            cs = _get_poly_coeffs(poly, 4)
            a, b, c, d, e = cs

            a1 = powdenest(sqrt(a), force=True)
            c1 = powdenest(sqrt(e), force=True)
            b1 = powdenest(sqrt(c - 2*a1*c1), force=True)

            is_type2 = (b == 2*a1*b1) and (d == 2*b1*c1)
            term = a1*t**2 + b1*t + c1

        else:
            is_type2 = False

    return is_type2, term


def _get_poly_coeffs(poly, order):
    cs = [0 for _ in range(order+1)]
    for c, m in zip(poly.coeffs(), poly.monoms()):
        cs[-1-m[0]] = c
    return cs


def _match_second_order_type(A1, A0, t, b=None):
    r"""
    Works only for second order system in its canonical form.

    Type 0: Constant coefficient matrix, can be simply solved by
            introducing dummy variables.
    Type 1: When the substitution: $U = t*X' - X$ works for reducing
            the second order system to first order system.
    Type 2: When the system is of the form: $poly * X'' = A*X$ where
            $poly$ is square of a quadratic polynomial with respect to
            *t* and $A$ is a constant coefficient matrix.

    """
    match = {"type_of_equation": "type0"}
    n = A1.shape[0]

    if _matrix_is_constant(A1, t) and _matrix_is_constant(A0, t):
        return match

    if (A1 + A0*t).applyfunc(expand_mul).is_zero_matrix:
        match.update({"type_of_equation": "type1", "A1": A1})

    elif A1.is_zero_matrix and (b is None or b.is_zero_matrix):
        is_type2, term = _is_second_order_type2(A0, t)
        if is_type2:
            a, b, c = _get_poly_coeffs(Poly(term, t), 2)
            A = (A0*(term**2).expand()).applyfunc(ratsimp) + (b**2/4 - a*c)*eye(n, n)
            tau = integrate(1/term, t)
            t_ = Symbol("{}_".format(t))
            match.update({"type_of_equation": "type2", "A0": A,
                          "g(t)": sqrt(term), "tau": tau, "is_transformed": True,
                          "t_": t_})

    return match


def _second_order_subs_type1(A, b, funcs, t):
    r"""
    For a linear, second order system of ODEs, a particular substitution.

    A system of the below form can be reduced to a linear first order system of
    ODEs:
    .. math::
        X'' = A(t) * (t*X' - X) + b(t)

    By substituting:
    .. math::  U = t*X' - X

    To get the system:
    .. math::  U' = t*(A(t)*U + b(t))

    Where $U$ is the vector of dependent variables, $X$ is the vector of dependent
    variables in `funcs` and $X'$ is the first order derivative of $X$ with respect to
    $t$. It may or may not reduce the system into linear first order system of ODEs.

    Then a check is made to determine if the system passed can be reduced or not, if
    this substitution works, then the system is reduced and its solved for the new
    substitution. After we get the solution for $U$:

    .. math::  U = a(t)

    We substitute and return the reduced system:

    .. math::
        a(t) = t*X' - X

    Parameters
    ==========

    A: Matrix
        Coefficient matrix($A(t)*t$) of the second order system of this form.
    b: Matrix
        Non-homogeneous term($b(t)$) of the system of ODEs.
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the system of ODEs.

    Returns
    =======

    List

    """

    U = Matrix([t*func.diff(t) - func for func in funcs])

    sol = linodesolve(A, t, t*b)
    reduced_eqs = [Eq(u, s) for s, u in zip(sol, U)]
    reduced_eqs = canonical_odes(reduced_eqs, funcs, t)[0]

    return reduced_eqs


def _second_order_subs_type2(A, funcs, t_):
    r"""
    Returns a second order system based on the coefficient matrix passed.

    Explanation
    ===========

    This function returns a system of second order ODE of the following form:

    .. math::
        X'' = A * X

    Here, $X$ is the vector of dependent variables, but a bit modified, $A$ is the
    coefficient matrix passed.

    Along with returning the second order system, this function also returns the new
    dependent variables with the new independent variable `t_` passed.

    Parameters
    ==========

    A: Matrix
        Coefficient matrix of the system
    funcs: List
        List of old dependent variables
    t_: Symbol
        New independent variable

    Returns
    =======

    List, List

    """
    func_names = [func.func.__name__ for func in funcs]
    new_funcs = [Function(Dummy("{}_".format(name)))(t_) for name in func_names]
    rhss = A * Matrix(new_funcs)
    new_eqs = [Eq(func.diff(t_, 2), rhs) for func, rhs in zip(new_funcs, rhss)]

    return new_eqs, new_funcs


def _is_euler_system(As, t):
    return all(_matrix_is_constant((A*t**i).applyfunc(ratsimp), t) for i, A in enumerate(As))


def _classify_linear_system(eqs, funcs, t, is_canon=False):
    r"""
    Returns a dictionary with details of the eqs if the system passed is linear
    and can be classified by this function else returns None

    Explanation
    ===========

    This function takes the eqs, converts it into a form Ax = b where x is a vector of terms
    containing dependent variables and their derivatives till their maximum order. If it is
    possible to convert eqs into Ax = b, then all the equations in eqs are linear otherwise
    they are non-linear.

    To check if the equations are constant coefficient, we need to check if all the terms in
    A obtained above are constant or not.

    To check if the equations are homogeneous or not, we need to check if b is a zero matrix
    or not.

    Parameters
    ==========

    eqs: List
        List of ODEs
    funcs: List
        List of dependent variables
    t: Symbol
        Independent variable of the equations in eqs
    is_canon: Boolean
        If True, then this function will not try to get the
        system in canonical form. Default value is False

    Returns
    =======

    match = {
        'no_of_equation': len(eqs),
        'eq': eqs,
        'func': funcs,
        'order': order,
        'is_linear': is_linear,
        'is_constant': is_constant,
        'is_homogeneous': is_homogeneous,
    }

    Dict or list of Dicts or None
        Dict with values for keys:
            1. no_of_equation: Number of equations
            2. eq: The set of equations
            3. func: List of dependent variables
            4. order: A dictionary that gives the order of the
                      dependent variable in eqs
            5. is_linear: Boolean value indicating if the set of
                          equations are linear or not.
            6. is_constant: Boolean value indicating if the set of
                          equations have constant coefficients or not.
            7. is_homogeneous: Boolean value indicating if the set of
                          equations are homogeneous or not.
            8. commutative_antiderivative: Antiderivative of the coefficient
                          matrix if the coefficient matrix is non-constant
                          and commutative with its antiderivative. This key
                          may or may not exist.
            9. is_general: Boolean value indicating if the system of ODEs is
                           solvable using one of the general case solvers or not.
            10. rhs: rhs of the non-homogeneous system of ODEs in Matrix form. This
                     key may or may not exist.
            11. is_higher_order: True if the system passed has an order greater than 1.
                                 This key may or may not exist.
            12. is_second_order: True if the system passed is a second order ODE. This
                                 key may or may not exist.
        This Dict is the answer returned if the eqs are linear and constant
        coefficient. Otherwise, None is returned.

    """

    # Error for i == 0 can be added but isn't for now

    # Check for len(funcs) == len(eqs)
    if len(funcs) != len(eqs):
        raise ValueError("Number of functions given is not equal to the number of equations %s" % funcs)

    # ValueError when functions have more than one arguments
    for func in funcs:
        if len(func.args) != 1:
            raise ValueError("dsolve() and classify_sysode() work with "
            "functions of one variable only, not %s" % func)

    # Getting the func_dict and order using the helper
    # function
    order = _get_func_order(eqs, funcs)
    system_order = max(order[func] for func in funcs)
    is_higher_order = system_order > 1
    is_second_order = system_order == 2 and all(order[func] == 2 for func in funcs)

    # Not adding the check if the len(func.args) for
    # every func in funcs is 1

    # Linearity check
    try:

        canon_eqs = canonical_odes(eqs, funcs, t) if not is_canon else [eqs]
        if len(canon_eqs) == 1:
            As, b = linear_ode_to_matrix(canon_eqs[0], funcs, t, system_order)
        else:

            match = {
                'is_implicit': True,
                'canon_eqs': canon_eqs
            }

            return match

    # When the system of ODEs is non-linear, an ODENonlinearError is raised.
    # This function catches the error and None is returned.
    except ODENonlinearError:
        return None

    is_linear = True

    # Homogeneous check
    is_homogeneous = True if b.is_zero_matrix else False

    # Is general key is used to identify if the system of ODEs can be solved by
    # one of the general case solvers or not.
    match = {
        'no_of_equation': len(eqs),
        'eq': eqs,
        'func': funcs,
        'order': order,
        'is_linear': is_linear,
        'is_homogeneous': is_homogeneous,
        'is_general': True
    }

    if not is_homogeneous:
        match['rhs'] = b

    is_constant = all(_matrix_is_constant(A_, t) for A_ in As)

    # The match['is_linear'] check will be added in the future when this
    # function becomes ready to deal with non-linear systems of ODEs

    if not is_higher_order:
        A = As[1]
        match['func_coeff'] = A

        # Constant coefficient check
        is_constant = _matrix_is_constant(A, t)
        match['is_constant'] = is_constant

        try:
            system_info = linodesolve_type(A, t, b=b)
        except NotImplementedError:
            return None

        match.update(system_info)
        antiderivative = match.pop("antiderivative")

        if not is_constant:
            match['commutative_antiderivative'] = antiderivative

        return match
    else:
        match['type_of_equation'] = "type0"

        if is_second_order:
            A1, A0 = As[1:]

            match_second_order = _match_second_order_type(A1, A0, t)
            match.update(match_second_order)

            match['is_second_order'] = True

        # If system is constant, then no need to check if its in euler
        # form or not. It will be easier and faster to directly proceed
        # to solve it.
        if match['type_of_equation'] == "type0" and not is_constant:
            is_euler = _is_euler_system(As, t)
            if is_euler:
                t_ = Symbol('{}_'.format(t))
                match.update({'is_transformed': True, 'type_of_equation': 'type1',
                              't_': t_})
            else:
                is_jordan = lambda M: M == Matrix.jordan_block(M.shape[0], M[0, 0])
                terms = _factor_matrix(As[-1], t)
                if all(A.is_zero_matrix for A in As[1:-1]) and terms is not None and not is_jordan(terms[1]):
                    P, J = terms[1].jordan_form()
                    match.update({'type_of_equation': 'type2', 'J': J,
                                  'f(t)': terms[0], 'P': P, 'is_transformed': True})

            if match['type_of_equation'] != 'type0' and is_second_order:
                match.pop('is_second_order', None)

        match['is_higher_order'] = is_higher_order

        return match

def _preprocess_eqs(eqs):
    processed_eqs = []
    for eq in eqs:
        processed_eqs.append(eq if isinstance(eq, Equality) else Eq(eq, 0))

    return processed_eqs


def _eqs2dict(eqs, funcs):
    eqsorig = {}
    eqsmap = {}
    funcset = set(funcs)
    for eq in eqs:
        f1, = eq.lhs.atoms(AppliedUndef)
        f2s = (eq.rhs.atoms(AppliedUndef) - {f1}) & funcset
        eqsmap[f1] = f2s
        eqsorig[f1] = eq
    return eqsmap, eqsorig


def _dict2graph(d):
    nodes = list(d)
    edges = [(f1, f2) for f1, f2s in d.items() for f2 in f2s]
    G = (nodes, edges)
    return G


def _is_type1(scc, t):
    eqs, funcs = scc

    try:
        (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, 1)
    except (ODENonlinearError, ODEOrderError):
        return False

    if _matrix_is_constant(A0, t) and b.is_zero_matrix:
        return True

    return False


def _combine_type1_subsystems(subsystem, funcs, t):
    indices = [i for i, sys in enumerate(zip(subsystem, funcs)) if _is_type1(sys, t)]
    remove = set()
    for ip, i in enumerate(indices):
        for j in indices[ip+1:]:
            if any(eq2.has(funcs[i]) for eq2 in subsystem[j]):
                subsystem[j] = subsystem[i] + subsystem[j]
                remove.add(i)
    subsystem = [sys for i, sys in enumerate(subsystem) if i not in remove]
    return subsystem


def _component_division(eqs, funcs, t):

    # Assuming that each eq in eqs is in canonical form,
    # that is, [f(x).diff(x) = .., g(x).diff(x) = .., etc]
    # and that the system passed is in its first order
    eqsmap, eqsorig = _eqs2dict(eqs, funcs)

    subsystems = []
    for cc in connected_components(_dict2graph(eqsmap)):
        eqsmap_c = {f: eqsmap[f] for f in cc}
        sccs = strongly_connected_components(_dict2graph(eqsmap_c))
        subsystem = [[eqsorig[f] for f in scc] for scc in sccs]
        subsystem = _combine_type1_subsystems(subsystem, sccs, t)
        subsystems.append(subsystem)

    return subsystems


# Returns: List of equations
def _linear_ode_solver(match):
    t = match['t']
    funcs = match['func']

    rhs = match.get('rhs', None)
    tau = match.get('tau', None)
    t = match['t_'] if 't_' in match else t
    A = match['func_coeff']

    # Note: To make B None when the matrix has constant
    # coefficient
    B = match.get('commutative_antiderivative', None)
    type = match['type_of_equation']

    sol_vector = linodesolve(A, t, b=rhs, B=B,
                             type=type, tau=tau)

    sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]

    return sol


def _select_equations(eqs, funcs, key=lambda x: x):
    eq_dict = {e.lhs: e.rhs for e in eqs}
    return [Eq(f, eq_dict[key(f)]) for f in funcs]


def _higher_order_ode_solver(match):
    eqs = match["eq"]
    funcs = match["func"]
    t = match["t"]
    sysorder = match['order']
    type = match.get('type_of_equation', "type0")

    is_second_order = match.get('is_second_order', False)
    is_transformed = match.get('is_transformed', False)
    is_euler = is_transformed and type == "type1"
    is_higher_order_type2 = is_transformed and type == "type2" and 'P' in match

    if is_second_order:
        new_eqs, new_funcs = _second_order_to_first_order(eqs, funcs, t,
                                                          A1=match.get("A1", None), A0=match.get("A0", None),
                                                          b=match.get("rhs", None), type=type,
                                                          t_=match.get("t_", None))
    else:
        new_eqs, new_funcs = _higher_order_to_first_order(eqs, sysorder, t, funcs=funcs,
                                                          type=type, J=match.get('J', None),
                                                          f_t=match.get('f(t)', None),
                                                          P=match.get('P', None), b=match.get('rhs', None))

    if is_transformed:
        t = match.get('t_', t)

    if not is_higher_order_type2:
        new_eqs = _select_equations(new_eqs, [f.diff(t) for f in new_funcs])

    sol = None

    # NotImplementedError may be raised when the system may be actually
    # solvable if it can be just divided into sub-systems
    try:
        if not is_higher_order_type2:
            sol = _strong_component_solver(new_eqs, new_funcs, t)
    except NotImplementedError:
        sol = None

    # Dividing the system only when it becomes essential
    if sol is None:
        try:
            sol = _component_solver(new_eqs, new_funcs, t)
        except NotImplementedError:
            sol = None

    if sol is None:
        return sol

    is_second_order_type2 = is_second_order and type == "type2"

    underscores = '__' if is_transformed else '_'

    sol = _select_equations(sol, funcs,
                            key=lambda x: Function(Dummy('{}{}0'.format(x.func.__name__, underscores)))(t))

    if match.get("is_transformed", False):
        if is_second_order_type2:
            g_t = match["g(t)"]
            tau = match["tau"]
            sol = [Eq(s.lhs, s.rhs.subs(t, tau) * g_t) for s in sol]
        elif is_euler:
            t = match['t']
            tau = match['t_']
            sol = [s.subs(tau, log(t)) for s in sol]
        elif is_higher_order_type2:
            P = match['P']
            sol_vector = P * Matrix([s.rhs for s in sol])
            sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]

    return sol


# Returns: List of equations or None
# If None is returned by this solver, then the system
# of ODEs cannot be solved directly by dsolve_system.
def _strong_component_solver(eqs, funcs, t):
    from sympy.solvers.ode.ode import dsolve, constant_renumber

    match = _classify_linear_system(eqs, funcs, t, is_canon=True)
    sol = None

    # Assuming that we can't get an implicit system
    # since we are already canonical equations from
    # dsolve_system
    if match:
        match['t'] = t

        if match.get('is_higher_order', False):
            sol = _higher_order_ode_solver(match)

        elif match.get('is_linear', False):
            sol = _linear_ode_solver(match)

        # Note: For now, only linear systems are handled by this function
        # hence, the match condition is added. This can be removed later.
        if sol is None and len(eqs) == 1:
            sol = dsolve(eqs[0], func=funcs[0])
            variables = Tuple(eqs[0]).free_symbols
            new_constants = [Dummy() for _ in range(ode_order(eqs[0], funcs[0]))]
            sol = constant_renumber(sol, variables=variables, newconstants=new_constants)
            sol = [sol]

        # To add non-linear case here in future

    return sol


def _get_funcs_from_canon(eqs):
    return [eq.lhs.args[0] for eq in eqs]


# Returns: List of Equations(a solution)
def _weak_component_solver(wcc, t):

    # We will divide the systems into sccs
    # only when the wcc cannot be solved as
    # a whole
    eqs = []
    for scc in wcc:
        eqs += scc
    funcs = _get_funcs_from_canon(eqs)

    sol = _strong_component_solver(eqs, funcs, t)
    if sol:
        return sol

    sol = []

    for scc in wcc:
        eqs = scc
        funcs = _get_funcs_from_canon(eqs)

        # Substituting solutions for the dependent
        # variables solved in previous SCC, if any solved.
        comp_eqs = [eq.subs({s.lhs: s.rhs for s in sol}) for eq in eqs]
        scc_sol = _strong_component_solver(comp_eqs, funcs, t)

        if scc_sol is None:
            raise NotImplementedError(filldedent('''
                The system of ODEs passed cannot be solved by dsolve_system.
            '''))

        # scc_sol: List of equations
        # scc_sol is a solution
        sol += scc_sol

    return sol


# Returns: List of Equations(a solution)
def _component_solver(eqs, funcs, t):
    components = _component_division(eqs, funcs, t)
    sol = []

    for wcc in components:

        # wcc_sol: List of Equations
        sol += _weak_component_solver(wcc, t)

    # sol: List of Equations
    return sol


def _second_order_to_first_order(eqs, funcs, t, type="auto", A1=None,
                                 A0=None, b=None, t_=None):
    r"""
    Expects the system to be in second order and in canonical form

    Explanation
    ===========

    Reduces a second order system into a first order one depending on the type of second
    order system.
    1. "type0": If this is passed, then the system will be reduced to first order by
                introducing dummy variables.
    2. "type1": If this is passed, then a particular substitution will be used to reduce the
                the system into first order.
    3. "type2": If this is passed, then the system will be transformed with new dependent
                variables and independent variables. This transformation is a part of solving
                the corresponding system of ODEs.

    `A1` and `A0` are the coefficient matrices from the system and it is assumed that the
    second order system has the form given below:

    .. math::
        A2 * X'' = A1 * X' + A0 * X + b

    Here, $A2$ is the coefficient matrix for the vector $X''$ and $b$ is the non-homogeneous
    term.

    Default value for `b` is None but if `A1` and `A0` are passed and `b` is not passed, then the
    system will be assumed homogeneous.

    """
    is_a1 = A1 is None
    is_a0 = A0 is None

    if (type == "type1" and is_a1) or (type == "type2" and is_a0)\
        or (type == "auto" and (is_a1 or is_a0)):
        (A2, A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, 2)

        if not A2.is_Identity:
            raise ValueError(filldedent('''
                The system must be in its canonical form.
            '''))

    if type == "auto":
        match = _match_second_order_type(A1, A0, t)
        type = match["type_of_equation"]
        A1 = match.get("A1", None)
        A0 = match.get("A0", None)

    sys_order = dict.fromkeys(funcs, 2)

    if type == "type1":
        if b is None:
            b = zeros(len(eqs))
        eqs = _second_order_subs_type1(A1, b, funcs, t)
        sys_order = dict.fromkeys(funcs, 1)

    if type == "type2":
        if t_ is None:
            t_ = Symbol("{}_".format(t))
        t = t_
        eqs, funcs = _second_order_subs_type2(A0, funcs, t_)
        sys_order = dict.fromkeys(funcs, 2)

    return _higher_order_to_first_order(eqs, sys_order, t, funcs=funcs)


def _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, b=None, P=None):

    # Note: To add a test for this ValueError
    if J is None or f_t is None or not _matrix_is_constant(J, t):
        raise ValueError(filldedent('''
            Correctly input for args 'A' and 'f_t' for Linear, Higher Order,
            Type 2
        '''))

    if P is None and b is not None and not b.is_zero_matrix:
        raise ValueError(filldedent('''
            Provide the keyword 'P' for matrix P in A = P * J * P-1.
        '''))

    new_funcs = Matrix([Function(Dummy('{}__0'.format(f.func.__name__)))(t) for f in funcs])
    new_eqs = new_funcs.diff(t, max_order) - f_t * J * new_funcs

    if b is not None and not b.is_zero_matrix:
        new_eqs -= P.inv() * b

    new_eqs = canonical_odes(new_eqs, new_funcs, t)[0]

    return new_eqs, new_funcs


def _higher_order_to_first_order(eqs, sys_order, t, funcs=None, type="type0", **kwargs):
    if funcs is None:
        funcs = sys_order.keys()

    # Standard Cauchy Euler system
    if type == "type1":
        t_ = Symbol('{}_'.format(t))
        new_funcs = [Function(Dummy('{}_'.format(f.func.__name__)))(t_) for f in funcs]
        max_order = max(sys_order[func] for func in funcs)
        subs_dict = dict(zip(funcs, new_funcs))
        subs_dict[t] = exp(t_)

        free_function = Function(Dummy())

        def _get_coeffs_from_subs_expression(expr):
            if isinstance(expr, Subs):
                free_symbol = expr.args[1][0]
                term = expr.args[0]
                return {ode_order(term, free_symbol): 1}

            if isinstance(expr, Mul):
                coeff = expr.args[0]
                order = list(_get_coeffs_from_subs_expression(expr.args[1]).keys())[0]
                return {order: coeff}

            if isinstance(expr, Add):
                coeffs = {}
                for arg in expr.args:

                    if isinstance(arg, Mul):
                        coeffs.update(_get_coeffs_from_subs_expression(arg))

                    else:
                        order = list(_get_coeffs_from_subs_expression(arg).keys())[0]
                        coeffs[order] = 1

                return coeffs

        for o in range(1, max_order + 1):
            expr = free_function(log(t_)).diff(t_, o)*t_**o
            coeff_dict = _get_coeffs_from_subs_expression(expr)
            coeffs = [coeff_dict[order] if order in coeff_dict else 0 for order in range(o + 1)]
            expr_to_subs = sum(free_function(t_).diff(t_, i) * c for i, c in
                        enumerate(coeffs)) / t**o
            subs_dict.update({f.diff(t, o): expr_to_subs.subs(free_function(t_), nf)
                              for f, nf in zip(funcs, new_funcs)})

        new_eqs = [eq.subs(subs_dict) for eq in eqs]
        new_sys_order = {nf: sys_order[f] for f, nf in zip(funcs, new_funcs)}

        new_eqs = canonical_odes(new_eqs, new_funcs, t_)[0]

        return _higher_order_to_first_order(new_eqs, new_sys_order, t_, funcs=new_funcs)

    # Systems of the form: X(n)(t) = f(t)*A*X + b
    # where X(n)(t) is the nth derivative of the vector of dependent variables
    # with respect to the independent variable and A is a constant matrix.
    if type == "type2":
        J = kwargs.get('J', None)
        f_t = kwargs.get('f_t', None)
        b = kwargs.get('b', None)
        P = kwargs.get('P', None)
        max_order = max(sys_order[func] for func in funcs)

        return _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, P=P, b=b)

        # Note: To be changed to this after doit option is disabled for default cases
        # new_sysorder = _get_func_order(new_eqs, new_funcs)
        #
        # return _higher_order_to_first_order(new_eqs, new_sysorder, t, funcs=new_funcs)

    new_funcs = []

    for prev_func in funcs:
        func_name = prev_func.func.__name__
        func = Function(Dummy('{}_0'.format(func_name)))(t)
        new_funcs.append(func)
        subs_dict = {prev_func: func}
        new_eqs = []

        for i in range(1, sys_order[prev_func]):
            new_func = Function(Dummy('{}_{}'.format(func_name, i)))(t)
            subs_dict[prev_func.diff(t, i)] = new_func
            new_funcs.append(new_func)

            prev_f = subs_dict[prev_func.diff(t, i-1)]
            new_eq = Eq(prev_f.diff(t), new_func)
            new_eqs.append(new_eq)

        eqs = [eq.subs(subs_dict) for eq in eqs] + new_eqs

    return eqs, new_funcs


def dsolve_system(eqs, funcs=None, t=None, ics=None, doit=False, simplify=True):
    r"""
    Solves any(supported) system of Ordinary Differential Equations

    Explanation
    ===========

    This function takes a system of ODEs as an input, determines if the
    it is solvable by this function, and returns the solution if found any.

    This function can handle:
    1. Linear, First Order, Constant coefficient homogeneous system of ODEs
    2. Linear, First Order, Constant coefficient non-homogeneous system of ODEs
    3. Linear, First Order, non-constant coefficient homogeneous system of ODEs
    4. Linear, First Order, non-constant coefficient non-homogeneous system of ODEs
    5. Any implicit system which can be divided into system of ODEs which is of the above 4 forms
    6. Any higher order linear system of ODEs that can be reduced to one of the 5 forms of systems described above.

    The types of systems described above are not limited by the number of equations, i.e. this
    function can solve the above types irrespective of the number of equations in the system passed.
    But, the bigger the system, the more time it will take to solve the system.

    This function returns a list of solutions. Each solution is a list of equations where LHS is
    the dependent variable and RHS is an expression in terms of the independent variable.

    Among the non constant coefficient types, not all the systems are solvable by this function. Only
    those which have either a coefficient matrix with a commutative antiderivative or those systems which
    may be divided further so that the divided systems may have coefficient matrix with commutative antiderivative.

    Parameters
    ==========

    eqs : List
        system of ODEs to be solved
    funcs : List or None
        List of dependent variables that make up the system of ODEs
    t : Symbol or None
        Independent variable in the system of ODEs
    ics : Dict or None
        Set of initial boundary/conditions for the system of ODEs
    doit : Boolean
        Evaluate the solutions if True. Default value is True. Can be
        set to false if the integral evaluation takes too much time and/or
        is not required.
