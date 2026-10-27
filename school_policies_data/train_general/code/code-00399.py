    simplify: Boolean
        Simplify the solutions for the systems. Default value is True.
        Can be set to false if simplification takes too much time and/or
        is not required.

    Examples
    ========

    >>> from sympy import symbols, Eq, Function
    >>> from sympy.solvers.ode.systems import dsolve_system
    >>> f, g = symbols("f g", cls=Function)
    >>> x = symbols("x")

    >>> eqs = [Eq(f(x).diff(x), g(x)), Eq(g(x).diff(x), f(x))]
    >>> dsolve_system(eqs)
    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]

    You can also pass the initial conditions for the system of ODEs:

    >>> dsolve_system(eqs, ics={f(0): 1, g(0): 0})
    [[Eq(f(x), exp(x)/2 + exp(-x)/2), Eq(g(x), exp(x)/2 - exp(-x)/2)]]

    Optionally, you can pass the dependent variables and the independent
    variable for which the system is to be solved:

    >>> funcs = [f(x), g(x)]
    >>> dsolve_system(eqs, funcs=funcs, t=x)
    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]

    Lets look at an implicit system of ODEs:

    >>> eqs = [Eq(f(x).diff(x)**2, g(x)**2), Eq(g(x).diff(x), g(x))]
    >>> dsolve_system(eqs)
    [[Eq(f(x), C1 - C2*exp(x)), Eq(g(x), C2*exp(x))], [Eq(f(x), C1 + C2*exp(x)), Eq(g(x), C2*exp(x))]]

    Returns
    =======

    List of List of Equations

    Raises
    ======

    NotImplementedError
        When the system of ODEs is not solvable by this function.
    ValueError
        When the parameters passed are not in the required form.

    """
    from sympy.solvers.ode.ode import solve_ics, _extract_funcs, constant_renumber

    if not iterable(eqs):
        raise ValueError(filldedent('''
            List of equations should be passed. The input is not valid.
        '''))

    eqs = _preprocess_eqs(eqs)

    if funcs is not None and not isinstance(funcs, list):
        raise ValueError(filldedent('''
            Input to the funcs should be a list of functions.
        '''))

    if funcs is None:
        funcs = _extract_funcs(eqs)

    if any(len(func.args) != 1 for func in funcs):
        raise ValueError(filldedent('''
            dsolve_system can solve a system of ODEs with only one independent
            variable.
        '''))

    if len(eqs) != len(funcs):
        raise ValueError(filldedent('''
            Number of equations and number of functions do not match
        '''))

    if t is not None and not isinstance(t, Symbol):
        raise ValueError(filldedent('''
            The independent variable must be of type Symbol
        '''))

    if t is None:
        t = list(list(eqs[0].atoms(Derivative))[0].atoms(Symbol))[0]

    sols = []
    canon_eqs = canonical_odes(eqs, funcs, t)

    for canon_eq in canon_eqs:
        try:
            sol = _strong_component_solver(canon_eq, funcs, t)
        except NotImplementedError:
            sol = None

        if sol is None:
            sol = _component_solver(canon_eq, funcs, t)

        sols.append(sol)

    if sols:
        final_sols = []
        variables = Tuple(*eqs).free_symbols

        for sol in sols:

            sol = _select_equations(sol, funcs)
            sol = constant_renumber(sol, variables=variables)

            if ics:
                constants = Tuple(*sol).free_symbols - variables
                solved_constants = solve_ics(sol, funcs, constants, ics)
                sol = [s.subs(solved_constants) for s in sol]

            if simplify:
                constants = Tuple(*sol).free_symbols - variables
                sol = simpsol(sol, [t], constants, doit=doit)

            final_sols.append(sol)

        sols = final_sols

    return sols
<|endoftext|>from .ode import (allhints, checkinfsol, classify_ode,
        constantsimp, dsolve, homogeneous_order)

from .lie_group import infinitesimals

from .subscheck import checkodesol

from .systems import (canonical_odes, linear_ode_to_matrix,
        linodesolve)


__all__ = [
    'allhints', 'checkinfsol', 'checkodesol', 'classify_ode', 'constantsimp',
    'dsolve', 'homogeneous_order', 'infinitesimals', 'canonical_odes', 'linear_ode_to_matrix',
    'linodesolve'
]
<|endoftext|>r"""
This module contains :py:meth:`~sympy.solvers.ode.riccati.solve_riccati`,
a function which gives all rational particular solutions to first order
Riccati ODEs. A general first order Riccati ODE is given by -

.. math:: y' = b_0(x) + b_1(x)w + b_2(x)w^2

where `b_0, b_1` and `b_2` can be arbitrary rational functions of `x`
with `b_2 \ne 0`. When `b_2 = 0`, the equation is not a Riccati ODE
anymore and becomes a Linear ODE. Similarly, when `b_0 = 0`, the equation
is a Bernoulli ODE. The algorithm presented below can find rational
solution(s) to all ODEs with `b_2 \ne 0` that have a rational solution,
or prove that no rational solution exists for the equation.

Background
==========

A Riccati equation can be transformed to its normal form

.. math:: y' + y^2 = a(x)

using the transformation

.. math:: y = -b_2(x) - \frac{b'_2(x)}{2 b_2(x)} - \frac{b_1(x)}{2}

where `a(x)` is given by

.. math:: a(x) = \frac{1}{4}\left(\frac{b_2'}{b_2} + b_1\right)^2 - \frac{1}{2}\left(\frac{b_2'}{b_2} + b_1\right)' - b_0 b_2

Thus, we can develop an algorithm to solve for the Riccati equation
in its normal form, which would in turn give us the solution for
the original Riccati equation.

Algorithm
=========

The algorithm implemented here is presented in the Ph.D thesis
"Rational and Algebraic Solutions of First-Order Algebraic ODEs"
by N. Thieu Vo. The entire thesis can be found here -
https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf

We have only implemented the Rational Riccati solver (Algorithm 11,
Pg 78-82 in Thesis). Before we proceed towards the implementation
of the algorithm, a few definitions to understand are -

1. Valuation of a Rational Function at `\infty`:
    The valuation of a rational function `p(x)` at `\infty` is equal
    to the difference between the degree of the denominator and the
    numerator of `p(x)`.

    NOTE: A general definition of valuation of a rational function
    at any value of `x` can be found in Pg 63 of the thesis, but
    is not of any interest for this algorithm.

2. Zeros and Poles of a Rational Function:
    Let `a(x) = \frac{S(x)}{T(x)}, T \ne 0` be a rational function
    of `x`. Then -

    a. The Zeros of `a(x)` are the roots of `S(x)`.
    b. The Poles of `a(x)` are the roots of `T(x)`. However, `\infty`
    can also be a pole of a(x). We say that `a(x)` has a pole at
    `\infty` if `a(\frac{1}{x})` has a pole at 0.

Every pole is associated with an order that is equal to the multiplicity
of its appearance as a root of `T(x)`. A pole is called a simple pole if
it has an order 1. Similarly, a pole is called a multiple pole if it has
an order `\ge` 2.

Necessary Conditions
====================

For a Riccati equation in its normal form,

.. math:: y' + y^2 = a(x)

we can define

a. A pole is called a movable pole if it is a pole of `y(x)` and is not
a pole of `a(x)`.
b. Similarly, a pole is called a non-movable pole if it is a pole of both
`y(x)` and `a(x)`.

Then, the algorithm states that a rational solution exists only if -

a. Every pole of `a(x)` must be either a simple pole or a multiple pole
of even order.
b. The valuation of `a(x)` at `\infty` must be even or be `\ge` 2.

This algorithm finds all possible rational solutions for the Riccati ODE.
If no rational solutions are found, it means that no rational solutions
exist.

The algorithm works for Riccati ODEs where the coefficients are rational
functions in the independent variable `x` with rational number coefficients
i.e. in `Q(x)`. The coefficients in the rational function cannot be floats,
irrational numbers, symbols or any other kind of expression. The reasons
for this are -

1. When using symbols, different symbols could take the same value and this
would affect the multiplicity of poles if symbols are present here.

2. An integer degree bound is required to calculate a polynomial solution
to an auxiliary differential equation, which in turn gives the particular
solution for the original ODE. If symbols/floats/irrational numbers are
present, we cannot determine if the expression for the degree bound is an
integer or not.

Solution
========

With these definitions, we can state a general form for the solution of
the equation. `y(x)` must have the form -

.. math:: y(x) = \sum_{i=1}^{n} \sum_{j=1}^{r_i} \frac{c_{ij}}{(x - x_i)^j} + \sum_{i=1}^{m} \frac{1}{x - \chi_i} + \sum_{i=0}^{N} d_i x^i

where `x_1, x_2, \dots, x_n` are non-movable poles of `a(x)`,
`\chi_1, \chi_2, \dots, \chi_m` are movable poles of `a(x)`, and the values
of `N, n, r_1, r_2, \dots, r_n` can be determined from `a(x)`. The
coefficient vectors `(d_0, d_1, \dots, d_N)` and `(c_{i1}, c_{i2}, \dots, c_{i r_i})`
can be determined from `a(x)`. We will have 2 choices each of these vectors
and part of the procedure is figuring out which of the 2 should be used
to get the solution correctly.

Implementation
==============

In this implementation, we use ``Poly`` to represent a rational function
rather than using ``Expr`` since ``Poly`` is much faster. Since we cannot
represent rational functions directly using ``Poly``, we instead represent
a rational function with 2 ``Poly`` objects - one for its numerator and
the other for its denominator.

The code is written to match the steps given in the thesis (Pg 82)

Step 0 : Match the equation -
Find `b_0, b_1` and `b_2`. If `b_2 = 0` or no such functions exist, raise
an error

Step 1 : Transform the equation to its normal form as explained in the
theory section.

Step 2 : Initialize an empty set of solutions, ``sol``.

Step 3 : If `a(x) = 0`, append `\frac{1}/{(x - C1)}` to ``sol``.

Step 4 : If `a(x)` is a rational non-zero number, append `\pm \sqrt{a}`
to ``sol``.

Step 5 : Find the poles and their multiplicities of `a(x)`. Let
the number of poles be `n`. Also find the valuation of `a(x)` at
`\infty` using ``val_at_inf``.

NOTE: Although the algorithm considers `\infty` as a pole, it is
not mentioned if it a part of the set of finite poles. `\infty`
is NOT a part of the set of finite poles. If a pole exists at
`\infty`, we use its multiplicity to find the laurent series of
`a(x)` about `\infty`.

Step 6 : Find `n` c-vectors (one for each pole) and 1 d-vector using
``construct_c`` and ``construct_d``. Now, determine all the ``2**(n + 1)``
combinations of choosing between 2 choices for each of the `n` c-vectors
and 1 d-vector.

NOTE: The equation for `d_{-1}` in Case 4 (Pg 80) has a printinig
mistake. The term `- d_N` must be replaced with `-N d_N`. The same
has been explained in the code as well.

For each of these above combinations, do

Step 8 : Compute `m` in ``compute_m_ybar``. `m` is the degree bound of
the polynomial solution we must find for the auxiliary equation.

Step 9 : In ``compute_m_ybar``, compute ybar as well where ``ybar`` is
one part of y(x) -

.. math:: \overline{y}(x) = \sum_{i=1}^{n} \sum_{j=1}^{r_i} \frac{c_{ij}}{(x - x_i)^j} + \sum_{i=0}^{N} d_i x^i

Step 10 : If `m` is a non-negative integer -

Step 11: Find a polynomial solution of degree `m` for the auxiliary equation.

There are 2 cases possible -

    a. `m` is a non-negative integer: We can solve for the coefficients
    in `p(x)` using Undetermined Coefficients.

    b. `m` is not a non-negative integer: In this case, we cannot find
    a polynomial solution to the auxiliary equation, and hence, we ignore
    this value of `m`.

Step 12 : For each `p(x)` that exists, append `ybar + \frac{p'(x)}{p(x)}`
to ``sol``.

Step 13 : For each solution in ``sol``, apply an inverse transformation,
so that the solutions of the original equation are found using the
solutions of the equation in its normal form.
"""


from itertools import product
from sympy.core import S
from sympy.core.add import Add
from sympy.core.numbers import oo, Float
from sympy.core.function import count_ops
from sympy.core.relational import Eq
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions import sqrt, exp
from sympy.functions.elementary.complexes import sign
from sympy.integrals.integrals import Integral
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyroots import roots
from sympy.solvers.solveset import linsolve


def riccati_normal(w, x, b1, b2):
    """
    Given a solution `w(x)` to the equation

    .. math:: w'(x) = b_0(x) + b_1(x)*w(x) + b_2(x)*w(x)^2

    and rational function coefficients `b_1(x)` and
    `b_2(x)`, this function transforms the solution to
    give a solution `y(x)` for its corresponding normal
    Riccati ODE

    .. math:: y'(x) + y(x)^2 = a(x)

    using the transformation

    .. math:: y(x) = -b_2(x)*w(x) - b'_2(x)/(2*b_2(x)) - b_1(x)/2
    """
    return -b2*w - b2.diff(x)/(2*b2) - b1/2


def riccati_inverse_normal(y, x, b1, b2, bp=None):
    """
    Inverse transforming the solution to the normal
    Riccati ODE to get the solution to the Riccati ODE.
    """
    # bp is the expression which is independent of the solution
    # and hence, it need not be computed again
    if bp is None:
        bp = -b2.diff(x)/(2*b2**2) - b1/(2*b2)
    # w(x) = -y(x)/b2(x) - b2'(x)/(2*b2(x)^2) - b1(x)/(2*b2(x))
    return -y/b2 + bp


def riccati_reduced(eq, f, x):
    """
    Convert a Riccati ODE into its corresponding
    normal Riccati ODE.
    """
    match, funcs = match_riccati(eq, f, x)
    # If equation is not a Riccati ODE, exit
    if not match:
        return False
    # Using the rational functions, find the expression for a(x)
    b0, b1, b2 = funcs
    a = -b0*b2 + b1**2/4 - b1.diff(x)/2 + 3*b2.diff(x)**2/(4*b2**2) + b1*b2.diff(x)/(2*b2) - \
        b2.diff(x, 2)/(2*b2)
    # Normal form of Riccati ODE is f'(x) + f(x)^2 = a(x)
    return f(x).diff(x) + f(x)**2 - a

def linsolve_dict(eq, syms):
    """
    Get the output of linsolve as a dict
    """
    # Convert tuple type return value of linsolve
    # to a dictionary for ease of use
    sol = linsolve(eq, syms)
    if not sol:
        return {}
    return dict(zip(syms, list(sol)[0]))


def match_riccati(eq, f, x):
    """
    A function that matches and returns the coefficients
    if an equation is a Riccati ODE

    Parameters
    ==========

    eq: Equation to be matched
    f: Dependent variable
    x: Independent variable

    Returns
    =======

    match: True if equation is a Riccati ODE, False otherwise
    funcs: [b0, b1, b2] if match is True, [] otherwise. Here,
    b0, b1 and b2 are rational functions which match the equation.
    """
    # Group terms based on f(x)
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    eq = eq.expand().collect(f(x))
    cf = eq.coeff(f(x).diff(x))

    # There must be an f(x).diff(x) term.
    # eq must be an Add object since we are using the expanded
    # equation and it must have atleast 2 terms (b2 != 0)
    if cf != 0 and isinstance(eq, Add):

        # Divide all coefficients by the coefficient of f(x).diff(x)
        # and add the terms again to get the same equation
        eq = Add(*((x/cf).cancel() for x in eq.args)).collect(f(x))

        # Match the equation with the pattern
        b1 = -eq.coeff(f(x))
        b2 = -eq.coeff(f(x)**2)
        b0 = (f(x).diff(x) - b1*f(x) - b2*f(x)**2 - eq).expand()
        funcs = [b0, b1, b2]

        # Check if coefficients are not symbols and floats
        if any(len(x.atoms(Symbol)) > 1 or len(x.atoms(Float)) for x in funcs):
            return False, []

        # If b_0(x) contains f(x), it is not a Riccati ODE
        if len(b0.atoms(f)) or not all((b2 != 0, b0.is_rational_function(x),
            b1.is_rational_function(x), b2.is_rational_function(x))):
            return False, []
        return True, funcs
    return False, []


def val_at_inf(num, den, x):
    # Valuation of a rational function at oo = deg(denom) - deg(numer)
    return den.degree(x) - num.degree(x)


def check_necessary_conds(val_inf, muls):
    """
    The necessary conditions for a rational solution
    to exist are as follows -

    i) Every pole of a(x) must be either a simple pole
    or a multiple pole of even order.

    ii) The valuation of a(x) at infinity must be even
    or be greater than or equal to 2.

    Here, a simple pole is a pole with multiplicity 1
    and a multiple pole is a pole with multiplicity
    greater than 1.
    """
    return (val_inf >= 2 or (val_inf <= 0 and val_inf%2 == 0)) and \
        all(mul == 1 or (mul%2 == 0 and mul >= 2) for mul in muls)


def inverse_transform_poly(num, den, x):
    """
    A function to make the substitution
    x -> 1/x in a rational function that
    is represented using Poly objects for
    numerator and denominator.
    """
    # Declare for reuse
    one = Poly(1, x)
    xpoly = Poly(x, x)

    # Check if degree of numerator is same as denominator
    pwr = val_at_inf(num, den, x)
    if pwr >= 0:
        # Denominator has greater degree. Substituting x with
        # 1/x would make the extra power go to the numerator
        if num.expr != 0:
            num = num.transform(one, xpoly) * x**pwr
            den = den.transform(one, xpoly)
    else:
        # Numerator has greater degree. Substituting x with
        # 1/x would make the extra power go to the denominator
        num = num.transform(one, xpoly)
        den = den.transform(one, xpoly) * x**(-pwr)
    return num.cancel(den, include=True)


def limit_at_inf(num, den, x):
    """
    Find the limit of a rational function
    at oo
    """
    # pwr = degree(num) - degree(den)
    pwr = -val_at_inf(num, den, x)
    # Numerator has a greater degree than denominator
    # Limit at infinity would depend on the sign of the
    # leading coefficients of numerator and denominator
    if pwr > 0:
        return oo*sign(num.LC()/den.LC())
    # Degree of numerator is equal to that of denominator
    # Limit at infinity is just the ratio of leading coeffs
    elif pwr == 0:
        return num.LC()/den.LC()
    # Degree of numerator is less than that of denominator
    # Limit at infinity is just 0
    else:
        return 0


def construct_c_case_1(num, den, x, pole):
    # Find the coefficient of 1/(x - pole)**2 in the
    # Laurent series expansion of a(x) about pole.
    num1, den1 = (num*Poly((x - pole)**2, x, extension=True)).cancel(den, include=True)
    r = (num1.subs(x, pole))/(den1.subs(x, pole))

    # If multiplicity is 2, the coefficient to be added
    # in the c-vector is c = (1 +- sqrt(1 + 4*r))/2
    if r != -S(1)/4:
        return [[(1 + sqrt(1 + 4*r))/2], [(1 - sqrt(1 + 4*r))/2]]
    return [[S.Half]]


def construct_c_case_2(num, den, x, pole, mul):
    # Generate the coefficients using the recurrence
    # relation mentioned in (5.14) in the thesis (Pg 80)

    # r_i = mul/2
    ri = mul//2

    # Find the Laurent series coefficients about the pole
    ser = rational_laurent_series(num, den, x, pole, mul, 6)

    # Start with an empty memo to store the coefficients
    # This is for the plus case
    cplus = [0 for i in range(ri)]

    # Base Case
    cplus[ri-1] = sqrt(ser[2*ri])

    # Iterate backwards to find all coefficients
    s = ri - 1
    sm = 0
    for s in range(ri-1, 0, -1):
        sm = 0
        for j in range(s+1, ri):
            sm += cplus[j-1]*cplus[ri+s-j-1]
        if s!= 1:
            cplus[s-1] = (ser[ri+s] - sm)/(2*cplus[ri-1])

    # Memo for the minus case
    cminus = [-x for x in cplus]

    # Find the 0th coefficient in the recurrence
    cplus[0] = (ser[ri+s] - sm - ri*cplus[ri-1])/(2*cplus[ri-1])
    cminus[0] = (ser[ri+s] - sm  - ri*cminus[ri-1])/(2*cminus[ri-1])

    # Add both the plus and minus cases' coefficients
    if cplus != cminus:
        return [cplus, cminus]
    return cplus


def construct_c_case_3():
    # If multiplicity is 1, the coefficient to be added
    # in the c-vector is 1 (no choice)
    return [[1]]


def construct_c(num, den, x, poles, muls):
    """
    Helper function to calculate the coefficients
    in the c-vector for each pole.
    """
    c = []
    for pole, mul in zip(poles, muls):
        c.append([])

        # Case 3
        if mul == 1:
            # Add the coefficients from Case 3
            c[-1].extend(construct_c_case_3())

        # Case 1
        elif mul == 2:
            # Add the coefficients from Case 1
            c[-1].extend(construct_c_case_1(num, den, x, pole))

        # Case 2
        else:
            # Add the coefficients from Case 2
            c[-1].extend(construct_c_case_2(num, den, x, pole, mul))

    return c


def construct_d_case_4(ser, N):
    # Initialize an empty vector
    dplus = [0 for i in range(N+2)]
    # d_N = sqrt(a_{2*N})
    dplus[N] = sqrt(ser[2*N])

    # Use the recurrence relations to find
    # the value of d_s
    for s in range(N-1, -2, -1):
        sm = 0
        for j in range(s+1, N):
            sm += dplus[j]*dplus[N+s-j]
        if s != -1:
            dplus[s] = (ser[N+s] - sm)/(2*dplus[N])

    # Coefficients for the case of d_N = -sqrt(a_{2*N})
    dminus = [-x for x in dplus]

    # The third equation in Eq 5.15 of the thesis is WRONG!
    # d_N must be replaced with N*d_N in that equation.
    dplus[-1] = (ser[N+s] - N*dplus[N] - sm)/(2*dplus[N])
    dminus[-1] = (ser[N+s] - N*dminus[N] - sm)/(2*dminus[N])

    if dplus != dminus:
        return [dplus, dminus]
    return dplus


def construct_d_case_5(ser):
    # List to store coefficients for plus case
    dplus = [0, 0]

    # d_0  = sqrt(a_0)
    dplus[0] = sqrt(ser[0])

    # d_(-1) = a_(-1)/(2*d_0)
    dplus[-1] = ser[-1]/(2*dplus[0])

    # Coefficients for the minus case are just the negative
    # of the coefficients for the positive case.
    dminus = [-x for x in dplus]

    if dplus != dminus:
        return [dplus, dminus]
    return dplus


def construct_d_case_6(num, den, x):
    # s_oo = lim x->0 1/x**2 * a(1/x) which is equivalent to
    # s_oo = lim x->oo x**2 * a(x)
    s_inf = limit_at_inf(Poly(x**2, x)*num, den, x)

    # d_(-1) = (1 +- sqrt(1 + 4*s_oo))/2
    if s_inf != -S(1)/4:
        return [[(1 + sqrt(1 + 4*s_inf))/2], [(1 - sqrt(1 + 4*s_inf))/2]]
    return [[S.Half]]


def construct_d(num, den, x, val_inf):
    """
    Helper function to calculate the coefficients
    in the d-vector based on the valuation of the
    function at oo.
    """
    N = -val_inf//2
    # Multiplicity of oo as a pole
    mul = -val_inf if val_inf < 0 else 0
    ser = rational_laurent_series(num, den, x, oo, mul, 1)

    # Case 4
    if val_inf < 0:
        d = construct_d_case_4(ser, N)

    # Case 5
    elif val_inf == 0:
        d = construct_d_case_5(ser)

    # Case 6
    else:
        d = construct_d_case_6(num, den, x)

    return d


def rational_laurent_series(num, den, x, r, m, n):
    r"""
    The function computes the Laurent series coefficients
    of a rational function.

    Parameters
    ==========

    num: A Poly object that is the numerator of `f(x)`.
    den: A Poly object that is the denominator of `f(x)`.
    x: The variable of expansion of the series.
    r: The point of expansion of the series.
    m: Multiplicity of r if r is a pole of `f(x)`. Should
    be zero otherwise.
    n: Order of the term upto which the series is expanded.

    Returns
    =======

    series: A dictionary that has power of the term as key
    and coefficient of that term as value.

    Below is a basic outline of how the Laurent series of a
    rational function `f(x)` about `x_0` is being calculated -

    1. Substitute `x + x_0` in place of `x`. If `x_0`
    is a pole of `f(x)`, multiply the expression by `x^m`
    where `m` is the multiplicity of `x_0`. Denote the
    the resulting expression as g(x). We do this substitution
    so that we can now find the Laurent series of g(x) about
    `x = 0`.

    2. We can then assume that the Laurent series of `g(x)`
    takes the following form -

    .. math:: g(x) = \frac{num(x)}{den(x)} = \sum_{m = 0}^{\infty} a_m x^m

    where `a_m` denotes the Laurent series coefficients.

    3. Multiply the denominator to the RHS of the equation
    and form a recurrence relation for the coefficients `a_m`.
    """
    one = Poly(1, x, extension=True)

    if r == oo:
        # Series at x = oo is equal to first transforming
        # the function from x -> 1/x and finding the
        # series at x = 0
        num, den = inverse_transform_poly(num, den, x)
        r = S(0)

    if r:
        # For an expansion about a non-zero point, a
        # transformation from x -> x + r must be made
        num = num.transform(Poly(x + r, x, extension=True), one)
        den = den.transform(Poly(x + r, x, extension=True), one)

    # Remove the pole from the denominator if the series
    # expansion is about one of the poles
    num, den = (num*x**m).cancel(den, include=True)

    # Equate coefficients for the first terms (base case)
    maxdegree = 1 + max(num.degree(), den.degree())
    syms = symbols(f'a:{maxdegree}', cls=Dummy)
    diff = num - den * Poly(syms[::-1], x)
    coeff_diffs = diff.all_coeffs()[::-1][:maxdegree]
    (coeffs, ) = linsolve(coeff_diffs, syms)

    # Use the recursion relation for the rest
    recursion = den.all_coeffs()[::-1]
    div, rec_rhs = recursion[0], recursion[1:]
    series = list(coeffs)
    while len(series) < n:
        next_coeff = Add(*(c*series[-1-n] for n, c in enumerate(rec_rhs))) / div
        series.append(-next_coeff)
    series = {m - i: val for i, val in enumerate(series)}
    return series

def compute_m_ybar(x, poles, choice, N):
    """
    Helper function to calculate -

    1. m - The degree bound for the polynomial
    solution that must be found for the auxiliary
    differential equation.

    2. ybar - Part of the solution which can be
    computed using the poles, c and d vectors.
    """
    ybar = 0
    m = Poly(choice[-1][-1], x, extension=True)

    # Calculate the first (nested) summation for ybar
    # as given in Step 9 of the Thesis (Pg 82)
    dybar = []
    for i, polei in enumerate(poles):
        for j, cij in enumerate(choice[i]):
            dybar.append(cij/(x - polei)**(j + 1))
        m -=Poly(choice[i][0], x, extension=True)  # can't accumulate Poly and use with Add
    ybar += Add(*dybar)

    # Calculate the second summation for ybar
    for i in range(N+1):
        ybar += choice[-1][i]*x**i
    return (m.expr, ybar)


def solve_aux_eq(numa, dena, numy, deny, x, m):
    """
    Helper function to find a polynomial solution
    of degree m for the auxiliary differential
    equation.
    """
    # Assume that the solution is of the type
    # p(x) = C_0 + C_1*x + ... + C_{m-1}*x**(m-1) + x**m
    psyms = symbols(f'C0:{m}', cls=Dummy)
    K = ZZ[psyms]
    psol = Poly(K.gens, x, domain=K) + Poly(x**m, x, domain=K)

    # Eq (5.16) in Thesis - Pg 81
    auxeq = (dena*(numy.diff(x)*deny - numy*deny.diff(x) + numy**2) - numa*deny**2)*psol
    if m >= 1:
        px = psol.diff(x)
        auxeq += px*(2*numy*deny*dena)
    if m >= 2:
        auxeq += px.diff(x)*(deny**2*dena)
    if m != 0:
        # m is a non-zero integer. Find the constant terms using undetermined coefficients
        return psol, linsolve_dict(auxeq.all_coeffs(), psyms), True
    else:
        # m == 0 . Check if 1 (x**0) is a solution to the auxiliary equation
        return S.One, auxeq, auxeq == 0


def remove_redundant_sols(sol1, sol2, x):
    """
    Helper function to remove redundant
    solutions to the differential equation.
    """
    # If y1 and y2 are redundant solutions, there is
    # some value of the arbitrary constant for which
    # they will be equal

    syms1 = sol1.atoms(Symbol, Dummy)
    syms2 = sol2.atoms(Symbol, Dummy)
    num1, den1 = [Poly(e, x, extension=True) for e in sol1.together().as_numer_denom()]
    num2, den2 = [Poly(e, x, extension=True) for e in sol2.together().as_numer_denom()]
    # Cross multiply
    e = num1*den2 - den1*num2
    # Check if there are any constants
    syms = list(e.atoms(Symbol, Dummy))
    if len(syms):
        # Find values of constants for which solutions are equal
        redn = linsolve(e.all_coeffs(), syms)
        if len(redn):
            # Return the general solution over a particular solution
            if len(syms1) > len(syms2):
                return sol2
            # If both have constants, return the lesser complex solution
            elif len(syms1) == len(syms2):
                return sol1 if count_ops(syms1) >= count_ops(syms2) else sol2
            else:
                return sol1


def get_gen_sol_from_part_sol(part_sols, a, x):
    """"
    Helper function which computes the general
    solution for a Riccati ODE from its particular
    solutions.

    There are 3 cases to find the general solution
    from the particular solutions for a Riccati ODE
    depending on the number of particular solution(s)
    we have - 1, 2 or 3.

    For more information, see Section 6 of
    "Methods of Solution of the Riccati Differential Equation"
    by D. R. Haaheim and F. M. Stein
    """

    # If no particular solutions are found, a general
    # solution cannot be found
    if len(part_sols) == 0:
        return []

    # In case of a single particular solution, the general
    # solution can be found by using the substitution
    # y = y1 + 1/z and solving a Bernoulli ODE to find z.
    elif len(part_sols) == 1:
        y1 = part_sols[0]
        i = exp(Integral(2*y1, x))
        z = i * Integral(a/i, x)
        z = z.doit()
        if a == 0 or z == 0:
            return y1
        return y1 + 1/z

    # In case of 2 particular solutions, the general solution
    # can be found by solving a separable equation. This is
    # the most common case, i.e. most Riccati ODEs have 2
    # rational particular solutions.
    elif len(part_sols) == 2:
        y1, y2 = part_sols
        # One of them already has a constant
        if len(y1.atoms(Dummy)) + len(y2.atoms(Dummy)) > 0:
            u = exp(Integral(y2 - y1, x)).doit()
        # Introduce a constant
        else:
            C1 = Dummy('C1')
            u = C1*exp(Integral(y2 - y1, x)).doit()
        if u == 1:
            return y2
        return (y2*u - y1)/(u - 1)

    # In case of 3 particular solutions, a closed form
    # of the general solution can be obtained directly
    else:
        y1, y2, y3 = part_sols[:3]
        C1 = Dummy('C1')
        return (C1 + 1)*y2*(y1 - y3)/(C1*y1 + y2 - (C1 + 1)*y3)


def solve_riccati(fx, x, b0, b1, b2, gensol=False):
    """
    The main function that gives particular/general
    solutions to Riccati ODEs that have atleast 1
    rational particular solution.
    """
    # Step 1 : Convert to Normal Form
    a = -b0*b2 + b1**2/4 - b1.diff(x)/2 + 3*b2.diff(x)**2/(4*b2**2) + b1*b2.diff(x)/(2*b2) - \
        b2.diff(x, 2)/(2*b2)
    a_t = a.together()
    num, den = [Poly(e, x, extension=True) for e in a_t.as_numer_denom()]
    num, den = num.cancel(den, include=True)

    # Step 2
    presol = []

    # Step 3 : a(x) is 0
    if num == 0:
        presol.append(1/(x + Dummy('C1')))

    # Step 4 : a(x) is a non-zero constant
    elif x not in num.free_symbols.union(den.free_symbols):
        presol.extend([sqrt(a), -sqrt(a)])

    # Step 5 : Find poles and valuation at infinity
    poles = roots(den, x)
    poles, muls = list(poles.keys()), list(poles.values())
    val_inf = val_at_inf(num, den, x)

    if len(poles):
        # Check necessary conditions (outlined in the module docstring)
        if not check_necessary_conds(val_inf, muls):
            raise ValueError("Rational Solution doesn't exist")

        # Step 6
        # Construct c-vectors for each singular point
        c = construct_c(num, den, x, poles, muls)

        # Construct d vectors for each singular point
        d = construct_d(num, den, x, val_inf)

        # Step 7 : Iterate over all possible combinations and return solutions
        # For each possible combination, generate an array of 0's and 1's
        # where 0 means pick 1st choice and 1 means pick the second choice.

        # NOTE: We could exit from the loop if we find 3 particular solutions,
        # but it is not implemented here as -
        #   a. Finding 3 particular solutions is very rare. Most of the time,
        #      only 2 particular solutions are found.
        #   b. In case we exit after finding 3 particular solutions, it might
        #      happen that 1 or 2 of them are redundant solutions. So, instead of
        #      spending some more time in computing the particular solutions,
        #      we will end up computing the general solution from a single
        #      particular solution which is usually slower than computing the
        #      general solution from 2 or 3 particular solutions.
        c.append(d)
        choices = product(*c)
        for choice in choices:
            m, ybar = compute_m_ybar(x, poles, choice, -val_inf//2)
            numy, deny = [Poly(e, x, extension=True) for e in ybar.together().as_numer_denom()]
            # Step 10 : Check if a valid solution exists. If yes, also check
            # if m is a non-negative integer
            if m.is_nonnegative == True and m.is_integer == True:

                # Step 11 : Find polynomial solutions of degree m for the auxiliary equation
                psol, coeffs, exists = solve_aux_eq(num, den, numy, deny, x, m)

                # Step 12 : If valid polynomial solution exists, append solution.
                if exists:
                    # m == 0 case
                    if psol == 1 and coeffs == 0:
                        # p(x) = 1, so p'(x)/p(x) term need not be added
                        presol.append(ybar)
                    # m is a positive integer and there are valid coefficients
                    elif len(coeffs):
                        # Substitute the valid coefficients to get p(x)
                        psol = psol.xreplace(coeffs)
                        # y(x) = ybar(x) + p'(x)/p(x)
                        presol.append(ybar + psol.diff(x)/psol)

    # Remove redundant solutions from the list of existing solutions
    remove = set()
    for i in range(len(presol)):
        for j in range(i+1, len(presol)):
            rem = remove_redundant_sols(presol[i], presol[j], x)
            if rem is not None:
                remove.add(rem)
    sols = [x for x in presol if x not in remove]

    # Step 15 : Inverse transform the solutions of the equation in normal form
    bp = -b2.diff(x)/(2*b2**2) - b1/(2*b2)

    # If general solution is required, compute it from the particular solutions
    if gensol:
        sols = [get_gen_sol_from_part_sol(sols, a, x)]

    # Inverse transform the particular solutions
    presol = [Eq(fx, riccati_inverse_normal(y, x, b1, b2, bp).cancel(extension=True)) for y in sols]
    return presol
<|endoftext|>r"""
This File contains helper functions for nth_linear_constant_coeff_undetermined_coefficients,
nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients,
nth_linear_constant_coeff_variation_of_parameters,
and nth_linear_euler_eq_nonhomogeneous_variation_of_parameters.

All the functions in this file are used by more than one solvers so, instead of creating
instances in other classes for using them it is better to keep it here as separate helpers.

"""
from collections import Counter
from sympy.core import Add, S
from sympy.core.function import diff, expand, _mexpand, expand_mul
from sympy.core.relational import Eq
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Wild
from sympy.functions import exp, cos, cosh, im, log, re, sin, sinh, \
    atan2, conjugate
from sympy.integrals import Integral
from sympy.polys import (Poly, RootOf, rootof, roots)
from sympy.simplify import collect, simplify, separatevars, powsimp, trigsimp # type: ignore
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.matrices import wronskian
from .subscheck import sub_func_doit
from sympy.solvers.ode.ode import get_numbered_constants


def _test_term(coeff, func, order):
    r"""
    Linear Euler ODEs have the form  K*x**order*diff(y(x), x, order) = F(x),
    where K is independent of x and y(x), order>= 0.
    So we need to check that for each term, coeff == K*x**order from
    some K.  We have a few cases, since coeff may have several
    different types.
    """
    x = func.args[0]
    f = func.func
    if order < 0:
        raise ValueError("order should be greater than 0")
    if coeff == 0:
        return True
    if order == 0:
        if x in coeff.free_symbols:
            return False
        return True
    if coeff.is_Mul:
        if coeff.has(f(x)):
            return False
        return x**order in coeff.args
    elif coeff.is_Pow:
        return coeff.as_base_exp() == (x, order)
    elif order == 1:
        return x == coeff
    return False


def _get_euler_characteristic_eq_sols(eq, func, match_obj):
    r"""
    Returns the solution of homogeneous part of the linear euler ODE and
    the list of roots of characteristic equation.

    The parameter ``match_obj`` is a dict of order:coeff terms, where order is the order
    of the derivative on each term, and coeff is the coefficient of that derivative.

    """
    x = func.args[0]
    f = func.func

    # First, set up characteristic equation.
    chareq, symbol = S.Zero, Dummy('x')

    for i in match_obj:
        if i >= 0:
            chareq += (match_obj[i]*diff(x**symbol, x, i)*x**-symbol).expand()

    chareq = Poly(chareq, symbol)
    chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]
    collectterms = []

    # A generator of constants
    constants = list(get_numbered_constants(eq, num=chareq.degree()*2))
    constants.reverse()

    # Create a dict root: multiplicity or charroots
    charroots = Counter(chareqroots)
    gsol = S.Zero
    ln = log
    for root, multiplicity in charroots.items():
        for i in range(multiplicity):
            if isinstance(root, RootOf):
                gsol += (x**root) * constants.pop()
                if multiplicity != 1:
                    raise ValueError("Value should be 1")
                collectterms = [(0, root, 0)] + collectterms
            elif root.is_real:
                gsol += ln(x)**i*(x**root) * constants.pop()
                collectterms = [(i, root, 0)] + collectterms
            else:
                reroot = re(root)
                imroot = im(root)
                gsol += ln(x)**i * (x**reroot) * (
                    constants.pop() * sin(abs(imroot)*ln(x))
                    + constants.pop() * cos(imroot*ln(x)))
                collectterms = [(i, reroot, imroot)] + collectterms

    gsol = Eq(f(x), gsol)

    gensols = []
    # Keep track of when to use sin or cos for nonzero imroot
    for i, reroot, imroot in collectterms:
        if imroot == 0:
            gensols.append(ln(x)**i*x**reroot)
        else:
            sin_form = ln(x)**i*x**reroot*sin(abs(imroot)*ln(x))
            if sin_form in gensols:
                cos_form = ln(x)**i*x**reroot*cos(imroot*ln(x))
                gensols.append(cos_form)
            else:
                gensols.append(sin_form)
    return gsol, gensols


def _solve_variation_of_parameters(eq, func, roots, homogen_sol, order, match_obj, simplify_flag=True):
    r"""
    Helper function for the method of variation of parameters and nonhomogeneous euler eq.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffVariationOfParameters`
    docstring for more information on this method.

    The parameter are ``match_obj`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
    f = func.func
    x = func.args[0]
    r = match_obj
    psol = 0
    wr = wronskian(roots, x)

    if simplify_flag:
        wr = simplify(wr)  # We need much better simplification for
                        # some ODEs. See issue 4662, for example.
        # To reduce commonly occurring sin(x)**2 + cos(x)**2 to 1
        wr = trigsimp(wr, deep=True, recursive=True)
    if not wr:
        # The wronskian will be 0 iff the solutions are not linearly
        # independent.
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply " +
        "variation of parameters to " + str(eq) + " (Wronskian == 0)")
    if len(roots) != order:
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply " +
        "variation of parameters to " +
        str(eq) + " (number of terms != order)")
    negoneterm = S.NegativeOne**(order)
    for i in roots:
        psol += negoneterm*Integral(wronskian([sol for sol in roots if sol != i], x)*r[-1]/wr, x)*i/r[order]
        negoneterm *= -1

    if simplify_flag:
        psol = simplify(psol)
        psol = trigsimp(psol, deep=True)
    return Eq(f(x), homogen_sol.rhs + psol)


def _get_const_characteristic_eq_sols(r, func, order):
    r"""
    Returns the roots of characteristic equation of constant coefficient
    linear ODE and list of collectterms which is later on used by simplification
    to use collect on solution.

    The parameter `r` is a dict of order:coeff terms, where order is the order of the
    derivative on each term, and coeff is the coefficient of that derivative.

    """
    x = func.args[0]
    # First, set up characteristic equation.
    chareq, symbol = S.Zero, Dummy('x')

    for i in r.keys():
        if isinstance(i, str) or i < 0:
            pass
        else:
            chareq += r[i]*symbol**i

    chareq = Poly(chareq, symbol)
    # Can't just call roots because it doesn't return rootof for unsolveable
    # polynomials.
    chareqroots = roots(chareq, multiple=True)
    if len(chareqroots) != order:
        chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]

    chareq_is_complex = not all(i.is_real for i in chareq.all_coeffs())

    # Create a dict root: multiplicity or charroots
    charroots = Counter(chareqroots)
    # We need to keep track of terms so we can run collect() at the end.
    # This is necessary for constantsimp to work properly.
    collectterms = []
    gensols = []
    conjugate_roots = [] # used to prevent double-use of conjugate roots
    # Loop over roots in theorder provided by roots/rootof...
    for root in chareqroots:
        # but don't repoeat multiple roots.
        if root not in charroots:
            continue
        multiplicity = charroots.pop(root)
        for i in range(multiplicity):
            if chareq_is_complex:
                gensols.append(x**i*exp(root*x))
                collectterms = [(i, root, 0)] + collectterms
                continue
            reroot = re(root)
            imroot = im(root)
            if imroot.has(atan2) and reroot.has(atan2):
                # Remove this condition when re and im stop returning
                # circular atan2 usages.
                gensols.append(x**i*exp(root*x))
                collectterms = [(i, root, 0)] + collectterms
            else:
                if root in conjugate_roots:
                    collectterms = [(i, reroot, imroot)] + collectterms
                    continue
                if imroot == 0:
                    gensols.append(x**i*exp(reroot*x))
                    collectterms = [(i, reroot, 0)] + collectterms
                    continue
                conjugate_roots.append(conjugate(root))
                gensols.append(x**i*exp(reroot*x) * sin(abs(imroot) * x))
                gensols.append(x**i*exp(reroot*x) * cos(    imroot  * x))

                # This ordering is important
                collectterms = [(i, reroot, imroot)] + collectterms
    return gensols, collectterms


# Ideally these kind of simplification functions shouldn't be part of solvers.
# odesimp should be improved to handle these kind of specific simplifications.
def _get_simplified_sol(sol, func, collectterms):
    r"""
    Helper function which collects the solution on
    collectterms. Ideally this should be handled by odesimp.It is used
    only when the simplify is set to True in dsolve.

    The parameter ``collectterms`` is a list of tuple (i, reroot, imroot) where `i` is
    the multiplicity of the root, reroot is real part and imroot being the imaginary part.

    """
    f = func.func
    x = func.args[0]
    collectterms.sort(key=default_sort_key)
    collectterms.reverse()
    assert len(sol) == 1 and sol[0].lhs == f(x)
    sol = sol[0].rhs
    sol = expand_mul(sol)
    for i, reroot, imroot in collectterms:
        sol = collect(sol, x**i*exp(reroot*x)*sin(abs(imroot)*x))
        sol = collect(sol, x**i*exp(reroot*x)*cos(imroot*x))
    for i, reroot, imroot in collectterms:
        sol = collect(sol, x**i*exp(reroot*x))
    sol = powsimp(sol)
    return Eq(f(x), sol)


def _undetermined_coefficients_match(expr, x, func=None, eq_homogeneous=S.Zero):
    r"""
    Returns a trial function match if undetermined coefficients can be applied
    to ``expr``, and ``None`` otherwise.

    A trial expression can be found for an expression for use with the method
    of undetermined coefficients if the expression is an
    additive/multiplicative combination of constants, polynomials in `x` (the
    independent variable of expr), `\sin(a x + b)`, `\cos(a x + b)`, and
    `e^{a x}` terms (in other words, it has a finite number of linearly
    independent derivatives).

    Note that you may still need to multiply each term returned here by
    sufficient `x` to make it linearly independent with the solutions to the
    homogeneous equation.

    This is intended for internal use by ``undetermined_coefficients`` hints.

    SymPy currently has no way to convert `\sin^n(x) \cos^m(y)` into a sum of
    only `\sin(a x)` and `\cos(b x)` terms, so these are not implemented.  So,
    for example, you will need to manually convert `\sin^2(x)` into `[1 +
    \cos(2 x)]/2` to properly apply the method of undetermined coefficients on
    it.

    Examples
    ========

    >>> from sympy import log, exp
    >>> from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
    >>> from sympy.abc import x
    >>> _undetermined_coefficients_match(9*x*exp(x) + exp(-x), x)
    {'test': True, 'trialset': {x*exp(x), exp(-x), exp(x)}}
    >>> _undetermined_coefficients_match(log(x), x)
    {'test': False}

    """
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    expr = powsimp(expr, combine='exp')  # exp(x)*exp(2*x + 1) => exp(3*x + 1)
    retdict = {}

    def _test_term(expr, x) -> bool:
        r"""
        Test if ``expr`` fits the proper form for undetermined coefficients.
        """
        if not expr.has(x):
            return True
        if expr.is_Add:
            return all(_test_term(i, x) for i in expr.args)
        if expr.is_Mul:
            if expr.has(sin, cos):
                foundtrig = False
                # Make sure that there is only one trig function in the args.
                # See the docstring.
                for i in expr.args:
                    if i.has(sin, cos):
                        if foundtrig:
                            return False
                        else:
                            foundtrig = True
            return all(_test_term(i, x) for i in expr.args)
        if expr.is_Function:
            return expr.func in (sin, cos, exp, sinh, cosh) and \
                   bool(expr.args[0].match(a*x + b))
        if expr.is_Pow and expr.base.is_Symbol and expr.exp.is_Integer and \
                expr.exp >= 0:
            return True
        if expr.is_Pow and expr.base.is_number:
            return bool(expr.exp.match(a*x + b))
        return expr.is_Symbol or bool(expr.is_number)

    def _get_trial_set(expr, x, exprs=set()):
        r"""
        Returns a set of trial terms for undetermined coefficients.

        The idea behind undetermined coefficients is that the terms expression
        repeat themselves after a finite number of derivatives, except for the
        coefficients (they are linearly dependent).  So if we collect these,
        we should have the terms of our trial function.
        """
        def _remove_coefficient(expr, x):
            r"""
            Returns the expression without a coefficient.

            Similar to expr.as_independent(x)[1], except it only works
            multiplicatively.
            """
            term = S.One
            if expr.is_Mul:
                for i in expr.args:
                    if i.has(x):
                        term *= i
            elif expr.has(x):
                term = expr
            return term

        expr = expand_mul(expr)
        if expr.is_Add:
            for term in expr.args:
                if _remove_coefficient(term, x) in exprs:
                    pass
                else:
                    exprs.add(_remove_coefficient(term, x))
                    exprs = exprs.union(_get_trial_set(term, x, exprs))
        else:
            term = _remove_coefficient(expr, x)
            tmpset = exprs.union({term})
            oldset = set()
            while tmpset != oldset:
                # If you get stuck in this loop, then _test_term is probably
                # broken
                oldset = tmpset.copy()
                expr = expr.diff(x)
                term = _remove_coefficient(expr, x)
                if term.is_Add:
                    tmpset = tmpset.union(_get_trial_set(term, x, tmpset))
                else:
                    tmpset.add(term)
            exprs = tmpset
        return exprs

    def is_homogeneous_solution(term):
        r""" This function checks whether the given trialset contains any root
            of homogeneous equation"""
        return expand(sub_func_doit(eq_homogeneous, func, term)).is_zero

    retdict['test'] = _test_term(expr, x)
    if retdict['test']:
        # Try to generate a list of trial solutions that will have the
        # undetermined coefficients. Note that if any of these are not linearly
        # independent with any of the solutions to the homogeneous equation,
        # then they will need to be multiplied by sufficient x to make them so.
        # This function DOES NOT do that (it doesn't even look at the
        # homogeneous equation).
        temp_set = set()
        for i in Add.make_args(expr):
            act = _get_trial_set(i, x)
            if eq_homogeneous is not S.Zero:
                while any(is_homogeneous_solution(ts) for ts in act):
                    act = {x*ts for ts in act}
            temp_set = temp_set.union(act)

        retdict['trialset'] = temp_set
    return retdict


def _solve_undetermined_coefficients(eq, func, order, match, trialset):
    r"""
    Helper function for the method of undetermined coefficients.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffUndeterminedCoefficients`
    docstring for more information on this method.

    The parameter ``trialset`` is the set of trial functions as returned by
    ``_undetermined_coefficients_match()['trialset']``.

    The parameter ``match`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
    r = match
    coeffs = numbered_symbols('a', cls=Dummy)
    coefflist = []
    gensols = r['list']
    gsol = r['sol']
    f = func.func
    x = func.args[0]

    if len(gensols) != order:
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply" +
        " undetermined coefficients to " + str(eq) +
        " (number of terms != order)")

    trialfunc = 0
    for i in trialset:
        c = next(coeffs)
        coefflist.append(c)
        trialfunc += c*i

    eqs = sub_func_doit(eq, f(x), trialfunc)

    coeffsdict = dict(list(zip(trialset, [0]*(len(trialset) + 1))))

    eqs = _mexpand(eqs)

    for i in Add.make_args(eqs):
        s = separatevars(i, dict=True, symbols=[x])
        if coeffsdict.get(s[x]):
            coeffsdict[s[x]] += s['coeff']
        else:
            coeffsdict[s[x]] = s['coeff']

    coeffvals = solve(list(coeffsdict.values()), coefflist)

    if not coeffvals:
        raise NotImplementedError(
            "Could not solve `%s` using the "
            "method of undetermined coefficients "
            "(unable to solve for coefficients)." % eq)

    psol = trialfunc.subs(coeffvals)

    return Eq(f(x), gsol.rhs + psol)
<|endoftext|>r'''
This module contains the implementation of the 2nd_hypergeometric hint for
dsolve. This is an incomplete implementation of the algorithm described in [1].
The algorithm solves 2nd order linear ODEs of the form

.. math:: y'' + A(x) y' + B(x) y = 0\text{,}

where `A` and `B` are rational functions. The algorithm should find any
solution of the form

.. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
Currently only the 2F1 case is implemented in SymPy but the other cases are
described in the paper and could be implemented in future (contributions
welcome!).

References
==========

.. [1] L. Chan, E.S. Cheb-Terrab, Non-Liouvillian solutions for second order
       linear ODEs, (2004).
       https://arxiv.org/abs/math-ph/0402063
'''

from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants


def match_2nd_hypergeometric(eq, func):
    x = func.args[0]
    df = func.diff(x)
    a3 = Wild('a3', exclude=[func, func.diff(x), func.diff(x, 2)])
    b3 = Wild('b3', exclude=[func, func.diff(x), func.diff(x, 2)])
    c3 = Wild('c3', exclude=[func, func.diff(x), func.diff(x, 2)])
    deq = a3*(func.diff(x, 2)) + b3*df + c3*func
    r = collect(eq,
        [func.diff(x, 2), func.diff(x), func]).match(deq)
    if r:
        if not all(val.is_polynomial() for val in r.values()):
            n, d = eq.as_numer_denom()
            eq = expand(n)
            r = collect(eq, [func.diff(x, 2), func.diff(x), func]).match(deq)

    if r and r[a3]!=0:
        A = cancel(r[b3]/r[a3])
        B = cancel(r[c3]/r[a3])
        return [A, B]
    else:
        return []


def equivalence_hypergeometric(A, B, func):
    # This method for finding the equivalence is only for 2F1 type.
    # We can extend it for 1F1 and 0F1 type also.
    x = func.args[0]

    # making given equation in normal form
    I1 = factor(cancel(A.diff(x)/2 + A**2/4 - B))

    # computing shifted invariant(J1) of the equation
    J1 = factor(cancel(x**2*I1 + S(1)/4))
    num, dem = J1.as_numer_denom()
    num = powdenest(expand(num))
    dem = powdenest(expand(dem))
    # this function will compute the different powers of variable(x) in J1.
    # then it will help in finding value of k. k is power of x such that we can express
    # J1 = x**k * J0(x**k) then all the powers in J0 become integers.
    def _power_counting(num):
        _pow = {0}
        for val in num:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    _pow.add(val.as_base_exp()[1])
                elif val == x:
                    _pow.add(val.as_base_exp()[1])
                else:
                    _pow.update(_power_counting(val.args))
        return _pow

    pow_num = _power_counting((num, ))
    pow_dem = _power_counting((dem, ))
    pow_dem.update(pow_num)

    _pow = pow_dem
    k = gcd(_pow)

    # computing I0 of the given equation
    I0 = powdenest(simplify(factor(((J1/k**2) - S(1)/4)/((x**k)**2))), force=True)
    I0 = factor(cancel(powdenest(I0.subs(x, x**(S(1)/k)), force=True)))

    # Before this point I0, J1 might be functions of e.g. sqrt(x) but replacing
    # x with x**(1/k) should result in I0 being a rational function of x or
    # otherwise the hypergeometric solver cannot be used. Note that k can be a
    # non-integer rational such as 2/7.
    if not I0.is_rational_function(x):
        return None

    num, dem = I0.as_numer_denom()

    max_num_pow = max(_power_counting((num, )))
    dem_args = dem.args
    sing_point = []
    dem_pow = []
    # calculating singular point of I0.
    for arg in dem_args:
        if arg.has(x):
            if isinstance(arg, Pow):
                # (x-a)**n
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg.as_base_exp()[0], x).keys())[0])
            else:
                # (x-a) type
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg, x).keys())[0])

    dem_pow.sort()
    # checking if equivalence is exists or not.

    if equivalence(max_num_pow, dem_pow) == "2F1":
        return {'I0':I0, 'k':k, 'sing_point':sing_point, 'type':"2F1"}
    else:
        return None


def match_2nd_2F1_hypergeometric(I, k, sing_point, func):
    x = func.args[0]
    a = Wild("a")
    b = Wild("b")
    c = Wild("c")
    t = Wild("t")
    s = Wild("s")
    r = Wild("r")
    alpha = Wild("alpha")
    beta = Wild("beta")
    gamma = Wild("gamma")
    delta = Wild("delta")
    # I0 of the standard 2F1 equation.
    I0 = ((a-b+1)*(a-b-1)*x**2 + 2*((1-a-b)*c + 2*a*b)*x + c*(c-2))/(4*x**2*(x-1)**2)
    if sing_point != [0, 1]:
        # If singular point is [0, 1] then we have standard equation.
        eqs = []
        sing_eqs = [-beta/alpha, -delta/gamma, (delta-beta)/(alpha-gamma)]
        # making equations for the finding the mobius transformation
        for i in range(3):
            if i<len(sing_point):
                eqs.append(Eq(sing_eqs[i], sing_point[i]))
            else:
                eqs.append(Eq(1/sing_eqs[i], 0))
        # solving above equations for the mobius transformation
        _beta = -alpha*sing_point[0]
        _delta = -gamma*sing_point[1]
        _gamma = alpha
        if len(sing_point) == 3:
            _gamma = (_beta + sing_point[2]*alpha)/(sing_point[2] - sing_point[1])
        mob = (alpha*x + beta)/(gamma*x + delta)
        mob = mob.subs(beta, _beta)
        mob = mob.subs(delta, _delta)
        mob = mob.subs(gamma, _gamma)
        mob = cancel(mob)
        t = (beta - delta*x)/(gamma*x - alpha)
        t = cancel(((t.subs(beta, _beta)).subs(delta, _delta)).subs(gamma, _gamma))
    else:
        mob = x
        t = x

    # applying mobius transformation in I to make it into I0.
    I = I.subs(x, t)
    I = I*(t.diff(x))**2
    I = factor(I)
    dict_I = {x**2:0, x:0, 1:0}
    I0_num, I0_dem = I0.as_numer_denom()
    # collecting coeff of (x**2, x), of the standard equation.
    # substituting (a-b) = s, (a+b) = r
    dict_I0 = {x**2:s**2 - 1, x:(2*(1-r)*c + (r+s)*(r-s)), 1:c*(c-2)}
    # collecting coeff of (x**2, x) from I0 of the given equation.
    dict_I.update(collect(expand(cancel(I*I0_dem)), [x**2, x], evaluate=False))
    eqs = []
    # We are comparing the coeff of powers of different x, for finding the values of
    # parameters of standard equation.
    for key in [x**2, x, 1]:
        eqs.append(Eq(dict_I[key], dict_I0[key]))

    # We can have many possible roots for the equation.
    # I am selecting the root on the basis that when we have
    # standard equation eq = x*(x-1)*f(x).diff(x, 2) + ((a+b+1)*x-c)*f(x).diff(x) + a*b*f(x)
    # then root should be a, b, c.

    _c = 1 - factor(sqrt(1+eqs[2].lhs))
    if not _c.has(Symbol):
        _c = min(list(roots(eqs[2], c)))
    _s = factor(sqrt(eqs[0].lhs + 1))
    _r = _c - factor(sqrt(_c**2 + _s**2 + eqs[1].lhs - 2*_c))
    _a = (_r + _s)/2
    _b = (_r - _s)/2

    rn = {'a':simplify(_a), 'b':simplify(_b), 'c':simplify(_c), 'k':k, 'mobius':mob, 'type':"2F1"}
    return rn


def equivalence(max_num_pow, dem_pow):
    # this function is made for checking the equivalence with 2F1 type of equation.
    # max_num_pow is the value of maximum power of x in numerator
    # and dem_pow is list of powers of different factor of form (a*x b).
    # reference from table 1 in paper - "Non-Liouvillian solutions for second order
    # linear ODEs" by L. Chan, E.S. Cheb-Terrab.
    # We can extend it for 1F1 and 0F1 type also.

    if max_num_pow == 2:
        if dem_pow in [[2, 2], [2, 2, 2]]:
            return "2F1"
    elif max_num_pow == 1:
        if dem_pow in [[1, 2, 2], [2, 2, 2], [1, 2], [2, 2]]:
            return "2F1"
    elif max_num_pow == 0:
        if dem_pow in [[1, 1, 2], [2, 2], [1, 2, 2], [1, 1], [2], [1, 2], [2, 2]]:
            return "2F1"

    return None


def get_sol_2F1_hypergeometric(eq, func, match_object):
    x = func.args[0]
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.polys.polytools import factor
    C0, C1 = get_numbered_constants(eq, num=2)
    a = match_object['a']
    b = match_object['b']
    c = match_object['c']
    A = match_object['A']

    sol = None

    if c.is_integer == False:
        sol = C0*hyper([a, b], [c], x) + C1*hyper([a-c+1, b-c+1], [2-c], x)*x**(1-c)
    elif c == 1:
        y2 = Integral(exp(Integral((-(a+b+1)*x + c)/(x**2-x), x))/(hyperexpand(hyper([a, b], [c], x))**2), x)*hyper([a, b], [c], x)
        sol = C0*hyper([a, b], [c], x) + C1*y2
    elif (c-a-b).is_integer == False:
        sol = C0*hyper([a, b], [1+a+b-c], 1-x) + C1*hyper([c-a, c-b], [1+c-a-b], 1-x)*(1-x)**(c-a-b)

    if sol:
        # applying transformation in the solution
        subs = match_object['mobius']
        dtdx = simplify(1/(subs.diff(x)))
        _B = ((a + b + 1)*x - c).subs(x, subs)*dtdx
        _B = factor(_B + ((x**2 -x).subs(x, subs))*(dtdx.diff(x)*dtdx))
        _A = factor((x**2 - x).subs(x, subs)*(dtdx**2))
        e = exp(logcombine(Integral(cancel(_B/(2*_A)), x), force=True))
        sol = sol.subs(x, match_object['mobius'])
        sol = sol.subs(x, x**match_object['k'])
        e = e.subs(x, x**match_object['k'])

        if not A.is_zero:
            e1 = Integral(A/2, x)
            e1 = exp(logcombine(e1, force=True))
            sol = cancel((e/e1)*x**((-match_object['k']+1)/2))*sol
            sol = Eq(func, sol)
            return sol

        sol = cancel((e)*x**((-match_object['k']+1)/2))*sol
        sol = Eq(func, sol)
    return sol
<|endoftext|>#
# This is the module for ODE solver classes for single ODEs.
#

from __future__ import annotations
from typing import ClassVar, Iterator

from .riccati import match_riccati, solve_riccati
from sympy.core import Add, S, Pow, Rational
from sympy.core.cache import cached_property
from sympy.core.exprtools import factor_terms
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, diff, Function, expand, Subs, _mexpand
from sympy.core.numbers import zoo
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Dummy, Wild
from sympy.core.mul import Mul
from sympy.functions import exp, tan, log, sqrt, besselj, bessely, cbrt, airyai, airybi
from sympy.integrals import Integral
from sympy.polys import Poly
from sympy.polys.polytools import cancel, factor, degree
from sympy.simplify import collect, simplify, separatevars, logcombine, posify # type: ignore
from sympy.simplify.radsimp import fraction
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.solvers.deutils import ode_order, _preprocess
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.solvers import PolyNonlinearError
from .hypergeometric import equivalence_hypergeometric, match_2nd_2F1_hypergeometric, \
    get_sol_2F1_hypergeometric, match_2nd_hypergeometric
from .nonhomogeneous import _get_euler_characteristic_eq_sols, _get_const_characteristic_eq_sols, \
    _solve_undetermined_coefficients, _solve_variation_of_parameters, _test_term, _undetermined_coefficients_match, \
        _get_simplified_sol
from .lie_group import _ode_lie_group


class ODEMatchError(NotImplementedError):
    """Raised if a SingleODESolver is asked to solve an ODE it does not match"""
    pass


class SingleODEProblem:
    """Represents an ordinary differential equation (ODE)

    This class is used internally in the by dsolve and related
    functions/classes so that properties of an ODE can be computed
    efficiently.

    Examples
    ========

    This class is used internally by dsolve. To instantiate an instance
    directly first define an ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now you can create a SingleODEProblem instance and query its properties:

    >>> from sympy.solvers.ode.single import SingleODEProblem
    >>> problem = SingleODEProblem(f(x).diff(x), f(x), x)
    >>> problem.eq
    Derivative(f(x), x)
    >>> problem.func
    f(x)
    >>> problem.sym
    x
    """

    # Instance attributes:
    eq: Expr
    func: AppliedUndef
    sym: Symbol
    _order: int
    _eq_expanded: Expr
    _eq_preprocessed: Expr
    _eq_high_order_free = None

    def __init__(self, eq, func, sym, prep=True, **kwargs):
        assert isinstance(eq, Expr)
        assert isinstance(func, AppliedUndef)
        assert isinstance(sym, Symbol)
        assert isinstance(prep, bool)
        self.eq = eq
        self.func = func
        self.sym = sym
        self.prep = prep
        self.params = kwargs

    @cached_property
    def order(self) -> int:
        return ode_order(self.eq, self.func)

    @cached_property
    def eq_preprocessed(self) -> Expr:
        return self._get_eq_preprocessed()

    @cached_property
    def eq_high_order_free(self) -> Expr:
        a = Wild('a', exclude=[self.func])
        c1 = Wild('c1', exclude=[self.sym])
        # Precondition to try remove f(x) from highest order derivative
        reduced_eq = None
        if self.eq.is_Add:
            deriv_coef = self.eq.coeff(self.func.diff(self.sym, self.order))
            if deriv_coef not in (1, 0):
                r = deriv_coef.match(a*self.func**c1)
                if r and r[c1]:
                    den = self.func**r[c1]
                    reduced_eq = Add(*[arg/den for arg in self.eq.args])
        if reduced_eq is None:
            reduced_eq = expand(self.eq)
        return reduced_eq

    @cached_property
    def eq_expanded(self) -> Expr:
        return expand(self.eq_preprocessed)

    def _get_eq_preprocessed(self) -> Expr:
        if self.prep:
            process_eq, process_func = _preprocess(self.eq, self.func)
            if process_func != self.func:
                raise ValueError
        else:
            process_eq = self.eq
        return process_eq

    def get_numbered_constants(self, num=1, start=1, prefix='C') -> list[Symbol]:
        """
        Returns a list of constants that do not occur
        in eq already.
        """
        ncs = self.iter_numbered_constants(start, prefix)
        Cs = [next(ncs) for i in range(num)]
        return Cs

    def iter_numbered_constants(self, start=1, prefix='C') -> Iterator[Symbol]:
        """
        Returns an iterator of constants that do not occur
        in eq already.
        """
        atom_set = self.eq.free_symbols
        func_set = self.eq.atoms(Function)
        if func_set:
            atom_set |= {Symbol(str(f.func)) for f in func_set}
        return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)

    @cached_property
    def is_autonomous(self):
        u = Dummy('u')
        x = self.sym
        syms = self.eq.subs(self.func, u).free_symbols
        return x not in syms

    def get_linear_coefficients(self, eq, func, order):
        r"""
        Matches a differential equation to the linear form:

        .. math:: a_n(x) y^{(n)} + \cdots + a_1(x)y' + a_0(x) y + B(x) = 0

        Returns a dict of order:coeff terms, where order is the order of the
        derivative on each term, and coeff is the coefficient of that derivative.
        The key ``-1`` holds the function `B(x)`. Returns ``None`` if the ODE is
        not linear.  This function assumes that ``func`` has already been checked
        to be good.

        Examples
        ========

        >>> from sympy import Function, cos, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import SingleODEProblem
        >>> f = Function('f')
        >>> eq = f(x).diff(x, 3) + 2*f(x).diff(x) + \
        ... x*f(x).diff(x, 2) + cos(x)*f(x).diff(x) + x - f(x) - \
        ... sin(x)
        >>> obj = SingleODEProblem(eq, f(x), x)
        >>> obj.get_linear_coefficients(eq, f(x), 3)
        {-1: x - sin(x), 0: -1, 1: cos(x) + 2, 2: x, 3: 1}
        >>> eq = f(x).diff(x, 3) + 2*f(x).diff(x) + \
        ... x*f(x).diff(x, 2) + cos(x)*f(x).diff(x) + x - f(x) - \
        ... sin(f(x))
        >>> obj = SingleODEProblem(eq, f(x), x)
        >>> obj.get_linear_coefficients(eq, f(x), 3) == None
        True

        """
        f = func.func
        x = func.args[0]
        symset = {Derivative(f(x), x, i) for i in range(order+1)}
        try:
            rhs, lhs_terms = _lin_eq2dict(eq, symset)
        except PolyNonlinearError:
            return None

        if rhs.has(func) or any(c.has(func) for c in lhs_terms.values()):
            return None
        terms = {i: lhs_terms.get(f(x).diff(x, i), S.Zero) for i in range(order+1)}
        terms[-1] = rhs
        return terms

    # TODO: Add methods that can be used by many ODE solvers:
    # order
    # is_linear()
    # get_linear_coefficients()
    # eq_prepared (the ODE in prepared form)


class SingleODESolver:
    """
    Base class for Single ODE solvers.

    Subclasses should implement the _matches and _get_general_solution
    methods. This class is not intended to be instantiated directly but its
    subclasses are as part of dsolve.

    Examples
    ========

    You can use a subclass of SingleODEProblem to solve a particular type of
    ODE. We first define a particular ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now we solve this problem using the NthAlgebraic solver which is a
    subclass of SingleODESolver:

    >>> from sympy.solvers.ode.single import NthAlgebraic, SingleODEProblem
    >>> problem = SingleODEProblem(eq, f(x), x)
    >>> solver = NthAlgebraic(problem)
    >>> solver.get_general_solution()
    [Eq(f(x), _C*x + _C)]

    The normal way to solve an ODE is to use dsolve (which would use
    NthAlgebraic and other solvers internally). When using dsolve a number of
    other things are done such as evaluating integrals, simplifying the
    solution and renumbering the constants:

    >>> from sympy import dsolve
    >>> dsolve(eq, hint='nth_algebraic')
    Eq(f(x), C1 + C2*x)
    """

    # Subclasses should store the hint name (the argument to dsolve) in this
    # attribute
    hint: ClassVar[str]

    # Subclasses should define this to indicate if they support an _Integral
    # hint.
    has_integral: ClassVar[bool]

    # The ODE to be solved
    ode_problem: SingleODEProblem

    # Cache whether or not the equation has matched the method
    _matched: bool | None = None

    # Subclasses should store in this attribute the list of order(s) of ODE
    # that subclass can solve or leave it to None if not specific to any order
    order: list | None = None

    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def matches(self) -> bool:
        if self.order is not None and self.ode_problem.order not in self.order:
            self._matched = False
            return self._matched

        if self._matched is None:
            self._matched = self._matches()
        return self._matched

    def get_general_solution(self, *, simplify: bool = True) -> list[Equality]:
        if not self.matches():
            msg = "%s solver cannot solve:\n%s"
            raise ODEMatchError(msg % (self.hint, self.ode_problem.eq))
        return self._get_general_solution(simplify_flag=simplify)

    def _matches(self) -> bool:
        msg = "Subclasses of SingleODESolver should implement matches."
        raise NotImplementedError(msg)

    def _get_general_solution(self, *, simplify_flag: bool = True) -> list[Equality]:
        msg = "Subclasses of SingleODESolver should implement get_general_solution."
        raise NotImplementedError(msg)


class SinglePatternODESolver(SingleODESolver):
    '''Superclass for ODE solvers based on pattern matching'''

    def wilds(self):
        prob = self.ode_problem
        f = prob.func.func
        x = prob.sym
        order = prob.order
        return self._wilds(f, x, order)

    def wilds_match(self):
        match = self._wilds_match
        return [match.get(w, S.Zero) for w in self.wilds()]

    def _matches(self):
        eq = self.ode_problem.eq_expanded
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        df = f(x).diff(x, order)

        if order not in [1, 2]:
            return False

        pattern = self._equation(f(x), x, order)

        if not pattern.coeff(df).has(Wild):
            eq = expand(eq / eq.coeff(df))
        eq = eq.collect([f(x).diff(x), f(x)], func = cancel)

        self._wilds_match = match = eq.match(pattern)
        if match is not None:
            return self._verify(f(x))
        return False

    def _verify(self, fx) -> bool:
        return True

    def _wilds(self, f, x, order):
        msg = "Subclasses of SingleODESolver should implement _wilds"
        raise NotImplementedError(msg)

    def _equation(self, fx, x, order):
        msg = "Subclasses of SingleODESolver should implement _equation"
        raise NotImplementedError(msg)


class NthAlgebraic(SingleODESolver):
    r"""
    Solves an `n`\th order ordinary differential equation using algebra and
    integrals.

    There is no general form for the kind of equation that this can solve. The
    the equation is solved algebraically treating differentiation as an
    invertible algebraic function.

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = Eq(f(x) * (f(x).diff(x)**2 - 1), 0)
    >>> dsolve(eq, f(x), hint='nth_algebraic')
    [Eq(f(x), 0), Eq(f(x), C1 - x), Eq(f(x), C1 + x)]

    Note that this solver can return algebraic solutions that do not have any
    integration constants (f(x) = 0 in the above example).
    """

    hint = 'nth_algebraic'
    has_integral = True  # nth_algebraic_Integral hint

    def _matches(self):
        r"""
        Matches any differential equation that nth_algebraic can solve. Uses
        `sympy.solve` but teaches it how to integrate derivatives.

        This involves calling `sympy.solve` and does most of the work of finding a
        solution (apart from evaluating the integrals).
        """
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        var = self.ode_problem.sym

        # Derivative that solve can handle:
        diffx = self._get_diffx(var)

        # Replace derivatives wrt the independent variable with diffx
        def replace(eq, var):
            def expand_diffx(*args):
                differand, diffs = args[0], args[1:]
                toreplace = differand
                for v, n in diffs:
                    for _ in range(n):
                        if v == var:
                            toreplace = diffx(toreplace)
                        else:
                            toreplace = Derivative(toreplace, v)
                return toreplace
            return eq.replace(Derivative, expand_diffx)

        # Restore derivatives in solution afterwards
        def unreplace(eq, var):
            return eq.replace(diffx, lambda e: Derivative(e, var))

        subs_eqn = replace(eq, var)
        try:
            # turn off simplification to protect Integrals that have
            # _t instead of fx in them and would otherwise factor
            # as t_*Integral(1, x)
            solns = solve(subs_eqn, func, simplify=False)
        except NotImplementedError:
            solns = []

        solns = [simplify(unreplace(soln, var)) for soln in solns]
        solns = [Equality(func, soln) for soln in solns]

        self.solutions = solns
        return len(solns) != 0

    def _get_general_solution(self, *, simplify_flag: bool = True):
        return self.solutions

    # This needs to produce an invertible function but the inverse depends
    # which variable we are integrating with respect to. Since the class can
    # be stored in cached results we need to ensure that we always get the
    # same class back for each particular integration variable so we store these
    # classes in a global dict:
    _diffx_stored: dict[Symbol, type[Function]] = {}

    @staticmethod
    def _get_diffx(var):
        diffcls = NthAlgebraic._diffx_stored.get(var, None)

        if diffcls is None:
            # A class that behaves like Derivative wrt var but is "invertible".
            class diffx(Function):
                def inverse(self):
                    # don't use integrate here because fx has been replaced by _t
                    # in the equation; integrals will not be correct while solve
                    # is at work.
                    return lambda expr: Integral(expr, var) + Dummy('C')

            diffcls = NthAlgebraic._diffx_stored.setdefault(var, diffx)

        return diffcls


class FirstExact(SinglePatternODESolver):
    r"""
    Solves 1st order exact ordinary differential equations.

    A 1st order differential equation is called exact if it is the total
    differential of a function. That is, the differential equation

    .. math:: P(x, y) \,\partial{}x + Q(x, y) \,\partial{}y = 0

    is exact if there is some function `F(x, y)` such that `P(x, y) =
    \partial{}F/\partial{}x` and `Q(x, y) = \partial{}F/\partial{}y`.  It can
    be shown that a necessary and sufficient condition for a first order ODE
    to be exact is that `\partial{}P/\partial{}y = \partial{}Q/\partial{}x`.
    Then, the solution will be as given below::

        >>> from sympy import Function, Eq, Integral, symbols, pprint
        >>> x, y, t, x0, y0, C1= symbols('x,y,t,x0,y0,C1')
        >>> P, Q, F= map(Function, ['P', 'Q', 'F'])
        >>> pprint(Eq(Eq(F(x, y), Integral(P(t, y), (t, x0, x)) +
        ... Integral(Q(x0, t), (t, y0, y))), C1))
                    x                y
                    /                /
                   |                |
        F(x, y) =  |  P(t, y) dt +  |  Q(x0, t) dt = C1
                   |                |
                  /                /
                  x0               y0

    Where the first partials of `P` and `Q` exist and are continuous in a
    simply connected region.

    A note: SymPy currently has no way to represent inert substitution on an
    expression, so the hint ``1st_exact_Integral`` will return an integral
    with `dy`.  This is supposed to represent the function that you are
    solving for.

    Examples
    ========

    >>> from sympy import Function, dsolve, cos, sin
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x),
    ... f(x), hint='1st_exact')
    Eq(x*cos(f(x)) + f(x)**3/3, C1)

    References
    ==========

    - https://en.wikipedia.org/wiki/Exact_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 73

    # indirect doctest

    """
    hint = "1st_exact"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return P + Q*fx.diff(x)

    def _verify(self, fx) -> bool:
        P, Q = self.wilds()
        x = self.ode_problem.sym
        y = Dummy('y')

        m, n = self.wilds_match()

        m = m.subs(fx, y)
        n = n.subs(fx, y)
        numerator = cancel(m.diff(y) - n.diff(x))

        if numerator.is_zero:
            # Is exact
            return True
        else:
            # The following few conditions try to convert a non-exact
            # differential equation into an exact one.
            # References:
            # 1. Differential equations with applications
            # and historical notes - George E. Simmons
            # 2. https://math.okstate.edu/people/binegar/2233-S99/2233-l12.pdf

            factor_n = cancel(numerator/n)
            factor_m = cancel(-numerator/m)
            if y not in factor_n.free_symbols:
                # If (dP/dy - dQ/dx) / Q = f(x)
                # then exp(integral(f(x))*equation becomes exact
                factor = factor_n
                integration_variable = x
            elif x not in factor_m.free_symbols:
                # If (dP/dy - dQ/dx) / -P = f(y)
                # then exp(integral(f(y))*equation becomes exact
                factor = factor_m
                integration_variable = y
            else:
                # Couldn't convert to exact
                return False

            factor = exp(Integral(factor, integration_variable))
            m *= factor
            n *= factor
            self._wilds_match[P] = m.subs(y, fx)
            self._wilds_match[Q] = n.subs(y, fx)
            return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        m, n = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        y = Dummy('y')

        m = m.subs(fx, y)
        n = n.subs(fx, y)

        gen_sol = Eq(Subs(Integral(m, x)
                          + Integral(n - Integral(m, x).diff(y), y), y, fx), C1)
        return [gen_sol]


class FirstLinear(SinglePatternODESolver):
    r"""
    Solves 1st order linear differential equations.

    These are differential equations of the form

    .. math:: dy/dx + P(x) y = Q(x)\text{.}

    These kinds of differential equations can be solved in a general way.  The
    integrating factor `e^{\int P(x) \,dx}` will turn the equation into a
    separable equation.  The general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint, diff, sin
        >>> from sympy.abc import x
        >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
        >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x))
        >>> pprint(genform)
                    d
        P(x)*f(x) + --(f(x)) = Q(x)
                    dx
        >>> pprint(dsolve(genform, f(x), hint='1st_linear_Integral'))
                /       /                   \
                |      |                    |
                |      |         /          |     /
                |      |        |           |    |
                |      |        | P(x) dx   |  - | P(x) dx
                |      |        |           |    |
                |      |       /            |   /
        f(x) = |C1 +  | Q(x)*e           dx|*e
                |      |                    |
                \     /                     /


    Examples
    ========

    >>> f = Function('f')
    >>> pprint(dsolve(Eq(x*diff(f(x), x) - f(x), x**2*sin(x)),
    ... f(x), '1st_linear'))
    f(x) = x*(C1 - cos(x))

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation#First-order_equation_with_variable_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 92

    # indirect doctest

    """
    hint = '1st_linear'
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x), f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return fx.diff(x) + P*fx - Q

    def _get_general_solution(self, *, simplify_flag: bool = True):
        P, Q = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        gensol = Eq(fx, ((C1 + Integral(Q*exp(Integral(P, x)), x))
            * exp(-Integral(P, x))))
        return [gensol]


class AlmostLinear(SinglePatternODESolver):
    r"""
    Solves an almost-linear differential equation.

    The general form of an almost linear differential equation is

    .. math:: a(x) g'(f(x)) f'(x) + b(x) g(f(x)) + c(x)

    Here `f(x)` is the function to be solved for (the dependent variable).
    The substitution `g(f(x)) = u(x)` leads to a linear differential equation
    for `u(x)` of the form `a(x) u' + b(x) u + c(x) = 0`. This can be solved
    for `u(x)` by the `first_linear` hint and then `f(x)` is found by solving
    `g(f(x)) = u(x)`.

    See Also
    ========
    :obj:`sympy.solvers.ode.single.FirstLinear`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint, sin, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = x*d + x*f(x) + 1
    >>> dsolve(eq, f(x), hint='almost_linear')
    Eq(f(x), (C1 - Ei(x))*exp(-x))
    >>> pprint(dsolve(eq, f(x), hint='almost_linear'))
                        -x
    f(x) = (C1 - Ei(x))*e
    >>> example = cos(f(x))*f(x).diff(x) + sin(f(x)) + 1
    >>> pprint(example)
                        d
    sin(f(x)) + cos(f(x))*--(f(x)) + 1
                        dx
    >>> pprint(dsolve(example, f(x), hint='almost_linear'))
                    /    -x    \             /    -x    \
    [f(x) = pi - asin\C1*e   - 1/, f(x) = asin\C1*e   - 1/]


    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "almost_linear"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return P*fx.diff(x) + Q

    def _verify(self, fx):
        a, b = self.wilds_match()
        c, b = b.as_independent(fx) if b.is_Add else (S.Zero, b)
        # a, b and c are the function a(x), b(x) and c(x) respectively.
        # c(x) is obtained by separating out b as terms with and without fx i.e, l(y)
        # The following conditions checks if the given equation is an almost-linear differential equation using the fact that
        # a(x)*(l(y))' / l(y)' is independent of l(y)

        if b.diff(fx) != 0 and not simplify(b.diff(fx)/a).has(fx):
            self.ly = factor_terms(b).as_independent(fx, as_Add=False)[1] # Gives the term containing fx i.e., l(y)
            self.ax = a / self.ly.diff(fx)
            self.cx = -c  # cx is taken as -c(x) to simplify expression in the solution integral
            self.bx = factor_terms(b) / self.ly
            return True

        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        x = self.ode_problem.sym
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        gensol = Eq(self.ly, ((C1 + Integral((self.cx/self.ax)*exp(Integral(self.bx/self.ax, x)), x))
                * exp(-Integral(self.bx/self.ax, x))))

        return [gensol]


class Bernoulli(SinglePatternODESolver):
    r"""
    Solves Bernoulli differential equations.

    These are equations of the form

    .. math:: dy/dx + P(x) y = Q(x) y^n\text{, }n \ne 1`\text{.}

    The substitution `w = 1/y^{1-n}` will transform an equation of this form
    into one that is linear (see the docstring of
    :obj:`~sympy.solvers.ode.single.FirstLinear`).  The general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint
        >>> from sympy.abc import x, n
        >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
        >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x)*f(x)**n)
        >>> pprint(genform)
                    d                n
        P(x)*f(x) + --(f(x)) = Q(x)*f (x)
                    dx
        >>> pprint(dsolve(genform, f(x), hint='Bernoulli_Integral'), num_columns=110)
                                                                                                                -1
                                                                                                               -----
                                                                                                               n - 1
               //         /                                 /                            \                    \
               ||        |                                 |                             |                    |
               ||        |                  /              |                  /          |            /       |
               ||        |                 |               |                 |           |           |        |
               ||        |       -(n - 1)* | P(x) dx       |       -(n - 1)* | P(x) dx   |  (n - 1)* | P(x) dx|
               ||        |                 |               |                 |           |           |        |
               ||        |                /                |                /            |          /         |
        f(x) = ||C1 - n* | Q(x)*e                    dx +  | Q(x)*e                    dx|*e                  |
               ||        |                                 |                             |                    |
               \\       /                                 /                              /                    /


    Note that the equation is separable when `n = 1` (see the docstring of
    :obj:`~sympy.solvers.ode.single.Separable`).

    >>> pprint(dsolve(Eq(f(x).diff(x) + P(x)*f(x), Q(x)*f(x)), f(x),
    ... hint='separable_Integral'))
    f(x)
        /
    |                /
    |  1            |
    |  - dy = C1 +  | (-P(x) + Q(x)) dx
    |  y            |
    |              /
    /


    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, pprint, log
    >>> from sympy.abc import x
    >>> f = Function('f')

    >>> pprint(dsolve(Eq(x*f(x).diff(x) + f(x), log(x)*f(x)**2),
    ... f(x), hint='Bernoulli'))
                    1
    f(x) =  -----------------
            C1*x + log(x) + 1

    References
    ==========

    - https://en.wikipedia.org/wiki/Bernoulli_differential_equation

    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 95

    # indirect doctest

    """
    hint = "Bernoulli"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x)])
        n = Wild('n', exclude=[x, f(x), f(x).diff(x)])
        return P, Q, n

    def _equation(self, fx, x, order):
        P, Q, n = self.wilds()
        return fx.diff(x) + P*fx - Q*fx**n

    def _get_general_solution(self, *, simplify_flag: bool = True):
        P, Q, n = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        if n==1:
            gensol = Eq(log(fx), (
            C1 + Integral((-P + Q), x)
        ))
        else:
            gensol = Eq(fx**(1-n), (
                (C1 - (n - 1) * Integral(Q*exp(-n*Integral(P, x))
                            * exp(Integral(P, x)), x)
                ) * exp(-(1 - n)*Integral(P, x)))
            )
        return [gensol]


class Factorable(SingleODESolver):
    r"""
        Solves equations having a solvable factor.

        This function is used to solve the equation having factors. Factors may be of type algebraic or ode. It
        will try to solve each factor independently. Factors will be solved by calling dsolve. We will return the
        list of solutions.

        Examples
        ========

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x
        >>> f = Function('f')
        >>> eq = (f(x)**2-4)*(f(x).diff(x)+f(x))
        >>> pprint(dsolve(eq, f(x)))
                                        -x
        [f(x) = 2, f(x) = -2, f(x) = C1*e  ]


        """
    hint = "factorable"
    has_integral = False

    def _matches(self):
        eq_orig = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        df = f(x).diff(x)
        self.eqs = []
        eq = eq_orig.collect(f(x), func = cancel)
        eq = fraction(factor(eq))[0]
        factors = Mul.make_args(factor(eq))
        roots = [fac.as_base_exp() for fac in factors if len(fac.args)!=0]
        if len(roots)>1 or roots[0][1]>1:
            for base, expo in roots:
                if base.has(f(x)):
                    self.eqs.append(base)
            if len(self.eqs)>0:
                return True
        roots = solve(eq, df)
        if len(roots)>0:
            self.eqs = [(df - root) for root in roots]
            # Avoid infinite recursion
            matches = self.eqs != [eq_orig]
            return matches
        for i in factors:
            if i.has(f(x)):
                self.eqs.append(i)
        return len(self.eqs)>0 and len(factors)>1

    def _get_general_solution(self, *, simplify_flag: bool = True):
        func = self.ode_problem.func.func
        x = self.ode_problem.sym
        eqns = self.eqs
        sols = []
        for eq in eqns:
            try:
                sol = dsolve(eq, func(x))
            except NotImplementedError:
                continue
            else:
                if isinstance(sol, list):
                    sols.extend(sol)
                else:
                    sols.append(sol)

        if sols == []:
            raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                + " the factorable group method")
        return sols


class RiccatiSpecial(SinglePatternODESolver):
    r"""
    The general Riccati equation has the form

    .. math:: dy/dx = f(x) y^2 + g(x) y + h(x)\text{.}

    While it does not have a general solution [1], the "special" form, `dy/dx
    = a y^2 - b x^c`, does have solutions in many cases [2].  This routine
    returns a solution for `a(dy/dx) = b y^2 + c y/x + d/x^2` that is obtained
    by using a suitable change of variables to reduce it to the special form
    and is valid when neither `a` nor `b` are zero and either `c` or `d` is
    zero.

    >>> from sympy.abc import x, a, b, c, d
    >>> from sympy import dsolve, checkodesol, pprint, Function
    >>> f = Function('f')
    >>> y = f(x)
    >>> genform = a*y.diff(x) - (b*y**2 + c*y/x + d/x**2)
    >>> sol = dsolve(genform, y, hint="Riccati_special_minus2")
    >>> pprint(sol, wrap_line=False)
            /                                 /        __________________       \\
            |           __________________    |       /                2        ||
            |          /                2     |     \/  4*b*d - (a + c)  *log(x)||
           -|a + c - \/  4*b*d - (a + c)  *tan|C1 + ----------------------------||
            \                                 \                 2*a             //
    f(x) = ------------------------------------------------------------------------
                                            2*b*x

    >>> checkodesol(genform, sol, order=1)[0]
    True

    References
    ==========

    - https://www.maplesoft.com/support/help/Maple/view.aspx?path=odeadvisor/Riccati
    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0106.pdf -
      https://eqworld.ipmnet.ru/en/solutions/ode/ode0123.pdf
    """
    hint = "Riccati_special_minus2"
    has_integral = False
    order = [1]

    def _wilds(self, f, x, order):
        a = Wild('a', exclude=[x, f(x), f(x).diff(x), 0])
        b = Wild('b', exclude=[x, f(x), f(x).diff(x), 0])
        c = Wild('c', exclude=[x, f(x), f(x).diff(x)])
        d = Wild('d', exclude=[x, f(x), f(x).diff(x)])
        return a, b, c, d

    def _equation(self, fx, x, order):
        a, b, c, d = self.wilds()
        return a*fx.diff(x) + b*fx**2 + c*fx/x + d/x**2

    def _get_general_solution(self, *, simplify_flag: bool = True):
        a, b, c, d = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        mu = sqrt(4*d*b - (a - c)**2)

        gensol = Eq(fx, (a - c - mu*tan(mu/(2*a)*log(x) + C1))/(2*b*x))
        return [gensol]


class RationalRiccati(SinglePatternODESolver):
    r"""
    Gives general solutions to the first order Riccati differential
    equations that have atleast one rational particular solution.

    .. math :: y' = b_0(x) + b_1(x) y + b_2(x) y^2

    where `b_0`, `b_1` and `b_2` are rational functions of `x`
    with `b_2 \ne 0` (`b_2 = 0` would make it a Bernoulli equation).

    Examples
    ========

    >>> from sympy import Symbol, Function, dsolve, checkodesol
    >>> f = Function('f')
    >>> x = Symbol('x')

    >>> eq = -x**4*f(x)**2 + x**3*f(x).diff(x) + x**2*f(x) + 20
    >>> sol = dsolve(eq, hint="1st_rational_riccati")
    >>> sol
    Eq(f(x), (4*C1 - 5*x**9 - 4)/(x**2*(C1 + x**9 - 1)))
    >>> checkodesol(eq, sol)
    (True, 0)

    References
    ==========

    - Riccati ODE:  https://en.wikipedia.org/wiki/Riccati_equation
    - N. Thieu Vo - Rational and Algebraic Solutions of First-Order Algebraic ODEs:
      Algorithm 11, pp. 78 - https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf
    """
    has_integral = False
    hint = "1st_rational_riccati"
    order = [1]

    def _wilds(self, f, x, order):
        b0 = Wild('b0', exclude=[f(x), f(x).diff(x)])
        b1 = Wild('b1', exclude=[f(x), f(x).diff(x)])
        b2 = Wild('b2', exclude=[f(x), f(x).diff(x)])
        return (b0, b1, b2)

    def _equation(self, fx, x, order):
        b0, b1, b2 = self.wilds()
        return fx.diff(x) - b0 - b1*fx - b2*fx**2

    def _matches(self):
        eq = self.ode_problem.eq_expanded
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order

        if order != 1:
            return False

        match, funcs = match_riccati(eq, f, x)
        if not match:
            return False
        _b0, _b1, _b2 = funcs
        b0, b1, b2 = self.wilds()
        self._wilds_match = match = {b0: _b0, b1: _b1, b2: _b2}
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # Match the equation
        b0, b1, b2 = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return solve_riccati(fx, x, b0, b1, b2, gensol=True)


class SecondNonlinearAutonomousConserved(SinglePatternODESolver):
    r"""
    Gives solution for the autonomous second order nonlinear
    differential equation of the form

    .. math :: f''(x) = g(f(x))

    The solution for this differential equation can be computed
    by multiplying by `f'(x)` and integrating on both sides,
    converting it into a first order differential equation.

    Examples
    ========

    >>> from sympy import Function, symbols, dsolve
    >>> f, g = symbols('f g', cls=Function)
    >>> x = symbols('x')

    >>> eq = f(x).diff(x, 2) - g(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 - x)]

    >>> from sympy import exp, log
    >>> eq = f(x).diff(x, 2) - exp(f(x)) + log(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 - x)]

    References
    ==========

    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0301.pdf
    """
    hint = "2nd_nonlinear_autonomous_conserved"
    has_integral = True
    order = [2]

    def _wilds(self, f, x, order):
        fy = Wild('fy', exclude=[0, f(x).diff(x), f(x).diff(x, 2)])
        return (fy, )

    def _equation(self, fx, x, order):
        fy = self.wilds()[0]
        return fx.diff(x, 2) + fy

    def _verify(self, fx):
        return self.ode_problem.is_autonomous

    def _get_general_solution(self, *, simplify_flag: bool = True):
        g = self.wilds_match()[0]
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        u = Dummy('u')
        g = g.subs(fx, u)
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        inside = -2*Integral(g, u) + C1
        lhs = Integral(1/sqrt(inside), (u, fx))
        return [Eq(lhs, C2 + x), Eq(lhs, C2 - x)]


class Liouville(SinglePatternODESolver):
    r"""
    Solves 2nd order Liouville differential equations.

    The general form of a Liouville ODE is

    .. math:: \frac{d^2 y}{dx^2} + g(y) \left(\!
                \frac{dy}{dx}\!\right)^2 + h(x)
                \frac{dy}{dx}\text{.}

    The general solution is:

        >>> from sympy import Function, dsolve, Eq, pprint, diff
        >>> from sympy.abc import x
        >>> f, g, h = map(Function, ['f', 'g', 'h'])
        >>> genform = Eq(diff(f(x),x,x) + g(f(x))*diff(f(x),x)**2 +
        ... h(x)*diff(f(x),x), 0)
        >>> pprint(genform)
                          2                    2
                /d       \         d          d
        g(f(x))*|--(f(x))|  + h(x)*--(f(x)) + ---(f(x)) = 0
                \dx      /         dx           2
                                              dx
        >>> pprint(dsolve(genform, f(x), hint='Liouville_Integral'))
                                          f(x)
                  /                     /
                 |                     |
                 |     /               |     /
                 |    |                |    |
                 |  - | h(x) dx        |    | g(y) dy
                 |    |                |    |
                 |   /                 |   /
        C1 + C2* | e            dx +   |  e           dy = 0
                 |                     |
                /                     /

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(diff(f(x), x, x) + diff(f(x), x)**2/f(x) +
    ... diff(f(x), x)/x, f(x), hint='Liouville'))
               ________________           ________________
    [f(x) = -\/ C1 + C2*log(x) , f(x) = \/ C1 + C2*log(x) ]

    References
    ==========

    - Goldstein and Braun, "Advanced Methods for the Solution of Differential
      Equations", pp. 98
    - https://www.maplesoft.com/support/help/Maple/view.aspx?path=odeadvisor/Liouville

    # indirect doctest

    """
    hint = "Liouville"
    has_integral = True
    order = [2]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        k = Wild('k', exclude=[f(x).diff(x)])
        return d, e, k

    def _equation(self, fx, x, order):
        # Liouville ODE in the form
        # f(x).diff(x, 2) + g(f(x))*(f(x).diff(x))**2 + h(x)*f(x).diff(x)
        # See Goldstein and Braun, "Advanced Methods for the Solution of
        # Differential Equations", pg. 98
        d, e, k = self.wilds()
        return d*fx.diff(x, 2) + e*fx.diff(x)**2 + k*fx.diff(x)

    def _verify(self, fx):
        d, e, k = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.g = simplify(e/d).subs(fx, self.y)
        self.h = simplify(k/d).subs(fx, self.y)
        if self.y in self.h.free_symbols or x in self.g.free_symbols:
            return False
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, k = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        int = Integral(exp(Integral(self.g, self.y)), (self.y, None, fx))
        gen_sol = Eq(int + C1*Integral(exp(-Integral(self.h, x)), x) + C2, 0)

        return [gen_sol]


class Separable(SinglePatternODESolver):
    r"""
    Solves separable 1st order differential equations.

    This is any differential equation that can be written as `P(y)
    \tfrac{dy}{dx} = Q(x)`.  The solution can then just be found by
    rearranging terms and integrating: `\int P(y) \,dy = \int Q(x) \,dx`.
    This hint uses :py:meth:`sympy.simplify.simplify.separatevars` as its back
    end, so if a separable equation is not caught by this solver, it is most
    likely the fault of that function.
    :py:meth:`~sympy.simplify.simplify.separatevars` is
    smart enough to do most expansion and factoring necessary to convert a
    separable equation `F(x, y)` into the proper form `P(x)\cdot{}Q(y)`.  The
    general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint
        >>> from sympy.abc import x
        >>> a, b, c, d, f = map(Function, ['a', 'b', 'c', 'd', 'f'])
        >>> genform = Eq(a(x)*b(f(x))*f(x).diff(x), c(x)*d(f(x)))
        >>> pprint(genform)
                     d
        a(x)*b(f(x))*--(f(x)) = c(x)*d(f(x))
                     dx
        >>> pprint(dsolve(genform, f(x), hint='separable_Integral'))
             f(x)
           /                  /
          |                  |
          |  b(y)            | c(x)
          |  ---- dy = C1 +  | ---- dx
          |  d(y)            | a(x)
          |                  |
         /                  /

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(Eq(f(x)*f(x).diff(x) + x, 3*x*f(x)**2), f(x),
    ... hint='separable', simplify=False))
       /   2       \         2
    log\3*f (x) - 1/        x
    ---------------- = C1 + --
           6                2

    References
    ==========

    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 52

    # indirect doctest

    """
    hint = "separable"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        d, e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        d = separatevars(d.subs(fx, self.y))
        e = separatevars(e.subs(fx, self.y))
        # m1[coeff]*m1[x]*m1[y] + m2[coeff]*m2[x]*m2[y]*y'
        self.m1 = separatevars(d, dict=True, symbols=(x, self.y))
        self.m2 = separatevars(e, dict=True, symbols=(x, self.y))
        return bool(self.m1 and self.m2)

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return self.m1, self.m2, x, fx

    def _get_general_solution(self, *, simplify_flag: bool = True):
        m1, m2, x, fx = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(m2['coeff']*m2[self.y]/m1[self.y],
        (self.y, None, fx))
        gen_sol = Eq(int, Integral(-m1['coeff']*m1[x]/
        m2[x], x) + C1)
        return [gen_sol]


class SeparableReduced(Separable):
    r"""
    Solves a differential equation that can be reduced to the separable form.

    The general form of this equation is

    .. math:: y' + (y/x) H(x^n y) = 0\text{}.

    This can be solved by substituting `u(y) = x^n y`.  The equation then
    reduces to the separable form `\frac{u'}{u (\mathrm{power} - H(u))} -
    \frac{1}{x} = 0`.

    The general solution is:

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x, n
        >>> f, g = map(Function, ['f', 'g'])
        >>> genform = f(x).diff(x) + (f(x)/x)*g(x**n*f(x))
        >>> pprint(genform)
                         / n     \
        d          f(x)*g\x *f(x)/
        --(f(x)) + ---------------
        dx                x
        >>> pprint(dsolve(genform, hint='separable_reduced'))
         n
        x *f(x)
          /
         |
         |         1
         |    ------------ dy = C1 + log(x)
         |    y*(n - g(y))
         |
         /

    See Also
    ========
    :obj:`sympy.solvers.ode.single.Separable`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = (x - x**2*f(x))*d - f(x)
    >>> dsolve(eq, hint='separable_reduced')
    [Eq(f(x), (1 - sqrt(C1*x**2 + 1))/x), Eq(f(x), (sqrt(C1*x**2 + 1) + 1)/x)]
    >>> pprint(dsolve(eq, hint='separable_reduced'))
                   ___________            ___________
                  /     2                /     2
            1 - \/  C1*x  + 1          \/  C1*x  + 1  + 1
    [f(x) = ------------------, f(x) = ------------------]
                    x                          x

    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "separable_reduced"
    has_integral = True
    order = [1]

    def _degree(self, expr, x):
        # Made this function to calculate the degree of
        # x in an expression. If expr will be of form
        # x**p*y, (wheare p can be variables/rationals) then it
        # will return p.
        for val in expr:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    return (val.as_base_exp()[1])
                elif val == x:
                    return (val.as_base_exp()[1])
                else:
                    return self._degree(val.args, x)
        return 0

    def _powers(self, expr):
        # this function will return all the different relative power of x w.r.t f(x).
        # expr = x**p * f(x)**q then it will return {p/q}.
        pows = set()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.y = Dummy('y')
        if isinstance(expr, Add):
            exprs = expr.atoms(Add)
        elif isinstance(expr, Mul):
            exprs = expr.atoms(Mul)
        elif isinstance(expr, Pow):
            exprs = expr.atoms(Pow)
        else:
            exprs = {expr}

        for arg in exprs:
            if arg.has(x):
                _, u = arg.as_independent(x, fx)
                pow = self._degree((u.subs(fx, self.y), ), x)/self._degree((u.subs(fx, self.y), ), self.y)
                pows.add(pow)
        return pows

    def _verify(self, fx):
        num, den = self.wilds_match()
        x = self.ode_problem.sym
        factor = simplify(x/fx*num/den)
        # Try representing factor in terms of x^n*y
        # where n is lowest power of x in factor;
        # first remove terms like sqrt(2)*3 from factor.atoms(Mul)
        num, dem = factor.as_numer_denom()
        num = expand(num)
        dem = expand(dem)
        pows = self._powers(num)
        pows.update(self._powers(dem))
        pows = list(pows)
        if(len(pows)==1) and pows[0]!=zoo:
            self.t = Dummy('t')
            self.r2 = {'t': self.t}
            num = num.subs(x**pows[0]*fx, self.t)
            dem = dem.subs(x**pows[0]*fx, self.t)
            test = num/dem
            free = test.free_symbols
            if len(free) == 1 and free.pop() == self.t:
                self.r2.update({'power' : pows[0], 'u' : test})
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        u = self.r2['u'].subs(self.r2['t'], self.y)
        ycoeff = 1/(self.y*(self.r2['power'] - u))
        m1 = {self.y: 1, x: -1/x, 'coeff': 1}
        m2 = {self.y: ycoeff, x: 1, 'coeff': 1}
        return m1, m2, x, x**self.r2['power']*fx


class HomogeneousCoeffSubsDepDivIndep(SinglePatternODESolver):
    r"""
    Solves a 1st order differential equation with homogeneous coefficients
    using the substitution `u_1 = \frac{\text{<dependent
    variable>}}{\text{<independent variable>}}`.

    This is a differential equation

    .. math:: P(x, y) + Q(x, y) dy/dx = 0

    such that `P` and `Q` are homogeneous and of the same order.  A function
    `F(x, y)` is homogeneous of order `n` if `F(x t, y t) = t^n F(x, y)`.
    Equivalently, `F(x, y)` can be rewritten as `G(y/x)` or `H(x/y)`.  See
    also the docstring of :py:meth:`~sympy.solvers.ode.homogeneous_order`.

    If the coefficients `P` and `Q` in the differential equation above are
    homogeneous functions of the same order, then it can be shown that the
    substitution `y = u_1 x` (i.e. `u_1 = y/x`) will turn the differential
    equation into an equation separable in the variables `x` and `u`.  If
    `h(u_1)` is the function that results from making the substitution `u_1 =
    f(x)/x` on `P(x, f(x))` and `g(u_2)` is the function that results from the
    substitution on `Q(x, f(x))` in the differential equation `P(x, f(x)) +
    Q(x, f(x)) f'(x) = 0`, then the general solution is::

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x
        >>> f, g, h = map(Function, ['f', 'g', 'h'])
        >>> genform = g(f(x)/x) + h(f(x)/x)*f(x).diff(x)
        >>> pprint(genform)
         /f(x)\    /f(x)\ d
        g|----| + h|----|*--(f(x))
         \ x  /    \ x  / dx
        >>> pprint(dsolve(genform, f(x),
        ... hint='1st_homogeneous_coeff_subs_dep_div_indep_Integral'))
                       f(x)
                       ----
                        x
                         /
                        |
                        |       -h(u1)
        log(x) = C1 +   |  ---------------- d(u1)
                        |  u1*h(u1) + g(u1)
                        |
                       /

    Where `u_1 h(u_1) + g(u_1) \ne 0` and `x \ne 0`.

    See also the docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`.

    Examples
    ========

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_subs_dep_div_indep', simplify=False))
                          /          3   \
                          |3*f(x)   f (x)|
                       log|------ + -----|
                          |  x         3 |
                          \           x  /
    log(x) = log(C1) - -------------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_subs_dep_div_indep"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.d = separatevars(self.d.subs(fx, self.y))
        self.e = separatevars(self.e.subs(fx, self.y))
        ordera = homogeneous_order(self.d, x, self.y)
        orderb = homogeneous_order(self.e, x, self.y)
        if ordera == orderb and ordera is not None:
            self.u = Dummy('u')
            if simplify((self.d + self.u*self.e).subs({x: 1, self.y: self.u})) != 0:
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        xarg = 0
        yarg = 0
        return [self.d, self.e, fx, x, self.u, self.u1, self.y, xarg, yarg]

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, fx, x, u, u1, y, xarg, yarg = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(
            (-e/(d + u1*e)).subs({x: 1, y: u1}),
            (u1, None, fx/x))
        sol = logcombine(Eq(log(x), int + log(C1)), force=True)
        gen_sol = sol.subs(fx, u).subs(((u, u - yarg), (x, x - xarg), (u, fx)))
        return [gen_sol]


class HomogeneousCoeffSubsIndepDivDep(SinglePatternODESolver):
    r"""
    Solves a 1st order differential equation with homogeneous coefficients
    using the substitution `u_2 = \frac{\text{<independent
    variable>}}{\text{<dependent variable>}}`.

    This is a differential equation

    .. math:: P(x, y) + Q(x, y) dy/dx = 0

    such that `P` and `Q` are homogeneous and of the same order.  A function
    `F(x, y)` is homogeneous of order `n` if `F(x t, y t) = t^n F(x, y)`.
    Equivalently, `F(x, y)` can be rewritten as `G(y/x)` or `H(x/y)`.  See
    also the docstring of :py:meth:`~sympy.solvers.ode.homogeneous_order`.

    If the coefficients `P` and `Q` in the differential equation above are
    homogeneous functions of the same order, then it can be shown that the
    substitution `x = u_2 y` (i.e. `u_2 = x/y`) will turn the differential
    equation into an equation separable in the variables `y` and `u_2`.  If
    `h(u_2)` is the function that results from making the substitution `u_2 =
    x/f(x)` on `P(x, f(x))` and `g(u_2)` is the function that results from the
    substitution on `Q(x, f(x))` in the differential equation `P(x, f(x)) +
    Q(x, f(x)) f'(x) = 0`, then the general solution is:

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f, g, h = map(Function, ['f', 'g', 'h'])
    >>> genform = g(x/f(x)) + h(x/f(x))*f(x).diff(x)
    >>> pprint(genform)
     / x  \    / x  \ d
    g|----| + h|----|*--(f(x))
     \f(x)/    \f(x)/ dx
    >>> pprint(dsolve(genform, f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral'))
                 x
                ----
                f(x)
                  /
                 |
                 |       -g(u1)
                 |  ---------------- d(u1)
                 |  u1*g(u1) + h(u1)
                 |
                /
    <BLANKLINE>
    f(x) = C1*e

    Where `u_1 g(u_1) + h(u_1) \ne 0` and `f(x) \ne 0`.

    See also the docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`.

    Examples
    ========

    >>> from sympy import Function, pprint, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep',
    ... simplify=False))
                             /   2     \
                             |3*x      |
                          log|----- + 1|
                             | 2       |
                             \f (x)    /
    log(f(x)) = log(C1) - --------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_subs_indep_div_dep"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.d = separatevars(self.d.subs(fx, self.y))
        self.e = separatevars(self.e.subs(fx, self.y))
        ordera = homogeneous_order(self.d, x, self.y)
        orderb = homogeneous_order(self.e, x, self.y)
        if ordera == orderb and ordera is not None:
            self.u = Dummy('u')
            if simplify((self.e + self.u*self.d).subs({x: self.u, self.y: 1})) != 0:
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        xarg = 0
        yarg = 0
        return [self.d, self.e, fx, x, self.u, self.u1, self.y, xarg, yarg]

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, fx, x, u, u1, y, xarg, yarg = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(simplify((-d/(e + u1*d)).subs({x: u1, y: 1})), (u1, None, x/fx)) # type: ignore
        sol = logcombine(Eq(log(fx), int + log(C1)), force=True)
        gen_sol = sol.subs(fx, u).subs(((u, u - yarg), (x, x - xarg), (u, fx)))
        return [gen_sol]


class HomogeneousCoeffBest(HomogeneousCoeffSubsIndepDivDep, HomogeneousCoeffSubsDepDivIndep):
    r"""
    Returns the best solution to an ODE from the two hints
    ``1st_homogeneous_coeff_subs_dep_div_indep`` and
    ``1st_homogeneous_coeff_subs_indep_div_dep``.

    This is as determined by :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity`.

    See the
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`
    and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`
    docstrings for more information on these hints.  Note that there is no
    ``ode_1st_homogeneous_coeff_best_Integral`` hint.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_best', simplify=False))
                             /   2     \
                             |3*x      |
                          log|----- + 1|
                             | 2       |
                             \f (x)    /
    log(f(x)) = log(C1) - --------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_best"
    has_integral = False
    order = [1]

    def _verify(self, fx):
        return HomogeneousCoeffSubsIndepDivDep._verify(self, fx) and \
               HomogeneousCoeffSubsDepDivIndep._verify(self, fx)

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # There are two substitutions that solve the equation, u1=y/x and u2=x/y
        # # They produce different integrals, so try them both and see which
        # # one is easier
        sol1 = HomogeneousCoeffSubsIndepDivDep._get_general_solution(self)
        sol2 = HomogeneousCoeffSubsDepDivIndep._get_general_solution(self)
        fx = self.ode_problem.func
        if simplify_flag:
            sol1 = odesimp(self.ode_problem.eq, *sol1, fx, "1st_homogeneous_coeff_subs_indep_div_dep")
            sol2 = odesimp(self.ode_problem.eq, *sol2, fx, "1st_homogeneous_coeff_subs_dep_div_indep")
        # XXX: not simplify should be not simplify_flag. mypy correctly complains
        return min([sol1, sol2], key=lambda x: ode_sol_simplicity(x, fx, trysolving=not simplify)) # type: ignore


class LinearCoefficients(HomogeneousCoeffBest):
    r"""
    Solves a differential equation with linear coefficients.

    The general form of a differential equation with linear coefficients is

    .. math:: y' + F\left(\!\frac{a_1 x + b_1 y + c_1}{a_2 x + b_2 y +
                c_2}\!\right) = 0\text{,}

    where `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are constants and `a_1 b_2
    - a_2 b_1 \ne 0`.

    This can be solved by substituting:

    .. math:: x = x' + \frac{b_2 c_1 - b_1 c_2}{a_2 b_1 - a_1 b_2}

              y = y' + \frac{a_1 c_2 - a_2 c_1}{a_2 b_1 - a_1
                  b_2}\text{.}

    This substitution reduces the equation to a homogeneous differential
    equation.

    See Also
    ========
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffBest`
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> df = f(x).diff(x)
    >>> eq = (x + f(x) + 1)*df + (f(x) - 6*x + 1)
    >>> dsolve(eq, hint='linear_coefficients')
    [Eq(f(x), -x - sqrt(C1 + 7*x**2) - 1), Eq(f(x), -x + sqrt(C1 + 7*x**2) - 1)]
    >>> pprint(dsolve(eq, hint='linear_coefficients'))
                      ___________                     ___________
                   /         2                     /         2
    [f(x) = -x - \/  C1 + 7*x   - 1, f(x) = -x + \/  C1 + 7*x   - 1]


    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "linear_coefficients"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        a, b = self.wilds()
        F = self.d/self.e
        x = self.ode_problem.sym
        params = self._linear_coeff_match(F, fx)
        if params:
            self.xarg, self.yarg = params
            u = Dummy('u')
            t = Dummy('t')
            self.y = Dummy('y')
            # Dummy substitution for df and f(x).
            dummy_eq = self.ode_problem.eq.subs(((fx.diff(x), t), (fx, u)))
            reps = ((x, x + self.xarg), (u, u + self.yarg), (t, fx.diff(x)), (u, fx))
            dummy_eq = simplify(dummy_eq.subs(reps))
            # get the re-cast values for e and d
            r2 = collect(expand(dummy_eq), [fx.diff(x), fx]).match(a*fx.diff(x) + b)
            if r2:
                self.d, self.e = r2[b], r2[a]
                orderd = homogeneous_order(self.d, x, fx)
                ordere = homogeneous_order(self.e, x, fx)
                if orderd == ordere and orderd is not None:
                    self.d = self.d.subs(fx, self.y)
                    self.e = self.e.subs(fx, self.y)
                    return True
                return False
            return False

    def _linear_coeff_match(self, expr, func):
        r"""
        Helper function to match hint ``linear_coefficients``.

        Matches the expression to the form `(a_1 x + b_1 f(x) + c_1)/(a_2 x + b_2
        f(x) + c_2)` where the following conditions hold:

        1. `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are Rationals;
        2. `c_1` or `c_2` are not equal to zero;
        3. `a_2 b_1 - a_1 b_2` is not equal to zero.

        Return ``xarg``, ``yarg`` where

        1. ``xarg`` = `(b_2 c_1 - b_1 c_2)/(a_2 b_1 - a_1 b_2)`
        2. ``yarg`` = `(a_1 c_2 - a_2 c_1)/(a_2 b_1 - a_1 b_2)`


        Examples
        ========

        >>> from sympy import Function, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import LinearCoefficients
        >>> f = Function('f')
        >>> eq = (-25*f(x) - 8*x + 62)/(4*f(x) + 11*x - 11)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (1/9, 22/9)
        >>> eq = sin((-5*f(x) - 8*x + 6)/(4*f(x) + x - 1))
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (19/27, 2/27)
        >>> eq = sin(f(x)/x)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))

        """
        f = func.func
        x = func.args[0]
        def abc(eq):
            r'''
            Internal function of _linear_coeff_match
            that returns Rationals a, b, c
            if eq is a*x + b*f(x) + c, else None.
            '''
            eq = _mexpand(eq)
            c = eq.as_independent(x, f(x), as_Add=True)[0]
            if not c.is_Rational:
                return
            a = eq.coeff(x)
            if not a.is_Rational:
                return
            b = eq.coeff(f(x))
            if not b.is_Rational:
                return
            if eq == a*x + b*f(x) + c:
                return a, b, c

        def match(arg):
            r'''
            Internal function of _linear_coeff_match that returns Rationals a1,
            b1, c1, a2, b2, c2 and a2*b1 - a1*b2 of the expression (a1*x + b1*f(x)
            + c1)/(a2*x + b2*f(x) + c2) if one of c1 or c2 and a2*b1 - a1*b2 is
            non-zero, else None.
            '''
            n, d = arg.together().as_numer_denom()
            m = abc(n)
            if m is not None:
                a1, b1, c1 = m
                m = abc(d)
                if m is not None:
                    a2, b2, c2 = m
                    d = a2*b1 - a1*b2
                    if (c1 or c2) and d:
                        return a1, b1, c1, a2, b2, c2, d

        m = [fi.args[0] for fi in expr.atoms(Function) if fi.func != f and
            len(fi.args) == 1 and not fi.args[0].is_Function] or {expr}
        m1 = match(m.pop())
        if m1 and all(match(mi) == m1 for mi in m):
            a1, b1, c1, a2, b2, c2, denom = m1
            return (b2*c1 - b1*c2)/denom, (a1*c2 - a2*c1)/denom

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        u = Dummy('u')
        return [self.d, self.e, fx, x, u, self.u1, self.y, self.xarg, self.yarg]


class NthOrderReducible(SingleODESolver):
    r"""
    Solves ODEs that only involve derivatives of the dependent variable using
    a substitution of the form `f^n(x) = g(x)`.

    For example any second order ODE of the form `f''(x) = h(f'(x), x)` can be
    transformed into a pair of 1st order ODEs `g'(x) = h(g(x), x)` and
    `f'(x) = g(x)`. Usually the 1st order ODE for `g` is easier to solve. If
    that gives an explicit solution for `g` then `f` is found simply by
    integration.


    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = Eq(x*f(x).diff(x)**2 + f(x).diff(x, 2), 0)
    >>> dsolve(eq, f(x), hint='nth_order_reducible')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), C1 - sqrt(-1/C2)*log(-C2*sqrt(-1/C2) + x) + sqrt(-1/C2)*log(C2*sqrt(-1/C2) + x))

    """
    hint = "nth_order_reducible"
    has_integral = False

    def _matches(self):
        # Any ODE that can be solved with a substitution and
        # repeated integration e.g.:
        # `d^2/dx^2(y) + x*d/dx(y) = constant
        #f'(x) must be finite for this to work
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        x = self.ode_problem.sym
        r"""
        Matches any differential equation that can be rewritten with a smaller
        order. Only derivatives of ``func`` alone, wrt a single variable,
        are considered, and only in them should ``func`` appear.
        """
        # ODE only handles functions of 1 variable so this affirms that state
        assert len(func.args) == 1
        vc = [d.variable_count[0] for d in eq.atoms(Derivative)
            if d.expr == func and len(d.variable_count) == 1]
        ords = [c for v, c in vc if v == x]
        if len(ords) < 2:
            return False
        self.smallest = min(ords)
        # make sure func does not appear outside of derivatives
        D = Dummy()
        if eq.subs(func.diff(x, self.smallest), D).has(func):
            return False
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        n = self.smallest
        # get a unique function name for g
        names = [a.name for a in eq.atoms(AppliedUndef)]
        while True:
            name = Dummy().name
            if name not in names:
                g = Function(name)
                break
        w = f(x).diff(x, n)
        geq = eq.subs(w, g(x))
        gsol = dsolve(geq, g(x))

        if not isinstance(gsol, list):
            gsol = [gsol]

        # Might be multiple solutions to the reduced ODE:
        fsol = []
        for gsoli in gsol:
            fsoli = dsolve(gsoli.subs(g(x), w), f(x))  # or do integration n times
            fsol.append(fsoli)

        return fsol


class SecondHypergeometric(SingleODESolver):
    r"""
    Solves 2nd order linear differential equations.

    It computes special function solutions which can be expressed using the
    2F1, 1F1 or 0F1 hypergeometric functions.

    .. math:: y'' + A(x) y' + B(x) y = 0\text{,}

    where `A` and `B` are rational functions.

    These kinds of differential equations have solution of non-Liouvillian form.

    Given linear ODE can be obtained from 2F1 given by

    .. math:: (x^2 - x) y'' + ((a + b + 1) x - c) y' + b a y = 0\text{,}

    where {a, b, c} are arbitrary constants.

    Notes
    =====

    The algorithm should find any solution of the form

    .. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

    where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
    Currently only the 2F1 case is implemented in SymPy but the other cases are
    described in the paper and could be implemented in future (contributions
    welcome!).


    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = (x*x - x)*f(x).diff(x,2) + (5*x - 1)*f(x).diff(x) + 4*f(x)
    >>> pprint(dsolve(eq, f(x), '2nd_hypergeometric'))
                                        _
           /        /           4  \\  |_  /-1, -1 |  \
           |C1 + C2*|log(x) + -----||* |   |       | x|
           \        \         x + 1// 2  1 \  1    |  /
    f(x) = --------------------------------------------
                                    3
                             (x - 1)


    References
    ==========

    - "Non-Liouvillian solutions for second order linear ODEs" by L. Chan, E.S. Cheb-Terrab

    """
    hint = "2nd_hypergeometric"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        r = match_2nd_hypergeometric(eq, func)
        self.match_object = None
        if r:
            A, B = r
            d = equivalence_hypergeometric(A, B, func)
            if d:
                if d['type'] == "2F1":
                    self.match_object = match_2nd_2F1_hypergeometric(d['I0'], d['k'], d['sing_point'], func)
                    if self.match_object is not None:
                        self.match_object.update({'A':A, 'B':B})
            # We can extend it for 1F1 and 0F1 type also.
        return self.match_object is not None

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        if self.match_object['type'] == "2F1":
            sol = get_sol_2F1_hypergeometric(eq, func, self.match_object)
            if sol is None:
                raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                    + " the hypergeometric method")

        return [sol]


class NthLinearConstantCoeffHomogeneous(SingleODESolver):
    r"""
    Solves an `n`\th order linear homogeneous differential equation with
    constant coefficients.

    This is an equation of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x)
                + a_0 f(x) = 0\text{.}

    These equations can be solved in a general manner, by taking the roots of
    the characteristic equation `a_n m^n + a_{n-1} m^{n-1} + \cdots + a_1 m +
    a_0 = 0`.  The solution will then be the sum of `C_n x^i e^{r x}` terms,
    for each where `C_n` is an arbitrary constant, `r` is a root of the
    characteristic equation and `i` is one of each from 0 to the multiplicity
    of the root - 1 (for example, a root 3 of multiplicity 2 would create the
    terms `C_1 e^{3 x} + C_2 x e^{3 x}`).  The exponential is usually expanded
    for complex roots using Euler's equation `e^{I x} = \cos(x) + I \sin(x)`.
    Complex roots always come in conjugate pairs in polynomials with real
    coefficients, so the two roots will be represented (after simplifying the
    constants) as `e^{a x} \left(C_1 \cos(b x) + C_2 \sin(b x)\right)`.

    If SymPy cannot find exact roots to the characteristic equation, a
    :py:class:`~sympy.polys.rootoftools.ComplexRootOf` instance will be return
    instead.

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(f(x).diff(x, 5) + 10*f(x).diff(x) - 2*f(x), f(x),
    ... hint='nth_linear_constant_coeff_homogeneous')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), C5*exp(x*CRootOf(_x**5 + 10*_x - 2, 0))
    + (C1*sin(x*im(CRootOf(_x**5 + 10*_x - 2, 1)))
    + C2*cos(x*im(CRootOf(_x**5 + 10*_x - 2, 1))))*exp(x*re(CRootOf(_x**5 + 10*_x - 2, 1)))
    + (C3*sin(x*im(CRootOf(_x**5 + 10*_x - 2, 3)))
    + C4*cos(x*im(CRootOf(_x**5 + 10*_x - 2, 3))))*exp(x*re(CRootOf(_x**5 + 10*_x - 2, 3))))

    Note that because this method does not involve integration, there is no
    ``nth_linear_constant_coeff_homogeneous_Integral`` hint.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 4) + 2*f(x).diff(x, 3) -
    ... 2*f(x).diff(x, 2) - 6*f(x).diff(x) + 5*f(x), f(x),
    ... hint='nth_linear_constant_coeff_homogeneous'))
                        x                            -2*x
    f(x) = (C1 + C2*x)*e  + (C3*sin(x) + C4*cos(x))*e

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation section:
      Nonhomogeneous_equation_with_constant_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 211

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_homogeneous"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if not self.r[-1]:
                return True
            else:
                return False
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        fx = self.ode_problem.func
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, fx, order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        gsol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        gsol = Eq(fx, gsol_rhs)
        if simplify_flag:
            gsol = _get_simplified_sol([gsol], fx, collectterms)

        return [gsol]


class NthLinearConstantCoeffVariationOfParameters(SingleODESolver):
    r"""
    Solves an `n`\th order linear differential equation with constant
    coefficients using the method of variation of parameters.

    This method works on any differential equations of the form

    .. math:: f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x) + a_0
                f(x) = P(x)\text{.}

    This method works by assuming that the particular solution takes the form

    .. math:: \sum_{x=1}^{n} c_i(x) y_i(x)\text{,}

    where `y_i` is the `i`\th solution to the homogeneous equation.  The
    solution is then solved using Wronskian's and Cramer's Rule.  The
    particular solution is given by

    .. math:: \sum_{x=1}^n \left( \int \frac{W_i(x)}{W(x)} \,dx
                \right) y_i(x) \text{,}

    where `W(x)` is the Wronskian of the fundamental system (the system of `n`
    linearly independent solutions to the homogeneous equation), and `W_i(x)`
    is the Wronskian of the fundamental system with the `i`\th column replaced
    with `[0, 0, \cdots, 0, P(x)]`.

    This method is general enough to solve any `n`\th order inhomogeneous
    linear differential equation with constant coefficients, but sometimes
    SymPy cannot simplify the Wronskian well enough to integrate it.  If this
    method hangs, try using the
    ``nth_linear_constant_coeff_variation_of_parameters_Integral`` hint and
    simplifying the integrals manually.  Also, prefer using
    ``nth_linear_constant_coeff_undetermined_coefficients`` when it
    applies, because it does not use integration, making it faster and more
    reliable.

    Warning, using simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters' in
    :py:meth:`~sympy.solvers.ode.dsolve` may cause it to hang, because it will
    not attempt to simplify the Wronskian before integrating.  It is
    recommended that you only use simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters_Integral' for this
    method, especially if the solution to the homogeneous equation has
    trigonometric functions in it.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, log
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 3) - 3*f(x).diff(x, 2) +
    ... 3*f(x).diff(x) - f(x) - exp(x)*log(x), f(x),
    ... hint='nth_linear_constant_coeff_variation_of_parameters'))
           /       /       /     x*log(x)   11*x\\\  x
    f(x) = |C1 + x*|C2 + x*|C3 + -------- - ----|||*e
           \       \       \        6        36 ///

    References
    ==========

    - https://en.wikipedia.org/wiki/Variation_of_parameters
    - https://planetmath.org/VariationOfParameters
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 233

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_variation_of_parameters"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)

        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if self.r[-1]:
                return True
            else:
                return False
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        homogen_sol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        homogen_sol = Eq(f(x), homogen_sol_rhs)
        homogen_sol = _solve_variation_of_parameters(eq, f(x), roots, homogen_sol, order, self.r, simplify_flag)
        if simplify_flag:
            homogen_sol = _get_simplified_sol([homogen_sol], f(x), collectterms)
        return [homogen_sol]


class NthLinearConstantCoeffUndeterminedCoefficients(SingleODESolver):
    r"""
    Solves an `n`\th order linear differential equation with constant
    coefficients using the method of undetermined coefficients.

    This method works on differential equations of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x)
                + a_0 f(x) = P(x)\text{,}

    where `P(x)` is a function that has a finite number of linearly
    independent derivatives.

    Functions that fit this requirement are finite sums functions of the form
    `a x^i e^{b x} \sin(c x + d)` or `a x^i e^{b x} \cos(c x + d)`, where `i`
    is a non-negative integer and `a`, `b`, `c`, and `d` are constants.  For
    example any polynomial in `x`, functions like `x^2 e^{2 x}`, `x \sin(x)`,
    and `e^x \cos(x)` can all be used.  Products of `\sin`'s and `\cos`'s have
    a finite number of derivatives, because they can be expanded into `\sin(a
    x)` and `\cos(b x)` terms.  However, SymPy currently cannot do that
    expansion, so you will need to manually rewrite the expression in terms of
    the above to use this method.  So, for example, you will need to manually
    convert `\sin^2(x)` into `(1 + \cos(2 x))/2` to properly apply the method
    of undetermined coefficients on it.

    This method works by creating a trial function from the expression and all
    of its linear independent derivatives and substituting them into the
    original ODE.  The coefficients for each term will be a system of linear
    equations, which are be solved for and substituted, giving the solution.
    If any of the trial functions are linearly dependent on the solution to
    the homogeneous equation, they are multiplied by sufficient `x` to make
    them linearly independent.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 2) + 2*f(x).diff(x) + f(x) -
    ... 4*exp(-x)*x**2 + cos(2*x), f(x),
    ... hint='nth_linear_constant_coeff_undetermined_coefficients'))
           /       /      3\\
           |       |     x ||  -x   4*sin(2*x)   3*cos(2*x)
    f(x) = |C1 + x*|C2 + --||*e   - ---------- + ----------
           \       \     3 //           25           25

    References
    ==========

    - https://en.wikipedia.org/wiki/Method_of_undetermined_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 221

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_undetermined_coefficients"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        does_match = False
        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if self.r[-1]:
                eq_homogeneous = Add(eq, -self.r[-1])
                undetcoeff = _undetermined_coefficients_match(self.r[-1], x, func, eq_homogeneous)
                if undetcoeff['test']:
                    self.trialset = undetcoeff['trialset']
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        homogen_sol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        homogen_sol = Eq(f(x), homogen_sol_rhs)
        self.r.update({'list': roots, 'sol': homogen_sol, 'simpliy_flag': simplify_flag})
        gsol = _solve_undetermined_coefficients(eq, f(x), order, self.r, self.trialset)
        if simplify_flag:
            gsol = _get_simplified_sol([gsol], f(x), collectterms)
        return [gsol]


class NthLinearEulerEqHomogeneous(SingleODESolver):
    r"""
    Solves an `n`\th order linear homogeneous variable-coefficient
    Cauchy-Euler equidimensional ordinary differential equation.

    This is an equation with form `0 = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    These equations can be solved in a general manner, by substituting
    solutions of the form `f(x) = x^r`, and deriving a characteristic equation
    for `r`.  When there are repeated roots, we include extra terms of the
    form `C_{r k} \ln^k(x) x^r`, where `C_{r k}` is an arbitrary integration
    constant, `r` is a root of the characteristic equation, and `k` ranges
    over the multiplicity of `r`.  In the cases where the roots are complex,
    solutions of the form `C_1 x^a \sin(b \log(x)) + C_2 x^a \cos(b \log(x))`
    are returned, based on expansions with Euler's formula.  The general
    solution is the sum of the terms found.  If SymPy cannot find exact roots
    to the characteristic equation, a
    :py:obj:`~.ComplexRootOf` instance will be returned
    instead.

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(4*x**2*f(x).diff(x, 2) + f(x), f(x),
    ... hint='nth_linear_euler_eq_homogeneous')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), sqrt(x)*(C1 + C2*log(x)))

    Note that because this method does not involve integration, there is no
    ``nth_linear_euler_eq_homogeneous_Integral`` hint.

    The following is for internal use:

    - ``returns = 'sol'`` returns the solution to the ODE.
    - ``returns = 'list'`` returns a list of linearly independent solutions,
      corresponding to the fundamental solution set, for use with non
      homogeneous solution methods like variation of parameters and
      undetermined coefficients.  Note that, though the solutions should be
      linearly independent, this function does not explicitly check that.  You
      can do ``assert simplify(wronskian(sollist)) != 0`` to check for linear
      independence.  Also, ``assert len(sollist) == order`` will need to pass.
    - ``returns = 'both'``, return a dictionary ``{'sol': <solution to ODE>,
      'list': <list of linearly independent solutions>}``.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)*x**2 - 4*f(x).diff(x)*x + 6*f(x)
    >>> pprint(dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_homogeneous'))
            2
    f(x) = x *(C1 + C2*x)

    References
    ==========

    - https://en.wikipedia.org/wiki/Cauchy%E2%80%93Euler_equation
    - C. Bender & S. Orszag, "Advanced Mathematical Methods for Scientists and
      Engineers", Springer 1999, pp. 12

    # indirect doctest

    """
    hint = "nth_linear_euler_eq_homogeneous"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if not self.r[-1]:
                does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        fx = self.ode_problem.func
        eq = self.ode_problem.eq
        homogen_sol = _get_euler_characteristic_eq_sols(eq, fx, self.r)[0]
        return [homogen_sol]


class NthLinearEulerEqNonhomogeneousVariationOfParameters(SingleODESolver):
    r"""
    Solves an `n`\th order linear non homogeneous Cauchy-Euler equidimensional
    ordinary differential equation using variation of parameters.

    This is an equation with form `g(x) = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    This method works by assuming that the particular solution takes the form

    .. math:: \sum_{x=1}^{n} c_i(x) y_i(x) {a_n} {x^n} \text{, }

    where `y_i` is the `i`\th solution to the homogeneous equation.  The
    solution is then solved using Wronskian's and Cramer's Rule.  The
    particular solution is given by multiplying eq given below with `a_n x^{n}`

    .. math:: \sum_{x=1}^n \left( \int \frac{W_i(x)}{W(x)} \, dx
                \right) y_i(x) \text{, }

    where `W(x)` is the Wronskian of the fundamental system (the system of `n`
    linearly independent solutions to the homogeneous equation), and `W_i(x)`
    is the Wronskian of the fundamental system with the `i`\th column replaced
    with `[0, 0, \cdots, 0, \frac{x^{- n}}{a_n} g{\left(x \right)}]`.

    This method is general enough to solve any `n`\th order inhomogeneous
    linear differential equation, but sometimes SymPy cannot simplify the
    Wronskian well enough to integrate it.  If this method hangs, try using the
    ``nth_linear_constant_coeff_variation_of_parameters_Integral`` hint and
    simplifying the integrals manually.  Also, prefer using
    ``nth_linear_constant_coeff_undetermined_coefficients`` when it
    applies, because it does not use integration, making it faster and more
    reliable.

    Warning, using simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters' in
    :py:meth:`~sympy.solvers.ode.dsolve` may cause it to hang, because it will
    not attempt to simplify the Wronskian before integrating.  It is
    recommended that you only use simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters_Integral' for this
    method, especially if the solution to the homogeneous equation has
    trigonometric functions in it.

    Examples
    ========

    >>> from sympy import Function, dsolve, Derivative
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - x**4
    >>> dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_nonhomogeneous_variation_of_parameters').expand()
    Eq(f(x), C1*x + C2*x**2 + x**4/6)

    """
    hint = "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if self.r[-1]:
                does_match = True

        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        homogen_sol, roots = _get_euler_characteristic_eq_sols(eq, f(x), self.r)
        self.r[-1] = self.r[-1]/self.r[order]
        sol = _solve_variation_of_parameters(eq, f(x), roots, homogen_sol, order, self.r, simplify_flag)

        return [Eq(f(x), homogen_sol.rhs + (sol.rhs - homogen_sol.rhs)*self.r[order])]


class NthLinearEulerEqNonhomogeneousUndeterminedCoefficients(SingleODESolver):
    r"""
    Solves an `n`\th order linear non homogeneous Cauchy-Euler equidimensional
    ordinary differential equation using undetermined coefficients.

    This is an equation with form `g(x) = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    These equations can be solved in a general manner, by substituting
    solutions of the form `x = exp(t)`, and deriving a characteristic equation
    of form `g(exp(t)) = b_0 f(t) + b_1 f'(t) + b_2 f''(t) \cdots` which can
    be then solved by nth_linear_constant_coeff_undetermined_coefficients if
    g(exp(t)) has finite number of linearly independent derivatives.

    Functions that fit this requirement are finite sums functions of the form
    `a x^i e^{b x} \sin(c x + d)` or `a x^i e^{b x} \cos(c x + d)`, where `i`
    is a non-negative integer and `a`, `b`, `c`, and `d` are constants.  For
    example any polynomial in `x`, functions like `x^2 e^{2 x}`, `x \sin(x)`,
    and `e^x \cos(x)` can all be used.  Products of `\sin`'s and `\cos`'s have
    a finite number of derivatives, because they can be expanded into `\sin(a
    x)` and `\cos(b x)` terms.  However, SymPy currently cannot do that
    expansion, so you will need to manually rewrite the expression in terms of
    the above to use this method.  So, for example, you will need to manually
    convert `\sin^2(x)` into `(1 + \cos(2 x))/2` to properly apply the method
    of undetermined coefficients on it.

    After replacement of x by exp(t), this method works by creating a trial function
    from the expression and all of its linear independent derivatives and
    substituting them into the original ODE.  The coefficients for each term
    will be a system of linear equations, which are be solved for and
    substituted, giving the solution. If any of the trial functions are linearly
    dependent on the solution to the homogeneous equation, they are multiplied
    by sufficient `x` to make them linearly independent.

    Examples
    ========

    >>> from sympy import dsolve, Function, Derivative, log
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - log(x)
    >>> dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients').expand()
    Eq(f(x), C1*x + C2*x**2 + log(x)/2 + 3/4)

    """
    hint = "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if self.r[-1]:
                e, re = posify(self.r[-1].subs(x, exp(x)))
                undetcoeff = _undetermined_coefficients_match(e.subs(re), x)
                if undetcoeff['test']:
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        chareq, eq, symbol = S.Zero, S.Zero, Dummy('x')
        for i in self.r.keys():
            if i >= 0:
                chareq += (self.r[i]*diff(x**symbol, x, i)*x**-symbol).expand()

        for i in range(1, degree(Poly(chareq, symbol))+1):
            eq += chareq.coeff(symbol**i)*diff(f(x), x, i)

        if chareq.as_coeff_add(symbol)[0]:
            eq += chareq.as_coeff_add(symbol)[0]*f(x)
        e, re = posify(self.r[-1].subs(x, exp(x)))
        eq += e.subs(re)

        self.const_undet_instance = NthLinearConstantCoeffUndeterminedCoefficients(SingleODEProblem(eq, f(x), x))
        sol = self.const_undet_instance.get_general_solution(simplify = simplify_flag)[0]
        sol = sol.subs(x, log(x)) # type: ignore
        sol = sol.subs(f(log(x)), f(x)).expand() # type: ignore

        return [sol]


class SecondLinearBessel(SingleODESolver):
    r"""
    Gives solution of the Bessel differential equation

    .. math :: x^2 \frac{d^2y}{dx^2} + x \frac{dy}{dx} y(x) + (x^2-n^2) y(x)

    if `n` is integer then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))`` as both the solutions are linearly independent else if
    `n` is a fraction then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 besselj(-n,x))`` which can also transform into ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))``.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import Symbol
    >>> v = Symbol('v', positive=True)
    >>> from sympy import dsolve, Function
    >>> f = Function('f')
    >>> y = f(x)
    >>> genform = x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - v**2)*y
    >>> dsolve(genform)
    Eq(f(x), C1*besselj(v, x) + C2*bessely(v, x))

    References
    ==========

    https://math24.net/bessel-differential-equation.html

    """
    hint = "2nd_linear_bessel"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f.diff(x)
        a = Wild('a', exclude=[f,df])
        b = Wild('b', exclude=[x, f,df])
        a4 = Wild('a4', exclude=[x,f,df])
        b4 = Wild('b4', exclude=[x,f,df])
        c4 = Wild('c4', exclude=[x,f,df])
        d4 = Wild('d4', exclude=[x,f,df])
        a3 = Wild('a3', exclude=[f, df, f.diff(x, 2)])
        b3 = Wild('b3', exclude=[f, df, f.diff(x, 2)])
        c3 = Wild('c3', exclude=[f, df, f.diff(x, 2)])
        deq = a3*(f.diff(x, 2)) + b3*df + c3*f
        r = collect(eq,
            [f.diff(x, 2), df, f]).match(deq)
        if order == 2 and r:
            if not all(r[key].is_polynomial() for key in r):
                n, d = eq.as_numer_denom()
                eq = expand(n)
                r = collect(eq,
                    [f.diff(x, 2), df, f]).match(deq)

        if r and r[a3] != 0:
            # leading coeff of f(x).diff(x, 2)
            coeff = factor(r[a3]).match(a4*(x-b)**b4)

            if coeff:
            # if coeff[b4] = 0 means constant coefficient
                if coeff[b4] == 0:
                    return False
                point = coeff[b]
            else:
                return False

            if point:
                r[a3] = simplify(r[a3].subs(x, x+point))
                r[b3] = simplify(r[b3].subs(x, x+point))
                r[c3] = simplify(r[c3].subs(x, x+point))

            # making a3 in the form of x**2
            r[a3] = cancel(r[a3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            r[b3] = cancel(r[b3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            r[c3] = cancel(r[c3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            # checking if b3 is of form c*(x-b)
            coeff1 = factor(r[b3]).match(a4*(x))
            if coeff1 is None:
                return False
            # c3 maybe of very complex form so I am simply checking (a - b) form
            # if yes later I will match with the standard form of bessel in a and b
            # a, b are wild variable defined above.
            _coeff2 = expand(r[c3]).match(a - b)
            if _coeff2 is None:
                return False
            # matching with standard form for c3
            coeff2 = factor(_coeff2[a]).match(c4**2*(x)**(2*a4))
            if coeff2 is None:
                return False

            if _coeff2[b] == 0:
                coeff2[d4] = 0
            else:
                coeff2[d4] = factor(_coeff2[b]).match(d4**2)[d4]

            self.rn = {'n':coeff2[d4], 'a4':coeff2[c4], 'd4':coeff2[a4]}
            self.rn['c4'] = coeff1[a4]
            self.rn['b4'] = point
            return True
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        n = self.rn['n']
        a4 = self.rn['a4']
        c4 = self.rn['c4']
        d4 = self.rn['d4']
        b4 = self.rn['b4']
        n = sqrt(n**2 + Rational(1, 4)*(c4 - 1)**2)
        (C1, C2) = self.ode_problem.get_numbered_constants(num=2)
        return [Eq(f(x), ((x**(Rational(1-c4,2)))*(C1*besselj(n/d4,a4*x**d4/d4)
            + C2*bessely(n/d4,a4*x**d4/d4))).subs(x, x-b4))]


class SecondLinearAiry(SingleODESolver):
    r"""
    Gives solution of the Airy differential equation

    .. math :: \frac{d^2y}{dx^2} + (a + b x) y(x) = 0

    in terms of Airy special functions airyai and airybi.

    Examples
    ========

    >>> from sympy import dsolve, Function
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = f(x).diff(x, 2) - x*f(x)
    >>> dsolve(eq)
    Eq(f(x), C1*airyai(x) + C2*airybi(x))
    """
    hint = "2nd_linear_airy"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f.diff(x)
        a4 = Wild('a4', exclude=[x,f,df])
        b4 = Wild('b4', exclude=[x,f,df])
        match = self.ode_problem.get_linear_coefficients(eq, f, order)
        does_match = False
        if order == 2 and match and match[2] != 0:
            if match[1].is_zero:
                self.rn = cancel(match[0]/match[2]).match(a4+b4*x)
                if self.rn and self.rn[b4] != 0:
                    self.rn = {'b':self.rn[a4],'m':self.rn[b4]}
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        (C1, C2) = self.ode_problem.get_numbered_constants(num=2)
        b = self.rn['b']
        m = self.rn['m']
        if m.is_positive:
            arg = - b/cbrt(m)**2 - cbrt(m)*x
        elif m.is_negative:
            arg = - b/cbrt(-m)**2 + cbrt(-m)*x
        else:
            arg = - b/cbrt(-m)**2 + cbrt(-m)*x

        return [Eq(f(x), C1*airyai(arg) + C2*airybi(arg))]


class LieGroup(SingleODESolver):
    r"""
    This hint implements the Lie group method of solving first order differential
    equations. The aim is to convert the given differential equation from the
    given coordinate system into another coordinate system where it becomes
    invariant under the one-parameter Lie group of translations. The converted
    ODE can be easily solved by quadrature. It makes use of the
    :py:meth:`sympy.solvers.ode.infinitesimals` function which returns the
    infinitesimals of the transformation.

    The coordinates `r` and `s` can be found by solving the following Partial
    Differential Equations.

    .. math :: \xi\frac{\partial r}{\partial x} + \eta\frac{\partial r}{\partial y}
                  = 0

    .. math :: \xi\frac{\partial s}{\partial x} + \eta\frac{\partial s}{\partial y}
                  = 1

    The differential equation becomes separable in the new coordinate system

    .. math :: \frac{ds}{dr} = \frac{\frac{\partial s}{\partial x} +
                 h(x, y)\frac{\partial s}{\partial y}}{
                 \frac{\partial r}{\partial x} + h(x, y)\frac{\partial r}{\partial y}}

    After finding the solution by integration, it is then converted back to the original
    coordinate system by substituting `r` and `s` in terms of `x` and `y` again.

    Examples
    ========

    >>> from sympy import Function, dsolve, exp, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x) + 2*x*f(x) - x*exp(-x**2), f(x),
    ... hint='lie_group'))
           /      2\    2
           |     x |  -x
    f(x) = |C1 + --|*e
           \     2 /


    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14

    """
    hint = "lie_group"
    has_integral = False

    def _has_additional_params(self):
        return 'xi' in self.ode_problem.params and 'eta' in self.ode_problem.params

    def _matches(self):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f(x).diff(x)
        y = Dummy('y')
        d = Wild('d', exclude=[df, f(x).diff(x, 2)])
        e = Wild('e', exclude=[df])
        does_match = False
        if self._has_additional_params() and order == 1:
            xi = self.ode_problem.params['xi']
            eta = self.ode_problem.params['eta']
            self.r3 = {'xi': xi, 'eta': eta}
            r = collect(eq, df, exact=True).match(d + e * df)
            if r:
                r['d'] = d
                r['e'] = e
                r['y'] = y
                r[d] = r[d].subs(f(x), y)
                r[e] = r[e].subs(f(x), y)
                self.r3.update(r)
            does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        x = self.ode_problem.sym
        func = self.ode_problem.func
        order = self.ode_problem.order
        df = func.diff(x)

        try:
            eqsol = solve(eq, df)
        except NotImplementedError:
            eqsol = []

        desols = []
        for s in eqsol:
            sol = _ode_lie_group(s, func, order, match=self.r3)
            if sol:
                desols.extend(sol)

        if desols == []:
            raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                + " the lie group method")
        return desols


solver_map = {
    'factorable': Factorable,
    'nth_linear_constant_coeff_homogeneous': NthLinearConstantCoeffHomogeneous,
    'nth_linear_euler_eq_homogeneous': NthLinearEulerEqHomogeneous,
    'nth_linear_constant_coeff_undetermined_coefficients': NthLinearConstantCoeffUndeterminedCoefficients,
    'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients': NthLinearEulerEqNonhomogeneousUndeterminedCoefficients,
    'separable': Separable,
    '1st_exact': FirstExact,
    '1st_linear': FirstLinear,
    'Bernoulli': Bernoulli,
    'Riccati_special_minus2': RiccatiSpecial,
    '1st_rational_riccati': RationalRiccati,
    '1st_homogeneous_coeff_best': HomogeneousCoeffBest,
    '1st_homogeneous_coeff_subs_indep_div_dep': HomogeneousCoeffSubsIndepDivDep,
    '1st_homogeneous_coeff_subs_dep_div_indep': HomogeneousCoeffSubsDepDivIndep,
    'almost_linear': AlmostLinear,
    'linear_coefficients': LinearCoefficients,
    'separable_reduced': SeparableReduced,
    'nth_linear_constant_coeff_variation_of_parameters': NthLinearConstantCoeffVariationOfParameters,
    'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters': NthLinearEulerEqNonhomogeneousVariationOfParameters,
    'Liouville': Liouville,
    '2nd_linear_airy': SecondLinearAiry,
    '2nd_linear_bessel': SecondLinearBessel,
    '2nd_hypergeometric': SecondHypergeometric,
    'nth_order_reducible': NthOrderReducible,
    '2nd_nonlinear_autonomous_conserved': SecondNonlinearAutonomousConserved,
    'nth_algebraic': NthAlgebraic,
    'lie_group': LieGroup,
    }

# Avoid circular import:
from .ode import dsolve, ode_sol_simplicity, odesimp, homogeneous_order
<|endoftext|>r"""
This module contains :py:meth:`~sympy.solvers.ode.dsolve` and different helper
functions that it uses.

:py:meth:`~sympy.solvers.ode.dsolve` solves ordinary differential equations.
See the docstring on the various functions for their uses.  Note that partial
differential equations support is in ``pde.py``.  Note that hint functions
have docstrings describing their various methods, but they are intended for
internal use.  Use ``dsolve(ode, func, hint=hint)`` to solve an ODE using a
specific hint.  See also the docstring on
:py:meth:`~sympy.solvers.ode.dsolve`.

**Functions in this module**

    These are the user functions in this module:

    - :py:meth:`~sympy.solvers.ode.dsolve` - Solves ODEs.
    - :py:meth:`~sympy.solvers.ode.classify_ode` - Classifies ODEs into
      possible hints for :py:meth:`~sympy.solvers.ode.dsolve`.
    - :py:meth:`~sympy.solvers.ode.checkodesol` - Checks if an equation is the
      solution to an ODE.
    - :py:meth:`~sympy.solvers.ode.homogeneous_order` - Returns the
      homogeneous order of an expression.
    - :py:meth:`~sympy.solvers.ode.infinitesimals` - Returns the infinitesimals
      of the Lie group of point transformations of an ODE, such that it is
      invariant.
    - :py:meth:`~sympy.solvers.ode.checkinfsol` - Checks if the given infinitesimals
      are the actual infinitesimals of a first order ODE.

    These are the non-solver helper functions that are for internal use.  The
    user should use the various options to
    :py:meth:`~sympy.solvers.ode.dsolve` to obtain the functionality provided
    by these functions:

    - :py:meth:`~sympy.solvers.ode.ode.odesimp` - Does all forms of ODE
      simplification.
    - :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity` - A key function for
      comparing solutions by simplicity.
    - :py:meth:`~sympy.solvers.ode.constantsimp` - Simplifies arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode.constant_renumber` - Renumber arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode._handle_Integral` - Evaluate unevaluated
      Integrals.

    See also the docstrings of these functions.

**Currently implemented solver methods**

The following methods are implemented for solving ordinary differential
equations.  See the docstrings of the various hint functions for more
information on each (run ``help(ode)``):

  - 1st order separable differential equations.
  - 1st order differential equations whose coefficients or `dx` and `dy` are
    functions homogeneous of the same order.
  - 1st order exact differential equations.
  - 1st order linear differential equations.
  - 1st order Bernoulli differential equations.
  - Power series solutions for first order differential equations.
  - Lie Group method of solving first order differential equations.
  - 2nd order Liouville differential equations.
  - Power series solutions for second order differential equations
    at ordinary and regular singular points.
  - `n`\th order differential equation that can be solved with algebraic
    rearrangement and integration.
  - `n`\th order linear homogeneous differential equation with constant
    coefficients.
  - `n`\th order linear inhomogeneous differential equation with constant
    coefficients using the method of undetermined coefficients.
  - `n`\th order linear inhomogeneous differential equation with constant
    coefficients using the method of variation of parameters.

**Philosophy behind this module**

This module is designed to make it easy to add new ODE solving methods without
having to mess with the solving code for other methods.  The idea is that
there is a :py:meth:`~sympy.solvers.ode.classify_ode` function, which takes in
an ODE and tells you what hints, if any, will solve the ODE.  It does this
without attempting to solve the ODE, so it is fast.  Each solving method is a
hint, and it has its own function, named ``ode_<hint>``.  That function takes
in the ODE and any match expression gathered by
:py:meth:`~sympy.solvers.ode.classify_ode` and returns a solved result.  If
this result has any integrals in it, the hint function will return an
unevaluated :py:class:`~sympy.integrals.integrals.Integral` class.
:py:meth:`~sympy.solvers.ode.dsolve`, which is the user wrapper function
around all of this, will then call :py:meth:`~sympy.solvers.ode.ode.odesimp` on
the result, which, among other things, will attempt to solve the equation for
the dependent variable (the function we are solving for), simplify the
arbitrary constants in the expression, and evaluate any integrals, if the hint
allows it.

**How to add new solution methods**

If you have an ODE that you want :py:meth:`~sympy.solvers.ode.dsolve` to be
able to solve, try to avoid adding special case code here.  Instead, try
finding a general method that will solve your ODE, as well as others.  This
way, the :py:mod:`~sympy.solvers.ode` module will become more robust, and
unhindered by special case hacks.  WolphramAlpha and Maple's
DETools[odeadvisor] function are two resources you can use to classify a
specific ODE.  It is also better for a method to work with an `n`\th order ODE
instead of only with specific orders, if possible.

To add a new method, there are a few things that you need to do.  First, you
need a hint name for your method.  Try to name your hint so that it is
unambiguous with all other methods, including ones that may not be implemented
yet.  If your method uses integrals, also include a ``hint_Integral`` hint.
If there is more than one way to solve ODEs with your method, include a hint
for each one, as well as a ``<hint>_best`` hint.  Your ``ode_<hint>_best()``
function should choose the best using min with ``ode_sol_simplicity`` as the
key argument.  See
:obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest`, for example.
The function that uses your method will be called ``ode_<hint>()``, so the
hint must only use characters that are allowed in a Python function name
(alphanumeric characters and the underscore '``_``' character).  Include a
function for every hint, except for ``_Integral`` hints
(:py:meth:`~sympy.solvers.ode.dsolve` takes care of those automatically).
Hint names should be all lowercase, unless a word is commonly capitalized
(such as Integral or Bernoulli).  If you have a hint that you do not want to
run with ``all_Integral`` that does not have an ``_Integral`` counterpart (such
as a best hint that would defeat the purpose of ``all_Integral``), you will
need to remove it manually in the :py:meth:`~sympy.solvers.ode.dsolve` code.
See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for
guidelines on writing a hint name.

Determine *in general* how the solutions returned by your method compare with
other methods that can potentially solve the same ODEs.  Then, put your hints
in the :py:data:`~sympy.solvers.ode.allhints` tuple in the order that they
should be called.  The ordering of this tuple determines which hints are
default.  Note that exceptions are ok, because it is easy for the user to
choose individual hints with :py:meth:`~sympy.solvers.ode.dsolve`.  In
general, ``_Integral`` variants should go at the end of the list, and
``_best`` variants should go before the various hints they apply to.  For
example, the ``undetermined_coefficients`` hint comes before the
``variation_of_parameters`` hint because, even though variation of parameters
is more general than undetermined coefficients, undetermined coefficients
generally returns cleaner results for the ODEs that it can solve than
variation of parameters does, and it does not require integration, so it is
much faster.

Next, you need to have a match expression or a function that matches the type
of the ODE, which you should put in :py:meth:`~sympy.solvers.ode.classify_ode`
(if the match function is more than just a few lines.  It should match the
ODE without solving for it as much as possible, so that
:py:meth:`~sympy.solvers.ode.classify_ode` remains fast and is not hindered by
bugs in solving code.  Be sure to consider corner cases.  For example, if your
solution method involves dividing by something, make sure you exclude the case
where that division will be 0.

In most cases, the matching of the ODE will also give you the various parts
that you need to solve it.  You should put that in a dictionary (``.match()``
will do this for you), and add that as ``matching_hints['hint'] = matchdict``
in the relevant part of :py:meth:`~sympy.solvers.ode.classify_ode`.
:py:meth:`~sympy.solvers.ode.classify_ode` will then send this to
:py:meth:`~sympy.solvers.ode.dsolve`, which will send it to your function as
the ``match`` argument.  Your function should be named ``ode_<hint>(eq, func,
order, match)`.  If you need to send more information, put it in the ``match``
dictionary.  For example, if you had to substitute in a dummy variable in
:py:meth:`~sympy.solvers.ode.classify_ode` to match the ODE, you will need to
pass it to your function using the `match` dict to access it.  You can access
the independent variable using ``func.args[0]``, and the dependent variable
(the function you are trying to solve for) as ``func.func``.  If, while trying
to solve the ODE, you find that you cannot, raise ``NotImplementedError``.
:py:meth:`~sympy.solvers.ode.dsolve` will catch this error with the ``all``
meta-hint, rather than causing the whole routine to fail.

Add a docstring to your function that describes the method employed.  Like
with anything else in SymPy, you will need to add a doctest to the docstring,
in addition to real tests in ``test_ode.py``.  Try to maintain consistency
with the other hint functions' docstrings.  Add your method to the list at the
top of this docstring.  Also, add your method to ``ode.rst`` in the
``docs/src`` directory, so that the Sphinx docs will pull its docstring into
the main SymPy documentation.  Be sure to make the Sphinx documentation by
running ``make html`` from within the doc directory to verify that the
docstring formats correctly.

If your solution method involves integrating, use :py:obj:`~.Integral` instead of
:py:meth:`~sympy.core.expr.Expr.integrate`.  This allows the user to bypass
hard/slow integration by using the ``_Integral`` variant of your hint.  In
most cases, calling :py:meth:`sympy.core.basic.Basic.doit` will integrate your
solution.  If this is not the case, you will need to write special code in
:py:meth:`~sympy.solvers.ode.ode._handle_Integral`.  Arbitrary constants should be
symbols named ``C1``, ``C2``, and so on.  All solution methods should return
an equality instance.  If you need an arbitrary number of arbitrary constants,
you can use ``constants = numbered_symbols(prefix='C', cls=Symbol, start=1)``.
If it is possible to solve for the dependent function in a general way, do so.
Otherwise, do as best as you can, but do not call solve in your
``ode_<hint>()`` function.  :py:meth:`~sympy.solvers.ode.ode.odesimp` will attempt
to solve the solution for you, so you do not need to do that.  Lastly, if your
ODE has a common simplification that can be applied to your solutions, you can
add a special case in :py:meth:`~sympy.solvers.ode.ode.odesimp` for it.  For
example, solutions returned from the ``1st_homogeneous_coeff`` hints often
have many :obj:`~sympy.functions.elementary.exponential.log` terms, so
:py:meth:`~sympy.solvers.ode.ode.odesimp` calls
:py:meth:`~sympy.simplify.simplify.logcombine` on them (it also helps to write
the arbitrary constant as ``log(C1)`` instead of ``C1`` in this case).  Also
consider common ways that you can rearrange your solution to have
:py:meth:`~sympy.solvers.ode.constantsimp` take better advantage of it.  It is
better to put simplification in :py:meth:`~sympy.solvers.ode.ode.odesimp` than in
your method, because it can then be turned off with the simplify flag in
:py:meth:`~sympy.solvers.ode.dsolve`.  If you have any extraneous
simplification in your function, be sure to only run it using ``if
match.get('simplify', True):``, especially if it can be slow or if it can
reduce the domain of the solution.

Finally, as with every contribution to SymPy, your method will need to be
tested.  Add a test for each method in ``test_ode.py``.  Follow the
conventions there, i.e., test the solver using ``dsolve(eq, f(x),
hint=your_hint)``, and also test the solution using
:py:meth:`~sympy.solvers.ode.checkodesol` (you can put these in a separate
tests and skip/XFAIL if it runs too slow/does not work).  Be sure to call your
hint specifically in :py:meth:`~sympy.solvers.ode.dsolve`, that way the test
will not be broken simply by the introduction of another matching hint.  If your
method works for higher order (>1) ODEs, you will need to run ``sol =
constant_renumber(sol, 'C', 1, order)`` for each solution, where ``order`` is
the order of the ODE.  This is because ``constant_renumber`` renumbers the
arbitrary constants by printing order, which is platform dependent.  Try to
test every corner case of your solver, including a range of orders if it is a
`n`\th order solver, but if your solver is slow, such as if it involves hard
integration, try to keep the test run time down.

Feel free to refactor existing hints to avoid duplicating code or creating
inconsistencies.  If you can show that your method exactly duplicates an
existing method, including in the simplicity and speed of obtaining the
solutions, then you can remove the old, less general method.  The existing
code is tested extensively in ``test_ode.py``, so if anything is broken, one
of those tests will surely fail.

"""

from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import (Function, Derivative, AppliedUndef, diff,
    expand, expand_mul, Subs)
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal

from sympy.logic.boolalg import (BooleanAtom, BooleanTrue,
                                BooleanFalse)
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import (Poly, terms_gcd, PolynomialError, lcm)
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import (collect, logcombine, powsimp,  # type: ignore
    separatevars, simplify, cse)
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve

from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve


#: This is a list of hints in the order that they should be preferred by
#: :py:meth:`~sympy.solvers.ode.classify_ode`. In general, hints earlier in the
#: list should produce simpler solutions than those later in the list (for
#: ODEs that fit both).  For now, the order of this list is based on empirical
#: observations by the developers of SymPy.
#:
#: The hint used by :py:meth:`~sympy.solvers.ode.dsolve` for a specific ODE
#: can be overridden (see the docstring).
#:
#: In general, ``_Integral`` hints are grouped at the end of the list, unless
#: there is a method that returns an unevaluable integral most of the time
#: (which go near the end of the list anyway).  ``default``, ``all``,
#: ``best``, and ``all_Integral`` meta-hints should not be included in this
#: list, but ``_best`` and ``_Integral`` hints should be included.
allhints = (
    "factorable",
    "nth_algebraic",
    "separable",
    "1st_exact",
    "1st_linear",
    "Bernoulli",
    "1st_rational_riccati",
    "Riccati_special_minus2",
    "1st_homogeneous_coeff_best",
    "1st_homogeneous_coeff_subs_indep_div_dep",
    "1st_homogeneous_coeff_subs_dep_div_indep",
    "almost_linear",
    "linear_coefficients",
    "separable_reduced",
    "1st_power_series",
    "lie_group",
    "nth_linear_constant_coeff_homogeneous",
    "nth_linear_euler_eq_homogeneous",
    "nth_linear_constant_coeff_undetermined_coefficients",
    "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients",
    "nth_linear_constant_coeff_variation_of_parameters",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters",
    "Liouville",
    "2nd_linear_airy",
    "2nd_linear_bessel",
    "2nd_hypergeometric",
    "2nd_hypergeometric_Integral",
    "nth_order_reducible",
    "2nd_power_series_ordinary",
    "2nd_power_series_regular",
    "nth_algebraic_Integral",
    "separable_Integral",
    "1st_exact_Integral",
    "1st_linear_Integral",
    "Bernoulli_Integral",
    "1st_homogeneous_coeff_subs_indep_div_dep_Integral",
    "1st_homogeneous_coeff_subs_dep_div_indep_Integral",
    "almost_linear_Integral",
    "linear_coefficients_Integral",
    "separable_reduced_Integral",
    "nth_linear_constant_coeff_variation_of_parameters_Integral",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters_Integral",
    "Liouville_Integral",
    "2nd_nonlinear_autonomous_conserved",
    "2nd_nonlinear_autonomous_conserved_Integral",
    )



def get_numbered_constants(eq, num=1, start=1, prefix='C'):
    """
    Returns a list of constants that do not occur
    in eq already.
    """

    ncs = iter_numbered_constants(eq, start, prefix)
    Cs = [next(ncs) for i in range(num)]
    return (Cs[0] if num == 1 else tuple(Cs))


def iter_numbered_constants(eq, start=1, prefix='C'):
    """
    Returns an iterator of constants that do not occur
    in eq already.
    """

    if isinstance(eq, (Expr, Eq)):
        eq = [eq]
    elif not iterable(eq):
        raise ValueError("Expected Expr or iterable but got %s" % eq)

    atom_set = set().union(*[i.free_symbols for i in eq])
    func_set = set().union(*[i.atoms(Function) for i in eq])
    if func_set:
        atom_set |= {Symbol(str(f.func)) for f in func_set}
    return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)


def dsolve(eq, func=None, hint="default", simplify=True,
    ics= None, xi=None, eta=None, x0=0, n=6, **kwargs):
    r"""
    Solves any (supported) kind of ordinary differential equation and
    system of ordinary differential equations.

    For single ordinary differential equation
    =========================================

    It is classified under this when number of equation in ``eq`` is one.
    **Usage**

        ``dsolve(eq, f(x), hint)`` -> Solve ordinary differential equation
        ``eq`` for function ``f(x)``, using method ``hint``.

    **Details**

        ``eq`` can be any supported ordinary differential equation (see the
            :py:mod:`~sympy.solvers.ode` docstring for supported methods).
            This can either be an :py:class:`~sympy.core.relational.Equality`,
            or an expression, which is assumed to be equal to ``0``.

        ``f(x)`` is a function of one variable whose derivatives in that
            variable make up the ordinary differential equation ``eq``.  In
            many cases it is not necessary to provide this; it will be
            autodetected (and an error raised if it could not be detected).

        ``hint`` is the solving method that you want dsolve to use.  Use
            ``classify_ode(eq, f(x))`` to get all of the possible hints for an
            ODE.  The default hint, ``default``, will use whatever hint is
            returned first by :py:meth:`~sympy.solvers.ode.classify_ode`.  See
            Hints below for more options that you can use for hint.

        ``simplify`` enables simplification by
            :py:meth:`~sympy.solvers.ode.ode.odesimp`.  See its docstring for more
            information.  Turn this off, for example, to disable solving of
            solutions for ``func`` or simplification of arbitrary constants.
            It will still integrate with this hint. Note that the solution may
            contain more arbitrary constants than the order of the ODE with
            this option enabled.

        ``xi`` and ``eta`` are the infinitesimal functions of an ordinary
            differential equation. They are the infinitesimals of the Lie group
            of point transformations for which the differential equation is
            invariant. The user can specify values for the infinitesimals. If
            nothing is specified, ``xi`` and ``eta`` are calculated using
            :py:meth:`~sympy.solvers.ode.infinitesimals` with the help of various
            heuristics.

        ``ics`` is the set of initial/boundary conditions for the differential equation.
          It should be given in the form of ``{f(x0): x1, f(x).diff(x).subs(x, x2):
          x3}`` and so on.  For power series solutions, if no initial
          conditions are specified ``f(0)`` is assumed to be ``C0`` and the power
          series solution is calculated about 0.

        ``x0`` is the point about which the power series solution of a differential
          equation is to be evaluated.

        ``n`` gives the exponent of the dependent variable up to which the power series
          solution of a differential equation is to be evaluated.

    **Hints**

        Aside from the various solving methods, there are also some meta-hints
        that you can pass to :py:meth:`~sympy.solvers.ode.dsolve`:

        ``default``:
                This uses whatever hint is returned first by
                :py:meth:`~sympy.solvers.ode.classify_ode`. This is the
                default argument to :py:meth:`~sympy.solvers.ode.dsolve`.

        ``all``:
                To make :py:meth:`~sympy.solvers.ode.dsolve` apply all
                relevant classification hints, use ``dsolve(ODE, func,
                hint="all")``.  This will return a dictionary of
                ``hint:solution`` terms.  If a hint causes dsolve to raise the
                ``NotImplementedError``, value of that hint's key will be the
                exception object raised.  The dictionary will also include
                some special keys:

                - ``order``: The order of the ODE.  See also
                  :py:meth:`~sympy.solvers.deutils.ode_order` in
                  ``deutils.py``.
                - ``best``: The simplest hint; what would be returned by
                  ``best`` below.
                - ``best_hint``: The hint that would produce the solution
                  given by ``best``.  If more than one hint produces the best
                  solution, the first one in the tuple returned by
                  :py:meth:`~sympy.solvers.ode.classify_ode` is chosen.
                - ``default``: The solution that would be returned by default.
                  This is the one produced by the hint that appears first in
                  the tuple returned by
                  :py:meth:`~sympy.solvers.ode.classify_ode`.

        ``all_Integral``:
                This is the same as ``all``, except if a hint also has a
                corresponding ``_Integral`` hint, it only returns the
                ``_Integral`` hint.  This is useful if ``all`` causes
                :py:meth:`~sympy.solvers.ode.dsolve` to hang because of a
                difficult or impossible integral.  This meta-hint will also be
                much faster than ``all``, because
                :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive
                routine.

        ``best``:
                To have :py:meth:`~sympy.solvers.ode.dsolve` try all methods
                and return the simplest one.  This takes into account whether
                the solution is solvable in the function, whether it contains
                any Integral classes (i.e.  unevaluatable integrals), and
                which one is the shortest in size.

        See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for
        more info on hints, and the :py:mod:`~sympy.solvers.ode` docstring for
        a list of all supported hints.

    **Tips**

        - You can declare the derivative of an unknown function this way:

            >>> from sympy import Function, Derivative
            >>> from sympy.abc import x # x is the independent variable
            >>> f = Function("f")(x) # f is a function of x
            >>> # f_ will be the derivative of f with respect to x
            >>> f_ = Derivative(f, x)

        - See ``test_ode.py`` for many tests, which serves also as a set of
          examples for how to use :py:meth:`~sympy.solvers.ode.dsolve`.
        - :py:meth:`~sympy.solvers.ode.dsolve` always returns an
          :py:class:`~sympy.core.relational.Equality` class (except for the
          case when the hint is ``all`` or ``all_Integral``).  If possible, it
          solves the solution explicitly for the function being solved for.
          Otherwise, it returns an implicit solution.
        - Arbitrary constants are symbols named ``C1``, ``C2``, and so on.
        - Because all solutions should be mathematically equivalent, some
          hints may return the exact same result for an ODE. Often, though,
          two different hints will return the same solution formatted
          differently.  The two should be equivalent. Also note that sometimes
          the values of the arbitrary constants in two different solutions may
          not be the same, because one constant may have "absorbed" other
          constants into it.
        - Do ``help(ode.ode_<hintname>)`` to get help more information on a
          specific hint, where ``<hintname>`` is the name of a hint without
          ``_Integral``.

    For system of ordinary differential equations
    =============================================

    **Usage**
        ``dsolve(eq, func)`` -> Solve a system of ordinary differential
        equations ``eq`` for ``func`` being list of functions including
        `x(t)`, `y(t)`, `z(t)` where number of functions in the list depends
        upon the number of equations provided in ``eq``.

    **Details**

        ``eq`` can be any supported system of ordinary differential equations
        This can either be an :py:class:`~sympy.core.relational.Equality`,
        or an expression, which is assumed to be equal to ``0``.

        ``func`` holds ``x(t)`` and ``y(t)`` being functions of one variable which
        together with some of their derivatives make up the system of ordinary
        differential equation ``eq``. It is not necessary to provide this; it
        will be autodetected (and an error raised if it could not be detected).

    **Hints**

        The hints are formed by parameters returned by classify_sysode, combining
        them give hints name used later for forming method name.

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))
    Eq(f(x), C1*sin(3*x) + C2*cos(3*x))

    >>> eq = sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x)
    >>> dsolve(eq, hint='1st_exact')
    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
    >>> dsolve(eq, hint='almost_linear')
    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
    >>> t = symbols('t')
    >>> x, y = symbols('x, y', cls=Function)
    >>> eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))
