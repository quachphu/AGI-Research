            for m in Add.make_args(d):  # d might have become non-Add
                p2 = []
                other = []
                for i in Mul.make_args(m):
                    if ispow2(i, log2=True):
                        p2.append(i.base if i.exp is S.Half else i.base**(2*i.exp))
                    elif i is S.ImaginaryUnit:
                        p2.append(S.NegativeOne)
                    else:
                        other.append(i)
                collected[tuple(ordered(p2))].append(Mul(*other))
            rterms = list(ordered(list(collected.items())))
            rterms = [(Mul(*i), Add(*j)) for i, j in rterms]
            nrad = len(rterms) - (1 if rterms[0][0] is S.One else 0)
            if nrad < 1:
                break
            elif nrad > max_terms:
                # there may have been invalid operations leading to this point
                # so don't keep changes, e.g. this expression is troublesome
                # in collecting terms so as not to raise the issue of 2834:
                # r = sqrt(sqrt(5) + 5)
                # eq = 1/(sqrt(5)*r + 2*sqrt(5)*sqrt(-sqrt(5) + 5) + 5*r)
                keep = False
                break
            if len(rterms) > 4:
                # in general, only 4 terms can be removed with repeated squaring
                # but other considerations can guide selection of radical terms
                # so that radicals are removed
                if all(x.is_Integer and (y**2).is_Rational for x, y in rterms):
                    nd, d = rad_rationalize(S.One, Add._from_args(
                        [sqrt(x)*y for x, y in rterms]))
                    n *= nd
                else:
                    # is there anything else that might be attempted?
                    keep = False
                break
            from sympy.simplify.powsimp import powsimp, powdenest

            num = powsimp(_num(rterms))
            n *= num
            d *= num
            d = powdenest(_mexpand(d), force=symbolic)
            if d.has(S.Zero, nan, zoo):
                return expr
            if d.is_Atom:
                break

        if not keep:
            return expr
        return _unevaluated_Mul(n, 1/d)

    if not isinstance(expr, Expr):
        return expr.func(*[radsimp(a, symbolic=symbolic, max_terms=max_terms) for a in expr.args])

    coeff, expr = expr.as_coeff_Add()
    expr = expr.normal()
    old = fraction(expr)
    n, d = fraction(handle(expr))
    if old != (n, d):
        if not d.is_Atom:
            was = (n, d)
            n = signsimp(n, evaluate=False)
            d = signsimp(d, evaluate=False)
            u = Factors(_unevaluated_Mul(n, 1/d))
            u = _unevaluated_Mul(*[k**v for k, v in u.factors.items()])
            n, d = fraction(u)
            if old == (n, d):
                n, d = was
        n = expand_mul(n)
        if d.is_Number or d.is_Add:
            n2, d2 = fraction(gcd_terms(_unevaluated_Mul(n, 1/d)))
            if d2.is_Number or (d2.count_ops() <= d.count_ops()):
                n, d = [signsimp(i) for i in (n2, d2)]
                if n.is_Mul and n.args[0].is_Number:
                    n = n.func(*n.args)

    return coeff + _unevaluated_Mul(n, 1/d)


def rad_rationalize(num, den):
    """
    Rationalize ``num/den`` by removing square roots in the denominator;
    num and den are sum of terms whose squares are positive rationals.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import rad_rationalize
    >>> rad_rationalize(sqrt(3), 1 + sqrt(2)/3)
    (-sqrt(3) + sqrt(6)/3, -7/9)
    """
    if not den.is_Add:
        return num, den
    g, a, b = split_surds(den)
    a = a*sqrt(g)
    num = _mexpand((a - b)*num)
    den = _mexpand(a**2 - b**2)
    return rad_rationalize(num, den)


def fraction(expr, exact=False):
    """Returns a pair with expression's numerator and denominator.
       If the given expression is not a fraction then this function
       will return the tuple (expr, 1).

       This function will not make any attempt to simplify nested
       fractions or to do any term rewriting at all.

       If only one of the numerator/denominator pair is needed then
       use numer(expr) or denom(expr) functions respectively.

       >>> from sympy import fraction, Rational, Symbol
       >>> from sympy.abc import x, y

       >>> fraction(x/y)
       (x, y)
       >>> fraction(x)
       (x, 1)

       >>> fraction(1/y**2)
       (1, y**2)

       >>> fraction(x*y/2)
       (x*y, 2)
       >>> fraction(Rational(1, 2))
       (1, 2)

       This function will also work fine with assumptions:

       >>> k = Symbol('k', negative=True)
       >>> fraction(x * y**k)
       (x, y**(-k))

       If we know nothing about sign of some exponent and ``exact``
       flag is unset, then the exponent's structure will
       be analyzed and pretty fraction will be returned:

       >>> from sympy import exp, Mul
       >>> fraction(2*x**(-y))
       (2, x**y)

       >>> fraction(exp(-x))
       (1, exp(x))

       >>> fraction(exp(-x), exact=True)
       (exp(-x), 1)

       The ``exact`` flag will also keep any unevaluated Muls from
       being evaluated:

       >>> u = Mul(2, x + 1, evaluate=False)
       >>> fraction(u)
       (2*x + 2, 1)
       >>> fraction(u, exact=True)
       (2*(x  + 1), 1)
    """
    expr = sympify(expr)

    numer, denom = [], []

    for term in Mul.make_args(expr):
        if term.is_commutative and (term.is_Pow or isinstance(term, exp)):
            b, ex = term.as_base_exp()
            if ex.is_negative:
                if ex is S.NegativeOne:
                    denom.append(b)
                elif exact:
                    if ex.is_constant():
                        denom.append(Pow(b, -ex))
                    else:
                        numer.append(term)
                else:
                    denom.append(Pow(b, -ex))
            elif ex.is_positive:
                numer.append(term)
            elif not exact and ex.is_Mul:
                n, d = term.as_numer_denom()  # this will cause evaluation
                if n != 1:
                    numer.append(n)
                denom.append(d)
            else:
                numer.append(term)
        elif term.is_Rational and not term.is_Integer:
            if term.p != 1:
                numer.append(term.p)
            denom.append(term.q)
        else:
            numer.append(term)
    return Mul(*numer, evaluate=not exact), Mul(*denom, evaluate=not exact)


def numer(expr, exact=False):  # default matches fraction's default
    return fraction(expr, exact=exact)[0]


def denom(expr, exact=False):  # default matches fraction's default
    return fraction(expr, exact=exact)[1]


def fraction_expand(expr, **hints):
    return expr.expand(frac=True, **hints)


def numer_expand(expr, **hints):
    # default matches fraction's default
    a, b = fraction(expr, exact=hints.get('exact', False))
    return a.expand(numer=True, **hints) / b


def denom_expand(expr, **hints):
    # default matches fraction's default
    a, b = fraction(expr, exact=hints.get('exact', False))
    return a / b.expand(denom=True, **hints)


expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand


def split_surds(expr):
    """
    Split an expression with terms whose squares are positive rationals
    into a sum of terms whose surds squared have gcd equal to g
    and a sum of terms with surds squared prime with g.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import split_surds
    >>> split_surds(3*sqrt(3) + sqrt(5)/7 + sqrt(6) + sqrt(10) + sqrt(15))
    (3, sqrt(2) + sqrt(5) + 3, sqrt(5)/7 + sqrt(10))
    """
    args = sorted(expr.args, key=default_sort_key)
    coeff_muls = [x.as_coeff_Mul() for x in args]
    surds = [x[1]**2 for x in coeff_muls if x[1].is_Pow]
    surds.sort(key=default_sort_key)
    g, b1, b2 = _split_gcd(*surds)
    g2 = g
    if not b2 and len(b1) >= 2:
        b1n = [x/g for x in b1]
        b1n = [x for x in b1n if x != 1]
        # only a common factor has been factored; split again
        g1, b1n, b2 = _split_gcd(*b1n)
        g2 = g*g1
    a1v, a2v = [], []
    for c, s in coeff_muls:
        if s.is_Pow and s.exp == S.Half:
            s1 = s.base
            if s1 in b1:
                a1v.append(c*sqrt(s1/g2))
            else:
                a2v.append(c*s)
        else:
            a2v.append(c*s)
    a = Add(*a1v)
    b = Add(*a2v)
    return g2, a, b


def _split_gcd(*a):
    """
    Split the list of integers ``a`` into a list of integers, ``a1`` having
    ``g = gcd(a1)``, and a list ``a2`` whose elements are not divisible by
    ``g``.  Returns ``g, a1, a2``.

    Examples
    ========

    >>> from sympy.simplify.radsimp import _split_gcd
    >>> _split_gcd(55, 35, 22, 14, 77, 10)
    (5, [55, 35, 10], [22, 14, 77])
    """
    g = a[0]
    b1 = [g]
    b2 = []
    for x in a[1:]:
        g1 = gcd(g, x)
        if g1 == 1:
            b2.append(x)
        else:
            g = g1
            b1.append(x)
    return g, b1, b2
<|endoftext|>from sympy.core import Mul
from sympy.core.function import count_ops
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions import gamma
from sympy.simplify.gammasimp import gammasimp, _gammasimp

from sympy.utilities.timeutils import timethis


@timethis('combsimp')
def combsimp(expr):
    r"""
    Simplify combinatorial expressions.

    Explanation
    ===========

    This function takes as input an expression containing factorials,
    binomials, Pochhammer symbol and other "combinatorial" functions,
    and tries to minimize the number of those functions and reduce
    the size of their arguments.

    The algorithm works by rewriting all combinatorial functions as
    gamma functions and applying gammasimp() except simplification
    steps that may make an integer argument non-integer. See docstring
    of gammasimp for more information.

    Then it rewrites expression in terms of factorials and binomials by
    rewriting gammas as factorials and converting (a+b)!/a!b! into
    binomials.

    If expression has gamma functions or combinatorial functions
    with non-integer argument, it is automatically passed to gammasimp.

    Examples
    ========

    >>> from sympy.simplify import combsimp
    >>> from sympy import factorial, binomial, symbols
    >>> n, k = symbols('n k', integer = True)

    >>> combsimp(factorial(n)/factorial(n - 3))
    n*(n - 2)*(n - 1)
    >>> combsimp(binomial(n+1, k+1)/binomial(n, k))
    (n + 1)/(k + 1)

    """

    expr = expr.rewrite(gamma, piecewise=False)
    if any(isinstance(node, gamma) and not node.args[0].is_integer
        for node in preorder_traversal(expr)):
        return gammasimp(expr)

    expr = _gammasimp(expr, as_comb = True)
    expr = _gamma_as_comb(expr)
    return expr


def _gamma_as_comb(expr):
    """
    Helper function for combsimp.

    Rewrites expression in terms of factorials and binomials
    """

    expr = expr.rewrite(factorial)

    def f(rv):
        if not rv.is_Mul:
            return rv
        rvd = rv.as_powers_dict()
        nd_fact_args = [[], []] # numerator, denominator

        for k in rvd:
            if isinstance(k, factorial) and rvd[k].is_Integer:
                if rvd[k].is_positive:
                    nd_fact_args[0].extend([k.args[0]]*rvd[k])
                else:
                    nd_fact_args[1].extend([k.args[0]]*-rvd[k])
                rvd[k] = 0
        if not nd_fact_args[0] or not nd_fact_args[1]:
            return rv

        hit = False
        for m in range(2):
            i = 0
            while i < len(nd_fact_args[m]):
                ai = nd_fact_args[m][i]
                for j in range(i + 1, len(nd_fact_args[m])):
                    aj = nd_fact_args[m][j]

                    sum = ai + aj
                    if sum in nd_fact_args[1 - m]:
                        hit = True

                        nd_fact_args[1 - m].remove(sum)
                        del nd_fact_args[m][j]
                        del nd_fact_args[m][i]

                        rvd[binomial(sum, ai if count_ops(ai) <
                                count_ops(aj) else aj)] += (
                                -1 if m == 0 else 1)
                        break
                else:
                    i += 1

        if hit:
            return Mul(*([k**rvd[k] for k in rvd] + [factorial(k)
                    for k in nd_fact_args[0]]))/Mul(*[factorial(k)
                    for k in nd_fact_args[1]])
        return rv

    return bottom_up(expr, f)
<|endoftext|>"""Module for differentiation using CSE."""

from sympy import cse, Matrix, Derivative, MatrixBase
from sympy.utilities.iterables import iterable


def _remove_cse_from_derivative(replacements, reduced_expressions):
    """
    This function is designed to postprocess the output of a common subexpression
    elimination (CSE) operation. Specifically, it removes any CSE replacement
    symbols from the arguments of ``Derivative`` terms in the expression. This
    is necessary to ensure that the forward Jacobian function correctly handles
    derivative terms.

    Parameters
    ==========

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation.

    reduced_expressions : list of SymPy expressions
        The reduced expressions with all the replacements from the
        replacements list above.

    Returns
    =======

    processed_replacements : list of (Symbol, expression) pairs
        Processed replacement list, in the same format of the
        ``replacements`` input list.

    processed_reduced : list of SymPy expressions
        Processed reduced list, in the same format of the
        ``reduced_expressions`` input list.
    """

    def traverse(node, repl_dict):
        if isinstance(node, Derivative):
            return replace_all(node, repl_dict)
        if not node.args:
            return node
        new_args = [traverse(arg, repl_dict) for arg in node.args]
        return node.func(*new_args)

    def replace_all(node, repl_dict):
        result = node
        while True:
            free_symbols = result.free_symbols
            symbols_dict = {k: repl_dict[k] for k in free_symbols if k in repl_dict}
            if not symbols_dict:
                break
            result = result.xreplace(symbols_dict)
        return result

    repl_dict = dict(replacements)
    processed_replacements = [
        (rep_sym, traverse(sub_exp, repl_dict))
        for rep_sym, sub_exp in replacements
    ]
    processed_reduced = [
        red_exp.__class__([traverse(exp, repl_dict) for exp in red_exp])
        for red_exp in reduced_expressions
    ]

    return processed_replacements, processed_reduced


def _forward_jacobian_cse(replacements, reduced_expr, wrt):
    """
    Core function to compute the Jacobian of an input Matrix of expressions
    through forward accumulation. Takes directly the output of a CSE operation
    (replacements and reduced_expr), and an iterable of variables (wrt) with
    respect to which to differentiate the reduced expression and returns the
    reduced Jacobian matrix and the ``replacements`` list.

    The function also returns a list of precomputed free symbols for each
    subexpression, which are useful in the substitution process.

    Parameters
    ==========

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation.

    reduced_expr : list of SymPy expressions
        The reduced expressions with all the replacements from the
        replacements list above.

    wrt : iterable
        Iterable of expressions with respect to which to compute the
        Jacobian matrix.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation. Compared to the input replacement list,
        the output one doesn't contain replacement symbols inside
        ``Derivative``'s arguments.

    jacobian : list of SymPy expressions
        The list only contains one element, which is the Jacobian matrix with
        elements in reduced form (replacement symbols are present).

    precomputed_fs: list
        List of sets, which store the free symbols present in each sub-expression.
        Useful in the substitution process.
    """

    if not isinstance(reduced_expr[0], MatrixBase):
        raise TypeError("``expr`` must be of matrix type")

    if not (reduced_expr[0].shape[0] == 1 or reduced_expr[0].shape[1] == 1):
        raise TypeError("``expr`` must be a row or a column matrix")

    if not iterable(wrt):
        raise TypeError("``wrt`` must be an iterable of variables")

    elif not isinstance(wrt, MatrixBase):
        wrt = Matrix(wrt)

    if not (wrt.shape[0] == 1 or wrt.shape[1] == 1):
        raise TypeError("``wrt`` must be a row or a column matrix")

    replacements, reduced_expr = _remove_cse_from_derivative(replacements, reduced_expr)

    if replacements:
        rep_sym, sub_expr = map(Matrix, zip(*replacements))
    else:
        rep_sym, sub_expr = Matrix([]), Matrix([])

    l_sub, l_wrt, l_red = len(sub_expr), len(wrt), len(reduced_expr[0])

    f1 = reduced_expr[0].__class__.from_dok(l_red, l_wrt,
        {
            (i, j): diff_value
            for i, r in enumerate(reduced_expr[0])
            for j, w in enumerate(wrt)
            if (diff_value := r.diff(w)) != 0
        },
    )

    if not replacements:
        return [], [f1], []

    f2 = Matrix.from_dok(l_red, l_sub,
        {
            (i, j): diff_value
            for i, (r, fs) in enumerate([(r, r.free_symbols) for r in reduced_expr[0]])
            for j, s in enumerate(rep_sym)
            if s in fs and (diff_value := r.diff(s)) != 0
        },
    )

    rep_sym_set = set(rep_sym)
    precomputed_fs = [s.free_symbols & rep_sym_set for s in sub_expr ]

    c_matrix = Matrix.from_dok(1, l_wrt,
                               {(0, j): diff_value for j, w in enumerate(wrt)
                                if (diff_value := sub_expr[0].diff(w)) != 0})

    for i in range(1, l_sub):

        bi_matrix = Matrix.from_dok(1, i,
                                    {(0, j): diff_value for j in range(i + 1)
                                     if rep_sym[j] in precomputed_fs[i]
                                     and (diff_value := sub_expr[i].diff(rep_sym[j])) != 0})

        ai_matrix = Matrix.from_dok(1, l_wrt,
                                    {(0, j): diff_value for j, w in enumerate(wrt)
                                     if (diff_value := sub_expr[i].diff(w)) != 0})

        if bi_matrix._rep.nnz():
            ci_matrix = bi_matrix.multiply(c_matrix).add(ai_matrix)
            c_matrix = Matrix.vstack(c_matrix, ci_matrix)
        else:
            c_matrix = Matrix.vstack(c_matrix, ai_matrix)

    jacobian = f2.multiply(c_matrix).add(f1)
    jacobian = [reduced_expr[0].__class__(jacobian)]

    return replacements, jacobian, precomputed_fs


def _forward_jacobian_norm_in_cse_out(expr, wrt):
    """
    Function to compute the Jacobian of an input Matrix of expressions through
    forward accumulation. Takes a sympy Matrix of expressions (expr) as input
    and an iterable of variables (wrt) with respect to which to compute the
    Jacobian matrix. The matrix is returned in reduced form (containing
    replacement symbols) along with the ``replacements`` list.

    The function also returns a list of precomputed free symbols for each
    subexpression, which are useful in the substitution process.

    Parameters
    ==========

    expr : Matrix
        The vector to be differentiated.

    wrt : iterable
        The vector with respect to which to perform the differentiation.
        Can be a matrix or an iterable of variables.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation. The output replacement list doesn't
        contain replacement symbols inside ``Derivative``'s arguments.

    jacobian : list of SymPy expressions
        The list only contains one element, which is the Jacobian matrix with
        elements in reduced form (replacement symbols are present).

    precomputed_fs: list
        List of sets, which store the free symbols present in each
        sub-expression. Useful in the substitution process.
    """

    replacements, reduced_expr = cse(expr)
    replacements, jacobian, precomputed_fs = _forward_jacobian_cse(replacements, reduced_expr, wrt)

    return replacements, jacobian, precomputed_fs


def _forward_jacobian(expr, wrt):
    """
    Function to compute the Jacobian of an input Matrix of expressions through
    forward accumulation. Takes a sympy Matrix of expressions (expr) as input
    and an iterable of variables (wrt) with respect to which to compute the
    Jacobian matrix.

    Explanation
    ===========

    Expressions often contain repeated subexpressions. Using a tree structure,
    these subexpressions are duplicated and differentiated multiple times,
    leading to inefficiency.

    Instead, if a data structure called a directed acyclic graph (DAG) is used
    then each of these repeated subexpressions will only exist a single time.
    This function uses a combination of representing the expression as a DAG and
    a forward accumulation algorithm (repeated application of the chain rule
    symbolically) to more efficiently calculate the Jacobian matrix of a target
    expression ``expr`` with respect to an expression or set of expressions
    ``wrt``.

    Note that this function is intended to improve performance when
    differentiating large expressions that contain many common subexpressions.
    For small and simple expressions it is likely less performant than using
    SymPy's standard differentiation functions and methods.

    Parameters
    ==========

    expr : Matrix
        The vector to be differentiated.

    wrt : iterable
        The vector with respect to which to do the differentiation.
        Can be a matrix or an iterable of variables.

    See Also
    ========

    Direct Acyclic Graph : https://en.wikipedia.org/wiki/Directed_acyclic_graph
    """

    replacements, reduced_expr = cse(expr)

    if replacements:
        rep_sym, _ = map(Matrix, zip(*replacements))
    else:
        rep_sym = Matrix([])

    replacements, jacobian, precomputed_fs = _forward_jacobian_cse(replacements, reduced_expr, wrt)

    if not replacements: return jacobian[0]

    sub_rep = dict(replacements)
    for i, ik in enumerate(precomputed_fs):
        sub_dict = {j: sub_rep[j] for j in ik}
        sub_rep[rep_sym[i]] = sub_rep[rep_sym[i]].xreplace(sub_dict)

    return jacobian[0].xreplace(sub_rep)
<|endoftext|>"""
Expand Hypergeometric (and Meijer G) functions into named
special functions.

The algorithm for doing this uses a collection of lookup tables of
hypergeometric functions, and various of their properties, to expand
many hypergeometric functions in terms of special functions.

It is based on the following paper:
      Kelly B. Roach.  Meijer G Function Representations.
      In: Proceedings of the 1997 International Symposium on Symbolic and
      Algebraic Computation, pages 205-211, New York, 1997. ACM.

It is described in great(er) detail in the Sphinx documentation.
"""
# SUMMARY OF EXTENSIONS FOR MEIJER G FUNCTIONS
#
# o z**rho G(ap, bq; z) = G(ap + rho, bq + rho; z)
#
# o denote z*d/dz by D
#
# o It is helpful to keep in mind that ap and bq play essentially symmetric
#   roles: G(1/z) has slightly altered parameters, with ap and bq interchanged.
#
# o There are four shift operators:
#   A_J = b_J - D,     J = 1, ..., n
#   B_J = 1 - a_j + D, J = 1, ..., m
#   C_J = -b_J + D,    J = m+1, ..., q
#   D_J = a_J - 1 - D, J = n+1, ..., p
#
#   A_J, C_J increment b_J
#   B_J, D_J decrement a_J
#
# o The corresponding four inverse-shift operators are defined if there
#   is no cancellation. Thus e.g. an index a_J (upper or lower) can be
#   incremented if a_J != b_i for i = 1, ..., q.
#
# o Order reduction: if b_j - a_i is a non-negative integer, where
#   j <= m and i > n, the corresponding quotient of gamma functions reduces
#   to a polynomial. Hence the G function can be expressed using a G-function
#   of lower order.
#   Similarly if j > m and i <= n.
#
#   Secondly, there are paired index theorems [Adamchik, The evaluation of
#   integrals of Bessel functions via G-function identities]. Suppose there
#   are three parameters a, b, c, where a is an a_i, i <= n, b is a b_j,
#   j <= m and c is a denominator parameter (i.e. a_i, i > n or b_j, j > m).
#   Suppose further all three differ by integers.
#   Then the order can be reduced.
#   TODO work this out in detail.
#
# o An index quadruple is called suitable if its order cannot be reduced.
#   If there exists a sequence of shift operators transforming one index
#   quadruple into another, we say one is reachable from the other.
#
# o Deciding if one index quadruple is reachable from another is tricky. For
#   this reason, we use hand-built routines to match and instantiate formulas.
#
from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod

from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
    EulerGamma, oo, zoo, expand_func, Add, nan, Expr, Rational)
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
        besseli, gamma, uppergamma, expint, erf, sin, besselj, Ei, Ci, Si, Shi,
        sinh, cosh, Chi, fresnels, fresnelc, polar_lift, exp_polar, floor, ceiling,
        rf, factorial, lerchphi, Piecewise, re, elliptic_k, elliptic_e)
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
        HyperRep_power1, HyperRep_power2, HyperRep_log1, HyperRep_asin1,
        HyperRep_asin2, HyperRep_sqrts1, HyperRep_sqrts2, HyperRep_log2,
        HyperRep_cosasin, HyperRep_sinasin, meijerg)
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift

# function to define "buckets"
def _mod1(x):
    # TODO see if this can work as Mod(x, 1); this will require
    # different handling of the "buckets" since these need to
    # be sorted and that fails when there is a mixture of
    # integers and expressions with parameters. With the current
    # Mod behavior, Mod(k, 1) == Mod(1, 1) == 0 if k is an integer.
    # Although the sorting can be done with Basic.compare, this may
    # still require different handling of the sorted buckets.
    if x.is_Number:
        return Mod(x, 1)
    c, x = x.as_coeff_Add()
    return Mod(c, 1) + x


# leave add formulae at the top for easy reference
def add_formulae(formulae):
    """ Create our knowledge base. """
    a, b, c, z = symbols('a b c, z', cls=Dummy)

    def add(ap, bq, res):
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, res, (a, b, c)))

    def addb(ap, bq, B, C, M):
        func = Hyper_Function(ap, bq)
        formulae.append(Formula(func, z, None, (a, b, c), B, C, M))

    # Luke, Y. L. (1969), The Special Functions and Their Approximations,
    # Volume 1, section 6.2

    # 0F0
    add((), (), exp(z))

    # 1F0
    add((a, ), (), HyperRep_power1(-a, z))

    # 2F1
    addb((a, a - S.Half), (2*a, ),
         Matrix([HyperRep_power2(a, z),
                 HyperRep_power2(a + S.Half, z)/2]),
         Matrix([[1, 0]]),
         Matrix([[(a - S.Half)*z/(1 - z), (S.Half - a)*z/(1 - z)],
                 [a/(1 - z), a*(z - 2)/(1 - z)]]))
    addb((1, 1), (2, ),
         Matrix([HyperRep_log1(z), 1]), Matrix([[-1/z, 0]]),
         Matrix([[0, z/(z - 1)], [0, 0]]))
    addb((S.Half, 1), (S('3/2'), ),
         Matrix([HyperRep_atanh(z), 1]),
         Matrix([[1, 0]]),
         Matrix([[Rational(-1, 2), 1/(1 - z)/2], [0, 0]]))
    addb((S.Half, S.Half), (S('3/2'), ),
         Matrix([HyperRep_asin1(z), HyperRep_power1(Rational(-1, 2), z)]),
         Matrix([[1, 0]]),
         Matrix([[Rational(-1, 2), S.Half], [0, z/(1 - z)/2]]))
    addb((a, S.Half + a), (S.Half, ),
         Matrix([HyperRep_sqrts1(-a, z), -HyperRep_sqrts2(-a - S.Half, z)]),
         Matrix([[1, 0]]),
         Matrix([[0, -a],
                 [z*(-2*a - 1)/2/(1 - z), S.Half - z*(-2*a - 1)/(1 - z)]]))

    # A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    # Integrals and Series: More Special Functions, Vol. 3,.
    # Gordon and Breach Science Publisher
    addb([a, -a], [S.Half],
         Matrix([HyperRep_cosasin(a, z), HyperRep_sinasin(a, z)]),
         Matrix([[1, 0]]),
         Matrix([[0, -a], [a*z/(1 - z), 1/(1 - z)/2]]))
    addb([1, 1], [3*S.Half],
         Matrix([HyperRep_asin2(z), 1]), Matrix([[1, 0]]),
         Matrix([[(z - S.Half)/(1 - z), 1/(1 - z)/2], [0, 0]]))

    # Complete elliptic integrals K(z) and E(z), both a 2F1 function
    addb([S.Half, S.Half], [S.One],
         Matrix([elliptic_k(z), elliptic_e(z)]),
         Matrix([[2/pi, 0]]),
         Matrix([[Rational(-1, 2), -1/(2*z-2)],
                 [Rational(-1, 2), S.Half]]))
    addb([Rational(-1, 2), S.Half], [S.One],
         Matrix([elliptic_k(z), elliptic_e(z)]),
         Matrix([[0, 2/pi]]),
         Matrix([[Rational(-1, 2), -1/(2*z-2)],
                 [Rational(-1, 2), S.Half]]))

    # 3F2
    addb([Rational(-1, 2), 1, 1], [S.Half, 2],
         Matrix([z*HyperRep_atanh(z), HyperRep_log1(z), 1]),
         Matrix([[Rational(-2, 3), -S.One/(3*z), Rational(2, 3)]]),
         Matrix([[S.Half, 0, z/(1 - z)/2],
                 [0, 0, z/(z - 1)],
                 [0, 0, 0]]))
    # actually the formula for 3/2 is much nicer ...
    addb([Rational(-1, 2), 1, 1], [2, 2],
         Matrix([HyperRep_power1(S.Half, z), HyperRep_log2(z), 1]),
         Matrix([[Rational(4, 9) - 16/(9*z), 4/(3*z), 16/(9*z)]]),
         Matrix([[z/2/(z - 1), 0, 0], [1/(2*(z - 1)), 0, S.Half], [0, 0, 0]]))

    # 1F1
    addb([1], [b], Matrix([z**(1 - b) * exp(z) * lowergamma(b - 1, z), 1]),
         Matrix([[b - 1, 0]]), Matrix([[1 - b + z, 1], [0, 0]]))
    addb([a], [2*a],
         Matrix([z**(S.Half - a)*exp(z/2)*besseli(a - S.Half, z/2)
                 * gamma(a + S.Half)/4**(S.Half - a),
                 z**(S.Half - a)*exp(z/2)*besseli(a + S.Half, z/2)
                 * gamma(a + S.Half)/4**(S.Half - a)]),
         Matrix([[1, 0]]),
         Matrix([[z/2, z/2], [z/2, (z/2 - 2*a)]]))
    mz = polar_lift(-1)*z
    addb([a], [a + 1],
         Matrix([mz**(-a)*a*lowergamma(a, mz), a*exp(z)]),
         Matrix([[1, 0]]),
         Matrix([[-a, 1], [0, z]]))
    # This one is redundant.
    add([Rational(-1, 2)], [S.Half], exp(z) - sqrt(pi*z)*(-I)*erf(I*sqrt(z)))

    # Added to get nice results for Laplace transform of Fresnel functions
    # https://functions.wolfram.com/07.22.03.6437.01
    # Basic rule
    #add([1], [Rational(3, 4), Rational(5, 4)],
    #    sqrt(pi) * (cos(2*sqrt(polar_lift(-1)*z))*fresnelc(2*root(polar_lift(-1)*z,4)/sqrt(pi)) +
    #                sin(2*sqrt(polar_lift(-1)*z))*fresnels(2*root(polar_lift(-1)*z,4)/sqrt(pi)))
    #    / (2*root(polar_lift(-1)*z,4)))
    # Manually tuned rule
    addb([1], [Rational(3, 4), Rational(5, 4)],
         Matrix([ sqrt(pi)*(I*sinh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                            + cosh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
                  * exp(-I*pi/4)/(2*root(z, 4)),
                  sqrt(pi)*root(z, 4)*(sinh(2*sqrt(z))*fresnelc(2*root(z, 4)*exp(I*pi/4)/sqrt(pi))
                                      + I*cosh(2*sqrt(z))*fresnels(2*root(z, 4)*exp(I*pi/4)/sqrt(pi)))
                  *exp(-I*pi/4)/2,
                  1 ]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 4),              1, Rational(1, 4)],
                 [              z, Rational(1, 4),              0],
                 [              0,              0,              0]]))

    # 2F2
    addb([S.Half, a], [Rational(3, 2), a + 1],
         Matrix([a/(2*a - 1)*(-I)*sqrt(pi/z)*erf(I*sqrt(z)),
                 a/(2*a - 1)*(polar_lift(-1)*z)**(-a)*
                 lowergamma(a, polar_lift(-1)*z),
                 a/(2*a - 1)*exp(z)]),
         Matrix([[1, -1, 0]]),
         Matrix([[Rational(-1, 2), 0, 1], [0, -a, 1], [0, 0, z]]))
    # We make a "basis" of four functions instead of three, and give EulerGamma
    # an extra slot (it could just be a coefficient to 1). The advantage is
    # that this way Polys will not see multivariate polynomials (it treats
    # EulerGamma as an indeterminate), which is *way* faster.
    addb([1, 1], [2, 2],
         Matrix([Ei(z) - log(z), exp(z), 1, EulerGamma]),
         Matrix([[1/z, 0, 0, -1/z]]),
         Matrix([[0, 1, -1, 0], [0, z, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))

    # 0F1
    add((), (S.Half, ), cosh(2*sqrt(z)))
    addb([], [b],
         Matrix([gamma(b)*z**((1 - b)/2)*besseli(b - 1, 2*sqrt(z)),
                 gamma(b)*z**(1 - b/2)*besseli(b, 2*sqrt(z))]),
         Matrix([[1, 0]]), Matrix([[0, 1], [z, (1 - b)]]))

    # 0F3
    x = 4*z**Rational(1, 4)

    def fp(a, z):
        return besseli(a, x) + besselj(a, x)

    def fm(a, z):
        return besseli(a, x) - besselj(a, x)

    # TODO branching
    addb([], [S.Half, a, a + S.Half],
         Matrix([fp(2*a - 1, z), fm(2*a, z)*z**Rational(1, 4),
                 fm(2*a - 1, z)*sqrt(z), fp(2*a, z)*z**Rational(3, 4)])
         * 2**(-2*a)*gamma(2*a)*z**((1 - 2*a)/4),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, 1, 0, 0],
                 [0, S.Half - a, 1, 0],
                 [0, 0, S.Half, 1],
                 [z, 0, 0, 1 - a]]))
    x = 2*(4*z)**Rational(1, 4)*exp_polar(I*pi/4)
    addb([], [a, a + S.Half, 2*a],
         (2*sqrt(polar_lift(-1)*z))**(1 - 2*a)*gamma(2*a)**2 *
         Matrix([besselj(2*a - 1, x)*besseli(2*a - 1, x),
                 x*(besseli(2*a, x)*besselj(2*a - 1, x)
                    - besseli(2*a - 1, x)*besselj(2*a, x)),
                 x**2*besseli(2*a, x)*besselj(2*a, x),
                 x**3*(besseli(2*a, x)*besselj(2*a - 1, x)
                       + besseli(2*a - 1, x)*besselj(2*a, x))]),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, Rational(1, 4), 0, 0],
                 [0, (1 - 2*a)/2, Rational(-1, 2), 0],
                 [0, 0, 1 - 2*a, Rational(1, 4)],
                 [-32*z, 0, 0, 1 - a]]))

    # 1F2
    addb([a], [a - S.Half, 2*a],
         Matrix([z**(S.Half - a)*besseli(a - S.Half, sqrt(z))**2,
                 z**(1 - a)*besseli(a - S.Half, sqrt(z))
                 *besseli(a - Rational(3, 2), sqrt(z)),
                 z**(Rational(3, 2) - a)*besseli(a - Rational(3, 2), sqrt(z))**2]),
         Matrix([[-gamma(a + S.Half)**2/4**(S.Half - a),
                 2*gamma(a - S.Half)*gamma(a + S.Half)/4**(1 - a),
                 0]]),
         Matrix([[1 - 2*a, 1, 0], [z/2, S.Half - a, S.Half], [0, z, 0]]))
    addb([S.Half], [b, 2 - b],
         pi*(1 - b)/sin(pi*b)*
         Matrix([besseli(1 - b, sqrt(z))*besseli(b - 1, sqrt(z)),
                 sqrt(z)*(besseli(-b, sqrt(z))*besseli(b - 1, sqrt(z))
                          + besseli(1 - b, sqrt(z))*besseli(b, sqrt(z))),
                 besseli(-b, sqrt(z))*besseli(b, sqrt(z))]),
         Matrix([[1, 0, 0]]),
         Matrix([[b - 1, S.Half, 0],
                 [z, 0, z],
                 [0, S.Half, -b]]))
    addb([S.Half], [Rational(3, 2), Rational(3, 2)],
         Matrix([Shi(2*sqrt(z))/2/sqrt(z), sinh(2*sqrt(z))/2/sqrt(z),
                 cosh(2*sqrt(z))]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 2), S.Half, 0], [0, Rational(-1, 2), S.Half], [0, 2*z, 0]]))

    # FresnelS
    # Basic rule
    #add([Rational(3, 4)], [Rational(3, 2),Rational(7, 4)], 6*fresnels( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) / ( pi * (exp(pi*I/4)*root(z,4)*2/sqrt(pi))**3 ) )
    # Manually tuned rule
    addb([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)],
         Matrix(
             [ fresnels(
                 exp(
                     pi*I/4)*root(
                         z, 4)*2/sqrt(
                             pi) ) / (
                                 pi * (exp(pi*I/4)*root(z, 4)*2/sqrt(pi))**3 ),
               sinh(2*sqrt(z))/sqrt(z),
               cosh(2*sqrt(z)) ]),
         Matrix([[6, 0, 0]]),
         Matrix([[Rational(-3, 4),  Rational(1, 16), 0],
                 [ 0,      Rational(-1, 2),  1],
                 [ 0,       z,       0]]))

    # FresnelC
    # Basic rule
    #add([Rational(1, 4)], [S.Half,Rational(5, 4)], fresnelc( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) / ( exp(pi*I/4)*root(z,4)*2/sqrt(pi) ) )
    # Manually tuned rule
    addb([Rational(1, 4)], [S.Half, Rational(5, 4)],
         Matrix(
             [ sqrt(
                 pi)*exp(
                     -I*pi/4)*fresnelc(
                         2*root(z, 4)*exp(I*pi/4)/sqrt(pi))/(2*root(z, 4)),
               cosh(2*sqrt(z)),
               sinh(2*sqrt(z))*sqrt(z) ]),
         Matrix([[1, 0, 0]]),
         Matrix([[Rational(-1, 4),  Rational(1, 4), 0     ],
                 [ 0,       0,      1     ],
                 [ 0,       z,      S.Half]]))

    # 2F3
    # XXX with this five-parameter formula is pretty slow with the current
    #     Formula.find_instantiations (creates 2!*3!*3**(2+3) ~ 3000
    #     instantiations ... But it's not too bad.
    addb([a, a + S.Half], [2*a, b, 2*a - b + 1],
         gamma(b)*gamma(2*a - b + 1) * (sqrt(z)/2)**(1 - 2*a) *
         Matrix([besseli(b - 1, sqrt(z))*besseli(2*a - b, sqrt(z)),
                 sqrt(z)*besseli(b, sqrt(z))*besseli(2*a - b, sqrt(z)),
                 sqrt(z)*besseli(b - 1, sqrt(z))*besseli(2*a - b + 1, sqrt(z)),
                 besseli(b, sqrt(z))*besseli(2*a - b + 1, sqrt(z))]),
         Matrix([[1, 0, 0, 0]]),
         Matrix([[0, S.Half, S.Half, 0],
                 [z/2, 1 - b, 0, z/2],
                 [z/2, 0, b - 2*a, z/2],
                 [0, S.Half, S.Half, -2*a]]))
    # (C/f above comment about eulergamma in the basis).
    addb([1, 1], [2, 2, Rational(3, 2)],
         Matrix([Chi(2*sqrt(z)) - log(2*sqrt(z)),
                 cosh(2*sqrt(z)), sqrt(z)*sinh(2*sqrt(z)), 1, EulerGamma]),
         Matrix([[1/z, 0, 0, 0, -1/z]]),
         Matrix([[0, S.Half, 0, Rational(-1, 2), 0],
                 [0, 0, 1, 0, 0],
                 [0, z, S.Half, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]))

    # 3F3
    # This is rule: https://functions.wolfram.com/07.31.03.0134.01
    # Initial reason to add it was a nice solution for
    # integrate(erf(a*z)/z**2, z) and same for erfc and erfi.
    # Basic rule
    # add([1, 1, a], [2, 2, a+1], (a/(z*(a-1)**2)) *
    #     (1 - (-z)**(1-a) * (gamma(a) - uppergamma(a,-z))
    #      - (a-1) * (EulerGamma + uppergamma(0,-z) + log(-z))
    #      - exp(z)))
    # Manually tuned rule
    addb([1, 1, a], [2, 2, a+1],
         Matrix([a*(log(-z) + expint(1, -z) + EulerGamma)/(z*(a**2 - 2*a + 1)),
                 a*(-z)**(-a)*(gamma(a) - uppergamma(a, -z))/(a - 1)**2,
                 a*exp(z)/(a**2 - 2*a + 1),
                 a/(z*(a**2 - 2*a + 1))]),
         Matrix([[1-a, 1, -1/z, 1]]),
         Matrix([[-1,0,-1/z,1],
                 [0,-a,1,0],
                 [0,0,z,0],
                 [0,0,0,-1]]))


def add_meijerg_formulae(formulae):
    a, b, c, z = list(map(Dummy, 'abcz'))
    rho = Dummy('rho')

    def add(an, ap, bm, bq, B, C, M, matcher):
        formulae.append(MeijerFormula(an, ap, bm, bq, z, [a, b, c, rho],
                                      B, C, M, matcher))

    def detect_uppergamma(func):
        x = func.an[0]
        y, z = func.bm
        swapped = False
        if not _mod1((x - y).simplify()):
            swapped = True
            (y, z) = (z, y)
        if _mod1((x - z).simplify()) or x - z > 0:
            return None
        l = [y, x]
        if swapped:
            l = [x, y]
        return {rho: y, a: x - y}, G_Function([x], [], l, [])

    add([a + rho], [], [rho, a + rho], [],
        Matrix([gamma(1 - a)*z**rho*exp(z)*uppergamma(a, z),
                gamma(1 - a)*z**(a + rho)]),
        Matrix([[1, 0]]),
        Matrix([[rho + z, -1], [0, a + rho]]),
        detect_uppergamma)

    def detect_3113(func):
        """https://functions.wolfram.com/07.34.03.0984.01"""
        x = func.an[0]
        u, v, w = func.bm
        if _mod1((u - v).simplify()) == 0:
            if _mod1((v - w).simplify()) == 0:
                return
            sig = (S.Half, S.Half, S.Zero)
            x1, x2, y = u, v, w
        else:
            if _mod1((x - u).simplify()) == 0:
                sig = (S.Half, S.Zero, S.Half)
                x1, y, x2 = u, v, w
            else:
                sig = (S.Zero, S.Half, S.Half)
                y, x1, x2 = u, v, w

        if (_mod1((x - x1).simplify()) != 0 or
            _mod1((x - x2).simplify()) != 0 or
            _mod1((x - y).simplify()) != S.Half or
                x - x1 > 0 or x - x2 > 0):
            return

        return {a: x}, G_Function([x], [], [x - S.Half + t for t in sig], [])

    s = sin(2*sqrt(z))
    c_ = cos(2*sqrt(z))
    S_ = Si(2*sqrt(z)) - pi/2
    C = Ci(2*sqrt(z))
    add([a], [], [a, a, a - S.Half], [],
        Matrix([sqrt(pi)*z**(a - S.Half)*(c_*S_ - s*C),
                sqrt(pi)*z**a*(s*S_ + c_*C),
                sqrt(pi)*z**a]),
        Matrix([[-2, 0, 0]]),
        Matrix([[a - S.Half, -1, 0], [z, a, S.Half], [0, 0, a]]),
        detect_3113)


def make_simp(z):
    """ Create a function that simplifies rational functions in ``z``. """

    def simp(expr):
        """ Efficiently simplify the rational function ``expr``. """
        numer, denom = expr.as_numer_denom()
        numer = numer.expand()
        # denom = denom.expand()  # is this needed?
        c, numer, denom = poly(numer, z).cancel(poly(denom, z))
        return c * numer.as_expr() / denom.as_expr()

    return simp


def debug(*args):
    if SYMPY_DEBUG:
        for a in args:
            print(a, end="")
        print()


class Hyper_Function(Expr):
    """ A generalized hypergeometric function. """

    def __new__(cls, ap, bq):
        obj = super().__new__(cls)
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        return (self.ap, self.bq)

    @property
    def sizes(self):
        return (len(self.ap), len(self.bq))

    @property
    def gamma(self):
        """
        Number of upper parameters that are negative integers

        This is a transformation invariant.
        """
        return sum(bool(x.is_integer and x.is_negative) for x in self.ap)

    def _hashable_content(self):
        return super()._hashable_content() + (self.ap,
                self.bq)

    def __call__(self, arg):
        return hyper(self.ap, self.bq, arg)

    def build_invariants(self):
        """
        Compute the invariant vector.

        Explanation
        ===========

        The invariant vector is:
            (gamma, ((s1, n1), ..., (sk, nk)), ((t1, m1), ..., (tr, mr)))
        where gamma is the number of integer a < 0,
              s1 < ... < sk
              nl is the number of parameters a_i congruent to sl mod 1
              t1 < ... < tr
              ml is the number of parameters b_i congruent to tl mod 1

        If the index pair contains parameters, then this is not truly an
        invariant, since the parameters cannot be sorted uniquely mod1.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import Hyper_Function
        >>> from sympy import S
        >>> ap = (S.Half, S.One/3, S(-1)/2, -2)
        >>> bq = (1, 2)

        Here gamma = 1,
             k = 3, s1 = 0, s2 = 1/3, s3 = 1/2
                    n1 = 1, n2 = 1,   n2 = 2
             r = 1, t1 = 0
                    m1 = 2:

        >>> Hyper_Function(ap, bq).build_invariants()
        (1, ((0, 1), (1/3, 1), (1/2, 2)), ((0, 2),))
        """
        abuckets, bbuckets = sift(self.ap, _mod1), sift(self.bq, _mod1)

        def tr(bucket):
            bucket = list(bucket.items())
            if not any(isinstance(x[0], Mod) for x in bucket):
                bucket.sort(key=lambda x: default_sort_key(x[0]))
            bucket = tuple([(mod, len(values)) for mod, values in bucket if
                    values])
            return bucket

        return (self.gamma, tr(abuckets), tr(bbuckets))

    def difficulty(self, func):
        """ Estimate how many steps it takes to reach ``func`` from self.
            Return -1 if impossible. """
        if self.gamma != func.gamma:
            return -1
        oabuckets, obbuckets, abuckets, bbuckets = [sift(params, _mod1) for
                params in (self.ap, self.bq, func.ap, func.bq)]

        diff = 0
        for bucket, obucket in [(abuckets, oabuckets), (bbuckets, obbuckets)]:
            for mod in set(list(bucket.keys()) + list(obucket.keys())):
                if (mod not in bucket) or (mod not in obucket) \
                        or len(bucket[mod]) != len(obucket[mod]):
                    return -1
                l1 = list(bucket[mod])
                l2 = list(obucket[mod])
                l1.sort()
                l2.sort()
                for i, j in zip(l1, l2):
                    diff += abs(i - j)

        return diff

    def _is_suitable_origin(self):
        """
        Decide if ``self`` is a suitable origin.

        Explanation
        ===========

        A function is a suitable origin iff:
        * none of the ai equals bj + n, with n a non-negative integer
        * none of the ai is zero
        * none of the bj is a non-positive integer

        Note that this gives meaningful results only when none of the indices
        are symbolic.

        """
        for a in self.ap:
            for b in self.bq:
                if (a - b).is_integer and (a - b).is_negative is False:
                    return False
        for a in self.ap:
            if a == 0:
                return False
        for b in self.bq:
            if b.is_integer and b.is_nonpositive:
                return False
        return True


class G_Function(Expr):
    """ A Meijer G-function. """

    def __new__(cls, an, ap, bm, bq):
        obj = super().__new__(cls)
        obj.an = Tuple(*list(map(expand, an)))
        obj.ap = Tuple(*list(map(expand, ap)))
        obj.bm = Tuple(*list(map(expand, bm)))
        obj.bq = Tuple(*list(map(expand, bq)))
        return obj

    @property
    def args(self):
        return (self.an, self.ap, self.bm, self.bq)

    def _hashable_content(self):
        return super()._hashable_content() + self.args

    def __call__(self, z):
        return meijerg(self.an, self.ap, self.bm, self.bq, z)

    def compute_buckets(self):
        """
        Compute buckets for the fours sets of parameters.

        Explanation
        ===========

        We guarantee that any two equal Mod objects returned are actually the
        same, and that the buckets are sorted by real part (an and bq
        descendending, bm and ap ascending).

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import G_Function
        >>> from sympy.abc import y
        >>> from sympy import S

        >>> a, b = [1, 3, 2, S(3)/2], [1 + y, y, 2, y + 3]
        >>> G_Function(a, b, [2], [y]).compute_buckets()
        ({0: [3, 2, 1], 1/2: [3/2]},
        {0: [2], y: [y, y + 1, y + 3]}, {0: [2]}, {y: [y]})

        """
        dicts = pan, pap, pbm, pbq = [defaultdict(list) for i in range(4)]
        for dic, lis in zip(dicts, (self.an, self.ap, self.bm, self.bq)):
            for x in lis:
                dic[_mod1(x)].append(x)

        for dic, flip in zip(dicts, (True, False, False, True)):
            for m, items in dic.items():
                x0 = items[0]
                items.sort(key=lambda x: x - x0, reverse=flip)
                dic[m] = items

        return tuple([dict(w) for w in dicts])

    @property
    def signature(self):
        return (len(self.an), len(self.ap), len(self.bm), len(self.bq))


# Dummy variable.
_x = Dummy('x')

class Formula:
    """
    This class represents hypergeometric formulae.

    Explanation
    ===========

    Its data members are:
    - z, the argument
    - closed_form, the closed form expression
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (see _compute_basis)

    Examples
    ========

    >>> from sympy.abc import a, b, z
    >>> from sympy.simplify.hyperexpand import Formula, Hyper_Function
    >>> func = Hyper_Function((a/2, a/3 + b, (1+a)/2), (a, b, (a+b)/7))
    >>> f = Formula(func, z, None, [a, b])

    """

    def _compute_basis(self, closed_form):
        """
        Compute a set of functions B=(f1, ..., fn), a nxn matrix M
        and a 1xn matrix C such that:
           closed_form = C B
           z d/dz B = M B.
        """
        afactors = [_x + a for a in self.func.ap]
        bfactors = [_x + b - 1 for b in self.func.bq]
        expr = _x*Mul(*bfactors) - self.z*Mul(*afactors)
        poly = Poly(expr, _x)

        n = poly.degree() - 1
        b = [closed_form]
        for _ in range(n):
            b.append(self.z*b[-1].diff(self.z))

        self.B = Matrix(b)
        self.C = Matrix([[1] + [0]*n])

        m = eye(n)
        m = m.col_insert(0, zeros(n, 1))
        l = poly.all_coeffs()[1:]
        l.reverse()
        self.M = m.row_insert(n, -Matrix([l])/poly.all_coeffs()[0])

    def __init__(self, func, z, res, symbols, B=None, C=None, M=None):
        z = sympify(z)
        res = sympify(res)
        symbols = [x for x in sympify(symbols) if func.has(x)]

        self.z = z
        self.symbols = symbols
        self.B = B
        self.C = C
        self.M = M
        self.func = func

        # TODO with symbolic parameters, it could be advantageous
        #      (for prettier answers) to compute a basis only *after*
        #      instantiation
        if res is not None:
            self._compute_basis(res)

    @property
    def closed_form(self):
        return reduce(lambda s,m: s+m[0]*m[1], zip(self.C, self.B), S.Zero)

    def find_instantiations(self, func):
        """
        Find substitutions of the free symbols that match ``func``.

        Return the substitution dictionaries as a list. Note that the returned
        instantiations need not actually match, or be valid!

        """
        from sympy.solvers import solve
        ap = func.ap
        bq = func.bq
        if len(ap) != len(self.func.ap) or len(bq) != len(self.func.bq):
            raise TypeError('Cannot instantiate other number of parameters')
        symbol_values = []
        for a in self.symbols:
            if a in self.func.ap.args:
                symbol_values.append(ap)
            elif a in self.func.bq.args:
                symbol_values.append(bq)
            else:
                raise ValueError("At least one of the parameters of the "
                        "formula must be equal to %s" % (a,))
        base_repl = [dict(list(zip(self.symbols, values)))
                for values in product(*symbol_values)]
        abuckets, bbuckets = [sift(params, _mod1) for params in [ap, bq]]
        a_inv, b_inv = [{a: len(vals) for a, vals in bucket.items()}
                for bucket in [abuckets, bbuckets]]
        critical_values = [[0] for _ in self.symbols]
        result = []
        _n = Dummy()
        for repl in base_repl:
            symb_a, symb_b = [sift(params, lambda x: _mod1(x.xreplace(repl)))
                for params in [self.func.ap, self.func.bq]]
            for bucket, obucket in [(abuckets, symb_a), (bbuckets, symb_b)]:
                for mod in set(list(bucket.keys()) + list(obucket.keys())):
                    if (mod not in bucket) or (mod not in obucket) \
                            or len(bucket[mod]) != len(obucket[mod]):
                        break
                    for a, vals in zip(self.symbols, critical_values):
                        if repl[a].free_symbols:
                            continue
                        exprs = [expr for expr in obucket[mod] if expr.has(a)]
                        repl0 = repl.copy()
                        repl0[a] += _n
                        for expr in exprs:
                            for target in bucket[mod]:
                                n0, = solve(expr.xreplace(repl0) - target, _n)
                                if n0.free_symbols:
                                    raise ValueError("Value should not be true")
                                vals.append(n0)
            else:
                values = []
                for a, vals in zip(self.symbols, critical_values):
                    a0 = repl[a]
                    min_ = floor(min(vals))
                    max_ = ceiling(max(vals))
                    values.append([a0 + n for n in range(min_, max_ + 1)])
                result.extend(dict(list(zip(self.symbols, l))) for l in product(*values))
        return result




class FormulaCollection:
    """ A collection of formulae to use as origins. """

    def __init__(self):
        """ Doing this globally at module init time is a pain ... """
        self.symbolic_formulae = {}
        self.concrete_formulae = {}
        self.formulae = []

        add_formulae(self.formulae)

        # Now process the formulae into a helpful form.
        # These dicts are indexed by (p, q).

        for f in self.formulae:
            sizes = f.func.sizes
            if len(f.symbols) > 0:
                self.symbolic_formulae.setdefault(sizes, []).append(f)
            else:
                inv = f.func.build_invariants()
                self.concrete_formulae.setdefault(sizes, {})[inv] = f

    def lookup_origin(self, func):
        """
        Given the suitable target ``func``, try to find an origin in our
        knowledge base.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import (FormulaCollection,
        ...     Hyper_Function)
        >>> f = FormulaCollection()
        >>> f.lookup_origin(Hyper_Function((), ())).closed_form
        exp(_z)
        >>> f.lookup_origin(Hyper_Function([1], ())).closed_form
        HyperRep_power1(-1, _z)

        >>> from sympy import S
        >>> i = Hyper_Function([S('1/4'), S('3/4 + 4')], [S.Half])
        >>> f.lookup_origin(i).closed_form
        HyperRep_sqrts1(-1/4, _z)
        """
        inv = func.build_invariants()
        sizes = func.sizes
        if sizes in self.concrete_formulae and \
                inv in self.concrete_formulae[sizes]:
            return self.concrete_formulae[sizes][inv]

        # We don't have a concrete formula. Try to instantiate.
        if sizes not in self.symbolic_formulae:
            return None  # Too bad...

        possible = []
        for f in self.symbolic_formulae[sizes]:
            repls = f.find_instantiations(func)
            for repl in repls:
                func2 = f.func.xreplace(repl)
                if not func2._is_suitable_origin():
                    continue
                diff = func2.difficulty(func)
                if diff == -1:
                    continue
                possible.append((diff, repl, f, func2))

        # find the nearest origin
        possible.sort(key=lambda x: x[0])
        for _, repl, f, func2 in possible:
            f2 = Formula(func2, f.z, None, [], f.B.subs(repl),
                    f.C.subs(repl), f.M.subs(repl))
            if not any(e.has(S.NaN, oo, -oo, zoo) for e in [f2.B, f2.M, f2.C]):
                return f2

        return None


class MeijerFormula:
    """
    This class represents a Meijer G-function formula.

    Its data members are:
    - z, the argument
    - symbols, the free symbols (parameters) in the formula
    - func, the function
    - B, C, M (c/f ordinary Formula)
    """

    def __init__(self, an, ap, bm, bq, z, symbols, B, C, M, matcher):
        an, ap, bm, bq = [Tuple(*list(map(expand, w))) for w in [an, ap, bm, bq]]
        self.func = G_Function(an, ap, bm, bq)
        self.z = z
        self.symbols = symbols
        self._matcher = matcher
        self.B = B
        self.C = C
        self.M = M

    @property
    def closed_form(self):
        return reduce(lambda s,m: s+m[0]*m[1], zip(self.C, self.B), S.Zero)

    def try_instantiate(self, func):
        """
        Try to instantiate the current formula to (almost) match func.
        This uses the _matcher passed on init.
        """
        if func.signature != self.func.signature:
            return None
        res = self._matcher(func)
        if res is not None:
            subs, newfunc = res
            return MeijerFormula(newfunc.an, newfunc.ap, newfunc.bm, newfunc.bq,
                                 self.z, [],
                                 self.B.subs(subs), self.C.subs(subs),
                                 self.M.subs(subs), None)


class MeijerFormulaCollection:
    """
    This class holds a collection of meijer g formulae.
    """

    def __init__(self):
        formulae = []
        add_meijerg_formulae(formulae)
        self.formulae = defaultdict(list)
        for formula in formulae:
            self.formulae[formula.func.signature].append(formula)
        self.formulae = dict(self.formulae)

    def lookup_origin(self, func):
        """ Try to find a formula that matches func. """
        if func.signature not in self.formulae:
            return None
        for formula in self.formulae[func.signature]:
            res = formula.try_instantiate(func)
            if res is not None:
                return res


class Operator:
    """
    Base class for operators to be applied to our functions.

    Explanation
    ===========

    These operators are differential operators. They are by convention
    expressed in the variable D = z*d/dz (although this base class does
    not actually care).
    Note that when the operator is applied to an object, we typically do
    *not* blindly differentiate but instead use a different representation
    of the z*d/dz operator (see make_derivative_operator).

    To subclass from this, define a __init__ method that initializes a
    self._poly variable. This variable stores a polynomial. By convention
    the generator is z*d/dz, and acts to the right of all coefficients.

    Thus this poly
        x**2 + 2*z*x + 1
    represents the differential operator
        (z*d/dz)**2 + 2*z**2*d/dz.

    This class is used only in the implementation of the hypergeometric
    function expansion algorithm.
    """

    def apply(self, obj, op):
        """
        Apply ``self`` to the object ``obj``, where the generator is ``op``.

        Examples
        ========

        >>> from sympy.simplify.hyperexpand import Operator
        >>> from sympy.polys.polytools import Poly
        >>> from sympy.abc import x, y, z
        >>> op = Operator()
        >>> op._poly = Poly(x**2 + z*x + y, x)
        >>> op.apply(z**7, lambda f: f.diff(z))
        y*z**7 + 7*z**7 + 42*z**5
        """
        coeffs = self._poly.all_coeffs()
        coeffs.reverse()
        diffs = [obj]
        for c in coeffs[1:]:
            diffs.append(op(diffs[-1]))
        r = coeffs[0]*diffs[0]
        for c, d in zip(coeffs[1:], diffs[1:]):
            r += c*d
        return r


class MultOperator(Operator):
    """ Simply multiply by a "constant" """

    def __init__(self, p):
        self._poly = Poly(p, _x)


class ShiftA(Operator):
    """ Increment an upper index. """

    def __init__(self, ai):
        ai = sympify(ai)
        if ai == 0:
            raise ValueError('Cannot increment zero upper index.')
        self._poly = Poly(_x/ai + 1, _x)

    def __str__(self):
        return '<Increment upper %s.>' % (1/self._poly.all_coeffs()[0])


class ShiftB(Operator):
    """ Decrement a lower index. """

    def __init__(self, bi):
        bi = sympify(bi)
        if bi == 1:
            raise ValueError('Cannot decrement unit lower index.')
        self._poly = Poly(_x/(bi - 1) + 1, _x)

    def __str__(self):
        return '<Decrement lower %s.>' % (1/self._poly.all_coeffs()[0] + 1)


class UnShiftA(Operator):
    """ Decrement an upper index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        ap, bq, i = list(map(sympify, [ap, bq, i]))

        self._ap = ap
        self._bq = bq
        self._i = i

        ap = list(ap)
        bq = list(bq)
        ai = ap.pop(i) - 1

        if ai == 0:
            raise ValueError('Cannot decrement unit upper index.')

        m = Poly(z*ai, _x)
        for a in ap:
            m *= Poly(_x + a, _x)

        A = Dummy('A')
        n = D = Poly(ai*A - ai, A)
        for b in bq:
            n *= D + (b - 1).as_poly(A)

        b0 = -n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper index: '
                             'cancels with lower')

        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, _x/ai + 1), _x)

        self._poly = Poly((n - m)/b0, _x)

    def __str__(self):
        return '<Decrement upper index #%s of %s, %s.>' % (self._i,
                                                        self._ap, self._bq)


class UnShiftB(Operator):
    """ Increment a lower index. """

    def __init__(self, ap, bq, i, z):
        """ Note: i counts from zero! """
        ap, bq, i = list(map(sympify, [ap, bq, i]))

        self._ap = ap
        self._bq = bq
        self._i = i

        ap = list(ap)
        bq = list(bq)
        bi = bq.pop(i) + 1

        if bi == 0:
            raise ValueError('Cannot increment -1 lower index.')

        m = Poly(_x*(bi - 1), _x)
        for b in bq:
            m *= Poly(_x + b - 1, _x)

        B = Dummy('B')
        D = Poly((bi - 1)*B - bi + 1, B)
        n = Poly(z, B)
        for a in ap:
            n *= (D + a.as_poly(B))

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment index: cancels with upper')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, _x/(bi - 1) + 1), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment lower index #%s of %s, %s.>' % (self._i,
                                                        self._ap, self._bq)


class MeijerShiftA(Operator):
    """ Increment an upper b index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(bi - _x, _x)

    def __str__(self):
        return '<Increment upper b=%s.>' % (self._poly.all_coeffs()[1])


class MeijerShiftB(Operator):
    """ Decrement an upper a index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(1 - bi + _x, _x)

    def __str__(self):
        return '<Decrement upper a=%s.>' % (1 - self._poly.all_coeffs()[1])


class MeijerShiftC(Operator):
    """ Increment a lower b index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(-bi + _x, _x)

    def __str__(self):
        return '<Increment lower b=%s.>' % (-self._poly.all_coeffs()[1])


class MeijerShiftD(Operator):
    """ Decrement a lower a index. """

    def __init__(self, bi):
        bi = sympify(bi)
        self._poly = Poly(bi - 1 - _x, _x)

    def __str__(self):
        return '<Decrement lower a=%s.>' % (self._poly.all_coeffs()[1] + 1)


class MeijerUnShiftA(Operator):
    """ Decrement an upper b index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bm.pop(i) - 1

        m = Poly(1, _x) * prod(Poly(b - _x, _x) for b in bm) * prod(Poly(_x - b, _x) for b in bq)

        A = Dummy('A')
        D = Poly(bi - A, A)
        n = Poly(z, A) * prod((D + 1 - a) for a in an) * prod((-D + a - 1) for a in ap)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement upper b index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], A).as_expr().subs(A, bi - _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Decrement upper b index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftB(Operator):
    """ Increment an upper a index. """

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        ai = an.pop(i) + 1

        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)

        B = Dummy('B')
        D = Poly(B + ai - 1, B)
        n = Poly(1, B)
        for b in bm:
            n *= (-D + b)
        for b in bq:
            n *= (D - b)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment upper a index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, 1 - ai + _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment upper a index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftC(Operator):
    """ Decrement a lower b index. """
    # XXX this is "essentially" the same as MeijerUnShiftA. This "essentially"
    #     can be made rigorous using the functional equation G(1/z) = G'(z),
    #     where G' denotes a G function of slightly altered parameters.
    #     However, sorting out the details seems harder than just coding it
    #     again.

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        bi = bq.pop(i) - 1

        m = Poly(1, _x)
        for b in bm:
            m *= Poly(b - _x, _x)
        for b in bq:
            m *= Poly(_x - b, _x)

        C = Dummy('C')
        D = Poly(bi + C, C)
        n = Poly(z, C)
        for a in an:
            n *= (D + 1 - a)
        for a in ap:
            n *= (-D + a - 1)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot decrement lower b index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], C).as_expr().subs(C, _x - bi), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Decrement lower b index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class MeijerUnShiftD(Operator):
    """ Increment a lower a index. """
    # XXX This is essentially the same as MeijerUnShiftA.
    #     See comment at MeijerUnShiftC.

    def __init__(self, an, ap, bm, bq, i, z):
        """ Note: i counts from zero! """
        an, ap, bm, bq, i = list(map(sympify, [an, ap, bm, bq, i]))

        self._an = an
        self._ap = ap
        self._bm = bm
        self._bq = bq
        self._i = i

        an = list(an)
        ap = list(ap)
        bm = list(bm)
        bq = list(bq)
        ai = ap.pop(i) + 1

        m = Poly(z, _x)
        for a in an:
            m *= Poly(1 - a + _x, _x)
        for a in ap:
            m *= Poly(a - 1 - _x, _x)

        B = Dummy('B')  # - this is the shift operator `D_I`
        D = Poly(ai - 1 - B, B)
        n = Poly(1, B)
        for b in bm:
            n *= (-D + b)
        for b in bq:
            n *= (D - b)

        b0 = n.nth(0)
        if b0 == 0:
            raise ValueError('Cannot increment lower a index (cancels)')

        n = Poly(Poly(n.all_coeffs()[:-1], B).as_expr().subs(
            B, ai - 1 - _x), _x)

        self._poly = Poly((m - n)/b0, _x)

    def __str__(self):
        return '<Increment lower a index #%s of %s, %s, %s, %s.>' % (self._i,
                                      self._an, self._ap, self._bm, self._bq)


class ReduceOrder(Operator):
    """ Reduce Order by cancelling an upper and a lower index. """

    def __new__(cls, ai, bj):
        """ For convenience if reduction is not possible, return None. """
        ai = sympify(ai)
        bj = sympify(bj)
        n = ai - bj
        if not n.is_Integer or n < 0:
            return None
        if bj.is_integer and bj.is_nonpositive:
            return None

        expr = Operator.__new__(cls)

        p = S.One
        for k in range(n):
            p *= (_x + bj + k)/(bj + k)

        expr._poly = Poly(p, _x)
        expr._a = ai
        expr._b = bj

        return expr

    @classmethod
    def _meijer(cls, b, a, sign):
        """ Cancel b + sign*s and a + sign*s
            This is for meijer G functions. """
        b = sympify(b)
        a = sympify(a)
        n = b - a
        if n.is_negative or not n.is_Integer:
            return None

        expr = Operator.__new__(cls)

        p = S.One
        for k in range(n):
            p *= (sign*_x + a + k)

        expr._poly = Poly(p, _x)
        if sign == -1:
            expr._a = b
            expr._b = a
        else:
            expr._b = Add(1, a - 1, evaluate=False)
            expr._a = Add(1, b - 1, evaluate=False)

        return expr

    @classmethod
    def meijer_minus(cls, b, a):
        return cls._meijer(b, a, -1)

    @classmethod
    def meijer_plus(cls, a, b):
        return cls._meijer(1 - a, 1 - b, 1)

    def __str__(self):
        return '<Reduce order by cancelling upper %s with lower %s.>' % \
            (self._a, self._b)


def _reduce_order(ap, bq, gen, key):
    """ Order reduction algorithm used in Hypergeometric and Meijer G """
    ap = list(ap)
    bq = list(bq)

    ap.sort(key=key)
    bq.sort(key=key)

    nap = []
    # we will edit bq in place
    operators = []
    for a in ap:
        op = None
        for i in range(len(bq)):
            op = gen(a, bq[i])
            if op is not None:
                bq.pop(i)
                break
        if op is None:
            nap.append(a)
        else:
            operators.append(op)

    return nap, bq, operators


def reduce_order(func):
    """
    Given the hypergeometric function ``func``, find a sequence of operators to
    reduces order as much as possible.

    Explanation
    ===========

    Return (newfunc, [operators]), where applying the operators to the
    hypergeometric function newfunc yields func.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import reduce_order, Hyper_Function
    >>> reduce_order(Hyper_Function((1, 2), (3, 4)))
    (Hyper_Function((1, 2), (3, 4)), [])
    >>> reduce_order(Hyper_Function((1,), (1,)))
    (Hyper_Function((), ()), [<Reduce order by cancelling upper 1 with lower 1.>])
    >>> reduce_order(Hyper_Function((2, 4), (3, 3)))
    (Hyper_Function((2,), (3,)), [<Reduce order by cancelling
    upper 4 with lower 3.>])
    """
    nap, nbq, operators = _reduce_order(func.ap, func.bq, ReduceOrder, default_sort_key)

    return Hyper_Function(Tuple(*nap), Tuple(*nbq)), operators


def reduce_order_meijer(func):
    """
    Given the Meijer G function parameters, ``func``, find a sequence of
    operators that reduces order as much as possible.

    Return newfunc, [operators].

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import (reduce_order_meijer,
    ...                                         G_Function)
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 2]))[0]
    G_Function((4, 3), (5, 6), (3, 4), (2, 1))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [3, 4], [1, 8]))[0]
    G_Function((3,), (5, 6), (3, 4), (1,))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [1, 5]))[0]
    G_Function((3,), (), (), (1,))
    >>> reduce_order_meijer(G_Function([3, 4], [5, 6], [7, 5], [5, 3]))[0]
    G_Function((), (), (), ())
    """

    nan, nbq, ops1 = _reduce_order(func.an, func.bq, ReduceOrder.meijer_plus,
                                   lambda x: default_sort_key(-x))
    nbm, nap, ops2 = _reduce_order(func.bm, func.ap, ReduceOrder.meijer_minus,
                                   default_sort_key)

    return G_Function(nan, nap, nbm, nbq), ops1 + ops2


def make_derivative_operator(M, z):
    """ Create a derivative operator, to be passed to Operator.apply. """
    def doit(C):
        r = z*C.diff(z) + C*M
        r = r.applyfunc(make_simp(z))
        return r
    return doit


def apply_operators(obj, ops, op):
    """
    Apply the list of operators ``ops`` to object ``obj``, substituting
    ``op`` for the generator.
    """
    res = obj
    for o in reversed(ops):
        res = o.apply(res, op)
    return res


def devise_plan(target, origin, z):
    """
    Devise a plan (consisting of shift and un-shift operators) to be applied
    to the hypergeometric function ``target`` to yield ``origin``.
    Returns a list of operators.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import devise_plan, Hyper_Function
    >>> from sympy.abc import z

    Nothing to do:

    >>> devise_plan(Hyper_Function((1, 2), ()), Hyper_Function((1, 2), ()), z)
    []
    >>> devise_plan(Hyper_Function((), (1, 2)), Hyper_Function((), (1, 2)), z)
    []

    Very simple plans:

    >>> devise_plan(Hyper_Function((2,), ()), Hyper_Function((1,), ()), z)
    [<Increment upper 1.>]
    >>> devise_plan(Hyper_Function((), (2,)), Hyper_Function((), (1,)), z)
    [<Increment lower index #0 of [], [1].>]

    Several buckets:

    >>> from sympy import S
    >>> devise_plan(Hyper_Function((1, S.Half), ()),
    ...             Hyper_Function((2, S('3/2')), ()), z) #doctest: +NORMALIZE_WHITESPACE
    [<Decrement upper index #0 of [3/2, 1], [].>,
    <Decrement upper index #0 of [2, 3/2], [].>]

    A slightly more complicated plan:

    >>> devise_plan(Hyper_Function((1, 3), ()), Hyper_Function((2, 2), ()), z)
    [<Increment upper 2.>, <Decrement upper index #0 of [2, 2], [].>]

    Another more complicated plan: (note that the ap have to be shifted first!)

    >>> devise_plan(Hyper_Function((1, -1), (2,)), Hyper_Function((3, -2), (4,)), z)
    [<Decrement lower 3.>, <Decrement lower 4.>,
    <Decrement upper index #1 of [-1, 2], [4].>,
    <Decrement upper index #1 of [-1, 3], [4].>, <Increment upper -2.>]
    """
    abuckets, bbuckets, nabuckets, nbbuckets = [sift(params, _mod1) for
            params in (target.ap, target.bq, origin.ap, origin.bq)]

    if len(list(abuckets.keys())) != len(list(nabuckets.keys())) or \
            len(list(bbuckets.keys())) != len(list(nbbuckets.keys())):
        raise ValueError('%s not reachable from %s' % (target, origin))

    ops = []

    def do_shifts(fro, to, inc, dec):
        ops = []
        for i in range(len(fro)):
            if to[i] - fro[i] > 0:
                sh = inc
                ch = 1
            else:
                sh = dec
                ch = -1

            while to[i] != fro[i]:
                ops += [sh(fro, i)]
                fro[i] += ch

        return ops

    def do_shifts_a(nal, nbk, al, aother, bother):
        """ Shift us from (nal, nbk) to (al, nbk). """
        return do_shifts(nal, al, lambda p, i: ShiftA(p[i]),
                         lambda p, i: UnShiftA(p + aother, nbk + bother, i, z))

    def do_shifts_b(nal, nbk, bk, aother, bother):
        """ Shift us from (nal, nbk) to (nal, bk). """
        return do_shifts(nbk, bk,
                         lambda p, i: UnShiftB(nal + aother, p + bother, i, z),
                         lambda p, i: ShiftB(p[i]))

    for r in sorted(list(abuckets.keys()) + list(bbuckets.keys()), key=default_sort_key):
        al = ()
        nal = ()
        bk = ()
        nbk = ()
        if r in abuckets:
            al = abuckets[r]
            nal = nabuckets[r]
        if r in bbuckets:
            bk = bbuckets[r]
            nbk = nbbuckets[r]
        if len(al) != len(nal) or len(bk) != len(nbk):
            raise ValueError('%s not reachable from %s' % (target, origin))

        al, nal, bk, nbk = [sorted(w, key=default_sort_key)
            for w in [al, nal, bk, nbk]]

        def others(dic, key):
            l = []
            for k in dic:
                if k != key:
                    l.extend(dic[k])
            return l
        aother = others(nabuckets, r)
        bother = others(nbbuckets, r)

        if len(al) == 0:
            # there can be no complications, just shift the bs as we please
            ops += do_shifts_b([], nbk, bk, aother, bother)
        elif len(bk) == 0:
            # there can be no complications, just shift the as as we please
            ops += do_shifts_a(nal, [], al, aother, bother)
        else:
            namax = nal[-1]
            amax = al[-1]

            if nbk[0] - namax <= 0 or bk[0] - amax <= 0:
                raise ValueError('Non-suitable parameters.')

            if namax - amax > 0:
                # we are going to shift down - first do the as, then the bs
                ops += do_shifts_a(nal, nbk, al, aother, bother)
                ops += do_shifts_b(al, nbk, bk, aother, bother)
            else:
                # we are going to shift up - first do the bs, then the as
                ops += do_shifts_b(nal, nbk, bk, aother, bother)
                ops += do_shifts_a(nal, bk, al, aother, bother)

        nabuckets[r] = al
        nbbuckets[r] = bk

    ops.reverse()
    return ops


def try_shifted_sum(func, z):
    """ Try to recognise a hypergeometric sum that starts from k > 0. """
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)
    if len(abuckets[S.Zero]) != 1:
        return None
    r = abuckets[S.Zero][0]
    if r <= 0:
        return None
    if S.Zero not in bbuckets:
        return None
    l = list(bbuckets[S.Zero])
    l.sort()
    k = l[0]
    if k <= 0:
        return None

    nap = list(func.ap)
    nap.remove(r)
    nbq = list(func.bq)
    nbq.remove(k)
    k -= 1
    nap = [x - k for x in nap]
    nbq = [x - k for x in nbq]

    ops = []
    for n in range(r - 1):
        ops.append(ShiftA(n + 1))
    ops.reverse()

    fac = factorial(k)/z**k
    fac *= Mul(*[rf(b, k) for b in nbq])
    fac /= Mul(*[rf(a, k) for a in nap])

    ops += [MultOperator(fac)]

    p = 0
    for n in range(k):
        m = z**n/factorial(n)
        m *= Mul(*[rf(a, n) for a in nap])
        m /= Mul(*[rf(b, n) for b in nbq])
        p += m

    return Hyper_Function(nap, nbq), ops, -p


def try_polynomial(func, z):
    """ Recognise polynomial cases. Returns None if not such a case.
        Requires order to be fully reduced. """
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)
    a0 = abuckets[S.Zero]
    b0 = bbuckets[S.Zero]
    a0.sort()
    b0.sort()
    al0 = [x for x in a0 if x <= 0]
    bl0 = [x for x in b0 if x <= 0]

    if bl0 and all(a < bl0[-1] for a in al0):
        return oo
    if not al0:
        return None

    a = al0[-1]
    fac = 1
    res = S.One
    for n in Tuple(*list(range(-a))):
        fac *= z
        fac /= n + 1
        fac *= Mul(*[a + n for a in func.ap])
        fac /= Mul(*[b + n for b in func.bq])
        res += fac
    return res


def try_lerchphi(func):
    """
    Try to find an expression for Hyper_Function ``func`` in terms of Lerch
    Transcendents.

    Return None if no such expression can be found.
    """
    # This is actually quite simple, and is described in Roach's paper,
    # section 18.
    # We don't need to implement the reduction to polylog here, this
    # is handled by expand_func.

    # First we need to figure out if the summation coefficient is a rational
    # function of the summation index, and construct that rational function.
    abuckets, bbuckets = sift(func.ap, _mod1), sift(func.bq, _mod1)

    paired = {}
    for key, value in abuckets.items():
        if key != 0 and key not in bbuckets:
            return None
        bvalue = bbuckets[key]
        paired[key] = (list(value), list(bvalue))
        bbuckets.pop(key, None)
    if bbuckets != {}:
        return None
    if S.Zero not in abuckets:
        return None
    aints, bints = paired[S.Zero]
    # Account for the additional n! in denominator
    paired[S.Zero] = (aints, bints + [1])

    t = Dummy('t')
    numer = S.One
    denom = S.One
    for key, (avalue, bvalue) in paired.items():
        if len(avalue) != len(bvalue):
            return None
        # Note that since order has been reduced fully, all the b are
        # bigger than all the a they differ from by an integer. In particular
        # if there are any negative b left, this function is not well-defined.
        for a, b in zip(avalue, bvalue):
            if (a - b).is_positive:
                k = a - b
                numer *= rf(b + t, k)
                denom *= rf(b, k)
            else:
                k = b - a
                numer *= rf(a, k)
                denom *= rf(a + t, k)

    # Now do a partial fraction decomposition.
    # We assemble two structures: a list monomials of pairs (a, b) representing
    # a*t**b (b a non-negative integer), and a dict terms, where
    # terms[a] = [(b, c)] means that there is a term b/(t-a)**c.
    part = apart(numer/denom, t)
    args = Add.make_args(part)
    monomials = []
    terms = {}
    for arg in args:
        numer, denom = arg.as_numer_denom()
        if not denom.has(t):
            p = Poly(numer, t)
            if not p.is_monomial:
                raise TypeError("p should be monomial")
            ((b, ), a) = p.LT()
            monomials += [(a/denom, b)]
            continue
        if numer.has(t):
            raise NotImplementedError('Need partial fraction decomposition'
                                      ' with linear denominators')
        indep, [dep] = denom.as_coeff_mul(t)
        n = 1
        if dep.is_Pow:
            n = dep.exp
            dep = dep.base
        if dep == t:
            a = 0
        elif dep.is_Add:
            a, tmp = dep.as_independent(t)
            b = 1
            if tmp != t:
                b, _ = tmp.as_independent(t)
            if dep != b*t + a:
                raise NotImplementedError('unrecognised form %s' % dep)
            a /= b
            indep *= b**n
        else:
            raise NotImplementedError('unrecognised form of partial fraction')
        terms.setdefault(a, []).append((numer/indep, n))

    # Now that we have this information, assemble our formula. All the
    # monomials yield rational functions and go into one basis element.
    # The terms[a] are related by differentiation. If the largest exponent is
    # n, we need lerchphi(z, k, a) for k = 1, 2, ..., n.
    # deriv maps a basis to its derivative, expressed as a C(z)-linear
    # combination of other basis elements.
    deriv = {}
    coeffs = {}
    z = Dummy('z')
    monomials.sort(key=lambda x: x[1])
    mon = {0: 1/(1 - z)}
    if monomials:
        for k in range(monomials[-1][1]):
            mon[k + 1] = z*mon[k].diff(z)
    for a, n in monomials:
        coeffs.setdefault(S.One, []).append(a*mon[n])
    for a, l in terms.items():
        for c, k in l:
            coeffs.setdefault(lerchphi(z, k, a), []).append(c)
        l.sort(key=lambda x: x[1])
        for k in range(2, l[-1][1] + 1):
            deriv[lerchphi(z, k, a)] = [(-a, lerchphi(z, k, a)),
                                        (1, lerchphi(z, k - 1, a))]
        deriv[lerchphi(z, 1, a)] = [(-a, lerchphi(z, 1, a)),
                                    (1/(1 - z), S.One)]
    trans = {}
    for n, b in enumerate([S.One] + list(deriv.keys())):
        trans[b] = n
    basis = [expand_func(b) for (b, _) in sorted(trans.items(),
                                                 key=lambda x:x[1])]
    B = Matrix(basis)
    C = Matrix([[0]*len(B)])
    for b, c in coeffs.items():
        C[trans[b]] = Add(*c)
    M = zeros(len(B))
    for b, l in deriv.items():
        for c, b2 in l:
            M[trans[b], trans[b2]] = c
    return Formula(func, z, None, [], B, C, M)


def build_hypergeometric_formula(func):
    """
    Create a formula object representing the hypergeometric function ``func``.

    """
    # We know that no `ap` are negative integers, otherwise "detect poly"
    # would have kicked in. However, `ap` could be empty. In this case we can
    # use a different basis.
    # I'm not aware of a basis that works in all cases.
    z = Dummy('z')
    if func.ap:
        afactors = [_x + a for a in func.ap]
        bfactors = [_x + b - 1 for b in func.bq]
        expr = _x*Mul(*bfactors) - z*Mul(*afactors)
        poly = Poly(expr, _x)
        n = poly.degree()
        basis = []
        M = zeros(n)
        for k in range(n):
            a = func.ap[0] + k
            basis += [hyper([a] + list(func.ap[1:]), func.bq, z)]
            if k < n - 1:
                M[k, k] = -a
                M[k, k + 1] = a
        B = Matrix(basis)
        C = Matrix([[1] + [0]*(n - 1)])
        derivs = [eye(n)]
        for k in range(n):
            derivs.append(M*derivs[k])
        l = poly.all_coeffs()
        l.reverse()
        res = [0]*n
        for k, c in enumerate(l):
            for r, d in enumerate(C*derivs[k]):
                res[r] += c*d
        for k, c in enumerate(res):
            M[n - 1, k] = -c/derivs[n - 1][0, n - 1]/poly.all_coeffs()[0]
        return Formula(func, z, None, [], B, C, M)
    else:
        # Since there are no `ap`, none of the `bq` can be non-positive
        # integers.
        basis = []
        bq = list(func.bq[:])
        for i in range(len(bq)):
            basis += [hyper([], bq, z)]
            bq[i] += 1
        basis += [hyper([], bq, z)]
        B = Matrix(basis)
        n = len(B)
        C = Matrix([[1] + [0]*(n - 1)])
        M = zeros(n)
        M[0, n - 1] = z/Mul(*func.bq)
        for k in range(1, n):
            M[k, k - 1] = func.bq[k - 1]
            M[k, k] = -func.bq[k - 1]
        return Formula(func, z, None, [], B, C, M)


def hyperexpand_special(ap, bq, z):
    """
    Try to find a closed-form expression for hyper(ap, bq, z), where ``z``
    is supposed to be a "special" value, e.g. 1.

    This function tries various of the classical summation formulae
    (Gauss, Saalschuetz, etc).
    """
    # This code is very ad-hoc. There are many clever algorithms
    # (notably Zeilberger's) related to this problem.
    # For now we just want a few simple cases to work.
    p, q = len(ap), len(bq)
    z_ = z
    z = unpolarify(z)
    if z == 0:
        return S.One
    from sympy.simplify.simplify import simplify
    if p == 2 and q == 1:
        # 2F1
        a, b, c = ap + bq
        if z == 1:
            # Gauss
            return gamma(c - a - b)*gamma(c)/gamma(c - a)/gamma(c - b)
        if z == -1 and simplify(b - a + c) == 1:
            b, a = a, b
        if z == -1 and simplify(a - b + c) == 1:
            # Kummer
            if b.is_integer and b.is_negative:
                return 2*cos(pi*b/2)*gamma(-b)*gamma(b - a + 1) \
                    /gamma(-b/2)/gamma(b/2 - a + 1)
            else:
                return gamma(b/2 + 1)*gamma(b - a + 1) \
                    /gamma(b + 1)/gamma(b/2 - a + 1)
    # TODO tons of more formulae
    #      investigate what algorithms exist
    return hyper(ap, bq, z_)

_collection = None


def _hyperexpand(func, z, ops0=[], z0=Dummy('z0'), premult=1, prem=0,
                 rewrite='default'):
    """
    Try to find an expression for the hypergeometric function ``func``.

    Explanation
    ===========

    The result is expressed in terms of a dummy variable ``z0``. Then it
    is multiplied by ``premult``. Then ``ops0`` is applied.
    ``premult`` must be a*z**prem for some a independent of ``z``.
    """

    if z.is_zero:
        return S.One

    from sympy.simplify.simplify import simplify

    z = polarify(z, subs=False)
    if rewrite == 'default':
        rewrite = 'nonrepsmall'

    def carryout_plan(f, ops):
        C = apply_operators(f.C.subs(f.z, z0), ops,
                            make_derivative_operator(f.M.subs(f.z, z0), z0))
        C = apply_operators(C, ops0,
                            make_derivative_operator(f.M.subs(f.z, z0)
                                         + prem*eye(f.M.shape[0]), z0))

        if premult == 1:
            C = C.applyfunc(make_simp(z0))
        r = reduce(lambda s,m: s+m[0]*m[1], zip(C, f.B.subs(f.z, z0)), S.Zero)*premult
        res = r.subs(z0, z)
        if rewrite:
            res = res.rewrite(rewrite)
        return res

    # TODO
    # The following would be possible:
    # *) PFD Duplication (see Kelly Roach's paper)
    # *) In a similar spirit, try_lerchphi() can be generalised considerably.

    global _collection
    if _collection is None:
        _collection = FormulaCollection()

    debug('Trying to expand hypergeometric function ', func)

    # First reduce order as much as possible.
    func, ops = reduce_order(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')

    # Now try polynomial cases
    res = try_polynomial(func, z0)
    if res is not None:
        debug('  Recognised polynomial.')
        p = apply_operators(res, ops, lambda f: z0*f.diff(z0))
        p = apply_operators(p*premult, ops0, lambda f: z0*f.diff(z0))
        return unpolarify(simplify(p).subs(z0, z))

    # Try to recognise a shifted sum.
    p = S.Zero
    res = try_shifted_sum(func, z0)
    if res is not None:
        func, nops, p = res
        debug('  Recognised shifted sum, reduced order to ', func)
        ops += nops

    # apply the plan for poly
    p = apply_operators(p, ops, lambda f: z0*f.diff(z0))
    p = apply_operators(p*premult, ops0, lambda f: z0*f.diff(z0))
    p = simplify(p).subs(z0, z)

    # Try special expansions early.
    if unpolarify(z) in [1, -1] and (len(func.ap), len(func.bq)) == (2, 1):
        f = build_hypergeometric_formula(func)
        r = carryout_plan(f, ops).replace(hyper, hyperexpand_special)
        if not r.has(hyper):
            return r + p

    # Try to find a formula in our collection
    formula = _collection.lookup_origin(func)

    # Now try a lerch phi formula
    if formula is None:
        formula = try_lerchphi(func)

    if formula is None:
        debug('  Could not find an origin. ',
              'Will return answer in terms of '
              'simpler hypergeometric functions.')
        formula = build_hypergeometric_formula(func)

    debug('  Found an origin: ', formula.closed_form, ' ', formula.func)

    # We need to find the operators that convert formula into func.
    ops += devise_plan(func, formula.func, z0)

    # Now carry out the plan.
    r = carryout_plan(formula, ops) + p

    return powdenest(r, polar=True).replace(hyper, hyperexpand_special)


def devise_plan_meijer(fro, to, z):
    """
    Find operators to convert G-function ``fro`` into G-function ``to``.

    Explanation
    ===========

    It is assumed that ``fro`` and ``to`` have the same signatures, and that in fact
    any corresponding pair of parameters differs by integers, and a direct path
    is possible. I.e. if there are parameters a1 b1 c1  and a2 b2 c2 it is
    assumed that a1 can be shifted to a2, etc. The only thing this routine
    determines is the order of shifts to apply, nothing clever will be tried.
    It is also assumed that ``fro`` is suitable.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import (devise_plan_meijer,
    ...                                         G_Function)
    >>> from sympy.abc import z

    Empty plan:

    >>> devise_plan_meijer(G_Function([1], [2], [3], [4]),
    ...                    G_Function([1], [2], [3], [4]), z)
    []

    Very simple plans:

    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([1], [], [], []), z)
    [<Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([-1], [], [], []), z)
    [<Decrement upper a=0.>]
    >>> devise_plan_meijer(G_Function([], [1], [], []),
    ...                    G_Function([], [2], [], []), z)
    [<Increment lower a index #0 of [], [1], [], [].>]

    Slightly more complicated plans:

    >>> devise_plan_meijer(G_Function([0], [], [], []),
    ...                    G_Function([2], [], [], []), z)
    [<Increment upper a index #0 of [1], [], [], [].>,
    <Increment upper a index #0 of [0], [], [], [].>]
    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([-1], [], [1], []), z)
    [<Increment upper b=0.>, <Decrement upper a=0.>]

    Order matters:

    >>> devise_plan_meijer(G_Function([0], [], [0], []),
    ...                    G_Function([1], [], [1], []), z)
    [<Increment upper a index #0 of [0], [], [1], [].>, <Increment upper b=0.>]
    """
    # TODO for now, we use the following simple heuristic: inverse-shift
    #      when possible, shift otherwise. Give up if we cannot make progress.

    def try_shift(f, t, shifter, diff, counter):
        """ Try to apply ``shifter`` in order to bring some element in ``f``
            nearer to its counterpart in ``to``. ``diff`` is +/- 1 and
            determines the effect of ``shifter``. Counter is a list of elements
            blocking the shift.

            Return an operator if change was possible, else None.
        """
        for idx, (a, b) in enumerate(zip(f, t)):
            if (
                (a - b).is_integer and (b - a)/diff > 0 and
                    all(a != x for x in counter)):
                sh = shifter(idx)
                f[idx] += diff
                return sh
    fan = list(fro.an)
    fap = list(fro.ap)
    fbm = list(fro.bm)
    fbq = list(fro.bq)
    ops = []
    change = True
    while change:
        change = False
        op = try_shift(fan, to.an,
                       lambda i: MeijerUnShiftB(fan, fap, fbm, fbq, i, z),
                       1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap,
                       lambda i: MeijerUnShiftD(fan, fap, fbm, fbq, i, z),
                       1, fbm + fbq)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm,
                       lambda i: MeijerUnShiftA(fan, fap, fbm, fbq, i, z),
                       -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq,
                       lambda i: MeijerUnShiftC(fan, fap, fbm, fbq, i, z),
                       -1, fan + fap)
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fan, to.an, lambda i: MeijerShiftB(fan[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fap, to.ap, lambda i: MeijerShiftD(fap[i]), -1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbm, to.bm, lambda i: MeijerShiftA(fbm[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
        op = try_shift(fbq, to.bq, lambda i: MeijerShiftC(fbq[i]), 1, [])
        if op is not None:
            ops += [op]
            change = True
            continue
    if fan != list(to.an) or fap != list(to.ap) or fbm != list(to.bm) or \
            fbq != list(to.bq):
        raise NotImplementedError('Could not devise plan.')
    ops.reverse()
    return ops

_meijercollection = None


def _meijergexpand(func, z0, allow_hyper=False, rewrite='default',
                   place=None):
    """
    Try to find an expression for the Meijer G function specified
    by the G_Function ``func``. If ``allow_hyper`` is True, then returning
    an expression in terms of hypergeometric functions is allowed.

    Currently this just does Slater's theorem.
    If expansions exist both at zero and at infinity, ``place``
    can be set to ``0`` or ``zoo`` for the preferred choice.
    """
    global _meijercollection
    if _meijercollection is None:
        _meijercollection = MeijerFormulaCollection()
    if rewrite == 'default':
        rewrite = None

    func0 = func
    debug('Try to expand Meijer G function corresponding to ', func)

    # We will play games with analytic continuation - rather use a fresh symbol
    z = Dummy('z')

    func, ops = reduce_order_meijer(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')

    # Try to find a direct formula
    f = _meijercollection.lookup_origin(func)
    if f is not None:
        debug('  Found a Meijer G formula: ', f.func)
        ops += devise_plan_meijer(f.func, func, z)

        # Now carry out the plan.
        C = apply_operators(f.C.subs(f.z, z), ops,
                            make_derivative_operator(f.M.subs(f.z, z), z))

        C = C.applyfunc(make_simp(z))
        r = C*f.B.subs(f.z, z)
        r = r[0].subs(z, z0)
        return powdenest(r, polar=True)

    debug("  Could not find a direct formula. Trying Slater's theorem.")

    # TODO the following would be possible:
    # *) Paired Index Theorems
    # *) PFD Duplication
    #    (See Kelly Roach's paper for details on either.)
    #
    # TODO Also, we tend to create combinations of gamma functions that can be
    #      simplified.

    def can_do(pbm, pap):
        """ Test if slater applies. """
        for i in pbm:
            if len(pbm[i]) > 1:
                l = 0
                if i in pap:
                    l = len(pap[i])
                if l + 1 < len(pbm[i]):
                    return False
        return True

    def do_slater(an, bm, ap, bq, z, zfinal):
        # zfinal is the value that will eventually be substituted for z.
        # We pass it to _hyperexpand to improve performance.
        func = G_Function(an, bm, ap, bq)
        _, pbm, pap, _ = func.compute_buckets()
        if not can_do(pbm, pap):
            return S.Zero, False

        cond = len(an) + len(ap) < len(bm) + len(bq)
        if len(an) + len(ap) == len(bm) + len(bq):
            cond = abs(z) < 1
        if cond is False:
            return S.Zero, False

        res = S.Zero
        for m in pbm:
            if len(pbm[m]) == 1:
                bh = pbm[m][0]
                fac = 1
                bo = list(bm)
                bo.remove(bh)
                for bj in bo:
                    fac *= gamma(bj - bh)
                for aj in an:
                    fac *= gamma(1 + bh - aj)
                for bj in bq:
                    fac /= gamma(1 + bh - bj)
                for aj in ap:
                    fac /= gamma(aj - bh)
                nap = [1 + bh - a for a in list(an) + list(ap)]
                nbq = [1 + bh - b for b in list(bo) + list(bq)]

                k = polar_lift(S.NegativeOne**(len(ap) - len(bm)))
                harg = k*zfinal
                # NOTE even though k "is" +-1, this has to be t/k instead of
                #      t*k ... we are using polar numbers for consistency!
                premult = (t/k)**bh
                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops,
                                   t, premult, bh, rewrite=None)
                res += fac * hyp
            else:
                b_ = pbm[m][0]
                ki = [bi - b_ for bi in pbm[m][1:]]
                u = len(ki)
                li = [ai - b_ for ai in pap[m][:u + 1]]
                bo = list(bm)
                for b in pbm[m]:
                    bo.remove(b)
                ao = list(ap)
                for a in pap[m][:u]:
                    ao.remove(a)
                lu = li[-1]
                di = [l - k for (l, k) in zip(li, ki)]

                # We first work out the integrand:
                s = Dummy('s')
                integrand = z**s
                for b in bm:
                    if not Mod(b, 1) and b.is_Number:
                        b = int(round(b))
                    integrand *= gamma(b - s)
                for a in an:
                    integrand *= gamma(1 - a + s)
                for b in bq:
                    integrand /= gamma(1 - b + s)
                for a in ap:
                    integrand /= gamma(a - s)

                # Now sum the finitely many residues:
                # XXX This speeds up some cases - is it a good idea?
                integrand = expand_func(integrand)
                for r in range(int(round(lu))):
                    resid = residue(integrand, s, b_ + r)
                    resid = apply_operators(resid, ops, lambda f: z*f.diff(z))
                    res -= resid

                # Now the hypergeometric term.
                au = b_ + lu
                k = polar_lift(S.NegativeOne**(len(ao) + len(bo) + 1))
                harg = k*zfinal
                premult = (t/k)**au
                nap = [1 + au - a for a in list(an) + list(ap)] + [1]
                nbq = [1 + au - b for b in list(bm) + list(bq)]

                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops,
                                   t, premult, au, rewrite=None)

                C = S.NegativeOne**(lu)/factorial(lu)
                for i in range(u):
                    C *= S.NegativeOne**di[i]/rf(lu - li[i] + 1, di[i])
                for a in an:
                    C *= gamma(1 - a + au)
                for b in bo:
                    C *= gamma(b - au)
                for a in ao:
                    C /= gamma(a - au)
                for b in bq:
                    C /= gamma(1 - b + au)

                res += C*hyp

        return res, cond

    t = Dummy('t')
    slater1, cond1 = do_slater(func.an, func.bm, func.ap, func.bq, z, z0)

    def tr(l):
        return [1 - x for x in l]

    for op in ops:
        op._poly = Poly(op._poly.subs({z: 1/t, _x: -_x}), _x)
    slater2, cond2 = do_slater(tr(func.bm), tr(func.an), tr(func.bq), tr(func.ap),
                               t, 1/z0)

    slater1 = powdenest(slater1.subs(z, z0), polar=True)
    slater2 = powdenest(slater2.subs(t, 1/z0), polar=True)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(t, 1/z)

    m = func(z)
    if m.delta > 0 or \
        (m.delta == 0 and len(m.ap) == len(m.bq) and
            (re(m.nu) < -1) is not False and polar_lift(z0) == polar_lift(1)):
        # The condition delta > 0 means that the convergence region is
        # connected. Any expression we find can be continued analytically
        # to the entire convergence region.
        # The conditions delta==0, p==q, re(nu) < -1 imply that G is continuous
        # on the positive reals, so the values at z=1 agree.
        if cond1 is not False:
            cond1 = True
        if cond2 is not False:
            cond2 = True

    if cond1 is True:
        slater1 = slater1.rewrite(rewrite or 'nonrep')
    else:
        slater1 = slater1.rewrite(rewrite or 'nonrepsmall')
    if cond2 is True:
        slater2 = slater2.rewrite(rewrite or 'nonrep')
    else:
        slater2 = slater2.rewrite(rewrite or 'nonrepsmall')

    if cond1 is not False and cond2 is not False:
        # If one condition is False, there is no choice.
        if place == 0:
            cond2 = False
        if place == zoo:
            cond1 = False

    if not isinstance(cond1, bool):
        cond1 = cond1.subs(z, z0)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(z, z0)

    def weight(expr, cond):
        if cond is True:
            c0 = 0
        elif cond is False:
            c0 = 1
        else:
            c0 = 2
        if expr.has(oo, zoo, -oo, nan):
            # XXX this actually should not happen, but consider
            # S('meijerg(((0, -1/2, 0, -1/2, 1/2), ()), ((0,),
            #   (-1/2, -1/2, -1/2, -1)), exp_polar(I*pi))/4')
            c0 = 3
        return (c0, expr.count(hyper), expr.count_ops())

    w1 = weight(slater1, cond1)
    w2 = weight(slater2, cond2)
    if min(w1, w2) <= (0, 1, oo):
        if w1 < w2:
            return slater1
        else:
            return slater2
    if max(w1[0], w2[0]) <= 1 and max(w1[1], w2[1]) <= 1:
        return Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))

    # We couldn't find an expression without hypergeometric functions.
    # TODO it would be helpful to give conditions under which the integral
    #      is known to diverge.
    r = Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    if r.has(hyper) and not allow_hyper:
        debug('  Could express using hypergeometric functions, '
              'but not allowed.')
    if not r.has(hyper) or allow_hyper:
        return r

    return func0(z0)


def hyperexpand(f, allow_hyper=False, rewrite='default', place=None):
    """
    Expand hypergeometric functions. If allow_hyper is True, allow partial
    simplification (that is a result different from input,
    but still containing hypergeometric functions).

    If a G-function has expansions both at zero and at infinity,
    ``place`` can be set to ``0`` or ``zoo`` to indicate the
    preferred choice.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import hyperexpand
    >>> from sympy.functions import hyper
    >>> from sympy.abc import z
    >>> hyperexpand(hyper([], [], z))
    exp(z)

    Non-hyperegeometric parts of the expression and hypergeometric expressions
    that are not recognised are left unchanged:

    >>> hyperexpand(1 + hyper([1, 1, 1], [], z))
    hyper((1, 1, 1), (), z) + 1
    """
    f = sympify(f)

    def do_replace(ap, bq, z):
        r = _hyperexpand(Hyper_Function(ap, bq), z, rewrite=rewrite)
        if r is None:
            return hyper(ap, bq, z)
        else:
            return r

    def do_meijer(ap, bq, z):
        r = _meijergexpand(G_Function(ap[0], ap[1], bq[0], bq[1]), z,
                   allow_hyper, rewrite=rewrite, place=place)
        if not r.has(nan, zoo, oo, -oo):
            return r
    return f.replace(hyper, do_replace).replace(meijerg, do_meijer)
<|endoftext|>"""
Discrete Fourier Transform, Number Theoretic Transform,
Walsh Hadamard Transform, Mobius Transform
"""

from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int


#----------------------------------------------------------------------------#
#                                                                            #
#                         Discrete Fourier Transform                         #
#                                                                            #
#----------------------------------------------------------------------------#

def _fourier_transform(seq, dps, inverse=False):
    """Utility function for the Discrete Fourier Transform"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of numeric coefficients "
                        "for Fourier Transform")

    a = [sympify(arg) for arg in seq]
    if any(x.has(Symbol) for x in a):
        raise ValueError("Expected non-symbolic coefficients")

    n = len(a)
    if n < 2:
        return a

    b = n.bit_length() - 1
    if n&(n - 1): # not a power of 2
        b += 1
        n = 2**b

    a += [S.Zero]*(n - len(a))
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            a[i], a[j] = a[j], a[i]

    ang = -2*pi/n if inverse else 2*pi/n

    if dps is not None:
        ang = ang.evalf(dps + 2)

    w = [cos(ang*i) + I*sin(ang*i) for i in range(n // 2)]

    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], expand_mul(a[i + j + hf]*w[ut * j])
                a[i + j], a[i + j + hf] = u + v, u - v
        h *= 2

    if inverse:
        a = [(x/n).evalf(dps) for x in a] if dps is not None \
                            else [x/n for x in a]

    return a


def fft(seq, dps=None):
    r"""
    Performs the Discrete Fourier Transform (**DFT**) in the complex domain.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 FFT* requires the number of sample points to be a power of 2.

    This method should be used with default arguments only for short sequences
    as the complexity of expressions increases with the size of the sequence.

    Parameters
    ==========

    seq : iterable
        The sequence on which **DFT** is to be applied.
    dps : Integer
        Specifies the number of decimal digits for precision.

    Examples
    ========

    >>> from sympy import fft, ifft

    >>> fft([1, 2, 3, 4])
    [10, -2 - 2*I, -2, -2 + 2*I]
    >>> ifft(_)
    [1, 2, 3, 4]

    >>> ifft([1, 2, 3, 4])
    [5/2, -1/2 + I/2, -1/2, -1/2 - I/2]
    >>> fft(_)
    [1, 2, 3, 4]

    >>> ifft([1, 7, 3, 4], dps=15)
    [3.75, -0.5 - 0.75*I, -1.75, -0.5 + 0.75*I]
    >>> fft(_)
    [1.0, 7.0, 3.0, 4.0]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
    .. [2] https://mathworld.wolfram.com/FastFourierTransform.html

    """

    return _fourier_transform(seq, dps=dps)


def ifft(seq, dps=None):
    return _fourier_transform(seq, dps=dps, inverse=True)

ifft.__doc__ = fft.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                         Number Theoretic Transform                         #
#                                                                            #
#----------------------------------------------------------------------------#

def _number_theoretic_transform(seq, prime, inverse=False):
    """Utility function for the Number Theoretic Transform"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of integer coefficients "
                        "for Number Theoretic Transform")

    p = as_int(prime)
    if not isprime(p):
        raise ValueError("Expected prime modulus for "
                        "Number Theoretic Transform")

    a = [as_int(x) % p for x in seq]

    n = len(a)
    if n < 1:
        return a

    b = n.bit_length() - 1
    if n&(n - 1):
        b += 1
        n = 2**b

    if (p - 1) % n:
        raise ValueError("Expected prime modulus of the form (m*2**k + 1)")

    a += [0]*(n - len(a))
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            a[i], a[j] = a[j], a[i]

    pr = primitive_root(p)

    rt = pow(pr, (p - 1) // n, p)
    if inverse:
        rt = pow(rt, p - 2, p)

    w = [1]*(n // 2)
    for i in range(1, n // 2):
        w[i] = w[i - 1]*rt % p

    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], a[i + j + hf]*w[ut * j]
                a[i + j], a[i + j + hf] = (u + v) % p, (u - v) % p
        h *= 2

    if inverse:
        rv = pow(n, p - 2, p)
        a = [x*rv % p for x in a]

    return a


def ntt(seq, prime):
    r"""
    Performs the Number Theoretic Transform (**NTT**), which specializes the
    Discrete Fourier Transform (**DFT**) over quotient ring `Z/pZ` for prime
    `p` instead of complex numbers `C`.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 NTT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which **DFT** is to be applied.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for performing
        **NTT** on the sequence.

    Examples
    ========

    >>> from sympy import ntt, intt
    >>> ntt([1, 2, 3, 4], prime=3*2**8 + 1)
    [10, 643, 767, 122]
    >>> intt(_, 3*2**8 + 1)
    [1, 2, 3, 4]
    >>> intt([1, 2, 3, 4], prime=3*2**8 + 1)
    [387, 415, 384, 353]
    >>> ntt(_, prime=3*2**8 + 1)
    [1, 2, 3, 4]

    References
    ==========

    .. [1] http://www.apfloat.org/ntt.html
    .. [2] https://mathworld.wolfram.com/NumberTheoreticTransform.html
    .. [3] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """

    return _number_theoretic_transform(seq, prime=prime)


def intt(seq, prime):
    return _number_theoretic_transform(seq, prime=prime, inverse=True)

intt.__doc__ = ntt.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                          Walsh Hadamard Transform                          #
#                                                                            #
#----------------------------------------------------------------------------#

def _walsh_hadamard_transform(seq, inverse=False):
    """Utility function for the Walsh Hadamard Transform"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of coefficients "
                        "for Walsh Hadamard Transform")

    a = [sympify(arg) for arg in seq]
    n = len(a)
    if n < 2:
        return a

    if n&(n - 1):
        n = 2**n.bit_length()

    a += [S.Zero]*(n - len(a))
    h = 2
    while h <= n:
        hf = h // 2
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], a[i + j + hf]
                a[i + j], a[i + j + hf] = u + v, u - v
        h *= 2

    if inverse:
        a = [x/n for x in a]

    return a


def fwht(seq):
    r"""
    Performs the Walsh Hadamard Transform (**WHT**), and uses Hadamard
    ordering for the sequence.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which WHT is to be applied.

    Examples
    ========

    >>> from sympy import fwht, ifwht
    >>> fwht([4, 2, 2, 0, 0, 2, -2, 0])
    [8, 0, 8, 0, 8, 8, 0, 0]
    >>> ifwht(_)
    [4, 2, 2, 0, 0, 2, -2, 0]

    >>> ifwht([19, -1, 11, -9, -7, 13, -15, 5])
    [2, 0, 4, 0, 3, 10, 0, 0]
    >>> fwht(_)
    [19, -1, 11, -9, -7, 13, -15, 5]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hadamard_transform
    .. [2] https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

    """

    return _walsh_hadamard_transform(seq)


def ifwht(seq):
    return _walsh_hadamard_transform(seq, inverse=True)

ifwht.__doc__ = fwht.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                    Mobius Transform for Subset Lattice                     #
#                                                                            #
#----------------------------------------------------------------------------#

def _mobius_transform(seq, sgn, subset):
    r"""Utility function for performing Mobius Transform using
    Yate's Dynamic Programming method"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of coefficients")

    a = [sympify(arg) for arg in seq]

    n = len(a)
    if n < 2:
        return a

    if n&(n - 1):
        n = 2**n.bit_length()

    a += [S.Zero]*(n - len(a))

    if subset:
        i = 1
        while i < n:
            for j in range(n):
                if j & i:
                    a[j] += sgn*a[j ^ i]
            i *= 2

    else:
        i = 1
        while i < n:
            for j in range(n):
                if j & i:
                    continue
                a[j] += sgn*a[j ^ i]
            i *= 2

    return a


def mobius_transform(seq, subset=True):
    r"""
    Performs the Mobius Transform for subset lattice with indices of
    sequence as bitmasks.

    The indices of each argument, considered as bit strings, correspond
    to subsets of a finite set.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset/superset based on bitmasks (indices) requires
    the size of sequence to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which Mobius Transform is to be applied.
    subset : bool
        Specifies if Mobius Transform is applied by enumerating subsets
        or supersets of the given set.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy import mobius_transform, inverse_mobius_transform
    >>> x, y, z = symbols('x y z')

    >>> mobius_transform([x, y, z])
    [x, x + y, x + z, x + y + z]
    >>> inverse_mobius_transform(_)
    [x, y, z, 0]

    >>> mobius_transform([x, y, z], subset=False)
    [x + y + z, y, z, 0]
    >>> inverse_mobius_transform(_, subset=False)
    [x, y, z, 0]

    >>> mobius_transform([1, 2, 3, 4])
    [1, 3, 4, 10]
    >>> inverse_mobius_transform(_)
    [1, 2, 3, 4]
    >>> mobius_transform([1, 2, 3, 4], subset=False)
    [10, 6, 7, 4]
    >>> inverse_mobius_transform(_, subset=False)
    [1, 2, 3, 4]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_inversion_formula
    .. [2] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf
    .. [3] https://arxiv.org/pdf/1211.0189.pdf

    """

    return _mobius_transform(seq, sgn=+1, subset=subset)

def inverse_mobius_transform(seq, subset=True):
    return _mobius_transform(seq, sgn=-1, subset=subset)

inverse_mobius_transform.__doc__ = mobius_transform.__doc__
<|endoftext|>"""
Convolution (using **FFT**, **NTT**, **FWHT**), Subset Convolution,
Covering Product, Intersecting Product
"""

from sympy.core import S, sympify, Rational
from sympy.core.function import expand_mul
from sympy.discrete.transforms import (
    fft, ifft, ntt, intt, fwht, ifwht,
    mobius_transform, inverse_mobius_transform)
from sympy.external.gmpy import MPZ, lcm
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int


def convolution(a, b, cycle=0, dps=None, prime=None, dyadic=None, subset=None):
    """
    Performs convolution by determining the type of desired
    convolution using hints.

    Exactly one of ``dps``, ``prime``, ``dyadic``, ``subset`` arguments
    should be specified explicitly for identifying the type of convolution,
    and the argument ``cycle`` can be specified optionally.

    For the default arguments, linear convolution is performed using **FFT**.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    cycle : Integer
        Specifies the length for doing cyclic convolution.
    dps : Integer
        Specifies the number of decimal digits for precision for
        performing **FFT** on the sequence.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for
        performing **NTT** on the sequence.
    dyadic : bool
        Identifies the convolution type as dyadic (*bitwise-XOR*)
        convolution, which is performed using **FWHT**.
    subset : bool
        Identifies the convolution type as subset convolution.

    Examples
    ========

    >>> from sympy import convolution, symbols, S, I
    >>> u, v, w, x, y, z = symbols('u v w x y z')

    >>> convolution([1 + 2*I, 4 + 3*I], [S(5)/4, 6], dps=3)
    [1.25 + 2.5*I, 11.0 + 15.8*I, 24.0 + 18.0*I]
    >>> convolution([1, 2, 3], [4, 5, 6], cycle=3)
    [31, 31, 28]

    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1)
    [1283, 19351, 14219]
    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1, cycle=2)
    [15502, 19351]

    >>> convolution([u, v], [x, y, z], dyadic=True)
    [u*x + v*y, u*y + v*x, u*z, v*z]
    >>> convolution([u, v], [x, y, z], dyadic=True, cycle=2)
    [u*x + u*z + v*y, u*y + v*x + v*z]

    >>> convolution([u, v, w], [x, y, z], subset=True)
    [u*x, u*y + v*x, u*z + w*x, v*z + w*y]
    >>> convolution([u, v, w], [x, y, z], subset=True, cycle=3)
    [u*x + v*z + w*y, u*y + v*x, u*z + w*x]

    """

    c = as_int(cycle)
    if c < 0:
        raise ValueError("The length for cyclic convolution "
                        "must be non-negative")

    dyadic = True if dyadic else None
    subset = True if subset else None
    if sum(x is not None for x in (prime, dps, dyadic, subset)) > 1:
        raise TypeError("Ambiguity in determining the type of convolution")

    if prime is not None:
        ls = convolution_ntt(a, b, prime=prime)
        return ls if not c else [sum(ls[i::c]) % prime for i in range(c)]

    if dyadic:
        ls = convolution_fwht(a, b)
    elif subset:
        ls = convolution_subset(a, b)
    else:
        def loop(a):
            dens = []
            for i in a:
                if isinstance(i, Rational) and i.q - 1:
                    dens.append(i.q)
                elif not isinstance(i, int):
                    return
            if dens:
                l = lcm(*dens)
                return [i*l if type(i) is int else i.p*(l//i.q) for i in a], l
            # no lcm of den to deal with
            return a, 1
        ls = None
        da = loop(a)
        if da is not None:
            db = loop(b)
            if db is not None:
                (ia, ma), (ib, mb) = da, db
                den = ma*mb
                ls = convolution_int(ia, ib)
                if den != 1:
                    ls = [Rational(i, den) for i in ls]
        if ls is None:
            ls = convolution_fft(a, b, dps)

    return ls if not c else [sum(ls[i::c]) for i in range(c)]


#----------------------------------------------------------------------------#
#                                                                            #
#                       Convolution for Complex domain                       #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_fft(a, b, dps=None):
    """
    Performs linear convolution using Fast Fourier Transform.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    dps : Integer
        Specifies the number of decimal digits for precision.

    Examples
    ========

    >>> from sympy import S, I
    >>> from sympy.discrete.convolutions import convolution_fft

    >>> convolution_fft([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_fft([2, 5], [6, 7, 3])
    [12, 44, 41, 15]
    >>> convolution_fft([1 + 2*I, 4 + 3*I], [S(5)/4, 6])
    [5/4 + 5*I/2, 11 + 63*I/4, 24 + 18*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """

    a, b = a[:], b[:]
    n = m = len(a) + len(b) - 1 # convolution size

    if n > 0 and n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = fft(a, dps), fft(b, dps)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = ifft(a, dps)[:m]

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                           Convolution for GF(p)                            #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_ntt(a, b, prime):
    """
    Performs linear convolution using Number Theoretic Transform.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for performing
        **NTT** on the sequence.

    Examples
    ========

    >>> from sympy.discrete.convolutions import convolution_ntt
    >>> convolution_ntt([2, 3], [4, 5], prime=19*2**10 + 1)
    [8, 22, 15]
    >>> convolution_ntt([2, 5], [6, 7, 3], prime=19*2**10 + 1)
    [12, 44, 41, 15]
    >>> convolution_ntt([333, 555], [222, 666], prime=19*2**10 + 1)
    [15555, 14219, 19404]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """

    a, b, p = a[:], b[:], as_int(prime)
    n = m = len(a) + len(b) - 1 # convolution size

    if n > 0 and n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [0]*(n - len(a))
    b += [0]*(n - len(b))

    a, b = ntt(a, p), ntt(b, p)
    a = [x*y % p for x, y in zip(a, b)]
    a = intt(a, p)[:m]

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                         Convolution for 2**n-group                         #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_fwht(a, b):
    """
    Performs dyadic (*bitwise-XOR*) convolution using Fast Walsh Hadamard
    Transform.

    The convolution is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S, I
    >>> from sympy.discrete.convolutions import convolution_fwht

    >>> u, v, x, y = symbols('u v x y')
    >>> convolution_fwht([u, v], [x, y])
    [u*x + v*y, u*y + v*x]

    >>> convolution_fwht([2, 3], [4, 5])
    [23, 22]
    >>> convolution_fwht([2, 5 + 4*I, 7], [6*I, 7, 3 + 4*I])
    [56 + 68*I, -10 + 30*I, 6 + 50*I, 48 + 32*I]

    >>> convolution_fwht([S(33)/7, S(55)/6, S(7)/4], [S(2)/3, 5])
    [2057/42, 1870/63, 7/6, 35/4]

    References
    ==========

    .. [1] https://www.radioeng.cz/fulltexts/2002/02_03_40_42.pdf
    .. [2] https://en.wikipedia.org/wiki/Hadamard_transform

    """

    if not a or not b:
        return []

    a, b = a[:], b[:]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = fwht(a), fwht(b)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = ifwht(a)

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                            Subset Convolution                              #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_subset(a, b):
    """
    Performs Subset Convolution of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S
    >>> from sympy.discrete.convolutions import convolution_subset
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> convolution_subset([u, v], [x, y])
    [u*x, u*y + v*x]
    >>> convolution_subset([u, v, x], [y, z])
    [u*y, u*z + v*y, x*y, x*z]

    >>> convolution_subset([1, S(2)/3], [3, 4])
    [3, 6]
    >>> convolution_subset([1, 3, S(5)/7], [7])
    [7, 21, 5, 0]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """

    if not a or not b:
        return []

    if not iterable(a) or not iterable(b):
        raise TypeError("Expected a sequence of coefficients for convolution")

    a = [sympify(arg) for arg in a]
    b = [sympify(arg) for arg in b]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    c = [S.Zero]*n

    for mask in range(n):
        smask = mask
        while smask > 0:
            c[mask] += expand_mul(a[smask] * b[mask^smask])
            smask = (smask - 1)&mask

        c[mask] += expand_mul(a[smask] * b[mask^smask])

    return c


#----------------------------------------------------------------------------#
#                                                                            #
#                              Covering Product                              #
#                                                                            #
#----------------------------------------------------------------------------#

def covering_product(a, b):
    """
    Returns the covering product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The covering product of given sequences is a sequence which contains
    the sum of products of the elements of the given sequences grouped by
    the *bitwise-OR* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which covering product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, I, covering_product
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> covering_product([u, v], [x, y])
    [u*x, u*y + v*x + v*y]
    >>> covering_product([u, v, x], [y, z])
    [u*y, u*z + v*y + v*z, x*y, x*z]

    >>> covering_product([1, S(2)/3], [3, 4 + 5*I])
    [3, 26/3 + 25*I/3]
    >>> covering_product([1, 3, S(5)/7], [7, 8])
    [7, 53, 5, 40/7]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """

    if not a or not b:
        return []

    a, b = a[:], b[:]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = mobius_transform(a), mobius_transform(b)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = inverse_mobius_transform(a)

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                            Intersecting Product                            #
#                                                                            #
#----------------------------------------------------------------------------#

def intersecting_product(a, b):
    """
    Returns the intersecting product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The intersecting product of given sequences is the sequence which
    contains the sum of products of the elements of the given sequences
    grouped by the *bitwise-AND* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which intersecting product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, I, intersecting_product
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> intersecting_product([u, v], [x, y])
    [u*x + u*y + v*x, v*y]
    >>> intersecting_product([u, v, x], [y, z])
    [u*y + u*z + v*y + x*y + x*z, v*z, 0, 0]

    >>> intersecting_product([1, S(2)/3], [3, 4 + 5*I])
    [9 + 5*I, 8/3 + 10*I/3]
    >>> intersecting_product([1, 3, S(5)/7], [7, 8])
    [327/7, 24, 0, 0]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """

    if not a or not b:
        return []

    a, b = a[:], b[:]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = mobius_transform(a, subset=False), mobius_transform(b, subset=False)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = inverse_mobius_transform(a, subset=False)

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                            Integer Convolutions                            #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_int(a, b):
    """Return the convolution of two sequences as a list.

    The iterables must consist solely of integers.

    Parameters
    ==========

    a, b : Sequence
        The sequences for which convolution is performed.

    Explanation
    ===========

    This function performs the convolution of ``a`` and ``b`` by packing
    each into a single integer, multiplying them together, and then
    unpacking the result from the product.  The intuition behind this is
    that if we evaluate some polynomial [1]:

    .. math ::
        1156x^6 + 3808x^5 + 8440x^4 + 14856x^3 + 16164x^2 + 14040x + 8100

    at say $x = 10^5$ we obtain $1156038080844014856161641404008100$.
    Note we can read of the coefficients for each term every five digits.
    If the $x$ we chose to evaluate at is large enough, the same will hold
    for the product.

    The idea now is since big integer multiplication in libraries such
    as GMP is highly optimised, this will be reasonably fast.

    Examples
    ========

    >>> from sympy.discrete.convolutions import convolution_int

    >>> convolution_int([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_int([1, 1, -1], [1, 1])
    [1, 2, 0, -1]

    References
    ==========

    .. [1] Fateman, Richard J.
           Can you save time in multiplying polynomials by encoding them as integers?
           University of California, Berkeley, California (2004).
           https://people.eecs.berkeley.edu/~fateman/papers/polysbyGMP.pdf
    """
    # An upper bound on the largest coefficient in p(x)q(x) is given by (1 + min(dp, dq))N(p)N(q)
    # where dp = deg(p), dq = deg(q), N(f) denotes the coefficient of largest modulus in f [1]
    B = max(abs(c) for c in a)*max(abs(c) for c in b)*(1 + min(len(a) - 1, len(b) - 1))
    x, power = MPZ(1), 0
    while x <= (2*B):  # multiply by two for negative coefficients, see [1]
        x <<= 1
        power += 1

    def to_integer(poly):
        n, mul = MPZ(0), 0
        for c in reversed(poly):
            if c and not mul: mul = -1 if c < 0 else 1
            n <<= power
            n += mul*int(c)
        return mul, n

    # Perform packing and multiplication
    (a_mul, a_packed), (b_mul, b_packed) = to_integer(a), to_integer(b)
    result = a_packed * b_packed

    # Perform unpacking
    mul = a_mul * b_mul
    mask, half, borrow, poly = x - 1, x >> 1, 0, []
    while result or borrow:
        coeff = (result & mask) + borrow
        result >>= power
        borrow = coeff >= half
        poly.append(mul * int(coeff if coeff < half else coeff - x))
    return poly or [0]
<|endoftext|>"""
Recurrences
"""

from sympy.core import S, sympify
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int


def linrec(coeffs, init, n):
    r"""
    Evaluation of univariate linear recurrences of homogeneous type
    having coefficients independent of the recurrence variable.

    Parameters
    ==========

    coeffs : iterable
        Coefficients of the recurrence
    init : iterable
        Initial values of the recurrence
    n : Integer
        Point of evaluation for the recurrence

    Notes
    =====

    Let `y(n)` be the recurrence of given type, ``c`` be the sequence
    of coefficients, ``b`` be the sequence of initial/base values of the
    recurrence and ``k`` (equal to ``len(c)``) be the order of recurrence.
    Then,

    .. math :: y(n) = \begin{cases} b_n & 0 \le n < k \\
        c_0 y(n-1) + c_1 y(n-2) + \cdots + c_{k-1} y(n-k) & n \ge k
        \end{cases}

    Let `x_0, x_1, \ldots, x_n` be a sequence and consider the transformation
    that maps each polynomial `f(x)` to `T(f(x))` where each power `x^i` is
    replaced by the corresponding value `x_i`. The sequence is then a solution
    of the recurrence if and only if `T(x^i p(x)) = 0` for each `i \ge 0` where
    `p(x) = x^k - c_0 x^(k-1) - \cdots - c_{k-1}` is the characteristic
    polynomial.

    Then `T(f(x)p(x)) = 0` for each polynomial `f(x)` (as it is a linear
    combination of powers `x^i`). Now, if `x^n` is congruent to
    `g(x) = a_0 x^0 + a_1 x^1 + \cdots + a_{k-1} x^{k-1}` modulo `p(x)`, then
    `T(x^n) = x_n` is equal to
    `T(g(x)) = a_0 x_0 + a_1 x_1 + \cdots + a_{k-1} x_{k-1}`.

    Computation of `x^n`,
    given `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`
    is performed using exponentiation by squaring (refer to [1_]) with
    an additional reduction step performed to retain only first `k` powers
    of `x` in the representation of `x^n`.

    Examples
    ========

    >>> from sympy.discrete.recurrences import linrec
    >>> from sympy.abc import x, y, z

    >>> linrec(coeffs=[1, 1], init=[0, 1], n=10)
    55

    >>> linrec(coeffs=[1, 1], init=[x, y], n=10)
    34*x + 55*y

    >>> linrec(coeffs=[x, y], init=[0, 1], n=5)
    x**2*y + x*(x**3 + 2*x*y) + y**2

    >>> linrec(coeffs=[1, 2, 3, 0, 0, 4], init=[x, y, z], n=16)
    13576*x + 5676*y + 2356*z

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    .. [2] https://en.wikipedia.org/w/index.php?title=Modular_exponentiation&section=6#Matrices

    See Also
    ========

    sympy.polys.agca.extensions.ExtensionElement.__pow__

    """

    if not coeffs:
        return S.Zero

    if not iterable(coeffs):
        raise TypeError("Expected a sequence of coefficients for"
                        " the recurrence")

    if not iterable(init):
        raise TypeError("Expected a sequence of values for the initialization"
                        " of the recurrence")

    n = as_int(n)
    if n < 0:
        raise ValueError("Point of evaluation of recurrence must be a "
                        "non-negative integer")

    c = [sympify(arg) for arg in coeffs]
    b = [sympify(arg) for arg in init]
    k = len(c)

    if len(b) > k:
        raise TypeError("Count of initial values should not exceed the "
                        "order of the recurrence")
    else:
        b += [S.Zero]*(k - len(b)) # remaining initial values default to zero

    if n < k:
        return b[n]
    terms = [u*v for u, v in zip(linrec_coeffs(c, n), b)]
    return sum(terms[:-1], terms[-1])


def linrec_coeffs(c, n):
    r"""
    Compute the coefficients of n'th term in linear recursion
    sequence defined by c.

    `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`.

    It computes the coefficients by using binary exponentiation.
    This function is used by `linrec` and `_eval_pow_by_cayley`.

    Parameters
    ==========

    c = coefficients of the divisor polynomial
    n = exponent of x, so dividend is x^n

    """

    k = len(c)

    def _square_and_reduce(u, offset):
        # squares `(u_0 + u_1 x + u_2 x^2 + \cdots + u_{k-1} x^k)` (and
        # multiplies by `x` if offset is 1) and reduces the above result of
        # length upto `2k` to `k` using the characteristic equation of the
        # recurrence given by, `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`

        w = [S.Zero]*(2*len(u) - 1 + offset)
        for i, p in enumerate(u):
            for j, q in enumerate(u):
                w[offset + i + j] += p*q

        for j in range(len(w) - 1, k - 1, -1):
            for i in range(k):
                w[j - i - 1] += w[j]*c[i]

        return w[:k]

    def _final_coeffs(n):
        # computes the final coefficient list - `cf` corresponding to the
        # point at which recurrence is to be evalauted - `n`, such that,
        # `y(n) = cf_0 y(k-1) + cf_1 y(k-2) + \cdots + cf_{k-1} y(0)`

        if n < k:
            return [S.Zero]*n + [S.One] + [S.Zero]*(k - n - 1)
        else:
            return _square_and_reduce(_final_coeffs(n // 2), n % 2)

    return _final_coeffs(n)
<|endoftext|>from sympy.core import Basic, Integer
import operator


class OmegaPower(Basic):
    """
    Represents ordinal exponential and multiplication terms one of the
    building blocks of the :class:`Ordinal` class.
    In ``OmegaPower(a, b)``, ``a`` represents exponent and ``b`` represents multiplicity.
    """
    def __new__(cls, a, b):
        if isinstance(b, int):
            b = Integer(b)
        if not isinstance(b, Integer) or b <= 0:
            raise TypeError("multiplicity must be a positive integer")

        if not isinstance(a, Ordinal):
            a = Ordinal.convert(a)

        return Basic.__new__(cls, a, b)

    @property
    def exp(self):
        return self.args[0]

    @property
    def mult(self):
        return self.args[1]

    def _compare_term(self, other, op):
        if self.exp == other.exp:
            return op(self.mult, other.mult)
        else:
            return op(self.exp, other.exp)

    def __eq__(self, other):
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self.args == other.args

    def __hash__(self):
        return Basic.__hash__(self)

    def __lt__(self, other):
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self._compare_term(other, operator.lt)


class Ordinal(Basic):
    """
    Represents ordinals in Cantor normal form.

    Internally, this class is just a list of instances of OmegaPower.

    Examples
    ========
    >>> from sympy import Ordinal, OmegaPower
    >>> from sympy.sets.ordinals import omega
    >>> w = omega
    >>> w.is_limit_ordinal
    True
    >>> Ordinal(OmegaPower(w + 1, 1), OmegaPower(3, 2))
    w**(w + 1) + w**3*2
    >>> 3 + w
    w
    >>> (w + 1) * w
    w**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ordinal_arithmetic
    """
    def __new__(cls, *terms):
        obj = super().__new__(cls, *terms)
        powers = [i.exp for i in obj.args]
        if not all(powers[i] >= powers[i+1] for i in range(len(powers) - 1)):
            raise ValueError("powers must be in decreasing order")
        return obj

    @property
    def terms(self):
        return self.args

    @property
    def leading_term(self):
        if self == ord0:
            raise ValueError("ordinal zero has no leading term")
        return self.terms[0]

    @property
    def trailing_term(self):
        if self == ord0:
            raise ValueError("ordinal zero has no trailing term")
        return self.terms[-1]

    @property
    def is_successor_ordinal(self):
        try:
            return self.trailing_term.exp == ord0
        except ValueError:
            return False

    @property
    def is_limit_ordinal(self):
        try:
            return not self.trailing_term.exp == ord0
        except ValueError:
            return False

    @property
    def degree(self):
        return self.leading_term.exp

    @classmethod
    def convert(cls, integer_value):
        if integer_value == 0:
            return ord0
        return Ordinal(OmegaPower(0, integer_value))

    def __eq__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return self.terms == other.terms

    def __hash__(self):
        return hash(self.args)

    def __lt__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        for term_self, term_other in zip(self.terms, other.terms):
            if term_self != term_other:
                return term_self < term_other
        return len(self.terms) < len(other.terms)

    def __le__(self, other):
        return (self == other or self < other)

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __str__(self):
        net_str = ""
        plus_count = 0
        if self == ord0:
            return 'ord0'
        for i in self.terms:
            if plus_count:
                net_str += " + "

            if i.exp == ord0:
                net_str += str(i.mult)
            elif i.exp == 1:
                net_str += 'w'
            elif len(i.exp.terms) > 1 or i.exp.is_limit_ordinal:
                net_str += 'w**(%s)'%i.exp
            else:
                net_str += 'w**%s'%i.exp

            if not i.mult == 1 and not i.exp == ord0:
                net_str += '*%s'%i.mult

            plus_count += 1
        return(net_str)

    __repr__ = __str__

    def __add__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        if other == ord0:
            return self
        a_terms = list(self.terms)
        b_terms = list(other.terms)
        r = len(a_terms) - 1
        b_exp = other.degree
        while r >= 0 and a_terms[r].exp < b_exp:
            r -= 1
        if r < 0:
            terms = b_terms
        elif a_terms[r].exp == b_exp:
            sum_term = OmegaPower(b_exp, a_terms[r].mult + other.leading_term.mult)
            terms = a_terms[:r] + [sum_term] + b_terms[1:]
        else:
            terms = a_terms[:r+1] + b_terms
        return Ordinal(*terms)

    def __radd__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return other + self

    def __mul__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        if ord0 in (self, other):
            return ord0
        a_exp = self.degree
        a_mult = self.leading_term.mult
        summation = []
        if other.is_limit_ordinal:
            for arg in other.terms:
                summation.append(OmegaPower(a_exp + arg.exp, arg.mult))

        else:
            for arg in other.terms[:-1]:
                summation.append(OmegaPower(a_exp + arg.exp, arg.mult))
            b_mult = other.trailing_term.mult
            summation.append(OmegaPower(a_exp, a_mult*b_mult))
            summation += list(self.terms[1:])
        return Ordinal(*summation)

    def __rmul__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return other * self

    def __pow__(self, other):
        if not self == omega:
            return NotImplemented
        return Ordinal(OmegaPower(other, 1))


class OrdinalZero(Ordinal):
    """The ordinal zero.

    OrdinalZero can be imported as ``ord0``.
    """
    pass


class OrdinalOmega(Ordinal):
    """The ordinal omega which forms the base of all ordinals in cantor normal form.

    OrdinalOmega can be imported as ``omega``.

    Examples
    ========

    >>> from sympy.sets.ordinals import omega
    >>> omega + omega
    w*2
    """
    def __new__(cls):
        return Ordinal.__new__(cls)

    @property
    def terms(self):
        return (OmegaPower(1, 1),)


ord0 = OrdinalZero()
omega = OrdinalOmega()
<|endoftext|>from sympy.core import S
from sympy.core.sympify import sympify
from sympy.core.relational import Eq, Ne
from sympy.core.parameters import global_parameters
from sympy.logic.boolalg import Boolean
from sympy.utilities.misc import func_name
from .sets import Set


class Contains(Boolean):
    """
    Asserts that x is an element of the set S.

    Examples
    ========

    >>> from sympy import Symbol, Integer, S, Contains
    >>> Contains(Integer(2), S.Integers)
    True
    >>> Contains(Integer(-2), S.Naturals)
    False
    >>> i = Symbol('i', integer=True)
    >>> Contains(i, S.Naturals)
    Contains(i, Naturals)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Element_%28mathematics%29
    """
    def __new__(cls, x, s, evaluate=None):
        x = sympify(x)
        s = sympify(s)

        if evaluate is None:
            evaluate = global_parameters.evaluate

        if not isinstance(s, Set):
            raise TypeError('expecting Set, not %s' % func_name(s))

        if evaluate:
            # _contains can return symbolic booleans that would be returned by
            # s.contains(x) but here for Contains(x, s) we only evaluate to
            # true, false or return the unevaluated Contains.
            result = s._contains(x)

            if isinstance(result, Boolean):
                if result in (S.true, S.false):
                    return result
            elif result is not None:
                raise TypeError("_contains() should return Boolean or None")

        return super().__new__(cls, x, s)

    @property
    def binary_symbols(self):
        return set().union(*[i.binary_symbols
            for i in self.args[1].args
            if i.is_Boolean or i.is_Symbol or
            isinstance(i, (Eq, Ne))])

    def as_set(self):
        return self.args[1]
<|endoftext|>from sympy.core.decorators import _sympifyit
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.singleton import S
from sympy.core.sympify import _sympify

from .sets import Set, FiniteSet, SetKind


class PowerSet(Set):
    r"""A symbolic object representing a power set.

    Parameters
    ==========

    arg : Set
        The set to take power of.

    evaluate : bool
        The flag to control evaluation.

        If the evaluation is disabled for finite sets, it can take
        advantage of using subset test as a membership test.

    Notes
    =====

    Power set `\mathcal{P}(S)` is defined as a set containing all the
    subsets of `S`.

    If the set `S` is a finite set, its power set would have
    `2^{\left| S \right|}` elements, where `\left| S \right|` denotes
    the cardinality of `S`.

    Examples
    ========

    >>> from sympy import PowerSet, S, FiniteSet

    A power set of a finite set:

    >>> PowerSet(FiniteSet(1, 2, 3))
    PowerSet({1, 2, 3})

    A power set of an empty set:

    >>> PowerSet(S.EmptySet)
    PowerSet(EmptySet)
    >>> PowerSet(PowerSet(S.EmptySet))
    PowerSet(PowerSet(EmptySet))

    A power set of an infinite set:

    >>> PowerSet(S.Reals)
    PowerSet(Reals)

    Evaluating the power set of a finite set to its explicit form:

    >>> PowerSet(FiniteSet(1, 2, 3)).rewrite(FiniteSet)
    FiniteSet(EmptySet, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3})

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Power_set

    .. [2] https://en.wikipedia.org/wiki/Axiom_of_power_set
    """
    def __new__(cls, arg, evaluate=None):
        if evaluate is None:
            evaluate=global_parameters.evaluate

        arg = _sympify(arg)

        if not isinstance(arg, Set):
            raise ValueError('{} must be a set.'.format(arg))

        return super().__new__(cls, arg)

    @property
    def arg(self):
        return self.args[0]

    def _eval_rewrite_as_FiniteSet(self, *args, **kwargs):
        arg = self.arg
        if arg.is_FiniteSet:
            return arg.powerset()
        return None

    @_sympifyit('other', NotImplemented)
    def _contains(self, other):
        if not isinstance(other, Set):
            return None

        return fuzzy_bool(self.arg.is_superset(other))

    def _eval_is_subset(self, other):
        if isinstance(other, PowerSet):
            return self.arg.is_subset(other.arg)

    def __len__(self):
        return 2 ** len(self.arg)

    def __iter__(self):
        found = [S.EmptySet]
        yield S.EmptySet

        for x in self.arg:
            temp = []
            x = FiniteSet(x)
            for y in found:
                new = x + y
                yield new
                temp.append(new)
            found.extend(temp)

    @property
    def kind(self):
        return SetKind(self.arg.kind)
<|endoftext|>from .sets import (Set, Interval, Union, FiniteSet, ProductSet,
        Intersection, imageset, Complement, SymmetricDifference,
        DisjointUnion)

from .fancysets import ImageSet, Range, ComplexRegion
from .contains import Contains
from .conditionset import ConditionSet
from .ordinals import Ordinal, OmegaPower, ord0
from .powerset import PowerSet
from ..core.singleton import S
from .handlers.comparison import _eval_is_eq  # noqa:F401
Complexes = S.Complexes
EmptySet = S.EmptySet
Integers = S.Integers
Naturals = S.Naturals
Naturals0 = S.Naturals0
Rationals = S.Rationals
Reals = S.Reals
UniversalSet = S.UniversalSet

__all__ = [
    'Set', 'Interval', 'Union', 'EmptySet', 'FiniteSet', 'ProductSet',
    'Intersection', 'imageset', 'Complement', 'SymmetricDifference', 'DisjointUnion',

    'ImageSet', 'Range', 'ComplexRegion', 'Reals',

    'Contains',

    'ConditionSet',

    'Ordinal', 'OmegaPower', 'ord0',

    'PowerSet',

    'Reals', 'Naturals', 'Naturals0', 'UniversalSet', 'Integers', 'Rationals',
]
<|endoftext|>from functools import reduce
from itertools import product

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import fuzzy_not, fuzzy_or, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.intfunc import igcd
from sympy.core.numbers import oo, Rational, Integer
from sympy.core.relational import Eq, is_eq
from sympy.core.kind import NumberKind
from sympy.core.singleton import Singleton, S
from sympy.core.symbol import Dummy, symbols, Symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.logic.boolalg import And, Or
from .sets import tfn, Set, Interval, Union, FiniteSet, ProductSet, SetKind
from sympy.utilities.misc import filldedent


class Rationals(Set, metaclass=Singleton):
    """
    Represents the rational numbers. This set is also available as
    the singleton ``S.Rationals``.

    Examples
    ========

    >>> from sympy import S
    >>> S.Half in S.Rationals
    True
    >>> iterable = iter(S.Rationals)
    >>> [next(iterable) for i in range(12)]
    [0, 1, -1, 1/2, 2, -1/2, -2, 1/3, 3, -1/3, -3, 2/3]
    """

    is_iterable = True
    _inf = S.NegativeInfinity
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return tfn[other.is_rational]

    def __iter__(self):
        yield S.Zero
        yield S.One
        yield S.NegativeOne
        d = 2
        while True:
            for n in range(d):
                if igcd(n, d) == 1:
                    yield Rational(n, d)
                    yield Rational(d, n)
                    yield Rational(-n, d)
                    yield Rational(-d, n)
            d += 1

    @property
    def _boundary(self):
        return S.Reals

    def _kind(self):
        return SetKind(NumberKind)


class Naturals(Set, metaclass=Singleton):
    """
    Represents the natural numbers (or counting numbers) which are all
    positive integers starting from 1. This set is also available as
    the singleton ``S.Naturals``.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Naturals)
    >>> next(iterable)
    1
    >>> next(iterable)
    2
    >>> next(iterable)
    3
    >>> pprint(S.Naturals.intersect(Interval(0, 10)))
    {1, 2, ..., 10}

    See Also
    ========

    Naturals0 : non-negative integers (i.e. includes 0, too)
    Integers : also includes negative integers
    """

    is_iterable = True
    _inf: Integer = S.One
    _sup = S.Infinity
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_positive and other.is_integer:
            return S.true
        elif other.is_integer is False or other.is_positive is False:
            return S.false

    def _eval_is_subset(self, other):
        return Range(1, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(1, oo).is_superset(other)

    def __iter__(self):
        i = self._inf
        while True:
            yield i
            i = i + 1

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        return And(Eq(floor(x), x), x >= self.inf, x < oo)

    def _kind(self):
        return SetKind(NumberKind)


class Naturals0(Naturals):
    """Represents the whole numbers which are all the non-negative integers,
    inclusive of zero.

    See Also
    ========

    Naturals : positive integers; does not include 0
    Integers : also includes the negative integers
    """
    _inf = S.Zero

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        elif other.is_integer and other.is_nonnegative:
            return S.true
        elif other.is_integer is False or other.is_nonnegative is False:
            return S.false

    def _eval_is_subset(self, other):
        return Range(oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(oo).is_superset(other)


class Integers(Set, metaclass=Singleton):
    """
    Represents all integers: positive, negative and zero. This set is also
    available as the singleton ``S.Integers``.

    Examples
    ========

    >>> from sympy import S, Interval, pprint
    >>> 5 in S.Naturals
    True
    >>> iterable = iter(S.Integers)
    >>> next(iterable)
    0
    >>> next(iterable)
    1
    >>> next(iterable)
    -1
    >>> next(iterable)
    2

    >>> pprint(S.Integers.intersect(Interval(-4, 4)))
    {-4, -3, ..., 4}

    See Also
    ========

    Naturals0 : non-negative integers
    Integers : positive and negative integers and zero
    """

    is_iterable = True
    is_empty = False
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, Expr):
            return S.false
        return tfn[other.is_integer]

    def __iter__(self):
        yield S.Zero
        i = S.One
        while True:
            yield i
            yield -i
            i = i + 1

    @property
    def _inf(self):
        return S.NegativeInfinity

    @property
    def _sup(self):
        return S.Infinity

    @property
    def _boundary(self):
        return self

    def _kind(self):
        return SetKind(NumberKind)

    def as_relational(self, x):
        return And(Eq(floor(x), x), -oo < x, x < oo)

    def _eval_is_subset(self, other):
        return Range(-oo, oo).is_subset(other)

    def _eval_is_superset(self, other):
        return Range(-oo, oo).is_superset(other)


class Reals(Interval, metaclass=Singleton):
    """
    Represents all real numbers
    from negative infinity to positive infinity,
    including all integer, rational and irrational numbers.
    This set is also available as the singleton ``S.Reals``.


    Examples
    ========

    >>> from sympy import S, Rational, pi, I
    >>> 5 in S.Reals
    True
    >>> Rational(-1, 2) in S.Reals
    True
    >>> pi in S.Reals
    True
    >>> 3*I in S.Reals
    False
    >>> S.Reals.contains(pi)
    True


    See Also
    ========

    ComplexRegion
    """
    @property
    def start(self):
        return S.NegativeInfinity

    @property
    def end(self):
        return S.Infinity

    @property
    def left_open(self):
        return True

    @property
    def right_open(self):
        return True

    def __eq__(self, other):
        return other == Interval(S.NegativeInfinity, S.Infinity)

    def __hash__(self):
        return hash(Interval(S.NegativeInfinity, S.Infinity))


class ImageSet(Set):
    """
    Image of a set under a mathematical function. The transformation
    must be given as a Lambda function which has as many arguments
    as the elements of the set upon which it operates, e.g. 1 argument
    when acting on the set of integers or 2 arguments when acting on
    a complex region.

    This function is not normally called directly, but is called
    from ``imageset``.


    Examples
    ========

    >>> from sympy import Symbol, S, pi, Dummy, Lambda
    >>> from sympy import FiniteSet, ImageSet, Interval

    >>> x = Symbol('x')
    >>> N = S.Naturals
    >>> squares = ImageSet(Lambda(x, x**2), N) # {x**2 for x in N}
    >>> 4 in squares
    True
    >>> 5 in squares
    False

    >>> FiniteSet(0, 1, 2, 3, 4, 5, 6, 7, 9, 10).intersect(squares)
    {1, 4, 9}

    >>> square_iterable = iter(squares)
    >>> for i in range(4):
    ...     next(square_iterable)
    1
    4
    9
    16

    If you want to get value for `x` = 2, 1/2 etc. (Please check whether the
    `x` value is in ``base_set`` or not before passing it as args)

    >>> squares.lamda(2)
    4
    >>> squares.lamda(S(1)/2)
    1/4

    >>> n = Dummy('n')
    >>> solutions = ImageSet(Lambda(n, n*pi), S.Integers) # solutions of sin(x) = 0
    >>> dom = Interval(-1, 1)
    >>> dom.intersect(solutions)
    {0}

    See Also
    ========

    sympy.sets.sets.imageset
    """
    def __new__(cls, flambda, *sets):
        if not isinstance(flambda, Lambda):
            raise ValueError('First argument must be a Lambda')

        signature = flambda.signature

        if len(signature) != len(sets):
            raise ValueError('Incompatible signature')

        sets = [_sympify(s) for s in sets]

        if not all(isinstance(s, Set) for s in sets):
            raise TypeError("Set arguments to ImageSet should of type Set")

        if not all(cls._check_sig(sg, st) for sg, st in zip(signature, sets)):
            raise ValueError("Signature %s does not match sets %s" % (signature, sets))

        if flambda is S.IdentityFunction and len(sets) == 1:
            return sets[0]

        if not set(flambda.variables) & flambda.expr.free_symbols:
            is_empty = fuzzy_or(s.is_empty for s in sets)
            if is_empty == True:
                return S.EmptySet
            elif is_empty == False:
                return FiniteSet(flambda.expr)

        return Basic.__new__(cls, flambda, *sets)

    lamda = property(lambda self: self.args[0])
    base_sets = property(lambda self: self.args[1:])

    @property
    def base_set(self):
        # XXX: Maybe deprecate this? It is poorly defined in handling
        # the multivariate case...
        sets = self.base_sets
        if len(sets) == 1:
            return sets[0]
        else:
            return ProductSet(*sets).flatten()

    @property
    def base_pset(self):
        return ProductSet(*self.base_sets)

    @classmethod
    def _check_sig(cls, sig_i, set_i):
        if sig_i.is_symbol:
            return True
        elif isinstance(set_i, ProductSet):
            sets = set_i.sets
            if len(sig_i) != len(sets):
                return False
            # Recurse through the signature for nested tuples:
            return all(cls._check_sig(ts, ps) for ts, ps in zip(sig_i, sets))
        else:
            # XXX: Need a better way of checking whether a set is a set of
            # Tuples or not. For example a FiniteSet can contain Tuples
            # but so can an ImageSet or a ConditionSet. Others like
            # Integers, Reals etc can not contain Tuples. We could just
            # list the possibilities here... Current code for e.g.
            # _contains probably only works for ProductSet.
            return True # Give the benefit of the doubt

    def __iter__(self):
        already_seen = set()
        for i in self.base_pset:
            val = self.lamda(*i)
            if val in already_seen:
                continue
            else:
                already_seen.add(val)
                yield val

    def _is_multivariate(self):
        return len(self.lamda.variables) > 1

    def _contains(self, other):
        from sympy.solvers.solveset import _solveset_multi

        def get_symsetmap(signature, base_sets):
            '''Attempt to get a map of symbols to base_sets'''
            queue = list(zip(signature, base_sets))
            symsetmap = {}
            for sig, base_set in queue:
                if sig.is_symbol:
                    symsetmap[sig] = base_set
                elif base_set.is_ProductSet:
                    sets = base_set.sets
                    if len(sig) != len(sets):
                        raise ValueError("Incompatible signature")
                    # Recurse
                    queue.extend(zip(sig, sets))
                else:
                    # If we get here then we have something like sig = (x, y) and
                    # base_set = {(1, 2), (3, 4)}. For now we give up.
                    return None

            return symsetmap

        def get_equations(expr, candidate):
            '''Find the equations relating symbols in expr and candidate.'''
            queue = [(expr, candidate)]
            for e, c in queue:
                if not isinstance(e, Tuple):
                    yield Eq(e, c)
                elif not isinstance(c, Tuple) or len(e) != len(c):
                    yield False
                    return
                else:
                    queue.extend(zip(e, c))

        # Get the basic objects together:
        other = _sympify(other)
        expr = self.lamda.expr
        sig = self.lamda.signature
        variables = self.lamda.variables
        base_sets = self.base_sets

        # Use dummy symbols for ImageSet parameters so they don't match
        # anything in other
        rep = {v: Dummy(v.name) for v in variables}
        variables = [v.subs(rep) for v in variables]
        sig = sig.subs(rep)
        expr = expr.subs(rep)

        # Map the parts of other to those in the Lambda expr
        equations = []
        for eq in get_equations(expr, other):
            # Unsatisfiable equation?
            if eq is False:
                return S.false
            equations.append(eq)

        # Map the symbols in the signature to the corresponding domains
        symsetmap = get_symsetmap(sig, base_sets)
        if symsetmap is None:
            # Can't factor the base sets to a ProductSet
            return None

        # Which of the variables in the Lambda signature need to be solved for?
        symss = (eq.free_symbols for eq in equations)
        variables = set(variables) & reduce(set.union, symss, set())

        # Use internal multivariate solveset
        variables = tuple(variables)
        base_sets = [symsetmap[v] for v in variables]
        solnset = _solveset_multi(equations, variables, base_sets)
        if solnset is None:
            return None
        return tfn[fuzzy_not(solnset.is_empty)]

    @property
    def is_iterable(self):
        return all(s.is_iterable for s in self.base_sets)

    def doit(self, **hints):
        from sympy.sets.setexpr import SetExpr
        f = self.lamda
        sig = f.signature
        if len(sig) == 1 and sig[0].is_symbol and isinstance(f.expr, Expr):
            base_set = self.base_sets[0]
            return SetExpr(base_set)._eval_func(f).set
        if all(s.is_FiniteSet for s in self.base_sets):
            return FiniteSet(*(f(*a) for a in product(*self.base_sets)))
        return self

    def _kind(self):
        return SetKind(self.lamda.expr.kind)


class Range(Set):
    """
    Represents a range of integers. Can be called as ``Range(stop)``,
    ``Range(start, stop)``, or ``Range(start, stop, step)``; when ``step`` is
    not given it defaults to 1.

    ``Range(stop)`` is the same as ``Range(0, stop, 1)`` and the stop value
    (just as for Python ranges) is not included in the Range values.

        >>> from sympy import Range
        >>> list(Range(3))
        [0, 1, 2]

    The step can also be negative:

        >>> list(Range(10, 0, -2))
        [10, 8, 6, 4, 2]

    The stop value is made canonical so equivalent ranges always
    have the same args:

        >>> Range(0, 10, 3)
        Range(0, 12, 3)

    Infinite ranges are allowed. ``oo`` and ``-oo`` are never included in the
    set (``Range`` is always a subset of ``Integers``). If the starting point
    is infinite, then the final value is ``stop - step``. To iterate such a
    range, it needs to be reversed:

        >>> from sympy import oo
        >>> r = Range(-oo, 1)
        >>> r[-1]
        0
        >>> next(iter(r))
        Traceback (most recent call last):
        ...
        TypeError: Cannot iterate over Range with infinite start
        >>> next(iter(r.reversed))
        0

    Although ``Range`` is a :class:`Set` (and supports the normal set
    operations) it maintains the order of the elements and can
    be used in contexts where ``range`` would be used.

        >>> from sympy import Interval
        >>> Range(0, 10, 2).intersect(Interval(3, 7))
        Range(4, 8, 2)
        >>> list(_)
        [4, 6]

    Although slicing of a Range will always return a Range -- possibly
    empty -- an empty set will be returned from any intersection that
    is empty:

        >>> Range(3)[:0]
        Range(0, 0, 1)
        >>> Range(3).intersect(Interval(4, oo))
        EmptySet
        >>> Range(3).intersect(Range(4, oo))
        EmptySet

    Range will accept symbolic arguments but has very limited support
    for doing anything other than displaying the Range:

        >>> from sympy import Symbol, pprint
        >>> from sympy.abc import i, j, k
        >>> Range(i, j, k).start
        i
        >>> Range(i, j, k).inf
        Traceback (most recent call last):
        ...
        ValueError: invalid method for symbolic range

    Better success will be had when using integer symbols:

        >>> n = Symbol('n', integer=True)
        >>> r = Range(n, n + 20, 3)
        >>> r.inf
        n
        >>> pprint(r)
        {n, n + 3, ..., n + 18}
    """

    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], range):
                raise TypeError(
                    'use sympify(%s) to convert range to Range' % args[0])

        # expand range
        slc = slice(*args)

        if slc.step == 0:
            raise ValueError("step cannot be 0")

        start, stop, step = slc.start or 0, slc.stop, slc.step or 1
        try:
            ok = []
            for w in (start, stop, step):
                w = sympify(w)
                if w in [S.NegativeInfinity, S.Infinity] or (
                        w.has(Symbol) and w.is_integer != False):
                    ok.append(w)
                elif not w.is_Integer:
                    if w.is_infinite:
                        raise ValueError('infinite symbols not allowed')
                    raise ValueError
                else:
                    ok.append(w)
        except ValueError:
            raise ValueError(filldedent('''
    Finite arguments to Range must be integers; `imageset` can define
    other cases, e.g. use `imageset(i, i/10, Range(3))` to give
    [0, 1/10, 1/5].'''))
        start, stop, step = ok

        null = False
        if any(i.has(Symbol) for i in (start, stop, step)):
            dif = stop - start
            n = dif/step
            if n.is_Rational:
                if dif == 0:
                    null = True
                else:  # (x, x + 5, 2) or (x, 3*x, x)
                    n = floor(n)
                    end = start + n*step
                    if dif.is_Rational:  # (x, x + 5, 2)
                        if (end - stop).is_negative:
                            end += step
                    else:  # (x, 3*x, x)
                        if (end/stop - 1).is_negative:
                            end += step
            elif n.is_extended_negative:
                null = True
            else:
                end = stop  # other methods like sup and reversed must fail
        elif start.is_infinite:
            span = step*(stop - start)
            if span is S.NaN or span <= 0:
                null = True
            elif step.is_Integer and stop.is_infinite and abs(step) != 1:
                raise ValueError(filldedent('''
                    Step size must be %s in this case.''' % (1 if step > 0 else -1)))
            else:
                end = stop
        else:
            oostep = step.is_infinite
            if oostep:
                step = S.One if step > 0 else S.NegativeOne
            n = ceiling((stop - start)/step)
            if n <= 0:
                null = True
            elif oostep:
                step = S.One  # make it canonical
                end = start + step
            else:
                end = start + n*step
        if null:
            start = end = S.Zero
            step = S.One
        return Basic.__new__(cls, start, end, step)

    start = property(lambda self: self.args[0])
    stop = property(lambda self: self.args[1])
    step = property(lambda self: self.args[2])

    @property
    def reversed(self):
        """Return an equivalent Range in the opposite order.

        Examples
        ========

        >>> from sympy import Range
        >>> Range(10).reversed
        Range(9, -1, -1)
        """
        if self.has(Symbol):
            n = (self.stop - self.start)/self.step
            if not n.is_extended_positive or not all(
                    i.is_integer or i.is_infinite for i in self.args):
                raise ValueError('invalid method for symbolic range')
        if self.start == self.stop:
            return self
        return self.func(
            self.stop - self.step, self.start - self.step, -self.step)

    def _kind(self):
        return SetKind(NumberKind)

    def _contains(self, other):
        if self.start == self.stop:
            return S.false
        if other.is_infinite:
            return S.false
        if not other.is_integer:
            return tfn[other.is_integer]
        if self.has(Symbol):
            n = (self.stop - self.start)/self.step
            if not n.is_extended_positive or not all(
                    i.is_integer or i.is_infinite for i in self.args):
                return
        else:
            n = self.size
        if self.start.is_finite:
            ref = self.start
        elif self.stop.is_finite:
            ref = self.stop
        else:  # both infinite; step is +/- 1 (enforced by __new__)
            return S.true
        if n == 1:
            return Eq(other, self[0])
        res = (ref - other) % self.step
        if res == S.Zero:
            if self.has(Symbol):
                d = Dummy('i')
                return self.as_relational(d).subs(d, other)
            return And(other >= self.inf, other <= self.sup)
        elif res.is_Integer:  # off sequence
            return S.false
        else:  # symbolic/unsimplified residue modulo step
            return None

    def __iter__(self):
        n = self.size  # validate
        if not (n.has(S.Infinity) or n.has(S.NegativeInfinity) or n.is_Integer):
            raise TypeError("Cannot iterate over symbolic Range")
        if self.start in [S.NegativeInfinity, S.Infinity]:
            raise TypeError("Cannot iterate over Range with infinite start")
        elif self.start != self.stop:
            i = self.start
            if n.is_infinite:
                while True:
                    yield i
                    i += self.step
            else:
                for _ in range(n):
                    yield i
                    i += self.step

    @property
    def is_iterable(self):
        # Check that size can be determined, used by __iter__
        dif = self.stop - self.start
        n = dif/self.step
        if not (n.has(S.Infinity) or n.has(S.NegativeInfinity) or n.is_Integer):
            return False
        if self.start in [S.NegativeInfinity, S.Infinity]:
            return False
        if not (n.is_extended_nonnegative and all(i.is_integer for i in self.args)):
            return False
        return True

    def __len__(self):
        rv = self.size
        if rv is S.Infinity:
            raise ValueError('Use .size to get the length of an infinite Range')
        return int(rv)

    @property
    def size(self):
        if self.start == self.stop:
            return S.Zero
        dif = self.stop - self.start
        n = dif/self.step
        if n.is_infinite:
            return S.Infinity
        if  n.is_extended_nonnegative and all(i.is_integer for i in self.args):
            return abs(floor(n))
        raise ValueError('Invalid method for symbolic Range')

    @property
    def is_finite_set(self):
        if self.start.is_integer and self.stop.is_integer:
            return True
        return self.size.is_finite

    @property
    def is_empty(self):
        try:
            return self.size.is_zero
        except ValueError:
            return None

    def __bool__(self):
        # this only distinguishes between definite null range
        # and non-null/unknown null; getting True doesn't mean
        # that it actually is not null
        b = is_eq(self.start, self.stop)
        if b is None:
            raise ValueError('cannot tell if Range is null or not')
        return not bool(b)

    def __getitem__(self, i):
        ooslice = "cannot slice from the end with an infinite value"
        zerostep = "slice step cannot be zero"
        infinite = "slicing not possible on range with infinite start"
        # if we had to take every other element in the following
        # oo, ..., 6, 4, 2, 0
        # we might get oo, ..., 4, 0 or oo, ..., 6, 2
        ambiguous = "cannot unambiguously re-stride from the end " + \
            "with an infinite value"
        if isinstance(i, slice):
            if self.size.is_finite:  # validates, too
                if self.start == self.stop:
                    return Range(0)
                start, stop, step = i.indices(self.size)
                n = ceiling((stop - start)/step)
                if n <= 0:
                    return Range(0)
                canonical_stop = start + n*step
                end = canonical_stop - step
                ss = step*self.step
                return Range(self[start], self[end] + ss, ss)
            else:  # infinite Range
                start = i.start
                stop = i.stop
                if i.step == 0:
                    raise ValueError(zerostep)
                step = i.step or 1
                ss = step*self.step
                #---------------------
                # handle infinite Range
                #   i.e. Range(-oo, oo) or Range(oo, -oo, -1)
                # --------------------
                if self.start.is_infinite and self.stop.is_infinite:
                    raise ValueError(infinite)
                #---------------------
                # handle infinite on right
                #   e.g. Range(0, oo) or Range(0, -oo, -1)
                # --------------------
                if self.stop.is_infinite:
                    # start and stop are not interdependent --
                    # they only depend on step --so we use the
                    # equivalent reversed values
                    return self.reversed[
                        stop if stop is None else -stop + 1:
                        start if start is None else -start:
                        step].reversed
                #---------------------
                # handle infinite on the left
                #   e.g. Range(oo, 0, -1) or Range(-oo, 0)
                # --------------------
                # consider combinations of
                # start/stop {== None, < 0, == 0, > 0} and
                # step {< 0, > 0}
                if start is None:
                    if stop is None:
                        if step < 0:
                            return Range(self[-1], self.start, ss)
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step < 0:
                            return Range(self[-1], self[stop], ss)
                        else:  # > 0
                            return Range(self.start, self[stop], ss)
                    elif stop == 0:
                        if step > 0:
                            return Range(0)
                        else:  # < 0
                            raise ValueError(ooslice)
                    elif stop == 1:
                        if step > 0:
                            raise ValueError(ooslice)  # infinite singleton
                        else:  # < 0
                            raise ValueError(ooslice)
                    else:  # > 1
                        raise ValueError(ooslice)
                elif start < 0:
                    if stop is None:
                        if step < 0:
                            return Range(self[start], self.start, ss)
                        else:  # > 0
                            return Range(self[start], self.stop, ss)
                    elif stop < 0:
                        return Range(self[start], self[stop], ss)
                    elif stop == 0:
                        if step < 0:
                            raise ValueError(ooslice)
                        else:  # > 0
                            return Range(0)
                    elif stop > 0:
                        raise ValueError(ooslice)
                elif start == 0:
                    if stop is None:
                        if step < 0:
                            raise ValueError(ooslice)  # infinite singleton
                        elif step > 1:
                            raise ValueError(ambiguous)
                        else:  # == 1
                            return self
                    elif stop < 0:
                        if step > 1:
                            raise ValueError(ambiguous)
                        elif step == 1:
                            return Range(self.start, self[stop], ss)
                        else:  # < 0
                            return Range(0)
                    else:  # >= 0
                        raise ValueError(ooslice)
                elif start > 0:
                    raise ValueError(ooslice)
        else:
            if self.start == self.stop:
                raise IndexError('Range index out of range')
            if not (all(i.is_integer or i.is_infinite
                    for i in self.args) and ((self.stop - self.start)/
                    self.step).is_extended_positive):
                raise ValueError('Invalid method for symbolic Range')
            if i == 0:
                if self.start.is_infinite:
                    raise ValueError(ooslice)
                return self.start
            if i == -1:
                if self.stop.is_infinite:
                    raise ValueError(ooslice)
                return self.stop - self.step
            n = self.size  # must be known for any other index
            rv = (self.stop if i < 0 else self.start) + i*self.step
            if rv.is_infinite:
                raise ValueError(ooslice)
            val = (rv - self.start)/self.step
            rel = fuzzy_or([val.is_infinite,
                            fuzzy_and([val.is_nonnegative, (n-val).is_nonnegative])])
            if rel:
                return rv
            if rel is None:
                raise ValueError('Invalid method for symbolic Range')
            raise IndexError("Range index out of range")

    @property
    def _inf(self):
        if not self:
            return S.EmptySet.inf
        if self.has(Symbol):
            if all(i.is_integer or i.is_infinite for i in self.args):
                dif = self.stop - self.start
                if self.step.is_positive and dif.is_positive:
                    return self.start
                elif self.step.is_negative and dif.is_negative:
                    return self.stop - self.step
            raise ValueError('invalid method for symbolic range')
        if self.step > 0:
            return self.start
        else:
            return self.stop - self.step

    @property
    def _sup(self):
        if not self:
            return S.EmptySet.sup
        if self.has(Symbol):
            if all(i.is_integer or i.is_infinite for i in self.args):
                dif = self.stop - self.start
                if self.step.is_positive and dif.is_positive:
                    return self.stop - self.step
                elif self.step.is_negative and dif.is_negative:
                    return self.start
            raise ValueError('invalid method for symbolic range')
        if self.step > 0:
            return self.stop - self.step
        else:
            return self.start

    @property
    def _boundary(self):
        return self

    def as_relational(self, x):
        """Rewrite a Range in terms of equalities and logic operators. """
        if self.start.is_infinite:
            assert not self.stop.is_infinite  # by instantiation
            a = self.reversed.start
        else:
            a = self.start
        step = self.step
        in_seq = Eq(Mod(x - a, step), 0)
        ints = And(Eq(Mod(a, 1), 0), Eq(Mod(step, 1), 0))
        n = (self.stop - self.start)/self.step
        if n == 0:
            return S.EmptySet.as_relational(x)
        if n == 1:
            return And(Eq(x, a), ints)
        try:
            a, b = self.inf, self.sup
        except ValueError:
            a = None
        if a is not None:
            range_cond = And(
                x > a if a.is_infinite else x >= a,
                x < b if b.is_infinite else x <= b)
        else:
            a, b = self.start, self.stop - self.step
            range_cond = Or(
                And(self.step >= 1, x > a if a.is_infinite else x >= a,
                x < b if b.is_infinite else x <= b),
                And(self.step <= -1, x < a if a.is_infinite else x <= a,
                x > b if b.is_infinite else x >= b))
        return And(in_seq, ints, range_cond)


_sympy_converter[range] = lambda r: Range(r.start, r.stop, r.step)

def normalize_theta_set(theta):
    r"""
    Normalize a Real Set `theta` in the interval `[0, 2\pi)`. It returns
    a normalized value of theta in the Set. For Interval, a maximum of
    one cycle $[0, 2\pi]$, is returned i.e. for theta equal to $[0, 10\pi]$,
    returned normalized value would be $[0, 2\pi)$. As of now intervals
    with end points as non-multiples of ``pi`` is not supported.

    Raises
    ======

    NotImplementedError
        The algorithms for Normalizing theta Set are not yet
        implemented.
    ValueError
        The input is not valid, i.e. the input is not a real set.
    RuntimeError
        It is a bug, please report to the github issue tracker.

    Examples
    ========

    >>> from sympy.sets.fancysets import normalize_theta_set
    >>> from sympy import Interval, FiniteSet, pi
    >>> normalize_theta_set(Interval(9*pi/2, 5*pi))
    Interval(pi/2, pi)
    >>> normalize_theta_set(Interval(-3*pi/2, pi/2))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-pi/2, pi/2))
    Union(Interval(0, pi/2), Interval.Ropen(3*pi/2, 2*pi))
    >>> normalize_theta_set(Interval(-4*pi, 3*pi))
    Interval.Ropen(0, 2*pi)
    >>> normalize_theta_set(Interval(-3*pi/2, -pi/2))
    Interval(pi/2, 3*pi/2)
    >>> normalize_theta_set(FiniteSet(0, pi, 3*pi))
    {0, pi}

    """
    from sympy.functions.elementary.trigonometric import _pi_coeff

    if theta.is_Interval:
        interval_len = theta.measure
        # one complete circle
        if interval_len >= 2*S.Pi:
            if interval_len == 2*S.Pi and theta.left_open and theta.right_open:
                k = _pi_coeff(theta.start)
                return Union(Interval(0, k*S.Pi, False, True),
                        Interval(k*S.Pi, 2*S.Pi, True, True))
            return Interval(0, 2*S.Pi, False, True)

        k_start, k_end = _pi_coeff(theta.start), _pi_coeff(theta.end)

        if k_start is None or k_end is None:
            raise NotImplementedError("Normalizing theta without pi as coefficient is "
                                    "not yet implemented")
        new_start = k_start*S.Pi
        new_end = k_end*S.Pi

        if new_start > new_end:
            return Union(Interval(S.Zero, new_end, False, theta.right_open),
                         Interval(new_start, 2*S.Pi, theta.left_open, True))
        else:
            return Interval(new_start, new_end, theta.left_open, theta.right_open)

    elif theta.is_FiniteSet:
        new_theta = []
        for element in theta:
            k = _pi_coeff(element)
            if k is None:
                raise NotImplementedError('Normalizing theta without pi as '
                                          'coefficient, is not Implemented.')
            else:
                new_theta.append(k*S.Pi)
        return FiniteSet(*new_theta)

    elif theta.is_Union:
        return Union(*[normalize_theta_set(interval) for interval in theta.args])

    elif theta.is_subset(S.Reals):
        raise NotImplementedError("Normalizing theta when, it is of type %s is not "
                                  "implemented" % type(theta))
    else:
        raise ValueError(" %s is not a real set" % (theta))


class ComplexRegion(Set):
    r"""
    Represents the Set of all Complex Numbers. It can represent a
    region of Complex Plane in both the standard forms Polar and
    Rectangular coordinates.

    * Polar Form
      Input is in the form of the ProductSet or Union of ProductSets
      of the intervals of ``r`` and ``theta``, and use the flag ``polar=True``.

      .. math:: Z = \{z \in \mathbb{C} \mid z = r\times (\cos(\theta) + I\sin(\theta)), r \in [\texttt{r}], \theta \in [\texttt{theta}]\}

    * Rectangular Form
      Input is in the form of the ProductSet or Union of ProductSets
      of interval of x and y, the real and imaginary parts of the Complex numbers in a plane.
      Default input type is in rectangular form.

    .. math:: Z = \{z \in \mathbb{C} \mid z = x + Iy, x \in [\operatorname{re}(z)], y \in [\operatorname{im}(z)]\}

    Examples
    ========

    >>> from sympy import ComplexRegion, Interval, S, I, Union
    >>> a = Interval(2, 3)
    >>> b = Interval(4, 6)
    >>> c1 = ComplexRegion(a*b)  # Rectangular Form
    >>> c1
    CartesianComplexRegion(ProductSet(Interval(2, 3), Interval(4, 6)))

    * c1 represents the rectangular region in complex plane
      surrounded by the coordinates (2, 4), (3, 4), (3, 6) and
      (2, 6), of the four vertices.

    >>> c = Interval(1, 8)
    >>> c2 = ComplexRegion(Union(a*b, b*c))
    >>> c2
    CartesianComplexRegion(Union(ProductSet(Interval(2, 3), Interval(4, 6)), ProductSet(Interval(4, 6), Interval(1, 8))))

    * c2 represents the Union of two rectangular regions in complex
      plane. One of them surrounded by the coordinates of c1 and
      other surrounded by the coordinates (4, 1), (6, 1), (6, 8) and
      (4, 8).

    >>> 2.5 + 4.5*I in c1
    True
    >>> 2.5 + 6.5*I in c1
    False

    >>> r = Interval(0, 1)
    >>> theta = Interval(0, 2*S.Pi)
    >>> c2 = ComplexRegion(r*theta, polar=True)  # Polar Form
    >>> c2  # unit Disk
    PolarComplexRegion(ProductSet(Interval(0, 1), Interval.Ropen(0, 2*pi)))

    * c2 represents the region in complex plane inside the
      Unit Disk centered at the origin.

    >>> 0.5 + 0.5*I in c2
    True
    >>> 1 + 2*I in c2
    False

    >>> unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*S.Pi), polar=True)
    >>> upper_half_unit_disk = ComplexRegion(Interval(0, 1)*Interval(0, S.Pi), polar=True)
    >>> intersection = unit_disk.intersect(upper_half_unit_disk)
    >>> intersection
    PolarComplexRegion(ProductSet(Interval(0, 1), Interval(0, pi)))
    >>> intersection == upper_half_unit_disk
    True

    See Also
    ========

    CartesianComplexRegion
    PolarComplexRegion
    Complexes

    """
    is_ComplexRegion = True

    def __new__(cls, sets, polar=False):
        if polar is False:
            return CartesianComplexRegion(sets)
        elif polar is True:
            return PolarComplexRegion(sets)
        else:
            raise ValueError("polar should be either True or False")

    @property
    def sets(self):
        """
        Return raw input sets to the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.sets
        ProductSet(Interval(2, 3), Interval(4, 5))
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.sets
        Union(ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        return self.args[0]

    @property
    def psets(self):
        """
        Return a tuple of sets (ProductSets) input of the self.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)),)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.psets
        (ProductSet(Interval(2, 3), Interval(4, 5)), ProductSet(Interval(4, 5), Interval(1, 7)))

        """
        if self.sets.is_ProductSet:
            psets = ()
            psets = psets + (self.sets, )
        else:
            psets = self.sets.args
        return psets

    @property
    def a_interval(self):
        """
        Return the union of intervals of `x` when, self is in
        rectangular form, or the union of intervals of `r` when
        self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.a_interval
        Interval(2, 3)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.a_interval
        Union(Interval(2, 3), Interval(4, 5))

        """
        a_interval = []
        for element in self.psets:
            a_interval.append(element.args[0])

        a_interval = Union(*a_interval)
        return a_interval

    @property
    def b_interval(self):
        """
        Return the union of intervals of `y` when, self is in
        rectangular form, or the union of intervals of `theta`
        when self is in polar form.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, Union
        >>> a = Interval(2, 3)
        >>> b = Interval(4, 5)
        >>> c = Interval(1, 7)
        >>> C1 = ComplexRegion(a*b)
        >>> C1.b_interval
        Interval(4, 5)
        >>> C2 = ComplexRegion(Union(a*b, b*c))
        >>> C2.b_interval
        Interval(1, 7)

        """
        b_interval = []
        for element in self.psets:
            b_interval.append(element.args[1])

        b_interval = Union(*b_interval)
        return b_interval

    @property
    def _measure(self):
        """
        The measure of self.sets.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion, S
        >>> a, b = Interval(2, 5), Interval(4, 8)
        >>> c = Interval(0, 2*S.Pi)
        >>> c1 = ComplexRegion(a*b)
        >>> c1.measure
        12
        >>> c2 = ComplexRegion(a*c, polar=True)
        >>> c2.measure
        6*pi

        """
        return self.sets._measure

    def _kind(self):
        return self.args[0].kind

    @classmethod
    def from_real(cls, sets):
        """
        Converts given subset of real numbers to a complex region.

        Examples
        ========

        >>> from sympy import Interval, ComplexRegion
        >>> unit = Interval(0,1)
        >>> ComplexRegion.from_real(unit)
        CartesianComplexRegion(ProductSet(Interval(0, 1), {0}))

        """
        if not sets.is_subset(S.Reals):
            raise ValueError("sets must be a subset of the real line")

        return CartesianComplexRegion(sets * FiniteSet(0))

    def _contains(self, other):
        from sympy.functions import arg, Abs

        isTuple = isinstance(other, Tuple)
        if isTuple and len(other) != 2:
            raise ValueError('expecting Tuple of length 2')

