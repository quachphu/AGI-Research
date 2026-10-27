    .. [2] K. Geddes, L. Stefanus, On the Risch-Norman Integration
       Method and its Implementation in Maple, Proceedings of
       ISSAC'89, ACM Press, 212-217.

    .. [3] J. H. Davenport, On the Parallel Risch Algorithm (I),
       Proceedings of EUROCAM'82, LNCS 144, Springer, 144-157.

    .. [4] J. H. Davenport, On the Parallel Risch Algorithm (III):
       Use of Tangents, SIGSAM Bulletin 16 (1982), 3-6.

    .. [5] J. H. Davenport, B. M. Trager, On the Parallel Risch
       Algorithm (II), ACM Transactions on Mathematical
       Software 11 (1985), 356-362.

    See Also
    ========

    sympy.integrals.integrals.Integral.doit
    sympy.integrals.integrals.Integral
    sympy.integrals.heurisch.components
    """
    f = sympify(f)

    # There are some functions that Heurisch cannot currently handle,
    # so do not even try.
    # Set _try_heurisch=True to skip this check
    if _try_heurisch is not True:
        if f.has(Abs, re, im, sign, Heaviside, DiracDelta, floor, ceiling, arg):
            return

    if not f.has_free(x):
        return f*x

    if not f.is_Add:
        indep, f = f.as_independent(x)
    else:
        indep = S.One

    rewritables = {
        (sin, cos, cot): tan,
        (sinh, cosh, coth): tanh,
    }

    if rewrite:
        for candidates, rule in rewritables.items():
            f = f.rewrite(candidates, rule)
    else:
        for candidates in rewritables.keys():
            if f.has(*candidates):
                break
        else:
            rewrite = True

    terms = components(f, x)
    dcache = DiffCache(x)

    if hints is not None:
        if not hints:
            a = Wild('a', exclude=[x])
            b = Wild('b', exclude=[x])
            c = Wild('c', exclude=[x])

            for g in set(terms):  # using copy of terms
                if g.is_Function:
                    if isinstance(g, li):
                        M = g.args[0].match(a*x**b)

                        if M is not None:
                            terms.add( x*(li(M[a]*x**M[b]) - (M[a]*x**M[b])**(-1/M[b])*Ei((M[b]+1)*log(M[a]*x**M[b])/M[b])) )
                            #terms.add( x*(li(M[a]*x**M[b]) - (x**M[b])**(-1/M[b])*Ei((M[b]+1)*log(M[a]*x**M[b])/M[b])) )
                            #terms.add( x*(li(M[a]*x**M[b]) - x*Ei((M[b]+1)*log(M[a]*x**M[b])/M[b])) )
                            #terms.add( li(M[a]*x**M[b]) - Ei((M[b]+1)*log(M[a]*x**M[b])/M[b]) )

                    elif isinstance(g, exp):
                        M = g.args[0].match(a*x**2)

                        if M is not None:
                            if M[a].is_positive:
                                terms.add(erfi(sqrt(M[a])*x))
                            else: # M[a].is_negative or unknown
                                terms.add(erf(sqrt(-M[a])*x))

                        M = g.args[0].match(a*x**2 + b*x + c)

                        if M is not None:
                            if M[a].is_positive:
                                terms.add(sqrt(pi/4*(-M[a]))*exp(M[c] - M[b]**2/(4*M[a]))*
                                          erfi(sqrt(M[a])*x + M[b]/(2*sqrt(M[a]))))
                            elif M[a].is_negative:
                                terms.add(sqrt(pi/4*(-M[a]))*exp(M[c] - M[b]**2/(4*M[a]))*
                                          erf(sqrt(-M[a])*x - M[b]/(2*sqrt(-M[a]))))

                        M = g.args[0].match(a*log(x)**2)

                        if M is not None:
                            if M[a].is_positive:
                                terms.add(erfi(sqrt(M[a])*log(x) + 1/(2*sqrt(M[a]))))
                            if M[a].is_negative:
                                terms.add(erf(sqrt(-M[a])*log(x) - 1/(2*sqrt(-M[a]))))

                elif g.is_Pow:
                    if g.exp.is_Rational and g.exp.q == 2:
                        M = g.base.match(a*x**2 + b)

                        if M is not None and M[b].is_positive:
                            if M[a].is_positive:
                                terms.add(asinh(sqrt(M[a]/M[b])*x))
                            elif M[a].is_negative:
                                terms.add(asin(sqrt(-M[a]/M[b])*x))

                        M = g.base.match(a*x**2 - b)

                        if M is not None and M[b].is_positive:
                            if M[a].is_positive:
                                dF = 1/sqrt(M[a]*x**2 - M[b])
                                F = log(2*sqrt(M[a])*sqrt(M[a]*x**2 - M[b]) + 2*M[a]*x)/sqrt(M[a])
                                dcache.cache[F] = dF  # hack: F.diff(x) doesn't automatically simplify to f
                                terms.add(F)
                            elif M[a].is_negative:
                                terms.add(-M[b]/2*sqrt(-M[a])*
                                           atan(sqrt(-M[a])*x/sqrt(M[a]*x**2 - M[b])))

        else:
            terms |= set(hints)

    for g in set(terms):  # using copy of terms
        terms |= components(dcache.get_diff(g), x)

    # XXX: The commented line below makes heurisch more deterministic wrt
    # PYTHONHASHSEED and the iteration order of sets. There are other places
    # where sets are iterated over but this one is possibly the most important.
    # Theoretically the order here should not matter but different orderings
    # can expose potential bugs in the different code paths so potentially it
    # is better to keep the non-determinism.
    #
    # terms = list(ordered(terms))

    # TODO: caching is significant factor for why permutations work at all. Change this.
    V = _symbols('x', len(terms))


    # sort mapping expressions from largest to smallest (last is always x).
    mapping = list(reversed(list(zip(*ordered(                          #
        [(a[0].as_independent(x)[1], a) for a in zip(terms, V)])))[1])) #
    rev_mapping = {v: k for k, v in mapping}                            #
    if mappings is None:                                                #
        # optimizing the number of permutations of mapping              #
        assert mapping[-1][0] == x # if not, find it and correct this comment
        unnecessary_permutations = [mapping.pop(-1)]
        # permute types of objects
        types = defaultdict(list)
        for i in mapping:
            e, _ = i
            types[type(e)].append(i)
        mapping = [types[i] for i in types]
        def _iter_mappings():
            for i in permutations(mapping):
                # make the expression of a given type be ordered
                yield [j for i in i for j in ordered(i)]
        mappings = _iter_mappings()
    else:
        unnecessary_permutations = unnecessary_permutations or []

    def _substitute(expr):
        return expr.subs(mapping)

    for mapping in mappings:
        mapping = list(mapping)
        mapping = mapping + unnecessary_permutations
        diffs = [ _substitute(dcache.get_diff(g)) for g in terms ]
        denoms = [ g.as_numer_denom()[1] for g in diffs ]
        if all(h.is_polynomial(*V) for h in denoms) and _substitute(f).is_rational_function(*V):
            denom = reduce(lambda p, q: lcm(p, q, *V), denoms)
            break
    else:
        if not rewrite:
            result = heurisch(f, x, rewrite=True, hints=hints,
                unnecessary_permutations=unnecessary_permutations)

            if result is not None:
                return indep*result
        return None

    numers = [ cancel(denom*g) for g in diffs ]
    def _derivation(h):
        return Add(*[ d * h.diff(v) for d, v in zip(numers, V) ])

    def _deflation(p):
        for y in V:
            if not p.has(y):
                continue

            if _derivation(p) is not S.Zero:
                c, q = p.as_poly(y).primitive()
                return _deflation(c)*gcd(q, q.diff(y)).as_expr()

        return p

    def _splitter(p):
        for y in V:
            if not p.has(y):
                continue

            if _derivation(y) is not S.Zero:
                c, q = p.as_poly(y).primitive()

                q = q.as_expr()

                h = gcd(q, _derivation(q), y)
                s = quo(h, gcd(q, q.diff(y), y), y)

                c_split = _splitter(c)

                if s.as_poly(y).degree() == 0:
                    return (c_split[0], q * c_split[1])

                q_split = _splitter(cancel(q / s))

                return (c_split[0]*q_split[0]*s, c_split[1]*q_split[1])

        return (S.One, p)

    special = {}

    for term in terms:
        if term.is_Function:
            if isinstance(term, tan):
                special[1 + _substitute(term)**2] = False
            elif isinstance(term, tanh):
                special[1 + _substitute(term)] = False
                special[1 - _substitute(term)] = False
            elif isinstance(term, LambertW):
                special[_substitute(term)] = True

    F = _substitute(f)

    P, Q = F.as_numer_denom()

    u_split = _splitter(denom)
    v_split = _splitter(Q)

    polys = set(list(v_split) + [ u_split[0] ] + list(special.keys()))

    s = u_split[0] * Mul(*[ k for k, v in special.items() if v ])
    polified = [ p.as_poly(*V) for p in [s, P, Q] ]

    if None in polified:
        return None

    #--- definitions for _integrate
    a, b, c = [ p.total_degree() for p in polified ]

    poly_denom = (s * v_split[0] * _deflation(v_split[1])).as_expr()

    def _exponent(g):
        if g.is_Pow:
            if g.exp.is_Rational and g.exp.q != 1:
                if g.exp.p > 0:
                    return g.exp.p + g.exp.q - 1
                else:
                    return abs(g.exp.p + g.exp.q)
            else:
                return 1
        elif not g.is_Atom and g.args:
            return max(_exponent(h) for h in g.args)
        else:
            return 1

    A, B = _exponent(f), a + max(b, c)

    if A > 1 and B > 1:
        monoms = tuple(ordered(itermonomials(V, A + B - 1 + degree_offset)))
    else:
        monoms = tuple(ordered(itermonomials(V, A + B + degree_offset)))

    poly_coeffs = _symbols('A', len(monoms))

    poly_part = Add(*[ poly_coeffs[i]*monomial
        for i, monomial in enumerate(monoms) ])

    reducibles = set()

    for poly in ordered(polys):
        coeff, factors = factor_list(poly, *V)
        reducibles.add(coeff)
        reducibles.update(fact for fact, mul in factors)

    def _integrate(field=None):
        atans = set()
        pairs = set()

        if field == 'Q':
            irreducibles = set(reducibles)
        else:
            setV = set(V)
            irreducibles = set()
            for poly in ordered(reducibles):
                zV = setV & set(iterfreeargs(poly))
                for z in ordered(zV):
                    s = set(root_factors(poly, z, filter=field))
                    irreducibles |= s
                    break

        log_part, atan_part = [], []

        for poly in ordered(irreducibles):
            m = collect(poly, I, evaluate=False)
            y = m.get(I, S.Zero)
            if y:
                x = m.get(S.One, S.Zero)
                if x.has(I) or y.has(I):
                    continue  # nontrivial x + I*y
                pairs.add((x, y))
                irreducibles.remove(poly)

        while pairs:
            x, y = pairs.pop()
            if (x, -y) in pairs:
                pairs.remove((x, -y))
                # Choosing b with no minus sign
                if y.could_extract_minus_sign():
                    y = -y
                irreducibles.add(x*x + y*y)
                atans.add(atan(x/y))
            else:
                irreducibles.add(x + I*y)


        B = _symbols('B', len(irreducibles))
        C = _symbols('C', len(atans))

        # Note: the ordering matters here
        for poly, b in reversed(list(zip(ordered(irreducibles), B))):
            if poly.has(*V):
                poly_coeffs.append(b)
                log_part.append(b * log(poly))

        for poly, c in reversed(list(zip(ordered(atans), C))):
            if poly.has(*V):
                poly_coeffs.append(c)
                atan_part.append(c * poly)

        # TODO: Currently it's better to use symbolic expressions here instead
        # of rational functions, because it's simpler and FracElement doesn't
        # give big speed improvement yet. This is because cancellation is slow
        # due to slow polynomial GCD algorithms. If this gets improved then
        # revise this code.
        candidate = poly_part/poly_denom + Add(*log_part) + Add(*atan_part)
        h = F - _derivation(candidate) / denom
        raw_numer = h.as_numer_denom()[0]

        # Rewrite raw_numer as a polynomial in K[coeffs][V] where K is a field
        # that we have to determine. We can't use simply atoms() because log(3),
        # sqrt(y) and similar expressions can appear, leading to non-trivial
        # domains.
        syms = set(poly_coeffs) | set(V)
        non_syms = set()

        def find_non_syms(expr):
            if expr.is_Integer or expr.is_Rational:
                pass # ignore trivial numbers
            elif expr in syms:
                pass # ignore variables
            elif not expr.has_free(*syms):
                non_syms.add(expr)
            elif expr.is_Add or expr.is_Mul or expr.is_Pow:
                list(map(find_non_syms, expr.args))
            else:
                # TODO: Non-polynomial expression. This should have been
                # filtered out at an earlier stage.
                raise PolynomialError

        try:
            find_non_syms(raw_numer)
        except PolynomialError:
            return None
        else:
            ground, _ = construct_domain(non_syms, field=True)

        coeff_ring = PolyRing(poly_coeffs, ground)
        ring = PolyRing(V, coeff_ring)
        try:
            numer = ring.from_expr(raw_numer)
        except ValueError:
            raise PolynomialError
        solution = solve_lin_sys(numer.coeffs(), coeff_ring, _raw=False)

        if solution is None:
            return None
        else:
            return candidate.xreplace(solution).xreplace(
                dict(zip(poly_coeffs, [S.Zero]*len(poly_coeffs))))

    if all(isinstance(_, Symbol) for _ in V):
        more_free = F.free_symbols - set(V)
    else:
        Fd = F.as_dummy()
        more_free = Fd.xreplace(dict(zip(V, (Dummy() for _ in V)))
            ).free_symbols & Fd.free_symbols
    if not more_free:
        # all free generators are identified in V
        solution = _integrate('Q')

        if solution is None:
            solution = _integrate()
    else:
        solution = _integrate()

    if solution is not None:
        antideriv = solution.subs(rev_mapping)
        antideriv = cancel(antideriv).expand()

        if antideriv.is_Add:
            antideriv = antideriv.as_independent(x)[1]

        return indep*antideriv
    else:
        if retries >= 0:
            result = heurisch(f, x, mappings=mappings, rewrite=rewrite, hints=hints, retries=retries - 1, unnecessary_permutations=unnecessary_permutations)

            if result is not None:
                return indep*result

        return None
<|endoftext|>"""Integration method that emulates by-hand techniques.

This module also provides functionality to get the steps used to evaluate a
particular integral, in the ``integral_steps`` function. This will return
nested ``Rule`` s representing the integration rules used.

Each ``Rule`` class represents a (maybe parametrized) integration rule, e.g.
``SinRule`` for integrating ``sin(x)`` and ``ReciprocalSqrtQuadraticRule``
for integrating ``1/sqrt(a+b*x+c*x**2)``. The ``eval`` method returns the
integration result.

The ``manualintegrate`` function computes the integral by calling ``eval``
on the rule returned by ``integral_steps``.

The integrator can be extended with new heuristics and evaluation
techniques. To do so, extend the ``Rule`` class, implement ``eval`` method,
then write a function that accepts an ``IntegralInfo`` object and returns
either a ``Rule`` instance or ``None``. If the new technique requires a new
match, add the key and call to the antiderivative function to integral_steps.
To enable simple substitutions, add the match to find_substitutions.

"""

from __future__ import annotations
from typing import NamedTuple, Type, Callable, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Number, E
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne, Boolean
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction, csch,
    cosh, coth, sech, sinh, tanh, asinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (TrigonometricFunction,
    cos, sin, tan, cot, csc, sec, acos, asin, atan, acot, acsc, asec)
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.error_functions import (erf, erfi, fresnelc,
    fresnels, Ci, Chi, Si, Shi, Ei, li)
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.elliptic_integrals import elliptic_e, elliptic_f
from sympy.functions.special.polynomials import (chebyshevt, chebyshevu,
    legendre, hermite, laguerre, assoc_laguerre, gegenbauer, jacobi,
    OrthogonalPolynomial)
from sympy.functions.special.zeta_functions import polylog
from .integrals import Integral
from sympy.logic.boolalg import And
from sympy.ntheory.factor_ import primefactors
from sympy.polys.polytools import degree, lcm_list, gcd_list, Poly
from sympy.simplify.radsimp import fraction
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.strategies.core import switch, do_one, null_safe, condition
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug


@dataclass
class Rule(ABC):
    integrand: Expr
    variable: Symbol

    @abstractmethod
    def eval(self) -> Expr:
        pass

    @abstractmethod
    def contains_dont_know(self) -> bool:
        pass


@dataclass
class AtomicRule(Rule, ABC):
    """A simple rule that does not depend on other rules"""
    def contains_dont_know(self) -> bool:
        return False


@dataclass
class ConstantRule(AtomicRule):
    """integrate(a, x)  ->  a*x"""
    def eval(self) -> Expr:
        return self.integrand * self.variable


@dataclass
class ConstantTimesRule(Rule):
    """integrate(a*f(x), x)  ->  a*integrate(f(x), x)"""
    constant: Expr
    other: Expr
    substep: Rule

    def eval(self) -> Expr:
        return self.constant * self.substep.eval()

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class PowerRule(AtomicRule):
    """integrate(x**a, x)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        return Piecewise(
            ((self.base**(self.exp + 1))/(self.exp + 1), Ne(self.exp, -1)),
            (log(self.base), True),
        )


@dataclass
class NestedPowRule(AtomicRule):
    """integrate((x**a)**b, x)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        m = self.base * self.integrand
        return Piecewise((m / (self.exp + 1), Ne(self.exp, -1)),
                         (m * log(self.base), True))


@dataclass
class AddRule(Rule):
    """integrate(f(x) + g(x), x) -> integrate(f(x), x) + integrate(g(x), x)"""
    substeps: list[Rule]

    def eval(self) -> Expr:
        return Add(*(substep.eval() for substep in self.substeps))

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.substeps)


@dataclass
class URule(Rule):
    """integrate(f(g(x))*g'(x), x) -> integrate(f(u), u), u = g(x)"""
    u_var: Symbol
    u_func: Expr
    substep: Rule

    def eval(self) -> Expr:
        result = self.substep.eval()
        if self.u_func.is_Pow:
            base, exp_ = self.u_func.as_base_exp()
            if exp_ == -1:
                # avoid needless -log(1/x) from substitution
                result = result.subs(log(self.u_var), -log(base))
        return result.subs(self.u_var, self.u_func)

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class PartsRule(Rule):
    """integrate(u(x)*v'(x), x) -> u(x)*v(x) - integrate(u'(x)*v(x), x)"""
    u: Symbol
    dv: Expr
    v_step: Rule
    second_step: Rule | None  # None when is a substep of CyclicPartsRule

    def eval(self) -> Expr:
        assert self.second_step is not None
        v = self.v_step.eval()
        return self.u * v - self.second_step.eval()

    def contains_dont_know(self) -> bool:
        return self.v_step.contains_dont_know() or (
            self.second_step is not None and self.second_step.contains_dont_know())


@dataclass
class CyclicPartsRule(Rule):
    """Apply PartsRule multiple times to integrate exp(x)*sin(x)"""
    parts_rules: list[PartsRule]
    coefficient: Expr

    def eval(self) -> Expr:
        result = []
        sign = 1
        for rule in self.parts_rules:
            result.append(sign * rule.u * rule.v_step.eval())
            sign *= -1
        return Add(*result) / (1 - self.coefficient)

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.parts_rules)


@dataclass
class TrigRule(AtomicRule, ABC):
    pass


@dataclass
class SinRule(TrigRule):
    """integrate(sin(x), x) -> -cos(x)"""
    def eval(self) -> Expr:
        return -cos(self.variable)


@dataclass
class CosRule(TrigRule):
    """integrate(cos(x), x) -> sin(x)"""
    def eval(self) -> Expr:
        return sin(self.variable)


@dataclass
class SecTanRule(TrigRule):
    """integrate(sec(x)*tan(x), x) -> sec(x)"""
    def eval(self) -> Expr:
        return sec(self.variable)


@dataclass
class CscCotRule(TrigRule):
    """integrate(csc(x)*cot(x), x) -> -csc(x)"""
    def eval(self) -> Expr:
        return -csc(self.variable)


@dataclass
class Sec2Rule(TrigRule):
    """integrate(sec(x)**2, x) -> tan(x)"""
    def eval(self) -> Expr:
        return tan(self.variable)


@dataclass
class Csc2Rule(TrigRule):
    """integrate(csc(x)**2, x) -> -cot(x)"""
    def eval(self) -> Expr:
        return -cot(self.variable)


@dataclass
class HyperbolicRule(AtomicRule, ABC):
    pass


@dataclass
class SinhRule(HyperbolicRule):
    """integrate(sinh(x), x) -> cosh(x)"""
    def eval(self) -> Expr:
        return cosh(self.variable)


@dataclass
class CoshRule(HyperbolicRule):
    """integrate(cosh(x), x) -> sinh(x)"""
    def eval(self):
        return sinh(self.variable)


@dataclass
class ExpRule(AtomicRule):
    """integrate(a**x, x) -> a**x/ln(a)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        return self.integrand / log(self.base)


@dataclass
class ReciprocalRule(AtomicRule):
    """integrate(1/x, x) -> ln(x)"""
    base: Expr

    def eval(self) -> Expr:
        return log(self.base)


@dataclass
class ArcsinRule(AtomicRule):
    """integrate(1/sqrt(1-x**2), x) -> asin(x)"""
    def eval(self) -> Expr:
        return asin(self.variable)


@dataclass
class ArcsinhRule(AtomicRule):
    """integrate(1/sqrt(1+x**2), x) -> asin(x)"""
    def eval(self) -> Expr:
        return asinh(self.variable)


@dataclass
class ReciprocalSqrtQuadraticRule(AtomicRule):
    """integrate(1/sqrt(a+b*x+c*x**2), x) -> log(2*sqrt(c)*sqrt(a+b*x+c*x**2)+b+2*c*x)/sqrt(c)"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return log(2*sqrt(c)*sqrt(a+b*x+c*x**2)+b+2*c*x)/sqrt(c)


@dataclass
class SqrtQuadraticDenomRule(AtomicRule):
    """integrate(poly(x)/sqrt(a+b*x+c*x**2), x)"""
    a: Expr
    b: Expr
    c: Expr
    coeffs: list[Expr]

    def eval(self) -> Expr:
        a, b, c, coeffs, x = self.a, self.b, self.c, self.coeffs.copy(), self.variable
        # Integrate poly/sqrt(a+b*x+c*x**2) using recursion.
        # coeffs are coefficients of the polynomial.
        # Let I_n = x**n/sqrt(a+b*x+c*x**2), then
        # I_n = A * x**(n-1)*sqrt(a+b*x+c*x**2) - B * I_{n-1} - C * I_{n-2}
        # where A = 1/(n*c), B = (2*n-1)*b/(2*n*c), C = (n-1)*a/(n*c)
        # See https://github.com/sympy/sympy/pull/23608 for proof.
        result_coeffs = []
        coeffs = coeffs.copy()
        for i in range(len(coeffs)-2):
            n = len(coeffs)-1-i
            coeff = coeffs[i]/(c*n)
            result_coeffs.append(coeff)
            coeffs[i+1] -= (2*n-1)*b/2*coeff
            coeffs[i+2] -= (n-1)*a*coeff
        d, e = coeffs[-1], coeffs[-2]
        s = sqrt(a+b*x+c*x**2)
        constant = d-b*e/(2*c)
        if constant == 0:
            I0 = 0
        else:
            step = inverse_trig_rule(IntegralInfo(1/s, x), degenerate=False)
            I0 = constant*step.eval()
        return Add(*(result_coeffs[i]*x**(len(coeffs)-2-i)
                     for i in range(len(result_coeffs))), e/c)*s + I0


@dataclass
class SqrtQuadraticRule(AtomicRule):
    """integrate(sqrt(a+b*x+c*x**2), x)"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        step = sqrt_quadratic_rule(IntegralInfo(self.integrand, self.variable), degenerate=False)
        return step.eval()


@dataclass
class AlternativeRule(Rule):
    """Multiple ways to do integration."""
    alternatives: list[Rule]

    def eval(self) -> Expr:
        return self.alternatives[0].eval()

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.alternatives)


@dataclass
class DontKnowRule(Rule):
    """Leave the integral as is."""
    def eval(self) -> Expr:
        return Integral(self.integrand, self.variable)

    def contains_dont_know(self) -> bool:
        return True


@dataclass
class DerivativeRule(AtomicRule):
    """integrate(f'(x), x) -> f(x)"""
    def eval(self) -> Expr:
        assert isinstance(self.integrand, Derivative)
        variable_count = list(self.integrand.variable_count)
        for i, (var, count) in enumerate(variable_count):
            if var == self.variable:
                variable_count[i] = (var, count - 1)
                break
        return Derivative(self.integrand.expr, *variable_count)


@dataclass
class RewriteRule(Rule):
    """Rewrite integrand to another form that is easier to handle."""
    rewritten: Expr
    substep: Rule

    def eval(self) -> Expr:
        return self.substep.eval()

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class CompleteSquareRule(RewriteRule):
    """Rewrite a+b*x+c*x**2 to a-b**2/(4*c) + c*(x+b/(2*c))**2"""
    pass


@dataclass
class PiecewiseRule(Rule):
    subfunctions: Sequence[tuple[Rule, bool | Boolean]]

    def eval(self) -> Expr:
        return Piecewise(*[(substep.eval(), cond)
                           for substep, cond in self.subfunctions])

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep, _ in self.subfunctions)


@dataclass
class HeavisideRule(Rule):
    harg: Expr
    ibnd: Expr
    substep: Rule

    def eval(self) -> Expr:
        # If we are integrating over x and the integrand has the form
        #       Heaviside(m*x+b)*g(x) == Heaviside(harg)*g(symbol)
        # then there needs to be continuity at -b/m == ibnd,
        # so we subtract the appropriate term.
        result = self.substep.eval()
        return Heaviside(self.harg) * (result - result.subs(self.variable, self.ibnd))

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class DiracDeltaRule(AtomicRule):
    n: Expr
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        n, a, b, x = self.n, self.a, self.b, self.variable
        if n == 0:
            return Heaviside(a+b*x)/b
        return DiracDelta(a+b*x, n-1)/b


@dataclass
class TrigSubstitutionRule(Rule):
    theta: Expr
    func: Expr
    rewritten: Expr
    substep: Rule
    restriction: bool | Boolean

    def eval(self) -> Expr:
        theta, func, x = self.theta, self.func, self.variable
        func = func.subs(sec(theta), 1/cos(theta))
        func = func.subs(csc(theta), 1/sin(theta))
        func = func.subs(cot(theta), 1/tan(theta))

        trig_function = list(func.find(TrigonometricFunction))
        assert len(trig_function) == 1
        trig_function = trig_function[0]
        relation = solve(x - func, trig_function)
        assert len(relation) == 1
        numer, denom = fraction(relation[0])

        if isinstance(trig_function, sin):
            opposite = numer
            hypotenuse = denom
            adjacent = sqrt(denom**2 - numer**2)
            inverse = asin(relation[0])
        elif isinstance(trig_function, cos):
            adjacent = numer
            hypotenuse = denom
            opposite = sqrt(denom**2 - numer**2)
            inverse = acos(relation[0])
        else:  # tan
            opposite = numer
            adjacent = denom
            hypotenuse = sqrt(denom**2 + numer**2)
            inverse = atan(relation[0])

        substitution = [
            (sin(theta), opposite/hypotenuse),
            (cos(theta), adjacent/hypotenuse),
            (tan(theta), opposite/adjacent),
            (theta, inverse)
        ]
        return Piecewise(
                (self.substep.eval().subs(substitution).trigsimp(), self.restriction) # type: ignore
        )

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass
class ArctanRule(AtomicRule):
    """integrate(a/(b*x**2+c), x) -> a/b / sqrt(c/b) * atan(x/sqrt(c/b))"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return a/b / sqrt(c/b) * atan(x/sqrt(c/b))


@dataclass
class OrthogonalPolyRule(AtomicRule, ABC):
    n: Expr


@dataclass
class JacobiRule(OrthogonalPolyRule):
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        n, a, b, x = self.n, self.a, self.b, self.variable
        return Piecewise(
            (2*jacobi(n + 1, a - 1, b - 1, x)/(n + a + b), Ne(n + a + b, 0)),
            (x, Eq(n, 0)),
            ((a + b + 2)*x**2/4 + (a - b)*x/2, Eq(n, 1)))


@dataclass
class GegenbauerRule(OrthogonalPolyRule):
    a: Expr

    def eval(self) -> Expr:
        n, a, x = self.n, self.a, self.variable
        return Piecewise(
            (gegenbauer(n + 1, a - 1, x)/(2*(a - 1)), Ne(a, 1)),
            (chebyshevt(n + 1, x)/(n + 1), Ne(n, -1)),
            (S.Zero, True))


@dataclass
class ChebyshevTRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return Piecewise(
            ((chebyshevt(n + 1, x)/(n + 1) -
              chebyshevt(n - 1, x)/(n - 1))/2, Ne(Abs(n), 1)),
            (x**2/2, True))


@dataclass
class ChebyshevURule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return Piecewise(
            (chebyshevt(n + 1, x)/(n + 1), Ne(n, -1)),
            (S.Zero, True))


@dataclass
class LegendreRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return(legendre(n + 1, x) - legendre(n - 1, x))/(2*n + 1)


@dataclass
class HermiteRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return hermite(n + 1, x)/(2*(n + 1))


@dataclass
class LaguerreRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return laguerre(n, x) - laguerre(n + 1, x)


@dataclass
class AssocLaguerreRule(OrthogonalPolyRule):
    a: Expr

    def eval(self) -> Expr:
        return -assoc_laguerre(self.n + 1, self.a - 1, self.variable)


@dataclass
class IRule(AtomicRule, ABC):
    a: Expr
    b: Expr


@dataclass
class CiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return cos(b)*Ci(a*x) - sin(b)*Si(a*x)


@dataclass
class ChiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return cosh(b)*Chi(a*x) + sinh(b)*Shi(a*x)


@dataclass
class EiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return exp(b)*Ei(a*x)


@dataclass
class SiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return sin(b)*Ci(a*x) + cos(b)*Si(a*x)


@dataclass
class ShiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return sinh(b)*Chi(a*x) + cosh(b)*Shi(a*x)


@dataclass
class LiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return li(a*x + b)/a


@dataclass
class ErfRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        if a.is_extended_real:
            return Piecewise(
                (sqrt(S.Pi)/sqrt(-a)/2 * exp(c - b**2/(4*a)) *
                    erf((-2*a*x - b)/(2*sqrt(-a))), a < 0),
                (sqrt(S.Pi)/sqrt(a)/2 * exp(c - b**2/(4*a)) *
                    erfi((2*a*x + b)/(2*sqrt(a))), True))
        return sqrt(S.Pi)/sqrt(a)/2 * exp(c - b**2/(4*a)) * \
                erfi((2*a*x + b)/(2*sqrt(a)))


@dataclass
class FresnelCRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return sqrt(S.Pi)/sqrt(2*a) * (
            cos(b**2/(4*a) - c)*fresnelc((2*a*x + b)/sqrt(2*a*S.Pi)) +
            sin(b**2/(4*a) - c)*fresnels((2*a*x + b)/sqrt(2*a*S.Pi)))


@dataclass
class FresnelSRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return sqrt(S.Pi)/sqrt(2*a) * (
            cos(b**2/(4*a) - c)*fresnels((2*a*x + b)/sqrt(2*a*S.Pi)) -
            sin(b**2/(4*a) - c)*fresnelc((2*a*x + b)/sqrt(2*a*S.Pi)))


@dataclass
class PolylogRule(AtomicRule):
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        return polylog(self.b + 1, self.a * self.variable)


@dataclass
class UpperGammaRule(AtomicRule):
    a: Expr
    e: Expr

    def eval(self) -> Expr:
        a, e, x = self.a, self.e, self.variable
        return x**e * (-a*x)**(-e) * uppergamma(e + 1, -a*x)/a


@dataclass
class EllipticFRule(AtomicRule):
    a: Expr
    d: Expr

    def eval(self) -> Expr:
        return elliptic_f(self.variable, self.d/self.a)/sqrt(self.a)


@dataclass
class EllipticERule(AtomicRule):
    a: Expr
    d: Expr

    def eval(self) -> Expr:
        return elliptic_e(self.variable, self.d/self.a)*sqrt(self.a)


class IntegralInfo(NamedTuple):
    integrand: Expr
    symbol: Symbol


def manual_diff(f, symbol):
    """Derivative of f in form expected by find_substitutions

    SymPy's derivatives for some trig functions (like cot) are not in a form
    that works well with finding substitutions; this replaces the
    derivatives for those particular forms with something that works better.

    """
    if f.args:
        arg = f.args[0]
        if isinstance(f, tan):
            return arg.diff(symbol) * sec(arg)**2
        elif isinstance(f, cot):
            return -arg.diff(symbol) * csc(arg)**2
        elif isinstance(f, sec):
            return arg.diff(symbol) * sec(arg) * tan(arg)
        elif isinstance(f, csc):
            return -arg.diff(symbol) * csc(arg) * cot(arg)
        elif isinstance(f, Add):
            return sum(manual_diff(arg, symbol) for arg in f.args)
        elif isinstance(f, Mul):
            if len(f.args) == 2 and isinstance(f.args[0], Number):
                return f.args[0] * manual_diff(f.args[1], symbol)
    return f.diff(symbol)

def manual_subs(expr, *args):
    """
    A wrapper for `expr.subs(*args)` with additional logic for substitution
    of invertible functions.
    """
    if len(args) == 1:
        sequence = args[0]
        if isinstance(sequence, (Dict, Mapping)):
            sequence = sequence.items()
        elif not iterable(sequence):
            raise ValueError("Expected an iterable of (old, new) pairs")
    elif len(args) == 2:
        sequence = [args]
    else:
        raise ValueError("subs accepts either 1 or 2 arguments")

    new_subs = []
    for old, new in sequence:
        if isinstance(old, log):
            # If log(x) = y, then exp(a*log(x)) = exp(a*y)
            # that is, x**a = exp(a*y). Replace nontrivial powers of x
            # before subs turns them into `exp(y)**a`, but
            # do not replace x itself yet, to avoid `log(exp(y))`.
            x0 = old.args[0]
            expr = expr.replace(lambda x: x.is_Pow and x.base == x0,
                lambda x: exp(x.exp*new))
            new_subs.append((x0, exp(new)))

    return expr.subs(list(sequence) + new_subs)

# Method based on that on SIN, described in "Symbolic Integration: The
# Stormy Decade"

inverse_trig_functions = (atan, asin, acos, acot, acsc, asec)


def find_substitutions(integrand, symbol, u_var):
    results = []

    def test_subterm(u, u_diff):
        if u_diff == 0:
            return False
        substituted = integrand / u_diff
        debug("substituted: {}, u: {}, u_var: {}".format(substituted, u, u_var))
        substituted = manual_subs(substituted, u, u_var).cancel()

        if substituted.has_free(symbol):
            return False
        # avoid increasing the degree of a rational function
        if integrand.is_rational_function(symbol) and substituted.is_rational_function(u_var):
            deg_before = max(degree(t, symbol) for t in integrand.as_numer_denom())
            deg_after = max(degree(t, u_var) for t in substituted.as_numer_denom())
            if deg_after > deg_before:
                return False
        return substituted.as_independent(u_var, as_Add=False)

    def exp_subterms(term: Expr):
        linear_coeffs = []
        terms = []
        n = Wild('n', properties=[lambda n: n.is_Integer])
        for exp_ in term.find(exp):
            arg = exp_.args[0]
            if symbol not in arg.free_symbols:
                continue
            match = arg.match(n*symbol)
            if match:
                linear_coeffs.append(match[n])
            else:
                terms.append(exp_)
        if linear_coeffs:
            terms.append(exp(gcd_list(linear_coeffs)*symbol))
        return terms

    def possible_subterms(term):
        if isinstance(term, (TrigonometricFunction, HyperbolicFunction,
                             *inverse_trig_functions,
                             exp, log, Heaviside)):
            return [term.args[0]]
        elif isinstance(term, (chebyshevt, chebyshevu,
                        legendre, hermite, laguerre)):
            return [term.args[1]]
        elif isinstance(term, (gegenbauer, assoc_laguerre)):
            return [term.args[2]]
        elif isinstance(term, jacobi):
            return [term.args[3]]
        elif isinstance(term, Mul):
            r = []
            for u in term.args:
                r.append(u)
                r.extend(possible_subterms(u))
            return r
        elif isinstance(term, Pow):
            r = [arg for arg in term.args if arg.has(symbol)]
            if term.exp.is_Integer:
                r.extend([term.base**d for d in primefactors(term.exp)
                    if 1 < d < abs(term.args[1])])
                if term.base.is_Add:
                    r.extend([t for t in possible_subterms(term.base)
                        if t.is_Pow])
            return r
        elif isinstance(term, Add):
            r = []
            for arg in term.args:
                r.append(arg)
                r.extend(possible_subterms(arg))
            return r
        return []

    for u in list(dict.fromkeys(possible_subterms(integrand) + exp_subterms(integrand))):
        if u == symbol:
            continue
        u_diff = manual_diff(u, symbol)
        new_integrand = test_subterm(u, u_diff)
        if new_integrand is not False:
            constant, new_integrand = new_integrand
            if new_integrand == integrand.subs(symbol, u_var):
                continue
            substitution = (u, constant, new_integrand)
            if substitution not in results:
                results.append(substitution)

    return results

def rewriter(condition, rewrite):
    """Strategy that rewrites an integrand."""
    def _rewriter(integral):
        integrand, symbol = integral
        debug("Integral: {} is rewritten with {} on symbol: {}".format(integrand, rewrite, symbol))
        if condition(*integral):
            rewritten = rewrite(*integral)
            if rewritten != integrand:
                substep = integral_steps(rewritten, symbol)
                if not isinstance(substep, DontKnowRule) and substep:
                    return RewriteRule(integrand, symbol, rewritten, substep)
    return _rewriter

def proxy_rewriter(condition, rewrite):
    """Strategy that rewrites an integrand based on some other criteria."""
    def _proxy_rewriter(criteria):
        criteria, integral = criteria
        integrand, symbol = integral
        debug("Integral: {} is rewritten with {} on symbol: {} and criteria: {}".format(integrand, rewrite, symbol, criteria))
        args = criteria + list(integral)
        if condition(*args):
            rewritten = rewrite(*args)
            if rewritten != integrand:
                return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))
    return _proxy_rewriter

def multiplexer(conditions):
    """Apply the rule that matches the condition, else None"""
    def multiplexer_rl(expr):
        for key, rule in conditions.items():
            if key(expr):
                return rule(expr)
    return multiplexer_rl

def alternatives(*rules):
    """Strategy that makes an AlternativeRule out of multiple possible results."""
    def _alternatives(integral):
        alts = []
        count = 0
        debug("List of Alternative Rules")
        for rule in rules:
            count = count + 1
            debug("Rule {}: {}".format(count, rule))

            result = rule(integral)
            if (result and not isinstance(result, DontKnowRule) and
                result != integral and result not in alts):
                alts.append(result)
        if len(alts) == 1:
            return alts[0]
        elif alts:
            doable = [rule for rule in alts if not rule.contains_dont_know()]
            if doable:
                return AlternativeRule(*integral, doable)
            else:
                return AlternativeRule(*integral, alts)
    return _alternatives

def constant_rule(integral):
    return ConstantRule(*integral)

def power_rule(integral):
    integrand, symbol = integral
    base, expt = integrand.as_base_exp()

    if symbol not in expt.free_symbols and isinstance(base, Symbol):
        if simplify(expt + 1) == 0:
            return ReciprocalRule(integrand, symbol, base)
        return PowerRule(integrand, symbol, base, expt)
    elif symbol not in base.free_symbols and isinstance(expt, Symbol):
        rule = ExpRule(integrand, symbol, base, expt)

        if fuzzy_not(log(base).is_zero):
            return rule
        elif log(base).is_zero:
            return ConstantRule(1, symbol)

        return PiecewiseRule(integrand, symbol, [
            (rule, Ne(log(base), 0)),
            (ConstantRule(1, symbol), True)
        ])

def exp_rule(integral):
    integrand, symbol = integral
    if isinstance(integrand.args[0], Symbol):
        return ExpRule(integrand, symbol, E, integrand.args[0])


def orthogonal_poly_rule(integral):
    orthogonal_poly_classes = {
        jacobi: JacobiRule,
        gegenbauer: GegenbauerRule,
        chebyshevt: ChebyshevTRule,
        chebyshevu: ChebyshevURule,
        legendre: LegendreRule,
        hermite: HermiteRule,
        laguerre: LaguerreRule,
        assoc_laguerre: AssocLaguerreRule
        }
    orthogonal_poly_var_index = {
        jacobi: 3,
        gegenbauer: 2,
        assoc_laguerre: 2
        }
    integrand, symbol = integral
    for klass in orthogonal_poly_classes:
        if isinstance(integrand, klass):
            var_index = orthogonal_poly_var_index.get(klass, 1)
            if (integrand.args[var_index] is symbol and not
                any(v.has(symbol) for v in integrand.args[:var_index])):
                    return orthogonal_poly_classes[klass](integrand, symbol, *integrand.args[:var_index])


_special_function_patterns: list[tuple[Type, Expr, Callable | None, tuple]] = []
_wilds = []
_symbol = Dummy('x')


def special_function_rule(integral):
    integrand, symbol = integral
    if not _special_function_patterns:
        a = Wild('a', exclude=[_symbol], properties=[lambda x: not x.is_zero])
        b = Wild('b', exclude=[_symbol])
        c = Wild('c', exclude=[_symbol])
        d = Wild('d', exclude=[_symbol], properties=[lambda x: not x.is_zero])
        e = Wild('e', exclude=[_symbol], properties=[
            lambda x: not (x.is_nonnegative and x.is_integer)])
        _wilds.extend((a, b, c, d, e))
        # patterns consist of a SymPy class, a wildcard expr, an optional
        # condition coded as a lambda (when Wild properties are not enough),
        # followed by an applicable rule
        linear_pattern = a*_symbol + b
        quadratic_pattern = a*_symbol**2 + b*_symbol + c
        _special_function_patterns.extend((
            (Mul, exp(linear_pattern, evaluate=False)/_symbol, None, EiRule),
            (Mul, cos(linear_pattern, evaluate=False)/_symbol, None, CiRule),
            (Mul, cosh(linear_pattern, evaluate=False)/_symbol, None, ChiRule),
            (Mul, sin(linear_pattern, evaluate=False)/_symbol, None, SiRule),
            (Mul, sinh(linear_pattern, evaluate=False)/_symbol, None, ShiRule),
            (Pow, 1/log(linear_pattern, evaluate=False), None, LiRule),
            (exp, exp(quadratic_pattern, evaluate=False), None, ErfRule),
            (sin, sin(quadratic_pattern, evaluate=False), None, FresnelSRule),
            (cos, cos(quadratic_pattern, evaluate=False), None, FresnelCRule),
            (Mul, _symbol**e*exp(a*_symbol, evaluate=False), None, UpperGammaRule),
            (Mul, polylog(b, a*_symbol, evaluate=False)/_symbol, None, PolylogRule),
            (Pow, 1/sqrt(a - d*sin(_symbol, evaluate=False)**2),
                lambda a, d: a != d, EllipticFRule),
            (Pow, sqrt(a - d*sin(_symbol, evaluate=False)**2),
                lambda a, d: a != d, EllipticERule),
        ))
    _integrand = integrand.subs(symbol, _symbol)
    for type_, pattern, constraint, rule in _special_function_patterns:
        if isinstance(_integrand, type_):
            match = _integrand.match(pattern)
            if match:
                wild_vals = tuple(match.get(w) for w in _wilds
                                  if match.get(w) is not None)
                if constraint is None or constraint(*wild_vals):
                    return rule(integrand, symbol, *wild_vals)


def _add_degenerate_step(generic_cond, generic_step: Rule, degenerate_step: Rule | None) -> Rule:
    if degenerate_step is None:
        return generic_step
    if isinstance(generic_step, PiecewiseRule):
        subfunctions = [(substep, (cond & generic_cond).simplify())
                        for substep, cond in generic_step.subfunctions]
    else:
        subfunctions = [(generic_step, generic_cond)]
    if isinstance(degenerate_step, PiecewiseRule):
        subfunctions += degenerate_step.subfunctions
    else:
        subfunctions.append((degenerate_step, S.true))
    return PiecewiseRule(generic_step.integrand, generic_step.variable, subfunctions)


def nested_pow_rule(integral: IntegralInfo):
    # nested (c*(a+b*x)**d)**e
    integrand, x = integral

    a_ = Wild('a', exclude=[x])
    b_ = Wild('b', exclude=[x, 0])
    pattern = a_+b_*x
    generic_cond = S.true

    class NoMatch(Exception):
        pass

    def _get_base_exp(expr: Expr) -> tuple[Expr, Expr]:
        if not expr.has_free(x):
            return S.One, S.Zero
        if expr.is_Mul:
            _, terms = expr.as_coeff_mul()
            if not terms:
                return S.One, S.Zero
            results = [_get_base_exp(term) for term in terms]
            bases = {b for b, _ in results}
            bases.discard(S.One)
            if len(bases) == 1:
                return bases.pop(), Add(*(e for _, e in results))
            raise NoMatch
        if expr.is_Pow:
            b, e = expr.base, expr.exp  # type: ignore
            if e.has_free(x):
                raise NoMatch
            base_, sub_exp = _get_base_exp(b)
            return base_, sub_exp * e
        match = expr.match(pattern)
        if match:
            a, b = match[a_], match[b_]
            base_ = x + a/b
            nonlocal generic_cond
            generic_cond = Ne(b, 0)
            return base_, S.One
        raise NoMatch

    try:
        base, exp_ = _get_base_exp(integrand)
    except NoMatch:
        return
    if generic_cond is S.true:
        degenerate_step = None
    else:
        # equivalent with subs(b, 0) but no need to find b
        degenerate_step = ConstantRule(integrand.subs(x, 0), x)
    generic_step = NestedPowRule(integrand, x, base, exp_)
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


def inverse_trig_rule(integral: IntegralInfo, degenerate=True):
    """
    Set degenerate=False on recursive call where coefficient of quadratic term
    is assumed non-zero.
    """
    integrand, symbol = integral
    base, exp = integrand.as_base_exp()
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    c = Wild('c', exclude=[symbol, 0])
    match = base.match(a + b*symbol + c*symbol**2)

    if not match:
        return

    def make_inverse_trig(RuleClass, a, sign_a, c, sign_c, h) -> Rule:
        u_var = Dummy("u")
        rewritten = 1/sqrt(sign_a*a + sign_c*c*(symbol-h)**2)  # a>0, c>0
        quadratic_base = sqrt(c/a)*(symbol-h)
        constant = 1/sqrt(c)
        u_func = None
        if quadratic_base is not symbol:
            u_func = quadratic_base
            quadratic_base = u_var
        standard_form = 1/sqrt(sign_a + sign_c*quadratic_base**2)
        substep = RuleClass(standard_form, quadratic_base)
        if constant != 1:
            substep = ConstantTimesRule(constant*standard_form, symbol, constant, standard_form, substep)
        if u_func is not None:
            substep = URule(rewritten, symbol, u_var, u_func, substep)
        if h != 0:
            substep = CompleteSquareRule(integrand, symbol, rewritten, substep)
        return substep

    a, b, c = [match.get(i, S.Zero) for i in (a, b, c)]
    generic_cond = Ne(c, 0)
    if not degenerate or generic_cond is S.true:
        degenerate_step = None
    elif b.is_zero:
        degenerate_step = ConstantRule(a ** exp, symbol)
    else:
        degenerate_step = sqrt_linear_rule(IntegralInfo((a + b * symbol) ** exp, symbol))

    if simplify(2*exp + 1) == 0:
        h, k = -b/(2*c), a - b**2/(4*c)  # rewrite base to k + c*(symbol-h)**2
        non_square_cond = Ne(k, 0)
        square_step = None
        if non_square_cond is not S.true:
            square_step = NestedPowRule(1/sqrt(c*(symbol-h)**2), symbol, symbol-h, S.NegativeOne)
        if non_square_cond is S.false:
            return square_step
        generic_step = ReciprocalSqrtQuadraticRule(integrand, symbol, a, b, c)
        step = _add_degenerate_step(non_square_cond, generic_step, square_step)
        if k.is_real and c.is_real:
            # list of ((rule, base_exp, a, sign_a, b, sign_b), condition)
            rules = []
            for args, cond in (  # don't apply ArccoshRule to x**2-1
                ((ArcsinRule, k, 1, -c, -1, h), And(k > 0, c < 0)),  # 1-x**2
                ((ArcsinhRule, k, 1, c, 1, h), And(k > 0, c > 0)),  # 1+x**2
            ):
                if cond is S.true:
                    return make_inverse_trig(*args)
                if cond is not S.false:
                    rules.append((make_inverse_trig(*args), cond))
            if rules:
                if not k.is_positive:  # conditions are not thorough, need fall back rule
                    rules.append((generic_step, S.true))
                step = PiecewiseRule(integrand, symbol, rules)
            else:
                step = generic_step
        return _add_degenerate_step(generic_cond, step, degenerate_step)
    if exp == S.Half:
        step = SqrtQuadraticRule(integrand, symbol, a, b, c)
        return _add_degenerate_step(generic_cond, step, degenerate_step)


def add_rule(integral):
    integrand, symbol = integral
    results = [integral_steps(g, symbol)
              for g in integrand.as_ordered_terms()]
    return None if None in results else AddRule(integrand, symbol, results)


def mul_rule(integral: IntegralInfo):
    integrand, symbol = integral

    # Constant times function case
    coeff, f = integrand.as_independent(symbol)
    if coeff != 1:
        next_step = integral_steps(f, symbol)
        if next_step is not None:
            return ConstantTimesRule(integrand, symbol, coeff, f, next_step)


def _parts_rule(integrand, symbol) -> tuple[Expr, Expr, Expr, Expr, Rule] | None:
    # LIATE rule:
    # log, inverse trig, algebraic, trigonometric, exponential
    def pull_out_algebraic(integrand):
        integrand = integrand.cancel().together()
        # iterating over Piecewise args would not work here
        algebraic = ([] if isinstance(integrand, Piecewise) or not integrand.is_Mul
            else [arg for arg in integrand.args if arg.is_algebraic_expr(symbol)])
        if algebraic:
            u = Mul(*algebraic)
            dv = (integrand / u).cancel()
            return u, dv

    def pull_out_u(*functions) -> Callable[[Expr], tuple[Expr, Expr] | None]:
        def pull_out_u_rl(integrand: Expr) -> tuple[Expr, Expr] | None:
            if any(integrand.has(f) for f in functions):
                args = [arg for arg in integrand.args
                        if any(isinstance(arg, cls) for cls in functions)]
                if args:
                    u = Mul(*args) # type: ignore
                    dv = integrand / u
                    return u, dv
            return None

        return pull_out_u_rl

    liate_rules = [pull_out_u(log), pull_out_u(*inverse_trig_functions),
                   pull_out_algebraic, pull_out_u(sin, cos),
                   pull_out_u(exp)]


    dummy = Dummy("temporary")
    # we can integrate log(x) and atan(x) by setting dv = 1
    if isinstance(integrand, (log, *inverse_trig_functions)):
        integrand = dummy * integrand

    for index, rule in enumerate(liate_rules):
        result = rule(integrand)

        if result:
            u, dv = result

            # Don't pick u to be a constant if possible
            if symbol not in u.free_symbols and not u.has(dummy):
                return None

            u = u.subs(dummy, 1)
            dv = dv.subs(dummy, 1)

            # Don't pick a non-polynomial algebraic to be differentiated
            if rule == pull_out_algebraic and not u.is_polynomial(symbol):
                return None
            # Don't trade one logarithm for another
            if isinstance(u, log):
                rec_dv = 1/dv
                if (rec_dv.is_polynomial(symbol) and
                    degree(rec_dv, symbol) == 1):
                    return None

            # Can integrate a polynomial times OrthogonalPolynomial
            if rule == pull_out_algebraic:
                if dv.is_Derivative or dv.has(TrigonometricFunction) or \
                        isinstance(dv, OrthogonalPolynomial):
                    v_step = integral_steps(dv, symbol)
                    if v_step.contains_dont_know():
                        return None
                    else:
                        du = u.diff(symbol)
                        v = v_step.eval()
                        return u, dv, v, du, v_step

            # make sure dv is amenable to integration
            accept = False
            if index < 2:  # log and inverse trig are usually worth trying
                accept = True
            elif (rule == pull_out_algebraic and dv.args and
                all(isinstance(a, (sin, cos, exp))
                for a in dv.args)):
                    accept = True
            else:
                for lrule in liate_rules[index + 1:]:
                    r = lrule(integrand)
                    if r and r[0].subs(dummy, 1).equals(dv):
                        accept = True
                        break

            if accept:
                du = u.diff(symbol)
                v_step = integral_steps(simplify(dv), symbol)
                if not v_step.contains_dont_know():
                    v = v_step.eval()
                    return u, dv, v, du, v_step
    return None


def parts_rule(integral):
    integrand, symbol = integral
    constant, integrand = integrand.as_coeff_Mul()

    result = _parts_rule(integrand, symbol)

    steps = []
    if result:
        u, dv, v, du, v_step = result
        debug("u : {}, dv : {}, v : {}, du : {}, v_step: {}".format(u, dv, v, du, v_step))
        steps.append(result)

        if isinstance(v, Integral):
            return

        # Set a limit on the number of times u can be used
        if isinstance(u, (sin, cos, exp, sinh, cosh)):
            cachekey = u.xreplace({symbol: _cache_dummy})
            if _parts_u_cache[cachekey] > 2:
                return
            _parts_u_cache[cachekey] += 1

        # Try cyclic integration by parts a few times
        for _ in range(4):
            debug("Cyclic integration {} with v: {}, du: {}, integrand: {}".format(_, v, du, integrand))
            coefficient = ((v * du) / integrand).cancel()
            if coefficient == 1:
                break
            if symbol not in coefficient.free_symbols:
                rule = CyclicPartsRule(integrand, symbol,
                    [PartsRule(None, None, u, dv, v_step, None)
                     for (u, dv, v, du, v_step) in steps],
                    (-1) ** len(steps) * coefficient)
                if (constant != 1) and rule:
                    rule = ConstantTimesRule(constant * integrand, symbol, constant, integrand, rule)
                return rule

            # _parts_rule is sensitive to constants, factor it out
            next_constant, next_integrand = (v * du).as_coeff_Mul()
            result = _parts_rule(next_integrand, symbol)

            if result:
                u, dv, v, du, v_step = result
                u *= next_constant
                du *= next_constant
                steps.append((u, dv, v, du, v_step))
            else:
                break

    def make_second_step(steps, integrand):
        if steps:
            u, dv, v, du, v_step = steps[0]
            return PartsRule(integrand, symbol, u, dv, v_step, make_second_step(steps[1:], v * du))
        return integral_steps(integrand, symbol)

    if steps:
        u, dv, v, du, v_step = steps[0]
        rule = PartsRule(integrand, symbol, u, dv, v_step, make_second_step(steps[1:], v * du))
        if (constant != 1) and rule:
            rule = ConstantTimesRule(constant * integrand, symbol, constant, integrand, rule)
        return rule


def trig_rule(integral):
    integrand, symbol = integral
    if integrand == sin(symbol):
        return SinRule(integrand, symbol)
    if integrand == cos(symbol):
        return CosRule(integrand, symbol)
    if integrand == sec(symbol)**2:
        return Sec2Rule(integrand, symbol)
    if integrand == csc(symbol)**2:
        return Csc2Rule(integrand, symbol)

    if isinstance(integrand, tan):
        rewritten = sin(*integrand.args) / cos(*integrand.args)
    elif isinstance(integrand, cot):
        rewritten = cos(*integrand.args) / sin(*integrand.args)
    elif isinstance(integrand, sec):
        arg = integrand.args[0]
        rewritten = ((sec(arg)**2 + tan(arg) * sec(arg)) /
                     (sec(arg) + tan(arg)))
    elif isinstance(integrand, csc):
        arg = integrand.args[0]
        rewritten = ((csc(arg)**2 + cot(arg) * csc(arg)) /
                     (csc(arg) + cot(arg)))
    else:
        return

    return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))

def trig_product_rule(integral: IntegralInfo):
    integrand, symbol = integral
    if integrand == sec(symbol) * tan(symbol):
        return SecTanRule(integrand, symbol)
    if integrand == csc(symbol) * cot(symbol):
        return CscCotRule(integrand, symbol)


def quadratic_denom_rule(integral):
    integrand, symbol = integral
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    c = Wild('c', exclude=[symbol])

    match = integrand.match(a / (b * symbol ** 2 + c))

    if match:
        a, b, c = match[a], match[b], match[c]
        general_rule = ArctanRule(integrand, symbol, a, b, c)
        if b.is_extended_real and c.is_extended_real:
            positive_cond = c/b > 0
            if positive_cond is S.true:
                return general_rule
            coeff = a/(2*sqrt(-c)*sqrt(b))
            constant = sqrt(-c/b)
            r1 = 1/(symbol-constant)
            r2 = 1/(symbol+constant)
            log_steps = [ReciprocalRule(r1, symbol, symbol-constant),
                         ConstantTimesRule(-r2, symbol, -1, r2, ReciprocalRule(r2, symbol, symbol+constant))]
            rewritten = sub = r1 - r2
            negative_step = AddRule(sub, symbol, log_steps)
            if coeff != 1:
                rewritten = Mul(coeff, sub, evaluate=False)
                negative_step = ConstantTimesRule(rewritten, symbol, coeff, sub, negative_step)
            negative_step = RewriteRule(integrand, symbol, rewritten, negative_step)
            if positive_cond is S.false:
                return negative_step
            return PiecewiseRule(integrand, symbol, [(general_rule, positive_cond), (negative_step, S.true)])

        power = PowerRule(integrand, symbol, symbol, -2)
        if b != 1:
            power = ConstantTimesRule(integrand, symbol, 1/b, symbol**-2, power)

        return PiecewiseRule(integrand, symbol, [(general_rule, Ne(c, 0)), (power, True)])

    d = Wild('d', exclude=[symbol])
    match2 = integrand.match(a / (b * symbol ** 2 + c * symbol + d))
    if match2:
        b, c =  match2[b], match2[c]
        if b.is_zero:
            return
        u = Dummy('u')
        u_func = symbol + c/(2*b)
        integrand2 = integrand.subs(symbol, u - c / (2*b))
        next_step = integral_steps(integrand2, u)
        if next_step:
            return URule(integrand2, symbol, u, u_func, next_step)
        else:
            return
    e = Wild('e', exclude=[symbol])
    match3 = integrand.match((a* symbol + b) / (c * symbol ** 2 + d * symbol + e))
    if match3:
        a, b, c, d, e = match3[a], match3[b], match3[c], match3[d], match3[e]
        if c.is_zero:
            return
        denominator = c * symbol**2 + d * symbol + e
        const =  a/(2*c)
        numer1 =  (2*c*symbol+d)
        numer2 = - const*d + b
        u = Dummy('u')
        step1 = URule(integrand, symbol,
                      u, denominator, integral_steps(u**(-1), u))
        if const != 1:
            step1 = ConstantTimesRule(const*numer1/denominator, symbol,
                                      const, numer1/denominator, step1)
        if numer2.is_zero:
            return step1
        step2 = integral_steps(numer2/denominator, symbol)
        substeps = AddRule(integrand, symbol, [step1, step2])
        rewriten = const*numer1/denominator+numer2/denominator
        return RewriteRule(integrand, symbol, rewriten, substeps)

    return


def sqrt_linear_rule(integral: IntegralInfo):
    """
    Substitute common (a+b*x)**(1/n)
    """
    integrand, x = integral
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x, 0])
    a0 = b0 = 0
    bases, qs, bs = [], [], []
    for pow_ in integrand.find(Pow):  # collect all (a+b*x)**(p/q)
        base, exp_ = pow_.base, pow_.exp
        if exp_.is_Integer or x not in base.free_symbols:  # skip 1/x and sqrt(2)
            continue
        if not exp_.is_Rational:  # exclude x**pi
            return
        match = base.match(a+b*x)
        if not match:  # skip non-linear
            continue  # for sqrt(x+sqrt(x)), although base is non-linear, we can still substitute sqrt(x)
        a1, b1 = match[a], match[b]
        if a0*b1 != a1*b0 or not (b0/b1).is_nonnegative:  # cannot transform sqrt(x) to sqrt(x+1) or sqrt(-x)
            return
        if b0 == 0 or (b0/b1 > 1) is S.true:  # choose the latter of sqrt(2*x) and sqrt(x) as representative
            a0, b0 = a1, b1
        bases.append(base)
        bs.append(b1)
        qs.append(exp_.q)
    if b0 == 0:  # no such pattern found
        return
    q0: Integer = lcm_list(qs)
    u_x = (a0 + b0*x)**(1/q0)
    u = Dummy("u")
    substituted = integrand.subs({base**(S.One/q): (b/b0)**(S.One/q)*u**(q0/q)
                                  for base, b, q in zip(bases, bs, qs)}).subs(x, (u**q0-a0)/b0)
    substep = integral_steps(substituted*u**(q0-1)*q0/b0, u)
    if not substep.contains_dont_know():
        step: Rule = URule(integrand, x, u, u_x, substep)
        generic_cond = Ne(b0, 0)
        if generic_cond is not S.true:  # possible degenerate case
            simplified = integrand.subs(dict.fromkeys(bs, 0))
            degenerate_step = integral_steps(simplified, x)
            step = PiecewiseRule(integrand, x, [(step, generic_cond), (degenerate_step, S.true)])
        return step


def sqrt_quadratic_rule(integral: IntegralInfo, degenerate=True):
    integrand, x = integral
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    c = Wild('c', exclude=[x, 0])
    f = Wild('f')
    n = Wild('n', properties=[lambda n: n.is_Integer and n.is_odd])
    match = integrand.match(f*sqrt(a+b*x+c*x**2)**n)
    if not match:
        return
    a, b, c, f, n = match[a], match[b], match[c], match[f], match[n]
    f_poly = f.as_poly(x)
    if f_poly is None:
        return

    generic_cond = Ne(c, 0)
    if not degenerate or generic_cond is S.true:
        degenerate_step = None
    elif b.is_zero:
        degenerate_step = integral_steps(f*sqrt(a)**n, x)
    else:
        degenerate_step = sqrt_linear_rule(IntegralInfo(f*sqrt(a+b*x)**n, x))

    def sqrt_quadratic_denom_rule(numer_poly: Poly, integrand: Expr):
        denom = sqrt(a+b*x+c*x**2)
        deg = numer_poly.degree()
        if deg <= 1:
            # integrand == (d+e*x)/sqrt(a+b*x+c*x**2)
            e, d = numer_poly.all_coeffs() if deg == 1 else (S.Zero, numer_poly.as_expr())
            # rewrite numerator to A*(2*c*x+b) + B
            A = e/(2*c)
            B = d-A*b
            pre_substitute = (2*c*x+b)/denom
            constant_step: Rule | None = None
            linear_step: Rule | None = None
            if A != 0:
                u = Dummy("u")
                pow_rule = PowerRule(1/sqrt(u), u, u, -S.Half)
                linear_step = URule(pre_substitute, x, u, a+b*x+c*x**2, pow_rule)
                if A != 1:
                    linear_step = ConstantTimesRule(A*pre_substitute, x, A, pre_substitute, linear_step)
            if B != 0:
                constant_step = inverse_trig_rule(IntegralInfo(1/denom, x), degenerate=False)
                if B != 1:
                    constant_step = ConstantTimesRule(B/denom, x, B, 1/denom, constant_step)  # type: ignore
            if linear_step and constant_step:
                add = Add(A*pre_substitute, B/denom, evaluate=False)
                step: Rule | None = RewriteRule(integrand, x, add, AddRule(add, x, [linear_step, constant_step]))
            else:
                step = linear_step or constant_step
        else:
            coeffs = numer_poly.all_coeffs()
            step = SqrtQuadraticDenomRule(integrand, x, a, b, c, coeffs)
        return step

    if n > 0:  # rewrite poly * sqrt(s)**(2*k-1) to poly*s**k / sqrt(s)
        numer_poly = f_poly * (a+b*x+c*x**2)**((n+1)/2)
        rewritten = numer_poly.as_expr()/sqrt(a+b*x+c*x**2)
        substep = sqrt_quadratic_denom_rule(numer_poly, rewritten)
        generic_step = RewriteRule(integrand, x, rewritten, substep)
    elif n == -1:
        generic_step = sqrt_quadratic_denom_rule(f_poly, integrand)
    else:
        return  # todo: handle n < -1 case
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


def hyperbolic_rule(integral: tuple[Expr, Symbol]):
    integrand, symbol = integral
    if isinstance(integrand, HyperbolicFunction) and integrand.args[0] == symbol:
        if integrand.func == sinh:
            return SinhRule(integrand, symbol)
        if integrand.func == cosh:
            return CoshRule(integrand, symbol)
        u = Dummy('u')
        if integrand.func == tanh:
            rewritten = sinh(symbol)/cosh(symbol)
            return RewriteRule(integrand, symbol, rewritten,
                   URule(rewritten, symbol, u, cosh(symbol), ReciprocalRule(1/u, u, u)))
        if integrand.func == coth:
            rewritten = cosh(symbol)/sinh(symbol)
            return RewriteRule(integrand, symbol, rewritten,
                   URule(rewritten, symbol, u, sinh(symbol), ReciprocalRule(1/u, u, u)))
        else:
            rewritten = integrand.rewrite(tanh)
            if integrand.func == sech:
                return RewriteRule(integrand, symbol, rewritten,
                       URule(rewritten, symbol, u, tanh(symbol/2),
                       ArctanRule(2/(u**2 + 1), u, S(2), S.One, S.One)))
            if integrand.func == csch:
                return RewriteRule(integrand, symbol, rewritten,
                       URule(rewritten, symbol, u, tanh(symbol/2),
                       ReciprocalRule(1/u, u, u)))

@cacheit
def make_wilds(symbol):
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    m = Wild('m', exclude=[symbol], properties=[lambda n: isinstance(n, Integer)])
    n = Wild('n', exclude=[symbol], properties=[lambda n: isinstance(n, Integer)])

    return a, b, m, n

@cacheit
def sincos_pattern(symbol):
    a, b, m, n = make_wilds(symbol)
    pattern = sin(a*symbol)**m * cos(b*symbol)**n

    return pattern, a, b, m, n

@cacheit
def tansec_pattern(symbol):
    a, b, m, n = make_wilds(symbol)
    pattern = tan(a*symbol)**m * sec(b*symbol)**n

    return pattern, a, b, m, n

@cacheit
def cotcsc_pattern(symbol):
    a, b, m, n = make_wilds(symbol)
    pattern = cot(a*symbol)**m * csc(b*symbol)**n

    return pattern, a, b, m, n

@cacheit
def heaviside_pattern(symbol):
    m = Wild('m', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    g = Wild('g')
    pattern = Heaviside(m*symbol + b) * g

    return pattern, m, b, g

def uncurry(func):
    def uncurry_rl(args):
        return func(*args)
    return uncurry_rl

def trig_rewriter(rewrite):
    def trig_rewriter_rl(args):
        a, b, m, n, integrand, symbol = args
        rewritten = rewrite(a, b, m, n, integrand, symbol)
        if rewritten != integrand:
            return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))
    return trig_rewriter_rl

sincos_botheven_condition = uncurry(
    lambda a, b, m, n, i, s: m.is_even and n.is_even and
    m.is_nonnegative and n.is_nonnegative)

sincos_botheven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (((1 - cos(2*a*symbol)) / 2) ** (m / 2)) *
                                    (((1 + cos(2*b*symbol)) / 2) ** (n / 2)) ))

sincos_sinodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd and m >= 3)

sincos_sinodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 - cos(a*symbol)**2)**((m - 1) / 2) *
                                    sin(a*symbol) *
                                    cos(b*symbol) ** n))

sincos_cosodd_condition = uncurry(lambda a, b, m, n, i, s: n.is_odd and n >= 3)

sincos_cosodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 - sin(b*symbol)**2)**((n - 1) / 2) *
                                    cos(b*symbol) *
                                    sin(a*symbol) ** m))

tansec_seceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)
tansec_seceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + tan(b*symbol)**2) ** (n/2 - 1) *
                                    sec(b*symbol)**2 *
                                    tan(a*symbol) ** m ))

tansec_tanodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)
tansec_tanodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (sec(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                     tan(a*symbol) *
                                     sec(b*symbol) ** n ))

tan_tansquared_condition = uncurry(lambda a, b, m, n, i, s: m == 2 and n == 0)
tan_tansquared = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( sec(a*symbol)**2 - 1))

cotcsc_csceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)
cotcsc_csceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + cot(b*symbol)**2) ** (n/2 - 1) *
                                    csc(b*symbol)**2 *
                                    cot(a*symbol) ** m ))

cotcsc_cotodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)
cotcsc_cotodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (csc(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                    cot(a*symbol) *
                                    csc(b*symbol) ** n ))

def trig_sincos_rule(integral):
    integrand, symbol = integral

    if any(integrand.has(f) for f in (sin, cos)):
        pattern, a, b, m, n = sincos_pattern(symbol)
        match = integrand.match(pattern)
        if not match:
            return

        return multiplexer({
            sincos_botheven_condition: sincos_botheven,
            sincos_sinodd_condition: sincos_sinodd,
            sincos_cosodd_condition: sincos_cosodd
        })(tuple(
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            [integrand, symbol]))

def trig_tansec_rule(integral):
    integrand, symbol = integral

    integrand = integrand.subs({
        1 / cos(symbol): sec(symbol)
    })

    if any(integrand.has(f) for f in (tan, sec)):
        pattern, a, b, m, n = tansec_pattern(symbol)
        match = integrand.match(pattern)
        if not match:
            return

        return multiplexer({
            tansec_tanodd_condition: tansec_tanodd,
            tansec_seceven_condition: tansec_seceven,
            tan_tansquared_condition: tan_tansquared
        })(tuple(
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            [integrand, symbol]))

def trig_cotcsc_rule(integral):
    integrand, symbol = integral
    integrand = integrand.subs({
        1 / sin(symbol): csc(symbol),
        1 / tan(symbol): cot(symbol),
        cos(symbol) / tan(symbol): cot(symbol)
    })

    if any(integrand.has(f) for f in (cot, csc)):
        pattern, a, b, m, n = cotcsc_pattern(symbol)
        match = integrand.match(pattern)
        if not match:
            return

        return multiplexer({
            cotcsc_cotodd_condition: cotcsc_cotodd,
            cotcsc_csceven_condition: cotcsc_csceven
        })(tuple(
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            [integrand, symbol]))

def trig_sindouble_rule(integral):
    integrand, symbol = integral
    a = Wild('a', exclude=[sin(2*symbol)])
    match = integrand.match(sin(2*symbol)*a)
    if match:
        sin_double = 2*sin(symbol)*cos(symbol)/sin(2*symbol)
        return integral_steps(integrand * sin_double, symbol)

def trig_powers_products_rule(integral):
    return do_one(null_safe(trig_sincos_rule),
                  null_safe(trig_tansec_rule),
                  null_safe(trig_cotcsc_rule),
                  null_safe(trig_sindouble_rule))(integral)

def trig_substitution_rule(integral):
    integrand, symbol = integral
    A = Wild('a', exclude=[0, symbol])
    B = Wild('b', exclude=[0, symbol])
    theta = Dummy("theta")
    target_pattern = A + B*symbol**2

    matches = integrand.find(target_pattern)
    for expr in matches:
        match = expr.match(target_pattern)
        a = match.get(A, S.Zero)
        b = match.get(B, S.Zero)

        a_positive = ((a.is_number and a > 0) or a.is_positive)
        b_positive = ((b.is_number and b > 0) or b.is_positive)
        a_negative = ((a.is_number and a < 0) or a.is_negative)
        b_negative = ((b.is_number and b < 0) or b.is_negative)
        x_func = None
        if a_positive and b_positive:
            # a**2 + b*x**2. Assume sec(theta) > 0, -pi/2 < theta < pi/2
            x_func = (sqrt(a)/sqrt(b)) * tan(theta)
            # Do not restrict the domain: tan(theta) takes on any real
            # value on the interval -pi/2 < theta < pi/2 so x takes on
            # any value
            restriction = True
        elif a_positive and b_negative:
            # a**2 - b*x**2. Assume cos(theta) > 0, -pi/2 < theta < pi/2
            constant = sqrt(a)/sqrt(-b)
            x_func = constant * sin(theta)
            restriction = And(symbol > -constant, symbol < constant)
        elif a_negative and b_positive:
            # b*x**2 - a**2. Assume sin(theta) > 0, 0 < theta < pi
            constant = sqrt(-a)/sqrt(b)
            x_func = constant * sec(theta)
            restriction = And(symbol > -constant, symbol < constant)
        if x_func:
            # Manually simplify sqrt(trig(theta)**2) to trig(theta)
            # Valid due to assumed domain restriction
            substitutions = {}
            for f in [sin, cos, tan,
                      sec, csc, cot]:
                substitutions[sqrt(f(theta)**2)] = f(theta)
                substitutions[sqrt(f(theta)**(-2))] = 1/f(theta)

            replaced = integrand.subs(symbol, x_func).trigsimp()
            replaced = manual_subs(replaced, substitutions)
            if not replaced.has(symbol):
                replaced *= manual_diff(x_func, theta)
                replaced = replaced.trigsimp()
                secants = replaced.find(1/cos(theta))
                if secants:
                    replaced = replaced.xreplace({
                        1/cos(theta): sec(theta)
                    })

                substep = integral_steps(replaced, theta)
                if not substep.contains_dont_know():
                    return TrigSubstitutionRule(integrand, symbol,
                        theta, x_func, replaced, substep, restriction)

def heaviside_rule(integral):
    integrand, symbol = integral
    pattern, m, b, g = heaviside_pattern(symbol)
    match = integrand.match(pattern)
    if match and 0 != match[g]:
        # f = Heaviside(m*x + b)*g
        substep = integral_steps(match[g], symbol)
        m, b = match[m], match[b]
        return HeavisideRule(integrand, symbol, m*symbol + b, -b/m, substep)


def dirac_delta_rule(integral: IntegralInfo):
    integrand, x = integral
    if len(integrand.args) == 1:
        n = S.Zero
    else:
        n = integrand.args[1] # type: ignore
    if not n.is_Integer or n < 0:
        return
    a, b = Wild('a', exclude=[x]), Wild('b', exclude=[x, 0])
    match = integrand.args[0].match(a+b*x)
    if not match:
        return
    a, b = match[a], match[b]
    generic_cond = Ne(b, 0)
    if generic_cond is S.true:
        degenerate_step = None
    else:
        degenerate_step = ConstantRule(DiracDelta(a, n), x)
    generic_step = DiracDeltaRule(integrand, x, n, a, b)
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


def substitution_rule(integral):
    integrand, symbol = integral

    u_var = Dummy("u")
    substitutions = find_substitutions(integrand, symbol, u_var)
    count = 0
    if substitutions:
        debug("List of Substitution Rules")
        ways = []
        for u_func, c, substituted in substitutions:
            subrule = integral_steps(substituted, u_var)
            count = count + 1
            debug("Rule {}: {}".format(count, subrule))

            if subrule.contains_dont_know():
                continue

            if simplify(c - 1) != 0:
                _, denom = c.as_numer_denom()
                if subrule:
                    subrule = ConstantTimesRule(c * substituted, u_var, c, substituted, subrule)

                if denom.free_symbols:
                    piecewise = []
                    could_be_zero = []

                    if isinstance(denom, Mul):
                        could_be_zero = denom.args
                    else:
                        could_be_zero.append(denom)

                    for expr in could_be_zero:
                        if not fuzzy_not(expr.is_zero):
                            substep = integral_steps(manual_subs(integrand, expr, 0), symbol)

                            if substep:
                                piecewise.append((
                                    substep,
                                    Eq(expr, 0)
                                ))
                    piecewise.append((subrule, True))
                    subrule = PiecewiseRule(substituted, symbol, piecewise)

            ways.append(URule(integrand, symbol, u_var, u_func, subrule))

        if len(ways) > 1:
            return AlternativeRule(integrand, symbol, ways)
        elif ways:
            return ways[0]


partial_fractions_rule = rewriter(
    lambda integrand, symbol: integrand.is_rational_function(),
    lambda integrand, symbol: integrand.apart(symbol))

cancel_rule = rewriter(
    # lambda integrand, symbol: integrand.is_algebraic_expr(),
    # lambda integrand, symbol: isinstance(integrand, Mul),
    lambda integrand, symbol: True,
    lambda integrand, symbol: integrand.cancel())

distribute_expand_rule = rewriter(
    lambda integrand, symbol: (
        isinstance(integrand, (Pow, Mul)) or all(arg.is_Pow or arg.is_polynomial(symbol) for arg in integrand.args)),
    lambda integrand, symbol: integrand.expand())

trig_expand_rule = rewriter(
    # If there are trig functions with different arguments, expand them
    lambda integrand, symbol: (
        len({a.args[0] for a in integrand.atoms(TrigonometricFunction)}) > 1),
    lambda integrand, symbol: integrand.expand(trig=True))

def derivative_rule(integral):
    integrand = integral[0]
    diff_variables = integrand.variables
    undifferentiated_function = integrand.expr
    integrand_variables = undifferentiated_function.free_symbols

    if integral.symbol in integrand_variables:
        if integral.symbol in diff_variables:
            return DerivativeRule(*integral)
        else:
            return DontKnowRule(integrand, integral.symbol)
    else:
        return ConstantRule(*integral)

def rewrites_rule(integral):
    integrand, symbol = integral

    if integrand.match(1/cos(symbol)):
        rewritten = integrand.subs(1/cos(symbol), sec(symbol))
        return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))

def fallback_rule(integral):
    return DontKnowRule(*integral)

# Cache is used to break cyclic integrals.
# Need to use the same dummy variable in cached expressions for them to match.
# Also record "u" of integration by parts, to avoid infinite repetition.
_integral_cache: dict[Expr, Expr | None] = {}
_parts_u_cache: dict[Expr, int] = defaultdict(int)
_cache_dummy = Dummy("z")

def integral_steps(integrand, symbol, **options):
    """Returns the steps needed to compute an integral.

    Explanation
    ===========

    This function attempts to mirror what a student would do by hand as
    closely as possible.

    SymPy Gamma uses this to provide a step-by-step explanation of an
    integral. The code it uses to format the results of this function can be
    found at
    https://github.com/sympy/sympy_gamma/blob/master/app/logic/intsteps.py.

    Examples
    ========

    >>> from sympy import exp, sin
    >>> from sympy.integrals.manualintegrate import integral_steps
    >>> from sympy.abc import x
    >>> print(repr(integral_steps(exp(x) / (1 + exp(2 * x)), x))) \
    # doctest: +NORMALIZE_WHITESPACE
    URule(integrand=exp(x)/(exp(2*x) + 1), variable=x, u_var=_u, u_func=exp(x),
    substep=ArctanRule(integrand=1/(_u**2 + 1), variable=_u, a=1, b=1, c=1))
    >>> print(repr(integral_steps(sin(x), x))) \
    # doctest: +NORMALIZE_WHITESPACE
    SinRule(integrand=sin(x), variable=x)
    >>> print(repr(integral_steps((x**2 + 3)**2, x))) \
    # doctest: +NORMALIZE_WHITESPACE
    RewriteRule(integrand=(x**2 + 3)**2, variable=x, rewritten=x**4 + 6*x**2 + 9,
    substep=AddRule(integrand=x**4 + 6*x**2 + 9, variable=x,
    substeps=[PowerRule(integrand=x**4, variable=x, base=x, exp=4),
    ConstantTimesRule(integrand=6*x**2, variable=x, constant=6, other=x**2,
    substep=PowerRule(integrand=x**2, variable=x, base=x, exp=2)),
    ConstantRule(integrand=9, variable=x)]))

    Returns
    =======

    rule : Rule
        The first step; most rules have substeps that must also be
        considered. These substeps can be evaluated using ``manualintegrate``
        to obtain a result.

    """
    cachekey = integrand.xreplace({symbol: _cache_dummy})
    if cachekey in _integral_cache:
        if _integral_cache[cachekey] is None:
            # Stop this attempt, because it leads around in a loop
            return DontKnowRule(integrand, symbol)
        else:
            # TODO: This is for future development, as currently
            # _integral_cache gets no values other than None
            return (_integral_cache[cachekey].xreplace(_cache_dummy, symbol),
                symbol)
    else:
        _integral_cache[cachekey] = None

    integral = IntegralInfo(integrand, symbol)

    def key(integral):
        integrand = integral.integrand

        if symbol not in integrand.free_symbols:
            return Number
        for cls in (Symbol, TrigonometricFunction, OrthogonalPolynomial):
            if isinstance(integrand, cls):
                return cls
        return type(integrand)

    def integral_is_subclass(*klasses):
        def _integral_is_subclass(integral):
            k = key(integral)
            return k and issubclass(k, klasses)
        return _integral_is_subclass

    result = do_one(
        null_safe(special_function_rule),
        null_safe(switch(key, {
            Pow: do_one(null_safe(power_rule), null_safe(inverse_trig_rule),
                        null_safe(sqrt_linear_rule),
                        null_safe(quadratic_denom_rule)),
            Symbol: power_rule,
            exp: exp_rule,
            Add: add_rule,
            Mul: do_one(null_safe(mul_rule), null_safe(trig_product_rule),
                        null_safe(heaviside_rule), null_safe(quadratic_denom_rule),
                        null_safe(sqrt_linear_rule),
                        null_safe(sqrt_quadratic_rule)),
            Derivative: derivative_rule,
            TrigonometricFunction: trig_rule,
            Heaviside: heaviside_rule,
            DiracDelta: dirac_delta_rule,
            OrthogonalPolynomial: orthogonal_poly_rule,
            Number: constant_rule
        })),
        do_one(
            null_safe(trig_rule),
            null_safe(hyperbolic_rule),
            null_safe(alternatives(
                rewrites_rule,
                substitution_rule,
                condition(
                    integral_is_subclass(Mul, Pow),
                    partial_fractions_rule),
                condition(
                    integral_is_subclass(Mul, Pow),
                    cancel_rule),
                condition(
                    integral_is_subclass(Mul, log,
                    *inverse_trig_functions),
                    parts_rule),
                condition(
                    integral_is_subclass(Mul, Pow),
                    distribute_expand_rule),
                trig_powers_products_rule,
                trig_expand_rule
            )),
            null_safe(condition(integral_is_subclass(Mul, Pow), nested_pow_rule)),
            null_safe(trig_substitution_rule)
        ),
        fallback_rule)(integral)
    del _integral_cache[cachekey]
    return result


def manualintegrate(f, var):
    """manualintegrate(f, var)

    Explanation
    ===========

    Compute indefinite integral of a single variable using an algorithm that
    resembles what a student would do by hand.

    Unlike :func:`~.integrate`, var can only be a single symbol.

    Examples
    ========

    >>> from sympy import sin, cos, tan, exp, log, integrate
    >>> from sympy.integrals.manualintegrate import manualintegrate
    >>> from sympy.abc import x
    >>> manualintegrate(1 / x, x)
    log(x)
    >>> integrate(1/x)
    log(x)
    >>> manualintegrate(log(x), x)
    x*log(x) - x
    >>> integrate(log(x))
    x*log(x) - x
    >>> manualintegrate(exp(x) / (1 + exp(2 * x)), x)
    atan(exp(x))
    >>> integrate(exp(x) / (1 + exp(2 * x)))
    RootSum(4*_z**2 + 1, Lambda(_i, _i*log(2*_i + exp(x))))
    >>> manualintegrate(cos(x)**4 * sin(x), x)
    -cos(x)**5/5
    >>> integrate(cos(x)**4 * sin(x), x)
    -cos(x)**5/5
    >>> manualintegrate(cos(x)**4 * sin(x)**3, x)
    cos(x)**7/7 - cos(x)**5/5
    >>> integrate(cos(x)**4 * sin(x)**3, x)
    cos(x)**7/7 - cos(x)**5/5
    >>> manualintegrate(tan(x), x)
    -log(cos(x))
    >>> integrate(tan(x), x)
    -log(cos(x))

    See Also
    ========

    sympy.integrals.integrals.integrate
    sympy.integrals.integrals.Integral.doit
    sympy.integrals.integrals.Integral
    """
    result = integral_steps(f, var).eval()
    # Clear the cache of u-parts
    _parts_u_cache.clear()
    # If we got Piecewise with two parts, put generic first
    if isinstance(result, Piecewise) and len(result.args) == 2:
        cond = result.args[0][1]
        if isinstance(cond, Eq) and result.args[1][1] == True:
            result = result.func(
                (result.args[1][0], Ne(*cond.args)),
                (result.args[0][0], True))
    return result
<|endoftext|>"""Laplace Transforms"""
import sys
import sympy
from sympy.core import S, pi, I
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (
    AppliedUndef, Derivative, expand, expand_complex, expand_mul, expand_trig,
    Lambda, WildFunction, diff, Subs)
from sympy.core.mul import Mul, prod
from sympy.core.relational import (
    _canonical, Ge, Gt, Lt, Unequality, Eq, Ne, Relational)
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.functions.elementary.complexes import (
    re, im, arg, Abs, polar_lift, periodic_argument)
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, asinh
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import (
    Piecewise, piecewise_exclusive)
from sympy.functions.elementary.trigonometric import cos, sin, atan, sinc
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.special.gamma_functions import (
    digamma, gamma, lowergamma, uppergamma)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.integrals import integrate, Integral
from sympy.integrals.transforms import (
    _simplify, IntegralTransform, IntegralTransformError)
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.matrices.matrixbase import MatrixBase
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.utilities.exceptions import (
    sympy_deprecation_warning, SymPyDeprecationWarning, ignore_warnings)
from sympy.utilities.misc import debugf

_LT_level = 0


def DEBUG_WRAP(func):
    def wrap(*args, **kwargs):
        from sympy import SYMPY_DEBUG
        global _LT_level

        if not SYMPY_DEBUG:
            return func(*args, **kwargs)

        if _LT_level == 0:
            print('\n' + '-'*78, file=sys.stderr)
        print('-LT- %s%s%s' % ('  '*_LT_level, func.__name__, args),
              file=sys.stderr)
        _LT_level += 1
        if (
                func.__name__ == '_laplace_transform_integration' or
                func.__name__ == '_inverse_laplace_transform_integration'):
            sympy.SYMPY_DEBUG = False
            print('**** %sIntegrating ...' % ('  '*_LT_level), file=sys.stderr)
            result = func(*args, **kwargs)
            sympy.SYMPY_DEBUG = True
        else:
            result = func(*args, **kwargs)
        _LT_level -= 1
        print('-LT- %s---> %s' % ('  '*_LT_level, result), file=sys.stderr)
        if _LT_level == 0:
            print('-'*78 + '\n', file=sys.stderr)
        return result
    return wrap


def _debug(text):
    from sympy import SYMPY_DEBUG

    if SYMPY_DEBUG:
        print('-LT- %s%s' % ('  '*_LT_level, text), file=sys.stderr)


def _simplifyconds(expr, s, a):
    r"""
    Naively simplify some conditions occurring in ``expr``,
    given that `\operatorname{Re}(s) > a`.

    Examples
    ========

    >>> from sympy.integrals.laplace import _simplifyconds
    >>> from sympy.abc import x
    >>> from sympy import sympify as S
    >>> _simplifyconds(abs(x**2) < 1, x, 1)
    False
    >>> _simplifyconds(abs(x**2) < 1, x, 2)
    False
    >>> _simplifyconds(abs(x**2) < 1, x, 0)
    Abs(x**2) < 1
    >>> _simplifyconds(abs(1/x**2) < 1, x, 1)
    True
    >>> _simplifyconds(S(1) < abs(x), x, 1)
    True
    >>> _simplifyconds(S(1) < abs(1/x), x, 1)
    False

    >>> from sympy import Ne
    >>> _simplifyconds(Ne(1, x**3), x, 1)
    True
    >>> _simplifyconds(Ne(1, x**3), x, 2)
    True
    >>> _simplifyconds(Ne(1, x**3), x, 0)
    Ne(1, x**3)
    """

    def power(ex):
        if ex == s:
            return 1
        if ex.is_Pow and ex.base == s:
            return ex.exp
        return None

    def bigger(ex1, ex2):
        """ Return True only if |ex1| > |ex2|, False only if |ex1| < |ex2|.
            Else return None. """
        if ex1.has(s) and ex2.has(s):
            return None
        if isinstance(ex1, Abs):
            ex1 = ex1.args[0]
        if isinstance(ex2, Abs):
            ex2 = ex2.args[0]
        if ex1.has(s):
            return bigger(1/ex2, 1/ex1)
        n = power(ex2)
        if n is None:
            return None
        try:
            if n > 0 and (Abs(ex1) <= Abs(a)**n) == S.true:
                return False
            if n < 0 and (Abs(ex1) >= Abs(a)**n) == S.true:
                return True
        except TypeError:
            return None

    def replie(x, y):
        """ simplify x < y """
        if (not (x.is_positive or isinstance(x, Abs))
                or not (y.is_positive or isinstance(y, Abs))):
            return (x < y)
        r = bigger(x, y)
        if r is not None:
            return not r
        return (x < y)

    def replue(x, y):
        b = bigger(x, y)
        if b in (True, False):
            return True
        return Unequality(x, y)

    def repl(ex, *args):
        if ex in (True, False):
            return bool(ex)
        return ex.replace(*args)

    from sympy.simplify.radsimp import collect_abs
    expr = collect_abs(expr)
    expr = repl(expr, Lt, replie)
    expr = repl(expr, Gt, lambda x, y: replie(y, x))
    expr = repl(expr, Unequality, replue)
    return S(expr)


@DEBUG_WRAP
def expand_dirac_delta(expr):
    """
    Expand an expression involving DiractDelta to get it as a linear
    combination of DiracDelta functions.
    """
    return _lin_eq2dict(expr, expr.atoms(DiracDelta))


@DEBUG_WRAP
def _laplace_transform_integration(f, t, s_, *, simplify):
    """ The backend function for doing Laplace transforms by integration.

    This backend assumes that the frontend has already split sums
    such that `f` is to an addition anymore.
    """
    s = Dummy('s')

    if f.has(DiracDelta):
        return None

    F = integrate(f*exp(-s*t), (t, S.Zero, S.Infinity))

    if not F.has(Integral):
        return _simplify(F.subs(s, s_), simplify), S.NegativeInfinity, S.true

    if not F.is_Piecewise:
        return None

    F, cond = F.args[0]
    if F.has(Integral):
        return None

    def process_conds(conds):
        """ Turn ``conds`` into a strip and auxiliary conditions. """
        from sympy.solvers.inequalities import _solve_inequality
        a = S.NegativeInfinity
        aux = S.true
        conds = conjuncts(to_cnf(conds))
        p, q, w1, w2, w3, w4, w5 = symbols(
            'p q w1 w2 w3 w4 w5', cls=Wild, exclude=[s])
        patterns = (
            p*Abs(arg((s + w3)*q)) < w2,
            p*Abs(arg((s + w3)*q)) <= w2,
            Abs(periodic_argument((s + w3)**p*q, w1)) < w2,
            Abs(periodic_argument((s + w3)**p*q, w1)) <= w2,
            Abs(periodic_argument((polar_lift(s + w3))**p*q, w1)) < w2,
            Abs(periodic_argument((polar_lift(s + w3))**p*q, w1)) <= w2)
        for c in conds:
            a_ = S.Infinity
            aux_ = []
            for d in disjuncts(c):
                if d.is_Relational and s in d.rhs.free_symbols:
                    d = d.reversed
                if d.is_Relational and isinstance(d, (Ge, Gt)):
                    d = d.reversedsign
                for pat in patterns:
                    m = d.match(pat)
                    if m:
                        break
                if m and m[q].is_positive and m[w2]/m[p] == pi/2:
                    d = -re(s + m[w3]) < 0
                m = d.match(p - cos(w1*Abs(arg(s*w5))*w2)*Abs(s**w3)**w4 < 0)
                if not m:
                    m = d.match(
                        cos(p - Abs(periodic_argument(s**w1*w5, q))*w2) *
                        Abs(s**w3)**w4 < 0)
                if not m:
                    m = d.match(
                        p - cos(
                            Abs(periodic_argument(polar_lift(s)**w1*w5, q))*w2
                            )*Abs(s**w3)**w4 < 0)
                if m and all(m[wild].is_positive for wild in [
                        w1, w2, w3, w4, w5]):
                    d = re(s) > m[p]
                d_ = d.replace(
                    re, lambda x: x.expand().as_real_imag()[0]).subs(re(s), t)
                if (
                        not d.is_Relational or d.rel_op in ('==', '!=')
                        or d_.has(s) or not d_.has(t)):
                    aux_ += [d]
                    continue
                soln = _solve_inequality(d_, t)
                if not soln.is_Relational or soln.rel_op in ('==', '!='):
                    aux_ += [d]
                    continue
                if soln.lts == t:
                    return None
                else:
                    a_ = Min(soln.lts, a_)
            if a_ is not S.Infinity:
                a = Max(a_, a)
            else:
                aux = And(aux, Or(*aux_))
        return a, aux.canonical if aux.is_Relational else aux

    conds = [process_conds(c) for c in disjuncts(cond)]
    conds2 = [x for x in conds if x[1] !=
              S.false and x[0] is not S.NegativeInfinity]
    if not conds2:
        conds2 = [x for x in conds if x[1] != S.false]
    conds = list(ordered(conds2))

    def cnt(expr):
        if expr in (True, False):
            return 0
        return expr.count_ops()
    conds.sort(key=lambda x: (-x[0], cnt(x[1])))

    if not conds:
        return None
    a, aux = conds[0]  # XXX is [0] always the right one?

    def sbs(expr):
        return expr.subs(s, s_)
    if simplify:
        F = _simplifyconds(F, s, a)
        aux = _simplifyconds(aux, s, a)
    return _simplify(F.subs(s, s_), simplify), sbs(a), _canonical(sbs(aux))


@DEBUG_WRAP
def _laplace_deep_collect(f, t):
    """
    This is an internal helper function that traverses through the expression
    tree of `f(t)` and collects arguments. The purpose of it is that
    anything like `f(w*t-1*t-c)` will be written as `f((w-1)*t-c)` such that
    it can match `f(a*t+b)`.
    """
    if not isinstance(f, Expr):
        return f
    if (p := f.as_poly(t)) is not None:
        return p.as_expr()
    func = f.func
    args = [_laplace_deep_collect(arg, t) for arg in f.args]
    return func(*args)


@cacheit
def _laplace_build_rules():
    """
    This is an internal helper function that returns the table of Laplace
    transform rules in terms of the time variable `t` and the frequency
    variable `s`.  It is used by ``_laplace_apply_rules``.  Each entry is a
    tuple containing:

        (time domain pattern,
         frequency-domain replacement,
         condition for the rule to be applied,
         convergence plane,
         preparation function)

    The preparation function is a function with one argument that is applied
    to the expression before matching. For most rules it should be
    ``_laplace_deep_collect``.
    """
    t = Dummy('t')
    s = Dummy('s')
    a = Wild('a', exclude=[t])
    b = Wild('b', exclude=[t])
    n = Wild('n', exclude=[t])
    tau = Wild('tau', exclude=[t])
    omega = Wild('omega', exclude=[t])
    def dco(f): return _laplace_deep_collect(f, t)
    _debug('_laplace_build_rules is building rules')

    laplace_transform_rules = [
        (a, a/s,
         S.true, S.Zero, dco),  # 4.2.1
        (DiracDelta(a*t-b), exp(-s*b/a)/Abs(a),
         Or(And(a > 0, b >= 0), And(a < 0, b <= 0)),
         S.NegativeInfinity, dco),  # Not in Bateman54
        (DiracDelta(a*t-b), S(0),
         Or(And(a < 0, b >= 0), And(a > 0, b <= 0)),
         S.NegativeInfinity, dco),  # Not in Bateman54
        (Heaviside(a*t-b), exp(-s*b/a)/s,
         And(a > 0, b > 0), S.Zero, dco),  # 4.4.1
        (Heaviside(a*t-b), (1-exp(-s*b/a))/s,
         And(a < 0, b < 0), S.Zero, dco),  # 4.4.1
        (Heaviside(a*t-b), 1/s,
         And(a > 0, b <= 0), S.Zero, dco),  # 4.4.1
        (Heaviside(a*t-b), 0,
         And(a < 0, b > 0), S.Zero, dco),  # 4.4.1
        (t, 1/s**2,
         S.true, S.Zero, dco),  # 4.2.3
        (1/(a*t+b), -exp(-b/a*s)*Ei(-b/a*s)/a,
         Abs(arg(b/a)) < pi, S.Zero, dco),  # 4.2.6
        (1/sqrt(a*t+b), sqrt(a*pi/s)*exp(b/a*s)*erfc(sqrt(b/a*s))/a,
         Abs(arg(b/a)) < pi, S.Zero, dco),  # 4.2.18
        ((a*t+b)**(-S(3)/2),
         2*b**(-S(1)/2)-2*(pi*s/a)**(S(1)/2)*exp(b/a*s) * erfc(sqrt(b/a*s))/a,
         Abs(arg(b/a)) < pi, S.Zero, dco),  # 4.2.20
        (sqrt(t)/(t+b), sqrt(pi/s)-pi*sqrt(b)*exp(b*s)*erfc(sqrt(b*s)),
         Abs(arg(b)) < pi, S.Zero, dco),  # 4.2.22
        (1/(a*sqrt(t) + t**(3/2)), pi*a**(S(1)/2)*exp(a*s)*erfc(sqrt(a*s)),
         S.true, S.Zero, dco),  # Not in Bateman54
        (t**n, gamma(n+1)/s**(n+1),
         n > -1, S.Zero, dco),  # 4.3.1
        ((a*t+b)**n, uppergamma(n+1, b/a*s)*exp(-b/a*s)/s**(n+1)/a,
         And(n > -1, Abs(arg(b/a)) < pi), S.Zero, dco),  # 4.3.4
        (t**n/(t+a), a**n*gamma(n+1)*uppergamma(-n, a*s),
         And(n > -1, Abs(arg(a)) < pi), S.Zero, dco),  # 4.3.7
        (exp(a*t-tau), exp(-tau)/(s-a),
         S.true, re(a), dco),  # 4.5.1
        (t*exp(a*t-tau), exp(-tau)/(s-a)**2,
         S.true, re(a), dco),  # 4.5.2
        (t**n*exp(a*t), gamma(n+1)/(s-a)**(n+1),
         re(n) > -1, re(a), dco),  # 4.5.3
        (exp(-a*t**2), sqrt(pi/4/a)*exp(s**2/4/a)*erfc(s/sqrt(4*a)),
         re(a) > 0, S.Zero, dco),  # 4.5.21
        (t*exp(-a*t**2),
         1/(2*a)-2/sqrt(pi)/(4*a)**(S(3)/2)*s*erfc(s/sqrt(4*a)),
         re(a) > 0, S.Zero, dco),  # 4.5.22
        (exp(-a/t), 2*sqrt(a/s)*besselk(1, 2*sqrt(a*s)),
         re(a) >= 0, S.Zero, dco),  # 4.5.25
        (sqrt(t)*exp(-a/t),
         S(1)/2*sqrt(pi/s**3)*(1+2*sqrt(a*s))*exp(-2*sqrt(a*s)),
         re(a) >= 0, S.Zero, dco),  # 4.5.26
        (exp(-a/t)/sqrt(t), sqrt(pi/s)*exp(-2*sqrt(a*s)),
         re(a) >= 0, S.Zero, dco),  # 4.5.27
        (exp(-a/t)/(t*sqrt(t)), sqrt(pi/a)*exp(-2*sqrt(a*s)),
         re(a) > 0, S.Zero, dco),  # 4.5.28
        (t**n*exp(-a/t), 2*(a/s)**((n+1)/2)*besselk(n+1, 2*sqrt(a*s)),
         re(a) > 0, S.Zero, dco),  # 4.5.29
        # TODO: rules with sqrt(a*t) and sqrt(a/t) have stopped working after
        #       changes to as_base_exp
        # (exp(-2*sqrt(a*t)),
        #  s**(-1)-sqrt(pi*a)*s**(-S(3)/2)*exp(a/s) * erfc(sqrt(a/s)),
        #  Abs(arg(a)) < pi, S.Zero, dco),  # 4.5.31
        # (exp(-2*sqrt(a*t))/sqrt(t), (pi/s)**(S(1)/2)*exp(a/s)*erfc(sqrt(a/s)),
        #  Abs(arg(a)) < pi, S.Zero, dco),  # 4.5.33
        (exp(-a*exp(-t)), a**(-s)*lowergamma(s, a),
         S.true, S.Zero, dco),  # 4.5.36
        (exp(-a*exp(t)), a**s*uppergamma(-s, a),
         re(a) > 0, S.Zero, dco),  # 4.5.37
        (log(a*t), -log(exp(S.EulerGamma)*s/a)/s,
         a > 0, S.Zero, dco),  # 4.6.1
        (log(1+a*t), -exp(s/a)/s*Ei(-s/a),
         Abs(arg(a)) < pi, S.Zero, dco),  # 4.6.4
        (log(a*t+b), (log(b)-exp(s/b/a)/s*a*Ei(-s/b))/s*a,
         And(a > 0, Abs(arg(b)) < pi), S.Zero, dco),  # 4.6.5
        (log(t)/sqrt(t), -sqrt(pi/s)*log(4*s*exp(S.EulerGamma)),
         S.true, S.Zero, dco),  # 4.6.9
        (t**n*log(t), gamma(n+1)*s**(-n-1)*(digamma(n+1)-log(s)),
         re(n) > -1, S.Zero, dco),  # 4.6.11
        (log(a*t)**2, (log(exp(S.EulerGamma)*s/a)**2+pi**2/6)/s,
         a > 0, S.Zero, dco),  # 4.6.13
        (sin(omega*t), omega/(s**2+omega**2),
         S.true, Abs(im(omega)), dco),  # 4,7,1
        (Abs(sin(omega*t)), omega/(s**2+omega**2)*coth(pi*s/2/omega),
         omega > 0, S.Zero, dco),  # 4.7.2
        (sin(omega*t)/t, atan(omega/s),
         S.true, Abs(im(omega)), dco),  # 4.7.16
        (sin(omega*t)**2/t, log(1+4*omega**2/s**2)/4,
         S.true, 2*Abs(im(omega)), dco),  # 4.7.17
        (sin(omega*t)**2/t**2,
         omega*atan(2*omega/s)-s*log(1+4*omega**2/s**2)/4,
         S.true, 2*Abs(im(omega)), dco),  # 4.7.20
        # (sin(2*sqrt(a*t)), sqrt(pi*a)/s/sqrt(s)*exp(-a/s),
        #  S.true, S.Zero, dco),  # 4.7.32
        # (sin(2*sqrt(a*t))/t, pi*erf(sqrt(a/s)),
        #  S.true, S.Zero, dco),  # 4.7.34
        (cos(omega*t), s/(s**2+omega**2),
         S.true, Abs(im(omega)), dco),  # 4.7.43
        (cos(omega*t)**2, (s**2+2*omega**2)/(s**2+4*omega**2)/s,
         S.true, 2*Abs(im(omega)), dco),  # 4.7.45
        # (sqrt(t)*cos(2*sqrt(a*t)), sqrt(pi)/2*s**(-S(5)/2)*(s-2*a)*exp(-a/s),
        #  S.true, S.Zero, dco),  # 4.7.66
        # (cos(2*sqrt(a*t))/sqrt(t), sqrt(pi/s)*exp(-a/s),
        #  S.true, S.Zero, dco),  # 4.7.67
        (sin(a*t)*sin(b*t), 2*a*b*s/(s**2+(a+b)**2)/(s**2+(a-b)**2),
         S.true, Abs(im(a))+Abs(im(b)), dco),  # 4.7.78
        (cos(a*t)*sin(b*t), b*(s**2-a**2+b**2)/(s**2+(a+b)**2)/(s**2+(a-b)**2),
         S.true, Abs(im(a))+Abs(im(b)), dco),  # 4.7.79
        (cos(a*t)*cos(b*t), s*(s**2+a**2+b**2)/(s**2+(a+b)**2)/(s**2+(a-b)**2),
         S.true, Abs(im(a))+Abs(im(b)), dco),  # 4.7.80
        (sinh(a*t), a/(s**2-a**2),
         S.true, Abs(re(a)), dco),  # 4.9.1
        (cosh(a*t), s/(s**2-a**2),
         S.true, Abs(re(a)), dco),  # 4.9.2
        (sinh(a*t)**2, 2*a**2/(s**3-4*a**2*s),
         S.true, 2*Abs(re(a)), dco),  # 4.9.3
        (cosh(a*t)**2, (s**2-2*a**2)/(s**3-4*a**2*s),
         S.true, 2*Abs(re(a)), dco),  # 4.9.4
        (sinh(a*t)/t, log((s+a)/(s-a))/2,
         S.true, Abs(re(a)), dco),  # 4.9.12
        (t**n*sinh(a*t), gamma(n+1)/2*((s-a)**(-n-1)-(s+a)**(-n-1)),
         n > -2, Abs(a), dco),  # 4.9.18
        (t**n*cosh(a*t), gamma(n+1)/2*((s-a)**(-n-1)+(s+a)**(-n-1)),
         n > -1, Abs(a), dco),  # 4.9.19
        # TODO
        # (sinh(2*sqrt(a*t)), sqrt(pi*a)/s/sqrt(s)*exp(a/s),
        #  S.true, S.Zero, dco),  # 4.9.34
        # (cosh(2*sqrt(a*t)), 1/s+sqrt(pi*a)/s/sqrt(s)*exp(a/s)*erf(sqrt(a/s)),
        #  S.true, S.Zero, dco),  # 4.9.35
        # (
        #     sqrt(t)*sinh(2*sqrt(a*t)),
        #     pi**(S(1)/2)*s**(-S(5)/2)*(s/2+a) *
        #     exp(a/s)*erf(sqrt(a/s))-a**(S(1)/2)*s**(-2),
        #     S.true, S.Zero, dco),  # 4.9.36
        # (sqrt(t)*cosh(2*sqrt(a*t)), pi**(S(1)/2)*s**(-S(5)/2)*(s/2+a)*exp(a/s),
        #   S.true, S.Zero, dco),  # 4.9.37
        # (sinh(2*sqrt(a*t))/sqrt(t),
        #  pi**(S(1)/2)*s**(-S(1)/2)*exp(a/s) * erf(sqrt(a/s)),
        #     S.true, S.Zero, dco),  # 4.9.38
        # (cosh(2*sqrt(a*t))/sqrt(t), pi**(S(1)/2)*s**(-S(1)/2)*exp(a/s),
        #  S.true, S.Zero, dco),  # 4.9.39
        # (sinh(sqrt(a*t))**2/sqrt(t), pi**(S(1)/2)/2*s**(-S(1)/2)*(exp(a/s)-1),
        #  S.true, S.Zero, dco),  # 4.9.40
        # (cosh(sqrt(a*t))**2/sqrt(t), pi**(S(1)/2)/2*s**(-S(1)/2)*(exp(a/s)+1),
        #  S.true, S.Zero, dco),  # 4.9.41
        (erf(a*t), exp(s**2/(2*a)**2)*erfc(s/(2*a))/s,
         4*Abs(arg(a)) < pi, S.Zero, dco),  # 4.12.2
        # (erf(sqrt(a*t)), sqrt(a)/sqrt(s+a)/s,
        #  S.true, Max(S.Zero, -re(a)), dco),  # 4.12.4
        # (exp(a*t)*erf(sqrt(a*t)), sqrt(a)/sqrt(s)/(s-a),
        #  S.true, Max(S.Zero, re(a)), dco),  # 4.12.5
        # (erf(sqrt(a/t)/2), (1-exp(-sqrt(a*s)))/s,
        #  re(a) > 0, S.Zero, dco),  # 4.12.6
        # (erfc(sqrt(a*t)), (sqrt(s+a)-sqrt(a))/sqrt(s+a)/s,
        #  S.true, -re(a), dco),  # 4.12.9
        # (exp(a*t)*erfc(sqrt(a*t)), 1/(s+sqrt(a*s)),
        #  S.true, S.Zero, dco),  # 4.12.10
        # (erfc(sqrt(a/t)/2), exp(-sqrt(a*s))/s,
        #  re(a) > 0, S.Zero, dco),  # 4.2.11
        (besselj(n, a*t), a**n/(sqrt(s**2+a**2)*(s+sqrt(s**2+a**2))**n),
         re(n) > -1, Abs(im(a)), dco),  # 4.14.1
        (t**b*besselj(n, a*t),
         2**n/sqrt(pi)*gamma(n+S.Half)*a**n*(s**2+a**2)**(-n-S.Half),
         And(re(n) > -S.Half, Eq(b, n)), Abs(im(a)), dco),  # 4.14.7
        (t**b*besselj(n, a*t),
         2**(n+1)/sqrt(pi)*gamma(n+S(3)/2)*a**n*s*(s**2+a**2)**(-n-S(3)/2),
         And(re(n) > -1, Eq(b, n+1)), Abs(im(a)), dco),  # 4.14.8
        # (besselj(0, 2*sqrt(a*t)), exp(-a/s)/s,
        #  S.true, S.Zero, dco),  # 4.14.25
        # (t**(b)*besselj(n, 2*sqrt(a*t)), a**(n/2)*s**(-n-1)*exp(-a/s),
        #  And(re(n) > -1, Eq(b, n*S.Half)), S.Zero, dco),  # 4.14.30
        (besselj(0, a*sqrt(t**2+b*t)),
         exp(b*s-b*sqrt(s**2+a**2))/sqrt(s**2+a**2),
         Abs(arg(b)) < pi, Abs(im(a)), dco),  # 4.15.19
        (besseli(n, a*t), a**n/(sqrt(s**2-a**2)*(s+sqrt(s**2-a**2))**n),
         re(n) > -1, Abs(re(a)), dco),  # 4.16.1
        (t**b*besseli(n, a*t),
         2**n/sqrt(pi)*gamma(n+S.Half)*a**n*(s**2-a**2)**(-n-S.Half),
         And(re(n) > -S.Half, Eq(b, n)), Abs(re(a)), dco),  # 4.16.6
        (t**b*besseli(n, a*t),
         2**(n+1)/sqrt(pi)*gamma(n+S(3)/2)*a**n*s*(s**2-a**2)**(-n-S(3)/2),
         And(re(n) > -1, Eq(b, n+1)), Abs(re(a)), dco),  # 4.16.7
        # (t**(b)*besseli(n, 2*sqrt(a*t)), a**(n/2)*s**(-n-1)*exp(a/s),
        #  And(re(n) > -1, Eq(b, n*S.Half)), S.Zero, dco),  # 4.16.18
        (bessely(0, a*t), -2/pi*asinh(s/a)/sqrt(s**2+a**2),
         S.true, Abs(im(a)), dco),  # 4.15.44
        (besselk(0, a*t), log((s + sqrt(s**2-a**2))/a)/(sqrt(s**2-a**2)),
         S.true, -re(a), dco)  # 4.16.23
    ]
    return laplace_transform_rules, t, s


@DEBUG_WRAP
def _laplace_rule_timescale(f, t, s):
    """
    This function applies the time-scaling rule of the Laplace transform in
    a straight-forward way. For example, if it gets ``(f(a*t), t, s)``, it will
    compute ``LaplaceTransform(f(t)/a, t, s/a)`` if ``a>0``.
    """

    a = Wild('a', exclude=[t])
    g = WildFunction('g', nargs=1)
    ma1 = f.match(g)
    if ma1:
        arg = ma1[g].args[0].collect(t)
        ma2 = arg.match(a*t)
        if ma2 and ma2[a].is_positive and ma2[a] != 1:
            _debug('     rule: time scaling (4.1.4)')
            r, pr, cr = _laplace_transform(
                1/ma2[a]*ma1[g].func(t), t, s/ma2[a], simplify=False)
            return (r, pr, cr)
    return None


@DEBUG_WRAP
def _laplace_rule_heaviside(f, t, s):
    """
    This function deals with time-shifted Heaviside step functions. If the time
    shift is positive, it applies the time-shift rule of the Laplace transform.
    For example, if it gets ``(Heaviside(t-a)*f(t), t, s)``, it will compute
    ``exp(-a*s)*LaplaceTransform(f(t+a), t, s)``.

    If the time shift is negative, the Heaviside function is simply removed
    as it means nothing to the Laplace transform.

    The function does not remove a factor ``Heaviside(t)``; this is done by
    the simple rules.
    """

    a = Wild('a', exclude=[t])
    y = Wild('y')
    g = Wild('g')
    if ma1 := f.match(Heaviside(y) * g):
        if ma2 := ma1[y].match(t - a):
            if ma2[a].is_positive:
                _debug('     rule: time shift (4.1.4)')
                r, pr, cr = _laplace_transform(
                    ma1[g].subs(t, t + ma2[a]), t, s, simplify=False)
                return (exp(-ma2[a] * s) * r, pr, cr)
            if ma2[a].is_negative:
                _debug(
                    '     rule: Heaviside factor; negative time shift (4.1.4)')
                r, pr, cr = _laplace_transform(ma1[g], t, s, simplify=False)
                return (r, pr, cr)
        if ma2 := ma1[y].match(a - t):
            if ma2[a].is_positive:
                _debug('     rule: Heaviside window open')
                r, pr, cr = _laplace_transform(
                    (1 - Heaviside(t - ma2[a])) * ma1[g], t, s, simplify=False)
                return (r, pr, cr)
            if ma2[a].is_negative:
                _debug('     rule: Heaviside window closed')
                return (0, 0, S.true)
    return None


@DEBUG_WRAP
def _laplace_rule_exp(f, t, s):
    """
    If this function finds a factor ``exp(a*t)``, it applies the
    frequency-shift rule of the Laplace transform and adjusts the convergence
    plane accordingly.  For example, if it gets ``(exp(-a*t)*f(t), t, s)``, it
    will compute ``LaplaceTransform(f(t), t, s+a)``.
    """

    a = Wild('a', exclude=[t])
    y = Wild('y')
    z = Wild('z')
    ma1 = f.match(exp(y)*z)
    if ma1:
        ma2 = ma1[y].collect(t).match(a*t)
        if ma2:
            _debug('     rule: multiply with exp (4.1.5)')
            r, pr, cr = _laplace_transform(ma1[z], t, s-ma2[a],
                                           simplify=False)
            return (r, pr+re(ma2[a]), cr)
    return None


@DEBUG_WRAP
def _laplace_rule_delta(f, t, s):
    """
    If this function finds a factor ``DiracDelta(b*t-a)``, it applies the
    masking property of the delta distribution. For example, if it gets
    ``(DiracDelta(t-a)*f(t), t, s)``, it will return
    ``(f(a)*exp(-a*s), -a, True)``.
    """
    # This rule is not in Bateman54

    a = Wild('a', exclude=[t])
    b = Wild('b', exclude=[t])

    y = Wild('y')
    z = Wild('z')
    ma1 = f.match(DiracDelta(y)*z)
    if ma1 and not ma1[z].has(DiracDelta):
        ma2 = ma1[y].collect(t).match(b*t-a)
        if ma2:
            _debug('     rule: multiply with DiracDelta')
            loc = ma2[a]/ma2[b]
            if re(loc) >= 0 and im(loc) == 0:
                fn = exp(-ma2[a]/ma2[b]*s)*ma1[z]
                if fn.has(sin, cos):
                    # Then it may be possible that a sinc() is present in the
                    # term; let's try this:
                    fn = fn.rewrite(sinc).ratsimp()
                n, d = [x.subs(t, ma2[a]/ma2[b]) for x in fn.as_numer_denom()]
                if d != 0:
                    return (n/d/ma2[b], S.NegativeInfinity, S.true)
                else:
                    return None
            else:
                return (0, S.NegativeInfinity, S.true)
        if ma1[y].is_polynomial(t):
            ro = roots(ma1[y], t)
            if ro != {} and set(ro.values()) == {1}:
                slope = diff(ma1[y], t)
                r = Add(
                    *[exp(-x*s)*ma1[z].subs(t, s)/slope.subs(t, x)
                      for x in list(ro.keys()) if im(x) == 0 and re(x) >= 0])
                return (r, S.NegativeInfinity, S.true)
    return None


@DEBUG_WRAP
def _laplace_trig_split(fn):
    """
    Helper function for `_laplace_rule_trig`.  This function returns two terms
    `f` and `g`.  `f` contains all product terms with sin, cos, sinh, cosh in
    them; `g` contains everything else.
    """
    trigs = [S.One]
    other = [S.One]
    for term in Mul.make_args(fn):
        if term.has(sin, cos, sinh, cosh, exp):
            trigs.append(term)
        else:
            other.append(term)
    f = Mul(*trigs)
    g = Mul(*other)
    return f, g


@DEBUG_WRAP
def _laplace_trig_expsum(f, t):
    """
    Helper function for `_laplace_rule_trig`.  This function expects the `f`
    from `_laplace_trig_split`.  It returns two lists `xm` and `xn`.  `xm` is
    a list of dictionaries with keys `k` and `a` representing a function
    `k*exp(a*t)`.  `xn` is a list of all terms that cannot be brought into
    that form, which may happen, e.g., when a trigonometric function has
    another function in its argument.
    """
    c1 = Wild('c1', exclude=[t])
    c0 = Wild('c0', exclude=[t])
    p = Wild('p', exclude=[t])
    xm = []
    xn = []

    x1 = f.rewrite(exp).expand()

    for term in Add.make_args(x1):
        if not term.has(t):
            xm.append({'k': term, 'a': 0, re: 0, im: 0})
            continue
        term = _laplace_deep_collect(term.powsimp(combine='exp'), t)

        if (r := term.match(p*exp(c1*t+c0))) is not None:
            xm.append({
                'k': r[p]*exp(r[c0]), 'a': r[c1],
                re: re(r[c1]), im: im(r[c1])})
        else:
            xn.append(term)
    return xm, xn


@DEBUG_WRAP
def _laplace_trig_ltex(xm, t, s):
    """
    Helper function for `_laplace_rule_trig`.  This function takes the list of
    exponentials `xm` from `_laplace_trig_expsum` and simplifies complex
    conjugate and real symmetric poles.  It returns the result as a sum and
    the convergence plane.
    """
    results = []
    planes = []

    def _simpc(coeffs):
        nc = coeffs.copy()
        for k in range(len(nc)):
            ri = nc[k].as_real_imag()
            if ri[0].has(im):
                nc[k] = nc[k].rewrite(cos)
            else:
                nc[k] = (ri[0] + I*ri[1]).rewrite(cos)
        return nc

    def _quadpole(t1, k1, k2, k3, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [
            k0 + k1 + k2 + k3,
            a*(k0 + k1 - k2 - k3) - 2*I*a_i*k1 + 2*I*a_i*k2,
            (
                a**2*(-k0 - k1 - k2 - k3) +
                a*(4*I*a_i*k0 + 4*I*a_i*k3) +
                4*a_i**2*k0 + 4*a_i**2*k3),
            (
                a**3*(-k0 - k1 + k2 + k3) +
                a**2*(4*I*a_i*k0 + 2*I*a_i*k1 - 2*I*a_i*k2 - 4*I*a_i*k3) +
                a*(4*a_i**2*k0 - 4*a_i**2*k3))
        ]
        dc = [
            S.One, S.Zero, 2*a_i**2 - 2*a_r**2,
            S.Zero, a_i**4 + 2*a_i**2*a_r**2 + a_r**4]
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    def _ccpole(t1, k1, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [k0 + k1, -a*k0 - a*k1 + 2*I*a_i*k0]
        dc = [S.One, -2*a_r, a_i**2 + a_r**2]
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    def _rspole(t1, k2, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [k0 + k2, a*k0 - a*k2 - 2*I*a_i*k0]
        dc = [S.One, -2*I*a_i, -a_i**2 - a_r**2]
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    def _sypole(t1, k3, s):
        a, k0 = t1['a'], t1['k']
        nc = [k0 + k3, a*(k0 - k3)]
        dc = [S.One, S.Zero, -a**2]
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    def _simplepole(t1, s):
        a, k0 = t1['a'], t1['k']
        n = k0
        d = s - a
        return n/d

    while len(xm) > 0:
        t1 = xm.pop()
        i_imagsym = None
        i_realsym = None
        i_pointsym = None
        # The following code checks all remaining poles. If t1 is a pole at
        # a+b*I, then we check for a-b*I, -a+b*I, and -a-b*I, and
        # assign the respective indices to i_imagsym, i_realsym, i_pointsym.
        # -a-b*I / i_pointsym only applies if both a and b are != 0.
        for i in range(len(xm)):
            real_eq = t1[re] == xm[i][re]
            realsym = t1[re] == -xm[i][re]
            imag_eq = t1[im] == xm[i][im]
            imagsym = t1[im] == -xm[i][im]
            if realsym and imagsym and t1[re] != 0 and t1[im] != 0:
                i_pointsym = i
            elif realsym and imag_eq and t1[re] != 0:
                i_realsym = i
            elif real_eq and imagsym and t1[im] != 0:
                i_imagsym = i

        # The next part looks for four possible pole constellations:
        # quad:   a+b*I, a-b*I, -a+b*I, -a-b*I
        # cc:     a+b*I, a-b*I (a may be zero)
        # quad:   a+b*I, -a+b*I (b may be zero)
        # point:  a+b*I, -a-b*I (a!=0 and b!=0 is needed, but that has been
        #                        asserted when finding i_pointsym above.)
        # If none apply, then t1 is a simple pole.
        if (
                i_imagsym is not None and i_realsym is not None
                and i_pointsym is not None):
            results.append(
                _quadpole(t1,
                          xm[i_imagsym]['k'], xm[i_realsym]['k'],
                          xm[i_pointsym]['k'], s))
            planes.append(Abs(re(t1['a'])))
            # The three additional poles have now been used; to pop them
            # easily we have to do it from the back.
            indices_to_pop = [i_imagsym, i_realsym, i_pointsym]
            indices_to_pop.sort(reverse=True)
            for i in indices_to_pop:
                xm.pop(i)
        elif i_imagsym is not None:
            results.append(_ccpole(t1, xm[i_imagsym]['k'], s))
            planes.append(t1[re])
            xm.pop(i_imagsym)
        elif i_realsym is not None:
            results.append(_rspole(t1, xm[i_realsym]['k'], s))
            planes.append(Abs(t1[re]))
            xm.pop(i_realsym)
        elif i_pointsym is not None:
            results.append(_sypole(t1, xm[i_pointsym]['k'], s))
            planes.append(Abs(t1[re]))
            xm.pop(i_pointsym)
        else:
            results.append(_simplepole(t1, s))
            planes.append(t1[re])

    return Add(*results), Max(*planes)


@DEBUG_WRAP
def _laplace_rule_trig(fn, t_, s):
    """
    This rule covers trigonometric factors by splitting everything into a
    sum of exponential functions and collecting complex conjugate poles and
    real symmetric poles.
    """
    t = Dummy('t', real=True)

    if not fn.has(sin, cos, sinh, cosh):
        return None

    f, g = _laplace_trig_split(fn.subs(t_, t))
    xm, xn = _laplace_trig_expsum(f, t)

    if len(xn) > 0:
        # TODO not implemented yet, but also not important
        return None

    if not g.has(t):
        r, p = _laplace_trig_ltex(xm, t, s)
        return g*r, p, S.true
    else:
        # Just transform `g` and make frequency-shifted copies
        planes = []
        results = []
        G, G_plane, G_cond = _laplace_transform(g, t, s, simplify=False)
        for x1 in xm:
            results.append(x1['k']*G.subs(s, s-x1['a']))
            planes.append(G_plane+re(x1['a']))
    return Add(*results).subs(t, t_), Max(*planes), G_cond


@DEBUG_WRAP
def _laplace_rule_diff(f, t, s):
    """
    This function looks for derivatives in the time domain and replaces it
    by factors of `s` and initial conditions in the frequency domain. For
    example, if it gets ``(diff(f(t), t), t, s)``, it will compute
    ``s*LaplaceTransform(f(t), t, s) - f(0)``.
    """

    a = Wild('a', exclude=[t])
    n = Wild('n', exclude=[t])
    g = WildFunction('g')
    ma1 = f.match(a*Derivative(g, (t, n)))
    if ma1 and ma1[n].is_integer:
        m = [z.has(t) for z in ma1[g].args]
        if sum(m) == 1:
            _debug('     rule: time derivative (4.1.8)')
            d = []
            for k in range(ma1[n]):
                if k == 0:
                    y = ma1[g].subs(t, 0)
                else:
                    y = Derivative(ma1[g], (t, k)).subs(t, 0)
                d.append(s**(ma1[n]-k-1)*y)
            r, pr, cr = _laplace_transform(ma1[g], t, s, simplify=False)
            return (ma1[a]*(s**ma1[n]*r - Add(*d)),  pr, cr)
    return None


@DEBUG_WRAP
def _laplace_rule_sdiff(f, t, s):
    """
    This function looks for multiplications with polynoimials in `t` as they
    correspond to differentiation in the frequency domain. For example, if it
    gets ``(t*f(t), t, s)``, it will compute
    ``-Derivative(LaplaceTransform(f(t), t, s), s)``.
    """

    if f.is_Mul:
        pfac = [1]
        ofac = [1]
        for fac in Mul.make_args(f):
            if fac.is_polynomial(t):
                pfac.append(fac)
            else:
                ofac.append(fac)
        if len(pfac) > 1:
            pex = prod(pfac)
            pc = Poly(pex, t).all_coeffs()
            N = len(pc)
            if N > 1:
                oex = prod(ofac)
                r_, p_, c_ = _laplace_transform(oex, t, s, simplify=False)
                deri = [r_]
                d1 = False
                try:
                    d1 = -diff(deri[-1], s)
                except ValueError:
                    d1 = False
                if r_.has(LaplaceTransform):
                    for k in range(N-1):
                        deri.append((-1)**(k+1)*Derivative(r_, s, k+1))
                elif d1:
                    deri.append(d1)
                    for k in range(N-2):
                        deri.append(-diff(deri[-1], s))
                if d1:
                    r = Add(*[pc[N-n-1]*deri[n] for n in range(N)])
                    return (r, p_, c_)
    # We still have to cover the possibility that there is a symbolic positive
    # integer exponent.
    n = Wild('n', exclude=[t])
    g = Wild('g')
    if ma1 := f.match(t**n*g):
        if ma1[n].is_integer and ma1[n].is_positive:
            r_, p_, c_ = _laplace_transform(ma1[g], t, s, simplify=False)
            return (-1)**ma1[n]*diff(r_, (s, ma1[n])), p_, c_
    return None


@DEBUG_WRAP
def _laplace_expand(f, t, s):
    """
    This function tries to expand its argument with successively stronger
    methods: first it will expand on the top level, then it will expand any
    multiplications in depth, then it will try all available expansion methods,
    and finally it will try to expand trigonometric functions.

    If it can expand, it will then compute the Laplace transform of the
    expanded term.
    """

    r = expand(f, deep=False)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand_mul(f)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand(f)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    if r != f:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand(expand_trig(f))
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    return None


@DEBUG_WRAP
def _laplace_apply_prog_rules(f, t, s):
    """
    This function applies all program rules and returns the result if one
    of them gives a result.
    """

    prog_rules = [_laplace_rule_heaviside, _laplace_rule_delta,
                  _laplace_rule_timescale, _laplace_rule_exp,
                  _laplace_rule_trig,
                  _laplace_rule_diff, _laplace_rule_sdiff]

    for p_rule in prog_rules:
        if (L := p_rule(f, t, s)) is not None:
            return L
    return None


@DEBUG_WRAP
def _laplace_apply_simple_rules(f, t, s):
    """
    This function applies all simple rules and returns the result if one
    of them gives a result.
    """
    simple_rules, t_, s_ = _laplace_build_rules()
    prep_old = ''
    prep_f = ''
    for t_dom, s_dom, check, plane, prep in simple_rules:
        if prep_old != prep:
            prep_f = prep(f.subs({t: t_}))
            prep_old = prep
        ma = prep_f.match(t_dom)
        if ma:
            try:
                c = check.xreplace(ma)
            except TypeError:
                # This may happen if the time function has imaginary
                # numbers in it. Then we give up.
                continue
            if c == S.true:
                return (s_dom.xreplace(ma).subs({s_: s}),
                        plane.xreplace(ma), S.true)
    return None


@DEBUG_WRAP
def _piecewise_to_heaviside(f, t):
    """
    This function converts a Piecewise expression to an expression written
    with Heaviside. It is not exact, but valid in the context of the Laplace
    transform.
    """
    if not t.is_real:
        r = Dummy('r', real=True)
        return _piecewise_to_heaviside(f.xreplace({t: r}), r).xreplace({r: t})
    x = piecewise_exclusive(f)
    r = []
    for fn, cond in x.args:
        # Here we do not need to do many checks because piecewise_exclusive
        # has a clearly predictable output. However, if any of the conditions
        # is not relative to t, this function just returns the input argument.
        if isinstance(cond, Relational) and t in cond.args:
            if isinstance(cond, (Eq, Ne)):
                # We do not cover this case; these would be single-point
                # exceptions that do not play a role in Laplace practice,
                # except if they contain Dirac impulses, and then we can
                # expect users to not try to use Piecewise for writing it.
                return f
            else:
                r.append(Heaviside(cond.gts - cond.lts)*fn)
        elif isinstance(cond, Or) and len(cond.args) == 2:
            # Or(t<2, t>4), Or(t>4, t<=2), ... in any order with any <= >=
            for c2 in cond.args:
                if c2.lhs == t:
                    r.append(Heaviside(c2.gts - c2.lts)*fn)
                else:
                    return f
        elif isinstance(cond, And) and len(cond.args) == 2:
            # And(t>2, t<4), And(t>4, t<=2), ...  in any order with any <= >=
            c0, c1 = cond.args
            if c0.lhs == t and c1.lhs == t:
                if '>' in c0.rel_op:
                    c0, c1 = c1, c0
                r.append(
                    (Heaviside(c1.gts - c1.lts) -
                     Heaviside(c0.lts - c0.gts))*fn)
            else:
                return f
        else:
            return f
    return Add(*r)


def laplace_correspondence(f, fdict, /):
    """
    This helper function takes a function `f` that is the result of a
    ``laplace_transform`` or an ``inverse_laplace_transform``.  It replaces all
    unevaluated ``LaplaceTransform(y(t), t, s)`` by `Y(s)` for any `s` and
    all ``InverseLaplaceTransform(Y(s), s, t)`` by `y(t)` for any `t` if
    ``fdict`` contains a correspondence ``{y: Y}``.

    Parameters
    ==========

    f : sympy expression
        Expression containing unevaluated ``LaplaceTransform`` or
        ``LaplaceTransform`` objects.
    fdict : dictionary
        Dictionary containing one or more function correspondences,
        e.g., ``{x: X, y: Y}`` meaning that ``X`` and ``Y`` are the
        Laplace transforms of ``x`` and ``y``, respectively.

    Examples
    ========

    >>> from sympy import laplace_transform, diff, Function
    >>> from sympy import laplace_correspondence, inverse_laplace_transform
    >>> from sympy.abc import t, s
    >>> y = Function("y")
    >>> Y = Function("Y")
    >>> z = Function("z")
    >>> Z = Function("Z")
    >>> f = laplace_transform(diff(y(t), t, 1) + z(t), t, s, noconds=True)
    >>> laplace_correspondence(f, {y: Y, z: Z})
    s*Y(s) + Z(s) - y(0)
    >>> f = inverse_laplace_transform(Y(s), s, t)
    >>> laplace_correspondence(f, {y: Y})
    y(t)
    """
    p = Wild('p')
    s = Wild('s')
    t = Wild('t')
    a = Wild('a')
    if (
            not isinstance(f, Expr)
            or (not f.has(LaplaceTransform)
                and not f.has(InverseLaplaceTransform))):
        return f
    for y, Y in fdict.items():
        if (
                (m := f.match(LaplaceTransform(y(a), t, s))) is not None
                and m[a] == m[t]):
            return Y(m[s])
        if (
                (m := f.match(InverseLaplaceTransform(Y(a), s, t, p)))
                is not None
                and m[a] == m[s]):
            return y(m[t])
    func = f.func
    args = [laplace_correspondence(arg, fdict) for arg in f.args]
    return func(*args)


def laplace_initial_conds(f, t, fdict, /):
    """
    This helper function takes a function `f` that is the result of a
    ``laplace_transform``.  It takes an fdict of the form ``{y: [1, 4, 2]}``,
    where the values in the list are the initial value, the initial slope, the
    initial second derivative, etc., of the function `y(t)`, and replaces all
    unevaluated initial conditions.

    Parameters
    ==========

    f : sympy expression
        Expression containing initial conditions of unevaluated functions.
    t : sympy expression
        Variable for which the initial conditions are to be applied.
    fdict : dictionary
        Dictionary containing a list of initial conditions for every
        function, e.g., ``{y: [0, 1, 2], x: [3, 4, 5]}``. The order
        of derivatives is ascending, so `0`, `1`, `2` are `y(0)`, `y'(0)`,
        and `y''(0)`, respectively.

    Examples
    ========

    >>> from sympy import laplace_transform, diff, Function
    >>> from sympy import laplace_correspondence, laplace_initial_conds
    >>> from sympy.abc import t, s
    >>> y = Function("y")
    >>> Y = Function("Y")
    >>> f = laplace_transform(diff(y(t), t, 3), t, s, noconds=True)
    >>> g = laplace_correspondence(f, {y: Y})
    >>> laplace_initial_conds(g, t, {y: [2, 4, 8, 16, 32]})
    s**3*Y(s) - 2*s**2 - 4*s - 8
    """
    for y, ic in fdict.items():
        for k in range(len(ic)):
            if k == 0:
                f = f.replace(y(0), ic[0])
            elif k == 1:
                f = f.replace(Subs(Derivative(y(t), t), t, 0), ic[1])
            else:
                f = f.replace(Subs(Derivative(y(t), (t, k)), t, 0), ic[k])
    return f


@DEBUG_WRAP
def _laplace_transform(fn, t_, s_, *, simplify):
    """
    Front-end function of the Laplace transform. It tries to apply all known
    rules recursively, and if everything else fails, it tries to integrate.
    """

    terms_t = Add.make_args(fn)
    terms_s = []
    terms = []
    planes = []
    conditions = []

    for ff in terms_t:
        k, ft = ff.as_independent(t_, as_Add=False)
        if ft.has(SingularityFunction):
            _terms = Add.make_args(ft.rewrite(Heaviside))
            for _term in _terms:
                k1, f1 = _term.as_independent(t_, as_Add=False)
                terms.append((k*k1, f1))
        elif ft.func == Piecewise and not ft.has(DiracDelta(t_)):
            _terms = Add.make_args(_piecewise_to_heaviside(ft, t_))
            for _term in _terms:
                k1, f1 = _term.as_independent(t_, as_Add=False)
                terms.append((k*k1, f1))
        else:
            terms.append((k, ft))

    for k, ft in terms:
        if ft.has(SingularityFunction):
            r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
        else:
            if ft.has(Heaviside(t_)) and not ft.has(DiracDelta(t_)):
                # For t>=0, Heaviside(t)=1 can be used, except if there is also
                # a DiracDelta(t) present, in which case removing Heaviside(t)
                # is unnecessary because _laplace_rule_delta can deal with it.
                ft = ft.subs(Heaviside(t_), 1)
            if (
                    (r := _laplace_apply_simple_rules(ft, t_, s_))
                    is not None or
                    (r := _laplace_apply_prog_rules(ft, t_, s_))
                    is not None or
                    (r := _laplace_expand(ft, t_, s_)) is not None):
                pass
            elif any(undef.has(t_) for undef in ft.atoms(AppliedUndef)):
                # If there are undefined functions f(t) then integration is
                # unlikely to do anything useful so we skip it and given an
                # unevaluated LaplaceTransform.
                r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
            elif (r := _laplace_transform_integration(
                    ft, t_, s_, simplify=simplify)) is not None:
                pass
            else:
                r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
        (ri_, pi_, ci_) = r
        terms_s.append(k*ri_)
        planes.append(pi_)
        conditions.append(ci_)

    result = Add(*terms_s)
    if simplify:
        result = result.simplify(doit=False)
    plane = Max(*planes)
    condition = And(*conditions)

    return result, plane, condition


class LaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute Laplace transforms, see the :func:`laplace_transform`
    docstring.

    If this is called with ``.doit()``, it returns the Laplace transform as an
    expression. If it is called with ``.doit(noconds=False)``, it returns a
    tuple containing the same expression, a convergence plane, and conditions.
    """

    _name = 'Laplace'

    def _compute_transform(self, f, t, s, **hints):
        _simplify = hints.get('simplify', False)
        LT = _laplace_transform_integration(f, t, s, simplify=_simplify)
        return LT

    def _as_integral(self, f, t, s):
        return Integral(f*exp(-s*t), (t, S.Zero, S.Infinity))

    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        Standard hints are the following:
        - ``noconds``:  if True, do not return convergence conditions. The
        default setting is `True`.
        - ``simplify``: if True, it simplifies the final result. The
        default setting is `False`.
        """
        _noconds = hints.get('noconds', True)
        _simplify = hints.get('simplify', False)

        debugf('[LT doit] (%s, %s, %s)', (self.function,
                                          self.function_variable,
                                          self.transform_variable))

        t_ = self.function_variable
        s_ = self.transform_variable
        fn = self.function

        r = _laplace_transform(fn, t_, s_, simplify=_simplify)

        if _noconds:
            return r[0]
        else:
            return r


def laplace_transform(f, t, s, legacy_matrix=True, **hints):
    r"""
    Compute the Laplace Transform `F(s)` of `f(t)`,

    .. math :: F(s) = \int_{0^{-}}^\infty e^{-st} f(t) \mathrm{d}t.

    Explanation
    ===========

    For all sensible functions, this converges absolutely in a
    half-plane

    .. math :: a < \operatorname{Re}(s)

    This function returns ``(F, a, cond)`` where ``F`` is the Laplace
    transform of ``f``, `a` is the half-plane of convergence, and `cond` are
    auxiliary convergence conditions.

    The implementation is rule-based, and if you are interested in which
    rules are applied, and whether integration is attempted, you can switch
    debug information on by setting ``sympy.SYMPY_DEBUG=True``. The numbers
    of the rules in the debug information (and the code) refer to Bateman's
    Tables of Integral Transforms [1].

    The lower bound is `0-`, meaning that this bound should be approached
    from the lower side. This is only necessary if distributions are involved.
    At present, it is only done if `f(t)` contains ``DiracDelta``, in which
    case the Laplace transform is computed implicitly as

    .. math ::
        F(s) = \lim_{\tau\to 0^{-}} \int_{\tau}^\infty e^{-st}
        f(t) \mathrm{d}t

    by applying rules.

    If the Laplace transform cannot be fully computed in closed form, this
    function returns expressions containing unevaluated
    :class:`LaplaceTransform` objects.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`. If
    ``noconds=True``, only `F` will be returned (i.e. not ``cond``, and also
    not the plane ``a``).

    .. deprecated:: 1.9
        Legacy behavior for matrices where ``laplace_transform`` with
        ``noconds=False`` (the default) returns a Matrix whose elements are
        tuples. The behavior of ``laplace_transform`` for matrices will change
        in a future release of SymPy to return a tuple of the transformed
        Matrix and the convergence conditions for the matrix as a whole. Use
        ``legacy_matrix=False`` to enable the new behavior.

    Examples
    ========

    >>> from sympy import DiracDelta, exp, laplace_transform
    >>> from sympy.abc import t, s, a
    >>> laplace_transform(t**4, t, s)
    (24/s**5, 0, True)
    >>> laplace_transform(t**a, t, s)
    (gamma(a + 1)/(s*s**a), 0, re(a) > -1)
    >>> laplace_transform(DiracDelta(t)-a*exp(-a*t), t, s, simplify=True)
    (s/(a + s), -re(a), True)

    There are also helper functions that make it easy to solve differential
    equations by Laplace transform. For example, to solve

    .. math :: m x''(t) + d x'(t) + k x(t) = 0

    with initial value `0` and initial derivative `v`:

    >>> from sympy import Function, laplace_correspondence, diff, solve
    >>> from sympy import laplace_initial_conds, inverse_laplace_transform
    >>> from sympy.abc import d, k, m, v
    >>> x = Function('x')
    >>> X = Function('X')
    >>> f = m*diff(x(t), t, 2) + d*diff(x(t), t) + k*x(t)
    >>> F = laplace_transform(f, t, s, noconds=True)
    >>> F = laplace_correspondence(F, {x: X})
    >>> F = laplace_initial_conds(F, t, {x: [0, v]})
    >>> F
    d*s*X(s) + k*X(s) + m*(s**2*X(s) - v)
    >>> Xs = solve(F, X(s))[0]
    >>> Xs
    m*v/(d*s + k + m*s**2)
    >>> inverse_laplace_transform(Xs, s, t)
    2*v*exp(-d*t/(2*m))*sin(t*sqrt((-d**2 + 4*k*m)/m**2)/2)*Heaviside(t)/sqrt((-d**2 + 4*k*m)/m**2)

    References
    ==========

    .. [1] Erdelyi, A. (ed.), Tables of Integral Transforms, Volume 1,
           Bateman Manuscript Prooject, McGraw-Hill (1954), available:
           https://resolver.caltech.edu/CaltechAUTHORS:20140123-101456353

    See Also
    ========

    inverse_laplace_transform, mellin_transform, fourier_transform
    hankel_transform, inverse_hankel_transform

    """

    _noconds = hints.get('noconds', False)
    _simplify = hints.get('simplify', False)

    if isinstance(f, MatrixBase) and hasattr(f, 'applyfunc'):

        conds = not hints.get('noconds', False)

        if conds and legacy_matrix:
            adt = 'deprecated-laplace-transform-matrix'
            sympy_deprecation_warning(
                """
Calling laplace_transform() on a Matrix with noconds=False (the default) is
deprecated. Either noconds=True or use legacy_matrix=False to get the new
behavior.
                """,
                deprecated_since_version='1.9',
                active_deprecations_target=adt,
            )
            # Temporarily disable the deprecation warning for non-Expr objects
            # in Matrix
            with ignore_warnings(SymPyDeprecationWarning):
                return f.applyfunc(
                    lambda fij: laplace_transform(fij, t, s, **hints))
        else:
            elements_trans = [laplace_transform(
                fij, t, s, **hints) for fij in f]
            if conds:
                elements, avals, conditions = zip(*elements_trans)
                f_laplace = type(f)(*f.shape, elements)
                return f_laplace, Max(*avals), And(*conditions)
            else:
                return type(f)(*f.shape, elements_trans)

    LT, p, c = LaplaceTransform(f, t, s).doit(noconds=False,
                                              simplify=_simplify)

    if not _noconds:
        return LT, p, c
    else:
        return LT


@DEBUG_WRAP
def _inverse_laplace_transform_integration(F, s, t_, plane, *, simplify):
    """ The backend function for inverse Laplace transforms. """
    from sympy.integrals.meijerint import meijerint_inversion, _get_coeff_exp
    from sympy.integrals.transforms import inverse_mellin_transform

    # There are two strategies we can try:
    # 1) Use inverse mellin transform, related by a simple change of variables.
    # 2) Use the inversion integral.

    t = Dummy('t', real=True)

    def pw_simp(*args):
        """ Simplify a piecewise expression from hyperexpand. """
        if len(args) != 3:
            return Piecewise(*args)
        arg = args[2].args[0].argument
        coeff, exponent = _get_coeff_exp(arg, t)
        e1 = args[0].args[0]
        e2 = args[1].args[0]
        return (
            Heaviside(1/Abs(coeff) - t**exponent)*e1 +
            Heaviside(t**exponent - 1/Abs(coeff))*e2)

    if F.is_rational_function(s):
        F = F.apart(s)

    if F.is_Add:
        f = Add(
            *[_inverse_laplace_transform_integration(X, s, t, plane, simplify)
              for X in F.args])
        return _simplify(f.subs(t, t_), simplify), True

    try:
        f, cond = inverse_mellin_transform(F, s, exp(-t), (None, S.Infinity),
                                           needeval=True, noconds=False)
    except IntegralTransformError:
        f = None

    if f is None:
        f = meijerint_inversion(F, s, t)
        if f is None:
            return None
        if f.is_Piecewise:
            f, cond = f.args[0]
            if f.has(Integral):
                return None
        else:
            cond = S.true
        f = f.replace(Piecewise, pw_simp)

    if f.is_Piecewise:
        # many of the functions called below can't work with piecewise
        # (b/c it has a bool in args)
        return f.subs(t, t_), cond

    u = Dummy('u')

    def simp_heaviside(arg, H0=S.Half):
        a = arg.subs(exp(-t), u)
        if a.has(t):
            return Heaviside(arg, H0)
        from sympy.solvers.inequalities import _solve_inequality
        rel = _solve_inequality(a > 0, u)
        if rel.lts == u:
            k = log(rel.gts)
            return Heaviside(t + k, H0)
        else:
            k = log(rel.lts)
            return Heaviside(-(t + k), H0)

    f = f.replace(Heaviside, simp_heaviside)

    def simp_exp(arg):
        return expand_complex(exp(arg))

    f = f.replace(exp, simp_exp)

    return _simplify(f.subs(t, t_), simplify), cond


@DEBUG_WRAP
def _complete_the_square_in_denom(f, s):
    from sympy.simplify.radsimp import fraction
    [n, d] = fraction(f)
    if d.is_polynomial(s):
        cf = d.as_poly(s).all_coeffs()
        if len(cf) == 3:
            a, b, c = cf
            d = a*((s+b/(2*a))**2+c/a-(b/(2*a))**2)
    return n/d


@cacheit
def _inverse_laplace_build_rules():
    """
    This is an internal helper function that returns the table of inverse
    Laplace transform rules in terms of the time variable `t` and the
    frequency variable `s`.  It is used by `_inverse_laplace_apply_rules`.
    """
    s = Dummy('s')
    t = Dummy('t')
    a = Wild('a', exclude=[s])
    b = Wild('b', exclude=[s])
    c = Wild('c', exclude=[s])

    _debug('_inverse_laplace_build_rules is building rules')

    def _frac(f, s):
        try:
            return f.factor(s)
        except PolynomialError:
            return f

    def same(f): return f
    # This list is sorted according to the prep function needed.
    _ILT_rules = [
        (a/s, a, S.true, same, 1),
        (
            b*(s+a)**(-c), t**(c-1)*exp(-a*t)/gamma(c),
            S.true, same, 1),
        (1/(s**2+a**2)**2, (sin(a*t) - a*t*cos(a*t))/(2*a**3),
         S.true, same, 1),
        # The next two rules must be there in that order. For the second
        # one, the condition would be a != 0 or, respectively, to take the
        # limit a -> 0 after the transform if a == 0. It is much simpler if
        # the case a == 0 has its own rule.
        (1/(s**b), t**(b - 1)/gamma(b), S.true, same, 1),
        (1/(s*(s+a)**b), lowergamma(b, a*t)/(a**b*gamma(b)),
         S.true, same, 1)
    ]
    return _ILT_rules, s, t


@DEBUG_WRAP
def _inverse_laplace_apply_simple_rules(f, s, t):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    if f == 1:
        _debug('     rule: 1 o---o DiracDelta()')
        return DiracDelta(t), S.true

    _ILT_rules, s_, t_ = _inverse_laplace_build_rules()
    _prep = ''
    fsubs = f.subs({s: s_})

    for s_dom, t_dom, check, prep, fac in _ILT_rules:
        if _prep != (prep, fac):
            _F = prep(fsubs*fac)
            _prep = (prep, fac)
        ma = _F.match(s_dom)
        if ma:
            c = check
            if c is not S.true:
                args = [x.xreplace(ma) for x in c[0]]
                c = c[1](*args)
            if c == S.true:
                return Heaviside(t)*t_dom.xreplace(ma).subs({t_: t}), S.true

    return None


@DEBUG_WRAP
def _inverse_laplace_diff(f, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    a = Wild('a', exclude=[s])
    n = Wild('n', exclude=[s])
    g = Wild('g')
    ma = f.match(a*Derivative(g, (s, n)))
    if ma and ma[n].is_integer:
        _debug('     rule: t**n*f(t) o---o (-1)**n*diff(F(s), s, n)')
        r, c = _inverse_laplace_transform(
            ma[g], s, t, plane, simplify=False, dorational=False)
        return (-t)**ma[n]*r, c
    return None


@DEBUG_WRAP
def _inverse_laplace_time_shift(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    a = Wild('a', exclude=[s])
    g = Wild('g')

    if not F.has(s):
        return F*DiracDelta(t), S.true
    if not F.has(exp):
        return None

    ma1 = F.match(exp(a*s))
    if ma1:
        if ma1[a].is_negative:
            _debug('     rule: exp(-a*s) o---o DiracDelta(t-a)')
            return DiracDelta(t+ma1[a]), S.true
        else:
            return InverseLaplaceTransform(F, s, t, plane), S.true

    ma1 = F.match(exp(a*s)*g)
    if ma1:
        if ma1[a].is_negative:
            _debug('     rule: exp(-a*s)*F(s) o---o Heaviside(t-a)*f(t-a)')
            return _inverse_laplace_transform(
                ma1[g], s, t+ma1[a], plane, simplify=False, dorational=True)
        else:
            return InverseLaplaceTransform(F, s, t, plane), S.true
    return None


@DEBUG_WRAP
def _inverse_laplace_freq_shift(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    if not F.has(s):
        return F*DiracDelta(t), S.true
    if len(args := F.args) == 1:
        a = Wild('a', exclude=[s])
        if (ma := args[0].match(s-a)) and re(ma[a]).is_positive:
            _debug('     rule: F(s-a) o---o exp(-a*t)*f(t)')
            return (
                exp(-ma[a]*t) *
                InverseLaplaceTransform(F.func(s), s, t, plane), S.true)
    return None


@DEBUG_WRAP
def _inverse_laplace_time_diff(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    n = Wild('n', exclude=[s])
    g = Wild('g')

    ma1 = F.match(s**n*g)
    if ma1 and ma1[n].is_integer and ma1[n].is_positive:
        _debug('     rule: s**n*F(s) o---o diff(f(t), t, n)')
        r, c = _inverse_laplace_transform(
            ma1[g], s, t, plane, simplify=False, dorational=True)
        r = r.replace(Heaviside(t), 1)
        if r.has(InverseLaplaceTransform):
            return diff(r, t, ma1[n]), c
        else:
            return Heaviside(t)*diff(r, t, ma1[n]), c
    return None


@DEBUG_WRAP
def _inverse_laplace_irrational(fn, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """

    a = Wild('a', exclude=[s])
    b = Wild('b', exclude=[s])
    m = Wild('m', exclude=[s])
    n = Wild('n', exclude=[s])

    result = None
    condition = S.true

    fa = fn.as_ordered_factors()

    ma = [x.match((a*s**m+b)**n) for x in fa]

    if None in ma:
        return None

    constants = S.One
    zeros = []
    poles = []
    rest = []

    for term in ma:
        if term[a] == 0:
            constants = constants*term
        elif term[n].is_positive:
            zeros.append(term)
        elif term[n].is_negative:
            poles.append(term)
        else:
            rest.append(term)

    # The code below assumes that the poles are sorted in a specific way:
    poles = sorted(poles, key=lambda x: (x[n], x[b] != 0, x[b]))
    zeros = sorted(zeros, key=lambda x: (x[n], x[b] != 0, x[b]))

    if len(rest) != 0:
        return None

    if len(poles) == 1 and len(zeros) == 0:
        if poles[0][n] == -1 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0) == 1/a0 * 1/(sqrt(s)+b0/a0)
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]*constants
            if a_.is_positive:
                result = (
                    k_/sqrt(pi)/sqrt(t) -
                    k_*a_*exp(a_**2*t)*erfc(a_*sqrt(t)))
                _debug('     rule 5.3.4')
        elif poles[0][n] == -2 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**2 == 1/a0**2 * 1/(sqrt(s)+b0/a0)**2
            a_sq = poles[0][b]/poles[0][a]
            a_ = a_sq**2
            k_ = 1/poles[0][a]**2*constants
            if a_sq.is_positive:
                result = (
                    k_*(1 - 2/sqrt(pi)*sqrt(a_)*sqrt(t) +
                        (1-2*a_*t)*exp(a_*t)*(erf(sqrt(a_)*sqrt(t))-1)))
                _debug('     rule 5.3.10')
        elif poles[0][n] == -3 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**3 == 1/a0**3 * 1/(sqrt(s)+b0/a0)**3
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]**3*constants
            if a_.is_positive:
                result = (
                    k_*(2/sqrt(pi)*(a_**2*t+1)*sqrt(t) -
                        a_*t*exp(a_**2*t)*(2*a_**2*t+3)*erfc(a_*sqrt(t))))
                _debug('     rule 5.3.13')
        elif poles[0][n] == -4 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**4 == 1/a0**4 * 1/(sqrt(s)+b0/a0)**4
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]**4*constants/3
            if a_.is_positive:
                result = (
                    k_*(t*(4*a_**4*t**2+12*a_**2*t+3)*exp(a_**2*t) *
                        erfc(a_*sqrt(t)) -
                        2/sqrt(pi)*a_**3*t**(S(5)/2)*(2*a_**2*t+5)))
                _debug('     rule 5.3.16')
        elif poles[0][n] == -S.Half and poles[0][m] == 2:
            # 1/sqrt(a0*s**2+b0) == 1/sqrt(a0) * 1/sqrt(s**2+b0/a0)
            a_ = sqrt(poles[0][b]/poles[0][a])
            k_ = 1/sqrt(poles[0][a])*constants
            result = (k_*(besselj(0, a_*t)))
            _debug('     rule 5.3.35/44')

    elif len(poles) == 1 and len(zeros) == 1:
        if (
                poles[0][n] == -3 and poles[0][m] == S.Half and
                zeros[0][n] == S.Half and zeros[0][b] == 0):
            # sqrt(az*s)/(ap*sqrt(s+bp)**3)
            # == sqrt(az)/ap * sqrt(s)/(sqrt(s+bp)**3)
            a_ = poles[0][b]
            k_ = sqrt(zeros[0][a])/poles[0][a]*constants
            result = (
                k_*(2*a_**4*t**2+5*a_**2*t+1)*exp(a_**2*t) *
                erfc(a_*sqrt(t)) - 2/sqrt(pi)*a_*(a_**2*t+2)*sqrt(t))
            _debug('     rule 5.3.14')
        if (
                poles[0][n] == -1 and poles[0][m] == 1 and
                zeros[0][n] == S.Half and zeros[0][m] == 1):
            # sqrt(az*s+bz)/(ap*s+bp)
            # == sqrt(az)/ap * (sqrt(s+bz/az)/(s+bp/ap))
            a_ = zeros[0][b]/zeros[0][a]
            b_ = poles[0][b]/poles[0][a]
            k_ = sqrt(zeros[0][a])/poles[0][a]*constants
            result = (
                k_*(exp(-a_*t)/sqrt(t)/sqrt(pi)+sqrt(a_-b_) *
                    exp(-b_*t)*erf(sqrt(a_-b_)*sqrt(t))))
            _debug('     rule 5.3.22')

    elif len(poles) == 2 and len(zeros) == 0:
        if (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -S.Half and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1/((a0*s+b0)*sqrt(a1*s))
            # == 1/(a0*sqrt(a1)) * 1/((s+b0/a0)*sqrt(s))
            a_ = -poles[0][b]/poles[0][a]
            k_ = 1/sqrt(poles[1][a])/poles[0][a]*constants
            if a_.is_positive:
                result = (k_/sqrt(a_)*exp(a_*t)*erf(sqrt(a_)*sqrt(t)))
                _debug('     rule 5.3.1')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and poles[0][b] == 0 and
                poles[1][n] == -1 and poles[1][m] == S.Half):
            # 1/(a0*s*(a1*sqrt(s)+b1))
            # == 1/(a0*a1) * 1/(s*(sqrt(s)+b1/a1))
            a_ = poles[1][b]/poles[1][a]
            k_ = 1/poles[0][a]/poles[1][a]/a_*constants
            if a_.is_positive:
                result = k_*(1-exp(a_**2*t)*erfc(a_*sqrt(t)))
                _debug('     rule 5.3.5')
        elif (
                poles[0][n] == -1 and poles[0][m] == S.Half and
                poles[1][n] == -S.Half and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1/((a0*sqrt(s)+b0)*(sqrt(a1*s))
            # == 1/(a0*sqrt(a1)) * 1/((sqrt(s)+b0/a0)"sqrt(s))
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/(poles[0][a]*sqrt(poles[1][a]))*constants
            if a_.is_positive:
                result = k_*exp(a_**2*t)*erfc(a_*sqrt(t))
                _debug('     rule 5.3.7')
        elif (
                poles[0][n] == -S(3)/2 and poles[0][m] == 1 and
                poles[0][b] == 0 and poles[1][n] == -1 and
                poles[1][m] == S.Half):
            # 1/((a0**(3/2)*s**(3/2))*(a1*sqrt(s)+b1))
            # == 1/(a0**(3/2)*a1)  1/((s**(3/2))*(sqrt(s)+b1/a1))
            # Note that Bateman54 5.3 (8) is incorrect; there (sqrt(p)+a)
            # should be (sqrt(p)+a)**(-1).
            a_ = poles[1][b]/poles[1][a]
            k_ = 1/(poles[0][a]**(S(3)/2)*poles[1][a])/a_**2*constants
            if a_.is_positive:
                result = (
                    k_*(2/sqrt(pi)*a_*sqrt(t)+exp(a_**2*t)*erfc(a_*sqrt(t))-1))
                _debug('     rule 5.3.8')
        elif (
                poles[0][n] == -2 and poles[0][m] == S.Half and
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1/((a0*sqrt(s)+b0)**2*a1*s)
            # == 1/a0**2/a1 * 1/(sqrt(s)+b0/a0)**2/s
            a_sq = poles[0][b]/poles[0][a]
            a_ = a_sq**2
            k_ = 1/poles[0][a]**2/poles[1][a]*constants
            if a_sq.is_positive:
                result = (
                    k_*(1/a_ + (2*t-1/a_)*exp(a_*t)*erfc(sqrt(a_)*sqrt(t)) -
                        2/sqrt(pi)/sqrt(a_)*sqrt(t)))
                _debug('     rule 5.3.11')
        elif (
                poles[0][n] == -2 and poles[0][m] == S.Half and
                poles[1][n] == -S.Half and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1/((a0*sqrt(s)+b0)**2*sqrt(a1*s))
            # == 1/a0**2/sqrt(a1) * 1/(sqrt(s)+b0/a0)**2/sqrt(s)
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]**2/sqrt(poles[1][a])*constants
            if a_.is_positive:
                result = (
                    k_*(2/sqrt(pi)*sqrt(t) -
                        2*a_*t*exp(a_**2*t)*erfc(a_*sqrt(t))))
                _debug('     rule 5.3.12')
        elif (
                poles[0][n] == -3 and poles[0][m] == S.Half and
                poles[1][n] == -S.Half and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1 / (sqrt(a1*s)*(a0*sqrt(s+b0)**3))
            # == 1/(sqrt(a1)*a0) * 1/(sqrt(s)*(sqrt(s+b0)**3))
            a_ = poles[0][b]
            k_ = constants/sqrt(poles[1][a])/poles[0][a]
            result = k_*(
                (2*a_**2*t+1)*t*exp(a_**2*t)*erfc(a_*sqrt(t)) -
                2/sqrt(pi)*a_*t**(S(3)/2))
            _debug('     rule 5.3.15')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -S.Half and poles[1][m] == 1):
            # 1 / ( (a0*s+b0)* sqrt(a1*s+b1) )
            # == 1/(sqrt(a1)*a0) * 1 / ( (s+b0/a0)* sqrt(s+b1/a1) )
            a_ = poles[0][b]/poles[0][a]
            b_ = poles[1][b]/poles[1][a]
            k_ = constants/sqrt(poles[1][a])/poles[0][a]
            result = k_*(
                1/sqrt(b_-a_)*exp(-a_*t)*erf(sqrt(b_-a_)*sqrt(t)))
            _debug('     rule 5.3.23')

    elif len(poles) == 2 and len(zeros) == 1:
        if (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -1 and poles[1][m] == S.Half and
                zeros[0][n] == S.Half and zeros[0][m] == 1 and
                zeros[0][b] == 0):
            # sqrt(za0*s)/((a0*s+b0)*(a1*sqrt(s)+b1))
            # == sqrt(za0)/(a0*a1) * s/((s+b0/a0)*(sqrt(s)+b1/a1))
            a_sq = poles[1][b]/poles[1][a]
            a_ = a_sq**2
            b_ = -poles[0][b]/poles[0][a]
            k_ = sqrt(zeros[0][a])/poles[0][a]/poles[1][a]/(a_-b_)*constants
            if a_sq.is_positive and b_.is_positive:
                result = k_*(
                    a_*exp(a_*t)*erfc(sqrt(a_)*sqrt(t)) +
                    sqrt(a_)*sqrt(b_)*exp(b_*t)*erfc(sqrt(b_)*sqrt(t)) -
                    b_*exp(b_*t))
                _debug('     rule 5.3.6')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[0][b] == 0 and poles[1][n] == -1 and
                poles[1][m] == S.Half and zeros[0][n] == 1 and
                zeros[0][m] == S.Half):
            # (az*sqrt(s)+bz)/(a0*s*(a1*sqrt(s)+b1))
            # == az/a0/a1 * (sqrt(z)+bz/az)/(s*(sqrt(s)+b1/a1))
            a_num = zeros[0][b]/zeros[0][a]
            a_ = poles[1][b]/poles[1][a]
            if a_+a_num == 0:
                k_ = zeros[0][a]/poles[0][a]/poles[1][a]*constants
                result = k_*(
                    2*exp(a_**2*t)*erfc(a_*sqrt(t))-1)
                _debug('     rule 5.3.17')
        elif (
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[1][b] == 0 and poles[0][n] == -2 and
                poles[0][m] == S.Half and zeros[0][n] == 2 and
                zeros[0][m] == S.Half):
            # (az*sqrt(s)+bz)**2/(a1*s*(a0*sqrt(s)+b0)**2)
            # == az**2/a1/a0**2 * (sqrt(z)+bz/az)**2/(s*(sqrt(s)+b0/a0)**2)
            a_num = zeros[0][b]/zeros[0][a]
            a_ = poles[0][b]/poles[0][a]
            if a_+a_num == 0:
                k_ = zeros[0][a]**2/poles[1][a]/poles[0][a]**2*constants
                result = k_*(
                    1 + 8*a_**2*t*exp(a_**2*t)*erfc(a_*sqrt(t)) -
                    8/sqrt(pi)*a_*sqrt(t))
                _debug('     rule 5.3.18')
        elif (
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[1][b] == 0 and poles[0][n] == -3 and
                poles[0][m] == S.Half and zeros[0][n] == 3 and
                zeros[0][m] == S.Half):
            # (az*sqrt(s)+bz)**3/(a1*s*(a0*sqrt(s)+b0)**3)
            # == az**3/a1/a0**3 * (sqrt(z)+bz/az)**3/(s*(sqrt(s)+b0/a0)**3)
            a_num = zeros[0][b]/zeros[0][a]
            a_ = poles[0][b]/poles[0][a]
            if a_+a_num == 0:
                k_ = zeros[0][a]**3/poles[1][a]/poles[0][a]**3*constants
                result = k_*(
                    2*(8*a_**4*t**2+8*a_**2*t+1)*exp(a_**2*t) *
                    erfc(a_*sqrt(t))-8/sqrt(pi)*a_*sqrt(t)*(2*a_**2*t+1)-1)
                _debug('     rule 5.3.19')

    elif len(poles) == 3 and len(zeros) == 0:
        if (
                poles[0][n] == -1 and poles[0][b] == 0 and poles[0][m] == 1 and
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[2][n] == -S.Half and poles[2][m] == 1):
            # 1/((a0*s)*(a1*s+b1)*sqrt(a2*s))
            # == 1/(a0*a1*sqrt(a2)) * 1/((s)*(s+b1/a1)*sqrt(s))
            a_ = -poles[1][b]/poles[1][a]
            k_ = 1/poles[0][a]/poles[1][a]/sqrt(poles[2][a])*constants
            if a_.is_positive:
                result = k_ * (
                    a_**(-S(3)/2) * exp(a_*t) * erf(sqrt(a_)*sqrt(t)) -
                    2/a_/sqrt(pi)*sqrt(t))
                _debug('     rule 5.3.2')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -1 and poles[1][m] == S.Half and
                poles[2][n] == -S.Half and poles[2][m] == 1 and
                poles[2][b] == 0):
            # 1/((a0*s+b0)*(a1*sqrt(s)+b1)*(sqrt(a2)*sqrt(s)))
            # == 1/(a0*a1*sqrt(a2)) * 1/((s+b0/a0)*(sqrt(s)+b1/a1)*sqrt(s))
            a_sq = poles[1][b]/poles[1][a]
            a_ = a_sq**2
            b_ = -poles[0][b]/poles[0][a]
            k_ = (
                1/poles[0][a]/poles[1][a]/sqrt(poles[2][a]) /
                (sqrt(b_)*(a_-b_)))
            if a_sq.is_positive and b_.is_positive:
                result = k_ * (
                    sqrt(b_)*exp(a_*t)*erfc(sqrt(a_)*sqrt(t)) +
                    sqrt(a_)*exp(b_*t)*erf(sqrt(b_)*sqrt(t)) -
                    sqrt(b_)*exp(b_*t))
                _debug('     rule 5.3.9')

    if result is None:
        return None
    else:
        return Heaviside(t)*result, condition


@DEBUG_WRAP
def _inverse_laplace_early_prog_rules(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    prog_rules = [_inverse_laplace_irrational]

    for p_rule in prog_rules:
        if (r := p_rule(F, s, t, plane)) is not None:
            return r
    return None


@DEBUG_WRAP
def _inverse_laplace_apply_prog_rules(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    prog_rules = [_inverse_laplace_time_shift, _inverse_laplace_freq_shift,
                  _inverse_laplace_time_diff, _inverse_laplace_diff,
                  _inverse_laplace_irrational]

    for p_rule in prog_rules:
        if (r := p_rule(F, s, t, plane)) is not None:
            return r
    return None


@DEBUG_WRAP
def _inverse_laplace_expand(fn, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    if fn.is_Add:
        return None
    r = expand(fn, deep=False)
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    r = expand_mul(fn)
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    r = expand(fn)
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    if fn.is_rational_function(s):
        r = fn.apart(s).doit()
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    return None


@DEBUG_WRAP
def _inverse_laplace_rational(fn, s, t, plane, *, simplify):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    x_ = symbols('x_')
    f = fn.apart(s)
    terms = Add.make_args(f)
    terms_t = []
    conditions = [S.true]
    for term in terms:
        [n, d] = term.as_numer_denom()
        dc = d.as_poly(s).all_coeffs()
        dc_lead = dc[0]
        dc = [x/dc_lead for x in dc]
        nc = [x/dc_lead for x in n.as_poly(s).all_coeffs()]
        if len(dc) == 1:
            N = len(nc)-1
            for c in enumerate(nc):
                r = c[1]*DiracDelta(t, N-c[0])
                terms_t.append(r)
        elif len(dc) == 2:
            r = nc[0]*exp(-dc[1]*t)
            terms_t.append(Heaviside(t)*r)
        elif len(dc) == 3:
            a = dc[1]/2
            b = (dc[2]-a**2).factor()
            if len(nc) == 1:
                nc = [S.Zero] + nc
            l, m = tuple(nc)
            if b == 0:
                r = (m*t+l*(1-a*t))*exp(-a*t)
            else:
                hyp = False
                if b.is_negative:
                    b = -b
                    hyp = True
                b2 = list(roots(x_**2-b, x_).keys())[0]
                bs = sqrt(b).simplify()
                if hyp:
                    r = (
                        l*exp(-a*t)*cosh(b2*t) + (m-a*l) /
                        bs*exp(-a*t)*sinh(bs*t))
                else:
                    r = l*exp(-a*t)*cos(b2*t) + (m-a*l)/bs*exp(-a*t)*sin(bs*t)
            terms_t.append(Heaviside(t)*r)
        else:
            ft, cond = _inverse_laplace_transform(
                term, s, t, plane, simplify=simplify, dorational=False)
            terms_t.append(ft)
            conditions.append(cond)

    result = Add(*terms_t)
    if simplify:
        result = result.simplify(doit=False)
    return result, And(*conditions)


@DEBUG_WRAP
def _inverse_laplace_transform(fn, s_, t_, plane, *, simplify, dorational):
    """
    Front-end function of the inverse Laplace transform. It tries to apply all
    known rules recursively.  If everything else fails, it tries to integrate.
    """
    terms = Add.make_args(fn)
    terms_t = []
    conditions = []

    for term in terms:
        if term.has(exp):
            # Simplify expressions with exp() such that time-shifted
            # expressions have negative exponents in the numerator instead of
            # positive exponents in the numerator and denominator; this is a
            # (necessary) trick. It will, for example, convert
            # (s**2*exp(2*s) + 4*exp(s) - 4)*exp(-2*s)/(s*(s**2 + 1)) into
            # (s**2 + 4*exp(-s) - 4*exp(-2*s))/(s*(s**2 + 1))
            term = term.subs(s_, -s_).together().subs(s_, -s_)
        k, f = term.as_independent(s_, as_Add=False)
        if (
                dorational and term.is_rational_function(s_) and
                (r := _inverse_laplace_rational(
                    f, s_, t_, plane, simplify=simplify))
                is not None or
                (r := _inverse_laplace_apply_simple_rules(f, s_, t_))
                is not None or
                (r := _inverse_laplace_early_prog_rules(f, s_, t_, plane))
                is not None or
                (r := _inverse_laplace_expand(f, s_, t_, plane))
                is not None or
                (r := _inverse_laplace_apply_prog_rules(f, s_, t_, plane))
                is not None):
            pass
        elif any(undef.has(s_) for undef in f.atoms(AppliedUndef)):
            # If there are undefined functions f(t) then integration is
            # unlikely to do anything useful so we skip it and given an
            # unevaluated LaplaceTransform.
            r = (InverseLaplaceTransform(f, s_, t_, plane), S.true)
        elif (
                r := _inverse_laplace_transform_integration(
                    f, s_, t_, plane, simplify=simplify)) is not None:
            pass
        else:
            r = (InverseLaplaceTransform(f, s_, t_, plane), S.true)
        (ri_, ci_) = r
        terms_t.append(k*ri_)
        conditions.append(ci_)

    result = Add(*terms_t)
    if simplify:
        result = result.simplify(doit=False)
    condition = And(*conditions)

    return result, condition


class InverseLaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated inverse Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Laplace transforms, see the
    :func:`inverse_laplace_transform` docstring.
    """

    _name = 'Inverse Laplace'
    _none_sentinel = Dummy('None')
    _c = Dummy('c')

    def __new__(cls, F, s, x, plane, **opts):
        if plane is None:
            plane = InverseLaplaceTransform._none_sentinel
        return IntegralTransform.__new__(cls, F, s, x, plane, **opts)

    @property
    def fundamental_plane(self):
        plane = self.args[3]
        if plane is InverseLaplaceTransform._none_sentinel:
            plane = None
        return plane

    def _compute_transform(self, F, s, t, **hints):
        return _inverse_laplace_transform_integration(
            F, s, t, self.fundamental_plane, **hints)

    def _as_integral(self, F, s, t):
        c = self.__class__._c
        return (
            Integral(exp(s*t)*F, (s, c - S.ImaginaryUnit*S.Infinity,
                                  c + S.ImaginaryUnit*S.Infinity)) /
            (2*S.Pi*S.ImaginaryUnit))

    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        Standard hints are the following:
        - ``noconds``:  if True, do not return convergence conditions. The
        default setting is `True`.
        - ``simplify``: if True, it simplifies the final result. The
        default setting is `False`.
        """
        _noconds = hints.get('noconds', True)
        _simplify = hints.get('simplify', False)

        debugf('[ILT doit] (%s, %s, %s)', (self.function,
                                           self.function_variable,
                                           self.transform_variable))

        s_ = self.function_variable
        t_ = self.transform_variable
        fn = self.function
        plane = self.fundamental_plane

        r = _inverse_laplace_transform(
            fn, s_, t_, plane, simplify=_simplify, dorational=True)

        if _noconds:
            return r[0]
        else:
            return r


def inverse_laplace_transform(F, s, t, plane=None, **hints):
    r"""
    Compute the inverse Laplace transform of `F(s)`, defined as

    .. math ::
        f(t) = \frac{1}{2\pi i} \int_{c-i\infty}^{c+i\infty} e^{st}
        F(s) \mathrm{d}s,

    for `c` so large that `F(s)` has no singularites in the
    half-plane `\operatorname{Re}(s) > c-\epsilon`.

    Explanation
    ===========

    The plane can be specified by
    argument ``plane``, but will be inferred if passed as None.

    Under certain regularity conditions, this recovers `f(t)` from its
    Laplace Transform `F(s)`, for non-negative `t`, and vice
    versa.

    If the integral cannot be computed in closed form, this function returns
    an unevaluated :class:`InverseLaplaceTransform` object.

    Note that this function will always assume `t` to be real,
    regardless of the SymPy assumption on `t`.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.

    Examples
    ========

    >>> from sympy import inverse_laplace_transform, exp, Symbol
    >>> from sympy.abc import s, t
    >>> a = Symbol('a', positive=True)
    >>> inverse_laplace_transform(exp(-a*s)/s, s, t)
    Heaviside(-a + t)

    See Also
    ========

    laplace_transform
    hankel_transform, inverse_hankel_transform
    """
    _noconds = hints.get('noconds', True)
    _simplify = hints.get('simplify', False)

    if isinstance(F, MatrixBase) and hasattr(F, 'applyfunc'):
        return F.applyfunc(
            lambda Fij: inverse_laplace_transform(Fij, s, t, plane, **hints))

    r, c = InverseLaplaceTransform(F, s, t, plane).doit(
        noconds=False, simplify=_simplify)

    if _noconds:
        return r
    else:
        return r, c


def _fast_inverse_laplace(e, s, t):
    """Fast inverse Laplace transform of rational function including RootSum"""
    a, b, n = symbols('a, b, n', cls=Wild, exclude=[s])

    def _ilt(e):
        if not e.has(s):
            return e
        elif e.is_Add:
            return _ilt_add(e)
        elif e.is_Mul:
            return _ilt_mul(e)
        elif e.is_Pow:
            return _ilt_pow(e)
        elif isinstance(e, RootSum):
            return _ilt_rootsum(e)
        else:
            raise NotImplementedError

    def _ilt_add(e):
        return e.func(*map(_ilt, e.args))

    def _ilt_mul(e):
        coeff, expr = e.as_independent(s)
        if expr.is_Mul:
            raise NotImplementedError
        return coeff * _ilt(expr)

    def _ilt_pow(e):
        match = e.match((a*s + b)**n)
        if match is not None:
            nm, am, bm = match[n], match[a], match[b]
            if nm.is_Integer and nm < 0:
                return t**(-nm-1)*exp(-(bm/am)*t)/(am**-nm*gamma(-nm))
            if nm == 1:
                return exp(-(bm/am)*t) / am
        raise NotImplementedError

    def _ilt_rootsum(e):
        expr = e.fun.expr
        [variable] = e.fun.variables
        return RootSum(e.poly, Lambda(variable, together(_ilt(expr))))

    return _ilt(e)
<|endoftext|>""" This module cooks up a docstring when imported. Its only purpose is to
    be displayed in the sphinx documentation. """

from __future__ import annotations
from typing import Any

from sympy.integrals.meijerint import _create_lookup_table
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.printing.latex import latex

t: dict[tuple[type[Basic], ...], list[Any]] = {}
_create_lookup_table(t)


doc = ""
for about, category in t.items():
    if about == ():
        doc += 'Elementary functions:\n\n'
    else:
        doc += 'Functions involving ' + ', '.join('`%s`' % latex(
            list(category[0][0].atoms(func))[0]) for func in about) + ':\n\n'
    for formula, gs, cond, hint in category:
        if not isinstance(gs, list):
            g: Expr = Symbol('\\text{generated}')
        else:
            g = Add(*[fac*f for (fac, f) in gs])
        obj = Eq(formula, g)
        if cond is True:
            cond = ""
        else:
            cond = ',\\text{ if } %s' % latex(cond)
        doc += ".. math::\n  %s%s\n\n" % (latex(obj), cond)

__doc__ = doc
<|endoftext|>from sympy.functions import SingularityFunction, DiracDelta
from sympy.integrals import integrate


def singularityintegrate(f, x):
    """
    This function handles the indefinite integrations of Singularity functions.
    The ``integrate`` function calls this function internally whenever an
    instance of SingularityFunction is passed as argument.

    Explanation
    ===========

    The idea for integration is the following:

    - If we are dealing with a SingularityFunction expression,
      i.e. ``SingularityFunction(x, a, n)``, we just return
      ``SingularityFunction(x, a, n + 1)/(n + 1)`` if ``n >= 0`` and
      ``SingularityFunction(x, a, n + 1)`` if ``n < 0``.

    - If the node is a multiplication or power node having a
      SingularityFunction term we rewrite the whole expression in terms of
      Heaviside and DiracDelta and then integrate the output. Lastly, we
      rewrite the output of integration back in terms of SingularityFunction.

    - If none of the above case arises, we return None.

    Examples
    ========

    >>> from sympy.integrals.singularityfunctions import singularityintegrate
    >>> from sympy import SingularityFunction, symbols, Function
    >>> x, a, n, y = symbols('x a n y')
    >>> f = Function('f')
    >>> singularityintegrate(SingularityFunction(x, a, 3), x)
    SingularityFunction(x, a, 4)/4
    >>> singularityintegrate(5*SingularityFunction(x, 5, -2), x)
    5*SingularityFunction(x, 5, -1)
    >>> singularityintegrate(6*SingularityFunction(x, 5, -1), x)
    6*SingularityFunction(x, 5, 0)
    >>> singularityintegrate(x*SingularityFunction(x, 0, -1), x)
    0
    >>> singularityintegrate(SingularityFunction(x, 1, -1) * f(x), x)
    f(1)*SingularityFunction(x, 1, 0)

    """

    if not f.has(SingularityFunction):
        return None

    if isinstance(f, SingularityFunction):
        x, a, n = f.args
        if n.is_positive or n.is_zero:
            return SingularityFunction(x, a, n + 1)/(n + 1)
        elif n in (-1, -2, -3, -4):
            return SingularityFunction(x, a, n + 1)

    if f.is_Mul or f.is_Pow:

        expr = f.rewrite(DiracDelta)
        expr = integrate(expr, x)
        return expr.rewrite(SingularityFunction)
    return None
<|endoftext|>"""
Module to implement integration of uni/bivariate polynomials over
2D Polytopes and uni/bi/trivariate polynomials over 3D Polytopes.

Uses evaluation techniques as described in Chin et al. (2015) [1].


References
===========

.. [1] Chin, Eric B., Jean B. Lasserre, and N. Sukumar. "Numerical integration
of homogeneous functions on convex and nonconvex polygons and polyhedra."
Computational Mechanics 56.6 (2015): 967-981

PDF link : http://dilbert.engr.ucdavis.edu/~suku/quadrature/cls-integration.pdf
"""

from functools import cmp_to_key

from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify


def polytope_integrate(poly, expr=None, *, clockwise=False, max_degree=None):
    """Integrates polynomials over 2/3-Polytopes.

    Explanation
    ===========

    This function accepts the polytope in ``poly`` and the function in ``expr``
    (uni/bi/trivariate polynomials are implemented) and returns
    the exact integral of ``expr`` over ``poly``.

    Parameters
    ==========

    poly : The input Polygon.

    expr : The input polynomial.

    clockwise : Binary value to sort input points of 2-Polytope clockwise.(Optional)

    max_degree : The maximum degree of any monomial of the input polynomial.(Optional)

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Point, Polygon
    >>> from sympy.integrals.intpoly import polytope_integrate
    >>> polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    >>> polys = [1, x, y, x*y, x**2*y, x*y**2]
    >>> expr = x*y
    >>> polytope_integrate(polygon, expr)
    1/4
    >>> polytope_integrate(polygon, polys, max_degree=3)
    {1: 1, x: 1/2, y: 1/2, x*y: 1/4, x*y**2: 1/6, x**2*y: 1/6}
    """
    if clockwise:
        if isinstance(poly, Polygon):
            poly = Polygon(*point_sort(poly.vertices), evaluate=False)
        else:
            raise TypeError("clockwise=True works for only 2-Polytope"
                            "V-representation input")

    if isinstance(poly, Polygon):
        # For Vertex Representation(2D case)
        hp_params = hyperplane_parameters(poly)
        facets = poly.sides
    elif len(poly[0]) == 2:
        # For Hyperplane Representation(2D case)
        plen = len(poly)
        if len(poly[0][0]) == 2:
            intersections = [intersection(poly[(i - 1) % plen], poly[i],
                                          "plane2D")
                             for i in range(0, plen)]
            hp_params = poly
            lints = len(intersections)
            facets = [Segment2D(intersections[i],
                                intersections[(i + 1) % lints])
                      for i in range(lints)]
        else:
            raise NotImplementedError("Integration for H-representation 3D"
                                      "case not implemented yet.")
    else:
        # For Vertex Representation(3D case)
        vertices = poly[0]
        facets = poly[1:]
        hp_params = hyperplane_parameters(facets, vertices)

        if max_degree is None:
            if expr is None:
                raise TypeError('Input expression must be a valid SymPy expression')
            return main_integrate3d(expr, facets, vertices, hp_params)

    if max_degree is not None:
        result = {}
        if expr is not None:
            f_expr = []
            for e in expr:
                _ = decompose(e)
                if len(_) == 1 and not _.popitem()[0]:
                    f_expr.append(e)
                elif Poly(e).total_degree() <= max_degree:
                    f_expr.append(e)
            expr = f_expr

        if not isinstance(expr, list) and expr is not None:
            raise TypeError('Input polynomials must be list of expressions')

        if len(hp_params[0][0]) == 3:
            result_dict = main_integrate3d(0, facets, vertices, hp_params,
                                           max_degree)
        else:
            result_dict = main_integrate(0, facets, hp_params, max_degree)

        if expr is None:
            return result_dict

        for poly in expr:
            poly = _sympify(poly)
            if poly not in result:
                if poly.is_zero:
                    result[S.Zero] = S.Zero
                    continue
                integral_value = S.Zero
                monoms = decompose(poly, separate=True)
                for monom in monoms:
                    monom = nsimplify(monom)
                    coeff, m = strip(monom)
                    integral_value += result_dict[m] * coeff
                result[poly] = integral_value
        return result

    if expr is None:
        raise TypeError('Input expression must be a valid SymPy expression')

    return main_integrate(expr, facets, hp_params)


def strip(monom):
    if monom.is_zero:
        return S.Zero, S.Zero
    elif monom.is_number:
        return monom, S.One
    else:
        coeff = LC(monom)
        return coeff, monom / coeff

def _polynomial_integrate(polynomials, facets, hp_params):
    dims = (x, y)
    dim_length = len(dims)
    integral_value = S.Zero
    for deg in polynomials:
        poly_contribute = S.Zero
        facet_count = 0
        for hp in hp_params:
            value_over_boundary = integration_reduction(facets,
                                                        facet_count,
                                                        hp[0], hp[1],
                                                        polynomials[deg],
                                                        dims, deg)
            poly_contribute += value_over_boundary * (hp[1] / norm(hp[0]))
            facet_count += 1
        poly_contribute /= (dim_length + deg)
        integral_value += poly_contribute

    return integral_value


def main_integrate3d(expr, facets, vertices, hp_params, max_degree=None):
    """Function to translate the problem of integrating uni/bi/tri-variate
    polynomials over a 3-Polytope to integrating over its faces.
    This is done using Generalized Stokes' Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Faces of the 3-Polytope(expressed as indices of `vertices`).
    vertices :
        Vertices that constitute the Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        Max degree of constituent monomial in given list of polynomial.

    Examples
    ========

    >>> from sympy.integrals.intpoly import main_integrate3d, \
    hyperplane_parameters
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> vertices = cube[0]
    >>> faces = cube[1:]
    >>> hp_params = hyperplane_parameters(faces, vertices)
    >>> main_integrate3d(1, faces, vertices, hp_params)
    -125
    """
    result = {}
    dims = (x, y, z)
    dim_length = len(dims)
    if max_degree:
        grad_terms = gradient_terms(max_degree, 3)
        flat_list = [term for z_terms in grad_terms
                     for x_term in z_terms
                     for term in x_term]

        for term in flat_list:
            result[term[0]] = 0

        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            x0 = vertices[facets[facet_count][0]]

            for i, monom in enumerate(flat_list):
                #  Every monomial is a tuple :
                #  (term, x_degree, y_degree, z_degree, value over boundary)
                expr, x_d, y_d, z_d, z_index, y_index, x_index, _ = monom
                degree = x_d + y_d + z_d
                if b.is_zero:
                    value_over_face = S.Zero
                else:
                    value_over_face = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, expr, degree, dims,
                                                      x_index, y_index,
                                                      z_index, x0, grad_terms,
                                                      i, vertices, hp)
                monom[7] = value_over_face
                result[expr] += value_over_face * \
                    (b / norm(a)) / (dim_length + x_d + y_d + z_d)
        return result
    else:
        integral_value = S.Zero
        polynomials = decompose(expr)
        for deg in polynomials:
            poly_contribute = S.Zero
            facet_count = 0
            for i, facet in enumerate(facets):
                hp = hp_params[i]
                if hp[1].is_zero:
                    continue
                pi = polygon_integrate(facet, hp, i, facets, vertices, expr, deg)
                poly_contribute += pi *\
                    (hp[1] / norm(tuple(hp[0])))
                facet_count += 1
            poly_contribute /= (dim_length + deg)
            integral_value += poly_contribute
    return integral_value


def main_integrate(expr, facets, hp_params, max_degree=None):
    """Function to translate the problem of integrating univariate/bivariate
    polynomials over a 2-Polytope to integrating over its boundary facets.
    This is done using Generalized Stokes's Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Facets(Line Segments) of the 2-Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        The maximum degree of any monomial of the input polynomial.

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import main_integrate,\
    hyperplane_parameters
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> hp_params = hyperplane_parameters(triangle)
    >>> main_integrate(x**2 + y**2, facets, hp_params)
    325/6
    """
    dims = (x, y)
    dim_length = len(dims)
    result = {}

    if max_degree:
        grad_terms = [[0, 0, 0, 0]] + gradient_terms(max_degree)

        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            x0 = facets[facet_count].points[0]

            for i, monom in enumerate(grad_terms):
                #  Every monomial is a tuple :
                #  (term, x_degree, y_degree, value over boundary)
                m, x_d, y_d, _ = monom
                value = result.get(m, None)
                degree = S.Zero
                if b.is_zero:
                    value_over_boundary = S.Zero
                else:
                    degree = x_d + y_d
                    value_over_boundary = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, m, degree, dims, x_d,
                                                      y_d, max_degree, x0,
                                                      grad_terms, i)
                monom[3] = value_over_boundary
                if value is not None:
                    result[m] += value_over_boundary * \
                                        (b / norm(a)) / (dim_length + degree)
                else:
                    result[m] = value_over_boundary * \
                                (b / norm(a)) / (dim_length + degree)
        return result
    else:
        if not isinstance(expr, list):
            polynomials = decompose(expr)
            return _polynomial_integrate(polynomials, facets, hp_params)
        else:
            return {e: _polynomial_integrate(decompose(e), facets, hp_params) for e in expr}


def polygon_integrate(facet, hp_param, index, facets, vertices, expr, degree):
    """Helper function to integrate the input uni/bi/trivariate polynomial
    over a certain face of the 3-Polytope.

    Parameters
    ==========

    facet :
        Particular face of the 3-Polytope over which ``expr`` is integrated.
    index :
        The index of ``facet`` in ``facets``.
    facets :
        Faces of the 3-Polytope(expressed as indices of `vertices`).
    vertices :
        Vertices that constitute the facet.
    expr :
        The input polynomial.
    degree :
        Degree of ``expr``.

    Examples
    ========

    >>> from sympy.integrals.intpoly import polygon_integrate
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                 [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> facet = cube[1]
    >>> facets = cube[1:]
    >>> vertices = cube[0]
    >>> polygon_integrate(facet, [(0, 1, 0), 5], 0, facets, vertices, 1, 0)
    -25
    """
    expr = S(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = vertices[facet[0]]
    facet_len = len(facet)
    for i, fac in enumerate(facet):
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        result += distance_to_side(x0, side, hp_param[0]) *\
            lineseg_integrate(facet, i, side, expr, degree)
    if not expr.is_number:
        expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] +\
            diff(expr, z) * x0[2]
        result += polygon_integrate(facet, hp_param, index, facets, vertices,
                                    expr, degree - 1)
    result /= (degree + 2)
    return result


def distance_to_side(point, line_seg, A):
    """Helper function to compute the signed distance between given 3D point
    and a line segment.

    Parameters
    ==========

    point : 3D Point
    line_seg : Line Segment

    Examples
    ========

    >>> from sympy.integrals.intpoly import distance_to_side
    >>> point = (0, 0, 0)
    >>> distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0))
    -sqrt(2)/2
    """
    x1, x2 = line_seg
    rev_normal = [-1 * S(i)/norm(A) for i in A]
    vector = [x2[i] - x1[i] for i in range(0, 3)]
    vector = [vector[i]/norm(vector) for i in range(0, 3)]

    n_side = cross_product((0, 0, 0), rev_normal, vector)
    vectorx0 = [line_seg[0][i] - point[i] for i in range(0, 3)]
    dot_product = sum(vectorx0[i] * n_side[i] for i in range(0, 3))

    return dot_product


def lineseg_integrate(polygon, index, line_seg, expr, degree):
    """Helper function to compute the line integral of ``expr`` over ``line_seg``.

    Parameters
    ===========

    polygon :
        Face of a 3-Polytope.
    index :
        Index of line_seg in polygon.
    line_seg :
        Line Segment.

    Examples
    ========

    >>> from sympy.integrals.intpoly import lineseg_integrate
    >>> polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    >>> line_seg = [(0, 5, 0), (5, 5, 0)]
    >>> lineseg_integrate(polygon, 0, line_seg, 1, 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = line_seg[0]
    distance = norm(tuple([line_seg[1][i] - line_seg[0][i] for i in
                           range(3)]))
    if isinstance(expr, Expr):
        expr_dict = {x: line_seg[1][0],
                     y: line_seg[1][1],
                     z: line_seg[1][2]}
        result += distance * expr.subs(expr_dict)
    else:
        result += distance * expr

    expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] +\
        diff(expr, z) * x0[2]

    result += lineseg_integrate(polygon, index, line_seg, expr, degree - 1)
    result /= (degree + 1)
    return result


def integration_reduction(facets, index, a, b, expr, dims, degree):
    """Helper method for main_integrate. Returns the value of the input
    expression evaluated over the polytope facet referenced by a given index.

    Parameters
    ===========

    facets :
        List of facets of the polytope.
    index :
        Index referencing the facet to integrate the expression over.
    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    expr :
        The expression to integrate over the facet.
    dims :
        List of symbols denoting axes.
    degree :
        Degree of the homogeneous polynomial.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import integration_reduction,\
    hyperplane_parameters
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> integration_reduction(facets, 0, a, b, 1, (x, y), 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return expr

    value = S.Zero
    x0 = facets[index].points[0]
    m = len(facets)
    gens = (x, y)

    inner_product = diff(expr, gens[0]) * x0[0] + diff(expr, gens[1]) * x0[1]

    if inner_product != 0:
        value += integration_reduction(facets, index, a, b,
                                       inner_product, dims, degree - 1)

    value += left_integral2D(m, index, facets, x0, expr, gens)

    return value/(len(dims) + degree - 1)


def left_integral2D(m, index, facets, x0, expr, gens):
    """Computes the left integral of Eq 10 in Chin et al.
    For the 2D case, the integral is just an evaluation of the polynomial
    at the intersection of two facets which is multiplied by the distance
    between the first point of facet and that intersection.

    Parameters
    ==========

    m :
        No. of hyperplanes.
    index :
        Index of facet to find intersections with.
    facets :
        List of facets(Line Segments in 2D case).
    x0 :
        First point on facet referenced by index.
    expr :
        Input polynomial
    gens :
        Generators which generate the polynomial

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import left_integral2D
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> left_integral2D(3, 0, facets, facets[0].points[0], 1, (x, y))
    5
    """
    value = S.Zero
    for j in range(m):
        intersect = ()
        if j in ((index - 1) % m, (index + 1) % m):
