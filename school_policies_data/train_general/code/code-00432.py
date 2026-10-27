        Q.extended_nonpositive, Q.extended_nonzero, Q.extended_real,
        Q.infinite, Q.negative_infinite]), set([Q.algebraic,
        Q.complex, Q.composite, Q.even, Q.extended_nonnegative,
        Q.extended_positive, Q.finite, Q.imaginary, Q.integer,
        Q.irrational, Q.negative, Q.nonnegative, Q.nonpositive,
        Q.nonzero, Q.odd, Q.positive, Q.positive_infinite, Q.prime,
        Q.rational, Q.real, Q.transcendental, Q.zero])),
        Q.noninteger: (set([Q.noninteger]), set([])),
        Q.nonnegative: (set([Q.commutative, Q.complex, Q.extended_nonnegative,
        Q.extended_real, Q.finite, Q.hermitian, Q.nonnegative,
        Q.real]), set([Q.extended_negative, Q.imaginary, Q.infinite,
        Q.negative, Q.negative_infinite, Q.positive_infinite])),
        Q.nonpositive: (set([Q.commutative, Q.complex, Q.extended_nonpositive,
        Q.extended_real, Q.finite, Q.hermitian, Q.nonpositive,
        Q.real]), set([Q.composite, Q.extended_positive, Q.imaginary,
        Q.infinite, Q.negative_infinite, Q.positive,
        Q.positive_infinite, Q.prime])),
        Q.nonzero: (set([Q.commutative, Q.complex, Q.extended_nonzero,
        Q.extended_real, Q.finite, Q.hermitian, Q.nonzero, Q.real]),
        set([Q.imaginary, Q.infinite, Q.negative_infinite,
        Q.positive_infinite, Q.zero])),
        Q.normal: (set([Q.normal, Q.square]), set([])),
        Q.odd: (set([Q.algebraic, Q.commutative, Q.complex,
        Q.extended_nonzero, Q.extended_real, Q.finite, Q.hermitian,
        Q.integer, Q.nonzero, Q.odd, Q.rational, Q.real]),
        set([Q.even, Q.imaginary, Q.infinite, Q.irrational,
        Q.negative_infinite, Q.positive_infinite, Q.transcendental,
        Q.zero])),
        Q.orthogonal: (set([Q.fullrank, Q.invertible, Q.normal, Q.orthogonal,
        Q.positive_definite, Q.square, Q.unitary]), set([Q.singular])),
        Q.positive: (set([Q.commutative, Q.complex, Q.extended_nonnegative,
        Q.extended_nonzero, Q.extended_positive, Q.extended_real,
        Q.finite, Q.hermitian, Q.nonnegative, Q.nonzero, Q.positive,
        Q.real]), set([Q.extended_negative, Q.extended_nonpositive,
        Q.imaginary, Q.infinite, Q.negative, Q.negative_infinite,
        Q.nonpositive, Q.positive_infinite, Q.zero])),
        Q.positive_definite: (set([Q.fullrank, Q.invertible,
        Q.positive_definite, Q.square]), set([Q.singular])),
        Q.positive_infinite: (set([Q.commutative, Q.extended_nonnegative,
        Q.extended_nonzero, Q.extended_positive, Q.extended_real,
        Q.infinite, Q.positive_infinite]), set([Q.algebraic,
        Q.complex, Q.composite, Q.even, Q.extended_negative,
        Q.extended_nonpositive, Q.finite, Q.imaginary, Q.integer,
        Q.irrational, Q.negative, Q.negative_infinite, Q.nonnegative,
        Q.nonpositive, Q.nonzero, Q.odd, Q.positive, Q.prime,
        Q.rational, Q.real, Q.transcendental, Q.zero])),
        Q.prime: (set([Q.algebraic, Q.commutative, Q.complex,
        Q.extended_nonnegative, Q.extended_nonzero,
        Q.extended_positive, Q.extended_real, Q.finite, Q.hermitian,
        Q.integer, Q.nonnegative, Q.nonzero, Q.positive, Q.prime,
        Q.rational, Q.real]), set([Q.composite, Q.extended_negative,
        Q.extended_nonpositive, Q.imaginary, Q.infinite, Q.irrational,
        Q.negative, Q.negative_infinite, Q.nonpositive,
        Q.positive_infinite, Q.transcendental, Q.zero])),
        Q.rational: (set([Q.algebraic, Q.commutative, Q.complex,
        Q.extended_real, Q.finite, Q.hermitian, Q.rational, Q.real]),
        set([Q.imaginary, Q.infinite, Q.irrational,
        Q.negative_infinite, Q.positive_infinite, Q.transcendental])),
        Q.real: (set([Q.commutative, Q.complex, Q.extended_real, Q.finite,
        Q.hermitian, Q.real]), set([Q.imaginary, Q.infinite,
        Q.negative_infinite, Q.positive_infinite])),
        Q.real_elements: (set([Q.complex_elements, Q.real_elements]), set([])),
        Q.singular: (set([Q.singular]), set([Q.invertible, Q.orthogonal,
        Q.positive_definite, Q.unitary])),
        Q.square: (set([Q.square]), set([])),
        Q.symmetric: (set([Q.square, Q.symmetric]), set([])),
        Q.transcendental: (set([Q.commutative, Q.complex, Q.finite,
        Q.transcendental]), set([Q.algebraic, Q.composite, Q.even,
        Q.infinite, Q.integer, Q.negative_infinite, Q.odd,
        Q.positive_infinite, Q.prime, Q.rational, Q.zero])),
        Q.triangular: (set([Q.triangular]), set([])),
        Q.unit_triangular: (set([Q.triangular, Q.unit_triangular]), set([])),
        Q.unitary: (set([Q.fullrank, Q.invertible, Q.normal, Q.square,
        Q.unitary]), set([Q.singular])),
        Q.upper_triangular: (set([Q.triangular, Q.upper_triangular]), set([])),
        Q.zero: (set([Q.algebraic, Q.commutative, Q.complex, Q.even,
        Q.extended_nonnegative, Q.extended_nonpositive,
        Q.extended_real, Q.finite, Q.hermitian, Q.integer,
        Q.nonnegative, Q.nonpositive, Q.rational, Q.real, Q.zero]),
        set([Q.composite, Q.extended_negative, Q.extended_nonzero,
        Q.extended_positive, Q.imaginary, Q.infinite, Q.irrational,
        Q.negative, Q.negative_infinite, Q.nonzero, Q.odd, Q.positive,
        Q.positive_infinite, Q.prime, Q.transcendental])),
    }
<|endoftext|>"""
Functions and wrapper object to call assumption property and predicate
query with same syntax.

In SymPy, there are two assumption systems. Old assumption system is
defined in sympy/core/assumptions, and it can be accessed by attribute
such as ``x.is_even``. New assumption system is defined in
sympy/assumptions, and it can be accessed by predicates such as
``Q.even(x)``.

Old assumption is fast, while new assumptions can freely take local facts.
In general, old assumption is used in evaluation method and new assumption
is used in refinement method.

In most cases, both evaluation and refinement follow the same process, and
the only difference is which assumption system is used. This module provides
``is_[...]()`` functions and ``AssumptionsWrapper()`` class which allows
using two systems with same syntax so that parallel code implementation can be
avoided.

Examples
========

For multiple use, use ``AssumptionsWrapper()``.

>>> from sympy import Q, Symbol
>>> from sympy.assumptions.wrapper import AssumptionsWrapper
>>> x = Symbol('x')
>>> _x = AssumptionsWrapper(x, Q.even(x))
>>> _x.is_integer
True
>>> _x.is_odd
False

For single use, use ``is_[...]()`` functions.

>>> from sympy.assumptions.wrapper import is_infinite
>>> a = Symbol('a')
>>> print(is_infinite(a))
None
>>> is_infinite(a, Q.finite(a))
False

"""

from sympy.assumptions import ask, Q
from sympy.core.basic import Basic
from sympy.core.sympify import _sympify


def make_eval_method(fact):
    def getit(self):
        pred = getattr(Q, fact)
        ret = ask(pred(self.expr), self.assumptions)
        return ret
    return getit


# we subclass Basic to use the fact deduction and caching
class AssumptionsWrapper(Basic):
    """
    Wrapper over ``Basic`` instances to call predicate query by
    ``.is_[...]`` property

    Parameters
    ==========

    expr : Basic

    assumptions : Boolean, optional

    Examples
    ========

    >>> from sympy import Q, Symbol
    >>> from sympy.assumptions.wrapper import AssumptionsWrapper
    >>> x = Symbol('x', even=True)
    >>> AssumptionsWrapper(x).is_integer
    True
    >>> y = Symbol('y')
    >>> AssumptionsWrapper(y, Q.even(y)).is_integer
    True

    With ``AssumptionsWrapper``, both evaluation and refinement can be supported
    by single implementation.

    >>> from sympy import Function
    >>> class MyAbs(Function):
    ...     @classmethod
    ...     def eval(cls, x, assumptions=True):
    ...         _x = AssumptionsWrapper(x, assumptions)
    ...         if _x.is_nonnegative:
    ...             return x
    ...         if _x.is_negative:
    ...             return -x
    ...     def _eval_refine(self, assumptions):
    ...         return MyAbs.eval(self.args[0], assumptions)
    >>> MyAbs(x)
    MyAbs(x)
    >>> MyAbs(x).refine(Q.positive(x))
    x
    >>> MyAbs(Symbol('y', negative=True))
    -y

    """
    def __new__(cls, expr, assumptions=None):
        if assumptions is None:
            return expr
        obj = super().__new__(cls, expr, _sympify(assumptions))
        obj.expr = expr
        obj.assumptions = assumptions
        return obj

    _eval_is_algebraic = make_eval_method("algebraic")
    _eval_is_antihermitian = make_eval_method("antihermitian")
    _eval_is_commutative = make_eval_method("commutative")
    _eval_is_complex = make_eval_method("complex")
    _eval_is_composite = make_eval_method("composite")
    _eval_is_even = make_eval_method("even")
    _eval_is_extended_negative = make_eval_method("extended_negative")
    _eval_is_extended_nonnegative = make_eval_method("extended_nonnegative")
    _eval_is_extended_nonpositive = make_eval_method("extended_nonpositive")
    _eval_is_extended_nonzero = make_eval_method("extended_nonzero")
    _eval_is_extended_positive = make_eval_method("extended_positive")
    _eval_is_extended_real = make_eval_method("extended_real")
    _eval_is_finite = make_eval_method("finite")
    _eval_is_hermitian = make_eval_method("hermitian")
    _eval_is_imaginary = make_eval_method("imaginary")
    _eval_is_infinite = make_eval_method("infinite")
    _eval_is_integer = make_eval_method("integer")
    _eval_is_irrational = make_eval_method("irrational")
    _eval_is_negative = make_eval_method("negative")
    _eval_is_noninteger = make_eval_method("noninteger")
    _eval_is_nonnegative = make_eval_method("nonnegative")
    _eval_is_nonpositive = make_eval_method("nonpositive")
    _eval_is_nonzero = make_eval_method("nonzero")
    _eval_is_odd = make_eval_method("odd")
    _eval_is_polar = make_eval_method("polar")
    _eval_is_positive = make_eval_method("positive")
    _eval_is_prime = make_eval_method("prime")
    _eval_is_rational = make_eval_method("rational")
    _eval_is_real = make_eval_method("real")
    _eval_is_transcendental = make_eval_method("transcendental")
    _eval_is_zero = make_eval_method("zero")


# one shot functions which are faster than AssumptionsWrapper

def is_infinite(obj, assumptions=None):
    if assumptions is None:
        return obj.is_infinite
    return ask(Q.infinite(obj), assumptions)


def is_extended_real(obj, assumptions=None):
    if assumptions is None:
        return obj.is_extended_real
    return ask(Q.extended_real(obj), assumptions)


def is_extended_nonnegative(obj, assumptions=None):
    if assumptions is None:
        return obj.is_extended_nonnegative
    return ask(Q.extended_nonnegative(obj), assumptions)
<|endoftext|>from sympy.assumptions.assume import global_assumptions
from sympy.assumptions.cnf import CNF, EncodedCNF
from sympy.assumptions.ask import Q
from sympy.logic.inference import satisfiable
from sympy.logic.algorithms.lra_theory import UnhandledInput, ALLOWED_PRED
from sympy.matrices.kind import MatrixKind
from sympy.core.kind import NumberKind
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.mul import Mul
from sympy.core.singleton import S


def lra_satask(proposition, assumptions=True, context=global_assumptions):
    """
    Function to evaluate the proposition with assumptions using SAT algorithm
    in conjunction with an Linear Real Arithmetic theory solver.

    Used to handle inequalities. Should eventually be depreciated and combined
    into satask, but infinity handling and other things need to be implemented
    before that can happen.
    """
    props = CNF.from_prop(proposition)
    _props = CNF.from_prop(~proposition)

    cnf = CNF.from_prop(assumptions)
    assumptions = EncodedCNF()
    assumptions.from_cnf(cnf)

    context_cnf = CNF()
    if context:
        context_cnf = context_cnf.extend(context)

    assumptions.add_from_cnf(context_cnf)

    return check_satisfiability(props, _props, assumptions)

# Some predicates such as Q.prime can't be handled by lra_satask.
# For example, (x > 0) & (x < 1) & Q.prime(x) is unsat but lra_satask would think it was sat.
# WHITE_LIST is a list of predicates that can always be handled.
WHITE_LIST = ALLOWED_PRED | {Q.positive, Q.negative, Q.zero, Q.nonzero, Q.nonpositive, Q.nonnegative,
                                            Q.extended_positive, Q.extended_negative, Q.extended_nonpositive,
                                            Q.extended_negative, Q.extended_nonzero, Q.negative_infinite,
                                            Q.positive_infinite}


def check_satisfiability(prop, _prop, factbase):
    sat_true = factbase.copy()
    sat_false = factbase.copy()
    sat_true.add_from_cnf(prop)
    sat_false.add_from_cnf(_prop)

    all_pred, all_exprs = get_all_pred_and_expr_from_enc_cnf(sat_true)

    for pred in all_pred:
        if pred.function not in WHITE_LIST and pred.function != Q.ne:
            raise UnhandledInput(f"LRASolver: {pred} is an unhandled predicate")
    for expr in all_exprs:
        if expr.kind == MatrixKind(NumberKind):
            raise UnhandledInput(f"LRASolver: {expr} is of MatrixKind")
        if expr == S.NaN:
            raise UnhandledInput("LRASolver: nan")

    # convert old assumptions into predicates and add them to sat_true and sat_false
    # also check for unhandled predicates
    for assm in extract_pred_from_old_assum(all_exprs):
        n = len(sat_true.encoding)
        if assm not in sat_true.encoding:
            sat_true.encoding[assm] = n+1
        sat_true.data.append([sat_true.encoding[assm]])

        n = len(sat_false.encoding)
        if assm not in sat_false.encoding:
            sat_false.encoding[assm] = n+1
        sat_false.data.append([sat_false.encoding[assm]])


    sat_true = _preprocess(sat_true)
    sat_false = _preprocess(sat_false)

    can_be_true = satisfiable(sat_true, use_lra_theory=True) is not False
    can_be_false = satisfiable(sat_false, use_lra_theory=True) is not False

    if can_be_true and can_be_false:
        return None

    if can_be_true and not can_be_false:
        return True

    if not can_be_true and can_be_false:
        return False

    if not can_be_true and not can_be_false:
        raise ValueError("Inconsistent assumptions")


def _preprocess(enc_cnf):
    """
    Returns an encoded cnf with only Q.eq, Q.gt, Q.lt,
    Q.ge, and Q.le predicate.

    Converts every unequality into a disjunction of strict
    inequalities. For example, x != 3 would become
    x < 3 OR x > 3.

    Also converts all negated Q.ne predicates into
    equalities.
    """

    # loops through each literal in each clause
    # to construct a new, preprocessed encodedCNF

    enc_cnf = enc_cnf.copy()
    cur_enc = 1
    rev_encoding = {value: key for key, value in enc_cnf.encoding.items()}

    new_encoding = {}
    new_data = []
    for clause in enc_cnf.data:
        new_clause = []
        for lit in clause:
            if lit == 0:
                new_clause.append(lit)
                new_encoding[lit] = False
                continue
            prop = rev_encoding[abs(lit)]
            negated = lit < 0
            sign = (lit > 0) - (lit < 0)

            prop = _pred_to_binrel(prop)

            if not isinstance(prop, AppliedPredicate):
                if prop not in new_encoding:
                    new_encoding[prop] = cur_enc
                    cur_enc += 1
                lit = new_encoding[prop]
                new_clause.append(sign*lit)
                continue


            if negated and prop.function == Q.eq:
                negated = False
                prop = Q.ne(*prop.arguments)

            if prop.function == Q.ne:
                arg1, arg2 = prop.arguments
                if negated:
                    new_prop = Q.eq(arg1, arg2)
                    if new_prop not in new_encoding:
                        new_encoding[new_prop] = cur_enc
                        cur_enc += 1

                    new_enc = new_encoding[new_prop]
                    new_clause.append(new_enc)
                    continue
                else:
                    new_props = (Q.gt(arg1, arg2), Q.lt(arg1, arg2))
                    for new_prop in new_props:
                        if new_prop not in new_encoding:
                            new_encoding[new_prop] = cur_enc
                            cur_enc += 1

                        new_enc = new_encoding[new_prop]
                        new_clause.append(new_enc)
                    continue

            if prop.function == Q.eq and negated:
                assert False

            if prop not in new_encoding:
                new_encoding[prop] = cur_enc
                cur_enc += 1
            new_clause.append(new_encoding[prop]*sign)
        new_data.append(new_clause)

    assert len(new_encoding) >= cur_enc - 1

    enc_cnf = EncodedCNF(new_data, new_encoding)
    return enc_cnf


def _pred_to_binrel(pred):
    if not isinstance(pred, AppliedPredicate):
        return pred

    if pred.function in pred_to_pos_neg_zero:
        f = pred_to_pos_neg_zero[pred.function]
        if f is False:
            return False
        pred = f(pred.arguments[0])

    if pred.function == Q.positive:
        pred = Q.gt(pred.arguments[0], 0)
    elif pred.function == Q.negative:
        pred = Q.lt(pred.arguments[0], 0)
    elif pred.function == Q.zero:
        pred = Q.eq(pred.arguments[0], 0)
    elif pred.function == Q.nonpositive:
        pred = Q.le(pred.arguments[0], 0)
    elif pred.function == Q.nonnegative:
        pred = Q.ge(pred.arguments[0], 0)
    elif pred.function == Q.nonzero:
        pred = Q.ne(pred.arguments[0], 0)

    return pred

pred_to_pos_neg_zero = {
    Q.extended_positive: Q.positive,
    Q.extended_negative: Q.negative,
    Q.extended_nonpositive: Q.nonpositive,
    Q.extended_negative: Q.negative,
    Q.extended_nonzero: Q.nonzero,
    Q.negative_infinite: False,
    Q.positive_infinite: False
}

def get_all_pred_and_expr_from_enc_cnf(enc_cnf):
    all_exprs = set()
    all_pred = set()
    for pred in enc_cnf.encoding.keys():
        if isinstance(pred, AppliedPredicate):
            all_pred.add(pred)
            all_exprs.update(pred.arguments)

    return all_pred, all_exprs

def extract_pred_from_old_assum(all_exprs):
    """
    Returns a list of relevant new assumption predicate
    based on any old assumptions.

    Raises an UnhandledInput exception if any of the assumptions are
    unhandled.

    Ignored predicate:
    - commutative
    - complex
    - algebraic
    - transcendental
    - extended_real
    - real
    - all matrix predicate
    - rational
    - irrational

    Example
    =======
    >>> from sympy.assumptions.lra_satask import extract_pred_from_old_assum
    >>> from sympy import symbols
    >>> x, y = symbols("x y", positive=True)
    >>> extract_pred_from_old_assum([x, y, 2])
    [Q.positive(x), Q.positive(y)]
    """
    ret = []
    for expr in all_exprs:
        if not hasattr(expr, "free_symbols"):
            continue
        if len(expr.free_symbols) == 0:
            continue

        if expr.is_real is not True:
            raise UnhandledInput(f"LRASolver: {expr} must be real")
        # test for I times imaginary variable; such expressions are considered real
        if isinstance(expr, Mul) and any(arg.is_real is not True for arg in expr.args):
            raise UnhandledInput(f"LRASolver: {expr} must be real")

        if expr.is_integer == True and expr.is_zero != True:
            raise UnhandledInput(f"LRASolver: {expr} is an integer")
        if expr.is_integer == False:
            raise UnhandledInput(f"LRASolver: {expr} can't be an integer")
        if expr.is_rational == False:
            raise UnhandledInput(f"LRASolver: {expr} is irational")

        if expr.is_zero:
            ret.append(Q.zero(expr))
        elif expr.is_positive:
            ret.append(Q.positive(expr))
        elif expr.is_negative:
            ret.append(Q.negative(expr))
        elif expr.is_nonzero:
            ret.append(Q.nonzero(expr))
        elif expr.is_nonpositive:
            ret.append(Q.nonpositive(expr))
        elif expr.is_nonnegative:
            ret.append(Q.nonnegative(expr))

    return ret
<|endoftext|>"""A module which implements predicates and assumption context."""

from contextlib import contextmanager
import inspect
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import Boolean, false, true
from sympy.multipledispatch.dispatcher import Dispatcher, str_signature
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.source import get_class


class AssumptionsContext(set):
    """
    Set containing default assumptions which are applied to the ``ask()``
    function.

    Explanation
    ===========

    This is used to represent global assumptions, but you can also use this
    class to create your own local assumptions contexts. It is basically a thin
    wrapper to Python's set, so see its documentation for advanced usage.

    Examples
    ========

    The default assumption context is ``global_assumptions``, which is initially empty:

    >>> from sympy import ask, Q
    >>> from sympy.assumptions import global_assumptions
    >>> global_assumptions
    AssumptionsContext()

    You can add default assumptions:

    >>> from sympy.abc import x
    >>> global_assumptions.add(Q.real(x))
    >>> global_assumptions
    AssumptionsContext({Q.real(x)})
    >>> ask(Q.real(x))
    True

    And remove them:

    >>> global_assumptions.remove(Q.real(x))
    >>> print(ask(Q.real(x)))
    None

    The ``clear()`` method removes every assumption:

    >>> global_assumptions.add(Q.positive(x))
    >>> global_assumptions
    AssumptionsContext({Q.positive(x)})
    >>> global_assumptions.clear()
    >>> global_assumptions
    AssumptionsContext()

    See Also
    ========

    assuming

    """

    def add(self, *assumptions):
        """Add assumptions."""
        for a in assumptions:
            super().add(a)

    def _sympystr(self, printer):
        if not self:
            return "%s()" % self.__class__.__name__
        return "{}({})".format(self.__class__.__name__, printer._print_set(self))

global_assumptions = AssumptionsContext()


class AppliedPredicate(Boolean):
    """
    The class of expressions resulting from applying ``Predicate`` to
    the arguments. ``AppliedPredicate`` merely wraps its argument and
    remain unevaluated. To evaluate it, use the ``ask()`` function.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> Q.integer(1)
    Q.integer(1)

    The ``function`` attribute returns the predicate, and the ``arguments``
    attribute returns the tuple of arguments.

    >>> type(Q.integer(1))
    <class 'sympy.assumptions.assume.AppliedPredicate'>
    >>> Q.integer(1).function
    Q.integer
    >>> Q.integer(1).arguments
    (1,)

    Applied predicates can be evaluated to a boolean value with ``ask``:

    >>> ask(Q.integer(1))
    True

    """
    __slots__ = ()

    def __new__(cls, predicate, *args):
        if not isinstance(predicate, Predicate):
            raise TypeError("%s is not a Predicate." % predicate)
        args = map(_sympify, args)
        return super().__new__(cls, predicate, *args)

    @property
    def arg(self):
        """
        Return the expression used by this assumption.

        Examples
        ========

        >>> from sympy import Q, Symbol
        >>> x = Symbol('x')
        >>> a = Q.integer(x + 1)
        >>> a.arg
        x + 1

        """
        # Will be deprecated
        args = self._args
        if len(args) == 2:
            # backwards compatibility
            return args[1]
        raise TypeError("'arg' property is allowed only for unary predicates.")

    @property
    def function(self):
        """
        Return the predicate.
        """
        # Will be changed to self.args[0] after args overriding is removed
        return self._args[0]

    @property
    def arguments(self):
        """
        Return the arguments which are applied to the predicate.
        """
        # Will be changed to self.args[1:] after args overriding is removed
        return self._args[1:]

    def _eval_ask(self, assumptions):
        return self.function.eval(self.arguments, assumptions)

    @property
    def binary_symbols(self):
        from .ask import Q
        if self.function == Q.is_true:
            i = self.arguments[0]
            if i.is_Boolean or i.is_Symbol:
                return i.binary_symbols
        if self.function in (Q.eq, Q.ne):
            if true in self.arguments or false in self.arguments:
                if self.arguments[0].is_Symbol:
                    return {self.arguments[0]}
                elif self.arguments[1].is_Symbol:
                    return {self.arguments[1]}
        return set()


class PredicateMeta(type):
    def __new__(cls, clsname, bases, dct):
        # If handler is not defined, assign empty dispatcher.
        if "handler" not in dct:
            name = f"Ask{clsname.capitalize()}Handler"
            handler = Dispatcher(name, doc="Handler for key %s" % name)
            dct["handler"] = handler

        dct["_orig_doc"] = dct.get("__doc__", "")

        return super().__new__(cls, clsname, bases, dct)

    @property
    def __doc__(cls):
        handler = cls.handler
        doc = cls._orig_doc
        if cls is not Predicate and handler is not None:
            doc += "Handler\n"
            doc += "    =======\n\n"

            # Append the handler's doc without breaking sphinx documentation.
            docs = ["    Multiply dispatched method: %s" % handler.name]
            if handler.doc:
                for line in handler.doc.splitlines():
                    if not line:
                        continue
                    docs.append("    %s" % line)
            other = []
            for sig in handler.ordering[::-1]:
                func = handler.funcs[sig]
                if func.__doc__:
                    s = '    Inputs: <%s>' % str_signature(sig)
                    lines = []
                    for line in func.__doc__.splitlines():
                        lines.append("    %s" % line)
                    s += "\n".join(lines)
                    docs.append(s)
                else:
                    other.append(str_signature(sig))
            if other:
                othersig = "    Other signatures:"
                for line in other:
                    othersig += "\n        * %s" % line
                docs.append(othersig)

            doc += '\n\n'.join(docs)

        return doc


class Predicate(Boolean, metaclass=PredicateMeta):
    """
    Base class for mathematical predicates. It also serves as a
    constructor for undefined predicate objects.

    Explanation
    ===========

    Predicate is a function that returns a boolean value [1].

    Predicate function is object, and it is instance of predicate class.
    When a predicate is applied to arguments, ``AppliedPredicate``
    instance is returned. This merely wraps the argument and remain
    unevaluated. To obtain the truth value of applied predicate, use the
    function ``ask``.

    Evaluation of predicate is done by multiple dispatching. You can
    register new handler to the predicate to support new types.

    Every predicate in SymPy can be accessed via the property of ``Q``.
    For example, ``Q.even`` returns the predicate which checks if the
    argument is even number.

    To define a predicate which can be evaluated, you must subclass this
    class, make an instance of it, and register it to ``Q``. After then,
    dispatch the handler by argument types.

    If you directly construct predicate using this class, you will get
    ``UndefinedPredicate`` which cannot be dispatched. This is useful
    when you are building boolean expressions which do not need to be
    evaluated.

    Examples
    ========

    Applying and evaluating to boolean value:

    >>> from sympy import Q, ask
    >>> ask(Q.prime(7))
    True

    You can define a new predicate by subclassing and dispatching. Here,
    we define a predicate for sexy primes [2] as an example.

    >>> from sympy import Predicate, Integer
    >>> class SexyPrimePredicate(Predicate):
    ...     name = "sexyprime"
    >>> Q.sexyprime = SexyPrimePredicate()
    >>> @Q.sexyprime.register(Integer, Integer)
    ... def _(int1, int2, assumptions):
    ...     args = sorted([int1, int2])
    ...     if not all(ask(Q.prime(a), assumptions) for a in args):
    ...         return False
    ...     return args[1] - args[0] == 6
    >>> ask(Q.sexyprime(5, 11))
    True

    Direct constructing returns ``UndefinedPredicate``, which can be
    applied but cannot be dispatched.

    >>> from sympy import Predicate, Integer
    >>> Q.P = Predicate("P")
    >>> type(Q.P)
    <class 'sympy.assumptions.assume.UndefinedPredicate'>
    >>> Q.P(1)
    Q.P(1)
    >>> Q.P.register(Integer)(lambda expr, assump: True)
    Traceback (most recent call last):
      ...
    TypeError: <class 'sympy.assumptions.assume.UndefinedPredicate'> cannot be dispatched.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Predicate_%28mathematical_logic%29
    .. [2] https://en.wikipedia.org/wiki/Sexy_prime

    """

    is_Atom = True

    def __new__(cls, *args, **kwargs):
        if cls is Predicate:
            return UndefinedPredicate(*args, **kwargs)
        obj = super().__new__(cls, *args)
        return obj

    @property
    def name(self):
        # May be overridden
        return type(self).__name__

    @classmethod
    def register(cls, *types, **kwargs):
        """
        Register the signature to the handler.
        """
        if cls.handler is None:
            raise TypeError("%s cannot be dispatched." % type(cls))
        return cls.handler.register(*types, **kwargs)

    @classmethod
    def register_many(cls, *types, **kwargs):
        """
        Register multiple signatures to same handler.
        """
        def _(func):
            for t in types:
                if not is_sequence(t):
                    t = (t,)  # for convenience, allow passing `type` to mean `(type,)`
                cls.register(*t, **kwargs)(func)
        return _

    def __call__(self, *args):
        return AppliedPredicate(self, *args)

    def eval(self, args, assumptions=True):
        """
        Evaluate ``self(*args)`` under the given assumptions.

        This uses only direct resolution methods, not logical inference.
        """
        result = None
        try:
            result = self.handler(*args, assumptions=assumptions)
        except NotImplementedError:
            pass
        return result

    def _eval_refine(self, assumptions):
        # When Predicate is no longer Boolean, delete this method
        return self


class UndefinedPredicate(Predicate):
    """
    Predicate without handler.

    Explanation
    ===========

    This predicate is generated by using ``Predicate`` directly for
    construction. It does not have a handler, and evaluating this with
    arguments is done by SAT solver.

    Examples
    ========

    >>> from sympy import Predicate, Q
    >>> Q.P = Predicate('P')
    >>> Q.P.func
    <class 'sympy.assumptions.assume.UndefinedPredicate'>
    >>> Q.P.name
    Str('P')

    """

    handler = None

    def __new__(cls, name, handlers=None):
        # "handlers" parameter supports old design
        if not isinstance(name, Str):
            name = Str(name)
        obj = super(Boolean, cls).__new__(cls, name)
        obj.handlers = handlers or []
        return obj

    @property
    def name(self):
        return self.args[0]

    def _hashable_content(self):
        return (self.name,)

    def __getnewargs__(self):
        return (self.name,)

    def __call__(self, expr):
        return AppliedPredicate(self, expr)

    def add_handler(self, handler):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. Predicate.add_handler()
            should be replaced with the multipledispatch handler of Predicate.
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        self.handlers.append(handler)

    def remove_handler(self, handler):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. Predicate.remove_handler()
            should be replaced with the multipledispatch handler of Predicate.
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        self.handlers.remove(handler)

    def eval(self, args, assumptions=True):
        # Support for deprecated design
        # When old design is removed, this will always return None
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. Evaluating UndefinedPredicate
            objects should be replaced with the multipledispatch handler of
            Predicate.
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
            stacklevel=5,
        )
        expr, = args
        res, _res = None, None
        mro = inspect.getmro(type(expr))
        for handler in self.handlers:
            cls = get_class(handler)
            for subclass in mro:
                eval_ = getattr(cls, subclass.__name__, None)
                if eval_ is None:
                    continue
                res = eval_(expr, assumptions)
                # Do not stop if value returned is None
                # Try to check for higher classes
                if res is None:
                    continue
                if _res is None:
                    _res = res
                else:
                    # only check consistency if both resolutors have concluded
                    if _res != res:
                        raise ValueError('incompatible resolutors')
                break
        return res


@contextmanager
def assuming(*assumptions):
    """
    Context manager for assumptions.

    Examples
    ========

    >>> from sympy import assuming, Q, ask
    >>> from sympy.abc import x, y
    >>> print(ask(Q.integer(x + y)))
    None
    >>> with assuming(Q.integer(x), Q.integer(y)):
    ...     print(ask(Q.integer(x + y)))
    True
    """
    old_global_assumptions = global_assumptions.copy()
    global_assumptions.update(assumptions)
    try:
        yield
    finally:
        global_assumptions.clear()
        global_assumptions.update(old_global_assumptions)
<|endoftext|>"""
Known facts in assumptions module.

This module defines the facts between unary predicates in ``get_known_facts()``,
and supports functions to generate the contents in
``sympy.assumptions.ask_generated`` file.
"""

from sympy.assumptions.ask import Q
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,
    Exclusive,)
from sympy.logic.inference import satisfiable


@cacheit
def get_composite_predicates():
    # To reduce the complexity of sat solver, these predicates are
    # transformed into the combination of primitive predicates.
    return {
        Q.real : Q.negative | Q.zero | Q.positive,
        Q.integer : Q.even | Q.odd,
        Q.nonpositive : Q.negative | Q.zero,
        Q.nonzero : Q.negative | Q.positive,
        Q.nonnegative : Q.zero | Q.positive,
        Q.extended_real : Q.negative_infinite | Q.negative | Q.zero | Q.positive | Q.positive_infinite,
        Q.extended_positive: Q.positive | Q.positive_infinite,
        Q.extended_negative: Q.negative | Q.negative_infinite,
        Q.extended_nonzero: Q.negative_infinite | Q.negative | Q.positive | Q.positive_infinite,
        Q.extended_nonpositive: Q.negative_infinite | Q.negative | Q.zero,
        Q.extended_nonnegative: Q.zero | Q.positive | Q.positive_infinite,
        Q.complex : Q.algebraic | Q.transcendental
    }


@cacheit
def get_known_facts(x=None):
    """
    Facts between unary predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        get_number_facts(x),
        get_matrix_facts(x)
    )
    return fact


@cacheit
def get_number_facts(x = None):
    """
    Facts between unary number predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        # primitive predicates for extended real exclude each other.
        Exclusive(Q.negative_infinite(x), Q.negative(x), Q.zero(x),
            Q.positive(x), Q.positive_infinite(x)),

        # build complex plane
        Exclusive(Q.real(x), Q.imaginary(x)),
        Implies(Q.real(x) | Q.imaginary(x), Q.complex(x)),

        # other subsets of complex
        Exclusive(Q.transcendental(x), Q.algebraic(x)),
        Equivalent(Q.real(x), Q.rational(x) | Q.irrational(x)),
        Exclusive(Q.irrational(x), Q.rational(x)),
        Implies(Q.rational(x), Q.algebraic(x)),

        # integers
        Exclusive(Q.even(x), Q.odd(x)),
        Implies(Q.integer(x), Q.rational(x)),
        Implies(Q.zero(x), Q.even(x)),
        Exclusive(Q.composite(x), Q.prime(x)),
        Implies(Q.composite(x) | Q.prime(x), Q.integer(x) & Q.positive(x)),
        Implies(Q.even(x) & Q.positive(x) & ~Q.prime(x), Q.composite(x)),

        # hermitian and antihermitian
        Implies(Q.real(x), Q.hermitian(x)),
        Implies(Q.imaginary(x), Q.antihermitian(x)),
        Implies(Q.zero(x), Q.hermitian(x) | Q.antihermitian(x)),

        # define finity and infinity, and build extended real line
        Exclusive(Q.infinite(x), Q.finite(x)),
        Implies(Q.complex(x), Q.finite(x)),
        Implies(Q.negative_infinite(x) | Q.positive_infinite(x), Q.infinite(x)),

        # commutativity
        Implies(Q.finite(x) | Q.infinite(x), Q.commutative(x)),
    )
    return fact


@cacheit
def get_matrix_facts(x = None):
    """
    Facts between unary matrix predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        # matrices
        Implies(Q.orthogonal(x), Q.positive_definite(x)),
        Implies(Q.orthogonal(x), Q.unitary(x)),
        Implies(Q.unitary(x) & Q.real_elements(x), Q.orthogonal(x)),
        Implies(Q.unitary(x), Q.normal(x)),
        Implies(Q.unitary(x), Q.invertible(x)),
        Implies(Q.normal(x), Q.square(x)),
        Implies(Q.diagonal(x), Q.normal(x)),
        Implies(Q.positive_definite(x), Q.invertible(x)),
        Implies(Q.diagonal(x), Q.upper_triangular(x)),
        Implies(Q.diagonal(x), Q.lower_triangular(x)),
        Implies(Q.lower_triangular(x), Q.triangular(x)),
        Implies(Q.upper_triangular(x), Q.triangular(x)),
        Implies(Q.triangular(x), Q.upper_triangular(x) | Q.lower_triangular(x)),
        Implies(Q.upper_triangular(x) & Q.lower_triangular(x), Q.diagonal(x)),
        Implies(Q.diagonal(x), Q.symmetric(x)),
        Implies(Q.unit_triangular(x), Q.triangular(x)),
        Implies(Q.invertible(x), Q.fullrank(x)),
        Implies(Q.invertible(x), Q.square(x)),
        Implies(Q.symmetric(x), Q.square(x)),
        Implies(Q.fullrank(x) & Q.square(x), Q.invertible(x)),
        Equivalent(Q.invertible(x), ~Q.singular(x)),
        Implies(Q.integer_elements(x), Q.real_elements(x)),
        Implies(Q.real_elements(x), Q.complex_elements(x)),
    )
    return fact



def generate_known_facts_dict(keys, fact):
    """
    Computes and returns a dictionary which contains the relations between
    unary predicates.

    Each key is a predicate, and item is two groups of predicates.
    First group contains the predicates which are implied by the key, and
    second group contains the predicates which are rejected by the key.

    All predicates in *keys* and *fact* must be unary and have same placeholder
    symbol.

    Parameters
    ==========

    keys : list of AppliedPredicate instances.

    fact : Fact between predicates in conjugated normal form.

    Examples
    ========

    >>> from sympy import Q, And, Implies
    >>> from sympy.assumptions.facts import generate_known_facts_dict
    >>> from sympy.abc import x
    >>> keys = [Q.even(x), Q.odd(x), Q.zero(x)]
    >>> fact = And(Implies(Q.even(x), ~Q.odd(x)),
    ...     Implies(Q.zero(x), Q.even(x)))
    >>> generate_known_facts_dict(keys, fact)
    {Q.even: ({Q.even}, {Q.odd}),
     Q.odd: ({Q.odd}, {Q.even, Q.zero}),
     Q.zero: ({Q.even, Q.zero}, {Q.odd})}
    """
    fact_cnf = to_cnf(fact)
    mapping = single_fact_lookup(keys, fact_cnf)

    ret = {}
    for key, value in mapping.items():
        implied = set()
        rejected = set()
        for expr in value:
            if isinstance(expr, AppliedPredicate):
                implied.add(expr.function)
            elif isinstance(expr, Not):
                pred = expr.args[0]
                rejected.add(pred.function)
        ret[key.function] = (implied, rejected)
    return ret


@cacheit
def get_known_facts_keys():
    """
    Return every unary predicates registered to ``Q``.

    This function is used to generate the keys for
    ``generate_known_facts_dict``.

    """
    # exclude polyadic predicates
    exclude = {Q.eq, Q.ne, Q.gt, Q.lt, Q.ge, Q.le}

    result = []
    for attr in Q.__class__.__dict__:
        if attr.startswith('__'):
            continue
        pred = getattr(Q, attr)
        if pred in exclude:
            continue
        result.append(pred)
    return result


def single_fact_lookup(known_facts_keys, known_facts_cnf):
    # Return the dictionary for quick lookup of single fact
    mapping = {}
    for key in known_facts_keys:
        mapping[key] = {key}
        for other_key in known_facts_keys:
            if other_key != key:
                if ask_full_inference(other_key, key, known_facts_cnf):
                    mapping[key].add(other_key)
                if ask_full_inference(~other_key, key, known_facts_cnf):
                    mapping[key].add(~other_key)
    return mapping


def ask_full_inference(proposition, assumptions, known_facts_cnf):
    """
    Method for inferring properties about objects.

    """
    if not satisfiable(And(known_facts_cnf, assumptions, proposition)):
        return False
    if not satisfiable(And(known_facts_cnf, assumptions, Not(proposition))):
        return True
    return None
<|endoftext|>"""
A module to implement logical predicates and assumption system.
"""

from .assume import (
    AppliedPredicate, Predicate, AssumptionsContext, assuming,
    global_assumptions
)
from .ask import Q, ask, register_handler, remove_handler
from .refine import refine
from .relation import BinaryRelation, AppliedBinaryRelation

__all__ = [
    'AppliedPredicate', 'Predicate', 'AssumptionsContext', 'assuming',
    'global_assumptions', 'Q', 'ask', 'register_handler', 'remove_handler',
    'refine',
    'BinaryRelation', 'AppliedBinaryRelation'
]
<|endoftext|>"""
The classes used here are for the internal use of assumptions system
only and should not be used anywhere else as these do not possess the
signatures common to SymPy objects. For general use of logic constructs
please refer to sympy.logic classes And, Or, Not, etc.
"""
from itertools import combinations, product, zip_longest
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)


class Literal:
    """
    The smallest element of a CNF object.

    Parameters
    ==========

    lit : Boolean expression

    is_Not : bool

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import Literal
    >>> from sympy.abc import x
    >>> Literal(Q.even(x))
    Literal(Q.even(x), False)
    >>> Literal(~Q.even(x))
    Literal(Q.even(x), True)
    """

    def __new__(cls, lit, is_Not=False):
        if isinstance(lit, Not):
            lit = lit.args[0]
            is_Not = True
        elif isinstance(lit, (AND, OR, Literal)):
            return ~lit if is_Not else lit
        obj = super().__new__(cls)
        obj.lit = lit
        obj.is_Not = is_Not
        return obj

    @property
    def arg(self):
        return self.lit

    def rcall(self, expr):
        if callable(self.lit):
            lit = self.lit(expr)
        else:
            lit = self.lit.apply(expr)
        return type(self)(lit, self.is_Not)

    def __invert__(self):
        is_Not = not self.is_Not
        return Literal(self.lit, is_Not)

    def __str__(self):
        return '{}({}, {})'.format(type(self).__name__, self.lit, self.is_Not)

    __repr__ = __str__

    def __eq__(self, other):
        return self.arg == other.arg and self.is_Not == other.is_Not

    def __hash__(self):
        h = hash((type(self).__name__, self.arg, self.is_Not))
        return h


class OR:
    """
    A low-level implementation for Or
    """
    def __init__(self, *args):
        self._args = args

    @property
    def args(self):
        return sorted(self._args, key=str)

    def rcall(self, expr):
        return type(self)(*[arg.rcall(expr)
                            for arg in self._args
                            ])

    def __invert__(self):
        return AND(*[~arg for arg in self._args])

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(self, other):
        return self.args == other.args

    def __str__(self):
        s = '(' + ' | '.join([str(arg) for arg in self.args]) + ')'
        return s

    __repr__ = __str__


class AND:
    """
    A low-level implementation for And
    """
    def __init__(self, *args):
        self._args = args

    def __invert__(self):
        return OR(*[~arg for arg in self._args])

    @property
    def args(self):
        return sorted(self._args, key=str)

    def rcall(self, expr):
        return type(self)(*[arg.rcall(expr)
                            for arg in self._args
                            ])

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(self, other):
        return self.args == other.args

    def __str__(self):
        s = '('+' & '.join([str(arg) for arg in self.args])+')'
        return s

    __repr__ = __str__


def to_NNF(expr, composite_map=None):
    """
    Generates the Negation Normal Form of any boolean expression in terms
    of AND, OR, and Literal objects.

    Examples
    ========

    >>> from sympy import Q, Eq
    >>> from sympy.assumptions.cnf import to_NNF
    >>> from sympy.abc import x, y
    >>> expr = Q.even(x) & ~Q.positive(x)
    >>> to_NNF(expr)
    (Literal(Q.even(x), False) & Literal(Q.positive(x), True))

    Supported boolean objects are converted to corresponding predicates.

    >>> to_NNF(Eq(x, y))
    Literal(Q.eq(x, y), False)

    If ``composite_map`` argument is given, ``to_NNF`` decomposes the
    specified predicate into a combination of primitive predicates.

    >>> cmap = {Q.nonpositive: Q.negative | Q.zero}
    >>> to_NNF(Q.nonpositive, cmap)
    (Literal(Q.negative, False) | Literal(Q.zero, False))
    >>> to_NNF(Q.nonpositive(x), cmap)
    (Literal(Q.negative(x), False) | Literal(Q.zero(x), False))
    """
    from sympy.assumptions.ask import Q

    if composite_map is None:
        composite_map = {}


    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
    if type(expr) in binrelpreds:
        pred = binrelpreds[type(expr)]
        expr = pred(*expr.args)

    if isinstance(expr, Not):
        arg = expr.args[0]
        tmp = to_NNF(arg, composite_map)  # Strategy: negate the NNF of expr
        return ~tmp

    if isinstance(expr, Or):
        return OR(*[to_NNF(x, composite_map) for x in Or.make_args(expr)])

    if isinstance(expr, And):
        return AND(*[to_NNF(x, composite_map) for x in And.make_args(expr)])

    if isinstance(expr, Nand):
        tmp = AND(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp

    if isinstance(expr, Nor):
        tmp = OR(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp

    if isinstance(expr, Xor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map)
                          for s in expr.args]
                cnfs.append(OR(*clause))
        return AND(*cnfs)

    if isinstance(expr, Xnor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map)
                          for s in expr.args]
                cnfs.append(OR(*clause))
        return ~AND(*cnfs)

    if isinstance(expr, Implies):
        L, R = to_NNF(expr.args[0], composite_map), to_NNF(expr.args[1], composite_map)
        return OR(~L, R)

    if isinstance(expr, Equivalent):
        cnfs = []
        for a, b in zip_longest(expr.args, expr.args[1:], fillvalue=expr.args[0]):
            a = to_NNF(a, composite_map)
            b = to_NNF(b, composite_map)
            cnfs.append(OR(~a, b))
        return AND(*cnfs)

    if isinstance(expr, ITE):
        L = to_NNF(expr.args[0], composite_map)
        M = to_NNF(expr.args[1], composite_map)
        R = to_NNF(expr.args[2], composite_map)
        return AND(OR(~L, M), OR(L, R))

    if isinstance(expr, AppliedPredicate):
        pred, args = expr.function, expr.arguments
        newpred = composite_map.get(pred, None)
        if newpred is not None:
            return to_NNF(newpred.rcall(*args), composite_map)

    if isinstance(expr, Predicate):
        newpred = composite_map.get(expr, None)
        if newpred is not None:
            return to_NNF(newpred, composite_map)

    return Literal(expr)


def distribute_AND_over_OR(expr):
    """
    Distributes AND over OR in the NNF expression.
    Returns the result( Conjunctive Normal Form of expression)
    as a CNF object.
    """
    if not isinstance(expr, (AND, OR)):
        tmp = set()
        tmp.add(frozenset((expr,)))
        return CNF(tmp)

    if isinstance(expr, OR):
        return CNF.all_or(*[distribute_AND_over_OR(arg)
                            for arg in expr._args])

    if isinstance(expr, AND):
        return CNF.all_and(*[distribute_AND_over_OR(arg)
                             for arg in expr._args])


class CNF:
    """
    Class to represent CNF of a Boolean expression.
    Consists of set of clauses, which themselves are stored as
    frozenset of Literal objects.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.abc import x
    >>> cnf = CNF.from_prop(Q.real(x) & ~Q.zero(x))
    >>> cnf.clauses
    {frozenset({Literal(Q.zero(x), True)}),
    frozenset({Literal(Q.negative(x), False),
    Literal(Q.positive(x), False), Literal(Q.zero(x), False)})}
    """
    def __init__(self, clauses=None):
        if not clauses:
            clauses = set()
        self.clauses = clauses

    def add(self, prop):
        clauses = CNF.to_CNF(prop).clauses
        self.add_clauses(clauses)

    def __str__(self):
        s = ' & '.join(
            ['(' + ' | '.join([str(lit) for lit in clause]) +')'
            for clause in self.clauses]
        )
        return s

    def extend(self, props):
        for p in props:
            self.add(p)
        return self

    def copy(self):
        return CNF(set(self.clauses))

    def add_clauses(self, clauses):
        self.clauses |= clauses

    @classmethod
    def from_prop(cls, prop):
        res = cls()
        res.add(prop)
        return res

    def __iand__(self, other):
        self.add_clauses(other.clauses)
        return self

    def all_predicates(self):
        predicates = set()
        for c in self.clauses:
            predicates |= {arg.lit for arg in c}
        return predicates

    def _or(self, cnf):
        clauses = set()
        for a, b in product(self.clauses, cnf.clauses):
            tmp = set(a)
            tmp.update(b)
            clauses.add(frozenset(tmp))
        return CNF(clauses)

    def _and(self, cnf):
        clauses = self.clauses.union(cnf.clauses)
        return CNF(clauses)

    def _not(self):
        clss = list(self.clauses)
        ll = {frozenset((~x,)) for x in clss[-1]}
        ll = CNF(ll)

        for rest in clss[:-1]:
            p = {frozenset((~x,)) for x in rest}
            ll = ll._or(CNF(p))
        return ll

    def rcall(self, expr):
        clause_list = []
        for clause in self.clauses:
            lits = [arg.rcall(expr) for arg in clause]
            clause_list.append(OR(*lits))
        expr = AND(*clause_list)
        return distribute_AND_over_OR(expr)

    @classmethod
    def all_or(cls, *cnfs):
        b = cnfs[0].copy()
        for rest in cnfs[1:]:
            b = b._or(rest)
        return b

    @classmethod
    def all_and(cls, *cnfs):
        b = cnfs[0].copy()
        for rest in cnfs[1:]:
            b = b._and(rest)
        return b

    @classmethod
    def to_CNF(cls, expr):
        from sympy.assumptions.facts import get_composite_predicates
        expr = to_NNF(expr, get_composite_predicates())
        expr = distribute_AND_over_OR(expr)
        return expr

    @classmethod
    def CNF_to_cnf(cls, cnf):
        """
        Converts CNF object to SymPy's boolean expression
        retaining the form of expression.
        """
        def remove_literal(arg):
            return Not(arg.lit) if arg.is_Not else arg.lit

        return And(*(Or(*(remove_literal(arg) for arg in clause)) for clause in cnf.clauses))


class EncodedCNF:
    """
    Class for encoding the CNF expression.
    """
    def __init__(self, data=None, encoding=None):
        if not data and not encoding:
            data = []
            encoding = {}
        self.data = data
        self.encoding = encoding
        self._symbols = list(encoding.keys())

    def from_cnf(self, cnf):
        self._symbols = list(cnf.all_predicates())
        n = len(self._symbols)
        self.encoding = dict(zip(self._symbols, range(1, n + 1)))
        self.data = [self.encode(clause) for clause in cnf.clauses]

    @property
    def symbols(self):
        return self._symbols

    @property
    def variables(self):
        return range(1, len(self._symbols) + 1)

    def copy(self):
        new_data = [set(clause) for clause in self.data]
        return EncodedCNF(new_data, dict(self.encoding))

    def add_prop(self, prop):
        cnf = CNF.from_prop(prop)
        self.add_from_cnf(cnf)

    def add_from_cnf(self, cnf):
        clauses = [self.encode(clause) for clause in cnf.clauses]
        self.data += clauses

    def encode_arg(self, arg):
        literal = arg.lit
        value = self.encoding.get(literal, None)
        if value is None:
            n = len(self._symbols)
            self._symbols.append(literal)
            value = self.encoding[literal] = n + 1
        if arg.is_Not:
            return -value
        else:
            return value

    def encode(self, clause):
        return {self.encode_arg(arg) if not arg.lit == S.false else 0 for arg in clause}
<|endoftext|>from collections import defaultdict

from sympy.assumptions.ask import Q
from sympy.core import (Add, Mul, Pow, Number, NumberSymbol, Symbol)
from sympy.core.numbers import ImaginaryUnit
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import (Equivalent, And, Or, Implies)
from sympy.matrices.expressions import MatMul

# APIs here may be subject to change


### Helper functions ###

def allargs(symbol, fact, expr):
    """
    Apply all arguments of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import allargs
    >>> from sympy.abc import x, y
    >>> allargs(x, Q.negative(x) | Q.positive(x), x*y)
    (Q.negative(x) | Q.positive(x)) & (Q.negative(y) | Q.positive(y))

    """
    return And(*[fact.subs(symbol, arg) for arg in expr.args])


def anyarg(symbol, fact, expr):
    """
    Apply any argument of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import anyarg
    >>> from sympy.abc import x, y
    >>> anyarg(x, Q.negative(x) & Q.positive(x), x*y)
    (Q.negative(x) & Q.positive(x)) | (Q.negative(y) & Q.positive(y))

    """
    return Or(*[fact.subs(symbol, arg) for arg in expr.args])


def exactlyonearg(symbol, fact, expr):
    """
    Apply exactly one argument of the expression to the fact structure.

    Parameters
    ==========

    symbol : Symbol
        A placeholder symbol.

    fact : Boolean
        Resulting ``Boolean`` expression.

    expr : Expr

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.sathandlers import exactlyonearg
    >>> from sympy.abc import x, y
    >>> exactlyonearg(x, Q.positive(x), x*y)
    (Q.positive(x) & ~Q.positive(y)) | (Q.positive(y) & ~Q.positive(x))

    """
    pred_args = [fact.subs(symbol, arg) for arg in expr.args]
    res = Or(*[And(pred_args[i], *[~lit for lit in pred_args[:i] +
        pred_args[i+1:]]) for i in range(len(pred_args))])
    return res


### Fact registry ###

class ClassFactRegistry:
    """
    Register handlers against classes.

    Explanation
    ===========

    ``register`` method registers the handler function for a class. Here,
    handler function should return a single fact. ``multiregister`` method
    registers the handler function for multiple classes. Here, handler function
    should return a container of multiple facts.

    ``registry(expr)`` returns a set of facts for *expr*.

    Examples
    ========

    Here, we register the facts for ``Abs``.

    >>> from sympy import Abs, Equivalent, Q
    >>> from sympy.assumptions.sathandlers import ClassFactRegistry
    >>> reg = ClassFactRegistry()
    >>> @reg.register(Abs)
    ... def f1(expr):
    ...     return Q.nonnegative(expr)
    >>> @reg.register(Abs)
    ... def f2(expr):
    ...     arg = expr.args[0]
    ...     return Equivalent(~Q.zero(arg), ~Q.zero(expr))

    Calling the registry with expression returns the defined facts for the
    expression.

    >>> from sympy.abc import x
    >>> reg(Abs(x))
    {Q.nonnegative(Abs(x)), Equivalent(~Q.zero(x), ~Q.zero(Abs(x)))}

    Multiple facts can be registered at once by ``multiregister`` method.

    >>> reg2 = ClassFactRegistry()
    >>> @reg2.multiregister(Abs)
    ... def _(expr):
    ...     arg = expr.args[0]
    ...     return [Q.even(arg) >> Q.even(expr), Q.odd(arg) >> Q.odd(expr)]
    >>> reg2(Abs(x))
    {Implies(Q.even(x), Q.even(Abs(x))), Implies(Q.odd(x), Q.odd(Abs(x)))}

    """
    def __init__(self):
        self.singlefacts = defaultdict(frozenset)
        self.multifacts = defaultdict(frozenset)

    def register(self, cls):
        def _(func):
            self.singlefacts[cls] |= {func}
            return func
        return _

    def multiregister(self, *classes):
        def _(func):
            for cls in classes:
                self.multifacts[cls] |= {func}
            return func
        return _

    def __getitem__(self, key):
        ret1 = self.singlefacts[key]
        for k in self.singlefacts:
            if issubclass(key, k):
                ret1 |= self.singlefacts[k]

        ret2 = self.multifacts[key]
        for k in self.multifacts:
            if issubclass(key, k):
                ret2 |= self.multifacts[k]

        return ret1, ret2

    def __call__(self, expr):
        ret = set()

        handlers1, handlers2 = self[type(expr)]

        ret.update(h(expr) for h in handlers1)
        for h in handlers2:
            ret.update(h(expr))
        return ret

class_fact_registry = ClassFactRegistry()



### Class fact registration ###

x = Symbol('x')

## Abs ##

@class_fact_registry.multiregister(Abs)
def _(expr):
    arg = expr.args[0]
    return [Q.nonnegative(expr),
            Equivalent(~Q.zero(arg), ~Q.zero(expr)),
            Q.even(arg) >> Q.even(expr),
            Q.odd(arg) >> Q.odd(expr),
            Q.integer(arg) >> Q.integer(expr),
            ]


### Add ##

@class_fact_registry.multiregister(Add)
def _(expr):
    return [allargs(x, Q.positive(x), expr) >> Q.positive(expr),
            allargs(x, Q.negative(x), expr) >> Q.negative(expr),
            allargs(x, Q.real(x), expr) >> Q.real(expr),
            allargs(x, Q.rational(x), expr) >> Q.rational(expr),
            allargs(x, Q.integer(x), expr) >> Q.integer(expr),
            exactlyonearg(x, ~Q.integer(x), expr) >> ~Q.integer(expr),
            ]

@class_fact_registry.register(Add)
def _(expr):
    allargs_real = allargs(x, Q.real(x), expr)
    onearg_irrational = exactlyonearg(x, Q.irrational(x), expr)
    return Implies(allargs_real, Implies(onearg_irrational, Q.irrational(expr)))


### Mul ###

@class_fact_registry.multiregister(Mul)
def _(expr):
    return [Equivalent(Q.zero(expr), anyarg(x, Q.zero(x), expr)),
            allargs(x, Q.positive(x), expr) >> Q.positive(expr),
            allargs(x, Q.real(x), expr) >> Q.real(expr),
            allargs(x, Q.rational(x), expr) >> Q.rational(expr),
            allargs(x, Q.integer(x), expr) >> Q.integer(expr),
            exactlyonearg(x, ~Q.rational(x), expr) >> ~Q.integer(expr),
            allargs(x, Q.commutative(x), expr) >> Q.commutative(expr),
            ]

@class_fact_registry.register(Mul)
def _(expr):
    # Implicitly assumes Mul has more than one arg
    # Would be allargs(x, Q.prime(x) | Q.composite(x)) except 1 is composite
    # More advanced prime assumptions will require inequalities, as 1 provides
    # a corner case.
    allargs_prime = allargs(x, Q.prime(x), expr)
    return Implies(allargs_prime, ~Q.prime(expr))

@class_fact_registry.register(Mul)
def _(expr):
    # General Case: Odd number of imaginary args implies mul is imaginary(To be implemented)
    allargs_imag_or_real = allargs(x, Q.imaginary(x) | Q.real(x), expr)
    onearg_imaginary = exactlyonearg(x, Q.imaginary(x), expr)
    return Implies(allargs_imag_or_real, Implies(onearg_imaginary, Q.imaginary(expr)))

@class_fact_registry.register(Mul)
def _(expr):
    allargs_real = allargs(x, Q.real(x), expr)
    onearg_irrational = exactlyonearg(x, Q.irrational(x), expr)
    return Implies(allargs_real, Implies(onearg_irrational, Q.irrational(expr)))

@class_fact_registry.register(Mul)
def _(expr):
    # Including the integer qualification means we don't need to add any facts
    # for odd, since the assumptions already know that every integer is
    # exactly one of even or odd.
    allargs_integer = allargs(x, Q.integer(x), expr)
    anyarg_even = anyarg(x, Q.even(x), expr)
    return Implies(allargs_integer, Equivalent(anyarg_even, Q.even(expr)))


### MatMul ###

@class_fact_registry.register(MatMul)
def _(expr):
    allargs_square = allargs(x, Q.square(x), expr)
    allargs_invertible = allargs(x, Q.invertible(x), expr)
    return Implies(allargs_square, Equivalent(Q.invertible(expr), allargs_invertible))


### Pow ###

@class_fact_registry.multiregister(Pow)
def _(expr):
    base, exp = expr.base, expr.exp
    return [
        (Q.real(base) & Q.even(exp) & Q.nonnegative(exp)) >> Q.nonnegative(expr),
        (Q.nonnegative(base) & Q.odd(exp) & Q.nonnegative(exp)) >> Q.nonnegative(expr),
        (Q.nonpositive(base) & Q.odd(exp) & Q.nonnegative(exp)) >> Q.nonpositive(expr),
        Equivalent(Q.zero(expr), Q.zero(base) & Q.positive(exp))
    ]


### Numbers ###

_old_assump_getters = {
    Q.positive: lambda o: o.is_positive,
    Q.zero: lambda o: o.is_zero,
    Q.negative: lambda o: o.is_negative,
    Q.rational: lambda o: o.is_rational,
    Q.irrational: lambda o: o.is_irrational,
    Q.even: lambda o: o.is_even,
    Q.odd: lambda o: o.is_odd,
    Q.imaginary: lambda o: o.is_imaginary,
    Q.prime: lambda o: o.is_prime,
    Q.composite: lambda o: o.is_composite,
}

@class_fact_registry.multiregister(Number, NumberSymbol, ImaginaryUnit)
def _(expr):
    ret = []
    for p, getter in _old_assump_getters.items():
        pred = p(expr)
        prop = getter(expr)
        if prop is not None:
            ret.append(Equivalent(pred, prop))
    return ret
<|endoftext|>"""
Module to evaluate the proposition with assumptions using SAT algorithm.
"""

from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.assumptions.ask_generated import get_all_known_matrix_facts, get_all_known_number_facts
from sympy.assumptions.assume import global_assumptions, AppliedPredicate
from sympy.assumptions.sathandlers import class_fact_registry
from sympy.core import oo
from sympy.logic.inference import satisfiable
from sympy.assumptions.cnf import CNF, EncodedCNF
from sympy.matrices.kind import MatrixKind


def satask(proposition, assumptions=True, context=global_assumptions,
        use_known_facts=True, iterations=oo):
    """
    Function to evaluate the proposition with assumptions using SAT algorithm.

    This function extracts every fact relevant to the expressions composing
    proposition and assumptions. For example, if a predicate containing
    ``Abs(x)`` is proposed, then ``Q.zero(Abs(x)) | Q.positive(Abs(x))``
    will be found and passed to SAT solver because ``Q.nonnegative`` is
    registered as a fact for ``Abs``.

    Proposition is evaluated to ``True`` or ``False`` if the truth value can be
    determined. If not, ``None`` is returned.

    Parameters
    ==========

    proposition : Any boolean expression.
        Proposition which will be evaluated to boolean value.

    assumptions : Any boolean expression, optional.
        Local assumptions to evaluate the *proposition*.

    context : AssumptionsContext, optional.
        Default assumptions to evaluate the *proposition*. By default,
        this is ``sympy.assumptions.global_assumptions`` variable.

    use_known_facts : bool, optional.
        If ``True``, facts from ``sympy.assumptions.ask_generated``
        module are passed to SAT solver as well.

    iterations : int, optional.
        Number of times that relevant facts are recursively extracted.
        Default is infinite times until no new fact is found.

    Returns
    =======

    ``True``, ``False``, or ``None``

    Examples
    ========

    >>> from sympy import Abs, Q
    >>> from sympy.assumptions.satask import satask
    >>> from sympy.abc import x
    >>> satask(Q.zero(Abs(x)), Q.zero(x))
    True

    """
    props = CNF.from_prop(proposition)
    _props = CNF.from_prop(~proposition)

    assumptions = CNF.from_prop(assumptions)

    context_cnf = CNF()
    if context:
        context_cnf = context_cnf.extend(context)

    sat = get_all_relevant_facts(props, assumptions, context_cnf,
        use_known_facts=use_known_facts, iterations=iterations)
    sat.add_from_cnf(assumptions)
    if context:
        sat.add_from_cnf(context_cnf)

    return check_satisfiability(props, _props, sat)


def check_satisfiability(prop, _prop, factbase):
    sat_true = factbase.copy()
    sat_false = factbase.copy()
    sat_true.add_from_cnf(prop)
    sat_false.add_from_cnf(_prop)
    can_be_true = satisfiable(sat_true)
    can_be_false = satisfiable(sat_false)

    if can_be_true and can_be_false:
        return None

    if can_be_true and not can_be_false:
        return True

    if not can_be_true and can_be_false:
        return False

    if not can_be_true and not can_be_false:
        # TODO: Run additional checks to see which combination of the
        # assumptions, global_assumptions, and relevant_facts are
        # inconsistent.
        raise ValueError("Inconsistent assumptions")


def extract_predargs(proposition, assumptions=None, context=None):
    """
    Extract every expression in the argument of predicates from *proposition*,
    *assumptions* and *context*.

    Parameters
    ==========

    proposition : sympy.assumptions.cnf.CNF

    assumptions : sympy.assumptions.cnf.CNF, optional.

    context : sympy.assumptions.cnf.CNF, optional.
        CNF generated from assumptions context.

    Examples
    ========

    >>> from sympy import Q, Abs
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.satask import extract_predargs
    >>> from sympy.abc import x, y
    >>> props = CNF.from_prop(Q.zero(Abs(x*y)))
    >>> assump = CNF.from_prop(Q.zero(x) & Q.zero(y))
    >>> extract_predargs(props, assump)
    {x, y, Abs(x*y)}

    """
    req_keys = find_symbols(proposition)
    keys = proposition.all_predicates()
    # XXX: We need this since True/False are not Basic
    lkeys = set()
    if assumptions:
        lkeys |= assumptions.all_predicates()
    if context:
        lkeys |= context.all_predicates()

    lkeys = lkeys - {S.true, S.false}
    tmp_keys = None
    while tmp_keys != set():
        tmp = set()
        for l in lkeys:
            syms = find_symbols(l)
            if (syms & req_keys) != set():
                tmp |= syms
        tmp_keys = tmp - req_keys
        req_keys |= tmp_keys
    keys |= {l for l in lkeys if find_symbols(l) & req_keys != set()}

    exprs = set()
    for key in keys:
        if isinstance(key, AppliedPredicate):
            exprs |= set(key.arguments)
        else:
            exprs.add(key)
    return exprs

def find_symbols(pred):
    """
    Find every :obj:`~.Symbol` in *pred*.

    Parameters
    ==========

    pred : sympy.assumptions.cnf.CNF, or any Expr.

    """
    if isinstance(pred, CNF):
        symbols = set()
        for a in pred.all_predicates():
            symbols |= find_symbols(a)
        return symbols
    return pred.atoms(Symbol)


def get_relevant_clsfacts(exprs, relevant_facts=None):
    """
    Extract relevant facts from the items in *exprs*. Facts are defined in
    ``assumptions.sathandlers`` module.

    This function is recursively called by ``get_all_relevant_facts()``.

    Parameters
    ==========

    exprs : set
        Expressions whose relevant facts are searched.

    relevant_facts : sympy.assumptions.cnf.CNF, optional.
        Pre-discovered relevant facts.

    Returns
    =======

    exprs : set
        Candidates for next relevant fact searching.

    relevant_facts : sympy.assumptions.cnf.CNF
        Updated relevant facts.

    Examples
    ========

    Here, we will see how facts relevant to ``Abs(x*y)`` are recursively
    extracted. On the first run, set containing the expression is passed
    without pre-discovered relevant facts. The result is a set containing
    candidates for next run, and ``CNF()`` instance containing facts
    which are relevant to ``Abs`` and its argument.

    >>> from sympy import Abs
    >>> from sympy.assumptions.satask import get_relevant_clsfacts
    >>> from sympy.abc import x, y
    >>> exprs = {Abs(x*y)}
    >>> exprs, facts = get_relevant_clsfacts(exprs)
    >>> exprs
    {x*y}
    >>> facts.clauses #doctest: +SKIP
    {frozenset({Literal(Q.odd(Abs(x*y)), False), Literal(Q.odd(x*y), True)}),
    frozenset({Literal(Q.zero(Abs(x*y)), False), Literal(Q.zero(x*y), True)}),
    frozenset({Literal(Q.even(Abs(x*y)), False), Literal(Q.even(x*y), True)}),
    frozenset({Literal(Q.zero(Abs(x*y)), True), Literal(Q.zero(x*y), False)}),
    frozenset({Literal(Q.even(Abs(x*y)), False),
                Literal(Q.odd(Abs(x*y)), False),
                Literal(Q.odd(x*y), True)}),
    frozenset({Literal(Q.even(Abs(x*y)), False),
                Literal(Q.even(x*y), True),
                Literal(Q.odd(Abs(x*y)), False)}),
    frozenset({Literal(Q.positive(Abs(x*y)), False),
                Literal(Q.zero(Abs(x*y)), False)})}

    We pass the first run's results to the second run, and get the expressions
    for next run and updated facts.

    >>> exprs, facts = get_relevant_clsfacts(exprs, relevant_facts=facts)
    >>> exprs
    {x, y}

    On final run, no more candidate is returned thus we know that all
    relevant facts are successfully retrieved.

    >>> exprs, facts = get_relevant_clsfacts(exprs, relevant_facts=facts)
    >>> exprs
    set()

    """
    if not relevant_facts:
        relevant_facts = CNF()

    newexprs = set()
    for expr in exprs:
        for fact in class_fact_registry(expr):
            newfact = CNF.to_CNF(fact)
            relevant_facts = relevant_facts._and(newfact)
            for key in newfact.all_predicates():
                if isinstance(key, AppliedPredicate):
                    newexprs |= set(key.arguments)

    return newexprs - exprs, relevant_facts


def get_all_relevant_facts(proposition, assumptions, context,
        use_known_facts=True, iterations=oo):
    """
    Extract all relevant facts from *proposition* and *assumptions*.

    This function extracts the facts by recursively calling
    ``get_relevant_clsfacts()``. Extracted facts are converted to
    ``EncodedCNF`` and returned.

    Parameters
    ==========

    proposition : sympy.assumptions.cnf.CNF
        CNF generated from proposition expression.

    assumptions : sympy.assumptions.cnf.CNF
        CNF generated from assumption expression.

    context : sympy.assumptions.cnf.CNF
        CNF generated from assumptions context.

    use_known_facts : bool, optional.
        If ``True``, facts from ``sympy.assumptions.ask_generated``
        module are encoded as well.

    iterations : int, optional.
        Number of times that relevant facts are recursively extracted.
        Default is infinite times until no new fact is found.

    Returns
    =======

    sympy.assumptions.cnf.EncodedCNF

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.satask import get_all_relevant_facts
    >>> from sympy.abc import x, y
    >>> props = CNF.from_prop(Q.nonzero(x*y))
    >>> assump = CNF.from_prop(Q.nonzero(x))
    >>> context = CNF.from_prop(Q.nonzero(y))
    >>> get_all_relevant_facts(props, assump, context) #doctest: +SKIP
    <sympy.assumptions.cnf.EncodedCNF at 0x7f09faa6ccd0>

    """
    # The relevant facts might introduce new keys, e.g., Q.zero(x*y) will
    # introduce the keys Q.zero(x) and Q.zero(y), so we need to run it until
    # we stop getting new things. Hopefully this strategy won't lead to an
    # infinite loop in the future.
    i = 0
    relevant_facts = CNF()
    all_exprs = set()
    while True:
        if i == 0:
            exprs = extract_predargs(proposition, assumptions, context)
        all_exprs |= exprs
        exprs, relevant_facts = get_relevant_clsfacts(exprs, relevant_facts)
        i += 1
        if i >= iterations:
            break
        if not exprs:
            break

    if use_known_facts:
        known_facts_CNF = CNF()

        if any(expr.kind == MatrixKind(NumberKind) for expr in all_exprs):
            known_facts_CNF.add_clauses(get_all_known_matrix_facts())
        # check for undefinedKind since kind system isn't fully implemented
        if any(((expr.kind == NumberKind) or (expr.kind == UndefinedKind)) for expr in all_exprs):
            known_facts_CNF.add_clauses(get_all_known_number_facts())

        kf_encoded = EncodedCNF()
        kf_encoded.from_cnf(known_facts_CNF)

        def translate_literal(lit, delta):
            if lit > 0:
                return lit + delta
            else:
                return lit - delta

        def translate_data(data, delta):
            return [{translate_literal(i, delta) for i in clause} for clause in data]
        data = []
        symbols = []
        n_lit = len(kf_encoded.symbols)
        for i, expr in enumerate(all_exprs):
            symbols += [pred(expr) for pred in kf_encoded.symbols]
            data += translate_data(kf_encoded.data, i * n_lit)

        encoding = dict(list(zip(symbols, range(1, len(symbols)+1))))
        ctx = EncodedCNF(data, encoding)
    else:
        ctx = EncodedCNF()

    ctx.add_from_cnf(relevant_facts)

    return ctx
<|endoftext|>from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher


class PrimePredicate(Predicate):
    """
    Prime number predicate.

    Explanation
    ===========

    ``ask(Q.prime(x))`` is true iff ``x`` is a natural number greater
    than 1 that has no positive divisors other than ``1`` and the
    number itself.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> ask(Q.prime(0))
    False
    >>> ask(Q.prime(1))
    False
    >>> ask(Q.prime(2))
    True
    >>> ask(Q.prime(20))
    False
    >>> ask(Q.prime(-3))
    False

    """
    name = 'prime'
    handler = Dispatcher(
        "PrimeHandler",
        doc=("Handler for key 'prime'. Test that an expression represents a prime"
        " number. When the expression is an exact number, the result (when True)"
        " is subject to the limitations of isprime() which is used to return the "
        "result.")
    )


class CompositePredicate(Predicate):
    """
    Composite number predicate.

    Explanation
    ===========

    ``ask(Q.composite(x))`` is true iff ``x`` is a positive integer and has
    at least one positive divisor other than ``1`` and the number itself.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> ask(Q.composite(0))
    False
    >>> ask(Q.composite(1))
    False
    >>> ask(Q.composite(2))
    False
    >>> ask(Q.composite(20))
    True

    """
    name = 'composite'
    handler = Dispatcher("CompositeHandler", doc="Handler for key 'composite'.")


class EvenPredicate(Predicate):
    """
    Even number predicate.

    Explanation
    ===========

    ``ask(Q.even(x))`` is true iff ``x`` belongs to the set of even
    integers.

    Examples
    ========

    >>> from sympy import Q, ask, pi
    >>> ask(Q.even(0))
    True
    >>> ask(Q.even(2))
    True
    >>> ask(Q.even(3))
    False
    >>> ask(Q.even(pi))
    False

    """
    name = 'even'
    handler = Dispatcher("EvenHandler", doc="Handler for key 'even'.")


class OddPredicate(Predicate):
    """
    Odd number predicate.

    Explanation
    ===========

    ``ask(Q.odd(x))`` is true iff ``x`` belongs to the set of odd numbers.

    Examples
    ========

    >>> from sympy import Q, ask, pi
    >>> ask(Q.odd(0))
    False
    >>> ask(Q.odd(2))
    False
    >>> ask(Q.odd(3))
    True
    >>> ask(Q.odd(pi))
    False

    """
    name = 'odd'
    handler = Dispatcher(
        "OddHandler",
        doc=("Handler for key 'odd'. Test that an expression represents an odd"
        " number.")
    )
<|endoftext|>from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher

class FinitePredicate(Predicate):
    """
    Finite number predicate.

    Explanation
    ===========

    ``Q.finite(x)`` is true if ``x`` is a number but neither an infinity
    nor a ``NaN``. In other words, ``ask(Q.finite(x))`` is true for all
    numerical ``x`` having a bounded absolute value.

    Examples
    ========

    >>> from sympy import Q, ask, S, oo, I, zoo
    >>> from sympy.abc import x
    >>> ask(Q.finite(oo))
    False
    >>> ask(Q.finite(-oo))
    False
    >>> ask(Q.finite(zoo))
    False
    >>> ask(Q.finite(1))
    True
    >>> ask(Q.finite(2 + 3*I))
    True
    >>> ask(Q.finite(x), Q.positive(x))
    True
    >>> print(ask(Q.finite(S.NaN)))
    None

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Finite

    """
    name = 'finite'
    handler = Dispatcher(
        "FiniteHandler",
        doc=("Handler for Q.finite. Test that an expression is bounded respect"
        " to all its variables.")
    )


class InfinitePredicate(Predicate):
    """
    Infinite number predicate.

    ``Q.infinite(x)`` is true iff the absolute value of ``x`` is
    infinity.

    """
    # TODO: Add examples
    name = 'infinite'
    handler = Dispatcher(
        "InfiniteHandler",
        doc="""Handler for Q.infinite key."""
    )


class PositiveInfinitePredicate(Predicate):
    """
    Positive infinity predicate.

    ``Q.positive_infinite(x)`` is true iff ``x`` is positive infinity ``oo``.
    """
    name = 'positive_infinite'
    handler = Dispatcher("PositiveInfiniteHandler")


class NegativeInfinitePredicate(Predicate):
    """
    Negative infinity predicate.

    ``Q.negative_infinite(x)`` is true iff ``x`` is negative infinity ``-oo``.
    """
    name = 'negative_infinite'
    handler = Dispatcher("NegativeInfiniteHandler")
<|endoftext|>from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher


class NegativePredicate(Predicate):
    r"""
    Negative number predicate.

    Explanation
    ===========

    ``Q.negative(x)`` is true iff ``x`` is a real number and :math:`x < 0`, that is,
    it is in the interval :math:`(-\infty, 0)`.  Note in particular that negative
    infinity is not negative.

    A few important facts about negative numbers:

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.negative(x), Q.real(x) & ~Q.positive(x) & ~Q.zero(x))
    True
    >>> ask(Q.negative(-1))
    True
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(~Q.negative(I))
    True

    """
    name = 'negative'
    handler = Dispatcher(
        "NegativeHandler",
        doc=("Handler for Q.negative. Test that an expression is strictly less"
        " than zero.")
    )


class NonNegativePredicate(Predicate):
    """
    Nonnegative real number predicate.

    Explanation
    ===========

    ``ask(Q.nonnegative(x))`` is true iff ``x`` belongs to the set of
    positive numbers including zero.

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.nonnegative(1))
    True
    >>> ask(Q.nonnegative(0))
    True
    >>> ask(Q.nonnegative(-1))
    False
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(Q.nonnegative(-I))
    False

    """
    name = 'nonnegative'
    handler = Dispatcher(
        "NonNegativeHandler",
        doc=("Handler for Q.nonnegative.")
    )


class NonZeroPredicate(Predicate):
    """
    Nonzero real number predicate.

    Explanation
    ===========

    ``ask(Q.nonzero(x))`` is true iff ``x`` is real and ``x`` is not zero.  Note in
    particular that ``Q.nonzero(x)`` is false if ``x`` is not real.  Use
    ``~Q.zero(x)`` if you want the negation of being zero without any real
    assumptions.

    A few important facts about nonzero numbers:

    - ``Q.nonzero`` is logically equivalent to ``Q.positive | Q.negative``.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I, oo
    >>> x = symbols('x')
    >>> print(ask(Q.nonzero(x), ~Q.zero(x)))
    None
    >>> ask(Q.nonzero(x), Q.positive(x))
    True
    >>> ask(Q.nonzero(x), Q.zero(x))
    False
    >>> ask(Q.nonzero(0))
    False
    >>> ask(Q.nonzero(I))
    False
    >>> ask(~Q.zero(I))
    True
    >>> ask(Q.nonzero(oo))
    False

    """
    name = 'nonzero'
    handler = Dispatcher(
        "NonZeroHandler",
        doc=("Handler for key 'nonzero'. Test that an expression is not identically"
        " zero.")
    )


class ZeroPredicate(Predicate):
    """
    Zero number predicate.

    Explanation
    ===========

    ``ask(Q.zero(x))`` is true iff the value of ``x`` is zero.

    Examples
    ========

    >>> from sympy import ask, Q, oo, symbols
    >>> x, y = symbols('x, y')
    >>> ask(Q.zero(0))
    True
    >>> ask(Q.zero(1/oo))
    True
    >>> print(ask(Q.zero(0*oo)))
    None
    >>> ask(Q.zero(1))
    False
    >>> ask(Q.zero(x*y), Q.zero(x) | Q.zero(y))
    True

    """
    name = 'zero'
    handler = Dispatcher(
        "ZeroHandler",
        doc="Handler for key 'zero'."
    )


class NonPositivePredicate(Predicate):
    """
    Nonpositive real number predicate.

    Explanation
    ===========

    ``ask(Q.nonpositive(x))`` is true iff ``x`` belongs to the set of
    negative numbers including zero.

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I

    >>> ask(Q.nonpositive(-1))
    True
    >>> ask(Q.nonpositive(0))
    True
    >>> ask(Q.nonpositive(1))
    False
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(Q.nonpositive(-I))
    False

    """
    name = 'nonpositive'
    handler = Dispatcher(
        "NonPositiveHandler",
        doc="Handler for key 'nonpositive'."
    )


class PositivePredicate(Predicate):
    r"""
    Positive real number predicate.

    Explanation
    ===========

    ``Q.positive(x)`` is true iff ``x`` is real and `x > 0`, that is if ``x``
    is in the interval `(0, \infty)`.  In particular, infinity is not
    positive.

    A few important facts about positive numbers:

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.positive(x), Q.real(x) & ~Q.negative(x) & ~Q.zero(x))
    True
    >>> ask(Q.positive(1))
    True
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(~Q.positive(I))
    True

    """
    name = 'positive'
    handler = Dispatcher(
        "PositiveHandler",
        doc=("Handler for key 'positive'. Test that an expression is strictly"
        " greater than zero.")
    )


class ExtendedPositivePredicate(Predicate):
    r"""
    Positive extended real number predicate.

    Explanation
    ===========

    ``Q.extended_positive(x)`` is true iff ``x`` is extended real and
    `x > 0`, that is if ``x`` is in the interval `(0, \infty]`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_positive(1))
    True
    >>> ask(Q.extended_positive(oo))
    True
    >>> ask(Q.extended_positive(I))
    False

    """
    name = 'extended_positive'
    handler = Dispatcher("ExtendedPositiveHandler")


class ExtendedNegativePredicate(Predicate):
    r"""
    Negative extended real number predicate.

    Explanation
    ===========

    ``Q.extended_negative(x)`` is true iff ``x`` is extended real and
    `x < 0`, that is if ``x`` is in the interval `[-\infty, 0)`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_negative(-1))
    True
    >>> ask(Q.extended_negative(-oo))
    True
    >>> ask(Q.extended_negative(-I))
    False

    """
    name = 'extended_negative'
    handler = Dispatcher("ExtendedNegativeHandler")


class ExtendedNonZeroPredicate(Predicate):
    """
    Nonzero extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonzero(x))`` is true iff ``x`` is extended real and
    ``x`` is not zero.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonzero(-1))
    True
    >>> ask(Q.extended_nonzero(oo))
    True
    >>> ask(Q.extended_nonzero(I))
    False

    """
    name = 'extended_nonzero'
    handler = Dispatcher("ExtendedNonZeroHandler")


class ExtendedNonPositivePredicate(Predicate):
    """
    Nonpositive extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonpositive(x))`` is true iff ``x`` is extended real and
    ``x`` is not positive.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonpositive(-1))
    True
    >>> ask(Q.extended_nonpositive(oo))
    False
    >>> ask(Q.extended_nonpositive(0))
    True
    >>> ask(Q.extended_nonpositive(I))
    False

    """
    name = 'extended_nonpositive'
    handler = Dispatcher("ExtendedNonPositiveHandler")


class ExtendedNonNegativePredicate(Predicate):
    """
    Nonnegative extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonnegative(x))`` is true iff ``x`` is extended real and
    ``x`` is not negative.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonnegative(-1))
    False
    >>> ask(Q.extended_nonnegative(oo))
    True
    >>> ask(Q.extended_nonnegative(0))
    True
    >>> ask(Q.extended_nonnegative(I))
    False

    """
    name = 'extended_nonnegative'
    handler = Dispatcher("ExtendedNonNegativeHandler")
<|endoftext|>from sympy.assumptions import Predicate, AppliedPredicate, Q
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.multipledispatch import Dispatcher


class CommutativePredicate(Predicate):
    """
    Commutative predicate.

    Explanation
    ===========

    ``ask(Q.commutative(x))`` is true iff ``x`` commutes with any other
    object with respect to multiplication operation.

    """
    # TODO: Add examples
    name = 'commutative'
    handler = Dispatcher("CommutativeHandler", doc="Handler for key 'commutative'.")


binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}

class IsTruePredicate(Predicate):
    """
    Generic predicate.

    Explanation
    ===========

    ``ask(Q.is_true(x))`` is true iff ``x`` is true. This only makes
    sense if ``x`` is a boolean object.

    Examples
    ========

    >>> from sympy import ask, Q
    >>> from sympy.abc import x, y
    >>> ask(Q.is_true(True))
    True

    Wrapping another applied predicate just returns the applied predicate.

    >>> Q.is_true(Q.even(x))
    Q.even(x)

    Wrapping binary relation classes in SymPy core returns applied binary
    relational predicates.

    >>> from sympy import Eq, Gt
    >>> Q.is_true(Eq(x, y))
    Q.eq(x, y)
    >>> Q.is_true(Gt(x, y))
    Q.gt(x, y)

    Notes
    =====

    This class is designed to wrap the boolean objects so that they can
    behave as if they are applied predicates. Consequently, wrapping another
    applied predicate is unnecessary and thus it just returns the argument.
    Also, binary relation classes in SymPy core have binary predicates to
    represent themselves and thus wrapping them with ``Q.is_true`` converts them
    to these applied predicates.

    """
    name = 'is_true'
    handler = Dispatcher(
        "IsTrueHandler",
        doc="Wrapper allowing to query the truth value of a boolean expression."
    )

    def __call__(self, arg):
        # No need to wrap another predicate
        if isinstance(arg, AppliedPredicate):
            return arg
        # Convert relational predicates instead of wrapping them
        if getattr(arg, "is_Relational", False):
            pred = binrelpreds[type(arg)]
            return pred(*arg.args)
        return super().__call__(arg)
<|endoftext|>from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher


class IntegerPredicate(Predicate):
    """
    Integer predicate.

    Explanation
    ===========

    ``Q.integer(x)`` is true iff ``x`` belongs to the set of integer
    numbers.

    Examples
    ========

    >>> from sympy import Q, ask, S
    >>> ask(Q.integer(5))
    True
    >>> ask(Q.integer(S(1)/2))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Integer

    """
    name = 'integer'
    handler = Dispatcher(
        "IntegerHandler",
        doc=("Handler for Q.integer.\n\n"
        "Test that an expression belongs to the field of integer numbers.")
    )


class NonIntegerPredicate(Predicate):
    """
    Non-integer extended real predicate.
    """
    name = 'noninteger'
    handler = Dispatcher(
        "NonIntegerHandler",
        doc=("Handler for Q.noninteger.\n\n"
        "Test that an expression is a non-integer extended real number.")
    )


class RationalPredicate(Predicate):
    """
    Rational number predicate.

    Explanation
    ===========

    ``Q.rational(x)`` is true iff ``x`` belongs to the set of
    rational numbers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S
    >>> ask(Q.rational(0))
    True
    >>> ask(Q.rational(S(1)/2))
    True
    >>> ask(Q.rational(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rational_number

    """
    name = 'rational'
    handler = Dispatcher(
        "RationalHandler",
        doc=("Handler for Q.rational.\n\n"
        "Test that an expression belongs to the field of rational numbers.")
    )


class IrrationalPredicate(Predicate):
    """
    Irrational number predicate.

    Explanation
    ===========

    ``Q.irrational(x)`` is true iff ``x``  is any real number that
    cannot be expressed as a ratio of integers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S, I
    >>> ask(Q.irrational(0))
    False
    >>> ask(Q.irrational(S(1)/2))
    False
    >>> ask(Q.irrational(pi))
    True
    >>> ask(Q.irrational(I))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Irrational_number

    """
    name = 'irrational'
    handler = Dispatcher(
        "IrrationalHandler",
        doc=("Handler for Q.irrational.\n\n"
        "Test that an expression is irrational numbers.")
    )


class RealPredicate(Predicate):
    r"""
    Real number predicate.

    Explanation
    ===========

    ``Q.real(x)`` is true iff ``x`` is a real number, i.e., it is in the
    interval `(-\infty, \infty)`.  Note that, in particular the
    infinities are not real. Use ``Q.extended_real`` if you want to
    consider those as well.

    A few important facts about reals:

    - Every real number is positive, negative, or zero.  Furthermore,
        because these sets are pairwise disjoint, each real number is
        exactly one of those three.

    - Every real number is also complex.

    - Every real number is finite.

    - Every real number is either rational or irrational.

    - Every real number is either algebraic or transcendental.

    - The facts ``Q.negative``, ``Q.zero``, ``Q.positive``,
        ``Q.nonnegative``, ``Q.nonpositive``, ``Q.nonzero``,
        ``Q.integer``, ``Q.rational``, and ``Q.irrational`` all imply
        ``Q.real``, as do all facts that imply those facts.

    - The facts ``Q.algebraic``, and ``Q.transcendental`` do not imply
        ``Q.real``; they imply ``Q.complex``. An algebraic or
        transcendental number may or may not be real.

    - The "non" facts (i.e., ``Q.nonnegative``, ``Q.nonzero``,
        ``Q.nonpositive`` and ``Q.noninteger``) are not equivalent to
        not the fact, but rather, not the fact *and* ``Q.real``.
        For example, ``Q.nonnegative`` means ``~Q.negative & Q.real``.
        So for example, ``I`` is not nonnegative, nonzero, or
        nonpositive.

    Examples
    ========

    >>> from sympy import Q, ask, symbols
    >>> x = symbols('x')
    >>> ask(Q.real(x), Q.positive(x))
    True
    >>> ask(Q.real(0))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Real_number

    """
    name = 'real'
    handler = Dispatcher(
        "RealHandler",
        doc=("Handler for Q.real.\n\n"
        "Test that an expression belongs to the field of real numbers.")
    )


class ExtendedRealPredicate(Predicate):
    r"""
    Extended real predicate.

    Explanation
    ===========

    ``Q.extended_real(x)`` is true iff ``x`` is a real number or
    `\{-\infty, \infty\}`.

    See documentation of ``Q.real`` for more information about related
    facts.

    Examples
    ========

    >>> from sympy import ask, Q, oo, I
    >>> ask(Q.extended_real(1))
    True
    >>> ask(Q.extended_real(I))
    False
    >>> ask(Q.extended_real(oo))
    True

    """
    name = 'extended_real'
    handler = Dispatcher(
        "ExtendedRealHandler",
        doc=("Handler for Q.extended_real.\n\n"
        "Test that an expression belongs to the field of extended real\n"
        "numbers, that is real numbers union {Infinity, -Infinity}.")
    )


class HermitianPredicate(Predicate):
    """
    Hermitian predicate.

    Explanation
    ===========

    ``ask(Q.hermitian(x))`` is true iff ``x`` belongs to the set of
    Hermitian operators.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    # TODO: Add examples
    name = 'hermitian'
    handler = Dispatcher(
        "HermitianHandler",
        doc=("Handler for Q.hermitian.\n\n"
        "Test that an expression belongs to the field of Hermitian operators.")
    )


class ComplexPredicate(Predicate):
    """
    Complex number predicate.

    Explanation
    ===========

    ``Q.complex(x)`` is true iff ``x`` belongs to the set of complex
    numbers. Note that every complex number is finite.

    Examples
    ========

    >>> from sympy import Q, Symbol, ask, I, oo
    >>> x = Symbol('x')
    >>> ask(Q.complex(0))
    True
    >>> ask(Q.complex(2 + 3*I))
    True
    >>> ask(Q.complex(oo))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_number

    """
    name = 'complex'
    handler = Dispatcher(
        "ComplexHandler",
        doc=("Handler for Q.complex.\n\n"
        "Test that an expression belongs to the field of complex numbers.")
    )


class ImaginaryPredicate(Predicate):
    """
    Imaginary number predicate.

    Explanation
    ===========

    ``Q.imaginary(x)`` is true iff ``x`` can be written as a real
    number multiplied by the imaginary unit ``I``. Please note that ``0``
    is not considered to be an imaginary number.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.imaginary(3*I))
    True
    >>> ask(Q.imaginary(2 + 3*I))
    False
    >>> ask(Q.imaginary(0))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_number

    """
    name = 'imaginary'
    handler = Dispatcher(
        "ImaginaryHandler",
        doc=("Handler for Q.imaginary.\n\n"
        "Test that an expression belongs to the field of imaginary numbers,\n"
        "that is, numbers in the form x*I, where x is real.")
    )


class AntihermitianPredicate(Predicate):
    """
    Antihermitian predicate.

    Explanation
    ===========

    ``Q.antihermitian(x)`` is true iff ``x`` belongs to the field of
    antihermitian operators, i.e., operators in the form ``x*I``, where
    ``x`` is Hermitian.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    # TODO: Add examples
    name = 'antihermitian'
    handler = Dispatcher(
        "AntiHermitianHandler",
        doc=("Handler for Q.antihermitian.\n\n"
        "Test that an expression belongs to the field of anti-Hermitian\n"
        "operators, that is, operators in the form x*I, where x is Hermitian.")
    )


class AlgebraicPredicate(Predicate):
    r"""
    Algebraic number predicate.

    Explanation
    ===========

    ``Q.algebraic(x)`` is true iff ``x`` belongs to the set of
    algebraic numbers. ``x`` is algebraic if there is some polynomial
    in ``p(x)\in \mathbb\{Q\}[x]`` such that ``p(x) = 0``.

    Examples
    ========

    >>> from sympy import ask, Q, sqrt, I, pi
    >>> ask(Q.algebraic(sqrt(2)))
    True
    >>> ask(Q.algebraic(I))
    True
    >>> ask(Q.algebraic(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Algebraic_number

    """
    name = 'algebraic'
    AlgebraicHandler = Dispatcher(
        "AlgebraicHandler",
        doc="""Handler for Q.algebraic key."""
    )


class TranscendentalPredicate(Predicate):
    """
    Transcedental number predicate.

    Explanation
    ===========

    ``Q.transcendental(x)`` is true iff ``x`` belongs to the set of
    transcendental numbers. A transcendental number is a real
    or complex number that is not algebraic.

    """
    # TODO: Add examples
    name = 'transcendental'
    handler = Dispatcher(
        "Transcendental",
        doc="""Handler for Q.transcendental key."""
    )
<|endoftext|>from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher

class SquarePredicate(Predicate):
    """
    Square matrix predicate.

    Explanation
    ===========

    ``Q.square(x)`` is true iff ``x`` is a square matrix. A square matrix
    is a matrix with the same number of rows and columns.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('X', 2, 3)
    >>> ask(Q.square(X))
    True
    >>> ask(Q.square(Y))
    False
    >>> ask(Q.square(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.square(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_matrix

    """
    name = 'square'
    handler = Dispatcher("SquareHandler", doc="Handler for Q.square.")


class SymmetricPredicate(Predicate):
    """
    Symmetric matrix predicate.

    Explanation
    ===========

    ``Q.symmetric(x)`` is true iff ``x`` is a square matrix and is equal to
    its transpose. Every square diagonal matrix is a symmetric matrix.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.symmetric(X*Z), Q.symmetric(X) & Q.symmetric(Z))
    True
    >>> ask(Q.symmetric(X + Z), Q.symmetric(X) & Q.symmetric(Z))
    True
    >>> ask(Q.symmetric(Y))
    False


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Symmetric_matrix

    """
    # TODO: Add handlers to make these keys work with
    # actual matrices and add more examples in the docstring.
    name = 'symmetric'
    handler = Dispatcher("SymmetricHandler", doc="Handler for Q.symmetric.")


class InvertiblePredicate(Predicate):
    """
    Invertible matrix predicate.

    Explanation
    ===========

    ``Q.invertible(x)`` is true iff ``x`` is an invertible matrix.
    A square matrix is called invertible only if its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.invertible(X*Y), Q.invertible(X))
    False
    >>> ask(Q.invertible(X*Z), Q.invertible(X) & Q.invertible(Z))
    True
    >>> ask(Q.invertible(X), Q.fullrank(X) & Q.square(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Invertible_matrix

    """
    name = 'invertible'
    handler = Dispatcher("InvertibleHandler", doc="Handler for Q.invertible.")


class OrthogonalPredicate(Predicate):
    """
    Orthogonal matrix predicate.

    Explanation
    ===========

    ``Q.orthogonal(x)`` is true iff ``x`` is an orthogonal matrix.
    A square matrix ``M`` is an orthogonal matrix if it satisfies
    ``M^TM = MM^T = I`` where ``M^T`` is the transpose matrix of
    ``M`` and ``I`` is an identity matrix. Note that an orthogonal
    matrix is necessarily invertible.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.orthogonal(Y))
    False
    >>> ask(Q.orthogonal(X*Z*X), Q.orthogonal(X) & Q.orthogonal(Z))
    True
    >>> ask(Q.orthogonal(Identity(3)))
    True
    >>> ask(Q.invertible(X), Q.orthogonal(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Orthogonal_matrix

    """
    name = 'orthogonal'
    handler = Dispatcher("OrthogonalHandler", doc="Handler for key 'orthogonal'.")


class UnitaryPredicate(Predicate):
    """
    Unitary matrix predicate.

    Explanation
    ===========

    ``Q.unitary(x)`` is true iff ``x`` is a unitary matrix.
    Unitary matrix is an analogue to orthogonal matrix. A square
    matrix ``M`` with complex elements is unitary if :math:``M^TM = MM^T= I``
    where :math:``M^T`` is the conjugate transpose matrix of ``M``.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.unitary(Y))
    False
    >>> ask(Q.unitary(X*Z*X), Q.unitary(X) & Q.unitary(Z))
    True
    >>> ask(Q.unitary(Identity(3)))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Unitary_matrix

    """
    name = 'unitary'
    handler = Dispatcher("UnitaryHandler", doc="Handler for key 'unitary'.")


class FullRankPredicate(Predicate):
    """
    Fullrank matrix predicate.

    Explanation
    ===========

    ``Q.fullrank(x)`` is true iff ``x`` is a full rank matrix.
    A matrix is full rank if all rows and columns of the matrix
    are linearly independent. A square matrix is full rank iff
    its determinant is nonzero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.fullrank(X.T), Q.fullrank(X))
    True
    >>> ask(Q.fullrank(ZeroMatrix(3, 3)))
    False
    >>> ask(Q.fullrank(Identity(3)))
    True

    """
    name = 'fullrank'
    handler = Dispatcher("FullRankHandler", doc="Handler for key 'fullrank'.")


class PositiveDefinitePredicate(Predicate):
    r"""
    Positive definite matrix predicate.

    Explanation
    ===========

    If $M$ is a :math:`n \times n` symmetric real matrix, it is said
    to be positive definite if :math:`Z^TMZ` is positive for
    every non-zero column vector $Z$ of $n$ real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.positive_definite(Y))
    False
    >>> ask(Q.positive_definite(Identity(3)))
    True
    >>> ask(Q.positive_definite(X + Z), Q.positive_definite(X) &
    ...     Q.positive_definite(Z))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Positive-definite_matrix

    """
    name = "positive_definite"
    handler = Dispatcher("PositiveDefiniteHandler", doc="Handler for key 'positive_definite'.")


class UpperTriangularPredicate(Predicate):
    """
    Upper triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called upper triangular matrix if :math:`M_{ij}=0`
    for :math:`i<j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.upper_triangular(Identity(3)))
    True
    >>> ask(Q.upper_triangular(ZeroMatrix(3, 3)))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UpperTriangularMatrix.html

    """
    name = "upper_triangular"
    handler = Dispatcher("UpperTriangularHandler", doc="Handler for key 'upper_triangular'.")


class LowerTriangularPredicate(Predicate):
    """
    Lower triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called lower triangular matrix if :math:`M_{ij}=0`
    for :math:`i>j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.lower_triangular(Identity(3)))
    True
    >>> ask(Q.lower_triangular(ZeroMatrix(3, 3)))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/LowerTriangularMatrix.html

    """
    name = "lower_triangular"
    handler = Dispatcher("LowerTriangularHandler", doc="Handler for key 'lower_triangular'.")


class DiagonalPredicate(Predicate):
    """
    Diagonal matrix predicate.

    Explanation
    ===========

    ``Q.diagonal(x)`` is true iff ``x`` is a diagonal matrix. A diagonal
    matrix is a matrix in which the entries outside the main diagonal
    are all zero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.diagonal(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.diagonal(X), Q.lower_triangular(X) &
    ...     Q.upper_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Diagonal_matrix

    """
    name = "diagonal"
    handler = Dispatcher("DiagonalHandler", doc="Handler for key 'diagonal'.")


class IntegerElementsPredicate(Predicate):
    """
    Integer elements matrix predicate.

    Explanation
    ===========

    ``Q.integer_elements(x)`` is true iff all the elements of ``x``
    are integers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.integer(X[1, 2]), Q.integer_elements(X))
    True

    """
    name = "integer_elements"
    handler = Dispatcher("IntegerElementsHandler", doc="Handler for key 'integer_elements'.")


class RealElementsPredicate(Predicate):
    """
    Real elements matrix predicate.

    Explanation
    ===========

    ``Q.real_elements(x)`` is true iff all the elements of ``x``
    are real numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.real(X[1, 2]), Q.real_elements(X))
    True

    """
    name = "real_elements"
    handler = Dispatcher("RealElementsHandler", doc="Handler for key 'real_elements'.")


class ComplexElementsPredicate(Predicate):
    """
    Complex elements matrix predicate.

    Explanation
    ===========

    ``Q.complex_elements(x)`` is true iff all the elements of ``x``
    are complex numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.complex(X[1, 2]), Q.complex_elements(X))
    True
    >>> ask(Q.complex_elements(X), Q.integer_elements(X))
    True

    """
    name = "complex_elements"
    handler = Dispatcher("ComplexElementsHandler", doc="Handler for key 'complex_elements'.")


class SingularPredicate(Predicate):
    """
    Singular matrix predicate.

    A matrix is singular iff the value of its determinant is 0.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.singular(X), Q.invertible(X))
    False
    >>> ask(Q.singular(X), ~Q.invertible(X))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/SingularMatrix.html

    """
    name = "singular"
    handler = Dispatcher("SingularHandler", doc="Predicate fore key 'singular'.")


class NormalPredicate(Predicate):
    """
    Normal matrix predicate.

    A matrix is normal if it commutes with its conjugate transpose.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.normal(X), Q.unitary(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_matrix

    """
    name = "normal"
    handler = Dispatcher("NormalHandler", doc="Predicate fore key 'normal'.")


class TriangularPredicate(Predicate):
    """
    Triangular matrix predicate.

    Explanation
    ===========

    ``Q.triangular(X)`` is true if ``X`` is one that is either lower
    triangular or upper triangular.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.upper_triangular(X))
    True
    >>> ask(Q.triangular(X), Q.lower_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Triangular_matrix

    """
    name = "triangular"
    handler = Dispatcher("TriangularHandler", doc="Predicate fore key 'triangular'.")


class UnitTriangularPredicate(Predicate):
    """
    Unit triangular matrix predicate.

    Explanation
    ===========

    A unit triangular matrix is a triangular matrix with 1s
    on the diagonal.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.unit_triangular(X))
    True

    """
    name = "unit_triangular"
    handler = Dispatcher("UnitTriangularHandler", doc="Predicate fore key 'unit_triangular'.")
<|endoftext|>"""
Module for mathematical equality [1] and inequalities [2].

The purpose of this module is to provide the instances which represent the
binary predicates in order to combine the relationals into logical inference
system. Objects such as ``Q.eq``, ``Q.lt`` should remain internal to
assumptions module, and user must use the classes such as :obj:`~.Eq()`,
:obj:`~.Lt()` instead to construct the relational expressions.

References
==========

.. [1] https://en.wikipedia.org/wiki/Equality_(mathematics)
.. [2] https://en.wikipedia.org/wiki/Inequality_(mathematics)
"""
from sympy.assumptions import Q
from sympy.core.relational import is_eq, is_neq, is_gt, is_ge, is_lt, is_le

from .binrel import BinaryRelation

__all__ = ['EqualityPredicate', 'UnequalityPredicate', 'StrictGreaterThanPredicate',
    'GreaterThanPredicate', 'StrictLessThanPredicate', 'LessThanPredicate']


class EqualityPredicate(BinaryRelation):
    """
    Binary predicate for $=$.

    The purpose of this class is to provide the instance which represent
    the equality predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Eq()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_eq`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.eq(0, 0)
    Q.eq(0, 0)
    >>> ask(_)
    True

    See Also
    ========

    sympy.core.relational.Eq

    """
    is_reflexive = True
    is_symmetric = True

    name = 'eq'
    handler = None  # Do not allow dispatching by this predicate

    @property
    def negated(self):
        return Q.ne

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_eq is None
            assumptions = None
        return is_eq(*args, assumptions)


class UnequalityPredicate(BinaryRelation):
    r"""
    Binary predicate for $\neq$.

    The purpose of this class is to provide the instance which represent
    the inequation predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Ne()` instead to construct the inequation expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_neq`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.ne(0, 0)
    Q.ne(0, 0)
    >>> ask(_)
    False

    See Also
    ========

    sympy.core.relational.Ne

    """
    is_reflexive = False
    is_symmetric = True

    name = 'ne'
    handler = None

    @property
    def negated(self):
        return Q.eq

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_neq is None
            assumptions = None
        return is_neq(*args, assumptions)


class StrictGreaterThanPredicate(BinaryRelation):
    """
    Binary predicate for $>$.

    The purpose of this class is to provide the instance which represent
    the ">" predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Gt()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_gt`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.gt(0, 0)
    Q.gt(0, 0)
    >>> ask(_)
    False

    See Also
    ========

    sympy.core.relational.Gt

    """
    is_reflexive = False
    is_symmetric = False

    name = 'gt'
    handler = None

    @property
    def reversed(self):
        return Q.lt

    @property
    def negated(self):
        return Q.le

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_gt is None
            assumptions = None
        return is_gt(*args, assumptions)


class GreaterThanPredicate(BinaryRelation):
    """
    Binary predicate for $>=$.

    The purpose of this class is to provide the instance which represent
    the ">=" predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Ge()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_ge`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.ge(0, 0)
    Q.ge(0, 0)
    >>> ask(_)
    True

    See Also
    ========

    sympy.core.relational.Ge

    """
    is_reflexive = True
    is_symmetric = False

    name = 'ge'
    handler = None

    @property
    def reversed(self):
        return Q.le

    @property
    def negated(self):
        return Q.lt

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_ge is None
            assumptions = None
        return is_ge(*args, assumptions)


class StrictLessThanPredicate(BinaryRelation):
    """
    Binary predicate for $<$.

    The purpose of this class is to provide the instance which represent
    the "<" predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Lt()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_lt`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.lt(0, 0)
    Q.lt(0, 0)
    >>> ask(_)
    False

    See Also
    ========

    sympy.core.relational.Lt

    """
    is_reflexive = False
    is_symmetric = False

    name = 'lt'
    handler = None

    @property
    def reversed(self):
        return Q.gt

    @property
    def negated(self):
        return Q.ge

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_lt is None
            assumptions = None
        return is_lt(*args, assumptions)


class LessThanPredicate(BinaryRelation):
    """
    Binary predicate for $<=$.

    The purpose of this class is to provide the instance which represent
    the "<=" predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Le()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_le`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.le(0, 0)
    Q.le(0, 0)
    >>> ask(_)
    True

    See Also
    ========

    sympy.core.relational.Le

    """
    is_reflexive = True
    is_symmetric = False

    name = 'le'
    handler = None

    @property
    def reversed(self):
        return Q.ge

    @property
    def negated(self):
        return Q.gt

    def eval(self, args, assumptions=True):
        if assumptions == True:
            # default assumptions for is_le is None
            assumptions = None
        return is_le(*args, assumptions)
<|endoftext|>"""
General binary relations.
"""
from typing import Optional

from sympy.core.singleton import S
from sympy.assumptions import AppliedPredicate, ask, Predicate, Q  # type: ignore
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.boolalg import conjuncts, Not

__all__ = ["BinaryRelation", "AppliedBinaryRelation"]


class BinaryRelation(Predicate):
    """
    Base class for all binary relational predicates.

    Explanation
    ===========

    Binary relation takes two arguments and returns ``AppliedBinaryRelation``
    instance. To evaluate it to boolean value, use :obj:`~.ask()` or
    :obj:`~.refine()` function.

    You can add support for new types by registering the handler to dispatcher.
    See :obj:`~.Predicate()` for more information about predicate dispatching.

    Examples
    ========

    Applying and evaluating to boolean value:

    >>> from sympy import Q, ask, sin, cos
    >>> from sympy.abc import x
    >>> Q.eq(sin(x)**2+cos(x)**2, 1)
    Q.eq(sin(x)**2 + cos(x)**2, 1)
    >>> ask(_)
    True

    You can define a new binary relation by subclassing and dispatching.
    Here, we define a relation $R$ such that $x R y$ returns true if
    $x = y + 1$.

    >>> from sympy import ask, Number, Q
    >>> from sympy.assumptions import BinaryRelation
    >>> class MyRel(BinaryRelation):
    ...     name = "R"
    ...     is_reflexive = False
    >>> Q.R = MyRel()
    >>> @Q.R.register(Number, Number)
    ... def _(n1, n2, assumptions):
    ...     return ask(Q.zero(n1 - n2 - 1), assumptions)
    >>> Q.R(2, 1)
    Q.R(2, 1)

    Now, we can use ``ask()`` to evaluate it to boolean value.

    >>> ask(Q.R(2, 1))
    True
    >>> ask(Q.R(1, 2))
    False

    ``Q.R`` returns ``False`` with minimum cost if two arguments have same
    structure because it is antireflexive relation [1] by
    ``is_reflexive = False``.

    >>> ask(Q.R(x, x))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Reflexive_relation
    """

    is_reflexive: Optional[bool] = None
    is_symmetric: Optional[bool] = None

    def __call__(self, *args):
        if not len(args) == 2:
            raise ValueError("Binary relation takes two arguments, but got %s." % len(args))
        return AppliedBinaryRelation(self, *args)

    @property
    def reversed(self):
        if self.is_symmetric:
            return self
        return None

    @property
    def negated(self):
        return None

    def _compare_reflexive(self, lhs, rhs):
        # quick exit for structurally same arguments
        # do not check != here because it cannot catch the
        # equivalent arguments with different structures.

        # reflexivity does not hold to NaN
        if lhs is S.NaN or rhs is S.NaN:
            return None

        reflexive = self.is_reflexive
        if reflexive is None:
            pass
        elif reflexive and (lhs == rhs):
            return True
        elif not reflexive and (lhs == rhs):
            return False
        return None

    def eval(self, args, assumptions=True):
        # quick exit for structurally same arguments
        ret = self._compare_reflexive(*args)
        if ret is not None:
            return ret

        # don't perform simplify on args here. (done by AppliedBinaryRelation._eval_ask)
        # evaluate by multipledispatch
        lhs, rhs = args
        ret = self.handler(lhs, rhs, assumptions=assumptions)
        if ret is not None:
            return ret

        # check reversed order if the relation is reflexive
        if self.is_reflexive:
            types = (type(lhs), type(rhs))
            if self.handler.dispatch(*types) is not self.handler.dispatch(*reversed(types)):
                ret = self.handler(rhs, lhs, assumptions=assumptions)

        return ret


class AppliedBinaryRelation(AppliedPredicate):
    """
    The class of expressions resulting from applying ``BinaryRelation``
    to the arguments.

    """

    @property
    def lhs(self):
        """The left-hand side of the relation."""
        return self.arguments[0]

    @property
    def rhs(self):
        """The right-hand side of the relation."""
        return self.arguments[1]

    @property
    def reversed(self):
        """
        Try to return the relationship with sides reversed.
        """
        revfunc = self.function.reversed
        if revfunc is None:
            return self
        return revfunc(self.rhs, self.lhs)

    @property
    def reversedsign(self):
        """
        Try to return the relationship with signs reversed.
        """
        revfunc = self.function.reversed
        if revfunc is None:
            return self
        if not any(side.kind is BooleanKind for side in self.arguments):
            return revfunc(-self.lhs, -self.rhs)
        return self

    @property
    def negated(self):
        neg_rel = self.function.negated
        if neg_rel is None:
            return Not(self, evaluate=False)
        return neg_rel(*self.arguments)

    def _eval_ask(self, assumptions):
        conj_assumps = set()
        binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
        for a in conjuncts(assumptions):
            if a.func in binrelpreds:
                conj_assumps.add(binrelpreds[type(a)](*a.args))
            else:
                conj_assumps.add(a)

        # After CNF in assumptions module is modified to take polyadic
        # predicate, this will be removed
        if any(rel in conj_assumps for rel in (self, self.reversed)):
            return True
        neg_rels = (self.negated, self.reversed.negated, Not(self, evaluate=False),
            Not(self.reversed, evaluate=False))
        if any(rel in conj_assumps for rel in neg_rels):
            return False

        # evaluation using multipledispatching
        ret = self.function.eval(self.arguments, assumptions)
        if ret is not None:
            return ret

        # simplify the args and try again
        args = tuple(a.simplify() for a in self.arguments)
        return self.function.eval(args, assumptions)

    def __bool__(self):
        ret = ask(self)
        if ret is None:
            raise TypeError("Cannot determine truth value of %s" % self)
        return ret
<|endoftext|>"""
Handlers for keys related to number theory: prime, even, odd, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.core.numbers import (ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, NumberSymbol, Rational, int_valued)
from sympy.functions import Abs, im, re
from sympy.ntheory import isprime

from sympy.multipledispatch import MDNotImplementedError

from ..predicates.ntheory import (PrimePredicate, CompositePredicate,
    EvenPredicate, OddPredicate)


# PrimePredicate

def _PrimePredicate_number(expr, assumptions):
    # helper method
    exact = not expr.atoms(Float)
    try:
        i = int(expr.round())
        if (expr - i).equals(0) is False:
            raise TypeError
    except TypeError:
        return False
    if exact:
        return isprime(i)
    # when not exact, we won't give a True or False
    # since the number represents an approximate value

@PrimePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_prime
    if ret is None:
        raise MDNotImplementedError
    return ret

@PrimePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    for arg in expr.args:
        if not ask(Q.integer(arg), assumptions):
            return None
    for arg in expr.args:
        if arg.is_number and arg.is_composite:
            return False

@PrimePredicate.register(Pow)
def _(expr, assumptions):
    """
    Integer**Integer     -> !Prime
    """
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    if ask(Q.integer(expr.exp), assumptions) and \
            ask(Q.integer(expr.base), assumptions):
        prime_base = ask(Q.prime(expr.base), assumptions)
        if prime_base is False:
            return False
        is_exp_one = ask(Q.eq(expr.exp, 1), assumptions)
        if is_exp_one is False:
            return False
        if prime_base is True and is_exp_one is True:
            return True

@PrimePredicate.register(Integer)
def _(expr, assumptions):
    return isprime(expr)

@PrimePredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    return False

@PrimePredicate.register(Float)
def _(expr, assumptions):
    return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(NumberSymbol)
def _(expr, assumptions):
    return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(NaN)
def _(expr, assumptions):
    return None


# CompositePredicate

@CompositePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_composite
    if ret is None:
        raise MDNotImplementedError
    return ret

@CompositePredicate.register(Basic)
def _(expr, assumptions):
    _positive = ask(Q.positive(expr), assumptions)
    if _positive:
        _integer = ask(Q.integer(expr), assumptions)
        if _integer:
            _prime = ask(Q.prime(expr), assumptions)
            if _prime is None:
                return
            # Positive integer which is not prime is not
            # necessarily composite
            _is_one = ask(Q.eq(expr, 1), assumptions)
            if _is_one:
                return False
            if _is_one is None:
                return None
            return not _prime
        else:
            return _integer
    else:
        return _positive


# EvenPredicate

def _EvenPredicate_number(expr, assumptions):
    # helper method
    if isinstance(expr, (float, Float)):
        if int_valued(expr):
            return None
        return False
    try:
        i = int(expr.round())
    except TypeError:
        return False
    if not (expr - i).equals(0):
        return False
    return i % 2 == 0

@EvenPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_even
    if ret is None:
        raise MDNotImplementedError
    return ret

@EvenPredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)

@EvenPredicate.register(Mul)
def _(expr, assumptions):
    """
    Even * Integer    -> Even
    Even * Odd        -> Even
    Integer * Odd     -> ?
    Odd * Odd         -> Odd
    Even * Even       -> Even
    Integer * Integer -> Even if Integer + Integer = Odd
    otherwise         -> ?
    """
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    even, odd, irrational, acc = False, 0, False, 1
    for arg in expr.args:
        # check for all integers and at least one even
        if ask(Q.integer(arg), assumptions):
            if ask(Q.even(arg), assumptions):
                even = True
            elif ask(Q.odd(arg), assumptions):
                odd += 1
            elif not even and acc != 1:
                if ask(Q.odd(acc + arg), assumptions):
                    even = True
        elif ask(Q.irrational(arg), assumptions):
            # one irrational makes the result False
            # two makes it undefined
            if irrational:
                break
            irrational = True
        else:
            break
        acc = arg
    else:
        if irrational:
            return False
        if even:
            return True
        if odd == len(expr.args):
            return False

@EvenPredicate.register(Add)
def _(expr, assumptions):
    """
    Even + Odd  -> Odd
    Even + Even -> Even
    Odd  + Odd  -> Even

    """
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    _result = True
    for arg in expr.args:
        if ask(Q.even(arg), assumptions):
            pass
        elif ask(Q.odd(arg), assumptions):
            _result = not _result
        else:
            break
    else:
        return _result

@EvenPredicate.register(Pow)
def _(expr, assumptions):
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    if ask(Q.integer(expr.exp), assumptions):
        if ask(Q.positive(expr.exp), assumptions):
            return ask(Q.even(expr.base), assumptions)
        elif ask(~Q.negative(expr.exp) & Q.odd(expr.base), assumptions):
            return False
        elif expr.base is S.NegativeOne:
            return False

@EvenPredicate.register(Integer)
def _(expr, assumptions):
    return not bool(expr.p & 1)

@EvenPredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    return False

@EvenPredicate.register(NumberSymbol)
def _(expr, assumptions):
    return _EvenPredicate_number(expr, assumptions)

@EvenPredicate.register(Abs)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

@EvenPredicate.register(re)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

@EvenPredicate.register(im)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return True

@EvenPredicate.register(NaN)
def _(expr, assumptions):
    return None


# OddPredicate

@OddPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_odd
    if ret is None:
        raise MDNotImplementedError
    return ret

@OddPredicate.register(Basic)
def _(expr, assumptions):
    _integer = ask(Q.integer(expr), assumptions)
    if _integer:
        _even = ask(Q.even(expr), assumptions)
        if _even is None:
            return None
        return not _even
    return _integer
<|endoftext|>"""
This module contains query handlers responsible for calculus queries:
infinitesimal, finite, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Expr, Add, Mul, Pow, Symbol
from sympy.core.numbers import (NegativeInfinity, GoldenRatio,
    Infinity, Exp1, ComplexInfinity, ImaginaryUnit, NaN, Number, Pi, E,
    TribonacciConstant)
from sympy.functions import cos, exp, log, sign, sin
from sympy.logic.boolalg import conjuncts

from ..predicates.calculus import (FinitePredicate, InfinitePredicate,
    PositiveInfinitePredicate, NegativeInfinitePredicate)


# FinitePredicate


@FinitePredicate.register(Symbol)
def _(expr, assumptions):
    """
    Handles Symbol.
    """
    if expr.is_finite is not None:
        return expr.is_finite
    if Q.finite(expr) in conjuncts(assumptions):
        return True
    return None

@FinitePredicate.register(Add)
def _(expr, assumptions):
    """
    Return True if expr is bounded, False if not and None if unknown.

    Truth Table:

    +-------+-----+-----------+-----------+
    |       |     |           |           |
    |       |  B  |     U     |     ?     |
    |       |     |           |           |
    +-------+-----+---+---+---+---+---+---+
    |       |     |   |   |   |   |   |   |
    |       |     |'+'|'-'|'x'|'+'|'-'|'x'|
    |       |     |   |   |   |   |   |   |
    +-------+-----+---+---+---+---+---+---+
    |       |     |           |           |
    |   B   |  B  |     U     |     ?     |
    |       |     |           |           |
    +---+---+-----+---+---+---+---+---+---+
    |   |   |     |   |   |   |   |   |   |
    |   |'+'|     | U | ? | ? | U | ? | ? |
    |   |   |     |   |   |   |   |   |   |
    |   +---+-----+---+---+---+---+---+---+
    |   |   |     |   |   |   |   |   |   |
    | U |'-'|     | ? | U | ? | ? | U | ? |
    |   |   |     |   |   |   |   |   |   |
    |   +---+-----+---+---+---+---+---+---+
    |   |   |     |           |           |
    |   |'x'|     |     ?     |     ?     |
    |   |   |     |           |           |
    +---+---+-----+---+---+---+---+---+---+
    |       |     |           |           |
    |   ?   |     |           |     ?     |
    |       |     |           |           |
    +-------+-----+-----------+---+---+---+

        * 'B' = Bounded

        * 'U' = Unbounded

        * '?' = unknown boundedness

        * '+' = positive sign

        * '-' = negative sign

        * 'x' = sign unknown

        * All Bounded -> True

        * 1 Unbounded and the rest Bounded -> False

        * >1 Unbounded, all with same known sign -> False

        * Any Unknown and unknown sign -> None

        * Else -> None

    When the signs are not the same you can have an undefined
    result as in oo - oo, hence 'bounded' is also undefined.
    """
    sign = -1  # sign of unknown or infinite
    result = True
    for arg in expr.args:
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            continue
        s = ask(Q.extended_positive(arg), assumptions)
        # if there has been more than one sign or if the sign of this arg
        # is None and Bounded is None or there was already
        # an unknown sign, return None
        if sign != -1 and s != sign or \
                s is None and None in (_bounded, sign):
            return None
        else:
            sign = s
        # once False, do not change
        if result is not False:
            result = _bounded
    return result

@FinitePredicate.register(Mul)
def _(expr, assumptions):
    """
    Return True if expr is bounded, False if not and None if unknown.

    Truth Table:

    +---+---+---+--------+
    |   |   |   |        |
    |   | B | U |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+
    |   |   |   |   |    |
    |   |   |   | s | /s |
    |   |   |   |   |    |
    +---+---+---+---+----+
    |   |   |   |        |
    | B | B | U |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+
    |   |   |   |   |    |
    | U |   | U | U | ?  |
    |   |   |   |   |    |
    +---+---+---+---+----+
    |   |   |   |        |
    | ? |   |   |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+

        * B = Bounded

        * U = Unbounded

        * ? = unknown boundedness

        * s = signed (hence nonzero)

        * /s = not signed
    """
    result = True
    possible_zero = False
    for arg in expr.args:
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            if ask(Q.zero(arg), assumptions) is not False:
                if result is False:
                    return None
                possible_zero = True
        elif _bounded is None:
            if result is None:
                return None
            if ask(Q.extended_nonzero(arg), assumptions) is None:
                return None
            if result is not False:
                result = None
        else:
            if possible_zero:
                return None
            result = False
    return result

@FinitePredicate.register(Pow)
def _(expr, assumptions):
    """
    * Unbounded ** NonZero -> Unbounded

    * Bounded ** Bounded -> Bounded

    * Abs()<=1 ** Positive -> Bounded

    * Abs()>=1 ** Negative -> Bounded

    * Otherwise unknown
    """
    if expr.base == E:
        return ask(Q.finite(expr.exp), assumptions)

    base_bounded = ask(Q.finite(expr.base), assumptions)
    exp_bounded = ask(Q.finite(expr.exp), assumptions)
    if base_bounded is None and exp_bounded is None:  # Common Case
        return None
    if base_bounded is False and ask(Q.extended_nonzero(expr.exp), assumptions):
        return False
    if base_bounded and exp_bounded:
        is_base_zero = ask(Q.zero(expr.base),assumptions)
        is_exp_negative = ask(Q.negative(expr.exp),assumptions)
        if is_base_zero is True and is_exp_negative is True:
            return False
        if is_base_zero is not False and is_exp_negative is not False:
            return None
        return True
    if (abs(expr.base) <= 1) == True and ask(Q.extended_positive(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and ask(Q.extended_negative(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and exp_bounded is False:
        return False
    return None

@FinitePredicate.register(exp)
def _(expr, assumptions):
    return ask(Q.finite(expr.exp), assumptions)

@FinitePredicate.register(log)
def _(expr, assumptions):
    # After complex -> finite fact is registered to new assumption system,
    # querying Q.infinite may be removed.
    if ask(Q.infinite(expr.args[0]), assumptions):
        return False
    return ask(~Q.zero(expr.args[0]), assumptions)

@FinitePredicate.register_many(cos, sin, Number, Pi, Exp1, GoldenRatio,
    TribonacciConstant, ImaginaryUnit, sign)
def _(expr, assumptions):
    return True

@FinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions):
    return False

@FinitePredicate.register(NaN)
def _(expr, assumptions):
    return None


# InfinitePredicate


@InfinitePredicate.register(Expr)
def _(expr, assumptions):
    is_finite = Q.finite(expr)._eval_ask(assumptions)
    if is_finite is None:
        return None
    return not is_finite


# PositiveInfinitePredicate


@PositiveInfinitePredicate.register(Infinity)
def _(expr, assumptions):
    return True


@PositiveInfinitePredicate.register_many(NegativeInfinity, ComplexInfinity)
def _(expr, assumptions):
    return False


# NegativeInfinitePredicate


@NegativeInfinitePredicate.register(NegativeInfinity)
def _(expr, assumptions):
    return True


@NegativeInfinitePredicate.register_many(Infinity, ComplexInfinity)
def _(expr, assumptions):
    return False
<|endoftext|>"""
Handlers related to order relations: positive, negative, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Mul, Pow, S
from sympy.core.logic import fuzzy_not, fuzzy_and, fuzzy_or
from sympy.core.numbers import E, ImaginaryUnit, NaN, I, pi
from sympy.functions import Abs, acos, acot, asin, atan, exp, factorial, log
from sympy.matrices import Determinant, Trace
from sympy.matrices.expressions.matexpr import MatrixElement

from sympy.multipledispatch import MDNotImplementedError

from ..predicates.order import (NegativePredicate, NonNegativePredicate,
    NonZeroPredicate, ZeroPredicate, NonPositivePredicate, PositivePredicate,
    ExtendedNegativePredicate, ExtendedNonNegativePredicate,
    ExtendedNonPositivePredicate, ExtendedNonZeroPredicate,
    ExtendedPositivePredicate,)


# NegativePredicate

def _NegativePredicate_number(expr, assumptions):
    r, i = expr.as_real_imag()

    if r == S.NaN or i == S.NaN:
        return None

    # If the imaginary part can symbolically be shown to be zero then
    # we just evaluate the real part; otherwise we evaluate the imaginary
    # part to see if it actually evaluates to zero and if it does then
    # we make the comparison between the real part and zero.
    if not i:
        r = r.evalf(2)
        if r._prec != 1:
            return r < 0
    else:
        i = i.evalf(2)
        if i._prec != 1:
            if i != 0:
                return False
            r = r.evalf(2)
            if r._prec != 1:
                return r < 0

@NegativePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)

@NegativePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_negative
    if ret is None:
        raise MDNotImplementedError
    return ret

@NegativePredicate.register(Add)
def _(expr, assumptions):
    """
    Positive + Positive -> Positive,
    Negative + Negative -> Negative
    """
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)

    r = ask(Q.real(expr), assumptions)
    if r is not True:
        return r

    nonpos = 0
    for arg in expr.args:
        if ask(Q.negative(arg), assumptions) is not True:
            if ask(Q.positive(arg), assumptions) is False:
                nonpos += 1
            else:
                break
    else:
        if nonpos < len(expr.args):
            return True

@NegativePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)
    result = None
    for arg in expr.args:
        if result is None:
            result = False
        if ask(Q.negative(arg), assumptions):
            result = not result
        elif ask(Q.positive(arg), assumptions):
            pass
        else:
            return
    return result

@NegativePredicate.register(Pow)
def _(expr, assumptions):
    """
    Real ** Even -> NonNegative
    Real ** Odd  -> same_as_base
    NonNegative ** Positive -> NonNegative
    """
    if expr.base == E:
        # Exponential is always positive:
        if ask(Q.real(expr.exp), assumptions):
            return False
        return

    if expr.is_number:
        return _NegativePredicate_number(expr, assumptions)
    if ask(Q.real(expr.base), assumptions):
        if ask(Q.positive(expr.base), assumptions):
            if ask(Q.real(expr.exp), assumptions):
                return False
        if ask(Q.even(expr.exp), assumptions):
            return False
        if ask(Q.odd(expr.exp), assumptions):
            return ask(Q.negative(expr.base), assumptions)

@NegativePredicate.register_many(Abs, ImaginaryUnit)
def _(expr, assumptions):
    return False

@NegativePredicate.register(exp)
def _(expr, assumptions):
    if ask(Q.real(expr.exp), assumptions):
        return False
    raise MDNotImplementedError


# NonNegativePredicate

@NonNegativePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        notnegative = fuzzy_not(_NegativePredicate_number(expr, assumptions))
        if notnegative:
            return ask(Q.real(expr), assumptions)
        else:
            return notnegative

@NonNegativePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_nonnegative
    if ret is None:
        raise MDNotImplementedError
    return ret


# NonZeroPredicate

@NonZeroPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_nonzero
    if ret is None:
        raise MDNotImplementedError
    return ret

@NonZeroPredicate.register(Basic)
def _(expr, assumptions):
    if ask(Q.real(expr)) is False:
        return False
    if expr.is_number:
        # if there are no symbols just evalf
        i = expr.evalf(2)
        def nonz(i):
            if i._prec != 1:
                return i != 0
        return fuzzy_or(nonz(i) for i in i.as_real_imag())

@NonZeroPredicate.register(Add)
def _(expr, assumptions):
    if all(ask(Q.positive(x), assumptions) for x in expr.args) \
            or all(ask(Q.negative(x), assumptions) for x in expr.args):
        return True

@NonZeroPredicate.register(Mul)
def _(expr, assumptions):
    for arg in expr.args:
        result = ask(Q.nonzero(arg), assumptions)
        if result:
            continue
        return result
    return True

@NonZeroPredicate.register(Pow)
def _(expr, assumptions):
    return ask(Q.nonzero(expr.base), assumptions)

@NonZeroPredicate.register(Abs)
def _(expr, assumptions):
    return ask(Q.nonzero(expr.args[0]), assumptions)

@NonZeroPredicate.register(NaN)
def _(expr, assumptions):
    return None


# ZeroPredicate

@ZeroPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_zero
    if ret is None:
        raise MDNotImplementedError
    return ret

@ZeroPredicate.register(Basic)
def _(expr, assumptions):
    return fuzzy_and([fuzzy_not(ask(Q.nonzero(expr), assumptions)),
        ask(Q.real(expr), assumptions)])

@ZeroPredicate.register(Mul)
def _(expr, assumptions):
    # TODO: This should be deducible from the nonzero handler
    return fuzzy_or(ask(Q.zero(arg), assumptions) for arg in expr.args)


# NonPositivePredicate

@NonPositivePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_nonpositive
    if ret is None:
        raise MDNotImplementedError
    return ret

@NonPositivePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        notpositive = fuzzy_not(_PositivePredicate_number(expr, assumptions))
        if notpositive:
            return ask(Q.real(expr), assumptions)
        else:
            return notpositive


# PositivePredicate

def _PositivePredicate_number(expr, assumptions):
    r, i = expr.as_real_imag()
    # If the imaginary part can symbolically be shown to be zero then
    # we just evaluate the real part; otherwise we evaluate the imaginary
    # part to see if it actually evaluates to zero and if it does then
    # we make the comparison between the real part and zero.
    if not i:
        r = r.evalf(2)
        if r._prec != 1:
            return r > 0
    else:
        i = i.evalf(2)
        if i._prec != 1:
            if i != 0:
                return False
            r = r.evalf(2)
            if r._prec != 1:
                return r > 0

@PositivePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_positive
    if ret is None:
        raise MDNotImplementedError
    return ret

@PositivePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)

@PositivePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)
    result = True
    for arg in expr.args:
        if ask(Q.positive(arg), assumptions):
            continue
        elif ask(Q.negative(arg), assumptions):
            result = result ^ True
        else:
            return
    return result

@PositivePredicate.register(Add)
def _(expr, assumptions):
    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)

    r = ask(Q.real(expr), assumptions)
    if r is not True:
        return r

    nonneg = 0
    for arg in expr.args:
        if ask(Q.positive(arg), assumptions) is not True:
            if ask(Q.negative(arg), assumptions) is False:
                nonneg += 1
            else:
                break
    else:
        if nonneg < len(expr.args):
            return True

@PositivePredicate.register(Pow)
def _(expr, assumptions):
    if expr.base == E:
        if ask(Q.real(expr.exp), assumptions):
            return True
        if ask(Q.imaginary(expr.exp), assumptions):
            return ask(Q.even(expr.exp/(I*pi)), assumptions)
        return

    if expr.is_number:
        return _PositivePredicate_number(expr, assumptions)
    if ask(Q.positive(expr.base), assumptions):
        if ask(Q.real(expr.exp), assumptions):
            return True
    if ask(Q.negative(expr.base), assumptions):
        if ask(Q.even(expr.exp), assumptions):
            return True
        if ask(Q.odd(expr.exp), assumptions):
            return False

@PositivePredicate.register(exp)
def _(expr, assumptions):
    if ask(Q.real(expr.exp), assumptions):
        return True
    if ask(Q.imaginary(expr.exp), assumptions):
        return ask(Q.even(expr.exp/(I*pi)), assumptions)

@PositivePredicate.register(log)
def _(expr, assumptions):
    r = ask(Q.real(expr.args[0]), assumptions)
    if r is not True:
        return r
    if ask(Q.positive(expr.args[0] - 1), assumptions):
        return True
    if ask(Q.negative(expr.args[0] - 1), assumptions):
        return False

@PositivePredicate.register(factorial)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.integer(x) & Q.positive(x), assumptions):
            return True

@PositivePredicate.register(ImaginaryUnit)
def _(expr, assumptions):
    return False

@PositivePredicate.register(Abs)
def _(expr, assumptions):
    return ask(Q.nonzero(expr), assumptions)

@PositivePredicate.register(Trace)
def _(expr, assumptions):
    if ask(Q.positive_definite(expr.arg), assumptions):
        return True

@PositivePredicate.register(Determinant)
def _(expr, assumptions):
    if ask(Q.positive_definite(expr.arg), assumptions):
        return True

@PositivePredicate.register(MatrixElement)
def _(expr, assumptions):
    if (expr.i == expr.j
            and ask(Q.positive_definite(expr.parent), assumptions)):
        return True

@PositivePredicate.register(atan)
def _(expr, assumptions):
    return ask(Q.positive(expr.args[0]), assumptions)

@PositivePredicate.register(asin)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.positive(x) & Q.nonpositive(x - 1), assumptions):
        return True
    if ask(Q.negative(x) & Q.nonnegative(x + 1), assumptions):
        return False

@PositivePredicate.register(acos)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.nonpositive(x - 1) & Q.nonnegative(x + 1), assumptions):
        return True

@PositivePredicate.register(acot)
def _(expr, assumptions):
    return ask(Q.real(expr.args[0]), assumptions)

@PositivePredicate.register(NaN)
def _(expr, assumptions):
    return None


# ExtendedNegativePredicate

@ExtendedNegativePredicate.register(object)
def _(expr, assumptions):
    return ask(Q.negative(expr) | Q.negative_infinite(expr), assumptions)


# ExtendedPositivePredicate

@ExtendedPositivePredicate.register(object)
def _(expr, assumptions):
    return ask(Q.positive(expr) | Q.positive_infinite(expr), assumptions)


# ExtendedNonZeroPredicate

@ExtendedNonZeroPredicate.register(object)
def _(expr, assumptions):
    return ask(
        Q.negative_infinite(expr) | Q.negative(expr) | Q.positive(expr) | Q.positive_infinite(expr),
        assumptions)


# ExtendedNonPositivePredicate

@ExtendedNonPositivePredicate.register(object)
def _(expr, assumptions):
    return ask(
        Q.negative_infinite(expr) | Q.negative(expr) | Q.zero(expr),
        assumptions)


# ExtendedNonNegativePredicate

@ExtendedNonNegativePredicate.register(object)
def _(expr, assumptions):
    return ask(
        Q.zero(expr) | Q.positive(expr) | Q.positive_infinite(expr),
        assumptions)
<|endoftext|>"""
This module defines base class for handlers and some core handlers:
``Q.commutative`` and ``Q.is_true``.
"""

from sympy.assumptions import Q, ask, AppliedPredicate
from sympy.core import Basic, Symbol
from sympy.core.logic import _fuzzy_group, fuzzy_and, fuzzy_or
from sympy.core.numbers import NaN, Number
from sympy.logic.boolalg import (And, BooleanTrue, BooleanFalse, conjuncts,
    Equivalent, Implies, Not, Or)
from sympy.utilities.exceptions import sympy_deprecation_warning

from ..predicates.common import CommutativePredicate, IsTruePredicate


class AskHandler:
    """Base class that all Ask Handlers must inherit."""
    def __new__(cls, *args, **kwargs):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. The AskHandler class should
            be replaced with the multipledispatch handler of Predicate
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        return super().__new__(cls, *args, **kwargs)


class CommonHandler(AskHandler):
    # Deprecated
    """Defines some useful methods common to most Handlers. """

    @staticmethod
    def AlwaysTrue(expr, assumptions):
        return True

    @staticmethod
    def AlwaysFalse(expr, assumptions):
        return False

    @staticmethod
    def AlwaysNone(expr, assumptions):
        return None

    NaN = AlwaysFalse


# CommutativePredicate

@CommutativePredicate.register(Symbol)
def _(expr, assumptions):
    """Objects are expected to be commutative unless otherwise stated"""
    assumps = conjuncts(assumptions)
    if expr.is_commutative is not None:
        return expr.is_commutative and not ~Q.commutative(expr) in assumps
    if Q.commutative(expr) in assumps:
        return True
    elif ~Q.commutative(expr) in assumps:
        return False
    return True

@CommutativePredicate.register(Basic)
def _(expr, assumptions):
    for arg in expr.args:
        if not ask(Q.commutative(arg), assumptions):
            return False
    return True

@CommutativePredicate.register(Number)
def _(expr, assumptions):
    return True

@CommutativePredicate.register(NaN)
def _(expr, assumptions):
    return True


# IsTruePredicate

@IsTruePredicate.register(bool)
def _(expr, assumptions):
    return expr

@IsTruePredicate.register(BooleanTrue)
def _(expr, assumptions):
    return True

@IsTruePredicate.register(BooleanFalse)
def _(expr, assumptions):
    return False

@IsTruePredicate.register(AppliedPredicate)
def _(expr, assumptions):
    return ask(expr, assumptions)

@IsTruePredicate.register(Not)
def _(expr, assumptions):
    arg = expr.args[0]
    if arg.is_Symbol:
        # symbol used as abstract boolean object
        return None
    value = ask(arg, assumptions=assumptions)
    if value in (True, False):
        return not value
    else:
        return None

@IsTruePredicate.register(Or)
def _(expr, assumptions):
    result = False
    for arg in expr.args:
        p = ask(arg, assumptions=assumptions)
        if p is True:
            return True
        if p is None:
            result = None
    return result

@IsTruePredicate.register(And)
def _(expr, assumptions):
    result = True
    for arg in expr.args:
        p = ask(arg, assumptions=assumptions)
        if p is False:
            return False
        if p is None:
            result = None
    return result

@IsTruePredicate.register(Implies)
def _(expr, assumptions):
    p, q = expr.args
    return ask(~p | q, assumptions=assumptions)

@IsTruePredicate.register(Equivalent)
def _(expr, assumptions):
    p, q = expr.args
    pt = ask(p, assumptions=assumptions)
    if pt is None:
        return None
    qt = ask(q, assumptions=assumptions)
    if qt is None:
        return None
    return pt == qt


#### Helper methods
def test_closed_group(expr, assumptions, key):
    """
    Test for membership in a group with respect
    to the current operation.
    """
    return _fuzzy_group(
        (ask(key(a), assumptions) for a in expr.args), quick_exit=True)

def ask_all(*queries, assumptions):
    return fuzzy_and(
        (ask(query, assumptions) for query in queries))

def ask_any(*queries, assumptions):
    return fuzzy_or(
        (ask(query, assumptions) for query in queries))
<|endoftext|>"""
Handlers for predicates related to set membership: integer, rational, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Mul, Pow, S
from sympy.core.numbers import (AlgebraicNumber, ComplexInfinity, Exp1, Float,
    GoldenRatio, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity,
    Number, NumberSymbol, Pi, pi, Rational, TribonacciConstant, E)
from sympy.core.logic import fuzzy_bool
from sympy.functions import (Abs, acos, acot, asin, atan, cos, cot, exp, im,
    log, re, sin, tan)
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices import Determinant, MatrixBase, Trace
from sympy.matrices.expressions.matexpr import MatrixElement

from sympy.multipledispatch import MDNotImplementedError

from .common import test_closed_group, ask_all, ask_any
from ..predicates.sets import (IntegerPredicate, RationalPredicate,
    IrrationalPredicate, RealPredicate, ExtendedRealPredicate,
    HermitianPredicate, ComplexPredicate, ImaginaryPredicate,
    AntihermitianPredicate, AlgebraicPredicate)


# IntegerPredicate

def _IntegerPredicate_number(expr, assumptions):
    # helper function
        try:
            i = int(expr.round())
            if not (expr - i).equals(0):
                raise TypeError
            return True
        except TypeError:
            return False

@IntegerPredicate.register_many(int, Integer) # type:ignore
def _(expr, assumptions):
    return True

@IntegerPredicate.register_many(Exp1, GoldenRatio, ImaginaryUnit, Infinity,
        NegativeInfinity, Pi, Rational, TribonacciConstant)
def _(expr, assumptions):
    return False

@IntegerPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_integer
    if ret is None:
        raise MDNotImplementedError
    return ret

@IntegerPredicate.register(Add)
def _(expr, assumptions):
    """
    * Integer + Integer       -> Integer
    * Integer + !Integer      -> !Integer
    * !Integer + !Integer -> ?
    """
    if expr.is_number:
        return _IntegerPredicate_number(expr, assumptions)
    return test_closed_group(expr, assumptions, Q.integer)

@IntegerPredicate.register(Pow)
def _(expr,assumptions):
    if expr.is_number:
        return _IntegerPredicate_number(expr, assumptions)
    if ask_all(~Q.zero(expr.base), Q.finite(expr.base), Q.zero(expr.exp), assumptions=assumptions):
        return True
    if ask_all(Q.integer(expr.base), Q.integer(expr.exp), assumptions=assumptions):
        if ask_any(Q.positive(expr.exp), Q.nonnegative(expr.exp) & ~Q.zero(expr.base), Q.zero(expr.base-1), Q.zero(expr.base+1), assumptions=assumptions):
            return True

@IntegerPredicate.register(Mul)
def _(expr, assumptions):
    """
    * Integer*Integer      -> Integer
    * Integer*Irrational   -> !Integer
    * Odd/Even             -> !Integer
    * Integer*Rational     -> ?
    """
    if expr.is_number:
        return _IntegerPredicate_number(expr, assumptions)
    _output = True
    for arg in expr.args:
        if not ask(Q.integer(arg), assumptions):
            if arg.is_Rational:
                if arg.q == 2:
                    return ask(Q.even(2*expr), assumptions)
                if ~(arg.q & 1):
                    return None
            elif ask(Q.irrational(arg), assumptions):
                if _output:
                    _output = False
                else:
                    return
            else:
                return

    return _output

@IntegerPredicate.register(Abs)
def _(expr, assumptions):
    if ask(Q.integer(expr.args[0]), assumptions):
        return True

@IntegerPredicate.register_many(Determinant, MatrixElement, Trace)
def _(expr, assumptions):
    return ask(Q.integer_elements(expr.args[0]), assumptions)


# RationalPredicate

@RationalPredicate.register(Rational)
def _(expr, assumptions):
    return True

@RationalPredicate.register(Float)
def _(expr, assumptions):
    return None

@RationalPredicate.register_many(Exp1, GoldenRatio, ImaginaryUnit, Infinity,
    NegativeInfinity, Pi, TribonacciConstant)
def _(expr, assumptions):
    return False

@RationalPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_rational
    if ret is None:
        raise MDNotImplementedError
    return ret

@RationalPredicate.register_many(Add, Mul)
def _(expr, assumptions):
    """
    * Rational + Rational     -> Rational
    * Rational + !Rational    -> !Rational
    * !Rational + !Rational   -> ?
    """
    if expr.is_number:
        if expr.as_real_imag()[1]:
            return False
    return test_closed_group(expr, assumptions, Q.rational)

@RationalPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Rational ** Integer      -> Rational
    * Irrational ** Rational   -> Irrational
    * Rational ** Irrational   -> ?
    """
    if expr.base == E:
        x = expr.exp
        if ask(Q.rational(x), assumptions):
            return ask(Q.zero(x), assumptions)
        return

    is_exp_integer = ask(Q.integer(expr.exp), assumptions)
    if is_exp_integer:
        is_base_rational = ask(Q.rational(expr.base),assumptions)
        if is_base_rational:
            is_base_zero = ask(Q.zero(expr.base),assumptions)
            if is_base_zero is False:
                return True
            if is_base_zero and ask(Q.positive(expr.exp)):
                return True
        if ask(Q.algebraic(expr.base),assumptions) is False:
            return ask(Q.zero(expr.exp), assumptions)
        if ask(Q.irrational(expr.base),assumptions) and ask(Q.eq(expr.exp,-1)):
            return False
        return
    elif ask(Q.rational(expr.exp), assumptions):
        if ask(Q.prime(expr.base), assumptions) and is_exp_integer is False:
            return False
        if ask(Q.zero(expr.base)) and ask(Q.positive(expr.exp)):
            return True
        if ask(Q.eq(expr.base,1)):
            return True

@RationalPredicate.register_many(asin, atan, cos, sin, tan)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@RationalPredicate.register(exp)
def _(expr, assumptions):
    x = expr.exp
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@RationalPredicate.register_many(acot, cot)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return False

@RationalPredicate.register_many(acos, log)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x - 1), assumptions)


# IrrationalPredicate

@IrrationalPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_irrational
    if ret is None:
        raise MDNotImplementedError
    return ret

@IrrationalPredicate.register(Basic)
def _(expr, assumptions):
    _real = ask(Q.real(expr), assumptions)
    if _real:
        _rational = ask(Q.rational(expr), assumptions)
        if _rational is None:
            return None
        return not _rational
    else:
        return _real


# RealPredicate

def _RealPredicate_number(expr, assumptions):
    # let as_real_imag() work first since the expression may
    # be simpler to evaluate
    i = expr.as_real_imag()[1].evalf(2)
    if i._prec != 1:
        return not i
    # allow None to be returned if we couldn't show for sure
    # that i was 0

@RealPredicate.register_many(Abs, Exp1, Float, GoldenRatio, im, Pi, Rational,
    re, TribonacciConstant)
def _(expr, assumptions):
    return True

@RealPredicate.register_many(ImaginaryUnit, Infinity, NegativeInfinity)
def _(expr, assumptions):
    return False

@RealPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_real
    if ret is None:
        raise MDNotImplementedError
    return ret

@RealPredicate.register(Add)
def _(expr, assumptions):
    """
    * Real + Real              -> Real
    * Real + (Complex & !Real) -> !Real
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)
    return test_closed_group(expr, assumptions, Q.real)

@RealPredicate.register(Mul)
def _(expr, assumptions):
    """
    * Real*Real               -> Real
    * Real*Imaginary          -> !Real
    * Imaginary*Imaginary     -> Real
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)
    result = True
    for arg in expr.args:
        if ask(Q.real(arg), assumptions):
            pass
        elif ask(Q.imaginary(arg), assumptions):
            result = result ^ True
        else:
            break
    else:
        return result

@RealPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Real**Integer              -> Real
    * Positive**Real             -> Real
    * Negative**Real             -> ?
    * Real**(Integer/Even)       -> Real if base is nonnegative
    * Real**(Integer/Odd)        -> Real
    * Imaginary**(Integer/Even)  -> Real
    * Imaginary**(Integer/Odd)   -> not Real
    * Imaginary**Real            -> ? since Real could be 0 (giving real)
                                    or 1 (giving imaginary)
    * b**Imaginary               -> Real if log(b) is imaginary and b != 0
                                    and exponent != integer multiple of
                                    I*pi/log(b)
    * Real**Real                 -> ? e.g. sqrt(-1) is imaginary and
                                    sqrt(2) is not
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)

    if expr.base == E:
        return ask(
            Q.integer(expr.exp/I/pi) | Q.real(expr.exp), assumptions
        )

    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return True
        # If the i = (exp's arg)/(I*pi) is an integer or half-integer
        # multiple of I*pi then 2*i will be an integer. In addition,
        # exp(i*I*pi) = (-1)**i so the overall realness of the expr
        # can be determined by replacing exp(i*I*pi) with (-1)**i.
        i = expr.base.exp/I/pi
        if ask(Q.integer(2*i), assumptions):
            return ask(Q.real((S.NegativeOne**i)**expr.exp), assumptions)
        return

    if ask(Q.imaginary(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                return not odd
            return

    if ask(Q.imaginary(expr.exp), assumptions):
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # I**i -> real, log(I) is imag;
            # (2*I)**i -> complex, log(2*I) is not imag
            return imlog

    if ask(Q.real(expr.base), assumptions):
        if ask(Q.real(expr.exp), assumptions):
            if ask(Q.zero(expr.base), assumptions) is not False:
                if ask(Q.positive(expr.exp), assumptions):
                    return True
                return
            if expr.exp.is_Rational and \
                    ask(Q.even(expr.exp.q), assumptions):
                return ask(Q.positive(expr.base), assumptions)
            elif ask(Q.integer(expr.exp), assumptions):
                return True
            elif ask(Q.positive(expr.base), assumptions):
                return True

@RealPredicate.register_many(cos, sin)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
            return True

@RealPredicate.register(exp)
def _(expr, assumptions):
    return ask(
        Q.integer(expr.exp/I/pi) | Q.real(expr.exp), assumptions
    )

@RealPredicate.register(log)
def _(expr, assumptions):
    return ask(Q.positive(expr.args[0]), assumptions)

@RealPredicate.register_many(Determinant, MatrixElement, Trace)
def _(expr, assumptions):
    return ask(Q.real_elements(expr.args[0]), assumptions)


# ExtendedRealPredicate

@ExtendedRealPredicate.register(object)
def _(expr, assumptions):
    return ask(Q.negative_infinite(expr)
               | Q.negative(expr)
               | Q.zero(expr)
               | Q.positive(expr)
               | Q.positive_infinite(expr),
            assumptions)

@ExtendedRealPredicate.register_many(Infinity, NegativeInfinity)
def _(expr, assumptions):
    return True

@ExtendedRealPredicate.register_many(Add, Mul, Pow) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.extended_real)


# HermitianPredicate

@HermitianPredicate.register(object) # type:ignore
def _(expr, assumptions):
    if isinstance(expr, MatrixBase):
        return None
    return ask(Q.real(expr), assumptions)

@HermitianPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian + Hermitian  -> Hermitian
    * Hermitian + !Hermitian -> !Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    return test_closed_group(expr, assumptions, Q.hermitian)

@HermitianPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    As long as there is at most only one noncommutative term:

    * Hermitian*Hermitian         -> Hermitian
    * Hermitian*Antihermitian     -> !Hermitian
    * Antihermitian*Antihermitian -> Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    nccount = 0
    result = True
    for arg in expr.args:
        if ask(Q.antihermitian(arg), assumptions):
            result = result ^ True
        elif not ask(Q.hermitian(arg), assumptions):
            break
        if ask(~Q.commutative(arg), assumptions):
            nccount += 1
            if nccount > 1:
                break
    else:
        return result

@HermitianPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian**Integer -> Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    if expr.base == E:
        if ask(Q.hermitian(expr.exp), assumptions):
            return True
        raise MDNotImplementedError
    if ask(Q.hermitian(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            return True
    raise MDNotImplementedError

@HermitianPredicate.register_many(cos, sin) # type:ignore
def _(expr, assumptions):
    if ask(Q.hermitian(expr.args[0]), assumptions):
        return True
    raise MDNotImplementedError

@HermitianPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    if ask(Q.hermitian(expr.exp), assumptions):
        return True
    raise MDNotImplementedError

@HermitianPredicate.register(MatrixBase) # type:ignore
def _(mat, assumptions):
    rows, cols = mat.shape
    ret_val = True
    for i in range(rows):
        for j in range(i, cols):
            cond = fuzzy_bool(Eq(mat[i, j], conjugate(mat[j, i])))
            if cond is None:
                ret_val = None
            if cond == False:
                return False
    if ret_val is None:
        raise MDNotImplementedError
    return ret_val


# ComplexPredicate

@ComplexPredicate.register_many(Abs, cos, exp, im, ImaginaryUnit, log, Number, # type:ignore
    NumberSymbol, re, sin)
def _(expr, assumptions):
    return True

@ComplexPredicate.register_many(Infinity, NegativeInfinity) # type:ignore
def _(expr, assumptions):
    return False

@ComplexPredicate.register(Expr) # type:ignore
def _(expr, assumptions):
    ret = expr.is_complex
    if ret is None:
        raise MDNotImplementedError
    return ret

@ComplexPredicate.register_many(Add, Mul) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.complex)

@ComplexPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    if expr.base == E:
        return True
    return test_closed_group(expr, assumptions, Q.complex)

@ComplexPredicate.register_many(Determinant, MatrixElement, Trace) # type:ignore
def _(expr, assumptions):
    return ask(Q.complex_elements(expr.args[0]), assumptions)

@ComplexPredicate.register(NaN) # type:ignore
def _(expr, assumptions):
    return None


# ImaginaryPredicate

def _Imaginary_number(expr, assumptions):
    # let as_real_imag() work first since the expression may
    # be simpler to evaluate
    r = expr.as_real_imag()[0].evalf(2)
    if r._prec != 1:
        return not r
    # allow None to be returned if we couldn't show for sure
    # that r was 0

@ImaginaryPredicate.register(ImaginaryUnit) # type:ignore
def _(expr, assumptions):
    return True

@ImaginaryPredicate.register(Expr) # type:ignore
def _(expr, assumptions):
    ret = expr.is_imaginary
    if ret is None:
        raise MDNotImplementedError
    return ret

@ImaginaryPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Imaginary + Imaginary -> Imaginary
    * Imaginary + Complex   -> ?
    * Imaginary + Real      -> !Imaginary
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    reals = 0
    for arg in expr.args:
        if ask(Q.imaginary(arg), assumptions):
            pass
        elif ask(Q.real(arg), assumptions):
            reals += 1
        else:
            break
    else:
        if reals == 0:
            return True
        if reals in (1, len(expr.args)):
            # two reals could sum 0 thus giving an imaginary
            return False

@ImaginaryPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    * Real*Imaginary      -> Imaginary
    * Imaginary*Imaginary -> Real
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)
    result = False
    reals = 0
    for arg in expr.args:
        if ask(Q.imaginary(arg), assumptions):
            result = result ^ True
        elif not ask(Q.real(arg), assumptions):
            break
    else:
        if reals == len(expr.args):
            return False
        return result

@ImaginaryPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Imaginary**Odd        -> Imaginary
    * Imaginary**Even       -> Real
    * b**Imaginary          -> !Imaginary if exponent is an integer
                               multiple of I*pi/log(b)
    * Imaginary**Real       -> ?
    * Positive**Real        -> Real
    * Negative**Integer     -> Real
    * Negative**(Integer/2) -> Imaginary
    * Negative**Real        -> not Imaginary if exponent is not Rational
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    if expr.base == E:
        a = expr.exp/I/pi
        return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return False
            i = expr.base.exp/I/pi
            if ask(Q.integer(2*i), assumptions):
                return ask(Q.imaginary((S.NegativeOne**i)**expr.exp), assumptions)

    if ask(Q.imaginary(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                return odd
            return

    if ask(Q.imaginary(expr.exp), assumptions):
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # I**i -> real; (2*I)**i -> complex ==> not imaginary
            return False

    if ask(Q.real(expr.base) & Q.real(expr.exp), assumptions):
        if ask(Q.positive(expr.base), assumptions):
            return False
        else:
            rat = ask(Q.rational(expr.exp), assumptions)
            if not rat:
                return rat
            if ask(Q.integer(expr.exp), assumptions):
                return False
            else:
                half = ask(Q.integer(2*expr.exp), assumptions)
                if half:
                    return ask(Q.negative(expr.base), assumptions)
                return half

@ImaginaryPredicate.register(log) # type:ignore
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        if ask(Q.positive(expr.args[0]), assumptions):
            return False
        return
    # XXX it should be enough to do
    # return ask(Q.nonpositive(expr.args[0]), assumptions)
    # but ask(Q.nonpositive(exp(x)), Q.imaginary(x)) -> None;
    # it should return True since exp(x) will be either 0 or complex
    if expr.args[0].func == exp or (expr.args[0].is_Pow and expr.args[0].base == E):
        if expr.args[0].exp in [I, -I]:
            return True
    im = ask(Q.imaginary(expr.args[0]), assumptions)
    if im is False:
        return False

@ImaginaryPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    a = expr.exp/I/pi
    return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

@ImaginaryPredicate.register_many(Number, NumberSymbol) # type:ignore
def _(expr, assumptions):
    return not (expr.as_real_imag()[1] == 0)

@ImaginaryPredicate.register(NaN) # type:ignore
def _(expr, assumptions):
    return None


# AntihermitianPredicate

@AntihermitianPredicate.register(object) # type:ignore
def _(expr, assumptions):
    if isinstance(expr, MatrixBase):
        return None
    if ask(Q.zero(expr), assumptions):
        return True
    return ask(Q.imaginary(expr), assumptions)

@AntihermitianPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Antihermitian + Antihermitian  -> Antihermitian
    * Antihermitian + !Antihermitian -> !Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    return test_closed_group(expr, assumptions, Q.antihermitian)

@AntihermitianPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    As long as there is at most only one noncommutative term:

    * Hermitian*Hermitian         -> !Antihermitian
    * Hermitian*Antihermitian     -> Antihermitian
    * Antihermitian*Antihermitian -> !Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    nccount = 0
    result = False
    for arg in expr.args:
        if ask(Q.antihermitian(arg), assumptions):
            result = result ^ True
        elif not ask(Q.hermitian(arg), assumptions):
            break
        if ask(~Q.commutative(arg), assumptions):
            nccount += 1
            if nccount > 1:
                break
    else:
        return result

@AntihermitianPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian**Integer  -> !Antihermitian
    * Antihermitian**Even -> !Antihermitian
    * Antihermitian**Odd  -> Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    if ask(Q.hermitian(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            return False
    elif ask(Q.antihermitian(expr.base), assumptions):
        if ask(Q.even(expr.exp), assumptions):
            return False
        elif ask(Q.odd(expr.exp), assumptions):
            return True
    raise MDNotImplementedError

@AntihermitianPredicate.register(MatrixBase) # type:ignore
def _(mat, assumptions):
    rows, cols = mat.shape
    ret_val = True
    for i in range(rows):
        for j in range(i, cols):
            cond = fuzzy_bool(Eq(mat[i, j], -conjugate(mat[j, i])))
            if cond is None:
                ret_val = None
            if cond == False:
                return False
    if ret_val is None:
        raise MDNotImplementedError
    return ret_val


# AlgebraicPredicate

@AlgebraicPredicate.register_many(AlgebraicNumber, Float, GoldenRatio, # type:ignore
    ImaginaryUnit, TribonacciConstant)
def _(expr, assumptions):
    return True

@AlgebraicPredicate.register_many(ComplexInfinity, Exp1, Infinity, # type:ignore
    NegativeInfinity, Pi)
def _(expr, assumptions):
    return False

@AlgebraicPredicate.register_many(Add, Mul) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.algebraic)

@AlgebraicPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    if expr.base == E:
        if ask(Q.algebraic(expr.exp), assumptions):
            return ask(~Q.nonzero(expr.exp), assumptions)
        return
    if expr.base == pi:
        if ask(Q.integer(expr.exp), assumptions) and ask(Q.positive(expr.exp), assumptions):
            return False
        return
    exp_rational = ask(Q.rational(expr.exp), assumptions)
    base_algebraic = ask(Q.algebraic(expr.base), assumptions)
    exp_algebraic = ask(Q.algebraic(expr.exp),assumptions)
    if base_algebraic and exp_algebraic:
        if exp_rational:
            return True
        # Check based on the Gelfond-Schneider theorem:
        # If the base is algebraic and not equal to 0 or 1, and the exponent
        # is irrational,then the result is transcendental.
        if ask(Q.ne(expr.base,0) & Q.ne(expr.base,1)) and exp_rational is False:
            return False

@AlgebraicPredicate.register(Rational) # type:ignore
def _(expr, assumptions):
    return expr.q != 0

@AlgebraicPredicate.register_many(asin, atan, cos, sin, tan) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@AlgebraicPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    x = expr.exp
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@AlgebraicPredicate.register_many(acot, cot) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return False

@AlgebraicPredicate.register_many(acos, log) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x - 1), assumptions)
<|endoftext|>"""
This module contains query handlers responsible for Matrices queries:
Square, Symmetric, Invertible etc.
"""

from sympy.logic.boolalg import conjuncts
from sympy.assumptions import Q, ask
from sympy.assumptions.handlers import test_closed_group
from sympy.matrices import MatrixBase
from sympy.matrices.expressions import (BlockMatrix, BlockDiagMatrix, Determinant,
    DiagMatrix, DiagonalMatrix, HadamardProduct, Identity, Inverse, MatAdd, MatMul,
    MatPow, MatrixExpr, MatrixSlice, MatrixSymbol, OneMatrix, Trace, Transpose,
    ZeroMatrix)
from sympy.matrices.expressions.blockmatrix import reblock_2x2
from sympy.matrices.expressions.factorizations import Factorization
from sympy.matrices.expressions.fourier import DFT
from sympy.core.logic import fuzzy_and
from sympy.utilities.iterables import sift
from sympy.core import Basic

from ..predicates.matrices import (SquarePredicate, SymmetricPredicate,
    InvertiblePredicate, OrthogonalPredicate, UnitaryPredicate,
    FullRankPredicate, PositiveDefinitePredicate, UpperTriangularPredicate,
    LowerTriangularPredicate, DiagonalPredicate, IntegerElementsPredicate,
    RealElementsPredicate, ComplexElementsPredicate)


def _Factorization(predicate, expr, assumptions):
    if predicate in expr.predicates:
        return True


# SquarePredicate

@SquarePredicate.register(MatrixExpr)
def _(expr, assumptions):
    return expr.shape[0] == expr.shape[1]


# SymmetricPredicate

@SymmetricPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if all(ask(Q.symmetric(arg), assumptions) for arg in mmul.args):
        return True
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if len(mmul.args) >= 2 and mmul.args[0] == mmul.args[-1].T:
        if len(mmul.args) == 2:
            return True
        return ask(Q.symmetric(MatMul(*mmul.args[1:-1])), assumptions)

@SymmetricPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.symmetric(base), assumptions)
    return None

@SymmetricPredicate.register(MatAdd)
def _(expr, assumptions):
    return all(ask(Q.symmetric(arg), assumptions) for arg in expr.args)

@SymmetricPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if Q.symmetric(expr) in conjuncts(assumptions):
        return True

@SymmetricPredicate.register_many(OneMatrix, ZeroMatrix)
def _(expr, assumptions):
    return ask(Q.square(expr), assumptions)

@SymmetricPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.symmetric(expr.arg), assumptions)

@SymmetricPredicate.register(MatrixSlice)
def _(expr, assumptions):
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if not expr.on_diag:
        return None
    else:
        return ask(Q.symmetric(expr.parent), assumptions)

@SymmetricPredicate.register(Identity)
def _(expr, assumptions):
    return True


# InvertiblePredicate

@InvertiblePredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if all(ask(Q.invertible(arg), assumptions) for arg in mmul.args):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@InvertiblePredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    if exp.is_negative == False:
        return ask(Q.invertible(base), assumptions)
    return None

@InvertiblePredicate.register(MatAdd)
def _(expr, assumptions):
    return None

@InvertiblePredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if Q.invertible(expr) in conjuncts(assumptions):
        return True

@InvertiblePredicate.register_many(Identity, Inverse)
def _(expr, assumptions):
    return True

@InvertiblePredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@InvertiblePredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@InvertiblePredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.invertible(expr.arg), assumptions)

@InvertiblePredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.invertible(expr.parent), assumptions)

@InvertiblePredicate.register(MatrixBase)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    return expr.rank() == expr.rows

@InvertiblePredicate.register(MatrixExpr)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    return None

@InvertiblePredicate.register(BlockMatrix)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if expr.blockshape == (1, 1):
        return ask(Q.invertible(expr.blocks[0, 0]), assumptions)
    expr = reblock_2x2(expr)
    if expr.blockshape == (2, 2):
        [[A, B], [C, D]] = expr.blocks.tolist()
        if ask(Q.invertible(A), assumptions) == True:
            invertible = ask(Q.invertible(D - C * A.I * B), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(B), assumptions) == True:
            invertible = ask(Q.invertible(C - D * B.I * A), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(C), assumptions) == True:
            invertible = ask(Q.invertible(B - A * C.I * D), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(D), assumptions) == True:
            invertible = ask(Q.invertible(A - B * D.I * C), assumptions)
            if invertible is not None:
                return invertible
    return None

@InvertiblePredicate.register(BlockDiagMatrix)
def _(expr, assumptions):
    if expr.rowblocksizes != expr.colblocksizes:
        return None
    return fuzzy_and([ask(Q.invertible(a), assumptions) for a in expr.diag])


# OrthogonalPredicate

@OrthogonalPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.orthogonal(arg), assumptions) for arg in mmul.args) and
            factor == 1):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@OrthogonalPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp:
        return ask(Q.orthogonal(base), assumptions)
    return None

@OrthogonalPredicate.register(MatAdd)
def _(expr, assumptions):
    if (len(expr.args) == 1 and
            ask(Q.orthogonal(expr.args[0]), assumptions)):
        return True

@OrthogonalPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if (not expr.is_square or
                    ask(Q.invertible(expr), assumptions) is False):
        return False
    if Q.orthogonal(expr) in conjuncts(assumptions):
        return True

@OrthogonalPredicate.register(Identity)
def _(expr, assumptions):
    return True

@OrthogonalPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@OrthogonalPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.orthogonal(expr.arg), assumptions)

@OrthogonalPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.orthogonal(expr.parent), assumptions)

@OrthogonalPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.orthogonal, expr, assumptions)


# UnitaryPredicate

@UnitaryPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.unitary(arg), assumptions) for arg in mmul.args) and
            abs(factor) == 1):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@UnitaryPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp:
        return ask(Q.unitary(base), assumptions)
    return None

@UnitaryPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if (not expr.is_square or
                    ask(Q.invertible(expr), assumptions) is False):
        return False
    if Q.unitary(expr) in conjuncts(assumptions):
        return True

@UnitaryPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.unitary(expr.arg), assumptions)

@UnitaryPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.unitary(expr.parent), assumptions)

@UnitaryPredicate.register_many(DFT, Identity)
def _(expr, assumptions):
    return True

@UnitaryPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@UnitaryPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.unitary, expr, assumptions)


# FullRankPredicate

@FullRankPredicate.register(MatMul)
def _(expr, assumptions):
    if all(ask(Q.fullrank(arg), assumptions) for arg in expr.args):
        return True

@FullRankPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp and ask(~Q.negative(exp), assumptions):
        return ask(Q.fullrank(base), assumptions)
    return None

@FullRankPredicate.register(Identity)
def _(expr, assumptions):
    return True

@FullRankPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@FullRankPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@FullRankPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.fullrank(expr.arg), assumptions)

@FullRankPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if ask(Q.orthogonal(expr.parent), assumptions):
        return True


# PositiveDefinitePredicate

@PositiveDefinitePredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.positive_definite(arg), assumptions)
            for arg in mmul.args) and factor > 0):
        return True
    if (len(mmul.args) >= 2
            and mmul.args[0] == mmul.args[-1].T
            and ask(Q.fullrank(mmul.args[0]), assumptions)):
        return ask(Q.positive_definite(
            MatMul(*mmul.args[1:-1])), assumptions)

@PositiveDefinitePredicate.register(MatPow)
def _(expr, assumptions):
    # a power of a positive definite matrix is positive definite
    if ask(Q.positive_definite(expr.args[0]), assumptions):
        return True

@PositiveDefinitePredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.positive_definite(arg), assumptions)
            for arg in expr.args):
        return True

@PositiveDefinitePredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if Q.positive_definite(expr) in conjuncts(assumptions):
        return True

@PositiveDefinitePredicate.register(Identity)
def _(expr, assumptions):
    return True

@PositiveDefinitePredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@PositiveDefinitePredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@PositiveDefinitePredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.positive_definite(expr.arg), assumptions)

@PositiveDefinitePredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.positive_definite(expr.parent), assumptions)


# UpperTriangularPredicate

@UpperTriangularPredicate.register(MatMul)
def _(expr, assumptions):
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.upper_triangular(m), assumptions) for m in matrices):
        return True

@UpperTriangularPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.upper_triangular(arg), assumptions) for arg in expr.args):
        return True

@UpperTriangularPredicate.register(MatPow)
def _(expr, assumptions):
