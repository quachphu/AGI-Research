            var = self._print(var, **kwargs)
            rv = tt.Rop(rv, var, tt.ones_like(var))
        return rv

    def emptyPrinter(self, expr):
        return expr

    def doprint(self, expr, dtypes=None, broadcastables=None):
        """ Convert a SymPy expression to a Theano graph variable.

        The ``dtypes`` and ``broadcastables`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Theano variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        SymPy symbols to the value of the corresponding argument to
        ``theano.tensor.Tensor``.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Theano.

        .. __: http://deeplearning.net/software/theano/tutorial/broadcasting.html

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            SymPy expression to print.

        dtypes : dict
            Mapping from SymPy symbols to Theano datatypes to use when creating
            new Theano variables for those symbols. Corresponds to the ``dtype``
            argument to ``theano.tensor.Tensor``. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastables : dict
            Mapping from SymPy symbols to the value of the ``broadcastable``
            argument to ``theano.tensor.Tensor`` to use when creating Theano
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        theano.gof.graph.Variable
            A variable corresponding to the expression's value in a Theano
            symbolic expression graph.

        """
        if dtypes is None:
            dtypes = {}
        if broadcastables is None:
            broadcastables = {}

        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)


global_cache: dict[Any, Any] = {}


def theano_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Theano graph variable.

    .. deprecated:: 1.8

      ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
      Aesara. Use ``sympy.printing.aesaracode`` instead. See
      :ref:`theanocode-deprecated` for more information.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict
        Cached Theano variables (see :class:`TheanoPrinter.cache
        <TheanoPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    Returns
    =======

    theano.gof.graph.Variable
        A variable corresponding to the expression's value in a Theano symbolic
        expression graph.

    """
    sympy_deprecation_warning(
        """
        sympy.printing.theanocode is deprecated. Theano has been renamed to
        Aesara. Use sympy.printing.aesaracode instead.""",
        deprecated_since_version="1.8",
    active_deprecations_target='theanocode-deprecated')

    if not theano:
        raise ImportError("theano is required for theano_code")

    if cache is None:
        cache = global_cache

    return TheanoPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    r"""
    Get value of ``broadcastables`` argument to :func:`.theano_code` from
    keyword arguments to :func:`.theano_function`.

    Included for backwards compatibility.

    Parameters
    ==========

    inputs
        Sequence of input symbols.

    dim : int
        Common number of dimensions for all inputs. Overrides other arguments
        if given.

    dims : dict
        Mapping from input symbols to number of dimensions. Overrides
        ``broadcastables`` argument if given.

    broadcastables : dict
        Explicit value of ``broadcastables`` argument to
        :meth:`.TheanoPrinter.doprint`. If not None function will return this value unchanged.

    Returns
    =======
    dict
        Dictionary mapping elements of ``inputs`` to their "broadcastable"
        values (tuple of ``bool``\ s).
    """
    if dim is not None:
        return dict.fromkeys(inputs, (False,) * dim)

    if dims is not None:
        maxdim = max(dims.values())
        return {
            s: (False,) * d + (True,) * (maxdim - d)
            for s, d in dims.items()
        }

    if broadcastables is not None:
        return broadcastables

    return {}


@doctest_depends_on(modules=('theano',))
def theano_function(inputs, outputs, scalar=False, *,
        dim=None, dims=None, broadcastables=None, **kwargs):
    """
    Create a Theano function from SymPy expressions.

    .. deprecated:: 1.8

      ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
      Aesara. Use ``sympy.printing.aesaracode`` instead. See
      :ref:`theanocode-deprecated` for more information.

    The inputs and outputs are converted to Theano variables using
    :func:`.theano_code` and then passed to ``theano.function``.

    Parameters
    ==========

    inputs
        Sequence of symbols which constitute the inputs of the function.

    outputs
        Sequence of expressions which constitute the outputs(s) of the
        function. The free symbols of each expression must be a subset of
        ``inputs``.

    scalar : bool
        Convert 0-dimensional arrays in output to scalars. This will return a
        Python wrapper function around the Theano function object.

    cache : dict
        Cached Theano variables (see :class:`TheanoPrinter.cache
        <TheanoPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    dims : dict
        Alternative to ``broadcastables`` argument. Mapping from elements of
        ``inputs`` to integers indicating the dimension of their associated
        arrays/tensors. Overrides ``broadcastables`` argument if given.

    dim : int
        Another alternative to the ``broadcastables`` argument. Common number of
        dimensions to use for all arrays/tensors.
        ``theano_function([x, y], [...], dim=2)`` is equivalent to using
        ``broadcastables={x: (False, False), y: (False, False)}``.

    Returns
    =======
    callable
        A callable object which takes values of ``inputs`` as positional
        arguments and returns an output array for each of the expressions
        in ``outputs``. If ``outputs`` is a single expression the function will
        return a Numpy array, if it is a list of multiple expressions the
        function will return a list of arrays. See description of the ``squeeze``
        argument above for the behavior when a single output is passed in a list.
        The returned object will either be an instance of
        ``theano.compile.function_module.Function`` or a Python wrapper
        function around one. In both cases, the returned value will have a
        ``theano_function`` attribute which points to the return value of
        ``theano.function``.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.printing.theanocode import theano_function

    A simple function with one input and one output:

    >>> f1 = theano_function([x], [x**2 - 1], scalar=True)
    >>> f1(3)
    8.0

    A function with multiple inputs and one output:

    >>> f2 = theano_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)
    >>> f2(3, 4, 2)
    5.0

    A function with multiple inputs and multiple outputs:

    >>> f3 = theano_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
    >>> f3(2, 3)
    [13.0, -5.0]

    See also
    ========

    dim_handling

    """
    sympy_deprecation_warning(
        """
        sympy.printing.theanocode is deprecated. Theano has been renamed to Aesara. Use sympy.printing.aesaracode instead""",
        deprecated_since_version="1.8",
    active_deprecations_target='theanocode-deprecated')

    if not theano:
        raise ImportError("theano is required for theano_function")

    # Pop off non-theano keyword args
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})

    broadcastables = dim_handling(
        inputs, dim=dim, dims=dims, broadcastables=broadcastables,
    )

    # Print inputs/outputs
    code = partial(theano_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs = list(map(code, inputs))
    toutputs = list(map(code, outputs))

    #fix constant expressions as variables
    toutputs = [output if isinstance(output, theano.Variable) else tt.as_tensor_variable(output) for output in toutputs]

    if len(toutputs) == 1:
        toutputs = toutputs[0]

    # Compile theano func
    func = theano.function(tinputs, toutputs, **kwargs)

    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]

    # No wrapper required
    if not scalar or not any(is_0d):
        func.theano_function = func
        return func

    # Create wrapper to convert 0-dimensional outputs to scalars
    def wrapper(*args):
        out = func(*args)
        # out can be array(1.0) or [array(1.0), array(2.0)]

        if is_sequence(out):
            return [o[()] if is_0d[i] else o for i, o in enumerate(out)]
        else:
            return out[()]

    wrapper.__wrapped__ = func
    wrapper.__doc__ = func.__doc__
    wrapper.theano_function = func
    return wrapper
<|endoftext|>from .pycode import (
    PythonCodePrinter,
    MpmathPrinter,
)
from .numpy import NumPyPrinter  # NumPyPrinter is imported for backward compatibility
from sympy.core.sorting import default_sort_key


__all__ = [
    'PythonCodePrinter',
    'MpmathPrinter',  # MpmathPrinter is published for backward compatibility
    'NumPyPrinter',
    'LambdaPrinter',
    'NumPyPrinter',
    'IntervalPrinter',
    'lambdarepr',
]


class LambdaPrinter(PythonCodePrinter):
    """
    This printer converts expressions into strings that can be used by
    lambdify.
    """
    printmethod = "_lambdacode"


    def _print_And(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' and ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Or(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' or ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Not(self, expr):
        result = ['(', 'not (', self._print(expr.args[0]), '))']
        return ''.join(result)

    def _print_BooleanTrue(self, expr):
        return "True"

    def _print_BooleanFalse(self, expr):
        return "False"

    def _print_ITE(self, expr):
        result = [
            '((', self._print(expr.args[1]),
            ') if (', self._print(expr.args[0]),
            ') else (', self._print(expr.args[2]), '))'
        ]
        return ''.join(result)

    def _print_NumberSymbol(self, expr):
        return str(expr)

    def _print_Pow(self, expr, **kwargs):
        # XXX Temporary workaround. Should Python math printer be
        # isolated from PythonCodePrinter?
        return super(PythonCodePrinter, self)._print_Pow(expr, **kwargs)


# numexpr works by altering the string passed to numexpr.evaluate
# rather than by populating a namespace.  Thus a special printer...
class NumExprPrinter(LambdaPrinter):
    # key, value pairs correspond to SymPy name and numexpr name
    # functions not appearing in this dict will raise a TypeError
    printmethod = "_numexprcode"

    _numexpr_functions = {
        'sin' : 'sin',
        'cos' : 'cos',
        'tan' : 'tan',
        'asin': 'arcsin',
        'acos': 'arccos',
        'atan': 'arctan',
        'atan2' : 'arctan2',
        'sinh' : 'sinh',
        'cosh' : 'cosh',
        'tanh' : 'tanh',
        'asinh': 'arcsinh',
        'acosh': 'arccosh',
        'atanh': 'arctanh',
        'ln' : 'log',
        'log': 'log',
        'exp': 'exp',
        'sqrt' : 'sqrt',
        'Abs' : 'abs',
        'conjugate' : 'conj',
        'im' : 'imag',
        're' : 'real',
        'where' : 'where',
        'complex' : 'complex',
        'contains' : 'contains',
    }

    module = 'numexpr'

    def _print_ImaginaryUnit(self, expr):
        return '1j'

    def _print_seq(self, seq, delimiter=', '):
        # simplified _print_seq taken from pretty.py
        s = [self._print(item) for item in seq]
        if s:
            return delimiter.join(s)
        else:
            return ""

    def _print_Function(self, e):
        func_name = e.func.__name__

        nstr = self._numexpr_functions.get(func_name, None)
        if nstr is None:
            # check for implemented_function
            if hasattr(e, '_imp_'):
                return "(%s)" % self._print(e._imp_(*e.args))
            else:
                raise TypeError("numexpr does not support function '%s'" %
                                func_name)
        return "%s(%s)" % (nstr, self._print_seq(e.args))

    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        exprs = [self._print(arg.expr) for arg in expr.args]
        conds = [self._print(arg.cond) for arg in expr.args]
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.
        ans = []
        parenthesis_count = 0
        is_last_cond_True = False
        for cond, expr in zip(conds, exprs):
            if cond == 'True':
                ans.append(expr)
                is_last_cond_True = True
                break
            else:
                ans.append('where(%s, %s, ' % (cond, expr))
                parenthesis_count += 1
        if not is_last_cond_True:
            # See https://github.com/pydata/numexpr/issues/298
            #
            # simplest way to put a nan but raises
            # 'RuntimeWarning: invalid value encountered in log'
            #
            # There are other ways to do this such as
            #
            #   >>> import numexpr as ne
            #   >>> nan = float('nan')
            #   >>> ne.evaluate('where(x < 0, -1, nan)', {'x': [-1, 2, 3], 'nan':nan})
            #   array([-1., nan, nan])
            #
            # That needs to be handled in the lambdified function though rather
            # than here in the printer.
            ans.append('log(-1)')
        return ''.join(ans) + ')' * parenthesis_count

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def blacklisted(self, expr):
        raise TypeError("numexpr cannot be used with %s" %
                        expr.__class__.__name__)

    # blacklist all Matrix printing
    _print_SparseRepMatrix = \
    _print_MutableSparseMatrix = \
    _print_ImmutableSparseMatrix = \
    _print_Matrix = \
    _print_DenseMatrix = \
    _print_MutableDenseMatrix = \
    _print_ImmutableMatrix = \
    _print_ImmutableDenseMatrix = \
    blacklisted
    # blacklist some Python expressions
    _print_list = \
    _print_tuple = \
    _print_Tuple = \
    _print_dict = \
    _print_Dict = \
    blacklisted

    def _print_NumExprEvaluate(self, expr):
        evaluate = self._module_format(self.module +".evaluate")
        return "%s('%s', truediv=True)" % (evaluate, self._print(expr.expr))

    def doprint(self, expr):
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        if not isinstance(expr, CodegenAST):
            expr = NumExprEvaluate(expr)
        return super().doprint(expr)

    def _print_Return(self, expr):
        from sympy.codegen.pynodes import NumExprEvaluate
        r, = expr.args
        if not isinstance(r, NumExprEvaluate):
            expr = expr.func(NumExprEvaluate(r))
        return super()._print_Return(expr)

    def _print_Assignment(self, expr):
        from sympy.codegen.pynodes import NumExprEvaluate
        lhs, rhs, *args = expr.args
        if not isinstance(rhs, NumExprEvaluate):
            expr = expr.func(lhs, NumExprEvaluate(rhs), *args)
        return super()._print_Assignment(expr)

    def _print_CodeBlock(self, expr):
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        args = [ arg if isinstance(arg, CodegenAST) else NumExprEvaluate(arg) for arg in expr.args ]
        return super()._print_CodeBlock(self, expr.func(*args))


class IntervalPrinter(MpmathPrinter, LambdaPrinter):
    """Use ``lambda`` printer but print numbers as ``mpi`` intervals. """

    def _print_Integer(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Integer(expr)

    def _print_Rational(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Half(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Pow(self, expr):
        return super(MpmathPrinter, self)._print_Pow(expr, rational=True)


for k in NumExprPrinter._numexpr_functions:
    setattr(NumExprPrinter, '_print_%s' % k, NumExprPrinter._print_Function)

def lambdarepr(expr, **settings):
    """
    Returns a string usable for lambdifying.
    """
    return LambdaPrinter(settings).doprint(expr)
<|endoftext|>import typing

import sympy
from sympy.core import Add, Mul
from sympy.core import Symbol, Expr, Float, Rational, Integer, Basic
from sympy.core.function import UndefinedFunction, Function
from sympy.core.relational import Relational, Unequality, Equality, LessThan, GreaterThan, StrictLessThan, StrictGreaterThan
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log, Pow
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan, atan2
from sympy.logic.boolalg import And, Or, Xor, Implies, Boolean
from sympy.logic.boolalg import BooleanTrue, BooleanFalse, BooleanFunction, Not, ITE
from sympy.printing.printer import Printer
from sympy.sets import Interval
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.assumptions.assume import AppliedPredicate
from sympy.assumptions.relation.binrel import AppliedBinaryRelation
from sympy.assumptions.ask import Q
from sympy.assumptions.relation.equality import StrictGreaterThanPredicate, StrictLessThanPredicate, GreaterThanPredicate, LessThanPredicate, EqualityPredicate


class SMTLibPrinter(Printer):
    printmethod = "_smtlib"

    # based on dReal, an automated reasoning tool for solving problems that can be encoded as first-order logic formulas over the real numbers.
    # dReal's special strength is in handling problems that involve a wide range of nonlinear real functions.
    _default_settings: dict = {
        'precision': None,
        'known_types': {
            bool: 'Bool',
            int: 'Int',
            float: 'Real'
        },
        'known_constants': {
            # pi: 'MY_VARIABLE_PI_DECLARED_ELSEWHERE',
        },
        'known_functions': {
            Add: '+',
            Mul: '*',

            Equality: '=',
            LessThan: '<=',
            GreaterThan: '>=',
            StrictLessThan: '<',
            StrictGreaterThan: '>',

            EqualityPredicate(): '=',
            LessThanPredicate(): '<=',
            GreaterThanPredicate(): '>=',
            StrictLessThanPredicate(): '<',
            StrictGreaterThanPredicate(): '>',

            exp: 'exp',
            log: 'log',
            Abs: 'abs',
            sin: 'sin',
            cos: 'cos',
            tan: 'tan',
            asin: 'arcsin',
            acos: 'arccos',
            atan: 'arctan',
            atan2: 'arctan2',
            sinh: 'sinh',
            cosh: 'cosh',
            tanh: 'tanh',
            Min: 'min',
            Max: 'max',
            Pow: 'pow',

            And: 'and',
            Or: 'or',
            Xor: 'xor',
            Not: 'not',
            ITE: 'ite',
            Implies: '=>',
        }
    }

    symbol_table: dict

    def __init__(self, settings: typing.Optional[dict] = None,
                 symbol_table=None):
        settings = settings or {}
        self.symbol_table = symbol_table or {}
        Printer.__init__(self, settings)
        self._precision = self._settings['precision']
        self._known_types = dict(self._settings['known_types'])
        self._known_constants = dict(self._settings['known_constants'])
        self._known_functions = dict(self._settings['known_functions'])

        for _ in self._known_types.values(): assert self._is_legal_name(_)
        for _ in self._known_constants.values(): assert self._is_legal_name(_)
        # for _ in self._known_functions.values(): assert self._is_legal_name(_)  # +, *, <, >, etc.

    def _is_legal_name(self, s: str):
        if not s: return False
        if s[0].isnumeric(): return False
        return all(_.isalnum() or _ == '_' for _ in s)

    def _s_expr(self, op: str, args: typing.Union[list, tuple]) -> str:
        args_str = ' '.join(
            a if isinstance(a, str)
            else self._print(a)
            for a in args
        )
        return f'({op} {args_str})'

    def _print_Function(self, e):
        if e in self._known_functions:
            op = self._known_functions[e]
        elif type(e) in self._known_functions:
            op = self._known_functions[type(e)]
        elif type(type(e)) == UndefinedFunction:
            op = e.name
        elif isinstance(e, AppliedBinaryRelation) and e.function in self._known_functions:
            op = self._known_functions[e.function]
            return self._s_expr(op, e.arguments)
        else:
            op = self._known_functions[e]  # throw KeyError

        return self._s_expr(op, e.args)

    def _print_Relational(self, e: Relational):
        return self._print_Function(e)

    def _print_BooleanFunction(self, e: BooleanFunction):
        return self._print_Function(e)

    def _print_Expr(self, e: Expr):
        return self._print_Function(e)

    def _print_Unequality(self, e: Unequality):
        if type(e) in self._known_functions:
            return self._print_Relational(e)  # default
        else:
            eq_op = self._known_functions[Equality]
            not_op = self._known_functions[Not]
            return self._s_expr(not_op, [self._s_expr(eq_op, e.args)])

    def _print_Piecewise(self, e: Piecewise):
        def _print_Piecewise_recursive(args: typing.Union[list, tuple]):
            e, c = args[0]
            if len(args) == 1:
                assert (c is True) or isinstance(c, BooleanTrue)
                return self._print(e)
            else:
                ite = self._known_functions[ITE]
                return self._s_expr(ite, [
                    c, e, _print_Piecewise_recursive(args[1:])
                ])

        return _print_Piecewise_recursive(e.args)

    def _print_Interval(self, e: Interval):
        if e.start.is_infinite and e.end.is_infinite:
            return ''
        elif e.start.is_infinite != e.end.is_infinite:
            raise ValueError(f'One-sided intervals (`{e}`) are not supported in SMT.')
        else:
            return f'[{e.start}, {e.end}]'

    def _print_AppliedPredicate(self, e: AppliedPredicate):
        if e.function == Q.positive:
            rel = Q.gt(e.arguments[0],0)
        elif e.function == Q.negative:
            rel = Q.lt(e.arguments[0], 0)
        elif e.function == Q.zero:
            rel = Q.eq(e.arguments[0], 0)
        elif e.function == Q.nonpositive:
            rel = Q.le(e.arguments[0], 0)
        elif e.function == Q.nonnegative:
            rel = Q.ge(e.arguments[0], 0)
        elif e.function == Q.nonzero:
            rel = Q.ne(e.arguments[0], 0)
        else:
            raise ValueError(f"Predicate (`{e}`) is not handled.")

        return self._print_AppliedBinaryRelation(rel)

    def _print_AppliedBinaryRelation(self, e: AppliedPredicate):
        if e.function == Q.ne:
            return self._print_Unequality(Unequality(*e.arguments))
        else:
            return self._print_Function(e)

    # todo: Sympy does not support quantifiers yet as of 2022, but quantifiers can be handy in SMT.
    # For now, users can extend this class and build in their own quantifier support.
    # See `test_quantifier_extensions()` in test_smtlib.py for an example of how this might look.

    # def _print_ForAll(self, e: ForAll):
    #     return self._s('forall', [
    #         self._s('', [
    #             self._s(sym.name, [self._type_name(sym), Interval(start, end)])
    #             for sym, start, end in e.limits
    #         ]),
    #         e.function
    #     ])

    def _print_BooleanTrue(self, x: BooleanTrue):
        return 'true'

    def _print_BooleanFalse(self, x: BooleanFalse):
        return 'false'

    def _print_Float(self, x: Float):
        dps = prec_to_dps(x._prec)
        str_real = mlib_to_str(x._mpf_, dps, strip_zeros=True, min_fixed=None, max_fixed=None)

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            mul = self._known_functions[Mul]
            pow = self._known_functions[Pow]

            return r"(%s %s (%s 10 %s))" % (mul, mant, pow, exp)
        elif str_real in ["+inf", "-inf"]:
            raise ValueError("Infinite values are not supported in SMT.")
        else:
            return str_real

    def _print_float(self, x: float):
        return self._print(Float(x))

    def _print_Rational(self, x: Rational):
        return self._s_expr('/', [x.p, x.q])

    def _print_Integer(self, x: Integer):
        assert x.q == 1
        return str(x.p)

    def _print_int(self, x: int):
        return str(x)

    def _print_Symbol(self, x: Symbol):
        assert self._is_legal_name(x.name)
        return x.name

    def _print_NumberSymbol(self, x):
        name = self._known_constants.get(x)
        if name:
            return name
        else:
            f = x.evalf(self._precision) if self._precision else x.evalf()
            return self._print_Float(f)

    def _print_UndefinedFunction(self, x):
        assert self._is_legal_name(x.name)
        return x.name

    def _print_Exp1(self, x):
        return (
            self._print_Function(exp(1, evaluate=False))
            if exp in self._known_functions else
            self._print_NumberSymbol(x)
        )

    def emptyPrinter(self, expr):
        raise NotImplementedError(f'Cannot convert `{repr(expr)}` of type `{type(expr)}` to SMT.')


def smtlib_code(
    expr,
    auto_assert=True, auto_declare=True,
    precision=None,
    symbol_table=None,
    known_types=None, known_constants=None, known_functions=None,
    prefix_expressions=None, suffix_expressions=None,
    log_warn=None
):
    r"""Converts ``expr`` to a string of smtlib code.

    Parameters
    ==========

    expr : Expr | List[Expr]
        A SymPy expression or system to be converted.
    auto_assert : bool, optional
        If false, do not modify expr and produce only the S-Expression equivalent of expr.
        If true, assume expr is a system and assert each boolean element.
    auto_declare : bool, optional
        If false, do not produce declarations for the symbols used in expr.
        If true, prepend all necessary declarations for variables used in expr based on symbol_table.
    precision : integer, optional
        The ``evalf(..)`` precision for numbers such as pi.
    symbol_table : dict, optional
        A dictionary where keys are ``Symbol`` or ``Function`` instances and values are their Python type i.e. ``bool``, ``int``, ``float``, or ``Callable[...]``.
        If incomplete, an attempt will be made to infer types from ``expr``.
    known_types: dict, optional
        A dictionary where keys are ``bool``, ``int``, ``float`` etc. and values are their corresponding SMT type names.
        If not given, a partial listing compatible with several solvers will be used.
    known_functions : dict, optional
        A dictionary where keys are ``Function``, ``Relational``, ``BooleanFunction``, or ``Expr`` instances and values are their SMT string representations.
        If not given, a partial listing optimized for dReal solver (but compatible with others) will be used.
    known_constants: dict, optional
        A dictionary where keys are ``NumberSymbol`` instances and values are their SMT variable names.
        When using this feature, extra caution must be taken to avoid naming collisions between user symbols and listed constants.
        If not given, constants will be expanded inline i.e. ``3.14159`` instead of ``MY_SMT_VARIABLE_FOR_PI``.
    prefix_expressions: list, optional
        A list of lists of ``str`` and/or expressions to convert into SMTLib and prefix to the output.
    suffix_expressions: list, optional
        A list of lists of ``str`` and/or expressions to convert into SMTLib and postfix to the output.
    log_warn: lambda function, optional
        A function to record all warnings during potentially risky operations.
        Soundness is a core value in SMT solving, so it is good to log all assumptions made.

    Examples
    ========
    >>> from sympy import smtlib_code, symbols, sin, Eq
    >>> x = symbols('x')
    >>> smtlib_code(sin(x).series(x).removeO(), log_warn=print)
    Could not infer type of `x`. Defaulting to float.
    Non-Boolean expression `x**5/120 - x**3/6 + x` will not be asserted. Converting to SMTLib verbatim.
    '(declare-const x Real)\n(+ x (* (/ -1 6) (pow x 3)) (* (/ 1 120) (pow x 5)))'

    >>> from sympy import Rational
    >>> x, y, tau = symbols("x, y, tau")
    >>> smtlib_code((2*tau)**Rational(7, 2), log_warn=print)
    Could not infer type of `tau`. Defaulting to float.
    Non-Boolean expression `8*sqrt(2)*tau**(7/2)` will not be asserted. Converting to SMTLib verbatim.
    '(declare-const tau Real)\n(* 8 (pow 2 (/ 1 2)) (pow tau (/ 7 2)))'

    ``Piecewise`` expressions are implemented with ``ite`` expressions by default.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> smtlib_code(Eq(pw, 3), symbol_table={x: float}, log_warn=print)
    '(declare-const x Real)\n(assert (= (ite (> x 0) (+ 1 x) x) 3))'

    Custom printing can be defined for certain types by passing a dictionary of
    PythonType : "SMT Name" to the ``known_types``, ``known_constants``, and ``known_functions`` kwargs.

    >>> from typing import Callable
    >>> from sympy import Function, Add
    >>> f = Function('f')
    >>> g = Function('g')
    >>> smt_builtin_funcs = {  # functions our SMT solver will understand
    ...   f: "existing_smtlib_fcn",
    ...   Add: "sum",
    ... }
    >>> user_def_funcs = {  # functions defined by the user must have their types specified explicitly
    ...   g: Callable[[int], float],
    ... }
    >>> smtlib_code(f(x) + g(x), symbol_table=user_def_funcs, known_functions=smt_builtin_funcs, log_warn=print)
    Non-Boolean expression `f(x) + g(x)` will not be asserted. Converting to SMTLib verbatim.
    '(declare-const x Int)\n(declare-fun g (Int) Real)\n(sum (existing_smtlib_fcn x) (g x))'
    """
    log_warn = log_warn or (lambda _: None)

    if not isinstance(expr, list): expr = [expr]
    expr = [
        sympy.sympify(_, strict=True, evaluate=False, convert_xor=False)
        for _ in expr
    ]

    if not symbol_table: symbol_table = {}
    symbol_table = _auto_infer_smtlib_types(
        *expr, symbol_table=symbol_table
    )
    # See [FALLBACK RULES]
    # Need SMTLibPrinter to populate known_functions and known_constants first.

    settings = {}
    if precision: settings['precision'] = precision
    del precision

    if known_types: settings['known_types'] = known_types
    del known_types

    if known_functions: settings['known_functions'] = known_functions
    del known_functions

    if known_constants: settings['known_constants'] = known_constants
    del known_constants

    if not prefix_expressions: prefix_expressions = []
    if not suffix_expressions: suffix_expressions = []

    p = SMTLibPrinter(settings, symbol_table)
    del symbol_table

    # [FALLBACK RULES]
    for e in expr:
        for sym in e.atoms(Symbol, Function):
            if (
                sym.is_Symbol and
                sym not in p._known_constants and
                sym not in p.symbol_table
            ):
                log_warn(f"Could not infer type of `{sym}`. Defaulting to float.")
                p.symbol_table[sym] = float
            if (
                sym.is_Function and
                type(sym) not in p._known_functions and
                type(sym) not in p.symbol_table and
                not sym.is_Piecewise
            ): raise TypeError(
                f"Unknown type of undefined function `{sym}`. "
                f"Must be mapped to ``str`` in known_functions or mapped to ``Callable[..]`` in symbol_table."
            )

    declarations = []
    if auto_declare:
        constants = {sym.name: sym for e in expr for sym in e.free_symbols
                     if sym not in p._known_constants}
        functions = {fnc.name: fnc for e in expr for fnc in e.atoms(Function)
                     if type(fnc) not in p._known_functions and not fnc.is_Piecewise}
        declarations = \
            [
                _auto_declare_smtlib(sym, p, log_warn)
                for sym in constants.values()
            ] + [
                _auto_declare_smtlib(fnc, p, log_warn)
                for fnc in functions.values()
            ]
        declarations = [decl for decl in declarations if decl]

    if auto_assert:
        expr = [_auto_assert_smtlib(e, p, log_warn) for e in expr]

    # return SMTLibPrinter().doprint(expr)
    return '\n'.join([
        # ';; PREFIX EXPRESSIONS',
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in prefix_expressions
        ],

        # ';; DECLARATIONS',
        *sorted(e for e in declarations),

        # ';; EXPRESSIONS',
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in expr
        ],

        # ';; SUFFIX EXPRESSIONS',
        *[
            e if isinstance(e, str) else p.doprint(e)
            for e in suffix_expressions
        ],
    ])


def _auto_declare_smtlib(sym: typing.Union[Symbol, Function], p: SMTLibPrinter, log_warn: typing.Callable[[str], None]):
    if sym.is_Symbol:
        type_signature = p.symbol_table[sym]
        assert isinstance(type_signature, type)
        type_signature = p._known_types[type_signature]
        return p._s_expr('declare-const', [sym, type_signature])

    elif sym.is_Function:
        type_signature = p.symbol_table[type(sym)]
        assert callable(type_signature)
        type_signature = [p._known_types[_] for _ in type_signature.__args__]
        assert len(type_signature) > 0
        params_signature = f"({' '.join(type_signature[:-1])})"
        return_signature = type_signature[-1]
        return p._s_expr('declare-fun', [type(sym), params_signature, return_signature])

    else:
        log_warn(f"Non-Symbol/Function `{sym}` will not be declared.")
        return None


def _auto_assert_smtlib(e: Expr, p: SMTLibPrinter, log_warn: typing.Callable[[str], None]):
    if isinstance(e, Boolean) or (
        e in p.symbol_table and p.symbol_table[e] == bool
    ) or (
        e.is_Function and
        type(e) in p.symbol_table and
        p.symbol_table[type(e)].__args__[-1] == bool
    ):
        return p._s_expr('assert', [e])
    else:
        log_warn(f"Non-Boolean expression `{e}` will not be asserted. Converting to SMTLib verbatim.")
        return e


def _auto_infer_smtlib_types(
    *exprs: Basic,
    symbol_table: typing.Optional[dict] = None
) -> dict:
    # [TYPE INFERENCE RULES]
    # X is alone in an expr => X is bool
    # X in BooleanFunction.args => X is bool
    # X matches to a bool param of a symbol_table function => X is bool
    # X matches to an int param of a symbol_table function => X is int
    # X.is_integer => X is int
    # X == Y, where X is T => Y is T

    # [FALLBACK RULES]
    # see _auto_declare_smtlib(..)
    # X is not bool and X is not int and X is Symbol => X is float
    # else (e.g. X is Function) => error. must be specified explicitly.

    _symbols = dict(symbol_table) if symbol_table else {}

    def safe_update(syms: set, inf):
        for s in syms:
            assert s.is_Symbol
            if (old_type := _symbols.setdefault(s, inf)) != inf:
                raise TypeError(f"Could not infer type of `{s}`. Apparently both `{old_type}` and `{inf}`?")

    # EXPLICIT TYPES
    safe_update({
        e
        for e in exprs
        if e.is_Symbol
    }, bool)

    safe_update({
        symbol
        for e in exprs
        for boolfunc in e.atoms(BooleanFunction)
        for symbol in boolfunc.args
        if symbol.is_Symbol
    }, bool)

    safe_update({
        symbol
        for e in exprs
        for boolfunc in e.atoms(Function)
        if type(boolfunc) in _symbols
        for symbol, param in zip(boolfunc.args, _symbols[type(boolfunc)].__args__)
        if symbol.is_Symbol and param == bool
    }, bool)

    safe_update({
        symbol
        for e in exprs
        for intfunc in e.atoms(Function)
        if type(intfunc) in _symbols
        for symbol, param in zip(intfunc.args, _symbols[type(intfunc)].__args__)
        if symbol.is_Symbol and param == int
    }, int)

    safe_update({
        symbol
        for e in exprs
        for symbol in e.atoms(Symbol)
        if symbol.is_integer
    }, int)

    safe_update({
        symbol
        for e in exprs
        for symbol in e.atoms(Symbol)
        if symbol.is_real and not symbol.is_integer
    }, float)

    # EQUALITY RELATION RULE
    rels_eq = [rel for expr in exprs for rel in expr.atoms(Equality)]
    rels = [
               (rel.lhs, rel.rhs) for rel in rels_eq if rel.lhs.is_Symbol
           ] + [
               (rel.rhs, rel.lhs) for rel in rels_eq if rel.rhs.is_Symbol
           ]
    for infer, reltd in rels:
        inference = (
            _symbols[infer] if infer in _symbols else
            _symbols[reltd] if reltd in _symbols else

            _symbols[type(reltd)].__args__[-1]
            if reltd.is_Function and type(reltd) in _symbols else

            bool if reltd.is_Boolean else
            int if reltd.is_integer or reltd.is_Integer else
            float if reltd.is_real else
            None
        )
        if inference: safe_update({infer}, inference)

    return _symbols
<|endoftext|>import os
from os.path import join
import shutil
import tempfile
from pathlib import Path

try:
    from subprocess import STDOUT, CalledProcessError, check_output
except ImportError:
    pass

from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex

__doctest_requires__ = {('preview',): ['pyglet']}


def _check_output_no_window(*args, **kwargs):
    # Avoid showing a cmd.exe window when running this
    # on Windows
    if os.name == 'nt':
        creation_flag = 0x08000000 # CREATE_NO_WINDOW
    else:
        creation_flag = 0 # Default value
    return check_output(*args, creationflags=creation_flag, **kwargs)


def system_default_viewer(fname, fmt):
    """ Open fname with the default system viewer.

    In practice, it is impossible for python to know when the system viewer is
    done. For this reason, we ensure the passed file will not be deleted under
    it, and this function does not attempt to block.
    """
    # copy to a new temporary file that will not be deleted
    with tempfile.NamedTemporaryFile(prefix='sympy-preview-',
                                     suffix=os.path.splitext(fname)[1],
                                     delete=False) as temp_f:
        with open(fname, 'rb') as f:
            shutil.copyfileobj(f, temp_f)

    import platform
    if platform.system() == 'Darwin':
        import subprocess
        subprocess.call(('open', temp_f.name))
    elif platform.system() == 'Windows':
        os.startfile(temp_f.name)
    else:
        import subprocess
        subprocess.call(('xdg-open', temp_f.name))


def pyglet_viewer(fname, fmt):
    try:
        from pyglet import window, image, gl
        from pyglet.window import key
        from pyglet.image.codecs import ImageDecodeException
    except ImportError:
        raise ImportError("pyglet is required for preview.\n visit https://pyglet.org/")

    try:
        img = image.load(fname)
    except ImageDecodeException:
        raise ValueError("pyglet preview does not work for '{}' files.".format(fmt))

    offset = 25

    config = gl.Config(double_buffer=False)
    win = window.Window(
        width=img.width + 2*offset,
        height=img.height + 2*offset,
        caption="SymPy",
        resizable=False,
        config=config
    )

    win.set_vsync(False)

    try:
        def on_close():
            win.has_exit = True

        win.on_close = on_close

        def on_key_press(symbol, modifiers):
            if symbol in [key.Q, key.ESCAPE]:
                on_close()

        win.on_key_press = on_key_press

        def on_expose():
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            img.blit(
                (win.width - img.width) / 2,
                (win.height - img.height) / 2
            )

        win.on_expose = on_expose

        while not win.has_exit:
            win.dispatch_events()
            win.flip()
    except KeyboardInterrupt:
        pass

    win.close()


def _get_latex_main(expr, *, preamble=None, packages=(), extra_preamble=None,
                    euler=True, fontsize=None, **latex_settings):
    """
    Generate string of a LaTeX document rendering ``expr``.
    """
    if preamble is None:
        actual_packages = packages + ("amsmath", "amsfonts")
        if euler:
            actual_packages += ("euler",)
        package_includes = "\n" + "\n".join(["\\usepackage{%s}" % p
                                             for p in actual_packages])
        if extra_preamble:
            package_includes += extra_preamble

        if not fontsize:
            fontsize = "12pt"
        elif isinstance(fontsize, int):
            fontsize = "{}pt".format(fontsize)
        preamble = r"""\documentclass[varwidth,%s]{standalone}
%s

\begin{document}
""" % (fontsize, package_includes)
    else:
        if packages or extra_preamble:
            raise ValueError("The \"packages\" or \"extra_preamble\" keywords"
                             "must not be set if a "
                             "custom LaTeX preamble was specified")

    if isinstance(expr, str):
        latex_string = expr
    else:
        latex_string = ('$\\displaystyle ' +
                        latex(expr, mode='plain', **latex_settings) +
                        '$')

    return preamble + '\n' + latex_string + '\n\n' + r"\end{document}"


@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',),
            disable_viewers=('evince', 'gimp', 'superior-dvi-viewer'))
def preview(expr, output='png', viewer=None, euler=True, packages=(),
            filename=None, outputbuffer=None, preamble=None, dvioptions=None,
            outputTexFile=None, extra_preamble=None, fontsize=None,
            **latex_settings):
    r"""
    View expression or LaTeX markup in PNG, DVI, PostScript or PDF form.

    If the expr argument is an expression, it will be exported to LaTeX and
    then compiled using the available TeX distribution.  The first argument,
    'expr', may also be a LaTeX string.  The function will then run the
    appropriate viewer for the given output format or use the user defined
    one. By default png output is generated.

    By default pretty Euler fonts are used for typesetting (they were used to
    typeset the well known "Concrete Mathematics" book). For that to work, you
    need the 'eulervm.sty' LaTeX style (in Debian/Ubuntu, install the
    texlive-fonts-extra package). If you prefer default AMS fonts or your
    system lacks 'eulervm' LaTeX package then unset the 'euler' keyword
    argument.

    To use viewer auto-detection, lets say for 'png' output, issue

    >>> from sympy import symbols, preview, Symbol
    >>> x, y = symbols("x,y")

    >>> preview(x + y, output='png')

    This will choose 'pyglet' by default. To select a different one, do

    >>> preview(x + y, output='png', viewer='gimp')

    The 'png' format is considered special. For all other formats the rules
    are slightly different. As an example we will take 'dvi' output format. If
    you would run

    >>> preview(x + y, output='dvi')

    then 'view' will look for available 'dvi' viewers on your system
    (predefined in the function, so it will try evince, first, then kdvi and
    xdvi). If nothing is found, it will fall back to using a system file
    association (via ``open`` and ``xdg-open``). To always use your system file
    association without searching for the above readers, use

    >>> from sympy.printing.preview import system_default_viewer
    >>> preview(x + y, output='dvi', viewer=system_default_viewer)

    If this still does not find the viewer you want, it can be set explicitly.

    >>> preview(x + y, output='dvi', viewer='superior-dvi-viewer')

    This will skip auto-detection and will run user specified
    'superior-dvi-viewer'. If ``view`` fails to find it on your system it will
    gracefully raise an exception.

    You may also enter ``'file'`` for the viewer argument. Doing so will cause
    this function to return a file object in read-only mode, if ``filename``
    is unset. However, if it was set, then 'preview' writes the generated
    file to this filename instead.

    There is also support for writing to a ``io.BytesIO`` like object, which
    needs to be passed to the ``outputbuffer`` argument.

    >>> from io import BytesIO
    >>> obj = BytesIO()
    >>> preview(x + y, output='png', viewer='BytesIO',
    ...         outputbuffer=obj)

    The LaTeX preamble can be customized by setting the 'preamble' keyword
    argument. This can be used, e.g., to set a different font size, use a
    custom documentclass or import certain set of LaTeX packages.

    >>> preamble = "\\documentclass[10pt]{article}\n" \
    ...            "\\usepackage{amsmath,amsfonts}\\begin{document}"
    >>> preview(x + y, output='png', preamble=preamble)

    It is also possible to use the standard preamble and provide additional
    information to the preamble using the ``extra_preamble`` keyword argument.

    >>> from sympy import sin
    >>> extra_preamble = "\\renewcommand{\\sin}{\\cos}"
    >>> preview(sin(x), output='png', extra_preamble=extra_preamble)

    If the value of 'output' is different from 'dvi' then command line
    options can be set ('dvioptions' argument) for the execution of the
    'dvi'+output conversion tool. These options have to be in the form of a
    list of strings (see ``subprocess.Popen``).

    Additional keyword args will be passed to the :func:`~sympy.printing.latex.latex` call,
    e.g., the ``symbol_names`` flag.

    >>> phidd = Symbol('phidd')
    >>> preview(phidd, symbol_names={phidd: r'\ddot{\varphi}'})

    For post-processing the generated TeX File can be written to a file by
    passing the desired filename to the 'outputTexFile' keyword
    argument. To write the TeX code to a file named
    ``"sample.tex"`` and run the default png viewer to display the resulting
    bitmap, do

    >>> preview(x + y, outputTexFile="sample.tex")


    """
    # pyglet is the default for png
    if viewer is None and output == "png":
        try:
            import pyglet  # noqa: F401
        except ImportError:
            pass
        else:
            viewer = pyglet_viewer

    # look up a known application
    if viewer is None:
        # sorted in order from most pretty to most ugly
        # very discussable, but indeed 'gv' looks awful :)
        candidates = {
            "dvi": [ "evince", "okular", "kdvi", "xdvi" ],
            "ps": [ "evince", "okular", "gsview", "gv" ],
            "pdf": [ "evince", "okular", "kpdf", "acroread", "xpdf", "gv" ],
        }

        for candidate in candidates.get(output, []):
            path = shutil.which(candidate)
            if path is not None:
                viewer = path
                break

    # otherwise, use the system default for file association
    if viewer is None:
        viewer = system_default_viewer

    if viewer == "file":
        if filename is None:
            raise ValueError("filename has to be specified if viewer=\"file\"")
    elif viewer == "BytesIO":
        if outputbuffer is None:
            raise ValueError("outputbuffer has to be a BytesIO "
                             "compatible object if viewer=\"BytesIO\"")
    elif not callable(viewer) and not shutil.which(viewer):
        raise OSError("Unrecognized viewer: %s" % viewer)

    latex_main = _get_latex_main(expr, preamble=preamble, packages=packages,
                                 euler=euler, extra_preamble=extra_preamble,
                                 fontsize=fontsize, **latex_settings)

    debug("Latex code:")
    debug(latex_main)
    with tempfile.TemporaryDirectory() as workdir:
        Path(join(workdir, 'texput.tex')).write_text(latex_main, encoding='utf-8')

        if outputTexFile is not None:
            shutil.copyfile(join(workdir, 'texput.tex'), outputTexFile)

        if not shutil.which('latex'):
            raise RuntimeError("latex program is not installed")

        try:
            _check_output_no_window(
                ['latex', '-halt-on-error', '-interaction=nonstopmode',
                 'texput.tex'],
                cwd=workdir,
                stderr=STDOUT)
        except CalledProcessError as e:
            raise RuntimeError(
                "'latex' exited abnormally with the following output:\n%s" %
                e.output)

        src = "texput.%s" % (output)

        if output != "dvi":
            # in order of preference
            commandnames = {
                "ps": ["dvips"],
                "pdf": ["dvipdfmx", "dvipdfm", "dvipdf"],
                "png": ["dvipng"],
                "svg": ["dvisvgm"],
            }
            try:
                cmd_variants = commandnames[output]
            except KeyError:
                raise ValueError("Invalid output format: %s" % output) from None

            # find an appropriate command
            for cmd_variant in cmd_variants:
                cmd_path = shutil.which(cmd_variant)
                if cmd_path:
                    cmd = [cmd_path]
                    break
            else:
                if len(cmd_variants) > 1:
                    raise RuntimeError("None of %s are installed" % ", ".join(cmd_variants))
                else:
                    raise RuntimeError("%s is not installed" % cmd_variants[0])

            defaultoptions = {
                "dvipng": ["-T", "tight", "-z", "9", "--truecolor"],
                "dvisvgm": ["--no-fonts"],
            }

            commandend = {
                "dvips": ["-o", src, "texput.dvi"],
                "dvipdf": ["texput.dvi", src],
                "dvipdfm": ["-o", src, "texput.dvi"],
                "dvipdfmx": ["-o", src, "texput.dvi"],
                "dvipng": ["-o", src, "texput.dvi"],
                "dvisvgm": ["-o", src, "texput.dvi"],
            }

            if dvioptions is not None:
                cmd.extend(dvioptions)
            else:
                cmd.extend(defaultoptions.get(cmd_variant, []))
            cmd.extend(commandend[cmd_variant])

            try:
                _check_output_no_window(cmd, cwd=workdir, stderr=STDOUT)
            except CalledProcessError as e:
                raise RuntimeError(
                    "'%s' exited abnormally with the following output:\n%s" %
                    (' '.join(cmd), e.output))


        if viewer == "file":
            shutil.move(join(workdir, src), filename)
        elif viewer == "BytesIO":
            s = Path(join(workdir, src)).read_bytes()
            outputbuffer.write(s)
        elif callable(viewer):
            viewer(join(workdir, src), fmt=output)
        else:
            try:
                _check_output_no_window(
                    [viewer, src], cwd=workdir, stderr=STDOUT)
            except CalledProcessError as e:
                raise RuntimeError(
                    "'%s %s' exited abnormally with the following output:\n%s" %
                    (viewer, src, e.output))
<|endoftext|>"""
Fortran code printer

The FCodePrinter converts single SymPy expressions into single Fortran
expressions, using the functions defined in the Fortran 77 standard where
possible. Some useful pointers to Fortran can be found on wikipedia:

https://en.wikipedia.org/wiki/Fortran

Most of the code below is based on the "Professional Programmer\'s Guide to
Fortran77" by Clive G. Page:

https://www.star.le.ac.uk/~cgp/prof77.html

Fortran is a case-insensitive language. This might cause trouble because
SymPy is case sensitive. So, fcode adds underscores to variable names when
it is necessary to make them different for Fortran.
"""

from __future__ import annotations
from typing import Any

from collections import defaultdict
from itertools import chain
import string

from sympy.codegen.ast import (
    Assignment, Declaration, Pointer, value_const,
    float32, float64, float80, complex64, complex128, int8, int16, int32,
    int64, intc, real, integer,  bool_, complex_, none, stderr, stdout
)
from sympy.codegen.fnodes import (
    allocatable, isign, dsign, cmplx, merge, literal_dp, elemental, pure,
    intent_in, intent_out, intent_inout
)
from sympy.core import S, Add, N, Float, Symbol
from sympy.core.function import Function
from sympy.core.numbers import equal_valued
from sympy.core.relational import Eq
from sympy.sets import Range
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.printing.printer import printer_context

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import fcode, print_fcode # noqa:F401

known_functions = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "log": "log",
    "exp": "exp",
    "erf": "erf",
    "Abs": "abs",
    "conjugate": "conjg",
    "Max": "max",
    "Min": "min",
}


class FCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Fortran code"""
    printmethod = "_fcode"
    language = "Fortran"

    type_aliases = {
        integer: int32,
        real: float64,
        complex_: complex128,
    }

    type_mappings = {
        intc: 'integer(c_int)',
        float32: 'real*4',  # real(kind(0.e0))
        float64: 'real*8',  # real(kind(0.d0))
        float80: 'real*10', # real(kind(????))
        complex64: 'complex*8',
        complex128: 'complex*16',
        int8: 'integer*1',
        int16: 'integer*2',
        int32: 'integer*4',
        int64: 'integer*8',
        bool_: 'logical'
    }

    type_modules = {
        intc: {'iso_c_binding': 'c_int'}
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'source_format': 'fixed',
        'contract': True,
        'standard': 77,
        'name_mangling': True,
    })

    _operators = {
        'and': '.and.',
        'or': '.or.',
        'xor': '.neqv.',
        'equivalent': '.eqv.',
        'not': '.not. ',
    }

    _relationals = {
        '!=': '/=',
    }

    def __init__(self, settings=None):
        if not settings:
            settings = {}
        self.mangled_symbols = {}         # Dict showing mapping of all words
        self.used_name = []
        self.type_aliases = dict(chain(self.type_aliases.items(),
                                       settings.pop('type_aliases', {}).items()))
        self.type_mappings = dict(chain(self.type_mappings.items(),
                                        settings.pop('type_mappings', {}).items()))
        super().__init__(settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        # leading columns depend on fixed or free format
        standards = {66, 77, 90, 95, 2003, 2008}
        if self._settings['standard'] not in standards:
            raise ValueError("Unknown Fortran standard: %s" % self._settings[
                             'standard'])
        self.module_uses = defaultdict(set)  # e.g.: use iso_c_binding, only: c_int

    @property
    def _lead(self):
        if self._settings['source_format'] == 'fixed':
            return {'code': "      ", 'cont': "     @ ", 'comment': "C     "}
        elif self._settings['source_format'] == 'free':
            return {'code': "", 'cont': "      ", 'comment': "! "}
        else:
            raise ValueError("Unknown source format: %s" % self._settings['source_format'])

    def _print_Symbol(self, expr):
        if self._settings['name_mangling'] == True:
            if expr not in self.mangled_symbols:
                name = expr.name
                while name.lower() in self.used_name:
                    name += '_'
                self.used_name.append(name.lower())
                if name == expr.name:
                    self.mangled_symbols[expr] = expr
                else:
                    self.mangled_symbols[expr] = Symbol(name)

            expr = expr.xreplace(self.mangled_symbols)

        name = super()._print_Symbol(expr)
        return name

    def _rate_index_position(self, p):
        return -p*5

    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {}".format(text)

    def _declare_number_const(self, name, value):
        return "parameter ({} = {})".format(name, self._print(value))

    def _print_NumberSymbol(self, expr):
        # A Number symbol that is not implemented here or with _printmethod
        # is registered and evaluated
        self._number_symbols.add((expr, Float(expr.evalf(self._settings['precision']))))
        return str(expr)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # fortran arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("do %s = %s, %s" % (var, start, stop))
            close_lines.append("end do")
        return open_lines, close_lines

    def _print_sign(self, expr):
        from sympy.functions.elementary.complexes import Abs
        arg, = expr.args
        if arg.is_integer:
            new_expr = merge(0, isign(1, arg), Eq(arg, 0))
        elif (arg.is_complex or arg.is_infinite):
            new_expr = merge(cmplx(literal_dp(0), literal_dp(0)), arg/Abs(arg), Eq(Abs(arg), literal_dp(0)))
        else:
            new_expr = merge(literal_dp(0), dsign(literal_dp(1), arg), Eq(arg, literal_dp(0)))
        return self._print(new_expr)


    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) then" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("else if (%s) then" % self._print(c))
                lines.append(self._print(e))
            lines.append("end if")
            return "\n".join(lines)
        elif self._settings["standard"] >= 95:
            # Only supported in F95 and newer:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            pattern = "merge({T}, {F}, {COND})"
            code = self._print(expr.args[-1].expr)
            terms = list(expr.args[:-1])
            while terms:
                e, c = terms.pop()
                expr = self._print(e)
                cond = self._print(c)
                code = pattern.format(T=expr, F=code, COND=cond)
            return code
        else:
            # `merge` is not supported prior to F95
            raise NotImplementedError("Using Piecewise as an expression using "
                                      "inline operators is not supported in "
                                      "standards earlier than Fortran95.")

    def _print_MatrixElement(self, expr):
        return "{}({}, {})".format(self.parenthesize(expr.parent,
                PRECEDENCE["Atom"], strict=True), expr.i + 1, expr.j + 1)

    def _print_Add(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # collect the purely real and purely imaginary parts:
        pure_real = []
        pure_imaginary = []
        mixed = []
        for arg in expr.args:
            if arg.is_number and arg.is_real:
                pure_real.append(arg)
            elif arg.is_number and arg.is_imaginary:
                pure_imaginary.append(arg)
            else:
                mixed.append(arg)
        if pure_imaginary:
            if mixed:
                PREC = precedence(expr)
                term = Add(*mixed)
                t = self._print(term)
                if t.startswith('-'):
                    sign = "-"
                    t = t[1:]
                else:
                    sign = "+"
                if precedence(term) < PREC:
                    t = "(%s)" % t

                return "cmplx(%s,%s) %s %s" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                    sign, t,
                )
            else:
                return "cmplx(%s,%s)" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                )
        else:
            return CodePrinter._print_Add(self, expr)

    def _print_Function(self, expr):
        # All constant function args are evaluated as floats
        prec =  self._settings['precision']
        args = [N(a, prec) for a in expr.args]
        eval_expr = expr.func(*args)
        if not isinstance(eval_expr, Function):
            return self._print(eval_expr)
        else:
            return CodePrinter._print_Function(self, expr.func(*args))

    def _print_Mod(self, expr):
        # NOTE : Fortran has the functions mod() and modulo(). modulo() behaves
        # the same wrt to the sign of the arguments as Python and SymPy's
        # modulus computations (% and Mod()) but is not available in Fortran 66
        # or Fortran 77, thus we raise an error.
        if self._settings['standard'] in [66, 77]:
            msg = ("Python % operator and SymPy's Mod() function are not "
                   "supported by Fortran 66 or 77 standards.")
            raise NotImplementedError(msg)
        else:
            x, y = expr.args
            return "      modulo({}, {})".format(self._print(x), self._print(y))

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        return "cmplx(0,1)"

    def _print_int(self, expr):
        return str(expr)

    def _print_Mul(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        if expr.is_number and expr.is_imaginary:
            return "cmplx(0,%s)" % (
                self._print(-S.ImaginaryUnit*expr)
            )
        else:
            return CodePrinter._print_Mul(self, expr)

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '%s/%s' % (
                self._print(literal_dp(1)),
                self.parenthesize(expr.base, PREC)
            )
        elif equal_valued(expr.exp, 0.5):
            if expr.base.is_integer:
                # Fortran intrinsic sqrt() does not accept integer argument
                if expr.base.is_Number:
                    return 'sqrt(%s.0d0)' % self._print(expr.base)
                else:
                    return 'sqrt(dble(%s))' % self._print(expr.base)
            else:
                return 'sqrt(%s)' % self._print(expr.base)
        else:
            return CodePrinter._print_Pow(self, expr)

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return "%d.0d0/%d.0d0" % (p, q)

    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])
        return "%sd0" % printed

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        op = op if op not in self._relationals else self._relationals[op]
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement("{0} = {0} {1} {2}".format(
            self._print(lhs_code), self._print(expr.binop), self._print(rhs_code)))

    def _print_sum_(self, sm):
        params = self._print(sm.array)
        if sm.dim != None: # Must use '!= None', cannot use 'is not None'
            params += ', ' + self._print(sm.dim)
        if sm.mask != None: # Must use '!= None', cannot use 'is not None'
            params += ', mask=' + self._print(sm.mask)
        return '%s(%s)' % (sm.__class__.__name__.rstrip('_'), params)

    def _print_product_(self, prod):
        return self._print_sum_(prod)

    def _print_Do(self, do):
        excl = ['concurrent']
        if do.step == 1:
            excl.append('step')
            step = ''
        else:
            step = ', {step}'

        return (
            'do {concurrent}{counter} = {first}, {last}'+step+'\n'
            '{body}\n'
            'end do\n'
        ).format(
            concurrent='concurrent ' if do.concurrent else '',
            **do.kwargs(apply=lambda arg: self._print(arg), exclude=excl)
        )

    def _print_ImpliedDoLoop(self, idl):
        step = '' if idl.step == 1 else ', {step}'
        return ('({expr}, {counter} = {first}, {last}'+step+')').format(
            **idl.kwargs(apply=lambda arg: self._print(arg))
        )

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return ('do {target} = {start}, {stop}, {step}\n'
                '{body}\n'
                'end do').format(target=target, start=start, stop=stop - 1,
                        step=step, body=body)

    def _print_Type(self, type_):
        type_ = self.type_aliases.get(type_, type_)
        type_str = self.type_mappings.get(type_, type_.name)
        module_uses = self.type_modules.get(type_)
        if module_uses:
            for k, v in module_uses:
                self.module_uses[k].add(v)
        return type_str

    def _print_Element(self, elem):
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.symbol),
            idxs=', '.join((self._print(arg) for arg in elem.indices))
        )

    def _print_Extent(self, ext):
        return str(ext)

    def _print_Declaration(self, expr):
        var = expr.variable
        val = var.value
        dim = var.attr_params('dimension')
        intents = [intent in var.attrs for intent in (intent_in, intent_out, intent_inout)]
        if intents.count(True) == 0:
            intent = ''
        elif intents.count(True) == 1:
            intent = ', intent(%s)' % ['in', 'out', 'inout'][intents.index(True)]
        else:
            raise ValueError("Multiple intents specified for %s" % self)

        if isinstance(var, Pointer):
            raise NotImplementedError("Pointers are not available by default in Fortran.")
        if self._settings["standard"] >= 90:
            result = '{t}{vc}{dim}{intent}{alloc} :: {s}'.format(
                t=self._print(var.type),
                vc=', parameter' if value_const in var.attrs else '',
                dim=', dimension(%s)' % ', '.join((self._print(arg) for arg in dim)) if dim else '',
                intent=intent,
                alloc=', allocatable' if allocatable in var.attrs else '',
                s=self._print(var.symbol)
            )
            if val != None: # Must be "!= None", cannot be "is not None"
                result += ' = %s' % self._print(val)
        else:
            if value_const in var.attrs or val:
                raise NotImplementedError("F77 init./parameter statem. req. multiple lines.")
            result = ' '.join((self._print(arg) for arg in [var.type, var.symbol]))

        return result


    def _print_Infinity(self, expr):
        return '(huge(%s) + 1)' % self._print(literal_dp(0))

    def _print_While(self, expr):
        return 'do while ({condition})\n{body}\nend do'.format(**expr.kwargs(
            apply=lambda arg: self._print(arg)))

    def _print_BooleanTrue(self, expr):
        return '.true.'

    def _print_BooleanFalse(self, expr):
        return '.false.'

    def _pad_leading_columns(self, lines):
        result = []
        for line in lines:
            if line.startswith('!'):
                result.append(self._lead['comment'] + line[1:].lstrip())
            else:
                result.append(self._lead['code'] + line)
        return result

    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (without \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # routine to find split point in a code line
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos
        # split line by line and add the split lines to result
        result = []
        if self._settings['source_format'] == 'free':
            trailing = ' &'
        else:
            trailing = ''
        for line in lines:
            if line.startswith(self._lead['comment']):
                # comment line
                if len(line) > 72:
                    pos = line.rfind(" ", 6, 72)
                    if pos == -1:
                        pos = 72
                    hunk = line[:pos]
                    line = line[pos:].lstrip()
                    result.append(hunk)
                    while line:
                        pos = line.rfind(" ", 0, 66)
                        if pos == -1 or len(line) < 66:
                            pos = 66
                        hunk = line[:pos]
                        line = line[pos:].lstrip()
                        result.append("%s%s" % (self._lead['comment'], hunk))
                else:
                    result.append(line)
            elif line.startswith(self._lead['code']):
                # code line
                pos = split_pos_code(line, 72)
                hunk = line[:pos].rstrip()
                line = line[pos:].lstrip()
                if line:
                    hunk += trailing
                result.append(hunk)
                while line:
                    pos = split_pos_code(line, 65)
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                    if line:
                        hunk += trailing
                    result.append("%s%s" % (self._lead['cont'], hunk))
            else:
                result.append(line)
        return result

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        free = self._settings['source_format'] == 'free'
        code = [ line.lstrip(' \t') for line in code ]

        inc_keyword = ('do ', 'if(', 'if ', 'do\n', 'else', 'program', 'interface')
        dec_keyword = ('end do', 'enddo', 'end if', 'endif', 'else', 'end program', 'end interface')

        increase = [ int(any(map(line.startswith, inc_keyword)))
                     for line in code ]
        decrease = [ int(any(map(line.startswith, dec_keyword)))
                     for line in code ]
        continuation = [ int(any(map(line.endswith, ['&', '&\n'])))
                         for line in code ]

        level = 0
        cont_padding = 0
        tabwidth = 3
        new_code = []
        for i, line in enumerate(code):
            if line in ('', '\n'):
                new_code.append(line)
                continue
            level -= decrease[i]

            if free:
                padding = " "*(level*tabwidth + cont_padding)
            else:
                padding = " "*level*tabwidth

            line = "%s%s" % (padding, line)
            if not free:
                line = self._pad_leading_columns([line])[0]

            new_code.append(line)

            if continuation[i]:
                cont_padding = 2*tabwidth
            else:
                cont_padding = 0
            level += increase[i]

        if not free:
            return self._wrap_fortran(new_code)
        return new_code

    def _print_GoTo(self, goto):
        if goto.expr:  # computed goto
            return "go to ({labels}), {expr}".format(
                labels=', '.join((self._print(arg) for arg in goto.labels)),
                expr=self._print(goto.expr)
            )
        else:
            lbl, = goto.labels
            return "go to %s" % self._print(lbl)

    def _print_Program(self, prog):
        return (
            "program {name}\n"
            "{body}\n"
            "end program\n"
        ).format(**prog.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Module(self, mod):
        return (
            "module {name}\n"
            "{declarations}\n"
            "\ncontains\n\n"
            "{definitions}\n"
            "end module\n"
        ).format(**mod.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Stream(self, strm):
        if strm.name == 'stdout' and self._settings["standard"] >= 2003:
            self.module_uses['iso_c_binding'].add('stdint=>input_unit')
            return 'input_unit'
        elif strm.name == 'stderr' and self._settings["standard"] >= 2003:
            self.module_uses['iso_c_binding'].add('stdint=>error_unit')
            return 'error_unit'
        else:
            if strm.name == 'stdout':
                return '*'
            else:
                return strm.name

    def _print_Print(self, ps):
        if ps.format_string == none: # Must be '!= None', cannot be 'is not None'
            template = "print {fmt}, {iolist}"
            fmt = '*'
        else:
            template = 'write(%(out)s, fmt="{fmt}", advance="no"), {iolist}' % {
                'out': {stderr: '0', stdout: '6'}.get(ps.file, '*')
            }
            fmt = self._print(ps.format_string)
        return template.format(fmt=fmt, iolist=', '.join(
            (self._print(arg) for arg in ps.print_args)))

    def _print_Return(self, rs):
        arg, = rs.args
        return "{result_name} = {arg}".format(
            result_name=self._context.get('result_name', 'sympy_result'),
            arg=self._print(arg)
        )

    def _print_FortranReturn(self, frs):
        arg, = frs.args
        if arg:
            return 'return %s' % self._print(arg)
        else:
            return 'return'

    def _head(self, entity, fp, **kwargs):
        bind_C_params = fp.attr_params('bind_C')
        if bind_C_params is None:
            bind = ''
        else:
            bind = ' bind(C, name="%s")' % bind_C_params[0] if bind_C_params else ' bind(C)'
        result_name = self._settings.get('result_name', None)
        return (
            "{entity}{name}({arg_names}){result}{bind}\n"
            "{arg_declarations}"
        ).format(
            entity=entity,
            name=self._print(fp.name),
            arg_names=', '.join([self._print(arg.symbol) for arg in fp.parameters]),
            result=(' result(%s)' % result_name) if result_name else '',
            bind=bind,
            arg_declarations='\n'.join((self._print(Declaration(arg)) for arg in fp.parameters))
        )

    def _print_FunctionPrototype(self, fp):
        entity = "{} function ".format(self._print(fp.return_type))
        return (
            "interface\n"
            "{function_head}\n"
            "end function\n"
            "end interface"
        ).format(function_head=self._head(entity, fp))

    def _print_FunctionDefinition(self, fd):
        if elemental in fd.attrs:
            prefix = 'elemental '
        elif pure in fd.attrs:
            prefix = 'pure '
        else:
            prefix = ''

        entity = "{} function ".format(self._print(fd.return_type))
        with printer_context(self, result_name=fd.name):
            return (
                "{prefix}{function_head}\n"
                "{body}\n"
                "end function\n"
            ).format(
                prefix=prefix,
                function_head=self._head(entity, fd),
                body=self._print(fd.body)
            )

    def _print_Subroutine(self, sub):
        return (
            '{subroutine_head}\n'
            '{body}\n'
            'end subroutine\n'
        ).format(
            subroutine_head=self._head('subroutine ', sub),
            body=self._print(sub.body)
        )

    def _print_SubroutineCall(self, scall):
        return 'call {name}({args})'.format(
            name=self._print(scall.name),
            args=', '.join((self._print(arg) for arg in scall.subroutine_args))
        )

    def _print_use_rename(self, rnm):
        return "%s => %s" % tuple((self._print(arg) for arg in rnm.args))

    def _print_use(self, use):
        result = 'use %s' % self._print(use.namespace)
        if use.rename != None: # Must be '!= None', cannot be 'is not None'
            result += ', ' + ', '.join([self._print(rnm) for rnm in use.rename])
        if use.only != None: # Must be '!= None', cannot be 'is not None'
            result += ', only: ' + ', '.join([self._print(nly) for nly in use.only])
        return result

    def _print_BreakToken(self, _):
        return 'exit'

    def _print_ContinueToken(self, _):
        return 'cycle'

    def _print_ArrayConstructor(self, ac):
        fmtstr = "[%s]" if self._settings["standard"] >= 2003 else '(/%s/)'
        return fmtstr % ', '.join((self._print(arg) for arg in ac.elements))

    def _print_ArrayElement(self, elem):
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.name),
            idxs=', '.join((self._print(arg) for arg in elem.indices))
        )
<|endoftext|>"""
Maple code printer

The MapleCodePrinter converts single SymPy expressions into single
Maple expressions, using the functions defined in the Maple objects where possible.


FIXME: This module is still under actively developed. Some functions may be not completed.
"""

from sympy.core import S
from sympy.core.numbers import Integer, IntegerConstant, equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE

import sympy

_known_func_same_name = (
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'sech',
    'csch', 'coth', 'exp', 'floor', 'factorial', 'bernoulli',  'euler',
    'fibonacci', 'gcd', 'lcm', 'conjugate', 'Ci', 'Chi', 'Ei', 'Li', 'Si', 'Shi',
    'erf', 'erfc', 'harmonic', 'LambertW',
    'sqrt', # For automatic rewrites
)

known_functions = {
    # SymPy -> Maple
    'Abs': 'abs',
    'log': 'ln',
    'asin': 'arcsin',
    'acos': 'arccos',
    'atan': 'arctan',
    'asec': 'arcsec',
    'acsc': 'arccsc',
    'acot': 'arccot',
    'asinh': 'arcsinh',
    'acosh': 'arccosh',
    'atanh': 'arctanh',
    'asech': 'arcsech',
    'acsch': 'arccsch',
    'acoth': 'arccoth',
    'ceiling': 'ceil',
    'Max' : 'max',
    'Min' : 'min',

    'factorial2': 'doublefactorial',
    'RisingFactorial': 'pochhammer',
    'besseli': 'BesselI',
    'besselj': 'BesselJ',
    'besselk': 'BesselK',
    'bessely': 'BesselY',
    'hankelh1': 'HankelH1',
    'hankelh2': 'HankelH2',
    'airyai': 'AiryAi',
    'airybi': 'AiryBi',
    'appellf1': 'AppellF1',
    'fresnelc': 'FresnelC',
    'fresnels': 'FresnelS',
    'lerchphi' : 'LerchPhi',
}

for _func in _known_func_same_name:
    known_functions[_func] = _func

number_symbols = {
    # SymPy -> Maple
    S.Pi: 'Pi',
    S.Exp1: 'exp(1)',
    S.Catalan: 'Catalan',
    S.EulerGamma: 'gamma',
    S.GoldenRatio: '(1/2 + (1/2)*sqrt(5))'
}

spec_relational_ops = {
    # SymPy -> Maple
    '==': '=',
    '!=': '<>'
}

not_supported_symbol = [
    S.ComplexInfinity
]

class MapleCodePrinter(CodePrinter):
    """
    Printer which converts a SymPy expression into a maple code.
    """
    printmethod = "_maple"
    language = "maple"

    _operators = {
        'and': 'and',
        'or': 'or',
        'not': 'not ',
    }

    _default_settings = dict(CodePrinter._default_settings, **{
        'inline': True,
        'allow_unknown_functions': True,
    })

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        super().__init__(settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "# {}".format(text)

    def _declare_number_const(self, name, value):
        return "{} := {};".format(name,
                                    value.evalf(self._settings['precision']))

    def _format_code(self, lines):
        return lines

    def _print_tuple(self, expr):
        return self._print(list(expr))

    def _print_Tuple(self, expr):
        return self._print(list(expr))

    def _print_Assignment(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return "{lhs} := {rhs}".format(lhs=lhs, rhs=rhs)

    def _print_Pow(self, expr, **kwargs):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        elif equal_valued(expr.exp, -0.5):
            return '1/sqrt(%s)' % self._print(expr.base)
        else:
            return '{base}^{exp}'.format(
                base=self.parenthesize(expr.base, PREC),
                exp=self.parenthesize(expr.exp, PREC))

    def _print_Piecewise(self, expr):
        if (expr.args[-1].cond is not True) and (expr.args[-1].cond != S.BooleanTrue):
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        _coup_list = [
            ("{c}, {e}".format(c=self._print(c),
                               e=self._print(e)) if c is not True and c is not S.BooleanTrue else "{e}".format(
                e=self._print(e)))
            for e, c in expr.args]
        _inbrace = ', '.join(_coup_list)
        return 'piecewise({_inbrace})'.format(_inbrace=_inbrace)

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return "{p}/{q}".format(p=str(p), q=str(q))

    def _print_Relational(self, expr):
        PREC=precedence(expr)
        lhs_code = self.parenthesize(expr.lhs, PREC)
        rhs_code = self.parenthesize(expr.rhs, PREC)
        op = expr.rel_op
        if op in spec_relational_ops:
            op = spec_relational_ops[op]
        return "{lhs} {rel_op} {rhs}".format(lhs=lhs_code, rel_op=op, rhs=rhs_code)

    def _print_NumberSymbol(self, expr):
        return number_symbols[expr]

    def _print_NegativeInfinity(self, expr):
        return '-infinity'

    def _print_Infinity(self, expr):
        return 'infinity'

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_bool(self, expr):
        return 'true' if expr else 'false'

    def _print_NaN(self, expr):
        return 'undefined'

    def _get_matrix(self, expr, sparse=False):
        if S.Zero in expr.shape:
            _strM = 'Matrix([], storage = {storage})'.format(
                storage='sparse' if sparse else 'rectangular')
        else:
            _strM = 'Matrix({list}, storage = {storage})'.format(
                list=self._print(expr.tolist()),
                storage='sparse' if sparse else 'rectangular')
        return _strM

    def _print_MatrixElement(self, expr):
        return "{parent}[{i_maple}, {j_maple}]".format(
            parent=self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True),
            i_maple=self._print(expr.i + 1),
            j_maple=self._print(expr.j + 1))

    def _print_MatrixBase(self, expr):
        return self._get_matrix(expr, sparse=False)

    def _print_SparseRepMatrix(self, expr):
        return self._get_matrix(expr, sparse=True)

    def _print_Identity(self, expr):
        if isinstance(expr.rows, (Integer, IntegerConstant)):
            return self._print(sympy.SparseMatrix(expr))
        else:
            return "Matrix({var_size}, shape = identity)".format(var_size=self._print(expr.rows))

    def _print_MatMul(self, expr):
        PREC=precedence(expr)
        _fact_list = list(expr.args)
        _const = None
        if not isinstance(_fact_list[0], (sympy.MatrixBase, sympy.MatrixExpr,
                                          sympy.MatrixSlice, sympy.MatrixSymbol)):
            _const, _fact_list = _fact_list[0], _fact_list[1:]

        if _const is None or _const == 1:
            return '.'.join(self.parenthesize(_m, PREC) for _m in _fact_list)
        else:
            return '{c}*{m}'.format(c=_const, m='.'.join(self.parenthesize(_m, PREC) for _m in _fact_list))

    def _print_MatPow(self, expr):
        # This function requires LinearAlgebra Function in Maple
        return 'MatrixPower({A}, {n})'.format(A=self._print(expr.base), n=self._print(expr.exp))

    def _print_HadamardProduct(self, expr):
        PREC = precedence(expr)
        _fact_list = list(expr.args)
        return '*'.join(self.parenthesize(_m, PREC) for _m in _fact_list)

    def _print_Derivative(self, expr):
        _f, (_var, _order) = expr.args

        if _order != 1:
            _second_arg = '{var}${order}'.format(var=self._print(_var),
                                                 order=self._print(_order))
        else:
            _second_arg = '{var}'.format(var=self._print(_var))
        return 'diff({func_expr}, {sec_arg})'.format(func_expr=self._print(_f), sec_arg=_second_arg)


def maple_code(expr, assign_to=None, **settings):
    r"""Converts ``expr`` to a string of Maple code.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    """
    return MapleCodePrinter(settings).doprint(expr, assign_to)


def print_maple_code(expr, **settings):
    """Prints the Maple representation of the given expression.

    See :func:`maple_code` for the meaning of the optional arguments.

    Examples
    ========

    >>> from sympy import print_maple_code, symbols
    >>> x, y = symbols('x y')
    >>> print_maple_code(x, assign_to=y)
    y := x
    """
    print(maple_code(expr, **settings))
<|endoftext|>from __future__ import annotations

from sympy.core import Basic, S
from sympy.core.function import Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
from functools import reduce

known_functions = {
    'Abs': 'abs',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'acos': 'acos',
    'asin': 'asin',
    'atan': 'atan',
    'atan2': 'atan',
    'ceiling': 'ceil',
    'floor': 'floor',
    'sign': 'sign',
    'exp': 'exp',
    'log': 'log',
    'add': 'add',
    'sub': 'sub',
    'mul': 'mul',
    'pow': 'pow'
}

class GLSLPrinter(CodePrinter):
    """
    Rudimentary, generic GLSL printing tools.

    Additional settings:
    'use_operators': Boolean (should the printer use operators for +,-,*, or functions?)
    """
    _not_supported: set[Basic] = set()
    printmethod = "_glsl"
    language = "GLSL"

    _default_settings = dict(CodePrinter._default_settings, **{
        'use_operators': True,
        'zero': 0,
        'mat_nested': False,
        'mat_separator': ',\n',
        'mat_transpose': False,
        'array_type': 'float',
        'glsl_types': True,

        'precision': 9,
        'user_functions': {},
        'contract': True,
    })

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {}".format(text)

    def _declare_number_const(self, name, value):
        return "float {} = {};".format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [line.lstrip(' \t') for line in code]

        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token))) for line in code]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty

    def _print_MatrixBase(self, mat):
        mat_separator = self._settings['mat_separator']
        mat_transpose = self._settings['mat_transpose']
        column_vector = (mat.rows == 1) if mat_transpose else (mat.cols == 1)
        A = mat.transpose() if mat_transpose != column_vector else mat

        glsl_types = self._settings['glsl_types']
        array_type = self._settings['array_type']
        array_size = A.cols*A.rows
        array_constructor = "{}[{}]".format(array_type, array_size)

        if A.cols == 1:
            return self._print(A[0])
        if A.rows <= 4 and A.cols <= 4 and glsl_types:
            if A.rows == 1:
                return "vec{}{}".format(
                    A.cols, A.table(self,rowstart='(',rowend=')')
                )
            elif A.rows == A.cols:
                return "mat{}({})".format(
                    A.rows, A.table(self,rowsep=', ',
                    rowstart='',rowend='')
                )
            else:
                return "mat{}x{}({})".format(
                    A.cols, A.rows,
                    A.table(self,rowsep=', ',
                    rowstart='',rowend='')
                )
        elif S.One in A.shape:
            return "{}({})".format(
                array_constructor,
                A.table(self,rowsep=mat_separator,rowstart='',rowend='')
            )
        elif not self._settings['mat_nested']:
            return "{}(\n{}\n) /* a {}x{} matrix */".format(
                array_constructor,
                A.table(self,rowsep=mat_separator,rowstart='',rowend=''),
                A.rows, A.cols
            )
        elif self._settings['mat_nested']:
            return "{}[{}][{}](\n{}\n)".format(
                array_type, A.rows, A.cols,
                A.table(self,rowsep=mat_separator,rowstart='float[](',rowend=')')
            )

    def _print_SparseRepMatrix(self, mat):
        # do not allow sparse matrices to be made dense
        return self._print_not_supported(mat)

    def _traverse_matrix_indices(self, mat):
        mat_transpose = self._settings['mat_transpose']
        if mat_transpose:
            rows,cols = mat.shape
        else:
            cols,rows = mat.shape
        return ((i, j) for i in range(cols) for j in range(rows))

    def _print_MatrixElement(self, expr):
        # print('begin _print_MatrixElement')
        nest = self._settings['mat_nested']
        glsl_types = self._settings['glsl_types']
        mat_transpose = self._settings['mat_transpose']
        if mat_transpose:
            cols,rows = expr.parent.shape
            i,j = expr.j,expr.i
        else:
            rows,cols = expr.parent.shape
            i,j = expr.i,expr.j
        pnt = self._print(expr.parent)
        if glsl_types and ((rows <= 4 and cols <=4) or nest):
            return "{}[{}][{}]".format(pnt, i, j)
        else:
            return "{}[{}]".format(pnt, i + j*rows)

    def _print_list(self, expr):
        l = ', '.join(self._print(item) for item in expr)
        glsl_types = self._settings['glsl_types']
        array_type = self._settings['array_type']
        array_size = len(expr)
        array_constructor = '{}[{}]'.format(array_type, array_size)

        if array_size <= 4 and glsl_types:
            return 'vec{}({})'.format(array_size, l)
        else:
            return '{}({})'.format(array_constructor, l)

    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (int %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){"
        for i in indices:
            # GLSL arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'varble': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_Function_with_args(self, func, func_args):
        if func in self.known_functions:
            cond_func = self.known_functions[func]
            func = None
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for cond, func in cond_func:
                    if cond(func_args):
                        break
            if func is not None:
                try:
                    return func(*[self.parenthesize(item, 0) for item in func_args])
                except TypeError:
                    return '{}({})'.format(func, self.stringify(func_args, ", "))
        elif isinstance(func, Lambda):
            # inlined function
            return self._print(func(*func_args))
        else:
            return self._print_not_supported(func)

    def _print_Piecewise(self, expr):
        from sympy.codegen.ast import Assignment
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                else:
                    lines.append("else if (%s) {" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            return "\n".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c),
                                               self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "{}[{}]".format(
            self._print(expr.base.label),
            self._print(elem)
        )

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            try:
                e = self._print(float(expr.exp))
            except TypeError:
                e = self._print(expr.exp)
            return self._print_Function_with_args('pow', (
                self._print(expr.base),
                e
            ))

    def _print_int(self, expr):
        return str(float(expr))

    def _print_Rational(self, expr):
        return "{}.0/{}.0".format(expr.p, expr.q)

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Add(self, expr, order=None):
        if self._settings['use_operators']:
            return CodePrinter._print_Add(self, expr, order=order)

        terms = expr.as_ordered_terms()

        def partition(p,l):
            return reduce(lambda x, y: (x[0]+[y], x[1]) if p(y) else (x[0], x[1]+[y]), l,  ([], []))
        def add(a,b):
            return self._print_Function_with_args('add', (a, b))
            # return self.known_functions['add']+'(%s, %s)' % (a,b)
        neg, pos = partition(lambda arg: arg.could_extract_minus_sign(), terms)
        if pos:
            s = pos = reduce(lambda a,b: add(a,b), (self._print(t) for t in pos))
        else:
            s = pos = self._print(self._settings['zero'])

        if neg:
            # sum the absolute values of the negative terms
            neg = reduce(lambda a,b: add(a,b), (self._print(-n) for n in neg))
            # then subtract them from the positive terms
            s = self._print_Function_with_args('sub', (pos,neg))
            # s = self.known_functions['sub']+'(%s, %s)' % (pos,neg)
        return s

    def _print_Mul(self, expr, **kwargs):
        if self._settings['use_operators']:
            return CodePrinter._print_Mul(self, expr, **kwargs)
        terms = expr.as_ordered_factors()
        def mul(a,b):
            # return self.known_functions['mul']+'(%s, %s)' % (a,b)
            return self._print_Function_with_args('mul', (a,b))

        s = reduce(lambda a,b: mul(a,b), (self._print(t) for t in terms))
        return s

def glsl_code(expr,assign_to=None,**settings):
    """Converts an expr to a string of GLSL code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used for naming the variable or variables
        to which the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol`` or ``Indexed`` type object. In cases where ``expr``
        would be printed as an array, a list of string or ``Symbol`` objects
        can also be passed.

        This is helpful in case of line-wrapping, or for expressions that
        generate multi-line statements.  It can also be used to spread an array-like
        expression into multiple assignments.
    use_operators: bool, optional
        If set to False, then *,/,+,- operators will be replaced with functions
        mul, add, and sub, which must be implemented by the user, e.g. for
        implementing non-standard rings or emulated quad/octal precision.
        [default=True]
    glsl_types: bool, optional
        Set this argument to ``False`` in order to avoid using the ``vec`` and ``mat``
        types.  The printer will instead use arrays (or nested arrays).
        [default=True]
    mat_nested: bool, optional
        GLSL version 4.3 and above support nested arrays (arrays of arrays).  Set this to ``True``
        to render matrices as nested arrays.
        [default=False]
    mat_separator: str, optional
        By default, matrices are rendered with newlines using this separator,
        making them easier to read, but less compact.  By removing the newline
        this option can be used to make them more vertically compact.
        [default=',\n']
    mat_transpose: bool, optional
        GLSL's matrix multiplication implementation assumes column-major indexing.
        By default, this printer ignores that convention. Setting this option to
        ``True`` transposes all matrix output.
        [default=False]
    array_type: str, optional
        The GLSL array constructor type.
        [default='float']
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, js_function_string)]. See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import glsl_code, symbols, Rational, sin, ceiling, Abs
    >>> x, tau = symbols("x, tau")
    >>> glsl_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*pow(tau, 3.5)'
    >>> glsl_code(sin(x), assign_to="float y")
    'float y = sin(x);'

    Various GLSL types are supported:
    >>> from sympy import Matrix, glsl_code
    >>> glsl_code(Matrix([1,2,3]))
    'vec3(1, 2, 3)'

    >>> glsl_code(Matrix([[1, 2],[3, 4]]))
    'mat2(1, 2, 3, 4)'

    Pass ``mat_transpose = True`` to switch to column-major indexing:
    >>> glsl_code(Matrix([[1, 2],[3, 4]]), mat_transpose = True)
    'mat2(1, 3, 2, 4)'

    By default, larger matrices get collapsed into float arrays:
    >>> print(glsl_code( Matrix([[1,2,3,4,5],[6,7,8,9,10]]) ))
    float[10](
       1, 2, 3, 4,  5,
       6, 7, 8, 9, 10
    ) /* a 2x5 matrix */

    The type of array constructor used to print GLSL arrays can be controlled
    via the ``array_type`` parameter:
    >>> glsl_code(Matrix([1,2,3,4,5]), array_type='int')
    'int[5](1, 2, 3, 4, 5)'

    Passing a list of strings or ``symbols`` to the ``assign_to`` parameter will yield
    a multi-line assignment for each item in an array-like expression:
    >>> x_struct_members = symbols('x.a x.b x.c x.d')
    >>> print(glsl_code(Matrix([1,2,3,4]), assign_to=x_struct_members))
    x.a = 1;
    x.b = 2;
    x.c = 3;
    x.d = 4;

    This could be useful in cases where it's desirable to modify members of a
    GLSL ``Struct``.  It could also be used to spread items from an array-like
    expression into various miscellaneous assignments:
    >>> misc_assignments = ('x[0]', 'x[1]', 'float y', 'float z')
    >>> print(glsl_code(Matrix([1,2,3,4]), assign_to=misc_assignments))
    x[0] = 1;
    x[1] = 2;
    float y = 3;
    float z = 4;

    Passing ``mat_nested = True`` instead prints out nested float arrays, which are
    supported in GLSL 4.3 and above.
    >>> mat = Matrix([
    ... [ 0,  1,  2],
    ... [ 3,  4,  5],
    ... [ 6,  7,  8],
    ... [ 9, 10, 11],
    ... [12, 13, 14]])
    >>> print(glsl_code( mat, mat_nested = True ))
    float[5][3](
       float[]( 0,  1,  2),
       float[]( 3,  4,  5),
       float[]( 6,  7,  8),
       float[]( 9, 10, 11),
       float[](12, 13, 14)
    )



    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    js_function_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")]
    ... }
    >>> glsl_code(Abs(x) + ceiling(x), user_functions=custom_functions)
    'fabs(x) + CEIL(x)'

    If further control is needed, addition, subtraction, multiplication and
    division operators can be replaced with ``add``, ``sub``, and ``mul``
    functions.  This is done by passing ``use_operators = False``:

    >>> x,y,z = symbols('x,y,z')
    >>> glsl_code(x*(y+z), use_operators = False)
    'mul(x, add(y, z))'
    >>> glsl_code(x*(y+z*(x-y)**z), use_operators = False)
    'mul(x, add(y, mul(z, pow(sub(x, y), z))))'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(glsl_code(expr, tau))
    if (x > 0) {
       tau = x + 1;
    }
    else {
       tau = x;
    }

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> glsl_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(glsl_code(mat, A))
    A[0][0] = pow(x, 2.0);
    if (x > 0) {
       A[1][0] = x + 1;
    }
    else {
       A[1][0] = x;
    }
    A[2][0] = sin(x);
    """
    return GLSLPrinter(settings).doprint(expr,assign_to)

def print_glsl(expr, **settings):
    """Prints the GLSL representation of the given expression.

       See GLSLPrinter init function for settings.
    """
    print(glsl_code(expr, **settings))
<|endoftext|>"""Printing subsystem driver

SymPy's printing system works the following way: Any expression can be
passed to a designated Printer who then is responsible to return an
adequate representation of that expression.

**The basic concept is the following:**

1.  Let the object print itself if it knows how.
2.  Take the best fitting method defined in the printer.
3.  As fall-back use the emptyPrinter method for the printer.

Which Method is Responsible for Printing?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The whole printing process is started by calling ``.doprint(expr)`` on the printer
which you want to use. This method looks for an appropriate method which can
print the given expression in the given style that the printer defines.
While looking for the method, it follows these steps:

1.  **Let the object print itself if it knows how.**

    The printer looks for a specific method in every object. The name of that method
    depends on the specific printer and is defined under ``Printer.printmethod``.
    For example, StrPrinter calls ``_sympystr`` and LatexPrinter calls ``_latex``.
    Look at the documentation of the printer that you want to use.
    The name of the method is specified there.

    This was the original way of doing printing in sympy. Every class had
    its own latex, mathml, str and repr methods, but it turned out that it
    is hard to produce a high quality printer, if all the methods are spread
    out that far. Therefore all printing code was combined into the different
    printers, which works great for built-in SymPy objects, but not that
    good for user defined classes where it is inconvenient to patch the
    printers.

2.  **Take the best fitting method defined in the printer.**

    The printer loops through expr classes (class + its bases), and tries
    to dispatch the work to ``_print_<EXPR_CLASS>``

    e.g., suppose we have the following class hierarchy::

            Basic
            |
            Atom
            |
            Number
            |
        Rational

    then, for ``expr=Rational(...)``, the Printer will try
    to call printer methods in the order as shown in the figure below::

        p._print(expr)
        |
        |-- p._print_Rational(expr)
        |
        |-- p._print_Number(expr)
        |
        |-- p._print_Atom(expr)
        |
        `-- p._print_Basic(expr)

    if ``._print_Rational`` method exists in the printer, then it is called,
    and the result is returned back. Otherwise, the printer tries to call
    ``._print_Number`` and so on.

3.  **As a fall-back use the emptyPrinter method for the printer.**

    As fall-back ``self.emptyPrinter`` will be called with the expression. If
    not defined in the Printer subclass this will be the same as ``str(expr)``.

.. _printer_example:

Example of Custom Printer
^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, we have a printer which prints the derivative of a function
in a shorter form.

.. code-block:: python

    from sympy.core.symbol import Symbol
    from sympy.printing.latex import LatexPrinter, print_latex
    from sympy.core.function import UndefinedFunction, Function


    class MyLatexPrinter(LatexPrinter):
        \"\"\"Print derivative of a function of symbols in a shorter form.
        \"\"\"
        def _print_Derivative(self, expr):
            function, *vars = expr.args
            if not isinstance(type(function), UndefinedFunction) or \\
               not all(isinstance(i, Symbol) for i in vars):
                return super()._print_Derivative(expr)

            # If you want the printer to work correctly for nested
            # expressions then use self._print() instead of str() or latex().
            # See the example of nested modulo below in the custom printing
            # method section.
            return "{}_{{{}}}".format(
                self._print(Symbol(function.func.__name__)),
                            ''.join(self._print(i) for i in vars))


    def print_my_latex(expr):
        \"\"\" Most of the printers define their own wrappers for print().
        These wrappers usually take printer settings. Our printer does not have
        any settings.
        \"\"\"
        print(MyLatexPrinter().doprint(expr))


    y = Symbol("y")
    x = Symbol("x")
    f = Function("f")
    expr = f(x, y).diff(x, y)

    # Print the expression using the normal latex printer and our custom
    # printer.
    print_latex(expr)
    print_my_latex(expr)

The output of the code above is::

    \\frac{\\partial^{2}}{\\partial x\\partial y}  f{\\left(x,y \\right)}
    f_{xy}

.. _printer_method_example:

Example of Custom Printing Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, the latex printing of the modulo operator is modified.
This is done by overriding the method ``_latex`` of ``Mod``.

>>> from sympy import Symbol, Mod, Integer, print_latex

>>> # Always use printer._print()
>>> class ModOp(Mod):
...     def _latex(self, printer):
...         a, b = [printer._print(i) for i in self.args]
...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

Comparing the output of our custom operator to the builtin one:

>>> x = Symbol('x')
>>> m = Symbol('m')
>>> print_latex(Mod(x, m))
x \\bmod m
>>> print_latex(ModOp(x, m))
\\operatorname{Mod}{\\left(x, m\\right)}

Common mistakes
~~~~~~~~~~~~~~~
It's important to always use ``self._print(obj)`` to print subcomponents of
an expression when customizing a printer. Mistakes include:

1.  Using ``self.doprint(obj)`` instead:

    >>> # This example does not work properly, as only the outermost call may use
    >>> # doprint.
    >>> class ModOpModeWrong(Mod):
    ...     def _latex(self, printer):
    ...         a, b = [printer.doprint(i) for i in self.args]
    ...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

    This fails when the ``mode`` argument is passed to the printer:

    >>> print_latex(ModOp(x, m), mode='inline')  # ok
    $\\operatorname{Mod}{\\left(x, m\\right)}$
    >>> print_latex(ModOpModeWrong(x, m), mode='inline')  # bad
    $\\operatorname{Mod}{\\left($x$, $m$\\right)}$

2.  Using ``str(obj)`` instead:

    >>> class ModOpNestedWrong(Mod):
    ...     def _latex(self, printer):
    ...         a, b = [str(i) for i in self.args]
    ...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

    This fails on nested objects:

    >>> # Nested modulo.
    >>> print_latex(ModOp(ModOp(x, m), Integer(7)))  # ok
    \\operatorname{Mod}{\\left(\\operatorname{Mod}{\\left(x, m\\right)}, 7\\right)}
    >>> print_latex(ModOpNestedWrong(ModOpNestedWrong(x, m), Integer(7)))  # bad
    \\operatorname{Mod}{\\left(ModOpNestedWrong(x, m), 7\\right)}

3.  Using ``LatexPrinter()._print(obj)`` instead.

    >>> from sympy.printing.latex import LatexPrinter
    >>> class ModOpSettingsWrong(Mod):
    ...     def _latex(self, printer):
    ...         a, b = [LatexPrinter()._print(i) for i in self.args]
    ...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

    This causes all the settings to be discarded in the subobjects. As an
    example, the ``full_prec`` setting which shows floats to full precision is
    ignored:

    >>> from sympy import Float
    >>> print_latex(ModOp(Float(1) * x, m), full_prec=True)  # ok
    \\operatorname{Mod}{\\left(1.00000000000000 x, m\\right)}
    >>> print_latex(ModOpSettingsWrong(Float(1) * x, m), full_prec=True)  # bad
    \\operatorname{Mod}{\\left(1.0 x, m\\right)}

"""

from __future__ import annotations
import sys
from typing import Any, Type
import inspect
from contextlib import contextmanager
from functools import cmp_to_key, update_wrapper

from sympy.core.add import Add
from sympy.core.basic import Basic

from sympy.core.function import AppliedUndef, UndefinedFunction, Function



@contextmanager
def printer_context(printer, **kwargs):
    original = printer._context.copy()
    try:
        printer._context.update(kwargs)
        yield
    finally:
        printer._context = original


class Printer:
    """ Generic printer

    Its job is to provide infrastructure for implementing new printers easily.

    If you want to define your custom Printer or your custom printing method
    for your custom class then see the example above: printer_example_ .
    """

    _global_settings: dict[str, Any] = {}

    _default_settings: dict[str, Any] = {}

    # must be initialized to pass tests and cannot be set to '| None' to pass mypy
    printmethod = None  # type: str

    @classmethod
    def _get_initial_settings(cls):
        settings = cls._default_settings.copy()
        for key, val in cls._global_settings.items():
            if key in cls._default_settings:
                settings[key] = val
        return settings

    def __init__(self, settings=None):
        self._str = str

        self._settings = self._get_initial_settings()
        self._context = {}  # mutable during printing

        if settings is not None:
            self._settings.update(settings)

            if len(self._settings) > len(self._default_settings):
                for key in self._settings:
                    if key not in self._default_settings:
                        raise TypeError("Unknown setting '%s'." % key)

        # _print_level is the number of times self._print() was recursively
        # called. See StrPrinter._print_Float() for an example of usage
        self._print_level = 0

    @classmethod
    def set_global_settings(cls, **settings):
        """Set system-wide printing settings. """
        for key, val in settings.items():
            if val is not None:
                cls._global_settings[key] = val

    @property
    def order(self):
        if 'order' in self._settings:
            return self._settings['order']
        else:
            raise AttributeError("No order defined.")

    def doprint(self, expr):
        """Returns printer's representation for expr (as a string)"""
        return self._str(self._print(expr))

    def _print(self, expr, **kwargs) -> str:
        """Internal dispatcher

        Tries the following concepts to print an expression:
            1. Let the object print itself if it knows how.
            2. Take the best fitting method defined in the printer.
            3. As fall-back use the emptyPrinter method for the printer.
        """
        self._print_level += 1
        try:
            # If the printer defines a name for a printing method
            # (Printer.printmethod) and the object knows for itself how it
            # should be printed, use that method.
            if self.printmethod and hasattr(expr, self.printmethod):
                if not (isinstance(expr, type) and issubclass(expr, Basic)):
                    return getattr(expr, self.printmethod)(self, **kwargs)

            # See if the class of expr is known, or if one of its super
            # classes is known, and use that print function
            # Exception: ignore the subclasses of Undefined, so that, e.g.,
            # Function('gamma') does not get dispatched to _print_gamma
            classes = type(expr).__mro__
            if AppliedUndef in classes:
                classes = classes[classes.index(AppliedUndef):]
            if UndefinedFunction in classes:
                classes = classes[classes.index(UndefinedFunction):]
            # Another exception: if someone subclasses a known function, e.g.,
            # gamma, and changes the name, then ignore _print_gamma
            if Function in classes:
                i = classes.index(Function)
                classes = tuple(c for c in classes[:i] if \
                    c.__name__ == classes[0].__name__ or \
                    c.__name__.endswith("Base")) + classes[i:]
            for cls in classes:
                printmethodname = '_print_' + cls.__name__
                printmethod = getattr(self, printmethodname, None)
                if printmethod is not None:
                    return printmethod(expr, **kwargs)
            # Unknown object, fall back to the emptyPrinter.
            return self.emptyPrinter(expr)
        finally:
            self._print_level -= 1

    def emptyPrinter(self, expr):
        return str(expr)

    def _as_ordered_terms(self, expr, order=None):
        """A compatibility function for ordering terms in Add. """
        order = order or self.order

        if order == 'old':
            return sorted(Add.make_args(expr), key=cmp_to_key(self._compare_pretty))
        elif order == 'none':
            return list(expr.args)
        else:
            return expr.as_ordered_terms(order=order)

    def _compare_pretty(self, a, b):
        """return -1, 0, 1 if a is canonically less, equal or
        greater than b. This is used when 'order=old' is selected
        for printing. This puts Order last, orders Rationals
        according to value, puts terms in order wrt the power of
        the last power appearing in a term. Ties are broken using
        Basic.compare.
        """
        from sympy.core.numbers import Rational
        from sympy.core.symbol import Wild
        from sympy.series.order import Order
        if isinstance(a, Order) and not isinstance(b, Order):
            return 1
        if not isinstance(a, Order) and isinstance(b, Order):
            return -1

        if isinstance(a, Rational) and isinstance(b, Rational):
            l = a.p * b.q
            r = b.p * a.q
            return (l > r) - (l < r)
        else:
            p1, p2, p3 = Wild("p1"), Wild("p2"), Wild("p3")
            r_a = a.match(p1 * p2**p3)
            if r_a and p3 in r_a:
                a3 = r_a[p3]
                r_b = b.match(p1 * p2**p3)
                if r_b and p3 in r_b:
                    b3 = r_b[p3]
                    c = Basic.compare(a3, b3)
                    if c != 0:
                        return c

        # break ties
        return Basic.compare(a, b)


class _PrintFunction:
    """
    Function wrapper to replace ``**settings`` in the signature with printer defaults
    """
    def __init__(self, f, print_cls: Type[Printer]):
        # find all the non-setting arguments
        params = list(inspect.signature(f).parameters.values())
        assert params.pop(-1).kind == inspect.Parameter.VAR_KEYWORD
        self.__other_params = params

        self.__print_cls = print_cls
        update_wrapper(self, f)

    def __reduce__(self):
        # Since this is used as a decorator, it replaces the original function.
        # The default pickling will try to pickle self.__wrapped__ and fail
        # because the wrapped function can't be retrieved by name.
        return self.__wrapped__.__qualname__

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        settings = self.__print_cls._get_initial_settings()
        return inspect.Signature(
            parameters=self.__other_params + [
                inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
                for k, v in settings.items()
            ],
            return_annotation=self.__wrapped__.__annotations__.get('return', inspect.Signature.empty)  # type:ignore
        )


def print_function(print_cls):
    """ A decorator to replace kwargs with the printer settings in __signature__ """
    def decorator(f):
        if sys.version_info < (3, 9):
            # We have to create a subclass so that `help` actually shows the docstring in older Python versions.
            # IPython and Sphinx do not need this, only a raw Python console.
            cls = type(f'{f.__qualname__}_PrintFunction', (_PrintFunction,), {"__doc__": f.__doc__})
        else:
            cls = _PrintFunction
        return cls(f, print_cls)
    return decorator
<|endoftext|>from sympy.core import S
from sympy.core.function import Lambda
from sympy.core.power import Pow
from .pycode import PythonCodePrinter, _known_functions_math, _print_known_const, _print_known_func, _unpack_integral_limits, ArrayPrinter
from .codeprinter import CodePrinter


_not_in_numpy = 'erf erfc factorial gamma loggamma'.split()
_in_numpy = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_numpy]
_known_functions_numpy = dict(_in_numpy, **{
    'acos': 'arccos',
    'acosh': 'arccosh',
    'asin': 'arcsin',
    'asinh': 'arcsinh',
    'atan': 'arctan',
    'atan2': 'arctan2',
    'atanh': 'arctanh',
    'exp2': 'exp2',
    'sign': 'sign',
    'logaddexp': 'logaddexp',
    'logaddexp2': 'logaddexp2',
    'isinf': 'isinf',
    'isnan': 'isnan',

})
_known_constants_numpy = {
    'Exp1': 'e',
    'Pi': 'pi',
    'EulerGamma': 'euler_gamma',
    'NaN': 'nan',
    'Infinity': 'inf',
}

_numpy_known_functions = {k: 'numpy.' + v for k, v in _known_functions_numpy.items()}
_numpy_known_constants = {k: 'numpy.' + v for k, v in _known_constants_numpy.items()}

class NumPyPrinter(ArrayPrinter, PythonCodePrinter):
    """
    Numpy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """

    _module = 'numpy'
    _kf = _numpy_known_functions
    _kc = _numpy_known_constants

    def __init__(self, settings=None):
        """
        `settings` is passed to CodePrinter.__init__()
        `module` specifies the array module to use, currently 'NumPy', 'CuPy'
        or 'JAX'.
        """
        self.language = "Python with {}".format(self._module)
        self.printmethod = "_{}code".format(self._module)

        self._kf = {**PythonCodePrinter._kf, **self._kf}

        super().__init__(settings=settings)


    def _print_seq(self, seq):
        "General sequence printer: converts to tuple"
        # Print tuples here instead of lists because numba supports
        #     tuples in nopython mode.
        delimiter=', '
        return '({},)'.format(delimiter.join(self._print(item) for item in seq))

    def _print_NegativeInfinity(self, expr):
        return '-' + self._print(S.Infinity)

    def _print_MatMul(self, expr):
        "Matrix multiplication printer"
        if expr.as_coeff_matrices()[0] is not S.One:
            expr_list = expr.as_coeff_matrices()[1]+[(expr.as_coeff_matrices()[0])]
            return '({})'.format(').dot('.join(self._print(i) for i in expr_list))
        return '({})'.format(').dot('.join(self._print(i) for i in expr.args))

    def _print_MatPow(self, expr):
        "Matrix power printer"
        return '{}({}, {})'.format(self._module_format(self._module + '.linalg.matrix_power'),
            self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Inverse(self, expr):
        "Matrix inverse printer"
        return '{}({})'.format(self._module_format(self._module + '.linalg.inv'),
            self._print(expr.args[0]))

    def _print_DotProduct(self, expr):
        # DotProduct allows any shape order, but numpy.dot does matrix
        # multiplication, so we have to make sure it gets 1 x n by n x 1.
        arg1, arg2 = expr.args
        if arg1.shape[0] != 1:
            arg1 = arg1.T
        if arg2.shape[1] != 1:
            arg2 = arg2.T

        return "%s(%s, %s)" % (self._module_format(self._module + '.dot'),
                               self._print(arg1),
                               self._print(arg2))

    def _print_MatrixSolve(self, expr):
        return "%s(%s, %s)" % (self._module_format(self._module + '.linalg.solve'),
                               self._print(expr.matrix),
                               self._print(expr.vector))

    def _print_ZeroMatrix(self, expr):
        return '{}({})'.format(self._module_format(self._module + '.zeros'),
            self._print(expr.shape))

    def _print_OneMatrix(self, expr):
        return '{}({})'.format(self._module_format(self._module + '.ones'),
            self._print(expr.shape))

    def _print_FunctionMatrix(self, expr):
        from sympy.abc import i, j
        lamda = expr.lamda
        if not isinstance(lamda, Lambda):
            lamda = Lambda((i, j), lamda(i, j))
        return '{}(lambda {}: {}, {})'.format(self._module_format(self._module + '.fromfunction'),
            ', '.join(self._print(arg) for arg in lamda.args[0]),
            self._print(lamda.args[1]), self._print(expr.shape))

    def _print_HadamardProduct(self, expr):
        func = self._module_format(self._module + '.multiply')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_KroneckerProduct(self, expr):
        func = self._module_format(self._module + '.kron')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_Adjoint(self, expr):
        return '{}({}({}))'.format(
            self._module_format(self._module + '.conjugate'),
            self._module_format(self._module + '.transpose'),
            self._print(expr.args[0]))

    def _print_DiagonalOf(self, expr):
        vect = '{}({})'.format(
            self._module_format(self._module + '.diag'),
            self._print(expr.arg))
        return '{}({}, (-1, 1))'.format(
            self._module_format(self._module + '.reshape'), vect)

    def _print_DiagMatrix(self, expr):
        return '{}({})'.format(self._module_format(self._module + '.diagflat'),
            self._print(expr.args[0]))

    def _print_DiagonalMatrix(self, expr):
        return '{}({}, {}({}, {}))'.format(self._module_format(self._module + '.multiply'),
            self._print(expr.arg), self._module_format(self._module + '.eye'),
            self._print(expr.shape[0]), self._print(expr.shape[1]))

    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        from sympy.logic.boolalg import ITE, simplify_logic
        def print_cond(cond):
            """ Problem having an ITE in the cond. """
            if cond.has(ITE):
                return self._print(simplify_logic(cond))
            else:
                return self._print(cond)
        exprs = '[{}]'.format(','.join(self._print(arg.expr) for arg in expr.args))
        conds = '[{}]'.format(','.join(print_cond(arg.cond) for arg in expr.args))
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.
        return '{}({}, {}, default={})'.format(
            self._module_format(self._module + '.select'), conds, exprs,
            self._print(S.NaN))

    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        op = {
            '==' :'equal',
            '!=' :'not_equal',
            '<'  :'less',
            '<=' :'less_equal',
            '>'  :'greater',
            '>=' :'greater_equal',
        }
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '{op}({lhs}, {rhs})'.format(op=self._module_format(self._module + '.'+op[expr.rel_op]),
                                               lhs=lhs, rhs=rhs)
        return super()._print_Relational(expr)

    def _print_And(self, expr):
        "Logical And printer"
        # We have to override LambdaPrinter because it uses Python 'and' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_and' to NUMPY_TRANSLATIONS.
        return '{}.reduce(({}))'.format(self._module_format(self._module + '.logical_and'), ','.join(self._print(i) for i in expr.args))

    def _print_Or(self, expr):
        "Logical Or printer"
        # We have to override LambdaPrinter because it uses Python 'or' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_or' to NUMPY_TRANSLATIONS.
        return '{}.reduce(({}))'.format(self._module_format(self._module + '.logical_or'), ','.join(self._print(i) for i in expr.args))

    def _print_Not(self, expr):
        "Logical Not printer"
        # We have to override LambdaPrinter because it uses Python 'not' keyword.
        # If LambdaPrinter didn't define it, we would still have to define our
        #     own because StrPrinter doesn't define it.
        return '{}({})'.format(self._module_format(self._module + '.logical_not'), ','.join(self._print(i) for i in expr.args))

    def _print_Pow(self, expr, rational=False):
        # XXX Workaround for negative integer power error
        if expr.exp.is_integer and expr.exp.is_negative:
            expr = Pow(expr.base, expr.exp.evalf(), evaluate=False)
        return self._hprint_Pow(expr, rational=rational, sqrt=self._module + '.sqrt')

    def _helper_minimum_maximum(self, op: str, *args):
        if len(args) == 0:
            raise NotImplementedError(f"Need at least one argument for {op}")
        elif len(args) == 1:
            return self._print(args[0])
        _reduce = self._module_format('functools.reduce')
        s_args = [self._print(arg) for arg in args]
        return f"{_reduce}({op}, [{', '.join(s_args)}])"

    def _print_Min(self, expr):
        return self._print_minimum(expr)

    def _print_amin(self, expr):
        return '{}({}, axis={})'.format(self._module_format(self._module + '.amin'), self._print(expr.array), self._print(expr.axis))

    def _print_minimum(self, expr):
        op = self._module_format(self._module + '.minimum')
        return self._helper_minimum_maximum(op, *expr.args)

    def _print_Max(self, expr):
        return self._print_maximum(expr)

    def _print_amax(self, expr):
        return '{}({}, axis={})'.format(self._module_format(self._module + '.amax'), self._print(expr.array), self._print(expr.axis))

    def _print_maximum(self, expr):
        op = self._module_format(self._module + '.maximum')
        return self._helper_minimum_maximum(op, *expr.args)

    def _print_arg(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.angle'), self._print(expr.args[0]))

    def _print_im(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.imag'), self._print(expr.args[0]))

    def _print_Mod(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.mod'), ', '.join(
            (self._print(arg) for arg in expr.args)))

    def _print_re(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.real'), self._print(expr.args[0]))

    def _print_sinc(self, expr):
        return "%s(%s)" % (self._module_format(self._module + '.sinc'), self._print(expr.args[0]/S.Pi))

    def _print_MatrixBase(self, expr):
        if 0 in expr.shape:
            func = self._module_format(f'{self._module}.{self._zeros}')
            return f"{func}({self._print(expr.shape)})"
        func = self.known_functions.get(expr.__class__.__name__, None)
        if func is None:
            func = self._module_format(f'{self._module}.array')
        return "%s(%s)" % (func, self._print(expr.tolist()))

    def _print_Identity(self, expr):
        shape = expr.shape
        if all(dim.is_Integer for dim in shape):
            return "%s(%s)" % (self._module_format(self._module + '.eye'), self._print(expr.shape[0]))
        else:
            raise NotImplementedError("Symbolic matrix dimensions are not yet supported for identity matrices")

    def _print_BlockMatrix(self, expr):
        return '{}({})'.format(self._module_format(self._module + '.block'),
                                 self._print(expr.args[0].tolist()))

    def _print_NDimArray(self, expr):
        if expr.rank() == 0:
            func = self._module_format(f'{self._module}.array')
            return f"{func}({self._print(expr[()])})"
        if 0 in expr.shape:
            func = self._module_format(f'{self._module}.{self._zeros}')
            return f"{func}({self._print(expr.shape)})"
        func = self._module_format(f'{self._module}.array')
        return f"{func}({self._print(expr.tolist())})"

    _add = "add"
    _einsum = "einsum"
    _transpose = "transpose"
    _ones = "ones"
    _zeros = "zeros"

    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported

for func in _numpy_known_functions:
    setattr(NumPyPrinter, f'_print_{func}', _print_known_func)

for const in _numpy_known_constants:
    setattr(NumPyPrinter, f'_print_{const}', _print_known_const)


_known_functions_scipy_special = {
    'Ei': 'expi',
    'erf': 'erf',
    'erfc': 'erfc',
    'besselj': 'jv',
    'bessely': 'yv',
    'besseli': 'iv',
    'besselk': 'kv',
    'cosm1': 'cosm1',
    'powm1': 'powm1',
    'factorial': 'factorial',
    'gamma': 'gamma',
    'loggamma': 'gammaln',
    'digamma': 'psi',
    'polygamma': 'polygamma',
    'RisingFactorial': 'poch',
    'jacobi': 'eval_jacobi',
    'gegenbauer': 'eval_gegenbauer',
    'chebyshevt': 'eval_chebyt',
    'chebyshevu': 'eval_chebyu',
    'legendre': 'eval_legendre',
    'hermite': 'eval_hermite',
    'laguerre': 'eval_laguerre',
    'assoc_laguerre': 'eval_genlaguerre',
    'beta': 'beta',
    'LambertW' : 'lambertw',
}

_known_constants_scipy_constants = {
    'GoldenRatio': 'golden_ratio',
    'Pi': 'pi',
}
_scipy_known_functions = {k : "scipy.special." + v for k, v in _known_functions_scipy_special.items()}
_scipy_known_constants = {k : "scipy.constants." + v for k, v in _known_constants_scipy_constants.items()}

class SciPyPrinter(NumPyPrinter):

    _kf = {**NumPyPrinter._kf, **_scipy_known_functions}
    _kc = {**NumPyPrinter._kc, **_scipy_known_constants}

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.language = "Python with SciPy and NumPy"

    def _print_SparseRepMatrix(self, expr):
        i, j, data = [], [], []
        for (r, c), v in expr.todok().items():
            i.append(r)
            j.append(c)
            data.append(v)

        return "{name}(({data}, ({i}, {j})), shape={shape})".format(
            name=self._module_format('scipy.sparse.coo_matrix'),
            data=data, i=i, j=j, shape=expr.shape
        )

    _print_ImmutableSparseMatrix = _print_SparseRepMatrix

    # SciPy's lpmv has a different order of arguments from assoc_legendre
    def _print_assoc_legendre(self, expr):
        return "{0}({2}, {1}, {3})".format(
            self._module_format('scipy.special.lpmv'),
            self._print(expr.args[0]),
            self._print(expr.args[1]),
            self._print(expr.args[2]))

    def _print_lowergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammainc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    def _print_uppergamma(self, expr):
        return "{0}({2})*{1}({2}, {3})".format(
            self._module_format('scipy.special.gamma'),
            self._module_format('scipy.special.gammaincc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]))

    def _print_betainc(self, expr):
        betainc = self._module_format('scipy.special.betainc')
        beta = self._module_format('scipy.special.beta')
        args = [self._print(arg) for arg in expr.args]
        return f"({betainc}({args[0]}, {args[1]}, {args[3]}) - {betainc}({args[0]}, {args[1]}, {args[2]})) \
            * {beta}({args[0]}, {args[1]})"

    def _print_betainc_regularized(self, expr):
        return "{0}({1}, {2}, {4}) - {0}({1}, {2}, {3})".format(
            self._module_format('scipy.special.betainc'),
            self._print(expr.args[0]),
            self._print(expr.args[1]),
            self._print(expr.args[2]),
            self._print(expr.args[3]))

    def _print_fresnels(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    def _print_fresnelc(self, expr):
        return "{}({})[1]".format(
                self._module_format("scipy.special.fresnel"),
                self._print(expr.args[0]))

    def _print_airyai(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airyaiprime(self, expr):
        return "{}({})[1]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airybi(self, expr):
        return "{}({})[2]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_airybiprime(self, expr):
        return "{}({})[3]".format(
                self._module_format("scipy.special.airy"),
                self._print(expr.args[0]))

    def _print_bernoulli(self, expr):
        # scipy's bernoulli is inconsistent with SymPy's so rewrite
        return self._print(expr._eval_rewrite_as_zeta(*expr.args))

    def _print_harmonic(self, expr):
        return self._print(expr._eval_rewrite_as_zeta(*expr.args))

    def _print_Integral(self, e):
        integration_vars, limits = _unpack_integral_limits(e)

        if len(limits) == 1:
            # nicer (but not necessary) to prefer quad over nquad for 1D case
            module_str = self._module_format("scipy.integrate.quad")
            limit_str = "%s, %s" % tuple(map(self._print, limits[0]))
        else:
            module_str = self._module_format("scipy.integrate.nquad")
            limit_str = "({})".format(", ".join(
                "(%s, %s)" % tuple(map(self._print, l)) for l in limits))

        return "{}(lambda {}: {}, {})[0]".format(
                module_str,
                ", ".join(map(self._print, integration_vars)),
                self._print(e.args[0]),
                limit_str)

    def _print_Si(self, expr):
        return "{}({})[0]".format(
                self._module_format("scipy.special.sici"),
                self._print(expr.args[0]))

    def _print_Ci(self, expr):
        return "{}({})[1]".format(
                self._module_format("scipy.special.sici"),
                self._print(expr.args[0]))

for func in _scipy_known_functions:
    setattr(SciPyPrinter, f'_print_{func}', _print_known_func)

for const in _scipy_known_constants:
    setattr(SciPyPrinter, f'_print_{const}', _print_known_const)


_cupy_known_functions = {k : "cupy." + v for k, v in _known_functions_numpy.items()}
_cupy_known_constants = {k : "cupy." + v for k, v in _known_constants_numpy.items()}

class CuPyPrinter(NumPyPrinter):
    """
    CuPy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """

    _module = 'cupy'
    _kf = _cupy_known_functions
    _kc = _cupy_known_constants

    def __init__(self, settings=None):
        super().__init__(settings=settings)

for func in _cupy_known_functions:
    setattr(CuPyPrinter, f'_print_{func}', _print_known_func)

for const in _cupy_known_constants:
    setattr(CuPyPrinter, f'_print_{const}', _print_known_const)


_jax_known_functions = {k: 'jax.numpy.' + v for k, v in _known_functions_numpy.items()}
_jax_known_constants = {k: 'jax.numpy.' + v for k, v in _known_constants_numpy.items()}

class JaxPrinter(NumPyPrinter):
    """
    JAX printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    _module = "jax.numpy"

    _kf = _jax_known_functions
    _kc = _jax_known_constants

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.printmethod = '_jaxcode'

    # These need specific override to allow for the lack of "jax.numpy.reduce"
    def _print_And(self, expr):
        "Logical And printer"
        return "{}({}.asarray([{}]), axis=0)".format(
            self._module_format(self._module + ".all"),
            self._module_format(self._module),
            ",".join(self._print(i) for i in expr.args),
        )

    def _print_Or(self, expr):
        "Logical Or printer"
        return "{}({}.asarray([{}]), axis=0)".format(
            self._module_format(self._module + ".any"),
            self._module_format(self._module),
            ",".join(self._print(i) for i in expr.args),
        )

for func in _jax_known_functions:
    setattr(JaxPrinter, f'_print_{func}', _print_known_func)

for const in _jax_known_constants:
    setattr(JaxPrinter, f'_print_{const}', _print_known_const)
<|endoftext|>"""
A MathML printer.
"""

from __future__ import annotations
from typing import Any

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import \
    precedence_traditional, PRECEDENCE, PRECEDENCE_TRADITIONAL
from sympy.printing.pretty.pretty_symbology import greek_unicode
from sympy.printing.printer import Printer, print_function

from mpmath.libmp import prec_to_dps, repr_dps, to_str as mlib_to_str


class MathMLPrinterBase(Printer):
    """Contains common code required for MathMLContentPrinter and
    MathMLPresentationPrinter.
    """

    _default_settings: dict[str, Any] = {
        "order": None,
        "encoding": "utf-8",
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_symbol_style": "plain",
        "mul_symbol": None,
        "root_notation": True,
        "symbol_names": {},
        "mul_symbol_mathml_numbers": '&#xB7;',
        "disable_split_super_sub": False,
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)
        from xml.dom.minidom import Document, Text

        self.dom = Document()

        # Workaround to allow strings to remain unescaped
        # Based on
        # https://stackoverflow.com/questions/38015864/python-xml-dom-minidom-\
        #                              please-dont-escape-my-strings/38041194
        class RawText(Text):
            def writexml(self, writer, indent='', addindent='', newl=''):
                if self.data:
                    writer.write('{}{}{}'.format(indent, self.data, newl))

        def createRawTextNode(data):
            r = RawText()
            r.data = data
            r.ownerDocument = self.dom
            return r

        self.dom.createTextNode = createRawTextNode

    def doprint(self, expr):
        """
        Prints the expression as MathML.
        """
        mathML = Printer._print(self, expr)
        unistr = mathML.toxml()
        xmlbstr = unistr.encode('ascii', 'xmlcharrefreplace')
        res = xmlbstr.decode()
        return res

    def _split_super_sub(self, name):
        if self._settings["disable_split_super_sub"]:
            return (name, [], [])
        else:
            return split_super_sub(name)


class MathMLContentPrinter(MathMLPrinterBase):
    """Prints an expression to the Content MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter4.html
    """
    printmethod = "_mathml_content"

    def mathml_tag(self, e):
        """Returns the MathML tag for an expression."""
        translate = {
            'Add': 'plus',
            'Mul': 'times',
            'Derivative': 'diff',
            'Number': 'cn',
            'int': 'cn',
            'Pow': 'power',
            'Max': 'max',
            'Min': 'min',
            'Abs': 'abs',
            'And': 'and',
            'Or': 'or',
            'Xor': 'xor',
            'Not': 'not',
            'Implies': 'implies',
            'Symbol': 'ci',
            'MatrixSymbol': 'ci',
            'RandomSymbol': 'ci',
            'Integral': 'int',
            'Sum': 'sum',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'cot': 'cot',
            'csc': 'csc',
            'sec': 'sec',
            'sinh': 'sinh',
            'cosh': 'cosh',
            'tanh': 'tanh',
            'coth': 'coth',
            'csch': 'csch',
            'sech': 'sech',
            'asin': 'arcsin',
            'asinh': 'arcsinh',
            'acos': 'arccos',
            'acosh': 'arccosh',
            'atan': 'arctan',
            'atanh': 'arctanh',
            'atan2': 'arctan',
            'acot': 'arccot',
            'acoth': 'arccoth',
            'asec': 'arcsec',
            'asech': 'arcsech',
            'acsc': 'arccsc',
            'acsch': 'arccsch',
            'log': 'ln',
            'Equality': 'eq',
            'Unequality': 'neq',
            'GreaterThan': 'geq',
            'LessThan': 'leq',
            'StrictGreaterThan': 'gt',
            'StrictLessThan': 'lt',
            'Union': 'union',
            'Intersection': 'intersect',
        }

        for cls in e.__class__.__mro__:
            n = cls.__name__
            if n in translate:
                return translate[n]
        # Not found in the MRO set
        n = e.__class__.__name__
        return n.lower()

    def _print_Mul(self, expr):

        if expr.could_extract_minus_sign():
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('minus'))
            x.appendChild(self._print_Mul(-expr))
            return x

        from sympy.simplify import fraction
        numer, denom = fraction(expr)

        if denom is not S.One:
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('divide'))
            x.appendChild(self._print(numer))
            x.appendChild(self._print(denom))
            return x

        coeff, terms = expr.as_coeff_mul()
        if coeff is S.One and len(terms) == 1:
            # XXX since the negative coefficient has been handled, I don't
            # think a coeff of 1 can remain
            return self._print(terms[0])

        if self.order != 'old':
            terms = Mul._from_args(terms).as_ordered_factors()

        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('times'))
        if coeff != 1:
            x.appendChild(self._print(coeff))
        for term in terms:
            x.appendChild(self._print(term))
        return x

    def _print_Add(self, expr, order=None):
        args = self._as_ordered_terms(expr, order=order)
        lastProcessed = self._print(args[0])
        plusNodes = []
        for arg in args[1:]:
            if arg.could_extract_minus_sign():
                # use minus
                x = self.dom.createElement('apply')
                x.appendChild(self.dom.createElement('minus'))
                x.appendChild(lastProcessed)
                x.appendChild(self._print(-arg))
                # invert expression since this is now minused
                lastProcessed = x
                if arg == args[-1]:
                    plusNodes.append(lastProcessed)
            else:
                plusNodes.append(lastProcessed)
                lastProcessed = self._print(arg)
                if arg == args[-1]:
                    plusNodes.append(self._print(arg))
        if len(plusNodes) == 1:
            return lastProcessed
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('plus'))
        while plusNodes:
            x.appendChild(plusNodes.pop(0))
        return x

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        root = self.dom.createElement('piecewise')
        for i, (e, c) in enumerate(expr.args):
            if i == len(expr.args) - 1 and c == True:
                piece = self.dom.createElement('otherwise')
                piece.appendChild(self._print(e))
            else:
                piece = self.dom.createElement('piece')
                piece.appendChild(self._print(e))
                piece.appendChild(self._print(c))
            root.appendChild(piece)
        return root

    def _print_MatrixBase(self, m):
        x = self.dom.createElement('matrix')
        for i in range(m.rows):
            x_r = self.dom.createElement('matrixrow')
            for j in range(m.cols):
                x_r.appendChild(self._print(m[i, j]))
            x.appendChild(x_r)
        return x

    def _print_Rational(self, e):
        if e.q == 1:
            # don't divide
            x = self.dom.createElement('cn')
            x.appendChild(self.dom.createTextNode(str(e.p)))
            return x
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('divide'))
        # numerator
        xnum = self.dom.createElement('cn')
        xnum.appendChild(self.dom.createTextNode(str(e.p)))
        # denominator
        xdenom = self.dom.createElement('cn')
        xdenom.appendChild(self.dom.createTextNode(str(e.q)))
        x.appendChild(xnum)
        x.appendChild(xdenom)
        return x

    def _print_Limit(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))

        x_1 = self.dom.createElement('bvar')
        x_2 = self.dom.createElement('lowlimit')
        x_1.appendChild(self._print(e.args[1]))
        x_2.appendChild(self._print(e.args[2]))

        x.appendChild(x_1)
        x.appendChild(x_2)
        x.appendChild(self._print(e.args[0]))
        return x

    def _print_ImaginaryUnit(self, e):
        return self.dom.createElement('imaginaryi')

    def _print_EulerGamma(self, e):
        return self.dom.createElement('eulergamma')

    def _print_GoldenRatio(self, e):
        """We use unicode #x3c6 for Greek letter phi as defined here
        https://www.w3.org/2003/entities/2007doc/isogrk1.html"""
        x = self.dom.createElement('cn')
        x.appendChild(self.dom.createTextNode("\N{GREEK SMALL LETTER PHI}"))
        return x

    def _print_Exp1(self, e):
        return self.dom.createElement('exponentiale')

    def _print_Pi(self, e):
        return self.dom.createElement('pi')

    def _print_Infinity(self, e):
        return self.dom.createElement('infinity')

    def _print_NaN(self, e):
        return self.dom.createElement('notanumber')

    def _print_EmptySet(self, e):
        return self.dom.createElement('emptyset')

    def _print_BooleanTrue(self, e):
        return self.dom.createElement('true')

    def _print_BooleanFalse(self, e):
        return self.dom.createElement('false')

    def _print_NegativeInfinity(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('minus'))
        x.appendChild(self.dom.createElement('infinity'))
        return x

    def _print_Integral(self, e):
        def lime_recur(limits):
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement(self.mathml_tag(e)))
            bvar_elem = self.dom.createElement('bvar')
            bvar_elem.appendChild(self._print(limits[0][0]))
            x.appendChild(bvar_elem)

            if len(limits[0]) == 3:
                low_elem = self.dom.createElement('lowlimit')
                low_elem.appendChild(self._print(limits[0][1]))
                x.appendChild(low_elem)
                up_elem = self.dom.createElement('uplimit')
                up_elem.appendChild(self._print(limits[0][2]))
                x.appendChild(up_elem)
            if len(limits[0]) == 2:
                up_elem = self.dom.createElement('uplimit')
                up_elem.appendChild(self._print(limits[0][1]))
                x.appendChild(up_elem)
            if len(limits) == 1:
                x.appendChild(self._print(e.function))
            else:
                x.appendChild(lime_recur(limits[1:]))
            return x

        limits = list(e.limits)
        limits.reverse()
        return lime_recur(limits)

    def _print_Sum(self, e):
        # Printer can be shared because Sum and Integral have the
        # same internal representation.
        return self._print_Integral(e)

    def _print_Symbol(self, sym):
        ci = self.dom.createElement(self.mathml_tag(sym))

        def join(items):
            if len(items) > 1:
                mrow = self.dom.createElement('mml:mrow')
                for i, item in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mml:mo')
                        mo.appendChild(self.dom.createTextNode(" "))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mml:mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mml:mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi

        # translate name, supers and subs to unicode characters
        def translate(s):
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s

        name, supers, subs = self._split_super_sub(sym.name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        mname = self.dom.createElement('mml:mi')
        mname.appendChild(self.dom.createTextNode(name))
        if not supers:
            if not subs:
                ci.appendChild(self.dom.createTextNode(name))
            else:
                msub = self.dom.createElement('mml:msub')
                msub.appendChild(mname)
                msub.appendChild(join(subs))
                ci.appendChild(msub)
        else:
            if not subs:
                msup = self.dom.createElement('mml:msup')
                msup.appendChild(mname)
                msup.appendChild(join(supers))
                ci.appendChild(msup)
            else:
                msubsup = self.dom.createElement('mml:msubsup')
                msubsup.appendChild(mname)
                msubsup.appendChild(join(subs))
                msubsup.appendChild(join(supers))
                ci.appendChild(msubsup)
        return ci

    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Pow(self, e):
        # Here we use root instead of power if the exponent is the reciprocal
        # of an integer
        if (self._settings['root_notation'] and e.exp.is_Rational
                and e.exp.p == 1):
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('root'))
            if e.exp.q != 2:
                xmldeg = self.dom.createElement('degree')
                xmlcn = self.dom.createElement('cn')
                xmlcn.appendChild(self.dom.createTextNode(str(e.exp.q)))
                xmldeg.appendChild(xmlcn)
                x.appendChild(xmldeg)
            x.appendChild(self._print(e.base))
            return x

        x = self.dom.createElement('apply')
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        x.appendChild(self._print(e.base))
        x.appendChild(self._print(e.exp))
        return x

    def _print_Number(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    def _print_Float(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        repr_e = mlib_to_str(e._mpf_, repr_dps(e._prec))
        x.appendChild(self.dom.createTextNode(repr_e))
        return x

    def _print_Derivative(self, e):
        x = self.dom.createElement('apply')
        diff_symbol = self.mathml_tag(e)
        if requires_partial(e.expr):
            diff_symbol = 'partialdiff'
        x.appendChild(self.dom.createElement(diff_symbol))
        x_1 = self.dom.createElement('bvar')

        for sym, times in reversed(e.variable_count):
            x_1.appendChild(self._print(sym))
            if times > 1:
                degree = self.dom.createElement('degree')
                degree.appendChild(self._print(sympify(times)))
                x_1.appendChild(degree)

        x.appendChild(x_1)
        x.appendChild(self._print(e.expr))
        return x

    def _print_Function(self, e):
        x = self.dom.createElement("apply")
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Basic(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_AssocOp(self, e):
        x = self.dom.createElement('apply')
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Relational(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        x.appendChild(self._print(e.lhs))
        x.appendChild(self._print(e.rhs))
        return x

    def _print_list(self, seq):
        """MathML reference for the <list> element:
        https://www.w3.org/TR/MathML2/chapter4.html#contm.list"""
        dom_element = self.dom.createElement('list')
        for item in seq:
            dom_element.appendChild(self._print(item))
        return dom_element

    def _print_int(self, p):
        dom_element = self.dom.createElement(self.mathml_tag(p))
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        return dom_element

    _print_Implies = _print_AssocOp
    _print_Not = _print_AssocOp
    _print_Xor = _print_AssocOp

    def _print_FiniteSet(self, e):
        x = self.dom.createElement('set')
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Complement(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('setdiff'))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_ProductSet(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('cartesianproduct'))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Lambda(self, e):
        # MathML reference for the lambda element:
        # https://www.w3.org/TR/MathML2/chapter4.html#id.4.2.1.7
        x = self.dom.createElement(self.mathml_tag(e))
        for arg in e.signature:
            x_1 = self.dom.createElement('bvar')
            x_1.appendChild(self._print(arg))
            x.appendChild(x_1)
        x.appendChild(self._print(e.expr))
        return x

    # XXX Symmetric difference is not supported for MathML content printers.


class MathMLPresentationPrinter(MathMLPrinterBase):
    """Prints an expression to the Presentation MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter3.html
    """
    printmethod = "_mathml_presentation"

    def mathml_tag(self, e):
        """Returns the MathML tag for an expression."""
        translate = {
            'Number': 'mn',
            'Limit': '&#x2192;',
            'Derivative': '&dd;',
            'int': 'mn',
            'Symbol': 'mi',
            'Integral': '&int;',
            'Sum': '&#x2211;',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'cot': 'cot',
            'asin': 'arcsin',
            'asinh': 'arcsinh',
            'acos': 'arccos',
            'acosh': 'arccosh',
            'atan': 'arctan',
            'atanh': 'arctanh',
            'acot': 'arccot',
            'atan2': 'arctan',
            'Equality': '=',
            'Unequality': '&#x2260;',
            'GreaterThan': '&#x2265;',
            'LessThan': '&#x2264;',
            'StrictGreaterThan': '>',
            'StrictLessThan': '<',
            'lerchphi': '&#x3A6;',
            'zeta': '&#x3B6;',
            'dirichlet_eta': '&#x3B7;',
            'elliptic_k': '&#x39A;',
            'lowergamma': '&#x3B3;',
            'uppergamma': '&#x393;',
            'gamma': '&#x393;',
            'totient': '&#x3D5;',
            'reduced_totient': '&#x3BB;',
            'primenu': '&#x3BD;',
            'primeomega': '&#x3A9;',
            'fresnels': 'S',
            'fresnelc': 'C',
            'LambertW': 'W',
            'Heaviside': '&#x398;',
            'BooleanTrue': 'True',
            'BooleanFalse': 'False',
            'NoneType': 'None',
            'mathieus': 'S',
            'mathieuc': 'C',
            'mathieusprime': 'S&#x2032;',
            'mathieucprime': 'C&#x2032;',
            'Lambda': 'lambda',
        }

        def mul_symbol_selection():
            if (self._settings["mul_symbol"] is None or
                    self._settings["mul_symbol"] == 'None'):
                return '&InvisibleTimes;'
            elif self._settings["mul_symbol"] == 'times':
                return '&#xD7;'
            elif self._settings["mul_symbol"] == 'dot':
                return '&#xB7;'
            elif self._settings["mul_symbol"] == 'ldot':
                return '&#x2024;'
            elif not isinstance(self._settings["mul_symbol"], str):
                raise TypeError
            else:
                return self._settings["mul_symbol"]
        for cls in e.__class__.__mro__:
            n = cls.__name__
            if n in translate:
                return translate[n]
        # Not found in the MRO set
        if e.__class__.__name__ == "Mul":
            return mul_symbol_selection()
        n = e.__class__.__name__
        return n.lower()

    def _l_paren(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('('))
        return mo

    def _r_paren(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(')'))
        return mo

    def _l_brace(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('{'))
        return mo

    def _r_brace(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('}'))
        return mo

    def _comma(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(','))
        return mo

    def _bar(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('|'))
        return mo

    def _semicolon(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(';'))
        return mo

    def _paren_comma_separated(self, *args):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._l_paren())
        for i, arg in enumerate(args):
            if i:
                mrow.appendChild(self._comma())
            mrow.appendChild(self._print(arg))
        mrow.appendChild(self._r_paren())
        return mrow

    def _paren_bar_separated(self, *args):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._l_paren())
        for i, arg in enumerate(args):
            if i:
                mrow.appendChild(self._bar())
            mrow.appendChild(self._print(arg))
        mrow.appendChild(self._r_paren())
        return mrow

    def parenthesize(self, item, level, strict=False):
        prec_val = precedence_traditional(item)
        if (prec_val < level) or ((not strict) and prec_val <= level):
            mrow = self.dom.createElement('mrow')
            mrow.appendChild(self._l_paren())
            mrow.appendChild(self._print(item))
            mrow.appendChild(self._r_paren())
            return mrow
        return self._print(item)

    def _print_Mul(self, expr):

        def multiply(expr, mrow):
            from sympy.simplify import fraction
            numer, denom = fraction(expr)
            if denom is not S.One:
                frac = self.dom.createElement('mfrac')
                if self._settings["fold_short_frac"] and len(str(expr)) < 7:
                    frac.setAttribute('bevelled', 'true')
                xnum = self._print(numer)
                xden = self._print(denom)
                frac.appendChild(xnum)
                frac.appendChild(xden)
                mrow.appendChild(frac)
                return mrow

            coeff, terms = expr.as_coeff_mul()
            if coeff is S.One and len(terms) == 1:
                mrow.appendChild(self._print(terms[0]))
                return mrow
            if self.order != 'old':
                terms = Mul._from_args(terms).as_ordered_factors()

            if coeff != 1:
                x = self._print(coeff)
                y = self.dom.createElement('mo')
                y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                mrow.appendChild(x)
                mrow.appendChild(y)
            for term in terms:
                mrow.appendChild(self.parenthesize(term, PRECEDENCE['Mul']))
                if not term == terms[-1]:
                    y = self.dom.createElement('mo')
                    y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                    mrow.appendChild(y)
            return mrow
        mrow = self.dom.createElement('mrow')
        if expr.could_extract_minus_sign():
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode('-'))
            mrow.appendChild(x)
            mrow = multiply(-expr, mrow)
        else:
            mrow = multiply(expr, mrow)

        return mrow

    def _print_Add(self, expr, order=None):
        mrow = self.dom.createElement('mrow')
        args = self._as_ordered_terms(expr, order=order)
        mrow.appendChild(self._print(args[0]))
        for arg in args[1:]:
            if arg.could_extract_minus_sign():
                # use minus
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('-'))
                y = self._print(-arg)
                # invert expression since this is now minused
            else:
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('+'))
                y = self._print(arg)
            mrow.appendChild(x)
            mrow.appendChild(y)

        return mrow

    def _print_MatrixBase(self, m):
        table = self.dom.createElement('mtable')
        for i in range(m.rows):
            x = self.dom.createElement('mtr')
            for j in range(m.cols):
                y = self.dom.createElement('mtd')
                y.appendChild(self._print(m[i, j]))
                x.appendChild(y)
            table.appendChild(x)
        mat_delim = self._settings["mat_delim"]
        if mat_delim == '':
            return table
        left = self.dom.createElement('mo')
        right = self.dom.createElement('mo')
        if mat_delim == "[":
            left.appendChild(self.dom.createTextNode("["))
            right.appendChild(self.dom.createTextNode("]"))
        else:
            left.appendChild(self.dom.createTextNode("("))
            right.appendChild(self.dom.createTextNode(")"))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(left)
        mrow.appendChild(table)
        mrow.appendChild(right)
        return mrow

    def _get_printed_Rational(self, e, folded=None):
        if e.p < 0:
            p = -e.p
        else:
            p = e.p
        x = self.dom.createElement('mfrac')
        if folded or self._settings["fold_short_frac"]:
            x.setAttribute('bevelled', 'true')
        x.appendChild(self._print(p))
        x.appendChild(self._print(e.q))
        if e.p < 0:
            mrow = self.dom.createElement('mrow')
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('-'))
            mrow.appendChild(mo)
            mrow.appendChild(x)
            return mrow
        else:
            return x

    def _print_Rational(self, e):
        if e.q == 1:
            # don't divide
            return self._print(e.p)

        return self._get_printed_Rational(e, self._settings["fold_short_frac"])

    def _print_Limit(self, e):
        mrow = self.dom.createElement('mrow')
        munder = self.dom.createElement('munder')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('lim'))

        x = self.dom.createElement('mrow')
        x_1 = self._print(e.args[1])
        arrow = self.dom.createElement('mo')
        arrow.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        x_2 = self._print(e.args[2])
        x.appendChild(x_1)
        x.appendChild(arrow)
        x.appendChild(x_2)

        munder.appendChild(mi)
        munder.appendChild(x)
        mrow.appendChild(munder)
        mrow.appendChild(self._print(e.args[0]))

        return mrow

    def _print_ImaginaryUnit(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&ImaginaryI;'))
        return x

    def _print_GoldenRatio(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x3A6;'))
        return x

    def _print_Exp1(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&ExponentialE;'))
        return x

    def _print_Pi(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&pi;'))
        return x

    def _print_Infinity(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x221E;'))
        return x

    def _print_NegativeInfinity(self, e):
        mrow = self.dom.createElement('mrow')
        y = self.dom.createElement('mo')
        y.appendChild(self.dom.createTextNode('-'))
        x = self._print_Infinity(e)
        mrow.appendChild(y)
        mrow.appendChild(x)
        return mrow

    def _print_HBar(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x210F;'))
        return x

    def _print_EulerGamma(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x3B3;'))
        return x

    def _print_TribonacciConstant(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('TribonacciConstant'))
        return x

    def _print_Dagger(self, e):
        msup = self.dom.createElement('msup')
        msup.appendChild(self._print(e.args[0]))
        msup.appendChild(self.dom.createTextNode('&#x2020;'))
        return msup

    def _print_Contains(self, e):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._print(e.args[0]))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2208;'))
        mrow.appendChild(mo)
        mrow.appendChild(self._print(e.args[1]))
        return mrow

    def _print_HilbertSpace(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x210B;'))
        return x

    def _print_ComplexSpace(self, e):
        msup = self.dom.createElement('msup')
        msup.appendChild(self.dom.createTextNode('&#x1D49E;'))
        msup.appendChild(self._print(e.args[0]))
        return msup

    def _print_FockSpace(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x2131;'))
        return x


    def _print_Integral(self, expr):
        intsymbols = {1: "&#x222B;", 2: "&#x222C;", 3: "&#x222D;"}

        mrow = self.dom.createElement('mrow')
        if len(expr.limits) <= 3 and all(len(lim) == 1 for lim in expr.limits):
            # Only up to three-integral signs exists
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode(intsymbols[len(expr.limits)]))
            mrow.appendChild(mo)
        else:
            # Either more than three or limits provided
            for lim in reversed(expr.limits):
                mo = self.dom.createElement('mo')
                mo.appendChild(self.dom.createTextNode(intsymbols[1]))
                if len(lim) == 1:
                    mrow.appendChild(mo)
                if len(lim) == 2:
                    msup = self.dom.createElement('msup')
                    msup.appendChild(mo)
                    msup.appendChild(self._print(lim[1]))
                    mrow.appendChild(msup)
                if len(lim) == 3:
                    msubsup = self.dom.createElement('msubsup')
                    msubsup.appendChild(mo)
                    msubsup.appendChild(self._print(lim[1]))
                    msubsup.appendChild(self._print(lim[2]))
                    mrow.appendChild(msubsup)
        # print function
        mrow.appendChild(self.parenthesize(expr.function, PRECEDENCE["Mul"],
                                           strict=True))
        # print integration variables
        for lim in reversed(expr.limits):
            d = self.dom.createElement('mo')
            d.appendChild(self.dom.createTextNode('&dd;'))
            mrow.appendChild(d)
            mrow.appendChild(self._print(lim[0]))
        return mrow

    def _print_Sum(self, e):
        limits = list(e.limits)
        subsup = self.dom.createElement('munderover')
        low_elem = self._print(limits[0][1])
        up_elem = self._print(limits[0][2])
        summand = self.dom.createElement('mo')
        summand.appendChild(self.dom.createTextNode(self.mathml_tag(e)))

        low = self.dom.createElement('mrow')
        var = self._print(limits[0][0])
        equal = self.dom.createElement('mo')
        equal.appendChild(self.dom.createTextNode('='))
        low.appendChild(var)
        low.appendChild(equal)
        low.appendChild(low_elem)

        subsup.appendChild(summand)
        subsup.appendChild(low)
        subsup.appendChild(up_elem)

        mrow = self.dom.createElement('mrow')
        mrow.appendChild(subsup)
        mrow.appendChild(self.parenthesize(e.function, precedence_traditional(e)))
        return mrow

    def _print_Symbol(self, sym, style='plain'):
        def join(items):
            if len(items) > 1:
                mrow = self.dom.createElement('mrow')
                for i, item in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode(" "))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi

        # translate name, supers and subs to unicode characters
        def translate(s):
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s

        name, supers, subs = self._split_super_sub(sym.name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        mname = self.dom.createElement('mi')
        mname.appendChild(self.dom.createTextNode(name))
        if len(supers) == 0:
            if len(subs) == 0:
                x = mname
            else:
                x = self.dom.createElement('msub')
                x.appendChild(mname)
                x.appendChild(join(subs))
        else:
            if len(subs) == 0:
                x = self.dom.createElement('msup')
                x.appendChild(mname)
                x.appendChild(join(supers))
            else:
                x = self.dom.createElement('msubsup')
                x.appendChild(mname)
                x.appendChild(join(subs))
                x.appendChild(join(supers))
        # Set bold font?
        if style == 'bold':
            x.setAttribute('mathvariant', 'bold')
        return x

    def _print_MatrixSymbol(self, sym):
        return self._print_Symbol(sym,
                                  style=self._settings['mat_symbol_style'])

    _print_RandomSymbol = _print_Symbol

    def _print_conjugate(self, expr):
        enc = self.dom.createElement('menclose')
        enc.setAttribute('notation', 'top')
        enc.appendChild(self._print(expr.args[0]))
        return enc

    def _print_operator_after(self, op, expr):
        row = self.dom.createElement('mrow')
        row.appendChild(self.parenthesize(expr, PRECEDENCE["Func"]))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(op))
        row.appendChild(mo)
        return row

    def _print_factorial(self, expr):
        return self._print_operator_after('!', expr.args[0])

    def _print_factorial2(self, expr):
        return self._print_operator_after('!!', expr.args[0])

    def _print_binomial(self, expr):
        frac = self.dom.createElement('mfrac')
        frac.setAttribute('linethickness', '0')
        frac.appendChild(self._print(expr.args[0]))
        frac.appendChild(self._print(expr.args[1]))
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_paren())
        brac.appendChild(frac)
        brac.appendChild(self._r_paren())
        return brac

    def _print_Pow(self, e):
        # Here we use root instead of power if the exponent is the
        # reciprocal of an integer
        if (e.exp.is_Rational and abs(e.exp.p) == 1 and e.exp.q != 1 and
                self._settings['root_notation']):
            if e.exp.q == 2:
                x = self.dom.createElement('msqrt')
                x.appendChild(self._print(e.base))
            if e.exp.q != 2:
                x = self.dom.createElement('mroot')
                x.appendChild(self._print(e.base))
                x.appendChild(self._print(e.exp.q))
            if e.exp.p == -1:
                frac = self.dom.createElement('mfrac')
                frac.appendChild(self._print(1))
                frac.appendChild(x)
                return frac
            else:
                return x

        if e.exp.is_Rational and e.exp.q != 1:
            if e.exp.is_negative:
                top = self.dom.createElement('mfrac')
                top.appendChild(self._print(1))
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._get_printed_Rational(-e.exp,
                                    self._settings['fold_frac_powers']))
                top.appendChild(x)
                return top
            else:
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._get_printed_Rational(e.exp,
                                    self._settings['fold_frac_powers']))
                return x

        if e.exp.is_negative:
                top = self.dom.createElement('mfrac')
                top.appendChild(self._print(1))
                if e.exp == -1:
                    top.appendChild(self._print(e.base))
                else:
                    x = self.dom.createElement('msup')
                    x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                    x.appendChild(self._print(-e.exp))
                    top.appendChild(x)
                return top

        x = self.dom.createElement('msup')
        x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
        x.appendChild(self._print(e.exp))
        return x

    def _print_Number(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    def _print_AccumulationBounds(self, i):
        left = self.dom.createElement('mo')
        left.appendChild(self.dom.createTextNode('\u27e8'))
        right = self.dom.createElement('mo')
        right.appendChild(self.dom.createTextNode('\u27e9'))
        brac = self.dom.createElement('mrow')
        brac.appendChild(left)
        brac.appendChild(self._print(i.min))
        brac.appendChild(self._comma())
        brac.appendChild(self._print(i.max))
        brac.appendChild(right)
        return brac

    def _print_Derivative(self, e):

        if requires_partial(e.expr):
            d = '&#x2202;'
        else:
            d = self.mathml_tag(e)

        # Determine denominator
        m = self.dom.createElement('mrow')
        dim = 0  # Total diff dimension, for numerator
        for sym, num in reversed(e.variable_count):
            dim += num
            if num >= 2:
                x = self.dom.createElement('msup')
                xx = self.dom.createElement('mo')
                xx.appendChild(self.dom.createTextNode(d))
                x.appendChild(xx)
                x.appendChild(self._print(num))
            else:
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode(d))
            m.appendChild(x)
            y = self._print(sym)
            m.appendChild(y)

        mnum = self.dom.createElement('mrow')
        if dim >= 2:
            x = self.dom.createElement('msup')
            xx = self.dom.createElement('mo')
            xx.appendChild(self.dom.createTextNode(d))
            x.appendChild(xx)
            x.appendChild(self._print(dim))
        else:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(d))

        mnum.appendChild(x)
        mrow = self.dom.createElement('mrow')
        frac = self.dom.createElement('mfrac')
        frac.appendChild(mnum)
        frac.appendChild(m)
        mrow.appendChild(frac)

        # Print function
        mrow.appendChild(self._print(e.expr))

        return mrow

    def _print_Function(self, e):
        x = self.dom.createElement('mi')
        if self.mathml_tag(e) == 'log' and self._settings["ln_notation"]:
            x.appendChild(self.dom.createTextNode('ln'))
        else:
            x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(x)
        mrow.appendChild(self._paren_comma_separated(*e.args))
        return mrow

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=True)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_mathml_numbers']
        mrow = self.dom.createElement('mrow')
        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(mant))
            mrow.appendChild(mn)
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode(separator))
            mrow.appendChild(mo)
            msup = self.dom.createElement('msup')
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode("10"))
            msup.appendChild(mn)
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(exp))
            msup.appendChild(mn)
            mrow.appendChild(msup)
            return mrow
        elif str_real == "+inf":
            return self._print_Infinity(None)
        elif str_real == "-inf":
            return self._print_NegativeInfinity(None)
        else:
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(str_real))
            return mn

    def _print_polylog(self, expr):
        mrow = self.dom.createElement('mrow')
        m = self.dom.createElement('msub')

        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('Li'))
        m.appendChild(mi)
        m.appendChild(self._print(expr.args[0]))
        mrow.appendChild(m)
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_paren())
        brac.appendChild(self._print(expr.args[1]))
        brac.appendChild(self._r_paren())
        mrow.appendChild(brac)
        return mrow

    def _print_Basic(self, e):
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        mrow.appendChild(self._paren_comma_separated(*e.args))
        return mrow

    def _print_Tuple(self, e):
        return self._paren_comma_separated(*e.args)

    def _print_Interval(self, i):
        right = self.dom.createElement('mo')
        if i.right_open:
            right.appendChild(self.dom.createTextNode(')'))
        else:
            right.appendChild(self.dom.createTextNode(']'))
        left = self.dom.createElement('mo')
        if i.left_open:
            left.appendChild(self.dom.createTextNode('('))
        else:
            left.appendChild(self.dom.createTextNode('['))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(left)
        mrow.appendChild(self._print(i.start))
        mrow.appendChild(self._comma())
        mrow.appendChild(self._print(i.end))
        mrow.appendChild(right)
        return mrow

    def _print_Abs(self, expr, exp=None):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._bar())
        mrow.appendChild(self._print(expr.args[0]))
        mrow.appendChild(self._bar())
        return mrow

    _print_Determinant = _print_Abs

    def _print_re_im(self, c, expr):
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_paren())
        brac.appendChild(self._print(expr))
        brac.appendChild(self._r_paren())
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(c))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(mi)
        mrow.appendChild(brac)
        return mrow

    def _print_re(self, expr, exp=None):
        return self._print_re_im('\u211C', expr.args[0])

    def _print_im(self, expr, exp=None):
        return self._print_re_im('\u2111', expr.args[0])

    def _print_AssocOp(self, e):
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        for arg in e.args:
            mrow.appendChild(self._print(arg))
        return mrow

    def _print_SetOp(self, expr, symbol, prec):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self.parenthesize(expr.args[0], prec))
        for arg in expr.args[1:]:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(symbol))
            y = self.parenthesize(arg, prec)
            mrow.appendChild(x)
            mrow.appendChild(y)
        return mrow

    def _print_Union(self, expr):
        prec = PRECEDENCE_TRADITIONAL['Union']
        return self._print_SetOp(expr, '&#x222A;', prec)

    def _print_Intersection(self, expr):
        prec = PRECEDENCE_TRADITIONAL['Intersection']
        return self._print_SetOp(expr, '&#x2229;', prec)

    def _print_Complement(self, expr):
        prec = PRECEDENCE_TRADITIONAL['Complement']
        return self._print_SetOp(expr, '&#x2216;', prec)

    def _print_SymmetricDifference(self, expr):
        prec = PRECEDENCE_TRADITIONAL['SymmetricDifference']
        return self._print_SetOp(expr, '&#x2206;', prec)

    def _print_ProductSet(self, expr):
        prec = PRECEDENCE_TRADITIONAL['ProductSet']
        return self._print_SetOp(expr, '&#x00d7;', prec)

    def _print_FiniteSet(self, s):
        return self._print_set(s.args)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_brace())
        for i, item in enumerate(items):
            if i:
                brac.appendChild(self._comma())
            brac.appendChild(self._print(item))
        brac.appendChild(self._r_brace())
        return brac

    _print_frozenset = _print_set

    def _print_LogOp(self, args, symbol):
        mrow = self.dom.createElement('mrow')
        if args[0].is_Boolean and not args[0].is_Not:
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(args[0]))
            brac.appendChild(self._r_paren())
            mrow.appendChild(brac)
        else:
            mrow.appendChild(self._print(args[0]))
        for arg in args[1:]:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(symbol))
            if arg.is_Boolean and not arg.is_Not:
                y = self.dom.createElement('mrow')
                y.appendChild(self._l_paren())
                y.appendChild(self._print(arg))
                y.appendChild(self._r_paren())
            else:
                y = self._print(arg)
            mrow.appendChild(x)
            mrow.appendChild(y)
