        """function to calculate score based on position among indices

        This method is used to sort loops in an optimized order, see
        CodePrinter._sort_optimized()
        """
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_statement(self, codestring):
        """Formats a codestring with the proper line ending."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_comment(self, text):
        """Formats a text string as a comment."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _declare_number_const(self, name, value):
        """Declare a numeric constant at the top of a function"""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _format_code(self, lines):
        """Take in a list of lines of code, and format them accordingly.

        This may include indenting, wrapping long lines, etc..."""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _get_loop_opening_ending(self, indices):
        """Returns a tuple (open_lines, close_lines) containing lists
        of codelines"""
        raise NotImplementedError("This function must be implemented by "
                                  "subclass of CodePrinter.")

    def _print_Dummy(self, expr):
        if expr.name.startswith('Dummy_'):
            return '_' + expr.name
        else:
            return '%s_%d' % (expr.name, expr.dummy_index)

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_CodeBlock(self, expr):
        return '\n'.join([self._print(i) for i in expr.args])

    def _print_String(self, string):
        return str(string)

    def _print_QuotedString(self, arg):
        return '"%s"' % arg.text

    def _print_Comment(self, string):
        return self._get_comment(str(string))

    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        elif isinstance(lhs, MatrixSymbol):
            # Here we form an Assignment for each element in the array,
            # printing each one.
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return "\n".join(lines)
        elif self._settings.get("contract", False) and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement("{} {} {}".format(
            *(self._print(arg) for arg in [lhs_code, expr.op, rhs_code])))

    def _print_FunctionCall(self, expr):
        return '%s(%s)' % (
            expr.name,
            ', '.join((self._print(arg) for arg in expr.function_args)))

    def _print_Variable(self, expr):
        return self._print(expr.symbol)

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)

        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = ('This expression includes the symbol "{}" which is a '
                       'reserved keyword in this language.')
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        else:
            return name

    def _can_print(self, name):
        """ Check if function ``name`` is either a known function or has its own
            printing method. Used to check if rewriting is possible."""
        return name in self.known_functions or getattr(self, '_print_{}'.format(name), False)

    def _print_Function(self, expr):
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            if isinstance(cond_func, str):
                return "%s(%s)" % (cond_func, self.stringify(expr.args, ", "))
            else:
                for cond, func in cond_func:
                    if cond(*expr.args):
                        break
                if func is not None:
                    try:
                        return func(*[self.parenthesize(item, 0) for item in expr.args])
                    except TypeError:
                        return "%s(%s)" % (func, self.stringify(expr.args, ", "))
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            # inlined function
            return self._print(expr._imp_(*expr.args))
        elif expr.func.__name__ in self._rewriteable_functions:
            # Simple rewrite to supported function possible
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all(self._can_print(f) for f in required_fs):
                return '(' + self._print(expr.rewrite(target_f)) + ')'

        if expr.is_Function and self._settings.get('allow_unknown_functions', False):
            return '%s(%s)' % (self._print(expr.func), ', '.join(map(self._print, expr.args)))
        else:
            return self._print_not_supported(expr)

    _print_Expr = _print_Function

    def _print_Derivative(self, expr):
        obj, *wrt_order_pairs = expr.args
        for func_arg in obj.args:
            if not func_arg.is_Symbol:
                raise ValueError("%s._print_Derivative(...) only supports functions with symbols as arguments." %
                                 self.__class__.__name__)
        meth_name = '_print_Derivative_%s' % obj.func.__name__
        pmeth = getattr(self, meth_name, None)
        if pmeth is None:
            if self._settings.get('strict', False):
                raise PrintMethodNotImplementedError(
                    f"Unsupported by {type(self)}: {type(expr)}" +
                    f"\nPrinter has no method: {meth_name}" +
                    "\nSet the printer option 'strict' to False in order to generate partially printed code."
                )
            return self._print_not_supported(expr)
        orders = dict(wrt_order_pairs)
        seq_orders = [orders[arg] for arg in obj.args]
        return pmeth(obj.args, seq_orders)

    # Don't inherit the str-printer method for Heaviside to the code printers
    _print_Heaviside = None

    def _print_NumberSymbol(self, expr):
        if self._settings.get("inline", False):
            return self._print(Float(expr.evalf(self._settings["precision"])))
        else:
            # A Number symbol that is not implemented here or with _printmethod
            # is registered and evaluated
            self._number_symbols.add((expr,
                Float(expr.evalf(self._settings["precision"]))))
            return str(expr)

    def _print_Catalan(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_EulerGamma(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_GoldenRatio(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_TribonacciConstant(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_Exp1(self, expr):
        return self._print_NumberSymbol(expr)
    def _print_Pi(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_And(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['and']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return (" %s " % self._operators['or']).join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Xor(self, expr):
        if self._operators.get('xor') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (" %s " % self._operators['xor']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    def _print_Equivalent(self, expr):
        if self._operators.get('equivalent') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (" %s " % self._operators['equivalent']).join(self.parenthesize(a, PREC)
                for a in expr.args)

    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)

    def _print_BooleanFunction(self, expr):
        return self._print(expr.to_nnf())

    def _print_isnan(self, arg):
        return 'isnan(%s)' % self._print(*arg.args)

    def _print_isinf(self, arg):
        return 'isinf(%s)' % self._print(*arg.args)

    def _print_Mul(self, expr):

        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            else:
                a.append(item)

        a = a or [S.One]

        if len(a) == 1 and sign == "-":
            # Unary minus does not have a SymPy class, and hence there's no
            # precedence weight associated with it, Python's unary minus has
            # an operator precedence between multiplication and exponentiation,
            # so we use this to compute a weight.
            a_str = [self.parenthesize(a[0], 0.5*(PRECEDENCE["Pow"]+PRECEDENCE["Mul"]))]
        else:
            a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)

    def _print_not_supported(self, expr):
        if self._settings.get('strict', False):
            raise PrintMethodNotImplementedError(
                f"Unsupported by {type(self)}: {type(expr)}" +
                "\nSet the printer option 'strict' to False in order to generate partially printed code."
            )
        try:
            self._not_supported.add(expr)
        except TypeError:
            # not hashable
            pass
        return self.emptyPrinter(expr)

    # The following can not be simply translated into C or Fortran
    _print_Basic = _print_not_supported
    _print_ComplexInfinity = _print_not_supported
    _print_ExprCondPair = _print_not_supported
    _print_GeometryEntity = _print_not_supported
    _print_Infinity = _print_not_supported
    _print_Integral = _print_not_supported
    _print_Interval = _print_not_supported
    _print_AccumulationBounds = _print_not_supported
    _print_Limit = _print_not_supported
    _print_MatrixBase = _print_not_supported
    _print_DeferredVector = _print_not_supported
    _print_NaN = _print_not_supported
    _print_NegativeInfinity = _print_not_supported
    _print_Order = _print_not_supported
    _print_RootOf = _print_not_supported
    _print_RootsOf = _print_not_supported
    _print_RootSum = _print_not_supported
    _print_Uniform = _print_not_supported
    _print_Unit = _print_not_supported
    _print_Wild = _print_not_supported
    _print_WildFunction = _print_not_supported
    _print_Relational = _print_not_supported


# Code printer functions. These are included in this file so that they can be
# imported in the top-level __init__.py without importing the sympy.codegen
# module.

def ccode(expr, assign_to=None, standard='c99', **settings):
    """Converts an expr to a string of c code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    standard : str, optional
        String specifying the standard. If your compiler supports a more modern
        standard you may set this to 'c99' to allow the printer to use more math
        functions. [default='c89'].
    precision : integer, optional
        The precision for numbers such as pi [default=17].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)] or [(argument_test, cfunction_formater)]. See below
        for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
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

    >>> from sympy import ccode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> expr = (2*tau)**Rational(7, 2)
    >>> ccode(expr)
    '8*M_SQRT2*pow(tau, 7.0/2.0)'
    >>> ccode(expr, math_macros={})
    '8*sqrt(2)*pow(tau, 7.0/2.0)'
    >>> ccode(sin(x), assign_to="s")
    's = sin(x);'
    >>> from sympy.codegen.ast import real, float80
    >>> ccode(expr, type_aliases={real: float80})
    '8*M_SQRT2l*powl(tau, 7.0L/2.0L)'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> ccode(func(Abs(x) + ceiling(x)), standard='C89', user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    or if the C-function takes a subset of the original arguments:

    >>> ccode(2**x + 3**x, standard='C99', user_functions={'Pow': [
    ...   (lambda b, e: b == 2, lambda b, e: 'exp2(%s)' % e),
    ...   (lambda b, e: b != 2, 'pow')]})
    'exp2(x) + pow(3, x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(ccode(expr, tau, standard='C89'))
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
    >>> ccode(e.rhs, assign_to=e.lhs, contract=False, standard='C89')
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(ccode(mat, A, standard='C89'))
    A[0] = pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = sin(x);
    """
    from sympy.printing.c import c_code_printers
    return c_code_printers[standard.lower()](settings).doprint(expr, assign_to)

def print_ccode(expr, **settings):
    """Prints C representation of the given expression."""
    print(ccode(expr, **settings))

def fcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of fortran code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        DEPRECATED. Use type_mappings instead. The precision for numbers such
        as pi [default=17].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
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
    source_format : optional
        The source format can be either 'fixed' or 'free'. [default='fixed']
    standard : integer, optional
        The Fortran standard to be followed. This is specified as an integer.
        Acceptable standards are 66, 77, 90, 95, 2003, and 2008. Default is 77.
        Note that currently the only distinction internally is between
        standards before 95, and those 95 and after. This may change later as
        more features are added.
    name_mangling : bool, optional
        If True, then the variables that would become identical in
        case-insensitive Fortran are mangled by appending different number
        of ``_`` at the end. If False, SymPy Will not interfere with naming of
        variables. [default=True]

    Examples
    ========

    >>> from sympy import fcode, symbols, Rational, sin, ceiling, floor
    >>> x, tau = symbols("x, tau")
    >>> fcode((2*tau)**Rational(7, 2))
    '      8*sqrt(2.0d0)*tau**(7.0d0/2.0d0)'
    >>> fcode(sin(x), assign_to="s")
    '      s = sin(x)'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "floor": [(lambda x: not x.is_integer, "FLOOR1"),
    ...             (lambda x: x.is_integer, "FLOOR2")]
    ... }
    >>> fcode(floor(x) + ceiling(x), user_functions=custom_functions)
    '      CEIL(x) + FLOOR1(x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(fcode(expr, tau))
          if (x > 0) then
             tau = x + 1
          else
             tau = x
          end if

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
    >>> fcode(e.rhs, assign_to=e.lhs, contract=False)
    '      Dy(i) = (y(i + 1) - y(i))/(t(i + 1) - t(i))'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(fcode(mat, A))
          A(1, 1) = x**2
             if (x > 0) then
          A(2, 1) = x + 1
             else
          A(2, 1) = x
             end if
          A(3, 1) = sin(x)
    """
    from sympy.printing.fortran import FCodePrinter
    return FCodePrinter(settings).doprint(expr, assign_to)


def print_fcode(expr, **settings):
    """Prints the Fortran representation of the given expression.

       See fcode for the meaning of the optional arguments.
    """
    print(fcode(expr, **settings))

def cxxcode(expr, assign_to=None, standard='c++11', **settings):
    """ C++ equivalent of :func:`~.ccode`. """
    from sympy.printing.cxx import cxx_code_printers
    return cxx_code_printers[standard.lower()](settings).doprint(expr, assign_to)


def rust_code(expr, assign_to=None, **settings):
    """Converts an expr to a string of Rust code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)].  See below for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
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

    >>> from sympy import rust_code, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rust_code((2*tau)**Rational(7, 2))
    '8.0*1.4142135623731*tau.powf(7_f64/2.0)'
    >>> rust_code(sin(x), assign_to="s")
    's = x.sin();'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs", 4),
    ...           (lambda x: x.is_integer, "ABS", 4)],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> rust_code(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    '(fabs(x) + x.ceil()).f()'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(rust_code(expr, tau))
    tau = if (x > 0.0) {
        x + 1
    } else {
        x
    };

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
    >>> rust_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(rust_code(mat, A))
    A = [x.powi(2), if (x > 0.0) {
        x + 1
    } else {
        x
    }, x.sin()];
    """
    from sympy.printing.rust import RustCodePrinter
    printer = RustCodePrinter(settings)
    expr = printer._rewrite_known_functions(expr)
    if isinstance(expr, Expr):
        for src_func, dst_func in printer.function_overrides.values():
            expr = expr.replace(src_func, dst_func)
    return printer.doprint(expr, assign_to)


def print_rust_code(expr, **settings):
    """Prints Rust representation of the given expression."""
    print(rust_code(expr, **settings))
<|endoftext|>'''
Use llvmlite to create executable functions from SymPy expressions

This module requires llvmlite (https://github.com/numba/llvmlite).
'''

import ctypes

from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on

llvmlite = import_module('llvmlite')
if llvmlite:
    ll = import_module('llvmlite.ir').ir
    llvm = import_module('llvmlite.binding').binding
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()


__doctest_requires__ = {('llvm_callable'): ['llvmlite']}


class LLVMJitPrinter(Printer):
    '''Convert expressions to LLVM IR'''
    def __init__(self, module, builder, fn, *args, **kwargs):
        self.func_arg_map = kwargs.pop("func_arg_map", {})
        if not llvmlite:
            raise ImportError("llvmlite is required for LLVMJITPrinter")
        super().__init__(*args, **kwargs)
        self.fp_type = ll.DoubleType()
        self.module = module
        self.builder = builder
        self.fn = fn
        self.ext_fn = {}  # keep track of wrappers to external functions
        self.tmp_var = {}

    def _add_tmp_var(self, name, value):
        self.tmp_var[name] = value

    def _print_Number(self, n):
        return ll.Constant(self.fp_type, float(n))

    def _print_Integer(self, expr):
        return ll.Constant(self.fp_type, float(expr.p))

    def _print_Symbol(self, s):
        val = self.tmp_var.get(s)
        if not val:
            # look up parameter with name s
            val = self.func_arg_map.get(s)
        if not val:
            raise LookupError("Symbol not found: %s" % s)
        return val

    def _print_Pow(self, expr):
        base0 = self._print(expr.base)
        if expr.exp == S.NegativeOne:
            return self.builder.fdiv(ll.Constant(self.fp_type, 1.0), base0)
        if expr.exp == S.Half:
            fn = self.ext_fn.get("sqrt")
            if not fn:
                fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
                fn = ll.Function(self.module, fn_type, "sqrt")
                self.ext_fn["sqrt"] = fn
            return self.builder.call(fn, [base0], "sqrt")
        if expr.exp == 2:
            return self.builder.fmul(base0, base0)

        exp0 = self._print(expr.exp)
        fn = self.ext_fn.get("pow")
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type, self.fp_type])
            fn = ll.Function(self.module, fn_type, "pow")
            self.ext_fn["pow"] = fn
        return self.builder.call(fn, [base0, exp0], "pow")

    def _print_Mul(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fmul(e, node)
        return e

    def _print_Add(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fadd(e, node)
        return e

    # TODO - assumes all called functions take one double precision argument.
    #        Should have a list of math library functions to validate this.
    def _print_Function(self, expr):
        name = expr.func.__name__
        e0 = self._print(expr.args[0])
        fn = self.ext_fn.get(name)
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
            fn = ll.Function(self.module, fn_type, name)
            self.ext_fn[name] = fn
        return self.builder.call(fn, [e0], name)

    def emptyPrinter(self, expr):
        raise TypeError("Unsupported type for LLVM JIT conversion: %s"
                        % type(expr))


# Used when parameters are passed by array.  Often used in callbacks to
# handle a variable number of parameters.
class LLVMJitCallbackPrinter(LLVMJitPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_Indexed(self, expr):
        array, idx = self.func_arg_map[expr.base]
        offset = int(expr.indices[0].evalf())
        array_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32), offset)])
        fp_array_ptr = self.builder.bitcast(array_ptr, ll.PointerType(self.fp_type))
        value = self.builder.load(fp_array_ptr)
        return value

    def _print_Symbol(self, s):
        val = self.tmp_var.get(s)
        if val:
            return val

        array, idx = self.func_arg_map.get(s, [None, 0])
        if not array:
            raise LookupError("Symbol not found: %s" % s)
        array_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32), idx)])
        fp_array_ptr = self.builder.bitcast(array_ptr,
                                            ll.PointerType(self.fp_type))
        value = self.builder.load(fp_array_ptr)
        return value


# ensure lifetime of the execution engine persists (else call to compiled
#   function will seg fault)
exe_engines = []

# ensure names for generated functions are unique
link_names = set()
current_link_suffix = 0


class LLVMJitCode:
    def __init__(self, signature):
        self.signature = signature
        self.fp_type = ll.DoubleType()
        self.module = ll.Module('mod1')
        self.fn = None
        self.llvm_arg_types = []
        self.llvm_ret_type = self.fp_type
        self.param_dict = {}  # map symbol name to LLVM function argument
        self.link_name = ''

    def _from_ctype(self, ctype):
        if ctype == ctypes.c_int:
            return ll.IntType(32)
        if ctype == ctypes.c_double:
            return self.fp_type
        if ctype == ctypes.POINTER(ctypes.c_double):
            return ll.PointerType(self.fp_type)
        if ctype == ctypes.c_void_p:
            return ll.PointerType(ll.IntType(32))
        if ctype == ctypes.py_object:
            return ll.PointerType(ll.IntType(32))

        print("Unhandled ctype = %s" % str(ctype))

    def _create_args(self, func_args):
        """Create types for function arguments"""
        self.llvm_ret_type = self._from_ctype(self.signature.ret_type)
        self.llvm_arg_types = \
            [self._from_ctype(a) for a in self.signature.arg_ctypes]

    def _create_function_base(self):
        """Create function with name and type signature"""
        global current_link_suffix
        default_link_name = 'jit_func'
        current_link_suffix += 1
        self.link_name = default_link_name + str(current_link_suffix)
        link_names.add(self.link_name)

        fn_type = ll.FunctionType(self.llvm_ret_type, self.llvm_arg_types)
        self.fn = ll.Function(self.module, fn_type, name=self.link_name)

    def _create_param_dict(self, func_args):
        """Mapping of symbolic values to function arguments"""
        for i, a in enumerate(func_args):
            self.fn.args[i].name = str(a)
            self.param_dict[a] = self.fn.args[i]

    def _create_function(self, expr):
        """Create function body and return LLVM IR"""
        bb_entry = self.fn.append_basic_block('entry')
        builder = ll.IRBuilder(bb_entry)

        lj = LLVMJitPrinter(self.module, builder, self.fn,
                            func_arg_map=self.param_dict)

        ret = self._convert_expr(lj, expr)
        lj.builder.ret(self._wrap_return(lj, ret))

        strmod = str(self.module)
        return strmod

    def _wrap_return(self, lj, vals):
        # Return a single double if there is one return value,
        #  else return a tuple of doubles.

        # Don't wrap return value in this case
        if self.signature.ret_type == ctypes.c_double:
            return vals[0]

        # Use this instead of a real PyObject*
        void_ptr = ll.PointerType(ll.IntType(32))

        # Create a wrapped double: PyObject* PyFloat_FromDouble(double v)
        wrap_type = ll.FunctionType(void_ptr, [self.fp_type])
        wrap_fn = ll.Function(lj.module, wrap_type, "PyFloat_FromDouble")

        wrapped_vals = [lj.builder.call(wrap_fn, [v]) for v in vals]
        if len(vals) == 1:
            final_val = wrapped_vals[0]
        else:
            # Create a tuple: PyObject* PyTuple_Pack(Py_ssize_t n, ...)

            # This should be Py_ssize_t
            tuple_arg_types = [ll.IntType(32)]

            tuple_arg_types.extend([void_ptr]*len(vals))
            tuple_type = ll.FunctionType(void_ptr, tuple_arg_types)
            tuple_fn = ll.Function(lj.module, tuple_type, "PyTuple_Pack")

            tuple_args = [ll.Constant(ll.IntType(32), len(wrapped_vals))]
            tuple_args.extend(wrapped_vals)

            final_val = lj.builder.call(tuple_fn, tuple_args)

        return final_val

    def _convert_expr(self, lj, expr):
        try:
            # Match CSE return data structure.
            if len(expr) == 2:
                tmp_exprs = expr[0]
                final_exprs = expr[1]
                if len(final_exprs) != 1 and self.signature.ret_type == ctypes.c_double:
                    raise NotImplementedError("Return of multiple expressions not supported for this callback")
                for name, e in tmp_exprs:
                    val = lj._print(e)
                    lj._add_tmp_var(name, val)
        except TypeError:
            final_exprs = [expr]

        vals = [lj._print(e) for e in final_exprs]

        return vals

    def _compile_function(self, strmod):
        llmod = llvm.parse_assembly(strmod)

        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 2
        pass_manager = llvm.create_module_pass_manager()
        pmb.populate(pass_manager)

        pass_manager.run(llmod)

        target_machine = \
            llvm.Target.from_default_triple().create_target_machine()
        exe_eng = llvm.create_mcjit_compiler(llmod, target_machine)
        exe_eng.finalize_object()
        exe_engines.append(exe_eng)

        if False:
            print("Assembly")
            print(target_machine.emit_assembly(llmod))

        fptr = exe_eng.get_function_address(self.link_name)

        return fptr


class LLVMJitCodeCallback(LLVMJitCode):
    def __init__(self, signature):
        super().__init__(signature)

    def _create_param_dict(self, func_args):
        for i, a in enumerate(func_args):
            if isinstance(a, IndexedBase):
                self.param_dict[a] = (self.fn.args[i], i)
                self.fn.args[i].name = str(a)
            else:
                self.param_dict[a] = (self.fn.args[self.signature.input_arg],
                                      i)

    def _create_function(self, expr):
        """Create function body and return LLVM IR"""
        bb_entry = self.fn.append_basic_block('entry')
        builder = ll.IRBuilder(bb_entry)

        lj = LLVMJitCallbackPrinter(self.module, builder, self.fn,
                                    func_arg_map=self.param_dict)

        ret = self._convert_expr(lj, expr)

        if self.signature.ret_arg:
            output_fp_ptr = builder.bitcast(self.fn.args[self.signature.ret_arg],
                                            ll.PointerType(self.fp_type))
            for i, val in enumerate(ret):
                index = ll.Constant(ll.IntType(32), i)
                output_array_ptr = builder.gep(output_fp_ptr, [index])
                builder.store(val, output_array_ptr)
            builder.ret(ll.Constant(ll.IntType(32), 0))  # return success
        else:
            lj.builder.ret(self._wrap_return(lj, ret))

        strmod = str(self.module)
        return strmod


class CodeSignature:
    def __init__(self, ret_type):
        self.ret_type = ret_type
        self.arg_ctypes = []

        # Input argument array element index
        self.input_arg = 0

        # For the case output value is referenced through a parameter rather
        # than the return value
        self.ret_arg = None


def _llvm_jit_code(args, expr, signature, callback_type):
    """Create a native code function from a SymPy expression"""
    if callback_type is None:
        jit = LLVMJitCode(signature)
    else:
        jit = LLVMJitCodeCallback(signature)

    jit._create_args(args)
    jit._create_function_base()
    jit._create_param_dict(args)
    strmod = jit._create_function(expr)
    if False:
        print("LLVM IR")
        print(strmod)
    fptr = jit._compile_function(strmod)
    return fptr


@doctest_depends_on(modules=('llvmlite', 'scipy'))
def llvm_callable(args, expr, callback_type=None):
    '''Compile function from a SymPy expression

    Expressions are evaluated using double precision arithmetic.
    Some single argument math functions (exp, sin, cos, etc.) are supported
    in expressions.

    Parameters
    ==========

    args : List of Symbol
        Arguments to the generated function.  Usually the free symbols in
        the expression.  Currently each one is assumed to convert to
        a double precision scalar.
    expr : Expr, or (Replacements, Expr) as returned from 'cse'
        Expression to compile.
    callback_type : string
        Create function with signature appropriate to use as a callback.
        Currently supported:
           'scipy.integrate'
           'scipy.integrate.test'
           'cubature'

    Returns
    =======

    Compiled function that can evaluate the expression.

    Examples
    ========

    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy.abc import a
    >>> e = a*a + a + 1
    >>> e1 = jit.llvm_callable([a], e)
    >>> e.subs(a, 1.1)   # Evaluate via substitution
    3.31000000000000
    >>> e1(1.1)  # Evaluate using JIT-compiled code
    3.3100000000000005


    Callbacks for integration functions can be JIT compiled.

    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy.abc import a
    >>> from sympy import integrate
    >>> from scipy.integrate import quad
    >>> e = a*a
    >>> e1 = jit.llvm_callable([a], e, callback_type='scipy.integrate')
    >>> integrate(e, (a, 0.0, 2.0))
    2.66666666666667
    >>> quad(e1, 0.0, 2.0)[0]
    2.66666666666667

    The 'cubature' callback is for the Python wrapper around the
    cubature package ( https://github.com/saullocastro/cubature )
    and ( http://ab-initio.mit.edu/wiki/index.php/Cubature )

    There are two signatures for the SciPy integration callbacks.
    The first ('scipy.integrate') is the function to be passed to the
    integration routine, and will pass the signature checks.
    The second ('scipy.integrate.test') is only useful for directly calling
    the function using ctypes variables. It will not pass the signature checks
    for scipy.integrate.

    The return value from the cse module can also be compiled.  This
    can improve the performance of the compiled function.  If multiple
    expressions are given to cse, the compiled function returns a tuple.
    The 'cubature' callback handles multiple expressions (set `fdim`
    to match in the integration call.)

    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy import cse
    >>> from sympy.abc import x,y
    >>> e1 = x*x + y*y
    >>> e2 = 4*(x*x + y*y) + 8.0
    >>> after_cse = cse([e1,e2])
    >>> after_cse
    ([(x0, x**2), (x1, y**2)], [x0 + x1, 4*x0 + 4*x1 + 8.0])
    >>> j1 = jit.llvm_callable([x,y], after_cse)
    >>> j1(1.0, 2.0)
    (5.0, 28.0)
    '''

    if not llvmlite:
        raise ImportError("llvmlite is required for llvmjitcode")

    signature = CodeSignature(ctypes.py_object)

    arg_ctypes = []
    if callback_type is None:
        for _ in args:
            arg_ctype = ctypes.c_double
            arg_ctypes.append(arg_ctype)
    elif callback_type in ('scipy.integrate', 'scipy.integrate.test'):
        signature.ret_type = ctypes.c_double
        arg_ctypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        arg_ctypes_formal = [ctypes.c_int, ctypes.c_double]
        signature.input_arg = 1
    elif callback_type == 'cubature':
        arg_ctypes = [ctypes.c_int,
                      ctypes.POINTER(ctypes.c_double),
                      ctypes.c_void_p,
                      ctypes.c_int,
                      ctypes.POINTER(ctypes.c_double)
                      ]
        signature.ret_type = ctypes.c_int
        signature.input_arg = 1
        signature.ret_arg = 4
    else:
        raise ValueError("Unknown callback type: %s" % callback_type)

    signature.arg_ctypes = arg_ctypes

    fptr = _llvm_jit_code(args, expr, signature, callback_type)

    if callback_type and callback_type == 'scipy.integrate':
        arg_ctypes = arg_ctypes_formal

    # PYFUNCTYPE holds the GIL which is needed to prevent a segfault when
    # calling PyFloat_FromDouble on Python 3.10. Probably it is better to use
    # ctypes.c_double when returning a float rather than using ctypes.py_object
    # and returning a PyFloat from inside the jitted function (i.e. let ctypes
    # handle the conversion from double to PyFloat).
    if signature.ret_type == ctypes.py_object:
        FUNCTYPE = ctypes.PYFUNCTYPE
    else:
        FUNCTYPE = ctypes.CFUNCTYPE

    cfunc = FUNCTYPE(signature.ret_type, *arg_ctypes)(fptr)
    return cfunc
<|endoftext|>"""
Python code printers

This module contains Python code printers for plain Python as well as NumPy & SciPy enabled code.
"""
from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter

_kw = {
    'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif',
    'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
    'is', 'lambda', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
    'with', 'yield', 'None', 'False', 'nonlocal', 'True'
}

_known_functions = {
    'Abs': 'abs',
    'Min': 'min',
    'Max': 'max',
}
_known_functions_math = {
    'acos': 'acos',
    'acosh': 'acosh',
    'asin': 'asin',
    'asinh': 'asinh',
    'atan': 'atan',
    'atan2': 'atan2',
    'atanh': 'atanh',
    'ceiling': 'ceil',
    'cos': 'cos',
    'cosh': 'cosh',
    'erf': 'erf',
    'erfc': 'erfc',
    'exp': 'exp',
    'expm1': 'expm1',
    'factorial': 'factorial',
    'floor': 'floor',
    'gamma': 'gamma',
    'hypot': 'hypot',
    'isinf': 'isinf',
    'isnan': 'isnan',
    'loggamma': 'lgamma',
    'log': 'log',
    'ln': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'sin': 'sin',
    'sinh': 'sinh',
    'Sqrt': 'sqrt',
    'tan': 'tan',
    'tanh': 'tanh'
}  # Not used from ``math``: [copysign isclose isfinite isinf ldexp frexp pow modf
# radians trunc fmod fsum gcd degrees fabs]
_known_constants_math = {
    'Exp1': 'e',
    'Pi': 'pi',
    'E': 'e',
    'Infinity': 'inf',
    'NaN': 'nan',
    'ComplexInfinity': 'nan'
}

def _print_known_func(self, expr):
    known = self.known_functions[expr.__class__.__name__]
    return '{name}({args})'.format(name=self._module_format(known),
                                   args=', '.join((self._print(arg) for arg in expr.args)))


def _print_known_const(self, expr):
    known = self.known_constants[expr.__class__.__name__]
    return self._module_format(known)


class AbstractPythonCodePrinter(CodePrinter):
    printmethod = "_pythoncode"
    language = "Python"
    reserved_words = _kw
    modules = None  # initialized to a set in __init__
    tab = '    '
    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'math.' + v) for k, v in _known_functions_math.items()]
    ))
    _kc = {k: 'math.'+v for k, v in _known_constants_math.items()}
    _operators = {'and': 'and', 'or': 'or', 'not': 'not'}
    _default_settings = dict(
        CodePrinter._default_settings,
        user_functions={},
        precision=17,
        inline=True,
        fully_qualified_modules=True,
        contract=False,
        standard='python3',
    )

    def __init__(self, settings=None):
        super().__init__(settings)

        # Python standard handler
        std = self._settings['standard']
        if std is None:
            import sys
            std = 'python{}'.format(sys.version_info.major)
        if std != 'python3':
            raise ValueError('Only Python 3 is supported.')
        self.standard = std

        self.module_imports = defaultdict(set)

        # Known functions and constants handler
        self.known_functions = dict(self._kf, **(settings or {}).get(
            'user_functions', {}))
        self.known_constants = dict(self._kc, **(settings or {}).get(
            'user_constants', {}))

    def _declare_number_const(self, name, value):
        return "%s = %s" % (name, value)

    def _module_format(self, fqn, register=True):
        parts = fqn.split('.')
        if register and len(parts) > 1:
            self.module_imports['.'.join(parts[:-1])].add(parts[-1])

        if self._settings['fully_qualified_modules']:
            return fqn
        else:
            return fqn.split('(')[0].split('[')[0].split('.')[-1]

    def _format_code(self, lines):
        return lines

    def _get_statement(self, codestring):
        return "{}".format(codestring)

    def _get_comment(self, text):
        return "  # {}".format(text)

    def _expand_fold_binary_op(self, op, args):
        """
        This method expands a fold on binary operations.

        ``functools.reduce`` is an example of a folded operation.

        For example, the expression

        `A + B + C + D`

        is folded into

        `((A + B) + C) + D`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_fold_binary_op(op, args[:-1]),
                self._print(args[-1]),
            )

    def _expand_reduce_binary_op(self, op, args):
        """
        This method expands a reduction on binary operations.

        Notice: this is NOT the same as ``functools.reduce``.

        For example, the expression

        `A + B + C + D`

        is reduced into:

        `(A + B) + (C + D)`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            N = len(args)
            Nhalf = N // 2
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_reduce_binary_op(args[:Nhalf]),
                self._expand_reduce_binary_op(args[Nhalf:]),
            )

    def _print_NaN(self, expr):
        return "float('nan')"

    def _print_Infinity(self, expr):
        return "float('inf')"

    def _print_NegativeInfinity(self, expr):
        return "float('-inf')"

    def _print_ComplexInfinity(self, expr):
        return self._print_NaN(expr)

    def _print_Mod(self, expr):
        PREC = precedence(expr)
        return ('{} % {}'.format(*(self.parenthesize(x, PREC) for x in expr.args)))

    def _print_Piecewise(self, expr):
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append('(')
            result.append('(')
            result.append(self._print(e))
            result.append(')')
            result.append(' if ')
            result.append(self._print(c))
            result.append(' else ')
            i += 1
        result = result[:-1]
        if result[-1] == 'True':
            result = result[:-2]
            result.append(')')
        else:
            result.append(' else None)')
        return ''.join(result)

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
            return '({lhs} {op} {rhs})'.format(op=expr.rel_op, lhs=lhs, rhs=rhs)
        return super()._print_Relational(expr)

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_Sum(self, expr):
        loops = (
            'for {i} in range({a}, {b}+1)'.format(
                i=self._print(i),
                a=self._print(a),
                b=self._print(b))
            for i, a, b in expr.limits[::-1])
        return '(builtins.sum({function} {loops}))'.format(
            function=self._print(expr.function),
            loops=' '.join(loops))

    def _print_ImaginaryUnit(self, expr):
        return '1j'

    def _print_KroneckerDelta(self, expr):
        a, b = expr.args

        return '(1 if {a} == {b} else 0)'.format(
            a = self._print(a),
            b = self._print(b)
        )

    def _print_MatrixBase(self, expr):
        name = expr.__class__.__name__
        func = self.known_functions.get(name, name)
        return "%s(%s)" % (func, self._print(expr.tolist()))

    _print_SparseRepMatrix = \
        _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        lambda self, expr: self._print_MatrixBase(expr)

    def _indent_codestring(self, codestring):
        return '\n'.join([self.tab + line for line in codestring.split('\n')])

    def _print_FunctionDefinition(self, fd):
        body = '\n'.join((self._print(arg) for arg in fd.body))
        return "def {name}({parameters}):\n{body}".format(
            name=self._print(fd.name),
            parameters=', '.join([self._print(var.symbol) for var in fd.parameters]),
            body=self._indent_codestring(body)
        )

    def _print_While(self, whl):
        body = '\n'.join((self._print(arg) for arg in whl.body))
        return "while {cond}:\n{body}".format(
            cond=self._print(whl.condition),
            body=self._indent_codestring(body)
        )

    def _print_Declaration(self, decl):
        return '%s = %s' % (
            self._print(decl.variable.symbol),
            self._print(decl.variable.value)
        )

    def _print_BreakToken(self, bt):
        return 'break'

    def _print_Return(self, ret):
        arg, = ret.args
        return 'return %s' % self._print(arg)

    def _print_Raise(self, rs):
        arg, = rs.args
        return 'raise %s' % self._print(arg)

    def _print_RuntimeError_(self, re):
        message, = re.args
        return "RuntimeError(%s)" % self._print(message)

    def _print_Print(self, prnt):
        print_args = ', '.join((self._print(arg) for arg in prnt.print_args))
        from sympy.codegen.ast import none
        if prnt.format_string != none:
            print_args = '{} % ({}), end=""'.format(
                self._print(prnt.format_string),
                print_args
            )
        if prnt.file != None: # Must be '!= None', cannot be 'is not None'
            print_args += ', file=%s' % self._print(prnt.file)
        return 'print(%s)' % print_args

    def _print_Stream(self, strm):
        if str(strm.name) == 'stdout':
            return self._module_format('sys.stdout')
        elif str(strm.name) == 'stderr':
            return self._module_format('sys.stderr')
        else:
            return self._print(strm.name)

    def _print_NoneToken(self, arg):
        return 'None'

    def _hprint_Pow(self, expr, rational=False, sqrt='math.sqrt'):
        """Printing helper function for ``Pow``

        Notes
        =====

        This preprocesses the ``sqrt`` as math formatter and prints division

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.printing.pycode import PythonCodePrinter
        >>> from sympy.abc import x

        Python code printer automatically looks up ``math.sqrt``.

        >>> printer = PythonCodePrinter()
        >>> printer._hprint_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._hprint_Pow(sqrt(x), rational=False)
        'math.sqrt(x)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=False)
        '1/math.sqrt(x)'
        >>> printer._hprint_Pow(1/x, rational=False)
        '1/x'
        >>> printer._hprint_Pow(1/x, rational=True)
        'x**(-1)'

        Using sqrt from numpy or mpmath

        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')
        'numpy.sqrt(x)'
        >>> printer._hprint_Pow(sqrt(x), sqrt='mpmath.sqrt')
        'mpmath.sqrt(x)'

        See Also
        ========

        sympy.printing.str.StrPrinter._print_Pow
        """
        PREC = precedence(expr)

        if expr.exp == S.Half and not rational:
            func = self._module_format(sqrt)
            arg = self._print(expr.base)
            return '{func}({arg})'.format(func=func, arg=arg)

        if expr.is_commutative and not rational:
            if -expr.exp is S.Half:
                func = self._module_format(sqrt)
                num = self._print(S.One)
                arg = self._print(expr.base)
                return f"{num}/{func}({arg})"
            if expr.exp is S.NegativeOne:
                num = self._print(S.One)
                arg = self.parenthesize(expr.base, PREC, strict=False)
                return f"{num}/{arg}"


        base_str = self.parenthesize(expr.base, PREC, strict=False)
        exp_str = self.parenthesize(expr.exp, PREC, strict=False)
        return "{}**{}".format(base_str, exp_str)


class ArrayPrinter:

    def _arrayify(self, indexed):
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        try:
            return convert_indexed_to_array(indexed)
        except Exception:
            return indexed

    def _get_einsum_string(self, subranks, contraction_indices):
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ""
        counter = 0
        d = {j: min(i) for i in contraction_indices for j in i}
        indices = []
        for rank_arg in subranks:
            lindices = []
            for i in range(rank_arg):
                if counter in d:
                    lindices.append(d[counter])
                else:
                    lindices.append(counter)
                counter += 1
            indices.append(lindices)
        mapping = {}
        letters_free = []
        letters_dum = []
        for i in indices:
            for j in i:
                if j not in mapping:
                    l = next(letters)
                    mapping[j] = l
                else:
                    l = mapping[j]
                contraction_string += l
                if j in d:
                    if l not in letters_dum:
                        letters_dum.append(l)
                else:
                    letters_free.append(l)
            contraction_string += ","
        contraction_string = contraction_string[:-1]
        return contraction_string, letters_free, letters_dum

    def _get_letter_generator_for_einsum(self):
        for i in range(97, 123):
            yield chr(i)
        for i in range(65, 91):
            yield chr(i)
        raise ValueError("out of letters")

    def _print_ArrayTensorProduct(self, expr):
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ",".join(["".join([next(letters) for j in range(i)]) for i in expr.subranks])
        return '%s("%s", %s)' % (
                self._module_format(self._module + "." + self._einsum),
                contraction_string,
                ", ".join([self._print(arg) for arg in expr.args])
        )

    def _print_ArrayContraction(self, expr):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        base = expr.expr
        contraction_indices = expr.contraction_indices

        if isinstance(base, ArrayTensorProduct):
            elems = ",".join(["%s" % (self._print(arg)) for arg in base.args])
            ranks = base.subranks
        else:
            elems = self._print(base)
            ranks = [len(base.shape)]

        contraction_string, letters_free, letters_dum = self._get_einsum_string(ranks, contraction_indices)

        if not contraction_indices:
            return self._print(base)
        if isinstance(base, ArrayTensorProduct):
            elems = ",".join(["%s" % (self._print(arg)) for arg in base.args])
        else:
            elems = self._print(base)
        return "%s(\"%s\", %s)" % (
            self._module_format(self._module + "." + self._einsum),
            "{}->{}".format(contraction_string, "".join(sorted(letters_free))),
            elems,
        )

    def _print_ArrayDiagonal(self, expr):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        diagonal_indices = list(expr.diagonal_indices)
        if isinstance(expr.expr, ArrayTensorProduct):
            subranks = expr.expr.subranks
            elems = expr.expr.args
        else:
            subranks = expr.subranks
            elems = [expr.expr]
        diagonal_string, letters_free, letters_dum = self._get_einsum_string(subranks, diagonal_indices)
        elems = [self._print(i) for i in elems]
        return '%s("%s", %s)' % (
            self._module_format(self._module + "." + self._einsum),
            "{}->{}".format(diagonal_string, "".join(letters_free+letters_dum)),
            ", ".join(elems)
        )

    def _print_PermuteDims(self, expr):
        return "%s(%s, %s)" % (
            self._module_format(self._module + "." + self._transpose),
            self._print(expr.expr),
            self._print(expr.permutation.array_form),
        )

    def _print_ArrayAdd(self, expr):
        return self._expand_fold_binary_op(self._module + "." + self._add, expr.args)

    def _print_OneArray(self, expr):
        return "%s((%s,))" % (
            self._module_format(self._module+ "." + self._ones),
            ','.join(map(self._print,expr.args))
        )

    def _print_ZeroArray(self, expr):
        return "%s((%s,))" % (
            self._module_format(self._module+ "." + self._zeros),
            ','.join(map(self._print,expr.args))
        )

    def _print_Assignment(self, expr):
        #XXX: maybe this needs to happen at a higher level e.g. at _print or
        #doprint?
        lhs = self._print(self._arrayify(expr.lhs))
        rhs = self._print(self._arrayify(expr.rhs))
        return "%s = %s" % ( lhs, rhs )

    def _print_IndexedBase(self, expr):
        return self._print_ArraySymbol(expr)


class PythonCodePrinter(AbstractPythonCodePrinter):

    def _print_sign(self, e):
        return '(0.0 if {e} == 0 else {f}(1, {e}))'.format(
            f=self._module_format('math.copysign'), e=self._print(e.args[0]))

    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + ' ' + self.parenthesize(expr.args[0], PREC)

    def _print_IndexedBase(self, expr):
        return expr.name

    def _print_Indexed(self, expr):
        base = expr.args[0]
        index = expr.args[1:]
        return "{}[{}]".format(str(base), ", ".join([self._print(ind) for ind in index]))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational)

    def _print_Rational(self, expr):
        return '{}/{}'.format(expr.p, expr.q)

    def _print_Half(self, expr):
        return self._print_Rational(expr)

    def _print_frac(self, expr):
        return self._print_Mod(Mod(expr.args[0], 1))

    def _print_Symbol(self, expr):

        name = super()._print_Symbol(expr)

        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = ('This expression includes the symbol "{}" which is a '
                       'reserved keyword in this language.')
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        elif '{' in name:   # Remove curly braces from subscripted variables
            return name.replace('{', '').replace('}', '')
        else:
            return name

    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported


for k in PythonCodePrinter._kf:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_func)

for k in _known_constants_math:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_const)


def pycode(expr, **settings):
    """ Converts an expr to a string of Python code

    Parameters
    ==========

    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    standard : str or None, optional
        Only 'python3' (default) is supported.
        This parameter may be removed in the future.

    Examples
    ========

    >>> from sympy import pycode, tan, Symbol
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'

    """
    return PythonCodePrinter(settings).doprint(expr)


from itertools import chain
from sympy.printing.pycode import PythonCodePrinter

_known_functions_cmath = {
    'exp': 'exp',
    'sqrt': 'sqrt',
    'log': 'log',
    'cos': 'cos',
    'sin': 'sin',
    'tan': 'tan',
    'acos': 'acos',
    'asin': 'asin',
    'atan': 'atan',
    'cosh': 'cosh',
    'sinh': 'sinh',
    'tanh': 'tanh',
    'acosh': 'acosh',
    'asinh': 'asinh',
    'atanh': 'atanh',
}

_known_constants_cmath = {
    'Pi': 'pi',
    'E': 'e',
    'Infinity': 'inf',
    'NegativeInfinity': '-inf',
}

class CmathPrinter(PythonCodePrinter):
    """ Printer for Python's cmath module """
    printmethod = "_cmathcode"
    language = "Python with cmath"

    _kf = dict(chain(
        _known_functions_cmath.items()
    ))

    _kc = {k: 'cmath.' + v for k, v in _known_constants_cmath.items()}

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='cmath.sqrt')

    def _print_Float(self, e):
        return '{func}({val})'.format(func=self._module_format('cmath.mpf'), val=self._print(e))

    def _print_known_func(self, expr):
        func_name = expr.func.__name__
        if func_name in self._kf:
            return f"cmath.{self._kf[func_name]}({', '.join(map(self._print, expr.args))})"
        return super()._print_Function(expr)

    def _print_known_const(self, expr):
        return self._kc[expr.__class__.__name__]

    def _print_re(self, expr):
        """Prints `re(z)` as `z.real`"""
        return f"({self._print(expr.args[0])}).real"

    def _print_im(self, expr):
        """Prints `im(z)` as `z.imag`"""
        return f"({self._print(expr.args[0])}).imag"


for k in CmathPrinter._kf:
    setattr(CmathPrinter, '_print_%s' % k, CmathPrinter._print_known_func)

for k in _known_constants_cmath:
    setattr(CmathPrinter, '_print_%s' % k, CmathPrinter._print_known_const)


_not_in_mpmath = 'log1p log2'.split()
_in_mpmath = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_mpmath]
_known_functions_mpmath = dict(_in_mpmath, **{
    'beta': 'beta',
    'frac': 'frac',
    'fresnelc': 'fresnelc',
    'fresnels': 'fresnels',
    'sign': 'sign',
    'loggamma': 'loggamma',
    'hyper': 'hyper',
    'meijerg': 'meijerg',
    'besselj': 'besselj',
    'bessely': 'bessely',
    'besseli': 'besseli',
    'besselk': 'besselk',
})
_known_constants_mpmath = {
    'Exp1': 'e',
    'Pi': 'pi',
    'GoldenRatio': 'phi',
    'EulerGamma': 'euler',
    'Catalan': 'catalan',
    'NaN': 'nan',
    'Infinity': 'inf',
    'NegativeInfinity': 'ninf'
}


def _unpack_integral_limits(integral_expr):
    """ helper function for _print_Integral that
        - accepts an Integral expression
        - returns a tuple of
           - a list variables of integration
           - a list of tuples of the upper and lower limits of integration
    """
    integration_vars = []
    limits = []
    for integration_range in integral_expr.limits:
        if len(integration_range) == 3:
            integration_var, lower_limit, upper_limit = integration_range
        else:
            raise NotImplementedError("Only definite integrals are supported")
        integration_vars.append(integration_var)
        limits.append((lower_limit, upper_limit))
    return integration_vars, limits


class MpmathPrinter(PythonCodePrinter):
    """
    Lambda printer for mpmath which maintains precision for floats
    """
    printmethod = "_mpmathcode"

    language = "Python with mpmath"

    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'mpmath.' + v) for k, v in _known_functions_mpmath.items()]
    ))
    _kc = {k: 'mpmath.'+v for k, v in _known_constants_mpmath.items()}

    def _print_Float(self, e):
        # XXX: This does not handle setting mpmath.mp.dps. It is assumed that
        # the caller of the lambdified function will have set it to sufficient
        # precision to match the Floats in the expression.

        # Remove 'mpz' if gmpy is installed.
        args = str(tuple(map(int, e._mpf_)))
        return '{func}({args})'.format(func=self._module_format('mpmath.mpf'), args=args)


    def _print_Rational(self, e):
        return "{func}({p})/{func}({q})".format(
            func=self._module_format('mpmath.mpf'),
            q=self._print(e.q),
            p=self._print(e.p)
        )

    def _print_Half(self, e):
        return self._print_Rational(e)

    def _print_uppergamma(self, e):
        return "{}({}, {}, {})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]),
            self._module_format('mpmath.inf'))

    def _print_lowergamma(self, e):
        return "{}({}, 0, {})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]))

    def _print_log2(self, e):
        return '{0}({1})/{0}(2)'.format(
            self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_log1p(self, e):
        return '{}({})'.format(
            self._module_format('mpmath.log1p'), self._print(e.args[0]))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='mpmath.sqrt')

    def _print_Integral(self, e):
        integration_vars, limits = _unpack_integral_limits(e)

        return "{}(lambda {}: {}, {})".format(
                self._module_format("mpmath.quad"),
                ", ".join(map(self._print, integration_vars)),
                self._print(e.args[0]),
                ", ".join("(%s, %s)" % tuple(map(self._print, l)) for l in limits))


    def _print_Derivative_zeta(self, args, seq_orders):
        arg, = args
        deriv_order, = seq_orders
        return '{}({}, derivative={})'.format(
            self._module_format('mpmath.zeta'),
            self._print(arg), deriv_order
        )


for k in MpmathPrinter._kf:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_func)

for k in _known_constants_mpmath:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_const)


class SymPyPrinter(AbstractPythonCodePrinter):

    language = "Python with SymPy"

    _default_settings = dict(
        AbstractPythonCodePrinter._default_settings,
        strict=False   # any class name will per definition be what we target in SymPyPrinter.
    )

    def _print_Function(self, expr):
        mod = expr.func.__module__ or ''
        return '%s(%s)' % (self._module_format(mod + ('.' if mod else '') + expr.func.__name__),
                           ', '.join((self._print(arg) for arg in expr.args)))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='sympy.sqrt')
<|endoftext|>"""
Rust code printer

The `RustCodePrinter` converts SymPy expressions into Rust expressions.

A complete code generator, which uses `rust_code` extensively, can be found
in `sympy.utilities.codegen`. The `codegen` module can be used to generate
complete source code files.

"""

# Possible Improvement
#
# * make sure we follow Rust Style Guidelines_
# * make use of pattern matching
# * better support for reference
# * generate generic code and use trait to make sure they have specific methods
# * use crates_ to get more math support
#     - num_
#         + BigInt_, BigUint_
#         + Complex_
#         + Rational64_, Rational32_, BigRational_
#
# .. _crates: https://crates.io/
# .. _Guidelines: https://github.com/rust-lang/rust/tree/master/src/doc/style
# .. _num: http://rust-num.github.io/num/num/
# .. _BigInt: http://rust-num.github.io/num/num/bigint/struct.BigInt.html
# .. _BigUint: http://rust-num.github.io/num/num/bigint/struct.BigUint.html
# .. _Complex: http://rust-num.github.io/num/num/complex/struct.Complex.html
# .. _Rational32: http://rust-num.github.io/num/num/rational/type.Rational32.html
# .. _Rational64: http://rust-num.github.io/num/num/rational/type.Rational64.html
# .. _BigRational: http://rust-num.github.io/num/num/rational/type.BigRational.html

from __future__ import annotations
from functools import reduce
import operator
from typing import Any

from sympy.codegen.ast import (
    float32, float64, int32,
    real, integer,  bool_
)
from sympy.core import S, Rational, Float, Lambda
from sympy.core.expr import Expr
from sympy.core.numbers import equal_valued
from sympy.functions.elementary.integers import ceiling, floor
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import PRECEDENCE

# Rust's methods for integer and float can be found at here :
#
# * `Rust - Primitive Type f64 <https://doc.rust-lang.org/std/primitive.f64.html>`_
# * `Rust - Primitive Type i64 <https://doc.rust-lang.org/std/primitive.i64.html>`_
#
# Function Style :
#
# 1. args[0].func(args[1:]), method with arguments
# 2. args[0].func(), method without arguments
# 3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())
# 4. func(args), function with arguments

# dictionary mapping SymPy function to (argument_conditions, Rust_function).
# Used in RustCodePrinter._print_Function(self)

class float_floor(floor):
    """
    Same as `sympy.floor`, but mimics the Rust behavior of returning a float rather than an integer
    """
    def _eval_is_integer(self):
        return False

class float_ceiling(ceiling):
    """
    Same as `sympy.ceiling`, but mimics the Rust behavior of returning a float rather than an integer
    """
    def _eval_is_integer(self):
        return False


function_overrides = {
    "floor": (floor, float_floor),
    "ceiling": (ceiling, float_ceiling),
}

# f64 method in Rust
known_functions = {
    # "": "is_nan",
    # "": "is_infinite",
    # "": "is_finite",
    # "": "is_normal",
    # "": "classify",
    "float_floor": "floor",
    "float_ceiling": "ceil",
    # "": "round",
    # "": "trunc",
    # "": "fract",
    "Abs": "abs",
    # "": "signum",
    # "": "is_sign_positive",
    # "": "is_sign_negative",
    # "": "mul_add",
    "Pow": [(lambda base, exp: equal_valued(exp, -1), "recip", 2),   # 1.0/x
            (lambda base, exp: equal_valued(exp, 0.5), "sqrt", 2),   # x ** 0.5
            (lambda base, exp: equal_valued(exp, -0.5), "sqrt().recip", 2),   # 1/(x ** 0.5)
            (lambda base, exp: exp == Rational(1, 3), "cbrt", 2),    # x ** (1/3)
            (lambda base, exp: equal_valued(base, 2), "exp2", 3),    # 2 ** x
            (lambda base, exp: exp.is_integer, "powi", 1),           # x ** y, for i32
            (lambda base, exp: not exp.is_integer, "powf", 1)],      # x ** y, for f64
    "exp": [(lambda exp: True, "exp", 2)],   # e ** x
    "log": "ln",
    # "": "log",          # number.log(base)
    # "": "log2",
    # "": "log10",
    # "": "to_degrees",
    # "": "to_radians",
    "Max": "max",
    "Min": "min",
    # "": "hypot",        # (x**2 + y**2) ** 0.5
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    # "": "sin_cos",
    # "": "exp_m1",       # e ** x - 1
    # "": "ln_1p",        # ln(1 + x)
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "sqrt": "sqrt",  # To enable automatic rewrites
}

# i64 method in Rust
# known_functions_i64 = {
#     "": "min_value",
#     "": "max_value",
#     "": "from_str_radix",
#     "": "count_ones",
#     "": "count_zeros",
#     "": "leading_zeros",
#     "": "trainling_zeros",
#     "": "rotate_left",
#     "": "rotate_right",
#     "": "swap_bytes",
#     "": "from_be",
#     "": "from_le",
#     "": "to_be",    # to big endian
#     "": "to_le",    # to little endian
#     "": "checked_add",
#     "": "checked_sub",
#     "": "checked_mul",
#     "": "checked_div",
#     "": "checked_rem",
#     "": "checked_neg",
#     "": "checked_shl",
#     "": "checked_shr",
#     "": "checked_abs",
#     "": "saturating_add",
#     "": "saturating_sub",
#     "": "saturating_mul",
#     "": "wrapping_add",
#     "": "wrapping_sub",
#     "": "wrapping_mul",
#     "": "wrapping_div",
#     "": "wrapping_rem",
#     "": "wrapping_neg",
#     "": "wrapping_shl",
#     "": "wrapping_shr",
#     "": "wrapping_abs",
#     "": "overflowing_add",
#     "": "overflowing_sub",
#     "": "overflowing_mul",
#     "": "overflowing_div",
#     "": "overflowing_rem",
#     "": "overflowing_neg",
#     "": "overflowing_shl",
#     "": "overflowing_shr",
#     "": "overflowing_abs",
#     "Pow": "pow",
#     "Abs": "abs",
#     "sign": "signum",
#     "": "is_positive",
#     "": "is_negnative",
# }

# These are the core reserved words in the Rust language. Taken from:
# https://doc.rust-lang.org/reference/keywords.html

reserved_words = ['abstract',
                  'as',
                  'async',
                  'await',
                  'become',
                  'box',
                  'break',
                  'const',
                  'continue',
                  'crate',
                  'do',
                  'dyn',
                  'else',
                  'enum',
                  'extern',
                  'false',
                  'final',
                  'fn',
                  'for',
                  'gen',
                  'if',
                  'impl',
                  'in',
                  'let',
                  'loop',
                  'macro',
                  'match',
                  'mod',
                  'move',
                  'mut',
                  'override',
                  'priv',
                  'pub',
                  'ref',
                  'return',
                  'Self',
                  'self',
                  'static',
                  'struct',
                  'super',
                  'trait',
                  'true',
                  'try',
                  'type',
                  'typeof',
                  'unsafe',
                  'unsized',
                  'use',
                  'virtual',
                  'where',
                  'while',
                  'yield']


class TypeCast(Expr):
    """
    The type casting operator of the Rust language.
    """

    def __init__(self, expr, type_) -> None:
        super().__init__()
        self.explicit = expr.is_integer and type_ is not integer
        self._assumptions = expr._assumptions
        if self.explicit:
            setattr(self, 'precedence', PRECEDENCE["Func"] + 10)

    @property
    def expr(self):
        return self.args[0]

    @property
    def type_(self):
        return self.args[1]

    def sort_key(self, order=None):
        return self.args[0].sort_key(order=order)


class RustCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Rust code"""
    printmethod = "_rust_code"
    language = "Rust"

    type_aliases = {
        integer: int32,
        real: float64,
    }

    type_mappings = {
        int32: 'i32',
        float32: 'f32',
        float64: 'f64',
        bool_: 'bool'
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
    })

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)
        self.function_overrides = function_overrides

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// %s" % text

    def _declare_number_const(self, name, value):
        type_ = self.type_mappings[self.type_aliases[real]]
        return "const %s: %s = %s;" % (name, type_, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for %(var)s in %(start)s..%(end)s {"
        for i in indices:
            # Rust arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'var': self._print(i),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_caller_var(self, expr):
        if len(expr.args) > 1:
            # for something like `sin(x + y + z)`,
            # make sure we can get '(x + y + z).sin()'
            # instead of 'x + y + z.sin()'
            return '(' + self._print(expr) + ')'
        elif expr.is_number:
            return self._print(expr, _type=True)
        else:
            return self._print(expr)

    def _print_Function(self, expr):
        """
        basic function for printing `Function`

        Function Style :

        1. args[0].func(args[1:]), method with arguments
        2. args[0].func(), method without arguments
        3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())
        4. func(args), function with arguments
        """

        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            func = None
            style = 1
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for cond, func, style in cond_func:
                    if cond(*expr.args):
                        break
            if func is not None:
                if style == 1:
                    ret = "%(var)s.%(method)s(%(args)s)" % {
                        'var': self._print_caller_var(expr.args[0]),
                        'method': func,
                        'args': self.stringify(expr.args[1:], ", ") if len(expr.args) > 1 else ''
                    }
                elif style == 2:
                    ret = "%(var)s.%(method)s()" % {
                        'var': self._print_caller_var(expr.args[0]),
                        'method': func,
                    }
                elif style == 3:
                    ret = "%(var)s.%(method)s()" % {
                        'var': self._print_caller_var(expr.args[1]),
                        'method': func,
                    }
                else:
                    ret = "%(func)s(%(args)s)" % {
                        'func': func,
                        'args': self.stringify(expr.args, ", "),
                    }
                return ret
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            # inlined function
            return self._print(expr._imp_(*expr.args))
        else:
            return self._print_not_supported(expr)

    def _print_Mul(self, expr):
        contains_floats = any(arg.is_real and not arg.is_integer for arg in expr.args)
        if contains_floats:
            expr = reduce(operator.mul,(self._cast_to_float(arg) if arg != -1 else arg for arg in expr.args))

        return super()._print_Mul(expr)

    def _print_Add(self, expr, order=None):
        contains_floats = any(arg.is_real and not arg.is_integer for arg in expr.args)
        if contains_floats:
            expr = reduce(operator.add, (self._cast_to_float(arg) for arg in expr.args))

        return super()._print_Add(expr, order)

    def _print_Pow(self, expr):
        if expr.base.is_integer and not expr.exp.is_integer:
            expr = type(expr)(Float(expr.base), expr.exp)
            return self._print(expr)
        return self._print_Function(expr)

    def _print_TypeCast(self, expr):
        if not expr.explicit:
            return self._print(expr.expr)
        else:
            return self._print(expr.expr) + ' as %s' % self.type_mappings[self.type_aliases[expr.type_]]

    def _print_Float(self, expr, _type=False):
        ret = super()._print_Float(expr)
        if _type:
            return ret + '_%s' % self.type_mappings[self.type_aliases[real]]
        else:
            return ret

    def _print_Integer(self, expr, _type=False):
        ret = super()._print_Integer(expr)
        if _type:
            return ret + '_%s' % self.type_mappings[self.type_aliases[integer]]
        else:
            return ret

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        float_suffix = self.type_mappings[self.type_aliases[real]]
        return '%d_%s/%d.0' % (p, float_suffix, q)

    def _print_Relational(self, expr):
        if (expr.lhs.is_integer and not expr.rhs.is_integer) or (expr.rhs.is_integer and not expr.lhs.is_integer):
            lhs = self._cast_to_float(expr.lhs)
            rhs = self._cast_to_float(expr.rhs)
        else:
            lhs = expr.lhs
            rhs = expr.rhs
        lhs_code = self._print(lhs)
        rhs_code = self._print(rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

    def _print_Idx(self, expr):
        return expr.label.name

    def _print_Dummy(self, expr):
        return expr.name

    def _print_Exp1(self, expr, _type=False):
        return "E"

    def _print_Pi(self, expr, _type=False):
        return 'PI'

    def _print_Infinity(self, expr, _type=False):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr, _type=False):
        return 'NEG_INFINITY'

    def _print_BooleanTrue(self, expr, _type=False):
        return "true"

    def _print_BooleanFalse(self, expr, _type=False):
        return "false"

    def _print_bool(self, expr, _type=False):
        return str(expr).lower()

    def _print_NaN(self, expr, _type=False):
        return "NAN"

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

        for i, (e, c) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s) {" % self._print(c))
            elif i == len(expr.args) - 1 and c == True:
                lines[-1] += " else {"
            else:
                lines[-1] += " else if (%s) {" % self._print(c)
            code0 = self._print(e)
            lines.append(code0)
            lines.append("}")

        if self._settings['inline']:
            return " ".join(lines)
        else:
            return "\n".join(lines)

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixBase(self, A):
        if A.cols == 1:
            return "[%s]" % ", ".join(self._print(a) for a in A)
        else:
            raise ValueError("Full Matrix Support in Rust need Crates (https://crates.io/keywords/matrix).")

    def _print_SparseRepMatrix(self, mat):
        # do not allow sparse matrices to be made dense
        return self._print_not_supported(mat)

    def _print_MatrixElement(self, expr):
        return "%s[%s]" % (expr.parent,
                           expr.j + expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):

        name = super()._print_Symbol(expr)

        if expr in self._dereference:
            return '(*%s)' % name
        else:
            return name

    def _print_Assignment(self, expr):
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def _print_sign(self, expr):
        arg = self._print(expr.args[0])
        return "(if (%s == 0.0) { 0.0 } else { (%s).signum() })" % (arg, arg)

    def _cast_to_float(self, expr):
        if not expr.is_number:
            return TypeCast(expr, real)
        elif expr.is_integer:
            return Float(expr)
        return expr

    def _can_print(self, name):
        """ Check if function ``name`` is either a known function or has its own
            printing method. Used to check if rewriting is possible."""

        # since the whole point of function_overrides is to enable proper printing,
        # we presume they all are printable

        return name in self.known_functions or name in function_overrides or getattr(self, '_print_{}'.format(name), False)

    def _collect_functions(self, expr):
        functions = set()
        if isinstance(expr, Expr):
            if expr.is_Function:
                functions.add(expr.func)
            for arg in expr.args:
                functions = functions.union(self._collect_functions(arg))
        return functions

    def _rewrite_known_functions(self, expr):
        if not isinstance(expr, Expr):
            return expr

        expression_functions = self._collect_functions(expr)
        rewriteable_functions = {
            name: (target_f, required_fs)
            for name, (target_f, required_fs) in self._rewriteable_functions.items()
            if self._can_print(target_f)
            and all(self._can_print(f) for f in required_fs)
        }
        for func in expression_functions:
            target_f, _ = rewriteable_functions.get(func.__name__, (None, None))
            if target_f:
                expr = expr.rewrite(target_f)
        return expr

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "    "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

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
<|endoftext|>from __future__ import annotations
import math
from typing import Any

from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial


aesara = import_module('aesara')

if aesara:
    aes = aesara.scalar
    aet = aesara.tensor
    from aesara.tensor import nlinalg
    from aesara.tensor.elemwise import Elemwise
    from aesara.tensor.elemwise import DimShuffle

    # `true_divide` replaced `true_div` in Aesara 2.8.11 (released 2023) to
    # match NumPy
    # XXX: Remove this when not needed to support older versions.
    true_divide = getattr(aet, 'true_divide', None)
    if true_divide is None:
        true_divide = aet.true_div

    mapping = {
            sympy.Add: aet.add,
            sympy.Mul: aet.mul,
            sympy.Abs: aet.abs,
            sympy.sign: aet.sgn,
            sympy.ceiling: aet.ceil,
            sympy.floor: aet.floor,
            sympy.log: aet.log,
            sympy.exp: aet.exp,
            sympy.sqrt: aet.sqrt,
            sympy.cos: aet.cos,
            sympy.acos: aet.arccos,
            sympy.sin: aet.sin,
            sympy.asin: aet.arcsin,
            sympy.tan: aet.tan,
            sympy.atan: aet.arctan,
            sympy.atan2: aet.arctan2,
            sympy.cosh: aet.cosh,
            sympy.acosh: aet.arccosh,
            sympy.sinh: aet.sinh,
            sympy.asinh: aet.arcsinh,
            sympy.tanh: aet.tanh,
            sympy.atanh: aet.arctanh,
            sympy.re: aet.real,
            sympy.im: aet.imag,
            sympy.arg: aet.angle,
            sympy.erf: aet.erf,
            sympy.gamma: aet.gamma,
            sympy.loggamma: aet.gammaln,
            sympy.Pow: aet.pow,
            sympy.Eq: aet.eq,
            sympy.StrictGreaterThan: aet.gt,
            sympy.StrictLessThan: aet.lt,
            sympy.LessThan: aet.le,
            sympy.GreaterThan: aet.ge,
            sympy.And: aet.bitwise_and,  # bitwise
            sympy.Or: aet.bitwise_or,  # bitwise
            sympy.Not: aet.invert,  # bitwise
            sympy.Xor: aet.bitwise_xor,  # bitwise
            sympy.Max: aet.maximum,  # Sympy accept >2 inputs, Aesara only 2
            sympy.Min: aet.minimum,  # Sympy accept >2 inputs, Aesara only 2
            sympy.conjugate: aet.conj,
            sympy.core.numbers.ImaginaryUnit: lambda:aet.complex(0,1),
            # Matrices
            sympy.MatAdd: Elemwise(aes.add),
            sympy.HadamardProduct: Elemwise(aes.mul),
            sympy.Trace: nlinalg.trace,
            sympy.Determinant : nlinalg.det,
            sympy.Inverse: nlinalg.matrix_inverse,
            sympy.Transpose: DimShuffle((False, False), [1, 0]),
    }


class AesaraPrinter(Printer):
    """
    .. deprecated:: 1.14.
        The ``Aesara Code printing`` is deprecated.See its documentation for
        more information. See :ref:`deprecated-aesaraprinter` for details.

    Code printer which creates Aesara symbolic expression graphs.

    Parameters
    ==========

    cache : dict
        Cache dictionary to use. If None (default) will use
        the global cache. To create a printer which does not depend on or alter
        global state pass an empty dictionary. Note: the dictionary is not
        copied on initialization of the printer and will be updated in-place,
        so using the same dict object when creating multiple printers or making
        multiple calls to :func:`.aesara_code` or :func:`.aesara_function` means
        the cache is shared between all these applications.

    Attributes
    ==========

    cache : dict
        A cache of Aesara variables which have been created for SymPy
        symbol-like objects (e.g. :class:`sympy.core.symbol.Symbol` or
        :class:`sympy.matrices.expressions.MatrixSymbol`). This is used to
        ensure that all references to a given symbol in an expression (or
        multiple expressions) are printed as the same Aesara variable, which is
        created only once. Symbols are differentiated only by name and type. The
        format of the cache's contents should be considered opaque to the user.
    """
    printmethod = "_aesara"

    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop('cache', {})
        super().__init__(*args, **kwargs)

    def _get_key(self, s, name=None, dtype=None, broadcastable=None):
        """ Get the cache key for a SymPy object.

        Parameters
        ==========

        s : sympy.core.basic.Basic
            SymPy object to get key for.

        name : str
            Name of object, if it does not have a ``name`` attribute.
        """

        if name is None:
            name = s.name

        return (name, type(s), s.args, dtype, broadcastable)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        Get the Aesara variable for a SymPy symbol from the cache, or create it
        if it does not exist.
        """

        # Defaults
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)

        if key in self.cache:
            return self.cache[key]

        value = aet.tensor(name=name, dtype=dtype, shape=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        op = mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        # Integers already taken care of below, interpret as float
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        if not hasattr(aet, 'stacklists'):
            raise NotImplementedError(
               "Matrix translation not yet supported in this version of Aesara")

        return aet.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = aet.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = 1
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = aet.dot(result, children[0])
        else:
            raise NotImplementedError('''Only non-negative integer
           powers of matrices can be handled by Aesara at the moment''')
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        return aet.join(0, *[aet.join(1, *row) for row in blocks])


    def _print_slice(self, expr, **kwargs):
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        return math.pi

    def _print_Piecewise(self, expr, **kwargs):
        import numpy as np
        e, cond = expr.args[0].args  # First condition and corresponding value

        # Print conditional expression and value for first condition
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)

        # One condition only
        if len(expr.args) == 1:
            # Return value if condition else NaN
            return aet.switch(p_cond, p_e, np.nan)

        # Return value_1 if condition_1 else evaluate remaining conditions
        p_remaining = self._print(sympy.Piecewise(*expr.args[1:]), **kwargs)
        return aet.switch(p_cond, p_e, p_remaining)

    def _print_Rational(self, expr, **kwargs):
        return true_divide(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    def _print_Integer(self, expr, **kwargs):
        return expr.p

    def _print_factorial(self, expr, **kwargs):
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    def _print_Derivative(self, deriv, **kwargs):
        from aesara.gradient import Rop

        rv = self._print(deriv.expr, **kwargs)
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            rv = Rop(rv, var, aet.ones_like(var))
        return rv

    def emptyPrinter(self, expr):
        return expr

    def doprint(self, expr, dtypes=None, broadcastables=None):
        """ Convert a SymPy expression to a Aesara graph variable.

        The ``dtypes`` and ``broadcastables`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Aesara variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        SymPy symbols to the value of the corresponding argument to
        ``aesara.tensor.var.TensorVariable``.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Aesara.


        .. __: https://aesara.readthedocs.io/en/latest/reference/tensor/shapes.html#broadcasting

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            SymPy expression to print.

        dtypes : dict
            Mapping from SymPy symbols to Aesara datatypes to use when creating
            new Aesara variables for those symbols. Corresponds to the ``dtype``
            argument to ``aesara.tensor.var.TensorVariable``. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastables : dict
            Mapping from SymPy symbols to the value of the ``broadcastable``
            argument to ``aesara.tensor.var.TensorVariable`` to use when creating Aesara
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        aesara.graph.basic.Variable
            A variable corresponding to the expression's value in a Aesara
            symbolic expression graph.

        """
        if dtypes is None:
            dtypes = {}
        if broadcastables is None:
            broadcastables = {}

        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)


global_cache: dict[Any, Any] = {}


def aesara_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Aesara graph variable.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict
        Cached Aesara variables (see :class:`AesaraPrinter.cache
        <AesaraPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    Returns
    =======

    aesara.graph.basic.Variable
        A variable corresponding to the expression's value in a Aesara symbolic
        expression graph.

    """
    sympy_deprecation_warning(
        """
        The aesara_code function is deprecated.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-aesaraprinter',
    )

    if not aesara:
        raise ImportError("aesara is required for aesara_code")

    if cache is None:
        cache = global_cache

    return AesaraPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    r"""
    Get value of ``broadcastables`` argument to :func:`.aesara_code` from
    keyword arguments to :func:`.aesara_function`.

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
        :meth:`.AesaraPrinter.doprint`. If not None function will return this value unchanged.

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


def aesara_function(inputs, outputs, scalar=False, *,
        dim=None, dims=None, broadcastables=None, **kwargs):
    """
    Create a Aesara function from SymPy expressions.

    The inputs and outputs are converted to Aesara variables using
    :func:`.aesara_code` and then passed to ``aesara.function``.

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
        Python wrapper function around the Aesara function object.

    cache : dict
        Cached Aesara variables (see :class:`AesaraPrinter.cache
        <AesaraPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.AesaraPrinter.doprint`.

    dims : dict
        Alternative to ``broadcastables`` argument. Mapping from elements of
        ``inputs`` to integers indicating the dimension of their associated
        arrays/tensors. Overrides ``broadcastables`` argument if given.

    dim : int
        Another alternative to the ``broadcastables`` argument. Common number of
        dimensions to use for all arrays/tensors.
        ``aesara_function([x, y], [...], dim=2)`` is equivalent to using
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
        ``aesara.compile.function.types.Function`` or a Python wrapper
        function around one. In both cases, the returned value will have a
        ``aesara_function`` attribute which points to the return value of
        ``aesara.function``.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.printing.aesaracode import aesara_function

    A simple function with one input and one output:

    >>> f1 = aesara_function([x], [x**2 - 1], scalar=True)
    >>> f1(3)
    8.0

    A function with multiple inputs and one output:

    >>> f2 = aesara_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)
    >>> f2(3, 4, 2)
    5.0

    A function with multiple inputs and multiple outputs:

    >>> f3 = aesara_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
    >>> f3(2, 3)
    [13.0, -5.0]

    See also
    ========

    dim_handling

    """
    sympy_deprecation_warning(
        """
        The aesara_function function is deprecated.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-aesaraprinter',
    )

    if not aesara:
        raise ImportError("Aesara is required for aesara_function")

    # Pop off non-aesara keyword args
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})

    broadcastables = dim_handling(
        inputs, dim=dim, dims=dims, broadcastables=broadcastables,
    )

    # Print inputs/outputs
    code = partial(aesara_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs = list(map(code, inputs))
    toutputs = list(map(code, outputs))

    #fix constant expressions as variables
    toutputs = [output if isinstance(output, aesara.graph.basic.Variable) else aet.as_tensor_variable(output) for output in toutputs]

    if len(toutputs) == 1:
        toutputs = toutputs[0]

    # Compile aesara func
    func = aesara.function(tinputs, toutputs, **kwargs)

    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]

    # No wrapper required
    if not scalar or not any(is_0d):
        func.aesara_function = func
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
    wrapper.aesara_function = func
    return wrapper
<|endoftext|>"""
C code printer

The C89CodePrinter & C99CodePrinter converts single SymPy expressions into
single C expressions, using the functions defined in math.h where possible.

A complete code generator, which uses ccode extensively, can be found in
sympy.utilities.codegen. The codegen module can be used to generate complete
source code files that are compilable without further modifications.


"""

from __future__ import annotations
from typing import Any

from functools import wraps
from itertools import chain

from sympy.core import S
from sympy.core.numbers import equal_valued, Float
from sympy.codegen.ast import (
    Assignment, Pointer, Variable, Declaration, Type,
    real, complex_, integer, bool_, float32, float64, float80,
    complex64, complex128, intc, value_const, pointer_const,
    int8, int16, int32, int64, uint8, uint16, uint32, uint64, untyped,
    none
)
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401

# dictionary mapping SymPy function to (argument_conditions, C_function).
# Used in C89CodePrinter._print_Function(self)
known_functions_C89 = {
    "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "log10": "log10",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "floor": "floor",
    "ceiling": "ceil",
    "sqrt": "sqrt", # To enable automatic rewrites
}

known_functions_C99 = dict(known_functions_C89, **{
    'exp2': 'exp2',
    'expm1': 'expm1',
    'log2': 'log2',
    'log1p': 'log1p',
    'Cbrt': 'cbrt',
    'hypot': 'hypot',
    'fma': 'fma',
    'loggamma': 'lgamma',
    'erfc': 'erfc',
    'Max': 'fmax',
    'Min': 'fmin',
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "erf": "erf",
    "gamma": "tgamma",
})

# These are the core reserved words in the C language. Taken from:
# https://en.cppreference.com/w/c/keyword

reserved_words = [
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int',
    'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
    'struct', 'entry',  # never standardized, we'll leave it here anyway
    'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'
]

reserved_words_c99 = ['inline', 'restrict']

def get_math_macros():
    """ Returns a dictionary with math-related macros from math.h/cmath

    Note that these macros are not strictly required by the C/C++-standard.
    For MSVC they are enabled by defining "_USE_MATH_DEFINES" (preferably
    via a compilation flag).

    Returns
    =======

    Dictionary mapping SymPy expressions to strings (macro names)

    """
    from sympy.codegen.cfunctions import log2, Sqrt
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.miscellaneous import sqrt

    return {
        S.Exp1: 'M_E',
        log2(S.Exp1): 'M_LOG2E',
        1/log(2): 'M_LOG2E',
        log(2): 'M_LN2',
        log(10): 'M_LN10',
        S.Pi: 'M_PI',
        S.Pi/2: 'M_PI_2',
        S.Pi/4: 'M_PI_4',
        1/S.Pi: 'M_1_PI',
        2/S.Pi: 'M_2_PI',
        2/sqrt(S.Pi): 'M_2_SQRTPI',
        2/Sqrt(S.Pi): 'M_2_SQRTPI',
        sqrt(2): 'M_SQRT2',
        Sqrt(2): 'M_SQRT2',
        1/sqrt(2): 'M_SQRT1_2',
        1/Sqrt(2): 'M_SQRT1_2'
    }


def _as_macro_if_defined(meth):
    """ Decorator for printer methods

    When a Printer's method is decorated using this decorator the expressions printed
    will first be looked for in the attribute ``math_macros``, and if present it will
    print the macro name in ``math_macros`` followed by a type suffix for the type
    ``real``. e.g. printing ``sympy.pi`` would print ``M_PIl`` if real is mapped to float80.

    """
    @wraps(meth)
    def _meth_wrapper(self, expr, **kwargs):
        if expr in self.math_macros:
            return '%s%s' % (self.math_macros[expr], self._get_math_macro_suffix(real))
        else:
            return meth(self, expr, **kwargs)

    return _meth_wrapper


class C89CodePrinter(CodePrinter):
    """A printer to convert Python expressions to strings of C code"""
    printmethod = "_ccode"
    language = "C"
    standard = "C89"
    reserved_words = set(reserved_words)

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
        'error_on_reserved': False,
    })

    type_aliases = {
        real: float64,
        complex_: complex128,
        integer: intc
    }

    type_mappings: dict[Type, Any] = {
        real: 'double',
        intc: 'int',
        float32: 'float',
        float64: 'double',
        integer: 'int',
        bool_: 'bool',
        int8: 'int8_t',
        int16: 'int16_t',
        int32: 'int32_t',
        int64: 'int64_t',
        uint8: 'int8_t',
        uint16: 'int16_t',
        uint32: 'int32_t',
        uint64: 'int64_t',
    }

    type_headers = {
        bool_: {'stdbool.h'},
        int8: {'stdint.h'},
        int16: {'stdint.h'},
        int32: {'stdint.h'},
        int64: {'stdint.h'},
        uint8: {'stdint.h'},
        uint16: {'stdint.h'},
        uint32: {'stdint.h'},
        uint64: {'stdint.h'},
    }

    # Macros needed to be defined when using a Type
    type_macros: dict[Type, tuple[str, ...]] = {}

    type_func_suffixes = {
        float32: 'f',
        float64: '',
        float80: 'l'
    }

    type_literal_suffixes = {
        float32: 'F',
        float64: '',
        float80: 'L'
    }

    type_math_macro_suffixes = {
        float80: 'l'
    }

    math_macros = None

    _ns = ''  # namespace, C++ uses 'std::'
    # known_functions-dict to copy
    _kf: dict[str, Any] = known_functions_C89

    def __init__(self, settings=None):
        settings = settings or {}
        if self.math_macros is None:
            self.math_macros = settings.pop('math_macros', get_math_macros())
        self.type_aliases = dict(chain(self.type_aliases.items(),
                                       settings.pop('type_aliases', {}).items()))
        self.type_mappings = dict(chain(self.type_mappings.items(),
                                        settings.pop('type_mappings', {}).items()))
        self.type_headers = dict(chain(self.type_headers.items(),
                                       settings.pop('type_headers', {}).items()))
        self.type_macros = dict(chain(self.type_macros.items(),
                                       settings.pop('type_macros', {}).items()))
        self.type_func_suffixes = dict(chain(self.type_func_suffixes.items(),
                                        settings.pop('type_func_suffixes', {}).items()))
        self.type_literal_suffixes = dict(chain(self.type_literal_suffixes.items(),
                                        settings.pop('type_literal_suffixes', {}).items()))
        self.type_math_macro_suffixes = dict(chain(self.type_math_macro_suffixes.items(),
                                        settings.pop('type_math_macro_suffixes', {}).items()))
        super().__init__(settings)
        self.known_functions = dict(self._kf, **settings.get('user_functions', {}))
        self._dereference = set(settings.get('dereference', []))
        self.headers = set()
        self.libraries = set()
        self.macros = set()

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        """ Get code string as a statement - i.e. ending with a semicolon. """
        return codestring if codestring.endswith(';') else codestring + ';'

    def _get_comment(self, text):
        return "/* {} */".format(text)

    def _declare_number_const(self, name, value):
        type_ = self.type_aliases[real]
        var = Variable(name, type=type_, value=value.evalf(type_.decimal_dig), attrs={value_const})
        decl = Declaration(var)
        return self._get_statement(self._print(decl))

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    @_as_macro_if_defined
    def _print_Mul(self, expr, **kwargs):
        return super()._print_Mul(expr, **kwargs)

    @_as_macro_if_defined
    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)
        if equal_valued(expr.exp, -1):
            return '%s/%s' % (self._print_Float(Float(1.0)), self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        elif expr.exp == S.One/3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base),
                                   self._print(expr.exp))

    def _print_Mod(self, expr):
        num, den = expr.args
        if num.is_integer and den.is_integer:
            PREC = precedence(expr)
            snum, sden = [self.parenthesize(arg, PREC) for arg in expr.args]
            # % is remainder (same sign as numerator), not modulo (same sign as
            # denominator), in C. Hence, % only works as modulo if both numbers
            # have the same sign
            if (num.is_nonnegative and den.is_nonnegative or
                num.is_nonpositive and den.is_nonpositive):
                return f"{snum} % {sden}"
            return f"(({snum} % {sden}) + {sden}) % {sden}"
        # Not guaranteed integer
        return self._print_math_func(expr, known='fmod')

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        suffix = self._get_literal_suffix(real)
        return '%d.0%s/%d.0%s' % (p, suffix, q, suffix)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        offset = getattr(expr.base, 'offset', S.Zero)
        strides = getattr(expr.base, 'strides', None)
        indices = expr.indices

        if strides is None or isinstance(strides, str):
            dims = expr.shape
            shift = S.One
            temp = ()
            if strides == 'C' or strides is None:
                traversal = reversed(range(expr.rank))
                indices = indices[::-1]
            elif strides == 'F':
                traversal = range(expr.rank)

            for i in traversal:
                temp += (shift,)
                shift *= dims[i]
            strides = temp
        flat_index = sum(x[0]*x[1] for x in zip(indices, strides)) + offset
        return "%s[%s]" % (self._print(expr.base.label),
                           self._print(flat_index))

    @_as_macro_if_defined
    def _print_NumberSymbol(self, expr):
        return super()._print_NumberSymbol(expr)

    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'

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

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixElement(self, expr):
        return "{}[{}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
            strict=True), expr.j + expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if expr in self._settings['dereference']:
            return '(*{})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return ('for ({target} = {start}; {target} < {stop}; {target} += '
                '{step}) {{\n{body}\n}}').format(target=target, start=start,
                stop=stop, step=step, body=body)

    def _print_sign(self, func):
        return '((({0}) > 0) - (({0}) < 0))'.format(self._print(func.args[0]))

    def _print_Max(self, expr):
        if "Max" in self.known_functions:
            return self._print_Function(expr)
        def inner_print_max(args): # The more natural abstraction of creating
            if len(args) == 1:     # and printing smaller Max objects is slow
                return self._print(args[0]) # when there are many arguments.
            half = len(args) // 2
            return "((%(a)s > %(b)s) ? %(a)s : %(b)s)" % {
                'a': inner_print_max(args[:half]),
                'b': inner_print_max(args[half:])
            }
        return inner_print_max(expr.args)

    def _print_Min(self, expr):
        if "Min" in self.known_functions:
            return self._print_Function(expr)
        def inner_print_min(args): # The more natural abstraction of creating
            if len(args) == 1:     # and printing smaller Min objects is slow
                return self._print(args[0]) # when there are many arguments.
            half = len(args) // 2
            return "((%(a)s < %(b)s) ? %(a)s : %(b)s)" % {
                'a': inner_print_min(args[:half]),
                'b': inner_print_min(args[half:])
            }
        return inner_print_min(expr.args)

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

    def _get_func_suffix(self, type_):
        return self.type_func_suffixes[self.type_aliases.get(type_, type_)]

    def _get_literal_suffix(self, type_):
        return self.type_literal_suffixes[self.type_aliases.get(type_, type_)]

    def _get_math_macro_suffix(self, type_):
        alias = self.type_aliases.get(type_, type_)
        dflt = self.type_math_macro_suffixes.get(alias, '')
        return self.type_math_macro_suffixes.get(type_, dflt)

    def _print_Tuple(self, expr):
        return '{'+', '.join(self._print(e) for e in expr)+'}'

    _print_List = _print_Tuple

    def _print_Type(self, type_):
        self.headers.update(self.type_headers.get(type_, set()))
        self.macros.update(self.type_macros.get(type_, set()))
        return self._print(self.type_mappings.get(type_, type_.name))

    def _print_Declaration(self, decl):
        from sympy.codegen.cnodes import restrict
        var = decl.variable
        val = var.value
        if var.type == untyped:
            raise ValueError("C does not support untyped variables")

        if isinstance(var, Pointer):
            result = '{vc}{t} *{pc} {r}{s}'.format(
                vc='const ' if value_const in var.attrs else '',
                t=self._print(var.type),
                pc=' const' if pointer_const in var.attrs else '',
                r='restrict ' if restrict in var.attrs else '',
                s=self._print(var.symbol)
            )
        elif isinstance(var, Variable):
            result = '{vc}{t} {s}'.format(
                vc='const ' if value_const in var.attrs else '',
                t=self._print(var.type),
                s=self._print(var.symbol)
            )
        else:
            raise NotImplementedError("Unknown type of var: %s" % type(var))
        if val != None: # Must be "!= None", cannot be "is not None"
            result += ' = %s' % self._print(val)
        return result

    def _print_Float(self, flt):
        type_ = self.type_aliases.get(real, real)
        self.macros.update(self.type_macros.get(type_, set()))
        suffix = self._get_literal_suffix(type_)
        num = str(flt.evalf(type_.decimal_dig))
        if 'e' not in num and '.' not in num:
            num += '.0'
        num_parts = num.split('e')
        num_parts[0] = num_parts[0].rstrip('0')
        if num_parts[0].endswith('.'):
            num_parts[0] += '0'
        return 'e'.join(num_parts) + suffix

    @requires(headers={'stdbool.h'})
    def _print_BooleanTrue(self, expr):
        return 'true'

    @requires(headers={'stdbool.h'})
    def _print_BooleanFalse(self, expr):
        return 'false'

    def _print_Element(self, elem):
        if elem.strides == None: # Must be "== None", cannot be "is None"
            if elem.offset != None: # Must be "!= None", cannot be "is not None"
                raise ValueError("Expected strides when offset is given")
            idxs = ']['.join((self._print(arg) for arg in elem.indices))
        else:
            global_idx = sum(i*s for i, s in zip(elem.indices, elem.strides))
            if elem.offset != None: # Must be "!= None", cannot be "is not None"
                global_idx += elem.offset
            idxs = self._print(global_idx)

        return "{symb}[{idxs}]".format(
            symb=self._print(elem.symbol),
            idxs=idxs
        )

    def _print_CodeBlock(self, expr):
        """ Elements of code blocks printed as statements. """
        return '\n'.join([self._get_statement(self._print(i)) for i in expr.args])

    def _print_While(self, expr):
        return 'while ({condition}) {{\n{body}\n}}'.format(**expr.kwargs(
            apply=lambda arg: self._print(arg)))

    def _print_Scope(self, expr):
        return '{\n%s\n}' % self._print_CodeBlock(expr.body)

    @requires(headers={'stdio.h'})
    def _print_Print(self, expr):
        if expr.file == none:
            template = 'printf({fmt}, {pargs})'
        else:
            template = 'fprintf(%(out)s, {fmt}, {pargs})' % {
                'out': self._print(expr.file)
            }
        return template.format(
            fmt="%s\n" if expr.format_string == none else self._print(expr.format_string),
            pargs=', '.join((self._print(arg) for arg in expr.print_args))
        )

    def _print_Stream(self, strm):
        return strm.name

    def _print_FunctionPrototype(self, expr):
        pars = ', '.join((self._print(Declaration(arg)) for arg in expr.parameters))
        return "%s %s(%s)" % (
            tuple((self._print(arg) for arg in (expr.return_type, expr.name))) + (pars,)
        )

    def _print_FunctionDefinition(self, expr):
        return "%s%s" % (self._print_FunctionPrototype(expr),
                         self._print_Scope(expr))

    def _print_Return(self, expr):
        arg, = expr.args
        return 'return %s' % self._print(arg)

    def _print_CommaOperator(self, expr):
        return '(%s)' % ', '.join((self._print(arg) for arg in expr.args))

    def _print_Label(self, expr):
        if expr.body == none:
            return '%s:' % str(expr.name)
        if len(expr.body.args) == 1:
            return '%s:\n%s' % (str(expr.name), self._print_CodeBlock(expr.body))
        return '%s:\n{\n%s\n}' % (str(expr.name), self._print_CodeBlock(expr.body))

    def _print_goto(self, expr):
        return 'goto %s' % expr.label.name

    def _print_PreIncrement(self, expr):
        arg, = expr.args
        return '++(%s)' % self._print(arg)

    def _print_PostIncrement(self, expr):
        arg, = expr.args
        return '(%s)++' % self._print(arg)

    def _print_PreDecrement(self, expr):
        arg, = expr.args
        return '--(%s)' % self._print(arg)

    def _print_PostDecrement(self, expr):
        arg, = expr.args
        return '(%s)--' % self._print(arg)

    def _print_struct(self, expr):
        return "%(keyword)s %(name)s {\n%(lines)s}" % {
            "keyword": expr.__class__.__name__, "name": expr.name, "lines": ';\n'.join(
                [self._print(decl) for decl in expr.declarations] + [''])
        }

    def _print_BreakToken(self, _):
        return 'break'

    def _print_ContinueToken(self, _):
        return 'continue'

    _print_union = _print_struct

class C99CodePrinter(C89CodePrinter):
    standard = 'C99'
    reserved_words = set(reserved_words + reserved_words_c99)
    type_mappings=dict(chain(C89CodePrinter.type_mappings.items(), {
        complex64: 'float complex',
        complex128: 'double complex',
    }.items()))
    type_headers = dict(chain(C89CodePrinter.type_headers.items(), {
        complex64: {'complex.h'},
        complex128: {'complex.h'}
    }.items()))

    # known_functions-dict to copy
    _kf: dict[str, Any] = known_functions_C99

    # functions with versions with 'f' and 'l' suffixes:
    _prec_funcs = ('fabs fmod remainder remquo fma fmax fmin fdim nan exp exp2'
                   ' expm1 log log10 log2 log1p pow sqrt cbrt hypot sin cos tan'
                   ' asin acos atan atan2 sinh cosh tanh asinh acosh atanh erf'
                   ' erfc tgamma lgamma ceil floor trunc round nearbyint rint'
                   ' frexp ldexp modf scalbn ilogb logb nextafter copysign').split()

    def _print_Infinity(self, expr):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr):
        return '-INFINITY'

    def _print_NaN(self, expr):
        return 'NAN'

    # tgamma was already covered by 'known_functions' dict

    @requires(headers={'math.h'}, libraries={'m'})
    @_as_macro_if_defined
    def _print_math_func(self, expr, nest=False, known=None):
        if known is None:
            known = self.known_functions[expr.__class__.__name__]
        if not isinstance(known, str):
            for cb, name in known:
                if cb(*expr.args):
                    known = name
                    break
            else:
                raise ValueError("No matching printer")
        try:
            return known(self, *expr.args)
        except TypeError:
            suffix = self._get_func_suffix(real) if self._ns + known in self._prec_funcs else ''

        if nest:
            args = self._print(expr.args[0])
            if len(expr.args) > 1:
                paren_pile = ''
                for curr_arg in expr.args[1:-1]:
                    paren_pile += ')'
                    args += ', {ns}{name}{suffix}({next}'.format(
                        ns=self._ns,
                        name=known,
                        suffix=suffix,
                        next = self._print(curr_arg)
                    )
                args += ', %s%s' % (
                    self._print(expr.func(expr.args[-1])),
                    paren_pile
                )
        else:
            args = ', '.join((self._print(arg) for arg in expr.args))
        return '{ns}{name}{suffix}({args})'.format(
            ns=self._ns,
            name=known,
            suffix=suffix,
            args=args
        )

    def _print_Max(self, expr):
        return self._print_math_func(expr, nest=True)

    def _print_Min(self, expr):
        return self._print_math_func(expr, nest=True)

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (int %(var)s=%(start)s; %(var)s<%(end)s; %(var)s++){"  # C99
        for i in indices:
            # C arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'var': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines


for k in ('Abs Sqrt exp exp2 expm1 log log10 log2 log1p Cbrt hypot fma'
          ' loggamma sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh '
          'atanh erf erfc loggamma gamma ceiling floor').split():
    setattr(C99CodePrinter, '_print_%s' % k, C99CodePrinter._print_math_func)


class C11CodePrinter(C99CodePrinter):

    @requires(headers={'stdalign.h'})
    def _print_alignof(self, expr):
        arg, = expr.args
        return 'alignof(%s)' % self._print(arg)


c_code_printers = {
    'c89': C89CodePrinter,
    'c99': C99CodePrinter,
    'c11': C11CodePrinter
}
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer, Rational, Float
from sympy.printing.repr import srepr

__all__ = ['dotprint']

default_styles = (
    (Basic, {'color': 'blue', 'shape': 'ellipse'}),
    (Expr,  {'color': 'black'})
)

slotClasses = (Symbol, Integer, Rational, Float)
def purestr(x, with_args=False):
    """A string that follows ```obj = type(obj)(*obj.args)``` exactly.

    Parameters
    ==========

    with_args : boolean, optional
        If ``True``, there will be a second argument for the return
        value, which is a tuple containing ``purestr`` applied to each
        of the subnodes.

        If ``False``, there will not be a second argument for the
        return.

        Default is ``False``

    Examples
    ========

    >>> from sympy import Float, Symbol, MatrixSymbol
    >>> from sympy import Integer # noqa: F401
    >>> from sympy.core.symbol import Str # noqa: F401
    >>> from sympy.printing.dot import purestr

    Applying ``purestr`` for basic symbolic object:
    >>> code = purestr(Symbol('x'))
    >>> code
    "Symbol('x')"
    >>> eval(code) == Symbol('x')
    True

    For basic numeric object:
    >>> purestr(Float(2))
    "Float('2.0', precision=53)"

    For matrix symbol:
    >>> code = purestr(MatrixSymbol('x', 2, 2))
    >>> code
    "MatrixSymbol(Str('x'), Integer(2), Integer(2))"
    >>> eval(code) == MatrixSymbol('x', 2, 2)
    True

    With ``with_args=True``:
    >>> purestr(Float(2), with_args=True)
    ("Float('2.0', precision=53)", ())
    >>> purestr(MatrixSymbol('x', 2, 2), with_args=True)
    ("MatrixSymbol(Str('x'), Integer(2), Integer(2))",
     ("Str('x')", 'Integer(2)', 'Integer(2)'))
    """
    sargs = ()
    if not isinstance(x, Basic):
        rv = str(x)
    elif not x.args:
        rv = srepr(x)
    else:
        args = x.args
        sargs = tuple(map(purestr, args))
        rv = "%s(%s)"%(type(x).__name__, ', '.join(sargs))
    if with_args:
        rv = rv, sargs
    return rv


def styleof(expr, styles=default_styles):
    """ Merge style dictionaries in order

    Examples
    ========

    >>> from sympy import Symbol, Basic, Expr, S
    >>> from sympy.printing.dot import styleof
    >>> styles = [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
    ...           (Expr,  {'color': 'black'})]

    >>> styleof(Basic(S(1)), styles)
    {'color': 'blue', 'shape': 'ellipse'}

    >>> x = Symbol('x')
    >>> styleof(x + 1, styles)  # this is an Expr
    {'color': 'black', 'shape': 'ellipse'}
    """
    style = {}
    for typ, sty in styles:
        if isinstance(expr, typ):
            style.update(sty)
    return style


def attrprint(d, delimiter=', '):
    """ Print a dictionary of attributes

    Examples
    ========

    >>> from sympy.printing.dot import attrprint
    >>> print(attrprint({'color': 'blue', 'shape': 'ellipse'}))
    "color"="blue", "shape"="ellipse"
    """
    return delimiter.join('"%s"="%s"'%item for item in sorted(d.items()))


def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
    """ String defining a node

    Examples
    ========

    >>> from sympy.printing.dot import dotnode
    >>> from sympy.abc import x
    >>> print(dotnode(x))
    "Symbol('x')_()" ["color"="black", "label"="x", "shape"="ellipse"];
    """
    style = styleof(expr, styles)

    if isinstance(expr, Basic) and not expr.is_Atom:
        label = str(expr.__class__.__name__)
    else:
        label = labelfunc(expr)
    style['label'] = label
    expr_str = purestr(expr)
    if repeat:
        expr_str += '_%s' % str(pos)
    return '"%s" [%s];' % (expr_str, attrprint(style))


def dotedges(expr, atom=lambda x: not isinstance(x, Basic), pos=(), repeat=True):
    """ List of strings for all expr->expr.arg pairs

    See the docstring of dotprint for explanations of the options.

    Examples
    ========

    >>> from sympy.printing.dot import dotedges
    >>> from sympy.abc import x
    >>> for e in dotedges(x+2):
    ...     print(e)
    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
    """
    if atom(expr):
        return []
    else:
        expr_str, arg_strs = purestr(expr, with_args=True)
        if repeat:
            expr_str += '_%s' % str(pos)
            arg_strs = ['%s_%s' % (a, str(pos + (i,)))
                for i, a in enumerate(arg_strs)]
        return ['"%s" -> "%s";' % (expr_str, a) for a in arg_strs]

template = \
"""digraph{

# Graph style
%(graphstyle)s

#########
# Nodes #
#########

%(nodes)s

#########
# Edges #
#########

%(edges)s
}"""

_graphstyle = {'rankdir': 'TD', 'ordering': 'out'}

def dotprint(expr,
    styles=default_styles, atom=lambda x: not isinstance(x, Basic),
    maxdepth=None, repeat=True, labelfunc=str, **kwargs):
    """DOT description of a SymPy expression tree

    Parameters
    ==========

    styles : list of lists composed of (Class, mapping), optional
        Styles for different classes.

        The default is

        .. code-block:: python

            (
                (Basic, {'color': 'blue', 'shape': 'ellipse'}),
                (Expr,  {'color': 'black'})
            )

    atom : function, optional
        Function used to determine if an arg is an atom.

        A good choice is ``lambda x: not x.args``.

        The default is ``lambda x: not isinstance(x, Basic)``.

    maxdepth : integer, optional
        The maximum depth.

        The default is ``None``, meaning no limit.

    repeat : boolean, optional
        Whether to use different nodes for common subexpressions.

        The default is ``True``.

        For example, for ``x + x*y`` with ``repeat=True``, it will have
        two nodes for ``x``; with ``repeat=False``, it will have one
        node.

        .. warning::
            Even if a node appears twice in the same object like ``x`` in
            ``Pow(x, x)``, it will still only appear once.
            Hence, with ``repeat=False``, the number of arrows out of an
            object might not equal the number of args it has.

    labelfunc : function, optional
        A function to create a label for a given leaf node.

        The default is ``str``.

        Another good option is ``srepr``.

        For example with ``str``, the leaf nodes of ``x + 1`` are labeled,
        ``x`` and ``1``.  With ``srepr``, they are labeled ``Symbol('x')``
        and ``Integer(1)``.

    **kwargs : optional
        Additional keyword arguments are included as styles for the graph.

    Examples
    ========

    >>> from sympy import dotprint
    >>> from sympy.abc import x
    >>> print(dotprint(x+2)) # doctest: +NORMALIZE_WHITESPACE
    digraph{
    <BLANKLINE>
    # Graph style
    "ordering"="out"
    "rankdir"="TD"
    <BLANKLINE>
    #########
    # Nodes #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol('x'))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
    "Integer(2)_(0,)" ["color"="black", "label"="2", "shape"="ellipse"];
    "Symbol('x')_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
    <BLANKLINE>
    #########
    # Edges #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
    }

    """
    # repeat works by adding a signature tuple to the end of each node for its
    # position in the graph. For example, for expr = Add(x, Pow(x, 2)), the x in the
    # Pow will have the tuple (1, 0), meaning it is expr.args[1].args[0].
    graphstyle = _graphstyle.copy()
    graphstyle.update(kwargs)

    nodes = []
    edges = []
    def traverse(e, depth, pos=()):
        nodes.append(dotnode(e, styles, labelfunc=labelfunc, pos=pos, repeat=repeat))
        if maxdepth and depth >= maxdepth:
            return
        edges.extend(dotedges(e, atom=atom, pos=pos, repeat=repeat))
        [traverse(arg, depth+1, pos + (i,)) for i, arg in enumerate(e.args) if not atom(arg)]
    traverse(expr, 0)

    return template%{'graphstyle': attrprint(graphstyle, delimiter='\n'),
                     'nodes': '\n'.join(nodes),
                     'edges': '\n'.join(edges)}
<|endoftext|>"""
C++ code printer
"""

from itertools import chain
from sympy.codegen.ast import Type, none
from .codeprinter import requires
from .c import C89CodePrinter, C99CodePrinter

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import cxxcode # noqa:F401

# from https://en.cppreference.com/w/cpp/keyword
reserved = {
    'C++98': [
        'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break',
        'case', 'catch,', 'char', 'class', 'compl', 'const', 'const_cast',
        'continue', 'default', 'delete', 'do', 'double', 'dynamic_cast',
        'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float',
        'for', 'friend', 'goto', 'if', 'inline', 'int', 'long', 'mutable',
        'namespace', 'new', 'not', 'not_eq', 'operator', 'or', 'or_eq',
        'private', 'protected', 'public', 'register', 'reinterpret_cast',
        'return', 'short', 'signed', 'sizeof', 'static', 'static_cast',
        'struct', 'switch', 'template', 'this', 'throw', 'true', 'try',
        'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using',
        'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'
    ]
}

reserved['C++11'] = reserved['C++98'][:] + [
    'alignas', 'alignof', 'char16_t', 'char32_t', 'constexpr', 'decltype',
    'noexcept', 'nullptr', 'static_assert', 'thread_local'
]
reserved['C++17'] = reserved['C++11'][:]
reserved['C++17'].remove('register')
# TM TS: atomic_cancel, atomic_commit, atomic_noexcept, synchronized
# concepts TS: concept, requires
# module TS: import, module


_math_functions = {
    'C++98': {
        'Mod': 'fmod',
        'ceiling': 'ceil',
    },
    'C++11': {
        'gamma': 'tgamma',
    },
    'C++17': {
        'beta': 'beta',
        'Ei': 'expint',
        'zeta': 'riemann_zeta',
    }
}

# from https://en.cppreference.com/w/cpp/header/cmath
for k in ('Abs', 'exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 'tan',  # 'Pow'
          'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'floor'):
    _math_functions['C++98'][k] = k.lower()


for k in ('asinh', 'acosh', 'atanh', 'erf', 'erfc'):
    _math_functions['C++11'][k] = k.lower()


def _attach_print_method(cls, sympy_name, func_name):
    meth_name = '_print_%s' % sympy_name
    if hasattr(cls, meth_name):
        raise ValueError("Edit method (or subclass) instead of overwriting.")
    def _print_method(self, expr):
        return '{}{}({})'.format(self._ns, func_name, ', '.join(map(self._print, expr.args)))
    _print_method.__doc__ = "Prints code for %s" % k
    setattr(cls, meth_name, _print_method)


def _attach_print_methods(cls, cont):
    for sympy_name, cxx_name in cont[cls.standard].items():
        _attach_print_method(cls, sympy_name, cxx_name)


class _CXXCodePrinterBase:
    printmethod = "_cxxcode"
    language = 'C++'
    _ns = 'std::'  # namespace

    def __init__(self, settings=None):
        super().__init__(settings or {})

    @requires(headers={'algorithm'})
    def _print_Max(self, expr):
        from sympy.functions.elementary.miscellaneous import Max
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "%smax(%s, %s)" % (self._ns, self._print(expr.args[0]),
                                  self._print(Max(*expr.args[1:])))

    @requires(headers={'algorithm'})
    def _print_Min(self, expr):
        from sympy.functions.elementary.miscellaneous import Min
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "%smin(%s, %s)" % (self._ns, self._print(expr.args[0]),
                                  self._print(Min(*expr.args[1:])))

    def _print_using(self, expr):
        if expr.alias == none:
            return 'using %s' % expr.type
        else:
            raise ValueError("C++98 does not support type aliases")

    def _print_Raise(self, rs):
        arg, = rs.args
        return 'throw %s' % self._print(arg)

    @requires(headers={'stdexcept'})
    def _print_RuntimeError_(self, re):
        message, = re.args
        return "%sruntime_error(%s)" % (self._ns, self._print(message))


class CXX98CodePrinter(_CXXCodePrinterBase, C89CodePrinter):
    standard = 'C++98'
    reserved_words = set(reserved['C++98'])


# _attach_print_methods(CXX98CodePrinter, _math_functions)


class CXX11CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard = 'C++11'
    reserved_words = set(reserved['C++11'])
    type_mappings = dict(chain(
        CXX98CodePrinter.type_mappings.items(),
        {
            Type('int8'): ('int8_t', {'cstdint'}),
            Type('int16'): ('int16_t', {'cstdint'}),
            Type('int32'): ('int32_t', {'cstdint'}),
            Type('int64'): ('int64_t', {'cstdint'}),
            Type('uint8'): ('uint8_t', {'cstdint'}),
            Type('uint16'): ('uint16_t', {'cstdint'}),
            Type('uint32'): ('uint32_t', {'cstdint'}),
            Type('uint64'): ('uint64_t', {'cstdint'}),
            Type('complex64'): ('std::complex<float>', {'complex'}),
            Type('complex128'): ('std::complex<double>', {'complex'}),
            Type('bool'): ('bool', None),
        }.items()
    ))

    def _print_using(self, expr):
        if expr.alias == none:
            return super()._print_using(expr)
        else:
            return 'using %(alias)s = %(type)s' % expr.kwargs(apply=self._print)

# _attach_print_methods(CXX11CodePrinter, _math_functions)


class CXX17CodePrinter(_CXXCodePrinterBase, C99CodePrinter):
    standard = 'C++17'
    reserved_words = set(reserved['C++17'])

    _kf = dict(C99CodePrinter._kf, **_math_functions['C++17'])

    def _print_beta(self, expr):
        return self._print_math_func(expr)

    def _print_Ei(self, expr):
        return self._print_math_func(expr)

    def _print_zeta(self, expr):
        return self._print_math_func(expr)


# _attach_print_methods(CXX17CodePrinter, _math_functions)

cxx_code_printers = {
    'c++98': CXX98CodePrinter,
    'c++11': CXX11CodePrinter,
    'c++17': CXX17CodePrinter
}
<|endoftext|>"""A module providing information about the necessity of brackets"""


# Default precedence values for some basic types
PRECEDENCE = {
    "Lambda": 1,
    "Xor": 10,
    "Or": 20,
    "And": 30,
    "Relational": 35,
    "Add": 40,
    "Mul": 50,
    "Pow": 60,
    "Func": 70,
    "Not": 100,
    "Atom": 1000,
    "BitwiseOr": 36,
    "BitwiseXor": 37,
    "BitwiseAnd": 38
}

# A dictionary assigning precedence values to certain classes. These values are
# treated like they were inherited, so not every single class has to be named
# here.
# Do not use this with printers other than StrPrinter
PRECEDENCE_VALUES = {
    "Equivalent": PRECEDENCE["Xor"],
    "Xor": PRECEDENCE["Xor"],
    "Implies": PRECEDENCE["Xor"],
    "Or": PRECEDENCE["Or"],
    "And": PRECEDENCE["And"],
    "Add": PRECEDENCE["Add"],
    "Pow": PRECEDENCE["Pow"],
    "Relational": PRECEDENCE["Relational"],
    "Sub": PRECEDENCE["Add"],
    "Not": PRECEDENCE["Not"],
    "Function" : PRECEDENCE["Func"],
    "NegativeInfinity": PRECEDENCE["Add"],
    "MatAdd": PRECEDENCE["Add"],
    "MatPow": PRECEDENCE["Pow"],
    "MatrixSolve": PRECEDENCE["Mul"],
    "Mod": PRECEDENCE["Mul"],
    "TensAdd": PRECEDENCE["Add"],
    # As soon as `TensMul` is a subclass of `Mul`, remove this:
    "TensMul": PRECEDENCE["Mul"],
    "HadamardProduct": PRECEDENCE["Mul"],
    "HadamardPower": PRECEDENCE["Pow"],
    "KroneckerProduct": PRECEDENCE["Mul"],
    "Equality": PRECEDENCE["Mul"],
    "Unequality": PRECEDENCE["Mul"],
}

# Sometimes it's not enough to assign a fixed precedence value to a
# class. Then a function can be inserted in this dictionary that takes
# an instance of this class as argument and returns the appropriate
# precedence value.

# Precedence functions


def precedence_Mul(item):
    from sympy.core.function import Function
    if any(hasattr(arg, 'precedence') and isinstance(arg, Function) and
           arg.precedence < PRECEDENCE["Mul"] for arg in item.args):
        return PRECEDENCE["Mul"]

    if item.could_extract_minus_sign():
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


def precedence_Rational(item):
    if item.p < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]


def precedence_Integer(item):
    if item.p < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Atom"]


def precedence_Float(item):
    if item < 0:
        return PRECEDENCE["Add"]
    return PRECEDENCE["Atom"]


def precedence_PolyElement(item):
    if item.is_generator:
        return PRECEDENCE["Atom"]
    elif item.is_ground:
        return precedence(item.coeff(1))
    elif item.is_term:
        return PRECEDENCE["Mul"]
    else:
        return PRECEDENCE["Add"]


def precedence_FracElement(item):
    if item.denom == 1:
        return precedence_PolyElement(item.numer)
    else:
        return PRECEDENCE["Mul"]


def precedence_UnevaluatedExpr(item):
    return precedence(item.args[0]) - 0.5


PRECEDENCE_FUNCTIONS = {
    "Integer": precedence_Integer,
    "Mul": precedence_Mul,
    "Rational": precedence_Rational,
    "Float": precedence_Float,
    "PolyElement": precedence_PolyElement,
    "FracElement": precedence_FracElement,
    "UnevaluatedExpr": precedence_UnevaluatedExpr,
}


def precedence(item):
    """Returns the precedence of a given object.

    This is the precedence for StrPrinter.
    """
    if hasattr(item, "precedence"):
        return item.precedence
    if not isinstance(item, type):
        for i in type(item).mro():
            n = i.__name__
            if n in PRECEDENCE_FUNCTIONS:
                return PRECEDENCE_FUNCTIONS[n](item)
            elif n in PRECEDENCE_VALUES:
                return PRECEDENCE_VALUES[n]
    return PRECEDENCE["Atom"]


PRECEDENCE_TRADITIONAL = PRECEDENCE.copy()
PRECEDENCE_TRADITIONAL['Integral'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Sum'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Product'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Limit'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Derivative'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['TensorProduct'] = PRECEDENCE["Mul"]
PRECEDENCE_TRADITIONAL['Transpose'] = PRECEDENCE["Pow"]
PRECEDENCE_TRADITIONAL['Adjoint'] = PRECEDENCE["Pow"]
PRECEDENCE_TRADITIONAL['Dot'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Cross'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Gradient'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Divergence'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Curl'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Laplacian'] = PRECEDENCE["Mul"] - 1
PRECEDENCE_TRADITIONAL['Union'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['Intersection'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['Complement'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['SymmetricDifference'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['ProductSet'] = PRECEDENCE['Xor']
PRECEDENCE_TRADITIONAL['DotProduct'] = PRECEDENCE_TRADITIONAL['Dot']


def precedence_traditional(item):
    """Returns the precedence of a given object according to the
    traditional rules of mathematics.

    This is the precedence for the LaTeX and pretty printer.
    """
    # Integral, Sum, Product, Limit have the precedence of Mul in LaTeX,
    # the precedence of Atom for other printers:
    from sympy.core.expr import UnevaluatedExpr

    if isinstance(item, UnevaluatedExpr):
        return precedence_traditional(item.args[0])

    n = item.__class__.__name__
    if n in PRECEDENCE_TRADITIONAL:
        return PRECEDENCE_TRADITIONAL[n]

    return precedence(item)
<|endoftext|>"""
Javascript code printer

The JavascriptCodePrinter converts single SymPy expressions into single
Javascript expressions, using the functions defined in the Javascript
Math object where possible.

"""

from __future__ import annotations
from typing import Any

from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE


# dictionary mapping SymPy function to (argument_conditions, Javascript_function).
# Used in JavascriptCodePrinter._print_Function(self)
known_functions = {
    'Abs': 'Math.abs',
    'acos': 'Math.acos',
    'acosh': 'Math.acosh',
    'asin': 'Math.asin',
    'asinh': 'Math.asinh',
    'atan': 'Math.atan',
    'atan2': 'Math.atan2',
    'atanh': 'Math.atanh',
    'ceiling': 'Math.ceil',
    'cos': 'Math.cos',
    'cosh': 'Math.cosh',
    'exp': 'Math.exp',
    'floor': 'Math.floor',
    'log': 'Math.log',
    'Max': 'Math.max',
    'Min': 'Math.min',
    'sign': 'Math.sign',
    'sin': 'Math.sin',
    'sinh': 'Math.sinh',
    'tan': 'Math.tan',
    'tanh': 'Math.tanh',
}


class JavascriptCodePrinter(CodePrinter):
    """"A Printer to convert Python expressions to strings of JavaScript code
    """
    printmethod = '_javascript'
    language = 'JavaScript'

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
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
        return "var {} = {};".format(name, value.evalf(self._settings['precision']))

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (var %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){"
        for i in indices:
            # Javascript arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'varble': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'Math.sqrt(%s)' % self._print(expr.base)
        elif expr.exp == S.One/3:
            return 'Math.cbrt(%s)' % self._print(expr.base)
        else:
            return 'Math.pow(%s, %s)' % (self._print(expr.base),
                                 self._print(expr.exp))

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d/%d' % (p, q)

    def _print_Mod(self, expr):
        num, den = expr.args
        PREC = precedence(expr)
        snum, sden = [self.parenthesize(arg, PREC) for arg in expr.args]
        # % is remainder (same sign as numerator), not modulo (same sign as
        # denominator), in js. Hence, % only works as modulo if both numbers
        # have the same sign
        if (num.is_nonnegative and den.is_nonnegative or
            num.is_nonpositive and den.is_nonpositive):
            return f"{snum} % {sden}"
        return f"(({snum} % {sden}) + {sden}) % {sden}"

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

    def _print_Exp1(self, expr):
        return "Math.E"

    def _print_Pi(self, expr):
        return 'Math.PI'

    def _print_Infinity(self, expr):
        return 'Number.POSITIVE_INFINITY'

    def _print_NegativeInfinity(self, expr):
        return 'Number.NEGATIVE_INFINITY'

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
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_MatrixElement(self, expr):
        return "{}[{}]".format(self.parenthesize(expr.parent,
            PRECEDENCE["Atom"], strict=True),
            expr.j + expr.i*expr.parent.shape[1])

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

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


def jscode(expr, assign_to=None, **settings):
    """Converts an expr to a string of javascript code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
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

    >>> from sympy import jscode, symbols, Rational, sin, ceiling, Abs
    >>> x, tau = symbols("x, tau")
    >>> jscode((2*tau)**Rational(7, 2))
    '8*Math.sqrt(2)*Math.pow(tau, 7/2)'
    >>> jscode(sin(x), assign_to="s")
    's = Math.sin(x);'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    js_function_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")]
    ... }
    >>> jscode(Abs(x) + ceiling(x), user_functions=custom_functions)
    'fabs(x) + CEIL(x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(jscode(expr, tau))
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
    >>> jscode(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(jscode(mat, A))
    A[0] = Math.pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = Math.sin(x);
    """

    return JavascriptCodePrinter(settings).doprint(expr, assign_to)


def print_jscode(expr, **settings):
    """Prints the Javascript representation of the given expression.

       See jscode for the meaning of the optional arguments.
    """
    print(jscode(expr, **settings))
<|endoftext|>"""Printing subsystem"""

from .pretty import pager_print, pretty, pretty_print, pprint, pprint_use_unicode, pprint_try_use_unicode

from .latex import latex, print_latex, multiline_latex

from .mathml import mathml, print_mathml

from .python import python, print_python

from .pycode import pycode

from .codeprinter import print_ccode, print_fcode

from .codeprinter import ccode, fcode, cxxcode, rust_code # noqa:F811

from .smtlib import smtlib_code

from .glsl import glsl_code, print_glsl

from .rcode import rcode, print_rcode

from .jscode import jscode, print_jscode

from .julia import julia_code

from .mathematica import mathematica_code

from .octave import octave_code

from .gtk import print_gtk

from .preview import preview

from .repr import srepr

from .tree import print_tree

from .str import StrPrinter, sstr, sstrrepr

from .tableform import TableForm

from .dot import dotprint

from .maple import maple_code, print_maple_code

__all__ = [
    # sympy.printing.pretty
    'pager_print', 'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode',

    # sympy.printing.latex
    'latex', 'print_latex', 'multiline_latex',

    # sympy.printing.mathml
    'mathml', 'print_mathml',

    # sympy.printing.python
    'python', 'print_python',

    # sympy.printing.pycode
    'pycode',

    # sympy.printing.codeprinter
    'ccode', 'print_ccode', 'cxxcode', 'fcode', 'print_fcode', 'rust_code',

    # sympy.printing.smtlib
    'smtlib_code',

    # sympy.printing.glsl
    'glsl_code', 'print_glsl',

    # sympy.printing.rcode
    'rcode', 'print_rcode',

    # sympy.printing.jscode
    'jscode', 'print_jscode',

    # sympy.printing.julia
    'julia_code',

    # sympy.printing.mathematica
    'mathematica_code',

    # sympy.printing.octave
    'octave_code',

    # sympy.printing.gtk
    'print_gtk',

    # sympy.printing.preview
    'preview',

    # sympy.printing.repr
    'srepr',

    # sympy.printing.tree
    'print_tree',

    # sympy.printing.str
    'StrPrinter', 'sstr', 'sstrrepr',

    # sympy.printing.tableform
    'TableForm',

    # sympy.printing.dot
    'dotprint',

    # sympy.printing.maple
    'maple_code', 'print_maple_code',
]
<|endoftext|>from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError

from types import FunctionType


class TableForm:
    r"""
    Create a nice table representation of data.

    Examples
    ========

    >>> from sympy import TableForm
    >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
    >>> print(t)
    5  7
    4  2
    10 3

    You can use the SymPy's printing system to produce tables in any
    format (ascii, latex, html, ...).

    >>> print(t.as_latex())
    \begin{tabular}{l l}
    $5$ & $7$ \\
    $4$ & $2$ \\
    $10$ & $3$ \\
    \end{tabular}

    """

    def __init__(self, data, **kwarg):
        """
        Creates a TableForm.

        Parameters:

            data ...
                            2D data to be put into the table; data can be
                            given as a Matrix

            headings ...
                            gives the labels for rows and columns:

                            Can be a single argument that applies to both
                            dimensions:

                                - None ... no labels
                                - "automatic" ... labels are 1, 2, 3, ...

                            Can be a list of labels for rows and columns:
                            The labels for each dimension can be given
                            as None, "automatic", or [l1, l2, ...] e.g.
                            ["automatic", None] will number the rows

                            [default: None]

            alignments ...
                            alignment of the columns with:

                                - "left" or "<"
                                - "center" or "^"
                                - "right" or ">"

                            When given as a single value, the value is used for
                            all columns. The row headings (if given) will be
                            right justified unless an explicit alignment is
                            given for it and all other columns.

                            [default: "left"]

            formats ...
                            a list of format strings or functions that accept
                            3 arguments (entry, row number, col number) and
                            return a string for the table entry. (If a function
                            returns None then the _print method will be used.)

            wipe_zeros ...
                            Do not show zeros in the table.

                            [default: True]

            pad ...
                            the string to use to indicate a missing value (e.g.
                            elements that are None or those that are missing
                            from the end of a row (i.e. any row that is shorter
                            than the rest is assumed to have missing values).
                            When None, nothing will be shown for values that
                            are missing from the end of a row; values that are
                            None, however, will be shown.

                            [default: None]

        Examples
        ========

        >>> from sympy import TableForm, Symbol
        >>> TableForm([[5, 7], [4, 2], [10, 3]])
        5  7
        4  2
        10 3
        >>> TableForm([list('.'*i) for i in range(1, 4)], headings='automatic')
          | 1 2 3
        ---------
        1 | .
        2 | . .
        3 | . . .
        >>> TableForm([[Symbol('.'*(j if not i%2 else 1)) for i in range(3)]
        ...            for j in range(4)], alignments='rcl')
            .
          . . .
         .. . ..
        ... . ...
        """
        from sympy.matrices.dense import Matrix

        # We only support 2D data. Check the consistency:
        if isinstance(data, Matrix):
            data = data.tolist()
        _h = len(data)

        # fill out any short lines
        pad = kwarg.get('pad', None)
        ok_None = False
        if pad is None:
            pad = " "
            ok_None = True
        pad = Symbol(pad)
        _w = max(len(line) for line in data)
        for i, line in enumerate(data):
            if len(line) != _w:
                line.extend([pad]*(_w - len(line)))
            for j, lj in enumerate(line):
                if lj is None:
                    if not ok_None:
                        lj = pad
                else:
                    try:
                        lj = S(lj)
                    except SympifyError:
                        lj = Symbol(str(lj))
                line[j] = lj
            data[i] = line
        _lines = Tuple(*[Tuple(*d) for d in data])

        headings = kwarg.get("headings", [None, None])
        if headings == "automatic":
            _headings = [range(1, _h + 1), range(1, _w + 1)]
        else:
            h1, h2 = headings
            if h1 == "automatic":
                h1 = range(1, _h + 1)
            if h2 == "automatic":
                h2 = range(1, _w + 1)
            _headings = [h1, h2]

        allow = ('l', 'r', 'c')
        alignments = kwarg.get("alignments", "l")

        def _std_align(a):
            a = a.strip().lower()
            if len(a) > 1:
                return {'left': 'l', 'right': 'r', 'center': 'c'}.get(a, a)
            else:
                return {'<': 'l', '>': 'r', '^': 'c'}.get(a, a)
        std_align = _std_align(alignments)
        if std_align in allow:
            _alignments = [std_align]*_w
        else:
            _alignments = []
            for a in alignments:
                std_align = _std_align(a)
                _alignments.append(std_align)
                if std_align not in ('l', 'r', 'c'):
                    raise ValueError('alignment "%s" unrecognized' %
                        alignments)
        if _headings[0] and len(_alignments) == _w + 1:
            _head_align = _alignments[0]
            _alignments = _alignments[1:]
        else:
            _head_align = 'r'
        if len(_alignments) != _w:
            raise ValueError(
                'wrong number of alignments: expected %s but got %s' %
                (_w, len(_alignments)))

        _column_formats = kwarg.get("formats", [None]*_w)

        _wipe_zeros = kwarg.get("wipe_zeros", True)

        self._w = _w
        self._h = _h
        self._lines = _lines
        self._headings = _headings
        self._head_align = _head_align
        self._alignments = _alignments
        self._column_formats = _column_formats
        self._wipe_zeros = _wipe_zeros

    def __repr__(self):
        from .str import sstr
        return sstr(self, order=None)

    def __str__(self):
        from .str import sstr
        return sstr(self, order=None)

    def as_matrix(self):
        """Returns the data of the table in Matrix form.

        Examples
        ========

        >>> from sympy import TableForm
        >>> t = TableForm([[5, 7], [4, 2], [10, 3]], headings='automatic')
        >>> t
          | 1  2
        --------
        1 | 5  7
        2 | 4  2
        3 | 10 3
        >>> t.as_matrix()
        Matrix([
        [ 5, 7],
        [ 4, 2],
        [10, 3]])
        """
        from sympy.matrices.dense import Matrix
        return Matrix(self._lines)

    def as_str(self):
        # XXX obsolete ?
        return str(self)

    def as_latex(self):
        from .latex import latex
        return latex(self)

    def _sympystr(self, p):
        """
        Returns the string representation of 'self'.

        Examples
        ========

        >>> from sympy import TableForm
        >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
        >>> s = t.as_str()

        """
        column_widths = [0] * self._w
        lines = []
        for line in self._lines:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                s = str(line[i])
                if self._wipe_zeros and (s == "0"):
                    s = " "
                w = len(s)
                if w > column_widths[i]:
                    column_widths[i] = w
                new_line.append(s)
            lines.append(new_line)

        # Check heading:
        if self._headings[0]:
            self._headings[0] = [str(x) for x in self._headings[0]]
            _head_width = max(len(x) for x in self._headings[0])

        if self._headings[1]:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                s = str(self._headings[1][i])
                w = len(s)
                if w > column_widths[i]:
                    column_widths[i] = w
                new_line.append(s)
            self._headings[1] = new_line

        format_str = []

        def _align(align, w):
            return '%%%s%ss' % (
                ("-" if align == "l" else ""),
                str(w))
        format_str = [_align(align, w) for align, w in
                      zip(self._alignments, column_widths)]
        if self._headings[0]:
            format_str.insert(0, _align(self._head_align, _head_width))
            format_str.insert(1, '|')
        format_str = ' '.join(format_str) + '\n'

        s = []
        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [""] + d
            first_line = format_str % tuple(d)
            s.append(first_line)
            s.append("-" * (len(first_line) - 1) + "\n")
        for i, line in enumerate(lines):
            d = [l if self._alignments[j] != 'c' else
                 l.center(column_widths[j]) for j, l in enumerate(line)]
            if self._headings[0]:
                l = self._headings[0][i]
                l = (l if self._head_align != 'c' else
                     l.center(_head_width))
                d = [l] + d
            s.append(format_str % tuple(d))
        return ''.join(s)[:-1]  # don't include trailing newline

    def _latex(self, printer):
        """
        Returns the string representation of 'self'.
        """
        # Check heading:
        if self._headings[1]:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                new_line.append(str(self._headings[1][i]))
            self._headings[1] = new_line

        alignments = []
        if self._headings[0]:
            self._headings[0] = [str(x) for x in self._headings[0]]
            alignments = [self._head_align]
        alignments.extend(self._alignments)

        s = r"\begin{tabular}{" + " ".join(alignments) + "}\n"

        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [""] + d
            first_line = " & ".join(d) + r" \\" + "\n"
            s += first_line
            s += r"\hline" + "\n"
        for i, line in enumerate(self._lines):
            d = []
            for j, x in enumerate(line):
                if self._wipe_zeros and (x in (0, "0")):
                    d.append(" ")
                    continue
                f = self._column_formats[j]
                if f:
                    if isinstance(f, FunctionType):
                        v = f(x, i, j)
                        if v is None:
                            v = printer._print(x)
                    else:
                        v = f % x
                    d.append(v)
                else:
                    v = printer._print(x)
                    d.append("$%s$" % v)
            if self._headings[0]:
                d = [self._headings[0][i]] + d
            s += " & ".join(d) + r" \\" + "\n"
        s += r"\end{tabular}"
        return s
<|endoftext|>import keyword as kw
import sympy
from .repr import ReprPrinter
from .str import StrPrinter

# A list of classes that should be printed using StrPrinter
STRPRINT = ("Add", "Infinity", "Integer", "Mul", "NegativeInfinity", "Pow")


class PythonPrinter(ReprPrinter, StrPrinter):
    """A printer which converts an expression into its Python interpretation."""

    def __init__(self, settings=None):
        super().__init__(settings)
        self.symbols = []
        self.functions = []

        # Create print methods for classes that should use StrPrinter instead
        # of ReprPrinter.
        for name in STRPRINT:
            f_name = "_print_%s" % name
            f = getattr(StrPrinter, f_name)
            setattr(PythonPrinter, f_name, f)

    def _print_Function(self, expr):
        func = expr.func.__name__
        if not hasattr(sympy, func) and func not in self.functions:
            self.functions.append(func)
        return StrPrinter._print_Function(self, expr)

    # procedure (!) for defining symbols which have be defined in print_python()
    def _print_Symbol(self, expr):
        symbol = self._str(expr)
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        return StrPrinter._print_Symbol(self, expr)

    def _print_module(self, expr):
        raise ValueError('Modules in the expression are unacceptable')


def python(expr, **settings):
    """Return Python interpretation of passed expression
    (can be passed to the exec() function without any modifications)"""

    printer = PythonPrinter(settings)
    exprp = printer.doprint(expr)

    result = ''
    # Returning found symbols and functions
    renamings = {}
    for symbolname in printer.symbols:
        # Remove curly braces from subscripted variables
        if '{' in symbolname:
            newsymbolname = symbolname.replace('{', '').replace('}', '')
            renamings[sympy.Symbol(symbolname)] = newsymbolname
        else:
            newsymbolname = symbolname

        # Escape symbol names that are reserved Python keywords
        if kw.iskeyword(newsymbolname):
            while True:
                newsymbolname += "_"
                if (newsymbolname not in printer.symbols and
                        newsymbolname not in printer.functions):
                    renamings[sympy.Symbol(
                        symbolname)] = sympy.Symbol(newsymbolname)
                    break
        result += newsymbolname + ' = Symbol(\'' + symbolname + '\')\n'

    for functionname in printer.functions:
        newfunctionname = functionname
        # Escape function names that are reserved Python keywords
        if kw.iskeyword(newfunctionname):
            while True:
                newfunctionname += "_"
                if (newfunctionname not in printer.symbols and
                        newfunctionname not in printer.functions):
                    renamings[sympy.Function(
                        functionname)] = sympy.Function(newfunctionname)
                    break
        result += newfunctionname + ' = Function(\'' + functionname + '\')\n'

    if renamings:
        exprp = expr.subs(renamings)
    result += 'e = ' + printer._str(exprp)
    return result


def print_python(expr, **settings):
    """Print output of python() function"""
    print(python(expr, **settings))
<|endoftext|>"""
.. deprecated:: 1.8

  ``sympy.printing.theanocode`` is deprecated. Theano has been renamed to
  Aesara. Use ``sympy.printing.aesaracode`` instead. See
  :ref:`theanocode-deprecated` for more information.

"""
from __future__ import annotations
import math
from typing import Any

from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial

from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning


__doctest_requires__ = {('theano_function',): ['theano']}


theano = import_module('theano')


if theano:
    ts = theano.scalar
    tt = theano.tensor
    from theano.sandbox import linalg as tlinalg

    mapping = {
            sympy.Add: tt.add,
            sympy.Mul: tt.mul,
            sympy.Abs: tt.abs_,
            sympy.sign: tt.sgn,
            sympy.ceiling: tt.ceil,
            sympy.floor: tt.floor,
            sympy.log: tt.log,
            sympy.exp: tt.exp,
            sympy.sqrt: tt.sqrt,
            sympy.cos: tt.cos,
            sympy.acos: tt.arccos,
            sympy.sin: tt.sin,
            sympy.asin: tt.arcsin,
            sympy.tan: tt.tan,
            sympy.atan: tt.arctan,
            sympy.atan2: tt.arctan2,
            sympy.cosh: tt.cosh,
            sympy.acosh: tt.arccosh,
            sympy.sinh: tt.sinh,
            sympy.asinh: tt.arcsinh,
            sympy.tanh: tt.tanh,
            sympy.atanh: tt.arctanh,
            sympy.re: tt.real,
            sympy.im: tt.imag,
            sympy.arg: tt.angle,
            sympy.erf: tt.erf,
            sympy.gamma: tt.gamma,
            sympy.loggamma: tt.gammaln,
            sympy.Pow: tt.pow,
            sympy.Eq: tt.eq,
            sympy.StrictGreaterThan: tt.gt,
            sympy.StrictLessThan: tt.lt,
            sympy.LessThan: tt.le,
            sympy.GreaterThan: tt.ge,
            sympy.And: tt.and_,
            sympy.Or: tt.or_,
            sympy.Max: tt.maximum,  # SymPy accept >2 inputs, Theano only 2
            sympy.Min: tt.minimum,  # SymPy accept >2 inputs, Theano only 2
            sympy.conjugate: tt.conj,
            sympy.core.numbers.ImaginaryUnit: lambda:tt.complex(0,1),
            # Matrices
            sympy.MatAdd: tt.Elemwise(ts.add),
            sympy.HadamardProduct: tt.Elemwise(ts.mul),
            sympy.Trace: tlinalg.trace,
            sympy.Determinant : tlinalg.det,
            sympy.Inverse: tlinalg.matrix_inverse,
            sympy.Transpose: tt.DimShuffle((False, False), [1, 0]),
    }


class TheanoPrinter(Printer):
    """ Code printer which creates Theano symbolic expression graphs.

    Parameters
    ==========

    cache : dict
        Cache dictionary to use. If None (default) will use
        the global cache. To create a printer which does not depend on or alter
        global state pass an empty dictionary. Note: the dictionary is not
        copied on initialization of the printer and will be updated in-place,
        so using the same dict object when creating multiple printers or making
        multiple calls to :func:`.theano_code` or :func:`.theano_function` means
        the cache is shared between all these applications.

    Attributes
    ==========

    cache : dict
        A cache of Theano variables which have been created for SymPy
        symbol-like objects (e.g. :class:`sympy.core.symbol.Symbol` or
        :class:`sympy.matrices.expressions.MatrixSymbol`). This is used to
        ensure that all references to a given symbol in an expression (or
        multiple expressions) are printed as the same Theano variable, which is
        created only once. Symbols are differentiated only by name and type. The
        format of the cache's contents should be considered opaque to the user.
    """
    printmethod = "_theano"

    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop('cache', {})
        super().__init__(*args, **kwargs)

    def _get_key(self, s, name=None, dtype=None, broadcastable=None):
        """ Get the cache key for a SymPy object.

        Parameters
        ==========

        s : sympy.core.basic.Basic
            SymPy object to get key for.

        name : str
            Name of object, if it does not have a ``name`` attribute.
        """

        if name is None:
            name = s.name

        return (name, type(s), s.args, dtype, broadcastable)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        Get the Theano variable for a SymPy symbol from the cache, or create it
        if it does not exist.
        """

        # Defaults
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)

        if key in self.cache:
            return self.cache[key]

        value = tt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        op = mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        # Integers already taken care of below, interpret as float
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        if not hasattr(tt, 'stacklists'):
            raise NotImplementedError(
               "Matrix translation not yet supported in this version of Theano")

        return tt.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = tt.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = 1
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = tt.dot(result, children[0])
        else:
            raise NotImplementedError('''Only non-negative integer
           powers of matrices can be handled by Theano at the moment''')
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        return tt.join(0, *[tt.join(1, *row) for row in blocks])


    def _print_slice(self, expr, **kwargs):
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        return math.pi

    def _print_Exp1(self, expr, **kwargs):
        return ts.exp(1)

    def _print_Piecewise(self, expr, **kwargs):
        import numpy as np
        e, cond = expr.args[0].args  # First condition and corresponding value

        # Print conditional expression and value for first condition
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)

        # One condition only
        if len(expr.args) == 1:
            # Return value if condition else NaN
            return tt.switch(p_cond, p_e, np.nan)

        # Return value_1 if condition_1 else evaluate remaining conditions
        p_remaining = self._print(sympy.Piecewise(*expr.args[1:]), **kwargs)
        return tt.switch(p_cond, p_e, p_remaining)

    def _print_Rational(self, expr, **kwargs):
        return tt.true_div(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    def _print_Integer(self, expr, **kwargs):
        return expr.p

    def _print_factorial(self, expr, **kwargs):
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    def _print_Derivative(self, deriv, **kwargs):
        rv = self._print(deriv.expr, **kwargs)
        for var in deriv.variables:
