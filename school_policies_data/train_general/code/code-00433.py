    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.upper_triangular(base), assumptions)
    return None

@UpperTriangularPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if Q.upper_triangular(expr) in conjuncts(assumptions):
        return True

@UpperTriangularPredicate.register_many(Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@UpperTriangularPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@UpperTriangularPredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.lower_triangular(expr.arg), assumptions)

@UpperTriangularPredicate.register(Inverse)
def _(expr, assumptions):
    return ask(Q.upper_triangular(expr.arg), assumptions)

@UpperTriangularPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.upper_triangular(expr.parent), assumptions)

@UpperTriangularPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.upper_triangular, expr, assumptions)

# LowerTriangularPredicate

@LowerTriangularPredicate.register(MatMul)
def _(expr, assumptions):
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.lower_triangular(m), assumptions) for m in matrices):
        return True

@LowerTriangularPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.lower_triangular(arg), assumptions) for arg in expr.args):
        return True

@LowerTriangularPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.lower_triangular(base), assumptions)
    return None

@LowerTriangularPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if Q.lower_triangular(expr) in conjuncts(assumptions):
        return True

@LowerTriangularPredicate.register_many(Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@LowerTriangularPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@LowerTriangularPredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.upper_triangular(expr.arg), assumptions)

@LowerTriangularPredicate.register(Inverse)
def _(expr, assumptions):
    return ask(Q.lower_triangular(expr.arg), assumptions)

@LowerTriangularPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.lower_triangular(expr.parent), assumptions)

@LowerTriangularPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.lower_triangular, expr, assumptions)


# DiagonalPredicate

def _is_empty_or_1x1(expr):
    return expr.shape in ((0, 0), (1, 1))

@DiagonalPredicate.register(MatMul)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.diagonal(m), assumptions) for m in matrices):
        return True

@DiagonalPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.diagonal(base), assumptions)
    return None

@DiagonalPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.diagonal(arg), assumptions) for arg in expr.args):
        return True

@DiagonalPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    if Q.diagonal(expr) in conjuncts(assumptions):
        return True

@DiagonalPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@DiagonalPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.diagonal(expr.arg), assumptions)

@DiagonalPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    if not expr.on_diag:
        return None
    else:
        return ask(Q.diagonal(expr.parent), assumptions)

@DiagonalPredicate.register_many(DiagonalMatrix, DiagMatrix, Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@DiagonalPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.diagonal, expr, assumptions)


# IntegerElementsPredicate

def BM_elements(predicate, expr, assumptions):
    """ Block Matrix elements. """
    return all(ask(predicate(b), assumptions) for b in expr.blocks)

def MS_elements(predicate, expr, assumptions):
    """ Matrix Slice elements. """
    return ask(predicate(expr.parent), assumptions)

def MatMul_elements(matrix_predicate, scalar_predicate, expr, assumptions):
    d = sift(expr.args, lambda x: isinstance(x, MatrixExpr))
    factors, matrices = d[False], d[True]
    return fuzzy_and([
        test_closed_group(Basic(*factors), assumptions, scalar_predicate),
        test_closed_group(Basic(*matrices), assumptions, matrix_predicate)])


@IntegerElementsPredicate.register_many(Determinant, HadamardProduct, MatAdd,
    Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.integer_elements)

@IntegerElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    if exp.is_negative == False:
        return ask(Q.integer_elements(base), assumptions)
    return None

@IntegerElementsPredicate.register_many(Identity, OneMatrix, ZeroMatrix)
def _(expr, assumptions):
    return True

@IntegerElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.integer_elements, Q.integer, expr, assumptions)

@IntegerElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.integer_elements, expr, assumptions)

@IntegerElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.integer_elements, expr, assumptions)


# RealElementsPredicate

@RealElementsPredicate.register_many(Determinant, Factorization, HadamardProduct,
    MatAdd, Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.real_elements)

@RealElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.real_elements(base), assumptions)
    return None

@RealElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.real_elements, Q.real, expr, assumptions)

@RealElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.real_elements, expr, assumptions)

@RealElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.real_elements, expr, assumptions)


# ComplexElementsPredicate

@ComplexElementsPredicate.register_many(Determinant, Factorization, HadamardProduct,
    Inverse, MatAdd, Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.complex_elements)

@ComplexElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.complex_elements(base), assumptions)
    return None

@ComplexElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.complex_elements, Q.complex, expr, assumptions)

@ComplexElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.complex_elements, expr, assumptions)

@ComplexElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.complex_elements, expr, assumptions)

@ComplexElementsPredicate.register(DFT)
def _(expr, assumptions):
    return True
<|endoftext|>"""Functions that involve magic. """

def pollute(names, objects):
    """Pollute the global namespace with symbols -> objects mapping. """
    from inspect import currentframe
    frame = currentframe().f_back.f_back

    try:
        for name, obj in zip(names, objects):
            frame.f_globals[name] = obj
    finally:
        del frame  # break cyclic dependencies as stated in inspect docs
<|endoftext|>"""
This module adds several functions for interactive source code inspection.
"""


def get_class(lookup_view):
    """
    Convert a string version of a class name to the object.

    For example, get_class('sympy.core.Basic') will return
    class Basic located in module sympy.core
    """
    if isinstance(lookup_view, str):
        mod_name, func_name = get_mod_func(lookup_view)
        if func_name != '':
            lookup_view = getattr(
                __import__(mod_name, {}, {}, ['*']), func_name)
            if not callable(lookup_view):
                raise AttributeError(
                    "'%s.%s' is not a callable." % (mod_name, func_name))
    return lookup_view


def get_mod_func(callback):
    """
    splits the string path to a class into a string path to the module
    and the name of the class.

    Examples
    ========

    >>> from sympy.utilities.source import get_mod_func
    >>> get_mod_func('sympy.core.basic.Basic')
    ('sympy.core.basic', 'Basic')

    """
    dot = callback.rfind('.')
    if dot == -1:
        return callback, ''
    return callback[:dot], callback[dot + 1:]
<|endoftext|>"""
module for generating C, C++, Fortran77, Fortran90, Julia, Rust
and Octave/Matlab routines that evaluate SymPy expressions.
This module is work in progress.
Only the milestones with a '+' character in the list below have been completed.

--- How is sympy.utilities.codegen different from sympy.printing.ccode? ---

We considered the idea to extend the printing routines for SymPy functions in
such a way that it prints complete compilable code, but this leads to a few
unsurmountable issues that can only be tackled with dedicated code generator:

- For C, one needs both a code and a header file, while the printing routines
  generate just one string. This code generator can be extended to support
  .pyf files for f2py.

- SymPy functions are not concerned with programming-technical issues, such
  as input, output and input-output arguments. Other examples are contiguous
  or non-contiguous arrays, including headers of other libraries such as gsl
  or others.

- It is highly interesting to evaluate several SymPy functions in one C
  routine, eventually sharing common intermediate results with the help
  of the cse routine. This is more than just printing.

- From the programming perspective, expressions with constants should be
  evaluated in the code generator as much as possible. This is different
  for printing.

--- Basic assumptions ---

* A generic Routine data structure describes the routine that must be
  translated into C/Fortran/... code. This data structure covers all
  features present in one or more of the supported languages.

* Descendants from the CodeGen class transform multiple Routine instances
  into compilable code. Each derived class translates into a specific
  language.

* In many cases, one wants a simple workflow. The friendly functions in the
  last part are a simple api on top of the Routine/CodeGen stuff. They are
  easier to use, but are less powerful.

--- Milestones ---

+ First working version with scalar input arguments, generating C code,
  tests
+ Friendly functions that are easier to use than the rigorous
  Routine/CodeGen workflow.
+ Integer and Real numbers as input and output
+ Output arguments
+ InputOutput arguments
+ Sort input/output arguments properly
+ Contiguous array arguments (numpy matrices)
+ Also generate .pyf code for f2py (in autowrap module)
+ Isolate constants and evaluate them beforehand in double precision
+ Fortran 90
+ Octave/Matlab

- Common Subexpression Elimination
- User defined comments in the generated code
- Optional extra include lines for libraries/objects that can eval special
  functions
- Test other C compilers and libraries: gcc, tcc, libtcc, gcc+gsl, ...
- Contiguous array arguments (SymPy matrices)
- Non-contiguous array arguments (SymPy matrices)
- ccode must raise an error when it encounters something that cannot be
  translated into c. ccode(integrate(sin(x)/x, x)) does not make sense.
- Complex numbers as input and output
- A default complex datatype
- Include extra information in the header: date, user, hostname, sha1
  hash, ...
- Fortran 77
- C++
- Python
- Julia
- Rust
- ...

"""

import os
import textwrap
from io import StringIO

from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.printing.c import c_code_printers
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.fortran import FCodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)
from sympy.utilities.iterables import is_sequence


__all__ = [
    # description of routines
    "Routine", "DataType", "default_datatypes", "get_default_datatype",
    "Argument", "InputArgument", "OutputArgument", "Result",
    # routines -> code
    "CodeGen", "CCodeGen", "FCodeGen", "JuliaCodeGen", "OctaveCodeGen",
    "RustCodeGen",
    # friendly functions
    "codegen", "make_routine",
]


#
# Description of routines
#


class Routine:
    """Generic description of evaluation routine for set of expressions.

    A CodeGen class can translate instances of this class into code in a
    particular language.  The routine specification covers all the features
    present in these languages.  The CodeGen part must raise an exception
    when certain features are not present in the target language.  For
    example, multiple return values are possible in Python, but not in C or
    Fortran.  Another example: Fortran and Python support complex numbers,
    while C does not.

    """

    def __init__(self, name, arguments, results, local_vars, global_vars):
        """Initialize a Routine instance.

        Parameters
        ==========

        name : string
            Name of the routine.

        arguments : list of Arguments
            These are things that appear in arguments of a routine, often
            appearing on the right-hand side of a function call.  These are
            commonly InputArguments but in some languages, they can also be
            OutputArguments or InOutArguments (e.g., pass-by-reference in C
            code).

        results : list of Results
            These are the return values of the routine, often appearing on
            the left-hand side of a function call.  The difference between
            Results and OutputArguments and when you should use each is
            language-specific.

        local_vars : list of Results
            These are variables that will be defined at the beginning of the
            function.

        global_vars : list of Symbols
            Variables which will not be passed into the function.

        """

        # extract all input symbols and all symbols appearing in an expression
        input_symbols = set()
        symbols = set()
        for arg in arguments:
            if isinstance(arg, OutputArgument):
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            elif isinstance(arg, InputArgument):
                input_symbols.add(arg.name)
            elif isinstance(arg, InOutArgument):
                input_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            else:
                raise ValueError("Unknown Routine argument: %s" % arg)

        for r in results:
            if not isinstance(r, Result):
                raise ValueError("Unknown Routine result: %s" % r)
            symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))

        local_symbols = set()
        for r in local_vars:
            if isinstance(r, Result):
                symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))
                local_symbols.add(r.name)
            else:
                local_symbols.add(r)

        symbols = {s.label if isinstance(s, Idx) else s for s in symbols}

        # Check that all symbols in the expressions are covered by
        # InputArguments/InOutArguments---subset because user could
        # specify additional (unused) InputArguments or local_vars.
        notcovered = symbols.difference(
            input_symbols.union(local_symbols).union(global_vars))
        if notcovered != set():
            raise ValueError("Symbols needed for output are not in input " +
                             ", ".join([str(x) for x in notcovered]))

        self.name = name
        self.arguments = arguments
        self.results = results
        self.local_vars = local_vars
        self.global_vars = global_vars

    def __str__(self):
        return self.__class__.__name__ + "({name!r}, {arguments}, {results}, {local_vars}, {global_vars})".format(**self.__dict__)

    __repr__ = __str__

    @property
    def variables(self):
        """Returns a set of all variables possibly used in the routine.

        For routines with unnamed return values, the dummies that may or
        may not be used will be included in the set.

        """
        v = set(self.local_vars)
        v.update(arg.name for arg in self.arguments)
        v.update(res.result_var for res in self.results)
        return v

    @property
    def result_variables(self):
        """Returns a list of OutputArgument, InOutArgument and Result.

        If return values are present, they are at the end of the list.
        """
        args = [arg for arg in self.arguments if isinstance(
            arg, (OutputArgument, InOutArgument))]
        args.extend(self.results)
        return args


class DataType:
    """Holds strings for a certain datatype in different languages."""
    def __init__(self, cname, fname, pyname, jlname, octname, rsname):
        self.cname = cname
        self.fname = fname
        self.pyname = pyname
        self.jlname = jlname
        self.octname = octname
        self.rsname = rsname


default_datatypes = {
    "int": DataType("int", "INTEGER*4", "int", "", "", "i32"),
    "float": DataType("double", "REAL*8", "float", "", "", "f64"),
    "complex": DataType("double", "COMPLEX*16", "complex", "", "", "float") #FIXME:
       # complex is only supported in fortran, python, julia, and octave.
       # So to not break c or rust code generation, we stick with double or
       # float, respectively (but actually should raise an exception for
       # explicitly complex variables (x.is_complex==True))
}


COMPLEX_ALLOWED = False
def get_default_datatype(expr, complex_allowed=None):
    """Derives an appropriate datatype based on the expression."""
    if complex_allowed is None:
        complex_allowed = COMPLEX_ALLOWED
    if complex_allowed:
        final_dtype = "complex"
    else:
        final_dtype = "float"
    if expr.is_integer:
        return default_datatypes["int"]
    elif expr.is_real:
        return default_datatypes["float"]
    elif isinstance(expr, MatrixBase):
        #check all entries
        dt = "int"
        for element in expr:
            if dt == "int" and not element.is_integer:
                dt = "float"
            if dt == "float" and not element.is_real:
                return default_datatypes[final_dtype]
        return default_datatypes[dt]
    else:
        return default_datatypes[final_dtype]


class Variable:
    """Represents a typed variable."""

    def __init__(self, name, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol or MatrixSymbol

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimensions : sequence containing tuples, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        if not isinstance(name, (Symbol, MatrixSymbol)):
            raise TypeError("The first argument must be a SymPy symbol.")
        if datatype is None:
            datatype = get_default_datatype(name)
        elif not isinstance(datatype, DataType):
            raise TypeError("The (optional) `datatype' argument must be an "
                            "instance of the DataType class.")
        if dimensions and not isinstance(dimensions, (tuple, list)):
            raise TypeError(
                "The dimensions argument must be a sequence of tuples")

        self._name = name
        self._datatype = {
            'C': datatype.cname,
            'FORTRAN': datatype.fname,
            'JULIA': datatype.jlname,
            'OCTAVE': datatype.octname,
            'PYTHON': datatype.pyname,
            'RUST': datatype.rsname,
        }
        self.dimensions = dimensions
        self.precision = precision

    def __str__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)

    __repr__ = __str__

    @property
    def name(self):
        return self._name

    def get_datatype(self, language):
        """Returns the datatype string for the requested language.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.utilities.codegen import Variable
        >>> x = Variable(Symbol('x'))
        >>> x.get_datatype('c')
        'double'
        >>> x.get_datatype('fortran')
        'REAL*8'

        """
        try:
            return self._datatype[language.upper()]
        except KeyError:
            raise CodeGenError("Has datatypes for languages: %s" %
                    ", ".join(self._datatype))


class Argument(Variable):
    """An abstract Argument data structure: a name and a data type.

    This structure is refined in the descendants below.

    """
    pass


class InputArgument(Argument):
    pass


class ResultBase:
    """Base class for all "outgoing" information from a routine.

    Objects of this class stores a SymPy expression, and a SymPy object
    representing a result variable that will be used in the generated code
    only if necessary.

    """
    def __init__(self, expr, result_var):
        self.expr = expr
        self.result_var = result_var

    def __str__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.expr,
            self.result_var)

    __repr__ = __str__


class OutputArgument(Argument, ResultBase):
    """OutputArgument are always initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol, MatrixSymbol
            The name of this variable.  When used for code generation, this
            might appear, for example, in the prototype of function in the
            argument list.

        result_var : Symbol, Indexed
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".

        expr : object
            The expression that should be output, typically a SymPy
            expression.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimensions : sequence containing tuples, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """

        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.result_var, self.expr)

    __repr__ = __str__


class InOutArgument(Argument, ResultBase):
    """InOutArgument are never initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        if not datatype:
            datatype = get_default_datatype(expr)
        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)
    __init__.__doc__ = OutputArgument.__init__.__doc__


    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.expr,
            self.result_var)

    __repr__ = __str__


class Result(Variable, ResultBase):
    """An expression for a return value.

    The name result is used to avoid conflicts with the reserved word
    "return" in the Python language.  It is also shorter than ReturnValue.

    These may or may not need a name in the destination (e.g., "return(x*y)"
    might return a value without ever naming it).

    """

    def __init__(self, expr, name=None, result_var=None, datatype=None,
                 dimensions=None, precision=None):
        """Initialize a return value.

        Parameters
        ==========

        expr : SymPy expression

        name : Symbol, MatrixSymbol, optional
            The name of this return variable.  When used for code generation,
            this might appear, for example, in the prototype of function in a
            list of return values.  A dummy name is generated if omitted.

        result_var : Symbol, Indexed, optional
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".  Defaults to
            `name` if omitted.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the expr argument.

        dimensions : sequence containing tuples, optional
            If present, this variable is interpreted as an array,
            where this sequence of tuples specifies (lower, upper)
            bounds for each index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        # Basic because it is the base class for all types of expressions
        if not isinstance(expr, (Basic, MatrixBase)):
            raise TypeError("The first argument must be a SymPy expression.")

        if name is None:
            name = 'result_%d' % abs(hash(expr))

        if datatype is None:
            #try to infer data type from the expression
            datatype = get_default_datatype(expr)

        if isinstance(name, str):
            if isinstance(expr, (MatrixBase, MatrixExpr)):
                name = MatrixSymbol(name, *expr.shape)
            else:
                name = Symbol(name)

        if result_var is None:
            result_var = name

        Variable.__init__(self, name, datatype=datatype,
                          dimensions=dimensions, precision=precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.expr, self.name,
            self.result_var)

    __repr__ = __str__


#
# Transformation of routine objects into code
#

class CodeGen:
    """Abstract class for the code generators."""

    printer = None  # will be set to an instance of a CodePrinter subclass

    def _indent_code(self, codelines):
        return self.printer.indent_code(codelines)

    def _printer_method_with_settings(self, method, settings=None, *args, **kwargs):
        settings = settings or {}
        ori = {k: self.printer._settings[k] for k in settings}
        for k, v in settings.items():
            self.printer._settings[k] = v
        result = getattr(self.printer, method)(*args, **kwargs)
        for k, v in ori.items():
            self.printer._settings[k] = v
        return result

    def _get_symbol(self, s):
        """Returns the symbol as fcode prints it."""
        if self.printer._settings['human']:
            expr_str = self.printer.doprint(s)
        else:
            constants, not_supported, expr_str = self.printer.doprint(s)
            if constants or not_supported:
                raise ValueError("Failed to print %s" % str(s))
        return expr_str.strip()

    def __init__(self, project="project", cse=False):
        """Initialize a code generator.

        Derived classes will offer more options that affect the generated
        code.

        """
        self.project = project
        self.cse = cse

    def routine(self, name, expr, argument_sequence=None, global_vars=None):
        """Creates an Routine object that is appropriate for this language.

        This implementation is appropriate for at least C/Fortran.  Subclasses
        can override this if necessary.

        Here, we assume at most one return value (the l-value) which must be
        scalar.  Additional outputs are OutputArguments (e.g., pointers on
        right-hand-side or pass-by-reference).  Matrices are always returned
        via OutputArguments.  If ``argument_sequence`` is None, arguments will
        be ordered alphabetically, but with all InputArguments first, and then
        OutputArgument and InOutArguments.

        """

        if self.cse:
            from sympy.simplify.cse_main import cse

            if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
                if not expr:
                    raise ValueError("No expression given")
                for e in expr:
                    if not e.is_Equality:
                        raise CodeGenError("Lists of expressions must all be Equalities. {} is not.".format(e))

                # create a list of right hand sides and simplify them
                rhs = [e.rhs for e in expr]
                common, simplified = cse(rhs)

                # pack the simplified expressions back up with their left hand sides
                expr = [Equality(e.lhs, rhs) for e, rhs in zip(expr, simplified)]
            else:
                if isinstance(expr, Equality):
                    common, simplified = cse(expr.rhs) #, ignore=in_out_args)
                    expr = Equality(expr.lhs, simplified[0])
                else:
                    common, simplified = cse(expr)
                    expr = simplified

            local_vars = [Result(b,a) for a,b in common]
            local_symbols = {a for a,_ in common}
            local_expressions = Tuple(*[b for _,b in common])
        else:
            local_expressions = Tuple()

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        if self.cse:
            if {i.label for i in expressions.atoms(Idx)} != set():
                raise CodeGenError("CSE and Indexed expressions do not play well together yet")
        else:
            # local variables for indexed expressions
            local_vars = {i.label for i in expressions.atoms(Idx)}
            local_symbols = local_vars

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        symbols = (expressions.free_symbols | local_expressions.free_symbols) - local_symbols - global_vars
        new_symbols = set()
        new_symbols.update(symbols)

        for symbol in symbols:
            if isinstance(symbol, Idx):
                new_symbols.remove(symbol)
                new_symbols.update(symbol.args[1].free_symbols)
            if isinstance(symbol, Indexed):
                new_symbols.remove(symbol)
        symbols = new_symbols

        # Decide whether to use output argument or return value
        return_val = []
        output_args = []
        for expr in expressions:
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                elif isinstance(out_arg, Symbol):
                    dims = []
                    symbol = out_arg
                elif isinstance(out_arg, MatrixSymbol):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg
                else:
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                if expr.has(symbol):
                    output_args.append(
                        InOutArgument(symbol, out_arg, expr, dimensions=dims))
                else:
                    output_args.append(
                        OutputArgument(symbol, out_arg, expr, dimensions=dims))

                # remove duplicate arguments when they are not local variables
                if symbol not in local_vars:
                    # avoid duplicate arguments
                    symbols.remove(symbol)
            elif isinstance(expr, (ImmutableMatrix, MatrixSlice)):
                # Create a "dummy" MatrixSymbol to use as the Output arg
                out_arg = MatrixSymbol('out_%s' % abs(hash(expr)), *expr.shape)
                dims = tuple([(S.Zero, dim - 1) for dim in out_arg.shape])
                output_args.append(
                    OutputArgument(out_arg, out_arg, expr, dimensions=dims))
            else:
                return_val.append(Result(expr))

        arg_list = []

        # setup input argument list

        # helper to get dimensions for data for array-like args
        def dimensions(s):
            return [(S.Zero, dim - 1) for dim in s.shape]

        array_symbols = {}
        for array in expressions.atoms(Indexed) | local_expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol) | local_expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            if symbol in array_symbols:
                array = array_symbols[symbol]
                metadata = {'dimensions': dimensions(array)}
            else:
                metadata = {}

            arg_list.append(InputArgument(symbol, **metadata))

        output_args.sort(key=lambda x: str(x.name))
        arg_list.extend(output_args)

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    if isinstance(symbol, (IndexedBase, MatrixSymbol)):
                        metadata = {'dimensions': dimensions(symbol)}
                    else:
                        metadata = {}
                    new_args.append(InputArgument(symbol, **metadata))
            arg_list = new_args

        return Routine(name, arg_list, return_val, local_vars, global_vars)

    def write(self, routines, prefix, to_files=False, header=True, empty=True):
        """Writes all the source code files for the given routines.

        The generated source is returned as a list of (filename, contents)
        tuples, or is written to files (see below).  Each filename consists
        of the given prefix, appended with an appropriate extension.

        Parameters
        ==========

        routines : list
            A list of Routine instances to be written

        prefix : string
            The prefix for the output files

        to_files : bool, optional
            When True, the output is written to files.  Otherwise, a list
            of (filename, contents) tuples is returned.  [default: False]

        header : bool, optional
            When True, a header comment is included on top of each source
            file. [default: True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files. [default: True]

        """
        if to_files:
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                with open(filename, "w") as f:
                    dump_fn(self, routines, f, prefix, header, empty)
        else:
            result = []
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                contents = StringIO()
                dump_fn(self, routines, contents, prefix, header, empty)
                result.append((filename, contents.getvalue()))
            return result

    def dump_code(self, routines, f, prefix, header=True, empty=True):
        """Write the code by calling language specific methods.

        The generated file contains all the definitions of the routines in
        low-level code and refers to the header file if appropriate.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """

        code_lines = self._preprocessor_statements(prefix)

        for routine in routines:
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_opening(routine))
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._call_printer(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_ending(routine))

        code_lines = self._indent_code(''.join(code_lines))

        if header:
            code_lines = ''.join(self._get_header() + [code_lines])

        if code_lines:
            f.write(code_lines)


class CodeGenError(Exception):
    pass


class CodeGenArgumentListError(Exception):
    @property
    def missing_args(self):
        return self.args[1]


header_comment = """Code generated with SymPy %(version)s

See http://www.sympy.org/ for more information.

This file is part of '%(project)s'
"""


class CCodeGen(CodeGen):
    """Generator for C code.

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.c and <prefix>.h respectively.

    """

    code_extension = "c"
    interface_extension = "h"
    standard = 'c99'

    def __init__(self, project="project", printer=None,
                 preprocessor_statements=None, cse=False):
        super().__init__(project=project, cse=cse)
        self.printer = printer or c_code_printers[self.standard.lower()]()

        self.preprocessor_statements = preprocessor_statements
        if preprocessor_statements is None:
            self.preprocessor_statements = ['#include <math.h>']

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("/" + "*"*78 + '\n')
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append(" *%s*\n" % line.center(76))
        code_lines.append(" " + "*"*78 + "/\n")
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        if len(routine.results) > 1:
            raise CodeGenError("C only supports a single or no return value.")
        elif len(routine.results) == 1:
            ctype = routine.results[0].get_datatype('C')
        else:
            ctype = "void"

        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append((arg.get_datatype('C'), "*%s" % name))
            else:
                type_args.append((arg.get_datatype('C'), name))
        arguments = ", ".join([ "%s %s" % t for t in type_args])
        return "%s %s(%s)" % (ctype, routine.name, arguments)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        code_lines.append('#include "{}.h"'.format(os.path.basename(prefix)))
        code_lines.extend(self.preprocessor_statements)
        code_lines = ['{}\n'.format(l) for l in code_lines]
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ["%s {\n" % prototype]

    def _declare_arguments(self, routine):
        # arguments are declared in prototype
        return []

    def _declare_globals(self, routine):
        # global variables are not explicitly declared within C functions
        return []

    def _declare_locals(self, routine):

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        code_lines = []
        for result in routine.local_vars:

            # local variables that are simple symbols such as those used as indices into
            # for loops are defined declared elsewhere.
            if not isinstance(result, Result):
                continue

            if result.name != result.result_var:
                raise CodeGen("Result variable and name should match: {}".format(result))
            assign_to = result.name
            t = result.get_datatype('c')
            if isinstance(result.expr, (MatrixBase, MatrixExpr)):
                dims = result.expr.shape
                code_lines.append("{} {}[{}];\n".format(t, str(assign_to), dims[0]*dims[1]))
                prefix = ""
            else:
                prefix = "const {} ".format(t)

            constants, not_c, c_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "dereference": dereference, "strict": False},
                result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))

            code_lines.append("{}{}\n".format(prefix, c_expr))

        return code_lines

    def _call_printer(self, routine):
        code_lines = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        return_val = None
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name + "_result"
                t = result.get_datatype('c')
                code_lines.append("{} {};\n".format(t, str(assign_to)))
                return_val = assign_to
            else:
                assign_to = result.result_var

            try:
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)
            except AssignmentError:
                assign_to = result.result_var
                code_lines.append(
                    "%s %s;\n" % (result.get_datatype('c'), str(assign_to)))
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            code_lines.append("%s\n" % c_expr)

        if return_val:
            code_lines.append("   return %s;\n" % return_val)
        return code_lines

    def _get_routine_ending(self, routine):
        return ["}\n"]

    def dump_c(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)
    dump_c.extension = code_extension  # type: ignore
    dump_c.__doc__ = CodeGen.dump_code.__doc__

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the C header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to construct the include guards.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        guard_name = "%s__%s__H" % (self.project.replace(
            " ", "_").upper(), prefix.replace("/", "_").upper())
        # include guards
        if empty:
            print(file=f)
        print("#ifndef %s" % guard_name, file=f)
        print("#define %s" % guard_name, file=f)
        if empty:
            print(file=f)
        # declaration of the function prototypes
        for routine in routines:
            prototype = self.get_prototype(routine)
            print("%s;" % prototype, file=f)
        # end if include guards
        if empty:
            print(file=f)
        print("#endif", file=f)
        if empty:
            print(file=f)
    dump_h.extension = interface_extension  # type: ignore

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_c, dump_h]

class C89CodeGen(CCodeGen):
    standard = 'C89'

class C99CodeGen(CCodeGen):
    standard = 'C99'

class FCodeGen(CodeGen):
    """Generator for Fortran 95 code

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.f90 and <prefix>.h respectively.

    """

    code_extension = "f90"
    interface_extension = "h"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or FCodePrinter()

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("!" + "*"*78 + '\n')
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            code_lines.append("!*%s*\n" % line.center(76))
        code_lines.append("!" + "*"*78 + '\n')
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the fortran routine."""
        code_list = []
        if len(routine.results) > 1:
            raise CodeGenError(
                "Fortran only supports a single or no return value.")
        elif len(routine.results) == 1:
            result = routine.results[0]
            code_list.append(result.get_datatype('fortran'))
            code_list.append("function")
        else:
            code_list.append("subroutine")

        args = ", ".join("%s" % self._get_symbol(arg.name)
                        for arg in routine.arguments)

        call_sig = "{}({})\n".format(routine.name, args)
        # Fortran 95 requires all lines be less than 132 characters, so wrap
        # this line before appending.
        call_sig = ' &\n'.join(textwrap.wrap(call_sig,
                                             width=60,
                                             break_long_words=False)) + '\n'
        code_list.append(call_sig)
        code_list = [' '.join(code_list)]
        code_list.append('implicit none\n')
        return code_list

    def _declare_arguments(self, routine):
        # argument type declarations
        code_list = []
        array_list = []
        scalar_list = []
        for arg in routine.arguments:

            if isinstance(arg, InputArgument):
                typeinfo = "%s, intent(in)" % arg.get_datatype('fortran')
            elif isinstance(arg, InOutArgument):
                typeinfo = "%s, intent(inout)" % arg.get_datatype('fortran')
            elif isinstance(arg, OutputArgument):
                typeinfo = "%s, intent(out)" % arg.get_datatype('fortran')
            else:
                raise CodeGenError("Unknown Argument type: %s" % type(arg))

            fprint = self._get_symbol

            if arg.dimensions:
                # fortran arrays start at 1
                dimstr = ", ".join(["%s:%s" % (
                    fprint(dim[0] + 1), fprint(dim[1] + 1))
                    for dim in arg.dimensions])
                typeinfo += ", dimension(%s)" % dimstr
                array_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))
            else:
                scalar_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))

        # scalars first, because they can be used in array declarations
        code_list.extend(scalar_list)
        code_list.extend(array_list)

        return code_list

    def _declare_globals(self, routine):
        # Global variables not explicitly declared within Fortran 90 functions.
        # Note: a future F77 mode may need to generate "common" blocks.
        return []

    def _declare_locals(self, routine):
        code_list = []
        for var in sorted(routine.local_vars, key=str):
            typeinfo = get_default_datatype(var)
            code_list.append("%s :: %s\n" % (
                typeinfo.fname, self._get_symbol(var)))
        return code_list

    def _get_routine_ending(self, routine):
        """Returns the closing statements of the fortran routine."""
        if len(routine.results) == 1:
            return ["end function\n"]
        else:
            return ["end subroutine\n"]

    def get_interface(self, routine):
        """Returns a string for the function interface.

        The routine should have a single result object, which can be None.
        If the routine has multiple result objects, a CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        prototype = [ "interface\n" ]
        prototype.extend(self._get_routine_opening(routine))
        prototype.extend(self._declare_arguments(routine))
        prototype.extend(self._get_routine_ending(routine))
        prototype.append("end interface\n")

        return "".join(prototype)

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name
            elif isinstance(result, (OutputArgument, InOutArgument)):
                assign_to = result.result_var

            constants, not_fortran, f_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "source_format": 'free', "standard": 95, "strict": False},
                result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                t = get_default_datatype(obj)
                declarations.append(
                    "%s, parameter :: %s = %s\n" % (t.fname, obj, v))
            for obj in sorted(not_fortran, key=str):
                t = get_default_datatype(obj)
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("%s :: %s\n" % (t.fname, name))

            code_lines.append("%s\n" % f_expr)
        return declarations + code_lines

    def _indent_code(self, codelines):
        return self._printer_method_with_settings(
            'indent_code', {"human": False, "source_format": 'free', "strict": False}, codelines)

    def dump_f95(self, routines, f, prefix, header=True, empty=True):
        # check that symbols are unique with ignorecase
        for r in routines:
            lowercase = {str(x).lower() for x in r.variables}
            orig_case = {str(x) for x in r.variables}
            if len(lowercase) < len(orig_case):
                raise CodeGenError("Fortran ignores case. Got symbols: %s" %
                        (", ".join([str(var) for var in r.variables])))
        self.dump_code(routines, f, prefix, header, empty)
    dump_f95.extension = code_extension  # type: ignore
    dump_f95.__doc__ = CodeGen.dump_code.__doc__

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the interface to a header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        if empty:
            print(file=f)
        # declaration of the function prototypes
        for routine in routines:
            prototype = self.get_interface(routine)
            f.write(prototype)
        if empty:
            print(file=f)
    dump_h.extension = interface_extension  # type: ignore

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_f95, dump_h]


class JuliaCodeGen(CodeGen):
    """Generator for Julia code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.jl.

    """

    code_extension = "jl"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or JuliaCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Julia."""

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        old_symbols = expressions.free_symbols - local_vars - global_vars
        symbols = set()
        for s in old_symbols:
            if isinstance(s, Idx):
                symbols.update(s.args[1].free_symbols)
            elif not isinstance(s, Indexed):
                symbols.add(s)

        # Julia supports multiple return values
        return_vals = []
        output_args = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.One, dim) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                    output_args.append(InOutArgument(symbol, out_arg, expr, dimensions=dims))
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        output_args.sort(key=lambda x: str(x.name))
        arg_list = list(output_args)
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append("#\n")
            else:
                code_lines.append("#   %s\n" % line)
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []
        code_list.append("function ")

        # Inputs
        args = []
        for arg in routine.arguments:
            if isinstance(arg, OutputArgument):
                raise CodeGenError("Julia: invalid argument of type %s" %
                                   str(type(arg)))
            if isinstance(arg, (InputArgument, InOutArgument)):
                args.append("%s" % self._get_symbol(arg.name))
        args = ", ".join(args)
        code_list.append("%s(%s)\n" % (routine.name, args))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        outs = []
        for result in routine.results:
            if isinstance(result, Result):
                # Note: name not result_var; want `y` not `y[i]` for Indexed
                s = self._get_symbol(result.name)
            else:
                raise CodeGenError("unexpected object in Routine results")
            outs.append(s)
        return ["return " + ", ".join(outs) + "\nend\n"]

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, jl_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "%s = %s\n" % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "# unsupported: %s\n" % (name))
            code_lines.append("%s\n" % (jl_expr))
        return declarations + code_lines

    def _indent_code(self, codelines):
        # Note that indenting seems to happen twice, first
        # statement-by-statement by JuliaPrinter then again here.
        p = JuliaCodePrinter({'human': False, "strict": False})
        return p.indent_code(codelines)

    def dump_jl(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_jl.extension = code_extension  # type: ignore
    dump_jl.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_jl]


class OctaveCodeGen(CodeGen):
    """Generator for Octave code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.m.

    Octave .m files usually contain one function.  That function name should
    match the filename (``prefix``).  If you pass multiple ``name_expr`` pairs,
    the latter ones are presumed to be private functions accessed by the
    primary function.

    You should only pass inputs to ``argument_sequence``: outputs are ordered
    according to their order in ``name_expr``.

    """

    code_extension = "m"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or OctaveCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Octave."""

        # FIXME: this is probably general enough for other high-level
        # languages, perhaps its the C/Fortran one that is specialized!

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        old_symbols = expressions.free_symbols - local_vars - global_vars
        symbols = set()
        for s in old_symbols:
            if isinstance(s, Idx):
                symbols.update(s.args[1].free_symbols)
            elif not isinstance(s, Indexed):
                symbols.add(s)

        # Octave supports multiple return values
        return_vals = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    symbol = out_arg.base.label
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        arg_list = []
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append("%\n")
            else:
                code_lines.append("%%   %s\n" % line)
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []
        code_list.append("function ")

        # Outputs
        outs = []
        for result in routine.results:
            if isinstance(result, Result):
                # Note: name not result_var; want `y` not `y(i)` for Indexed
                s = self._get_symbol(result.name)
            else:
                raise CodeGenError("unexpected object in Routine results")
            outs.append(s)
        if len(outs) > 1:
            code_list.append("[" + (", ".join(outs)) + "]")
        else:
            code_list.append("".join(outs))
        code_list.append(" = ")

        # Inputs
        args = []
        for arg in routine.arguments:
            if isinstance(arg, (OutputArgument, InOutArgument)):
                raise CodeGenError("Octave: invalid argument of type %s" %
                                   str(type(arg)))
            if isinstance(arg, InputArgument):
                args.append("%s" % self._get_symbol(arg.name))
        args = ", ".join(args)
        code_list.append("%s(%s)\n" % (routine.name, args))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        if not routine.global_vars:
            return []
        s = " ".join(sorted([self._get_symbol(g) for g in routine.global_vars]))
        return ["global " + s + "\n"]

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        return ["end\n"]

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, oct_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "  %s = %s;  %% constant\n" % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "  %% unsupported: %s\n" % (name))
            code_lines.append("%s\n" % (oct_expr))
        return declarations + code_lines

    def _indent_code(self, codelines):
        return self._printer_method_with_settings(
            'indent_code', {"human": False, "strict": False}, codelines)

    def dump_m(self, routines, f, prefix, header=True, empty=True, inline=True):
        # Note used to call self.dump_code() but we need more control for header

        code_lines = self._preprocessor_statements(prefix)

        for i, routine in enumerate(routines):
            if i > 0:
                if empty:
                    code_lines.append("\n")
            code_lines.extend(self._get_routine_opening(routine))
            if i == 0:
                if routine.name != prefix:
                    raise ValueError('Octave function name should match prefix')
                if header:
                    code_lines.append("%" + prefix.upper() +
                                      "  Autogenerated by SymPy\n")
                    code_lines.append(''.join(self._get_header()))
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._call_printer(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_ending(routine))

        code_lines = self._indent_code(''.join(code_lines))

        if code_lines:
            f.write(code_lines)

    dump_m.extension = code_extension  # type: ignore
    dump_m.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_m]

class RustCodeGen(CodeGen):
    """Generator for Rust code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.rs

    """

    code_extension = "rs"

    def __init__(self, project="project", printer=None):
        super().__init__(project=project)
        self.printer = printer or RustCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Rust."""

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        symbols = expressions.free_symbols - local_vars - global_vars - expressions.atoms(Indexed)

        # Rust supports multiple return values
        return_vals = []
        output_args = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.One, dim) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                    output_args.append(InOutArgument(symbol, out_arg, expr, dimensions=dims))
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        output_args.sort(key=lambda x: str(x.name))
        arg_list = list(output_args)
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)


    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("/*\n")
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append((" *%s" % line.center(76)).rstrip() + "\n")
        code_lines.append(" */\n")
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        results = [i.get_datatype('Rust') for i in routine.results]

        if len(results) == 1:
            rstype = " -> " + results[0]
        elif len(routine.results) > 1:
            rstype = " -> (" + ", ".join(results) + ")"
        else:
            rstype = ""

        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append(("*%s" % name, arg.get_datatype('Rust')))
            else:
                type_args.append((name, arg.get_datatype('Rust')))
        arguments = ", ".join([ "%s: %s" % t for t in type_args])
        return "fn %s(%s)%s" % (routine.name, arguments, rstype)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        # code_lines.append("use std::f64::consts::*;\n")
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ["%s {\n" % prototype]

    def _declare_arguments(self, routine):
        # arguments are declared in prototype
        return []

    def _declare_globals(self, routine):
        # global variables are not explicitly declared within C functions
        return []

    def _declare_locals(self, routine):
        # loop variables are declared in loop statement
        return []

    def _call_printer(self, routine):

        code_lines = []
        declarations = []
        returns = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
                returns.append(str(result.result_var))
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, rs_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                declarations.append("const %s: f64 = %s;\n" % (name, value))

            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("// unsupported: %s\n" % (name))

            code_lines.append("let %s\n" % rs_expr)

        if len(returns) > 1:
            returns = ['(' + ', '.join(returns) + ')']

        returns.append('\n')

        return declarations + code_lines + returns

    def _get_routine_ending(self, routine):
        return ["}\n"]

    def dump_rs(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_rs.extension = code_extension  # type: ignore
    dump_rs.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_rs]




def get_code_generator(language, project=None, standard=None, printer = None):
    if language == 'C':
        if standard is None:
            pass
        elif standard.lower() == 'c89':
            language = 'C89'
        elif standard.lower() == 'c99':
            language = 'C99'
    CodeGenClass = {"C": CCodeGen, "C89": C89CodeGen, "C99": C99CodeGen,
                    "F95": FCodeGen, "JULIA": JuliaCodeGen,
                    "OCTAVE": OctaveCodeGen,
                    "RUST": RustCodeGen}.get(language.upper())
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    return CodeGenClass(project, printer)


#
# Friendly functions
#


def codegen(name_expr, language=None, prefix=None, project="project",
            to_files=False, header=True, empty=True, argument_sequence=None,
            global_vars=None, standard=None, code_gen=None, printer=None):
    """Generate source code for expressions in a given language.

    Parameters
    ==========

    name_expr : tuple, or list of tuples
        A single (name, expression) tuple or a list of (name, expression)
        tuples.  Each tuple corresponds to a routine.  If the expression is
        an equality (an instance of class Equality) the left hand side is
        considered an output argument.  If expression is an iterable, then
        the routine will have multiple outputs.

    language : string,
        A string that indicates the source code language.  This is case
        insensitive.  Currently, 'C', 'F95' and 'Octave' are supported.
        'Octave' generates code compatible with both Octave and Matlab.

    prefix : string, optional
        A prefix for the names of the files that contain the source code.
        Language-dependent suffixes will be appended.  If omitted, the name
        of the first name_expr tuple is used.

    project : string, optional
        A project name, used for making unique preprocessor instructions.
        [default: "project"]

    to_files : bool, optional
        When True, the code will be written to one or more files with the
        given prefix, otherwise strings with the names and contents of
        these files are returned. [default: False]

    header : bool, optional
        When True, a header is written on top of each source file.
        [default: True]

    empty : bool, optional
        When True, empty lines are used to structure the code.
        [default: True]

    argument_sequence : iterable, optional
        Sequence of arguments for the routine in a preferred order.  A
        CodeGenError is raised if required arguments are missing.
        Redundant arguments are used without warning.  If omitted,
        arguments will be ordered alphabetically, but with all input
        arguments first, and then output or in-out arguments.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    standard : string, optional

    code_gen : CodeGen instance, optional
        An instance of a CodeGen subclass. Overrides ``language``.

    printer : Printer instance, optional
        An instance of a Printer subclass.

    Examples
    ========

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...     ("f", x+y*z), "C89", "test", header=False, empty=False)
    >>> print(c_name)
    test.c
    >>> print(c_code)
    #include "test.h"
    #include <math.h>
    double f(double x, double y, double z) {
       double f_result;
       f_result = x + y*z;
       return f_result;
    }
    <BLANKLINE>
    >>> print(h_name)
    test.h
    >>> print(c_header)
    #ifndef PROJECT__TEST__H
    #define PROJECT__TEST__H
    double f(double x, double y, double z);
    #endif
    <BLANKLINE>

    Another example using Equality objects to give named outputs.  Here the
    filename (prefix) is taken from the first (name, expr) pair.

    >>> from sympy.abc import f, g
    >>> from sympy import Eq
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...      [("myfcn", x + y), ("fcn2", [Eq(f, 2*x), Eq(g, y)])],
    ...      "C99", header=False, empty=False)
    >>> print(c_name)
    myfcn.c
    >>> print(c_code)
    #include "myfcn.h"
    #include <math.h>
    double myfcn(double x, double y) {
       double myfcn_result;
       myfcn_result = x + y;
       return myfcn_result;
    }
    void fcn2(double x, double y, double *f, double *g) {
       (*f) = 2*x;
       (*g) = y;
    }
    <BLANKLINE>

    If the generated function(s) will be part of a larger project where various
    global variables have been defined, the 'global_vars' option can be used
    to remove the specified variables from the function signature

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(f_name, f_code), header] = codegen(
    ...     ("f", x+y*z), "F95", header=False, empty=False,
    ...     argument_sequence=(x, y), global_vars=(z,))
    >>> print(f_code)
    REAL*8 function f(x, y)
    implicit none
    REAL*8, intent(in) :: x
    REAL*8, intent(in) :: y
    f = x + y*z
    end function
    <BLANKLINE>

    """

    # Initialize the code generator.
    if language is None:
        if code_gen is None:
            raise ValueError("Need either language or code_gen")
    else:
        if code_gen is not None:
            raise ValueError("You cannot specify both language and code_gen.")
        code_gen = get_code_generator(language, project, standard, printer)

    if isinstance(name_expr[0], str):
        # single tuple is given, turn it into a singleton list with a tuple.
        name_expr = [name_expr]

    if prefix is None:
        prefix = name_expr[0][0]

    # Construct Routines appropriate for this code_gen from (name, expr) pairs.
    routines = []
    for name, expr in name_expr:
        routines.append(code_gen.routine(name, expr, argument_sequence,
                                         global_vars))

    # Write the code.
    return code_gen.write(routines, prefix, to_files, header, empty)


def make_routine(name, expr, argument_sequence=None,
                 global_vars=None, language="F95"):
    """A factory that makes an appropriate Routine from an expression.

    Parameters
    ==========

    name : string
        The name of this routine in the generated code.

    expr : expression or list/tuple of expressions
        A SymPy expression that the Routine instance will represent.  If
        given a list or tuple of expressions, the routine will be
        considered to have multiple return values and/or output arguments.

    argument_sequence : list or tuple, optional
        List arguments for the routine in a preferred order.  If omitted,
        the results are language dependent, for example, alphabetical order
        or in the same order as the given expressions.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    language : string, optional
        Specify a target language.  The Routine itself should be
        language-agnostic but the precise way one is created, error
        checking, etc depend on the language.  [default: "F95"].

    Notes
    =====

    A decision about whether to use output arguments or return values is made
    depending on both the language and the particular mathematical expressions.
    For an expression of type Equality, the left hand side is typically made
    into an OutputArgument (or perhaps an InOutArgument if appropriate).
    Otherwise, typically, the calculated expression is made a return values of
    the routine.

    Examples
    ========

    >>> from sympy.utilities.codegen import make_routine
    >>> from sympy.abc import x, y, f, g
    >>> from sympy import Eq
    >>> r = make_routine('test', [Eq(f, 2*x), Eq(g, x + y)])
    >>> [arg.result_var for arg in r.results]
    []
    >>> [arg.name for arg in r.arguments]
    [x, y, f, g]
    >>> [arg.name for arg in r.result_variables]
    [f, g]
    >>> r.local_vars
    set()

    Another more complicated example with a mixture of specified and
    automatically-assigned names.  Also has Matrix output.

    >>> from sympy import Matrix
    >>> r = make_routine('fcn', [x*y, Eq(f, 1), Eq(g, x + g), Matrix([[x, 2]])])
    >>> [arg.result_var for arg in r.results]  # doctest: +SKIP
    [result_5397460570204848505]
    >>> [arg.expr for arg in r.results]
    [x*y]
    >>> [arg.name for arg in r.arguments]  # doctest: +SKIP
    [x, y, f, g, out_8598435338387848786]

    We can examine the various arguments more closely:

    >>> from sympy.utilities.codegen import (InputArgument, OutputArgument,
    ...                                      InOutArgument)
    >>> [a.name for a in r.arguments if isinstance(a, InputArgument)]
    [x, y]

    >>> [a.name for a in r.arguments if isinstance(a, OutputArgument)]  # doctest: +SKIP
    [f, out_8598435338387848786]
    >>> [a.expr for a in r.arguments if isinstance(a, OutputArgument)]
    [1, Matrix([[x, 2]])]

    >>> [a.name for a in r.arguments if isinstance(a, InOutArgument)]
    [g]
    >>> [a.expr for a in r.arguments if isinstance(a, InOutArgument)]
    [g + x]

    """

    # initialize a new code generator
    code_gen = get_code_generator(language)

    return code_gen.routine(name, expr, argument_sequence, global_vars)
<|endoftext|>"""Module for compiling codegen output, and wrap the binary for use in
python.

.. note:: To use the autowrap module it must first be imported

   >>> from sympy.utilities.autowrap import autowrap

This module provides a common interface for different external backends, such
as f2py, fwrap, Cython, SWIG(?) etc. (Currently only f2py and Cython are
implemented) The goal is to provide access to compiled binaries of acceptable
performance with a one-button user interface, e.g.,

    >>> from sympy.abc import x,y
    >>> expr = (x - y)**25
    >>> flat = expr.expand()
    >>> binary_callable = autowrap(flat)
    >>> binary_callable(2, 3)
    -1.0

Although a SymPy user might primarily be interested in working with
mathematical expressions and not in the details of wrapping tools
needed to evaluate such expressions efficiently in numerical form,
the user cannot do so without some understanding of the
limits in the target language. For example, the expanded expression
contains large coefficients which result in loss of precision when
computing the expression:

    >>> binary_callable(3, 2)
    0.0
    >>> binary_callable(4, 5), binary_callable(5, 4)
    (-22925376.0, 25165824.0)

Wrapping the unexpanded expression gives the expected behavior:

    >>> e = autowrap(expr)
    >>> e(4, 5), e(5, 4)
    (-1.0, 1.0)

The callable returned from autowrap() is a binary Python function, not a
SymPy object.  If it is desired to use the compiled function in symbolic
expressions, it is better to use binary_function() which returns a SymPy
Function object.  The binary callable is attached as the _imp_ attribute and
invoked when a numerical evaluation is requested with evalf(), or with
lambdify().

    >>> from sympy.utilities.autowrap import binary_function
    >>> f = binary_function('f', expr)
    >>> 2*f(x, y) + y
    y + 2*f(x, y)
    >>> (2*f(x, y) + y).evalf(2, subs={x: 1, y:2})
    0.e-110

When is this useful?

    1) For computations on large arrays, Python iterations may be too slow,
       and depending on the mathematical expression, it may be difficult to
       exploit the advanced index operations provided by NumPy.

    2) For *really* long expressions that will be called repeatedly, the
       compiled binary should be significantly faster than SymPy's .evalf()

    3) If you are generating code with the codegen utility in order to use
       it in another project, the automatic Python wrappers let you test the
       binaries immediately from within SymPy.

    4) To create customized ufuncs for use with numpy arrays.
       See *ufuncify*.

When is this module NOT the best approach?

    1) If you are really concerned about speed or memory optimizations,
       you will probably get better results by working directly with the
       wrapper tools and the low level code.  However, the files generated
       by this utility may provide a useful starting point and reference
       code. Temporary files will be left intact if you supply the keyword
       tempdir="path/to/files/".

    2) If the array computation can be handled easily by numpy, and you
       do not need the binaries for another project.

"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from string import Template
from warnings import warn

from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,
                                     OutputArgument, InOutArgument,
                                     InputArgument, CodeGenArgumentListError,
                                     Result, ResultBase, C99CodeGen)
from sympy.utilities.iterables import iterable
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.decorator import doctest_depends_on

_doctest_depends_on = {'exe': ('f2py', 'gfortran', 'gcc'),
                       'modules': ('numpy',)}


class CodeWrapError(Exception):
    pass


class CodeWrapper:
    """Base Class for code wrappers"""
    _filename = "wrapped_code"
    _module_basename = "wrapper_module"
    _module_counter = 0

    @property
    def filename(self):
        return "%s_%s" % (self._filename, CodeWrapper._module_counter)

    @property
    def module_name(self):
        return "%s_%s" % (self._module_basename, CodeWrapper._module_counter)

    def __init__(self, generator, filepath=None, flags=[], verbose=False):
        """
        generator -- the code generator to use
        """
        self.generator = generator
        self.filepath = filepath
        self.flags = flags
        self.quiet = not verbose

    @property
    def include_header(self):
        return bool(self.filepath)

    @property
    def include_empty(self):
        return bool(self.filepath)

    def _generate_code(self, main_routine, routines):
        routines.append(main_routine)
        self.generator.write(
            routines, self.filename, True, self.include_header,
            self.include_empty)

    def wrap_code(self, routine, helpers=None):
        helpers = helpers or []
        if self.filepath:
            workdir = os.path.abspath(self.filepath)
        else:
            workdir = tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
            self._generate_code(routine, helpers)
            self._prepare_files(routine)
            self._process_files(routine)
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    # Could be some issues on Windows
                    pass

        return self._get_wrapped_function(mod, routine.name)

    def _process_files(self, routine):
        command = self.command
        command.extend(self.flags)
        try:
            retoutput = check_output(command, stderr=STDOUT)
        except CalledProcessError as e:
            raise CodeWrapError(
                "Error while executing command: %s. Command output is:\n%s" % (
                    " ".join(command), e.output.decode('utf-8')))
        if not self.quiet:
            print(retoutput)


class DummyWrapper(CodeWrapper):
    """Class used for testing independent of backends """

    template = """# dummy module for testing of SymPy
def %(name)s():
    return "%(expr)s"
%(name)s.args = "%(args)s"
%(name)s.returns = "%(retvals)s"
"""

    def _prepare_files(self, routine):
        return

    def _generate_code(self, routine, helpers):
        with open('%s.py' % self.module_name, 'w') as f:
            printed = ", ".join(
                [str(res.expr) for res in routine.result_variables])
            # convert OutputArguments to return value like f2py
            args = filter(lambda x: not isinstance(
                x, OutputArgument), routine.arguments)
            retvals = []
            for val in routine.result_variables:
                if isinstance(val, Result):
                    retvals.append('nameless')
                else:
                    retvals.append(val.result_var)

            print(DummyWrapper.template % {
                'name': routine.name,
                'expr': printed,
                'args': ", ".join([str(a.name) for a in args]),
                'retvals': ", ".join([str(val) for val in retvals])
            }, end="", file=f)

    def _process_files(self, routine):
        return

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)


class CythonCodeWrapper(CodeWrapper):
    """Wrapper that uses Cython"""

    setup_template = """\
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
cy_opts = {cythonize_options}
{np_import}
ext_mods = [Extension(
    {ext_args},
    include_dirs={include_dirs},
    library_dirs={library_dirs},
    libraries={libraries},
    extra_compile_args={extra_compile_args},
    extra_link_args={extra_link_args}
)]
setup(ext_modules=cythonize(ext_mods, **cy_opts))
"""

    _cythonize_options = {'compiler_directives':{'language_level' : "3"}}

    pyx_imports = (
        "import numpy as np\n"
        "cimport numpy as np\n\n")

    pyx_header = (
        "cdef extern from '{header_file}.h':\n"
        "    {prototype}\n\n")

    pyx_func = (
        "def {name}_c({arg_string}):\n"
        "\n"
        "{declarations}"
        "{body}")

    std_compile_flag = '-std=c99'

    def __init__(self, *args, **kwargs):
        """Instantiates a Cython code wrapper.

        The following optional parameters get passed to ``setuptools.Extension``
        for building the Python extension module. Read its documentation to
        learn more.

        Parameters
        ==========
        include_dirs : [list of strings]
            A list of directories to search for C/C++ header files (in Unix
            form for portability).
        library_dirs : [list of strings]
            A list of directories to search for C/C++ libraries at link time.
        libraries : [list of strings]
            A list of library names (not filenames or paths) to link against.
        extra_compile_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            compiling the source files in 'sources'.  For platforms and
            compilers where "command line" makes sense, this is typically a
            list of command-line arguments, but for other platforms it could be
            anything. Note that the attribute ``std_compile_flag`` will be
            appended to this list.
        extra_link_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            linking object files together to create the extension (or to create
            a new static Python interpreter). Similar interpretation as for
            'extra_compile_args'.
        cythonize_options : [dictionary]
            Keyword arguments passed on to cythonize.

        """

        self._include_dirs = kwargs.pop('include_dirs', [])
        self._library_dirs = kwargs.pop('library_dirs', [])
        self._libraries = kwargs.pop('libraries', [])
        self._extra_compile_args = kwargs.pop('extra_compile_args', [])
        self._extra_compile_args.append(self.std_compile_flag)
        self._extra_link_args = kwargs.pop('extra_link_args', [])
        self._cythonize_options = kwargs.pop('cythonize_options', self._cythonize_options)

        self._need_numpy = False

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command

    def _prepare_files(self, routine, build_dir=os.curdir):
        # NOTE : build_dir is used for testing purposes.
        pyxfilename = self.module_name + '.pyx'
        codefilename = "%s.%s" % (self.filename, self.generator.code_extension)

        # pyx
        with open(os.path.join(build_dir, pyxfilename), 'w') as f:
            self.dump_pyx([routine], f, self.filename)

        # setup.py
        ext_args = [repr(self.module_name), repr([pyxfilename, codefilename])]
        if self._need_numpy:
            np_import = 'import numpy as np\n'
            self._include_dirs.append('np.get_include()')
        else:
            np_import = ''

        includes = str(self._include_dirs).replace("'np.get_include()'",
                                                    'np.get_include()')
        code = self.setup_template.format(
                ext_args=", ".join(ext_args),
                np_import=np_import,
                include_dirs=includes,
                library_dirs=self._library_dirs,
                libraries=self._libraries,
                extra_compile_args=self._extra_compile_args,
                extra_link_args=self._extra_link_args,
                cythonize_options=self._cythonize_options)
        Path(os.path.join(build_dir, 'setup.py')).write_text(code)

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name + '_c')

    def dump_pyx(self, routines, f, prefix):
        """Write a Cython file with Python wrappers

        This file contains all the definitions of the routines in c code and
        refers to the header file.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.
        """
        headers = []
        functions = []
        for routine in routines:
            prototype = self.generator.get_prototype(routine)

            # C Function Header Import
            headers.append(self.pyx_header.format(header_file=prefix,
                                                  prototype=prototype))

            # Partition the C function arguments into categories
            py_rets, py_args, py_loc, py_inf = self._partition_args(routine.arguments)

            # Function prototype
            name = routine.name
            arg_string = ", ".join(self._prototype_arg(arg) for arg in py_args)

            # Local Declarations
            local_decs = []
            for arg, val in py_inf.items():
                proto = self._prototype_arg(arg)
                mat, ind = [self._string_var(v) for v in val]
                local_decs.append("    cdef {} = {}.shape[{}]".format(proto, mat, ind))
            local_decs.extend(["    cdef {}".format(self._declare_arg(a)) for a in py_loc])
            declarations = "\n".join(local_decs)
            if declarations:
                declarations = declarations + "\n"

            # Function Body
            args_c = ", ".join([self._call_arg(a) for a in routine.arguments])
            rets = ", ".join([self._string_var(r.name) for r in py_rets])
            if routine.results:
                body = '    return %s(%s)' % (routine.name, args_c)
                if rets:
                    body = body + ', ' + rets
            else:
                body = '    %s(%s)\n' % (routine.name, args_c)
                body = body + '    return ' + rets

            functions.append(self.pyx_func.format(name=name, arg_string=arg_string,
                    declarations=declarations, body=body))

        # Write text to file
        if self._need_numpy:
            # Only import numpy if required
            f.write(self.pyx_imports)
        f.write('\n'.join(headers))
        f.write('\n'.join(functions))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_args = []
        py_returns = []
        py_locals = []
        py_inferred = {}
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_returns.append(arg)
                py_locals.append(arg)
            elif isinstance(arg, InOutArgument):
                py_returns.append(arg)
                py_args.append(arg)
            else:
                py_args.append(arg)
        # Find arguments that are array dimensions. These can be inferred
        # locally in the Cython code.
            if isinstance(arg, (InputArgument, InOutArgument)) and arg.dimensions:
                dims = [d[1] + 1 for d in arg.dimensions]
                sym_dims = [(i, d) for (i, d) in enumerate(dims) if
                            isinstance(d, Symbol)]
                for (i, d) in sym_dims:
                    py_inferred[d] = (arg.name, i)
        for arg in args:
            if arg.name in py_inferred:
                py_inferred[arg] = py_inferred.pop(arg.name)
        # Filter inferred arguments from py_args
        py_args = [a for a in py_args if a not in py_inferred]
        return py_returns, py_args, py_locals, py_inferred

    def _prototype_arg(self, arg):
        mat_dec = "np.ndarray[{mtype}, ndim={ndim}] {name}"
        np_types = {'double': 'np.double_t',
                    'int': 'np.int_t'}
        t = arg.get_datatype('c')
        if arg.dimensions:
            self._need_numpy = True
            ndim = len(arg.dimensions)
            mtype = np_types[t]
            return mat_dec.format(mtype=mtype, ndim=ndim, name=self._string_var(arg.name))
        else:
            return "%s %s" % (t, self._string_var(arg.name))

    def _declare_arg(self, arg):
        proto = self._prototype_arg(arg)
        if arg.dimensions:
            shape = '(' + ','.join(self._string_var(i[1] + 1) for i in arg.dimensions) + ')'
            return proto + " = np.empty({shape})".format(shape=shape)
        else:
            return proto + " = 0"

    def _call_arg(self, arg):
        if arg.dimensions:
            t = arg.get_datatype('c')
            return "<{}*> {}.data".format(t, self._string_var(arg.name))
        elif isinstance(arg, ResultBase):
            return "&{}".format(self._string_var(arg.name))
        else:
            return self._string_var(arg.name)

    def _string_var(self, var):
        printer = self.generator.printer.doprint
        return printer(var)


class F2PyCodeWrapper(CodeWrapper):
    """Wrapper that uses f2py"""

    def __init__(self, *args, **kwargs):

        ext_keys = ['include_dirs', 'library_dirs', 'libraries',
                    'extra_compile_args', 'extra_link_args']
        msg = ('The compilation option kwarg {} is not supported with the f2py '
               'backend.')

        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            kwargs.pop(k, None)

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        filename = self.filename + '.' + self.generator.code_extension
        args = ['-c', '-m', self.module_name, filename]
        command = [sys.executable, "-c", "import numpy.f2py as f2py2e;f2py2e.main()"]+args
        return command

    def _prepare_files(self, routine):
        pass

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)


# Here we define a lookup of backends -> tuples of languages. For now, each
# tuple is of length 1, but if a backend supports more than one language,
# the most preferable language is listed first.
_lang_lookup = {'CYTHON': ('C99', 'C89', 'C'),
                'F2PY': ('F95',),
                'NUMPY': ('C99', 'C89', 'C'),
                'DUMMY': ('F95',)}     # Dummy here just for testing


def _infer_language(backend):
    """For a given backend, return the top choice of language"""
    langs = _lang_lookup.get(backend.upper(), False)
    if not langs:
        raise ValueError("Unrecognized backend: " + backend)
    return langs[0]


def _validate_backend_language(backend, language):
    """Throws error if backend and language are incompatible"""
    langs = _lang_lookup.get(backend.upper(), False)
    if not langs:
        raise ValueError("Unrecognized backend: " + backend)
    if language.upper() not in langs:
        raise ValueError(("Backend {} and language {} are "
                          "incompatible").format(backend, language))


@cacheit
@doctest_depends_on(exe=('f2py', 'gfortran'), modules=('numpy',))
def autowrap(expr, language=None, backend='f2py', tempdir=None, args=None,
             flags=None, verbose=False, helpers=None, code_gen=None, **kwargs):
    """Generates Python callable binaries based on the math expression.

    Parameters
    ==========

    expr
        The SymPy expression that should be wrapped as a binary routine.
    language : string, optional
        If supplied, (options: 'C' or 'F95'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either 'f2py' [default],
        or 'cython'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in the
        specified path.
    args : iterable, optional
        An ordered iterable of symbols. Specifies the argument sequence for the
        function.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can be
        helpful for debugging.
    helpers : 3-tuple or iterable of 3-tuples, optional
        Used to define auxiliary functions needed for the main expression.
        Each tuple should be of the form (name, expr, args) where:

        - name : str, the function name
        - expr : sympy expression, the function
        - args : iterable, the function arguments (can be any iterable of symbols)

    code_gen : CodeGen instance
        An instance of a CodeGen subclass. Overrides ``language``.
    include_dirs : [string]
        A list of directories to search for C/C++ header files (in Unix form
        for portability).
    library_dirs : [string]
        A list of directories to search for C/C++ libraries at link time.
    libraries : [string]
        A list of library names (not filenames or paths) to link against.
    extra_compile_args : [string]
        Any extra platform- and compiler-specific information to use when
        compiling the source files in 'sources'.  For platforms and compilers
        where "command line" makes sense, this is typically a list of
        command-line arguments, but for other platforms it could be anything.
    extra_link_args : [string]
        Any extra platform- and compiler-specific information to use when
        linking object files together to create the extension (or to create a
        new static Python interpreter).  Similar interpretation as for
        'extra_compile_args'.

    Examples
    ========

    Basic usage:

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.autowrap import autowrap
    >>> expr = ((x - y + z)**(13)).expand()
    >>> binary_func = autowrap(expr)
    >>> binary_func(1, 4, 2)
    -1.0

    Using helper functions:

    >>> from sympy.abc import x, t
    >>> from sympy import Function
    >>> helper_func = Function('helper_func')  # Define symbolic function
    >>> expr = 3*x + helper_func(t)  # Main expression using helper function
    >>> # Define helper_func(x) = 4*x using f2py backend
    >>> binary_func = autowrap(expr, args=[x, t],
    ...                       helpers=('helper_func', 4*x, [x]))
    >>> binary_func(2, 5)  # 3*2 + helper_func(5) = 6 + 20
    26.0
    >>> # Same example using cython backend
    >>> binary_func = autowrap(expr, args=[x, t], backend='cython',
    ...                       helpers=[('helper_func', 4*x, [x])])
    >>> binary_func(2, 5)  # 3*2 + helper_func(5) = 6 + 20
    26.0

    Type handling example:

    >>> import numpy as np
    >>> expr = x + y
    >>> f_cython = autowrap(expr, backend='cython')
    >>> f_cython(1, 2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Argument '_x' has incorrect type (expected numpy.ndarray, got int)
    >>> f_cython(np.array([1.0]), np.array([2.0]))
    array([ 3.])

    """
    if language:
        if not isinstance(language, type):
            _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    # two cases 1) helpers is an iterable of 3-tuples and 2) helpers is a
    # 3-tuple
    if iterable(helpers) and len(helpers) != 0 and iterable(helpers[0]):
        helpers = helpers if helpers else ()
    else:
        helpers = [helpers] if helpers else ()
    args = list(args) if iterable(args, exclude=set) else args

    if code_gen is None:
        code_gen = get_code_generator(language, "autowrap")

    CodeWrapperClass = {
        'F2PY': F2PyCodeWrapper,
        'CYTHON': CythonCodeWrapper,
        'DUMMY': DummyWrapper
    }[backend.upper()]
    code_wrapper = CodeWrapperClass(code_gen, tempdir, flags if flags else (),
                                    verbose, **kwargs)

    helps = []
    for name_h, expr_h, args_h in helpers:
        helps.append(code_gen.routine(name_h, expr_h, args_h))

    for name_h, expr_h, args_h in helpers:
        if expr.has(expr_h):
            name_h = binary_function(name_h, expr_h, backend='dummy')
            expr = expr.subs(expr_h, name_h(*args_h))
    try:
        routine = code_gen.routine('autofunc', expr, args)
    except CodeGenArgumentListError as e:
        # if all missing arguments are for pure output, we simply attach them
        # at the end and try again, because the wrappers will silently convert
        # them to return values anyway.
        new_args = []
        for missing in e.missing_args:
            if not isinstance(missing, OutputArgument):
                raise
            new_args.append(missing.name)
        routine = code_gen.routine('autofunc', expr, args + new_args)

    return code_wrapper.wrap_code(routine, helpers=helps)


@doctest_depends_on(exe=('f2py', 'gfortran'), modules=('numpy',))
def binary_function(symfunc, expr, **kwargs):
    """Returns a SymPy function with expr as binary implementation

    This is a convenience function that automates the steps needed to
    autowrap the SymPy expression and attaching it to a Function object
    with implemented_function().

    Parameters
    ==========

    symfunc : SymPy Function
        The function to bind the callable to.
    expr : SymPy Expression
        The expression used to generate the function.
    kwargs : dict
        Any kwargs accepted by autowrap.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.utilities.autowrap import binary_function
    >>> expr = ((x - y)**(25)).expand()
    >>> f = binary_function('f', expr)
    >>> type(f)
    <class 'sympy.core.function.UndefinedFunction'>
    >>> 2*f(x, y)
    2*f(x, y)
    >>> f(x, y).evalf(2, subs={x: 1, y: 2})
    -1.0

    """
    binary = autowrap(expr, **kwargs)
    return implemented_function(symfunc, binary)

#################################################################
#                           UFUNCIFY                            #
#################################################################

_ufunc_top = Template("""\
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include ${include_file}

static PyMethodDef ${module}Methods[] = {
        {NULL, NULL, 0, NULL}
};""")

_ufunc_outcalls = Template("*((double *)out${outnum}) = ${funcname}(${call_args});")

_ufunc_body = Template("""\
#ifdef NPY_1_19_API_VERSION
static void ${funcname}_ufunc(char **args, const npy_intp *dimensions, const npy_intp* steps, void* data)
#else
static void ${funcname}_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
#endif
{
    npy_intp i;
    npy_intp n = dimensions[0];
    ${declare_args}
    ${declare_steps}
    for (i = 0; i < n; i++) {
        ${outcalls}
        ${step_increments}
    }
}
PyUFuncGenericFunction ${funcname}_funcs[1] = {&${funcname}_ufunc};
static char ${funcname}_types[${n_types}] = ${types}
static void *${funcname}_data[1] = {NULL};""")

_ufunc_bottom = Template("""\
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "${module}",
    NULL,
    -1,
    ${module}Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
    return m;
}
#else
PyMODINIT_FUNC init${module}(void)
{
    PyObject *m, *d;
    ${function_creation}
    m = Py_InitModule("${module}", ${module}Methods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ${ufunc_init}
}
#endif\
""")

_ufunc_init_form = Template("""\
ufunc${ind} = PyUFunc_FromFuncAndData(${funcname}_funcs, ${funcname}_data, ${funcname}_types, 1, ${n_in}, ${n_out},
            PyUFunc_None, "${module}", ${docstring}, 0);
    PyDict_SetItemString(d, "${funcname}", ufunc${ind});
    Py_DECREF(ufunc${ind});""")

_ufunc_setup = Template("""\
from setuptools.extension import Extension
from setuptools import setup

from numpy import get_include

if __name__ == "__main__":
    setup(ext_modules=[
        Extension('${module}',
                  sources=['${module}.c', '${filename}.c'],
                  include_dirs=[get_include()])])
""")


class UfuncifyCodeWrapper(CodeWrapper):
    """Wrapper for Ufuncify"""

    def __init__(self, *args, **kwargs):

        ext_keys = ['include_dirs', 'library_dirs', 'libraries',
                    'extra_compile_args', 'extra_link_args']
        msg = ('The compilation option kwarg {} is not supported with the numpy'
               ' backend.')

        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            kwargs.pop(k, None)

        super().__init__(*args, **kwargs)

    @property
    def command(self):
        command = [sys.executable, "setup.py", "build_ext", "--inplace"]
        return command

    def wrap_code(self, routines, helpers=None):
        # This routine overrides CodeWrapper because we can't assume funcname == routines[0].name
        # Therefore we have to break the CodeWrapper private API.
        # There isn't an obvious way to extend multi-expr support to
        # the other autowrap backends, so we limit this change to ufuncify.
        helpers = helpers if helpers is not None else []
        # We just need a consistent name
        funcname = 'wrapped_' + str(id(routines) + id(helpers))

        workdir = self.filepath or tempfile.mkdtemp("_sympy_compile")
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
            self._generate_code(routines, helpers)
            self._prepare_files(routines, funcname)
            self._process_files(routines)
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    # Could be some issues on Windows
                    pass

        return self._get_wrapped_function(mod, funcname)

    def _generate_code(self, main_routines, helper_routines):
        all_routines = main_routines + helper_routines
        self.generator.write(
            all_routines, self.filename, True, self.include_header,
            self.include_empty)

    def _prepare_files(self, routines, funcname):

        # C
        codefilename = self.module_name + '.c'
        with open(codefilename, 'w') as f:
            self.dump_c(routines, f, self.filename, funcname=funcname)

        # setup.py
        with open('setup.py', 'w') as f:
            self.dump_setup(f)

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)

    def dump_setup(self, f):
        setup = _ufunc_setup.substitute(module=self.module_name,
                                        filename=self.filename)
        f.write(setup)

    def dump_c(self, routines, f, prefix, funcname=None):
        """Write a C file with Python wrappers

        This file contains all the definitions of the routines in c code.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to name the imported module.
        funcname
            Name of the main function to be returned.
        """
        if funcname is None:
            if len(routines) == 1:
                funcname = routines[0].name
            else:
                msg = 'funcname must be specified for multiple output routines'
                raise ValueError(msg)
        functions = []
        function_creation = []
        ufunc_init = []
        module = self.module_name
        include_file = "\"{}.h\"".format(prefix)
        top = _ufunc_top.substitute(include_file=include_file, module=module)

        name = funcname

        # Partition the C function arguments into categories
        # Here we assume all routines accept the same arguments
        r_index = 0
        py_in, _ = self._partition_args(routines[0].arguments)
        n_in = len(py_in)
        n_out = len(routines)

        # Declare Args
        form = "char *{0}{1} = args[{2}];"
        arg_decs = [form.format('in', i, i) for i in range(n_in)]
        arg_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_args = '\n    '.join(arg_decs)

        # Declare Steps
        form = "npy_intp {0}{1}_step = steps[{2}];"
        step_decs = [form.format('in', i, i) for i in range(n_in)]
        step_decs.extend([form.format('out', i, i+n_in) for i in range(n_out)])
        declare_steps = '\n    '.join(step_decs)

        # Call Args
        form = "*(double *)in{0}"
        call_args = ', '.join([form.format(a) for a in range(n_in)])

        # Step Increments
        form = "{0}{1} += {0}{1}_step;"
        step_incs = [form.format('in', i) for i in range(n_in)]
        step_incs.extend([form.format('out', i, i) for i in range(n_out)])
        step_increments = '\n        '.join(step_incs)

        # Types
        n_types = n_in + n_out
        types = "{" + ', '.join(["NPY_DOUBLE"]*n_types) + "};"

        # Docstring
        docstring = '"Created in SymPy with Ufuncify"'

        # Function Creation
        function_creation.append("PyObject *ufunc{};".format(r_index))

        # Ufunc initialization
        init_form = _ufunc_init_form.substitute(module=module,
                                                funcname=name,
                                                docstring=docstring,
                                                n_in=n_in, n_out=n_out,
                                                ind=r_index)
        ufunc_init.append(init_form)

        outcalls = [_ufunc_outcalls.substitute(
            outnum=i, call_args=call_args, funcname=routines[i].name) for i in
            range(n_out)]

        body = _ufunc_body.substitute(module=module, funcname=name,
                                      declare_args=declare_args,
                                      declare_steps=declare_steps,
                                      call_args=call_args,
                                      step_increments=step_increments,
                                      n_types=n_types, types=types,
                                      outcalls='\n        '.join(outcalls))
        functions.append(body)

        body = '\n\n'.join(functions)
        ufunc_init = '\n    '.join(ufunc_init)
        function_creation = '\n    '.join(function_creation)
        bottom = _ufunc_bottom.substitute(module=module,
                                          ufunc_init=ufunc_init,
                                          function_creation=function_creation)
        text = [top, body, bottom]
        f.write('\n\n'.join(text))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_in = []
        py_out = []
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_out.append(arg)
            elif isinstance(arg, InOutArgument):
                raise ValueError("Ufuncify doesn't support InOutArguments")
            else:
                py_in.append(arg)
        return py_in, py_out


@cacheit
@doctest_depends_on(exe=('f2py', 'gfortran', 'gcc'), modules=('numpy',))
def ufuncify(args, expr, language=None, backend='numpy', tempdir=None,
             flags=None, verbose=False, helpers=None, **kwargs):
    """Generates a binary function that supports broadcasting on numpy arrays.

    Parameters
    ==========

    args : iterable
        Either a Symbol or an iterable of symbols. Specifies the argument
        sequence for the function.
    expr
        A SymPy expression that defines the element wise operation.
    language : string, optional
        If supplied, (options: 'C' or 'F95'), specifies the language of the
        generated code. If ``None`` [default], the language is inferred based
        upon the specified backend.
    backend : string, optional
        Backend used to wrap the generated code. Either 'numpy' [default],
        'cython', or 'f2py'.
    tempdir : string, optional
        Path to directory for temporary files. If this argument is supplied,
        the generated code and the wrapper input files are left intact in
        the specified path.
    flags : iterable, optional
        Additional option flags that will be passed to the backend.
    verbose : bool, optional
        If True, autowrap will not mute the command line backends. This can
        be helpful for debugging.
    helpers : 3-tuple or iterable of 3-tuples, optional
        Used to define auxiliary functions needed for the main expression.
        Each tuple should be of the form (name, expr, args) where:

        - name : str, the function name
        - expr : sympy expression, the function
        - args : iterable, the function arguments (can be any iterable of symbols)

    kwargs : dict
        These kwargs will be passed to autowrap if the `f2py` or `cython`
        backend is used and ignored if the `numpy` backend is used.

    Notes
    =====

    The default backend ('numpy') will create actual instances of
    ``numpy.ufunc``. These support ndimensional broadcasting, and implicit type
    conversion. Use of the other backends will result in a "ufunc-like"
    function, which requires equal length 1-dimensional arrays for all
    arguments, and will not perform any type conversions.

    References
    ==========

    .. [1] https://numpy.org/doc/stable/reference/ufuncs.html

    Examples
    ========

    Basic usage:

    >>> from sympy.utilities.autowrap import ufuncify
    >>> from sympy.abc import x, y
    >>> import numpy as np
    >>> f = ufuncify((x, y), y + x**2)
    >>> type(f)
    <class 'numpy.ufunc'>
    >>> f([1, 2, 3], 2)
    array([  3.,   6.,  11.])
    >>> f(np.arange(5), 3)
    array([  3.,   4.,   7.,  12.,  19.])

    Using helper functions:

    >>> from sympy import Function
    >>> helper_func = Function('helper_func')  # Define symbolic function
    >>> expr = x**2 + y*helper_func(x)  # Main expression using helper function
    >>> # Define helper_func(x) = x**3
    >>> f = ufuncify((x, y), expr, helpers=[('helper_func', x**3, [x])])
    >>> f([1, 2], [3, 4])
    array([  4.,  36.])

    Type handling with different backends:

    For the 'f2py' and 'cython' backends, inputs are required to be equal length
    1-dimensional arrays. The 'f2py' backend will perform type conversion, but
    the Cython backend will error if the inputs are not of the expected type.

    >>> f_fortran = ufuncify((x, y), y + x**2, backend='f2py')
    >>> f_fortran(1, 2)
    array([ 3.])
    >>> f_fortran(np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
    array([  2.,   6.,  12.])
    >>> f_cython = ufuncify((x, y), y + x**2, backend='Cython')
    >>> f_cython(1, 2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Argument '_x' has incorrect type (expected numpy.ndarray, got int)
    >>> f_cython(np.array([1.0]), np.array([2.0]))
    array([ 3.])

    """

    if isinstance(args, Symbol):
        args = (args,)
    else:
        args = tuple(args)

    if language:
        _validate_backend_language(backend, language)
    else:
        language = _infer_language(backend)

    helpers = helpers if helpers else ()
    flags = flags if flags else ()

    if backend.upper() == 'NUMPY':
        # maxargs is set by numpy compile-time constant NPY_MAXARGS
        # If a future version of numpy modifies or removes this restriction
        # this variable should be changed or removed
        maxargs = 32
        helps = []
        for name, expr, args in helpers:
            helps.append(make_routine(name, expr, args))
        code_wrapper = UfuncifyCodeWrapper(C99CodeGen("ufuncify"), tempdir,
                                           flags, verbose)
        if not isinstance(expr, (list, tuple)):
            expr = [expr]
        if len(expr) == 0:
            raise ValueError('Expression iterable has zero length')
        if len(expr) + len(args) > maxargs:
            msg = ('Cannot create ufunc with more than {0} total arguments: '
                   'got {1} in, {2} out')
            raise ValueError(msg.format(maxargs, len(args), len(expr)))
        routines = [make_routine('autofunc{}'.format(idx), exprx, args) for
                    idx, exprx in enumerate(expr)]
        return code_wrapper.wrap_code(routines, helpers=helps)
    else:
        # Dummies are used for all added expressions to prevent name clashes
        # within the original expression.
        y = IndexedBase(Dummy('y'))
        m = Dummy('m', integer=True)
        i = Idx(Dummy('i', integer=True), m)
        f_dummy = Dummy('f')
        f = implemented_function('%s_%d' % (f_dummy.name, f_dummy.dummy_index), Lambda(args, expr))
        # For each of the args create an indexed version.
        indexed_args = [IndexedBase(Dummy(str(a))) for a in args]
        # Order the arguments (out, args, dim)
        args = [y] + indexed_args + [m]
        args_with_indices = [a[i] for a in indexed_args]
        return autowrap(Eq(y[i], f(*args_with_indices)), language, backend,
                        tempdir, args, flags, verbose, helpers, **kwargs)
<|endoftext|>"""Useful utility decorators. """

from typing import TypeVar
import sys
import types
import inspect
from functools import wraps, update_wrapper

from sympy.utilities.exceptions import sympy_deprecation_warning


T = TypeVar('T')
"""A generic type"""


def threaded_factory(func, use_add):
    """A factory for ``threaded`` decorators. """
    from sympy.core import sympify
    from sympy.matrices import MatrixBase
    from sympy.utilities.iterables import iterable

    @wraps(func)
    def threaded_func(expr, *args, **kwargs):
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(lambda f: func(f, *args, **kwargs))
        elif iterable(expr):
            try:
                return expr.__class__([func(f, *args, **kwargs) for f in expr])
            except TypeError:
                return expr
        else:
            expr = sympify(expr)

            if use_add and expr.is_Add:
                return expr.__class__(*[ func(f, *args, **kwargs) for f in expr.args ])
            elif expr.is_Relational:
                return expr.__class__(func(expr.lhs, *args, **kwargs),
                                      func(expr.rhs, *args, **kwargs))
            else:
                return func(expr, *args, **kwargs)

    return threaded_func


def threaded(func):
    """Apply ``func`` to sub--elements of an object, including :class:`~.Add`.

    This decorator is intended to make it uniformly possible to apply a
    function to all elements of composite objects, e.g. matrices, lists, tuples
    and other iterable containers, or just expressions.

    This version of :func:`threaded` decorator allows threading over
    elements of :class:`~.Add` class. If this behavior is not desirable
    use :func:`xthreaded` decorator.

    Functions using this decorator must have the following signature::

      @threaded
      def function(expr, *args, **kwargs):

    """
    return threaded_factory(func, True)


def xthreaded(func):
    """Apply ``func`` to sub--elements of an object, excluding :class:`~.Add`.

    This decorator is intended to make it uniformly possible to apply a
    function to all elements of composite objects, e.g. matrices, lists, tuples
    and other iterable containers, or just expressions.

    This version of :func:`threaded` decorator disallows threading over
    elements of :class:`~.Add` class. If this behavior is not desirable
    use :func:`threaded` decorator.

    Functions using this decorator must have the following signature::

      @xthreaded
      def function(expr, *args, **kwargs):

    """
    return threaded_factory(func, False)


def conserve_mpmath_dps(func):
    """After the function finishes, resets the value of ``mpmath.mp.dps`` to
    the value it had before the function was run."""
    import mpmath

    def func_wrapper(*args, **kwargs):
        dps = mpmath.mp.dps
        try:
            return func(*args, **kwargs)
        finally:
            mpmath.mp.dps = dps

    func_wrapper = update_wrapper(func_wrapper, func)
    return func_wrapper


class no_attrs_in_subclass:
    """Don't 'inherit' certain attributes from a base class

    >>> from sympy.utilities.decorator import no_attrs_in_subclass

    >>> class A(object):
    ...     x = 'test'

    >>> A.x = no_attrs_in_subclass(A, A.x)

    >>> class B(A):
    ...     pass

    >>> hasattr(A, 'x')
    True
    >>> hasattr(B, 'x')
    False

    """
    def __init__(self, cls, f):
        self.cls = cls
        self.f = f

    def __get__(self, instance, owner=None):
        if owner == self.cls:
            if hasattr(self.f, '__get__'):
                return self.f.__get__(instance, owner)
            return self.f
        raise AttributeError


def doctest_depends_on(exe=None, modules=None, disable_viewers=None,
                       python_version=None, ground_types=None):
    """
    Adds metadata about the dependencies which need to be met for doctesting
    the docstrings of the decorated objects.

    ``exe`` should be a list of executables

    ``modules`` should be a list of modules

    ``disable_viewers`` should be a list of viewers for :func:`~sympy.printing.preview.preview` to disable

    ``python_version`` should be the minimum Python version required, as a tuple
    (like ``(3, 0)``)
    """
    dependencies = {}
    if exe is not None:
        dependencies['executables'] = exe
    if modules is not None:
        dependencies['modules'] = modules
    if disable_viewers is not None:
        dependencies['disable_viewers'] = disable_viewers
    if python_version is not None:
        dependencies['python_version'] = python_version
    if ground_types is not None:
        dependencies['ground_types'] = ground_types

    def skiptests():
        from sympy.testing.runtests import DependencyError, SymPyDocTests, PyTestReporter # lazy import
        r = PyTestReporter()
        t = SymPyDocTests(r, None)
        try:
            t._check_dependencies(**dependencies)
        except DependencyError:
            return True  # Skip doctests
        else:
            return False # Run doctests

    def depends_on_deco(fn):
        fn._doctest_depends_on = dependencies
        fn.__doctest_skip__ = skiptests

        if inspect.isclass(fn):
            fn._doctest_depdends_on = no_attrs_in_subclass(
                fn, fn._doctest_depends_on)
            fn.__doctest_skip__ = no_attrs_in_subclass(
                fn, fn.__doctest_skip__)
        return fn

    return depends_on_deco


def public(obj: T) -> T:
    """
    Append ``obj``'s name to global ``__all__`` variable (call site).

    By using this decorator on functions or classes you achieve the same goal
    as by filling ``__all__`` variables manually, you just do not have to repeat
    yourself (object's name). You also know if object is public at definition
    site, not at some random location (where ``__all__`` was set).

    Note that in multiple decorator setup (in almost all cases) ``@public``
    decorator must be applied before any other decorators, because it relies
    on the pointer to object's global namespace. If you apply other decorators
    first, ``@public`` may end up modifying the wrong namespace.

    Examples
    ========

    >>> from sympy.utilities.decorator import public

    >>> __all__ # noqa: F821
    Traceback (most recent call last):
    ...
    NameError: name '__all__' is not defined

    >>> @public
    ... def some_function():
    ...     pass

    >>> __all__ # noqa: F821
    ['some_function']

    """
    if isinstance(obj, types.FunctionType):
        ns = obj.__globals__
        name = obj.__name__
    elif isinstance(obj, (type(type), type)):
        ns = sys.modules[obj.__module__].__dict__
        name = obj.__name__
    else:
        raise TypeError("expected a function or a class, got %s" % obj)

    if "__all__" not in ns:
        ns["__all__"] = [name]
    else:
        ns["__all__"].append(name)

    return obj


def memoize_property(propfunc):
    """Property decorator that caches the value of potentially expensive
    ``propfunc`` after the first evaluation. The cached value is stored in
    the corresponding property name with an attached underscore."""
    attrname = '_' + propfunc.__name__
    sentinel = object()

    @wraps(propfunc)
    def accessor(self):
        val = getattr(self, attrname, sentinel)
        if val is sentinel:
            val = propfunc(self)
            setattr(self, attrname, val)
        return val

    return property(accessor)


def deprecated(message, *, deprecated_since_version,
               active_deprecations_target, stacklevel=3):
    '''
    Mark a function as deprecated.

    This decorator should be used if an entire function or class is
    deprecated. If only a certain functionality is deprecated, you should use
    :func:`~.warns_deprecated_sympy` directly. This decorator is just a
    convenience. There is no functional difference between using this
    decorator and calling ``warns_deprecated_sympy()`` at the top of the
    function.

    The decorator takes the same arguments as
    :func:`~.warns_deprecated_sympy`. See its
    documentation for details on what the keywords to this decorator do.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    Examples
    ========

    >>> from sympy.utilities.decorator import deprecated
    >>> from sympy import simplify
    >>> @deprecated("""\
    ... The simplify_this(expr) function is deprecated. Use simplify(expr)
    ... instead.""", deprecated_since_version="1.1",
    ... active_deprecations_target='simplify-this-deprecation')
    ... def simplify_this(expr):
    ...     """
    ...     Simplify ``expr``.
    ...
    ...     .. deprecated:: 1.1
    ...
    ...        The ``simplify_this`` function is deprecated. Use :func:`simplify`
    ...        instead. See its documentation for more information. See
    ...        :ref:`simplify-this-deprecation` for details.
    ...
    ...     """
    ...     return simplify(expr)
    >>> from sympy.abc import x
    >>> simplify_this(x*(x + 1) - x**2) # doctest: +SKIP
    <stdin>:1: SymPyDeprecationWarning:
    <BLANKLINE>
    The simplify_this(expr) function is deprecated. Use simplify(expr)
    instead.
    <BLANKLINE>
    See https://docs.sympy.org/latest/explanation/active-deprecations.html#simplify-this-deprecation
    for details.
    <BLANKLINE>
    This has been deprecated since SymPy version 1.1. It
    will be removed in a future version of SymPy.
    <BLANKLINE>
      simplify_this(x)
    x

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.exceptions.ignore_warnings
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    decorator_kwargs = {"deprecated_since_version": deprecated_since_version,
               "active_deprecations_target": active_deprecations_target}
    def deprecated_decorator(wrapped):
        if hasattr(wrapped, '__mro__'):  # wrapped is actually a class
            class wrapper(wrapped):
                __doc__ = wrapped.__doc__
                __module__ = wrapped.__module__
                _sympy_deprecated_func = wrapped
                if '__new__' in wrapped.__dict__:
                    def __new__(cls, *args, **kwargs):
                        sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                        return super().__new__(cls, *args, **kwargs)
                else:
                    def __init__(self, *args, **kwargs):
                        sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                        super().__init__(*args, **kwargs)
            wrapper.__name__ = wrapped.__name__
        else:
            @wraps(wrapped)
            def wrapper(*args, **kwargs):
                sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                return wrapped(*args, **kwargs)
            wrapper._sympy_deprecated_func = wrapped
        return wrapper
    return deprecated_decorator
<|endoftext|>"""
This module provides convenient functions to transform SymPy expressions to
lambda functions which can be used to calculate numerical values very fast.
"""

from __future__ import annotations
from typing import Any

import builtins
import inspect
import keyword
import textwrap
import linecache
import weakref

# Required despite static analysis claiming it is not used
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (is_sequence, iterable,
    NotIterable, flatten)
from sympy.utilities.misc import filldedent


__doctest_requires__ = {('lambdify',): ['numpy', 'tensorflow']}


# Default namespaces, letting us define translations that can't be defined
# by simple variable maps, like I => 1j
MATH_DEFAULT: dict[str, Any] = {}
CMATH_DEFAULT: dict[str,Any] = {}
MPMATH_DEFAULT: dict[str, Any] = {}
NUMPY_DEFAULT: dict[str, Any] = {"I": 1j}
SCIPY_DEFAULT: dict[str, Any] = {"I": 1j}
CUPY_DEFAULT: dict[str, Any] = {"I": 1j}
JAX_DEFAULT: dict[str, Any] = {"I": 1j}
TENSORFLOW_DEFAULT: dict[str, Any] = {}
TORCH_DEFAULT: dict[str, Any] = {"I": 1j}
SYMPY_DEFAULT: dict[str, Any] = {}
NUMEXPR_DEFAULT: dict[str, Any] = {}

# These are the namespaces the lambda functions will use.
# These are separate from the names above because they are modified
# throughout this file, whereas the defaults should remain unmodified.

MATH = MATH_DEFAULT.copy()
CMATH = CMATH_DEFAULT.copy()
MPMATH = MPMATH_DEFAULT.copy()
NUMPY = NUMPY_DEFAULT.copy()
SCIPY = SCIPY_DEFAULT.copy()
CUPY = CUPY_DEFAULT.copy()
JAX = JAX_DEFAULT.copy()
TENSORFLOW = TENSORFLOW_DEFAULT.copy()
TORCH = TORCH_DEFAULT.copy()
SYMPY = SYMPY_DEFAULT.copy()
NUMEXPR = NUMEXPR_DEFAULT.copy()


# Mappings between SymPy and other modules function names.
MATH_TRANSLATIONS = {
    "ceiling": "ceil",
    "E": "e",
    "ln": "log",
}

CMATH_TRANSLATIONS: dict[str, str] = {}

# NOTE: This dictionary is reused in Function._eval_evalf to allow subclasses
# of Function to automatically evalf.
MPMATH_TRANSLATIONS = {
    "Abs": "fabs",
    "elliptic_k": "ellipk",
    "elliptic_f": "ellipf",
    "elliptic_e": "ellipe",
    "elliptic_pi": "ellippi",
    "ceiling": "ceil",
    "chebyshevt": "chebyt",
    "chebyshevu": "chebyu",
    "assoc_legendre": "legenp",
    "E": "e",
    "I": "j",
    "ln": "log",
    #"lowergamma":"lower_gamma",
    "oo": "inf",
    #"uppergamma":"upper_gamma",
    "LambertW": "lambertw",
    "MutableDenseMatrix": "matrix",
    "ImmutableDenseMatrix": "matrix",
    "conjugate": "conj",
    "dirichlet_eta": "altzeta",
    "Ei": "ei",
    "Shi": "shi",
    "Chi": "chi",
    "Si": "si",
    "Ci": "ci",
    "RisingFactorial": "rf",
    "FallingFactorial": "ff",
    "betainc_regularized": "betainc",
}

NUMPY_TRANSLATIONS: dict[str, str] = {
    "Heaviside": "heaviside",
}
SCIPY_TRANSLATIONS: dict[str, str] = {
    "jn" : "spherical_jn",
    "yn" : "spherical_yn"
}
CUPY_TRANSLATIONS: dict[str, str] = {}
JAX_TRANSLATIONS: dict[str, str] = {}

TENSORFLOW_TRANSLATIONS: dict[str, str] = {}
TORCH_TRANSLATIONS: dict[str, str] = {}

NUMEXPR_TRANSLATIONS: dict[str, str] = {}

# Available modules:
MODULES = {
    "math": (MATH, MATH_DEFAULT, MATH_TRANSLATIONS, ("from math import *",)),
    "cmath": (CMATH, CMATH_DEFAULT, CMATH_TRANSLATIONS, ("import cmath; from cmath import *",)),
    "mpmath": (MPMATH, MPMATH_DEFAULT, MPMATH_TRANSLATIONS, ("from mpmath import *",)),
    "numpy": (NUMPY, NUMPY_DEFAULT, NUMPY_TRANSLATIONS, ("import numpy; from numpy import *; from numpy.linalg import *",)),
    "scipy": (SCIPY, SCIPY_DEFAULT, SCIPY_TRANSLATIONS, ("import scipy; import numpy; from scipy.special import *",)),
    "cupy": (CUPY, CUPY_DEFAULT, CUPY_TRANSLATIONS, ("import cupy",)),
    "jax": (JAX, JAX_DEFAULT, JAX_TRANSLATIONS, ("import jax",)),
    "tensorflow": (TENSORFLOW, TENSORFLOW_DEFAULT, TENSORFLOW_TRANSLATIONS, ("import tensorflow",)),
    "torch": (TORCH, TORCH_DEFAULT, TORCH_TRANSLATIONS, ("import torch",)),
    "sympy": (SYMPY, SYMPY_DEFAULT, {}, (
        "from sympy.functions import *",
        "from sympy.matrices import *",
        "from sympy import Integral, pi, oo, nan, zoo, E, I",)),
    "numexpr" : (NUMEXPR, NUMEXPR_DEFAULT, NUMEXPR_TRANSLATIONS,
                 ("import_module('numexpr')", )),
}


def _import(module, reload=False):
    """
    Creates a global translation dictionary for module.

    The argument module has to be one of the following strings: "math","cmath"
    "mpmath", "numpy", "sympy", "tensorflow", "jax".
    These dictionaries map names of Python functions to their equivalent in
    other modules.
    """
    try:
        namespace, namespace_default, translations, import_commands = MODULES[
            module]
    except KeyError:
        raise NameError(
            "'%s' module cannot be used for lambdification" % module)

    # Clear namespace or exit
    if namespace != namespace_default:
        # The namespace was already generated, don't do it again if not forced.
        if reload:
            namespace.clear()
            namespace.update(namespace_default)
        else:
            return

    for import_command in import_commands:
        if import_command.startswith('import_module'):
            module = eval(import_command)

            if module is not None:
                namespace.update(module.__dict__)
                continue
        else:
            try:
                exec(import_command, {}, namespace)
                continue
            except ImportError:
                pass

        raise ImportError(
            "Cannot import '%s' with '%s' command" % (module, import_command))

    # Add translated names to namespace
    for sympyname, translation in translations.items():
        namespace[sympyname] = namespace[translation]

    # For computing the modulus of a SymPy expression we use the builtin abs
    # function, instead of the previously used fabs function for all
    # translation modules. This is because the fabs function in the math
    # module does not accept complex valued arguments. (see issue 9474). The
    # only exception, where we don't use the builtin abs function is the
    # mpmath translation module, because mpmath.fabs returns mpf objects in
    # contrast to abs().
    if 'Abs' not in namespace:
        namespace['Abs'] = abs

# Used for dynamically generated filenames that are inserted into the
# linecache.
_lambdify_generated_counter = 1


@doctest_depends_on(modules=('numpy', 'scipy', 'tensorflow',), python_version=(3,))
def lambdify(args, expr, modules=None, printer=None, use_imps=True,
             dummify=False, cse=False, docstring_limit=1000):
    """Convert a SymPy expression into a function that allows for fast
    numeric evaluation.

    .. warning::
       This function uses ``exec``, and thus should not be used on
       unsanitized input.

    .. deprecated:: 1.7
       Passing a set for the *args* parameter is deprecated as sets are
       unordered. Use an ordered iterable such as a list or tuple.

    Explanation
    ===========

    For example, to convert the SymPy expression ``sin(x) + cos(x)`` to an
    equivalent NumPy function that numerically evaluates it:

    >>> from sympy import sin, cos, symbols, lambdify
    >>> import numpy as np
    >>> x = symbols('x')
    >>> expr = sin(x) + cos(x)
    >>> expr
    sin(x) + cos(x)
    >>> f = lambdify(x, expr, 'numpy')
    >>> a = np.array([1, 2])
    >>> f(a)
    [1.38177329 0.49315059]

    The primary purpose of this function is to provide a bridge from SymPy
    expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,
    and tensorflow. In general, SymPy functions do not work with objects from
    other libraries, such as NumPy arrays, and functions from numeric
    libraries like NumPy or mpmath do not work on SymPy expressions.
    ``lambdify`` bridges the two by converting a SymPy expression to an
    equivalent numeric function.

    The basic workflow with ``lambdify`` is to first create a SymPy expression
    representing whatever mathematical function you wish to evaluate. This
    should be done using only SymPy functions and expressions. Then, use
    ``lambdify`` to convert this to an equivalent function for numerical
    evaluation. For instance, above we created ``expr`` using the SymPy symbol
    ``x`` and SymPy functions ``sin`` and ``cos``, then converted it to an
    equivalent NumPy function ``f``, and called it on a NumPy array ``a``.

    Parameters
    ==========

    args : List[Symbol]
        A variable or a list of variables whose nesting represents the
        nesting of the arguments that will be passed to the function.

        Variables can be symbols, undefined functions, or matrix symbols.

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z

        The list of variables should match the structure of how the
        arguments will be passed to the function. Simply enclose the
        parameters as they will be passed in a list.

        To call a function like ``f(x)`` then ``[x]``
        should be the first argument to ``lambdify``; for this
        case a single ``x`` can also be used:

        >>> f = lambdify(x, x + 1)
        >>> f(1)
        2
        >>> f = lambdify([x], x + 1)
        >>> f(1)
        2

        To call a function like ``f(x, y)`` then ``[x, y]`` will
        be the first argument of the ``lambdify``:

        >>> f = lambdify([x, y], x + y)
        >>> f(1, 1)
        2

        To call a function with a single 3-element tuple like
        ``f((x, y, z))`` then ``[(x, y, z)]`` will be the first
        argument of the ``lambdify``:

        >>> f = lambdify([(x, y, z)], Eq(z**2, x**2 + y**2))
        >>> f((3, 4, 5))
        True

        If two args will be passed and the first is a scalar but
        the second is a tuple with two arguments then the items
        in the list should match that structure:

        >>> f = lambdify([x, (y, z)], x + y + z)
        >>> f(1, (2, 3))
        6

    expr : Expr
        An expression, list of expressions, or matrix to be evaluated.

        Lists may be nested.
        If the expression is a list, the output will also be a list.

        >>> f = lambdify(x, [x, [x + 1, x + 2]])
        >>> f(1)
        [1, [2, 3]]

        If it is a matrix, an array will be returned (for the NumPy module).

        >>> from sympy import Matrix
        >>> f = lambdify(x, Matrix([x, x + 1]))
        >>> f(1)
        [[1]
        [2]]

        Note that the argument order here (variables then expression) is used
        to emulate the Python ``lambda`` keyword. ``lambdify(x, expr)`` works
        (roughly) like ``lambda x: expr``
        (see :ref:`lambdify-how-it-works` below).

    modules : str, optional
        Specifies the numeric library to use.

        If not specified, *modules* defaults to:

        - ``["scipy", "numpy"]`` if SciPy is installed
        - ``["numpy"]`` if only NumPy is installed
        - ``["math","cmath", "mpmath", "sympy"]`` if neither is installed.

        That is, SymPy functions are replaced as far as possible by
        either ``scipy`` or ``numpy`` functions if available, and Python's
        standard library ``math`` and ``cmath``, or ``mpmath`` functions otherwise.

        *modules* can be one of the following types:

        - The strings ``"math"``, ``"cmath"``, ``"mpmath"``, ``"numpy"``, ``"numexpr"``,
          ``"scipy"``, ``"sympy"``, or ``"tensorflow"`` or ``"jax"``. This uses the
          corresponding printer and namespace mapping for that module.
        - A module (e.g., ``math``). This uses the global namespace of the
          module. If the module is one of the above known modules, it will
          also use the corresponding printer and namespace mapping
          (i.e., ``modules=numpy`` is equivalent to ``modules="numpy"``).
        - A dictionary that maps names of SymPy functions to arbitrary
          functions
          (e.g., ``{'sin': custom_sin}``).
        - A list that contains a mix of the arguments above, with higher
          priority given to entries appearing first
          (e.g., to use the NumPy module but override the ``sin`` function
          with a custom version, you can use
          ``[{'sin': custom_sin}, 'numpy']``).

    dummify : bool, optional
        Whether or not the variables in the provided expression that are not
        valid Python identifiers are substituted with dummy symbols.

        This allows for undefined functions like ``Function('f')(t)`` to be
        supplied as arguments. By default, the variables are only dummified
        if they are not valid Python identifiers.

        Set ``dummify=True`` to replace all arguments with dummy symbols
        (if ``args`` is not a string) - for example, to ensure that the
        arguments do not redefine any built-in names.

    cse : bool, or callable, optional
        Large expressions can be computed more efficiently when
        common subexpressions are identified and precomputed before
        being used multiple time. Finding the subexpressions will make
        creation of the 'lambdify' function slower, however.

        When ``True``, ``sympy.simplify.cse`` is used, otherwise (the default)
        the user may pass a function matching the ``cse`` signature.

    docstring_limit : int or None
        When lambdifying large expressions, a significant proportion of the time
        spent inside ``lambdify`` is spent producing a string representation of
        the expression for use in the automatically generated docstring of the
        returned function. For expressions containing hundreds or more nodes the
        resulting docstring often becomes so long and dense that it is difficult
        to read. To reduce the runtime of lambdify, the rendering of the full
        expression inside the docstring can be disabled.

        When ``None``, the full expression is rendered in the docstring. When
        ``0`` or a negative ``int``, an ellipsis is rendering in the docstring
        instead of the expression. When a strictly positive ``int``, if the
        number of nodes in the expression exceeds ``docstring_limit`` an
        ellipsis is rendered in the docstring, otherwise a string representation
        of the expression is rendered as normal. The default is ``1000``.

    Examples
    ========

    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import sqrt, sin, Matrix
    >>> from sympy import Function
    >>> from sympy.abc import w, x, y, z

    >>> f = lambdify(x, x**2)
    >>> f(2)
    4
    >>> f = lambdify((x, y, z), [z, y, x])
    >>> f(1,2,3)
    [3, 2, 1]
    >>> f = lambdify(x, sqrt(x))
    >>> f(4)
    2.0
    >>> f = lambdify((x, y), sin(x*y)**2)
    >>> f(0, 5)
    0.0
    >>> row = lambdify((x, y), Matrix((x, x + y)).T, modules='sympy')
    >>> row(1, 2)
    Matrix([[1, 3]])

    ``lambdify`` can be used to translate SymPy expressions into mpmath
    functions. This may be preferable to using ``evalf`` (which uses mpmath on
    the backend) in some cases.

    >>> f = lambdify(x, sin(x), 'mpmath')
    >>> f(1)
    0.8414709848078965

    Tuple arguments are handled and the lambdified function should
    be called with the same type of arguments as were used to create
    the function:

    >>> f = lambdify((x, (y, z)), x + y)
    >>> f(1, (2, 4))
    3

    The ``flatten`` function can be used to always work with flattened
    arguments:

    >>> from sympy.utilities.iterables import flatten
    >>> args = w, (x, (y, z))
    >>> vals = 1, (2, (3, 4))
    >>> f = lambdify(flatten(args), w + x + y + z)
    >>> f(*flatten(vals))
    10

    Functions present in ``expr`` can also carry their own numerical
    implementations, in a callable attached to the ``_imp_`` attribute. This
    can be used with undefined functions using the ``implemented_function``
    factory:

    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> func = lambdify(x, f(x))
    >>> func(4)
    5

    ``lambdify`` always prefers ``_imp_`` implementations to implementations
    in other namespaces, unless the ``use_imps`` input parameter is False.

    Usage with Tensorflow:

    >>> import tensorflow as tf
    >>> from sympy import Max, sin, lambdify
    >>> from sympy.abc import x

    >>> f = Max(x, sin(x))
    >>> func = lambdify(x, f, 'tensorflow')

    After tensorflow v2, eager execution is enabled by default.
    If you want to get the compatible result across tensorflow v1 and v2
    as same as this tutorial, run this line.

    >>> tf.compat.v1.enable_eager_execution()

    If you have eager execution enabled, you can get the result out
    immediately as you can use numpy.

    If you pass tensorflow objects, you may get an ``EagerTensor``
    object instead of value.

    >>> result = func(tf.constant(1.0))
    >>> print(result)
    tf.Tensor(1.0, shape=(), dtype=float32)
    >>> print(result.__class__)
    <class 'tensorflow.python.framework.ops.EagerTensor'>

    You can use ``.numpy()`` to get the numpy value of the tensor.

    >>> result.numpy()
    1.0

    >>> var = tf.Variable(2.0)
    >>> result = func(var) # also works for tf.Variable and tf.Placeholder
    >>> result.numpy()
    2.0

    And it works with any shape array.

    >>> tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> result = func(tensor)
    >>> result.numpy()
    [[1. 2.]
     [3. 4.]]

    Notes
    =====

    - For functions involving large array calculations, numexpr can provide a
      significant speedup over numpy. Please note that the available functions
      for numexpr are more limited than numpy but can be expanded with
      ``implemented_function`` and user defined subclasses of Function. If
      specified, numexpr may be the only option in modules. The official list
      of numexpr functions can be found at:
      https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions

    - In the above examples, the generated functions can accept scalar
      values or numpy arrays as arguments.  However, in some cases
      the generated function relies on the input being a numpy array:

      >>> import numpy
      >>> from sympy import Piecewise
      >>> from sympy.testing.pytest import ignore_warnings
      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "numpy")

      >>> with ignore_warnings(RuntimeWarning):
      ...     f(numpy.array([-1, 0, 1, 2]))
      [-1.   0.   1.   0.5]

      >>> f(0)
      Traceback (most recent call last):
          ...
      ZeroDivisionError: division by zero

      In such cases, the input should be wrapped in a numpy array:

      >>> with ignore_warnings(RuntimeWarning):
      ...     float(f(numpy.array([0])))
      0.0

      Or if numpy functionality is not required another module can be used:

      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "math")
      >>> f(0)
      0

    .. _lambdify-how-it-works:

    How it works
    ============

    When using this function, it helps a great deal to have an idea of what it
    is doing. At its core, lambdify is nothing more than a namespace
    translation, on top of a special printer that makes some corner cases work
    properly.

    To understand lambdify, first we must properly understand how Python
    namespaces work. Say we had two files. One called ``sin_cos_sympy.py``,
    with

    .. code:: python

        # sin_cos_sympy.py

        from sympy.functions.elementary.trigonometric import (cos, sin)

        def sin_cos(x):
            return sin(x) + cos(x)


    and one called ``sin_cos_numpy.py`` with

    .. code:: python

        # sin_cos_numpy.py

        from numpy import sin, cos

        def sin_cos(x):
            return sin(x) + cos(x)

    The two files define an identical function ``sin_cos``. However, in the
    first file, ``sin`` and ``cos`` are defined as the SymPy ``sin`` and
    ``cos``. In the second, they are defined as the NumPy versions.

    If we were to import the first file and use the ``sin_cos`` function, we
    would get something like

    >>> from sin_cos_sympy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    cos(1) + sin(1)

    On the other hand, if we imported ``sin_cos`` from the second file, we
    would get

    >>> from sin_cos_numpy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    1.38177329068

    In the first case we got a symbolic output, because it used the symbolic
    ``sin`` and ``cos`` functions from SymPy. In the second, we got a numeric
    result, because ``sin_cos`` used the numeric ``sin`` and ``cos`` functions
    from NumPy. But notice that the versions of ``sin`` and ``cos`` that were
    used was not inherent to the ``sin_cos`` function definition. Both
    ``sin_cos`` definitions are exactly the same. Rather, it was based on the
    names defined at the module where the ``sin_cos`` function was defined.

    The key point here is that when function in Python references a name that
    is not defined in the function, that name is looked up in the "global"
    namespace of the module where that function is defined.

    Now, in Python, we can emulate this behavior without actually writing a
    file to disk using the ``exec`` function. ``exec`` takes a string
    containing a block of Python code, and a dictionary that should contain
    the global variables of the module. It then executes the code "in" that
    dictionary, as if it were the module globals. The following is equivalent
    to the ``sin_cos`` defined in ``sin_cos_sympy.py``:

    >>> import sympy
    >>> module_dictionary = {'sin': sympy.sin, 'cos': sympy.cos}
    >>> exec('''
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... ''', module_dictionary)
    >>> sin_cos = module_dictionary['sin_cos']
    >>> sin_cos(1)
    cos(1) + sin(1)

    and similarly with ``sin_cos_numpy``:

    >>> import numpy
    >>> module_dictionary = {'sin': numpy.sin, 'cos': numpy.cos}
    >>> exec('''
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... ''', module_dictionary)
    >>> sin_cos = module_dictionary['sin_cos']
    >>> sin_cos(1)
    1.38177329068

    So now we can get an idea of how ``lambdify`` works. The name "lambdify"
    comes from the fact that we can think of something like ``lambdify(x,
    sin(x) + cos(x), 'numpy')`` as ``lambda x: sin(x) + cos(x)``, where
    ``sin`` and ``cos`` come from the ``numpy`` namespace. This is also why
    the symbols argument is first in ``lambdify``, as opposed to most SymPy
    functions where it comes after the expression: to better mimic the
    ``lambda`` keyword.

    ``lambdify`` takes the input expression (like ``sin(x) + cos(x)``) and

    1. Converts it to a string
    2. Creates a module globals dictionary based on the modules that are
       passed in (by default, it uses the NumPy module)
    3. Creates the string ``"def func({vars}): return {expr}"``, where ``{vars}`` is the
       list of variables separated by commas, and ``{expr}`` is the string
       created in step 1., then ``exec``s that string with the module globals
       namespace and returns ``func``.

    In fact, functions returned by ``lambdify`` support inspection. So you can
    see exactly how they are defined by using ``inspect.getsource``, or ``??`` if you
    are using IPython or the Jupyter notebook.

    >>> f = lambdify(x, sin(x) + cos(x))
    >>> import inspect
    >>> print(inspect.getsource(f))
    def _lambdifygenerated(x):
        return sin(x) + cos(x)

    This shows us the source code of the function, but not the namespace it
    was defined in. We can inspect that by looking at the ``__globals__``
    attribute of ``f``:

    >>> f.__globals__['sin']
    <ufunc 'sin'>
    >>> f.__globals__['cos']
    <ufunc 'cos'>
    >>> f.__globals__['sin'] is numpy.sin
    True

    This shows us that ``sin`` and ``cos`` in the namespace of ``f`` will be
    ``numpy.sin`` and ``numpy.cos``.

    Note that there are some convenience layers in each of these steps, but at
    the core, this is how ``lambdify`` works. Step 1 is done using the
    ``LambdaPrinter`` printers defined in the printing module (see
    :mod:`sympy.printing.lambdarepr`). This allows different SymPy expressions
    to define how they should be converted to a string for different modules.
    You can change which printer ``lambdify`` uses by passing a custom printer
    in to the ``printer`` argument.

    Step 2 is augmented by certain translations. There are default
    translations for each module, but you can provide your own by passing a
    list to the ``modules`` argument. For instance,

    >>> def mysin(x):
    ...     print('taking the sin of', x)
    ...     return numpy.sin(x)
    ...
    >>> f = lambdify(x, sin(x), [{'sin': mysin}, 'numpy'])
    >>> f(1)
    taking the sin of 1
    0.8414709848078965

    The globals dictionary is generated from the list by merging the
    dictionary ``{'sin': mysin}`` and the module dictionary for NumPy. The
    merging is done so that earlier items take precedence, which is why
    ``mysin`` is used above instead of ``numpy.sin``.

    If you want to modify the way ``lambdify`` works for a given function, it
    is usually easiest to do so by modifying the globals dictionary as such.
    In more complicated cases, it may be necessary to create and pass in a
    custom printer.

    Finally, step 3 is augmented with certain convenience operations, such as
    the addition of a docstring.

    Understanding how ``lambdify`` works can make it easier to avoid certain
    gotchas when using it. For instance, a common mistake is to create a
    lambdified function for one module (say, NumPy), and pass it objects from
    another (say, a SymPy expression).

    For instance, say we create

    >>> from sympy.abc import x
    >>> f = lambdify(x, x + 1, 'numpy')

    Now if we pass in a NumPy array, we get that array plus 1

    >>> import numpy
    >>> a = numpy.array([1, 2])
    >>> f(a)
    [2 3]

    But what happens if you make the mistake of passing in a SymPy expression
    instead of a NumPy array:

    >>> f(x + 1)
    x + 2

    This worked, but it was only by accident. Now take a different lambdified
    function:

    >>> from sympy import sin
    >>> g = lambdify(x, x + sin(x), 'numpy')

    This works as expected on NumPy arrays:

    >>> g(a)
    [1.84147098 2.90929743]

    But if we try to pass in a SymPy expression, it fails

    >>> g(x + 1)
    Traceback (most recent call last):
    ...
    TypeError: loop of ufunc does not support argument 0 of type Add which has
               no callable sin method

    Now, let's look at what happened. The reason this fails is that ``g``
    calls ``numpy.sin`` on the input expression, and ``numpy.sin`` does not
    know how to operate on a SymPy object. **As a general rule, NumPy
    functions do not know how to operate on SymPy expressions, and SymPy
    functions do not know how to operate on NumPy arrays. This is why lambdify
    exists: to provide a bridge between SymPy and NumPy.**

    However, why is it that ``f`` did work? That's because ``f`` does not call
    any functions, it only adds 1. So the resulting function that is created,
    ``def _lambdifygenerated(x): return x + 1`` does not depend on the globals
    namespace it is defined in. Thus it works, but only by accident. A future
    version of ``lambdify`` may remove this behavior.

    Be aware that certain implementation details described here may change in
    future versions of SymPy. The API of passing in custom modules and
    printers will not change, but the details of how a lambda function is
    created may change. However, the basic idea will remain the same, and
    understanding it will be helpful to understanding the behavior of
    lambdify.

    **In general: you should create lambdified functions for one module (say,
    NumPy), and only pass it input types that are compatible with that module
    (say, NumPy arrays).** Remember that by default, if the ``module``
    argument is not provided, ``lambdify`` creates functions using the NumPy
    and SciPy namespaces.
    """
    from sympy.core.symbol import Symbol
    from sympy.core.expr import Expr

    # If the user hasn't specified any modules, use what is available.
    if modules is None:
        try:
            _import("scipy")
        except ImportError:
            try:
                _import("numpy")
            except ImportError:
                # Use either numpy (if available) or python.math where possible.
                # XXX: This leads to different behaviour on different systems and
                #      might be the reason for irreproducible errors.
                modules = ["math", "mpmath", "sympy"]
            else:
                modules = ["numpy"]
        else:
            modules = ["numpy", "scipy"]

    # Get the needed namespaces.
    namespaces = []
    # First find any function implementations
    if use_imps:
        namespaces.append(_imp_namespace(expr))
    # Check for dict before iterating
    if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
        namespaces.append(modules)
    else:
        # consistency check
        if _module_present('numexpr', modules) and len(modules) > 1:
            raise TypeError("numexpr must be the only item in 'modules'")
        namespaces += list(modules)
    # fill namespace with first having highest priority
    namespace = {}
    for m in namespaces[::-1]:
        buf = _get_namespace(m)
        namespace.update(buf)

    if hasattr(expr, "atoms"):
        #Try if you can extract symbols from the expression.
        #Move on if expr.atoms in not implemented.
        syms = expr.atoms(Symbol)
        for term in syms:
            namespace.update({str(term): term})

    if printer is None:
        if _module_present('mpmath', namespaces):
            from sympy.printing.pycode import MpmathPrinter as Printer # type: ignore
        elif _module_present('scipy', namespaces):
            from sympy.printing.numpy import SciPyPrinter as Printer # type: ignore
        elif _module_present('numpy', namespaces):
            from sympy.printing.numpy import NumPyPrinter as Printer # type: ignore
        elif _module_present('cupy', namespaces):
            from sympy.printing.numpy import CuPyPrinter as Printer # type: ignore
        elif _module_present('jax', namespaces):
            from sympy.printing.numpy import JaxPrinter as Printer # type: ignore
        elif _module_present('numexpr', namespaces):
            from sympy.printing.lambdarepr import NumExprPrinter as Printer # type: ignore
        elif _module_present('tensorflow', namespaces):
            from sympy.printing.tensorflow import TensorflowPrinter as Printer # type: ignore
        elif _module_present('torch', namespaces):
            from sympy.printing.pytorch import TorchPrinter as Printer  # type: ignore
        elif _module_present('sympy', namespaces):
            from sympy.printing.pycode import SymPyPrinter as Printer # type: ignore
        elif _module_present('cmath', namespaces):
            from sympy.printing.pycode import CmathPrinter as Printer # type: ignore
        else:
            from sympy.printing.pycode import PythonCodePrinter as Printer # type: ignore
        user_functions = {}
        for m in namespaces[::-1]:
            if isinstance(m, dict):
                for k in m:
                    user_functions[k] = k
        printer = Printer({'fully_qualified_modules': False, 'inline': True,
                           'allow_unknown_functions': True,
                           'user_functions': user_functions})

    if isinstance(args, set):
        sympy_deprecation_warning(
            """
Passing the function arguments to lambdify() as a set is deprecated. This
leads to unpredictable results since sets are unordered. Instead, use a list
or tuple for the function arguments.
            """,
            deprecated_since_version="1.6.3",
            active_deprecations_target="deprecated-lambdify-arguments-set",
                )

    # Get the names of the args, for creating a docstring
    iterable_args = (args,) if isinstance(args, Expr) else args
    names = []

    # Grab the callers frame, for getting the names by inspection (if needed)
    callers_local_vars = inspect.currentframe().f_back.f_locals.items() # type: ignore
    for n, var in enumerate(iterable_args):
        if hasattr(var, 'name'):
            names.append(var.name)
        else:
            # It's an iterable. Try to get name by inspection of calling frame.
            name_list = [var_name for var_name, var_val in callers_local_vars
                    if var_val is var]
            if len(name_list) == 1:
                names.append(name_list[0])
            else:
                # Cannot infer name with certainty. arg_# will have to do.
                names.append('arg_' + str(n))

    # Create the function definition code and execute it
    funcname = '_lambdifygenerated'
    if _module_present('tensorflow', namespaces):
        funcprinter = _TensorflowEvaluatorPrinter(printer, dummify)
    else:
        funcprinter = _EvaluatorPrinter(printer, dummify)

    if cse == True:
        from sympy.simplify.cse_main import cse as _cse
        cses, _expr = _cse(expr, list=False)
    elif callable(cse):
        cses, _expr = cse(expr)
    else:
        cses, _expr = (), expr
    funcstr = funcprinter.doprint(funcname, iterable_args, _expr, cses=cses)

    # Collect the module imports from the code printers.
    imp_mod_lines = []
    for mod, keys in (getattr(printer, 'module_imports', None) or {}).items():
        for k in keys:
            if k not in namespace:
                ln = "from %s import %s" % (mod, k)
                try:
                    exec(ln, {}, namespace)
                except ImportError:
                    # Tensorflow 2.0 has issues with importing a specific
                    # function from its submodule.
                    # https://github.com/tensorflow/tensorflow/issues/33022
                    ln = "%s = %s.%s" % (k, mod, k)
                    exec(ln, {}, namespace)
                imp_mod_lines.append(ln)

    # Provide lambda expression with builtins, and compatible implementation of range
    namespace.update({'builtins':builtins, 'range':range})

    funclocals = {}
    global _lambdify_generated_counter
    filename = '<lambdifygenerated-%s>' % _lambdify_generated_counter
    _lambdify_generated_counter += 1
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)
    # mtime has to be None or else linecache.checkcache will remove it
    linecache.cache[filename] = (len(funcstr), None, funcstr.splitlines(True), filename) # type: ignore

    # Remove the entry from the linecache when the object is garbage collected
    def cleanup_linecache(filename):
        def _cleanup():
            if filename in linecache.cache:
                del linecache.cache[filename]
        return _cleanup

    func = funclocals[funcname]

    weakref.finalize(func, cleanup_linecache(filename))

    # Apply the docstring
    sig = "func({})".format(", ".join(str(i) for i in names))
    sig = textwrap.fill(sig, subsequent_indent=' '*8)
    if _too_large_for_docstring(expr, docstring_limit):
        expr_str = "EXPRESSION REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        src_str = "SOURCE CODE REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
    else:
        expr_str = str(expr)
        if len(expr_str) > 78:
            expr_str = textwrap.wrap(expr_str, 75)[0] + '...'
        src_str = funcstr
    func.__doc__ = (
        "Created with lambdify. Signature:\n\n"
        "{sig}\n\n"
        "Expression:\n\n"
        "{expr}\n\n"
        "Source code:\n\n"
        "{src}\n\n"
        "Imported modules:\n\n"
        "{imp_mods}"
        ).format(sig=sig, expr=expr_str, src=src_str, imp_mods='\n'.join(imp_mod_lines))
    return func

def _module_present(modname, modlist):
    if modname in modlist:
        return True
    for m in modlist:
        if hasattr(m, '__name__') and m.__name__ == modname:
            return True
    return False

def _get_namespace(m):
    """
    This is used by _lambdify to parse its arguments.
    """
    if isinstance(m, str):
        _import(m)
        return MODULES[m][0]
    elif isinstance(m, dict):
        return m
    elif hasattr(m, "__dict__"):
        return m.__dict__
    else:
        raise TypeError("Argument must be either a string, dict or module but it is: %s" % m)


def _recursive_to_string(doprint, arg):
    """Functions in lambdify accept both SymPy types and non-SymPy types such as python
    lists and tuples. This method ensures that we only call the doprint method of the
    printer with SymPy types (so that the printer safely can use SymPy-methods)."""
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.core.basic import Basic

    if isinstance(arg, (Basic, MatrixBase)):
        return doprint(arg)
    elif iterable(arg):
        if isinstance(arg, list):
            left, right = "[", "]"
        elif isinstance(arg, tuple):
            left, right = "(", ",)"
            if not arg:
                return "()"
        else:
            raise NotImplementedError("unhandled type: %s, %s" % (type(arg), arg))
        return left +', '.join(_recursive_to_string(doprint, e) for e in arg) + right
    elif isinstance(arg, str):
        return arg
    else:
        return doprint(arg)


def lambdastr(args, expr, printer=None, dummify=None):
    """
    Returns a string that can be evaluated to a lambda function.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import lambdastr
    >>> lambdastr(x, x**2)
    'lambda x: (x**2)'
    >>> lambdastr((x,y,z), [z,y,x])
    'lambda x,y,z: ([z, y, x])'

    Although tuples may not appear as arguments to lambda in Python 3,
    lambdastr will create a lambda function that will unpack the original
    arguments so that nested arguments can be handled:

    >>> lambdastr((x, (y, z)), x + y)
    'lambda _0,_1: (lambda x,y,z: (x + y))(_0,_1[0],_1[1])'
    """
    # Transforming everything to strings.
    from sympy.matrices import DeferredVector
    from sympy.core.basic import Basic
    from sympy.core.function import (Derivative, Function)
    from sympy.core.symbol import (Dummy, Symbol)
    from sympy.core.sympify import sympify

    if printer is not None:
        if inspect.isfunction(printer):
            lambdarepr = printer
        else:
            if inspect.isclass(printer):
                lambdarepr = lambda expr: printer().doprint(expr)
            else:
                lambdarepr = lambda expr: printer.doprint(expr)
    else:
        #XXX: This has to be done here because of circular imports
        from sympy.printing.lambdarepr import lambdarepr

    def sub_args(args, dummies_dict):
        if isinstance(args, str):
            return args
        elif isinstance(args, DeferredVector):
            return str(args)
        elif iterable(args):
            dummies = flatten([sub_args(a, dummies_dict) for a in args])
            return ",".join(str(a) for a in dummies)
        else:
            # replace these with Dummy symbols
            if isinstance(args, (Function, Symbol, Derivative)):
                dummies = Dummy()
                dummies_dict.update({args : dummies})
                return str(dummies)
            else:
                return str(args)

    def sub_expr(expr, dummies_dict):
        expr = sympify(expr)
        # dict/tuple are sympified to Basic
        if isinstance(expr, Basic):
            expr = expr.xreplace(dummies_dict)
        # list is not sympified to Basic
        elif isinstance(expr, list):
            expr = [sub_expr(a, dummies_dict) for a in expr]
        return expr

    # Transform args
    def isiter(l):
        return iterable(l, exclude=(str, DeferredVector, NotIterable))

    def flat_indexes(iterable):
        n = 0

        for el in iterable:
            if isiter(el):
                for ndeep in flat_indexes(el):
                    yield (n,) + ndeep
            else:
                yield (n,)

            n += 1

    if dummify is None:
        dummify = any(isinstance(a, Basic) and
            a.atoms(Function, Derivative) for a in (
            args if isiter(args) else [args]))

    if isiter(args) and any(isiter(i) for i in args):
        dum_args = [str(Dummy(str(i))) for i in range(len(args))]

        indexed_args = ','.join([
            dum_args[ind[0]] + ''.join(["[%s]" % k for k in ind[1:]])
                    for ind in flat_indexes(args)])

        lstr = lambdastr(flatten(args), expr, printer=printer, dummify=dummify)

        return 'lambda %s: (%s)(%s)' % (','.join(dum_args), lstr, indexed_args)

    dummies_dict = {}
    if dummify:
        args = sub_args(args, dummies_dict)
    else:
        if isinstance(args, str):
            pass
        elif iterable(args, exclude=DeferredVector):
            args = ",".join(str(a) for a in args)

    # Transform expr
    if dummify:
        if isinstance(expr, str):
            pass
        else:
            expr = sub_expr(expr, dummies_dict)
    expr = _recursive_to_string(lambdarepr, expr)
    return "lambda %s: (%s)" % (args, expr)

class _EvaluatorPrinter:
    def __init__(self, printer=None, dummify=False):
        self._dummify = dummify

        #XXX: This has to be done here because of circular imports
        from sympy.printing.lambdarepr import LambdaPrinter

        if printer is None:
            printer = LambdaPrinter()

        if inspect.isfunction(printer):
            self._exprrepr = printer
        else:
            if inspect.isclass(printer):
                printer = printer()

            self._exprrepr = printer.doprint

            #if hasattr(printer, '_print_Symbol'):
            #    symbolrepr = printer._print_Symbol

            #if hasattr(printer, '_print_Dummy'):
            #    dummyrepr = printer._print_Dummy

        # Used to print the generated function arguments in a standard way
        self._argrepr = LambdaPrinter().doprint

    def doprint(self, funcname, args, expr, *, cses=()):
        """
        Returns the function definition code as a string.
        """
        from sympy.core.symbol import Dummy

        funcbody = []

        if not iterable(args):
            args = [args]

        if cses:
            cses = list(cses)
            subvars, subexprs = zip(*cses)
            exprs = [expr] + list(subexprs)
            argstrs, exprs = self._preprocess(args, exprs, cses=cses)
            expr, subexprs = exprs[0], exprs[1:]
            cses = zip(subvars, subexprs)
        else:
            argstrs, expr = self._preprocess(args, expr)

        # Generate argument unpacking and final argument list
        funcargs = []
        unpackings = []

        for argstr in argstrs:
            if iterable(argstr):
                funcargs.append(self._argrepr(Dummy()))
                unpackings.extend(self._print_unpacking(argstr, funcargs[-1]))
            else:
                funcargs.append(argstr)

        funcsig = 'def {}({}):'.format(funcname, ', '.join(funcargs))

        # Wrap input arguments before unpacking
        funcbody.extend(self._print_funcargwrapping(funcargs))

        funcbody.extend(unpackings)

        for s, e in cses:
            if e is None:
                funcbody.append('del {}'.format(self._exprrepr(s)))
            else:
                funcbody.append('{} = {}'.format(self._exprrepr(s), self._exprrepr(e)))

        # Subs may appear in expressions generated by .diff()
        subs_assignments = []
        expr = self._handle_Subs(expr, out=subs_assignments)
        for lhs, rhs in subs_assignments:
            funcbody.append('{} = {}'.format(self._exprrepr(lhs), self._exprrepr(rhs)))

        str_expr = _recursive_to_string(self._exprrepr, expr)

        if '\n' in str_expr:
            str_expr = '({})'.format(str_expr)
        funcbody.append('return {}'.format(str_expr))

        funclines = [funcsig]
        funclines.extend(['    ' + line for line in funcbody])

        return '\n'.join(funclines) + '\n'

    @classmethod
    def _is_safe_ident(cls, ident):
        return isinstance(ident, str) and ident.isidentifier() \
                and not keyword.iskeyword(ident)

    def _preprocess(self, args, expr, cses=(), _dummies_dict=None):
        """Preprocess args, expr to replace arguments that do not map
        to valid Python identifiers.

        Returns string form of args, and updated expr.
        """
        from sympy.core.basic import Basic
        from sympy.core.sorting import ordered
        from sympy.core.function import (Derivative, Function)
        from sympy.core.symbol import Dummy, uniquely_named_symbol
        from sympy.matrices import DeferredVector
        from sympy.core.expr import Expr

        # Args of type Dummy can cause name collisions with args
        # of type Symbol.  Force dummify of everything in this
        # situation.
        dummify = self._dummify or any(
            isinstance(arg, Dummy) for arg in flatten(args))

        argstrs = [None]*len(args)
        if _dummies_dict is None:
            _dummies_dict = {}

        def update_dummies(arg, dummy):
            _dummies_dict[arg] = dummy
            for repl, sub in cses:
                arg = arg.xreplace({sub: repl})
                _dummies_dict[arg] = dummy

        for arg, i in reversed(list(ordered(zip(args, range(len(args)))))):
            if iterable(arg):
                s, expr = self._preprocess(arg, expr, cses=cses, _dummies_dict=_dummies_dict)
            elif isinstance(arg, DeferredVector):
                s = str(arg)
            elif isinstance(arg, Basic) and arg.is_symbol:
                s = str(arg)
                if dummify or not self._is_safe_ident(s):
                    dummy = Dummy()
                    if isinstance(expr, Expr):
                        dummy = uniquely_named_symbol(
                            dummy.name, expr, modify=lambda s: '_' + s)
                    s = self._argrepr(dummy)
                    update_dummies(arg, dummy)
                    expr = self._subexpr(expr, _dummies_dict)
            elif dummify or isinstance(arg, (Function, Derivative)):
                dummy = Dummy()
                s = self._argrepr(dummy)
                update_dummies(arg, dummy)
                expr = self._subexpr(expr, _dummies_dict)
            else:
                s = str(arg)
            argstrs[i] = s
        return argstrs, expr

    def _subexpr(self, expr, dummies_dict):
        from sympy.matrices import DeferredVector
        from sympy.core.sympify import sympify

        expr = sympify(expr)
        xreplace = getattr(expr, 'xreplace', None)
        if xreplace is not None:
            expr = xreplace(dummies_dict)
        else:
            if isinstance(expr, DeferredVector):
                pass
            elif isinstance(expr, dict):
                k = [self._subexpr(sympify(a), dummies_dict) for a in expr.keys()]
                v = [self._subexpr(sympify(a), dummies_dict) for a in expr.values()]
                expr = dict(zip(k, v))
            elif isinstance(expr, tuple):
                expr = tuple(self._subexpr(sympify(a), dummies_dict) for a in expr)
            elif isinstance(expr, list):
                expr = [self._subexpr(sympify(a), dummies_dict) for a in expr]
        return expr

    def _print_funcargwrapping(self, args):
        """Generate argument wrapping code.

        args is the argument list of the generated function (strings).

        Return value is a list of lines of code that will be inserted  at
        the beginning of the function definition.
        """
        return []

    def _print_unpacking(self, unpackto, arg):
        """Generate argument unpacking code.

        arg is the function argument to be unpacked (a string), and
        unpackto is a list or nested lists of the variable names (strings) to
        unpack to.
        """
        def unpack_lhs(lvalues):
            return '[{}]'.format(', '.join(
                unpack_lhs(val) if iterable(val) else val for val in lvalues))

        return ['{} = {}'.format(unpack_lhs(unpackto), arg)]

    def _handle_Subs(self, expr, out):
        """Any instance of Subs is extracted and returned as assignment pairs."""
        from sympy.core.basic import Basic
        from sympy.core.function import Subs
        from sympy.core.symbol import Dummy
        from sympy.matrices.matrixbase import MatrixBase

        def _replace(ex, variables, point):
            safe = {}
            for lhs, rhs in zip(variables, point):
                dummy = Dummy()
                safe[lhs] = dummy
                out.append((dummy, rhs))
            return ex.xreplace(safe)

        if isinstance(expr, (Basic, MatrixBase)):
            expr = expr.replace(Subs, _replace)
        elif iterable(expr):
            expr = type(expr)([self._handle_Subs(e, out) for e in expr])
        return expr

class _TensorflowEvaluatorPrinter(_EvaluatorPrinter):
    def _print_unpacking(self, lvalues, rvalue):
        """Generate argument unpacking code.

        This method is used when the input value is not iterable,
        but can be indexed (see issue #14655).
        """

        def flat_indexes(elems):
            n = 0

            for el in elems:
                if iterable(el):
                    for ndeep in flat_indexes(el):
                        yield (n,) + ndeep
                else:
                    yield (n,)

                n += 1

        indexed = ', '.join('{}[{}]'.format(rvalue, ']['.join(map(str, ind)))
                                for ind in flat_indexes(lvalues))

        return ['[{}] = [{}]'.format(', '.join(flatten(lvalues)), indexed)]

def _imp_namespace(expr, namespace=None):
    """ Return namespace dict with function implementations

    We need to search for functions in anything that can be thrown at
    us - that is - anything that could be passed as ``expr``.  Examples
    include SymPy expressions, as well as tuples, lists and dicts that may
    contain SymPy expressions.

    Parameters
    ----------
    expr : object
       Something passed to lambdify, that will generate valid code from
       ``str(expr)``.
    namespace : None or mapping
       Namespace to fill.  None results in new empty dict

    Returns
    -------
    namespace : dict
       dict with keys of implemented function names within ``expr`` and
       corresponding values being the numerical implementation of
       function

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function, _imp_namespace
    >>> from sympy import Function
    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> g = implemented_function(Function('g'), lambda x: x*10)
    >>> namespace = _imp_namespace(f(g(x)))
    >>> sorted(namespace.keys())
    ['f', 'g']
    """
    # Delayed import to avoid circular imports
    from sympy.core.function import FunctionClass
    if namespace is None:
        namespace = {}
    # tuples, lists, dicts are valid expressions
    if is_sequence(expr):
        for arg in expr:
            _imp_namespace(arg, namespace)
        return namespace
    elif isinstance(expr, dict):
        for key, val in expr.items():
            # functions can be in dictionary keys
            _imp_namespace(key, namespace)
            _imp_namespace(val, namespace)
        return namespace
    # SymPy expressions may be Functions themselves
    func = getattr(expr, 'func', None)
    if isinstance(func, FunctionClass):
        imp = getattr(func, '_imp_', None)
        if imp is not None:
            name = expr.func.__name__
            if name in namespace and namespace[name] != imp:
                raise ValueError('We found more than one '
                                 'implementation with name '
                                 '"%s"' % name)
            namespace[name] = imp
    # and / or they may take Functions as arguments
    if hasattr(expr, 'args'):
        for arg in expr.args:
            _imp_namespace(arg, namespace)
    return namespace


def implemented_function(symfunc, implementation):
    """ Add numerical ``implementation`` to function ``symfunc``.

    ``symfunc`` can be an ``UndefinedFunction`` instance, or a name string.
    In the latter case we create an ``UndefinedFunction`` instance with that
    name.

    Be aware that this is a quick workaround, not a general method to create
    special symbolic functions. If you want to create a symbolic function to be
    used by all the machinery of SymPy you should subclass the ``Function``
    class.

    Parameters
    ----------
    symfunc : ``str`` or ``UndefinedFunction`` instance
       If ``str``, then create new ``UndefinedFunction`` with this as
       name.  If ``symfunc`` is an Undefined function, create a new function
       with the same name and the implemented function attached.
    implementation : callable
       numerical implementation to be called by ``evalf()`` or ``lambdify``

    Returns
    -------
    afunc : sympy.FunctionClass instance
       function with attached implementation

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import lambdify
