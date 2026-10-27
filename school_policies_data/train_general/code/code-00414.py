    # XXX: Catching TypeError like this is a bad way of distinguishing between
    # classes and instances. The logic using this function should be rewritten
    # somehow.
    try:
        ret = expr()
    except TypeError:
        ret = expr

    return ret


def _get_state(state_class, ops, **options):
    # Try to get a state instance from the operator INSTANCES.
    # If this fails, get the class
    try:
        ret = state_class._operators_to_state(ops, **options)
    except NotImplementedError:
        ret = _make_default(state_class)

    return ret


def _get_ops(state_inst, op_classes, **options):
    # Try to get operator instances from the state INSTANCE.
    # If this fails, just return the classes
    try:
        ret = state_inst._state_to_operators(op_classes, **options)
    except NotImplementedError:
        if isinstance(op_classes, (set, tuple, frozenset)):
            ret = tuple(_make_default(x) for x in op_classes)
        else:
            ret = _make_default(op_classes)

    if isinstance(ret, set) and len(ret) == 1:
        return ret[0]

    return ret


def _make_set(ops):
    if isinstance(ops, (tuple, list, frozenset)):
        return set(ops)
    else:
        return ops
<|endoftext|>from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.matrices import Matrix


def _is_scalar(e):
    """ Helper method used in Tr"""

    # sympify to set proper attributes
    e = sympify(e)
    if isinstance(e, Expr):
        if (e.is_Integer or e.is_Float or
            e.is_Rational or e.is_Number or
            (e.is_Symbol and e.is_commutative)
                ):
            return True

    return False


def _cycle_permute(l):
    """ Cyclic permutations based on canonical ordering

    Explanation
    ===========

    This method does the sort based ascii values while
    a better approach would be to used lexicographic sort.

    TODO: Handle condition such as symbols have subscripts/superscripts
    in case of lexicographic sort

    """

    if len(l) == 1:
        return l

    min_item = min(l, key=default_sort_key)
    indices = [i for i, x in enumerate(l) if x == min_item]

    le = list(l)
    le.extend(l)  # duplicate and extend string for easy processing

    # adding the first min_item index back for easier looping
    indices.append(len(l) + indices[0])

    # create sublist of items with first item as min_item and last_item
    # in each of the sublist is item just before the next occurrence of
    # minitem in the cycle formed.
    sublist = [[le[indices[i]:indices[i + 1]]] for i in
               range(len(indices) - 1)]

    # we do comparison of strings by comparing elements
    # in each sublist
    idx = sublist.index(min(sublist))
    ordered_l = le[indices[idx]:indices[idx] + len(l)]

    return ordered_l


def _rearrange_args(l):
    """ this just moves the last arg to first position
     to enable expansion of args
     A,B,A ==> A**2,B
    """
    if len(l) == 1:
        return l

    x = list(l[-1:])
    x.extend(l[0:-1])
    return Mul(*x).args


class Tr(Expr):
    """ Generic Trace operation than can trace over:

    a) SymPy matrix
    b) operators
    c) outer products

    Parameters
    ==========
    o : operator, matrix, expr
    i : tuple/list indices (optional)

    Examples
    ========

    # TODO: Need to handle printing

    a) Trace(A+B) = Tr(A) + Tr(B)
    b) Trace(scalar*Operator) = scalar*Trace(Operator)

    >>> from sympy.physics.quantum.trace import Tr
    >>> from sympy import symbols, Matrix
    >>> a, b = symbols('a b', commutative=True)
    >>> A, B = symbols('A B', commutative=False)
    >>> Tr(a*A,[2])
    a*Tr(A)
    >>> m = Matrix([[1,2],[1,1]])
    >>> Tr(m)
    2

    """
    def __new__(cls, *args):
        """ Construct a Trace object.

        Parameters
        ==========
        args = SymPy expression
        indices = tuple/list if indices, optional

        """

        # expect no indices,int or a tuple/list/Tuple
        if (len(args) == 2):
            if not isinstance(args[1], (list, Tuple, tuple)):
                indices = Tuple(args[1])
            else:
                indices = Tuple(*args[1])

            expr = args[0]
        elif (len(args) == 1):
            indices = Tuple()
            expr = args[0]
        else:
            raise ValueError("Arguments to Tr should be of form "
                             "(expr[, [indices]])")

        if isinstance(expr, Matrix):
            return expr.trace()
        elif hasattr(expr, 'trace') and callable(expr.trace):
            #for any objects that have trace() defined e.g numpy
            return expr.trace()
        elif isinstance(expr, Add):
            return Add(*[Tr(arg, indices) for arg in expr.args])
        elif isinstance(expr, Mul):
            c_part, nc_part = expr.args_cnc()
            if len(nc_part) == 0:
                return Mul(*c_part)
            else:
                obj = Expr.__new__(cls, Mul(*nc_part), indices )
                #this check is needed to prevent cached instances
                #being returned even if len(c_part)==0
                return Mul(*c_part)*obj if len(c_part) > 0 else obj
        elif isinstance(expr, Pow):
            if (_is_scalar(expr.args[0]) and
                    _is_scalar(expr.args[1])):
                return expr
            else:
                return Expr.__new__(cls, expr, indices)
        else:
            if (_is_scalar(expr)):
                return expr

            return Expr.__new__(cls, expr, indices)

    @property
    def kind(self):
        expr = self.args[0]
        expr_kind = expr.kind
        return expr_kind.element_kind

    def doit(self, **hints):
        """ Perform the trace operation.

        #TODO: Current version ignores the indices set for partial trace.

        >>> from sympy.physics.quantum.trace import Tr
        >>> from sympy.physics.quantum.operator import OuterProduct
        >>> from sympy.physics.quantum.spin import JzKet, JzBra
        >>> t = Tr(OuterProduct(JzKet(1,1), JzBra(1,1)))
        >>> t.doit()
        1

        """
        if hasattr(self.args[0], '_eval_trace'):
            return self.args[0]._eval_trace(indices=self.args[1])

        return self

    @property
    def is_number(self):
        # TODO : improve this implementation
        return True

    #TODO: Review if the permute method is needed
    # and if it needs to return a new instance
    def permute(self, pos):
        """ Permute the arguments cyclically.

        Parameters
        ==========

        pos : integer, if positive, shift-right, else shift-left

        Examples
        ========

        >>> from sympy.physics.quantum.trace import Tr
        >>> from sympy import symbols
        >>> A, B, C, D = symbols('A B C D', commutative=False)
        >>> t = Tr(A*B*C*D)
        >>> t.permute(2)
        Tr(C*D*A*B)
        >>> t.permute(-2)
        Tr(C*D*A*B)

        """
        if pos > 0:
            pos = pos % len(self.args[0].args)
        else:
            pos = -(abs(pos) % len(self.args[0].args))

        args = list(self.args[0].args[-pos:] + self.args[0].args[0:-pos])

        return Tr(Mul(*(args)))

    def _hashable_content(self):
        if isinstance(self.args[0], Mul):
            args = _cycle_permute(_rearrange_args(self.args[0].args))
        else:
            args = [self.args[0]]

        return tuple(args) + (self.args[1], )
<|endoftext|>"""Simple Harmonic Oscillator 1-Dimension"""

from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.matrixutils import matrix_zeros

#------------------------------------------------------------------------------

class SHOOp(Operator):
    """A base class for the SHO Operators.

    We are limiting the number of arguments to be 1.

    """

    @classmethod
    def _eval_args(cls, args):
        args = QExpr._eval_args(args)
        if len(args) == 1:
            return args
        else:
            raise ValueError("Too many arguments")

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(S.Infinity)

class RaisingOp(SHOOp):
    """The Raising Operator or a^dagger.

    When a^dagger acts on a state it raises the state up by one. Taking
    the adjoint of a^dagger returns 'a', the Lowering Operator. a^dagger
    can be rewritten in terms of position and momentum. We can represent
    a^dagger as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Raising Operator and rewrite it in terms of position and
    momentum, and show that taking its adjoint returns 'a':

        >>> from sympy.physics.quantum.sho1d import RaisingOp
        >>> from sympy.physics.quantum import Dagger

        >>> ad = RaisingOp('a')
        >>> ad.rewrite('xp').doit()
        sqrt(2)*(m*omega*X - I*Px)/(2*sqrt(hbar)*sqrt(m*omega))

        >>> Dagger(ad)
        a

    Taking the commutator of a^dagger with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import RaisingOp, LoweringOp
        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> ad = RaisingOp('a')
        >>> a = LoweringOp('a')
        >>> N = NumberOp('N')
        >>> Commutator(ad, a).doit()
        -1
        >>> Commutator(ad, N).doit()
        -RaisingOp(a)

    Apply a^dagger to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import RaisingOp, SHOKet

        >>> ad = RaisingOp('a')
        >>> k = SHOKet('k')
        >>> qapply(ad*k)
        sqrt(k + 1)*|k + 1>

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import RaisingOp
        >>> from sympy.physics.quantum.represent import represent
        >>> ad = RaisingOp('a')
        >>> represent(ad, basis=N, ndim=4, format='sympy')
        Matrix([
        [0,       0,       0, 0],
        [1,       0,       0, 0],
        [0, sqrt(2),       0, 0],
        [0,       0, sqrt(3), 0]])

    """

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/sqrt(Integer(2)*hbar*m*omega))*(
            S.NegativeOne*I*Px + m*omega*X)

    def _eval_adjoint(self):
        return LoweringOp(*self.args)

    def _eval_commutator_LoweringOp(self, other):
        return S.NegativeOne

    def _eval_commutator_NumberOp(self, other):
        return S.NegativeOne*self

    def _apply_operator_SHOKet(self, ket, **options):
        temp = ket.n + S.One
        return sqrt(temp)*SHOKet(temp)

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # This logic is good but the underlying position
        # representation logic is broken.
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        # return result
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format','sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info - 1):
            value = sqrt(i + 1)
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i + 1, i] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix

    #--------------------------------------------------------------------------
    # Printing Methods
    #--------------------------------------------------------------------------

    def _print_contents(self, printer, *args):
        arg0 = printer._print(self.args[0], *args)
        return '%s(%s)' % (self.__class__.__name__, arg0)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        pform = pform**prettyForm('\N{DAGGER}')
        return pform

    def _print_contents_latex(self, printer, *args):
        arg = printer._print(self.args[0])
        return '%s^{\\dagger}' % arg

class LoweringOp(SHOOp):
    """The Lowering Operator or 'a'.

    When 'a' acts on a state it lowers the state up by one. Taking
    the adjoint of 'a' returns a^dagger, the Raising Operator. 'a'
    can be rewritten in terms of position and momentum. We can
    represent 'a' as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Lowering Operator and rewrite it in terms of position and
    momentum, and show that taking its adjoint returns a^dagger:

        >>> from sympy.physics.quantum.sho1d import LoweringOp
        >>> from sympy.physics.quantum import Dagger

        >>> a = LoweringOp('a')
        >>> a.rewrite('xp').doit()
        sqrt(2)*(m*omega*X + I*Px)/(2*sqrt(hbar)*sqrt(m*omega))

        >>> Dagger(a)
        RaisingOp(a)

    Taking the commutator of 'a' with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import LoweringOp, RaisingOp
        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> a = LoweringOp('a')
        >>> ad = RaisingOp('a')
        >>> N = NumberOp('N')
        >>> Commutator(a, ad).doit()
        1
        >>> Commutator(a, N).doit()
        a

    Apply 'a' to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import LoweringOp, SHOKet

        >>> a = LoweringOp('a')
        >>> k = SHOKet('k')
        >>> qapply(a*k)
        sqrt(k)*|k - 1>

    Taking 'a' of the lowest state will return 0:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import LoweringOp, SHOKet

        >>> a = LoweringOp('a')
        >>> k = SHOKet(0)
        >>> qapply(a*k)
        0

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import LoweringOp
        >>> from sympy.physics.quantum.represent import represent
        >>> a = LoweringOp('a')
        >>> represent(a, basis=N, ndim=4, format='sympy')
        Matrix([
        [0, 1,       0,       0],
        [0, 0, sqrt(2),       0],
        [0, 0,       0, sqrt(3)],
        [0, 0,       0,       0]])

    """

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/sqrt(Integer(2)*hbar*m*omega))*(
            I*Px + m*omega*X)

    def _eval_adjoint(self):
        return RaisingOp(*self.args)

    def _eval_commutator_RaisingOp(self, other):
        return S.One

    def _eval_commutator_NumberOp(self, other):
        return self

    def _apply_operator_SHOKet(self, ket, **options):
        temp = ket.n - Integer(1)
        if ket.n is S.Zero:
            return S.Zero
        else:
            return sqrt(ket.n)*SHOKet(temp)

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # This logic is good but the underlying position
        # representation logic is broken.
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        # return result
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info - 1):
            value = sqrt(i + 1)
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i,i + 1] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix


class NumberOp(SHOOp):
    """The Number Operator is simply a^dagger*a

    It is often useful to write a^dagger*a as simply the Number Operator
    because the Number Operator commutes with the Hamiltonian. And can be
    expressed using the Number Operator. Also the Number Operator can be
    applied to states. We can represent the Number Operator as a matrix,
    which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Number Operator and rewrite it in terms of the ladder
    operators, position and momentum operators, and Hamiltonian:

        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> N = NumberOp('N')
        >>> N.rewrite('a').doit()
        RaisingOp(a)*a
        >>> N.rewrite('xp').doit()
        -1/2 + (m**2*omega**2*X**2 + Px**2)/(2*hbar*m*omega)
        >>> N.rewrite('H').doit()
        -1/2 + H/(hbar*omega)

    Take the Commutator of the Number Operator with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import NumberOp, Hamiltonian
        >>> from sympy.physics.quantum.sho1d import RaisingOp, LoweringOp

        >>> N = NumberOp('N')
        >>> H = Hamiltonian('H')
        >>> ad = RaisingOp('a')
        >>> a = LoweringOp('a')
        >>> Commutator(N,H).doit()
        0
        >>> Commutator(N,ad).doit()
        RaisingOp(a)
        >>> Commutator(N,a).doit()
        -a

    Apply the Number Operator to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import NumberOp, SHOKet

        >>> N = NumberOp('N')
        >>> k = SHOKet('k')
        >>> qapply(N*k)
        k*|k>

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import NumberOp
        >>> from sympy.physics.quantum.represent import represent
        >>> N = NumberOp('N')
        >>> represent(N, basis=N, ndim=4, format='sympy')
        Matrix([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3]])

    """

    def _eval_rewrite_as_a(self, *args, **kwargs):
        return ad*a

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/(Integer(2)*m*hbar*omega))*(Px**2 + (
            m*omega*X)**2) - S.Half

    def _eval_rewrite_as_H(self, *args, **kwargs):
        return H/(hbar*omega) - S.Half

    def _apply_operator_SHOKet(self, ket, **options):
        return ket.n*ket

    def _eval_commutator_Hamiltonian(self, other):
        return S.Zero

    def _eval_commutator_RaisingOp(self, other):
        return other

    def _eval_commutator_LoweringOp(self, other):
        return S.NegativeOne*other

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # This logic is good but the underlying position
        # representation logic is broken.
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        # return result
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info):
            value = i
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i,i] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix


class Hamiltonian(SHOOp):
    """The Hamiltonian Operator.

    The Hamiltonian is used to solve the time-independent Schrodinger
    equation. The Hamiltonian can be expressed using the ladder operators,
    as well as by position and momentum. We can represent the Hamiltonian
    Operator as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Hamiltonian Operator and rewrite it in terms of the ladder
    operators, position and momentum, and the Number Operator:

        >>> from sympy.physics.quantum.sho1d import Hamiltonian

        >>> H = Hamiltonian('H')
        >>> H.rewrite('a').doit()
        hbar*omega*(1/2 + RaisingOp(a)*a)
        >>> H.rewrite('xp').doit()
        (m**2*omega**2*X**2 + Px**2)/(2*m)
        >>> H.rewrite('N').doit()
        hbar*omega*(1/2 + N)

    Take the Commutator of the Hamiltonian and the Number Operator:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import Hamiltonian, NumberOp

        >>> H = Hamiltonian('H')
        >>> N = NumberOp('N')
        >>> Commutator(H,N).doit()
        0

    Apply the Hamiltonian Operator to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import Hamiltonian, SHOKet

        >>> H = Hamiltonian('H')
        >>> k = SHOKet('k')
        >>> qapply(H*k)
        hbar*k*omega*|k> + hbar*omega*|k>/2

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import Hamiltonian
        >>> from sympy.physics.quantum.represent import represent

        >>> H = Hamiltonian('H')
        >>> represent(H, basis=N, ndim=4, format='sympy')
        Matrix([
        [hbar*omega/2,              0,              0,              0],
        [           0, 3*hbar*omega/2,              0,              0],
        [           0,              0, 5*hbar*omega/2,              0],
        [           0,              0,              0, 7*hbar*omega/2]])

    """

    def _eval_rewrite_as_a(self, *args, **kwargs):
        return hbar*omega*(ad*a + S.Half)

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return (S.One/(Integer(2)*m))*(Px**2 + (m*omega*X)**2)

    def _eval_rewrite_as_N(self, *args, **kwargs):
        return hbar*omega*(N + S.Half)

    def _apply_operator_SHOKet(self, ket, **options):
        return (hbar*omega*(ket.n + S.Half))*ket

    def _eval_commutator_NumberOp(self, other):
        return S.Zero

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        # This logic is good but the underlying position
        # representation logic is broken.
        # temp = self.rewrite('xp').doit()
        # result = represent(temp, basis=X)
        # return result
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info):
            value = i + S.Half
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i,i] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return hbar*omega*matrix

#------------------------------------------------------------------------------

class SHOState(State):
    """State class for SHO states"""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(S.Infinity)

    @property
    def n(self):
        return self.args[0]


class SHOKet(SHOState, Ket):
    """1D eigenket.

    Inherits from SHOState and Ket.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket
        This is usually its quantum numbers or its symbol.

    Examples
    ========

    Ket's know about their associated bra:

        >>> from sympy.physics.quantum.sho1d import SHOKet

        >>> k = SHOKet('k')
        >>> k.dual
        <k|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.sho1d.SHOBra'>

    Take the Inner Product with a bra:

        >>> from sympy.physics.quantum import InnerProduct
        >>> from sympy.physics.quantum.sho1d import SHOKet, SHOBra

        >>> k = SHOKet('k')
        >>> b = SHOBra('b')
        >>> InnerProduct(b,k).doit()
        KroneckerDelta(b, k)

    Vector representation of a numerical state ket:

        >>> from sympy.physics.quantum.sho1d import SHOKet, NumberOp
        >>> from sympy.physics.quantum.represent import represent

        >>> k = SHOKet(3)
        >>> N = NumberOp('N')
        >>> represent(k, basis=N, ndim=4)
        Matrix([
        [0],
        [0],
        [0],
        [1]])

    """

    @classmethod
    def dual_class(self):
        return SHOBra

    def _eval_innerproduct_SHOBra(self, bra, **hints):
        result = KroneckerDelta(self.n, bra.n)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        options['spmatrix'] = 'lil'
        vector = matrix_zeros(ndim_info, 1, **options)
        if isinstance(self.n, Integer):
            if self.n >= ndim_info:
                return ValueError("N-Dimension too small")
            if format == 'scipy.sparse':
                vector[int(self.n), 0] = 1.0
                vector = vector.tocsr()
            elif format == 'numpy':
                vector[int(self.n), 0] = 1.0
            else:
                vector[self.n, 0] = S.One
            return vector
        else:
            return ValueError("Not Numerical State")


class SHOBra(SHOState, Bra):
    """A time-independent Bra in SHO.

    Inherits from SHOState and Bra.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket
        This is usually its quantum numbers or its symbol.

    Examples
    ========

    Bra's know about their associated ket:

        >>> from sympy.physics.quantum.sho1d import SHOBra

        >>> b = SHOBra('b')
        >>> b.dual
        |b>
        >>> b.dual_class()
        <class 'sympy.physics.quantum.sho1d.SHOKet'>

    Vector representation of a numerical state bra:

        >>> from sympy.physics.quantum.sho1d import SHOBra, NumberOp
        >>> from sympy.physics.quantum.represent import represent

        >>> b = SHOBra(3)
        >>> N = NumberOp('N')
        >>> represent(b, basis=N, ndim=4)
        Matrix([[0, 0, 0, 1]])

    """

    @classmethod
    def dual_class(self):
        return SHOKet

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        options['spmatrix'] = 'lil'
        vector = matrix_zeros(1, ndim_info, **options)
        if isinstance(self.n, Integer):
            if self.n >= ndim_info:
                return ValueError("N-Dimension too small")
            if format == 'scipy.sparse':
                vector[0, int(self.n)] = 1.0
                vector = vector.tocsr()
            elif format == 'numpy':
                vector[0, int(self.n)] = 1.0
            else:
                vector[0, self.n] = S.One
            return vector
        else:
            return ValueError("Not Numerical State")


ad = RaisingOp('a')
a = LoweringOp('a')
H = Hamiltonian('H')
N = NumberOp('N')
omega = Symbol('omega')
m = Symbol('m')
<|endoftext|>"""An implementation of qubits and gates acting on them.

Todo:

* Update docstrings.
* Update tests.
* Implement apply using decompose.
* Implement represent using decompose or something smarter. For this to
  work we first have to implement represent for SWAP.
* Decide if we want upper index to be inclusive in the constructor.
* Fix the printing of Rk gates in plotting.
"""

from sympy.core.expr import Expr
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.functions import sqrt

from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError, QExpr
from sympy.matrices import eye
from sympy.physics.quantum.tensorproduct import matrix_tensor_product

from sympy.physics.quantum.gate import (
    Gate, HadamardGate, SwapGate, OneQubitGate, CGate, PhaseGate, TGate, ZGate
)

from sympy.functions.elementary.complexes import sign

__all__ = [
    'QFT',
    'IQFT',
    'RkGate',
    'Rk'
]

#-----------------------------------------------------------------------------
# Fourier stuff
#-----------------------------------------------------------------------------


class RkGate(OneQubitGate):
    """This is the R_k gate of the QTF."""
    gate_name = 'Rk'
    gate_name_latex = 'R'

    def __new__(cls, *args):
        if len(args) != 2:
            raise QuantumError(
                'Rk gates only take two arguments, got: %r' % args
            )
        # For small k, Rk gates simplify to other gates, using these
        # substitutions give us familiar results for the QFT for small numbers
        # of qubits.
        target = args[0]
        k = args[1]
        if k == 1:
            return ZGate(target)
        elif k == 2:
            return PhaseGate(target)
        elif k == 3:
            return TGate(target)
        args = cls._eval_args(args)
        inst = Expr.__new__(cls, *args)
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _eval_args(cls, args):
        # Fall back to this, because Gate._eval_args assumes that args is
        # all targets and can't contain duplicates.
        return QExpr._eval_args(args)

    @property
    def k(self):
        return self.label[1]

    @property
    def targets(self):
        return self.label[:1]

    @property
    def gate_name_plot(self):
        return r'$%s_%s$' % (self.gate_name_latex, str(self.k))

    def get_target_matrix(self, format='sympy'):
        if format == 'sympy':
            return Matrix([[1, 0], [0, exp(sign(self.k)*Integer(2)*pi*I/(Integer(2)**abs(self.k)))]])
        raise NotImplementedError(
            'Invalid format for the R_k gate: %r' % format)


Rk = RkGate


class Fourier(Gate):
    """Superclass of Quantum Fourier and Inverse Quantum Fourier Gates."""

    @classmethod
    def _eval_args(self, args):
        if len(args) != 2:
            raise QuantumError(
                'QFT/IQFT only takes two arguments, got: %r' % args
            )
        if args[0] >= args[1]:
            raise QuantumError("Start must be smaller than finish")
        return Gate._eval_args(args)

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        """
            Represents the (I)QFT In the Z Basis
        """
        nqubits = options.get('nqubits', 0)
        if nqubits == 0:
            raise QuantumError(
                'The number of qubits must be given as nqubits.')
        if nqubits < self.min_qubits:
            raise QuantumError(
                'The number of qubits %r is too small for the gate.' % nqubits
            )
        size = self.size
        omega = self.omega

        #Make a matrix that has the basic Fourier Transform Matrix
        arrayFT = [[omega**(
            i*j % size)/sqrt(size) for i in range(size)] for j in range(size)]
        matrixFT = Matrix(arrayFT)

        #Embed the FT Matrix in a higher space, if necessary
        if self.label[0] != 0:
            matrixFT = matrix_tensor_product(eye(2**self.label[0]), matrixFT)
        if self.min_qubits < nqubits:
            matrixFT = matrix_tensor_product(
                matrixFT, eye(2**(nqubits - self.min_qubits)))

        return matrixFT

    @property
    def targets(self):
        return range(self.label[0], self.label[1])

    @property
    def min_qubits(self):
        return self.label[1]

    @property
    def size(self):
        """Size is the size of the QFT matrix"""
        return 2**(self.label[1] - self.label[0])

    @property
    def omega(self):
        return Symbol('omega')


class QFT(Fourier):
    """The forward quantum Fourier transform."""

    gate_name = 'QFT'
    gate_name_latex = 'QFT'

    def decompose(self):
        """Decomposes QFT into elementary gates."""
        start = self.label[0]
        finish = self.label[1]
        circuit = 1
        for level in reversed(range(start, finish)):
            circuit = HadamardGate(level)*circuit
            for i in range(level - start):
                circuit = CGate(level - i - 1, RkGate(level, i + 2))*circuit
        for i in range((finish - start)//2):
            circuit = SwapGate(i + start, finish - i - 1)*circuit
        return circuit

    def _apply_operator_Qubit(self, qubits, **options):
        return qapply(self.decompose()*qubits)

    def _eval_inverse(self):
        return IQFT(*self.args)

    @property
    def omega(self):
        return exp(2*pi*I/self.size)


class IQFT(Fourier):
    """The inverse quantum Fourier transform."""

    gate_name = 'IQFT'
    gate_name_latex = '{QFT^{-1}}'

    def decompose(self):
        """Decomposes IQFT into elementary gates."""
        start = self.args[0]
        finish = self.args[1]
        circuit = 1
        for i in range((finish - start)//2):
            circuit = SwapGate(i + start, finish - i - 1)*circuit
        for level in range(start, finish):
            for i in reversed(range(level - start)):
                circuit = CGate(level - i - 1, RkGate(level, -i - 2))*circuit
            circuit = HadamardGate(level)*circuit
        return circuit

    def _eval_inverse(self):
        return QFT(*self.args)

    @property
    def omega(self):
        return exp(-2*pi*I/self.size)
<|endoftext|>"""Utilities to deal with sympy.Matrix, numpy and scipy.sparse."""

from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module

__all__ = [
    'numpy_ndarray',
    'scipy_sparse_matrix',
    'sympy_to_numpy',
    'sympy_to_scipy_sparse',
    'numpy_to_sympy',
    'scipy_sparse_to_sympy',
    'flatten_scalar',
    'matrix_dagger',
    'to_sympy',
    'to_numpy',
    'to_scipy_sparse',
    'matrix_tensor_product',
    'matrix_zeros'
]

# Conditionally define the base classes for numpy and scipy.sparse arrays
# for use in isinstance tests.

np = import_module('numpy')
if not np:
    class numpy_ndarray:
        pass
else:
    numpy_ndarray = np.ndarray  # type: ignore

scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
if not scipy:
    class scipy_sparse_matrix:
        pass
    sparse = None
else:
    sparse = scipy.sparse
    scipy_sparse_matrix = sparse.spmatrix # type: ignore


def sympy_to_numpy(m, **options):
    """Convert a SymPy Matrix/complex number to a numpy matrix or scalar."""
    if not np:
        raise ImportError
    dtype = options.get('dtype', 'complex')
    if isinstance(m, MatrixBase):
        return np.array(m.tolist(), dtype=dtype)
    elif isinstance(m, Expr):
        if m.is_Number or m.is_NumberSymbol or m == I:
            return complex(m)
    raise TypeError('Expected MatrixBase or complex scalar, got: %r' % m)


def sympy_to_scipy_sparse(m, **options):
    """Convert a SymPy Matrix/complex number to a numpy matrix or scalar."""
    if not np or not sparse:
        raise ImportError
    dtype = options.get('dtype', 'complex')
    if isinstance(m, MatrixBase):
        return sparse.csr_matrix(np.array(m.tolist(), dtype=dtype))
    elif isinstance(m, Expr):
        if m.is_Number or m.is_NumberSymbol or m == I:
            return complex(m)
    raise TypeError('Expected MatrixBase or complex scalar, got: %r' % m)


def scipy_sparse_to_sympy(m, **options):
    """Convert a scipy.sparse matrix to a SymPy matrix."""
    return MatrixBase(m.todense())


def numpy_to_sympy(m, **options):
    """Convert a numpy matrix to a SymPy matrix."""
    return MatrixBase(m)


def to_sympy(m, **options):
    """Convert a numpy/scipy.sparse matrix to a SymPy matrix."""
    if isinstance(m, MatrixBase):
        return m
    elif isinstance(m, numpy_ndarray):
        return numpy_to_sympy(m)
    elif isinstance(m, scipy_sparse_matrix):
        return scipy_sparse_to_sympy(m)
    elif isinstance(m, Expr):
        return m
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)


def to_numpy(m, **options):
    """Convert a sympy/scipy.sparse matrix to a numpy matrix."""
    dtype = options.get('dtype', 'complex')
    if isinstance(m, (MatrixBase, Expr)):
        return sympy_to_numpy(m, dtype=dtype)
    elif isinstance(m, numpy_ndarray):
        return m
    elif isinstance(m, scipy_sparse_matrix):
        return m.todense()
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)


def to_scipy_sparse(m, **options):
    """Convert a sympy/numpy matrix to a scipy.sparse matrix."""
    dtype = options.get('dtype', 'complex')
    if isinstance(m, (MatrixBase, Expr)):
        return sympy_to_scipy_sparse(m, dtype=dtype)
    elif isinstance(m, numpy_ndarray):
        if not sparse:
            raise ImportError
        return sparse.csr_matrix(m)
    elif isinstance(m, scipy_sparse_matrix):
        return m
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)


def flatten_scalar(e):
    """Flatten a 1x1 matrix to a scalar, return larger matrices unchanged."""
    if isinstance(e, MatrixBase):
        if e.shape == (1, 1):
            e = e[0]
    if isinstance(e, (numpy_ndarray, scipy_sparse_matrix)):
        if e.shape == (1, 1):
            e = complex(e[0, 0])
    return e


def matrix_dagger(e):
    """Return the dagger of a sympy/numpy/scipy.sparse matrix."""
    if isinstance(e, MatrixBase):
        return e.H
    elif isinstance(e, (numpy_ndarray, scipy_sparse_matrix)):
        return e.conjugate().transpose()
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % e)


# TODO: Move this into sympy.matrices.
def _sympy_tensor_product(*matrices):
    """Compute the kronecker product of a sequence of SymPy Matrices.
    """
    from sympy.matrices.expressions.kronecker import matrix_kronecker_product

    return matrix_kronecker_product(*matrices)


def _numpy_tensor_product(*product):
    """numpy version of tensor product of multiple arguments."""
    if not np:
        raise ImportError
    answer = product[0]
    for item in product[1:]:
        answer = np.kron(answer, item)
    return answer


def _scipy_sparse_tensor_product(*product):
    """scipy.sparse version of tensor product of multiple arguments."""
    if not sparse:
        raise ImportError
    answer = product[0]
    for item in product[1:]:
        answer = sparse.kron(answer, item)
    # The final matrices will just be multiplied, so csr is a good final
    # sparse format.
    return sparse.csr_matrix(answer)


def matrix_tensor_product(*product):
    """Compute the matrix tensor product of sympy/numpy/scipy.sparse matrices."""
    if isinstance(product[0], MatrixBase):
        return _sympy_tensor_product(*product)
    elif isinstance(product[0], numpy_ndarray):
        return _numpy_tensor_product(*product)
    elif isinstance(product[0], scipy_sparse_matrix):
        return _scipy_sparse_tensor_product(*product)


def _numpy_eye(n):
    """numpy version of complex eye."""
    if not np:
        raise ImportError
    return np.array(np.eye(n, dtype='complex'))


def _scipy_sparse_eye(n):
    """scipy.sparse version of complex eye."""
    if not sparse:
        raise ImportError
    return sparse.eye(n, n, dtype='complex')


def matrix_eye(n, **options):
    """Get the version of eye and tensor_product for a given format."""
    format = options.get('format', 'sympy')
    if format == 'sympy':
        return eye(n)
    elif format == 'numpy':
        return _numpy_eye(n)
    elif format == 'scipy.sparse':
        return _scipy_sparse_eye(n)
    raise NotImplementedError('Invalid format: %r' % format)


def _numpy_zeros(m, n, **options):
    """numpy version of zeros."""
    dtype = options.get('dtype', 'float64')
    if not np:
        raise ImportError
    return np.zeros((m, n), dtype=dtype)


def _scipy_sparse_zeros(m, n, **options):
    """scipy.sparse version of zeros."""
    spmatrix = options.get('spmatrix', 'csr')
    dtype = options.get('dtype', 'float64')
    if not sparse:
        raise ImportError
    if spmatrix == 'lil':
        return sparse.lil_matrix((m, n), dtype=dtype)
    elif spmatrix == 'csr':
        return sparse.csr_matrix((m, n), dtype=dtype)


def matrix_zeros(m, n, **options):
    """"Get a zeros matrix for a given format."""
    format = options.get('format', 'sympy')
    if format == 'sympy':
        return zeros(m, n)
    elif format == 'numpy':
        return _numpy_zeros(m, n, **options)
    elif format == 'scipy.sparse':
        return _scipy_sparse_zeros(m, n, **options)
    raise NotImplementedError('Invaild format: %r' % format)


def _numpy_matrix_to_zero(e):
    """Convert a numpy zero matrix to the zero scalar."""
    if not np:
        raise ImportError
    test = np.zeros_like(e)
    if np.allclose(e, test):
        return 0.0
    else:
        return e


def _scipy_sparse_matrix_to_zero(e):
    """Convert a scipy.sparse zero matrix to the zero scalar."""
    if not np:
        raise ImportError
    edense = e.todense()
    test = np.zeros_like(edense)
    if np.allclose(edense, test):
        return 0.0
    else:
        return e


def matrix_to_zero(e):
    """Convert a zero matrix to the scalar zero."""
    if isinstance(e, MatrixBase):
        if zeros(*e.shape) == e:
            e = S.Zero
    elif isinstance(e, numpy_ndarray):
        e = _numpy_matrix_to_zero(e)
    elif isinstance(e, scipy_sparse_matrix):
        e = _scipy_sparse_matrix_to_zero(e)
    return e
<|endoftext|>"""The anti-commutator: ``{A,B} = A*B + B*A``."""

from sympy.core.expr import Expr
from sympy.core.kind import KindDispatcher
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.kind import _OperatorKind, OperatorKind

__all__ = [
    'AntiCommutator'
]

#-----------------------------------------------------------------------------
# Anti-commutator
#-----------------------------------------------------------------------------


class AntiCommutator(Expr):
    """The standard anticommutator, in an unevaluated state.

    Explanation
    ===========

    Evaluating an anticommutator is defined [1]_ as: ``{A, B} = A*B + B*A``.
    This class returns the anticommutator in an unevaluated form.  To evaluate
    the anticommutator, use the ``.doit()`` method.

    Canonical ordering of an anticommutator is ``{A, B}`` for ``A < B``. The
    arguments of the anticommutator are put into canonical order using
    ``__cmp__``. If ``B < A``, then ``{A, B}`` is returned as ``{B, A}``.

    Parameters
    ==========

    A : Expr
        The first argument of the anticommutator {A,B}.
    B : Expr
        The second argument of the anticommutator {A,B}.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.quantum import AntiCommutator
    >>> from sympy.physics.quantum import Operator, Dagger
    >>> x, y = symbols('x,y')
    >>> A = Operator('A')
    >>> B = Operator('B')

    Create an anticommutator and use ``doit()`` to multiply them out.

    >>> ac = AntiCommutator(A,B); ac
    {A,B}
    >>> ac.doit()
    A*B + B*A

    The commutator orders it arguments in canonical order:

    >>> ac = AntiCommutator(B,A); ac
    {A,B}

    Commutative constants are factored out:

    >>> AntiCommutator(3*x*A,x*y*B)
    3*x**2*y*{A,B}

    Adjoint operations applied to the anticommutator are properly applied to
    the arguments:

    >>> Dagger(AntiCommutator(A,B))
    {Dagger(A),Dagger(B)}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Commutator
    """
    is_commutative = False

    _kind_dispatcher = KindDispatcher("AntiCommutator_kind_dispatcher", commutative=True)

    @property
    def kind(self):
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    def __new__(cls, A, B):
        r = cls.eval(A, B)
        if r is not None:
            return r
        obj = Expr.__new__(cls, A, B)
        return obj

    @classmethod
    def eval(cls, a, b):
        if not (a and b):
            return S.Zero
        if a == b:
            return Integer(2)*a**2
        if a.is_commutative or b.is_commutative:
            return Integer(2)*a*b

        # [xA,yB]  ->  xy*[A,B]
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = ca + cb
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))

        # Canonical ordering of arguments
        #The Commutator [A,B] is on canonical form if A < B.
        if a.compare(b) == 1:
            return cls(b, a)

    def doit(self, **hints):
        """ Evaluate anticommutator """
        # Keep the import of Operator here to avoid problems with
        # circular imports.
        from sympy.physics.quantum.operator import Operator
        A = self.args[0]
        B = self.args[1]
        if isinstance(A, Operator) and isinstance(B, Operator):
            try:
                comm = A._eval_anticommutator(B, **hints)
            except NotImplementedError:
                try:
                    comm = B._eval_anticommutator(A, **hints)
                except NotImplementedError:
                    comm = None
            if comm is not None:
                return comm.doit(**hints)
        return (A*B + B*A).doit(**hints)

    def _eval_adjoint(self):
        return AntiCommutator(Dagger(self.args[0]), Dagger(self.args[1]))

    def _sympyrepr(self, printer, *args):
        return "%s(%s,%s)" % (
            self.__class__.__name__, printer._print(
                self.args[0]), printer._print(self.args[1])
        )

    def _sympystr(self, printer, *args):
        return "{%s,%s}" % (
            printer._print(self.args[0]), printer._print(self.args[1]))

    def _pretty(self, printer, *args):
        pform = printer._print(self.args[0], *args)
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(printer._print(self.args[1], *args)))
        pform = prettyForm(*pform.parens(left='{', right='}'))
        return pform

    def _latex(self, printer, *args):
        return "\\left\\{%s,%s\\right\\}" % tuple([
            printer._print(arg, *args) for arg in self.args])


@AntiCommutator._kind_dispatcher.register(_OperatorKind, _OperatorKind)
def find_op_kind(e1, e2):
    """Find the kind of an anticommutator of two OperatorKinds."""
    return OperatorKind
<|endoftext|>"""Primitive circuit operations on quantum circuits."""

from functools import reduce

from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate

__all__ = [
    'kmp_table',
    'find_subcircuit',
    'replace_subcircuit',
    'convert_to_symbolic_indices',
    'convert_to_real_indices',
    'random_reduce',
    'random_insert'
]


def kmp_table(word):
    """Build the 'partial match' table of the Knuth-Morris-Pratt algorithm.

    Note: This is applicable to strings or
    quantum circuits represented as tuples.
    """

    # Current position in subcircuit
    pos = 2
    # Beginning position of candidate substring that
    # may reappear later in word
    cnd = 0
    # The 'partial match' table that helps one determine
    # the next location to start substring search
    table = []
    table.append(-1)
    table.append(0)

    while pos < len(word):
        if word[pos - 1] == word[cnd]:
            cnd = cnd + 1
            table.append(cnd)
            pos = pos + 1
        elif cnd > 0:
            cnd = table[cnd]
        else:
            table.append(0)
            pos = pos + 1

    return table


def find_subcircuit(circuit, subcircuit, start=0, end=0):
    """Finds the subcircuit in circuit, if it exists.

    Explanation
    ===========

    If the subcircuit exists, the index of the start of
    the subcircuit in circuit is returned; otherwise,
    -1 is returned.  The algorithm that is implemented
    is the Knuth-Morris-Pratt algorithm.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A tuple of Gates or Mul representing a quantum circuit
    subcircuit : tuple, Gate or Mul
        A tuple of Gates or Mul to find in circuit
    start : int
        The location to start looking for subcircuit.
        If start is the same or past end, -1 is returned.
    end : int
        The last place to look for a subcircuit.  If end
        is less than 1 (one), then the length of circuit
        is taken to be end.

    Examples
    ========

    Find the first instance of a subcircuit:

    >>> from sympy.physics.quantum.circuitutils import find_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> find_subcircuit(circuit, subcircuit)
    1

    Find the first instance starting at a specific position:

    >>> find_subcircuit(circuit, subcircuit, start=1)
    1

    >>> find_subcircuit(circuit, subcircuit, start=2)
    -1

    >>> circuit = circuit*subcircuit
    >>> find_subcircuit(circuit, subcircuit, start=2)
    4

    Find the subcircuit within some interval:

    >>> find_subcircuit(circuit, subcircuit, start=2, end=2)
    -1
    """

    if isinstance(circuit, Mul):
        circuit = circuit.args

    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args

    if len(subcircuit) == 0 or len(subcircuit) > len(circuit):
        return -1

    if end < 1:
        end = len(circuit)

    # Location in circuit
    pos = start
    # Location in the subcircuit
    index = 0
    # 'Partial match' table
    table = kmp_table(subcircuit)

    while (pos + index) < end:
        if subcircuit[index] == circuit[pos + index]:
            index = index + 1
        else:
            pos = pos + index - table[index]
            index = table[index] if table[index] > -1 else 0

        if index == len(subcircuit):
            return pos

    return -1


def replace_subcircuit(circuit, subcircuit, replace=None, pos=0):
    """Replaces a subcircuit with another subcircuit in circuit,
    if it exists.

    Explanation
    ===========

    If multiple instances of subcircuit exists, the first instance is
    replaced.  The position to being searching from (if different from
    0) may be optionally given.  If subcircuit cannot be found, circuit
    is returned.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A quantum circuit.
    subcircuit : tuple, Gate or Mul
        The circuit to be replaced.
    replace : tuple, Gate or Mul
        The replacement circuit.
    pos : int
        The location to start search and replace
        subcircuit, if it exists.  This may be used
        if it is known beforehand that multiple
        instances exist, and it is desirable to
        replace a specific instance.  If a negative number
        is given, pos will be defaulted to 0.

    Examples
    ========

    Find and remove the subcircuit:

    >>> from sympy.physics.quantum.circuitutils import replace_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)*X(0)*H(0)*Y(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> replace_subcircuit(circuit, subcircuit)
    (X(0), H(0), X(0), H(0), Y(0))

    Remove the subcircuit given a starting search point:

    >>> replace_subcircuit(circuit, subcircuit, pos=1)
    (X(0), H(0), X(0), H(0), Y(0))

    >>> replace_subcircuit(circuit, subcircuit, pos=2)
    (X(0), Z(0), Y(0), H(0), X(0), H(0), Y(0))

    Replace the subcircuit:

    >>> replacement = H(0)*Z(0)
    >>> replace_subcircuit(circuit, subcircuit, replace=replacement)
    (X(0), H(0), Z(0), H(0), X(0), H(0), Y(0))
    """

    if pos < 0:
        pos = 0

    if isinstance(circuit, Mul):
        circuit = circuit.args

    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args

    if isinstance(replace, Mul):
        replace = replace.args
    elif replace is None:
        replace = ()

    # Look for the subcircuit starting at pos
    loc = find_subcircuit(circuit, subcircuit, start=pos)

    # If subcircuit was found
    if loc > -1:
        # Get the gates to the left of subcircuit
        left = circuit[0:loc]
        # Get the gates to the right of subcircuit
        right = circuit[loc + len(subcircuit):len(circuit)]
        # Recombine the left and right side gates into a circuit
        circuit = left + replace + right

    return circuit


def _sympify_qubit_map(mapping):
    new_map = {}
    for key in mapping:
        new_map[key] = sympify(mapping[key])
    return new_map


def convert_to_symbolic_indices(seq, start=None, gen=None, qubit_map=None):
    """Returns the circuit with symbolic indices and the
    dictionary mapping symbolic indices to real indices.

    The mapping is 1 to 1 and onto (bijective).

    Parameters
    ==========

    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects, or a Mul
    start : Symbol
        An optional starting symbolic index
    gen : object
        An optional numbered symbol generator
    qubit_map : dict
        An existing mapping of symbolic indices to real indices

    All symbolic indices have the format 'i#', where # is
    some number >= 0.
    """

    if isinstance(seq, Mul):
        seq = seq.args

    # A numbered symbol generator
    index_gen = numbered_symbols(prefix='i', start=-1)
    cur_ndx = next(index_gen)

    # keys are symbolic indices; values are real indices
    ndx_map = {}

    def create_inverse_map(symb_to_real_map):
        rev_items = lambda item: (item[1], item[0])
        return dict(map(rev_items, symb_to_real_map.items()))

    if start is not None:
        if not isinstance(start, Symbol):
            msg = 'Expected Symbol for starting index, got %r.' % start
            raise TypeError(msg)
        cur_ndx = start

    if gen is not None:
        if not isinstance(gen, numbered_symbols().__class__):
            msg = 'Expected a generator, got %r.' % gen
            raise TypeError(msg)
        index_gen = gen

    if qubit_map is not None:
        if not isinstance(qubit_map, dict):
            msg = ('Expected dict for existing map, got ' +
                   '%r.' % qubit_map)
            raise TypeError(msg)
        ndx_map = qubit_map

    ndx_map = _sympify_qubit_map(ndx_map)
    # keys are real indices; keys are symbolic indices
    inv_map = create_inverse_map(ndx_map)

    sym_seq = ()
    for item in seq:
        # Nested items, so recurse
        if isinstance(item, Gate):
            result = convert_to_symbolic_indices(item.args,
                                                 qubit_map=ndx_map,
                                                 start=cur_ndx,
                                                 gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)

        elif isinstance(item, (tuple, Tuple)):
            result = convert_to_symbolic_indices(item,
                                                 qubit_map=ndx_map,
                                                 start=cur_ndx,
                                                 gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)

        elif item in inv_map:
            sym_item = inv_map[item]

        else:
            cur_ndx = next(gen)
            ndx_map[cur_ndx] = item
            inv_map[item] = cur_ndx
            sym_item = cur_ndx

        if isinstance(item, Gate):
            sym_item = item.__class__(*sym_item)

        sym_seq = sym_seq + (sym_item,)

    return sym_seq, ndx_map, cur_ndx, index_gen


def convert_to_real_indices(seq, qubit_map):
    """Returns the circuit with real indices.

    Parameters
    ==========

    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects or a Mul
    qubit_map : dict
        A dictionary mapping symbolic indices to real indices.

    Examples
    ========

    Change the symbolic indices to real integers:

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.circuitutils import convert_to_real_indices
    >>> from sympy.physics.quantum.gate import X, Y, H
    >>> i0, i1 = symbols('i:2')
    >>> index_map = {i0 : 0, i1 : 1}
    >>> convert_to_real_indices(X(i0)*Y(i1)*H(i0)*X(i1), index_map)
    (X(0), Y(1), H(0), X(1))
    """

    if isinstance(seq, Mul):
        seq = seq.args

    if not isinstance(qubit_map, dict):
        msg = 'Expected dict for qubit_map, got %r.' % qubit_map
        raise TypeError(msg)

    qubit_map = _sympify_qubit_map(qubit_map)
    real_seq = ()
    for item in seq:
        # Nested items, so recurse
        if isinstance(item, Gate):
            real_item = convert_to_real_indices(item.args, qubit_map)

        elif isinstance(item, (tuple, Tuple)):
            real_item = convert_to_real_indices(item, qubit_map)

        else:
            real_item = qubit_map[item]

        if isinstance(item, Gate):
            real_item = item.__class__(*real_item)

        real_seq = real_seq + (real_item,)

    return real_seq


def random_reduce(circuit, gate_ids, seed=None):
    """Shorten the length of a quantum circuit.

    Explanation
    ===========

    random_reduce looks for circuit identities in circuit, randomly chooses
    one to remove, and returns a shorter yet equivalent circuit.  If no
    identities are found, the same circuit is returned.

    Parameters
    ==========

    circuit : Gate tuple of Mul
        A tuple of Gates representing a quantum circuit
    gate_ids : list, GateIdentity
        List of gate identities to find in circuit
    seed : int or list
        seed used for _randrange; to override the random selection, provide a
        list of integers: the elements of gate_ids will be tested in the order
        given by the list

    """
    from sympy.core.random import _randrange

    if not gate_ids:
        return circuit

    if isinstance(circuit, Mul):
        circuit = circuit.args

    ids = flatten_ids(gate_ids)

    # Create the random integer generator with the seed
    randrange = _randrange(seed)

    # Look for an identity in the circuit
    while ids:
        i = randrange(len(ids))
        id = ids.pop(i)
        if find_subcircuit(circuit, id) != -1:
            break
    else:
        # no identity was found
        return circuit

    # return circuit with the identity removed
    return replace_subcircuit(circuit, id)


def random_insert(circuit, choices, seed=None):
    """Insert a circuit into another quantum circuit.

    Explanation
    ===========

    random_insert randomly chooses a location in the circuit to insert
    a randomly selected circuit from amongst the given choices.

    Parameters
    ==========

    circuit : Gate tuple or Mul
        A tuple or Mul of Gates representing a quantum circuit
    choices : list
        Set of circuit choices
    seed : int or list
        seed used for _randrange; to override the random selections, give
        a list two integers, [i, j] where i is the circuit location where
        choice[j] will be inserted.

    Notes
    =====

    Indices for insertion should be [0, n] if n is the length of the
    circuit.
    """
    from sympy.core.random import _randrange

    if not choices:
        return circuit

    if isinstance(circuit, Mul):
        circuit = circuit.args

    # get the location in the circuit and the element to insert from choices
    randrange = _randrange(seed)
    loc = randrange(len(circuit) + 1)
    choice = choices[randrange(len(choices))]

    circuit = list(circuit)
    circuit[loc: loc] = choice
    return tuple(circuit)

# Flatten the GateIdentity objects (with gate rules) into one single list


def flatten_ids(ids):
    collapse = lambda acc, an_id: acc + sorted(an_id.equivalent_ids,
                                        key=default_sort_key)
    ids = reduce(collapse, ids, [])
    ids.sort(key=default_sort_key)
    return ids
<|endoftext|>"""Logic for applying operators to states.

Todo:
* Sometimes the final result needs to be expanded, we should do this by hand.
"""

from sympy.concrete import Sum
from sympy.core.add import Add
from sympy.core.kind import NumberKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sympify import sympify, _sympify

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.state import State, KetBase, BraBase, Wavefunction
from sympy.physics.quantum.tensorproduct import TensorProduct

__all__ = [
    'qapply'
]


#-----------------------------------------------------------------------------
# Main code
#-----------------------------------------------------------------------------


def ip_doit_func(e):
    """Transform the inner products in an expression by calling ``.doit()``."""
    return e.replace(InnerProduct, lambda *args: InnerProduct(*args).doit())


def sum_doit_func(e):
    """Transform the sums in an expression by calling ``.doit()``."""
    return e.replace(Sum, lambda *args: Sum(*args).doit())


def qapply(e, **options):
    """Apply operators to states in a quantum expression.

    Parameters
    ==========

    e : Expr
        The expression containing operators and states. This expression tree
        will be walked to find operators acting on states symbolically.
    options : dict
        A dict of key/value pairs that determine how the operator actions
        are carried out.

        The following options are valid:

        * ``dagger``: try to apply Dagger operators to the left
          (default: False).
        * ``ip_doit``: call ``.doit()`` in inner products when they are
          encountered (default: True).
        * ``sum_doit``: call ``.doit()`` on sums when they are encountered
          (default: False). This is helpful for collapsing sums over Kronecker
          delta's that are created when calling ``qapply``.

    Returns
    =======

    e : Expr
        The original expression, but with the operators applied to states.

    Examples
    ========

        >>> from sympy.physics.quantum import qapply, Ket, Bra
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> A = k * b
        >>> A
        |k><b|
        >>> qapply(A * b.dual / (b * b.dual))
        |k>
        >>> qapply(k.dual * A / (k.dual * k))
        <b|
    """
    from sympy.physics.quantum.density import Density

    dagger = options.get('dagger', False)
    sum_doit = options.get('sum_doit', False)
    ip_doit = options.get('ip_doit', True)

    e = _sympify(e)

    # Using the kind API here helps us to narrow what types of expressions
    # we call ``ip_doit_func`` on.
    if e.kind == NumberKind:
        return ip_doit_func(e) if ip_doit else e

    # This may be a bit aggressive but ensures that everything gets expanded
    # to its simplest form before trying to apply operators. This includes
    # things like (A+B+C)*|a> and A*(|a>+|b>) and all Commutators and
    # TensorProducts. The only problem with this is that if we can't apply
    # all the Operators, we have just expanded everything.
    # TODO: don't expand the scalars in front of each Mul.
    e = e.expand(commutator=True, tensorproduct=True)

    # If we just have a raw ket, return it.
    if isinstance(e, KetBase):
        return e

    # We have an Add(a, b, c, ...) and compute
    # Add(qapply(a), qapply(b), ...)
    elif isinstance(e, Add):
        result = 0
        for arg in e.args:
            result += qapply(arg, **options)
        return result.expand()

    # For a Density operator call qapply on its state
    elif isinstance(e, Density):
        new_args = [(qapply(state, **options), prob) for (state,
                     prob) in e.args]
        return Density(*new_args)

    # For a raw TensorProduct, call qapply on its args.
    elif isinstance(e, TensorProduct):
        return TensorProduct(*[qapply(t, **options) for t in e.args])

    # For a Sum, call qapply on its function.
    elif isinstance(e, Sum):
        result = Sum(qapply(e.function, **options), *e.limits)
        result = sum_doit_func(result) if sum_doit else result
        return result

    # For a Pow, call qapply on its base.
    elif isinstance(e, Pow):
        return qapply(e.base, **options)**e.exp

    # We have a Mul where there might be actual operators to apply to kets.
    elif isinstance(e, Mul):
        c_part, nc_part = e.args_cnc()
        c_mul = Mul(*c_part)
        nc_mul = Mul(*nc_part)
        if not nc_part: # If we only have a commuting part, just return it.
            result = c_mul
        elif isinstance(nc_mul, Mul):
            result = c_mul*qapply_Mul(nc_mul, **options)
        else:
            result = c_mul*qapply(nc_mul, **options)
        if result == e and dagger:
            result = Dagger(qapply_Mul(Dagger(e), **options))
        result = ip_doit_func(result) if ip_doit else result
        result = sum_doit_func(result) if sum_doit else result
        return result

    # In all other cases (State, Operator, Pow, Commutator, InnerProduct,
    # OuterProduct) we won't ever have operators to apply to kets.
    else:
        return e


def qapply_Mul(e, **options):

    args = list(e.args)
    extra = S.One
    result = None

    # If we only have 0 or 1 args, we have nothing to do and return.
    if len(args) <= 1 or not isinstance(e, Mul):
        return e
    rhs = args.pop()
    lhs = args.pop()

    # Make sure we have two non-commutative objects before proceeding.
    if (not isinstance(rhs, Wavefunction) and sympify(rhs).is_commutative) or \
            (not isinstance(lhs, Wavefunction) and sympify(lhs).is_commutative):
        return e

    # For a Pow with an integer exponent, apply one of them and reduce the
    # exponent by one.
    if isinstance(lhs, Pow) and lhs.exp.is_Integer:
        args.append(lhs.base**(lhs.exp - 1))
        lhs = lhs.base

    # Pull OuterProduct apart
    if isinstance(lhs, OuterProduct):
        args.append(lhs.ket)
        lhs = lhs.bra

    if isinstance(rhs, OuterProduct):
        extra = rhs.bra # Append to the right of the result
        rhs = rhs.ket

    # Call .doit() on Commutator/AntiCommutator.
    if isinstance(lhs, (Commutator, AntiCommutator)):
        comm = lhs.doit()
        if isinstance(comm, Add):
            return qapply(
                e.func(*(args + [comm.args[0], rhs])) +
                e.func(*(args + [comm.args[1], rhs])),
                **options
            )*extra
        else:
            return qapply(e.func(*args)*comm*rhs, **options)*extra

    # Apply tensor products of operators to states
    if isinstance(lhs, TensorProduct) and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in lhs.args) and \
            isinstance(rhs, TensorProduct) and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in rhs.args) and \
            len(lhs.args) == len(rhs.args):
        result = TensorProduct(*[qapply(lhs.args[n]*rhs.args[n], **options) for n in range(len(lhs.args))]).expand(tensorproduct=True)
        return qapply_Mul(e.func(*args), **options)*result*extra

    # For Sums, move the Sum to the right.
    if isinstance(rhs, Sum):
        if isinstance(lhs, Sum):
            if set(lhs.variables).intersection(set(rhs.variables)):
                raise ValueError('Duplicated dummy indices in separate sums in qapply.')
            limits = lhs.limits + rhs.limits
            result = Sum(qapply(lhs.function*rhs.function, **options), *limits)
            return qapply_Mul(e.func(*args)*result, **options)
        else:
            result = Sum(qapply(lhs*rhs.function, **options), *rhs.limits)
            return qapply_Mul(e.func(*args)*result, **options)

    if isinstance(lhs, Sum):
        result = Sum(qapply(lhs.function*rhs, **options), *lhs.limits)
        return qapply_Mul(e.func(*args)*result, **options)

    # Now try to actually apply the operator and build an inner product.
    _apply = getattr(lhs, '_apply_operator', None)
    if _apply is not None:
        try:
            result = _apply(rhs, **options)
        except NotImplementedError:
            result = None
    else:
        result = None

    if result is None:
        _apply_right = getattr(rhs, '_apply_from_right_to', None)
        if _apply_right is not None:
            try:
                result = _apply_right(lhs, **options)
            except NotImplementedError:
                result = None

    if result is None:
        if isinstance(lhs, BraBase) and isinstance(rhs, KetBase):
            result = InnerProduct(lhs, rhs)

    # TODO: I may need to expand before returning the final result.
    if isinstance(result, (int, complex, float)):
        return _sympify(result)
    elif result is None:
        if len(args) == 0:
            # We had two args to begin with so args=[].
            return e
        else:
            return qapply_Mul(e.func(*(args + [lhs])), **options)*rhs*extra
    elif isinstance(result, InnerProduct):
        return result*qapply_Mul(e.func(*args), **options)*extra
    else:  # result is a scalar times a Mul, Add or TensorProduct
        return qapply(e.func(*args)*result, **options)*extra
<|endoftext|>"""1D quantum particle in a box."""

from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.sets.sets import Interval

from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.constants import hbar
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import L2

m = Symbol('m')
L = Symbol('L')


__all__ = [
    'PIABHamiltonian',
    'PIABKet',
    'PIABBra'
]


class PIABHamiltonian(HermitianOperator):
    """Particle in a box Hamiltonian operator."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PIABKet(self, ket, **options):
        n = ket.label[0]
        return (n**2*pi**2*hbar**2)/(2*m*L**2)*ket


class PIABKet(Ket):
    """Particle in a box eigenket."""

    @classmethod
    def _eval_hilbert_space(cls, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    @classmethod
    def dual_class(self):
        return PIABBra

    def _represent_default_basis(self, **options):
        return self._represent_XOp(None, **options)

    def _represent_XOp(self, basis, **options):
        x = Symbol('x')
        n = Symbol('n')
        subs_info = options.get('subs', {})
        return sqrt(2/L)*sin(n*pi*x/L).subs(subs_info)

    def _eval_innerproduct_PIABBra(self, bra):
        return KroneckerDelta(bra.label[0], self.label[0])


class PIABBra(Bra):
    """Particle in a box eigenbra."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    @classmethod
    def dual_class(self):
        return PIABKet
<|endoftext|>"""Constants (like hbar) related to quantum mechanics."""

from sympy.core.numbers import NumberSymbol
from sympy.core.singleton import Singleton
from sympy.printing.pretty.stringpict import prettyForm
import mpmath.libmp as mlib

#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

__all__ = [
    'hbar',
    'HBar',
]


class HBar(NumberSymbol, metaclass=Singleton):
    """Reduced Plank's constant in numerical and symbolic form [1]_.

    Examples
    ========

        >>> from sympy.physics.quantum.constants import hbar
        >>> hbar.evalf()
        1.05457162000000e-34

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Planck_constant
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True

    __slots__ = ()

    def _as_mpf_val(self, prec):
        return mlib.from_float(1.05457162e-34, prec)

    def _sympyrepr(self, printer, *args):
        return 'HBar()'

    def _sympystr(self, printer, *args):
        return 'hbar'

    def _pretty(self, printer, *args):
        if printer._use_unicode:
            return prettyForm('\N{PLANCK CONSTANT OVER TWO PI}')
        return prettyForm('hbar')

    def _latex(self, printer, *args):
        return r'\hbar'

# Create an instance for everyone to use.
hbar = HBar()
<|endoftext|>#TODO:
# -Implement Clebsch-Gordan symmetries
# -Improve simplification method
# -Implement new simplifications
"""Clebsch-Gordon Coefficients."""

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict

from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE

__all__ = [
    'CG',
    'Wigner3j',
    'Wigner6j',
    'Wigner9j',
    'cg_simp'
]

#-----------------------------------------------------------------------------
# CG Coefficients
#-----------------------------------------------------------------------------


class Wigner3j(Expr):
    """Class for the Wigner-3j symbols.

    Explanation
    ===========

    Wigner 3j-symbols are coefficients determined by the coupling of
    two angular momenta. When created, they are expressed as symbolic
    quantities that, for numerical parameters, can be evaluated using the
    ``.doit()`` method [1]_.

    Parameters
    ==========

    j1, m1, j2, m2, j3, m3 : Number, Symbol
        Terms determining the angular momentum of coupled angular momentum
        systems.

    Examples
    ========

    Declare a Wigner-3j coefficient and calculate its value

        >>> from sympy.physics.quantum.cg import Wigner3j
        >>> w3j = Wigner3j(6,0,4,0,2,0)
        >>> w3j
        Wigner3j(6, 0, 4, 0, 2, 0)
        >>> w3j.doit()
        sqrt(715)/143

    See Also
    ========

    CG: Clebsch-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    is_commutative = True

    def __new__(cls, j1, m1, j2, m2, j3, m3):
        args = map(sympify, (j1, m1, j2, m2, j3, m3))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def m1(self):
        return self.args[1]

    @property
    def j2(self):
        return self.args[2]

    @property
    def m2(self):
        return self.args[3]

    @property
    def j3(self):
        return self.args[4]

    @property
    def m3(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.m1)),
            (printer._print(self.j2), printer._print(self.m2)),
            (printer._print(self.j3), printer._print(self.m3)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(2))
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens())
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j3,
                    self.m1, self.m2, self.m3))
        return r'\left(\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right)' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_3j(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)


class CG(Wigner3j):
    r"""Class for Clebsch-Gordan coefficient.

    Explanation
    ===========

    Clebsch-Gordan coefficients describe the angular momentum coupling between
    two systems. The coefficients give the expansion of a coupled total angular
    momentum state and an uncoupled tensor product state. The Clebsch-Gordan
    coefficients are defined as [1]_:

    .. math ::
        C^{j_3,m_3}_{j_1,m_1,j_2,m_2} = \left\langle j_1,m_1;j_2,m_2 | j_3,m_3\right\rangle

    Parameters
    ==========

    j1, m1, j2, m2 : Number, Symbol
        Angular momenta of states 1 and 2.

    j3, m3: Number, Symbol
        Total angular momentum of the coupled system.

    Examples
    ========

    Define a Clebsch-Gordan coefficient and evaluate its value

        >>> from sympy.physics.quantum.cg import CG
        >>> from sympy import S
        >>> cg = CG(S(3)/2, S(3)/2, S(1)/2, -S(1)/2, 1, 1)
        >>> cg
        CG(3/2, 3/2, 1/2, -1/2, 1, 1)
        >>> cg.doit()
        sqrt(3)/2
        >>> CG(j1=S(1)/2, m1=-S(1)/2, j2=S(1)/2, m2=+S(1)/2, j3=1, m3=0).doit()
        sqrt(2)/2


    Compare [2]_.

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    .. [2] `Clebsch-Gordan Coefficients, Spherical Harmonics, and d Functions
        <https://pdg.lbl.gov/2020/reviews/rpp2020-rev-clebsch-gordan-coefs.pdf>`_
        in P.A. Zyla *et al.* (Particle Data Group), Prog. Theor. Exp. Phys.
        2020, 083C01 (2020).
    """
    precedence = PRECEDENCE["Pow"] - 1

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return clebsch_gordan(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)

    def _pretty(self, printer, *args):
        bot = printer._print_seq(
            (self.j1, self.m1, self.j2, self.m2), delimiter=',')
        top = printer._print_seq((self.j3, self.m3), delimiter=',')

        pad = max(top.width(), bot.width())
        bot = prettyForm(*bot.left(' '))
        top = prettyForm(*top.left(' '))

        if not pad == bot.width():
            bot = prettyForm(*bot.right(' '*(pad - bot.width())))
        if not pad == top.width():
            top = prettyForm(*top.right(' '*(pad - top.width())))
        s = stringPict('C' + ' '*pad)
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.above(top))
        return s

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j3, self.m3, self.j1,
                    self.m1, self.j2, self.m2))
        return r'C^{%s,%s}_{%s,%s,%s,%s}' % tuple(label)


class Wigner6j(Expr):
    """Class for the Wigner-6j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j, j23):
        args = map(sympify, (j1, j2, j12, j3, j, j23))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j(self):
        return self.args[4]

    @property
    def j23(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.j3)),
            (printer._print(self.j2), printer._print(self.j)),
            (printer._print(self.j12), printer._print(self.j23)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(2))
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12,
                    self.j3, self.j, self.j23))
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_6j(self.j1, self.j2, self.j12, self.j3, self.j, self.j23)


class Wigner9j(Expr):
    """Class for the Wigner-9j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j4, j34, j13, j24, j):
        args = map(sympify, (j1, j2, j12, j3, j4, j34, j13, j24, j))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j4(self):
        return self.args[4]

    @property
    def j34(self):
        return self.args[5]

    @property
    def j13(self):
        return self.args[6]

    @property
    def j24(self):
        return self.args[7]

    @property
    def j(self):
        return self.args[8]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = (
            (printer._print(
                self.j1), printer._print(self.j3), printer._print(self.j13)),
            (printer._print(
                self.j2), printer._print(self.j4), printer._print(self.j24)),
            (printer._print(self.j12), printer._print(self.j34), printer._print(self.j)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(3))
        D = None
        for i in range(3):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12, self.j3,
                self.j4, self.j34, self.j13, self.j24, self.j))
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_9j(self.j1, self.j2, self.j12, self.j3, self.j4, self.j34, self.j13, self.j24, self.j)


def cg_simp(e):
    """Simplify and combine CG coefficients.

    Explanation
    ===========

    This function uses various symmetry and properties of sums and
    products of Clebsch-Gordan coefficients to simplify statements
    involving these terms [1]_.

    Examples
    ========

    Simplify the sum over CG(a,alpha,0,0,a,alpha) for all alpha to
    2*a+1

        >>> from sympy.physics.quantum.cg import CG, cg_simp
        >>> a = CG(1,1,0,0,1,1)
        >>> b = CG(1,0,0,0,1,0)
        >>> c = CG(1,-1,0,0,1,-1)
        >>> cg_simp(a+b+c)
        3

    See Also
    ========

    CG: Clebsh-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """
    if isinstance(e, Add):
        return _cg_simp_add(e)
    elif isinstance(e, Sum):
        return _cg_simp_sum(e)
    elif isinstance(e, Mul):
        return Mul(*[cg_simp(arg) for arg in e.args])
    elif isinstance(e, Pow):
        return Pow(cg_simp(e.base), e.exp)
    else:
        return e


def _cg_simp_add(e):
    #TODO: Improve simplification method
    """Takes a sum of terms involving Clebsch-Gordan coefficients and
    simplifies the terms.

    Explanation
    ===========

    First, we create two lists, cg_part, which is all the terms involving CG
    coefficients, and other_part, which is all other terms. The cg_part list
    is then passed to the simplification methods, which return the new cg_part
    and any additional terms that are added to other_part
    """
    cg_part = []
    other_part = []

    e = expand(e)
    for arg in e.args:
        if arg.has(CG):
            if isinstance(arg, Sum):
                other_part.append(_cg_simp_sum(arg))
            elif isinstance(arg, Mul):
                terms = 1
                for term in arg.args:
                    if isinstance(term, Sum):
                        terms *= _cg_simp_sum(term)
                    else:
                        terms *= term
                if terms.has(CG):
                    cg_part.append(terms)
                else:
                    other_part.append(terms)
            else:
                cg_part.append(arg)
        else:
            other_part.append(arg)

    cg_part, other = _check_varsh_871_1(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_871_2(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_872_9(cg_part)
    other_part.append(other)
    return Add(*cg_part) + Add(*other_part)


def _check_varsh_871_1(term_list):
    # Sum( CG(a,alpha,b,0,a,alpha), (alpha, -a, a)) == KroneckerDelta(b,0)
    a, alpha, b, lt = map(Wild, ('a', 'alpha', 'b', 'lt'))
    expr = lt*CG(a, alpha, b, 0, a, alpha)
    simp = (2*a + 1)*KroneckerDelta(b, 0)
    sign = lt/abs(lt)
    build_expr = 2*a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, lt), (a, b), build_expr, index_expr)


def _check_varsh_871_2(term_list):
    # Sum((-1)**(a-alpha)*CG(a,alpha,a,-alpha,c,0),(alpha,-a,a))
    a, alpha, c, lt = map(Wild, ('a', 'alpha', 'c', 'lt'))
    expr = lt*CG(a, alpha, a, -alpha, c, 0)
    simp = sqrt(2*a + 1)*KroneckerDelta(c, 0)
    sign = (-1)**(a - alpha)*lt/abs(lt)
    build_expr = 2*a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, c, lt), (a, c), build_expr, index_expr)


def _check_varsh_872_9(term_list):
    # Sum( CG(a,alpha,b,beta,c,gamma)*CG(a,alpha',b,beta',c,gamma), (gamma, -c, c), (c, abs(a-b), a+b))
    a, alpha, alphap, b, beta, betap, c, gamma, lt = map(Wild, (
        'a', 'alpha', 'alphap', 'b', 'beta', 'betap', 'c', 'gamma', 'lt'))
    # Case alpha==alphap, beta==betap

    # For numerical alpha,beta
    expr = lt*CG(a, alpha, b, beta, c, gamma)**2
    simp = S.One
    sign = lt/abs(lt)
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    term_list, other1 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)

    # For symbolic alpha,beta
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x)*(x + y + 1)
    index_expr = (c - x)*(x + c) + c + gamma
    term_list, other2 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)

    # Case alpha!=alphap or beta!=betap
    # Note: this only works with leading term of 1, pattern matching is unable to match when there is a Wild leading term
    # For numerical alpha,alphap,beta,betap
    expr = CG(a, alpha, b, beta, c, gamma)*CG(a, alphap, b, betap, c, gamma)
    simp = KroneckerDelta(alpha, alphap)*KroneckerDelta(beta, betap)
    sign = S.One
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    term_list, other3 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)

    # For symbolic alpha,alphap,beta,betap
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x)*(x + y + 1)
    index_expr = (c - x)*(x + c) + c + gamma
    term_list, other4 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)

    return term_list, other1 + other2 + other4


def _check_cg_simp(expr, simp, sign, lt, term_list, variables, dep_variables, build_index_expr, index_expr):
    """ Checks for simplifications that can be made, returning a tuple of the
    simplified list of terms and any terms generated by simplification.

    Parameters
    ==========

    expr: expression
        The expression with Wild terms that will be matched to the terms in
        the sum

    simp: expression
        The expression with Wild terms that is substituted in place of the CG
        terms in the case of simplification

    sign: expression
        The expression with Wild terms denoting the sign that is on expr that
        must match

    lt: expression
        The expression with Wild terms that gives the leading term of the
        matched expr

    term_list: list
        A list of all of the terms is the sum to be simplified

    variables: list
        A list of all the variables that appears in expr

    dep_variables: list
        A list of the variables that must match for all the terms in the sum,
        i.e. the dependent variables

    build_index_expr: expression
        Expression with Wild terms giving the number of elements in cg_index

    index_expr: expression
        Expression with Wild terms giving the index terms have when storing
        them to cg_index

    """
    other_part = 0
    i = 0
    while i < len(term_list):
        sub_1 = _check_cg(term_list[i], expr, len(variables))
        if sub_1 is None:
            i += 1
            continue
        if not build_index_expr.subs(sub_1).is_number:
            i += 1
            continue
        sub_dep = [(x, sub_1[x]) for x in dep_variables]
        cg_index = [None]*build_index_expr.subs(sub_1)
        for j in range(i, len(term_list)):
            sub_2 = _check_cg(term_list[j], expr.subs(sub_dep), len(variables) - len(dep_variables), sign=(sign.subs(sub_1), sign.subs(sub_dep)))
            if sub_2 is None:
                continue
            if not index_expr.subs(sub_dep).subs(sub_2).is_number:
                continue
            cg_index[index_expr.subs(sub_dep).subs(sub_2)] = j, expr.subs(lt, 1).subs(sub_dep).subs(sub_2), lt.subs(sub_2), sign.subs(sub_dep).subs(sub_2)
        if not any(i is None for i in cg_index):
            min_lt = min(*[ abs(term[2]) for term in cg_index ])
            indices = [ term[0] for term in cg_index]
            indices.sort()
            indices.reverse()
            [ term_list.pop(j) for j in indices ]
            for term in cg_index:
                if abs(term[2]) > min_lt:
                    term_list.append( (term[2] - min_lt*term[3])*term[1] )
            other_part += min_lt*(sign*simp).subs(sub_1)
        else:
            i += 1
    return term_list, other_part


def _check_cg(cg_term, expr, length, sign=None):
    """Checks whether a term matches the given expression"""
    # TODO: Check for symmetries
    matches = cg_term.match(expr)
    if matches is None:
        return
    if sign is not None:
        if not isinstance(sign, tuple):
            raise TypeError('sign must be a tuple')
        if not sign[0] == (sign[1]).subs(matches):
            return
    if len(matches) == length:
        return matches


def _cg_simp_sum(e):
    e = _check_varsh_sum_871_1(e)
    e = _check_varsh_sum_871_2(e)
    e = _check_varsh_sum_872_4(e)
    return e


def _check_varsh_sum_871_1(e):
    a = Wild('a')
    alpha = symbols('alpha')
    b = Wild('b')
    match = e.match(Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)))
    if match is not None and len(match) == 2:
        return ((2*a + 1)*KroneckerDelta(b, 0)).subs(match)
    return e


def _check_varsh_sum_871_2(e):
    a = Wild('a')
    alpha = symbols('alpha')
    c = Wild('c')
    match = e.match(
        Sum((-1)**(a - alpha)*CG(a, alpha, a, -alpha, c, 0), (alpha, -a, a)))
    if match is not None and len(match) == 2:
        return (sqrt(2*a + 1)*KroneckerDelta(c, 0)).subs(match)
    return e


def _check_varsh_sum_872_4(e):
    alpha = symbols('alpha')
    beta = symbols('beta')
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    cp = Wild('cp')
    gamma = Wild('gamma')
    gammap = Wild('gammap')
    cg1 = CG(a, alpha, b, beta, c, gamma)
    cg2 = CG(a, alpha, b, beta, cp, gammap)
    match1 = e.match(Sum(cg1*cg2, (alpha, -a, a), (beta, -b, b)))
    if match1 is not None and len(match1) == 6:
        return (KroneckerDelta(c, cp)*KroneckerDelta(gamma, gammap)).subs(match1)
    match2 = e.match(Sum(cg1**2, (alpha, -a, a), (beta, -b, b)))
    if match2 is not None and len(match2) == 4:
        return S.One
    return e


def _cg_list(term):
    if isinstance(term, CG):
        return (term,), 1, 1
    cg = []
    coeff = 1
    if not isinstance(term, (Mul, Pow)):
        raise NotImplementedError('term must be CG, Add, Mul or Pow')
    if isinstance(term, Pow) and term.exp.is_number:
        if term.exp.is_number:
            [ cg.append(term.base) for _ in range(term.exp) ]
        else:
            return (term,), 1, 1
    if isinstance(term, Mul):
        for arg in term.args:
            if isinstance(arg, CG):
                cg.append(arg)
            else:
                coeff *= arg
        return cg, coeff, coeff/abs(coeff)
<|endoftext|>"""Pauli operators and states"""

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum import Operator, Ket, Bra
from sympy.physics.quantum import ComplexSpace
from sympy.matrices import Matrix
from sympy.functions.special.tensor_functions import KroneckerDelta

__all__ = [
    'SigmaX', 'SigmaY', 'SigmaZ', 'SigmaMinus', 'SigmaPlus', 'SigmaZKet',
    'SigmaZBra', 'qsimplify_pauli'
]


class SigmaOpBase(Operator):
    """Pauli sigma operator, base class"""

    @property
    def name(self):
        return self.args[0]

    @property
    def use_name(self):
        return bool(self.args[0]) is not False

    @classmethod
    def default_args(self):
        return (False,)

    def __new__(cls, *args, **hints):
        return Operator.__new__(cls, *args, **hints)

    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero


class SigmaX(SigmaOpBase):
    """Pauli sigma x operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaX
    >>> sx = SigmaX()
    >>> sx
    SigmaX()
    >>> represent(sx)
    Matrix([
    [0, 1],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args, **hints)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return 2 * I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return - 2 * I * SigmaY(self.name)

    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_x^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_x}'

    def _print_contents(self, printer, *args):
        return 'SigmaX()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaX(self.name).__pow__(int(e) % 2)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaY(SigmaOpBase):
    """Pauli sigma y operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaY
    >>> sy = SigmaY()
    >>> sy
    SigmaY()
    >>> represent(sy)
    Matrix([
    [0, -I],
    [I,  0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return 2 * I * SigmaX(self.name)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return - 2 * I * SigmaZ(self.name)

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_y^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_y}'

    def _print_contents(self, printer, *args):
        return 'SigmaY()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaY(self.name).__pow__(int(e) % 2)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, -I], [I, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZ(SigmaOpBase):
    """Pauli sigma z operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent
    >>> from sympy.physics.quantum.pauli import SigmaZ
    >>> sz = SigmaZ()
    >>> sz ** 3
    SigmaZ()
    >>> represent(sz)
    Matrix([
    [1,  0],
    [0, -1]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return 2 * I * SigmaY(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return - 2 * I * SigmaX(self.name)

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return S.Zero

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_z^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_z}'

    def _print_contents(self, printer, *args):
        return 'SigmaZ()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaZ(self.name).__pow__(int(e) % 2)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[1, 0], [0, -1]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaMinus(SigmaOpBase):
    """Pauli sigma minus operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent, Dagger
    >>> from sympy.physics.quantum.pauli import SigmaMinus
    >>> sm = SigmaMinus()
    >>> sm
    SigmaMinus()
    >>> Dagger(sm)
    SigmaPlus()
    >>> represent(sm)
    Matrix([
    [0, 0],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return -SigmaZ(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        return 2 * self

    def _eval_commutator_SigmaMinus(self, other, **hints):
        return SigmaZ(self.name)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.One

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return I * S.NegativeOne

    def _eval_anticommutator_SigmaPlus(self, other, **hints):
        return S.One

    def _eval_adjoint(self):
        return SigmaPlus(self.name)

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return S.Zero

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_-^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_-}'

    def _print_contents(self, printer, *args):
        return 'SigmaMinus()'

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 0], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaPlus(SigmaOpBase):
    """Pauli sigma plus operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympy.physics.quantum import represent, Dagger
    >>> from sympy.physics.quantum.pauli import SigmaPlus
    >>> sp = SigmaPlus()
    >>> sp
    SigmaPlus()
    >>> Dagger(sp)
    SigmaMinus()
    >>> represent(sp)
    Matrix([
    [0, 1],
    [0, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return SigmaZ(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return S.Zero
        else:
            return -2 * self

    def _eval_commutator_SigmaMinus(self, other, **hints):
        return SigmaZ(self.name)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return S.Zero

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return S.One

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return I

    def _eval_anticommutator_SigmaMinus(self, other, **hints):
        return S.One

    def _eval_adjoint(self):
        return SigmaMinus(self.name)

    def _eval_mul(self, other):
        return self * other

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return S.Zero

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_+^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_+}'

    def _print_contents(self, printer, *args):
        return 'SigmaPlus()'

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [0, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZKet(Ket):
    """Ket for a two-level system quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return SigmaZBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2)

    def _eval_innerproduct_SigmaZBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_from_right_to_SigmaZ(self, op, **options):
        if self.n == 0:
            return self
        else:
            return S.NegativeOne * self

    def _apply_from_right_to_SigmaX(self, op, **options):
        return SigmaZKet(1) if self.n == 0 else SigmaZKet(0)

    def _apply_from_right_to_SigmaY(self, op, **options):
        return I * SigmaZKet(1) if self.n == 0 else (-I) * SigmaZKet(0)

    def _apply_from_right_to_SigmaMinus(self, op, **options):
        if self.n == 0:
            return SigmaZKet(1)
        else:
            return S.Zero

    def _apply_from_right_to_SigmaPlus(self, op, **options):
        if self.n == 0:
            return S.Zero
        else:
            return SigmaZKet(0)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[1], [0]]) if self.n == 0 else Matrix([[0], [1]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZBra(Bra):
    """Bra for a two-level quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        if n not in (0, 1):
            raise ValueError("n must be 0 or 1")
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return SigmaZKet


def _qsimplify_pauli_product(a, b):
    """
    Internal helper function for simplifying products of Pauli operators.
    """
    if not (isinstance(a, SigmaOpBase) and isinstance(b, SigmaOpBase)):
        return Mul(a, b)

    if a.name != b.name:
        # Pauli matrices with different labels commute; sort by name
        if a.name < b.name:
            return Mul(a, b)
        else:
            return Mul(b, a)

    elif isinstance(a, SigmaX):

        if isinstance(b, SigmaX):
            return S.One

        if isinstance(b, SigmaY):
            return I * SigmaZ(a.name)

        if isinstance(b, SigmaZ):
            return - I * SigmaY(a.name)

        if isinstance(b, SigmaMinus):
            return (S.Half + SigmaZ(a.name)/2)

        if isinstance(b, SigmaPlus):
            return (S.Half - SigmaZ(a.name)/2)

    elif isinstance(a, SigmaY):

        if isinstance(b, SigmaX):
            return - I * SigmaZ(a.name)

        if isinstance(b, SigmaY):
            return S.One

        if isinstance(b, SigmaZ):
            return I * SigmaX(a.name)

        if isinstance(b, SigmaMinus):
            return -I * (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaPlus):
            return I * (S.One - SigmaZ(a.name))/2

    elif isinstance(a, SigmaZ):

        if isinstance(b, SigmaX):
            return I * SigmaY(a.name)

        if isinstance(b, SigmaY):
            return - I * SigmaX(a.name)

        if isinstance(b, SigmaZ):
            return S.One

        if isinstance(b, SigmaMinus):
            return - SigmaMinus(a.name)

        if isinstance(b, SigmaPlus):
            return SigmaPlus(a.name)

    elif isinstance(a, SigmaMinus):

        if isinstance(b, SigmaX):
            return (S.One - SigmaZ(a.name))/2

        if isinstance(b, SigmaY):
            return - I * (S.One - SigmaZ(a.name))/2

        if isinstance(b, SigmaZ):
            # (SigmaX(a.name) - I * SigmaY(a.name))/2
            return SigmaMinus(b.name)

        if isinstance(b, SigmaMinus):
            return S.Zero

        if isinstance(b, SigmaPlus):
            return S.Half - SigmaZ(a.name)/2

    elif isinstance(a, SigmaPlus):

        if isinstance(b, SigmaX):
            return (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaY):
            return I * (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaZ):
            #-(SigmaX(a.name) + I * SigmaY(a.name))/2
            return -SigmaPlus(a.name)

        if isinstance(b, SigmaMinus):
            return (S.One + SigmaZ(a.name))/2

        if isinstance(b, SigmaPlus):
            return S.Zero

    else:
        return a * b


def qsimplify_pauli(e):
    """
    Simplify an expression that includes products of pauli operators.

    Parameters
    ==========

    e : expression
        An expression that contains products of Pauli operators that is
        to be simplified.

    Examples
    ========

    >>> from sympy.physics.quantum.pauli import SigmaX, SigmaY
    >>> from sympy.physics.quantum.pauli import qsimplify_pauli
    >>> sx, sy = SigmaX(), SigmaY()
    >>> sx * sy
    SigmaX()*SigmaY()
    >>> qsimplify_pauli(sx * sy)
    I*SigmaZ()
    """
    if isinstance(e, Operator):
        return e

    if isinstance(e, (Add, Pow, exp)):
        t = type(e)
        return t(*(qsimplify_pauli(arg) for arg in e.args))

    if isinstance(e, Mul):

        c, nc = e.args_cnc()

        nc_s = []
        while nc:
            curr = nc.pop(0)

            while (len(nc) and
                   isinstance(curr, SigmaOpBase) and
                   isinstance(nc[0], SigmaOpBase) and
                   curr.name == nc[0].name):

                x = nc.pop(0)
                y = _qsimplify_pauli_product(curr, x)
                c1, nc1 = y.args_cnc()
                curr = Mul(*nc1)
                c = c + c1

            nc_s.append(curr)

        return Mul(*c) * Mul(*nc_s)

    return e
<|endoftext|># Names exposed by 'from sympy.physics.quantum import *'

__all__ = [
    'AntiCommutator',

    'qapply',

    'Commutator',

    'Dagger',

    'HilbertSpaceError', 'HilbertSpace', 'TensorProductHilbertSpace',
    'TensorPowerHilbertSpace', 'DirectSumHilbertSpace', 'ComplexSpace', 'L2',
    'FockSpace',

    'InnerProduct',

    'Operator', 'HermitianOperator', 'UnitaryOperator', 'IdentityOperator',
    'OuterProduct', 'DifferentialOperator',

    'represent', 'rep_innerproduct', 'rep_expectation', 'integrate_result',
    'get_basis', 'enumerate_states',

    'KetBase', 'BraBase', 'StateBase', 'State', 'Ket', 'Bra', 'TimeDepState',
    'TimeDepBra', 'TimeDepKet', 'OrthogonalKet', 'OrthogonalBra',
    'OrthogonalState', 'Wavefunction',

    'TensorProduct', 'tensor_product_simp',

    'hbar', 'HBar',

    '_postprocess_state_mul', '_postprocess_state_pow'
]

from .anticommutator import AntiCommutator

from .qapply import qapply

from .commutator import Commutator

from .dagger import Dagger

from .hilbert import (HilbertSpaceError, HilbertSpace,
        TensorProductHilbertSpace, TensorPowerHilbertSpace,
        DirectSumHilbertSpace, ComplexSpace, L2, FockSpace)

from .innerproduct import InnerProduct

from .operator import (Operator, HermitianOperator, UnitaryOperator,
        IdentityOperator, OuterProduct, DifferentialOperator)

from .represent import (represent, rep_innerproduct, rep_expectation,
        integrate_result, get_basis, enumerate_states)

from .state import (KetBase, BraBase, StateBase, State, Ket, Bra,
        TimeDepState, TimeDepBra, TimeDepKet, OrthogonalKet,
        OrthogonalBra, OrthogonalState, Wavefunction)

from .tensorproduct import TensorProduct, tensor_product_simp

from .constants import hbar, HBar

# These are private, but need to be imported so they are registered
# as postprocessing transformers with Mul and Pow.
from .transforms import _postprocess_state_mul, _postprocess_state_pow
<|endoftext|>from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray, scipy_sparse_matrix,
    to_sympy, to_numpy, to_scipy_sparse
)

__all__ = [
    'QuantumError',
    'QExpr'
]


#-----------------------------------------------------------------------------
# Error handling
#-----------------------------------------------------------------------------

class QuantumError(Exception):
    pass


def _qsympify_sequence(seq):
    """Convert elements of a sequence to standard form.

    This is like sympify, but it performs special logic for arguments passed
    to QExpr. The following conversions are done:

    * (list, tuple, Tuple) => _qsympify_sequence each element and convert
      sequence to a Tuple.
    * basestring => Symbol
    * Matrix => Matrix
    * other => sympify

    Strings are passed to Symbol, not sympify to make sure that variables like
    'pi' are kept as Symbols, not the SymPy built-in number subclasses.

    Examples
    ========

    >>> from sympy.physics.quantum.qexpr import _qsympify_sequence
    >>> _qsympify_sequence((1,2,[3,4,[1,]]))
    (1, 2, (3, 4, (1,)))

    """

    return tuple(__qsympify_sequence_helper(seq))


def __qsympify_sequence_helper(seq):
    """
       Helper function for _qsympify_sequence
       This function does the actual work.
    """
    #base case. If not a list, do Sympification
    if not is_sequence(seq):
        if isinstance(seq, Matrix):
            return seq
        elif isinstance(seq, str):
            return Symbol(seq)
        else:
            return sympify(seq)

    # base condition, when seq is QExpr and also
    # is iterable.
    if isinstance(seq, QExpr):
        return seq

    #if list, recurse on each item in the list
    result = [__qsympify_sequence_helper(item) for item in seq]

    return Tuple(*result)


#-----------------------------------------------------------------------------
# Basic Quantum Expression from which all objects descend
#-----------------------------------------------------------------------------

class QExpr(Expr):
    """A base class for all quantum object like operators and states."""

    # In sympy, slots are for instance attributes that are computed
    # dynamically by the __new__ method. They are not part of args, but they
    # derive from args.

    # The Hilbert space a quantum Object belongs to.
    __slots__ = ('hilbert_space', )

    is_commutative = False

    # The separator used in printing the label.
    _label_separator = ''

    def __new__(cls, *args, **kwargs):
        """Construct a new quantum object.

        Parameters
        ==========

        args : tuple
            The list of numbers or parameters that uniquely specify the
            quantum object. For a state, this will be its symbol or its
            set of quantum numbers.

        Examples
        ========

        >>> from sympy.physics.quantum.qexpr import QExpr
        >>> q = QExpr(0)
        >>> q
        0
        >>> q.label
        (0,)
        >>> q.hilbert_space
        H
        >>> q.args
        (0,)
        >>> q.is_commutative
        False
        """

        # First compute args and call Expr.__new__ to create the instance
        args = cls._eval_args(args, **kwargs)
        if len(args) == 0:
            args = cls._eval_args(tuple(cls.default_args()), **kwargs)
        inst = Expr.__new__(cls, *args)
        # Now set the slots on the instance
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _new_rawargs(cls, hilbert_space, *args, **old_assumptions):
        """Create new instance of this class with hilbert_space and args.

        This is used to bypass the more complex logic in the ``__new__``
        method in cases where you already have the exact ``hilbert_space``
        and ``args``. This should be used when you are positive these
        arguments are valid, in their final, proper form and want to optimize
        the creation of the object.
        """

        obj = Expr.__new__(cls, *args, **old_assumptions)
        obj.hilbert_space = hilbert_space
        return obj

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def label(self):
        """The label is the unique set of identifiers for the object.

        Usually, this will include all of the information about the state
        *except* the time (in the case of time-dependent objects).

        This must be a tuple, rather than a Tuple.
        """
        if len(self.args) == 0:  # If there is no label specified, return the default
            return self._eval_args(list(self.default_args()))
        else:
            return self.args

    @property
    def is_symbolic(self):
        return True

    @classmethod
    def default_args(self):
        """If no arguments are specified, then this will return a default set
        of arguments to be run through the constructor.

        NOTE: Any classes that override this MUST return a tuple of arguments.
        Should be overridden by subclasses to specify the default arguments for kets and operators
        """
        raise NotImplementedError("No default arguments for this class!")

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------

    def _eval_adjoint(self):
        obj = Expr._eval_adjoint(self)
        if obj is None:
            obj = Expr.__new__(Dagger, self)
        if isinstance(obj, QExpr):
            obj.hilbert_space = self.hilbert_space
        return obj

    @classmethod
    def _eval_args(cls, args):
        """Process the args passed to the __new__ method.

        This simply runs args through _qsympify_sequence.
        """
        return _qsympify_sequence(args)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """Compute the Hilbert space instance from the args.
        """
        from sympy.physics.quantum.hilbert import HilbertSpace
        return HilbertSpace()

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    # Utilities for printing: these operate on raw SymPy objects

    def _print_sequence(self, seq, sep, printer, *args):
        result = []
        for item in seq:
            result.append(printer._print(item, *args))
        return sep.join(result)

    def _print_sequence_pretty(self, seq, sep, printer, *args):
        pform = printer._print(seq[0], *args)
        for item in seq[1:]:
            pform = prettyForm(*pform.right(sep))
            pform = prettyForm(*pform.right(printer._print(item, *args)))
        return pform

    # Utilities for printing: these operate prettyForm objects

    def _print_subscript_pretty(self, a, b):
        top = prettyForm(*b.left(' '*a.width()))
        bot = prettyForm(*a.right(' '*b.width()))
        return prettyForm(binding=prettyForm.POW, *bot.below(top))

    def _print_superscript_pretty(self, a, b):
        return a**b

    def _print_parens_pretty(self, pform, left='(', right=')'):
        return prettyForm(*pform.parens(left=left, right=right))

    # Printing of labels (i.e. args)

    def _print_label(self, printer, *args):
        """Prints the label of the QExpr

        This method prints self.label, using self._label_separator to separate
        the elements. This method should not be overridden, instead, override
        _print_contents to change printing behavior.
        """
        return self._print_sequence(
            self.label, self._label_separator, printer, *args
        )

    def _print_label_repr(self, printer, *args):
        return self._print_sequence(
            self.label, ',', printer, *args
        )

    def _print_label_pretty(self, printer, *args):
        return self._print_sequence_pretty(
            self.label, self._label_separator, printer, *args
        )

    def _print_label_latex(self, printer, *args):
        return self._print_sequence(
            self.label, self._label_separator, printer, *args
        )

    # Printing of contents (default to label)

    def _print_contents(self, printer, *args):
        """Printer for contents of QExpr

        Handles the printing of any unique identifying contents of a QExpr to
        print as its contents, such as any variables or quantum numbers. The
        default is to print the label, which is almost always the args. This
        should not include printing of any brackets or parentheses.
        """
        return self._print_label(printer, *args)

    def _print_contents_pretty(self, printer, *args):
        return self._print_label_pretty(printer, *args)

    def _print_contents_latex(self, printer, *args):
        return self._print_label_latex(printer, *args)

    # Main printing methods

    def _sympystr(self, printer, *args):
        """Default printing behavior of QExpr objects

        Handles the default printing of a QExpr. To add other things to the
        printing of the object, such as an operator name to operators or
        brackets to states, the class should override the _print/_pretty/_latex
        functions directly and make calls to _print_contents where appropriate.
        This allows things like InnerProduct to easily control its printing the
        printing of contents.
        """
        return self._print_contents(printer, *args)

    def _sympyrepr(self, printer, *args):
        classname = self.__class__.__name__
        label = self._print_label_repr(printer, *args)
        return '%s(%s)' % (classname, label)

    def _pretty(self, printer, *args):
        pform = self._print_contents_pretty(printer, *args)
        return pform

    def _latex(self, printer, *args):
        return self._print_contents_latex(printer, *args)

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_default_basis(self, **options):
        raise NotImplementedError('This object does not have a default basis')

    def _represent(self, *, basis=None, **options):
        """Represent this object in a given basis.

        This method dispatches to the actual methods that perform the
        representation. Subclases of QExpr should define various methods to
        determine how the object will be represented in various bases. The
        format of these methods is::

            def _represent_BasisName(self, basis, **options):

        Thus to define how a quantum object is represented in the basis of
        the operator Position, you would define::

            def _represent_Position(self, basis, **options):

        Usually, basis object will be instances of Operator subclasses, but
        there is a chance we will relax this in the future to accommodate other
        types of basis sets that are not associated with an operator.

        If the ``format`` option is given it can be ("sympy", "numpy",
        "scipy.sparse"). This will ensure that any matrices that result from
        representing the object are returned in the appropriate matrix format.

        Parameters
        ==========

        basis : Operator
            The Operator whose basis functions will be used as the basis for
            representation.
        options : dict
            A dictionary of key/value pairs that give options and hints for
            the representation, such as the number of basis functions to
            be used.
        """
        if basis is None:
            result = self._represent_default_basis(**options)
        else:
            result = dispatch_method(self, '_represent', basis, **options)

        # If we get a matrix representation, convert it to the right format.
        format = options.get('format', 'sympy')
        result = self._format_represent(result, format)
        return result

    def _format_represent(self, result, format):
        if format == 'sympy' and not isinstance(result, Matrix):
            return to_sympy(result)
        elif format == 'numpy' and not isinstance(result, numpy_ndarray):
            return to_numpy(result)
        elif format == 'scipy.sparse' and \
                not isinstance(result, scipy_sparse_matrix):
            return to_scipy_sparse(result)

        return result


def split_commutative_parts(e):
    """Split into commutative and non-commutative parts."""
    c_part, nc_part = e.args_cnc()
    c_part = list(c_part)
    return c_part, nc_part


def split_qexpr_parts(e):
    """Split an expression into Expr and noncommutative QExpr parts."""
    expr_part = []
    qexpr_part = []
    for arg in e.args:
        if not isinstance(arg, QExpr):
            expr_part.append(arg)
        else:
            qexpr_part.append(arg)
    return expr_part, qexpr_part


def dispatch_method(self, basename, arg, **options):
    """Dispatch a method to the proper handlers."""
    method_name = '%s_%s' % (basename, arg.__class__.__name__)
    if hasattr(self, method_name):
        f = getattr(self, method_name)
        # This can raise and we will allow it to propagate.
        result = f(arg, **options)
        if result is not None:
            return result
    raise NotImplementedError(
        "%s.%s cannot handle: %r" %
        (self.__class__.__name__, basename, arg)
    )
<|endoftext|>"""An implementation of gates that act on qubits.

Gates are unitary operators that act on the space of qubits.

Medium Term Todo:

* Optimize Gate._apply_operators_Qubit to remove the creation of many
  intermediate Qubit objects.
* Add commutation relationships to all operators and use this in gate_sort.
* Fix gate_sort and gate_simp.
* Get multi-target UGates plotting properly.
* Get UGate to work with either sympy/numpy matrices and output either
  format. This should also use the matrix slots.
"""

from itertools import chain
import random

from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.power import Pow
from sympy.core.numbers import Number
from sympy.core.singleton import S as _S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.pretty.stringpict import prettyForm, stringPict

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import (UnitaryOperator, Operator,
                                            HermitianOperator)
from sympy.physics.quantum.matrixutils import matrix_tensor_product, matrix_eye
from sympy.physics.quantum.matrixcache import matrix_cache

from sympy.matrices.matrixbase import MatrixBase

from sympy.utilities.iterables import is_sequence

__all__ = [
    'Gate',
    'CGate',
    'UGate',
    'OneQubitGate',
    'TwoQubitGate',
    'IdentityGate',
    'HadamardGate',
    'XGate',
    'YGate',
    'ZGate',
    'TGate',
    'PhaseGate',
    'SwapGate',
    'CNotGate',
    # Aliased gate names
    'CNOT',
    'SWAP',
    'H',
    'X',
    'Y',
    'Z',
    'T',
    'S',
    'Phase',
    'normalized',
    'gate_sort',
    'gate_simp',
    'random_circuit',
    'CPHASE',
    'CGateS',
]

#-----------------------------------------------------------------------------
# Gate Super-Classes
#-----------------------------------------------------------------------------

_normalized = True


def _max(*args, **kwargs):
    if "key" not in kwargs:
        kwargs["key"] = default_sort_key
    return max(*args, **kwargs)


def _min(*args, **kwargs):
    if "key" not in kwargs:
        kwargs["key"] = default_sort_key
    return min(*args, **kwargs)


def normalized(normalize):
    r"""Set flag controlling normalization of Hadamard gates by `1/\sqrt{2}`.

    This is a global setting that can be used to simplify the look of various
    expressions, by leaving off the leading `1/\sqrt{2}` of the Hadamard gate.

    Parameters
    ----------
    normalize : bool
        Should the Hadamard gate include the `1/\sqrt{2}` normalization factor?
        When True, the Hadamard gate will have the `1/\sqrt{2}`. When False, the
        Hadamard gate will not have this factor.
    """
    global _normalized
    _normalized = normalize


def _validate_targets_controls(tandc):
    tandc = list(tandc)
    # Check for integers
    for bit in tandc:
        if not bit.is_Integer and not bit.is_Symbol:
            raise TypeError('Integer expected, got: %r' % tandc[bit])
    # Detect duplicates
    if len(set(tandc)) != len(tandc):
        raise QuantumError(
            'Target/control qubits in a gate cannot be duplicated'
        )


class Gate(UnitaryOperator):
    """Non-controlled unitary gate operator that acts on qubits.

    This is a general abstract gate that needs to be subclassed to do anything
    useful.

    Parameters
    ----------
    label : tuple, int
        A list of the target qubits (as ints) that the gate will apply to.

    Examples
    ========


    """

    _label_separator = ','

    gate_name = 'G'
    gate_name_latex = 'G'

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        args = Tuple(*UnitaryOperator._eval_args(args))
        _validate_targets_controls(args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
        return len(self.targets)

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(self.targets) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return self.label

    @property
    def gate_name_plot(self):
        return r'$%s$' % self.gate_name_latex

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        """The matrix representation of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        raise NotImplementedError(
            'get_target_matrix is not implemented in Gate.')

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_IntQubit(self, qubits, **options):
        """Redirect an apply from IntQubit to Qubit"""
        return self._apply_operator_Qubit(qubits, **options)

    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this gate to a Qubit."""

        # Check number of qubits this gate acts on.
        if qubits.nqubits < self.min_qubits:
            raise QuantumError(
                'Gate needs a minimum of %r qubits to act on, got: %r' %
                (self.min_qubits, qubits.nqubits)
            )

        # If the controls are not met, just return
        if isinstance(self, CGate):
            if not self.eval_controls(qubits):
                return qubits

        targets = self.targets
        target_matrix = self.get_target_matrix(format='sympy')

        # Find which column of the target matrix this applies to.
        column_index = 0
        n = 1
        for target in targets:
            column_index += n*qubits[target]
            n = n << 1
        column = target_matrix[:, int(column_index)]

        # Now apply each column element to the qubit.
        result = 0
        for index in range(column.rows):
            # TODO: This can be optimized to reduce the number of Qubit
            # creations. We should simply manipulate the raw list of qubit
            # values and then build the new Qubit object once.
            # Make a copy of the incoming qubits.
            new_qubit = qubits.__class__(*qubits.args)
            # Flip the bits that need to be flipped.
            for bit, target in enumerate(targets):
                if new_qubit[target] != (index >> bit) & 1:
                    new_qubit = new_qubit.flip(target)
            # The value in that row and column times the flipped-bit qubit
            # is the result for that part.
            result += column[index]*new_qubit
        return result

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        format = options.get('format', 'sympy')
        nqubits = options.get('nqubits', 0)
        if nqubits == 0:
            raise QuantumError(
                'The number of qubits must be given as nqubits.')

        # Make sure we have enough qubits for the gate.
        if nqubits < self.min_qubits:
            raise QuantumError(
                'The number of qubits %r is too small for the gate.' % nqubits
            )

        target_matrix = self.get_target_matrix(format)
        targets = self.targets
        if isinstance(self, CGate):
            controls = self.controls
        else:
            controls = []
        m = represent_zbasis(
            controls, targets, target_matrix, nqubits, format
        )
        return m

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------

    def _sympystr(self, printer, *args):
        label = self._print_label(printer, *args)
        return '%s(%s)' % (self.gate_name, label)

    def _pretty(self, printer, *args):
        a = stringPict(self.gate_name)
        b = self._print_label_pretty(printer, *args)
        return self._print_subscript_pretty(a, b)

    def _latex(self, printer, *args):
        label = self._print_label(printer, *args)
        return '%s_{%s}' % (self.gate_name_latex, label)

    def plot_gate(self, axes, gate_idx, gate_grid, wire_grid):
        raise NotImplementedError('plot_gate is not implemented.')


class CGate(Gate):
    """A general unitary gate with control qubits.

    A general control gate applies a target gate to a set of targets if all
    of the control qubits have a particular values (set by
    ``CGate.control_value``).

    Parameters
    ----------
    label : tuple
        The label in this case has the form (controls, gate), where controls
        is a tuple/list of control qubits (as ints) and gate is a ``Gate``
        instance that is the target operator.

    Examples
    ========

    """

    gate_name = 'C'
    gate_name_latex = 'C'

    # The values this class controls for.
    control_value = _S.One

    simplify_cgate = False

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # _eval_args has the right logic for the controls argument.
        controls = args[0]
        gate = args[1]
        if not is_sequence(controls):
            controls = (controls,)
        controls = UnitaryOperator._eval_args(controls)
        _validate_targets_controls(chain(controls, gate.targets))
        return (Tuple(*controls), gate)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**_max(_max(args[0]) + 1, args[1].min_qubits)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
        return len(self.targets) + len(self.controls)

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(_max(self.controls), _max(self.targets)) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return self.gate.targets

    @property
    def controls(self):
        """A tuple of control qubits."""
        return tuple(self.label[0])

    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
        return self.label[1]

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        return self.gate.get_target_matrix(format)

    def eval_controls(self, qubit):
        """Return True/False to indicate if the controls are satisfied."""
        return all(qubit[bit] == self.control_value for bit in self.controls)

    def decompose(self, **options):
        """Decompose the controlled gate into CNOT and single qubits gates."""
        if len(self.controls) == 1:
            c = self.controls[0]
            t = self.gate.targets[0]
            if isinstance(self.gate, YGate):
                g1 = PhaseGate(t)
                g2 = CNotGate(c, t)
                g3 = PhaseGate(t)
                g4 = ZGate(t)
                return g1*g2*g3*g4
            if isinstance(self.gate, ZGate):
                g1 = HadamardGate(t)
                g2 = CNotGate(c, t)
                g3 = HadamardGate(t)
                return g1*g2*g3
        else:
            return self

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------

    def _print_label(self, printer, *args):
        controls = self._print_sequence(self.controls, ',', printer, *args)
        gate = printer._print(self.gate, *args)
        return '(%s),%s' % (controls, gate)

    def _pretty(self, printer, *args):
        controls = self._print_sequence_pretty(
            self.controls, ',', printer, *args)
        gate = printer._print(self.gate)
        gate_name = stringPict(self.gate_name)
        first = self._print_subscript_pretty(gate_name, controls)
        gate = self._print_parens_pretty(gate)
        final = prettyForm(*first.right(gate))
        return final

    def _latex(self, printer, *args):
        controls = self._print_sequence(self.controls, ',', printer, *args)
        gate = printer._print(self.gate, *args)
        return r'%s_{%s}{\left(%s\right)}' % \
            (self.gate_name_latex, controls, gate)

    def plot_gate(self, circ_plot, gate_idx):
        """
        Plot the controlled gate. If *simplify_cgate* is true, simplify
        C-X and C-Z gates into their more familiar forms.
        """
        min_wire = int(_min(chain(self.controls, self.targets)))
        max_wire = int(_max(chain(self.controls, self.targets)))
        circ_plot.control_line(gate_idx, min_wire, max_wire)
        for c in self.controls:
            circ_plot.control_point(gate_idx, int(c))
        if self.simplify_cgate:
            if self.gate.gate_name == 'X':
                self.gate.plot_gate_plus(circ_plot, gate_idx)
            elif self.gate.gate_name == 'Z':
                circ_plot.control_point(gate_idx, self.targets[0])
            else:
                self.gate.plot_gate(circ_plot, gate_idx)
        else:
            self.gate.plot_gate(circ_plot, gate_idx)

    #-------------------------------------------------------------------------
    # Miscellaneous
    #-------------------------------------------------------------------------

    def _eval_dagger(self):
        if isinstance(self.gate, HermitianOperator):
            return self
        else:
            return Gate._eval_dagger(self)

    def _eval_inverse(self):
        if isinstance(self.gate, HermitianOperator):
            return self
        else:
            return Gate._eval_inverse(self)

    def _eval_power(self, exp):
        if isinstance(self.gate, HermitianOperator):
            if exp == -1:
                return Gate._eval_power(self, exp)
            elif abs(exp) % 2 == 0:
                return self*(Gate._eval_inverse(self))
            else:
                return self
        else:
            return Gate._eval_power(self, exp)

class CGateS(CGate):
    """Version of CGate that allows gate simplifications.
    I.e. cnot looks like an oplus, cphase has dots, etc.
    """
    simplify_cgate=True


class UGate(Gate):
    """General gate specified by a set of targets and a target matrix.

    Parameters
    ----------
    label : tuple
        A tuple of the form (targets, U), where targets is a tuple of the
        target qubits and U is a unitary matrix with dimension of
        len(targets).
    """
    gate_name = 'U'
    gate_name_latex = 'U'

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        targets = args[0]
        if not is_sequence(targets):
            targets = (targets,)
        targets = Gate._eval_args(targets)
        _validate_targets_controls(targets)
        mat = args[1]
        if not isinstance(mat, MatrixBase):
            raise TypeError('Matrix expected, got: %r' % mat)
        #make sure this matrix is of a Basic type
        mat = _sympify(mat)
        dim = 2**len(targets)
        if not all(dim == shape for shape in mat.shape):
            raise IndexError(
                'Number of targets must match the matrix size: %r %r' %
                (targets, mat)
            )
        return (targets, mat)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args[0]) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def targets(self):
        """A tuple of target qubits."""
        return tuple(self.label[0])

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        """The matrix rep. of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        return self.label[1]

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------
    def _pretty(self, printer, *args):
        targets = self._print_sequence_pretty(
            self.targets, ',', printer, *args)
        gate_name = stringPict(self.gate_name)
        return self._print_subscript_pretty(gate_name, targets)

    def _latex(self, printer, *args):
        targets = self._print_sequence(self.targets, ',', printer, *args)
        return r'%s_{%s}' % (self.gate_name_latex, targets)

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(
            self.gate_name_plot,
            gate_idx, int(self.targets[0])
        )


class OneQubitGate(Gate):
    """A single qubit unitary gate base class."""

    nqubits = _S.One

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(
            self.gate_name_plot,
            gate_idx, int(self.targets[0])
        )

    def _eval_commutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return _S.Zero
        return Operator._eval_commutator(self, other, **hints)

    def _eval_anticommutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return Integer(2)*self*other
        return Operator._eval_anticommutator(self, other, **hints)


class TwoQubitGate(Gate):
    """A two qubit unitary gate base class."""

    nqubits = Integer(2)

#-----------------------------------------------------------------------------
# Single Qubit Gates
#-----------------------------------------------------------------------------


class IdentityGate(OneQubitGate):
    """The single qubit identity gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian = True
    gate_name = '1'
    gate_name_latex = '1'

    # Short cut version of gate._apply_operator_Qubit
    def _apply_operator_Qubit(self, qubits, **options):
        # Check number of qubits this gate acts on (see gate._apply_operator_Qubit)
        if qubits.nqubits < self.min_qubits:
            raise QuantumError(
                'Gate needs a minimum of %r qubits to act on, got: %r' %
                (self.min_qubits, qubits.nqubits)
            )
        return qubits # no computation required for IdentityGate

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('eye2', format)

    def _eval_commutator(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator(self, other, **hints):
        return Integer(2)*other


class HadamardGate(HermitianOperator, OneQubitGate):
    """The single qubit Hadamard gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> from sympy.physics.quantum.gate import HadamardGate
    >>> from sympy.physics.quantum.qapply import qapply
    >>> qapply(HadamardGate(0)*Qubit('1'))
    sqrt(2)*|0>/2 - sqrt(2)*|1>/2
    >>> # Hadamard on bell state, applied on 2 qubits.
    >>> psi = 1/sqrt(2)*(Qubit('00')+Qubit('11'))
    >>> qapply(HadamardGate(0)*HadamardGate(1)*psi)
    sqrt(2)*|00>/2 + sqrt(2)*|11>/2

    """
    gate_name = 'H'
    gate_name_latex = 'H'

    def get_target_matrix(self, format='sympy'):
        if _normalized:
            return matrix_cache.get_matrix('H', format)
        else:
            return matrix_cache.get_matrix('Hsqrt2', format)

    def _eval_commutator_XGate(self, other, **hints):
        return I*sqrt(2)*YGate(self.targets[0])

    def _eval_commutator_YGate(self, other, **hints):
        return I*sqrt(2)*(ZGate(self.targets[0]) - XGate(self.targets[0]))

    def _eval_commutator_ZGate(self, other, **hints):
        return -I*sqrt(2)*YGate(self.targets[0])

    def _eval_anticommutator_XGate(self, other, **hints):
        return sqrt(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator_ZGate(self, other, **hints):
        return sqrt(2)*IdentityGate(self.targets[0])


class XGate(HermitianOperator, OneQubitGate):
    """The single qubit X, or NOT, gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'X'
    gate_name_latex = 'X'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('X', format)

    def plot_gate(self, circ_plot, gate_idx):
        OneQubitGate.plot_gate(self,circ_plot,gate_idx)

    def plot_gate_plus(self, circ_plot, gate_idx):
        circ_plot.not_point(
            gate_idx, int(self.label[0])
        )

    def _eval_commutator_YGate(self, other, **hints):
        return Integer(2)*I*ZGate(self.targets[0])

    def _eval_anticommutator_XGate(self, other, **hints):
        return Integer(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator_ZGate(self, other, **hints):
        return _S.Zero


class YGate(HermitianOperator, OneQubitGate):
    """The single qubit Y gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'Y'
    gate_name_latex = 'Y'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('Y', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return Integer(2)*I*XGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return Integer(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_ZGate(self, other, **hints):
        return _S.Zero


class ZGate(HermitianOperator, OneQubitGate):
    """The single qubit Z gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'Z'
    gate_name_latex = 'Z'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('Z', format)

    def _eval_commutator_XGate(self, other, **hints):
        return Integer(2)*I*YGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero


class PhaseGate(OneQubitGate):
    """The single qubit phase, or S, gate.

    This gate rotates the phase of the state by pi/2 if the state is ``|1>`` and
    does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian =  False
    gate_name = 'S'
    gate_name_latex = 'S'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('S', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return _S.Zero

    def _eval_commutator_TGate(self, other, **hints):
        return _S.Zero


class TGate(OneQubitGate):
    """The single qubit pi/8 gate.

    This gate rotates the phase of the state by pi/4 if the state is ``|1>`` and
    does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian = False
    gate_name = 'T'
    gate_name_latex = 'T'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('T', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return _S.Zero

    def _eval_commutator_PhaseGate(self, other, **hints):
        return _S.Zero


# Aliases for gate names.
H = HadamardGate
X = XGate
Y = YGate
Z = ZGate
T = TGate
Phase = S = PhaseGate


#-----------------------------------------------------------------------------
# 2 Qubit Gates
#-----------------------------------------------------------------------------


class CNotGate(HermitianOperator, CGate, TwoQubitGate):
    """Two qubit controlled-NOT.

    This gate performs the NOT or X gate on the target qubit if the control
    qubits all have the value 1.

    Parameters
    ----------
    label : tuple
        A tuple of the form (control, target).

    Examples
    ========

    >>> from sympy.physics.quantum.gate import CNOT
    >>> from sympy.physics.quantum.qapply import qapply
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> c = CNOT(1,0)
    >>> qapply(c*Qubit('10')) # note that qubits are indexed from right to left
    |11>

    """
    gate_name = 'CNOT'
    gate_name_latex = r'\text{CNOT}'
    simplify_cgate = True

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        args = Gate._eval_args(args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(self.label) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return (self.label[1],)

    @property
    def controls(self):
        """A tuple of control qubits."""
        return (self.label[0],)

    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
        return XGate(self.label[1])

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    # The default printing of Gate works better than those of CGate, so we
    # go around the overridden methods in CGate.

    def _print_label(self, printer, *args):
        return Gate._print_label(self, printer, *args)

    def _pretty(self, printer, *args):
        return Gate._pretty(self, printer, *args)

    def _latex(self, printer, *args):
        return Gate._latex(self, printer, *args)

    #-------------------------------------------------------------------------
    # Commutator/AntiCommutator
    #-------------------------------------------------------------------------

    def _eval_commutator_ZGate(self, other, **hints):
        """[CNOT(i, j), Z(i)] == 0."""
        if self.controls[0] == other.targets[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)

    def _eval_commutator_TGate(self, other, **hints):
        """[CNOT(i, j), T(i)] == 0."""
        return self._eval_commutator_ZGate(other, **hints)

    def _eval_commutator_PhaseGate(self, other, **hints):
        """[CNOT(i, j), S(i)] == 0."""
        return self._eval_commutator_ZGate(other, **hints)

    def _eval_commutator_XGate(self, other, **hints):
        """[CNOT(i, j), X(j)] == 0."""
        if self.targets[0] == other.targets[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)

    def _eval_commutator_CNotGate(self, other, **hints):
        """[CNOT(i, j), CNOT(i,k)] == 0."""
        if self.controls[0] == other.controls[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)


class SwapGate(TwoQubitGate):
    """Two qubit SWAP gate.

    This gate swap the values of the two qubits.

    Parameters
    ----------
    label : tuple
        A tuple of the form (target1, target2).

    Examples
    ========

    """
    is_hermitian = True
    gate_name = 'SWAP'
    gate_name_latex = r'\text{SWAP}'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('SWAP', format)

    def decompose(self, **options):
        """Decompose the SWAP gate into CNOT gates."""
        i, j = self.targets[0], self.targets[1]
        g1 = CNotGate(i, j)
        g2 = CNotGate(j, i)
        return g1*g2*g1

    def plot_gate(self, circ_plot, gate_idx):
        min_wire = int(_min(self.targets))
        max_wire = int(_max(self.targets))
        circ_plot.control_line(gate_idx, min_wire, max_wire)
        circ_plot.swap_point(gate_idx, min_wire)
        circ_plot.swap_point(gate_idx, max_wire)

    def _represent_ZGate(self, basis, **options):
        """Represent the SWAP gate in the computational basis.

        The following representation is used to compute this:

        SWAP = |1><1|x|1><1| + |0><0|x|0><0| + |1><0|x|0><1| + |0><1|x|1><0|
        """
        format = options.get('format', 'sympy')
        targets = [int(t) for t in self.targets]
        min_target = _min(targets)
        max_target = _max(targets)
        nqubits = options.get('nqubits', self.min_qubits)

        op01 = matrix_cache.get_matrix('op01', format)
        op10 = matrix_cache.get_matrix('op10', format)
        op11 = matrix_cache.get_matrix('op11', format)
        op00 = matrix_cache.get_matrix('op00', format)
        eye2 = matrix_cache.get_matrix('eye2', format)

        result = None
        for i, j in ((op01, op10), (op10, op01), (op00, op00), (op11, op11)):
            product = nqubits*[eye2]
            product[nqubits - min_target - 1] = i
            product[nqubits - max_target - 1] = j
            new_result = matrix_tensor_product(*product)
            if result is None:
                result = new_result
            else:
                result = result + new_result

        return result


# Aliases for gate names.
CNOT = CNotGate
SWAP = SwapGate
def CPHASE(a,b): return CGateS((a,),Z(b))


#-----------------------------------------------------------------------------
# Represent
#-----------------------------------------------------------------------------


def represent_zbasis(controls, targets, target_matrix, nqubits, format='sympy'):
    """Represent a gate with controls, targets and target_matrix.

    This function does the low-level work of representing gates as matrices
    in the standard computational basis (ZGate). Currently, we support two
    main cases:

    1. One target qubit and no control qubits.
    2. One target qubits and multiple control qubits.

    For the base of multiple controls, we use the following expression [1]:

    1_{2**n} + (|1><1|)^{(n-1)} x (target-matrix - 1_{2})

    Parameters
    ----------
    controls : list, tuple
        A sequence of control qubits.
    targets : list, tuple
        A sequence of target qubits.
    target_matrix : sympy.Matrix, numpy.matrix, scipy.sparse
        The matrix form of the transformation to be performed on the target
        qubits.  The format of this matrix must match that passed into
        the `format` argument.
    nqubits : int
        The total number of qubits used for the representation.
    format : str
        The format of the final matrix ('sympy', 'numpy', 'scipy.sparse').

    Examples
    ========

    References
    ----------
    [1] http://www.johnlapeyre.com/qinf/qinf_html/node6.html.
    """
    controls = [int(x) for x in controls]
    targets = [int(x) for x in targets]
    nqubits = int(nqubits)

    # This checks for the format as well.
    op11 = matrix_cache.get_matrix('op11', format)
    eye2 = matrix_cache.get_matrix('eye2', format)

    # Plain single qubit case
    if len(controls) == 0 and len(targets) == 1:
        product = []
        bit = targets[0]
        # Fill product with [I1,Gate,I2] such that the unitaries,
        # I, cause the gate to be applied to the correct Qubit
        if bit != nqubits - 1:
            product.append(matrix_eye(2**(nqubits - bit - 1), format=format))
        product.append(target_matrix)
        if bit != 0:
            product.append(matrix_eye(2**bit, format=format))
        return matrix_tensor_product(*product)

    # Single target, multiple controls.
    elif len(targets) == 1 and len(controls) >= 1:
        target = targets[0]

        # Build the non-trivial part.
        product2 = []
        for i in range(nqubits):
            product2.append(matrix_eye(2, format=format))
        for control in controls:
            product2[nqubits - 1 - control] = op11
        product2[nqubits - 1 - target] = target_matrix - eye2

        return matrix_eye(2**nqubits, format=format) + \
            matrix_tensor_product(*product2)

    # Multi-target, multi-control is not yet implemented.
    else:
        raise NotImplementedError(
            'The representation of multi-target, multi-control gates '
            'is not implemented.'
        )


#-----------------------------------------------------------------------------
# Gate manipulation functions.
#-----------------------------------------------------------------------------


def gate_simp(circuit):
    """Simplifies gates symbolically

    It first sorts gates using gate_sort. It then applies basic
    simplification rules to the circuit, e.g., XGate**2 = Identity
    """

    # Bubble sort out gates that commute.
    circuit = gate_sort(circuit)

    # Do simplifications by subing a simplification into the first element
    # which can be simplified. We recursively call gate_simp with new circuit
    # as input more simplifications exist.
    if isinstance(circuit, Add):
        return sum(gate_simp(t) for t in circuit.args)
    elif isinstance(circuit, Mul):
        circuit_args = circuit.args
    elif isinstance(circuit, Pow):
        b, e = circuit.as_base_exp()
        circuit_args = (gate_simp(b)**e,)
    else:
        return circuit

    # Iterate through each element in circuit, simplify if possible.
    for i in range(len(circuit_args)):
        # H,X,Y or Z squared is 1.
        # T**2 = S, S**2 = Z
        if isinstance(circuit_args[i], Pow):
            if isinstance(circuit_args[i].base,
                (HadamardGate, XGate, YGate, ZGate)) \
                    and isinstance(circuit_args[i].exp, Number):
                # Build a new circuit taking replacing the
                # H,X,Y,Z squared with one.
                newargs = (circuit_args[:i] +
                          (circuit_args[i].base**(circuit_args[i].exp % 2),) +
                           circuit_args[i + 1:])
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
            elif isinstance(circuit_args[i].base, PhaseGate):
                # Build a new circuit taking old circuit but splicing
                # in simplification.
                newargs = circuit_args[:i]
                # Replace PhaseGate**2 with ZGate.
                newargs = newargs + (ZGate(circuit_args[i].base.args[0])**
                (Integer(circuit_args[i].exp/2)), circuit_args[i].base**
                (circuit_args[i].exp % 2))
                # Append the last elements.
                newargs = newargs + circuit_args[i + 1:]
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
            elif isinstance(circuit_args[i].base, TGate):
                # Build a new circuit taking all the old elements.
                newargs = circuit_args[:i]

                # Put an Phasegate in place of any TGate**2.
                newargs = newargs + (PhaseGate(circuit_args[i].base.args[0])**
                Integer(circuit_args[i].exp/2), circuit_args[i].base**
                    (circuit_args[i].exp % 2))

                # Append the last elements.
                newargs = newargs + circuit_args[i + 1:]
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
    return circuit


def gate_sort(circuit):
    """Sorts the gates while keeping track of commutation relations

    This function uses a bubble sort to rearrange the order of gate
    application. Keeps track of Quantum computations special commutation
    relations (e.g. things that apply to the same Qubit do not commute with
    each other)

    circuit is the Mul of gates that are to be sorted.
    """
    # Make sure we have an Add or Mul.
    if isinstance(circuit, Add):
        return sum(gate_sort(t) for t in circuit.args)
    if isinstance(circuit, Pow):
        return gate_sort(circuit.base)**circuit.exp
    elif isinstance(circuit, Gate):
        return circuit
    if not isinstance(circuit, Mul):
        return circuit

    changes = True
    while changes:
        changes = False
        circ_array = circuit.args
        for i in range(len(circ_array) - 1):
            # Go through each element and switch ones that are in wrong order
            if isinstance(circ_array[i], (Gate, Pow)) and \
                    isinstance(circ_array[i + 1], (Gate, Pow)):
                # If we have a Pow object, look at only the base
                first_base, first_exp = circ_array[i].as_base_exp()
                second_base, second_exp = circ_array[i + 1].as_base_exp()

                # Use SymPy's hash based sorting. This is not mathematical
                # sorting, but is rather based on comparing hashes of objects.
                # See Basic.compare for details.
                if first_base.compare(second_base) > 0:
                    if Commutator(first_base, second_base).doit() == 0:
                        new_args = (circuit.args[:i] + (circuit.args[i + 1],) +
                                   (circuit.args[i],) + circuit.args[i + 2:])
                        circuit = Mul(*new_args)
                        changes = True
                        break
                    if AntiCommutator(first_base, second_base).doit() == 0:
                        new_args = (circuit.args[:i] + (circuit.args[i + 1],) +
                                   (circuit.args[i],) + circuit.args[i + 2:])
                        sign = _S.NegativeOne**(first_exp*second_exp)
                        circuit = sign*Mul(*new_args)
                        changes = True
                        break
    return circuit


#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------


def random_circuit(ngates, nqubits, gate_space=(X, Y, Z, S, T, H, CNOT, SWAP)):
    """Return a random circuit of ngates and nqubits.

    This uses an equally weighted sample of (X, Y, Z, S, T, H, CNOT, SWAP)
    gates.

    Parameters
    ----------
    ngates : int
        The number of gates in the circuit.
    nqubits : int
        The number of qubits in the circuit.
    gate_space : tuple
        A tuple of the gate classes that will be used in the circuit.
        Repeating gate classes multiple times in this tuple will increase
        the frequency they appear in the random circuit.
    """
    qubit_space = range(nqubits)
    result = []
    for i in range(ngates):
        g = random.choice(gate_space)
        if g == CNotGate or g == SwapGate:
            qubits = random.sample(qubit_space, 2)
            g = g(*qubits)
        else:
            qubit = random.choice(qubit_space)
            g = g(qubit)
        result.append(g)
    return Mul(*result)


def zx_basis_transform(self, format='sympy'):
    """Transformation matrix from Z to X basis."""
    return matrix_cache.get_matrix('ZX', format)


def zy_basis_transform(self, format='sympy'):
    """Transformation matrix from Z to Y basis."""
    return matrix_cache.get_matrix('ZY', format)
<|endoftext|>"""Operators and states for 1D cartesian position and momentum.

TODO:

* Add 3D classes to mappings in operatorset.py

"""

from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval

from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import L2
from sympy.physics.quantum.operator import DifferentialOperator, HermitianOperator
from sympy.physics.quantum.state import Ket, Bra, State

__all__ = [
    'XOp',
    'YOp',
    'ZOp',
    'PxOp',
    'X',
    'Y',
    'Z',
    'Px',
    'XKet',
    'XBra',
    'PxKet',
    'PxBra',
    'PositionState3D',
    'PositionKet3D',
    'PositionBra3D'
]

#-------------------------------------------------------------------------
# Position operators
#-------------------------------------------------------------------------


class XOp(HermitianOperator):
    """1D cartesian position operator."""

    @classmethod
    def default_args(self):
        return ("X",)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _eval_commutator_PxOp(self, other):
        return I*hbar

    def _apply_operator_XKet(self, ket, **options):
        return ket.position*ket

    def _apply_operator_PositionKet3D(self, ket, **options):
        return ket.position_x*ket

    def _represent_PxKet(self, basis, *, index=1, **options):
        states = basis._enumerate_state(2, start_index=index)
        coord1 = states[0].momentum
        coord2 = states[1].momentum
        d = DifferentialOperator(coord1)
        delta = DiracDelta(coord1 - coord2)

        return I*hbar*(d*delta)


class YOp(HermitianOperator):
    """ Y cartesian coordinate operator (for 2D or 3D systems) """

    @classmethod
    def default_args(self):
        return ("Y",)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PositionKet3D(self, ket, **options):
        return ket.position_y*ket


class ZOp(HermitianOperator):
    """ Z cartesian coordinate operator (for 3D systems) """

    @classmethod
    def default_args(self):
        return ("Z",)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PositionKet3D(self, ket, **options):
        return ket.position_z*ket

#-------------------------------------------------------------------------
# Momentum operators
#-------------------------------------------------------------------------


class PxOp(HermitianOperator):
    """1D cartesian momentum operator."""

    @classmethod
    def default_args(self):
        return ("Px",)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PxKet(self, ket, **options):
        return ket.momentum*ket

    def _represent_XKet(self, basis, *, index=1, **options):
        states = basis._enumerate_state(2, start_index=index)
        coord1 = states[0].position
        coord2 = states[1].position
        d = DifferentialOperator(coord1)
        delta = DiracDelta(coord1 - coord2)

        return -I*hbar*(d*delta)

X = XOp('X')
Y = YOp('Y')
Z = ZOp('Z')
Px = PxOp('Px')

#-------------------------------------------------------------------------
# Position eigenstates
#-------------------------------------------------------------------------


class XKet(Ket):
    """1D cartesian position eigenket."""

    @classmethod
    def _operators_to_state(self, op, **options):
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        return ("x",)

    @classmethod
    def dual_class(self):
        return XBra

    @property
    def position(self):
        """The position of the state."""
        return self.label[0]

    def _enumerate_state(self, num_states, **options):
        return _enumerate_continuous_1D(self, num_states, **options)

    def _eval_innerproduct_XBra(self, bra, **hints):
        return DiracDelta(self.position - bra.position)

    def _eval_innerproduct_PxBra(self, bra, **hints):
        return exp(-I*self.position*bra.momentum/hbar)/sqrt(2*pi*hbar)


class XBra(Bra):
    """1D cartesian position eigenbra."""

    @classmethod
    def default_args(self):
        return ("x",)

    @classmethod
    def dual_class(self):
        return XKet

    @property
    def position(self):
        """The position of the state."""
        return self.label[0]


class PositionState3D(State):
    """ Base class for 3D cartesian position eigenstates """

    @classmethod
    def _operators_to_state(self, op, **options):
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        return ("x", "y", "z")

    @property
    def position_x(self):
        """ The x coordinate of the state """
        return self.label[0]

    @property
    def position_y(self):
        """ The y coordinate of the state """
        return self.label[1]

    @property
    def position_z(self):
        """ The z coordinate of the state """
        return self.label[2]


class PositionKet3D(Ket, PositionState3D):
    """ 3D cartesian position eigenket """

    def _eval_innerproduct_PositionBra3D(self, bra, **options):
        x_diff = self.position_x - bra.position_x
        y_diff = self.position_y - bra.position_y
        z_diff = self.position_z - bra.position_z

        return DiracDelta(x_diff)*DiracDelta(y_diff)*DiracDelta(z_diff)

    @classmethod
    def dual_class(self):
        return PositionBra3D


# XXX: The type:ignore here is because mypy gives Definition of
# "_state_to_operators" in base class "PositionState3D" is incompatible with
# definition in base class "BraBase"
class PositionBra3D(Bra, PositionState3D):  # type: ignore
    """ 3D cartesian position eigenbra """

    @classmethod
    def dual_class(self):
        return PositionKet3D

#-------------------------------------------------------------------------
# Momentum eigenstates
#-------------------------------------------------------------------------


class PxKet(Ket):
    """1D cartesian momentum eigenket."""

    @classmethod
    def _operators_to_state(self, op, **options):
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        return ("px",)

    @classmethod
    def dual_class(self):
        return PxBra

    @property
    def momentum(self):
        """The momentum of the state."""
        return self.label[0]

    def _enumerate_state(self, *args, **options):
        return _enumerate_continuous_1D(self, *args, **options)

    def _eval_innerproduct_XBra(self, bra, **hints):
        return exp(I*self.momentum*bra.position/hbar)/sqrt(2*pi*hbar)

    def _eval_innerproduct_PxBra(self, bra, **hints):
        return DiracDelta(self.momentum - bra.momentum)


class PxBra(Bra):
    """1D cartesian momentum eigenbra."""

    @classmethod
    def default_args(self):
        return ("px",)

    @classmethod
    def dual_class(self):
        return PxKet

    @property
    def momentum(self):
        """The momentum of the state."""
        return self.label[0]

#-------------------------------------------------------------------------
# Global helper functions
#-------------------------------------------------------------------------


def _enumerate_continuous_1D(*args, **options):
    state = args[0]
    num_states = args[1]
    state_class = state.__class__
    index_list = options.pop('index_list', [])

    if len(index_list) == 0:
        start_index = options.pop('start_index', 1)
        index_list = list(range(start_index, start_index + num_states))

    enum_states = [0 for i in range(len(index_list))]

    for i, ind in enumerate(index_list):
        label = state.args[0]
        enum_states[i] = state_class(str(label) + "_" + str(ind), **options)

    return enum_states


def _lowercase_labels(ops):
    if not isinstance(ops, set):
        ops = [ops]

    return [str(arg.label[0]).lower() for arg in ops]


def _uppercase_labels(ops):
    if not isinstance(ops, set):
        ops = [ops]

    new_args = [str(arg.label[0])[0].upper() +
                str(arg.label[0])[1:] for arg in ops]

    return new_args
<|endoftext|>"""Hilbert spaces for quantum mechanics.

Authors:
* Brian Granger
* Matt Curry
"""

from functools import reduce

from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError


__all__ = [
    'HilbertSpaceError',
    'HilbertSpace',
    'TensorProductHilbertSpace',
    'TensorPowerHilbertSpace',
    'DirectSumHilbertSpace',
    'ComplexSpace',
    'L2',
    'FockSpace'
]

#-----------------------------------------------------------------------------
# Main objects
#-----------------------------------------------------------------------------


class HilbertSpaceError(QuantumError):
    pass

#-----------------------------------------------------------------------------
# Main objects
#-----------------------------------------------------------------------------


class HilbertSpace(Basic):
    """An abstract Hilbert space for quantum mechanics.

    In short, a Hilbert space is an abstract vector space that is complete
    with inner products defined [1]_.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import HilbertSpace
    >>> hs = HilbertSpace()
    >>> hs
    H

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space
    """

    def __new__(cls):
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        """Return the Hilbert dimension of the space."""
        raise NotImplementedError('This Hilbert space has no dimension.')

    def __add__(self, other):
        return DirectSumHilbertSpace(self, other)

    def __radd__(self, other):
        return DirectSumHilbertSpace(other, self)

    def __mul__(self, other):
        return TensorProductHilbertSpace(self, other)

    def __rmul__(self, other):
        return TensorProductHilbertSpace(other, self)

    def __pow__(self, other, mod=None):
        if mod is not None:
            raise ValueError('The third argument to __pow__ is not supported \
            for Hilbert spaces.')
        return TensorPowerHilbertSpace(self, other)

    def __contains__(self, other):
        """Is the operator or state in this Hilbert space.

        This is checked by comparing the classes of the Hilbert spaces, not
        the instances. This is to allow Hilbert Spaces with symbolic
        dimensions.
        """
        if other.hilbert_space.__class__ == self.__class__:
            return True
        else:
            return False

    def _sympystr(self, printer, *args):
        return 'H'

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER H}'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        return r'\mathcal{H}'


class ComplexSpace(HilbertSpace):
    """Finite dimensional Hilbert space of complex vectors.

    The elements of this Hilbert space are n-dimensional complex valued
    vectors with the usual inner product that takes the complex conjugate
    of the vector on the right.

    A classic example of this type of Hilbert space is spin-1/2, which is
    ``ComplexSpace(2)``. Generalizing to spin-s, the space is
    ``ComplexSpace(2*s+1)``.  Quantum computing with N qubits is done with the
    direct product space ``ComplexSpace(2)**N``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.hilbert import ComplexSpace
    >>> c1 = ComplexSpace(2)
    >>> c1
    C(2)
    >>> c1.dimension
    2

    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> c2
    C(n)
    >>> c2.dimension
    n

    """

    def __new__(cls, dimension):
        dimension = sympify(dimension)
        r = cls.eval(dimension)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, dimension)
        return obj

    @classmethod
    def eval(cls, dimension):
        if len(dimension.atoms()) == 1:
            if not (dimension.is_Integer and dimension > 0 or dimension is S.Infinity
            or dimension.is_Symbol):
                raise TypeError('The dimension of a ComplexSpace can only'
                                'be a positive integer, oo, or a Symbol: %r'
                                % dimension)
        else:
            for dim in dimension.atoms():
                if not (dim.is_Integer or dim is S.Infinity or dim.is_Symbol):
                    raise TypeError('The dimension of a ComplexSpace can only'
                                    ' contain integers, oo, or a Symbol: %r'
                                    % dim)

    @property
    def dimension(self):
        return self.args[0]

    def _sympyrepr(self, printer, *args):
        return "%s(%s)" % (self.__class__.__name__,
                           printer._print(self.dimension, *args))

    def _sympystr(self, printer, *args):
        return "C(%s)" % printer._print(self.dimension, *args)

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER C}'
        pform_exp = printer._print(self.dimension, *args)
        pform_base = prettyForm(ustr)
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        return r'\mathcal{C}^{%s}' % printer._print(self.dimension, *args)


class L2(HilbertSpace):
    """The Hilbert space of square integrable functions on an interval.

    An L2 object takes in a single SymPy Interval argument which represents
    the interval its functions (vectors) are defined on.

    Examples
    ========

    >>> from sympy import Interval, oo
    >>> from sympy.physics.quantum.hilbert import L2
    >>> hs = L2(Interval(0,oo))
    >>> hs
    L2(Interval(0, oo))
    >>> hs.dimension
    oo
    >>> hs.interval
    Interval(0, oo)

    """

    def __new__(cls, interval):
        if not isinstance(interval, Interval):
            raise TypeError('L2 interval must be an Interval instance: %r'
            % interval)
        obj = Basic.__new__(cls, interval)
        return obj

    @property
    def dimension(self):
        return S.Infinity

    @property
    def interval(self):
        return self.args[0]

    def _sympyrepr(self, printer, *args):
        return "L2(%s)" % printer._print(self.interval, *args)

    def _sympystr(self, printer, *args):
        return "L2(%s)" % printer._print(self.interval, *args)

    def _pretty(self, printer, *args):
        pform_exp = prettyForm('2')
        pform_base = prettyForm('L')
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        interval = printer._print(self.interval, *args)
        return r'{\mathcal{L}^2}\left( %s \right)' % interval


class FockSpace(HilbertSpace):
    """The Hilbert space for second quantization.

    Technically, this Hilbert space is a infinite direct sum of direct
    products of single particle Hilbert spaces [1]_. This is a mess, so we have
    a class to represent it directly.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import FockSpace
    >>> hs = FockSpace()
    >>> hs
    F
    >>> hs.dimension
    oo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fock_space
    """

    def __new__(cls):
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        return S.Infinity

    def _sympyrepr(self, printer, *args):
        return "FockSpace()"

    def _sympystr(self, printer, *args):
        return "F"

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER F}'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        return r'\mathcal{F}'


class TensorProductHilbertSpace(HilbertSpace):
    """A tensor product of Hilbert spaces [1]_.

    The tensor product between Hilbert spaces is represented by the
    operator ``*`` Products of the same Hilbert space will be combined into
    tensor powers.

    A ``TensorProductHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. In addition, multiplication of
    ``HilbertSpace`` objects will automatically return this tensor product
    object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c*f
    >>> hs
    C(2)*F
    >>> hs.dimension
    oo
    >>> hs.spaces
    (C(2), F)

    >>> c1 = ComplexSpace(2)
    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> hs = c1*c2
    >>> hs
    C(2)*C(n)
    >>> hs.dimension
    2*n

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, *args)
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct product."""
        new_args = []
        recall = False
        #flatten arguments
        for arg in args:
            if isinstance(arg, TensorProductHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, (HilbertSpace, TensorPowerHilbertSpace)):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be multiplied by \
                other Hilbert spaces: %r' % arg)
        #combine like arguments into direct powers
        comb_args = []
        prev_arg = None
        for new_arg in new_args:
            if prev_arg is not None:
                if isinstance(new_arg, TensorPowerHilbertSpace) and \
                    isinstance(prev_arg, TensorPowerHilbertSpace) and \
                        new_arg.base == prev_arg.base:
                    prev_arg = new_arg.base**(new_arg.exp + prev_arg.exp)
                elif isinstance(new_arg, TensorPowerHilbertSpace) and \
                        new_arg.base == prev_arg:
                    prev_arg = prev_arg**(new_arg.exp + 1)
                elif isinstance(prev_arg, TensorPowerHilbertSpace) and \
                        new_arg == prev_arg.base:
                    prev_arg = new_arg**(prev_arg.exp + 1)
                elif new_arg == prev_arg:
                    prev_arg = new_arg**2
                else:
                    comb_args.append(prev_arg)
                    prev_arg = new_arg
            elif prev_arg is None:
                prev_arg = new_arg
        comb_args.append(prev_arg)
        if recall:
            return TensorProductHilbertSpace(*comb_args)
        elif len(comb_args) == 1:
            return TensorPowerHilbertSpace(comb_args[0].base, comb_args[0].exp)
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x*y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this tensor product."""
        return self.args

    def _spaces_printer(self, printer, *args):
        spaces_strs = []
        for arg in self.args:
            s = printer._print(arg, *args)
            if isinstance(arg, DirectSumHilbertSpace):
                s = '(%s)' % s
            spaces_strs.append(s)
        return spaces_strs

    def _sympyrepr(self, printer, *args):
        spaces_reprs = self._spaces_printer(printer, *args)
        return "TensorProductHilbertSpace(%s)" % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        spaces_strs = self._spaces_printer(printer, *args)
        return '*'.join(spaces_strs)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                          TensorProductHilbertSpace)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right(' ' + '\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right(' x '))
        return pform

    def _latex(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            arg_s = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                 TensorProductHilbertSpace)):
                arg_s = r'\left(%s\right)' % arg_s
            s = s + arg_s
            if i != length - 1:
                s = s + r'\otimes '
        return s


class DirectSumHilbertSpace(HilbertSpace):
    """A direct sum of Hilbert spaces [1]_.

    This class uses the ``+`` operator to represent direct sums between
    different Hilbert spaces.

    A ``DirectSumHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. Also, addition of
    ``HilbertSpace`` objects will automatically return a direct sum object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c+f
    >>> hs
    C(2)+F
    >>> hs.dimension
    oo
    >>> list(hs.spaces)
    [C(2), F]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Direct_sums
    """
    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, *args)
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct product."""
        new_args = []
        recall = False
        #flatten arguments
        for arg in args:
            if isinstance(arg, DirectSumHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, HilbertSpace):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be summed with other \
                Hilbert spaces: %r' % arg)
        if recall:
            return DirectSumHilbertSpace(*new_args)
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x + y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this direct sum."""
        return self.args

    def _sympyrepr(self, printer, *args):
        spaces_reprs = [printer._print(arg, *args) for arg in self.args]
        return "DirectSumHilbertSpace(%s)" % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        spaces_strs = [printer._print(arg, *args) for arg in self.args]
        return '+'.join(spaces_strs)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                          TensorProductHilbertSpace)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right(' \N{CIRCLED PLUS} '))
                else:
                    pform = prettyForm(*pform.right(' + '))
        return pform

    def _latex(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            arg_s = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                 TensorProductHilbertSpace)):
                arg_s = r'\left(%s\right)' % arg_s
            s = s + arg_s
            if i != length - 1:
                s = s + r'\oplus '
        return s


class TensorPowerHilbertSpace(HilbertSpace):
    """An exponentiated Hilbert space [1]_.

    Tensor powers (repeated tensor products) are represented by the
    operator ``**`` Identical Hilbert spaces that are multiplied together
