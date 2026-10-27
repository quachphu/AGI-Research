        # If this function is an Undefined function of t, it is probably a
        # dynamic symbol, so we'll skip the (t). The rest of the code is
        # identical to the normal PrettyPrinter code
        if not (isinstance(func, UndefinedFunction) and (args == (t,))):
            return super()._print_Function(e)
        return pform


def vprint(expr, **settings):
    r"""Function for printing of expressions generated in the
    sympy.physics vector package.

    Extends SymPy's StrPrinter, takes the same setting accepted by SymPy's
    :func:`~.sstr`, and is equivalent to ``print(sstr(foo))``.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    >>> from sympy.physics.vector import vprint, dynamicsymbols
    >>> u1 = dynamicsymbols('u1')
    >>> print(u1)
    u1(t)
    >>> vprint(u1)
    u1

    """

    outstr = vsprint(expr, **settings)

    import builtins
    if (outstr != 'None'):
        builtins._ = outstr
        print(outstr)


def vsstrrepr(expr, **settings):
    """Function for displaying expression representation's with vector
    printing enabled.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstrrepr().

    """
    p = VectorStrReprPrinter(settings)
    return p.doprint(expr)


def vsprint(expr, **settings):
    r"""Function for displaying expressions generated in the
    sympy.physics vector package.

    Returns the output of vprint() as a string.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    >>> from sympy.physics.vector import vsprint, dynamicsymbols
    >>> u1, u2 = dynamicsymbols('u1 u2')
    >>> u2d = dynamicsymbols('u2', level=1)
    >>> print("%s = %s" % (u1, u2 + u2d))
    u1(t) = u2(t) + Derivative(u2(t), t)
    >>> print("%s = %s" % (vsprint(u1), vsprint(u2 + u2d)))
    u1 = u2 + u2'

    """

    string_printer = VectorStrPrinter(settings)
    return string_printer.doprint(expr)


def vpprint(expr, **settings):
    r"""Function for pretty printing of expressions generated in the
    sympy.physics vector package.

    Mainly used for expressions not inside a vector; the output of running
    scripts and generating equations of motion. Takes the same options as
    SymPy's :func:`~.pretty_print`; see that function for more information.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to pretty print
    settings : args
        Same as those accepted by SymPy's pretty_print.


    """

    pp = VectorPrettyPrinter(settings)

    # Note that this is copied from sympy.printing.pretty.pretty_print:

    # XXX: this is an ugly hack, but at least it works
    use_unicode = pp._settings['use_unicode']
    from sympy.printing.pretty.pretty_symbology import pretty_use_unicode
    uflag = pretty_use_unicode(use_unicode)

    try:
        return pp.doprint(expr)
    finally:
        pretty_use_unicode(uflag)


def vlatex(expr, **settings):
    r"""Function for printing latex representation of sympy.physics.vector
    objects.

    For latex representation of Vectors, Dyadics, and dynamicsymbols. Takes the
    same options as SymPy's :func:`~.latex`; see that function for more
    information;

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to represent in LaTeX form
    settings : args
        Same as latex()

    Examples
    ========

    >>> from sympy.physics.vector import vlatex, ReferenceFrame, dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q1, q2 = dynamicsymbols('q1 q2')
    >>> q1d, q2d = dynamicsymbols('q1 q2', 1)
    >>> q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    >>> vlatex(N.x + N.y)
    '\\mathbf{\\hat{n}_x} + \\mathbf{\\hat{n}_y}'
    >>> vlatex(q1 + q2)
    'q_{1} + q_{2}'
    >>> vlatex(q1d)
    '\\dot{q}_{1}'
    >>> vlatex(q1 * q2d)
    'q_{1} \\dot{q}_{2}'
    >>> vlatex(q1dd * q1 / q1d)
    '\\frac{q_{1} \\ddot{q}_{1}}{\\dot{q}_{1}}'

    """
    latex_printer = VectorLatexPrinter(settings)

    return latex_printer.doprint(expr)


def init_vprinting(**kwargs):
    """Initializes time derivative printing for all SymPy objects, i.e. any
    functions of time will be displayed in a more compact notation. The main
    benefit of this is for printing of time derivatives; instead of
    displaying as ``Derivative(f(t),t)``, it will display ``f'``. This is
    only actually needed for when derivatives are present and are not in a
    physics.vector.Vector or physics.vector.Dyadic object. This function is a
    light wrapper to :func:`~.init_printing`. Any keyword
    arguments for it are valid here.

    {0}

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> t, x = symbols('t, x')
    >>> omega = Function('omega')
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    Derivative(omega(t), t)

    Now use the string printer:

    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    omega'

    """
    kwargs['str_printer'] = vsstrrepr
    kwargs['pretty_printer'] = vpprint
    kwargs['latex_printer'] = vlatex
    init_printing(**kwargs)


params = init_printing.__doc__.split('Examples\n    ========')[0]  # type: ignore
init_vprinting.__doc__ = init_vprinting.__doc__.format(params)  # type: ignore
<|endoftext|>from sympy import (S, sympify, expand, sqrt, Add, zeros, acos,
                   ImmutableMatrix as Matrix, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.printing.defaults import Printable
from sympy.utilities.misc import filldedent
from sympy.core.evalf import EvalfMixin

from mpmath.libmp.libmpf import prec_to_dps


__all__ = ['Vector']


class Vector(Printable, EvalfMixin):
    """The class used to define vectors.

    It along with ReferenceFrame are the building blocks of describing a
    classical mechanics system in PyDy and sympy.physics.vector.

    Attributes
    ==========

    simp : Boolean
        Let certain methods use trigsimp on their outputs

    """

    simp = False
    is_number = False

    def __init__(self, inlist):
        """This is the constructor for the Vector class. You should not be
        calling this, it should only be used by other functions. You should be
        treating Vectors like you would with if you were doing the math by
        hand, and getting the first 3 from the standard basis vectors from a
        ReferenceFrame.

        The only exception is to create a zero vector:
        zv = Vector(0)

        """

        self.args = []
        if inlist == 0:
            inlist = []
        if isinstance(inlist, dict):
            d = inlist
        else:
            d = {}
            for inp in inlist:
                if inp[1] in d:
                    d[inp[1]] += inp[0]
                else:
                    d[inp[1]] = inp[0]

        for k, v in d.items():
            if v != Matrix([0, 0, 0]):
                self.args.append((v, k))

    @property
    def func(self):
        """Returns the class Vector. """
        return Vector

    def __hash__(self):
        return hash(tuple(self.args))

    def __add__(self, other):
        """The add operator for Vector. """
        if other == 0:
            return self
        other = _check_vector(other)
        return Vector(self.args + other.args)

    def dot(self, other):
        """Dot product of two vectors.

        Returns a scalar, the dot product of the two Vectors

        Parameters
        ==========

        other : Vector
            The Vector which we are dotting with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dot
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> dot(N.x, N.x)
        1
        >>> dot(N.x, N.y)
        0
        >>> A = N.orientnew('A', 'Axis', [q1, N.x])
        >>> dot(N.y, A.y)
        cos(q1)

        """

        from sympy.physics.vector.dyadic import Dyadic, _check_dyadic
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Vector(0)
            for v in other.args:
                ol += v[0] * v[2] * (v[1].dot(self))
            return ol
        other = _check_vector(other)
        out = S.Zero
        for v1 in self.args:
            for v2 in other.args:
                out += ((v2[0].T) * (v2[1].dcm(v1[1])) * (v1[0]))[0]
        if Vector.simp:
            return trigsimp(out, recursive=True)
        else:
            return out

    def __truediv__(self, other):
        """This uses mul and inputs self and 1 divided by other. """
        return self.__mul__(S.One / other)

    def __eq__(self, other):
        """Tests for equality.

        It is very import to note that this is only as good as the SymPy
        equality test; False does not always mean they are not equivalent
        Vectors.
        If other is 0, and self is empty, returns True.
        If other is 0 and self is not empty, returns False.
        If none of the above, only accepts other as a Vector.

        """

        if other == 0:
            other = Vector(0)
        try:
            other = _check_vector(other)
        except TypeError:
            return False
        if (self.args == []) and (other.args == []):
            return True
        elif (self.args == []) or (other.args == []):
            return False

        frame = self.args[0][1]
        for v in frame:
            if expand((self - other).dot(v)) != 0:
                return False
        return True

    def __mul__(self, other):
        """Multiplies the Vector by a sympifyable expression.

        Parameters
        ==========

        other : Sympifyable
            The scalar to multiply this Vector with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> b = Symbol('b')
        >>> V = 10 * b * N.x
        >>> print(V)
        10*b*N.x

        """

        newlist = list(self.args)
        other = sympify(other)
        for i in range(len(newlist)):
            newlist[i] = (other * newlist[i][0], newlist[i][1])
        return Vector(newlist)

    def __neg__(self):
        return self * -1

    def outer(self, other):
        """Outer product between two Vectors.

        A rank increasing operation, which returns a Dyadic from two Vectors

        Parameters
        ==========

        other : Vector
            The Vector to take the outer product with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> outer(N.x, N.x)
        (N.x|N.x)

        """

        from sympy.physics.vector.dyadic import Dyadic
        other = _check_vector(other)
        ol = Dyadic(0)
        for v in self.args:
            for v2 in other.args:
                # it looks this way because if we are in the same frame and
                # use the enumerate function on the same frame in a nested
                # fashion, then bad things happen
                ol += Dyadic([(v[0][0] * v2[0][0], v[1].x, v2[1].x)])
                ol += Dyadic([(v[0][0] * v2[0][1], v[1].x, v2[1].y)])
                ol += Dyadic([(v[0][0] * v2[0][2], v[1].x, v2[1].z)])
                ol += Dyadic([(v[0][1] * v2[0][0], v[1].y, v2[1].x)])
                ol += Dyadic([(v[0][1] * v2[0][1], v[1].y, v2[1].y)])
                ol += Dyadic([(v[0][1] * v2[0][2], v[1].y, v2[1].z)])
                ol += Dyadic([(v[0][2] * v2[0][0], v[1].z, v2[1].x)])
                ol += Dyadic([(v[0][2] * v2[0][1], v[1].z, v2[1].y)])
                ol += Dyadic([(v[0][2] * v2[0][2], v[1].z, v2[1].z)])
        return ol

    def _latex(self, printer):
        """Latex Printing method. """

        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return str(0)
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            for j in 0, 1, 2:
                # if the coef of the basis vector is 1, we skip the 1
                if v[0][j] == 1:
                    ol.append(' + ' + v[1].latex_vecs[j])
                # if the coef of the basis vector is -1, we skip the 1
                elif v[0][j] == -1:
                    ol.append(' - ' + v[1].latex_vecs[j])
                elif v[0][j] != 0:
                    # If the coefficient of the basis vector is not 1 or -1;
                    # also, we might wrap it in parentheses, for readability.
                    arg_str = printer._print(v[0][j])
                    if isinstance(v[0][j], Add):
                        arg_str = "(%s)" % arg_str
                    if arg_str[0] == '-':
                        arg_str = arg_str[1:]
                        str_start = ' - '
                    else:
                        str_start = ' + '
                    ol.append(str_start + arg_str + v[1].latex_vecs[j])
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def _pretty(self, printer):
        """Pretty Printing method. """
        from sympy.printing.pretty.stringpict import prettyForm

        terms = []

        def juxtapose(a, b):
            pa = printer._print(a)
            pb = printer._print(b)
            if a.is_Add:
                pa = prettyForm(*pa.parens())
            return printer._print_seq([pa, pb], delimiter=' ')

        for M, N in self.args:
            for i in range(3):
                if M[i] == 0:
                    continue
                elif M[i] == 1:
                    terms.append(prettyForm(N.pretty_vecs[i]))
                elif M[i] == -1:
                    terms.append(prettyForm("-1") * prettyForm(N.pretty_vecs[i]))
                else:
                    terms.append(juxtapose(M[i], N.pretty_vecs[i]))

        if terms:
            pretty_result = prettyForm.__add__(*terms)
        else:
            pretty_result = prettyForm("0")

        return pretty_result

    def __rsub__(self, other):
        return (-1 * self) + other

    def _sympystr(self, printer, order=True):
        """Printing method. """
        if not order or len(self.args) == 1:
            ar = list(self.args)
        elif len(self.args) == 0:
            return printer._print(0)
        else:
            d = {v[1]: v[0] for v in self.args}
            keys = sorted(d.keys(), key=lambda x: x.index)
            ar = []
            for key in keys:
                ar.append((d[key], key))
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            for j in 0, 1, 2:
                # if the coef of the basis vector is 1, we skip the 1
                if v[0][j] == 1:
                    ol.append(' + ' + v[1].str_vecs[j])
                # if the coef of the basis vector is -1, we skip the 1
                elif v[0][j] == -1:
                    ol.append(' - ' + v[1].str_vecs[j])
                elif v[0][j] != 0:
                    # If the coefficient of the basis vector is not 1 or -1;
                    # also, we might wrap it in parentheses, for readability.
                    arg_str = printer._print(v[0][j])
                    if isinstance(v[0][j], Add):
                        arg_str = "(%s)" % arg_str
                    if arg_str[0] == '-':
                        arg_str = arg_str[1:]
                        str_start = ' - '
                    else:
                        str_start = ' + '
                    ol.append(str_start + arg_str + '*' + v[1].str_vecs[j])
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def __sub__(self, other):
        """The subtraction operator. """
        return self.__add__(other * -1)

    def cross(self, other):
        """The cross product operator for two Vectors.

        Returns a Vector, expressed in the same ReferenceFrames as self.

        Parameters
        ==========

        other : Vector
            The Vector which we are crossing with

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame, cross
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> cross(N.x, N.y)
        N.z
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, q1, N.x)
        >>> cross(A.x, N.y)
        N.z
        >>> cross(N.y, A.x)
        - sin(q1)*A.y - cos(q1)*A.z

        """

        from sympy.physics.vector.dyadic import Dyadic, _check_dyadic
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Dyadic(0)
            for i, v in enumerate(other.args):
                ol += v[0] * ((self.cross(v[1])).outer(v[2]))
            return ol
        other = _check_vector(other)
        if other.args == []:
            return Vector(0)

        def _det(mat):
            """This is needed as a little method for to find the determinant
            of a list in python; needs to work for a 3x3 list.
            SymPy's Matrix will not take in Vector, so need a custom function.
            You should not be calling this.

            """

            return (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
                    + mat[0][1] * (mat[1][2] * mat[2][0] - mat[1][0] *
                    mat[2][2]) + mat[0][2] * (mat[1][0] * mat[2][1] -
                    mat[1][1] * mat[2][0]))

        outlist = []
        ar = other.args  # For brevity
        for v in ar:
            tempx = v[1].x
            tempy = v[1].y
            tempz = v[1].z
            tempm = ([[tempx, tempy, tempz],
                      [self.dot(tempx), self.dot(tempy), self.dot(tempz)],
                      [Vector([v]).dot(tempx), Vector([v]).dot(tempy),
                       Vector([v]).dot(tempz)]])
            outlist += _det(tempm).args
        return Vector(outlist)

    __radd__ = __add__
    __rmul__ = __mul__

    def separate(self):
        """
        The constituents of this vector in different reference frames,
        as per its definition.

        Returns a dict mapping each ReferenceFrame to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> R1 = ReferenceFrame('R1')
        >>> R2 = ReferenceFrame('R2')
        >>> v = R1.x + R2.x
        >>> v.separate() == {R1: R1.x, R2: R2.x}
        True

        """

        components = {}
        for x in self.args:
            components[x[1]] = Vector([x])
        return components

    def __and__(self, other):
        return self.dot(other)
    __and__.__doc__ = dot.__doc__
    __rand__ = __and__

    def __xor__(self, other):
        return self.cross(other)
    __xor__.__doc__ = cross.__doc__

    def __or__(self, other):
        return self.outer(other)
    __or__.__doc__ = outer.__doc__

    def diff(self, var, frame, var_in_dcm=True):
        """Returns the partial derivative of the vector with respect to a
        variable in the provided reference frame.

        Parameters
        ==========
        var : Symbol
            What the partial derivative is taken with respect to.
        frame : ReferenceFrame
            The reference frame that the partial derivative is taken in.
        var_in_dcm : boolean
            If true, the differentiation algorithm assumes that the variable
            may be present in any of the direction cosine matrices that relate
            the frame to the frames of any component of the vector. But if it
            is known that the variable is not present in the direction cosine
            matrices, false can be set to skip full reexpression in the desired
            frame.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.vector import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> t = Symbol('t')
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.diff(t, N)
        - sin(q1)*q1'*N.x - cos(q1)*q1'*N.z
        >>> A.x.diff(t, N).express(A).simplify()
        - q1'*A.z
        >>> B = ReferenceFrame('B')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> v = u1 * A.x + u2 * B.y
        >>> v.diff(u2, N, var_in_dcm=False)
        B.y

        """

        from sympy.physics.vector.frame import _check_frame

        _check_frame(frame)
        var = sympify(var)

        inlist = []

        for vector_component in self.args:
            measure_number = vector_component[0]
            component_frame = vector_component[1]
            if component_frame == frame:
                inlist += [(measure_number.diff(var), frame)]
            else:
                # If the direction cosine matrix relating the component frame
                # with the derivative frame does not contain the variable.
                if not var_in_dcm or (frame.dcm(component_frame).diff(var) ==
                                      zeros(3, 3)):
                    inlist += [(measure_number.diff(var), component_frame)]
                else:  # else express in the frame
                    reexp_vec_comp = Vector([vector_component]).express(frame)
                    deriv = reexp_vec_comp.args[0][0].diff(var)
                    inlist += Vector([(deriv, frame)]).args

        return Vector(inlist)

    def express(self, otherframe, variables=False):
        """
        Returns a Vector equivalent to this one, expressed in otherframe.
        Uses the global express method.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame for this Vector to be described in

        variables : boolean
            If True, the coordinate symbols(if present) in this Vector
            are re-expressed in terms otherframe

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.express(N)
        cos(q1)*N.x - sin(q1)*N.z

        """
        from sympy.physics.vector import express
        return express(self, otherframe, variables=variables)

    def to_matrix(self, reference_frame):
        """Returns the matrix form of the vector with respect to the given
        frame.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows of the matrix correspond to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,1)
            The matrix that gives the 1D vector.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> a, b, c = symbols('a, b, c')
        >>> N = ReferenceFrame('N')
        >>> vector = a * N.x + b * N.y + c * N.z
        >>> vector.to_matrix(N)
        Matrix([
        [a],
        [b],
        [c]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> vector.to_matrix(A)
        Matrix([
        [                         a],
        [ b*cos(beta) + c*sin(beta)],
        [-b*sin(beta) + c*cos(beta)]])

        """

        return Matrix([self.dot(unit_vec) for unit_vec in
                       reference_frame]).reshape(3, 1)

    def doit(self, **hints):
        """Calls .doit() on each term in the Vector"""
        d = {}
        for v in self.args:
            d[v[1]] = v[0].applyfunc(lambda x: x.doit(**hints))
        return Vector(d)

    def dt(self, otherframe):
        """
        Returns a Vector which is the time derivative of
        the self Vector, taken in frame otherframe.

        Calls the global time_derivative method

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame to calculate the time derivative in

        """
        from sympy.physics.vector import time_derivative
        return time_derivative(self, otherframe)

    def simplify(self):
        """Returns a simplified Vector."""
        d = {}
        for v in self.args:
            d[v[1]] = simplify(v[0])
        return Vector(d)

    def subs(self, *args, **kwargs):
        """Substitution on the Vector.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = N.x * s
        >>> a.subs({s: 2})
        2*N.x

        """

        d = {}
        for v in self.args:
            d[v[1]] = v[0].subs(*args, **kwargs)
        return Vector(d)

    def magnitude(self):
        """Returns the magnitude (Euclidean norm) of self.

        Warnings
        ========

        Python ignores the leading negative sign so that might
        give wrong results.
        ``-A.x.magnitude()`` would be treated as ``-(A.x.magnitude())``,
        instead of ``(-A.x).magnitude()``.

        """
        return sqrt(self.dot(self))

    def normalize(self):
        """Returns a Vector of magnitude 1, codirectional with self."""
        return Vector(self.args + []) / self.magnitude()

    def applyfunc(self, f):
        """Apply a function to each component of a vector."""
        if not callable(f):
            raise TypeError("`f` must be callable.")

        d = {}
        for v in self.args:
            d[v[1]] = v[0].applyfunc(f)
        return Vector(d)

    def angle_between(self, vec):
        """
        Returns the smallest angle between Vector 'vec' and self.

        Parameter
        =========

        vec : Vector
            The Vector between which angle is needed.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame("A")
        >>> v1 = A.x
        >>> v2 = A.y
        >>> v1.angle_between(v2)
        pi/2

        >>> v3 = A.x + A.y + A.z
        >>> v1.angle_between(v3)
        acos(sqrt(3)/3)

        Warnings
        ========

        Python ignores the leading negative sign so that might give wrong
        results. ``-A.x.angle_between()`` would be treated as
        ``-(A.x.angle_between())``, instead of ``(-A.x).angle_between()``.

        """

        vec1 = self.normalize()
        vec2 = vec.normalize()
        angle = acos(vec1.dot(vec2))
        return angle

    def free_symbols(self, reference_frame):
        """Returns the free symbols in the measure numbers of the vector
        expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free symbols of the given
            vector is to be determined.

        Returns
        =======
        set of Symbol
            set of symbols present in the measure numbers of
            ``reference_frame``.

        """

        return self.to_matrix(reference_frame).free_symbols

    def free_dynamicsymbols(self, reference_frame):
        """Returns the free dynamic symbols (functions of time ``t``) in the
        measure numbers of the vector expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free dynamic symbols of the
            given vector is to be determined.

        Returns
        =======
        set
            Set of functions of time ``t``, e.g.
            ``Function('f')(me.dynamicsymbols._t)``.

        """
        # TODO : Circular dependency if imported at top. Should move
        # find_dynamicsymbols into physics.vector.functions.
        from sympy.physics.mechanics.functions import find_dynamicsymbols

        return find_dynamicsymbols(self, reference_frame=reference_frame)

    def _eval_evalf(self, prec):
        if not self.args:
            return self
        new_args = []
        dps = prec_to_dps(prec)
        for mat, frame in self.args:
            new_args.append([mat.evalf(n=dps), frame])
        return Vector(new_args)

    def xreplace(self, rule):
        """Replace occurrences of objects within the measure numbers of the
        vector.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Vector
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame('A')
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * A.x).xreplace({x: pi})
        (pi*y + 1)*A.x
        >>> ((1 + x*y) * A.x).xreplace({x: pi, y: 2})
        (1 + 2*pi)*A.x

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * A.x).xreplace({x*y: pi})
        (z + pi)*A.x
        >>> ((x*y*z) * A.x).xreplace({x*y: pi})
        x*y*z*A.x

        """

        new_args = []
        for mat, frame in self.args:
            mat = mat.xreplace(rule)
            new_args.append([mat, frame])
        return Vector(new_args)


class VectorTypeError(TypeError):

    def __init__(self, other, want):
        msg = filldedent("Expected an instance of %s, but received object "
                         "'%s' of %s." % (type(want), other, type(other)))
        super().__init__(msg)


def _check_vector(other):
    if not isinstance(other, Vector):
        raise TypeError('A Vector must be supplied')
    return other
<|endoftext|>from functools import reduce

from sympy import (sympify, diff, sin, cos, Matrix, symbols,
                                Function, S, Symbol, linear_eq_to_matrix)
from sympy.integrals.integrals import integrate
from sympy.simplify.trigsimp import trigsimp
from .vector import Vector, _check_vector
from .frame import CoordinateSym, _check_frame
from .dyadic import Dyadic
from .printing import vprint, vsprint, vpprint, vlatex, init_vprinting
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import translate

__all__ = ['cross', 'dot', 'express', 'time_derivative', 'outer',
           'kinematic_equations', 'get_motion_params', 'partial_velocity',
           'dynamicsymbols', 'vprint', 'vsprint', 'vpprint', 'vlatex',
           'init_vprinting']


def cross(vec1, vec2):
    """Cross product convenience wrapper for Vector.cross(): \n"""
    if not isinstance(vec1, (Vector, Dyadic)):
        raise TypeError('Cross product is between two vectors')
    return vec1 ^ vec2


cross.__doc__ += Vector.cross.__doc__  # type: ignore


def dot(vec1, vec2):
    """Dot product convenience wrapper for Vector.dot(): \n"""
    if not isinstance(vec1, (Vector, Dyadic)):
        raise TypeError('Dot product is between two vectors')
    return vec1 & vec2


dot.__doc__ += Vector.dot.__doc__  # type: ignore


def express(expr, frame, frame2=None, variables=False):
    """
    Global function for 'express' functionality.

    Re-expresses a Vector, scalar(sympyfiable) or Dyadic in given frame.

    Refer to the local methods of Vector and Dyadic for details.
    If 'variables' is True, then the coordinate variables (CoordinateSym
    instances) of other frames present in the vector/scalar field or
    dyadic expression are also substituted in terms of the base scalars of
    this frame.

    Parameters
    ==========

    expr : Vector/Dyadic/scalar(sympyfiable)
        The expression to re-express in ReferenceFrame 'frame'

    frame: ReferenceFrame
        The reference frame to express expr in

    frame2 : ReferenceFrame
        The other frame required for re-expression(only for Dyadic expr)

    variables : boolean
        Specifies whether to substitute the coordinate variables present
        in expr, in terms of those of frame

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> B = N.orientnew('B', 'Axis', [q, N.z])
    >>> d = outer(N.x, N.x)
    >>> from sympy.physics.vector import express
    >>> express(d, B, N)
    cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)
    >>> express(B.x, N)
    cos(q)*N.x + sin(q)*N.y
    >>> express(N[0], B, variables=True)
    B_x*cos(q) - B_y*sin(q)

    """

    _check_frame(frame)

    if expr == 0:
        return expr

    if isinstance(expr, Vector):
        # Given expr is a Vector
        if variables:
            # If variables attribute is True, substitute the coordinate
            # variables in the Vector
            frame_list = [x[-1] for x in expr.args]
            subs_dict = {}
            for f in frame_list:
                subs_dict.update(f.variable_map(frame))
            expr = expr.subs(subs_dict)
        # Re-express in this frame
        outvec = Vector([])
        for v in expr.args:
            if v[1] != frame:
                temp = frame.dcm(v[1]) * v[0]
                if Vector.simp:
                    temp = temp.applyfunc(lambda x:
                                          trigsimp(x, method='fu'))
                outvec += Vector([(temp, frame)])
            else:
                outvec += Vector([v])
        return outvec

    if isinstance(expr, Dyadic):
        if frame2 is None:
            frame2 = frame
        _check_frame(frame2)
        ol = Dyadic(0)
        for v in expr.args:
            ol += express(v[0], frame, variables=variables) * \
                  (express(v[1], frame, variables=variables) |
                   express(v[2], frame2, variables=variables))
        return ol

    else:
        if variables:
            # Given expr is a scalar field
            frame_set = set()
            expr = sympify(expr)
            # Substitute all the coordinate variables
            for x in expr.free_symbols:
                if isinstance(x, CoordinateSym) and x.frame != frame:
                    frame_set.add(x.frame)
            subs_dict = {}
            for f in frame_set:
                subs_dict.update(f.variable_map(frame))
            return expr.subs(subs_dict)
        return expr


def time_derivative(expr, frame, order=1):
    """
    Calculate the time derivative of a vector/scalar field function
    or dyadic expression in given frame.

    References
    ==========

    https://en.wikipedia.org/wiki/Rotating_reference_frame#Time_derivatives_in_the_two_frames

    Parameters
    ==========

    expr : Vector/Dyadic/sympifyable
        The expression whose time derivative is to be calculated

    frame : ReferenceFrame
        The reference frame to calculate the time derivative in

    order : integer
        The order of the derivative to be calculated

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import Symbol
    >>> q1 = Symbol('q1')
    >>> u1 = dynamicsymbols('u1')
    >>> N = ReferenceFrame('N')
    >>> A = N.orientnew('A', 'Axis', [q1, N.x])
    >>> v = u1 * N.x
    >>> A.set_ang_vel(N, 10*A.x)
    >>> from sympy.physics.vector import time_derivative
    >>> time_derivative(v, N)
    u1'*N.x
    >>> time_derivative(u1*A[0], N)
    N_x*u1'
    >>> B = N.orientnew('B', 'Axis', [u1, N.z])
    >>> from sympy.physics.vector import outer
    >>> d = outer(N.x, N.x)
    >>> time_derivative(d, B)
    - u1'*(N.y|N.x) - u1'*(N.x|N.y)

    """

    t = dynamicsymbols._t
    _check_frame(frame)

    if order == 0:
        return expr
    if order % 1 != 0 or order < 0:
        raise ValueError("Unsupported value of order entered")

    if isinstance(expr, Vector):
        outlist = []
        for v in expr.args:
            if v[1] == frame:
                outlist += [(express(v[0], frame, variables=True).diff(t),
                             frame)]
            else:
                outlist += (time_derivative(Vector([v]), v[1]) +
                            (v[1].ang_vel_in(frame) ^ Vector([v]))).args
        outvec = Vector(outlist)
        return time_derivative(outvec, frame, order - 1)

    if isinstance(expr, Dyadic):
        ol = Dyadic(0)
        for v in expr.args:
            ol += (v[0].diff(t) * (v[1] | v[2]))
            ol += (v[0] * (time_derivative(v[1], frame) | v[2]))
            ol += (v[0] * (v[1] | time_derivative(v[2], frame)))
        return time_derivative(ol, frame, order - 1)

    else:
        return diff(express(expr, frame, variables=True), t, order)


def outer(vec1, vec2):
    """Outer product convenience wrapper for Vector.outer():\n"""
    if not isinstance(vec1, Vector):
        raise TypeError('Outer product is between two Vectors')
    return vec1.outer(vec2)


outer.__doc__ += Vector.outer.__doc__  # type: ignore


def kinematic_equations(speeds, coords, rot_type, rot_order=''):
    """Gives equations relating the qdot's to u's for a rotation type.

    Supply rotation type and order as in orient. Speeds are assumed to be
    body-fixed; if we are defining the orientation of B in A using by rot_type,
    the angular velocity of B in A is assumed to be in the form: speed[0]*B.x +
    speed[1]*B.y + speed[2]*B.z

    Parameters
    ==========

    speeds : list of length 3
        The body fixed angular velocity measure numbers.
    coords : list of length 3 or 4
        The coordinates used to define the orientation of the two frames.
    rot_type : str
        The type of rotation used to create the equations. Body, Space, or
        Quaternion only
    rot_order : str or int
        If applicable, the order of a series of rotations.

    Examples
    ========

    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy.physics.vector import kinematic_equations, vprint
    >>> u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    >>> q1, q2, q3 = dynamicsymbols('q1 q2 q3')
    >>> vprint(kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', '313'),
    ...     order=None)
    [-(u1*sin(q3) + u2*cos(q3))/sin(q2) + q1', -u1*cos(q3) + u2*sin(q3) + q2', (u1*sin(q3) + u2*cos(q3))*cos(q2)/sin(q2) - u3 + q3']

    """

    # Code below is checking and sanitizing input
    approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131',
                       '212', '232', '313', '323', '1', '2', '3', '')
    # make sure XYZ => 123 and rot_type is in lower case
    rot_order = translate(str(rot_order), 'XYZxyz', '123123')
    rot_type = rot_type.lower()

    if not isinstance(speeds, (list, tuple)):
        raise TypeError('Need to supply speeds in a list')
    if len(speeds) != 3:
        raise TypeError('Need to supply 3 body-fixed speeds')
    if not isinstance(coords, (list, tuple)):
        raise TypeError('Need to supply coordinates in a list')
    if rot_type in ['body', 'space']:
        if rot_order not in approved_orders:
            raise ValueError('Not an acceptable rotation order')
        if len(coords) != 3:
            raise ValueError('Need 3 coordinates for body or space')
        # Actual hard-coded kinematic differential equations
        w1, w2, w3 = speeds
        if w1 == w2 == w3 == 0:
            return [S.Zero]*3
        q1, q2, q3 = coords
        q1d, q2d, q3d = [diff(i, dynamicsymbols._t) for i in coords]
        s1, s2, s3 = [sin(q1), sin(q2), sin(q3)]
        c1, c2, c3 = [cos(q1), cos(q2), cos(q3)]
        if rot_type == 'body':
            if rot_order == '123':
                return [q1d - (w1 * c3 - w2 * s3) / c2, q2d - w1 * s3 - w2 *
                        c3, q3d - (-w1 * c3 + w2 * s3) * s2 / c2 - w3]
            if rot_order == '231':
                return [q1d - (w2 * c3 - w3 * s3) / c2, q2d - w2 * s3 - w3 *
                        c3, q3d - w1 - (- w2 * c3 + w3 * s3) * s2 / c2]
            if rot_order == '312':
                return [q1d - (-w1 * s3 + w3 * c3) / c2, q2d - w1 * c3 - w3 *
                        s3, q3d - (w1 * s3 - w3 * c3) * s2 / c2 - w2]
            if rot_order == '132':
                return [q1d - (w1 * c3 + w3 * s3) / c2, q2d + w1 * s3 - w3 *
                        c3, q3d - (w1 * c3 + w3 * s3) * s2 / c2 - w2]
            if rot_order == '213':
                return [q1d - (w1 * s3 + w2 * c3) / c2, q2d - w1 * c3 + w2 *
                        s3, q3d - (w1 * s3 + w2 * c3) * s2 / c2 - w3]
            if rot_order == '321':
                return [q1d - (w2 * s3 + w3 * c3) / c2, q2d - w2 * c3 + w3 *
                        s3, q3d - w1 - (w2 * s3 + w3 * c3) * s2 / c2]
            if rot_order == '121':
                return [q1d - (w2 * s3 + w3 * c3) / s2, q2d - w2 * c3 + w3 *
                        s3, q3d - w1 + (w2 * s3 + w3 * c3) * c2 / s2]
            if rot_order == '131':
                return [q1d - (-w2 * c3 + w3 * s3) / s2, q2d - w2 * s3 - w3 *
                        c3, q3d - w1 - (w2 * c3 - w3 * s3) * c2 / s2]
            if rot_order == '212':
                return [q1d - (w1 * s3 - w3 * c3) / s2, q2d - w1 * c3 - w3 *
                        s3, q3d - (-w1 * s3 + w3 * c3) * c2 / s2 - w2]
            if rot_order == '232':
                return [q1d - (w1 * c3 + w3 * s3) / s2, q2d + w1 * s3 - w3 *
                        c3, q3d + (w1 * c3 + w3 * s3) * c2 / s2 - w2]
            if rot_order == '313':
                return [q1d - (w1 * s3 + w2 * c3) / s2, q2d - w1 * c3 + w2 *
                        s3, q3d + (w1 * s3 + w2 * c3) * c2 / s2 - w3]
            if rot_order == '323':
                return [q1d - (-w1 * c3 + w2 * s3) / s2, q2d - w1 * s3 - w2 *
                        c3, q3d - (w1 * c3 - w2 * s3) * c2 / s2 - w3]
        if rot_type == 'space':
            if rot_order == '123':
                return [q1d - w1 - (w2 * s1 + w3 * c1) * s2 / c2, q2d - w2 *
                        c1 + w3 * s1, q3d - (w2 * s1 + w3 * c1) / c2]
            if rot_order == '231':
                return [q1d - (w1 * c1 + w3 * s1) * s2 / c2 - w2, q2d + w1 *
                        s1 - w3 * c1, q3d - (w1 * c1 + w3 * s1) / c2]
            if rot_order == '312':
                return [q1d - (w1 * s1 + w2 * c1) * s2 / c2 - w3, q2d - w1 *
                        c1 + w2 * s1, q3d - (w1 * s1 + w2 * c1) / c2]
            if rot_order == '132':
                return [q1d - w1 - (-w2 * c1 + w3 * s1) * s2 / c2, q2d - w2 *
                        s1 - w3 * c1, q3d - (w2 * c1 - w3 * s1) / c2]
            if rot_order == '213':
                return [q1d - (w1 * s1 - w3 * c1) * s2 / c2 - w2, q2d - w1 *
                        c1 - w3 * s1, q3d - (-w1 * s1 + w3 * c1) / c2]
            if rot_order == '321':
                return [q1d - (-w1 * c1 + w2 * s1) * s2 / c2 - w3, q2d - w1 *
                        s1 - w2 * c1, q3d - (w1 * c1 - w2 * s1) / c2]
            if rot_order == '121':
                return [q1d - w1 + (w2 * s1 + w3 * c1) * c2 / s2, q2d - w2 *
                        c1 + w3 * s1, q3d - (w2 * s1 + w3 * c1) / s2]
            if rot_order == '131':
                return [q1d - w1 - (w2 * c1 - w3 * s1) * c2 / s2, q2d - w2 *
                        s1 - w3 * c1, q3d - (-w2 * c1 + w3 * s1) / s2]
            if rot_order == '212':
                return [q1d - (-w1 * s1 + w3 * c1) * c2 / s2 - w2, q2d - w1 *
                        c1 - w3 * s1, q3d - (w1 * s1 - w3 * c1) / s2]
            if rot_order == '232':
                return [q1d + (w1 * c1 + w3 * s1) * c2 / s2 - w2, q2d + w1 *
                        s1 - w3 * c1, q3d - (w1 * c1 + w3 * s1) / s2]
            if rot_order == '313':
                return [q1d + (w1 * s1 + w2 * c1) * c2 / s2 - w3, q2d - w1 *
                        c1 + w2 * s1, q3d - (w1 * s1 + w2 * c1) / s2]
            if rot_order == '323':
                return [q1d - (w1 * c1 - w2 * s1) * c2 / s2 - w3, q2d - w1 *
                        s1 - w2 * c1, q3d - (-w1 * c1 + w2 * s1) / s2]
    elif rot_type == 'quaternion':
        if rot_order != '':
            raise ValueError('Cannot have rotation order for quaternion')
        if len(coords) != 4:
            raise ValueError('Need 4 coordinates for quaternion')
        # Actual hard-coded kinematic differential equations
        e0, e1, e2, e3 = coords
        w = Matrix(speeds + [0])
        E = Matrix([[e0, -e3, e2, e1],
                    [e3, e0, -e1, e2],
                    [-e2, e1, e0, e3],
                    [-e1, -e2, -e3, e0]])
        edots = Matrix([diff(i, dynamicsymbols._t) for i in [e1, e2, e3, e0]])
        return list(edots.T - 0.5 * w.T * E.T)
    else:
        raise ValueError('Not an approved rotation type for this function')


def get_motion_params(frame, **kwargs):
    """
    Returns the three motion parameters - (acceleration, velocity, and
    position) as vectorial functions of time in the given frame.

    If a higher order differential function is provided, the lower order
    functions are used as boundary conditions. For example, given the
    acceleration, the velocity and position parameters are taken as
    boundary conditions.

    The values of time at which the boundary conditions are specified
    are taken from timevalue1(for position boundary condition) and
    timevalue2(for velocity boundary condition).

    If any of the boundary conditions are not provided, they are taken
    to be zero by default (zero vectors, in case of vectorial inputs). If
    the boundary conditions are also functions of time, they are converted
    to constants by substituting the time values in the dynamicsymbols._t
    time Symbol.

    This function can also be used for calculating rotational motion
    parameters. Have a look at the Parameters and Examples for more clarity.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame to express the motion parameters in

    acceleration : Vector
        Acceleration of the object/frame as a function of time

    velocity : Vector
        Velocity as function of time or as boundary condition
        of velocity at time = timevalue1

    position : Vector
        Velocity as function of time or as boundary condition
        of velocity at time = timevalue1

    timevalue1 : sympyfiable
        Value of time for position boundary condition

    timevalue2 : sympyfiable
        Value of time for velocity boundary condition

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, get_motion_params, dynamicsymbols
    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> from sympy import symbols
    >>> R = ReferenceFrame('R')
    >>> v1, v2, v3 = dynamicsymbols('v1 v2 v3')
    >>> v = v1*R.x + v2*R.y + v3*R.z
    >>> get_motion_params(R, position = v)
    (v1''*R.x + v2''*R.y + v3''*R.z, v1'*R.x + v2'*R.y + v3'*R.z, v1*R.x + v2*R.y + v3*R.z)
    >>> a, b, c = symbols('a b c')
    >>> v = a*R.x + b*R.y + c*R.z
    >>> get_motion_params(R, velocity = v)
    (0, a*R.x + b*R.y + c*R.z, a*t*R.x + b*t*R.y + c*t*R.z)
    >>> parameters = get_motion_params(R, acceleration = v)
    >>> parameters[1]
    a*t*R.x + b*t*R.y + c*t*R.z
    >>> parameters[2]
    a*t**2/2*R.x + b*t**2/2*R.y + c*t**2/2*R.z

    """

    def _process_vector_differential(vectdiff, condition, variable, ordinate,
                                     frame):
        """
        Helper function for get_motion methods. Finds derivative of vectdiff
        wrt variable, and its integral using the specified boundary condition
        at value of variable = ordinate.
        Returns a tuple of - (derivative, function and integral) wrt vectdiff

        """

        # Make sure boundary condition is independent of 'variable'
        if condition != 0:
            condition = express(condition, frame, variables=True)
        # Special case of vectdiff == 0
        if vectdiff == Vector(0):
            return (0, 0, condition)
        # Express vectdiff completely in condition's frame to give vectdiff1
        vectdiff1 = express(vectdiff, frame)
        # Find derivative of vectdiff
        vectdiff2 = time_derivative(vectdiff, frame)
        # Integrate and use boundary condition
        vectdiff0 = Vector(0)
        lims = (variable, ordinate, variable)
        for dim in frame:
            function1 = vectdiff1.dot(dim)
            abscissa = dim.dot(condition).subs({variable: ordinate})
            # Indefinite integral of 'function1' wrt 'variable', using
            # the given initial condition (ordinate, abscissa).
            vectdiff0 += (integrate(function1, lims) + abscissa) * dim
        # Return tuple
        return (vectdiff2, vectdiff, vectdiff0)

    _check_frame(frame)
    # Decide mode of operation based on user's input
    if 'acceleration' in kwargs:
        mode = 2
    elif 'velocity' in kwargs:
        mode = 1
    else:
        mode = 0
    # All the possible parameters in kwargs
    # Not all are required for every case
    # If not specified, set to default values(may or may not be used in
    # calculations)
    conditions = ['acceleration', 'velocity', 'position',
                  'timevalue', 'timevalue1', 'timevalue2']
    for i, x in enumerate(conditions):
        if x not in kwargs:
            if i < 3:
                kwargs[x] = Vector(0)
            else:
                kwargs[x] = S.Zero
        elif i < 3:
            _check_vector(kwargs[x])
        else:
            kwargs[x] = sympify(kwargs[x])
    if mode == 2:
        vel = _process_vector_differential(kwargs['acceleration'],
                                           kwargs['velocity'],
                                           dynamicsymbols._t,
                                           kwargs['timevalue2'], frame)[2]
        pos = _process_vector_differential(vel, kwargs['position'],
                                           dynamicsymbols._t,
                                           kwargs['timevalue1'], frame)[2]
        return (kwargs['acceleration'], vel, pos)
    elif mode == 1:
        return _process_vector_differential(kwargs['velocity'],
                                            kwargs['position'],
                                            dynamicsymbols._t,
                                            kwargs['timevalue1'], frame)
    else:
        vel = time_derivative(kwargs['position'], frame)
        acc = time_derivative(vel, frame)
        return (acc, vel, kwargs['position'])


def partial_velocity(vel_vecs, gen_speeds, frame):
    """Returns a list of partial velocities with respect to the provided
    generalized speeds in the given reference frame for each of the supplied
    velocity vectors.

    The output is a list of lists. The outer list has a number of elements
    equal to the number of supplied velocity vectors. The inner lists are, for
    each velocity vector, the partial derivatives of that velocity vector with
    respect to the generalized speeds supplied.

    Parameters
    ==========

    vel_vecs : iterable
        An iterable of velocity vectors (angular or linear).
    gen_speeds : iterable
        An iterable of generalized speeds.
    frame : ReferenceFrame
        The reference frame that the partial derivatives are going to be taken
        in.

    Examples
    ========

    >>> from sympy.physics.vector import Point, ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy.physics.vector import partial_velocity
    >>> u = dynamicsymbols('u')
    >>> N = ReferenceFrame('N')
    >>> P = Point('P')
    >>> P.set_vel(N, u * N.x)
    >>> vel_vecs = [P.vel(N)]
    >>> gen_speeds = [u]
    >>> partial_velocity(vel_vecs, gen_speeds, N)
    [[N.x]]

    """

    if not iterable(vel_vecs):
        raise TypeError('Velocity vectors must be contained in an iterable.')

    if not iterable(gen_speeds):
        raise TypeError('Generalized speeds must be contained in an iterable')

    vec_partials = []
    gen_speeds = list(gen_speeds)
    for vel in vel_vecs:
        partials = [Vector(0) for _ in gen_speeds]
        for components, ref in vel.args:
            mat, _ = linear_eq_to_matrix(components, gen_speeds)
            for i in range(len(gen_speeds)):
                for dim, direction in enumerate(ref):
                    if mat[dim, i] != 0:
                        partials[i] += direction * mat[dim, i]

        vec_partials.append(partials)

    return vec_partials


def dynamicsymbols(names, level=0, **assumptions):
    """Uses symbols and Function for functions of time.

    Creates a SymPy UndefinedFunction, which is then initialized as a function
    of a variable, the default being Symbol('t').

    Parameters
    ==========

    names : str
        Names of the dynamic symbols you want to create; works the same way as
        inputs to symbols
    level : int
        Level of differentiation of the returned function; d/dt once of t,
        twice of t, etc.
    assumptions :
        - real(bool) : This is used to set the dynamicsymbol as real,
                    by default is False.
        - positive(bool) : This is used to set the dynamicsymbol as positive,
                    by default is False.
        - commutative(bool) : This is used to set the commutative property of
                    a dynamicsymbol, by default is True.
        - integer(bool) : This is used to set the dynamicsymbol as integer,
                    by default is False.

    Examples
    ========

    >>> from sympy.physics.vector import dynamicsymbols
    >>> from sympy import diff, Symbol
    >>> q1 = dynamicsymbols('q1')
    >>> q1
    q1(t)
    >>> q2 = dynamicsymbols('q2', real=True)
    >>> q2.is_real
    True
    >>> q3 = dynamicsymbols('q3', positive=True)
    >>> q3.is_positive
    True
    >>> q4, q5 = dynamicsymbols('q4,q5', commutative=False)
    >>> bool(q4*q5 != q5*q4)
    True
    >>> q6 = dynamicsymbols('q6', integer=True)
    >>> q6.is_integer
    True
    >>> diff(q1, Symbol('t'))
    Derivative(q1(t), t)

    """
    esses = symbols(names, cls=Function, **assumptions)
    t = dynamicsymbols._t
    if iterable(esses):
        esses = [reduce(diff, [t] * level, e(t)) for e in esses]
        return esses
    else:
        return reduce(diff, [t] * level, esses(t))


dynamicsymbols._t = Symbol('t')  # type: ignore
dynamicsymbols._str = '\''  # type: ignore
<|endoftext|>from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.integrals.integrals import integrate
from sympy.physics.vector import Vector, express
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import _check_vector


__all__ = ['curl', 'divergence', 'gradient', 'is_conservative',
           'is_solenoidal', 'scalar_potential',
           'scalar_potential_difference']


def curl(vect, frame):
    """
    Returns the curl of a vector field computed wrt the coordinate
    symbols of the given frame.

    Parameters
    ==========

    vect : Vector
        The vector operand

    frame : ReferenceFrame
        The reference frame to calculate the curl in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import curl
    >>> R = ReferenceFrame('R')
    >>> v1 = R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z
    >>> curl(v1, R)
    0
    >>> v2 = R[0]*R[1]*R[2]*R.x
    >>> curl(v2, R)
    R_x*R_y*R.y - R_x*R_z*R.z

    """

    _check_vector(vect)
    if vect == 0:
        return Vector(0)
    vect = express(vect, frame, variables=True)
    # A mechanical approach to avoid looping overheads
    vectx = vect.dot(frame.x)
    vecty = vect.dot(frame.y)
    vectz = vect.dot(frame.z)
    outvec = Vector(0)
    outvec += (diff(vectz, frame[1]) - diff(vecty, frame[2])) * frame.x
    outvec += (diff(vectx, frame[2]) - diff(vectz, frame[0])) * frame.y
    outvec += (diff(vecty, frame[0]) - diff(vectx, frame[1])) * frame.z
    return outvec


def divergence(vect, frame):
    """
    Returns the divergence of a vector field computed wrt the coordinate
    symbols of the given frame.

    Parameters
    ==========

    vect : Vector
        The vector operand

    frame : ReferenceFrame
        The reference frame to calculate the divergence in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import divergence
    >>> R = ReferenceFrame('R')
    >>> v1 = R[0]*R[1]*R[2] * (R.x+R.y+R.z)
    >>> divergence(v1, R)
    R_x*R_y + R_x*R_z + R_y*R_z
    >>> v2 = 2*R[1]*R[2]*R.y
    >>> divergence(v2, R)
    2*R_z

    """

    _check_vector(vect)
    if vect == 0:
        return S.Zero
    vect = express(vect, frame, variables=True)
    vectx = vect.dot(frame.x)
    vecty = vect.dot(frame.y)
    vectz = vect.dot(frame.z)
    out = S.Zero
    out += diff(vectx, frame[0])
    out += diff(vecty, frame[1])
    out += diff(vectz, frame[2])
    return out


def gradient(scalar, frame):
    """
    Returns the vector gradient of a scalar field computed wrt the
    coordinate symbols of the given frame.

    Parameters
    ==========

    scalar : sympifiable
        The scalar field to take the gradient of

    frame : ReferenceFrame
        The frame to calculate the gradient in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import gradient
    >>> R = ReferenceFrame('R')
    >>> s1 = R[0]*R[1]*R[2]
    >>> gradient(s1, R)
    R_y*R_z*R.x + R_x*R_z*R.y + R_x*R_y*R.z
    >>> s2 = 5*R[0]**2*R[2]
    >>> gradient(s2, R)
    10*R_x*R_z*R.x + 5*R_x**2*R.z

    """

    _check_frame(frame)
    outvec = Vector(0)
    scalar = express(scalar, frame, variables=True)
    for i, x in enumerate(frame):
        outvec += diff(scalar, frame[i]) * x  # noqa: PLR1736
    return outvec


def is_conservative(field):
    """
    Checks if a field is conservative.

    Parameters
    ==========

    field : Vector
        The field to check for conservative property

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import is_conservative
    >>> R = ReferenceFrame('R')
    >>> is_conservative(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z)
    True
    >>> is_conservative(R[2] * R.y)
    False

    """

    # Field is conservative irrespective of frame
    # Take the first frame in the result of the separate method of Vector
    if field == Vector(0):
        return True
    frame = list(field.separate())[0]
    return curl(field, frame).simplify() == Vector(0)


def is_solenoidal(field):
    """
    Checks if a field is solenoidal.

    Parameters
    ==========

    field : Vector
        The field to check for solenoidal property

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import is_solenoidal
    >>> R = ReferenceFrame('R')
    >>> is_solenoidal(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z)
    True
    >>> is_solenoidal(R[1] * R.y)
    False

    """

    # Field is solenoidal irrespective of frame
    # Take the first frame in the result of the separate method in Vector
    if field == Vector(0):
        return True
    frame = list(field.separate())[0]
    return divergence(field, frame).simplify() is S.Zero


def scalar_potential(field, frame):
    """
    Returns the scalar potential function of a field in a given frame
    (without the added integration constant).

    Parameters
    ==========

    field : Vector
        The vector field whose scalar potential function is to be
        calculated

    frame : ReferenceFrame
        The frame to do the calculation in

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame
    >>> from sympy.physics.vector import scalar_potential, gradient
    >>> R = ReferenceFrame('R')
    >>> scalar_potential(R.z, R) == R[2]
    True
    >>> scalar_field = 2*R[0]**2*R[1]*R[2]
    >>> grad_field = gradient(scalar_field, R)
    >>> scalar_potential(grad_field, R)
    2*R_x**2*R_y*R_z

    """

    # Check whether field is conservative
    if not is_conservative(field):
        raise ValueError("Field is not conservative")
    if field == Vector(0):
        return S.Zero
    # Express the field exntirely in frame
    # Substitute coordinate variables also
    _check_frame(frame)
    field = express(field, frame, variables=True)
    # Make a list of dimensions of the frame
    dimensions = list(frame)
    # Calculate scalar potential function
    temp_function = integrate(field.dot(dimensions[0]), frame[0])
    for i, dim in enumerate(dimensions[1:]):
        partial_diff = diff(temp_function, frame[i + 1])
        partial_diff = field.dot(dim) - partial_diff
        temp_function += integrate(partial_diff, frame[i + 1])
    return temp_function


def scalar_potential_difference(field, frame, point1, point2, origin):
    """
    Returns the scalar potential difference between two points in a
    certain frame, wrt a given field.

    If a scalar field is provided, its values at the two points are
    considered. If a conservative vector field is provided, the values
    of its scalar potential function at the two points are used.

    Returns (potential at position 2) - (potential at position 1)

    Parameters
    ==========

    field : Vector/sympyfiable
        The field to calculate wrt

    frame : ReferenceFrame
        The frame to do the calculations in

    point1 : Point
        The initial Point in given frame

    position2 : Point
        The second Point in the given frame

    origin : Point
        The Point to use as reference point for position vector
        calculation

    Examples
    ========

    >>> from sympy.physics.vector import ReferenceFrame, Point
    >>> from sympy.physics.vector import scalar_potential_difference
    >>> R = ReferenceFrame('R')
    >>> O = Point('O')
    >>> P = O.locatenew('P', R[0]*R.x + R[1]*R.y + R[2]*R.z)
    >>> vectfield = 4*R[0]*R[1]*R.x + 2*R[0]**2*R.y
    >>> scalar_potential_difference(vectfield, R, O, P, O)
    2*R_x**2*R_y
    >>> Q = O.locatenew('O', 3*R.x + R.y + 2*R.z)
    >>> scalar_potential_difference(vectfield, R, P, Q, O)
    -2*R_x**2*R_y + 18

    """

    _check_frame(frame)
    if isinstance(field, Vector):
        # Get the scalar potential function
        scalar_fn = scalar_potential(field, frame)
    else:
        # Field is a scalar
        scalar_fn = field
    # Express positions in required frame
    position1 = express(point1.pos_from(origin), frame, variables=True)
    position2 = express(point2.pos_from(origin), frame, variables=True)
    # Get the two positions as substitution dicts for coordinate variables
    subs_dict1 = {}
    subs_dict2 = {}
    for i, x in enumerate(frame):
        subs_dict1[frame[i]] = x.dot(position1)
        subs_dict2[frame[i]] = x.dot(position2)
    return scalar_fn.subs(subs_dict2) - scalar_fn.subs(subs_dict1)
<|endoftext|>from sympy import sympify, Add, ImmutableMatrix as Matrix
from sympy.core.evalf import EvalfMixin
from sympy.printing.defaults import Printable

from mpmath.libmp.libmpf import prec_to_dps


__all__ = ['Dyadic']


class Dyadic(Printable, EvalfMixin):
    """A Dyadic object.

    See:
    https://en.wikipedia.org/wiki/Dyadic_tensor
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    A more powerful way to represent a rigid body's inertia. While it is more
    complex, by choosing Dyadic components to be in body fixed basis vectors,
    the resulting matrix is equivalent to the inertia tensor.

    """

    is_number = False

    def __init__(self, inlist):
        """
        Just like Vector's init, you should not call this unless creating a
        zero dyadic.

        zd = Dyadic(0)

        Stores a Dyadic as a list of lists; the inner list has the measure
        number and the two unit vectors; the outerlist holds each unique
        unit vector pair.

        """

        self.args = []
        if inlist == 0:
            inlist = []
        while len(inlist) != 0:
            added = 0
            for i, v in enumerate(self.args):
                if ((str(inlist[0][1]) == str(self.args[i][1])) and
                        (str(inlist[0][2]) == str(self.args[i][2]))):
                    self.args[i] = (self.args[i][0] + inlist[0][0],
                                    inlist[0][1], inlist[0][2])
                    inlist.remove(inlist[0])
                    added = 1
                    break
            if added != 1:
                self.args.append(inlist[0])
                inlist.remove(inlist[0])
        i = 0
        # This code is to remove empty parts from the list
        while i < len(self.args):
            if ((self.args[i][0] == 0) | (self.args[i][1] == 0) |
                    (self.args[i][2] == 0)):
                self.args.remove(self.args[i])
                i -= 1
            i += 1

    @property
    def func(self):
        """Returns the class Dyadic. """
        return Dyadic

    def __add__(self, other):
        """The add operator for Dyadic. """
        other = _check_dyadic(other)
        return Dyadic(self.args + other.args)

    __radd__ = __add__

    def __mul__(self, other):
        """Multiplies the Dyadic by a sympifyable expression.

        Parameters
        ==========

        other : Sympafiable
            The scalar to multiply this Dyadic with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> 5 * d
        5*(N.x|N.x)

        """
        newlist = list(self.args)
        other = sympify(other)
        for i in range(len(newlist)):
            newlist[i] = (other * newlist[i][0], newlist[i][1],
                          newlist[i][2])
        return Dyadic(newlist)

    __rmul__ = __mul__

    def dot(self, other):
        """The inner product operator for a Dyadic and a Dyadic or Vector.

        Parameters
        ==========

        other : Dyadic or Vector
            The other Dyadic or Vector to take the inner product with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D1 = outer(N.x, N.y)
        >>> D2 = outer(N.y, N.y)
        >>> D1.dot(D2)
        (N.x|N.y)
        >>> D1.dot(N.y)
        N.x

        """
        from sympy.physics.vector.vector import Vector, _check_vector
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Dyadic(0)
            for v in self.args:
                for v2 in other.args:
                    ol += v[0] * v2[0] * (v[2].dot(v2[1])) * (v[1].outer(v2[2]))
        else:
            other = _check_vector(other)
            ol = Vector(0)
            for v in self.args:
                ol += v[0] * v[1] * (v[2].dot(other))
        return ol

    # NOTE : supports non-advertised Dyadic & Dyadic, Dyadic & Vector notation
    __and__ = dot

    def __truediv__(self, other):
        """Divides the Dyadic by a sympifyable expression. """
        return self.__mul__(1 / other)

    def __eq__(self, other):
        """Tests for equality.

        Is currently weak; needs stronger comparison testing

        """

        if other == 0:
            other = Dyadic(0)
        other = _check_dyadic(other)
        if (self.args == []) and (other.args == []):
            return True
        elif (self.args == []) or (other.args == []):
            return False
        return set(self.args) == set(other.args)

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return self * -1

    def _latex(self, printer):
        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return str(0)
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            # if the coef of the dyadic is 1, we skip the 1
            if v[0] == 1:
                ol.append(' + ' + printer._print(v[1]) + r"\otimes " +
                          printer._print(v[2]))
            # if the coef of the dyadic is -1, we skip the 1
            elif v[0] == -1:
                ol.append(' - ' +
                          printer._print(v[1]) +
                          r"\otimes " +
                          printer._print(v[2]))
            # If the coefficient of the dyadic is not 1 or -1,
            # we might wrap it in parentheses, for readability.
            elif v[0] != 0:
                arg_str = printer._print(v[0])
                if isinstance(v[0], Add):
                    arg_str = '(%s)' % arg_str
                if arg_str.startswith('-'):
                    arg_str = arg_str[1:]
                    str_start = ' - '
                else:
                    str_start = ' + '
                ol.append(str_start + arg_str + printer._print(v[1]) +
                          r"\otimes " + printer._print(v[2]))
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def _pretty(self, printer):
        e = self

        class Fake:
            baseline = 0

            def render(self, *args, **kwargs):
                ar = e.args  # just to shorten things
                mpp = printer
                if len(ar) == 0:
                    return str(0)
                bar = "\N{CIRCLED TIMES}" if printer._use_unicode else "|"
                ol = []  # output list, to be concatenated to a string
                for v in ar:
                    # if the coef of the dyadic is 1, we skip the 1
                    if v[0] == 1:
                        ol.extend([" + ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                    # if the coef of the dyadic is -1, we skip the 1
                    elif v[0] == -1:
                        ol.extend([" - ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                    # If the coefficient of the dyadic is not 1 or -1,
                    # we might wrap it in parentheses, for readability.
                    elif v[0] != 0:
                        if isinstance(v[0], Add):
                            arg_str = mpp._print(
                                v[0]).parens()[0]
                        else:
                            arg_str = mpp.doprint(v[0])
                        if arg_str.startswith("-"):
                            arg_str = arg_str[1:]
                            str_start = " - "
                        else:
                            str_start = " + "
                        ol.extend([str_start, arg_str, " ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                outstr = "".join(ol)
                if outstr.startswith(" + "):
                    outstr = outstr[3:]
                elif outstr.startswith(" "):
                    outstr = outstr[1:]
                return outstr
        return Fake()

    def __rsub__(self, other):
        return (-1 * self) + other

    def _sympystr(self, printer):
        """Printing method. """
        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return printer._print(0)
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            # if the coef of the dyadic is 1, we skip the 1
            if v[0] == 1:
                ol.append(' + (' + printer._print(v[1]) + '|' +
                          printer._print(v[2]) + ')')
            # if the coef of the dyadic is -1, we skip the 1
            elif v[0] == -1:
                ol.append(' - (' + printer._print(v[1]) + '|' +
                          printer._print(v[2]) + ')')
            # If the coefficient of the dyadic is not 1 or -1,
            # we might wrap it in parentheses, for readability.
            elif v[0] != 0:
                arg_str = printer._print(v[0])
                if isinstance(v[0], Add):
                    arg_str = "(%s)" % arg_str
                if arg_str[0] == '-':
                    arg_str = arg_str[1:]
                    str_start = ' - '
                else:
                    str_start = ' + '
                ol.append(str_start + arg_str + '*(' +
                          printer._print(v[1]) +
                          '|' + printer._print(v[2]) + ')')
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def __sub__(self, other):
        """The subtraction operator. """
        return self.__add__(other * -1)

    def cross(self, other):
        """Returns the dyadic resulting from the dyadic vector cross product:
        Dyadic x Vector.

        Parameters
        ==========
        other : Vector
            Vector to cross with.

        Examples
        ========
        >>> from sympy.physics.vector import ReferenceFrame, outer, cross
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> cross(d, N.y)
        (N.x|N.z)

        """
        from sympy.physics.vector.vector import _check_vector
        other = _check_vector(other)
        ol = Dyadic(0)
        for v in self.args:
            ol += v[0] * (v[1].outer((v[2].cross(other))))
        return ol

    # NOTE : supports non-advertised Dyadic ^ Vector notation
    __xor__ = cross

    def express(self, frame1, frame2=None):
        """Expresses this Dyadic in alternate frame(s)

        The first frame is the list side expression, the second frame is the
        right side; if Dyadic is in form A.x|B.y, you can express it in two
        different frames. If no second frame is given, the Dyadic is
        expressed in only one frame.

        Calls the global express function

        Parameters
        ==========

        frame1 : ReferenceFrame
            The frame to express the left side of the Dyadic in
        frame2 : ReferenceFrame
            If provided, the frame to express the right side of the Dyadic in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.express(B, N)
        cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)

        """
        from sympy.physics.vector.functions import express
        return express(self, frame1, frame2)

    def to_matrix(self, reference_frame, second_reference_frame=None):
        """Returns the matrix form of the dyadic with respect to one or two
        reference frames.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows and columns of the matrix
            correspond to. If a second reference frame is provided, this
            only corresponds to the rows of the matrix.
        second_reference_frame : ReferenceFrame, optional, default=None
            The reference frame that the columns of the matrix correspond
            to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,3)
            The matrix that gives the 2D tensor form.

        Examples
        ========

        >>> from sympy import symbols, trigsimp
        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy.physics.mechanics import inertia
        >>> Ixx, Iyy, Izz, Ixy, Iyz, Ixz = symbols('Ixx, Iyy, Izz, Ixy, Iyz, Ixz')
        >>> N = ReferenceFrame('N')
        >>> inertia_dyadic = inertia(N, Ixx, Iyy, Izz, Ixy, Iyz, Ixz)
        >>> inertia_dyadic.to_matrix(N)
        Matrix([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> trigsimp(inertia_dyadic.to_matrix(A))
        Matrix([
        [                           Ixx,                                           Ixy*cos(beta) + Ixz*sin(beta),                                           -Ixy*sin(beta) + Ixz*cos(beta)],
        [ Ixy*cos(beta) + Ixz*sin(beta), Iyy*cos(2*beta)/2 + Iyy/2 + Iyz*sin(2*beta) - Izz*cos(2*beta)/2 + Izz/2,                 -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2],
        [-Ixy*sin(beta) + Ixz*cos(beta),                -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2, -Iyy*cos(2*beta)/2 + Iyy/2 - Iyz*sin(2*beta) + Izz*cos(2*beta)/2 + Izz/2]])

        """

        if second_reference_frame is None:
            second_reference_frame = reference_frame

        return Matrix([i.dot(self).dot(j) for i in reference_frame for j in
                      second_reference_frame]).reshape(3, 3)

    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""
        return sum([Dyadic([(v[0].doit(**hints), v[1], v[2])])
                    for v in self.args], Dyadic(0))

    def dt(self, frame):
        """Take the time derivative of this Dyadic in a frame.

        This function calls the global time_derivative method

        Parameters
        ==========

        frame : ReferenceFrame
            The frame to take the time derivative in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.dt(B)
        - q'*(N.y|N.x) - q'*(N.x|N.y)

        """
        from sympy.physics.vector.functions import time_derivative
        return time_derivative(self, frame)

    def simplify(self):
        """Returns a simplified Dyadic."""
        out = Dyadic(0)
        for v in self.args:
            out += Dyadic([(v[0].simplify(), v[1], v[2])])
        return out

    def subs(self, *args, **kwargs):
        """Substitution on the Dyadic.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = s*(N.x|N.x)
        >>> a.subs({s: 2})
        2*(N.x|N.x)

        """

        return sum([Dyadic([(v[0].subs(*args, **kwargs), v[1], v[2])])
                    for v in self.args], Dyadic(0))

    def applyfunc(self, f):
        """Apply a function to each component of a Dyadic."""
        if not callable(f):
            raise TypeError("`f` must be callable.")

        out = Dyadic(0)
        for a, b, c in self.args:
            out += f(a) * (b.outer(c))
        return out

    def _eval_evalf(self, prec):
        if not self.args:
            return self
        new_args = []
        dps = prec_to_dps(prec)
        for inlist in self.args:
            new_inlist = list(inlist)
            new_inlist[0] = inlist[0].evalf(n=dps)
            new_args.append(tuple(new_inlist))
        return Dyadic(new_args)

    def xreplace(self, rule):
        """
        Replace occurrences of objects within the measure numbers of the
        Dyadic.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Dyadic
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D = outer(N.x, N.x)
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * D).xreplace({x: pi})
        (pi*y + 1)*(N.x|N.x)
        >>> ((1 + x*y) * D).xreplace({x: pi, y: 2})
        (1 + 2*pi)*(N.x|N.x)

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * D).xreplace({x*y: pi})
        (z + pi)*(N.x|N.x)
        >>> ((x*y*z) * D).xreplace({x*y: pi})
        x*y*z*(N.x|N.x)

        """

        new_args = []
        for inlist in self.args:
            new_inlist = list(inlist)
            new_inlist[0] = new_inlist[0].xreplace(rule)
            new_args.append(tuple(new_inlist))
        return Dyadic(new_args)


def _check_dyadic(other):
    if not isinstance(other, Dyadic):
        raise TypeError('A Dyadic must be supplied')
    return other
<|endoftext|>"""
This module can be used to solve 2D beam bending problems with
singularity functions in mechanics.
"""

from sympy.core import S, Symbol, diff, symbols
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.sympify import sympify
from sympy.solvers import linsolve
from sympy.solvers.ode.ode import dsolve
from sympy.solvers.solvers import solve
from sympy.printing import sstr
from sympy.functions import SingularityFunction, Piecewise, factorial
from sympy.integrals import integrate
from sympy.series import limit
from sympy.plotting import plot, PlotGrid
from sympy.geometry.entity import GeometryEntity
from sympy.external import import_module
from sympy.sets.sets import Interval
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import iterable
import warnings


__doctest_requires__ = {
    ('Beam.draw',
     'Beam.plot_bending_moment',
     'Beam.plot_deflection',
     'Beam.plot_ild_moment',
     'Beam.plot_ild_shear',
     'Beam.plot_shear_force',
     'Beam.plot_shear_stress',
     'Beam.plot_slope'): ['matplotlib'],
}


numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})


class Beam:
    """
    A Beam is a structural element that is capable of withstanding load
    primarily by resisting against bending. Beams are characterized by
    their cross sectional profile(Second moment of area), their length
    and their material.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention. However, the
       chosen sign convention must respect the rule that, on the positive
       side of beam's axis (in respect to current section), a loading force
       giving positive shear yields a negative moment, as below (the
       curved arrow shows the positive moment and rotation):

    .. image:: allowed-sign-conventions.png

    Examples
    ========
    There is a beam of length 4 meters. A constant distributed load of 6 N/m
    is applied from half of the beam till the end. There are two simple supports
    below the beam, one at the starting point and another at the ending point
    of the beam. The deflection of the beam at the end is restricted.

    Using the sign convention of downwards forces being positive.

    >>> from sympy.physics.continuum_mechanics.beam import Beam
    >>> from sympy import symbols, Piecewise
    >>> E, I = symbols('E, I')
    >>> R1, R2 = symbols('R1, R2')
    >>> b = Beam(4, E, I)
    >>> b.apply_load(R1, 0, -1)
    >>> b.apply_load(6, 2, 0)
    >>> b.apply_load(R2, 4, -1)
    >>> b.bc_deflection = [(0, 0), (4, 0)]
    >>> b.boundary_conditions
    {'bending_moment': [], 'deflection': [(0, 0), (4, 0)], 'shear_force': [], 'slope': []}
    >>> b.load
    R1*SingularityFunction(x, 0, -1) + R2*SingularityFunction(x, 4, -1) + 6*SingularityFunction(x, 2, 0)
    >>> b.solve_for_reaction_loads(R1, R2)
    >>> b.load
    -3*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 2, 0) - 9*SingularityFunction(x, 4, -1)
    >>> b.shear_force()
    3*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 2, 1) + 9*SingularityFunction(x, 4, 0)
    >>> b.bending_moment()
    3*SingularityFunction(x, 0, 1) - 3*SingularityFunction(x, 2, 2) + 9*SingularityFunction(x, 4, 1)
    >>> b.slope()
    (-3*SingularityFunction(x, 0, 2)/2 + SingularityFunction(x, 2, 3) - 9*SingularityFunction(x, 4, 2)/2 + 7)/(E*I)
    >>> b.deflection()
    (7*x - SingularityFunction(x, 0, 3)/2 + SingularityFunction(x, 2, 4)/4 - 3*SingularityFunction(x, 4, 3)/2)/(E*I)
    >>> b.deflection().rewrite(Piecewise)
    (7*x - Piecewise((x**3, x >= 0), (0, True))/2
         - 3*Piecewise(((x - 4)**3, x >= 4), (0, True))/2
         + Piecewise(((x - 2)**4, x >= 2), (0, True))/4)/(E*I)

    Calculate the support reactions for a fully symbolic beam of length L.
    There are two simple supports below the beam, one at the starting point
    and another at the ending point of the beam. The deflection of the beam
    at the end is restricted. The beam is loaded with:

    * a downward point load P1 applied at L/4
    * an upward point load P2 applied at L/8
    * a counterclockwise moment M1 applied at L/2
    * a clockwise moment M2 applied at 3*L/4
    * a distributed constant load q1, applied downward, starting from L/2
      up to 3*L/4
    * a distributed constant load q2, applied upward, starting from 3*L/4
      up to L

    No assumptions are needed for symbolic loads. However, defining a positive
    length will help the algorithm to compute the solution.

    >>> E, I = symbols('E, I')
    >>> L = symbols("L", positive=True)
    >>> P1, P2, M1, M2, q1, q2 = symbols("P1, P2, M1, M2, q1, q2")
    >>> R1, R2 = symbols('R1, R2')
    >>> b = Beam(L, E, I)
    >>> b.apply_load(R1, 0, -1)
    >>> b.apply_load(R2, L, -1)
    >>> b.apply_load(P1, L/4, -1)
    >>> b.apply_load(-P2, L/8, -1)
    >>> b.apply_load(M1, L/2, -2)
    >>> b.apply_load(-M2, 3*L/4, -2)
    >>> b.apply_load(q1, L/2, 0, 3*L/4)
    >>> b.apply_load(-q2, 3*L/4, 0, L)
    >>> b.bc_deflection = [(0, 0), (L, 0)]
    >>> b.solve_for_reaction_loads(R1, R2)
    >>> print(b.reaction_loads[R1])
    (-3*L**2*q1 + L**2*q2 - 24*L*P1 + 28*L*P2 - 32*M1 + 32*M2)/(32*L)
    >>> print(b.reaction_loads[R2])
    (-5*L**2*q1 + 7*L**2*q2 - 8*L*P1 + 4*L*P2 + 32*M1 - 32*M2)/(32*L)
    """

    def __init__(self, length, elastic_modulus, second_moment, area=Symbol('A'), variable=Symbol('x'), base_char='C', ild_variable=Symbol('a')):
        """Initializes the class.

        Parameters
        ==========

        length : Sympifyable
            A Symbol or value representing the Beam's length.

        elastic_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of Elasticity.
            It is a measure of the stiffness of the Beam material. It can
            also be a continuous function of position along the beam.

        second_moment : Sympifyable or Geometry object
            Describes the cross-section of the beam via a SymPy expression
            representing the Beam's second moment of area. It is a geometrical
            property of an area which reflects how its points are distributed
            with respect to its neutral axis. It can also be a continuous
            function of position along the beam. Alternatively ``second_moment``
            can be a shape object such as a ``Polygon`` from the geometry module
            representing the shape of the cross-section of the beam. In such cases,
            it is assumed that the x-axis of the shape object is aligned with the
            bending axis of the beam. The second moment of area will be computed
            from the shape object internally.

        area : Symbol/float
            Represents the cross-section area of beam

        variable : Symbol, optional
            A Symbol object that will be used as the variable along the beam
            while representing the load, shear, moment, slope and deflection
            curve. By default, it is set to ``Symbol('x')``.

        base_char : String, optional
            A String that will be used as base character to generate sequential
            symbols for integration constants in cases where boundary conditions
            are not sufficient to solve them.

        ild_variable : Symbol, optional
            A Symbol object that will be used as the variable specifying the
            location of the moving load in ILD calculations. By default, it
            is set to ``Symbol('a')``.
        """
        self.length = length
        self.elastic_modulus = elastic_modulus
        if isinstance(second_moment, GeometryEntity):
            self.cross_section = second_moment
        else:
            self.cross_section = None
            self.second_moment = second_moment
        self.variable = variable
        self.ild_variable = ild_variable
        self._base_char = base_char
        self._boundary_conditions = {'deflection': [], 'slope': [], 'bending_moment': [], 'shear_force': []}
        self._load = 0
        self.area = area
        self._applied_supports = []
        self._applied_rotation_hinges = []
        self._applied_sliding_hinges = []
        self._rotation_hinge_symbols = []
        self._sliding_hinge_symbols = []
        self._support_as_loads = []
        self._applied_loads = []
        self._reaction_loads = {}
        self._ild_reactions = {}
        self._ild_shear = 0
        self._ild_moment = 0
        # _original_load is a copy of _load equations with unsubstituted reaction
        # forces. It is used for calculating reaction forces in case of I.L.D.
        self._original_load = 0
        self._joined_beam = False

    def __str__(self):
        shape_description = self._cross_section if self._cross_section else self._second_moment
        str_sol = 'Beam({}, {}, {})'.format(sstr(self._length), sstr(self._elastic_modulus), sstr(shape_description))
        return str_sol

    @property
    def reaction_loads(self):
        """ Returns the reaction forces in a dictionary."""
        return self._reaction_loads

    @property
    def rotation_jumps(self):
        """
        Returns the value for the rotation jumps in rotation hinges in a dictionary.
        The rotation jump is the rotation (in radian) in a rotation hinge. This can
        be seen as a jump in the slope plot.
        """
        return self._rotation_jumps

    @property
    def deflection_jumps(self):
        """
        Returns the deflection jumps in sliding hinges in a dictionary.
        The deflection jump is the deflection (in meters) in a sliding hinge.
        This can be seen as a jump in the deflection plot.
        """
        return self._deflection_jumps

    @property
    def ild_shear(self):
        """ Returns the I.L.D. shear equation."""
        return self._ild_shear

    @property
    def ild_reactions(self):
        """ Returns the I.L.D. reaction forces in a dictionary."""
        return self._ild_reactions

    @property
    def ild_rotation_jumps(self):
        """
        Returns the I.L.D. rotation jumps in rotation hinges in a dictionary.
        The rotation jump is the rotation (in radian) in a rotation hinge. This can
        be seen as a jump in the slope plot.
        """
        return self._ild_rotations_jumps

    @property
    def ild_deflection_jumps(self):
        """
        Returns the I.L.D. deflection jumps in sliding hinges in a dictionary.
        The deflection jump is the deflection (in meters) in a sliding hinge.
        This can be seen as a jump in the deflection plot.
        """
        return self._ild_deflection_jumps

    @property
    def ild_moment(self):
        """ Returns the I.L.D. moment equation."""
        return self._ild_moment

    @property
    def length(self):
        """Length of the Beam."""
        return self._length

    @length.setter
    def length(self, l):
        self._length = sympify(l)

    @property
    def area(self):
        """Cross-sectional area of the Beam. """
        return self._area

    @area.setter
    def area(self, a):
        self._area = sympify(a)

    @property
    def variable(self):
        """
        A symbol that can be used as a variable along the length of the beam
        while representing load distribution, shear force curve, bending
        moment, slope curve and the deflection curve. By default, it is set
        to ``Symbol('x')``, but this property is mutable.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I, A = symbols('E, I, A')
        >>> x, y, z = symbols('x, y, z')
        >>> b = Beam(4, E, I)
        >>> b.variable
        x
        >>> b.variable = y
        >>> b.variable
        y
        >>> b = Beam(4, E, I, A, z)
        >>> b.variable
        z
        """
        return self._variable

    @variable.setter
    def variable(self, v):
        if isinstance(v, Symbol):
            self._variable = v
        else:
            raise TypeError("""The variable should be a Symbol object.""")

    @property
    def elastic_modulus(self):
        """Young's Modulus of the Beam. """
        return self._elastic_modulus

    @elastic_modulus.setter
    def elastic_modulus(self, e):
        self._elastic_modulus = sympify(e)

    @property
    def second_moment(self):
        """Second moment of area of the Beam. """
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        self._cross_section = None
        if isinstance(i, GeometryEntity):
            raise ValueError("To update cross-section geometry use `cross_section` attribute")
        else:
            self._second_moment = sympify(i)

    @property
    def cross_section(self):
        """Cross-section of the beam"""
        return self._cross_section

    @cross_section.setter
    def cross_section(self, s):
        if s:
            self._second_moment = s.second_moment_of_area()[0]
        self._cross_section = s

    @property
    def boundary_conditions(self):
        """
        Returns a dictionary of boundary conditions applied on the beam.
        The dictionary has three keywords namely moment, slope and deflection.
        The value of each keyword is a list of tuple, where each tuple
        contains location and value of a boundary condition in the format
        (location, value).

        Examples
        ========
        There is a beam of length 4 meters. The bending moment at 0 should be 4
        and at 4 it should be 0. The slope of the beam should be 1 at 0. The
        deflection should be 2 at 0.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.bc_deflection = [(0, 2)]
        >>> b.bc_slope = [(0, 1)]
        >>> b.boundary_conditions
        {'bending_moment': [], 'deflection': [(0, 2)], 'shear_force': [], 'slope': [(0, 1)]}

        Here the deflection of the beam should be ``2`` at ``0``.
        Similarly, the slope of the beam should be ``1`` at ``0``.
        """
        return self._boundary_conditions

    @property
    def bc_shear_force(self):
        return self._boundary_conditions['shear_force']

    @bc_shear_force.setter
    def bc_shear_force(self, sf_bcs):
        self._boundary_conditions['shear_force'] = sf_bcs

    @property
    def bc_bending_moment(self):
        return self._boundary_conditions['bending_moment']

    @bc_bending_moment.setter
    def bc_bending_moment(self, bm_bcs):
        self._boundary_conditions['bending_moment'] = bm_bcs

    @property
    def bc_slope(self):
        return self._boundary_conditions['slope']

    @bc_slope.setter
    def bc_slope(self, s_bcs):
        self._boundary_conditions['slope'] = s_bcs

    @property
    def bc_deflection(self):
        return self._boundary_conditions['deflection']

    @bc_deflection.setter
    def bc_deflection(self, d_bcs):
        self._boundary_conditions['deflection'] = d_bcs

    def join(self, beam, via="fixed"):
        """
        This method joins two beams to make a new composite beam system.
        Passed Beam class instance is attached to the right end of calling
        object. This method can be used to form beams having Discontinuous
        values of Elastic modulus or Second moment.

        Parameters
        ==========
        beam : Beam class object
            The Beam object which would be connected to the right of calling
            object.
        via : String
            States the way two Beam object would get connected
            - For axially fixed Beams, via="fixed"
            - For Beams connected via rotation hinge, via="hinge"

        Examples
        ========
        There is a cantilever beam of length 4 meters. For first 2 meters
        its moment of inertia is `1.5*I` and `I` for the other end.
        A pointload of magnitude 4 N is applied from the top at its free end.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b1 = Beam(2, E, 1.5*I)
        >>> b2 = Beam(2, E, I)
        >>> b = b1.join(b2, "fixed")
        >>> b.apply_load(20, 4, -1)
        >>> b.apply_load(R1, 0, -1)
        >>> b.apply_load(R2, 0, -2)
        >>> b.bc_slope = [(0, 0)]
        >>> b.bc_deflection = [(0, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.load
        80*SingularityFunction(x, 0, -2) - 20*SingularityFunction(x, 0, -1) + 20*SingularityFunction(x, 4, -1)
        >>> b.slope()
        (-((-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))/I + 120/I)/E + 80.0/(E*I))*SingularityFunction(x, 2, 0)
        - 0.666666666666667*(-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))*SingularityFunction(x, 0, 0)/(E*I)
        + 0.666666666666667*(-80*SingularityFunction(x, 0, 1) + 10*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 4, 2))*SingularityFunction(x, 2, 0)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        new_length = self.length + beam.length
        if self.elastic_modulus != beam.elastic_modulus:
            raise NotImplementedError('Joining beams with different Elastic modulus is not implemented.')

        if self.second_moment != beam.second_moment:
            new_second_moment = Piecewise((self.second_moment, x<=self.length),
                                    (beam.second_moment, x<=new_length))
        else:
            new_second_moment = self.second_moment

        if via == "fixed":
            new_beam = Beam(new_length, E, new_second_moment, x)
            new_beam._joined_beam = True
            return new_beam

        if via == "hinge":
            new_beam = Beam(new_length, E, new_second_moment, x)
            new_beam._joined_beam = True
            new_beam.apply_rotation_hinge(self.length)
            return new_beam

    def apply_support(self, loc, type="fixed"):
        """
        This method applies support to a particular beam object and returns
        the symbol of the unknown reaction load(s).

        Parameters
        ==========
        loc : Sympifyable
            Location of point at which support is applied.
        type : String
            Determines type of Beam support applied. To apply support structure
            with
            - zero degree of freedom, type = "fixed"
            - one degree of freedom, type = "pin"
            - two degrees of freedom, type = "roller"

        Returns
        =======
        Symbol or tuple of Symbol
            The unknown reaction load as a symbol.
            - Symbol(reaction_force) if type = "pin" or "roller"
            - Symbol(reaction_force), Symbol(reaction_moment) if type = "fixed"

        Examples
        ========
        There is a beam of length 20 meters. A moment of magnitude 100 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at a distance of 10 meters.
        There is one fixed support at the start of the beam and a roller at the end.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(20, E, I)
        >>> p0, m0 = b.apply_support(0, 'fixed')
        >>> p1 = b.apply_support(20, 'roller')
        >>> b.apply_load(-8, 10, -1)
        >>> b.apply_load(100, 20, -2)
        >>> b.solve_for_reaction_loads(p0, m0, p1)
        >>> b.reaction_loads
        {M_0: 20, R_0: -2, R_20: 10}
        >>> b.reaction_loads[p0]
        -2
        >>> b.load
        20*SingularityFunction(x, 0, -2) - 2*SingularityFunction(x, 0, -1)
        - 8*SingularityFunction(x, 10, -1) + 100*SingularityFunction(x, 20, -2)
        + 10*SingularityFunction(x, 20, -1)
        """
        loc = sympify(loc)

        self._applied_supports.append((loc, type))
        if type in ("pin", "roller"):
            reaction_load = Symbol('R_'+str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.bc_deflection.append((loc, 0))
        else:
            reaction_load = Symbol('R_'+str(loc))
            reaction_moment = Symbol('M_'+str(loc))
            self.apply_load(reaction_load, loc, -1)
            self.apply_load(reaction_moment, loc, -2)
            self.bc_deflection.append((loc, 0))
            self.bc_slope.append((loc, 0))
            self._support_as_loads.append((reaction_moment, loc, -2, None))

        self._support_as_loads.append((reaction_load, loc, -1, None))

        if type in ("pin", "roller"):
            return reaction_load
        else:
            return reaction_load, reaction_moment

    def _get_I(self, loc):
        """
        Helper function that returns the Second moment (I) at a location in the beam.
        """
        I = self.second_moment
        if not isinstance(I, Piecewise):
            return I
        else:
            for i in range(len(I.args)):
                if loc <= I.args[i][1].args[1]:
                    return I.args[i][0]

    def apply_rotation_hinge(self, loc):
        """
        This method applies a rotation hinge at a single location on the beam.

        Parameters
        ----------
        loc : Sympifyable
            Location of point at which hinge is applied.

        Returns
        =======
        Symbol
            The unknown rotation jump multiplied by the elastic modulus and second moment as a symbol.

        Examples
        ========
        There is a beam of length 15 meters. Pin supports are placed at distances
        of 0 and 10 meters. There is a fixed support at the end. There are two rotation hinges
        in the structure, one at 5 meters and one at 10 meters. A pointload of magnitude
        10 kN is applied on the hinge at 5 meters. A distributed load of 5 kN works on
        the structure from 10 meters to the end.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import Symbol
        >>> E = Symbol('E')
        >>> I = Symbol('I')
        >>> b = Beam(15, E, I)
        >>> r0 = b.apply_support(0, type='pin')
        >>> r10 = b.apply_support(10, type='pin')
        >>> r15, m15 = b.apply_support(15, type='fixed')
        >>> p5 = b.apply_rotation_hinge(5)
        >>> p12 = b.apply_rotation_hinge(12)
        >>> b.apply_load(-10, 5, -1)
        >>> b.apply_load(-5, 10, 0, 15)
        >>> b.solve_for_reaction_loads(r0, r10, r15, m15)
        >>> b.reaction_loads
        {M_15: -75/2, R_0: 0, R_10: 40, R_15: -5}
        >>> b.rotation_jumps
        {P_12: -1875/(16*E*I), P_5: 9625/(24*E*I)}
        >>> b.rotation_jumps[p12]
        -1875/(16*E*I)
        >>> b.bending_moment()
        -9625*SingularityFunction(x, 5, -1)/24 + 10*SingularityFunction(x, 5, 1)
        - 40*SingularityFunction(x, 10, 1) + 5*SingularityFunction(x, 10, 2)/2
        + 1875*SingularityFunction(x, 12, -1)/16 + 75*SingularityFunction(x, 15, 0)/2
        + 5*SingularityFunction(x, 15, 1) - 5*SingularityFunction(x, 15, 2)/2
        """
        loc = sympify(loc)
        E = self.elastic_modulus
        I = self._get_I(loc)

        rotation_jump = Symbol('P_'+str(loc))
        self._applied_rotation_hinges.append(loc)
        self._rotation_hinge_symbols.append(rotation_jump)
        self.apply_load(E * I * rotation_jump, loc, -3)
        self.bc_bending_moment.append((loc, 0))
        return rotation_jump

    def apply_sliding_hinge(self, loc):
        """
        This method applies a sliding hinge at a single location on the beam.

        Parameters
        ----------
        loc : Sympifyable
            Location of point at which hinge is applied.

        Returns
        =======
        Symbol
            The unknown deflection jump multiplied by the elastic modulus and second moment as a symbol.

        Examples
        ========
        There is a beam of length 13 meters. A fixed support is placed at the beginning.
        There is a pin support at the end. There is a sliding hinge at a location of 8 meters.
        A pointload of magnitude 10 kN is applied on the hinge at 5 meters.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> b = Beam(13, 20, 20)
        >>> r0, m0 = b.apply_support(0, type="fixed")
        >>> s8 = b.apply_sliding_hinge(8)
        >>> r13 = b.apply_support(13, type="pin")
        >>> b.apply_load(-10, 5, -1)
        >>> b.solve_for_reaction_loads(r0, m0, r13)
        >>> b.reaction_loads
        {M_0: -50, R_0: 10, R_13: 0}
        >>> b.deflection_jumps
        {W_8: 85/24}
        >>> b.deflection_jumps[s8]
        85/24
        >>> b.bending_moment()
        50*SingularityFunction(x, 0, 0) - 10*SingularityFunction(x, 0, 1)
        + 10*SingularityFunction(x, 5, 1) - 4250*SingularityFunction(x, 8, -2)/3
        >>> b.deflection()
        -SingularityFunction(x, 0, 2)/16 + SingularityFunction(x, 0, 3)/240
        - SingularityFunction(x, 5, 3)/240 + 85*SingularityFunction(x, 8, 0)/24
        """
        loc = sympify(loc)
        E = self.elastic_modulus
        I = self._get_I(loc)

        deflection_jump = Symbol('W_' + str(loc))
        self._applied_sliding_hinges.append(loc)
        self._sliding_hinge_symbols.append(deflection_jump)
        self.apply_load(E * I * deflection_jump, loc, -4)
        self.bc_shear_force.append((loc, 0))
        return deflection_jump

    def apply_load(self, value, start, order, end=None):
        """
        This method adds up the loads given to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The value inserted should have the units [Force/(Distance**(n+1)]
            where n is the order of applied load.
            Units for applied loads:

               - For moments, unit = kN*m
               - For point loads, unit = kN
               - For constant distributed load, unit = kN/m
               - For ramp loads, unit = kN/m/m
               - For parabolic ramp loads, unit = kN/m/m/m
               - ... so on.

        start : Sympifyable
            The starting point of the applied load. For point moments and
            point forces this is the location of application.
        order : Integer
            The order of the applied load.

               - For moments, order = -2
               - For point loads, order =-1
               - For constant distributed load, order = 0
               - For ramp loads, order = 1
               - For parabolic ramp loads, order = 2
               - ... so on.

        end : Sympifyable, optional
            An optional argument that can be used if the load has an end point
            within the length of the beam.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A point load of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 2 meters to 3 meters
        away from the starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 2, 2, end=3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)

        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        self._applied_loads.append((value, start, order, end))
        self._load += value*SingularityFunction(x, start, order)
        self._original_load += value*SingularityFunction(x, start, order)

        if end:
            # load has an end point within the length of the beam.
            self._handle_end(x, value, start, order, end, type="apply")

    def remove_load(self, value, start, order, end=None):
        """
        This method removes a particular load present on the beam object.
        Returns a ValueError if the load passed as an argument is not
        present on the beam.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        start : Sympifyable
            The starting point of the applied load. For point moments and
            point forces this is the location of application.
        order : Integer
            The order of the applied load.
            - For moments, order= -2
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        end : Sympifyable, optional
            An optional argument that can be used if the load has an end point
            within the length of the beam.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A pointload of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 2 meters to 3 meters
        away from the starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 2, 2, end=3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)
        >>> b.remove_load(-2, 2, 2, end = 3)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1)
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if (value, start, order, end) in self._applied_loads:
            self._load -= value*SingularityFunction(x, start, order)
            self._original_load -= value*SingularityFunction(x, start, order)
            self._applied_loads.remove((value, start, order, end))
        else:
            msg = "No such load distribution exists on the beam object."
            raise ValueError(msg)

        if end:
            # load has an end point within the length of the beam.
            self._handle_end(x, value, start, order, end, type="remove")

    def _handle_end(self, x, value, start, order, end, type):
        """
        This functions handles the optional `end` value in the
        `apply_load` and `remove_load` functions. When the value
        of end is not NULL, this function will be executed.
        """
        if order.is_negative:
            msg = ("If 'end' is provided the 'order' of the load cannot "
                    "be negative, i.e. 'end' is only valid for distributed "
                    "loads.")
            raise ValueError(msg)
        # NOTE : A Taylor series can be used to define the summation of
        # singularity functions that subtract from the load past the end
        # point such that it evaluates to zero past 'end'.
        f = value*x**order

        if type == "apply":
            # iterating for "apply_load" method
            for i in range(0, order + 1):
                self._load -= (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
                self._original_load -= (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
        elif type == "remove":
            # iterating for "remove_load" method
            for i in range(0, order + 1):
                self._load += (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))
                self._original_load += (f.diff(x, i).subs(x, end - start) *
                                SingularityFunction(x, end, i)/factorial(i))


    @property
    def load(self):
        """
        Returns a Singularity Function expression which represents
        the load distribution curve of the Beam object.

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A point load of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point and a parabolic ramp load of magnitude
        2 N/m is applied below the beam starting from 3 meters away from the
        starting point of the beam.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(-2, 3, 2)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1) - 2*SingularityFunction(x, 3, 2)
        """
        return self._load

    @property
    def applied_loads(self):
        """
        Returns a list of all loads applied on the beam object.
        Each load in the list is a tuple of form (value, start, order, end).

        Examples
        ========
        There is a beam of length 4 meters. A moment of magnitude 3 Nm is
        applied in the clockwise direction at the starting point of the beam.
        A pointload of magnitude 4 N is applied from the top of the beam at
        2 meters from the starting point. Another pointload of magnitude 5 N
        is applied at same position.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(4, E, I)
        >>> b.apply_load(-3, 0, -2)
        >>> b.apply_load(4, 2, -1)
        >>> b.apply_load(5, 2, -1)
        >>> b.load
        -3*SingularityFunction(x, 0, -2) + 9*SingularityFunction(x, 2, -1)
        >>> b.applied_loads
        [(-3, 0, -2, None), (4, 2, -1, None), (5, 2, -1, None)]
        """
        return self._applied_loads

    def solve_for_reaction_loads(self, *reactions):
        """
        Solves for the reaction forces.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)  # Reaction force at x = 10
        >>> b.apply_load(R2, 30, -1)  # Reaction force at x = 30
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.load
        R1*SingularityFunction(x, 10, -1) + R2*SingularityFunction(x, 30, -1)
            - 8*SingularityFunction(x, 0, -1) + 120*SingularityFunction(x, 30, -2)
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.reaction_loads
        {R1: 6, R2: 2}
        >>> b.load
        -8*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 10, -1)
            + 120*SingularityFunction(x, 30, -2) + 2*SingularityFunction(x, 30, -1)
        """

        x = self.variable
        l = self.length
        C3 = Symbol('C3')
        C4 = Symbol('C4')
        rotation_jumps = tuple(self._rotation_hinge_symbols)
        deflection_jumps = tuple(self._sliding_hinge_symbols)

        shear_curve = limit(self.shear_force(), x, l)
        moment_curve = limit(self.bending_moment(), x, l)

        shear_force_eqs = []
        bending_moment_eqs = []
        slope_eqs = []
        deflection_eqs = []

        for position, value in self._boundary_conditions['shear_force']:
            eqs = self.shear_force().subs(x, position) - value
            new_eqs = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            shear_force_eqs.append(new_eqs)

        for position, value in self._boundary_conditions['bending_moment']:
            eqs = self.bending_moment().subs(x, position) - value
            new_eqs = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            bending_moment_eqs.append(new_eqs)

        slope_curve = integrate(self.bending_moment(), x) + C3
        for position, value in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            slope_eqs.append(eqs)

        deflection_curve = integrate(slope_curve, x) + C4
        for position, value in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            deflection_eqs.append(eqs)

        solution = list((linsolve([shear_curve, moment_curve] + shear_force_eqs + bending_moment_eqs + slope_eqs
                            + deflection_eqs, (C3, C4) + reactions + rotation_jumps + deflection_jumps).args)[0])
        reaction_index = 2+len(reactions)
        rotation_index = reaction_index + len(rotation_jumps)
        reaction_solution = solution[2:reaction_index]
        rotation_solution = solution[reaction_index:rotation_index]
        deflection_solution = solution[rotation_index:]

        self._reaction_loads = dict(zip(reactions, reaction_solution))
        self._rotation_jumps = dict(zip(rotation_jumps, rotation_solution))
        self._deflection_jumps = dict(zip(deflection_jumps, deflection_solution))
        self._load = self._load.subs(self._reaction_loads)
        self._load = self._load.subs(self._rotation_jumps)
        self._load = self._load.subs(self._deflection_jumps)

    def shear_force(self):
        """
        Returns a Singularity Function expression which represents
        the shear force curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.shear_force()
        8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) - 120*SingularityFunction(x, 30, -1) - 2*SingularityFunction(x, 30, 0)
        """
        x = self.variable
        return -integrate(self.load, x)

    def max_shear_force(self):
        """Returns maximum Shear force and its coordinate
        in the Beam object."""
        shear_curve = self.shear_force()
        x = self.variable

        terms = shear_curve.args
        singularity = []        # Points at which shear function changes
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]    # SingularityFunction in the term
            singularity.append(term.args[1])
        singularity = list(set(singularity))
        singularity.sort()

        intervals = []    # List of Intervals with discrete value of shear force
        shear_values = []   # List of values of shear force in each interval
        for i, s in enumerate(singularity):
            if s == 0:
                continue
            try:
                shear_slope = Piecewise((float("nan"), x<=singularity[i-1]),(self._load.rewrite(Piecewise), x<s), (float("nan"), True))
                points = solve(shear_slope, x)
                val = []
                for point in points:
                    val.append(abs(shear_curve.subs(x, point)))
                points.extend([singularity[i-1], s])
                val += [abs(limit(shear_curve, x, singularity[i-1], '+')), abs(limit(shear_curve, x, s, '-'))]
                max_shear = max(val)
                shear_values.append(max_shear)
                intervals.append(points[val.index(max_shear)])
            # If shear force in a particular Interval has zero or constant
            # slope, then above block gives NotImplementedError as
            # solve can't represent Interval solutions.
            except NotImplementedError:
                initial_shear = limit(shear_curve, x, singularity[i-1], '+')
                final_shear = limit(shear_curve, x, s, '-')
                # If shear_curve has a constant slope(it is a line).
                if shear_curve.subs(x, (singularity[i-1] + s)/2) == (initial_shear + final_shear)/2 and initial_shear != final_shear:
                    shear_values.extend([initial_shear, final_shear])
                    intervals.extend([singularity[i-1], s])
                else:    # shear_curve has same value in whole Interval
                    shear_values.append(final_shear)
                    intervals.append(Interval(singularity[i-1], s))

        shear_values = list(map(abs, shear_values))
        maximum_shear = max(shear_values)
        point = intervals[shear_values.index(maximum_shear)]
        return (point, maximum_shear)

    def bending_moment(self):
        """
        Returns a Singularity Function expression which represents
        the bending moment curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.bending_moment()
        8*SingularityFunction(x, 0, 1) - 6*SingularityFunction(x, 10, 1) - 120*SingularityFunction(x, 30, 0) - 2*SingularityFunction(x, 30, 1)
        """
        x = self.variable
        return integrate(self.shear_force(), x)

    def max_bmoment(self):
        """Returns maximum Shear force and its coordinate
        in the Beam object."""
        bending_curve = self.bending_moment()
        x = self.variable

        terms = bending_curve.args
        singularity = []        # Points at which bending moment changes
        for term in terms:
            if isinstance(term, Mul):
                term = term.args[-1]    # SingularityFunction in the term
            singularity.append(term.args[1])
        singularity = list(set(singularity))
        singularity.sort()

        intervals = []    # List of Intervals with discrete value of bending moment
        moment_values = []   # List of values of bending moment in each interval
        for i, s in enumerate(singularity):
            if s == 0:
                continue
            try:
                moment_slope = Piecewise(
                    (float("nan"), x <= singularity[i - 1]),
                    (self.shear_force().rewrite(Piecewise), x < s),
                    (float("nan"), True))
                points = solve(moment_slope, x)
                val = []
                for point in points:
                    val.append(abs(bending_curve.subs(x, point)))
                points.extend([singularity[i-1], s])
                val += [abs(limit(bending_curve, x, singularity[i-1], '+')), abs(limit(bending_curve, x, s, '-'))]
                max_moment = max(val)
                moment_values.append(max_moment)
                intervals.append(points[val.index(max_moment)])

            # If bending moment in a particular Interval has zero or constant
            # slope, then above block gives NotImplementedError as solve
            # can't represent Interval solutions.
            except NotImplementedError:
                initial_moment = limit(bending_curve, x, singularity[i-1], '+')
                final_moment = limit(bending_curve, x, s, '-')
                # If bending_curve has a constant slope(it is a line).
                if bending_curve.subs(x, (singularity[i-1] + s)/2) == (initial_moment + final_moment)/2 and initial_moment != final_moment:
                    moment_values.extend([initial_moment, final_moment])
                    intervals.extend([singularity[i-1], s])
                else:    # bending_curve has same value in whole Interval
                    moment_values.append(final_moment)
                    intervals.append(Interval(singularity[i-1], s))

        moment_values = list(map(abs, moment_values))
        maximum_moment = max(moment_values)
        point = intervals[moment_values.index(maximum_moment)]
        return (point, maximum_moment)

    def point_cflexure(self):
        """
        Returns a Set of point(s) with zero bending moment and
        where bending moment curve of the beam object changes
        its sign from negative to positive or vice versa.

        Examples
        ========
        There is is 10 meter long overhanging beam. There are
        two simple supports below the beam. One at the start
        and another one at a distance of 6 meters from the start.
        Point loads of magnitude 10KN and 20KN are applied at
        2 meters and 4 meters from start respectively. A Uniformly
        distribute load of magnitude of magnitude 3KN/m is also
        applied on top starting from 6 meters away from starting
        point till end.
        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> b = Beam(10, E, I)
        >>> b.apply_load(-4, 0, -1)
        >>> b.apply_load(-46, 6, -1)
        >>> b.apply_load(10, 2, -1)
        >>> b.apply_load(20, 4, -1)
        >>> b.apply_load(3, 6, 0)
        >>> b.point_cflexure()
        [10/3]
        """
        #Removes the singularity functions of order < 0 from the bending moment equation used in this method
        non_singular_bending_moment = sum(arg for arg in self.bending_moment().args if not arg.args[1].args[2] < 0)

        # To restrict the range within length of the Beam
        moment_curve = Piecewise((float("nan"), self.variable<=0),
                (non_singular_bending_moment, self.variable<self.length),
                (float("nan"), True))
        try:
            points = solve(moment_curve.rewrite(Piecewise), self.variable,
                           domain=S.Reals)
        except NotImplementedError as e:
            if "An expression is already zero when" in str(e):
                raise NotImplementedError("This method cannot be used when a whole region of "
                                          "the bending moment line is equal to 0.")
            else:
                raise

        return points

    def slope(self):
        """
        Returns a Singularity Function expression which represents
        the slope the elastic curve of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.slope()
        (-4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2)
            + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + 4000/3)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        I = self.second_moment

        if not self._boundary_conditions['slope']:
            return diff(self.deflection(), x)
        if isinstance(I, Piecewise) and self._joined_beam:
            args = I.args
            slope = 0
            prev_slope = 0
            prev_end = 0
            for i in range(len(args)):
                if i != 0:
                    prev_end = args[i-1][1].args[1]
                slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                if i != len(args) - 1:
                    slope += (prev_slope + slope_value)*SingularityFunction(x, prev_end, 0) - \
                        (prev_slope + slope_value)*SingularityFunction(x, args[i][1].args[1], 0)
                else:
                    slope += (prev_slope + slope_value)*SingularityFunction(x, prev_end, 0)
                prev_slope = slope_value.subs(x, args[i][1].args[1])
            return slope

        C3 = Symbol('C3')
        slope_curve = -integrate(S.One/(E*I)*self.bending_moment(), x) + C3

        bc_eqs = []
        for position, value in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - value
            bc_eqs.append(eqs)
        constants = list(linsolve(bc_eqs, C3))
        slope_curve = slope_curve.subs({C3: constants[0][0]})
        return slope_curve

    def deflection(self):
        """
        Returns a Singularity Function expression which represents
        the elastic curve or deflection of the Beam object.

        Examples
        ========
        There is a beam of length 30 meters. A moment of magnitude 120 Nm is
        applied in the clockwise direction at the end of the beam. A pointload
        of magnitude 8 N is applied from the top of the beam at the starting
        point. There are two simple supports below the beam. One at the end
        and another one at a distance of 10 meters from the start. The
        deflection is restricted at both the supports.

        Using the sign convention of upward forces and clockwise moment
        being positive.

        >>> from sympy.physics.continuum_mechanics.beam import Beam
        >>> from sympy import symbols
        >>> E, I = symbols('E, I')
        >>> R1, R2 = symbols('R1, R2')
        >>> b = Beam(30, E, I)
        >>> b.apply_load(-8, 0, -1)
        >>> b.apply_load(R1, 10, -1)
        >>> b.apply_load(R2, 30, -1)
        >>> b.apply_load(120, 30, -2)
        >>> b.bc_deflection = [(10, 0), (30, 0)]
        >>> b.solve_for_reaction_loads(R1, R2)
        >>> b.deflection()
        (4000*x/3 - 4*SingularityFunction(x, 0, 3)/3 + SingularityFunction(x, 10, 3)
            + 60*SingularityFunction(x, 30, 2) + SingularityFunction(x, 30, 3)/3 - 12000)/(E*I)
        """
        x = self.variable
        E = self.elastic_modulus
        I = self.second_moment
        if not self._boundary_conditions['deflection'] and not self._boundary_conditions['slope']:
            if isinstance(I, Piecewise) and self._joined_beam:
                args = I.args
                prev_slope = 0
                prev_def = 0
                prev_end = 0
                deflection = 0
                for i in range(len(args)):
                    if i != 0:
                        prev_end = args[i-1][1].args[1]
                    slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                    recent_segment_slope = prev_slope + slope_value
                    deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                    if i != len(args) - 1:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                            - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                    else:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                    prev_slope = slope_value.subs(x, args[i][1].args[1])
                    prev_def = deflection_value.subs(x, args[i][1].args[1])
                return deflection
            base_char = self._base_char
            constants = symbols(base_char + '3:5')
            return S.One/(E*I)*integrate(-integrate(self.bending_moment(), x), x) + constants[0]*x + constants[1]
        elif not self._boundary_conditions['deflection']:
            base_char = self._base_char
            constant = symbols(base_char + '4')
            return integrate(self.slope(), x) + constant
        elif not self._boundary_conditions['slope'] and self._boundary_conditions['deflection']:
            if isinstance(I, Piecewise) and self._joined_beam:
                args = I.args
                prev_slope = 0
                prev_def = 0
                prev_end = 0
                deflection = 0
                for i in range(len(args)):
                    if i != 0:
                        prev_end = args[i-1][1].args[1]
                    slope_value = -S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                    recent_segment_slope = prev_slope + slope_value
                    deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                    if i != len(args) - 1:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                            - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                    else:
                        deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                    prev_slope = slope_value.subs(x, args[i][1].args[1])
                    prev_def = deflection_value.subs(x, args[i][1].args[1])
                return deflection
            base_char = self._base_char
            C3, C4 = symbols(base_char + '3:5')    # Integration constants
            slope_curve = -integrate(self.bending_moment(), x) + C3
            deflection_curve = integrate(slope_curve, x) + C4
            bc_eqs = []
            for position, value in self._boundary_conditions['deflection']:
                eqs = deflection_curve.subs(x, position) - value
                bc_eqs.append(eqs)
            constants = list(linsolve(bc_eqs, (C3, C4)))
            deflection_curve = deflection_curve.subs({C3: constants[0][0], C4: constants[0][1]})
            return S.One/(E*I)*deflection_curve

        if isinstance(I, Piecewise) and self._joined_beam:
            args = I.args
            prev_slope = 0
            prev_def = 0
            prev_end = 0
            deflection = 0
            for i in range(len(args)):
                if i != 0:
                    prev_end = args[i-1][1].args[1]
                slope_value = S.One/E*integrate(self.bending_moment()/args[i][0], (x, prev_end, x))
                recent_segment_slope = prev_slope + slope_value
                deflection_value = integrate(recent_segment_slope, (x, prev_end, x))
                if i != len(args) - 1:
                    deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0) \
                        - (prev_def + deflection_value)*SingularityFunction(x, args[i][1].args[1], 0)
                else:
                    deflection += (prev_def + deflection_value)*SingularityFunction(x, prev_end, 0)
                prev_slope = slope_value.subs(x, args[i][1].args[1])
                prev_def = deflection_value.subs(x, args[i][1].args[1])
            return deflection

        C4 = Symbol('C4')
        deflection_curve = integrate(self.slope(), x) + C4

        bc_eqs = []
        for position, value in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - value
            bc_eqs.append(eqs)

        constants = list(linsolve(bc_eqs, C4))
        deflection_curve = deflection_curve.subs({C4: constants[0][0]})
        return deflection_curve

    def max_deflection(self):
        """
        Returns point of max deflection and its corresponding deflection value
        in a Beam object.
        """

        # To restrict the range within length of the Beam
        slope_curve = Piecewise((float("nan"), self.variable<=0),
                (self.slope(), self.variable<self.length),
                (float("nan"), True))

        points = solve(slope_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        deflection_curve = self.deflection()
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))
        if len(deflections) != 0:
            max_def = max(deflections)
            return (points[deflections.index(max_def)], max_def)
        else:
            return None

    def shear_stress(self):
        """
        Returns an expression representing the Shear Stress
        curve of the Beam object.
        """
        return self.shear_force()/self._area

    def plot_shear_stress(self, subs=None):
        """

        Returns a plot of shear stress present in the beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters and area of cross section 2 square
        meters. A constant distributed load of 10 KN/m is applied from half of
        the beam till the end. There are two simple supports below the beam,
        one at the starting point and another at the ending point of the beam.
        A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6), 2)
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_shear_stress()
            Plot object containing:
            [0]: cartesian line: 6875*SingularityFunction(x, 0, 0) - 2500*SingularityFunction(x, 2, 0)
            - 5000*SingularityFunction(x, 4, 1) + 15625*SingularityFunction(x, 8, 0)
            + 5000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)
        """

        shear_stress = self.shear_stress()
        x = self.variable
        length = self.length

        if subs is None:
            subs = {}
        for sym in shear_stress.atoms(Symbol):
            if sym != x and sym not in subs:
                raise ValueError('value of %s was not passed.' %sym)

        if length in subs:
            length = subs[length]

        # Returns Plot of Shear Stress
        return plot (shear_stress.subs(subs), (x, 0, length),
        title='Shear Stress', xlabel=r'$\mathrm{x}$', ylabel=r'$\tau$',
        line_color='r')


    def plot_shear_force(self, subs=None):
        """

        Returns a plot for Shear force present in the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_shear_force()
            Plot object containing:
            [0]: cartesian line: 13750*SingularityFunction(x, 0, 0) - 5000*SingularityFunction(x, 2, 0)
            - 10000*SingularityFunction(x, 4, 1) + 31250*SingularityFunction(x, 8, 0)
            + 10000*SingularityFunction(x, 8, 1) for x over (0.0, 8.0)
        """
        shear_force = self.shear_force()
        if subs is None:
            subs = {}
        for sym in shear_force.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(shear_force.subs(subs), (self.variable, 0, length), title='Shear Force',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{V}$', line_color='g')

    def plot_bending_moment(self, subs=None):
        """

        Returns a plot for Bending moment present in the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_bending_moment()
            Plot object containing:
            [0]: cartesian line: 13750*SingularityFunction(x, 0, 1) - 5000*SingularityFunction(x, 2, 1)
            - 5000*SingularityFunction(x, 4, 2) + 31250*SingularityFunction(x, 8, 1)
            + 5000*SingularityFunction(x, 8, 2) for x over (0.0, 8.0)
        """
        bending_moment = self.bending_moment()
        if subs is None:
            subs = {}
        for sym in bending_moment.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(bending_moment.subs(subs), (self.variable, 0, length), title='Bending Moment',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{M}$', line_color='b')

    def plot_slope(self, subs=None):
        """

        Returns a plot for slope of deflection curve of the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_slope()
            Plot object containing:
            [0]: cartesian line: -8.59375e-5*SingularityFunction(x, 0, 2) + 3.125e-5*SingularityFunction(x, 2, 2)
            + 2.08333333333333e-5*SingularityFunction(x, 4, 3) - 0.0001953125*SingularityFunction(x, 8, 2)
            - 2.08333333333333e-5*SingularityFunction(x, 8, 3) + 0.00138541666666667 for x over (0.0, 8.0)
        """
        slope = self.slope()
        if subs is None:
            subs = {}
        for sym in slope.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(slope.subs(subs), (self.variable, 0, length), title='Slope',
                xlabel=r'$\mathrm{x}$', ylabel=r'$\theta$', line_color='m')

    def plot_deflection(self, subs=None):
        """

        Returns a plot for deflection curve of the Beam object.

        Parameters
        ==========
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> b.plot_deflection()
            Plot object containing:
            [0]: cartesian line: 0.00138541666666667*x - 2.86458333333333e-5*SingularityFunction(x, 0, 3)
            + 1.04166666666667e-5*SingularityFunction(x, 2, 3) + 5.20833333333333e-6*SingularityFunction(x, 4, 4)
            - 6.51041666666667e-5*SingularityFunction(x, 8, 3) - 5.20833333333333e-6*SingularityFunction(x, 8, 4)
            for x over (0.0, 8.0)
        """
        deflection = self.deflection()
        if subs is None:
            subs = {}
        for sym in deflection.atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length
        return plot(deflection.subs(subs), (self.variable, 0, length),
                    title='Deflection', xlabel=r'$\mathrm{x}$', ylabel=r'$\delta$',
                    line_color='r')


    def plot_loading_results(self, subs=None):
        """
        Returns a subplot of Shear Force, Bending Moment,
        Slope and Deflection of the Beam object.

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========

        There is a beam of length 8 meters. A constant distributed load of 10 KN/m
        is applied from half of the beam till the end. There are two simple supports
        below the beam, one at the starting point and another at the ending point
        of the beam. A pointload of magnitude 5 KN is also applied from top of the
        beam, at a distance of 4 meters from the starting point.
        Take E = 200 GPa and I = 400*(10**-6) meter**4.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> R1, R2 = symbols('R1, R2')
            >>> b = Beam(8, 200*(10**9), 400*(10**-6))
            >>> b.apply_load(5000, 2, -1)
            >>> b.apply_load(R1, 0, -1)
            >>> b.apply_load(R2, 8, -1)
            >>> b.apply_load(10000, 4, 0, end=8)
            >>> b.bc_deflection = [(0, 0), (8, 0)]
            >>> b.solve_for_reaction_loads(R1, R2)
            >>> axes = b.plot_loading_results()
        """
        length = self.length
        variable = self.variable
        if subs is None:
            subs = {}
        for sym in self.deflection().atoms(Symbol):
            if sym == self.variable:
                continue
            if sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if length in subs:
            length = subs[length]
        ax1 = plot(self.shear_force().subs(subs), (variable, 0, length),
                   title="Shear Force", xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{V}$',
                   line_color='g', show=False)
        ax2 = plot(self.bending_moment().subs(subs), (variable, 0, length),
                   title="Bending Moment", xlabel=r'$\mathrm{x}$', ylabel=r'$\mathrm{M}$',
                   line_color='b', show=False)
        ax3 = plot(self.slope().subs(subs), (variable, 0, length),
                   title="Slope", xlabel=r'$\mathrm{x}$', ylabel=r'$\theta$',
                   line_color='m', show=False)
        ax4 = plot(self.deflection().subs(subs), (variable, 0, length),
                   title="Deflection", xlabel=r'$\mathrm{x}$', ylabel=r'$\delta$',
                   line_color='r', show=False)

        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _solve_for_ild_equations(self, value):
        """

        Helper function for I.L.D. It takes the unsubstituted
        copy of the load equation and uses it to calculate shear force and bending
        moment equations.
        """
        x = self.variable
        a = self.ild_variable
        load = self._load + value * SingularityFunction(x, a, -1)
        shear_force = -integrate(load, x)
        bending_moment = integrate(shear_force, x)

        return shear_force, bending_moment

    def solve_for_ild_reactions(self, value, *reactions):
        """

        Determines the Influence Line Diagram equations for reaction
        forces under the effect of a moving load.

        Parameters
        ==========
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 10 meters. There are two simple supports
        below the beam, one at the starting point and another at the ending
        point of the beam. Calculate the I.L.D. equations for reaction forces
        under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_10 = symbols('R_0, R_10')
            >>> b = Beam(10, E, I)
            >>> p0 = b.apply_support(0, 'pin')
            >>> p10 = b.apply_support(10, 'roller')
            >>> b.solve_for_ild_reactions(1,R_0,R_10)
            >>> b.ild_reactions
            {R_0: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/10 - SingularityFunction(a, 10, 1)/10,
            R_10: -SingularityFunction(a, 0, 1)/10 + SingularityFunction(a, 10, 0) + SingularityFunction(a, 10, 1)/10}

        """
        shear_force, bending_moment = self._solve_for_ild_equations(value)
        x = self.variable
        l = self.length
        a = self.ild_variable

        rotation_jumps = tuple(self._rotation_hinge_symbols)
        deflection_jumps = tuple(self._sliding_hinge_symbols)

        C3 = Symbol('C3')
        C4 = Symbol('C4')

        shear_curve = limit(shear_force, x, l) - value*(SingularityFunction(a, 0, 0) - SingularityFunction(a, l, 0))
        moment_curve = (limit(bending_moment, x, l) - value * (l * SingularityFunction(a, 0, 0)
                                                               - SingularityFunction(a, 0, 1)
                                                               + SingularityFunction(a, l, 1)))

        shear_force_eqs = []
        bending_moment_eqs = []
        slope_eqs = []
        deflection_eqs = []

        for position, val in self._boundary_conditions['shear_force']:
            eqs = self.shear_force().subs(x, position) - val
            eqs_without_inf = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            shear_sinc = value * (SingularityFunction(- a, - position, 0) - SingularityFunction(-a, 0, 0))
            eqs_with_shear_sinc = eqs_without_inf - shear_sinc
            shear_force_eqs.append(eqs_with_shear_sinc)

        for position, val in self._boundary_conditions['bending_moment']:
            eqs = self.bending_moment().subs(x, position) - val
            eqs_without_inf = sum(arg for arg in eqs.args if not any(num.is_infinite for num in arg.args))
            moment_sinc = value * (position * SingularityFunction(a, 0, 0)
                                   - SingularityFunction(a, 0, 1) + SingularityFunction(a, position, 1))
            eqs_with_moment_sinc = eqs_without_inf - moment_sinc
            bending_moment_eqs.append(eqs_with_moment_sinc)

        slope_curve = integrate(bending_moment, x) + C3
        for position, val in self._boundary_conditions['slope']:
            eqs = slope_curve.subs(x, position) - val + value * (SingularityFunction(-a, 0, 1) + position * SingularityFunction(-a, 0, 0))**2 / 2
            slope_eqs.append(eqs)

        deflection_curve = integrate(slope_curve, x) + C4
        for position, val in self._boundary_conditions['deflection']:
            eqs = deflection_curve.subs(x, position) - val + value * (SingularityFunction(-a, 0, 1) + position * SingularityFunction(-a, 0, 0)) ** 3 / 6
            deflection_eqs.append(eqs)

        solution = list((linsolve([shear_curve, moment_curve] + shear_force_eqs + bending_moment_eqs + slope_eqs
                                  + deflection_eqs, (C3, C4) + reactions + rotation_jumps + deflection_jumps).args)[0])

        reaction_index = 2 + len(reactions)
        rotation_index = reaction_index + len(rotation_jumps)
        reaction_solution = solution[2:reaction_index]
        rotation_solution = solution[reaction_index:rotation_index]
        deflection_solution = solution[rotation_index:]

        self._ild_reactions = dict(zip(reactions, reaction_solution))
        self._ild_rotations_jumps = dict(zip(rotation_jumps, rotation_solution))
        self._ild_deflection_jumps = dict(zip(deflection_jumps, deflection_solution))

    def plot_ild_reactions(self, subs=None):
        """

        Plots the Influence Line Diagram of Reaction Forces
        under the effect of a moving load. This function
        should be called after calling solve_for_ild_reactions().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 10 meters. A point load of magnitude 5KN
        is also applied from top of the beam, at a distance of 4 meters
        from the starting point. There are two simple supports below the
        beam, located at the starting point and at a distance of 7 meters
        from the starting point. Plot the I.L.D. equations for reactions
        at both support points under the effect of a moving load
        of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_7 = symbols('R_0, R_7')
            >>> b = Beam(10, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p7 = b.apply_support(7, 'roller')
            >>> b.apply_load(5,4,-1)
            >>> b.solve_for_ild_reactions(1,R_0,R_7)
            >>> b.ild_reactions
            {R_0: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/7
            - 3*SingularityFunction(a, 10, 0)/7  - SingularityFunction(a, 10, 1)/7 - 15/7,
            R_7: -SingularityFunction(a, 0, 1)/7 + 10*SingularityFunction(a, 10, 0)/7 + SingularityFunction(a, 10, 1)/7 - 20/7}
            >>> b.plot_ild_reactions()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/7
            - 3*SingularityFunction(a, 10, 0)/7 - SingularityFunction(a, 10, 1)/7 - 15/7 for a over (0.0, 10.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -SingularityFunction(a, 0, 1)/7 + 10*SingularityFunction(a, 10, 0)/7
            + SingularityFunction(a, 10, 1)/7 - 20/7 for a over (0.0, 10.0)

        """
        if not self._ild_reactions:
            raise ValueError("I.L.D. reaction equations not found. Please use solve_for_ild_reactions() to generate the I.L.D. reaction equations.")

        a = self.ild_variable
        ildplots = []

        if subs is None:
            subs = {}

        for reaction in self._ild_reactions:
            for sym in self._ild_reactions[reaction].atoms(Symbol):
                if sym != a and sym not in subs:
                    raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for reaction in self._ild_reactions:
            ildplots.append(plot(self._ild_reactions[reaction].subs(subs),
            (a, 0, self._length.subs(subs)), title='I.L.D. for Reactions',
            xlabel=a, ylabel=reaction, line_color='blue', show=False))

        return PlotGrid(len(ildplots), 1, *ildplots)

    def solve_for_ild_shear(self, distance, value, *reactions):
        """

        Determines the Influence Line Diagram equations for shear at a
        specified point under the effect of a moving load.

        Parameters
        ==========
        distance : Integer
            Distance of the point from the start of the beam
            for which equations are to be determined
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Calculate the I.L.D. equations for Shear at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)
            >>> b.ild_shear
            -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1

        """

        x = self.variable
        l = self.length
        a = self.ild_variable

        shear_force, _ = self._solve_for_ild_equations(value)

        shear_curve1 = value - limit(shear_force, x, distance)
        shear_curve2 = (limit(shear_force, x, l) - limit(shear_force, x, distance)) - value

        for reaction in reactions:
            shear_curve1 = shear_curve1.subs(reaction,self._ild_reactions[reaction])
            shear_curve2 = shear_curve2.subs(reaction,self._ild_reactions[reaction])

        shear_eq = (shear_curve1 - (shear_curve1 - shear_curve2) * SingularityFunction(a, distance, 0)
                    - value * SingularityFunction(-a, 0, 0) + value * SingularityFunction(a, l, 0))

        self._ild_shear = shear_eq

    def plot_ild_shear(self,subs=None):
        """

        Plots the Influence Line Diagram for Shear under the effect
        of a moving load. This function should be called after
        calling solve_for_ild_shear().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Plot the I.L.D. for Shear at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_shear(4, 1, R_0, R_8)
            >>> b.ild_shear
            -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1
            >>> b.plot_ild_shear()
            Plot object containing:
            [0]: cartesian line: -(-SingularityFunction(a, 0, 0) + SingularityFunction(a, 12, 0) + 2)*SingularityFunction(a, 4, 0)
            - SingularityFunction(-a, 0, 0) - SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/8
            + SingularityFunction(a, 12, 0)/2 - SingularityFunction(a, 12, 1)/8 + 1 for a over (0.0, 12.0)

        """

        if not self._ild_shear:
            raise ValueError("I.L.D. shear equation not found. Please use solve_for_ild_shear() to generate the I.L.D. shear equations.")

        l = self._length
        a = self.ild_variable

        if subs is None:
            subs = {}

        for sym in self._ild_shear.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        return plot(self._ild_shear.subs(subs), (a, 0, l),  title='I.L.D. for Shear',
               xlabel=r'$\mathrm{a}$', ylabel=r'$\mathrm{V}$', line_color='blue',show=True)

    def solve_for_ild_moment(self, distance, value, *reactions):
        """

        Determines the Influence Line Diagram equations for moment at a
        specified point under the effect of a moving load.

        Parameters
        ==========
        distance : Integer
            Distance of the point from the start of the beam
            for which equations are to be determined
        value : Integer
            Magnitude of moving load
        reactions :
            The reaction forces applied on the beam.

        Warning
        =======
        This method creates equations that can give incorrect results when
        substituting a = 0 or a = l, with l the length of the beam.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Calculate the I.L.D. equations for Moment at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)
            >>> b.ild_moment
            -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1) + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0)
            - SingularityFunction(a, 0, 1)/2 + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0)
            - SingularityFunction(a, 12, 1)/2

        """

        x = self.variable
        l = self.length
        a = self.ild_variable

        _, moment = self._solve_for_ild_equations(value)

        moment_curve1 = value*(distance * SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
                               + SingularityFunction(a, distance, 1)) - limit(moment, x, distance)
        moment_curve2 = (limit(moment, x, l)-limit(moment, x, distance)
                         - value * (l * SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
                                    + SingularityFunction(a, l, 1)))

        for reaction in reactions:
            moment_curve1 = moment_curve1.subs(reaction, self._ild_reactions[reaction])
            moment_curve2 = moment_curve2.subs(reaction, self._ild_reactions[reaction])

        moment_eq = moment_curve1 - (moment_curve1 - moment_curve2) * SingularityFunction(a, distance, 0)

        self._ild_moment = moment_eq

    def plot_ild_moment(self,subs=None):
        """

        Plots the Influence Line Diagram for Moment under the effect
        of a moving load. This function should be called after
        calling solve_for_ild_moment().

        Parameters
        ==========

        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Warning
        =======
        The values for a = 0 and a = l, with l the length of the beam, in
        the plot can be incorrect.

        Examples
        ========

        There is a beam of length 12 meters. There are two simple supports
        below the beam, one at the starting point and another at a distance
        of 8 meters. Plot the I.L.D. for Moment at a distance
        of 4 meters under the effect of a moving load of magnitude 1kN.

        Using the sign convention of downwards forces being positive.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy import symbols
            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> E, I = symbols('E, I')
            >>> R_0, R_8 = symbols('R_0, R_8')
            >>> b = Beam(12, E, I)
            >>> p0 = b.apply_support(0, 'roller')
            >>> p8 = b.apply_support(8, 'roller')
            >>> b.solve_for_ild_reactions(1, R_0, R_8)
            >>> b.solve_for_ild_moment(4, 1, R_0, R_8)
            >>> b.ild_moment
            -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1) + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0)
            - SingularityFunction(a, 0, 1)/2 + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0)
            - SingularityFunction(a, 12, 1)/2
            >>> b.plot_ild_moment()
            Plot object containing:
            [0]: cartesian line: -(4*SingularityFunction(a, 0, 0) - SingularityFunction(a, 0, 1)
            + SingularityFunction(a, 4, 1))*SingularityFunction(a, 4, 0) - SingularityFunction(a, 0, 1)/2
            + SingularityFunction(a, 4, 1) - 2*SingularityFunction(a, 12, 0) - SingularityFunction(a, 12, 1)/2 for a over (0.0, 12.0)

        """

        if not self._ild_moment:
            raise ValueError("I.L.D. moment equation not found. Please use solve_for_ild_moment() to generate the I.L.D. moment equations.")

        a = self.ild_variable

        if subs is None:
            subs = {}

        for sym in self._ild_moment.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)

        for sym in self._length.atoms(Symbol):
            if sym != a and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        return plot(self._ild_moment.subs(subs), (a, 0, self._length), title='I.L.D. for Moment',
               xlabel=r'$\mathrm{a}$', ylabel=r'$\mathrm{M}$', line_color='blue', show=True)

    @doctest_depends_on(modules=('numpy',))
    def draw(self, pictorial=True):
        """
        Returns a plot object representing the beam diagram of the beam.
        In particular, the diagram might include:

        * the beam.
        * vertical black arrows represent point loads and support reaction
          forces (the latter if they have been added with the ``apply_load``
          method).
        * circular arrows represent moments.
        * shaded areas represent distributed loads.
        * the support, if ``apply_support`` has been executed.
        * if a composite beam has been created with the ``join`` method and
          a hinge has been specified, it will be shown with a white disc.

        The diagram shows positive loads on the upper side of the beam,
        and negative loads on the lower side. If two or more distributed
        loads acts along the same direction over the same region, the
        function will add them up together.

        .. note::
            The user must be careful while entering load values.
            The draw function assumes a sign convention which is used
            for plotting loads.
            Given a right handed coordinate system with XYZ coordinates,
            the beam's length is assumed to be along the positive X axis.
            The draw function recognizes positive loads(with n>-2) as loads
            acting along negative Y direction and positive moments acting
            along positive Z direction.

        Parameters
        ==========

        pictorial: Boolean (default=True)
            Setting ``pictorial=True`` would simply create a pictorial (scaled)
            view of the beam diagram. On the other hand, ``pictorial=False``
            would create a beam diagram with the exact dimensions on the plot.

        Examples
        ========

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam
            >>> from sympy import symbols
            >>> P1, P2, M = symbols('P1, P2, M')
            >>> E, I = symbols('E, I')
            >>> b = Beam(50, 20, 30)
            >>> b.apply_load(-10, 2, -1)
            >>> b.apply_load(15, 26, -1)
            >>> b.apply_load(P1, 10, -1)
            >>> b.apply_load(-P2, 40, -1)
            >>> b.apply_load(90, 5, 0, 23)
            >>> b.apply_load(10, 30, 1, 50)
            >>> b.apply_load(M, 15, -2)
            >>> b.apply_load(-M, 30, -2)
            >>> p50 = b.apply_support(50, "pin")
            >>> p0, m0 = b.apply_support(0, "fixed")
            >>> p20 = b.apply_support(20, "roller")
            >>> p = b.draw()  # doctest: +SKIP
            >>> p  # doctest: +ELLIPSIS,+SKIP
            Plot object containing:
            [0]: cartesian line: 25*SingularityFunction(x, 5, 0) - 25*SingularityFunction(x, 23, 0)
            + SingularityFunction(x, 30, 1) - 20*SingularityFunction(x, 50, 0)
            - SingularityFunction(x, 50, 1) + 5 for x over (0.0, 50.0)
            [1]: cartesian line: 5 for x over (0.0, 50.0)
            ...
            >>> p.show() # doctest: +SKIP

        """
        if not numpy:
            raise ImportError("To use this function numpy module is required")

        loads = list(set(self.applied_loads) - set(self._support_as_loads))
        if (not pictorial) and any((len(l[0].free_symbols) > 0) and (l[2] >= 0) for l in loads):
            raise ValueError("`pictorial=False` requires numerical "
                "distributed loads. Instead, symbolic loads were found. "
                "Cannot continue.")

        x = self.variable

        # checking whether length is an expression in terms of any Symbol.
        if isinstance(self.length, Expr):
            l = list(self.length.atoms(Symbol))
            # assigning every Symbol a default value of 10
            l = dict.fromkeys(l, 10)
            length = self.length.subs(l)
        else:
            l = {}
            length = self.length
        height = length/10

        rectangles = []
        rectangles.append({'xy':(0, 0), 'width':length, 'height': height, 'facecolor':"brown"})
        annotations, markers, load_eq,load_eq1, fill = self._draw_load(pictorial, length, l)
        support_markers, support_rectangles = self._draw_supports(length, l)

        rectangles += support_rectangles
        markers += support_markers

        for loc in self._applied_rotation_hinges:
            ratio = loc / self.length
            x_pos = float(ratio) * length
            markers += [{'args':[[x_pos], [height / 2]], 'marker':'o', 'markersize':6, 'color':"white"}]

        for loc in self._applied_sliding_hinges:
            ratio = loc / self.length
            x_pos = float(ratio) * length
            markers += [{'args': [[x_pos], [height / 2]], 'marker':'|', 'markersize':12, 'color':"white"}]

        ylim = (-length, 1.25*length)
        if fill:
            # when distributed loads are presents, they might get clipped out
            # in the figure by the ylim settings.
            # It might be necessary to compute new limits.
            _min = min(min(fill["y2"]), min(r["xy"][1] for r in rectangles))
            _max = max(max(fill["y1"]), max(r["xy"][1] for r in rectangles))
            if (_min < ylim[0]) or (_max > ylim[1]):
                offset = abs(_max - _min) * 0.1
                ylim = (_min - offset, _max + offset)

        sing_plot = plot(height + load_eq, height + load_eq1, (x, 0, length),
            xlim=(-height, length + height), ylim=ylim,
            annotations=annotations, markers=markers, rectangles=rectangles,
            line_color='brown', fill=fill, axis=False, show=False)

        return sing_plot


    def _is_load_negative(self, load):
        """Try to determine if a load is negative or positive, using
        expansion and doit if necessary.

        Returns
        =======
        True: if the load is negative
        False: if the load is positive
        None: if it is indeterminate

        """
        rv = load.is_negative
        if load.is_Atom or rv is not None:
            return rv
        return load.doit().expand().is_negative

    def _draw_load(self, pictorial, length, l):
        loads = list(set(self.applied_loads) - set(self._support_as_loads))
        height = length/10
        x = self.variable

        annotations = []
        markers = []
        load_args = []
        scaled_load = 0
        load_args1 = []
        scaled_load1 = 0
        load_eq = S.Zero     # For positive valued higher order loads
        load_eq1 = S.Zero    # For negative valued higher order loads
        fill = None

        # schematic view should use the class convention as much as possible.
        # However, users can add expressions as symbolic loads, for example
        # P1 - P2: is this load positive or negative? We can't say.
        # On these occasions it is better to inform users about the
        # indeterminate state of those loads.
        warning_head = "Please, note that this schematic view might not be " \
            "in agreement with the sign convention used by the Beam class " \
            "for load-related computations, because it was not possible " \
            "to determine the sign (hence, the direction) of the " \
            "following loads:\n"
        warning_body = ""

        for load in loads:
            # check if the position of load is in terms of the beam length.
            if l:
                pos =  load[1].subs(l)
            else:
                pos = load[1]

            # point loads
            if load[2] == -1:
                iln = self._is_load_negative(load[0])
                if iln is None:
                    warning_body += "* Point load %s located at %s\n" % (load[0], load[1])
                if iln:
                    annotations.append({'text':'', 'xy':(pos, 0), 'xytext':(pos, height - 4*height), 'arrowprops':{'width': 1.5, 'headlength': 5, 'headwidth': 5, 'facecolor': 'black'}})
                else:
                    annotations.append({'text':'', 'xy':(pos, height),  'xytext':(pos, height*4), 'arrowprops':{"width": 1.5, "headlength": 4, "headwidth": 4, "facecolor": 'black'}})
            # moment loads
            elif load[2] == -2:
                iln = self._is_load_negative(load[0])
                if iln is None:
                    warning_body += "* Moment %s located at %s\n" % (load[0], load[1])
                if self._is_load_negative(load[0]):
                    markers.append({'args':[[pos], [height/2]], 'marker': r'$\circlearrowright$', 'markersize':15})
                else:
                    markers.append({'args':[[pos], [height/2]], 'marker': r'$\circlearrowleft$', 'markersize':15})
            # higher order loads
            elif load[2] >= 0:
                # `fill` will be assigned only when higher order loads are present
                value, start, order, end = load

                iln = self._is_load_negative(value)
                if iln is None:
                    warning_body += "* Distributed load %s from %s to %s\n" % (value, start, end)

                # Positive loads have their separate equations
                if not iln:
                    # if pictorial is True we remake the load equation again with
                    # some constant magnitude values.
                    if pictorial:
                        # remake the load equation again with some constant
                        # magnitude values.
                        value = 10**(1-order) if order > 0 else length/2
                    scaled_load += value*SingularityFunction(x, start, order)
                    if end:
                        f2 = value*x**order if order >= 0 else length/2*x**order
                        for i in range(0, order + 1):
                            scaled_load -= (f2.diff(x, i).subs(x, end - start)*
                                            SingularityFunction(x, end, i)/factorial(i))

                    if isinstance(scaled_load, Add):
                        load_args = scaled_load.args
                    else:
                        # when the load equation consists of only a single term
                        load_args = (scaled_load,)
                    load_eq = Add(*[i.subs(l) for i in load_args])

                # For loads with negative value
                else:
                    if pictorial:
                        # remake the load equation again with some constant
                        # magnitude values.
                        value = 10**(1-order) if order > 0 else length/2
                    scaled_load1 += abs(value)*SingularityFunction(x, start, order)
                    if end:
                        f2 = abs(value)*x**order if order >= 0 else length/2*x**order
                        for i in range(0, order + 1):
                            scaled_load1 -= (f2.diff(x, i).subs(x, end - start)*
                                            SingularityFunction(x, end, i)/factorial(i))

                    if isinstance(scaled_load1, Add):
                        load_args1 = scaled_load1.args
                    else:
                        # when the load equation consists of only a single term
                        load_args1 = (scaled_load1,)
                    load_eq1 = [i.subs(l) for i in load_args1]
                    load_eq1 = -Add(*load_eq1) - height

        if len(warning_body) > 0:
            warnings.warn(warning_head + warning_body)

        xx = numpy.arange(0, float(length), 0.001)
        yy1 = lambdify([x], height + load_eq.rewrite(Piecewise))(xx)
        yy2 = lambdify([x], height + load_eq1.rewrite(Piecewise))(xx)
        if not isinstance(yy1, numpy.ndarray):
            yy1 *= numpy.ones_like(xx)
        if not isinstance(yy2, numpy.ndarray):
            yy2 *= numpy.ones_like(xx)
        fill = {'x': xx, 'y1': yy1, 'y2': yy2,
            'color':'darkkhaki', "zorder": -1}
        return annotations, markers, load_eq, load_eq1, fill


    def _draw_supports(self, length, l):
        height = float(length/10)

        support_markers = []
        support_rectangles = []
        for support in self._applied_supports:
            if l:
                pos =  support[0].subs(l)
            else:
                pos = support[0]

            if support[1] == "pin":
                support_markers.append({'args':[pos, [0]], 'marker':6, 'markersize':13, 'color':"black"})

            elif support[1] == "roller":
                support_markers.append({'args':[pos, [-height/2.5]], 'marker':'o', 'markersize':11, 'color':"black"})

            elif support[1] == "fixed":
                if pos == 0:
                    support_rectangles.append({'xy':(0, -3*height), 'width':-length/20, 'height':6*height + height, 'fill':False, 'hatch':'/////'})
                else:
                    support_rectangles.append({'xy':(length, -3*height), 'width':length/20, 'height': 6*height + height, 'fill':False, 'hatch':'/////'})

        return support_markers, support_rectangles


class Beam3D(Beam):
    """
    This class handles loads applied in any direction of a 3D space along
    with unequal values of Second moment along different axes.

    .. note::
       A consistent sign convention must be used while solving a beam
       bending problem; the results will
       automatically follow the chosen sign convention.
       This class assumes that any kind of distributed load/moment is
       applied through out the span of a beam.

    Examples
    ========
    There is a beam of l meters long. A constant distributed load of magnitude q
    is applied along y-axis from start till the end of beam. A constant distributed
    moment of magnitude m is also applied along z-axis from start till the end of beam.
    Beam is fixed at both of its end. So, deflection of the beam at the both ends
    is restricted.

    >>> from sympy.physics.continuum_mechanics.beam import Beam3D
    >>> from sympy import symbols, simplify, collect, factor
    >>> l, E, G, I, A = symbols('l, E, G, I, A')
    >>> b = Beam3D(l, E, G, I, A)
    >>> x, q, m = symbols('x, q, m')
    >>> b.apply_load(q, 0, 0, dir="y")
    >>> b.apply_moment_load(m, 0, -1, dir="z")
    >>> b.shear_force()
    [0, -q*x, 0]
    >>> b.bending_moment()
    [0, 0, -m*x + q*x**2/2]
    >>> b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    >>> b.solve_slope_deflection()
    >>> factor(b.slope())
    [0, 0, x*(-l + x)*(-A*G*l**3*q + 2*A*G*l**2*q*x - 12*E*I*l*q
        - 72*E*I*m + 24*E*I*q*x)/(12*E*I*(A*G*l**2 + 12*E*I))]
    >>> dx, dy, dz = b.deflection()
    >>> dy = collect(simplify(dy), x)
    >>> dx == dz == 0
    True
    >>> dy == (x*(12*E*I*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q)
    ... + x*(A*G*l*(3*l*(A*G*l**2*q - 2*A*G*l*m + 12*E*I*q) + x*(-2*A*G*l**2*q + 4*A*G*l*m - 24*E*I*q))
    ... + A*G*(A*G*l**2 + 12*E*I)*(-2*l**2*q + 6*l*m - 4*m*x + q*x**2)
    ... - 12*E*I*q*(A*G*l**2 + 12*E*I)))/(24*A*E*G*I*(A*G*l**2 + 12*E*I)))
    True

    References
    ==========

    .. [1] https://homes.civil.aau.dk/jc/FemteSemester/Beams3D.pdf

    """

    def __init__(self, length, elastic_modulus, shear_modulus, second_moment,
                 area, variable=Symbol('x')):
        """Initializes the class.

        Parameters
        ==========
        length : Sympifyable
            A Symbol or value representing the Beam's length.
        elastic_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of Elasticity.
            It is a measure of the stiffness of the Beam material.
        shear_modulus : Sympifyable
            A SymPy expression representing the Beam's Modulus of rigidity.
            It is a measure of rigidity of the Beam material.
        second_moment : Sympifyable or list
            A list of two elements having SymPy expression representing the
            Beam's Second moment of area. First value represent Second moment
            across y-axis and second across z-axis.
            Single SymPy expression can be passed if both values are same
        area : Sympifyable
            A SymPy expression representing the Beam's cross-sectional area
            in a plane perpendicular to length of the Beam.
        variable : Symbol, optional
            A Symbol object that will be used as the variable along the beam
            while representing the load, shear, moment, slope and deflection
            curve. By default, it is set to ``Symbol('x')``.
        """
        super().__init__(length, elastic_modulus, second_moment, variable)
        self.shear_modulus = shear_modulus
        self.area = area
        self._load_vector = [0, 0, 0]
        self._moment_load_vector = [0, 0, 0]
        self._torsion_moment = {}
        self._load_Singularity = [0, 0, 0]
        self._slope = [0, 0, 0]
        self._deflection = [0, 0, 0]
        self._angular_deflection = 0

    @property
    def shear_modulus(self):
        """Young's Modulus of the Beam. """
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, e):
        self._shear_modulus = sympify(e)

    @property
    def second_moment(self):
        """Second moment of area of the Beam. """
        return self._second_moment

    @second_moment.setter
    def second_moment(self, i):
        if isinstance(i, list):
            i = [sympify(x) for x in i]
            self._second_moment = i
        else:
            self._second_moment = sympify(i)

    @property
    def area(self):
        """Cross-sectional area of the Beam. """
        return self._area

    @area.setter
    def area(self, a):
        self._area = sympify(a)

    @property
    def load_vector(self):
        """
        Returns a three element list representing the load vector.
        """
        return self._load_vector

    @property
    def moment_load_vector(self):
        """
        Returns a three element list representing moment loads on Beam.
        """
        return self._moment_load_vector

    @property
    def boundary_conditions(self):
        """
        Returns a dictionary of boundary conditions applied on the beam.
        The dictionary has two keywords namely slope and deflection.
        The value of each keyword is a list of tuple, where each tuple
        contains location and value of a boundary condition in the format
        (location, value). Further each value is a list corresponding to
        slope or deflection(s) values along three axes at that location.

        Examples
        ========
        There is a beam of length 4 meters. The slope at 0 should be 4 along
        the x-axis and 0 along others. At the other end of beam, deflection
        along all the three axes should be zero.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.bc_slope = [(0, (4, 0, 0))]
        >>> b.bc_deflection = [(4, [0, 0, 0])]
        >>> b.boundary_conditions
        {'bending_moment': [], 'deflection': [(4, [0, 0, 0])], 'shear_force': [], 'slope': [(0, (4, 0, 0))]}

        Here the deflection of the beam should be ``0`` along all the three axes at ``4``.
        Similarly, the slope of the beam should be ``4`` along x-axis and ``0``
        along y and z axis at ``0``.
        """
        return self._boundary_conditions

    def polar_moment(self):
        """
        Returns the polar moment of area of the beam
        about the X axis with respect to the centroid.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A = symbols('l, E, G, I, A')
        >>> b = Beam3D(l, E, G, I, A)
        >>> b.polar_moment()
        2*I
        >>> I1 = [9, 15]
        >>> b = Beam3D(l, E, G, I1, A)
        >>> b.polar_moment()
        24
        """
        if not iterable(self.second_moment):
            return 2*self.second_moment
        return sum(self.second_moment)

    def apply_load(self, value, start, order, dir="y"):
        """
        This method adds up the force load to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied load.
        dir : String
            Axis along which load is applied.
        order : Integer
            The order of the applied load.
            - For point loads, order=-1
            - For constant distributed load, order=0
            - For ramp loads, order=1
            - For parabolic ramp loads, order=2
            - ... so on.
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if dir == "x":
            if not order == -1:
                self._load_vector[0] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)

        elif dir == "y":
            if not order == -1:
                self._load_vector[1] += value
            self._load_Singularity[1] += value*SingularityFunction(x, start, order)

        else:
            if not order == -1:
                self._load_vector[2] += value
            self._load_Singularity[2] += value*SingularityFunction(x, start, order)

    def apply_moment_load(self, value, start, order, dir="y"):
        """
        This method adds up the moment loads to a particular beam object.

        Parameters
        ==========
        value : Sympifyable
            The magnitude of an applied moment.
        dir : String
