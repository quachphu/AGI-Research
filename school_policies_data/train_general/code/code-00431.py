        i, j, k = coord_sys.base_vectors()
        x, y, z = coord_sys.base_scalars()
        h1, h2, h3 = coord_sys.lame_coefficients()
        vectx = vect.dot(i)
        vecty = vect.dot(j)
        vectz = vect.dot(k)
        outvec = Vector.zero
        outvec += (Derivative(vectz * h3, y) -
                   Derivative(vecty * h2, z)) * i / (h2 * h3)
        outvec += (Derivative(vectx * h1, z) -
                   Derivative(vectz * h3, x)) * j / (h1 * h3)
        outvec += (Derivative(vecty * h2, x) -
                   Derivative(vectx * h1, y)) * k / (h2 * h1)

        if doit:
            return outvec.doit()
        return outvec
    else:
        if isinstance(vect, (Add, VectorAdd)):
            from sympy.vector import express
            try:
                cs = next(iter(coord_sys))
                args = [express(i, cs, variables=True) for i in vect.args]
            except ValueError:
                args = vect.args
            return VectorAdd.fromiter(curl(i, doit=doit) for i in args)
        elif isinstance(vect, (Mul, VectorMul)):
            vector = [i for i in vect.args if isinstance(i, (Vector, Cross, Gradient))][0]
            scalar = Mul.fromiter(i for i in vect.args if not isinstance(i, (Vector, Cross, Gradient)))
            res = Cross(gradient(scalar), vector).doit() + scalar*curl(vector, doit=doit)
            if doit:
                return res.doit()
            return res
        elif isinstance(vect, (Cross, Curl, Gradient)):
            return Curl(vect)
        else:
            raise ValueError("Invalid argument for curl")


def divergence(vect, doit=True):
    """
    Returns the divergence of a vector field computed wrt the base
    scalars of the given coordinate system.

    Parameters
    ==========

    vector : Vector
        The vector operand

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, divergence
    >>> R = CoordSys3D('R')
    >>> v1 = R.x*R.y*R.z * (R.i+R.j+R.k)

    >>> divergence(v1)
    R.x*R.y + R.x*R.z + R.y*R.z
    >>> v2 = 2*R.y*R.z*R.j
    >>> divergence(v2)
    2*R.z

    """
    coord_sys = _get_coord_systems(vect)
    if len(coord_sys) == 0:
        return S.Zero
    elif len(coord_sys) == 1:
        if isinstance(vect, (Cross, Curl, Gradient)):
            return Divergence(vect)
        # TODO: is case of many coord systems, this gets a random one:
        coord_sys = next(iter(coord_sys))
        i, j, k = coord_sys.base_vectors()
        x, y, z = coord_sys.base_scalars()
        h1, h2, h3 = coord_sys.lame_coefficients()
        vx = _diff_conditional(vect.dot(i), x, h2, h3) \
             / (h1 * h2 * h3)
        vy = _diff_conditional(vect.dot(j), y, h3, h1) \
             / (h1 * h2 * h3)
        vz = _diff_conditional(vect.dot(k), z, h1, h2) \
             / (h1 * h2 * h3)
        res = vx + vy + vz
        if doit:
            return res.doit()
        return res
    else:
        if isinstance(vect, (Add, VectorAdd)):
            return Add.fromiter(divergence(i, doit=doit) for i in vect.args)
        elif isinstance(vect, (Mul, VectorMul)):
            vector = [i for i in vect.args if isinstance(i, (Vector, Cross, Gradient))][0]
            scalar = Mul.fromiter(i for i in vect.args if not isinstance(i, (Vector, Cross, Gradient)))
            res = Dot(vector, gradient(scalar)) + scalar*divergence(vector, doit=doit)
            if doit:
                return res.doit()
            return res
        elif isinstance(vect, (Cross, Curl, Gradient)):
            return Divergence(vect)
        else:
            raise ValueError("Invalid argument for divergence")


def gradient(scalar_field, doit=True):
    """
    Returns the vector gradient of a scalar field computed wrt the
    base scalars of the given coordinate system.

    Parameters
    ==========

    scalar_field : SymPy Expr
        The scalar field to compute the gradient of

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, gradient
    >>> R = CoordSys3D('R')
    >>> s1 = R.x*R.y*R.z
    >>> gradient(s1)
    R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k
    >>> s2 = 5*R.x**2*R.z
    >>> gradient(s2)
    10*R.x*R.z*R.i + 5*R.x**2*R.k

    """
    coord_sys = _get_coord_systems(scalar_field)

    if len(coord_sys) == 0:
        return Vector.zero
    elif len(coord_sys) == 1:
        coord_sys = next(iter(coord_sys))
        h1, h2, h3 = coord_sys.lame_coefficients()
        i, j, k = coord_sys.base_vectors()
        x, y, z = coord_sys.base_scalars()
        vx = Derivative(scalar_field, x) / h1
        vy = Derivative(scalar_field, y) / h2
        vz = Derivative(scalar_field, z) / h3

        if doit:
            return (vx * i + vy * j + vz * k).doit()
        return vx * i + vy * j + vz * k
    else:
        if isinstance(scalar_field, (Add, VectorAdd)):
            return VectorAdd.fromiter(gradient(i) for i in scalar_field.args)
        if isinstance(scalar_field, (Mul, VectorMul)):
            s = _split_mul_args_wrt_coordsys(scalar_field)
            return VectorAdd.fromiter(scalar_field / i * gradient(i) for i in s)
        return Gradient(scalar_field)


class Laplacian(Expr):
    """
    Represents unevaluated Laplacian.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Laplacian
    >>> R = CoordSys3D('R')
    >>> v = 3*R.x**3*R.y**2*R.z**3
    >>> Laplacian(v)
    Laplacian(3*R.x**3*R.y**2*R.z**3)

    """

    def __new__(cls, expr):
        expr = sympify(expr)
        obj = Expr.__new__(cls, expr)
        obj._expr = expr
        return obj

    def doit(self, **hints):
        from sympy.vector.functions import laplacian
        return laplacian(self._expr)


def _diff_conditional(expr, base_scalar, coeff_1, coeff_2):
    """
    First re-expresses expr in the system that base_scalar belongs to.
    If base_scalar appears in the re-expressed form, differentiates
    it wrt base_scalar.
    Else, returns 0
    """
    from sympy.vector.functions import express
    new_expr = express(expr, base_scalar.system, variables=True)
    arg = coeff_1 * coeff_2 * new_expr
    return Derivative(arg, base_scalar) if arg else S.Zero
<|endoftext|>from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import (Vector, VectorAdd, VectorMul,
                                 BaseVector, VectorZero, Cross, Dot, cross, dot)
from sympy.vector.dyadic import (Dyadic, DyadicAdd, DyadicMul,
                                 BaseDyadic, DyadicZero)
from sympy.vector.scalar import BaseScalar
from sympy.vector.deloperator import Del
from sympy.vector.functions import (express, matrix_to_vector,
                                    laplacian, is_conservative,
                                    is_solenoidal, scalar_potential,
                                    directional_derivative,
                                    scalar_potential_difference)
from sympy.vector.point import Point
from sympy.vector.orienters import (AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)
from sympy.vector.operators import Gradient, Divergence, Curl, Laplacian, gradient, curl, divergence
from sympy.vector.implicitregion import ImplicitRegion
from sympy.vector.parametricregion import (ParametricRegion, parametric_region_list)
from sympy.vector.integrals import (ParametricIntegral, vector_integrate)
from sympy.vector.kind import VectorKind

__all__ = [
    'Vector', 'VectorAdd', 'VectorMul', 'BaseVector', 'VectorZero', 'Cross',
    'Dot', 'cross', 'dot',

    'VectorKind',

    'Dyadic', 'DyadicAdd', 'DyadicMul', 'BaseDyadic', 'DyadicZero',

    'BaseScalar',

    'Del',

    'CoordSys3D',

    'express', 'matrix_to_vector', 'laplacian', 'is_conservative',
    'is_solenoidal', 'scalar_potential', 'directional_derivative',
    'scalar_potential_difference',

    'Point',

    'AxisOrienter', 'BodyOrienter', 'SpaceOrienter', 'QuaternionOrienter',

    'Gradient', 'Divergence', 'Curl', 'Laplacian', 'gradient', 'curl',
    'divergence',

    'ParametricRegion', 'parametric_region_list', 'ImplicitRegion',

    'ParametricIntegral', 'vector_integrate',
]
<|endoftext|>from collections.abc import Callable

from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda
from sympy.core.symbol import Str
from sympy.core.symbol import symbols
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.matrices.matrixbase import MatrixBase
from sympy.solvers import solve
from sympy.vector.scalar import BaseScalar
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import eye
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
import sympy.vector
from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)


class CoordSys3D(Basic):
    """
    Represents a coordinate system in 3-D space.
    """

    def __new__(cls, name, transformation=None, parent=None, location=None,
                rotation_matrix=None, vector_names=None, variable_names=None):
        """
        The orientation/location parameters are necessary if this system
        is being defined at a certain orientation or location wrt another.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        location : Vector
            The position vector of the new system's origin wrt the parent
            instance.

        rotation_matrix : SymPy ImmutableMatrix
            The rotation matrix of the new coordinate system with respect
            to the parent. In other words, the output of
            new_system.rotation_matrix(parent).

        parent : CoordSys3D
            The coordinate system wrt which the orientation/location
            (or both) is being defined.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        """

        name = str(name)
        Vector = sympy.vector.Vector
        Point = sympy.vector.Point

        if not isinstance(name, str):
            raise TypeError("name should be a string")

        if transformation is not None:
            if (location is not None) or (rotation_matrix is not None):
                raise ValueError("specify either `transformation` or "
                                 "`location`/`rotation_matrix`")
            if isinstance(transformation, (Tuple, tuple, list)):
                if isinstance(transformation[0], MatrixBase):
                    rotation_matrix = transformation[0]
                    location = transformation[1]
                else:
                    transformation = Lambda(transformation[0],
                                            transformation[1])
            elif isinstance(transformation, Callable):
                x1, x2, x3 = symbols('x1 x2 x3', cls=Dummy)
                transformation = Lambda((x1, x2, x3),
                                        transformation(x1, x2, x3))
            elif isinstance(transformation, str):
                transformation = Str(transformation)
            elif isinstance(transformation, (Str, Lambda)):
                pass
            else:
                raise TypeError("transformation: "
                                "wrong type {}".format(type(transformation)))

        # If orientation information has been provided, store
        # the rotation matrix accordingly
        if rotation_matrix is None:
            rotation_matrix = ImmutableDenseMatrix(eye(3))
        else:
            if not isinstance(rotation_matrix, MatrixBase):
                raise TypeError("rotation_matrix should be an Immutable" +
                                "Matrix instance")
            rotation_matrix = rotation_matrix.as_immutable()

        # If location information is not given, adjust the default
        # location as Vector.zero
        if parent is not None:
            if not isinstance(parent, CoordSys3D):
                raise TypeError("parent should be a " +
                                "CoordSys3D/None")
            if location is None:
                location = Vector.zero
            else:
                if not isinstance(location, Vector):
                    raise TypeError("location should be a Vector")
                # Check that location does not contain base
                # scalars
                for x in location.free_symbols:
                    if isinstance(x, BaseScalar):
                        raise ValueError("location should not contain" +
                                         " BaseScalars")
            origin = parent.origin.locate_new(name + '.origin',
                                              location)
        else:
            location = Vector.zero
            origin = Point(name + '.origin')

        if transformation is None:
            transformation = Tuple(rotation_matrix, location)

        if isinstance(transformation, Tuple):
            lambda_transformation = CoordSys3D._compose_rotation_and_translation(
                transformation[0],
                transformation[1],
                parent
            )
            r, l = transformation
            l = l._projections
            lambda_lame = CoordSys3D._get_lame_coeff('cartesian')
            lambda_inverse = lambda x, y, z: r.inv()*Matrix(
                [x-l[0], y-l[1], z-l[2]])
        elif isinstance(transformation, Str):
            trname = transformation.name
            lambda_transformation = CoordSys3D._get_transformation_lambdas(trname)
            if parent is not None:
                if parent.lame_coefficients() != (S.One, S.One, S.One):
                    raise ValueError('Parent for pre-defined coordinate '
                                 'system should be Cartesian.')
            lambda_lame = CoordSys3D._get_lame_coeff(trname)
            lambda_inverse = CoordSys3D._set_inv_trans_equations(trname)
        elif isinstance(transformation, Lambda):
            if not CoordSys3D._check_orthogonality(transformation):
                raise ValueError("The transformation equation does not "
                                 "create orthogonal coordinate system")
            lambda_transformation = transformation
            lambda_lame = CoordSys3D._calculate_lame_coeff(lambda_transformation)
            lambda_inverse = None
        else:
            lambda_transformation = lambda x, y, z: transformation(x, y, z)
            lambda_lame = CoordSys3D._get_lame_coeff(transformation)
            lambda_inverse = None

        if variable_names is None:
            if isinstance(transformation, Lambda):
                variable_names = ["x1", "x2", "x3"]
            elif isinstance(transformation, Str):
                if transformation.name == 'spherical':
                    variable_names = ["r", "theta", "phi"]
                elif transformation.name == 'cylindrical':
                    variable_names = ["r", "theta", "z"]
                else:
                    variable_names = ["x", "y", "z"]
            else:
                variable_names = ["x", "y", "z"]
        if vector_names is None:
            vector_names = ["i", "j", "k"]

        # All systems that are defined as 'roots' are unequal, unless
        # they have the same name.
        # Systems defined at same orientation/position wrt the same
        # 'parent' are equal, irrespective of the name.
        # This is true even if the same orientation is provided via
        # different methods like Axis/Body/Space/Quaternion.
        # However, coincident systems may be seen as unequal if
        # positioned/oriented wrt different parents, even though
        # they may actually be 'coincident' wrt the root system.
        if parent is not None:
            obj = super().__new__(
                cls, Str(name), transformation, parent)
        else:
            obj = super().__new__(
                cls, Str(name), transformation)
        obj._name = name
        # Initialize the base vectors

        _check_strings('vector_names', vector_names)
        vector_names = list(vector_names)
        latex_vects = [(r'\mathbf{\hat{%s}_{%s}}' % (x, name)) for
                           x in vector_names]
        pretty_vects = ['%s_%s' % (x, name) for x in vector_names]

        obj._vector_names = vector_names

        v1 = BaseVector(0, obj, pretty_vects[0], latex_vects[0])
        v2 = BaseVector(1, obj, pretty_vects[1], latex_vects[1])
        v3 = BaseVector(2, obj, pretty_vects[2], latex_vects[2])

        obj._base_vectors = (v1, v2, v3)

        # Initialize the base scalars

        _check_strings('variable_names', vector_names)
        variable_names = list(variable_names)
        latex_scalars = [(r"\mathbf{{%s}_{%s}}" % (x, name)) for
                         x in variable_names]
        pretty_scalars = ['%s_%s' % (x, name) for x in variable_names]

        obj._variable_names = variable_names
        obj._vector_names = vector_names

        x1 = BaseScalar(0, obj, pretty_scalars[0], latex_scalars[0])
        x2 = BaseScalar(1, obj, pretty_scalars[1], latex_scalars[1])
        x3 = BaseScalar(2, obj, pretty_scalars[2], latex_scalars[2])

        obj._base_scalars = (x1, x2, x3)

        obj._transformation = transformation
        obj._transformation_lambda = lambda_transformation
        obj._lame_coefficients = lambda_lame(x1, x2, x3)
        obj._transformation_from_parent_lambda = lambda_inverse

        setattr(obj, variable_names[0], x1)
        setattr(obj, variable_names[1], x2)
        setattr(obj, variable_names[2], x3)

        setattr(obj, vector_names[0], v1)
        setattr(obj, vector_names[1], v2)
        setattr(obj, vector_names[2], v3)

        # Assign params
        obj._parent = parent
        if obj._parent is not None:
            obj._root = obj._parent._root
        else:
            obj._root = obj

        obj._parent_rotation_matrix = rotation_matrix
        obj._origin = origin

        # Return the instance
        return obj

    def _sympystr(self, printer):
        return self._name

    def __iter__(self):
        return iter(self.base_vectors())

    @staticmethod
    def _check_orthogonality(equations):
        """
        Helper method for _connect_to_cartesian. It checks if
        set of transformation equations create orthogonal curvilinear
        coordinate system

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations

        """

        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy)
        equations = equations(x1, x2, x3)
        v1 = Matrix([diff(equations[0], x1),
                     diff(equations[1], x1), diff(equations[2], x1)])

        v2 = Matrix([diff(equations[0], x2),
                     diff(equations[1], x2), diff(equations[2], x2)])

        v3 = Matrix([diff(equations[0], x3),
                     diff(equations[1], x3), diff(equations[2], x3)])

        if any(simplify(i[0] + i[1] + i[2]) == 0 for i in (v1, v2, v3)):
            return False
        else:
            if simplify(v1.dot(v2)) == 0 and simplify(v2.dot(v3)) == 0 \
                and simplify(v3.dot(v1)) == 0:
                return True
            else:
                return False

    @staticmethod
    def _set_inv_trans_equations(curv_coord_name):
        """
        Store information about inverse transformation equations for
        pre-defined coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if curv_coord_name == 'cartesian':
            return lambda x, y, z: (x, y, z)

        if curv_coord_name == 'spherical':
            return lambda x, y, z: (
                sqrt(x**2 + y**2 + z**2),
                acos(z/sqrt(x**2 + y**2 + z**2)),
                atan2(y, x)
            )
        if curv_coord_name == 'cylindrical':
            return lambda x, y, z: (
                sqrt(x**2 + y**2),
                atan2(y, x),
                z
            )
        raise ValueError('Wrong set of parameters.'
                         'Type of coordinate system is defined')

    def _calculate_inv_trans_equations(self):
        """
        Helper method for set_coordinate_type. It calculates inverse
        transformation equations for given transformations equations.

        """
        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy, reals=True)
        x, y, z = symbols("x, y, z", cls=Dummy)

        equations = self._transformation(x1, x2, x3)

        solved = solve([equations[0] - x,
                        equations[1] - y,
                        equations[2] - z], (x1, x2, x3), dict=True)[0]
        solved = solved[x1], solved[x2], solved[x3]
        self._transformation_from_parent_lambda = \
            lambda x1, x2, x3: tuple(i.subs(list(zip((x, y, z), (x1, x2, x3)))) for i in solved)

    @staticmethod
    def _get_lame_coeff(curv_coord_name):
        """
        Store information about Lame coefficients for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if isinstance(curv_coord_name, str):
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (S.One, S.One, S.One)
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (S.One, r, r*sin(theta))
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (S.One, r, S.One)
            raise ValueError('Wrong set of parameters.'
                             ' Type of coordinate system is not defined')
        return CoordSys3D._calculate_lame_coefficients(curv_coord_name)

    @staticmethod
    def _calculate_lame_coeff(equations):
        """
        It calculates Lame coefficients
        for given transformations equations.

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations.

        """
        return lambda x1, x2, x3: (
                          sqrt(diff(equations(x1, x2, x3)[0], x1)**2 +
                               diff(equations(x1, x2, x3)[1], x1)**2 +
                               diff(equations(x1, x2, x3)[2], x1)**2),
                          sqrt(diff(equations(x1, x2, x3)[0], x2)**2 +
                               diff(equations(x1, x2, x3)[1], x2)**2 +
                               diff(equations(x1, x2, x3)[2], x2)**2),
                          sqrt(diff(equations(x1, x2, x3)[0], x3)**2 +
                               diff(equations(x1, x2, x3)[1], x3)**2 +
                               diff(equations(x1, x2, x3)[2], x3)**2)
                      )

    def _inverse_rotation_matrix(self):
        """
        Returns inverse rotation matrix.
        """
        return simplify(self._parent_rotation_matrix**-1)

    @staticmethod
    def _get_transformation_lambdas(curv_coord_name):
        """
        Store information about transformation equations for pre-defined
        coordinate systems.

        Parameters
        ==========

        curv_coord_name : str
            Name of coordinate system

        """
        if isinstance(curv_coord_name, str):
            if curv_coord_name == 'cartesian':
                return lambda x, y, z: (x, y, z)
            if curv_coord_name == 'spherical':
                return lambda r, theta, phi: (
                    r*sin(theta)*cos(phi),
                    r*sin(theta)*sin(phi),
                    r*cos(theta)
                )
            if curv_coord_name == 'cylindrical':
                return lambda r, theta, h: (
                    r*cos(theta),
                    r*sin(theta),
                    h
                )
            raise ValueError('Wrong set of parameters.'
                             'Type of coordinate system is defined')

    @classmethod
    def _rotation_trans_equations(cls, matrix, equations):
        """
        Returns the transformation equations obtained from rotation matrix.

        Parameters
        ==========

        matrix : Matrix
            Rotation matrix

        equations : tuple
            Transformation equations

        """
        return tuple(matrix * Matrix(equations))

    @property
    def origin(self):
        return self._origin

    def base_vectors(self):
        return self._base_vectors

    def base_scalars(self):
        return self._base_scalars

    def lame_coefficients(self):
        return self._lame_coefficients

    def transformation_to_parent(self):
        return self._transformation_lambda(*self.base_scalars())

    def transformation_from_parent(self):
        if self._parent is None:
            raise ValueError("no parent coordinate system, use "
                             "`transformation_from_parent_function()`")
        return self._transformation_from_parent_lambda(
                            *self._parent.base_scalars())

    def transformation_from_parent_function(self):
        return self._transformation_from_parent_lambda

    def rotation_matrix(self, other):
        """
        Returns the direction cosine matrix(DCM), also known as the
        'rotation matrix' of this coordinate system with respect to
        another system.

        If v_a is a vector defined in system 'A' (in matrix format)
        and v_b is the same vector defined in system 'B', then
        v_a = A.rotation_matrix(B) * v_b.

        A SymPy Matrix is returned.

        Parameters
        ==========

        other : CoordSys3D
            The system which the DCM is generated to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> A = N.orient_new_axis('A', q1, N.i)
        >>> N.rotation_matrix(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        """
        from sympy.vector.functions import _path
        if not isinstance(other, CoordSys3D):
            raise TypeError(str(other) +
                            " is not a CoordSys3D")
        # Handle special cases
        if other == self:
            return eye(3)
        elif other == self._parent:
            return self._parent_rotation_matrix
        elif other._parent == self:
            return other._parent_rotation_matrix.T
        # Else, use tree to calculate position
        rootindex, path = _path(self, other)
        result = eye(3)
        for i in range(rootindex):
            result *= path[i]._parent_rotation_matrix
        for i in range(rootindex + 1, len(path)):
            result *= path[i]._parent_rotation_matrix.T
        return result

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of the origin of this coordinate
        system with respect to another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this system's origin
            wrt it is returned. If its an instance of CoordSyRect,
            the position wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N1 = N.locate_new('N1', 10 * N.i)
        >>> N.position_wrt(N1)
        (-10)*N.i

        """
        return self.origin.position_wrt(other)

    def scalar_map(self, other):
        """
        Returns a dictionary which expresses the coordinate variables
        (base scalars) of this frame in terms of the variables of
        otherframe.

        Parameters
        ==========

        otherframe : CoordSys3D
            The other system to map the variables to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import Symbol
        >>> A = CoordSys3D('A')
        >>> q = Symbol('q')
        >>> B = A.orient_new_axis('B', q, A.k)
        >>> A.scalar_map(B)
        {A.x: B.x*cos(q) - B.y*sin(q), A.y: B.x*sin(q) + B.y*cos(q), A.z: B.z}

        """

        origin_coords = tuple(self.position_wrt(other).to_matrix(other))
        relocated_scalars = [x - origin_coords[i]
                             for i, x in enumerate(other.base_scalars())]

        vars_matrix = (self.rotation_matrix(other) *
                       Matrix(relocated_scalars))
        return {x: trigsimp(vars_matrix[i])
                for i, x in enumerate(self.base_scalars())}

    def locate_new(self, name, position, vector_names=None,
                   variable_names=None):
        """
        Returns a CoordSys3D with its origin located at the given
        position wrt this coordinate system's origin.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        position : Vector
            The position vector of the new system's origin wrt this
            one.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> A = CoordSys3D('A')
        >>> B = A.locate_new('B', 10 * A.i)
        >>> B.origin.position_wrt(A.origin)
        10*A.i

        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        return CoordSys3D(name, location=position,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          parent=self)

    def orient_new(self, name, orienters, location=None,
                   vector_names=None, variable_names=None):
        """
        Creates a new CoordSys3D oriented in the user-specified way
        with respect to this system.

        Please refer to the documentation of the orienter classes
        for more information about the orientation procedure.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        orienters : iterable/Orienter
            An Orienter or an iterable of Orienters for orienting the
            new coordinate system.
            If an Orienter is provided, it is applied to get the new
            system.
            If an iterable is provided, the orienters will be applied
            in the order in which they appear in the iterable.

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')

        Using an AxisOrienter

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> A = N.orient_new('A', (axis_orienter, ))

        Using a BodyOrienter

        >>> from sympy.vector import BodyOrienter
        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')
        >>> B = N.orient_new('B', (body_orienter, ))

        Using a SpaceOrienter

        >>> from sympy.vector import SpaceOrienter
        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')
        >>> C = N.orient_new('C', (space_orienter, ))

        Using a QuaternionOrienter

        >>> from sympy.vector import QuaternionOrienter
        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)
        >>> D = N.orient_new('D', (q_orienter, ))
        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        if isinstance(orienters, Orienter):
            if isinstance(orienters, AxisOrienter):
                final_matrix = orienters.rotation_matrix(self)
            else:
                final_matrix = orienters.rotation_matrix()
            # TODO: trigsimp is needed here so that the matrix becomes
            # canonical (scalar_map also calls trigsimp; without this, you can
            # end up with the same CoordinateSystem that compares differently
            # due to a differently formatted matrix). However, this is
            # probably not so good for performance.
            final_matrix = trigsimp(final_matrix)
        else:
            final_matrix = Matrix(eye(3))
            for orienter in orienters:
                if isinstance(orienter, AxisOrienter):
                    final_matrix *= orienter.rotation_matrix(self)
                else:
                    final_matrix *= orienter.rotation_matrix()

        return CoordSys3D(name, rotation_matrix=final_matrix,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          location=location,
                          parent=self)

    def orient_new_axis(self, name, angle, axis, location=None,
                        vector_names=None, variable_names=None):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_axis('B', q1, N.i + 2 * N.j)

        """
        if variable_names is None:
            variable_names = self._variable_names
        if vector_names is None:
            vector_names = self._vector_names

        orienter = AxisOrienter(angle, axis)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_body(self, name, angle1, angle2, angle3,
                        rotation_order, location=None,
                        vector_names=None, variable_names=None):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> D = N.orient_new_body('D', q1, q2, q3, '123')

        is same as

        >>> D = N.orient_new_axis('D', q1, N.i)
        >>> D = D.orient_new_axis('D', q2, D.j)
        >>> D = D.orient_new_axis('D', q3, D.k)

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> B = N.orient_new_body('B', q1, q2, q3, '123')
        >>> B = N.orient_new_body('B', q1, q2, 0, 'ZXZ')
        >>> B = N.orient_new_body('B', 0, 0, 0, 'XYX')

        """

        orienter = BodyOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_space(self, name, angle1, angle2, angle3,
                         rotation_order, location=None,
                         vector_names=None, variable_names=None):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        See Also
        ========

        CoordSys3D.orient_new_body : method to orient via Euler
            angles

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> D = N.orient_new_space('D', q1, q2, q3, '312')

        is same as

        >>> B = N.orient_new_axis('B', q1, N.i)
        >>> C = B.orient_new_axis('C', q2, N.j)
        >>> D = C.orient_new_axis('D', q3, N.k)

        """

        orienter = SpaceOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_quaternion(self, name, q0, q1, q2, q3, location=None,
                              vector_names=None, variable_names=None):
        """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_quaternion('B', q0, q1, q2, q3)

        """

        orienter = QuaternionOrienter(q0, q1, q2, q3)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def create_new(self, name, transformation, variable_names=None, vector_names=None):
        """
        Returns a CoordSys3D which is connected to self by transformation.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> a = CoordSys3D('a')
        >>> b = a.create_new('b', transformation='spherical')
        >>> b.transformation_to_parent()
        (b.r*sin(b.theta)*cos(b.phi), b.r*sin(b.phi)*sin(b.theta), b.r*cos(b.theta))
        >>> b.transformation_from_parent()
        (sqrt(a.x**2 + a.y**2 + a.z**2), acos(a.z/sqrt(a.x**2 + a.y**2 + a.z**2)), atan2(a.y, a.x))

        """
        return CoordSys3D(name, parent=self, transformation=transformation,
                          variable_names=variable_names, vector_names=vector_names)

    def __init__(self, name, location=None, rotation_matrix=None,
                 parent=None, vector_names=None, variable_names=None,
                 latex_vects=None, pretty_vects=None, latex_scalars=None,
                 pretty_scalars=None, transformation=None):
        # Dummy initializer for setting docstring
        pass

    __init__.__doc__ = __new__.__doc__

    @staticmethod
    def _compose_rotation_and_translation(rot, translation, parent):
        r = lambda x, y, z: CoordSys3D._rotation_trans_equations(rot, (x, y, z))
        if parent is None:
            return r

        dx, dy, dz = [translation.dot(i) for i in parent.base_vectors()]
        t = lambda x, y, z: (
            x + dx,
            y + dy,
            z + dz,
        )
        return lambda x, y, z: t(*r(x, y, z))


def _check_strings(arg_name, arg):
    errorstr = arg_name + " must be an iterable of 3 string-types"
    if len(arg) != 3:
        raise ValueError(errorstr)
    for s in arg:
        if not isinstance(s, str):
            raise TypeError(errorstr)


# Delayed import to avoid cyclic import problems:
from sympy.vector.vector import BaseVector
<|endoftext|>from __future__ import annotations
from itertools import product

from sympy.core import Add, Basic
from sympy.core.assumptions import StdFactKB
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.basisdependent import (BasisDependentZero,
    BasisDependent, BasisDependentMul, BasisDependentAdd)
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.dyadic import Dyadic, BaseDyadic, DyadicAdd
from sympy.vector.kind import VectorKind


class Vector(BasisDependent):
    """
    Super class for all Vector classes.
    Ideally, neither this class nor any of its subclasses should be
    instantiated by the user.
    """

    is_scalar = False
    is_Vector = True
    _op_priority = 12.0

    _expr_type: type[Vector]
    _mul_func: type[Vector]
    _add_func: type[Vector]
    _zero_func: type[Vector]
    _base_func: type[Vector]
    zero: VectorZero

    kind: VectorKind = VectorKind()

    @property
    def components(self):
        """
        Returns the components of this vector in the form of a
        Python dictionary mapping BaseVector instances to the
        corresponding measure numbers.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.components
        {C.i: 3, C.j: 4, C.k: 5}

        """
        # The '_components' attribute is defined according to the
        # subclass of Vector the instance belongs to.
        return self._components

    def magnitude(self):
        """
        Returns the magnitude of this vector.
        """
        return sqrt(self & self)

    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        return self / self.magnitude()

    def equals(self, other):
        """
        Check if ``self`` and ``other`` are identically equal vectors.

        Explanation
        ===========

        Checks if two vector expressions are equal for all possible values of
        the symbols present in the expressions.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy.abc import x, y
        >>> from sympy import pi
        >>> C = CoordSys3D('C')

        Compare vectors that are equal or not:

        >>> C.i.equals(C.j)
        False
        >>> C.i.equals(C.i)
        True

        These two vectors are equal if `x = y` but are not identically equal
        as expressions since for some values of `x` and `y` they are unequal:

        >>> v1 = x*C.i + C.j
        >>> v2 = y*C.i + C.j
        >>> v1.equals(v1)
        True
        >>> v1.equals(v2)
        False

        Vectors from different coordinate systems can be compared:

        >>> D = C.orient_new_axis('D', pi/2, C.i)
        >>> D.j.equals(C.j)
        False
        >>> D.j.equals(C.k)
        True

        Parameters
        ==========

        other: Vector
            The other vector expression to compare with.

        Returns
        =======

        ``True``, ``False`` or ``None``. A return value of ``True`` indicates
        that the two vectors are identically equal. A return value of ``False``
        indicates that they are not. In some cases it is not possible to
        determine if the two vectors are identically equal and ``None`` is
        returned.

        See Also
        ========

        sympy.core.expr.Expr.equals
        """
        diff = self - other
        diff_mag2 = diff.dot(diff)
        return diff_mag2.equals(0)

    def dot(self, other):
        """
        Returns the dot product of this Vector, either with another
        Vector, or a Dyadic, or a Del operator.
        If 'other' is a Vector, returns the dot product scalar (SymPy
        expression).
        If 'other' is a Dyadic, the dot product is returned as a Vector.
        If 'other' is an instance of Del, returns the directional
        derivative operator as a Python function. If this function is
        applied to a scalar expression, it returns the directional
        derivative of the scalar field wrt this Vector.

        Parameters
        ==========

        other: Vector/Dyadic/Del
            The Vector or Dyadic we are dotting with, or a Del operator .

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> C.i.dot(C.j)
        0
        >>> C.i & C.i
        1
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.dot(C.k)
        5
        >>> (C.i & delop)(C.x*C.y*C.z)
        C.y*C.z
        >>> d = C.i.outer(C.i)
        >>> C.i.dot(d)
        C.i

        """

        # Check special cases
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Vector.zero
            outvec = Vector.zero
            for k, v in other.components.items():
                vect_dot = k.args[0].dot(self)
                outvec += vect_dot * v * k.args[1]
            return outvec
        from sympy.vector.deloperator import Del
        if not isinstance(other, (Del, Vector)):
            raise TypeError(str(other) + " is not a vector, dyadic or " +
                            "del operator")

        # Check if the other is a del operator
        if isinstance(other, Del):
            def directional_derivative(field):
                from sympy.vector.functions import directional_derivative
                return directional_derivative(field, self)
            return directional_derivative

        return dot(self, other)

    def __and__(self, other):
        return self.dot(other)

    __and__.__doc__ = dot.__doc__

    def cross(self, other):
        """
        Returns the cross product of this Vector with another Vector or
        Dyadic instance.
        The cross product is a Vector, if 'other' is a Vector. If 'other'
        is a Dyadic, this returns a Dyadic instance.

        Parameters
        ==========

        other: Vector/Dyadic
            The Vector or Dyadic we are crossing with.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> C.i.cross(C.j)
        C.k
        >>> C.i ^ C.i
        0
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v ^ C.i
        5*C.j + (-4)*C.k
        >>> d = C.i.outer(C.i)
        >>> C.j.cross(d)
        (-1)*(C.k|C.i)

        """

        # Check special cases
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Dyadic.zero
            outdyad = Dyadic.zero
            for k, v in other.components.items():
                cross_product = self.cross(k.args[0])
                outer = cross_product.outer(k.args[1])
                outdyad += v * outer
            return outdyad

        return cross(self, other)

    def __xor__(self, other):
        return self.cross(other)

    __xor__.__doc__ = cross.__doc__

    def outer(self, other):
        """
        Returns the outer product of this vector with another, in the
        form of a Dyadic instance.

        Parameters
        ==========

        other : Vector
            The Vector with respect to which the outer product is to
            be computed.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N.i.outer(N.j)
        (N.i|N.j)

        """

        # Handle the special cases
        if not isinstance(other, Vector):
            raise TypeError("Invalid operand for outer product")
        elif (isinstance(self, VectorZero) or
                isinstance(other, VectorZero)):
            return Dyadic.zero

        # Iterate over components of both the vectors to generate
        # the required Dyadic instance
        args = [(v1 * v2) * BaseDyadic(k1, k2) for (k1, v1), (k2, v2)
                in product(self.components.items(), other.components.items())]

        return DyadicAdd(*args)

    def projection(self, other, scalar=False):
        """
        Returns the vector or scalar projection of the 'other' on 'self'.

        Examples
        ========

        >>> from sympy.vector.coordsysrect import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> i, j, k = C.base_vectors()
        >>> v1 = i + j + k
        >>> v2 = 3*i + 4*j
        >>> v1.projection(v2)
        7/3*C.i + 7/3*C.j + 7/3*C.k
        >>> v1.projection(v2, scalar=True)
        7/3

        """
        if self.equals(Vector.zero):
            return S.Zero if scalar else Vector.zero

        if scalar:
            return self.dot(other) / self.dot(self)
        else:
            return self.dot(other) / self.dot(self) * self

    @property
    def _projections(self):
        """
        Returns the components of this vector but the output includes
        also zero values components.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Vector
        >>> C = CoordSys3D('C')
        >>> v1 = 3*C.i + 4*C.j + 5*C.k
        >>> v1._projections
        (3, 4, 5)
        >>> v2 = C.x*C.y*C.z*C.i
        >>> v2._projections
        (C.x*C.y*C.z, 0, 0)
        >>> v3 = Vector.zero
        >>> v3._projections
        (0, 0, 0)
        """

        from sympy.vector.operators import _get_coord_systems
        if isinstance(self, VectorZero):
            return (S.Zero, S.Zero, S.Zero)
        base_vec = next(iter(_get_coord_systems(self))).base_vectors()
        return tuple([self.dot(i) for i in base_vec])

    def __or__(self, other):
        return self.outer(other)

    __or__.__doc__ = outer.__doc__

    def to_matrix(self, system):
        """
        Returns the matrix form of this vector with respect to the
        specified coordinate system.

        Parameters
        ==========

        system : CoordSys3D
            The system wrt which the matrix form is to be computed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> from sympy.abc import a, b, c
        >>> v = a*C.i + b*C.j + c*C.k
        >>> v.to_matrix(C)
        Matrix([
        [a],
        [b],
        [c]])

        """

        return Matrix([self.dot(unit_vec) for unit_vec in
                       system.base_vectors()])

    def separate(self):
        """
        The constituents of this vector in different coordinate systems,
        as per its definition.

        Returns a dict mapping each CoordSys3D to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> R1 = CoordSys3D('R1')
        >>> R2 = CoordSys3D('R2')
        >>> v = R1.i + R2.i
        >>> v.separate() == {R1: R1.i, R2: R2.i}
        True

        """

        parts = {}
        for vect, measure in self.components.items():
            parts[vect.system] = (parts.get(vect.system, Vector.zero) +
                                  vect * measure)
        return parts

    def _div_helper(one, other):
        """ Helper for division involving vectors. """
        if isinstance(one, Vector) and isinstance(other, Vector):
            raise TypeError("Cannot divide two vectors")
        elif isinstance(one, Vector):
            if other == S.Zero:
                raise ValueError("Cannot divide a vector by zero")
            return VectorMul(one, Pow(other, S.NegativeOne))
        else:
            raise TypeError("Invalid division involving a vector")

# The following is adapted from the matrices.expressions.matexpr file

def get_postprocessor(cls):
    def _postprocessor(expr):
        vec_class = {Add: VectorAdd}[cls]
        vectors = []
        for term in expr.args:
            if isinstance(term.kind, VectorKind):
                vectors.append(term)

        if vec_class == VectorAdd:
            return VectorAdd(*vectors).doit(deep=False)
    return _postprocessor


Basic._constructor_postprocessor_mapping[Vector] = {
    "Add": [get_postprocessor(Add)],
}

class BaseVector(Vector, AtomicExpr):
    """
    Class to denote a base vector.

    """

    def __new__(cls, index, system, pretty_str=None, latex_str=None):
        if pretty_str is None:
            pretty_str = "x{}".format(index)
        if latex_str is None:
            latex_str = "x_{}".format(index)
        pretty_str = str(pretty_str)
        latex_str = str(latex_str)
        # Verify arguments
        if index not in range(0, 3):
            raise ValueError("index must be 0, 1 or 2")
        if not isinstance(system, CoordSys3D):
            raise TypeError("system should be a CoordSys3D")
        name = system._vector_names[index]
        # Initialize an object
        obj = super().__new__(cls, S(index), system)
        # Assign important attributes
        obj._base_instance = obj
        obj._components = {obj: S.One}
        obj._measure_number = S.One
        obj._name = system._name + '.' + name
        obj._pretty_form = '' + pretty_str
        obj._latex_form = latex_str
        obj._system = system
        # The _id is used for printing purposes
        obj._id = (index, system)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)

        # This attr is used for re-expression to one of the systems
        # involved in the definition of the Vector. Applies to
        # VectorMul and VectorAdd too.
        obj._sys = system

        return obj

    @property
    def system(self):
        return self._system

    def _sympystr(self, printer):
        return self._name

    def _sympyrepr(self, printer):
        index, system = self._id
        return printer._print(system) + '.' + system._vector_names[index]

    @property
    def free_symbols(self):
        return {self}

    def _eval_conjugate(self):
        return self


class VectorAdd(BasisDependentAdd, Vector):
    """
    Class to denote sum of Vector instances.
    """

    def __new__(cls, *args, **options):
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    def _sympystr(self, printer):
        ret_str = ''
        items = list(self.separate().items())
        items.sort(key=lambda x: x[0].__str__())
        for system, vect in items:
            base_vects = system.base_vectors()
            for x in base_vects:
                if x in vect.components:
                    temp_vect = self.components[x] * x
                    ret_str += printer._print(temp_vect) + " + "
        return ret_str[:-3]


class VectorMul(BasisDependentMul, Vector):
    """
    Class to denote products of scalars and BaseVectors.
    """

    def __new__(cls, *args, **options):
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    @property
    def base_vector(self):
        """ The BaseVector involved in the product. """
        return self._base_instance

    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this VectorMul.
        """
        return self._measure_number


class VectorZero(BasisDependentZero, Vector):
    """
    Class to denote a zero vector
    """

    _op_priority = 12.1
    _pretty_form = '0'
    _latex_form = r'\mathbf{\hat{0}}'

    def __new__(cls):
        obj = BasisDependentZero.__new__(cls)
        return obj


class Cross(Vector):
    """
    Represents unevaluated Cross product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> Cross(v1, v2)
    Cross(R.i + R.j + R.k, R.x*R.i + R.y*R.j + R.z*R.k)
    >>> Cross(v1, v2).doit()
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        if default_sort_key(expr1) > default_sort_key(expr2):
            return -Cross(expr2, expr1)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        return cross(self._expr1, self._expr2)


class Dot(Expr):
    """
    Represents unevaluated Dot product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Dot
    >>> from sympy import symbols
    >>> R = CoordSys3D('R')
    >>> a, b, c = symbols('a b c')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = a * R.i + b * R.j + c * R.k
    >>> Dot(v1, v2)
    Dot(R.i + R.j + R.k, a*R.i + b*R.j + c*R.k)
    >>> Dot(v1, v2).doit()
    a + b + c

    """

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        expr1, expr2 = sorted([expr1, expr2], key=default_sort_key)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        return dot(self._expr1, self._expr2)


def cross(vect1, vect2):
    """
    Returns cross product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> cross(v1, v2)
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """
    if isinstance(vect1, Add):
        return VectorAdd.fromiter(cross(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add):
        return VectorAdd.fromiter(cross(vect1, i) for i in vect2.args)
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            n1 = vect1.args[0]
            n2 = vect2.args[0]
            if n1 == n2:
                return Vector.zero
            n3 = ({0,1,2}.difference({n1, n2})).pop()
            sign = 1 if ((n1 + 1) % 3 == n2) else -1
            return sign*vect1._sys.base_vectors()[n3]
        from .functions import express
        try:
            v = express(vect1, vect2._sys)
        except ValueError:
            return Cross(vect1, vect2)
        else:
            return cross(v, vect2)
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return Vector.zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1*cross(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2*cross(vect1, v2)

    return Cross(vect1, vect2)


def dot(vect1, vect2):
    """
    Returns dot product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import dot
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> dot(v1, v2)
    R.x + R.y + R.z

    """
    if isinstance(vect1, Add):
        return Add.fromiter(dot(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add):
        return Add.fromiter(dot(vect1, i) for i in vect2.args)
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            return S.One if vect1 == vect2 else S.Zero
        from .functions import express
        try:
            v = express(vect2, vect1._sys)
        except ValueError:
            return Dot(vect1, vect2)
        else:
            return dot(vect1, v)
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return S.Zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1*dot(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2*dot(vect1, v2)

    return Dot(vect1, vect2)


Vector._expr_type = Vector
Vector._mul_func = VectorMul
Vector._add_func = VectorAdd
Vector._zero_func = VectorZero
Vector._base_func = BaseVector
Vector.zero = VectorZero()
<|endoftext|>from sympy.core import Basic, diff
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.matrices import Matrix
from sympy.integrals import Integral, integrate
from sympy.geometry.entity import GeometryEntity
from sympy.simplify.simplify import simplify
from sympy.utilities.iterables import topological_sort
from sympy.vector import (CoordSys3D, Vector, ParametricRegion,
                        parametric_region_list, ImplicitRegion)
from sympy.vector.operators import _get_coord_systems


class ParametricIntegral(Basic):
    """
    Represents integral of a scalar or vector field
    over a Parametric Region

    Examples
    ========

    >>> from sympy import cos, sin, pi
    >>> from sympy.vector import CoordSys3D, ParametricRegion, ParametricIntegral
    >>> from sympy.abc import r, t, theta, phi

    >>> C = CoordSys3D('C')
    >>> curve = ParametricRegion((3*t - 2, t + 1), (t, 1, 2))
    >>> ParametricIntegral(C.x, curve)
    5*sqrt(10)/2
    >>> length = ParametricIntegral(1, curve)
    >>> length
    sqrt(10)
    >>> semisphere = ParametricRegion((2*sin(phi)*cos(theta), 2*sin(phi)*sin(theta), 2*cos(phi)),\
                            (theta, 0, 2*pi), (phi, 0, pi/2))
    >>> ParametricIntegral(C.z, semisphere)
    8*pi

    >>> ParametricIntegral(C.j + C.k, ParametricRegion((r*cos(theta), r*sin(theta)), r, theta))
    0

    """

    def __new__(cls, field, parametricregion):

        coord_set = _get_coord_systems(field)

        if len(coord_set) == 0:
            coord_sys = CoordSys3D('C')
        elif len(coord_set) > 1:
            raise ValueError
        else:
            coord_sys = next(iter(coord_set))

        if parametricregion.dimensions == 0:
            return S.Zero

        base_vectors = coord_sys.base_vectors()
        base_scalars = coord_sys.base_scalars()

        parametricfield = field

        r = Vector.zero
        for i in range(len(parametricregion.definition)):
            r += base_vectors[i]*parametricregion.definition[i]

        if len(coord_set) != 0:
            for i in range(len(parametricregion.definition)):
                parametricfield = parametricfield.subs(base_scalars[i], parametricregion.definition[i])

        if parametricregion.dimensions == 1:
            parameter = parametricregion.parameters[0]

            r_diff = diff(r, parameter)
            lower, upper = parametricregion.limits[parameter][0], parametricregion.limits[parameter][1]

            if isinstance(parametricfield, Vector):
                integrand = simplify(r_diff.dot(parametricfield))
            else:
                integrand = simplify(r_diff.magnitude()*parametricfield)

            result = integrate(integrand, (parameter, lower, upper))

        elif parametricregion.dimensions == 2:
            u, v = cls._bounds_case(parametricregion.parameters, parametricregion.limits)

            r_u = diff(r, u)
            r_v = diff(r, v)
            normal_vector = simplify(r_u.cross(r_v))

            if isinstance(parametricfield, Vector):
                integrand = parametricfield.dot(normal_vector)
            else:
                integrand = parametricfield*normal_vector.magnitude()

            integrand = simplify(integrand)

            lower_u, upper_u = parametricregion.limits[u][0], parametricregion.limits[u][1]
            lower_v, upper_v = parametricregion.limits[v][0], parametricregion.limits[v][1]

            result = integrate(integrand, (u, lower_u, upper_u), (v, lower_v, upper_v))

        else:
            variables = cls._bounds_case(parametricregion.parameters, parametricregion.limits)
            coeff = Matrix(parametricregion.definition).jacobian(variables).det()
            integrand = simplify(parametricfield*coeff)

            l = [(var, parametricregion.limits[var][0], parametricregion.limits[var][1]) for var in variables]
            result = integrate(integrand, *l)

        if not isinstance(result, Integral):
            return result
        else:
            return super().__new__(cls, field, parametricregion)

    @classmethod
    def _bounds_case(cls, parameters, limits):

        V = list(limits.keys())
        E = []

        for p in V:
            lower_p = limits[p][0]
            upper_p = limits[p][1]

            lower_p = lower_p.atoms()
            upper_p = upper_p.atoms()
            E.extend((p, q) for q in V if p != q and
                     (lower_p.issuperset({q}) or upper_p.issuperset({q})))

        if not E:
            return parameters
        else:
            return topological_sort((V, E), key=default_sort_key)

    @property
    def field(self):
        return self.args[0]

    @property
    def parametricregion(self):
        return self.args[1]


def vector_integrate(field, *region):
    """
    Compute the integral of a vector/scalar field
    over a a region or a set of parameters.

    Examples
    ========
    >>> from sympy.vector import CoordSys3D, ParametricRegion, vector_integrate
    >>> from sympy.abc import x, y, t
    >>> C = CoordSys3D('C')

    >>> region = ParametricRegion((t, t**2), (t, 1, 5))
    >>> vector_integrate(C.x*C.i, region)
    12

    Integrals over some objects of geometry module can also be calculated.

    >>> from sympy.geometry import Point, Circle, Triangle
    >>> c = Circle(Point(0, 2), 5)
    >>> vector_integrate(C.x**2 + C.y**2, c)
    290*pi
    >>> triangle = Triangle(Point(-2, 3), Point(2, 3), Point(0, 5))
    >>> vector_integrate(3*C.x**2*C.y*C.i + C.j, triangle)
    -8

    Integrals over some simple implicit regions can be computed. But in most cases,
    it takes too long to compute over them. This is due to the expressions of parametric
    representation becoming large.

    >>> from sympy.vector import ImplicitRegion
    >>> c2 = ImplicitRegion((x, y), (x - 2)**2 + (y - 1)**2 - 9)
    >>> vector_integrate(1, c2)
    6*pi

    Integral of fields with respect to base scalars:

    >>> vector_integrate(12*C.y**3, (C.y, 1, 3))
    240
    >>> vector_integrate(C.x**2*C.z, C.x)
    C.x**3*C.z/3
    >>> vector_integrate(C.x*C.i - C.y*C.k, C.x)
    (Integral(C.x, C.x))*C.i + (Integral(-C.y, C.x))*C.k
    >>> _.doit()
    C.x**2/2*C.i + (-C.x*C.y)*C.k

    """
    if len(region) == 1:
        if isinstance(region[0], ParametricRegion):
            return ParametricIntegral(field, region[0])

        if isinstance(region[0], ImplicitRegion):
            region = parametric_region_list(region[0])[0]
            return vector_integrate(field, region)

        if isinstance(region[0], GeometryEntity):
            regions_list = parametric_region_list(region[0])

            result = 0
            for reg in regions_list:
                result += vector_integrate(field, reg)
            return result

    return integrate(field, *region)
<|endoftext|>from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import gcd
from sympy.sets.sets import Complement
from sympy.core import Basic, Tuple, diff, expand, Eq, Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol
from sympy.solvers import solveset, nonlinsolve, diophantine
from sympy.polys import total_degree
from sympy.geometry import Point
from sympy.ntheory.factor_ import core


class ImplicitRegion(Basic):
    """
    Represents an implicit region in space.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z, t
    >>> from sympy.vector import ImplicitRegion

    >>> ImplicitRegion((x, y), x**2 + y**2 - 4)
    ImplicitRegion((x, y), x**2 + y**2 - 4)
    >>> ImplicitRegion((x, y), Eq(y*x, 1))
    ImplicitRegion((x, y), x*y - 1)

    >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
    >>> parabola.degree
    2
    >>> parabola.equation
    -4*x + y**2
    >>> parabola.rational_parametrization(t)
    (4/t**2, 4/t)

    >>> r = ImplicitRegion((x, y, z), Eq(z, x**2 + y**2))
    >>> r.variables
    (x, y, z)
    >>> r.singular_points()
    EmptySet
    >>> r.regular_point()
    (-10, -10, 200)

    Parameters
    ==========

    variables : tuple to map variables in implicit equation to base scalars.

    equation : An expression or Eq denoting the implicit equation of the region.

    """
    def __new__(cls, variables, equation):
        if not isinstance(variables, Tuple):
            variables = Tuple(*variables)

        if isinstance(equation, Eq):
            equation = equation.lhs - equation.rhs

        return super().__new__(cls, variables, equation)

    @property
    def variables(self):
        return self.args[0]

    @property
    def equation(self):
        return self.args[1]

    @property
    def degree(self):
        return total_degree(self.equation)

    def regular_point(self):
        """
        Returns a point on the implicit region.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> circle = ImplicitRegion((x, y), (x + 2)**2 + (y - 3)**2 - 16)
        >>> circle.regular_point()
        (-2, -1)
        >>> parabola = ImplicitRegion((x, y), x**2 - 4*y)
        >>> parabola.regular_point()
        (0, 0)
        >>> r = ImplicitRegion((x, y, z), (x + y + z)**4)
        >>> r.regular_point()
        (-10, -10, 20)

        References
        ==========

        - Erik Hillgarter, "Rational Points on Conics", Diploma Thesis, RISC-Linz,
          J. Kepler Universitat Linz, 1996. Available:
          https://www3.risc.jku.at/publications/download/risc_1355/Rational%20Points%20on%20Conics.pdf

        """
        equation = self.equation

        if len(self.variables) == 1:
            return (list(solveset(equation, self.variables[0], domain=S.Reals))[0],)
        elif len(self.variables) == 2:

            if self.degree == 2:
                coeffs = a, b, c, d, e, f = conic_coeff(self.variables, equation)

                if b**2 == 4*a*c:
                    x_reg, y_reg = self._regular_point_parabola(*coeffs)
                else:
                    x_reg, y_reg = self._regular_point_ellipse(*coeffs)
                return x_reg, y_reg

        if len(self.variables) == 3:
            x, y, z = self.variables

            for x_reg in range(-10, 10):
                for y_reg in range(-10, 10):
                    if not solveset(equation.subs({x: x_reg, y: y_reg}), self.variables[2], domain=S.Reals).is_empty:
                        return (x_reg, y_reg, list(solveset(equation.subs({x: x_reg, y: y_reg})))[0])

        if len(self.singular_points()) != 0:
            return list[self.singular_points()][0]

        raise NotImplementedError()

    def _regular_point_parabola(self, a, b, c, d, e, f):
            ok = (a, d) != (0, 0) and (c, e) != (0, 0) and b**2 == 4*a*c and (a, c) != (0, 0)

            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            if a != 0:
                d_dash, f_dash = (4*a*e - 2*b*d, 4*a*f - d**2)
                if d_dash != 0:
                    y_reg = -f_dash/d_dash
                    x_reg = -(d + b*y_reg)/(2*a)
                else:
                    ok = False
            elif c != 0:
                d_dash, f_dash = (4*c*d - 2*b*e, 4*c*f - e**2)
                if d_dash != 0:
                    x_reg = -f_dash/d_dash
                    y_reg = -(e + b*x_reg)/(2*c)
                else:
                    ok = False

            if ok:
                return x_reg, y_reg
            else:
                raise ValueError("Rational Point on the conic does not exist")

    def _regular_point_ellipse(self, a, b, c, d, e, f):
            D = 4*a*c - b**2
            ok = D

            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            if a == 0 and c == 0:
                K = -1
                L = 4*(d*e - b*f)
            elif c != 0:
                K = D
                L = 4*c**2*d**2 - 4*b*c*d*e + 4*a*c*e**2 + 4*b**2*c*f - 16*a*c**2*f
            else:
                K = D
                L = 4*a**2*e**2 - 4*b*a*d*e + 4*b**2*a*f

            ok = L != 0 and not(K > 0 and L < 0)
            if not ok:
                raise ValueError("Rational Point on the conic does not exist")

            K = Rational(K).limit_denominator(10**12)
            L = Rational(L).limit_denominator(10**12)

            k1, k2 = K.p, K.q
            l1, l2 = L.p, L.q
            g = gcd(k2, l2)

            a1 = (l2*k2)/g
            b1 = (k1*l2)/g
            c1 = -(l1*k2)/g
            a2 = sign(a1)*core(abs(a1), 2)
            r1 = sqrt(a1/a2)
            b2 = sign(b1)*core(abs(b1), 2)
            r2 = sqrt(b1/b2)
            c2 = sign(c1)*core(abs(c1), 2)
            r3 = sqrt(c1/c2)

            g = gcd(gcd(a2, b2), c2)
            a2 = a2/g
            b2 = b2/g
            c2 = c2/g

            g1 = gcd(a2, b2)
            a2 = a2/g1
            b2 = b2/g1
            c2 = c2*g1

            g2 = gcd(a2,c2)
            a2 = a2/g2
            b2 = b2*g2
            c2 = c2/g2

            g3 = gcd(b2, c2)
            a2 = a2*g3
            b2 = b2/g3
            c2 = c2/g3

            x, y, z = symbols("x y z")
            eq = a2*x**2 + b2*y**2 + c2*z**2

            solutions = diophantine(eq)

            if len(solutions) == 0:
                raise ValueError("Rational Point on the conic does not exist")

            flag = False
            for sol in solutions:
                syms = Tuple(*sol).free_symbols
                rep = dict.fromkeys(syms, 3)
                sol_z = sol[2]

                if sol_z == 0:
                    flag = True
                    continue

                if not isinstance(sol_z, (int, Integer)):
                    syms_z = sol_z.free_symbols

                    if len(syms_z) == 1:
                        p = next(iter(syms_z))
                        p_values = Complement(S.Integers, solveset(Eq(sol_z, 0), p, S.Integers))
                        rep[p] = next(iter(p_values))

                    if len(syms_z) == 2:
                        p, q = list(ordered(syms_z))

                        for i in S.Integers:
                            subs_sol_z = sol_z.subs(p, i)
                            q_values = Complement(S.Integers, solveset(Eq(subs_sol_z, 0), q, S.Integers))

                            if not q_values.is_empty:
                                rep[p] = i
                                rep[q] = next(iter(q_values))
                                break

                    if len(syms) != 0:
                        x, y, z = tuple(s.subs(rep) for s in sol)
                    else:
                        x, y, z =   sol
                    flag = False
                    break

            if flag:
                raise ValueError("Rational Point on the conic does not exist")

            x = (x*g3)/r1
            y = (y*g2)/r2
            z = (z*g1)/r3
            x = x/z
            y = y/z

            if a == 0 and c == 0:
                x_reg = (x + y - 2*e)/(2*b)
                y_reg = (x - y - 2*d)/(2*b)
            elif c != 0:
                x_reg = (x - 2*d*c + b*e)/K
                y_reg = (y - b*x_reg - e)/(2*c)
            else:
                y_reg = (x - 2*e*a + b*d)/K
                x_reg = (y - b*y_reg - d)/(2*a)

            return x_reg, y_reg

    def singular_points(self):
        """
        Returns a set of singular points of the region.

        The singular points are those points on the region
        where all partial derivatives vanish.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y), (y-1)**2 -x**3 + 2*x**2 -x)
        >>> I.singular_points()
        {(1, 1)}

        """
        eq_list = [self.equation]
        for var in self.variables:
            eq_list += [diff(self.equation, var)]

        return nonlinsolve(eq_list, list(self.variables))

    def multiplicity(self, point):
        """
        Returns the multiplicity of a singular point on the region.

        A singular point (x,y) of region is said to be of multiplicity m
        if all the partial derivatives off to order m - 1 vanish there.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.vector import ImplicitRegion
        >>> I = ImplicitRegion((x, y, z), x**2 + y**3 - z**4)
        >>> I.singular_points()
        {(0, 0, 0)}
        >>> I.multiplicity((0, 0, 0))
        2

        """
        if isinstance(point, Point):
            point = point.args

        modified_eq = self.equation

        for i, var in enumerate(self.variables):
            modified_eq = modified_eq.subs(var, var + point[i])
        modified_eq = expand(modified_eq)

        if len(modified_eq.args) != 0:
            terms = modified_eq.args
            m = min(total_degree(term) for term in terms)
        else:
            terms = modified_eq
            m = total_degree(terms)

        return m

    def rational_parametrization(self, parameters=('t', 's'), reg_point=None):
        """
        Returns the rational parametrization of implicit region.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z, s, t
        >>> from sympy.vector import ImplicitRegion

        >>> parabola = ImplicitRegion((x, y), y**2 - 4*x)
        >>> parabola.rational_parametrization()
        (4/t**2, 4/t)

        >>> circle = ImplicitRegion((x, y), Eq(x**2 + y**2, 4))
        >>> circle.rational_parametrization()
        (4*t/(t**2 + 1), 4*t**2/(t**2 + 1) - 2)

        >>> I = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> I.rational_parametrization()
        (t**2 - 1, t*(t**2 - 1))

        >>> cubic_curve = ImplicitRegion((x, y), x**3 + x**2 - y**2)
        >>> cubic_curve.rational_parametrization(parameters=(t))
        (t**2 - 1, t*(t**2 - 1))

        >>> sphere = ImplicitRegion((x, y, z), x**2 + y**2 + z**2 - 4)
        >>> sphere.rational_parametrization(parameters=(t, s))
        (-2 + 4/(s**2 + t**2 + 1), 4*s/(s**2 + t**2 + 1), 4*t/(s**2 + t**2 + 1))

        For some conics, regular_points() is unable to find a point on curve.
        To calulcate the parametric representation in such cases, user need
        to determine a point on the region and pass it using reg_point.

        >>> c = ImplicitRegion((x, y), (x  - 1/2)**2 + (y)**2 - (1/4)**2)
        >>> c.rational_parametrization(reg_point=(3/4, 0))
        (0.75 - 0.5/(t**2 + 1), -0.5*t/(t**2 + 1))

        References
        ==========

        - Christoph M. Hoffmann, "Conversion Methods between Parametric and
          Implicit Curves and Surfaces", Purdue e-Pubs, 1990. Available:
          https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1827&context=cstech

        """
        equation = self.equation
        degree = self.degree

        if degree == 1:
            if len(self.variables) == 1:
                return (equation,)
            elif len(self.variables) == 2:
                x, y = self.variables
                y_par = list(solveset(equation, y))[0]
                return x, y_par
            else:
                raise NotImplementedError()

        point = ()

        # Finding the (n - 1) fold point of the monoid of degree
        if degree == 2:
            # For degree 2 curves, either a regular point or a singular point can be used.
            if reg_point is not None:
                # Using point provided by the user as regular point
                point = reg_point
            else:
                if len(self.singular_points()) != 0:
                    point = list(self.singular_points())[0]
                else:
                    point = self.regular_point()

        if len(self.singular_points()) != 0:
            singular_points = self.singular_points()
            for spoint in singular_points:
                syms = Tuple(*spoint).free_symbols
                rep = dict.fromkeys(syms, 2)

                if len(syms) != 0:
                    spoint = tuple(s.subs(rep) for s in spoint)

                if self.multiplicity(spoint) == degree - 1:
                    point = spoint
                    break

        if len(point) == 0:
            # The region in not a monoid
            raise NotImplementedError()

        modified_eq = equation

        # Shifting the region such that fold point moves to origin
        for i, var in enumerate(self.variables):
            modified_eq = modified_eq.subs(var, var + point[i])
        modified_eq = expand(modified_eq)

        hn = hn_1 = 0
        for term in modified_eq.args:
            if total_degree(term) == degree:
                hn += term
            else:
                hn_1 += term

        hn_1 = -1*hn_1

        if not isinstance(parameters, tuple):
            parameters = (parameters,)

        if len(self.variables) == 2:

            parameter1 = parameters[0]
            if parameter1 == 's':
                # To avoid name conflict between parameters
                s = _symbol('s_', real=True)
            else:
                s = _symbol('s', real=True)
            t = _symbol(parameter1, real=True)

            hn = hn.subs({self.variables[0]: s, self.variables[1]: t})
            hn_1 = hn_1.subs({self.variables[0]: s, self.variables[1]: t})

            x_par = (s*(hn_1/hn)).subs(s, 1) + point[0]
            y_par = (t*(hn_1/hn)).subs(s, 1) + point[1]

            return x_par, y_par

        elif len(self.variables) == 3:

            parameter1, parameter2 = parameters
            if 'r' in parameters:
                # To avoid name conflict between parameters
                r = _symbol('r_', real=True)
            else:
                r = _symbol('r', real=True)
            s = _symbol(parameter2, real=True)
            t = _symbol(parameter1, real=True)

            hn = hn.subs({self.variables[0]: r, self.variables[1]: s, self.variables[2]: t})
            hn_1 = hn_1.subs({self.variables[0]: r, self.variables[1]: s, self.variables[2]: t})

            x_par = (r*(hn_1/hn)).subs(r, 1) + point[0]
            y_par = (s*(hn_1/hn)).subs(r, 1) + point[1]
            z_par = (t*(hn_1/hn)).subs(r, 1) + point[2]

            return x_par, y_par, z_par

        raise NotImplementedError()

def conic_coeff(variables, equation):
    if total_degree(equation) != 2:
        raise ValueError()
    x = variables[0]
    y = variables[1]

    equation = expand(equation)
    a = equation.coeff(x**2)
    b = equation.coeff(x*y)
    c = equation.coeff(y**2)
    d = equation.coeff(x, 1).coeff(y, 0)
    e = equation.coeff(y, 1).coeff(x, 0)
    f = equation.coeff(x, 0).coeff(y, 0)
    return a, b, c, d, e, f
<|endoftext|>from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.deloperator import Del
from sympy.vector.scalar import BaseScalar
from sympy.vector.vector import Vector, BaseVector
from sympy.vector.operators import gradient, curl, divergence
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.integrals.integrals import integrate
from sympy.core import sympify
from sympy.vector.dyadic import Dyadic


def express(expr, system, system2=None, variables=False):
    """
    Global function for 'express' functionality.

    Re-expresses a Vector, Dyadic or scalar(sympyfiable) in the given
    coordinate system.

    If 'variables' is True, then the coordinate variables (base scalars)
    of other coordinate systems present in the vector/scalar field or
    dyadic are also substituted in terms of the base scalars of the
    given system.

    Parameters
    ==========

    expr : Vector/Dyadic/scalar(sympyfiable)
        The expression to re-express in CoordSys3D 'system'

    system: CoordSys3D
        The coordinate system the expr is to be expressed in

    system2: CoordSys3D
        The other coordinate system required for re-expression
        (only for a Dyadic Expr)

    variables : boolean
        Specifies whether to substitute the coordinate variables present
        in expr, in terms of those of parameter system

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy import Symbol, cos, sin
    >>> N = CoordSys3D('N')
    >>> q = Symbol('q')
    >>> B = N.orient_new_axis('B', q, N.k)
    >>> from sympy.vector import express
    >>> express(B.i, N)
    (cos(q))*N.i + (sin(q))*N.j
    >>> express(N.x, B, variables=True)
    B.x*cos(q) - B.y*sin(q)
    >>> d = N.i.outer(N.i)
    >>> express(d, B, N) == (cos(q))*(B.i|N.i) + (-sin(q))*(B.j|N.i)
    True

    """

    if expr in (0, Vector.zero):
        return expr

    if not isinstance(system, CoordSys3D):
        raise TypeError("system should be a CoordSys3D \
                        instance")

    if isinstance(expr, Vector):
        if system2 is not None:
            raise ValueError("system2 should not be provided for \
                                Vectors")
        # Given expr is a Vector
        if variables:
            # If variables attribute is True, substitute
            # the coordinate variables in the Vector
            system_list = {x.system for x in expr.atoms(BaseScalar, BaseVector)} - {system}
            subs_dict = {}
            for f in system_list:
                subs_dict.update(f.scalar_map(system))
            expr = expr.subs(subs_dict)
        # Re-express in this coordinate system
        outvec = Vector.zero
        parts = expr.separate()
        for x in parts:
            if x != system:
                temp = system.rotation_matrix(x) * parts[x].to_matrix(x)
                outvec += matrix_to_vector(temp, system)
            else:
                outvec += parts[x]
        return outvec

    elif isinstance(expr, Dyadic):
        if system2 is None:
            system2 = system
        if not isinstance(system2, CoordSys3D):
            raise TypeError("system2 should be a CoordSys3D \
                            instance")
        outdyad = Dyadic.zero
        var = variables
        for k, v in expr.components.items():
            outdyad += (express(v, system, variables=var) *
                        (express(k.args[0], system, variables=var) |
                         express(k.args[1], system2, variables=var)))

        return outdyad

    else:
        if system2 is not None:
            raise ValueError("system2 should not be provided for \
                                Vectors")
        if variables:
            # Given expr is a scalar field
            system_set = set()
            expr = sympify(expr)
            # Substitute all the coordinate variables
            for x in expr.atoms(BaseScalar):
                if x.system != system:
                    system_set.add(x.system)
            subs_dict = {}
            for f in system_set:
                subs_dict.update(f.scalar_map(system))
            return expr.subs(subs_dict)
        return expr


def directional_derivative(field, direction_vector):
    """
    Returns the directional derivative of a scalar or vector field computed
    along a given vector in coordinate system which parameters are expressed.

    Parameters
    ==========

    field : Vector or Scalar
        The scalar or vector field to compute the directional derivative of

    direction_vector : Vector
        The vector to calculated directional derivative along them.


    Examples
    ========

    >>> from sympy.vector import CoordSys3D, directional_derivative
    >>> R = CoordSys3D('R')
    >>> f1 = R.x*R.y*R.z
    >>> v1 = 3*R.i + 4*R.j + R.k
    >>> directional_derivative(f1, v1)
    R.x*R.y + 4*R.x*R.z + 3*R.y*R.z
    >>> f2 = 5*R.x**2*R.z
    >>> directional_derivative(f2, v1)
    5*R.x**2 + 30*R.x*R.z

    """
    from sympy.vector.operators import _get_coord_systems
    coord_sys = _get_coord_systems(field)
    if len(coord_sys) > 0:
        # TODO: This gets a random coordinate system in case of multiple ones:
        coord_sys = next(iter(coord_sys))
        field = express(field, coord_sys, variables=True)
        i, j, k = coord_sys.base_vectors()
        x, y, z = coord_sys.base_scalars()
        out = Vector.dot(direction_vector, i) * diff(field, x)
        out += Vector.dot(direction_vector, j) * diff(field, y)
        out += Vector.dot(direction_vector, k) * diff(field, z)
        if out == 0 and isinstance(field, Vector):
            out = Vector.zero
        return out
    elif isinstance(field, Vector):
        return Vector.zero
    else:
        return S.Zero


def laplacian(expr):
    """
    Return the laplacian of the given field computed in terms of
    the base scalars of the given coordinate system.

    Parameters
    ==========

    expr : SymPy Expr or Vector
        expr denotes a scalar or vector field.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, laplacian
    >>> R = CoordSys3D('R')
    >>> f = R.x**2*R.y**5*R.z
    >>> laplacian(f)
    20*R.x**2*R.y**3*R.z + 2*R.y**5*R.z
    >>> f = R.x**2*R.i + R.y**3*R.j + R.z**4*R.k
    >>> laplacian(f)
    2*R.i + 6*R.y*R.j + 12*R.z**2*R.k

    """

    delop = Del()
    if expr.is_Vector:
        return (gradient(divergence(expr)) - curl(curl(expr))).doit()
    return delop.dot(delop(expr)).doit()


def is_conservative(field):
    """
    Checks if a field is conservative.

    Parameters
    ==========

    field : Vector
        The field to check for conservative property

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector import is_conservative
    >>> R = CoordSys3D('R')
    >>> is_conservative(R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k)
    True
    >>> is_conservative(R.z*R.j)
    False

    """

    # Field is conservative irrespective of system
    # Take the first coordinate system in the result of the
    # separate method of Vector
    if not isinstance(field, Vector):
        raise TypeError("field should be a Vector")
    if field == Vector.zero:
        return True
    return curl(field).simplify() == Vector.zero


def is_solenoidal(field):
    """
    Checks if a field is solenoidal.

    Parameters
    ==========

    field : Vector
        The field to check for solenoidal property

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector import is_solenoidal
    >>> R = CoordSys3D('R')
    >>> is_solenoidal(R.y*R.z*R.i + R.x*R.z*R.j + R.x*R.y*R.k)
    True
    >>> is_solenoidal(R.y * R.j)
    False

    """

    # Field is solenoidal irrespective of system
    # Take the first coordinate system in the result of the
    # separate method in Vector
    if not isinstance(field, Vector):
        raise TypeError("field should be a Vector")
    if field == Vector.zero:
        return True
    return divergence(field).simplify() is S.Zero


def scalar_potential(field, coord_sys):
    """
    Returns the scalar potential function of a field in a given
    coordinate system (without the added integration constant).

    Parameters
    ==========

    field : Vector
        The vector field whose scalar potential function is to be
        calculated

    coord_sys : CoordSys3D
        The coordinate system to do the calculation in

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector import scalar_potential, gradient
    >>> R = CoordSys3D('R')
    >>> scalar_potential(R.k, R) == R.z
    True
    >>> scalar_field = 2*R.x**2*R.y*R.z
    >>> grad_field = gradient(scalar_field)
    >>> scalar_potential(grad_field, R)
    2*R.x**2*R.y*R.z

    """

    # Check whether field is conservative
    if not is_conservative(field):
        raise ValueError("Field is not conservative")
    if field == Vector.zero:
        return S.Zero
    # Express the field exntirely in coord_sys
    # Substitute coordinate variables also
    if not isinstance(coord_sys, CoordSys3D):
        raise TypeError("coord_sys must be a CoordSys3D")
    field = express(field, coord_sys, variables=True)
    dimensions = coord_sys.base_vectors()
    scalars = coord_sys.base_scalars()
    # Calculate scalar potential function
    temp_function = integrate(field.dot(dimensions[0]), scalars[0])
    for i, dim in enumerate(dimensions[1:]):
        partial_diff = diff(temp_function, scalars[i + 1])
        partial_diff = field.dot(dim) - partial_diff
        temp_function += integrate(partial_diff, scalars[i + 1])
    return temp_function


def scalar_potential_difference(field, coord_sys, point1, point2):
    """
    Returns the scalar potential difference between two points in a
    certain coordinate system, wrt a given field.

    If a scalar field is provided, its values at the two points are
    considered. If a conservative vector field is provided, the values
    of its scalar potential function at the two points are used.

    Returns (potential at point2) - (potential at point1)

    The position vectors of the two Points are calculated wrt the
    origin of the coordinate system provided.

    Parameters
    ==========

    field : Vector/Expr
        The field to calculate wrt

    coord_sys : CoordSys3D
        The coordinate system to do the calculations in

    point1 : Point
        The initial Point in given coordinate system

    position2 : Point
        The second Point in the given coordinate system

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector import scalar_potential_difference
    >>> R = CoordSys3D('R')
    >>> P = R.origin.locate_new('P', R.x*R.i + R.y*R.j + R.z*R.k)
    >>> vectfield = 4*R.x*R.y*R.i + 2*R.x**2*R.j
    >>> scalar_potential_difference(vectfield, R, R.origin, P)
    2*R.x**2*R.y
    >>> Q = R.origin.locate_new('O', 3*R.i + R.j + 2*R.k)
    >>> scalar_potential_difference(vectfield, R, P, Q)
    -2*R.x**2*R.y + 18

    """

    if not isinstance(coord_sys, CoordSys3D):
        raise TypeError("coord_sys must be a CoordSys3D")
    if isinstance(field, Vector):
        # Get the scalar potential function
        scalar_fn = scalar_potential(field, coord_sys)
    else:
        # Field is a scalar
        scalar_fn = field
    # Express positions in required coordinate system
    origin = coord_sys.origin
    position1 = express(point1.position_wrt(origin), coord_sys,
                        variables=True)
    position2 = express(point2.position_wrt(origin), coord_sys,
                        variables=True)
    # Get the two positions as substitution dicts for coordinate variables
    subs_dict1 = {}
    subs_dict2 = {}
    scalars = coord_sys.base_scalars()
    for i, x in enumerate(coord_sys.base_vectors()):
        subs_dict1[scalars[i]] = x.dot(position1)
        subs_dict2[scalars[i]] = x.dot(position2)
    return scalar_fn.subs(subs_dict2) - scalar_fn.subs(subs_dict1)


def matrix_to_vector(matrix, system):
    """
    Converts a vector in matrix form to a Vector instance.

    It is assumed that the elements of the Matrix represent the
    measure numbers of the components of the vector along basis
    vectors of 'system'.

    Parameters
    ==========

    matrix : SymPy Matrix, Dimensions: (3, 1)
        The matrix to be converted to a vector

    system : CoordSys3D
        The coordinate system the vector is to be defined in

    Examples
    ========

    >>> from sympy import ImmutableMatrix as Matrix
    >>> m = Matrix([1, 2, 3])
    >>> from sympy.vector import CoordSys3D, matrix_to_vector
    >>> C = CoordSys3D('C')
    >>> v = matrix_to_vector(m, C)
    >>> v
    C.i + 2*C.j + 3*C.k
    >>> v.to_matrix(C) == m
    True

    """

    outvec = Vector.zero
    vects = system.base_vectors()
    for i, x in enumerate(matrix):
        outvec += x * vects[i]
    return outvec


def _path(from_object, to_object):
    """
    Calculates the 'path' of objects starting from 'from_object'
    to 'to_object', along with the index of the first common
    ancestor in the tree.

    Returns (index, list) tuple.
    """

    if from_object._root != to_object._root:
        raise ValueError("No connecting path found between " +
                         str(from_object) + " and " + str(to_object))

    other_path = []
    obj = to_object
    while obj._parent is not None:
        other_path.append(obj)
        obj = obj._parent
    other_path.append(obj)
    object_set = set(other_path)
    from_path = []
    obj = from_object
    while obj not in object_set:
        from_path.append(obj)
        obj = obj._parent
    index = len(from_path)
    from_path.extend(other_path[other_path.index(obj)::-1])
    return index, from_path


def orthogonalize(*vlist, orthonormal=False):
    """
    Takes a sequence of independent vectors and orthogonalizes them
    using the Gram - Schmidt process. Returns a list of
    orthogonal or orthonormal vectors.

    Parameters
    ==========

    vlist : sequence of independent vectors to be made orthogonal.

    orthonormal : Optional parameter
                  Set to True if the vectors returned should be
                  orthonormal.
                  Default: False

    Examples
    ========

    >>> from sympy.vector.coordsysrect import CoordSys3D
    >>> from sympy.vector.functions import orthogonalize
    >>> C = CoordSys3D('C')
    >>> i, j, k = C.base_vectors()
    >>> v1 = i + 2*j
    >>> v2 = 2*i + 3*j
    >>> orthogonalize(v1, v2)
    [C.i + 2*C.j, 2/5*C.i + (-1/5)*C.j]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gram-Schmidt_process

    """

    if not all(isinstance(vec, Vector) for vec in vlist):
        raise TypeError('Each element must be of Type Vector')

    ortho_vlist = []
    for i, term in enumerate(vlist):
        for j in range(i):
            term -= ortho_vlist[j].projection(vlist[i])
        # TODO : The following line introduces a performance issue
        # and needs to be changed once a good solution for issue #10279 is
        # found.
        if term.equals(Vector.zero):
            raise ValueError("Vector set not linearly independent")
        ortho_vlist.append(term)

    if orthonormal:
        ortho_vlist = [vec.normalize() for vec in ortho_vlist]

    return ortho_vlist
<|endoftext|>from functools import singledispatch
from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import tan
from sympy.simplify import trigsimp
from sympy.core import Basic, Tuple
from sympy.core.symbol import _symbol
from sympy.solvers import solve
from sympy.geometry import Point, Segment, Curve, Ellipse, Polygon
from sympy.vector import ImplicitRegion


class ParametricRegion(Basic):
    """
    Represents a parametric region in space.

    Examples
    ========

    >>> from sympy import cos, sin, pi
    >>> from sympy.abc import r, theta, t, a, b, x, y
    >>> from sympy.vector import ParametricRegion

    >>> ParametricRegion((t, t**2), (t, -1, 2))
    ParametricRegion((t, t**2), (t, -1, 2))
    >>> ParametricRegion((x, y), (x, 3, 4), (y, 5, 6))
    ParametricRegion((x, y), (x, 3, 4), (y, 5, 6))
    >>> ParametricRegion((r*cos(theta), r*sin(theta)), (r, -2, 2), (theta, 0, pi))
    ParametricRegion((r*cos(theta), r*sin(theta)), (r, -2, 2), (theta, 0, pi))
    >>> ParametricRegion((a*cos(t), b*sin(t)), t)
    ParametricRegion((a*cos(t), b*sin(t)), t)

    >>> circle = ParametricRegion((r*cos(theta), r*sin(theta)), r, (theta, 0, pi))
    >>> circle.parameters
    (r, theta)
    >>> circle.definition
    (r*cos(theta), r*sin(theta))
    >>> circle.limits
    {theta: (0, pi)}

    Dimension of a parametric region determines whether a region is a curve, surface
    or volume region. It does not represent its dimensions in space.

    >>> circle.dimensions
    1

    Parameters
    ==========

    definition : tuple to define base scalars in terms of parameters.

    bounds : Parameter or a tuple of length 3 to define parameter and corresponding lower and upper bound.

    """
    def __new__(cls, definition, *bounds):
        parameters = ()
        limits = {}

        if not isinstance(bounds, Tuple):
            bounds = Tuple(*bounds)

        for bound in bounds:
            if isinstance(bound, (tuple, Tuple)):
                if len(bound) != 3:
                    raise ValueError("Tuple should be in the form (parameter, lowerbound, upperbound)")
                parameters += (bound[0],)
                limits[bound[0]] = (bound[1], bound[2])
            else:
                parameters += (bound,)

        if not isinstance(definition, (tuple, Tuple)):
            definition = (definition,)

        obj = super().__new__(cls, Tuple(*definition), *bounds)
        obj._parameters = parameters
        obj._limits = limits

        return obj

    @property
    def definition(self):
        return self.args[0]

    @property
    def limits(self):
        return self._limits

    @property
    def parameters(self):
        return self._parameters

    @property
    def dimensions(self):
        return len(self.limits)


@singledispatch
def parametric_region_list(reg):
    """
    Returns a list of ParametricRegion objects representing the geometric region.

    Examples
    ========

    >>> from sympy.abc import t
    >>> from sympy.vector import parametric_region_list
    >>> from sympy.geometry import Point, Curve, Ellipse, Segment, Polygon

    >>> p = Point(2, 5)
    >>> parametric_region_list(p)
    [ParametricRegion((2, 5))]

    >>> c = Curve((t**3, 4*t), (t, -3, 4))
    >>> parametric_region_list(c)
    [ParametricRegion((t**3, 4*t), (t, -3, 4))]

    >>> e = Ellipse(Point(1, 3), 2, 3)
    >>> parametric_region_list(e)
    [ParametricRegion((2*cos(t) + 1, 3*sin(t) + 3), (t, 0, 2*pi))]

    >>> s = Segment(Point(1, 3), Point(2, 6))
    >>> parametric_region_list(s)
    [ParametricRegion((t + 1, 3*t + 3), (t, 0, 1))]

    >>> p1, p2, p3, p4 = [(0, 1), (2, -3), (5, 3), (-2, 3)]
    >>> poly = Polygon(p1, p2, p3, p4)
    >>> parametric_region_list(poly)
    [ParametricRegion((2*t, 1 - 4*t), (t, 0, 1)), ParametricRegion((3*t + 2, 6*t - 3), (t, 0, 1)),\
     ParametricRegion((5 - 7*t, 3), (t, 0, 1)), ParametricRegion((2*t - 2, 3 - 2*t),  (t, 0, 1))]

    """
    raise ValueError("SymPy cannot determine parametric representation of the region.")


@parametric_region_list.register(Point)
def _(obj):
    return [ParametricRegion(obj.args)]


@parametric_region_list.register(Curve)  # type: ignore
def _(obj):
    definition = obj.arbitrary_point(obj.parameter).args
    bounds = obj.limits
    return [ParametricRegion(definition, bounds)]


@parametric_region_list.register(Ellipse) # type: ignore
def _(obj, parameter='t'):
    definition = obj.arbitrary_point(parameter).args
    t = _symbol(parameter, real=True)
    bounds = (t, 0, 2*pi)
    return [ParametricRegion(definition, bounds)]


@parametric_region_list.register(Segment) # type: ignore
def _(obj, parameter='t'):
    t = _symbol(parameter, real=True)
    definition = obj.arbitrary_point(t).args

    for i in range(0, 3):
        lower_bound = solve(definition[i] - obj.points[0].args[i], t)
        upper_bound = solve(definition[i] - obj.points[1].args[i], t)

        if len(lower_bound) == 1 and len(upper_bound) == 1:
            bounds = t, lower_bound[0], upper_bound[0]
            break

    definition_tuple = obj.arbitrary_point(parameter).args
    return [ParametricRegion(definition_tuple, bounds)]


@parametric_region_list.register(Polygon) # type: ignore
def _(obj, parameter='t'):
    l = [parametric_region_list(side, parameter)[0] for side in obj.sides]
    return l


@parametric_region_list.register(ImplicitRegion) # type: ignore
def _(obj, parameters=('t', 's')):
    definition = obj.rational_parametrization(parameters)
    bounds = []

    for i in range(len(obj.variables) - 1):
        # Each parameter is replaced by its tangent to simplify integration
        parameter = _symbol(parameters[i], real=True)
        definition = [trigsimp(elem.subs(parameter, tan(parameter/2))) for elem in definition]
        bounds.append((parameter, 0, 2*pi),)

    definition = Tuple(*definition)
    return [ParametricRegion(definition, *bounds)]
<|endoftext|>from __future__ import annotations

from sympy.vector.basisdependent import (BasisDependent, BasisDependentAdd,
                                         BasisDependentMul, BasisDependentZero)
from sympy.core import S, Pow
from sympy.core.expr import AtomicExpr
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
import sympy.vector


class Dyadic(BasisDependent):
    """
    Super class for all Dyadic-classes.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dyadic_tensor
    .. [2] Kane, T., Levinson, D. Dynamics Theory and Applications. 1985
           McGraw-Hill

    """

    _op_priority = 13.0

    _expr_type: type[Dyadic]
    _mul_func: type[Dyadic]
    _add_func: type[Dyadic]
    _zero_func: type[Dyadic]
    _base_func: type[Dyadic]
    zero: DyadicZero

    @property
    def components(self):
        """
        Returns the components of this dyadic in the form of a
        Python dictionary mapping BaseDyadic instances to the
        corresponding measure numbers.

        """
        # The '_components' attribute is defined according to the
        # subclass of Dyadic the instance belongs to.
        return self._components

    def dot(self, other):
        """
        Returns the dot product(also called inner product) of this
        Dyadic, with another Dyadic or Vector.
        If 'other' is a Dyadic, this returns a Dyadic. Else, it returns
        a Vector (unless an error is encountered).

        Parameters
        ==========

        other : Dyadic/Vector
            The other Dyadic or Vector to take the inner product with

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> D1 = N.i.outer(N.j)
        >>> D2 = N.j.outer(N.j)
        >>> D1.dot(D2)
        (N.i|N.j)
        >>> D1.dot(N.j)
        N.i

        """

        Vector = sympy.vector.Vector
        if isinstance(other, BasisDependentZero):
            return Vector.zero
        elif isinstance(other, Vector):
            outvec = Vector.zero
            for k, v in self.components.items():
                vect_dot = k.args[1].dot(other)
                outvec += vect_dot * v * k.args[0]
            return outvec
        elif isinstance(other, Dyadic):
            outdyad = Dyadic.zero
            for k1, v1 in self.components.items():
                for k2, v2 in other.components.items():
                    vect_dot = k1.args[1].dot(k2.args[0])
                    outer_product = k1.args[0].outer(k2.args[1])
                    outdyad += vect_dot * v1 * v2 * outer_product
            return outdyad
        else:
            raise TypeError("Inner product is not defined for " +
                            str(type(other)) + " and Dyadics.")

    def __and__(self, other):
        return self.dot(other)

    __and__.__doc__ = dot.__doc__

    def cross(self, other):
        """
        Returns the cross product between this Dyadic, and a Vector, as a
        Vector instance.

        Parameters
        ==========

        other : Vector
            The Vector that we are crossing this Dyadic with

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> d = N.i.outer(N.i)
        >>> d.cross(N.j)
        (N.i|N.k)

        """

        Vector = sympy.vector.Vector
        if other == Vector.zero:
            return Dyadic.zero
        elif isinstance(other, Vector):
            outdyad = Dyadic.zero
            for k, v in self.components.items():
                cross_product = k.args[1].cross(other)
                outer = k.args[0].outer(cross_product)
                outdyad += v * outer
            return outdyad
        else:
            raise TypeError(str(type(other)) + " not supported for " +
                            "cross with dyadics")

    def __xor__(self, other):
        return self.cross(other)

    __xor__.__doc__ = cross.__doc__

    def to_matrix(self, system, second_system=None):
        """
        Returns the matrix form of the dyadic with respect to one or two
        coordinate systems.

        Parameters
        ==========

        system : CoordSys3D
            The coordinate system that the rows and columns of the matrix
            correspond to. If a second system is provided, this
            only corresponds to the rows of the matrix.
        second_system : CoordSys3D, optional, default=None
            The coordinate system that the columns of the matrix correspond
            to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> v = N.i + 2*N.j
        >>> d = v.outer(N.i)
        >>> d.to_matrix(N)
        Matrix([
        [1, 0, 0],
        [2, 0, 0],
        [0, 0, 0]])
        >>> from sympy import Symbol
        >>> q = Symbol('q')
        >>> P = N.orient_new_axis('P', q, N.k)
        >>> d.to_matrix(N, P)
        Matrix([
        [  cos(q),   -sin(q), 0],
        [2*cos(q), -2*sin(q), 0],
        [       0,         0, 0]])

        """

        if second_system is None:
            second_system = system

        return Matrix([i.dot(self).dot(j) for i in system for j in
                       second_system]).reshape(3, 3)

    def _div_helper(one, other):
        """ Helper for division involving dyadics """
        if isinstance(one, Dyadic) and isinstance(other, Dyadic):
            raise TypeError("Cannot divide two dyadics")
        elif isinstance(one, Dyadic):
            return DyadicMul(one, Pow(other, S.NegativeOne))
        else:
            raise TypeError("Cannot divide by a dyadic")


class BaseDyadic(Dyadic, AtomicExpr):
    """
    Class to denote a base dyadic tensor component.
    """

    def __new__(cls, vector1, vector2):
        Vector = sympy.vector.Vector
        BaseVector = sympy.vector.BaseVector
        VectorZero = sympy.vector.VectorZero
        # Verify arguments
        if not isinstance(vector1, (BaseVector, VectorZero)) or \
                not isinstance(vector2, (BaseVector, VectorZero)):
            raise TypeError("BaseDyadic cannot be composed of non-base " +
                            "vectors")
        # Handle special case of zero vector
        elif vector1 == Vector.zero or vector2 == Vector.zero:
            return Dyadic.zero
        # Initialize instance
        obj = super().__new__(cls, vector1, vector2)
        obj._base_instance = obj
        obj._measure_number = 1
        obj._components = {obj: S.One}
        obj._sys = vector1._sys
        obj._pretty_form = ('(' + vector1._pretty_form + '|' +
                             vector2._pretty_form + ')')
        obj._latex_form = (r'\left(' + vector1._latex_form + r"{\middle|}" +
                           vector2._latex_form + r'\right)')

        return obj

    def _sympystr(self, printer):
        return "({}|{})".format(
            printer._print(self.args[0]), printer._print(self.args[1]))

    def _sympyrepr(self, printer):
        return "BaseDyadic({}, {})".format(
            printer._print(self.args[0]), printer._print(self.args[1]))


class DyadicMul(BasisDependentMul, Dyadic):
    """ Products of scalars and BaseDyadics """

    def __new__(cls, *args, **options):
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    @property
    def base_dyadic(self):
        """ The BaseDyadic involved in the product. """
        return self._base_instance

    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this DyadicMul.
        """
        return self._measure_number


class DyadicAdd(BasisDependentAdd, Dyadic):
    """ Class to hold dyadic sums """

    def __new__(cls, *args, **options):
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    def _sympystr(self, printer):
        items = list(self.components.items())
        items.sort(key=lambda x: x[0].__str__())
        return " + ".join(printer._print(k * v) for k, v in items)


class DyadicZero(BasisDependentZero, Dyadic):
    """
    Class to denote a zero dyadic
    """

    _op_priority = 13.1
    _pretty_form = '(0|0)'
    _latex_form = r'(\mathbf{\hat{0}}|\mathbf{\hat{0}})'

    def __new__(cls):
        obj = BasisDependentZero.__new__(cls)
        return obj


Dyadic._expr_type = Dyadic
Dyadic._mul_func = DyadicMul
Dyadic._add_func = DyadicAdd
Dyadic._zero_func = DyadicZero
Dyadic._base_func = BaseDyadic
Dyadic.zero = DyadicZero()
<|endoftext|>from __future__ import annotations
from typing import TYPE_CHECKING

from sympy.simplify import simplify as simp, trigsimp as tsimp  # type: ignore
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.assumptions import StdFactKB
from sympy.core.function import diff as df
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor as fctr
from sympy.core import S, Add, Mul
from sympy.core.expr import Expr

if TYPE_CHECKING:
    from sympy.vector.vector import BaseVector


class BasisDependent(Expr):
    """
    Super class containing functionality common to vectors and
    dyadics.
    Named so because the representation of these quantities in
    sympy.vector is dependent on the basis they are expressed in.
    """

    zero: BasisDependentZero

    @call_highest_priority('__radd__')
    def __add__(self, other):
        return self._add_func(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self._add_func(other, self)

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return self._add_func(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return self._add_func(other, -self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return self._mul_func(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self._mul_func(other, self)

    def __neg__(self):
        return self._mul_func(S.NegativeOne, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self._div_helper(other)

    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return TypeError("Invalid divisor for division")

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """
        Implements the SymPy evalf routine for this quantity.

        evalf's documentation
        =====================

        """
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        vec = self.zero
        for k, v in self.components.items():
            vec += v.evalf(n, **options) * k
        return vec

    evalf.__doc__ += Expr.evalf.__doc__  # type: ignore

    n = evalf # type: ignore

    def simplify(self, **kwargs):
        """
        Implements the SymPy simplify routine for this quantity.

        simplify's documentation
        ========================

        """
        simp_components = [simp(v, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*simp_components)

    simplify.__doc__ += simp.__doc__  # type: ignore

    def trigsimp(self, **opts):
        """
        Implements the SymPy trigsimp routine, for this quantity.

        trigsimp's documentation
        ========================

        """
        trig_components = [tsimp(v, **opts) * k for
                           k, v in self.components.items()]
        return self._add_func(*trig_components)

    trigsimp.__doc__ += tsimp.__doc__  # type: ignore

    def _eval_simplify(self, **kwargs):
        return self.simplify(**kwargs)

    def _eval_trigsimp(self, **opts):
        return self.trigsimp(**opts)

    def _eval_derivative(self, wrt):
        return self.diff(wrt)

    def _eval_Integral(self, *symbols, **assumptions):
        integral_components = [Integral(v, *symbols, **assumptions) * k
                               for k, v in self.components.items()]
        return self._add_func(*integral_components)

    def as_numer_denom(self):
        """
        Returns the expression as a tuple wrt the following
        transformation -

        expression -> a/b -> a, b

        """
        return self, S.One

    def factor(self, *args, **kwargs):
        """
        Implements the SymPy factor routine, on the scalar parts
        of a basis-dependent expression.

        factor's documentation
        ========================

        """
        fctr_components = [fctr(v, *args, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*fctr_components)

    factor.__doc__ += fctr.__doc__  # type: ignore

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return (S.One, self)

    def as_coeff_add(self, *deps):
        """Efficiently extract the coefficient of a summation."""
        return 0, tuple(x * self.components[x] for x in self.components)

    def diff(self, *args, **kwargs):
        """
        Implements the SymPy diff routine, for vectors.

        diff's documentation
        ========================

        """
        for x in args:
            if isinstance(x, BasisDependent):
                raise TypeError("Invalid arg for differentiation")
        diff_components = [df(v, *args, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*diff_components)

    diff.__doc__ += df.__doc__  # type: ignore

    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""
        doit_components = [self.components[x].doit(**hints) * x
                           for x in self.components]
        return self._add_func(*doit_components)


class BasisDependentAdd(BasisDependent, Add):
    """
    Denotes sum of basis dependent quantities such that they cannot
    be expressed as base or Mul instances.
    """

    def __new__(cls, *args, **options):
        components = {}

        # Check each arg and simultaneously learn the components
        for arg in args:
            if not isinstance(arg, cls._expr_type):
                if isinstance(arg, Mul):
                    arg = cls._mul_func(*(arg.args))
                elif isinstance(arg, Add):
                    arg = cls._add_func(*(arg.args))
                else:
                    raise TypeError(str(arg) +
                                    " cannot be interpreted correctly")
            # If argument is zero, ignore
            if arg == cls.zero:
                continue
            # Else, update components accordingly
            for x in arg.components:
                components[x] = components.get(x, 0) + arg.components[x]

        temp = list(components.keys())
        for x in temp:
            if components[x] == 0:
                del components[x]

        # Handle case of zero vector
        if len(components) == 0:
            return cls.zero

        # Build object
        newargs = [x * components[x] for x in components]
        obj = super().__new__(cls, *newargs, **options)
        if isinstance(obj, Mul):
            return cls._mul_func(*obj.args)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = components
        obj._sys = (list(components.keys()))[0]._sys

        return obj


class BasisDependentMul(BasisDependent, Mul):
    """
    Denotes product of base- basis dependent quantity with a scalar.
    """

    def __new__(cls, *args, **options):
        obj = cls._new(*args, **options)
        return obj

    def _new_rawargs(self, *args):
        # XXX: This is needed because Add.flatten() uses it but the default
        # implementation does not work for Vectors because they assign
        # attributes outside of .args.
        return type(self)(*args)

    @classmethod
    def _new(cls, *args, **options):
        from sympy.vector import Cross, Dot, Curl, Gradient
        count = 0
        measure_number = S.One
        zeroflag = False
        extra_args = []

        # Determine the component and check arguments
        # Also keep a count to ensure two vectors aren't
        # being multiplied
        for arg in args:
            if isinstance(arg, cls._zero_func):
                count += 1
                zeroflag = True
            elif arg == S.Zero:
                zeroflag = True
            elif isinstance(arg, (cls._base_func, cls._mul_func)):
                count += 1
                expr = arg._base_instance
                measure_number *= arg._measure_number
            elif isinstance(arg, cls._add_func):
                count += 1
                expr = arg
            elif isinstance(arg, (Cross, Dot, Curl, Gradient)):
                extra_args.append(arg)
            else:
                measure_number *= arg
        # Make sure incompatible types weren't multiplied
        if count > 1:
            raise ValueError("Invalid multiplication")
        elif count == 0:
            return Mul(*args, **options)
        # Handle zero vector case
        if zeroflag:
            return cls.zero

        # If one of the args was a VectorAdd, return an
        # appropriate VectorAdd instance
        if isinstance(expr, cls._add_func):
            newargs = [cls._mul_func(measure_number, x) for
                       x in expr.args]
            return cls._add_func(*newargs)

        obj = super().__new__(cls, measure_number,
                              expr._base_instance,
                              *extra_args,
                              **options)
        if isinstance(obj, Add):
            return cls._add_func(*obj.args)
        obj._base_instance = expr._base_instance
        obj._measure_number = measure_number
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = {expr._base_instance: measure_number}
        obj._sys = expr._base_instance._sys

        return obj

    def _sympystr(self, printer):
        measure_str = printer._print(self._measure_number)
        if ('(' in measure_str or '-' in measure_str or
                '+' in measure_str):
            measure_str = '(' + measure_str + ')'
        return measure_str + '*' + printer._print(self._base_instance)


class BasisDependentZero(BasisDependent):
    """
    Class to denote a zero basis dependent instance.
    """
    components: dict['BaseVector', Expr] = {}
    _latex_form: str

    def __new__(cls):
        obj = super().__new__(cls)
        # Pre-compute a specific hash value for the zero vector
        # Use the same one always
        obj._hash = (S.Zero, cls).__hash__()
        return obj

    def __hash__(self):
        return self._hash

    @call_highest_priority('__req__')
    def __eq__(self, other):
        return isinstance(other, self._zero_func)

    __req__ = __eq__

    @call_highest_priority('__radd__')
    def __add__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for addition")

    @call_highest_priority('__add__')
    def __radd__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for addition")

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if isinstance(other, self._expr_type):
            return -other
        else:
            raise TypeError("Invalid argument types for subtraction")

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for subtraction")

    def __neg__(self):
        return self

    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        return self

    def _sympystr(self, printer):
        return '0'
<|endoftext|>from .utils import _toposort, groupby

class AmbiguityWarning(Warning):
    pass


def supercedes(a, b):
    """ A is consistent and strictly more specific than B """
    return len(a) == len(b) and all(map(issubclass, a, b))


def consistent(a, b):
    """ It is possible for an argument list to satisfy both A and B """
    return (len(a) == len(b) and
            all(issubclass(aa, bb) or issubclass(bb, aa)
                           for aa, bb in zip(a, b)))


def ambiguous(a, b):
    """ A is consistent with B but neither is strictly more specific """
    return consistent(a, b) and not (supercedes(a, b) or supercedes(b, a))


def ambiguities(signatures):
    """ All signature pairs such that A is ambiguous with B """
    signatures = list(map(tuple, signatures))
    return {(a, b) for a in signatures for b in signatures
                       if hash(a) < hash(b)
                       and ambiguous(a, b)
                       and not any(supercedes(c, a) and supercedes(c, b)
                                    for c in signatures)}


def super_signature(signatures):
    """ A signature that would break ambiguities """
    n = len(signatures[0])
    assert all(len(s) == n for s in signatures)

    return [max([type.mro(sig[i]) for sig in signatures], key=len)[0]
               for i in range(n)]


def edge(a, b, tie_breaker=hash):
    """ A should be checked before B

    Tie broken by tie_breaker, defaults to ``hash``
    """
    if supercedes(a, b):
        if supercedes(b, a):
            return tie_breaker(a) > tie_breaker(b)
        else:
            return True
    return False


def ordering(signatures):
    """ A sane ordering of signatures to check, first to last

    Topoological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(lambda x: x[0], edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    edges = {k: [b for a, b in v] for k, v in edges.items()}
    return _toposort(edges)
<|endoftext|>from collections import OrderedDict


def expand_tuples(L):
    """
    >>> from sympy.multipledispatch.utils import expand_tuples
    >>> expand_tuples([1, (2, 3)])
    [(1, 2), (1, 3)]

    >>> expand_tuples([1, 2])
    [(1, 2)]
    """
    if not L:
        return [()]
    elif not isinstance(L[0], tuple):
        rest = expand_tuples(L[1:])
        return [(L[0],) + t for t in rest]
    else:
        rest = expand_tuples(L[1:])
        return [(item,) + t for t in rest for item in L[0]]


# Taken from theano/theano/gof/sched.py
# Avoids licensing issues because this was written by Matthew Rocklin
def _toposort(edges):
    """ Topological sort algorithm by Kahn [1] - O(nodes + vertices)

    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges

    >>> from sympy.multipledispatch.utils import _toposort
    >>> _toposort({1: (2, 3), 2: (3, )})
    [1, 2, 3]

    Closely follows the wikipedia page [2]

    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] https://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = {k: set(val) for k, val in incoming_edges.items()}
    S = OrderedDict.fromkeys(v for v in edges if v not in incoming_edges)
    L = []

    while S:
        n, _ = S.popitem()
        L.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S[m] = None
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L


def reverse_dict(d):
    """Reverses direction of dependence dict

    >>> d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}

    :note: dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.

    """
    result = {}
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, ()) + (key, )
    return result


# Taken from toolz
# Avoids licensing issues because this version was authored by Matthew Rocklin
def groupby(func, seq):
    """ Group a collection by a key function

    >>> from sympy.multipledispatch.utils import groupby
    >>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    >>> groupby(len, names)  # doctest: +SKIP
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}

    >>> iseven = lambda x: x % 2 == 0
    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}

    See Also:
        ``countby``
    """

    d = {}
    for item in seq:
        key = func(item)
        if key not in d:
            d[key] = []
        d[key].append(item)
    return d
<|endoftext|>from __future__ import annotations
from typing import Any

import inspect

from .dispatcher import Dispatcher, MethodDispatcher, ambiguity_warn

# XXX: This parameter to dispatch isn't documented and isn't used anywhere in
# sympy. Maybe it should just be removed.
global_namespace: dict[str, Any] = {}


def dispatch(*types, namespace=global_namespace, on_ambiguity=ambiguity_warn):
    """ Dispatch function on the types of the inputs

    Supports dispatch on all non-keyword arguments.

    Collects implementations based on the function name.  Ignores namespaces.

    If ambiguous type signatures occur a warning is raised when the function is
    defined suggesting the additional method to break the ambiguity.

    Examples
    --------

    >>> from sympy.multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1

    >>> @dispatch(float)
    ... def f(x): # noqa: F811
    ...     return x - 1

    >>> f(3)
    4
    >>> f(3.0)
    2.0

    Specify an isolated namespace with the namespace keyword argument

    >>> my_namespace = dict()
    >>> @dispatch(int, namespace=my_namespace)
    ... def foo(x):
    ...     return x + 1

    Dispatch on instance methods within classes

    >>> class MyClass(object):
    ...     @dispatch(list)
    ...     def __init__(self, data):
    ...         self.data = data
    ...     @dispatch(int)
    ...     def __init__(self, datum): # noqa: F811
    ...         self.data = [datum]
    """
    types = tuple(types)

    def _(func):
        name = func.__name__

        if ismethod(func):
            dispatcher = inspect.currentframe().f_back.f_locals.get(
                name,
                MethodDispatcher(name))
        else:
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        dispatcher.add(types, func, on_ambiguity=on_ambiguity)
        return dispatcher
    return _


def ismethod(func):
    """ Is func a method?

    Note that this has to work as the method is defined but before the class is
    defined.  At this stage methods look like functions.
    """
    signature = inspect.signature(func)
    return signature.parameters.get('self', None) is not None
<|endoftext|>from __future__ import annotations

from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl


class MDNotImplementedError(NotImplementedError):
    """ A NotImplementedError for multiple dispatch """


### Functions for on_ambiguity

def ambiguity_warn(dispatcher, ambiguities):
    """ Raise warning when ambiguity is detected

    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher

    See Also:
        Dispatcher.add
        warning_text
    """
    warn(warning_text(dispatcher.name, ambiguities), AmbiguityWarning)


class RaiseNotImplementedError:
    """Raise ``NotImplementedError`` when called."""

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        types = tuple(type(a) for a in args)
        raise NotImplementedError(
            "Ambiguous signature for %s: <%s>" % (
            self.dispatcher.name, str_signature(types)
        ))

def ambiguity_register_error_ignore_dup(dispatcher, ambiguities):
    """
    If super signature for ambiguous types is duplicate types, ignore it.
    Else, register instance of ``RaiseNotImplementedError`` for ambiguous types.

    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher

    See Also:
        Dispatcher.add
        ambiguity_warn
    """
    for amb in ambiguities:
        signature = tuple(super_signature(amb))
        if len(set(signature)) == 1:
            continue
        dispatcher.add(
            signature, RaiseNotImplementedError(dispatcher),
            on_ambiguity=ambiguity_register_error_ignore_dup
        )

###


_unresolved_dispatchers: set[Dispatcher] = set()
_resolve = [True]


def halt_ordering():
    _resolve[0] = False


def restart_ordering(on_ambiguity=ambiguity_warn):
    _resolve[0] = True
    while _unresolved_dispatchers:
        dispatcher = _unresolved_dispatchers.pop()
        dispatcher.reorder(on_ambiguity=on_ambiguity)


class Dispatcher:
    """ Dispatch methods based on type signature

    Use ``dispatch`` to add implementations

    Examples
    --------

    >>> from sympy.multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1

    >>> @dispatch(float)
    ... def f(x): # noqa: F811
    ...     return x - 1

    >>> f(3)
    4
    >>> f(3.0)
    2.0
    """
    __slots__ = '__name__', 'name', 'funcs', 'ordering', '_cache', 'doc'

    def __init__(self, name, doc=None):
        self.name = self.__name__ = name
        self.funcs = {}
        self._cache = {}
        self.ordering = []
        self.doc = doc

    def register(self, *types, **kwargs):
        """ Register dispatcher with new implementation

        >>> from sympy.multipledispatch.dispatcher import Dispatcher
        >>> f = Dispatcher('f')
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1

        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1

        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]

        >>> f(1)
        2

        >>> f(1.0)
        0.0

        >>> f([1, 2, 3])
        [3, 2, 1]
        """
        def _(func):
            self.add(types, func, **kwargs)
            return func
        return _

    @classmethod
    def get_func_params(cls, func):
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return sig.parameters.values()

    @classmethod
    def get_func_annotations(cls, func):
        """ Get annotations of function positional parameters
        """
        params = cls.get_func_params(func)
        if params:
            Parameter = inspect.Parameter

            params = (param for param in params
                      if param.kind in
                      (Parameter.POSITIONAL_ONLY,
                       Parameter.POSITIONAL_OR_KEYWORD))

            annotations = tuple(
                param.annotation
                for param in params)

            if not any(ann is Parameter.empty for ann in annotations):
                return annotations

    def add(self, signature, func, on_ambiguity=ambiguity_warn):
        """ Add new types/method pair to dispatcher

        >>> from sympy.multipledispatch import Dispatcher
        >>> D = Dispatcher('add')
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)

        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>

        When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        with a dispatcher/itself, and a set of ambiguous type signature pairs
        as inputs.  See ``ambiguity_warn`` for an example.
        """
        # Handle annotations
        if not signature:
            annotations = self.get_func_annotations(func)
            if annotations:
                signature = annotations

        # Handle union types
        if any(isinstance(typ, tuple) for typ in signature):
            for typs in expand_tuples(signature):
                self.add(typs, func, on_ambiguity)
            return

        for typ in signature:
            if not isinstance(typ, type):
                str_sig = ', '.join(c.__name__ if isinstance(c, type)
                                    else str(c) for c in signature)
                raise TypeError("Tried to dispatch on non-type: %s\n"
                                "In signature: <%s>\n"
                                "In function: %s" %
                                (typ, str_sig, self.name))

        self.funcs[signature] = func
        self.reorder(on_ambiguity=on_ambiguity)
        self._cache.clear()

    def reorder(self, on_ambiguity=ambiguity_warn):
        if _resolve[0]:
            self.ordering = ordering(self.funcs)
            amb = ambiguities(self.funcs)
            if amb:
                on_ambiguity(self, amb)
        else:
            _unresolved_dispatchers.add(self)

    def __call__(self, *args, **kwargs):
        types = tuple([type(arg) for arg in args])
        try:
            func = self._cache[types]
        except KeyError:
            func = self.dispatch(*types)
            if not func:
                raise NotImplementedError(
                    'Could not find signature for %s: <%s>' %
                    (self.name, str_signature(types)))
            self._cache[types] = func
        try:
            return func(*args, **kwargs)

        except MDNotImplementedError:
            funcs = self.dispatch_iter(*types)
            next(funcs)  # burn first
            for func in funcs:
                try:
                    return func(*args, **kwargs)
                except MDNotImplementedError:
                    pass
            raise NotImplementedError("Matching functions for "
                                      "%s: <%s> found, but none completed successfully"
                                      % (self.name, str_signature(types)))

    def __str__(self):
        return "<dispatched %s>" % self.name
    __repr__ = __str__

    def dispatch(self, *types):
        """ Deterimine appropriate implementation for this type signature

        This method is internal.  Users should call this object as a function.
        Implementation resolution occurs within the ``__call__`` method.

        >>> from sympy.multipledispatch import dispatch
        >>> @dispatch(int)
        ... def inc(x):
        ...     return x + 1

        >>> implementation = inc.dispatch(int)
        >>> implementation(3)
        4

        >>> print(inc.dispatch(float))
        None

        See Also:
            ``sympy.multipledispatch.conflict`` - module to determine resolution order
        """

        if types in self.funcs:
            return self.funcs[types]

        try:
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    def dispatch_iter(self, *types):
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):
                result = self.funcs[signature]
                yield result

    def resolve(self, types):
        """ Deterimine appropriate implementation for this type signature

        .. deprecated:: 0.4.4
            Use ``dispatch(*types)`` instead
        """
        warn("resolve() is deprecated, use dispatch(*types)",
             DeprecationWarning)

        return self.dispatch(*types)

    def __getstate__(self):
        return {'name': self.name,
                'funcs': self.funcs}

    def __setstate__(self, d):
        self.name = d['name']
        self.funcs = d['funcs']
        self.ordering = ordering(self.funcs)
        self._cache = {}

    @property
    def __doc__(self):
        docs = ["Multiply dispatched method: %s" % self.name]

        if self.doc:
            docs.append(self.doc)

        other = []
        for sig in self.ordering[::-1]:
            func = self.funcs[sig]
            if func.__doc__:
                s = 'Inputs: <%s>\n' % str_signature(sig)
                s += '-' * len(s) + '\n'
                s += func.__doc__.strip()
                docs.append(s)
            else:
                other.append(str_signature(sig))

        if other:
            docs.append('Other signatures:\n    ' + '\n    '.join(other))

        return '\n\n'.join(docs)

    def _help(self, *args):
        return self.dispatch(*map(type, args)).__doc__

    def help(self, *args, **kwargs):
        """ Print docstring for the function corresponding to inputs """
        print(self._help(*args))

    def _source(self, *args):
        func = self.dispatch(*map(type, args))
        if not func:
            raise TypeError("No function found")
        return source(func)

    def source(self, *args, **kwargs):
        """ Print source code for the function corresponding to inputs """
        print(self._source(*args))


def source(func):
    s = 'File: %s\n\n' % inspect.getsourcefile(func)
    s = s + inspect.getsource(func)
    return s


class MethodDispatcher(Dispatcher):
    """ Dispatch methods based on type signature

    See Also:
        Dispatcher
    """

    @classmethod
    def get_func_params(cls, func):
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return itl.islice(sig.parameters.values(), 1, None)

    def __get__(self, instance, owner):
        self.obj = instance
        self.cls = owner
        return self

    def __call__(self, *args, **kwargs):
        types = tuple([type(arg) for arg in args])
        func = self.dispatch(*types)
        if not func:
            raise NotImplementedError('Could not find signature for %s: <%s>' %
                                      (self.name, str_signature(types)))
        return func(self.obj, *args, **kwargs)


def str_signature(sig):
    """ String representation of type signature

    >>> from sympy.multipledispatch.dispatcher import str_signature
    >>> str_signature((int, float))
    'int, float'
    """
    return ', '.join(cls.__name__ for cls in sig)


def warning_text(name, amb):
    """ The text for ambiguity warnings """
    text = "\nAmbiguities exist in dispatched function %s\n\n" % (name)
    text += "The following signatures may result in ambiguous behavior:\n"
    for pair in amb:
        text += "\t" + \
            ', '.join('[' + str_signature(s) + ']' for s in pair) + "\n"
    text += "\n\nConsider making the following additions:\n\n"
    text += '\n\n'.join(['@dispatch(' + str_signature(super_signature(s))
                         + ')\ndef %s(...)' % name for s in amb])
    return text
<|endoftext|>"""Module for querying SymPy objects about assumptions."""

from sympy.assumptions.assume import (global_assumptions, Predicate,
        AppliedPredicate)
from sympy.assumptions.cnf import CNF, EncodedCNF, Literal
from sympy.core import sympify
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.inference import satisfiable
from sympy.utilities.decorator import memoize_property
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning,
                                        ignore_warnings)


# Memoization is necessary for the properties of AssumptionKeys to
# ensure that only one object of Predicate objects are created.
# This is because assumption handlers are registered on those objects.


class AssumptionKeys:
    """
    This class contains all the supported keys by ``ask``.
    It should be accessed via the instance ``sympy.Q``.

    """

    # DO NOT add methods or properties other than predicate keys.
    # SAT solver checks the properties of Q and use them to compute the
    # fact system. Non-predicate attributes will break this.

    @memoize_property
    def hermitian(self):
        from .handlers.sets import HermitianPredicate
        return HermitianPredicate()

    @memoize_property
    def antihermitian(self):
        from .handlers.sets import AntihermitianPredicate
        return AntihermitianPredicate()

    @memoize_property
    def real(self):
        from .handlers.sets import RealPredicate
        return RealPredicate()

    @memoize_property
    def extended_real(self):
        from .handlers.sets import ExtendedRealPredicate
        return ExtendedRealPredicate()

    @memoize_property
    def imaginary(self):
        from .handlers.sets import ImaginaryPredicate
        return ImaginaryPredicate()

    @memoize_property
    def complex(self):
        from .handlers.sets import ComplexPredicate
        return ComplexPredicate()

    @memoize_property
    def algebraic(self):
        from .handlers.sets import AlgebraicPredicate
        return AlgebraicPredicate()

    @memoize_property
    def transcendental(self):
        from .predicates.sets import TranscendentalPredicate
        return TranscendentalPredicate()

    @memoize_property
    def integer(self):
        from .handlers.sets import IntegerPredicate
        return IntegerPredicate()

    @memoize_property
    def noninteger(self):
        from .predicates.sets import NonIntegerPredicate
        return NonIntegerPredicate()

    @memoize_property
    def rational(self):
        from .handlers.sets import RationalPredicate
        return RationalPredicate()

    @memoize_property
    def irrational(self):
        from .handlers.sets import IrrationalPredicate
        return IrrationalPredicate()

    @memoize_property
    def finite(self):
        from .handlers.calculus import FinitePredicate
        return FinitePredicate()

    @memoize_property
    def infinite(self):
        from .handlers.calculus import InfinitePredicate
        return InfinitePredicate()

    @memoize_property
    def positive_infinite(self):
        from .handlers.calculus import PositiveInfinitePredicate
        return PositiveInfinitePredicate()

    @memoize_property
    def negative_infinite(self):
        from .handlers.calculus import NegativeInfinitePredicate
        return NegativeInfinitePredicate()

    @memoize_property
    def positive(self):
        from .handlers.order import PositivePredicate
        return PositivePredicate()

    @memoize_property
    def negative(self):
        from .handlers.order import NegativePredicate
        return NegativePredicate()

    @memoize_property
    def zero(self):
        from .handlers.order import ZeroPredicate
        return ZeroPredicate()

    @memoize_property
    def extended_positive(self):
        from .handlers.order import ExtendedPositivePredicate
        return ExtendedPositivePredicate()

    @memoize_property
    def extended_negative(self):
        from .handlers.order import ExtendedNegativePredicate
        return ExtendedNegativePredicate()

    @memoize_property
    def nonzero(self):
        from .handlers.order import NonZeroPredicate
        return NonZeroPredicate()

    @memoize_property
    def nonpositive(self):
        from .handlers.order import NonPositivePredicate
        return NonPositivePredicate()

    @memoize_property
    def nonnegative(self):
        from .handlers.order import NonNegativePredicate
        return NonNegativePredicate()

    @memoize_property
    def extended_nonzero(self):
        from .handlers.order import ExtendedNonZeroPredicate
        return ExtendedNonZeroPredicate()

    @memoize_property
    def extended_nonpositive(self):
        from .handlers.order import ExtendedNonPositivePredicate
        return ExtendedNonPositivePredicate()

    @memoize_property
    def extended_nonnegative(self):
        from .handlers.order import ExtendedNonNegativePredicate
        return ExtendedNonNegativePredicate()

    @memoize_property
    def even(self):
        from .handlers.ntheory import EvenPredicate
        return EvenPredicate()

    @memoize_property
    def odd(self):
        from .handlers.ntheory import OddPredicate
        return OddPredicate()

    @memoize_property
    def prime(self):
        from .handlers.ntheory import PrimePredicate
        return PrimePredicate()

    @memoize_property
    def composite(self):
        from .handlers.ntheory import CompositePredicate
        return CompositePredicate()

    @memoize_property
    def commutative(self):
        from .handlers.common import CommutativePredicate
        return CommutativePredicate()

    @memoize_property
    def is_true(self):
        from .handlers.common import IsTruePredicate
        return IsTruePredicate()

    @memoize_property
    def symmetric(self):
        from .handlers.matrices import SymmetricPredicate
        return SymmetricPredicate()

    @memoize_property
    def invertible(self):
        from .handlers.matrices import InvertiblePredicate
        return InvertiblePredicate()

    @memoize_property
    def orthogonal(self):
        from .handlers.matrices import OrthogonalPredicate
        return OrthogonalPredicate()

    @memoize_property
    def unitary(self):
        from .handlers.matrices import UnitaryPredicate
        return UnitaryPredicate()

    @memoize_property
    def positive_definite(self):
        from .handlers.matrices import PositiveDefinitePredicate
        return PositiveDefinitePredicate()

    @memoize_property
    def upper_triangular(self):
        from .handlers.matrices import UpperTriangularPredicate
        return UpperTriangularPredicate()

    @memoize_property
    def lower_triangular(self):
        from .handlers.matrices import LowerTriangularPredicate
        return LowerTriangularPredicate()

    @memoize_property
    def diagonal(self):
        from .handlers.matrices import DiagonalPredicate
        return DiagonalPredicate()

    @memoize_property
    def fullrank(self):
        from .handlers.matrices import FullRankPredicate
        return FullRankPredicate()

    @memoize_property
    def square(self):
        from .handlers.matrices import SquarePredicate
        return SquarePredicate()

    @memoize_property
    def integer_elements(self):
        from .handlers.matrices import IntegerElementsPredicate
        return IntegerElementsPredicate()

    @memoize_property
    def real_elements(self):
        from .handlers.matrices import RealElementsPredicate
        return RealElementsPredicate()

    @memoize_property
    def complex_elements(self):
        from .handlers.matrices import ComplexElementsPredicate
        return ComplexElementsPredicate()

    @memoize_property
    def singular(self):
        from .predicates.matrices import SingularPredicate
        return SingularPredicate()

    @memoize_property
    def normal(self):
        from .predicates.matrices import NormalPredicate
        return NormalPredicate()

    @memoize_property
    def triangular(self):
        from .predicates.matrices import TriangularPredicate
        return TriangularPredicate()

    @memoize_property
    def unit_triangular(self):
        from .predicates.matrices import UnitTriangularPredicate
        return UnitTriangularPredicate()

    @memoize_property
    def eq(self):
        from .relation.equality import EqualityPredicate
        return EqualityPredicate()

    @memoize_property
    def ne(self):
        from .relation.equality import UnequalityPredicate
        return UnequalityPredicate()

    @memoize_property
    def gt(self):
        from .relation.equality import StrictGreaterThanPredicate
        return StrictGreaterThanPredicate()

    @memoize_property
    def ge(self):
        from .relation.equality import GreaterThanPredicate
        return GreaterThanPredicate()

    @memoize_property
    def lt(self):
        from .relation.equality import StrictLessThanPredicate
        return StrictLessThanPredicate()

    @memoize_property
    def le(self):
        from .relation.equality import LessThanPredicate
        return LessThanPredicate()


Q = AssumptionKeys()

def _extract_all_facts(assump, exprs):
    """
    Extract all relevant assumptions from *assump* with respect to given *exprs*.

    Parameters
    ==========

    assump : sympy.assumptions.cnf.CNF

    exprs : tuple of expressions

    Returns
    =======

    sympy.assumptions.cnf.CNF

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.ask import _extract_all_facts
    >>> from sympy.abc import x, y
    >>> assump = CNF.from_prop(Q.positive(x) & Q.integer(y))
    >>> exprs = (x,)
    >>> cnf = _extract_all_facts(assump, exprs)
    >>> cnf.clauses
    {frozenset({Literal(Q.positive, False)})}

    """
    facts = set()

    for clause in assump.clauses:
        args = []
        for literal in clause:
            if isinstance(literal.lit, AppliedPredicate) and len(literal.lit.arguments) == 1:
                if literal.lit.arg in exprs:
                    # Add literal if it has matching in it
                    args.append(Literal(literal.lit.function, literal.is_Not))
                else:
                    # If any of the literals doesn't have matching expr don't add the whole clause.
                    break
            else:
                # If any of the literals aren't unary predicate don't add the whole clause.
                break

        else:
            if args:
                facts.add(frozenset(args))
    return CNF(facts)


def ask(proposition, assumptions=True, context=global_assumptions):
    """
    Function to evaluate the proposition with assumptions.

    Explanation
    ===========

    This function evaluates the proposition to ``True`` or ``False`` if
    the truth value can be determined. If not, it returns ``None``.

    It should be discerned from :func:`~.refine` which, when applied to a
    proposition, simplifies the argument to symbolic ``Boolean`` instead of
    Python built-in ``True``, ``False`` or ``None``.

    **Syntax**

        * ask(proposition)
            Evaluate the *proposition* in global assumption context.

        * ask(proposition, assumptions)
            Evaluate the *proposition* with respect to *assumptions* in
            global assumption context.

    Parameters
    ==========

    proposition : Boolean
        Proposition which will be evaluated to boolean value. If this is
        not ``AppliedPredicate``, it will be wrapped by ``Q.is_true``.

    assumptions : Boolean, optional
        Local assumptions to evaluate the *proposition*.

    context : AssumptionsContext, optional
        Default assumptions to evaluate the *proposition*. By default,
        this is ``sympy.assumptions.global_assumptions`` variable.

    Returns
    =======

    ``True``, ``False``, or ``None``

    Raises
    ======

    TypeError : *proposition* or *assumptions* is not valid logical expression.

    ValueError : assumptions are inconsistent.

    Examples
    ========

    >>> from sympy import ask, Q, pi
    >>> from sympy.abc import x, y
    >>> ask(Q.rational(pi))
    False
    >>> ask(Q.even(x*y), Q.even(x) & Q.integer(y))
    True
    >>> ask(Q.prime(4*x), Q.integer(x))
    False

    If the truth value cannot be determined, ``None`` will be returned.

    >>> print(ask(Q.odd(3*x))) # cannot determine unless we know x
    None

    ``ValueError`` is raised if assumptions are inconsistent.

    >>> ask(Q.integer(x), Q.even(x) & Q.odd(x))
    Traceback (most recent call last):
      ...
    ValueError: inconsistent assumptions Q.even(x) & Q.odd(x)

    Notes
    =====

    Relations in assumptions are not implemented (yet), so the following
    will not give a meaningful result.

    >>> ask(Q.positive(x), x > 0)

    It is however a work in progress.

    See Also
    ========

    sympy.assumptions.refine.refine : Simplification using assumptions.
        Proposition is not reduced to ``None`` if the truth value cannot
        be determined.
    """
    from sympy.assumptions.satask import satask
    from sympy.assumptions.lra_satask import lra_satask
    from sympy.logic.algorithms.lra_theory import UnhandledInput

    proposition = sympify(proposition)
    assumptions = sympify(assumptions)

    if isinstance(proposition, Predicate) or proposition.kind is not BooleanKind:
        raise TypeError("proposition must be a valid logical expression")

    if isinstance(assumptions, Predicate) or assumptions.kind is not BooleanKind:
        raise TypeError("assumptions must be a valid logical expression")

    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
    if isinstance(proposition, AppliedPredicate):
        key, args = proposition.function, proposition.arguments
    elif proposition.func in binrelpreds:
        key, args = binrelpreds[type(proposition)], proposition.args
    else:
        key, args = Q.is_true, (proposition,)

    # convert local and global assumptions to CNF
    assump_cnf = CNF.from_prop(assumptions)
    assump_cnf.extend(context)

    # extract the relevant facts from assumptions with respect to args
    local_facts = _extract_all_facts(assump_cnf, args)

    # convert default facts and assumed facts to encoded CNF
    known_facts_cnf = get_all_known_facts()
    enc_cnf = EncodedCNF()
    enc_cnf.from_cnf(CNF(known_facts_cnf))
    enc_cnf.add_from_cnf(local_facts)

    # check the satisfiability of given assumptions
    if local_facts.clauses and satisfiable(enc_cnf) is False:
        raise ValueError("inconsistent assumptions %s" % assumptions)

    # quick computation for single fact
    res = _ask_single_fact(key, local_facts)
    if res is not None:
        return res

    # direct resolution method, no logic
    res = key(*args)._eval_ask(assumptions)
    if res is not None:
        return bool(res)

    # using satask (still costly)
    res = satask(proposition, assumptions=assumptions, context=context)
    if res is not None:
        return res

    try:
        res = lra_satask(proposition, assumptions=assumptions, context=context)
    except UnhandledInput:
        return None

    return res


def _ask_single_fact(key, local_facts):
    """
    Compute the truth value of single predicate using assumptions.

    Parameters
    ==========

    key : sympy.assumptions.assume.Predicate
        Proposition predicate.

    local_facts : sympy.assumptions.cnf.CNF
        Local assumption in CNF form.

    Returns
    =======

    ``True``, ``False`` or ``None``

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.ask import _ask_single_fact

    If prerequisite of proposition is rejected by the assumption,
    return ``False``.

    >>> key, assump = Q.zero, ~Q.zero
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False
    >>> key, assump = Q.zero, ~Q.even
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False

    If assumption implies the proposition, return ``True``.

    >>> key, assump = Q.even, Q.zero
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    True

    If proposition rejects the assumption, return ``False``.

    >>> key, assump = Q.even, Q.odd
    >>> local_facts = CNF.from_prop(assump)
    >>> _ask_single_fact(key, local_facts)
    False
    """
    if local_facts.clauses:

        known_facts_dict = get_known_facts_dict()

        if len(local_facts.clauses) == 1:
            cl, = local_facts.clauses
            if len(cl) == 1:
                f, = cl
                prop_facts = known_facts_dict.get(key, None)
                prop_req = prop_facts[0] if prop_facts is not None else set()
                if f.is_Not and f.arg in prop_req:
                    # the prerequisite of proposition is rejected
                    return False

        for clause in local_facts.clauses:
            if len(clause) == 1:
                f, = clause
                prop_facts = known_facts_dict.get(f.arg, None) if not f.is_Not else None
                if prop_facts is None:
                    continue

                prop_req, prop_rej = prop_facts
                if key in prop_req:
                    # assumption implies the proposition
                    return True
                elif key in prop_rej:
                    # proposition rejects the assumption
                    return False

    return None


def register_handler(key, handler):
    """
    Register a handler in the ask system. key must be a string and handler a
    class inheriting from AskHandler.

    .. deprecated:: 1.8.
        Use multipledispatch handler instead. See :obj:`~.Predicate`.

    """
    sympy_deprecation_warning(
        """
        The AskHandler system is deprecated. The register_handler() function
        should be replaced with the multipledispatch handler of Predicate.
        """,
        deprecated_since_version="1.8",
        active_deprecations_target='deprecated-askhandler',
    )
    if isinstance(key, Predicate):
        key = key.name.name
    Qkey = getattr(Q, key, None)
    if Qkey is not None:
        Qkey.add_handler(handler)
    else:
        setattr(Q, key, Predicate(key, handlers=[handler]))


def remove_handler(key, handler):
    """
    Removes a handler from the ask system.

    .. deprecated:: 1.8.
        Use multipledispatch handler instead. See :obj:`~.Predicate`.

    """
    sympy_deprecation_warning(
        """
        The AskHandler system is deprecated. The remove_handler() function
        should be replaced with the multipledispatch handler of Predicate.
        """,
        deprecated_since_version="1.8",
        active_deprecations_target='deprecated-askhandler',
    )
    if isinstance(key, Predicate):
        key = key.name.name
    # Don't show the same warning again recursively
    with ignore_warnings(SymPyDeprecationWarning):
        getattr(Q, key).remove_handler(handler)


from sympy.assumptions.ask_generated import (get_all_known_facts,
    get_known_facts_dict)
<|endoftext|>from __future__ import annotations
from typing import Callable

from sympy.core import S, Add, Expr, Basic, Mul, Pow, Rational
from sympy.core.logic import fuzzy_not
from sympy.logic.boolalg import Boolean

from sympy.assumptions import ask, Q  # type: ignore


def refine(expr, assumptions=True):
    """
    Simplify an expression using assumptions.

    Explanation
    ===========

    Unlike :func:`~.simplify` which performs structural simplification
    without any assumption, this function transforms the expression into
    the form which is only valid under certain assumptions. Note that
    ``simplify()`` is generally not done in refining process.

    Refining boolean expression involves reducing it to ``S.true`` or
    ``S.false``. Unlike :func:`~.ask`, the expression will not be reduced
    if the truth value cannot be determined.

    Examples
    ========

    >>> from sympy import refine, sqrt, Q
    >>> from sympy.abc import x
    >>> refine(sqrt(x**2), Q.real(x))
    Abs(x)
    >>> refine(sqrt(x**2), Q.positive(x))
    x

    >>> refine(Q.real(x), Q.positive(x))
    True
    >>> refine(Q.positive(x), Q.real(x))
    Q.positive(x)

    See Also
    ========

    sympy.simplify.simplify.simplify : Structural simplification without assumptions.
    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.
    """
    if not isinstance(expr, Basic):
        return expr

    if not expr.is_Atom:
        args = [refine(arg, assumptions) for arg in expr.args]
        # TODO: this will probably not work with Integral or Polynomial
        expr = expr.func(*args)
    if hasattr(expr, '_eval_refine'):
        ref_expr = expr._eval_refine(assumptions)
        if ref_expr is not None:
            return ref_expr
    name = expr.__class__.__name__
    handler = handlers_dict.get(name, None)
    if handler is None:
        return expr
    new_expr = handler(expr, assumptions)
    if (new_expr is None) or (expr == new_expr):
        return expr
    if not isinstance(new_expr, Expr):
        return new_expr
    return refine(new_expr, assumptions)


def refine_abs(expr, assumptions):
    """
    Handler for the absolute value.

    Examples
    ========

    >>> from sympy import Q, Abs
    >>> from sympy.assumptions.refine import refine_abs
    >>> from sympy.abc import x
    >>> refine_abs(Abs(x), Q.real(x))
    >>> refine_abs(Abs(x), Q.positive(x))
    x
    >>> refine_abs(Abs(x), Q.negative(x))
    -x

    """
    from sympy.functions.elementary.complexes import Abs
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions) and \
            fuzzy_not(ask(Q.negative(arg), assumptions)):
        # if it's nonnegative
        return arg
    if ask(Q.negative(arg), assumptions):
        return -arg
    # arg is Mul
    if isinstance(arg, Mul):
        r = [refine(abs(a), assumptions) for a in arg.args]
        non_abs = []
        in_abs = []
        for i in r:
            if isinstance(i, Abs):
                in_abs.append(i.args[0])
            else:
                non_abs.append(i)
        return Mul(*non_abs) * Abs(Mul(*in_abs))


def refine_Pow(expr, assumptions):
    """
    Handler for instances of Pow.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.refine import refine_Pow
    >>> from sympy.abc import x,y,z
    >>> refine_Pow((-1)**x, Q.real(x))
    >>> refine_Pow((-1)**x, Q.even(x))
    1
    >>> refine_Pow((-1)**x, Q.odd(x))
    -1

    For powers of -1, even parts of the exponent can be simplified:

    >>> refine_Pow((-1)**(x+y), Q.even(x))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+z), Q.odd(x) & Q.odd(z))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+2), Q.odd(x))
    (-1)**(y + 1)
    >>> refine_Pow((-1)**(x+3), True)
    (-1)**(x + 1)

    """
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions import sign
    if isinstance(expr.base, Abs):
        if ask(Q.real(expr.base.args[0]), assumptions) and \
                ask(Q.even(expr.exp), assumptions):
            return expr.base.args[0] ** expr.exp
    if ask(Q.real(expr.base), assumptions):
        if expr.base.is_number:
            if ask(Q.even(expr.exp), assumptions):
                return abs(expr.base) ** expr.exp
            if ask(Q.odd(expr.exp), assumptions):
                return sign(expr.base) * abs(expr.base) ** expr.exp
        if isinstance(expr.exp, Rational):
            if isinstance(expr.base, Pow):
                return abs(expr.base.base) ** (expr.base.exp * expr.exp)

        if expr.base is S.NegativeOne:
            if expr.exp.is_Add:

                old = expr

                # For powers of (-1) we can remove
                #  - even terms
                #  - pairs of odd terms
                #  - a single odd term + 1
                #  - A numerical constant N can be replaced with mod(N,2)

                coeff, terms = expr.exp.as_coeff_add()
                terms = set(terms)
                even_terms = set()
                odd_terms = set()
                initial_number_of_terms = len(terms)

                for t in terms:
                    if ask(Q.even(t), assumptions):
                        even_terms.add(t)
                    elif ask(Q.odd(t), assumptions):
                        odd_terms.add(t)

                terms -= even_terms
                if len(odd_terms) % 2:
                    terms -= odd_terms
                    new_coeff = (coeff + S.One) % 2
                else:
                    terms -= odd_terms
                    new_coeff = coeff % 2

                if new_coeff != coeff or len(terms) < initial_number_of_terms:
                    terms.add(new_coeff)
                    expr = expr.base**(Add(*terms))

                # Handle (-1)**((-1)**n/2 + m/2)
                e2 = 2*expr.exp
                if ask(Q.even(e2), assumptions):
                    if e2.could_extract_minus_sign():
                        e2 *= expr.base
                if e2.is_Add:
                    i, p = e2.as_two_terms()
                    if p.is_Pow and p.base is S.NegativeOne:
                        if ask(Q.integer(p.exp), assumptions):
                            i = (i + 1)/2
                            if ask(Q.even(i), assumptions):
                                return expr.base**p.exp
                            elif ask(Q.odd(i), assumptions):
                                return expr.base**(p.exp + 1)
                            else:
                                return expr.base**(p.exp + i)

                if old != expr:
                    return expr


def refine_atan2(expr, assumptions):
    """
    Handler for the atan2 function.

    Examples
    ========

    >>> from sympy import Q, atan2
    >>> from sympy.assumptions.refine import refine_atan2
    >>> from sympy.abc import x, y
    >>> refine_atan2(atan2(y,x), Q.real(y) & Q.positive(x))
    atan(y/x)
    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.negative(x))
    atan(y/x) - pi
    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.negative(x))
    atan(y/x) + pi
    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.negative(x))
    pi
    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.zero(x))
    pi/2
    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.zero(x))
    -pi/2
    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.zero(x))
    nan
    """
    from sympy.functions.elementary.trigonometric import atan
    y, x = expr.args
    if ask(Q.real(y) & Q.positive(x), assumptions):
        return atan(y / x)
    elif ask(Q.negative(y) & Q.negative(x), assumptions):
        return atan(y / x) - S.Pi
    elif ask(Q.positive(y) & Q.negative(x), assumptions):
        return atan(y / x) + S.Pi
    elif ask(Q.zero(y) & Q.negative(x), assumptions):
        return S.Pi
    elif ask(Q.positive(y) & Q.zero(x), assumptions):
        return S.Pi/2
    elif ask(Q.negative(y) & Q.zero(x), assumptions):
        return -S.Pi/2
    elif ask(Q.zero(y) & Q.zero(x), assumptions):
        return S.NaN
    else:
        return expr


def refine_re(expr, assumptions):
    """
    Handler for real part.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_re
    >>> from sympy import Q, re
    >>> from sympy.abc import x
    >>> refine_re(re(x), Q.real(x))
    x
    >>> refine_re(re(x), Q.imaginary(x))
    0
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return arg
    if ask(Q.imaginary(arg), assumptions):
        return S.Zero
    return _refine_reim(expr, assumptions)


def refine_im(expr, assumptions):
    """
    Handler for imaginary part.

    Explanation
    ===========

    >>> from sympy.assumptions.refine import refine_im
    >>> from sympy import Q, im
    >>> from sympy.abc import x
    >>> refine_im(im(x), Q.real(x))
    0
    >>> refine_im(im(x), Q.imaginary(x))
    -I*x
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return S.Zero
    if ask(Q.imaginary(arg), assumptions):
        return - S.ImaginaryUnit * arg
    return _refine_reim(expr, assumptions)

def refine_arg(expr, assumptions):
    """
    Handler for complex argument

    Explanation
    ===========

    >>> from sympy.assumptions.refine import refine_arg
    >>> from sympy import Q, arg
    >>> from sympy.abc import x
    >>> refine_arg(arg(x), Q.positive(x))
    0
    >>> refine_arg(arg(x), Q.negative(x))
    pi
    """
    rg = expr.args[0]
    if ask(Q.positive(rg), assumptions):
        return S.Zero
    if ask(Q.negative(rg), assumptions):
        return S.Pi
    return None


def _refine_reim(expr, assumptions):
    # Helper function for refine_re & refine_im
    expanded = expr.expand(complex = True)
    if expanded != expr:
        refined = refine(expanded, assumptions)
        if refined != expanded:
            return refined
    # Best to leave the expression as is
    return None


def refine_sign(expr, assumptions):
    """
    Handler for sign.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_sign
    >>> from sympy import Symbol, Q, sign, im
    >>> x = Symbol('x', real = True)
    >>> expr = sign(x)
    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))
    1
    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))
    -1
    >>> refine_sign(expr, Q.zero(x))
    0
    >>> y = Symbol('y', imaginary = True)
    >>> expr = sign(y)
    >>> refine_sign(expr, Q.positive(im(y)))
    I
    >>> refine_sign(expr, Q.negative(im(y)))
    -I
    """
    arg = expr.args[0]
    if ask(Q.zero(arg), assumptions):
        return S.Zero
    if ask(Q.real(arg)):
        if ask(Q.positive(arg), assumptions):
            return S.One
        if ask(Q.negative(arg), assumptions):
            return S.NegativeOne
    if ask(Q.imaginary(arg)):
        arg_re, arg_im = arg.as_real_imag()
        if ask(Q.positive(arg_im), assumptions):
            return S.ImaginaryUnit
        if ask(Q.negative(arg_im), assumptions):
            return -S.ImaginaryUnit
    return expr


def refine_matrixelement(expr, assumptions):
    """
    Handler for symmetric part.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_matrixelement
    >>> from sympy import MatrixSymbol, Q
    >>> X = MatrixSymbol('X', 3, 3)
    >>> refine_matrixelement(X[0, 1], Q.symmetric(X))
    X[0, 1]
    >>> refine_matrixelement(X[1, 0], Q.symmetric(X))
    X[0, 1]
    """
    from sympy.matrices.expressions.matexpr import MatrixElement
    matrix, i, j = expr.args
    if ask(Q.symmetric(matrix), assumptions):
        if (i - j).could_extract_minus_sign():
            return expr
        return MatrixElement(matrix, j, i)

handlers_dict: dict[str, Callable[[Expr, Boolean], Expr]] = {
    'Abs': refine_abs,
    'Pow': refine_Pow,
    'atan2': refine_atan2,
    're': refine_re,
    'im': refine_im,
    'arg': refine_arg,
    'sign': refine_sign,
    'MatrixElement': refine_matrixelement
}
<|endoftext|>"""
Do NOT manually edit this file.
Instead, run ./bin/ask_update.py.
"""

from sympy.assumptions.ask import Q
from sympy.assumptions.cnf import Literal
from sympy.core.cache import cacheit

@cacheit
def get_all_known_facts():
    """
    Known facts between unary predicates as CNF clauses.
    """
    return {
        frozenset((Literal(Q.algebraic, False), Literal(Q.imaginary, True), Literal(Q.transcendental, False))),
        frozenset((Literal(Q.algebraic, False), Literal(Q.negative, True), Literal(Q.transcendental, False))),
        frozenset((Literal(Q.algebraic, False), Literal(Q.positive, True), Literal(Q.transcendental, False))),
        frozenset((Literal(Q.algebraic, False), Literal(Q.rational, True))),
        frozenset((Literal(Q.algebraic, False), Literal(Q.transcendental, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.algebraic, True), Literal(Q.finite, False))),
        frozenset((Literal(Q.algebraic, True), Literal(Q.transcendental, True))),
        frozenset((Literal(Q.antihermitian, False), Literal(Q.hermitian, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.antihermitian, False), Literal(Q.imaginary, True))),
        frozenset((Literal(Q.commutative, False), Literal(Q.finite, True))),
        frozenset((Literal(Q.commutative, False), Literal(Q.infinite, True))),
        frozenset((Literal(Q.complex_elements, False), Literal(Q.real_elements, True))),
        frozenset((Literal(Q.composite, False), Literal(Q.even, True), Literal(Q.positive, True), Literal(Q.prime, False))),
        frozenset((Literal(Q.composite, True), Literal(Q.even, False), Literal(Q.odd, False))),
        frozenset((Literal(Q.composite, True), Literal(Q.positive, False))),
        frozenset((Literal(Q.composite, True), Literal(Q.prime, True))),
        frozenset((Literal(Q.diagonal, False), Literal(Q.lower_triangular, True), Literal(Q.upper_triangular, True))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.lower_triangular, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.normal, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.symmetric, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.upper_triangular, False))),
        frozenset((Literal(Q.even, False), Literal(Q.odd, False), Literal(Q.prime, True))),
        frozenset((Literal(Q.even, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.even, True), Literal(Q.odd, True))),
        frozenset((Literal(Q.even, True), Literal(Q.rational, False))),
        frozenset((Literal(Q.finite, False), Literal(Q.transcendental, True))),
        frozenset((Literal(Q.finite, True), Literal(Q.infinite, True))),
        frozenset((Literal(Q.fullrank, False), Literal(Q.invertible, True))),
        frozenset((Literal(Q.fullrank, True), Literal(Q.invertible, False), Literal(Q.square, True))),
        frozenset((Literal(Q.hermitian, False), Literal(Q.negative, True))),
        frozenset((Literal(Q.hermitian, False), Literal(Q.positive, True))),
        frozenset((Literal(Q.hermitian, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.imaginary, True), Literal(Q.negative, True))),
        frozenset((Literal(Q.imaginary, True), Literal(Q.positive, True))),
        frozenset((Literal(Q.imaginary, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.infinite, False), Literal(Q.negative_infinite, True))),
        frozenset((Literal(Q.infinite, False), Literal(Q.positive_infinite, True))),
        frozenset((Literal(Q.integer_elements, True), Literal(Q.real_elements, False))),
        frozenset((Literal(Q.invertible, False), Literal(Q.positive_definite, True))),
        frozenset((Literal(Q.invertible, False), Literal(Q.singular, False))),
        frozenset((Literal(Q.invertible, False), Literal(Q.unitary, True))),
        frozenset((Literal(Q.invertible, True), Literal(Q.singular, True))),
        frozenset((Literal(Q.invertible, True), Literal(Q.square, False))),
        frozenset((Literal(Q.irrational, False), Literal(Q.negative, True), Literal(Q.rational, False))),
        frozenset((Literal(Q.irrational, False), Literal(Q.positive, True), Literal(Q.rational, False))),
        frozenset((Literal(Q.irrational, False), Literal(Q.rational, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.irrational, True), Literal(Q.negative, False), Literal(Q.positive, False), Literal(Q.zero, False))),
        frozenset((Literal(Q.irrational, True), Literal(Q.rational, True))),
        frozenset((Literal(Q.lower_triangular, False), Literal(Q.triangular, True), Literal(Q.upper_triangular, False))),
        frozenset((Literal(Q.lower_triangular, True), Literal(Q.triangular, False))),
        frozenset((Literal(Q.negative, False), Literal(Q.positive, False), Literal(Q.rational, True), Literal(Q.zero, False))),
        frozenset((Literal(Q.negative, True), Literal(Q.negative_infinite, True))),
        frozenset((Literal(Q.negative, True), Literal(Q.positive, True))),
        frozenset((Literal(Q.negative, True), Literal(Q.positive_infinite, True))),
        frozenset((Literal(Q.negative, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.negative_infinite, True), Literal(Q.positive, True))),
        frozenset((Literal(Q.negative_infinite, True), Literal(Q.positive_infinite, True))),
        frozenset((Literal(Q.negative_infinite, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.normal, False), Literal(Q.unitary, True))),
        frozenset((Literal(Q.normal, True), Literal(Q.square, False))),
        frozenset((Literal(Q.odd, True), Literal(Q.rational, False))),
        frozenset((Literal(Q.orthogonal, False), Literal(Q.real_elements, True), Literal(Q.unitary, True))),
        frozenset((Literal(Q.orthogonal, True), Literal(Q.positive_definite, False))),
        frozenset((Literal(Q.orthogonal, True), Literal(Q.unitary, False))),
        frozenset((Literal(Q.positive, False), Literal(Q.prime, True))),
        frozenset((Literal(Q.positive, True), Literal(Q.positive_infinite, True))),
        frozenset((Literal(Q.positive, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.positive_infinite, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.square, False), Literal(Q.symmetric, True))),
        frozenset((Literal(Q.triangular, False), Literal(Q.unit_triangular, True))),
        frozenset((Literal(Q.triangular, False), Literal(Q.upper_triangular, True)))
    }

@cacheit
def get_all_known_matrix_facts():
    """
    Known facts between unary predicates for matrices as CNF clauses.
    """
    return {
        frozenset((Literal(Q.complex_elements, False), Literal(Q.real_elements, True))),
        frozenset((Literal(Q.diagonal, False), Literal(Q.lower_triangular, True), Literal(Q.upper_triangular, True))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.lower_triangular, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.normal, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.symmetric, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.upper_triangular, False))),
        frozenset((Literal(Q.fullrank, False), Literal(Q.invertible, True))),
        frozenset((Literal(Q.fullrank, True), Literal(Q.invertible, False), Literal(Q.square, True))),
        frozenset((Literal(Q.integer_elements, True), Literal(Q.real_elements, False))),
        frozenset((Literal(Q.invertible, False), Literal(Q.positive_definite, True))),
        frozenset((Literal(Q.invertible, False), Literal(Q.singular, False))),
        frozenset((Literal(Q.invertible, False), Literal(Q.unitary, True))),
        frozenset((Literal(Q.invertible, True), Literal(Q.singular, True))),
        frozenset((Literal(Q.invertible, True), Literal(Q.square, False))),
        frozenset((Literal(Q.lower_triangular, False), Literal(Q.triangular, True), Literal(Q.upper_triangular, False))),
        frozenset((Literal(Q.lower_triangular, True), Literal(Q.triangular, False))),
        frozenset((Literal(Q.normal, False), Literal(Q.unitary, True))),
        frozenset((Literal(Q.normal, True), Literal(Q.square, False))),
        frozenset((Literal(Q.orthogonal, False), Literal(Q.real_elements, True), Literal(Q.unitary, True))),
        frozenset((Literal(Q.orthogonal, True), Literal(Q.positive_definite, False))),
        frozenset((Literal(Q.orthogonal, True), Literal(Q.unitary, False))),
        frozenset((Literal(Q.square, False), Literal(Q.symmetric, True))),
        frozenset((Literal(Q.triangular, False), Literal(Q.unit_triangular, True))),
        frozenset((Literal(Q.triangular, False), Literal(Q.upper_triangular, True)))
    }

@cacheit
def get_all_known_number_facts():
    """
    Known facts between unary predicates for numbers as CNF clauses.
    """
    return {
        frozenset((Literal(Q.algebraic, False), Literal(Q.imaginary, True), Literal(Q.transcendental, False))),
        frozenset((Literal(Q.algebraic, False), Literal(Q.negative, True), Literal(Q.transcendental, False))),
        frozenset((Literal(Q.algebraic, False), Literal(Q.positive, True), Literal(Q.transcendental, False))),
        frozenset((Literal(Q.algebraic, False), Literal(Q.rational, True))),
        frozenset((Literal(Q.algebraic, False), Literal(Q.transcendental, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.algebraic, True), Literal(Q.finite, False))),
        frozenset((Literal(Q.algebraic, True), Literal(Q.transcendental, True))),
        frozenset((Literal(Q.antihermitian, False), Literal(Q.hermitian, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.antihermitian, False), Literal(Q.imaginary, True))),
        frozenset((Literal(Q.commutative, False), Literal(Q.finite, True))),
        frozenset((Literal(Q.commutative, False), Literal(Q.infinite, True))),
        frozenset((Literal(Q.composite, False), Literal(Q.even, True), Literal(Q.positive, True), Literal(Q.prime, False))),
        frozenset((Literal(Q.composite, True), Literal(Q.even, False), Literal(Q.odd, False))),
        frozenset((Literal(Q.composite, True), Literal(Q.positive, False))),
        frozenset((Literal(Q.composite, True), Literal(Q.prime, True))),
        frozenset((Literal(Q.even, False), Literal(Q.odd, False), Literal(Q.prime, True))),
        frozenset((Literal(Q.even, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.even, True), Literal(Q.odd, True))),
        frozenset((Literal(Q.even, True), Literal(Q.rational, False))),
        frozenset((Literal(Q.finite, False), Literal(Q.transcendental, True))),
        frozenset((Literal(Q.finite, True), Literal(Q.infinite, True))),
        frozenset((Literal(Q.hermitian, False), Literal(Q.negative, True))),
        frozenset((Literal(Q.hermitian, False), Literal(Q.positive, True))),
        frozenset((Literal(Q.hermitian, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.imaginary, True), Literal(Q.negative, True))),
        frozenset((Literal(Q.imaginary, True), Literal(Q.positive, True))),
        frozenset((Literal(Q.imaginary, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.infinite, False), Literal(Q.negative_infinite, True))),
        frozenset((Literal(Q.infinite, False), Literal(Q.positive_infinite, True))),
        frozenset((Literal(Q.irrational, False), Literal(Q.negative, True), Literal(Q.rational, False))),
        frozenset((Literal(Q.irrational, False), Literal(Q.positive, True), Literal(Q.rational, False))),
        frozenset((Literal(Q.irrational, False), Literal(Q.rational, False), Literal(Q.zero, True))),
        frozenset((Literal(Q.irrational, True), Literal(Q.negative, False), Literal(Q.positive, False), Literal(Q.zero, False))),
        frozenset((Literal(Q.irrational, True), Literal(Q.rational, True))),
        frozenset((Literal(Q.negative, False), Literal(Q.positive, False), Literal(Q.rational, True), Literal(Q.zero, False))),
        frozenset((Literal(Q.negative, True), Literal(Q.negative_infinite, True))),
        frozenset((Literal(Q.negative, True), Literal(Q.positive, True))),
        frozenset((Literal(Q.negative, True), Literal(Q.positive_infinite, True))),
        frozenset((Literal(Q.negative, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.negative_infinite, True), Literal(Q.positive, True))),
        frozenset((Literal(Q.negative_infinite, True), Literal(Q.positive_infinite, True))),
        frozenset((Literal(Q.negative_infinite, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.odd, True), Literal(Q.rational, False))),
        frozenset((Literal(Q.positive, False), Literal(Q.prime, True))),
        frozenset((Literal(Q.positive, True), Literal(Q.positive_infinite, True))),
        frozenset((Literal(Q.positive, True), Literal(Q.zero, True))),
        frozenset((Literal(Q.positive_infinite, True), Literal(Q.zero, True)))
    }

@cacheit
def get_known_facts_dict():
    """
    Logical relations between unary predicates as dictionary.

    Each key is a predicate, and item is two groups of predicates.
    First group contains the predicates which are implied by the key, and
    second group contains the predicates which are rejected by the key.

    """
    return {
        Q.algebraic: (set([Q.algebraic, Q.commutative, Q.complex, Q.finite]),
        set([Q.infinite, Q.negative_infinite, Q.positive_infinite,
        Q.transcendental])),
        Q.antihermitian: (set([Q.antihermitian]), set([])),
        Q.commutative: (set([Q.commutative]), set([])),
        Q.complex: (set([Q.commutative, Q.complex, Q.finite]),
        set([Q.infinite, Q.negative_infinite, Q.positive_infinite])),
        Q.complex_elements: (set([Q.complex_elements]), set([])),
        Q.composite: (set([Q.algebraic, Q.commutative, Q.complex, Q.composite,
        Q.extended_nonnegative, Q.extended_nonzero,
        Q.extended_positive, Q.extended_real, Q.finite, Q.hermitian,
        Q.integer, Q.nonnegative, Q.nonzero, Q.positive, Q.rational,
        Q.real]), set([Q.extended_negative, Q.extended_nonpositive,
        Q.imaginary, Q.infinite, Q.irrational, Q.negative,
        Q.negative_infinite, Q.nonpositive, Q.positive_infinite,
        Q.prime, Q.transcendental, Q.zero])),
        Q.diagonal: (set([Q.diagonal, Q.lower_triangular, Q.normal, Q.square,
        Q.symmetric, Q.triangular, Q.upper_triangular]), set([])),
        Q.even: (set([Q.algebraic, Q.commutative, Q.complex, Q.even,
        Q.extended_real, Q.finite, Q.hermitian, Q.integer, Q.rational,
        Q.real]), set([Q.imaginary, Q.infinite, Q.irrational,
        Q.negative_infinite, Q.odd, Q.positive_infinite,
        Q.transcendental])),
        Q.extended_negative: (set([Q.commutative, Q.extended_negative,
        Q.extended_nonpositive, Q.extended_nonzero, Q.extended_real]),
        set([Q.composite, Q.extended_nonnegative, Q.extended_positive,
        Q.imaginary, Q.nonnegative, Q.positive, Q.positive_infinite,
        Q.prime, Q.zero])),
        Q.extended_nonnegative: (set([Q.commutative, Q.extended_nonnegative,
        Q.extended_real]), set([Q.extended_negative, Q.imaginary,
        Q.negative, Q.negative_infinite])),
        Q.extended_nonpositive: (set([Q.commutative, Q.extended_nonpositive,
        Q.extended_real]), set([Q.composite, Q.extended_positive,
        Q.imaginary, Q.positive, Q.positive_infinite, Q.prime])),
        Q.extended_nonzero: (set([Q.commutative, Q.extended_nonzero,
        Q.extended_real]), set([Q.imaginary, Q.zero])),
        Q.extended_positive: (set([Q.commutative, Q.extended_nonnegative,
        Q.extended_nonzero, Q.extended_positive, Q.extended_real]),
        set([Q.extended_negative, Q.extended_nonpositive, Q.imaginary,
        Q.negative, Q.negative_infinite, Q.nonpositive, Q.zero])),
        Q.extended_real: (set([Q.commutative, Q.extended_real]),
        set([Q.imaginary])),
        Q.finite: (set([Q.commutative, Q.finite]), set([Q.infinite,
        Q.negative_infinite, Q.positive_infinite])),
        Q.fullrank: (set([Q.fullrank]), set([])),
        Q.hermitian: (set([Q.hermitian]), set([])),
        Q.imaginary: (set([Q.antihermitian, Q.commutative, Q.complex,
        Q.finite, Q.imaginary]), set([Q.composite, Q.even,
        Q.extended_negative, Q.extended_nonnegative,
        Q.extended_nonpositive, Q.extended_nonzero,
        Q.extended_positive, Q.extended_real, Q.infinite, Q.integer,
        Q.irrational, Q.negative, Q.negative_infinite, Q.nonnegative,
        Q.nonpositive, Q.nonzero, Q.odd, Q.positive,
        Q.positive_infinite, Q.prime, Q.rational, Q.real, Q.zero])),
        Q.infinite: (set([Q.commutative, Q.infinite]), set([Q.algebraic,
        Q.complex, Q.composite, Q.even, Q.finite, Q.imaginary,
        Q.integer, Q.irrational, Q.negative, Q.nonnegative,
        Q.nonpositive, Q.nonzero, Q.odd, Q.positive, Q.prime,
        Q.rational, Q.real, Q.transcendental, Q.zero])),
        Q.integer: (set([Q.algebraic, Q.commutative, Q.complex,
        Q.extended_real, Q.finite, Q.hermitian, Q.integer, Q.rational,
        Q.real]), set([Q.imaginary, Q.infinite, Q.irrational,
        Q.negative_infinite, Q.positive_infinite, Q.transcendental])),
        Q.integer_elements: (set([Q.complex_elements, Q.integer_elements,
        Q.real_elements]), set([])),
        Q.invertible: (set([Q.fullrank, Q.invertible, Q.square]),
        set([Q.singular])),
        Q.irrational: (set([Q.commutative, Q.complex, Q.extended_nonzero,
        Q.extended_real, Q.finite, Q.hermitian, Q.irrational,
        Q.nonzero, Q.real]), set([Q.composite, Q.even, Q.imaginary,
        Q.infinite, Q.integer, Q.negative_infinite, Q.odd,
        Q.positive_infinite, Q.prime, Q.rational, Q.zero])),
        Q.is_true: (set([Q.is_true]), set([])),
        Q.lower_triangular: (set([Q.lower_triangular, Q.triangular]), set([])),
        Q.negative: (set([Q.commutative, Q.complex, Q.extended_negative,
        Q.extended_nonpositive, Q.extended_nonzero, Q.extended_real,
        Q.finite, Q.hermitian, Q.negative, Q.nonpositive, Q.nonzero,
        Q.real]), set([Q.composite, Q.extended_nonnegative,
        Q.extended_positive, Q.imaginary, Q.infinite,
        Q.negative_infinite, Q.nonnegative, Q.positive,
        Q.positive_infinite, Q.prime, Q.zero])),
        Q.negative_infinite: (set([Q.commutative, Q.extended_negative,
