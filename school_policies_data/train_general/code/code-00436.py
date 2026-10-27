        and vertical distances to point p.

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> p1.taxicab_distance(p2)
        7

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return Add(*(abs(a - b) for a, b in zip(s, p)))

    def canberra_distance(self, p):
        """The Canberra Distance from self to point p.

        Returns the weighted sum of horizontal and vertical distances to
        point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        canberra_distance : The weighted sum of horizontal and vertical
        distances to point p. The weight used is the sum of absolute values
        of the coordinates.

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(3, 3)
        >>> p1.canberra_distance(p2)
        1
        >>> p1, p2 = Point(0, 0), Point(3, 3)
        >>> p1.canberra_distance(p2)
        2

        Raises
        ======

        ValueError when both vectors are zero.

        See Also
        ========

        sympy.geometry.point.Point.distance

        """

        s, p = Point._normalize_dimension(self, Point(p))
        if self.is_zero and p.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        return Add(*((abs(a - b)/(abs(a) + abs(b))) for a, b in zip(s, p)))

    @property
    def unit(self):
        """Return the Point that is in the same direction as `self`
        and a distance of 1 from the origin"""
        return self / abs(self)


class Point2D(Point):
    """A point in a 2-dimensional Euclidean space.

    Parameters
    ==========

    coords
        A sequence of 2 coordinate values.

    Attributes
    ==========

    x
    y
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When trying to create a point with more than two dimensions.
        When `intersection` is called with object other than a Point.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy import Point2D
    >>> from sympy.abc import x
    >>> Point2D(1, 2)
    Point2D(1, 2)
    >>> Point2D([1, 2])
    Point2D(1, 2)
    >>> Point2D(0, x)
    Point2D(0, x)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point2D(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point2D(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    _ambient_dimension = 2

    def __new__(cls, *args, _nocheck=False, **kwargs):
        if not _nocheck:
            kwargs['dim'] = 2
            args = Point(*args, **kwargs)
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        return item == self

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        return (self.x, self.y, self.x, self.y)

    def rotate(self, angle, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        See Also
        ========

        translate, scale

        Examples
        ========

        >>> from sympy import Point2D, pi
        >>> t = Point2D(1, 0)
        >>> t.rotate(pi/2)
        Point2D(0, 1)
        >>> t.rotate(pi/2, (2, 0))
        Point2D(2, -1)

        """
        c = cos(angle)
        s = sin(angle)

        rv = self
        if pt is not None:
            pt = Point(pt, dim=2)
            rv -= pt
        x, y = rv.args
        rv = Point(c*x - s*y, s*x + c*y)
        if pt is not None:
            rv += pt
        return rv

    def scale(self, x=1, y=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        rotate, translate

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(1, 1)
        >>> t.scale(2)
        Point2D(2, 1)
        >>> t.scale(2, 2)
        Point2D(2, 2)

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        return Point(self.x*x, self.y*y)

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 3x3 Matrix, ``matrix``.

        See Also
        ========
        sympy.geometry.point.Point2D.rotate
        sympy.geometry.point.Point2D.scale
        sympy.geometry.point.Point2D.translate
        """
        if not (matrix.is_Matrix and matrix.shape == (3, 3)):
            raise ValueError("matrix must be a 3x3 matrix")
        x, y = self.args
        return Point(*(Matrix(1, 3, [x, y, 1])*matrix).tolist()[0][:2])

    def translate(self, x=0, y=0):
        """Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========

        sympy.geometry.point.Point2D.rotate, scale

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(0, 1)
        >>> t.translate(2)
        Point2D(2, 1)
        >>> t.translate(2, 2)
        Point2D(2, 3)
        >>> t + Point2D(2, 2)
        Point2D(2, 3)

        """
        return Point(self.x + x, self.y + y)

    @property
    def coordinates(self):
        """
        Returns the two coordinates of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.coordinates
        (0, 1)
        """
        return self.args

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.y
        1
        """
        return self.args[1]

class Point3D(Point):
    """A point in a 3-dimensional Euclidean space.

    Parameters
    ==========

    coords
        A sequence of 3 coordinate values.

    Attributes
    ==========

    x
    y
    z
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When `intersection` is called with object other than a Point.

    Examples
    ========

    >>> from sympy import Point3D
    >>> from sympy.abc import x
    >>> Point3D(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point3D([1, 2, 3])
    Point3D(1, 2, 3)
    >>> Point3D(0, x, 3)
    Point3D(0, x, 3)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point3D(0.5, 0.25, 2)
    Point3D(1/2, 1/4, 2)
    >>> Point3D(0.5, 0.25, 3, evaluate=False)
    Point3D(0.5, 0.25, 3)

    """

    _ambient_dimension = 3

    def __new__(cls, *args, _nocheck=False, **kwargs):
        if not _nocheck:
            kwargs['dim'] = 3
            args = Point(*args, **kwargs)
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        return item == self

    @staticmethod
    def are_collinear(*points):
        """Is a sequence of points collinear?

        Test whether or not a set of points are collinear. Returns True if
        the set of points are collinear, or False otherwise.

        Parameters
        ==========

        points : sequence of Point

        Returns
        =======

        are_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line3D

        Examples
        ========

        >>> from sympy import Point3D
        >>> from sympy.abc import x
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(1, 1, 1)
        >>> p3, p4, p5 = Point3D(2, 2, 2), Point3D(x, x, x), Point3D(1, 2, 6)
        >>> Point3D.are_collinear(p1, p2, p3, p4)
        True
        >>> Point3D.are_collinear(p1, p2, p3, p5)
        False
        """
        return Point.is_collinear(*points)

    def direction_cosine(self, point):
        """
        Gives the direction cosine between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_cosine(Point3D(2, 3, 5))
        [sqrt(6)/6, sqrt(6)/6, sqrt(6)/3]
        """
        a = self.direction_ratio(point)
        b = sqrt(Add(*(i**2 for i in a)))
        return [(point.x - self.x) / b,(point.y - self.y) / b,
                (point.z - self.z) / b]

    def direction_ratio(self, point):
        """
        Gives the direction ratio between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_ratio(Point3D(2, 3, 5))
        [1, 1, 2]
        """
        return [(point.x - self.x),(point.y - self.y),(point.z - self.z)]

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : GeometryEntity or sequence of coordinates

        Returns
        =======

        intersection : list of Points

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point3D(0, 0, 0)]

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=3)
        if isinstance(other, Point3D):
            if self == other:
                return [self]
            return []
        return other.intersection(self)

    def scale(self, x=1, y=1, z=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        translate

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(1, 1, 1)
        >>> t.scale(2)
        Point3D(2, 1, 1)
        >>> t.scale(2, 2)
        Point3D(2, 2, 1)

        """
        if pt:
            pt = Point3D(pt)
            return self.translate(*(-pt).args).scale(x, y, z).translate(*pt.args)
        return Point3D(self.x*x, self.y*y, self.z*z)

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 4x4 Matrix, ``matrix``.

        See Also
        ========
        sympy.geometry.point.Point3D.scale
        sympy.geometry.point.Point3D.translate
        """
        if not (matrix.is_Matrix and matrix.shape == (4, 4)):
            raise ValueError("matrix must be a 4x4 matrix")
        x, y, z = self.args
        m = Transpose(matrix)
        return Point3D(*(Matrix(1, 4, [x, y, z, 1])*m).tolist()[0][:3])

    def translate(self, x=0, y=0, z=0):
        """Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========

        scale

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(0, 1, 1)
        >>> t.translate(2)
        Point3D(2, 1, 1)
        >>> t.translate(2, 2)
        Point3D(2, 3, 1)
        >>> t + Point3D(2, 2, 2)
        Point3D(2, 3, 3)

        """
        return Point3D(self.x + x, self.y + y, self.z + z)

    @property
    def coordinates(self):
        """
        Returns the three coordinates of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.coordinates
        (0, 1, 2)
        """
        return self.args

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 3)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.y
        1
        """
        return self.args[1]

    @property
    def z(self):
        """
        Returns the Z coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 1)
        >>> p.z
        1
        """
        return self.args[2]
<|endoftext|>"""The definition of the base geometrical entity with attributes common to
all derived geometrical entities.

Contains
========

GeometryEntity
GeometricSet

Notes
=====

A GeometryEntity is any object that has special geometric properties.
A GeometrySet is a superclass of any GeometryEntity that can also
be viewed as a sympy.sets.Set.  In particular, points are the only
GeometryEntity not considered a Set.

Rn is a GeometrySet representing n-dimensional Euclidean space. R2 and
R3 are currently the only ambient spaces implemented.

"""
from __future__ import annotations

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin, N
from sympy.core.numbers import oo
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin, atan
from sympy.matrices import eye
from sympy.multipledispatch import dispatch
from sympy.printing import sstr
from sympy.sets import Set, Union, FiniteSet
from sympy.sets.handlers.intersection import intersection_sets
from sympy.sets.handlers.union import union_sets
from sympy.solvers.solvers import solve
from sympy.utilities.misc import func_name
from sympy.utilities.iterables import is_sequence


# How entities are ordered; used by __cmp__ in GeometryEntity
ordering_of_classes = [
    "Point2D",
    "Point3D",
    "Point",
    "Segment2D",
    "Ray2D",
    "Line2D",
    "Segment3D",
    "Line3D",
    "Ray3D",
    "Segment",
    "Ray",
    "Line",
    "Plane",
    "Triangle",
    "RegularPolygon",
    "Polygon",
    "Circle",
    "Ellipse",
    "Curve",
    "Parabola"
]


x, y = [Dummy('entity_dummy') for i in range(2)]
T = Dummy('entity_dummy', real=True)


class GeometryEntity(Basic, EvalfMixin):
    """The base class for all geometrical entities.

    This class does not represent any particular geometric entity, it only
    provides the implementation of some methods common to all subclasses.

    """

    __slots__: tuple[str, ...] = ()

    def __cmp__(self, other):
        """Comparison of two GeometryEntities."""
        n1 = self.__class__.__name__
        n2 = other.__class__.__name__
        c = (n1 > n2) - (n1 < n2)
        if not c:
            return 0

        i1 = -1
        for cls in self.__class__.__mro__:
            try:
                i1 = ordering_of_classes.index(cls.__name__)
                break
            except ValueError:
                i1 = -1
        if i1 == -1:
            return c

        i2 = -1
        for cls in other.__class__.__mro__:
            try:
                i2 = ordering_of_classes.index(cls.__name__)
                break
            except ValueError:
                i2 = -1
        if i2 == -1:
            return c

        return (i1 > i2) - (i1 < i2)

    def __contains__(self, other):
        """Subclasses should implement this method for anything more complex than equality."""
        if type(self) is type(other):
            return self == other
        raise NotImplementedError()

    def __getnewargs__(self):
        """Returns a tuple that will be passed to __new__ on unpickling."""
        return tuple(self.args)

    def __ne__(self, o):
        """Test inequality of two geometrical entities."""
        return not self == o

    def __new__(cls, *args, **kwargs):
        # Points are sequences, but they should not
        # be converted to Tuples, so use this detection function instead.
        def is_seq_and_not_point(a):
            # we cannot use isinstance(a, Point) since we cannot import Point
            if hasattr(a, 'is_Point') and a.is_Point:
                return False
            return is_sequence(a)

        args = [Tuple(*a) if is_seq_and_not_point(a) else sympify(a) for a in args]
        return Basic.__new__(cls, *args)

    def __radd__(self, a):
        """Implementation of reverse add method."""
        return a.__add__(self)

    def __rtruediv__(self, a):
        """Implementation of reverse division method."""
        return a.__truediv__(self)

    def __repr__(self):
        """String representation of a GeometryEntity that can be evaluated
        by sympy."""
        return type(self).__name__ + repr(self.args)

    def __rmul__(self, a):
        """Implementation of reverse multiplication method."""
        return a.__mul__(self)

    def __rsub__(self, a):
        """Implementation of reverse subtraction method."""
        return a.__sub__(self)

    def __str__(self):
        """String representation of a GeometryEntity."""
        return type(self).__name__ + sstr(self.args)

    def _eval_subs(self, old, new):
        from sympy.geometry.point import Point, Point3D
        if is_sequence(old) or is_sequence(new):
            if isinstance(self, Point3D):
                old = Point3D(old)
                new = Point3D(new)
            else:
                old = Point(old)
                new = Point(new)
            return  self._subs(old, new)

    def _repr_svg_(self):
        """SVG representation of a GeometryEntity suitable for IPython"""

        try:
            bounds = self.bounds
        except (NotImplementedError, TypeError):
            # if we have no SVG representation, return None so IPython
            # will fall back to the next representation
            return None

        if not all(x.is_number and x.is_finite for x in bounds):
            return None

        svg_top = '''<svg xmlns="http://www.w3.org/2000/svg"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            width="{1}" height="{2}" viewBox="{0}"
            preserveAspectRatio="xMinYMin meet">
            <defs>
                <marker id="markerCircle" markerWidth="8" markerHeight="8"
                    refx="5" refy="5" markerUnits="strokeWidth">
                    <circle cx="5" cy="5" r="1.5" style="stroke: none; fill:#000000;"/>
                </marker>
                <marker id="markerArrow" markerWidth="13" markerHeight="13" refx="2" refy="4"
                       orient="auto" markerUnits="strokeWidth">
                    <path d="M2,2 L2,6 L6,4" style="fill: #000000;" />
                </marker>
                <marker id="markerReverseArrow" markerWidth="13" markerHeight="13" refx="6" refy="4"
                       orient="auto" markerUnits="strokeWidth">
                    <path d="M6,2 L6,6 L2,4" style="fill: #000000;" />
                </marker>
            </defs>'''

        # Establish SVG canvas that will fit all the data + small space
        xmin, ymin, xmax, ymax = map(N, bounds)
        if xmin == xmax and ymin == ymax:
            # This is a point; buffer using an arbitrary size
            xmin, ymin, xmax, ymax = xmin - .5, ymin -.5, xmax + .5, ymax + .5
        else:
            # Expand bounds by a fraction of the data ranges
            expand = 0.1  # or 10%; this keeps arrowheads in view (R plots use 4%)
            widest_part = max([xmax - xmin, ymax - ymin])
            expand_amount = widest_part * expand
            xmin -= expand_amount
            ymin -= expand_amount
            xmax += expand_amount
            ymax += expand_amount
        dx = xmax - xmin
        dy = ymax - ymin
        width = min([max([100., dx]), 300])
        height = min([max([100., dy]), 300])

        scale_factor = 1. if max(width, height) == 0 else max(dx, dy) / max(width, height)
        try:
            svg = self._svg(scale_factor)
        except (NotImplementedError, TypeError):
            # if we have no SVG representation, return None so IPython
            # will fall back to the next representation
            return None

        view_box = "{} {} {} {}".format(xmin, ymin, dx, dy)
        transform = "matrix(1,0,0,-1,0,{})".format(ymax + ymin)
        svg_top = svg_top.format(view_box, width, height)

        return svg_top + (
            '<g transform="{}">{}</g></svg>'
            ).format(transform, svg)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the GeometryEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        raise NotImplementedError()

    def _sympy_(self):
        return self

    @property
    def ambient_dimension(self):
        """What is the dimension of the space that the object is contained in?"""
        raise NotImplementedError()

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        raise NotImplementedError()

    def encloses(self, o):
        """
        Return True if o is inside (not on or outside) the boundaries of self.

        The object will be decomposed into Points and individual Entities need
        only define an encloses_point method for their class.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point
        sympy.geometry.polygon.Polygon.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t  = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t2 = Polygon(*RegularPolygon(Point(0, 0), 2, 3).vertices)
        >>> t2.encloses(t)
        True
        >>> t.encloses(t2)
        False

        """

        from sympy.geometry.point import Point
        from sympy.geometry.line import Segment, Ray, Line
        from sympy.geometry.ellipse import Ellipse
        from sympy.geometry.polygon import Polygon, RegularPolygon

        if isinstance(o, Point):
            return self.encloses_point(o)
        elif isinstance(o, Segment):
            return all(self.encloses_point(x) for x in o.points)
        elif isinstance(o, (Ray, Line)):
            return False
        elif isinstance(o, Ellipse):
            return self.encloses_point(o.center) and \
                self.encloses_point(
                Point(o.center.x + o.hradius, o.center.y)) and \
                not self.intersection(o)
        elif isinstance(o, Polygon):
            if isinstance(o, RegularPolygon):
                if not self.encloses_point(o.center):
                    return False
            return all(self.encloses_point(v) for v in o.vertices)
        raise NotImplementedError()

    def equals(self, o):
        return self == o

    def intersection(self, o):
        """
        Returns a list of all of the intersections of self with o.

        Notes
        =====

        An entity is not required to implement this method.

        If two different types of entities can intersect, the item with
        higher index in ordering_of_classes should implement
        intersections with anything having a lower index.

        See Also
        ========

        sympy.geometry.util.intersection

        """
        raise NotImplementedError()

    def is_similar(self, other):
        """Is this geometrical entity similar to another geometrical entity?

        Two entities are similar if a uniform scaling (enlarging or
        shrinking) of one of the entities will allow one to obtain the other.

        Notes
        =====

        This method is not intended to be used directly but rather
        through the `are_similar` function found in util.py.
        An entity is not required to implement this method.
        If two different types of entities can be similar, it is only
        required that one of them be able to determine this.

        See Also
        ========

        scale

        """
        raise NotImplementedError()

    def reflect(self, line):
        """
        Reflects an object across a line.

        Parameters
        ==========

        line: Line

        Examples
        ========

        >>> from sympy import pi, sqrt, Line, RegularPolygon
        >>> l = Line((0, pi), slope=sqrt(2))
        >>> pent = RegularPolygon((1, 2), 1, 5)
        >>> rpent = pent.reflect(l)
        >>> rpent
        RegularPolygon(Point2D(-2*sqrt(2)*pi/3 - 1/3 + 4*sqrt(2)/3, 2/3 + 2*sqrt(2)/3 + 2*pi/3), -1, 5, -atan(2*sqrt(2)) + 3*pi/5)

        >>> from sympy import pi, Line, Circle, Point
        >>> l = Line((0, pi), slope=1)
        >>> circ = Circle(Point(0, 0), 5)
        >>> rcirc = circ.reflect(l)
        >>> rcirc
        Circle(Point2D(-pi, pi), -5)

        """
        from sympy.geometry.point import Point

        g = self
        l = line
        o = Point(0, 0)
        if l.slope.is_zero:
            v = l.args[0].y
            if not v:  # x-axis
                return g.scale(y=-1)
            reps = [(p, p.translate(y=2*(v - p.y))) for p in g.atoms(Point)]
        elif l.slope is oo:
            v = l.args[0].x
            if not v:  # y-axis
                return g.scale(x=-1)
            reps = [(p, p.translate(x=2*(v - p.x))) for p in g.atoms(Point)]
        else:
            if not hasattr(g, 'reflect') and not all(
                    isinstance(arg, Point) for arg in g.args):
                raise NotImplementedError(
                    'reflect undefined or non-Point args in %s' % g)
            a = atan(l.slope)
            c = l.coefficients
            d = -c[-1]/c[1]  # y-intercept
            # apply the transform to a single point
            xf = Point(x, y)
            xf = xf.translate(y=-d).rotate(-a, o).scale(y=-1
                ).rotate(a, o).translate(y=d)
            # replace every point using that transform
            reps = [(p, xf.xreplace({x: p.x, y: p.y})) for p in g.atoms(Point)]
        return g.xreplace(dict(reps))

    def rotate(self, angle, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        The default pt is the origin, Point(0, 0)

        See Also
        ========

        scale, translate

        Examples
        ========

        >>> from sympy import Point, RegularPolygon, Polygon, pi
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t # vertex on x axis
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.rotate(pi/2) # vertex on y axis now
        Triangle(Point2D(0, 1), Point2D(-sqrt(3)/2, -1/2), Point2D(sqrt(3)/2, -1/2))

        """
        newargs = []
        for a in self.args:
            if isinstance(a, GeometryEntity):
                newargs.append(a.rotate(angle, pt))
            else:
                newargs.append(a)
        return type(self)(*newargs)

    def scale(self, x=1, y=1, pt=None):
        """Scale the object by multiplying the x,y-coordinates by x and y.

        If pt is given, the scaling is done relative to that point; the
        object is shifted by -pt, scaled, and shifted by pt.

        See Also
        ========

        rotate, translate

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.scale(2)
        Triangle(Point2D(2, 0), Point2D(-1, sqrt(3)/2), Point2D(-1, -sqrt(3)/2))
        >>> t.scale(2, 2)
        Triangle(Point2D(2, 0), Point2D(-1, sqrt(3)), Point2D(-1, -sqrt(3)))

        """
        from sympy.geometry.point import Point
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        return type(self)(*[a.scale(x, y) for a in self.args])  # if this fails, override this class

    def translate(self, x=0, y=0):
        """Shift the object by adding to the x,y-coordinates the values x and y.

        See Also
        ========

        rotate, scale

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.translate(2)
        Triangle(Point2D(3, 0), Point2D(3/2, sqrt(3)/2), Point2D(3/2, -sqrt(3)/2))
        >>> t.translate(2, 2)
        Triangle(Point2D(3, 2), Point2D(3/2, sqrt(3)/2 + 2), Point2D(3/2, 2 - sqrt(3)/2))

        """
        newargs = []
        for a in self.args:
            if isinstance(a, GeometryEntity):
                newargs.append(a.translate(x, y))
            else:
                newargs.append(a)
        return self.func(*newargs)

    def parameter_value(self, other, t):
        """Return the parameter corresponding to the given point.
        Evaluating an arbitrary point of the entity at this parameter
        value will return the given point.

        Examples
        ========

        >>> from sympy import Line, Point
        >>> from sympy.abc import t
        >>> a = Point(0, 0)
        >>> b = Point(2, 2)
        >>> Line(a, b).parameter_value((1, 1), t)
        {t: 1/2}
        >>> Line(a, b).arbitrary_point(t).subs(_)
        Point2D(1, 1)
        """
        from sympy.geometry.point import Point
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other, Point):
            raise ValueError("other must be a point")
        sol = solve(self.arbitrary_point(T) - other, T, dict=True)
        if not sol:
            raise ValueError("Given point is not on %s" % func_name(self))
        return {t: sol[0][T]}


class GeometrySet(GeometryEntity, Set):
    """Parent class of all GeometryEntity that are also Sets
    (compatible with sympy.sets)
    """
    __slots__ = ()

    def _contains(self, other):
        """sympy.sets uses the _contains method, so include it for compatibility."""

        if isinstance(other, Set) and other.is_FiniteSet:
            return all(self.__contains__(i) for i in other)

        return self.__contains__(other)

@dispatch(GeometrySet, Set)  # type:ignore # noqa:F811
def union_sets(self, o): # noqa:F811
    """ Returns the union of self and o
    for use with sympy.sets.Set, if possible. """


    # if its a FiniteSet, merge any points
    # we contain and return a union with the rest
    if o.is_FiniteSet:
        other_points = [p for p in o if not self._contains(p)]
        if len(other_points) == len(o):
            return None
        return Union(self, FiniteSet(*other_points))
    if self._contains(o):
        return self
    return None


@dispatch(GeometrySet, Set)  # type: ignore # noqa:F811
def intersection_sets(self, o): # noqa:F811
    """ Returns a sympy.sets.Set of intersection objects,
    if possible. """

    from sympy.geometry.point import Point

    try:
        # if o is a FiniteSet, find the intersection directly
        # to avoid infinite recursion
        if o.is_FiniteSet:
            inter = FiniteSet(*(p for p in o if self.contains(p)))
        else:
            inter = self.intersection(o)
    except NotImplementedError:
        # sympy.sets.Set.reduce expects None if an object
        # doesn't know how to simplify
        return None

    # put the points in a FiniteSet
    points = FiniteSet(*[p for p in inter if isinstance(p, Point)])
    non_points = [p for p in inter if not isinstance(p, Point)]

    return Union(*(non_points + [points]))

def translate(x, y):
    """Return the matrix to translate a 2-D point by x and y."""
    rv = eye(3)
    rv[2, 0] = x
    rv[2, 1] = y
    return rv


def scale(x, y, pt=None):
    """Return the matrix to multiply a 2-D point's coordinates by x and y.

    If pt is given, the scaling is done relative to that point."""
    rv = eye(3)
    rv[0, 0] = x
    rv[1, 1] = y
    if pt:
        from sympy.geometry.point import Point
        pt = Point(pt, dim=2)
        tr1 = translate(*(-pt).args)
        tr2 = translate(*pt.args)
        return tr1*rv*tr2
    return rv


def rotate(th):
    """Return the matrix to rotate a 2-D point about the origin by ``angle``.

    The angle is measured in radians. To Point a point about a point other
    then the origin, translate the Point, do the rotation, and
    translate it back:

    >>> from sympy.geometry.entity import rotate, translate
    >>> from sympy import Point, pi
    >>> rot_about_11 = translate(-1, -1)*rotate(pi/2)*translate(1, 1)
    >>> Point(1, 1).transform(rot_about_11)
    Point2D(1, 1)
    >>> Point(0, 0).transform(rot_about_11)
    Point2D(2, 0)
    """
    s = sin(th)
    rv = eye(3)*cos(th)
    rv[0, 1] = s
    rv[1, 0] = -s
    rv[2, 2] = 1
    return rv
<|endoftext|>"""Utility functions for geometrical entities.

Contains
========
intersection
convex_hull
closest_points
farthest_points
are_coplanar
are_similar

"""

from collections import deque
from math import sqrt as _sqrt

from sympy import nsimplify
from .entity import GeometryEntity
from .exceptions import GeometryError
from .point import Point, Point2D, Point3D
from sympy.core.containers import OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, expand_mul
from sympy.core.numbers import Float
from sympy.core.sorting import ordered
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.polys.polytools import cancel
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.iterables import is_sequence

from mpmath.libmp.libmpf import prec_to_dps


def find(x, equation):
    """
    Checks whether a Symbol matching ``x`` is present in ``equation``
    or not. If present, the matching symbol is returned, else a
    ValueError is raised. If ``x`` is a string the matching symbol
    will have the same name; if ``x`` is a Symbol then it will be
    returned if found.

    Examples
    ========

    >>> from sympy.geometry.util import find
    >>> from sympy import Dummy
    >>> from sympy.abc import x
    >>> find('x', x)
    x
    >>> find('x', Dummy('x'))
    _x

    The dummy symbol is returned since it has a matching name:

    >>> _.name == 'x'
    True
    >>> find(x, Dummy('x'))
    Traceback (most recent call last):
    ...
    ValueError: could not find x
    """

    free = equation.free_symbols
    xs = [i for i in free if (i.name if isinstance(x, str) else i) == x]
    if not xs:
        raise ValueError('could not find %s' % x)
    if len(xs) != 1:
        raise ValueError('ambiguous %s' % x)
    return xs[0]


def _ordered_points(p):
    """Return the tuple of points sorted numerically according to args"""
    return tuple(sorted(p, key=lambda x: x.args))


def are_coplanar(*e):
    """ Returns True if the given entities are coplanar otherwise False

    Parameters
    ==========

    e: entities to be checked for being coplanar

    Returns
    =======

    Boolean

    Examples
    ========

    >>> from sympy import Point3D, Line3D
    >>> from sympy.geometry.util import are_coplanar
    >>> a = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    >>> b = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    >>> c = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))
    >>> are_coplanar(a, b, c)
    False

    """
    from .line import LinearEntity3D
    from .plane import Plane
    # XXX update tests for coverage

    e = set(e)
    # first work with a Plane if present
    for i in list(e):
        if isinstance(i, Plane):
            e.remove(i)
            return all(p.is_coplanar(i) for p in e)

    if all(isinstance(i, Point3D) for i in e):
        if len(e) < 3:
            return False

        # remove pts that are collinear with 2 pts
        a, b = e.pop(), e.pop()
        for i in list(e):
            if Point3D.are_collinear(a, b, i):
                e.remove(i)

        if not e:
            return False
        else:
            # define a plane
            p = Plane(a, b, e.pop())
            for i in e:
                if i not in p:
                    return False
            return True
    else:
        pt3d = []
        for i in e:
            if isinstance(i, Point3D):
                pt3d.append(i)
            elif isinstance(i, LinearEntity3D):
                pt3d.extend(i.args)
            elif isinstance(i, GeometryEntity):  # XXX we should have a GeometryEntity3D class so we can tell the difference between 2D and 3D -- here we just want to deal with 2D objects; if new 3D objects are encountered that we didn't handle above, an error should be raised
                # all 2D objects have some Point that defines them; so convert those points to 3D pts by making z=0
                for p in i.args:
                    if isinstance(p, Point):
                        pt3d.append(Point3D(*(p.args + (0,))))
        return are_coplanar(*pt3d)


def are_similar(e1, e2):
    """Are two geometrical entities similar.

    Can one geometrical entity be uniformly scaled to the other?

    Parameters
    ==========

    e1 : GeometryEntity
    e2 : GeometryEntity

    Returns
    =======

    are_similar : boolean

    Raises
    ======

    GeometryError
        When `e1` and `e2` cannot be compared.

    Notes
    =====

    If the two objects are equal then they are similar.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity.is_similar

    Examples
    ========

    >>> from sympy import Point, Circle, Triangle, are_similar
    >>> c1, c2 = Circle(Point(0, 0), 4), Circle(Point(1, 4), 3)
    >>> t1 = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
    >>> t2 = Triangle(Point(0, 0), Point(2, 0), Point(0, 2))
    >>> t3 = Triangle(Point(0, 0), Point(3, 0), Point(0, 1))
    >>> are_similar(t1, t2)
    True
    >>> are_similar(t1, t3)
    False

    """
    if e1 == e2:
        return True
    is_similar1 = getattr(e1, 'is_similar', None)
    if is_similar1:
        return is_similar1(e2)
    is_similar2 = getattr(e2, 'is_similar', None)
    if is_similar2:
        return is_similar2(e1)
    n1 = e1.__class__.__name__
    n2 = e2.__class__.__name__
    raise GeometryError(
        "Cannot test similarity between %s and %s" % (n1, n2))


def centroid(*args):
    """Find the centroid (center of mass) of the collection containing only Points,
    Segments or Polygons. The centroid is the weighted average of the individual centroid
    where the weights are the lengths (of segments) or areas (of polygons).
    Overlapping regions will add to the weight of that region.

    If there are no objects (or a mixture of objects) then None is returned.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.line.Segment,
    sympy.geometry.polygon.Polygon

    Examples
    ========

    >>> from sympy import Point, Segment, Polygon
    >>> from sympy.geometry.util import centroid
    >>> p = Polygon((0, 0), (10, 0), (10, 10))
    >>> q = p.translate(0, 20)
    >>> p.centroid, q.centroid
    (Point2D(20/3, 10/3), Point2D(20/3, 70/3))
    >>> centroid(p, q)
    Point2D(20/3, 40/3)
    >>> p, q = Segment((0, 0), (2, 0)), Segment((0, 0), (2, 2))
    >>> centroid(p, q)
    Point2D(1, 2 - sqrt(2))
    >>> centroid(Point(0, 0), Point(2, 0))
    Point2D(1, 0)

    Stacking 3 polygons on top of each other effectively triples the
    weight of that polygon:

    >>> p = Polygon((0, 0), (1, 0), (1, 1), (0, 1))
    >>> q = Polygon((1, 0), (3, 0), (3, 1), (1, 1))
    >>> centroid(p, q)
    Point2D(3/2, 1/2)
    >>> centroid(p, p, p, q) # centroid x-coord shifts left
    Point2D(11/10, 1/2)

    Stacking the squares vertically above and below p has the same
    effect:

    >>> centroid(p, p.translate(0, 1), p.translate(0, -1), q)
    Point2D(11/10, 1/2)

    """
    from .line import Segment
    from .polygon import Polygon
    if args:
        if all(isinstance(g, Point) for g in args):
            c = Point(0, 0)
            for g in args:
                c += g
            den = len(args)
        elif all(isinstance(g, Segment) for g in args):
            c = Point(0, 0)
            L = 0
            for g in args:
                l = g.length
                c += g.midpoint*l
                L += l
            den = L
        elif all(isinstance(g, Polygon) for g in args):
            c = Point(0, 0)
            A = 0
            for g in args:
                a = g.area
                c += g.centroid*a
                A += a
            den = A
        c /= den
        return c.func(*[i.simplify() for i in c.args])


def closest_points(*args):
    """Return the subset of points from a set of points that were
    the closest to each other in the 2D plane.

    Parameters
    ==========

    args
        A collection of Points on 2D plane.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line. If there are no ties then a single
    pair of Points will be in the set.

    Examples
    ========

    >>> from sympy import closest_points, Triangle
    >>> Triangle(sss=(3, 4, 5)).args
    (Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> closest_points(*_)
    {(Point2D(0, 0), Point2D(3, 0))}

    References
    ==========

    .. [1] https://www.cs.mcgill.ca/~cs251/ClosestPair/ClosestPairPS.html

    .. [2] Sweep line algorithm
        https://en.wikipedia.org/wiki/Sweep_line_algorithm

    """
    p = [Point2D(i) for i in set(args)]
    if len(p) < 2:
        raise ValueError('At least 2 distinct points must be given.')

    try:
        p.sort(key=lambda x: x.args)
    except TypeError:
        raise ValueError("The points could not be sorted.")

    if not all(i.is_Rational for j in p for i in j.args):
        def hypot(x, y):
            arg = x*x + y*y
            if arg.is_Rational:
                return _sqrt(arg)
            return sqrt(arg)
    else:
        from math import hypot

    rv = [(0, 1)]
    best_dist = hypot(p[1].x - p[0].x, p[1].y - p[0].y)
    left = 0
    box = deque([0, 1])
    for i in range(2, len(p)):
        while left < i and p[i][0] - p[left][0] > best_dist:
            box.popleft()
            left += 1

        for j in box:
            d = hypot(p[i].x - p[j].x, p[i].y - p[j].y)
            if d < best_dist:
                rv = [(j, i)]
            elif d == best_dist:
                rv.append((j, i))
            else:
                continue
            best_dist = d
        box.append(i)

    return {tuple([p[i] for i in pair]) for pair in rv}


def convex_hull(*args, polygon=True):
    """The convex hull surrounding the Points contained in the list of entities.

    Parameters
    ==========

    args : a collection of Points, Segments and/or Polygons

    Optional parameters
    ===================

    polygon : Boolean. If True, returns a Polygon, if false a tuple, see below.
              Default is True.

    Returns
    =======

    convex_hull : Polygon if ``polygon`` is True else as a tuple `(U, L)` where
                  ``L`` and ``U`` are the lower and upper hulls, respectively.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.polygon.Polygon

    Examples
    ========

    >>> from sympy import convex_hull
    >>> points = [(1, 1), (1, 2), (3, 1), (-5, 2), (15, 4)]
    >>> convex_hull(*points)
    Polygon(Point2D(-5, 2), Point2D(1, 1), Point2D(3, 1), Point2D(15, 4))
    >>> convex_hull(*points, **dict(polygon=False))
    ([Point2D(-5, 2), Point2D(15, 4)],
     [Point2D(-5, 2), Point2D(1, 1), Point2D(3, 1), Point2D(15, 4)])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Graham_scan

    .. [2] Andrew's Monotone Chain Algorithm
      (A.M. Andrew,
      "Another Efficient Algorithm for Convex Hulls in Two Dimensions", 1979)
      https://web.archive.org/web/20210511015444/http://geomalgorithms.com/a10-_hull-1.html

    """
    from .line import Segment
    from .polygon import Polygon
    p = OrderedSet()
    for e in args:
        if not isinstance(e, GeometryEntity):
            try:
                e = Point(e)
            except NotImplementedError:
                raise ValueError('%s is not a GeometryEntity and cannot be made into Point' % str(e))
        if isinstance(e, Point):
            p.add(e)
        elif isinstance(e, Segment):
            p.update(e.points)
        elif isinstance(e, Polygon):
            p.update(e.vertices)
        else:
            raise NotImplementedError(
                'Convex hull for %s not implemented.' % type(e))

    # make sure all our points are of the same dimension
    if any(len(x) != 2 for x in p):
        raise ValueError('Can only compute the convex hull in two dimensions')

    p = list(p)
    if len(p) == 1:
        return p[0] if polygon else (p[0], None)
    elif len(p) == 2:
        s = Segment(p[0], p[1])
        return s if polygon else (s, None)

    def _orientation(p, q, r):
        '''Return positive if p-q-r are clockwise, neg if ccw, zero if
        collinear.'''
        return (q.y - p.y)*(r.x - p.x) - (q.x - p.x)*(r.y - p.y)

    # scan to find upper and lower convex hulls of a set of 2d points.
    U = []
    L = []
    try:
        p.sort(key=lambda x: x.args)
    except TypeError:
        raise ValueError("The points could not be sorted.")
    for p_i in p:
        while len(U) > 1 and _orientation(U[-2], U[-1], p_i) <= 0:
            U.pop()
        while len(L) > 1 and _orientation(L[-2], L[-1], p_i) >= 0:
            L.pop()
        U.append(p_i)
        L.append(p_i)
    U.reverse()
    convexHull = tuple(L + U[1:-1])

    if len(convexHull) == 2:
        s = Segment(convexHull[0], convexHull[1])
        return s if polygon else (s, None)
    if polygon:
        return Polygon(*convexHull)
    else:
        U.reverse()
        return (U, L)

def farthest_points(*args):
    """Return the subset of points from a set of points that were
    the furthest apart from each other in the 2D plane.

    Parameters
    ==========

    args
        A collection of Points on 2D plane.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line. If there are no ties then a single
    pair of Points will be in the set.

    Examples
    ========

    >>> from sympy.geometry import farthest_points, Triangle
    >>> Triangle(sss=(3, 4, 5)).args
    (Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> farthest_points(*_)
    {(Point2D(0, 0), Point2D(3, 4))}

    References
    ==========

    .. [1] https://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/

    .. [2] Rotating Callipers Technique
        https://en.wikipedia.org/wiki/Rotating_calipers

    """

    def rotatingCalipers(Points):
        U, L = convex_hull(*Points, **{"polygon": False})

        if L is None:
            if isinstance(U, Point):
                raise ValueError('At least two distinct points must be given.')
            yield U.args
        else:
            i = 0
            j = len(L) - 1
            while i < len(U) - 1 or j > 0:
                yield U[i], L[j]
                # if all the way through one side of hull, advance the other side
                if i == len(U) - 1:
                    j -= 1
                elif j == 0:
                    i += 1
                # still points left on both lists, compare slopes of next hull edges
                # being careful to avoid divide-by-zero in slope calculation
                elif (U[i+1].y - U[i].y) * (L[j].x - L[j-1].x) > \
                        (L[j].y - L[j-1].y) * (U[i+1].x - U[i].x):
                    i += 1
                else:
                    j -= 1

    p = [Point2D(i) for i in set(args)]

    if not all(i.is_Rational for j in p for i in j.args):
        def hypot(x, y):
            arg = x*x + y*y
            if arg.is_Rational:
                return _sqrt(arg)
            return sqrt(arg)
    else:
        from math import hypot

    rv = []
    diam = 0
    for pair in rotatingCalipers(args):
        h, q = _ordered_points(pair)
        d = hypot(h.x - q.x, h.y - q.y)
        if d > diam:
            rv = [(h, q)]
        elif d == diam:
            rv.append((h, q))
        else:
            continue
        diam = d

    return set(rv)


def idiff(eq, y, x, n=1):
    """Return ``dy/dx`` assuming that ``eq == 0``.

    Parameters
    ==========

    y : the dependent variable or a list of dependent variables (with y first)
    x : the variable that the derivative is being taken with respect to
    n : the order of the derivative (default is 1)

    Examples
    ========

    >>> from sympy.abc import x, y, a
    >>> from sympy.geometry.util import idiff

    >>> circ = x**2 + y**2 - 4
    >>> idiff(circ, y, x)
    -x/y
    >>> idiff(circ, y, x, 2).simplify()
    (-x**2 - y**2)/y**3

    Here, ``a`` is assumed to be independent of ``x``:

    >>> idiff(x + a + y, y, x)
    -1

    Now the x-dependence of ``a`` is made explicit by listing ``a`` after
    ``y`` in a list.

    >>> idiff(x + a + y, [y, a], x)
    -Derivative(a, x) - 1

    See Also
    ========

    sympy.core.function.Derivative: represents unevaluated derivatives
    sympy.core.function.diff: explicitly differentiates wrt symbols

    """
    if is_sequence(y):
        dep = set(y)
        y = y[0]
    elif isinstance(y, Symbol):
        dep = {y}
    elif isinstance(y, Function):
        pass
    else:
        raise ValueError("expecting x-dependent symbol(s) or function(s) but got: %s" % y)

    f = {s: Function(s.name)(x) for s in eq.free_symbols
        if s != x and s in dep}

    if isinstance(y, Symbol):
        dydx = Function(y.name)(x).diff(x)
    else:
        dydx = y.diff(x)

    eq = eq.subs(f)
    derivs = {}
    for i in range(n):
        # equation will be linear in dydx, a*dydx + b, so dydx = -b/a
        deq = eq.diff(x)
        b = deq.xreplace({dydx: S.Zero})
        a = (deq - b).xreplace({dydx: S.One})
        yp = factor_terms(expand_mul(cancel((-b/a).subs(derivs)), deep=False))
        if i == n - 1:
            return yp.subs([(v, k) for k, v in f.items()])
        derivs[dydx] = yp
        eq = dydx - yp
        dydx = dydx.diff(x)


def intersection(*entities, pairwise=False, **kwargs):
    """The intersection of a collection of GeometryEntity instances.

    Parameters
    ==========
    entities : sequence of GeometryEntity
    pairwise (keyword argument) : Can be either True or False

    Returns
    =======
    intersection : list of GeometryEntity

    Raises
    ======
    NotImplementedError
        When unable to calculate intersection.

    Notes
    =====
    The intersection of any geometrical entity with itself should return
    a list with one item: the entity in question.
    An intersection requires two or more entities. If only a single
    entity is given then the function will return an empty list.
    It is possible for `intersection` to miss intersections that one
    knows exists because the required quantities were not fully
    simplified internally.
    Reals should be converted to Rationals, e.g. Rational(str(real_num))
    or else failures due to floating point issues may result.

    Case 1: When the keyword argument 'pairwise' is False (default value):
    In this case, the function returns a list of intersections common to
    all entities.

    Case 2: When the keyword argument 'pairwise' is True:
    In this case, the functions returns a list intersections that occur
    between any pair of entities.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity.intersection

    Examples
    ========

    >>> from sympy import Ray, Circle, intersection
    >>> c = Circle((0, 1), 1)
    >>> intersection(c, c.center)
    []
    >>> right = Ray((0, 0), (1, 0))
    >>> up = Ray((0, 0), (0, 1))
    >>> intersection(c, right, up)
    [Point2D(0, 0)]
    >>> intersection(c, right, up, pairwise=True)
    [Point2D(0, 0), Point2D(0, 2)]
    >>> left = Ray((1, 0), (0, 0))
    >>> intersection(right, left)
    [Segment2D(Point2D(0, 0), Point2D(1, 0))]

    """
    if len(entities) <= 1:
        return []

    entities = list(entities)
    prec = None
    for i, e in enumerate(entities):
        if not isinstance(e, GeometryEntity):
            # entities may be an immutable tuple
            e = Point(e)
        # convert to exact Rationals
        d = {}
        for f in e.atoms(Float):
            prec = f._prec if prec is None else min(f._prec, prec)
            d.setdefault(f, nsimplify(f, rational=True))
        entities[i] = e.xreplace(d)

    if not pairwise:
        # find the intersection common to all objects
        res = entities[0].intersection(entities[1])
        for entity in entities[2:]:
            newres = []
            for x in res:
                newres.extend(x.intersection(entity))
            res = newres
    else:
        # find all pairwise intersections
        ans = []
        for j in range(len(entities)):
            for k in range(j + 1, len(entities)):
                ans.extend(intersection(entities[j], entities[k]))
        res = list(ordered(set(ans)))

    # convert back to Floats
    if prec is not None:
        p = prec_to_dps(prec)
        res = [i.n(p) for i in res]
    return res
<|endoftext|>"""Elliptical geometrical entities.

Contains
* Ellipse
* Circle

"""

from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core import S, pi, sympify
from sympy.core.evalf import N
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.numbers import Rational, oo
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, uniquely_named_symbol, _symbol
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.elliptic_integrals import elliptic_e
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray2D, Segment2D, Line2D, LinearEntity3D
from .point import Point, Point2D, Point3D
from .util import idiff, find
from sympy.polys import DomainError, Poly, PolynomialError
from sympy.polys.polyutils import _not_a_coeff, _nsort
from sympy.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import filldedent, func_name

from mpmath.libmp.libmpf import prec_to_dps

import random

x, y = [Dummy('ellipse_dummy', real=True) for i in range(2)]


class Ellipse(GeometrySet):
    """An elliptical GeometryEntity.

    Parameters
    ==========

    center : Point, optional
        Default value is Point(0, 0)
    hradius : number or SymPy expression, optional
    vradius : number or SymPy expression, optional
    eccentricity : number or SymPy expression, optional
        Two of `hradius`, `vradius` and `eccentricity` must be supplied to
        create an Ellipse. The third is derived from the two supplied.

    Attributes
    ==========

    center
    hradius
    vradius
    area
    circumference
    eccentricity
    periapsis
    apoapsis
    focus_distance
    foci

    Raises
    ======

    GeometryError
        When `hradius`, `vradius` and `eccentricity` are incorrectly supplied
        as parameters.
    TypeError
        When `center` is not a Point.

    See Also
    ========

    Circle

    Notes
    -----
    Constructed from a center and two radii, the first being the horizontal
    radius (along the x-axis) and the second being the vertical radius (along
    the y-axis).

    When symbolic value for hradius and vradius are used, any calculation that
    refers to the foci or the major or minor axis will assume that the ellipse
    has its major radius on the x-axis. If this is not true then a manual
    rotation is necessary.

    Examples
    ========

    >>> from sympy import Ellipse, Point, Rational
    >>> e1 = Ellipse(Point(0, 0), 5, 1)
    >>> e1.hradius, e1.vradius
    (5, 1)
    >>> e2 = Ellipse(Point(3, 1), hradius=3, eccentricity=Rational(4, 5))
    >>> e2
    Ellipse(Point2D(3, 1), 3, 9/5)

    """

    def __contains__(self, o):
        if isinstance(o, Point):
            res = self.equation(x, y).subs({x: o.x, y: o.y})
            return trigsimp(simplify(res)) is S.Zero
        elif isinstance(o, Ellipse):
            return self == o
        return False

    def __eq__(self, o):
        """Is the other GeometryEntity the same as this ellipse?"""
        return isinstance(o, Ellipse) and (self.center == o.center and
                                           self.hradius == o.hradius and
                                           self.vradius == o.vradius)

    def __hash__(self):
        return super().__hash__()

    def __new__(
        cls, center=None, hradius=None, vradius=None, eccentricity=None, **kwargs):

        hradius = sympify(hradius)
        vradius = sympify(vradius)

        if center is None:
            center = Point(0, 0)
        else:
            if len(center) != 2:
                raise ValueError('The center of "{}" must be a two dimensional point'.format(cls))
            center = Point(center, dim=2)

        if len(list(filter(lambda x: x is not None, (hradius, vradius, eccentricity)))) != 2:
            raise ValueError(filldedent('''
                Exactly two arguments of "hradius", "vradius", and
                "eccentricity" must not be None.'''))

        if eccentricity is not None:
            eccentricity = sympify(eccentricity)
            if eccentricity.is_negative:
                raise GeometryError("Eccentricity of ellipse/circle should lie between [0, 1)")
            elif hradius is None:
                hradius = vradius / sqrt(1 - eccentricity**2)
            elif vradius is None:
                vradius = hradius * sqrt(1 - eccentricity**2)

        if hradius == vradius:
            return Circle(center, hradius, **kwargs)

        if S.Zero in (hradius, vradius):
            return Segment(Point(center[0] - hradius, center[1] - vradius), Point(center[0] + hradius, center[1] + vradius))

        if hradius.is_real is False or vradius.is_real is False:
            raise GeometryError("Invalid value encountered when computing hradius / vradius.")

        return GeometryEntity.__new__(cls, center, hradius, vradius, **kwargs)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG ellipse element for the Ellipse.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """

        c = N(self.center)
        h, v = N(self.hradius), N(self.vradius)
        return (
            '<ellipse fill="{1}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" cx="{2}" cy="{3}" rx="{4}" ry="{5}"/>'
        ).format(2. * scale_factor, fill_color, c.x, c.y, h, v)

    @property
    def ambient_dimension(self):
        return 2

    @property
    def apoapsis(self):
        """The apoapsis of the ellipse.

        The greatest distance between the focus and the contour.

        Returns
        =======

        apoapsis : number

        See Also
        ========

        periapsis : Returns shortest distance between foci and contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.apoapsis
        2*sqrt(2) + 3

        """
        return self.major * (1 + self.eccentricity)

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the ellipse.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        arbitrary_point : Point

        Raises
        ======

        ValueError
            When `parameter` already appears in the functions.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.arbitrary_point()
        Point2D(3*cos(t), 2*sin(t))

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('Symbol %s already appears in object '
                                        'and cannot be used as a parameter.' % t.name))
        return Point(self.center.x + self.hradius*cos(t),
                     self.center.y + self.vradius*sin(t))

    @property
    def area(self):
        """The area of the ellipse.

        Returns
        =======

        area : number

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.area
        3*pi

        """
        return simplify(S.Pi * self.hradius * self.vradius)

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        h, v = self.hradius, self.vradius
        return (self.center.x - h, self.center.y - v, self.center.x + h, self.center.y + v)

    @property
    def center(self):
        """The center of the ellipse.

        Returns
        =======

        center : number

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.center
        Point2D(0, 0)

        """
        return self.args[0]

    @property
    def circumference(self):
        """The circumference of the ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.circumference
        12*elliptic_e(8/9)

        """
        if self.eccentricity == 1:
            # degenerate
            return 4*self.major
        elif self.eccentricity == 0:
            # circle
            return 2*pi*self.hradius
        else:
            return 4*self.major*elliptic_e(self.eccentricity**2)

    @property
    def eccentricity(self):
        """The eccentricity of the ellipse.

        Returns
        =======

        eccentricity : number

        Examples
        ========

        >>> from sympy import Point, Ellipse, sqrt
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, sqrt(2))
        >>> e1.eccentricity
        sqrt(7)/3

        """
        return self.focus_distance / self.major

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        -----
        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Ellipse, S
        >>> from sympy.abc import t
        >>> e = Ellipse((0, 0), 3, 2)
        >>> e.encloses_point((0, 0))
        True
        >>> e.encloses_point(e.arbitrary_point(t).subs(t, S.Half))
        False
        >>> e.encloses_point((4, 0))
        False

        """
        p = Point(p, dim=2)
        if p in self:
            return False

        if len(self.foci) == 2:
            # if the combined distance from the foci to p (h1 + h2) is less
            # than the combined distance from the foci to the minor axis
            # (which is the same as the major axis length) then p is inside
            # the ellipse
            h1, h2 = [f.distance(p) for f in self.foci]
            test = 2*self.major - (h1 + h2)
        else:
            test = self.radius - self.center.distance(p)

        return fuzzy_bool(test.is_positive)

    def equation(self, x='x', y='y', _slope=None):
        """
        Returns the equation of an ellipse aligned with the x and y axes;
        when slope is given, the equation returned corresponds to an ellipse
        with a major axis having that slope.

        Parameters
        ==========

        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.
        _slope : Expr, optional
                The slope of the major axis. Ignored when 'None'.

        Returns
        =======

        equation : SymPy expression

        See Also
        ========

        arbitrary_point : Returns parameterized point on ellipse

        Examples
        ========

        >>> from sympy import Point, Ellipse, pi
        >>> from sympy.abc import x, y
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> eq1 = e1.equation(x, y); eq1
        y**2/4 + (x/3 - 1/3)**2 - 1
        >>> eq2 = e1.equation(x, y, _slope=1); eq2
        (-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1

        A point on e1 satisfies eq1. Let's use one on the x-axis:

        >>> p1 = e1.center + Point(e1.major, 0)
        >>> assert eq1.subs(x, p1.x).subs(y, p1.y) == 0

        When rotated the same as the rotated ellipse, about the center
        point of the ellipse, it will satisfy the rotated ellipse's
        equation, too:

        >>> r1 = p1.rotate(pi/4, e1.center)
        >>> assert eq2.subs(x, r1.x).subs(y, r1.y) == 0

        References
        ==========

        .. [1] https://math.stackexchange.com/questions/108270/what-is-the-equation-of-an-ellipse-that-is-not-aligned-with-the-axis
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Shifted_ellipse

        """

        x = _symbol(x, real=True)
        y = _symbol(y, real=True)

        dx = x - self.center.x
        dy = y - self.center.y

        if _slope is not None:
            L = (dy - _slope*dx)**2
            l = (_slope*dy + dx)**2
            h = 1 + _slope**2
            b = h*self.major**2
            a = h*self.minor**2
            return l/b + L/a - 1

        else:
            t1 = (dx/self.hradius)**2
            t2 = (dy/self.vradius)**2
            return t1 + t2 - 1

    def evolute(self, x='x', y='y'):
        """The equation of evolute of the ellipse.

        Parameters
        ==========

        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> e1.evolute()
        2**(2/3)*y**(2/3) + (3*x - 3)**(2/3) - 5**(2/3)
        """
        if len(self.args) != 3:
            raise NotImplementedError('Evolute of arbitrary Ellipse is not supported.')
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        t1 = (self.hradius*(x - self.center.x))**Rational(2, 3)
        t2 = (self.vradius*(y - self.center.y))**Rational(2, 3)
        return t1 + t2 - (self.hradius**2 - self.vradius**2)**Rational(2, 3)

    @property
    def foci(self):
        """The foci of the ellipse.

        Notes
        -----
        The foci can only be calculated if the major/minor axes are known.

        Raises
        ======

        ValueError
            When the major and minor axis cannot be determined.

        See Also
        ========

        sympy.geometry.point.Point
        focus_distance : Returns the distance between focus and center

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.foci
        (Point2D(-2*sqrt(2), 0), Point2D(2*sqrt(2), 0))

        """
        c = self.center
        hr, vr = self.hradius, self.vradius
        if hr == vr:
            return (c, c)

        # calculate focus distance manually, since focus_distance calls this
        # routine
        fd = sqrt(self.major**2 - self.minor**2)
        if hr == self.minor:
            # foci on the y-axis
            return (c + Point(0, -fd), c + Point(0, fd))
        elif hr == self.major:
            # foci on the x-axis
            return (c + Point(-fd, 0), c + Point(fd, 0))

    @property
    def focus_distance(self):
        """The focal distance of the ellipse.

        The distance between the center and one focus.

        Returns
        =======

        focus_distance : number

        See Also
        ========

        foci

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.focus_distance
        2*sqrt(2)

        """
        return Point.distance(self.center, self.foci[0])

    @property
    def hradius(self):
        """The horizontal radius of the ellipse.

        Returns
        =======

        hradius : number

        See Also
        ========

        vradius, major, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.hradius
        3

        """
        return self.args[1]

    def intersection(self, o):
        """The intersection of this ellipse and another geometrical entity
        `o`.

        Parameters
        ==========

        o : GeometryEntity

        Returns
        =======

        intersection : list of GeometryEntity objects

        Notes
        -----
        Currently supports intersections with Point, Line, Segment, Ray,
        Circle and Ellipse types.

        See Also
        ========

        sympy.geometry.entity.GeometryEntity

        Examples
        ========

        >>> from sympy import Ellipse, Point, Line
        >>> e = Ellipse(Point(0, 0), 5, 7)
        >>> e.intersection(Point(0, 0))
        []
        >>> e.intersection(Point(5, 0))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(0,0), Point(0, 1)))
        [Point2D(0, -7), Point2D(0, 7)]
        >>> e.intersection(Line(Point(5,0), Point(5, 1)))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(6,0), Point(6, 1)))
        []
        >>> e = Ellipse(Point(-1, 0), 4, 3)
        >>> e.intersection(Ellipse(Point(1, 0), 4, 3))
        [Point2D(0, -3*sqrt(15)/4), Point2D(0, 3*sqrt(15)/4)]
        >>> e.intersection(Ellipse(Point(5, 0), 4, 3))
        [Point2D(2, -3*sqrt(7)/4), Point2D(2, 3*sqrt(7)/4)]
        >>> e.intersection(Ellipse(Point(100500, 0), 4, 3))
        []
        >>> e.intersection(Ellipse(Point(0, 0), 3, 4))
        [Point2D(3, 0), Point2D(-363/175, -48*sqrt(111)/175), Point2D(-363/175, 48*sqrt(111)/175)]
        >>> e.intersection(Ellipse(Point(-1, 0), 3, 4))
        [Point2D(-17/5, -12/5), Point2D(-17/5, 12/5), Point2D(7/5, -12/5), Point2D(7/5, 12/5)]
        """
        # TODO: Replace solve with nonlinsolve, when nonlinsolve will be able to solve in real domain

        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []

        elif isinstance(o, (Segment2D, Ray2D)):
            ellipse_equation = self.equation(x, y)
            result = solve([ellipse_equation, Line(
                o.points[0], o.points[1]).equation(x, y)], [x, y],
                set=True)[1]
            return list(ordered([Point(i) for i in result if i in o]))

        elif isinstance(o, Polygon):
            return o.intersection(self)

        elif isinstance(o, (Ellipse, Line2D)):
            if o == self:
                return self
            else:
                ellipse_equation = self.equation(x, y)
                return list(ordered([Point(i) for i in solve(
                    [ellipse_equation, o.equation(x, y)], [x, y],
                    set=True)[1]]))
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Intersection not handled for %s' % func_name(o))

    def is_tangent(self, o):
        """Is `o` tangent to the ellipse?

        Parameters
        ==========

        o : GeometryEntity
            An Ellipse, LinearEntity or Polygon

        Raises
        ======

        NotImplementedError
            When the wrong type of argument is supplied.

        Returns
        =======

        is_tangent: boolean
            True if o is tangent to the ellipse, False otherwise.

        See Also
        ========

        tangent_lines

        Examples
        ========

        >>> from sympy import Point, Ellipse, Line
        >>> p0, p1, p2 = Point(0, 0), Point(3, 0), Point(3, 3)
        >>> e1 = Ellipse(p0, 3, 2)
        >>> l1 = Line(p1, p2)
        >>> e1.is_tangent(l1)
        True

        """
        if isinstance(o, Point2D):
            return False
        elif isinstance(o, Ellipse):
            intersect = self.intersection(o)
            if isinstance(intersect, Ellipse):
                return True
            elif intersect:
                return all((self.tangent_lines(i)[0]).equals(o.tangent_lines(i)[0]) for i in intersect)
            else:
                return False
        elif isinstance(o, Line2D):
            hit = self.intersection(o)
            if not hit:
                return False
            if len(hit) == 1:
                return True
            # might return None if it can't decide
            return hit[0].equals(hit[1])
        elif isinstance(o, (Segment2D, Ray2D)):
            intersect = self.intersection(o)
            if len(intersect) == 1:
                return o in self.tangent_lines(intersect[0])[0]
            else:
                return False
        elif isinstance(o, Polygon):
            return all(self.is_tangent(s) for s in o.sides)
        elif isinstance(o, (LinearEntity3D, Point3D)):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Is_tangent not handled for %s' % func_name(o))

    @property
    def major(self):
        """Longer axis of the ellipse (if it can be determined) else hradius.

        Returns
        =======

        major : number or expression

        See Also
        ========

        hradius, vradius, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.major
        3

        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).major
        a
        >>> Ellipse(p1, b, a).major
        b

        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).major
        m + 1

        """
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        a, b = ab
        o = b - a < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.hradius

    @property
    def minor(self):
        """Shorter axis of the ellipse (if it can be determined) else vradius.

        Returns
        =======

        minor : number or expression

        See Also
        ========

        hradius, vradius, major

        Examples
        ========

        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.minor
        1

        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).minor
        b
        >>> Ellipse(p1, b, a).minor
        a

        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).minor
        m

        """
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        a, b = ab
        o = a - b < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.vradius

    def normal_lines(self, p, prec=None):
        """Normal lines between `p` and the ellipse.

        Parameters
        ==========

        p : Point

        Returns
        =======

        normal_lines : list with 1, 2 or 4 Lines

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e = Ellipse((0, 0), 2, 3)
        >>> c = e.center
        >>> e.normal_lines(c + Point(1, 0))
        [Line2D(Point2D(0, 0), Point2D(1, 0))]
        >>> e.normal_lines(c)
        [Line2D(Point2D(0, 0), Point2D(0, 1)), Line2D(Point2D(0, 0), Point2D(1, 0))]

        Off-axis points require the solution of a quartic equation. This
        often leads to very large expressions that may be of little practical
        use. An approximate solution of `prec` digits can be obtained by
        passing in the desired value:

        >>> e.normal_lines((3, 3), prec=2)
        [Line2D(Point2D(-0.81, -2.7), Point2D(0.19, -1.2)),
        Line2D(Point2D(1.5, -2.0), Point2D(2.5, -2.7))]

        Whereas the above solution has an operation count of 12, the exact
        solution has an operation count of 2020.
        """
        p = Point(p, dim=2)

        # XXX change True to something like self.angle == 0 if the arbitrarily
        # rotated ellipse is introduced.
        # https://github.com/sympy/sympy/issues/2815)
        if True:
            rv = []
            if p.x == self.center.x:
                rv.append(Line(self.center, slope=oo))
            if p.y == self.center.y:
                rv.append(Line(self.center, slope=0))
            if rv:
                # at these special orientations of p either 1 or 2 normals
                # exist and we are done
                return rv

        # find the 4 normal points and construct lines through them with
        # the corresponding slope
        eq = self.equation(x, y)
        dydx = idiff(eq, y, x)
        norm = -1/dydx
        slope = Line(p, (x, y)).slope
        seq = slope - norm

        # TODO: Replace solve with solveset, when this line is tested
        yis = solve(seq, y)[0]
        xeq = eq.subs(y, yis).as_numer_denom()[0].expand()
        if len(xeq.free_symbols) == 1:
            try:
                # this is so much faster, it's worth a try
                xsol = Poly(xeq, x).real_roots()
            except (DomainError, PolynomialError, NotImplementedError):
                # TODO: Replace solve with solveset, when these lines are tested
                xsol = _nsort(solve(xeq, x), separated=True)[0]
            points = [Point(i, solve(eq.subs(x, i), y)[0]) for i in xsol]
        else:
            raise NotImplementedError(
                'intersections for the general ellipse are not supported')
        slopes = [norm.subs(zip((x, y), pt.args)) for pt in points]
        if prec is not None:
            points = [pt.n(prec) for pt in points]
            slopes = [i if _not_a_coeff(i) else i.n(prec) for i in slopes]
        return [Line(pt, slope=s) for pt, s in zip(points, slopes)]

    @property
    def periapsis(self):
        """The periapsis of the ellipse.

        The shortest distance between the focus and the contour.

        Returns
        =======

        periapsis : number

        See Also
        ========

        apoapsis : Returns greatest distance between focus and contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.periapsis
        3 - 2*sqrt(2)

        """
        return self.major * (1 - self.eccentricity)

    @property
    def semilatus_rectum(self):
        """
        Calculates the semi-latus rectum of the Ellipse.

        Semi-latus rectum is defined as one half of the chord through a
        focus parallel to the conic section directrix of a conic section.

        Returns
        =======

        semilatus_rectum : number

        See Also
        ========

        apoapsis : Returns greatest distance between focus and contour

        periapsis : The shortest distance between the focus and the contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.semilatus_rectum
        1/3

        References
        ==========

        .. [1] https://mathworld.wolfram.com/SemilatusRectum.html
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum

        """
        return self.major * (1 - self.eccentricity ** 2)

    def auxiliary_circle(self):
        """Returns a Circle whose diameter is the major axis of the ellipse.

        Examples
        ========

        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(1, 2)
        >>> Ellipse(c, 8, 7).auxiliary_circle()
        Circle(Point2D(1, 2), 8)
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).auxiliary_circle()
        Circle(Point2D(1, 2), Max(a, b))
        """
        return Circle(self.center, Max(self.hradius, self.vradius))

    def director_circle(self):
        """
        Returns a Circle consisting of all points where two perpendicular
        tangent lines to the ellipse cross each other.

        Returns
        =======

        Circle
            A director circle returned as a geometric object.

        Examples
        ========

        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(3,8)
        >>> Ellipse(c, 7, 9).director_circle()
        Circle(Point2D(3, 8), sqrt(130))
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).director_circle()
        Circle(Point2D(3, 8), sqrt(a**2 + b**2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Director_circle

        """
        return Circle(self.center, sqrt(self.hradius**2 + self.vradius**2))

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the Ellipse.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.plot_interval()
        [t, -pi, pi]

        """
        t = _symbol(parameter, real=True)
        return [t, -S.Pi, S.Pi]

    def random_point(self, seed=None):
        """A random point on the ellipse.

        Returns
        =======

        point : Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.random_point() # gives some random point
        Point2D(...)
        >>> p1 = e1.random_point(seed=0); p1.n(2)
        Point2D(2.1, 1.4)

        Notes
        =====

        When creating a random point, one may simply replace the
        parameter with a random number. When doing so, however, the
        random number should be made a Rational or else the point
        may not test as being in the ellipse:

        >>> from sympy.abc import t
        >>> from sympy import Rational
        >>> arb = e1.arbitrary_point(t); arb
        Point2D(3*cos(t), 2*sin(t))
        >>> arb.subs(t, .1) in e1
        False
        >>> arb.subs(t, Rational(.1)) in e1
        True
        >>> arb.subs(t, Rational('.1')) in e1
        True

        See Also
        ========
        sympy.geometry.point.Point
        arbitrary_point : Returns parameterized point on ellipse
        """
        t = _symbol('t', real=True)
        x, y = self.arbitrary_point(t).args
        # get a random value in [-1, 1) corresponding to cos(t)
        # and confirm that it will test as being in the ellipse
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        # simplify this now or else the Float will turn s into a Float
        r = Rational(rng.random())
        c = 2*r - 1
        s = sqrt(1 - c**2)
        return Point(x.subs(cos(t), c), y.subs(sin(t), s))

    def reflect(self, line):
        """Override GeometryEntity.reflect since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle, Line
        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))
        Circle(Point2D(1, 0), -1)
        >>> from sympy import Ellipse, Line, Point
        >>> Ellipse(Point(3, 4), 1, 3).reflect(Line(Point(0, -4), Point(5, 0)))
        Traceback (most recent call last):
        ...
        NotImplementedError:
        General Ellipse is not supported but the equation of the reflected
        Ellipse is given by the zeros of: f(x, y) = (9*x/41 + 40*y/41 +
        37/41)**2 + (40*x/123 - 3*y/41 - 364/123)**2 - 1

        Notes
        =====

        Until the general ellipse (with no axis parallel to the x-axis) is
        supported a NotImplemented error is raised and the equation whose
        zeros define the rotated ellipse is given.

        """

        if line.slope in (0, oo):
            c = self.center
            c = c.reflect(line)
            return self.func(c, -self.hradius, self.vradius)
        else:
            x, y = [uniquely_named_symbol(
                name, (self, line), modify=lambda s: '_' + s, real=True)
                for name in 'xy']
            expr = self.equation(x, y)
            p = Point(x, y).reflect(line)
            result = expr.subs(zip((x, y), p.args
                                   ), simultaneous=True)
            raise NotImplementedError(filldedent(
                'General Ellipse is not supported but the equation '
                'of the reflected Ellipse is given by the zeros of: ' +
                "f(%s, %s) = %s" % (str(x), str(y), str(result))))

    def rotate(self, angle=0, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        Note: since the general ellipse is not supported, only rotations that
        are integer multiples of pi/2 are allowed.

        Examples
        ========

        >>> from sympy import Ellipse, pi
        >>> Ellipse((1, 0), 2, 1).rotate(pi/2)
        Ellipse(Point2D(0, 1), 1, 2)
        >>> Ellipse((1, 0), 2, 1).rotate(pi)
        Ellipse(Point2D(-1, 0), 2, 1)
        """
        if self.hradius == self.vradius:
            return self.func(self.center.rotate(angle, pt), self.hradius)
        if (angle/S.Pi).is_integer:
            return super().rotate(angle, pt)
        if (2*angle/S.Pi).is_integer:
            return self.func(self.center.rotate(angle, pt), self.vradius, self.hradius)
        # XXX see https://github.com/sympy/sympy/issues/2815 for general ellipes
        raise NotImplementedError('Only rotations of pi/2 are currently supported for Ellipse.')

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since it is the major and minor
        axes which must be scaled and they are not GeometryEntities.

        Examples
        ========

        >>> from sympy import Ellipse
        >>> Ellipse((0, 0), 2, 1).scale(2, 4)
        Circle(Point2D(0, 0), 4)
        >>> Ellipse((0, 0), 2, 1).scale(2)
        Ellipse(Point2D(0, 0), 4, 1)
        """
        c = self.center
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        h = self.hradius
        v = self.vradius
        return self.func(c.scale(x, y), hradius=h*x, vradius=v*y)

    def tangent_lines(self, p):
        """Tangent lines between `p` and the ellipse.

        If `p` is on the ellipse, returns the tangent line through point `p`.
        Otherwise, returns the tangent line(s) from `p` to the ellipse, or
        None if no tangent line is possible (e.g., `p` inside ellipse).

        Parameters
        ==========

        p : Point

        Returns
        =======

        tangent_lines : list with 1 or 2 Lines

        Raises
        ======

        NotImplementedError
            Can only find tangent lines for a point, `p`, on the ellipse.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.tangent_lines(Point(3, 0))
        [Line2D(Point2D(3, 0), Point2D(3, -12))]

        """
        p = Point(p, dim=2)
        if self.encloses_point(p):
            return []

        if p in self:
            delta = self.center - p
            rise = (self.vradius**2)*delta.x
            run = -(self.hradius**2)*delta.y
            p2 = Point(simplify(p.x + run),
                       simplify(p.y + rise))
            return [Line(p, p2)]
        else:
            if len(self.foci) == 2:
                f1, f2 = self.foci
                maj = self.hradius
                test = (2*maj -
                        Point.distance(f1, p) -
                        Point.distance(f2, p))
            else:
                test = self.radius - Point.distance(self.center, p)
            if test.is_number and test.is_positive:
                return []
            # else p is outside the ellipse or we can't tell. In case of the
            # latter, the solutions returned will only be valid if
            # the point is not inside the ellipse; if it is, nan will result.
            eq = self.equation(x, y)
            dydx = idiff(eq, y, x)
            slope = Line(p, Point(x, y)).slope

            # TODO: Replace solve with solveset, when this line is tested
            tangent_points = solve([slope - dydx, eq], [x, y])

            # handle horizontal and vertical tangent lines
            if len(tangent_points) == 1:
                if tangent_points[0][
                           0] == p.x or tangent_points[0][1] == p.y:
                    return [Line(p, p + Point(1, 0)), Line(p, p + Point(0, 1))]
                else:
                    return [Line(p, p + Point(0, 1)), Line(p, tangent_points[0])]

            # others
            return [Line(p, tangent_points[0]), Line(p, tangent_points[1])]

    @property
    def vradius(self):
        """The vertical radius of the ellipse.

        Returns
        =======

        vradius : number

        See Also
        ========

        hradius, major, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.vradius
        1

        """
        return self.args[2]


    def second_moment_of_area(self, point=None):
        """Returns the second moment and product moment area of an ellipse.

        Parameters
        ==========

        point : Point, two-tuple of sympifiable objects, or None(default=None)
            point is the point about which second moment of area is to be found.
            If "point=None" it will be calculated about the axis passing through the
            centroid of the ellipse.

        Returns
        =======

        I_xx, I_yy, I_xy : number or SymPy expression
            I_xx, I_yy are second moment of area of an ellise.
            I_xy is product moment of area of an ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.second_moment_of_area()
        (3*pi/4, 27*pi/4, 0)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/List_of_second_moments_of_area

        """

        I_xx = (S.Pi*(self.hradius)*(self.vradius**3))/4
        I_yy = (S.Pi*(self.hradius**3)*(self.vradius))/4
        I_xy = 0

        if point is None:
            return I_xx, I_yy, I_xy

        # parallel axis theorem
        I_xx = I_xx + self.area*((point[1] - self.center.y)**2)
        I_yy = I_yy + self.area*((point[0] - self.center.x)**2)
        I_xy = I_xy + self.area*(point[0] - self.center.x)*(point[1] - self.center.y)

        return I_xx, I_yy, I_xy


    def polar_second_moment_of_area(self):
        """Returns the polar second moment of area of an Ellipse

        It is a constituent of the second moment of area, linked through
        the perpendicular axis theorem. While the planar second moment of
        area describes an object's resistance to deflection (bending) when
        subjected to a force applied to a plane parallel to the central
        axis, the polar second moment of area describes an object's
        resistance to deflection when subjected to a moment applied in a
        plane perpendicular to the object's central axis (i.e. parallel to
        the cross-section)

        Examples
        ========

        >>> from sympy import symbols, Circle, Ellipse
        >>> c = Circle((5, 5), 4)
        >>> c.polar_second_moment_of_area()
        128*pi
        >>> a, b = symbols('a, b')
        >>> e = Ellipse((0, 0), a, b)
        >>> e.polar_second_moment_of_area()
        pi*a**3*b/4 + pi*a*b**3/4

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia

        """
        second_moment = self.second_moment_of_area()
        return second_moment[0] + second_moment[1]


    def section_modulus(self, point=None):
        """Returns a tuple with the section modulus of an ellipse

        Section modulus is a geometric property of an ellipse defined as the
        ratio of second moment of area to the distance of the extreme end of
        the ellipse from the centroidal axis.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point at which section modulus is to be found.
            If "point=None" section modulus will be calculated for the
            point farthest from the centroidal axis of the ellipse.

        Returns
        =======

        S_x, S_y: numbers or SymPy expressions
                  S_x is the section modulus with respect to the x-axis
                  S_y is the section modulus with respect to the y-axis
                  A negative sign indicates that the section modulus is
                  determined for a point below the centroidal axis.

        Examples
        ========

        >>> from sympy import Symbol, Ellipse, Circle, Point2D
        >>> d = Symbol('d', positive=True)
        >>> c = Circle((0, 0), d/2)
        >>> c.section_modulus()
        (pi*d**3/32, pi*d**3/32)
        >>> e = Ellipse(Point2D(0, 0), 2, 4)
        >>> e.section_modulus()
        (8*pi, 4*pi)
        >>> e.section_modulus((2, 2))
        (16*pi, 4*pi)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Section_modulus

        """
        x_c, y_c = self.center
        if point is None:
            # taking x and y as maximum distances from centroid
            x_min, y_min, x_max, y_max = self.bounds
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            # taking x and y as distances of the given point from the center
            point = Point2D(point)
            y = point.y - y_c
            x = point.x - x_c

        second_moment = self.second_moment_of_area()
        S_x = second_moment[0]/y
        S_y = second_moment[1]/x

        return S_x, S_y


class Circle(Ellipse):
    r"""A circle in space.

    Constructed simply from a center and a radius, from three
    non-collinear points, or the equation of a circle.

    Parameters
    ==========

    center : Point
    radius : number or SymPy expression
    points : sequence of three Points
    equation : equation of a circle

    Attributes
    ==========

    radius (synonymous with hradius, vradius, major and minor)
    circumference
    equation

    Raises
    ======

    GeometryError
        When the given equation is not that of a circle.
        When trying to construct circle from incorrect parameters.

    See Also
    ========

    Ellipse, sympy.geometry.point.Point

    Examples
    ========

    >>> from sympy import Point, Circle, Eq
    >>> from sympy.abc import x, y, a, b

    A circle constructed from a center and radius:

    >>> c1 = Circle(Point(0, 0), 5)
    >>> c1.hradius, c1.vradius, c1.radius
    (5, 5, 5)

    A circle constructed from three points:

    >>> c2 = Circle(Point(0, 0), Point(1, 1), Point(1, 0))
    >>> c2.hradius, c2.vradius, c2.radius, c2.center
    (sqrt(2)/2, sqrt(2)/2, sqrt(2)/2, Point2D(1/2, 1/2))

    A circle can be constructed from an equation in the form
    `ax^2 + by^2 + gx + hy + c = 0`, too:

    >>> Circle(x**2 + y**2 - 25)
    Circle(Point2D(0, 0), 5)

    If the variables corresponding to x and y are named something
    else, their name or symbol can be supplied:

    >>> Circle(Eq(a**2 + b**2, 25), x='a', y=b)
    Circle(Point2D(0, 0), 5)
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
            x = kwargs.get('x', 'x')
            y = kwargs.get('y', 'y')
            equation = args[0].expand()
            if isinstance(equation, Eq):
                equation = equation.lhs - equation.rhs
            x = find(x, equation)
            y = find(y, equation)

            try:
                a, b, c, d, e = linear_coeffs(equation, x**2, y**2, x, y)
            except ValueError:
                raise GeometryError("The given equation is not that of a circle.")

            if S.Zero in (a, b) or a != b:
                raise GeometryError("The given equation is not that of a circle.")

            center_x = -c/a/2
            center_y = -d/b/2
            r2 = (center_x**2) + (center_y**2) - e/a

            return Circle((center_x, center_y), sqrt(r2), evaluate=evaluate)

        else:
            c, r = None, None
            if len(args) == 3:
                args = [Point(a, dim=2, evaluate=evaluate) for a in args]
                t = Triangle(*args)
                if not isinstance(t, Triangle):
                    return t
                c = t.circumcenter
                r = t.circumradius
            elif len(args) == 2:
                # Assume (center, radius) pair
                c = Point(args[0], dim=2, evaluate=evaluate)
                r = args[1]
                # this will prohibit imaginary radius
                try:
                    r = Point(r, 0, evaluate=evaluate).x
                except ValueError:
                    raise GeometryError("Circle with imaginary radius is not permitted")

            if not (c is None or r is None):
                if r == 0:
                    return c
                return GeometryEntity.__new__(cls, c, r, **kwargs)

            raise GeometryError("Circle.__new__ received unknown arguments")

    def _eval_evalf(self, prec=15, **options):
        pt, r = self.args
        dps = prec_to_dps(prec)
        pt = pt.evalf(n=dps, **options)
        r = r.evalf(n=dps, **options)
        return self.func(pt, r, evaluate=False)

    @property
    def circumference(self):
        """The circumference of the circle.

        Returns
        =======

        circumference : number or SymPy expression

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.circumference
        12*pi

        """
        return 2 * S.Pi * self.radius

    def equation(self, x='x', y='y'):
        """The equation of the circle.

        Parameters
        ==========

        x : str or Symbol, optional
            Default value is 'x'.
        y : str or Symbol, optional
            Default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(0, 0), 5)
        >>> c1.equation()
        x**2 + y**2 - 25

        """
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        t1 = (x - self.center.x)**2
        t2 = (y - self.center.y)**2
        return t1 + t2 - self.major**2

    def intersection(self, o):
        """The intersection of this circle with another geometrical entity.

        Parameters
        ==========

        o : GeometryEntity

        Returns
        =======

        intersection : list of GeometryEntities

        Examples
        ========

        >>> from sympy import Point, Circle, Line, Ray
        >>> p1, p2, p3 = Point(0, 0), Point(5, 5), Point(6, 0)
        >>> p4 = Point(5, 0)
        >>> c1 = Circle(p1, 5)
        >>> c1.intersection(p2)
        []
        >>> c1.intersection(p4)
        [Point2D(5, 0)]
        >>> c1.intersection(Ray(p1, p2))
        [Point2D(5*sqrt(2)/2, 5*sqrt(2)/2)]
        >>> c1.intersection(Line(p2, p3))
        []

        """
        return Ellipse.intersection(self, o)

    @property
    def radius(self):
        """The radius of the circle.

        Returns
        =======

        radius : number or SymPy expression

        See Also
        ========

        Ellipse.major, Ellipse.minor, Ellipse.hradius, Ellipse.vradius

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.radius
        6

        """
        return self.args[1]

    def reflect(self, line):
        """Override GeometryEntity.reflect since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle, Line
        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))
        Circle(Point2D(1, 0), -1)
        """
        c = self.center
        c = c.reflect(line)
        return self.func(c, -self.radius)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle
        >>> Circle((0, 0), 1).scale(2, 2)
        Circle(Point2D(0, 0), 2)
        >>> Circle((0, 0), 1).scale(2, 4)
        Ellipse(Point2D(0, 0), 2, 4)
        """
        c = self.center
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        c = c.scale(x, y)
        x, y = [abs(i) for i in (x, y)]
        if x == y:
            return self.func(c, x*self.radius)
        h = v = self.radius
        return Ellipse(c, hradius=h*x, vradius=v*y)

    @property
    def vradius(self):
        """
        This Ellipse property is an alias for the Circle's radius.

        Whereas hradius, major and minor can use Ellipse's conventions,
        the vradius does not exist for a circle. It is always a positive
        value in order that the Circle, like Polygons, will have an
        area that can be positive or negative as determined by the sign
        of the hradius.

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.vradius
        6
        """
        return abs(self.radius)


from .polygon import Polygon, Triangle
<|endoftext|>"""Geometrical Planes.

Contains
========
Plane

"""

from sympy.core import Dummy, Rational, S, Symbol
from sympy.core.symbol import _symbol
from sympy.functions.elementary.trigonometric import cos, sin, acos, asin, sqrt
from .entity import GeometryEntity
from .line import (Line, Ray, Segment, Line3D, LinearEntity, LinearEntity3D,
                   Ray3D, Segment3D)
from .point import Point, Point3D
from sympy.matrices import Matrix
from sympy.polys.polytools import cancel
from sympy.solvers import solve, linsolve
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable

from mpmath.libmp.libmpf import prec_to_dps

import random


x, y, z, t = [Dummy('plane_dummy') for i in range(4)]


class Plane(GeometryEntity):
    """
    A plane is a flat, two-dimensional surface. A plane is the two-dimensional
    analogue of a point (zero-dimensions), a line (one-dimension) and a solid
    (three-dimensions). A plane can generally be constructed by two types of
    inputs. They are:
    - three non-collinear points
    - a point and the plane's normal vector

    Attributes
    ==========

    p1
    normal_vector

    Examples
    ========

    >>> from sympy import Plane, Point3D
    >>> Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
    Plane(Point3D(1, 1, 1), (-1, 2, -1))
    >>> Plane((1, 1, 1), (2, 3, 4), (2, 2, 2))
    Plane(Point3D(1, 1, 1), (-1, 2, -1))
    >>> Plane(Point3D(1, 1, 1), normal_vector=(1,4,7))
    Plane(Point3D(1, 1, 1), (1, 4, 7))

    """
    def __new__(cls, p1, a=None, b=None, **kwargs):
        p1 = Point3D(p1, dim=3)
        if a and b:
            p2 = Point(a, dim=3)
            p3 = Point(b, dim=3)
            if Point3D.are_collinear(p1, p2, p3):
                raise ValueError('Enter three non-collinear points')
            a = p1.direction_ratio(p2)
            b = p1.direction_ratio(p3)
            normal_vector = tuple(Matrix(a).cross(Matrix(b)))
        else:
            a = kwargs.pop('normal_vector', a)
            evaluate = kwargs.get('evaluate', True)
            if is_sequence(a) and len(a) == 3:
                normal_vector = Point3D(a).args if evaluate else a
            else:
                raise ValueError(filldedent('''
                    Either provide 3 3D points or a point with a
                    normal vector expressed as a sequence of length 3'''))
            if all(coord.is_zero for coord in normal_vector):
                raise ValueError('Normal vector cannot be zero vector')
        return GeometryEntity.__new__(cls, p1, normal_vector, **kwargs)

    def __contains__(self, o):
        k = self.equation(x, y, z)
        if isinstance(o, (LinearEntity, LinearEntity3D)):
            d = Point3D(o.arbitrary_point(t))
            e = k.subs([(x, d.x), (y, d.y), (z, d.z)])
            return e.equals(0)
        try:
            o = Point(o, dim=3, strict=True)
            d = k.xreplace(dict(zip((x, y, z), o.args)))
            return d.equals(0)
        except TypeError:
            return False

    def _eval_evalf(self, prec=15, **options):
        pt, tup = self.args
        dps = prec_to_dps(prec)
        pt = pt.evalf(n=dps, **options)
        tup = tuple([i.evalf(n=dps, **options) for i in tup])
        return self.func(pt, normal_vector=tup, evaluate=False)

    def angle_between(self, o):
        """Angle between the plane and other geometric entity.

        Parameters
        ==========

        LinearEntity3D, Plane.

        Returns
        =======

        angle : angle in radians

        Notes
        =====

        This method accepts only 3D entities as it's parameter, but if you want
        to calculate the angle between a 2D entity and a plane you should
        first convert to a 3D entity by projecting onto a desired plane and
        then proceed to calculate the angle.

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 2, 2), normal_vector=(1, 2, 3))
        >>> b = Line3D(Point3D(1, 3, 4), Point3D(2, 2, 2))
        >>> a.angle_between(b)
        -asin(sqrt(21)/6)

        """
        if isinstance(o, LinearEntity3D):
            a = Matrix(self.normal_vector)
            b = Matrix(o.direction_ratio)
            c = a.dot(b)
            d = sqrt(sum(i**2 for i in self.normal_vector))
            e = sqrt(sum(i**2 for i in o.direction_ratio))
            return asin(c/(d*e))
        if isinstance(o, Plane):
            a = Matrix(self.normal_vector)
            b = Matrix(o.normal_vector)
            c = a.dot(b)
            d = sqrt(sum(i**2 for i in self.normal_vector))
            e = sqrt(sum(i**2 for i in o.normal_vector))
            return acos(c/(d*e))


    def arbitrary_point(self, u=None, v=None):
        """ Returns an arbitrary point on the Plane. If given two
        parameters, the point ranges over the entire plane. If given 1
        or no parameters, returns a point with one parameter which,
        when varying from 0 to 2*pi, moves the point in a circle of
        radius 1 about p1 of the Plane.

        Examples
        ========

        >>> from sympy import Plane, Ray
        >>> from sympy.abc import u, v, t, r
        >>> p = Plane((1, 1, 1), normal_vector=(1, 0, 0))
        >>> p.arbitrary_point(u, v)
        Point3D(1, u + 1, v + 1)
        >>> p.arbitrary_point(t)
        Point3D(1, cos(t) + 1, sin(t) + 1)

        While arbitrary values of u and v can move the point anywhere in
        the plane, the single-parameter point can be used to construct a
        ray whose arbitrary point can be located at angle t and radius
        r from p.p1:

        >>> Ray(p.p1, _).arbitrary_point(r)
        Point3D(1, r*cos(t) + 1, r*sin(t) + 1)

        Returns
        =======

        Point3D

        """
        circle = v is None
        if circle:
            u = _symbol(u or 't', real=True)
        else:
            u = _symbol(u or 'u', real=True)
            v = _symbol(v or 'v', real=True)
        x, y, z = self.normal_vector
        a, b, c = self.p1.args
        # x1, y1, z1 is a nonzero vector parallel to the plane
        if x.is_zero and y.is_zero:
            x1, y1, z1 = S.One, S.Zero, S.Zero
        else:
            x1, y1, z1 = -y, x, S.Zero
        # x2, y2, z2 is also parallel to the plane, and orthogonal to x1, y1, z1
        x2, y2, z2 = tuple(Matrix((x, y, z)).cross(Matrix((x1, y1, z1))))
        if circle:
            x1, y1, z1 = (w/sqrt(x1**2 + y1**2 + z1**2) for w in (x1, y1, z1))
            x2, y2, z2 = (w/sqrt(x2**2 + y2**2 + z2**2) for w in (x2, y2, z2))
            p = Point3D(a + x1*cos(u) + x2*sin(u), \
                        b + y1*cos(u) + y2*sin(u), \
                        c + z1*cos(u) + z2*sin(u))
        else:
            p = Point3D(a + x1*u + x2*v, b + y1*u + y2*v, c + z1*u + z2*v)
        return p


    @staticmethod
    def are_concurrent(*planes):
        """Is a sequence of Planes concurrent?

        Two or more Planes are concurrent if their intersections
        are a common line.

        Parameters
        ==========

        planes: list

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(5, 0, 0), normal_vector=(1, -1, 1))
        >>> b = Plane(Point3D(0, -2, 0), normal_vector=(3, 1, 1))
        >>> c = Plane(Point3D(0, -1, 0), normal_vector=(5, -1, 9))
        >>> Plane.are_concurrent(a, b)
        True
        >>> Plane.are_concurrent(a, b, c)
        False

        """
        planes = list(uniq(planes))
        for i in planes:
            if not isinstance(i, Plane):
                raise ValueError('All objects should be Planes but got %s' % i.func)
        if len(planes) < 2:
            return False
        planes = list(planes)
        first = planes.pop(0)
        sol = first.intersection(planes[0])
        if sol == []:
            return False
        else:
            line = sol[0]
            for i in planes[1:]:
                l = first.intersection(i)
                if not l or l[0] not in line:
                    return False
            return True


    def distance(self, o):
        """Distance between the plane and another geometric entity.

        Parameters
        ==========

        Point3D, LinearEntity3D, Plane.

        Returns
        =======

        distance

        Notes
        =====

        This method accepts only 3D entities as it's parameter, but if you want
        to calculate the distance between a 2D entity and a plane you should
        first convert to a 3D entity by projecting onto a desired plane and
        then proceed to calculate the distance.

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))
        >>> b = Point3D(1, 2, 3)
        >>> a.distance(b)
        sqrt(3)
        >>> c = Line3D(Point3D(2, 3, 1), Point3D(1, 2, 2))
        >>> a.distance(c)
        0

        """
        if self.intersection(o) != []:
            return S.Zero

        if isinstance(o, (Segment3D, Ray3D)):
            a, b = o.p1, o.p2
            pi, = self.intersection(Line3D(a, b))
            if pi in o:
                return self.distance(pi)
            elif a in Segment3D(pi, b):
                return self.distance(a)
            else:
                assert isinstance(o, Segment3D) is True
                return self.distance(b)

        # following code handles `Point3D`, `LinearEntity3D`, `Plane`
        a = o if isinstance(o, Point3D) else o.p1
        n = Point3D(self.normal_vector).unit
        d = (a - self.p1).dot(n)
        return abs(d)


    def equals(self, o):
        """
        Returns True if self and o are the same mathematical entities.

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))
        >>> b = Plane(Point3D(1, 2, 3), normal_vector=(2, 2, 2))
        >>> c = Plane(Point3D(1, 2, 3), normal_vector=(-1, 4, 6))
        >>> a.equals(a)
        True
        >>> a.equals(b)
        True
        >>> a.equals(c)
        False
        """
        if isinstance(o, Plane):
            a = self.equation()
            b = o.equation()
            return cancel(a/b).is_constant()
        else:
            return False


    def equation(self, x=None, y=None, z=None):
        """The equation of the Plane.

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 2), Point3D(2, 4, 7), Point3D(3, 5, 1))
        >>> a.equation()
        -23*x + 11*y - 2*z + 16
        >>> a = Plane(Point3D(1, 4, 2), normal_vector=(6, 6, 6))
        >>> a.equation()
        6*x + 6*y + 6*z - 42

        """
        x, y, z = [i if i else Symbol(j, real=True) for i, j in zip((x, y, z), 'xyz')]
        a = Point3D(x, y, z)
        b = self.p1.direction_ratio(a)
        c = self.normal_vector
        return (sum(i*j for i, j in zip(b, c)))


    def intersection(self, o):
        """ The intersection with other geometrical entity.

        Parameters
        ==========

        Point, Point3D, LinearEntity, LinearEntity3D, Plane

        Returns
        =======

        List

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))
        >>> b = Point3D(1, 2, 3)
        >>> a.intersection(b)
        [Point3D(1, 2, 3)]
        >>> c = Line3D(Point3D(1, 4, 7), Point3D(2, 2, 2))
        >>> a.intersection(c)
        [Point3D(2, 2, 2)]
        >>> d = Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))
        >>> e = Plane(Point3D(2, 0, 0), normal_vector=(3, 4, -3))
        >>> d.intersection(e)
        [Line3D(Point3D(78/23, -24/23, 0), Point3D(147/23, 321/23, 23))]

        """
        if not isinstance(o, GeometryEntity):
            o = Point(o, dim=3)
        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []
        if isinstance(o, (LinearEntity, LinearEntity3D)):
            # recast to 3D
            p1, p2 = o.p1, o.p2
            if isinstance(o, Segment):
                o = Segment3D(p1, p2)
            elif isinstance(o, Ray):
                o = Ray3D(p1, p2)
            elif isinstance(o, Line):
                o = Line3D(p1, p2)
            else:
                raise ValueError('unhandled linear entity: %s' % o.func)
            if o in self:
                return [o]
            else:
                a = Point3D(o.arbitrary_point(t))
                p1, n = self.p1, Point3D(self.normal_vector)

                # TODO: Replace solve with solveset, when this line is tested
                c = solve((a - p1).dot(n), t)
                if not c:
                    return []
                else:
                    c = [i for i in c if i.is_real is not False]
                    if len(c) > 1:
                        c = [i for i in c if i.is_real]
                    if len(c) != 1:
                        raise Undecidable("not sure which point is real")
                    p = a.subs(t, c[0])
                    if p not in o:
                        return []  # e.g. a segment might not intersect a plane
                    return [p]
        if isinstance(o, Plane):
            if self.equals(o):
                return [self]
            if self.is_parallel(o):
                return []
            else:
                x, y, z = map(Dummy, 'xyz')
                a, b = Matrix([self.normal_vector]), Matrix([o.normal_vector])
                c = list(a.cross(b))
                d = self.equation(x, y, z)
                e = o.equation(x, y, z)
                result = list(linsolve([d, e], x, y, z))[0]
                for i in (x, y, z): result = result.subs(i, 0)
                return [Line3D(Point3D(result), direction_ratio=c)]


    def is_coplanar(self, o):
        """ Returns True if `o` is coplanar with self, else False.

        Examples
        ========

        >>> from sympy import Plane
        >>> o = (0, 0, 0)
        >>> p = Plane(o, (1, 1, 1))
        >>> p2 = Plane(o, (2, 2, 2))
        >>> p == p2
        False
        >>> p.is_coplanar(p2)
        True
        """
        if isinstance(o, Plane):
            return not cancel(self.equation(x, y, z)/o.equation(x, y, z)).has(x, y, z)
        if isinstance(o, Point3D):
            return o in self
        elif isinstance(o, LinearEntity3D):
            return all(i in self for i in self)
        elif isinstance(o, GeometryEntity):  # XXX should only be handling 2D objects now
            return all(i == 0 for i in self.normal_vector[:2])


    def is_parallel(self, l):
        """Is the given geometric entity parallel to the plane?

        Parameters
        ==========

        LinearEntity3D or Plane

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> b = Plane(Point3D(3,1,3), normal_vector=(4, 8, 12))
        >>> a.is_parallel(b)
        True

        """
        if isinstance(l, LinearEntity3D):
            a = l.direction_ratio
            b = self.normal_vector
            return sum(i*j for i, j in zip(a, b)) == 0
        if isinstance(l, Plane):
            a = Matrix(l.normal_vector)
            b = Matrix(self.normal_vector)
            return bool(a.cross(b).is_zero_matrix)


    def is_perpendicular(self, l):
        """Is the given geometric entity perpendicualar to the given plane?

        Parameters
        ==========

        LinearEntity3D or Plane

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> b = Plane(Point3D(2, 2, 2), normal_vector=(-1, 2, -1))
        >>> a.is_perpendicular(b)
        True

        """
        if isinstance(l, LinearEntity3D):
            a = Matrix(l.direction_ratio)
            b = Matrix(self.normal_vector)
            if a.cross(b).is_zero_matrix:
                return True
            else:
                return False
        elif isinstance(l, Plane):
            a = Matrix(l.normal_vector)
            b = Matrix(self.normal_vector)
            if a.dot(b) == 0:
                return True
            else:
                return False
        else:
            return False

    @property
    def normal_vector(self):
        """Normal vector of the given plane.

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
        >>> a.normal_vector
        (-1, 2, -1)
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 4, 7))
        >>> a.normal_vector
        (1, 4, 7)

        """
        return self.args[1]

    @property
    def p1(self):
        """The only defining point of the plane. Others can be obtained from the
        arbitrary_point method.

        See Also
        ========

        sympy.geometry.point.Point3D

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
        >>> a.p1
        Point3D(1, 1, 1)

        """
        return self.args[0]

    def parallel_plane(self, pt):
        """
        Plane parallel to the given plane and passing through the point pt.

        Parameters
        ==========

        pt: Point3D

        Returns
        =======

        Plane

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1, 4, 6), normal_vector=(2, 4, 6))
        >>> a.parallel_plane(Point3D(2, 3, 5))
        Plane(Point3D(2, 3, 5), (2, 4, 6))

        """
        a = self.normal_vector
        return Plane(pt, normal_vector=a)

    def perpendicular_line(self, pt):
        """A line perpendicular to the given plane.

        Parameters
        ==========

        pt: Point3D

        Returns
        =======

        Line3D

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> a.perpendicular_line(Point3D(9, 8, 7))
        Line3D(Point3D(9, 8, 7), Point3D(11, 12, 13))

        """
        a = self.normal_vector
        return Line3D(pt, direction_ratio=a)

    def perpendicular_plane(self, *pts):
        """
        Return a perpendicular passing through the given points. If the
        direction ratio between the points is the same as the Plane's normal
        vector then, to select from the infinite number of possible planes,
        a third point will be chosen on the z-axis (or the y-axis
        if the normal vector is already parallel to the z-axis). If less than
        two points are given they will be supplied as follows: if no point is
        given then pt1 will be self.p1; if a second point is not given it will
        be a point through pt1 on a line parallel to the z-axis (if the normal
        is not already the z-axis, otherwise on the line parallel to the
        y-axis).

        Parameters
        ==========

        pts: 0, 1 or 2 Point3D

        Returns
        =======

        Plane

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a, b = Point3D(0, 0, 0), Point3D(0, 1, 0)
        >>> Z = (0, 0, 1)
        >>> p = Plane(a, normal_vector=Z)
        >>> p.perpendicular_plane(a, b)
        Plane(Point3D(0, 0, 0), (1, 0, 0))
        """
        if len(pts) > 2:
            raise ValueError('No more than 2 pts should be provided.')

        pts = list(pts)
        if len(pts) == 0:
            pts.append(self.p1)
        if len(pts) == 1:
            x, y, z = self.normal_vector
            if x == y == 0:
                dir = (0, 1, 0)
            else:
                dir = (0, 0, 1)
            pts.append(pts[0] + Point3D(*dir))

        p1, p2 = [Point(i, dim=3) for i in pts]
        l = Line3D(p1, p2)
        n = Line3D(p1, direction_ratio=self.normal_vector)
        if l in n:  # XXX should an error be raised instead?
            # there are infinitely many perpendicular planes;
            x, y, z = self.normal_vector
            if x == y == 0:
                # the z axis is the normal so pick a pt on the y-axis
                p3 = Point3D(0, 1, 0)  # case 1
            else:
                # else pick a pt on the z axis
                p3 = Point3D(0, 0, 1)  # case 2
            # in case that point is already given, move it a bit
            if p3 in l:
                p3 *= 2  # case 3
        else:
            p3 = p1 + Point3D(*self.normal_vector)  # case 4
        return Plane(p1, p2, p3)

    def projection_line(self, line):
        """Project the given line onto the plane through the normal plane
        containing the line.

        Parameters
        ==========

        LinearEntity or LinearEntity3D

        Returns
        =======

        Point3D, Line3D, Ray3D or Segment3D

        Notes
        =====

        For the interaction between 2D and 3D lines(segments, rays), you should
        convert the line to 3D by using this method. For example for finding the
        intersection between a 2D and a 3D line, convert the 2D line to a 3D line
        by projecting it on a required plane and then proceed to find the
        intersection between those lines.

        Examples
        ========

        >>> from sympy import Plane, Line, Line3D, Point3D
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))
        >>> b = Line(Point3D(1, 1), Point3D(2, 2))
        >>> a.projection_line(b)
        Line3D(Point3D(4/3, 4/3, 1/3), Point3D(5/3, 5/3, -1/3))
        >>> c = Line3D(Point3D(1, 1, 1), Point3D(2, 2, 2))
        >>> a.projection_line(c)
        Point3D(1, 1, 1)

        """
        if not isinstance(line, (LinearEntity, LinearEntity3D)):
            raise NotImplementedError('Enter a linear entity only')
        a, b = self.projection(line.p1), self.projection(line.p2)
        if a == b:
            # projection does not imply intersection so for
            # this case (line parallel to plane's normal) we
            # return the projection point
            return a
        if isinstance(line, (Line, Line3D)):
            return Line3D(a, b)
        if isinstance(line, (Ray, Ray3D)):
            return Ray3D(a, b)
        if isinstance(line, (Segment, Segment3D)):
            return Segment3D(a, b)

    def projection(self, pt):
        """Project the given point onto the plane along the plane normal.

        Parameters
        ==========

        Point or Point3D

        Returns
        =======

        Point3D

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> A = Plane(Point3D(1, 1, 2), normal_vector=(1, 1, 1))

        The projection is along the normal vector direction, not the z
        axis, so (1, 1) does not project to (1, 1, 2) on the plane A:

        >>> b = Point3D(1, 1)
        >>> A.projection(b)
        Point3D(5/3, 5/3, 2/3)
        >>> _ in A
        True

        But the point (1, 1, 2) projects to (1, 1) on the XY-plane:

        >>> XY = Plane((0, 0, 0), (0, 0, 1))
        >>> XY.projection((1, 1, 2))
        Point3D(1, 1, 0)
        """
        rv = Point(pt, dim=3)
        if rv in self:
            return rv
        return self.intersection(Line3D(rv, rv + Point3D(self.normal_vector)))[0]

    def random_point(self, seed=None):
        """ Returns a random point on the Plane.

        Returns
        =======

        Point3D

        Examples
        ========

        >>> from sympy import Plane
        >>> p = Plane((1, 0, 0), normal_vector=(0, 1, 0))
        >>> r = p.random_point(seed=42)  # seed value is optional
        >>> r.n(3)
        Point3D(2.29, 0, -1.35)

        The random point can be moved to lie on the circle of radius
        1 centered on p1:

        >>> c = p.p1 + (r - p.p1).unit
        >>> c.distance(p.p1).equals(1)
        True
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        params = {
            x: 2*Rational(rng.gauss(0, 1)) - 1,
            y: 2*Rational(rng.gauss(0, 1)) - 1}
        return self.arbitrary_point(x, y).subs(params)

    def parameter_value(self, other, u, v=None):
        """Return the parameter(s) corresponding to the given point.

        Examples
        ========

        >>> from sympy import pi, Plane
        >>> from sympy.abc import t, u, v
        >>> p = Plane((2, 0, 0), (0, 0, 1), (0, 1, 0))

        By default, the parameter value returned defines a point
        that is a distance of 1 from the Plane's p1 value and
        in line with the given point:

        >>> on_circle = p.arbitrary_point(t).subs(t, pi/4)
        >>> on_circle.distance(p.p1)
        1
        >>> p.parameter_value(on_circle, t)
        {t: pi/4}

        Moving the point twice as far from p1 does not change
        the parameter value:

        >>> off_circle = p.p1 + (on_circle - p.p1)*2
        >>> off_circle.distance(p.p1)
        2
        >>> p.parameter_value(off_circle, t)
        {t: pi/4}

        If the 2-value parameter is desired, supply the two
        parameter symbols and a replacement dictionary will
        be returned:

        >>> p.parameter_value(on_circle, u, v)
        {u: sqrt(10)/10, v: sqrt(10)/30}
        >>> p.parameter_value(off_circle, u, v)
        {u: sqrt(10)/5, v: sqrt(10)/15}
        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other, Point):
            raise ValueError("other must be a point")
        if other == self.p1:
            return other
        if isinstance(u, Symbol) and v is None:
            delta = self.arbitrary_point(u) - self.p1
            eq = delta - (other - self.p1).unit
            sol = solve(eq, u, dict=True)
        elif isinstance(u, Symbol) and isinstance(v, Symbol):
            pt = self.arbitrary_point(u, v)
            sol = solve(pt - other, (u, v), dict=True)
        else:
            raise ValueError('expecting 1 or 2 symbols')
        if not sol:
            raise ValueError("Given point is not on %s" % func_name(self))
        return sol[0]  # {t: tval} or {u: uval, v: vval}

    @property
    def ambient_dimension(self):
        return self.p1.ambient_dimension
<|endoftext|>from sympy.core import Expr, S, oo, pi, sympify
from sympy.core.evalf import N
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import _symbol, Dummy, Symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, tan
from .ellipse import Circle
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray
from .point import Point
from sympy.logic import And
from sympy.matrices import Matrix
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import has_dups, has_variety, uniq, rotate_left, least_rotation
from sympy.utilities.misc import as_int, func_name

from mpmath.libmp.libmpf import prec_to_dps

import warnings


x, y, T = [Dummy('polygon_dummy', real=True) for i in range(3)]


class Polygon(GeometrySet):
    """A two-dimensional polygon.

    A simple polygon in space. Can be constructed from a sequence of points
    or from a center, radius, number of sides and rotation angle.

    Parameters
    ==========

    vertices
        A sequence of points.

    n : int, optional
        If $> 0$, an n-sided RegularPolygon is created.
        Default value is $0$.

    Attributes
    ==========

    area
    angles
    perimeter
    vertices
    centroid
    sides

    Raises
    ======

    GeometryError
        If all parameters are not Points.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.line.Segment, Triangle

    Notes
    =====

    Polygons are treated as closed paths rather than 2D areas so
    some calculations can be be negative or positive (e.g., area)
    based on the orientation of the points.

    Any consecutive identical points are reduced to a single point
    and any points collinear and between two points will be removed
    unless they are needed to define an explicit intersection (see examples).

    A Triangle, Segment or Point will be returned when there are 3 or
    fewer points provided.

    Examples
    ========

    >>> from sympy import Polygon, pi
    >>> p1, p2, p3, p4, p5 = [(0, 0), (1, 0), (5, 1), (0, 1), (3, 0)]
    >>> Polygon(p1, p2, p3, p4)
    Polygon(Point2D(0, 0), Point2D(1, 0), Point2D(5, 1), Point2D(0, 1))
    >>> Polygon(p1, p2)
    Segment2D(Point2D(0, 0), Point2D(1, 0))
    >>> Polygon(p1, p2, p5)
    Segment2D(Point2D(0, 0), Point2D(3, 0))

    The area of a polygon is calculated as positive when vertices are
    traversed in a ccw direction. When the sides of a polygon cross the
    area will have positive and negative contributions. The following
    defines a Z shape where the bottom right connects back to the top
    left.

    >>> Polygon((0, 2), (2, 2), (0, 0), (2, 0)).area
    0

    When the keyword `n` is used to define the number of sides of the
    Polygon then a RegularPolygon is created and the other arguments are
    interpreted as center, radius and rotation. The unrotated RegularPolygon
    will always have a vertex at Point(r, 0) where `r` is the radius of the
    circle that circumscribes the RegularPolygon. Its method `spin` can be
    used to increment that angle.

    >>> p = Polygon((0,0), 1, n=3)
    >>> p
    RegularPolygon(Point2D(0, 0), 1, 3, 0)
    >>> p.vertices[0]
    Point2D(1, 0)
    >>> p.args[0]
    Point2D(0, 0)
    >>> p.spin(pi/2)
    >>> p.vertices[0]
    Point2D(0, 1)

    """

    __slots__ = ()

    def __new__(cls, *args, n = 0, **kwargs):
        if n:
            args = list(args)
            # return a virtual polygon with n sides
            if len(args) == 2:  # center, radius
                args.append(n)
            elif len(args) == 3:  # center, radius, rotation
                args.insert(2, n)
            return RegularPolygon(*args, **kwargs)

        vertices = [Point(a, dim=2, **kwargs) for a in args]

        # remove consecutive duplicates
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()  # last point was same as first

        # remove collinear points
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = nodup[i], nodup[i + 1], nodup[i + 2]
            if Point.is_collinear(a, b, c):
                nodup.pop(i + 1)
                if a == c:
                    nodup.pop(i)
            else:
                i += 1

        vertices = list(nodup)

        if len(vertices) > 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 3:
            return Triangle(*vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)

    @property
    def area(self):
        """
        The area of the polygon.

        Notes
        =====

        The area calculation can be positive or negative based on the
        orientation of the points. If any side of the polygon crosses
        any other side, there will be areas having opposite signs.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.area

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.area
        3

        In the Z shaped polygon (with the lower right connecting back
        to the upper left) the areas cancel out:

        >>> Z = Polygon((0, 1), (1, 1), (0, 0), (1, 0))
        >>> Z.area
        0

        In the M shaped polygon, areas do not cancel because no side
        crosses any other (though there is a point of contact).

        >>> M = Polygon((0, 0), (0, 1), (2, 0), (3, 1), (3, 0))
        >>> M.area
        -3/2

        """
        area = 0
        args = self.args
        for i in range(len(args)):
            x1, y1 = args[i - 1].args
            x2, y2 = args[i].args
            area += x1*y2 - x2*y1
        return simplify(area) / 2

    @staticmethod
    def _is_clockwise(a, b, c):
        """Return True/False for cw/ccw orientation.

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b, c = [Point(i) for i in [(0, 0), (1, 1), (1, 0)]]
        >>> Polygon._is_clockwise(a, b, c)
        True
        >>> Polygon._is_clockwise(a, c, b)
        False
        """
        ba = b - a
        ca = c - a
        t_area = simplify(ba.x*ca.y - ca.x*ba.y)
        res = t_area.is_nonpositive
        if res is None:
            raise ValueError("Can't determine orientation")
        return res

    @property
    def angles(self):
        """The internal angle at each vertex.

        Returns
        =======

        angles : dict
            A dictionary where each key is a vertex and each value is the
            internal angle at that vertex. The vertices are represented as
            Points.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.angles[p1]
        pi/2
        >>> poly.angles[p2]
        acos(-4*sqrt(17)/17)

        """

        args = self.vertices
        n = len(args)
        ret = {}
        for i in range(n):
            a, b, c = args[i - 2], args[i - 1], args[i]
            reflex_ang = Ray(b, a).angle_between(Ray(b, c))
            if self._is_clockwise(a, b, c):
                ret[b] = 2*S.Pi - reflex_ang
            else:
                ret[b] = reflex_ang

        # internal sum should be pi*(n - 2), not pi*(n+2)
        # so if ratio is (n+2)/(n-2) > 1 it is wrong
        wrong = ((sum(ret.values())/S.Pi-1)/(n - 2) - 1).is_positive
        if wrong:
            two_pi = 2*S.Pi
            for b in ret:
                ret[b] = two_pi - ret[b]
        elif wrong is None:
            raise ValueError("could not determine Polygon orientation.")
        return ret

    @property
    def ambient_dimension(self):
        return self.vertices[0].ambient_dimension

    @property
    def perimeter(self):
        """The perimeter of the polygon.

        Returns
        =======

        perimeter : number or Basic instance

        See Also
        ========

        sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.perimeter
        sqrt(17) + 7
        """
        p = 0
        args = self.vertices
        for i in range(len(args)):
            p += args[i - 1].distance(args[i])
        return simplify(p)

    @property
    def vertices(self):
        """The vertices of the polygon.

        Returns
        =======

        vertices : list of Points

        Notes
        =====

        When iterating over the vertices, it is more efficient to index self
        rather than to request the vertices and index them. Only use the
        vertices when you want to process all of them at once. This is even
        more important with RegularPolygons that calculate each vertex.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.vertices
        [Point2D(0, 0), Point2D(1, 0), Point2D(5, 1), Point2D(0, 1)]
        >>> poly.vertices[0]
        Point2D(0, 0)

        """
        return list(self.args)

    @property
    def centroid(self):
        """The centroid of the polygon.

        Returns
        =======

        centroid : Point

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.util.centroid

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.centroid
        Point2D(31/18, 11/18)

        """
        A = 1/(6*self.area)
        cx, cy = 0, 0
        args = self.args
        for i in range(len(args)):
            x1, y1 = args[i - 1].args
            x2, y2 = args[i].args
            v = x1*y2 - x2*y1
            cx += v*(x1 + x2)
            cy += v*(y1 + y2)
        return Point(simplify(A*cx), simplify(A*cy))


    def second_moment_of_area(self, point=None):
        """Returns the second moment and product moment of area of a two dimensional polygon.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point about which second moment of area is to be found.
            If "point=None" it will be calculated about the axis passing through the
            centroid of the polygon.

        Returns
        =======

        I_xx, I_yy, I_xy : number or SymPy expression
                           I_xx, I_yy are second moment of area of a two dimensional polygon.
                           I_xy is product moment of area of a two dimensional polygon.

        Examples
        ========

        >>> from sympy import Polygon, symbols
        >>> a, b = symbols('a, b')
        >>> p1, p2, p3, p4, p5 = [(0, 0), (a, 0), (a, b), (0, b), (a/3, b/3)]
        >>> rectangle = Polygon(p1, p2, p3, p4)
        >>> rectangle.second_moment_of_area()
        (a*b**3/12, a**3*b/12, 0)
        >>> rectangle.second_moment_of_area(p5)
        (a*b**3/9, a**3*b/9, a**2*b**2/36)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Second_moment_of_area

        """

        I_xx, I_yy, I_xy = 0, 0, 0
        args = self.vertices
        for i in range(len(args)):
            x1, y1 = args[i-1].args
            x2, y2 = args[i].args
            v = x1*y2 - x2*y1
            I_xx += (y1**2 + y1*y2 + y2**2)*v
            I_yy += (x1**2 + x1*x2 + x2**2)*v
            I_xy += (x1*y2 + 2*x1*y1 + 2*x2*y2 + x2*y1)*v
        A = self.area
        c_x = self.centroid[0]
        c_y = self.centroid[1]
        # parallel axis theorem
        I_xx_c = (I_xx/12) - (A*(c_y**2))
        I_yy_c = (I_yy/12) - (A*(c_x**2))
        I_xy_c = (I_xy/24) - (A*(c_x*c_y))
        if point is None:
            return I_xx_c, I_yy_c, I_xy_c

        I_xx = (I_xx_c + A*((point[1]-c_y)**2))
        I_yy = (I_yy_c + A*((point[0]-c_x)**2))
        I_xy = (I_xy_c + A*((point[0]-c_x)*(point[1]-c_y)))

        return I_xx, I_yy, I_xy


    def first_moment_of_area(self, point=None):
        """
        Returns the first moment of area of a two-dimensional polygon with
        respect to a certain point of interest.

        First moment of area is a measure of the distribution of the area
        of a polygon in relation to an axis. The first moment of area of
        the entire polygon about its own centroid is always zero. Therefore,
        here it is calculated for an area, above or below a certain point
        of interest, that makes up a smaller portion of the polygon. This
        area is bounded by the point of interest and the extreme end
        (top or bottom) of the polygon. The first moment for this area is
        is then determined about the centroidal axis of the initial polygon.

        References
        ==========

        .. [1] https://skyciv.com/docs/tutorials/section-tutorials/calculating-the-statical-or-first-moment-of-area-of-beam-sections/?cc=BMD
        .. [2] https://mechanicalc.com/reference/cross-sections

        Parameters
        ==========

        point: Point, two-tuple of sympifyable objects, or None (default=None)
            point is the point above or below which the area of interest lies
            If ``point=None`` then the centroid acts as the point of interest.

        Returns
        =======

        Q_x, Q_y: number or SymPy expressions
            Q_x is the first moment of area about the x-axis
            Q_y is the first moment of area about the y-axis
            A negative sign indicates that the section modulus is
            determined for a section below (or left of) the centroidal axis

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b = 50, 10
        >>> p1, p2, p3, p4 = [(0, b), (0, 0), (a, 0), (a, b)]
        >>> p = Polygon(p1, p2, p3, p4)
        >>> p.first_moment_of_area()
        (625, 3125)
        >>> p.first_moment_of_area(point=Point(30, 7))
        (525, 3000)
        """
        if point:
            xc, yc = self.centroid
        else:
            point = self.centroid
            xc, yc = point

        h_line = Line(point, slope=0)
        v_line = Line(point, slope=S.Infinity)

        h_poly = self.cut_section(h_line)
        v_poly = self.cut_section(v_line)

        poly_1 = h_poly[0] if h_poly[0].area <= h_poly[1].area else h_poly[1]
        poly_2 = v_poly[0] if v_poly[0].area <= v_poly[1].area else v_poly[1]

        Q_x = (poly_1.centroid.y - yc)*poly_1.area
        Q_y = (poly_2.centroid.x - xc)*poly_2.area

        return Q_x, Q_y


    def polar_second_moment_of_area(self):
        """Returns the polar modulus of a two-dimensional polygon

        It is a constituent of the second moment of area, linked through
        the perpendicular axis theorem. While the planar second moment of
        area describes an object's resistance to deflection (bending) when
        subjected to a force applied to a plane parallel to the central
        axis, the polar second moment of area describes an object's
        resistance to deflection when subjected to a moment applied in a
        plane perpendicular to the object's central axis (i.e. parallel to
        the cross-section)

        Examples
        ========

        >>> from sympy import Polygon, symbols
        >>> a, b = symbols('a, b')
        >>> rectangle = Polygon((0, 0), (a, 0), (a, b), (0, b))
        >>> rectangle.polar_second_moment_of_area()
        a**3*b/12 + a*b**3/12

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia

        """
        second_moment = self.second_moment_of_area()
        return second_moment[0] + second_moment[1]


    def section_modulus(self, point=None):
        """Returns a tuple with the section modulus of a two-dimensional
        polygon.

        Section modulus is a geometric property of a polygon defined as the
        ratio of second moment of area to the distance of the extreme end of
        the polygon from the centroidal axis.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point at which section modulus is to be found.
            If "point=None" it will be calculated for the point farthest from the
            centroidal axis of the polygon.

        Returns
        =======

        S_x, S_y: numbers or SymPy expressions
                  S_x is the section modulus with respect to the x-axis
                  S_y is the section modulus with respect to the y-axis
                  A negative sign indicates that the section modulus is
                  determined for a point below the centroidal axis

        Examples
        ========

        >>> from sympy import symbols, Polygon, Point
        >>> a, b = symbols('a, b', positive=True)
        >>> rectangle = Polygon((0, 0), (a, 0), (a, b), (0, b))
        >>> rectangle.section_modulus()
        (a*b**2/6, a**2*b/6)
        >>> rectangle.section_modulus(Point(a/4, b/4))
        (-a*b**2/3, -a**2*b/3)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Section_modulus

        """
        x_c, y_c = self.centroid
        if point is None:
            # taking x and y as maximum distances from centroid
            x_min, y_min, x_max, y_max = self.bounds
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            # taking x and y as distances of the given point from the centroid
            y = point.y - y_c
            x = point.x - x_c

        second_moment= self.second_moment_of_area()
        S_x = second_moment[0]/y
        S_y = second_moment[1]/x

        return S_x, S_y


    @property
    def sides(self):
        """The directed line segments that form the sides of the polygon.

        Returns
        =======

        sides : list of sides
            Each side is a directed Segment.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.sides
        [Segment2D(Point2D(0, 0), Point2D(1, 0)),
        Segment2D(Point2D(1, 0), Point2D(5, 1)),
        Segment2D(Point2D(5, 1), Point2D(0, 1)), Segment2D(Point2D(0, 1), Point2D(0, 0))]

        """
        res = []
        args = self.vertices
        for i in range(-len(args), 0):
            res.append(Segment(args[i], args[i + 1]))
        return res

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        verts = self.vertices
        xs = [p.x for p in verts]
        ys = [p.y for p in verts]
        return (min(xs), min(ys), max(xs), max(ys))

    def is_convex(self):
        """Is the polygon convex?

        A polygon is convex if all its interior angles are less than 180
        degrees and there are no intersections between sides.

        Returns
        =======

        is_convex : boolean
            True if this polygon is convex, False otherwise.

        See Also
        ========

        sympy.geometry.util.convex_hull

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.is_convex()
        True

        """
        # Determine orientation of points
        args = self.vertices
        cw = self._is_clockwise(args[-2], args[-1], args[0])
        for i in range(1, len(args)):
            if cw ^ self._is_clockwise(args[i - 2], args[i - 1], args[i]):
                return False
        # check for intersecting sides
        sides = self.sides
        for i, si in enumerate(sides):
            pts = si.args
            # exclude the sides connected to si
            for j in range(1 if i == len(sides) - 1 else 0, i - 1):
                sj = sides[j]
                if sj.p1 not in pts and sj.p2 not in pts:
                    hit = si.intersection(sj)
                    if hit:
                        return False
        return True

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.encloses_point

        Examples
        ========

        >>> from sympy import Polygon, Point
        >>> p = Polygon((0, 0), (4, 0), (4, 4))
        >>> p.encloses_point(Point(2, 1))
        True
        >>> p.encloses_point(Point(2, 2))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        References
        ==========

        .. [1] https://paulbourke.net/geometry/polygonmesh/#insidepoly

        """
        p = Point(p, dim=2)
        if p in self.vertices or any(p in s for s in self.sides):
            return False

        # move to p, checking that the result is numeric
        lit = []
        for v in self.vertices:
            lit.append(v - p)  # the difference is simplified
            if lit[-1].free_symbols:
                return None

        poly = Polygon(*lit)

        # polygon closure is assumed in the following test but Polygon removes duplicate pts so
        # the last point has to be added so all sides are computed. Using Polygon.sides is
        # not good since Segments are unordered.
        args = poly.args
        indices = list(range(-len(args), 1))

        if poly.is_convex():
            orientation = None
            for i in indices:
                a = args[i]
                b = args[i + 1]
                test = ((-a.y)*(b.x - a.x) - (-a.x)*(b.y - a.y)).is_negative
                if orientation is None:
                    orientation = test
                elif test is not orientation:
                    return False
            return True

        hit_odd = False
        p1x, p1y = args[0].args
        for i in indices[1:]:
            p2x, p2y = args[i].args
            if 0 > min(p1y, p2y):
                if 0 <= max(p1y, p2y):
                    if 0 <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (-p1y)*(p2x - p1x)/(p2y - p1y) + p1x
                            if p1x == p2x or 0 <= xinters:
                                hit_odd = not hit_odd
            p1x, p1y = p2x, p2y
        return hit_odd

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the polygon.

        The parameter, varying from 0 to 1, assigns points to the position on
        the perimeter that is that fraction of the total perimeter. So the
        point evaluated at t=1/2 would return the point from the first vertex
        that is 1/2 way around the polygon.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        arbitrary_point : Point

        Raises
        ======

        ValueError
            When `parameter` already appears in the Polygon's definition.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Polygon, Symbol
        >>> t = Symbol('t', real=True)
        >>> tri = Polygon((0, 0), (1, 0), (1, 1))
        >>> p = tri.arbitrary_point('t')
        >>> perimeter = tri.perimeter
        >>> s1, s2 = [s.length for s in tri.sides[:2]]
        >>> p.subs(t, (s1 + s2/2)/perimeter)
        Point2D(1, 1/2)

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError('Symbol %s already appears in object and cannot be used as a parameter.' % t.name)
        sides = []
        perimeter = self.perimeter
        perim_fraction_start = 0
        for s in self.sides:
            side_perim_fraction = s.length/perimeter
            perim_fraction_end = perim_fraction_start + side_perim_fraction
            pt = s.arbitrary_point(parameter).subs(
                t, (t - perim_fraction_start)/side_perim_fraction)
            sides.append(
                (pt, (And(perim_fraction_start <= t, t < perim_fraction_end))))
            perim_fraction_start = perim_fraction_end
        return Piecewise(*sides)

    def parameter_value(self, other, t):
        if not isinstance(other,GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other,Point):
            raise ValueError("other must be a point")
        if other.free_symbols:
            raise NotImplementedError('non-numeric coordinates')
        unknown = False
        p = self.arbitrary_point(T)
        for pt, cond in p.args:
            sol = solve(pt - other, T, dict=True)
            if not sol:
                continue
            value = sol[0][T]
            if simplify(cond.subs(T, value)) == True:
                return {t: value}
            unknown = True
        if unknown:
            raise ValueError("Given point may not be on %s" % func_name(self))
        raise ValueError("Given point is not on %s" % func_name(self))

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the polygon.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list (plot interval)
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Polygon
        >>> p = Polygon((0, 0), (1, 0), (1, 1))
        >>> p.plot_interval()
        [t, 0, 1]

        """
        t = Symbol(parameter, real=True)
        return [t, 0, 1]

    def intersection(self, o):
        """The intersection of polygon and geometry entity.

        The intersection may be empty and can contain individual Points and
        complete Line Segments.

        Parameters
        ==========

        other: GeometryEntity

        Returns
        =======

        intersection : list
            The list of Segments and Points

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon, Line
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly1 = Polygon(p1, p2, p3, p4)
        >>> p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
        >>> poly2 = Polygon(p5, p6, p7)
        >>> poly1.intersection(poly2)
        [Point2D(1/3, 1), Point2D(2/3, 0), Point2D(9/5, 1/5), Point2D(7/3, 1)]
        >>> poly1.intersection(Line(p1, p2))
        [Segment2D(Point2D(0, 0), Point2D(1, 0))]
        >>> poly1.intersection(p1)
        [Point2D(0, 0)]
        """
        intersection_result = []
        k = o.sides if isinstance(o, Polygon) else [o]
        for side in self.sides:
            for side1 in k:
                intersection_result.extend(side.intersection(side1))

        intersection_result = list(uniq(intersection_result))
        points = [entity for entity in intersection_result if isinstance(entity, Point)]
        segments = [entity for entity in intersection_result if isinstance(entity, Segment)]

        if points and segments:
            points_in_segments = list(uniq([point for point in points for segment in segments if point in segment]))
            if points_in_segments:
                for i in points_in_segments:
                    points.remove(i)
            return list(ordered(segments + points))
        else:
            return list(ordered(intersection_result))


    def cut_section(self, line):
        """
        Returns a tuple of two polygon segments that lie above and below
        the intersecting line respectively.

        Parameters
        ==========

        line: Line object of geometry module
            line which cuts the Polygon. The part of the Polygon that lies
            above and below this line is returned.

        Returns
        =======

        upper_polygon, lower_polygon: Polygon objects or None
            upper_polygon is the polygon that lies above the given line.
            lower_polygon is the polygon that lies below the given line.
            upper_polygon and lower polygon are ``None`` when no polygon
            exists above the line or below the line.

        Raises
        ======

        ValueError: When the line does not intersect the polygon

        Examples
        ========

        >>> from sympy import Polygon, Line
        >>> a, b = 20, 10
        >>> p1, p2, p3, p4 = [(0, b), (0, 0), (a, 0), (a, b)]
        >>> rectangle = Polygon(p1, p2, p3, p4)
        >>> t = rectangle.cut_section(Line((0, 5), slope=0))
        >>> t
        (Polygon(Point2D(0, 10), Point2D(0, 5), Point2D(20, 5), Point2D(20, 10)),
        Polygon(Point2D(0, 5), Point2D(0, 0), Point2D(20, 0), Point2D(20, 5)))
        >>> upper_segment, lower_segment = t
        >>> upper_segment.area
        100
        >>> upper_segment.centroid
        Point2D(10, 15/2)
        >>> lower_segment.centroid
        Point2D(10, 5/2)

        References
        ==========

        .. [1] https://github.com/sympy/sympy/wiki/A-method-to-return-a-cut-section-of-any-polygon-geometry

        """
        intersection_points = self.intersection(line)
        if not intersection_points:
            raise ValueError("This line does not intersect the polygon")

        points = list(self.vertices)
        points.append(points[0])

        eq = line.equation(x, y)

        # considering equation of line to be `ax +by + c`
        a = eq.coeff(x)
        b = eq.coeff(y)

        upper_vertices = []
        lower_vertices = []
        # prev is true when previous point is above the line
        prev = True
        prev_point = None
        for point in points:
            # when coefficient of y is 0, right side of the line is
            # considered
            compare = eq.subs({x: point.x, y: point.y})/b if b \
                    else eq.subs(x, point.x)/a

            # if point lies above line
            if compare > 0:
                if not prev:
                    # if previous point lies below the line, the intersection
                    # point of the polygon edge and the line has to be included
                    edge = Line(point, prev_point)
                    new_point = edge.intersection(line)
                    upper_vertices.append(new_point[0])
                    lower_vertices.append(new_point[0])

                upper_vertices.append(point)
                prev = True
            else:
                if prev and prev_point:
                    edge = Line(point, prev_point)
                    new_point = edge.intersection(line)
                    upper_vertices.append(new_point[0])
                    lower_vertices.append(new_point[0])
                lower_vertices.append(point)
                prev = False
            prev_point = point

        upper_polygon, lower_polygon = None, None
        if upper_vertices and isinstance(Polygon(*upper_vertices), Polygon):
            upper_polygon = Polygon(*upper_vertices)
        if lower_vertices and isinstance(Polygon(*lower_vertices), Polygon):
            lower_polygon = Polygon(*lower_vertices)

        return upper_polygon, lower_polygon


    def distance(self, o):
        """
        Returns the shortest distance between self and o.

        If o is a point, then self does not need to be convex.
        If o is another polygon self and o must be convex.

        Examples
        ========

        >>> from sympy import Point, Polygon, RegularPolygon
        >>> p1, p2 = map(Point, [(0, 0), (7, 5)])
        >>> poly = Polygon(*RegularPolygon(p1, 1, 3).vertices)
        >>> poly.distance(p2)
        sqrt(61)
        """
        if isinstance(o, Point):
            dist = oo
            for side in self.sides:
                current = side.distance(o)
                if current == 0:
                    return S.Zero
                elif current < dist:
                    dist = current
            return dist
        elif isinstance(o, Polygon) and self.is_convex() and o.is_convex():
            return self._do_poly_distance(o)
        raise NotImplementedError()

    def _do_poly_distance(self, e2):
        """
        Calculates the least distance between the exteriors of two
        convex polygons e1 and e2. Does not check for the convexity
        of the polygons as this is checked by Polygon.distance.

        Notes
        =====

            - Prints a warning if the two polygons possibly intersect as the return
              value will not be valid in such a case. For a more through test of
              intersection use intersection().

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> square = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
        >>> triangle = Polygon(Point(1, 2), Point(2, 2), Point(2, 1))
        >>> square._do_poly_distance(triangle)
        sqrt(2)/2

        Description of method used
        ==========================

        Method:
        [1] https://web.archive.org/web/20150509035744/http://cgm.cs.mcgill.ca/~orm/mind2p.html
        Uses rotating calipers:
        [2] https://en.wikipedia.org/wiki/Rotating_calipers
        and antipodal points:
        [3] https://en.wikipedia.org/wiki/Antipodal_point
        """
        e1 = self

        '''Tests for a possible intersection between the polygons and outputs a warning'''
        e1_center = e1.centroid
        e2_center = e2.centroid
        e1_max_radius = S.Zero
        e2_max_radius = S.Zero
        for vertex in e1.vertices:
            r = Point.distance(e1_center, vertex)
            if e1_max_radius < r:
                e1_max_radius = r
        for vertex in e2.vertices:
            r = Point.distance(e2_center, vertex)
            if e2_max_radius < r:
                e2_max_radius = r
        center_dist = Point.distance(e1_center, e2_center)
        if center_dist <= e1_max_radius + e2_max_radius:
            warnings.warn("Polygons may intersect producing erroneous output",
                          stacklevel=3)

        '''
        Find the upper rightmost vertex of e1 and the lowest leftmost vertex of e2
        '''
        e1_ymax = Point(0, -oo)
        e2_ymin = Point(0, oo)

        for vertex in e1.vertices:
            if vertex.y > e1_ymax.y or (vertex.y == e1_ymax.y and vertex.x > e1_ymax.x):
                e1_ymax = vertex
        for vertex in e2.vertices:
            if vertex.y < e2_ymin.y or (vertex.y == e2_ymin.y and vertex.x < e2_ymin.x):
                e2_ymin = vertex
        min_dist = Point.distance(e1_ymax, e2_ymin)

        '''
        Produce a dictionary with vertices of e1 as the keys and, for each vertex, the points
        to which the vertex is connected as its value. The same is then done for e2.
        '''
        e1_connections = {}
        e2_connections = {}

        for side in e1.sides:
            if side.p1 in e1_connections:
                e1_connections[side.p1].append(side.p2)
            else:
                e1_connections[side.p1] = [side.p2]

            if side.p2 in e1_connections:
                e1_connections[side.p2].append(side.p1)
            else:
                e1_connections[side.p2] = [side.p1]

        for side in e2.sides:
            if side.p1 in e2_connections:
                e2_connections[side.p1].append(side.p2)
            else:
                e2_connections[side.p1] = [side.p2]

            if side.p2 in e2_connections:
                e2_connections[side.p2].append(side.p1)
            else:
                e2_connections[side.p2] = [side.p1]

        e1_current = e1_ymax
        e2_current = e2_ymin
        support_line = Line(Point(S.Zero, S.Zero), Point(S.One, S.Zero))

        '''
        Determine which point in e1 and e2 will be selected after e2_ymin and e1_ymax,
        this information combined with the above produced dictionaries determines the
        path that will be taken around the polygons
        '''
        point1 = e1_connections[e1_ymax][0]
        point2 = e1_connections[e1_ymax][1]
        angle1 = support_line.angle_between(Line(e1_ymax, point1))
        angle2 = support_line.angle_between(Line(e1_ymax, point2))
        if angle1 < angle2:
            e1_next = point1
        elif angle2 < angle1:
            e1_next = point2
        elif Point.distance(e1_ymax, point1) > Point.distance(e1_ymax, point2):
            e1_next = point2
        else:
            e1_next = point1

        point1 = e2_connections[e2_ymin][0]
        point2 = e2_connections[e2_ymin][1]
        angle1 = support_line.angle_between(Line(e2_ymin, point1))
        angle2 = support_line.angle_between(Line(e2_ymin, point2))
        if angle1 > angle2:
            e2_next = point1
        elif angle2 > angle1:
            e2_next = point2
        elif Point.distance(e2_ymin, point1) > Point.distance(e2_ymin, point2):
            e2_next = point2
        else:
            e2_next = point1

        '''
        Loop which determines the distance between anti-podal pairs and updates the
        minimum distance accordingly. It repeats until it reaches the starting position.
        '''
        while True:
            e1_angle = support_line.angle_between(Line(e1_current, e1_next))
            e2_angle = pi - support_line.angle_between(Line(
                e2_current, e2_next))

            if (e1_angle < e2_angle) is True:
                support_line = Line(e1_current, e1_next)
                e1_segment = Segment(e1_current, e1_next)
                min_dist_current = e1_segment.distance(e2_current)

                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e1_connections[e1_next][0] != e1_current:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][0]
                else:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][1]
            elif (e1_angle > e2_angle) is True:
                support_line = Line(e2_next, e2_current)
                e2_segment = Segment(e2_current, e2_next)
                min_dist_current = e2_segment.distance(e1_current)

                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e2_connections[e2_next][0] != e2_current:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][0]
                else:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][1]
            else:
                support_line = Line(e1_current, e1_next)
                e1_segment = Segment(e1_current, e1_next)
                e2_segment = Segment(e2_current, e2_next)
                min1 = e1_segment.distance(e2_next)
                min2 = e2_segment.distance(e1_next)

                min_dist_current = min(min1, min2)
                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e1_connections[e1_next][0] != e1_current:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][0]
                else:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][1]

                if e2_connections[e2_next][0] != e2_current:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][0]
                else:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][1]
            if e1_current == e1_ymax and e2_current == e2_ymin:
                break
        return min_dist

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the Polygon.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        verts = map(N, self.vertices)
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        path = "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
            ).format(2. * scale_factor, path, fill_color)

    def _hashable_content(self):

        D = {}
        def ref_list(point_list):
            kee = {}
            for i, p in enumerate(ordered(set(point_list))):
                kee[p] = i
                D[i] = p
            return [kee[p] for p in point_list]

        S1 = ref_list(self.args)
        r_nor = rotate_left(S1, least_rotation(S1))
        S2 = ref_list(list(reversed(self.args)))
        r_rev = rotate_left(S2, least_rotation(S2))
        if r_nor < r_rev:
            r = r_nor
        else:
            r = r_rev
        canonical_args = [ D[order] for order in r ]
        return tuple(canonical_args)

    def __contains__(self, o):
        """
        Return True if o is contained within the boundary lines of self.altitudes

        Parameters
        ==========

        other : GeometryEntity

        Returns
        =======

        contained in : bool
            The points (and sides, if applicable) are contained in self.

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.encloses

        Examples
        ========

        >>> from sympy import Line, Segment, Point
        >>> p = Point(0, 0)
        >>> q = Point(1, 1)
        >>> s = Segment(p, q*2)
        >>> l = Line(p, q)
        >>> p in q
        False
        >>> p in s
        True
        >>> q*3 in s
        False
        >>> s in l
        True

        """

        if isinstance(o, Polygon):
            return self == o
        elif isinstance(o, Segment):
            return any(o in s for s in self.sides)
        elif isinstance(o, Point):
            if o in self.vertices:
                return True
            for side in self.sides:
                if o in side:
                    return True

        return False

    def bisectors(p, prec=None):
        """Returns angle bisectors of a polygon. If prec is given
        then approximate the point defining the ray to that precision.

        The distance between the points defining the bisector ray is 1.

        Examples
        ========

        >>> from sympy import Polygon, Point
        >>> p = Polygon(Point(0, 0), Point(2, 0), Point(1, 1), Point(0, 3))
        >>> p.bisectors(2)
        {Point2D(0, 0): Ray2D(Point2D(0, 0), Point2D(0.71, 0.71)),
         Point2D(0, 3): Ray2D(Point2D(0, 3), Point2D(0.23, 2.0)),
         Point2D(1, 1): Ray2D(Point2D(1, 1), Point2D(0.19, 0.42)),
         Point2D(2, 0): Ray2D(Point2D(2, 0), Point2D(1.1, 0.38))}
        """
        b = {}
        pts = list(p.args)
        pts.append(pts[0])  # close it
        cw = Polygon._is_clockwise(*pts[:3])
        if cw:
            pts = list(reversed(pts))
        for v, a in p.angles.items():
            i = pts.index(v)
            p1, p2 = Point._normalize_dimension(pts[i], pts[i + 1])
            ray = Ray(p1, p2).rotate(a/2, v)
            dir = ray.direction
            ray = Ray(ray.p1, ray.p1 + dir/dir.distance((0, 0)))
            if prec is not None:
                ray = Ray(ray.p1, ray.p2.n(prec))
            b[v] = ray
        return b


class RegularPolygon(Polygon):
    """
    A regular polygon.

    Such a polygon has all internal angles equal and all sides the same length.

    Parameters
    ==========

    center : Point
    radius : number or Basic instance
        The distance from the center to a vertex
    n : int
        The number of sides

    Attributes
    ==========

    vertices
    center
    radius
    rotation
    apothem
    interior_angle
    exterior_angle
    circumcircle
    incircle
    angles

    Raises
    ======

    GeometryError
        If the `center` is not a Point, or the `radius` is not a number or Basic
        instance, or the number of sides, `n`, is less than three.

    Notes
    =====

    A RegularPolygon can be instantiated with Polygon with the kwarg n.

    Regular polygons are instantiated with a center, radius, number of sides
    and a rotation angle. Whereas the arguments of a Polygon are vertices, the
    vertices of the RegularPolygon must be obtained with the vertices method.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import RegularPolygon, Point
    >>> r = RegularPolygon(Point(0, 0), 5, 3)
    >>> r
    RegularPolygon(Point2D(0, 0), 5, 3, 0)
    >>> r.vertices[0]
    Point2D(5, 0)

    """

    __slots__ = ('_n', '_center', '_radius', '_rot')

    def __new__(self, c, r, n, rot=0, **kwargs):
        r, n, rot = map(sympify, (r, n, rot))
        c = Point(c, dim=2, **kwargs)
        if not isinstance(r, Expr):
            raise GeometryError("r must be an Expr object, not %s" % r)
        if n.is_Number:
            as_int(n)  # let an error raise if necessary
            if n < 3:
                raise GeometryError("n must be a >= 3, not %s" % n)

        obj = GeometryEntity.__new__(self, c, r, n, **kwargs)
        obj._n = n
        obj._center = c
        obj._radius = r
        obj._rot = rot % (2*S.Pi/n) if rot.is_number else rot
        return obj

    def _eval_evalf(self, prec=15, **options):
        c, r, n, a = self.args
        dps = prec_to_dps(prec)
        c, r, a = [i.evalf(n=dps, **options) for i in (c, r, a)]
        return self.func(c, r, n, a)

    @property
    def args(self):
        """
        Returns the center point, the radius,
        the number of sides, and the orientation angle.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.args
        (Point2D(0, 0), 5, 3, 0)
        """
        return self._center, self._radius, self._n, self._rot

    def __str__(self):
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    def __repr__(self):
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    @property
    def area(self):
        """Returns the area.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> square = RegularPolygon((0, 0), 1, 4)
        >>> square.area
        2
        >>> _ == square.length**2
        True
        """
        c, r, n, rot = self.args
        return sign(r)*n*self.length**2/(4*tan(pi/n))

    @property
    def length(self):
        """Returns the length of the sides.

        The half-length of the side and the apothem form two legs
        of a right triangle whose hypotenuse is the radius of the
        regular polygon.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> from sympy import sqrt
        >>> s = square_in_unit_circle = RegularPolygon((0, 0), 1, 4)
        >>> s.length
        sqrt(2)
        >>> sqrt((_/2)**2 + s.apothem**2) == s.radius
        True

        """
        return self.radius*2*sin(pi/self._n)

    @property
    def center(self):
        """The center of the RegularPolygon

        This is also the center of the circumscribing circle.

        Returns
        =======

        center : Point

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.center

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.center
        Point2D(0, 0)
        """
        return self._center

    centroid = center

    @property
    def circumcenter(self):
        """
        Alias for center.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.circumcenter
        Point2D(0, 0)
        """
        return self.center

    @property
    def radius(self):
        """Radius of the RegularPolygon

        This is also the radius of the circumscribing circle.

        Returns
        =======

        radius : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.radius
        r

        """
        return self._radius

    @property
    def circumradius(self):
        """
        Alias for radius.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.circumradius
        r
        """
        return self.radius

    @property
    def rotation(self):
        """CCW angle by which the RegularPolygon is rotated

        Returns
        =======

        rotation : number or instance of Basic

        Examples
        ========

        >>> from sympy import pi
        >>> from sympy.abc import a
        >>> from sympy import RegularPolygon, Point
        >>> RegularPolygon(Point(0, 0), 3, 4, pi/4).rotation
        pi/4

        Numerical rotation angles are made canonical:

        >>> RegularPolygon(Point(0, 0), 3, 4, a).rotation
        a
        >>> RegularPolygon(Point(0, 0), 3, 4, pi).rotation
        0

        """
        return self._rot

    @property
    def apothem(self):
        """The inradius of the RegularPolygon.

        The apothem/inradius is the radius of the inscribed circle.

        Returns
        =======

        apothem : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.apothem
        sqrt(2)*r/2

        """
        return self.radius * cos(S.Pi/self._n)

    @property
    def inradius(self):
        """
        Alias for apothem.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.inradius
        sqrt(2)*r/2
        """
        return self.apothem

    @property
    def interior_angle(self):
        """Measure of the interior angles.

        Returns
        =======

        interior_angle : number

        See Also
        ========

        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.interior_angle
        3*pi/4

        """
        return (self._n - 2)*S.Pi/self._n

    @property
    def exterior_angle(self):
        """Measure of the exterior angles.

        Returns
        =======

        exterior_angle : number

        See Also
        ========

        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.exterior_angle
        pi/4

        """
        return 2*S.Pi/self._n

    @property
    def circumcircle(self):
        """The circumcircle of the RegularPolygon.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        circumcenter, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.circumcircle
        Circle(Point2D(0, 0), 4)

        """
        return Circle(self.center, self.radius)

    @property
    def incircle(self):
        """The incircle of the RegularPolygon.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        inradius, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 7)
        >>> rp.incircle
        Circle(Point2D(0, 0), 4*cos(pi/7))

        """
        return Circle(self.center, self.apothem)

    @property
    def angles(self):
        """
        Returns a dictionary with keys, the vertices of the Polygon,
        and values, the interior angle at each vertex.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.angles
        {Point2D(-5/2, -5*sqrt(3)/2): pi/3,
         Point2D(-5/2, 5*sqrt(3)/2): pi/3,
         Point2D(5, 0): pi/3}
        """
        ret = {}
        ang = self.interior_angle
        for v in self.vertices:
            ret[v] = ang
        return ret

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        The general Polygon.encloses_point method is called only if
        a point is not within or beyond the incircle or circumcircle,
        respectively.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, S, Point, Symbol
        >>> p = RegularPolygon((0, 0), 3, 4)
        >>> p.encloses_point(Point(0, 0))
        True
        >>> r, R = p.inradius, p.circumradius
        >>> p.encloses_point(Point((r + R)/2, 0))
        True
        >>> p.encloses_point(Point(R/2, R/2 + (R - r)/10))
        False
        >>> t = Symbol('t', real=True)
        >>> p.encloses_point(p.arbitrary_point().subs(t, S.Half))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        """

        c = self.center
        d = Segment(c, p).length
        if d >= self.radius:
            return False
        elif d < self.inradius:
            return True
        else:
            # now enumerate the RegularPolygon like a general polygon.
            return Polygon.encloses_point(self, p)

    def spin(self, angle):
        """Increment *in place* the virtual Polygon's rotation by ccw angle.

        See also: rotate method which moves the center.

        >>> from sympy import Polygon, Point, pi
        >>> r = Polygon(Point(0,0), 1, n=3)
        >>> r.vertices[0]
        Point2D(1, 0)
        >>> r.spin(pi/6)
        >>> r.vertices[0]
        Point2D(sqrt(3)/2, 1/2)

        See Also
        ========

        rotation
        rotate : Creates a copy of the RegularPolygon rotated about a Point

        """
        self._rot += angle

    def rotate(self, angle, pt=None):
        """Override GeometryEntity.rotate to first rotate the RegularPolygon
        about its center.

        >>> from sympy import Point, RegularPolygon, pi
        >>> t = RegularPolygon(Point(1, 0), 1, 3)
        >>> t.vertices[0] # vertex on x-axis
        Point2D(2, 0)
        >>> t.rotate(pi/2).vertices[0] # vertex on y axis now
        Point2D(0, 2)

        See Also
        ========

        rotation
        spin : Rotates a RegularPolygon in place

        """

        r = type(self)(*self.args)  # need a copy or else changes are in-place
        r._rot += angle
        return GeometryEntity.rotate(r, angle, pt)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since it is the radius that must be
        scaled (if x == y) or else a new Polygon must be returned.

        >>> from sympy import RegularPolygon

        Symmetric scaling returns a RegularPolygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 2)
        RegularPolygon(Point2D(0, 0), 2, 4, 0)

        Asymmetric scaling returns a kite as a Polygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 1)
        Polygon(Point2D(2, 0), Point2D(0, 1), Point2D(-2, 0), Point2D(0, -1))

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        if x != y:
            return Polygon(*self.vertices).scale(x, y)
        c, r, n, rot = self.args
        r *= x
        return self.func(c, r, n, rot)

    def reflect(self, line):
        """Override GeometryEntity.reflect since this is not made of only
        points.

        Examples
        ========

        >>> from sympy import RegularPolygon, Line

        >>> RegularPolygon((0, 0), 1, 4).reflect(Line((0, 1), slope=-2))
        RegularPolygon(Point2D(4/5, 2/5), -1, 4, atan(4/3))

        """
        c, r, n, rot = self.args
        v = self.vertices[0]
        d = v - c
        cc = c.reflect(line)
        vv = v.reflect(line)
        dd = vv - cc
        # calculate rotation about the new center
        # which will align the vertices
        l1 = Ray((0, 0), dd)
        l2 = Ray((0, 0), d)
        ang = l1.closing_angle(l2)
        rot += ang
        # change sign of radius as point traversal is reversed
        return self.func(cc, -r, n, rot)

    @property
    def vertices(self):
        """The vertices of the RegularPolygon.

        Returns
        =======

        vertices : list
            Each vertex is a Point.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.vertices
        [Point2D(5, 0), Point2D(0, 5), Point2D(-5, 0), Point2D(0, -5)]

        """
        c = self._center
        r = abs(self._radius)
        rot = self._rot
        v = 2*S.Pi/self._n

        return [Point(c.x + r*cos(k*v + rot), c.y + r*sin(k*v + rot))
                for k in range(self._n)]

    def __eq__(self, o):
        if not isinstance(o, Polygon):
            return False
        elif not isinstance(o, RegularPolygon):
            return Polygon.__eq__(o, self)
        return self.args == o.args

    def __hash__(self):
        return super().__hash__()


class Triangle(Polygon):
    """
    A polygon with three vertices and three sides.

    Parameters
    ==========

    points : sequence of Points
    keyword: asa, sas, or sss to specify sides/angles of the triangle

    Attributes
    ==========

    vertices
    altitudes
    orthocenter
    circumcenter
    circumradius
    circumcircle
    inradius
    incircle
    exradii
    medians
    medial
    nine_point_circle

    Raises
    ======

    GeometryError
        If the number of vertices is not equal to three, or one of the vertices
        is not a Point, or a valid keyword is not given.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import Triangle, Point
    >>> Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
    Triangle(Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

    Keywords sss, sas, or asa can be used to give the desired
    side lengths (in order) and interior angles (in degrees) that
    define the triangle:

    >>> Triangle(sss=(3, 4, 5))
    Triangle(Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> Triangle(asa=(30, 1, 30))
    Triangle(Point2D(0, 0), Point2D(1, 0), Point2D(1/2, sqrt(3)/6))
    >>> Triangle(sas=(1, 45, 2))
    Triangle(Point2D(0, 0), Point2D(2, 0), Point2D(sqrt(2)/2, sqrt(2)/2))

    """

    def __new__(cls, *args, **kwargs):
        if len(args) != 3:
            if 'sss' in kwargs:
                return _sss(*[simplify(a) for a in kwargs['sss']])
            if 'asa' in kwargs:
                return _asa(*[simplify(a) for a in kwargs['asa']])
            if 'sas' in kwargs:
                return _sas(*[simplify(a) for a in kwargs['sas']])
            msg = "Triangle instantiates with three points or a valid keyword."
            raise GeometryError(msg)

        vertices = [Point(a, dim=2, **kwargs) for a in args]

        # remove consecutive duplicates
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()  # last point was same as first

        # remove collinear points
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = sorted(
                [nodup[i], nodup[i + 1], nodup[i + 2]], key=default_sort_key)
            if Point.is_collinear(a, b, c):
                nodup[i] = a
                nodup[i + 1] = None
                nodup.pop(i + 1)
            i += 1

        vertices = list(filter(lambda x: x is not None, nodup))

        if len(vertices) == 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)

    @property
    def vertices(self):
        """The triangle's vertices

        Returns
        =======

        vertices : tuple
            Each element in the tuple is a Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t.vertices
        (Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

        """
        return self.args

    def is_similar(t1, t2):
        """Is another triangle similar to this one.

        Two triangles are similar if one can be uniformly scaled to the other.

        Parameters
        ==========

        other: Triangle

        Returns
        =======

        is_similar : boolean

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.is_similar

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -3))
        >>> t1.is_similar(t2)
        True

        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -4))
        >>> t1.is_similar(t2)
        False

        """
        if not isinstance(t2, Polygon):
            return False

        s1_1, s1_2, s1_3 = [side.length for side in t1.sides]
        s2 = [side.length for side in t2.sides]

        def _are_similar(u1, u2, u3, v1, v2, v3):
            e1 = simplify(u1/v1)
            e2 = simplify(u2/v2)
            e3 = simplify(u3/v3)
            return bool(e1 == e2) and bool(e2 == e3)

        # There's only 6 permutations, so write them out
        return _are_similar(s1_1, s1_2, s1_3, *s2) or \
            _are_similar(s1_1, s1_3, s1_2, *s2) or \
            _are_similar(s1_2, s1_1, s1_3, *s2) or \
            _are_similar(s1_2, s1_3, s1_1, *s2) or \
            _are_similar(s1_3, s1_1, s1_2, *s2) or \
            _are_similar(s1_3, s1_2, s1_1, *s2)

    def is_equilateral(self):
        """Are all the sides the same length?

        Returns
        =======

        is_equilateral : boolean

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.is_similar, RegularPolygon
        is_isosceles, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_equilateral()
        False

        >>> from sympy import sqrt
        >>> t2 = Triangle(Point(0, 0), Point(10, 0), Point(5, 5*sqrt(3)))
        >>> t2.is_equilateral()
        True

        """
        return not has_variety(s.length for s in self.sides)

    def is_isosceles(self):
        """Are two or more of the sides the same length?

        Returns
        =======

        is_isosceles : boolean

        See Also
        ========

        is_equilateral, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(2, 4))
        >>> t1.is_isosceles()
        True

        """
        return has_dups(s.length for s in self.sides)

    def is_scalene(self):
        """Are all the sides of the triangle of different lengths?

        Returns
        =======

        is_scalene : boolean

        See Also
        ========

        is_equilateral, is_isosceles, is_right

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(1, 4))
        >>> t1.is_scalene()
        True

        """
        return not has_dups(s.length for s in self.sides)

    def is_right(self):
        """Is the triangle right-angled.

        Returns
        =======

        is_right : boolean

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular
        is_equilateral, is_isosceles, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_right()
        True

        """
        s = self.sides
        return Segment.is_perpendicular(s[0], s[1]) or \
            Segment.is_perpendicular(s[1], s[2]) or \
            Segment.is_perpendicular(s[0], s[2])

    @property
    def altitudes(self):
        """The altitudes of the triangle.

        An altitude of a triangle is a segment through a vertex,
        perpendicular to the opposite side, with length being the
        height of the vertex measured from the line containing the side.

        Returns
        =======

        altitudes : dict
