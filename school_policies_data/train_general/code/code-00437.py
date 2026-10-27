            The dictionary consists of keys which are vertices and values
            which are Segments.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.altitudes[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        return {v[0]: s[1].perpendicular_segment(v[0]),
                v[1]: s[2].perpendicular_segment(v[1]),
                v[2]: s[0].perpendicular_segment(v[2])}

    @property
    def orthocenter(self):
        """The orthocenter of the triangle.

        The orthocenter is the intersection of the altitudes of a triangle.
        It may lie inside, outside or on the triangle.

        Returns
        =======

        orthocenter : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.orthocenter
        Point2D(0, 0)

        """
        a = self.altitudes
        v = self.vertices
        return Line(a[v[0]]).intersection(Line(a[v[1]]))[0]

    @property
    def circumcenter(self):
        """The circumcenter of the triangle

        The circumcenter is the center of the circumcircle.

        Returns
        =======

        circumcenter : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcenter
        Point2D(1/2, 1/2)
        """
        a, b, c = [x.perpendicular_bisector() for x in self.sides]
        return a.intersection(b)[0]

    @property
    def circumradius(self):
        """The radius of the circumcircle of the triangle.

        Returns
        =======

        circumradius : number of Basic instance

        See Also
        ========

        sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import Point, Triangle
        >>> a = Symbol('a')
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, a)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumradius
        sqrt(a**2/4 + 1/4)
        """
        return Point.distance(self.circumcenter, self.vertices[0])

    @property
    def circumcircle(self):
        """The circle which passes through the three vertices of the triangle.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcircle
        Circle(Point2D(1/2, 1/2), sqrt(2)/2)

        """
        return Circle(self.circumcenter, self.circumradius)

    def bisectors(self):
        """The angle bisectors of the triangle.

        An angle bisector of a triangle is a straight line through a vertex
        which cuts the corresponding angle in half.

        Returns
        =======

        bisectors : dict
            Each key is a vertex (Point) and each value is the corresponding
            bisector (Segment).

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Triangle, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> from sympy import sqrt
        >>> t.bisectors()[p2] == Segment(Point(1, 0), Point(0, sqrt(2) - 1))
        True

        """
        # use lines containing sides so containment check during
        # intersection calculation can be avoided, thus reducing
        # the processing time for calculating the bisectors
        s = [Line(l) for l in self.sides]
        v = self.vertices
        c = self.incenter
        l1 = Segment(v[0], Line(v[0], c).intersection(s[1])[0])
        l2 = Segment(v[1], Line(v[1], c).intersection(s[2])[0])
        l3 = Segment(v[2], Line(v[2], c).intersection(s[0])[0])
        return {v[0]: l1, v[1]: l2, v[2]: l3}

    @property
    def incenter(self):
        """The center of the incircle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incenter : Point

        See Also
        ========

        incircle, sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incenter
        Point2D(1 - sqrt(2)/2, 1 - sqrt(2)/2)

        """
        s = self.sides
        l = Matrix([s[i].length for i in [1, 2, 0]])
        p = sum(l)
        v = self.vertices
        x = simplify(l.dot(Matrix([vi.x for vi in v]))/p)
        y = simplify(l.dot(Matrix([vi.y for vi in v]))/p)
        return Point(x, y)

    @property
    def inradius(self):
        """The radius of the incircle.

        Returns
        =======

        inradius : number of Basic instance

        See Also
        ========

        incircle, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(4, 0), Point(0, 3)
        >>> t = Triangle(p1, p2, p3)
        >>> t.inradius
        1

        """
        return simplify(2 * self.area / self.perimeter)

    @property
    def incircle(self):
        """The incircle of the triangle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(2, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incircle
        Circle(Point2D(2 - sqrt(2), 2 - sqrt(2)), 2 - sqrt(2))

        """
        return Circle(self.incenter, self.inradius)

    @property
    def exradii(self):
        """The radius of excircles of a triangle.

        An excircle of the triangle is a circle lying outside the triangle,
        tangent to one of its sides and tangent to the extensions of the
        other two.

        Returns
        =======

        exradii : dict

        See Also
        ========

        sympy.geometry.polygon.Triangle.inradius

        Examples
        ========

        The exradius touches the side of the triangle to which it is keyed, e.g.
        the exradius touching side 2 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.exradii[t.sides[2]]
        -2 + sqrt(10)

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Exradius.html
        .. [2] https://mathworld.wolfram.com/Excircles.html

        """

        side = self.sides
        a = side[0].length
        b = side[1].length
        c = side[2].length
        s = (a+b+c)/2
        area = self.area
        exradii = {self.sides[0]: simplify(area/(s-a)),
                   self.sides[1]: simplify(area/(s-b)),
                   self.sides[2]: simplify(area/(s-c))}

        return exradii

    @property
    def excenters(self):
        """Excenters of the triangle.

        An excenter is the center of a circle that is tangent to a side of the
        triangle and the extensions of the other two sides.

        Returns
        =======

        excenters : dict


        Examples
        ========

        The excenters are keyed to the side of the triangle to which their corresponding
        excircle is tangent: The center is keyed, e.g. the excenter of a circle touching
        side 0 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.excenters[t.sides[0]]
        Point2D(12*sqrt(10), 2/3 + sqrt(10)/3)

        See Also
        ========

        sympy.geometry.polygon.Triangle.exradii

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Excircles.html

        """

        s = self.sides
        v = self.vertices
        a = s[0].length
        b = s[1].length
        c = s[2].length
        x = [v[0].x, v[1].x, v[2].x]
        y = [v[0].y, v[1].y, v[2].y]

        exc_coords = {
            "x1": simplify(-a*x[0]+b*x[1]+c*x[2]/(-a+b+c)),
            "x2": simplify(a*x[0]-b*x[1]+c*x[2]/(a-b+c)),
            "x3": simplify(a*x[0]+b*x[1]-c*x[2]/(a+b-c)),
            "y1": simplify(-a*y[0]+b*y[1]+c*y[2]/(-a+b+c)),
            "y2": simplify(a*y[0]-b*y[1]+c*y[2]/(a-b+c)),
            "y3": simplify(a*y[0]+b*y[1]-c*y[2]/(a+b-c))
        }

        excenters = {
            s[0]: Point(exc_coords["x1"], exc_coords["y1"]),
            s[1]: Point(exc_coords["x2"], exc_coords["y2"]),
            s[2]: Point(exc_coords["x3"], exc_coords["y3"])
        }

        return excenters

    @property
    def medians(self):
        """The medians of the triangle.

        A median of a triangle is a straight line through a vertex and the
        midpoint of the opposite side, and divides the triangle into two
        equal areas.

        Returns
        =======

        medians : dict
            Each key is a vertex (Point) and each value is the median (Segment)
            at that point.

        See Also
        ========

        sympy.geometry.point.Point.midpoint, sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medians[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        return {v[0]: Segment(v[0], s[1].midpoint),
                v[1]: Segment(v[1], s[2].midpoint),
                v[2]: Segment(v[2], s[0].midpoint)}

    @property
    def medial(self):
        """The medial triangle of the triangle.

        The triangle which is formed from the midpoints of the three sides.

        Returns
        =======

        medial : Triangle

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medial
        Triangle(Point2D(1/2, 0), Point2D(1/2, 1/2), Point2D(0, 1/2))

        """
        s = self.sides
        return Triangle(s[0].midpoint, s[1].midpoint, s[2].midpoint)

    @property
    def nine_point_circle(self):
        """The nine-point circle of the triangle.

        Nine-point circle is the circumcircle of the medial triangle, which
        passes through the feet of altitudes and the middle points of segments
        connecting the vertices and the orthocenter.

        Returns
        =======

        nine_point_circle : Circle

        See also
        ========

        sympy.geometry.line.Segment.midpoint
        sympy.geometry.polygon.Triangle.medial
        sympy.geometry.polygon.Triangle.orthocenter

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.nine_point_circle
        Circle(Point2D(1/4, 1/4), sqrt(2)/4)

        """
        return Circle(*self.medial.vertices)

    @property
    def eulerline(self):
        """The Euler line of the triangle.

        The line which passes through circumcenter, centroid and orthocenter.

        Returns
        =======

        eulerline : Line (or Point for equilateral triangles in which case all
                    centers coincide)

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.eulerline
        Line2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        if self.is_equilateral():
            return self.orthocenter
        return Line(self.orthocenter, self.circumcenter)

def rad(d):
    """Return the radian value for the given degrees (pi = 180 degrees)."""
    return d*pi/180


def deg(r):
    """Return the degree value for the given radians (pi = 180 degrees)."""
    return r/pi*180


def _slope(d):
    rv = tan(rad(d))
    return rv


def _asa(d1, l, d2):
    """Return triangle having side with length l on the x-axis."""
    xy = Line((0, 0), slope=_slope(d1)).intersection(
        Line((l, 0), slope=_slope(180 - d2)))[0]
    return Triangle((0, 0), (l, 0), xy)


def _sss(l1, l2, l3):
    """Return triangle having side of length l1 on the x-axis."""
    c1 = Circle((0, 0), l3)
    c2 = Circle((l1, 0), l2)
    inter = [a for a in c1.intersection(c2) if a.y.is_nonnegative]
    if not inter:
        return None
    pt = inter[0]
    return Triangle((0, 0), (l1, 0), pt)


def _sas(l1, d, l2):
    """Return triangle having side with length l2 on the x-axis."""
    p1 = Point(0, 0)
    p2 = Point(l2, 0)
    p3 = Point(cos(rad(d))*l1, sin(rad(d))*l1)
    return Triangle(p1, p2, p3)
<|endoftext|>"""
A geometry module for the SymPy library. This module contains all of the
entities and functions needed to construct basic geometrical data and to
perform simple informational queries.

Usage:
======

Examples
========

"""
from sympy.geometry.point import Point, Point2D, Point3D
from sympy.geometry.line import Line, Ray, Segment, Line2D, Segment2D, Ray2D, \
    Line3D, Segment3D, Ray3D
from sympy.geometry.plane import Plane
from sympy.geometry.ellipse import Ellipse, Circle
from sympy.geometry.polygon import Polygon, RegularPolygon, Triangle, rad, deg
from sympy.geometry.util import are_similar, centroid, convex_hull, idiff, \
    intersection, closest_points, farthest_points
from sympy.geometry.exceptions import GeometryError
from sympy.geometry.curve import Curve
from sympy.geometry.parabola import Parabola

__all__ = [
    'Point', 'Point2D', 'Point3D',

    'Line', 'Ray', 'Segment', 'Line2D', 'Segment2D', 'Ray2D', 'Line3D',
    'Segment3D', 'Ray3D',

    'Plane',

    'Ellipse', 'Circle',

    'Polygon', 'RegularPolygon', 'Triangle', 'rad', 'deg',

    'are_similar', 'centroid', 'convex_hull', 'idiff', 'intersection',
    'closest_points', 'farthest_points',

    'GeometryError',

    'Curve',

    'Parabola',
]
<|endoftext|>"""Parabolic geometrical entity.

Contains
* Parabola

"""

from sympy.core import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, symbols
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point, Point2D
from sympy.geometry.line import Line, Line2D, Ray2D, Segment2D, LinearEntity3D
from sympy.geometry.ellipse import Ellipse
from sympy.functions import sign
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve


class Parabola(GeometrySet):
    """A parabolic GeometryEntity.

    A parabola is declared with a point, that is called 'focus', and
    a line, that is called 'directrix'.
    Only vertical or horizontal parabolas are currently supported.

    Parameters
    ==========

    focus : Point
        Default value is Point(0, 0)
    directrix : Line

    Attributes
    ==========

    focus
    directrix
    axis of symmetry
    focal length
    p parameter
    vertex
    eccentricity

    Raises
    ======
    ValueError
        When `focus` is not a two dimensional point.
        When `focus` is a point of directrix.
    NotImplementedError
        When `directrix` is neither horizontal nor vertical.

    Examples
    ========

    >>> from sympy import Parabola, Point, Line
    >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7,8)))
    >>> p1.focus
    Point2D(0, 0)
    >>> p1.directrix
    Line2D(Point2D(5, 8), Point2D(7, 8))

    """

    def __new__(cls, focus=None, directrix=None, **kwargs):

        if focus:
            focus = Point(focus, dim=2)
        else:
            focus = Point(0, 0)

        directrix = Line(directrix)

        if directrix.contains(focus):
            raise ValueError('The focus must not be a point of directrix')

        return GeometryEntity.__new__(cls, focus, directrix, **kwargs)

    @property
    def ambient_dimension(self):
        """Returns the ambient dimension of parabola.

        Returns
        =======

        ambient_dimension : integer

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> f1 = Point(0, 0)
        >>> p1 = Parabola(f1, Line(Point(5, 8), Point(7, 8)))
        >>> p1.ambient_dimension
        2

        """
        return 2

    @property
    def axis_of_symmetry(self):
        """Return the axis of symmetry of the parabola: a line
        perpendicular to the directrix passing through the focus.

        Returns
        =======

        axis_of_symmetry : Line

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.axis_of_symmetry
        Line2D(Point2D(0, 0), Point2D(0, 1))

        """
        return self.directrix.perpendicular_line(self.focus)

    @property
    def directrix(self):
        """The directrix of the parabola.

        Returns
        =======

        directrix : Line

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> l1 = Line(Point(5, 8), Point(7, 8))
        >>> p1 = Parabola(Point(0, 0), l1)
        >>> p1.directrix
        Line2D(Point2D(5, 8), Point2D(7, 8))

        """
        return self.args[1]

    @property
    def eccentricity(self):
        """The eccentricity of the parabola.

        Returns
        =======

        eccentricity : number

        A parabola may also be characterized as a conic section with an
        eccentricity of 1. As a consequence of this, all parabolas are
        similar, meaning that while they can be different sizes,
        they are all the same shape.

        See Also
        ========

        https://en.wikipedia.org/wiki/Parabola


        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.eccentricity
        1

        Notes
        -----
        The eccentricity for every Parabola is 1 by definition.

        """
        return S.One

    def equation(self, x='x', y='y'):
        """The equation of the parabola.

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

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.equation()
        -x**2 - 16*y + 64
        >>> p1.equation('f')
        -f**2 - 16*y + 64
        >>> p1.equation(y='z')
        -x**2 - 16*z + 64

        """
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)

        m = self.directrix.slope
        if m is S.Infinity:
            t1 = 4 * (self.p_parameter) * (x - self.vertex.x)
            t2 = (y - self.vertex.y)**2
        elif m == 0:
            t1 = 4 * (self.p_parameter) * (y - self.vertex.y)
            t2 = (x - self.vertex.x)**2
        else:
            a, b = self.focus
            c, d = self.directrix.coefficients[:2]
            t1 = (x - a)**2 + (y - b)**2
            t2 = self.directrix.equation(x, y)**2/(c**2 + d**2)
        return t1 - t2

    @property
    def focal_length(self):
        """The focal length of the parabola.

        Returns
        =======

        focal_lenght : number or symbolic expression

        Notes
        =====

        The distance between the vertex and the focus
        (or the vertex and directrix), measured along the axis
        of symmetry, is the "focal length".

        See Also
        ========

        https://en.wikipedia.org/wiki/Parabola

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.focal_length
        4

        """
        distance = self.directrix.distance(self.focus)
        focal_length = distance/2

        return focal_length

    @property
    def focus(self):
        """The focus of the parabola.

        Returns
        =======

        focus : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> f1 = Point(0, 0)
        >>> p1 = Parabola(f1, Line(Point(5, 8), Point(7, 8)))
        >>> p1.focus
        Point2D(0, 0)

        """
        return self.args[0]

    def intersection(self, o):
        """The intersection of the parabola and another geometrical entity `o`.

        Parameters
        ==========

        o : GeometryEntity, LinearEntity

        Returns
        =======

        intersection : list of GeometryEntity objects

        Examples
        ========

        >>> from sympy import Parabola, Point, Ellipse, Line, Segment
        >>> p1 = Point(0,0)
        >>> l1 = Line(Point(1, -2), Point(-1,-2))
        >>> parabola1 = Parabola(p1, l1)
        >>> parabola1.intersection(Ellipse(Point(0, 0), 2, 5))
        [Point2D(-2, 0), Point2D(2, 0)]
        >>> parabola1.intersection(Line(Point(-7, 3), Point(12, 3)))
        [Point2D(-4, 3), Point2D(4, 3)]
        >>> parabola1.intersection(Segment((-12, -65), (14, -68)))
        []

        """
        x, y = symbols('x y', real=True)
        parabola_eq = self.equation()
        if isinstance(o, Parabola):
            if o in self:
                return [o]
            else:
                return list(ordered([Point(i) for i in solve(
                    [parabola_eq, o.equation()], [x, y], set=True)[1]]))
        elif isinstance(o, Point2D):
            if simplify(parabola_eq.subs([(x, o._args[0]), (y, o._args[1])])) == 0:
                return [o]
            else:
                return []
        elif isinstance(o, (Segment2D, Ray2D)):
            result = solve([parabola_eq,
                Line2D(o.points[0], o.points[1]).equation()],
                [x, y], set=True)[1]
            return list(ordered([Point2D(i) for i in result if i in o]))
        elif isinstance(o, (Line2D, Ellipse)):
            return list(ordered([Point2D(i) for i in solve(
                [parabola_eq, o.equation()], [x, y], set=True)[1]]))
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Wrong type of argument were put')

    @property
    def p_parameter(self):
        """P is a parameter of parabola.

        Returns
        =======

        p : number or symbolic expression

        Notes
        =====

        The absolute value of p is the focal length. The sign on p tells
        which way the parabola faces. Vertical parabolas that open up
        and horizontal that open right, give a positive value for p.
        Vertical parabolas that open down and horizontal that open left,
        give a negative value for p.


        See Also
        ========

        https://www.sparknotes.com/math/precalc/conicsections/section2/

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.p_parameter
        -4

        """
        m = self.directrix.slope
        if m is S.Infinity:
            x = self.directrix.coefficients[2]
            p = sign(self.focus.args[0] + x)
        elif m == 0:
            y = self.directrix.coefficients[2]
            p = sign(self.focus.args[1] + y)
        else:
            d = self.directrix.projection(self.focus)
            p = sign(self.focus.x - d.x)
        return p * self.focal_length

    @property
    def vertex(self):
        """The vertex of the parabola.

        Returns
        =======

        vertex : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Parabola, Point, Line
        >>> p1 = Parabola(Point(0, 0), Line(Point(5, 8), Point(7, 8)))
        >>> p1.vertex
        Point2D(0, 4)

        """
        focus = self.focus
        m = self.directrix.slope
        if m is S.Infinity:
            vertex = Point(focus.args[0] - self.p_parameter, focus.args[1])
        elif m == 0:
            vertex = Point(focus.args[0], focus.args[1] - self.p_parameter)
        else:
            vertex = self.axis_of_symmetry.intersection(self)[0]
        return vertex
<|endoftext|>"""Curves in 2-dimensional Euclidean space.

Contains
========
Curve

"""

from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import diff
from sympy.core.containers import Tuple
from sympy.core.symbol import _symbol
from sympy.geometry.entity import GeometryEntity, GeometrySet
from sympy.geometry.point import Point
from sympy.integrals import integrate
from sympy.matrices import Matrix, rot_axis3
from sympy.utilities.iterables import is_sequence

from mpmath.libmp.libmpf import prec_to_dps


class Curve(GeometrySet):
    """A curve in space.

    A curve is defined by parametric functions for the coordinates, a
    parameter and the lower and upper bounds for the parameter value.

    Parameters
    ==========

    function : list of functions
    limits : 3-tuple
        Function parameter and lower and upper bounds.

    Attributes
    ==========

    functions
    parameter
    limits

    Raises
    ======

    ValueError
        When `functions` are specified incorrectly.
        When `limits` are specified incorrectly.

    Examples
    ========

    >>> from sympy import Curve, sin, cos, interpolate
    >>> from sympy.abc import t, a
    >>> C = Curve((sin(t), cos(t)), (t, 0, 2))
    >>> C.functions
    (sin(t), cos(t))
    >>> C.limits
    (t, 0, 2)
    >>> C.parameter
    t
    >>> C = Curve((t, interpolate([1, 4, 9, 16], t)), (t, 0, 1)); C
    Curve((t, t**2), (t, 0, 1))
    >>> C.subs(t, 4)
    Point2D(4, 16)
    >>> C.arbitrary_point(a)
    Point2D(a, a**2)

    See Also
    ========

    sympy.core.function.Function
    sympy.polys.polyfuncs.interpolate

    """

    def __new__(cls, function, limits):
        if not is_sequence(function) or len(function) != 2:
            raise ValueError("Function argument should be (x(t), y(t)) "
                "but got %s" % str(function))
        if not is_sequence(limits) or len(limits) != 3:
            raise ValueError("Limit argument should be (t, tmin, tmax) "
                "but got %s" % str(limits))

        return GeometryEntity.__new__(cls, Tuple(*function), Tuple(*limits))

    def __call__(self, f):
        return self.subs(self.parameter, f)

    def _eval_subs(self, old, new):
        if old == self.parameter:
            return Point(*[f.subs(old, new) for f in self.functions])

    def _eval_evalf(self, prec=15, **options):
        f, (t, a, b) = self.args
        dps = prec_to_dps(prec)
        f = tuple([i.evalf(n=dps, **options) for i in f])
        a, b = [i.evalf(n=dps, **options) for i in (a, b)]
        return self.func(f, (t, a, b))

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the curve.

        Parameters
        ==========

        parameter : str or Symbol, optional
            Default value is 't'.
            The Curve's parameter is selected with None or self.parameter
            otherwise the provided symbol is used.

        Returns
        =======

        Point :
            Returns a point in parametric form.

        Raises
        ======

        ValueError
            When `parameter` already appears in the functions.

        Examples
        ========

        >>> from sympy import Curve, Symbol
        >>> from sympy.abc import s
        >>> C = Curve([2*s, s**2], (s, 0, 2))
        >>> C.arbitrary_point()
        Point2D(2*t, t**2)
        >>> C.arbitrary_point(C.parameter)
        Point2D(2*s, s**2)
        >>> C.arbitrary_point(None)
        Point2D(2*s, s**2)
        >>> C.arbitrary_point(Symbol('a'))
        Point2D(2*a, a**2)

        See Also
        ========

        sympy.geometry.point.Point

        """
        if parameter is None:
            return Point(*self.functions)

        tnew = _symbol(parameter, self.parameter, real=True)
        t = self.parameter
        if (tnew.name != t.name and
                tnew.name in (f.name for f in self.free_symbols)):
            raise ValueError('Symbol %s already appears in object '
                'and cannot be used as a parameter.' % tnew.name)
        return Point(*[w.subs(t, tnew) for w in self.functions])

    @property
    def free_symbols(self):
        """Return a set of symbols other than the bound symbols used to
        parametrically define the Curve.

        Returns
        =======

        set :
            Set of all non-parameterized symbols.

        Examples
        ========

        >>> from sympy.abc import t, a
        >>> from sympy import Curve
        >>> Curve((t, t**2), (t, 0, 2)).free_symbols
        set()
        >>> Curve((t, t**2), (t, a, 2)).free_symbols
        {a}

        """
        free = set()
        for a in self.functions + self.limits[1:]:
            free |= a.free_symbols
        free = free.difference({self.parameter})
        return free

    @property
    def ambient_dimension(self):
        """The dimension of the curve.

        Returns
        =======

        int :
            the dimension of curve.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve((t, t**2), (t, 0, 2))
        >>> C.ambient_dimension
        2

        """

        return len(self.args[0])

    @property
    def functions(self):
        """The functions specifying the curve.

        Returns
        =======

        functions :
            list of parameterized coordinate functions.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve((t, t**2), (t, 0, 2))
        >>> C.functions
        (t, t**2)

        See Also
        ========

        parameter

        """
        return self.args[0]

    @property
    def limits(self):
        """The limits for the curve.

        Returns
        =======

        limits : tuple
            Contains parameter and lower and upper limits.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve([t, t**3], (t, -2, 2))
        >>> C.limits
        (t, -2, 2)

        See Also
        ========

        plot_interval

        """
        return self.args[1]

    @property
    def parameter(self):
        """The curve function variable.

        Returns
        =======

        Symbol :
            returns a bound symbol.

        Examples
        ========

        >>> from sympy.abc import t
        >>> from sympy import Curve
        >>> C = Curve([t, t**2], (t, 0, 2))
        >>> C.parameter
        t

        See Also
        ========

        functions

        """
        return self.args[1][0]

    @property
    def length(self):
        """The curve length.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import t
        >>> Curve((t, t), (t, 0, 1)).length
        sqrt(2)

        """
        integrand = sqrt(sum(diff(func, self.limits[0])**2 for func in self.functions))
        return integrate(integrand, self.limits)

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the curve.

        Parameters
        ==========

        parameter : str or Symbol, optional
            Default value is 't';
            otherwise the provided symbol is used.

        Returns
        =======

        List :
            the plot interval as below:
                [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Curve, sin
        >>> from sympy.abc import x, s
        >>> Curve((x, sin(x)), (x, 1, 2)).plot_interval()
        [t, 1, 2]
        >>> Curve((x, sin(x)), (x, 1, 2)).plot_interval(s)
        [s, 1, 2]

        See Also
        ========

        limits : Returns limits of the parameter interval

        """
        t = _symbol(parameter, self.parameter, real=True)
        return [t] + list(self.limits[1:])

    def rotate(self, angle=0, pt=None):
        """This function is used to rotate a curve along given point ``pt`` at given angle(in radian).

        Parameters
        ==========

        angle :
            the angle at which the curve will be rotated(in radian) in counterclockwise direction.
            default value of angle is 0.

        pt : Point
            the point along which the curve will be rotated.
            If no point given, the curve will be rotated around origin.

        Returns
        =======

        Curve :
            returns a curve rotated at given angle along given point.

        Examples
        ========

        >>> from sympy import Curve, pi
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).rotate(pi/2)
        Curve((-x, x), (x, 0, 1))

        """
        if pt:
            pt = -Point(pt, dim=2)
        else:
            pt = Point(0,0)
        rv = self.translate(*pt.args)
        f = list(rv.functions)
        f.append(0)
        f = Matrix(1, 3, f)
        f *= rot_axis3(angle)
        rv = self.func(f[0, :2].tolist()[0], self.limits)
        pt = -pt
        return rv.translate(*pt.args)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since Curve is not made up of Points.

        Returns
        =======

        Curve :
            returns scaled curve.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).scale(2)
        Curve((2*x, x), (x, 0, 1))

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        fx, fy = self.functions
        return self.func((fx*x, fy*y), self.limits)

    def translate(self, x=0, y=0):
        """Translate the Curve by (x, y).

        Returns
        =======

        Curve :
            returns a translated curve.

        Examples
        ========

        >>> from sympy import Curve
        >>> from sympy.abc import x
        >>> Curve((x, x), (x, 0, 1)).translate(1, 2)
        Curve((x + 1, x + 2), (x, 0, 1))

        """
        fx, fy = self.functions
        return self.func((fx + x, fy + y), self.limits)
<|endoftext|>from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable

import sympy
from sympy import Mul, Add, Pow, Rational, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
    acosh, atanh, acoth, asech, acsch, expand, im, flatten, polylog, cancel, expand_trig, sign, simplify, \
    UnevaluatedExpr, S, atan, atan2, Mod, Max, Min, rf, Ei, Si, Ci, airyai, airyaiprime, airybi, primepi, prime, \
    isprime, cot, sec, csc, csch, sech, coth, Function, I, pi, Tuple, GreaterThan, StrictGreaterThan, StrictLessThan, \
    LessThan, Equality, Or, And, Lambda, Integer, Dummy, symbols
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning


def mathematica(s, additional_translations=None):
    sympy_deprecation_warning(
        """The ``mathematica`` function for the Mathematica parser is now
deprecated. Use ``parse_mathematica`` instead.
The parameter ``additional_translation`` can be replaced by SymPy's
.replace( ) or .subs( ) methods on the output expression instead.""",
        deprecated_since_version="1.11",
        active_deprecations_target="mathematica-parser-new",
    )
    parser = MathematicaParser(additional_translations)
    return sympify(parser._parse_old(s))


def parse_mathematica(s):
    """
    Translate a string containing a Wolfram Mathematica expression to a SymPy
    expression.

    If the translator is unable to find a suitable SymPy expression, the
    ``FullForm`` of the Mathematica expression will be output, using SymPy
    ``Function`` objects as nodes of the syntax tree.

    Examples
    ========

    >>> from sympy.parsing.mathematica import parse_mathematica
    >>> parse_mathematica("Sin[x]^2 Tan[y]")
    sin(x)**2*tan(y)
    >>> e = parse_mathematica("F[7,5,3]")
    >>> e
    F(7, 5, 3)
    >>> from sympy import Function, Max, Min
    >>> e.replace(Function("F"), lambda *x: Max(*x)*Min(*x))
    21

    Both standard input form and Mathematica full form are supported:

    >>> parse_mathematica("x*(a + b)")
    x*(a + b)
    >>> parse_mathematica("Times[x, Plus[a, b]]")
    x*(a + b)

    To get a matrix from Wolfram's code:

    >>> m = parse_mathematica("{{a, b}, {c, d}}")
    >>> m
    ((a, b), (c, d))
    >>> from sympy import Matrix
    >>> Matrix(m)
    Matrix([
    [a, b],
    [c, d]])

    If the translation into equivalent SymPy expressions fails, an SymPy
    expression equivalent to Wolfram Mathematica's "FullForm" will be created:

    >>> parse_mathematica("x_.")
    Optional(Pattern(x, Blank()))
    >>> parse_mathematica("Plus @@ {x, y, z}")
    Apply(Plus, (x, y, z))
    >>> parse_mathematica("f[x_, 3] := x^3 /; x > 0")
    SetDelayed(f(Pattern(x, Blank()), 3), Condition(x**3, x > 0))
    """
    parser = MathematicaParser()
    return parser.parse(s)


def _parse_Function(*args):
    if len(args) == 1:
        arg = args[0]
        Slot = Function("Slot")
        slots = arg.atoms(Slot)
        numbers = [a.args[0] for a in slots]
        number_of_arguments = max(numbers)
        if isinstance(number_of_arguments, Integer):
            variables = symbols(f"dummy0:{number_of_arguments}", cls=Dummy)
            return Lambda(variables, arg.xreplace({Slot(i+1): v for i, v in enumerate(variables)}))
        return Lambda((), arg)
    elif len(args) == 2:
        variables = args[0]
        body = args[1]
        return Lambda(variables, body)
    else:
        raise SyntaxError("Function node expects 1 or 2 arguments")


def _deco(cls):
    cls._initialize_class()
    return cls


@_deco
class MathematicaParser:
    """
    An instance of this class converts a string of a Wolfram Mathematica
    expression to a SymPy expression.

    The main parser acts internally in three stages:

    1. tokenizer: tokenizes the Mathematica expression and adds the missing *
        operators. Handled by ``_from_mathematica_to_tokens(...)``
    2. full form list: sort the list of strings output by the tokenizer into a
        syntax tree of nested lists and strings, equivalent to Mathematica's
        ``FullForm`` expression output. This is handled by the function
        ``_from_tokens_to_fullformlist(...)``.
    3. SymPy expression: the syntax tree expressed as full form list is visited
        and the nodes with equivalent classes in SymPy are replaced. Unknown
        syntax tree nodes are cast to SymPy ``Function`` objects. This is
        handled by ``_from_fullformlist_to_sympy(...)``.

    """

    # left: Mathematica, right: SymPy
    CORRESPONDENCES = {
        'Sqrt[x]': 'sqrt(x)',
        'Rational[x,y]': 'Rational(x,y)',
        'Exp[x]': 'exp(x)',
        'Log[x]': 'log(x)',
        'Log[x,y]': 'log(y,x)',
        'Log2[x]': 'log(x,2)',
        'Log10[x]': 'log(x,10)',
        'Mod[x,y]': 'Mod(x,y)',
        'Max[*x]': 'Max(*x)',
        'Min[*x]': 'Min(*x)',
        'Pochhammer[x,y]':'rf(x,y)',
        'ArcTan[x,y]':'atan2(y,x)',
        'ExpIntegralEi[x]': 'Ei(x)',
        'SinIntegral[x]': 'Si(x)',
        'CosIntegral[x]': 'Ci(x)',
        'AiryAi[x]': 'airyai(x)',
        'AiryAiPrime[x]': 'airyaiprime(x)',
        'AiryBi[x]' :'airybi(x)',
        'AiryBiPrime[x]' :'airybiprime(x)',
        'LogIntegral[x]':' li(x)',
        'PrimePi[x]': 'primepi(x)',
        'Prime[x]': 'prime(x)',
        'PrimeQ[x]': 'isprime(x)'
    }

    # trigonometric, e.t.c.
    for arc, tri, h in product(('', 'Arc'), (
            'Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc'), ('', 'h')):
        fm = arc + tri + h + '[x]'
        if arc:  # arc func
            fs = 'a' + tri.lower() + h + '(x)'
        else:    # non-arc func
            fs = tri.lower() + h + '(x)'
        CORRESPONDENCES.update({fm: fs})

    REPLACEMENTS = {
        ' ': '',
        '^': '**',
        '{': '[',
        '}': ']',
    }

    RULES = {
        # a single whitespace to '*'
        'whitespace': (
            re.compile(r'''
                (?:(?<=[a-zA-Z\d])|(?<=\d\.))     # a letter or a number
                \s+                               # any number of whitespaces
                (?:(?=[a-zA-Z\d])|(?=\.\d))       # a letter or a number
                ''', re.VERBOSE),
            '*'),

        # add omitted '*' character
        'add*_1': (
            re.compile(r'''
                (?:(?<=[])\d])|(?<=\d\.))       # ], ) or a number
                                                # ''
                (?=[(a-zA-Z])                   # ( or a single letter
                ''', re.VERBOSE),
            '*'),

        # add omitted '*' character (variable letter preceding)
        'add*_2': (
            re.compile(r'''
                (?<=[a-zA-Z])       # a letter
                \(                  # ( as a character
                (?=.)               # any characters
                ''', re.VERBOSE),
            '*('),

        # convert 'Pi' to 'pi'
        'Pi': (
            re.compile(r'''
                (?:
                \A|(?<=[^a-zA-Z])
                )
                Pi                  # 'Pi' is 3.14159... in Mathematica
                (?=[^a-zA-Z])
                ''', re.VERBOSE),
            'pi'),
    }

    # Mathematica function name pattern
    FM_PATTERN = re.compile(r'''
                (?:
                \A|(?<=[^a-zA-Z])   # at the top or a non-letter
                )
                [A-Z][a-zA-Z\d]*    # Function
                (?=\[)              # [ as a character
                ''', re.VERBOSE)

    # list or matrix pattern (for future usage)
    ARG_MTRX_PATTERN = re.compile(r'''
                \{.*\}
                ''', re.VERBOSE)

    # regex string for function argument pattern
    ARGS_PATTERN_TEMPLATE = r'''
                (?:
                \A|(?<=[^a-zA-Z])
                )
                {arguments}         # model argument like x, y,...
                (?=[^a-zA-Z])
                '''

    # will contain transformed CORRESPONDENCES dictionary
    TRANSLATIONS: dict[tuple[str, int], dict[str, Any]] = {}

    # cache for a raw users' translation dictionary
    cache_original: dict[tuple[str, int], dict[str, Any]] = {}

    # cache for a compiled users' translation dictionary
    cache_compiled: dict[tuple[str, int], dict[str, Any]] = {}

    @classmethod
    def _initialize_class(cls):
        # get a transformed CORRESPONDENCES dictionary
        d = cls._compile_dictionary(cls.CORRESPONDENCES)
        cls.TRANSLATIONS.update(d)

    def __init__(self, additional_translations=None):
        self.translations = {}

        # update with TRANSLATIONS (class constant)
        self.translations.update(self.TRANSLATIONS)

        if additional_translations is None:
            additional_translations = {}

        # check the latest added translations
        if self.__class__.cache_original != additional_translations:
            if not isinstance(additional_translations, dict):
                raise ValueError('The argument must be dict type')

            # get a transformed additional_translations dictionary
            d = self._compile_dictionary(additional_translations)

            # update cache
            self.__class__.cache_original = additional_translations
            self.__class__.cache_compiled = d

        # merge user's own translations
        self.translations.update(self.__class__.cache_compiled)

    @classmethod
    def _compile_dictionary(cls, dic):
        # for return
        d = {}

        for fm, fs in dic.items():
            # check function form
            cls._check_input(fm)
            cls._check_input(fs)

            # uncover '*' hiding behind a whitespace
            fm = cls._apply_rules(fm, 'whitespace')
            fs = cls._apply_rules(fs, 'whitespace')

            # remove whitespace(s)
            fm = cls._replace(fm, ' ')
            fs = cls._replace(fs, ' ')

            # search Mathematica function name
            m = cls.FM_PATTERN.search(fm)

            # if no-hit
            if m is None:
                err = "'{f}' function form is invalid.".format(f=fm)
                raise ValueError(err)

            # get Mathematica function name like 'Log'
            fm_name = m.group()

            # get arguments of Mathematica function
            args, end = cls._get_args(m)

            # function side check. (e.g.) '2*Func[x]' is invalid.
            if m.start() != 0 or end != len(fm):
                err = "'{f}' function form is invalid.".format(f=fm)
                raise ValueError(err)

            # check the last argument's 1st character
            if args[-1][0] == '*':
                key_arg = '*'
            else:
                key_arg = len(args)

            key = (fm_name, key_arg)

            # convert '*x' to '\\*x' for regex
            re_args = [x if x[0] != '*' else '\\' + x for x in args]

            # for regex. Example: (?:(x|y|z))
            xyz = '(?:(' + '|'.join(re_args) + '))'

            # string for regex compile
            patStr = cls.ARGS_PATTERN_TEMPLATE.format(arguments=xyz)

            pat = re.compile(patStr, re.VERBOSE)

            # update dictionary
            d[key] = {}
            d[key]['fs'] = fs  # SymPy function template
            d[key]['args'] = args  # args are ['x', 'y'] for example
            d[key]['pat'] = pat

        return d

    def _convert_function(self, s):
        '''Parse Mathematica function to SymPy one'''

        # compiled regex object
        pat = self.FM_PATTERN

        scanned = ''                # converted string
        cur = 0                     # position cursor
        while True:
            m = pat.search(s)

            if m is None:
                # append the rest of string
                scanned += s
                break

            # get Mathematica function name
            fm = m.group()

            # get arguments, and the end position of fm function
            args, end = self._get_args(m)

            # the start position of fm function
            bgn = m.start()

            # convert Mathematica function to SymPy one
            s = self._convert_one_function(s, fm, args, bgn, end)

            # update cursor
            cur = bgn

            # append converted part
            scanned += s[:cur]

            # shrink s
            s = s[cur:]

        return scanned

    def _convert_one_function(self, s, fm, args, bgn, end):
        # no variable-length argument
        if (fm, len(args)) in self.translations:
            key = (fm, len(args))

            # x, y,... model arguments
            x_args = self.translations[key]['args']

            # make CORRESPONDENCES between model arguments and actual ones
            d = dict(zip(x_args, args))

        # with variable-length argument
        elif (fm, '*') in self.translations:
            key = (fm, '*')

            # x, y,..*args (model arguments)
            x_args = self.translations[key]['args']

            # make CORRESPONDENCES between model arguments and actual ones
            d = {}
            for i, x in enumerate(x_args):
                if x[0] == '*':
                    d[x] = ','.join(args[i:])
                    break
                d[x] = args[i]

        # out of self.translations
        else:
            err = "'{f}' is out of the whitelist.".format(f=fm)
            raise ValueError(err)

        # template string of converted function
        template = self.translations[key]['fs']

        # regex pattern for x_args
        pat = self.translations[key]['pat']

        scanned = ''
        cur = 0
        while True:
            m = pat.search(template)

            if m is None:
                scanned += template
                break

            # get model argument
            x = m.group()

            # get a start position of the model argument
            xbgn = m.start()

            # add the corresponding actual argument
            scanned += template[:xbgn] + d[x]

            # update cursor to the end of the model argument
            cur = m.end()

            # shrink template
            template = template[cur:]

        # update to swapped string
        s = s[:bgn] + scanned + s[end:]

        return s

    @classmethod
    def _get_args(cls, m):
        '''Get arguments of a Mathematica function'''

        s = m.string                # whole string
        anc = m.end() + 1           # pointing the first letter of arguments
        square, curly = [], []      # stack for brackets
        args = []

        # current cursor
        cur = anc
        for i, c in enumerate(s[anc:], anc):
            # extract one argument
            if c == ',' and (not square) and (not curly):
                args.append(s[cur:i])       # add an argument
                cur = i + 1                 # move cursor

            # handle list or matrix (for future usage)
            if c == '{':
                curly.append(c)
            elif c == '}':
                curly.pop()

            # seek corresponding ']' with skipping irrevant ones
            if c == '[':
                square.append(c)
            elif c == ']':
                if square:
                    square.pop()
                else:   # empty stack
                    args.append(s[cur:i])
                    break

        # the next position to ']' bracket (the function end)
        func_end = i + 1

        return args, func_end

    @classmethod
    def _replace(cls, s, bef):
        aft = cls.REPLACEMENTS[bef]
        s = s.replace(bef, aft)
        return s

    @classmethod
    def _apply_rules(cls, s, bef):
        pat, aft = cls.RULES[bef]
        return pat.sub(aft, s)

    @classmethod
    def _check_input(cls, s):
        for bracket in (('[', ']'), ('{', '}'), ('(', ')')):
            if s.count(bracket[0]) != s.count(bracket[1]):
                err = "'{f}' function form is invalid.".format(f=s)
                raise ValueError(err)

        if '{' in s:
            err = "Currently list is not supported."
            raise ValueError(err)

    def _parse_old(self, s):
        # input check
        self._check_input(s)

        # uncover '*' hiding behind a whitespace
        s = self._apply_rules(s, 'whitespace')

        # remove whitespace(s)
        s = self._replace(s, ' ')

        # add omitted '*' character
        s = self._apply_rules(s, 'add*_1')
        s = self._apply_rules(s, 'add*_2')

        # translate function
        s = self._convert_function(s)

        # '^' to '**'
        s = self._replace(s, '^')

        # 'Pi' to 'pi'
        s = self._apply_rules(s, 'Pi')

        # '{', '}' to '[', ']', respectively
#        s = cls._replace(s, '{')   # currently list is not taken into account
#        s = cls._replace(s, '}')

        return s

    def parse(self, s):
        s2 = self._from_mathematica_to_tokens(s)
        s3 = self._from_tokens_to_fullformlist(s2)
        s4 = self._from_fullformlist_to_sympy(s3)
        return s4

    INFIX = "Infix"
    PREFIX = "Prefix"
    POSTFIX = "Postfix"
    FLAT = "Flat"
    RIGHT = "Right"
    LEFT = "Left"

    _mathematica_op_precedence: list[tuple[str, str | None, dict[str, str | Callable]]] = [
        (POSTFIX, None, {";": lambda x: x + ["Null"] if isinstance(x, list) and x and x[0] == "CompoundExpression" else ["CompoundExpression", x, "Null"]}),
        (INFIX, FLAT, {";": "CompoundExpression"}),
        (INFIX, RIGHT, {"=": "Set", ":=": "SetDelayed", "+=": "AddTo", "-=": "SubtractFrom", "*=": "TimesBy", "/=": "DivideBy"}),
        (INFIX, LEFT, {"//": lambda x, y: [x, y]}),
        (POSTFIX, None, {"&": "Function"}),
        (INFIX, LEFT, {"/.": "ReplaceAll"}),
        (INFIX, RIGHT, {"->": "Rule", ":>": "RuleDelayed"}),
        (INFIX, LEFT, {"/;": "Condition"}),
        (INFIX, FLAT, {"|": "Alternatives"}),
        (POSTFIX, None, {"..": "Repeated", "...": "RepeatedNull"}),
        (INFIX, FLAT, {"||": "Or"}),
        (INFIX, FLAT, {"&&": "And"}),
        (PREFIX, None, {"!": "Not"}),
        (INFIX, FLAT, {"===": "SameQ", "=!=": "UnsameQ"}),
        (INFIX, FLAT, {"==": "Equal", "!=": "Unequal", "<=": "LessEqual", "<": "Less", ">=": "GreaterEqual", ">": "Greater"}),
        (INFIX, None, {";;": "Span"}),
        (INFIX, FLAT, {"+": "Plus", "-": "Plus"}),
        (INFIX, FLAT, {"*": "Times", "/": "Times"}),
        (INFIX, FLAT, {".": "Dot"}),
        (PREFIX, None, {"-": lambda x: MathematicaParser._get_neg(x),
                        "+": lambda x: x}),
        (INFIX, RIGHT, {"^": "Power"}),
        (INFIX, RIGHT, {"@@": "Apply", "/@": "Map", "//@": "MapAll", "@@@": lambda x, y: ["Apply", x, y, ["List", "1"]]}),
        (POSTFIX, None, {"'": "Derivative", "!": "Factorial", "!!": "Factorial2", "--": "Decrement"}),
        (INFIX, None, {"[": lambda x, y: [x, *y], "[[": lambda x, y: ["Part", x, *y]}),
        (PREFIX, None, {"{": lambda x: ["List", *x], "(": lambda x: x[0]}),
        (INFIX, None, {"?": "PatternTest"}),
        (POSTFIX, None, {
            "_": lambda x: ["Pattern", x, ["Blank"]],
            "_.": lambda x: ["Optional", ["Pattern", x, ["Blank"]]],
            "__": lambda x: ["Pattern", x, ["BlankSequence"]],
            "___": lambda x: ["Pattern", x, ["BlankNullSequence"]],
        }),
        (INFIX, None, {"_": lambda x, y: ["Pattern", x, ["Blank", y]]}),
        (PREFIX, None, {"#": "Slot", "##": "SlotSequence"}),
    ]

    _missing_arguments_default = {
        "#": lambda: ["Slot", "1"],
        "##": lambda: ["SlotSequence", "1"],
    }

    _literal = r"[A-Za-z][A-Za-z0-9]*"
    _number = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"

    _enclosure_open = ["(", "[", "[[", "{"]
    _enclosure_close = [")", "]", "]]", "}"]

    @classmethod
    def _get_neg(cls, x):
        return f"-{x}" if isinstance(x, str) and re.match(MathematicaParser._number, x) else ["Times", "-1", x]

    @classmethod
    def _get_inv(cls, x):
        return ["Power", x, "-1"]

    _regex_tokenizer = None

    def _get_tokenizer(self):
        if self._regex_tokenizer is not None:
            # Check if the regular expression has already been compiled:
            return self._regex_tokenizer
        tokens = [self._literal, self._number]
        tokens_escape = self._enclosure_open[:] + self._enclosure_close[:]
        for typ, strat, symdict in self._mathematica_op_precedence:
            for k in symdict:
                tokens_escape.append(k)
        tokens_escape.sort(key=lambda x: -len(x))
        tokens.extend(map(re.escape, tokens_escape))
        tokens.append(",")
        tokens.append("\n")
        tokenizer = re.compile("(" + "|".join(tokens) + ")")
        self._regex_tokenizer = tokenizer
        return self._regex_tokenizer

    def _from_mathematica_to_tokens(self, code: str):
        tokenizer = self._get_tokenizer()

        # Find strings:
        code_splits: list[str | list] = []
        while True:
            string_start = code.find("\"")
            if string_start == -1:
                if len(code) > 0:
                    code_splits.append(code)
                break
            match_end = re.search(r'(?<!\\)"', code[string_start+1:])
            if match_end is None:
                raise SyntaxError('mismatch in string "  " expression')
            string_end = string_start + match_end.start() + 1
            if string_start > 0:
                code_splits.append(code[:string_start])
            code_splits.append(["_Str", code[string_start+1:string_end].replace('\\"', '"')])
            code = code[string_end+1:]

        # Remove comments:
        for i, code_split in enumerate(code_splits):
            if isinstance(code_split, list):
                continue
            while True:
                pos_comment_start = code_split.find("(*")
                if pos_comment_start == -1:
                    break
                pos_comment_end = code_split.find("*)")
                if pos_comment_end == -1 or pos_comment_end < pos_comment_start:
                    raise SyntaxError("mismatch in comment (*  *) code")
                code_split = code_split[:pos_comment_start] + code_split[pos_comment_end+2:]
            code_splits[i] = code_split

        # Tokenize the input strings with a regular expression:
        token_lists = [tokenizer.findall(i) if isinstance(i, str) and i.isascii() else [i] for i in code_splits]
        tokens = [j for i in token_lists for j in i]

        # Remove newlines at the beginning
        while tokens and tokens[0] == "\n":
            tokens.pop(0)
        # Remove newlines at the end
        while tokens and tokens[-1] == "\n":
            tokens.pop(-1)

        return tokens

    def _is_op(self, token: str | list) -> bool:
        if isinstance(token, list):
            return False
        if re.match(self._literal, token):
            return False
        if re.match("-?" + self._number, token):
            return False
        return True

    def _is_valid_star1(self, token: str | list) -> bool:
        if token in (")", "}"):
            return True
        return not self._is_op(token)

    def _is_valid_star2(self, token: str | list) -> bool:
        if token in ("(", "{"):
            return True
        return not self._is_op(token)

    def _from_tokens_to_fullformlist(self, tokens: list):
        stack: list[list] = [[]]
        open_seq = []
        pointer: int = 0
        while pointer < len(tokens):
            token = tokens[pointer]
            if token in self._enclosure_open:
                stack[-1].append(token)
                open_seq.append(token)
                stack.append([])
            elif token == ",":
                if len(stack[-1]) == 0 and stack[-2][-1] == open_seq[-1]:
                    raise SyntaxError("%s cannot be followed by comma ," % open_seq[-1])
                stack[-1] = self._parse_after_braces(stack[-1])
                stack.append([])
            elif token in self._enclosure_close:
                ind = self._enclosure_close.index(token)
                if self._enclosure_open[ind] != open_seq[-1]:
                    unmatched_enclosure = SyntaxError("unmatched enclosure")
                    if token == "]]" and open_seq[-1] == "[":
                        if open_seq[-2] == "[":
                            # These two lines would be logically correct, but are
                            # unnecessary:
                            # token = "]"
                            # tokens[pointer] = "]"
                            tokens.insert(pointer+1, "]")
                        elif open_seq[-2] == "[[":
                            if tokens[pointer+1] == "]":
                                tokens[pointer+1] = "]]"
                            elif tokens[pointer+1] == "]]":
                                tokens[pointer+1] = "]]"
                                tokens.insert(pointer+2, "]")
                            else:
                                raise unmatched_enclosure
                    else:
                        raise unmatched_enclosure
                if len(stack[-1]) == 0 and stack[-2][-1] == "(":
                    raise SyntaxError("( ) not valid syntax")
                last_stack = self._parse_after_braces(stack[-1], True)
                stack[-1] = last_stack
                new_stack_element = []
                while stack[-1][-1] != open_seq[-1]:
                    new_stack_element.append(stack.pop())
                new_stack_element.reverse()
                if open_seq[-1] == "(" and len(new_stack_element) != 1:
                    raise SyntaxError("( must be followed by one expression, %i detected" % len(new_stack_element))
                stack[-1].append(new_stack_element)
                open_seq.pop(-1)
            else:
                stack[-1].append(token)
            pointer += 1
        if len(stack) != 1:
            raise RuntimeError("Stack should have only one element")
        return self._parse_after_braces(stack[0])

    def _util_remove_newlines(self, lines: list, tokens: list, inside_enclosure: bool):
        pointer = 0
        size = len(tokens)
        while pointer < size:
            token = tokens[pointer]
            if token == "\n":
                if inside_enclosure:
                    # Ignore newlines inside enclosures
                    tokens.pop(pointer)
                    size -= 1
                    continue
                if pointer == 0:
                    tokens.pop(0)
                    size -= 1
                    continue
                if pointer > 1:
                    try:
                        prev_expr = self._parse_after_braces(tokens[:pointer], inside_enclosure)
                    except SyntaxError:
                        tokens.pop(pointer)
                        size -= 1
                        continue
                else:
                    prev_expr = tokens[0]
                if len(prev_expr) > 0 and prev_expr[0] == "CompoundExpression":
                    lines.extend(prev_expr[1:])
                else:
                    lines.append(prev_expr)
                for i in range(pointer):
                    tokens.pop(0)
                size -= pointer
                pointer = 0
                continue
            pointer += 1

    def _util_add_missing_asterisks(self, tokens: list):
        size: int = len(tokens)
        pointer: int = 0
        while pointer < size:
            if (pointer > 0 and
                    self._is_valid_star1(tokens[pointer - 1]) and
                    self._is_valid_star2(tokens[pointer])):
                # This is a trick to add missing * operators in the expression,
                # `"*" in op_dict` makes sure the precedence level is the same as "*",
                # while `not self._is_op( ... )` makes sure this and the previous
                # expression are not operators.
                if tokens[pointer] == "(":
                    # ( has already been processed by now, replace:
                    tokens[pointer] = "*"
                    tokens[pointer + 1] = tokens[pointer + 1][0]
                else:
                    tokens.insert(pointer, "*")
                    pointer += 1
                    size += 1
            pointer += 1

    def _parse_after_braces(self, tokens: list, inside_enclosure: bool = False):
        op_dict: dict
        changed: bool = False
        lines: list = []

        self._util_remove_newlines(lines, tokens, inside_enclosure)

        for op_type, grouping_strat, op_dict in reversed(self._mathematica_op_precedence):
            if "*" in op_dict:
                self._util_add_missing_asterisks(tokens)
            size: int = len(tokens)
            pointer: int = 0
            while pointer < size:
                token = tokens[pointer]
                if isinstance(token, str) and token in op_dict:
                    op_name: str | Callable = op_dict[token]
                    node: list
                    first_index: int
                    if isinstance(op_name, str):
                        node = [op_name]
                        first_index = 1
                    else:
                        node = []
                        first_index = 0
                    if token in ("+", "-") and op_type == self.PREFIX and pointer > 0 and not self._is_op(tokens[pointer - 1]):
                        # Make sure that PREFIX + - don't match expressions like a + b or a - b,
                        # the INFIX + - are supposed to match that expression:
                        pointer += 1
                        continue
                    if op_type == self.INFIX:
                        if pointer == 0 or pointer == size - 1 or self._is_op(tokens[pointer - 1]) or self._is_op(tokens[pointer + 1]):
                            pointer += 1
                            continue
                    changed = True
                    tokens[pointer] = node
                    if op_type == self.INFIX:
                        arg1 = tokens.pop(pointer-1)
                        arg2 = tokens.pop(pointer)
                        if token == "/":
                            arg2 = self._get_inv(arg2)
                        elif token == "-":
                            arg2 = self._get_neg(arg2)
                        pointer -= 1
                        size -= 2
                        node.append(arg1)
                        node_p = node
                        if grouping_strat == self.FLAT:
                            while pointer + 2 < size and self._check_op_compatible(tokens[pointer+1], token):
                                node_p.append(arg2)
                                other_op = tokens.pop(pointer+1)
                                arg2 = tokens.pop(pointer+1)
                                if other_op == "/":
                                    arg2 = self._get_inv(arg2)
                                elif other_op == "-":
                                    arg2 = self._get_neg(arg2)
                                size -= 2
                            node_p.append(arg2)
                        elif grouping_strat == self.RIGHT:
                            while pointer + 2 < size and tokens[pointer+1] == token:
                                node_p.append([op_name, arg2])
                                node_p = node_p[-1]
                                tokens.pop(pointer+1)
                                arg2 = tokens.pop(pointer+1)
                                size -= 2
                            node_p.append(arg2)
                        elif grouping_strat == self.LEFT:
                            while pointer + 1 < size and tokens[pointer+1] == token:
                                if isinstance(op_name, str):
                                    node_p[first_index] = [op_name, node_p[first_index], arg2]
                                else:
                                    node_p[first_index] = op_name(node_p[first_index], arg2)
                                tokens.pop(pointer+1)
                                arg2 = tokens.pop(pointer+1)
                                size -= 2
                            node_p.append(arg2)
                        else:
                            node.append(arg2)
                    elif op_type == self.PREFIX:
                        if grouping_strat is not None:
                            raise TypeError("'Prefix' op_type should not have a grouping strat")
                        if pointer == size - 1 or self._is_op(tokens[pointer + 1]):
                            tokens[pointer] = self._missing_arguments_default[token]()
                        else:
                            node.append(tokens.pop(pointer+1))
                            size -= 1
                    elif op_type == self.POSTFIX:
                        if grouping_strat is not None:
                            raise TypeError("'Prefix' op_type should not have a grouping strat")
                        if pointer == 0 or self._is_op(tokens[pointer - 1]):
                            tokens[pointer] = self._missing_arguments_default[token]()
                        else:
                            node.append(tokens.pop(pointer-1))
                            pointer -= 1
                            size -= 1
                    if isinstance(op_name, Callable):  # type: ignore
                        op_call: Callable = typing.cast(Callable, op_name)
                        new_node = op_call(*node)
                        node.clear()
                        if isinstance(new_node, list):
                            node.extend(new_node)
                        else:
                            tokens[pointer] = new_node
                pointer += 1
        if len(tokens) > 1 or (len(lines) == 0 and len(tokens) == 0):
            if changed:
                # Trick to deal with cases in which an operator with lower
                # precedence should be transformed before an operator of higher
                # precedence. Such as in the case of `#&[x]` (that is
                # equivalent to `Lambda(d_, d_)(x)` in SymPy). In this case the
                # operator `&` has lower precedence than `[`, but needs to be
                # evaluated first because otherwise `# (&[x])` is not a valid
                # expression:
                return self._parse_after_braces(tokens, inside_enclosure)
            raise SyntaxError("unable to create a single AST for the expression")
        if len(lines) > 0:
            if tokens[0] and tokens[0][0] == "CompoundExpression":
                tokens = tokens[0][1:]
            compound_expression = ["CompoundExpression", *lines, *tokens]
            return compound_expression
        return tokens[0]

    def _check_op_compatible(self, op1: str, op2: str):
        if op1 == op2:
            return True
        muldiv = {"*", "/"}
        addsub = {"+", "-"}
        if op1 in muldiv and op2 in muldiv:
            return True
        if op1 in addsub and op2 in addsub:
            return True
        return False

    def _from_fullform_to_fullformlist(self, wmexpr: str):
        """
        Parses FullForm[Downvalues[]] generated by Mathematica
        """
        out: list = []
        stack = [out]
        generator = re.finditer(r'[\[\],]', wmexpr)
        last_pos = 0
        for match in generator:
            if match is None:
                break
            position = match.start()
            last_expr = wmexpr[last_pos:position].replace(',', '').replace(']', '').replace('[', '').strip()

            if match.group() == ',':
                if last_expr != '':
                    stack[-1].append(last_expr)
            elif match.group() == ']':
                if last_expr != '':
                    stack[-1].append(last_expr)
                stack.pop()
            elif match.group() == '[':
                stack[-1].append([last_expr])
                stack.append(stack[-1][-1])
            last_pos = match.end()
        return out[0]

    def _from_fullformlist_to_fullformsympy(self, pylist: list):
        from sympy import Function, Symbol

        def converter(expr):
            if isinstance(expr, list):
                if len(expr) > 0:
                    head = expr[0]
                    args = [converter(arg) for arg in expr[1:]]
                    return Function(head)(*args)
                else:
                    raise ValueError("Empty list of expressions")
            elif isinstance(expr, str):
                return Symbol(expr)
            else:
                return _sympify(expr)

        return converter(pylist)

    _node_conversions = {
        "Times": Mul,
        "Plus": Add,
        "Power": Pow,
        "Rational": Rational,
        "Log": lambda *a: log(*reversed(a)),
        "Log2": lambda x: log(x, 2),
        "Log10": lambda x: log(x, 10),
        "Exp": exp,
        "Sqrt": sqrt,

        "Sin": sin,
        "Cos": cos,
        "Tan": tan,
        "Cot": cot,
        "Sec": sec,
        "Csc": csc,

        "ArcSin": asin,
        "ArcCos": acos,
        "ArcTan": lambda *a: atan2(*reversed(a)) if len(a) == 2 else atan(*a),
        "ArcCot": acot,
        "ArcSec": asec,
        "ArcCsc": acsc,

        "Sinh": sinh,
        "Cosh": cosh,
        "Tanh": tanh,
        "Coth": coth,
        "Sech": sech,
        "Csch": csch,

        "ArcSinh": asinh,
        "ArcCosh": acosh,
        "ArcTanh": atanh,
        "ArcCoth": acoth,
        "ArcSech": asech,
        "ArcCsch": acsch,

        "Expand": expand,
        "Im": im,
        "Re": sympy.re,
        "Flatten": flatten,
        "Polylog": polylog,
        "Cancel": cancel,
        # Gamma=gamma,
        "TrigExpand": expand_trig,
        "Sign": sign,
        "Simplify": simplify,
        "Defer": UnevaluatedExpr,
        "Identity": S,
        # Sum=Sum_doit,
        # Module=With,
        # Block=With,
        "Null": lambda *a: S.Zero,
        "Mod": Mod,
        "Max": Max,
        "Min": Min,
        "Pochhammer": rf,
        "ExpIntegralEi": Ei,
        "SinIntegral": Si,
        "CosIntegral": Ci,
        "AiryAi": airyai,
        "AiryAiPrime": airyaiprime,
        "AiryBi": airybi,
        "AiryBiPrime": airybiprime,
        "LogIntegral": li,
        "PrimePi": primepi,
        "Prime": prime,
        "PrimeQ": isprime,

        "List": Tuple,
        "Greater": StrictGreaterThan,
        "GreaterEqual": GreaterThan,
        "Less": StrictLessThan,
        "LessEqual": LessThan,
        "Equal": Equality,
        "Or": Or,
        "And": And,

        "Function": _parse_Function,
    }

    _atom_conversions = {
        "I": I,
        "Pi": pi,
    }

    def _from_fullformlist_to_sympy(self, full_form_list):

        def recurse(expr):
            if isinstance(expr, list):
                if isinstance(expr[0], list):
                    head = recurse(expr[0])
                else:
                    head = self._node_conversions.get(expr[0], Function(expr[0]))
                return head(*[recurse(arg) for arg in expr[1:]])
            else:
                return self._atom_conversions.get(expr, sympify(expr))

        return recurse(full_form_list)

    def _from_fullformsympy_to_sympy(self, mform):

        expr = mform
        for mma_form, sympy_node in self._node_conversions.items():
            expr = expr.replace(Function(mma_form), sympy_node)
        return expr
<|endoftext|>import re
from sympy.concrete.products import product
from sympy.concrete.summations import Sum
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)


class MaximaHelpers:
    def maxima_expand(expr):
        return expr.expand()

    def maxima_float(expr):
        return expr.evalf()

    def maxima_trigexpand(expr):
        return expr.expand(trig=True)

    def maxima_sum(a1, a2, a3, a4):
        return Sum(a1, (a2, a3, a4)).doit()

    def maxima_product(a1, a2, a3, a4):
        return product(a1, (a2, a3, a4))

    def maxima_csc(expr):
        return 1/sin(expr)

    def maxima_sec(expr):
        return 1/cos(expr)

sub_dict = {
    'pi': re.compile(r'%pi'),
    'E': re.compile(r'%e'),
    'I': re.compile(r'%i'),
    '**': re.compile(r'\^'),
    'oo': re.compile(r'\binf\b'),
    '-oo': re.compile(r'\bminf\b'),
    "'-'": re.compile(r'\bminus\b'),
    'maxima_expand': re.compile(r'\bexpand\b'),
    'maxima_float': re.compile(r'\bfloat\b'),
    'maxima_trigexpand': re.compile(r'\btrigexpand'),
    'maxima_sum': re.compile(r'\bsum\b'),
    'maxima_product': re.compile(r'\bproduct\b'),
    'cancel': re.compile(r'\bratsimp\b'),
    'maxima_csc': re.compile(r'\bcsc\b'),
    'maxima_sec': re.compile(r'\bsec\b')
}

var_name = re.compile(r'^\s*(\w+)\s*:')


def parse_maxima(str, globals=None, name_dict={}):
    str = str.strip()
    str = str.rstrip('; ')

    for k, v in sub_dict.items():
        str = v.sub(k, str)

    assign_var = None
    var_match = var_name.search(str)
    if var_match:
        assign_var = var_match.group(1)
        str = str[var_match.end():].strip()

    dct = MaximaHelpers.__dict__.copy()
    dct.update(name_dict)
    obj = sympify(str, locals=dct)

    if assign_var and globals:
        globals[assign_var] = obj

    return obj
<|endoftext|>from sympy.printing import pycode, ccode, fcode
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on

lfortran = import_module('lfortran')
cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

if lfortran:
    from sympy.parsing.fortran.fortran_parser import src_to_sympy
if cin:
    from sympy.parsing.c.c_parser import parse_c

@doctest_depends_on(modules=['lfortran', 'clang.cindex'])
class SymPyExpression:  # type: ignore
    """Class to store and handle SymPy expressions

    This class will hold SymPy Expressions and handle the API for the
    conversion to and from different languages.

    It works with the C and the Fortran Parser to generate SymPy expressions
    which are stored here and which can be converted to multiple language's
    source code.

    Notes
    =====

    The module and its API are currently under development and experimental
    and can be changed during development.

    The Fortran parser does not support numeric assignments, so all the
    variables have been Initialized to zero.

    The module also depends on external dependencies:

    - LFortran which is required to use the Fortran parser
    - Clang which is required for the C parser

    Examples
    ========

    Example of parsing C code:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... int a,b;
    ... float c = 2, d =4;
    ... '''
    >>> a = SymPyExpression(src, 'c')
    >>> a.return_expr()
    [Declaration(Variable(a, type=intc)),
    Declaration(Variable(b, type=intc)),
    Declaration(Variable(c, type=float32, value=2.0)),
    Declaration(Variable(d, type=float32, value=4.0))]

    An example of variable definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src2 = '''
    ... integer :: a, b, c, d
    ... real :: p, q, r, s
    ... '''
    >>> p = SymPyExpression()
    >>> p.convert_to_expr(src2, 'f')
    >>> p.convert_to_c()
    ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0']

    An example of Assignment:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src3 = '''
    ... integer :: a, b, c, d, e
    ... d = a + b - c
    ... e = b * d + c * e / a
    ... '''
    >>> p = SymPyExpression(src3, 'f')
    >>> p.convert_to_python()
    ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'e = 0', 'd = a + b - c', 'e = b*d + c*e/a']

    An example of function definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... integer function f(a,b)
    ... integer, intent(in) :: a, b
    ... integer :: r
    ... end function
    ... '''
    >>> a = SymPyExpression(src, 'f')
    >>> a.convert_to_python()
    ['def f(a, b):\\n   f = 0\\n    r = 0\\n    return f']

    """

    def __init__(self, source_code = None, mode = None):
        """Constructor for SymPyExpression class"""
        super().__init__()
        if not(mode or source_code):
            self._expr = []
        elif mode:
            if source_code:
                if mode.lower() == 'f':
                    if lfortran:
                        self._expr = src_to_sympy(source_code)
                    else:
                        raise ImportError("LFortran is not installed, cannot parse Fortran code")
                elif mode.lower() == 'c':
                    if cin:
                        self._expr = parse_c(source_code)
                    else:
                        raise ImportError("Clang is not installed, cannot parse C code")
                else:
                    raise NotImplementedError(
                        'Parser for specified language is not implemented'
                    )
            else:
                raise ValueError('Source code not present')
        else:
            raise ValueError('Please specify a mode for conversion')

    def convert_to_expr(self, src_code, mode):
        """Converts the given source code to SymPy Expressions

        Attributes
        ==========

        src_code : String
            the source code or filename of the source code that is to be
            converted

        mode: String
            the mode to determine which parser is to be used according to
            the language of the source code
            f or F for Fortran
            c or C for C/C++

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b) result(r)
        ... integer, intent(in) :: a, b
        ... integer :: x
        ... r = a + b -x
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(r, type=integer, value=0)),
        Declaration(Variable(x, type=integer, value=0)),
        Assignment(Variable(r), a + b - x),
        Return(Variable(r))
        ))]




        """
        if mode.lower() == 'f':
            if lfortran:
                self._expr = src_to_sympy(src_code)
            else:
                raise ImportError("LFortran is not installed, cannot parse Fortran code")
        elif mode.lower() == 'c':
            if cin:
                self._expr = parse_c(src_code)
            else:
                raise ImportError("Clang is not installed, cannot parse C code")
        else:
            raise NotImplementedError(
                "Parser for specified language has not been implemented"
            )

    def convert_to_python(self):
        """Returns a list with Python code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression(src2, 'f')
        >>> p.convert_to_python()
        ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'p = 0.0', 'q = 0.0', 'r = 0.0', 's = 0.0', 'c = a/b', 'd = c/a', 's = p/q', 'r = q/p']

        """
        self._pycode = []
        for iter in self._expr:
            self._pycode.append(pycode(iter))
        return self._pycode

    def convert_to_c(self):
        """Returns a list with the c source code for the SymPy expressions


        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src2, 'f')
        >>> p.convert_to_c()
        ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0', 'c = a/b;', 'd = c/a;', 's = p/q;', 'r = q/p;']

        """
        self._ccode = []
        for iter in self._expr:
            self._ccode.append(ccode(iter))
        return self._ccode

    def convert_to_fortran(self):
        """Returns a list with the fortran source code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression(src2, 'f')
        >>> p.convert_to_fortran()
        ['      integer*4 a', '      integer*4 b', '      integer*4 c', '      integer*4 d', '      real*8 p', '      real*8 q', '      real*8 r', '      real*8 s', '      c = a/b', '      d = c/a', '      s = p/q', '      r = q/p']

        """
        self._fcode = []
        for iter in self._expr:
            self._fcode.append(fcode(iter))
        return self._fcode

    def return_expr(self):
        """Returns the expression list

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b)
        ... integer, intent(in) :: a, b
        ... integer :: r
        ... r = a+b
        ... f = r
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(f, type=integer, value=0)),
        Declaration(Variable(r, type=integer, value=0)),
        Assignment(Variable(f), Variable(r)),
        Return(Variable(f))
        ))]

        """
        return self._expr
<|endoftext|>"""
This module implements the functionality to take any Python expression as a
string and fix all numbers and other things before evaluating it,
thus

1/2

returns

Integer(1)/Integer(2)

We use the ast module for this. It is well documented at docs.python.org.

Some tips to understand how this works: use dump() to get a nice
representation of any node. Then write a string of what you want to get,
e.g. "Integer(1)", parse it, dump it and you'll see that you need to do
"Call(Name('Integer', Load()), [node], [], None, None)". You do not need
to bother with lineno and col_offset, just call fix_missing_locations()
before returning the node.
"""

from sympy.core.basic import Basic
from sympy.core.sympify import SympifyError

from ast import parse, NodeTransformer, Call, Name, Load, \
    fix_missing_locations, Constant, Tuple

class Transform(NodeTransformer):

    def __init__(self, local_dict, global_dict):
        NodeTransformer.__init__(self)
        self.local_dict = local_dict
        self.global_dict = global_dict

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return fix_missing_locations(Call(func=Name('Integer', Load()),
                    args=[node], keywords=[]))
        elif isinstance(node.value, float):
            return fix_missing_locations(Call(func=Name('Float', Load()),
                    args=[node], keywords=[]))
        return node

    def visit_Name(self, node):
        if node.id in self.local_dict:
            return node
        elif node.id in self.global_dict:
            name_obj = self.global_dict[node.id]

            if isinstance(name_obj, (Basic, type)) or callable(name_obj):
                return node
        elif node.id in ['True', 'False']:
            return node
        return fix_missing_locations(Call(func=Name('Symbol', Load()),
                args=[Constant(node.id)], keywords=[]))

    def visit_Lambda(self, node):
        args = [self.visit(arg) for arg in node.args.args]
        body = self.visit(node.body)
        n = Call(func=Name('Lambda', Load()),
            args=[Tuple(args, Load()), body], keywords=[])
        return fix_missing_locations(n)

def parse_expr(s, local_dict):
    """
    Converts the string "s" to a SymPy expression, in local_dict.

    It converts all numbers to Integers before feeding it to Python and
    automatically creates Symbols.
    """
    global_dict = {}
    exec('from sympy import *', global_dict)
    try:
        a = parse(s.strip(), mode="eval")
    except SyntaxError:
        raise SympifyError("Cannot parse %s." % repr(s))
    a = Transform(local_dict, global_dict).visit(a)
    e = compile(a, "<string>", "eval")
    return eval(e, global_dict, local_dict)
<|endoftext|>"""Transform a string with Python-like source code into SymPy expression. """
from __future__ import annotations
from tokenize import (generate_tokens, untokenize, TokenError,
    NUMBER, STRING, NAME, OP, ENDMARKER, ERRORTOKEN, NEWLINE)

from keyword import iskeyword

import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Any, Callable
from functools import reduce
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min


null = ''

TOKEN = tuple[int, str]
DICT = dict[str, Any]
TRANS = Callable[[list[TOKEN], DICT, DICT], list[TOKEN]]

def _token_splittable(token_name: str) -> bool:
    """
    Predicate for whether a token name can be split into multiple tokens.

    A token is splittable if it does not contain an underscore character and
    it is not the name of a Greek letter. This is used to implicitly convert
    expressions like 'xyz' into 'x*y*z'.
    """
    if '_' in token_name:
        return False
    try:
        return not unicodedata.lookup('GREEK SMALL LETTER ' + token_name)
    except KeyError:
        return len(token_name) > 1


def _token_callable(token: TOKEN, local_dict: DICT, global_dict: DICT, nextToken=None):
    """
    Predicate for whether a token name represents a callable function.

    Essentially wraps ``callable``, but looks up the token name in the
    locals and globals.
    """
    func = local_dict.get(token[1])
    if not func:
        func = global_dict.get(token[1])
    return callable(func) and not isinstance(func, Symbol)


def _add_factorial_tokens(name: str, result: list[TOKEN]) -> list[TOKEN]:
    if result == [] or result[-1][1] == '(':
        raise TokenError()

    beginning = [(NAME, name), (OP, '(')]
    end = [(OP, ')')]

    diff = 0
    length = len(result)

    for index, token in enumerate(result[::-1]):
        toknum, tokval = token
        i = length - index - 1

        if tokval == ')':
            diff += 1
        elif tokval == '(':
            diff -= 1

        if diff == 0:
            if i - 1 >= 0 and result[i - 1][0] == NAME:
                return result[:i - 1] + beginning + result[i - 1:] + end
            else:
                return result[:i] + beginning + result[i:] + end

    return result


class ParenthesisGroup(list[TOKEN]):
    """List of tokens representing an expression in parentheses."""
    pass


class AppliedFunction:
    """
    A group of tokens representing a function and its arguments.

    `exponent` is for handling the shorthand sin^2, ln^2, etc.
    """
    def __init__(self, function: TOKEN, args: ParenthesisGroup, exponent=None):
        if exponent is None:
            exponent = []
        self.function = function
        self.args = args
        self.exponent = exponent
        self.items = ['function', 'args', 'exponent']

    def expand(self) -> list[TOKEN]:
        """Return a list of tokens representing the function"""
        return [self.function, *self.args]

    def __getitem__(self, index):
        return getattr(self, self.items[index])

    def __repr__(self):
        return "AppliedFunction(%s, %s, %s)" % (self.function, self.args,
                                                self.exponent)


def _flatten(result: list[TOKEN | AppliedFunction]):
    result2: list[TOKEN] = []
    for tok in result:
        if isinstance(tok, AppliedFunction):
            result2.extend(tok.expand())
        else:
            result2.append(tok)
    return result2


def _group_parentheses(recursor: TRANS):
    def _inner(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
        """Group tokens between parentheses with ParenthesisGroup.

        Also processes those tokens recursively.

        """
        result: list[TOKEN | ParenthesisGroup] = []
        stacks: list[ParenthesisGroup] = []
        stacklevel = 0
        for token in tokens:
            if token[0] == OP:
                if token[1] == '(':
                    stacks.append(ParenthesisGroup([]))
                    stacklevel += 1
                elif token[1] == ')':
                    stacks[-1].append(token)
                    stack = stacks.pop()

                    if len(stacks) > 0:
                        # We don't recurse here since the upper-level stack
                        # would reprocess these tokens
                        stacks[-1].extend(stack)
                    else:
                        # Recurse here to handle nested parentheses
                        # Strip off the outer parentheses to avoid an infinite loop
                        inner = stack[1:-1]
                        inner = recursor(inner,
                                         local_dict,
                                         global_dict)
                        parenGroup = [stack[0]] + inner + [stack[-1]]
                        result.append(ParenthesisGroup(parenGroup))
                    stacklevel -= 1
                    continue
            if stacklevel:
                stacks[-1].append(token)
            else:
                result.append(token)
        if stacklevel:
            raise TokenError("Mismatched parentheses")
        return result
    return _inner


def _apply_functions(tokens: list[TOKEN | ParenthesisGroup], local_dict: DICT, global_dict: DICT):
    """Convert a NAME token + ParenthesisGroup into an AppliedFunction.

    Note that ParenthesisGroups, if not applied to any function, are
    converted back into lists of tokens.

    """
    result: list[TOKEN | AppliedFunction] = []
    symbol = None
    for tok in tokens:
        if isinstance(tok, ParenthesisGroup):
            if symbol and _token_callable(symbol, local_dict, global_dict):
                result[-1] = AppliedFunction(symbol, tok)
                symbol = None
            else:
                result.extend(tok)
        elif tok[0] == NAME:
            symbol = tok
            result.append(tok)
        else:
            symbol = None
            result.append(tok)
    return result


def _implicit_multiplication(tokens: list[TOKEN | AppliedFunction], local_dict: DICT, global_dict: DICT):
    """Implicitly adds '*' tokens.

    Cases:

    - Two AppliedFunctions next to each other ("sin(x)cos(x)")

    - AppliedFunction next to an open parenthesis ("sin x (cos x + 1)")

    - A close parenthesis next to an AppliedFunction ("(x+2)sin x")\

    - A close parenthesis next to an open parenthesis ("(x+2)(x+3)")

    - AppliedFunction next to an implicitly applied function ("sin(x)cos x")

    """
    result: list[TOKEN | AppliedFunction] = []
    skip = False
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if skip:
            skip = False
            continue
        if tok[0] == OP and tok[1] == '.' and nextTok[0] == NAME:
            # Dotted name. Do not do implicit multiplication
            skip = True
            continue
        if isinstance(tok, AppliedFunction):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                # Applied function followed by an open parenthesis
                if tok.function[1] == "Function":
                    tok.function = (tok.function[0], 'Symbol')
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                # Applied function followed by implicitly applied function
                result.append((OP, '*'))
        else:
            if tok == (OP, ')'):
                if isinstance(nextTok, AppliedFunction):
                    # Close parenthesis followed by an applied function
                    result.append((OP, '*'))
                elif nextTok[0] == NAME:
                    # Close parenthesis followed by an implicitly applied function
                    result.append((OP, '*'))
                elif nextTok == (OP, '('):
                    # Close parenthesis followed by an open parenthesis
                    result.append((OP, '*'))
            elif tok[0] == NAME and not _token_callable(tok, local_dict, global_dict):
                if isinstance(nextTok, AppliedFunction) or \
                    (nextTok[0] == NAME and _token_callable(nextTok, local_dict, global_dict)):
                    # Constant followed by (implicitly applied) function
                    result.append((OP, '*'))
                elif nextTok == (OP, '('):
                    # Constant followed by parenthesis
                    result.append((OP, '*'))
                elif nextTok[0] == NAME:
                    # Constant followed by constant
                    result.append((OP, '*'))
    if tokens:
        result.append(tokens[-1])
    return result


def _implicit_application(tokens: list[TOKEN | AppliedFunction], local_dict: DICT, global_dict: DICT):
    """Adds parentheses as needed after functions."""
    result: list[TOKEN | AppliedFunction] = []
    appendParen = 0  # number of closing parentheses to add
    skip = 0  # number of tokens to delay before adding a ')' (to
              # capture **, ^, etc.)
    exponentSkip = False  # skipping tokens before inserting parentheses to
                          # work with function exponentiation
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if (tok[0] == NAME and nextTok[0] not in [OP, ENDMARKER, NEWLINE]):
            if _token_callable(tok, local_dict, global_dict, nextTok):  # type: ignore
                result.append((OP, '('))
                appendParen += 1
        # name followed by exponent - function exponentiation
        elif (tok[0] == NAME and nextTok[0] == OP and nextTok[1] == '**'):
            if _token_callable(tok, local_dict, global_dict):  # type: ignore
                exponentSkip = True
        elif exponentSkip:
            # if the last token added was an applied function (i.e. the
            # power of the function exponent) OR a multiplication (as
            # implicit multiplication would have added an extraneous
            # multiplication)
            if (isinstance(tok, AppliedFunction)
                or (tok[0] == OP and tok[1] == '*')):
                # don't add anything if the next token is a multiplication
                # or if there's already a parenthesis (if parenthesis, still
                # stop skipping tokens)
                if not (nextTok[0] == OP and nextTok[1] == '*'):
                    if not(nextTok[0] == OP and nextTok[1] == '('):
                        result.append((OP, '('))
                        appendParen += 1
                    exponentSkip = False
        elif appendParen:
            if nextTok[0] == OP and nextTok[1] in ('^', '**', '*'):
                skip = 1
                continue
            if skip:
                skip -= 1
                continue
            result.append((OP, ')'))
            appendParen -= 1

    if tokens:
        result.append(tokens[-1])

    if appendParen:
        result.extend([(OP, ')')] * appendParen)
    return result


def function_exponentiation(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows functions to be exponentiated, e.g. ``cos**2(x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, function_exponentiation)
    >>> transformations = standard_transformations + (function_exponentiation,)
    >>> parse_expr('sin**4(x)', transformations=transformations)
    sin(x)**4
    """
    result: list[TOKEN] = []
    exponent: list[TOKEN] = []
    consuming_exponent = False
    level = 0
    for tok, nextTok in zip(tokens, tokens[1:]):
        if tok[0] == NAME and nextTok[0] == OP and nextTok[1] == '**':
            if _token_callable(tok, local_dict, global_dict):
                consuming_exponent = True
        elif consuming_exponent:
            if tok[0] == NAME and tok[1] == 'Function':
                tok = (NAME, 'Symbol')
            exponent.append(tok)

            # only want to stop after hitting )
            if tok[0] == nextTok[0] == OP and tok[1] == ')' and nextTok[1] == '(':
                consuming_exponent = False
            # if implicit multiplication was used, we may have )*( instead
            if tok[0] == nextTok[0] == OP and tok[1] == '*' and nextTok[1] == '(':
                consuming_exponent = False
                del exponent[-1]
            continue
        elif exponent and not consuming_exponent:
            if tok[0] == OP:
                if tok[1] == '(':
                    level += 1
                elif tok[1] == ')':
                    level -= 1
            if level == 0:
                result.append(tok)
                result.extend(exponent)
                exponent = []
                continue
        result.append(tok)
    if tokens:
        result.append(tokens[-1])
    if exponent:
        result.extend(exponent)
    return result


def split_symbols_custom(predicate: Callable[[str], bool]):
    """Creates a transformation that splits symbol names.

    ``predicate`` should return True if the symbol name is to be split.

    For instance, to retain the default behavior but avoid splitting certain
    symbol names, a predicate like this would work:


    >>> from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,
    ... standard_transformations, implicit_multiplication,
    ... split_symbols_custom)
    >>> def can_split(symbol):
    ...     if symbol not in ('list', 'of', 'unsplittable', 'names'):
    ...             return _token_splittable(symbol)
    ...     return False
    ...
    >>> transformation = split_symbols_custom(can_split)
    >>> parse_expr('unsplittable', transformations=standard_transformations +
    ... (transformation, implicit_multiplication))
    unsplittable
    """
    def _split_symbols(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
        result: list[TOKEN] = []
        split = False
        split_previous=False

        for tok in tokens:
            if split_previous:
                # throw out closing parenthesis of Symbol that was split
                split_previous=False
                continue
            split_previous=False

            if tok[0] == NAME and tok[1] in ['Symbol', 'Function']:
                split = True

            elif split and tok[0] == NAME:
                symbol = tok[1][1:-1]

                if predicate(symbol):
                    tok_type = result[-2][1]  # Symbol or Function
                    del result[-2:]  # Get rid of the call to Symbol

                    i = 0
                    while i < len(symbol):
                        char = symbol[i]
                        if char in local_dict or char in global_dict:
                            result.append((NAME, "%s" % char))
                        elif char.isdigit():
                            chars = [char]
                            for i in range(i + 1, len(symbol)):
                                if not symbol[i].isdigit():
                                    i -= 1
                                    break
                                chars.append(symbol[i])
                            char = ''.join(chars)
                            result.extend([(NAME, 'Number'), (OP, '('),
                                           (NAME, "'%s'" % char), (OP, ')')])
                        else:
                            use = tok_type if i == len(symbol) else 'Symbol'
                            result.extend([(NAME, use), (OP, '('),
                                           (NAME, "'%s'" % char), (OP, ')')])
                        i += 1

                    # Set split_previous=True so will skip
                    # the closing parenthesis of the original Symbol
                    split = False
                    split_previous = True
                    continue

                else:
                    split = False

            result.append(tok)

        return result

    return _split_symbols


#: Splits symbol names for implicit multiplication.
#:
#: Intended to let expressions like ``xyz`` be parsed as ``x*y*z``. Does not
#: split Greek character names, so ``theta`` will *not* become
#: ``t*h*e*t*a``. Generally this should be used with
#: ``implicit_multiplication``.
split_symbols = split_symbols_custom(_token_splittable)


def implicit_multiplication(tokens: list[TOKEN], local_dict: DICT,
                            global_dict: DICT) -> list[TOKEN]:
    """Makes the multiplication operator optional in most cases.

    Use this before :func:`implicit_application`, otherwise expressions like
    ``sin 2x`` will be parsed as ``x * sin(2)`` rather than ``sin(2*x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication)
    >>> transformations = standard_transformations + (implicit_multiplication,)
    >>> parse_expr('3 x y', transformations=transformations)
    3*x*y
    """
    # These are interdependent steps, so we don't expose them separately
    res1 = _group_parentheses(implicit_multiplication)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_multiplication(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result


def implicit_application(tokens: list[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> list[TOKEN]:
    """Makes parentheses optional in some cases for function calls.

    Use this after :func:`implicit_multiplication`, otherwise expressions
    like ``sin 2x`` will be parsed as ``x * sin(2)`` rather than
    ``sin(2*x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_application)
    >>> transformations = standard_transformations + (implicit_application,)
    >>> parse_expr('cot z + csc z', transformations=transformations)
    cot(z) + csc(z)
    """
    res1 = _group_parentheses(implicit_application)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_application(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result


def implicit_multiplication_application(result: list[TOKEN], local_dict: DICT,
                                        global_dict: DICT) -> list[TOKEN]:
    """Allows a slightly relaxed syntax.

    - Parentheses for single-argument method calls are optional.

    - Multiplication is implicit.

    - Symbol names can be split (i.e. spaces are not needed between
      symbols).

    - Functions can be exponentiated.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication_application)
    >>> parse_expr("10sin**2 x**2 + 3xyz + tan theta",
    ... transformations=(standard_transformations +
    ... (implicit_multiplication_application,)))
    3*x*y*z + 10*sin(x**2)**2 + tan(theta)

    """
    for step in (split_symbols, implicit_multiplication,
                 implicit_application, function_exponentiation):
        result = step(result, local_dict, global_dict)

    return result


def auto_symbol(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Inserts calls to ``Symbol``/``Function`` for undefined variables."""
    result: list[TOKEN] = []
    prevTok = (-1, '')

    tokens.append((-1, ''))  # so zip traverses all tokens
    for tok, nextTok in zip(tokens, tokens[1:]):
        tokNum, tokVal = tok
        nextTokNum, nextTokVal = nextTok
        if tokNum == NAME:
            name = tokVal

            if (name in ['True', 'False', 'None']
                    or iskeyword(name)
                    # Don't convert attribute access
                    or (prevTok[0] == OP and prevTok[1] == '.')
                    # Don't convert keyword arguments
                    or (prevTok[0] == OP and prevTok[1] in ('(', ',')
                        and nextTokNum == OP and nextTokVal == '=')
                    # the name has already been defined
                    or name in local_dict and local_dict[name] is not null):
                result.append((NAME, name))
                continue
            elif name in local_dict:
                local_dict.setdefault(null, set()).add(name)
                if nextTokVal == '(':
                    local_dict[name] = Function(name)
                else:
                    local_dict[name] = Symbol(name)
                result.append((NAME, name))
                continue
            elif name in global_dict:
                obj = global_dict[name]
                if isinstance(obj, (AssumptionKeys, Basic, type)) or callable(obj):
                    result.append((NAME, name))
                    continue

            result.extend([
                (NAME, 'Symbol' if nextTokVal != '(' else 'Function'),
                (OP, '('),
                (NAME, repr(str(name))),
                (OP, ')'),
            ])
        else:
            result.append((tokNum, tokVal))

        prevTok = (tokNum, tokVal)

    return result


def lambda_notation(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Substitutes "lambda" with its SymPy equivalent Lambda().
    However, the conversion does not take place if only "lambda"
    is passed because that is a syntax error.

    """
    result: list[TOKEN] = []
    flag = False
    toknum, tokval = tokens[0]
    tokLen = len(tokens)

    if toknum == NAME and tokval == 'lambda':
        if tokLen == 2 or tokLen == 3 and tokens[1][0] == NEWLINE:
            # In Python 3.6.7+, inputs without a newline get NEWLINE added to
            # the tokens
            result.extend(tokens)
        elif tokLen > 2:
            result.extend([
                (NAME, 'Lambda'),
                (OP, '('),
                (OP, '('),
                (OP, ')'),
                (OP, ')'),
            ])
            for tokNum, tokVal in tokens[1:]:
                if tokNum == OP and tokVal == ':':
                    tokVal = ','
                    flag = True
                if not flag and tokNum == OP and tokVal in ('*', '**'):
                    raise TokenError("Starred arguments in lambda not supported")
                if flag:
                    result.insert(-1, (tokNum, tokVal))
                else:
                    result.insert(-2, (tokNum, tokVal))
    else:
        result.extend(tokens)

    return result


def factorial_notation(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows standard notation for factorial."""
    result: list[TOKEN] = []
    nfactorial = 0
    for toknum, tokval in tokens:
        if toknum == OP and tokval == "!":
            # In Python 3.12 "!" are OP instead of ERRORTOKEN
            nfactorial += 1
        elif toknum == ERRORTOKEN:
            op = tokval
            if op == '!':
                nfactorial += 1
            else:
                nfactorial = 0
                result.append((OP, op))
        else:
            if nfactorial == 1:
                result = _add_factorial_tokens('factorial', result)
            elif nfactorial == 2:
                result = _add_factorial_tokens('factorial2', result)
            elif nfactorial > 2:
                raise TokenError
            nfactorial = 0
            result.append((toknum, tokval))
    return result


def convert_xor(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Treats XOR, ``^``, as exponentiation, ``**``."""
    result: list[TOKEN] = []
    for toknum, tokval in tokens:
        if toknum == OP:
            if tokval == '^':
                result.append((OP, '**'))
            else:
                result.append((toknum, tokval))
        else:
            result.append((toknum, tokval))

    return result


def repeated_decimals(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Allows 0.2[1] notation to represent the repeated decimal 0.2111... (19/90)

    Run this before auto_number.

    """
    result: list[TOKEN] = []

    def is_digit(s):
        return all(i in '0123456789_' for i in s)

    # num will running match any DECIMAL [ INTEGER ]
    num: list[TOKEN] = []
    for toknum, tokval in tokens:
        if toknum == NUMBER:
            if (not num and '.' in tokval and 'e' not in tokval.lower() and
                'j' not in tokval.lower()):
                num.append((toknum, tokval))
            elif is_digit(tokval) and (len(num) == 2 or
                    len(num) == 3 and is_digit(num[-1][1])):
                num.append((toknum, tokval))
            else:
                num = []
        elif toknum == OP:
            if tokval == '[' and len(num) == 1:
                num.append((OP, tokval))
            elif tokval == ']' and len(num) >= 3:
                num.append((OP, tokval))
            elif tokval == '.' and not num:
                # handle .[1]
                num.append((NUMBER, '0.'))
            else:
                num = []
        else:
            num = []

        result.append((toknum, tokval))

        if num and num[-1][1] == ']':
            # pre.post[repetend] = a + b/c + d/e where a = pre, b/c = post,
            # and d/e = repetend
            result = result[:-len(num)]
            pre, post = num[0][1].split('.')
            repetend = num[2][1]
            if len(num) == 5:
                repetend += num[3][1]

            pre = pre.replace('_', '')
            post = post.replace('_', '')
            repetend = repetend.replace('_', '')

            zeros = '0'*len(post)
            post, repetends = [w.lstrip('0') for w in [post, repetend]]
                                        # or else interpreted as octal

            a = pre or '0'
            b, c = post or '0', '1' + zeros
            d, e = repetends, ('9'*len(repetend)) + zeros

            seq = [
                (OP, '('),
                    (NAME, 'Integer'),
                    (OP, '('),
                        (NUMBER, a),
                    (OP, ')'),
                    (OP, '+'),
                    (NAME, 'Rational'),
                    (OP, '('),
                        (NUMBER, b),
                        (OP, ','),
                        (NUMBER, c),
                    (OP, ')'),
                    (OP, '+'),
                    (NAME, 'Rational'),
                    (OP, '('),
                        (NUMBER, d),
                        (OP, ','),
                        (NUMBER, e),
                    (OP, ')'),
                (OP, ')'),
            ]
            result.extend(seq)
            num = []

    return result


def auto_number(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Converts numeric literals to use SymPy equivalents.

    Complex numbers use ``I``, integer literals use ``Integer``, and float
    literals use ``Float``.

    """
    result: list[TOKEN] = []

    for toknum, tokval in tokens:
        if toknum == NUMBER:
            number = tokval
            postfix = []

            if number.endswith(('j', 'J')):
                number = number[:-1]
                postfix = [(OP, '*'), (NAME, 'I')]

            if '.' in number or (('e' in number or 'E' in number) and
                    not (number.startswith(('0x', '0X')))):
                seq = [(NAME, 'Float'), (OP, '('),
                    (NUMBER, repr(str(number))), (OP, ')')]
            else:
                seq = [(NAME, 'Integer'), (OP, '('), (
                    NUMBER, number), (OP, ')')]

            result.extend(seq + postfix)
        else:
            result.append((toknum, tokval))

    return result


def rationalize(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Converts floats into ``Rational``. Run AFTER ``auto_number``."""
    result: list[TOKEN] = []
    passed_float = False
    for toknum, tokval in tokens:
        if toknum == NAME:
            if tokval == 'Float':
                passed_float = True
                tokval = 'Rational'
            result.append((toknum, tokval))
        elif passed_float == True and toknum == NUMBER:
            passed_float = False
            result.append((STRING, tokval))
        else:
            result.append((toknum, tokval))

    return result


def _transform_equals_sign(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Transforms the equals sign ``=`` to instances of Eq.

    This is a helper function for ``convert_equals_signs``.
    Works with expressions containing one equals sign and no
    nesting. Expressions like ``(1=2)=False`` will not work with this
    and should be used with ``convert_equals_signs``.

    Examples: 1=2     to Eq(1,2)
              1*2=x   to Eq(1*2, x)

    This does not deal with function arguments yet.

    """
    result: list[TOKEN] = []
    if (OP, "=") in tokens:
        result.append((NAME, "Eq"))
        result.append((OP, "("))
        for token in tokens:
            if token == (OP, "="):
                result.append((OP, ","))
                continue
            result.append(token)
        result.append((OP, ")"))
    else:
        result = tokens
    return result


def convert_equals_signs(tokens: list[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> list[TOKEN]:
    """ Transforms all the equals signs ``=`` to instances of Eq.

    Parses the equals signs in the expression and replaces them with
    appropriate Eq instances. Also works with nested equals signs.

    Does not yet play well with function arguments.
    For example, the expression ``(x=y)`` is ambiguous and can be interpreted
    as x being an argument to a function and ``convert_equals_signs`` will not
    work for this.

    See also
    ========
    convert_equality_operators

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, convert_equals_signs)
    >>> parse_expr("1*2=x", transformations=(
    ... standard_transformations + (convert_equals_signs,)))
    Eq(2, x)
    >>> parse_expr("(1*2=x)=False", transformations=(
    ... standard_transformations + (convert_equals_signs,)))
    Eq(Eq(2, x), False)

    """
    res1 = _group_parentheses(convert_equals_signs)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _transform_equals_sign(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result


#: Standard transformations for :func:`parse_expr`.
#: Inserts calls to :class:`~.Symbol`, :class:`~.Integer`, and other SymPy
#: datatypes and allows the use of standard factorial notation (e.g. ``x!``).
standard_transformations: tuple[TRANS, ...] \
    = (lambda_notation, auto_symbol, repeated_decimals, auto_number,
       factorial_notation)


def stringify_expr(s: str, local_dict: DICT, global_dict: DICT,
        transformations: tuple[TRANS, ...]) -> str:
    """
    Converts the string ``s`` to Python code, in ``local_dict``

    Generally, ``parse_expr`` should be used.
    """

    tokens = []
    input_code = StringIO(s.strip())
    for toknum, tokval, _, _, _ in generate_tokens(input_code.readline):
        tokens.append((toknum, tokval))

    for transform in transformations:
        tokens = transform(tokens, local_dict, global_dict)

    return untokenize(tokens)


def eval_expr(code, local_dict: DICT, global_dict: DICT):
    """
    Evaluate Python code generated by ``stringify_expr``.

    Generally, ``parse_expr`` should be used.
    """
    expr = eval(
        code, global_dict, local_dict)  # take local objects in preference
    return expr


def parse_expr(s: str, local_dict: DICT | None = None,
               transformations: tuple[TRANS, ...] | str \
                   = standard_transformations,
               global_dict: DICT | None = None, evaluate=True):
    """Converts the string ``s`` to a SymPy expression, in ``local_dict``.

    .. warning::
        Note that this function uses ``eval``, and thus shouldn't be used on
        unsanitized input.

    Parameters
    ==========

    s : str
        The string to parse.

    local_dict : dict, optional
        A dictionary of local variables to use when parsing.

    global_dict : dict, optional
        A dictionary of global variables. By default, this is initialized
        with ``from sympy import *``; provide this parameter to override
        this behavior (for instance, to parse ``"Q & S"``).

    transformations : tuple or str
        A tuple of transformation functions used to modify the tokens of the
        parsed expression before evaluation. The default transformations
        convert numeric literals into their SymPy equivalents, convert
        undefined variables into SymPy symbols, and allow the use of standard
        mathematical factorial notation (e.g. ``x!``). Selection via
        string is available (see below).

    evaluate : bool, optional
        When False, the order of the arguments will remain as they were in the
        string and automatic simplification that would normally occur is
        suppressed. (see examples)

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import parse_expr
    >>> parse_expr("1/2")
    1/2
    >>> type(_)
    <class 'sympy.core.numbers.Half'>
    >>> from sympy.parsing.sympy_parser import standard_transformations,\\
    ... implicit_multiplication_application
    >>> transformations = (standard_transformations +
    ...     (implicit_multiplication_application,))
    >>> parse_expr("2x", transformations=transformations)
    2*x

    When evaluate=False, some automatic simplifications will not occur:

    >>> parse_expr("2**3"), parse_expr("2**3", evaluate=False)
    (8, 2**3)

    In addition the order of the arguments will not be made canonical.
    This feature allows one to tell exactly how the expression was entered:

    >>> a = parse_expr('1 + x', evaluate=False)
    >>> b = parse_expr('x + 1', evaluate=False)
    >>> a == b
    False
    >>> a.args
    (1, x)
    >>> b.args
    (x, 1)

    Note, however, that when these expressions are printed they will
    appear the same:

    >>> assert str(a) == str(b)

    As a convenience, transformations can be seen by printing ``transformations``:

    >>> from sympy.parsing.sympy_parser import transformations

    >>> print(transformations)
    0: lambda_notation
    1: auto_symbol
    2: repeated_decimals
    3: auto_number
    4: factorial_notation
    5: implicit_multiplication_application
    6: convert_xor
    7: implicit_application
    8: implicit_multiplication
    9: convert_equals_signs
    10: function_exponentiation
    11: rationalize

    The ``T`` object provides a way to select these transformations:

    >>> from sympy.parsing.sympy_parser import T

    If you print it, you will see the same list as shown above.

    >>> str(T) == str(transformations)
    True

    Standard slicing will return a tuple of transformations:

    >>> T[:5] == standard_transformations
    True

    So ``T`` can be used to specify the parsing transformations:

    >>> parse_expr("2x", transformations=T[:5])
    Traceback (most recent call last):
    ...
    SyntaxError: invalid syntax
    >>> parse_expr("2x", transformations=T[:6])
    2*x
    >>> parse_expr('.3', transformations=T[3, 11])
    3/10
    >>> parse_expr('.3x', transformations=T[:])
    3*x/10

    As a further convenience, strings 'implicit' and 'all' can be used
    to select 0-5 and all the transformations, respectively.

    >>> parse_expr('.3x', transformations='all')
    3*x/10

    See Also
    ========

    stringify_expr, eval_expr, standard_transformations,
    implicit_multiplication_application

    """

    if local_dict is None:
        local_dict = {}
    elif not isinstance(local_dict, dict):
        raise TypeError('expecting local_dict to be a dict')
    elif null in local_dict:
        raise ValueError('cannot use "" in local_dict')

    if global_dict is None:
        global_dict = {}
        exec('from sympy import *', global_dict)

        builtins_dict = vars(builtins)
        for name, obj in builtins_dict.items():
            if isinstance(obj, types.BuiltinFunctionType):
                global_dict[name] = obj
        global_dict['max'] = Max
        global_dict['min'] = Min

    elif not isinstance(global_dict, dict):
        raise TypeError('expecting global_dict to be a dict')

    transformations = transformations or ()
    if isinstance(transformations, str):
        if transformations == 'all':
            _transformations = T[:]
        elif transformations == 'implicit':
            _transformations = T[:6]
        else:
            raise ValueError('unknown transformation group name')
    else:
        _transformations = transformations

    code = stringify_expr(s, local_dict, global_dict, _transformations)

    if not evaluate:
        code = compile(evaluateFalse(code), '<string>', 'eval') # type: ignore

    try:
        rv = eval_expr(code, local_dict, global_dict)
        # restore neutral definitions for names
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        return rv
    except Exception as e:
        # restore neutral definitions for names
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        raise e from ValueError(f"Error from parse_expr with transformed code: {code!r}")


def evaluateFalse(s: str):
    """
    Replaces operators with the SymPy equivalent and sets evaluate=False.
    """
    node = ast.parse(s)
    transformed_node = EvaluateFalseTransformer().visit(node)
    # node is a Module, we want an Expression
    transformed_node = ast.Expression(transformed_node.body[0].value)

    return ast.fix_missing_locations(transformed_node)


class EvaluateFalseTransformer(ast.NodeTransformer):
    operators = {
        ast.Add: 'Add',
        ast.Mult: 'Mul',
        ast.Pow: 'Pow',
        ast.Sub: 'Add',
        ast.Div: 'Mul',
        ast.BitOr: 'Or',
        ast.BitAnd: 'And',
        ast.BitXor: 'Not',
    }
    functions = (
        'Abs', 'im', 're', 'sign', 'arg', 'conjugate',
        'acos', 'acot', 'acsc', 'asec', 'asin', 'atan',
        'acosh', 'acoth', 'acsch', 'asech', 'asinh', 'atanh',
        'cos', 'cot', 'csc', 'sec', 'sin', 'tan',
        'cosh', 'coth', 'csch', 'sech', 'sinh', 'tanh',
        'exp', 'ln', 'log', 'sqrt', 'cbrt',
    )

    relational_operators = {
        ast.NotEq: 'Ne',
        ast.Lt: 'Lt',
        ast.LtE: 'Le',
        ast.Gt: 'Gt',
        ast.GtE: 'Ge',
        ast.Eq: 'Eq'
    }
    def visit_Compare(self, node):
        def reducer(acc, op_right):
            result, left = acc
            op, right = op_right
            if op.__class__ not in self.relational_operators:
                raise ValueError("Only equation or inequality operators are supported")
            new = ast.Call(
                func=ast.Name(
                    id=self.relational_operators[op.__class__], ctx=ast.Load()
                ),
                args=[self.visit(left), self.visit(right)],
                keywords=[ast.keyword(arg="evaluate", value=ast.Constant(value=False))],
            )
            return result + [new], right

        args, _ = reduce(
            reducer, zip(node.ops, node.comparators), ([], node.left)
        )
        if len(args) == 1:
            return args[0]
        return ast.Call(
            func=ast.Name(id=self.operators[ast.BitAnd], ctx=ast.Load()),
            args=args,
            keywords=[ast.keyword(arg="evaluate", value=ast.Constant(value=False))],
        )

    def flatten(self, args, func):
        result = []
        for arg in args:
            if isinstance(arg, ast.Call):
                arg_func = arg.func
                if isinstance(arg_func, ast.Call):
                    arg_func = arg_func.func
                if arg_func.id == func:
                    result.extend(self.flatten(arg.args, func))
                else:
                    result.append(arg)
            else:
                result.append(arg)
        return result

    def visit_BinOp(self, node):
        if node.op.__class__ in self.operators:
            sympy_class = self.operators[node.op.__class__]
            right = self.visit(node.right)
            left = self.visit(node.left)

            rev = False
            if isinstance(node.op, ast.Sub):
                right = ast.Call(
                    func=ast.Name(id='Mul', ctx=ast.Load()),
                    args=[ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1)), right],
                    keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                )
            elif isinstance(node.op, ast.Div):
                if isinstance(node.left, ast.UnaryOp):
                    left, right = right, left
                    rev = True
                    left = ast.Call(
                    func=ast.Name(id='Pow', ctx=ast.Load()),
                    args=[left, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))],
                    keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                )
                else:
                    right = ast.Call(
                    func=ast.Name(id='Pow', ctx=ast.Load()),
                    args=[right, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))],
                    keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                )

            if rev:  # undo reversal
                left, right = right, left
            new_node = ast.Call(
                func=ast.Name(id=sympy_class, ctx=ast.Load()),
                args=[left, right],
                keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
            )

            if sympy_class in ('Add', 'Mul'):
                # Denest Add or Mul as appropriate
                new_node.args = self.flatten(new_node.args, sympy_class)

            return new_node
        return node

    def visit_Call(self, node):
        new_node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            new_node.keywords.append(ast.keyword(arg='evaluate', value=ast.Constant(value=False)))
        return new_node


_transformation = {  # items can be added but never re-ordered
0: lambda_notation,
1: auto_symbol,
2: repeated_decimals,
3: auto_number,
4: factorial_notation,
5: implicit_multiplication_application,
6: convert_xor,
7: implicit_application,
8: implicit_multiplication,
9: convert_equals_signs,
10: function_exponentiation,
11: rationalize}

transformations = '\n'.join('%s: %s' % (i, func_name(f)) for i, f in _transformation.items())


class _T():
    """class to retrieve transformations from a given slice

    EXAMPLES
    ========

    >>> from sympy.parsing.sympy_parser import T, standard_transformations
    >>> assert T[:5] == standard_transformations
    """
    def __init__(self):
        self.N = len(_transformation)

    def __str__(self):
        return transformations

    def __getitem__(self, t):
        if not type(t) is tuple:
            t = (t,)
        i = []
        for ti in t:
            if type(ti) is int:
                i.append(range(self.N)[ti])
            elif type(ti) is slice:
                i.extend(range(*ti.indices(self.N)))
            else:
                raise TypeError('unexpected slice arg')
        return tuple([_transformation[_] for _ in i])

T = _T()
<|endoftext|>import os
import subprocess
import glob

from sympy.utilities.misc import debug

here = os.path.dirname(__file__)
grammar_file = os.path.abspath(os.path.join(here, "LaTeX.g4"))
dir_latex_antlr = os.path.join(here, "_antlr")

header = '''\
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated from ../LaTeX.g4, derived from latex2sympy
#     latex2sympy is licensed under the MIT license
#     https://github.com/augustt198/latex2sympy/blob/master/LICENSE.txt
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt
'''


def check_antlr_version():
    debug("Checking antlr4 version...")

    try:
        debug(subprocess.check_output(["antlr4"])
              .decode('utf-8').split("\n")[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        debug("The 'antlr4' command line tool is not installed, "
              "or not on your PATH.\n"
              "> Please refer to the README.md file for more information.")
        return False


def build_parser(output_dir=dir_latex_antlr):
    check_antlr_version()

    debug("Updating ANTLR-generated code in {}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "__init__.py"), "w+") as fp:
        fp.write(header)

    args = [
        "antlr4",
        grammar_file,
        "-o", output_dir,
        # for now, not generating these as latex2sympy did not use them
        "-no-visitor",
        "-no-listener",
    ]

    debug("Running code generation...\n\t$ {}".format(" ".join(args)))
    subprocess.check_output(args, cwd=output_dir)

    debug("Applying headers, removing unnecessary files and renaming...")
    # Handle case insensitive file systems. If the files are already
    # generated, they will be written to latex* but LaTeX*.* won't match them.
    for path in (glob.glob(os.path.join(output_dir, "LaTeX*.*")) or
        glob.glob(os.path.join(output_dir, "latex*.*"))):

        # Remove files ending in .interp or .tokens as they are not needed.
        if not path.endswith(".py"):
            os.unlink(path)
            continue

        new_path = os.path.join(output_dir, os.path.basename(path).lower())
        with open(path, 'r') as f:
            lines = [line.rstrip() + '\n' for line in f]

        os.unlink(path)

        with open(new_path, "w") as out_file:
            offset = 0
            while lines[offset].startswith('#'):
                offset += 1
            out_file.write(header)
            out_file.writelines(lines[offset:])

        debug("\t{}".format(new_path))

    return True


if __name__ == "__main__":
    build_parser()
<|endoftext|>from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from re import compile as rcompile

from sympy.parsing.latex.lark import LarkLaTeXParser, TransformToSymPyExpr, parse_latex_lark # noqa

from .errors import LaTeXParsingError  # noqa


IGNORE_L = r"\s*[{]*\s*"
IGNORE_R = r"\s*[}]*\s*"
NO_LEFT = r"(?<!\\left)"
BEGIN_AMS_MAT = r"\\begin{matrix}"
END_AMS_MAT = r"\\end{matrix}"
BEGIN_ARR = r"\\begin{array}{.*?}"
END_ARR = r"\\end{array}"

# begin_delim_regex: end_delim_regex
MATRIX_DELIMS = {fr"\\left\({IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\)",
                 fr"{NO_LEFT}\({IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\)",
                 fr"\\left\[{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\]",
                 fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\]",
                 fr"\\left\|{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\\right\|",
                 fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_AMS_MAT}": fr"{END_AMS_MAT}{IGNORE_R}\|",
                 r"\\begin{pmatrix}": r"\\end{pmatrix}",
                 r"\\begin{bmatrix}": r"\\end{bmatrix}",
                 r"\\begin{vmatrix}": r"\\end{vmatrix}",
                 fr"\\left\({IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\)",
                 fr"{NO_LEFT}\({IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\)",
                 fr"\\left\[{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\]",
                 fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\]",
                 fr"\\left\|{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\\right\|",
                 fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_ARR}": fr"{END_ARR}{IGNORE_R}\|"
                 }

MATRIX_DELIMS_INV = {v: k for k, v in MATRIX_DELIMS.items()}

# begin_delim_regex: ideal_begin_delim_representative
BEGIN_DELIM_REPR = {fr"\\left\({IGNORE_L}{BEGIN_AMS_MAT}": "\\left(\\begin{matrix}",
                    fr"{NO_LEFT}\({IGNORE_L}{BEGIN_AMS_MAT}": "(\\begin{matrix}",
                    fr"\\left\[{IGNORE_L}{BEGIN_AMS_MAT}": "\\left[\\begin{matrix}",
                    fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_AMS_MAT}": "[\\begin{matrix}",
                    fr"\\left\|{IGNORE_L}{BEGIN_AMS_MAT}": "\\left|\\begin{matrix}",
                    fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_AMS_MAT}": "|\\begin{matrix}",
                    r"\\begin{pmatrix}": "\\begin{pmatrix}",
                    r"\\begin{bmatrix}": "\\begin{bmatrix}",
                    r"\\begin{vmatrix}": "\\begin{vmatrix}",
                    fr"\\left\({IGNORE_L}{BEGIN_ARR}": "\\left(\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\({IGNORE_L}{BEGIN_ARR}": "(\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"\\left\[{IGNORE_L}{BEGIN_ARR}": "\\left[\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\[{IGNORE_L}{BEGIN_ARR}": "[\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"\\left\|{IGNORE_L}{BEGIN_ARR}": "\\left|\\begin{array}{COLUMN_SPECIFIERS}",
                    fr"{NO_LEFT}\|{IGNORE_L}{BEGIN_ARR}": "|\\begin{array}{COLUMN_SPECIFIERS}"
                    }

# end_delim_regex: ideal_end_delim_representative
END_DELIM_REPR = {fr"{END_AMS_MAT}{IGNORE_R}\\right\)": "\\end{matrix}\\right)",
                  fr"{END_AMS_MAT}{IGNORE_R}\)": "\\end{matrix})",
                  fr"{END_AMS_MAT}{IGNORE_R}\\right\]": "\\end{matrix}\\right]",
                  fr"{END_AMS_MAT}{IGNORE_R}\]": "\\end{matrix}]",
                  fr"{END_AMS_MAT}{IGNORE_R}\\right\|": "\\end{matrix}\\right|",
                  fr"{END_AMS_MAT}{IGNORE_R}\|": "\\end{matrix}|",
                  r"\\end{pmatrix}": "\\end{pmatrix}",
                  r"\\end{bmatrix}": "\\end{bmatrix}",
                  r"\\end{vmatrix}": "\\end{vmatrix}",
                  fr"{END_ARR}{IGNORE_R}\\right\)": "\\end{array}\\right)",
                  fr"{END_ARR}{IGNORE_R}\)": "\\end{array})",
                  fr"{END_ARR}{IGNORE_R}\\right\]": "\\end{array}\\right]",
                  fr"{END_ARR}{IGNORE_R}\]": "\\end{array}]",
                  fr"{END_ARR}{IGNORE_R}\\right\|": "\\end{array}\\right|",
                  fr"{END_ARR}{IGNORE_R}\|": "\\end{array}|"
                  }


def check_matrix_delimiters(latex_str):
    """Report mismatched, excess, or missing matrix delimiters."""
    spans = []
    for begin_delim in MATRIX_DELIMS:
        end_delim = MATRIX_DELIMS[begin_delim]

        p = rcompile(begin_delim)
        q = rcompile(end_delim)

        spans.extend([(*m.span(), m.group(),
                       begin_delim) for m in p.finditer(latex_str)])
        spans.extend([(*m.span(), m.group(),
                       end_delim) for m in q.finditer(latex_str)])

    spans.sort(key=(lambda x: x[0]))
    if len(spans) % 2 == 1:
        # Odd number of delimiters; therefore something
        # is wrong. We do not complain yet; let's see if
        # we can pinpoint the actual error.
        spans.append((None, None, None, None))

    spans = [(*x, *y) for (x, y) in zip(spans[::2], spans[1::2])]
    for x in spans:
        # x is supposed to be an 8-tuple of the following form:
        #
        # (begin_delim_span_start, begin_delim_span_end,
        # begin_delim_match, begin_delim_regex,
        # end_delim_span_start, end_delim_span_end,
        # end_delim_match, end_delim_regex)

        sellipsis = "..."
        s = x[0] - 10
        if s < 0:
            s = 0
            sellipsis = ""

        eellipsis = "..."
        e = x[1] + 10
        if e > len(latex_str):
            e = len(latex_str)
            eellipsis = ""

        if x[3] in END_DELIM_REPR:
            err = (f"Extra '{x[2]}' at index {x[0]} or "
                   "missing corresponding "
                   f"'{BEGIN_DELIM_REPR[MATRIX_DELIMS_INV[x[3]]]}' "
                   f"in LaTeX string: {sellipsis}{latex_str[s:e]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

        if x[7] is None:
            err = (f"Extra '{x[2]}' at index {x[0]} or "
                   "missing corresponding "
                   f"'{END_DELIM_REPR[MATRIX_DELIMS[x[3]]]}' "
                   f"in LaTeX string: {sellipsis}{latex_str[s:e]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

        correct_end_regex = MATRIX_DELIMS[x[3]]
        sellipsis = "..." if x[0] > 0 else ""
        eellipsis = "..." if x[5] < len(latex_str) else ""
        if x[7] != correct_end_regex:
            err = ("Expected "
                   f"'{END_DELIM_REPR[correct_end_regex]}' "
                   f"to close the '{x[2]}' at index {x[0]} but "
                   f"found '{x[6]}' at index {x[4]} of LaTeX "
                   f"string instead: {sellipsis}{latex_str[x[0]:x[5]]}"
                   f"{eellipsis}")
            raise LaTeXParsingError(err)

__doctest_requires__ = {('parse_latex',): ['antlr4', 'lark']}


@doctest_depends_on(modules=('antlr4', 'lark'))
def parse_latex(s, strict=False, backend="antlr"):
    r"""Converts the input LaTeX string ``s`` to a SymPy ``Expr``.

    Parameters
    ==========

    s : str
        The LaTeX string to parse. In Python source containing LaTeX,
        *raw strings* (denoted with ``r"``, like this one) are preferred,
        as LaTeX makes liberal use of the ``\`` character, which would
        trigger escaping in normal Python strings.
    backend : str, optional
        Currently, there are two backends supported: ANTLR, and Lark.
        The default setting is to use the ANTLR backend, which can be
        changed to Lark if preferred.

        Use ``backend="antlr"`` for the ANTLR-based parser, and
        ``backend="lark"`` for the Lark-based parser.

        The ``backend`` option is case-sensitive, and must be in
        all lowercase.
    strict : bool, optional
        This option is only available with the ANTLR backend.

        If True, raise an exception if the string cannot be parsed as
        valid LaTeX. If False, try to recover gracefully from common
        mistakes.

    Examples
    ========

    >>> from sympy.parsing.latex import parse_latex
    >>> expr = parse_latex(r"\frac {1 + \sqrt {\a}} {\b}")
    >>> expr
    (sqrt(a) + 1)/b
    >>> expr.evalf(4, subs=dict(a=5, b=2))
    1.618
    >>> func = parse_latex(r"\int_1^\alpha \dfrac{\mathrm{d}t}{t}", backend="lark")
    >>> func.evalf(subs={"alpha": 2})
    0.693147180559945
    """

    check_matrix_delimiters(s)

    if backend == "antlr":
        _latex = import_module(
            'sympy.parsing.latex._parse_latex_antlr',
            import_kwargs={'fromlist': ['X']})

        if _latex is not None:
            return _latex.parse_latex(s, strict)
    elif backend == "lark":
        return parse_latex_lark(s)
    else:
        raise NotImplementedError(f"Using the '{backend}' backend in the LaTeX" \
                                   " parser is not supported.")
<|endoftext|># Ported from latex2sympy by @augustt198
# https://github.com/augustt198/latex2sympy
# See license in LICENSE.txt
from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket

from .errors import LaTeXParsingError


LaTeXParser = LaTeXLexer = MathErrorListener = None

try:
    LaTeXParser = import_module('sympy.parsing.latex._antlr.latexparser',
                                import_kwargs={'fromlist': ['LaTeXParser']}).LaTeXParser
    LaTeXLexer = import_module('sympy.parsing.latex._antlr.latexlexer',
                               import_kwargs={'fromlist': ['LaTeXLexer']}).LaTeXLexer
except Exception:
    pass

ErrorListener = import_module('antlr4.error.ErrorListener',
                              warn_not_installed=True,
                              import_kwargs={'fromlist': ['ErrorListener']}
                              )



if ErrorListener:
    class MathErrorListener(ErrorListener.ErrorListener):  # type:ignore # noqa:F811
        def __init__(self, src):
            super(ErrorListener.ErrorListener, self).__init__()
            self.src = src

        def syntaxError(self, recog, symbol, line, col, msg, e):
            fmt = "%s\n%s\n%s"
            marker = "~" * col + "^"

            if msg.startswith("missing"):
                err = fmt % (msg, self.src, marker)
            elif msg.startswith("no viable"):
                err = fmt % ("I expected something else here", self.src, marker)
            elif msg.startswith("mismatched"):
                names = LaTeXParser.literalNames
                expected = [
                    names[i] for i in e.getExpectedTokens() if i < len(names)
                ]
                if len(expected) < 10:
                    expected = " ".join(expected)
                    err = (fmt % ("I expected one of these: " + expected, self.src,
                                  marker))
                else:
                    err = (fmt % ("I expected something else here", self.src,
                                  marker))
            else:
                err = fmt % ("I don't understand this", self.src, marker)
            raise LaTeXParsingError(err)


def parse_latex(sympy, strict=False):
    antlr4 = import_module('antlr4')

    if None in [antlr4, MathErrorListener] or \
            not version('antlr4-python3-runtime').startswith('4.11'):
        raise ImportError("LaTeX parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime) or"
                          " conda (antlr-python-runtime), version 4.11")

    sympy = sympy.strip()
    matherror = MathErrorListener(sympy)

    stream = antlr4.InputStream(sympy)
    lex = LaTeXLexer(stream)
    lex.removeErrorListeners()
    lex.addErrorListener(matherror)

    tokens = antlr4.CommonTokenStream(lex)
    parser = LaTeXParser(tokens)

    # remove default console error listener
    parser.removeErrorListeners()
    parser.addErrorListener(matherror)

    relation = parser.math().relation()
    if strict and (relation.start.start != 0 or relation.stop.stop != len(sympy) - 1):
        raise LaTeXParsingError("Invalid LaTeX")
    expr = convert_relation(relation)

    return expr


def convert_relation(rel):
    if rel.expr():
        return convert_expr(rel.expr())

    lh = convert_relation(rel.relation(0))
    rh = convert_relation(rel.relation(1))
    if rel.LT():
        return sympy.StrictLessThan(lh, rh)
    elif rel.LTE():
        return sympy.LessThan(lh, rh)
    elif rel.GT():
        return sympy.StrictGreaterThan(lh, rh)
    elif rel.GTE():
        return sympy.GreaterThan(lh, rh)
    elif rel.EQUAL():
        return sympy.Eq(lh, rh)
    elif rel.NEQ():
        return sympy.Ne(lh, rh)


def convert_expr(expr):
    return convert_add(expr.additive())


def convert_add(add):
    if add.ADD():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.Add(lh, rh, evaluate=False)
    elif add.SUB():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        if hasattr(rh, "is_Atom") and rh.is_Atom:
            return sympy.Add(lh, -1 * rh, evaluate=False)
        return sympy.Add(lh, sympy.Mul(-1, rh, evaluate=False), evaluate=False)
    else:
        return convert_mp(add.mp())


def convert_mp(mp):
    if hasattr(mp, 'mp'):
        mp_left = mp.mp(0)
        mp_right = mp.mp(1)
    else:
        mp_left = mp.mp_nofunc(0)
        mp_right = mp.mp_nofunc(1)

    if mp.MUL() or mp.CMD_TIMES() or mp.CMD_CDOT():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return sympy.Mul(lh, rh, evaluate=False)
    elif mp.DIV() or mp.CMD_DIV() or mp.COLON():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return sympy.Mul(lh, sympy.Pow(rh, -1, evaluate=False), evaluate=False)
    else:
        if hasattr(mp, 'unary'):
            return convert_unary(mp.unary())
        else:
            return convert_unary(mp.unary_nofunc())


def convert_unary(unary):
    if hasattr(unary, 'unary'):
        nested_unary = unary.unary()
    else:
        nested_unary = unary.unary_nofunc()
    if hasattr(unary, 'postfix_nofunc'):
        first = unary.postfix()
        tail = unary.postfix_nofunc()
        postfix = [first] + tail
    else:
        postfix = unary.postfix()

    if unary.ADD():
        return convert_unary(nested_unary)
    elif unary.SUB():
        numabs = convert_unary(nested_unary)
        # Use Integer(-n) instead of Mul(-1, n)
        return -numabs
    elif postfix:
        return convert_postfix_list(postfix)


def convert_postfix_list(arr, i=0):
    if i >= len(arr):
        raise LaTeXParsingError("Index out of bounds")

    res = convert_postfix(arr[i])
    if isinstance(res, sympy.Expr):
        if i == len(arr) - 1:
            return res  # nothing to multiply by
        else:
            if i > 0:
                left = convert_postfix(arr[i - 1])
                right = convert_postfix(arr[i + 1])
                if isinstance(left, sympy.Expr) and isinstance(
                        right, sympy.Expr):
                    left_syms = convert_postfix(arr[i - 1]).atoms(sympy.Symbol)
                    right_syms = convert_postfix(arr[i + 1]).atoms(
                        sympy.Symbol)
                    # if the left and right sides contain no variables and the
                    # symbol in between is 'x', treat as multiplication.
                    if not (left_syms or right_syms) and str(res) == 'x':
                        return convert_postfix_list(arr, i + 1)
            # multiply by next
            return sympy.Mul(
                res, convert_postfix_list(arr, i + 1), evaluate=False)
    else:  # must be derivative
        wrt = res[0]
        if i == len(arr) - 1:
            raise LaTeXParsingError("Expected expression for derivative")
        else:
            expr = convert_postfix_list(arr, i + 1)
            return sympy.Derivative(expr, wrt)


def do_subs(expr, at):
    if at.expr():
        at_expr = convert_expr(at.expr())
        syms = at_expr.atoms(sympy.Symbol)
        if len(syms) == 0:
            return expr
        elif len(syms) > 0:
            sym = next(iter(syms))
            return expr.subs(sym, at_expr)
    elif at.equality():
        lh = convert_expr(at.equality().expr(0))
        rh = convert_expr(at.equality().expr(1))
        return expr.subs(lh, rh)


def convert_postfix(postfix):
    if hasattr(postfix, 'exp'):
        exp_nested = postfix.exp()
    else:
        exp_nested = postfix.exp_nofunc()

    exp = convert_exp(exp_nested)
    for op in postfix.postfix_op():
        if op.BANG():
            if isinstance(exp, list):
                raise LaTeXParsingError("Cannot apply postfix to derivative")
            exp = sympy.factorial(exp, evaluate=False)
        elif op.eval_at():
            ev = op.eval_at()
            at_b = None
            at_a = None
            if ev.eval_at_sup():
                at_b = do_subs(exp, ev.eval_at_sup())
            if ev.eval_at_sub():
                at_a = do_subs(exp, ev.eval_at_sub())
            if at_b is not None and at_a is not None:
                exp = sympy.Add(at_b, -1 * at_a, evaluate=False)
            elif at_b is not None:
                exp = at_b
            elif at_a is not None:
                exp = at_a

    return exp


def convert_exp(exp):
    if hasattr(exp, 'exp'):
        exp_nested = exp.exp()
    else:
        exp_nested = exp.exp_nofunc()

    if exp_nested:
        base = convert_exp(exp_nested)
        if isinstance(base, list):
            raise LaTeXParsingError("Cannot raise derivative to power")
        if exp.atom():
            exponent = convert_atom(exp.atom())
        elif exp.expr():
            exponent = convert_expr(exp.expr())
        return sympy.Pow(base, exponent, evaluate=False)
    else:
        if hasattr(exp, 'comp'):
            return convert_comp(exp.comp())
        else:
            return convert_comp(exp.comp_nofunc())


def convert_comp(comp):
    if comp.group():
        return convert_expr(comp.group().expr())
    elif comp.abs_group():
        return sympy.Abs(convert_expr(comp.abs_group().expr()), evaluate=False)
    elif comp.atom():
        return convert_atom(comp.atom())
    elif comp.floor():
        return convert_floor(comp.floor())
    elif comp.ceil():
        return convert_ceil(comp.ceil())
    elif comp.func():
        return convert_func(comp.func())


def convert_atom(atom):
    if atom.LETTER():
        sname = atom.LETTER().getText()
        if atom.subexpr():
            if atom.subexpr().expr():  # subscript is expr
                subscript = convert_expr(atom.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(atom.subexpr().atom())
            sname += '_{' + StrPrinter().doprint(subscript) + '}'
        if atom.SINGLE_QUOTES():
            sname += atom.SINGLE_QUOTES().getText()  # put after subscript for easy identify
        return sympy.Symbol(sname)
    elif atom.SYMBOL():
        s = atom.SYMBOL().getText()[1:]
        if s == "infty":
            return sympy.oo
        else:
            if atom.subexpr():
                subscript = None
                if atom.subexpr().expr():  # subscript is expr
                    subscript = convert_expr(atom.subexpr().expr())
                else:  # subscript is atom
                    subscript = convert_atom(atom.subexpr().atom())
                subscriptName = StrPrinter().doprint(subscript)
                s += '_{' + subscriptName + '}'
            return sympy.Symbol(s)
    elif atom.number():
        s = atom.number().getText().replace(",", "")
        return sympy.Number(s)
    elif atom.DIFFERENTIAL():
        var = get_differential_var(atom.DIFFERENTIAL())
        return sympy.Symbol('d' + var.name)
    elif atom.mathit():
        text = rule2text(atom.mathit().mathit_text())
        return sympy.Symbol(text)
    elif atom.frac():
        return convert_frac(atom.frac())
    elif atom.binom():
        return convert_binom(atom.binom())
    elif atom.bra():
        val = convert_expr(atom.bra().expr())
        return Bra(val)
    elif atom.ket():
        val = convert_expr(atom.ket().expr())
        return Ket(val)


def rule2text(ctx):
    stream = ctx.start.getInputStream()
    # starting index of starting token
    startIdx = ctx.start.start
    # stopping index of stopping token
    stopIdx = ctx.stop.stop

    return stream.getText(startIdx, stopIdx)


def convert_frac(frac):
    diff_op = False
    partial_op = False
    if frac.lower and frac.upper:
        lower_itv = frac.lower.getSourceInterval()
        lower_itv_len = lower_itv[1] - lower_itv[0] + 1
        if (frac.lower.start == frac.lower.stop
                and frac.lower.start.type == LaTeXLexer.DIFFERENTIAL):
            wrt = get_differential_var_str(frac.lower.start.text)
            diff_op = True
        elif (lower_itv_len == 2 and frac.lower.start.type == LaTeXLexer.SYMBOL
              and frac.lower.start.text == '\\partial'
              and (frac.lower.stop.type == LaTeXLexer.LETTER
                   or frac.lower.stop.type == LaTeXLexer.SYMBOL)):
            partial_op = True
            wrt = frac.lower.stop.text
            if frac.lower.stop.type == LaTeXLexer.SYMBOL:
                wrt = wrt[1:]

        if diff_op or partial_op:
            wrt = sympy.Symbol(wrt)
            if (diff_op and frac.upper.start == frac.upper.stop
                    and frac.upper.start.type == LaTeXLexer.LETTER
                    and frac.upper.start.text == 'd'):
                return [wrt]
            elif (partial_op and frac.upper.start == frac.upper.stop
                  and frac.upper.start.type == LaTeXLexer.SYMBOL
                  and frac.upper.start.text == '\\partial'):
                return [wrt]
            upper_text = rule2text(frac.upper)

            expr_top = None
            if diff_op and upper_text.startswith('d'):
                expr_top = parse_latex(upper_text[1:])
            elif partial_op and frac.upper.start.text == '\\partial':
                expr_top = parse_latex(upper_text[len('\\partial'):])
            if expr_top:
                return sympy.Derivative(expr_top, wrt)
    if frac.upper:
        expr_top = convert_expr(frac.upper)
    else:
        expr_top = sympy.Number(frac.upperd.text)
    if frac.lower:
        expr_bot = convert_expr(frac.lower)
    else:
        expr_bot = sympy.Number(frac.lowerd.text)
    inverse_denom = sympy.Pow(expr_bot, -1, evaluate=False)
    if expr_top == 1:
        return inverse_denom
    else:
        return sympy.Mul(expr_top, inverse_denom, evaluate=False)

def convert_binom(binom):
    expr_n = convert_expr(binom.n)
    expr_k = convert_expr(binom.k)
    return sympy.binomial(expr_n, expr_k, evaluate=False)

def convert_floor(floor):
    val = convert_expr(floor.val)
    return sympy.floor(val, evaluate=False)

def convert_ceil(ceil):
    val = convert_expr(ceil.val)
    return sympy.ceiling(val, evaluate=False)

def convert_func(func):
    if func.func_normal():
        if func.L_PAREN():  # function called with parenthesis
            arg = convert_func_arg(func.func_arg())
        else:
            arg = convert_func_arg(func.func_arg_noparens())

        name = func.func_normal().start.text[1:]

        # change arc<trig> -> a<trig>
        if name in [
                "arcsin", "arccos", "arctan", "arccsc", "arcsec", "arccot"
        ]:
            name = "a" + name[3:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)
        if name in ["arsinh", "arcosh", "artanh"]:
            name = "a" + name[2:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        if name == "exp":
            expr = sympy.exp(arg, evaluate=False)

        if name in ("log", "lg", "ln"):
            if func.subexpr():
                if func.subexpr().expr():
                    base = convert_expr(func.subexpr().expr())
                else:
                    base = convert_atom(func.subexpr().atom())
            elif name == "lg":  # ISO 80000-2:2019
                base = 10
            elif name in ("ln", "log"):  # SymPy's latex printer prints ln as log by default
                base = sympy.E
            expr = sympy.log(arg, base, evaluate=False)

        func_pow = None
        should_pow = True
        if func.supexpr():
            if func.supexpr().expr():
                func_pow = convert_expr(func.supexpr().expr())
            else:
                func_pow = convert_atom(func.supexpr().atom())

        if name in [
                "sin", "cos", "tan", "csc", "sec", "cot", "sinh", "cosh",
                "tanh"
        ]:
            if func_pow == -1:
                name = "a" + name
                should_pow = False
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        if func_pow and should_pow:
            expr = sympy.Pow(expr, func_pow, evaluate=False)

        return expr
    elif func.LETTER() or func.SYMBOL():
        if func.LETTER():
            fname = func.LETTER().getText()
        elif func.SYMBOL():
            fname = func.SYMBOL().getText()[1:]
        fname = str(fname)  # can't be unicode
        if func.subexpr():
            if func.subexpr().expr():  # subscript is expr
                subscript = convert_expr(func.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(func.subexpr().atom())
            subscriptName = StrPrinter().doprint(subscript)
            fname += '_{' + subscriptName + '}'
        if func.SINGLE_QUOTES():
            fname += func.SINGLE_QUOTES().getText()
        input_args = func.args()
        output_args = []
        while input_args.args():  # handle multiple arguments to function
            output_args.append(convert_expr(input_args.expr()))
            input_args = input_args.args()
        output_args.append(convert_expr(input_args.expr()))
        return sympy.Function(fname)(*output_args)
    elif func.FUNC_INT():
        return handle_integral(func)
    elif func.FUNC_SQRT():
        expr = convert_expr(func.base)
        if func.root:
            r = convert_expr(func.root)
            return sympy.root(expr, r, evaluate=False)
        else:
            return sympy.sqrt(expr, evaluate=False)
    elif func.FUNC_OVERLINE():
        expr = convert_expr(func.base)
        return sympy.conjugate(expr, evaluate=False)
    elif func.FUNC_SUM():
        return handle_sum_or_prod(func, "summation")
    elif func.FUNC_PROD():
        return handle_sum_or_prod(func, "product")
    elif func.FUNC_LIM():
        return handle_limit(func)


def convert_func_arg(arg):
    if hasattr(arg, 'expr'):
        return convert_expr(arg.expr())
    else:
        return convert_mp(arg.mp_nofunc())


def handle_integral(func):
    if func.additive():
        integrand = convert_add(func.additive())
    elif func.frac():
        integrand = convert_frac(func.frac())
    else:
        integrand = 1

    int_var = None
    if func.DIFFERENTIAL():
        int_var = get_differential_var(func.DIFFERENTIAL())
    else:
        for sym in integrand.atoms(sympy.Symbol):
            s = str(sym)
            if len(s) > 1 and s[0] == 'd':
                if s[1] == '\\':
                    int_var = sympy.Symbol(s[2:])
                else:
                    int_var = sympy.Symbol(s[1:])
                int_sym = sym
        if int_var:
            integrand = integrand.subs(int_sym, 1)
        else:
            # Assume dx by default
            int_var = sympy.Symbol('x')

    if func.subexpr():
        if func.subexpr().atom():
            lower = convert_atom(func.subexpr().atom())
        else:
            lower = convert_expr(func.subexpr().expr())
        if func.supexpr().atom():
            upper = convert_atom(func.supexpr().atom())
        else:
            upper = convert_expr(func.supexpr().expr())
        return sympy.Integral(integrand, (int_var, lower, upper))
    else:
        return sympy.Integral(integrand, int_var)


def handle_sum_or_prod(func, name):
    val = convert_mp(func.mp())
    iter_var = convert_expr(func.subeq().equality().expr(0))
    start = convert_expr(func.subeq().equality().expr(1))
    if func.supexpr().expr():  # ^{expr}
        end = convert_expr(func.supexpr().expr())
    else:  # ^atom
        end = convert_atom(func.supexpr().atom())

    if name == "summation":
        return sympy.Sum(val, (iter_var, start, end))
    elif name == "product":
        return sympy.Product(val, (iter_var, start, end))


def handle_limit(func):
    sub = func.limit_sub()
    if sub.LETTER():
        var = sympy.Symbol(sub.LETTER().getText())
    elif sub.SYMBOL():
        var = sympy.Symbol(sub.SYMBOL().getText()[1:])
    else:
        var = sympy.Symbol('x')
    if sub.SUB():
        direction = "-"
    elif sub.ADD():
        direction = "+"
    else:
        direction = "+-"
    approaching = convert_expr(sub.expr())
    content = convert_mp(func.mp())

    return sympy.Limit(content, var, approaching, direction)


def get_differential_var(d):
    text = get_differential_var_str(d.getText())
    return sympy.Symbol(text)


def get_differential_var_str(text):
    for i in range(1, len(text)):
        c = text[i]
        if not (c == " " or c == "\r" or c == "\n" or c == "\t"):
            idx = i
            break
    text = text[idx:]
    if text[0] == "\\":
        text = text[1:]
    return text
<|endoftext|>import os
import logging
import re
from pathlib import Path

from sympy.external import import_module
from sympy.parsing.latex.lark.transformer import TransformToSymPyExpr

_lark = import_module("lark")


class LarkLaTeXParser:
    r"""Class for converting input `\mathrm{\LaTeX}` strings into SymPy Expressions.
    It holds all the necessary internal data for doing so, and exposes hooks for
    customizing its behavior.

    Parameters
    ==========

    print_debug_output : bool, optional

        If set to ``True``, prints debug output to the logger. Defaults to ``False``.

    transform : bool, optional

        If set to ``True``, the class runs the Transformer class on the parse tree
        generated by running ``Lark.parse`` on the input string. Defaults to ``True``.

        Setting it to ``False`` can help with debugging the `\mathrm{\LaTeX}` grammar.

    grammar_file : str, optional

        The path to the grammar file that the parser should use. If set to ``None``,
        it uses the default grammar, which is in ``grammar/latex.lark``, relative to
        the ``sympy/parsing/latex/lark/`` directory.

    transformer : str, optional

        The name of the Transformer class to use. If set to ``None``, it uses the
        default transformer class, which is :py:func:`TransformToSymPyExpr`.

    """
    def __init__(self, print_debug_output=False, transform=True, grammar_file=None, transformer=None):
        grammar_dir_path = os.path.join(os.path.dirname(__file__), "grammar/")

        if grammar_file is None:
            latex_grammar = Path(os.path.join(grammar_dir_path, "latex.lark")).read_text(encoding="utf-8")
        else:
            latex_grammar = Path(grammar_file).read_text(encoding="utf-8")

        self.parser = _lark.Lark(
            latex_grammar,
            source_path=grammar_dir_path,
            parser="earley",
            start="latex_string",
            lexer="auto",
            ambiguity="explicit",
            propagate_positions=False,
            maybe_placeholders=False,
            keep_all_tokens=True)

        self.print_debug_output = print_debug_output
        self.transform_expr = transform

        if transformer is None:
            self.transformer = TransformToSymPyExpr()
        else:
            self.transformer = transformer()

    def doparse(self, s: str):
        if self.print_debug_output:
            _lark.logger.setLevel(logging.DEBUG)

        parse_tree = self.parser.parse(s)

        if not self.transform_expr:
            # exit early and return the parse tree
            _lark.logger.debug("expression = %s", s)
            _lark.logger.debug(parse_tree)
            _lark.logger.debug(parse_tree.pretty())
            return parse_tree

        if self.print_debug_output:
            # print this stuff before attempting to run the transformer
            _lark.logger.debug("expression = %s", s)
            # print the `parse_tree` variable
            _lark.logger.debug(parse_tree.pretty())

        sympy_expression = self.transformer.transform(parse_tree)

        if self.print_debug_output:
            _lark.logger.debug("SymPy expression = %s", sympy_expression)

        return sympy_expression


if _lark is not None:
    _lark_latex_parser = LarkLaTeXParser()


def parse_latex_lark(s: str):
    """
    Experimental LaTeX parser using Lark.

    This function is still under development and its API may change with the
    next releases of SymPy.
    """
    if _lark is None:
        raise ImportError("Lark is probably not installed")
    return _lark_latex_parser.doparse(s)


def _pretty_print_lark_trees(tree, indent=0, show_expr=True):
    if isinstance(tree, _lark.Token):
        return tree.value

    data = str(tree.data)

    is_expr = data.startswith("expression")

    if is_expr:
        data = re.sub(r"^expression", "E", data)

    is_ambig = (data == "_ambig")

    if is_ambig:
        new_indent = indent + 2
    else:
        new_indent = indent

    output = ""
    show_node = not is_expr or show_expr

    if show_node:
        output += str(data) + "("

    if is_ambig:
        output += "\n" + "\n".join([" " * new_indent + _pretty_print_lark_trees(i, new_indent, show_expr) for i in tree.children])
    else:
        output += ",".join([_pretty_print_lark_trees(i, new_indent, show_expr) for i in tree.children])

    if show_node:
        output += ")"

    return output
<|endoftext|>import re

import sympy
from sympy.external import import_module
from sympy.parsing.latex.errors import LaTeXParsingError

lark = import_module("lark")

if lark:
    from lark import Transformer, Token, Tree  # type: ignore
else:
    class Transformer:  # type: ignore
        def transform(self, *args):
            pass


    class Token:  # type: ignore
        pass


    class Tree:  # type: ignore
        pass


# noinspection PyPep8Naming,PyMethodMayBeStatic
class TransformToSymPyExpr(Transformer):
    """Returns a SymPy expression that is generated by traversing the ``lark.Tree``
    passed to the ``.transform()`` function.

    Notes
    =====

    **This class is never supposed to be used directly.**

    In order to tweak the behavior of this class, it has to be subclassed and then after
    the required modifications are made, the name of the new class should be passed to
    the :py:class:`LarkLaTeXParser` class by using the ``transformer`` argument in the
    constructor.

    Parameters
    ==========

    visit_tokens : bool, optional
        For information about what this option does, see `here
        <https://lark-parser.readthedocs.io/en/latest/visitors.html#lark.visitors.Transformer>`_.

        Note that the option must be set to ``True`` for the default parser to work.
    """

    SYMBOL = sympy.Symbol
    DIGIT = sympy.core.numbers.Integer

    def CMD_INFTY(self, tokens):
        return sympy.oo

    def GREEK_SYMBOL_WITH_PRIMES(self, tokens):
        # we omit the first character because it is a backslash. Also, if the variable name has "var" in it,
        # like "varphi" or "varepsilon", we remove that too
        variable_name = re.sub("var", "", tokens[1:])

        return sympy.Symbol(variable_name)

    def LATIN_SYMBOL_WITH_LATIN_SUBSCRIPT(self, tokens):
        base, sub = tokens.value.split("_")
        if sub.startswith("{"):
            return sympy.Symbol("%s_{%s}" % (base, sub[1:-1]))
        else:
            return sympy.Symbol("%s_{%s}" % (base, sub))

    def GREEK_SYMBOL_WITH_LATIN_SUBSCRIPT(self, tokens):
        base, sub = tokens.value.split("_")
        greek_letter = re.sub("var", "", base[1:])

        if sub.startswith("{"):
            return sympy.Symbol("%s_{%s}" % (greek_letter, sub[1:-1]))
        else:
            return sympy.Symbol("%s_{%s}" % (greek_letter, sub))

    def LATIN_SYMBOL_WITH_GREEK_SUBSCRIPT(self, tokens):
        base, sub = tokens.value.split("_")
        if sub.startswith("{"):
            greek_letter = sub[2:-1]
        else:
            greek_letter = sub[1:]

        greek_letter = re.sub("var", "", greek_letter)
        return sympy.Symbol("%s_{%s}" % (base, greek_letter))


    def GREEK_SYMBOL_WITH_GREEK_SUBSCRIPT(self, tokens):
        base, sub = tokens.value.split("_")
        greek_base = re.sub("var", "", base[1:])

        if sub.startswith("{"):
            greek_sub = sub[2:-1]
        else:
            greek_sub = sub[1:]

        greek_sub = re.sub("var", "", greek_sub)
        return sympy.Symbol("%s_{%s}" % (greek_base, greek_sub))

    def multi_letter_symbol(self, tokens):
        if len(tokens) == 4: # no primes (single quotes) on symbol
            return sympy.Symbol(tokens[2])
        if len(tokens) == 5: # there are primes on the symbol
            return sympy.Symbol(tokens[2] + tokens[4])

    def number(self, tokens):
        if tokens[0].type == "CMD_IMAGINARY_UNIT":
            return sympy.I

        if "." in tokens[0]:
            return sympy.core.numbers.Float(tokens[0])
        else:
            return sympy.core.numbers.Integer(tokens[0])

    def latex_string(self, tokens):
        return tokens[0]

    def group_round_parentheses(self, tokens):
        return tokens[1]

    def group_square_brackets(self, tokens):
        return tokens[1]

    def group_curly_parentheses(self, tokens):
        return tokens[1]

    def eq(self, tokens):
        return sympy.Eq(tokens[0], tokens[2])

    def ne(self, tokens):
        return sympy.Ne(tokens[0], tokens[2])

    def lt(self, tokens):
        return sympy.Lt(tokens[0], tokens[2])

    def lte(self, tokens):
        return sympy.Le(tokens[0], tokens[2])

    def gt(self, tokens):
        return sympy.Gt(tokens[0], tokens[2])

    def gte(self, tokens):
        return sympy.Ge(tokens[0], tokens[2])

    def add(self, tokens):
        if len(tokens) == 2: # +a
            return tokens[1]
        if len(tokens) == 3: # a + b
            lh = tokens[0]
            rh = tokens[2]

            if self._obj_is_sympy_Matrix(lh) or self._obj_is_sympy_Matrix(rh):
                return sympy.MatAdd(lh, rh)

            return sympy.Add(lh, rh)

    def sub(self, tokens):
        if len(tokens) == 2: # -a
            x = tokens[1]

            if self._obj_is_sympy_Matrix(x):
                return sympy.MatMul(-1, x)

            return -x
        if len(tokens) == 3: # a - b
            lh = tokens[0]
            rh = tokens[2]

            if self._obj_is_sympy_Matrix(lh) or self._obj_is_sympy_Matrix(rh):
                return sympy.MatAdd(lh, sympy.MatMul(-1, rh))

            return sympy.Add(lh, -rh)

    def mul(self, tokens):
        lh = tokens[0]
        rh = tokens[2]

        if self._obj_is_sympy_Matrix(lh) or self._obj_is_sympy_Matrix(rh):
            return sympy.MatMul(lh, rh)

        return sympy.Mul(lh, rh)

    def div(self, tokens):
        return self._handle_division(tokens[0], tokens[2])

    def adjacent_expressions(self, tokens):
        # Most of the time, if two expressions are next to each other, it means implicit multiplication,
        # but not always
        from sympy.physics.quantum import Bra, Ket
        if isinstance(tokens[0], Ket) and isinstance(tokens[1], Bra):
            from sympy.physics.quantum import OuterProduct
            return OuterProduct(tokens[0], tokens[1])
        elif tokens[0] == sympy.Symbol("d"):
            # If the leftmost token is a "d", then it is highly likely that this is a differential
            return tokens[0], tokens[1]
        elif isinstance(tokens[0], tuple):
            # then we have a derivative
            return sympy.Derivative(tokens[1], tokens[0][1])
        else:
            return sympy.Mul(tokens[0], tokens[1])

    def superscript(self, tokens):
        def isprime(x):
            return isinstance(x, Token) and x.type == "PRIMES"

        def iscmdprime(x):
            return isinstance(x, Token) and (x.type == "PRIMES_VIA_CMD"
                                             or x.type == "CMD_PRIME")

        def isstar(x):
            return isinstance(x, Token) and x.type == "STARS"

        def iscmdstar(x):
            return isinstance(x, Token) and (x.type == "STARS_VIA_CMD"
                                             or x.type == "CMD_ASTERISK")

        base = tokens[0]
        if len(tokens) == 3: # a^b OR a^\prime OR a^\ast
            sup = tokens[2]
        if len(tokens) == 5:
            # a^{'}, a^{''}, ... OR
            # a^{*}, a^{**}, ... OR
            # a^{\prime}, a^{\prime\prime}, ... OR
            # a^{\ast}, a^{\ast\ast}, ...
            sup = tokens[3]

        if self._obj_is_sympy_Matrix(base):
            if sup == sympy.Symbol("T"):
                return sympy.Transpose(base)
            if sup == sympy.Symbol("H"):
                return sympy.adjoint(base)
            if isprime(sup):
                sup = sup.value
                if len(sup) % 2 == 0:
                    return base
                return sympy.Transpose(base)
            if iscmdprime(sup):
                sup = sup.value
                if (len(sup)/len(r"\prime")) % 2 == 0:
                    return base
                return sympy.Transpose(base)
            if isstar(sup):
                sup = sup.value
                # need .doit() in order to be consistent with
                # sympy.adjoint() which returns the evaluated adjoint
                # of a matrix
                if len(sup) % 2 == 0:
                    return base.doit()
                return sympy.adjoint(base)
            if iscmdstar(sup):
                sup = sup.value
                # need .doit() for same reason as above
                if (len(sup)/len(r"\ast")) % 2 == 0:
                    return base.doit()
                return sympy.adjoint(base)

        if isprime(sup) or iscmdprime(sup) or isstar(sup) or iscmdstar(sup):
            raise LaTeXParsingError(f"{base} with superscript {sup} is not understood.")

        return sympy.Pow(base, sup)

    def matrix_prime(self, tokens):
        base = tokens[0]
        primes = tokens[1].value

        if not self._obj_is_sympy_Matrix(base):
            raise LaTeXParsingError(f"({base}){primes} is not understood.")

        if len(primes) % 2 == 0:
            return base

        return sympy.Transpose(base)

    def symbol_prime(self, tokens):
        base = tokens[0]
        primes = tokens[1].value

        return sympy.Symbol(f"{base.name}{primes}")

    def fraction(self, tokens):
        numerator = tokens[1]
        if isinstance(tokens[2], tuple):
            # we only need the variable w.r.t. which we are differentiating
            _, variable = tokens[2]

            # we will pass this information upwards
            return "derivative", variable
        else:
            denominator = tokens[2]
            return self._handle_division(numerator, denominator)

    def binomial(self, tokens):
        return sympy.binomial(tokens[1], tokens[2])

    def normal_integral(self, tokens):
        underscore_index = None
        caret_index = None

        if "_" in tokens:
            # we need to know the index because the next item in the list is the
            # arguments for the lower bound of the integral
            underscore_index = tokens.index("_")

        if "^" in tokens:
            # we need to know the index because the next item in the list is the
            # arguments for the upper bound of the integral
            caret_index = tokens.index("^")

        lower_bound = tokens[underscore_index + 1] if underscore_index else None
        upper_bound = tokens[caret_index + 1] if caret_index else None

        differential_symbol = self._extract_differential_symbol(tokens)

        if differential_symbol is None:
            raise LaTeXParsingError("Differential symbol was not found in the expression."
                                    "Valid differential symbols are \"d\", \"\\text{d}, and \"\\mathrm{d}\".")

        # else we can assume that a differential symbol was found
        differential_variable_index = tokens.index(differential_symbol) + 1
        differential_variable = tokens[differential_variable_index]

        # we can't simply do something like `if (lower_bound and not upper_bound) ...` because this would
        # evaluate to `True` if the `lower_bound` is 0 and upper bound is non-zero
        if lower_bound is not None and upper_bound is None:
            # then one was given and the other wasn't
            raise LaTeXParsingError("Lower bound for the integral was found, but upper bound was not found.")

        if upper_bound is not None and lower_bound is None:
            # then one was given and the other wasn't
            raise LaTeXParsingError("Upper bound for the integral was found, but lower bound was not found.")

        # check if any expression was given or not. If it wasn't, then set the integrand to 1.
        if underscore_index is not None and underscore_index == differential_variable_index - 3:
            # The Token at differential_variable_index - 2 should be the integrand. However, if going one more step
            # backwards after that gives us the underscore, then that means that there _was_ no integrand.
            # Example: \int^7_0 dx
            integrand = 1
        elif caret_index is not None and caret_index == differential_variable_index - 3:
            # The Token at differential_variable_index - 2 should be the integrand. However, if going one more step
            # backwards after that gives us the caret, then that means that there _was_ no integrand.
            # Example: \int_0^7 dx
            integrand = 1
        elif differential_variable_index == 2:
            # this means we have something like "\int dx", because the "\int" symbol will always be
            # at index 0 in `tokens`
            integrand = 1
        else:
            # The Token at differential_variable_index - 1 is the differential symbol itself, so we need to go one
            # more step before that.
            integrand = tokens[differential_variable_index - 2]

        if lower_bound is not None:
            # then we have a definite integral

            # we can assume that either both the lower and upper bounds are given, or
            # neither of them are
            return sympy.Integral(integrand, (differential_variable, lower_bound, upper_bound))
        else:
            # we have an indefinite integral
            return sympy.Integral(integrand, differential_variable)

    def group_curly_parentheses_int(self, tokens):
        # return signature is a tuple consisting of the expression in the numerator, along with the variable of
        # integration
        if len(tokens) == 3:
            return 1, tokens[1]
        elif len(tokens) == 4:
            return tokens[1], tokens[2]
        # there are no other possibilities

    def special_fraction(self, tokens):
        numerator, variable = tokens[1]
        denominator = tokens[2]

        # We pass the integrand, along with information about the variable of integration, upw
        return sympy.Mul(numerator, sympy.Pow(denominator, -1)), variable

    def integral_with_special_fraction(self, tokens):
        underscore_index = None
        caret_index = None

        if "_" in tokens:
            # we need to know the index because the next item in the list is the
            # arguments for the lower bound of the integral
            underscore_index = tokens.index("_")

        if "^" in tokens:
            # we need to know the index because the next item in the list is the
            # arguments for the upper bound of the integral
            caret_index = tokens.index("^")

        lower_bound = tokens[underscore_index + 1] if underscore_index else None
        upper_bound = tokens[caret_index + 1] if caret_index else None

        # we can't simply do something like `if (lower_bound and not upper_bound) ...` because this would
        # evaluate to `True` if the `lower_bound` is 0 and upper bound is non-zero
        if lower_bound is not None and upper_bound is None:
            # then one was given and the other wasn't
            raise LaTeXParsingError("Lower bound for the integral was found, but upper bound was not found.")

        if upper_bound is not None and lower_bound is None:
            # then one was given and the other wasn't
            raise LaTeXParsingError("Upper bound for the integral was found, but lower bound was not found.")

        integrand, differential_variable = tokens[-1]

        if lower_bound is not None:
            # then we have a definite integral

            # we can assume that either both the lower and upper bounds are given, or
            # neither of them are
            return sympy.Integral(integrand, (differential_variable, lower_bound, upper_bound))
        else:
            # we have an indefinite integral
            return sympy.Integral(integrand, differential_variable)

    def group_curly_parentheses_special(self, tokens):
        underscore_index = tokens.index("_")
        caret_index = tokens.index("^")

        # given the type of expressions we are parsing, we can assume that the lower limit
        # will always use braces around its arguments. This is because we don't support
        # converting unconstrained sums into SymPy expressions.

        # first we isolate the bottom limit
        left_brace_index = tokens.index("{", underscore_index)
        right_brace_index = tokens.index("}", underscore_index)

        bottom_limit = tokens[left_brace_index + 1: right_brace_index]

        # next, we isolate the upper limit
        top_limit = tokens[caret_index + 1:]

        # the code below will be useful for supporting things like `\sum_{n = 0}^{n = 5} n^2`
        # if "{" in top_limit:
        #     left_brace_index = tokens.index("{", caret_index)
        #     if left_brace_index != -1:
        #         # then there's a left brace in the string, and we need to find the closing right brace
        #         right_brace_index = tokens.index("}", caret_index)
        #         top_limit = tokens[left_brace_index + 1: right_brace_index]

        # print(f"top  limit = {top_limit}")

        index_variable = bottom_limit[0]
        lower_limit = bottom_limit[-1]
        upper_limit = top_limit[0]  # for now, the index will always be 0

        # print(f"return value = ({index_variable}, {lower_limit}, {upper_limit})")

        return index_variable, lower_limit, upper_limit

    def summation(self, tokens):
        return sympy.Sum(tokens[2], tokens[1])

    def product(self, tokens):
        return sympy.Product(tokens[2], tokens[1])

    def limit_dir_expr(self, tokens):
        caret_index = tokens.index("^")

        if "{" in tokens:
            left_curly_brace_index = tokens.index("{", caret_index)
            direction = tokens[left_curly_brace_index + 1]
        else:
            direction = tokens[caret_index + 1]

        if direction == "+":
            return tokens[0], "+"
        elif direction == "-":
            return tokens[0], "-"
        else:
            return tokens[0], "+-"

    def group_curly_parentheses_lim(self, tokens):
        limit_variable = tokens[1]
        if isinstance(tokens[3], tuple):
            destination, direction = tokens[3]
        else:
            destination = tokens[3]
            direction = "+-"

        return limit_variable, destination, direction

    def limit(self, tokens):
        limit_variable, destination, direction = tokens[2]

        return sympy.Limit(tokens[-1], limit_variable, destination, direction)

    def differential(self, tokens):
        return tokens[1]

    def derivative(self, tokens):
        return sympy.Derivative(tokens[-1], tokens[5])

    def list_of_expressions(self, tokens):
        if len(tokens) == 1:
            # we return it verbatim because the function_applied node expects
            # a list
            return tokens
        else:
            def remove_tokens(args):
                if isinstance(args, Token):
                    if args.type != "COMMA":
                        # An unexpected token was encountered
                        raise LaTeXParsingError("A comma token was expected, but some other token was encountered.")
                    return False
                return True

            return filter(remove_tokens, tokens)

    def function_applied(self, tokens):
        return sympy.Function(tokens[0])(*tokens[2])

    def min(self, tokens):
        return sympy.Min(*tokens[2])

    def max(self, tokens):
        return sympy.Max(*tokens[2])

    def bra(self, tokens):
        from sympy.physics.quantum import Bra
        return Bra(tokens[1])

    def ket(self, tokens):
        from sympy.physics.quantum import Ket
        return Ket(tokens[1])

    def inner_product(self, tokens):
        from sympy.physics.quantum import Bra, Ket, InnerProduct
        return InnerProduct(Bra(tokens[1]), Ket(tokens[3]))

    def sin(self, tokens):
        return sympy.sin(tokens[1])

    def cos(self, tokens):
        return sympy.cos(tokens[1])

    def tan(self, tokens):
        return sympy.tan(tokens[1])

    def csc(self, tokens):
        return sympy.csc(tokens[1])

    def sec(self, tokens):
        return sympy.sec(tokens[1])

    def cot(self, tokens):
        return sympy.cot(tokens[1])

    def sin_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.asin(tokens[-1])
        else:
            return sympy.Pow(sympy.sin(tokens[-1]), exponent)

    def cos_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.acos(tokens[-1])
        else:
            return sympy.Pow(sympy.cos(tokens[-1]), exponent)

    def tan_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.atan(tokens[-1])
        else:
            return sympy.Pow(sympy.tan(tokens[-1]), exponent)

    def csc_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.acsc(tokens[-1])
        else:
            return sympy.Pow(sympy.csc(tokens[-1]), exponent)

    def sec_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.asec(tokens[-1])
        else:
            return sympy.Pow(sympy.sec(tokens[-1]), exponent)

    def cot_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.acot(tokens[-1])
        else:
            return sympy.Pow(sympy.cot(tokens[-1]), exponent)

    def arcsin(self, tokens):
        return sympy.asin(tokens[1])

    def arccos(self, tokens):
        return sympy.acos(tokens[1])

    def arctan(self, tokens):
        return sympy.atan(tokens[1])

    def arccsc(self, tokens):
        return sympy.acsc(tokens[1])

    def arcsec(self, tokens):
        return sympy.asec(tokens[1])

    def arccot(self, tokens):
        return sympy.acot(tokens[1])

    def sinh(self, tokens):
        return sympy.sinh(tokens[1])

    def cosh(self, tokens):
        return sympy.cosh(tokens[1])

    def tanh(self, tokens):
        return sympy.tanh(tokens[1])

    def asinh(self, tokens):
        return sympy.asinh(tokens[1])

    def acosh(self, tokens):
        return sympy.acosh(tokens[1])

    def atanh(self, tokens):
        return sympy.atanh(tokens[1])

    def abs(self, tokens):
        return sympy.Abs(tokens[1])

    def floor(self, tokens):
        return sympy.floor(tokens[1])

    def ceil(self, tokens):
        return sympy.ceiling(tokens[1])

    def factorial(self, tokens):
        return sympy.factorial(tokens[0])

    def conjugate(self, tokens):
        return sympy.conjugate(tokens[1])

    def square_root(self, tokens):
        if len(tokens) == 2:
            # then there was no square bracket argument
            return sympy.sqrt(tokens[1])
        elif len(tokens) == 3:
            # then there _was_ a square bracket argument
            return sympy.root(tokens[2], tokens[1])

    def exponential(self, tokens):
        return sympy.exp(tokens[1])

    def log(self, tokens):
        if tokens[0].type == "FUNC_LG":
            # we don't need to check if there's an underscore or not because having one
            # in this case would be meaningless
            # TODO: ANTLR refers to ISO 80000-2:2019. should we keep base 10 or base 2?
            return sympy.log(tokens[1], 10)
        elif tokens[0].type == "FUNC_LN":
            return sympy.log(tokens[1])
        elif tokens[0].type == "FUNC_LOG":
            # we check if a base was specified or not
            if "_" in tokens:
                # then a base was specified
                return sympy.log(tokens[3], tokens[2])
            else:
                # a base was not specified
                return sympy.log(tokens[1])

    def _extract_differential_symbol(self, s: str):
        differential_symbols = {"d", r"\text{d}", r"\mathrm{d}"}

        differential_symbol = next((symbol for symbol in differential_symbols if symbol in s), None)

        return differential_symbol

    def matrix(self, tokens):
        def is_matrix_row(x):
            return (isinstance(x, Tree) and x.data == "matrix_row")

        def is_not_col_delim(y):
            return (not isinstance(y, Token) or y.type != "MATRIX_COL_DELIM")

        matrix_body = tokens[1].children
        return sympy.Matrix([[y for y in x.children if is_not_col_delim(y)]
                             for x in matrix_body if is_matrix_row(x)])

    def determinant(self, tokens):
        if len(tokens) == 2: # \det A
            if not self._obj_is_sympy_Matrix(tokens[1]):
                raise LaTeXParsingError("Cannot take determinant of non-matrix.")

            return tokens[1].det()

        if len(tokens) == 3: # | A |
            return self.matrix(tokens).det()

    def trace(self, tokens):
        if not self._obj_is_sympy_Matrix(tokens[1]):
            raise LaTeXParsingError("Cannot take trace of non-matrix.")

        return sympy.Trace(tokens[1])

    def adjugate(self, tokens):
        if not self._obj_is_sympy_Matrix(tokens[1]):
            raise LaTeXParsingError("Cannot take adjugate of non-matrix.")

        # need .doit() since MatAdd does not support .adjugate() method
        return tokens[1].doit().adjugate()

    def _obj_is_sympy_Matrix(self, obj):
        if hasattr(obj, "is_Matrix"):
            return obj.is_Matrix

        return isinstance(obj, sympy.Matrix)

    def _handle_division(self, numerator, denominator):
        if self._obj_is_sympy_Matrix(denominator):
            raise LaTeXParsingError("Cannot divide by matrices like this since "
                                    "it is not clear if left or right multiplication "
                                    "by the inverse is intended. Try explicitly "
                                    "multiplying by the inverse instead.")

        if self._obj_is_sympy_Matrix(numerator):
            return sympy.MatMul(numerator, sympy.Pow(denominator, -1))

        return sympy.Mul(numerator, sympy.Pow(denominator, -1))
<|endoftext|>from sympy.external import import_module
import os

cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

"""
This module contains all the necessary Classes and Function used to Parse C and
C++ code into SymPy expression
The module serves as a backend for SymPyExpression to parse C code
It is also dependent on Clang's AST and SymPy's Codegen AST.
The module only supports the features currently supported by the Clang and
codegen AST which will be updated as the development of codegen AST and this
module progresses.
You might find unexpected bugs and exceptions while using the module, feel free
to report them to the SymPy Issue Tracker

Features Supported
==================

- Variable Declarations (integers and reals)
- Assignment (using integer & floating literal and function calls)
- Function Definitions and Declaration
- Function Calls
- Compound statements, Return statements

Notes
=====

The module is dependent on an external dependency which needs to be installed
to use the features of this module.

Clang: The C and C++ compiler which is used to extract an AST from the provided
C source code.

References
==========

.. [1] https://github.com/sympy/sympy/issues
.. [2] https://clang.llvm.org/docs/
.. [3] https://clang.llvm.org/docs/IntroductionToTheClangAST.html

"""

if cin:
    from sympy.codegen.ast import (Variable, Integer, Float,
        FunctionPrototype, FunctionDefinition, FunctionCall,
        none, Return, Assignment, intc, int8, int16, int64,
        uint8, uint16, uint32, uint64, float32, float64, float80,
        aug_assign, bool_, While, CodeBlock)
    from sympy.codegen.cnodes import (PreDecrement, PostDecrement,
        PreIncrement, PostIncrement)
    from sympy.core import Add, Mod, Mul, Pow, Rel
    from sympy.logic.boolalg import And, as_Boolean, Not, Or
    from sympy.core.symbol import Symbol
    from sympy.core.sympify import sympify
    from sympy.logic.boolalg import (false, true)
    import sys
    import tempfile

    class BaseParser:
        """Base Class for the C parser"""

        def __init__(self):
            """Initializes the Base parser creating a Clang AST index"""
            self.index = cin.Index.create()

        def diagnostics(self, out):
            """Diagostics function for the Clang AST"""
            for diag in self.tu.diagnostics:
                # tu = translation unit
                print('%s %s (line %s, col %s) %s' % (
                        {
                            4: 'FATAL',
                            3: 'ERROR',
                            2: 'WARNING',
                            1: 'NOTE',
                            0: 'IGNORED',
                        }[diag.severity],
                        diag.location.file,
                        diag.location.line,
                        diag.location.column,
                        diag.spelling
                    ), file=out)

    class CCodeConverter(BaseParser):
        """The Code Convereter for Clang AST

        The converter object takes the C source code or file as input and
        converts them to SymPy Expressions.
        """

        def __init__(self):
            """Initializes the code converter"""
            super().__init__()
            self._py_nodes = []
            self._data_types = {
                "void": {
                    cin.TypeKind.VOID: none
                },
                "bool": {
                    cin.TypeKind.BOOL: bool_
                },
                "int": {
                    cin.TypeKind.SCHAR: int8,
                    cin.TypeKind.SHORT: int16,
                    cin.TypeKind.INT: intc,
                    cin.TypeKind.LONG: int64,
                    cin.TypeKind.UCHAR: uint8,
                    cin.TypeKind.USHORT: uint16,
                    cin.TypeKind.UINT: uint32,
                    cin.TypeKind.ULONG: uint64
                },
                "float": {
                    cin.TypeKind.FLOAT: float32,
                    cin.TypeKind.DOUBLE: float64,
                    cin.TypeKind.LONGDOUBLE: float80
                }
            }

        def parse(self, filename, flags):
