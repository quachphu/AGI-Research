            intersect = intersection(facets[index], facets[j], "segment2D")
        if intersect:
            distance_origin = norm(tuple(map(lambda x, y: x - y,
                                             intersect, x0)))
            if is_vertex(intersect):
                if isinstance(expr, Expr):
                    if len(gens) == 3:
                        expr_dict = {gens[0]: intersect[0],
                                     gens[1]: intersect[1],
                                     gens[2]: intersect[2]}
                    else:
                        expr_dict = {gens[0]: intersect[0],
                                     gens[1]: intersect[1]}
                    value += distance_origin * expr.subs(expr_dict)
                else:
                    value += distance_origin * expr
    return value


def integration_reduction_dynamic(facets, index, a, b, expr, degree, dims,
                                  x_index, y_index, max_index, x0,
                                  monomial_values, monom_index, vertices=None,
                                  hp_param=None):
    """The same integration_reduction function which uses a dynamic
    programming approach to compute terms by using the values of the integral
    of previously computed terms.

    Parameters
    ==========

    facets :
        Facets of the Polytope.
    index :
        Index of facet to find intersections with.(Used in left_integral()).
    a, b :
        Hyperplane parameters.
    expr :
        Input monomial.
    degree :
        Total degree of ``expr``.
    dims :
        Tuple denoting axes variables.
    x_index :
        Exponent of 'x' in ``expr``.
    y_index :
        Exponent of 'y' in ``expr``.
    max_index :
        Maximum exponent of any monomial in ``monomial_values``.
    x0 :
        First point on ``facets[index]``.
    monomial_values :
        List of monomial values constituting the polynomial.
    monom_index :
        Index of monomial whose integration is being found.
    vertices : optional
        Coordinates of vertices constituting the 3-Polytope.
    hp_param : optional
        Hyperplane Parameter of the face of the facets[index].

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import (integration_reduction_dynamic, \
            hyperplane_parameters)
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> x0 = facets[0].points[0]
    >>> monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],\
                           [y, 0, 1, 15], [x, 1, 0, None]]
    >>> integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1, 0, 1,\
                                      x0, monomial_values, 3)
    25/2
    """
    value = S.Zero
    m = len(facets)

    if expr == S.Zero:
        return expr

    if len(dims) == 2:
        if not expr.is_number:
            _, x_degree, y_degree, _ = monomial_values[monom_index]
            x_index = monom_index - max_index + \
                x_index - 2 if x_degree > 0 else 0
            y_index = monom_index - 1 if y_degree > 0 else 0
            x_value, y_value =\
                monomial_values[x_index][3], monomial_values[y_index][3]

            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1]

        value += left_integral2D(m, index, facets, x0, expr, dims)
    else:
        # For 3D use case the max_index contains the z_degree of the term
        z_index = max_index
        if not expr.is_number:
            x_degree, y_degree, z_degree = y_index,\
                                           z_index - x_index - y_index, x_index
            x_value = monomial_values[z_index - 1][y_index - 1][x_index][7]\
                if x_degree > 0 else 0
            y_value = monomial_values[z_index - 1][y_index][x_index][7]\
                if y_degree > 0 else 0
            z_value = monomial_values[z_index - 1][y_index][x_index - 1][7]\
                if z_degree > 0 else 0

            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1] \
                + z_degree * z_value * x0[2]

        value += left_integral3D(facets, index, expr,
                                 vertices, hp_param, degree)
    return value / (len(dims) + degree - 1)


def left_integral3D(facets, index, expr, vertices, hp_param, degree):
    """Computes the left integral of Eq 10 in Chin et al.

    Explanation
    ===========

    For the 3D case, this is the sum of the integral values over constituting
    line segments of the face (which is accessed by facets[index]) multiplied
    by the distance between the first point of facet and that line segment.

    Parameters
    ==========

    facets :
        List of faces of the 3-Polytope.
    index :
        Index of face over which integral is to be calculated.
    expr :
        Input polynomial.
    vertices :
        List of vertices that constitute the 3-Polytope.
    hp_param :
        The hyperplane parameters of the face.
    degree :
        Degree of the ``expr``.

    Examples
    ========

    >>> from sympy.integrals.intpoly import left_integral3D
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                 [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> facets = cube[1:]
    >>> vertices = cube[0]
    >>> left_integral3D(facets, 3, 1, vertices, ([0, -1, 0], -5), 0)
    -50
    """
    value = S.Zero
    facet = facets[index]
    x0 = vertices[facet[0]]
    facet_len = len(facet)
    for i, fac in enumerate(facet):
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        value += distance_to_side(x0, side, hp_param[0]) * \
            lineseg_integrate(facet, i, side, expr, degree)
    return value


def gradient_terms(binomial_power=0, no_of_gens=2):
    """Returns a list of all the possible monomials between
    0 and y**binomial_power for 2D case and z**binomial_power
    for 3D case.

    Parameters
    ==========

    binomial_power :
        Power upto which terms are generated.
    no_of_gens :
        Denotes whether terms are being generated for 2D or 3D case.

    Examples
    ========

    >>> from sympy.integrals.intpoly import gradient_terms
    >>> gradient_terms(2)
    [[1, 0, 0, 0], [y, 0, 1, 0], [y**2, 0, 2, 0], [x, 1, 0, 0],
    [x*y, 1, 1, 0], [x**2, 2, 0, 0]]
    >>> gradient_terms(2, 3)
    [[[[1, 0, 0, 0, 0, 0, 0, 0]]], [[[y, 0, 1, 0, 1, 0, 0, 0],
    [z, 0, 0, 1, 1, 0, 1, 0]], [[x, 1, 0, 0, 1, 1, 0, 0]]],
    [[[y**2, 0, 2, 0, 2, 0, 0, 0], [y*z, 0, 1, 1, 2, 0, 1, 0],
    [z**2, 0, 0, 2, 2, 0, 2, 0]], [[x*y, 1, 1, 0, 2, 1, 0, 0],
    [x*z, 1, 0, 1, 2, 1, 1, 0]], [[x**2, 2, 0, 0, 2, 2, 0, 0]]]]
    """
    if no_of_gens == 2:
        count = 0
        terms = [None] * int((binomial_power ** 2 + 3 * binomial_power + 2) / 2)
        for x_count in range(0, binomial_power + 1):
            for y_count in range(0, binomial_power - x_count + 1):
                terms[count] = [x**x_count*y**y_count,
                                x_count, y_count, 0]
                count += 1
    else:
        terms = [[[[x ** x_count * y ** y_count *
                    z ** (z_count - y_count - x_count),
                    x_count, y_count, z_count - y_count - x_count,
                    z_count, x_count, z_count - y_count - x_count, 0]
                 for y_count in range(z_count - x_count, -1, -1)]
                 for x_count in range(0, z_count + 1)]
                 for z_count in range(0, binomial_power + 1)]
    return terms


def hyperplane_parameters(poly, vertices=None):
    """A helper function to return the hyperplane parameters
    of which the facets of the polytope are a part of.

    Parameters
    ==========

    poly :
        The input 2/3-Polytope.
    vertices :
        Vertex indices of 3-Polytope.

    Examples
    ========

    >>> from sympy import Point, Polygon
    >>> from sympy.integrals.intpoly import hyperplane_parameters
    >>> hyperplane_parameters(Polygon(Point(0, 3), Point(5, 3), Point(1, 1)))
    [((0, 1), 3), ((1, -2), -1), ((-2, -1), -3)]
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> hyperplane_parameters(cube[1:], cube[0])
    [([0, -1, 0], -5), ([0, 0, -1], -5), ([-1, 0, 0], -5),
    ([0, 1, 0], 0), ([1, 0, 0], 0), ([0, 0, 1], 0)]
    """
    if isinstance(poly, Polygon):
        vertices = list(poly.vertices) + [poly.vertices[0]]  # Close the polygon
        params = [None] * (len(vertices) - 1)

        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            a1 = v1[1] - v2[1]
            a2 = v2[0] - v1[0]
            b = v2[0] * v1[1] - v2[1] * v1[0]

            factor = gcd_list([a1, a2, b])

            b = S(b) / factor
            a = (S(a1) / factor, S(a2) / factor)
            params[i] = (a, b)
    else:
        params = [None] * len(poly)
        for i, polygon in enumerate(poly):
            v1, v2, v3 = [vertices[vertex] for vertex in polygon[:3]]
            normal = cross_product(v1, v2, v3)
            b = sum(normal[j] * v1[j] for j in range(0, 3))
            fac = gcd_list(normal)
            if fac.is_zero:
                fac = 1
            normal = [j / fac for j in normal]
            b = b / fac
            params[i] = (normal, b)
    return params


def cross_product(v1, v2, v3):
    """Returns the cross-product of vectors (v2 - v1) and (v3 - v1)
    That is : (v2 - v1) X (v3 - v1)
    """
    v2 = [v2[j] - v1[j] for j in range(0, 3)]
    v3 = [v3[j] - v1[j] for j in range(0, 3)]
    return [v3[2] * v2[1] - v3[1] * v2[2],
            v3[0] * v2[2] - v3[2] * v2[0],
            v3[1] * v2[0] - v3[0] * v2[1]]


def best_origin(a, b, lineseg, expr):
    """Helper method for polytope_integrate. Currently not used in the main
    algorithm.

    Explanation
    ===========

    Returns a point on the lineseg whose vector inner product with the
    divergence of `expr` yields an expression with the least maximum
    total power.

    Parameters
    ==========

    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    lineseg :
        Line segment on which to find the origin.
    expr :
        The expression which determines the best point.

    Algorithm(currently works only for 2D use case)
    ===============================================

    1 > Firstly, check for edge cases. Here that would refer to vertical
        or horizontal lines.

    2 > If input expression is a polynomial containing more than one generator
        then find out the total power of each of the generators.

        x**2 + 3 + x*y + x**4*y**5 ---> {x: 7, y: 6}

        If expression is a constant value then pick the first boundary point
        of the line segment.

    3 > First check if a point exists on the line segment where the value of
        the highest power generator becomes 0. If not check if the value of
        the next highest becomes 0. If none becomes 0 within line segment
        constraints then pick the first boundary point of the line segment.
        Actually, any point lying on the segment can be picked as best origin
        in the last case.

    Examples
    ========

    >>> from sympy.integrals.intpoly import best_origin
    >>> from sympy.abc import x, y
    >>> from sympy import Point, Segment2D
    >>> l = Segment2D(Point(0, 3), Point(1, 1))
    >>> expr = x**3*y**7
    >>> best_origin((2, 1), 3, l, expr)
    (0, 3.0)
    """
    a1, b1 = lineseg.points[0]

    def x_axis_cut(ls):
        """Returns the point where the input line segment
        intersects the x-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        p, q = ls.points
        if p.y.is_zero:
            return tuple(p)
        elif q.y.is_zero:
            return tuple(q)
        elif p.y/q.y < S.Zero:
            return p.y * (p.x - q.x)/(q.y - p.y) + p.x, S.Zero
        else:
            return ()

    def y_axis_cut(ls):
        """Returns the point where the input line segment
        intersects the y-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        p, q = ls.points
        if p.x.is_zero:
            return tuple(p)
        elif q.x.is_zero:
            return tuple(q)
        elif p.x/q.x < S.Zero:
            return S.Zero, p.x * (p.y - q.y)/(q.x - p.x) + p.y
        else:
            return ()

    gens = (x, y)
    power_gens = {}

    for i in gens:
        power_gens[i] = S.Zero

    if len(gens) > 1:
        # Special case for vertical and horizontal lines
        if len(gens) == 2:
            if a[0] == 0:
                if y_axis_cut(lineseg):
                    return S.Zero, b/a[1]
                else:
                    return a1, b1
            elif a[1] == 0:
                if x_axis_cut(lineseg):
                    return b/a[0], S.Zero
                else:
                    return a1, b1

        if isinstance(expr, Expr):  # Find the sum total of power of each
            if expr.is_Add:         # generator and store in a dictionary.
                for monomial in expr.args:
                    if monomial.is_Pow:
                        if monomial.args[0] in gens:
                            power_gens[monomial.args[0]] += monomial.args[1]
                    else:
                        for univariate in monomial.args:
                            term_type = len(univariate.args)
                            if term_type == 0 and univariate in gens:
                                power_gens[univariate] += 1
                            elif term_type == 2 and univariate.args[0] in gens:
                                power_gens[univariate.args[0]] +=\
                                           univariate.args[1]
            elif expr.is_Mul:
                for term in expr.args:
                    term_type = len(term.args)
                    if term_type == 0 and term in gens:
                        power_gens[term] += 1
                    elif term_type == 2 and term.args[0] in gens:
                        power_gens[term.args[0]] += term.args[1]
            elif expr.is_Pow:
                power_gens[expr.args[0]] = expr.args[1]
            elif expr.is_Symbol:
                power_gens[expr] += 1
        else:  # If `expr` is a constant take first vertex of the line segment.
            return a1, b1

        #  TODO : This part is quite hacky. Should be made more robust with
        #  TODO : respect to symbol names and scalable w.r.t higher dimensions.
        power_gens = sorted(power_gens.items(), key=lambda k: str(k[0]))
        if power_gens[0][1] >= power_gens[1][1]:
            if y_axis_cut(lineseg):
                x0 = (S.Zero, b / a[1])
            elif x_axis_cut(lineseg):
                x0 = (b / a[0], S.Zero)
            else:
                x0 = (a1, b1)
        else:
            if x_axis_cut(lineseg):
                x0 = (b/a[0], S.Zero)
            elif y_axis_cut(lineseg):
                x0 = (S.Zero, b/a[1])
            else:
                x0 = (a1, b1)
    else:
        x0 = (b/a[0])
    return x0


def decompose(expr, separate=False):
    """Decomposes an input polynomial into homogeneous ones of
    smaller or equal degree.

    Explanation
    ===========

    Returns a dictionary with keys as the degree of the smaller
    constituting polynomials. Values are the constituting polynomials.

    Parameters
    ==========

    expr : Expr
        Polynomial(SymPy expression).
    separate : bool
        If True then simply return a list of the constituent monomials
        If not then break up the polynomial into constituent homogeneous
        polynomials.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import decompose
    >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5)
    {1: x + y, 2: x**2 + x*y, 5: x**3*y**2 + y**5}
    >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5, True)
    {x, x**2, y, y**5, x*y, x**3*y**2}
    """
    poly_dict = {}

    if isinstance(expr, Expr) and not expr.is_number:
        if expr.is_Symbol:
            poly_dict[1] = expr
        elif expr.is_Add:
            symbols = expr.atoms(Symbol)
            degrees = [(sum(degree_list(monom, *symbols)), monom)
                       for monom in expr.args]
            if separate:
                return {monom[1] for monom in degrees}
            else:
                for monom in degrees:
                    degree, term = monom
                    if poly_dict.get(degree):
                        poly_dict[degree] += term
                    else:
                        poly_dict[degree] = term
        elif expr.is_Pow:
            _, degree = expr.args
            poly_dict[degree] = expr
        else:  # Now expr can only be of `Mul` type
            degree = 0
            for term in expr.args:
                term_type = len(term.args)
                if term_type == 0 and term.is_Symbol:
                    degree += 1
                elif term_type == 2:
                    degree += term.args[1]
            poly_dict[degree] = expr
    else:
        poly_dict[0] = expr

    if separate:
        return set(poly_dict.values())
    return poly_dict


def point_sort(poly, normal=None, clockwise=True):
    """Returns the same polygon with points sorted in clockwise or
    anti-clockwise order.

    Note that it's necessary for input points to be sorted in some order
    (clockwise or anti-clockwise) for the integration algorithm to work.
    As a convention algorithm has been implemented keeping clockwise
    orientation in mind.

    Parameters
    ==========

    poly:
        2D or 3D Polygon.
    normal : optional
        The normal of the plane which the 3-Polytope is a part of.
    clockwise : bool, optional
        Returns points sorted in clockwise order if True and
        anti-clockwise if False.

    Examples
    ========

    >>> from sympy.integrals.intpoly import point_sort
    >>> from sympy import Point
    >>> point_sort([Point(0, 0), Point(1, 0), Point(1, 1)])
    [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]
    """
    pts = poly.vertices if isinstance(poly, Polygon) else poly
    n = len(pts)
    if n < 2:
        return list(pts)

    order = S.One if clockwise else S.NegativeOne
    dim = len(pts[0])
    if dim == 2:
        center = Point(sum((vertex.x for vertex in pts)) / n,
                        sum((vertex.y for vertex in pts)) / n)
    else:
        center = Point(sum((vertex.x for vertex in pts)) / n,
                        sum((vertex.y for vertex in pts)) / n,
                        sum((vertex.z for vertex in pts)) / n)

    def compare(a, b):
        if a.x - center.x >= S.Zero and b.x - center.x < S.Zero:
            return -order
        elif a.x - center.x < 0 and b.x - center.x >= 0:
            return order
        elif a.x - center.x == 0 and b.x - center.x == 0:
            if a.y - center.y >= 0 or b.y - center.y >= 0:
                return -order if a.y > b.y else order
            return -order if b.y > a.y else order

        det = (a.x - center.x) * (b.y - center.y) -\
              (b.x - center.x) * (a.y - center.y)
        if det < 0:
            return -order
        elif det > 0:
            return order

        first = (a.x - center.x) * (a.x - center.x) +\
                (a.y - center.y) * (a.y - center.y)
        second = (b.x - center.x) * (b.x - center.x) +\
                 (b.y - center.y) * (b.y - center.y)
        return -order if first > second else order

    def compare3d(a, b):
        det = cross_product(center, a, b)
        dot_product = sum(det[i] * normal[i] for i in range(0, 3))
        if dot_product < 0:
            return -order
        elif dot_product > 0:
            return order

    return sorted(pts, key=cmp_to_key(compare if dim==2 else compare3d))


def norm(point):
    """Returns the Euclidean norm of a point from origin.

    Parameters
    ==========

    point:
        This denotes a point in the dimension_al spac_e.

    Examples
    ========

    >>> from sympy.integrals.intpoly import norm
    >>> from sympy import Point
    >>> norm(Point(2, 7))
    sqrt(53)
    """
    half = S.Half
    if isinstance(point, (list, tuple)):
        return sum(coord ** 2 for coord in point) ** half
    elif isinstance(point, Point):
        if isinstance(point, Point2D):
            return (point.x ** 2 + point.y ** 2) ** half
        else:
            return (point.x ** 2 + point.y ** 2 + point.z) ** half
    elif isinstance(point, dict):
        return sum(i**2 for i in point.values()) ** half


def intersection(geom_1, geom_2, intersection_type):
    """Returns intersection between geometric objects.

    Explanation
    ===========

    Note that this function is meant for use in integration_reduction and
    at that point in the calling function the lines denoted by the segments
    surely intersect within segment boundaries. Coincident lines are taken
    to be non-intersecting. Also, the hyperplane intersection for 2D case is
    also implemented.

    Parameters
    ==========

    geom_1, geom_2:
        The input line segments.

    Examples
    ========

    >>> from sympy.integrals.intpoly import intersection
    >>> from sympy import Point, Segment2D
    >>> l1 = Segment2D(Point(1, 1), Point(3, 5))
    >>> l2 = Segment2D(Point(2, 0), Point(2, 5))
    >>> intersection(l1, l2, "segment2D")
    (2, 3)
    >>> p1 = ((-1, 0), 0)
    >>> p2 = ((0, 1), 1)
    >>> intersection(p1, p2, "plane2D")
    (0, 1)
    """
    if intersection_type[:-2] == "segment":
        if intersection_type == "segment2D":
            x1, y1 = geom_1.points[0]
            x2, y2 = geom_1.points[1]
            x3, y3 = geom_2.points[0]
            x4, y4 = geom_2.points[1]
        elif intersection_type == "segment3D":
            x1, y1, z1 = geom_1.points[0]
            x2, y2, z2 = geom_1.points[1]
            x3, y3, z3 = geom_2.points[0]
            x4, y4, z4 = geom_2.points[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom:
            t1 = x1 * y2 - y1 * x2
            t2 = x3 * y4 - x4 * y3
            return (S(t1 * (x3 - x4) - t2 * (x1 - x2)) / denom,
                    S(t1 * (y3 - y4) - t2 * (y1 - y2)) / denom)
    if intersection_type[:-2] == "plane":
        if intersection_type == "plane2D":  # Intersection of hyperplanes
            a1x, a1y = geom_1[0]
            a2x, a2y = geom_2[0]
            b1, b2 = geom_1[1], geom_2[1]

            denom = a1x * a2y - a2x * a1y
            if denom:
                return (S(b1 * a2y - b2 * a1y) / denom,
                        S(b2 * a1x - b1 * a2x) / denom)


def is_vertex(ent):
    """If the input entity is a vertex return True.

    Parameter
    =========

    ent :
        Denotes a geometric entity representing a point.

    Examples
    ========

    >>> from sympy import Point
    >>> from sympy.integrals.intpoly import is_vertex
    >>> is_vertex((2, 3))
    True
    >>> is_vertex((2, 3, 6))
    True
    >>> is_vertex(Point(2, 3))
    True
    """
    if isinstance(ent, tuple):
        if len(ent) in [2, 3]:
            return True
    elif isinstance(ent, Point):
        return True
    return False


def plot_polytope(poly):
    """Plots the 2D polytope using the functions written in plotting
    module which in turn uses matplotlib backend.

    Parameter
    =========

    poly:
        Denotes a 2-Polytope.
    """
    from sympy.plotting.plot import Plot, List2DSeries

    xl = [vertex.x for vertex in poly.vertices]
    yl = [vertex.y for vertex in poly.vertices]

    xl.append(poly.vertices[0].x)  # Closing the polygon
    yl.append(poly.vertices[0].y)

    l2ds = List2DSeries(xl, yl)
    p = Plot(l2ds, axes='label_axes=True')
    p.show()


def plot_polynomial(expr):
    """Plots the polynomial using the functions written in
    plotting module which in turn uses matplotlib backend.

    Parameter
    =========

    expr:
        Denotes a polynomial(SymPy expression).
    """
    from sympy.plotting.plot import plot3d, plot
    gens = expr.free_symbols
    if len(gens) == 2:
        plot3d(expr)
    else:
        plot(expr)
<|endoftext|>"""
Algorithms for solving the Risch differential equation.

Given a differential field K of characteristic 0 that is a simple
monomial extension of a base field k and f, g in K, the Risch
Differential Equation problem is to decide if there exist y in K such
that Dy + f*y == g and to find one if there are some.  If t is a
monomial over k and the coefficients of f and g are in k(t), then y is
in k(t), and the outline of the algorithm here is given as:

1. Compute the normal part n of the denominator of y.  The problem is
then reduced to finding y' in k<t>, where y == y'/n.
2. Compute the special part s of the denominator of y.   The problem is
then reduced to finding y'' in k[t], where y == y''/(n*s)
3. Bound the degree of y''.
4. Reduce the equation Dy + f*y == g to a similar equation with f, g in
k[t].
5. Find the solutions in k[t] of bounded degree of the reduced equation.

See Chapter 6 of "Symbolic Integration I: Transcendental Functions" by
Manuel Bronstein.  See also the docstring of risch.py.
"""

from operator import mul
from functools import reduce

from sympy.core import oo
from sympy.core.symbol import Dummy

from sympy.polys import Poly, gcd, ZZ, cancel

from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt

from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
    splitfactor, NonElementaryIntegralException, DecrementLevel, recognize_log_derivative)

# TODO: Add messages to NonElementaryIntegralException errors


def order_at(a, p, t):
    """
    Computes the order of a at p, with respect to t.

    Explanation
    ===========

    For a, p in k[t], the order of a at p is defined as nu_p(a) = max({n
    in Z+ such that p**n|a}), where a != 0.  If a == 0, nu_p(a) = +oo.

    To compute the order at a rational function, a/b, use the fact that
    nu_p(a/b) == nu_p(a) - nu_p(b).
    """
    if a.is_zero:
        return oo
    if p == Poly(t, t):
        return a.as_poly(t).ET()[0][0]

    # Uses binary search for calculating the power. power_list collects the tuples
    # (p^k,k) where each k is some power of 2. After deciding the largest k
    # such that k is power of 2 and p^k|a the loop iteratively calculates
    # the actual power.
    power_list = []
    p1 = p
    r = a.rem(p1)
    tracks_power = 1
    while r.is_zero:
        power_list.append((p1,tracks_power))
        p1 = p1*p1
        tracks_power *= 2
        r = a.rem(p1)
    n = 0
    product = Poly(1, t)
    while len(power_list) != 0:
        final = power_list.pop()
        productf = product*final[0]
        r = a.rem(productf)
        if r.is_zero:
            n += final[1]
            product = productf
    return n


def order_at_oo(a, d, t):
    """
    Computes the order of a/d at oo (infinity), with respect to t.

    For f in k(t), the order or f at oo is defined as deg(d) - deg(a), where
    f == a/d.
    """
    if a.is_zero:
        return oo
    return d.degree(t) - a.degree(t)


def weak_normalizer(a, d, DE, z=None):
    """
    Weak normalization.

    Explanation
    ===========

    Given a derivation D on k[t] and f == a/d in k(t), return q in k[t]
    such that f - Dq/q is weakly normalized with respect to t.

    f in k(t) is said to be "weakly normalized" with respect to t if
    residue_p(f) is not a positive integer for any normal irreducible p
    in k[t] such that f is in R_p (Definition 6.1.1).  If f has an
    elementary integral, this is equivalent to no logarithm of
    integral(f) whose argument depends on t has a positive integer
    coefficient, where the arguments of the logarithms not in k(t) are
    in k[t].

    Returns (q, f - Dq/q)
    """
    z = z or Dummy('z')
    dn, ds = splitfactor(d, DE)

    # Compute d1, where dn == d1*d2**2*...*dn**n is a square-free
    # factorization of d.
    g = gcd(dn, dn.diff(DE.t))
    d_sqf_part = dn.quo(g)
    d1 = d_sqf_part.quo(gcd(d_sqf_part, g))

    a1, b = gcdex_diophantine(d.quo(d1).as_poly(DE.t), d1.as_poly(DE.t),
        a.as_poly(DE.t))
    r = (a - Poly(z, DE.t)*derivation(d1, DE)).as_poly(DE.t).resultant(
        d1.as_poly(DE.t))
    r = Poly(r, z)

    if not r.expr.has(z):
        return (Poly(1, DE.t), (a, d))

    N = [i for i in r.real_roots() if i in ZZ and i > 0]

    q = reduce(mul, [gcd(a - Poly(n, DE.t)*derivation(d1, DE), d1) for n in N],
        Poly(1, DE.t))

    dq = derivation(q, DE)
    sn = q*a - d*dq
    sd = q*d
    sn, sd = sn.cancel(sd, include=True)

    return (q, (sn, sd))


def normal_denom(fa, fd, ga, gd, DE):
    """
    Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g in k(t) with f weakly
    normalized with respect to t, either raise NonElementaryIntegralException,
    in which case the equation Dy + f*y == g has no solution in k(t), or the
    quadruplet (a, b, c, h) such that a, h in k[t], b, c in k<t>, and for any
    solution y in k(t) of Dy + f*y == g, q = y*h in k<t> satisfies
    a*Dq + b*q == c.

    This constitutes step 1 in the outline given in the rde.py docstring.
    """
    dn, ds = splitfactor(fd, DE)
    en, es = splitfactor(gd, DE)

    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    a = dn*h
    c = a*h
    if c.div(en)[1]:
        # en does not divide dn*h**2
        raise NonElementaryIntegralException
    ca = c*ga
    ca, cd = ca.cancel(gd, include=True)

    ba = a*fa - dn*derivation(h, DE)*fd
    ba, bd = ba.cancel(fd, include=True)

    # (dn*h, dn*h*f - dn*Dh, dn*h**2*g, h)
    return (a, (ba, bd), (ca, cd), h)


def special_denom(a, ba, bd, ca, cd, DE, case='auto'):
    """
    Special part of the denominator.

    Explanation
    ===========

    case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the
    hyperexponential (resp. hypertangent) case, given a derivation D on
    k[t] and a in k[t], b, c, in k<t> with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the quadruplet (A, B, C, 1/h) such that
    A, B, C, h in k[t] and for any solution q in k<t> of a*Dq + b*q == c,
    r = qh in k[t] satisfies A*Dr + B*r == C.

    For ``case == 'primitive'``, k<t> == k[t], so it returns (a, b, c, 1) in
    this case.

    This constitutes step 2 of the outline given in the rde.py docstring.
    """
    # TODO: finish writing this and write tests

    if case == 'auto':
        case = DE.case

    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t**2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        B = ba.to_field().quo(bd)
        C = ca.to_field().quo(cd)
        return (a, B, C, Poly(1, DE.t))
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base'}, not %s." % case)

    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = order_at(ca, p, DE.t) - order_at(cd, p, DE.t)

    n = min(0, nc - min(0, nb))
    if not nb:
        # Possible cancellation.
        from .prde import parametric_log_deriv
        if case == 'exp':
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                alphaa, alphad = frac_in(-ba.eval(0)/bd.eval(0)/a.eval(0), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    Q, m, z = A
                    if Q == 1:
                        n = min(n, m)

        elif case == 'tan':
            dcoeff = DE.d.quo(Poly(DE.t**2+1, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                alphaa, alphad = frac_in(im(-ba.eval(sqrt(-1))/bd.eval(sqrt(-1))/a.eval(sqrt(-1))), DE.t)
                betaa, betad = frac_in(re(-ba.eval(sqrt(-1))/bd.eval(sqrt(-1))/a.eval(sqrt(-1))), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)

                if recognize_log_derivative(Poly(2, DE.t)*betaa, betad, DE):
                    A = parametric_log_deriv(alphaa*Poly(sqrt(-1), DE.t)*betad+alphad*betaa, alphad*betad, etaa, etad, DE)
                    if A is not None:
                        Q, m, z = A
                        if Q == 1:
                            n = min(n, m)
    N = max(0, -nb, n - nc)
    pN = p**N
    pn = p**-n

    A = a*pN
    B = ba*pN.quo(bd) + Poly(n, DE.t)*a*derivation(p, DE).quo(p)*pN
    C = (ca*pN*pn).quo(cd)
    h = pn

    # (a*p**N, (b + n*a*Dp/p)*p**N, c*p**(N - n), p**-n)
    return (A, B, C, h)


def bound_degree(a, b, cQ, DE, case='auto', parametric=False):
    """
    Bound on polynomial solutions.

    Explanation
    ===========

    Given a derivation D on k[t] and ``a``, ``b``, ``c`` in k[t] with ``a != 0``, return
    n in ZZ such that deg(q) <= n for any solution q in k[t] of
    a*Dq + b*q == c, when parametric=False, or deg(q) <= n for any solution
    c1, ..., cm in Const(k) and q in k[t] of a*Dq + b*q == Sum(ci*gi, (i, 1, m))
    when parametric=True.

    For ``parametric=False``, ``cQ`` is ``c``, a ``Poly``; for ``parametric=True``, ``cQ`` is Q ==
    [q1, ..., qm], a list of Polys.

    This constitutes step 3 of the outline given in the rde.py docstring.
    """
    # TODO: finish writing this and write tests

    if case == 'auto':
        case = DE.case

    da = a.degree(DE.t)
    db = b.degree(DE.t)

    # The parametric and regular cases are identical, except for this part
    if parametric:
        dc = max(i.degree(DE.t) for i in cQ)
    else:
        dc = cQ.degree(DE.t)

    alpha = cancel(-b.as_poly(DE.t).LC().as_expr()/
        a.as_poly(DE.t).LC().as_expr())

    if case == 'base':
        n = max(0, dc - max(db, da - 1))
        if db == da - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)

    elif case == 'primitive':
        if db > da:
            n = max(0, dc - db)
        else:
            n = max(0, dc - da + 1)

        etaa, etad = frac_in(DE.d, DE.T[DE.level - 1])

        t1 = DE.t
        with DecrementLevel(DE):
            alphaa, alphad = frac_in(alpha, DE.t)
            if db == da - 1:
                from .prde import limited_integrate
                # if alpha == m*Dt + Dz for z in k and m in ZZ:
                try:
                    (za, zd), m = limited_integrate(alphaa, alphad, [(etaa, etad)],
                        DE)
                except NonElementaryIntegralException:
                    pass
                else:
                    if len(m) != 1:
                        raise ValueError("Length of m should be 1")
                    n = max(n, m[0])

            elif db == da:
                # if alpha == Dz/z for z in k*:
                    # beta = -lc(a*Dz + b*z)/(z*lc(a))
                    # if beta == m*Dt + Dw for w in k and m in ZZ:
                        # n = max(n, m)
                from .prde import is_log_deriv_k_t_radical_in_field
                A = is_log_deriv_k_t_radical_in_field(alphaa, alphad, DE)
                if A is not None:
                    aa, z = A
                    if aa == 1:
                        beta = -(a*derivation(z, DE).as_poly(t1) +
                            b*z.as_poly(t1)).LC()/(z.as_expr()*a.LC())
                        betaa, betad = frac_in(beta, DE.t)
                        from .prde import limited_integrate
                        try:
                            (za, zd), m = limited_integrate(betaa, betad,
                                [(etaa, etad)], DE)
                        except NonElementaryIntegralException:
                            pass
                        else:
                            if len(m) != 1:
                                raise ValueError("Length of m should be 1")
                            n = max(n, m[0].as_expr())

    elif case == 'exp':
        from .prde import parametric_log_deriv

        n = max(0, dc - max(db, da))
        if da == db:
            etaa, etad = frac_in(DE.d.quo(Poly(DE.t, DE.t)), DE.T[DE.level - 1])
            with DecrementLevel(DE):
                alphaa, alphad = frac_in(alpha, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    # if alpha == m*Dt/t + Dz/z for z in k* and m in ZZ:
                        # n = max(n, m)
                    a, m, z = A
                    if a == 1:
                        n = max(n, m)

    elif case in ('tan', 'other_nonlinear'):
        delta = DE.d.degree(DE.t)
        lam = DE.d.LC()
        alpha = cancel(alpha/lam)
        n = max(0, dc - max(da + delta - 1, db))
        if db == da + delta - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)

    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'other_nonlinear', 'base'}, not %s." % case)

    return n


def spde(a, b, c, n, DE):
    """
    Rothstein's Special Polynomial Differential Equation algorithm.

    Explanation
    ===========

    Given a derivation D on k[t], an integer n and ``a``,``b``,``c`` in k[t] with
    ``a != 0``, either raise NonElementaryIntegralException, in which case the
    equation a*Dq + b*q == c has no solution of degree at most ``n`` in
    k[t], or return the tuple (B, C, m, alpha, beta) such that B, C,
    alpha, beta in k[t], m in ZZ, and any solution q in k[t] of degree
    at most n of a*Dq + b*q == c must be of the form
    q == alpha*h + beta, where h in k[t], deg(h) <= m, and Dh + B*h == C.

    This constitutes step 4 of the outline given in the rde.py docstring.
    """
    zero = Poly(0, DE.t)

    alpha = Poly(1, DE.t)
    beta = Poly(0, DE.t)

    while True:
        if c.is_zero:
            return (zero, zero, 0, zero, beta)  # -1 is more to the point
        if (n < 0) is True:
            raise NonElementaryIntegralException

        g = a.gcd(b)
        if not c.rem(g).is_zero:  # g does not divide c
            raise NonElementaryIntegralException

        a, b, c = a.quo(g), b.quo(g), c.quo(g)

        if a.degree(DE.t) == 0:
            b = b.to_field().quo(a)
            c = c.to_field().quo(a)
            return (b, c, n, alpha, beta)

        r, z = gcdex_diophantine(b, a, c)
        b += derivation(a, DE)
        c = z - derivation(r, DE)
        n -= a.degree(DE.t)

        beta += alpha * r
        alpha *= a

def no_cancel_b_large(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) large enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with ``b != 0`` and either D == d/dt or
    deg(b) > max(0, deg(D) - 1), either raise NonElementaryIntegralException, in
    which case the equation ``Dq + b*q == c`` has no solution of degree at
    most n in k[t], or a solution q in k[t] of this equation with
    ``deg(q) < n``.
    """
    q = Poly(0, DE.t)

    while not c.is_zero:
        m = c.degree(DE.t) - b.degree(DE.t)
        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        p = Poly(c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()*DE.t**m, DE.t,
            expand=False)
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


def no_cancel_b_small(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with deg(b) < deg(D) - 1 and either D == d/dt or
    deg(D) >= 2, either raise NonElementaryIntegralException, in which case the
    equation Dq + b*q == c has no solution of degree at most n in k[t],
    or a solution q in k[t] of this equation with deg(q) <= n, or the
    tuple (h, b0, c0) such that h in k[t], b0, c0, in k, and for any
    solution q in k[t] of degree at most n of Dq + bq == c, y == q - h
    is a solution in k of Dy + b0*y == c0.
    """
    q = Poly(0, DE.t)

    while not c.is_zero:
        if n == 0:
            m = 0
        else:
            m = c.degree(DE.t) - DE.d.degree(DE.t) + 1

        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        if m > 0:
            p = Poly(c.as_poly(DE.t).LC()/(m*DE.d.as_poly(DE.t).LC())*DE.t**m,
                DE.t, expand=False)
        else:
            if b.degree(DE.t) != c.degree(DE.t):
                raise NonElementaryIntegralException
            if b.degree(DE.t) == 0:
                return (q, b.as_poly(DE.T[DE.level - 1]),
                    c.as_poly(DE.T[DE.level - 1]))
            p = Poly(c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC(), DE.t,
                expand=False)

        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


# TODO: better name for this function
def no_cancel_equal(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) == deg(D) - 1

    Explanation
    ===========

    Given a derivation D on k[t] with deg(D) >= 2, n either an integer
    or +oo, and b, c in k[t] with deg(b) == deg(D) - 1, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c has
    no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n, or the tuple (h, m, C) such that h
    in k[t], m in ZZ, and C in k[t], and for any solution q in k[t] of
    degree at most n of Dq + b*q == c, y == q - h is a solution in k[t]
    of degree at most m of Dy + b*y == C.
    """
    q = Poly(0, DE.t)
    lc = cancel(-b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC())
    if lc.is_Integer and lc.is_positive:
        M = lc
    else:
        M = -1

    while not c.is_zero:
        m = max(M, c.degree(DE.t) - DE.d.degree(DE.t) + 1)

        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        u = cancel(m*DE.d.as_poly(DE.t).LC() + b.as_poly(DE.t).LC())
        if u.is_zero:
            return (q, m, c)
        if m > 0:
            p = Poly(c.as_poly(DE.t).LC()/u*DE.t**m, DE.t, expand=False)
        else:
            if c.degree(DE.t) != DE.d.degree(DE.t) - 1:
                raise NonElementaryIntegralException
            else:
                p = c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()

        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


def cancel_primitive(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Primitive case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.
    """
    # Delayed imports
    from .prde import is_log_deriv_k_t_radical_in_field
    with DecrementLevel(DE):
        ba, bd = frac_in(b, DE.t)
        A = is_log_deriv_k_t_radical_in_field(ba, bd, DE)
        if A is not None:
            n, z = A
            if n == 1:  # b == Dz/z
                raise NotImplementedError("is_deriv_in_field() is required to "
                    " solve this problem.")
                # if z*c == Dp for p in k[t] and deg(p) <= n:
                #     return p/z
                # else:
                #     raise NonElementaryIntegralException

    if c.is_zero:
        return c  # return 0

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException

    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        with DecrementLevel(DE):
            a2a, a2d = frac_in(c.LC(), DE.t)
            sa, sd = rischDE(ba, bd, a2a, a2d, DE)
        stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b*stm + derivation(stm, DE)

    return q


def cancel_exp(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Hyperexponential case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt/t in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.
    """
    from .prde import parametric_log_deriv
    eta = DE.d.quo(Poly(DE.t, DE.t)).as_expr()

    with DecrementLevel(DE):
        etaa, etad = frac_in(eta, DE.t)
        ba, bd = frac_in(b, DE.t)
        A = parametric_log_deriv(ba, bd, etaa, etad, DE)
        if A is not None:
            a, m, z = A
            if a == 1:
                raise NotImplementedError("is_deriv_in_field() is required to "
                    "solve this problem.")
                # if c*z*t**m == Dp for p in k<t> and q = p/(z*t**m) in k[t] and
                # deg(q) <= n:
                #     return q
                # else:
                #     raise NonElementaryIntegralException

    if c.is_zero:
        return c  # return 0

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException

    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        # a1 = b + m*Dt/t
        a1 = b.as_expr()
        with DecrementLevel(DE):
            # TODO: Write a dummy function that does this idiom
            a1a, a1d = frac_in(a1, DE.t)
            a1a = a1a*etad + etaa*a1d*Poly(m, DE.t)
            a1d = a1d*etad

            a2a, a2d = frac_in(c.LC(), DE.t)

            sa, sd = rischDE(a1a, a1d, a2a, a2d, DE)
        stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b*stm + derivation(stm, DE)  # deg(c) becomes smaller
    return q


def solve_poly_rde(b, cQ, n, DE, parametric=False):
    """
    Solve a Polynomial Risch Differential Equation with degree bound ``n``.

    This constitutes step 4 of the outline given in the rde.py docstring.

    For parametric=False, cQ is c, a Poly; for parametric=True, cQ is Q ==
    [q1, ..., qm], a list of Polys.
    """
    # No cancellation
    if not b.is_zero and (DE.case == 'base' or
            b.degree(DE.t) > max(0, DE.d.degree(DE.t) - 1)):

        if parametric:
            # Delayed imports
            from .prde import prde_no_cancel_b_large
            return prde_no_cancel_b_large(b, cQ, n, DE)
        return no_cancel_b_large(b, cQ, n, DE)

    elif (b.is_zero or b.degree(DE.t) < DE.d.degree(DE.t) - 1) and \
            (DE.case == 'base' or DE.d.degree(DE.t) >= 2):

        if parametric:
            from .prde import prde_no_cancel_b_small
            return prde_no_cancel_b_small(b, cQ, n, DE)

        R = no_cancel_b_small(b, cQ, n, DE)

        if isinstance(R, Poly):
            return R
        else:
            # XXX: Might k be a field? (pg. 209)
            h, b0, c0 = R
            with DecrementLevel(DE):
                b0, c0 = b0.as_poly(DE.t), c0.as_poly(DE.t)
                if b0 is None:  # See above comment
                    raise ValueError("b0 should be a non-Null value")
                if c0 is  None:
                    raise ValueError("c0 should be a non-Null value")
                y = solve_poly_rde(b0, c0, n, DE).as_poly(DE.t)
            return h + y

    elif DE.d.degree(DE.t) >= 2 and b.degree(DE.t) == DE.d.degree(DE.t) - 1 and \
            n > -b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC():

        # TODO: Is this check necessary, and if so, what should it do if it fails?
        # b comes from the first element returned from spde()
        if not b.as_poly(DE.t).LC().is_number:
            raise TypeError("Result should be a number")

        if parametric:
            raise NotImplementedError("prde_no_cancel_b_equal() is not yet "
                "implemented.")

        R = no_cancel_equal(b, cQ, n, DE)

        if isinstance(R, Poly):
            return R
        else:
            h, m, C = R
            # XXX: Or should it be rischDE()?
            y = solve_poly_rde(b, C, m, DE)
            return h + y

    else:
        # Cancellation
        if b.is_zero:
            raise NotImplementedError("Remaining cases for Poly (P)RDE are "
            "not yet implemented (is_deriv_in_field() required).")
        else:
            if DE.case == 'exp':
                if parametric:
                    raise NotImplementedError("Parametric RDE cancellation "
                        "hyperexponential case is not yet implemented.")
                return cancel_exp(b, cQ, n, DE)

            elif DE.case == 'primitive':
                if parametric:
                    raise NotImplementedError("Parametric RDE cancellation "
                        "primitive case is not yet implemented.")
                return cancel_primitive(b, cQ, n, DE)

            else:
                raise NotImplementedError("Other Poly (P)RDE cancellation "
                    "cases are not yet implemented (%s)." % DE.case)

        if parametric:
            raise NotImplementedError("Remaining cases for Poly PRDE not yet "
                "implemented.")
        raise NotImplementedError("Remaining cases for Poly RDE not yet "
            "implemented.")


def rischDE(fa, fd, ga, gd, DE):
    """
    Solve a Risch Differential Equation: Dy + f*y == g.

    Explanation
    ===========

    See the outline in the docstring of rde.py for more information
    about the procedure used.  Either raise NonElementaryIntegralException, in
    which case there is no solution y in the given differential field,
    or return y in k(t) satisfying Dy + f*y == g, or raise
    NotImplementedError, in which case, the algorithms necessary to
    solve the given Risch Differential Equation have not yet been
    implemented.
    """
    _, (fa, fd) = weak_normalizer(fa, fd, DE)
    a, (ba, bd), (ca, cd), hn = normal_denom(fa, fd, ga, gd, DE)
    A, B, C, hs = special_denom(a, ba, bd, ca, cd, DE)
    try:
        # Until this is fully implemented, use oo.  Note that this will almost
        # certainly cause non-termination in spde() (unless A == 1), and
        # *might* lead to non-termination in the next step for a nonelementary
        # integral (I don't know for certain yet).  Fortunately, spde() is
        # currently written recursively, so this will just give
        # RuntimeError: maximum recursion depth exceeded.
        n = bound_degree(A, B, C, DE)
    except NotImplementedError:
        # Useful for debugging:
        # import warnings
        # warnings.warn("rischDE: Proceeding with n = oo; may cause "
        #     "non-termination.")
        n = oo

    B, C, m, alpha, beta = spde(A, B, C, n, DE)
    if C.is_zero:
        y = C
    else:
        y = solve_poly_rde(B, C, m, DE)

    return (alpha*y + beta, hn*hs)
<|endoftext|>from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf


def gauss_legendre(n, n_digits):
    r"""
    Computes the Gauss-Legendre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Legendre quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2}{\left(1-x_i^2\right) \left(P'_n(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_legendre
    >>> x, w = gauss_legendre(3, 5)
    >>> x
    [-0.7746, 0, 0.7746]
    >>> w
    [0.55556, 0.88889, 0.55556]
    >>> x, w = gauss_legendre(4, 5)
    >>> x
    [-0.86114, -0.33998, 0.33998, 0.86114]
    >>> w
    [0.34785, 0.65215, 0.65215, 0.34785]

    See Also
    ========

    gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/legendre_rule/legendre_rule.html
    """
    x = Dummy("x")
    p = legendre_poly(n, x, polys=True)
    pd = p.diff(x)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((2/((1-r**2) * pd.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_laguerre(n, n_digits):
    r"""
    Computes the Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \int_0^{\infty} e^{-x} f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)


    The nodes `x_i` of an order `n` quadrature rule are the roots of `L_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{x_i}{(n+1)^2 \left(L_{n+1}(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : The ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_laguerre
    >>> x, w = gauss_laguerre(3, 5)
    >>> x
    [0.41577, 2.2943, 6.2899]
    >>> w
    [0.71109, 0.27852, 0.010389]
    >>> x, w = gauss_laguerre(6, 5)
    >>> x
    [0.22285, 1.1889, 2.9927, 5.7751, 9.8375, 15.983]
    >>> w
    [0.45896, 0.417, 0.11337, 0.010399, 0.00026102, 8.9855e-7]

    See Also
    ========

    gauss_legendre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/laguerre_rule/laguerre_rule.html
    """
    x = Dummy("x")
    p = laguerre_poly(n, x, polys=True)
    p1 = laguerre_poly(n+1, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((r/((n+1)**2 * p1.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_hermite(n, n_digits):
    r"""
    Computes the Gauss-Hermite quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Hermite quadrature approximates the integral:

    .. math::
        \int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `H_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2^{n-1} n! \sqrt{\pi}}{n^2 \left(H_{n-1}(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : The ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_hermite
    >>> x, w = gauss_hermite(3, 5)
    >>> x
    [-1.2247, 0, 1.2247]
    >>> w
    [0.29541, 1.1816, 0.29541]

    >>> x, w = gauss_hermite(6, 5)
    >>> x
    [-2.3506, -1.3358, -0.43608, 0.43608, 1.3358, 2.3506]
    >>> w
    [0.00453, 0.15707, 0.72463, 0.72463, 0.15707, 0.00453]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_gen_laguerre, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss-Hermite_Quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/hermite_rule/hermite_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gen_hermite_rule/gen_hermite_rule.html
    """
    x = Dummy("x")
    p = hermite_poly(n, x, polys=True)
    p1 = hermite_poly(n-1, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append(((2**(n-1) * factorial(n) * sqrt(pi)) /
                 (n**2 * p1.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_gen_laguerre(n, alpha, n_digits):
    r"""
    Computes the generalized Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The generalized Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \int_{0}^\infty x^{\alpha} e^{-x} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `L^{\alpha}_n` and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\Gamma(\alpha+n)}
                {n \Gamma(n) L^{\alpha}_{n-1}(x_i) L^{\alpha+1}_{n-1}(x_i)}

    Parameters
    ==========

    n :
        The order of quadrature.

    alpha :
        The exponent of the singularity, `\alpha > -1`.

    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_gen_laguerre
    >>> x, w = gauss_gen_laguerre(3, -S.Half, 5)
    >>> x
    [0.19016, 1.7845, 5.5253]
    >>> w
    [1.4493, 0.31413, 0.00906]

    >>> x, w = gauss_gen_laguerre(4, 3*S.Half, 5)
    >>> x
    [0.97851, 2.9904, 6.3193, 11.712]
    >>> w
    [0.53087, 0.67721, 0.11895, 0.0023152]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/gen_laguerre_rule/gen_laguerre_rule.html
    """
    x = Dummy("x")
    p = laguerre_poly(n, x, alpha=alpha, polys=True)
    p1 = laguerre_poly(n-1, x, alpha=alpha, polys=True)
    p2 = laguerre_poly(n-1, x, alpha=alpha+1, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((gamma(alpha+n) /
                 (n*gamma(n)*p1.subs(x, r)*p2.subs(x, r))).n(n_digits))
    return xi, w


def gauss_chebyshev_t(n, n_digits):
    r"""
    Computes the Gauss-Chebyshev quadrature [1]_ points and weights of
    the first kind.

    Explanation
    ===========

    The Gauss-Chebyshev quadrature of the first kind approximates the integral:

    .. math::
        \int_{-1}^{1} \frac{1}{\sqrt{1-x^2}} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `T_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\pi}{n}

    Parameters
    ==========

    n :
        The order of quadrature.

    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_chebyshev_t
    >>> x, w = gauss_chebyshev_t(3, 5)
    >>> x
    [0.86602, 0, -0.86602]
    >>> w
    [1.0472, 1.0472, 1.0472]

    >>> x, w = gauss_chebyshev_t(6, 5)
    >>> x
    [0.96593, 0.70711, 0.25882, -0.25882, -0.70711, -0.96593]
    >>> w
    [0.5236, 0.5236, 0.5236, 0.5236, 0.5236, 0.5236]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/chebyshev1_rule/chebyshev1_rule.html
    """
    xi = []
    w = []
    for i in range(1, n+1):
        xi.append((cos((2*i-S.One)/(2*n)*S.Pi)).n(n_digits))
        w.append((S.Pi/n).n(n_digits))
    return xi, w


def gauss_chebyshev_u(n, n_digits):
    r"""
    Computes the Gauss-Chebyshev quadrature [1]_ points and weights of
    the second kind.

    Explanation
    ===========

    The Gauss-Chebyshev quadrature of the second kind approximates the
    integral:

    .. math::
        \int_{-1}^{1} \sqrt{1-x^2} f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `U_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\pi}{n+1} \sin^2 \left(\frac{i}{n+1}\pi\right)

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_chebyshev_u
    >>> x, w = gauss_chebyshev_u(3, 5)
    >>> x
    [0.70711, 0, -0.70711]
    >>> w
    [0.3927, 0.7854, 0.3927]

    >>> x, w = gauss_chebyshev_u(6, 5)
    >>> x
    [0.90097, 0.62349, 0.22252, -0.22252, -0.62349, -0.90097]
    >>> w
    [0.084489, 0.27433, 0.42658, 0.42658, 0.27433, 0.084489]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre, gauss_chebyshev_t, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/chebyshev2_rule/chebyshev2_rule.html
    """
    xi = []
    w = []
    for i in range(1, n+1):
        xi.append((cos(i/(n+S.One)*S.Pi)).n(n_digits))
        w.append((S.Pi/(n+S.One)*sin(i*S.Pi/(n+S.One))**2).n(n_digits))
    return xi, w


def gauss_jacobi(n, alpha, beta, n_digits):
    r"""
    Computes the Gauss-Jacobi quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Jacobi quadrature of the first kind approximates the integral:

    .. math::
        \int_{-1}^1 (1-x)^\alpha (1+x)^\beta f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `P^{(\alpha,\beta)}_n` and the weights `w_i` are given by:

    .. math::
        w_i = -\frac{2n+\alpha+\beta+2}{n+\alpha+\beta+1}
              \frac{\Gamma(n+\alpha+1)\Gamma(n+\beta+1)}
              {\Gamma(n+\alpha+\beta+1)(n+1)!}
              \frac{2^{\alpha+\beta}}{P'_n(x_i)
              P^{(\alpha,\beta)}_{n+1}(x_i)}

    Parameters
    ==========

    n : the order of quadrature

    alpha : the first parameter of the Jacobi Polynomial, `\alpha > -1`

    beta : the second parameter of the Jacobi Polynomial, `\beta > -1`

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_jacobi
    >>> x, w = gauss_jacobi(3, S.Half, -S.Half, 5)
    >>> x
    [-0.90097, -0.22252, 0.62349]
    >>> w
    [1.7063, 1.0973, 0.33795]

    >>> x, w = gauss_jacobi(6, 1, 1, 5)
    >>> x
    [-0.87174, -0.5917, -0.2093, 0.2093, 0.5917, 0.87174]
    >>> w
    [0.050584, 0.22169, 0.39439, 0.39439, 0.22169, 0.050584]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre,
    gauss_chebyshev_t, gauss_chebyshev_u, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_rule/jacobi_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gegenbauer_rule/gegenbauer_rule.html
    """
    x = Dummy("x")
    p = jacobi_poly(n, alpha, beta, x, polys=True)
    pd = p.diff(x)
    pn = jacobi_poly(n+1, alpha, beta, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((
            - (2*n+alpha+beta+2) / (n+alpha+beta+S.One) *
            (gamma(n+alpha+1)*gamma(n+beta+1)) /
            (gamma(n+alpha+beta+S.One)*gamma(n+2)) *
            2**(alpha+beta) / (pd.subs(x, r) * pn.subs(x, r))).n(n_digits))
    return xi, w


def gauss_lobatto(n, n_digits):
    r"""
    Computes the Gauss-Lobatto quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Lobatto quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P'_(n-1)`
    and the weights `w_i` are given by:

    .. math::
        &w_i = \frac{2}{n(n-1) \left[P_{n-1}(x_i)\right]^2},\quad x\neq\pm 1\\
        &w_i = \frac{2}{n(n-1)},\quad x=\pm 1

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_lobatto
    >>> x, w = gauss_lobatto(3, 5)
    >>> x
    [-1, 0, 1]
    >>> w
    [0.33333, 1.3333, 0.33333]
    >>> x, w = gauss_lobatto(4, 5)
    >>> x
    [-1, -0.44721, 0.44721, 1]
    >>> w
    [0.16667, 0.83333, 0.83333, 0.16667]

    See Also
    ========

    gauss_legendre,gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
    .. [2] https://web.archive.org/web/20200118141346/http://people.math.sfu.ca/~cbm/aands/page_888.htm
    """
    x = Dummy("x")
    p = legendre_poly(n-1, x, polys=True)
    pd = p.diff(x)
    xi = []
    w = []
    for r in pd.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((2/(n*(n-1) * p.subs(x, r)**2)).n(n_digits))

    xi.insert(0, -1)
    xi.append(1)
    w.insert(0, (S(2)/(n*(n-1))).n(n_digits))
    w.append((S(2)/(n*(n-1))).n(n_digits))
    return xi, w
<|endoftext|>"""Integration functions that integrate a SymPy expression.

    Examples
    ========

    >>> from sympy import integrate, sin
    >>> from sympy.abc import x
    >>> integrate(1/x,x)
    log(x)
    >>> integrate(sin(x),x)
    -cos(x)
"""
from .integrals import integrate, Integral, line_integrate
from .transforms import (mellin_transform, inverse_mellin_transform,
                        MellinTransform, InverseMellinTransform,
                        laplace_transform, inverse_laplace_transform,
                        laplace_correspondence, laplace_initial_conds,
                        LaplaceTransform, InverseLaplaceTransform,
                        fourier_transform, inverse_fourier_transform,
                        FourierTransform, InverseFourierTransform,
                        sine_transform, inverse_sine_transform,
                        SineTransform, InverseSineTransform,
                        cosine_transform, inverse_cosine_transform,
                        CosineTransform, InverseCosineTransform,
                        hankel_transform, inverse_hankel_transform,
                        HankelTransform, InverseHankelTransform)
from .singularityfunctions import singularityintegrate

__all__ = [
    'integrate', 'Integral', 'line_integrate',

    'mellin_transform', 'inverse_mellin_transform', 'MellinTransform',
    'InverseMellinTransform', 'laplace_transform',
    'inverse_laplace_transform', 'LaplaceTransform',
    'laplace_correspondence', 'laplace_initial_conds',
    'InverseLaplaceTransform', 'fourier_transform',
    'inverse_fourier_transform', 'FourierTransform',
    'InverseFourierTransform', 'sine_transform', 'inverse_sine_transform',
    'SineTransform', 'InverseSineTransform', 'cosine_transform',
    'inverse_cosine_transform', 'CosineTransform', 'InverseCosineTransform',
    'hankel_transform', 'inverse_hankel_transform', 'HankelTransform',
    'InverseHankelTransform',

    'singularityintegrate',
]
<|endoftext|>"""
Integrate functions by rewriting them as Meijer G-functions.

There are three user-visible functions that can be used by other parts of the
sympy library to solve various integration problems:

- meijerint_indefinite
- meijerint_definite
- meijerint_inversion

They can be used to compute, respectively, indefinite integrals, definite
integrals over intervals of the real line, and inverse laplace-type integrals
(from c-I*oo to c+I*oo). See the respective docstrings for details.

The main references for this are:

[L] Luke, Y. L. (1969), The Special Functions and Their Approximations,
    Volume 1

[R] Kelly B. Roach.  Meijer G Function Representations.
    In: Proceedings of the 1997 International Symposium on Symbolic and
    Algebraic Computation, pages 205-211, New York, 1997. ACM.

[P] A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    Integrals and Series: More Special Functions, Vol. 3,.
    Gordon and Breach Science Publisher
"""

from __future__ import annotations
import itertools

from sympy import SYMPY_DEBUG
from sympy.core import S, Expr
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand, expand_mul, expand_power_base,
                                 expand_trig, Function)
from sympy.core.mul import Mul
from sympy.core.intfunc import ilcm
from sympy.core.numbers import Rational, pi
from sympy.core.relational import Eq, Ne, _canonical_coeff
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, symbols, Wild, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (re, im, arg, Abs, sign,
        unpolarify, polarify, polar_lift, principal_branch, unbranched_argument,
        periodic_argument)
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.hyperbolic import (cosh, sinh,
        _rewrite_hyperbolics_as_exp, HyperbolicFunction)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc,
        TrigonometricFunction)
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e
from sympy.functions.special.error_functions import (erf, erfc, erfi, Ei,
        expint, Si, Ci, Shi, Chi, fresnels, fresnelc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.functions.special.singularity_functions import SingularityFunction
from .integrals import Integral
from sympy.logic.boolalg import And, Or, BooleanAtom, Not, BooleanFunction
from sympy.polys import cancel, factor
from sympy.utilities.iterables import multiset_partitions
from sympy.utilities.misc import debug as _debug
from sympy.utilities.misc import debugf as _debugf

# keep this at top for easy reference
z = Dummy('z')


def _has(res, *f):
    # return True if res has f; in the case of Piecewise
    # only return True if *all* pieces have f
    res = piecewise_fold(res)
    if getattr(res, 'is_Piecewise', False):
        return all(_has(i, *f) for i in res.args)
    return res.has(*f)


def _create_lookup_table(table):
    """ Add formulae for the function -> meijerg lookup table. """
    def wild(n):
        return Wild(n, exclude=[z])
    p, q, a, b, c = list(map(wild, 'pqabc'))
    n = Wild('n', properties=[lambda x: x.is_Integer and x > 0])
    t = p*z**q

    def add(formula, an, ap, bm, bq, arg=t, fac=S.One, cond=True, hint=True):
        table.setdefault(_mytype(formula, z), []).append((formula,
                                     [(fac, meijerg(an, ap, bm, bq, arg))], cond, hint))

    def addi(formula, inst, cond, hint=True):
        table.setdefault(
            _mytype(formula, z), []).append((formula, inst, cond, hint))

    def constant(a):
        return [(a, meijerg([1], [], [], [0], z)),
                (a, meijerg([], [1], [0], [], z))]
    table[()] = [(a, constant(a), True, True)]

    # [P], Section 8.
    class IsNonPositiveInteger(Function):

        @classmethod
        def eval(cls, arg):
            arg = unpolarify(arg)
            if arg.is_Integer is True:
                return arg <= 0

    # Section 8.4.2
    # TODO this needs more polar_lift (c/f entry for exp)
    add(Heaviside(t - b)*(t - b)**(a - 1), [a], [], [], [0], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside(b - t)*(b - t)**(a - 1), [], [a], [0], [], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside(z - (b/p)**(1/q))*(t - b)**(a - 1), [a], [], [], [0], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside((b/p)**(1/q) - z)*(b - t)**(a - 1), [], [a], [0], [], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add((b + t)**(-a), [1 - a], [], [0], [], t/b, b**(-a)/gamma(a),
        hint=Not(IsNonPositiveInteger(a)))
    add(Abs(b - t)**(-a), [1 - a], [(1 - a)/2], [0], [(1 - a)/2], t/b,
        2*sin(pi*a/2)*gamma(1 - a)*Abs(b)**(-a), re(a) < 1)
    add((t**a - b**a)/(t - b), [0, a], [], [0, a], [], t/b,
        b**(a - 1)*sin(a*pi)/pi)

    # 12
    def A1(r, sign, nu):
        return pi**Rational(-1, 2)*(-sign*nu/2)**(1 - 2*r)

    def tmpadd(r, sgn):
        # XXX the a**2 is bad for matching
        add((sqrt(a**2 + t) + sgn*a)**b/(a**2 + t)**r,
            [(1 + b)/2, 1 - 2*r + b/2], [],
            [(b - sgn*b)/2], [(b + sgn*b)/2], t/a**2,
            a**(b - 2*r)*A1(r, sgn, b))
    tmpadd(0, 1)
    tmpadd(0, -1)
    tmpadd(S.Half, 1)
    tmpadd(S.Half, -1)

    # 13
    def tmpadd(r, sgn):
        add((sqrt(a + p*z**q) + sgn*sqrt(p)*z**(q/2))**b/(a + p*z**q)**r,
            [1 - r + sgn*b/2], [1 - r - sgn*b/2], [0, S.Half], [],
            p*z**q/a, a**(b/2 - r)*A1(r, sgn, b))
    tmpadd(0, 1)
    tmpadd(0, -1)
    tmpadd(S.Half, 1)
    tmpadd(S.Half, -1)
    # (those after look obscure)

    # Section 8.4.3
    add(exp(polar_lift(-1)*t), [], [], [0], [])

    # TODO can do sin^n, sinh^n by expansion ... where?
    # 8.4.4 (hyperbolic functions)
    add(sinh(t), [], [1], [S.Half], [1, 0], t**2/4, pi**Rational(3, 2))
    add(cosh(t), [], [S.Half], [0], [S.Half, S.Half], t**2/4, pi**Rational(3, 2))

    # Section 8.4.5
    # TODO can do t + a. but can also do by expansion... (XXX not really)
    add(sin(t), [], [], [S.Half], [0], t**2/4, sqrt(pi))
    add(cos(t), [], [], [0], [S.Half], t**2/4, sqrt(pi))

    # Section 8.4.6 (sinc function)
    add(sinc(t), [], [], [0], [Rational(-1, 2)], t**2/4, sqrt(pi)/2)

    # Section 8.5.5
    def make_log1(subs):
        N = subs[n]
        return [(S.NegativeOne**N*factorial(N),
                 meijerg([], [1]*(N + 1), [0]*(N + 1), [], t))]

    def make_log2(subs):
        N = subs[n]
        return [(factorial(N),
                 meijerg([1]*(N + 1), [], [], [0]*(N + 1), t))]
    # TODO these only hold for positive p, and can be made more general
    #      but who uses log(x)*Heaviside(a-x) anyway ...
    # TODO also it would be nice to derive them recursively ...
    addi(log(t)**n*Heaviside(1 - t), make_log1, True)
    addi(log(t)**n*Heaviside(t - 1), make_log2, True)

    def make_log3(subs):
        return make_log1(subs) + make_log2(subs)
    addi(log(t)**n, make_log3, True)
    addi(log(t + a),
         constant(log(a)) + [(S.One, meijerg([1, 1], [], [1], [0], t/a))],
         True)
    addi(log(Abs(t - a)), constant(log(Abs(a))) +
         [(pi, meijerg([1, 1], [S.Half], [1], [0, S.Half], t/a))],
         True)
    # TODO log(x)/(x+a) and log(x)/(x-1) can also be done. should they
    #      be derivable?
    # TODO further formulae in this section seem obscure

    # Sections 8.4.9-10
    # TODO

    # Section 8.4.11
    addi(Ei(t),
         constant(-S.ImaginaryUnit*pi) + [(S.NegativeOne, meijerg([], [1], [0, 0], [],
                  t*polar_lift(-1)))],
         True)

    # Section 8.4.12
    add(Si(t), [1], [], [S.Half], [0, 0], t**2/4, sqrt(pi)/2)
    add(Ci(t), [], [1], [0, 0], [S.Half], t**2/4, -sqrt(pi)/2)

    # Section 8.4.13
    add(Shi(t), [S.Half], [], [0], [Rational(-1, 2), Rational(-1, 2)], polar_lift(-1)*t**2/4,
        t*sqrt(pi)/4)
    add(Chi(t), [], [S.Half, 1], [0, 0], [S.Half, S.Half], t**2/4, -
        pi**S('3/2')/2)

    # generalized exponential integral
    add(expint(a, t), [], [a], [a - 1, 0], [], t)

    # Section 8.4.14
    add(erf(t), [1], [], [S.Half], [0], t**2, 1/sqrt(pi))
    # TODO exp(-x)*erf(I*x) does not work
    add(erfc(t), [], [1], [0, S.Half], [], t**2, 1/sqrt(pi))
    # This formula for erfi(z) yields a wrong(?) minus sign
    #add(erfi(t), [1], [], [S.Half], [0], -t**2, I/sqrt(pi))
    add(erfi(t), [S.Half], [], [0], [Rational(-1, 2)], -t**2, t/sqrt(pi))

    # Fresnel Integrals
    add(fresnels(t), [1], [], [Rational(3, 4)], [0, Rational(1, 4)], pi**2*t**4/16, S.Half)
    add(fresnelc(t), [1], [], [Rational(1, 4)], [0, Rational(3, 4)], pi**2*t**4/16, S.Half)

    ##### bessel-type functions #####
    # Section 8.4.19
    add(besselj(a, t), [], [], [a/2], [-a/2], t**2/4)

    # all of the following are derivable
    #add(sin(t)*besselj(a, t), [Rational(1, 4), Rational(3, 4)], [], [(1+a)/2],
    #    [-a/2, a/2, (1-a)/2], t**2, 1/sqrt(2))
    #add(cos(t)*besselj(a, t), [Rational(1, 4), Rational(3, 4)], [], [a/2],
    #    [-a/2, (1+a)/2, (1-a)/2], t**2, 1/sqrt(2))
    #add(besselj(a, t)**2, [S.Half], [], [a], [-a, 0], t**2, 1/sqrt(pi))
    #add(besselj(a, t)*besselj(b, t), [0, S.Half], [], [(a + b)/2],
    #    [-(a+b)/2, (a - b)/2, (b - a)/2], t**2, 1/sqrt(pi))

    # Section 8.4.20
    add(bessely(a, t), [], [-(a + 1)/2], [a/2, -a/2], [-(a + 1)/2], t**2/4)

    # TODO all of the following should be derivable
    #add(sin(t)*bessely(a, t), [Rational(1, 4), Rational(3, 4)], [(1 - a - 1)/2],
    #    [(1 + a)/2, (1 - a)/2], [(1 - a - 1)/2, (1 - 1 - a)/2, (1 - 1 + a)/2],
    #    t**2, 1/sqrt(2))
    #add(cos(t)*bessely(a, t), [Rational(1, 4), Rational(3, 4)], [(0 - a - 1)/2],
    #    [(0 + a)/2, (0 - a)/2], [(0 - a - 1)/2, (1 - 0 - a)/2, (1 - 0 + a)/2],
    #    t**2, 1/sqrt(2))
    #add(besselj(a, t)*bessely(b, t), [0, S.Half], [(a - b - 1)/2],
    #    [(a + b)/2, (a - b)/2], [(a - b - 1)/2, -(a + b)/2, (b - a)/2],
    #    t**2, 1/sqrt(pi))
    #addi(bessely(a, t)**2,
    #     [(2/sqrt(pi), meijerg([], [S.Half, S.Half - a], [0, a, -a],
    #                           [S.Half - a], t**2)),
    #      (1/sqrt(pi), meijerg([S.Half], [], [a], [-a, 0], t**2))],
    #     True)
    #addi(bessely(a, t)*bessely(b, t),
    #     [(2/sqrt(pi), meijerg([], [0, S.Half, (1 - a - b)/2],
    #                           [(a + b)/2, (a - b)/2, (b - a)/2, -(a + b)/2],
    #                           [(1 - a - b)/2], t**2)),
    #      (1/sqrt(pi), meijerg([0, S.Half], [], [(a + b)/2],
    #                           [-(a + b)/2, (a - b)/2, (b - a)/2], t**2))],
    #     True)

    # Section 8.4.21 ?
    # Section 8.4.22
    add(besseli(a, t), [], [(1 + a)/2], [a/2], [-a/2, (1 + a)/2], t**2/4, pi)
    # TODO many more formulas. should all be derivable

    # Section 8.4.23
    add(besselk(a, t), [], [], [a/2, -a/2], [], t**2/4, S.Half)
    # TODO many more formulas. should all be derivable

    # Complete elliptic integrals K(z) and E(z)
    add(elliptic_k(t), [S.Half, S.Half], [], [0], [0], -t, S.Half)
    add(elliptic_e(t), [S.Half, 3*S.Half], [], [0], [0], -t, Rational(-1, 2)/2)


####################################################################
# First some helper functions.
####################################################################

from sympy.utilities.timeutils import timethis
timeit = timethis('meijerg')


def _mytype(f: Basic, x: Symbol) -> tuple[type[Basic], ...]:
    """ Create a hashable entity describing the type of f. """
    def key(x: type[Basic]) -> tuple[int, int, str]:
        return x.class_key()

    if x not in f.free_symbols:
        return ()
    elif f.is_Function:
        return type(f),
    return tuple(sorted((t for a in f.args for t in _mytype(a, x)), key=key))


class _CoeffExpValueError(ValueError):
    """
    Exception raised by _get_coeff_exp, for internal use only.
    """
    pass


def _get_coeff_exp(expr, x):
    """
    When expr is known to be of the form c*x**b, with c and/or b possibly 1,
    return c, b.

    Examples
    ========

    >>> from sympy.abc import x, a, b
    >>> from sympy.integrals.meijerint import _get_coeff_exp
    >>> _get_coeff_exp(a*x**b, x)
    (a, b)
    >>> _get_coeff_exp(x, x)
    (1, 1)
    >>> _get_coeff_exp(2*x, x)
    (2, 1)
    >>> _get_coeff_exp(x**3, x)
    (1, 3)
    """
    from sympy.simplify import powsimp
    (c, m) = expand_power_base(powsimp(expr)).as_coeff_mul(x)
    if not m:
        return c, S.Zero
    [m] = m
    if m.is_Pow:
        if m.base != x:
            raise _CoeffExpValueError('expr not of form a*x**b')
        return c, m.exp
    elif m == x:
        return c, S.One
    else:
        raise _CoeffExpValueError('expr not of form a*x**b: %s' % expr)


def _exponents(expr, x):
    """
    Find the exponents of ``x`` (not including zero) in ``expr``.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _exponents
    >>> from sympy.abc import x, y
    >>> from sympy import sin
    >>> _exponents(x, x)
    {1}
    >>> _exponents(x**2, x)
    {2}
    >>> _exponents(x**2 + x, x)
    {1, 2}
    >>> _exponents(x**3*sin(x + x**y) + 1/x, x)
    {-1, 1, 3, y}
    """
    def _exponents_(expr, x, res):
        if expr == x:
            res.update([1])
            return
        if expr.is_Pow and expr.base == x:
            res.update([expr.exp])
            return
        for argument in expr.args:
            _exponents_(argument, x, res)
    res = set()
    _exponents_(expr, x, res)
    return res


def _functions(expr, x):
    """ Find the types of functions in expr, to estimate the complexity. """
    return {e.func for e in expr.atoms(Function) if x in e.free_symbols}


def _find_splitting_points(expr, x):
    """
    Find numbers a such that a linear substitution x -> x + a would
    (hopefully) simplify expr.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _find_splitting_points as fsp
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> fsp(x, x)
    {0}
    >>> fsp((x-1)**3, x)
    {1}
    >>> fsp(sin(x+3)*x, x)
    {-3, 0}
    """
    p, q = [Wild(n, exclude=[x]) for n in 'pq']

    def compute_innermost(expr, res):
        if not isinstance(expr, Expr):
            return
        m = expr.match(p*x + q)
        if m and m[p] != 0:
            res.add(-m[q]/m[p])
            return
        if expr.is_Atom:
            return
        for argument in expr.args:
            compute_innermost(argument, res)
    innermost = set()
    compute_innermost(expr, innermost)
    return innermost


def _split_mul(f, x):
    """
    Split expression ``f`` into fac, po, g, where fac is a constant factor,
    po = x**s for some s independent of s, and g is "the rest".

    Examples
    ========

    >>> from sympy.integrals.meijerint import _split_mul
    >>> from sympy import sin
    >>> from sympy.abc import s, x
    >>> _split_mul((3*x)**s*sin(x**2)*x, x)
    (3**s, x*x**s, sin(x**2))
    """
    fac = S.One
    po = S.One
    g = S.One
    f = expand_power_base(f)

    args = Mul.make_args(f)
    for a in args:
        if a == x:
            po *= x
        elif x not in a.free_symbols:
            fac *= a
        else:
            if a.is_Pow and x not in a.exp.free_symbols:
                c, t = a.base.as_coeff_mul(x)
                if t != (x,):
                    c, t = expand_mul(a.base).as_coeff_mul(x)
                if t == (x,):
                    po *= x**a.exp
                    fac *= unpolarify(polarify(c**a.exp, subs=False))
                    continue
            g *= a

    return fac, po, g


def _mul_args(f):
    """
    Return a list ``L`` such that ``Mul(*L) == f``.

    If ``f`` is not a ``Mul`` or ``Pow``, ``L=[f]``.
    If ``f=g**n`` for an integer ``n``, ``L=[g]*n``.
    If ``f`` is a ``Mul``, ``L`` comes from applying ``_mul_args`` to all factors of ``f``.
    """
    args = Mul.make_args(f)
    gs = []
    for g in args:
        if g.is_Pow and g.exp.is_Integer:
            n = g.exp
            base = g.base
            if n < 0:
                n = -n
                base = 1/base
            gs += [base]*n
        else:
            gs.append(g)
    return gs


def _mul_as_two_parts(f):
    """
    Find all the ways to split ``f`` into a product of two terms.
    Return None on failure.

    Explanation
    ===========

    Although the order is canonical from multiset_partitions, this is
    not necessarily the best order to process the terms. For example,
    if the case of len(gs) == 2 is removed and multiset is allowed to
    sort the terms, some tests fail.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _mul_as_two_parts
    >>> from sympy import sin, exp, ordered
    >>> from sympy.abc import x
    >>> list(ordered(_mul_as_two_parts(x*sin(x)*exp(x))))
    [(x, exp(x)*sin(x)), (x*exp(x), sin(x)), (x*sin(x), exp(x))]
    """

    gs = _mul_args(f)
    if len(gs) < 2:
        return None
    if len(gs) == 2:
        return [tuple(gs)]
    return [(Mul(*x), Mul(*y)) for (x, y) in multiset_partitions(gs, 2)]


def _inflate_g(g, n):
    """ Return C, h such that h is a G function of argument z**n and
        g = C*h. """
    # TODO should this be a method of meijerg?
    # See: [L, page 150, equation (5)]
    def inflate(params, n):
        """ (a1, .., ak) -> (a1/n, (a1+1)/n, ..., (ak + n-1)/n) """
        return [(a + i)/n for a, i in itertools.product(params, range(n))]
    v = S(len(g.ap) - len(g.bq))
    C = n**(1 + g.nu + v/2)
    C /= (2*pi)**((n - 1)*g.delta)
    return C, meijerg(inflate(g.an, n), inflate(g.aother, n),
                      inflate(g.bm, n), inflate(g.bother, n),
                      g.argument**n * n**(n*v))


def _flip_g(g):
    """ Turn the G function into one of inverse argument
        (i.e. G(1/x) -> G'(x)) """
    # See [L], section 5.2
    def tr(l):
        return [1 - a for a in l]
    return meijerg(tr(g.bm), tr(g.bother), tr(g.an), tr(g.aother), 1/g.argument)


def _inflate_fox_h(g, a):
    r"""
    Let d denote the integrand in the definition of the G function ``g``.
    Consider the function H which is defined in the same way, but with
    integrand d/Gamma(a*s) (contour conventions as usual).

    If ``a`` is rational, the function H can be written as C*G, for a constant C
    and a G-function G.

    This function returns C, G.
    """
    if a < 0:
        return _inflate_fox_h(_flip_g(g), -a)
    p = S(a.p)
    q = S(a.q)
    # We use the substitution s->qs, i.e. inflate g by q. We are left with an
    # extra factor of Gamma(p*s), for which we use Gauss' multiplication
    # theorem.
    D, g = _inflate_g(g, q)
    z = g.argument
    D /= (2*pi)**((1 - p)/2)*p**Rational(-1, 2)
    z /= p**p
    bs = [(n + 1)/p for n in range(p)]
    return D, meijerg(g.an, g.aother, g.bm, list(g.bother) + bs, z)


_dummies: dict[tuple[str, str], Dummy]  = {}


def _dummy(name, token, expr, **kwargs):
    """
    Return a dummy. This will return the same dummy if the same token+name is
    requested more than once, and it is not already in expr.
    This is for being cache-friendly.
    """
    d = _dummy_(name, token, **kwargs)
    if d in expr.free_symbols:
        return Dummy(name, **kwargs)
    return d


def _dummy_(name, token, **kwargs):
    """
    Return a dummy associated to name and token. Same effect as declaring
    it globally.
    """
    if not (name, token) in _dummies:
        _dummies[(name, token)] = Dummy(name, **kwargs)
    return _dummies[(name, token)]


def _is_analytic(f, x):
    """ Check if f(x), when expressed using G functions on the positive reals,
        will in fact agree with the G functions almost everywhere """
    return not any(x in expr.free_symbols for expr in f.atoms(Heaviside, Abs))


def _condsimp(cond, first=True):
    """
    Do naive simplifications on ``cond``.

    Explanation
    ===========

    Note that this routine is completely ad-hoc, simplification rules being
    added as need arises rather than following any logical pattern.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _condsimp as simp
    >>> from sympy import Or, Eq
    >>> from sympy.abc import x, y
    >>> simp(Or(x < y, Eq(x, y)))
    x <= y
    """
    if first:
        cond = cond.replace(lambda _: _.is_Relational, _canonical_coeff)
        first = False
    if not isinstance(cond, BooleanFunction):
        return cond
    p, q, r = symbols('p q r', cls=Wild)
    # transforms tests use 0, 4, 5 and 11-14
    # meijer tests use 0, 2, 11, 14
    # joint_rv uses 6, 7
    rules = [
        (Or(p < q, Eq(p, q)), p <= q),  # 0
        # The next two obviously are instances of a general pattern, but it is
        # easier to spell out the few cases we care about.
        (And(Abs(arg(p)) <= pi, Abs(arg(p) - 2*pi) <= pi),
         Eq(arg(p) - pi, 0)),  # 1
        (And(Abs(2*arg(p) + pi) <= pi, Abs(2*arg(p) - pi) <= pi),
         Eq(arg(p), 0)), # 2
        (And(Abs(2*arg(p) + pi) < pi, Abs(2*arg(p) - pi) <= pi),
         S.false),  # 3
        (And(Abs(arg(p) - pi/2) <= pi/2, Abs(arg(p) + pi/2) <= pi/2),
         Eq(arg(p), 0)),  # 4
        (And(Abs(arg(p) - pi/2) <= pi/2, Abs(arg(p) + pi/2) < pi/2),
         S.false),  # 5
        (And(Abs(arg(p**2/2 + 1)) < pi, Ne(Abs(arg(p**2/2 + 1)), pi)),
         S.true),  # 6
        (Or(Abs(arg(p**2/2 + 1)) < pi, Ne(1/(p**2/2 + 1), 0)),
         S.true),  # 7
        (And(Abs(unbranched_argument(p)) <= pi,
           Abs(unbranched_argument(exp_polar(-2*pi*S.ImaginaryUnit)*p)) <= pi),
         Eq(unbranched_argument(exp_polar(-S.ImaginaryUnit*pi)*p), 0)),  # 8
        (And(Abs(unbranched_argument(p)) <= pi/2,
           Abs(unbranched_argument(exp_polar(-pi*S.ImaginaryUnit)*p)) <= pi/2),
         Eq(unbranched_argument(exp_polar(-S.ImaginaryUnit*pi/2)*p), 0)),  # 9
        (Or(p <= q, And(p < q, r)), p <= q),  # 10
        (Ne(p**2, 1) & (p**2 > 1), p**2 > 1),  # 11
        (Ne(1/p, 1) & (cos(Abs(arg(p)))*Abs(p) > 1), Abs(p) > 1),  # 12
        (Ne(p, 2) & (cos(Abs(arg(p)))*Abs(p) > 2), Abs(p) > 2),  # 13
        ((Abs(arg(p)) < pi/2) & (cos(Abs(arg(p)))*sqrt(Abs(p**2)) > 1), p**2 > 1),  # 14
    ]
    cond = cond.func(*[_condsimp(_, first) for _ in cond.args])
    change = True
    while change:
        change = False
        for irule, (fro, to) in enumerate(rules):
            if fro.func != cond.func:
                continue
            for n, arg1 in enumerate(cond.args):
                if r in fro.args[0].free_symbols:
                    m = arg1.match(fro.args[1])
                    num = 1
                else:
                    num = 0
                    m = arg1.match(fro.args[0])
                if not m:
                    continue
                otherargs = [x.subs(m) for x in fro.args[:num] + fro.args[num + 1:]]
                otherlist = [n]
                for arg2 in otherargs:
                    for k, arg3 in enumerate(cond.args):
                        if k in otherlist:
                            continue
                        if arg2 == arg3:
                            otherlist += [k]
                            break
                        if isinstance(arg3, And) and arg2.args[1] == r and \
                                isinstance(arg2, And) and arg2.args[0] in arg3.args:
                            otherlist += [k]
                            break
                        if isinstance(arg3, And) and arg2.args[0] == r and \
                                isinstance(arg2, And) and arg2.args[1] in arg3.args:
                            otherlist += [k]
                            break
                if len(otherlist) != len(otherargs) + 1:
                    continue
                newargs = [arg_ for (k, arg_) in enumerate(cond.args)
                           if k not in otherlist] + [to.subs(m)]
                if SYMPY_DEBUG:
                    if irule not in (0, 2, 4, 5, 6, 7, 11, 12, 13, 14):
                        print('used new rule:', irule)
                cond = cond.func(*newargs)
                change = True
                break

    # final tweak
    def rel_touchup(rel):
        if rel.rel_op != '==' or rel.rhs != 0:
            return rel

        # handle Eq(*, 0)
        LHS = rel.lhs
        m = LHS.match(arg(p)**q)
        if not m:
            m = LHS.match(unbranched_argument(polar_lift(p)**q))
        if not m:
            if isinstance(LHS, periodic_argument) and not LHS.args[0].is_polar \
                    and LHS.args[1] is S.Infinity:
                return (LHS.args[0] > 0)
            return rel
        return (m[p] > 0)
    cond = cond.replace(lambda _: _.is_Relational, rel_touchup)
    if SYMPY_DEBUG:
        print('_condsimp: ', cond)
    return cond

def _eval_cond(cond):
    """ Re-evaluate the conditions. """
    if isinstance(cond, bool):
        return cond
    return _condsimp(cond.doit())

####################################################################
# Now the "backbone" functions to do actual integration.
####################################################################


def _my_principal_branch(expr, period, full_pb=False):
    """ Bring expr nearer to its principal branch by removing superfluous
        factors.
        This function does *not* guarantee to yield the principal branch,
        to avoid introducing opaque principal_branch() objects,
        unless full_pb=True. """
    res = principal_branch(expr, period)
    if not full_pb:
        res = res.replace(principal_branch, lambda x, y: x)
    return res


def _rewrite_saxena_1(fac, po, g, x):
    """
    Rewrite the integral fac*po*g dx, from zero to infinity, as
    integral fac*G, where G has argument a*x. Note po=x**s.
    Return fac, G.
    """
    _, s = _get_coeff_exp(po, x)
    a, b = _get_coeff_exp(g.argument, x)
    period = g.get_period()
    a = _my_principal_branch(a, period)

    # We substitute t = x**b.
    C = fac/(Abs(b)*a**((s + 1)/b - 1))
    # Absorb a factor of (at)**((1 + s)/b - 1).

    def tr(l):
        return [a + (1 + s)/b - 1 for a in l]
    return C, meijerg(tr(g.an), tr(g.aother), tr(g.bm), tr(g.bother),
                      a*x)


def _check_antecedents_1(g, x, helper=False):
    r"""
    Return a condition under which the mellin transform of g exists.
    Any power of x has already been absorbed into the G function,
    so this is just $\int_0^\infty g\, dx$.

    See [L, section 5.6.1]. (Note that s=1.)

    If ``helper`` is True, only check if the MT exists at infinity, i.e. if
    $\int_1^\infty g\, dx$ exists.
    """
    # NOTE if you update these conditions, please update the documentation as well
    delta = g.delta
    eta, _ = _get_coeff_exp(g.argument, x)
    m, n, p, q = S([len(g.bm), len(g.an), len(g.ap), len(g.bq)])

    if p > q:
        def tr(l):
            return [1 - x for x in l]
        return _check_antecedents_1(meijerg(tr(g.bm), tr(g.bother),
                                            tr(g.an), tr(g.aother), x/eta),
                                    x)

    tmp = [-re(b) < 1 for b in g.bm] + [1 < 1 - re(a) for a in g.an]
    cond_3 = And(*tmp)

    tmp += [-re(b) < 1 for b in g.bother]
    tmp += [1 < 1 - re(a) for a in g.aother]
    cond_3_star = And(*tmp)

    cond_4 = (-re(g.nu) + (q + 1 - p)/2 > q - p)

    def debug(*msg):
        _debug(*msg)

    def debugf(string, arg):
        _debugf(string, arg)

    debug('Checking antecedents for 1 function:')
    debugf('  delta=%s, eta=%s, m=%s, n=%s, p=%s, q=%s',
           (delta, eta, m, n, p, q))
    debugf('  ap = %s, %s', (list(g.an), list(g.aother)))
    debugf('  bq = %s, %s', (list(g.bm), list(g.bother)))
    debugf('  cond_3=%s, cond_3*=%s, cond_4=%s', (cond_3, cond_3_star, cond_4))

    conds = []

    # case 1
    case1 = []
    tmp1 = [1 <= n, p < q, 1 <= m]
    tmp2 = [1 <= p, 1 <= m, Eq(q, p + 1), Not(And(Eq(n, 0), Eq(m, p + 1)))]
    tmp3 = [1 <= p, Eq(q, p)]
    for k in range(ceiling(delta/2) + 1):
        tmp3 += [Ne(Abs(unbranched_argument(eta)), (delta - 2*k)*pi)]
    tmp = [delta > 0, Abs(unbranched_argument(eta)) < delta*pi]
    extra = [Ne(eta, 0), cond_3]
    if helper:
        extra = []
    for t in [tmp1, tmp2, tmp3]:
        case1 += [And(*(t + tmp + extra))]
    conds += case1
    debug('  case 1:', case1)

    # case 2
    extra = [cond_3]
    if helper:
        extra = []
    case2 = [And(Eq(n, 0), p + 1 <= m, m <= q,
                 Abs(unbranched_argument(eta)) < delta*pi, *extra)]
    conds += case2
    debug('  case 2:', case2)

    # case 3
    extra = [cond_3, cond_4]
    if helper:
        extra = []
    case3 = [And(p < q, 1 <= m, delta > 0, Eq(Abs(unbranched_argument(eta)), delta*pi),
                 *extra)]
    case3 += [And(p <= q - 2, Eq(delta, 0), Eq(Abs(unbranched_argument(eta)), 0), *extra)]
    conds += case3
    debug('  case 3:', case3)

    # TODO altered cases 4-7

    # extra case from wofram functions site:
    # (reproduced verbatim from Prudnikov, section 2.24.2)
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/01/
    case_extra = []
    case_extra += [Eq(p, q), Eq(delta, 0), Eq(unbranched_argument(eta), 0), Ne(eta, 0)]
    if not helper:
        case_extra += [cond_3]
    s = []
    for a, b in zip(g.ap, g.bq):
        s += [b - a]
    case_extra += [re(Add(*s)) < 0]
    case_extra = And(*case_extra)
    conds += [case_extra]
    debug('  extra case:', [case_extra])

    case_extra_2 = [And(delta > 0, Abs(unbranched_argument(eta)) < delta*pi)]
    if not helper:
        case_extra_2 += [cond_3]
    case_extra_2 = And(*case_extra_2)
    conds += [case_extra_2]
    debug('  second extra case:', [case_extra_2])

    # TODO This leaves only one case from the three listed by Prudnikov.
    #      Investigate if these indeed cover everything; if so, remove the rest.

    return Or(*conds)


def _int0oo_1(g, x):
    r"""
    Evaluate $\int_0^\infty g\, dx$ using G functions,
    assuming the necessary conditions are fulfilled.

    Examples
    ========

    >>> from sympy.abc import a, b, c, d, x, y
    >>> from sympy import meijerg
    >>> from sympy.integrals.meijerint import _int0oo_1
    >>> _int0oo_1(meijerg([a], [b], [c], [d], x*y), x)
    gamma(-a)*gamma(c + 1)/(y*gamma(-d)*gamma(b + 1))
    """
    from sympy.simplify import gammasimp
    # See [L, section 5.6.1]. Note that s=1.
    eta, _ = _get_coeff_exp(g.argument, x)
    res = 1/eta
    # XXX TODO we should reduce order first
    for b in g.bm:
        res *= gamma(b + 1)
    for a in g.an:
        res *= gamma(1 - a - 1)
    for b in g.bother:
        res /= gamma(1 - b - 1)
    for a in g.aother:
        res /= gamma(a + 1)
    return gammasimp(unpolarify(res))


def _rewrite_saxena(fac, po, g1, g2, x, full_pb=False):
    """
    Rewrite the integral ``fac*po*g1*g2`` from 0 to oo in terms of G
    functions with argument ``c*x``.

    Explanation
    ===========

    Return C, f1, f2 such that integral C f1 f2 from 0 to infinity equals
    integral fac ``po``, ``g1``, ``g2`` from 0 to infinity.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _rewrite_saxena
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg
    >>> g1 = meijerg([], [], [0], [], s*t)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t**2/4)
    >>> r = _rewrite_saxena(1, t**0, g1, g2, t)
    >>> r[0]
    s/(4*sqrt(pi))
    >>> r[1]
    meijerg(((), ()), ((-1/2, 0), ()), s**2*t/4)
    >>> r[2]
    meijerg(((), ()), ((m/2,), (-m/2,)), t/4)
    """
    def pb(g):
        a, b = _get_coeff_exp(g.argument, x)
        per = g.get_period()
        return meijerg(g.an, g.aother, g.bm, g.bother,
                       _my_principal_branch(a, per, full_pb)*x**b)

    _, s = _get_coeff_exp(po, x)
    _, b1 = _get_coeff_exp(g1.argument, x)
    _, b2 = _get_coeff_exp(g2.argument, x)
    if (b1 < 0) == True:
        b1 = -b1
        g1 = _flip_g(g1)
    if (b2 < 0) == True:
        b2 = -b2
        g2 = _flip_g(g2)
    if not b1.is_Rational or not b2.is_Rational:
        return
    m1, n1 = b1.p, b1.q
    m2, n2 = b2.p, b2.q
    tau = ilcm(m1*n2, m2*n1)
    r1 = tau//(m1*n2)
    r2 = tau//(m2*n1)

    C1, g1 = _inflate_g(g1, r1)
    C2, g2 = _inflate_g(g2, r2)
    g1 = pb(g1)
    g2 = pb(g2)

    fac *= C1*C2
    a1, b = _get_coeff_exp(g1.argument, x)
    a2, _ = _get_coeff_exp(g2.argument, x)

    # arbitrarily tack on the x**s part to g1
    # TODO should we try both?
    exp = (s + 1)/b - 1
    fac = fac/(Abs(b) * a1**exp)

    def tr(l):
        return [a + exp for a in l]
    g1 = meijerg(tr(g1.an), tr(g1.aother), tr(g1.bm), tr(g1.bother), a1*x)
    g2 = meijerg(g2.an, g2.aother, g2.bm, g2.bother, a2*x)

    from sympy.simplify import powdenest
    return powdenest(fac, polar=True), g1, g2


def _check_antecedents(g1, g2, x):
    """ Return a condition under which the integral theorem applies. """
    #  Yes, this is madness.
    # XXX TODO this is a testing *nightmare*
    # NOTE if you update these conditions, please update the documentation as well

    # The following conditions are found in
    # [P], Section 2.24.1
    #
    # They are also reproduced (verbatim!) at
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/03/
    #
    # Note: k=l=r=alpha=1
    sigma, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)
    s, t, u, v = S([len(g1.bm), len(g1.an), len(g1.ap), len(g1.bq)])
    m, n, p, q = S([len(g2.bm), len(g2.an), len(g2.ap), len(g2.bq)])
    bstar = s + t - (u + v)/2
    cstar = m + n - (p + q)/2
    rho = g1.nu + (u - v)/2 + 1
    mu = g2.nu + (p - q)/2 + 1
    phi = q - p - (v - u)
    eta = 1 - (v - u) - mu - rho
    psi = (pi*(q - m - n) + Abs(unbranched_argument(omega)))/(q - p)
    theta = (pi*(v - s - t) + Abs(unbranched_argument(sigma)))/(v - u)

    _debug('Checking antecedents:')
    _debugf('  sigma=%s, s=%s, t=%s, u=%s, v=%s, b*=%s, rho=%s',
            (sigma, s, t, u, v, bstar, rho))
    _debugf('  omega=%s, m=%s, n=%s, p=%s, q=%s, c*=%s, mu=%s,',
            (omega, m, n, p, q, cstar, mu))
    _debugf('  phi=%s, eta=%s, psi=%s, theta=%s', (phi, eta, psi, theta))

    def _c1():
        for g in [g1, g2]:
            for i, j in itertools.product(g.an, g.bm):
                diff = i - j
                if diff.is_integer and diff.is_positive:
                    return False
        return True
    c1 = _c1()
    c2 = And(*[re(1 + i + j) > 0 for i in g1.bm for j in g2.bm])
    c3 = And(*[re(1 + i + j) < 1 + 1 for i in g1.an for j in g2.an])
    c4 = And(*[(p - q)*re(1 + i - 1) - re(mu) > Rational(-3, 2) for i in g1.an])
    c5 = And(*[(p - q)*re(1 + i) - re(mu) > Rational(-3, 2) for i in g1.bm])
    c6 = And(*[(u - v)*re(1 + i - 1) - re(rho) > Rational(-3, 2) for i in g2.an])
    c7 = And(*[(u - v)*re(1 + i) - re(rho) > Rational(-3, 2) for i in g2.bm])
    c8 = (Abs(phi) + 2*re((rho - 1)*(q - p) + (v - u)*(q - p) + (mu -
          1)*(v - u)) > 0)
    c9 = (Abs(phi) - 2*re((rho - 1)*(q - p) + (v - u)*(q - p) + (mu -
          1)*(v - u)) > 0)
    c10 = (Abs(unbranched_argument(sigma)) < bstar*pi)
    c11 = Eq(Abs(unbranched_argument(sigma)), bstar*pi)
    c12 = (Abs(unbranched_argument(omega)) < cstar*pi)
    c13 = Eq(Abs(unbranched_argument(omega)), cstar*pi)

    # The following condition is *not* implemented as stated on the wolfram
    # function site. In the book of Prudnikov there is an additional part
    # (the And involving re()). However, I only have this book in russian, and
    # I don't read any russian. The following condition is what other people
    # have told me it means.
    # Worryingly, it is different from the condition implemented in REDUCE.
    # The REDUCE implementation:
    #   https://reduce-algebra.svn.sourceforge.net/svnroot/reduce-algebra/trunk/packages/defint/definta.red
    #   (search for tst14)
    # The Wolfram alpha version:
    #   https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/03/03/0014/
    z0 = exp(-(bstar + cstar)*pi*S.ImaginaryUnit)
    zos = unpolarify(z0*omega/sigma)
    zso = unpolarify(z0*sigma/omega)
    if zos == 1/zso:
        c14 = And(Eq(phi, 0), bstar + cstar <= 1,
                  Or(Ne(zos, 1), re(mu + rho + v - u) < 1,
                     re(mu + rho + q - p) < 1))
    else:
        def _cond(z):
            '''Returns True if abs(arg(1-z)) < pi, avoiding arg(0).

            Explanation
            ===========

            If ``z`` is 1 then arg is NaN. This raises a
            TypeError on `NaN < pi`. Previously this gave `False` so
            this behavior has been hardcoded here but someone should
            check if this NaN is more serious! This NaN is triggered by
            test_meijerint() in test_meijerint.py:
            `meijerint_definite(exp(x), x, 0, I)`
            '''
            return z != 1 and Abs(arg(1 - z)) < pi

        c14 = And(Eq(phi, 0), bstar - 1 + cstar <= 0,
                  Or(And(Ne(zos, 1), _cond(zos)),
                     And(re(mu + rho + v - u) < 1, Eq(zos, 1))))

        c14_alt = And(Eq(phi, 0), cstar - 1 + bstar <= 0,
                  Or(And(Ne(zso, 1), _cond(zso)),
                     And(re(mu + rho + q - p) < 1, Eq(zso, 1))))

        # Since r=k=l=1, in our case there is c14_alt which is the same as calling
        # us with (g1, g2) = (g2, g1). The conditions below enumerate all cases
        # (i.e. we don't have to try arguments reversed by hand), and indeed try
        # all symmetric cases. (i.e. whenever there is a condition involving c14,
        # there is also a dual condition which is exactly what we would get when g1,
        # g2 were interchanged, *but c14 was unaltered*).
        # Hence the following seems correct:
        c14 = Or(c14, c14_alt)

    '''
    When `c15` is NaN (e.g. from `psi` being NaN as happens during
    'test_issue_4992' and/or `theta` is NaN as in 'test_issue_6253',
    both in `test_integrals.py`) the comparison to 0 formerly gave False
    whereas now an error is raised. To keep the old behavior, the value
    of NaN is replaced with False but perhaps a closer look at this condition
    should be made: XXX how should conditions leading to c15=NaN be handled?
    '''
    try:
        lambda_c = (q - p)*Abs(omega)**(1/(q - p))*cos(psi) \
            + (v - u)*Abs(sigma)**(1/(v - u))*cos(theta)
        # the TypeError might be raised here, e.g. if lambda_c is NaN
        if _eval_cond(lambda_c > 0) != False:
            c15 = (lambda_c > 0)
        else:
            def lambda_s0(c1, c2):
                return c1*(q - p)*Abs(omega)**(1/(q - p))*sin(psi) \
                    + c2*(v - u)*Abs(sigma)**(1/(v - u))*sin(theta)
            lambda_s = Piecewise(
                ((lambda_s0(+1, +1)*lambda_s0(-1, -1)),
                 And(Eq(unbranched_argument(sigma), 0), Eq(unbranched_argument(omega), 0))),
                (lambda_s0(sign(unbranched_argument(omega)), +1)*lambda_s0(sign(unbranched_argument(omega)), -1),
                 And(Eq(unbranched_argument(sigma), 0), Ne(unbranched_argument(omega), 0))),
                (lambda_s0(+1, sign(unbranched_argument(sigma)))*lambda_s0(-1, sign(unbranched_argument(sigma))),
                 And(Ne(unbranched_argument(sigma), 0), Eq(unbranched_argument(omega), 0))),
                (lambda_s0(sign(unbranched_argument(omega)), sign(unbranched_argument(sigma))), True))
            tmp = [lambda_c > 0,
                   And(Eq(lambda_c, 0), Ne(lambda_s, 0), re(eta) > -1),
                   And(Eq(lambda_c, 0), Eq(lambda_s, 0), re(eta) > 0)]
            c15 = Or(*tmp)
    except TypeError:
        c15 = False
    for cond, i in [(c1, 1), (c2, 2), (c3, 3), (c4, 4), (c5, 5), (c6, 6),
                    (c7, 7), (c8, 8), (c9, 9), (c10, 10), (c11, 11),
                    (c12, 12), (c13, 13), (c14, 14), (c15, 15)]:
        _debugf('  c%s: %s', (i, cond))

    # We will return Or(*conds)
    conds = []

    def pr(count):
        _debugf('  case %s: %s', (count, conds[-1]))
    conds += [And(m*n*s*t != 0, bstar.is_positive is True, cstar.is_positive is True, c1, c2, c3, c10,
                  c12)]  # 1
    pr(1)
    conds += [And(Eq(u, v), Eq(bstar, 0), cstar.is_positive is True, sigma.is_positive is True, re(rho) < 1,
                  c1, c2, c3, c12)]  # 2
    pr(2)
    conds += [And(Eq(p, q), Eq(cstar, 0), bstar.is_positive is True, omega.is_positive is True, re(mu) < 1,
                  c1, c2, c3, c10)]  # 3
    pr(3)
    conds += [And(Eq(p, q), Eq(u, v), Eq(bstar, 0), Eq(cstar, 0),
                  sigma.is_positive is True, omega.is_positive is True, re(mu) < 1, re(rho) < 1,
                  Ne(sigma, omega), c1, c2, c3)]  # 4
    pr(4)
    conds += [And(Eq(p, q), Eq(u, v), Eq(bstar, 0), Eq(cstar, 0),
                  sigma.is_positive is True, omega.is_positive is True, re(mu + rho) < 1,
                  Ne(omega, sigma), c1, c2, c3)]  # 5
    pr(5)
    conds += [And(p > q, s.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  c1, c2, c3, c5, c10, c13)]  # 6
    pr(6)
    conds += [And(p < q, t.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  c1, c2, c3, c4, c10, c13)]  # 7
    pr(7)
    conds += [And(u > v, m.is_positive is True, cstar.is_positive is True, bstar >= 0,
                  c1, c2, c3, c7, c11, c12)]  # 8
    pr(8)
    conds += [And(u < v, n.is_positive is True, cstar.is_positive is True, bstar >= 0,
                  c1, c2, c3, c6, c11, c12)]  # 9
    pr(9)
    conds += [And(p > q, Eq(u, v), Eq(bstar, 0), cstar >= 0, sigma.is_positive is True,
                  re(rho) < 1, c1, c2, c3, c5, c13)]  # 10
    pr(10)
    conds += [And(p < q, Eq(u, v), Eq(bstar, 0), cstar >= 0, sigma.is_positive is True,
                  re(rho) < 1, c1, c2, c3, c4, c13)]  # 11
    pr(11)
    conds += [And(Eq(p, q), u > v, bstar >= 0, Eq(cstar, 0), omega.is_positive is True,
                  re(mu) < 1, c1, c2, c3, c7, c11)]  # 12
    pr(12)
    conds += [And(Eq(p, q), u < v, bstar >= 0, Eq(cstar, 0), omega.is_positive is True,
                  re(mu) < 1, c1, c2, c3, c6, c11)]  # 13
    pr(13)
    conds += [And(p < q, u > v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c4, c7, c11, c13)]  # 14
    pr(14)
    conds += [And(p > q, u < v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c5, c6, c11, c13)]  # 15
    pr(15)
    conds += [And(p > q, u > v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c5, c7, c8, c11, c13, c14)]  # 16
    pr(16)
    conds += [And(p < q, u < v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c4, c6, c9, c11, c13, c14)]  # 17
    pr(17)
    conds += [And(Eq(t, 0), s.is_positive is True, bstar.is_positive is True, phi.is_positive is True, c1, c2, c10)]  # 18
    pr(18)
    conds += [And(Eq(s, 0), t.is_positive is True, bstar.is_positive is True, phi.is_negative is True, c1, c3, c10)]  # 19
    pr(19)
    conds += [And(Eq(n, 0), m.is_positive is True, cstar.is_positive is True, phi.is_negative is True, c1, c2, c12)]  # 20
    pr(20)
    conds += [And(Eq(m, 0), n.is_positive is True, cstar.is_positive is True, phi.is_positive is True, c1, c3, c12)]  # 21
    pr(21)
    conds += [And(Eq(s*t, 0), bstar.is_positive is True, cstar.is_positive is True,
                  c1, c2, c3, c10, c12)]  # 22
    pr(22)
    conds += [And(Eq(m*n, 0), bstar.is_positive is True, cstar.is_positive is True,
                  c1, c2, c3, c10, c12)]  # 23
    pr(23)

    # The following case is from [Luke1969]. As far as I can tell, it is *not*
    # covered by Prudnikov's.
    # Let G1 and G2 be the two G-functions. Suppose the integral exists from
    # 0 to a > 0 (this is easy the easy part), that G1 is exponential decay at
    # infinity, and that the mellin transform of G2 exists.
    # Then the integral exists.
    mt1_exists = _check_antecedents_1(g1, x, helper=True)
    mt2_exists = _check_antecedents_1(g2, x, helper=True)
    conds += [And(mt2_exists, Eq(t, 0), u < s, bstar.is_positive is True, c10, c1, c2, c3)]
    pr('E1')
    conds += [And(mt2_exists, Eq(s, 0), v < t, bstar.is_positive is True, c10, c1, c2, c3)]
    pr('E2')
    conds += [And(mt1_exists, Eq(n, 0), p < m, cstar.is_positive is True, c12, c1, c2, c3)]
    pr('E3')
    conds += [And(mt1_exists, Eq(m, 0), q < n, cstar.is_positive is True, c12, c1, c2, c3)]
    pr('E4')

    # Let's short-circuit if this worked ...
    # the rest is corner-cases and terrible to read.
    r = Or(*conds)
    if _eval_cond(r) != False:
        return r

    conds += [And(m + n > p, Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True, cstar.is_negative is True,
                  Abs(unbranched_argument(omega)) < (m + n - p + 1)*pi,
                  c1, c2, c10, c14, c15)]  # 24
    pr(24)
    conds += [And(m + n > q, Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True, cstar.is_negative is True,
                  Abs(unbranched_argument(omega)) < (m + n - q + 1)*pi,
                  c1, c3, c10, c14, c15)]  # 25
    pr(25)
    conds += [And(Eq(p, q - 1), Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  c1, c2, c10, c14, c15)]  # 26
    pr(26)
    conds += [And(Eq(p, q + 1), Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  c1, c3, c10, c14, c15)]  # 27
    pr(27)
    conds += [And(p < q - 1, Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  Abs(unbranched_argument(omega)) < (m + n - p + 1)*pi,
                  c1, c2, c10, c14, c15)]  # 28
    pr(28)
    conds += [And(
        p > q + 1, Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  cstar*pi < Abs(unbranched_argument(omega)),
                  Abs(unbranched_argument(omega)) < (m + n - q + 1)*pi,
                  c1, c3, c10, c14, c15)]  # 29
    pr(29)
    conds += [And(Eq(n, 0), Eq(phi, 0), s + t > 0, m.is_positive is True, cstar.is_positive is True, bstar.is_negative is True,
                  Abs(unbranched_argument(sigma)) < (s + t - u + 1)*pi,
                  c1, c2, c12, c14, c15)]  # 30
    pr(30)
    conds += [And(Eq(m, 0), Eq(phi, 0), s + t > v, n.is_positive is True, cstar.is_positive is True, bstar.is_negative is True,
                  Abs(unbranched_argument(sigma)) < (s + t - v + 1)*pi,
                  c1, c3, c12, c14, c15)]  # 31
    pr(31)
    conds += [And(Eq(n, 0), Eq(phi, 0), Eq(u, v - 1), m.is_positive is True, cstar.is_positive is True,
                  bstar >= 0, bstar*pi < Abs(unbranched_argument(sigma)),
                  Abs(unbranched_argument(sigma)) < (bstar + 1)*pi,
                  c1, c2, c12, c14, c15)]  # 32
    pr(32)
    conds += [And(Eq(m, 0), Eq(phi, 0), Eq(u, v + 1), n.is_positive is True, cstar.is_positive is True,
                  bstar >= 0, bstar*pi < Abs(unbranched_argument(sigma)),
                  Abs(unbranched_argument(sigma)) < (bstar + 1)*pi,
                  c1, c3, c12, c14, c15)]  # 33
    pr(33)
    conds += [And(
        Eq(n, 0), Eq(phi, 0), u < v - 1, m.is_positive is True, cstar.is_positive is True, bstar >= 0,
        bstar*pi < Abs(unbranched_argument(sigma)),
        Abs(unbranched_argument(sigma)) < (s + t - u + 1)*pi,
        c1, c2, c12, c14, c15)]  # 34
    pr(34)
    conds += [And(
        Eq(m, 0), Eq(phi, 0), u > v + 1, n.is_positive is True, cstar.is_positive is True, bstar >= 0,
        bstar*pi < Abs(unbranched_argument(sigma)),
        Abs(unbranched_argument(sigma)) < (s + t - v + 1)*pi,
        c1, c3, c12, c14, c15)]  # 35
    pr(35)

    return Or(*conds)

    # NOTE An alternative, but as far as I can tell weaker, set of conditions
    #      can be found in [L, section 5.6.2].


def _int0oo(g1, g2, x):
    """
    Express integral from zero to infinity g1*g2 using a G function,
    assuming the necessary conditions are fulfilled.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _int0oo
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg, S
    >>> g1 = meijerg([], [], [-S(1)/2, 0], [], s**2*t/4)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t/4)
    >>> _int0oo(g1, g2, t)
    4*meijerg(((0, 1/2), ()), ((m/2,), (-m/2,)), s**(-2))/s**2
    """
    # See: [L, section 5.6.2, equation (1)]
    eta, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)

    def neg(l):
        return [-x for x in l]
    a1 = neg(g1.bm) + list(g2.an)
    a2 = list(g2.aother) + neg(g1.bother)
    b1 = neg(g1.an) + list(g2.bm)
    b2 = list(g2.bother) + neg(g1.aother)
    return meijerg(a1, a2, b1, b2, omega/eta)/eta


def _rewrite_inversion(fac, po, g, x):
    """ Absorb ``po`` == x**s into g. """
    _, s = _get_coeff_exp(po, x)
    a, b = _get_coeff_exp(g.argument, x)

    def tr(l):
        return [t + s/b for t in l]
    from sympy.simplify import powdenest
    return (powdenest(fac/a**(s/b), polar=True),
            meijerg(tr(g.an), tr(g.aother), tr(g.bm), tr(g.bother), g.argument))


def _check_antecedents_inversion(g, x):
    """ Check antecedents for the laplace inversion integral. """
    _debug('Checking antecedents for inversion:')
    z = g.argument
    _, e = _get_coeff_exp(z, x)
    if e < 0:
        _debug('  Flipping G.')
        # We want to assume that argument gets large as |x| -> oo
        return _check_antecedents_inversion(_flip_g(g), x)

    def statement_half(a, b, c, z, plus):
        coeff, exponent = _get_coeff_exp(z, x)
        a *= exponent
        b *= coeff**c
        c *= exponent
        conds = []
        wp = b*exp(S.ImaginaryUnit*re(c)*pi/2)
        wm = b*exp(-S.ImaginaryUnit*re(c)*pi/2)
        if plus:
            w = wp
        else:
            w = wm
        conds += [And(Or(Eq(b, 0), re(c) <= 0), re(a) <= -1)]
        conds += [And(Ne(b, 0), Eq(im(c), 0), re(c) > 0, re(w) < 0)]
        conds += [And(Ne(b, 0), Eq(im(c), 0), re(c) > 0, re(w) <= 0,
                      re(a) <= -1)]
        return Or(*conds)

    def statement(a, b, c, z):
        """ Provide a convergence statement for z**a * exp(b*z**c),
             c/f sphinx docs. """
        return And(statement_half(a, b, c, z, True),
                   statement_half(a, b, c, z, False))

    # Notations from [L], section 5.7-10
    m, n, p, q = S([len(g.bm), len(g.an), len(g.ap), len(g.bq)])
    tau = m + n - p
    nu = q - m - n
    rho = (tau - nu)/2
    sigma = q - p
    if sigma == 1:
        epsilon = S.Half
    elif sigma > 1:
        epsilon = 1
    else:
        epsilon = S.NaN
    theta = ((1 - sigma)/2 + Add(*g.bq) - Add(*g.ap))/sigma
    delta = g.delta
    _debugf('  m=%s, n=%s, p=%s, q=%s, tau=%s, nu=%s, rho=%s, sigma=%s',
            (m, n, p, q, tau, nu, rho, sigma))
    _debugf('  epsilon=%s, theta=%s, delta=%s', (epsilon, theta, delta))

    # First check if the computation is valid.
    if not (g.delta >= e/2 or (p >= 1 and p >= q)):
        _debug('  Computation not valid for these parameters.')
        return False

    # Now check if the inversion integral exists.

    # Test "condition A"
    for a, b in itertools.product(g.an, g.bm):
        if (a - b).is_integer and a > b:
            _debug('  Not a valid G function.')
            return False

    # There are two cases. If p >= q, we can directly use a slater expansion
    # like [L], 5.2 (11). Note in particular that the asymptotics of such an
    # expansion even hold when some of the parameters differ by integers, i.e.
    # the formula itself would not be valid! (b/c G functions are cts. in their
    # parameters)
    # When p < q, we need to use the theorems of [L], 5.10.

    if p >= q:
        _debug('  Using asymptotic Slater expansion.')
        return And(*[statement(a - 1, 0, 0, z) for a in g.an])

    def E(z):
        return And(*[statement(a - 1, 0, 0, z) for a in g.an])

    def H(z):
        return statement(theta, -sigma, 1/sigma, z)

    def Hp(z):
        return statement_half(theta, -sigma, 1/sigma, z, True)

    def Hm(z):
        return statement_half(theta, -sigma, 1/sigma, z, False)

    # [L], section 5.10
    conds = []
    # Theorem 1 -- p < q from test above
    conds += [And(1 <= n, 1 <= m, rho*pi - delta >= pi/2, delta > 0,
                  E(z*exp(S.ImaginaryUnit*pi*(nu + 1))))]
    # Theorem 2, statements (2) and (3)
    conds += [And(p + 1 <= m, m + 1 <= q, delta > 0, delta < pi/2, n == 0,
                  (m - p + 1)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*(q - m))),
                  Hm(z*exp(-S.ImaginaryUnit*pi*(q - m))))]
    # Theorem 2, statement (5)  -- p < q from test above
    conds += [And(m == q, n == 0, delta > 0,
                  (sigma + epsilon)*pi - delta >= pi/2, H(z))]
    # Theorem 3, statements (6) and (7)
    conds += [And(Or(And(p <= q - 2, 1 <= tau, tau <= sigma/2),
                     And(p + 1 <= m + n, m + n <= (p + q)/2)),
                  delta > 0, delta < pi/2, (tau + 1)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*nu)),
                  Hm(z*exp(-S.ImaginaryUnit*pi*nu)))]
    # Theorem 4, statements (10) and (11)  -- p < q from test above
    conds += [And(1 <= m, rho > 0, delta > 0, delta + rho*pi < pi/2,
                  (tau + epsilon)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*nu)),
                  Hm(z*exp(-S.ImaginaryUnit*pi*nu)))]
    # Trivial case
    conds += [m == 0]

    # TODO
    # Theorem 5 is quite general
    # Theorem 6 contains special cases for q=p+1

    return Or(*conds)


def _int_inversion(g, x, t):
    """
    Compute the laplace inversion integral, assuming the formula applies.
    """
    b, a = _get_coeff_exp(g.argument, x)
    C, g = _inflate_fox_h(meijerg(g.an, g.aother, g.bm, g.bother, b/t**a), -a)
    return C/t*g


####################################################################
# Finally, the real meat.
####################################################################

_lookup_table = None


@cacheit
@timeit
def _rewrite_single(f, x, recursive=True):
    """
    Try to rewrite f as a sum of single G functions of the form
    C*x**s*G(a*x**b), where b is a rational number and C is independent of x.
    We guarantee that result.argument.as_coeff_mul(x) returns (a, (x**b,))
    or (a, ()).
    Returns a list of tuples (C, s, G) and a condition cond.
    Returns None on failure.
    """
    from .transforms import (mellin_transform, inverse_mellin_transform,
        IntegralTransformError, MellinTransformStripError)

    global _lookup_table
    if not _lookup_table:
        _lookup_table = {}
        _create_lookup_table(_lookup_table)

    if isinstance(f, meijerg):
        coeff, m = factor(f.argument, x).as_coeff_mul(x)
        if len(m) > 1:
            return None
        m = m[0]
        if m.is_Pow:
            if m.base != x or not m.exp.is_Rational:
                return None
        elif m != x:
            return None
        return [(1, 0, meijerg(f.an, f.aother, f.bm, f.bother, coeff*m))], True

    f_ = f
    f = f.subs(x, z)
    t = _mytype(f, z)
    if t in _lookup_table:
        l = _lookup_table[t]
        for formula, terms, cond, hint in l:
            subs = f.match(formula, old=True)
            if subs:
                subs_ = {}
                for fro, to in subs.items():
                    subs_[fro] = unpolarify(polarify(to, lift=True),
                                            exponents_only=True)
                subs = subs_
                if not isinstance(hint, bool):
                    hint = hint.subs(subs)
                if hint == False:
                    continue
                if not isinstance(cond, (bool, BooleanAtom)):
                    cond = unpolarify(cond.subs(subs))
                if _eval_cond(cond) == False:
                    continue
                if not isinstance(terms, list):
                    terms = terms(subs)
                res = []
                for fac, g in terms:
                    r1 = _get_coeff_exp(unpolarify(fac.subs(subs).subs(z, x),
                                                   exponents_only=True), x)
                    try:
                        g = g.subs(subs).subs(z, x)
                    except ValueError:
                        continue
                    # NOTE these substitutions can in principle introduce oo,
                    #      zoo and other absurdities. It shouldn't matter,
                    #      but better be safe.
                    if Tuple(*(r1 + (g,))).has(S.Infinity, S.ComplexInfinity, S.NegativeInfinity):
                        continue
                    g = meijerg(g.an, g.aother, g.bm, g.bother,
                                unpolarify(g.argument, exponents_only=True))
                    res.append(r1 + (g,))
                if res:
                    return res, cond

    # try recursive mellin transform
    if not recursive:
        return None
    _debug('Trying recursive Mellin transform method.')

    def my_imt(F, s, x, strip):
        """ Calling simplify() all the time is slow and not helpful, since
            most of the time it only factors things in a way that has to be
            un-done anyway. But sometimes it can remove apparent poles. """
        # XXX should this be in inverse_mellin_transform?
        try:
            return inverse_mellin_transform(F, s, x, strip,
                                            as_meijerg=True, needeval=True)
        except MellinTransformStripError:
            from sympy.simplify import simplify
            return inverse_mellin_transform(
                simplify(cancel(expand(F))), s, x, strip,
                as_meijerg=True, needeval=True)
    f = f_
    s = _dummy('s', 'rewrite-single', f)
    # to avoid infinite recursion, we have to force the two g functions case

    def my_integrator(f, x):
        r = _meijerint_definite_4(f, x, only_double=True)
        if r is not None:
            from sympy.simplify import hyperexpand
            res, cond = r
            res = _my_unpolarify(hyperexpand(res, rewrite='nonrepsmall'))
            return Piecewise((res, cond),
                             (Integral(f, (x, S.Zero, S.Infinity)), True))
        return Integral(f, (x, S.Zero, S.Infinity))
    try:
        F, strip, _ = mellin_transform(f, x, s, integrator=my_integrator,
                                       simplify=False, needeval=True)
        g = my_imt(F, s, x, strip)
    except IntegralTransformError:
        g = None
    if g is None:
        # We try to find an expression by analytic continuation.
        # (also if the dummy is already in the expression, there is no point in
        #  putting in another one)
        a = _dummy_('a', 'rewrite-single')
        if a not in f.free_symbols and _is_analytic(f, x):
            try:
                F, strip, _ = mellin_transform(f.subs(x, a*x), x, s,
                                               integrator=my_integrator,
                                               needeval=True, simplify=False)
                g = my_imt(F, s, x, strip).subs(a, 1)
            except IntegralTransformError:
                g = None
    if g is None or g.has(S.Infinity, S.NaN, S.ComplexInfinity):
        _debug('Recursive Mellin transform failed.')
        return None
    args = Add.make_args(g)
    res = []
    for f in args:
        c, m = f.as_coeff_mul(x)
        if len(m) > 1:
            raise NotImplementedError('Unexpected form...')
        g = m[0]
        a, b = _get_coeff_exp(g.argument, x)
        res += [(c, 0, meijerg(g.an, g.aother, g.bm, g.bother,
                               unpolarify(polarify(
                                   a, lift=True), exponents_only=True)
                               *x**b))]
    _debug('Recursive Mellin transform worked:', g)
    return res, True


def _rewrite1(f, x, recursive=True):
    """
    Try to rewrite ``f`` using a (sum of) single G functions with argument a*x**b.
    Return fac, po, g such that f = fac*po*g, fac is independent of ``x``.
    and po = x**s.
    Here g is a result from _rewrite_single.
    Return None on failure.
    """
    fac, po, g = _split_mul(f, x)
    g = _rewrite_single(g, x, recursive)
    if g:
        return fac, po, g[0], g[1]


def _rewrite2(f, x):
    """
    Try to rewrite ``f`` as a product of two G functions of arguments a*x**b.
    Return fac, po, g1, g2 such that f = fac*po*g1*g2, where fac is
    independent of x and po is x**s.
    Here g1 and g2 are results of _rewrite_single.
    Returns None on failure.
    """
    fac, po, g = _split_mul(f, x)
    if any(_rewrite_single(expr, x, False) is None for expr in _mul_args(g)):
        return None
    l = _mul_as_two_parts(g)
    if not l:
        return None
    l = list(ordered(l, [
        lambda p: max(len(_exponents(p[0], x)), len(_exponents(p[1], x))),
        lambda p: max(len(_functions(p[0], x)), len(_functions(p[1], x))),
        lambda p: max(len(_find_splitting_points(p[0], x)),
                      len(_find_splitting_points(p[1], x)))]))

    for recursive, (fac1, fac2) in itertools.product((False, True), l):
        g1 = _rewrite_single(fac1, x, recursive)
        g2 = _rewrite_single(fac2, x, recursive)
        if g1 and g2:
            cond = And(g1[1], g2[1])
            if cond != False:
                return fac, po, g1[0], g2[0], cond


def meijerint_indefinite(f, x):
    """
    Compute an indefinite integral of ``f`` by rewriting it as a G function.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_indefinite
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> meijerint_indefinite(sin(x), x)
    -cos(x)
    """
    f = sympify(f)
    results = []
    for a in sorted(_find_splitting_points(f, x) | {S.Zero}, key=default_sort_key):
        res = _meijerint_indefinite_1(f.subs(x, x + a), x)
        if not res:
            continue
        res = res.subs(x, x - a)
        if _has(res, hyper, meijerg):
            results.append(res)
        else:
            return res
    if f.has(HyperbolicFunction):
        _debug('Try rewriting hyperbolics in terms of exp.')
        rv = meijerint_indefinite(
            _rewrite_hyperbolics_as_exp(f), x)
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                return collect(factor_terms(rv), rv.atoms(exp))
            results.extend(rv)
    if results:
        return next(ordered(results))


def _meijerint_indefinite_1(f, x):
    """ Helper that does not attempt any substitution. """
    _debug('Trying to compute the indefinite integral of', f, 'wrt', x)
    from sympy.simplify import hyperexpand, powdenest

    gs = _rewrite1(f, x)
    if gs is None:
        # Note: the code that calls us will do expand() and try again
        return None

    fac, po, gl, cond = gs
    _debug(' could rewrite:', gs)
    res = S.Zero
    for C, s, g in gl:
        a, b = _get_coeff_exp(g.argument, x)
        _, c = _get_coeff_exp(po, x)
        c += s

        # we do a substitution t=a*x**b, get integrand fac*t**rho*g
        fac_ = fac * C * x**(1 + c) / b
        rho = (c + 1)/b

        # we now use t**rho*G(params, t) = G(params + rho, t)
        # [L, page 150, equation (4)]
        # and integral G(params, t) dt = G(1, params+1, 0, t)
        #   (or a similar expression with 1 and 0 exchanged ... pick the one
        #    which yields a well-defined function)
        # [R, section 5]
        # (Note that this dummy will immediately go away again, so we
        #  can safely pass S.One for ``expr``.)
        t = _dummy('t', 'meijerint-indefinite', S.One)

        def tr(p):
            return [a + rho for a in p]
        if any(b.is_integer and (b <= 0) == True for b in tr(g.bm)):
            r = -meijerg(
                list(g.an), list(g.aother) + [1-rho], list(g.bm) + [-rho], list(g.bother), t)
        else:
            r = meijerg(
                list(g.an) + [1-rho], list(g.aother), list(g.bm), list(g.bother) + [-rho], t)
        # The antiderivative is most often expected to be defined
        # in the neighborhood of  x = 0.
        if b.is_extended_nonnegative and not f.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            place = 0  # Assume we can expand at zero
        else:
            place = None
        r = hyperexpand(r.subs(t, a*x**b), place=place)

        # now substitute back
        # Note: we really do want the powers of x to combine.
        res += powdenest(fac_*r, polar=True)

    def _clean(res):
        """This multiplies out superfluous powers of x we created, and chops off
        constants:

            >> _clean(x*(exp(x)/x - 1/x) + 3)
            exp(x)

        cancel is used before mul_expand since it is possible for an
        expression to have an additive constant that does not become isolated
        with simple expansion. Such a situation was identified in issue 6369:

        Examples
        ========

        >>> from sympy import sqrt, cancel
        >>> from sympy.abc import x
        >>> a = sqrt(2*x + 1)
        >>> bad = (3*x*a**5 + 2*x - a**5 + 1)/a**2
        >>> bad.expand().as_independent(x)[0]
        0
        >>> cancel(bad).expand().as_independent(x)[0]
        1
        """
        res = expand_mul(cancel(res), deep=False)
        return Add._from_args(res.as_coeff_add(x)[1])

    res = piecewise_fold(res, evaluate=None)
    if res.is_Piecewise:
        newargs = []
        for e, c in res.args:
            e = _my_unpolarify(_clean(e))
            newargs += [(e, c)]
        res = Piecewise(*newargs, evaluate=False)
    else:
        res = _my_unpolarify(_clean(res))
    return Piecewise((res, _my_unpolarify(cond)), (Integral(f, x), True))


@timeit
def meijerint_definite(f, x, a, b):
    """
    Integrate ``f`` over the interval [``a``, ``b``], by rewriting it as a product
    of two G functions, or as a single G function.

    Return res, cond, where cond are convergence conditions.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_definite
    >>> from sympy import exp, oo
    >>> from sympy.abc import x
    >>> meijerint_definite(exp(-x**2), x, -oo, oo)
    (sqrt(pi), True)

    This function is implemented as a succession of functions
    meijerint_definite, _meijerint_definite_2, _meijerint_definite_3,
    _meijerint_definite_4. Each function in the list calls the next one
    (presumably) several times. This means that calling meijerint_definite
    can be very costly.
    """
    # This consists of three steps:
    # 1) Change the integration limits to 0, oo
    # 2) Rewrite in terms of G functions
    # 3) Evaluate the integral
    #
    # There are usually several ways of doing this, and we want to try all.
    # This function does (1), calls _meijerint_definite_2 for step (2).
    _debugf('Integrating %s wrt %s from %s to %s.', (f, x, a, b))
    f = sympify(f)
    if f.has(DiracDelta):
        _debug('Integrand has DiracDelta terms - giving up.')
        return None

    if f.has(SingularityFunction):
        _debug('Integrand has Singularity Function terms - giving up.')
        return None

    f_, x_, a_, b_ = f, x, a, b

    # Let's use a dummy in case any of the boundaries has x.
    d = Dummy('x')
    f = f.subs(x, d)
    x = d

    if a == b:
        return (S.Zero, True)

    results = []
    if a is S.NegativeInfinity and b is not S.Infinity:
        return meijerint_definite(f.subs(x, -x), x, -b, -a)

    elif a is S.NegativeInfinity:
        # Integrating -oo to oo. We need to find a place to split the integral.
        _debug('  Integrating -oo to +oo.')
        innermost = _find_splitting_points(f, x)
        _debug('  Sensible splitting points:', innermost)
        for c in sorted(innermost, key=default_sort_key, reverse=True) + [S.Zero]:
            _debug('  Trying to split at', c)
            if not c.is_extended_real:
                _debug('  Non-real splitting point.')
                continue
            res1 = _meijerint_definite_2(f.subs(x, x + c), x)
            if res1 is None:
                _debug('  But could not compute first integral.')
                continue
            res2 = _meijerint_definite_2(f.subs(x, c - x), x)
            if res2 is None:
                _debug('  But could not compute second integral.')
                continue
            res1, cond1 = res1
            res2, cond2 = res2
            cond = _condsimp(And(cond1, cond2))
            if cond == False:
                _debug('  But combined condition is always false.')
                continue
            res = res1 + res2
            return res, cond

    elif a is S.Infinity:
        res = meijerint_definite(f, x, b, S.Infinity)
        return -res[0], res[1]

    elif (a, b) == (S.Zero, S.Infinity):
        # This is a common case - try it directly first.
        res = _meijerint_definite_2(f, x)
        if res:
            if _has(res[0], meijerg):
                results.append(res)
            else:
                return res

    else:
        if b is S.Infinity:
            for split in _find_splitting_points(f, x):
                if (a - split >= 0) == True:
                    _debugf('Trying x -> x + %s', split)
                    res = _meijerint_definite_2(f.subs(x, x + split)
                                                *Heaviside(x + split - a), x)
                    if res:
                        if _has(res[0], meijerg):
                            results.append(res)
                        else:
                            return res

        f = f.subs(x, x + a)
        b = b - a
        a = 0
        if b is not S.Infinity:
            phi = exp(S.ImaginaryUnit*arg(b))
            b = Abs(b)
            f = f.subs(x, phi*x)
            f *= Heaviside(b - x)*phi
            b = S.Infinity

        _debug('Changed limits to', a, b)
        _debug('Changed function to', f)
        res = _meijerint_definite_2(f, x)
        if res:
            if _has(res[0], meijerg):
                results.append(res)
            else:
                return res
    if f_.has(HyperbolicFunction):
        _debug('Try rewriting hyperbolics in terms of exp.')
        rv = meijerint_definite(
            _rewrite_hyperbolics_as_exp(f_), x_, a_, b_)
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                rv = (collect(factor_terms(rv[0]), rv[0].atoms(exp)),) + rv[1:]
                return rv
            results.extend(rv)
    if results:
        return next(ordered(results))


def _guess_expansion(f, x):
    """ Try to guess sensible rewritings for integrand f(x). """
    res = [(f, 'original integrand')]

    orig = res[-1][0]
    saw = {orig}
    expanded = expand_mul(orig)
    if expanded not in saw:
        res += [(expanded, 'expand_mul')]
        saw.add(expanded)

    expanded = expand(orig)
    if expanded not in saw:
        res += [(expanded, 'expand')]
        saw.add(expanded)

    if orig.has(TrigonometricFunction, HyperbolicFunction):
        expanded = expand_mul(expand_trig(orig))
        if expanded not in saw:
            res += [(expanded, 'expand_trig, expand_mul')]
            saw.add(expanded)

    if orig.has(cos, sin):
        from sympy.simplify.fu import sincos_to_sum
        reduced = sincos_to_sum(orig)
        if reduced not in saw:
            res += [(reduced, 'trig power reduction')]
            saw.add(reduced)

    return res


def _meijerint_definite_2(f, x):
    """
    Try to integrate f dx from zero to infinity.

    The body of this function computes various 'simplifications'
    f1, f2, ... of f (e.g. by calling expand_mul(), trigexpand()
    - see _guess_expansion) and calls _meijerint_definite_3 with each of
    these in succession.
    If _meijerint_definite_3 succeeds with any of the simplified functions,
    returns this result.
    """
    # This function does preparation for (2), calls
    # _meijerint_definite_3 for (2) and (3) combined.

    # use a positive dummy - we integrate from 0 to oo
    # XXX if a nonnegative symbol is used there will be test failures
    dummy = _dummy('x', 'meijerint-definite2', f, positive=True)
    f = f.subs(x, dummy)
    x = dummy

    if f == 0:
        return S.Zero, True

    for g, explanation in _guess_expansion(f, x):
        _debug('Trying', explanation)
        res = _meijerint_definite_3(g, x)
        if res:
            return res


def _meijerint_definite_3(f, x):
    """
    Try to integrate f dx from zero to infinity.

    This function calls _meijerint_definite_4 to try to compute the
    integral. If this fails, it tries using linearity.
    """
    res = _meijerint_definite_4(f, x)
    if res and res[1] != False:
        return res
    if f.is_Add:
        _debug('Expanding and evaluating all terms.')
        ress = [_meijerint_definite_4(g, x) for g in f.args]
        if all(r is not None for r in ress):
            conds = []
            res = S.Zero
            for r, c in ress:
                res += r
                conds += [c]
            c = And(*conds)
            if c != False:
                return res, c


def _my_unpolarify(f):
    return _eval_cond(unpolarify(f))


@timeit
def _meijerint_definite_4(f, x, only_double=False):
    """
    Try to integrate f dx from zero to infinity.

    Explanation
    ===========

    This function tries to apply the integration theorems found in literature,
    i.e. it tries to rewrite f as either one or a product of two G-functions.

    The parameter ``only_double`` is used internally in the recursive algorithm
    to disable trying to rewrite f as a single G-function.
    """
    from sympy.simplify import hyperexpand
    # This function does (2) and (3)
    _debug('Integrating', f)
    # Try single G function.
    if not only_double:
        gs = _rewrite1(f, x, recursive=False)
        if gs is not None:
            fac, po, g, cond = gs
            _debug('Could rewrite as single G function:', fac, po, g)
            res = S.Zero
            for C, s, f in g:
                if C == 0:
                    continue
                C, f = _rewrite_saxena_1(fac*C, po*x**s, f, x)
                res += C*_int0oo_1(f, x)
                cond = And(cond, _check_antecedents_1(f, x))
                if cond == False:
                    break
            cond = _my_unpolarify(cond)
            if cond == False:
                _debug('But cond is always False.')
            else:
                _debug('Result before branch substitutions is:', res)
                return _my_unpolarify(hyperexpand(res)), cond

    # Try two G functions.
    gs = _rewrite2(f, x)
    if gs is not None:
        for full_pb in [False, True]:
            fac, po, g1, g2, cond = gs
            _debug('Could rewrite as two G functions:', fac, po, g1, g2)
            res = S.Zero
            for C1, s1, f1 in g1:
                for C2, s2, f2 in g2:
                    r = _rewrite_saxena(fac*C1*C2, po*x**(s1 + s2),
                                        f1, f2, x, full_pb)
                    if r is None:
                        _debug('Non-rational exponents.')
                        return
                    C, f1_, f2_ = r
                    _debug('Saxena subst for yielded:', C, f1_, f2_)
                    cond = And(cond, _check_antecedents(f1_, f2_, x))
                    if cond == False:
                        break
                    res += C*_int0oo(f1_, f2_, x)
                else:
                    continue
                break
            cond = _my_unpolarify(cond)
            if cond == False:
                _debugf('But cond is always False (full_pb=%s).', full_pb)
            else:
                _debugf('Result before branch substitutions is: %s', (res, ))
                if only_double:
                    return res, cond
                return _my_unpolarify(hyperexpand(res)), cond


def meijerint_inversion(f, x, t):
    r"""
    Compute the inverse laplace transform
    $\int_{c+i\infty}^{c-i\infty} f(x) e^{tx}\, dx$,
    for real c larger than the real part of all singularities of ``f``.

    Note that ``t`` is always assumed real and positive.

    Return None if the integral does not exist or could not be evaluated.

    Examples
    ========

    >>> from sympy.abc import x, t
    >>> from sympy.integrals.meijerint import meijerint_inversion
    >>> meijerint_inversion(1/x, x, t)
    Heaviside(t)
    """
    f_ = f
    t_ = t
    t = Dummy('t', polar=True)  # We don't want sqrt(t**2) = abs(t) etc
    f = f.subs(t_, t)
    _debug('Laplace-inverting', f)
    if not _is_analytic(f, x):
        _debug('But expression is not analytic.')
        return None
    # Exponentials correspond to shifts; we filter them out and then
    # shift the result later.  If we are given an Add this will not
    # work, but the calling code will take care of that.
    shift = S.Zero

    if f.is_Mul:
        args = list(f.args)
    elif isinstance(f, exp):
        args = [f]
    else:
        args = None

    if args:
        newargs = []
        exponentials = []
        while args:
            arg = args.pop()
            if isinstance(arg, exp):
                arg2 = expand(arg)
                if arg2.is_Mul:
                    args += arg2.args
                    continue
                try:
                    a, b = _get_coeff_exp(arg.args[0], x)
                except _CoeffExpValueError:
                    b = 0
                if b == 1:
                    exponentials.append(a)
                else:
                    newargs.append(arg)
            elif arg.is_Pow:
                arg2 = expand(arg)
                if arg2.is_Mul:
                    args += arg2.args
                    continue
                if x not in arg.base.free_symbols:
                    try:
                        a, b = _get_coeff_exp(arg.exp, x)
                    except _CoeffExpValueError:
                        b = 0
                    if b == 1:
                        exponentials.append(a*log(arg.base))
                newargs.append(arg)
            else:
                newargs.append(arg)
        shift = Add(*exponentials)
        f = Mul(*newargs)

    if x not in f.free_symbols:
        _debug('Expression consists of constant and exp shift:', f, shift)
        cond = Eq(im(shift), 0)
        if cond == False:
            _debug('but shift is nonreal, cannot be a Laplace transform')
            return None
        res = f*DiracDelta(t + shift)
        _debug('Result is a delta function, possibly conditional:', res, cond)
        # cond is True or Eq
        return Piecewise((res.subs(t, t_), cond))

    gs = _rewrite1(f, x)
    if gs is not None:
        fac, po, g, cond = gs
        _debug('Could rewrite as single G function:', fac, po, g)
        res = S.Zero
        for C, s, f in g:
            C, f = _rewrite_inversion(fac*C, po*x**s, f, x)
            res += C*_int_inversion(f, x, t)
            cond = And(cond, _check_antecedents_inversion(f, x))
            if cond == False:
                break
        cond = _my_unpolarify(cond)
        if cond == False:
            _debug('But cond is always False.')
        else:
            _debug('Result before branch substitution:', res)
            from sympy.simplify import hyperexpand
            res = _my_unpolarify(hyperexpand(res))
            if not res.has(Heaviside):
                res *= Heaviside(t)
            res = res.subs(t, t + shift)
            if not isinstance(cond, bool):
                cond = cond.subs(t, t + shift)
            from .transforms import InverseLaplaceTransform
            return Piecewise((res.subs(t, t_), cond),
                             (InverseLaplaceTransform(f_.subs(t, t_), x, t_, None), True))
<|endoftext|>from __future__ import annotations

from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import diff
from sympy.core.logic import fuzzy_bool
from sympy.core.mul import Mul
from sympy.core.numbers import oo, pi
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.core.sympify import sympify
from sympy.functions import Piecewise, sqrt, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.special.singularity_functions import Heaviside
from .rationaltools import ratint
from sympy.matrices import MatrixBase
from sympy.polys import Poly, PolynomialError
from sympy.series.formal import FormalPowerSeries
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.tensor.functions import shape
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent


class Integral(AddWithLimits):
    """Represents unevaluated integral."""

    __slots__ = ()

    args: tuple[Expr, Tuple] # type: ignore

    def __new__(cls, function, *symbols, **assumptions) -> Integral:
        """Create an unevaluated integral.

        Explanation
        ===========

        Arguments are an integrand followed by one or more limits.

        If no limits are given and there is only one free symbol in the
        expression, that symbol will be used, otherwise an error will be
        raised.

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x)
        Integral(x, x)
        >>> Integral(y)
        Integral(y, y)

        When limits are provided, they are interpreted as follows (using
        ``x`` as though it were the variable of integration):

            (x,) or x - indefinite integral
            (x, a) - "evaluate at" integral is an abstract antiderivative
            (x, a, b) - definite integral

        The ``as_dummy`` method can be used to see which symbols cannot be
        targeted by subs: those with a prepended underscore cannot be
        changed with ``subs``. (Also, the integration variables themselves --
        the first element of a limit -- can never be changed by subs.)

        >>> i = Integral(x, x)
        >>> at = Integral(x, (x, x))
        >>> i.as_dummy()
        Integral(x, x)
        >>> at.as_dummy()
        Integral(_0, (_0, x))

        """

        #This will help other classes define their own definitions
        #of behaviour with Integral.
        if hasattr(function, '_eval_Integral'):
            return function._eval_Integral(*symbols, **assumptions)

        if isinstance(function, Poly):
            sympy_deprecation_warning(
                """
                integrate(Poly) and Integral(Poly) are deprecated. Instead,
                use the Poly.integrate() method, or convert the Poly to an
                Expr first with the Poly.as_expr() method.
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-integrate-poly")

        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        return obj

    def __getnewargs__(self):
        return (self.function,) + tuple([tuple(xab) for xab in self.limits])

    @property
    def free_symbols(self):
        """
        This method returns the symbols that will exist when the
        integral is evaluated. This is useful if one is trying to
        determine whether an integral depends on a certain
        symbol or not.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x, (x, y, 1)).free_symbols
        {y}

        See Also
        ========

        sympy.concrete.expr_with_limits.ExprWithLimits.function
        sympy.concrete.expr_with_limits.ExprWithLimits.limits
        sympy.concrete.expr_with_limits.ExprWithLimits.variables
        """
        return super().free_symbols

    def _eval_is_zero(self):
        # This is a very naive and quick test, not intended to do the integral to
        # answer whether it is zero or not, e.g. Integral(sin(x), (x, 0, 2*pi))
        # is zero but this routine should return None for that case. But, like
        # Mul, there are trivial situations for which the integral will be
        # zero so we check for those.
        if self.function.is_zero:
            return True
        got_none = False
        for l in self.limits:
            if len(l) == 3:
                z = (l[1] == l[2]) or (l[1] - l[2]).is_zero
                if z:
                    return True
                elif z is None:
                    got_none = True
        free = self.function.free_symbols
        for xab in self.limits:
            if len(xab) == 1:
                free.add(xab[0])
                continue
            if len(xab) == 2 and xab[0] not in free:
                if xab[1].is_zero:
                    return True
                elif xab[1].is_zero is None:
                    got_none = True
            # take integration symbol out of free since it will be replaced
            # with the free symbols in the limits
            free.discard(xab[0])
            # add in the new symbols
            for i in xab[1:]:
                free.update(i.free_symbols)
        if self.function.is_zero is False and got_none is False:
            return False

    def transform(self, x, u):
        r"""
        Performs a change of variables from `x` to `u` using the relationship
        given by `x` and `u` which will define the transformations `f` and `F`
        (which are inverses of each other) as follows:

        1) If `x` is a Symbol (which is a variable of integration) then `u`
           will be interpreted as some function, f(u), with inverse F(u).
           This, in effect, just makes the substitution of x with f(x).

        2) If `u` is a Symbol then `x` will be interpreted as some function,
           F(x), with inverse f(u). This is commonly referred to as
           u-substitution.

        Once f and F have been identified, the transformation is made as
        follows:

        .. math:: \int_a^b x \mathrm{d}x \rightarrow \int_{F(a)}^{F(b)} f(x)
                  \frac{\mathrm{d}}{\mathrm{d}x}

        where `F(x)` is the inverse of `f(x)` and the limits and integrand have
        been corrected so as to retain the same value after integration.

        Notes
        =====

        The mappings, F(x) or f(u), must lead to a unique integral. Linear
        or rational linear expression, ``2*x``, ``1/x`` and ``sqrt(x)``, will
        always work; quadratic expressions like ``x**2 - 1`` are acceptable
        as long as the resulting integrand does not depend on the sign of
        the solutions (see examples).

        The integral will be returned unchanged if ``x`` is not a variable of
        integration.

        ``x`` must be (or contain) only one of of the integration variables. If
        ``u`` has more than one free symbol then it should be sent as a tuple
        (``u``, ``uvar``) where ``uvar`` identifies which variable is replacing
        the integration variable.
        XXX can it contain another integration variable?

        Examples
        ========

        >>> from sympy.abc import a, x, u
        >>> from sympy import Integral, cos, sqrt

        >>> i = Integral(x*cos(x**2 - 1), (x, 0, 1))

        transform can change the variable of integration

        >>> i.transform(x, u)
        Integral(u*cos(u**2 - 1), (u, 0, 1))

        transform can perform u-substitution as long as a unique
        integrand is obtained:

        >>> ui = i.transform(x**2 - 1, u)
        >>> ui
        Integral(cos(u)/2, (u, -1, 0))

        This attempt fails because x = +/-sqrt(u + 1) and the
        sign does not cancel out of the integrand:

        >>> Integral(cos(x**2 - 1), (x, 0, 1)).transform(x**2 - 1, u)
        Traceback (most recent call last):
        ...
        ValueError:
        The mapping between F(x) and f(u) did not give a unique integrand.

        transform can do a substitution. Here, the previous
        result is transformed back into the original expression
        using "u-substitution":

        >>> ui.transform(sqrt(u + 1), x) == i
        True

        We can accomplish the same with a regular substitution:

        >>> ui.transform(u, x**2 - 1) == i
        True

        If the `x` does not contain a symbol of integration then
        the integral will be returned unchanged. Integral `i` does
        not have an integration variable `a` so no change is made:

        >>> i.transform(a, x) == i
        True

        When `u` has more than one free symbol the symbol that is
        replacing `x` must be identified by passing `u` as a tuple:

        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, u))
        Integral(a + u, (u, -a, 1 - a))
        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, a))
        Integral(a + u, (a, -u, 1 - u))

        See Also
        ========

        sympy.concrete.expr_with_limits.ExprWithLimits.variables : Lists the integration variables
        as_dummy : Replace integration variables with dummy ones
        """
        d = Dummy('d')

        xfree = x.free_symbols.intersection(self.variables)
        if len(xfree) > 1:
            raise ValueError(
                'F(x) can only contain one of: %s' % self.variables)
        xvar = xfree.pop() if xfree else d

        if xvar not in self.variables:
            return self

        u = sympify(u)
        if isinstance(u, Expr):
            ufree = u.free_symbols
            if len(ufree) == 0:
                raise ValueError(filldedent('''
                f(u) cannot be a constant'''))
            if len(ufree) > 1:
                raise ValueError(filldedent('''
                When f(u) has more than one free symbol, the one replacing x
                must be identified: pass f(u) as (f(u), u)'''))
            uvar = ufree.pop()
        else:
            u, uvar = u
            if uvar not in u.free_symbols:
                raise ValueError(filldedent('''
                Expecting a tuple (expr, symbol) where symbol identified
                a free symbol in expr, but symbol is not in expr's free
                symbols.'''))
            if not isinstance(uvar, Symbol):
                # This probably never evaluates to True
                raise ValueError(filldedent('''
                Expecting a tuple (expr, symbol) but didn't get
                a symbol; got %s''' % uvar))

        if x.is_Symbol and u.is_Symbol:
            return self.xreplace({x: u})

        if not x.is_Symbol and not u.is_Symbol:
            raise ValueError('either x or u must be a symbol')

        if uvar == xvar:
            return self.transform(x, (u.subs(uvar, d), d)).xreplace({d: uvar})

        if uvar in self.limits:
            raise ValueError(filldedent('''
            u must contain the same variable as in x
            or a variable that is not already an integration variable'''))

        from sympy.solvers.solvers import solve
        if not x.is_Symbol:
            F = [x.subs(xvar, d)]
            soln = solve(u - x, xvar, check=False)
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), x)')
            f = [fi.subs(uvar, d) for fi in soln]
        else:
            f = [u.subs(uvar, d)]
            from sympy.simplify.simplify import posify
            pdiff, reps = posify(u - x)
            puvar = uvar.subs([(v, k) for k, v in reps.items()])
            soln = [s.subs(reps) for s in solve(pdiff, puvar)]
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), u)')
            F = [fi.subs(xvar, d) for fi in soln]

        newfuncs = {(self.function.subs(xvar, fi)*fi.diff(d)
                        ).subs(d, uvar) for fi in f}
        if len(newfuncs) > 1:
            raise ValueError(filldedent('''
            The mapping between F(x) and f(u) did not give
            a unique integrand.'''))
        newfunc = newfuncs.pop()

        def _calc_limit_1(F, a, b):
            """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
            wok = F.subs(d, a)
            if wok is S.NaN or wok.is_finite is False and a.is_finite:
                return limit(sign(b)*F, d, a)
            return wok

        def _calc_limit(a, b):
            """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
            avals = list({_calc_limit_1(Fi, a, b) for Fi in F})
            if len(avals) > 1:
                raise ValueError(filldedent('''
                The mapping between F(x) and f(u) did not
                give a unique limit.'''))
            return avals[0]

        newlimits = []
        for xab in self.limits:
            sym = xab[0]
            if sym == xvar:
                if len(xab) == 3:
                    a, b = xab[1:]
                    a, b = _calc_limit(a, b), _calc_limit(b, a)
                    if fuzzy_bool(a - b > 0):
                        a, b = b, a
                        newfunc = -newfunc
                    newlimits.append((uvar, a, b))
                elif len(xab) == 2:
                    a = _calc_limit(xab[1], 1)
                    newlimits.append((uvar, a))
                else:
                    newlimits.append(uvar)
            else:
                newlimits.append(xab)

        return self.func(newfunc, *newlimits)

    def doit(self, **hints):
        """
        Perform the integration using any hints given.

        Examples
        ========

        >>> from sympy import Piecewise, S
        >>> from sympy.abc import x, t
        >>> p = x**2 + Piecewise((0, x/t < 0), (1, True))
        >>> p.integrate((t, S(4)/5, 1), (x, -1, 1))
        1/3

        See Also
        ========

        sympy.integrals.trigonometry.trigintegrate
        sympy.integrals.heurisch.heurisch
        sympy.integrals.rationaltools.ratint
        as_sum : Approximate the integral using a sum
        """
        if not hints.get('integrals', True):
            return self

        deep = hints.get('deep', True)
        meijerg = hints.get('meijerg', None)
        conds = hints.get('conds', 'piecewise')
        risch = hints.get('risch', None)
        heurisch = hints.get('heurisch', None)
        manual = hints.get('manual', None)
        if len(list(filter(None, (manual, meijerg, risch, heurisch)))) > 1:
            raise ValueError("At most one of manual, meijerg, risch, heurisch can be True")
        elif manual:
            meijerg = risch = heurisch = False
        elif meijerg:
            manual = risch = heurisch = False
        elif risch:
            manual = meijerg = heurisch = False
        elif heurisch:
            manual = meijerg = risch = False
        eval_kwargs = {"meijerg": meijerg, "risch": risch, "manual": manual, "heurisch": heurisch,
            "conds": conds}

        if conds not in ('separate', 'piecewise', 'none'):
            raise ValueError('conds must be one of "separate", "piecewise", '
                             '"none", got: %s' % conds)

        if risch and any(len(xab) > 1 for xab in self.limits):
            raise ValueError('risch=True is only allowed for indefinite integrals.')

        # check for the trivial zero
        if self.is_zero:
            return S.Zero

        # hacks to handle integrals of
        # nested summations
        from sympy.concrete.summations import Sum
        if isinstance(self.function, Sum):
            if any(v in self.function.limits[0] for v in self.variables):
                raise ValueError('Limit of the sum cannot be an integration variable.')
            if any(l.is_infinite for l in self.function.limits[0][1:]):
                return self
            _i = self
            _sum = self.function
            return _sum.func(_i.func(_sum.function, *_i.limits).doit(), *_sum.limits).doit()

        # now compute and check the function
        function = self.function

        # hack to use a consistent Heaviside(x, 1/2)
        function = function.replace(
            lambda x: isinstance(x, Heaviside) and x.args[1]*2 != 1,
            lambda x: Heaviside(x.args[0]))

        if deep:
            function = function.doit(**hints)
        if function.is_zero:
            return S.Zero

        # hacks to handle special cases
        if isinstance(function, MatrixBase):
            return function.applyfunc(
                lambda f: self.func(f, *self.limits).doit(**hints))

        if isinstance(function, FormalPowerSeries):
            if len(self.limits) > 1:
                raise NotImplementedError
            xab = self.limits[0]
            if len(xab) > 1:
                return function.integrate(xab, **eval_kwargs)
            else:
                return function.integrate(xab[0], **eval_kwargs)

        # There is no trivial answer and special handling
        # is done so continue

        # first make sure any definite limits have integration
        # variables with matching assumptions
        reps = {}
        for xab in self.limits:
            if len(xab) != 3:
                # it makes sense to just make
                # all x real but in practice with the
                # current state of integration...this
                # doesn't work out well
                # x = xab[0]
                # if x not in reps and not x.is_real:
                #     reps[x] = Dummy(real=True)
                continue
            x, a, b = xab
            l = (a, b)
            if all(i.is_nonnegative for i in l) and not x.is_nonnegative:
                d = Dummy(positive=True)
            elif all(i.is_nonpositive for i in l) and not x.is_nonpositive:
                d = Dummy(negative=True)
            elif all(i.is_real for i in l) and not x.is_real:
                d = Dummy(real=True)
            else:
                d = None
            if d:
                reps[x] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # when separate=True
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did

        # continue with existing assumptions
        undone_limits = []
        # ulj = free symbols of any undone limits' upper and lower limits
        ulj = set()
        for xab in self.limits:
            # compute uli, the free symbols in the
            # Upper and Lower limits of limit I
            if len(xab) == 1:
                uli = set(xab[:1])
            elif len(xab) == 2:
                uli = xab[1].free_symbols
            elif len(xab) == 3:
                uli = xab[1].free_symbols.union(xab[2].free_symbols)
            # this integral can be done as long as there is no blocking
            # limit that has been undone. An undone limit is blocking if
            # it contains an integration variable that is in this limit's
            # upper or lower free symbols or vice versa
            if xab[0] in ulj or any(v[0] in uli for v in undone_limits):
                undone_limits.append(xab)
                ulj.update(uli)
                function = self.func(*([function] + [xab]))
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue

            if function.has(Abs, sign) and (
                (len(xab) < 3 and all(x.is_extended_real for x in xab)) or
                (len(xab) == 3 and all(x.is_extended_real and not x.is_infinite for
                 x in xab[1:]))):
                    # some improper integrals are better off with Abs
                    xr = Dummy("xr", real=True)
                    function = (function.xreplace({xab[0]: xr})
                        .rewrite(Piecewise).xreplace({xr: xab[0]}))
            elif function.has(Min, Max):
                function = function.rewrite(Piecewise)
            if (function.has(Piecewise) and
                not isinstance(function, Piecewise)):
                    function = piecewise_fold(function)
            if isinstance(function, Piecewise):
                if len(xab) == 1:
                    antideriv = function._eval_integral(xab[0],
                        **eval_kwargs)
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
            else:
                # There are a number of tradeoffs in using the
                # Meijer G method. It can sometimes be a lot faster
                # than other methods, and sometimes slower. And
                # there are certain types of integrals for which it
                # is more likely to work than others. These
                # heuristics are incorporated in deciding what
                # integration methods to try, in what order. See the
                # integrate() docstring for details.
                def try_meijerg(function, xab):
                    ret = None
                    if len(xab) == 3 and meijerg is not False:
                        x, a, b = xab
                        try:
                            res = meijerint_definite(function, x, a, b)
                        except NotImplementedError:
                            _debug('NotImplementedError '
                                'from meijerint_definite')
                            res = None
                        if res is not None:
                            f, cond = res
                            if conds == 'piecewise':
                                u = self.func(function, (x, a, b))
                                # if Piecewise modifies cond too
                                # much it may not be recognized by
                                # _condsimp pattern matching so just
                                # turn off all evaluation
                                return Piecewise((f, cond), (u, True),
                                    evaluate=False)
                            elif conds == 'separate':
                                if len(self.limits) != 1:
                                    raise ValueError(filldedent('''
                                        conds=separate not supported in
                                        multiple integrals'''))
                                ret = f, cond
                            else:
                                ret = f
                    return ret

                meijerg1 = meijerg
                if (meijerg is not False and
                        len(xab) == 3 and xab[1].is_extended_real and xab[2].is_extended_real
                        and not function.is_Poly and
                        (xab[1].has(oo, -oo) or xab[2].has(oo, -oo))):
                    ret = try_meijerg(function, xab)
                    if ret is not None:
                        function = ret
                        continue
                    meijerg1 = False
                # If the special meijerg code did not succeed in
                # finding a definite integral, then the code using
                # meijerint_indefinite will not either (it might
                # find an antiderivative, but the answer is likely
                # to be nonsensical). Thus if we are requested to
                # only use Meijer G-function methods, we give up at
                # this stage. Otherwise we just disable G-function
                # methods.
                if meijerg1 is False and meijerg is True:
                    antideriv = None
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
                    if antideriv is None and meijerg is True:
                        ret = try_meijerg(function, xab)
                        if ret is not None:
                            function = ret
                            continue

            final = hints.get('final', True)
            # dotit may be iterated but floor terms making atan and acot
            # continuous should only be added in the final round
            if (final and not isinstance(antideriv, Integral) and
                antideriv is not None):
                for atan_term in antideriv.atoms(atan):
                    atan_arg = atan_term.args[0]
                    # Checking `atan_arg` to be linear combination of `tan` or `cot`
                    for tan_part in atan_arg.atoms(tan):
                        x1 = Dummy('x1')
                        tan_exp1 = atan_arg.subs(tan_part, x1)
                        # The coefficient of `tan` should be constant
                        coeff = tan_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = tan_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a-pi/2)/pi)))
                    for cot_part in atan_arg.atoms(cot):
                        x1 = Dummy('x1')
                        cot_exp1 = atan_arg.subs(cot_part, x1)
                        # The coefficient of `cot` should be constant
                        coeff = cot_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = cot_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a)/pi)))

            if antideriv is None:
                undone_limits.append(xab)
                function = self.func(*([function] + [xab])).factor()
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            else:
                if len(xab) == 1:
                    function = antideriv
                else:
                    if len(xab) == 3:
                        x, a, b = xab
                    elif len(xab) == 2:
                        x, b = xab
                        a = None
                    else:
                        raise NotImplementedError

                    if deep:
                        if isinstance(a, Basic):
                            a = a.doit(**hints)
                        if isinstance(b, Basic):
                            b = b.doit(**hints)

                    if antideriv.is_Poly:
                        gens = list(antideriv.gens)
                        gens.remove(x)

                        antideriv = antideriv.as_expr()

                        function = antideriv._eval_interval(x, a, b)
                        function = Poly(function, *gens)
                    else:
                        def is_indef_int(g, x):
                            return (isinstance(g, Integral) and
                                    any(i == (x,) for i in g.limits))

                        def eval_factored(f, x, a, b):
                            # _eval_interval for integrals with
                            # (constant) factors
                            # a single indefinite integral is assumed
                            args = []
                            for g in Mul.make_args(f):
                                if is_indef_int(g, x):
                                    args.append(g._eval_interval(x, a, b))
                                else:
                                    args.append(g)
                            return Mul(*args)

                        integrals, others, piecewises = [], [], []
                        for f in Add.make_args(antideriv):
                            if any(is_indef_int(g, x)
                                   for g in Mul.make_args(f)):
                                integrals.append(f)
                            elif any(isinstance(g, Piecewise)
                                     for g in Mul.make_args(f)):
                                piecewises.append(piecewise_fold(f))
                            else:
                                others.append(f)
                        uneval = Add(*[eval_factored(f, x, a, b)
                                       for f in integrals])
                        try:
                            evalued = Add(*others)._eval_interval(x, a, b)
                            evalued_pw = piecewise_fold(Add(*piecewises))._eval_interval(x, a, b)
                            function = uneval + evalued + evalued_pw
                        except NotImplementedError:
                            # This can happen if _eval_interval depends in a
                            # complicated way on limits that cannot be computed
                            undone_limits.append(xab)
                            function = self.func(*([function] + [xab]))
                            factored_function = function.factor()
                            if not isinstance(factored_function, Integral):
                                function = factored_function
        return function

    def _eval_derivative(self, sym):
        """Evaluate the derivative of the current Integral object by
        differentiating under the integral sign [1], using the Fundamental
        Theorem of Calculus [2] when possible.

        Explanation
        ===========

        Whenever an Integral is encountered that is equivalent to zero or
        has an integrand that is independent of the variable of integration
        those integrals are performed. All others are returned as Integral
        instances which can be resolved with doit() (provided they are integrable).

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Differentiation_under_the_integral_sign
        .. [2] https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> i = Integral(x + y, y, (y, 1, x))
        >>> i.diff(x)
        Integral(x + y, (y, x)) + Integral(1, y, (y, 1, x))
        >>> i.doit().diff(x) == i.diff(x).doit()
        True
        >>> i.diff(y)
        0

        The previous must be true since there is no y in the evaluated integral:

        >>> i.free_symbols
        {x}
        >>> i.doit()
        2*x**3/3 - x/2 - 1/6

        """

        # differentiate under the integral sign; we do not
        # check for regularity conditions (TODO), see issue 4215

        # get limits and the function
        f, limits = self.function, list(self.limits)

        # the order matters if variables of integration appear in the limits
        # so work our way in from the outside to the inside.
        limit = limits.pop(-1)
        if len(limit) == 3:
            x, a, b = limit
        elif len(limit) == 2:
            x, b = limit
            a = None
        else:
            a = b = None
            x = limit[0]

        if limits:  # f is the argument to an integral
            f = self.func(f, *tuple(limits))

        # assemble the pieces
        def _do(f, ab):
            dab_dsym = diff(ab, sym)
            if not dab_dsym:
                return S.Zero
            if isinstance(f, Integral):
                limits = [(x, x) if (len(l) == 1 and l[0] == x) else l
                          for l in f.limits]
                f = self.func(f.function, *limits)
            return f.subs(x, ab)*dab_dsym

        rv = S.Zero
        if b is not None:
            rv += _do(f, b)
        if a is not None:
            rv -= _do(f, a)
        if len(limit) == 1 and sym == x:
            # the dummy variable *is* also the real-world variable
            arg = f
            rv += arg
        else:
            # the dummy variable might match sym but it's
            # only a dummy and the actual variable is determined
            # by the limits, so mask off the variable of integration
            # while differentiating
            u = Dummy('u')
            arg = f.subs(x, u).diff(sym).subs(u, x)
            if arg:
                rv += self.func(arg, (x, a, b))
        return rv

    def _eval_integral(self, f, x, meijerg=None, risch=None, manual=None,
                       heurisch=None, conds='piecewise',final=None):
        """
        Calculate the anti-derivative to the function f(x).

        Explanation
        ===========

        The following algorithms are applied (roughly in this order):

        1. Simple heuristics (based on pattern matching and integral table):

           - most frequently used functions (e.g. polynomials, products of
             trig functions)

        2. Integration of rational functions:

           - A complete algorithm for integrating rational functions is
             implemented (the Lazard-Rioboo-Trager algorithm).  The algorithm
             also uses the partial fraction decomposition algorithm
             implemented in apart() as a preprocessor to make this process
             faster.  Note that the integral of a rational function is always
             elementary, but in general, it may include a RootSum.

        3. Full Risch algorithm:

           - The Risch algorithm is a complete decision
             procedure for integrating elementary functions, which means that
             given any elementary function, it will either compute an
             elementary antiderivative, or else prove that none exists.
             Currently, part of transcendental case is implemented, meaning
             elementary integrals containing exponentials, logarithms, and
             (soon!) trigonometric functions can be computed.  The algebraic
             case, e.g., functions containing roots, is much more difficult
             and is not implemented yet.

           - If the routine fails (because the integrand is not elementary, or
             because a case is not implemented yet), it continues on to the
             next algorithms below.  If the routine proves that the integrals
             is nonelementary, it still moves on to the algorithms below,
             because we might be able to find a closed-form solution in terms
             of special functions.  If risch=True, however, it will stop here.

        4. The Meijer G-Function algorithm:

           - This algorithm works by first rewriting the integrand in terms of
             very general Meijer G-Function (meijerg in SymPy), integrating
             it, and then rewriting the result back, if possible.  This
             algorithm is particularly powerful for definite integrals (which
             is actually part of a different method of Integral), since it can
             compute closed-form solutions of definite integrals even when no
             closed-form indefinite integral exists.  But it also is capable
             of computing many indefinite integrals as well.

           - Another advantage of this method is that it can use some results
             about the Meijer G-Function to give a result in terms of a
             Piecewise expression, which allows to express conditionally
             convergent integrals.

           - Setting meijerg=True will cause integrate() to use only this
             method.

        5. The "manual integration" algorithm:

           - This algorithm tries to mimic how a person would find an
             antiderivative by hand, for example by looking for a
             substitution or applying integration by parts. This algorithm
             does not handle as many integrands but can return results in a
             more familiar form.

           - Sometimes this algorithm can evaluate parts of an integral; in
             this case integrate() will try to evaluate the rest of the
             integrand using the other methods here.

           - Setting manual=True will cause integrate() to use only this
             method.

        6. The Heuristic Risch algorithm:

           - This is a heuristic version of the Risch algorithm, meaning that
             it is not deterministic.  This is tried as a last resort because
             it can be very slow.  It is still used because not enough of the
             full Risch algorithm is implemented, so that there are still some
             integrals that can only be computed using this method.  The goal
             is to implement enough of the Risch and Meijer G-function methods
             so that this can be deleted.

             Setting heurisch=True will cause integrate() to use only this
             method. Set heurisch=False to not use it.

        """

        from sympy.integrals.risch import risch_integrate, NonElementaryIntegral
        from sympy.integrals.manualintegrate import manualintegrate

        if risch:
            try:
                return risch_integrate(f, x, conds=conds)
            except NotImplementedError:
                return None

        if manual:
            try:
                result = manualintegrate(f, x)
                if result is not None and result.func != Integral:
                    return result
            except (ValueError, PolynomialError):
                pass

        eval_kwargs = {"meijerg": meijerg, "risch": risch, "manual": manual,
            "heurisch": heurisch, "conds": conds}

        # if it is a poly(x) then let the polynomial integrate itself (fast)
        #
        # It is important to make this check first, otherwise the other code
        # will return a SymPy expression instead of a Polynomial.
        #
        # see Polynomial for details.
        if isinstance(f, Poly) and not (manual or meijerg or risch):
            # Note: this is deprecated, but the deprecation warning is already
            # issued in the Integral constructor.
            return f.integrate(x)

        # Piecewise antiderivatives need to call special integrate.
        if isinstance(f, Piecewise):
            return f.piecewise_integrate(x, **eval_kwargs)

        # let's cut it short if `f` does not depend on `x`; if
        # x is only a dummy, that will be handled below
        if not f.has(x):
            return f*x

        # try to convert to poly(x) and then integrate if successful (fast)
        poly = f.as_poly(x)
        if poly is not None and not (manual or meijerg or risch):
            return poly.integrate().as_expr()

        if risch is not False:
            try:
                result, i = risch_integrate(f, x, separate_integral=True,
                    conds=conds)
            except NotImplementedError:
                pass
            else:
                if i:
                    # There was a nonelementary integral. Try integrating it.

                    # if no part of the NonElementaryIntegral is integrated by
                    # the Risch algorithm, then use the original function to
                    # integrate, instead of re-written one
                    if result == 0:
                        return NonElementaryIntegral(f, x).doit(risch=False)
                    else:
                        return result + i.doit(risch=False)
                else:
                    return result

        # since Integral(f=g1+g2+...) == Integral(g1) + Integral(g2) + ...
        # we are going to handle Add terms separately,
        # if `f` is not Add -- we only have one term

        # Note that in general, this is a bad idea, because Integral(g1) +
        # Integral(g2) might not be computable, even if Integral(g1 + g2) is.
        # For example, Integral(x**x + x**x*log(x)).  But many heuristics only
        # work term-wise.  So we compute this step last, after trying
        # risch_integrate.  We also try risch_integrate again in this loop,
        # because maybe the integral is a sum of an elementary part and a
        # nonelementary part (like erf(x) + exp(x)).  risch_integrate() is
        # quite fast, so this is acceptable.
        from sympy.simplify.fu import sincos_to_sum
        parts = []
        args = Add.make_args(f)
        for g in args:
            coeff, g = g.as_independent(x)

            # g(x) = const
            if g is S.One and not meijerg:
                parts.append(coeff*x)
                continue

            # g(x) = expr + O(x**n)
            order_term = g.getO()

            if order_term is not None:
                h = self._eval_integral(g.removeO(), x, **eval_kwargs)

                if h is not None:
                    h_order_expr = self._eval_integral(order_term.expr, x, **eval_kwargs)

                    if h_order_expr is not None:
                        h_order_term = order_term.func(
                            h_order_expr, *order_term.variables)
                        parts.append(coeff*(h + h_order_term))
                        continue

                # NOTE: if there is O(x**n) and we fail to integrate then
                # there is no point in trying other methods because they
                # will fail, too.
                return None

            #               c
            # g(x) = (a*x+b)
            if g.is_Pow and not g.exp.has(x) and not meijerg:
                a = Wild('a', exclude=[x])
                b = Wild('b', exclude=[x])

                M = g.base.match(a*x + b)

                if M is not None:
                    if g.exp == -1:
                        h = log(g.base)
                    elif conds != 'piecewise':
                        h = g.base**(g.exp + 1) / (g.exp + 1)
                    else:
                        h1 = log(g.base)
                        h2 = g.base**(g.exp + 1) / (g.exp + 1)
                        h = Piecewise((h2, Ne(g.exp, -1)), (h1, True))

                    parts.append(coeff * h / M[a])
                    continue

            #        poly(x)
            # g(x) = -------
            #        poly(x)
            if g.is_rational_function(x) and not (manual or meijerg or risch):
                parts.append(coeff * ratint(g, x))
                continue

            if not (manual or meijerg or risch):
                # g(x) = Mul(trig)
                h = trigintegrate(g, x, conds=conds)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # g(x) has at least a DiracDelta term
                h = deltaintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                from .singularityfunctions import singularityintegrate
                # g(x) has at least a Singularity Function term
                h = singularityintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # Try risch again.
                if risch is not False:
                    try:
                        h, i = risch_integrate(g, x,
                            separate_integral=True, conds=conds)
                    except NotImplementedError:
                        h = None
                    else:
                        if i:
                            h = h + i.doit(risch=False)

                        parts.append(coeff*h)
                        continue

                # fall back to heurisch
                if heurisch is not False:
                    from sympy.integrals.heurisch import (heurisch as heurisch_,
                                                          heurisch_wrapper)
                    try:
                        if conds == 'piecewise':
                            h = heurisch_wrapper(g, x, hints=[])
                        else:
                            h = heurisch_(g, x, hints=[])
                    except PolynomialError:
                        # XXX: this exception means there is a bug in the
                        # implementation of heuristic Risch integration
                        # algorithm.
                        h = None
            else:
                h = None

            if meijerg is not False and h is None:
                # rewrite using G functions
                try:
                    h = meijerint_indefinite(g, x)
                except NotImplementedError:
                    _debug('NotImplementedError from meijerint_definite')
                if h is not None:
                    parts.append(coeff * h)
                    continue

            if h is None and manual is not False:
                try:
                    result = manualintegrate(g, x)
                    if result is not None and not isinstance(result, Integral):
                        if result.has(Integral) and not manual:
                            # Try to have other algorithms do the integrals
                            # manualintegrate can't handle,
                            # unless we were asked to use manual only.
                            # Keep the rest of eval_kwargs in case another
                            # method was set to False already
                            new_eval_kwargs = eval_kwargs
                            new_eval_kwargs["manual"] = False
                            new_eval_kwargs["final"] = False
                            result = result.func(*[
                                arg.doit(**new_eval_kwargs) if
                                arg.has(Integral) else arg
                                for arg in result.args
                            ]).expand(multinomial=False,
                                      log=False,
                                      power_exp=False,
                                      power_base=False)
                        if not result.has(Integral):
                            parts.append(coeff * result)
                            continue
                except (ValueError, PolynomialError):
                    # can't handle some SymPy expressions
                    pass

            # if we failed maybe it was because we had
            # a product that could have been expanded,
            # so let's try an expansion of the whole
            # thing before giving up; we don't try this
            # at the outset because there are things
            # that cannot be solved unless they are
            # NOT expanded e.g., x**x*(1+log(x)). There
            # should probably be a checker somewhere in this
            # routine to look for such cases and try to do
            # collection on the expressions if they are already
            # in an expanded form
            if not h and len(args) == 1:
                f = sincos_to_sum(f).expand(mul=True, deep=False)
                if f.is_Add:
                    # Note: risch will be identical on the expanded
                    # expression, but maybe it will be able to pick out parts,
                    # like x*(exp(x) + erf(x)).
                    return self._eval_integral(f, x, **eval_kwargs)

            if h is not None:
                parts.append(coeff * h)
            else:
                return None

        return Add(*parts)

    def _eval_lseries(self, x, logx=None, cdir=0):
        expr = self.as_dummy()
        symb = x
