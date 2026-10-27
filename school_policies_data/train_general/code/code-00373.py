            Total length (diameter) of horizontal axis.
        height : float
            Total length (diameter) of vertical axis.
        angle : float, default: 0
            Rotation in degrees anti-clockwise.

        Notes
        -----
        Valid keyword arguments are:

        %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)

        self._center = xy
        self._width, self._height = width, height
        self._angle = angle
        self._path = Path.unit_circle()
        # Required for EllipseSelector with axes aspect ratio != 1
        # The patch is defined in data coordinates and when changing the
        # selector with square modifier and not in data coordinates, we need
        # to correct for the aspect ratio difference between the data and
        # display coordinate systems.
        self._aspect_ratio_correction = 1.0
        # Note: This cannot be calculated until this is added to an Axes
        self._patch_transform = transforms.IdentityTransform()

    def _recompute_transform(self):
        """
        Notes
        -----
        This cannot be called until after this has been added to an Axes,
        otherwise unit conversion will fail. This makes it very important to
        call the accessor method and not directly access the transformation
        member variable.
        """
        center = (self.convert_xunits(self._center[0]),
                  self.convert_yunits(self._center[1]))
        width = self.convert_xunits(self._width)
        height = self.convert_yunits(self._height)
        self._patch_transform = transforms.Affine2D() \
            .scale(width * 0.5, height * 0.5 * self._aspect_ratio_correction) \
            .rotate_deg(self.angle) \
            .scale(1, 1 / self._aspect_ratio_correction) \
            .translate(*center)

    def get_path(self):
        """Return the path of the ellipse."""
        return self._path

    def get_patch_transform(self):
        self._recompute_transform()
        return self._patch_transform

    def set_center(self, xy):
        """
        Set the center of the ellipse.

        Parameters
        ----------
        xy : (float, float)
        """
        self._center = xy
        self.stale = True

    def get_center(self):
        """Return the center of the ellipse."""
        return self._center

    center = property(get_center, set_center)

    def set_width(self, width):
        """
        Set the width of the ellipse.

        Parameters
        ----------
        width : float
        """
        self._width = width
        self.stale = True

    def get_width(self):
        """
        Return the width of the ellipse.
        """
        return self._width

    width = property(get_width, set_width)

    def set_height(self, height):
        """
        Set the height of the ellipse.

        Parameters
        ----------
        height : float
        """
        self._height = height
        self.stale = True

    def get_height(self):
        """Return the height of the ellipse."""
        return self._height

    height = property(get_height, set_height)

    def set_angle(self, angle):
        """
        Set the angle of the ellipse.

        Parameters
        ----------
        angle : float
        """
        self._angle = angle
        self.stale = True

    def get_angle(self):
        """Return the angle of the ellipse."""
        return self._angle

    angle = property(get_angle, set_angle)

    def get_corners(self):
        """
        Return the corners of the ellipse bounding box.

        The bounding box orientation is moving anti-clockwise from the
        lower left corner defined before rotation.
        """
        return self.get_patch_transform().transform(
            [(-1, -1), (1, -1), (1, 1), (-1, 1)])

    def get_vertices(self):
        """
        Return the vertices coordinates of the ellipse.

        The definition can be found `here <https://en.wikipedia.org/wiki/Ellipse>`_

        .. versionadded:: 3.8
        """
        if self.width < self.height:
            ret = self.get_patch_transform().transform([(0, 1), (0, -1)])
        else:
            ret = self.get_patch_transform().transform([(1, 0), (-1, 0)])
        return [tuple(x) for x in ret]

    def get_co_vertices(self):
        """
        Return the co-vertices coordinates of the ellipse.

        The definition can be found `here <https://en.wikipedia.org/wiki/Ellipse>`_

        .. versionadded:: 3.8
        """
        if self.width < self.height:
            ret = self.get_patch_transform().transform([(1, 0), (-1, 0)])
        else:
            ret = self.get_patch_transform().transform([(0, 1), (0, -1)])
        return [tuple(x) for x in ret]


class Annulus(Patch):
    """
    An elliptical annulus.
    """

    @_docstring.interpd
    def __init__(self, xy, r, width, angle=0.0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            xy coordinates of annulus centre.
        r : float or (float, float)
            The radius, or semi-axes:

            - If float: radius of the outer circle.
            - If two floats: semi-major and -minor axes of outer ellipse.
        width : float
            Width (thickness) of the annular ring. The width is measured inward
            from the outer ellipse so that for the inner ellipse the semi-axes
            are given by ``r - width``. *width* must be less than or equal to
            the semi-minor axis.
        angle : float, default: 0
            Rotation angle in degrees (anti-clockwise from the positive
            x-axis). Ignored for circular annuli (i.e., if *r* is a scalar).
        **kwargs
            Keyword arguments control the `Patch` properties:

            %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)

        self.set_radii(r)
        self.center = xy
        self.width = width
        self.angle = angle
        self._path = None

    def __str__(self):
        if self.a == self.b:
            r = self.a
        else:
            r = (self.a, self.b)

        return "Annulus(xy=(%s, %s), r=%s, width=%s, angle=%s)" % \
                (*self.center, r, self.width, self.angle)

    def set_center(self, xy):
        """
        Set the center of the annulus.

        Parameters
        ----------
        xy : (float, float)
        """
        self._center = xy
        self._path = None
        self.stale = True

    def get_center(self):
        """Return the center of the annulus."""
        return self._center

    center = property(get_center, set_center)

    def set_width(self, width):
        """
        Set the width (thickness) of the annulus ring.

        The width is measured inwards from the outer ellipse.

        Parameters
        ----------
        width : float
        """
        if width > min(self.a, self.b):
            raise ValueError(
                'Width of annulus must be less than or equal to semi-minor axis')

        self._width = width
        self._path = None
        self.stale = True

    def get_width(self):
        """Return the width (thickness) of the annulus ring."""
        return self._width

    width = property(get_width, set_width)

    def set_angle(self, angle):
        """
        Set the tilt angle of the annulus.

        Parameters
        ----------
        angle : float
        """
        self._angle = angle
        self._path = None
        self.stale = True

    def get_angle(self):
        """Return the angle of the annulus."""
        return self._angle

    angle = property(get_angle, set_angle)

    def set_semimajor(self, a):
        """
        Set the semi-major axis *a* of the annulus.

        Parameters
        ----------
        a : float
        """
        self.a = float(a)
        self._path = None
        self.stale = True

    def set_semiminor(self, b):
        """
        Set the semi-minor axis *b* of the annulus.

        Parameters
        ----------
        b : float
        """
        self.b = float(b)
        self._path = None
        self.stale = True

    def set_radii(self, r):
        """
        Set the semi-major (*a*) and semi-minor radii (*b*) of the annulus.

        Parameters
        ----------
        r : float or (float, float)
            The radius, or semi-axes:

            - If float: radius of the outer circle.
            - If two floats: semi-major and -minor axes of outer ellipse.
        """
        if np.shape(r) == (2,):
            self.a, self.b = r
        elif np.shape(r) == ():
            self.a = self.b = float(r)
        else:
            raise ValueError("Parameter 'r' must be one or two floats.")

        self._path = None
        self.stale = True

    def get_radii(self):
        """Return the semi-major and semi-minor radii of the annulus."""
        return self.a, self.b

    radii = property(get_radii, set_radii)

    def _transform_verts(self, verts, a, b):
        return transforms.Affine2D() \
            .scale(*self._convert_xy_units((a, b))) \
            .rotate_deg(self.angle) \
            .translate(*self._convert_xy_units(self.center)) \
            .transform(verts)

    def _recompute_path(self):
        # circular arc
        arc = Path.arc(0, 360)

        # annulus needs to draw an outer ring
        # followed by a reversed and scaled inner ring
        a, b, w = self.a, self.b, self.width
        v1 = self._transform_verts(arc.vertices, a, b)
        v2 = self._transform_verts(arc.vertices[::-1], a - w, b - w)
        v = np.vstack([v1, v2, v1[0, :], (0, 0)])
        c = np.hstack([arc.codes, Path.MOVETO,
                       arc.codes[1:], Path.MOVETO,
                       Path.CLOSEPOLY])
        self._path = Path(v, c)

    def get_path(self):
        if self._path is None:
            self._recompute_path()
        return self._path


class Circle(Ellipse):
    """
    A circle patch.
    """
    def __str__(self):
        pars = self.center[0], self.center[1], self.radius
        fmt = "Circle(xy=(%g, %g), radius=%g)"
        return fmt % pars

    @_docstring.interpd
    def __init__(self, xy, radius=5, **kwargs):
        """
        Create a true circle at center *xy* = (*x*, *y*) with given *radius*.

        Unlike `CirclePolygon` which is a polygonal approximation, this uses
        Bezier splines and is much closer to a scale-free circle.

        Valid keyword arguments are:

        %(Patch:kwdoc)s
        """
        super().__init__(xy, radius * 2, radius * 2, **kwargs)
        self.radius = radius

    def set_radius(self, radius):
        """
        Set the radius of the circle.

        Parameters
        ----------
        radius : float
        """
        self.width = self.height = 2 * radius
        self.stale = True

    def get_radius(self):
        """Return the radius of the circle."""
        return self.width / 2.

    radius = property(get_radius, set_radius)


class Arc(Ellipse):
    """
    An elliptical arc, i.e. a segment of an ellipse.

    Due to internal optimizations, the arc cannot be filled.
    """

    def __str__(self):
        pars = (self.center[0], self.center[1], self.width,
                self.height, self.angle, self.theta1, self.theta2)
        fmt = ("Arc(xy=(%g, %g), width=%g, "
               "height=%g, angle=%g, theta1=%g, theta2=%g)")
        return fmt % pars

    @_docstring.interpd
    def __init__(self, xy, width, height, *,
                 angle=0.0, theta1=0.0, theta2=360.0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            The center of the ellipse.

        width : float
            The length of the horizontal axis.

        height : float
            The length of the vertical axis.

        angle : float
            Rotation of the ellipse in degrees (counterclockwise).

        theta1, theta2 : float, default: 0, 360
            Starting and ending angles of the arc in degrees. These values
            are relative to *angle*, e.g. if *angle* = 45 and *theta1* = 90
            the absolute starting angle is 135.
            Default *theta1* = 0, *theta2* = 360, i.e. a complete ellipse.
            The arc is drawn in the counterclockwise direction.
            Angles greater than or equal to 360, or smaller than 0, are
            represented by an equivalent angle in the range [0, 360), by
            taking the input value mod 360.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` properties
            Most `.Patch` properties are supported as keyword arguments,
            except *fill* and *facecolor* because filling is not supported.

        %(Patch:kwdoc)s
        """
        fill = kwargs.setdefault('fill', False)
        if fill:
            raise ValueError("Arc objects cannot be filled")

        super().__init__(xy, width, height, angle=angle, **kwargs)

        self.theta1 = theta1
        self.theta2 = theta2
        (self._theta1, self._theta2, self._stretched_width,
         self._stretched_height) = self._theta_stretch()
        self._path = Path.arc(self._theta1, self._theta2)

    @artist.allow_rasterization
    def draw(self, renderer):
        """
        Draw the arc to the given *renderer*.

        Notes
        -----
        Ellipses are normally drawn using an approximation that uses
        eight cubic Bezier splines.  The error of this approximation
        is 1.89818e-6, according to this unverified source:

          Lancaster, Don.  *Approximating a Circle or an Ellipse Using
          Four Bezier Cubic Splines.*

          https://www.tinaja.com/glib/ellipse4.pdf

        There is a use case where very large ellipses must be drawn
        with very high accuracy, and it is too expensive to render the
        entire ellipse with enough segments (either splines or line
        segments).  Therefore, in the case where either radius of the
        ellipse is large enough that the error of the spline
        approximation will be visible (greater than one pixel offset
        from the ideal), a different technique is used.

        In that case, only the visible parts of the ellipse are drawn,
        with each visible arc using a fixed number of spline segments
        (8).  The algorithm proceeds as follows:

        1. The points where the ellipse intersects the axes (or figure)
           bounding box are located.  (This is done by performing an inverse
           transformation on the bbox such that it is relative to the unit
           circle -- this makes the intersection calculation much easier than
           doing rotated ellipse intersection directly.)

           This uses the "line intersecting a circle" algorithm from:

               Vince, John.  *Geometry for Computer Graphics: Formulae,
               Examples & Proofs.*  London: Springer-Verlag, 2005.

        2. The angles of each of the intersection points are calculated.

        3. Proceeding counterclockwise starting in the positive
           x-direction, each of the visible arc-segments between the
           pairs of vertices are drawn using the Bezier arc
           approximation technique implemented in `.Path.arc`.
        """
        if not self.get_visible():
            return

        self._recompute_transform()

        self._update_path()
        # Get width and height in pixels we need to use
        # `self.get_data_transform` rather than `self.get_transform`
        # because we want the transform from dataspace to the
        # screen space to estimate how big the arc will be in physical
        # units when rendered (the transform that we get via
        # `self.get_transform()` goes from an idealized unit-radius
        # space to screen space).
        data_to_screen_trans = self.get_data_transform()
        pwidth, pheight = (
            data_to_screen_trans.transform((self._stretched_width,
                                            self._stretched_height)) -
            data_to_screen_trans.transform((0, 0)))
        inv_error = (1.0 / 1.89818e-6) * 0.5

        if pwidth < inv_error and pheight < inv_error:
            return Patch.draw(self, renderer)

        def line_circle_intersect(x0, y0, x1, y1):
            dx = x1 - x0
            dy = y1 - y0
            dr2 = dx * dx + dy * dy
            D = x0 * y1 - x1 * y0
            D2 = D * D
            discrim = dr2 - D2
            if discrim >= 0.0:
                sign_dy = np.copysign(1, dy)  # +/-1, never 0.
                sqrt_discrim = np.sqrt(discrim)
                return np.array(
                    [[(D * dy + sign_dy * dx * sqrt_discrim) / dr2,
                      (-D * dx + abs(dy) * sqrt_discrim) / dr2],
                     [(D * dy - sign_dy * dx * sqrt_discrim) / dr2,
                      (-D * dx - abs(dy) * sqrt_discrim) / dr2]])
            else:
                return np.empty((0, 2))

        def segment_circle_intersect(x0, y0, x1, y1):
            epsilon = 1e-9
            if x1 < x0:
                x0e, x1e = x1, x0
            else:
                x0e, x1e = x0, x1
            if y1 < y0:
                y0e, y1e = y1, y0
            else:
                y0e, y1e = y0, y1
            xys = line_circle_intersect(x0, y0, x1, y1)
            xs, ys = xys.T
            return xys[
                (x0e - epsilon < xs) & (xs < x1e + epsilon)
                & (y0e - epsilon < ys) & (ys < y1e + epsilon)
            ]

        # Transform the Axes (or figure) box_path so that it is relative to
        # the unit circle in the same way that it is relative to the desired
        # ellipse.
        box_path_transform = (
            transforms.BboxTransformTo((self.axes or self.get_figure(root=False)).bbox)
            - self.get_transform())
        box_path = Path.unit_rectangle().transformed(box_path_transform)

        thetas = set()
        # For each of the point pairs, there is a line segment
        for p0, p1 in zip(box_path.vertices[:-1], box_path.vertices[1:]):
            xy = segment_circle_intersect(*p0, *p1)
            x, y = xy.T
            # arctan2 return [-pi, pi), the rest of our angles are in
            # [0, 360], adjust as needed.
            theta = (np.rad2deg(np.arctan2(y, x)) + 360) % 360
            thetas.update(
                theta[(self._theta1 < theta) & (theta < self._theta2)])
        thetas = sorted(thetas) + [self._theta2]
        last_theta = self._theta1
        theta1_rad = np.deg2rad(self._theta1)
        inside = box_path.contains_point(
            (np.cos(theta1_rad), np.sin(theta1_rad))
        )

        # save original path
        path_original = self._path
        for theta in thetas:
            if inside:
                self._path = Path.arc(last_theta, theta, 8)
                Patch.draw(self, renderer)
                inside = False
            else:
                inside = True
            last_theta = theta

        # restore original path
        self._path = path_original

    def _update_path(self):
        # Compute new values and update and set new _path if any value changed
        stretched = self._theta_stretch()
        if any(a != b for a, b in zip(
                stretched, (self._theta1, self._theta2, self._stretched_width,
                            self._stretched_height))):
            (self._theta1, self._theta2, self._stretched_width,
             self._stretched_height) = stretched
            self._path = Path.arc(self._theta1, self._theta2)

    def _theta_stretch(self):
        # If the width and height of ellipse are not equal, take into account
        # stretching when calculating angles to draw between
        def theta_stretch(theta, scale):
            theta = np.deg2rad(theta)
            x = np.cos(theta)
            y = np.sin(theta)
            stheta = np.rad2deg(np.arctan2(scale * y, x))
            # arctan2 has the range [-pi, pi], we expect [0, 2*pi]
            return (stheta + 360) % 360

        width = self.convert_xunits(self.width)
        height = self.convert_yunits(self.height)
        if (
            # if we need to stretch the angles because we are distorted
            width != height
            # and we are not doing a full circle.
            #
            # 0 and 360 do not exactly round-trip through the angle
            # stretching (due to both float precision limitations and
            # the difference between the range of arctan2 [-pi, pi] and
            # this method [0, 360]) so avoid doing it if we don't have to.
            and not (self.theta1 != self.theta2 and
                     self.theta1 % 360 == self.theta2 % 360)
        ):
            theta1 = theta_stretch(self.theta1, width / height)
            theta2 = theta_stretch(self.theta2, width / height)
            return theta1, theta2, width, height
        return self.theta1, self.theta2, width, height


def bbox_artist(artist, renderer, props=None, fill=True):
    """
    A debug function to draw a rectangle around the bounding
    box returned by an artist's `.Artist.get_window_extent`
    to test whether the artist is returning the correct bbox.

    *props* is a dict of rectangle props with the additional property
    'pad' that sets the padding around the bbox in points.
    """
    if props is None:
        props = {}
    props = props.copy()  # don't want to alter the pad externally
    pad = props.pop('pad', 4)
    pad = renderer.points_to_pixels(pad)
    bbox = artist.get_window_extent(renderer)
    r = Rectangle(
        xy=(bbox.x0 - pad / 2, bbox.y0 - pad / 2),
        width=bbox.width + pad, height=bbox.height + pad,
        fill=fill, transform=transforms.IdentityTransform(), clip_on=False)
    r.update(props)
    r.draw(renderer)


def draw_bbox(bbox, renderer, color='k', trans=None):
    """
    A debug function to draw a rectangle around the bounding
    box returned by an artist's `.Artist.get_window_extent`
    to test whether the artist is returning the correct bbox.
    """
    r = Rectangle(xy=bbox.p0, width=bbox.width, height=bbox.height,
                  edgecolor=color, fill=False, clip_on=False)
    if trans is not None:
        r.set_transform(trans)
    r.draw(renderer)


class _Style:
    """
    A base class for the Styles. It is meant to be a container class,
    where actual styles are declared as subclass of it, and it
    provides some helper functions.
    """

    def __init_subclass__(cls):
        # Automatically perform docstring interpolation on the subclasses:
        # This allows listing the supported styles via
        # - %(BoxStyle:table)s
        # - %(ConnectionStyle:table)s
        # - %(ArrowStyle:table)s
        # and additionally adding .. ACCEPTS: blocks via
        # - %(BoxStyle:table_and_accepts)s
        # - %(ConnectionStyle:table_and_accepts)s
        # - %(ArrowStyle:table_and_accepts)s
        _docstring.interpd.register(**{
            f"{cls.__name__}:table": cls.pprint_styles(),
            f"{cls.__name__}:table_and_accepts": (
                cls.pprint_styles()
                + "\n\n    .. ACCEPTS: ["
                + "|".join(map(" '{}' ".format, cls._style_list))
                + "]")
        })

    def __new__(cls, stylename, **kwargs):
        """Return the instance of the subclass with the given style name."""
        # The "class" should have the _style_list attribute, which is a mapping
        # of style names to style classes.
        _list = stylename.replace(" ", "").split(",")
        _name = _list[0].lower()
        try:
            _cls = cls._style_list[_name]
        except KeyError as err:
            raise ValueError(f"Unknown style: {stylename!r}") from err
        try:
            _args_pair = [cs.split("=") for cs in _list[1:]]
            _args = {k: float(v) for k, v in _args_pair}
        except ValueError as err:
            raise ValueError(
                f"Incorrect style argument: {stylename!r}") from err
        return _cls(**{**_args, **kwargs})

    @classmethod
    def get_styles(cls):
        """Return a dictionary of available styles."""
        return cls._style_list

    @classmethod
    def pprint_styles(cls):
        """Return the available styles as pretty-printed string."""
        table = [('Class', 'Name', 'Parameters'),
                 *[(cls.__name__,
                    # Add backquotes, as - and | have special meaning in reST.
                    f'``{name}``',
                    # [1:-1] drops the surrounding parentheses.
                    str(inspect.signature(cls))[1:-1] or 'None')
                   for name, cls in cls._style_list.items()]]
        # Convert to rst table.
        col_len = [max(len(cell) for cell in column) for column in zip(*table)]
        table_formatstr = '  '.join('=' * cl for cl in col_len)
        rst_table = '\n'.join([
            '',
            table_formatstr,
            '  '.join(cell.ljust(cl) for cell, cl in zip(table[0], col_len)),
            table_formatstr,
            *['  '.join(cell.ljust(cl) for cell, cl in zip(row, col_len))
              for row in table[1:]],
            table_formatstr,
        ])
        return textwrap.indent(rst_table, prefix=' ' * 4)

    @classmethod
    @_api.deprecated(
        '3.10.0',
        message="This method is never used internally.",
        alternative="No replacement.  Please open an issue if you use this."
    )
    def register(cls, name, style):
        """Register a new style."""
        if not issubclass(style, cls._Base):
            raise ValueError(f"{style} must be a subclass of {cls._Base}")
        cls._style_list[name] = style


def _register_style(style_list, cls=None, *, name=None):
    """Class decorator that stashes a class in a (style) dictionary."""
    if cls is None:
        return functools.partial(_register_style, style_list, name=name)
    style_list[name or cls.__name__.lower()] = cls
    return cls


@_docstring.interpd
class BoxStyle(_Style):
    """
    `BoxStyle` is a container class which defines several
    boxstyle classes, which are used for `FancyBboxPatch`.

    A style object can be created as::

           BoxStyle.Round(pad=0.2)

    or::

           BoxStyle("Round", pad=0.2)

    or::

           BoxStyle("Round, pad=0.2")

    The following boxstyle classes are defined.

    %(BoxStyle:table)s

    An instance of a boxstyle class is a callable object, with the signature ::

       __call__(self, x0, y0, width, height, mutation_size) -> Path

    *x0*, *y0*, *width* and *height* specify the location and size of the box
    to be drawn; *mutation_size* scales the outline properties such as padding.
    """

    _style_list = {}

    @_register_style(_style_list)
    class Square:
        """A square box."""

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            pad = mutation_size * self.pad
            # width and height with padding added.
            width, height = width + 2 * pad, height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad
            x1, y1 = x0 + width, y0 + height
            return Path._create_closed(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    @_register_style(_style_list)
    class Circle:
        """A circular box."""

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            pad = mutation_size * self.pad
            width, height = width + 2 * pad, height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad
            return Path.circle((x0 + width / 2, y0 + height / 2),
                                max(width, height) / 2)

    @_register_style(_style_list)
    class Ellipse:
        """
        An elliptical box.

        .. versionadded:: 3.7
        """

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            pad = mutation_size * self.pad
            width, height = width + 2 * pad, height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad
            a = width / math.sqrt(2)
            b = height / math.sqrt(2)
            trans = Affine2D().scale(a, b).translate(x0 + width / 2,
                                                     y0 + height / 2)
            return trans.transform_path(Path.unit_circle())

    @_register_style(_style_list)
    class LArrow:
        """A box in the shape of a left-pointing arrow."""

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            # padding
            pad = mutation_size * self.pad
            # width and height with padding added.
            width, height = width + 2 * pad, height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad,
            x1, y1 = x0 + width, y0 + height

            dx = (y1 - y0) / 2
            dxx = dx / 2
            x0 = x0 + pad / 1.4  # adjust by ~sqrt(2)

            return Path._create_closed(
                [(x0 + dxx, y0), (x1, y0), (x1, y1), (x0 + dxx, y1),
                 (x0 + dxx, y1 + dxx), (x0 - dx, y0 + dx),
                 (x0 + dxx, y0 - dxx),  # arrow
                 (x0 + dxx, y0)])

    @_register_style(_style_list)
    class RArrow(LArrow):
        """A box in the shape of a right-pointing arrow."""

        def __call__(self, x0, y0, width, height, mutation_size):
            p = BoxStyle.LArrow.__call__(
                self, x0, y0, width, height, mutation_size)
            p.vertices[:, 0] = 2 * x0 + width - p.vertices[:, 0]
            return p

    @_register_style(_style_list)
    class DArrow:
        """A box in the shape of a two-way arrow."""
        # Modified from LArrow to add a right arrow to the bbox.

        def __init__(self, pad=0.3):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            """
            self.pad = pad

        def __call__(self, x0, y0, width, height, mutation_size):
            # padding
            pad = mutation_size * self.pad
            # width and height with padding added.
            # The width is padded by the arrows, so we don't need to pad it.
            height = height + 2 * pad
            # boundary of the padded box
            x0, y0 = x0 - pad, y0 - pad
            x1, y1 = x0 + width, y0 + height

            dx = (y1 - y0) / 2
            dxx = dx / 2
            x0 = x0 + pad / 1.4  # adjust by ~sqrt(2)

            return Path._create_closed([
                (x0 + dxx, y0), (x1, y0),  # bot-segment
                (x1, y0 - dxx), (x1 + dx + dxx, y0 + dx),
                (x1, y1 + dxx),  # right-arrow
                (x1, y1), (x0 + dxx, y1),  # top-segment
                (x0 + dxx, y1 + dxx), (x0 - dx, y0 + dx),
                (x0 + dxx, y0 - dxx),  # left-arrow
                (x0 + dxx, y0)])

    @_register_style(_style_list)
    class Round:
        """A box with round corners."""

        def __init__(self, pad=0.3, rounding_size=None):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            rounding_size : float, default: *pad*
                Radius of the corners.
            """
            self.pad = pad
            self.rounding_size = rounding_size

        def __call__(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # size of the rounding corner
            if self.rounding_size:
                dr = mutation_size * self.rounding_size
            else:
                dr = pad

            width, height = width + 2 * pad, height + 2 * pad

            x0, y0 = x0 - pad, y0 - pad,
            x1, y1 = x0 + width, y0 + height

            # Round corners are implemented as quadratic Bezier, e.g.,
            # [(x0, y0-dr), (x0, y0), (x0+dr, y0)] for lower left corner.
            cp = [(x0 + dr, y0),
                  (x1 - dr, y0),
                  (x1, y0), (x1, y0 + dr),
                  (x1, y1 - dr),
                  (x1, y1), (x1 - dr, y1),
                  (x0 + dr, y1),
                  (x0, y1), (x0, y1 - dr),
                  (x0, y0 + dr),
                  (x0, y0), (x0 + dr, y0),
                  (x0 + dr, y0)]

            com = [Path.MOVETO,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.CLOSEPOLY]

            return Path(cp, com)

    @_register_style(_style_list)
    class Round4:
        """A box with rounded edges."""

        def __init__(self, pad=0.3, rounding_size=None):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            rounding_size : float, default: *pad*/2
                Rounding of edges.
            """
            self.pad = pad
            self.rounding_size = rounding_size

        def __call__(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # Rounding size; defaults to half of the padding.
            if self.rounding_size:
                dr = mutation_size * self.rounding_size
            else:
                dr = pad / 2.

            width = width + 2 * pad - 2 * dr
            height = height + 2 * pad - 2 * dr

            x0, y0 = x0 - pad + dr, y0 - pad + dr,
            x1, y1 = x0 + width, y0 + height

            cp = [(x0, y0),
                  (x0 + dr, y0 - dr), (x1 - dr, y0 - dr), (x1, y0),
                  (x1 + dr, y0 + dr), (x1 + dr, y1 - dr), (x1, y1),
                  (x1 - dr, y1 + dr), (x0 + dr, y1 + dr), (x0, y1),
                  (x0 - dr, y1 - dr), (x0 - dr, y0 + dr), (x0, y0),
                  (x0, y0)]

            com = [Path.MOVETO,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CLOSEPOLY]

            return Path(cp, com)

    @_register_style(_style_list)
    class Sawtooth:
        """A box with a sawtooth outline."""

        def __init__(self, pad=0.3, tooth_size=None):
            """
            Parameters
            ----------
            pad : float, default: 0.3
                The amount of padding around the original box.
            tooth_size : float, default: *pad*/2
                Size of the sawtooth.
            """
            self.pad = pad
            self.tooth_size = tooth_size

        def _get_sawtooth_vertices(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # size of sawtooth
            if self.tooth_size is None:
                tooth_size = self.pad * .5 * mutation_size
            else:
                tooth_size = self.tooth_size * mutation_size

            hsz = tooth_size / 2
            width = width + 2 * pad - tooth_size
            height = height + 2 * pad - tooth_size

            # the sizes of the vertical and horizontal sawtooth are
            # separately adjusted to fit the given box size.
            dsx_n = round((width - tooth_size) / (tooth_size * 2)) * 2
            dsy_n = round((height - tooth_size) / (tooth_size * 2)) * 2

            x0, y0 = x0 - pad + hsz, y0 - pad + hsz
            x1, y1 = x0 + width, y0 + height

            xs = [
                x0, *np.linspace(x0 + hsz, x1 - hsz, 2 * dsx_n + 1),  # bottom
                *([x1, x1 + hsz, x1, x1 - hsz] * dsy_n)[:2*dsy_n+2],  # right
                x1, *np.linspace(x1 - hsz, x0 + hsz, 2 * dsx_n + 1),  # top
                *([x0, x0 - hsz, x0, x0 + hsz] * dsy_n)[:2*dsy_n+2],  # left
            ]
            ys = [
                *([y0, y0 - hsz, y0, y0 + hsz] * dsx_n)[:2*dsx_n+2],  # bottom
                y0, *np.linspace(y0 + hsz, y1 - hsz, 2 * dsy_n + 1),  # right
                *([y1, y1 + hsz, y1, y1 - hsz] * dsx_n)[:2*dsx_n+2],  # top
                y1, *np.linspace(y1 - hsz, y0 + hsz, 2 * dsy_n + 1),  # left
            ]

            return [*zip(xs, ys), (xs[0], ys[0])]

        def __call__(self, x0, y0, width, height, mutation_size):
            saw_vertices = self._get_sawtooth_vertices(x0, y0, width,
                                                       height, mutation_size)
            return Path(saw_vertices, closed=True)

    @_register_style(_style_list)
    class Roundtooth(Sawtooth):
        """A box with a rounded sawtooth outline."""

        def __call__(self, x0, y0, width, height, mutation_size):
            saw_vertices = self._get_sawtooth_vertices(x0, y0,
                                                       width, height,
                                                       mutation_size)
            # Add a trailing vertex to allow us to close the polygon correctly
            saw_vertices = np.concatenate([saw_vertices, [saw_vertices[0]]])
            codes = ([Path.MOVETO] +
                     [Path.CURVE3, Path.CURVE3] * ((len(saw_vertices)-1)//2) +
                     [Path.CLOSEPOLY])
            return Path(saw_vertices, codes)


@_docstring.interpd
class ConnectionStyle(_Style):
    """
    `ConnectionStyle` is a container class which defines
    several connectionstyle classes, which is used to create a path
    between two points.  These are mainly used with `FancyArrowPatch`.

    A connectionstyle object can be either created as::

           ConnectionStyle.Arc3(rad=0.2)

    or::

           ConnectionStyle("Arc3", rad=0.2)

    or::

           ConnectionStyle("Arc3, rad=0.2")

    The following classes are defined

    %(ConnectionStyle:table)s

    An instance of any connection style class is a callable object,
    whose call signature is::

        __call__(self, posA, posB,
                 patchA=None, patchB=None,
                 shrinkA=2., shrinkB=2.)

    and it returns a `.Path` instance. *posA* and *posB* are
    tuples of (x, y) coordinates of the two points to be
    connected. *patchA* (or *patchB*) is given, the returned path is
    clipped so that it start (or end) from the boundary of the
    patch. The path is further shrunk by *shrinkA* (or *shrinkB*)
    which is given in points.
    """

    _style_list = {}

    class _Base:
        """
        A base class for connectionstyle classes. The subclass needs
        to implement a *connect* method whose call signature is::

          connect(posA, posB)

        where posA and posB are tuples of x, y coordinates to be
        connected.  The method needs to return a path connecting two
        points. This base class defines a __call__ method, and a few
        helper methods.
        """
        def _in_patch(self, patch):
            """
            Return a predicate function testing whether a point *xy* is
            contained in *patch*.
            """
            return lambda xy: patch.contains(
                SimpleNamespace(x=xy[0], y=xy[1]))[0]

        def _clip(self, path, in_start, in_stop):
            """
            Clip *path* at its start by the region where *in_start* returns
            True, and at its stop by the region where *in_stop* returns True.

            The original path is assumed to start in the *in_start* region and
            to stop in the *in_stop* region.
            """
            if in_start:
                try:
                    _, path = split_path_inout(path, in_start)
                except ValueError:
                    pass
            if in_stop:
                try:
                    path, _ = split_path_inout(path, in_stop)
                except ValueError:
                    pass
            return path

        def __call__(self, posA, posB,
                     shrinkA=2., shrinkB=2., patchA=None, patchB=None):
            """
            Call the *connect* method to create a path between *posA* and
            *posB*; then clip and shrink the path.
            """
            path = self.connect(posA, posB)
            path = self._clip(
                path,
                self._in_patch(patchA) if patchA else None,
                self._in_patch(patchB) if patchB else None,
            )
            path = self._clip(
                path,
                inside_circle(*path.vertices[0], shrinkA) if shrinkA else None,
                inside_circle(*path.vertices[-1], shrinkB) if shrinkB else None
            )
            return path

    @_register_style(_style_list)
    class Arc3(_Base):
        """
        Creates a simple quadratic Bézier curve between two
        points. The curve is created so that the middle control point
        (C1) is located at the same distance from the start (C0) and
        end points(C2) and the distance of the C1 to the line
        connecting C0-C2 is *rad* times the distance of C0-C2.
        """

        def __init__(self, rad=0.):
            """
            Parameters
            ----------
            rad : float
              Curvature of the curve.
            """
            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB
            x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
            dx, dy = x2 - x1, y2 - y1

            f = self.rad

            cx, cy = x12 + f * dy, y12 - f * dx

            vertices = [(x1, y1),
                        (cx, cy),
                        (x2, y2)]
            codes = [Path.MOVETO,
                     Path.CURVE3,
                     Path.CURVE3]

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Angle3(_Base):
        """
        Creates a simple quadratic Bézier curve between two points. The middle
        control point is placed at the intersecting point of two lines which
        cross the start and end point, and have a slope of *angleA* and
        *angleB*, respectively.
        """

        def __init__(self, angleA=90, angleB=0):
            """
            Parameters
            ----------
            angleA : float
              Starting angle of the path.

            angleB : float
              Ending angle of the path.
            """

            self.angleA = angleA
            self.angleB = angleB

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            cosA = math.cos(math.radians(self.angleA))
            sinA = math.sin(math.radians(self.angleA))
            cosB = math.cos(math.radians(self.angleB))
            sinB = math.sin(math.radians(self.angleB))

            cx, cy = get_intersection(x1, y1, cosA, sinA,
                                      x2, y2, cosB, sinB)

            vertices = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Angle(_Base):
        """
        Creates a piecewise continuous quadratic Bézier path between two
        points. The path has a one passing-through point placed at the
        intersecting point of two lines which cross the start and end point,
        and have a slope of *angleA* and *angleB*, respectively.
        The connecting edges are rounded with *rad*.
        """

        def __init__(self, angleA=90, angleB=0, rad=0.):
            """
            Parameters
            ----------
            angleA : float
              Starting angle of the path.

            angleB : float
              Ending angle of the path.

            rad : float
              Rounding radius of the edge.
            """

            self.angleA = angleA
            self.angleB = angleB

            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            cosA = math.cos(math.radians(self.angleA))
            sinA = math.sin(math.radians(self.angleA))
            cosB = math.cos(math.radians(self.angleB))
            sinB = math.sin(math.radians(self.angleB))

            cx, cy = get_intersection(x1, y1, cosA, sinA,
                                      x2, y2, cosB, sinB)

            vertices = [(x1, y1)]
            codes = [Path.MOVETO]

            if self.rad == 0.:
                vertices.append((cx, cy))
                codes.append(Path.LINETO)
            else:
                dx1, dy1 = x1 - cx, y1 - cy
                d1 = np.hypot(dx1, dy1)
                f1 = self.rad / d1
                dx2, dy2 = x2 - cx, y2 - cy
                d2 = np.hypot(dx2, dy2)
                f2 = self.rad / d2
                vertices.extend([(cx + dx1 * f1, cy + dy1 * f1),
                                 (cx, cy),
                                 (cx + dx2 * f2, cy + dy2 * f2)])
                codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])

            vertices.append((x2, y2))
            codes.append(Path.LINETO)

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Arc(_Base):
        """
        Creates a piecewise continuous quadratic Bézier path between two
        points. The path can have two passing-through points, a
        point placed at the distance of *armA* and angle of *angleA* from
        point A, another point with respect to point B. The edges are
        rounded with *rad*.
        """

        def __init__(self, angleA=0, angleB=0, armA=None, armB=None, rad=0.):
            """
            Parameters
            ----------
            angleA : float
              Starting angle of the path.

            angleB : float
              Ending angle of the path.

            armA : float or None
              Length of the starting arm.

            armB : float or None
              Length of the ending arm.

            rad : float
              Rounding radius of the edges.
            """

            self.angleA = angleA
            self.angleB = angleB
            self.armA = armA
            self.armB = armB

            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            vertices = [(x1, y1)]
            rounded = []
            codes = [Path.MOVETO]

            if self.armA:
                cosA = math.cos(math.radians(self.angleA))
                sinA = math.sin(math.radians(self.angleA))
                # x_armA, y_armB
                d = self.armA - self.rad
                rounded.append((x1 + d * cosA, y1 + d * sinA))
                d = self.armA
                rounded.append((x1 + d * cosA, y1 + d * sinA))

            if self.armB:
                cosB = math.cos(math.radians(self.angleB))
                sinB = math.sin(math.radians(self.angleB))
                x_armB, y_armB = x2 + self.armB * cosB, y2 + self.armB * sinB

                if rounded:
                    xp, yp = rounded[-1]
                    dx, dy = x_armB - xp, y_armB - yp
                    dd = (dx * dx + dy * dy) ** .5

                    rounded.append((xp + self.rad * dx / dd,
                                    yp + self.rad * dy / dd))
                    vertices.extend(rounded)
                    codes.extend([Path.LINETO,
                                  Path.CURVE3,
                                  Path.CURVE3])
                else:
                    xp, yp = vertices[-1]
                    dx, dy = x_armB - xp, y_armB - yp
                    dd = (dx * dx + dy * dy) ** .5

                d = dd - self.rad
                rounded = [(xp + d * dx / dd, yp + d * dy / dd),
                           (x_armB, y_armB)]

            if rounded:
                xp, yp = rounded[-1]
                dx, dy = x2 - xp, y2 - yp
                dd = (dx * dx + dy * dy) ** .5

                rounded.append((xp + self.rad * dx / dd,
                                yp + self.rad * dy / dd))
                vertices.extend(rounded)
                codes.extend([Path.LINETO,
                              Path.CURVE3,
                              Path.CURVE3])

            vertices.append((x2, y2))
            codes.append(Path.LINETO)

            return Path(vertices, codes)

    @_register_style(_style_list)
    class Bar(_Base):
        """
        A line with *angle* between A and B with *armA* and *armB*. One of the
        arms is extended so that they are connected in a right angle. The
        length of *armA* is determined by (*armA* + *fraction* x AB distance).
        Same for *armB*.
        """

        def __init__(self, armA=0., armB=0., fraction=0.3, angle=None):
            """
            Parameters
            ----------
            armA : float
                Minimum length of armA.

            armB : float
                Minimum length of armB.

            fraction : float
                A fraction of the distance between two points that will be
                added to armA and armB.

            angle : float or None
                Angle of the connecting line (if None, parallel to A and B).
            """
            self.armA = armA
            self.armB = armB
            self.fraction = fraction
            self.angle = angle

        def connect(self, posA, posB):
            x1, y1 = posA
            x20, y20 = x2, y2 = posB

            theta1 = math.atan2(y2 - y1, x2 - x1)
            dx, dy = x2 - x1, y2 - y1
            dd = (dx * dx + dy * dy) ** .5
            ddx, ddy = dx / dd, dy / dd

            armA, armB = self.armA, self.armB

            if self.angle is not None:
                theta0 = np.deg2rad(self.angle)
                dtheta = theta1 - theta0
                dl = dd * math.sin(dtheta)
                dL = dd * math.cos(dtheta)
                x2, y2 = x1 + dL * math.cos(theta0), y1 + dL * math.sin(theta0)
                armB = armB - dl

                # update
                dx, dy = x2 - x1, y2 - y1
                dd2 = (dx * dx + dy * dy) ** .5
                ddx, ddy = dx / dd2, dy / dd2

            arm = max(armA, armB)
            f = self.fraction * dd + arm

            cx1, cy1 = x1 + f * ddy, y1 - f * ddx
            cx2, cy2 = x2 + f * ddy, y2 - f * ddx

            vertices = [(x1, y1),
                        (cx1, cy1),
                        (cx2, cy2),
                        (x20, y20)]
            codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO]

            return Path(vertices, codes)


def _point_along_a_line(x0, y0, x1, y1, d):
    """
    Return the point on the line connecting (*x0*, *y0*) -- (*x1*, *y1*) whose
    distance from (*x0*, *y0*) is *d*.
    """
    dx, dy = x0 - x1, y0 - y1
    ff = d / (dx * dx + dy * dy) ** .5
    x2, y2 = x0 - ff * dx, y0 - ff * dy

    return x2, y2


@_docstring.interpd
class ArrowStyle(_Style):
    """
    `ArrowStyle` is a container class which defines several
    arrowstyle classes, which is used to create an arrow path along a
    given path.  These are mainly used with `FancyArrowPatch`.

    An arrowstyle object can be either created as::

           ArrowStyle.Fancy(head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy", head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy, head_length=.4, head_width=.4, tail_width=.4")

    The following classes are defined

    %(ArrowStyle:table)s

    For an overview of the visual appearance, see
    :doc:`/gallery/text_labels_and_annotations/fancyarrow_demo`.

    An instance of any arrow style class is a callable object,
    whose call signature is::

        __call__(self, path, mutation_size, linewidth, aspect_ratio=1.)

    and it returns a tuple of a `.Path` instance and a boolean
    value. *path* is a `.Path` instance along which the arrow
    will be drawn. *mutation_size* and *aspect_ratio* have the same
    meaning as in `BoxStyle`. *linewidth* is a line width to be
    stroked. This is meant to be used to correct the location of the
    head so that it does not overshoot the destination point, but not all
    classes support it.

    Notes
    -----
    *angleA* and *angleB* specify the orientation of the bracket, as either a
    clockwise or counterclockwise angle depending on the arrow type. 0 degrees
    means perpendicular to the line connecting the arrow's head and tail.

    .. plot:: gallery/text_labels_and_annotations/angles_on_bracket_arrows.py
    """

    _style_list = {}

    class _Base:
        """
        Arrow Transmuter Base class

        ArrowTransmuterBase and its derivatives are used to make a fancy
        arrow around a given path. The __call__ method returns a path
        (which will be used to create a PathPatch instance) and a boolean
        value indicating the path is open therefore is not fillable.  This
        class is not an artist and actual drawing of the fancy arrow is
        done by the FancyArrowPatch class.
        """

        # The derived classes are required to be able to be initialized
        # w/o arguments, i.e., all its argument (except self) must have
        # the default values.

        @staticmethod
        def ensure_quadratic_bezier(path):
            """
            Some ArrowStyle classes only works with a simple quadratic
            Bézier curve (created with `.ConnectionStyle.Arc3` or
            `.ConnectionStyle.Angle3`). This static method checks if the
            provided path is a simple quadratic Bézier curve and returns its
            control points if true.
            """
            segments = list(path.iter_segments())
            if (len(segments) != 2 or segments[0][1] != Path.MOVETO or
                    segments[1][1] != Path.CURVE3):
                raise ValueError(
                    "'path' is not a valid quadratic Bezier curve")
            return [*segments[0][0], *segments[1][0]]

        def transmute(self, path, mutation_size, linewidth):
            """
            The transmute method is the very core of the ArrowStyle class and
            must be overridden in the subclasses. It receives the *path*
            object along which the arrow will be drawn, and the
            *mutation_size*, with which the arrow head etc. will be scaled.
            The *linewidth* may be used to adjust the path so that it does not
            pass beyond the given points. It returns a tuple of a `.Path`
            instance and a boolean. The boolean value indicate whether the
            path can be filled or not. The return value can also be a list of
            paths and list of booleans of the same length.
            """
            raise NotImplementedError('Derived must override')

        def __call__(self, path, mutation_size, linewidth,
                     aspect_ratio=1.):
            """
            The __call__ method is a thin wrapper around the transmute method
            and takes care of the aspect ratio.
            """

            if aspect_ratio is not None:
                # Squeeze the given height by the aspect_ratio
                vertices = path.vertices / [1, aspect_ratio]
                path_shrunk = Path(vertices, path.codes)
                # call transmute method with squeezed height.
                path_mutated, fillable = self.transmute(path_shrunk,
                                                        mutation_size,
                                                        linewidth)
                if np.iterable(fillable):
                    # Restore the height
                    path_list = [Path(p.vertices * [1, aspect_ratio], p.codes)
                                 for p in path_mutated]
                    return path_list, fillable
                else:
                    return path_mutated, fillable
            else:
                return self.transmute(path, mutation_size, linewidth)

    class _Curve(_Base):
        """
        A simple arrow which will work with any path instance. The
        returned path is the concatenation of the original path, and at
        most two paths representing the arrow head or bracket at the start
        point and at the end point. The arrow heads can be either open
        or closed.
        """

        arrow = "-"
        fillbegin = fillend = False  # Whether arrows are filled.

        def __init__(self, head_length=.4, head_width=.2, widthA=1., widthB=1.,
                     lengthA=0.2, lengthB=0.2, angleA=0, angleB=0, scaleA=None,
                     scaleB=None):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head, relative to *mutation_size*.
            head_width : float, default: 0.2
                Width of the arrow head, relative to *mutation_size*.
            widthA, widthB : float, default: 1.0
                Width of the bracket.
            lengthA, lengthB : float, default: 0.2
                Length of the bracket.
            angleA, angleB : float, default: 0
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            scaleA, scaleB : float, default: *mutation_size*
                The scale of the brackets.
            """

            self.head_length, self.head_width = head_length, head_width
            self.widthA, self.widthB = widthA, widthB
            self.lengthA, self.lengthB = lengthA, lengthB
            self.angleA, self.angleB = angleA, angleB
            self.scaleA, self.scaleB = scaleA, scaleB

            self._beginarrow_head = False
            self._beginarrow_bracket = False
            self._endarrow_head = False
            self._endarrow_bracket = False

            if "-" not in self.arrow:
                raise ValueError("arrow must have the '-' between "
                                 "the two heads")

            beginarrow, endarrow = self.arrow.split("-", 1)

            if beginarrow == "<":
                self._beginarrow_head = True
                self._beginarrow_bracket = False
            elif beginarrow == "<|":
                self._beginarrow_head = True
                self._beginarrow_bracket = False
                self.fillbegin = True
            elif beginarrow in ("]", "|"):
                self._beginarrow_head = False
                self._beginarrow_bracket = True

            if endarrow == ">":
                self._endarrow_head = True
                self._endarrow_bracket = False
            elif endarrow == "|>":
                self._endarrow_head = True
                self._endarrow_bracket = False
                self.fillend = True
            elif endarrow in ("[", "|"):
                self._endarrow_head = False
                self._endarrow_bracket = True

            super().__init__()

        def _get_arrow_wedge(self, x0, y0, x1, y1,
                             head_dist, cos_t, sin_t, linewidth):
            """
            Return the paths for arrow heads. Since arrow lines are
            drawn with capstyle=projected, The arrow goes beyond the
            desired point. This method also returns the amount of the path
            to be shrunken so that it does not overshoot.
            """

            # arrow from x0, y0 to x1, y1
            dx, dy = x0 - x1, y0 - y1

            cp_distance = np.hypot(dx, dy)

            # pad_projected : amount of pad to account the
            # overshooting of the projection of the wedge
            pad_projected = (.5 * linewidth / sin_t)

            # Account for division by zero
            if cp_distance == 0:
                cp_distance = 1

            # apply pad for projected edge
            ddx = pad_projected * dx / cp_distance
            ddy = pad_projected * dy / cp_distance

            # offset for arrow wedge
            dx = dx / cp_distance * head_dist
            dy = dy / cp_distance * head_dist

            dx1, dy1 = cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy
            dx2, dy2 = cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy

            vertices_arrow = [(x1 + ddx + dx1, y1 + ddy + dy1),
                              (x1 + ddx, y1 + ddy),
                              (x1 + ddx + dx2, y1 + ddy + dy2)]
            codes_arrow = [Path.MOVETO,
                           Path.LINETO,
                           Path.LINETO]

            return vertices_arrow, codes_arrow, ddx, ddy

        def _get_bracket(self, x0, y0,
                         x1, y1, width, length, angle):

            cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)

            # arrow from x0, y0 to x1, y1
            from matplotlib.bezier import get_normal_points
            x1, y1, x2, y2 = get_normal_points(x0, y0, cos_t, sin_t, width)

            dx, dy = length * cos_t, length * sin_t

            vertices_arrow = [(x1 + dx, y1 + dy),
                              (x1, y1),
                              (x2, y2),
                              (x2 + dx, y2 + dy)]
            codes_arrow = [Path.MOVETO,
                           Path.LINETO,
                           Path.LINETO,
                           Path.LINETO]

            if angle:
                trans = transforms.Affine2D().rotate_deg_around(x0, y0, angle)
                vertices_arrow = trans.transform(vertices_arrow)

            return vertices_arrow, codes_arrow

        def transmute(self, path, mutation_size, linewidth):
            # docstring inherited
            if self._beginarrow_head or self._endarrow_head:
                head_length = self.head_length * mutation_size
                head_width = self.head_width * mutation_size
                head_dist = np.hypot(head_length, head_width)
                cos_t, sin_t = head_length / head_dist, head_width / head_dist

            scaleA = mutation_size if self.scaleA is None else self.scaleA
            scaleB = mutation_size if self.scaleB is None else self.scaleB

            # begin arrow
            x0, y0 = path.vertices[0]
            x1, y1 = path.vertices[1]

            # If there is no room for an arrow and a line, then skip the arrow
            has_begin_arrow = self._beginarrow_head and (x0, y0) != (x1, y1)
            verticesA, codesA, ddxA, ddyA = (
                self._get_arrow_wedge(x1, y1, x0, y0,
                                      head_dist, cos_t, sin_t, linewidth)
                if has_begin_arrow
                else ([], [], 0, 0)
            )

            # end arrow
            x2, y2 = path.vertices[-2]
            x3, y3 = path.vertices[-1]

            # If there is no room for an arrow and a line, then skip the arrow
            has_end_arrow = self._endarrow_head and (x2, y2) != (x3, y3)
            verticesB, codesB, ddxB, ddyB = (
                self._get_arrow_wedge(x2, y2, x3, y3,
                                      head_dist, cos_t, sin_t, linewidth)
                if has_end_arrow
                else ([], [], 0, 0)
            )

            # This simple code will not work if ddx, ddy is greater than the
            # separation between vertices.
            paths = [Path(np.concatenate([[(x0 + ddxA, y0 + ddyA)],
                                          path.vertices[1:-1],
                                          [(x3 + ddxB, y3 + ddyB)]]),
                          path.codes)]
            fills = [False]

            if has_begin_arrow:
                if self.fillbegin:
                    paths.append(
                        Path([*verticesA, (0, 0)], [*codesA, Path.CLOSEPOLY]))
                    fills.append(True)
                else:
                    paths.append(Path(verticesA, codesA))
                    fills.append(False)
            elif self._beginarrow_bracket:
                x0, y0 = path.vertices[0]
                x1, y1 = path.vertices[1]
                verticesA, codesA = self._get_bracket(x0, y0, x1, y1,
                                                      self.widthA * scaleA,
                                                      self.lengthA * scaleA,
                                                      self.angleA)

                paths.append(Path(verticesA, codesA))
                fills.append(False)

            if has_end_arrow:
                if self.fillend:
                    fills.append(True)
                    paths.append(
                        Path([*verticesB, (0, 0)], [*codesB, Path.CLOSEPOLY]))
                else:
                    fills.append(False)
                    paths.append(Path(verticesB, codesB))
            elif self._endarrow_bracket:
                x0, y0 = path.vertices[-1]
                x1, y1 = path.vertices[-2]
                verticesB, codesB = self._get_bracket(x0, y0, x1, y1,
                                                      self.widthB * scaleB,
                                                      self.lengthB * scaleB,
                                                      self.angleB)

                paths.append(Path(verticesB, codesB))
                fills.append(False)

            return paths, fills

    @_register_style(_style_list, name="-")
    class Curve(_Curve):
        """A simple curve without any arrow head."""

        def __init__(self):  # hide head_length, head_width
            # These attributes (whose values come from backcompat) only matter
            # if someone modifies beginarrow/etc. on an ArrowStyle instance.
            super().__init__(head_length=.2, head_width=.1)

    @_register_style(_style_list, name="<-")
    class CurveA(_Curve):
        """An arrow with a head at its start point."""
        arrow = "<-"

    @_register_style(_style_list, name="->")
    class CurveB(_Curve):
        """An arrow with a head at its end point."""
        arrow = "->"

    @_register_style(_style_list, name="<->")
    class CurveAB(_Curve):
        """An arrow with heads both at the start and the end point."""
        arrow = "<->"

    @_register_style(_style_list, name="<|-")
    class CurveFilledA(_Curve):
        """An arrow with filled triangle head at the start."""
        arrow = "<|-"

    @_register_style(_style_list, name="-|>")
    class CurveFilledB(_Curve):
        """An arrow with filled triangle head at the end."""
        arrow = "-|>"

    @_register_style(_style_list, name="<|-|>")
    class CurveFilledAB(_Curve):
        """An arrow with filled triangle heads at both ends."""
        arrow = "<|-|>"

    @_register_style(_style_list, name="]-")
    class BracketA(_Curve):
        """An arrow with an outward square bracket at its start."""
        arrow = "]-"

        def __init__(self, widthA=1., lengthA=0.2, angleA=0):
            """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.
            lengthA : float, default: 0.2
                Length of the bracket.
            angleA : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA)

    @_register_style(_style_list, name="-[")
    class BracketB(_Curve):
        """An arrow with an outward square bracket at its end."""
        arrow = "-["

        def __init__(self, widthB=1., lengthB=0.2, angleB=0):
            """
            Parameters
            ----------
            widthB : float, default: 1.0
                Width of the bracket.
            lengthB : float, default: 0.2
                Length of the bracket.
            angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list, name="]-[")
    class BracketAB(_Curve):
        """An arrow with outward square brackets at both ends."""
        arrow = "]-["

        def __init__(self,
                     widthA=1., lengthA=0.2, angleA=0,
                     widthB=1., lengthB=0.2, angleB=0):
            """
            Parameters
            ----------
            widthA, widthB : float, default: 1.0
                Width of the bracket.
            lengthA, lengthB : float, default: 0.2
                Length of the bracket.
            angleA, angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA,
                             widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list, name="|-|")
    class BarAB(_Curve):
        """An arrow with vertical bars ``|`` at both ends."""
        arrow = "|-|"

        def __init__(self, widthA=1., angleA=0, widthB=1., angleB=0):
            """
            Parameters
            ----------
            widthA, widthB : float, default: 1.0
                Width of the bracket.
            angleA, angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=0, angleA=angleA,
                             widthB=widthB, lengthB=0, angleB=angleB)

    @_register_style(_style_list, name=']->')
    class BracketCurve(_Curve):
        """
        An arrow with an outward square bracket at its start and a head at
        the end.
        """
        arrow = "]->"

        def __init__(self, widthA=1., lengthA=0.2, angleA=None):
            """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.
            lengthA : float, default: 0.2
                Length of the bracket.
            angleA : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA)

    @_register_style(_style_list, name='<-[')
    class CurveBracket(_Curve):
        """
        An arrow with an outward square bracket at its end and a head at
        the start.
        """
        arrow = "<-["

        def __init__(self, widthB=1., lengthB=0.2, angleB=None):
            """
            Parameters
            ----------
            widthB : float, default: 1.0
                Width of the bracket.
            lengthB : float, default: 0.2
                Length of the bracket.
            angleB : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
            super().__init__(widthB=widthB, lengthB=lengthB, angleB=angleB)

    @_register_style(_style_list)
    class Simple(_Base):
        """A simple arrow. Only works with a quadratic Bézier curve."""

        def __init__(self, head_length=.5, head_width=.5, tail_width=.2):
            """
            Parameters
            ----------
            head_length : float, default: 0.5
                Length of the arrow head.

            head_width : float, default: 0.5
                Width of the arrow head.

            tail_width : float, default: 0.2
                Width of the arrow tail.
            """
            self.head_length, self.head_width, self.tail_width = \
                head_length, head_width, tail_width
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):
            # docstring inherited
            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            # divide the path into a head and a tail
            head_length = self.head_length * mutation_size
            in_f = inside_circle(x2, y2, head_length)
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]

            try:
                arrow_out, arrow_in = \
                    split_bezier_intersecting_with_closedpath(arrow_path, in_f)
            except NonIntersectingPathException:
                # if this happens, make a straight line of the head_length
                # long.
                x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
                x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
                arrow_in = [(x0, y0), (x1n, y1n), (x2, y2)]
                arrow_out = None

            # head
            head_width = self.head_width * mutation_size
            head_left, head_right = make_wedged_bezier2(arrow_in,
                                                        head_width / 2., wm=.5)

            # tail
            if arrow_out is not None:
                tail_width = self.tail_width * mutation_size
                tail_left, tail_right = get_parallels(arrow_out,
                                                      tail_width / 2.)

                patch_path = [(Path.MOVETO, tail_right[0]),
                              (Path.CURVE3, tail_right[1]),
                              (Path.CURVE3, tail_right[2]),
                              (Path.LINETO, head_right[0]),
                              (Path.CURVE3, head_right[1]),
                              (Path.CURVE3, head_right[2]),
                              (Path.CURVE3, head_left[1]),
                              (Path.CURVE3, head_left[0]),
                              (Path.LINETO, tail_left[2]),
                              (Path.CURVE3, tail_left[1]),
                              (Path.CURVE3, tail_left[0]),
                              (Path.LINETO, tail_right[0]),
                              (Path.CLOSEPOLY, tail_right[0]),
                              ]
            else:
                patch_path = [(Path.MOVETO, head_right[0]),
                              (Path.CURVE3, head_right[1]),
                              (Path.CURVE3, head_right[2]),
                              (Path.CURVE3, head_left[1]),
                              (Path.CURVE3, head_left[0]),
                              (Path.CLOSEPOLY, head_left[0]),
                              ]

            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True

    @_register_style(_style_list)
    class Fancy(_Base):
        """A fancy arrow. Only works with a quadratic Bézier curve."""

        def __init__(self, head_length=.4, head_width=.4, tail_width=.4):
            """
            Parameters
            ----------
            head_length : float, default: 0.4
                Length of the arrow head.

            head_width : float, default: 0.4
                Width of the arrow head.

            tail_width : float, default: 0.4
                Width of the arrow tail.
            """
            self.head_length, self.head_width, self.tail_width = \
                head_length, head_width, tail_width
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):
            # docstring inherited
            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            # divide the path into a head and a tail
            head_length = self.head_length * mutation_size
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]

            # path for head
            in_f = inside_circle(x2, y2, head_length)
            try:
                path_out, path_in = split_bezier_intersecting_with_closedpath(
                    arrow_path, in_f)
            except NonIntersectingPathException:
                # if this happens, make a straight line of the head_length
                # long.
                x0, y0 = _point_along_a_line(x2, y2, x1, y1, head_length)
                x1n, y1n = 0.5 * (x0 + x2), 0.5 * (y0 + y2)
                arrow_path = [(x0, y0), (x1n, y1n), (x2, y2)]
                path_head = arrow_path
            else:
                path_head = path_in

            # path for head
            in_f = inside_circle(x2, y2, head_length * .8)
            path_out, path_in = split_bezier_intersecting_with_closedpath(
                arrow_path, in_f)
            path_tail = path_out

            # head
            head_width = self.head_width * mutation_size
            head_l, head_r = make_wedged_bezier2(path_head,
                                                 head_width / 2.,
                                                 wm=.6)

            # tail
            tail_width = self.tail_width * mutation_size
            tail_left, tail_right = make_wedged_bezier2(path_tail,
                                                        tail_width * .5,
                                                        w1=1., wm=0.6, w2=0.3)

            # path for head
            in_f = inside_circle(x0, y0, tail_width * .3)
            path_in, path_out = split_bezier_intersecting_with_closedpath(
                arrow_path, in_f)
            tail_start = path_in[-1]

            head_right, head_left = head_r, head_l
            patch_path = [(Path.MOVETO, tail_start),
                          (Path.LINETO, tail_right[0]),
                          (Path.CURVE3, tail_right[1]),
                          (Path.CURVE3, tail_right[2]),
                          (Path.LINETO, head_right[0]),
                          (Path.CURVE3, head_right[1]),
                          (Path.CURVE3, head_right[2]),
                          (Path.CURVE3, head_left[1]),
                          (Path.CURVE3, head_left[0]),
                          (Path.LINETO, tail_left[2]),
                          (Path.CURVE3, tail_left[1]),
                          (Path.CURVE3, tail_left[0]),
                          (Path.LINETO, tail_start),
                          (Path.CLOSEPOLY, tail_start),
                          ]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True

    @_register_style(_style_list)
    class Wedge(_Base):
        """
        Wedge(?) shape. Only works with a quadratic Bézier curve.  The
        start point has a width of the *tail_width* and the end point has a
        width of 0. At the middle, the width is *shrink_factor*x*tail_width*.
        """

        def __init__(self, tail_width=.3, shrink_factor=0.5):
            """
            Parameters
            ----------
            tail_width : float, default: 0.3
                Width of the tail.

            shrink_factor : float, default: 0.5
                Fraction of the arrow width at the middle point.
            """
            self.tail_width = tail_width
            self.shrink_factor = shrink_factor
            super().__init__()

        def transmute(self, path, mutation_size, linewidth):
            # docstring inherited
            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
            b_plus, b_minus = make_wedged_bezier2(
                                    arrow_path,
                                    self.tail_width * mutation_size / 2.,
                                    wm=self.shrink_factor)

            patch_path = [(Path.MOVETO, b_plus[0]),
                          (Path.CURVE3, b_plus[1]),
                          (Path.CURVE3, b_plus[2]),
                          (Path.LINETO, b_minus[2]),
                          (Path.CURVE3, b_minus[1]),
                          (Path.CURVE3, b_minus[0]),
                          (Path.CLOSEPOLY, b_minus[0]),
                          ]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True


class FancyBboxPatch(Patch):
    """
    A fancy box around a rectangle with lower left at *xy* = (*x*, *y*)
    with specified width and height.

    `.FancyBboxPatch` is similar to `.Rectangle`, but it draws a fancy box
    around the rectangle. The transformation of the rectangle box to the
    fancy box is delegated to the style classes defined in `.BoxStyle`.
    """

    _edge_default = True

    def __str__(self):
        s = self.__class__.__name__ + "((%g, %g), width=%g, height=%g)"
        return s % (self._x, self._y, self._width, self._height)

    @_docstring.interpd
    def __init__(self, xy, width, height, boxstyle="round", *,
                 mutation_scale=1, mutation_aspect=1, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
          The lower left corner of the box.

        width : float
            The width of the box.

        height : float
            The height of the box.

        boxstyle : str or `~matplotlib.patches.BoxStyle`
            The style of the fancy box. This can either be a `.BoxStyle`
            instance or a string of the style name and optionally comma
            separated attributes (e.g. "Round, pad=0.2"). This string is
            passed to `.BoxStyle` to construct a `.BoxStyle` object. See
            there for a full documentation.

            The following box styles are available:

            %(BoxStyle:table)s

        mutation_scale : float, default: 1
            Scaling factor applied to the attributes of the box style
            (e.g. pad or rounding_size).

        mutation_aspect : float, default: 1
            The height of the rectangle will be squeezed by this value before
            the mutation and the mutated box will be stretched by the inverse
            of it. For example, this allows different horizontal and vertical
            padding.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` properties

        %(Patch:kwdoc)s
        """

        super().__init__(**kwargs)
        self._x, self._y = xy
        self._width = width
        self._height = height
        self.set_boxstyle(boxstyle)
        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect
        self.stale = True

    @_docstring.interpd
    def set_boxstyle(self, boxstyle=None, **kwargs):
        """
        Set the box style, possibly with further attributes.

        Attributes from the previous box style are not reused.

        Without argument (or with ``boxstyle=None``), the available box styles
        are returned as a human-readable string.

        Parameters
        ----------
        boxstyle : str or `~matplotlib.patches.BoxStyle`
            The style of the box: either a `.BoxStyle` instance, or a string,
            which is the style name and optionally comma separated attributes
            (e.g. "Round,pad=0.2"). Such a string is used to construct a
            `.BoxStyle` object, as documented in that class.

            The following box styles are available:

            %(BoxStyle:table_and_accepts)s

        **kwargs
            Additional attributes for the box style. See the table above for
            supported parameters.

        Examples
        --------
        ::

            set_boxstyle("Round,pad=0.2")
            set_boxstyle("round", pad=0.2)
        """
        if boxstyle is None:
            return BoxStyle.pprint_styles()
        self._bbox_transmuter = (
            BoxStyle(boxstyle, **kwargs)
            if isinstance(boxstyle, str) else boxstyle)
        self.stale = True

    def get_boxstyle(self):
        """Return the boxstyle object."""
        return self._bbox_transmuter

    def set_mutation_scale(self, scale):
        """
        Set the mutation scale.

        Parameters
        ----------
        scale : float
        """
        self._mutation_scale = scale
        self.stale = True

    def get_mutation_scale(self):
        """Return the mutation scale."""
        return self._mutation_scale

    def set_mutation_aspect(self, aspect):
        """
        Set the aspect ratio of the bbox mutation.

        Parameters
        ----------
        aspect : float
        """
        self._mutation_aspect = aspect
        self.stale = True

    def get_mutation_aspect(self):
        """Return the aspect ratio of the bbox mutation."""
        return (self._mutation_aspect if self._mutation_aspect is not None
                else 1)  # backcompat.

    def get_path(self):
        """Return the mutated path of the rectangle."""
        boxstyle = self.get_boxstyle()
        m_aspect = self.get_mutation_aspect()
        # Call boxstyle with y, height squeezed by aspect_ratio.
        path = boxstyle(self._x, self._y / m_aspect,
                        self._width, self._height / m_aspect,
                        self.get_mutation_scale())
        return Path(path.vertices * [1, m_aspect], path.codes)  # Unsqueeze y.

    # Following methods are borrowed from the Rectangle class.

    def get_x(self):
        """Return the left coord of the rectangle."""
        return self._x

    def get_y(self):
        """Return the bottom coord of the rectangle."""
        return self._y

    def get_width(self):
        """Return the width of the rectangle."""
        return self._width

    def get_height(self):
        """Return the height of the rectangle."""
        return self._height

    def set_x(self, x):
        """
        Set the left coord of the rectangle.

        Parameters
        ----------
        x : float
        """
        self._x = x
        self.stale = True

    def set_y(self, y):
        """
        Set the bottom coord of the rectangle.

        Parameters
        ----------
        y : float
        """
        self._y = y
        self.stale = True

    def set_width(self, w):
        """
        Set the rectangle width.

        Parameters
        ----------
        w : float
        """
        self._width = w
        self.stale = True

    def set_height(self, h):
        """
        Set the rectangle height.

        Parameters
        ----------
        h : float
        """
        self._height = h
        self.stale = True

    def set_bounds(self, *args):
        """
        Set the bounds of the rectangle.

        Call signatures::

            set_bounds(left, bottom, width, height)
            set_bounds((left, bottom, width, height))

        Parameters
        ----------
        left, bottom : float
            The coordinates of the bottom left corner of the rectangle.
        width, height : float
            The width/height of the rectangle.
        """
        if len(args) == 1:
            l, b, w, h = args[0]
        else:
            l, b, w, h = args
        self._x = l
        self._y = b
        self._width = w
        self._height = h
        self.stale = True

    def get_bbox(self):
        """Return the `.Bbox`."""
        return transforms.Bbox.from_bounds(self._x, self._y,
                                           self._width, self._height)


class FancyArrowPatch(Patch):
    """
    A fancy arrow patch.

    It draws an arrow using the `ArrowStyle`. It is primarily used by the
    `~.axes.Axes.annotate` method.  For most purposes, use the annotate method for
    drawing arrows.

    The head and tail positions are fixed at the specified start and end points
    of the arrow, but the size and shape (in display coordinates) of the arrow
    does not change when the axis is moved or zoomed.
    """
    _edge_default = True

    def __str__(self):
        if self._posA_posB is not None:
            (x1, y1), (x2, y2) = self._posA_posB
            return f"{type(self).__name__}(({x1:g}, {y1:g})->({x2:g}, {y2:g}))"
        else:
            return f"{type(self).__name__}({self._path_original})"

    @_docstring.interpd
    def __init__(self, posA=None, posB=None, *,
                 path=None, arrowstyle="simple", connectionstyle="arc3",
                 patchA=None, patchB=None, shrinkA=2, shrinkB=2,
                 mutation_scale=1, mutation_aspect=1, **kwargs):
        """
        **Defining the arrow position and path**

        There are two ways to define the arrow position and path:

        - **Start, end and connection**:
          The typical approach is to define the start and end points of the
          arrow using *posA* and *posB*. The curve between these two can
          further be configured using *connectionstyle*.

          If given, the arrow curve is clipped by *patchA* and *patchB*,
          allowing it to start/end at the border of these patches.
          Additionally, the arrow curve can be shortened by *shrinkA* and *shrinkB*
          to create a margin between start/end (after possible clipping) and the
          drawn arrow.

        - **path**: Alternatively if *path* is provided, an arrow is drawn along
          this Path. In this case, *connectionstyle*, *patchA*, *patchB*,
          *shrinkA*, and *shrinkB* are ignored.

        **Styling**

        The *arrowstyle* defines the styling of the arrow head, tail and shaft.
        The resulting arrows can be styled further by setting the `.Patch`
        properties such as *linewidth*, *color*, *facecolor*, *edgecolor*
        etc. via keyword arguments.

        Parameters
        ----------
        posA, posB : (float, float), optional
            (x, y) coordinates of start and end point of the arrow.
            The actually drawn start and end positions may be modified
            through *patchA*, *patchB*, *shrinkA*, and *shrinkB*.

            *posA*, *posB* are exclusive of *path*.

        path : `~matplotlib.path.Path`, optional
            If provided, an arrow is drawn along this path and *patchA*,
            *patchB*, *shrinkA*, and *shrinkB* are ignored.

            *path* is exclusive of *posA*, *posB*.

        arrowstyle : str or `.ArrowStyle`, default: 'simple'
            The styling of arrow head, tail and shaft. This can be

            - `.ArrowStyle` or one of its subclasses
            - The shorthand string name (e.g. "->") as given in the table below,
              optionally containing a comma-separated list of style parameters,
              e.g. "->, head_length=10, head_width=5".

            The style parameters are scaled by *mutation_scale*.

            The following arrow styles are available. See also
            :doc:`/gallery/text_labels_and_annotations/fancyarrow_demo`.

            %(ArrowStyle:table)s

            Only the styles ``<|-``, ``-|>``, ``<|-|>`` ``simple``, ``fancy``
            and ``wedge`` contain closed paths and can be filled.

        connectionstyle : str or `.ConnectionStyle` or None, optional, \
default: 'arc3'
            `.ConnectionStyle` with which *posA* and *posB* are connected.
            This can be

            - `.ConnectionStyle` or one of its subclasses
            - The shorthand string name as given in the table below, e.g. "arc3".

            %(ConnectionStyle:table)s

            Ignored if *path* is provided.

        patchA, patchB : `~matplotlib.patches.Patch`, default: None
            Optional Patches at *posA* and *posB*, respectively. If given,
            the arrow path is clipped by these patches such that head and tail
            are at the border of the patches.

            Ignored if *path* is provided.

        shrinkA, shrinkB : float, default: 2
            Shorten the arrow path at *posA* and *posB* by this amount in points.
            This allows to add a margin between the intended start/end points and
            the arrow.

            Ignored if *path* is provided.

        mutation_scale : float, default: 1
            Value with which attributes of *arrowstyle* (e.g., *head_length*)
            will be scaled.

        mutation_aspect : None or float, default: None
            The height of the rectangle will be squeezed by this value before
            the mutation and the mutated box will be stretched by the inverse
            of it.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` properties, optional
            Here is a list of available `.Patch` properties:

        %(Patch:kwdoc)s

            In contrast to other patches, the default ``capstyle`` and
            ``joinstyle`` for `FancyArrowPatch` are set to ``"round"``.
        """
        # Traditionally, the cap- and joinstyle for FancyArrowPatch are round
        kwargs.setdefault("joinstyle", JoinStyle.round)
        kwargs.setdefault("capstyle", CapStyle.round)

        super().__init__(**kwargs)

        if posA is not None and posB is not None and path is None:
            self._posA_posB = [posA, posB]

            if connectionstyle is None:
                connectionstyle = "arc3"
            self.set_connectionstyle(connectionstyle)

        elif posA is None and posB is None and path is not None:
            self._posA_posB = None
        else:
            raise ValueError("Either posA and posB, or path need to provided")

        self.patchA = patchA
        self.patchB = patchB
        self.shrinkA = shrinkA
        self.shrinkB = shrinkB

        self._path_original = path

        self.set_arrowstyle(arrowstyle)

        self._mutation_scale = mutation_scale
        self._mutation_aspect = mutation_aspect

        self._dpi_cor = 1.0

    def set_positions(self, posA, posB):
        """
        Set the start and end positions of the connecting path.

        Parameters
        ----------
        posA, posB : None, tuple
            (x, y) coordinates of arrow tail and arrow head respectively. If
            `None` use current value.
        """
        if posA is not None:
            self._posA_posB[0] = posA
        if posB is not None:
            self._posA_posB[1] = posB
        self.stale = True

    def set_patchA(self, patchA):
        """
        Set the tail patch.

        Parameters
        ----------
        patchA : `.patches.Patch`
        """
        self.patchA = patchA
        self.stale = True

    def set_patchB(self, patchB):
        """
        Set the head patch.

        Parameters
        ----------
        patchB : `.patches.Patch`
        """
        self.patchB = patchB
        self.stale = True

    @_docstring.interpd
    def set_connectionstyle(self, connectionstyle=None, **kwargs):
        """
        Set the connection style, possibly with further attributes.

        Attributes from the previous connection style are not reused.

        Without argument (or with ``connectionstyle=None``), the available box
        styles are returned as a human-readable string.

        Parameters
        ----------
        connectionstyle : str or `~matplotlib.patches.ConnectionStyle`
            The style of the connection: either a `.ConnectionStyle` instance,
            or a string, which is the style name and optionally comma separated
            attributes (e.g. "Arc,armA=30,rad=10"). Such a string is used to
            construct a `.ConnectionStyle` object, as documented in that class.

            The following connection styles are available:

            %(ConnectionStyle:table_and_accepts)s

        **kwargs
            Additional attributes for the connection style. See the table above
            for supported parameters.

        Examples
        --------
        ::

            set_connectionstyle("Arc,armA=30,rad=10")
            set_connectionstyle("arc", armA=30, rad=10)
        """
        if connectionstyle is None:
            return ConnectionStyle.pprint_styles()
        self._connector = (
            ConnectionStyle(connectionstyle, **kwargs)
            if isinstance(connectionstyle, str) else connectionstyle)
        self.stale = True

    def get_connectionstyle(self):
        """Return the `ConnectionStyle` used."""
        return self._connector

    @_docstring.interpd
    def set_arrowstyle(self, arrowstyle=None, **kwargs):
        """
        Set the arrow style, possibly with further attributes.

        Attributes from the previous arrow style are not reused.

        Without argument (or with ``arrowstyle=None``), the available box
        styles are returned as a human-readable string.

        Parameters
        ----------
        arrowstyle : str or `~matplotlib.patches.ArrowStyle`
            The style of the arrow: either a `.ArrowStyle` instance, or a
            string, which is the style name and optionally comma separated
            attributes (e.g. "Fancy,head_length=0.2"). Such a string is used to
            construct a `.ArrowStyle` object, as documented in that class.

            The following arrow styles are available:

            %(ArrowStyle:table_and_accepts)s

        **kwargs
            Additional attributes for the arrow style. See the table above for
            supported parameters.

        Examples
        --------
        ::

            set_arrowstyle("Fancy,head_length=0.2")
            set_arrowstyle("fancy", head_length=0.2)
        """
        if arrowstyle is None:
            return ArrowStyle.pprint_styles()
        self._arrow_transmuter = (
            ArrowStyle(arrowstyle, **kwargs)
            if isinstance(arrowstyle, str) else arrowstyle)
        self.stale = True

    def get_arrowstyle(self):
        """Return the arrowstyle object."""
        return self._arrow_transmuter

    def set_mutation_scale(self, scale):
        """
        Set the mutation scale.

        Parameters
        ----------
        scale : float
        """
        self._mutation_scale = scale
        self.stale = True

    def get_mutation_scale(self):
        """
        Return the mutation scale.

        Returns
        -------
        scalar
        """
        return self._mutation_scale

    def set_mutation_aspect(self, aspect):
        """
        Set the aspect ratio of the bbox mutation.

        Parameters
        ----------
        aspect : float
        """
        self._mutation_aspect = aspect
        self.stale = True

    def get_mutation_aspect(self):
        """Return the aspect ratio of the bbox mutation."""
        return (self._mutation_aspect if self._mutation_aspect is not None
                else 1)  # backcompat.

    def get_path(self):
        """Return the path of the arrow in the data coordinates."""
        # The path is generated in display coordinates, then converted back to
        # data coordinates.
        _path, fillable = self._get_path_in_displaycoord()
        if np.iterable(fillable):
            _path = Path.make_compound_path(*_path)
        return self.get_transform().inverted().transform_path(_path)

    def _get_path_in_displaycoord(self):
        """Return the mutated path of the arrow in display coordinates."""
        dpi_cor = self._dpi_cor

        if self._posA_posB is not None:
            posA = self._convert_xy_units(self._posA_posB[0])
            posB = self._convert_xy_units(self._posA_posB[1])
            (posA, posB) = self.get_transform().transform((posA, posB))
            _path = self.get_connectionstyle()(posA, posB,
                                               patchA=self.patchA,
                                               patchB=self.patchB,
                                               shrinkA=self.shrinkA * dpi_cor,
                                               shrinkB=self.shrinkB * dpi_cor
                                               )
        else:
            _path = self.get_transform().transform_path(self._path_original)

        _path, fillable = self.get_arrowstyle()(
            _path,
            self.get_mutation_scale() * dpi_cor,
            self.get_linewidth() * dpi_cor,
            self.get_mutation_aspect())

        return _path, fillable

    def draw(self, renderer):
        if not self.get_visible():
            return

        # FIXME: dpi_cor is for the dpi-dependency of the linewidth.  There
        # could be room for improvement.  Maybe _get_path_in_displaycoord could
        # take a renderer argument, but get_path should be adapted too.
        self._dpi_cor = renderer.points_to_pixels(1.)
        path, fillable = self._get_path_in_displaycoord()

        if not np.iterable(fillable):
            path = [path]
            fillable = [fillable]

        affine = transforms.IdentityTransform()

        self._draw_paths_with_artist_properties(
            renderer,
            [(p, affine, self._facecolor if f and self._facecolor[3] else None)
             for p, f in zip(path, fillable)])


class ConnectionPatch(FancyArrowPatch):
    """A patch that connects two points (possibly in different Axes)."""

    def __str__(self):
        return "ConnectionPatch((%g, %g), (%g, %g))" % \
               (self.xy1[0], self.xy1[1], self.xy2[0], self.xy2[1])

    @_docstring.interpd
    def __init__(self, xyA, xyB, coordsA, coordsB=None, *,
                 axesA=None, axesB=None,
                 arrowstyle="-",
                 connectionstyle="arc3",
                 patchA=None,
                 patchB=None,
                 shrinkA=0.,
                 shrinkB=0.,
                 mutation_scale=10.,
                 mutation_aspect=None,
                 clip_on=False,
                 **kwargs):
        """
        Connect point *xyA* in *coordsA* with point *xyB* in *coordsB*.

        Valid keys are

        ===============  ======================================================
        Key              Description
        ===============  ======================================================
        arrowstyle       the arrow style
        connectionstyle  the connection style
        relpos           default is (0.5, 0.5)
        patchA           default is bounding box of the text
        patchB           default is None
        shrinkA          default is 2 points
        shrinkB          default is 2 points
        mutation_scale   default is text size (in points)
        mutation_aspect  default is 1.
        ?                any key for `matplotlib.patches.PathPatch`
        ===============  ======================================================

        *coordsA* and *coordsB* are strings that indicate the
        coordinates of *xyA* and *xyB*.

        ==================== ==================================================
        Property             Description
        ==================== ==================================================
        'figure points'      points from the lower left corner of the figure
        'figure pixels'      pixels from the lower left corner of the figure
        'figure fraction'    0, 0 is lower left of figure and 1, 1 is upper
                             right
        'subfigure points'   points from the lower left corner of the subfigure
        'subfigure pixels'   pixels from the lower left corner of the subfigure
        'subfigure fraction' fraction of the subfigure, 0, 0 is lower left.
        'axes points'        points from lower left corner of the Axes
        'axes pixels'        pixels from lower left corner of the Axes
        'axes fraction'      0, 0 is lower left of Axes and 1, 1 is upper right
        'data'               use the coordinate system of the object being
                             annotated (default)
        'offset points'      offset (in points) from the *xy* value
        'polar'              you can specify *theta*, *r* for the annotation,
                             even in cartesian plots.  Note that if you are
                             using a polar Axes, you do not need to specify
                             polar for the coordinate system since that is the
                             native "data" coordinate system.
        ==================== ==================================================

        Alternatively they can be set to any valid
        `~matplotlib.transforms.Transform`.

        Note that 'subfigure pixels' and 'figure pixels' are the same
        for the parent figure, so users who want code that is usable in
        a subfigure can use 'subfigure pixels'.

        .. note::

           Using `ConnectionPatch` across two `~.axes.Axes` instances
           is not directly compatible with :ref:`constrained layout
           <constrainedlayout_guide>`. Add the artist
           directly to the `.Figure` instead of adding it to a specific Axes,
           or exclude it from the layout using ``con.set_in_layout(False)``.

           .. code-block:: default

              fig, ax = plt.subplots(1, 2, constrained_layout=True)
              con = ConnectionPatch(..., axesA=ax[0], axesB=ax[1])
              fig.add_artist(con)

        """
        if coordsB is None:
            coordsB = coordsA
        # we'll draw ourself after the artist we annotate by default
        self.xy1 = xyA
        self.xy2 = xyB
        self.coords1 = coordsA
        self.coords2 = coordsB

        self.axesA = axesA
        self.axesB = axesB

        super().__init__(posA=(0, 0), posB=(1, 1),
                         arrowstyle=arrowstyle,
                         connectionstyle=connectionstyle,
                         patchA=patchA, patchB=patchB,
                         shrinkA=shrinkA, shrinkB=shrinkB,
                         mutation_scale=mutation_scale,
                         mutation_aspect=mutation_aspect,
                         clip_on=clip_on,
                         **kwargs)
        # if True, draw annotation only if self.xy is inside the Axes
        self._annotation_clip = None

    def _get_xy(self, xy, s, axes=None):
        """Calculate the pixel position of given point."""
        s0 = s  # For the error message, if needed.
        if axes is None:
            axes = self.axes

        # preserve mixed type input (such as str, int)
        x = np.array(xy[0])
        y = np.array(xy[1])

        fig = self.get_figure(root=False)
        if s in ["figure points", "axes points"]:
            x = x * fig.dpi / 72
            y = y * fig.dpi / 72
            s = s.replace("points", "pixels")
        elif s == "figure fraction":
            s = fig.transFigure
        elif s == "subfigure fraction":
            s = fig.transSubfigure
        elif s == "axes fraction":
            s = axes.transAxes

        if s == 'data':
            trans = axes.transData
            x = cbook._to_unmasked_float_array(axes.xaxis.convert_units(x))
            y = cbook._to_unmasked_float_array(axes.yaxis.convert_units(y))
            return trans.transform((x, y))
        elif s == 'offset points':
            if self.xycoords == 'offset points':  # prevent recursion
                return self._get_xy(self.xy, 'data')
            return (
                self._get_xy(self.xy, self.xycoords)  # converted data point
                + xy * self.get_figure(root=True).dpi / 72)  # converted offset
        elif s == 'polar':
            theta, r = x, y
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            trans = axes.transData
            return trans.transform((x, y))
        elif s == 'figure pixels':
            # pixels from the lower left corner of the figure
            bb = self.get_figure(root=False).figbbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif s == 'subfigure pixels':
            # pixels from the lower left corner of the figure
            bb = self.get_figure(root=False).bbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif s == 'axes pixels':
            # pixels from the lower left corner of the Axes
            bb = axes.bbox
            x = bb.x0 + x if x >= 0 else bb.x1 + x
            y = bb.y0 + y if y >= 0 else bb.y1 + y
            return x, y
        elif isinstance(s, transforms.Transform):
            return s.transform(xy)
        else:
            raise ValueError(f"{s0} is not a valid coordinate transformation")

    def set_annotation_clip(self, b):
        """
        Set the annotation's clipping behavior.

        Parameters
        ----------
        b : bool or None
            - True: The annotation will be clipped when ``self.xy`` is
              outside the Axes.
            - False: The annotation will always be drawn.
            - None: The annotation will be clipped when ``self.xy`` is
              outside the Axes and ``self.xycoords == "data"``.
        """
        self._annotation_clip = b
        self.stale = True

    def get_annotation_clip(self):
        """
        Return the clipping behavior.

        See `.set_annotation_clip` for the meaning of the return value.
        """
        return self._annotation_clip

    def _get_path_in_displaycoord(self):
        """Return the mutated path of the arrow in display coordinates."""
        dpi_cor = self._dpi_cor
        posA = self._get_xy(self.xy1, self.coords1, self.axesA)
        posB = self._get_xy(self.xy2, self.coords2, self.axesB)
        path = self.get_connectionstyle()(
            posA, posB,
            patchA=self.patchA, patchB=self.patchB,
            shrinkA=self.shrinkA * dpi_cor, shrinkB=self.shrinkB * dpi_cor,
        )
        path, fillable = self.get_arrowstyle()(
            path,
            self.get_mutation_scale() * dpi_cor,
            self.get_linewidth() * dpi_cor,
            self.get_mutation_aspect()
        )
        return path, fillable

    def _check_xy(self, renderer):
        """Check whether the annotation needs to be drawn."""

        b = self.get_annotation_clip()

        if b or (b is None and self.coords1 == "data"):
            xy_pixel = self._get_xy(self.xy1, self.coords1, self.axesA)
            if self.axesA is None:
                axes = self.axes
            else:
                axes = self.axesA
            if not axes.contains_point(xy_pixel):
                return False

        if b or (b is None and self.coords2 == "data"):
            xy_pixel = self._get_xy(self.xy2, self.coords2, self.axesB)
            if self.axesB is None:
                axes = self.axes
            else:
                axes = self.axesB
            if not axes.contains_point(xy_pixel):
                return False

        return True

    def draw(self, renderer):
        if not self.get_visible() or not self._check_xy(renderer):
            return
        super().draw(renderer)
<|endoftext|>from collections import namedtuple
import contextlib
from functools import cache, reduce, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import operator
import re
import warnings

import numpy as np

import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
                         TransformedPatchPath, TransformedPath)

_log = logging.getLogger(__name__)


def _prevent_rasterization(draw):
    # We assume that by default artists are not allowed to rasterize (unless
    # its draw method is explicitly decorated). If it is being drawn after a
    # rasterized artist and it has reached a raster_depth of 0, we stop
    # rasterization so that it does not affect the behavior of normal artist
    # (e.g., change in dpi).

    @wraps(draw)
    def draw_wrapper(artist, renderer, *args, **kwargs):
        if renderer._raster_depth == 0 and renderer._rasterizing:
            # Only stop when we are not in a rasterized parent
            # and something has been rasterized since last stop.
            renderer.stop_rasterizing()
            renderer._rasterizing = False

        return draw(artist, renderer, *args, **kwargs)

    draw_wrapper._supports_rasterization = False
    return draw_wrapper


def allow_rasterization(draw):
    """
    Decorator for Artist.draw method. Provides routines
    that run before and after the draw call. The before and after functions
    are useful for changing artist-dependent renderer attributes or making
    other setup function calls, such as starting and flushing a mixed-mode
    renderer.
    """

    @wraps(draw)
    def draw_wrapper(artist, renderer):
        try:
            if artist.get_rasterized():
                if renderer._raster_depth == 0 and not renderer._rasterizing:
                    renderer.start_rasterizing()
                    renderer._rasterizing = True
                renderer._raster_depth += 1
            else:
                if renderer._raster_depth == 0 and renderer._rasterizing:
                    # Only stop when we are not in a rasterized parent
                    # and something has be rasterized since last stop
                    renderer.stop_rasterizing()
                    renderer._rasterizing = False

            if artist.get_agg_filter() is not None:
                renderer.start_filter()

            return draw(artist, renderer)
        finally:
            if artist.get_agg_filter() is not None:
                renderer.stop_filter(artist.get_agg_filter())
            if artist.get_rasterized():
                renderer._raster_depth -= 1
            if (renderer._rasterizing and (fig := artist.get_figure(root=True)) and
                    fig.suppressComposite):
                # restart rasterizing to prevent merging
                renderer.stop_rasterizing()
                renderer.start_rasterizing()

    draw_wrapper._supports_rasterization = True
    return draw_wrapper


def _finalize_rasterization(draw):
    """
    Decorator for Artist.draw method. Needed on the outermost artist, i.e.
    Figure, to finish up if the render is still in rasterized mode.
    """
    @wraps(draw)
    def draw_wrapper(artist, renderer, *args, **kwargs):
        result = draw(artist, renderer, *args, **kwargs)
        if renderer._rasterizing:
            renderer.stop_rasterizing()
            renderer._rasterizing = False
        return result
    return draw_wrapper


def _stale_axes_callback(self, val):
    if self.axes:
        self.axes.stale = val


_XYPair = namedtuple("_XYPair", "x y")


class _Unset:
    def __repr__(self):
        return "<UNSET>"
_UNSET = _Unset()


class Artist:
    """
    Abstract base class for objects that render into a FigureCanvas.

    Typically, all visible elements in a figure are subclasses of Artist.
    """

    zorder = 0

    def __init_subclass__(cls):

        # Decorate draw() method so that all artists are able to stop
        # rastrization when necessary. If the artist's draw method is already
        # decorated (has a `_supports_rasterization` attribute), it won't be
        # decorated.

        if not hasattr(cls.draw, "_supports_rasterization"):
            cls.draw = _prevent_rasterization(cls.draw)

        # Inject custom set() methods into the subclass with signature and
        # docstring based on the subclasses' properties.

        if not hasattr(cls.set, '_autogenerated_signature'):
            # Don't overwrite cls.set if the subclass or one of its parents
            # has defined a set method set itself.
            # If there was no explicit definition, cls.set is inherited from
            # the hierarchy of auto-generated set methods, which hold the
            # flag _autogenerated_signature.
            return

        cls.set = lambda self, **kwargs: Artist.set(self, **kwargs)
        cls.set.__name__ = "set"
        cls.set.__qualname__ = f"{cls.__qualname__}.set"
        cls._update_set_signature_and_docstring()

    _PROPERTIES_EXCLUDED_FROM_SET = [
        'navigate_mode',  # not a user-facing function
        'figure',         # changing the figure is such a profound operation
                          # that we don't want this in set()
        '3d_properties',  # cannot be used as a keyword due to leading digit
    ]

    @classmethod
    def _update_set_signature_and_docstring(cls):
        """
        Update the signature of the set function to list all properties
        as keyword arguments.

        Property aliases are not listed in the signature for brevity, but
        are still accepted as keyword arguments.
        """
        cls.set.__signature__ = Signature(
            [Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
             *[Parameter(prop, Parameter.KEYWORD_ONLY, default=_UNSET)
               for prop in ArtistInspector(cls).get_setters()
               if prop not in Artist._PROPERTIES_EXCLUDED_FROM_SET]])
        cls.set._autogenerated_signature = True

        cls.set.__doc__ = (
            "Set multiple properties at once.\n\n"
            "Supported properties are\n\n"
            + kwdoc(cls))

    def __init__(self):
        self._stale = True
        self.stale_callback = None
        self._axes = None
        self._parent_figure = None

        self._transform = None
        self._transformSet = False
        self._visible = True
        self._animated = False
        self._alpha = None
        self.clipbox = None
        self._clippath = None
        self._clipon = True
        self._label = ''
        self._picker = None
        self._rasterized = False
        self._agg_filter = None
        # Normally, artist classes need to be queried for mouseover info if and
        # only if they override get_cursor_data.
        self._mouseover = type(self).get_cursor_data != Artist.get_cursor_data
        self._callbacks = cbook.CallbackRegistry(signals=["pchanged"])
        try:
            self.axes = None
        except AttributeError:
            # Handle self.axes as a read-only property, as in Figure.
            pass
        self._remove_method = None
        self._url = None
        self._gid = None
        self._snap = None
        self._sketch = mpl.rcParams['path.sketch']
        self._path_effects = mpl.rcParams['path.effects']
        self._sticky_edges = _XYPair([], [])
        self._in_layout = True

    def __getstate__(self):
        d = self.__dict__.copy()
        d['stale_callback'] = None
        return d

    def remove(self):
        """
        Remove the artist from the figure if possible.

        The effect will not be visible until the figure is redrawn, e.g.,
        with `.FigureCanvasBase.draw_idle`.  Call `~.axes.Axes.relim` to
        update the Axes limits if desired.

        Note: `~.axes.Axes.relim` will not see collections even if the
        collection was added to the Axes with *autolim* = True.

        Note: there is no support for removing the artist's legend entry.
        """

        # There is no method to set the callback.  Instead, the parent should
        # set the _remove_method attribute directly.  This would be a
        # protected attribute if Python supported that sort of thing.  The
        # callback has one parameter, which is the child to be removed.
        if self._remove_method is not None:
            self._remove_method(self)
            # clear stale callback
            self.stale_callback = None
            _ax_flag = False
            ax = getattr(self, 'axes', None)
            mouseover_set = getattr(ax, '_mouseover_set', None)
            if mouseover_set is not None:
                # remove from the mouse hit list
                mouseover_set.discard(self)
                ax.stale = True
                self.axes = None  # decouple the artist from the Axes
                _ax_flag = True

            if (fig := self.get_figure(root=False)) is not None:
                if not _ax_flag:
                    fig.stale = True
                self._parent_figure = None

        else:
            raise NotImplementedError('cannot remove artist')
        # TODO: the fix for the collections relim problem is to move the
        # limits calculation into the artist itself, including the property of
        # whether or not the artist should affect the limits.  Then there will
        # be no distinction between axes.add_line, axes.add_patch, etc.
        # TODO: add legend support

    def have_units(self):
        """Return whether units are set on any axis."""
        ax = self.axes
        return ax and any(axis.have_units() for axis in ax._axis_map.values())

    def convert_xunits(self, x):
        """
        Convert *x* using the unit type of the xaxis.

        If the artist is not contained in an Axes or if the xaxis does not
        have units, *x* itself is returned.
        """
        ax = getattr(self, 'axes', None)
        if ax is None or ax.xaxis is None:
            return x
        return ax.xaxis.convert_units(x)

    def convert_yunits(self, y):
        """
        Convert *y* using the unit type of the yaxis.

        If the artist is not contained in an Axes or if the yaxis does not
        have units, *y* itself is returned.
        """
        ax = getattr(self, 'axes', None)
        if ax is None or ax.yaxis is None:
            return y
        return ax.yaxis.convert_units(y)

    @property
    def axes(self):
        """The `~.axes.Axes` instance the artist resides in, or *None*."""
        return self._axes

    @axes.setter
    def axes(self, new_axes):
        if (new_axes is not None and self._axes is not None
                and new_axes != self._axes):
            raise ValueError("Can not reset the Axes. You are probably trying to reuse "
                             "an artist in more than one Axes which is not supported")
        self._axes = new_axes
        if new_axes is not None and new_axes is not self:
            self.stale_callback = _stale_axes_callback

    @property
    def stale(self):
        """
        Whether the artist is 'stale' and needs to be re-drawn for the output
        to match the internal state of the artist.
        """
        return self._stale

    @stale.setter
    def stale(self, val):
        self._stale = val

        # if the artist is animated it does not take normal part in the
        # draw stack and is not expected to be drawn as part of the normal
        # draw loop (when not saving) so do not propagate this change
        if self._animated:
            return

        if val and self.stale_callback is not None:
            self.stale_callback(self, val)

    def get_window_extent(self, renderer=None):
        """
        Get the artist's bounding box in display space.

        The bounding box' width and height are nonnegative.

        Subclasses should override for inclusion in the bounding box
        "tight" calculation. Default is to return an empty bounding
        box at 0, 0.

        Be careful when using this function, the results will not update
        if the artist window extent of the artist changes.  The extent
        can change due to any changes in the transform stack, such as
        changing the Axes limits, the figure size, or the canvas used
        (as is done when saving a figure).  This can lead to unexpected
        behavior where interactive figures will look fine on the screen,
        but will save incorrectly.
        """
        return Bbox([[0, 0], [0, 0]])

    def get_tightbbox(self, renderer=None):
        """
        Like `.Artist.get_window_extent`, but includes any clipping.

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase` subclass, optional
            renderer that will be used to draw the figures (i.e.
            ``fig.canvas.get_renderer()``)

        Returns
        -------
        `.Bbox` or None
            The enclosing bounding box (in figure pixel coordinates).
            Returns None if clipping results in no intersection.
        """
        bbox = self.get_window_extent(renderer)
        if self.get_clip_on():
            clip_box = self.get_clip_box()
            if clip_box is not None:
                bbox = Bbox.intersection(bbox, clip_box)
            clip_path = self.get_clip_path()
            if clip_path is not None and bbox is not None:
                clip_path = clip_path.get_fully_transformed_path()
                bbox = Bbox.intersection(bbox, clip_path.get_extents())
        return bbox

    def add_callback(self, func):
        """
        Add a callback function that will be called whenever one of the
        `.Artist`'s properties changes.

        Parameters
        ----------
        func : callable
            The callback function. It must have the signature::

                def func(artist: Artist) -> Any

            where *artist* is the calling `.Artist`. Return values may exist
            but are ignored.

        Returns
        -------
        int
            The observer id associated with the callback. This id can be
            used for removing the callback with `.remove_callback` later.

        See Also
        --------
        remove_callback
        """
        # Wrapping func in a lambda ensures it can be connected multiple times
        # and never gets weakref-gc'ed.
        return self._callbacks.connect("pchanged", lambda: func(self))

    def remove_callback(self, oid):
        """
        Remove a callback based on its observer id.

        See Also
        --------
        add_callback
        """
        self._callbacks.disconnect(oid)

    def pchanged(self):
        """
        Call all of the registered callbacks.

        This function is triggered internally when a property is changed.

        See Also
        --------
        add_callback
        remove_callback
        """
        self._callbacks.process("pchanged")

    def is_transform_set(self):
        """
        Return whether the Artist has an explicitly set transform.

        This is *True* after `.set_transform` has been called.
        """
        return self._transformSet

    def set_transform(self, t):
        """
        Set the artist transform.

        Parameters
        ----------
        t : `~matplotlib.transforms.Transform`
        """
        self._transform = t
        self._transformSet = True
        self.pchanged()
        self.stale = True

    def get_transform(self):
        """Return the `.Transform` instance used by this artist."""
        if self._transform is None:
            self._transform = IdentityTransform()
        elif (not isinstance(self._transform, Transform)
              and hasattr(self._transform, '_as_mpl_transform')):
            self._transform = self._transform._as_mpl_transform(self.axes)
        return self._transform

    def get_children(self):
        r"""Return a list of the child `.Artist`\s of this `.Artist`."""
        return []

    def _different_canvas(self, event):
        """
        Check whether an *event* occurred on a canvas other that this artist's canvas.

        If this method returns True, the event definitely occurred on a different
        canvas; if it returns False, either it occurred on the same canvas, or we may
        not have enough information to know.

        Subclasses should start their definition of `contains` as follows::

            if self._different_canvas(mouseevent):
                return False, {}
            # subclass-specific implementation follows
        """
        return (getattr(event, "canvas", None) is not None
                and (fig := self.get_figure(root=True)) is not None
                and event.canvas is not fig.canvas)

    def contains(self, mouseevent):
        """
        Test whether the artist contains the mouse event.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`

        Returns
        -------
        contains : bool
            Whether any values are within the radius.
        details : dict
            An artist-specific dictionary of details of the event context,
            such as which points are contained in the pick radius. See the
            individual Artist subclasses for details.
        """
        _log.warning("%r needs 'contains' method", self.__class__.__name__)
        return False, {}

    def pickable(self):
        """
        Return whether the artist is pickable.

        See Also
        --------
        .Artist.set_picker, .Artist.get_picker, .Artist.pick
        """
        return self.get_figure(root=False) is not None and self._picker is not None

    def pick(self, mouseevent):
        """
        Process a pick event.

        Each child artist will fire a pick event if *mouseevent* is over
        the artist and the artist has picker set.

        See Also
        --------
        .Artist.set_picker, .Artist.get_picker, .Artist.pickable
        """
        from .backend_bases import PickEvent  # Circular import.
        # Pick self
        if self.pickable():
            picker = self.get_picker()
            if callable(picker):
                inside, prop = picker(self, mouseevent)
            else:
                inside, prop = self.contains(mouseevent)
            if inside:
                PickEvent("pick_event", self.get_figure(root=True).canvas,
                          mouseevent, self, **prop)._process()

        # Pick children
        for a in self.get_children():
            # make sure the event happened in the same Axes
            ax = getattr(a, 'axes', None)
            if (isinstance(a, mpl.figure.SubFigure)
                    or mouseevent.inaxes is None or ax is None
                    or mouseevent.inaxes == ax):
                # we need to check if mouseevent.inaxes is None
                # because some objects associated with an Axes (e.g., a
                # tick label) can be outside the bounding box of the
                # Axes and inaxes will be None
                # also check that ax is None so that it traverse objects
                # which do not have an axes property but children might
                a.pick(mouseevent)

    def set_picker(self, picker):
        """
        Define the picking behavior of the artist.

        Parameters
        ----------
        picker : None or bool or float or callable
            This can be one of the following:

            - *None*: Picking is disabled for this artist (default).

            - A boolean: If *True* then picking will be enabled and the
              artist will fire a pick event if the mouse event is over
              the artist.

            - A float: If picker is a number it is interpreted as an
              epsilon tolerance in points and the artist will fire
              off an event if its data is within epsilon of the mouse
              event.  For some artists like lines and patch collections,
              the artist may provide additional data to the pick event
              that is generated, e.g., the indices of the data within
              epsilon of the pick event

            - A function: If picker is callable, it is a user supplied
              function which determines whether the artist is hit by the
              mouse event::

                hit, props = picker(artist, mouseevent)

              to determine the hit test.  if the mouse event is over the
              artist, return *hit=True* and props is a dictionary of
              properties you want added to the PickEvent attributes.
        """
        self._picker = picker

    def get_picker(self):
        """
        Return the picking behavior of the artist.

        The possible values are described in `.Artist.set_picker`.

        See Also
        --------
        .Artist.set_picker, .Artist.pickable, .Artist.pick
        """
        return self._picker

    def get_url(self):
        """Return the url."""
        return self._url

    def set_url(self, url):
        """
        Set the url for the artist.

        Parameters
        ----------
        url : str
        """
        self._url = url

    def get_gid(self):
        """Return the group id."""
        return self._gid

    def set_gid(self, gid):
        """
        Set the (group) id for the artist.

        Parameters
        ----------
        gid : str
        """
        self._gid = gid

    def get_snap(self):
        """
        Return the snap setting.

        See `.set_snap` for details.
        """
        if mpl.rcParams['path.snap']:
            return self._snap
        else:
            return False

    def set_snap(self, snap):
        """
        Set the snapping behavior.

        Snapping aligns positions with the pixel grid, which results in
        clearer images. For example, if a black line of 1px width was
        defined at a position in between two pixels, the resulting image
        would contain the interpolated value of that line in the pixel grid,
        which would be a grey value on both adjacent pixel positions. In
        contrast, snapping will move the line to the nearest integer pixel
        value, so that the resulting image will really contain a 1px wide
        black line.

        Snapping is currently only supported by the Agg and MacOSX backends.

        Parameters
        ----------
        snap : bool or None
            Possible values:

            - *True*: Snap vertices to the nearest pixel center.
            - *False*: Do not modify vertex positions.
            - *None*: (auto) If the path contains only rectilinear line
              segments, round to the nearest pixel center.
        """
        self._snap = snap
        self.stale = True

    def get_sketch_params(self):
        """
        Return the sketch parameters for the artist.

        Returns
        -------
        tuple or None

            A 3-tuple with the following elements:

            - *scale*: The amplitude of the wiggle perpendicular to the
              source line.
            - *length*: The length of the wiggle along the line.
            - *randomness*: The scale factor by which the length is
              shrunken or expanded.

            Returns *None* if no sketch parameters were set.
        """
        return self._sketch

    def set_sketch_params(self, scale=None, length=None, randomness=None):
        """
        Set the sketch parameters.

        Parameters
        ----------
        scale : float, optional
            The amplitude of the wiggle perpendicular to the source
            line, in pixels.  If scale is `None`, or not provided, no
            sketch filter will be provided.
        length : float, optional
             The length of the wiggle along the line, in pixels
             (default 128.0)
        randomness : float, optional
            The scale factor by which the length is shrunken or
            expanded (default 16.0)

            The PGF backend uses this argument as an RNG seed and not as
            described above. Using the same seed yields the same random shape.

            .. ACCEPTS: (scale: float, length: float, randomness: float)
        """
        if scale is None:
            self._sketch = None
        else:
            self._sketch = (scale, length or 128.0, randomness or 16.0)
        self.stale = True

    def set_path_effects(self, path_effects):
        """
        Set the path effects.

        Parameters
        ----------
        path_effects : list of `.AbstractPathEffect`
        """
        self._path_effects = path_effects
        self.stale = True

    def get_path_effects(self):
        return self._path_effects

    def get_figure(self, root=False):
        """
        Return the `.Figure` or `.SubFigure` instance the artist belongs to.

        Parameters
        ----------
        root : bool, default=False
            If False, return the (Sub)Figure this artist is on.  If True,
            return the root Figure for a nested tree of SubFigures.
        """
        if root and self._parent_figure is not None:
            return self._parent_figure.get_figure(root=True)

        return self._parent_figure

    def set_figure(self, fig):
        """
        Set the `.Figure` or `.SubFigure` instance the artist belongs to.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure` or `~matplotlib.figure.SubFigure`
        """
        # if this is a no-op just return
        if self._parent_figure is fig:
            return
        # if we currently have a figure (the case of both `self.figure`
        # and *fig* being none is taken care of above) we then user is
        # trying to change the figure an artist is associated with which
        # is not allowed for the same reason as adding the same instance
        # to more than one Axes
        if self._parent_figure is not None:
            raise RuntimeError("Can not put single artist in "
                               "more than one figure")
        self._parent_figure = fig
        if self._parent_figure and self._parent_figure is not self:
            self.pchanged()
        self.stale = True

    figure = property(get_figure, set_figure,
                      doc=("The (Sub)Figure that the artist is on.  For more "
                           "control, use the `get_figure` method."))

    def set_clip_box(self, clipbox):
        """
        Set the artist's clip `.Bbox`.

        Parameters
        ----------
        clipbox : `~matplotlib.transforms.BboxBase` or None
            Will typically be created from a `.TransformedBbox`. For instance,
            ``TransformedBbox(Bbox([[0, 0], [1, 1]]), ax.transAxes)`` is the default
            clipping for an artist added to an Axes.

        """
        _api.check_isinstance((BboxBase, None), clipbox=clipbox)
        if clipbox != self.clipbox:
            self.clipbox = clipbox
            self.pchanged()
            self.stale = True

    def set_clip_path(self, path, transform=None):
        """
        Set the artist's clip path.

        Parameters
        ----------
        path : `~matplotlib.patches.Patch` or `.Path` or `.TransformedPath` or None
            The clip path. If given a `.Path`, *transform* must be provided as
            well. If *None*, a previously set clip path is removed.
        transform : `~matplotlib.transforms.Transform`, optional
            Only used if *path* is a `.Path`, in which case the given `.Path`
            is converted to a `.TransformedPath` using *transform*.

        Notes
        -----
        For efficiency, if *path* is a `.Rectangle` this method will set the
        clipping box to the corresponding rectangle and set the clipping path
        to ``None``.

        For technical reasons (support of `~.Artist.set`), a tuple
        (*path*, *transform*) is also accepted as a single positional
        parameter.

        .. ACCEPTS: Patch or (Path, Transform) or None
        """
        from matplotlib.patches import Patch, Rectangle

        success = False
        if transform is None:
            if isinstance(path, Rectangle):
                self.clipbox = TransformedBbox(Bbox.unit(),
                                               path.get_transform())
                self._clippath = None
                success = True
            elif isinstance(path, Patch):
                self._clippath = TransformedPatchPath(path)
                success = True
            elif isinstance(path, tuple):
                path, transform = path

        if path is None:
            self._clippath = None
            success = True
        elif isinstance(path, Path):
            self._clippath = TransformedPath(path, transform)
            success = True
        elif isinstance(path, TransformedPatchPath):
            self._clippath = path
            success = True
        elif isinstance(path, TransformedPath):
            self._clippath = path
            success = True

        if not success:
            raise TypeError(
                "Invalid arguments to set_clip_path, of type "
                f"{type(path).__name__} and {type(transform).__name__}")
        # This may result in the callbacks being hit twice, but guarantees they
        # will be hit at least once.
        self.pchanged()
        self.stale = True

    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on all
        backends.
        """
        return self._alpha

    def get_visible(self):
        """Return the visibility."""
        return self._visible

    def get_animated(self):
        """Return whether the artist is animated."""
        return self._animated

    def get_in_layout(self):
        """
        Return boolean flag, ``True`` if artist is included in layout
        calculations.

        E.g. :ref:`constrainedlayout_guide`,
        `.Figure.tight_layout()`, and
        ``fig.savefig(fname, bbox_inches='tight')``.
        """
        return self._in_layout

    def _fully_clipped_to_axes(self):
        """
        Return a boolean flag, ``True`` if the artist is clipped to the Axes
        and can thus be skipped in layout calculations. Requires `get_clip_on`
        is True, one of `clip_box` or `clip_path` is set, ``clip_box.extents``
        is equivalent to ``ax.bbox.extents`` (if set), and ``clip_path._patch``
        is equivalent to ``ax.patch`` (if set).
        """
        # Note that ``clip_path.get_fully_transformed_path().get_extents()``
        # cannot be directly compared to ``axes.bbox.extents`` because the
        # extents may be undefined (i.e. equivalent to ``Bbox.null()``)
        # before the associated artist is drawn, and this method is meant
        # to determine whether ``axes.get_tightbbox()`` may bypass drawing
        clip_box = self.get_clip_box()
        clip_path = self.get_clip_path()
        return (self.axes is not None
                and self.get_clip_on()
                and (clip_box is not None or clip_path is not None)
                and (clip_box is None
                     or np.all(clip_box.extents == self.axes.bbox.extents))
                and (clip_path is None
                     or isinstance(clip_path, TransformedPatchPath)
                     and clip_path._patch is self.axes.patch))

    def get_clip_on(self):
        """Return whether the artist uses clipping."""
        return self._clipon

    def get_clip_box(self):
        """Return the clipbox."""
        return self.clipbox

    def get_clip_path(self):
        """Return the clip path."""
        return self._clippath

    def get_transformed_clip_path_and_affine(self):
        """
        Return the clip path with the non-affine part of its
        transformation applied, and the remaining affine part of its
        transformation.
        """
        if self._clippath is not None:
            return self._clippath.get_transformed_path_and_affine()
        return None, None

    def set_clip_on(self, b):
        """
        Set whether the artist uses clipping.

        When False, artists will be visible outside the Axes which
        can lead to unexpected results.

        Parameters
        ----------
        b : bool
        """
        self._clipon = b
        # This may result in the callbacks being hit twice, but ensures they
        # are hit at least once
        self.pchanged()
        self.stale = True

    def _set_gc_clip(self, gc):
        """Set the clip properly for the gc."""
        if self._clipon:
            if self.clipbox is not None:
                gc.set_clip_rectangle(self.clipbox)
            gc.set_clip_path(self._clippath)
        else:
            gc.set_clip_rectangle(None)
            gc.set_clip_path(None)

    def get_rasterized(self):
        """Return whether the artist is to be rasterized."""
        return self._rasterized

    def set_rasterized(self, rasterized):
        """
        Force rasterized (bitmap) drawing for vector graphics output.

        Rasterized drawing is not supported by all artists. If you try to
        enable this on an artist that does not support it, the command has no
        effect and a warning will be issued.

        This setting is ignored for pixel-based output.

        See also :doc:`/gallery/misc/rasterization_demo`.

        Parameters
        ----------
        rasterized : bool
        """
        supports_rasterization = getattr(self.draw,
                                         "_supports_rasterization", False)
        if rasterized and not supports_rasterization:
            _api.warn_external(f"Rasterization of '{self}' will be ignored")

        self._rasterized = rasterized

    def get_agg_filter(self):
        """Return filter function to be used for agg filter."""
        return self._agg_filter

    def set_agg_filter(self, filter_func):
        """
        Set the agg filter.

        Parameters
        ----------
        filter_func : callable
            A filter function, which takes a (m, n, depth) float array
            and a dpi value, and returns a (m, n, depth) array and two
            offsets from the bottom left corner of the image

            .. ACCEPTS: a filter function, which takes a (m, n, 3) float array
                and a dpi value, and returns a (m, n, 3) array and two offsets
                from the bottom left corner of the image
        """
        self._agg_filter = filter_func
        self.stale = True

    def draw(self, renderer):
        """
        Draw the Artist (and its children) using the given renderer.

        This has no effect if the artist is not visible (`.Artist.get_visible`
        returns False).

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase` subclass.

        Notes
        -----
        This method is overridden in the Artist subclasses.
        """
        if not self.get_visible():
            return
        self.stale = False

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on all backends.

        Parameters
        ----------
        alpha : float or None
            *alpha* must be within the 0-1 range, inclusive.
        """
        if alpha is not None and not isinstance(alpha, Real):
            raise TypeError(
                f'alpha must be numeric or None, not {type(alpha)}')
        if alpha is not None and not (0 <= alpha <= 1):
            raise ValueError(f'alpha ({alpha}) is outside 0-1 range')
        if alpha != self._alpha:
            self._alpha = alpha
            self.pchanged()
            self.stale = True

    def _set_alpha_for_array(self, alpha):
        """
        Set the alpha value used for blending - not supported on all backends.

        Parameters
        ----------
        alpha : array-like or float or None
            All values must be within the 0-1 range, inclusive.
            Masked values and nans are not supported.
        """
        if isinstance(alpha, str):
            raise TypeError("alpha must be numeric or None, not a string")
        if not np.iterable(alpha):
            Artist.set_alpha(self, alpha)
            return
        alpha = np.asarray(alpha)
        if not (0 <= alpha.min() and alpha.max() <= 1):
            raise ValueError('alpha must be between 0 and 1, inclusive, '
                             f'but min is {alpha.min()}, max is {alpha.max()}')
        self._alpha = alpha
        self.pchanged()
        self.stale = True

    def set_visible(self, b):
        """
        Set the artist's visibility.

        Parameters
        ----------
        b : bool
        """
        if b != self._visible:
            self._visible = b
            self.pchanged()
            self.stale = True

    def set_animated(self, b):
        """
        Set whether the artist is intended to be used in an animation.

        If True, the artist is excluded from regular drawing of the figure.
        You have to call `.Figure.draw_artist` / `.Axes.draw_artist`
        explicitly on the artist. This approach is used to speed up animations
        using blitting.

        See also `matplotlib.animation` and
        :ref:`blitting`.

        Parameters
        ----------
        b : bool
        """
        if self._animated != b:
            self._animated = b
            self.pchanged()

    def set_in_layout(self, in_layout):
        """
        Set if artist is to be included in layout calculations,
        E.g. :ref:`constrainedlayout_guide`,
        `.Figure.tight_layout()`, and
        ``fig.savefig(fname, bbox_inches='tight')``.

        Parameters
        ----------
        in_layout : bool
        """
        self._in_layout = in_layout

    def get_label(self):
        """Return the label used for this artist in the legend."""
        return self._label

    def set_label(self, s):
        """
        Set a label that will be displayed in the legend.

        Parameters
        ----------
        s : object
            *s* will be converted to a string by calling `str`.
        """
        label = str(s) if s is not None else None
        if label != self._label:
            self._label = label
            self.pchanged()
            self.stale = True

    def get_zorder(self):
        """Return the artist's zorder."""
        return self.zorder

    def set_zorder(self, level):
        """
        Set the zorder for the artist.  Artists with lower zorder
        values are drawn first.

        Parameters
        ----------
        level : float
        """
        if level is None:
            level = self.__class__.zorder
        if level != self.zorder:
            self.zorder = level
            self.pchanged()
            self.stale = True

    @property
    def sticky_edges(self):
        """
        ``x`` and ``y`` sticky edge lists for autoscaling.

        When performing autoscaling, if a data limit coincides with a value in
        the corresponding sticky_edges list, then no margin will be added--the
        view limit "sticks" to the edge. A typical use case is histograms,
        where one usually expects no margin on the bottom edge (0) of the
        histogram.

        Moreover, margin expansion "bumps" against sticky edges and cannot
        cross them.  For example, if the upper data limit is 1.0, the upper
        view limit computed by simple margin application is 1.2, but there is a
        sticky edge at 1.1, then the actual upper view limit will be 1.1.

        This attribute cannot be assigned to; however, the ``x`` and ``y``
        lists can be modified in place as needed.

        Examples
        --------
        >>> artist.sticky_edges.x[:] = (xmin, xmax)
        >>> artist.sticky_edges.y[:] = (ymin, ymax)

        """
        return self._sticky_edges

    def update_from(self, other):
        """Copy properties from *other* to *self*."""
        self._transform = other._transform
        self._transformSet = other._transformSet
        self._visible = other._visible
        self._alpha = other._alpha
        self.clipbox = other.clipbox
        self._clipon = other._clipon
        self._clippath = other._clippath
        self._label = other._label
        self._sketch = other._sketch
        self._path_effects = other._path_effects
        self.sticky_edges.x[:] = other.sticky_edges.x.copy()
        self.sticky_edges.y[:] = other.sticky_edges.y.copy()
        self.pchanged()
        self.stale = True

    def properties(self):
        """Return a dictionary of all the properties of the artist."""
        return ArtistInspector(self).properties()

    def _update_props(self, props, errfmt):
        """
        Helper for `.Artist.set` and `.Artist.update`.

        *errfmt* is used to generate error messages for invalid property
        names; it gets formatted with ``type(self)`` for "{cls}" and the
        property name for "{prop_name}".
        """
        ret = []
        with cbook._setattr_cm(self, eventson=False):
            for k, v in props.items():
                # Allow attributes we want to be able to update through
                # art.update, art.set, setp.
                if k == "axes":
                    ret.append(setattr(self, k, v))
                else:
                    func = getattr(self, f"set_{k}", None)
                    if not callable(func):
                        raise AttributeError(
                            errfmt.format(cls=type(self), prop_name=k),
                            name=k)
                    ret.append(func(v))
        if ret:
            self.pchanged()
            self.stale = True
        return ret

    def update(self, props):
        """
        Update this artist's properties from the dict *props*.

        Parameters
        ----------
        props : dict
        """
        return self._update_props(
            props, "{cls.__name__!r} object has no property {prop_name!r}")

    def _internal_update(self, kwargs):
        """
        Update artist properties without prenormalizing them, but generating
        errors as if calling `set`.

        The lack of prenormalization is to maintain backcompatibility.
        """
        return self._update_props(
            kwargs, "{cls.__name__}.set() got an unexpected keyword argument "
            "{prop_name!r}")

    def set(self, **kwargs):
        # docstring and signature are auto-generated via
        # Artist._update_set_signature_and_docstring() at the end of the
        # module.
        return self._internal_update(cbook.normalize_kwargs(kwargs, self))

    @contextlib.contextmanager
    def _cm_set(self, **kwargs):
        """
        `.Artist.set` context-manager that restores original values at exit.
        """
        orig_vals = {k: getattr(self, f"get_{k}")() for k in kwargs}
        try:
            self.set(**kwargs)
            yield
        finally:
            self.set(**orig_vals)

    def findobj(self, match=None, include_self=True):
        """
        Find artist objects.

        Recursively find all `.Artist` instances contained in the artist.

        Parameters
        ----------
        match
            A filter criterion for the matches. This can be

            - *None*: Return all objects contained in artist.
            - A function with signature ``def match(artist: Artist) -> bool``.
              The result will only contain artists for which the function
              returns *True*.
            - A class instance: e.g., `.Line2D`. The result will only contain
              artists of this class or its subclasses (``isinstance`` check).

        include_self : bool
            Include *self* in the list to be checked for a match.

        Returns
        -------
        list of `.Artist`

        """
        if match is None:  # always return True
            def matchfunc(x):
                return True
        elif isinstance(match, type) and issubclass(match, Artist):
            def matchfunc(x):
                return isinstance(x, match)
        elif callable(match):
            matchfunc = match
        else:
            raise ValueError('match must be None, a matplotlib.artist.Artist '
                             'subclass, or a callable')

        artists = reduce(operator.iadd,
                         [c.findobj(matchfunc) for c in self.get_children()], [])
        if include_self and matchfunc(self):
            artists.append(self)
        return artists

    def get_cursor_data(self, event):
        """
        Return the cursor data for a given event.

        .. note::
            This method is intended to be overridden by artist subclasses.
            As an end-user of Matplotlib you will most likely not call this
            method yourself.

        Cursor data can be used by Artists to provide additional context
        information for a given event. The default implementation just returns
        *None*.

        Subclasses can override the method and return arbitrary data. However,
        when doing so, they must ensure that `.format_cursor_data` can convert
        the data to a string representation.

        The only current use case is displaying the z-value of an `.AxesImage`
        in the status bar of a plot window, while moving the mouse.

        Parameters
        ----------
        event : `~matplotlib.backend_bases.MouseEvent`

        See Also
        --------
        format_cursor_data

        """
        return None

    def format_cursor_data(self, data):
        """
        Return a string representation of *data*.

        .. note::
            This method is intended to be overridden by artist subclasses.
            As an end-user of Matplotlib you will most likely not call this
            method yourself.

        The default implementation converts ints and floats and arrays of ints
        and floats into a comma-separated string enclosed in square brackets,
        unless the artist has an associated colorbar, in which case scalar
        values are formatted using the colorbar's formatter.

        See Also
        --------
        get_cursor_data
        """
        if np.ndim(data) == 0 and hasattr(self, "_format_cursor_data_override"):
            # workaround for ScalarMappable to be able to define its own
            # format_cursor_data(). See ScalarMappable._format_cursor_data_override
            # for details.
            return self._format_cursor_data_override(data)
        else:
            try:
                data[0]
            except (TypeError, IndexError):
                data = [data]
            data_str = ', '.join(f'{item:0.3g}' for item in data
                                 if isinstance(item, Number))
            return "[" + data_str + "]"

    def get_mouseover(self):
        """
        Return whether this artist is queried for custom context information
        when the mouse cursor moves over it.
        """
        return self._mouseover

    def set_mouseover(self, mouseover):
        """
        Set whether this artist is queried for custom context information when
        the mouse cursor moves over it.

        Parameters
        ----------
        mouseover : bool

        See Also
        --------
        get_cursor_data
        .ToolCursorPosition
        .NavigationToolbar2
        """
        self._mouseover = bool(mouseover)
        ax = self.axes
        if ax:
            if self._mouseover:
                ax._mouseover_set.add(self)
            else:
                ax._mouseover_set.discard(self)

    mouseover = property(get_mouseover, set_mouseover)  # backcompat.


def _get_tightbbox_for_layout_only(obj, *args, **kwargs):
    """
    Matplotlib's `.Axes.get_tightbbox` and `.Axis.get_tightbbox` support a
    *for_layout_only* kwarg; this helper tries to use the kwarg but skips it
    when encountering third-party subclasses that do not support it.
    """
    try:
        return obj.get_tightbbox(*args, **{**kwargs, "for_layout_only": True})
    except TypeError:
        return obj.get_tightbbox(*args, **kwargs)


class ArtistInspector:
    """
    A helper class to inspect an `~matplotlib.artist.Artist` and return
    information about its settable properties and their current values.
    """

    def __init__(self, o):
        r"""
        Initialize the artist inspector with an `Artist` or an iterable of
        `Artist`\s.  If an iterable is used, we assume it is a homogeneous
        sequence (all `Artist`\s are of the same type) and it is your
        responsibility to make sure this is so.
        """
        if not isinstance(o, Artist):
            if np.iterable(o):
                o = list(o)
                if len(o):
                    o = o[0]

        self.oorig = o
        if not isinstance(o, type):
            o = type(o)
        self.o = o

        self.aliasd = self.get_aliases()

    def get_aliases(self):
        """
        Get a dict mapping property fullnames to sets of aliases for each alias
        in the :class:`~matplotlib.artist.ArtistInspector`.

        e.g., for lines::

          {'markerfacecolor': {'mfc'},
           'linewidth'      : {'lw'},
          }
        """
        names = [name for name in dir(self.o)
                 if name.startswith(('set_', 'get_'))
                    and callable(getattr(self.o, name))]
        aliases = {}
        for name in names:
            func = getattr(self.o, name)
            if not self.is_alias(func):
                continue
            propname = re.search(f"`({name[:4]}.*)`",  # get_.*/set_.*
                                 inspect.getdoc(func)).group(1)
            aliases.setdefault(propname[4:], set()).add(name[4:])
        return aliases

    _get_valid_values_regex = re.compile(
        r"\n\s*(?:\.\.\s+)?ACCEPTS:\s*((?:.|\n)*?)(?:$|(?:\n\n))"
    )

    def get_valid_values(self, attr):
        """
        Get the legal arguments for the setter associated with *attr*.

        This is done by querying the docstring of the setter for a line that
        begins with "ACCEPTS:" or ".. ACCEPTS:", and then by looking for a
        numpydoc-style documentation for the setter's first argument.
        """

        name = 'set_%s' % attr
        if not hasattr(self.o, name):
            raise AttributeError(f'{self.o} has no function {name}')
        func = getattr(self.o, name)

        if hasattr(func, '_kwarg_doc'):
            return func._kwarg_doc

        docstring = inspect.getdoc(func)
        if docstring is None:
            return 'unknown'

        if docstring.startswith('Alias for '):
            return None

        match = self._get_valid_values_regex.search(docstring)
        if match is not None:
            return re.sub("\n *", " ", match.group(1))

        # Much faster than list(inspect.signature(func).parameters)[1],
        # although barely relevant wrt. matplotlib's total import time.
        param_name = func.__code__.co_varnames[1]
        # We could set the presence * based on whether the parameter is a
        # varargs (it can't be a varkwargs) but it's not really worth it.
        match = re.search(fr"(?m)^ *\*?{param_name} : (.+)", docstring)
        if match:
            return match.group(1)

        return 'unknown'

    def _replace_path(self, source_class):
        """
        Changes the full path to the public API path that is used
        in sphinx. This is needed for links to work.
        """
        replace_dict = {'_base._AxesBase': 'Axes',
                        '_axes.Axes': 'Axes'}
        for key, value in replace_dict.items():
            source_class = source_class.replace(key, value)
        return source_class

    def get_setters(self):
        """
        Get the attribute strings with setters for object.

        For example, for a line, return ``['markerfacecolor', 'linewidth',
        ....]``.
        """
        setters = []
        for name in dir(self.o):
            if not name.startswith('set_'):
                continue
            func = getattr(self.o, name)
            if (not callable(func)
                    or self.number_of_parameters(func) < 2
                    or self.is_alias(func)):
                continue
            setters.append(name[4:])
        return setters

    @staticmethod
    @cache
    def number_of_parameters(func):
        """Return number of parameters of the callable *func*."""
        return len(inspect.signature(func).parameters)

    @staticmethod
    @cache
    def is_alias(method):
        """
        Return whether the object *method* is an alias for another method.
        """

        ds = inspect.getdoc(method)
        if ds is None:
            return False

        return ds.startswith('Alias for ')

    def aliased_name(self, s):
        """
        Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME'.

        For example, for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'.
        """
        aliases = ''.join(' or %s' % x for x in sorted(self.aliasd.get(s, [])))
        return s + aliases

    _NOT_LINKABLE = {
        # A set of property setter methods that are not available in our
        # current docs. This is a workaround used to prevent trying to link
        # these setters which would lead to "target reference not found"
        # warnings during doc build.
        'matplotlib.image._ImageBase.set_alpha',
        'matplotlib.image._ImageBase.set_array',
        'matplotlib.image._ImageBase.set_data',
        'matplotlib.image._ImageBase.set_filternorm',
        'matplotlib.image._ImageBase.set_filterrad',
        'matplotlib.image._ImageBase.set_interpolation',
        'matplotlib.image._ImageBase.set_interpolation_stage',
        'matplotlib.image._ImageBase.set_resample',
        'matplotlib.text._AnnotationBase.set_annotation_clip',
    }

    def aliased_name_rest(self, s, target):
        """
        Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME',
        formatted for reST.

        For example, for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'.
        """
        # workaround to prevent "reference target not found"
        if target in self._NOT_LINKABLE:
            return f'``{s}``'

        aliases = ''.join(
            f' or :meth:`{a} <{target}>`' for a in sorted(self.aliasd.get(s, [])))
        return f':meth:`{s} <{target}>`{aliases}'

    def pprint_setters(self, prop=None, leadingspace=2):
        """
        If *prop* is *None*, return a list of strings of all settable
        properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of property : valid
        values.
        """
        if leadingspace:
            pad = ' ' * leadingspace
        else:
            pad = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return f'{pad}{prop}: {accepts}'

        lines = []
        for prop in sorted(self.get_setters()):
            accepts = self.get_valid_values(prop)
            name = self.aliased_name(prop)
            lines.append(f'{pad}{name}: {accepts}')
        return lines

    def pprint_setters_rest(self, prop=None, leadingspace=4):
        """
        If *prop* is *None*, return a list of reST-formatted strings of all
        settable properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of "property : valid"
        values.
        """
        if leadingspace:
            pad = ' ' * leadingspace
        else:
            pad = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return f'{pad}{prop}: {accepts}'

        prop_and_qualnames = []
        for prop in sorted(self.get_setters()):
            # Find the parent method which actually provides the docstring.
            for cls in self.o.__mro__:
                method = getattr(cls, f"set_{prop}", None)
                if method and method.__doc__ is not None:
                    break
            else:  # No docstring available.
                method = getattr(self.o, f"set_{prop}")
            prop_and_qualnames.append(
                (prop, f"{method.__module__}.{method.__qualname__}"))

        names = [self.aliased_name_rest(prop, target)
                 .replace('_base._AxesBase', 'Axes')
                 .replace('_axes.Axes', 'Axes')
                 for prop, target in prop_and_qualnames]
        accepts = [self.get_valid_values(prop)
                   for prop, _ in prop_and_qualnames]

        col0_len = max(len(n) for n in names)
        col1_len = max(len(a) for a in accepts)
        table_formatstr = pad + '   ' + '=' * col0_len + '   ' + '=' * col1_len

        return [
            '',
            pad + '.. table::',
            pad + '   :class: property-table',
            '',
            table_formatstr,
            pad + '   ' + 'Property'.ljust(col0_len)
            + '   ' + 'Description'.ljust(col1_len),
            table_formatstr,
            *[pad + '   ' + n.ljust(col0_len) + '   ' + a.ljust(col1_len)
              for n, a in zip(names, accepts)],
            table_formatstr,
            '',
        ]

    def properties(self):
        """Return a dictionary mapping property name -> value."""
        o = self.oorig
        getters = [name for name in dir(o)
                   if name.startswith('get_') and callable(getattr(o, name))]
        getters.sort()
        d = {}
        for name in getters:
            func = getattr(o, name)
            if self.is_alias(func):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    val = func()
            except Exception:
                continue
            else:
                d[name[4:]] = val
        return d

    def pprint_getters(self):
        """Return the getters and actual values as list of strings."""
        lines = []
        for name, val in sorted(self.properties().items()):
            if getattr(val, 'shape', ()) != () and len(val) > 6:
                s = str(val[:6]) + '...'
            else:
                s = str(val)
            s = s.replace('\n', ' ')
            if len(s) > 50:
                s = s[:50] + '...'
            name = self.aliased_name(name)
            lines.append(f'    {name} = {s}')
        return lines


def getp(obj, property=None):
    """
    Return the value of an `.Artist`'s *property*, or print all of them.

    Parameters
    ----------
    obj : `~matplotlib.artist.Artist`
        The queried artist; e.g., a `.Line2D`, a `.Text`, or an `~.axes.Axes`.

    property : str or None, default: None
        If *property* is 'somename', this function returns
        ``obj.get_somename()``.

        If it's None (or unset), it *prints* all gettable properties from
        *obj*.  Many properties have aliases for shorter typing, e.g. 'lw' is
        an alias for 'linewidth'.  In the output, aliases and full property
        names will be listed as:

          property or alias = value

        e.g.:

          linewidth or lw = 2

    See Also
    --------
    setp
    """
    if property is None:
        insp = ArtistInspector(obj)
        ret = insp.pprint_getters()
        print('\n'.join(ret))
        return
    return getattr(obj, 'get_' + property)()

# alias
get = getp


def setp(obj, *args, file=None, **kwargs):
    """
    Set one or more properties on an `.Artist`, or list allowed values.

    Parameters
    ----------
    obj : `~matplotlib.artist.Artist` or list of `.Artist`
        The artist(s) whose properties are being set or queried.  When setting
        properties, all artists are affected; when querying the allowed values,
        only the first instance in the sequence is queried.

        For example, two lines can be made thicker and red with a single call:

        >>> x = arange(0, 1, 0.01)
        >>> lines = plot(x, sin(2*pi*x), x, sin(4*pi*x))
        >>> setp(lines, linewidth=2, color='r')

    file : file-like, default: `sys.stdout`
        Where `setp` writes its output when asked to list allowed values.

        >>> with open('output.log') as file:
        ...     setp(line, file=file)

        The default, ``None``, means `sys.stdout`.

    *args, **kwargs
        The properties to set.  The following combinations are supported:

        - Set the linestyle of a line to be dashed:

          >>> line, = plot([1, 2, 3])
          >>> setp(line, linestyle='--')

        - Set multiple properties at once:

          >>> setp(line, linewidth=2, color='r')

        - List allowed values for a line's linestyle:

          >>> setp(line, 'linestyle')
          linestyle: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}

        - List all properties that can be set, and their allowed values:

          >>> setp(line)
          agg_filter: a filter function, ...
          [long output listing omitted]

        `setp` also supports MATLAB style string/value pairs.  For example, the
        following are equivalent:

        >>> setp(lines, 'linewidth', 2, 'color', 'r')  # MATLAB style
        >>> setp(lines, linewidth=2, color='r')        # Python style

    See Also
    --------
    getp
    """

    if isinstance(obj, Artist):
        objs = [obj]
    else:
        objs = list(cbook.flatten(obj))

    if not objs:
        return

    insp = ArtistInspector(objs[0])

    if not kwargs and len(args) < 2:
        if args:
            print(insp.pprint_setters(prop=args[0]), file=file)
        else:
            print('\n'.join(insp.pprint_setters()), file=file)
        return

    if len(args) % 2:
        raise ValueError('The set args must be string, value pairs')

    funcvals = dict(zip(args[::2], args[1::2]))
    ret = [o.update(funcvals) for o in objs] + [o.set(**kwargs) for o in objs]
    return list(cbook.flatten(ret))


def kwdoc(artist):
    r"""
    Inspect an `~matplotlib.artist.Artist` class (using `.ArtistInspector`) and
    return information about its settable properties and their current values.

    Parameters
    ----------
    artist : `~matplotlib.artist.Artist` or an iterable of `Artist`\s

    Returns
    -------
    str
        The settable properties of *artist*, as plain text if
        :rc:`docstring.hardcopy` is False and as a rst table (intended for
        use in Sphinx) if it is True.
    """
    ai = ArtistInspector(artist)
    return ('\n'.join(ai.pprint_setters_rest(leadingspace=4))
            if mpl.rcParams['docstring.hardcopy'] else
            'Properties:\n' + '\n'.join(ai.pprint_setters(leadingspace=4)))

# We defer this to the end of them module, because it needs ArtistInspector
# to be defined.
Artist._update_set_signature_and_docstring()
<|endoftext|>"""
Plotting of string "category" data: ``plot(['d', 'f', 'a'], [1, 2, 3])`` will
plot three points with x-axis values of 'd', 'f', 'a'.

See :doc:`/gallery/lines_bars_and_markers/categorical_variables` for an
example.

The module uses Matplotlib's `matplotlib.units` mechanism to convert from
strings to integers and provides a tick locator, a tick formatter, and the
`.UnitData` class that creates and stores the string-to-integer mapping.
"""

from collections import OrderedDict
import dateutil.parser
import itertools
import logging

import numpy as np

from matplotlib import _api, cbook, ticker, units


_log = logging.getLogger(__name__)


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """
        Convert strings in *value* to floats using mapping information stored
        in the *unit* object.

        Parameters
        ----------
        value : str or iterable
            Value or list of values to be converted.
        unit : `.UnitData`
            An object mapping strings to integers.
        axis : `~matplotlib.axis.Axis`
            The axis on which the converted value is plotted.

            .. note:: *axis* is unused.

        Returns
        -------
        float or `~numpy.ndarray` of float
        """
        if unit is None:
            raise ValueError(
                'Missing category information for StrCategoryConverter; '
                'this might be caused by unintendedly mixing categorical and '
                'numeric data')
        StrCategoryConverter._validate_unit(unit)
        # dtype = object preserves numerical pass throughs
        values = np.atleast_1d(np.array(value, dtype=object))
        # force an update so it also does type checking
        unit.update(values)
        s = np.vectorize(unit._mapping.__getitem__, otypes=[float])(values)
        return s if not cbook.is_scalar_or_string(value) else s[0]

    @staticmethod
    def axisinfo(unit, axis):
        """
        Set the default axis ticks and labels.

        Parameters
        ----------
        unit : `.UnitData`
            object string unit information for value
        axis : `~matplotlib.axis.Axis`
            axis for which information is being set

            .. note:: *axis* is not used

        Returns
        -------
        `~matplotlib.units.AxisInfo`
            Information to support default tick labeling

        """
        StrCategoryConverter._validate_unit(unit)
        # locator and formatter take mapping dict because
        # args need to be pass by reference for updates
        majloc = StrCategoryLocator(unit._mapping)
        majfmt = StrCategoryFormatter(unit._mapping)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def default_units(data, axis):
        """
        Set and update the `~matplotlib.axis.Axis` units.

        Parameters
        ----------
        data : str or iterable of str
        axis : `~matplotlib.axis.Axis`
            axis on which the data is plotted

        Returns
        -------
        `.UnitData`
            object storing string to integer mapping
        """
        # the conversion call stack is default_units -> axis_info -> convert
        if axis.units is None:
            axis.set_units(UnitData(data))
        else:
            axis.units.update(data)
        return axis.units

    @staticmethod
    def _validate_unit(unit):
        if not hasattr(unit, '_mapping'):
            raise ValueError(
                f'Provided unit "{unit}" is not valid for a categorical '
                'converter, as it does not have a _mapping attribute.')


class StrCategoryLocator(ticker.Locator):
    """Tick at every integer mapping of the string data."""
    def __init__(self, units_mapping):
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
        self._units = units_mapping

    def __call__(self):
        # docstring inherited
        return list(self._units.values())

    def tick_values(self, vmin, vmax):
        # docstring inherited
        return self()


class StrCategoryFormatter(ticker.Formatter):
    """String representation of the data at every tick."""
    def __init__(self, units_mapping):
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
        self._units = units_mapping

    def __call__(self, x, pos=None):
        # docstring inherited
        return self.format_ticks([x])[0]

    def format_ticks(self, values):
        # docstring inherited
        r_mapping = {v: self._text(k) for k, v in self._units.items()}
        return [r_mapping.get(round(val), '') for val in values]

    @staticmethod
    def _text(value):
        """Convert text values into utf-8 or ascii strings."""
        if isinstance(value, bytes):
            value = value.decode(encoding='utf-8')
        elif not isinstance(value, str):
            value = str(value)
        return value


class UnitData:
    def __init__(self, data=None):
        """
        Create mapping between unique categorical values and integer ids.

        Parameters
        ----------
        data : iterable
            sequence of string values
        """
        self._mapping = OrderedDict()
        self._counter = itertools.count()
        if data is not None:
            self.update(data)

    @staticmethod
    def _str_is_convertible(val):
        """
        Helper method to check whether a string can be parsed as float or date.
        """
        try:
            float(val)
        except ValueError:
            try:
                dateutil.parser.parse(val)
            except (ValueError, TypeError):
                # TypeError if dateutil >= 2.8.1 else ValueError
                return False
        return True

    def update(self, data):
        """
        Map new values to integer identifiers.

        Parameters
        ----------
        data : iterable of str or bytes

        Raises
        ------
        TypeError
            If elements in *data* are neither str nor bytes.
        """
        data = np.atleast_1d(np.array(data, dtype=object))
        # check if convertible to number:
        convertible = True
        for val in OrderedDict.fromkeys(data):
            # OrderedDict just iterates over unique values in data.
            _api.check_isinstance((str, bytes), value=val)
            if convertible:
                # this will only be called so long as convertible is True.
                convertible = self._str_is_convertible(val)
            if val not in self._mapping:
                self._mapping[val] = next(self._counter)
        if data.size and convertible:
            _log.info('Using categorical units to plot a list of strings '
                      'that are all parsable as floats or dates. If these '
                      'strings should be plotted as numbers, cast to the '
                      'appropriate data type before plotting.')


# Register the converter with Matplotlib's unit framework
# Intentionally set to a single instance
units.registry[str] = \
    units.registry[np.str_] = \
    units.registry[bytes] = \
    units.registry[np.bytes_] = StrCategoryConverter()
<|endoftext|>"""
Default legend handlers.

.. important::

    This is a low-level legend API, which most end users do not need.

    We recommend that you are familiar with the :ref:`legend guide
    <legend_guide>` before reading this documentation.

Legend handlers are expected to be a callable object with a following
signature::

    legend_handler(legend, orig_handle, fontsize, handlebox)

Where *legend* is the legend itself, *orig_handle* is the original
plot, *fontsize* is the fontsize in pixels, and *handlebox* is an
`.OffsetBox` instance. Within the call, you should create relevant
artists (using relevant properties from the *legend* and/or
*orig_handle*) and add them into the *handlebox*. The artists need to
be scaled according to the *fontsize* (note that the size is in pixels,
i.e., this is dpi-scaled value).

This module includes definition of several legend handler classes
derived from the base class (HandlerBase) with the following method::

    def legend_artist(self, legend, orig_handle, fontsize, handlebox)
"""

from itertools import cycle

import numpy as np

from matplotlib import cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll


def update_from_first_child(tgt, src):
    first_child = next(iter(src.get_children()), None)
    if first_child is not None:
        tgt.update_from(first_child)


class HandlerBase:
    """
    A base class for default legend handlers.

    The derived classes are meant to override *create_artists* method, which
    has the following signature::

      def create_artists(self, legend, orig_handle,
                         xdescent, ydescent, width, height, fontsize,
                         trans):

    The overridden method needs to create artists of the given
    transform that fits in the given dimension (xdescent, ydescent,
    width, height) that are scaled by fontsize if necessary.

    """
    def __init__(self, xpad=0., ypad=0., update_func=None):
        """
        Parameters
        ----------
        xpad : float, optional
            Padding in x-direction.
        ypad : float, optional
            Padding in y-direction.
        update_func : callable, optional
            Function for updating the legend handler properties from another
            legend handler, used by `~HandlerBase.update_prop`.
        """
        self._xpad, self._ypad = xpad, ypad
        self._update_prop_func = update_func

    def _update_prop(self, legend_handle, orig_handle):
        if self._update_prop_func is None:
            self._default_update_prop(legend_handle, orig_handle)
        else:
            self._update_prop_func(legend_handle, orig_handle)

    def _default_update_prop(self, legend_handle, orig_handle):
        legend_handle.update_from(orig_handle)

    def update_prop(self, legend_handle, orig_handle, legend):

        self._update_prop(legend_handle, orig_handle)

        legend._set_artist_props(legend_handle)
        legend_handle.set_clip_box(None)
        legend_handle.set_clip_path(None)

    def adjust_drawing_area(self, legend, orig_handle,
                            xdescent, ydescent, width, height, fontsize,
                            ):
        xdescent = xdescent - self._xpad * fontsize
        ydescent = ydescent - self._ypad * fontsize
        width = width - self._xpad * fontsize
        height = height - self._ypad * fontsize
        return xdescent, ydescent, width, height

    def legend_artist(self, legend, orig_handle,
                      fontsize, handlebox):
        """
        Return the artist that this HandlerBase generates for the given
        original artist/handle.

        Parameters
        ----------
        legend : `~matplotlib.legend.Legend`
            The legend for which these legend artists are being created.
        orig_handle : :class:`matplotlib.artist.Artist` or similar
            The object for which these legend artists are being created.
        fontsize : int
            The fontsize in pixels. The artists being created should
            be scaled according to the given fontsize.
        handlebox : `~matplotlib.offsetbox.OffsetBox`
            The box which has been created to hold this legend entry's
            artists. Artists created in the `legend_artist` method must
            be added to this handlebox inside this method.

        """
        xdescent, ydescent, width, height = self.adjust_drawing_area(
                 legend, orig_handle,
                 handlebox.xdescent, handlebox.ydescent,
                 handlebox.width, handlebox.height,
                 fontsize)
        artists = self.create_artists(legend, orig_handle,
                                      xdescent, ydescent, width, height,
                                      fontsize, handlebox.get_transform())

        # create_artists will return a list of artists.
        for a in artists:
            handlebox.add_artist(a)

        # we only return the first artist
        return artists[0]

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        """
        Return the legend artists generated.

        Parameters
        ----------
        legend : `~matplotlib.legend.Legend`
            The legend for which these legend artists are being created.
        orig_handle : `~matplotlib.artist.Artist` or similar
            The object for which these legend artists are being created.
        xdescent, ydescent, width, height : int
            The rectangle (*xdescent*, *ydescent*, *width*, *height*) that the
            legend artists being created should fit within.
        fontsize : int
            The fontsize in pixels. The legend artists being created should
            be scaled according to the given fontsize.
        trans : `~matplotlib.transforms.Transform`
            The transform that is applied to the legend artists being created.
            Typically from unit coordinates in the handler box to screen
            coordinates.
        """
        raise NotImplementedError('Derived must override')


class HandlerNpoints(HandlerBase):
    """
    A legend handler that shows *numpoints* points in the legend entry.
    """

    def __init__(self, marker_pad=0.3, numpoints=None, **kwargs):
        """
        Parameters
        ----------
        marker_pad : float
            Padding between points in legend entry.
        numpoints : int
            Number of points to show in legend entry.
        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
        """
        super().__init__(**kwargs)

        self._numpoints = numpoints
        self._marker_pad = marker_pad

    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.numpoints
        else:
            return self._numpoints

    def get_xdata(self, legend, xdescent, ydescent, width, height, fontsize):
        numpoints = self.get_numpoints(legend)
        if numpoints > 1:
            # we put some pad here to compensate the size of the marker
            pad = self._marker_pad * fontsize
            xdata = np.linspace(-xdescent + pad,
                                -xdescent + width - pad,
                                numpoints)
            xdata_marker = xdata
        else:
            xdata = [-xdescent, -xdescent + width]
            xdata_marker = [-xdescent + 0.5 * width]
        return xdata, xdata_marker


class HandlerNpointsYoffsets(HandlerNpoints):
    """
    A legend handler that shows *numpoints* in the legend, and allows them to
    be individually offset in the y-direction.
    """

    def __init__(self, numpoints=None, yoffsets=None, **kwargs):
        """
        Parameters
        ----------
        numpoints : int
            Number of points to show in legend entry.
        yoffsets : array of floats
            Length *numpoints* list of y offsets for each point in
            legend entry.
        **kwargs
            Keyword arguments forwarded to `.HandlerNpoints`.
        """
        super().__init__(numpoints=numpoints, **kwargs)
        self._yoffsets = yoffsets

    def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
        if self._yoffsets is None:
            ydata = height * legend._scatteryoffsets
        else:
            ydata = height * np.asarray(self._yoffsets)

        return ydata


class HandlerLine2DCompound(HandlerNpoints):
    """
    Original handler for `.Line2D` instances, that relies on combining
    a line-only with a marker-only artist.  May be deprecated in the future.
    """

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # docstring inherited
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = np.full_like(xdata, ((height - ydescent) / 2))
        legline = Line2D(xdata, ydata)

        self.update_prop(legline, orig_handle, legend)
        legline.set_drawstyle('default')
        legline.set_marker("")

        legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        self.update_prop(legline_marker, orig_handle, legend)
        legline_marker.set_linestyle('None')
        if legend.markerscale != 1:
            newsz = legline_marker.get_markersize() * legend.markerscale
            legline_marker.set_markersize(newsz)
        # we don't want to add this to the return list because
        # the texts and handles are assumed to be in one-to-one
        # correspondence.
        legline._legmarker = legline_marker

        legline.set_transform(trans)
        legline_marker.set_transform(trans)

        return [legline, legline_marker]


class HandlerLine2D(HandlerNpoints):
    """
    Handler for `.Line2D` instances.

    See Also
    --------
    HandlerLine2DCompound : An earlier handler implementation, which used one
                            artist for the line and another for the marker(s).
    """

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # docstring inherited
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        markevery = None
        if self.get_numpoints(legend) == 1:
            # Special case: one wants a single marker in the center
            # and a line that extends on both sides. One will use a
            # 3 points line, but only mark the #1 (i.e. middle) point.
            xdata = np.linspace(xdata[0], xdata[-1], 3)
            markevery = [1]

        ydata = np.full_like(xdata, (height - ydescent) / 2)
        legline = Line2D(xdata, ydata, markevery=markevery)

        self.update_prop(legline, orig_handle, legend)

        if legend.markerscale != 1:
            newsz = legline.get_markersize() * legend.markerscale
            legline.set_markersize(newsz)

        legline.set_transform(trans)

        return [legline]


class HandlerPatch(HandlerBase):
    """
    Handler for `.Patch` instances.
    """

    def __init__(self, patch_func=None, **kwargs):
        """
        Parameters
        ----------
        patch_func : callable, optional
            The function that creates the legend key artist.
            *patch_func* should have the signature::

                def patch_func(legend=legend, orig_handle=orig_handle,
                               xdescent=xdescent, ydescent=ydescent,
                               width=width, height=height, fontsize=fontsize)

            Subsequently, the created artist will have its ``update_prop``
            method called and the appropriate transform will be applied.

        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
        """
        super().__init__(**kwargs)
        self._patch_func = patch_func

    def _create_patch(self, legend, orig_handle,
                      xdescent, ydescent, width, height, fontsize):
        if self._patch_func is None:
            p = Rectangle(xy=(-xdescent, -ydescent),
                          width=width, height=height)
        else:
            p = self._patch_func(legend=legend, orig_handle=orig_handle,
                                 xdescent=xdescent, ydescent=ydescent,
                                 width=width, height=height, fontsize=fontsize)
        return p

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # docstring inherited
        p = self._create_patch(legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerStepPatch(HandlerBase):
    """
    Handler for `~.matplotlib.patches.StepPatch` instances.
    """

    @staticmethod
    def _create_patch(orig_handle, xdescent, ydescent, width, height):
        return Rectangle(xy=(-xdescent, -ydescent), width=width,
                         height=height, color=orig_handle.get_facecolor())

    @staticmethod
    def _create_line(orig_handle, width, height):
        # Unfilled StepPatch should show as a line
        legline = Line2D([0, width], [height/2, height/2],
                         color=orig_handle.get_edgecolor(),
                         linestyle=orig_handle.get_linestyle(),
                         linewidth=orig_handle.get_linewidth(),
                         )

        # Overwrite manually because patch and line properties don't mix
        legline.set_drawstyle('default')
        legline.set_marker("")
        return legline

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # docstring inherited
        if orig_handle.get_fill() or (orig_handle.get_hatch() is not None):
            p = self._create_patch(orig_handle, xdescent, ydescent, width,
                                   height)
            self.update_prop(p, orig_handle, legend)
        else:
            p = self._create_line(orig_handle, width, height)
        p.set_transform(trans)
        return [p]


class HandlerLineCollection(HandlerLine2D):
    """
    Handler for `.LineCollection` instances.
    """
    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.scatterpoints
        else:
            return self._numpoints

    def _default_update_prop(self, legend_handle, orig_handle):
        lw = orig_handle.get_linewidths()[0]
        dashes = orig_handle._us_linestyles[0]
        color = orig_handle.get_colors()[0]
        legend_handle.set_color(color)
        legend_handle.set_linestyle(dashes)
        legend_handle.set_linewidth(lw)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # docstring inherited
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        ydata = np.full_like(xdata, (height - ydescent) / 2)
        legline = Line2D(xdata, ydata)

        self.update_prop(legline, orig_handle, legend)
        legline.set_transform(trans)

        return [legline]


class HandlerRegularPolyCollection(HandlerNpointsYoffsets):
    r"""Handler for `.RegularPolyCollection`\s."""

    def __init__(self, yoffsets=None, sizes=None, **kwargs):
        super().__init__(yoffsets=yoffsets, **kwargs)

        self._sizes = sizes

    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.scatterpoints
        else:
            return self._numpoints

    def get_sizes(self, legend, orig_handle,
                  xdescent, ydescent, width, height, fontsize):
        if self._sizes is None:
            handle_sizes = orig_handle.get_sizes()
            if not len(handle_sizes):
                handle_sizes = [1]
            size_max = max(handle_sizes) * legend.markerscale ** 2
            size_min = min(handle_sizes) * legend.markerscale ** 2

            numpoints = self.get_numpoints(legend)
            if numpoints < 4:
                sizes = [.5 * (size_max + size_min), size_max,
                         size_min][:numpoints]
            else:
                rng = (size_max - size_min)
                sizes = rng * np.linspace(0, 1, numpoints) + size_min
        else:
            sizes = self._sizes

        return sizes

    def update_prop(self, legend_handle, orig_handle, legend):

        self._update_prop(legend_handle, orig_handle)

        legend_handle.set_figure(legend.get_figure(root=False))
        # legend._set_artist_props(legend_handle)
        legend_handle.set_clip_box(None)
        legend_handle.set_clip_path(None)

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        return type(orig_handle)(
            orig_handle.get_numsides(),
            rotation=orig_handle.get_rotation(), sizes=sizes,
