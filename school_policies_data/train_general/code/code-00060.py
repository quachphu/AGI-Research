        The last point specified is on-curve, all others are off-curve
        (control) points. If the number of control points is > 2, the
        segment is split into multiple bezier segments. This works
        like this:

        Let n be the number of control points (which is the number of
        arguments to this call minus 1). If n==2, a plain vanilla cubic
        bezier is drawn. If n==1, we fall back to a quadratic segment and
        if n==0 we draw a straight line. It gets interesting when n>2:
        n-1 PostScript-style cubic segments will be drawn as if it were
        one curve. See decomposeSuperBezierSegment().

        The conversion algorithm used for n>2 is inspired by NURB
        splines, and is conceptually equivalent to the TrueType "implied
        points" principle. See also decomposeQuadraticSegment().
        """
        raise NotImplementedError

    def qCurveTo(self, *points: Tuple[float, float]) -> None:
        """Draw a whole string of quadratic curve segments.

        The last point specified is on-curve, all others are off-curve
        points.

        This method implements TrueType-style curves, breaking up curves
        using 'implied points': between each two consequtive off-curve points,
        there is one implied point exactly in the middle between them. See
        also decomposeQuadraticSegment().

        The last argument (normally the on-curve point) may be None.
        This is to support contours that have NO on-curve points (a rarely
        seen feature of TrueType outlines).
        """
        raise NotImplementedError

    def closePath(self) -> None:
        """Close the current sub path. You must call either pen.closePath()
        or pen.endPath() after each sub path.
        """
        pass

    def endPath(self) -> None:
        """End the current sub path, but don't close it. You must call
        either pen.closePath() or pen.endPath() after each sub path.
        """
        pass

    def addComponent(
        self,
        glyphName: str,
        transformation: Tuple[float, float, float, float, float, float],
    ) -> None:
        """Add a sub glyph. The 'transformation' argument must be a 6-tuple
        containing an affine transformation, or a Transform object from the
        fontTools.misc.transform module. More precisely: it should be a
        sequence containing 6 numbers.
        """
        raise NotImplementedError

    def addVarComponent(
        self,
        glyphName: str,
        transformation: DecomposedTransform,
        location: Dict[str, float],
    ) -> None:
        """Add a VarComponent sub glyph. The 'transformation' argument
        must be a DecomposedTransform from the fontTools.misc.transform module,
        and the 'location' argument must be a dictionary mapping axis tags
        to their locations.
        """
        # GlyphSet decomposes for us
        raise AttributeError


class NullPen(AbstractPen):
    """A pen that does nothing."""

    def moveTo(self, pt):
        pass

    def lineTo(self, pt):
        pass

    def curveTo(self, *points):
        pass

    def qCurveTo(self, *points):
        pass

    def closePath(self):
        pass

    def endPath(self):
        pass

    def addComponent(self, glyphName, transformation):
        pass

    def addVarComponent(self, glyphName, transformation, location):
        pass


class LoggingPen(LogMixin, AbstractPen):
    """A pen with a ``log`` property (see fontTools.misc.loggingTools.LogMixin)"""

    pass


class MissingComponentError(KeyError):
    """Indicates a component pointing to a non-existent glyph in the glyphset."""


class DecomposingPen(LoggingPen):
    """Implements a 'addComponent' method that decomposes components
    (i.e. draws them onto self as simple contours).
    It can also be used as a mixin class (e.g. see ContourRecordingPen).

    You must override moveTo, lineTo, curveTo and qCurveTo. You may
    additionally override closePath, endPath and addComponent.

    By default a warning message is logged when a base glyph is missing;
    set the class variable ``skipMissingComponents`` to False if you want
    all instances of a sub-class to raise a :class:`MissingComponentError`
    exception by default.
    """

    skipMissingComponents = True
    # alias error for convenience
    MissingComponentError = MissingComponentError

    def __init__(
        self,
        glyphSet,
        *args,
        skipMissingComponents=None,
        reverseFlipped=False,
        **kwargs,
    ):
        """Takes a 'glyphSet' argument (dict), in which the glyphs that are referenced
        as components are looked up by their name.

        If the optional 'reverseFlipped' argument is True, components whose transformation
        matrix has a negative determinant will be decomposed with a reversed path direction
        to compensate for the flip.

        The optional 'skipMissingComponents' argument can be set to True/False to
        override the homonymous class attribute for a given pen instance.
        """
        super(DecomposingPen, self).__init__(*args, **kwargs)
        self.glyphSet = glyphSet
        self.skipMissingComponents = (
            self.__class__.skipMissingComponents
            if skipMissingComponents is None
            else skipMissingComponents
        )
        self.reverseFlipped = reverseFlipped

    def addComponent(self, glyphName, transformation):
        """Transform the points of the base glyph and draw it onto self."""
        from fontTools.pens.transformPen import TransformPen

        try:
            glyph = self.glyphSet[glyphName]
        except KeyError:
            if not self.skipMissingComponents:
                raise MissingComponentError(glyphName)
            self.log.warning("glyph '%s' is missing from glyphSet; skipped" % glyphName)
        else:
            pen = self
            if transformation != Identity:
                pen = TransformPen(pen, transformation)
            if self.reverseFlipped:
                # if the transformation has a negative determinant, it will
                # reverse the contour direction of the component
                a, b, c, d = transformation[:4]
                det = a * d - b * c
                if det < 0:
                    from fontTools.pens.reverseContourPen import ReverseContourPen

                    pen = ReverseContourPen(pen)
            glyph.draw(pen)

    def addVarComponent(self, glyphName, transformation, location):
        # GlyphSet decomposes for us
        raise AttributeError


class BasePen(DecomposingPen):
    """Base class for drawing pens. You must override _moveTo, _lineTo and
    _curveToOne. You may additionally override _closePath, _endPath,
    addComponent, addVarComponent, and/or _qCurveToOne. You should not
    override any other methods.
    """

    def __init__(self, glyphSet=None):
        super(BasePen, self).__init__(glyphSet)
        self.__currentPoint = None

    # must override

    def _moveTo(self, pt):
        raise NotImplementedError

    def _lineTo(self, pt):
        raise NotImplementedError

    def _curveToOne(self, pt1, pt2, pt3):
        raise NotImplementedError

    # may override

    def _closePath(self):
        pass

    def _endPath(self):
        pass

    def _qCurveToOne(self, pt1, pt2):
        """This method implements the basic quadratic curve type. The
        default implementation delegates the work to the cubic curve
        function. Optionally override with a native implementation.
        """
        pt0x, pt0y = self.__currentPoint
        pt1x, pt1y = pt1
        pt2x, pt2y = pt2
        mid1x = pt0x + 0.66666666666666667 * (pt1x - pt0x)
        mid1y = pt0y + 0.66666666666666667 * (pt1y - pt0y)
        mid2x = pt2x + 0.66666666666666667 * (pt1x - pt2x)
        mid2y = pt2y + 0.66666666666666667 * (pt1y - pt2y)
        self._curveToOne((mid1x, mid1y), (mid2x, mid2y), pt2)

    # don't override

    def _getCurrentPoint(self):
        """Return the current point. This is not part of the public
        interface, yet is useful for subclasses.
        """
        return self.__currentPoint

    def closePath(self):
        self._closePath()
        self.__currentPoint = None

    def endPath(self):
        self._endPath()
        self.__currentPoint = None

    def moveTo(self, pt):
        self._moveTo(pt)
        self.__currentPoint = pt

    def lineTo(self, pt):
        self._lineTo(pt)
        self.__currentPoint = pt

    def curveTo(self, *points):
        n = len(points) - 1  # 'n' is the number of control points
        assert n >= 0
        if n == 2:
            # The common case, we have exactly two BCP's, so this is a standard
            # cubic bezier. Even though decomposeSuperBezierSegment() handles
            # this case just fine, we special-case it anyway since it's so
            # common.
            self._curveToOne(*points)
            self.__currentPoint = points[-1]
        elif n > 2:
            # n is the number of control points; split curve into n-1 cubic
            # bezier segments. The algorithm used here is inspired by NURB
            # splines and the TrueType "implied point" principle, and ensures
            # the smoothest possible connection between two curve segments,
            # with no disruption in the curvature. It is practical since it
            # allows one to construct multiple bezier segments with a much
            # smaller amount of points.
            _curveToOne = self._curveToOne
            for pt1, pt2, pt3 in decomposeSuperBezierSegment(points):
                _curveToOne(pt1, pt2, pt3)
                self.__currentPoint = pt3
        elif n == 1:
            self.qCurveTo(*points)
        elif n == 0:
            self.lineTo(points[0])
        else:
            raise AssertionError("can't get there from here")

    def qCurveTo(self, *points):
        n = len(points) - 1  # 'n' is the number of control points
        assert n >= 0
        if points[-1] is None:
            # Special case for TrueType quadratics: it is possible to
            # define a contour with NO on-curve points. BasePen supports
            # this by allowing the final argument (the expected on-curve
            # point) to be None. We simulate the feature by making the implied
            # on-curve point between the last and the first off-curve points
            # explicit.
            x, y = points[-2]  # last off-curve point
            nx, ny = points[0]  # first off-curve point
            impliedStartPoint = (0.5 * (x + nx), 0.5 * (y + ny))
            self.__currentPoint = impliedStartPoint
            self._moveTo(impliedStartPoint)
            points = points[:-1] + (impliedStartPoint,)
        if n > 0:
            # Split the string of points into discrete quadratic curve
            # segments. Between any two consecutive off-curve points
            # there's an implied on-curve point exactly in the middle.
            # This is where the segment splits.
            _qCurveToOne = self._qCurveToOne
            for pt1, pt2 in decomposeQuadraticSegment(points):
                _qCurveToOne(pt1, pt2)
                self.__currentPoint = pt2
        else:
            self.lineTo(points[0])


def decomposeSuperBezierSegment(points):
    """Split the SuperBezier described by 'points' into a list of regular
    bezier segments. The 'points' argument must be a sequence with length
    3 or greater, containing (x, y) coordinates. The last point is the
    destination on-curve point, the rest of the points are off-curve points.
    The start point should not be supplied.

    This function returns a list of (pt1, pt2, pt3) tuples, which each
    specify a regular curveto-style bezier segment.
    """
    n = len(points) - 1
    assert n > 1
    bezierSegments = []
    pt1, pt2, pt3 = points[0], None, None
    for i in range(2, n + 1):
        # calculate points in between control points.
        nDivisions = min(i, 3, n - i + 2)
        for j in range(1, nDivisions):
            factor = j / nDivisions
            temp1 = points[i - 1]
            temp2 = points[i - 2]
            temp = (
                temp2[0] + factor * (temp1[0] - temp2[0]),
                temp2[1] + factor * (temp1[1] - temp2[1]),
            )
            if pt2 is None:
                pt2 = temp
            else:
                pt3 = (0.5 * (pt2[0] + temp[0]), 0.5 * (pt2[1] + temp[1]))
                bezierSegments.append((pt1, pt2, pt3))
                pt1, pt2, pt3 = temp, None, None
    bezierSegments.append((pt1, points[-2], points[-1]))
    return bezierSegments


def decomposeQuadraticSegment(points):
    """Split the quadratic curve segment described by 'points' into a list
    of "atomic" quadratic segments. The 'points' argument must be a sequence
    with length 2 or greater, containing (x, y) coordinates. The last point
    is the destination on-curve point, the rest of the points are off-curve
    points. The start point should not be supplied.

    This function returns a list of (pt1, pt2) tuples, which each specify a
    plain quadratic bezier segment.
    """
    n = len(points) - 1
    assert n > 0
    quadSegments = []
    for i in range(n - 1):
        x, y = points[i]
        nx, ny = points[i + 1]
        impliedPt = (0.5 * (x + nx), 0.5 * (y + ny))
        quadSegments.append((points[i], impliedPt))
    quadSegments.append((points[-2], points[-1]))
    return quadSegments


class _TestPen(BasePen):
    """Test class that prints PostScript to stdout."""

    def _moveTo(self, pt):
        print("%s %s moveto" % (pt[0], pt[1]))

    def _lineTo(self, pt):
        print("%s %s lineto" % (pt[0], pt[1]))

    def _curveToOne(self, bcp1, bcp2, pt):
        print(
            "%s %s %s %s %s %s curveto"
            % (bcp1[0], bcp1[1], bcp2[0], bcp2[1], pt[0], pt[1])
        )

    def _closePath(self):
        print("closepath")


if __name__ == "__main__":
    pen = _TestPen(None)
    pen.moveTo((0, 0))
    pen.lineTo((0, 100))
    pen.curveTo((50, 75), (60, 50), (50, 25), (0, 0))
    pen.closePath()

    pen = _TestPen(None)
    # testing the "no on-curve point" scenario
    pen.qCurveTo((0, 0), (0, 100), (100, 100), (100, 0), None)
    pen.closePath()
<|endoftext|># -*- coding: utf-8 -*-

"""Pen to rasterize paths with FreeType."""

__all__ = ["FreeTypePen"]

import os
import ctypes
import platform
import subprocess
import collections
import math

import freetype
from freetype.raw import FT_Outline_Get_Bitmap, FT_Outline_Get_BBox, FT_Outline_Get_CBox
from freetype.ft_types import FT_Pos
from freetype.ft_structs import FT_Vector, FT_BBox, FT_Bitmap, FT_Outline
from freetype.ft_enums import (
    FT_OUTLINE_NONE,
    FT_OUTLINE_EVEN_ODD_FILL,
    FT_PIXEL_MODE_GRAY,
    FT_CURVE_TAG_ON,
    FT_CURVE_TAG_CONIC,
    FT_CURVE_TAG_CUBIC,
)
from freetype.ft_errors import FT_Exception

from fontTools.pens.basePen import BasePen, PenError
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform

Contour = collections.namedtuple("Contour", ("points", "tags"))


class FreeTypePen(BasePen):
    """Pen to rasterize paths with FreeType. Requires `freetype-py` module.

    Constructs ``FT_Outline`` from the paths, and renders it within a bitmap
    buffer.

    For ``array()`` and ``show()``, `numpy` and `matplotlib` must be installed.
    For ``image()``, `Pillow` is required. Each module is lazily loaded when the
    corresponding method is called.

    Args:
        glyphSet: a dictionary of drawable glyph objects keyed by name
            used to resolve component references in composite glyphs.

    Examples:
        If `numpy` and `matplotlib` is available, the following code will
        show the glyph image of `fi` in a new window::

            from fontTools.ttLib import TTFont
            from fontTools.pens.freetypePen import FreeTypePen
            from fontTools.misc.transform import Offset
            pen = FreeTypePen(None)
            font = TTFont('SourceSansPro-Regular.otf')
            glyph = font.getGlyphSet()['fi']
            glyph.draw(pen)
            width, ascender, descender = glyph.width, font['OS/2'].usWinAscent, -font['OS/2'].usWinDescent
            height = ascender - descender
            pen.show(width=width, height=height, transform=Offset(0, -descender))

        Combining with `uharfbuzz`, you can typeset a chunk of glyphs in a pen::

            import uharfbuzz as hb
            from fontTools.pens.freetypePen import FreeTypePen
            from fontTools.pens.transformPen import TransformPen
            from fontTools.misc.transform import Offset

            en1, en2, ar, ja = 'Typesetting', 'Jeff', 'صف الحروف', 'たいぷせっと'
            for text, font_path, direction, typo_ascender, typo_descender, vhea_ascender, vhea_descender, contain, features in (
                (en1, 'NotoSans-Regular.ttf',       'ltr', 2189, -600, None, None, False, {"kern": True, "liga": True}),
                (en2, 'NotoSans-Regular.ttf',       'ltr', 2189, -600, None, None, True,  {"kern": True, "liga": True}),
                (ar,  'NotoSansArabic-Regular.ttf', 'rtl', 1374, -738, None, None, False, {"kern": True, "liga": True}),
                (ja,  'NotoSansJP-Regular.otf',     'ltr', 880,  -120, 500,  -500, False, {"palt": True, "kern": True}),
                (ja,  'NotoSansJP-Regular.otf',     'ttb', 880,  -120, 500,  -500, False, {"vert": True, "vpal": True, "vkrn": True})
            ):
                blob = hb.Blob.from_file_path(font_path)
                face = hb.Face(blob)
                font = hb.Font(face)
                buf = hb.Buffer()
                buf.direction = direction
                buf.add_str(text)
                buf.guess_segment_properties()
                hb.shape(font, buf, features)

                x, y = 0, 0
                pen = FreeTypePen(None)
                for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
                    gid = info.codepoint
                    transformed = TransformPen(pen, Offset(x + pos.x_offset, y + pos.y_offset))
                    font.draw_glyph_with_pen(gid, transformed)
                    x += pos.x_advance
                    y += pos.y_advance

                offset, width, height = None, None, None
                if direction in ('ltr', 'rtl'):
                    offset = (0, -typo_descender)
                    width  = x
                    height = typo_ascender - typo_descender
                else:
                    offset = (-vhea_descender, -y)
                    width  = vhea_ascender - vhea_descender
                    height = -y
                pen.show(width=width, height=height, transform=Offset(*offset), contain=contain)

        For Jupyter Notebook, the rendered image will be displayed in a cell if
        you replace ``show()`` with ``image()`` in the examples.
    """

    def __init__(self, glyphSet):
        BasePen.__init__(self, glyphSet)
        self.contours = []

    def outline(self, transform=None, evenOdd=False):
        """Converts the current contours to ``FT_Outline``.

        Args:
            transform: An optional 6-tuple containing an affine transformation,
                or a ``Transform`` object from the ``fontTools.misc.transform``
                module.
            evenOdd: Pass ``True`` for even-odd fill instead of non-zero.
        """
        transform = transform or Transform()
        if not hasattr(transform, "transformPoint"):
            transform = Transform(*transform)
        n_contours = len(self.contours)
        n_points = sum((len(contour.points) for contour in self.contours))
        points = []
        for contour in self.contours:
            for point in contour.points:
                point = transform.transformPoint(point)
                points.append(
                    FT_Vector(
                        FT_Pos(otRound(point[0] * 64)), FT_Pos(otRound(point[1] * 64))
                    )
                )
        tags = []
        for contour in self.contours:
            for tag in contour.tags:
                tags.append(tag)
        contours = []
        contours_sum = 0
        for contour in self.contours:
            contours_sum += len(contour.points)
            contours.append(contours_sum - 1)
        flags = FT_OUTLINE_EVEN_ODD_FILL if evenOdd else FT_OUTLINE_NONE
        return FT_Outline(
            (ctypes.c_short)(n_contours),
            (ctypes.c_short)(n_points),
            (FT_Vector * n_points)(*points),
            (ctypes.c_ubyte * n_points)(*tags),
            (ctypes.c_short * n_contours)(*contours),
            (ctypes.c_int)(flags),
        )

    def buffer(
        self, width=None, height=None, transform=None, contain=False, evenOdd=False
    ):
        """Renders the current contours within a bitmap buffer.

        Args:
            width: Image width of the bitmap in pixels. If omitted, it
                automatically fits to the bounding box of the contours.
            height: Image height of the bitmap in pixels. If omitted, it
                automatically fits to the bounding box of the contours.
            transform: An optional 6-tuple containing an affine transformation,
                or a ``Transform`` object from the ``fontTools.misc.transform``
                module. The bitmap size is not affected by this matrix.
            contain: If ``True``, the image size will be automatically expanded
                so that it fits to the bounding box of the paths. Useful for
                rendering glyphs with negative sidebearings without clipping.
            evenOdd: Pass ``True`` for even-odd fill instead of non-zero.

        Returns:
            A tuple of ``(buffer, size)``, where ``buffer`` is a ``bytes``
            object of the resulted bitmap and ``size`` is a 2-tuple of its
            dimension.

        Notes:
            The image size should always be given explicitly if you need to get
            a proper glyph image. When ``width`` and ``height`` are omitted, it
            forcifully fits to the bounding box and the side bearings get
            cropped. If you pass ``0`` to both ``width`` and ``height`` and set
            ``contain`` to ``True``, it expands to the bounding box while
            maintaining the origin of the contours, meaning that LSB will be
            maintained but RSB won’t. The difference between the two becomes
            more obvious when rotate or skew transformation is applied.

        Example:
            .. code-block:: pycon

                >>>
                >> pen = FreeTypePen(None)
                >> glyph.draw(pen)
                >> buf, size = pen.buffer(width=500, height=1000)
                >> type(buf), len(buf), size
                (<class 'bytes'>, 500000, (500, 1000))
        """
        transform = transform or Transform()
        if not hasattr(transform, "transformPoint"):
            transform = Transform(*transform)
        contain_x, contain_y = contain or width is None, contain or height is None
        if contain_x or contain_y:
            dx, dy = transform.dx, transform.dy
            bbox = self.bbox
            p1, p2, p3, p4 = (
                transform.transformPoint((bbox[0], bbox[1])),
                transform.transformPoint((bbox[2], bbox[1])),
                transform.transformPoint((bbox[0], bbox[3])),
                transform.transformPoint((bbox[2], bbox[3])),
            )
            px, py = (p1[0], p2[0], p3[0], p4[0]), (p1[1], p2[1], p3[1], p4[1])
            if contain_x:
                if width is None:
                    dx = dx - min(*px)
                    width = max(*px) - min(*px)
                else:
                    dx = dx - min(min(*px), 0.0)
                    width = max(width, max(*px) - min(min(*px), 0.0))
            if contain_y:
                if height is None:
                    dy = dy - min(*py)
                    height = max(*py) - min(*py)
                else:
                    dy = dy - min(min(*py), 0.0)
                    height = max(height, max(*py) - min(min(*py), 0.0))
            transform = Transform(*transform[:4], dx, dy)
        width, height = math.ceil(width), math.ceil(height)
        buf = ctypes.create_string_buffer(width * height)
        bitmap = FT_Bitmap(
            (ctypes.c_int)(height),
            (ctypes.c_int)(width),
            (ctypes.c_int)(width),
            (ctypes.POINTER(ctypes.c_ubyte))(buf),
            (ctypes.c_short)(256),
            (ctypes.c_ubyte)(FT_PIXEL_MODE_GRAY),
            (ctypes.c_char)(0),
            (ctypes.c_void_p)(None),
        )
        outline = self.outline(transform=transform, evenOdd=evenOdd)
        err = FT_Outline_Get_Bitmap(
            freetype.get_handle(), ctypes.byref(outline), ctypes.byref(bitmap)
        )
        if err != 0:
            raise FT_Exception(err)
        return buf.raw, (width, height)

    def array(
        self, width=None, height=None, transform=None, contain=False, evenOdd=False
    ):
        """Returns the rendered contours as a numpy array. Requires `numpy`.

        Args:
            width: Image width of the bitmap in pixels. If omitted, it
                automatically fits to the bounding box of the contours.
            height: Image height of the bitmap in pixels. If omitted, it
                automatically fits to the bounding box of the contours.
            transform: An optional 6-tuple containing an affine transformation,
                or a ``Transform`` object from the ``fontTools.misc.transform``
                module. The bitmap size is not affected by this matrix.
            contain: If ``True``, the image size will be automatically expanded
                so that it fits to the bounding box of the paths. Useful for
                rendering glyphs with negative sidebearings without clipping.
            evenOdd: Pass ``True`` for even-odd fill instead of non-zero.

        Returns:
            A ``numpy.ndarray`` object with a shape of ``(height, width)``.
            Each element takes a value in the range of ``[0.0, 1.0]``.

        Notes:
            The image size should always be given explicitly if you need to get
            a proper glyph image. When ``width`` and ``height`` are omitted, it
            forcifully fits to the bounding box and the side bearings get
            cropped. If you pass ``0`` to both ``width`` and ``height`` and set
            ``contain`` to ``True``, it expands to the bounding box while
            maintaining the origin of the contours, meaning that LSB will be
            maintained but RSB won’t. The difference between the two becomes
            more obvious when rotate or skew transformation is applied.

        Example:
            .. code-block:: pycon

                >>>
                >> pen = FreeTypePen(None)
                >> glyph.draw(pen)
                >> arr = pen.array(width=500, height=1000)
                >> type(a), a.shape
                (<class 'numpy.ndarray'>, (1000, 500))
        """

        import numpy as np

        buf, size = self.buffer(
            width=width,
            height=height,
            transform=transform,
            contain=contain,
            evenOdd=evenOdd,
        )
        return np.frombuffer(buf, "B").reshape((size[1], size[0])) / 255.0

    def show(
        self, width=None, height=None, transform=None, contain=False, evenOdd=False
    ):
        """Plots the rendered contours with `pyplot`. Requires `numpy` and
        `matplotlib`.

        Args:
            width: Image width of the bitmap in pixels. If omitted, it
                automatically fits to the bounding box of the contours.
            height: Image height of the bitmap in pixels. If omitted, it
                automatically fits to the bounding box of the contours.
            transform: An optional 6-tuple containing an affine transformation,
                or a ``Transform`` object from the ``fontTools.misc.transform``
                module. The bitmap size is not affected by this matrix.
            contain: If ``True``, the image size will be automatically expanded
                so that it fits to the bounding box of the paths. Useful for
                rendering glyphs with negative sidebearings without clipping.
            evenOdd: Pass ``True`` for even-odd fill instead of non-zero.

        Notes:
            The image size should always be given explicitly if you need to get
            a proper glyph image. When ``width`` and ``height`` are omitted, it
            forcifully fits to the bounding box and the side bearings get
            cropped. If you pass ``0`` to both ``width`` and ``height`` and set
            ``contain`` to ``True``, it expands to the bounding box while
            maintaining the origin of the contours, meaning that LSB will be
            maintained but RSB won’t. The difference between the two becomes
            more obvious when rotate or skew transformation is applied.

        Example:
            .. code-block:: pycon

                >>>
                >> pen = FreeTypePen(None)
                >> glyph.draw(pen)
                >> pen.show(width=500, height=1000)
        """
        from matplotlib import pyplot as plt

        a = self.array(
            width=width,
            height=height,
            transform=transform,
            contain=contain,
            evenOdd=evenOdd,
        )
        plt.imshow(a, cmap="gray_r", vmin=0, vmax=1)
        plt.show()

    def image(
        self, width=None, height=None, transform=None, contain=False, evenOdd=False
    ):
        """Returns the rendered contours as a PIL image. Requires `Pillow`.
        Can be used to display a glyph image in Jupyter Notebook.

        Args:
            width: Image width of the bitmap in pixels. If omitted, it
                automatically fits to the bounding box of the contours.
            height: Image height of the bitmap in pixels. If omitted, it
                automatically fits to the bounding box of the contours.
            transform: An optional 6-tuple containing an affine transformation,
                or a ``Transform`` object from the ``fontTools.misc.transform``
                module. The bitmap size is not affected by this matrix.
            contain: If ``True``, the image size will be automatically expanded
                so that it fits to the bounding box of the paths. Useful for
                rendering glyphs with negative sidebearings without clipping.
            evenOdd: Pass ``True`` for even-odd fill instead of non-zero.

        Returns:
            A ``PIL.image`` object. The image is filled in black with alpha
            channel obtained from the rendered bitmap.

        Notes:
            The image size should always be given explicitly if you need to get
            a proper glyph image. When ``width`` and ``height`` are omitted, it
            forcifully fits to the bounding box and the side bearings get
            cropped. If you pass ``0`` to both ``width`` and ``height`` and set
            ``contain`` to ``True``, it expands to the bounding box while
            maintaining the origin of the contours, meaning that LSB will be
            maintained but RSB won’t. The difference between the two becomes
            more obvious when rotate or skew transformation is applied.

        Example:
            .. code-block:: pycon

                >>>
                >> pen = FreeTypePen(None)
                >> glyph.draw(pen)
                >> img = pen.image(width=500, height=1000)
                >> type(img), img.size
                (<class 'PIL.Image.Image'>, (500, 1000))
        """
        from PIL import Image

        buf, size = self.buffer(
            width=width,
            height=height,
            transform=transform,
            contain=contain,
            evenOdd=evenOdd,
        )
        img = Image.new("L", size, 0)
        img.putalpha(Image.frombuffer("L", size, buf))
        return img

    @property
    def bbox(self):
        """Computes the exact bounding box of an outline.

        Returns:
            A tuple of ``(xMin, yMin, xMax, yMax)``.
        """
        bbox = FT_BBox()
        outline = self.outline()
        FT_Outline_Get_BBox(ctypes.byref(outline), ctypes.byref(bbox))
        return (bbox.xMin / 64.0, bbox.yMin / 64.0, bbox.xMax / 64.0, bbox.yMax / 64.0)

    @property
    def cbox(self):
        """Returns an outline's ‘control box’.

        Returns:
            A tuple of ``(xMin, yMin, xMax, yMax)``.
        """
        cbox = FT_BBox()
        outline = self.outline()
        FT_Outline_Get_CBox(ctypes.byref(outline), ctypes.byref(cbox))
        return (cbox.xMin / 64.0, cbox.yMin / 64.0, cbox.xMax / 64.0, cbox.yMax / 64.0)

    def _moveTo(self, pt):
        contour = Contour([], [])
        self.contours.append(contour)
        contour.points.append(pt)
        contour.tags.append(FT_CURVE_TAG_ON)

    def _lineTo(self, pt):
        if not (self.contours and len(self.contours[-1].points) > 0):
            raise PenError("Contour missing required initial moveTo")
        contour = self.contours[-1]
        contour.points.append(pt)
        contour.tags.append(FT_CURVE_TAG_ON)

    def _curveToOne(self, p1, p2, p3):
        if not (self.contours and len(self.contours[-1].points) > 0):
            raise PenError("Contour missing required initial moveTo")
        t1, t2, t3 = FT_CURVE_TAG_CUBIC, FT_CURVE_TAG_CUBIC, FT_CURVE_TAG_ON
        contour = self.contours[-1]
        for p, t in ((p1, t1), (p2, t2), (p3, t3)):
            contour.points.append(p)
            contour.tags.append(t)

    def _qCurveToOne(self, p1, p2):
        if not (self.contours and len(self.contours[-1].points) > 0):
            raise PenError("Contour missing required initial moveTo")
        t1, t2 = FT_CURVE_TAG_CONIC, FT_CURVE_TAG_ON
        contour = self.contours[-1]
        for p, t in ((p1, t1), (p2, t2)):
            contour.points.append(p)
            contour.tags.append(t)
<|endoftext|>from fontTools.misc.arrayTools import updateBounds, pointInRect, unionRect
from fontTools.misc.bezierTools import calcCubicBounds, calcQuadraticBounds
from fontTools.pens.basePen import BasePen

__all__ = ["BoundsPen", "ControlBoundsPen"]


class ControlBoundsPen(BasePen):
    """Pen to calculate the "control bounds" of a shape. This is the
    bounding box of all control points, so may be larger than the
    actual bounding box if there are curves that don't have points
    on their extremes.

    When the shape has been drawn, the bounds are available as the
    ``bounds`` attribute of the pen object. It's a 4-tuple::

            (xMin, yMin, xMax, yMax).

    If ``ignoreSinglePoints`` is True, single points are ignored.
    """

    def __init__(self, glyphSet, ignoreSinglePoints=False):
        BasePen.__init__(self, glyphSet)
        self.ignoreSinglePoints = ignoreSinglePoints
        self.init()

    def init(self):
        self.bounds = None
        self._start = None

    def _moveTo(self, pt):
        self._start = pt
        if not self.ignoreSinglePoints:
            self._addMoveTo()

    def _addMoveTo(self):
        if self._start is None:
            return
        bounds = self.bounds
        if bounds:
            self.bounds = updateBounds(bounds, self._start)
        else:
            x, y = self._start
            self.bounds = (x, y, x, y)
        self._start = None

    def _lineTo(self, pt):
        self._addMoveTo()
        self.bounds = updateBounds(self.bounds, pt)

    def _curveToOne(self, bcp1, bcp2, pt):
        self._addMoveTo()
        bounds = self.bounds
        bounds = updateBounds(bounds, bcp1)
        bounds = updateBounds(bounds, bcp2)
        bounds = updateBounds(bounds, pt)
        self.bounds = bounds

    def _qCurveToOne(self, bcp, pt):
        self._addMoveTo()
        bounds = self.bounds
        bounds = updateBounds(bounds, bcp)
        bounds = updateBounds(bounds, pt)
        self.bounds = bounds


class BoundsPen(ControlBoundsPen):
    """Pen to calculate the bounds of a shape. It calculates the
    correct bounds even when the shape contains curves that don't
    have points on their extremes. This is somewhat slower to compute
    than the "control bounds".

    When the shape has been drawn, the bounds are available as the
    ``bounds`` attribute of the pen object. It's a 4-tuple::

            (xMin, yMin, xMax, yMax)
    """

    def _curveToOne(self, bcp1, bcp2, pt):
        self._addMoveTo()
        bounds = self.bounds
        bounds = updateBounds(bounds, pt)
        if not pointInRect(bcp1, bounds) or not pointInRect(bcp2, bounds):
            bounds = unionRect(
                bounds, calcCubicBounds(self._getCurrentPoint(), bcp1, bcp2, pt)
            )
        self.bounds = bounds

    def _qCurveToOne(self, bcp, pt):
        self._addMoveTo()
        bounds = self.bounds
        bounds = updateBounds(bounds, pt)
        if not pointInRect(bcp, bounds):
            bounds = unionRect(
                bounds, calcQuadraticBounds(self._getCurrentPoint(), bcp, pt)
            )
        self.bounds = bounds
<|endoftext|>from fontTools.pens.basePen import BasePen

__all__ = ["CocoaPen"]


class CocoaPen(BasePen):
    def __init__(self, glyphSet, path=None):
        BasePen.__init__(self, glyphSet)
        if path is None:
            from AppKit import NSBezierPath

            path = NSBezierPath.bezierPath()
        self.path = path

    def _moveTo(self, p):
        self.path.moveToPoint_(p)

    def _lineTo(self, p):
        self.path.lineToPoint_(p)

    def _curveToOne(self, p1, p2, p3):
        self.path.curveToPoint_controlPoint1_controlPoint2_(p3, p1, p2)

    def _closePath(self):
        self.path.closePath()
<|endoftext|>"""Pen calculating area, center of mass, variance and standard-deviation,
covariance and correlation, and slant, of glyph shapes."""

from math import sqrt, degrees, atan
from fontTools.pens.basePen import BasePen, OpenContourError
from fontTools.pens.momentsPen import MomentsPen

__all__ = ["StatisticsPen", "StatisticsControlPen"]


class StatisticsBase:
    def __init__(self):
        self._zero()

    def _zero(self):
        self.area = 0
        self.meanX = 0
        self.meanY = 0
        self.varianceX = 0
        self.varianceY = 0
        self.stddevX = 0
        self.stddevY = 0
        self.covariance = 0
        self.correlation = 0
        self.slant = 0

    def _update(self):
        # XXX The variance formulas should never produce a negative value,
        # but due to reasons I don't understand, both of our pens do.
        # So we take the absolute value here.
        self.varianceX = abs(self.varianceX)
        self.varianceY = abs(self.varianceY)

        self.stddevX = stddevX = sqrt(self.varianceX)
        self.stddevY = stddevY = sqrt(self.varianceY)

        # Correlation(X,Y) = Covariance(X,Y) / ( stddev(X) * stddev(Y) )
        # https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
        if stddevX * stddevY == 0:
            correlation = float("NaN")
        else:
            # XXX The above formula should never produce a value outside
            # the range [-1, 1], but due to reasons I don't understand,
            # (probably the same issue as above), it does. So we clamp.
            correlation = self.covariance / (stddevX * stddevY)
            correlation = max(-1, min(1, correlation))
        self.correlation = correlation if abs(correlation) > 1e-3 else 0

        slant = (
            self.covariance / self.varianceY if self.varianceY != 0 else float("NaN")
        )
        self.slant = slant if abs(slant) > 1e-3 else 0


class StatisticsPen(StatisticsBase, MomentsPen):
    """Pen calculating area, center of mass, variance and
    standard-deviation, covariance and correlation, and slant,
    of glyph shapes.

    Note that if the glyph shape is self-intersecting, the values
    are not correct (but well-defined). Moreover, area will be
    negative if contour directions are clockwise."""

    def __init__(self, glyphset=None):
        MomentsPen.__init__(self, glyphset=glyphset)
        StatisticsBase.__init__(self)

    def _closePath(self):
        MomentsPen._closePath(self)
        self._update()

    def _update(self):
        area = self.area
        if not area:
            self._zero()
            return

        # Center of mass
        # https://en.wikipedia.org/wiki/Center_of_mass#A_continuous_volume
        self.meanX = meanX = self.momentX / area
        self.meanY = meanY = self.momentY / area

        # Var(X) = E[X^2] - E[X]^2
        self.varianceX = self.momentXX / area - meanX * meanX
        self.varianceY = self.momentYY / area - meanY * meanY

        # Covariance(X,Y) = (E[X.Y] - E[X]E[Y])
        self.covariance = self.momentXY / area - meanX * meanY

        StatisticsBase._update(self)


class StatisticsControlPen(StatisticsBase, BasePen):
    """Pen calculating area, center of mass, variance and
    standard-deviation, covariance and correlation, and slant,
    of glyph shapes, using the control polygon only.

    Note that if the glyph shape is self-intersecting, the values
    are not correct (but well-defined). Moreover, area will be
    negative if contour directions are clockwise."""

    def __init__(self, glyphset=None):
        BasePen.__init__(self, glyphset)
        StatisticsBase.__init__(self)
        self._nodes = []

    def _moveTo(self, pt):
        self._nodes.append(complex(*pt))
        self._startPoint = pt

    def _lineTo(self, pt):
        self._nodes.append(complex(*pt))

    def _qCurveToOne(self, pt1, pt2):
        for pt in (pt1, pt2):
            self._nodes.append(complex(*pt))

    def _curveToOne(self, pt1, pt2, pt3):
        for pt in (pt1, pt2, pt3):
            self._nodes.append(complex(*pt))

    def _closePath(self):
        p0 = self._getCurrentPoint()
        if p0 != self._startPoint:
            self._lineTo(self._startPoint)
        self._update()

    def _endPath(self):
        p0 = self._getCurrentPoint()
        if p0 != self._startPoint:
            raise OpenContourError("Glyph statistics not defined on open contours.")
        self._update()

    def _update(self):
        nodes = self._nodes
        n = len(nodes)

        # Triangle formula
        self.area = (
            sum(
                (p0.real * p1.imag - p1.real * p0.imag)
                for p0, p1 in zip(nodes, nodes[1:] + nodes[:1])
            )
            / 2
        )

        # Center of mass
        # https://en.wikipedia.org/wiki/Center_of_mass#A_system_of_particles
        sumNodes = sum(nodes)
        self.meanX = meanX = sumNodes.real / n
        self.meanY = meanY = sumNodes.imag / n

        if n > 1:
            # Var(X) = (sum[X^2] - sum[X]^2 / n) / (n - 1)
            # https://www.statisticshowto.com/probability-and-statistics/descriptive-statistics/sample-variance/
            self.varianceX = varianceX = (
                sum(p.real * p.real for p in nodes)
                - (sumNodes.real * sumNodes.real) / n
            ) / (n - 1)
            self.varianceY = varianceY = (
                sum(p.imag * p.imag for p in nodes)
                - (sumNodes.imag * sumNodes.imag) / n
            ) / (n - 1)

            # Covariance(X,Y) = (sum[X.Y] - sum[X].sum[Y] / n) / (n - 1)
            self.covariance = covariance = (
                sum(p.real * p.imag for p in nodes)
                - (sumNodes.real * sumNodes.imag) / n
            ) / (n - 1)
        else:
            self.varianceX = varianceX = 0
            self.varianceY = varianceY = 0
            self.covariance = covariance = 0

        StatisticsBase._update(self)


def _test(glyphset, upem, glyphs, quiet=False, *, control=False):
    from fontTools.pens.transformPen import TransformPen
    from fontTools.misc.transform import Scale

    wght_sum = 0
    wght_sum_perceptual = 0
    wdth_sum = 0
    slnt_sum = 0
    slnt_sum_perceptual = 0
    for glyph_name in glyphs:
        glyph = glyphset[glyph_name]
        if control:
            pen = StatisticsControlPen(glyphset=glyphset)
        else:
            pen = StatisticsPen(glyphset=glyphset)
        transformer = TransformPen(pen, Scale(1.0 / upem))
        glyph.draw(transformer)

        area = abs(pen.area)
        width = glyph.width
        wght_sum += area
        wght_sum_perceptual += pen.area * width
        wdth_sum += width
        slnt_sum += pen.slant
        slnt_sum_perceptual += pen.slant * width

        if quiet:
            continue

        print()
        print("glyph:", glyph_name)

        for item in [
            "area",
            "momentX",
            "momentY",
            "momentXX",
            "momentYY",
            "momentXY",
            "meanX",
            "meanY",
            "varianceX",
            "varianceY",
            "stddevX",
            "stddevY",
            "covariance",
            "correlation",
            "slant",
        ]:
            print("%s: %g" % (item, getattr(pen, item)))

    if not quiet:
        print()
        print("font:")

    print("weight: %g" % (wght_sum * upem / wdth_sum))
    print("weight (perceptual): %g" % (wght_sum_perceptual / wdth_sum))
    print("width:  %g" % (wdth_sum / upem / len(glyphs)))
    slant = slnt_sum / len(glyphs)
    print("slant:  %g" % slant)
    print("slant angle:  %g" % -degrees(atan(slant)))
    slant_perceptual = slnt_sum_perceptual / wdth_sum
    print("slant (perceptual):  %g" % slant_perceptual)
    print("slant (perceptual) angle:  %g" % -degrees(atan(slant_perceptual)))


def main(args):
    """Report font glyph shape geometricsl statistics"""

    if args is None:
        import sys

        args = sys.argv[1:]

    import argparse

    parser = argparse.ArgumentParser(
        "fonttools pens.statisticsPen",
        description="Report font glyph shape geometricsl statistics",
    )
    parser.add_argument("font", metavar="font.ttf", help="Font file.")
    parser.add_argument("glyphs", metavar="glyph-name", help="Glyph names.", nargs="*")
    parser.add_argument(
        "-y",
        metavar="<number>",
        help="Face index into a collection to open. Zero based.",
    )
    parser.add_argument(
        "-c",
        "--control",
        action="store_true",
        help="Use the control-box pen instead of the Green therem.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only report font-wide statistics."
    )
    parser.add_argument(
        "--variations",
        metavar="AXIS=LOC",
        default="",
        help="List of space separated locations. A location consist in "
        "the name of a variation axis, followed by '=' and a number. E.g.: "
        "wght=700 wdth=80. The default is the location of the base master.",
    )

    options = parser.parse_args(args)

    glyphs = options.glyphs
    fontNumber = int(options.y) if options.y is not None else 0

    location = {}
    for tag_v in options.variations.split():
        fields = tag_v.split("=")
        tag = fields[0].strip()
        v = int(fields[1])
        location[tag] = v

    from fontTools.ttLib import TTFont

    font = TTFont(options.font, fontNumber=fontNumber)
    if not glyphs:
        glyphs = font.getGlyphOrder()
    _test(
        font.getGlyphSet(location=location),
        font["head"].unitsPerEm,
        glyphs,
        quiet=options.quiet,
        control=options.control,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
<|endoftext|>from array import array
from typing import Any, Callable, Dict, Optional, Tuple
from fontTools.misc.fixedTools import MAX_F2DOT14, floatToFixedToFloat
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.pointPen import AbstractPointPen
from fontTools.misc.roundTools import otRound
from fontTools.pens.basePen import LoggingPen, PenError
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import flagOnCurve, flagCubic
from fontTools.ttLib.tables._g_l_y_f import Glyph
from fontTools.ttLib.tables._g_l_y_f import GlyphComponent
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.ttLib.tables._g_l_y_f import dropImpliedOnCurvePoints
import math

__all__ = ["TTGlyphPen", "TTGlyphPointPen"]


class _TTGlyphBasePen:
    def __init__(
        self,
        glyphSet: Optional[Dict[str, Any]],
        handleOverflowingTransforms: bool = True,
    ) -> None:
        """
        Construct a new pen.

        Args:
            glyphSet (Dict[str, Any]): A glyphset object, used to resolve components.
            handleOverflowingTransforms (bool): See below.

        If ``handleOverflowingTransforms`` is True, the components' transform values
        are checked that they don't overflow the limits of a F2Dot14 number:
        -2.0 <= v < +2.0. If any transform value exceeds these, the composite
        glyph is decomposed.

        An exception to this rule is done for values that are very close to +2.0
        (both for consistency with the -2.0 case, and for the relative frequency
        these occur in real fonts). When almost +2.0 values occur (and all other
        values are within the range -2.0 <= x <= +2.0), they are clamped to the
        maximum positive value that can still be encoded as an F2Dot14: i.e.
        1.99993896484375.

        If False, no check is done and all components are translated unmodified
        into the glyf table, followed by an inevitable ``struct.error`` once an
        attempt is made to compile them.

        If both contours and components are present in a glyph, the components
        are decomposed.
        """
        self.glyphSet = glyphSet
        self.handleOverflowingTransforms = handleOverflowingTransforms
        self.init()

    def _decompose(
        self,
        glyphName: str,
        transformation: Tuple[float, float, float, float, float, float],
    ):
        tpen = self.transformPen(self, transformation)
        getattr(self.glyphSet[glyphName], self.drawMethod)(tpen)

    def _isClosed(self):
        """
        Check if the current path is closed.
        """
        raise NotImplementedError

    def init(self) -> None:
        self.points = []
        self.endPts = []
        self.types = []
        self.components = []

    def addComponent(
        self,
        baseGlyphName: str,
        transformation: Tuple[float, float, float, float, float, float],
        identifier: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add a sub glyph.
        """
        self.components.append((baseGlyphName, transformation))

    def _buildComponents(self, componentFlags):
        if self.handleOverflowingTransforms:
            # we can't encode transform values > 2 or < -2 in F2Dot14,
            # so we must decompose the glyph if any transform exceeds these
            overflowing = any(
                s > 2 or s < -2
                for (glyphName, transformation) in self.components
                for s in transformation[:4]
            )
        components = []
        for glyphName, transformation in self.components:
            if glyphName not in self.glyphSet:
                self.log.warning(f"skipped non-existing component '{glyphName}'")
                continue
            if self.points or (self.handleOverflowingTransforms and overflowing):
                # can't have both coordinates and components, so decompose
                self._decompose(glyphName, transformation)
                continue

            component = GlyphComponent()
            component.glyphName = glyphName
            component.x, component.y = (otRound(v) for v in transformation[4:])
            # quantize floats to F2Dot14 so we get same values as when decompiled
            # from a binary glyf table
            transformation = tuple(
                floatToFixedToFloat(v, 14) for v in transformation[:4]
            )
            if transformation != (1, 0, 0, 1):
                if self.handleOverflowingTransforms and any(
                    MAX_F2DOT14 < s <= 2 for s in transformation
                ):
                    # clamp values ~= +2.0 so we can keep the component
                    transformation = tuple(
                        MAX_F2DOT14 if MAX_F2DOT14 < s <= 2 else s
                        for s in transformation
                    )
                component.transform = (transformation[:2], transformation[2:])
            component.flags = componentFlags
            components.append(component)
        return components

    def glyph(
        self,
        componentFlags: int = 0x04,
        dropImpliedOnCurves: bool = False,
        *,
        round: Callable[[float], int] = otRound,
    ) -> Glyph:
        """
        Returns a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.

        Args:
            componentFlags: Flags to use for component glyphs. (default: 0x04)

            dropImpliedOnCurves: Whether to remove implied-oncurve points. (default: False)
        """
        if not self._isClosed():
            raise PenError("Didn't close last contour.")
        components = self._buildComponents(componentFlags)

        glyph = Glyph()
        glyph.coordinates = GlyphCoordinates(self.points)
        glyph.endPtsOfContours = self.endPts
        glyph.flags = array("B", self.types)
        self.init()

        if components:
            # If both components and contours were present, they have by now
            # been decomposed by _buildComponents.
            glyph.components = components
            glyph.numberOfContours = -1
        else:
            glyph.numberOfContours = len(glyph.endPtsOfContours)
            glyph.program = ttProgram.Program()
            glyph.program.fromBytecode(b"")
            if dropImpliedOnCurves:
                dropImpliedOnCurvePoints(glyph)
            glyph.coordinates.toInt(round=round)

        return glyph


class TTGlyphPen(_TTGlyphBasePen, LoggingPen):
    """
    Pen used for drawing to a TrueType glyph.

    This pen can be used to construct or modify glyphs in a TrueType format
    font. After using the pen to draw, use the ``.glyph()`` method to retrieve
    a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.
    """

    drawMethod = "draw"
    transformPen = TransformPen

    def __init__(
        self,
        glyphSet: Optional[Dict[str, Any]] = None,
        handleOverflowingTransforms: bool = True,
        outputImpliedClosingLine: bool = False,
    ) -> None:
        super().__init__(glyphSet, handleOverflowingTransforms)
        self.outputImpliedClosingLine = outputImpliedClosingLine

    def _addPoint(self, pt: Tuple[float, float], tp: int) -> None:
        self.points.append(pt)
        self.types.append(tp)

    def _popPoint(self) -> None:
        self.points.pop()
        self.types.pop()

    def _isClosed(self) -> bool:
        return (not self.points) or (
            self.endPts and self.endPts[-1] == len(self.points) - 1
        )

    def lineTo(self, pt: Tuple[float, float]) -> None:
        self._addPoint(pt, flagOnCurve)

    def moveTo(self, pt: Tuple[float, float]) -> None:
        if not self._isClosed():
            raise PenError('"move"-type point must begin a new contour.')
        self._addPoint(pt, flagOnCurve)

    def curveTo(self, *points) -> None:
        assert len(points) % 2 == 1
        for pt in points[:-1]:
            self._addPoint(pt, flagCubic)

        # last point is None if there are no on-curve points
        if points[-1] is not None:
            self._addPoint(points[-1], 1)

    def qCurveTo(self, *points) -> None:
        assert len(points) >= 1
        for pt in points[:-1]:
            self._addPoint(pt, 0)

        # last point is None if there are no on-curve points
        if points[-1] is not None:
            self._addPoint(points[-1], 1)

    def closePath(self) -> None:
        endPt = len(self.points) - 1

        # ignore anchors (one-point paths)
        if endPt == 0 or (self.endPts and endPt == self.endPts[-1] + 1):
            self._popPoint()
            return

        if not self.outputImpliedClosingLine:
            # if first and last point on this path are the same, remove last
            startPt = 0
            if self.endPts:
                startPt = self.endPts[-1] + 1
            if self.points[startPt] == self.points[endPt]:
                self._popPoint()
                endPt -= 1

        self.endPts.append(endPt)

    def endPath(self) -> None:
        # TrueType contours are always "closed"
        self.closePath()


class TTGlyphPointPen(_TTGlyphBasePen, LogMixin, AbstractPointPen):
    """
    Point pen used for drawing to a TrueType glyph.

    This pen can be used to construct or modify glyphs in a TrueType format
    font. After using the pen to draw, use the ``.glyph()`` method to retrieve
    a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.
    """

    drawMethod = "drawPoints"
    transformPen = TransformPointPen

    def init(self) -> None:
        super().init()
        self._currentContourStartIndex = None

    def _isClosed(self) -> bool:
        return self._currentContourStartIndex is None

    def beginPath(self, identifier: Optional[str] = None, **kwargs: Any) -> None:
        """
        Start a new sub path.
        """
        if not self._isClosed():
            raise PenError("Didn't close previous contour.")
        self._currentContourStartIndex = len(self.points)

    def endPath(self) -> None:
        """
        End the current sub path.
        """
        # TrueType contours are always "closed"
        if self._isClosed():
            raise PenError("Contour is already closed.")
        if self._currentContourStartIndex == len(self.points):
            # ignore empty contours
            self._currentContourStartIndex = None
            return

        contourStart = self.endPts[-1] + 1 if self.endPts else 0
        self.endPts.append(len(self.points) - 1)
        self._currentContourStartIndex = None

        # Resolve types for any cubic segments
        flags = self.types
        for i in range(contourStart, len(flags)):
            if flags[i] == "curve":
                j = i - 1
                if j < contourStart:
                    j = len(flags) - 1
                while flags[j] == 0:
                    flags[j] = flagCubic
                    j -= 1
                flags[i] = flagOnCurve

    def addPoint(
        self,
        pt: Tuple[float, float],
        segmentType: Optional[str] = None,
        smooth: bool = False,
        name: Optional[str] = None,
        identifier: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add a point to the current sub path.
        """
        if self._isClosed():
            raise PenError("Can't add a point to a closed contour.")
        if segmentType is None:
            self.types.append(0)
        elif segmentType in ("line", "move"):
            self.types.append(flagOnCurve)
        elif segmentType == "qcurve":
            self.types.append(flagOnCurve)
        elif segmentType == "curve":
            self.types.append("curve")
        else:
            raise AssertionError(segmentType)

        self.points.append(pt)
<|endoftext|>"""
=========
PointPens
=========

Where **SegmentPens** have an intuitive approach to drawing
(if you're familiar with postscript anyway), the **PointPen**
is geared towards accessing all the data in the contours of
the glyph. A PointPen has a very simple interface, it just
steps through all the points in a call from glyph.drawPoints().
This allows the caller to provide more data for each point.
For instance, whether or not a point is smooth, and its name.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from fontTools.misc.enumTools import StrEnum
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
from fontTools.pens.basePen import AbstractPen, MissingComponentError, PenError

__all__ = [
    "AbstractPointPen",
    "BasePointToSegmentPen",
    "PointToSegmentPen",
    "SegmentToPointPen",
    "GuessSmoothPointPen",
    "ReverseContourPointPen",
    "ReverseFlipped",
]

# Some type aliases to make it easier below
Point = Tuple[float, float]
PointName = Optional[str]
# [(pt, smooth, name, kwargs)]
SegmentPointList = List[Tuple[Optional[Point], bool, PointName, Any]]
SegmentType = Optional[str]
SegmentList = List[Tuple[SegmentType, SegmentPointList]]


class ReverseFlipped(StrEnum):
    """How to handle flipped components during decomposition.

    NO: Don't reverse flipped components
    KEEP_START: Reverse flipped components, keeping original starting point
    ON_CURVE_FIRST: Reverse flipped components, ensuring first point is on-curve
    """

    NO = "no"
    KEEP_START = "keep_start"
    ON_CURVE_FIRST = "on_curve_first"


class AbstractPointPen:
    """Baseclass for all PointPens."""

    def beginPath(self, identifier: Optional[str] = None, **kwargs: Any) -> None:
        """Start a new sub path."""
        raise NotImplementedError

    def endPath(self) -> None:
        """End the current sub path."""
        raise NotImplementedError

    def addPoint(
        self,
        pt: Tuple[float, float],
        segmentType: Optional[str] = None,
        smooth: bool = False,
        name: Optional[str] = None,
        identifier: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Add a point to the current sub path."""
        raise NotImplementedError

    def addComponent(
        self,
        baseGlyphName: str,
        transformation: Tuple[float, float, float, float, float, float],
        identifier: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Add a sub glyph."""
        raise NotImplementedError

    def addVarComponent(
        self,
        glyphName: str,
        transformation: DecomposedTransform,
        location: Dict[str, float],
        identifier: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Add a VarComponent sub glyph. The 'transformation' argument
        must be a DecomposedTransform from the fontTools.misc.transform module,
        and the 'location' argument must be a dictionary mapping axis tags
        to their locations.
        """
        # ttGlyphSet decomposes for us
        raise AttributeError


class BasePointToSegmentPen(AbstractPointPen):
    """
    Base class for retrieving the outline in a segment-oriented
    way. The PointPen protocol is simple yet also a little tricky,
    so when you need an outline presented as segments but you have
    as points, do use this base implementation as it properly takes
    care of all the edge cases.
    """

    def __init__(self) -> None:
        self.currentPath = None

    def beginPath(self, identifier=None, **kwargs):
        if self.currentPath is not None:
            raise PenError("Path already begun.")
        self.currentPath = []

    def _flushContour(self, segments: SegmentList) -> None:
        """Override this method.

        It will be called for each non-empty sub path with a list
        of segments: the 'segments' argument.

        The segments list contains tuples of length 2:
                (segmentType, points)

        segmentType is one of "move", "line", "curve" or "qcurve".
        "move" may only occur as the first segment, and it signifies
        an OPEN path. A CLOSED path does NOT start with a "move", in
        fact it will not contain a "move" at ALL.

        The 'points' field in the 2-tuple is a list of point info
        tuples. The list has 1 or more items, a point tuple has
        four items:
                (point, smooth, name, kwargs)
        'point' is an (x, y) coordinate pair.

        For a closed path, the initial moveTo point is defined as
        the last point of the last segment.

        The 'points' list of "move" and "line" segments always contains
        exactly one point tuple.
        """
        raise NotImplementedError

    def endPath(self) -> None:
        if self.currentPath is None:
            raise PenError("Path not begun.")
        points = self.currentPath
        self.currentPath = None
        if not points:
            return
        if len(points) == 1:
            # Not much more we can do than output a single move segment.
            pt, segmentType, smooth, name, kwargs = points[0]
            segments: SegmentList = [("move", [(pt, smooth, name, kwargs)])]
            self._flushContour(segments)
            return
        segments = []
        if points[0][1] == "move":
            # It's an open contour, insert a "move" segment for the first
            # point and remove that first point from the point list.
            pt, segmentType, smooth, name, kwargs = points[0]
            segments.append(("move", [(pt, smooth, name, kwargs)]))
            points.pop(0)
        else:
            # It's a closed contour. Locate the first on-curve point, and
            # rotate the point list so that it _ends_ with an on-curve
            # point.
            firstOnCurve = None
            for i in range(len(points)):
                segmentType = points[i][1]
                if segmentType is not None:
                    firstOnCurve = i
                    break
            if firstOnCurve is None:
                # Special case for quadratics: a contour with no on-curve
                # points. Add a "None" point. (See also the Pen protocol's
                # qCurveTo() method and fontTools.pens.basePen.py.)
                points.append((None, "qcurve", None, None, None))
            else:
                points = points[firstOnCurve + 1 :] + points[: firstOnCurve + 1]

        currentSegment: SegmentPointList = []
        for pt, segmentType, smooth, name, kwargs in points:
            currentSegment.append((pt, smooth, name, kwargs))
            if segmentType is None:
                continue
            segments.append((segmentType, currentSegment))
            currentSegment = []

        self._flushContour(segments)

    def addPoint(
        self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs
    ):
        if self.currentPath is None:
            raise PenError("Path not begun")
        self.currentPath.append((pt, segmentType, smooth, name, kwargs))


class PointToSegmentPen(BasePointToSegmentPen):
    """
    Adapter class that converts the PointPen protocol to the
    (Segment)Pen protocol.

    NOTE: The segment pen does not support and will drop point names, identifiers
    and kwargs.
    """

    def __init__(self, segmentPen, outputImpliedClosingLine: bool = False) -> None:
        BasePointToSegmentPen.__init__(self)
        self.pen = segmentPen
        self.outputImpliedClosingLine = outputImpliedClosingLine

    def _flushContour(self, segments):
        if not segments:
            raise PenError("Must have at least one segment.")
        pen = self.pen
        if segments[0][0] == "move":
            # It's an open path.
            closed = False
            points = segments[0][1]
            if len(points) != 1:
                raise PenError(f"Illegal move segment point count: {len(points)}")
            movePt, _, _, _ = points[0]
            del segments[0]
        else:
            # It's a closed path, do a moveTo to the last
            # point of the last segment.
            closed = True
            segmentType, points = segments[-1]
            movePt, _, _, _ = points[-1]
        if movePt is None:
            # quad special case: a contour with no on-curve points contains
            # one "qcurve" segment that ends with a point that's None. We
            # must not output a moveTo() in that case.
            pass
        else:
            pen.moveTo(movePt)
        outputImpliedClosingLine = self.outputImpliedClosingLine
        nSegments = len(segments)
        lastPt = movePt
        for i in range(nSegments):
            segmentType, points = segments[i]
            points = [pt for pt, _, _, _ in points]
            if segmentType == "line":
                if len(points) != 1:
                    raise PenError(f"Illegal line segment point count: {len(points)}")
                pt = points[0]
                # For closed contours, a 'lineTo' is always implied from the last oncurve
                # point to the starting point, thus we can omit it when the last and
                # starting point don't overlap.
                # However, when the last oncurve point is a "line" segment and has same
                # coordinates as the starting point of a closed contour, we need to output
                # the closing 'lineTo' explicitly (regardless of the value of the
                # 'outputImpliedClosingLine' option) in order to disambiguate this case from
                # the implied closing 'lineTo', otherwise the duplicate point would be lost.
                # See https://github.com/googlefonts/fontmake/issues/572.
                if (
                    i + 1 != nSegments
                    or outputImpliedClosingLine
                    or not closed
                    or pt == lastPt
                ):
                    pen.lineTo(pt)
                    lastPt = pt
            elif segmentType == "curve":
                pen.curveTo(*points)
                lastPt = points[-1]
            elif segmentType == "qcurve":
                pen.qCurveTo(*points)
                lastPt = points[-1]
            else:
                raise PenError(f"Illegal segmentType: {segmentType}")
        if closed:
            pen.closePath()
        else:
            pen.endPath()

    def addComponent(self, glyphName, transform, identifier=None, **kwargs):
        del identifier  # unused
        del kwargs  # unused
        self.pen.addComponent(glyphName, transform)


class SegmentToPointPen(AbstractPen):
    """
    Adapter class that converts the (Segment)Pen protocol to the
    PointPen protocol.
    """

    def __init__(self, pointPen, guessSmooth=True) -> None:
        if guessSmooth:
            self.pen = GuessSmoothPointPen(pointPen)
        else:
            self.pen = pointPen
        self.contour: Optional[List[Tuple[Point, SegmentType]]] = None

    def _flushContour(self) -> None:
        pen = self.pen
        pen.beginPath()
        for pt, segmentType in self.contour:
            pen.addPoint(pt, segmentType=segmentType)
        pen.endPath()

    def moveTo(self, pt):
        self.contour = []
        self.contour.append((pt, "move"))

    def lineTo(self, pt):
        if self.contour is None:
            raise PenError("Contour missing required initial moveTo")
        self.contour.append((pt, "line"))

    def curveTo(self, *pts):
        if not pts:
            raise TypeError("Must pass in at least one point")
        if self.contour is None:
            raise PenError("Contour missing required initial moveTo")
        for pt in pts[:-1]:
            self.contour.append((pt, None))
        self.contour.append((pts[-1], "curve"))

    def qCurveTo(self, *pts):
        if not pts:
            raise TypeError("Must pass in at least one point")
        if pts[-1] is None:
            self.contour = []
        else:
            if self.contour is None:
                raise PenError("Contour missing required initial moveTo")
        for pt in pts[:-1]:
            self.contour.append((pt, None))
        if pts[-1] is not None:
            self.contour.append((pts[-1], "qcurve"))

    def closePath(self):
        if self.contour is None:
            raise PenError("Contour missing required initial moveTo")

        # Remove the last point if it's a duplicate of the first, but only if both
        # are on-curve points (segmentType is not None); for quad blobs
        # (all off-curve) every point must be preserved:
        # https://github.com/fonttools/fonttools/issues/4014
        if (
            len(self.contour) > 1
            and (self.contour[0][0] == self.contour[-1][0])
            and self.contour[0][1] is not None
            and self.contour[-1][1] is not None
        ):
            self.contour[0] = self.contour[-1]
            del self.contour[-1]
        else:
            # There's an implied line at the end, replace "move" with "line"
            # for the first point
            pt, tp = self.contour[0]
            if tp == "move":
                self.contour[0] = pt, "line"
        self._flushContour()
        self.contour = None

    def endPath(self):
        if self.contour is None:
            raise PenError("Contour missing required initial moveTo")
        self._flushContour()
        self.contour = None

    def addComponent(self, glyphName, transform):
        if self.contour is not None:
            raise PenError("Components must be added before or after contours")
        self.pen.addComponent(glyphName, transform)


class GuessSmoothPointPen(AbstractPointPen):
    """
    Filtering PointPen that tries to determine whether an on-curve point
    should be "smooth", ie. that it's a "tangent" point or a "curve" point.
    """

    def __init__(self, outPen, error=0.05):
        self._outPen = outPen
        self._error = error
        self._points = None

    def _flushContour(self):
        if self._points is None:
            raise PenError("Path not begun")
        points = self._points
        nPoints = len(points)
        if not nPoints:
            return
        if points[0][1] == "move":
            # Open path.
            indices = range(1, nPoints - 1)
        elif nPoints > 1:
            # Closed path. To avoid having to mod the contour index, we
            # simply abuse Python's negative index feature, and start at -1
            indices = range(-1, nPoints - 1)
        else:
            # closed path containing 1 point (!), ignore.
            indices = []
        for i in indices:
            pt, segmentType, _, name, kwargs = points[i]
            if segmentType is None:
                continue
            prev = i - 1
            next = i + 1
            if points[prev][1] is not None and points[next][1] is not None:
                continue
            # At least one of our neighbors is an off-curve point
            pt = points[i][0]
            prevPt = points[prev][0]
            nextPt = points[next][0]
            if pt != prevPt and pt != nextPt:
                dx1, dy1 = pt[0] - prevPt[0], pt[1] - prevPt[1]
                dx2, dy2 = nextPt[0] - pt[0], nextPt[1] - pt[1]
                a1 = math.atan2(dy1, dx1)
                a2 = math.atan2(dy2, dx2)
                if abs(a1 - a2) < self._error:
                    points[i] = pt, segmentType, True, name, kwargs

        for pt, segmentType, smooth, name, kwargs in points:
            self._outPen.addPoint(pt, segmentType, smooth, name, **kwargs)

    def beginPath(self, identifier=None, **kwargs):
        if self._points is not None:
            raise PenError("Path already begun")
        self._points = []
        if identifier is not None:
            kwargs["identifier"] = identifier
        self._outPen.beginPath(**kwargs)

    def endPath(self):
        self._flushContour()
        self._outPen.endPath()
        self._points = None

    def addPoint(
        self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs
    ):
        if self._points is None:
            raise PenError("Path not begun")
        if identifier is not None:
            kwargs["identifier"] = identifier
        self._points.append((pt, segmentType, False, name, kwargs))

    def addComponent(self, glyphName, transformation, identifier=None, **kwargs):
        if self._points is not None:
            raise PenError("Components must be added before or after contours")
        if identifier is not None:
            kwargs["identifier"] = identifier
        self._outPen.addComponent(glyphName, transformation, **kwargs)

    def addVarComponent(
        self, glyphName, transformation, location, identifier=None, **kwargs
    ):
        if self._points is not None:
            raise PenError("VarComponents must be added before or after contours")
        if identifier is not None:
            kwargs["identifier"] = identifier
        self._outPen.addVarComponent(glyphName, transformation, location, **kwargs)


class ReverseContourPointPen(AbstractPointPen):
    """
    This is a PointPen that passes outline data to another PointPen, but
    reversing the winding direction of all contours. Components are simply
    passed through unchanged.

    Closed contours are reversed in such a way that the first point remains
    the first point.
    """

    def __init__(self, outputPointPen):
        self.pen = outputPointPen
        # a place to store the points for the current sub path
        self.currentContour = None

    def _flushContour(self):
        pen = self.pen
        contour = self.currentContour
        if not contour:
            pen.beginPath(identifier=self.currentContourIdentifier)
            pen.endPath()
            return

        closed = contour[0][1] != "move"
        if not closed:
            lastSegmentType = "move"
        else:
            # Remove the first point and insert it at the end. When
            # the list of points gets reversed, this point will then
            # again be at the start. In other words, the following
            # will hold:
            #   for N in range(len(originalContour)):
            #       originalContour[N] == reversedContour[-N]
            contour.append(contour.pop(0))
            # Find the first on-curve point.
            firstOnCurve = None
            for i in range(len(contour)):
                if contour[i][1] is not None:
                    firstOnCurve = i
                    break
            if firstOnCurve is None:
                # There are no on-curve points, be basically have to
                # do nothing but contour.reverse().
                lastSegmentType = None
            else:
                lastSegmentType = contour[firstOnCurve][1]

        contour.reverse()
        if not closed:
            # Open paths must start with a move, so we simply dump
            # all off-curve points leading up to the first on-curve.
            while contour[0][1] is None:
                contour.pop(0)
        pen.beginPath(identifier=self.currentContourIdentifier)
        for pt, nextSegmentType, smooth, name, kwargs in contour:
            if nextSegmentType is not None:
                segmentType = lastSegmentType
                lastSegmentType = nextSegmentType
            else:
                segmentType = None
            pen.addPoint(
                pt, segmentType=segmentType, smooth=smooth, name=name, **kwargs
            )
        pen.endPath()

    def beginPath(self, identifier=None, **kwargs):
        if self.currentContour is not None:
            raise PenError("Path already begun")
        self.currentContour = []
        self.currentContourIdentifier = identifier
        self.onCurve = []

    def endPath(self):
        if self.currentContour is None:
            raise PenError("Path not begun")
        self._flushContour()
        self.currentContour = None

    def addPoint(
        self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs
    ):
        if self.currentContour is None:
            raise PenError("Path not begun")
        if identifier is not None:
            kwargs["identifier"] = identifier
        self.currentContour.append((pt, segmentType, smooth, name, kwargs))

    def addComponent(self, glyphName, transform, identifier=None, **kwargs):
        if self.currentContour is not None:
            raise PenError("Components must be added before or after contours")
        self.pen.addComponent(glyphName, transform, identifier=identifier, **kwargs)


class DecomposingPointPen(LogMixin, AbstractPointPen):
    """Implements a 'addComponent' method that decomposes components
    (i.e. draws them onto self as simple contours).
    It can also be used as a mixin class (e.g. see DecomposingRecordingPointPen).

    You must override beginPath, addPoint, endPath. You may
    additionally override addVarComponent and addComponent.

    By default a warning message is logged when a base glyph is missing;
    set the class variable ``skipMissingComponents`` to False if you want
    all instances of a sub-class to raise a :class:`MissingComponentError`
    exception by default.
    """

    skipMissingComponents = True
    # alias error for convenience
    MissingComponentError = MissingComponentError

    def __init__(
        self,
        glyphSet,
        *args,
        skipMissingComponents=None,
        reverseFlipped: bool | ReverseFlipped = False,
        **kwargs,
    ):
        """Takes a 'glyphSet' argument (dict), in which the glyphs that are referenced
        as components are looked up by their name.

        If the optional 'reverseFlipped' argument is True or a ReverseFlipped enum value,
        components whose transformation matrix has a negative determinant will be decomposed
        with a reversed path direction to compensate for the flip.

        The reverseFlipped parameter can be:
        - False or ReverseFlipped.NO: Don't reverse flipped components
        - True or ReverseFlipped.KEEP_START: Reverse, keeping original starting point
        - ReverseFlipped.ON_CURVE_FIRST: Reverse, ensuring first point is on-curve

        The optional 'skipMissingComponents' argument can be set to True/False to
        override the homonymous class attribute for a given pen instance.
        """
        super().__init__(*args, **kwargs)
        self.glyphSet = glyphSet
        self.skipMissingComponents = (
            self.__class__.skipMissingComponents
            if skipMissingComponents is None
            else skipMissingComponents
        )
        # Handle backward compatibility and validate string inputs
        if reverseFlipped is False:
            self.reverseFlipped = ReverseFlipped.NO
        elif reverseFlipped is True:
            self.reverseFlipped = ReverseFlipped.KEEP_START
        else:
            self.reverseFlipped = ReverseFlipped(reverseFlipped)

    def addComponent(self, baseGlyphName, transformation, identifier=None, **kwargs):
        """Transform the points of the base glyph and draw it onto self.

        The `identifier` parameter and any extra kwargs are ignored.
        """
        from fontTools.pens.transformPen import TransformPointPen

        try:
            glyph = self.glyphSet[baseGlyphName]
        except KeyError:
            if not self.skipMissingComponents:
                raise MissingComponentError(baseGlyphName)
            self.log.warning(
                "glyph '%s' is missing from glyphSet; skipped" % baseGlyphName
            )
        else:
            pen = self
            if transformation != Identity:
                pen = TransformPointPen(pen, transformation)
            if self.reverseFlipped != ReverseFlipped.NO:
                # if the transformation has a negative determinant, it will
                # reverse the contour direction of the component
                a, b, c, d = transformation[:4]
                if a * d - b * c < 0:
                    pen = ReverseContourPointPen(pen)

                    if self.reverseFlipped == ReverseFlipped.ON_CURVE_FIRST:
                        from fontTools.pens.filterPen import OnCurveFirstPointPen

                        # Ensure the starting point is an on-curve.
                        # Wrap last so this filter runs first during drawPoints
                        pen = OnCurveFirstPointPen(pen)

            glyph.drawPoints(pen)
<|endoftext|># -*- coding: utf-8 -*-
"""Calculate the perimeter of a glyph."""

from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import (
    approximateQuadraticArcLengthC,
    calcQuadraticArcLengthC,
    approximateCubicArcLengthC,
    calcCubicArcLengthC,
)
import math

__all__ = ["PerimeterPen"]


def _distance(p0, p1):
    return math.hypot(p0[0] - p1[0], p0[1] - p1[1])


class PerimeterPen(BasePen):
    def __init__(self, glyphset=None, tolerance=0.005):
        BasePen.__init__(self, glyphset)
        self.value = 0
        self.tolerance = tolerance

        # Choose which algorithm to use for quadratic and for cubic.
        # Quadrature is faster but has fixed error characteristic with no strong
        # error bound.  The cutoff points are derived empirically.
        self._addCubic = (
            self._addCubicQuadrature if tolerance >= 0.0015 else self._addCubicRecursive
        )
        self._addQuadratic = (
            self._addQuadraticQuadrature
            if tolerance >= 0.00075
            else self._addQuadraticExact
        )

    def _moveTo(self, p0):
        self.__startPoint = p0

    def _closePath(self):
        p0 = self._getCurrentPoint()
        if p0 != self.__startPoint:
            self._lineTo(self.__startPoint)

    def _lineTo(self, p1):
        p0 = self._getCurrentPoint()
        self.value += _distance(p0, p1)

    def _addQuadraticExact(self, c0, c1, c2):
        self.value += calcQuadraticArcLengthC(c0, c1, c2)

    def _addQuadraticQuadrature(self, c0, c1, c2):
        self.value += approximateQuadraticArcLengthC(c0, c1, c2)

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        self._addQuadratic(complex(*p0), complex(*p1), complex(*p2))

    def _addCubicRecursive(self, c0, c1, c2, c3):
        self.value += calcCubicArcLengthC(c0, c1, c2, c3, self.tolerance)

    def _addCubicQuadrature(self, c0, c1, c2, c3):
        self.value += approximateCubicArcLengthC(c0, c1, c2, c3)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._getCurrentPoint()
        self._addCubic(complex(*p0), complex(*p1), complex(*p2), complex(*p3))
<|endoftext|>from fontTools.pens.basePen import BasePen

__all__ = ["WxPen"]


class WxPen(BasePen):
    def __init__(self, glyphSet, path=None):
        BasePen.__init__(self, glyphSet)
        if path is None:
            import wx

            path = wx.GraphicsRenderer.GetDefaultRenderer().CreatePath()
        self.path = path

    def _moveTo(self, p):
        self.path.MoveToPoint(*p)

    def _lineTo(self, p):
        self.path.AddLineToPoint(*p)

    def _curveToOne(self, p1, p2, p3):
        self.path.AddCurveToPoint(*p1 + p2 + p3)

    def _qCurveToOne(self, p1, p2):
        self.path.AddQuadCurveToPoint(*p1 + p2)

    def _closePath(self):
        self.path.CloseSubpath()
<|endoftext|>from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.transform import Transform
from fontTools.pens.filterPen import FilterPen, FilterPointPen

__all__ = ["RoundingPen", "RoundingPointPen"]


class RoundingPen(FilterPen):
    """
    Filter pen that rounds point coordinates and component XY offsets to integer. For
    rounding the component transform values, a separate round function can be passed to
    the pen.

    >>> from fontTools.pens.recordingPen import RecordingPen
    >>> recpen = RecordingPen()
    >>> roundpen = RoundingPen(recpen)
    >>> roundpen.moveTo((0.4, 0.6))
    >>> roundpen.lineTo((1.6, 2.5))
    >>> roundpen.qCurveTo((2.4, 4.6), (3.3, 5.7), (4.9, 6.1))
    >>> roundpen.curveTo((6.4, 8.6), (7.3, 9.7), (8.9, 10.1))
    >>> roundpen.addComponent("a", (1.5, 0, 0, 1.5, 10.5, -10.5))
    >>> recpen.value == [
    ...     ('moveTo', ((0, 1),)),
    ...     ('lineTo', ((2, 3),)),
    ...     ('qCurveTo', ((2, 5), (3, 6), (5, 6))),
    ...     ('curveTo', ((6, 9), (7, 10), (9, 10))),
    ...     ('addComponent', ('a', (1.5, 0, 0, 1.5, 11, -10))),
    ... ]
    True
    """

    def __init__(self, outPen, roundFunc=otRound, transformRoundFunc=noRound):
        super().__init__(outPen)
        self.roundFunc = roundFunc
        self.transformRoundFunc = transformRoundFunc

    def moveTo(self, pt):
        self._outPen.moveTo((self.roundFunc(pt[0]), self.roundFunc(pt[1])))

    def lineTo(self, pt):
        self._outPen.lineTo((self.roundFunc(pt[0]), self.roundFunc(pt[1])))

    def curveTo(self, *points):
        self._outPen.curveTo(
            *((self.roundFunc(x), self.roundFunc(y)) for x, y in points)
        )

    def qCurveTo(self, *points):
        self._outPen.qCurveTo(
            *((self.roundFunc(x), self.roundFunc(y)) for x, y in points)
        )

    def addComponent(self, glyphName, transformation):
        xx, xy, yx, yy, dx, dy = transformation
        self._outPen.addComponent(
            glyphName,
            Transform(
                self.transformRoundFunc(xx),
                self.transformRoundFunc(xy),
                self.transformRoundFunc(yx),
                self.transformRoundFunc(yy),
                self.roundFunc(dx),
                self.roundFunc(dy),
            ),
        )


class RoundingPointPen(FilterPointPen):
    """
    Filter point pen that rounds point coordinates and component XY offsets to integer.
    For rounding the component scale values, a separate round function can be passed to
    the pen.

    >>> from fontTools.pens.recordingPen import RecordingPointPen
    >>> recpen = RecordingPointPen()
    >>> roundpen = RoundingPointPen(recpen)
    >>> roundpen.beginPath()
    >>> roundpen.addPoint((0.4, 0.6), 'line')
    >>> roundpen.addPoint((1.6, 2.5), 'line')
    >>> roundpen.addPoint((2.4, 4.6))
    >>> roundpen.addPoint((3.3, 5.7))
    >>> roundpen.addPoint((4.9, 6.1), 'qcurve')
    >>> roundpen.endPath()
    >>> roundpen.addComponent("a", (1.5, 0, 0, 1.5, 10.5, -10.5))
    >>> recpen.value == [
    ...     ('beginPath', (), {}),
    ...     ('addPoint', ((0, 1), 'line', False, None), {}),
    ...     ('addPoint', ((2, 3), 'line', False, None), {}),
    ...     ('addPoint', ((2, 5), None, False, None), {}),
    ...     ('addPoint', ((3, 6), None, False, None), {}),
    ...     ('addPoint', ((5, 6), 'qcurve', False, None), {}),
    ...     ('endPath', (), {}),
    ...     ('addComponent', ('a', (1.5, 0, 0, 1.5, 11, -10)), {}),
    ... ]
    True
    """

    def __init__(self, outPen, roundFunc=otRound, transformRoundFunc=noRound):
        super().__init__(outPen)
        self.roundFunc = roundFunc
        self.transformRoundFunc = transformRoundFunc

    def addPoint(
        self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs
    ):
        self._outPen.addPoint(
            (self.roundFunc(pt[0]), self.roundFunc(pt[1])),
            segmentType=segmentType,
            smooth=smooth,
            name=name,
            identifier=identifier,
            **kwargs,
        )

    def addComponent(self, baseGlyphName, transformation, identifier=None, **kwargs):
        xx, xy, yx, yy, dx, dy = transformation
        self._outPen.addComponent(
            baseGlyphName,
            Transform(
                self.transformRoundFunc(xx),
                self.transformRoundFunc(xy),
                self.transformRoundFunc(yx),
                self.transformRoundFunc(yy),
                self.roundFunc(dx),
                self.roundFunc(dy),
            ),
            identifier=identifier,
            **kwargs,
        )
<|endoftext|>"""
colorLib.builder: Build COLR/CPAL tables from scratch

"""

import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder

# TODO move type aliases to colorLib.types?
T = TypeVar("T")
_Kwargs = Mapping[str, Any]
_PaintInput = Union[int, _Kwargs, ot.Paint, Tuple[str, "_PaintInput"]]
_PaintInputList = Sequence[_PaintInput]
_ColorGlyphsDict = Dict[str, Union[_PaintInputList, _PaintInput]]
_ColorGlyphsV0Dict = Dict[str, Sequence[Tuple[str, int]]]
_ClipBoxInput = Union[
    Tuple[int, int, int, int, int],  # format 1, variable
    Tuple[int, int, int, int],  # format 0, non-variable
    ot.ClipBox,
]


MAX_PAINT_COLR_LAYER_COUNT = 255
_DEFAULT_ALPHA = 1.0
_MAX_REUSE_LEN = 32


def _beforeBuildPaintRadialGradient(paint, source):
    x0 = source["x0"]
    y0 = source["y0"]
    r0 = source["r0"]
    x1 = source["x1"]
    y1 = source["y1"]
    r1 = source["r1"]

    # TODO apparently no builder_test confirms this works (?)

    # avoid abrupt change after rounding when c0 is near c1's perimeter
    c = round_start_circle_stable_containment((x0, y0), r0, (x1, y1), r1)
    x0, y0 = c.centre
    r0 = c.radius

    # update source to ensure paint is built with corrected values
    source["x0"] = x0
    source["y0"] = y0
    source["r0"] = r0
    source["x1"] = x1
    source["y1"] = y1
    source["r1"] = r1

    return paint, source


def _defaultColorStop():
    colorStop = ot.ColorStop()
    colorStop.Alpha = _DEFAULT_ALPHA
    return colorStop


def _defaultVarColorStop():
    colorStop = ot.VarColorStop()
    colorStop.Alpha = _DEFAULT_ALPHA
    return colorStop


def _defaultColorLine():
    colorLine = ot.ColorLine()
    colorLine.Extend = ExtendMode.PAD
    return colorLine


def _defaultVarColorLine():
    colorLine = ot.VarColorLine()
    colorLine.Extend = ExtendMode.PAD
    return colorLine


def _defaultPaintSolid():
    paint = ot.Paint()
    paint.Alpha = _DEFAULT_ALPHA
    return paint


def _buildPaintCallbacks():
    return {
        (
            BuildCallback.BEFORE_BUILD,
            ot.Paint,
            ot.PaintFormat.PaintRadialGradient,
        ): _beforeBuildPaintRadialGradient,
        (
            BuildCallback.BEFORE_BUILD,
            ot.Paint,
            ot.PaintFormat.PaintVarRadialGradient,
        ): _beforeBuildPaintRadialGradient,
        (BuildCallback.CREATE_DEFAULT, ot.ColorStop): _defaultColorStop,
        (BuildCallback.CREATE_DEFAULT, ot.VarColorStop): _defaultVarColorStop,
        (BuildCallback.CREATE_DEFAULT, ot.ColorLine): _defaultColorLine,
        (BuildCallback.CREATE_DEFAULT, ot.VarColorLine): _defaultVarColorLine,
        (
            BuildCallback.CREATE_DEFAULT,
            ot.Paint,
            ot.PaintFormat.PaintSolid,
        ): _defaultPaintSolid,
        (
            BuildCallback.CREATE_DEFAULT,
            ot.Paint,
            ot.PaintFormat.PaintVarSolid,
        ): _defaultPaintSolid,
    }


def populateCOLRv0(
    table: ot.COLR,
    colorGlyphsV0: _ColorGlyphsV0Dict,
    glyphMap: Optional[Mapping[str, int]] = None,
):
    """Build v0 color layers and add to existing COLR table.

    Args:
        table: a raw ``otTables.COLR()`` object (not ttLib's ``table_C_O_L_R_``).
        colorGlyphsV0: map of base glyph names to lists of (layer glyph names,
            color palette index) tuples. Can be empty.
        glyphMap: a map from glyph names to glyph indices, as returned from
            ``TTFont.getReverseGlyphMap()``, to optionally sort base records by GID.
    """
    if glyphMap is not None:
        _check_base_glyphs_exist(colorGlyphsV0, glyphMap, "populateCOLRv0")
        colorGlyphItems = sorted(
            colorGlyphsV0.items(), key=lambda item: glyphMap[item[0]]
        )
    else:
        colorGlyphItems = colorGlyphsV0.items()
    baseGlyphRecords = []
    layerRecords = []
    for baseGlyph, layers in colorGlyphItems:
        baseRec = ot.BaseGlyphRecord()
        baseRec.BaseGlyph = baseGlyph
        baseRec.FirstLayerIndex = len(layerRecords)
        baseRec.NumLayers = len(layers)
        baseGlyphRecords.append(baseRec)

        for layerGlyph, paletteIndex in layers:
            if glyphMap is not None and layerGlyph not in glyphMap:
                raise ColorLibError(
                    f"populateCOLRv0: layer glyph not found in glyphMap: {layerGlyph!r}"
                )
            layerRec = ot.LayerRecord()
            layerRec.LayerGlyph = layerGlyph
            layerRec.PaletteIndex = paletteIndex
            layerRecords.append(layerRec)

    table.BaseGlyphRecordArray = table.LayerRecordArray = None
    if baseGlyphRecords:
        table.BaseGlyphRecordArray = ot.BaseGlyphRecordArray()
        table.BaseGlyphRecordArray.BaseGlyphRecord = baseGlyphRecords
    if layerRecords:
        table.LayerRecordArray = ot.LayerRecordArray()
        table.LayerRecordArray.LayerRecord = layerRecords
    table.BaseGlyphRecordCount = len(baseGlyphRecords)
    table.LayerRecordCount = len(layerRecords)


def buildCOLR(
    colorGlyphs: _ColorGlyphsDict,
    version: Optional[int] = None,
    *,
    glyphMap: Optional[Mapping[str, int]] = None,
    varStore: Optional[ot.VarStore] = None,
    varIndexMap: Optional[ot.DeltaSetIndexMap] = None,
    clipBoxes: Optional[Dict[str, _ClipBoxInput]] = None,
    allowLayerReuse: bool = True,
) -> C_O_L_R_.table_C_O_L_R_:
    """Build COLR table from color layers mapping.

    Args:

        colorGlyphs: map of base glyph name to, either list of (layer glyph name,
            color palette index) tuples for COLRv0; or a single ``Paint`` (dict) or
            list of ``Paint`` for COLRv1.
        version: the version of COLR table. If None, the version is determined
            by the presence of COLRv1 paints or variation data (varStore), which
            require version 1; otherwise, if all base glyphs use only simple color
            layers, version 0 is used.
        glyphMap: a map from glyph names to glyph indices, as returned from
            TTFont.getReverseGlyphMap(), to optionally sort base records by GID.
        varStore: Optional ItemVarationStore for deltas associated with v1 layer.
        varIndexMap: Optional DeltaSetIndexMap for deltas associated with v1 layer.
        clipBoxes: Optional map of base glyph name to clip box 4- or 5-tuples:
            (xMin, yMin, xMax, yMax) or (xMin, yMin, xMax, yMax, varIndexBase).

    Returns:
        A new COLR table.
    """
    self = C_O_L_R_.table_C_O_L_R_()

    if varStore is not None and version == 0:
        raise ValueError("Can't add VarStore to COLRv0")

    if version in (None, 0) and not varStore:
        # split color glyphs into v0 and v1 and encode separately
        colorGlyphsV0, colorGlyphsV1 = _split_color_glyphs_by_version(colorGlyphs)
        if version == 0 and colorGlyphsV1:
            raise ValueError("Can't encode COLRv1 glyphs in COLRv0")
    else:
        # unless explicitly requested for v1 or have variations, in which case
        # we encode all color glyph as v1
        colorGlyphsV0, colorGlyphsV1 = {}, colorGlyphs

    colr = ot.COLR()

    populateCOLRv0(colr, colorGlyphsV0, glyphMap)

    colr.LayerList, colr.BaseGlyphList = buildColrV1(
        colorGlyphsV1,
        glyphMap,
        allowLayerReuse=allowLayerReuse,
    )

    if version is None:
        version = 1 if (varStore or colorGlyphsV1) else 0
    elif version not in (0, 1):
        raise NotImplementedError(version)
    self.version = colr.Version = version

    if version == 0:
        self.ColorLayers = self._decompileColorLayersV0(colr)
    else:
        colr.ClipList = buildClipList(clipBoxes) if clipBoxes else None
        colr.VarIndexMap = varIndexMap
        colr.VarStore = varStore
        self.table = colr

    return self


def buildClipList(clipBoxes: Dict[str, _ClipBoxInput]) -> ot.ClipList:
    clipList = ot.ClipList()
    clipList.Format = 1
    clipList.clips = {name: buildClipBox(box) for name, box in clipBoxes.items()}
    return clipList


def buildClipBox(clipBox: _ClipBoxInput) -> ot.ClipBox:
    if isinstance(clipBox, ot.ClipBox):
        return clipBox
    n = len(clipBox)
    clip = ot.ClipBox()
    if n not in (4, 5):
        raise ValueError(f"Invalid ClipBox: expected 4 or 5 values, found {n}")
    clip.xMin, clip.yMin, clip.xMax, clip.yMax = intRect(clipBox[:4])
    clip.Format = int(n == 5) + 1
    if n == 5:
        clip.VarIndexBase = int(clipBox[4])
    return clip


class ColorPaletteType(enum.IntFlag):
    USABLE_WITH_LIGHT_BACKGROUND = 0x0001
    USABLE_WITH_DARK_BACKGROUND = 0x0002

    @classmethod
    def _missing_(cls, value):
        # enforce reserved bits
        if isinstance(value, int) and (value < 0 or value & 0xFFFC != 0):
            raise ValueError(f"{value} is not a valid {cls.__name__}")
        return super()._missing_(value)


# None, 'abc' or {'en': 'abc', 'de': 'xyz'}
_OptionalLocalizedString = Union[None, str, Dict[str, str]]


def buildPaletteLabels(
    labels: Iterable[_OptionalLocalizedString], nameTable: _n_a_m_e.table__n_a_m_e
) -> List[Optional[int]]:
    return [
        (
            nameTable.addMultilingualName(l, mac=False)
            if isinstance(l, dict)
            else (
                C_P_A_L_.table_C_P_A_L_.NO_NAME_ID
                if l is None
                else nameTable.addMultilingualName({"en": l}, mac=False)
            )
        )
        for l in labels
    ]


def buildCPAL(
    palettes: Sequence[Sequence[Tuple[float, float, float, float]]],
    paletteTypes: Optional[Sequence[ColorPaletteType]] = None,
    paletteLabels: Optional[Sequence[_OptionalLocalizedString]] = None,
    paletteEntryLabels: Optional[Sequence[_OptionalLocalizedString]] = None,
    nameTable: Optional[_n_a_m_e.table__n_a_m_e] = None,
) -> C_P_A_L_.table_C_P_A_L_:
    """Build CPAL table from list of color palettes.

    Args:
        palettes: list of lists of colors encoded as tuples of (R, G, B, A) floats
            in the range [0..1].
        paletteTypes: optional list of ColorPaletteType, one for each palette.
        paletteLabels: optional list of palette labels. Each lable can be either:
            None (no label), a string (for for default English labels), or a
            localized string (as a dict keyed with BCP47 language codes).
        paletteEntryLabels: optional list of palette entry labels, one for each
            palette entry (see paletteLabels).
        nameTable: optional name table where to store palette and palette entry
            labels. Required if either paletteLabels or paletteEntryLabels is set.

    Return:
        A new CPAL v0 or v1 table, if custom palette types or labels are specified.
    """
    if len({len(p) for p in palettes}) != 1:
        raise ColorLibError("color palettes have different lengths")

    if (paletteLabels or paletteEntryLabels) and not nameTable:
        raise TypeError(
            "nameTable is required if palette or palette entries have labels"
        )

    cpal = C_P_A_L_.table_C_P_A_L_()
    cpal.numPaletteEntries = len(palettes[0])

    cpal.palettes = []
    for i, palette in enumerate(palettes):
        colors = []
        for j, color in enumerate(palette):
            if not isinstance(color, tuple) or len(color) != 4:
                raise ColorLibError(
                    f"In palette[{i}][{j}]: expected (R, G, B, A) tuple, got {color!r}"
                )
            if any(v > 1 or v < 0 for v in color):
                raise ColorLibError(
                    f"palette[{i}][{j}] has invalid out-of-range [0..1] color: {color!r}"
                )
            # input colors are RGBA, CPAL encodes them as BGRA
            red, green, blue, alpha = color
            colors.append(
                C_P_A_L_.Color(*(round(v * 255) for v in (blue, green, red, alpha)))
            )
        cpal.palettes.append(colors)

    if any(v is not None for v in (paletteTypes, paletteLabels, paletteEntryLabels)):
        cpal.version = 1

        if paletteTypes is not None:
            if len(paletteTypes) != len(palettes):
                raise ColorLibError(
                    f"Expected {len(palettes)} paletteTypes, got {len(paletteTypes)}"
                )
            cpal.paletteTypes = [ColorPaletteType(t).value for t in paletteTypes]
        else:
            cpal.paletteTypes = [C_P_A_L_.table_C_P_A_L_.DEFAULT_PALETTE_TYPE] * len(
                palettes
            )

        if paletteLabels is not None:
            if len(paletteLabels) != len(palettes):
                raise ColorLibError(
                    f"Expected {len(palettes)} paletteLabels, got {len(paletteLabels)}"
                )
            cpal.paletteLabels = buildPaletteLabels(paletteLabels, nameTable)
        else:
            cpal.paletteLabels = [C_P_A_L_.table_C_P_A_L_.NO_NAME_ID] * len(palettes)

        if paletteEntryLabels is not None:
            if len(paletteEntryLabels) != cpal.numPaletteEntries:
                raise ColorLibError(
                    f"Expected {cpal.numPaletteEntries} paletteEntryLabels, "
                    f"got {len(paletteEntryLabels)}"
                )
            cpal.paletteEntryLabels = buildPaletteLabels(paletteEntryLabels, nameTable)
        else:
            cpal.paletteEntryLabels = [
                C_P_A_L_.table_C_P_A_L_.NO_NAME_ID
            ] * cpal.numPaletteEntries
    else:
        cpal.version = 0

    return cpal


# COLR v1 tables
# See draft proposal at: https://github.com/googlefonts/colr-gradients-spec


def _is_colrv0_layer(layer: Any) -> bool:
    # Consider as COLRv0 layer any sequence of length 2 (be it tuple or list) in which
    # the first element is a str (the layerGlyph) and the second element is an int
    # (CPAL paletteIndex).
    # https://github.com/googlefonts/ufo2ft/issues/426
    try:
        layerGlyph, paletteIndex = layer
    except (TypeError, ValueError):
        return False
    else:
        return isinstance(layerGlyph, str) and isinstance(paletteIndex, int)


def _split_color_glyphs_by_version(
    colorGlyphs: _ColorGlyphsDict,
) -> Tuple[_ColorGlyphsV0Dict, _ColorGlyphsDict]:
    colorGlyphsV0 = {}
    colorGlyphsV1 = {}
    for baseGlyph, layers in colorGlyphs.items():
        if all(_is_colrv0_layer(l) for l in layers):
            colorGlyphsV0[baseGlyph] = layers
        else:
            colorGlyphsV1[baseGlyph] = layers

    # sanity check
    assert set(colorGlyphs) == (set(colorGlyphsV0) | set(colorGlyphsV1))

    return colorGlyphsV0, colorGlyphsV1


def _reuse_ranges(num_layers: int) -> Generator[Tuple[int, int], None, None]:
    # TODO feels like something itertools might have already
    for lbound in range(num_layers):
        # Reuse of very large #s of layers is relatively unlikely
        # +2: we want sequences of at least 2
        # otData handles single-record duplication
        for ubound in range(
            lbound + 2, min(num_layers + 1, lbound + 2 + _MAX_REUSE_LEN)
        ):
            yield (lbound, ubound)


class LayerReuseCache:
    reusePool: Mapping[Tuple[Any, ...], int]
    tuples: Mapping[int, Tuple[Any, ...]]
    keepAlive: List[ot.Paint]  # we need id to remain valid

    def __init__(self):
        self.reusePool = {}
        self.tuples = {}
        self.keepAlive = []

    def _paint_tuple(self, paint: ot.Paint):
        # start simple, who even cares about cyclic graphs or interesting field types
        def _tuple_safe(value):
            if isinstance(value, enum.Enum):
                return value
            elif hasattr(value, "__dict__"):
                return tuple(
                    (k, _tuple_safe(v)) for k, v in sorted(value.__dict__.items())
                )
            elif isinstance(value, collections.abc.MutableSequence):
                return tuple(_tuple_safe(e) for e in value)
            return value

        # Cache the tuples for individual Paint instead of the whole sequence
        # because the seq could be a transient slice
        result = self.tuples.get(id(paint), None)
        if result is None:
            result = _tuple_safe(paint)
            self.tuples[id(paint)] = result
            self.keepAlive.append(paint)
        return result

    def _as_tuple(self, paints: Sequence[ot.Paint]) -> Tuple[Any, ...]:
        return tuple(self._paint_tuple(p) for p in paints)

    def try_reuse(self, layers: List[ot.Paint]) -> List[ot.Paint]:
        found_reuse = True
        while found_reuse:
            found_reuse = False

            ranges = sorted(
                _reuse_ranges(len(layers)),
                key=lambda t: (t[1] - t[0], t[1], t[0]),
                reverse=True,
            )
            for lbound, ubound in ranges:
                reuse_lbound = self.reusePool.get(
                    self._as_tuple(layers[lbound:ubound]), -1
                )
                if reuse_lbound == -1:
                    continue
                new_slice = ot.Paint()
                new_slice.Format = int(ot.PaintFormat.PaintColrLayers)
                new_slice.NumLayers = ubound - lbound
                new_slice.FirstLayerIndex = reuse_lbound
                layers = layers[:lbound] + [new_slice] + layers[ubound:]
                found_reuse = True
                break
        return layers

    def add(self, layers: List[ot.Paint], first_layer_index: int):
        for lbound, ubound in _reuse_ranges(len(layers)):
            self.reusePool[self._as_tuple(layers[lbound:ubound])] = (
                lbound + first_layer_index
            )


class LayerListBuilder:
    layers: List[ot.Paint]
    cache: LayerReuseCache
    allowLayerReuse: bool

    def __init__(self, *, glyphMap=None, allowLayerReuse=True):
        self.glyphMap = glyphMap
        self.layers = []
        if allowLayerReuse:
            self.cache = LayerReuseCache()
        else:
            self.cache = None

        # We need to intercept construction of PaintColrLayers
        callbacks = _buildPaintCallbacks()
        callbacks[
            (
                BuildCallback.BEFORE_BUILD,
                ot.Paint,
                ot.PaintFormat.PaintColrLayers,
            )
        ] = self._beforeBuildPaintColrLayers
        # When a glyphMap is available, sanity check that glyphs referenced by
        # PaintGlyph and PaintColrGlyph actually exist, so a missing glyph raises
        # a legible error here instead of failing obscurely later (e.g. a
        # struct.error at compile time).
        # https://github.com/fonttools/fonttools/issues/2629
        if glyphMap is not None:
            for paintFormat in (
                ot.PaintFormat.PaintGlyph,
                ot.PaintFormat.PaintColrGlyph,
            ):
                callbacks[(BuildCallback.AFTER_BUILD, ot.Paint, paintFormat)] = (
                    self._checkPaintGlyphExists
                )
        self.tableBuilder = TableBuilder(callbacks)

    def _checkPaintGlyphExists(self, paint: ot.Paint) -> ot.Paint:
        if paint.Glyph not in self.glyphMap:
            paintName = ot.PaintFormat(paint.Format).name
            raise ColorLibError(
                f"{paintName}: glyph not found in glyphMap: {paint.Glyph!r}"
            )
        return paint

    # COLR layers is unusual in that it modifies shared state
    # so we need a callback into an object
    def _beforeBuildPaintColrLayers(self, dest, source):
        # Sketchy gymnastics: a sequence input will have dropped it's layers
        # into NumLayers; get it back
        if isinstance(source.get("NumLayers", None), collections.abc.Sequence):
            layers = source["NumLayers"]
        else:
            layers = source["Layers"]

        # Convert maps seqs or whatever into typed objects
        layers = [self.buildPaint(l) for l in layers]

        # No reason to have a colr layers with just one entry
        if len(layers) == 1:
            return layers[0], {}

        if self.cache is not None:
            # Look for reuse, with preference to longer sequences
            # This may make the layer list smaller
            layers = self.cache.try_reuse(layers)

        # The layer list is now final; if it's too big we need to tree it
        is_tree = len(layers) > MAX_PAINT_COLR_LAYER_COUNT
        layers = build_n_ary_tree(layers, n=MAX_PAINT_COLR_LAYER_COUNT)

        # We now have a tree of sequences with Paint leaves.
        # Convert the sequences into PaintColrLayers.
        def listToColrLayers(layer):
            if isinstance(layer, collections.abc.Sequence):
                return self.buildPaint(
                    {
                        "Format": ot.PaintFormat.PaintColrLayers,
                        "Layers": [listToColrLayers(l) for l in layer],
                    }
                )
            return layer

        layers = [listToColrLayers(l) for l in layers]

        # No reason to have a colr layers with just one entry
        if len(layers) == 1:
            return layers[0], {}

        paint = ot.Paint()
        paint.Format = int(ot.PaintFormat.PaintColrLayers)
        paint.NumLayers = len(layers)
        paint.FirstLayerIndex = len(self.layers)
        self.layers.extend(layers)

        # Register our parts for reuse provided we aren't a tree
        # If we are a tree the leaves registered for reuse and that will suffice
        if self.cache is not None and not is_tree:
            self.cache.add(layers, paint.FirstLayerIndex)

        # we've fully built dest; empty source prevents generalized build from kicking in
        return paint, {}

    def buildPaint(self, paint: _PaintInput) -> ot.Paint:
        return self.tableBuilder.build(ot.Paint, paint)

    def build(self) -> Optional[ot.LayerList]:
        if not self.layers:
            return None
        layers = ot.LayerList()
        layers.LayerCount = len(self.layers)
        layers.Paint = self.layers
        return layers


def buildBaseGlyphPaintRecord(
    baseGlyph: str, layerBuilder: LayerListBuilder, paint: _PaintInput
) -> ot.BaseGlyphList:
    self = ot.BaseGlyphPaintRecord()
    self.BaseGlyph = baseGlyph
    self.Paint = layerBuilder.buildPaint(paint)
    return self


def _check_base_glyphs_exist(colorGlyphs, glyphMap, where):
    """Checks that every base glyph name in colorGlyphs exists in glyphMap."""

    missing = []
    for baseGlyph in colorGlyphs.keys():
        if baseGlyph not in glyphMap:
            missing.append(baseGlyph)

    if missing:
        preview = ", ".join(missing[:10])
        extra = ""
        if len(missing) > 10:
            extra = f" (and {len(missing) - 10} more)"
        raise ColorLibError(
            f"{where}: base glyph(s) not found in glyphMap: {preview}{extra}"
        )


def _format_glyph_errors(errors: Mapping[str, Exception]) -> str:
    lines = []
    for baseGlyph, error in sorted(errors.items()):
        lines.append(f"    {baseGlyph} => {type(error).__name__}: {error}")
    return "\n".join(lines)


def buildColrV1(
    colorGlyphs: _ColorGlyphsDict,
    glyphMap: Optional[Mapping[str, int]] = None,
    *,
    allowLayerReuse: bool = True,
) -> Tuple[Optional[ot.LayerList], ot.BaseGlyphList]:
    if glyphMap is not None:
        _check_base_glyphs_exist(colorGlyphs, glyphMap, "buildColrV1")
        colorGlyphItems = sorted(
            colorGlyphs.items(), key=lambda item: glyphMap[item[0]]
        )
    else:
        colorGlyphItems = colorGlyphs.items()

    errors = {}
    baseGlyphs = []
    layerBuilder = LayerListBuilder(glyphMap=glyphMap, allowLayerReuse=allowLayerReuse)
    for baseGlyph, paint in colorGlyphItems:
        try:
            baseGlyphs.append(buildBaseGlyphPaintRecord(baseGlyph, layerBuilder, paint))

        except (ColorLibError, OverflowError, ValueError, TypeError) as e:
            errors[baseGlyph] = e

    if errors:
        failed_glyphs = _format_glyph_errors(errors)
        exc = ColorLibError(f"Failed to build BaseGlyphList:\n{failed_glyphs}")
        exc.errors = errors
        raise exc from next(iter(errors.values()))

    layers = layerBuilder.build()
    glyphs = ot.BaseGlyphList()
    glyphs.BaseGlyphCount = len(baseGlyphs)
    glyphs.BaseGlyphPaintRecord = baseGlyphs
    return (layers, glyphs)
<|endoftext|>"""
colorLib.table_builder: Generic helper for filling in BaseTable derivatives from tuples and maps and such.

"""

import collections
import enum
from fontTools.ttLib.tables.otBase import (
    BaseTable,
    FormatSwitchingBaseTable,
    UInt8FormatSwitchingBaseTable,
)
from fontTools.ttLib.tables.otConverters import (
    ComputedInt,
    SimpleValue,
    Struct,
    Short,
    UInt8,
    UShort,
    IntValue,
    FloatValue,
    OptionalValue,
)
from fontTools.misc.roundTools import otRound


class BuildCallback(enum.Enum):
    """Keyed on (BEFORE_BUILD, class[, Format if available]).
    Receives (dest, source).
    Should return (dest, source), which can be new objects.
    """

    BEFORE_BUILD = enum.auto()

    """Keyed on (AFTER_BUILD, class[, Format if available]).
    Receives (dest).
    Should return dest, which can be a new object.
    """
    AFTER_BUILD = enum.auto()

    """Keyed on (CREATE_DEFAULT, class[, Format if available]).
    Receives no arguments.
    Should return a new instance of class.
    """
    CREATE_DEFAULT = enum.auto()


def _assignable(convertersByName):
    return {k: v for k, v in convertersByName.items() if not isinstance(v, ComputedInt)}


def _isNonStrSequence(value):
    return isinstance(value, collections.abc.Sequence) and not isinstance(value, str)


def _split_format(cls, source):
    if _isNonStrSequence(source):
        assert len(source) > 0, f"{cls} needs at least format from {source}"
        fmt, remainder = source[0], source[1:]
    elif isinstance(source, collections.abc.Mapping):
        assert "Format" in source, f"{cls} needs at least Format from {source}"
        remainder = source.copy()
        fmt = remainder.pop("Format")
    else:
        raise ValueError(f"Not sure how to populate {cls} from {source}")

    assert isinstance(
        fmt, collections.abc.Hashable
    ), f"{cls} Format is not hashable: {fmt!r}"
    assert fmt in cls.convertersByName, f"{cls} invalid Format: {fmt!r}"

    return fmt, remainder


class TableBuilder:
    """
    Helps to populate things derived from BaseTable from maps, tuples, etc.

    A table of lifecycle callbacks may be provided to add logic beyond what is possible
    based on otData info for the target class. See BuildCallbacks.
    """

    def __init__(self, callbackTable=None):
        if callbackTable is None:
            callbackTable = {}
        self._callbackTable = callbackTable

    def _convert(self, dest, field, converter, value):
        enumClass = getattr(converter, "enumClass", None)

        if enumClass:
            if isinstance(value, enumClass):
                pass
            elif isinstance(value, str):
                try:
                    value = getattr(enumClass, value.upper())
                except AttributeError:
                    raise ValueError(f"{value} is not a valid {enumClass}")
            else:
                value = enumClass(value)

        elif isinstance(converter, IntValue):
            value = otRound(value)
        elif isinstance(converter, FloatValue):
            value = float(value)

        elif isinstance(converter, Struct):
            if converter.repeat:
                if _isNonStrSequence(value):
                    value = [self.build(converter.tableClass, v) for v in value]
                else:
                    value = [self.build(converter.tableClass, value)]
                setattr(dest, converter.repeat, len(value))
            else:
                value = self.build(converter.tableClass, value)
        elif callable(converter):
            value = converter(value)

        setattr(dest, field, value)

    def build(self, cls, source):
        assert issubclass(cls, BaseTable)

        if isinstance(source, cls):
            return source

        callbackKey = (cls,)
        fmt = None
        if issubclass(cls, FormatSwitchingBaseTable):
            fmt, source = _split_format(cls, source)
            callbackKey = (cls, fmt)

        dest = self._callbackTable.get(
            (BuildCallback.CREATE_DEFAULT,) + callbackKey, lambda: cls()
        )()
        assert isinstance(dest, cls)

        convByName = _assignable(cls.convertersByName)
        skippedFields = set()

        # For format switchers we need to resolve converters based on format
        if issubclass(cls, FormatSwitchingBaseTable):
            dest.Format = fmt
            convByName = _assignable(convByName[dest.Format])
            skippedFields.add("Format")

        # Convert sequence => mapping so before thunk only has to handle one format
        if _isNonStrSequence(source):
            # Sequence (typically list or tuple) assumed to match fields in declaration order
            assert len(source) <= len(
                convByName
            ), f"Sequence of {len(source)} too long for {cls}; expected <= {len(convByName)} values"
            source = dict(zip(convByName.keys(), source))

        dest, source = self._callbackTable.get(
            (BuildCallback.BEFORE_BUILD,) + callbackKey, lambda d, s: (d, s)
        )(dest, source)

        if isinstance(source, collections.abc.Mapping):
            for field, value in source.items():
                if field in skippedFields:
                    continue
                converter = convByName.get(field, None)
                if not converter:
                    raise ValueError(
                        f"Unrecognized field {field} for {cls}; expected one of {sorted(convByName.keys())}"
                    )
                self._convert(dest, field, converter, value)
        else:
            # let's try as a 1-tuple
            dest = self.build(cls, (source,))

        for field, conv in convByName.items():
            if not hasattr(dest, field) and isinstance(conv, OptionalValue):
                setattr(dest, field, conv.DEFAULT)

        dest = self._callbackTable.get(
            (BuildCallback.AFTER_BUILD,) + callbackKey, lambda d: d
        )(dest)

        return dest


class TableUnbuilder:
    def __init__(self, callbackTable=None):
        if callbackTable is None:
            callbackTable = {}
        self._callbackTable = callbackTable

    def unbuild(self, table):
        assert isinstance(table, BaseTable)

        source = {}

        callbackKey = (type(table),)
        if isinstance(table, FormatSwitchingBaseTable):
            source["Format"] = int(table.Format)
            callbackKey += (table.Format,)

        for converter in table.getConverters():
            if isinstance(converter, ComputedInt):
                continue
            value = getattr(table, converter.name)

            enumClass = getattr(converter, "enumClass", None)
            if enumClass:
                source[converter.name] = value.name.lower()
            elif isinstance(converter, Struct):
                if converter.repeat:
                    source[converter.name] = [self.unbuild(v) for v in value]
                else:
                    source[converter.name] = self.unbuild(value)
            elif isinstance(converter, SimpleValue):
                # "simple" values (e.g. int, float, str) need no further un-building
                source[converter.name] = value
            else:
                raise NotImplementedError(
                    "Don't know how unbuild {value!r} with {converter!r}"
                )

        source = self._callbackTable.get(callbackKey, lambda s: s)(source)

        return source
<|endoftext|>from fontTools.ttLib.tables import otTables as ot
from .table_builder import TableUnbuilder


def unbuildColrV1(layerList, baseGlyphList):
    layers = []
    if layerList:
        layers = layerList.Paint
    unbuilder = LayerListUnbuilder(layers)
    return {
        rec.BaseGlyph: unbuilder.unbuildPaint(rec.Paint)
        for rec in baseGlyphList.BaseGlyphPaintRecord
    }


def _flatten_layers(lst):
    for paint in lst:
        if paint["Format"] == ot.PaintFormat.PaintColrLayers:
            yield from _flatten_layers(paint["Layers"])
        else:
            yield paint


class LayerListUnbuilder:
    def __init__(self, layers):
        self.layers = layers

        callbacks = {
            (
                ot.Paint,
                ot.PaintFormat.PaintColrLayers,
            ): self._unbuildPaintColrLayers,
        }
        self.tableUnbuilder = TableUnbuilder(callbacks)

    def unbuildPaint(self, paint):
        assert isinstance(paint, ot.Paint)
        return self.tableUnbuilder.unbuild(paint)

    def _unbuildPaintColrLayers(self, source):
        assert source["Format"] == ot.PaintFormat.PaintColrLayers

        layers = list(
            _flatten_layers(
                [
                    self.unbuildPaint(childPaint)
                    for childPaint in self.layers[
                        source["FirstLayerIndex"] : source["FirstLayerIndex"]
                        + source["NumLayers"]
                    ]
                ]
            )
        )

        if len(layers) == 1:
            return layers[0]

        return {"Format": source["Format"], "Layers": layers}


if __name__ == "__main__":
    from pprint import pprint
    import sys
    from fontTools.ttLib import TTFont

    try:
        fontfile = sys.argv[1]
    except IndexError:
        sys.exit("usage: fonttools colorLib.unbuilder FONTFILE")

    font = TTFont(fontfile)
    colr = font["COLR"]
    if colr.version < 1:
        sys.exit(f"error: No COLR table version=1 found in {fontfile}")

    colorGlyphs = unbuildColrV1(
        colr.table.LayerList,
        colr.table.BaseGlyphList,
    )

    pprint(colorGlyphs)
<|endoftext|>"""Helpers for manipulating 2D points and vectors in COLR table."""

from math import copysign, cos, hypot, isclose, pi
from fontTools.misc.roundTools import otRound


def _vector_between(origin, target):
    return (target[0] - origin[0], target[1] - origin[1])


def _round_point(pt):
    return (otRound(pt[0]), otRound(pt[1]))


def _unit_vector(vec):
    length = hypot(*vec)
    if length == 0:
        return None
    return (vec[0] / length, vec[1] / length)


_CIRCLE_INSIDE_TOLERANCE = 1e-4


# The unit vector's X and Y components are respectively
#   U = (cos(α), sin(α))
# where α is the angle between the unit vector and the positive x axis.
_UNIT_VECTOR_THRESHOLD = cos(3 / 8 * pi)  # == sin(1/8 * pi) == 0.38268343236508984


def _rounding_offset(direction):
    # Return 2-tuple of -/+ 1.0 or 0.0 approximately based on the direction vector.
    # We divide the unit circle in 8 equal slices oriented towards the cardinal
    # (N, E, S, W) and intermediate (NE, SE, SW, NW) directions. To each slice we
    # map one of the possible cases: -1, 0, +1 for either X and Y coordinate.
    # E.g. Return (+1.0, -1.0) if unit vector is oriented towards SE, or
    # (-1.0, 0.0) if it's pointing West, etc.
    uv = _unit_vector(direction)
    if not uv:
        return (0, 0)

    result = []
    for uv_component in uv:
        if -_UNIT_VECTOR_THRESHOLD <= uv_component < _UNIT_VECTOR_THRESHOLD:
            # unit vector component near 0: direction almost orthogonal to the
            # direction of the current axis, thus keep coordinate unchanged
            result.append(0)
        else:
            # nudge coord by +/- 1.0 in direction of unit vector
            result.append(copysign(1.0, uv_component))
    return tuple(result)


class Circle:
    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius

    def __repr__(self):
        return f"Circle(centre={self.centre}, radius={self.radius})"

    def round(self):
        return Circle(_round_point(self.centre), otRound(self.radius))

    def inside(self, outer_circle, tolerance=_CIRCLE_INSIDE_TOLERANCE):
        dist = self.radius + hypot(*_vector_between(self.centre, outer_circle.centre))
        return (
            isclose(outer_circle.radius, dist, rel_tol=_CIRCLE_INSIDE_TOLERANCE)
            or outer_circle.radius > dist
        )

    def concentric(self, other):
        return self.centre == other.centre

    def move(self, dx, dy):
        self.centre = (self.centre[0] + dx, self.centre[1] + dy)


def round_start_circle_stable_containment(c0, r0, c1, r1):
    """Round start circle so that it stays inside/outside end circle after rounding.

    The rounding of circle coordinates to integers may cause an abrupt change
    if the start circle c0 is so close to the end circle c1's perimiter that
    it ends up falling outside (or inside) as a result of the rounding.
    To keep the gradient unchanged, we nudge it in the right direction.

    See:
    https://github.com/googlefonts/colr-gradients-spec/issues/204
    https://github.com/googlefonts/picosvg/issues/158
    """
    start, end = Circle(c0, r0), Circle(c1, r1)

    inside_before_round = start.inside(end)

    round_start = start.round()
    round_end = end.round()
    inside_after_round = round_start.inside(round_end)

    if inside_before_round == inside_after_round:
        return round_start
    elif inside_after_round:
        # start was outside before rounding: we need to push start away from end
        direction = _vector_between(round_end.centre, round_start.centre)
        radius_delta = +1.0
    else:
        # start was inside before rounding: we need to push start towards end
        direction = _vector_between(round_start.centre, round_end.centre)
        radius_delta = -1.0
    dx, dy = _rounding_offset(direction)

    # At most 2 iterations ought to be enough to converge. Before the loop, we
    # know the start circle didn't keep containment after normal rounding; thus
    # we continue adjusting by -/+ 1.0 until containment is restored.
    # Normal rounding can at most move each coordinates -/+0.5; in the worst case
    # both the start and end circle's centres and radii will be rounded in opposite
    # directions, e.g. when they move along a 45 degree diagonal:
    #   c0 = (1.5, 1.5) ===> (2.0, 2.0)
    #   r0 = 0.5 ===> 1.0
    #   c1 = (0.499, 0.499) ===> (0.0, 0.0)
    #   r1 = 2.499 ===> 2.0
    # In this example, the relative distance between the circles, calculated
    # as r1 - (r0 + distance(c0, c1)) is initially 0.57437 (c0 is inside c1), and
    # -1.82842 after rounding (c0 is now outside c1). Nudging c0 by -1.0 on both
    # x and y axes moves it towards c1 by hypot(-1.0, -1.0) = 1.41421. Two of these
    # moves cover twice that distance, which is enough to restore containment.
    max_attempts = 2
    for _ in range(max_attempts):
        if round_start.concentric(round_end):
            # can't move c0 towards c1 (they are the same), so we change the radius
            round_start.radius += radius_delta
            assert round_start.radius >= 0
        else:
            round_start.move(dx, dy)
        if inside_before_round == round_start.inside(round_end):
            break
    else:  # likely a bug
        raise AssertionError(
            f"Rounding circle {start} "
            f"{'inside' if inside_before_round else 'outside'} "
            f"{end} failed after {max_attempts} attempts!"
        )

    return round_start
<|endoftext|>"""fontTools.t1Lib.py -- Tools for PostScript Type 1 fonts.

Functions for reading and writing raw Type 1 data:

read(path)
        reads any Type 1 font file, returns the raw data and a type indicator:
        'LWFN', 'PFB' or 'OTHER', depending on the format of the file pointed
        to by 'path'.
        Raises an error when the file does not contain valid Type 1 data.

write(path, data, kind='OTHER', dohex=False)
        writes raw Type 1 data to the file pointed to by 'path'.
        'kind' can be one of 'LWFN', 'PFB' or 'OTHER'; it defaults to 'OTHER'.
        'dohex' is a flag which determines whether the eexec encrypted
        part should be written as hexadecimal or binary, but only if kind
        is 'OTHER'.
"""

import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
    _type1_pre_eexec_order,
    _type1_fontinfo_order,
    _type1_post_eexec_order,
)
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re

__author__ = "jvr"
__version__ = "1.0b3"
DEBUG = 0


try:
    try:
        from Carbon import Res
    except ImportError:
        import Res  # MacPython < 2.2
except ImportError:
    haveMacSupport = 0
else:
    haveMacSupport = 1


class T1Error(Exception):
    pass


class T1Font(object):
    """Type 1 font class.

    Uses a minimal interpeter that supports just about enough PS to parse
    Type 1 fonts.
    """

    def __init__(self, path, encoding="ascii", kind=None):
        if kind is None:
            self.data, _ = read(path)
        elif kind == "LWFN":
            self.data = readLWFN(path)
        elif kind == "PFB":
            self.data = readPFB(path)
        elif kind == "OTHER":
            self.data = readOther(path)
        else:
            raise ValueError(kind)
        self.encoding = encoding

    def saveAs(self, path, type, dohex=False):
        write(path, self.getData(), type, dohex)

    def getData(self):
        if not hasattr(self, "data"):
            self.data = self.createData()
        return self.data

    def getGlyphSet(self):
        """Return a generic GlyphSet, which is a dict-like object
        mapping glyph names to glyph objects. The returned glyph objects
        have a .draw() method that supports the Pen protocol, and will
        have an attribute named 'width', but only *after* the .draw() method
        has been called.

        In the case of Type 1, the GlyphSet is simply the CharStrings dict.
        """
        return self["CharStrings"]

    def __getitem__(self, key):
        if not hasattr(self, "font"):
            self.parse()
        return self.font[key]

    def parse(self):
        from fontTools.misc import psLib
        from fontTools.misc import psCharStrings

        self.font = psLib.suckfont(self.data, self.encoding)
        charStrings = self.font["CharStrings"]
        lenIV = self.font["Private"].get("lenIV", 4)
        assert lenIV >= 0
        subrs = self.font["Private"]["Subrs"]
        for glyphName, charString in charStrings.items():
            charString, R = eexec.decrypt(charString, 4330)
            charStrings[glyphName] = psCharStrings.T1CharString(
                charString[lenIV:], subrs=subrs
            )
        for i in range(len(subrs)):
            charString, R = eexec.decrypt(subrs[i], 4330)
            subrs[i] = psCharStrings.T1CharString(charString[lenIV:], subrs=subrs)
        del self.data

    def createData(self):
        sf = self.font

        eexec_began = False
        eexec_dict = {}
        lines = []
        lines.extend(
            [
                self._tobytes(f"%!FontType1-1.1: {sf['FontName']}"),
                self._tobytes(f"%t1Font: ({fontTools.version})"),
                self._tobytes(f"%%BeginResource: font {sf['FontName']}"),
            ]
        )
        # follow t1write.c:writeRegNameKeyedFont
        size = 3  # Headroom for new key addition
        size += 1  # FontMatrix is always counted
        size += 1 + 1  # Private, CharStings
        for key in font_dictionary_keys:
            size += int(key in sf)
        lines.append(self._tobytes(f"{size} dict dup begin"))

        for key, value in sf.items():
            if eexec_began:
                eexec_dict[key] = value
                continue

            if key == "FontInfo":
                fi = sf["FontInfo"]
                # follow t1write.c:writeFontInfoDict
                size = 3  # Headroom for new key addition
                for subkey in FontInfo_dictionary_keys:
                    size += int(subkey in fi)
                lines.append(self._tobytes(f"/FontInfo {size} dict dup begin"))

                for subkey, subvalue in fi.items():
                    lines.extend(self._make_lines(subkey, subvalue))
                lines.append(b"end def")
            elif key in _type1_post_eexec_order:  # usually 'Private'
                eexec_dict[key] = value
                eexec_began = True
            else:
                lines.extend(self._make_lines(key, value))
        lines.append(b"end")
        eexec_portion = self.encode_eexec(eexec_dict)
        lines.append(bytesjoin([b"currentfile eexec ", eexec_portion]))

        for _ in range(8):
            lines.append(self._tobytes("0" * 64))
        lines.extend([b"cleartomark", b"%%EndResource", b"%%EOF"])

        data = bytesjoin(lines, "\n")
        return data

    def encode_eexec(self, eexec_dict):
        lines = []

        # '-|', '|-', '|'
        RD_key, ND_key, NP_key = None, None, None
        lenIV = 4
        subrs = std_subrs

        # Ensure we look at Private first, because we need RD_key, ND_key, NP_key and lenIV
        sortedItems = sorted(eexec_dict.items(), key=lambda item: item[0] != "Private")

        for key, value in sortedItems:
            if key == "Private":
                pr = eexec_dict["Private"]
                # follow t1write.c:writePrivateDict
                size = 3  # for RD, ND, NP
                for subkey in Private_dictionary_keys:
                    size += int(subkey in pr)
                lines.append(b"dup /Private")
                lines.append(self._tobytes(f"{size} dict dup begin"))
                for subkey, subvalue in pr.items():
                    if not RD_key and subvalue == RD_value:
                        RD_key = subkey
                    elif not ND_key and subvalue in ND_values:
                        ND_key = subkey
                    elif not NP_key and subvalue in PD_values:
                        NP_key = subkey

                    if subkey == "lenIV":
                        lenIV = subvalue

                    if subkey == "OtherSubrs":
                        # XXX: assert that no flex hint is used
                        lines.append(self._tobytes(hintothers))
                    elif subkey == "Subrs":
                        for subr_bin in subvalue:
                            subr_bin.compile()
                        subrs = [subr_bin.bytecode for subr_bin in subvalue]
                        lines.append(f"/Subrs {len(subrs)} array".encode("ascii"))
                        for i, subr_bin in enumerate(subrs):
                            encrypted_subr, R = eexec.encrypt(
                                bytesjoin([char_IV[:lenIV], subr_bin]), 4330
                            )
                            lines.append(
                                bytesjoin(
                                    [
                                        self._tobytes(
                                            f"dup {i} {len(encrypted_subr)} {RD_key} "
                                        ),
                                        encrypted_subr,
                                        self._tobytes(f" {NP_key}"),
                                    ]
                                )
                            )
                        lines.append(b"def")

                        lines.append(b"put")
                    else:
                        lines.extend(self._make_lines(subkey, subvalue))
            elif key == "CharStrings":
                lines.append(b"dup /CharStrings")
                lines.append(
                    self._tobytes(f"{len(eexec_dict['CharStrings'])} dict dup begin")
                )
                for glyph_name, char_bin in eexec_dict["CharStrings"].items():
                    char_bin.compile()
                    encrypted_char, R = eexec.encrypt(
                        bytesjoin([char_IV[:lenIV], char_bin.bytecode]), 4330
                    )
                    lines.append(
                        bytesjoin(
                            [
                                self._tobytes(
                                    f"/{glyph_name} {len(encrypted_char)} {RD_key} "
                                ),
                                encrypted_char,
                                self._tobytes(f" {ND_key}"),
                            ]
                        )
                    )
                lines.append(b"end put")
            else:
                lines.extend(self._make_lines(key, value))

        lines.extend(
            [
                b"end",
                b"dup /FontName get exch definefont pop",
                b"mark",
                b"currentfile closefile\n",
            ]
        )

        eexec_portion = bytesjoin(lines, "\n")
        encrypted_eexec, R = eexec.encrypt(bytesjoin([eexec_IV, eexec_portion]), 55665)

        return encrypted_eexec

    def _make_lines(self, key, value):
        if key == "FontName":
            return [self._tobytes(f"/{key} /{value} def")]
        if key in ["isFixedPitch", "ForceBold", "RndStemUp"]:
            return [self._tobytes(f"/{key} {'true' if value else 'false'} def")]
        elif key == "Encoding":
            if value == StandardEncoding:
                return [self._tobytes(f"/{key} StandardEncoding def")]
            else:
                # follow fontTools.misc.psOperators._type1_Encoding_repr
                lines = []
                lines.append(b"/Encoding 256 array")
                lines.append(b"0 1 255 {1 index exch /.notdef put} for")
                for i in range(256):
                    name = value[i]
                    if name != ".notdef":
                        lines.append(self._tobytes(f"dup {i} /{name} put"))
                lines.append(b"def")
                return lines
        if isinstance(value, str):
            return [self._tobytes(f"/{key} ({value}) def")]
        elif isinstance(value, bool):
            return [self._tobytes(f"/{key} {'true' if value else 'false'} def")]
        elif isinstance(value, list):
            return [self._tobytes(f"/{key} [{' '.join(str(v) for v in value)}] def")]
        elif isinstance(value, tuple):
            return [self._tobytes(f"/{key} {{{' '.join(str(v) for v in value)}}} def")]
        else:
            return [self._tobytes(f"/{key} {value} def")]

    def _tobytes(self, s, errors="strict"):
        return tobytes(s, self.encoding, errors)


# low level T1 data read and write functions


def read(path, onlyHeader=False):
    """reads any Type 1 font file, returns raw data"""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    creator, typ = getMacCreatorAndType(path)
    if typ == "LWFN":
        return readLWFN(path, onlyHeader), "LWFN"
    if ext == ".pfb":
        return readPFB(path, onlyHeader), "PFB"
    else:
        return readOther(path), "OTHER"


def write(path, data, kind="OTHER", dohex=False):
    assertType1(data)
    kind = kind.upper()
    try:
        os.remove(path)
    except os.error:
        pass
    err = 1
    try:
        if kind == "LWFN":
            writeLWFN(path, data)
        elif kind == "PFB":
            writePFB(path, data)
        else:
            writeOther(path, data, dohex)
        err = 0
    finally:
        if err and not DEBUG:
            try:
                os.remove(path)
            except os.error:
                pass


# -- internal --

LWFNCHUNKSIZE = 2000
HEXLINELENGTH = 80


def readLWFN(path, onlyHeader=False):
    """reads an LWFN font file, returns raw data"""
    from fontTools.misc.macRes import ResourceReader

    reader = ResourceReader(path)
    try:
        data = []
        for res in reader.get("POST", []):
            code = byteord(res.data[0])
            if byteord(res.data[1]) != 0:
                raise T1Error("corrupt LWFN file")
            if code in [1, 2]:
                if onlyHeader and code == 2:
                    break
                data.append(res.data[2:])
            elif code in [3, 5]:
                break
            elif code == 4:
                with open(path, "rb") as f:
                    data.append(f.read())
            elif code == 0:
                pass  # comment, ignore
            else:
                raise T1Error("bad chunk code: " + repr(code))
    finally:
        reader.close()
    data = bytesjoin(data)
    assertType1(data)
    return data


def readPFB(path, onlyHeader=False):
    """reads a PFB font file, returns raw data"""
    data = []
    with open(path, "rb") as f:
        while True:
            if f.read(1) != bytechr(128):
                raise T1Error("corrupt PFB file")
            code = byteord(f.read(1))
            if code in [1, 2]:
                chunklen = stringToLong(f.read(4))
                chunk = f.read(chunklen)
                assert len(chunk) == chunklen
                data.append(chunk)
            elif code == 3:
                break
            else:
                raise T1Error("bad chunk code: " + repr(code))
            if onlyHeader:
                break
    data = bytesjoin(data)
    assertType1(data)
    return data


def readOther(path):
    """reads any (font) file, returns raw data"""
    with open(path, "rb") as f:
        data = f.read()
    assertType1(data)
    chunks = findEncryptedChunks(data)
    data = []
    for isEncrypted, chunk in chunks:
        if isEncrypted and isHex(chunk[:4]):
            data.append(deHexString(chunk))
        else:
            data.append(chunk)
    return bytesjoin(data)


# file writing tools


def writeLWFN(path, data):
    # Res.FSpCreateResFile was deprecated in OS X 10.5
    Res.FSpCreateResFile(path, "just", "LWFN", 0)
    resRef = Res.FSOpenResFile(path, 2)  # write-only
    try:
        Res.UseResFile(resRef)
        resID = 501
        chunks = findEncryptedChunks(data)
        for isEncrypted, chunk in chunks:
            if isEncrypted:
                code = 2
            else:
                code = 1
            while chunk:
                res = Res.Resource(bytechr(code) + "\0" + chunk[: LWFNCHUNKSIZE - 2])
                res.AddResource("POST", resID, "")
                chunk = chunk[LWFNCHUNKSIZE - 2 :]
                resID = resID + 1
        res = Res.Resource(bytechr(5) + "\0")
        res.AddResource("POST", resID, "")
    finally:
        Res.CloseResFile(resRef)


def writePFB(path, data):
    chunks = findEncryptedChunks(data)
    with open(path, "wb") as f:
        for isEncrypted, chunk in chunks:
            if isEncrypted:
                code = 2
            else:
                code = 1
            f.write(bytechr(128) + bytechr(code))
            f.write(longToString(len(chunk)))
            f.write(chunk)
        f.write(bytechr(128) + bytechr(3))


def writeOther(path, data, dohex=False):
    chunks = findEncryptedChunks(data)
    with open(path, "wb") as f:
        hexlinelen = HEXLINELENGTH // 2
        for isEncrypted, chunk in chunks:
            if isEncrypted:
                code = 2
            else:
                code = 1
            if code == 2 and dohex:
                while chunk:
                    f.write(eexec.hexString(chunk[:hexlinelen]))
                    f.write(b"\r")
                    chunk = chunk[hexlinelen:]
            else:
                f.write(chunk)


# decryption tools

EEXECBEGIN = b"currentfile eexec"
# The spec allows for 512 ASCII zeros interrupted by arbitrary whitespace to
# follow eexec
EEXECEND = re.compile(b"(0[ \t\r\n]*){512}", flags=re.M)
EEXECINTERNALEND = b"currentfile closefile"
EEXECBEGINMARKER = b"%-- eexec start\r"
EEXECENDMARKER = b"%-- eexec end\r"

_ishexRE = re.compile(b"[0-9A-Fa-f]*$")


def isHex(text):
    return _ishexRE.match(text) is not None


def decryptType1(data):
    chunks = findEncryptedChunks(data)
    data = []
    for isEncrypted, chunk in chunks:
        if isEncrypted:
            if isHex(chunk[:4]):
                chunk = deHexString(chunk)
            decrypted, R = eexec.decrypt(chunk, 55665)
            decrypted = decrypted[4:]
            if (
                decrypted[-len(EEXECINTERNALEND) - 1 : -1] != EEXECINTERNALEND
                and decrypted[-len(EEXECINTERNALEND) - 2 : -2] != EEXECINTERNALEND
            ):
                raise T1Error("invalid end of eexec part")
            decrypted = decrypted[: -len(EEXECINTERNALEND) - 2] + b"\r"
            data.append(EEXECBEGINMARKER + decrypted + EEXECENDMARKER)
        else:
            if chunk[-len(EEXECBEGIN) - 1 : -1] == EEXECBEGIN:
                data.append(chunk[: -len(EEXECBEGIN) - 1])
            else:
                data.append(chunk)
    return bytesjoin(data)


def findEncryptedChunks(data):
    chunks = []
    while True:
        eBegin = data.find(EEXECBEGIN)
        if eBegin < 0:
            break
        eBegin = eBegin + len(EEXECBEGIN) + 1
        endMatch = EEXECEND.search(data, eBegin)
        if endMatch is None:
            raise T1Error("can't find end of eexec part")
        eEnd = endMatch.start()
        cypherText = data[eBegin : eEnd + 2]
        if isHex(cypherText[:4]):
            cypherText = deHexString(cypherText)
        plainText, R = eexec.decrypt(cypherText, 55665)
        eEndLocal = plainText.find(EEXECINTERNALEND)
        if eEndLocal < 0:
            raise T1Error("can't find end of eexec part")
        chunks.append((0, data[:eBegin]))
        chunks.append((1, cypherText[: eEndLocal + len(EEXECINTERNALEND) + 1]))
        data = data[eEnd:]
    chunks.append((0, data))
    return chunks


def deHexString(hexstring):
    return eexec.deHexString(bytesjoin(hexstring.split()))


# Type 1 assertion

_fontType1RE = re.compile(rb"/FontType\s+1\s+def")


def assertType1(data):
    for head in [b"%!PS-AdobeFont", b"%!FontType1"]:
        if data[: len(head)] == head:
            break
    else:
        raise T1Error("not a PostScript font")
    if not _fontType1RE.search(data):
        raise T1Error("not a Type 1 font")
    if data.find(b"currentfile eexec") < 0:
        raise T1Error("not an encrypted Type 1 font")
    # XXX what else?
    return data


# pfb helpers


def longToString(long):
    s = b""
    for i in range(4):
        s += bytechr((long & (0xFF << (i * 8))) >> i * 8)
    return s


def stringToLong(s):
    if len(s) != 4:
        raise ValueError("string must be 4 bytes long")
    l = 0
    for i in range(4):
        l += byteord(s[i]) << (i * 8)
    return l


# PS stream helpers

font_dictionary_keys = list(_type1_pre_eexec_order)
# t1write.c:writeRegNameKeyedFont
# always counts following keys
font_dictionary_keys.remove("FontMatrix")

FontInfo_dictionary_keys = list(_type1_fontinfo_order)
# extend because AFDKO tx may use following keys
FontInfo_dictionary_keys.extend(
    [
        "FSType",
        "Copyright",
    ]
)

Private_dictionary_keys = [
    # We don't know what names will be actually used.
    # "RD",
    # "ND",
    # "NP",
    "Subrs",
    "OtherSubrs",
    "UniqueID",
    "BlueValues",
    "OtherBlues",
    "FamilyBlues",
    "FamilyOtherBlues",
    "BlueScale",
    "BlueShift",
    "BlueFuzz",
    "StdHW",
    "StdVW",
    "StemSnapH",
    "StemSnapV",
    "ForceBold",
    "LanguageGroup",
    "password",
    "lenIV",
    "MinFeature",
    "RndStemUp",
]

# t1write_hintothers.h
hintothers = """/OtherSubrs[{}{}{}{systemdict/internaldict known not{pop 3}{1183615869
systemdict/internaldict get exec dup/startlock known{/startlock get exec}{dup
/strtlck known{/strtlck get exec}{pop 3}ifelse}ifelse}ifelse}executeonly]def"""
# t1write.c:saveStdSubrs
std_subrs = [
    # 3 0 callother pop pop setcurrentpoint return
    b"\x8e\x8b\x0c\x10\x0c\x11\x0c\x11\x0c\x21\x0b",
    # 0 1 callother return
    b"\x8b\x8c\x0c\x10\x0b",
    # 0 2 callother return
    b"\x8b\x8d\x0c\x10\x0b",
    # return
    b"\x0b",
    # 3 1 3 callother pop callsubr return
    b"\x8e\x8c\x8e\x0c\x10\x0c\x11\x0a\x0b",
]
# follow t1write.c:writeRegNameKeyedFont
eexec_IV = b"cccc"
char_IV = b"\x0c\x0c\x0c\x0c"
RD_value = ("string", "currentfile", "exch", "readstring", "pop")
ND_values = [("def",), ("noaccess", "def")]
PD_values = [("put",), ("noaccess", "put")]
<|endoftext|>"""Extend the Python codecs module with a few encodings that are used in OpenType (name table)
but missing from Python.  See https://github.com/fonttools/fonttools/issues/236 for details.
"""

import codecs
import encodings


class ExtendCodec(codecs.Codec):
    def __init__(self, name, base_encoding, mapping):
        self.name = name
        self.base_encoding = base_encoding
        self.mapping = mapping
        self.reverse = {v: k for k, v in mapping.items()}
        self.max_len = max(len(v) for v in mapping.values())
        self.info = codecs.CodecInfo(
            name=self.name, encode=self.encode, decode=self.decode
        )
        codecs.register_error(name, self.error)

    def _map(self, mapper, output_type, exc_type, input, errors):
        base_error_handler = codecs.lookup_error(errors)
        length = len(input)
        out = output_type()
        while input:
            # first try to use self.error as the error handler
            try:
                part = mapper(input, self.base_encoding, errors=self.name)
                out += part
                break  # All converted
            except exc_type as e:
                # else convert the correct part, handle error as requested and continue
                out += mapper(input[: e.start], self.base_encoding, self.name)
                replacement, pos = base_error_handler(e)
                out += replacement
                input = input[pos:]
        return out, length

    def encode(self, input, errors="strict"):
        return self._map(codecs.encode, bytes, UnicodeEncodeError, input, errors)

    def decode(self, input, errors="strict"):
        return self._map(codecs.decode, str, UnicodeDecodeError, input, errors)

    def error(self, e):
        if isinstance(e, UnicodeDecodeError):
            for end in range(e.start + 1, e.end + 1):
                s = e.object[e.start : end]
                if s in self.mapping:
                    return self.mapping[s], end
        elif isinstance(e, UnicodeEncodeError):
            for end in range(e.start + 1, e.start + self.max_len + 1):
                s = e.object[e.start : end]
                if s in self.reverse:
                    return self.reverse[s], end
        e.encoding = self.name
        raise e


_extended_encodings = {
    "x_mac_japanese_ttx": (
        "shift_jis",
        {
            b"\xfc": chr(0x007C),
            b"\x7e": chr(0x007E),
            b"\x80": chr(0x005C),
            b"\xa0": chr(0x00A0),
            b"\xfd": chr(0x00A9),
            b"\xfe": chr(0x2122),
            b"\xff": chr(0x2026),
        },
    ),
    "x_mac_trad_chinese_ttx": (
        "big5",
        {
            b"\x80": chr(0x005C),
            b"\xa0": chr(0x00A0),
            b"\xfd": chr(0x00A9),
            b"\xfe": chr(0x2122),
            b"\xff": chr(0x2026),
        },
    ),
    "x_mac_korean_ttx": (
        "euc_kr",
        {
            b"\x80": chr(0x00A0),
            b"\x81": chr(0x20A9),
            b"\x82": chr(0x2014),
            b"\x83": chr(0x00A9),
            b"\xfe": chr(0x2122),
            b"\xff": chr(0x2026),
        },
    ),
    "x_mac_simp_chinese_ttx": (
        "gb2312",
        {
            b"\x80": chr(0x00FC),
            b"\xa0": chr(0x00A0),
            b"\xfd": chr(0x00A9),
            b"\xfe": chr(0x2122),
            b"\xff": chr(0x2026),
        },
    ),
}

_cache = {}


def search_function(name):
    name = encodings.normalize_encoding(name)  # Rather undocumented...
    if name in _extended_encodings:
        if name not in _cache:
            base_encoding, mapping = _extended_encodings[name]
            assert name[-4:] == "_ttx"
            # Python 2 didn't have any of the encodings that we are implementing
            # in this file.  Python 3 added aliases for the East Asian ones, mapping
            # them "temporarily" to the same base encoding as us, with a comment
            # suggesting that full implementation will appear some time later.
            # As such, try the Python version of the x_mac_... first, if that is found,
            # use *that* as our base encoding.  This would make our encoding upgrade
            # to the full encoding when and if Python finally implements that.
            # http://bugs.python.org/issue24041
            base_encodings = [name[:-4], base_encoding]
            for base_encoding in base_encodings:
                try:
                    codecs.lookup(base_encoding)
                except LookupError:
                    continue
                _cache[name] = ExtendCodec(name, base_encoding, mapping)
                break
        return _cache[name].info

    return None


codecs.register(search_function)
<|endoftext|>from typing import Dict, Text

ansicolors: Dict[Text, Text] = {
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "BOLD": "\033[1m",
    "RESET": "\033[0m",
}

green_start: Text = ansicolors["GREEN"]
red_start: Text = ansicolors["RED"]
cyan_start: Text = ansicolors["CYAN"]
reset: Text = ansicolors["RESET"]


def color_unified_diff_line(line: Text) -> Text:
    """Returns an ANSI escape code colored string with color based
    on the unified diff line type."""
    if line[0:2] == "+ ":
        return f"{green_start}{line}{reset}"
    elif line == "+\n":
        # some lines are formatted as hyphen only with no other characters
        # this indicates an added empty line
        return f"{green_start}{line}{reset}"
    elif line[0:2] == "- ":
        return f"{red_start}{line}{reset}"
    elif line == "-\n":
        # some lines are formatted as hyphen only with no other characters
        # this indicates a deleted empty line
        return f"{red_start}{line}{reset}"
    elif line[0:3] == "@@ ":
        return f"{cyan_start}{line}{reset}"
    elif line[0:4] == "--- ":
        return f"{red_start}{line}{reset}"
    elif line[0:4] == "+++ ":
        return f"{green_start}{line}{reset}"
    else:
        return line
<|endoftext|>import os
from datetime import datetime, timezone
from typing import List, Optional, Text, Union


def file_exists(path: Union[bytes, str, "os.PathLike[Text]"]) -> bool:
    """Validates file path as existing local file"""
    return os.path.isfile(path)


def get_file_modtime(path: Union[bytes, str, "os.PathLike[Text]"]) -> Text:
    """Returns ISO formatted file modification time in local system timezone"""
    return (
        datetime.fromtimestamp(os.stat(path).st_mtime, timezone.utc)
        .astimezone()
        .isoformat()
    )


def get_tables_argument_list(table_list: Optional[List[Text]]) -> Optional[List[Text]]:
    """Converts a list of OpenType table string into a Python list or
    return None if the table_list was not defined (i.e., it was not included
    in an option on the command line). Tables that are composed of three
    characters must be right padded with a space."""
    if table_list is None:
        return None
    else:
        return [table.ljust(4) for table in table_list]
<|endoftext|>import argparse
import os
import sys
import shutil
import subprocess
from typing import Iterable, Iterator, List, Optional, Text, Tuple

from .color import color_unified_diff_line
from .diff import run_external_diff, u_diff
from .utils import file_exists, get_tables_argument_list


def pipe_output(output: str) -> None:
    """Pipes output to a pager if stdout is a TTY and a pager is available."""

    if not output:
        return

    if not sys.stdout.isatty():
        sys.stdout.write(output)
        return

    pager = os.getenv("PAGER") or shutil.which("less")

    if not pager:
        sys.stdout.write(output)
        return

    pager_cmd = [pager]
    if "less" in os.path.basename(pager):
        pager_cmd.append("-R")

    proc = subprocess.Popen(pager_cmd, stdin=subprocess.PIPE, text=True)
    try:
        proc.stdin.write(output)
        proc.stdin.close()
        proc.wait()
    except (BrokenPipeError, KeyboardInterrupt):
        # Pager process was terminated before all output was written.
        # This is not an error. The main exception handler will deal with it.
        if proc.stdin:
            proc.stdin.close()
        # The process might still be running, but we have closed our side of the
        # pipe. The Popen destructor will send a SIGKILL to the child.
    except Exception:
        if proc.stdin:
            proc.stdin.close()
        raise


def _is_gnu_diff(diff_tool: str) -> bool:
    """Returns True if the provided diff executable is GNU diff."""
    try:
        proc = subprocess.run(
            [diff_tool, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return False

    version_output = (proc.stdout or "") + (proc.stderr or "")
    return "GNU diffutils" in version_output


def _iter_filtered_table_tags(
    tags: Iterable[str],
    include_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None,
) -> Iterator[str]:
    for tag in tags:
        if exclude_tables and tag in exclude_tables:
            continue
        if include_tables and tag not in include_tables:
            continue
        yield tag


def summarize(
    file1: str,
    file2: str,
    include_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None,
    font_number_1: int = -1,
    font_number_2: int = -1,
) -> Tuple[bool, str]:
    from fontTools.ttLib import TTFont

    with (
        TTFont(file1, lazy=True, fontNumber=font_number_1) as font1,
        TTFont(file2, lazy=True, fontNumber=font_number_2) as font2,
    ):
        tags1 = {str(tag) for tag in font1.reader.keys()}
        tags2 = {str(tag) for tag in font2.reader.keys()}

        all_tags = sorted(
            set(
                _iter_filtered_table_tags(
                    tags1 | tags2,
                    include_tables=include_tables,
                    exclude_tables=exclude_tables,
                )
            )
        )

        only1 = [tag for tag in all_tags if tag in tags1 and tag not in tags2]
        only2 = [tag for tag in all_tags if tag in tags2 and tag not in tags1]
        both = [tag for tag in all_tags if tag in tags1 and tag in tags2]

        identical = True
        lines: List[str] = []

        lines.append(f"Binary table summary:\n")
        lines.append(f"  file1: {file1}\n")
        lines.append(f"  file2: {file2}\n")

        if only1:
            identical = False
            lines.append(f"\nTables only in file1 ({len(only1)}):\n")
            for tag in only1:
                lines.append(f"- {tag} ({len(font1.reader[tag])} bytes)\n")
        if only2:
            identical = False
            lines.append(f"\nTables only in file2 ({len(only2)}):\n")
            for tag in only2:
                lines.append(f"+ {tag} ({len(font2.reader[tag])} bytes)\n")

        lines.append(f"\nTables in both ({len(both)}):\n")
        for tag in both:
            data1 = font1.reader[tag]
            data2 = font2.reader[tag]
            if data1 == data2:
                lines.append(f"  {tag}: SAME ({len(data1)} bytes)\n")
            else:
                identical = False
                lines.append(f"* {tag}: DIFF ({len(data1)} vs {len(data2)} bytes)\n")

        if identical:
            lines.append("\nResult: SAME\n")
        else:
            lines.append("\nResult: DIFFERENT\n")

        return identical, "".join(lines)


def get_binary_exclude_tables(
    file1: str,
    file2: str,
    include_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None,
    font_number_1: int = -1,
    font_number_2: int = -1,
) -> Tuple[bool, str]:
    from fontTools.ttLib import TTFont

    with (
        TTFont(file1, lazy=True, fontNumber=font_number_1) as font1,
        TTFont(file2, lazy=True, fontNumber=font_number_2) as font2,
    ):
        tags1 = {str(tag) for tag in font1.reader.keys()}
        tags2 = {str(tag) for tag in font2.reader.keys()}

        all_tags = sorted(
            set(
                _iter_filtered_table_tags(
                    tags1 | tags2,
                    include_tables=include_tables,
                    exclude_tables=exclude_tables,
                )
            )
        )

        both = [tag for tag in all_tags if tag in tags1 and tag in tags2]
        out = set()

        for tag in both:
            data1 = font1.reader[tag]
            data2 = font2.reader[tag]
            if data1 == data2:
                out.add(tag)

        return out


def main():
    """Compare two fonts for differences"""
    # try/except block rationale:
    # handles "premature" socket closure exception that is
    # raised by Python when stdout is piped to tools like
    # the `head` executable and socket is closed early
    # see: https://docs.python.org/3/library/signal.html#note-on-sigpipe
    ret = 0
    try:
        ret = run(sys.argv[1:])
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    return ret


def run(argv: List[Text]):
    # ------------------------------------------
    # argparse command line argument definitions
    # ------------------------------------------
    parser = argparse.ArgumentParser(
        description="An OpenType table diff tool for fonts."
    )
    parser.add_argument(
        "-l",
        "--summary",
        action="store_true",
        help="Report table presence and binary equality only",
    )
    parser.add_argument(
        "-U",
        "--lines",
        type=int,
        default=3,
        help="Number of context lines for unified diff (default: 3)",
    )
    parser.add_argument(
        "-t",
        "--include",
        type=str,
        nargs="+",
        default=None,
        help="Font tables to include. Multiple options are allowed.",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help="Font tables to exclude. Multiple options are allowed.",
    )
    parser.add_argument(
        "--diff", type=str, help="Run external diff tool command (default: diff)"
    )
    parser.add_argument(
        "--diff-arg",
        type=str,
        default=None,
        help="External diff tool arguments (default: -u)",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "never", "always"],
        default="auto",
        help="Whether to colorize output (default: auto)",
    )
    parser.add_argument(
        "--y1",
        type=int,
        default=-1,
        metavar="NUMBER",
        help="Select font number for TrueType Collection (.ttc/.otc) FILE1, starting from 0",
    )
    parser.add_argument(
        "--y2",
        type=int,
        default=-1,
        metavar="NUMBER",
        help="Select font number for TrueType Collection (.ttc/.otc) FILE2, starting from 0",
    )
    parser.add_argument(
        "-a",
        "--always",
        action="store_true",
        help="Compare tables even if binary identical",
    )
    parser.add_argument(
        "-b",
        "--binary",
        action="store_true",
        help="Compare tables only if binaries differ (default)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress all output"
    )
    parser.add_argument("FILE1", help="Font file path 1")
    parser.add_argument("FILE2", help="Font file path 2")

    args: argparse.Namespace = parser.parse_args(argv)

    # /////////////////////////////////////////////////////////
    #
    #  Validations
    #
    # /////////////////////////////////////////////////////////

    # ----------------------------------
    #  Incompatible argument validations
    # ----------------------------------

    if args.always and args.binary:
        if not args.quiet:
            sys.stderr.write(
                f"[*] Error: --always and --binary are mutually exclusive options. "
                f"Please use ONLY one of these options in your command.{os.linesep}"
            )
        return 2
    if not args.always:
        args.binary = True

    # -------------------------------
    #  File path argument validations
    # -------------------------------

    if not file_exists(args.FILE1):
        if not args.quiet:
            sys.stderr.write(
                f"[*] ERROR: The file path '{args.FILE1}' can not be found.{os.linesep}"
            )
        return 2
    if not file_exists(args.FILE2):
        if not args.quiet:
            sys.stderr.write(
                f"[*] ERROR: The file path '{args.FILE2}' can not be found.{os.linesep}"
            )
        return 2

    # /////////////////////////////////////////////////////////
    #
    #  Command line logic
    #
    # /////////////////////////////////////////////////////////

    # parse explicitly included or excluded tables in
    # the command line arguments
    # set as a Python list if it was defined on the command line
    # or as None if it was not set on the command line
    include_list: Optional[List[Text]] = get_tables_argument_list(args.include)
    exclude_list: Optional[List[Text]] = get_tables_argument_list(args.exclude)

    if args.summary:
        try:
            identical, output = summarize(
                args.FILE1,
                args.FILE2,
                include_tables=include_list,
                exclude_tables=exclude_list,
                font_number_1=args.y1,
                font_number_2=args.y2,
            )
            if not args.quiet:
                sys.stdout.write(output)
            return 0 if identical else 1
        except Exception as e:
            if not args.quiet:
                sys.stderr.write(f"[*] ERROR: {e}{os.linesep}")
            return 2

    if args.binary:
        excluded_binary_tables = get_binary_exclude_tables(
            args.FILE1,
            args.FILE2,
            include_tables=include_list,
            exclude_tables=exclude_list,
            font_number_1=args.y1,
            font_number_2=args.y2,
        )
        if include_list is not None:
            include_list = [
                tag for tag in include_list if tag not in excluded_binary_tables
            ]
        else:
            if exclude_list is None:
                exclude_list = []
            exclude_list.extend(sorted(excluded_binary_tables))

    diff_tool = args.diff
    color_output = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty
    )

    if diff_tool is None:
        diff_tool = shutil.which("diff")
    elif diff_tool:
        diff_tool = shutil.which(diff_tool)
        if diff_tool is None:
            if not args.quiet:
                sys.stderr.write(
                    f"[*] ERROR: The external diff tool executable "
                    f"'{args.diff}' was not found.{os.linesep}"
                )
            return 2

    try:
        if diff_tool:
            diff_arg = args.diff_arg
            if diff_arg is None:
                if args.lines == 3:
                    diff_arg = ["-u"]
                else:
                    diff_arg = ["-u{}".format(args.lines)]
                if _is_gnu_diff(diff_tool):
                    diff_arg.append(r"-F^\s\s<")
            else:
                diff_arg = diff_arg.split()

            output = run_external_diff(
                diff_tool,
                diff_arg,
                args.FILE1,
                args.FILE2,
                include_tables=include_list,
                exclude_tables=exclude_list,
                font_number_a=args.y1,
                font_number_b=args.y2,
                use_multiprocess=True,
            )
        else:
            output = u_diff(
                args.FILE1,
                args.FILE2,
                context_lines=args.lines,
                include_tables=include_list,
                exclude_tables=exclude_list,
                font_number_a=args.y1,
                font_number_b=args.y2,
                use_multiprocess=True,
            )

        if color_output:
            output = [color_unified_diff_line(line) for line in output]

        output = "".join(output)
        if not args.quiet:
            pipe_output(output)
        return 1 if output else 0

    except Exception as e:
        if not args.quiet:
            sys.stderr.write(f"[*] ERROR: {e}{os.linesep}")
        return 2
<|endoftext|>import os
import subprocess
import tempfile
from contextlib import contextmanager
from difflib import unified_diff
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Iterable, Iterator, List, Optional, Text, Tuple

from fontTools.ttLib import TTFont  # type: ignore

from .utils import get_file_modtime

#
#
#  Private functions
#
#


def _get_fonts_and_save_xml(
    filepath_a: Text,
    filepath_b: Text,
    tmpdirpath: Text,
    include_tables: Optional[List[Text]],
    exclude_tables: Optional[List[Text]],
    font_number_a: int,
    font_number_b: int,
    use_multiprocess: bool,
) -> Tuple[Text, Text, Text, Text, Text, Text]:
    post_pathname, postpath, pre_pathname, prepath = _get_pre_post_paths(
        filepath_a, filepath_b
    )
    # instantiate left and right fontTools.ttLib.TTFont objects
    tt_left = TTFont(prepath, fontNumber=font_number_a)
    tt_right = TTFont(postpath, fontNumber=font_number_b)
    left_ttxpath = os.path.join(tmpdirpath, "left.ttx")
    right_ttxpath = os.path.join(tmpdirpath, "right.ttx")
    _mp_save_ttx_xml(
        tt_left,
        tt_right,
        left_ttxpath,
        right_ttxpath,
        exclude_tables,
        include_tables,
        use_multiprocess,
    )
    return left_ttxpath, right_ttxpath, pre_pathname, prepath, post_pathname, postpath


def _get_pre_post_paths(
    filepath_a: Text,
    filepath_b: Text,
) -> Tuple[Text, Text, Text, Text]:
    prepath = filepath_a
    postpath = filepath_b
    pre_pathname = filepath_a
    post_pathname = filepath_b
    return post_pathname, postpath, pre_pathname, prepath


def _mp_save_ttx_xml(
    tt_left: Any,
    tt_right: Any,
    left_ttxpath: Text,
    right_ttxpath: Text,
    exclude_tables: Optional[List[Text]],
    include_tables: Optional[List[Text]],
    use_multiprocess: bool,
) -> None:
    if use_multiprocess and cpu_count() > 1:
        # Use parallel fontTools.ttLib.TTFont.saveXML dump
        # by default on multi CPU systems.  This is a performance
        # optimization. Profiling demonstrates that this can reduce
        # execution time by up to 30% for some fonts
        mp_args_list = [
            (tt_left, left_ttxpath, include_tables, exclude_tables),
            (tt_right, right_ttxpath, include_tables, exclude_tables),
        ]
        with Pool(processes=2) as pool:
            pool.starmap(_ttfont_save_xml, mp_args_list)
    else:
        # use sequential fontTools.ttLib.TTFont.saveXML dumps
        # when use_multiprocess is False or single CPU system
        # detected
        _ttfont_save_xml(tt_left, left_ttxpath, include_tables, exclude_tables)
        _ttfont_save_xml(tt_right, right_ttxpath, include_tables, exclude_tables)


def _ttfont_save_xml(
    ttf: Any,
    filepath: Text,
    include_tables: Optional[List[Text]],
    exclude_tables: Optional[List[Text]],
) -> bool:
    """Writes TTX specification formatted XML to disk on filepath."""
    ttf.saveXML(filepath, tables=include_tables, skipTables=exclude_tables)
    return True


@contextmanager
def _saved_ttx_files(
    filepath_a: Text,
    filepath_b: Text,
    include_tables: Optional[List[Text]],
    exclude_tables: Optional[List[Text]],
    font_number_a: int,
    font_number_b: int,
    use_multiprocess: bool,
) -> Iterator[Tuple[Text, Text, Text, Text, Text, Text]]:
    with tempfile.TemporaryDirectory() as tmpdirpath:
        yield _get_fonts_and_save_xml(
            filepath_a,
            filepath_b,
            tmpdirpath,
            include_tables,
            exclude_tables,
            font_number_a,
            font_number_b,
            use_multiprocess,
        )


def _diff_with_saved_ttx_files(
    filepath_a: Text,
    filepath_b: Text,
    include_tables: Optional[List[Text]],
    exclude_tables: Optional[List[Text]],
    font_number_a: int,
    font_number_b: int,
    use_multiprocess: bool,
    create_differ: Callable[[Text, Text, Text, Text, Text, Text], Iterable[Text]],
) -> Iterator[Text]:
    with _saved_ttx_files(
        filepath_a,
        filepath_b,
        include_tables,
        exclude_tables,
        font_number_a,
        font_number_b,
        use_multiprocess,
    ) as (
        left_ttxpath,
        right_ttxpath,
        pre_pathname,
        prepath,
        post_pathname,
        postpath,
    ):
        yield from create_differ(
            left_ttxpath,
            right_ttxpath,
            pre_pathname,
            prepath,
            post_pathname,
            postpath,
        )


#
#
#  Public functions
#
#


def u_diff(
    filepath_a: Text,
    filepath_b: Text,
    context_lines: int = 3,
    include_tables: Optional[List[Text]] = None,
    exclude_tables: Optional[List[Text]] = None,
    font_number_a: int = -1,
    font_number_b: int = -1,
    use_multiprocess: bool = True,
) -> Iterator[Text]:
    """Performs a unified diff on a TTX serialized data format dump of font binary data using
    a modified version of the Python standard libary difflib module.

    filepath_a: (string) pre-file local file path
    filepath_b: (string) post-file local file path
    context_lines: (int) number of context lines to include in the diff (default=3)
    include_tables: (list of str) Python list of OpenType tables to include in the diff
    exclude_tables: (list of str) Python list of OpentType tables to exclude from the diff
    use_multiprocess: (bool) use multi-processor optimizations (default=True)

    include_tables and exclude_tables are mutually exclusive arguments.  Only one should
    be defined

    :returns: Generator of ordered diff line strings that include newline line endings
    :raises: KeyError if include_tables or exclude_tables includes a mis-specified table
    that is not included in filepath_a OR filepath_b
    """

    def _create_unified_diff(
        left_ttxpath: Text,
        right_ttxpath: Text,
        pre_pathname: Text,
        prepath: Text,
        post_pathname: Text,
        postpath: Text,
    ) -> Iterable[Text]:
        with open(left_ttxpath) as ff:
            fromlines = ff.readlines()
        with open(right_ttxpath) as tf:
            tolines = tf.readlines()

        fromdate = get_file_modtime(prepath)
        todate = get_file_modtime(postpath)

        yield from unified_diff(
            fromlines,
            tolines,
            pre_pathname,
            post_pathname,
            fromdate,
            todate,
            n=context_lines,
        )

    yield from _diff_with_saved_ttx_files(
        filepath_a,
        filepath_b,
        include_tables,
        exclude_tables,
        font_number_a,
        font_number_b,
        use_multiprocess,
        _create_unified_diff,
    )


def run_external_diff(
    diff_tool: Text,
    diff_args: List[Text],
    filepath_a: Text,
    filepath_b: Text,
    include_tables: Optional[List[Text]] = None,
    exclude_tables: Optional[List[Text]] = None,
    font_number_a: int = -1,
    font_number_b: int = -1,
    use_multiprocess: bool = True,
) -> Iterator[Text]:
    """Performs a unified diff on a TTX serialized data format dump of font binary data using
    an external diff executable that is requested by the caller via `command`

    diff_tool: (string) command line executable string
    diff_args: (list of strings) arguments for the diff tool
    filepath_a: (string) pre-file local file path
    filepath_b: (string) post-file local file path
    include_tables: (list of str) Python list of OpenType tables to include in the diff
    exclude_tables: (list of str) Python list of OpentType tables to exclude from the diff
    use_multiprocess: (bool) use multi-processor optimizations (default=True)

    include_tables and exclude_tables are mutually exclusive arguments.  Only one should
    be defined

    :returns: Generator of ordered diff line strings that include newline line endings
    :raises: KeyError if include_tables or exclude_tables includes a mis-specified table
    that is not included in filepath_a OR filepath_b
    :raises: IOError if exception raised during execution of `command` on TTX files
    """

    def _create_external_diff(
        left_ttxpath: Text,
        right_ttxpath: Text,
        _pre_pathname: Text,
        _prepath: Text,
        _post_pathname: Text,
        _postpath: Text,
    ) -> Iterable[Text]:
        command = [diff_tool] + diff_args + [left_ttxpath, right_ttxpath]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf8",
        )

        for line in process.stdout:
            yield line
        err = process.stderr.read()
        if err:
            raise IOError(err)

    yield from _diff_with_saved_ttx_files(
        filepath_a,
        filepath_b,
        include_tables,
        exclude_tables,
        font_number_a,
        font_number_b,
        use_multiprocess,
        _create_external_diff,
    )
<|endoftext|>"""
Define all configuration options that can affect the working of fontTools
modules. E.g. optimization levels of varLib IUP, otlLib GPOS compression level,
etc. If this file gets too big, split it into smaller files per-module.

An instance of the Config class can be attached to a TTFont object, so that
the various modules can access their configuration options from it.
"""

from textwrap import dedent

from fontTools.misc.configTools import *


class Config(AbstractConfig):
    options = Options()


OPTIONS = Config.options


Config.register_option(
    name="fontTools.otlLib.optimize.gpos:COMPRESSION_LEVEL",
    help=dedent("""\
        GPOS Lookup type 2 (PairPos) compression level:
            0 = do not attempt to compact PairPos lookups;
            1 to 8 = create at most 1 to 8 new subtables for each existing
                subtable, provided that it would yield a 50%% file size saving;
            9 = create as many new subtables as needed to yield a file size saving.
        Default: 0.

        This compaction aims to save file size, by splitting large class
        kerning subtables (Format 2) that contain many zero values into
        smaller and denser subtables. It's a trade-off between the overhead
        of several subtables versus the sparseness of one big subtable.

        See the pull request: https://github.com/fonttools/fonttools/pull/2326
        """),
    default=0,
    parse=int,
    validate=lambda v: v in range(10),
)

Config.register_option(
    name="fontTools.ttLib.tables.otBase:USE_HARFBUZZ_REPACKER",
    help=dedent("""\
        FontTools tries to use the HarfBuzz Repacker to serialize GPOS/GSUB tables
        if the uharfbuzz python bindings are importable, otherwise falls back to its
        slower, less efficient serializer. Set to False to always use the latter.
        Set to True to explicitly request the HarfBuzz Repacker (will raise an
        error if uharfbuzz cannot be imported).
        """),
    default=None,
    parse=Option.parse_optional_bool,
    validate=Option.validate_optional_bool,
)

Config.register_option(
    name="fontTools.otlLib.builder:WRITE_GPOS7",
    help=dedent("""\
        macOS before 13.2 didn’t support GPOS LookupType 7 (non-chaining
        ContextPos lookups), so FontTools.otlLib.builder disables a file size
        optimisation that would use LookupType 7 instead of 8 when there is no
        chaining (no prefix or suffix). Set to True to enable the optimization.
        """),
    default=False,
    parse=Option.parse_optional_bool,
    validate=Option.validate_optional_bool,
)

Config.register_option(
    name="fontTools.ttLib:OPTIMIZE_FONT_SPEED",
    help=dedent("""\
        Enable optimizations that prioritize speed over file size. This
        mainly affects how glyf table and gvar / VARC tables are compiled.
        The produced fonts will be larger, but rendering performance will
        be improved with HarfBuzz and other text layout engines.
        """),
    default=False,
    parse=Option.parse_optional_bool,
    validate=Option.validate_optional_bool,
)
<|endoftext|>"""Compute name information for a given location in user-space coordinates
using STAT data. This can be used to fill-in automatically the names of an
instance:

.. code:: python

    instance = doc.instances[0]
    names = getStatNames(doc, instance.getFullUserLocation(doc))
    print(names.styleNames)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union
import logging

from fontTools.designspaceLib import (
    AxisDescriptor,
    AxisLabelDescriptor,
    DesignSpaceDocument,
    DiscreteAxisDescriptor,
    SimpleLocationDict,
    SourceDescriptor,
)

LOGGER = logging.getLogger(__name__)

RibbiStyleName = Union[
    Literal["regular"],
    Literal["bold"],
    Literal["italic"],
    Literal["bold italic"],
]

BOLD_ITALIC_TO_RIBBI_STYLE = {
    (False, False): "regular",
    (False, True): "italic",
    (True, False): "bold",
    (True, True): "bold italic",
}


@dataclass
class StatNames:
    """Name data generated from the STAT table information."""

    familyNames: Dict[str, str]
    styleNames: Dict[str, str]
    postScriptFontName: Optional[str]
    styleMapFamilyNames: Dict[str, str]
    styleMapStyleName: Optional[RibbiStyleName]


def getStatNames(
    doc: DesignSpaceDocument, userLocation: SimpleLocationDict
) -> StatNames:
    """Compute the family, style, PostScript names of the given ``userLocation``
    using the document's STAT information.

    Also computes localizations.

    If not enough STAT data is available for a given name, either its dict of
    localized names will be empty (family and style names), or the name will be
    None (PostScript name).

    Note: this method does not consider info attached to the instance, like
    family name. The user needs to override all names on an instance that STAT
    information would compute differently than desired.

    .. versionadded:: 5.0
    """
    familyNames: Dict[str, str] = {}
    defaultSource: Optional[SourceDescriptor] = doc.findDefault()
    if defaultSource is None:
        LOGGER.warning("Cannot determine default source to look up family name.")
    elif defaultSource.familyName is None:
        LOGGER.warning(
            "Cannot look up family name, assign the 'familyname' attribute to the default source."
        )
    else:
        familyNames = {
            "en": defaultSource.familyName,
            **defaultSource.localisedFamilyName,
        }

    styleNames: Dict[str, str] = {}
    # If a free-standing label matches the location, use it for name generation.
    label = doc.labelForUserLocation(userLocation)
    if label is not None:
        styleNames = {"en": label.name, **label.labelNames}
    # Otherwise, scour the axis labels for matches.
    else:
        # Gather all languages in which at least one translation is provided
        # Then build names for all these languages, but fallback to English
        # whenever a translation is missing.
        labels = _getAxisLabelsForUserLocation(doc.axes, userLocation)
        if labels:
            languages = set(
                language for label in labels for language in label.labelNames
            )
            languages.add("en")
            for language in languages:
                styleName = " ".join(
                    label.labelNames.get(language, label.defaultName)
                    for label in labels
                    if not label.elidable
                )
                if not styleName and doc.elidedFallbackName is not None:
                    styleName = doc.elidedFallbackName
                styleNames[language] = styleName

    if "en" not in familyNames or "en" not in styleNames:
        # Not enough information to compute PS names of styleMap names
        return StatNames(
            familyNames=familyNames,
            styleNames=styleNames,
            postScriptFontName=None,
            styleMapFamilyNames={},
            styleMapStyleName=None,
        )

    postScriptFontName = f"{familyNames['en']}-{styleNames['en']}".replace(" ", "")

    styleMapStyleName, regularUserLocation = _getRibbiStyle(doc, userLocation)

    styleNamesForStyleMap = styleNames
    if regularUserLocation != userLocation:
        regularStatNames = getStatNames(doc, regularUserLocation)
        styleNamesForStyleMap = regularStatNames.styleNames

    styleMapFamilyNames = {}
    for language in set(familyNames).union(styleNames.keys()):
        familyName = familyNames.get(language, familyNames["en"])
        styleName = styleNamesForStyleMap.get(language, styleNamesForStyleMap["en"])
        styleMapFamilyNames[language] = (familyName + " " + styleName).strip()

    return StatNames(
        familyNames=familyNames,
        styleNames=styleNames,
        postScriptFontName=postScriptFontName,
        styleMapFamilyNames=styleMapFamilyNames,
        styleMapStyleName=styleMapStyleName,
    )


def _getSortedAxisLabels(
    axes: list[Union[AxisDescriptor, DiscreteAxisDescriptor]],
) -> Dict[str, list[AxisLabelDescriptor]]:
    """Returns axis labels sorted by their ordering, with unordered ones appended as
    they are listed."""

    # First, get the axis labels with explicit ordering...
    sortedAxes = sorted(
        (axis for axis in axes if axis.axisOrdering is not None),
        key=lambda a: a.axisOrdering,
    )
    sortedLabels: Dict[str, list[AxisLabelDescriptor]] = {
        axis.name: axis.axisLabels for axis in sortedAxes
    }

    # ... then append the others in the order they appear.
    # NOTE: This relies on Python 3.7+ dict's preserved insertion order.
    for axis in axes:
        if axis.axisOrdering is None:
            sortedLabels[axis.name] = axis.axisLabels

    return sortedLabels


def _getAxisLabelsForUserLocation(
    axes: list[Union[AxisDescriptor, DiscreteAxisDescriptor]],
    userLocation: SimpleLocationDict,
) -> list[AxisLabelDescriptor]:
    labels: list[AxisLabelDescriptor] = []

    allAxisLabels = _getSortedAxisLabels(axes)
    if allAxisLabels.keys() != userLocation.keys():
        LOGGER.warning(
            f"Mismatch between user location '{userLocation.keys()}' and available "
            f"labels for '{allAxisLabels.keys()}'."
        )

    for axisName, axisLabels in allAxisLabels.items():
        userValue = userLocation[axisName]
        label: Optional[AxisLabelDescriptor] = next(
            (
                l
                for l in axisLabels
                if l.userValue == userValue
                or (
                    l.userMinimum is not None
                    and l.userMaximum is not None
                    and l.userMinimum <= userValue <= l.userMaximum
                )
            ),
            None,
        )
        if label is None:
            LOGGER.debug(
                f"Document needs a label for axis '{axisName}', user value '{userValue}'."
            )
        else:
            labels.append(label)

    return labels


def _getRibbiStyle(
    self: DesignSpaceDocument, userLocation: SimpleLocationDict
) -> Tuple[RibbiStyleName, SimpleLocationDict]:
    """Compute the RIBBI style name of the given user location,
    return the location of the matching Regular in the RIBBI group.

    .. versionadded:: 5.0
    """
    regularUserLocation = {}
    axes_by_tag = {axis.tag: axis for axis in self.axes}

    bold: bool = False
    italic: bool = False

    axis = axes_by_tag.get("wght")
    if axis is not None:
        for regular_label in axis.axisLabels:
            if (
                regular_label.linkedUserValue == userLocation[axis.name]
                # In the "recursive" case where both the Regular has
                # linkedUserValue pointing the Bold, and the Bold has
                # linkedUserValue pointing to the Regular, only consider the
                # first case: Regular (e.g. 400) has linkedUserValue pointing to
                # Bold (e.g. 700, higher than Regular)
                and regular_label.userValue < regular_label.linkedUserValue
            ):
                regularUserLocation[axis.name] = regular_label.userValue
                bold = True
                break

    axis = axes_by_tag.get("ital") or axes_by_tag.get("slnt")
    if axis is not None:
        for upright_label in axis.axisLabels:
            if (
                upright_label.linkedUserValue == userLocation[axis.name]
                # In the "recursive" case where both the Upright has
                # linkedUserValue pointing the Italic, and the Italic has
                # linkedUserValue pointing to the Upright, only consider the
                # first case: Upright (e.g. ital=0, slant=0) has
                # linkedUserValue pointing to Italic (e.g ital=1, slant=-12 or
                # slant=12 for backwards italics, in any case higher than
                # Upright in absolute value, hence the abs() below.
                and abs(upright_label.userValue) < abs(upright_label.linkedUserValue)
            ):
                regularUserLocation[axis.name] = upright_label.userValue
                italic = True
                break

    return BOLD_ITALIC_TO_RIBBI_STYLE[bold, italic], {
        **userLocation,
        **regularUserLocation,
    }
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast

from fontTools.designspaceLib import (
    AxisDescriptor,
    DesignSpaceDocument,
    DesignSpaceDocumentError,
    RangeAxisSubsetDescriptor,
    SimpleLocationDict,
    ValueAxisSubsetDescriptor,
    VariableFontDescriptor,
)


def clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)


@dataclass
class Range:
    minimum: float
    """Inclusive minimum of the range."""
    maximum: float
    """Inclusive maximum of the range."""
    default: float = 0
    """Default value"""

    def __post_init__(self):
        self.minimum, self.maximum = sorted((self.minimum, self.maximum))
        self.default = clamp(self.default, self.minimum, self.maximum)

    def __contains__(self, value: Union[float, Range]) -> bool:
        if isinstance(value, Range):
            return self.minimum <= value.minimum and value.maximum <= self.maximum
        return self.minimum <= value <= self.maximum

    def intersection(self, other: Range) -> Optional[Range]:
        if self.maximum < other.minimum or self.minimum > other.maximum:
            return None
        else:
            return Range(
                max(self.minimum, other.minimum),
                min(self.maximum, other.maximum),
                self.default,  # We don't care about the default in this use-case
            )


# A region selection is either a range or a single value, as a Designspace v5
# axis-subset element only allows a single discrete value or a range for a
# variable-font element.
Region = Dict[str, Union[Range, float]]

# A conditionset is a set of named ranges.
ConditionSet = Dict[str, Range]

# A rule is a list of conditionsets where any has to be relevant for the whole rule to be relevant.
Rule = List[ConditionSet]
Rules = Dict[str, Rule]


def locationInRegion(location: SimpleLocationDict, region: Region) -> bool:
    for name, value in location.items():
        if name not in region:
            return False
        regionValue = region[name]
        if isinstance(regionValue, (float, int)):
            if value != regionValue:
                return False
        else:
            if value not in regionValue:
                return False
    return True


def regionInRegion(region: Region, superRegion: Region) -> bool:
    for name, value in region.items():
        if not name in superRegion:
            return False
        superValue = superRegion[name]
        if isinstance(superValue, (float, int)):
            if value != superValue:
                return False
        else:
            if value not in superValue:
                return False
    return True


def userRegionToDesignRegion(doc: DesignSpaceDocument, userRegion: Region) -> Region:
    designRegion = {}
    for name, value in userRegion.items():
        axis = doc.getAxis(name)
        if axis is None:
            raise DesignSpaceDocumentError(
                f"Cannot find axis named '{name}' for region."
            )
        if isinstance(value, (float, int)):
            designRegion[name] = axis.map_forward(value)
        else:
            designRegion[name] = Range(
                axis.map_forward(value.minimum),
                axis.map_forward(value.maximum),
                axis.map_forward(value.default),
            )
    return designRegion


def getVFUserRegion(doc: DesignSpaceDocument, vf: VariableFontDescriptor) -> Region:
    vfUserRegion: Region = {}
    # For each axis, 2 cases:
    #  - it has a range = it's an axis in the VF DS
    #  - it's a single location = use it to know which rules should apply in the VF
    for axisSubset in vf.axisSubsets:
        axis = doc.getAxis(axisSubset.name)
        if axis is None:
            raise DesignSpaceDocumentError(
                f"Cannot find axis named '{axisSubset.name}' for variable font '{vf.name}'."
            )
        if hasattr(axisSubset, "userMinimum"):
            # Mypy doesn't support narrowing union types via hasattr()
            # TODO(Python 3.10): use TypeGuard
            # https://mypy.readthedocs.io/en/stable/type_narrowing.html
            axisSubset = cast(RangeAxisSubsetDescriptor, axisSubset)
            if not hasattr(axis, "minimum"):
                raise DesignSpaceDocumentError(
                    f"Cannot select a range over '{axis.name}' for variable font '{vf.name}' "
                    "because it's a discrete axis, use only 'userValue' instead."
                )
            axis = cast(AxisDescriptor, axis)
            vfUserRegion[axis.name] = Range(
                max(axisSubset.userMinimum, axis.minimum),
                min(axisSubset.userMaximum, axis.maximum),
                axisSubset.userDefault or axis.default,
            )
        else:
            axisSubset = cast(ValueAxisSubsetDescriptor, axisSubset)
