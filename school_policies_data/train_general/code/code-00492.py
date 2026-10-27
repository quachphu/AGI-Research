    fmt_d_ms = r"$%s%d" + deg_mark + r"\,%02d.%s" + min_mark + "$"

    fmt_d_m_partial = "$%s%d" + deg_mark + r"\,%02d" + min_mark + r"\,"
    fmt_s_partial = "%02d" + sec_mark + "$"
    fmt_ss_partial = "%02d.%s" + sec_mark + "$"

    def _get_number_fraction(self, factor):
        ## check for fractional numbers
        number_fraction = None
        # check for 60

        for threshold in [1, 60, 3600]:
            if factor <= threshold:
                break

            d = factor // threshold
            int_log_d = int(np.floor(np.log10(d)))
            if 10**int_log_d == d and d != 1:
                number_fraction = int_log_d
                factor = factor // 10**int_log_d
                return factor, number_fraction

        return factor, number_fraction

    def __call__(self, direction, factor, values):
        if len(values) == 0:
            return []

        ss = np.sign(values)
        signs = ["-" if v < 0 else "" for v in values]

        factor, number_fraction = self._get_number_fraction(factor)

        values = np.abs(values)

        if number_fraction is not None:
            values, frac_part = divmod(values, 10 ** number_fraction)
            frac_fmt = "%%0%dd" % (number_fraction,)
            frac_str = [frac_fmt % (f1,) for f1 in frac_part]

        if factor == 1:
            if number_fraction is None:
                return [self.fmt_d % (s * int(v),) for s, v in zip(ss, values)]
            else:
                return [self.fmt_ds % (s * int(v), f1)
                        for s, v, f1 in zip(ss, values, frac_str)]
        elif factor == 60:
            deg_part, min_part = divmod(values, 60)
            if number_fraction is None:
                return [self.fmt_d_m % (s1, d1, m1)
                        for s1, d1, m1 in zip(signs, deg_part, min_part)]
            else:
                return [self.fmt_d_ms % (s, d1, m1, f1)
                        for s, d1, m1, f1
                        in zip(signs, deg_part, min_part, frac_str)]

        elif factor == 3600:
            if ss[-1] == -1:
                inverse_order = True
                values = values[::-1]
                signs = signs[::-1]
            else:
                inverse_order = False

            l_hm_old = ""
            r = []

            deg_part, min_part_ = divmod(values, 3600)
            min_part, sec_part = divmod(min_part_, 60)

            if number_fraction is None:
                sec_str = [self.fmt_s_partial % (s1,) for s1 in sec_part]
            else:
                sec_str = [self.fmt_ss_partial % (s1, f1)
                           for s1, f1 in zip(sec_part, frac_str)]

            for s, d1, m1, s1 in zip(signs, deg_part, min_part, sec_str):
                l_hm = self.fmt_d_m_partial % (s, d1, m1)
                if l_hm != l_hm_old:
                    l_hm_old = l_hm
                    l = l_hm + s1
                else:
                    l = "$" + s + s1
                r.append(l)

            if inverse_order:
                return r[::-1]
            else:
                return r

        else:  # factor > 3600.
            return [r"$%s^{\circ}$" % v for v in ss*values]


class FormatterHMS(FormatterDMS):
    deg_mark = r"^\mathrm{h}"
    min_mark = r"^\mathrm{m}"
    sec_mark = r"^\mathrm{s}"

    fmt_d = "$%d" + deg_mark + "$"
    fmt_ds = r"$%d.%s" + deg_mark + "$"

    # %s for sign
    fmt_d_m = r"$%s%d" + deg_mark + r"\,%02d" + min_mark+"$"
    fmt_d_ms = r"$%s%d" + deg_mark + r"\,%02d.%s" + min_mark+"$"

    fmt_d_m_partial = "$%s%d" + deg_mark + r"\,%02d" + min_mark + r"\,"
    fmt_s_partial = "%02d" + sec_mark + "$"
    fmt_ss_partial = "%02d.%s" + sec_mark + "$"

    def __call__(self, direction, factor, values):  # hour
        return super().__call__(direction, factor, np.asarray(values) / 15)


class ExtremeFinderCycle(ExtremeFinderSimple):
    # docstring inherited

    def __init__(self, nx, ny,
                 lon_cycle=360., lat_cycle=None,
                 lon_minmax=None, lat_minmax=(-90, 90)):
        """
        This subclass handles the case where one or both coordinates should be
        taken modulo 360, or be restricted to not exceed a specific range.

        Parameters
        ----------
        nx, ny : int
            The number of samples in each direction.

        lon_cycle, lat_cycle : 360 or None
            If not None, values in the corresponding direction are taken modulo
            *lon_cycle* or *lat_cycle*; in theory this can be any number but
            the implementation actually assumes that it is 360 (if not None);
            other values give nonsensical results.

            This is done by "unwrapping" the transformed grid coordinates so
            that jumps are less than a half-cycle; then normalizing the span to
            no more than a full cycle.

            For example, if values are in the union of the [0, 2] and
            [358, 360] intervals (typically, angles measured modulo 360), the
            values in the second interval are normalized to [-2, 0] instead so
            that the values now cover [-2, 2].  If values are in a range of
            [5, 1000], this gets normalized to [5, 365].

        lon_minmax, lat_minmax : (float, float) or None
            If not None, the computed bounding box is clipped to the given
            range in the corresponding direction.
        """
        self.nx, self.ny = nx, ny
        self.lon_cycle, self.lat_cycle = lon_cycle, lat_cycle
        self.lon_minmax = lon_minmax
        self.lat_minmax = lat_minmax

    def __call__(self, transform_xy, x1, y1, x2, y2):
        # docstring inherited
        x, y = np.meshgrid(
            np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        lon, lat = transform_xy(np.ravel(x), np.ravel(y))

        # iron out jumps, but algorithm should be improved.
        # This is just naive way of doing and my fail for some cases.
        # Consider replacing this with numpy.unwrap
        # We are ignoring invalid warnings. They are triggered when
        # comparing arrays with NaNs using > We are already handling
        # that correctly using np.nanmin and np.nanmax
        with np.errstate(invalid='ignore'):
            if self.lon_cycle is not None:
                lon0 = np.nanmin(lon)
                lon -= 360. * ((lon - lon0) > 180.)
            if self.lat_cycle is not None:
                lat0 = np.nanmin(lat)
                lat -= 360. * ((lat - lat0) > 180.)

        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

        lon_min, lon_max, lat_min, lat_max = \
            self._add_pad(lon_min, lon_max, lat_min, lat_max)

        # check cycle
        if self.lon_cycle:
            lon_max = min(lon_max, lon_min + self.lon_cycle)
        if self.lat_cycle:
            lat_max = min(lat_max, lat_min + self.lat_cycle)

        if self.lon_minmax is not None:
            min0 = self.lon_minmax[0]
            lon_min = max(min0, lon_min)
            max0 = self.lon_minmax[1]
            lon_max = min(max0, lon_max)

        if self.lat_minmax is not None:
            min0 = self.lat_minmax[0]
            lat_min = max(min0, lat_min)
            max0 = self.lat_minmax[1]
            lat_max = min(max0, lat_max)

        return lon_min, lon_max, lat_min, lat_max
<|endoftext|>from .axislines import Axes
from .axislines import (  # noqa: F401
    AxesZero, AxisArtistHelper, AxisArtistHelperRectlinear,
    GridHelperBase, GridHelperRectlinear, Subplot, SubplotZero)
from .axis_artist import AxisArtist, GridlinesCollection  # noqa: F401
from .grid_helper_curvelinear import GridHelperCurveLinear  # noqa: F401
from .floating_axes import FloatingAxes, FloatingSubplot  # noqa: F401
from mpl_toolkits.axes_grid1.parasite_axes import (
    host_axes_class_factory, parasite_axes_class_factory)


ParasiteAxes = parasite_axes_class_factory(Axes)
HostAxes = host_axes_class_factory(Axes)
SubplotHost = HostAxes
<|endoftext|>"""
An experimental support for curvilinear grid.
"""

import functools

import numpy as np

import matplotlib as mpl
from matplotlib import _api
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, IdentityTransform
from .axislines import (
    _FixedAxisArtistHelperBase, _FloatingAxisArtistHelperBase, GridHelperBase)
from .axis_artist import AxisArtist
from .grid_finder import GridFinder


def _value_and_jacobian(func, xs, ys, xlims, ylims):
    """
    Compute *func* and its derivatives along x and y at positions *xs*, *ys*,
    while ensuring that finite difference calculations don't try to evaluate
    values outside of *xlims*, *ylims*.
    """
    eps = np.finfo(float).eps ** (1/2)  # see e.g. scipy.optimize.approx_fprime
    val = func(xs, ys)
    # Take the finite difference step in the direction where the bound is the
    # furthest; the step size is min of epsilon and distance to that bound.
    xlo, xhi = sorted(xlims)
    dxlo = xs - xlo
    dxhi = xhi - xs
    xeps = (np.take([-1, 1], dxhi >= dxlo)
            * np.minimum(eps, np.maximum(dxlo, dxhi)))
    val_dx = func(xs + xeps, ys)
    ylo, yhi = sorted(ylims)
    dylo = ys - ylo
    dyhi = yhi - ys
    yeps = (np.take([-1, 1], dyhi >= dylo)
            * np.minimum(eps, np.maximum(dylo, dyhi)))
    val_dy = func(xs, ys + yeps)
    return (val, (val_dx - val) / xeps, (val_dy - val) / yeps)


class FixedAxisArtistHelper(_FixedAxisArtistHelperBase):
    """
    Helper class for a fixed axis.
    """

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """

        super().__init__(loc=side)

        self.grid_helper = grid_helper
        if nth_coord_ticks is None:
            nth_coord_ticks = self.nth_coord
        self.nth_coord_ticks = nth_coord_ticks

        self.side = side

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

    def get_tick_transform(self, axes):
        return axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        v1, v2 = axes.get_ylim() if self.nth_coord == 0 else axes.get_xlim()
        if v1 > v2:  # Inverted limits.
            side = {"left": "right", "right": "left",
                    "top": "bottom", "bottom": "top"}[self.side]
        else:
            side = self.side

        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[side]

        def iter_major():
            for nth_coord, show_labels in [
                    (self.nth_coord_ticks, True), (1 - self.nth_coord_ticks, False)]:
                gi = self.grid_helper._grid_info[["lon", "lat"][nth_coord]]
                for tick in gi["ticks"][side]:
                    yield (*tick["loc"], angle_tangent,
                           (tick["label"] if show_labels else ""))

        return iter_major(), iter([])


class FloatingAxisArtistHelper(_FloatingAxisArtistHelperBase):

    def __init__(self, grid_helper, nth_coord, value, axis_direction=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        super().__init__(nth_coord, value)
        self.value = value
        self.grid_helper = grid_helper
        self._extremes = -np.inf, np.inf
        self._line_num_points = 100  # number of points to create a line

    def set_extremes(self, e1, e2):
        if e1 is None:
            e1 = -np.inf
        if e2 is None:
            e2 = np.inf
        self._extremes = e1, e2

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        grid_finder = self.grid_helper.grid_finder
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)

        lon_min, lon_max, lat_min, lat_max = extremes
        e_min, e_max = self._extremes  # ranges of other coordinates
        if self.nth_coord == 0:
            lat_min = max(e_min, lat_min)
            lat_max = min(e_max, lat_max)
        elif self.nth_coord == 1:
            lon_min = max(e_min, lon_min)
            lon_max = min(e_max, lon_max)

        lon_levs, lon_n, lon_factor = \
            grid_finder.grid_locator1(lon_min, lon_max)
        lat_levs, lat_n, lat_factor = \
            grid_finder.grid_locator2(lat_min, lat_max)

        if self.nth_coord == 0:
            xx0 = np.full(self._line_num_points, self.value)
            yy0 = np.linspace(lat_min, lat_max, self._line_num_points)
            xx, yy = grid_finder.transform_xy(xx0, yy0)
        elif self.nth_coord == 1:
            xx0 = np.linspace(lon_min, lon_max, self._line_num_points)
            yy0 = np.full(self._line_num_points, self.value)
            xx, yy = grid_finder.transform_xy(xx0, yy0)

        self._grid_info = {
            "extremes": (lon_min, lon_max, lat_min, lat_max),
            "lon_info": (lon_levs, lon_n, np.asarray(lon_factor)),
            "lat_info": (lat_levs, lat_n, np.asarray(lat_factor)),
            "lon_labels": grid_finder._format_ticks(
                1, "bottom", lon_factor, lon_levs),
            "lat_labels": grid_finder._format_ticks(
                2, "bottom", lat_factor, lat_levs),
            "line_xy": (xx, yy),
        }

    def get_axislabel_transform(self, axes):
        return Affine2D()  # axes.transData

    def get_axislabel_pos_angle(self, axes):
        def trf_xy(x, y):
            trf = self.grid_helper.grid_finder.get_transform() + axes.transData
            return trf.transform([x, y]).T

        xmin, xmax, ymin, ymax = self._grid_info["extremes"]
        if self.nth_coord == 0:
            xx0 = self.value
            yy0 = (ymin + ymax) / 2
        elif self.nth_coord == 1:
            xx0 = (xmin + xmax) / 2
            yy0 = self.value
        xy1, dxy1_dx, dxy1_dy = _value_and_jacobian(
            trf_xy, xx0, yy0, (xmin, xmax), (ymin, ymax))
        p = axes.transAxes.inverted().transform(xy1)
        if 0 <= p[0] <= 1 and 0 <= p[1] <= 1:
            d = [dxy1_dy, dxy1_dx][self.nth_coord]
            return xy1, np.rad2deg(np.arctan2(*d[::-1]))
        else:
            return None, None

    def get_tick_transform(self, axes):
        return IdentityTransform()  # axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""

        lat_levs, lat_n, lat_factor = self._grid_info["lat_info"]
        yy0 = lat_levs / lat_factor

        lon_levs, lon_n, lon_factor = self._grid_info["lon_info"]
        xx0 = lon_levs / lon_factor

        e0, e1 = self._extremes

        def trf_xy(x, y):
            trf = self.grid_helper.grid_finder.get_transform() + axes.transData
            return trf.transform(np.column_stack(np.broadcast_arrays(x, y))).T

        # find angles
        if self.nth_coord == 0:
            mask = (e0 <= yy0) & (yy0 <= e1)
            (xx1, yy1), (dxx1, dyy1), (dxx2, dyy2) = _value_and_jacobian(
                trf_xy, self.value, yy0[mask], (-np.inf, np.inf), (e0, e1))
            labels = self._grid_info["lat_labels"]

        elif self.nth_coord == 1:
            mask = (e0 <= xx0) & (xx0 <= e1)
            (xx1, yy1), (dxx2, dyy2), (dxx1, dyy1) = _value_and_jacobian(
                trf_xy, xx0[mask], self.value, (-np.inf, np.inf), (e0, e1))
            labels = self._grid_info["lon_labels"]

        labels = [l for l, m in zip(labels, mask) if m]

        angle_normal = np.arctan2(dyy1, dxx1)
        angle_tangent = np.arctan2(dyy2, dxx2)
        mm = (dyy1 == 0) & (dxx1 == 0)  # points with degenerate normal
        angle_normal[mm] = angle_tangent[mm] + np.pi / 2

        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
        in_01 = functools.partial(
            mpl.transforms._interval_contains_close, (0, 1))

        def iter_major():
            for x, y, normal, tangent, lab \
                    in zip(xx1, yy1, angle_normal, angle_tangent, labels):
                c2 = tick_to_axes.transform((x, y))
                if in_01(c2[0]) and in_01(c2[1]):
                    yield [x, y], *np.rad2deg([normal, tangent]), lab

        return iter_major(), iter([])

    def get_line_transform(self, axes):
        return axes.transData

    def get_line(self, axes):
        self.update_lim(axes)
        x, y = self._grid_info["line_xy"]
        return Path(np.column_stack([x, y]))


class GridHelperCurveLinear(GridHelperBase):
    def __init__(self, aux_trans,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        """
        Parameters
        ----------
        aux_trans : `.Transform` or tuple[Callable, Callable]
            The transform from curved coordinates to rectilinear coordinate:
            either a `.Transform` instance (which provides also its inverse),
            or a pair of callables ``(trans, inv_trans)`` that define the
            transform and its inverse.  The callables should have signature::

                x_rect, y_rect = trans(x_curved, y_curved)
                x_curved, y_curved = inv_trans(x_rect, y_rect)

        extreme_finder

        grid_locator1, grid_locator2
            Grid locators for each axis.

        tick_formatter1, tick_formatter2
            Tick formatters for each axis.
        """
        super().__init__()
        self._grid_info = None
        self.grid_finder = GridFinder(aux_trans,
                                      extreme_finder,
                                      grid_locator1,
                                      grid_locator2,
                                      tick_formatter1,
                                      tick_formatter2)

    def update_grid_finder(self, aux_trans=None, **kwargs):
        if aux_trans is not None:
            self.grid_finder.update_transform(aux_trans)
        self.grid_finder.update(**kwargs)
        self._old_limits = None  # Force revalidation.

    @_api.make_keyword_only("3.9", "nth_coord")
    def new_fixed_axis(
            self, loc, nth_coord=None, axis_direction=None, offset=None, axes=None):
        if axes is None:
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        helper = FixedAxisArtistHelper(self, loc, nth_coord_ticks=nth_coord)
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        # Why is clip not set on axisline, unlike in new_floating_axis or in
        # the floating_axig.GridHelperCurveLinear subclass?
        return axisline

    def new_floating_axis(self, nth_coord, value, axes=None, axis_direction="bottom"):
        if axes is None:
            axes = self.axes
        helper = FloatingAxisArtistHelper(
            self, nth_coord, value, axis_direction)
        axisline = AxisArtist(axes, helper)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        # axisline.major_ticklabels.set_visible(True)
        # axisline.minor_ticklabels.set_visible(False)
        return axisline

    def _update_grid(self, x1, y1, x2, y2):
        self._grid_info = self.grid_finder.get_grid_info(x1, y1, x2, y2)

    def get_gridlines(self, which="major", axis="both"):
        grid_lines = []
        if axis in ["both", "x"]:
            for gl in self._grid_info["lon"]["lines"]:
                grid_lines.extend(gl)
        if axis in ["both", "y"]:
            for gl in self._grid_info["lat"]["lines"]:
                grid_lines.extend(gl)
        return grid_lines

    @_api.deprecated("3.9")
    def get_tick_iterator(self, nth_coord, axis_side, minor=False):
        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
        lon_or_lat = ["lon", "lat"][nth_coord]
        if not minor:  # major ticks
            for tick in self._grid_info[lon_or_lat]["ticks"][axis_side]:
                yield *tick["loc"], angle_tangent, tick["label"]
        else:
            for tick in self._grid_info[lon_or_lat]["ticks"][axis_side]:
                yield *tick["loc"], angle_tangent, ""
<|endoftext|>"""
An experimental support for curvilinear grid.
"""

# TODO :
# see if tick_iterator method can be simplified by reusing the parent method.

import functools

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.patches as mpatches
from matplotlib.path import Path

from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory

from . import axislines, grid_helper_curvelinear
from .axis_artist import AxisArtist
from .grid_finder import ExtremeFinderSimple


class FloatingAxisArtistHelper(
        grid_helper_curvelinear.FloatingAxisArtistHelper):
    pass


class FixedAxisArtistHelper(grid_helper_curvelinear.FloatingAxisArtistHelper):

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """
        lon1, lon2, lat1, lat2 = grid_helper.grid_finder.extreme_finder(*[None] * 5)
        value, nth_coord = _api.check_getitem(
            dict(left=(lon1, 0), right=(lon2, 0), bottom=(lat1, 1), top=(lat2, 1)),
            side=side)
        super().__init__(grid_helper, nth_coord, value, axis_direction=side)
        if nth_coord_ticks is None:
            nth_coord_ticks = nth_coord
        self.nth_coord_ticks = nth_coord_ticks

        self.value = value
        self.grid_helper = grid_helper
        self._side = side

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)
        self._grid_info = self.grid_helper._grid_info

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""

        grid_finder = self.grid_helper.grid_finder

        lat_levs, lat_n, lat_factor = self._grid_info["lat_info"]
        yy0 = lat_levs / lat_factor

        lon_levs, lon_n, lon_factor = self._grid_info["lon_info"]
        xx0 = lon_levs / lon_factor

        extremes = self.grid_helper.grid_finder.extreme_finder(*[None] * 5)
        xmin, xmax = sorted(extremes[:2])
        ymin, ymax = sorted(extremes[2:])

        def trf_xy(x, y):
            trf = grid_finder.get_transform() + axes.transData
            return trf.transform(np.column_stack(np.broadcast_arrays(x, y))).T

        if self.nth_coord == 0:
            mask = (ymin <= yy0) & (yy0 <= ymax)
            (xx1, yy1), (dxx1, dyy1), (dxx2, dyy2) = \
                grid_helper_curvelinear._value_and_jacobian(
                    trf_xy, self.value, yy0[mask], (xmin, xmax), (ymin, ymax))
            labels = self._grid_info["lat_labels"]

        elif self.nth_coord == 1:
            mask = (xmin <= xx0) & (xx0 <= xmax)
            (xx1, yy1), (dxx2, dyy2), (dxx1, dyy1) = \
                grid_helper_curvelinear._value_and_jacobian(
                    trf_xy, xx0[mask], self.value, (xmin, xmax), (ymin, ymax))
            labels = self._grid_info["lon_labels"]

        labels = [l for l, m in zip(labels, mask) if m]

        angle_normal = np.arctan2(dyy1, dxx1)
        angle_tangent = np.arctan2(dyy2, dxx2)
        mm = (dyy1 == 0) & (dxx1 == 0)  # points with degenerate normal
        angle_normal[mm] = angle_tangent[mm] + np.pi / 2

        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
        in_01 = functools.partial(
            mpl.transforms._interval_contains_close, (0, 1))

        def f1():
            for x, y, normal, tangent, lab \
                    in zip(xx1, yy1, angle_normal, angle_tangent, labels):
                c2 = tick_to_axes.transform((x, y))
                if in_01(c2[0]) and in_01(c2[1]):
                    yield [x, y], *np.rad2deg([normal, tangent]), lab

        return f1(), iter([])

    def get_line(self, axes):
        self.update_lim(axes)
        k, v = dict(left=("lon_lines0", 0),
                    right=("lon_lines0", 1),
                    bottom=("lat_lines0", 0),
                    top=("lat_lines0", 1))[self._side]
        xx, yy = self._grid_info[k][v]
        return Path(np.column_stack([xx, yy]))


class ExtremeFinderFixed(ExtremeFinderSimple):
    # docstring inherited

    def __init__(self, extremes):
        """
        This subclass always returns the same bounding box.

        Parameters
        ----------
        extremes : (float, float, float, float)
            The bounding box that this helper always returns.
        """
        self._extremes = extremes

    def __call__(self, transform_xy, x1, y1, x2, y2):
        # docstring inherited
        return self._extremes


class GridHelperCurveLinear(grid_helper_curvelinear.GridHelperCurveLinear):

    def __init__(self, aux_trans, extremes,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        # docstring inherited
        super().__init__(aux_trans,
                         extreme_finder=ExtremeFinderFixed(extremes),
                         grid_locator1=grid_locator1,
                         grid_locator2=grid_locator2,
                         tick_formatter1=tick_formatter1,
                         tick_formatter2=tick_formatter2)

    def new_fixed_axis(
            self, loc, nth_coord=None, axis_direction=None, offset=None, axes=None):
        if axes is None:
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        # This is not the same as the FixedAxisArtistHelper class used by
        # grid_helper_curvelinear.GridHelperCurveLinear.new_fixed_axis!
        helper = FixedAxisArtistHelper(
            self, loc, nth_coord_ticks=nth_coord)
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        # Perhaps should be moved to the base class?
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        return axisline

    # new_floating_axis will inherit the grid_helper's extremes.

    # def new_floating_axis(self, nth_coord, value, axes=None, axis_direction="bottom"):
    #     axis = super(GridHelperCurveLinear,
    #                  self).new_floating_axis(nth_coord,
    #                                          value, axes=axes,
    #                                          axis_direction=axis_direction)
    #     # set extreme values of the axis helper
    #     if nth_coord == 1:
    #         axis.get_helper().set_extremes(*self._extremes[:2])
    #     elif nth_coord == 0:
    #         axis.get_helper().set_extremes(*self._extremes[2:])
    #     return axis

    def _update_grid(self, x1, y1, x2, y2):
        if self._grid_info is None:
            self._grid_info = dict()

        grid_info = self._grid_info

        grid_finder = self.grid_finder
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)

        lon_min, lon_max = sorted(extremes[:2])
        lat_min, lat_max = sorted(extremes[2:])
        grid_info["extremes"] = lon_min, lon_max, lat_min, lat_max  # extremes

        lon_levs, lon_n, lon_factor = \
            grid_finder.grid_locator1(lon_min, lon_max)
        lon_levs = np.asarray(lon_levs)
        lat_levs, lat_n, lat_factor = \
            grid_finder.grid_locator2(lat_min, lat_max)
        lat_levs = np.asarray(lat_levs)

        grid_info["lon_info"] = lon_levs, lon_n, lon_factor
        grid_info["lat_info"] = lat_levs, lat_n, lat_factor

        grid_info["lon_labels"] = grid_finder._format_ticks(
            1, "bottom", lon_factor, lon_levs)
        grid_info["lat_labels"] = grid_finder._format_ticks(
            2, "bottom", lat_factor, lat_levs)

        lon_values = lon_levs[:lon_n] / lon_factor
        lat_values = lat_levs[:lat_n] / lat_factor

        lon_lines, lat_lines = grid_finder._get_raw_grid_lines(
            lon_values[(lon_min < lon_values) & (lon_values < lon_max)],
            lat_values[(lat_min < lat_values) & (lat_values < lat_max)],
            lon_min, lon_max, lat_min, lat_max)

        grid_info["lon_lines"] = lon_lines
        grid_info["lat_lines"] = lat_lines

        lon_lines, lat_lines = grid_finder._get_raw_grid_lines(
            # lon_min, lon_max, lat_min, lat_max)
            extremes[:2], extremes[2:], *extremes)

        grid_info["lon_lines0"] = lon_lines
        grid_info["lat_lines0"] = lat_lines

    def get_gridlines(self, which="major", axis="both"):
        grid_lines = []
        if axis in ["both", "x"]:
            grid_lines.extend(self._grid_info["lon_lines"])
        if axis in ["both", "y"]:
            grid_lines.extend(self._grid_info["lat_lines"])
        return grid_lines


class FloatingAxesBase:

    def __init__(self, *args, grid_helper, **kwargs):
        _api.check_isinstance(GridHelperCurveLinear, grid_helper=grid_helper)
        super().__init__(*args, grid_helper=grid_helper, **kwargs)
        self.set_aspect(1.)

    def _gen_axes_patch(self):
        # docstring inherited
        x0, x1, y0, y1 = self.get_grid_helper().grid_finder.extreme_finder(*[None] * 5)
        patch = mpatches.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        patch.get_path()._interpolation_steps = 100
        return patch

    def clear(self):
        super().clear()
        self.patch.set_transform(
            self.get_grid_helper().grid_finder.get_transform()
            + self.transData)
        # The original patch is not in the draw tree; it is only used for
        # clipping purposes.
        orig_patch = super()._gen_axes_patch()
        orig_patch.set_figure(self.get_figure(root=False))
        orig_patch.set_transform(self.transAxes)
        self.patch.set_clip_path(orig_patch)
        self.gridlines.set_clip_path(orig_patch)
        self.adjust_axes_lim()

    def adjust_axes_lim(self):
        bbox = self.patch.get_path().get_extents(
            # First transform to pixel coords, then to parent data coords.
            self.patch.get_transform() - self.transData)
        bbox = bbox.expanded(1.02, 1.02)
        self.set_xlim(bbox.xmin, bbox.xmax)
        self.set_ylim(bbox.ymin, bbox.ymax)


floatingaxes_class_factory = cbook._make_class_factory(FloatingAxesBase, "Floating{}")
FloatingAxes = floatingaxes_class_factory(host_axes_class_factory(axislines.Axes))
FloatingSubplot = FloatingAxes
<|endoftext|>"""
The :mod:`.axis_artist` module implements custom artists to draw axis elements
(axis lines and labels, tick lines and labels, grid lines).

Axis lines and labels and tick lines and labels are managed by the `AxisArtist`
class; grid lines are managed by the `GridlinesCollection` class.

There is one `AxisArtist` per Axis; it can be accessed through
the ``axis`` dictionary of the parent Axes (which should be a
`mpl_toolkits.axislines.Axes`), e.g. ``ax.axis["bottom"]``.

Children of the AxisArtist are accessed as attributes: ``.line`` and ``.label``
for the axis line and label, ``.major_ticks``, ``.major_ticklabels``,
``.minor_ticks``, ``.minor_ticklabels`` for the tick lines and labels (e.g.
``ax.axis["bottom"].line``).

Children properties (colors, fonts, line widths, etc.) can be set using
setters, e.g. ::

  # Make the major ticks of the bottom axis red.
  ax.axis["bottom"].major_ticks.set_color("red")

However, things like the locations of ticks, and their ticklabels need to be
changed from the side of the grid_helper.

axis_direction
--------------

`AxisArtist`, `AxisLabel`, `TickLabels` have an *axis_direction* attribute,
which adjusts the location, angle, etc. The *axis_direction* must be one of
"left", "right", "bottom", "top", and follows the Matplotlib convention for
rectangular axis.

For example, for the *bottom* axis (the left and right is relative to the
direction of the increasing coordinate),

* ticklabels and axislabel are on the right
* ticklabels and axislabel have text angle of 0
* ticklabels are baseline, center-aligned
* axislabel is top, center-aligned

The text angles are actually relative to (90 + angle of the direction to the
ticklabel), which gives 0 for bottom axis.

=================== ====== ======== ====== ========
Property            left   bottom   right  top
=================== ====== ======== ====== ========
ticklabel location  left   right    right  left
axislabel location  left   right    right  left
ticklabel angle     90     0        -90    180
axislabel angle     180    0        0      180
ticklabel va        center baseline center baseline
axislabel va        center top      center bottom
ticklabel ha        right  center   right  center
axislabel ha        right  center   right  center
=================== ====== ======== ====== ========

Ticks are by default direct opposite side of the ticklabels. To make ticks to
the same side of the ticklabels, ::

  ax.axis["bottom"].major_ticks.set_tick_out(True)

The following attributes can be customized (use the ``set_xxx`` methods):

* `Ticks`: ticksize, tick_out
* `TickLabels`: pad
* `AxisLabel`: pad
"""

# FIXME :
# angles are given in data coordinate - need to convert it to canvas coordinate


from operator import methodcaller

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
    Affine2D, Bbox, IdentityTransform, ScaledTranslation)

from .axisline_style import AxislineStyle


class AttributeCopier:
    def get_ref_artist(self):
        """
        Return the underlying artist that actually defines some properties
        (e.g., color) of this artist.
        """
        raise RuntimeError("get_ref_artist must overridden")

    def get_attribute_from_ref_artist(self, attr_name):
        getter = methodcaller("get_" + attr_name)
        prop = getter(super())
        return getter(self.get_ref_artist()) if prop == "auto" else prop


class Ticks(AttributeCopier, Line2D):
    """
    Ticks are derived from `.Line2D`, and note that ticks themselves
    are markers. Thus, you should use set_mec, set_mew, etc.

    To change the tick size (length), you need to use
    `set_ticksize`. To change the direction of the ticks (ticks are
    in opposite direction of ticklabels by default), use
    ``set_tick_out(False)``
    """

    def __init__(self, ticksize, tick_out=False, *, axis=None, **kwargs):
        self._ticksize = ticksize
        self.locs_angles_labels = []

        self.set_tick_out(tick_out)

        self._axis = axis
        if self._axis is not None:
            if "color" not in kwargs:
                kwargs["color"] = "auto"
            if "mew" not in kwargs and "markeredgewidth" not in kwargs:
                kwargs["markeredgewidth"] = "auto"

        Line2D.__init__(self, [0.], [0.], **kwargs)
        self.set_snap(True)

    def get_ref_artist(self):
        # docstring inherited
        return self._axis.majorTicks[0].tick1line

    def set_color(self, color):
        # docstring inherited
        # Unlike the base Line2D.set_color, this also supports "auto".
        if not cbook._str_equal(color, "auto"):
            mcolors._check_color_like(color=color)
        self._color = color
        self.stale = True

    def get_color(self):
        return self.get_attribute_from_ref_artist("color")

    def get_markeredgecolor(self):
        return self.get_attribute_from_ref_artist("markeredgecolor")

    def get_markeredgewidth(self):
        return self.get_attribute_from_ref_artist("markeredgewidth")

    def set_tick_out(self, b):
        """Set whether ticks are drawn inside or outside the axes."""
        self._tick_out = b

    def get_tick_out(self):
        """Return whether ticks are drawn inside or outside the axes."""
        return self._tick_out

    def set_ticksize(self, ticksize):
        """Set length of the ticks in points."""
        self._ticksize = ticksize

    def get_ticksize(self):
        """Return length of the ticks in points."""
        return self._ticksize

    def set_locs_angles(self, locs_angles):
        self.locs_angles = locs_angles

    _tickvert_path = Path([[0., 0.], [1., 0.]])

    def draw(self, renderer):
        if not self.get_visible():
            return

        gc = renderer.new_gc()
        gc.set_foreground(self.get_markeredgecolor())
        gc.set_linewidth(self.get_markeredgewidth())
        gc.set_alpha(self._alpha)

        path_trans = self.get_transform()
        marker_transform = (Affine2D()
                            .scale(renderer.points_to_pixels(self._ticksize)))
        if self.get_tick_out():
            marker_transform.rotate_deg(180)

        for loc, angle in self.locs_angles:
            locs = path_trans.transform_non_affine(np.array([loc]))
            if self.axes and not self.axes.viewLim.contains(*locs[0]):
                continue
            renderer.draw_markers(
                gc, self._tickvert_path,
                marker_transform + Affine2D().rotate_deg(angle),
                Path(locs), path_trans.get_affine())

        gc.restore()


class LabelBase(mtext.Text):
    """
    A base class for `.AxisLabel` and `.TickLabels`. The position and
    angle of the text are calculated by the offset_ref_angle,
    text_ref_angle, and offset_radius attributes.
    """

    def __init__(self, *args, **kwargs):
        self.locs_angles_labels = []
        self._ref_angle = 0
        self._offset_radius = 0.

        super().__init__(*args, **kwargs)

        self.set_rotation_mode("anchor")
        self._text_follow_ref_angle = True

    @property
    def _text_ref_angle(self):
        if self._text_follow_ref_angle:
            return self._ref_angle + 90
        else:
            return 0

    @property
    def _offset_ref_angle(self):
        return self._ref_angle

    _get_opposite_direction = {"left": "right",
                               "right": "left",
                               "top": "bottom",
                               "bottom": "top"}.__getitem__

    def draw(self, renderer):
        if not self.get_visible():
            return

        # save original and adjust some properties
        tr = self.get_transform()
        angle_orig = self.get_rotation()
        theta = np.deg2rad(self._offset_ref_angle)
        dd = self._offset_radius
        dx, dy = dd * np.cos(theta), dd * np.sin(theta)

        self.set_transform(tr + Affine2D().translate(dx, dy))
        self.set_rotation(self._text_ref_angle + angle_orig)
        super().draw(renderer)
        # restore original properties
        self.set_transform(tr)
        self.set_rotation(angle_orig)

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()

        # save original and adjust some properties
        tr = self.get_transform()
        angle_orig = self.get_rotation()
        theta = np.deg2rad(self._offset_ref_angle)
        dd = self._offset_radius
        dx, dy = dd * np.cos(theta), dd * np.sin(theta)

        self.set_transform(tr + Affine2D().translate(dx, dy))
        self.set_rotation(self._text_ref_angle + angle_orig)
        bbox = super().get_window_extent(renderer).frozen()
        # restore original properties
        self.set_transform(tr)
        self.set_rotation(angle_orig)

        return bbox


class AxisLabel(AttributeCopier, LabelBase):
    """
    Axis label. Derived from `.Text`. The position of the text is updated
    in the fly, so changing text position has no effect. Otherwise, the
    properties can be changed as a normal `.Text`.

    To change the pad between tick labels and axis label, use `set_pad`.
    """

    def __init__(self, *args, axis_direction="bottom", axis=None, **kwargs):
        self._axis = axis
        self._pad = 5
        self._external_pad = 0  # in pixels
        LabelBase.__init__(self, *args, **kwargs)
        self.set_axis_direction(axis_direction)

    def set_pad(self, pad):
        """
        Set the internal pad in points.

        The actual pad will be the sum of the internal pad and the
        external pad (the latter is set automatically by the `.AxisArtist`).

        Parameters
        ----------
        pad : float
            The internal pad in points.
        """
        self._pad = pad

    def get_pad(self):
        """
        Return the internal pad in points.

        See `.set_pad` for more details.
        """
        return self._pad

    def get_ref_artist(self):
        # docstring inherited
        return self._axis.label

    def get_text(self):
        # docstring inherited
        t = super().get_text()
        if t == "__from_axes__":
            return self._axis.label.get_text()
        return self._text

    _default_alignments = dict(left=("bottom", "center"),
                               right=("top", "center"),
                               bottom=("top", "center"),
                               top=("bottom", "center"))

    def set_default_alignment(self, d):
        """
        Set the default alignment. See `set_axis_direction` for details.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        va, ha = _api.check_getitem(self._default_alignments, d=d)
        self.set_va(va)
        self.set_ha(ha)

    _default_angles = dict(left=180,
                           right=0,
                           bottom=0,
                           top=180)

    def set_default_angle(self, d):
        """
        Set the default angle. See `set_axis_direction` for details.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        self.set_rotation(_api.check_getitem(self._default_angles, d=d))

    def set_axis_direction(self, d):
        """
        Adjust the text angle and text alignment of axis label
        according to the matplotlib convention.

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        axislabel angle          180        0         0          180
        axislabel va             center     top       center     bottom
        axislabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the text angles are actually relative to (90 + angle
        of the direction to the ticklabel), which gives 0 for bottom
        axis.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        self.set_default_alignment(d)
        self.set_default_angle(d)

    def get_color(self):
        return self.get_attribute_from_ref_artist("color")

    def draw(self, renderer):
        if not self.get_visible():
            return

        self._offset_radius = \
            self._external_pad + renderer.points_to_pixels(self.get_pad())

        super().draw(renderer)

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()
        if not self.get_visible():
            return

        r = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r

        bb = super().get_window_extent(renderer)

        return bb


class TickLabels(AxisLabel):  # mtext.Text
    """
    Tick labels. While derived from `.Text`, this single artist draws all
    ticklabels. As in `.AxisLabel`, the position of the text is updated
    in the fly, so changing text position has no effect. Otherwise,
    the properties can be changed as a normal `.Text`. Unlike the
    ticklabels of the mainline Matplotlib, properties of a single
    ticklabel alone cannot be modified.

    To change the pad between ticks and ticklabels, use `~.AxisLabel.set_pad`.
    """

    def __init__(self, *, axis_direction="bottom", **kwargs):
        super().__init__(**kwargs)
        self.set_axis_direction(axis_direction)
        self._axislabel_pad = 0

    def get_ref_artist(self):
        # docstring inherited
        return self._axis.get_ticklabels()[0]

    def set_axis_direction(self, label_direction):
        """
        Adjust the text angle and text alignment of ticklabels
        according to the Matplotlib convention.

        The *label_direction* must be one of [left, right, bottom, top].

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        ticklabel angle          90         0         -90        180
        ticklabel va             center     baseline  center     baseline
        ticklabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the text angles are actually relative to (90 + angle
        of the direction to the ticklabel), which gives 0 for bottom
        axis.

        Parameters
        ----------
        label_direction : {"left", "bottom", "right", "top"}

        """
        self.set_default_alignment(label_direction)
        self.set_default_angle(label_direction)
        self._axis_direction = label_direction

    def invert_axis_direction(self):
        label_direction = self._get_opposite_direction(self._axis_direction)
        self.set_axis_direction(label_direction)

    def _get_ticklabels_offsets(self, renderer, label_direction):
        """
        Calculate the ticklabel offsets from the tick and their total heights.

        The offset only takes account the offset due to the vertical alignment
        of the ticklabels: if axis direction is bottom and va is 'top', it will
        return 0; if va is 'baseline', it will return (height-descent).
        """
        whd_list = self.get_texts_widths_heights_descents(renderer)

        if not whd_list:
            return 0, 0

        r = 0
        va, ha = self.get_va(), self.get_ha()

        if label_direction == "left":
            pad = max(w for w, h, d in whd_list)
            if ha == "left":
                r = pad
            elif ha == "center":
                r = .5 * pad
        elif label_direction == "right":
            pad = max(w for w, h, d in whd_list)
            if ha == "right":
                r = pad
            elif ha == "center":
                r = .5 * pad
        elif label_direction == "bottom":
            pad = max(h for w, h, d in whd_list)
            if va == "bottom":
                r = pad
            elif va == "center":
                r = .5 * pad
            elif va == "baseline":
                max_ascent = max(h - d for w, h, d in whd_list)
                max_descent = max(d for w, h, d in whd_list)
                r = max_ascent
                pad = max_ascent + max_descent
        elif label_direction == "top":
            pad = max(h for w, h, d in whd_list)
            if va == "top":
                r = pad
            elif va == "center":
                r = .5 * pad
            elif va == "baseline":
                max_ascent = max(h - d for w, h, d in whd_list)
                max_descent = max(d for w, h, d in whd_list)
                r = max_descent
                pad = max_ascent + max_descent

        # r : offset
        # pad : total height of the ticklabels. This will be used to
        # calculate the pad for the axislabel.
        return r, pad

    _default_alignments = dict(left=("center", "right"),
                               right=("center", "left"),
                               bottom=("baseline", "center"),
                               top=("baseline", "center"))

    _default_angles = dict(left=90,
                           right=-90,
                           bottom=0,
                           top=180)

    def draw(self, renderer):
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return

        r, total_width = self._get_ticklabels_offsets(renderer,
                                                      self._axis_direction)

        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad

        for (x, y), a, l in self._locs_angles_labels:
            if not l.strip():
                continue
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            LabelBase.draw(self, renderer)

        # the value saved will be used to draw axislabel.
        self._axislabel_pad = total_width + pad

    def set_locs_angles_labels(self, locs_angles_labels):
        self._locs_angles_labels = locs_angles_labels

    def get_window_extents(self, renderer=None):
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()

        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return []

        bboxes = []

        r, total_width = self._get_ticklabels_offsets(renderer,
                                                      self._axis_direction)

        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad

        for (x, y), a, l in self._locs_angles_labels:
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            bb = LabelBase.get_window_extent(self, renderer)
            bboxes.append(bb)

        # the value saved will be used to draw axislabel.
        self._axislabel_pad = total_width + pad

        return bboxes

    def get_texts_widths_heights_descents(self, renderer):
        """
        Return a list of ``(width, height, descent)`` tuples for ticklabels.

        Empty labels are left out.
        """
        whd_list = []
        for _loc, _angle, label in self._locs_angles_labels:
            if not label.strip():
                continue
            clean_line, ismath = self._preprocess_math(label)
            whd = mtext._get_text_metrics_with_cache(
                renderer, clean_line, self._fontproperties, ismath=ismath,
                dpi=self.get_figure(root=True).dpi)
            whd_list.append(whd)
        return whd_list


class GridlinesCollection(LineCollection):
    def __init__(self, *args, which="major", axis="both", **kwargs):
        """
        Collection of grid lines.

        Parameters
        ----------
        which : {"major", "minor"}
            Which grid to consider.
        axis : {"both", "x", "y"}
            Which axis to consider.
        *args, **kwargs
            Passed to `.LineCollection`.
        """
        self._which = which
        self._axis = axis
        super().__init__(*args, **kwargs)
        self.set_grid_helper(None)

    def set_which(self, which):
        """
        Select major or minor grid lines.

        Parameters
        ----------
        which : {"major", "minor"}
        """
        self._which = which

    def set_axis(self, axis):
        """
        Select axis.

        Parameters
        ----------
        axis : {"both", "x", "y"}
        """
        self._axis = axis

    def set_grid_helper(self, grid_helper):
        """
        Set grid helper.

        Parameters
        ----------
        grid_helper : `.GridHelperBase` subclass
        """
        self._grid_helper = grid_helper

    def draw(self, renderer):
        if self._grid_helper is not None:
            self._grid_helper.update_lim(self.axes)
            gl = self._grid_helper.get_gridlines(self._which, self._axis)
            self.set_segments([np.transpose(l) for l in gl])
        super().draw(renderer)


class AxisArtist(martist.Artist):
    """
    An artist which draws axis (a line along which the n-th axes coord
    is constant) line, ticks, tick labels, and axis label.
    """

    zorder = 2.5

    @property
    def LABELPAD(self):
        return self.label.get_pad()

    @LABELPAD.setter
    def LABELPAD(self, v):
        self.label.set_pad(v)

    def __init__(self, axes,
                 helper,
                 offset=None,
                 axis_direction="bottom",
                 **kwargs):
        """
        Parameters
        ----------
        axes : `mpl_toolkits.axisartist.axislines.Axes`
        helper : `~mpl_toolkits.axisartist.axislines.AxisArtistHelper`
        """
        # axes is also used to follow the axis attribute (tick color, etc).

        super().__init__(**kwargs)

        self.axes = axes

        self._axis_artist_helper = helper

        if offset is None:
            offset = (0, 0)
        self.offset_transform = ScaledTranslation(
            *offset,
            Affine2D().scale(1 / 72)  # points to inches.
            + self.axes.get_figure(root=False).dpi_scale_trans)

        if axis_direction in ["left", "right"]:
            self.axis = axes.yaxis
        else:
            self.axis = axes.xaxis

        self._axisline_style = None
        self._axis_direction = axis_direction

        self._init_line()
        self._init_ticks(**kwargs)
        self._init_offsetText(axis_direction)
        self._init_label()

        # axis direction
        self._ticklabel_add_angle = 0.
        self._axislabel_add_angle = 0.
        self.set_axis_direction(axis_direction)

    # axis direction

    def set_axis_direction(self, axis_direction):
        """
        Adjust the direction, text angle, and text alignment of tick labels
        and axis labels following the Matplotlib convention for the rectangle
        axes.

        The *axis_direction* must be one of [left, right, bottom, top].

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        ticklabel direction      "-"        "+"       "+"        "-"
        axislabel direction      "-"        "+"       "+"        "-"
        ticklabel angle          90         0         -90        180
        ticklabel va             center     baseline  center     baseline
        ticklabel ha             right      center    right      center
        axislabel angle          180        0         0          180
        axislabel va             center     top       center     bottom
        axislabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the direction "+" and "-" are relative to the direction of
        the increasing coordinate. Also, the text angles are actually
        relative to (90 + angle of the direction to the ticklabel),
        which gives 0 for bottom axis.

        Parameters
        ----------
        axis_direction : {"left", "bottom", "right", "top"}
        """
        self.major_ticklabels.set_axis_direction(axis_direction)
        self.label.set_axis_direction(axis_direction)
        self._axis_direction = axis_direction
        if axis_direction in ["left", "top"]:
            self.set_ticklabel_direction("-")
            self.set_axislabel_direction("-")
        else:
            self.set_ticklabel_direction("+")
            self.set_axislabel_direction("+")

    def set_ticklabel_direction(self, tick_direction):
        r"""
        Adjust the direction of the tick labels.

        Note that the *tick_direction*\s '+' and '-' are relative to the
        direction of the increasing coordinate.

        Parameters
        ----------
        tick_direction : {"+", "-"}
        """
        self._ticklabel_add_angle = _api.check_getitem(
            {"+": 0, "-": 180}, tick_direction=tick_direction)

    def invert_ticklabel_direction(self):
        self._ticklabel_add_angle = (self._ticklabel_add_angle + 180) % 360
        self.major_ticklabels.invert_axis_direction()
        self.minor_ticklabels.invert_axis_direction()

    def set_axislabel_direction(self, label_direction):
        r"""
        Adjust the direction of the axis label.

        Note that the *label_direction*\s '+' and '-' are relative to the
        direction of the increasing coordinate.

        Parameters
        ----------
        label_direction : {"+", "-"}
        """
        self._axislabel_add_angle = _api.check_getitem(
            {"+": 0, "-": 180}, label_direction=label_direction)

    def get_transform(self):
        return self.axes.transAxes + self.offset_transform

    def get_helper(self):
        """
        Return axis artist helper instance.
        """
        return self._axis_artist_helper

    def set_axisline_style(self, axisline_style=None, **kwargs):
        """
        Set the axisline style.

        The new style is completely defined by the passed attributes. Existing
        style attributes are forgotten.

        Parameters
        ----------
        axisline_style : str or None
            The line style, e.g. '->', optionally followed by a comma-separated
            list of attributes. Alternatively, the attributes can be provided
            as keywords.

            If *None* this returns a string containing the available styles.

        Examples
        --------
        The following two commands are equal:

        >>> set_axisline_style("->,size=1.5")
        >>> set_axisline_style("->", size=1.5)
        """
        if axisline_style is None:
            return AxislineStyle.pprint_styles()

        if isinstance(axisline_style, AxislineStyle._Base):
            self._axisline_style = axisline_style
        else:
            self._axisline_style = AxislineStyle(axisline_style, **kwargs)

        self._init_line()

    def get_axisline_style(self):
        """Return the current axisline style."""
        return self._axisline_style

    def _init_line(self):
        """
        Initialize the *line* artist that is responsible to draw the axis line.
        """
        tran = (self._axis_artist_helper.get_line_transform(self.axes)
                + self.offset_transform)

        axisline_style = self.get_axisline_style()
        if axisline_style is None:
            self.line = PathPatch(
                self._axis_artist_helper.get_line(self.axes),
                color=mpl.rcParams['axes.edgecolor'],
                fill=False,
                linewidth=mpl.rcParams['axes.linewidth'],
                capstyle=mpl.rcParams['lines.solid_capstyle'],
                joinstyle=mpl.rcParams['lines.solid_joinstyle'],
                transform=tran)
        else:
            self.line = axisline_style(self, transform=tran)

    def _draw_line(self, renderer):
        self.line.set_path(self._axis_artist_helper.get_line(self.axes))
        if self.get_axisline_style() is not None:
            self.line.set_line_mutation_scale(self.major_ticklabels.get_size())
        self.line.draw(renderer)

    def _init_ticks(self, **kwargs):
        axis_name = self.axis.axis_name

        trans = (self._axis_artist_helper.get_tick_transform(self.axes)
                 + self.offset_transform)

        self.major_ticks = Ticks(
            kwargs.get(
                "major_tick_size",
                mpl.rcParams[f"{axis_name}tick.major.size"]),
            axis=self.axis, transform=trans)
        self.minor_ticks = Ticks(
            kwargs.get(
                "minor_tick_size",
                mpl.rcParams[f"{axis_name}tick.minor.size"]),
            axis=self.axis, transform=trans)

        size = mpl.rcParams[f"{axis_name}tick.labelsize"]
        self.major_ticklabels = TickLabels(
            axis=self.axis,
            axis_direction=self._axis_direction,
            figure=self.axes.get_figure(root=False),
            transform=trans,
            fontsize=size,
            pad=kwargs.get(
                "major_tick_pad", mpl.rcParams[f"{axis_name}tick.major.pad"]),
        )
        self.minor_ticklabels = TickLabels(
            axis=self.axis,
            axis_direction=self._axis_direction,
            figure=self.axes.get_figure(root=False),
            transform=trans,
            fontsize=size,
            pad=kwargs.get(
                "minor_tick_pad", mpl.rcParams[f"{axis_name}tick.minor.pad"]),
        )

    def _get_tick_info(self, tick_iter):
        """
        Return a pair of:

        - list of locs and angles for ticks
        - list of locs, angles and labels for ticklabels.
        """
        ticks_loc_angle = []
        ticklabels_loc_angle_label = []

        ticklabel_add_angle = self._ticklabel_add_angle

        for loc, angle_normal, angle_tangent, label in tick_iter:
            angle_label = angle_tangent - 90 + ticklabel_add_angle
            angle_tick = (angle_normal
                          if 90 <= (angle_label - angle_normal) % 360 <= 270
                          else angle_normal + 180)
            ticks_loc_angle.append([loc, angle_tick])
            ticklabels_loc_angle_label.append([loc, angle_label, label])

        return ticks_loc_angle, ticklabels_loc_angle_label

    def _update_ticks(self, renderer=None):
        # set extra pad for major and minor ticklabels: use ticksize of
        # majorticks even for minor ticks. not clear what is best.

        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()

        dpi_cor = renderer.points_to_pixels(1.)
        if self.major_ticks.get_visible() and self.major_ticks.get_tick_out():
            ticklabel_pad = self.major_ticks._ticksize * dpi_cor
            self.major_ticklabels._external_pad = ticklabel_pad
            self.minor_ticklabels._external_pad = ticklabel_pad
        else:
            self.major_ticklabels._external_pad = 0
            self.minor_ticklabels._external_pad = 0

        majortick_iter, minortick_iter = \
            self._axis_artist_helper.get_tick_iterators(self.axes)

        tick_loc_angle, ticklabel_loc_angle_label = \
            self._get_tick_info(majortick_iter)
        self.major_ticks.set_locs_angles(tick_loc_angle)
        self.major_ticklabels.set_locs_angles_labels(ticklabel_loc_angle_label)

        tick_loc_angle, ticklabel_loc_angle_label = \
            self._get_tick_info(minortick_iter)
        self.minor_ticks.set_locs_angles(tick_loc_angle)
        self.minor_ticklabels.set_locs_angles_labels(ticklabel_loc_angle_label)

    def _draw_ticks(self, renderer):
        self._update_ticks(renderer)
        self.major_ticks.draw(renderer)
        self.major_ticklabels.draw(renderer)
        self.minor_ticks.draw(renderer)
        self.minor_ticklabels.draw(renderer)
        if (self.major_ticklabels.get_visible()
                or self.minor_ticklabels.get_visible()):
            self._draw_offsetText(renderer)

    _offsetText_pos = dict(left=(0, 1, "bottom", "right"),
                           right=(1, 1, "bottom", "left"),
                           bottom=(1, 0, "top", "right"),
                           top=(1, 1, "bottom", "right"))

    def _init_offsetText(self, direction):
        x, y, va, ha = self._offsetText_pos[direction]
        self.offsetText = mtext.Annotation(
            "",
            xy=(x, y), xycoords="axes fraction",
            xytext=(0, 0), textcoords="offset points",
            color=mpl.rcParams['xtick.color'],
            horizontalalignment=ha, verticalalignment=va,
        )
        self.offsetText.set_transform(IdentityTransform())
        self.axes._set_artist_props(self.offsetText)

    def _update_offsetText(self):
        self.offsetText.set_text(self.axis.major.formatter.get_offset())
        self.offsetText.set_size(self.major_ticklabels.get_size())
        offset = (self.major_ticklabels.get_pad()
                  + self.major_ticklabels.get_size()
                  + 2)
        self.offsetText.xyann = (0, offset)

    def _draw_offsetText(self, renderer):
        self._update_offsetText()
        self.offsetText.draw(renderer)

    def _init_label(self, **kwargs):
        tr = (self._axis_artist_helper.get_axislabel_transform(self.axes)
              + self.offset_transform)
        self.label = AxisLabel(
            0, 0, "__from_axes__",
            color="auto",
            fontsize=kwargs.get("labelsize", mpl.rcParams['axes.labelsize']),
            fontweight=mpl.rcParams['axes.labelweight'],
            axis=self.axis,
            transform=tr,
            axis_direction=self._axis_direction,
        )
        self.label.set_figure(self.axes.get_figure(root=False))
        labelpad = kwargs.get("labelpad", 5)
        self.label.set_pad(labelpad)

    def _update_label(self, renderer):
        if not self.label.get_visible():
            return

        if self._ticklabel_add_angle != self._axislabel_add_angle:
            if ((self.major_ticks.get_visible()
                 and not self.major_ticks.get_tick_out())
                or (self.minor_ticks.get_visible()
                    and not self.major_ticks.get_tick_out())):
                axislabel_pad = self.major_ticks._ticksize
            else:
                axislabel_pad = 0
        else:
            axislabel_pad = max(self.major_ticklabels._axislabel_pad,
                                self.minor_ticklabels._axislabel_pad)

        self.label._external_pad = axislabel_pad

        xy, angle_tangent = \
            self._axis_artist_helper.get_axislabel_pos_angle(self.axes)
        if xy is None:
            return

        angle_label = angle_tangent - 90

        x, y = xy
        self.label._ref_angle = angle_label + self._axislabel_add_angle
        self.label.set(x=x, y=y)

    def _draw_label(self, renderer):
        self._update_label(renderer)
        self.label.draw(renderer)

    def set_label(self, s):
        # docstring inherited
        self.label.set_text(s)

    def get_tightbbox(self, renderer=None):
        if not self.get_visible():
            return
        self._axis_artist_helper.update_lim(self.axes)
        self._update_ticks(renderer)
        self._update_label(renderer)

        self.line.set_path(self._axis_artist_helper.get_line(self.axes))
        if self.get_axisline_style() is not None:
            self.line.set_line_mutation_scale(self.major_ticklabels.get_size())

        bb = [
            *self.major_ticklabels.get_window_extents(renderer),
            *self.minor_ticklabels.get_window_extents(renderer),
            self.label.get_window_extent(renderer),
            self.offsetText.get_window_extent(renderer),
            self.line.get_window_extent(renderer),
        ]
        bb = [b for b in bb if b and (b.width != 0 or b.height != 0)]
        if bb:
            _bbox = Bbox.union(bb)
            return _bbox
        else:
            return None

    @martist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return
        renderer.open_group(__name__, gid=self.get_gid())
        self._axis_artist_helper.update_lim(self.axes)
        self._draw_ticks(renderer)
        self._draw_line(renderer)
        self._draw_label(renderer)
        renderer.close_group(__name__)

    def toggle(self, all=None, ticks=None, ticklabels=None, label=None):
        """
        Toggle visibility of ticks, ticklabels, and (axis) label.
        To turn all off, ::

          axis.toggle(all=False)

        To turn all off but ticks on ::

          axis.toggle(all=False, ticks=True)

        To turn all on but (axis) label off ::

          axis.toggle(all=True, label=False)

        """
        if all:
            _ticks, _ticklabels, _label = True, True, True
        elif all is not None:
            _ticks, _ticklabels, _label = False, False, False
        else:
            _ticks, _ticklabels, _label = None, None, None

        if ticks is not None:
            _ticks = ticks
        if ticklabels is not None:
            _ticklabels = ticklabels
        if label is not None:
            _label = label

        if _ticks is not None:
            self.major_ticks.set_visible(_ticks)
            self.minor_ticks.set_visible(_ticks)
        if _ticklabels is not None:
            self.major_ticklabels.set_visible(_ticklabels)
            self.minor_ticklabels.set_visible(_ticklabels)
        if _label is not None:
            self.label.set_visible(_label)
<|endoftext|>import numpy as np

from matplotlib import ticker as mticker, _api
from matplotlib.transforms import Bbox, Transform


def _find_line_box_crossings(xys, bbox):
    """
    Find the points where a polyline crosses a bbox, and the crossing angles.

    Parameters
    ----------
    xys : (N, 2) array
        The polyline coordinates.
    bbox : `.Bbox`
        The bounding box.

    Returns
    -------
    list of ((float, float), float)
        Four separate lists of crossings, for the left, right, bottom, and top
        sides of the bbox, respectively.  For each list, the entries are the
        ``((x, y), ccw_angle_in_degrees)`` of the crossing, where an angle of 0
        means that the polyline is moving to the right at the crossing point.

        The entries are computed by linearly interpolating at each crossing
        between the nearest points on either side of the bbox edges.
    """
    crossings = []
    dxys = xys[1:] - xys[:-1]
    for sl in [slice(None), slice(None, None, -1)]:
        us, vs = xys.T[sl]  # "this" coord, "other" coord
        dus, dvs = dxys.T[sl]
        umin, vmin = bbox.min[sl]
        umax, vmax = bbox.max[sl]
        for u0, inside in [(umin, us > umin), (umax, us < umax)]:
            cross = []
            idxs, = (inside[:-1] ^ inside[1:]).nonzero()
            for idx in idxs:
                v = vs[idx] + (u0 - us[idx]) * dvs[idx] / dus[idx]
                if not vmin <= v <= vmax:
                    continue
                crossing = (u0, v)[sl]
                theta = np.degrees(np.arctan2(*dxys[idx][::-1]))
                cross.append((crossing, theta))
            crossings.append(cross)
    return crossings


class ExtremeFinderSimple:
    """
    A helper class to figure out the range of grid lines that need to be drawn.
    """

    def __init__(self, nx, ny):
        """
        Parameters
        ----------
        nx, ny : int
            The number of samples in each direction.
        """
        self.nx = nx
        self.ny = ny

    def __call__(self, transform_xy, x1, y1, x2, y2):
        """
        Compute an approximation of the bounding box obtained by applying
        *transform_xy* to the box delimited by ``(x1, y1, x2, y2)``.

        The intended use is to have ``(x1, y1, x2, y2)`` in axes coordinates,
        and have *transform_xy* be the transform from axes coordinates to data
        coordinates; this method then returns the range of data coordinates
        that span the actual axes.

        The computation is done by sampling ``nx * ny`` equispaced points in
        the ``(x1, y1, x2, y2)`` box and finding the resulting points with
        extremal coordinates; then adding some padding to take into account the
        finite sampling.

        As each sampling step covers a relative range of *1/nx* or *1/ny*,
        the padding is computed by expanding the span covered by the extremal
        coordinates by these fractions.
        """
        x, y = np.meshgrid(
            np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        xt, yt = transform_xy(np.ravel(x), np.ravel(y))
        return self._add_pad(xt.min(), xt.max(), yt.min(), yt.max())

    def _add_pad(self, x_min, x_max, y_min, y_max):
        """Perform the padding mentioned in `__call__`."""
        dx = (x_max - x_min) / self.nx
        dy = (y_max - y_min) / self.ny
        return x_min - dx, x_max + dx, y_min - dy, y_max + dy


class _User2DTransform(Transform):
    """A transform defined by two user-set functions."""

    input_dims = output_dims = 2

    def __init__(self, forward, backward):
        """
        Parameters
        ----------
        forward, backward : callable
            The forward and backward transforms, taking ``x`` and ``y`` as
            separate arguments and returning ``(tr_x, tr_y)``.
        """
        # The normal Matplotlib convention would be to take and return an
        # (N, 2) array but axisartist uses the transposed version.
        super().__init__()
        self._forward = forward
        self._backward = backward

    def transform_non_affine(self, values):
        # docstring inherited
        return np.transpose(self._forward(*np.transpose(values)))

    def inverted(self):
        # docstring inherited
        return type(self)(self._backward, self._forward)


class GridFinder:
    """
    Internal helper for `~.grid_helper_curvelinear.GridHelperCurveLinear`, with
    the same constructor parameters; should not be directly instantiated.
    """

    def __init__(self,
                 transform,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        if extreme_finder is None:
            extreme_finder = ExtremeFinderSimple(20, 20)
        if grid_locator1 is None:
            grid_locator1 = MaxNLocator()
        if grid_locator2 is None:
            grid_locator2 = MaxNLocator()
        if tick_formatter1 is None:
            tick_formatter1 = FormatterPrettyPrint()
        if tick_formatter2 is None:
            tick_formatter2 = FormatterPrettyPrint()
        self.extreme_finder = extreme_finder
        self.grid_locator1 = grid_locator1
        self.grid_locator2 = grid_locator2
        self.tick_formatter1 = tick_formatter1
        self.tick_formatter2 = tick_formatter2
        self.set_transform(transform)

    def _format_ticks(self, idx, direction, factor, levels):
        """
        Helper to support both standard formatters (inheriting from
        `.mticker.Formatter`) and axisartist-specific ones; should be called instead of
        directly calling ``self.tick_formatter1`` and ``self.tick_formatter2``.  This
        method should be considered as a temporary workaround which will be removed in
        the future at the same time as axisartist-specific formatters.
        """
        fmt = _api.check_getitem(
            {1: self.tick_formatter1, 2: self.tick_formatter2}, idx=idx)
        return (fmt.format_ticks(levels) if isinstance(fmt, mticker.Formatter)
                else fmt(direction, factor, levels))

    def get_grid_info(self, x1, y1, x2, y2):
        """
        lon_values, lat_values : list of grid values. if integer is given,
                           rough number of grids in each direction.
        """

        extremes = self.extreme_finder(self.inv_transform_xy, x1, y1, x2, y2)

        # min & max rage of lat (or lon) for each grid line will be drawn.
        # i.e., gridline of lon=0 will be drawn from lat_min to lat_max.

        lon_min, lon_max, lat_min, lat_max = extremes
        lon_levs, lon_n, lon_factor = self.grid_locator1(lon_min, lon_max)
        lon_levs = np.asarray(lon_levs)
        lat_levs, lat_n, lat_factor = self.grid_locator2(lat_min, lat_max)
        lat_levs = np.asarray(lat_levs)

        lon_values = lon_levs[:lon_n] / lon_factor
        lat_values = lat_levs[:lat_n] / lat_factor

        lon_lines, lat_lines = self._get_raw_grid_lines(lon_values,
                                                        lat_values,
                                                        lon_min, lon_max,
                                                        lat_min, lat_max)

        bb = Bbox.from_extents(x1, y1, x2, y2).expanded(1 + 2e-10, 1 + 2e-10)

        grid_info = {
            "extremes": extremes,
            # "lon", "lat", filled below.
        }

        for idx, lon_or_lat, levs, factor, values, lines in [
                (1, "lon", lon_levs, lon_factor, lon_values, lon_lines),
                (2, "lat", lat_levs, lat_factor, lat_values, lat_lines),
        ]:
            grid_info[lon_or_lat] = gi = {
                "lines": [[l] for l in lines],
                "ticks": {"left": [], "right": [], "bottom": [], "top": []},
            }
            for (lx, ly), v, level in zip(lines, values, levs):
                all_crossings = _find_line_box_crossings(np.column_stack([lx, ly]), bb)
                for side, crossings in zip(
                        ["left", "right", "bottom", "top"], all_crossings):
                    for crossing in crossings:
                        gi["ticks"][side].append({"level": level, "loc": crossing})
            for side in gi["ticks"]:
                levs = [tick["level"] for tick in gi["ticks"][side]]
                labels = self._format_ticks(idx, side, factor, levs)
                for tick, label in zip(gi["ticks"][side], labels):
                    tick["label"] = label

        return grid_info

    def _get_raw_grid_lines(self,
                            lon_values, lat_values,
                            lon_min, lon_max, lat_min, lat_max):

        lons_i = np.linspace(lon_min, lon_max, 100)  # for interpolation
        lats_i = np.linspace(lat_min, lat_max, 100)

        lon_lines = [self.transform_xy(np.full_like(lats_i, lon), lats_i)
                     for lon in lon_values]
        lat_lines = [self.transform_xy(lons_i, np.full_like(lons_i, lat))
                     for lat in lat_values]

        return lon_lines, lat_lines

    def set_transform(self, aux_trans):
        if isinstance(aux_trans, Transform):
            self._aux_transform = aux_trans
        elif len(aux_trans) == 2 and all(map(callable, aux_trans)):
            self._aux_transform = _User2DTransform(*aux_trans)
        else:
            raise TypeError("'aux_trans' must be either a Transform "
                            "instance or a pair of callables")

    def get_transform(self):
        return self._aux_transform

    update_transform = set_transform  # backcompat alias.

    def transform_xy(self, x, y):
        return self._aux_transform.transform(np.column_stack([x, y])).T

    def inv_transform_xy(self, x, y):
        return self._aux_transform.inverted().transform(
            np.column_stack([x, y])).T

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in ["extreme_finder",
                     "grid_locator1",
                     "grid_locator2",
                     "tick_formatter1",
                     "tick_formatter2"]:
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown update property {k!r}")


class MaxNLocator(mticker.MaxNLocator):
    def __init__(self, nbins=10, steps=None,
                 trim=True,
                 integer=False,
                 symmetric=False,
                 prune=None):
        # trim argument has no effect. It has been left for API compatibility
        super().__init__(nbins, steps=steps, integer=integer,
                         symmetric=symmetric, prune=prune)
        self.create_dummy_axis()

    def __call__(self, v1, v2):
        locs = super().tick_values(v1, v2)
        return np.array(locs), len(locs), 1  # 1: factor (see angle_helper)


class FixedLocator:
    def __init__(self, locs):
        self._locs = locs

    def __call__(self, v1, v2):
        v1, v2 = sorted([v1, v2])
        locs = np.array([l for l in self._locs if v1 <= l <= v2])
        return locs, len(locs), 1  # 1: factor (see angle_helper)


# Tick Formatter

class FormatterPrettyPrint:
    def __init__(self, useMathText=True):
        self._fmt = mticker.ScalarFormatter(
            useMathText=useMathText, useOffset=False)
        self._fmt.create_dummy_axis()

    def __call__(self, direction, factor, values):
        return self._fmt.format_ticks(values)


class DictFormatter:
    def __init__(self, format_dict, formatter=None):
        """
        format_dict : dictionary for format strings to be used.
        formatter : fall-back formatter
        """
        super().__init__()
        self._format_dict = format_dict
        self._fallback_formatter = formatter

    def __call__(self, direction, factor, values):
        """
        factor is ignored if value is found in the dictionary
        """
        if self._fallback_formatter:
            fallback_strings = self._fallback_formatter(
                direction, factor, values)
        else:
            fallback_strings = [""] * len(values)
        return [self._format_dict.get(k, v)
                for k, v in zip(values, fallback_strings)]
<|endoftext|># don't import any costly modules
import os
import sys

report_url = (
    "https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml"
)


def warn_distutils_present():
    if 'distutils' not in sys.modules:
        return
    import warnings

    warnings.warn(
        "Distutils was imported before Setuptools, but importing Setuptools "
        "also replaces the `distutils` module in `sys.modules`. This may lead "
        "to undesirable behaviors or errors. To avoid these issues, avoid "
        "using distutils directly, ensure that setuptools is installed in the "
        "traditional way (e.g. not an editable install), and/or make sure "
        "that setuptools is always imported before distutils."
    )


def clear_distutils():
    if 'distutils' not in sys.modules:
        return
    import warnings

    warnings.warn(
        "Setuptools is replacing distutils. Support for replacing "
        "an already imported distutils is deprecated. In the future, "
        "this condition will fail. "
        f"Register concerns at {report_url}"
    )
    mods = [
        name
        for name in sys.modules
        if name == "distutils" or name.startswith("distutils.")
    ]
    for name in mods:
        del sys.modules[name]


def enabled():
    """
    Allow selection of distutils by environment variable.
    """
    which = os.environ.get('SETUPTOOLS_USE_DISTUTILS', 'local')
    if which == 'stdlib':
        import warnings

        warnings.warn(
            "Reliance on distutils from stdlib is deprecated. Users "
            "must rely on setuptools to provide the distutils module. "
            "Avoid importing distutils or import setuptools first, "
            "and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. "
            f"Register concerns at {report_url}"
        )
    return which == 'local'


def ensure_local_distutils():
    import importlib

    clear_distutils()

    # With the DistutilsMetaFinder in place,
    # perform an import to cause distutils to be
    # loaded from setuptools._distutils. Ref #2906.
    with shim():
        importlib.import_module('distutils')

    # check that submodules load as expected
    core = importlib.import_module('distutils.core')
    assert '_distutils' in core.__file__, core.__file__
    assert 'setuptools._distutils.log' not in sys.modules


def do_override():
    """
    Ensure that the local copy of distutils is preferred over stdlib.

    See https://github.com/pypa/setuptools/issues/417#issuecomment-392298401
    for more motivation.
    """
    if enabled():
        warn_distutils_present()
        ensure_local_distutils()


class _TrivialRe:
    def __init__(self, *patterns) -> None:
        self._patterns = patterns

    def match(self, string):
        return all(pat in string for pat in self._patterns)


class DistutilsMetaFinder:
    def find_spec(self, fullname, path, target=None):
        # optimization: only consider top level modules and those
        # found in the CPython test suite.
        if path is not None and not fullname.startswith('test.'):
            return None

        method_name = 'spec_for_{fullname}'.format(**locals())
        method = getattr(self, method_name, lambda: None)
        return method()

    def spec_for_distutils(self):
        if self.is_cpython():
            return None

        import importlib
        import importlib.abc
        import importlib.util

        try:
            mod = importlib.import_module('setuptools._distutils')
        except Exception:  # noqa: BLE001 # intentional broad fallback
            # There are a couple of cases where setuptools._distutils
            # may not be present:
            # - An older Setuptools without a local distutils is
            #   taking precedence. Ref #2957.
            # - Path manipulation during sitecustomize removes
            #   setuptools from the path but only after the hook
            #   has been loaded. Ref #2980.
            # In either case, fall back to stdlib behavior.
            return None

        class DistutilsLoader(importlib.abc.Loader):
            def create_module(self, spec):
                mod.__name__ = 'distutils'
                return mod

            def exec_module(self, module):
                pass

        return importlib.util.spec_from_loader(
            'distutils', DistutilsLoader(), origin=mod.__file__
        )

    @staticmethod
    def is_cpython():
        """
        Suppress supplying distutils for CPython (build and tests).
        Ref #2965 and #3007.
        """
        return os.path.isfile('pybuilddir.txt')

    def spec_for_pip(self):
        """
        Ensure stdlib distutils when running under pip.
        See pypa/pip#8761 for rationale.
        """
        if sys.version_info >= (3, 12) or self.pip_imported_during_build():
            return
        clear_distutils()
        self.spec_for_distutils = lambda: None

    @classmethod
    def pip_imported_during_build(cls):
        """
        Detect if pip is being imported in a build script. Ref #2355.
        """
        import traceback

        return any(
            cls.frame_file_is_setup(frame) for frame, line in traceback.walk_stack(None)
        )

    @staticmethod
    def frame_file_is_setup(frame):
        """
        Return True if the indicated frame suggests a setup.py file.
        """
        # some frames may not have __file__ (#2940)
        return frame.f_globals.get('__file__', '').endswith('setup.py')

    def spec_for_sensitive_tests(self):
        """
        Ensure stdlib distutils when running select tests under CPython.

        python/cpython#91169
        """
        clear_distutils()
        self.spec_for_distutils = lambda: None

    sensitive_tests = (
        [
            'test.test_distutils',
            'test.test_peg_generator',
            'test.test_importlib',
        ]
        if sys.version_info < (3, 10)
        else [
            'test.test_distutils',
        ]
    )


for name in DistutilsMetaFinder.sensitive_tests:
    setattr(
        DistutilsMetaFinder,
        f'spec_for_{name}',
        DistutilsMetaFinder.spec_for_sensitive_tests,
    )


DISTUTILS_FINDER = DistutilsMetaFinder()


def add_shim():
    DISTUTILS_FINDER in sys.meta_path or insert_shim()


class shim:
    def __enter__(self) -> None:
        insert_shim()

    def __exit__(self, exc: object, value: object, tb: object) -> None:
        _remove_shim()


def insert_shim():
    sys.meta_path.insert(0, DISTUTILS_FINDER)


def _remove_shim():
    try:
        sys.meta_path.remove(DISTUTILS_FINDER)
    except ValueError:
        pass


if sys.version_info < (3, 12):
    # DistutilsMetaFinder can only be disabled in Python < 3.12 (PEP 632)
    remove_shim = _remove_shim
<|endoftext|>from __future__ import annotations

import inspect
import logging
import os
import shutil
import uuid

from .asyn import AsyncFileSystem, _run_coros_in_chunks, sync_wrapper
from .callbacks import DEFAULT_CALLBACK
from .core import filesystem, get_filesystem_class, split_protocol, url_to_fs

_generic_fs = {}
logger = logging.getLogger("fsspec.generic")


def set_generic_fs(protocol, **storage_options):
    """Populate the dict used for method=="generic" lookups"""
    _generic_fs[protocol] = filesystem(protocol, **storage_options)


def _resolve_fs(url, method, protocol=None, storage_options=None):
    """Pick instance of backend FS"""
    url = url[0] if isinstance(url, (list, tuple)) else url
    protocol = protocol or split_protocol(url)[0]
    storage_options = storage_options or {}
    if method == "default":
        return filesystem(protocol)
    if method == "generic":
        return _generic_fs[protocol]
    if method == "current":
        cls = get_filesystem_class(protocol)
        return cls.current()
    if method == "options":
        fs, _ = url_to_fs(url, **storage_options.get(protocol, {}))
        return fs
    raise ValueError(f"Unknown FS resolution method: {method}")


def rsync(
    source,
    destination,
    delete_missing=False,
    source_field="size",
    dest_field="size",
    update_cond="different",
    inst_kwargs=None,
    fs=None,
    **kwargs,
):
    """Sync files between two directory trees

    (experimental)

    Parameters
    ----------
    source: str
        Root of the directory tree to take files from. This must be a directory, but
        do not include any terminating "/" character
    destination: str
        Root path to copy into. The contents of this location should be
        identical to the contents of ``source`` when done. This will be made a
        directory, and the terminal "/" should not be included.
    delete_missing: bool
        If there are paths in the destination that don't exist in the
        source and this is True, delete them. Otherwise, leave them alone.
    source_field: str | callable
        If ``update_field`` is "different", this is the key in the info
        of source files to consider for difference. Maybe a function of the
        info dict.
    dest_field: str | callable
        If ``update_field`` is "different", this is the key in the info
        of destination files to consider for difference. May be a function of
        the info dict.
    update_cond: "different"|"always"|"never"
        If "always", every file is copied, regardless of whether it exists in
        the destination. If "never", files that exist in the destination are
        not copied again. If "different" (default), only copy if the info
        fields given by ``source_field`` and ``dest_field`` (usually "size")
        are different. Other comparisons may be added in the future.
    inst_kwargs: dict|None
        If ``fs`` is None, use this set of keyword arguments to make a
        GenericFileSystem instance
    fs: GenericFileSystem|None
        Instance to use if explicitly given. The instance defines how to
        to make downstream file system instances from paths.

    Returns
    -------
    dict of the copy operations that were performed, {source: destination}
    """
    fs = fs or GenericFileSystem(**(inst_kwargs or {}))
    source = fs._strip_protocol(source)
    destination = fs._strip_protocol(destination)
    allfiles = fs.find(source, withdirs=True, detail=True)
    if not fs.isdir(source):
        raise ValueError("Can only rsync on a directory")
    otherfiles = fs.find(destination, withdirs=True, detail=True)
    dirs = [
        a
        for a, v in allfiles.items()
        if v["type"] == "directory" and a.replace(source, destination) not in otherfiles
    ]
    logger.debug(f"{len(dirs)} directories to create")
    if dirs:
        fs.make_many_dirs(
            [dirn.replace(source, destination) for dirn in dirs], exist_ok=True
        )
    allfiles = {a: v for a, v in allfiles.items() if v["type"] == "file"}
    logger.debug(f"{len(allfiles)} files to consider for copy")
    to_delete = [
        o
        for o, v in otherfiles.items()
        if o.replace(destination, source) not in allfiles and v["type"] == "file"
    ]
    for k, v in allfiles.copy().items():
        otherfile = k.replace(source, destination)
        if otherfile in otherfiles:
            if update_cond == "always":
                allfiles[k] = otherfile
            elif update_cond == "never":
                allfiles.pop(k)
            elif update_cond == "different":
                inf1 = source_field(v) if callable(source_field) else v[source_field]
                v2 = otherfiles[otherfile]
                inf2 = dest_field(v2) if callable(dest_field) else v2[dest_field]
                if inf1 != inf2:
                    # details mismatch, make copy
                    allfiles[k] = otherfile
                else:
                    # details match, don't copy
                    allfiles.pop(k)
        else:
            # file not in target yet
            allfiles[k] = otherfile
    logger.debug(f"{len(allfiles)} files to copy")
    if allfiles:
        source_files, target_files = zip(*allfiles.items())
        fs.cp(source_files, target_files, **kwargs)
    logger.debug(f"{len(to_delete)} files to delete")
    if delete_missing and to_delete:
        fs.rm(to_delete)
    return allfiles


class GenericFileSystem(AsyncFileSystem):
    """Wrapper over all other FS types

    <experimental!>

    This implementation is a single unified interface to be able to run FS operations
    over generic URLs, and dispatch to the specific implementations using the URL
    protocol prefix.

    Note: instances of this FS are always async, even if you never use it with any async
    backend.
    """

    protocol = "generic"  # there is no real reason to ever use a protocol with this FS

    def __init__(self, default_method="default", storage_options=None, **kwargs):
        """

        Parameters
        ----------
        default_method: str (optional)
            Defines how to configure backend FS instances. Options are:
            - "default": instantiate like FSClass(), with no
              extra arguments; this is the default instance of that FS, and can be
              configured via the config system
            - "generic": takes instances from the `_generic_fs` dict in this module,
              which you must populate before use. Keys are by protocol
            - "options": expects storage_options, a dict mapping protocol to
              kwargs to use when constructing the filesystem
            - "current": takes the most recently instantiated version of each FS
        """
        self.method = default_method
        self.st_opts = storage_options
        super().__init__(**kwargs)

    def _parent(self, path):
        fs = _resolve_fs(path, self.method, storage_options=self.st_opts)
        return fs.unstrip_protocol(fs._parent(path))

    def _strip_protocol(self, path):
        # normalization only
        fs = _resolve_fs(path, self.method, storage_options=self.st_opts)
        return fs.unstrip_protocol(fs._strip_protocol(path))

    async def _find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        fs = _resolve_fs(path, self.method, storage_options=self.st_opts)
        if fs.async_impl:
            out = await fs._find(
                path, maxdepth=maxdepth, withdirs=withdirs, detail=True, **kwargs
            )
        else:
            out = fs.find(
                path, maxdepth=maxdepth, withdirs=withdirs, detail=True, **kwargs
            )
        result = {}
        for k, v in out.items():
            v = v.copy()  # don't corrupt target FS dircache
            name = fs.unstrip_protocol(k)
            v["name"] = name
            result[name] = v
        if detail:
            return result
        return list(result)

    async def _info(self, url, **kwargs):
        fs = _resolve_fs(url, self.method, storage_options=self.st_opts)
        if fs.async_impl:
            out = await fs._info(url, **kwargs)
        else:
            out = fs.info(url, **kwargs)
        out = out.copy()  # don't edit originals
        out["name"] = fs.unstrip_protocol(out["name"])
        return out

    async def _ls(
        self,
        url,
        detail=True,
        **kwargs,
    ):
        fs = _resolve_fs(url, self.method, storage_options=self.st_opts)
        if fs.async_impl:
            out = await fs._ls(url, detail=True, **kwargs)
        else:
            out = fs.ls(url, detail=True, **kwargs)
        out = [o.copy() for o in out]  # don't edit originals
        for o in out:
            o["name"] = fs.unstrip_protocol(o["name"])
        if detail:
            return out
        else:
            return [o["name"] for o in out]

    async def _cat_file(
        self,
        url,
        **kwargs,
    ):
        fs = _resolve_fs(url, self.method, storage_options=self.st_opts)
        if fs.async_impl:
            return await fs._cat_file(url, **kwargs)
        else:
            return fs.cat_file(url, **kwargs)

    async def _pipe_file(
        self,
        path,
        value,
        **kwargs,
    ):
        fs = _resolve_fs(path, self.method, storage_options=self.st_opts)
        if fs.async_impl:
            return await fs._pipe_file(path, value, **kwargs)
        else:
            return fs.pipe_file(path, value, **kwargs)

    async def _rm(self, url, **kwargs):
        urls = url
        if isinstance(urls, str):
            urls = [urls]
        fs = _resolve_fs(urls[0], self.method, storage_options=self.st_opts)
        if fs.async_impl:
            await fs._rm(urls, **kwargs)
        else:
            fs.rm(url, **kwargs)

    async def _makedirs(self, path, exist_ok=False):
        logger.debug("Make dir %s", path)
        fs = _resolve_fs(path, self.method, storage_options=self.st_opts)
        if fs.async_impl:
            await fs._makedirs(path, exist_ok=exist_ok)
        else:
            fs.makedirs(path, exist_ok=exist_ok)

    def rsync(self, source, destination, **kwargs):
        """Sync files between two directory trees

        See `func:rsync` for more details.
        """
        rsync(source, destination, fs=self, **kwargs)

    async def _cp_file(
        self,
        url,
        url2,
        blocksize=2**20,
        callback=DEFAULT_CALLBACK,
        tempdir: str | None = None,
        **kwargs,
    ):
        fs = _resolve_fs(url, self.method, storage_options=self.st_opts)
        fs2 = _resolve_fs(url2, self.method, storage_options=self.st_opts)
        if fs is fs2:
            # pure remote
            if fs.async_impl:
                return await fs._copy(url, url2, **kwargs)
            else:
                return fs.copy(url, url2, **kwargs)
        await copy_file_op(fs, [url], fs2, [url2], tempdir, 1, on_error="raise")

    async def _make_many_dirs(self, urls, exist_ok=True):
        fs = _resolve_fs(urls[0], self.method, storage_options=self.st_opts)
        if fs.async_impl:
            coros = [fs._makedirs(u, exist_ok=exist_ok) for u in urls]
            await _run_coros_in_chunks(coros)
        else:
            for u in urls:
                fs.makedirs(u, exist_ok=exist_ok)

    make_many_dirs = sync_wrapper(_make_many_dirs)

    async def _copy(
        self,
        path1: list[str],
        path2: list[str],
        recursive: bool = False,
        on_error: str = "ignore",
        maxdepth: int | None = None,
        batch_size: int | None = None,
        tempdir: str | None = None,
        **kwargs,
    ):
        # TODO: special case for one FS being local, which can use get/put
        # TODO: special case for one being memFS, which can use cat/pipe
        if recursive:
            raise NotImplementedError("Please use fsspec.generic.rsync")
        path1 = [path1] if isinstance(path1, str) else path1
        path2 = [path2] if isinstance(path2, str) else path2

        fs = _resolve_fs(path1, self.method, storage_options=self.st_opts)
        fs2 = _resolve_fs(path2, self.method, storage_options=self.st_opts)

        if fs is fs2:
            if fs.async_impl:
                return await fs._copy(path1, path2, **kwargs)
            else:
                return fs.copy(path1, path2, **kwargs)

        await copy_file_op(
            fs, path1, fs2, path2, tempdir, batch_size, on_error=on_error
        )


async def copy_file_op(
    fs1, url1, fs2, url2, tempdir=None, batch_size=20, on_error="ignore"
):
    import tempfile

    tempdir = tempdir or tempfile.mkdtemp()
    try:
        coros = [
            _copy_file_op(
                fs1,
                u1,
                fs2,
                u2,
                os.path.join(tempdir, uuid.uuid4().hex),
            )
            for u1, u2 in zip(url1, url2)
        ]
        out = await _run_coros_in_chunks(
            coros, batch_size=batch_size, return_exceptions=True
        )
    finally:
        shutil.rmtree(tempdir)
    if on_error == "return":
        return out
    elif on_error == "raise":
        for o in out:
            if isinstance(o, Exception):
                raise o


async def _copy_file_op(fs1, url1, fs2, url2, local, on_error="ignore"):
    if fs1.async_impl:
        await fs1._get_file(url1, local)
    else:
        fs1.get_file(url1, local)
    if fs2.async_impl:
        await fs2._put_file(local, url2)
    else:
        fs2.put_file(local, url2)
    os.unlink(local)
    logger.debug("Copy %s -> %s; done", url1, url2)


async def maybe_await(cor):
    if inspect.iscoroutine(cor):
        return await cor
    else:
        return cor
<|endoftext|>import argparse
import logging
import os
import stat
import threading
import time
from errno import EIO, ENOENT

from fuse import FUSE, FuseOSError, LoggingMixIn, Operations

from fsspec import __version__
from fsspec.core import url_to_fs

logger = logging.getLogger("fsspec.fuse")


class FUSEr(Operations):
    def __init__(self, fs, path, ready_file=False):
        self.fs = fs
        self.cache = {}
        self.root = path.rstrip("/") + "/"
        self.counter = 0
        logger.info("Starting FUSE at %s", path)
        self._ready_file = ready_file

    def getattr(self, path, fh=None):
        logger.debug("getattr %s", path)
        if self._ready_file and path in ["/.fuse_ready", ".fuse_ready"]:
            return {"type": "file", "st_size": 5}

        path = "".join([self.root, path.lstrip("/")]).rstrip("/")
        try:
            info = self.fs.info(path)
        except FileNotFoundError as exc:
            raise FuseOSError(ENOENT) from exc

        data = {"st_uid": info.get("uid", 1000), "st_gid": info.get("gid", 1000)}
        perm = info.get("mode", 0o777)

        if info["type"] != "file":
            data["st_mode"] = stat.S_IFDIR | perm
            data["st_size"] = 0
            data["st_blksize"] = 0
        else:
            data["st_mode"] = stat.S_IFREG | perm
            data["st_size"] = info["size"]
            data["st_blksize"] = 5 * 2**20
            data["st_nlink"] = 1
        data["st_atime"] = info["atime"] if "atime" in info else time.time()
        data["st_ctime"] = info["ctime"] if "ctime" in info else time.time()
        data["st_mtime"] = info["mtime"] if "mtime" in info else time.time()
        return data

    def readdir(self, path, fh):
        logger.debug("readdir %s", path)
        path = "".join([self.root, path.lstrip("/")])
        files = self.fs.ls(path, False)
        files = [os.path.basename(f.rstrip("/")) for f in files]
        return [".", ".."] + files

    def mkdir(self, path, mode):
        path = "".join([self.root, path.lstrip("/")])
        self.fs.mkdir(path)
        return 0

    def rmdir(self, path):
        path = "".join([self.root, path.lstrip("/")])
        self.fs.rmdir(path)
        return 0

    def read(self, path, size, offset, fh):
        logger.debug("read %s", (path, size, offset))
        if self._ready_file and path in ["/.fuse_ready", ".fuse_ready"]:
            # status indicator
            return b"ready"

        f = self.cache[fh]
        f.seek(offset)
        out = f.read(size)
        return out

    def write(self, path, data, offset, fh):
        logger.debug("write %s", (path, offset))
        f = self.cache[fh]
        f.seek(offset)
        f.write(data)
        return len(data)

    def create(self, path, flags, fi=None):
        logger.debug("create %s", (path, flags))
        fn = "".join([self.root, path.lstrip("/")])
        self.fs.touch(fn)  # OS will want to get attributes immediately
        f = self.fs.open(fn, "wb")
        self.cache[self.counter] = f
        self.counter += 1
        return self.counter - 1

    def open(self, path, flags):
        logger.debug("open %s", (path, flags))
        fn = "".join([self.root, path.lstrip("/")])
        if flags % 2 == 0:
            # read
            mode = "rb"
        else:
            # write/create
            mode = "wb"
        self.cache[self.counter] = self.fs.open(fn, mode)
        self.counter += 1
        return self.counter - 1

    def truncate(self, path, length, fh=None):
        fn = "".join([self.root, path.lstrip("/")])
        if length != 0:
            raise NotImplementedError
        # maybe should be no-op since open with write sets size to zero anyway
        self.fs.touch(fn)

    def unlink(self, path):
        fn = "".join([self.root, path.lstrip("/")])
        try:
            self.fs.rm(fn, False)
        except (OSError, FileNotFoundError) as exc:
            raise FuseOSError(EIO) from exc

    def release(self, path, fh):
        try:
            if fh in self.cache:
                f = self.cache[fh]
                f.close()
                self.cache.pop(fh)
        except Exception as e:
            print(e)
        return 0

    def chmod(self, path, mode):
        if hasattr(self.fs, "chmod"):
            path = "".join([self.root, path.lstrip("/")])
            return self.fs.chmod(path, mode)
        raise NotImplementedError


def run(
    fs,
    path,
    mount_point,
    foreground=True,
    threads=False,
    ready_file=False,
    ops_class=FUSEr,
):
    """Mount stuff in a local directory

    This uses fusepy to make it appear as if a given path on an fsspec
    instance is in fact resident within the local file-system.

    This requires that fusepy by installed, and that FUSE be available on
    the system (typically requiring a package to be installed with
    apt, yum, brew, etc.).

    Parameters
    ----------
    fs: file-system instance
        From one of the compatible implementations
    path: str
        Location on that file-system to regard as the root directory to
        mount. Note that you typically should include the terminating "/"
        character.
    mount_point: str
        An empty directory on the local file-system where the contents of
        the remote path will appear.
    foreground: bool
        Whether or not calling this function will block. Operation will
        typically be more stable if True.
    threads: bool
        Whether or not to create threads when responding to file operations
        within the mounter directory. Operation will typically be more
        stable if False.
    ready_file: bool
        Whether the FUSE process is ready. The ``.fuse_ready`` file will
        exist in the ``mount_point`` directory if True. Debugging purpose.
    ops_class: FUSEr or Subclass of FUSEr
        To override the default behavior of FUSEr. For Example, logging
        to file.

    """
    func = lambda: FUSE(
        ops_class(fs, path, ready_file=ready_file),
        mount_point,
        nothreads=not threads,
        foreground=foreground,
    )
    if not foreground:
        th = threading.Thread(target=func)
        th.daemon = True
        th.start()
        return th
    else:  # pragma: no cover
        try:
            func()
        except KeyboardInterrupt:
            pass


def main(args):
    """Mount filesystem from chained URL to MOUNT_POINT.

    Examples:

    python3 -m fsspec.fuse memory /usr/share /tmp/mem

    python3 -m fsspec.fuse local /tmp/source /tmp/local \\
            -l /tmp/fsspecfuse.log

    You can also mount chained-URLs and use special settings:

    python3 -m fsspec.fuse 'filecache::zip::file://data.zip' \\
            / /tmp/zip \\
            -o 'filecache-cache_storage=/tmp/simplecache'

    You can specify the type of the setting by using `[int]` or `[bool]`,
    (`true`, `yes`, `1` represents the Boolean value `True`):

    python3 -m fsspec.fuse 'simplecache::ftp://ftp1.at.proftpd.org' \\
            /historic/packages/RPMS /tmp/ftp \\
            -o 'simplecache-cache_storage=/tmp/simplecache' \\
            -o 'simplecache-check_files=false[bool]' \\
            -o 'ftp-listings_expiry_time=60[int]' \\
            -o 'ftp-username=anonymous' \\
            -o 'ftp-password=xieyanbo'
    """

    class RawDescriptionArgumentParser(argparse.ArgumentParser):
        def format_help(self):
            usage = super().format_help()
            parts = usage.split("\n\n")
            parts[1] = self.description.rstrip()
            return "\n\n".join(parts)

    parser = RawDescriptionArgumentParser(prog="fsspec.fuse", description=main.__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("url", type=str, help="fs url")
    parser.add_argument("source_path", type=str, help="source directory in fs")
    parser.add_argument("mount_point", type=str, help="local directory")
    parser.add_argument(
        "-o",
        "--option",
        action="append",
        help="Any options of protocol included in the chained URL",
    )
    parser.add_argument(
        "-l", "--log-file", type=str, help="Logging FUSE debug info (Default: '')"
    )
    parser.add_argument(
        "-f",
        "--foreground",
        action="store_false",
        help="Running in foreground or not (Default: False)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        action="store_false",
        help="Running with threads support (Default: False)",
    )
    parser.add_argument(
        "-r",
        "--ready-file",
        action="store_false",
        help="The `.fuse_ready` file will exist after FUSE is ready. "
        "(Debugging purpose, Default: False)",
    )
    args = parser.parse_args(args)

    kwargs = {}
    for item in args.option or []:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(message=f"Wrong option: {item!r}")
        val = value.lower()
        if val.endswith("[int]"):
            value = int(value[: -len("[int]")])
        elif val.endswith("[bool]"):
            value = val[: -len("[bool]")] in ["1", "yes", "true"]

        if "-" in key:
            fs_name, setting_name = key.split("-", 1)
            if fs_name in kwargs:
                kwargs[fs_name][setting_name] = value
            else:
                kwargs[fs_name] = {setting_name: value}
        else:
            kwargs[key] = value

    if args.log_file:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=args.log_file,
            format="%(asctime)s %(message)s",
        )

        class LoggingFUSEr(FUSEr, LoggingMixIn):
            pass

        fuser = LoggingFUSEr
    else:
        fuser = FUSEr

    fs, url_path = url_to_fs(args.url, **kwargs)
    logger.debug("Mounting %s to %s", url_path, str(args.mount_point))
    run(
        fs,
        args.source_path,
        args.mount_point,
        foreground=args.foreground,
        threads=args.threads,
        ready_file=args.ready_file,
        ops_class=fuser,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
<|endoftext|>from __future__ import annotations

import importlib
import types
import warnings

__all__ = ["registry", "get_filesystem_class", "default"]

# internal, mutable
_registry: dict[str, type] = {}

# external, immutable
registry = types.MappingProxyType(_registry)
default = "file"


def register_implementation(name, cls, clobber=False, errtxt=None):
    """Add implementation class to the registry

    Parameters
    ----------
    name: str
        Protocol name to associate with the class
    cls: class or str
        if a class: fsspec-compliant implementation class (normally inherits from
        ``fsspec.AbstractFileSystem``, gets added straight to the registry. If a
        str, the full path to an implementation class like package.module.class,
        which gets added to known_implementations,
        so the import is deferred until the filesystem is actually used.
    clobber: bool (optional)
        Whether to overwrite a protocol with the same name; if False, will raise
        instead.
    errtxt: str (optional)
        If given, then a failure to import the given class will result in this
        text being given.
    """
    if isinstance(cls, str):
        if name in known_implementations and clobber is False:
            if cls != known_implementations[name]["class"]:
                raise ValueError(
                    f"Name ({name}) already in the known_implementations and clobber "
                    f"is False"
                )
        else:
            known_implementations[name] = {
                "class": cls,
                "err": errtxt or f"{cls} import failed for protocol {name}",
            }

    else:
        if name in registry and clobber is False:
            if _registry[name] is not cls:
                raise ValueError(
                    f"Name ({name}) already in the registry and clobber is False"
                )
        else:
            _registry[name] = cls


# protocols mapped to the class which implements them. This dict can be
# updated with register_implementation
known_implementations = {
    "abfs": {
        "class": "adlfs.AzureBlobFileSystem",
        "err": "Install adlfs to access Azure Datalake Gen2 and Azure Blob Storage",
    },
    "adl": {
        "class": "adlfs.AzureDatalakeFileSystem",
        "err": (
            "Azure Data Lake Storage Gen1 is retired and no longer supported. Please "
            "install adlfs and use the `az://` protocol to access Azure Blob Storage "
            "and Azure Data Lake Storage Gen2 instead."
        ),
    },
    "arrow_hdfs": {
        "class": "fsspec.implementations.arrow.HadoopFileSystem",
        "err": "pyarrow and local java libraries required for HDFS",
    },
    "async_wrapper": {
        "class": "fsspec.implementations.asyn_wrapper.AsyncFileSystemWrapper",
    },
    "asynclocal": {
        "class": "morefs.asyn_local.AsyncLocalFileSystem",
        "err": "Install 'morefs[asynclocalfs]' to use AsyncLocalFileSystem",
    },
    "asyncwrapper": {
        "class": "fsspec.implementations.asyn_wrapper.AsyncFileSystemWrapper",
    },
    "az": {
        "class": "adlfs.AzureBlobFileSystem",
        "err": "Install adlfs to access Azure Datalake Gen2 and Azure Blob Storage",
    },
    "blockcache": {"class": "fsspec.implementations.cached.CachingFileSystem"},
    "box": {
        "class": "boxfs.BoxFileSystem",
        "err": "Please install boxfs to access BoxFileSystem",
    },
    "cached": {"class": "fsspec.implementations.cached.CachingFileSystem"},
    "dask": {
        "class": "fsspec.implementations.dask.DaskWorkerFileSystem",
        "err": "Install dask distributed to access worker file system",
    },
    "data": {"class": "fsspec.implementations.data.DataFileSystem"},
    "dbfs": {
        "class": "fsspec.implementations.dbfs.DatabricksFileSystem",
        "err": "Install the requests package to use the DatabricksFileSystem",
    },
    "dir": {"class": "fsspec.implementations.dirfs.DirFileSystem"},
    "dropbox": {
        "class": "dropboxdrivefs.DropboxDriveFileSystem",
        "err": (
            'DropboxFileSystem requires "dropboxdrivefs","requests" and "'
            '"dropbox" to be installed'
        ),
    },
    "dvc": {
        "class": "dvc.api.DVCFileSystem",
        "err": "Install dvc to access DVCFileSystem",
    },
    "file": {"class": "fsspec.implementations.local.LocalFileSystem"},
    "filecache": {"class": "fsspec.implementations.cached.WholeFileCacheFileSystem"},
    "ftp": {"class": "fsspec.implementations.ftp.FTPFileSystem"},
    "gcs": {
        "class": "gcsfs.GCSFileSystem",
        "err": "Please install gcsfs to access Google Storage",
    },
    "gdrive": {
        "class": "gdrive_fsspec.GoogleDriveFileSystem",
        "err": "Please install gdrive_fs for access to Google Drive",
    },
    "generic": {"class": "fsspec.generic.GenericFileSystem"},
    "gist": {
        "class": "fsspec.implementations.gist.GistFileSystem",
        "err": "Install the requests package to use the gist FS",
    },
    "git": {
        "class": "fsspec.implementations.git.GitFileSystem",
        "err": "Install pygit2 to browse local git repos",
    },
    "github": {
        "class": "fsspec.implementations.github.GithubFileSystem",
        "err": "Install the requests package to use the github FS",
    },
    "gs": {
        "class": "gcsfs.GCSFileSystem",
        "err": "Please install gcsfs to access Google Storage",
    },
    "hdfs": {
        "class": "fsspec.implementations.arrow.HadoopFileSystem",
        "err": "pyarrow and local java libraries required for HDFS",
    },
    "hf": {
        "class": "huggingface_hub.HfFileSystem",
        "err": "Install huggingface_hub to access HfFileSystem",
    },
    "http": {
        "class": "fsspec.implementations.http.HTTPFileSystem",
        "err": 'HTTPFileSystem requires "requests" and "aiohttp" to be installed',
    },
    "https": {
        "class": "fsspec.implementations.http.HTTPFileSystem",
        "err": 'HTTPFileSystem requires "requests" and "aiohttp" to be installed',
    },
    "jlab": {
        "class": "fsspec.implementations.jupyter.JupyterFileSystem",
        "err": "Jupyter FS requires requests to be installed",
    },
    "jupyter": {
        "class": "fsspec.implementations.jupyter.JupyterFileSystem",
        "err": "Jupyter FS requires requests to be installed",
    },
    "lakefs": {
        "class": "lakefs_spec.LakeFSFileSystem",
        "err": "Please install lakefs-spec to access LakeFSFileSystem",
    },
    "libarchive": {
        "class": "fsspec.implementations.libarchive.LibArchiveFileSystem",
        "err": "LibArchive requires to be installed",
    },
    "local": {"class": "fsspec.implementations.local.LocalFileSystem"},
    "memory": {"class": "fsspec.implementations.memory.MemoryFileSystem"},
    "oci": {
        "class": "ocifs.OCIFileSystem",
        "err": "Install ocifs to access OCI Object Storage",
    },
    "ocilake": {
        "class": "ocifs.OCIFileSystem",
        "err": "Install ocifs to access OCI Data Lake",
    },
    "oss": {
        "class": "ossfs.OSSFileSystem",
        "err": "Install ossfs to access Alibaba Object Storage System",
    },
    "pyscript": {
        "class": "pyscript_fsspec_client.client.PyscriptFileSystem",
        "err": "This only runs in a pyscript context",
    },
    "reference": {"class": "fsspec.implementations.reference.ReferenceFileSystem"},
    "root": {
        "class": "fsspec_xrootd.XRootDFileSystem",
        "err": (
            "Install fsspec-xrootd to access xrootd storage system. "
            "Note: 'root' is the protocol name for xrootd storage systems, "
            "not referring to root directories"
        ),
    },
    "s3": {"class": "s3fs.S3FileSystem", "err": "Install s3fs to access S3"},
    "s3a": {"class": "s3fs.S3FileSystem", "err": "Install s3fs to access S3"},
    "sftp": {
        "class": "fsspec.implementations.sftp.SFTPFileSystem",
        "err": 'SFTPFileSystem requires "paramiko" to be installed',
    },
    "simplecache": {"class": "fsspec.implementations.cached.SimpleCacheFileSystem"},
    "smb": {
        "class": "fsspec.implementations.smb.SMBFileSystem",
        "err": 'SMB requires "smbprotocol" or "smbprotocol[kerberos]" installed',
    },
    "ssh": {
        "class": "fsspec.implementations.sftp.SFTPFileSystem",
        "err": 'SFTPFileSystem requires "paramiko" to be installed',
    },
    "tar": {"class": "fsspec.implementations.tar.TarFileSystem"},
    "tos": {
        "class": "tosfs.TosFileSystem",
        "err": "Install tosfs to access ByteDance volcano engine Tinder Object Storage",
    },
    "tosfs": {
        "class": "tosfs.TosFileSystem",
        "err": "Install tosfs to access ByteDance volcano engine Tinder Object Storage",
    },
    "wandb": {"class": "wandbfs.WandbFS", "err": "Install wandbfs to access wandb"},
    "webdav": {
        "class": "webdav4.fsspec.WebdavFileSystem",
        "err": "Install webdav4 to access WebDAV",
    },
    "webhdfs": {
        "class": "fsspec.implementations.webhdfs.WebHDFS",
        "err": 'webHDFS access requires "requests" to be installed',
    },
    "zip": {"class": "fsspec.implementations.zip.ZipFileSystem"},
}

assert list(known_implementations) == sorted(known_implementations), (
    "Not in alphabetical order"
)


def get_filesystem_class(protocol):
    """Fetch named protocol implementation from the registry

    The dict ``known_implementations`` maps protocol names to the locations
    of classes implementing the corresponding file-system. When used for the
    first time, appropriate imports will happen and the class will be placed in
    the registry. All subsequent calls will fetch directly from the registry.

    Some protocol implementations require additional dependencies, and so the
    import may fail. In this case, the string in the "err" field of the
    ``known_implementations`` will be given as the error message.
    """
    if not protocol:
        protocol = default

    if protocol not in registry:
        if protocol not in known_implementations:
            raise ValueError(f"Protocol not known: {protocol}")
        bit = known_implementations[protocol]
        try:
            register_implementation(protocol, _import_class(bit["class"]))
        except ImportError as e:
            raise ImportError(bit.get("err")) from e
    cls = registry[protocol]
    if getattr(cls, "protocol", None) in ("abstract", None):
        cls.protocol = protocol

    return cls


s3_msg = """Your installed version of s3fs is very old and known to cause
severe performance issues, see also https://github.com/dask/dask/issues/10276

To fix, you should specify a lower version bound on s3fs, or
update the current installation.
"""


def _import_class(fqp: str):
    """Take a fully-qualified path and return the imported class or identifier.

    ``fqp`` is of the form "package.module.klass" or
    "package.module:subobject.klass".

    Warnings
    --------
    This can import arbitrary modules. Make sure you haven't installed any modules
    that may execute malicious code at import time.
    """
    if ":" in fqp:
        mod, name = fqp.rsplit(":", 1)
    else:
        mod, name = fqp.rsplit(".", 1)

    is_s3 = mod == "s3fs"
    mod = importlib.import_module(mod)
    if is_s3 and mod.__version__.split(".") < ["0", "5"]:
        warnings.warn(s3_msg)
    for part in name.split("."):
        mod = getattr(mod, part)

    if not isinstance(mod, type):
        raise TypeError(f"{fqp} is not a class")

    return mod


def filesystem(protocol, **storage_options):
    """Instantiate filesystems for given protocol and arguments

    ``storage_options`` are specific to the protocol being chosen, and are
    passed directly to the class.
    """
    if protocol == "arrow_hdfs":
        warnings.warn(
            "The 'arrow_hdfs' protocol has been deprecated and will be "
            "removed in the future. Specify it as 'hdfs'.",
            DeprecationWarning,
        )

    cls = get_filesystem_class(protocol)
    return cls(**storage_options)


def available_protocols():
    """Return a list of the implemented protocols.

    Note that any given protocol may require extra packages to be importable.
    """
    return list(known_implementations)
<|endoftext|>import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import PurePath
from typing import Any, ClassVar

from .registry import _import_class, get_filesystem_class
from .spec import AbstractFileSystem


class FilesystemJSONEncoder(json.JSONEncoder):
    include_password: ClassVar[bool] = True

    def default(self, o: Any) -> Any:
        if isinstance(o, AbstractFileSystem):
            return o.to_dict(include_password=self.include_password)
        if isinstance(o, PurePath):
            cls = type(o)
            return {"cls": f"{cls.__module__}.{cls.__name__}", "str": str(o)}

        return super().default(o)

    def make_serializable(self, obj: Any) -> Any:
        """
        Recursively converts an object so that it can be JSON serialized via
        :func:`json.dumps` and :func:`json.dump`, without actually calling
        said functions.
        """
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Mapping):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, Sequence):
            return [self.make_serializable(v) for v in obj]

        return self.default(obj)


class FilesystemJSONDecoder(json.JSONDecoder):
    def __init__(
        self,
        *,
        object_hook: Callable[[dict[str, Any]], Any] | None = None,
        parse_float: Callable[[str], Any] | None = None,
        parse_int: Callable[[str], Any] | None = None,
        parse_constant: Callable[[str], Any] | None = None,
        strict: bool = True,
        object_pairs_hook: Callable[[list[tuple[str, Any]]], Any] | None = None,
    ) -> None:
        self.original_object_hook = object_hook

        super().__init__(
            object_hook=self.custom_object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            strict=strict,
            object_pairs_hook=object_pairs_hook,
        )

    @classmethod
    def try_resolve_path_cls(cls, dct: dict[str, Any]):
        with suppress(Exception):
            fqp = dct["cls"]

            path_cls = _import_class(fqp)

            if issubclass(path_cls, PurePath):
                return path_cls

        return None

    @classmethod
    def try_resolve_fs_cls(cls, dct: dict[str, Any]):
        with suppress(Exception):
            if "cls" in dct:
                try:
                    fs_cls = _import_class(dct["cls"])
                    if issubclass(fs_cls, AbstractFileSystem):
                        return fs_cls
                except Exception:
                    if "protocol" in dct:  # Fallback if cls cannot be imported
                        return get_filesystem_class(dct["protocol"])

                    raise

        return None

    def custom_object_hook(self, dct: dict[str, Any]):
        if "cls" in dct:
            if (obj_cls := self.try_resolve_fs_cls(dct)) is not None:
                return AbstractFileSystem.from_dict(dct)
            if (obj_cls := self.try_resolve_path_cls(dct)) is not None:
                return obj_cls(dct["str"])

        if self.original_object_hook is not None:
            return self.original_object_hook(dct)

        return dct

    def unmake_serializable(self, obj: Any) -> Any:
        """
        Inverse function of :meth:`FilesystemJSONEncoder.make_serializable`.
        """
        if isinstance(obj, dict):
            obj = self.custom_object_hook(obj)
        if isinstance(obj, dict):
            return {k: self.unmake_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.unmake_serializable(v) for v in obj]

        return obj
<|endoftext|>import operator

from fsspec import AbstractFileSystem
from fsspec.utils import tokenize


class AbstractArchiveFileSystem(AbstractFileSystem):
    """
    A generic superclass for implementing Archive-based filesystems.

    Currently, it is shared amongst
    :class:`~fsspec.implementations.zip.ZipFileSystem`,
    :class:`~fsspec.implementations.libarchive.LibArchiveFileSystem` and
    :class:`~fsspec.implementations.tar.TarFileSystem`.
    """

    def __str__(self):
        return f"<Archive-like object {type(self).__name__} at {id(self)}>"

    __repr__ = __str__

    def ukey(self, path):
        return tokenize(path, self.fo, self.protocol)

    def _all_dirnames(self, paths):
        """Returns *all* directory names for each path in paths, including intermediate
        ones.

        Parameters
        ----------
        paths: Iterable of path strings
        """
        if len(paths) == 0:
            return set()

        dirnames = {self._parent(path) for path in paths} - {self.root_marker}
        return dirnames | self._all_dirnames(dirnames)

    def info(self, path, **kwargs):
        self._get_dirs()
        path = self._strip_protocol(path)
        if path in {"", "/"} and self.dir_cache:
            return {"name": "", "type": "directory", "size": 0}
        if path in self.dir_cache:
            return self.dir_cache[path]
        elif path + "/" in self.dir_cache:
            return self.dir_cache[path + "/"]
        else:
            raise FileNotFoundError(path)

    def ls(self, path, detail=True, **kwargs):
        self._get_dirs()
        paths = {}
        for p, f in self.dir_cache.items():
            p = p.rstrip("/")
            if "/" in p:
                root = p.rsplit("/", 1)[0]
            else:
                root = ""
            if root == path.rstrip("/"):
                paths[p] = f
            elif all(
                (a == b)
                for a, b in zip(path.split("/"), [""] + p.strip("/").split("/"))
            ):
                # root directory entry
                ppath = p.rstrip("/").split("/", 1)[0]
                if ppath not in paths:
                    out = {"name": ppath, "size": 0, "type": "directory"}
                    paths[ppath] = out
        if detail:
            out = sorted(paths.values(), key=operator.itemgetter("name"))
            return out
        else:
            return sorted(paths)
<|endoftext|>from __future__ import annotations

import io
import json
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import Any, ClassVar

from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
    _unstrip_protocol,
    glob_translate,
    isfilelike,
    other_paths,
    read_block,
    stringify_path,
    tokenize,
)

logger = logging.getLogger("fsspec")


def make_instance(cls, args, kwargs):
    return cls(*args, **kwargs)


FORK_AVAILABLE = hasattr(os, "register_at_fork")


if FORK_AVAILABLE:
    _registered_classes = weakref.WeakSet()

    def _reset_instances_lock():
        for cls in _registered_classes:
            cls._instantiation_lock = threading.RLock()
            cls._cache.clear()
            cls._pid = os.getpid()

    os.register_at_fork(after_in_child=_reset_instances_lock)


class _Cached(type):
    """
    Metaclass for caching file system instances.

    Notes
    -----
    Instances are cached according to

    * The values of the class attributes listed in `_extra_tokenize_attributes`
    * The arguments passed to ``__init__``.

    This creates an additional reference to the filesystem, which prevents the
    filesystem from being garbage collected when all *user* references go away.
    A call to the :meth:`AbstractFileSystem.clear_instance_cache` must *also*
    be made for a filesystem instance to be garbage collected.
    """

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Note: we intentionally create a reference here, to avoid garbage
        # collecting instances when all other references are gone. To really
        # delete a FileSystem, the cache must be cleared.
        if conf.get("weakref_instance_cache"):  # pragma: no cover
            # debug option for analysing fork/spawn conditions
            cls._cache = weakref.WeakValueDictionary()
        else:
            cls._cache = {}
        cls._pid = os.getpid()
        cls._instantiation_lock = threading.RLock()

        if FORK_AVAILABLE:
            _registered_classes.add(cls)

    def _check_instance_cache(cls, token):
        inst = cls._cache.get(token)
        if inst is not None:
            cls._latest = token
        return inst

    def __call__(cls, *args, **kwargs):
        kwargs = apply_config(cls, kwargs)
        extra_tokens = tuple(
            getattr(cls, attr, None) for attr in cls._extra_tokenize_attributes
        )
        strip_tokenize_options = {
            k: kwargs.pop(k) for k in cls._strip_tokenize_options if k in kwargs
        }
        pid = os.getpid()

        if getattr(cls, "async_impl", False) and not kwargs.get("asynchronous", False):
            token = tokenize(cls, pid, *args, *extra_tokens, **kwargs)
        else:
            token = tokenize(
                cls, pid, threading.get_ident(), *args, *extra_tokens, **kwargs
            )
        skip = kwargs.pop("skip_instance_cache", False)

        if pid != cls._pid:
            with cls._instantiation_lock:
                if pid != cls._pid:
                    cls._cache.clear()
                    cls._pid = pid

        if not skip and cls.cachable:
            inst = cls._check_instance_cache(token)
            if inst is not None:
                return inst

            with cls._instantiation_lock:
                # protect against the race condition that a new instance was created
                # and inserted into the cache since the initial check just above
                inst = cls._check_instance_cache(token)
                if inst is not None:
                    return inst

        obj = super().__call__(*args, **kwargs, **strip_tokenize_options)
        # Setting _fs_token here causes some static linters to complain.
        obj._fs_token_ = token
        obj.storage_args = args
        obj.storage_options = kwargs
        if obj.async_impl and obj.mirror_sync_methods:
            from .asyn import mirror_sync_methods

            mirror_sync_methods(obj)

        if cls.cachable and not skip:
            with cls._instantiation_lock:
                # another thread may have created the instance while we were calling
                # super().__call__(), so we check again.
                inst = cls._check_instance_cache(token)
                if inst is not None:
                    return inst

                cls._latest = token
                cls._cache[token] = obj
        return obj


class AbstractFileSystem(metaclass=_Cached):
    """
    An abstract super-class for pythonic file-systems

    Implementations are expected to be compatible with or, better, subclass
    from here.
    """

    cachable = True  # this class can be cached, instances reused
    _cached = False
    blocksize = 2**22
    sep = "/"
    protocol: ClassVar[str | tuple[str, ...]] = "abstract"
    _latest = None
    async_impl = False
    mirror_sync_methods = False
    root_marker = ""  # For some FSs, may require leading '/' or other character
    transaction_type = Transaction

    #: Extra *class attributes* that should be considered when hashing.
    _extra_tokenize_attributes = ()
    #: *storage options* that should not be considered when hashing.
    _strip_tokenize_options = ()

    # Set by _Cached metaclass
    storage_args: tuple[Any, ...]
    storage_options: dict[str, Any]

    def __init__(self, *args, **storage_options):
        """Create and configure file-system instance

        Instances may be cachable, so if similar enough arguments are seen
        a new instance is not required. The token attribute exists to allow
        implementations to cache instances if they wish.

        A reasonable default should be provided if there are no arguments.

        Subclasses should call this method.

        Parameters
        ----------
        use_listings_cache, listings_expiry_time, max_paths:
            passed to ``DirCache``, if the implementation supports
            directory listing caching. Pass use_listings_cache=False
            to disable such caching.
        skip_instance_cache: bool
            If this is a cachable implementation, pass True here to force
            creating a new instance even if a matching instance exists, and prevent
            storing this instance.
        asynchronous: bool
        loop: asyncio-compatible IOLoop or None
        """
        if self._cached:
            # reusing instance, don't change
            return
        self._cached = True
        self._intrans = False
        self._transaction = None
        self._invalidated_caches_in_transaction = []
        self.dircache = DirCache(**storage_options)

        if storage_options.pop("add_docs", None):
            warnings.warn("add_docs is no longer supported.", FutureWarning)

        if storage_options.pop("add_aliases", None):
            warnings.warn("add_aliases has been removed.", FutureWarning)
        # This is set in _Cached
        self._fs_token_ = None

    @property
    def fsid(self):
        """Persistent filesystem id that can be used to compare filesystems
        across sessions.
        """
        raise NotImplementedError

    @property
    def _fs_token(self):
        return self._fs_token_

    def __dask_tokenize__(self):
        return self._fs_token

    def __hash__(self):
        return int(self._fs_token, 16)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self._fs_token == other._fs_token

    def __reduce__(self):
        return make_instance, (type(self), self.storage_args, self.storage_options)

    @classmethod
    def _strip_protocol(cls, path):
        """Turn path from fully-qualified to file-system-specific

        May require FS-specific handling, e.g., for relative paths or links.
        """
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        path = stringify_path(path)
        protos = (cls.protocol,) if isinstance(cls.protocol, str) else cls.protocol
        for protocol in protos:
            if path.startswith(protocol + "://"):
                path = path[len(protocol) + 3 :]
            elif path.startswith(protocol + "::"):
                path = path[len(protocol) + 2 :]
        path = path.rstrip("/")
        # use of root_marker to make minimum required path, e.g., "/"
        return path or cls.root_marker

    def unstrip_protocol(self, name: str) -> str:
        """Format FS-specific path to generic, including protocol"""
        protos = (self.protocol,) if isinstance(self.protocol, str) else self.protocol
        for protocol in protos:
            if name.startswith(f"{protocol}://"):
                return name
        return f"{protos[0]}://{name}"

    @staticmethod
    def _get_kwargs_from_urls(path):
        """If kwargs can be encoded in the paths, extract them here

        This should happen before instantiation of the class; incoming paths
        then should be amended to strip the options in methods.

        Examples may look like an sftp path "sftp://user@host:/my/path", where
        the user and host should become kwargs and later get stripped.
        """
        # by default, nothing happens
        return {}

    @classmethod
    def current(cls):
        """Return the most recently instantiated FileSystem

        If no instance has been created, then create one with defaults
        """
        inst = cls._cache.get(cls._latest)
        if inst is not None:
            return inst
        return cls()

    @property
    def transaction(self):
        """A context within which files are committed together upon exit

        Requires the file class to implement `.commit()` and `.discard()`
        for the normal and exception cases.
        """
        if self._transaction is None:
            self._transaction = self.transaction_type(self)
        return self._transaction

    def start_transaction(self):
        """Begin write transaction for deferring files, non-context version"""
        self._intrans = True
        self._transaction = self.transaction_type(self)
        return self.transaction

    def end_transaction(self):
        """Finish write transaction, non-context version"""
        self.transaction.complete()
        self._transaction = None
        # The invalid cache must be cleared after the transaction is completed.
        for path in self._invalidated_caches_in_transaction:
            self.invalidate_cache(path)
        self._invalidated_caches_in_transaction.clear()

    def invalidate_cache(self, path=None):
        """
        Discard any cached directory information

        Parameters
        ----------
        path: string or None
            If None, clear all listings cached else listings at or under given
            path.
        """
        # Not necessary to implement invalidation mechanism, may have no cache.
        # But if have, you should call this method of parent class from your
        # subclass to ensure expiring caches after transacations correctly.
        # See the implementation of FTPFileSystem in ftp.py
        if self._intrans:
            self._invalidated_caches_in_transaction.append(path)

    def mkdir(self, path, create_parents=True, **kwargs):
        """
        Create directory entry at path

        For systems that don't have true directories, may create an for
        this instance only and not touch the real filesystem

        Parameters
        ----------
        path: str
            location
        create_parents: bool
            if True, this is equivalent to ``makedirs``
        kwargs:
            may be permissions, etc.
        """
        pass  # not necessary to implement, may not have directories

    def makedirs(self, path, exist_ok=False):
        """Recursively make directories

        Creates directory at path and any intervening required directories.
        Raises exception if, for instance, the path already exists but is a
        file.

        Parameters
        ----------
        path: str
            leaf directory name
        exist_ok: bool (False)
            If False, will error if the target already exists
        """
        pass  # not necessary to implement, may not have directories

    def rmdir(self, path):
        """Remove a directory, if empty"""
        pass  # not necessary to implement, may not have directories

    def ls(self, path, detail=True, **kwargs):
        """List objects at path.

        This should include subdirectories and files at that location. The
        difference between a file and a directory must be clear when details
        are requested.

        The specific keys, or perhaps a FileInfo class, or similar, is TBD,
        but must be consistent across implementations.
        Must include:

        - full path to the entry (without protocol)
        - size of the entry, in bytes. If the value cannot be determined, will
          be ``None``.
        - type of entry, "file", "directory" or other

        Additional information
        may be present, appropriate to the file-system, e.g., generation,
        checksum, etc.

        May use refresh=True|False to allow use of self._ls_from_cache to
        check for a saved listing and avoid calling the backend. This would be
        common where listing may be expensive.

        Parameters
        ----------
        path: str
        detail: bool
            if True, gives a list of dictionaries, where each is the same as
            the result of ``info(path)``. If False, gives a list of paths
            (str).
        kwargs: may have additional backend-specific options, such as version
            information

        Returns
        -------
        List of strings if detail is False, or list of directory information
        dicts if detail is True.
        """
        raise NotImplementedError

    def _ls_from_cache(self, path):
        """Check cache for listing

        Returns listing, if found (may be empty list for a directly that exists
        but contains nothing), None if not in cache.
        """
        parent = self._parent(path)
        try:
            return self.dircache[path.rstrip("/")]
        except KeyError:
            pass
        try:
            files = [
                f
                for f in self.dircache[parent]
                if f["name"] == path
                or (f["name"] == path.rstrip("/") and f["type"] == "directory")
            ]
            if len(files) == 0:
                # parent dir was listed but did not contain this file
                raise FileNotFoundError(path)
            return files
        except KeyError:
            pass

    def walk(self, path, maxdepth=None, topdown=True, on_error="omit", **kwargs):
        """Return all files under the given path.

        List all files, recursing into subdirectories; output is iterator-style,
        like ``os.walk()``. For a simple list of files, ``find()`` is available.

        When topdown is True, the caller can modify the dirnames list in-place (perhaps
        using del or slice assignment), and walk() will
        only recurse into the subdirectories whose names remain in dirnames;
        this can be used to prune the search, impose a specific order of visiting,
        or even to inform walk() about directories the caller creates or renames before
        it resumes walk() again.
        Modifying dirnames when topdown is False has no effect. (see os.walk)

        Note that the "files" outputted will include anything that is not
        a directory, such as links.

        Parameters
        ----------
        path: str
            Root to recurse into
        maxdepth: int
            Maximum recursion depth. None means limitless, but not recommended
            on link-based file-systems.
        topdown: bool (True)
            Whether to walk the directory tree from the top downwards or from
            the bottom upwards.
        on_error: "omit", "raise", a callable
            if omit (default), path with exception will simply be empty;
            If raise, an underlying exception will be raised;
            if callable, it will be called with a single OSError instance as argument
        kwargs: passed to ``ls``
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        path = self._strip_protocol(path)
        full_dirs = {}
        dirs = {}
        files = {}

        detail = kwargs.pop("detail", False)
        try:
            listing = self.ls(path, detail=True, **kwargs)
        except (FileNotFoundError, OSError) as e:
            if on_error == "raise":
                raise
            if callable(on_error):
                on_error(e)
            return

        for info in listing:
            # each info name must be at least [path]/part , but here
            # we check also for names like [path]/part/
            pathname = info["name"].rstrip("/")
            name = pathname.rsplit("/", 1)[-1]
            if info["type"] == "directory" and pathname != path:
                # do not include "self" path
                full_dirs[name] = pathname
                dirs[name] = info
            elif pathname == path:
                # file-like with same name as give path
                files[""] = info
            else:
                files[name] = info

        if not detail:
            dirs = list(dirs)
            files = list(files)

        if topdown:
            # Yield before recursion if walking top down
            yield path, dirs, files

        if maxdepth is not None:
            maxdepth -= 1
            if maxdepth < 1:
                if not topdown:
                    yield path, dirs, files
                return

        for d in dirs:
            yield from self.walk(
                full_dirs[d],
                maxdepth=maxdepth,
                detail=detail,
                topdown=topdown,
                **kwargs,
            )

        if not topdown:
            # Yield after recursion if walking bottom up
            yield path, dirs, files

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        """List all files below path.

        Like posix ``find`` command without conditions

        Parameters
        ----------
        path : str
        maxdepth: int or None
            If not None, the maximum number of levels to descend
        withdirs: bool
            Whether to include directory paths in the output. This is True
            when used by glob, but users usually only want files.
        kwargs are passed to ``ls``.
        """
        # TODO: allow equivalent of -name parameter
        path = self._strip_protocol(path)
        out = {}

        # Add the root directory if withdirs is requested
        # This is needed for posix glob compliance
        if withdirs and path != "" and self.isdir(path):
            out[path] = self.info(path)

        for _, dirs, files in self.walk(path, maxdepth, detail=True, **kwargs):
            if withdirs:
                files.update(dirs)
            out.update({info["name"]: info for name, info in files.items()})
        if not out and self.isfile(path):
            # walk works on directories, but find should also return [path]
            # when path happens to be a file
            out[path] = {}
        names = sorted(out)
        if not detail:
            return names
        else:
            return {name: out[name] for name in names}

    def du(self, path, total=True, maxdepth=None, withdirs=False, **kwargs):
        """Space used by files and optionally directories within a path

        Directory size does not include the size of its contents.

        Parameters
        ----------
        path: str
        total: bool
            Whether to sum all the file sizes
        maxdepth: int or None
            Maximum number of directory levels to descend, None for unlimited.
        withdirs: bool
            Whether to include directory paths in the output.
        kwargs: passed to ``find``

        Returns
        -------
        Dict of {path: size} if total=False, or int otherwise, where numbers
        refer to bytes used.
        """
        sizes = {}
        if withdirs and self.isdir(path):
            # Include top-level directory in output
            info = self.info(path)
            sizes[info["name"]] = info["size"]
        for f in self.find(path, maxdepth=maxdepth, withdirs=withdirs, **kwargs):
            info = self.info(f)
            sizes[info["name"]] = info["size"]
        if total:
            return sum(sizes.values())
        else:
            return sizes

    def glob(self, path, maxdepth=None, **kwargs):
        """Find files by glob-matching.

        Pattern matching capabilities for finding files that match the given pattern.

        Parameters
        ----------
        path: str
            The glob pattern to match against
        maxdepth: int or None
            Maximum depth for ``'**'`` patterns. Applied on the first ``'**'`` found.
            Must be at least 1 if provided.
        kwargs:
            Additional arguments passed to ``find`` (e.g., detail=True)

        Returns
        -------
        List of matched paths, or dict of paths and their info if detail=True

        Notes
        -----
        Supported patterns:
        - '*': Matches any sequence of characters within a single directory level
        - ``'**'``: Matches any number of directory levels (must be an entire path component)
        - '?': Matches exactly one character
        - '[abc]': Matches any character in the set
        - '[a-z]': Matches any character in the range
        - '[!abc]': Matches any character NOT in the set

        Special behaviors:
        - If the path ends with '/', only folders are returned
        - Consecutive '*' characters are compressed into a single '*'
        - Empty set '[]' or negated empty negated set '[!]' never match anything
        - Special characters in character classes are escaped properly

        Limitations:
        - ``'**'`` must be a complete path component (e.g., ``'a/**/b'``, not ``'a**b'``)
        - No brace expansion ('{a,b}.txt')
        - No extended glob patterns ('+(pattern)', '!(pattern)')
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        import re

        seps = (os.path.sep, os.path.altsep) if os.path.altsep else (os.path.sep,)
        ends_with_sep = path.endswith(seps)  # _strip_protocol strips trailing slash
        path = self._strip_protocol(path)
        append_slash_to_dirname = ends_with_sep or path.endswith(
            tuple(sep + "**" for sep in seps)
        )
        idx_star = path.find("*") if path.find("*") >= 0 else len(path)
        idx_qmark = path.find("?") if path.find("?") >= 0 else len(path)
        idx_brace = path.find("[") if path.find("[") >= 0 else len(path)

        min_idx = min(idx_star, idx_qmark, idx_brace)

        detail = kwargs.pop("detail", False)
        withdirs = kwargs.pop("withdirs", True)

        if not has_magic(path):
            if self.exists(path, **kwargs):
                if not detail:
                    return [path]
                else:
                    return {path: self.info(path, **kwargs)}
            else:
                if not detail:
                    return []  # glob of non-existent returns empty
                else:
                    return {}
        elif "/" in path[:min_idx]:
            min_idx = path[:min_idx].rindex("/")
            root = path[: min_idx + 1]
            depth = path[min_idx + 1 :].count("/") + 1
        else:
            root = ""
            depth = path[min_idx + 1 :].count("/") + 1

        if "**" in path:
            if maxdepth is not None:
                idx_double_stars = path.find("**")
                depth_double_stars = path[idx_double_stars:].count("/") + 1
                depth = depth - depth_double_stars + maxdepth
            else:
                depth = None

        allpaths = self.find(
            root, maxdepth=depth, withdirs=withdirs, detail=True, **kwargs
        )

        pattern = glob_translate(path + ("/" if ends_with_sep else ""))
        pattern = re.compile(pattern)

        out = {
            p: info
            for p, info in sorted(allpaths.items())
            if pattern.match(
                p + "/"
                if append_slash_to_dirname and info["type"] == "directory"
                else p
            )
        }

        if detail:
            return out
        else:
            return list(out)

    def exists(self, path, **kwargs):
        """Is there a file at the given path"""
        try:
            self.info(path, **kwargs)
            return True
        except:  # noqa: E722
            # any exception allowed bar FileNotFoundError?
            return False

    def lexists(self, path, **kwargs):
        """If there is a file at the given path (including
        broken links)"""
        return self.exists(path)

    def info(self, path, **kwargs):
        """Give details of entry at path

        Returns a single dictionary, with exactly the same information as ``ls``
        would with ``detail=True``.

        The default implementation calls ls and could be overridden by a
        shortcut. kwargs are passed on to ```ls()``.

        Some file systems might not be able to measure the file's size, in
        which case, the returned dict will include ``'size': None``.

        Returns
        -------
        dict with keys: name (full path in the FS), size (in bytes), type (file,
        directory, or something else) and other FS-specific keys.
        """
        path = self._strip_protocol(path)
        out = self.ls(self._parent(path), detail=True, **kwargs)
        out = [o for o in out if o["name"].rstrip("/") == path]
        if out:
            return out[0]
        out = self.ls(path, detail=True, **kwargs)
        path = path.rstrip("/")
        out1 = [o for o in out if o["name"].rstrip("/") == path]
        if len(out1) == 1:
            if "size" not in out1[0]:
                out1[0]["size"] = None
            return out1[0]
        elif len(out1) > 1 or out:
            return {"name": path, "size": 0, "type": "directory"}
        else:
            raise FileNotFoundError(path)

    def checksum(self, path):
        """Unique value for current version of file

        If the checksum is the same from one moment to another, the contents
        are guaranteed to be the same. If the checksum changes, the contents
        *might* have changed.

        This should normally be overridden; default will probably capture
        creation/modification timestamp (which would be good) or maybe
        access timestamp (which would be bad)
        """
        return int(tokenize(self.info(path)), 16)

    def size(self, path):
        """Size in bytes of file"""
        return self.info(path).get("size", None)

    def sizes(self, paths):
        """Size in bytes of each file in a list of paths"""
        return [self.size(p) for p in paths]

    def isdir(self, path):
        """Is this entry directory-like?"""
        try:
            return self.info(path)["type"] == "directory"
        except OSError:
            return False

    def isfile(self, path):
        """Is this entry file-like?"""
        try:
            return self.info(path)["type"] == "file"
        except:  # noqa: E722
            return False

    def read_text(self, path, encoding=None, errors=None, newline=None, **kwargs):
        """Get the contents of the file as a string.

        Parameters
        ----------
        path: str
            URL of file on this filesystems
        encoding, errors, newline: same as `open`.
        """
        with self.open(
            path,
            mode="r",
            encoding=encoding,
            errors=errors,
            newline=newline,
            **kwargs,
        ) as f:
            return f.read()

    def write_text(
        self, path, value, encoding=None, errors=None, newline=None, **kwargs
    ):
        """Write the text to the given file.

        An existing file will be overwritten.

        Parameters
        ----------
        path: str
            URL of file on this filesystems
        value: str
            Text to write.
        encoding, errors, newline: same as `open`.
        """
        with self.open(
            path,
            mode="w",
            encoding=encoding,
            errors=errors,
            newline=newline,
            **kwargs,
        ) as f:
            return f.write(value)

    def cat_file(self, path, start=None, end=None, **kwargs):
        """Get the content of a file

        Parameters
        ----------
        path: URL of file on this filesystems
        start, end: int
            Bytes limits of the read. If negative, backwards from end,
            like usual python slices. Either can be None for start or
            end of file, respectively
        kwargs: passed to ``open()``.
        """
        # explicitly set buffering off?
        with self.open(path, "rb", **kwargs) as f:
            if start is not None:
                if start >= 0:
                    f.seek(start)
                else:
                    f.seek(max(0, f.size + start))
            if end is not None:
                if end < 0:
                    end = f.size + end
                return f.read(end - f.tell())
            return f.read()

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        """Set the bytes of given file"""
        if mode == "create" and self.exists(path):
            # non-atomic but simple way; or could use "xb" in open(), which is likely
            # not as well supported
            raise FileExistsError
        with self.open(path, "wb", **kwargs) as f:
            f.write(value)

    def pipe(self, path, value=None, **kwargs):
        """Put value into path

        (counterpart to ``cat``)

        Parameters
        ----------
        path: string or dict(str, bytes)
            If a string, a single remote location to put ``value`` bytes; if a dict,
            a mapping of {path: bytesvalue}.
        value: bytes, optional
            If using a single path, these are the bytes to put there. Ignored if
            ``path`` is a dict
        """
        if isinstance(path, str):
            self.pipe_file(self._strip_protocol(path), value, **kwargs)
        elif isinstance(path, dict):
            for k, v in path.items():
                self.pipe_file(self._strip_protocol(k), v, **kwargs)
        else:
            raise ValueError("path must be str or dict")

    def cat_ranges(
        self, paths, starts, ends, max_gap=None, on_error="return", **kwargs
    ):
        """Get the contents of byte ranges from one or more files

        Parameters
        ----------
        paths: list
            A list of of filepaths on this filesystems
        starts, ends: int or list
            Bytes limits of the read. If using a single int, the same value will be
            used to read all the specified files.
        """
        if max_gap is not None:
            raise NotImplementedError
        if not isinstance(paths, list):
            raise TypeError
        if not isinstance(starts, list):
            starts = [starts] * len(paths)
        if not isinstance(ends, list):
            ends = [ends] * len(paths)
        if len(starts) != len(paths) or len(ends) != len(paths):
            raise ValueError
        out = []
        for p, s, e in zip(paths, starts, ends):
            try:
                out.append(self.cat_file(p, s, e, **kwargs))
            except Exception as e:
                if on_error == "return":
                    out.append(e)
                else:
                    raise
        return out

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        """Fetch (potentially multiple) paths' contents

        Parameters
        ----------
        recursive: bool
            If True, assume the path(s) are directories, and get all the
            contained files
        on_error : "raise", "omit", "return"
            If raise, an underlying exception will be raised (converted to KeyError
            if the type is in self.missing_exceptions); if omit, keys with exception
            will simply not be included in the output; if "return", all keys are
            included in the output, but the value will be bytes or an exception
            instance.
        kwargs: passed to cat_file

        Returns
        -------
        dict of {path: contents} if there are multiple paths
        or the path has been otherwise expanded
        """
        paths = self.expand_path(path, recursive=recursive, **kwargs)
        if (
            len(paths) > 1
            or isinstance(path, list)
            or paths[0] != self._strip_protocol(path)
        ):
            out = {}
            for path in paths:
                try:
                    out[path] = self.cat_file(path, **kwargs)
                except Exception as e:
                    if on_error == "raise":
                        raise
                    if on_error == "return":
                        out[path] = e
            return out
        else:
            return self.cat_file(paths[0], **kwargs)

    def get_file(self, rpath, lpath, callback=DEFAULT_CALLBACK, outfile=None, **kwargs):
        """Copy single remote file to local"""
        from .implementations.local import LocalFileSystem

        if isfilelike(lpath):
            outfile = lpath
        elif self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
            return None

        fs = LocalFileSystem(auto_mkdir=True)
        fs.makedirs(fs._parent(lpath), exist_ok=True)

        with self.open(rpath, "rb", **kwargs) as f1:
            if outfile is None:
                outfile = open(lpath, "wb")

            try:
                callback.set_size(getattr(f1, "size", None))
                data = True
                while data:
                    data = f1.read(self.blocksize)
                    segment_len = outfile.write(data)
                    if segment_len is None:
                        segment_len = len(data)
                    callback.relative_update(segment_len)
            finally:
                if not isfilelike(lpath):
                    outfile.close()

    def get(
        self,
        rpath,
        lpath,
        recursive=False,
        callback=DEFAULT_CALLBACK,
        maxdepth=None,
        **kwargs,
    ):
        """Copy file(s) to local.

        Copies a specific file or tree of files (if recursive=True). If lpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within. Can submit a list of paths, which may be glob-patterns
        and will be expanded.

        Calls get_file for each source.
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            rpaths = rpath
            lpaths = lpath
        else:
            from .implementations.local import (
                LocalFileSystem,
                make_path_posix,
                trailing_sep,
            )

            source_is_str = isinstance(rpath, str)
            rpaths = self.expand_path(
                rpath, recursive=recursive, maxdepth=maxdepth, **kwargs
            )
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                rpaths = [p for p in rpaths if not (trailing_sep(p) or self.isdir(p))]
                if not rpaths:
                    return

            if isinstance(lpath, str):
                lpath = make_path_posix(lpath)

            source_is_file = len(rpaths) == 1
            dest_is_dir = isinstance(lpath, str) and (
                trailing_sep(lpath) or LocalFileSystem().isdir(lpath)
            )

            exists = source_is_str and (
                (has_magic(rpath) and source_is_file)
                or (not has_magic(rpath) and dest_is_dir and not trailing_sep(rpath))
            )
            lpaths = other_paths(
                rpaths,
                lpath,
                exists=exists,
                flatten=not source_is_str,
            )

        callback.set_size(len(lpaths))
        for lpath, rpath in callback.wrap(zip(lpaths, rpaths)):
            with callback.branched(rpath, lpath) as child:
                self.get_file(rpath, lpath, callback=child, **kwargs)

    def put_file(
        self, lpath, rpath, callback=DEFAULT_CALLBACK, mode="overwrite", **kwargs
    ):
        """Copy single file to remote"""
        if mode == "create" and self.exists(rpath):
            raise FileExistsError
        if os.path.isdir(lpath):
            self.makedirs(rpath, exist_ok=True)
            return None

        with open(lpath, "rb") as f1:
            size = f1.seek(0, 2)
            callback.set_size(size)
            f1.seek(0)

            self.mkdirs(self._parent(os.fspath(rpath)), exist_ok=True)
            with self.open(rpath, "wb", **kwargs) as f2:
                while f1.tell() < size:
                    data = f1.read(self.blocksize)
                    segment_len = f2.write(data)
                    if segment_len is None:
                        segment_len = len(data)
                    callback.relative_update(segment_len)

    def put(
        self,
        lpath,
        rpath,
        recursive=False,
        callback=DEFAULT_CALLBACK,
        maxdepth=None,
        **kwargs,
    ):
        """Copy file(s) from local.

        Copies a specific file or tree of files (if recursive=True). If rpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within.

        Calls put_file for each source.
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            rpaths = rpath
            lpaths = lpath
        else:
            from .implementations.local import (
                LocalFileSystem,
                make_path_posix,
                trailing_sep,
            )

            source_is_str = isinstance(lpath, str)
            if source_is_str:
                lpath = make_path_posix(lpath)
            fs = LocalFileSystem()
            lpaths = fs.expand_path(
                lpath, recursive=recursive, maxdepth=maxdepth, **kwargs
            )
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                lpaths = [p for p in lpaths if not (trailing_sep(p) or fs.isdir(p))]
                if not lpaths:
                    return

            source_is_file = len(lpaths) == 1
            dest_is_dir = isinstance(rpath, str) and (
                trailing_sep(rpath) or self.isdir(rpath)
            )

            rpath = (
                self._strip_protocol(rpath)
                if isinstance(rpath, str)
                else [self._strip_protocol(p) for p in rpath]
            )
            exists = source_is_str and (
                (has_magic(lpath) and source_is_file)
                or (not has_magic(lpath) and dest_is_dir and not trailing_sep(lpath))
            )
            rpaths = other_paths(
                lpaths,
                rpath,
                exists=exists,
                flatten=not source_is_str,
            )

        callback.set_size(len(rpaths))
        for lpath, rpath in callback.wrap(zip(lpaths, rpaths)):
            with callback.branched(lpath, rpath) as child:
                self.put_file(lpath, rpath, callback=child, **kwargs)

    def head(self, path, size=1024):
        """Get the first ``size`` bytes from file"""
        with self.open(path, "rb") as f:
            return f.read(size)

    def tail(self, path, size=1024):
        """Get the last ``size`` bytes from file"""
        with self.open(path, "rb") as f:
            f.seek(max(-size, -f.size), 2)
            return f.read()

    def cp_file(self, path1, path2, **kwargs):
        raise NotImplementedError

    def copy(
        self, path1, path2, recursive=False, maxdepth=None, on_error=None, **kwargs
    ):
        """Copy within two locations in the filesystem

        on_error : "raise", "ignore"
            If raise, any not-found exceptions will be raised; if ignore any
            not-found exceptions will cause the path to be skipped; defaults to
            raise unless recursive is true, where the default is ignore
        """
        if on_error is None and recursive:
            on_error = "ignore"
        elif on_error is None:
            on_error = "raise"

        if isinstance(path1, list) and isinstance(path2, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            paths1 = path1
            paths2 = path2
        else:
            from .implementations.local import trailing_sep

            source_is_str = isinstance(path1, str)
            paths1 = self.expand_path(
                path1, recursive=recursive, maxdepth=maxdepth, **kwargs
            )
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                paths1 = [p for p in paths1 if not (trailing_sep(p) or self.isdir(p))]
                if not paths1:
                    return

            source_is_file = len(paths1) == 1
            dest_is_dir = isinstance(path2, str) and (
                trailing_sep(path2) or self.isdir(path2)
            )

            exists = source_is_str and (
                (has_magic(path1) and source_is_file)
                or (not has_magic(path1) and dest_is_dir and not trailing_sep(path1))
            )
            paths2 = other_paths(
                paths1,
                path2,
                exists=exists,
                flatten=not source_is_str,
            )

        for p1, p2 in zip(paths1, paths2):
            try:
                self.cp_file(p1, p2, **kwargs)
            except FileNotFoundError:
                if on_error == "raise":
                    raise

    def expand_path(
        self, path, recursive=False, maxdepth=None, assume_literal=False, **kwargs
    ):
        """Turn one or more globs or directories into a list of all matching paths
        to files or directories.

        kwargs are passed to ``glob`` or ``find``, which may in turn call ``ls``
        """

        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        if isinstance(path, (str, os.PathLike)):
            out = self.expand_path([path], recursive, maxdepth, **kwargs)
        else:
            out = set()
            path = [self._strip_protocol(p) for p in path]
            for p in path:
                if not assume_literal and has_magic(p):
                    bit = set(self.glob(p, maxdepth=maxdepth, **kwargs))
                    out |= bit
                    if recursive:
                        # glob call above expanded one depth so if maxdepth is defined
                        # then decrement it in expand_path call below. If it is zero
                        # after decrementing then avoid expand_path call.
                        if maxdepth is not None and maxdepth <= 1:
                            continue
                        out |= set(
                            self.expand_path(
                                list(bit),
                                recursive=recursive,
                                maxdepth=maxdepth - 1 if maxdepth is not None else None,
                                assume_literal=True,
                                **kwargs,
                            )
                        )
                    continue
                elif recursive:
                    rec = set(
                        self.find(
                            p, maxdepth=maxdepth, withdirs=True, detail=False, **kwargs
                        )
                    )
                    out |= rec
                if p not in out and (recursive is False or self.exists(p)):
                    # should only check once, for the root
                    out.add(p)
        if not out:
            raise FileNotFoundError(path)
        return sorted(out)

    def mv(self, path1, path2, recursive=False, maxdepth=None, **kwargs):
        """Move file(s) from one location to another"""
        if path1 == path2:
            logger.debug("%s mv: The paths are the same, so no files were moved.", self)
        else:
            # explicitly raise exception to prevent data corruption
            self.copy(
                path1, path2, recursive=recursive, maxdepth=maxdepth, on_error="raise"
            )
            self.rm(path1, recursive=recursive)

    def rm_file(self, path):
        """Delete a file"""
        self._rm(path)

    def _rm(self, path):
        """Delete one file"""
        # this is the old name for the method, prefer rm_file
        raise NotImplementedError

    def rm(self, path, recursive=False, maxdepth=None):
        """Delete files.

        Parameters
        ----------
        path: str or list of str
            File(s) to delete.
        recursive: bool
            If file(s) are directories, recursively delete contents and then
            also remove the directory
        maxdepth: int or None
            Depth to pass to walk for finding files to delete, if recursive.
            If None, there will be no limit and infinite recursion may be
            possible.
        """
        path = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        for p in reversed(path):
            self.rm_file(p)

    @classmethod
    def _parent(cls, path):
        path = cls._strip_protocol(path)
        if "/" in path:
            parent = path.rsplit("/", 1)[0].lstrip(cls.root_marker)
            return cls.root_marker + parent
        else:
            return cls.root_marker

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        """Return raw bytes-mode file-like from the file-system"""
        return AbstractBufferedFile(
            self,
            path,
            mode,
            block_size,
            autocommit,
            cache_options=cache_options,
            **kwargs,
        )

    def open(
        self,
        path,
        mode="rb",
        block_size=None,
        cache_options=None,
        compression=None,
        **kwargs,
    ):
        """
        Return a file-like object from the filesystem

        The resultant instance must function correctly in a context ``with``
        block.

        Parameters
        ----------
        path: str
            Target file
        mode: str like 'rb', 'w'
            See builtin ``open()``
            Mode "x" (exclusive write) may be implemented by the backend. Even if
            it is, whether  it is checked up front or on commit, and whether it is
            atomic is implementation-dependent.
        block_size: int
            Some indication of buffering - this is a value in bytes
        cache_options : dict, optional
            Extra arguments to pass through to the cache.
        compression: string or None
            If given, open file using compression codec. Can either be a compression
            name (a key in ``fsspec.compression.compr``) or "infer" to guess the
            compression from the filename suffix.
        encoding, errors, newline: passed on to TextIOWrapper for text mode
        """
        import io

        path = self._strip_protocol(path)
        if "b" not in mode:
            mode = mode.replace("t", "") + "b"

            text_kwargs = {
                k: kwargs.pop(k)
                for k in ["encoding", "errors", "newline"]
                if k in kwargs
            }
            return io.TextIOWrapper(
                self.open(
                    path,
                    mode,
                    block_size=block_size,
                    cache_options=cache_options,
                    compression=compression,
                    **kwargs,
                ),
                **text_kwargs,
            )
        else:
            ac = kwargs.pop("autocommit", not self._intrans)
            f = self._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=ac,
                cache_options=cache_options,
                **kwargs,
            )
            if compression is not None:
                from fsspec.compression import compr
                from fsspec.core import get_compression

                compression = get_compression(path, compression)
                compress = compr[compression]
                f = compress(f, mode=mode[0])

            if not ac and "r" not in mode:
                self.transaction.files.append(f)
            return f

    def touch(self, path, truncate=True, **kwargs):
        """Create empty file, or update timestamp

        Parameters
        ----------
        path: str
            file location
        truncate: bool
            If True, always set file size to 0; if False, update timestamp and
            leave file unchanged, if backend allows this
        """
        if truncate or not self.exists(path):
            with self.open(path, "wb", **kwargs):
                pass
        else:
            raise NotImplementedError  # update timestamp, if possible

    def ukey(self, path):
        """Hash of file properties, to tell if it has changed"""
        return sha256(str(self.info(path)).encode()).hexdigest()

    def read_block(self, fn, offset, length, delimiter=None):
        """Read a block of bytes from

        Starting at ``offset`` of the file, read ``length`` bytes.  If
        ``delimiter`` is set then we ensure that the read starts and stops at
        delimiter boundaries that follow the locations ``offset`` and ``offset
        + length``.  If ``offset`` is zero then we start at zero.  The
        bytestring returned WILL include the end delimiter string.

        If offset+length is beyond the eof, reads to eof.

        Parameters
        ----------
        fn: string
            Path to filename
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read. If None, read to end.
        delimiter: bytes (optional)
            Ensure reading starts and stops at delimiter bytestring

        Examples
        --------
        >>> fs.read_block('data/file.csv', 0, 13)  # doctest: +SKIP
        b'Alice, 100\\nBo'
        >>> fs.read_block('data/file.csv', 0, 13, delimiter=b'\\n')  # doctest: +SKIP
        b'Alice, 100\\nBob, 200\\n'

        Use ``length=None`` to read to the end of the file.
        >>> fs.read_block('data/file.csv', 0, None, delimiter=b'\\n')  # doctest: +SKIP
        b'Alice, 100\\nBob, 200\\nCharlie, 300'

        See Also
        --------
        :func:`fsspec.utils.read_block`
        """
        with self.open(fn, "rb") as f:
            size = f.size
            if length is None:
                length = size
            if size is not None and offset + length > size:
                length = size - offset
            return read_block(f, offset, length, delimiter)

    def to_json(self, *, include_password: bool = True) -> str:
        """
        JSON representation of this filesystem instance.

        Parameters
        ----------
        include_password: bool, default True
            Whether to include the password (if any) in the output.

        Returns
        -------
        JSON string with keys ``cls`` (the python location of this class),
        protocol (text name of this class's protocol, first one in case of
        multiple), ``args`` (positional args, usually empty), and all other
        keyword arguments as their own keys.

        Warnings
        --------
        Serialized filesystems may contain sensitive information which have been
        passed to the constructor, such as passwords and tokens. Make sure you
        store and send them in a secure environment!
        """
        from .json import FilesystemJSONEncoder

        return json.dumps(
            self,
            cls=type(
                "_FilesystemJSONEncoder",
                (FilesystemJSONEncoder,),
                {"include_password": include_password},
            ),
        )

    @staticmethod
    def from_json(blob: str) -> AbstractFileSystem:
        """
        Recreate a filesystem instance from JSON representation.

        See ``.to_json()`` for the expected structure of the input.

        Parameters
        ----------
        blob: str

        Returns
        -------
        file system instance, not necessarily of this particular class.

        Warnings
        --------
        This can import arbitrary modules (as determined by the ``cls`` key).
        Make sure you haven't installed any modules that may execute malicious code
        at import time.
        """
        from .json import FilesystemJSONDecoder

        return json.loads(blob, cls=FilesystemJSONDecoder)

    def to_dict(self, *, include_password: bool = True) -> dict[str, Any]:
        """
        JSON-serializable dictionary representation of this filesystem instance.

        Parameters
        ----------
        include_password: bool, default True
            Whether to include the password (if any) in the output.

        Returns
        -------
        Dictionary with keys ``cls`` (the python location of this class),
        protocol (text name of this class's protocol, first one in case of
        multiple), ``args`` (positional args, usually empty), and all other
        keyword arguments as their own keys.

        Warnings
        --------
        Serialized filesystems may contain sensitive information which have been
        passed to the constructor, such as passwords and tokens. Make sure you
        store and send them in a secure environment!
        """
        from .json import FilesystemJSONEncoder

        json_encoder = FilesystemJSONEncoder()

        cls = type(self)
        proto = self.protocol

        storage_options = dict(self.storage_options)
        if not include_password:
            storage_options.pop("password", None)

        return dict(
            cls=f"{cls.__module__}:{cls.__name__}",
            protocol=proto[0] if isinstance(proto, (tuple, list)) else proto,
            args=json_encoder.make_serializable(self.storage_args),
            **json_encoder.make_serializable(storage_options),
        )

    @staticmethod
    def from_dict(dct: dict[str, Any]) -> AbstractFileSystem:
        """
        Recreate a filesystem instance from dictionary representation.

        See ``.to_dict()`` for the expected structure of the input.

        Parameters
        ----------
        dct: Dict[str, Any]

        Returns
        -------
        file system instance, not necessarily of this particular class.

        Warnings
        --------
        This can import arbitrary modules (as determined by the ``cls`` key).
        Make sure you haven't installed any modules that may execute malicious code
        at import time.
        """
        from .json import FilesystemJSONDecoder

        json_decoder = FilesystemJSONDecoder()

        dct = dict(dct)  # Defensive copy

        cls = FilesystemJSONDecoder.try_resolve_fs_cls(dct)
        if cls is None:
            raise ValueError("Not a serialized AbstractFileSystem")

        dct.pop("cls", None)
        dct.pop("protocol", None)

        return cls(
            *json_decoder.unmake_serializable(dct.pop("args", ())),
            **json_decoder.unmake_serializable(dct),
        )

    def _get_pyarrow_filesystem(self):
        """
        Make a version of the FS instance which will be acceptable to pyarrow
        """
        # all instances already also derive from pyarrow
        return self

    def get_mapper(self, root="", check=False, create=False, missing_exceptions=None):
        """Create key/value store based on this file-system

        Makes a MutableMapping interface to the FS at the given root path.
        See ``fsspec.mapping.FSMap`` for further details.
        """
        from .mapping import FSMap

        return FSMap(
            root,
            self,
            check=check,
            create=create,
            missing_exceptions=missing_exceptions,
        )

    @classmethod
    def clear_instance_cache(cls):
        """
        Clear the cache of filesystem instances.

        Notes
        -----
        Unless overridden by setting the ``cachable`` class attribute to False,
        the filesystem class stores a reference to newly created instances. This
        prevents Python's normal rules around garbage collection from working,
        since the instances refcount will not drop to zero until
        ``clear_instance_cache`` is called.
        """
        cls._cache.clear()

    def created(self, path):
        """Return the created timestamp of a file as a datetime.datetime"""
        raise NotImplementedError

    def modified(self, path):
        """Return the modified timestamp of a file as a datetime.datetime"""
        raise NotImplementedError

    def tree(
        self,
        path: str = "/",
        recursion_limit: int = 2,
        max_display: int = 25,
        display_size: bool = False,
        prefix: str = "",
        is_last: bool = True,
        first: bool = True,
        indent_size: int = 4,
    ) -> str:
        """
        Return a tree-like structure of the filesystem starting from the given path as a string.

        Parameters
        ----------
            path: Root path to start traversal from
            recursion_limit: Maximum depth of directory traversal
            max_display: Maximum number of items to display per directory
            display_size: Whether to display file sizes
            prefix: Current line prefix for visual tree structure
            is_last: Whether current item is last in its level
            first: Whether this is the first call (displays root path)
            indent_size: Number of spaces by indent

        Returns
        -------
            str: A string representing the tree structure.

        Example
        -------
            >>> from fsspec import filesystem

            >>> fs = filesystem('ftp', host='test.rebex.net', user='demo', password='password')
            >>> tree = fs.tree(display_size=True, recursion_limit=3, indent_size=8, max_display=10)
            >>> print(tree)
        """

        def format_bytes(n: int) -> str:
            """Format bytes as text."""
            for prefix, k in (
                ("P", 2**50),
                ("T", 2**40),
                ("G", 2**30),
                ("M", 2**20),
                ("k", 2**10),
            ):
                if n >= 0.9 * k:
                    return f"{n / k:.2f} {prefix}b"
            return f"{n}B"

        result = []

        if first:
            result.append(path)

        if recursion_limit:
            indent = " " * indent_size
            contents = self.ls(path, detail=True)
            contents.sort(
                key=lambda x: (x.get("type") != "directory", x.get("name", ""))
            )

            if max_display is not None and len(contents) > max_display:
                displayed_contents = contents[:max_display]
                remaining_count = len(contents) - max_display
            else:
                displayed_contents = contents
                remaining_count = 0

            for i, item in enumerate(displayed_contents):
                is_last_item = (i == len(displayed_contents) - 1) and (
                    remaining_count == 0
                )

                branch = (
                    "└" + ("─" * (indent_size - 2))
                    if is_last_item
                    else "├" + ("─" * (indent_size - 2))
                )
                branch += " "
                new_prefix = prefix + (
                    indent if is_last_item else "│" + " " * (indent_size - 1)
                )

                name = os.path.basename(item.get("name", ""))

                if display_size and item.get("type") == "directory":
                    sub_contents = self.ls(item.get("name", ""), detail=True)
                    num_files = sum(
                        1 for sub_item in sub_contents if sub_item.get("type") == "file"
                    )
                    num_folders = sum(
                        1
                        for sub_item in sub_contents
                        if sub_item.get("type") == "directory"
                    )

                    if num_files == 0 and num_folders == 0:
                        size = " (empty folder)"
                    elif num_files == 0:
                        size = f" ({num_folders} subfolder{'s' if num_folders > 1 else ''})"
                    elif num_folders == 0:
                        size = f" ({num_files} file{'s' if num_files > 1 else ''})"
                    else:
                        size = f" ({num_files} file{'s' if num_files > 1 else ''}, {num_folders} subfolder{'s' if num_folders > 1 else ''})"
                elif display_size and item.get("type") == "file":
                    size = f" ({format_bytes(item.get('size', 0))})"
                else:
                    size = ""

                result.append(f"{prefix}{branch}{name}{size}")

                if item.get("type") == "directory" and recursion_limit > 0:
                    result.append(
                        self.tree(
                            path=item.get("name", ""),
                            recursion_limit=recursion_limit - 1,
                            max_display=max_display,
                            display_size=display_size,
                            prefix=new_prefix,
                            is_last=is_last_item,
                            first=False,
                            indent_size=indent_size,
                        )
                    )

            if remaining_count > 0:
                more_message = f"{remaining_count} more item(s) not displayed."
                result.append(
                    f"{prefix}{'└' + ('─' * (indent_size - 2))} {more_message}"
                )

        return "\n".join(_ for _ in result if _)

    # ------------------------------------------------------------------------
    # Aliases

    def read_bytes(self, path, start=None, end=None, **kwargs):
        """Alias of `AbstractFileSystem.cat_file`."""
        return self.cat_file(path, start=start, end=end, **kwargs)

    def write_bytes(self, path, value, **kwargs):
        """Alias of `AbstractFileSystem.pipe_file`."""
        self.pipe_file(path, value, **kwargs)

    def makedir(self, path, create_parents=True, **kwargs):
        """Alias of `AbstractFileSystem.mkdir`."""
        return self.mkdir(path, create_parents=create_parents, **kwargs)

    def mkdirs(self, path, exist_ok=False):
        """Alias of `AbstractFileSystem.makedirs`."""
        return self.makedirs(path, exist_ok=exist_ok)

    def listdir(self, path, detail=True, **kwargs):
        """Alias of `AbstractFileSystem.ls`."""
        return self.ls(path, detail=detail, **kwargs)

    def cp(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.copy`."""
        return self.copy(path1, path2, **kwargs)

    def move(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.mv`."""
        return self.mv(path1, path2, **kwargs)

    def stat(self, path, **kwargs):
        """Alias of `AbstractFileSystem.info`."""
        return self.info(path, **kwargs)

    def disk_usage(self, path, total=True, maxdepth=None, **kwargs):
        """Alias of `AbstractFileSystem.du`."""
        return self.du(path, total=total, maxdepth=maxdepth, **kwargs)

    def rename(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.mv`."""
        return self.mv(path1, path2, **kwargs)

    def delete(self, path, recursive=False, maxdepth=None):
        """Alias of `AbstractFileSystem.rm`."""
        return self.rm(path, recursive=recursive, maxdepth=maxdepth)

    def upload(self, lpath, rpath, recursive=False, **kwargs):
        """Alias of `AbstractFileSystem.put`."""
        return self.put(lpath, rpath, recursive=recursive, **kwargs)

    def download(self, rpath, lpath, recursive=False, **kwargs):
        """Alias of `AbstractFileSystem.get`."""
        return self.get(rpath, lpath, recursive=recursive, **kwargs)

    def sign(self, path, expiration=100, **kwargs):
        """Create a signed URL representing the given path

        Some implementations allow temporary URLs to be generated, as a
        way of delegating credentials.

        Parameters
        ----------
        path : str
             The path on the filesystem
        expiration : int
            Number of seconds to enable the URL for (if supported)

        Returns
        -------
        URL : str
            The signed URL

        Raises
        ------
        NotImplementedError : if method is not implemented for a filesystem
        """
        raise NotImplementedError("Sign is not implemented for this filesystem")

    def _isfilestore(self):
        # Originally inherited from pyarrow DaskFileSystem. Keeping this
        # here for backwards compatibility as long as pyarrow uses its
        # legacy fsspec-compatible filesystems and thus accepts fsspec
        # filesystems as well
        return False


class AbstractBufferedFile(io.IOBase):
    """Convenient class to derive from to provide buffering

    In the case that the backend does not provide a pythonic file-like object
    already, this class contains much of the logic to build one. The only
    methods that need to be overridden are ``_upload_chunk``,
    ``_initiate_upload`` and ``_fetch_range``.
    """

    DEFAULT_BLOCK_SIZE = 5 * 2**20
    _details = None

    def __init__(
        self,
        fs,
        path,
        mode="rb",
        block_size="default",
        autocommit=True,
        cache_type="readahead",
        cache_options=None,
        size=None,
        **kwargs,
    ):
        """
        Template for files with buffered reading and writing

        Parameters
        ----------
        fs: instance of FileSystem
        path: str
            location in file-system
        mode: str
            Normal file modes. Currently only 'wb', 'ab' or 'rb'. Some file
            systems may be read-only, and some may not support append.
        block_size: int
            Buffer size for reading or writing, 'default' for class default
        autocommit: bool
            Whether to write to final destination; may only impact what
            happens when file is being closed.
        cache_type: {"readahead", "none", "mmap", "bytes"}, default "readahead"
            Caching policy in read mode. See the definitions in ``core``.
        cache_options : dict
            Additional options passed to the constructor for the cache specified
            by `cache_type`.
        size: int
            If given and in read mode, suppressed having to look up the file size
        kwargs:
            Gets stored as self.kwargs
        """
        from .core import caches

        self.path = path
        self.fs = fs
        self.mode = mode
        self.blocksize = (
            self.DEFAULT_BLOCK_SIZE if block_size in ["default", None] else block_size
        )
        self.loc = 0
        self.autocommit = autocommit
        self.end = None
        self.start = None
        self.closed = False

        if cache_options is None:
            cache_options = {}

        if "trim" in kwargs:
            warnings.warn(
                "Passing 'trim' to control the cache behavior has been deprecated. "
                "Specify it within the 'cache_options' argument instead.",
                FutureWarning,
            )
            cache_options["trim"] = kwargs.pop("trim")

        self.kwargs = kwargs

        if mode not in {"ab", "rb", "wb", "xb"}:
            raise NotImplementedError("File mode not supported")
        if mode == "rb":
            if size is not None:
                self.size = size
            else:
                self.size = self.details["size"]
            self.cache = caches[cache_type](
