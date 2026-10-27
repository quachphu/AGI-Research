    contains only plain integers — no DTensorSpec or DeviceMesh objects.
    """

    class RunDTensorRngOp(HigherOrderOperator):
        def __init__(self):
            # Not cacheable: the compiled wrapper bakes in the current device
            # (torch.cuda.set_device(N)) and the rank-dependent start/end offset
            # ints; sharing the cached graph across ranks routes every rank's
            # set_rng_state to one rank's device and offsets, leaving the other
            # ranks' generators unchanged (#185520).
            super().__init__("run_dtensor_rng_op", cacheable=False)

        def __call__(self, start_offset_incr, end_offset_incr, op, *args, **kwargs):
            # pyrefly: ignore [missing-attribute]
            return super().__call__(
                start_offset_incr, end_offset_incr, op, *args, **kwargs
            )

    run_dtensor_rng_op = RunDTensorRngOp()

    run_dtensor_rng_op.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(run_dtensor_rng_op, deferred_error=True)
    )

    @run_dtensor_rng_op.py_impl(DispatchKey.CUDA)
    def impl_cuda(start_offset_incr, end_offset_incr, op, *args, **kwargs):
        from torch.distributed.tensor._random import _PhiloxState

        state = _PhiloxState(torch.cuda.get_rng_state())
        old_offset = state.offset.clone()
        state.offset = old_offset + start_offset_incr

        device = (
            args[0].device
            if args and hasattr(args[0], "device")
            else torch.device(f"cuda:{torch.cuda.current_device()}")
        )
        with torch.random.fork_rng(devices=[device], device_type="cuda"):
            torch.cuda.set_rng_state(state.state)
            try:
                out = op(*args, **kwargs)
            finally:
                state.offset = old_offset + end_offset_incr

        torch.cuda.set_rng_state(state.state)
        return out

    @run_dtensor_rng_op.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(start_offset_incr, end_offset_incr, op, *args, **kwargs):
        device = get_device(args, kwargs)
        if device != "cuda":
            raise RuntimeError(
                f"run_dtensor_rng_op only supports CUDA device, got {device}. "
                f"This operator is designed for distributed random operations on CUDA."
            )
        return impl_cuda(start_offset_incr, end_offset_incr, op, *args, **kwargs)

    @register_fake(run_dtensor_rng_op, skip_cache=True)
    def impl_fake_tensor_mode(start_offset_incr, end_offset_incr, op, *args, **kwargs):
        return op(*args, **kwargs)

    @run_dtensor_rng_op.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(
        mode, start_offset_incr, end_offset_incr, op, *args, **kwargs
    ):
        with disable_proxy_modes_tracing():
            out = run_dtensor_rng_op(
                start_offset_incr, end_offset_incr, op, *args, **kwargs
            )
        proxy_args = pytree.tree_map(
            mode.tracer.unwrap_proxy, (start_offset_incr, end_offset_incr, op, *args)
        )
        proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
        out_proxy = mode.tracer.create_proxy(
            "call_function", run_dtensor_rng_op, proxy_args, proxy_kwargs
        )
        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

    @run_dtensor_rng_op.py_functionalize_impl
    def impl_functional(ctx, start_offset_incr, end_offset_incr, op, *args, **kwargs):
        unwrapped_args = ctx.unwrap_tensors(args)
        unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

        with ctx.redispatch_to_next():
            out = run_dtensor_rng_op(
                start_offset_incr,
                end_offset_incr,
                op,
                *unwrapped_args,
                **unwrapped_kwargs,
            )
            return ctx.wrap_tensors(out)

    return run_dtensor_rng_op


run_dtensor_rng_op = register_run_dtensor_rng_op()


def register_rng_prims():
    register_philox_rand()
<|endoftext|>from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack

from torch._prims.context import TorchRefsMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx


T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")


def execute(
    gm: GraphModule,
    *args: Unpack[Ts],
    executor: str = "aten",
    executor_parameters: dict | None = None,
) -> Any:
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    if executor == "aten":
        return gm.forward(*args)

    msg = f"Received unexpected value for 'executor': {executor}. Allowed values are: aten."
    raise ValueError(msg)


def make_traced(fn: Callable[P, T]) -> Callable[P, T]:
    """
    Returns a function that, when called, will
    trace its torch operations to prims and then
    execute those prims on the requested trace executor
    (possibly lowering them to that trace executor first).

    Only supports the torch operations defined in _torch_to_reference_map
    in context.py and operations with positional args. All args must
    be tensors.
    In the near future all these restrictions will be lifted.

    Example usage:

    def foo(a, b):
      return torch.add(a, b)

    traced_foo = make_traced(foo)

    a = torch.randn((1, 2, 3, 4, 5), device='cuda')
    b = torch.randn((1, 2, 3, 4, 5), device='cuda')
    result = traced_foo(a, b, executor='aten')
    """

    def _traced(*args: P.args, **kwargs: P.kwargs) -> T:
        executor = str(kwargs.pop("executor", "aten"))

        # TODO: caching
        wrapped, all_args = wrapper_and_args_for_make_fx(fn, args, kwargs)

        with TorchRefsMode():
            gm = make_fx(wrapped)(all_args)
        return execute(gm, all_args, executor=executor)

    return _traced  # type: ignore[return-value]
<|endoftext|>import os
import types
from collections.abc import MutableSequence
from functools import total_ordering

__version__ = "1.8.0"

__all__ = ("FrozenList", "PyFrozenList")  # type: Tuple[str, ...]


NO_EXTENSIONS = bool(os.environ.get("FROZENLIST_NO_EXTENSIONS"))  # type: bool


@total_ordering
class FrozenList(MutableSequence):
    __slots__ = ("_frozen", "_items")
    __class_getitem__ = classmethod(types.GenericAlias)

    def __init__(self, items=None):
        self._frozen = False
        if items is not None:
            items = list(items)
        else:
            items = []
        self._items = items

    @property
    def frozen(self):
        return self._frozen

    def freeze(self):
        self._frozen = True

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        if self._frozen:
            raise RuntimeError("Cannot modify frozen list.")
        self._items[index] = value

    def __delitem__(self, index):
        if self._frozen:
            raise RuntimeError("Cannot modify frozen list.")
        del self._items[index]

    def __len__(self):
        return self._items.__len__()

    def __iter__(self):
        return self._items.__iter__()

    def __reversed__(self):
        return self._items.__reversed__()

    def __eq__(self, other):
        return list(self) == other

    def __le__(self, other):
        return list(self) <= other

    def insert(self, pos, item):
        if self._frozen:
            raise RuntimeError("Cannot modify frozen list.")
        self._items.insert(pos, item)

    def __repr__(self):
        return f"<FrozenList(frozen={self._frozen}, {self._items!r})>"

    def __hash__(self):
        if self._frozen:
            return hash(tuple(self))
        else:
            raise RuntimeError("Cannot hash unfrozen list.")


PyFrozenList = FrozenList


if not NO_EXTENSIONS:
    try:
        from ._frozenlist import FrozenList as CFrozenList  # type: ignore
    except ImportError:  # pragma: no cover
        pass
    else:
        FrozenList = CFrozenList  # type: ignore
<|endoftext|>"""
Stacked area plot for 1D arrays inspired by Douglas Y'barbo's stackoverflow
answer:
https://stackoverflow.com/q/2225995/

(https://stackoverflow.com/users/66549/doug)
"""

import itertools

import numpy as np

from matplotlib import _api

__all__ = ['stackplot']


def stackplot(axes, x, *args,
              labels=(), colors=None, hatch=None, baseline='zero',
              **kwargs):
    """
    Draw a stacked area plot or a streamgraph.

    Parameters
    ----------
    x : (N,) array-like

    y : (M, N) array-like
        The data can be either stacked or unstacked. Each of the following
        calls is legal::

            stackplot(x, y)  # where y has shape (M, N) e.g. y = [y1, y2, y3, y4]
            stackplot(x, y1, y2, y3, y4)  # where y1, y2, y3, y4 have length N

    baseline : {'zero', 'sym', 'wiggle', 'weighted_wiggle'}
        Method used to calculate the baseline:

        - ``'zero'``: Constant zero baseline, i.e. a simple stacked plot.
        - ``'sym'``:  Symmetric around zero and is sometimes called
          'ThemeRiver'.
        - ``'wiggle'``: Minimizes the sum of the squared slopes.
        - ``'weighted_wiggle'``: Does the same but weights to account for
          size of each layer. It is also called 'Streamgraph'-layout. More
          details can be found at http://leebyron.com/streamgraph/.

    labels : list of str, optional
        A sequence of labels to assign to each data series. If unspecified,
        then no labels will be applied to artists.

    colors : list of :mpltype:`color`, optional
        A sequence of colors to be cycled through and used to color the stacked
        areas. The sequence need not be exactly the same length as the number
        of provided *y*, in which case the colors will repeat from the
        beginning.

        If not specified, the colors from the Axes property cycle will be used.

    hatch : list of str, default: None
        A sequence of hatching styles.  See
        :doc:`/gallery/shapes_and_collections/hatch_style_reference`.
        The sequence will be cycled through for filling the
        stacked areas from bottom to top.
        It need not be exactly the same length as the number
        of provided *y*, in which case the styles will repeat from the
        beginning.

        .. versionadded:: 3.9
           Support for list input

    data : indexable object, optional
        DATA_PARAMETER_PLACEHOLDER

    **kwargs
        All other keyword arguments are passed to `.Axes.fill_between`.

    Returns
    -------
    list of `.PolyCollection`
        A list of `.PolyCollection` instances, one for each element in the
        stacked area plot.
    """

    y = np.vstack(args)

    labels = iter(labels)
    if colors is not None:
        colors = itertools.cycle(colors)
    else:
        colors = (axes._get_lines.get_next_color() for _ in y)

    if hatch is None or isinstance(hatch, str):
        hatch = itertools.cycle([hatch])
    else:
        hatch = itertools.cycle(hatch)

    # Assume data passed has not been 'stacked', so stack it here.
    # We'll need a float buffer for the upcoming calculations.
    stack = np.cumsum(y, axis=0, dtype=np.promote_types(y.dtype, np.float32))

    _api.check_in_list(['zero', 'sym', 'wiggle', 'weighted_wiggle'],
                       baseline=baseline)
    if baseline == 'zero':
        first_line = 0.

    elif baseline == 'sym':
        first_line = -np.sum(y, 0) * 0.5
        stack += first_line[None, :]

    elif baseline == 'wiggle':
        m = y.shape[0]
        first_line = (y * (m - 0.5 - np.arange(m)[:, None])).sum(0)
        first_line /= -m
        stack += first_line

    elif baseline == 'weighted_wiggle':
        total = np.sum(y, 0)
        # multiply by 1/total (or zero) to avoid infinities in the division:
        inv_total = np.zeros_like(total)
        mask = total > 0
        inv_total[mask] = 1.0 / total[mask]
        increase = np.hstack((y[:, 0:1], np.diff(y)))
        below_size = total - stack
        below_size += 0.5 * y
        move_up = below_size * inv_total
        move_up[:, 0] = 0.5
        center = (move_up - 0.5) * increase
        center = np.cumsum(center.sum(0))
        first_line = center - 0.5 * total
        stack += first_line

    # Color between x = 0 and the first array.
    coll = axes.fill_between(x, first_line, stack[0, :],
                             facecolor=next(colors),
                             hatch=next(hatch),
                             label=next(labels, None),
                             **kwargs)
    coll.sticky_edges.y[:] = [0]
    r = [coll]

    # Color between array i-1 and array i
    for i in range(len(y) - 1):
        r.append(axes.fill_between(x, stack[i, :], stack[i + 1, :],
                                   facecolor=next(colors),
                                   hatch=next(hatch),
                                   label=next(labels, None),
                                   **kwargs))
    return r
<|endoftext|>"""
A module providing some utility functions regarding Bézier path manipulation.
"""

from functools import lru_cache
import math
import warnings

import numpy as np

from matplotlib import _api


# same algorithm as 3.8's math.comb
@np.vectorize
@lru_cache(maxsize=128)
def _comb(n, k):
    if k > n:
        return 0
    k = min(k, n - k)
    i = np.arange(1, k + 1)
    return np.prod((n + 1 - i)/i).astype(int)


class NonIntersectingPathException(ValueError):
    pass


# some functions


def get_intersection(cx1, cy1, cos_t1, sin_t1,
                     cx2, cy2, cos_t2, sin_t2):
    """
    Return the intersection between the line through (*cx1*, *cy1*) at angle
    *t1* and the line through (*cx2*, *cy2*) at angle *t2*.
    """

    # line1 => sin_t1 * (x - cx1) - cos_t1 * (y - cy1) = 0.
    # line1 => sin_t1 * x + cos_t1 * y = sin_t1*cx1 - cos_t1*cy1

    line1_rhs = sin_t1 * cx1 - cos_t1 * cy1
    line2_rhs = sin_t2 * cx2 - cos_t2 * cy2

    # rhs matrix
    a, b = sin_t1, -cos_t1
    c, d = sin_t2, -cos_t2

    ad_bc = a * d - b * c
    if abs(ad_bc) < 1e-12:
        raise ValueError("Given lines do not intersect. Please verify that "
                         "the angles are not equal or differ by 180 degrees.")

    # rhs_inverse
    a_, b_ = d, -b
    c_, d_ = -c, a
    a_, b_, c_, d_ = (k / ad_bc for k in [a_, b_, c_, d_])

    x = a_ * line1_rhs + b_ * line2_rhs
    y = c_ * line1_rhs + d_ * line2_rhs

    return x, y


def get_normal_points(cx, cy, cos_t, sin_t, length):
    """
    For a line passing through (*cx*, *cy*) and having an angle *t*, return
    locations of the two points located along its perpendicular line at the
    distance of *length*.
    """

    if length == 0.:
        return cx, cy, cx, cy

    cos_t1, sin_t1 = sin_t, -cos_t
    cos_t2, sin_t2 = -sin_t, cos_t

    x1, y1 = length * cos_t1 + cx, length * sin_t1 + cy
    x2, y2 = length * cos_t2 + cx, length * sin_t2 + cy

    return x1, y1, x2, y2


# BEZIER routines

# subdividing bezier curve
# http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html


def _de_casteljau1(beta, t):
    next_beta = beta[:-1] * (1 - t) + beta[1:] * t
    return next_beta


def split_de_casteljau(beta, t):
    """
    Split a Bézier segment defined by its control points *beta* into two
    separate segments divided at *t* and return their control points.
    """
    beta = np.asarray(beta)
    beta_list = [beta]
    while True:
        beta = _de_casteljau1(beta, t)
        beta_list.append(beta)
        if len(beta) == 1:
            break
    left_beta = [beta[0] for beta in beta_list]
    right_beta = [beta[-1] for beta in reversed(beta_list)]

    return left_beta, right_beta


def find_bezier_t_intersecting_with_closedpath(
        bezier_point_at_t, inside_closedpath, t0=0., t1=1., tolerance=0.01):
    """
    Find the intersection of the Bézier curve with a closed path.

    The intersection point *t* is approximated by two parameters *t0*, *t1*
    such that *t0* <= *t* <= *t1*.

    Search starts from *t0* and *t1* and uses a simple bisecting algorithm
    therefore one of the end points must be inside the path while the other
    doesn't. The search stops when the distance of the points parametrized by
    *t0* and *t1* gets smaller than the given *tolerance*.

    Parameters
    ----------
    bezier_point_at_t : callable
        A function returning x, y coordinates of the Bézier at parameter *t*.
        It must have the signature::

            bezier_point_at_t(t: float) -> tuple[float, float]

    inside_closedpath : callable
        A function returning True if a given point (x, y) is inside the
        closed path. It must have the signature::

            inside_closedpath(point: tuple[float, float]) -> bool

    t0, t1 : float
        Start parameters for the search.

    tolerance : float
        Maximal allowed distance between the final points.

    Returns
    -------
    t0, t1 : float
        The Bézier path parameters.
    """
    start = bezier_point_at_t(t0)
    end = bezier_point_at_t(t1)

    start_inside = inside_closedpath(start)
    end_inside = inside_closedpath(end)

    if start_inside == end_inside and start != end:
        raise NonIntersectingPathException(
            "Both points are on the same side of the closed path")

    while True:

        # return if the distance is smaller than the tolerance
        if np.hypot(start[0] - end[0], start[1] - end[1]) < tolerance:
            return t0, t1

        # calculate the middle point
        middle_t = 0.5 * (t0 + t1)
        middle = bezier_point_at_t(middle_t)
        middle_inside = inside_closedpath(middle)

        if start_inside ^ middle_inside:
            t1 = middle_t
            if end == middle:
                # Edge case where infinite loop is possible
                # Caused by large numbers relative to tolerance
                return t0, t1
            end = middle
        else:
            t0 = middle_t
            if start == middle:
                # Edge case where infinite loop is possible
                # Caused by large numbers relative to tolerance
                return t0, t1
            start = middle
            start_inside = middle_inside


class BezierSegment:
    """
    A d-dimensional Bézier segment.

    Parameters
    ----------
    control_points : (N, d) array
        Location of the *N* control points.
    """

    def __init__(self, control_points):
        self._cpoints = np.asarray(control_points)
        self._N, self._d = self._cpoints.shape
        self._orders = np.arange(self._N)
        coeff = [math.factorial(self._N - 1)
                 // (math.factorial(i) * math.factorial(self._N - 1 - i))
                 for i in range(self._N)]
        self._px = (self._cpoints.T * coeff).T

    def __call__(self, t):
        """
        Evaluate the Bézier curve at point(s) *t* in [0, 1].

        Parameters
        ----------
        t : (k,) array-like
            Points at which to evaluate the curve.

        Returns
        -------
        (k, d) array
            Value of the curve for each point in *t*.
        """
        t = np.asarray(t)
        return (np.power.outer(1 - t, self._orders[::-1])
                * np.power.outer(t, self._orders)) @ self._px

    def point_at_t(self, t):
        """
        Evaluate the curve at a single point, returning a tuple of *d* floats.
        """
        return tuple(self(t))

    @property
    def control_points(self):
        """The control points of the curve."""
        return self._cpoints

    @property
    def dimension(self):
        """The dimension of the curve."""
        return self._d

    @property
    def degree(self):
        """Degree of the polynomial. One less the number of control points."""
        return self._N - 1

    @property
    def polynomial_coefficients(self):
        r"""
        The polynomial coefficients of the Bézier curve.

        .. warning:: Follows opposite convention from `numpy.polyval`.

        Returns
        -------
        (n+1, d) array
            Coefficients after expanding in polynomial basis, where :math:`n`
            is the degree of the Bézier curve and :math:`d` its dimension.
            These are the numbers (:math:`C_j`) such that the curve can be
            written :math:`\sum_{j=0}^n C_j t^j`.

        Notes
        -----
        The coefficients are calculated as

        .. math::

            {n \choose j} \sum_{i=0}^j (-1)^{i+j} {j \choose i} P_i

        where :math:`P_i` are the control points of the curve.
        """
        n = self.degree
        # matplotlib uses n <= 4. overflow plausible starting around n = 15.
        if n > 10:
            warnings.warn("Polynomial coefficients formula unstable for high "
                          "order Bezier curves!", RuntimeWarning)
        P = self.control_points
        j = np.arange(n+1)[:, None]
        i = np.arange(n+1)[None, :]  # _comb is non-zero for i <= j
        prefactor = (-1)**(i + j) * _comb(j, i)  # j on axis 0, i on axis 1
        return _comb(n, j) * prefactor @ P  # j on axis 0, self.dimension on 1

    def axis_aligned_extrema(self):
        """
        Return the dimension and location of the curve's interior extrema.

        The extrema are the points along the curve where one of its partial
        derivatives is zero.

        Returns
        -------
        dims : array of int
            Index :math:`i` of the partial derivative which is zero at each
            interior extrema.
        dzeros : array of float
            Of same size as dims. The :math:`t` such that :math:`d/dx_i B(t) =
            0`
        """
        n = self.degree
        if n <= 1:
            return np.array([]), np.array([])
        Cj = self.polynomial_coefficients
        dCj = np.arange(1, n+1)[:, None] * Cj[1:]
        dims = []
        roots = []
        for i, pi in enumerate(dCj.T):
            r = np.roots(pi[::-1])
            roots.append(r)
            dims.append(np.full_like(r, i))
        roots = np.concatenate(roots)
        dims = np.concatenate(dims)
        in_range = np.isreal(roots) & (roots >= 0) & (roots <= 1)
        return dims[in_range], np.real(roots)[in_range]


def split_bezier_intersecting_with_closedpath(
        bezier, inside_closedpath, tolerance=0.01):
    """
    Split a Bézier curve into two at the intersection with a closed path.

    Parameters
    ----------
    bezier : (N, 2) array-like
        Control points of the Bézier segment. See `.BezierSegment`.
    inside_closedpath : callable
        A function returning True if a given point (x, y) is inside the
        closed path. See also `.find_bezier_t_intersecting_with_closedpath`.
    tolerance : float
        The tolerance for the intersection. See also
        `.find_bezier_t_intersecting_with_closedpath`.

    Returns
    -------
    left, right
        Lists of control points for the two Bézier segments.
    """

    bz = BezierSegment(bezier)
    bezier_point_at_t = bz.point_at_t

    t0, t1 = find_bezier_t_intersecting_with_closedpath(
        bezier_point_at_t, inside_closedpath, tolerance=tolerance)

    _left, _right = split_de_casteljau(bezier, (t0 + t1) / 2.)
    return _left, _right


# matplotlib specific


def split_path_inout(path, inside, tolerance=0.01, reorder_inout=False):
    """
    Divide a path into two segments at the point where ``inside(x, y)`` becomes
    False.
    """
    from .path import Path
    path_iter = path.iter_segments()

    ctl_points, command = next(path_iter)
    begin_inside = inside(ctl_points[-2:])  # true if begin point is inside

    ctl_points_old = ctl_points

    iold = 0
    i = 1

    for ctl_points, command in path_iter:
        iold = i
        i += len(ctl_points) // 2
        if inside(ctl_points[-2:]) != begin_inside:
            bezier_path = np.concatenate([ctl_points_old[-2:], ctl_points])
            break
        ctl_points_old = ctl_points
    else:
        raise ValueError("The path does not intersect with the patch")

    bp = bezier_path.reshape((-1, 2))
    left, right = split_bezier_intersecting_with_closedpath(
        bp, inside, tolerance)
    if len(left) == 2:
        codes_left = [Path.LINETO]
        codes_right = [Path.MOVETO, Path.LINETO]
    elif len(left) == 3:
        codes_left = [Path.CURVE3, Path.CURVE3]
        codes_right = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    elif len(left) == 4:
        codes_left = [Path.CURVE4, Path.CURVE4, Path.CURVE4]
        codes_right = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        raise AssertionError("This should never be reached")

    verts_left = left[1:]
    verts_right = right[:]

    if path.codes is None:
        path_in = Path(np.concatenate([path.vertices[:i], verts_left]))
        path_out = Path(np.concatenate([verts_right, path.vertices[i:]]))

    else:
        path_in = Path(np.concatenate([path.vertices[:iold], verts_left]),
                       np.concatenate([path.codes[:iold], codes_left]))

        path_out = Path(np.concatenate([verts_right, path.vertices[i:]]),
                        np.concatenate([codes_right, path.codes[i:]]))

    if reorder_inout and not begin_inside:
        path_in, path_out = path_out, path_in

    return path_in, path_out


def inside_circle(cx, cy, r):
    """
    Return a function that checks whether a point is in a circle with center
    (*cx*, *cy*) and radius *r*.

    The returned function has the signature::

        f(xy: tuple[float, float]) -> bool
    """
    r2 = r ** 2

    def _f(xy):
        x, y = xy
        return (x - cx) ** 2 + (y - cy) ** 2 < r2
    return _f


# quadratic Bezier lines

def get_cos_sin(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    d = (dx * dx + dy * dy) ** .5
    # Account for divide by zero
    if d == 0:
        return 0.0, 0.0
    return dx / d, dy / d


def check_if_parallel(dx1, dy1, dx2, dy2, tolerance=1.e-5):
    """
    Check if two lines are parallel.

    Parameters
    ----------
    dx1, dy1, dx2, dy2 : float
        The gradients *dy*/*dx* of the two lines.
    tolerance : float
        The angular tolerance in radians up to which the lines are considered
        parallel.

    Returns
    -------
    is_parallel
        - 1 if two lines are parallel in same direction.
        - -1 if two lines are parallel in opposite direction.
        - False otherwise.
    """
    theta1 = np.arctan2(dx1, dy1)
    theta2 = np.arctan2(dx2, dy2)
    dtheta = abs(theta1 - theta2)
    if dtheta < tolerance:
        return 1
    elif abs(dtheta - np.pi) < tolerance:
        return -1
    else:
        return False


def get_parallels(bezier2, width):
    """
    Given the quadratic Bézier control points *bezier2*, returns
    control points of quadratic Bézier lines roughly parallel to given
    one separated by *width*.
    """

    # The parallel Bezier lines are constructed by following ways.
    #  c1 and c2 are control points representing the start and end of the
    #  Bezier line.
    #  cm is the middle point

    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c2x, c2y = bezier2[2]

    parallel_test = check_if_parallel(c1x - cmx, c1y - cmy,
                                      cmx - c2x, cmy - c2y)

    if parallel_test == -1:
        _api.warn_external(
            "Lines do not intersect. A straight line is used instead.")
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, c2x, c2y)
        cos_t2, sin_t2 = cos_t1, sin_t1
    else:
        # t1 and t2 is the angle between c1 and cm, cm, c2.  They are
        # also an angle of the tangential line of the path at c1 and c2
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
        cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c2x, c2y)

    # find c1_left, c1_right which are located along the lines
    # through c1 and perpendicular to the tangential lines of the
    # Bezier path at a distance of width. Same thing for c2_left and
    # c2_right with respect to c2.
    c1x_left, c1y_left, c1x_right, c1y_right = (
        get_normal_points(c1x, c1y, cos_t1, sin_t1, width)
    )
    c2x_left, c2y_left, c2x_right, c2y_right = (
        get_normal_points(c2x, c2y, cos_t2, sin_t2, width)
    )

    # find cm_left which is the intersecting point of a line through
    # c1_left with angle t1 and a line through c2_left with angle
    # t2. Same with cm_right.
    try:
        cmx_left, cmy_left = get_intersection(c1x_left, c1y_left, cos_t1,
                                              sin_t1, c2x_left, c2y_left,
                                              cos_t2, sin_t2)
        cmx_right, cmy_right = get_intersection(c1x_right, c1y_right, cos_t1,
                                                sin_t1, c2x_right, c2y_right,
                                                cos_t2, sin_t2)
    except ValueError:
        # Special case straight lines, i.e., angle between two lines is
        # less than the threshold used by get_intersection (we don't use
        # check_if_parallel as the threshold is not the same).
        cmx_left, cmy_left = (
            0.5 * (c1x_left + c2x_left), 0.5 * (c1y_left + c2y_left)
        )
        cmx_right, cmy_right = (
            0.5 * (c1x_right + c2x_right), 0.5 * (c1y_right + c2y_right)
        )

    # the parallel Bezier lines are created with control points of
    # [c1_left, cm_left, c2_left] and [c1_right, cm_right, c2_right]
    path_left = [(c1x_left, c1y_left),
                 (cmx_left, cmy_left),
                 (c2x_left, c2y_left)]
    path_right = [(c1x_right, c1y_right),
                  (cmx_right, cmy_right),
                  (c2x_right, c2y_right)]

    return path_left, path_right


def find_control_points(c1x, c1y, mmx, mmy, c2x, c2y):
    """
    Find control points of the Bézier curve passing through (*c1x*, *c1y*),
    (*mmx*, *mmy*), and (*c2x*, *c2y*), at parametric values 0, 0.5, and 1.
    """
    cmx = .5 * (4 * mmx - (c1x + c2x))
    cmy = .5 * (4 * mmy - (c1y + c2y))
    return [(c1x, c1y), (cmx, cmy), (c2x, c2y)]


def make_wedged_bezier2(bezier2, width, w1=1., wm=0.5, w2=0.):
    """
    Being similar to `get_parallels`, returns control points of two quadratic
    Bézier lines having a width roughly parallel to given one separated by
    *width*.
    """

    # c1, cm, c2
    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c3x, c3y = bezier2[2]

    # t1 and t2 is the angle between c1 and cm, cm, c3.
    # They are also an angle of the tangential line of the path at c1 and c3
    cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
    cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c3x, c3y)

    # find c1_left, c1_right which are located along the lines
    # through c1 and perpendicular to the tangential lines of the
    # Bezier path at a distance of width. Same thing for c3_left and
    # c3_right with respect to c3.
    c1x_left, c1y_left, c1x_right, c1y_right = (
        get_normal_points(c1x, c1y, cos_t1, sin_t1, width * w1)
    )
    c3x_left, c3y_left, c3x_right, c3y_right = (
        get_normal_points(c3x, c3y, cos_t2, sin_t2, width * w2)
    )

    # find c12, c23 and c123 which are middle points of c1-cm, cm-c3 and
    # c12-c23
    c12x, c12y = (c1x + cmx) * .5, (c1y + cmy) * .5
    c23x, c23y = (cmx + c3x) * .5, (cmy + c3y) * .5
    c123x, c123y = (c12x + c23x) * .5, (c12y + c23y) * .5

    # tangential angle of c123 (angle between c12 and c23)
    cos_t123, sin_t123 = get_cos_sin(c12x, c12y, c23x, c23y)

    c123x_left, c123y_left, c123x_right, c123y_right = (
        get_normal_points(c123x, c123y, cos_t123, sin_t123, width * wm)
    )

    path_left = find_control_points(c1x_left, c1y_left,
                                    c123x_left, c123y_left,
                                    c3x_left, c3y_left)
    path_right = find_control_points(c1x_right, c1y_right,
                                     c123x_right, c123y_right,
                                     c3x_right, c3y_right)

    return path_left, path_right
<|endoftext|>"""
The inset module defines the InsetIndicator class, which draws the rectangle and
connectors required for `.Axes.indicate_inset` and `.Axes.indicate_inset_zoom`.
"""

from . import _api, artist, transforms
from matplotlib.patches import ConnectionPatch, PathPatch, Rectangle
from matplotlib.path import Path


_shared_properties = ('alpha', 'edgecolor', 'linestyle', 'linewidth')


class InsetIndicator(artist.Artist):
    """
    An artist to highlight an area of interest.

    An inset indicator is a rectangle on the plot at the position indicated by
    *bounds* that optionally has lines that connect the rectangle to an inset
    Axes (`.Axes.inset_axes`).

    .. versionadded:: 3.10
    """
    zorder = 4.99

    def __init__(self, bounds=None, inset_ax=None, zorder=None, **kwargs):
        """
        Parameters
        ----------
        bounds : [x0, y0, width, height], optional
            Lower-left corner of rectangle to be marked, and its width
            and height.  If not set, the bounds will be calculated from the
            data limits of inset_ax, which must be supplied.

        inset_ax : `~.axes.Axes`, optional
            An optional inset Axes to draw connecting lines to.  Two lines are
            drawn connecting the indicator box to the inset Axes on corners
            chosen so as to not overlap with the indicator box.

        zorder : float, default: 4.99
            Drawing order of the rectangle and connector lines.  The default,
            4.99, is just below the default level of inset Axes.

        **kwargs
            Other keyword arguments are passed on to the `.Rectangle` patch.
        """
        if bounds is None and inset_ax is None:
            raise ValueError("At least one of bounds or inset_ax must be supplied")

        self._inset_ax = inset_ax

        if bounds is None:
            # Work out bounds from inset_ax
            self._auto_update_bounds = True
            bounds = self._bounds_from_inset_ax()
        else:
            self._auto_update_bounds = False

        x, y, width, height = bounds

        self._rectangle = Rectangle((x, y), width, height, clip_on=False, **kwargs)

        # Connector positions cannot be calculated till the artist has been added
        # to an axes, so just make an empty list for now.
        self._connectors = []

        super().__init__()
        self.set_zorder(zorder)

        # Initial style properties for the artist should match the rectangle.
        for prop in _shared_properties:
            setattr(self, f'_{prop}', artist.getp(self._rectangle, prop))

    def _shared_setter(self, prop, val):
        """
        Helper function to set the same style property on the artist and its children.
        """
        setattr(self, f'_{prop}', val)

        artist.setp([self._rectangle, *self._connectors], prop, val)

    def set_alpha(self, alpha):
        # docstring inherited
        self._shared_setter('alpha', alpha)

    def set_edgecolor(self, color):
        """
        Set the edge color of the rectangle and the connectors.

        Parameters
        ----------
        color : :mpltype:`color` or None
        """
        self._shared_setter('edgecolor', color)

    def set_color(self, c):
        """
        Set the edgecolor of the rectangle and the connectors, and the
        facecolor for the rectangle.

        Parameters
        ----------
        c : :mpltype:`color`
        """
        self._shared_setter('edgecolor', c)
        self._shared_setter('facecolor', c)

    def set_linewidth(self, w):
        """
        Set the linewidth in points of the rectangle and the connectors.

        Parameters
        ----------
        w : float or None
        """
        self._shared_setter('linewidth', w)

    def set_linestyle(self, ls):
        """
        Set the linestyle of the rectangle and the connectors.

        ==========================================  =================
        linestyle                                   description
        ==========================================  =================
        ``'-'`` or ``'solid'``                      solid line
        ``'--'`` or ``'dashed'``                    dashed line
        ``'-.'`` or ``'dashdot'``                   dash-dotted line
        ``':'`` or ``'dotted'``                     dotted line
        ``'none'``, ``'None'``, ``' '``, or ``''``  draw nothing
        ==========================================  =================

        Alternatively a dash tuple of the following form can be provided::

            (offset, onoffseq)

        where ``onoffseq`` is an even length tuple of on and off ink in points.

        Parameters
        ----------
        ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            The line style.
        """
        self._shared_setter('linestyle', ls)

    def _bounds_from_inset_ax(self):
        xlim = self._inset_ax.get_xlim()
        ylim = self._inset_ax.get_ylim()
        return (xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0])

    def _update_connectors(self):
        (x, y) = self._rectangle.get_xy()
        width = self._rectangle.get_width()
        height = self._rectangle.get_height()

        existing_connectors = self._connectors or [None] * 4

        # connect the inset_axes to the rectangle
        for xy_inset_ax, existing in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                                         existing_connectors):
            # inset_ax positions are in axes coordinates
            # The 0, 1 values define the four edges if the inset_ax
            # lower_left, upper_left, lower_right upper_right.
            ex, ey = xy_inset_ax
            if self.axes.xaxis.get_inverted():
                ex = 1 - ex
            if self.axes.yaxis.get_inverted():
                ey = 1 - ey
            xy_data = x + ex * width, y + ey * height
            if existing is None:
                # Create new connection patch with styles inherited from the
                # parent artist.
                p = ConnectionPatch(
                    xyA=xy_inset_ax, coordsA=self._inset_ax.transAxes,
                    xyB=xy_data, coordsB=self.axes.transData,
                    arrowstyle="-",
                    edgecolor=self._edgecolor, alpha=self.get_alpha(),
                    linestyle=self._linestyle, linewidth=self._linewidth)
                self._connectors.append(p)
            else:
                # Only update positioning of existing connection patch.  We
                # do not want to override any style settings made by the user.
                existing.xy1 = xy_inset_ax
                existing.xy2 = xy_data
                existing.coords1 = self._inset_ax.transAxes
                existing.coords2 = self.axes.transData

        if existing is None:
            # decide which two of the lines to keep visible....
            pos = self._inset_ax.get_position()
            bboxins = pos.transformed(self.get_figure(root=False).transSubfigure)
            rectbbox = transforms.Bbox.from_bounds(x, y, width, height).transformed(
                self._rectangle.get_transform())
            x0 = rectbbox.x0 < bboxins.x0
            x1 = rectbbox.x1 < bboxins.x1
            y0 = rectbbox.y0 < bboxins.y0
            y1 = rectbbox.y1 < bboxins.y1
            self._connectors[0].set_visible(x0 ^ y0)
            self._connectors[1].set_visible(x0 == y1)
            self._connectors[2].set_visible(x1 == y0)
            self._connectors[3].set_visible(x1 ^ y1)

    @property
    def rectangle(self):
        """`.Rectangle`: the indicator frame."""
        return self._rectangle

    @property
    def connectors(self):
        """
        4-tuple of `.patches.ConnectionPatch` or None
            The four connector lines connecting to (lower_left, upper_left,
            lower_right upper_right) corners of *inset_ax*. Two lines are
            set with visibility to *False*,  but the user can set the
            visibility to True if the automatic choice is not deemed correct.
        """
        if self._inset_ax is None:
            return

        if self._auto_update_bounds:
            self._rectangle.set_bounds(self._bounds_from_inset_ax())
        self._update_connectors()
        return tuple(self._connectors)

    def draw(self, renderer):
        # docstring inherited
        conn_same_style = []

        # Figure out which connectors have the same style as the box, so should
        # be drawn as a single path.
        for conn in self.connectors or []:
            if conn.get_visible():
                drawn = False
                for s in _shared_properties:
                    if artist.getp(self._rectangle, s) != artist.getp(conn, s):
                        # Draw this connector by itself
                        conn.draw(renderer)
                        drawn = True
                        break

                if not drawn:
                    # Connector has same style as box.
                    conn_same_style.append(conn)

        if conn_same_style:
            # Since at least one connector has the same style as the rectangle, draw
            # them as a compound path.
            artists = [self._rectangle] + conn_same_style
            paths = [a.get_transform().transform_path(a.get_path()) for a in artists]
            path = Path.make_compound_path(*paths)

            # Create a temporary patch to draw the path.
            p = PathPatch(path)
            p.update_from(self._rectangle)
            p.set_transform(transforms.IdentityTransform())
            p.draw(renderer)

            return

        # Just draw the rectangle
        self._rectangle.draw(renderer)

    @_api.deprecated(
        '3.10',
        message=('Since Matplotlib 3.10 indicate_inset_[zoom] returns a single '
                 'InsetIndicator artist with a rectangle property and a connectors '
                 'property.  From 3.12 it will no longer be possible to unpack the '
                 'return value into two elements.'))
    def __getitem__(self, key):
        return [self._rectangle, self.connectors][key]
<|endoftext|>"""
2D lines with support for a variety of line styles, markers, colors, etc.
"""

import copy

from numbers import Integral, Number, Real
import logging

import numpy as np

import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
    _to_unmasked_float_array, ls_mapper, ls_mapper_r, STEP_LOOKUP_MAP)
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle

# Imported here for backward compatibility, even though they don't
# really belong.
from . import _path
from .markers import (  # noqa
    CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
    CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE,
    TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN)

_log = logging.getLogger(__name__)


def _get_dash_pattern(style):
    """Convert linestyle to dash pattern."""
    # go from short hand -> full strings
    if isinstance(style, str):
        style = ls_mapper.get(style, style)
    # un-dashed styles
    if style in ['solid', 'None']:
        offset = 0
        dashes = None
    # dashed styles
    elif style in ['dashed', 'dashdot', 'dotted']:
        offset = 0
        dashes = tuple(mpl.rcParams[f'lines.{style}_pattern'])
    #
    elif isinstance(style, tuple):
        offset, dashes = style
        if offset is None:
            raise ValueError(f'Unrecognized linestyle: {style!r}')
    else:
        raise ValueError(f'Unrecognized linestyle: {style!r}')

    # normalize offset to be positive and shorter than the dash cycle
    if dashes is not None:
        dsum = sum(dashes)
        if dsum:
            offset %= dsum

    return offset, dashes


def _get_dash_patterns(styles):
    """Convert linestyle or sequence of linestyles to list of dash patterns."""
    try:
        patterns = [_get_dash_pattern(styles)]
    except ValueError:
        try:
            patterns = [_get_dash_pattern(x) for x in styles]
        except ValueError as err:
            emsg = f'Do not know how to convert {styles!r} to dashes'
            raise ValueError(emsg) from err

    return patterns


def _get_inverse_dash_pattern(offset, dashes):
    """Return the inverse of the given dash pattern, for filling the gaps."""
    # Define the inverse pattern by moving the last gap to the start of the
    # sequence.
    gaps = dashes[-1:] + dashes[:-1]
    # Set the offset so that this new first segment is skipped
    # (see backend_bases.GraphicsContextBase.set_dashes for offset definition).
    offset_gaps = offset + dashes[-1]

    return offset_gaps, gaps


def _scale_dashes(offset, dashes, lw):
    if not mpl.rcParams['lines.scale_dashes']:
        return offset, dashes
    scaled_offset = offset * lw
    scaled_dashes = ([x * lw if x is not None else None for x in dashes]
                     if dashes is not None else None)
    return scaled_offset, scaled_dashes


def segment_hits(cx, cy, x, y, radius):
    """
    Return the indices of the segments in the polyline with coordinates (*cx*,
    *cy*) that are within a distance *radius* of the point (*x*, *y*).
    """
    # Process single points specially
    if len(x) <= 1:
        res, = np.nonzero((cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2)
        return res

    # We need to lop the last element off a lot.
    xr, yr = x[:-1], y[:-1]

    # Only look at line segments whose nearest point to C on the line
    # lies within the segment.
    dx, dy = x[1:] - xr, y[1:] - yr
    Lnorm_sq = dx ** 2 + dy ** 2  # Possibly want to eliminate Lnorm==0
    u = ((cx - xr) * dx + (cy - yr) * dy) / Lnorm_sq
    candidates = (u >= 0) & (u <= 1)

    # Note that there is a little area near one side of each point
    # which will be near neither segment, and another which will
    # be near both, depending on the angle of the lines.  The
    # following radius test eliminates these ambiguities.
    point_hits = (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2
    candidates = candidates & ~(point_hits[:-1] | point_hits[1:])

    # For those candidates which remain, determine how far they lie away
    # from the line.
    px, py = xr + u * dx, yr + u * dy
    line_hits = (cx - px) ** 2 + (cy - py) ** 2 <= radius ** 2
    line_hits = line_hits & candidates
    points, = point_hits.ravel().nonzero()
    lines, = line_hits.ravel().nonzero()
    return np.concatenate((points, lines))


def _mark_every_path(markevery, tpath, affine, ax):
    """
    Helper function that sorts out how to deal the input
    `markevery` and returns the points where markers should be drawn.

    Takes in the `markevery` value and the line path and returns the
    sub-sampled path.
    """
    # pull out the two bits of data we want from the path
    codes, verts = tpath.codes, tpath.vertices

    def _slice_or_none(in_v, slc):
        """Helper function to cope with `codes` being an ndarray or `None`."""
        if in_v is None:
            return None
        return in_v[slc]

    # if just an int, assume starting at 0 and make a tuple
    if isinstance(markevery, Integral):
        markevery = (0, markevery)
    # if just a float, assume starting at 0.0 and make a tuple
    elif isinstance(markevery, Real):
        markevery = (0.0, markevery)

    if isinstance(markevery, tuple):
        if len(markevery) != 2:
            raise ValueError('`markevery` is a tuple but its len is not 2; '
                             f'markevery={markevery}')
        start, step = markevery
        # if step is an int, old behavior
        if isinstance(step, Integral):
            # tuple of 2 int is for backwards compatibility,
            if not isinstance(start, Integral):
                raise ValueError(
                    '`markevery` is a tuple with len 2 and second element is '
                    'an int, but the first element is not an int; '
                    f'markevery={markevery}')
            # just return, we are done here

            return Path(verts[slice(start, None, step)],
                        _slice_or_none(codes, slice(start, None, step)))

        elif isinstance(step, Real):
            if not isinstance(start, Real):
                raise ValueError(
                    '`markevery` is a tuple with len 2 and second element is '
                    'a float, but the first element is not a float or an int; '
                    f'markevery={markevery}')
            if ax is None:
                raise ValueError(
                    "markevery is specified relative to the Axes size, but "
                    "the line does not have a Axes as parent")

            # calc cumulative distance along path (in display coords):
            fin = np.isfinite(verts).all(axis=1)
            fverts = verts[fin]
            disp_coords = affine.transform(fverts)

            delta = np.empty((len(disp_coords), 2))
            delta[0, :] = 0
            delta[1:, :] = disp_coords[1:, :] - disp_coords[:-1, :]
            delta = np.hypot(*delta.T).cumsum()
            # calc distance between markers along path based on the Axes
            # bounding box diagonal being a distance of unity:
            (x0, y0), (x1, y1) = ax.transAxes.transform([[0, 0], [1, 1]])
            scale = np.hypot(x1 - x0, y1 - y0)
            marker_delta = np.arange(start * scale, delta[-1], step * scale)
            # find closest actual data point that is closest to
            # the theoretical distance along the path:
            inds = np.abs(delta[np.newaxis, :] - marker_delta[:, np.newaxis])
            inds = inds.argmin(axis=1)
            inds = np.unique(inds)
            # return, we are done here
            return Path(fverts[inds], _slice_or_none(codes, inds))
        else:
            raise ValueError(
                f"markevery={markevery!r} is a tuple with len 2, but its "
                f"second element is not an int or a float")

    elif isinstance(markevery, slice):
        # mazol tov, it's already a slice, just return
        return Path(verts[markevery], _slice_or_none(codes, markevery))

    elif np.iterable(markevery):
        # fancy indexing
        try:
            return Path(verts[markevery], _slice_or_none(codes, markevery))
        except (ValueError, IndexError) as err:
            raise ValueError(
                f"markevery={markevery!r} is iterable but not a valid numpy "
                f"fancy index") from err
    else:
        raise ValueError(f"markevery={markevery!r} is not a recognized value")


@_docstring.interpd
@_api.define_aliases({
    "antialiased": ["aa"],
    "color": ["c"],
    "drawstyle": ["ds"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
    "markeredgecolor": ["mec"],
    "markeredgewidth": ["mew"],
    "markerfacecolor": ["mfc"],
    "markerfacecoloralt": ["mfcalt"],
    "markersize": ["ms"],
})
class Line2D(Artist):
    """
    A line - the line can have both a solid linestyle connecting all
    the vertices, and a marker at each vertex.  Additionally, the
    drawing of the solid line is influenced by the drawstyle, e.g., one
    can create "stepped" lines in various styles.
    """

    lineStyles = _lineStyles = {  # hidden names deprecated
        '-':    '_draw_solid',
        '--':   '_draw_dashed',
        '-.':   '_draw_dash_dot',
        ':':    '_draw_dotted',
        'None': '_draw_nothing',
        ' ':    '_draw_nothing',
        '':     '_draw_nothing',
    }

    _drawStyles_l = {
        'default':    '_draw_lines',
        'steps-mid':  '_draw_steps_mid',
        'steps-pre':  '_draw_steps_pre',
        'steps-post': '_draw_steps_post',
    }

    _drawStyles_s = {
        'steps': '_draw_steps_pre',
    }

    # drawStyles should now be deprecated.
    drawStyles = {**_drawStyles_l, **_drawStyles_s}
    # Need a list ordered with long names first:
    drawStyleKeys = [*_drawStyles_l, *_drawStyles_s]

    # Referenced here to maintain API.  These are defined in
    # MarkerStyle
    markers = MarkerStyle.markers
    filled_markers = MarkerStyle.filled_markers
    fillStyles = MarkerStyle.fillstyles

    zorder = 2

    _subslice_optim_min_size = 1000

    def __str__(self):
        if self._label != "":
            return f"Line2D({self._label})"
        elif self._x is None:
            return "Line2D()"
        elif len(self._x) > 3:
            return "Line2D(({:g},{:g}),({:g},{:g}),...,({:g},{:g}))".format(
                self._x[0], self._y[0],
                self._x[1], self._y[1],
                self._x[-1], self._y[-1])
        else:
            return "Line2D(%s)" % ",".join(
                map("({:g},{:g})".format, self._x, self._y))

    def __init__(self, xdata, ydata, *,
                 linewidth=None,  # all Nones default to rc
                 linestyle=None,
                 color=None,
                 gapcolor=None,
                 marker=None,
                 markersize=None,
                 markeredgewidth=None,
                 markeredgecolor=None,
                 markerfacecolor=None,
                 markerfacecoloralt='none',
                 fillstyle=None,
                 antialiased=None,
                 dash_capstyle=None,
                 solid_capstyle=None,
                 dash_joinstyle=None,
                 solid_joinstyle=None,
                 pickradius=5,
                 drawstyle=None,
                 markevery=None,
                 **kwargs
                 ):
        """
        Create a `.Line2D` instance with *x* and *y* data in sequences of
        *xdata*, *ydata*.

        Additional keyword arguments are `.Line2D` properties:

        %(Line2D:kwdoc)s

        See :meth:`set_linestyle` for a description of the line styles,
        :meth:`set_marker` for a description of the markers, and
        :meth:`set_drawstyle` for a description of the draw styles.

        """
        super().__init__()

        # Convert sequences to NumPy arrays.
        if not np.iterable(xdata):
            raise RuntimeError('xdata must be a sequence')
        if not np.iterable(ydata):
            raise RuntimeError('ydata must be a sequence')

        if linewidth is None:
            linewidth = mpl.rcParams['lines.linewidth']

        if linestyle is None:
            linestyle = mpl.rcParams['lines.linestyle']
        if marker is None:
            marker = mpl.rcParams['lines.marker']
        if color is None:
            color = mpl.rcParams['lines.color']

        if markersize is None:
            markersize = mpl.rcParams['lines.markersize']
        if antialiased is None:
            antialiased = mpl.rcParams['lines.antialiased']
        if dash_capstyle is None:
            dash_capstyle = mpl.rcParams['lines.dash_capstyle']
        if dash_joinstyle is None:
            dash_joinstyle = mpl.rcParams['lines.dash_joinstyle']
        if solid_capstyle is None:
            solid_capstyle = mpl.rcParams['lines.solid_capstyle']
        if solid_joinstyle is None:
            solid_joinstyle = mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None:
            drawstyle = 'default'

        self._dashcapstyle = None
        self._dashjoinstyle = None
        self._solidjoinstyle = None
        self._solidcapstyle = None
        self.set_dash_capstyle(dash_capstyle)
        self.set_dash_joinstyle(dash_joinstyle)
        self.set_solid_capstyle(solid_capstyle)
        self.set_solid_joinstyle(solid_joinstyle)

        self._linestyles = None
        self._drawstyle = None
        self._linewidth = linewidth
        self._unscaled_dash_pattern = (0, None)  # offset, dash
        self._dash_pattern = (0, None)  # offset, dash (scaled by linewidth)

        self.set_linewidth(linewidth)
        self.set_linestyle(linestyle)
        self.set_drawstyle(drawstyle)

        self._color = None
        self.set_color(color)
        if marker is None:
            marker = 'none'  # Default.
        if not isinstance(marker, MarkerStyle):
            self._marker = MarkerStyle(marker, fillstyle)
        else:
            self._marker = marker

        self._gapcolor = None
        self.set_gapcolor(gapcolor)

        self._markevery = None
        self._markersize = None
        self._antialiased = None

        self.set_markevery(markevery)
        self.set_antialiased(antialiased)
        self.set_markersize(markersize)

        self._markeredgecolor = None
        self._markeredgewidth = None
        self._markerfacecolor = None
        self._markerfacecoloralt = None

        self.set_markerfacecolor(markerfacecolor)  # Normalizes None to rc.
        self.set_markerfacecoloralt(markerfacecoloralt)
        self.set_markeredgecolor(markeredgecolor)  # Normalizes None to rc.
        self.set_markeredgewidth(markeredgewidth)

        # update kwargs before updating data to give the caller a
        # chance to init axes (and hence unit support)
        self._internal_update(kwargs)
        self.pickradius = pickradius
        self.ind_offset = 0
        if (isinstance(self._picker, Number) and
                not isinstance(self._picker, bool)):
            self._pickradius = self._picker

        self._xorig = np.asarray([])
        self._yorig = np.asarray([])
        self._invalidx = True
        self._invalidy = True
        self._x = None
        self._y = None
        self._xy = None
        self._path = None
        self._transformed_path = None
        self._subslice = False
        self._x_filled = None  # used in subslicing; only x is needed

        self.set_data(xdata, ydata)

    def contains(self, mouseevent):
        """
        Test whether *mouseevent* occurred on the line.

        An event is deemed to have occurred "on" the line if it is less
        than ``self.pickradius`` (default: 5 points) away from it.  Use
        `~.Line2D.get_pickradius` or `~.Line2D.set_pickradius` to get or set
        the pick radius.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`

        Returns
        -------
        contains : bool
            Whether any values are within the radius.
        details : dict
            A dictionary ``{'ind': pointlist}``, where *pointlist* is a
            list of points of the line that are within the pickradius around
            the event position.

            TODO: sort returned indices by distance
        """
        if self._different_canvas(mouseevent):
            return False, {}

        # Make sure we have data to plot
        if self._invalidy or self._invalidx:
            self.recache()
        if len(self._xy) == 0:
            return False, {}

        # Convert points to pixels
        transformed_path = self._get_transformed_path()
        path, affine = transformed_path.get_transformed_path_and_affine()
        path = affine.transform_path(path)
        xy = path.vertices
        xt = xy[:, 0]
        yt = xy[:, 1]

        # Convert pick radius from points to pixels
        fig = self.get_figure(root=True)
        if fig is None:
            _log.warning('no figure set when check if mouse is on line')
            pixels = self._pickradius
        else:
            pixels = fig.dpi / 72. * self._pickradius

        # The math involved in checking for containment (here and inside of
        # segment_hits) assumes that it is OK to overflow, so temporarily set
        # the error flags accordingly.
        with np.errstate(all='ignore'):
            # Check for collision
            if self._linestyle in ['None', None]:
                # If no line, return the nearby point(s)
                ind, = np.nonzero(
                    (xt - mouseevent.x) ** 2 + (yt - mouseevent.y) ** 2
                    <= pixels ** 2)
            else:
                # If line, return the nearby segment(s)
                ind = segment_hits(mouseevent.x, mouseevent.y, xt, yt, pixels)
                if self._drawstyle.startswith("steps"):
                    ind //= 2

        ind += self.ind_offset

        # Return the point(s) within radius
        return len(ind) > 0, dict(ind=ind)

    def get_pickradius(self):
        """
        Return the pick radius used for containment tests.

        See `.contains` for more details.
        """
        return self._pickradius

    def set_pickradius(self, pickradius):
        """
        Set the pick radius used for containment tests.

        See `.contains` for more details.

        Parameters
        ----------
        pickradius : float
            Pick radius, in points.
        """
        if not isinstance(pickradius, Real) or pickradius < 0:
            raise ValueError("pick radius should be a distance")
        self._pickradius = pickradius

    pickradius = property(get_pickradius, set_pickradius)

    def get_fillstyle(self):
        """
        Return the marker fill style.

        See also `~.Line2D.set_fillstyle`.
        """
        return self._marker.get_fillstyle()

    def set_fillstyle(self, fs):
        """
        Set the marker fill style.

        Parameters
        ----------
        fs : {'full', 'left', 'right', 'bottom', 'top', 'none'}
            Possible values:

            - 'full': Fill the whole marker with the *markerfacecolor*.
            - 'left', 'right', 'bottom', 'top': Fill the marker half at
              the given side with the *markerfacecolor*. The other
              half of the marker is filled with *markerfacecoloralt*.
            - 'none': No filling.

            For examples see :ref:`marker_fill_styles`.
        """
        self.set_marker(MarkerStyle(self._marker.get_marker(), fs))
        self.stale = True

    def set_markevery(self, every):
        """
        Set the markevery property to subsample the plot when using markers.

        e.g., if ``every=5``, every 5-th marker will be plotted.

        Parameters
        ----------
        every : None or int or (int, int) or slice or list[int] or float or \
(float, float) or list[bool]
            Which markers to plot.

            - ``every=None``: every point will be plotted.
            - ``every=N``: every N-th marker will be plotted starting with
              marker 0.
            - ``every=(start, N)``: every N-th marker, starting at index
              *start*, will be plotted.
            - ``every=slice(start, end, N)``: every N-th marker, starting at
              index *start*, up to but not including index *end*, will be
              plotted.
            - ``every=[i, j, m, ...]``: only markers at the given indices
              will be plotted.
            - ``every=[True, False, True, ...]``: only positions that are True
              will be plotted. The list must have the same length as the data
              points.
            - ``every=0.1``, (i.e. a float): markers will be spaced at
              approximately equal visual distances along the line; the distance
              along the line between markers is determined by multiplying the
              display-coordinate distance of the Axes bounding-box diagonal
              by the value of *every*.
            - ``every=(0.5, 0.1)`` (i.e. a length-2 tuple of float): similar
              to ``every=0.1`` but the first marker will be offset along the
              line by 0.5 multiplied by the
              display-coordinate-diagonal-distance along the line.

            For examples see
            :doc:`/gallery/lines_bars_and_markers/markevery_demo`.

        Notes
        -----
        Setting *markevery* will still only draw markers at actual data points.
        While the float argument form aims for uniform visual spacing, it has
        to coerce from the ideal spacing to the nearest available data point.
        Depending on the number and distribution of data points, the result
        may still not look evenly spaced.

        When using a start offset to specify the first marker, the offset will
        be from the first data point which may be different from the first
        the visible data point if the plot is zoomed in.

        If zooming in on a plot when using float arguments then the actual
        data points that have markers will change because the distance between
        markers is always determined from the display-coordinates
        axes-bounding-box-diagonal regardless of the actual axes data limits.

        """
        self._markevery = every
        self.stale = True

    def get_markevery(self):
        """
        Return the markevery setting for marker subsampling.

        See also `~.Line2D.set_markevery`.
        """
        return self._markevery

    def set_picker(self, p):
        """
        Set the event picker details for the line.

        Parameters
        ----------
        p : float or callable[[Artist, Event], tuple[bool, dict]]
            If a float, it is used as the pick radius in points.
        """
        if not callable(p):
            self.set_pickradius(p)
        self._picker = p

    def get_bbox(self):
        """Get the bounding box of this line."""
        bbox = Bbox([[0, 0], [0, 0]])
        bbox.update_from_data_xy(self.get_xydata())
        return bbox

    def get_window_extent(self, renderer=None):
        bbox = Bbox([[0, 0], [0, 0]])
        trans_data_to_xy = self.get_transform().transform
        bbox.update_from_data_xy(trans_data_to_xy(self.get_xydata()),
                                 ignore=True)
        # correct for marker size, if any
        if self._marker:
            ms = (self._markersize / 72.0 * self.get_figure(root=True).dpi) * 0.5
            bbox = bbox.padded(ms)
        return bbox

    def set_data(self, *args):
        """
        Set the x and y data.

        Parameters
        ----------
        *args : (2, N) array or two 1D arrays

        See Also
        --------
        set_xdata
        set_ydata
        """
        if len(args) == 1:
            (x, y), = args
        else:
            x, y = args

        self.set_xdata(x)
        self.set_ydata(y)

    def recache_always(self):
        self.recache(always=True)

    def recache(self, always=False):
        if always or self._invalidx:
            xconv = self.convert_xunits(self._xorig)
            x = _to_unmasked_float_array(xconv).ravel()
        else:
            x = self._x
        if always or self._invalidy:
            yconv = self.convert_yunits(self._yorig)
            y = _to_unmasked_float_array(yconv).ravel()
        else:
            y = self._y

        self._xy = np.column_stack(np.broadcast_arrays(x, y)).astype(float)
        self._x, self._y = self._xy.T  # views

        self._subslice = False
        if (self.axes
                and len(x) > self._subslice_optim_min_size
                and _path.is_sorted_and_has_non_nan(x)
                and self.axes.name == 'rectilinear'
                and self.axes.get_xscale() == 'linear'
                and self._markevery is None
                and self.get_clip_on()
                and self.get_transform() == self.axes.transData):
            self._subslice = True
            nanmask = np.isnan(x)
            if nanmask.any():
                self._x_filled = self._x.copy()
                indices = np.arange(len(x))
                self._x_filled[nanmask] = np.interp(
                    indices[nanmask], indices[~nanmask], self._x[~nanmask])
            else:
                self._x_filled = self._x

        if self._path is not None:
            interpolation_steps = self._path._interpolation_steps
        else:
            interpolation_steps = 1
        xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy.T)
        self._path = Path(np.asarray(xy).T,
                          _interpolation_steps=interpolation_steps)
        self._transformed_path = None
        self._invalidx = False
        self._invalidy = False

    def _transform_path(self, subslice=None):
        """
        Put a TransformedPath instance at self._transformed_path;
        all invalidation of the transform is then handled by the
        TransformedPath instance.
        """
        # Masked arrays are now handled by the Path class itself
        if subslice is not None:
            xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy[subslice, :].T)
            _path = Path(np.asarray(xy).T,
                         _interpolation_steps=self._path._interpolation_steps)
        else:
            _path = self._path
        self._transformed_path = TransformedPath(_path, self.get_transform())

    def _get_transformed_path(self):
        """Return this line's `~matplotlib.transforms.TransformedPath`."""
        if self._transformed_path is None:
            self._transform_path()
        return self._transformed_path

    def set_transform(self, t):
        # docstring inherited
        self._invalidx = True
        self._invalidy = True
        super().set_transform(t)

    @allow_rasterization
    def draw(self, renderer):
        # docstring inherited

        if not self.get_visible():
            return

        if self._invalidy or self._invalidx:
            self.recache()
        self.ind_offset = 0  # Needed for contains() method.
        if self._subslice and self.axes:
            x0, x1 = self.axes.get_xbound()
            i0 = self._x_filled.searchsorted(x0, 'left')
            i1 = self._x_filled.searchsorted(x1, 'right')
            subslice = slice(max(i0 - 1, 0), i1 + 1)
            self.ind_offset = subslice.start
            self._transform_path(subslice)
        else:
            subslice = None

        if self.get_path_effects():
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        renderer.open_group('line2d', self.get_gid())
        if self._lineStyles[self._linestyle] != '_draw_nothing':
            tpath, affine = (self._get_transformed_path()
                             .get_transformed_path_and_affine())
            if len(tpath.vertices):
                gc = renderer.new_gc()
                self._set_gc_clip(gc)
                gc.set_url(self.get_url())

                gc.set_antialiased(self._antialiased)
                gc.set_linewidth(self._linewidth)

                if self.is_dashed():
                    cap = self._dashcapstyle
                    join = self._dashjoinstyle
                else:
                    cap = self._solidcapstyle
                    join = self._solidjoinstyle
                gc.set_joinstyle(join)
                gc.set_capstyle(cap)
                gc.set_snap(self.get_snap())
                if self.get_sketch_params() is not None:
                    gc.set_sketch_params(*self.get_sketch_params())

                # We first draw a path within the gaps if needed.
                if self.is_dashed() and self._gapcolor is not None:
                    lc_rgba = mcolors.to_rgba(self._gapcolor, self._alpha)
                    gc.set_foreground(lc_rgba, isRGBA=True)

                    offset_gaps, gaps = _get_inverse_dash_pattern(
                        *self._dash_pattern)

                    gc.set_dashes(offset_gaps, gaps)
                    renderer.draw_path(gc, tpath, affine.frozen())

                lc_rgba = mcolors.to_rgba(self._color, self._alpha)
                gc.set_foreground(lc_rgba, isRGBA=True)

                gc.set_dashes(*self._dash_pattern)
                renderer.draw_path(gc, tpath, affine.frozen())
                gc.restore()

        if self._marker and self._markersize > 0:
            gc = renderer.new_gc()
            self._set_gc_clip(gc)
            gc.set_url(self.get_url())
            gc.set_linewidth(self._markeredgewidth)
            gc.set_antialiased(self._antialiased)

            ec_rgba = mcolors.to_rgba(
                self.get_markeredgecolor(), self._alpha)
            fc_rgba = mcolors.to_rgba(
                self._get_markerfacecolor(), self._alpha)
            fcalt_rgba = mcolors.to_rgba(
                self._get_markerfacecolor(alt=True), self._alpha)
            # If the edgecolor is "auto", it is set according to the *line*
            # color but inherits the alpha value of the *face* color, if any.
            if (cbook._str_equal(self._markeredgecolor, "auto")
                    and not cbook._str_lower_equal(
                        self.get_markerfacecolor(), "none")):
                ec_rgba = ec_rgba[:3] + (fc_rgba[3],)
            gc.set_foreground(ec_rgba, isRGBA=True)
            if self.get_sketch_params() is not None:
                scale, length, randomness = self.get_sketch_params()
                gc.set_sketch_params(scale/2, length/2, 2*randomness)

            marker = self._marker

            # Markers *must* be drawn ignoring the drawstyle (but don't pay the
            # recaching if drawstyle is already "default").
            if self.get_drawstyle() != "default":
                with cbook._setattr_cm(
                        self, _drawstyle="default", _transformed_path=None):
                    self.recache()
                    self._transform_path(subslice)
                    tpath, affine = (self._get_transformed_path()
                                     .get_transformed_points_and_affine())
            else:
                tpath, affine = (self._get_transformed_path()
                                 .get_transformed_points_and_affine())

            if len(tpath.vertices):
                # subsample the markers if markevery is not None
                markevery = self.get_markevery()
                if markevery is not None:
                    subsampled = _mark_every_path(
                        markevery, tpath, affine, self.axes)
                else:
                    subsampled = tpath

                snap = marker.get_snap_threshold()
                if isinstance(snap, Real):
                    snap = renderer.points_to_pixels(self._markersize) >= snap
                gc.set_snap(snap)
                gc.set_joinstyle(marker.get_joinstyle())
                gc.set_capstyle(marker.get_capstyle())
                marker_path = marker.get_path()
                marker_trans = marker.get_transform()
                w = renderer.points_to_pixels(self._markersize)

                if cbook._str_equal(marker.get_marker(), ","):
                    gc.set_linewidth(0)
                else:
                    # Don't scale for pixels, and don't stroke them
                    marker_trans = marker_trans.scale(w)
                renderer.draw_markers(gc, marker_path, marker_trans,
                                      subsampled, affine.frozen(),
                                      fc_rgba)

                alt_marker_path = marker.get_alt_path()
                if alt_marker_path:
                    alt_marker_trans = marker.get_alt_transform()
                    alt_marker_trans = alt_marker_trans.scale(w)
                    renderer.draw_markers(
                            gc, alt_marker_path, alt_marker_trans, subsampled,
                            affine.frozen(), fcalt_rgba)

            gc.restore()

        renderer.close_group('line2d')
        self.stale = False

    def get_antialiased(self):
        """Return whether antialiased rendering is used."""
        return self._antialiased

    def get_color(self):
        """
        Return the line color.

        See also `~.Line2D.set_color`.
        """
        return self._color

    def get_drawstyle(self):
        """
        Return the drawstyle.

        See also `~.Line2D.set_drawstyle`.
        """
        return self._drawstyle

    def get_gapcolor(self):
        """
        Return the line gapcolor.

        See also `~.Line2D.set_gapcolor`.
        """
        return self._gapcolor

    def get_linestyle(self):
        """
        Return the linestyle.

        See also `~.Line2D.set_linestyle`.
        """
        return self._linestyle

    def get_linewidth(self):
        """
        Return the linewidth in points.

        See also `~.Line2D.set_linewidth`.
        """
        return self._linewidth

    def get_marker(self):
        """
        Return the line marker.

        See also `~.Line2D.set_marker`.
        """
        return self._marker.get_marker()

    def get_markeredgecolor(self):
        """
        Return the marker edge color.

        See also `~.Line2D.set_markeredgecolor`.
        """
        mec = self._markeredgecolor
        if cbook._str_equal(mec, 'auto'):
            if mpl.rcParams['_internal.classic_mode']:
                if self._marker.get_marker() in ('.', ','):
                    return self._color
                if (self._marker.is_filled()
                        and self._marker.get_fillstyle() != 'none'):
                    return 'k'  # Bad hard-wired default...
            return self._color
        else:
            return mec

    def get_markeredgewidth(self):
        """
        Return the marker edge width in points.

        See also `~.Line2D.set_markeredgewidth`.
        """
        return self._markeredgewidth

    def _get_markerfacecolor(self, alt=False):
        if self._marker.get_fillstyle() == 'none':
            return 'none'
        fc = self._markerfacecoloralt if alt else self._markerfacecolor
        if cbook._str_lower_equal(fc, 'auto'):
            return self._color
        else:
            return fc

    def get_markerfacecolor(self):
        """
        Return the marker face color.

        See also `~.Line2D.set_markerfacecolor`.
        """
        return self._get_markerfacecolor(alt=False)

    def get_markerfacecoloralt(self):
        """
        Return the alternate marker face color.

        See also `~.Line2D.set_markerfacecoloralt`.
        """
        return self._get_markerfacecolor(alt=True)

    def get_markersize(self):
        """
        Return the marker size in points.

        See also `~.Line2D.set_markersize`.
        """
        return self._markersize

    def get_data(self, orig=True):
        """
        Return the line data as an ``(xdata, ydata)`` pair.

        If *orig* is *True*, return the original data.
        """
        return self.get_xdata(orig=orig), self.get_ydata(orig=orig)

    def get_xdata(self, orig=True):
        """
        Return the xdata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
        if orig:
            return self._xorig
        if self._invalidx:
            self.recache()
        return self._x

    def get_ydata(self, orig=True):
        """
        Return the ydata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
        if orig:
            return self._yorig
        if self._invalidy:
            self.recache()
        return self._y

    def get_path(self):
        """Return the `~matplotlib.path.Path` associated with this line."""
        if self._invalidy or self._invalidx:
            self.recache()
        return self._path

    def get_xydata(self):
        """Return the *xy* data as a (N, 2) array."""
        if self._invalidy or self._invalidx:
            self.recache()
        return self._xy

    def set_antialiased(self, b):
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        b : bool
        """
        if self._antialiased != b:
            self.stale = True
        self._antialiased = b

    def set_color(self, color):
        """
        Set the color of the line.

        Parameters
        ----------
        color : :mpltype:`color`
        """
        mcolors._check_color_like(color=color)
        self._color = color
        self.stale = True

    def set_drawstyle(self, drawstyle):
        """
        Set the drawstyle of the plot.

        The drawstyle determines how the points are connected.

        Parameters
        ----------
        drawstyle : {'default', 'steps', 'steps-pre', 'steps-mid', \
'steps-post'}, default: 'default'
            For 'default', the points are connected with straight lines.

            The steps variants connect the points with step-like lines,
            i.e. horizontal lines with vertical steps. They differ in the
            location of the step:

            - 'steps-pre': The step is at the beginning of the line segment,
              i.e. the line will be at the y-value of point to the right.
            - 'steps-mid': The step is halfway between the points.
            - 'steps-post: The step is at the end of the line segment,
              i.e. the line will be at the y-value of the point to the left.
            - 'steps' is equal to 'steps-pre' and is maintained for
              backward-compatibility.

            For examples see :doc:`/gallery/lines_bars_and_markers/step_demo`.
        """
        if drawstyle is None:
            drawstyle = 'default'
        _api.check_in_list(self.drawStyles, drawstyle=drawstyle)
        if self._drawstyle != drawstyle:
            self.stale = True
            # invalidate to trigger a recache of the path
            self._invalidx = True
        self._drawstyle = drawstyle

    def set_gapcolor(self, gapcolor):
        """
        Set a color to fill the gaps in the dashed line style.

        .. note::

            Striped lines are created by drawing two interleaved dashed lines.
            There can be overlaps between those two, which may result in
            artifacts when using transparency.

            This functionality is experimental and may change.

        Parameters
        ----------
        gapcolor : :mpltype:`color` or None
            The color with which to fill the gaps. If None, the gaps are
            unfilled.
        """
        if gapcolor is not None:
            mcolors._check_color_like(color=gapcolor)
        self._gapcolor = gapcolor
        self.stale = True

    def set_linewidth(self, w):
        """
        Set the line width in points.

        Parameters
        ----------
        w : float
            Line width, in points.
        """
        w = float(w)
        if self._linewidth != w:
            self.stale = True
        self._linewidth = w
        self._dash_pattern = _scale_dashes(*self._unscaled_dash_pattern, w)

    def set_linestyle(self, ls):
        """
        Set the linestyle of the line.

        Parameters
        ----------
        ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            Possible values:

            - A string:

              ==========================================  =================
              linestyle                                   description
              ==========================================  =================
              ``'-'`` or ``'solid'``                      solid line
              ``'--'`` or  ``'dashed'``                   dashed line
              ``'-.'`` or  ``'dashdot'``                  dash-dotted line
              ``':'`` or ``'dotted'``                     dotted line
              ``'none'``, ``'None'``, ``' '``, or ``''``  draw nothing
              ==========================================  =================

            - Alternatively a dash tuple of the following form can be
              provided::

                  (offset, onoffseq)

              where ``onoffseq`` is an even length tuple of on and off ink
              in points. See also :meth:`set_dashes`.

            For examples see :doc:`/gallery/lines_bars_and_markers/linestyles`.
        """
        if isinstance(ls, str):
            if ls in [' ', '', 'none']:
                ls = 'None'
            _api.check_in_list([*self._lineStyles, *ls_mapper_r], ls=ls)
            if ls not in self._lineStyles:
                ls = ls_mapper_r[ls]
            self._linestyle = ls
        else:
            self._linestyle = '--'
        self._unscaled_dash_pattern = _get_dash_pattern(ls)
        self._dash_pattern = _scale_dashes(
            *self._unscaled_dash_pattern, self._linewidth)
        self.stale = True

    @_docstring.interpd
    def set_marker(self, marker):
        """
        Set the line marker.

        Parameters
        ----------
        marker : marker style string, `~.path.Path` or `~.markers.MarkerStyle`
            See `~matplotlib.markers` for full description of possible
            arguments.
        """
        self._marker = MarkerStyle(marker, self._marker.get_fillstyle())
        self.stale = True

    def _set_markercolor(self, name, has_rcdefault, val):
        if val is None:
            val = mpl.rcParams[f"lines.{name}"] if has_rcdefault else "auto"
        attr = f"_{name}"
        current = getattr(self, attr)
        if current is None:
            self.stale = True
        else:
            neq = current != val
            # Much faster than `np.any(current != val)` if no arrays are used.
            if neq.any() if isinstance(neq, np.ndarray) else neq:
                self.stale = True
        setattr(self, attr, val)

    def set_markeredgecolor(self, ec):
        """
        Set the marker edge color.

        Parameters
        ----------
        ec : :mpltype:`color`
        """
        self._set_markercolor("markeredgecolor", True, ec)

    def set_markerfacecolor(self, fc):
        """
        Set the marker face color.

        Parameters
        ----------
        fc : :mpltype:`color`
        """
        self._set_markercolor("markerfacecolor", True, fc)

    def set_markerfacecoloralt(self, fc):
        """
        Set the alternate marker face color.

        Parameters
        ----------
        fc : :mpltype:`color`
        """
        self._set_markercolor("markerfacecoloralt", False, fc)

    def set_markeredgewidth(self, ew):
        """
        Set the marker edge width in points.

        Parameters
        ----------
        ew : float
             Marker edge width, in points.
        """
        if ew is None:
            ew = mpl.rcParams['lines.markeredgewidth']
        if self._markeredgewidth != ew:
            self.stale = True
        self._markeredgewidth = ew

    def set_markersize(self, sz):
        """
        Set the marker size in points.

        Parameters
        ----------
        sz : float
             Marker size, in points.
        """
        sz = float(sz)
        if self._markersize != sz:
            self.stale = True
        self._markersize = sz

    def set_xdata(self, x):
        """
        Set the data array for x.

        Parameters
        ----------
        x : 1D array

        See Also
        --------
        set_data
        set_ydata
        """
        if not np.iterable(x):
            raise RuntimeError('x must be a sequence')
        self._xorig = copy.copy(x)
        self._invalidx = True
        self.stale = True

    def set_ydata(self, y):
        """
        Set the data array for y.

        Parameters
        ----------
        y : 1D array

        See Also
        --------
        set_data
        set_xdata
        """
        if not np.iterable(y):
            raise RuntimeError('y must be a sequence')
        self._yorig = copy.copy(y)
        self._invalidy = True
        self.stale = True

    def set_dashes(self, seq):
        """
        Set the dash sequence.

        The dash sequence is a sequence of floats of even length describing
        the length of dashes and spaces in points.

        For example, (5, 2, 1, 2) describes a sequence of 5 point and 1 point
        dashes separated by 2 point spaces.

        See also `~.Line2D.set_gapcolor`, which allows those spaces to be
        filled with a color.

        Parameters
        ----------
        seq : sequence of floats (on/off ink in points) or (None, None)
            If *seq* is empty or ``(None, None)``, the linestyle will be set
            to solid.
        """
        if seq == (None, None) or len(seq) == 0:
            self.set_linestyle('-')
        else:
            self.set_linestyle((0, seq))

    def update_from(self, other):
        """Copy properties from *other* to self."""
        super().update_from(other)
        self._linestyle = other._linestyle
        self._linewidth = other._linewidth
        self._color = other._color
        self._gapcolor = other._gapcolor
        self._markersize = other._markersize
        self._markerfacecolor = other._markerfacecolor
        self._markerfacecoloralt = other._markerfacecoloralt
        self._markeredgecolor = other._markeredgecolor
        self._markeredgewidth = other._markeredgewidth
        self._unscaled_dash_pattern = other._unscaled_dash_pattern
        self._dash_pattern = other._dash_pattern
        self._dashcapstyle = other._dashcapstyle
        self._dashjoinstyle = other._dashjoinstyle
        self._solidcapstyle = other._solidcapstyle
        self._solidjoinstyle = other._solidjoinstyle

        self._linestyle = other._linestyle
        self._marker = MarkerStyle(marker=other._marker)
        self._drawstyle = other._drawstyle

    @_docstring.interpd
    def set_dash_joinstyle(self, s):
        """
        How to join segments of the line if it `~Line2D.is_dashed`.

        The default joinstyle is :rc:`lines.dash_joinstyle`.

        Parameters
        ----------
        s : `.JoinStyle` or %(JoinStyle)s
        """
        js = JoinStyle(s)
        if self._dashjoinstyle != js:
            self.stale = True
        self._dashjoinstyle = js

    @_docstring.interpd
    def set_solid_joinstyle(self, s):
        """
        How to join segments if the line is solid (not `~Line2D.is_dashed`).

        The default joinstyle is :rc:`lines.solid_joinstyle`.

        Parameters
        ----------
        s : `.JoinStyle` or %(JoinStyle)s
        """
        js = JoinStyle(s)
        if self._solidjoinstyle != js:
            self.stale = True
        self._solidjoinstyle = js

    def get_dash_joinstyle(self):
        """
        Return the `.JoinStyle` for dashed lines.

        See also `~.Line2D.set_dash_joinstyle`.
        """
        return self._dashjoinstyle.name

    def get_solid_joinstyle(self):
        """
        Return the `.JoinStyle` for solid lines.

        See also `~.Line2D.set_solid_joinstyle`.
        """
        return self._solidjoinstyle.name

    @_docstring.interpd
    def set_dash_capstyle(self, s):
        """
        How to draw the end caps if the line is `~Line2D.is_dashed`.

        The default capstyle is :rc:`lines.dash_capstyle`.

        Parameters
        ----------
        s : `.CapStyle` or %(CapStyle)s
        """
        cs = CapStyle(s)
        if self._dashcapstyle != cs:
            self.stale = True
        self._dashcapstyle = cs

    @_docstring.interpd
    def set_solid_capstyle(self, s):
        """
        How to draw the end caps if the line is solid (not `~Line2D.is_dashed`)

        The default capstyle is :rc:`lines.solid_capstyle`.

        Parameters
        ----------
        s : `.CapStyle` or %(CapStyle)s
        """
        cs = CapStyle(s)
        if self._solidcapstyle != cs:
            self.stale = True
        self._solidcapstyle = cs

    def get_dash_capstyle(self):
        """
        Return the `.CapStyle` for dashed lines.

        See also `~.Line2D.set_dash_capstyle`.
        """
        return self._dashcapstyle.name

    def get_solid_capstyle(self):
        """
        Return the `.CapStyle` for solid lines.

        See also `~.Line2D.set_solid_capstyle`.
        """
        return self._solidcapstyle.name

    def is_dashed(self):
        """
        Return whether line has a dashed linestyle.

        A custom linestyle is assumed to be dashed, we do not inspect the
        ``onoffseq`` directly.

        See also `~.Line2D.set_linestyle`.
        """
        return self._linestyle in ('--', '-.', ':')


class AxLine(Line2D):
    """
    A helper class that implements `~.Axes.axline`, by recomputing the artist
    transform at draw time.
    """

    def __init__(self, xy1, xy2, slope, **kwargs):
        """
        Parameters
        ----------
        xy1 : (float, float)
            The first set of (x, y) coordinates for the line to pass through.
        xy2 : (float, float) or None
            The second set of (x, y) coordinates for the line to pass through.
            Both *xy2* and *slope* must be passed, but one of them must be None.
        slope : float or None
            The slope of the line. Both *xy2* and *slope* must be passed, but one of
            them must be None.
        """
        super().__init__([0, 1], [0, 1], **kwargs)

        if (xy2 is None and slope is None or
                xy2 is not None and slope is not None):
            raise TypeError(
                "Exactly one of 'xy2' and 'slope' must be given")

        self._slope = slope
        self._xy1 = xy1
        self._xy2 = xy2

    def get_transform(self):
        ax = self.axes
        points_transform = self._transform - ax.transData + ax.transScale

        if self._xy2 is not None:
            # two points were given
            (x1, y1), (x2, y2) = \
                points_transform.transform([self._xy1, self._xy2])
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                if dy == 0:
                    raise ValueError(
                        f"Cannot draw a line through two identical points "
                        f"(x={(x1, x2)}, y={(y1, y2)})")
                slope = np.inf
            else:
                slope = dy / dx
        else:
            # one point and a slope were given
            x1, y1 = points_transform.transform(self._xy1)
            slope = self._slope
        (vxlo, vylo), (vxhi, vyhi) = ax.transScale.transform(ax.viewLim)
        # General case: find intersections with view limits in either
        # direction, and draw between the middle two points.
        if slope == 0:
            start = vxlo, y1
            stop = vxhi, y1
        elif np.isinf(slope):
            start = x1, vylo
            stop = x1, vyhi
        else:
            _, start, stop, _ = sorted([
                (vxlo, y1 + (vxlo - x1) * slope),
                (vxhi, y1 + (vxhi - x1) * slope),
                (x1 + (vylo - y1) / slope, vylo),
                (x1 + (vyhi - y1) / slope, vyhi),
            ])
        return (BboxTransformTo(Bbox([start, stop]))
                + ax.transLimits + ax.transAxes)

    def draw(self, renderer):
        self._transformed_path = None  # Force regen.
        super().draw(renderer)

    def get_xy1(self):
        """Return the *xy1* value of the line."""
        return self._xy1

    def get_xy2(self):
        """Return the *xy2* value of the line."""
        return self._xy2

    def get_slope(self):
        """Return the *slope* value of the line."""
        return self._slope

    def set_xy1(self, *args, **kwargs):
        """
        Set the *xy1* value of the line.

        Parameters
        ----------
        xy1 : tuple[float, float]
            Points for the line to pass through.
        """
        params = _api.select_matching_signature([
            lambda self, x, y: locals(), lambda self, xy1: locals(),
        ], self, *args, **kwargs)
        if "x" in params:
            _api.warn_deprecated("3.10", message=(
                "Passing x and y separately to AxLine.set_xy1 is deprecated since "
                "%(since)s; pass them as a single tuple instead."))
            xy1 = params["x"], params["y"]
        else:
            xy1 = params["xy1"]
        self._xy1 = xy1

    def set_xy2(self, *args, **kwargs):
        """
        Set the *xy2* value of the line.

        .. note::

            You can only set *xy2* if the line was created using the *xy2*
            parameter. If the line was created using *slope*, please use
            `~.AxLine.set_slope`.

        Parameters
        ----------
        xy2 : tuple[float, float]
            Points for the line to pass through.
        """
        if self._slope is None:
            params = _api.select_matching_signature([
                lambda self, x, y: locals(), lambda self, xy2: locals(),
            ], self, *args, **kwargs)
            if "x" in params:
                _api.warn_deprecated("3.10", message=(
                    "Passing x and y separately to AxLine.set_xy2 is deprecated since "
                    "%(since)s; pass them as a single tuple instead."))
                xy2 = params["x"], params["y"]
            else:
                xy2 = params["xy2"]
            self._xy2 = xy2
        else:
            raise ValueError("Cannot set an 'xy2' value while 'slope' is set;"
                             " they differ but their functionalities overlap")

    def set_slope(self, slope):
        """
        Set the *slope* value of the line.

        .. note::

            You can only set *slope* if the line was created using the *slope*
            parameter. If the line was created using *xy2*, please use
            `~.AxLine.set_xy2`.

        Parameters
        ----------
        slope : float
            The slope of the line.
        """
        if self._xy2 is None:
            self._slope = slope
        else:
            raise ValueError("Cannot set a 'slope' value while 'xy2' is set;"
                             " they differ but their functionalities overlap")


class VertexSelector:
    """
    Manage the callbacks to maintain a list of selected vertices for `.Line2D`.
    Derived classes should override the `process_selected` method to do
    something with the picks.

    Here is an example which highlights the selected verts with red circles::

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.lines as lines

        class HighlightSelected(lines.VertexSelector):
            def __init__(self, line, fmt='ro', **kwargs):
                super().__init__(line)
                self.markers, = self.axes.plot([], [], fmt, **kwargs)

            def process_selected(self, ind, xs, ys):
                self.markers.set_data(xs, ys)
                self.canvas.draw()

        fig, ax = plt.subplots()
        x, y = np.random.rand(2, 30)
        line, = ax.plot(x, y, 'bs-', picker=5)

        selector = HighlightSelected(line)
        plt.show()
    """

    def __init__(self, line):
        """
        Parameters
        ----------
        line : `~matplotlib.lines.Line2D`
            The line must already have been added to an `~.axes.Axes` and must
            have its picker property set.
        """
        if line.axes is None:
            raise RuntimeError('You must first add the line to the Axes')
        if line.get_picker() is None:
            raise RuntimeError('You must first set the picker property '
                               'of the line')
        self.axes = line.axes
        self.line = line
        self.cid = self.canvas.callbacks._connect_picklable(
            'pick_event', self.onpick)
        self.ind = set()

    canvas = property(lambda self: self.axes.get_figure(root=True).canvas)

    def process_selected(self, ind, xs, ys):
        """
        Default "do nothing" implementation of the `process_selected` method.

        Parameters
        ----------
        ind : list of int
            The indices of the selected vertices.
        xs, ys : array-like
            The coordinates of the selected vertices.
        """
        pass

    def onpick(self, event):
        """When the line is picked, update the set of selected indices."""
        if event.artist is not self.line:
            return
        self.ind ^= set(event.ind)
        ind = sorted(self.ind)
        xdata, ydata = self.line.get_data()
        self.process_selected(ind, xdata[ind], ydata[ind])


lineStyles = Line2D._lineStyles
lineMarkers = MarkerStyle.markers
drawStyles = Line2D.drawStyles
fillStyles = MarkerStyle.fillstyles
<|endoftext|>"""
Matplotlib includes a framework for arbitrary geometric transformations that is used to
determine the final position of all elements drawn on the canvas.

Transforms are composed into trees of `TransformNode` objects
whose actual value depends on their children.  When the contents of
children change, their parents are automatically invalidated.  The
next time an invalidated transform is accessed, it is recomputed to
reflect those changes.  This invalidation/caching approach prevents
unnecessary recomputations of transforms, and contributes to better
interactive performance.

For example, here is a graph of the transform tree used to plot data to the figure:

.. graphviz:: /api/transforms.dot
    :alt: Diagram of transform tree from data to figure coordinates.

The framework can be used for both affine and non-affine
transformations.  However, for speed, we want to use the backend
renderers to perform affine transformations whenever possible.
Therefore, it is possible to perform just the affine or non-affine
part of a transformation on a set of data.  The affine is always
assumed to occur after the non-affine.  For any transform::

  full transform == non-affine part + affine part

The backends are not expected to handle non-affine transformations
themselves.

See the tutorial :ref:`transforms_tutorial` for examples
of how to use transforms.
"""

# Note: There are a number of places in the code where we use `np.min` or
# `np.minimum` instead of the builtin `min`, and likewise for `max`.  This is
# done so that `nan`s are propagated, instead of being silently dropped.

import functools
import itertools
import textwrap
import weakref
import math

import numpy as np
from numpy.linalg import inv

from matplotlib import _api
from matplotlib._path import (
    affine_transform, count_bboxes_overlapping_bbox, update_path_extents)
from .path import Path

DEBUG = False


def _make_str_method(*args, **kwargs):
    """
    Generate a ``__str__`` method for a `.Transform` subclass.

    After ::

        class T:
            __str__ = _make_str_method("attr", key="other")

    ``str(T(...))`` will be

    .. code-block:: text

        {type(T).__name__}(
            {self.attr},
            key={self.other})
    """
    indent = functools.partial(textwrap.indent, prefix=" " * 4)
    def strrepr(x): return repr(x) if isinstance(x, str) else str(x)
    return lambda self: (
        type(self).__name__ + "("
        + ",".join([*(indent("\n" + strrepr(getattr(self, arg)))
                      for arg in args),
                    *(indent("\n" + k + "=" + strrepr(getattr(self, arg)))
                      for k, arg in kwargs.items())])
        + ")")


class TransformNode:
    """
    The base class for anything that participates in the transform tree
    and needs to invalidate its parents or be invalidated.  This includes
    classes that are not really transforms, such as bounding boxes, since some
    transforms depend on bounding boxes to compute their values.
    """

    # Invalidation may affect only the affine part.  If the
    # invalidation was "affine-only", the _invalid member is set to
    # INVALID_AFFINE_ONLY

    # Possible values for the _invalid attribute.
    _VALID, _INVALID_AFFINE_ONLY, _INVALID_FULL = range(3)

    # Some metadata about the transform, used to determine whether an
    # invalidation is affine-only
    is_affine = False
    is_bbox = _api.deprecated("3.9")(_api.classproperty(lambda cls: False))

    pass_through = False
    """
    If pass_through is True, all ancestors will always be
    invalidated, even if 'self' is already invalid.
    """

    def __init__(self, shorthand_name=None):
        """
        Parameters
        ----------
        shorthand_name : str
            A string representing the "name" of the transform. The name carries
            no significance other than to improve the readability of
            ``str(transform)`` when DEBUG=True.
        """
        self._parents = {}
        # Initially invalid, until first computation.
        self._invalid = self._INVALID_FULL
        self._shorthand_name = shorthand_name or ''

    if DEBUG:
        def __str__(self):
            # either just return the name of this TransformNode, or its repr
            return self._shorthand_name or repr(self)

    def __getstate__(self):
        # turn the dictionary with weak values into a normal dictionary
        return {**self.__dict__,
                '_parents': {k: v() for k, v in self._parents.items()}}

    def __setstate__(self, data_dict):
        self.__dict__ = data_dict
        # turn the normal dictionary back into a dictionary with weak values
        # The extra lambda is to provide a callback to remove dead
        # weakrefs from the dictionary when garbage collection is done.
        self._parents = {
            k: weakref.ref(v, lambda _, pop=self._parents.pop, k=k: pop(k))
            for k, v in self._parents.items() if v is not None}

    def __copy__(self):
        cls = type(self)
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)
        # If `c = a + b; a1 = copy(a)`, then modifications to `a1` do not
        # propagate back to `c`, i.e. we need to clear the parents of `a1`.
        other._parents = {}
        # If `c = a + b; c1 = copy(c)`, then modifications to `a` also need to
        # be propagated to `c1`.
        for key, val in vars(self).items():
            if isinstance(val, TransformNode) and id(self) in val._parents:
                other.set_children(val)  # val == getattr(other, key)
        return other

    def invalidate(self):
        """
        Invalidate this `TransformNode` and triggers an invalidation of its
        ancestors.  Should be called any time the transform changes.
        """
        return self._invalidate_internal(
            level=self._INVALID_AFFINE_ONLY if self.is_affine else self._INVALID_FULL,
            invalidating_node=self)

    def _invalidate_internal(self, level, invalidating_node):
        """
        Called by :meth:`invalidate` and subsequently ascends the transform
        stack calling each TransformNode's _invalidate_internal method.
        """
        # If we are already more invalid than the currently propagated invalidation,
        # then we don't need to do anything.
        if level <= self._invalid and not self.pass_through:
            return
        self._invalid = level
        for parent in list(self._parents.values()):
            parent = parent()  # Dereference the weak reference.
            if parent is not None:
                parent._invalidate_internal(level=level, invalidating_node=self)

    def set_children(self, *children):
        """
        Set the children of the transform, to let the invalidation
        system know which transforms can invalidate this transform.
        Should be called from the constructor of any transforms that
        depend on other transforms.
        """
        # Parents are stored as weak references, so that if the
        # parents are destroyed, references from the children won't
        # keep them alive.
        id_self = id(self)
        for child in children:
            # Use weak references so this dictionary won't keep obsolete nodes
            # alive; the callback deletes the dictionary entry. This is a
            # performance improvement over using WeakValueDictionary.
            ref = weakref.ref(
                self, lambda _, pop=child._parents.pop, k=id_self: pop(k))
            child._parents[id_self] = ref

    def frozen(self):
        """
        Return a frozen copy of this transform node.  The frozen copy will not
        be updated when its children change.  Useful for storing a previously
        known state of a transform where ``copy.deepcopy()`` might normally be
        used.
        """
        return self


class BboxBase(TransformNode):
    """
    The base class of all bounding boxes.

    This class is immutable; `Bbox` is a mutable subclass.

    The canonical representation is as two points, with no
    restrictions on their ordering.  Convenience properties are
    provided to get the left, bottom, right and top edges and width
    and height, but these are not stored explicitly.
    """

    is_bbox = _api.deprecated("3.9")(_api.classproperty(lambda cls: True))
    is_affine = True

    if DEBUG:
        @staticmethod
        def _check(points):
            if isinstance(points, np.ma.MaskedArray):
                _api.warn_external("Bbox bounds are a masked array.")
            points = np.asarray(points)
            if any((points[1, :] - points[0, :]) == 0):
                _api.warn_external("Singular Bbox.")

    def frozen(self):
        return Bbox(self.get_points().copy())
    frozen.__doc__ = TransformNode.__doc__

    def __array__(self, *args, **kwargs):
        return self.get_points()

    @property
    def x0(self):
        """
        The first of the pair of *x* coordinates that define the bounding box.

        This is not guaranteed to be less than :attr:`x1` (for that, use
        :attr:`xmin`).
        """
        return self.get_points()[0, 0]

    @property
    def y0(self):
        """
        The first of the pair of *y* coordinates that define the bounding box.

        This is not guaranteed to be less than :attr:`y1` (for that, use
        :attr:`ymin`).
        """
        return self.get_points()[0, 1]

    @property
    def x1(self):
        """
        The second of the pair of *x* coordinates that define the bounding box.

        This is not guaranteed to be greater than :attr:`x0` (for that, use
        :attr:`xmax`).
        """
        return self.get_points()[1, 0]

    @property
    def y1(self):
        """
        The second of the pair of *y* coordinates that define the bounding box.

        This is not guaranteed to be greater than :attr:`y0` (for that, use
        :attr:`ymax`).
        """
        return self.get_points()[1, 1]

    @property
    def p0(self):
        """
        The first pair of (*x*, *y*) coordinates that define the bounding box.

        This is not guaranteed to be the bottom-left corner (for that, use
        :attr:`min`).
        """
        return self.get_points()[0]

    @property
    def p1(self):
        """
        The second pair of (*x*, *y*) coordinates that define the bounding box.

        This is not guaranteed to be the top-right corner (for that, use
        :attr:`max`).
        """
        return self.get_points()[1]

    @property
    def xmin(self):
        """The left edge of the bounding box."""
        return np.min(self.get_points()[:, 0])

    @property
    def ymin(self):
        """The bottom edge of the bounding box."""
        return np.min(self.get_points()[:, 1])

    @property
    def xmax(self):
        """The right edge of the bounding box."""
        return np.max(self.get_points()[:, 0])

    @property
    def ymax(self):
        """The top edge of the bounding box."""
        return np.max(self.get_points()[:, 1])

    @property
    def min(self):
        """The bottom-left corner of the bounding box."""
        return np.min(self.get_points(), axis=0)

    @property
    def max(self):
        """The top-right corner of the bounding box."""
        return np.max(self.get_points(), axis=0)

    @property
    def intervalx(self):
        """
        The pair of *x* coordinates that define the bounding box.

        This is not guaranteed to be sorted from left to right.
        """
        return self.get_points()[:, 0]

    @property
    def intervaly(self):
        """
        The pair of *y* coordinates that define the bounding box.

        This is not guaranteed to be sorted from bottom to top.
        """
        return self.get_points()[:, 1]

    @property
    def width(self):
        """The (signed) width of the bounding box."""
        points = self.get_points()
        return points[1, 0] - points[0, 0]

    @property
    def height(self):
        """The (signed) height of the bounding box."""
        points = self.get_points()
        return points[1, 1] - points[0, 1]

    @property
    def size(self):
        """The (signed) width and height of the bounding box."""
        points = self.get_points()
        return points[1] - points[0]

    @property
    def bounds(self):
        """Return (:attr:`x0`, :attr:`y0`, :attr:`width`, :attr:`height`)."""
        (x0, y0), (x1, y1) = self.get_points()
        return (x0, y0, x1 - x0, y1 - y0)

    @property
    def extents(self):
        """Return (:attr:`x0`, :attr:`y0`, :attr:`x1`, :attr:`y1`)."""
        return self.get_points().flatten()  # flatten returns a copy.

    def get_points(self):
        raise NotImplementedError

    def containsx(self, x):
        """
        Return whether *x* is in the closed (:attr:`x0`, :attr:`x1`) interval.
        """
        x0, x1 = self.intervalx
        return x0 <= x <= x1 or x0 >= x >= x1

    def containsy(self, y):
        """
        Return whether *y* is in the closed (:attr:`y0`, :attr:`y1`) interval.
        """
        y0, y1 = self.intervaly
        return y0 <= y <= y1 or y0 >= y >= y1

    def contains(self, x, y):
        """
        Return whether ``(x, y)`` is in the bounding box or on its edge.
        """
        return self.containsx(x) and self.containsy(y)

    def overlaps(self, other):
        """
        Return whether this bounding box overlaps with the other bounding box.

        Parameters
        ----------
        other : `.BboxBase`
        """
        ax1, ay1, ax2, ay2 = self.extents
        bx1, by1, bx2, by2 = other.extents
        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2
        return ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2

    def fully_containsx(self, x):
        """
        Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.
        """
        x0, x1 = self.intervalx
        return x0 < x < x1 or x0 > x > x1

    def fully_containsy(self, y):
        """
        Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.
        """
        y0, y1 = self.intervaly
        return y0 < y < y1 or y0 > y > y1

    def fully_contains(self, x, y):
        """
        Return whether ``x, y`` is in the bounding box, but not on its edge.
        """
        return self.fully_containsx(x) and self.fully_containsy(y)

    def fully_overlaps(self, other):
        """
        Return whether this bounding box overlaps with the other bounding box,
        not including the edges.

        Parameters
        ----------
        other : `.BboxBase`
        """
        ax1, ay1, ax2, ay2 = self.extents
        bx1, by1, bx2, by2 = other.extents
        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2
        return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2

    def transformed(self, transform):
        """
        Construct a `Bbox` by statically transforming this one by *transform*.
        """
        pts = self.get_points()
        ll, ul, lr = transform.transform(np.array(
            [pts[0], [pts[0, 0], pts[1, 1]], [pts[1, 0], pts[0, 1]]]))
        return Bbox([ll, [lr[0], ul[1]]])

    coefs = {'C':  (0.5, 0.5),
             'SW': (0, 0),
             'S':  (0.5, 0),
             'SE': (1.0, 0),
             'E':  (1.0, 0.5),
             'NE': (1.0, 1.0),
             'N':  (0.5, 1.0),
             'NW': (0, 1.0),
             'W':  (0, 0.5)}

    def anchored(self, c, container):
        """
        Return a copy of the `Bbox` anchored to *c* within *container*.

        Parameters
        ----------
        c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).
        container : `Bbox`
            The box within which the `Bbox` is positioned.

        See Also
        --------
        .Axes.set_anchor
        """
        l, b, w, h = container.bounds
        L, B, W, H = self.bounds
        cx, cy = self.coefs[c] if isinstance(c, str) else c
        return Bbox(self._points +
                    [(l + cx * (w - W)) - L,
                     (b + cy * (h - H)) - B])

    def shrunk(self, mx, my):
        """
        Return a copy of the `Bbox`, shrunk by the factor *mx*
        in the *x* direction and the factor *my* in the *y* direction.
        The lower left corner of the box remains unchanged.  Normally
        *mx* and *my* will be less than 1, but this is not enforced.
        """
        w, h = self.size
        return Bbox([self._points[0],
                     self._points[0] + [mx * w, my * h]])

    def shrunk_to_aspect(self, box_aspect, container=None, fig_aspect=1.0):
        """
        Return a copy of the `Bbox`, shrunk so that it is as
        large as it can be while having the desired aspect ratio,
        *box_aspect*.  If the box coordinates are relative (i.e.
        fractions of a larger box such as a figure) then the
        physical aspect ratio of that figure is specified with
        *fig_aspect*, so that *box_aspect* can also be given as a
        ratio of the absolute dimensions, not the relative dimensions.
        """
        if box_aspect <= 0 or fig_aspect <= 0:
            raise ValueError("'box_aspect' and 'fig_aspect' must be positive")
        if container is None:
            container = self
        w, h = container.size
        H = w * box_aspect / fig_aspect
        if H <= h:
            W = w
        else:
            W = h * fig_aspect / box_aspect
            H = h
        return Bbox([self._points[0],
                     self._points[0] + (W, H)])

    def splitx(self, *args):
        """
        Return a list of new `Bbox` objects formed by splitting the original
        one with vertical lines at fractional positions given by *args*.
        """
        xf = [0, *args, 1]
        x0, y0, x1, y1 = self.extents
        w = x1 - x0
        return [Bbox([[x0 + xf0 * w, y0], [x0 + xf1 * w, y1]])
                for xf0, xf1 in itertools.pairwise(xf)]

    def splity(self, *args):
        """
        Return a list of new `Bbox` objects formed by splitting the original
        one with horizontal lines at fractional positions given by *args*.
        """
        yf = [0, *args, 1]
        x0, y0, x1, y1 = self.extents
        h = y1 - y0
        return [Bbox([[x0, y0 + yf0 * h], [x1, y0 + yf1 * h]])
                for yf0, yf1 in itertools.pairwise(yf)]

    def count_contains(self, vertices):
        """
        Count the number of vertices contained in the `Bbox`.
        Any vertices with a non-finite x or y value are ignored.

        Parameters
        ----------
        vertices : (N, 2) array
        """
        if len(vertices) == 0:
            return 0
        vertices = np.asarray(vertices)
        with np.errstate(invalid='ignore'):
            return (((self.min < vertices) &
                     (vertices < self.max)).all(axis=1).sum())

    def count_overlaps(self, bboxes):
        """
        Count the number of bounding boxes that overlap this one.

        Parameters
        ----------
        bboxes : sequence of `.BboxBase`
        """
        return count_bboxes_overlapping_bbox(
            self, np.atleast_3d([np.array(x) for x in bboxes]))

    def expanded(self, sw, sh):
        """
        Construct a `Bbox` by expanding this one around its center by the
        factors *sw* and *sh*.
        """
        width = self.width
        height = self.height
        deltaw = (sw * width - width) / 2.0
        deltah = (sh * height - height) / 2.0
        a = np.array([[-deltaw, -deltah], [deltaw, deltah]])
        return Bbox(self._points + a)

    def padded(self, w_pad, h_pad=None):
        """
        Construct a `Bbox` by padding this one on all four sides.

        Parameters
        ----------
        w_pad : float
            Width pad
        h_pad : float, optional
            Height pad.  Defaults to *w_pad*.

        """
        points = self.get_points()
        if h_pad is None:
            h_pad = w_pad
        return Bbox(points + [[-w_pad, -h_pad], [w_pad, h_pad]])

    def translated(self, tx, ty):
        """Construct a `Bbox` by translating this one by *tx* and *ty*."""
        return Bbox(self._points + (tx, ty))

    def corners(self):
        """
        Return the corners of this rectangle as an array of points.

        Specifically, this returns the array
        ``[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]``.
        """
        (x0, y0), (x1, y1) = self.get_points()
        return np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])

    def rotated(self, radians):
        """
        Return the axes-aligned bounding box that bounds the result of rotating
        this `Bbox` by an angle of *radians*.
        """
        corners = self.corners()
        corners_rotated = Affine2D().rotate(radians).transform(corners)
        bbox = Bbox.unit()
        bbox.update_from_data_xy(corners_rotated, ignore=True)
        return bbox

    @staticmethod
    def union(bboxes):
        """Return a `Bbox` that contains all of the given *bboxes*."""
        if not len(bboxes):
            raise ValueError("'bboxes' cannot be empty")
        x0 = np.min([bbox.xmin for bbox in bboxes])
        x1 = np.max([bbox.xmax for bbox in bboxes])
        y0 = np.min([bbox.ymin for bbox in bboxes])
        y1 = np.max([bbox.ymax for bbox in bboxes])
        return Bbox([[x0, y0], [x1, y1]])

    @staticmethod
    def intersection(bbox1, bbox2):
        """
        Return the intersection of *bbox1* and *bbox2* if they intersect, or
        None if they don't.
        """
        x0 = np.maximum(bbox1.xmin, bbox2.xmin)
        x1 = np.minimum(bbox1.xmax, bbox2.xmax)
        y0 = np.maximum(bbox1.ymin, bbox2.ymin)
        y1 = np.minimum(bbox1.ymax, bbox2.ymax)
        return Bbox([[x0, y0], [x1, y1]]) if x0 <= x1 and y0 <= y1 else None


_default_minpos = np.array([np.inf, np.inf])


class Bbox(BboxBase):
    """
    A mutable bounding box.

    Examples
    --------
    **Create from known bounds**

    The default constructor takes the boundary "points" ``[[xmin, ymin],
    [xmax, ymax]]``.

        >>> Bbox([[1, 1], [3, 7]])
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    Alternatively, a Bbox can be created from the flattened points array, the
    so-called "extents" ``(xmin, ymin, xmax, ymax)``

        >>> Bbox.from_extents(1, 1, 3, 7)
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    or from the "bounds" ``(xmin, ymin, width, height)``.

        >>> Bbox.from_bounds(1, 1, 2, 6)
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    **Create from collections of points**

    The "empty" object for accumulating Bboxs is the null bbox, which is a
    stand-in for the empty set.

        >>> Bbox.null()
        Bbox([[inf, inf], [-inf, -inf]])

    Adding points to the null bbox will give you the bbox of those points.

        >>> box = Bbox.null()
        >>> box.update_from_data_xy([[1, 1]])
        >>> box
        Bbox([[1.0, 1.0], [1.0, 1.0]])
        >>> box.update_from_data_xy([[2, 3], [3, 2]], ignore=False)
        >>> box
        Bbox([[1.0, 1.0], [3.0, 3.0]])

    Setting ``ignore=True`` is equivalent to starting over from a null bbox.

        >>> box.update_from_data_xy([[1, 1]], ignore=True)
        >>> box
        Bbox([[1.0, 1.0], [1.0, 1.0]])

    .. warning::

        It is recommended to always specify ``ignore`` explicitly.  If not, the
        default value of ``ignore`` can be changed at any time by code with
        access to your Bbox, for example using the method `~.Bbox.ignore`.

    **Properties of the ``null`` bbox**

    .. note::

        The current behavior of `Bbox.null()` may be surprising as it does
        not have all of the properties of the "empty set", and as such does
        not behave like a "zero" object in the mathematical sense. We may
        change that in the future (with a deprecation period).

    The null bbox is the identity for intersections

        >>> Bbox.intersection(Bbox([[1, 1], [3, 7]]), Bbox.null())
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    except with itself, where it returns the full space.

        >>> Bbox.intersection(Bbox.null(), Bbox.null())
        Bbox([[-inf, -inf], [inf, inf]])

    A union containing null will always return the full space (not the other
    set!)

        >>> Bbox.union([Bbox([[0, 0], [0, 0]]), Bbox.null()])
        Bbox([[-inf, -inf], [inf, inf]])
    """

    def __init__(self, points, **kwargs):
        """
        Parameters
        ----------
        points : `~numpy.ndarray`
            A (2, 2) array of the form ``[[x0, y0], [x1, y1]]``.
        """
        super().__init__(**kwargs)
        points = np.asarray(points, float)
        if points.shape != (2, 2):
            raise ValueError('Bbox points must be of the form '
                             '"[[x0, y0], [x1, y1]]".')
        self._points = points
        self._minpos = _default_minpos.copy()
        self._ignore = True
        # it is helpful in some contexts to know if the bbox is a
        # default or has been mutated; we store the orig points to
        # support the mutated methods
        self._points_orig = self._points.copy()
    if DEBUG:
        ___init__ = __init__

        def __init__(self, points, **kwargs):
            self._check(points)
            self.___init__(points, **kwargs)

        def invalidate(self):
            self._check(self._points)
            super().invalidate()

    def frozen(self):
        # docstring inherited
        frozen_bbox = super().frozen()
        frozen_bbox._minpos = self.minpos.copy()
        return frozen_bbox

    @staticmethod
    def unit():
        """Create a new unit `Bbox` from (0, 0) to (1, 1)."""
        return Bbox([[0, 0], [1, 1]])

    @staticmethod
    def null():
        """Create a new null `Bbox` from (inf, inf) to (-inf, -inf)."""
        return Bbox([[np.inf, np.inf], [-np.inf, -np.inf]])

    @staticmethod
    def from_bounds(x0, y0, width, height):
        """
        Create a new `Bbox` from *x0*, *y0*, *width* and *height*.

        *width* and *height* may be negative.
        """
        return Bbox.from_extents(x0, y0, x0 + width, y0 + height)

    @staticmethod
    def from_extents(*args, minpos=None):
        """
        Create a new Bbox from *left*, *bottom*, *right* and *top*.

        The *y*-axis increases upwards.

        Parameters
        ----------
        left, bottom, right, top : float
            The four extents of the bounding box.
        minpos : float or None
            If this is supplied, the Bbox will have a minimum positive value
            set. This is useful when dealing with logarithmic scales and other
            scales where negative bounds result in floating point errors.
        """
        bbox = Bbox(np.reshape(args, (2, 2)))
        if minpos is not None:
            bbox._minpos[:] = minpos
        return bbox

    def __format__(self, fmt):
        return (
            'Bbox(x0={0.x0:{1}}, y0={0.y0:{1}}, x1={0.x1:{1}}, y1={0.y1:{1}})'.
            format(self, fmt))

    def __str__(self):
        return format(self, '')

    def __repr__(self):
        return 'Bbox([[{0.x0}, {0.y0}], [{0.x1}, {0.y1}]])'.format(self)

    def ignore(self, value):
        """
        Set whether the existing bounds of the box should be ignored
        by subsequent calls to :meth:`update_from_data_xy`.

        value : bool
            - When ``True``, subsequent calls to `update_from_data_xy` will
              ignore the existing bounds of the `Bbox`.
            - When ``False``, subsequent calls to `update_from_data_xy` will
              include the existing bounds of the `Bbox`.
        """
        self._ignore = value

    def update_from_path(self, path, ignore=None, updatex=True, updatey=True):
        """
        Update the bounds of the `Bbox` to contain the vertices of the
        provided path. After updating, the bounds will have positive *width*
        and *height*; *x0* and *y0* will be the minimal values.

        Parameters
        ----------
        path : `~matplotlib.path.Path`
        ignore : bool, optional
           - When ``True``, ignore the existing bounds of the `Bbox`.
           - When ``False``, include the existing bounds of the `Bbox`.
           - When ``None``, use the last value passed to :meth:`ignore`.
        updatex, updatey : bool, default: True
            When ``True``, update the x/y values.
        """
        if ignore is None:
            ignore = self._ignore

        if path.vertices.size == 0:
            return

        points, minpos, changed = update_path_extents(
            path, None, self._points, self._minpos, ignore)

        if changed:
            self.invalidate()
            if updatex:
                self._points[:, 0] = points[:, 0]
                self._minpos[0] = minpos[0]
            if updatey:
                self._points[:, 1] = points[:, 1]
                self._minpos[1] = minpos[1]

    def update_from_data_x(self, x, ignore=None):
        """
        Update the x-bounds of the `Bbox` based on the passed in data. After
        updating, the bounds will have positive *width*, and *x0* will be the
        minimal value.

        Parameters
        ----------
        x : `~numpy.ndarray`
            Array of x-values.
        ignore : bool, optional
           - When ``True``, ignore the existing bounds of the `Bbox`.
           - When ``False``, include the existing bounds of the `Bbox`.
           - When ``None``, use the last value passed to :meth:`ignore`.
        """
        x = np.ravel(x)
        self.update_from_data_xy(np.column_stack([x, np.ones(x.size)]),
                                 ignore=ignore, updatey=False)

    def update_from_data_y(self, y, ignore=None):
        """
        Update the y-bounds of the `Bbox` based on the passed in data. After
        updating, the bounds will have positive *height*, and *y0* will be the
        minimal value.

        Parameters
        ----------
        y : `~numpy.ndarray`
            Array of y-values.
        ignore : bool, optional
            - When ``True``, ignore the existing bounds of the `Bbox`.
            - When ``False``, include the existing bounds of the `Bbox`.
            - When ``None``, use the last value passed to :meth:`ignore`.
        """
        y = np.ravel(y)
        self.update_from_data_xy(np.column_stack([np.ones(y.size), y]),
                                 ignore=ignore, updatex=False)

    def update_from_data_xy(self, xy, ignore=None, updatex=True, updatey=True):
        """
        Update the `Bbox` bounds based on the passed in *xy* coordinates.

        After updating, the bounds will have positive *width* and *height*;
        *x0* and *y0* will be the minimal values.

        Parameters
        ----------
        xy : (N, 2) array-like
            The (x, y) coordinates.
        ignore : bool, optional
            - When ``True``, ignore the existing bounds of the `Bbox`.
            - When ``False``, include the existing bounds of the `Bbox`.
            - When ``None``, use the last value passed to :meth:`ignore`.
        updatex, updatey : bool, default: True
             When ``True``, update the x/y values.
        """
        if len(xy) == 0:
            return

        path = Path(xy)
        self.update_from_path(path, ignore=ignore,
                              updatex=updatex, updatey=updatey)

    @BboxBase.x0.setter
    def x0(self, val):
        self._points[0, 0] = val
        self.invalidate()

    @BboxBase.y0.setter
    def y0(self, val):
        self._points[0, 1] = val
        self.invalidate()

    @BboxBase.x1.setter
    def x1(self, val):
        self._points[1, 0] = val
        self.invalidate()

    @BboxBase.y1.setter
    def y1(self, val):
        self._points[1, 1] = val
        self.invalidate()

    @BboxBase.p0.setter
    def p0(self, val):
        self._points[0] = val
        self.invalidate()

    @BboxBase.p1.setter
    def p1(self, val):
        self._points[1] = val
        self.invalidate()

    @BboxBase.intervalx.setter
    def intervalx(self, interval):
        self._points[:, 0] = interval
        self.invalidate()

    @BboxBase.intervaly.setter
    def intervaly(self, interval):
        self._points[:, 1] = interval
        self.invalidate()

    @BboxBase.bounds.setter
    def bounds(self, bounds):
        l, b, w, h = bounds
        points = np.array([[l, b], [l + w, b + h]], float)
        if np.any(self._points != points):
            self._points = points
            self.invalidate()

    @property
    def minpos(self):
        """
        The minimum positive value in both directions within the Bbox.

        This is useful when dealing with logarithmic scales and other scales
        where negative bounds result in floating point errors, and will be used
        as the minimum extent instead of *p0*.
        """
        return self._minpos

    @minpos.setter
    def minpos(self, val):
        self._minpos[:] = val

    @property
    def minposx(self):
        """
        The minimum positive value in the *x*-direction within the Bbox.

        This is useful when dealing with logarithmic scales and other scales
        where negative bounds result in floating point errors, and will be used
        as the minimum *x*-extent instead of *x0*.
        """
        return self._minpos[0]

    @minposx.setter
    def minposx(self, val):
        self._minpos[0] = val

    @property
    def minposy(self):
        """
        The minimum positive value in the *y*-direction within the Bbox.

        This is useful when dealing with logarithmic scales and other scales
        where negative bounds result in floating point errors, and will be used
        as the minimum *y*-extent instead of *y0*.
        """
        return self._minpos[1]

    @minposy.setter
    def minposy(self, val):
        self._minpos[1] = val

    def get_points(self):
        """
        Get the points of the bounding box as an array of the form
        ``[[x0, y0], [x1, y1]]``.
        """
        self._invalid = 0
        return self._points

    def set_points(self, points):
        """
        Set the points of the bounding box directly from an array of the form
        ``[[x0, y0], [x1, y1]]``.  No error checking is performed, as this
        method is mainly for internal use.
        """
        if np.any(self._points != points):
            self._points = points
            self.invalidate()

    def set(self, other):
        """
        Set this bounding box from the "frozen" bounds of another `Bbox`.
        """
        if np.any(self._points != other.get_points()):
            self._points = other.get_points()
            self.invalidate()

    def mutated(self):
        """Return whether the bbox has changed since init."""
        return self.mutatedx() or self.mutatedy()

    def mutatedx(self):
        """Return whether the x-limits have changed since init."""
        return (self._points[0, 0] != self._points_orig[0, 0] or
                self._points[1, 0] != self._points_orig[1, 0])

    def mutatedy(self):
        """Return whether the y-limits have changed since init."""
        return (self._points[0, 1] != self._points_orig[0, 1] or
                self._points[1, 1] != self._points_orig[1, 1])


class TransformedBbox(BboxBase):
    """
    A `Bbox` that is automatically transformed by a given
    transform.  When either the child bounding box or transform
    changes, the bounds of this bbox will update accordingly.
    """

    def __init__(self, bbox, transform, **kwargs):
        """
        Parameters
        ----------
        bbox : `Bbox`
        transform : `Transform`
        """
        _api.check_isinstance(BboxBase, bbox=bbox)
        _api.check_isinstance(Transform, transform=transform)
        if transform.input_dims != 2 or transform.output_dims != 2:
            raise ValueError(
                "The input and output dimensions of 'transform' must be 2")

        super().__init__(**kwargs)
        self._bbox = bbox
        self._transform = transform
        self.set_children(bbox, transform)
        self._points = None

    __str__ = _make_str_method("_bbox", "_transform")

    def get_points(self):
        # docstring inherited
        if self._invalid:
            p = self._bbox.get_points()
            # Transform all four points, then make a new bounding box
            # from the result, taking care to make the orientation the
            # same.
            points = self._transform.transform(
                [[p[0, 0], p[0, 1]],
                 [p[1, 0], p[0, 1]],
                 [p[0, 0], p[1, 1]],
                 [p[1, 0], p[1, 1]]])
            points = np.ma.filled(points, 0.0)

            xs = min(points[:, 0]), max(points[:, 0])
            if p[0, 0] > p[1, 0]:
                xs = xs[::-1]

            ys = min(points[:, 1]), max(points[:, 1])
            if p[0, 1] > p[1, 1]:
                ys = ys[::-1]

            self._points = np.array([
                [xs[0], ys[0]],
                [xs[1], ys[1]]
            ])

            self._invalid = 0
        return self._points

    if DEBUG:
        _get_points = get_points

        def get_points(self):
            points = self._get_points()
            self._check(points)
            return points

    def contains(self, x, y):
        # Docstring inherited.
        return self._bbox.contains(*self._transform.inverted().transform((x, y)))

    def fully_contains(self, x, y):
        # Docstring inherited.
        return self._bbox.fully_contains(*self._transform.inverted().transform((x, y)))


class LockableBbox(BboxBase):
    """
    A `Bbox` where some elements may be locked at certain values.

    When the child bounding box changes, the bounds of this bbox will update
    accordingly with the exception of the locked elements.
    """
    def __init__(self, bbox, x0=None, y0=None, x1=None, y1=None, **kwargs):
        """
        Parameters
        ----------
        bbox : `Bbox`
            The child bounding box to wrap.

        x0 : float or None
            The locked value for x0, or None to leave unlocked.

        y0 : float or None
            The locked value for y0, or None to leave unlocked.

        x1 : float or None
            The locked value for x1, or None to leave unlocked.

        y1 : float or None
            The locked value for y1, or None to leave unlocked.

        """
        _api.check_isinstance(BboxBase, bbox=bbox)
        super().__init__(**kwargs)
        self._bbox = bbox
        self.set_children(bbox)
        self._points = None
        fp = [x0, y0, x1, y1]
        mask = [val is None for val in fp]
        self._locked_points = np.ma.array(fp, float, mask=mask).reshape((2, 2))

    __str__ = _make_str_method("_bbox", "_locked_points")

    def get_points(self):
        # docstring inherited
        if self._invalid:
            points = self._bbox.get_points()
            self._points = np.where(self._locked_points.mask,
                                    points,
                                    self._locked_points)
            self._invalid = 0
        return self._points

    if DEBUG:
        _get_points = get_points

        def get_points(self):
            points = self._get_points()
            self._check(points)
            return points

    @property
    def locked_x0(self):
        """
        float or None: The value used for the locked x0.
        """
        if self._locked_points.mask[0, 0]:
            return None
        else:
            return self._locked_points[0, 0]

    @locked_x0.setter
    def locked_x0(self, x0):
        self._locked_points.mask[0, 0] = x0 is None
        self._locked_points.data[0, 0] = x0
        self.invalidate()

    @property
    def locked_y0(self):
        """
        float or None: The value used for the locked y0.
        """
        if self._locked_points.mask[0, 1]:
            return None
        else:
            return self._locked_points[0, 1]

    @locked_y0.setter
    def locked_y0(self, y0):
        self._locked_points.mask[0, 1] = y0 is None
        self._locked_points.data[0, 1] = y0
        self.invalidate()

    @property
    def locked_x1(self):
        """
        float or None: The value used for the locked x1.
        """
        if self._locked_points.mask[1, 0]:
            return None
        else:
            return self._locked_points[1, 0]

    @locked_x1.setter
    def locked_x1(self, x1):
        self._locked_points.mask[1, 0] = x1 is None
        self._locked_points.data[1, 0] = x1
        self.invalidate()

    @property
    def locked_y1(self):
        """
        float or None: The value used for the locked y1.
        """
        if self._locked_points.mask[1, 1]:
            return None
        else:
            return self._locked_points[1, 1]

    @locked_y1.setter
    def locked_y1(self, y1):
        self._locked_points.mask[1, 1] = y1 is None
        self._locked_points.data[1, 1] = y1
        self.invalidate()


class Transform(TransformNode):
    """
    The base class of all `TransformNode` instances that
    actually perform a transformation.

    All non-affine transformations should be subclasses of this class.
    New affine transformations should be subclasses of `Affine2D`.

    Subclasses of this class should override the following members (at
    minimum):

    - :attr:`input_dims`
    - :attr:`output_dims`
    - :meth:`transform`
    - :meth:`inverted` (if an inverse exists)

    The following attributes may be overridden if the default is unsuitable:

    - :attr:`is_separable` (defaults to True for 1D -> 1D transforms, False
      otherwise)
    - :attr:`has_inverse` (defaults to True if :meth:`inverted` is overridden,
      False otherwise)

    If the transform needs to do something non-standard with
    `matplotlib.path.Path` objects, such as adding curves
    where there were once line segments, it should override:

    - :meth:`transform_path`
    """

    input_dims = None
    """
    The number of input dimensions of this transform.
    Must be overridden (with integers) in the subclass.
    """

    output_dims = None
    """
    The number of output dimensions of this transform.
    Must be overridden (with integers) in the subclass.
    """

    is_separable = False
    """True if this transform is separable in the x- and y- dimensions."""

    has_inverse = False
    """True if this transform has a corresponding inverse transform."""

    def __init_subclass__(cls):
        # 1d transforms are always separable; we assume higher-dimensional ones
        # are not but subclasses can also directly set is_separable -- this is
        # verified by checking whether "is_separable" appears more than once in
        # the class's MRO (it appears once in Transform).
        if (sum("is_separable" in vars(parent) for parent in cls.__mro__) == 1
                and cls.input_dims == cls.output_dims == 1):
            cls.is_separable = True
        # Transform.inverted raises NotImplementedError; we assume that if this
        # is overridden then the transform is invertible but subclass can also
        # directly set has_inverse.
        if (sum("has_inverse" in vars(parent) for parent in cls.__mro__) == 1
                and hasattr(cls, "inverted")
                and cls.inverted is not Transform.inverted):
            cls.has_inverse = True

    def __add__(self, other):
        """
        Compose two transforms together so that *self* is followed by *other*.

        ``A + B`` returns a transform ``C`` so that
        ``C.transform(x) == B.transform(A.transform(x))``.
        """
        return (composite_transform_factory(self, other)
                if isinstance(other, Transform) else
                NotImplemented)

    # Equality is based on object identity for `Transform`s (so we don't
    # override `__eq__`), but some subclasses, such as TransformWrapper &
    # AffineBase, override this behavior.

    def _iter_break_from_left_to_right(self):
        """
        Return an iterator breaking down this transform stack from left to
        right recursively. If self == ((A, N), A) then the result will be an
        iterator which yields I : ((A, N), A), followed by A : (N, A),
        followed by (A, N) : (A), but not ((A, N), A) : I.

        This is equivalent to flattening the stack then yielding
        ``flat_stack[:i], flat_stack[i:]`` where i=0..(n-1).
        """
        yield IdentityTransform(), self

    @property
    def depth(self):
        """
        Return the number of transforms which have been chained
        together to form this Transform instance.

        .. note::

            For the special case of a Composite transform, the maximum depth
            of the two is returned.

        """
        return 1

    def contains_branch(self, other):
        """
        Return whether the given transform is a sub-tree of this transform.

        This routine uses transform equality to identify sub-trees, therefore
        in many situations it is object id which will be used.

        For the case where the given transform represents the whole
        of this transform, returns True.
        """
        if self.depth < other.depth:
            return False

        # check that a subtree is equal to other (starting from self)
        for _, sub_tree in self._iter_break_from_left_to_right():
            if sub_tree == other:
                return True
        return False

    def contains_branch_seperately(self, other_transform):
        """
        Return whether the given branch is a sub-tree of this transform on
        each separate dimension.

        A common use for this method is to identify if a transform is a blended
        transform containing an Axes' data transform. e.g.::

            x_isdata, y_isdata = trans.contains_branch_seperately(ax.transData)

        """
        if self.output_dims != 2:
            raise ValueError('contains_branch_seperately only supports '
                             'transforms with 2 output dimensions')
        # for a non-blended transform each separate dimension is the same, so
        # just return the appropriate shape.
        return (self.contains_branch(other_transform), ) * 2

    def __sub__(self, other):
        """
        Compose *self* with the inverse of *other*, cancelling identical terms
        if any::

            # In general:
            A - B == A + B.inverted()
            # (but see note regarding frozen transforms below).

            # If A "ends with" B (i.e. A == A' + B for some A') we can cancel
            # out B:
            (A' + B) - B == A'

            # Likewise, if B "starts with" A (B = A + B'), we can cancel out A:
            A - (A + B') == B'.inverted() == B'^-1

        Cancellation (rather than naively returning ``A + B.inverted()``) is
        important for multiple reasons:

        - It avoids floating-point inaccuracies when computing the inverse of
          B: ``B - B`` is guaranteed to cancel out exactly (resulting in the
          identity transform), whereas ``B + B.inverted()`` may differ by a
          small epsilon.
        - ``B.inverted()`` always returns a frozen transform: if one computes
          ``A + B + B.inverted()`` and later mutates ``B``, then
          ``B.inverted()`` won't be updated and the last two terms won't cancel
          out anymore; on the other hand, ``A + B - B`` will always be equal to
          ``A`` even if ``B`` is mutated.
        """
        # we only know how to do this operation if other is a Transform.
        if not isinstance(other, Transform):
            return NotImplemented
        for remainder, sub_tree in self._iter_break_from_left_to_right():
            if sub_tree == other:
                return remainder
        for remainder, sub_tree in other._iter_break_from_left_to_right():
            if sub_tree == self:
                if not remainder.has_inverse:
                    raise ValueError(
                        "The shortcut cannot be computed since 'other' "
                        "includes a non-invertible component")
                return remainder.inverted()
        # if we have got this far, then there was no shortcut possible
        if other.has_inverse:
            return self + other.inverted()
        else:
            raise ValueError('It is not possible to compute transA - transB '
                             'since transB cannot be inverted and there is no '
                             'shortcut possible.')

    def __array__(self, *args, **kwargs):
        """Array interface to get at this Transform's affine matrix."""
        return self.get_affine().get_matrix()

    def transform(self, values):
        """
        Apply this transformation on the given array of *values*.

        Parameters
        ----------
        values : array-like
            The input values as an array of length :attr:`input_dims` or
            shape (N, :attr:`input_dims`).

        Returns
        -------
        array
            The output values as an array of length :attr:`output_dims` or
            shape (N, :attr:`output_dims`), depending on the input.
        """
        # Ensure that values is a 2d array (but remember whether
        # we started with a 1d or 2d array).
        values = np.asanyarray(values)
        ndim = values.ndim
        values = values.reshape((-1, self.input_dims))

        # Transform the values
        res = self.transform_affine(self.transform_non_affine(values))

        # Convert the result back to the shape of the input values.
        if ndim == 0:
            assert not np.ma.is_masked(res)  # just to be on the safe side
            return res[0, 0]
        if ndim == 1:
            return res.reshape(-1)
        elif ndim == 2:
            return res
        raise ValueError(
            "Input values must have shape (N, {dims}) or ({dims},)"
            .format(dims=self.input_dims))

    def transform_affine(self, values):
        """
        Apply only the affine part of this transformation on the
        given array of values.

        ``transform(values)`` is always equivalent to
        ``transform_affine(transform_non_affine(values))``.

        In non-affine transformations, this is generally a no-op.  In
        affine transformations, this is equivalent to
        ``transform(values)``.

        Parameters
        ----------
        values : array
            The input values as an array of length :attr:`input_dims` or
            shape (N, :attr:`input_dims`).

        Returns
        -------
        array
            The output values as an array of length :attr:`output_dims` or
            shape (N, :attr:`output_dims`), depending on the input.
        """
        return self.get_affine().transform(values)

    def transform_non_affine(self, values):
        """
        Apply only the non-affine part of this transformation.

        ``transform(values)`` is always equivalent to
        ``transform_affine(transform_non_affine(values))``.

        In non-affine transformations, this is generally equivalent to
        ``transform(values)``.  In affine transformations, this is
        always a no-op.

        Parameters
        ----------
        values : array
            The input values as an array of length :attr:`input_dims` or
            shape (N, :attr:`input_dims`).

        Returns
        -------
        array
            The output values as an array of length :attr:`output_dims` or
            shape (N, :attr:`output_dims`), depending on the input.
        """
        return values

    def transform_bbox(self, bbox):
        """
        Transform the given bounding box.

        For smarter transforms including caching (a common requirement in
        Matplotlib), see `TransformedBbox`.
        """
        return Bbox(self.transform(bbox.get_points()))

    def get_affine(self):
        """Get the affine part of this transform."""
        return IdentityTransform()

    def get_matrix(self):
        """Get the matrix for the affine part of this transform."""
        return self.get_affine().get_matrix()

    def transform_point(self, point):
        """
        Return a transformed point.

        This function is only kept for backcompatibility; the more general
        `.transform` method is capable of transforming both a list of points
        and a single point.

        The point is given as a sequence of length :attr:`input_dims`.
        The transformed point is returned as a sequence of length
        :attr:`output_dims`.
        """
        if len(point) != self.input_dims:
            raise ValueError("The length of 'point' must be 'self.input_dims'")
        return self.transform(point)

    def transform_path(self, path):
        """
        Apply the transform to `.Path` *path*, returning a new `.Path`.

        In some cases, this transform may insert curves into the path
        that began as line segments.
        """
        return self.transform_path_affine(self.transform_path_non_affine(path))

    def transform_path_affine(self, path):
        """
        Apply the affine part of this transform to `.Path` *path*, returning a
        new `.Path`.

        ``transform_path(path)`` is equivalent to
        ``transform_path_affine(transform_path_non_affine(values))``.
        """
        return self.get_affine().transform_path_affine(path)

    def transform_path_non_affine(self, path):
        """
        Apply the non-affine part of this transform to `.Path` *path*,
        returning a new `.Path`.

        ``transform_path(path)`` is equivalent to
        ``transform_path_affine(transform_path_non_affine(values))``.
        """
        x = self.transform_non_affine(path.vertices)
        return Path._fast_from_codes_and_verts(x, path.codes, path)

    def transform_angles(self, angles, pts, radians=False, pushoff=1e-5):
        """
        Transform a set of angles anchored at specific locations.

        Parameters
        ----------
        angles : (N,) array-like
            The angles to transform.
        pts : (N, 2) array-like
            The points where the angles are anchored.
        radians : bool, default: False
            Whether *angles* are radians or degrees.
        pushoff : float
            For each point in *pts* and angle in *angles*, the transformed
            angle is computed by transforming a segment of length *pushoff*
            starting at that point and making that angle relative to the
            horizontal axis, and measuring the angle between the horizontal
            axis and the transformed segment.

        Returns
        -------
        (N,) array
        """
        # Must be 2D
        if self.input_dims != 2 or self.output_dims != 2:
            raise NotImplementedError('Only defined in 2D')
        angles = np.asarray(angles)
        pts = np.asarray(pts)
        _api.check_shape((None, 2), pts=pts)
        _api.check_shape((None,), angles=angles)
        if len(angles) != len(pts):
            raise ValueError("There must be as many 'angles' as 'pts'")
        # Convert to radians if desired
        if not radians:
            angles = np.deg2rad(angles)
        # Move a short distance away
        pts2 = pts + pushoff * np.column_stack([np.cos(angles),
                                                np.sin(angles)])
        # Transform both sets of points
        tpts = self.transform(pts)
        tpts2 = self.transform(pts2)
        # Calculate transformed angles
        d = tpts2 - tpts
        a = np.arctan2(d[:, 1], d[:, 0])
        # Convert back to degrees if desired
        if not radians:
            a = np.rad2deg(a)
        return a

    def inverted(self):
        """
        Return the corresponding inverse transformation.

        It holds ``x == self.inverted().transform(self.transform(x))``.

        The return value of this method should be treated as
        temporary.  An update to *self* does not cause a corresponding
        update to its inverted copy.
        """
        raise NotImplementedError()


class TransformWrapper(Transform):
    """
    A helper class that holds a single child transform and acts
    equivalently to it.

    This is useful if a node of the transform tree must be replaced at
    run time with a transform of a different type.  This class allows
    that replacement to correctly trigger invalidation.

    `TransformWrapper` instances must have the same input and output dimensions
    during their entire lifetime, so the child transform may only be replaced
    with another child transform of the same dimensions.
    """

    pass_through = True

    def __init__(self, child):
        """
        *child*: A `Transform` instance.  This child may later
        be replaced with :meth:`set`.
        """
        _api.check_isinstance(Transform, child=child)
        super().__init__()
        self.set(child)

    def __eq__(self, other):
        return self._child.__eq__(other)

    __str__ = _make_str_method("_child")

    def frozen(self):
        # docstring inherited
        return self._child.frozen()

    def set(self, child):
        """
        Replace the current child of this transform with another one.

        The new child must have the same number of input and output
        dimensions as the current child.
        """
        if hasattr(self, "_child"):  # Absent during init.
            self.invalidate()
            new_dims = (child.input_dims, child.output_dims)
            old_dims = (self._child.input_dims, self._child.output_dims)
            if new_dims != old_dims:
                raise ValueError(
                    f"The input and output dims of the new child {new_dims} "
                    f"do not match those of current child {old_dims}")
            self._child._parents.pop(id(self), None)

        self._child = child
        self.set_children(child)

        self.transform = child.transform
        self.transform_affine = child.transform_affine
        self.transform_non_affine = child.transform_non_affine
        self.transform_path = child.transform_path
        self.transform_path_affine = child.transform_path_affine
        self.transform_path_non_affine = child.transform_path_non_affine
        self.get_affine = child.get_affine
        self.inverted = child.inverted
        self.get_matrix = child.get_matrix
        # note we do not wrap other properties here since the transform's
        # child can be changed with WrappedTransform.set and so checking
        # is_affine and other such properties may be dangerous.

        self._invalid = 0
        self.invalidate()
        self._invalid = 0

    input_dims = property(lambda self: self._child.input_dims)
    output_dims = property(lambda self: self._child.output_dims)
    is_affine = property(lambda self: self._child.is_affine)
    is_separable = property(lambda self: self._child.is_separable)
    has_inverse = property(lambda self: self._child.has_inverse)


class AffineBase(Transform):
    """
    The base class of all affine transformations of any number of dimensions.
    """
    is_affine = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inverted = None

    def __array__(self, *args, **kwargs):
        # optimises the access of the transform matrix vs. the superclass
        return self.get_matrix()

    def __eq__(self, other):
        if getattr(other, "is_affine", False) and hasattr(other, "get_matrix"):
            return (self.get_matrix() == other.get_matrix()).all()
        return NotImplemented

    def transform(self, values):
        # docstring inherited
        return self.transform_affine(values)

    def transform_affine(self, values):
        # docstring inherited
        raise NotImplementedError('Affine subclasses should override this '
                                  'method.')

    def transform_non_affine(self, values):
        # docstring inherited
        return values

    def transform_path(self, path):
        # docstring inherited
        return self.transform_path_affine(path)

    def transform_path_affine(self, path):
        # docstring inherited
        return Path(self.transform_affine(path.vertices),
                    path.codes, path._interpolation_steps)

    def transform_path_non_affine(self, path):
        # docstring inherited
        return path

    def get_affine(self):
        # docstring inherited
        return self


class Affine2DBase(AffineBase):
    """
    The base class of all 2D affine transformations.

    2D affine transformations are performed using a 3x3 numpy array::

        a c e
        b d f
        0 0 1

    This class provides the read-only interface.  For a mutable 2D
    affine transformation, use `Affine2D`.

    Subclasses of this class will generally only need to override a
    constructor and `~.Transform.get_matrix` that generates a custom 3x3 matrix.
    """
    input_dims = 2
    output_dims = 2

    def frozen(self):
        # docstring inherited
        return Affine2D(self.get_matrix().copy())

    @property
    def is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == mtx[1, 0] == 0.0

    def to_values(self):
        """
        Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
        """
        mtx = self.get_matrix()
        return tuple(mtx[:2].swapaxes(0, 1).flat)

    def transform_affine(self, values):
        mtx = self.get_matrix()
        if isinstance(values, np.ma.MaskedArray):
            tpoints = affine_transform(values.data, mtx)
            return np.ma.MaskedArray(tpoints, mask=np.ma.getmask(values))
        return affine_transform(values, mtx)

    if DEBUG:
        _transform_affine = transform_affine

        def transform_affine(self, values):
            # docstring inherited
            # The major speed trap here is just converting to the
            # points to an array in the first place.  If we can use
            # more arrays upstream, that should help here.
            if not isinstance(values, np.ndarray):
                _api.warn_external(
                    f'A non-numpy array of type {type(values)} was passed in '
                    f'for transformation, which results in poor performance.')
            return self._transform_affine(values)

    def inverted(self):
        # docstring inherited
        if self._inverted is None or self._invalid:
            mtx = self.get_matrix()
            shorthand_name = None
            if self._shorthand_name:
                shorthand_name = '(%s)-1' % self._shorthand_name
            self._inverted = Affine2D(inv(mtx), shorthand_name=shorthand_name)
            self._invalid = 0
        return self._inverted


class Affine2D(Affine2DBase):
    """
    A mutable 2D affine transformation.
    """

    def __init__(self, matrix=None, **kwargs):
        """
        Initialize an Affine transform from a 3x3 numpy float array::

          a c e
          b d f
          0 0 1

        If *matrix* is None, initialize with the identity transform.
        """
        super().__init__(**kwargs)
        if matrix is None:
            # A bit faster than np.identity(3).
            matrix = IdentityTransform._mtx
        self._mtx = matrix.copy()
        self._invalid = 0

    _base_str = _make_str_method("_mtx")

    def __str__(self):
        return (self._base_str()
                if (self._mtx != np.diag(np.diag(self._mtx))).any()
                else f"Affine2D().scale({self._mtx[0, 0]}, {self._mtx[1, 1]})"
                if self._mtx[0, 0] != self._mtx[1, 1]
                else f"Affine2D().scale({self._mtx[0, 0]})")

    @staticmethod
    def from_values(a, b, c, d, e, f):
        """
        Create a new Affine2D instance from the given values::

          a c e
          b d f
          0 0 1

        .
        """
        return Affine2D(
            np.array([a, c, e, b, d, f, 0.0, 0.0, 1.0], float).reshape((3, 3)))

    def get_matrix(self):
        """
        Get the underlying transformation matrix as a 3x3 array::

          a c e
          b d f
          0 0 1

        .
        """
        if self._invalid:
            self._inverted = None
            self._invalid = 0
        return self._mtx

    def set_matrix(self, mtx):
        """
        Set the underlying transformation matrix from a 3x3 array::

          a c e
          b d f
          0 0 1

        .
        """
        self._mtx = mtx
        self.invalidate()

    def set(self, other):
        """
        Set this transformation from the frozen copy of another
        `Affine2DBase` object.
        """
        _api.check_isinstance(Affine2DBase, other=other)
        self._mtx = other.get_matrix()
        self.invalidate()

    def clear(self):
        """
        Reset the underlying matrix to the identity transform.
        """
        # A bit faster than np.identity(3).
        self._mtx = IdentityTransform._mtx.copy()
        self.invalidate()
        return self

    def rotate(self, theta):
        """
        Add a rotation (in radians) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        a = math.cos(theta)
        b = math.sin(theta)
        mtx = self._mtx
        # Operating and assigning one scalar at a time is much faster.
        (xx, xy, x0), (yx, yy, y0), _ = mtx.tolist()
        # mtx = [[a -b 0], [b a 0], [0 0 1]] * mtx
        mtx[0, 0] = a * xx - b * yx
        mtx[0, 1] = a * xy - b * yy
        mtx[0, 2] = a * x0 - b * y0
        mtx[1, 0] = b * xx + a * yx
        mtx[1, 1] = b * xy + a * yy
        mtx[1, 2] = b * x0 + a * y0
        self.invalidate()
        return self

    def rotate_deg(self, degrees):
        """
        Add a rotation (in degrees) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.rotate(math.radians(degrees))

    def rotate_around(self, x, y, theta):
        """
        Add a rotation (in radians) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.translate(-x, -y).rotate(theta).translate(x, y)

    def rotate_deg_around(self, x, y, degrees):
        """
        Add a rotation (in degrees) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # Cast to float to avoid wraparound issues with uint8's
        x, y = float(x), float(y)
        return self.translate(-x, -y).rotate_deg(degrees).translate(x, y)

    def translate(self, tx, ty):
        """
        Add a translation in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        self._mtx[0, 2] += tx
        self._mtx[1, 2] += ty
        self.invalidate()
        return self

    def scale(self, sx, sy=None):
        """
        Add a scale in place.

        If *sy* is None, the same scale is applied in both the *x*- and
        *y*-directions.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        if sy is None:
            sy = sx
        # explicit element-wise scaling is fastest
        self._mtx[0, 0] *= sx
        self._mtx[0, 1] *= sx
        self._mtx[0, 2] *= sx
        self._mtx[1, 0] *= sy
        self._mtx[1, 1] *= sy
        self._mtx[1, 2] *= sy
        self.invalidate()
        return self

    def skew(self, xShear, yShear):
        """
        Add a skew in place.

        *xShear* and *yShear* are the shear angles along the *x*- and
        *y*-axes, respectively, in radians.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        rx = math.tan(xShear)
        ry = math.tan(yShear)
        mtx = self._mtx
        # Operating and assigning one scalar at a time is much faster.
        (xx, xy, x0), (yx, yy, y0), _ = mtx.tolist()
        # mtx = [[1 rx 0], [ry 1 0], [0 0 1]] * mtx
        mtx[0, 0] += rx * yx
        mtx[0, 1] += rx * yy
        mtx[0, 2] += rx * y0
        mtx[1, 0] += ry * xx
        mtx[1, 1] += ry * xy
        mtx[1, 2] += ry * x0
        self.invalidate()
        return self

    def skew_deg(self, xShear, yShear):
        """
        Add a skew in place.

        *xShear* and *yShear* are the shear angles along the *x*- and
        *y*-axes, respectively, in degrees.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        return self.skew(math.radians(xShear), math.radians(yShear))


class IdentityTransform(Affine2DBase):
    """
    A special class that does one thing, the identity transform, in a
    fast way.
    """
    _mtx = np.identity(3)

    def frozen(self):
        # docstring inherited
        return self

    __str__ = _make_str_method()

    def get_matrix(self):
        # docstring inherited
        return self._mtx

    def transform(self, values):
        # docstring inherited
        return np.asanyarray(values)

    def transform_affine(self, values):
        # docstring inherited
        return np.asanyarray(values)

    def transform_non_affine(self, values):
        # docstring inherited
        return np.asanyarray(values)

    def transform_path(self, path):
        # docstring inherited
        return path

    def transform_path_affine(self, path):
        # docstring inherited
        return path

    def transform_path_non_affine(self, path):
        # docstring inherited
        return path

    def get_affine(self):
        # docstring inherited
        return self

    def inverted(self):
        # docstring inherited
        return self


class _BlendedMixin:
    """Common methods for `BlendedGenericTransform` and `BlendedAffine2D`."""

    def __eq__(self, other):
        if isinstance(other, (BlendedAffine2D, BlendedGenericTransform)):
            return (self._x == other._x) and (self._y == other._y)
        elif self._x == self._y:
            return self._x == other
        else:
            return NotImplemented

    def contains_branch_seperately(self, transform):
        return (self._x.contains_branch(transform),
                self._y.contains_branch(transform))

    __str__ = _make_str_method("_x", "_y")


class BlendedGenericTransform(_BlendedMixin, Transform):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This "generic" version can handle any given child transform in the
    *x*- and *y*-directions.
    """
    input_dims = 2
    output_dims = 2
    is_separable = True
    pass_through = True

    def __init__(self, x_transform, y_transform, **kwargs):
        """
        Create a new "blended" transform using *x_transform* to transform the
        *x*-axis and *y_transform* to transform the *y*-axis.

        You will generally not call this constructor directly but use the
        `blended_transform_factory` function instead, which can determine
        automatically which kind of blended transform to create.
        """
        Transform.__init__(self, **kwargs)
        self._x = x_transform
        self._y = y_transform
        self.set_children(x_transform, y_transform)
        self._affine = None

    @property
    def depth(self):
        return max(self._x.depth, self._y.depth)

    def contains_branch(self, other):
        # A blended transform cannot possibly contain a branch from two
        # different transforms.
        return False

    is_affine = property(lambda self: self._x.is_affine and self._y.is_affine)
    has_inverse = property(
        lambda self: self._x.has_inverse and self._y.has_inverse)

    def frozen(self):
        # docstring inherited
        return blended_transform_factory(self._x.frozen(), self._y.frozen())

    def transform_non_affine(self, values):
        # docstring inherited
        if self._x.is_affine and self._y.is_affine:
            return values
        x = self._x
        y = self._y

        if x == y and x.input_dims == 2:
            return x.transform_non_affine(values)

        if x.input_dims == 2:
            x_points = x.transform_non_affine(values)[:, 0:1]
        else:
            x_points = x.transform_non_affine(values[:, 0])
            x_points = x_points.reshape((len(x_points), 1))

        if y.input_dims == 2:
            y_points = y.transform_non_affine(values)[:, 1:]
        else:
            y_points = y.transform_non_affine(values[:, 1])
            y_points = y_points.reshape((len(y_points), 1))

        if (isinstance(x_points, np.ma.MaskedArray) or
                isinstance(y_points, np.ma.MaskedArray)):
            return np.ma.concatenate((x_points, y_points), 1)
        else:
            return np.concatenate((x_points, y_points), 1)

    def inverted(self):
        # docstring inherited
        return BlendedGenericTransform(self._x.inverted(), self._y.inverted())

    def get_affine(self):
        # docstring inherited
        if self._invalid or self._affine is None:
            if self._x == self._y:
                self._affine = self._x.get_affine()
            else:
                x_mtx = self._x.get_affine().get_matrix()
                y_mtx = self._y.get_affine().get_matrix()
                # We already know the transforms are separable, so we can skip
                # setting b and c to zero.
                mtx = np.array([x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]])
                self._affine = Affine2D(mtx)
            self._invalid = 0
        return self._affine


class BlendedAffine2D(_BlendedMixin, Affine2DBase):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This version is an optimization for the case where both child
    transforms are of type `Affine2DBase`.
    """

    is_separable = True

    def __init__(self, x_transform, y_transform, **kwargs):
        """
        Create a new "blended" transform using *x_transform* to transform the
        *x*-axis and *y_transform* to transform the *y*-axis.

        Both *x_transform* and *y_transform* must be 2D affine transforms.

        You will generally not call this constructor directly but use the
        `blended_transform_factory` function instead, which can determine
        automatically which kind of blended transform to create.
        """
        is_affine = x_transform.is_affine and y_transform.is_affine
        is_separable = x_transform.is_separable and y_transform.is_separable
        is_correct = is_affine and is_separable
        if not is_correct:
            raise ValueError("Both *x_transform* and *y_transform* must be 2D "
                             "affine transforms")

        Transform.__init__(self, **kwargs)
        self._x = x_transform
        self._y = y_transform
        self.set_children(x_transform, y_transform)

        Affine2DBase.__init__(self)
        self._mtx = None

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            if self._x == self._y:
                self._mtx = self._x.get_matrix()
            else:
                x_mtx = self._x.get_matrix()
                y_mtx = self._y.get_matrix()
                # We already know the transforms are separable, so we can skip
                # setting b and c to zero.
                self._mtx = np.array([x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]])
            self._inverted = None
            self._invalid = 0
        return self._mtx


def blended_transform_factory(x_transform, y_transform):
    """
    Create a new "blended" transform using *x_transform* to transform
    the *x*-axis and *y_transform* to transform the *y*-axis.

    A faster version of the blended transform is returned for the case
    where both child transforms are affine.
    """
    if (isinstance(x_transform, Affine2DBase) and
            isinstance(y_transform, Affine2DBase)):
        return BlendedAffine2D(x_transform, y_transform)
    return BlendedGenericTransform(x_transform, y_transform)


class CompositeGenericTransform(Transform):
    """
    A composite transform formed by applying transform *a* then
    transform *b*.

    This "generic" version can handle any two arbitrary
    transformations.
    """
    pass_through = True

    def __init__(self, a, b, **kwargs):
        """
        Create a new composite transform that is the result of
        applying transform *a* then transform *b*.

        You will generally not call this constructor directly but write ``a +
        b`` instead, which will automatically choose the best kind of composite
        transform instance to create.
        """
        if a.output_dims != b.input_dims:
            raise ValueError("The output dimension of 'a' must be equal to "
                             "the input dimensions of 'b'")
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims

        super().__init__(**kwargs)
        self._a = a
        self._b = b
        self.set_children(a, b)

    def frozen(self):
        # docstring inherited
        self._invalid = 0
        frozen = composite_transform_factory(
            self._a.frozen(), self._b.frozen())
        if not isinstance(frozen, CompositeGenericTransform):
            return frozen.frozen()
        return frozen

    def _invalidate_internal(self, level, invalidating_node):
        # When the left child is invalidated at AFFINE_ONLY level and the right child is
        # non-affine, the composite transform is FULLY invalidated.
        if invalidating_node is self._a and not self._b.is_affine:
            level = Transform._INVALID_FULL
        super()._invalidate_internal(level, invalidating_node)

    def __eq__(self, other):
        if isinstance(other, (CompositeGenericTransform, CompositeAffine2D)):
            return self is other or (self._a == other._a
                                     and self._b == other._b)
        else:
            return False

    def _iter_break_from_left_to_right(self):
        for left, right in self._a._iter_break_from_left_to_right():
            yield left, right + self._b
        for left, right in self._b._iter_break_from_left_to_right():
            yield self._a + left, right

    def contains_branch_seperately(self, other_transform):
        # docstring inherited
        if self.output_dims != 2:
            raise ValueError('contains_branch_seperately only supports '
                             'transforms with 2 output dimensions')
        if self == other_transform:
            return (True, True)
        return self._b.contains_branch_seperately(other_transform)

    depth = property(lambda self: self._a.depth + self._b.depth)
    is_affine = property(lambda self: self._a.is_affine and self._b.is_affine)
    is_separable = property(
        lambda self: self._a.is_separable and self._b.is_separable)
    has_inverse = property(
        lambda self: self._a.has_inverse and self._b.has_inverse)

    __str__ = _make_str_method("_a", "_b")

    def transform_affine(self, values):
        # docstring inherited
        return self.get_affine().transform(values)

    def transform_non_affine(self, values):
        # docstring inherited
        if self._a.is_affine and self._b.is_affine:
            return values
        elif not self._a.is_affine and self._b.is_affine:
            return self._a.transform_non_affine(values)
        else:
            return self._b.transform_non_affine(self._a.transform(values))

    def transform_path_non_affine(self, path):
        # docstring inherited
        if self._a.is_affine and self._b.is_affine:
            return path
        elif not self._a.is_affine and self._b.is_affine:
            return self._a.transform_path_non_affine(path)
        else:
            return self._b.transform_path_non_affine(
                                    self._a.transform_path(path))

    def get_affine(self):
        # docstring inherited
        if not self._b.is_affine:
            return self._b.get_affine()
        else:
            return Affine2D(np.dot(self._b.get_affine().get_matrix(),
                                   self._a.get_affine().get_matrix()))

    def inverted(self):
        # docstring inherited
        return CompositeGenericTransform(
            self._b.inverted(), self._a.inverted())


class CompositeAffine2D(Affine2DBase):
    """
    A composite transform formed by applying transform *a* then transform *b*.

    This version is an optimization that handles the case where both *a*
    and *b* are 2D affines.
    """
    def __init__(self, a, b, **kwargs):
        """
        Create a new composite transform that is the result of
        applying `Affine2DBase` *a* then `Affine2DBase` *b*.

        You will generally not call this constructor directly but write ``a +
        b`` instead, which will automatically choose the best kind of composite
        transform instance to create.
        """
        if not a.is_affine or not b.is_affine:
            raise ValueError("'a' and 'b' must be affine transforms")
        if a.output_dims != b.input_dims:
            raise ValueError("The output dimension of 'a' must be equal to "
                             "the input dimensions of 'b'")
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims

        super().__init__(**kwargs)
        self._a = a
        self._b = b
        self.set_children(a, b)
        self._mtx = None

    @property
    def depth(self):
        return self._a.depth + self._b.depth

    def _iter_break_from_left_to_right(self):
        for left, right in self._a._iter_break_from_left_to_right():
            yield left, right + self._b
        for left, right in self._b._iter_break_from_left_to_right():
            yield self._a + left, right

    __str__ = _make_str_method("_a", "_b")

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            self._mtx = np.dot(
                self._b.get_matrix(),
                self._a.get_matrix())
            self._inverted = None
            self._invalid = 0
        return self._mtx


def composite_transform_factory(a, b):
    """
    Create a new composite transform that is the result of applying
    transform a then transform b.

    Shortcut versions of the blended transform are provided for the
    case where both child transforms are affine, or one or the other
    is the identity transform.

    Composite transforms may also be created using the '+' operator,
    e.g.::

      c = a + b
    """
    # check to see if any of a or b are IdentityTransforms. We use
    # isinstance here to guarantee that the transforms will *always*
    # be IdentityTransforms. Since TransformWrappers are mutable,
    # use of equality here would be wrong.
    if isinstance(a, IdentityTransform):
        return b
    elif isinstance(b, IdentityTransform):
        return a
    elif isinstance(a, Affine2D) and isinstance(b, Affine2D):
        return CompositeAffine2D(a, b)
    return CompositeGenericTransform(a, b)


class BboxTransform(Affine2DBase):
    """
    `BboxTransform` linearly transforms points from one `Bbox` to another.
    """

    is_separable = True

    def __init__(self, boxin, boxout, **kwargs):
        """
        Create a new `BboxTransform` that linearly transforms
        points from *boxin* to *boxout*.
        """
        _api.check_isinstance(BboxBase, boxin=boxin, boxout=boxout)

        super().__init__(**kwargs)
        self._boxin = boxin
        self._boxout = boxout
        self.set_children(boxin, boxout)
        self._mtx = None
        self._inverted = None

    __str__ = _make_str_method("_boxin", "_boxout")

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            inl, inb, inw, inh = self._boxin.bounds
            outl, outb, outw, outh = self._boxout.bounds
            x_scale = outw / inw
            y_scale = outh / inh
            if DEBUG and (x_scale == 0 or y_scale == 0):
                raise ValueError(
                    "Transforming from or to a singular bounding box")
            self._mtx = np.array([[x_scale,     0.0, -inl*x_scale+outl],
                                  [    0.0, y_scale, -inb*y_scale+outb],
                                  [    0.0,     0.0,               1.0]],
                                 float)
            self._inverted = None
            self._invalid = 0
        return self._mtx


class BboxTransformTo(Affine2DBase):
    """
    `BboxTransformTo` is a transformation that linearly transforms points from
    the unit bounding box to a given `Bbox`.
    """

    is_separable = True

    def __init__(self, boxout, **kwargs):
        """
        Create a new `BboxTransformTo` that linearly transforms
        points from the unit bounding box to *boxout*.
        """
        _api.check_isinstance(BboxBase, boxout=boxout)

        super().__init__(**kwargs)
        self._boxout = boxout
        self.set_children(boxout)
        self._mtx = None
        self._inverted = None

    __str__ = _make_str_method("_boxout")

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            outl, outb, outw, outh = self._boxout.bounds
            if DEBUG and (outw == 0 or outh == 0):
                raise ValueError("Transforming to a singular bounding box.")
            self._mtx = np.array([[outw,  0.0, outl],
                                  [ 0.0, outh, outb],
                                  [ 0.0,  0.0,  1.0]],
                                 float)
            self._inverted = None
            self._invalid = 0
        return self._mtx


@_api.deprecated("3.9")
class BboxTransformToMaxOnly(BboxTransformTo):
    """
    `BboxTransformToMaxOnly` is a transformation that linearly transforms points from
    the unit bounding box to a given `Bbox` with a fixed upper left of (0, 0).
    """
    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            xmax, ymax = self._boxout.max
            if DEBUG and (xmax == 0 or ymax == 0):
                raise ValueError("Transforming to a singular bounding box.")
            self._mtx = np.array([[xmax,  0.0, 0.0],
                                  [ 0.0, ymax, 0.0],
                                  [ 0.0,  0.0, 1.0]],
                                 float)
            self._inverted = None
            self._invalid = 0
        return self._mtx


class BboxTransformFrom(Affine2DBase):
    """
    `BboxTransformFrom` linearly transforms points from a given `Bbox` to the
    unit bounding box.
    """
    is_separable = True

    def __init__(self, boxin, **kwargs):
        _api.check_isinstance(BboxBase, boxin=boxin)

        super().__init__(**kwargs)
        self._boxin = boxin
        self.set_children(boxin)
        self._mtx = None
        self._inverted = None

    __str__ = _make_str_method("_boxin")

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            inl, inb, inw, inh = self._boxin.bounds
            if DEBUG and (inw == 0 or inh == 0):
                raise ValueError("Transforming from a singular bounding box.")
            x_scale = 1.0 / inw
            y_scale = 1.0 / inh
            self._mtx = np.array([[x_scale,     0.0, -inl*x_scale],
                                  [    0.0, y_scale, -inb*y_scale],
                                  [    0.0,     0.0,          1.0]],
                                 float)
            self._inverted = None
            self._invalid = 0
        return self._mtx


class ScaledTranslation(Affine2DBase):
    """
    A transformation that translates by *xt* and *yt*, after *xt* and *yt*
    have been transformed by *scale_trans*.
    """
    def __init__(self, xt, yt, scale_trans, **kwargs):
        super().__init__(**kwargs)
        self._t = (xt, yt)
        self._scale_trans = scale_trans
        self.set_children(scale_trans)
        self._mtx = None
        self._inverted = None

    __str__ = _make_str_method("_t")

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            # A bit faster than np.identity(3).
            self._mtx = IdentityTransform._mtx.copy()
            self._mtx[:2, 2] = self._scale_trans.transform(self._t)
            self._invalid = 0
            self._inverted = None
        return self._mtx


class _ScaledRotation(Affine2DBase):
    """
    A transformation that applies rotation by *theta*, after transform by *trans_shift*.
    """
    def __init__(self, theta, trans_shift):
        super().__init__()
        self._theta = theta
        self._trans_shift = trans_shift
        self._mtx = None

    def get_matrix(self):
        if self._invalid:
            transformed_coords = self._trans_shift.transform([[self._theta, 0]])[0]
            adjusted_theta = transformed_coords[0]
            rotation = Affine2D().rotate(adjusted_theta)
            self._mtx = rotation.get_matrix()
        return self._mtx


class AffineDeltaTransform(Affine2DBase):
    r"""
    A transform wrapper for transforming displacements between pairs of points.

    This class is intended to be used to transform displacements ("position
    deltas") between pairs of points (e.g., as the ``offset_transform``
    of `.Collection`\s): given a transform ``t`` such that ``t =
    AffineDeltaTransform(t) + offset``, ``AffineDeltaTransform``
    satisfies ``AffineDeltaTransform(a - b) == AffineDeltaTransform(a) -
    AffineDeltaTransform(b)``.

    This is implemented by forcing the offset components of the transform
    matrix to zero.

    This class is experimental as of 3.3, and the API may change.
    """

    pass_through = True

    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)
        self._base_transform = transform
        self.set_children(transform)

    __str__ = _make_str_method("_base_transform")

    def get_matrix(self):
        if self._invalid:
            self._mtx = self._base_transform.get_matrix().copy()
            self._mtx[:2, -1] = 0
        return self._mtx


class TransformedPath(TransformNode):
    """
    A `TransformedPath` caches a non-affine transformed copy of the
    `~.path.Path`.  This cached copy is automatically updated when the
    non-affine part of the transform changes.

    .. note::

        Paths are considered immutable by this class. Any update to the
        path's vertices/codes will not trigger a transform recomputation.

    """
    def __init__(self, path, transform):
        """
        Parameters
        ----------
        path : `~.path.Path`
        transform : `Transform`
        """
        _api.check_isinstance(Transform, transform=transform)
        super().__init__()
        self._path = path
        self._transform = transform
        self.set_children(transform)
        self._transformed_path = None
        self._transformed_points = None

    def _revalidate(self):
        # only recompute if the invalidation includes the non_affine part of
        # the transform
        if (self._invalid == self._INVALID_FULL
                or self._transformed_path is None):
            self._transformed_path = \
                self._transform.transform_path_non_affine(self._path)
            self._transformed_points = \
                Path._fast_from_codes_and_verts(
                    self._transform.transform_non_affine(self._path.vertices),
                    None, self._path)
        self._invalid = 0

    def get_transformed_points_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.  Unlike
        :meth:`get_transformed_path_and_affine`, no interpolation will
        be performed.
        """
        self._revalidate()
        return self._transformed_points, self.get_affine()

    def get_transformed_path_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.
        """
        self._revalidate()
        return self._transformed_path, self.get_affine()

    def get_fully_transformed_path(self):
        """
        Return a fully-transformed copy of the child path.
        """
        self._revalidate()
        return self._transform.transform_path_affine(self._transformed_path)

    def get_affine(self):
        return self._transform.get_affine()


class TransformedPatchPath(TransformedPath):
    """
    A `TransformedPatchPath` caches a non-affine transformed copy of the
    `~.patches.Patch`. This cached copy is automatically updated when the
    non-affine part of the transform or the patch changes.
    """

    def __init__(self, patch):
        """
        Parameters
        ----------
        patch : `~.patches.Patch`
        """
        # Defer to TransformedPath.__init__.
        super().__init__(patch.get_path(), patch.get_transform())
        self._patch = patch

    def _revalidate(self):
        patch_path = self._patch.get_path()
        # Force invalidation if the patch path changed; otherwise, let base
        # class check invalidation.
        if patch_path != self._path:
            self._path = patch_path
            self._transformed_path = None
        super()._revalidate()


def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    """
    Modify the endpoints of a range as needed to avoid singularities.

    Parameters
    ----------
    vmin, vmax : float
        The initial endpoints.
    expander : float, default: 0.001
        Fractional amount by which *vmin* and *vmax* are expanded if
        the original interval is too small, based on *tiny*.
    tiny : float, default: 1e-15
        Threshold for the ratio of the interval to the maximum absolute
        value of its endpoints.  If the interval is smaller than
        this, it will be expanded.  This value should be around
        1e-15 or larger; otherwise the interval will be approaching
        the double precision resolution limit.
    increasing : bool, default: True
        If True, swap *vmin*, *vmax* if *vmin* > *vmax*.

    Returns
    -------
    vmin, vmax : float
        Endpoints, expanded and/or swapped if necessary.
        If either input is inf or NaN, or if both inputs are 0 or very
        close to zero, it returns -*expander*, *expander*.
    """

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        return -expander, expander

    swapped = False
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        swapped = True

    # Expand vmin, vmax to float: if they were integer types, they can wrap
    # around in abs (abs(np.int8(-128)) == -128) and vmax - vmin can overflow.
    vmin, vmax = map(float, [vmin, vmax])

    maxabsvalue = max(abs(vmin), abs(vmax))
    if maxabsvalue < (1e6 / tiny) * np.finfo(float).tiny:
        vmin = -expander
        vmax = expander

    elif vmax - vmin <= maxabsvalue * tiny:
        if vmax == 0 and vmin == 0:
            vmin = -expander
            vmax = expander
        else:
            vmin -= expander*abs(vmin)
            vmax += expander*abs(vmax)

    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax


def interval_contains(interval, val):
    """
    Check, inclusively, whether an interval includes a given value.

    Parameters
    ----------
    interval : (float, float)
        The endpoints of the interval.
    val : float
        Value to check is within interval.

    Returns
    -------
    bool
        Whether *val* is within the *interval*.
    """
    a, b = interval
    if a > b:
        a, b = b, a
    return a <= val <= b


def _interval_contains_close(interval, val, rtol=1e-10):
    """
    Check, inclusively, whether an interval includes a given value, with the
    interval expanded by a small tolerance to admit floating point errors.

    Parameters
    ----------
    interval : (float, float)
        The endpoints of the interval.
    val : float
        Value to check is within interval.
    rtol : float, default: 1e-10
        Relative tolerance slippage allowed outside of the interval.
        For an interval ``[a, b]``, values
        ``a - rtol * (b - a) <= val <= b + rtol * (b - a)`` are considered
        inside the interval.

    Returns
    -------
    bool
        Whether *val* is within the *interval* (with tolerance).
    """
    a, b = interval
    if a > b:
        a, b = b, a
    rtol = (b - a) * rtol
    return a - rtol <= val <= b + rtol


def interval_contains_open(interval, val):
    """
    Check, excluding endpoints, whether an interval includes a given value.

    Parameters
    ----------
    interval : (float, float)
        The endpoints of the interval.
    val : float
        Value to check is within interval.

    Returns
    -------
    bool
        Whether *val* is within the *interval*.
    """
    a, b = interval
    return a < val < b or a > val > b


def offset_copy(trans, fig=None, x=0.0, y=0.0, units='inches'):
    """
    Return a new transform with an added offset.

    Parameters
    ----------
    trans : `Transform` subclass
        Any transform, to which offset will be applied.
    fig : `~matplotlib.figure.Figure`, default: None
        Current figure. It can be None if *units* are 'dots'.
    x, y : float, default: 0.0
        The offset to apply.
    units : {'inches', 'points', 'dots'}, default: 'inches'
        Units of the offset.

    Returns
    -------
    `Transform` subclass
        Transform with applied offset.
    """
    _api.check_in_list(['dots', 'points', 'inches'], units=units)
    if units == 'dots':
        return trans + Affine2D().translate(x, y)
    if fig is None:
        raise ValueError('For units of inches or points a fig kwarg is needed')
    if units == 'points':
        x /= 72.0
        y /= 72.0
    # Default units are 'inches'
