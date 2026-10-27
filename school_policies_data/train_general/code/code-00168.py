    alpha_xu_pos = np.broadcast_to(np.atleast_1d(alpha_xu_pos), xpt.shape[1])

    for k in range(xpt.shape[1]):
        # Set alpha_tr to the step size for the trust-region constraint.
        if s_norm[k] > TINY * delta:
            alpha_tr = max(delta / s_norm[k], 0.0)
        else:
            # The current straight line is basically zero.
            continue

        alpha_bd_pos = max(min(alpha_xu_pos[k], alpha_xl_neg[k]), 0.0)
        alpha_bd_neg = min(max(alpha_xl_pos[k], alpha_xu_neg[k]), 0.0)

        # Set alpha_quad_pos and alpha_quad_neg to the step size to the extrema
        # of the quadratic function along the positive and negative directions.
        grad_step = grad @ xpt[:, k]
        curv_step = curv(xpt[:, k])
        if (
            grad_step >= 0.0
            and curv_step < -TINY * grad_step
            or grad_step <= 0.0
            and curv_step > -TINY * grad_step
        ):
            alpha_quad_pos = max(-grad_step / curv_step, 0.0)
        else:
            alpha_quad_pos = np.inf
        if (
            grad_step >= 0.0
            and curv_step > TINY * grad_step
            or grad_step <= 0.0
            and curv_step < TINY * grad_step
        ):
            alpha_quad_neg = min(-grad_step / curv_step, 0.0)
        else:
            alpha_quad_neg = -np.inf

        # Select the step that provides the largest value of the objective
        # function if it improves the current best. The best positive step is
        # either the one that reaches the constraints or the one that reaches
        # the extremum of the objective function along the current direction
        # (only possible if the resulting step is feasible). We test both, and
        # we perform similar calculations along the negative step.
        # N.B.: we select the largest possible step among all the ones that
        # maximize the objective function. This is to avoid returning the zero
        # step in some extreme cases.
        alpha_pos = min(alpha_tr, alpha_bd_pos)
        alpha_neg = max(-alpha_tr, alpha_bd_neg)
        q_val_pos = (
            const + alpha_pos * grad_step + 0.5 * alpha_pos**2.0 * curv_step
        )
        q_val_neg = (
            const + alpha_neg * grad_step + 0.5 * alpha_neg**2.0 * curv_step
        )
        if alpha_quad_pos < alpha_pos:
            q_val_quad_pos = (
                const
                + alpha_quad_pos * grad_step
                + 0.5 * alpha_quad_pos**2.0 * curv_step
            )
            if abs(q_val_quad_pos) > abs(q_val_pos):
                alpha_pos = alpha_quad_pos
                q_val_pos = q_val_quad_pos
        if alpha_quad_neg > alpha_neg:
            q_val_quad_neg = (
                const
                + alpha_quad_neg * grad_step
                + 0.5 * alpha_quad_neg**2.0 * curv_step
            )
            if abs(q_val_quad_neg) > abs(q_val_neg):
                alpha_neg = alpha_quad_neg
                q_val_neg = q_val_quad_neg
        if abs(q_val_pos) >= abs(q_val_neg) and abs(q_val_pos) > abs(q_val):
            step = np.clip(alpha_pos * xpt[:, k], xl, xu)
            q_val = q_val_pos
        elif abs(q_val_neg) > abs(q_val_pos) and abs(q_val_neg) > abs(q_val):
            step = np.clip(alpha_neg * xpt[:, k], xl, xu)
            q_val = q_val_neg

    if debug:
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.linalg.norm(step) < 1.1 * delta
    return step


def _cauchy_geom(const, grad, curv, xl, xu, delta, debug):
    """
    Same as `bound_constrained_cauchy_step` without the absolute value.
    """
    # Calculate the initial active set.
    fixed_xl = (xl < 0.0) & (grad > 0.0)
    fixed_xu = (xu > 0.0) & (grad < 0.0)

    # Calculate the Cauchy step.
    cauchy_step = np.zeros_like(grad)
    cauchy_step[fixed_xl] = xl[fixed_xl]
    cauchy_step[fixed_xu] = xu[fixed_xu]
    if np.linalg.norm(cauchy_step) > delta:
        working = fixed_xl | fixed_xu
        while True:
            # Calculate the Cauchy step for the directions in the working set.
            g_norm = np.linalg.norm(grad[working])
            delta_reduced = np.sqrt(
                delta**2.0 - cauchy_step[~working] @ cauchy_step[~working]
            )
            if g_norm > TINY * abs(delta_reduced):
                mu = max(delta_reduced / g_norm, 0.0)
            else:
                break
            cauchy_step[working] = mu * grad[working]

            # Update the working set.
            fixed_xl = working & (cauchy_step < xl)
            fixed_xu = working & (cauchy_step > xu)
            if not np.any(fixed_xl) and not np.any(fixed_xu):
                # Stop the calculations as the Cauchy step is now feasible.
                break
            cauchy_step[fixed_xl] = xl[fixed_xl]
            cauchy_step[fixed_xu] = xu[fixed_xu]
            working = working & ~(fixed_xl | fixed_xu)

    # Calculate the step that maximizes the quadratic along the Cauchy step.
    grad_step = grad @ cauchy_step
    if grad_step >= 0.0:
        # Set alpha_tr to the step size for the trust-region constraint.
        s_norm = np.linalg.norm(cauchy_step)
        if s_norm > TINY * delta:
            alpha_tr = max(delta / s_norm, 0.0)
        else:
            # The Cauchy step is basically zero.
            alpha_tr = 0.0

        # Set alpha_quad to the step size for the maximization problem.
        curv_step = curv(cauchy_step)
        if curv_step < -TINY * grad_step:
            alpha_quad = max(-grad_step / curv_step, 0.0)
        else:
            alpha_quad = np.inf

        # Set alpha_bd to the step size for the bound constraints.
        i_xl = (xl > -np.inf) & (cauchy_step < TINY * xl)
        i_xu = (xu < np.inf) & (cauchy_step > TINY * xu)
        alpha_xl = np.min(xl[i_xl] / cauchy_step[i_xl], initial=np.inf)
        alpha_xu = np.min(xu[i_xu] / cauchy_step[i_xu], initial=np.inf)
        alpha_bd = min(alpha_xl, alpha_xu)

        # Calculate the solution and the corresponding function value.
        alpha = min(alpha_tr, alpha_quad, alpha_bd)
        step = np.clip(alpha * cauchy_step, xl, xu)
        q_val = const + alpha * grad_step + 0.5 * alpha**2.0 * curv_step
    else:
        # This case is never reached in exact arithmetic. It prevents this
        # function to return a step that decreases the objective function.
        step = np.zeros_like(grad)
        q_val = const

    if debug:
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.linalg.norm(step) < 1.1 * delta
    return step, q_val
<|endoftext|>from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Array, ModuleType

__all__ = ["atleast_nd", "cov", "create_diagonal", "expand_dims", "kron", "sinc"]


def atleast_nd(x: Array, /, *, ndim: int, xp: ModuleType) -> Array:
    """
    Recursively expand the dimension of an array to at least `ndim`.

    Parameters
    ----------
    x : array
    ndim : int
        The minimum number of dimensions for the result.
    xp : array_namespace
        The standard-compatible namespace for `x`.

    Returns
    -------
    res : array
        An array with ``res.ndim`` >= `ndim`.
        If ``x.ndim`` >= `ndim`, `x` is returned.
        If ``x.ndim`` < `ndim`, `x` is expanded by prepending new axes
        until ``res.ndim`` equals `ndim`.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([1])
    >>> xpx.atleast_nd(x, ndim=3, xp=xp)
    Array([[[1]]], dtype=array_api_strict.int64)

    >>> x = xp.asarray([[[1, 2],
    ...                  [3, 4]]])
    >>> xpx.atleast_nd(x, ndim=1, xp=xp) is x
    True

    """
    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        x = atleast_nd(x, ndim=ndim, xp=xp)
    return x


def cov(m: Array, /, *, xp: ModuleType) -> Array:
    """
    Estimate a covariance matrix.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    This provides a subset of the functionality of ``numpy.cov``.

    Parameters
    ----------
    m : array
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables.
    xp : array_namespace
        The standard-compatible namespace for `m`.

    Returns
    -------
    res : array
        The covariance matrix of the variables.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx

    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = xp.asarray([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    Array([[0, 1, 2],
           [2, 1, 0]], dtype=array_api_strict.int64)

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> xpx.cov(x, xp=xp)
    Array([[ 1., -1.],
           [-1.,  1.]], dtype=array_api_strict.float64)


    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = xp.asarray([-2.1, -1,  4.3])
    >>> y = xp.asarray([3,  1.1,  0.12])
    >>> X = xp.stack((x, y), axis=0)
    >>> xpx.cov(X, xp=xp)
    Array([[11.71      , -4.286     ],
           [-4.286     ,  2.14413333]], dtype=array_api_strict.float64)

    >>> xpx.cov(x, xp=xp)
    Array(11.71, dtype=array_api_strict.float64)

    >>> xpx.cov(y, xp=xp)
    Array(2.14413333, dtype=array_api_strict.float64)

    """
    m = xp.asarray(m, copy=True)
    dtype = (
        xp.float64 if xp.isdtype(m.dtype, "integral") else xp.result_type(m, xp.float64)
    )

    m = atleast_nd(m, ndim=2, xp=xp)
    m = xp.astype(m, dtype)

    avg = _mean(m, axis=1, xp=xp)
    fact = m.shape[1] - 1

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        fact = 0.0

    m -= avg[:, None]
    m_transpose = m.T
    if xp.isdtype(m_transpose.dtype, "complex floating"):
        m_transpose = xp.conj(m_transpose)
    c = m @ m_transpose
    c /= fact
    axes = tuple(axis for axis, length in enumerate(c.shape) if length == 1)
    return xp.squeeze(c, axis=axes)


def create_diagonal(x: Array, /, *, offset: int = 0, xp: ModuleType) -> Array:
    """
    Construct a diagonal array.

    Parameters
    ----------
    x : array
        A 1-D array
    offset : int, optional
        Offset from the leading diagonal (default is ``0``).
        Use positive ints for diagonals above the leading diagonal,
        and negative ints for diagonals below the leading diagonal.
    xp : array_namespace
        The standard-compatible namespace for `x`.

    Returns
    -------
    res : array
        A 2-D array with `x` on the diagonal (offset by `offset`).

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([2, 4, 8])

    >>> xpx.create_diagonal(x, xp=xp)
    Array([[2, 0, 0],
           [0, 4, 0],
           [0, 0, 8]], dtype=array_api_strict.int64)

    >>> xpx.create_diagonal(x, offset=-2, xp=xp)
    Array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [2, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 8, 0, 0]], dtype=array_api_strict.int64)

    """
    if x.ndim != 1:
        err_msg = "`x` must be 1-dimensional."
        raise ValueError(err_msg)
    n = x.shape[0] + abs(offset)
    diag = xp.zeros(n**2, dtype=x.dtype)
    i = offset if offset >= 0 else abs(offset) * n
    diag[i : min(n * (n - offset), diag.shape[0]) : n + 1] = x
    return xp.reshape(diag, (n, n))


def _mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    xp: ModuleType,
) -> Array:
    """
    Complex mean, https://github.com/data-apis/array-api/issues/846.
    """
    if xp.isdtype(x.dtype, "complex floating"):
        x_real = xp.real(x)
        x_imag = xp.imag(x)
        mean_real = xp.mean(x_real, axis=axis, keepdims=keepdims)
        mean_imag = xp.mean(x_imag, axis=axis, keepdims=keepdims)
        return mean_real + (mean_imag * xp.asarray(1j))
    return xp.mean(x, axis=axis, keepdims=keepdims)


def expand_dims(
    a: Array, /, *, axis: int | tuple[int, ...] = (0,), xp: ModuleType
) -> Array:
    """
    Expand the shape of an array.

    Insert (a) new axis/axes that will appear at the position(s) specified by
    `axis` in the expanded array shape.

    This is ``xp.expand_dims`` for `axis` an int *or a tuple of ints*.
    Roughly equivalent to ``numpy.expand_dims`` for NumPy arrays.

    Parameters
    ----------
    a : array
    axis : int or tuple of ints, optional
        Position(s) in the expanded axes where the new axis (or axes) is/are placed.
        If multiple positions are provided, they should be unique (note that a position
        given by a positive index could also be referred to by a negative index -
        that will also result in an error).
        Default: ``(0,)``.
    xp : array_namespace
        The standard-compatible namespace for `a`.

    Returns
    -------
    res : array
        `a` with an expanded shape.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([1, 2])
    >>> x.shape
    (2,)

    The following is equivalent to ``x[xp.newaxis, :]`` or ``x[xp.newaxis]``:

    >>> y = xpx.expand_dims(x, axis=0, xp=xp)
    >>> y
    Array([[1, 2]], dtype=array_api_strict.int64)
    >>> y.shape
    (1, 2)

    The following is equivalent to ``x[:, xp.newaxis]``:

    >>> y = xpx.expand_dims(x, axis=1, xp=xp)
    >>> y
    Array([[1],
           [2]], dtype=array_api_strict.int64)
    >>> y.shape
    (2, 1)

    ``axis`` may also be a tuple:

    >>> y = xpx.expand_dims(x, axis=(0, 1), xp=xp)
    >>> y
    Array([[[1, 2]]], dtype=array_api_strict.int64)

    >>> y = xpx.expand_dims(x, axis=(2, 0), xp=xp)
    >>> y
    Array([[[1],
            [2]]], dtype=array_api_strict.int64)

    """
    if not isinstance(axis, tuple):
        axis = (axis,)
    ndim = a.ndim + len(axis)
    if axis != () and (min(axis) < -ndim or max(axis) >= ndim):
        err_msg = (
            f"a provided axis position is out of bounds for array of dimension {a.ndim}"
        )
        raise IndexError(err_msg)
    axis = tuple(dim % ndim for dim in axis)
    if len(set(axis)) != len(axis):
        err_msg = "Duplicate dimensions specified in `axis`."
        raise ValueError(err_msg)
    for i in sorted(axis):
        a = xp.expand_dims(a, axis=i)
    return a


def kron(a: Array, b: Array, /, *, xp: ModuleType) -> Array:
    """
    Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.

    Equivalent to ``numpy.kron`` for NumPy arrays.

    Parameters
    ----------
    a, b : array
    xp : array_namespace
        The standard-compatible namespace for `a` and `b`.

    Returns
    -------
    res : array
        The Kronecker product of `a` and `b`.

    Notes
    -----
    The function assumes that the number of dimensions of `a` and `b`
    are the same, if necessary prepending the smallest with ones.
    If ``a.shape = (r0,r1,..,rN)`` and ``b.shape = (s0,s1,...,sN)``,
    the Kronecker product has shape ``(r0*s0, r1*s1, ..., rN*SN)``.
    The elements are products of elements from `a` and `b`, organized
    explicitly by::

        kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]

    where::

        kt = it * st + jt,  t = 0,...,N

    In the common 2-D case (N=1), the block structure can be visualized::

        [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],
         [  ...                              ...   ],
         [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]


    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> xpx.kron(xp.asarray([1, 10, 100]), xp.asarray([5, 6, 7]), xp=xp)
    Array([  5,   6,   7,  50,  60,  70, 500,
           600, 700], dtype=array_api_strict.int64)

    >>> xpx.kron(xp.asarray([5, 6, 7]), xp.asarray([1, 10, 100]), xp=xp)
    Array([  5,  50, 500,   6,  60, 600,   7,
            70, 700], dtype=array_api_strict.int64)

    >>> xpx.kron(xp.eye(2), xp.ones((2, 2)), xp=xp)
    Array([[1., 1., 0., 0.],
           [1., 1., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 1., 1.]], dtype=array_api_strict.float64)


    >>> a = xp.reshape(xp.arange(100), (2, 5, 2, 5))
    >>> b = xp.reshape(xp.arange(24), (2, 3, 4))
    >>> c = xpx.kron(a, b, xp=xp)
    >>> c.shape
    (2, 10, 6, 20)
    >>> I = (1, 3, 0, 2)
    >>> J = (0, 2, 1)
    >>> J1 = (0,) + J             # extend to ndim=4
    >>> S1 = (1,) + b.shape
    >>> K = tuple(xp.asarray(I) * xp.asarray(S1) + xp.asarray(J1))
    >>> c[K] == a[I]*b[J]
    Array(True, dtype=array_api_strict.bool)

    """

    b = xp.asarray(b)
    singletons = (1,) * (b.ndim - a.ndim)
    a = xp.broadcast_to(xp.asarray(a), singletons + a.shape)

    nd_b, nd_a = b.ndim, a.ndim
    nd_max = max(nd_b, nd_a)
    if nd_a == 0 or nd_b == 0:
        return xp.multiply(a, b)

    a_shape = a.shape
    b_shape = b.shape

    # Equalise the shapes by prepending smaller one with 1s
    a_shape = (1,) * max(0, nd_b - nd_a) + a_shape
    b_shape = (1,) * max(0, nd_a - nd_b) + b_shape

    # Insert empty dimensions
    a_arr = expand_dims(a, axis=tuple(range(nd_b - nd_a)), xp=xp)
    b_arr = expand_dims(b, axis=tuple(range(nd_a - nd_b)), xp=xp)

    # Compute the product
    a_arr = expand_dims(a_arr, axis=tuple(range(1, nd_max * 2, 2)), xp=xp)
    b_arr = expand_dims(b_arr, axis=tuple(range(0, nd_max * 2, 2)), xp=xp)
    result = xp.multiply(a_arr, b_arr)

    # Reshape back and return
    a_shape = xp.asarray(a_shape)
    b_shape = xp.asarray(b_shape)
    return xp.reshape(result, tuple(xp.multiply(a_shape, b_shape)))


def sinc(x: Array, /, *, xp: ModuleType) -> Array:
    r"""
    Return the normalized sinc function.

    The sinc function is equal to :math:`\sin(\pi x)/(\pi x)` for any argument
    :math:`x\ne 0`. ``sinc(0)`` takes the limit value 1, making ``sinc`` not
    only everywhere continuous but also infinitely differentiable.

    .. note::

        Note the normalization factor of ``pi`` used in the definition.
        This is the most commonly used definition in signal processing.
        Use ``sinc(x / xp.pi)`` to obtain the unnormalized sinc function
        :math:`\sin(x)/x` that is more common in mathematics.

    Parameters
    ----------
    x : array
        Array (possibly multi-dimensional) of values for which to calculate
        ``sinc(x)``. Must have a real floating point dtype.
    xp : array_namespace
        The standard-compatible namespace for `x`.

    Returns
    -------
    res : array
        ``sinc(x)`` calculated elementwise, which has the same shape as the input.

    Notes
    -----
    The name sinc is short for "sine cardinal" or "sinus cardinalis".

    The sinc function is used in various signal processing applications,
    including in anti-aliasing, in the construction of a Lanczos resampling
    filter, and in interpolation.

    For bandlimited interpolation of discrete-time signals, the ideal
    interpolation kernel is proportional to the sinc function.

    References
    ----------
    .. [1] Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web
           Resource. https://mathworld.wolfram.com/SincFunction.html
    .. [2] Wikipedia, "Sinc function",
           https://en.wikipedia.org/wiki/Sinc_function

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.linspace(-4, 4, 41)
    >>> xpx.sinc(x, xp=xp)
    Array([-3.89817183e-17, -4.92362781e-02,
           -8.40918587e-02, -8.90384387e-02,
           -5.84680802e-02,  3.89817183e-17,
            6.68206631e-02,  1.16434881e-01,
            1.26137788e-01,  8.50444803e-02,
           -3.89817183e-17, -1.03943254e-01,
           -1.89206682e-01, -2.16236208e-01,
           -1.55914881e-01,  3.89817183e-17,
            2.33872321e-01,  5.04551152e-01,
            7.56826729e-01,  9.35489284e-01,
            1.00000000e+00,  9.35489284e-01,
            7.56826729e-01,  5.04551152e-01,
            2.33872321e-01,  3.89817183e-17,
           -1.55914881e-01, -2.16236208e-01,
           -1.89206682e-01, -1.03943254e-01,
           -3.89817183e-17,  8.50444803e-02,
            1.26137788e-01,  1.16434881e-01,
            6.68206631e-02,  3.89817183e-17,
           -5.84680802e-02, -8.90384387e-02,
           -8.40918587e-02, -4.92362781e-02,
           -3.89817183e-17], dtype=array_api_strict.float64)

    """
    if not xp.isdtype(x.dtype, "real floating"):
        err_msg = "`x` must have a real floating data type."
        raise ValueError(err_msg)
    # no scalars in `where` - array-api#807
    y = xp.pi * xp.where(
        x, x, xp.asarray(xp.finfo(x.dtype).smallest_normal, dtype=x.dtype)
    )
    return xp.sin(y) / y
<|endoftext|>"""
Internal helpers
"""

from functools import wraps
from inspect import signature

def get_xp(xp):
    """
    Decorator to automatically replace xp with the corresponding array module.

    Use like

    import numpy as np

    @get_xp(np)
    def func(x, /, xp, kwarg=None):
        return xp.func(x, kwarg=kwarg)

    Note that xp must be a keyword argument and come after all non-keyword
    arguments.

    """

    def inner(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            return f(*args, xp=xp, **kwargs)

        sig = signature(f)
        new_sig = sig.replace(
            parameters=[sig.parameters[i] for i in sig.parameters if i != "xp"]
        )

        if wrapped_f.__doc__ is None:
            wrapped_f.__doc__ = f"""\
Array API compatibility wrapper for {f.__name__}.

See the corresponding documentation in NumPy/CuPy and/or the array API
specification for more details.

"""
        wrapped_f.__signature__ = new_sig
        return wrapped_f

    return inner
<|endoftext|>from numpy.linalg import * # noqa: F403
from numpy.linalg import __all__ as linalg_all
import numpy as _np

from ..common import _linalg
from .._internal import get_xp

# These functions are in both the main and linalg namespaces
from ._aliases import matmul, matrix_transpose, tensordot, vecdot # noqa: F401

import numpy as np

cross = get_xp(np)(_linalg.cross)
outer = get_xp(np)(_linalg.outer)
EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
eigh = get_xp(np)(_linalg.eigh)
qr = get_xp(np)(_linalg.qr)
slogdet = get_xp(np)(_linalg.slogdet)
svd = get_xp(np)(_linalg.svd)
cholesky = get_xp(np)(_linalg.cholesky)
matrix_rank = get_xp(np)(_linalg.matrix_rank)
pinv = get_xp(np)(_linalg.pinv)
matrix_norm = get_xp(np)(_linalg.matrix_norm)
svdvals = get_xp(np)(_linalg.svdvals)
diagonal = get_xp(np)(_linalg.diagonal)
trace = get_xp(np)(_linalg.trace)

# Note: unlike np.linalg.solve, the array API solve() only accepts x2 as a
# vector when it is exactly 1-dimensional. All other cases treat x2 as a stack
# of matrices. The np.linalg.solve behavior of allowing stacks of both
# matrices and vectors is ambiguous c.f.
# https://github.com/numpy/numpy/issues/15349 and
# https://github.com/data-apis/array-api/issues/285.

# To workaround this, the below is the code from np.linalg.solve except
# only calling solve1 in the exactly 1D case.

# This code is here instead of in common because it is numpy specific. Also
# note that CuPy's solve() does not currently support broadcasting (see
# https://github.com/cupy/cupy/blob/main/cupy/cublas.py#L43).
def solve(x1: _np.ndarray, x2: _np.ndarray, /) -> _np.ndarray:
    try:
        from numpy.linalg._linalg import (
        _makearray, _assert_stacked_2d, _assert_stacked_square,
        _commonType, isComplexType, _raise_linalgerror_singular
        )
    except ImportError:
        from numpy.linalg.linalg import (
        _makearray, _assert_stacked_2d, _assert_stacked_square,
        _commonType, isComplexType, _raise_linalgerror_singular
        )
    from numpy.linalg import _umath_linalg

    x1, _ = _makearray(x1)
    _assert_stacked_2d(x1)
    _assert_stacked_square(x1)
    x2, wrap = _makearray(x2)
    t, result_t = _commonType(x1, x2)

    # This part is different from np.linalg.solve
    if x2.ndim == 1:
        gufunc = _umath_linalg.solve1
    else:
        gufunc = _umath_linalg.solve

    # This does nothing currently but is left in because it will be relevant
    # when complex dtype support is added to the spec in 2022.
    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    with _np.errstate(call=_raise_linalgerror_singular, invalid='call',
                      over='ignore', divide='ignore', under='ignore'):
        r = gufunc(x1, x2, signature=signature)

    return wrap(r.astype(result_t, copy=False))

# These functions are completely new here. If the library already has them
# (i.e., numpy 2.0), use the library version instead of our wrapper.
if hasattr(np.linalg, 'vector_norm'):
    vector_norm = np.linalg.vector_norm
else:
    vector_norm = get_xp(np)(_linalg.vector_norm)

__all__ = linalg_all + _linalg.__all__ + ['solve']

del get_xp
del np
del linalg_all
del _linalg
<|endoftext|>from __future__ import annotations

__all__ = [
    "ndarray",
    "Device",
    "Dtype",
]

import sys
from typing import (
    Literal,
    Union,
    TYPE_CHECKING,
)

from numpy import (
    ndarray,
    dtype,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
)

Device = Literal["cpu"]
if TYPE_CHECKING or sys.version_info >= (3, 9):
    Dtype = dtype[Union[
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
    ]]
else:
    Dtype = dtype
<|endoftext|>from numpy import * # noqa: F403

# from numpy import * doesn't overwrite these builtin names
from numpy import abs, max, min, round # noqa: F401

# These imports may overwrite names from the import * above.
from ._aliases import * # noqa: F403

# Don't know why, but we have to do an absolute import to import linalg. If we
# instead do
#
# from . import linalg
#
# It doesn't overwrite np.linalg from above. The import is generated
# dynamically so that the library can be vendored.
__import__(__package__ + '.linalg')

__import__(__package__ + '.fft')

from .linalg import matrix_transpose, vecdot # noqa: F401

from ..common._helpers import * # noqa: F403

try:
    # Used in asarray(). Not present in older versions.
    from numpy import _CopyMode # noqa: F401
except ImportError:
    pass

__array_api_version__ = '2023.12'
<|endoftext|>from numpy.fft import * # noqa: F403
from numpy.fft import __all__ as fft_all

from ..common import _fft
from .._internal import get_xp

import numpy as np

fft = get_xp(np)(_fft.fft)
ifft = get_xp(np)(_fft.ifft)
fftn = get_xp(np)(_fft.fftn)
ifftn = get_xp(np)(_fft.ifftn)
rfft = get_xp(np)(_fft.rfft)
irfft = get_xp(np)(_fft.irfft)
rfftn = get_xp(np)(_fft.rfftn)
irfftn = get_xp(np)(_fft.irfftn)
hfft = get_xp(np)(_fft.hfft)
ihfft = get_xp(np)(_fft.ihfft)
fftfreq = get_xp(np)(_fft.fftfreq)
rfftfreq = get_xp(np)(_fft.rfftfreq)
fftshift = get_xp(np)(_fft.fftshift)
ifftshift = get_xp(np)(_fft.ifftshift)

__all__ = fft_all + _fft.__all__

del get_xp
del np
del fft_all
del _fft
<|endoftext|>"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""
from numpy import (
    dtype,
    bool_ as bool,
    intp,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
)


class __array_namespace_info__:
    """
    Get the array API inspection namespace for NumPy.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for NumPy.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': numpy.float64,
     'complex floating': numpy.complex128,
     'integral': numpy.int64,
     'indexing': numpy.int64}

    """

    __module__ = 'numpy'

    def capabilities(self):
        """
        Return a dictionary of array API library capabilities.

        The resulting dictionary has the following keys:

        - **"boolean indexing"**: boolean indicating whether an array library
          supports boolean indexing. Always ``True`` for NumPy.

        - **"data-dependent shapes"**: boolean indicating whether an array
          library supports data-dependent output shapes. Always ``True`` for
          NumPy.

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.

        See Also
        --------
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        capabilities : dict
            A dictionary of array API library capabilities.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.capabilities()
        {'boolean indexing': True,
         'data-dependent shapes': True}

        """
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            # 'max rank' will be part of the 2024.12 standard
            # "max rank": 64,
        }

    def default_device(self):
        """
        The default device used for new NumPy arrays.

        For NumPy, this always returns ``'cpu'``.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        device : str
            The default device used for new NumPy arrays.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_device()
        'cpu'

        """
        return "cpu"

    def default_dtypes(self, *, device=None):
        """
        The default data types used for new NumPy arrays.

        For NumPy, this always returns the following dictionary:

        - **"real floating"**: ``numpy.float64``
        - **"complex floating"**: ``numpy.complex128``
        - **"integral"**: ``numpy.intp``
        - **"indexing"**: ``numpy.intp``

        Parameters
        ----------
        device : str, optional
            The device to get the default data types for. For NumPy, only
            ``'cpu'`` is allowed.

        Returns
        -------
        dtypes : dict
            A dictionary describing the default data types used for new NumPy
            arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_dtypes()
        {'real floating': numpy.float64,
         'complex floating': numpy.complex128,
         'integral': numpy.int64,
         'indexing': numpy.int64}

        """
        if device not in ["cpu", None]:
            raise ValueError(
                'Device not understood. Only "cpu" is allowed, but received:'
                f' {device}'
            )
        return {
            "real floating": dtype(float64),
            "complex floating": dtype(complex128),
            "integral": dtype(intp),
            "indexing": dtype(intp),
        }

    def dtypes(self, *, device=None, kind=None):
        """
        The array API data types supported by NumPy.

        Note that this function only returns data types that are defined by
        the array API.

        Parameters
        ----------
        device : str, optional
            The device to get the data types for. For NumPy, only ``'cpu'`` is
            allowed.
        kind : str or tuple of str, optional
            The kind of data types to return. If ``None``, all data types are
            returned. If a string, only data types of that kind are returned.
            If a tuple, a dictionary containing the union of the given kinds
            is returned. The following kinds are supported:

            - ``'bool'``: boolean data types (i.e., ``bool``).
            - ``'signed integer'``: signed integer data types (i.e., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (i.e.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (i.e., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (i.e.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

        Returns
        -------
        dtypes : dict
            A dictionary mapping the names of data types to the corresponding
            NumPy data types.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.dtypes(kind='signed integer')
        {'int8': numpy.int8,
         'int16': numpy.int16,
         'int32': numpy.int32,
         'int64': numpy.int64}

        """
        if device not in ["cpu", None]:
            raise ValueError(
                'Device not understood. Only "cpu" is allowed, but received:'
                f' {device}'
            )
        if kind is None:
            return {
                "bool": dtype(bool),
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
            }
        if kind == "unsigned integer":
            return {
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "integral":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "real floating":
            return {
                "float32": dtype(float32),
                "float64": dtype(float64),
            }
        if kind == "complex floating":
            return {
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "numeric":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self):
        """
        The devices supported by NumPy.

        For NumPy, this always returns ``['cpu']``.

        Returns
        -------
        devices : list of str
            The devices supported by NumPy.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.devices()
        ['cpu']

        """
        return ["cpu"]
<|endoftext|>from __future__ import annotations

from ..common import _aliases

from .._internal import get_xp

from ._info import __array_namespace_info__

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Union
    from ._typing import ndarray, Device, Dtype, NestedSequence, SupportsBufferProtocol

import numpy as np
bool = np.bool_

# Basic renames
acos = np.arccos
acosh = np.arccosh
asin = np.arcsin
asinh = np.arcsinh
atan = np.arctan
atan2 = np.arctan2
atanh = np.arctanh
bitwise_left_shift = np.left_shift
bitwise_invert = np.invert
bitwise_right_shift = np.right_shift
concat = np.concatenate
pow = np.power

arange = get_xp(np)(_aliases.arange)
empty = get_xp(np)(_aliases.empty)
empty_like = get_xp(np)(_aliases.empty_like)
eye = get_xp(np)(_aliases.eye)
full = get_xp(np)(_aliases.full)
full_like = get_xp(np)(_aliases.full_like)
linspace = get_xp(np)(_aliases.linspace)
ones = get_xp(np)(_aliases.ones)
ones_like = get_xp(np)(_aliases.ones_like)
zeros = get_xp(np)(_aliases.zeros)
zeros_like = get_xp(np)(_aliases.zeros_like)
UniqueAllResult = get_xp(np)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(np)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(np)(_aliases.UniqueInverseResult)
unique_all = get_xp(np)(_aliases.unique_all)
unique_counts = get_xp(np)(_aliases.unique_counts)
unique_inverse = get_xp(np)(_aliases.unique_inverse)
unique_values = get_xp(np)(_aliases.unique_values)
astype = _aliases.astype
std = get_xp(np)(_aliases.std)
var = get_xp(np)(_aliases.var)
cumulative_sum = get_xp(np)(_aliases.cumulative_sum)
clip = get_xp(np)(_aliases.clip)
permute_dims = get_xp(np)(_aliases.permute_dims)
reshape = get_xp(np)(_aliases.reshape)
argsort = get_xp(np)(_aliases.argsort)
sort = get_xp(np)(_aliases.sort)
nonzero = get_xp(np)(_aliases.nonzero)
ceil = get_xp(np)(_aliases.ceil)
floor = get_xp(np)(_aliases.floor)
trunc = get_xp(np)(_aliases.trunc)
matmul = get_xp(np)(_aliases.matmul)
matrix_transpose = get_xp(np)(_aliases.matrix_transpose)
tensordot = get_xp(np)(_aliases.tensordot)
sign = get_xp(np)(_aliases.sign)

def _supports_buffer_protocol(obj):
    try:
        memoryview(obj)
    except TypeError:
        return False
    return True

# asarray also adds the copy keyword, which is not present in numpy 1.0.
# asarray() is different enough between numpy, cupy, and dask, the logic
# complicated enough that it's easier to define it separately for each module
# rather than trying to combine everything into one function in common/
def asarray(
    obj: Union[
        ndarray,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsBufferProtocol,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: "Optional[Union[bool, np._CopyMode]]" = None,
    **kwargs,
) -> ndarray:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device for NumPy: {device!r}")

    if hasattr(np, '_CopyMode'):
        if copy is None:
            copy = np._CopyMode.IF_NEEDED
        elif copy is False:
            copy = np._CopyMode.NEVER
        elif copy is True:
            copy = np._CopyMode.ALWAYS
    else:
        # Not present in older NumPys. In this case, we cannot really support
        # copy=False.
        if copy is False:
            raise NotImplementedError("asarray(copy=False) requires a newer version of NumPy.")

    return np.array(obj, copy=copy, dtype=dtype, **kwargs)

# These functions are completely new here. If the library already has them
# (i.e., numpy 2.0), use the library version instead of our wrapper.
if hasattr(np, 'vecdot'):
    vecdot = np.vecdot
else:
    vecdot = get_xp(np)(_aliases.vecdot)

if hasattr(np, 'isdtype'):
    isdtype = np.isdtype
else:
    isdtype = get_xp(np)(_aliases.isdtype)

if hasattr(np, 'unstack'):
    unstack = np.unstack
else:
    unstack = get_xp(np)(_aliases.unstack)

__all__ = _aliases.__all__ + ['__array_namespace_info__', 'asarray', 'bool',
                              'acos', 'acosh', 'asin', 'asinh', 'atan',
                              'atan2', 'atanh', 'bitwise_left_shift',
                              'bitwise_invert', 'bitwise_right_shift',
                              'concat', 'pow']

_all_ignore = ['np', 'get_xp']
<|endoftext|>from __future__ import annotations

__all__ = [
    "NestedSequence",
    "SupportsBufferProtocol",
]

from typing import (
    Any,
    TypeVar,
    Protocol,
)

_T_co = TypeVar("_T_co", covariant=True)

class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...

SupportsBufferProtocol = Any

Array = Any
Device = Any
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
if TYPE_CHECKING:
    from typing import Literal, Optional, Tuple, Union
    from ._typing import ndarray

import math

import numpy as np
if np.__version__[0] == "2":
    from numpy.lib.array_utils import normalize_axis_tuple
else:
    from numpy.core.numeric import normalize_axis_tuple

from ._aliases import matmul, matrix_transpose, tensordot, vecdot, isdtype
from .._internal import get_xp

# These are in the main NumPy namespace but not in numpy.linalg
def cross(x1: ndarray, x2: ndarray, /, xp, *, axis: int = -1, **kwargs) -> ndarray:
    return xp.cross(x1, x2, axis=axis, **kwargs)

def outer(x1: ndarray, x2: ndarray, /, xp, **kwargs) -> ndarray:
    return xp.outer(x1, x2, **kwargs)

class EighResult(NamedTuple):
    eigenvalues: ndarray
    eigenvectors: ndarray

class QRResult(NamedTuple):
    Q: ndarray
    R: ndarray

class SlogdetResult(NamedTuple):
    sign: ndarray
    logabsdet: ndarray

class SVDResult(NamedTuple):
    U: ndarray
    S: ndarray
    Vh: ndarray

# These functions are the same as their NumPy counterparts except they return
# a namedtuple.
def eigh(x: ndarray, /, xp, **kwargs) -> EighResult:
    return EighResult(*xp.linalg.eigh(x, **kwargs))

def qr(x: ndarray, /, xp, *, mode: Literal['reduced', 'complete'] = 'reduced',
       **kwargs) -> QRResult:
    return QRResult(*xp.linalg.qr(x, mode=mode, **kwargs))

def slogdet(x: ndarray, /, xp, **kwargs) -> SlogdetResult:
    return SlogdetResult(*xp.linalg.slogdet(x, **kwargs))

def svd(x: ndarray, /, xp, *, full_matrices: bool = True, **kwargs) -> SVDResult:
    return SVDResult(*xp.linalg.svd(x, full_matrices=full_matrices, **kwargs))

# These functions have additional keyword arguments

# The upper keyword argument is new from NumPy
def cholesky(x: ndarray, /, xp, *, upper: bool = False, **kwargs) -> ndarray:
    L = xp.linalg.cholesky(x, **kwargs)
    if upper:
        U = get_xp(xp)(matrix_transpose)(L)
        if get_xp(xp)(isdtype)(U.dtype, 'complex floating'):
            U = xp.conj(U)
        return U
    return L

# The rtol keyword argument of matrix_rank() and pinv() is new from NumPy.
# Note that it has a different semantic meaning from tol and rcond.
def matrix_rank(x: ndarray,
                /,
                xp,
                *,
                rtol: Optional[Union[float, ndarray]] = None,
                **kwargs) -> ndarray:
    # this is different from xp.linalg.matrix_rank, which supports 1
    # dimensional arrays.
    if x.ndim < 2:
        raise xp.linalg.LinAlgError("1-dimensional array given. Array must be at least two-dimensional")
    S = get_xp(xp)(svdvals)(x, **kwargs)
    if rtol is None:
        tol = S.max(axis=-1, keepdims=True) * max(x.shape[-2:]) * xp.finfo(S.dtype).eps
    else:
        # this is different from xp.linalg.matrix_rank, which does not
        # multiply the tolerance by the largest singular value.
        tol = S.max(axis=-1, keepdims=True)*xp.asarray(rtol)[..., xp.newaxis]
    return xp.count_nonzero(S > tol, axis=-1)

def pinv(x: ndarray, /, xp, *, rtol: Optional[Union[float, ndarray]] = None, **kwargs) -> ndarray:
    # this is different from xp.linalg.pinv, which does not multiply the
    # default tolerance by max(M, N).
    if rtol is None:
        rtol = max(x.shape[-2:]) * xp.finfo(x.dtype).eps
    return xp.linalg.pinv(x, rcond=rtol, **kwargs)

# These functions are new in the array API spec

def matrix_norm(x: ndarray, /, xp, *, keepdims: bool = False, ord: Optional[Union[int, float, Literal['fro', 'nuc']]] = 'fro') -> ndarray:
    return xp.linalg.norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)

# svdvals is not in NumPy (but it is in SciPy). It is equivalent to
# xp.linalg.svd(compute_uv=False).
def svdvals(x: ndarray, /, xp) -> Union[ndarray, Tuple[ndarray, ...]]:
    return xp.linalg.svd(x, compute_uv=False)

def vector_norm(x: ndarray, /, xp, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ord: Optional[Union[int, float]] = 2) -> ndarray:
    # xp.linalg.norm tries to do a matrix norm whenever axis is a 2-tuple or
    # when axis=None and the input is 2-D, so to force a vector norm, we make
    # it so the input is 1-D (for axis=None), or reshape so that norm is done
    # on a single dimension.
    if axis is None:
        # Note: xp.linalg.norm() doesn't handle 0-D arrays
        _x = x.ravel()
        _axis = 0
    elif isinstance(axis, tuple):
        # Note: The axis argument supports any number of axes, whereas
        # xp.linalg.norm() only supports a single axis for vector norm.
        normalized_axis = normalize_axis_tuple(axis, x.ndim)
        rest = tuple(i for i in range(x.ndim) if i not in normalized_axis)
        newshape = axis + rest
        _x = xp.transpose(x, newshape).reshape(
            (math.prod([x.shape[i] for i in axis]), *[x.shape[i] for i in rest]))
        _axis = 0
    else:
        _x = x
        _axis = axis

    res = xp.linalg.norm(_x, axis=_axis, ord=ord)

    if keepdims:
        # We can't reuse xp.linalg.norm(keepdims) because of the reshape hacks
        # above to avoid matrix norm logic.
        shape = list(x.shape)
        _axis = normalize_axis_tuple(range(x.ndim) if axis is None else axis, x.ndim)
        for i in _axis:
            shape[i] = 1
        res = xp.reshape(res, tuple(shape))

    return res

# xp.diagonal and xp.trace operate on the first two axes whereas these
# operates on the last two

def diagonal(x: ndarray, /, xp, *, offset: int = 0, **kwargs) -> ndarray:
    return xp.diagonal(x, offset=offset, axis1=-2, axis2=-1, **kwargs)

def trace(x: ndarray, /, xp, *, offset: int = 0, dtype=None, **kwargs) -> ndarray:
    return xp.asarray(xp.trace(x, offset=offset, dtype=dtype, axis1=-2, axis2=-1, **kwargs))

__all__ = ['cross', 'matmul', 'outer', 'tensordot', 'EighResult',
           'QRResult', 'SlogdetResult', 'SVDResult', 'eigh', 'qr', 'slogdet',
           'svd', 'cholesky', 'matrix_rank', 'pinv', 'matrix_norm',
           'matrix_transpose', 'svdvals', 'vecdot', 'vector_norm', 'diagonal',
           'trace']
<|endoftext|>"""
Various helper functions which are not part of the spec.

Functions which start with an underscore are for internal use only but helpers
that are in __all__ are intended as additional helper functions for use by end
users of the compat library.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Union, Any
    from ._typing import Array, Device

import sys
import math
import inspect
import warnings

def _is_jax_zero_gradient_array(x):
    """Return True if `x` is a zero-gradient array.

    These arrays are a design quirk of Jax that may one day be removed.
    See https://github.com/google/jax/issues/20620.
    """
    if 'numpy' not in sys.modules or 'jax' not in sys.modules:
        return False

    import numpy as np
    import jax

    return isinstance(x, np.ndarray) and x.dtype == jax.float0

def is_numpy_array(x):
    """
    Return True if `x` is a NumPy array.

    This function does not import NumPy if it has not already been imported
    and is therefore cheap to use.

    This also returns True for `ndarray` subclasses and NumPy scalar objects.

    See Also
    --------

    array_namespace
    is_array_api_obj
    is_cupy_array
    is_torch_array
    is_ndonnx_array
    is_dask_array
    is_jax_array
    is_pydata_sparse_array
    """
    # Avoid importing NumPy if it isn't already
    if 'numpy' not in sys.modules:
        return False

    import numpy as np

    # TODO: Should we reject ndarray subclasses?
    return (isinstance(x, (np.ndarray, np.generic))
            and not _is_jax_zero_gradient_array(x))

def is_cupy_array(x):
    """
    Return True if `x` is a CuPy array.

    This function does not import CuPy if it has not already been imported
    and is therefore cheap to use.

    This also returns True for `cupy.ndarray` subclasses and CuPy scalar objects.

    See Also
    --------

    array_namespace
    is_array_api_obj
    is_numpy_array
    is_torch_array
    is_ndonnx_array
    is_dask_array
    is_jax_array
    is_pydata_sparse_array
    """
    # Avoid importing CuPy if it isn't already
    if 'cupy' not in sys.modules:
        return False

    import cupy as cp

    # TODO: Should we reject ndarray subclasses?
    return isinstance(x, (cp.ndarray, cp.generic))

def is_torch_array(x):
    """
    Return True if `x` is a PyTorch tensor.

    This function does not import PyTorch if it has not already been imported
    and is therefore cheap to use.

    See Also
    --------

    array_namespace
    is_array_api_obj
    is_numpy_array
    is_cupy_array
    is_dask_array
    is_jax_array
    is_pydata_sparse_array
    """
    # Avoid importing torch if it isn't already
    if 'torch' not in sys.modules:
        return False

    import torch

    # TODO: Should we reject ndarray subclasses?
    return isinstance(x, torch.Tensor)

def is_ndonnx_array(x):
    """
    Return True if `x` is a ndonnx Array.

    This function does not import ndonnx if it has not already been imported
    and is therefore cheap to use.

    See Also
    --------

    array_namespace
    is_array_api_obj
    is_numpy_array
    is_cupy_array
    is_ndonnx_array
    is_dask_array
    is_jax_array
    is_pydata_sparse_array
    """
    # Avoid importing torch if it isn't already
    if 'ndonnx' not in sys.modules:
        return False

    import ndonnx as ndx

    return isinstance(x, ndx.Array)

def is_dask_array(x):
    """
    Return True if `x` is a dask.array Array.

    This function does not import dask if it has not already been imported
    and is therefore cheap to use.

    See Also
    --------

    array_namespace
    is_array_api_obj
    is_numpy_array
    is_cupy_array
    is_torch_array
    is_ndonnx_array
    is_jax_array
    is_pydata_sparse_array
    """
    # Avoid importing dask if it isn't already
    if 'dask.array' not in sys.modules:
        return False

    import dask.array

    return isinstance(x, dask.array.Array)

def is_jax_array(x):
    """
    Return True if `x` is a JAX array.

    This function does not import JAX if it has not already been imported
    and is therefore cheap to use.


    See Also
    --------

    array_namespace
    is_array_api_obj
    is_numpy_array
    is_cupy_array
    is_torch_array
    is_ndonnx_array
    is_dask_array
    is_pydata_sparse_array
    """
    # Avoid importing jax if it isn't already
    if 'jax' not in sys.modules:
        return False

    import jax

    return isinstance(x, jax.Array) or _is_jax_zero_gradient_array(x)

def is_pydata_sparse_array(x) -> bool:
    """
    Return True if `x` is an array from the `sparse` package.

    This function does not import `sparse` if it has not already been imported
    and is therefore cheap to use.


    See Also
    --------

    array_namespace
    is_array_api_obj
    is_numpy_array
    is_cupy_array
    is_torch_array
    is_ndonnx_array
    is_dask_array
    is_jax_array
    """
    # Avoid importing jax if it isn't already
    if 'sparse' not in sys.modules:
        return False

    import sparse

    # TODO: Account for other backends.
    return isinstance(x, sparse.SparseArray)

def is_array_api_obj(x):
    """
    Return True if `x` is an array API compatible array object.

    See Also
    --------

    array_namespace
    is_numpy_array
    is_cupy_array
    is_torch_array
    is_ndonnx_array
    is_dask_array
    is_jax_array
    """
    return is_numpy_array(x) \
        or is_cupy_array(x) \
        or is_torch_array(x) \
        or is_dask_array(x) \
        or is_jax_array(x) \
        or is_pydata_sparse_array(x) \
        or hasattr(x, '__array_namespace__')

def _compat_module_name():
    assert __name__.endswith('.common._helpers')
    return __name__.removesuffix('.common._helpers')

def is_numpy_namespace(xp) -> bool:
    """
    Returns True if `xp` is a NumPy namespace.

    This includes both NumPy itself and the version wrapped by array-api-compat.

    See Also
    --------

    array_namespace
    is_cupy_namespace
    is_torch_namespace
    is_ndonnx_namespace
    is_dask_namespace
    is_jax_namespace
    is_pydata_sparse_namespace
    is_array_api_strict_namespace
    """
    return xp.__name__ in {'numpy', _compat_module_name() + '.numpy'}

def is_cupy_namespace(xp) -> bool:
    """
    Returns True if `xp` is a CuPy namespace.

    This includes both CuPy itself and the version wrapped by array-api-compat.

    See Also
    --------

    array_namespace
    is_numpy_namespace
    is_torch_namespace
    is_ndonnx_namespace
    is_dask_namespace
    is_jax_namespace
    is_pydata_sparse_namespace
    is_array_api_strict_namespace
    """
    return xp.__name__ in {'cupy', _compat_module_name() + '.cupy'}

def is_torch_namespace(xp) -> bool:
    """
    Returns True if `xp` is a PyTorch namespace.

    This includes both PyTorch itself and the version wrapped by array-api-compat.

    See Also
    --------

    array_namespace
    is_numpy_namespace
    is_cupy_namespace
    is_ndonnx_namespace
    is_dask_namespace
    is_jax_namespace
    is_pydata_sparse_namespace
    is_array_api_strict_namespace
    """
    return xp.__name__ in {'torch', _compat_module_name() + '.torch'}


def is_ndonnx_namespace(xp):
    """
    Returns True if `xp` is an NDONNX namespace.

    See Also
    --------

    array_namespace
    is_numpy_namespace
    is_cupy_namespace
    is_torch_namespace
    is_dask_namespace
    is_jax_namespace
    is_pydata_sparse_namespace
    is_array_api_strict_namespace
    """
    return xp.__name__ == 'ndonnx'

def is_dask_namespace(xp):
    """
    Returns True if `xp` is a Dask namespace.

    This includes both ``dask.array`` itself and the version wrapped by array-api-compat.

    See Also
    --------

    array_namespace
    is_numpy_namespace
    is_cupy_namespace
    is_torch_namespace
    is_ndonnx_namespace
    is_jax_namespace
    is_pydata_sparse_namespace
    is_array_api_strict_namespace
    """
    return xp.__name__ in {'dask.array', _compat_module_name() + '.dask.array'}

def is_jax_namespace(xp):
    """
    Returns True if `xp` is a JAX namespace.

    This includes ``jax.numpy`` and ``jax.experimental.array_api`` which existed in
    older versions of JAX.

    See Also
    --------

    array_namespace
    is_numpy_namespace
    is_cupy_namespace
    is_torch_namespace
    is_ndonnx_namespace
    is_dask_namespace
    is_pydata_sparse_namespace
    is_array_api_strict_namespace
    """
    return xp.__name__ in {'jax.numpy', 'jax.experimental.array_api'}

def is_pydata_sparse_namespace(xp):
    """
    Returns True if `xp` is a pydata/sparse namespace.

    See Also
    --------

    array_namespace
    is_numpy_namespace
    is_cupy_namespace
    is_torch_namespace
    is_ndonnx_namespace
    is_dask_namespace
    is_jax_namespace
    is_array_api_strict_namespace
    """
    return xp.__name__ == 'sparse'

def is_array_api_strict_namespace(xp):
    """
    Returns True if `xp` is an array-api-strict namespace.

    See Also
    --------

    array_namespace
    is_numpy_namespace
    is_cupy_namespace
    is_torch_namespace
    is_ndonnx_namespace
    is_dask_namespace
    is_jax_namespace
    is_pydata_sparse_namespace
    """
    return xp.__name__ == 'array_api_strict'

def _check_api_version(api_version):
    if api_version in ['2021.12', '2022.12']:
        warnings.warn(f"The {api_version} version of the array API specification was requested but the returned namespace is actually version 2023.12")
    elif api_version is not None and api_version not in ['2021.12', '2022.12',
                                                         '2023.12']:
        raise ValueError("Only the 2023.12 version of the array API specification is currently supported")

def array_namespace(*xs, api_version=None, use_compat=None):
    """
    Get the array API compatible namespace for the arrays `xs`.

    Parameters
    ----------
    xs: arrays
        one or more arrays.

    api_version: str
        The newest version of the spec that you need support for (currently
        the compat library wrapped APIs support v2023.12).

    use_compat: bool or None
        If None (the default), the native namespace will be returned if it is
        already array API compatible, otherwise a compat wrapper is used. If
        True, the compat library wrapped library will be returned. If False,
        the native library namespace is returned.

    Returns
    -------

    out: namespace
        The array API compatible namespace corresponding to the arrays in `xs`.

    Raises
    ------
    TypeError
        If `xs` contains arrays from different array libraries or contains a
        non-array.


    Typical usage is to pass the arguments of a function to
    `array_namespace()` at the top of a function to get the corresponding
    array API namespace:

    .. code:: python

       def your_function(x, y):
           xp = array_api_compat.array_namespace(x, y)
           # Now use xp as the array library namespace
           return xp.mean(x, axis=0) + 2*xp.std(y, axis=0)


    Wrapped array namespaces can also be imported directly. For example,
    `array_namespace(np.array(...))` will return `array_api_compat.numpy`.
    This function will also work for any array library not wrapped by
    array-api-compat if it explicitly defines `__array_namespace__
    <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html>`__
    (the wrapped namespace is always preferred if it exists).

    See Also
    --------

    is_array_api_obj
    is_numpy_array
    is_cupy_array
    is_torch_array
    is_dask_array
    is_jax_array
    is_pydata_sparse_array

    """
    if use_compat not in [None, True, False]:
        raise ValueError("use_compat must be None, True, or False")

    _use_compat = use_compat in [None, True]

    namespaces = set()
    for x in xs:
        if is_numpy_array(x):
            from .. import numpy as numpy_namespace
            import numpy as np
            if use_compat is True:
                _check_api_version(api_version)
                namespaces.add(numpy_namespace)
            elif use_compat is False:
                namespaces.add(np)
            else:
                # numpy 2.0+ have __array_namespace__, however, they are not yet fully array API
                # compatible.
                namespaces.add(numpy_namespace)
        elif is_cupy_array(x):
            if _use_compat:
                _check_api_version(api_version)
                from .. import cupy as cupy_namespace
                namespaces.add(cupy_namespace)
            else:
                import cupy as cp
                namespaces.add(cp)
        elif is_torch_array(x):
            if _use_compat:
                _check_api_version(api_version)
                from .. import torch as torch_namespace
                namespaces.add(torch_namespace)
            else:
                import torch
                namespaces.add(torch)
        elif is_dask_array(x):
            if _use_compat:
                _check_api_version(api_version)
                from ..dask import array as dask_namespace
                namespaces.add(dask_namespace)
            else:
                import dask.array as da
                namespaces.add(da)
        elif is_jax_array(x):
            if use_compat is True:
                _check_api_version(api_version)
                raise ValueError("JAX does not have an array-api-compat wrapper")
            elif use_compat is False:
                import jax.numpy as jnp
            else:
                # JAX v0.4.32 and newer implements the array API directly in jax.numpy.
                # For older JAX versions, it is available via jax.experimental.array_api.
                import jax.numpy
                if hasattr(jax.numpy, "__array_api_version__"):
                    jnp = jax.numpy
                else:
                    import jax.experimental.array_api as jnp
            namespaces.add(jnp)
        elif is_pydata_sparse_array(x):
            if use_compat is True:
                _check_api_version(api_version)
                raise ValueError("`sparse` does not have an array-api-compat wrapper")
            else:
                import sparse
            # `sparse` is already an array namespace. We do not have a wrapper
            # submodule for it.
            namespaces.add(sparse)
        elif hasattr(x, '__array_namespace__'):
            if use_compat is True:
                raise ValueError("The given array does not have an array-api-compat wrapper")
            namespaces.add(x.__array_namespace__(api_version=api_version))
        else:
            # TODO: Support Python scalars?
            raise TypeError(f"{type(x).__name__} is not a supported array type")

    if not namespaces:
        raise TypeError("Unrecognized array input")

    if len(namespaces) != 1:
        raise TypeError(f"Multiple namespaces for array inputs: {namespaces}")

    xp, = namespaces

    return xp

# backwards compatibility alias
get_namespace = array_namespace

def _check_device(xp, device):
    if xp == sys.modules.get('numpy'):
        if device not in ["cpu", None]:
            raise ValueError(f"Unsupported device for NumPy: {device!r}")

# Placeholder object to represent the dask device
# when the array backend is not the CPU.
# (since it is not easy to tell which device a dask array is on)
class _dask_device:
    def __repr__(self):
        return "DASK_DEVICE"

_DASK_DEVICE = _dask_device()

# device() is not on numpy.ndarray or dask.array and to_device() is not on numpy.ndarray
# or cupy.ndarray. They are not included in array objects of this library
# because this library just reuses the respective ndarray classes without
# wrapping or subclassing them. These helper functions can be used instead of
# the wrapper functions for libraries that need to support both NumPy/CuPy and
# other libraries that use devices.
def device(x: Array, /) -> Device:
    """
    Hardware device the array data resides on.

    This is equivalent to `x.device` according to the `standard
    <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.device.html>`__.
    This helper is included because some array libraries either do not have
    the `device` attribute or include it with an incompatible API.

    Parameters
    ----------
    x: array
        array instance from an array API compatible library.

    Returns
    -------
    out: device
        a ``device`` object (see the `Device Support <https://data-apis.org/array-api/latest/design_topics/device_support.html>`__
        section of the array API specification).

    Notes
    -----

    For NumPy the device is always `"cpu"`. For Dask, the device is always a
    special `DASK_DEVICE` object.

    See Also
    --------

    to_device : Move array data to a different device.

    """
    if is_numpy_array(x):
        return "cpu"
    elif is_dask_array(x):
        # Peek at the metadata of the jax array to determine type
        try:
            import numpy as np
            if isinstance(x._meta, np.ndarray):
                # Must be on CPU since backed by numpy
                return "cpu"
        except ImportError:
            pass
        return _DASK_DEVICE
    elif is_jax_array(x):
        # JAX has .device() as a method, but it is being deprecated so that it
        # can become a property, in accordance with the standard. In order for
        # this function to not break when JAX makes the flip, we check for
        # both here.
        if inspect.ismethod(x.device):
            return x.device()
        else:
            return x.device
    elif is_pydata_sparse_array(x):
        # `sparse` will gain `.device`, so check for this first.
        x_device = getattr(x, 'device', None)
        if x_device is not None:
            return x_device
        # Everything but DOK has this attr.
        try:
            inner = x.data
        except AttributeError:
            return "cpu"
        # Return the device of the constituent array
        return device(inner)
    return x.device

# Prevent shadowing, used below
_device = device

# Based on cupy.array_api.Array.to_device
def _cupy_to_device(x, device, /, stream=None):
    import cupy as cp
    from cupy.cuda import Device as _Device
    from cupy.cuda import stream as stream_module
    from cupy_backends.cuda.api import runtime

    if device == x.device:
        return x
    elif device == "cpu":
        # allowing us to use `to_device(x, "cpu")`
        # is useful for portable test swapping between
        # host and device backends
        return x.get()
    elif not isinstance(device, _Device):
        raise ValueError(f"Unsupported device {device!r}")
    else:
        # see cupy/cupy#5985 for the reason how we handle device/stream here
        prev_device = runtime.getDevice()
        prev_stream: stream_module.Stream = None
        if stream is not None:
            prev_stream = stream_module.get_current_stream()
            # stream can be an int as specified in __dlpack__, or a CuPy stream
            if isinstance(stream, int):
                stream = cp.cuda.ExternalStream(stream)
            elif isinstance(stream, cp.cuda.Stream):
                pass
            else:
                raise ValueError('the input stream is not recognized')
            stream.use()
        try:
            runtime.setDevice(device.id)
            arr = x.copy()
        finally:
            runtime.setDevice(prev_device)
            if stream is not None:
                prev_stream.use()
        return arr

def _torch_to_device(x, device, /, stream=None):
    if stream is not None:
        raise NotImplementedError
    return x.to(device)

def to_device(x: Array, device: Device, /, *, stream: Optional[Union[int, Any]] = None) -> Array:
    """
    Copy the array from the device on which it currently resides to the specified ``device``.

    This is equivalent to `x.to_device(device, stream=stream)` according to
    the `standard
    <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.to_device.html>`__.
    This helper is included because some array libraries do not have the
    `to_device` method.

    Parameters
    ----------

    x: array
        array instance from an array API compatible library.

    device: device
        a ``device`` object (see the `Device Support <https://data-apis.org/array-api/latest/design_topics/device_support.html>`__
        section of the array API specification).

    stream: Optional[Union[int, Any]]
        stream object to use during copy. In addition to the types supported
        in ``array.__dlpack__``, implementations may choose to support any
        library-specific stream object with the caveat that any code using
        such an object would not be portable.

    Returns
    -------

    out: array
        an array with the same data and data type as ``x`` and located on the
        specified ``device``.

    Notes
    -----

    For NumPy, this function effectively does nothing since the only supported
    device is the CPU. For CuPy, this method supports CuPy CUDA
    :external+cupy:class:`Device <cupy.cuda.Device>` and
    :external+cupy:class:`Stream <cupy.cuda.Stream>` objects. For PyTorch,
    this is the same as :external+torch:meth:`x.to(device) <torch.Tensor.to>`
    (the ``stream`` argument is not supported in PyTorch).

    See Also
    --------

    device : Hardware device the array data resides on.

    """
    if is_numpy_array(x):
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        if device == 'cpu':
            return x
        raise ValueError(f"Unsupported device {device!r}")
    elif is_cupy_array(x):
        # cupy does not yet have to_device
        return _cupy_to_device(x, device, stream=stream)
    elif is_torch_array(x):
        return _torch_to_device(x, device, stream=stream)
    elif is_dask_array(x):
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        # TODO: What if our array is on the GPU already?
        if device == 'cpu':
            return x
        raise ValueError(f"Unsupported device {device!r}")
    elif is_jax_array(x):
        if not hasattr(x, "__array_namespace__"):
            # In JAX v0.4.31 and older, this import adds to_device method to x.
            import jax.experimental.array_api # noqa: F401
        return x.to_device(device, stream=stream)
    elif is_pydata_sparse_array(x) and device == _device(x):
        # Perform trivial check to return the same array if
        # device is same instead of err-ing.
        return x
    return x.to_device(device, stream=stream)

def size(x):
    """
    Return the total number of elements of x.

    This is equivalent to `x.size` according to the `standard
    <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.size.html>`__.
    This helper is included because PyTorch defines `size` in an
    :external+torch:meth:`incompatible way <torch.Tensor.size>`.

    """
    if None in x.shape:
        return None
    return math.prod(x.shape)

__all__ = [
    "array_namespace",
    "device",
    "get_namespace",
    "is_array_api_obj",
    "is_array_api_strict_namespace",
    "is_cupy_array",
    "is_cupy_namespace",
    "is_dask_array",
    "is_dask_namespace",
    "is_jax_array",
    "is_jax_namespace",
    "is_numpy_array",
    "is_numpy_namespace",
    "is_torch_array",
    "is_torch_namespace",
    "is_ndonnx_array",
    "is_ndonnx_namespace",
    "is_pydata_sparse_array",
    "is_pydata_sparse_namespace",
    "size",
    "to_device",
]

_all_ignore = ['sys', 'math', 'inspect', 'warnings']
<|endoftext|>"""
These are functions that are just aliases of existing functions in NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Sequence, Tuple, Union
    from ._typing import ndarray, Device, Dtype

from typing import NamedTuple
import inspect

from ._helpers import array_namespace, _check_device, device, is_torch_array, is_cupy_namespace

# These functions are modified from the NumPy versions.

# Creation functions add the device keyword (which does nothing for NumPy)

def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    xp,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs
) -> ndarray:
    _check_device(xp, device)
    return xp.arange(start, stop=stop, step=step, dtype=dtype, **kwargs)

def empty(
    shape: Union[int, Tuple[int, ...]],
    xp,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs
) -> ndarray:
    _check_device(xp, device)
    return xp.empty(shape, dtype=dtype, **kwargs)

def empty_like(
    x: ndarray, /, xp, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None,
    **kwargs
) -> ndarray:
    _check_device(xp, device)
    return xp.empty_like(x, dtype=dtype, **kwargs)

def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    xp,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> ndarray:
    _check_device(xp, device)
    return xp.eye(n_rows, M=n_cols, k=k, dtype=dtype, **kwargs)

def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    xp,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> ndarray:
    _check_device(xp, device)
    return xp.full(shape, fill_value, dtype=dtype, **kwargs)

def full_like(
    x: ndarray,
    /,
    fill_value: Union[int, float],
    *,
    xp,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> ndarray:
    _check_device(xp, device)
    return xp.full_like(x, fill_value, dtype=dtype, **kwargs)

def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    xp,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
    **kwargs,
) -> ndarray:
    _check_device(xp, device)
    return xp.linspace(start, stop, num, dtype=dtype, endpoint=endpoint, **kwargs)

def ones(
    shape: Union[int, Tuple[int, ...]],
    xp,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> ndarray:
    _check_device(xp, device)
    return xp.ones(shape, dtype=dtype, **kwargs)

def ones_like(
    x: ndarray, /, xp, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None,
    **kwargs,
) -> ndarray:
    _check_device(xp, device)
    return xp.ones_like(x, dtype=dtype, **kwargs)

def zeros(
    shape: Union[int, Tuple[int, ...]],
    xp,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> ndarray:
    _check_device(xp, device)
    return xp.zeros(shape, dtype=dtype, **kwargs)

def zeros_like(
    x: ndarray, /, xp, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None,
    **kwargs,
) -> ndarray:
    _check_device(xp, device)
    return xp.zeros_like(x, dtype=dtype, **kwargs)

# np.unique() is split into four functions in the array API:
# unique_all, unique_counts, unique_inverse, and unique_values (this is done
# to remove polymorphic return types).

# The functions here return namedtuples (np.unique() returns a normal
# tuple).

# Note that these named tuples aren't actually part of the standard namespace,
# but I don't see any issue with exporting the names here regardless.
class UniqueAllResult(NamedTuple):
    values: ndarray
    indices: ndarray
    inverse_indices: ndarray
    counts: ndarray


class UniqueCountsResult(NamedTuple):
    values: ndarray
    counts: ndarray


class UniqueInverseResult(NamedTuple):
    values: ndarray
    inverse_indices: ndarray


def _unique_kwargs(xp):
    # Older versions of NumPy and CuPy do not have equal_nan. Rather than
    # trying to parse version numbers, just check if equal_nan is in the
    # signature.
    s = inspect.signature(xp.unique)
    if 'equal_nan' in s.parameters:
        return {'equal_nan': False}
    return {}

def unique_all(x: ndarray, /, xp) -> UniqueAllResult:
    kwargs = _unique_kwargs(xp)
    values, indices, inverse_indices, counts = xp.unique(
        x,
        return_counts=True,
        return_index=True,
        return_inverse=True,
        **kwargs,
    )
    # np.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueAllResult(
        values,
        indices,
        inverse_indices,
        counts,
    )


def unique_counts(x: ndarray, /, xp) -> UniqueCountsResult:
    kwargs = _unique_kwargs(xp)
    res = xp.unique(
        x,
        return_counts=True,
        return_index=False,
        return_inverse=False,
        **kwargs
    )

    return UniqueCountsResult(*res)


def unique_inverse(x: ndarray, /, xp) -> UniqueInverseResult:
    kwargs = _unique_kwargs(xp)
    values, inverse_indices = xp.unique(
        x,
        return_counts=False,
        return_index=False,
        return_inverse=True,
        **kwargs,
    )
    # xp.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueInverseResult(values, inverse_indices)


def unique_values(x: ndarray, /, xp) -> ndarray:
    kwargs = _unique_kwargs(xp)
    return xp.unique(
        x,
        return_counts=False,
        return_index=False,
        return_inverse=False,
        **kwargs,
    )

def astype(x: ndarray, dtype: Dtype, /, *, copy: bool = True) -> ndarray:
    if not copy and dtype == x.dtype:
        return x
    return x.astype(dtype=dtype, copy=copy)

# These functions have different keyword argument names

def std(
    x: ndarray,
    /,
    xp,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0, # correction instead of ddof
    keepdims: bool = False,
    **kwargs,
) -> ndarray:
    return xp.std(x, axis=axis, ddof=correction, keepdims=keepdims, **kwargs)

def var(
    x: ndarray,
    /,
    xp,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0, # correction instead of ddof
    keepdims: bool = False,
    **kwargs,
) -> ndarray:
    return xp.var(x, axis=axis, ddof=correction, keepdims=keepdims, **kwargs)

# cumulative_sum is renamed from cumsum, and adds the include_initial keyword
# argument

def cumulative_sum(
    x: ndarray,
    /,
    xp,
    *,
    axis: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    include_initial: bool = False,
    **kwargs
) -> ndarray:
    wrapped_xp = array_namespace(x)

    # TODO: The standard is not clear about what should happen when x.ndim == 0.
    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified in cumulative_sum for more than one dimension")
        axis = 0

    res = xp.cumsum(x, axis=axis, dtype=dtype, **kwargs)

    # np.cumsum does not support include_initial
    if include_initial:
        initial_shape = list(x.shape)
        initial_shape[axis] = 1
        res = xp.concatenate(
            [wrapped_xp.zeros(shape=initial_shape, dtype=res.dtype, device=device(res)), res],
            axis=axis,
        )
    return res

# The min and max argument names in clip are different and not optional in numpy, and type
# promotion behavior is different.
def clip(
    x: ndarray,
    /,
    min: Optional[Union[int, float, ndarray]] = None,
    max: Optional[Union[int, float, ndarray]] = None,
    *,
    xp,
    # TODO: np.clip has other ufunc kwargs
    out: Optional[ndarray] = None,
) -> ndarray:
    def _isscalar(a):
        return isinstance(a, (int, float, type(None)))
    min_shape = () if _isscalar(min) else min.shape
    max_shape = () if _isscalar(max) else max.shape

    wrapped_xp = array_namespace(x)

    result_shape = xp.broadcast_shapes(x.shape, min_shape, max_shape)

    # np.clip does type promotion but the array API clip requires that the
    # output have the same dtype as x. We do this instead of just downcasting
    # the result of xp.clip() to handle some corner cases better (e.g.,
    # avoiding uint64 -> float64 promotion).

    # Note: cases where min or max overflow (integer) or round (float) in the
    # wrong direction when downcasting to x.dtype are unspecified. This code
    # just does whatever NumPy does when it downcasts in the assignment, but
    # other behavior could be preferred, especially for integers. For example,
    # this code produces:

    # >>> clip(asarray(0, dtype=int8), asarray(128, dtype=int16), None)
    # -128

    # but an answer of 0 might be preferred. See
    # https://github.com/numpy/numpy/issues/24976 for more discussion on this issue.


    # At least handle the case of Python integers correctly (see
    # https://github.com/numpy/numpy/pull/26892).
    if type(min) is int and min <= wrapped_xp.iinfo(x.dtype).min:
        min = None
    if type(max) is int and max >= wrapped_xp.iinfo(x.dtype).max:
        max = None

    if out is None:
        out = wrapped_xp.asarray(xp.broadcast_to(x, result_shape),
                                 copy=True, device=device(x))
    if min is not None:
        if is_torch_array(x) and x.dtype == xp.float64 and _isscalar(min):
            # Avoid loss of precision due to torch defaulting to float32
            min = wrapped_xp.asarray(min, dtype=xp.float64)
        a = xp.broadcast_to(wrapped_xp.asarray(min, device=device(x)), result_shape)
        ia = (out < a) | xp.isnan(a)
        # torch requires an explicit cast here
        out[ia] = wrapped_xp.astype(a[ia], out.dtype)
    if max is not None:
        if is_torch_array(x) and x.dtype == xp.float64 and _isscalar(max):
            max = wrapped_xp.asarray(max, dtype=xp.float64)
        b = xp.broadcast_to(wrapped_xp.asarray(max, device=device(x)), result_shape)
        ib = (out > b) | xp.isnan(b)
        out[ib] = wrapped_xp.astype(b[ib], out.dtype)
    # Return a scalar for 0-D
    return out[()]

# Unlike transpose(), the axes argument to permute_dims() is required.
def permute_dims(x: ndarray, /, axes: Tuple[int, ...], xp) -> ndarray:
    return xp.transpose(x, axes)

# np.reshape calls the keyword argument 'newshape' instead of 'shape'
def reshape(x: ndarray,
            /,
            shape: Tuple[int, ...],
            xp, copy: Optional[bool] = None,
            **kwargs) -> ndarray:
    if copy is True:
        x = x.copy()
    elif copy is False:
        y = x.view()
        y.shape = shape
        return y
    return xp.reshape(x, shape, **kwargs)

# The descending keyword is new in sort and argsort, and 'kind' replaced with
# 'stable'
def argsort(
    x: ndarray, /, xp, *, axis: int = -1, descending: bool = False, stable: bool = True,
    **kwargs,
) -> ndarray:
    # Note: this keyword argument is different, and the default is different.
    # We set it in kwargs like this because numpy.sort uses kind='quicksort'
    # as the default whereas cupy.sort uses kind=None.
    if stable:
        kwargs['kind'] = "stable"
    if not descending:
        res = xp.argsort(x, axis=axis, **kwargs)
    else:
        # As NumPy has no native descending sort, we imitate it here. Note that
        # simply flipping the results of xp.argsort(x, ...) would not
        # respect the relative order like it would in native descending sorts.
        res = xp.flip(
            xp.argsort(xp.flip(x, axis=axis), axis=axis, **kwargs),
            axis=axis,
        )
        # Rely on flip()/argsort() to validate axis
        normalised_axis = axis if axis >= 0 else x.ndim + axis
        max_i = x.shape[normalised_axis] - 1
        res = max_i - res
    return res

def sort(
    x: ndarray, /, xp, *, axis: int = -1, descending: bool = False, stable: bool = True,
    **kwargs,
) -> ndarray:
    # Note: this keyword argument is different, and the default is different.
    # We set it in kwargs like this because numpy.sort uses kind='quicksort'
    # as the default whereas cupy.sort uses kind=None.
    if stable:
        kwargs['kind'] = "stable"
    res = xp.sort(x, axis=axis, **kwargs)
    if descending:
        res = xp.flip(res, axis=axis)
    return res

# nonzero should error for zero-dimensional arrays
def nonzero(x: ndarray, /, xp, **kwargs) -> Tuple[ndarray, ...]:
    if x.ndim == 0:
        raise ValueError("nonzero() does not support zero-dimensional arrays")
    return xp.nonzero(x, **kwargs)

# ceil, floor, and trunc return integers for integer inputs

def ceil(x: ndarray, /, xp, **kwargs) -> ndarray:
    if xp.issubdtype(x.dtype, xp.integer):
        return x
    return xp.ceil(x, **kwargs)

def floor(x: ndarray, /, xp, **kwargs) -> ndarray:
    if xp.issubdtype(x.dtype, xp.integer):
        return x
    return xp.floor(x, **kwargs)

def trunc(x: ndarray, /, xp, **kwargs) -> ndarray:
    if xp.issubdtype(x.dtype, xp.integer):
        return x
    return xp.trunc(x, **kwargs)

# linear algebra functions

def matmul(x1: ndarray, x2: ndarray, /, xp, **kwargs) -> ndarray:
    return xp.matmul(x1, x2, **kwargs)

# Unlike transpose, matrix_transpose only transposes the last two axes.
def matrix_transpose(x: ndarray, /, xp) -> ndarray:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return xp.swapaxes(x, -1, -2)

def tensordot(x1: ndarray,
              x2: ndarray,
              /,
              xp,
              *,
              axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
              **kwargs,
) -> ndarray:
    return xp.tensordot(x1, x2, axes=axes, **kwargs)

def vecdot(x1: ndarray, x2: ndarray, /, xp, *, axis: int = -1) -> ndarray:
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    if hasattr(xp, 'broadcast_tensors'):
        _broadcast = xp.broadcast_tensors
    else:
        _broadcast = xp.broadcast_arrays

    x1_ = xp.moveaxis(x1, axis, -1)
    x2_ = xp.moveaxis(x2, axis, -1)
    x1_, x2_ = _broadcast(x1_, x2_)

    res = x1_[..., None, :] @ x2_[..., None]
    return res[..., 0, 0]

# isdtype is a new function in the 2022.12 array API specification.

def isdtype(
    dtype: Dtype, kind: Union[Dtype, str, Tuple[Union[Dtype, str], ...]], xp,
    *, _tuple=True, # Disallow nested tuples
) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.

    Note that outside of this function, this compat library does not yet fully
    support complex numbers.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    for more details
    """
    if isinstance(kind, tuple) and _tuple:
        return any(isdtype(dtype, k, xp, _tuple=False) for k in kind)
    elif isinstance(kind, str):
        if kind == 'bool':
            return dtype == xp.bool_
        elif kind == 'signed integer':
            return xp.issubdtype(dtype, xp.signedinteger)
        elif kind == 'unsigned integer':
            return xp.issubdtype(dtype, xp.unsignedinteger)
        elif kind == 'integral':
            return xp.issubdtype(dtype, xp.integer)
        elif kind == 'real floating':
            return xp.issubdtype(dtype, xp.floating)
        elif kind == 'complex floating':
            return xp.issubdtype(dtype, xp.complexfloating)
        elif kind == 'numeric':
            return xp.issubdtype(dtype, xp.number)
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    else:
        # This will allow things that aren't required by the spec, like
        # isdtype(np.float64, float) or isdtype(np.int64, 'l'). Should we be
        # more strict here to match the type annotation? Note that the
        # array_api_strict implementation will be very strict.
        return dtype == kind

# unstack is a new function in the 2023.12 array API standard
def unstack(x: ndarray, /, xp, *, axis: int = 0) -> Tuple[ndarray, ...]:
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")
    return tuple(xp.moveaxis(x, axis, 0))

# numpy 1.26 does not use the standard definition for sign on complex numbers

def sign(x: ndarray, /, xp, **kwargs) -> ndarray:
    if isdtype(x.dtype, 'complex floating', xp=xp):
        out = (x/xp.abs(x, **kwargs))[...]
        # sign(0) = 0 but the above formula would give nan
        out[x == 0+0j] = 0+0j
    else:
        out = xp.sign(x, **kwargs)
    # CuPy sign() does not propagate nans. See
    # https://github.com/data-apis/array-api-compat/issues/136
    if is_cupy_namespace(xp) and isdtype(x.dtype, 'real floating', xp=xp):
        out[xp.isnan(x)] = xp.nan
    return out[()]

__all__ = ['arange', 'empty', 'empty_like', 'eye', 'full', 'full_like',
           'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like',
           'UniqueAllResult', 'UniqueCountsResult', 'UniqueInverseResult',
           'unique_all', 'unique_counts', 'unique_inverse', 'unique_values',
           'astype', 'std', 'var', 'cumulative_sum', 'clip', 'permute_dims',
           'reshape', 'argsort', 'sort', 'nonzero', 'ceil', 'floor', 'trunc',
           'matmul', 'matrix_transpose', 'tensordot', 'vecdot', 'isdtype',
           'unstack', 'sign']
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional, Literal

if TYPE_CHECKING:
    from ._typing import Device, ndarray
    from collections.abc import Sequence

# Note: NumPy fft functions improperly upcast float32 and complex64 to
# complex128, which is why we require wrapping them all here.

def fft(
    x: ndarray,
    /,
    xp,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.fft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def ifft(
    x: ndarray,
    /,
    xp,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.ifft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def fftn(
    x: ndarray,
    /,
    xp,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.fftn(x, s=s, axes=axes, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def ifftn(
    x: ndarray,
    /,
    xp,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.ifftn(x, s=s, axes=axes, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def rfft(
    x: ndarray,
    /,
    xp,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.rfft(x, n=n, axis=axis, norm=norm)
    if x.dtype == xp.float32:
        return res.astype(xp.complex64)
    return res

def irfft(
    x: ndarray,
    /,
    xp,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.irfft(x, n=n, axis=axis, norm=norm)
    if x.dtype == xp.complex64:
        return res.astype(xp.float32)
    return res

def rfftn(
    x: ndarray,
    /,
    xp,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.rfftn(x, s=s, axes=axes, norm=norm)
    if x.dtype == xp.float32:
        return res.astype(xp.complex64)
    return res

def irfftn(
    x: ndarray,
    /,
    xp,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.irfftn(x, s=s, axes=axes, norm=norm)
    if x.dtype == xp.complex64:
        return res.astype(xp.float32)
    return res

def hfft(
    x: ndarray,
    /,
    xp,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.hfft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.float32)
    return res

def ihfft(
    x: ndarray,
    /,
    xp,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> ndarray:
    res = xp.fft.ihfft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def fftfreq(n: int, /, xp, *, d: float = 1.0, device: Optional[Device] = None) -> ndarray:
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return xp.fft.fftfreq(n, d=d)

def rfftfreq(n: int, /, xp, *, d: float = 1.0, device: Optional[Device] = None) -> ndarray:
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    return xp.fft.rfftfreq(n, d=d)

def fftshift(x: ndarray, /, xp, *, axes: Union[int, Sequence[int]] = None) -> ndarray:
    return xp.fft.fftshift(x, axes=axes)

def ifftshift(x: ndarray, /, xp, *, axes: Union[int, Sequence[int]] = None) -> ndarray:
    return xp.fft.ifftshift(x, axes=axes)

__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
<|endoftext|>from __future__ import annotations

from ...common import _linalg
from ..._internal import get_xp

# Exports
from dask.array.linalg import * # noqa: F403
from dask.array import outer

# These functions are in both the main and linalg namespaces
from dask.array import matmul, tensordot
from ._aliases import matrix_transpose, vecdot

import dask.array as da

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...common._typing import Array
    from typing import Literal

# dask.array.linalg doesn't have __all__. If it is added, replace this with
#
# from dask.array.linalg import __all__ as linalg_all
_n = {}
exec('from dask.array.linalg import *', _n)
del _n['__builtins__']
if 'annotations' in _n:
    del _n['annotations']
linalg_all = list(_n)
del _n

EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
# TODO: use the QR wrapper once dask
# supports the mode keyword on QR
# https://github.com/dask/dask/issues/10388
#qr = get_xp(da)(_linalg.qr)
def qr(x: Array, mode: Literal['reduced', 'complete'] = 'reduced',
       **kwargs) -> QRResult:
    if mode != "reduced":
        raise ValueError("dask arrays only support using mode='reduced'")
    return QRResult(*da.linalg.qr(x, **kwargs))
trace = get_xp(da)(_linalg.trace)
cholesky = get_xp(da)(_linalg.cholesky)
matrix_rank = get_xp(da)(_linalg.matrix_rank)
matrix_norm = get_xp(da)(_linalg.matrix_norm)


# Wrap the svd functions to not pass full_matrices to dask
# when full_matrices=False (as that is the default behavior for dask),
# and dask doesn't have the full_matrices keyword
def svd(x: Array, full_matrices: bool = True, **kwargs) -> SVDResult:
    if full_matrices:
        raise ValueError("full_matrics=True is not supported by dask.")
    return da.linalg.svd(x, coerce_signs=False, **kwargs)

def svdvals(x: Array) -> Array:
    # TODO: can't avoid computing U or V for dask
    _, s, _ =  svd(x)
    return s

vector_norm = get_xp(da)(_linalg.vector_norm)
diagonal = get_xp(da)(_linalg.diagonal)

__all__ = linalg_all + ["trace", "outer", "matmul", "tensordot",
                        "matrix_transpose", "vecdot", "EighResult",
                        "QRResult", "SlogdetResult", "SVDResult", "qr",
                        "cholesky", "matrix_rank", "matrix_norm", "svdvals",
                        "vector_norm", "diagonal"]

_all_ignore = ['get_xp', 'da', 'linalg_all']
<|endoftext|>from dask.array.fft import * # noqa: F403
# dask.array.fft doesn't have __all__. If it is added, replace this with
#
# from dask.array.fft import __all__ as linalg_all
_n = {}
exec('from dask.array.fft import *', _n)
del _n['__builtins__']
fft_all = list(_n)
del _n

from ...common import _fft
from ..._internal import get_xp

import dask.array as da

fftfreq = get_xp(da)(_fft.fftfreq)
rfftfreq = get_xp(da)(_fft.rfftfreq)

__all__ = [elem for elem in fft_all if elem != "annotations"] + ["fftfreq", "rfftfreq"]

del get_xp
del da
del fft_all
del _fft
<|endoftext|>"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""
from numpy import (
    dtype,
    bool_ as bool,
    intp,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
)

from ...common._helpers import _DASK_DEVICE

class __array_namespace_info__:
    """
    Get the array API inspection namespace for Dask.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for Dask.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': dask.float64,
     'complex floating': dask.complex128,
     'integral': dask.int64,
     'indexing': dask.int64}

    """

    __module__ = 'dask.array'

    def capabilities(self):
        """
        Return a dictionary of array API library capabilities.

        The resulting dictionary has the following keys:

        - **"boolean indexing"**: boolean indicating whether an array library
          supports boolean indexing. Always ``False`` for Dask.

        - **"data-dependent shapes"**: boolean indicating whether an array
          library supports data-dependent output shapes. Always ``False`` for
          Dask.

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.

        See Also
        --------
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        capabilities : dict
            A dictionary of array API library capabilities.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.capabilities()
        {'boolean indexing': True,
         'data-dependent shapes': True}

        """
        return {
            "boolean indexing": False,
            "data-dependent shapes": False,
            # 'max rank' will be part of the 2024.12 standard
            # "max rank": 64,
        }

    def default_device(self):
        """
        The default device used for new Dask arrays.

        For Dask, this always returns ``'cpu'``.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        device : str
            The default device used for new Dask arrays.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_device()
        'cpu'

        """
        return "cpu"

    def default_dtypes(self, *, device=None):
        """
        The default data types used for new Dask arrays.

        For Dask, this always returns the following dictionary:

        - **"real floating"**: ``numpy.float64``
        - **"complex floating"**: ``numpy.complex128``
        - **"integral"**: ``numpy.intp``
        - **"indexing"**: ``numpy.intp``

        Parameters
        ----------
        device : str, optional
            The device to get the default data types for.

        Returns
        -------
        dtypes : dict
            A dictionary describing the default data types used for new Dask
            arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_dtypes()
        {'real floating': dask.float64,
         'complex floating': dask.complex128,
         'integral': dask.int64,
         'indexing': dask.int64}

        """
        if device not in ["cpu", _DASK_DEVICE, None]:
            raise ValueError(
                'Device not understood. Only "cpu" or _DASK_DEVICE is allowed, but received:'
                f' {device}'
            )
        return {
            "real floating": dtype(float64),
            "complex floating": dtype(complex128),
            "integral": dtype(intp),
            "indexing": dtype(intp),
        }

    def dtypes(self, *, device=None, kind=None):
        """
        The array API data types supported by Dask.

        Note that this function only returns data types that are defined by
        the array API.

        Parameters
        ----------
        device : str, optional
            The device to get the data types for.
        kind : str or tuple of str, optional
            The kind of data types to return. If ``None``, all data types are
            returned. If a string, only data types of that kind are returned.
            If a tuple, a dictionary containing the union of the given kinds
            is returned. The following kinds are supported:

            - ``'bool'``: boolean data types (i.e., ``bool``).
            - ``'signed integer'``: signed integer data types (i.e., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (i.e.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (i.e., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (i.e.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

        Returns
        -------
        dtypes : dict
            A dictionary mapping the names of data types to the corresponding
            Dask data types.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.dtypes(kind='signed integer')
        {'int8': dask.int8,
         'int16': dask.int16,
         'int32': dask.int32,
         'int64': dask.int64}

        """
        if device not in ["cpu", _DASK_DEVICE, None]:
            raise ValueError(
                'Device not understood. Only "cpu" or _DASK_DEVICE is allowed, but received:'
                f' {device}'
            )
        if kind is None:
            return {
                "bool": dtype(bool),
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
            }
        if kind == "unsigned integer":
            return {
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "integral":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "real floating":
            return {
                "float32": dtype(float32),
                "float64": dtype(float64),
            }
        if kind == "complex floating":
            return {
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "numeric":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self):
        """
        The devices supported by Dask.

        For Dask, this always returns ``['cpu', DASK_DEVICE]``.

        Returns
        -------
        devices : list of str
            The devices supported by Dask.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.devices()
        ['cpu', DASK_DEVICE]

        """
        return ["cpu", _DASK_DEVICE]
<|endoftext|>from __future__ import annotations

from ...common import _aliases
from ...common._helpers import _check_device

from ..._internal import get_xp

from ._info import __array_namespace_info__

import numpy as np
from numpy import (
    # Dtypes
    iinfo,
    finfo,
    bool_ as bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    complex64,
    complex128,
    can_cast,
    result_type,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Union

    from ...common._typing import Device, Dtype, Array, NestedSequence, SupportsBufferProtocol

import dask.array as da

isdtype = get_xp(np)(_aliases.isdtype)
unstack = get_xp(da)(_aliases.unstack)
astype = _aliases.astype

# Common aliases

# This arange func is modified from the common one to
# not pass stop/step as keyword arguments, which will cause
# an error with dask

# TODO: delete the xp stuff, it shouldn't be necessary
def _dask_arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    xp,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> Array:
    _check_device(xp, device)
    args = [start]
    if stop is not None:
        args.append(stop)
    else:
        # stop is None, so start is actually stop
        # prepend the default value for start which is 0
        args.insert(0, 0)
    args.append(step)
    return xp.arange(*args, dtype=dtype, **kwargs)

arange = get_xp(da)(_dask_arange)
eye = get_xp(da)(_aliases.eye)

linspace = get_xp(da)(_aliases.linspace)
eye = get_xp(da)(_aliases.eye)
UniqueAllResult = get_xp(da)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(da)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(da)(_aliases.UniqueInverseResult)
unique_all = get_xp(da)(_aliases.unique_all)
unique_counts = get_xp(da)(_aliases.unique_counts)
unique_inverse = get_xp(da)(_aliases.unique_inverse)
unique_values = get_xp(da)(_aliases.unique_values)
permute_dims = get_xp(da)(_aliases.permute_dims)
std = get_xp(da)(_aliases.std)
var = get_xp(da)(_aliases.var)
cumulative_sum = get_xp(da)(_aliases.cumulative_sum)
empty = get_xp(da)(_aliases.empty)
empty_like = get_xp(da)(_aliases.empty_like)
full = get_xp(da)(_aliases.full)
full_like = get_xp(da)(_aliases.full_like)
ones = get_xp(da)(_aliases.ones)
ones_like = get_xp(da)(_aliases.ones_like)
zeros = get_xp(da)(_aliases.zeros)
zeros_like = get_xp(da)(_aliases.zeros_like)
reshape = get_xp(da)(_aliases.reshape)
matrix_transpose = get_xp(da)(_aliases.matrix_transpose)
vecdot = get_xp(da)(_aliases.vecdot)

nonzero = get_xp(da)(_aliases.nonzero)
ceil = get_xp(np)(_aliases.ceil)
floor = get_xp(np)(_aliases.floor)
trunc = get_xp(np)(_aliases.trunc)
matmul = get_xp(np)(_aliases.matmul)
tensordot = get_xp(np)(_aliases.tensordot)
sign = get_xp(np)(_aliases.sign)

# asarray also adds the copy keyword, which is not present in numpy 1.0.
def asarray(
    obj: Union[
        Array,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsBufferProtocol,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: "Optional[Union[bool, np._CopyMode]]" = None,
    **kwargs,
) -> Array:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    if copy is False:
        # copy=False is not yet implemented in dask
        raise NotImplementedError("copy=False is not yet implemented")
    elif copy is True:
        if isinstance(obj, da.Array) and dtype is None:
            return obj.copy()
        # Go through numpy, since dask copy is no-op by default
        obj = np.array(obj, dtype=dtype, copy=True)
        return da.array(obj, dtype=dtype)
    else:
        if not isinstance(obj, da.Array) or dtype is not None and obj.dtype != dtype:
            obj = np.asarray(obj, dtype=dtype)
            return da.from_array(obj)
        return obj

    return da.asarray(obj, dtype=dtype, **kwargs)

from dask.array import (
    # Element wise aliases
    arccos as acos,
    arccosh as acosh,
    arcsin as asin,
    arcsinh as asinh,
    arctan as atan,
    arctan2 as atan2,
    arctanh as atanh,
    left_shift as bitwise_left_shift,
    right_shift as bitwise_right_shift,
    invert as bitwise_invert,
    power as pow,
    # Other
    concatenate as concat,
)

# dask.array.clip does not work unless all three arguments are provided.
# Furthermore, the masking workaround in common._aliases.clip cannot work with
# dask (meaning uint64 promoting to float64 is going to just be unfixed for
# now).
@get_xp(da)
def clip(
    x: Array,
    /,
    min: Optional[Union[int, float, Array]] = None,
    max: Optional[Union[int, float, Array]] = None,
    *,
    xp,
) -> Array:
    def _isscalar(a):
        return isinstance(a, (int, float, type(None)))
    min_shape = () if _isscalar(min) else min.shape
    max_shape = () if _isscalar(max) else max.shape

    # TODO: This won't handle dask unknown shapes
    import numpy as np
    result_shape = np.broadcast_shapes(x.shape, min_shape, max_shape)

    if min is not None:
        min = xp.broadcast_to(xp.asarray(min), result_shape)
    if max is not None:
        max = xp.broadcast_to(xp.asarray(max), result_shape)

    if min is None and max is None:
        return xp.positive(x)

    if min is None:
        return astype(xp.minimum(x, max), x.dtype)
    if max is None:
        return astype(xp.maximum(x, min), x.dtype)

    return astype(xp.minimum(xp.maximum(x, min), max), x.dtype)

# exclude these from all since dask.array has no sorting functions
_da_unsupported = ['sort', 'argsort']

_common_aliases = [alias for alias in _aliases.__all__ if alias not in _da_unsupported]

__all__ = _common_aliases + ['__array_namespace_info__', 'asarray', 'acos',
                    'acosh', 'asin', 'asinh', 'atan', 'atan2',
                    'atanh', 'bitwise_left_shift', 'bitwise_invert',
                    'bitwise_right_shift', 'concat', 'pow', 'iinfo', 'finfo', 'can_cast',
                    'result_type', 'bool', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64',
                    'uint8', 'uint16', 'uint32', 'uint64',
                    'complex64', 'complex128', 'iinfo', 'finfo',
                    'can_cast', 'result_type']

_all_ignore = ["get_xp", "da", "np"]
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    array = torch.Tensor
    from torch import dtype as Dtype
    from typing import Optional, Union, Tuple, Literal
    inf = float('inf')

from ._aliases import _fix_promotion, sum

from torch.linalg import * # noqa: F403

# torch.linalg doesn't define __all__
# from torch.linalg import __all__ as linalg_all
from torch import linalg as torch_linalg
linalg_all = [i for i in dir(torch_linalg) if not i.startswith('_')]

# outer is implemented in torch but aren't in the linalg namespace
from torch import outer
# These functions are in both the main and linalg namespaces
from ._aliases import matmul, matrix_transpose, tensordot

# Note: torch.linalg.cross does not default to axis=-1 (it defaults to the
# first axis with size 3), see https://github.com/pytorch/pytorch/issues/58743

# torch.cross also does not support broadcasting when it would add new
# dimensions https://github.com/pytorch/pytorch/issues/39656
def cross(x1: array, x2: array, /, *, axis: int = -1) -> array:
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    if not (-min(x1.ndim, x2.ndim) <= axis < max(x1.ndim, x2.ndim)):
        raise ValueError(f"axis {axis} out of bounds for cross product of arrays with shapes {x1.shape} and {x2.shape}")
    if not (x1.shape[axis] == x2.shape[axis] == 3):
        raise ValueError(f"cross product axis must have size 3, got {x1.shape[axis]} and {x2.shape[axis]}")
    x1, x2 = torch.broadcast_tensors(x1, x2)
    return torch_linalg.cross(x1, x2, dim=axis)

def vecdot(x1: array, x2: array, /, *, axis: int = -1, **kwargs) -> array:
    from ._aliases import isdtype

    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)

    # torch.linalg.vecdot incorrectly allows broadcasting along the contracted dimension
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    # torch.linalg.vecdot doesn't support integer dtypes
    if isdtype(x1.dtype, 'integral') or isdtype(x2.dtype, 'integral'):
        if kwargs:
            raise RuntimeError("vecdot kwargs not supported for integral dtypes")

        x1_ = torch.moveaxis(x1, axis, -1)
        x2_ = torch.moveaxis(x2, axis, -1)
        x1_, x2_ = torch.broadcast_tensors(x1_, x2_)

        res = x1_[..., None, :] @ x2_[..., None]
        return res[..., 0, 0]
    return torch.linalg.vecdot(x1, x2, dim=axis, **kwargs)

def solve(x1: array, x2: array, /, **kwargs) -> array:
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    # Torch tries to emulate NumPy 1 solve behavior by using batched 1-D solve
    # whenever
    # 1. x1.ndim - 1 == x2.ndim
    # 2. x1.shape[:-1] == x2.shape
    #
    # See linalg_solve_is_vector_rhs in
    # aten/src/ATen/native/LinearAlgebraUtils.h and
    # TORCH_META_FUNC(_linalg_solve_ex) in
    # aten/src/ATen/native/BatchLinearAlgebra.cpp in the PyTorch source code.
    #
    # The easiest way to work around this is to prepend a size 1 dimension to
    # x2, since x2 is already one dimension less than x1.
    #
    # See https://github.com/pytorch/pytorch/issues/52915
    if x2.ndim != 1 and x1.ndim - 1 == x2.ndim and x1.shape[:-1] == x2.shape:
        x2 = x2[None]
    return torch.linalg.solve(x1, x2, **kwargs)

# torch.trace doesn't support the offset argument and doesn't support stacking
def trace(x: array, /, *, offset: int = 0, dtype: Optional[Dtype] = None) -> array:
    # Use our wrapped sum to make sure it does upcasting correctly
    return sum(torch.diagonal(x, offset=offset, dim1=-2, dim2=-1), axis=-1, dtype=dtype)

def vector_norm(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    **kwargs,
) -> array:
    # torch.vector_norm incorrectly treats axis=() the same as axis=None
    if axis == ():
        out = kwargs.get('out')
        if out is None:
            dtype = None
            if x.dtype == torch.complex64:
                dtype = torch.float32
            elif x.dtype == torch.complex128:
                dtype = torch.float64

            out = torch.zeros_like(x, dtype=dtype)

        # The norm of a single scalar works out to abs(x) in every case except
        # for ord=0, which is x != 0.
        if ord == 0:
            out[:] = (x != 0)
        else:
            out[:] = torch.abs(x)
        return out
    return torch.linalg.vector_norm(x, ord=ord, axis=axis, keepdim=keepdims, **kwargs)

__all__ = linalg_all + ['outer', 'matmul', 'matrix_transpose', 'tensordot',
                        'cross', 'vecdot', 'solve', 'trace', 'vector_norm']

_all_ignore = ['torch_linalg', 'sum']

del linalg_all
<|endoftext|>from torch import * # noqa: F403

# Several names are not included in the above import *
import torch
for n in dir(torch):
    if (n.startswith('_')
        or n.endswith('_')
        or 'cuda' in n
        or 'cpu' in n
        or 'backward' in n):
        continue
    exec(n + ' = torch.' + n)

# These imports may overwrite names from the import * above.
from ._aliases import * # noqa: F403

# See the comment in the numpy __init__.py
__import__(__package__ + '.linalg')

__import__(__package__ + '.fft')

from ..common._helpers import * # noqa: F403

__array_api_version__ = '2023.12'
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    array = torch.Tensor
    from typing import Union, Sequence, Literal

from torch.fft import * # noqa: F403
import torch.fft

# Several torch fft functions do not map axes to dim

def fftn(
    x: array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array:
    return torch.fft.fftn(x, s=s, dim=axes, norm=norm, **kwargs)

def ifftn(
    x: array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array:
    return torch.fft.ifftn(x, s=s, dim=axes, norm=norm, **kwargs)

def rfftn(
    x: array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array:
    return torch.fft.rfftn(x, s=s, dim=axes, norm=norm, **kwargs)

def irfftn(
    x: array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array:
    return torch.fft.irfftn(x, s=s, dim=axes, norm=norm, **kwargs)

def fftshift(
    x: array,
    /,
    *,
    axes: Union[int, Sequence[int]] = None,
    **kwargs,
) -> array:
    return torch.fft.fftshift(x, dim=axes, **kwargs)

def ifftshift(
    x: array,
    /,
    *,
    axes: Union[int, Sequence[int]] = None,
    **kwargs,
) -> array:
    return torch.fft.ifftshift(x, dim=axes, **kwargs)


__all__ = torch.fft.__all__ + [
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "fftshift",
    "ifftshift",
]

_all_ignore = ['torch']
<|endoftext|>"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""
import torch

from functools import cache

class __array_namespace_info__:
    """
    Get the array API inspection namespace for PyTorch.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for PyTorch.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': numpy.float64,
     'complex floating': numpy.complex128,
     'integral': numpy.int64,
     'indexing': numpy.int64}

    """

    __module__ = 'torch'

    def capabilities(self):
        """
        Return a dictionary of array API library capabilities.

        The resulting dictionary has the following keys:

        - **"boolean indexing"**: boolean indicating whether an array library
          supports boolean indexing. Always ``True`` for PyTorch.

        - **"data-dependent shapes"**: boolean indicating whether an array
          library supports data-dependent output shapes. Always ``True`` for
          PyTorch.

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.

        See Also
        --------
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        capabilities : dict
            A dictionary of array API library capabilities.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.capabilities()
        {'boolean indexing': True,
         'data-dependent shapes': True}

        """
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            # 'max rank' will be part of the 2024.12 standard
            # "max rank": 64,
        }

    def default_device(self):
        """
        The default device used for new PyTorch arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        device : str
            The default device used for new PyTorch arrays.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_device()
        'cpu'

        """
        return torch.device("cpu")

    def default_dtypes(self, *, device=None):
        """
        The default data types used for new PyTorch arrays.

        Parameters
        ----------
        device : str, optional
            The device to get the default data types for. For PyTorch, only
            ``'cpu'`` is allowed.

        Returns
        -------
        dtypes : dict
            A dictionary describing the default data types used for new PyTorch
            arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_dtypes()
        {'real floating': torch.float32,
         'complex floating': torch.complex64,
         'integral': torch.int64,
         'indexing': torch.int64}

        """
        # Note: if the default is set to float64, the devices like MPS that
        # don't support float64 will error. We still return the default_dtype
        # value here because this error doesn't represent a different default
        # per-device.
        default_floating = torch.get_default_dtype()
        default_complex = torch.complex64 if default_floating == torch.float32 else torch.complex128
        default_integral = torch.int64
        return {
            "real floating": default_floating,
            "complex floating": default_complex,
            "integral": default_integral,
            "indexing": default_integral,
        }


    def _dtypes(self, kind):
        bool = torch.bool
        int8 = torch.int8
        int16 = torch.int16
        int32 = torch.int32
        int64 = torch.int64
        uint8 = torch.uint8
        # uint16, uint32, and uint64 are present in newer versions of pytorch,
        # but they aren't generally supported by the array API functions, so
        # we omit them from this function.
        float32 = torch.float32
        float64 = torch.float64
        complex64 = torch.complex64
        complex128 = torch.complex128

        if kind is None:
            return {
                "bool": bool,
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
                "float32": float32,
                "float64": float64,
                "complex64": complex64,
                "complex128": complex128,
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
            }
        if kind == "unsigned integer":
            return {
                "uint8": uint8,
            }
        if kind == "integral":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
            }
        if kind == "real floating":
            return {
                "float32": float32,
                "float64": float64,
            }
        if kind == "complex floating":
            return {
                "complex64": complex64,
                "complex128": complex128,
            }
        if kind == "numeric":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
                "float32": float32,
                "float64": float64,
                "complex64": complex64,
                "complex128": complex128,
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    @cache
    def dtypes(self, *, device=None, kind=None):
        """
        The array API data types supported by PyTorch.

        Note that this function only returns data types that are defined by
        the array API.

        Parameters
        ----------
        device : str, optional
            The device to get the data types for.
        kind : str or tuple of str, optional
            The kind of data types to return. If ``None``, all data types are
            returned. If a string, only data types of that kind are returned.
            If a tuple, a dictionary containing the union of the given kinds
            is returned. The following kinds are supported:

            - ``'bool'``: boolean data types (i.e., ``bool``).
            - ``'signed integer'``: signed integer data types (i.e., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (i.e.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (i.e., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (i.e.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

        Returns
        -------
        dtypes : dict
            A dictionary mapping the names of data types to the corresponding
            PyTorch data types.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.dtypes(kind='signed integer')
        {'int8': numpy.int8,
         'int16': numpy.int16,
         'int32': numpy.int32,
         'int64': numpy.int64}

        """
        res = self._dtypes(kind)
        for k, v in res.copy().items():
            try:
                torch.empty((0,), dtype=v, device=device)
            except:
                del res[k]
        return res

    @cache
    def devices(self):
        """
        The devices supported by PyTorch.

        Returns
        -------
        devices : list of str
            The devices supported by PyTorch.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.devices()
        [device(type='cpu'), device(type='mps', index=0), device(type='meta')]

        """
        # Torch doesn't have a straightforward way to get the list of all
        # currently supported devices. To do this, we first parse the error
        # message of torch.device to get the list of all possible types of
        # device:
        try:
            torch.device('notadevice')
        except RuntimeError as e:
            # The error message is something like:
            # "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: notadevice"
            devices_names = e.args[0].split('Expected one of ')[1].split(' device type')[0].split(', ')

        # Next we need to check for different indices for different devices.
        # device(device_name, index=index) doesn't actually check if the
        # device name or index is valid. We have to try to create a tensor
        # with it (which is why this function is cached).
        devices = []
        for device_name in devices_names:
            i = 0
            while True:
                try:
                    a = torch.empty((0,), device=torch.device(device_name, index=i))
                    if a.device in devices:
                        break
                    devices.append(a.device)
                except:
                    break
                i += 1

        return devices
<|endoftext|>from __future__ import annotations

from functools import wraps as _wraps
from builtins import all as _builtin_all, any as _builtin_any

from ..common._aliases import (matrix_transpose as _aliases_matrix_transpose,
                               vecdot as _aliases_vecdot,
                               clip as _aliases_clip,
                               unstack as _aliases_unstack,
                               cumulative_sum as _aliases_cumulative_sum,
                               )
from .._internal import get_xp

from ._info import __array_namespace_info__

import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union
    from ..common._typing import Device
    from torch import dtype as Dtype

    array = torch.Tensor

_int_dtypes = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

_array_api_dtypes = {
    torch.bool,
    *_int_dtypes,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
}

_promotion_table  = {
    # bool
    (torch.bool, torch.bool): torch.bool,
    # ints
    (torch.int8, torch.int8): torch.int8,
    (torch.int8, torch.int16): torch.int16,
    (torch.int8, torch.int32): torch.int32,
    (torch.int8, torch.int64): torch.int64,
    (torch.int16, torch.int8): torch.int16,
    (torch.int16, torch.int16): torch.int16,
    (torch.int16, torch.int32): torch.int32,
    (torch.int16, torch.int64): torch.int64,
    (torch.int32, torch.int8): torch.int32,
    (torch.int32, torch.int16): torch.int32,
    (torch.int32, torch.int32): torch.int32,
    (torch.int32, torch.int64): torch.int64,
    (torch.int64, torch.int8): torch.int64,
    (torch.int64, torch.int16): torch.int64,
    (torch.int64, torch.int32): torch.int64,
    (torch.int64, torch.int64): torch.int64,
    # uints
    (torch.uint8, torch.uint8): torch.uint8,
    # ints and uints (mixed sign)
    (torch.int8, torch.uint8): torch.int16,
    (torch.int16, torch.uint8): torch.int16,
    (torch.int32, torch.uint8): torch.int32,
    (torch.int64, torch.uint8): torch.int64,
    (torch.uint8, torch.int8): torch.int16,
    (torch.uint8, torch.int16): torch.int16,
    (torch.uint8, torch.int32): torch.int32,
    (torch.uint8, torch.int64): torch.int64,
    # floats
    (torch.float32, torch.float32): torch.float32,
    (torch.float32, torch.float64): torch.float64,
    (torch.float64, torch.float32): torch.float64,
    (torch.float64, torch.float64): torch.float64,
    # complexes
    (torch.complex64, torch.complex64): torch.complex64,
    (torch.complex64, torch.complex128): torch.complex128,
    (torch.complex128, torch.complex64): torch.complex128,
    (torch.complex128, torch.complex128): torch.complex128,
    # Mixed float and complex
    (torch.float32, torch.complex64): torch.complex64,
    (torch.float32, torch.complex128): torch.complex128,
    (torch.float64, torch.complex64): torch.complex128,
    (torch.float64, torch.complex128): torch.complex128,
}


def _two_arg(f):
    @_wraps(f)
    def _f(x1, x2, /, **kwargs):
        x1, x2 = _fix_promotion(x1, x2)
        return f(x1, x2, **kwargs)
    if _f.__doc__ is None:
        _f.__doc__ = f"""\
Array API compatibility wrapper for torch.{f.__name__}.

See the corresponding PyTorch documentation and/or the array API specification
for more details.

"""
    return _f

def _fix_promotion(x1, x2, only_scalar=True):
    if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
        return x1, x2
    if x1.dtype not in _array_api_dtypes or x2.dtype not in _array_api_dtypes:
        return x1, x2
    # If an argument is 0-D pytorch downcasts the other argument
    if not only_scalar or x1.shape == ():
        dtype = result_type(x1, x2)
        x2 = x2.to(dtype)
    if not only_scalar or x2.shape == ():
        dtype = result_type(x1, x2)
        x1 = x1.to(dtype)
    return x1, x2

def result_type(*arrays_and_dtypes: Union[array, Dtype]) -> Dtype:
    if len(arrays_and_dtypes) == 0:
        raise TypeError("At least one array or dtype must be provided")
    if len(arrays_and_dtypes) == 1:
        x = arrays_and_dtypes[0]
        if isinstance(x, torch.dtype):
            return x
        return x.dtype
    if len(arrays_and_dtypes) > 2:
        return result_type(arrays_and_dtypes[0], result_type(*arrays_and_dtypes[1:]))

    x, y = arrays_and_dtypes
    xdt = x.dtype if not isinstance(x, torch.dtype) else x
    ydt = y.dtype if not isinstance(y, torch.dtype) else y

    if (xdt, ydt) in _promotion_table:
        return _promotion_table[xdt, ydt]

    # This doesn't result_type(dtype, dtype) for non-array API dtypes
    # because torch.result_type only accepts tensors. This does however, allow
    # cross-kind promotion.
    x = torch.tensor([], dtype=x) if isinstance(x, torch.dtype) else x
    y = torch.tensor([], dtype=y) if isinstance(y, torch.dtype) else y
    return torch.result_type(x, y)

def can_cast(from_: Union[Dtype, array], to: Dtype, /) -> bool:
    if not isinstance(from_, torch.dtype):
        from_ = from_.dtype
    return torch.can_cast(from_, to)

# Basic renames
bitwise_invert = torch.bitwise_not
newaxis = None
# torch.conj sets the conjugation bit, which breaks conversion to other
# libraries. See https://github.com/data-apis/array-api-compat/issues/173
conj = torch.conj_physical

# Two-arg elementwise functions
# These require a wrapper to do the correct type promotion on 0-D tensors
add = _two_arg(torch.add)
atan2 = _two_arg(torch.atan2)
bitwise_and = _two_arg(torch.bitwise_and)
bitwise_left_shift = _two_arg(torch.bitwise_left_shift)
bitwise_or = _two_arg(torch.bitwise_or)
bitwise_right_shift = _two_arg(torch.bitwise_right_shift)
bitwise_xor = _two_arg(torch.bitwise_xor)
copysign = _two_arg(torch.copysign)
divide = _two_arg(torch.divide)
# Also a rename. torch.equal does not broadcast
equal = _two_arg(torch.eq)
floor_divide = _two_arg(torch.floor_divide)
greater = _two_arg(torch.greater)
greater_equal = _two_arg(torch.greater_equal)
hypot = _two_arg(torch.hypot)
less = _two_arg(torch.less)
less_equal = _two_arg(torch.less_equal)
logaddexp = _two_arg(torch.logaddexp)
# logical functions are not included here because they only accept bool in the
# spec, so type promotion is irrelevant.
maximum = _two_arg(torch.maximum)
minimum = _two_arg(torch.minimum)
multiply = _two_arg(torch.multiply)
not_equal = _two_arg(torch.not_equal)
pow = _two_arg(torch.pow)
remainder = _two_arg(torch.remainder)
subtract = _two_arg(torch.subtract)

# These wrappers are mostly based on the fact that pytorch uses 'dim' instead
# of 'axis'.

# torch.min and torch.max return a tuple and don't support multiple axes https://github.com/pytorch/pytorch/issues/58745
def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amax(x, axis, keepdims=keepdims)

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amin(x, axis, keepdims=keepdims)

clip = get_xp(torch)(_aliases_clip)
unstack = get_xp(torch)(_aliases_unstack)
cumulative_sum = get_xp(torch)(_aliases_cumulative_sum)

# torch.sort also returns a tuple
# https://github.com/pytorch/pytorch/issues/70921
def sort(x: array, /, *, axis: int = -1, descending: bool = False, stable: bool = True, **kwargs) -> array:
    return torch.sort(x, dim=axis, descending=descending, stable=stable, **kwargs).values

def _normalize_axes(axis, ndim):
    axes = []
    if ndim == 0 and axis:
        # Better error message in this case
        raise IndexError(f"Dimension out of range: {axis[0]}")
    lower, upper = -ndim, ndim - 1
    for a in axis:
        if a < lower or a > upper:
            # Match torch error message (e.g., from sum())
            raise IndexError(f"Dimension out of range (expected to be in range of [{lower}, {upper}], but got {a}")
        if a < 0:
            a = a + ndim
        if a in axes:
            # Use IndexError instead of RuntimeError, and "axis" instead of "dim"
            raise IndexError(f"Axis {a} appears multiple times in the list of axes")
        axes.append(a)
    return sorted(axes)

def _axis_none_keepdims(x, ndim, keepdims):
    # Apply keepdims when axis=None
    # (https://github.com/pytorch/pytorch/issues/71209)
    # Note that this is only valid for the axis=None case.
    if keepdims:
        for i in range(ndim):
            x = torch.unsqueeze(x, 0)
    return x

def _reduce_multiple_axes(f, x, axis, keepdims=False, **kwargs):
    # Some reductions don't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    axes = _normalize_axes(axis, x.ndim)
    for a in reversed(axes):
        x = torch.movedim(x, a, -1)
    x = torch.flatten(x, -len(axes))

    out = f(x, -1, **kwargs)

    if keepdims:
        for a in axes:
            out = torch.unsqueeze(out, a)
    return out

def prod(x: array,
         /,
         *,
         axis: Optional[Union[int, Tuple[int, ...]]] = None,
         dtype: Optional[Dtype] = None,
         keepdims: bool = False,
         **kwargs) -> array:
    x = torch.asarray(x)
    ndim = x.ndim

    # https://github.com/pytorch/pytorch/issues/29137. Separate from the logic
    # below because it still needs to upcast.
    if axis == ():
        if dtype is None:
            # We can't upcast uint8 according to the spec because there is no
            # torch.uint64, so at least upcast to int64 which is what sum does
            # when axis=None.
            if x.dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
                return x.to(torch.int64)
            return x.clone()
        return x.to(dtype)

    # torch.prod doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    if isinstance(axis, tuple):
        return _reduce_multiple_axes(torch.prod, x, axis, keepdims=keepdims, dtype=dtype, **kwargs)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.prod(x, dtype=dtype, **kwargs)
        res = _axis_none_keepdims(res, ndim, keepdims)
        return res

    return torch.prod(x, axis, dtype=dtype, keepdims=keepdims, **kwargs)


def sum(x: array,
         /,
         *,
         axis: Optional[Union[int, Tuple[int, ...]]] = None,
         dtype: Optional[Dtype] = None,
         keepdims: bool = False,
         **kwargs) -> array:
    x = torch.asarray(x)
    ndim = x.ndim

    # https://github.com/pytorch/pytorch/issues/29137.
    # Make sure it upcasts.
    if axis == ():
        if dtype is None:
            # We can't upcast uint8 according to the spec because there is no
            # torch.uint64, so at least upcast to int64 which is what sum does
            # when axis=None.
            if x.dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
                return x.to(torch.int64)
            return x.clone()
        return x.to(dtype)

    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.sum(x, dtype=dtype, **kwargs)
        res = _axis_none_keepdims(res, ndim, keepdims)
        return res

    return torch.sum(x, axis, dtype=dtype, keepdims=keepdims, **kwargs)

def any(x: array,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        **kwargs) -> array:
    x = torch.asarray(x)
    ndim = x.ndim
    if axis == ():
        return x.to(torch.bool)
    # torch.any doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    if isinstance(axis, tuple):
        res = _reduce_multiple_axes(torch.any, x, axis, keepdims=keepdims, **kwargs)
        return res.to(torch.bool)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.any(x, **kwargs)
        res = _axis_none_keepdims(res, ndim, keepdims)
        return res.to(torch.bool)

    # torch.any doesn't return bool for uint8
    return torch.any(x, axis, keepdims=keepdims).to(torch.bool)

def all(x: array,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
        **kwargs) -> array:
    x = torch.asarray(x)
    ndim = x.ndim
    if axis == ():
        return x.to(torch.bool)
    # torch.all doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    if isinstance(axis, tuple):
        res = _reduce_multiple_axes(torch.all, x, axis, keepdims=keepdims, **kwargs)
        return res.to(torch.bool)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.all(x, **kwargs)
        res = _axis_none_keepdims(res, ndim, keepdims)
        return res.to(torch.bool)

    # torch.all doesn't return bool for uint8
    return torch.all(x, axis, keepdims=keepdims).to(torch.bool)

def mean(x: array,
         /,
         *,
         axis: Optional[Union[int, Tuple[int, ...]]] = None,
         keepdims: bool = False,
         **kwargs) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.mean(x, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return torch.mean(x, axis, keepdims=keepdims, **kwargs)

def std(x: array,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        **kwargs) -> array:
    # Note, float correction is not supported
    # https://github.com/pytorch/pytorch/issues/61492. We don't try to
    # implement it here for now.

    if isinstance(correction, float):
        _correction = int(correction)
        if correction != _correction:
            raise NotImplementedError("float correction in torch std() is not yet supported")
    else:
        _correction = correction

    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.zeros_like(x)
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.std(x, tuple(range(x.ndim)), correction=_correction, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return torch.std(x, axis, correction=_correction, keepdims=keepdims, **kwargs)

def var(x: array,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        **kwargs) -> array:
    # Note, float correction is not supported
    # https://github.com/pytorch/pytorch/issues/61492. We don't try to
    # implement it here for now.

    # if isinstance(correction, float):
    #     correction = int(correction)

    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.zeros_like(x)
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.var(x, tuple(range(x.ndim)), correction=correction, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return torch.var(x, axis, correction=correction, keepdims=keepdims, **kwargs)

# torch.concat doesn't support dim=None
# https://github.com/pytorch/pytorch/issues/70925
def concat(arrays: Union[Tuple[array, ...], List[array]],
           /,
           *,
           axis: Optional[int] = 0,
           **kwargs) -> array:
    if axis is None:
        arrays = tuple(ar.flatten() for ar in arrays)
        axis = 0
    return torch.concat(arrays, axis, **kwargs)

# torch.squeeze only accepts int dim and doesn't require it
# https://github.com/pytorch/pytorch/issues/70924. Support for tuple dim was
# added at https://github.com/pytorch/pytorch/pull/89017.
def squeeze(x: array, /, axis: Union[int, Tuple[int, ...]]) -> array:
    if isinstance(axis, int):
        axis = (axis,)
    for a in axis:
        if x.shape[a] != 1:
            raise ValueError("squeezed dimensions must be equal to 1")
    axes = _normalize_axes(axis, x.ndim)
    # Remove this once pytorch 1.14 is released with the above PR #89017.
    sequence = [a - i for i, a in enumerate(axes)]
    for a in sequence:
        x = torch.squeeze(x, a)
    return x

# torch.broadcast_to uses size instead of shape
def broadcast_to(x: array, /, shape: Tuple[int, ...], **kwargs) -> array:
    return torch.broadcast_to(x, shape, **kwargs)

# torch.permute uses dims instead of axes
def permute_dims(x: array, /, axes: Tuple[int, ...]) -> array:
    return torch.permute(x, axes)

# The axis parameter doesn't work for flip() and roll()
# https://github.com/pytorch/pytorch/issues/71210. Also torch.flip() doesn't
# accept axis=None
def flip(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs) -> array:
    if axis is None:
        axis = tuple(range(x.ndim))
    # torch.flip doesn't accept dim as an int but the method does
    # https://github.com/pytorch/pytorch/issues/18095
    return x.flip(axis, **kwargs)

def roll(x: array, /, shift: Union[int, Tuple[int, ...]], *, axis: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs) -> array:
    return torch.roll(x, shift, axis, **kwargs)

def nonzero(x: array, /, **kwargs) -> Tuple[array, ...]:
    if x.ndim == 0:
        raise ValueError("nonzero() does not support zero-dimensional arrays")
    return torch.nonzero(x, as_tuple=True, **kwargs)

def where(condition: array, x1: array, x2: array, /) -> array:
    x1, x2 = _fix_promotion(x1, x2)
    return torch.where(condition, x1, x2)

# torch.reshape doesn't have the copy keyword
def reshape(x: array,
            /,
            shape: Tuple[int, ...],
            copy: Optional[bool] = None,
            **kwargs) -> array:
    if copy is not None:
        raise NotImplementedError("torch.reshape doesn't yet support the copy keyword")
    return torch.reshape(x, shape, **kwargs)

# torch.arange doesn't support returning empty arrays
# (https://github.com/pytorch/pytorch/issues/70915), and doesn't support some
# keyword argument combinations
# (https://github.com/pytorch/pytorch/issues/70914)
def arange(start: Union[int, float],
           /,
           stop: Optional[Union[int, float]] = None,
           step: Union[int, float] = 1,
           *,
           dtype: Optional[Dtype] = None,
           device: Optional[Device] = None,
           **kwargs) -> array:
    if stop is None:
        start, stop = 0, start
    if step > 0 and stop <= start or step < 0 and stop >= start:
        if dtype is None:
            if _builtin_all(isinstance(i, int) for i in [start, stop, step]):
                dtype = torch.int64
            else:
                dtype = torch.float32
        return torch.empty(0, dtype=dtype, device=device, **kwargs)
    return torch.arange(start, stop, step, dtype=dtype, device=device, **kwargs)

# torch.eye does not accept None as a default for the second argument and
# doesn't support off-diagonals (https://github.com/pytorch/pytorch/issues/70910)
def eye(n_rows: int,
        n_cols: Optional[int] = None,
        /,
        *,
        k: int = 0,
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
        **kwargs) -> array:
    if n_cols is None:
        n_cols = n_rows
    z = torch.zeros(n_rows, n_cols, dtype=dtype, device=device, **kwargs)
    if abs(k) <= n_rows + n_cols:
        z.diagonal(k).fill_(1)
    return z

# torch.linspace doesn't have the endpoint parameter
def linspace(start: Union[int, float],
             stop: Union[int, float],
             /,
             num: int,
             *,
             dtype: Optional[Dtype] = None,
             device: Optional[Device] = None,
             endpoint: bool = True,
             **kwargs) -> array:
    if not endpoint:
        return torch.linspace(start, stop, num+1, dtype=dtype, device=device, **kwargs)[:-1]
    return torch.linspace(start, stop, num, dtype=dtype, device=device, **kwargs)

# torch.full does not accept an int size
# https://github.com/pytorch/pytorch/issues/70906
def full(shape: Union[int, Tuple[int, ...]],
         fill_value: Union[bool, int, float, complex],
         *,
         dtype: Optional[Dtype] = None,
         device: Optional[Device] = None,
         **kwargs) -> array:
    if isinstance(shape, int):
        shape = (shape,)

    return torch.full(shape, fill_value, dtype=dtype, device=device, **kwargs)

# ones, zeros, and empty do not accept shape as a keyword argument
def ones(shape: Union[int, Tuple[int, ...]],
         *,
         dtype: Optional[Dtype] = None,
         device: Optional[Device] = None,
         **kwargs) -> array:
    return torch.ones(shape, dtype=dtype, device=device, **kwargs)

def zeros(shape: Union[int, Tuple[int, ...]],
         *,
         dtype: Optional[Dtype] = None,
         device: Optional[Device] = None,
         **kwargs) -> array:
    return torch.zeros(shape, dtype=dtype, device=device, **kwargs)

def empty(shape: Union[int, Tuple[int, ...]],
         *,
         dtype: Optional[Dtype] = None,
         device: Optional[Device] = None,
         **kwargs) -> array:
    return torch.empty(shape, dtype=dtype, device=device, **kwargs)

# tril and triu do not call the keyword argument k

def tril(x: array, /, *, k: int = 0) -> array:
    return torch.tril(x, k)

def triu(x: array, /, *, k: int = 0) -> array:
    return torch.triu(x, k)

# Functions that aren't in torch https://github.com/pytorch/pytorch/issues/58742
def expand_dims(x: array, /, *, axis: int = 0) -> array:
    return torch.unsqueeze(x, axis)

def astype(x: array, dtype: Dtype, /, *, copy: bool = True) -> array:
    return x.to(dtype, copy=copy)

def broadcast_arrays(*arrays: array) -> List[array]:
    shape = torch.broadcast_shapes(*[a.shape for a in arrays])
    return [torch.broadcast_to(a, shape) for a in arrays]

# Note that these named tuples aren't actually part of the standard namespace,
# but I don't see any issue with exporting the names here regardless.
from ..common._aliases import (UniqueAllResult, UniqueCountsResult,
                               UniqueInverseResult)

# https://github.com/pytorch/pytorch/issues/70920
def unique_all(x: array) -> UniqueAllResult:
    # torch.unique doesn't support returning indices.
    # https://github.com/pytorch/pytorch/issues/36748. The workaround
    # suggested in that issue doesn't actually function correctly (it relies
    # on non-deterministic behavior of scatter()).
    raise NotImplementedError("unique_all() not yet implemented for pytorch (see https://github.com/pytorch/pytorch/issues/36748)")

    # values, inverse_indices, counts = torch.unique(x, return_counts=True, return_inverse=True)
    # # torch.unique incorrectly gives a 0 count for nan values.
    # # https://github.com/pytorch/pytorch/issues/94106
    # counts[torch.isnan(values)] = 1
    # return UniqueAllResult(values, indices, inverse_indices, counts)

def unique_counts(x: array) -> UniqueCountsResult:
    values, counts = torch.unique(x, return_counts=True)

    # torch.unique incorrectly gives a 0 count for nan values.
    # https://github.com/pytorch/pytorch/issues/94106
    counts[torch.isnan(values)] = 1
    return UniqueCountsResult(values, counts)

def unique_inverse(x: array) -> UniqueInverseResult:
    values, inverse = torch.unique(x, return_inverse=True)
    return UniqueInverseResult(values, inverse)

def unique_values(x: array) -> array:
    return torch.unique(x)

def matmul(x1: array, x2: array, /, **kwargs) -> array:
    # torch.matmul doesn't type promote (but differently from _fix_promotion)
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return torch.matmul(x1, x2, **kwargs)

matrix_transpose = get_xp(torch)(_aliases_matrix_transpose)
_vecdot = get_xp(torch)(_aliases_vecdot)

def vecdot(x1: array, x2: array, /, *, axis: int = -1) -> array:
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return _vecdot(x1, x2, axis=axis)

# torch.tensordot uses dims instead of axes
def tensordot(x1: array, x2: array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2, **kwargs) -> array:
    # Note: torch.tensordot fails with integer dtypes when there is only 1
    # element in the axis (https://github.com/pytorch/pytorch/issues/84530).
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return torch.tensordot(x1, x2, dims=axes, **kwargs)


def isdtype(
    dtype: Dtype, kind: Union[Dtype, str, Tuple[Union[Dtype, str], ...]],
    *, _tuple=True, # Disallow nested tuples
) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.

    Note that outside of this function, this compat library does not yet fully
    support complex numbers.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    for more details
    """
    if isinstance(kind, tuple) and _tuple:
        return _builtin_any(isdtype(dtype, k, _tuple=False) for k in kind)
    elif isinstance(kind, str):
        if kind == 'bool':
            return dtype == torch.bool
        elif kind == 'signed integer':
            return dtype in _int_dtypes and dtype.is_signed
        elif kind == 'unsigned integer':
            return dtype in _int_dtypes and not dtype.is_signed
        elif kind == 'integral':
            return dtype in _int_dtypes
        elif kind == 'real floating':
            return dtype.is_floating_point
        elif kind == 'complex floating':
            return dtype.is_complex
        elif kind == 'numeric':
            return isdtype(dtype, ('integral', 'real floating', 'complex floating'))
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    else:
        return dtype == kind

def take(x: array, indices: array, /, *, axis: Optional[int] = None, **kwargs) -> array:
    if axis is None:
        if x.ndim != 1:
            raise ValueError("axis must be specified when ndim > 1")
        axis = 0
    return torch.index_select(x, axis, indices, **kwargs)

def sign(x: array, /) -> array:
    # torch sign() does not support complex numbers and does not propagate
    # nans. See https://github.com/data-apis/array-api-compat/issues/136
    if x.dtype.is_complex:
        out = x/torch.abs(x)
        # sign(0) = 0 but the above formula would give nan
        out[x == 0+0j] = 0+0j
        return out
    else:
        out = torch.sign(x)
        if x.dtype.is_floating_point:
            out[torch.isnan(x)] = torch.nan
        return out


__all__ = ['__array_namespace_info__', 'result_type', 'can_cast',
           'permute_dims', 'bitwise_invert', 'newaxis', 'conj', 'add',
           'atan2', 'bitwise_and', 'bitwise_left_shift', 'bitwise_or',
           'bitwise_right_shift', 'bitwise_xor', 'copysign', 'divide',
           'equal', 'floor_divide', 'greater', 'greater_equal', 'hypot',
           'less', 'less_equal', 'logaddexp', 'maximum', 'minimum',
           'multiply', 'not_equal', 'pow', 'remainder', 'subtract', 'max',
           'min', 'clip', 'unstack', 'cumulative_sum', 'sort', 'prod', 'sum',
           'any', 'all', 'mean', 'std', 'var', 'concat', 'squeeze',
           'broadcast_to', 'flip', 'roll', 'nonzero', 'where', 'reshape',
           'arange', 'eye', 'linspace', 'full', 'ones', 'zeros', 'empty',
           'tril', 'triu', 'expand_dims', 'astype', 'broadcast_arrays',
           'UniqueAllResult', 'UniqueCountsResult', 'UniqueInverseResult',
           'unique_all', 'unique_counts', 'unique_inverse', 'unique_values',
           'matmul', 'matrix_transpose', 'vecdot', 'tensordot', 'isdtype',
           'take', 'sign']

_all_ignore = ['torch', 'get_xp']
<|endoftext|>from cupy.linalg import * # noqa: F403
# cupy.linalg doesn't have __all__. If it is added, replace this with
#
# from cupy.linalg import __all__ as linalg_all
_n = {}
exec('from cupy.linalg import *', _n)
del _n['__builtins__']
linalg_all = list(_n)
del _n

from ..common import _linalg
from .._internal import get_xp

import cupy as cp

# These functions are in both the main and linalg namespaces
from ._aliases import matmul, matrix_transpose, tensordot, vecdot # noqa: F401

cross = get_xp(cp)(_linalg.cross)
outer = get_xp(cp)(_linalg.outer)
EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
eigh = get_xp(cp)(_linalg.eigh)
qr = get_xp(cp)(_linalg.qr)
slogdet = get_xp(cp)(_linalg.slogdet)
svd = get_xp(cp)(_linalg.svd)
cholesky = get_xp(cp)(_linalg.cholesky)
matrix_rank = get_xp(cp)(_linalg.matrix_rank)
pinv = get_xp(cp)(_linalg.pinv)
matrix_norm = get_xp(cp)(_linalg.matrix_norm)
svdvals = get_xp(cp)(_linalg.svdvals)
diagonal = get_xp(cp)(_linalg.diagonal)
trace = get_xp(cp)(_linalg.trace)

# These functions are completely new here. If the library already has them
# (i.e., numpy 2.0), use the library version instead of our wrapper.
if hasattr(cp.linalg, 'vector_norm'):
    vector_norm = cp.linalg.vector_norm
else:
    vector_norm = get_xp(cp)(_linalg.vector_norm)

__all__ = linalg_all + _linalg.__all__

del get_xp
del cp
del linalg_all
del _linalg
<|endoftext|>from __future__ import annotations

__all__ = [
    "ndarray",
    "Device",
    "Dtype",
]

import sys
from typing import (
    Union,
    TYPE_CHECKING,
)

from cupy import (
    ndarray,
    dtype,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
)

from cupy.cuda.device import Device

if TYPE_CHECKING or sys.version_info >= (3, 9):
    Dtype = dtype[Union[
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
    ]]
else:
    Dtype = dtype
<|endoftext|>from cupy import * # noqa: F403

# from cupy import * doesn't overwrite these builtin names
from cupy import abs, max, min, round # noqa: F401

# These imports may overwrite names from the import * above.
from ._aliases import * # noqa: F403

# See the comment in the numpy __init__.py
__import__(__package__ + '.linalg')

__import__(__package__ + '.fft')

from ..common._helpers import * # noqa: F401,F403

__array_api_version__ = '2023.12'
<|endoftext|>from cupy.fft import * # noqa: F403
# cupy.fft doesn't have __all__. If it is added, replace this with
#
# from cupy.fft import __all__ as linalg_all
_n = {}
exec('from cupy.fft import *', _n)
del _n['__builtins__']
fft_all = list(_n)
del _n

from ..common import _fft
from .._internal import get_xp

import cupy as cp

fft = get_xp(cp)(_fft.fft)
ifft = get_xp(cp)(_fft.ifft)
fftn = get_xp(cp)(_fft.fftn)
ifftn = get_xp(cp)(_fft.ifftn)
rfft = get_xp(cp)(_fft.rfft)
irfft = get_xp(cp)(_fft.irfft)
rfftn = get_xp(cp)(_fft.rfftn)
irfftn = get_xp(cp)(_fft.irfftn)
hfft = get_xp(cp)(_fft.hfft)
ihfft = get_xp(cp)(_fft.ihfft)
fftfreq = get_xp(cp)(_fft.fftfreq)
rfftfreq = get_xp(cp)(_fft.rfftfreq)
fftshift = get_xp(cp)(_fft.fftshift)
ifftshift = get_xp(cp)(_fft.ifftshift)

__all__ = fft_all + _fft.__all__

del get_xp
del cp
del fft_all
del _fft
<|endoftext|>"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""
from cupy import (
    dtype,
    cuda,
    bool_ as bool,
    intp,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
)

class __array_namespace_info__:
    """
    Get the array API inspection namespace for CuPy.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for CuPy.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': cupy.float64,
     'complex floating': cupy.complex128,
     'integral': cupy.int64,
     'indexing': cupy.int64}

    """

    __module__ = 'cupy'

    def capabilities(self):
        """
        Return a dictionary of array API library capabilities.

        The resulting dictionary has the following keys:

        - **"boolean indexing"**: boolean indicating whether an array library
          supports boolean indexing. Always ``True`` for CuPy.

        - **"data-dependent shapes"**: boolean indicating whether an array
          library supports data-dependent output shapes. Always ``True`` for
          CuPy.

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.

        See Also
        --------
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        capabilities : dict
            A dictionary of array API library capabilities.

        Examples
        --------
        >>> info = xp.__array_namespace_info__()
        >>> info.capabilities()
        {'boolean indexing': True,
         'data-dependent shapes': True}

        """
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            # 'max rank' will be part of the 2024.12 standard
            # "max rank": 64,
        }

    def default_device(self):
        """
        The default device used for new CuPy arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        device : str
            The default device used for new CuPy arrays.

        Examples
        --------
        >>> info = xp.__array_namespace_info__()
        >>> info.default_device()
        Device(0)

        """
        return cuda.Device(0)

    def default_dtypes(self, *, device=None):
        """
        The default data types used for new CuPy arrays.

        For CuPy, this always returns the following dictionary:

        - **"real floating"**: ``cupy.float64``
        - **"complex floating"**: ``cupy.complex128``
        - **"integral"**: ``cupy.intp``
        - **"indexing"**: ``cupy.intp``

        Parameters
        ----------
        device : str, optional
            The device to get the default data types for.

        Returns
        -------
        dtypes : dict
            A dictionary describing the default data types used for new CuPy
            arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = xp.__array_namespace_info__()
        >>> info.default_dtypes()
        {'real floating': cupy.float64,
         'complex floating': cupy.complex128,
         'integral': cupy.int64,
         'indexing': cupy.int64}

        """
        # TODO: Does this depend on device?
        return {
            "real floating": dtype(float64),
            "complex floating": dtype(complex128),
            "integral": dtype(intp),
            "indexing": dtype(intp),
        }

    def dtypes(self, *, device=None, kind=None):
        """
        The array API data types supported by CuPy.

        Note that this function only returns data types that are defined by
        the array API.

        Parameters
        ----------
        device : str, optional
            The device to get the data types for.
        kind : str or tuple of str, optional
            The kind of data types to return. If ``None``, all data types are
            returned. If a string, only data types of that kind are returned.
            If a tuple, a dictionary containing the union of the given kinds
            is returned. The following kinds are supported:

            - ``'bool'``: boolean data types (i.e., ``bool``).
            - ``'signed integer'``: signed integer data types (i.e., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (i.e.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (i.e., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (i.e.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

        Returns
        -------
        dtypes : dict
            A dictionary mapping the names of data types to the corresponding
            CuPy data types.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = xp.__array_namespace_info__()
        >>> info.dtypes(kind='signed integer')
        {'int8': cupy.int8,
         'int16': cupy.int16,
         'int32': cupy.int32,
         'int64': cupy.int64}

        """
        # TODO: Does this depend on device?
        if kind is None:
            return {
                "bool": dtype(bool),
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
            }
        if kind == "unsigned integer":
            return {
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "integral":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "real floating":
            return {
                "float32": dtype(float32),
                "float64": dtype(float64),
            }
        if kind == "complex floating":
            return {
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "numeric":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self):
        """
        The devices supported by CuPy.

        Returns
        -------
        devices : list of str
            The devices supported by CuPy.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes

        """
        return [cuda.Device(i) for i in range(cuda.runtime.getDeviceCount())]
<|endoftext|>from __future__ import annotations

import cupy as cp

from ..common import _aliases
from .._internal import get_xp

from ._info import __array_namespace_info__

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Union
    from ._typing import ndarray, Device, Dtype, NestedSequence, SupportsBufferProtocol

bool = cp.bool_

# Basic renames
acos = cp.arccos
acosh = cp.arccosh
asin = cp.arcsin
asinh = cp.arcsinh
atan = cp.arctan
atan2 = cp.arctan2
atanh = cp.arctanh
bitwise_left_shift = cp.left_shift
bitwise_invert = cp.invert
bitwise_right_shift = cp.right_shift
concat = cp.concatenate
pow = cp.power

arange = get_xp(cp)(_aliases.arange)
empty = get_xp(cp)(_aliases.empty)
empty_like = get_xp(cp)(_aliases.empty_like)
eye = get_xp(cp)(_aliases.eye)
full = get_xp(cp)(_aliases.full)
full_like = get_xp(cp)(_aliases.full_like)
linspace = get_xp(cp)(_aliases.linspace)
ones = get_xp(cp)(_aliases.ones)
ones_like = get_xp(cp)(_aliases.ones_like)
zeros = get_xp(cp)(_aliases.zeros)
zeros_like = get_xp(cp)(_aliases.zeros_like)
UniqueAllResult = get_xp(cp)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(cp)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(cp)(_aliases.UniqueInverseResult)
unique_all = get_xp(cp)(_aliases.unique_all)
unique_counts = get_xp(cp)(_aliases.unique_counts)
unique_inverse = get_xp(cp)(_aliases.unique_inverse)
unique_values = get_xp(cp)(_aliases.unique_values)
astype = _aliases.astype
std = get_xp(cp)(_aliases.std)
var = get_xp(cp)(_aliases.var)
cumulative_sum = get_xp(cp)(_aliases.cumulative_sum)
clip = get_xp(cp)(_aliases.clip)
permute_dims = get_xp(cp)(_aliases.permute_dims)
reshape = get_xp(cp)(_aliases.reshape)
argsort = get_xp(cp)(_aliases.argsort)
sort = get_xp(cp)(_aliases.sort)
nonzero = get_xp(cp)(_aliases.nonzero)
ceil = get_xp(cp)(_aliases.ceil)
floor = get_xp(cp)(_aliases.floor)
trunc = get_xp(cp)(_aliases.trunc)
matmul = get_xp(cp)(_aliases.matmul)
matrix_transpose = get_xp(cp)(_aliases.matrix_transpose)
tensordot = get_xp(cp)(_aliases.tensordot)
sign = get_xp(cp)(_aliases.sign)

_copy_default = object()

# asarray also adds the copy keyword, which is not present in numpy 1.0.
def asarray(
    obj: Union[
        ndarray,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsBufferProtocol,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = _copy_default,
    **kwargs,
) -> ndarray:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    with cp.cuda.Device(device):
        # cupy is like NumPy 1.26 (except without _CopyMode). See the comments
        # in asarray in numpy/_aliases.py.
        if copy is not _copy_default:
            # A future version of CuPy will change the meaning of copy=False
            # to mean no-copy. We don't know for certain what version it will
            # be yet, so to avoid breaking that version, we use a different
            # default value for copy so asarray(obj) with no copy kwarg will
            # always do the copy-if-needed behavior.

            # This will still need to be updated to remove the
            # NotImplementedError for copy=False, but at least this won't
            # break the default or existing behavior.
            if copy is None:
                copy = False
            elif copy is False:
                raise NotImplementedError("asarray(copy=False) is not yet supported in cupy")
            kwargs['copy'] = copy

        return cp.array(obj, dtype=dtype, **kwargs)

# These functions are completely new here. If the library already has them
# (i.e., numpy 2.0), use the library version instead of our wrapper.
if hasattr(cp, 'vecdot'):
    vecdot = cp.vecdot
else:
    vecdot = get_xp(cp)(_aliases.vecdot)

if hasattr(cp, 'isdtype'):
    isdtype = cp.isdtype
else:
    isdtype = get_xp(cp)(_aliases.isdtype)

if hasattr(cp, 'unstack'):
    unstack = cp.unstack
else:
    unstack = get_xp(cp)(_aliases.unstack)

__all__ = _aliases.__all__ + ['__array_namespace_info__', 'asarray', 'bool',
                              'acos', 'acosh', 'asin', 'asinh', 'atan',
                              'atan2', 'atanh', 'bitwise_left_shift',
                              'bitwise_invert', 'bitwise_right_shift',
                              'concat', 'pow', 'sign']

_all_ignore = ['cp', 'get_xp']
<|endoftext|>import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
import threading

from ._uarray import (  # type: ignore
    BackendNotImplementedError,
    _Function,
    _SkipBackendContext,
    _SetBackendContext,
    _BackendState,
)

__all__ = [
    "set_backend",
    "set_global_backend",
    "skip_backend",
    "register_backend",
    "determine_backend",
    "determine_backend_multi",
    "clear_backends",
    "create_multimethod",
    "generate_multimethod",
    "_Function",
    "BackendNotImplementedError",
    "Dispatchable",
    "wrap_single_convertor",
    "wrap_single_convertor_instance",
    "all_of_type",
    "mark_as",
    "set_state",
    "get_state",
    "reset_state",
    "_BackendState",
    "_SkipBackendContext",
    "_SetBackendContext",
]

ArgumentExtractorType = typing.Callable[..., tuple["Dispatchable", ...]]
ArgumentReplacerType = typing.Callable[
    [tuple, dict, tuple], tuple[tuple, dict]
]

def unpickle_function(mod_name, qname, self_):
    import importlib

    try:
        module = importlib.import_module(mod_name)
        qname = qname.split(".")
        func = module
        for q in qname:
            func = getattr(func, q)

        if self_ is not None:
            func = types.MethodType(func, self_)

        return func
    except (ImportError, AttributeError) as e:
        from pickle import UnpicklingError

        raise UnpicklingError from e


def pickle_function(func):
    mod_name = getattr(func, "__module__", None)
    qname = getattr(func, "__qualname__", None)
    self_ = getattr(func, "__self__", None)

    try:
        test = unpickle_function(mod_name, qname, self_)
    except pickle.UnpicklingError:
        test = None

    if test is not func:
        raise pickle.PicklingError(
            f"Can't pickle {func}: it's not the same object as {test}"
        )

    return unpickle_function, (mod_name, qname, self_)


def pickle_state(state):
    return _uarray._BackendState._unpickle, state._pickle()


def pickle_set_backend_context(ctx):
    return _SetBackendContext, ctx._pickle()


def pickle_skip_backend_context(ctx):
    return _SkipBackendContext, ctx._pickle()


copyreg.pickle(_Function, pickle_function)
copyreg.pickle(_uarray._BackendState, pickle_state)
copyreg.pickle(_SetBackendContext, pickle_set_backend_context)
copyreg.pickle(_SkipBackendContext, pickle_skip_backend_context)


def get_state():
    """
    Returns an opaque object containing the current state of all the backends.

    Can be used for synchronization between threads/processes.

    See Also
    --------
    set_state
        Sets the state returned by this function.
    """
    return _uarray.get_state()


@contextlib.contextmanager
def reset_state():
    """
    Returns a context manager that resets all state once exited.

    See Also
    --------
    set_state
        Context manager that sets the backend state.
    get_state
        Gets a state to be set by this context manager.
    """
    with set_state(get_state()):
        yield


@contextlib.contextmanager
def set_state(state):
    """
    A context manager that sets the state of the backends to one returned by :obj:`get_state`.

    See Also
    --------
    get_state
        Gets a state to be set by this context manager.
    """  # noqa: E501
    old_state = get_state()
    _uarray.set_state(state)
    try:
        yield
    finally:
        _uarray.set_state(old_state, True)


def create_multimethod(*args, **kwargs):
    """
    Creates a decorator for generating multimethods.

    This function creates a decorator that can be used with an argument
    extractor in order to generate a multimethod. Other than for the
    argument extractor, all arguments are passed on to
    :obj:`generate_multimethod`.

    See Also
    --------
    generate_multimethod
        Generates a multimethod.
    """

    def wrapper(a):
        return generate_multimethod(a, *args, **kwargs)

    return wrapper


def generate_multimethod(
    argument_extractor: ArgumentExtractorType,
    argument_replacer: ArgumentReplacerType,
    domain: str,
    default: typing.Callable | None = None,
):
    """
    Generates a multimethod.

    Parameters
    ----------
    argument_extractor : ArgumentExtractorType
        A callable which extracts the dispatchable arguments. Extracted arguments
        should be marked by the :obj:`Dispatchable` class. It has the same signature
        as the desired multimethod.
    argument_replacer : ArgumentReplacerType
        A callable with the signature (args, kwargs, dispatchables), which should also
        return an (args, kwargs) pair with the dispatchables replaced inside the
        args/kwargs.
    domain : str
        A string value indicating the domain of this multimethod.
    default: Optional[Callable], optional
        The default implementation of this multimethod, where ``None`` (the default)
        specifies there is no default implementation.

    Examples
    --------
    In this example, ``a`` is to be dispatched over, so we return it, while marking it
    as an ``int``.
    The trailing comma is needed because the args have to be returned as an iterable.

    >>> def override_me(a, b):
    ...   return Dispatchable(a, int),

    Next, we define the argument replacer that replaces the dispatchables inside
    args/kwargs with the supplied ones.

    >>> def override_replacer(args, kwargs, dispatchables):
    ...     return (dispatchables[0], args[1]), {}

    Next, we define the multimethod.

    >>> overridden_me = generate_multimethod(
    ...     override_me, override_replacer, "ua_examples"
    ... )

    Notice that there's no default implementation, unless you supply one.

    >>> overridden_me(1, "a")
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    >>> overridden_me2 = generate_multimethod(
    ...     override_me, override_replacer, "ua_examples", default=lambda x, y: (x, y)
    ... )
    >>> overridden_me2(1, "a")
    (1, 'a')

    See Also
    --------
    uarray
        See the module documentation for how to override the method by creating
        backends.
    """
    kw_defaults, arg_defaults, opts = get_defaults(argument_extractor)
    ua_func = _Function(
        argument_extractor,
        argument_replacer,
        domain,
        arg_defaults,
        kw_defaults,
        default,
    )

    return functools.update_wrapper(ua_func, argument_extractor)


def set_backend(backend, coerce=False, only=False):
    """
    A context manager that sets the preferred backend.

    Parameters
    ----------
    backend
        The backend to set.
    coerce
        Whether or not to coerce to a specific backend's types. Implies ``only``.
    only
        Whether or not this should be the last backend to try.

    See Also
    --------
    skip_backend: A context manager that allows skipping of backends.
    set_global_backend: Set a single, global backend for a domain.
    """
    tid = threading.get_native_id()
    try:
        return backend.__ua_cache__[tid, "set", coerce, only]
    except AttributeError:
        backend.__ua_cache__ = {}
    except KeyError:
        pass

    ctx = _SetBackendContext(backend, coerce, only)
    backend.__ua_cache__[tid, "set", coerce, only] = ctx
    return ctx


def skip_backend(backend):
    """
    A context manager that allows one to skip a given backend from processing
    entirely. This allows one to use another backend's code in a library that
    is also a consumer of the same backend.

    Parameters
    ----------
    backend
        The backend to skip.

    See Also
    --------
    set_backend: A context manager that allows setting of backends.
    set_global_backend: Set a single, global backend for a domain.
    """
    tid = threading.get_native_id()
    try:
        return backend.__ua_cache__[tid, "skip"]
    except AttributeError:
        backend.__ua_cache__ = {}
    except KeyError:
        pass

    ctx = _SkipBackendContext(backend)
    backend.__ua_cache__[tid, "skip"] = ctx
    return ctx


def get_defaults(f):
    sig = inspect.signature(f)
    kw_defaults = {}
    arg_defaults = []
    opts = set()
    for k, v in sig.parameters.items():
        if v.default is not inspect.Parameter.empty:
            kw_defaults[k] = v.default
        if v.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            arg_defaults.append(v.default)
        opts.add(k)

    return kw_defaults, tuple(arg_defaults), opts


def set_global_backend(backend, coerce=False, only=False, *, try_last=False):
    """
    This utility method replaces the default backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend. This will be the first tried
    backend outside the :obj:`set_backend` context manager.

    Note that this method is not thread-safe.

    .. warning::
        We caution library authors against using this function in
        their code. We do *not* support this use-case. This function
        is meant to be used only by users themselves, or by a reference
        implementation, if one exists.

    Parameters
    ----------
    backend
        The backend to register.
    coerce : bool
        Whether to coerce input types when trying this backend.
    only : bool
        If ``True``, no more backends will be tried if this fails.
        Implied by ``coerce=True``.
    try_last : bool
        If ``True``, the global backend is tried after registered backends.

    See Also
    --------
    set_backend: A context manager that allows setting of backends.
    skip_backend: A context manager that allows skipping of backends.
    """
    _uarray.set_global_backend(backend, coerce, only, try_last)


def register_backend(backend):
    """
    This utility method sets registers backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend.

    Note that this method is not thread-safe.

    Parameters
    ----------
    backend
        The backend to register.
    """
    _uarray.register_backend(backend)


def clear_backends(domain, registered=True, globals=False):
    """
    This utility method clears registered backends.

    .. warning::
        We caution library authors against using this function in
        their code. We do *not* support this use-case. This function
        is meant to be used only by users themselves.

    .. warning::
        Do NOT use this method inside a multimethod call, or the
        program is likely to crash.

    Parameters
    ----------
    domain : Optional[str]
        The domain for which to de-register backends. ``None`` means
        de-register for all domains.
    registered : bool
        Whether or not to clear registered backends. See :obj:`register_backend`.
    globals : bool
        Whether or not to clear global backends. See :obj:`set_global_backend`.

    See Also
    --------
    register_backend : Register a backend globally.
    set_global_backend : Set a global backend.
    """
    _uarray.clear_backends(domain, registered, globals)


class Dispatchable:
    """
    A utility class which marks an argument with a specific dispatch type.


    Attributes
    ----------
    value
        The value of the Dispatchable.

    type
        The type of the Dispatchable.

    Examples
    --------
    >>> x = Dispatchable(1, str)
    >>> x
    <Dispatchable: type=<class 'str'>, value=1>

    See Also
    --------
    all_of_type
        Marks all unmarked parameters of a function.

    mark_as
        Allows one to create a utility function to mark as a given type.
    """

    def __init__(self, value, dispatch_type, coercible=True):
        self.value = value
        self.type = dispatch_type
        self.coercible = coercible

    def __getitem__(self, index):
        return (self.type, self.value)[index]

    def __str__(self):
        return f"<{type(self).__name__}: type={self.type!r}, value={self.value!r}>"

    __repr__ = __str__


def mark_as(dispatch_type):
    """
    Creates a utility function to mark something as a specific type.

    Examples
    --------
    >>> mark_int = mark_as(int)
    >>> mark_int(1)
    <Dispatchable: type=<class 'int'>, value=1>
    """
    return functools.partial(Dispatchable, dispatch_type=dispatch_type)


def all_of_type(arg_type):
    """
    Marks all unmarked arguments as a given type.

    Examples
    --------
    >>> @all_of_type(str)
    ... def f(a, b):
    ...     return a, Dispatchable(b, int)
    >>> f('a', 1)
    (<Dispatchable: type=<class 'str'>, value='a'>,
     <Dispatchable: type=<class 'int'>, value=1>)
    """

    def outer(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            extracted_args = func(*args, **kwargs)
            return tuple(
                Dispatchable(arg, arg_type)
                if not isinstance(arg, Dispatchable)
                else arg
                for arg in extracted_args
            )

        return inner

    return outer


def wrap_single_convertor(convert_single):
    """
    Wraps a ``__ua_convert__`` defined for a single element to all elements.
    If any of them return ``NotImplemented``, the operation is assumed to be
    undefined.

    Accepts a signature of (value, type, coerce).
    """

    @functools.wraps(convert_single)
    def __ua_convert__(dispatchables, coerce):
        converted = []
        for d in dispatchables:
            c = convert_single(d.value, d.type, coerce and d.coercible)

            if c is NotImplemented:
                return NotImplemented

            converted.append(c)

        return converted

    return __ua_convert__


def wrap_single_convertor_instance(convert_single):
    """
    Wraps a ``__ua_convert__`` defined for a single element to all elements.
    If any of them return ``NotImplemented``, the operation is assumed to be
    undefined.

    Accepts a signature of (value, type, coerce).
    """

    @functools.wraps(convert_single)
    def __ua_convert__(self, dispatchables, coerce):
        converted = []
        for d in dispatchables:
            c = convert_single(self, d.value, d.type, coerce and d.coercible)

            if c is NotImplemented:
                return NotImplemented

            converted.append(c)

        return converted

    return __ua_convert__


def determine_backend(value, dispatch_type, *, domain, only=True, coerce=False):
    """Set the backend to the first active backend that supports ``value``

    This is useful for functions that call multimethods without any dispatchable
    arguments. You can use :func:`determine_backend` to ensure the same backend
    is used everywhere in a block of multimethod calls.

    Parameters
    ----------
    value
        The value being tested
    dispatch_type
        The dispatch type associated with ``value``, aka
        ":ref:`marking <MarkingGlossary>`".
    domain: string
        The domain to query for backends and set.
    coerce: bool
        Whether or not to allow coercion to the backend's types. Implies ``only``.
    only: bool
        Whether or not this should be the last backend to try.

    See Also
    --------
    set_backend: For when you know which backend to set

    Notes
    -----

    Support is determined by the ``__ua_convert__`` protocol. Backends not
    supporting the type must return ``NotImplemented`` from their
    ``__ua_convert__`` if they don't support input of that type.

    Examples
    --------

    Suppose we have two backends ``BackendA`` and ``BackendB`` each supporting
    different types, ``TypeA`` and ``TypeB``. Neither supporting the other type:

    >>> with ua.set_backend(ex.BackendA):
    ...     ex.call_multimethod(ex.TypeB(), ex.TypeB())
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    Now consider a multimethod that creates a new object of ``TypeA``, or
    ``TypeB`` depending on the active backend.

    >>> with ua.set_backend(ex.BackendA), ua.set_backend(ex.BackendB):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, ex.TypeA())
    Traceback (most recent call last):
        ...
    uarray.BackendNotImplementedError: ...

    ``res`` is an object of ``TypeB`` because ``BackendB`` is set in the
    innermost with statement. So, ``call_multimethod`` fails since the types
    don't match.

    Instead, we need to first find a backend suitable for all of our objects.

    >>> with ua.set_backend(ex.BackendA), ua.set_backend(ex.BackendB):
    ...     x = ex.TypeA()
    ...     with ua.determine_backend(x, "mark", domain="ua_examples"):
    ...         res = ex.creation_multimethod()
    ...         ex.call_multimethod(res, x)
    TypeA

    """
    dispatchables = (Dispatchable(value, dispatch_type, coerce),)
    backend = _uarray.determine_backend(domain, dispatchables, coerce)

    return set_backend(backend, coerce=coerce, only=only)


def determine_backend_multi(
    dispatchables, *, domain, only=True, coerce=False, **kwargs
):
    """Set a backend supporting all ``dispatchables``

    This is useful for functions that call multimethods without any dispatchable
    arguments. You can use :func:`determine_backend_multi` to ensure the same
    backend is used everywhere in a block of multimethod calls involving
    multiple arrays.

    Parameters
    ----------
    dispatchables: Sequence[Union[uarray.Dispatchable, Any]]
        The dispatchables that must be supported
    domain: string
        The domain to query for backends and set.
    coerce: bool
        Whether or not to allow coercion to the backend's types. Implies ``only``.
    only: bool
        Whether or not this should be the last backend to try.
    dispatch_type: Optional[Any]
        The default dispatch type associated with ``dispatchables``, aka
        ":ref:`marking <MarkingGlossary>`".

    See Also
    --------
    determine_backend: For a single dispatch value
    set_backend: For when you know which backend to set

    Notes
    -----

    Support is determined by the ``__ua_convert__`` protocol. Backends not
    supporting the type must return ``NotImplemented`` from their
    ``__ua_convert__`` if they don't support input of that type.

    Examples
    --------

    :func:`determine_backend` allows the backend to be set from a single
    object. :func:`determine_backend_multi` allows multiple objects to be
    checked simultaneously for support in the backend. Suppose we have a
    ``BackendAB`` which supports ``TypeA`` and ``TypeB`` in the same call,
    and a ``BackendBC`` that doesn't support ``TypeA``.

    >>> with ua.set_backend(ex.BackendAB), ua.set_backend(ex.BackendBC):
    ...     a, b = ex.TypeA(), ex.TypeB()
    ...     with ua.determine_backend_multi(
    ...         [ua.Dispatchable(a, "mark"), ua.Dispatchable(b, "mark")],
    ...         domain="ua_examples"
    ...     ):
