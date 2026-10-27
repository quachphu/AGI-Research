                    validated_bc.append((2, np.zeros(expected_deriv_shape)))
                elif bc in ['not-a-knot', 'periodic']:
                    validated_bc.append(bc)
                else:
                    raise ValueError(f"bc_type={bc} is not allowed.")
            else:
                try:
                    deriv_order, deriv_value = bc
                except Exception as e:
                    raise ValueError(
                        "A specified derivative value must be "
                        "given in the form (order, value)."
                    ) from e

                if deriv_order not in [1, 2]:
                    raise ValueError("The specified derivative order must "
                                     "be 1 or 2.")

                deriv_value = np.asarray(deriv_value)
                if deriv_value.shape != expected_deriv_shape:
                    raise ValueError(
                        f"`deriv_value` shape {deriv_value.shape} is not "
                        f"the expected one {expected_deriv_shape}."
                    )

                if np.issubdtype(deriv_value.dtype, np.complexfloating):
                    y = y.astype(complex, copy=False)

                validated_bc.append((deriv_order, deriv_value))

        return validated_bc, y
<|endoftext|># Copyright (c) 2017, The Chancellor, Masters and Scholars of the University
# of Oxford, and the Chebfun Developers. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Oxford nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import operator

import numpy as np
import scipy


__all__ = ["AAA", "FloaterHormannInterpolator"]


class _BarycentricRational:
    """Base class for Barycentric representation of a rational function."""
    def __init__(self, x, y, **kwargs):
        # input validation
        z = np.asarray(x)
        f = np.asarray(y)

        self._input_validation(z, f, **kwargs)

        # Remove infinite or NaN function values and repeated entries
        to_keep = np.logical_and.reduce(
            ((np.isfinite(f)) & (~np.isnan(f))).reshape(f.shape[0], -1),
            axis=-1
        )
        f = f[to_keep, ...]
        z = z[to_keep]
        z, uni = np.unique(z, return_index=True)
        f = f[uni, ...]

        self._shape = f.shape[1:]
        self._support_points, self._support_values, self.weights = (
            self._compute_weights(z, f, **kwargs)
        )

        # only compute once
        self._poles = None
        self._residues = None
        self._roots = None

    def _input_validation(self, x, y, **kwargs):
        if x.ndim != 1:
            raise ValueError("`x` must be 1-D.")

        if not y.ndim >= 1:
            raise ValueError("`y` must be at least 1-D.")

        if x.size != y.shape[0]:
            raise ValueError("`x` be the same size as the first dimension of `y`.")

        if not np.all(np.isfinite(x)):
            raise ValueError("`x` must be finite.")

    def _compute_weights(z, f, **kwargs):
        raise NotImplementedError

    def __call__(self, z):
        """Evaluate the rational approximation at given values.

        Parameters
        ----------
        z : array_like
            Input values.
        """
        # evaluate rational function in barycentric form.
        z = np.asarray(z)
        zv = np.ravel(z)

        support_values = self._support_values.reshape(
            (self._support_values.shape[0], -1)
        )
        weights = self.weights[..., np.newaxis]

        # Cauchy matrix
        # Ignore errors due to inf/inf at support points, these will be fixed later
        with np.errstate(invalid="ignore", divide="ignore"):
            CC = 1 / np.subtract.outer(zv, self._support_points)
            # Vector of values
            r = CC @ (weights * support_values) / (CC @ weights)

        # Deal with input inf: `r(inf) = lim r(z) = sum(w*f) / sum(w)`
        if np.any(np.isinf(zv)):
            r[np.isinf(zv)] = (np.sum(weights * support_values)
                               / np.sum(weights))

        # Deal with NaN
        ii = np.nonzero(np.isnan(r))[0]
        for jj in ii:
            if np.isnan(zv[jj]) or not np.any(zv[jj] == self._support_points):
                # r(NaN) = NaN is fine.
                # The second case may happen if `r(zv[ii]) = 0/0` at some point.
                pass
            else:
                # Clean up values `NaN = inf/inf` at support points.
                # Find the corresponding node and set entry to correct value:
                r[jj] = support_values[zv[jj] == self._support_points].squeeze()

        return np.reshape(r, z.shape + self._shape)

    def poles(self):
        """Compute the poles of the rational approximation.

        Returns
        -------
        poles : array
            Poles of the AAA approximation, repeated according to their multiplicity
            but not in any specific order.
        """
        if self._poles is None:
            # Compute poles via generalized eigenvalue problem
            m = self.weights.size
            B = np.eye(m + 1, dtype=self.weights.dtype)
            B[0, 0] = 0

            E = np.zeros_like(B, dtype=np.result_type(self.weights,
                                                      self._support_points))
            E[0, 1:] = self.weights
            E[1:, 0] = 1
            np.fill_diagonal(E[1:, 1:], self._support_points)

            pol = scipy.linalg.eigvals(E, B)
            self._poles = pol[np.isfinite(pol)]
        return self._poles

    def residues(self):
        """Compute the residues of the poles of the approximation.

        Returns
        -------
        residues : array
            Residues associated with the `poles` of the approximation
        """
        if self._residues is None:
            # Compute residues via formula for res of quotient of analytic functions
            with np.errstate(divide="ignore", invalid="ignore"):
                N = (1/(np.subtract.outer(self.poles(), self._support_points))) @ (
                    self._support_values * self.weights
                )
                Ddiff = (
                    -((1/np.subtract.outer(self.poles(), self._support_points))**2)
                    @ self.weights
                )
                self._residues = N / Ddiff
        return self._residues

    def roots(self):
        """Compute the zeros of the rational approximation.

        Returns
        -------
        zeros : array
            Zeros of the AAA approximation, repeated according to their multiplicity
            but not in any specific order.
        """
        if self._roots is None:
            # Compute zeros via generalized eigenvalue problem
            m = self.weights.size
            B = np.eye(m + 1, dtype=self.weights.dtype)
            B[0, 0] = 0
            E = np.zeros_like(B, dtype=np.result_type(self.weights,
                                                      self._support_values,
                                                      self._support_points))
            E[0, 1:] = self.weights * self._support_values
            E[1:, 0] = 1
            np.fill_diagonal(E[1:, 1:], self._support_points)

            zer = scipy.linalg.eigvals(E, B)
            self._roots = zer[np.isfinite(zer)]
        return self._roots


class AAA(_BarycentricRational):
    r"""
    AAA real or complex rational approximation.

    As described in [1]_, the AAA algorithm is a greedy algorithm for approximation by
    rational functions on a real or complex set of points. The rational approximation is
    represented in a barycentric form from which the roots (zeros), poles, and residues
    can be computed.

    Parameters
    ----------
    x : 1D array_like, shape (n,)
        1-D array containing values of the independent variable. Values may be real or
        complex but must be finite.
    y : 1D array_like, shape (n,)
        Function values ``f(x)``. Infinite and NaN values of `values` and
        corresponding values of `points` will be discarded.
    rtol : float, optional
        Relative tolerance, defaults to ``eps**0.75``. If a small subset of the entries
        in `values` are much larger than the rest the default tolerance may be too
        loose. If the tolerance is too tight then the approximation may contain
        Froissart doublets or the algorithm may fail to converge entirely.
    max_terms : int, optional
        Maximum number of terms in the barycentric representation, defaults to ``100``.
        Must be greater than or equal to one.
    clean_up : bool, optional
        Automatic removal of Froissart doublets, defaults to ``True``. See notes for
        more details.
    clean_up_tol : float, optional
        Poles with residues less than this number times the geometric mean
        of `values` times the minimum distance to `points` are deemed spurious by the
        cleanup procedure, defaults to 1e-13. See notes for more details.

    Attributes
    ----------
    support_points : array
        Support points of the approximation. These are a subset of the provided `x` at
        which the approximation strictly interpolates `y`.
        See notes for more details.
    support_values : array
        Value of the approximation at the `support_points`.
    weights : array
        Weights of the barycentric approximation.
    errors : array
        Error :math:`|f(z) - r(z)|_\infty` over `points` in the successive iterations
        of AAA.

    Warns
    -----
    RuntimeWarning
        If `rtol` is not achieved in `max_terms` iterations.

    See Also
    --------
    FloaterHormannInterpolator : Floater-Hormann barycentric rational interpolation.
    pade : Padé approximation.

    Notes
    -----
    At iteration :math:`m` (at which point there are :math:`m` terms in the both the
    numerator and denominator of the approximation), the
    rational approximation in the AAA algorithm takes the barycentric form

    .. math::

        r(z) = n(z)/d(z) =
        \frac{\sum_{j=1}^m\ w_j f_j / (z - z_j)}{\sum_{j=1}^m w_j / (z - z_j)},

    where :math:`z_1,\dots,z_m` are real or complex support points selected from
    `x`, :math:`f_1,\dots,f_m` are the corresponding real or complex data values
    from `y`, and :math:`w_1,\dots,w_m` are real or complex weights.

    Each iteration of the algorithm has two parts: the greedy selection the next support
    point and the computation of the weights. The first part of each iteration is to
    select the next support point to be added :math:`z_{m+1}` from the remaining
    unselected `x`, such that the nonlinear residual
    :math:`|f(z_{m+1}) - n(z_{m+1})/d(z_{m+1})|` is maximised. The algorithm terminates
    when this maximum is less than ``rtol * np.linalg.norm(f, ord=np.inf)``. This means
    the interpolation property is only satisfied up to a tolerance, except at the
    support points where approximation exactly interpolates the supplied data.

    In the second part of each iteration, the weights :math:`w_j` are selected to solve
    the least-squares problem

    .. math::

        \text{minimise}_{w_j}|fd - n| \quad \text{subject to} \quad
        \sum_{j=1}^{m+1} w_j = 1,

    over the unselected elements of `x`.

    One of the challenges with working with rational approximations is the presence of
    Froissart doublets, which are either poles with vanishingly small residues or
    pole-zero pairs that are close enough together to nearly cancel, see [2]_. The
    greedy nature of the AAA algorithm means Froissart doublets are rare. However, if
    `rtol` is set too tight then the approximation will stagnate and many Froissart
    doublets will appear. Froissart doublets can usually be removed by removing support
    points and then resolving the least squares problem. The support point :math:`z_j`,
    which is the closest support point to the pole :math:`a` with residue
    :math:`\alpha`, is removed if the following is satisfied

    .. math::

        |\alpha| / |z_j - a| < \verb|clean_up_tol| \cdot \tilde{f},

    where :math:`\tilde{f}` is the geometric mean of `support_values`.


    References
    ----------
    .. [1] Y. Nakatsukasa, O. Sete, and L. N. Trefethen, "The AAA algorithm for
            rational approximation", SIAM J. Sci. Comp. 40 (2018), A1494-A1522.
            :doi:`10.1137/16M1106122`
    .. [2] J. Gilewicz and M. Pindor, Pade approximants and noise: rational functions,
           J. Comp. Appl. Math. 105 (1999), pp. 285-297.
           :doi:`10.1016/S0377-0427(02)00674-X`

    Examples
    --------

    Here we reproduce a number of the numerical examples from [1]_ as a demonstration
    of the functionality offered by this method.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import AAA
    >>> import warnings

    For the first example we approximate the gamma function on ``[-3.5, 4.5]`` by
    extrapolating from 100 samples in ``[-1.5, 1.5]``.

    >>> from scipy.special import gamma
    >>> sample_points = np.linspace(-1.5, 1.5, num=100)
    >>> r = AAA(sample_points, gamma(sample_points))
    >>> z = np.linspace(-3.5, 4.5, num=1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(z, gamma(z), label="Gamma")
    >>> ax.plot(sample_points, gamma(sample_points), label="Sample points")
    >>> ax.plot(z, r(z).real, '--', label="AAA approximation")
    >>> ax.set(xlabel="z", ylabel="r(z)", ylim=[-8, 8], xlim=[-3.5, 4.5])
    >>> ax.legend()
    >>> plt.show()

    We can also view the poles of the rational approximation and their residues:

    >>> order = np.argsort(r.poles())
    >>> r.poles()[order]
    array([-3.81591039e+00+0.j        , -3.00269049e+00+0.j        ,
           -1.99999988e+00+0.j        , -1.00000000e+00+0.j        ,
            5.85842812e-17+0.j        ,  4.77485458e+00-3.06919376j,
            4.77485458e+00+3.06919376j,  5.29095868e+00-0.97373072j,
            5.29095868e+00+0.97373072j])
    >>> r.residues()[order]
    array([ 0.03658074 +0.j        , -0.16915426 -0.j        ,
            0.49999915 +0.j        , -1.         +0.j        ,
            1.         +0.j        , -0.81132013 -2.30193429j,
           -0.81132013 +2.30193429j,  0.87326839+10.70148546j,
            0.87326839-10.70148546j])

    For the second example, we call `AAA` with a spiral of 1000 points that wind 7.5
    times around the origin in the complex plane.

    >>> z = np.exp(np.linspace(-0.5, 0.5 + 15j*np.pi, 1000))
    >>> r = AAA(z, np.tan(np.pi*z/2), rtol=1e-13)

    We see that AAA takes 12 steps to converge with the following errors:

    >>> r.errors.size
    12
    >>> r.errors
    array([2.49261500e+01, 4.28045609e+01, 1.71346935e+01, 8.65055336e-02,
           1.27106444e-02, 9.90889874e-04, 5.86910543e-05, 1.28735561e-06,
           3.57007424e-08, 6.37007837e-10, 1.67103357e-11, 1.17112299e-13])

    We can also plot the computed poles:

    >>> fig, ax = plt.subplots()
    >>> ax.plot(z.real, z.imag, '.', markersize=2, label="Sample points")
    >>> ax.plot(r.poles().real, r.poles().imag, '.', markersize=5,
    ...         label="Computed poles")
    >>> ax.set(xlim=[-3.5, 3.5], ylim=[-3.5, 3.5], aspect="equal")
    >>> ax.legend()
    >>> plt.show()

    We now demonstrate the removal of Froissart doublets using the `clean_up` method
    using an example from [1]_. Here we approximate the function
    :math:`f(z)=\log(2 + z^4)/(1 + 16z^4)` by sampling it at 1000 roots of unity. The
    algorithm is run with ``rtol=0`` and ``clean_up=False`` to deliberately cause
    Froissart doublets to appear.

    >>> z = np.exp(1j*2*np.pi*np.linspace(0,1, num=1000))
    >>> def f(z):
    ...     return np.log(2 + z**4)/(1 - 16*z**4)
    >>> with warnings.catch_warnings():  # filter convergence warning due to rtol=0
    ...     warnings.simplefilter('ignore', RuntimeWarning)
    ...     r = AAA(z, f(z), rtol=0, max_terms=50, clean_up=False)
    >>> mask = np.abs(r.residues()) < 1e-13
    >>> fig, axs = plt.subplots(ncols=2)
    >>> axs[0].plot(r.poles().real[~mask], r.poles().imag[~mask], '.')
    >>> axs[0].plot(r.poles().real[mask], r.poles().imag[mask], 'r.')

    Now we call the `clean_up` method to remove Froissart doublets.

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', RuntimeWarning)
    ...     r.clean_up()
    4
    >>> mask = np.abs(r.residues()) < 1e-13
    >>> axs[1].plot(r.poles().real[~mask], r.poles().imag[~mask], '.')
    >>> axs[1].plot(r.poles().real[mask], r.poles().imag[mask], 'r.')
    >>> plt.show()

    The left image shows the poles prior of the approximation ``clean_up=False`` with
    poles with residue less than ``10^-13`` in absolute value shown in red. The right
    image then shows the poles after the `clean_up` method has been called.
    """
    def __init__(self, x, y, *, rtol=None, max_terms=100, clean_up=True,
                 clean_up_tol=1e-13):
        super().__init__(x, y, rtol=rtol, max_terms=max_terms)

        if clean_up:
            self.clean_up(clean_up_tol)

    def _input_validation(self, x, y, rtol=None, max_terms=100, clean_up=True,
                          clean_up_tol=1e-13):
        max_terms = operator.index(max_terms)
        if max_terms < 1:
            raise ValueError("`max_terms` must be an integer value greater than or "
                             "equal to one.")

        if y.ndim != 1:
            raise ValueError("`y` must be 1-D.")

        super()._input_validation(x, y)

    @property
    def support_points(self):
        return self._support_points

    @property
    def support_values(self):
        return self._support_values

    def _compute_weights(self, z, f, rtol, max_terms):
        # Initialization for AAA iteration
        M = np.size(z)
        mask = np.ones(M, dtype=np.bool_)
        dtype = np.result_type(z, f, 1.0)
        rtol = np.finfo(dtype).eps**0.75 if rtol is None else rtol
        atol = rtol * np.linalg.norm(f, ord=np.inf)
        zj = np.empty(max_terms, dtype=dtype)
        fj = np.empty(max_terms, dtype=dtype)
        # Cauchy matrix
        C = np.empty((M, max_terms), dtype=dtype)
        # Loewner matrix
        A = np.empty((M, max_terms), dtype=dtype)
        errors = np.empty(max_terms, dtype=A.real.dtype)
        R = np.repeat(np.mean(f), M)

        # AAA iteration
        for m in range(max_terms):
            # Introduce next support point
            # Select next support point
            jj = np.argmax(np.abs(f[mask] - R[mask]))
            # Update support points
            zj[m] = z[mask][jj]
            # Update data values
            fj[m] = f[mask][jj]
            # Next column of Cauchy matrix
            # Ignore errors as we manually interpolate at support points
            with np.errstate(divide="ignore", invalid="ignore"):
                C[:, m] = 1 / (z - z[mask][jj])
            # Update mask
            mask[np.nonzero(mask)[0][jj]] = False
            # Update Loewner matrix
            # Ignore errors as inf values will be masked out in SVD call
            with np.errstate(invalid="ignore"):
                A[:, m] = (f - fj[m]) * C[:, m]

            # Compute weights
            rows = mask.sum()
            if rows >= m + 1:
                # The usual tall-skinny case
                _, s, V = scipy.linalg.svd(
                    A[mask, : m + 1], full_matrices=False, check_finite=False,
                )
                # Treat case of multiple min singular values
                mm = s == np.min(s)
                # Aim for non-sparse weight vector
                wj = (V.conj()[mm, :].sum(axis=0) / np.sqrt(mm.sum())).astype(dtype)
            else:
                # Fewer rows than columns
                V = scipy.linalg.null_space(A[mask, : m + 1], check_finite=False)
                nm = V.shape[-1]
                # Aim for non-sparse wt vector
                wj = V.sum(axis=-1) / np.sqrt(nm)

            # Compute rational approximant
            # Omit columns with `wj == 0`
            i0 = wj != 0
            # Ignore errors as we manually interpolate at support points
            with np.errstate(invalid="ignore"):
                # Numerator
                N = C[:, : m + 1][:, i0] @ (wj[i0] * fj[: m + 1][i0])
                # Denominator
                D = C[:, : m + 1][:, i0] @ wj[i0]
            # Interpolate at support points with `wj !=0`
            D_inf = np.isinf(D) | np.isnan(D)
            D[D_inf] = 1
            N[D_inf] = f[D_inf]
            R = N / D

            # Check if converged
            max_error = np.linalg.norm(f - R, ord=np.inf)
            errors[m] = max_error
            if max_error <= atol:
                break

        if m == max_terms - 1:
            warnings.warn(f"AAA failed to converge within {max_terms} iterations.",
                          RuntimeWarning, stacklevel=2)

        # Trim off unused array allocation
        zj = zj[: m + 1]
        fj = fj[: m + 1]

        # Remove support points with zero weight
        i_non_zero = wj != 0
        self.errors = errors[: m + 1]
        self._points = z
        self._values = f
        return zj[i_non_zero], fj[i_non_zero], wj[i_non_zero]

    def clean_up(self, cleanup_tol=1e-13):
        """Automatic removal of Froissart doublets.

        Parameters
        ----------
        cleanup_tol : float, optional
            Poles with residues less than this number times the geometric mean
            of `values` times the minimum distance to `points` are deemed spurious by
            the cleanup procedure, defaults to 1e-13.

        Returns
        -------
        int
            Number of Froissart doublets detected
        """
        # Find negligible residues
        geom_mean_abs_f = scipy.stats.gmean(np.abs(self._values))

        Z_distances = np.min(
            np.abs(np.subtract.outer(self.poles(), self._points)), axis=1
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ii = np.nonzero(
                np.abs(self.residues()) / Z_distances < cleanup_tol * geom_mean_abs_f
            )

        ni = ii[0].size
        if ni == 0:
            return ni

        warnings.warn(f"{ni} Froissart doublets detected.", RuntimeWarning,
                        stacklevel=2)

        # For each spurious pole find and remove closest support point
        closest_spt_point = np.argmin(
            np.abs(np.subtract.outer(self._support_points, self.poles()[ii])), axis=0
        )
        self._support_points = np.delete(self._support_points, closest_spt_point)
        self._support_values = np.delete(self._support_values, closest_spt_point)

        # Remove support points z from sample set
        mask = np.logical_and.reduce(
            np.not_equal.outer(self._points, self._support_points), axis=1
        )
        f = self._values[mask]
        z = self._points[mask]

        # recompute weights, we resolve the least squares problem for the remaining
        # support points

        m = self._support_points.size

        # Cauchy matrix
        C = 1 / np.subtract.outer(z, self._support_points)
        # Loewner matrix
        A = f[:, np.newaxis] * C - C * self._support_values

        # Solve least-squares problem to obtain weights
        _, _, V = scipy.linalg.svd(A, check_finite=False)
        self.weights = np.conj(V[m - 1,:])

        # reset roots, poles, residues as cached values will be wrong with new weights
        self._poles = None
        self._residues = None
        self._roots = None

        return ni


class FloaterHormannInterpolator(_BarycentricRational):
    r"""
    Floater-Hormann barycentric rational interpolation.

    As described in [1]_, the method of Floater and Hormann computes weights for a
    Barycentric rational interpolant with no poles on the real axis.

    Parameters
    ----------
    x : 1D array_like, shape (n,)
        1-D array containing values of the independent variable. Values may be real or
        complex but must be finite.
    y : array_like, shape (n, ...)
        Array containing values of the dependent variable. Infinite and NaN values
        of `values` and corresponding values of `x` will be discarded.
    d : int, optional
        Blends ``n - d`` degree `d` polynomials together. For ``d = n - 1`` it is
        equivalent to polynomial interpolation. Must satisfy ``0 <= d < n``,
        defaults to 3.

    Attributes
    ----------
    weights : array
        Weights of the barycentric approximation.

    See Also
    --------
    AAA : Barycentric rational approximation of real and complex functions.
    pade : Padé approximation.

    Notes
    -----
    The Floater-Hormann interpolant is a rational function that interpolates the data
    with approximation order :math:`O(h^{d+1})`. The rational function blends ``n - d``
    polynomials of degree `d` together to produce a rational interpolant that contains
    no poles on the real axis, unlike `AAA`. The interpolant is given
    by

    .. math::

        r(x) = \frac{\sum_{i=0}^{n-d} \lambda_i(x) p_i(x)}
        {\sum_{i=0}^{n-d} \lambda_i(x)},

    where :math:`p_i(x)` is an interpolating polynomials of at most degree `d` through
    the points :math:`(x_i,y_i),\dots,(x_{i+d},y_{i+d}), and :math:`\lambda_i(z)` are
    blending functions defined by

    .. math::

        \lambda_i(x) = \frac{(-1)^i}{(x - x_i)\cdots(x - x_{i+d})}.

    When ``d = n - 1`` this reduces to polynomial interpolation.

    Due to its stability following barycentric representation of the above equation
    is used instead for computation

    .. math::

        r(z) = \frac{\sum_{k=1}^m\ w_k f_k / (x - x_k)}{\sum_{k=1}^m w_k / (x - x_k)},

    where the weights :math:`w_j` are computed as

    .. math::

        w_k &= (-1)^{k - d} \sum_{i \in J_k} \prod_{j = i, j \neq k}^{i + d}
        1/|x_k - x_j|, \\
        J_k &= \{ i \in I: k - d \leq i \leq k\},\\
        I &= \{0, 1, \dots, n - d\}.

    References
    ----------
    .. [1] M.S. Floater and K. Hormann, "Barycentric rational interpolation with no
           poles and high rates of approximation", Numer. Math. 107, 315 (2007).
           :doi:`10.1007/s00211-007-0093-y`

    Examples
    --------

    Here we compare the method against polynomial interpolation for an example where
    the polynomial interpolation fails due to Runge's phenomenon.

    >>> import numpy as np
    >>> from scipy.interpolate import (FloaterHormannInterpolator,
    ...                                BarycentricInterpolator)
    >>> def f(z):
    ...     return 1/(1 + z**2)
    >>> z = np.linspace(-5, 5, num=15)
    >>> r = FloaterHormannInterpolator(z, f(z))
    >>> p = BarycentricInterpolator(z, f(z))
    >>> zz = np.linspace(-5, 5, num=1000)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot(zz, r(zz), label="Floater=Hormann")
    >>> ax.plot(zz, p(zz), label="Polynomial")
    >>> ax.legend()
    >>> plt.show()
    """
    def __init__(self, points, values, *, d=3):
        super().__init__(points, values, d=d)

    def _input_validation(self, x, y, d):
        d = operator.index(d)
        if not (0 <= d < len(x)):
            raise ValueError("`d` must satisfy 0 <= d < n")

        super()._input_validation(x, y)

    def _compute_weights(self, z, f, d):
        # Floater and Hormann 2007 Eqn. (18) 3 equations later
        w = np.zeros_like(z, dtype=np.result_type(z, 1.0))
        n = w.size
        for k in range(n):
            for i in range(max(k-d, 0), min(k+1, n-d)):
                w[k] += 1/np.prod(np.abs(np.delete(z[k] - z[i : i + d + 1], k - i)))
        w *= (-1.)**(np.arange(n) - d)

        return z, f, w
<|endoftext|># Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections.abc import Iterable
import numbers
import warnings
import numpy as np
import operator

from scipy._lib._util import normalize_axis_index
from . import _ni_support
from . import _nd_image
from . import _ni_docstrings
from . import _rank_filter_1d

__all__ = ['correlate1d', 'convolve1d', 'gaussian_filter1d', 'gaussian_filter',
           'prewitt', 'sobel', 'generic_laplace', 'laplace',
           'gaussian_laplace', 'generic_gradient_magnitude',
           'gaussian_gradient_magnitude', 'correlate', 'convolve',
           'uniform_filter1d', 'uniform_filter', 'minimum_filter1d',
           'maximum_filter1d', 'minimum_filter', 'maximum_filter',
           'rank_filter', 'median_filter', 'percentile_filter',
           'generic_filter1d', 'generic_filter']


def _invalid_origin(origin, lenw):
    return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)


def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input and complex_weights:
        # real component of the output
        func(input.real, weights.real, output=output.real,
             cval=np.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None,
                            cval=np.imag(cval), **kwargs)
        # imaginary component of the output
        func(input.real, weights.imag, output=output.imag,
             cval=np.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None,
                            cval=np.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=np.real(cval),
             **kwargs)
        func(input.imag, weights, output=output.imag, cval=np.imag(cval),
             **kwargs)
    else:
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the "
                             "input is real.")
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    return output


def _expand_origin(ndim_image, axes, origin):
    num_axes = len(axes)
    origins = _ni_support._normalize_sequence(origin, num_axes)
    if num_axes < ndim_image:
        # set origin = 0 for any axes not being filtered
        origins_temp = [0,] * ndim_image
        for o, ax in zip(origins, axes):
            origins_temp[ax] = o
        origins = origins_temp
    return origins


def _expand_footprint(ndim_image, axes, footprint,
                      footprint_name="footprint"):
    num_axes = len(axes)
    if num_axes < ndim_image:
        if footprint.ndim != num_axes:
            raise RuntimeError(f"{footprint_name}.ndim ({footprint.ndim}) "
                               f"must match len(axes) ({num_axes})")

        footprint = np.expand_dims(
            footprint,
            tuple(ax for ax in range(ndim_image) if ax not in axes)
        )
    return footprint


def _expand_mode(ndim_image, axes, mode):
    num_axes = len(axes)
    if not isinstance(mode, str) and isinstance(mode, Iterable):
        # set mode = 'constant' for any axes not being filtered
        modes = _ni_support._normalize_sequence(mode, num_axes)
        modes_temp = ['constant'] * ndim_image
        for m, ax in zip(modes, axes):
            modes_temp[ax] = m
        mode = modes_temp
    return mode


@_ni_docstrings.docfiller
def correlate1d(input, weights, axis=-1, output=None, mode="reflect",
                cval=0.0, origin=0):
    """Calculate a 1-D correlation along the given axis.

    The lines of the array along the given axis are correlated with the
    given weights.

    Parameters
    ----------
    %(input)s
    weights : array
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Returns
    -------
    result : ndarray
        Correlation result. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
    input = np.asarray(input)
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input or complex_weights:
        if complex_weights:
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        kwargs = dict(axis=axis, mode=mode, origin=origin)
        output = _ni_support._get_output(output, input, complex_output=True)
        return _complex_via_real_components(correlate1d, input, weights,
                                            output, cval, **kwargs)

    output = _ni_support._get_output(output, input)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = normalize_axis_index(axis, input.ndim)
    if _invalid_origin(origin, len(weights)):
        raise ValueError('Invalid origin; origin must satisfy '
                         '-(len(weights) // 2) <= origin <= '
                         '(len(weights)-1) // 2')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate1d(input, weights, axis, output, mode, cval,
                          origin)
    return output


@_ni_docstrings.docfiller
def convolve1d(input, weights, axis=-1, output=None, mode="reflect",
               cval=0.0, origin=0):
    """Calculate a 1-D convolution along the given axis.

    The lines of the array along the given axis are convolved with the
    given weights.

    Parameters
    ----------
    %(input)s
    weights : ndarray
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Returns
    -------
    convolve1d : ndarray
        Convolved array with same shape as input

    Examples
    --------
    >>> from scipy.ndimage import convolve1d
    >>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([14, 24,  4, 13, 12, 36, 27,  0])
    """
    weights = np.asarray(weights)
    weights = weights[::-1]
    origin = -origin
    if not weights.shape[0] & 1:
        origin -= 1
    if weights.dtype.kind == 'c':
        # pre-conjugate here to counteract the conjugation in correlate1d
        weights = weights.conj()
    return correlate1d(input, weights, axis, output, mode, cval, origin)


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


@_ni_docstrings.docfiller
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """1-D Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode_reflect)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int, optional
        Radius of the Gaussian kernel. If specified, the size of
        the kernel will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.

    Returns
    -------
    gaussian_filter1d : ndarray

    Notes
    -----
    The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
    `radius` is None, a default ``radius = round(truncate * sigma)`` will be
    used.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter1d
    >>> import numpy as np
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(101).cumsum()
    >>> y3 = gaussian_filter1d(x, 3)
    >>> y6 = gaussian_filter1d(x, 6)
    >>> plt.plot(x, 'k', label='original data')
    >>> plt.plot(y3, '--', label='filtered, sigma=3')
    >>> plt.plot(y6, ':', label='filtered, sigma=6')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.show()

    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)


@_ni_docstrings.docfiller
def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0, *, radius=None,
                    axes=None):
    """Multidimensional Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : int or sequence of ints, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. A positive order
        corresponds to convolution with that derivative of a Gaussian.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int or sequence of ints, optional
        Radius of the Gaussian kernel. The radius are given for each axis
        as a sequence, or as a single number, in which case it is equal
        for all axes. If specified, the size of the kernel along each axis
        will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `sigma`, `order`, `mode` and/or `radius`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    gaussian_filter : ndarray
        Returned array of same shape as `input`.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.

    The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
    `radius` is None, the default ``radius = round(truncate * sigma)`` will be
    used.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter
    >>> import numpy as np
    >>> a = np.arange(50, step=2).reshape((5,5))
    >>> a
    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28],
           [30, 32, 34, 36, 38],
           [40, 42, 44, 46, 48]])
    >>> gaussian_filter(a, sigma=1)
    array([[ 4,  6,  8,  9, 11],
           [10, 12, 14, 15, 17],
           [20, 22, 24, 25, 27],
           [29, 31, 33, 34, 36],
           [35, 37, 39, 40, 42]])

    >>> from scipy import datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = gaussian_filter(ascent, sigma=5)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)

    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    radiuses = _ni_support._normalize_sequence(radius, num_axes)
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
            for ii in range(num_axes) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode, radius in axes:
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate, radius=radius)
            input = output
    else:
        output[...] = input[...]
    return output


@_ni_docstrings.docfiller
def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Prewitt filter.

    Parameters
    ----------
    %(input)s
    %(axis)s
    %(output)s
    %(mode_multiple)s
    %(cval)s

    Returns
    -------
    prewitt : ndarray
        Filtered array. Has the same shape as `input`.

    See Also
    --------
    sobel: Sobel filter

    Notes
    -----
    This function computes the one-dimensional Prewitt filter.
    Horizontal edges are emphasised with the horizontal transform (axis=0),
    vertical edges with the vertical transform (axis=1), and so on for higher
    dimensions. These can be combined to give the magnitude.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> ascent = datasets.ascent()
    >>> prewitt_h = ndimage.prewitt(ascent, axis=0)
    >>> prewitt_v = ndimage.prewitt(ascent, axis=1)
    >>> magnitude = np.sqrt(prewitt_h ** 2 + prewitt_v ** 2)
    >>> magnitude *= 255 / np.max(magnitude) # Normalization
    >>> fig, axes = plt.subplots(2, 2, figsize = (8, 8))
    >>> plt.gray()
    >>> axes[0, 0].imshow(ascent)
    >>> axes[0, 1].imshow(prewitt_h)
    >>> axes[1, 0].imshow(prewitt_v)
    >>> axes[1, 1].imshow(magnitude)
    >>> titles = ["original", "horizontal", "vertical", "magnitude"]
    >>> for i, ax in enumerate(axes.ravel()):
    ...     ax.set_title(titles[i])
    ...     ax.axis("off")
    >>> plt.show()

    """
    input = np.asarray(input)
    axis = normalize_axis_index(axis, input.ndim)
    output = _ni_support._get_output(output, input)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    correlate1d(input, [-1, 0, 1], axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        correlate1d(output, [1, 1, 1], ii, output, modes[ii], cval, 0,)
    return output


@_ni_docstrings.docfiller
def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Sobel filter.

    Parameters
    ----------
    %(input)s
    %(axis)s
    %(output)s
    %(mode_multiple)s
    %(cval)s

    Returns
    -------
    sobel : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    This function computes the axis-specific Sobel gradient.
    The horizontal edges can be emphasised with the horizontal transform (axis=0),
    the vertical edges with the vertical transform (axis=1) and so on for higher
    dimensions. These can be combined to give the magnitude.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> ascent = datasets.ascent().astype('int32')
    >>> sobel_h = ndimage.sobel(ascent, 0)  # horizontal gradient
    >>> sobel_v = ndimage.sobel(ascent, 1)  # vertical gradient
    >>> magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    >>> magnitude *= 255.0 / np.max(magnitude)  # normalization
    >>> fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    >>> plt.gray()  # show the filtered result in grayscale
    >>> axs[0, 0].imshow(ascent)
    >>> axs[0, 1].imshow(sobel_h)
    >>> axs[1, 0].imshow(sobel_v)
    >>> axs[1, 1].imshow(magnitude)
    >>> titles = ["original", "horizontal", "vertical", "magnitude"]
    >>> for i, ax in enumerate(axs.ravel()):
    ...     ax.set_title(titles[i])
    ...     ax.axis("off")
    >>> plt.show()

    """
    input = np.asarray(input)
    axis = normalize_axis_index(axis, input.ndim)
    output = _ni_support._get_output(output, input)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    correlate1d(input, [-1, 0, 1], axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        correlate1d(output, [1, 2, 1], ii, output, modes[ii], cval, 0)
    return output


@_ni_docstrings.docfiller
def generic_laplace(input, derivative2, output=None, mode="reflect",
                    cval=0.0,
                    extra_arguments=(),
                    extra_keywords=None,
                    *, axes=None):
    """
    N-D Laplace filter using a provided second derivative function.

    Parameters
    ----------
    %(input)s
    derivative2 : callable
        Callable with the following signature::

            derivative2(input, axis, output, mode, cval,
                        *extra_arguments, **extra_keywords)

        See `extra_arguments`, `extra_keywords` below.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(extra_keywords)s
    %(extra_arguments)s
    axes : tuple of int or None
        The axes over which to apply the filter. If a `mode` tuple is
        provided, its length must match the number of axes.

    Returns
    -------
    generic_laplace : ndarray
        Filtered array. Has the same shape as `input`.

    """
    if extra_keywords is None:
        extra_keywords = {}
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)
    axes = _ni_support._check_axes(axes, input.ndim)
    if len(axes) > 0:
        modes = _ni_support._normalize_sequence(mode, len(axes))
        derivative2(input, axes[0], output, modes[0], cval,
                    *extra_arguments, **extra_keywords)
        for ii in range(1, len(axes)):
            tmp = derivative2(input, axes[ii], output.dtype, modes[ii], cval,
                              *extra_arguments, **extra_keywords)
            output += tmp
    else:
        output[...] = input[...]
    return output


@_ni_docstrings.docfiller
def laplace(input, output=None, mode="reflect", cval=0.0, *, axes=None):
    """N-D Laplace filter based on approximate second derivatives.

    Parameters
    ----------
    %(input)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    axes : tuple of int or None
        The axes over which to apply the filter. If a `mode` tuple is
        provided, its length must match the number of axes.

    Returns
    -------
    laplace : ndarray
        Filtered array. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.laplace(ascent)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)
    return generic_laplace(input, derivative2, output, mode, cval, axes=axes)


@_ni_docstrings.docfiller
def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, *, axes=None, **kwargs):
    """Multidimensional Laplace filter using Gaussian second derivatives.

    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    axes : tuple of int or None
        The axes over which to apply the filter. If `sigma` or `mode` tuples
        are provided, their length must match the number of axes.
    Extra keyword arguments will be passed to gaussian_filter().

    Returns
    -------
    gaussian_laplace : ndarray
        Filtered array. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> ascent = datasets.ascent()

    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side

    >>> result = ndimage.gaussian_laplace(ascent, sigma=1)
    >>> ax1.imshow(result)

    >>> result = ndimage.gaussian_laplace(ascent, sigma=3)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)

    def derivative2(input, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)

    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    sigma = _ni_support._normalize_sequence(sigma, num_axes)
    if num_axes < input.ndim:
        # set sigma = 0 for any axes not being filtered
        sigma_temp = [0,] * input.ndim
        for s, ax in zip(sigma, axes):
            sigma_temp[ax] = s
        sigma = sigma_temp

    return generic_laplace(input, derivative2, output, mode, cval,
                           extra_arguments=(sigma,),
                           extra_keywords=kwargs,
                           axes=axes)


@_ni_docstrings.docfiller
def generic_gradient_magnitude(input, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None,
                               *, axes=None):
    """Gradient magnitude using a provided gradient function.

    Parameters
    ----------
    %(input)s
    derivative : callable
        Callable with the following signature::

            derivative(input, axis, output, mode, cval,
                       *extra_arguments, **extra_keywords)

        See `extra_arguments`, `extra_keywords` below.
        `derivative` can assume that `input` and `output` are ndarrays.
        Note that the output from `derivative` is modified inplace;
        be careful to copy important inputs before returning them.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(extra_keywords)s
    %(extra_arguments)s
    axes : tuple of int or None
        The axes over which to apply the filter. If a `mode` tuple is
        provided, its length must match the number of axes.

    Returns
    -------
    generic_gradient_matnitude : ndarray
        Filtered array. Has the same shape as `input`.

    """
    if extra_keywords is None:
        extra_keywords = {}
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)
    axes = _ni_support._check_axes(axes, input.ndim)
    if len(axes) > 0:
        modes = _ni_support._normalize_sequence(mode, len(axes))
        derivative(input, axes[0], output, modes[0], cval,
                   *extra_arguments, **extra_keywords)
        np.multiply(output, output, output)
        for ii in range(1, len(axes)):
            tmp = derivative(input, axes[ii], output.dtype, modes[ii], cval,
                             *extra_arguments, **extra_keywords)
            np.multiply(tmp, tmp, tmp)
            output += tmp
        # This allows the sqrt to work with a different default casting
        np.sqrt(output, output, casting='unsafe')
    else:
        output[...] = input[...]
    return output


@_ni_docstrings.docfiller
def gaussian_gradient_magnitude(input, sigma, output=None,
                                mode="reflect", cval=0.0, *, axes=None,
                                **kwargs):
    """Multidimensional gradient magnitude using Gaussian derivatives.

    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    axes : tuple of int or None
        The axes over which to apply the filter. If `sigma` or `mode` tuples
        are provided, their length must match the number of axes.
    Extra keyword arguments will be passed to gaussian_filter().

    Returns
    -------
    gaussian_gradient_magnitude : ndarray
        Filtered array. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.gaussian_gradient_magnitude(ascent, sigma=5)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)

    def derivative(input, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * input.ndim
        order[axis] = 1
        return gaussian_filter(input, sigma, order, output, mode,
                               cval, **kwargs)

    return generic_gradient_magnitude(input, derivative, output, mode,
                                      cval, extra_arguments=(sigma,),
                                      extra_keywords=kwargs, axes=axes)


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution, axes):
    input = np.asarray(input)
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input or complex_weights:
        if complex_weights and not convolution:
            # As for np.correlate, conjugate weights rather than input.
            weights = weights.conj()
        kwargs = dict(
            mode=mode, origin=origin, convolution=convolution, axes=axes
        )
        output = _ni_support._get_output(output, input, complex_output=True)

        return _complex_via_real_components(_correlate_or_convolve, input,
                                            weights, output, cval, **kwargs)

    axes = _ni_support._check_axes(axes, input.ndim)
    weights = np.asarray(weights, dtype=np.float64)

    # expand weights and origins if num_axes < input.ndim
    weights = _expand_footprint(input.ndim, axes, weights, "weights")
    origins = _expand_origin(input.ndim, axes, origin)

    wshape = [ii for ii in weights.shape if ii > 0]
    if len(wshape) != input.ndim:
        raise RuntimeError(f"weights.ndim ({len(wshape)}) must match "
                           f"len(axes) ({len(axes)})")
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        for ii in range(len(origins)):
            origins[ii] = -origins[ii]
            if not weights.shape[ii] & 1:
                origins[ii] -= 1
    for origin, lenw in zip(origins, wshape):
        if _invalid_origin(origin, lenw):
            raise ValueError('Invalid origin; origin must satisfy '
                             '-(weights.shape[k] // 2) <= origin[k] <= '
                             '(weights.shape[k]-1) // 2')

    if not weights.flags.contiguous:
        weights = weights.copy()
    output = _ni_support._get_output(output, input)
    temp_needed = np.may_share_memory(input, output)
    if temp_needed:
        # input and output arrays cannot share memory
        temp = output
        output = _ni_support._get_output(output.dtype, input)
    if not isinstance(mode, str) and isinstance(mode, Iterable):
        raise RuntimeError("A sequence of modes is not supported")
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate(input, weights, output, mode, cval, origins)
    if temp_needed:
        temp[...] = output
        output = temp
    return output


@_ni_docstrings.docfiller
def correlate(input, weights, output=None, mode='reflect', cval=0.0,
              origin=0, *, axes=None):
    """
    Multidimensional correlation.

    The array is correlated with the given kernel.

    Parameters
    ----------
    %(input)s
    weights : ndarray
        array of weights, same number of dimensions as input
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin_multiple)s
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `mode` or `origin` must match the length
        of `axes`. The ith entry in any of these tuples corresponds to the ith
        entry in `axes`.

    Returns
    -------
    result : ndarray
        The result of correlation of `input` with `weights`.

    See Also
    --------
    convolve : Convolve an image with a kernel.

    Examples
    --------
    Correlation is the process of moving a filter mask often referred to
    as kernel over the image and computing the sum of products at each location.

    >>> from scipy.ndimage import correlate
    >>> import numpy as np
    >>> input_img = np.arange(25).reshape(5,5)
    >>> print(input_img)
    [[ 0  1  2  3  4]
    [ 5  6  7  8  9]
    [10 11 12 13 14]
    [15 16 17 18 19]
    [20 21 22 23 24]]

    Define a kernel (weights) for correlation. In this example, it is for sum of
    center and up, down, left and right next elements.

    >>> weights = [[0, 1, 0],
    ...            [1, 1, 1],
    ...            [0, 1, 0]]

    We can calculate a correlation result:
    For example, element ``[2,2]`` is ``7 + 11 + 12 + 13 + 17 = 60``.

    >>> correlate(input_img, weights)
    array([[  6,  10,  15,  20,  24],
        [ 26,  30,  35,  40,  44],
        [ 51,  55,  60,  65,  69],
        [ 76,  80,  85,  90,  94],
        [ 96, 100, 105, 110, 114]])

    """
    return _correlate_or_convolve(input, weights, output, mode, cval,
                                  origin, False, axes)


@_ni_docstrings.docfiller
def convolve(input, weights, output=None, mode='reflect', cval=0.0,
             origin=0, *, axes=None):
    """
    Multidimensional convolution.

    The array is convolved with the given kernel.

    Parameters
    ----------
    %(input)s
    weights : array_like
        Array of weights, same number of dimensions as input
    %(output)s
    %(mode_reflect)s
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0
    origin : int or sequence, optional
        Controls the placement of the filter on the input array's pixels.
        A value of 0 (the default) centers the filter over the pixel, with
        positive values shifting the filter to the right, and negative ones
        to the left. By passing a sequence of origins with length equal to
        the number of dimensions of the input array, different shifts can
        be specified along each axis.
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `mode` or `origin` must match the length
        of `axes`. The ith entry in any of these tuples corresponds to the ith
        entry in `axes`.

    Returns
    -------
    result : ndarray
        The result of convolution of `input` with `weights`.

    See Also
    --------
    correlate : Correlate an image with a kernel.

    Notes
    -----
    Each value in result is :math:`C_i = \\sum_j{I_{i+k-j} W_j}`, where
    W is the `weights` kernel,
    j is the N-D spatial index over :math:`W`,
    I is the `input` and k is the coordinate of the center of
    W, specified by `origin` in the input parameters.

    Examples
    --------
    Perhaps the simplest case to understand is ``mode='constant', cval=0.0``,
    because in this case borders (i.e., where the `weights` kernel, centered
    on any one value, extends beyond an edge of `input`) are treated as zeros.

    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> k = np.array([[1,1,1],[1,1,0],[1,0,0]])
    >>> from scipy import ndimage
    >>> ndimage.convolve(a, k, mode='constant', cval=0.0)
    array([[11, 10,  7,  4],
           [10,  3, 11, 11],
           [15, 12, 14,  7],
           [12,  3,  7,  0]])

    Setting ``cval=1.0`` is equivalent to padding the outer edge of `input`
    with 1.0's (and then extracting only the original region of the result).

    >>> ndimage.convolve(a, k, mode='constant', cval=1.0)
    array([[13, 11,  8,  7],
           [11,  3, 11, 14],
           [16, 12, 14, 10],
           [15,  6, 10,  5]])

    With ``mode='reflect'`` (the default), outer values are reflected at the
    edge of `input` to fill in missing values.

    >>> b = np.array([[2, 0, 0],
    ...               [1, 0, 0],
    ...               [0, 0, 0]])
    >>> k = np.array([[0,1,0], [0,1,0], [0,1,0]])
    >>> ndimage.convolve(b, k, mode='reflect')
    array([[5, 0, 0],
           [3, 0, 0],
           [1, 0, 0]])

    This includes diagonally at the corners.

    >>> k = np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> ndimage.convolve(b, k)
    array([[4, 2, 0],
           [3, 2, 0],
           [1, 1, 0]])

    With ``mode='nearest'``, the single nearest value in to an edge in
    `input` is repeated as many times as needed to match the overlapping
    `weights`.

    >>> c = np.array([[2, 0, 1],
    ...               [1, 0, 0],
    ...               [0, 0, 0]])
    >>> k = np.array([[0, 1, 0],
    ...               [0, 1, 0],
    ...               [0, 1, 0],
    ...               [0, 1, 0],
    ...               [0, 1, 0]])
    >>> ndimage.convolve(c, k, mode='nearest')
    array([[7, 0, 3],
           [5, 0, 2],
           [3, 0, 1]])

    """
    return _correlate_or_convolve(input, weights, output, mode, cval,
                                  origin, True, axes)


@_ni_docstrings.docfiller
def uniform_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D uniform filter along the given axis.

    The lines of the array along the given axis are filtered with a
    uniform filter of given size.

    Parameters
    ----------
    %(input)s
    size : int
        length of uniform filter
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Returns
    -------
    result : ndarray
        Filtered array. Has same shape as `input`.

    Examples
    --------
    >>> from scipy.ndimage import uniform_filter1d
    >>> uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
    array([4, 3, 4, 1, 4, 6, 6, 3])
    """
    input = np.asarray(input)
    axis = normalize_axis_index(axis, input.ndim)
    if size < 1:
        raise RuntimeError('incorrect filter size')
    complex_output = input.dtype.kind == 'c'
    output = _ni_support._get_output(output, input,
                                     complex_output=complex_output)
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    if not complex_output:
        _nd_image.uniform_filter1d(input, size, axis, output, mode, cval,
                                   origin)
    else:
        _nd_image.uniform_filter1d(input.real, size, axis, output.real, mode,
                                   np.real(cval), origin)
        _nd_image.uniform_filter1d(input.imag, size, axis, output.imag, mode,
                                   np.imag(cval), origin)
    return output


@_ni_docstrings.docfiller
def uniform_filter(input, size=3, output=None, mode="reflect",
                   cval=0.0, origin=0, *, axes=None):
    """Multidimensional uniform filter.

    Parameters
    ----------
    %(input)s
    size : int or sequence of ints, optional
        The sizes of the uniform filter are given for each axis as a
        sequence, or as a single number, in which case the size is
        equal for all axes.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(origin_multiple)s
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `size`, `origin`, and/or `mode`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    uniform_filter : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D uniform filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.uniform_filter(ascent, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)
    output = _ni_support._get_output(output, input,
                                     complex_output=input.dtype.kind == 'c')
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    sizes = _ni_support._normalize_sequence(size, num_axes)
    origins = _ni_support._normalize_sequence(origin, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    axes = [(axes[ii], sizes[ii], origins[ii], modes[ii])
            for ii in range(num_axes) if sizes[ii] > 1]
    if len(axes) > 0:
        for axis, size, origin, mode in axes:
            uniform_filter1d(input, int(size), axis, output, mode,
                             cval, origin)
            input = output
    else:
        output[...] = input[...]
    return output


@_ni_docstrings.docfiller
def minimum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D minimum filter along the given axis.

    The lines of the array along the given axis are filtered with a
    minimum filter of given size.

    Parameters
    ----------
    %(input)s
    size : int
        length along which to calculate 1D minimum
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Returns
    -------
    result : ndarray.
        Filtered image. Has the same shape as `input`.

    Notes
    -----
    This function implements the MINLIST algorithm [1]_, as described by
    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
    the `input` length, regardless of filter size.

    References
    ----------
    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html


    Examples
    --------
    >>> from scipy.ndimage import minimum_filter1d
    >>> minimum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
    array([2, 0, 0, 0, 1, 1, 0, 0])
    """
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axis = normalize_axis_index(axis, input.ndim)
    if size < 1:
        raise RuntimeError('incorrect filter size')
    output = _ni_support._get_output(output, input)
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.min_or_max_filter1d(input, size, axis, output, mode, cval,
                                  origin, 1)
    return output


@_ni_docstrings.docfiller
def maximum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D maximum filter along the given axis.

    The lines of the array along the given axis are filtered with a
    maximum filter of given size.

    Parameters
    ----------
    %(input)s
    size : int
        Length along which to calculate the 1-D maximum.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Returns
    -------
    maximum1d : ndarray, None
        Maximum-filtered array with same shape as input.
        None if `output` is not None

    Notes
    -----
    This function implements the MAXLIST algorithm [1]_, as described by
    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
    the `input` length, regardless of filter size.

    References
    ----------
    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html

    Examples
    --------
    >>> from scipy.ndimage import maximum_filter1d
    >>> maximum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
    array([8, 8, 8, 4, 9, 9, 9, 9])
    """
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axis = normalize_axis_index(axis, input.ndim)
    if size < 1:
        raise RuntimeError('incorrect filter size')
    output = _ni_support._get_output(output, input)
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.min_or_max_filter1d(input, size, axis, output, mode, cval,
                                  origin, 0)
    return output


def _min_or_max_filter(input, size, footprint, structure, output, mode,
                       cval, origin, minimum, axes=None):
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=3)
    if structure is None:
        if footprint is None:
            if size is None:
                raise RuntimeError("no footprint provided")
            separable = True
        else:
            footprint = np.asarray(footprint, dtype=bool)
            if not footprint.any():
                raise ValueError("All-zero footprint is not supported.")
            if footprint.all():
                size = footprint.shape
                footprint = None
                separable = True
            else:
                separable = False
    else:
        structure = np.asarray(structure, dtype=np.float64)
        separable = False
        if footprint is None:
            footprint = np.ones(structure.shape, bool)
        else:
            footprint = np.asarray(footprint, dtype=bool)
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError("Complex type not supported")
    output = _ni_support._get_output(output, input)
    temp_needed = np.may_share_memory(input, output)
    if temp_needed:
        # input and output arrays cannot share memory
        temp = output
        output = _ni_support._get_output(output.dtype, input)
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    if separable:
        origins = _ni_support._normalize_sequence(origin, num_axes)
        sizes = _ni_support._normalize_sequence(size, num_axes)
        modes = _ni_support._normalize_sequence(mode, num_axes)
        axes = [(axes[ii], sizes[ii], origins[ii], modes[ii])
                for ii in range(len(axes)) if sizes[ii] > 1]
        if minimum:
            filter_ = minimum_filter1d
        else:
            filter_ = maximum_filter1d
        if len(axes) > 0:
            for axis, size, origin, mode in axes:
                filter_(input, int(size), axis, output, mode, cval, origin)
                input = output
        else:
            output[...] = input[...]
    else:
        # expand origins and footprint if num_axes < input.ndim
        footprint = _expand_footprint(input.ndim, axes, footprint)
        origins = _expand_origin(input.ndim, axes, origin)

        fshape = [ii for ii in footprint.shape if ii > 0]
        if len(fshape) != input.ndim:
            raise RuntimeError(f"footprint.ndim ({footprint.ndim}) must match "
                               f"len(axes) ({len(axes)})")
        for origin, lenf in zip(origins, fshape):
            if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
                raise ValueError("invalid origin")
        if not footprint.flags.contiguous:
            footprint = footprint.copy()
        if structure is not None:
            if len(structure.shape) != num_axes:
                raise RuntimeError("structure array has incorrect shape")
            if num_axes != structure.ndim:
                structure = np.expand_dims(
                    structure,
                    tuple(ax for ax in range(structure.ndim) if ax not in axes)
                )
            if not structure.flags.contiguous:
                structure = structure.copy()
        if not isinstance(mode, str) and isinstance(mode, Iterable):
            raise RuntimeError(
                "A sequence of modes is not supported for non-separable "
                "footprints")
        mode = _ni_support._extend_mode_to_code(mode)
        _nd_image.min_or_max_filter(input, footprint, structure, output,
                                    mode, cval, origins, minimum)
    if temp_needed:
        temp[...] = output
        output = temp
    return output


@_ni_docstrings.docfiller
def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0, *, axes=None):
    """Calculate a multidimensional minimum filter.

    Parameters
    ----------
    %(input)s
    %(size_foot)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(origin_multiple)s
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `size`, `origin`, and/or `mode`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    minimum_filter : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    A sequence of modes (one per axis) is only supported when the footprint is
    separable. Otherwise, a single mode string must be provided.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.minimum_filter(ascent, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 1, axes)


@_ni_docstrings.docfiller
def maximum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0, *, axes=None):
    """Calculate a multidimensional maximum filter.

    Parameters
    ----------
    %(input)s
    %(size_foot)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(origin_multiple)s
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `size`, `origin`, and/or `mode`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    maximum_filter : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    A sequence of modes (one per axis) is only supported when the footprint is
    separable. Otherwise, a single mode string must be provided.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.maximum_filter(ascent, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 0, axes)


@_ni_docstrings.docfiller
def _rank_filter(input, rank, size=None, footprint=None, output=None,
                 mode="reflect", cval=0.0, origin=0, operation='rank',
                 axes=None):
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=3)
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _ni_support._normalize_sequence(size, num_axes)
        footprint = np.ones(sizes, dtype=bool)
    else:
        footprint = np.asarray(footprint, dtype=bool)
    # expand origins, footprint and modes if num_axes < input.ndim
    footprint = _expand_footprint(input.ndim, axes, footprint)
    origins = _expand_origin(input.ndim, axes, origin)
    mode = _expand_mode(input.ndim, axes, mode)

    fshape = [ii for ii in footprint.shape if ii > 0]
    if len(fshape) != input.ndim:
        raise RuntimeError(f"footprint.ndim ({footprint.ndim}) must match "
                           f"len(axes) ({len(axes)})")
    for origin, lenf in zip(origins, fshape):
        if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
            raise ValueError('invalid origin')
    if not footprint.flags.contiguous:
        footprint = footprint.copy()
    filter_size = np.where(footprint, 1, 0).sum()
    if operation == 'median':
        rank = filter_size // 2
    elif operation == 'percentile':
        percentile = rank
        if percentile < 0.0:
            percentile += 100.0
        if percentile < 0 or percentile > 100:
            raise RuntimeError('invalid percentile')
        if percentile == 100.0:
            rank = filter_size - 1
        else:
            rank = int(float(filter_size) * percentile / 100.0)
    if rank < 0:
        rank += filter_size
    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    if rank == 0:
        return minimum_filter(input, None, footprint, output, mode, cval,
                              origins, axes=None)
    elif rank == filter_size - 1:
        return maximum_filter(input, None, footprint, output, mode, cval,
                              origins, axes=None)
    else:
        output = _ni_support._get_output(output, input)
        temp_needed = np.may_share_memory(input, output)
        if temp_needed:
            # input and output arrays cannot share memory
            temp = output
            output = _ni_support._get_output(output.dtype, input)
        if not isinstance(mode, str) and isinstance(mode, Iterable):
            raise RuntimeError(
                "A sequence of modes is not supported by non-separable rank "
                "filters")
        mode = _ni_support._extend_mode_to_code(mode, is_filter=True)
        if input.ndim == 1:
            if input.dtype in (np.int64, np.float64, np.float32):
                x = input
                x_out = output
            elif input.dtype == np.float16:
                x = input.astype('float32')
                x_out = np.empty(x.shape, dtype='float32')
            elif np.result_type(input, np.int64) == np.int64:
                x = input.astype('int64')
                x_out = np.empty(x.shape, dtype='int64')
            elif input.dtype.kind in 'biu':
                # cast any other boolean, integer or unsigned type to int64
                x = input.astype('int64')
                x_out = np.empty(x.shape, dtype='int64')
            else:
                raise RuntimeError('Unsupported array type')
            cval = x.dtype.type(cval)
            _rank_filter_1d.rank_filter(x, rank, footprint.size, x_out, mode, cval,
                                        origin)
            if input.dtype not in (np.int64, np.float64, np.float32):
                np.copyto(output, x_out, casting='unsafe')
        else:
            _nd_image.rank_filter(input, rank, footprint, output, mode, cval, origins)
        if temp_needed:
            temp[...] = output
            output = temp
        return output


@_ni_docstrings.docfiller
def rank_filter(input, rank, size=None, footprint=None, output=None,
                mode="reflect", cval=0.0, origin=0, *, axes=None):
    """Calculate a multidimensional rank filter.

    Parameters
    ----------
    %(input)s
    rank : int
        The rank parameter may be less than zero, i.e., rank = -1
        indicates the largest element.
    %(size_foot)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin_multiple)s
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `size`, `origin`, and/or `mode`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    rank_filter : ndarray
        Filtered array. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.rank_filter(ascent, rank=42, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    rank = operator.index(rank)
    return _rank_filter(input, rank, size, footprint, output, mode, cval,
                        origin, 'rank', axes=axes)


@_ni_docstrings.docfiller
def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0, *, axes=None):
    """
    Calculate a multidimensional median filter.

    Parameters
    ----------
    %(input)s
    %(size_foot)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin_multiple)s
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `size`, `origin`, and/or `mode`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    median_filter : ndarray
        Filtered array. Has the same shape as `input`.

    See Also
    --------
    scipy.signal.medfilt2d

    Notes
    -----
    For 2-dimensional images with ``uint8``, ``float32`` or ``float64`` dtypes
    the specialised function `scipy.signal.medfilt2d` may be faster. It is
    however limited to constant mode with ``cval=0``.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.median_filter(ascent, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    return _rank_filter(input, 0, size, footprint, output, mode, cval,
                        origin, 'median', axes=axes)


@_ni_docstrings.docfiller
def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0, *,
                      axes=None):
    """Calculate a multidimensional percentile filter.

    Parameters
    ----------
    %(input)s
    percentile : scalar
        The percentile parameter may be less than zero, i.e.,
        percentile = -20 equals percentile = 80
    %(size_foot)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin_multiple)s
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `size`, `origin`, and/or `mode`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    percentile_filter : ndarray
        Filtered array. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.percentile_filter(ascent, percentile=20, size=20)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    return _rank_filter(input, percentile, size, footprint, output, mode,
                        cval, origin, 'percentile', axes=axes)


@_ni_docstrings.docfiller
def generic_filter1d(input, function, filter_size, axis=-1,
                     output=None, mode="reflect", cval=0.0, origin=0,
                     extra_arguments=(), extra_keywords=None):
    """Calculate a 1-D filter along the given axis.

    `generic_filter1d` iterates over the lines of the array, calling the
    given function at each line. The arguments of the line are the
    input line, and the output line. The input and output lines are 1-D
    double arrays. The input line is extended appropriately according
    to the filter size and origin. The output line must be modified
    in-place with the result.

    Parameters
    ----------
    %(input)s
    function : {callable, scipy.LowLevelCallable}
        Function to apply along given axis.
    filter_size : scalar
        Length of the filter.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s
    %(extra_arguments)s
    %(extra_keywords)s

    Returns
    -------
    generic_filter1d : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    This function also accepts low-level callback functions with one of
    the following signatures and wrapped in `scipy.LowLevelCallable`:

    .. code:: c

       int function(double *input_line, npy_intp input_length,
                    double *output_line, npy_intp output_length,
                    void *user_data)
       int function(double *input_line, intptr_t input_length,
                    double *output_line, intptr_t output_length,
                    void *user_data)

    The calling function iterates over the lines of the input and output
    arrays, calling the callback function at each line. The current line
    is extended according to the border conditions set by the calling
    function, and the result is copied into the array that is passed
    through ``input_line``. The length of the input line (after extension)
    is passed through ``input_length``. The callback function should apply
    the filter and store the result in the array passed through
    ``output_line``. The length of the output line is passed through
    ``output_length``. ``user_data`` is the data pointer provided
    to `scipy.LowLevelCallable` as-is.

    The callback function must return an integer error status that is zero
    if something went wrong and one otherwise. If an error occurs, you should
    normally set the python error status with an informative message
    before returning, otherwise a default error message is set by the
    calling function.

    In addition, some other low-level function pointer specifications
    are accepted, but these are for backward compatibility only and should
    not be used in new code.

    """
    if extra_keywords is None:
        extra_keywords = {}
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    output = _ni_support._get_output(output, input)
    if filter_size < 1:
        raise RuntimeError('invalid filter size')
    axis = normalize_axis_index(axis, input.ndim)
    if (filter_size // 2 + origin < 0) or (filter_size // 2 + origin >=
                                           filter_size):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.generic_filter1d(input, function, filter_size, axis, output,
                               mode, cval, origin, extra_arguments,
                               extra_keywords)
    return output


@_ni_docstrings.docfiller
def generic_filter(input, function, size=None, footprint=None,
                   output=None, mode="reflect", cval=0.0, origin=0,
                   extra_arguments=(), extra_keywords=None, *, axes=None):
    """Calculate a multidimensional filter using the given function.

    At each element the provided function is called. The input values
    within the filter footprint at that element are passed to the function
    as a 1-D array of double values.

    Parameters
    ----------
    %(input)s
    function : {callable, scipy.LowLevelCallable}
        Function to apply at each element.
    %(size_foot)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin_multiple)s
    %(extra_arguments)s
    %(extra_keywords)s
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `size` or `origin` must match the length
        of `axes`. The ith entry in any of these tuples corresponds to the ith
        entry in `axes`.

    Returns
    -------
    generic_filter : ndarray
        Filtered array. Has the same shape as `input`.

    Notes
    -----
    This function also accepts low-level callback functions with one of
    the following signatures and wrapped in `scipy.LowLevelCallable`:

    .. code:: c

       int callback(double *buffer, npy_intp filter_size,
                    double *return_value, void *user_data)
       int callback(double *buffer, intptr_t filter_size,
                    double *return_value, void *user_data)

    The calling function iterates over the elements of the input and
    output arrays, calling the callback function at each element. The
    elements within the footprint of the filter at the current element are
    passed through the ``buffer`` parameter, and the number of elements
    within the footprint through ``filter_size``. The calculated value is
    returned in ``return_value``. ``user_data`` is the data pointer provided
    to `scipy.LowLevelCallable` as-is.

    The callback function must return an integer error status that is zero
    if something went wrong and one otherwise. If an error occurs, you should
    normally set the python error status with an informative message
    before returning, otherwise a default error message is set by the
    calling function.

    In addition, some other low-level function pointer specifications
    are accepted, but these are for backward compatibility only and should
    not be used in new code.

    Examples
    --------
    Import the necessary modules and load the example image used for
    filtering.

    >>> import numpy as np
    >>> from scipy import datasets
    >>> from scipy.ndimage import zoom, generic_filter
    >>> import matplotlib.pyplot as plt
    >>> ascent = zoom(datasets.ascent(), 0.5)

    Compute a maximum filter with kernel size 5 by passing a simple NumPy
    aggregation function as argument to `function`.

    >>> maximum_filter_result = generic_filter(ascent, np.amax, [5, 5])

    While a maximum filter could also directly be obtained using
    `maximum_filter`, `generic_filter` allows generic Python function or
    `scipy.LowLevelCallable` to be used as a filter. Here, we compute the
    range between maximum and minimum value as an example for a kernel size
    of 5.

    >>> def custom_filter(image):
    ...     return np.amax(image) - np.amin(image)
    >>> custom_filter_result = generic_filter(ascent, custom_filter, [5, 5])

    Plot the original and filtered images.

    >>> fig, axes = plt.subplots(3, 1, figsize=(3, 9))
    >>> plt.gray()  # show the filtered result in grayscale
    >>> top, middle, bottom = axes
    >>> for ax in axes:
    ...     ax.set_axis_off()  # remove coordinate system
    >>> top.imshow(ascent)
    >>> top.set_title("Original image")
    >>> middle.imshow(maximum_filter_result)
    >>> middle.set_title("Maximum filter, Kernel: 5x5")
    >>> bottom.imshow(custom_filter_result)
    >>> bottom.set_title("Custom filter, Kernel: 5x5")
    >>> fig.tight_layout()

    """
    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=2)
    if extra_keywords is None:
        extra_keywords = {}
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axes = _ni_support._check_axes(axes, input.ndim)
    num_axes = len(axes)
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _ni_support._normalize_sequence(size, num_axes)
        footprint = np.ones(sizes, dtype=bool)
    else:
        footprint = np.asarray(footprint, dtype=bool)

    # expand origins, footprint if num_axes < input.ndim
    footprint = _expand_footprint(input.ndim, axes, footprint)
    origins = _expand_origin(input.ndim, axes, origin)

    fshape = [ii for ii in footprint.shape if ii > 0]
    if len(fshape) != input.ndim:
        raise RuntimeError(f"footprint.ndim ({footprint.ndim}) "
                           f"must match len(axes) ({num_axes})")
    for origin, lenf in zip(origins, fshape):
        if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
            raise ValueError('invalid origin')
    if not footprint.flags.contiguous:
        footprint = footprint.copy()
    output = _ni_support._get_output(output, input)

    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.generic_filter(input, function, footprint, output, mode,
                             cval, origins, extra_arguments, extra_keywords)
    return output
<|endoftext|># Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools
import warnings

import numpy as np
from scipy._lib._util import normalize_axis_index

from scipy import special
from . import _ni_support
from . import _nd_image
from ._ni_docstrings import docfiller


__all__ = ['spline_filter1d', 'spline_filter', 'geometric_transform',
           'map_coordinates', 'affine_transform', 'shift', 'zoom', 'rotate']


@docfiller
def spline_filter1d(input, order=3, axis=-1, output=np.float64,
                    mode='mirror'):
    """
    Calculate a 1-D spline filter along the given axis.

    The lines of the array along the given axis are filtered by a
    spline filter. The order of the spline must be >= 2 and <= 5.

    Parameters
    ----------
    %(input)s
    order : int, optional
        The order of the spline, default is 3.
    axis : int, optional
        The axis along which the spline filter is applied. Default is the last
        axis.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array. Default is ``numpy.float64``.
    %(mode_interp_mirror)s

    Returns
    -------
    spline_filter1d : ndarray
        The filtered input.

    See Also
    --------
    spline_filter : Multidimensional spline filter.

    Notes
    -----
    All of the interpolation functions in `ndimage` do spline interpolation of
    the input image. If using B-splines of `order > 1`, the input image
    values have to be converted to B-spline coefficients first, which is
    done by applying this 1-D filter sequentially along all
    axes of the input. All functions that require B-spline coefficients
    will automatically filter their inputs, a behavior controllable with
    the `prefilter` keyword argument. For functions that accept a `mode`
    parameter, the result will only be correct if it matches the `mode`
    used when filtering.

    For complex-valued `input`, this function processes the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    We can filter an image using 1-D spline along the given axis:

    >>> from scipy.ndimage import spline_filter1d
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter_axis_0 = spline_filter1d(orig_img, axis=0)
    >>> sp_filter_axis_1 = spline_filter1d(orig_img, axis=1)
    >>> f, ax = plt.subplots(1, 3, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...             [sp_filter_axis_0, "spline filter (axis=0)"],
    ...             [sp_filter_axis_1, "spline filter (axis=1)"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input,
                                     complex_output=complex_output)
    if complex_output:
        spline_filter1d(input.real, order, axis, output.real, mode)
        spline_filter1d(input.imag, order, axis, output.imag, mode)
        return output
    if order in [0, 1]:
        output[...] = np.array(input)
    else:
        mode = _ni_support._extend_mode_to_code(mode)
        axis = normalize_axis_index(axis, input.ndim)
        _nd_image.spline_filter1d(input, order, axis, output, mode)
    return output

@docfiller
def spline_filter(input, order=3, output=np.float64, mode='mirror'):
    """
    Multidimensional spline filter.

    Parameters
    ----------
    %(input)s
    order : int, optional
        The order of the spline, default is 3.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array. Default is ``numpy.float64``.
    %(mode_interp_mirror)s

    Returns
    -------
    spline_filter : ndarray
        Filtered array. Has the same shape as `input`.

    See Also
    --------
    spline_filter1d : Calculate a 1-D spline filter along the given axis.

    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D spline filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    For complex-valued `input`, this function processes the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    We can filter an image using multidimensional splines:

    >>> from scipy.ndimage import spline_filter
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter = spline_filter(orig_img, order=3)
    >>> f, ax = plt.subplots(1, 2, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...                             [sp_filter, "spline filter"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if order < 2 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input,
                                     complex_output=complex_output)
    if complex_output:
        spline_filter(input.real, order, output.real, mode)
        spline_filter(input.imag, order, output.imag, mode)
        return output
    if order not in [0, 1] and input.ndim > 0:
        for axis in range(input.ndim):
            spline_filter1d(input, order, axis, output=output, mode=mode)
            input = output
    else:
        output[...] = input[...]
    return output


def _prepad_for_spline_filter(input, mode, cval):
    if mode in ['nearest', 'grid-constant']:
        npad = 12
        if mode == 'grid-constant':
            padded = np.pad(input, npad, mode='constant',
                               constant_values=cval)
        elif mode == 'nearest':
            padded = np.pad(input, npad, mode='edge')
    else:
        # other modes have exact boundary conditions implemented so
        # no prepadding is needed
        npad = 0
        padded = input
    return padded, npad


@docfiller
def geometric_transform(input, mapping, output_shape=None,
                        output=None, order=3,
                        mode='constant', cval=0.0, prefilter=True,
                        extra_arguments=(), extra_keywords=None):
    """
    Apply an arbitrary geometric transform.

    The given mapping function is used to find, for each point in the
    output, the corresponding coordinates in the input. The value of the
    input at those coordinates is determined by spline interpolation of
    the requested order.

    Parameters
    ----------
    %(input)s
    mapping : {callable, scipy.LowLevelCallable}
        A callable object that accepts a tuple of length equal to the output
        array rank, and returns the corresponding input coordinates as a tuple
        of length equal to the input array rank.
    output_shape : tuple of ints, optional
        Shape tuple.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s
    extra_arguments : tuple, optional
        Extra arguments passed to `mapping`.
    extra_keywords : dict, optional
        Extra keywords passed to `mapping`.

    Returns
    -------
    output : ndarray
        The filtered input.

    See Also
    --------
    map_coordinates, affine_transform, spline_filter1d


    Notes
    -----
    This function also accepts low-level callback functions with one
    the following signatures and wrapped in `scipy.LowLevelCallable`:

    .. code:: c

       int mapping(npy_intp *output_coordinates, double *input_coordinates,
                   int output_rank, int input_rank, void *user_data)
       int mapping(intptr_t *output_coordinates, double *input_coordinates,
                   int output_rank, int input_rank, void *user_data)

    The calling function iterates over the elements of the output array,
    calling the callback function at each element. The coordinates of the
    current output element are passed through ``output_coordinates``. The
    callback function must return the coordinates at which the input must
    be interpolated in ``input_coordinates``. The rank of the input and
    output arrays are given by ``input_rank`` and ``output_rank``
    respectively. ``user_data`` is the data pointer provided
    to `scipy.LowLevelCallable` as-is.

    The callback function must return an integer error status that is zero
    if something went wrong and one otherwise. If an error occurs, you should
    normally set the Python error status with an informative message
    before returning, otherwise a default error message is set by the
    calling function.

    In addition, some other low-level function pointer specifications
    are accepted, but these are for backward compatibility only and should
    not be used in new code.

    For complex-valued `input`, this function transforms the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import geometric_transform
    >>> a = np.arange(12.).reshape((4, 3))
    >>> def shift_func(output_coords):
    ...     return (output_coords[0] - 0.5, output_coords[1] - 0.5)
    ...
    >>> geometric_transform(a, shift_func)
    array([[ 0.   ,  0.   ,  0.   ],
           [ 0.   ,  1.362,  2.738],
           [ 0.   ,  4.812,  6.187],
           [ 0.   ,  8.263,  9.637]])

    >>> b = [1, 2, 3, 4, 5]
    >>> def shift_func(output_coords):
    ...     return (output_coords[0] - 3,)
    ...
    >>> geometric_transform(b, shift_func, mode='constant')
    array([0, 0, 0, 1, 2])
    >>> geometric_transform(b, shift_func, mode='nearest')
    array([1, 1, 1, 1, 2])
    >>> geometric_transform(b, shift_func, mode='reflect')
    array([3, 2, 1, 1, 2])
    >>> geometric_transform(b, shift_func, mode='wrap')
    array([2, 3, 4, 1, 2])

    """
    if extra_keywords is None:
        extra_keywords = {}
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if output_shape is None:
        output_shape = input.shape
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    if complex_output:
        kwargs = dict(order=order, mode=mode, prefilter=prefilter,
                      output_shape=output_shape,
                      extra_arguments=extra_arguments,
                      extra_keywords=extra_keywords)
        geometric_transform(input.real, mapping, output=output.real,
                            cval=np.real(cval), **kwargs)
        geometric_transform(input.imag, mapping, output=output.imag,
                            cval=np.imag(cval), **kwargs)
        return output

    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64,
                                 mode=mode)
    else:
        npad = 0
        filtered = input
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.geometric_transform(filtered, mapping, None, None, None, output,
                                  order, mode, cval, npad, extra_arguments,
                                  extra_keywords)
    return output


@docfiller
def map_coordinates(input, coordinates, output=None, order=3,
                    mode='constant', cval=0.0, prefilter=True):
    """
    Map the input array to new coordinates by interpolation.

    The array of coordinates is used to find, for each point in the output,
    the corresponding coordinates in the input. The value of the input at
    those coordinates is determined by spline interpolation of the
    requested order.

    The shape of the output is derived from that of the coordinate
    array by dropping the first axis. The values of the array along
    the first axis are the coordinates in the input array at which the
    output value is found.

    Parameters
    ----------
    %(input)s
    coordinates : array_like
        The coordinates at which `input` is evaluated.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    map_coordinates : ndarray
        The result of transforming the input. The shape of the output is
        derived from that of `coordinates` by dropping the first axis.

    See Also
    --------
    spline_filter, geometric_transform, scipy.interpolate

    Notes
    -----
    For complex-valued `input`, this function maps the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.arange(12.).reshape((4, 3))
    >>> a
    array([[  0.,   1.,   2.],
           [  3.,   4.,   5.],
           [  6.,   7.,   8.],
           [  9.,  10.,  11.]])
    >>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)
    array([ 2.,  7.])

    Above, the interpolated value of a[0.5, 0.5] gives output[0], while
    a[2, 1] is output[1].

    >>> inds = np.array([[0.5, 2], [0.5, 4]])
    >>> ndimage.map_coordinates(a, inds, order=1, cval=-33.3)
    array([  2. , -33.3])
    >>> ndimage.map_coordinates(a, inds, order=1, mode='nearest')
    array([ 2.,  8.])
    >>> ndimage.map_coordinates(a, inds, order=1, cval=0, output=bool)
    array([ True, False], dtype=bool)

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    coordinates = np.asarray(coordinates)
    if np.iscomplexobj(coordinates):
        raise TypeError('Complex type not supported')
    output_shape = coordinates.shape[1:]
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')
    if coordinates.shape[0] != input.ndim:
        raise RuntimeError('invalid shape for coordinate array')
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    if complex_output:
        kwargs = dict(order=order, mode=mode, prefilter=prefilter)
        map_coordinates(input.real, coordinates, output=output.real,
                        cval=np.real(cval), **kwargs)
        map_coordinates(input.imag, coordinates, output=output.imag,
                        cval=np.imag(cval), **kwargs)
        return output
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        npad = 0
        filtered = input
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.geometric_transform(filtered, None, coordinates, None, None,
                                  output, order, mode, cval, npad, None, None)
    return output


@docfiller
def affine_transform(input, matrix, offset=0.0, output_shape=None,
                     output=None, order=3,
                     mode='constant', cval=0.0, prefilter=True):
    """
    Apply an affine transformation.

    Given an output image pixel index vector ``o``, the pixel value
    is determined from the input image at position
    ``np.dot(matrix, o) + offset``.

    This does 'pull' (or 'backward') resampling, transforming the output space
    to the input to locate data. Affine transformations are often described in
    the 'push' (or 'forward') direction, transforming input to output. If you
    have a matrix for the 'push' transformation, use its inverse
    (:func:`numpy.linalg.inv`) in this function.

    Parameters
    ----------
    %(input)s
    matrix : ndarray
        The inverse coordinate transformation matrix, mapping output
        coordinates to input coordinates. If ``ndim`` is the number of
        dimensions of ``input``, the given matrix must have one of the
        following shapes:

            - ``(ndim, ndim)``: the linear transformation matrix for each
              output coordinate.
            - ``(ndim,)``: assume that the 2-D transformation matrix is
              diagonal, with the diagonal specified by the given value. A more
              efficient algorithm is then used that exploits the separability
              of the problem.
            - ``(ndim + 1, ndim + 1)``: assume that the transformation is
              specified using homogeneous coordinates [1]_. In this case, any
              value passed to ``offset`` is ignored.
            - ``(ndim, ndim + 1)``: as above, but the bottom row of a
              homogeneous transformation matrix is always ``[0, 0, ..., 1]``,
              and may be omitted.

    offset : float or sequence, optional
        The offset into the array where the transform is applied. If a float,
        `offset` is the same for each axis. If a sequence, `offset` should
        contain one value for each axis.
    output_shape : tuple of ints, optional
        Shape tuple.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    affine_transform : ndarray
        The transformed input.

    Notes
    -----
    The given matrix and offset are used to find for each point in the
    output the corresponding coordinates in the input by an affine
    transformation. The value of the input at those coordinates is
    determined by spline interpolation of the requested order. Points
    outside the boundaries of the input are filled according to the given
    mode.

    .. versionchanged:: 0.18.0
        Previously, the exact interpretation of the affine transformation
        depended on whether the matrix was supplied as a 1-D or a
        2-D array. If a 1-D array was supplied
        to the matrix parameter, the output pixel value at index ``o``
        was determined from the input image at position
        ``matrix * (o + offset)``.

    For complex-valued `input`, this function transforms the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates
    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if output_shape is None:
        if isinstance(output, np.ndarray):
            output_shape = output.shape
        else:
            output_shape = input.shape
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    if complex_output:
        kwargs = dict(offset=offset, output_shape=output_shape, order=order,
                      mode=mode, prefilter=prefilter)
        affine_transform(input.real, matrix, output=output.real,
                         cval=np.real(cval), **kwargs)
        affine_transform(input.imag, matrix, output=output.imag,
                         cval=np.imag(cval), **kwargs)
        return output
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        npad = 0
        filtered = input
    mode = _ni_support._extend_mode_to_code(mode)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim not in [1, 2] or matrix.shape[0] < 1:
        raise RuntimeError('no proper affine matrix provided')
    if (matrix.ndim == 2 and matrix.shape[1] == input.ndim + 1 and
            (matrix.shape[0] in [input.ndim, input.ndim + 1])):
        if matrix.shape[0] == input.ndim + 1:
            exptd = [0] * input.ndim + [1]
            if not np.all(matrix[input.ndim] == exptd):
                msg = (f'Expected homogeneous transformation matrix with '
                       f'shape {matrix.shape} for image shape {input.shape}, '
                       f'but bottom row was not equal to {exptd}')
                raise ValueError(msg)
        # assume input is homogeneous coordinate transformation matrix
        offset = matrix[:input.ndim, input.ndim]
        matrix = matrix[:input.ndim, :input.ndim]
    if matrix.shape[0] != input.ndim:
        raise RuntimeError('affine matrix has wrong number of rows')
    if matrix.ndim == 2 and matrix.shape[1] != output.ndim:
        raise RuntimeError('affine matrix has wrong number of columns')
    if not matrix.flags.contiguous:
        matrix = matrix.copy()
    offset = _ni_support._normalize_sequence(offset, input.ndim)
    offset = np.asarray(offset, dtype=np.float64)
    if offset.ndim != 1 or offset.shape[0] < 1:
        raise RuntimeError('no proper offset provided')
    if not offset.flags.contiguous:
        offset = offset.copy()
    if matrix.ndim == 1:
        warnings.warn(
            "The behavior of affine_transform with a 1-D "
            "array supplied for the matrix parameter has changed in "
            "SciPy 0.18.0.",
            stacklevel=2
        )
        _nd_image.zoom_shift(filtered, matrix, offset/matrix, output, order,
                             mode, cval, npad, False)
    else:
        _nd_image.geometric_transform(filtered, None, None, matrix, offset,
                                      output, order, mode, cval, npad, None,
                                      None)
    return output


@docfiller
def shift(input, shift, output=None, order=3, mode='constant', cval=0.0,
          prefilter=True):
    """
    Shift an array.

    The array is shifted using spline interpolation of the requested order.
    Points outside the boundaries of the input are filled according to the
    given mode.

    Parameters
    ----------
    %(input)s
    shift : float or sequence
        The shift along the axes. If a float, `shift` is the same for each
        axis. If a sequence, `shift` should contain one value for each axis.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    shift : ndarray
        The shifted input.

    See Also
    --------
    affine_transform : Affine transformations

    Notes
    -----
    For complex-valued `input`, this function shifts the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    Import the necessary modules and an exemplary image.

    >>> from scipy.ndimage import shift
    >>> import matplotlib.pyplot as plt
    >>> from scipy import datasets
    >>> image = datasets.ascent()

    Shift the image vertically by 20 pixels.

    >>> image_shifted_vertically = shift(image, (20, 0))

    Shift the image vertically by -200 pixels and horizontally by 100 pixels.

    >>> image_shifted_both_directions = shift(image, (-200, 100))

    Plot the original and the shifted images.

    >>> fig, axes = plt.subplots(3, 1, figsize=(4, 12))
    >>> plt.gray()  # show the filtered result in grayscale
    >>> top, middle, bottom = axes
    >>> for ax in axes:
    ...     ax.set_axis_off()  # remove coordinate system
    >>> top.imshow(image)
    >>> top.set_title("Original image")
    >>> middle.imshow(image_shifted_vertically)
    >>> middle.set_title("Vertically shifted image")
    >>> bottom.imshow(image_shifted_both_directions)
    >>> bottom.set_title("Image shifted in both directions")
    >>> fig.tight_layout()
    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if input.ndim < 1:
        raise RuntimeError('input and output rank must be > 0')
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input, complex_output=complex_output)
    if complex_output:
        # import under different name to avoid confusion with shift parameter
        from scipy.ndimage._interpolation import shift as _shift

        kwargs = dict(order=order, mode=mode, prefilter=prefilter)
        _shift(input.real, shift, output=output.real, cval=np.real(cval), **kwargs)
        _shift(input.imag, shift, output=output.imag, cval=np.imag(cval), **kwargs)
        return output
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        npad = 0
        filtered = input
    mode = _ni_support._extend_mode_to_code(mode)
    shift = _ni_support._normalize_sequence(shift, input.ndim)
    shift = [-ii for ii in shift]
    shift = np.asarray(shift, dtype=np.float64)
    if not shift.flags.contiguous:
        shift = shift.copy()
    _nd_image.zoom_shift(filtered, None, shift, output, order, mode, cval,
                         npad, False)
    return output


@docfiller
def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0,
         prefilter=True, *, grid_mode=False):
    """
    Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Parameters
    ----------
    %(input)s
    zoom : float or sequence
        The zoom factor along the axes. If a float, `zoom` is the same for each
        axis. If a sequence, `zoom` should contain one value for each axis.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s
    grid_mode : bool, optional
        If False, the distance from the pixel centers is zoomed. Otherwise, the
        distance including the full pixel extent is used. For example, a 1d
        signal of length 5 is considered to have length 4 when `grid_mode` is
        False, but length 5 when `grid_mode` is True. See the following
        visual illustration:

        .. code-block:: text

                | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                     |<-------------------------------------->|
                                        vs.
                |<----------------------------------------------->|

        The starting point of the arrow in the diagram above corresponds to
        coordinate location 0 in each mode.

    Returns
    -------
    zoom : ndarray
        The zoomed input.

    Notes
    -----
    For complex-valued `input`, this function zooms the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = ndimage.zoom(ascent, 3.0)
    >>> ax1.imshow(ascent, vmin=0, vmax=255)
    >>> ax2.imshow(result, vmin=0, vmax=255)
    >>> plt.show()

    >>> print(ascent.shape)
    (512, 512)

    >>> print(result.shape)
    (1536, 1536)
    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if input.ndim < 1:
        raise RuntimeError('input and output rank must be > 0')
    zoom = _ni_support._normalize_sequence(zoom, input.ndim)
    output_shape = tuple(
            [int(round(ii * jj)) for ii, jj in zip(input.shape, zoom)])
    complex_output = np.iscomplexobj(input)
    output = _ni_support._get_output(output, input, shape=output_shape,
                                     complex_output=complex_output)
    if complex_output:
        # import under different name to avoid confusion with zoom parameter
        from scipy.ndimage._interpolation import zoom as _zoom

        kwargs = dict(order=order, mode=mode, prefilter=prefilter)
        _zoom(input.real, zoom, output=output.real, cval=np.real(cval), **kwargs)
        _zoom(input.imag, zoom, output=output.imag, cval=np.imag(cval), **kwargs)
        return output
    if prefilter and order > 1:
        padded, npad = _prepad_for_spline_filter(input, mode, cval)
        filtered = spline_filter(padded, order, output=np.float64, mode=mode)
    else:
        npad = 0
        filtered = input
    if grid_mode:
        # warn about modes that may have surprising behavior
        suggest_mode = None
        if mode == 'constant':
            suggest_mode = 'grid-constant'
        elif mode == 'wrap':
            suggest_mode = 'grid-wrap'
        if suggest_mode is not None:
            warnings.warn(
                (f"It is recommended to use mode = {suggest_mode} instead of {mode} "
                 f"when grid_mode is True."),
                stacklevel=2
            )
    mode = _ni_support._extend_mode_to_code(mode)

    zoom_div = np.array(output_shape)
    zoom_nominator = np.array(input.shape)
    if not grid_mode:
        zoom_div -= 1
        zoom_nominator -= 1

    # Zooming to infinite values is unpredictable, so just choose
    # zoom factor 1 instead
    zoom = np.divide(zoom_nominator, zoom_div,
                     out=np.ones_like(input.shape, dtype=np.float64),
                     where=zoom_div != 0)
    zoom = np.ascontiguousarray(zoom)
    _nd_image.zoom_shift(filtered, zoom, None, output, order, mode, cval, npad,
                         grid_mode)
    return output


@docfiller
def rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=3,
           mode='constant', cval=0.0, prefilter=True):
    """
    Rotate an array.

    The array is rotated in the plane defined by the two axes given by the
    `axes` parameter using spline interpolation of the requested order.

    Parameters
    ----------
    %(input)s
    angle : float
        The rotation angle in degrees.
    axes : tuple of 2 ints, optional
        The two axes that define the plane of rotation. Default is the first
        two axes.
    reshape : bool, optional
        If `reshape` is true, the output shape is adapted so that the input
        array is contained completely in the output. Default is True.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    rotate : ndarray
        The rotated input.

    Notes
    -----
    For complex-valued `input`, this function rotates the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure(figsize=(10, 3))
    >>> ax1, ax2, ax3 = fig.subplots(1, 3)
    >>> img = datasets.ascent()
    >>> img_45 = ndimage.rotate(img, 45, reshape=False)
    >>> full_img_45 = ndimage.rotate(img, 45, reshape=True)
    >>> ax1.imshow(img, cmap='gray')
    >>> ax1.set_axis_off()
    >>> ax2.imshow(img_45, cmap='gray')
    >>> ax2.set_axis_off()
    >>> ax3.imshow(full_img_45, cmap='gray')
    >>> ax3.set_axis_off()
    >>> fig.set_layout_engine('tight')
    >>> plt.show()
    >>> print(img.shape)
    (512, 512)
    >>> print(img_45.shape)
    (512, 512)
    >>> print(full_img_45.shape)
    (724, 724)

    """
    input_arr = np.asarray(input)
    ndim = input_arr.ndim

    if ndim < 2:
        raise ValueError('input array should be at least 2D')

    axes = list(axes)

    if len(axes) != 2:
        raise ValueError('axes should contain exactly two values')

    if not all([float(ax).is_integer() for ax in axes]):
        raise ValueError('axes should contain only integer values')

    if axes[0] < 0:
        axes[0] += ndim
    if axes[1] < 0:
        axes[1] += ndim
    if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
        raise ValueError('invalid rotation plane specified')

    axes.sort()

    c, s = special.cosdg(angle), special.sindg(angle)

    rot_matrix = np.array([[c, s],
                           [-s, c]])

    img_shape = np.asarray(input_arr.shape)
    in_plane_shape = img_shape[axes]
    if reshape:
        # Compute transformed input bounds
        iy, ix = in_plane_shape
        out_bounds = rot_matrix @ [[0, 0, iy, iy],
                                   [0, ix, 0, ix]]
        # Compute the shape of the transformed input plane
        out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)
    else:
        out_plane_shape = img_shape[axes]

    out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
    in_center = (in_plane_shape - 1) / 2
    offset = in_center - out_center

    output_shape = img_shape
    output_shape[axes] = out_plane_shape
    output_shape = tuple(output_shape)

    complex_output = np.iscomplexobj(input_arr)
    output = _ni_support._get_output(output, input_arr, shape=output_shape,
                                     complex_output=complex_output)

    if ndim <= 2:
        affine_transform(input_arr, rot_matrix, offset, output_shape, output,
                         order, mode, cval, prefilter)
    else:
        # If ndim > 2, the rotation is applied over all the planes
        # parallel to axes
        planes_coord = itertools.product(
            *[[slice(None)] if ax in axes else range(img_shape[ax])
              for ax in range(ndim)])

        out_plane_shape = tuple(out_plane_shape)

        for coordinates in planes_coord:
            ia = input_arr[coordinates]
            oa = output[coordinates]
            affine_transform(ia, rot_matrix, offset, out_plane_shape,
                             oa, order, mode, cval, prefilter)

    return output
<|endoftext|># Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections.abc import Iterable
import operator
import warnings
import numpy as np


def _extend_mode_to_code(mode, is_filter=False):
    """Convert an extension mode to the corresponding integer code.
    """
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode in ['reflect', 'grid-mirror']:
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    elif mode == 'grid-wrap' and is_filter:
        return 1
    elif mode == 'grid-wrap':
        return 5
    elif mode == 'grid-constant' and is_filter:
        return 4
    elif mode == 'grid-constant':
        return 6
    else:
        raise RuntimeError('boundary mode not supported')


def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, str)
    if not is_str and np.iterable(input):
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized


def _get_output(output, input, shape=None, complex_output=False):
    if shape is None:
        shape = input.shape
    if output is None:
        if not complex_output:
            output = np.zeros(shape, dtype=input.dtype.name)
        else:
            complex_type = np.promote_types(input.dtype, np.complex64)
            output = np.zeros(shape, dtype=complex_type)
    elif isinstance(output, (type, np.dtype)):
        # Classes (like `np.float32`) and dtypes are interpreted as dtype
        if complex_output and np.dtype(output).kind != 'c':
            warnings.warn("promoting specified output dtype to complex", stacklevel=3)
            output = np.promote_types(output, np.complex64)
        output = np.zeros(shape, dtype=output)
    elif isinstance(output, str):
        output = np.dtype(output)
        if complex_output and output.kind != 'c':
            raise RuntimeError("output must have complex dtype")
        elif not issubclass(output.type, np.number):
            raise RuntimeError("output must have numeric dtype")
        output = np.zeros(shape, dtype=output)
    else:
        # output was supplied as an array
        output = np.asarray(output)
        if output.shape != shape:
            raise RuntimeError("output shape not correct")
        elif complex_output and output.dtype.kind != 'c':
            raise RuntimeError("output must have complex dtype")
    return output


def _check_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    elif np.isscalar(axes):
        axes = (operator.index(axes),)
    elif isinstance(axes, Iterable):
        for ax in axes:
            axes = tuple(operator.index(ax) for ax in axes)
            if ax < -ndim or ax > ndim - 1:
                raise ValueError(f"specified axis: {ax} is out of range")
        axes = tuple(ax % ndim if ax < 0 else ax for ax in axes)
    else:
        message = "axes must be an integer, iterable of integers, or None"
        raise ValueError(message)
    if len(tuple(set(axes))) != len(axes):
        raise ValueError("axes must be unique")
    return axes

def _skip_if_dtype(arg):
    """'array or dtype' polymorphism.

    Return None for np.int8, dtype('float32') or 'f' etc
           arg for np.empty(3) etc
    """
    if isinstance(arg, str):
        return None
    if type(arg) is type:
        return None if issubclass(arg, np.generic) else arg
    else:
        return None if isinstance(arg, np.dtype) else arg


def _skip_if_int(arg):
    return None if (arg is None or isinstance(arg, int)) else arg
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'label', 'find_objects', 'labeled_comprehension',
    'sum', 'mean', 'variance', 'standard_deviation',
    'minimum', 'maximum', 'median', 'minimum_position',
    'maximum_position', 'extrema', 'center_of_mass',
    'histogram', 'watershed_ift', 'sum_labels'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package='ndimage', module='measurements',
                                   private_modules=['_measurements'], all=__all__,
                                   attribute=name)
<|endoftext|># Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from scipy._lib._util import normalize_axis_index
from . import _ni_support
from . import _nd_image

__all__ = ['fourier_gaussian', 'fourier_uniform', 'fourier_ellipsoid',
           'fourier_shift']


def _get_output_fourier(output, input):
    if output is None:
        if input.dtype.type in [np.complex64, np.complex128, np.float32]:
            output = np.zeros(input.shape, dtype=input.dtype)
        else:
            output = np.zeros(input.shape, dtype=np.float64)
    elif type(output) is type:
        if output not in [np.complex64, np.complex128,
                          np.float32, np.float64]:
            raise RuntimeError("output type not supported")
        output = np.zeros(input.shape, dtype=output)
    elif output.shape != input.shape:
        raise RuntimeError("output shape not correct")
    return output


def _get_output_fourier_complex(output, input):
    if output is None:
        if input.dtype.type in [np.complex64, np.complex128]:
            output = np.zeros(input.shape, dtype=input.dtype)
        else:
            output = np.zeros(input.shape, dtype=np.complex128)
    elif type(output) is type:
        if output not in [np.complex64, np.complex128]:
            raise RuntimeError("output type not supported")
        output = np.zeros(input.shape, dtype=output)
    elif output.shape != input.shape:
        raise RuntimeError("output shape not correct")
    return output


def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
    """
    Multidimensional Gaussian fourier filter.

    The array is multiplied with the fourier transform of a Gaussian
    kernel.

    Parameters
    ----------
    input : array_like
        The input array.
    sigma : float or sequence
        The sigma of the Gaussian kernel. If a float, `sigma` is the same for
        all axes. If a sequence, `sigma` has to contain one value for each
        axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of filtering the input is placed in this array.

    Returns
    -------
    fourier_gaussian : ndarray
        The filtered input.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = datasets.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_gaussian(input_, sigma=4)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    input = np.asarray(input)
    output = _get_output_fourier(output, input)
    axis = normalize_axis_index(axis, input.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if not sigmas.flags.contiguous:
        sigmas = sigmas.copy()

    _nd_image.fourier_filter(input, sigmas, n, axis, output, 0)
    return output


def fourier_uniform(input, size, n=-1, axis=-1, output=None):
    """
    Multidimensional uniform fourier filter.

    The array is multiplied with the Fourier transform of a box of given
    size.

    Parameters
    ----------
    input : array_like
        The input array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of filtering the input is placed in this array.

    Returns
    -------
    fourier_uniform : ndarray
        The filtered input.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = datasets.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_uniform(input_, size=20)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    input = np.asarray(input)
    output = _get_output_fourier(output, input)
    axis = normalize_axis_index(axis, input.ndim)
    sizes = _ni_support._normalize_sequence(size, input.ndim)
    sizes = np.asarray(sizes, dtype=np.float64)
    if not sizes.flags.contiguous:
        sizes = sizes.copy()
    _nd_image.fourier_filter(input, sizes, n, axis, output, 1)
    return output


def fourier_ellipsoid(input, size, n=-1, axis=-1, output=None):
    """
    Multidimensional ellipsoid Fourier filter.

    The array is multiplied with the fourier transform of an ellipsoid of
    given sizes.

    Parameters
    ----------
    input : array_like
        The input array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of filtering the input is placed in this array.

    Returns
    -------
    fourier_ellipsoid : ndarray
        The filtered input.

    Notes
    -----
    This function is implemented for arrays of rank 1, 2, or 3.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = datasets.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_ellipsoid(input_, size=20)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    input = np.asarray(input)
    if input.ndim > 3:
        raise NotImplementedError("Only 1d, 2d and 3d inputs are supported")
    output = _get_output_fourier(output, input)
    if output.size == 0:
        # The C code has a bug that can result in a segfault with arrays
        # that have size 0 (gh-17270), so check here.
        return output
    axis = normalize_axis_index(axis, input.ndim)
    sizes = _ni_support._normalize_sequence(size, input.ndim)
    sizes = np.asarray(sizes, dtype=np.float64)
    if not sizes.flags.contiguous:
        sizes = sizes.copy()
    _nd_image.fourier_filter(input, sizes, n, axis, output, 2)
    return output


def fourier_shift(input, shift, n=-1, axis=-1, output=None):
    """
    Multidimensional Fourier shift filter.

    The array is multiplied with the Fourier transform of a shift operation.

    Parameters
    ----------
    input : array_like
        The input array.
    shift : float or sequence
        The size of the box used for filtering.
        If a float, `shift` is the same for all axes. If a sequence, `shift`
        has to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of shifting the input is placed in this array.

    Returns
    -------
    fourier_shift : ndarray
        The shifted input.

    Examples
    --------
    >>> from scipy import ndimage, datasets
    >>> import matplotlib.pyplot as plt
    >>> import numpy.fft
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = datasets.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_shift(input_, shift=200)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    input = np.asarray(input)
    output = _get_output_fourier_complex(output, input)
    axis = normalize_axis_index(axis, input.ndim)
    shifts = _ni_support._normalize_sequence(shift, input.ndim)
    shifts = np.asarray(shifts, dtype=np.float64)
    if not shifts.flags.contiguous:
        shifts = shifts.copy()
    _nd_image.fourier_shift(input, shifts, n, axis, output)
    return output
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.ndimage` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'spline_filter1d', 'spline_filter',
    'geometric_transform', 'map_coordinates',
    'affine_transform', 'shift', 'zoom', 'rotate',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package='ndimage', module='interpolation',
                                   private_modules=['_interpolation'], all=__all__,
                                   attribute=name)
<|endoftext|># Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from . import _ni_support
from . import _ni_label
from . import _nd_image
from . import _morphology

__all__ = ['label', 'find_objects', 'labeled_comprehension', 'sum', 'mean',
           'variance', 'standard_deviation', 'minimum', 'maximum', 'median',
           'minimum_position', 'maximum_position', 'extrema', 'center_of_mass',
           'histogram', 'watershed_ift', 'sum_labels', 'value_indices']


def label(input, structure=None, output=None):
    """
    Label features in an array.

    Parameters
    ----------
    input : array_like
        An array-like object to be labeled. Any non-zero values in `input` are
        counted as features and zero values are considered the background.
    structure : array_like, optional
        A structuring element that defines feature connections.
        `structure` must be centrosymmetric
        (see Notes).
        If no structuring element is provided,
        one is automatically generated with a squared connectivity equal to
        one.  That is, for a 2-D `input` array, the default structuring element
        is::

            [[0,1,0],
             [1,1,1],
             [0,1,0]]

    output : (None, data-type, array_like), optional
        If `output` is a data type, it specifies the type of the resulting
        labeled feature array.
        If `output` is an array-like object, then `output` will be updated
        with the labeled features from this function.  This function can
        operate in-place, by passing output=input.
        Note that the output must be able to store the largest label, or this
        function will raise an Exception.

    Returns
    -------
    label : ndarray or int
        An integer ndarray where each unique feature in `input` has a unique
        label in the returned array.
    num_features : int
        How many objects were found.

        If `output` is None, this function returns a tuple of
        (`labeled_array`, `num_features`).

        If `output` is a ndarray, then it will be updated with values in
        `labeled_array` and only `num_features` will be returned by this
        function.

    See Also
    --------
    find_objects : generate a list of slices for the labeled features (or
                   objects); useful for finding features' position or
                   dimensions

    Notes
    -----
    A centrosymmetric matrix is a matrix that is symmetric about the center.
    See [1]_ for more information.

    The `structure` matrix must be centrosymmetric to ensure
    two-way connections.
    For instance, if the `structure` matrix is not centrosymmetric
    and is defined as::

        [[0,1,0],
         [1,1,0],
         [0,0,0]]

    and the `input` is::

        [[1,2],
         [0,3]]

    then the structure matrix would indicate the
    entry 2 in the input is connected to 1,
    but 1 is not connected to 2.

    References
    ----------
    .. [1] James R. Weaver, "Centrosymmetric (cross-symmetric)
       matrices, their basic properties, eigenvalues, and
       eigenvectors." The American Mathematical Monthly 92.10
       (1985): 711-717.

    Examples
    --------
    Create an image with some features, then label it using the default
    (cross-shaped) structuring element:

    >>> from scipy.ndimage import label, generate_binary_structure
    >>> import numpy as np
    >>> a = np.array([[0,0,1,1,0,0],
    ...               [0,0,0,1,0,0],
    ...               [1,1,0,0,1,0],
    ...               [0,0,0,1,0,0]])
    >>> labeled_array, num_features = label(a)

    Each of the 4 features are labeled with a different integer:

    >>> num_features
    4
    >>> labeled_array
    array([[0, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [2, 2, 0, 0, 3, 0],
           [0, 0, 0, 4, 0, 0]], dtype=int32)

    Generate a structuring element that will consider features connected even
    if they touch diagonally:

    >>> s = generate_binary_structure(2,2)

    or,

    >>> s = [[1,1,1],
    ...      [1,1,1],
    ...      [1,1,1]]

    Label the image using the new structuring element:

    >>> labeled_array, num_features = label(a, structure=s)

    Show the 2 labeled features (note that features 1, 3, and 4 from above are
    now considered a single feature):

    >>> num_features
    2
    >>> labeled_array
    array([[0, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [2, 2, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 0]], dtype=int32)

    """
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = _morphology.generate_binary_structure(input.ndim, 1)
    structure = np.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for ii in structure.shape:
        if ii != 3:
            raise ValueError('structure dimensions must be equal to 3')

    # Use 32 bits if it's large enough for this image.
    # _ni_label.label() needs two entries for background and
    # foreground tracking
    need_64bits = input.size >= (2**31 - 2)

    if isinstance(output, np.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        else:
            output = np.empty(input.shape, output)

    # handle scalars, 0-D arrays
    if input.ndim == 0 or input.size == 0:
        if input.ndim == 0:
            # scalar
            maxlabel = 1 if (input != 0) else 0
            output[...] = maxlabel
        else:
            # 0-D
            maxlabel = 0
        if caller_provided_output:
            return maxlabel
        else:
            return output, maxlabel

    try:
        max_label = _ni_label._label(input, structure, output)
    except _ni_label.NeedMoreBits as e:
        # Make another attempt with enough bits, then try to cast to the
        # new type.
        tmp_output = np.empty(input.shape, np.intp if need_64bits else np.int32)
        max_label = _ni_label._label(input, structure, tmp_output)
        output[...] = tmp_output[...]
        if not np.all(output == tmp_output):
            # refuse to return bad results
            raise RuntimeError(
                "insufficient bit-depth in requested output type"
            ) from e

    if caller_provided_output:
        # result was written in-place
        return max_label
    else:
        return output, max_label


def find_objects(input, max_label=0):
    """
    Find objects in a labeled array.

    Parameters
    ----------
    input : ndarray of ints
        Array containing objects defined by different labels. Labels with
        value 0 are ignored.
    max_label : int, optional
        Maximum label to be searched for in `input`. If max_label is not
        given, the positions of all objects are returned.

    Returns
    -------
    object_slices : list of tuples
        A list of tuples, with each tuple containing N slices (with N the
        dimension of the input array). Slices correspond to the minimal
        parallelepiped that contains the object. If a number is missing,
        None is returned instead of a slice. The label ``l`` corresponds to
        the index ``l-1`` in the returned list.

    See Also
    --------
    label, center_of_mass

    Notes
    -----
    This function is very useful for isolating a volume of interest inside
    a 3-D array, that cannot be "seen through".

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.zeros((6,6), dtype=int)
    >>> a[2:4, 2:4] = 1
    >>> a[4, 4] = 1
    >>> a[:2, :3] = 2
    >>> a[0, 5] = 3
    >>> a
    array([[2, 2, 2, 0, 0, 3],
           [2, 2, 2, 0, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> ndimage.find_objects(a)
    [(slice(2, 5, None), slice(2, 5, None)),
     (slice(0, 2, None), slice(0, 3, None)),
     (slice(0, 1, None), slice(5, 6, None))]
    >>> ndimage.find_objects(a, max_label=2)
    [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None))]
    >>> ndimage.find_objects(a == 1, max_label=2)
    [(slice(2, 5, None), slice(2, 5, None)), None]

    >>> loc = ndimage.find_objects(a)[0]
    >>> a[loc]
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 1]])

    """
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')

    if max_label < 1:
        max_label = input.max()

    return _nd_image.find_objects(input, max_label)


def value_indices(arr, *, ignore_value=None):
    """
    Find indices of each distinct value in given array.

    Parameters
    ----------
    arr : ndarray of ints
        Array containing integer values.
    ignore_value : int, optional
        This value will be ignored in searching the `arr` array. If not
        given, all values found will be included in output. Default
        is None.

    Returns
    -------
    indices : dictionary
        A Python dictionary of array indices for each distinct value. The
        dictionary is keyed by the distinct values, the entries are array
        index tuples covering all occurrences of the value within the
        array.

        This dictionary can occupy significant memory, usually several times
        the size of the input array.

    See Also
    --------
    label, maximum, median, minimum_position, extrema, sum, mean, variance,
    standard_deviation, numpy.where, numpy.unique

    Notes
    -----
    For a small array with few distinct values, one might use
    `numpy.unique()` to find all possible values, and ``(arr == val)`` to
    locate each value within that array. However, for large arrays,
    with many distinct values, this can become extremely inefficient,
    as locating each value would require a new search through the entire
    array. Using this function, there is essentially one search, with
    the indices saved for all distinct values.

    This is useful when matching a categorical image (e.g. a segmentation
    or classification) to an associated image of other data, allowing
    any per-class statistic(s) to then be calculated. Provides a
    more flexible alternative to functions like ``scipy.ndimage.mean()``
    and ``scipy.ndimage.variance()``.

    Some other closely related functionality, with different strengths and
    weaknesses, can also be found in ``scipy.stats.binned_statistic()`` and
    the `scikit-image <https://scikit-image.org/>`_ function
    ``skimage.measure.regionprops()``.

    Note for IDL users: this provides functionality equivalent to IDL's
    REVERSE_INDICES option (as per the IDL documentation for the
    `HISTOGRAM <https://www.l3harrisgeospatial.com/docs/histogram.html>`_
    function).

    .. versionadded:: 1.10.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> a = np.zeros((6, 6), dtype=int)
    >>> a[2:4, 2:4] = 1
    >>> a[4, 4] = 1
    >>> a[:2, :3] = 2
    >>> a[0, 5] = 3
    >>> a
    array([[2, 2, 2, 0, 0, 3],
           [2, 2, 2, 0, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> val_indices = ndimage.value_indices(a)

    The dictionary `val_indices` will have an entry for each distinct
    value in the input array.

    >>> val_indices.keys()
    dict_keys([np.int64(0), np.int64(1), np.int64(2), np.int64(3)])

    The entry for each value is an index tuple, locating the elements
    with that value.

    >>> ndx1 = val_indices[1]
    >>> ndx1
    (array([2, 2, 3, 3, 4]), array([2, 3, 2, 3, 4]))

    This can be used to index into the original array, or any other
    array with the same shape.

    >>> a[ndx1]
    array([1, 1, 1, 1, 1])

    If the zeros were to be ignored, then the resulting dictionary
    would no longer have an entry for zero.

    >>> val_indices = ndimage.value_indices(a, ignore_value=0)
    >>> val_indices.keys()
    dict_keys([np.int64(1), np.int64(2), np.int64(3)])

    """
    # Cope with ignore_value being None, without too much extra complexity
    # in the C code. If not None, the value is passed in as a numpy array
    # with the same dtype as arr.
    arr = np.asarray(arr)
    ignore_value_arr = np.zeros((1,), dtype=arr.dtype)
    ignoreIsNone = (ignore_value is None)
    if not ignoreIsNone:
        ignore_value_arr[0] = ignore_value_arr.dtype.type(ignore_value)

    val_indices = _nd_image.value_indices(arr, ignoreIsNone, ignore_value_arr)
    return val_indices


def labeled_comprehension(input, labels, index, func, out_dtype, default,
                          pass_positions=False):
    """
    Roughly equivalent to [func(input[labels == i]) for i in index].

    Sequentially applies an arbitrary function (that works on array_like input)
    to subsets of an N-D image array specified by `labels` and `index`.
    The option exists to provide the function with positional parameters as the
    second argument.

    Parameters
    ----------
    input : array_like
        Data from which to select `labels` to process.
    labels : array_like or None
        Labels to objects in `input`.
        If not None, array must be same shape as `input`.
        If None, `func` is applied to raveled `input`.
    index : int, sequence of ints or None
        Subset of `labels` to which to apply `func`.
        If a scalar, a single value is returned.
        If None, `func` is applied to all non-zero values of `labels`.
    func : callable
        Python function to apply to `labels` from `input`.
    out_dtype : dtype
        Dtype to use for `result`.
    default : int, float or None
        Default return value when a element of `index` does not exist
        in `labels`.
    pass_positions : bool, optional
        If True, pass linear indices to `func` as a second argument.
        Default is False.

    Returns
    -------
    result : ndarray
        Result of applying `func` to each of `labels` to `input` in `index`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from scipy import ndimage
    >>> lbl, nlbl = ndimage.label(a)
    >>> lbls = np.arange(1, nlbl+1)
    >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, 0)
    array([ 2.75,  5.5 ,  6.  ])

    Falling back to `default`:

    >>> lbls = np.arange(1, nlbl+2)
    >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, -1)
    array([ 2.75,  5.5 ,  6.  , -1.  ])

    Passing positions:

    >>> def fn(val, pos):
    ...     print("fn says: %s : %s" % (val, pos))
    ...     return (val.sum()) if (pos.sum() % 2 == 0) else (-val.sum())
    ...
    >>> ndimage.labeled_comprehension(a, lbl, lbls, fn, float, 0, True)
    fn says: [1 2 5 3] : [0 1 4 5]
    fn says: [4 7] : [ 7 11]
    fn says: [9 3] : [12 13]
    array([ 11.,  11., -12.,   0.])

    """

    as_scalar = np.isscalar(index)
    input = np.asarray(input)

    if pass_positions:
        positions = np.arange(input.size).reshape(input.shape)

    if labels is None:
        if index is not None:
            raise ValueError("index without defined labels")
        if not pass_positions:
            return func(input.ravel())
        else:
            return func(input.ravel(), positions.ravel())

    labels = np.asarray(labels)

    try:
        input, labels = np.broadcast_arrays(input, labels)
    except ValueError as e:
        raise ValueError("input and labels must have the same shape "
                            "(excepting dimensions with width 1)") from e

    if index is None:
        if not pass_positions:
            return func(input[labels > 0])
        else:
            return func(input[labels > 0], positions[labels > 0])

    index = np.atleast_1d(index)
    if np.any(index.astype(labels.dtype).astype(index.dtype) != index):
        raise ValueError(f"Cannot convert index values from <{index.dtype}> to "
                         f"<{labels.dtype}> (labels' type) without loss of precision")

    index = index.astype(labels.dtype)

    # optimization: find min/max in index,
    # and select those parts of labels, input, and positions
    lo = index.min()
    hi = index.max()
    mask = (labels >= lo) & (labels <= hi)

    # this also ravels the arrays
    labels = labels[mask]
    input = input[mask]
    if pass_positions:
        positions = positions[mask]

    # sort everything by labels
    label_order = labels.argsort()
    labels = labels[label_order]
    input = input[label_order]
    if pass_positions:
        positions = positions[label_order]

    index_order = index.argsort()
    sorted_index = index[index_order]

    def do_map(inputs, output):
        """labels must be sorted"""
        nidx = sorted_index.size

        # Find boundaries for each stretch of constant labels
        # This could be faster, but we already paid N log N to sort labels.
        lo = np.searchsorted(labels, sorted_index, side='left')
        hi = np.searchsorted(labels, sorted_index, side='right')

        for i, l, h in zip(range(nidx), lo, hi):
            if l == h:
                continue
            output[i] = func(*[inp[l:h] for inp in inputs])

    temp = np.empty(index.shape, out_dtype)
    temp[:] = default
    if not pass_positions:
        do_map([input], temp)
    else:
        do_map([input, positions], temp)

    output = np.zeros(index.shape, out_dtype)
    output[index_order] = temp
    if as_scalar:
        output = output[0]

    return output


def _safely_castable_to_int(dt):
    """Test whether the NumPy data type `dt` can be safely cast to an int."""
    int_size = np.dtype(int).itemsize
    safe = ((np.issubdtype(dt, np.signedinteger) and dt.itemsize <= int_size) or
            (np.issubdtype(dt, np.unsignedinteger) and dt.itemsize < int_size))
    return safe


def _stats(input, labels=None, index=None, centered=False):
    """Count, sum, and optionally compute (sum - centre)^2 of input by label

    Parameters
    ----------
    input : array_like, N-D
        The input data to be analyzed.
    labels : array_like (N-D), optional
        The labels of the data in `input`. This array must be broadcast
        compatible with `input`; typically, it is the same shape as `input`.
        If `labels` is None, all nonzero values in `input` are treated as
        the single labeled group.
    index : label or sequence of labels, optional
        These are the labels of the groups for which the stats are computed.
        If `index` is None, the stats are computed for the single group where
        `labels` is greater than 0.
    centered : bool, optional
        If True, the centered sum of squares for each labeled group is
        also returned. Default is False.

    Returns
    -------
    counts : int or ndarray of ints
        The number of elements in each labeled group.
    sums : scalar or ndarray of scalars
        The sums of the values in each labeled group.
    sums_c : scalar or ndarray of scalars, optional
        The sums of mean-centered squares of the values in each labeled group.
        This is only returned if `centered` is True.

    """
    def single_group(vals):
        if centered:
            vals_c = vals - vals.mean()
            return vals.size, vals.sum(), (vals_c * vals_c.conjugate()).sum()
        else:
            return vals.size, vals.sum()

    input = np.asarray(input)
    if labels is None:
        return single_group(input)

    # ensure input and labels match sizes
    input, labels = np.broadcast_arrays(input, labels)

    if index is None:
        return single_group(input[labels > 0])

    if np.isscalar(index):
        return single_group(input[labels == index])

    def _sum_centered(labels):
        # `labels` is expected to be an ndarray with the same shape as `input`.
        # It must contain the label indices (which are not necessarily the labels
        # themselves).
        means = sums / counts
        centered_input = input - means[labels]
        # bincount expects 1-D inputs, so we ravel the arguments.
        bc = np.bincount(labels.ravel(),
                              weights=(centered_input *
                                       centered_input.conjugate()).ravel())
        return bc

    # Remap labels to unique integers if necessary, or if the largest
    # label is larger than the number of values.

    if (not _safely_castable_to_int(labels.dtype) or
            labels.min() < 0 or labels.max() > labels.size):
        # Use np.unique to generate the label indices.  `new_labels` will
        # be 1-D, but it should be interpreted as the flattened N-D array of
        # label indices.
        unique_labels, new_labels = np.unique(labels, return_inverse=True)
        new_labels = np.reshape(new_labels, (-1,))  # flatten, since it may be >1-D
        counts = np.bincount(new_labels)
        sums = np.bincount(new_labels, weights=input.ravel())
        if centered:
            # Compute the sum of the mean-centered squares.
            # We must reshape new_labels to the N-D shape of `input` before
            # passing it _sum_centered.
            sums_c = _sum_centered(new_labels.reshape(labels.shape))
        idxs = np.searchsorted(unique_labels, index)
        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = (unique_labels[idxs] == index)
    else:
        # labels are an integer type allowed by bincount, and there aren't too
        # many, so call bincount directly.
        counts = np.bincount(labels.ravel())
        sums = np.bincount(labels.ravel(), weights=input.ravel())
        if centered:
            sums_c = _sum_centered(labels)
        # make sure all index values are valid
        idxs = np.asanyarray(index, np.int_).copy()
        found = (idxs >= 0) & (idxs < counts.size)
        idxs[~found] = 0

    counts = counts[idxs]
    counts[~found] = 0
    sums = sums[idxs]
    sums[~found] = 0

    if not centered:
        return (counts, sums)
    else:
        sums_c = sums_c[idxs]
        sums_c[~found] = 0
        return (counts, sums, sums_c)


def sum(input, labels=None, index=None):
    """
    Calculate the sum of the values of the array.

    Notes
    -----
    This is an alias for `ndimage.sum_labels` kept for backwards compatibility
    reasons, for new code please prefer `sum_labels`.  See the `sum_labels`
    docstring for more details.

    """
    return sum_labels(input, labels, index)


def sum_labels(input, labels=None, index=None):
    """
    Calculate the sum of the values of the array.

    Parameters
    ----------
    input : array_like
        Values of `input` inside the regions defined by `labels`
        are summed together.
    labels : array_like of ints, optional
        Assign labels to the values of the array. Has to have the same shape as
        `input`.
    index : array_like, optional
        A single label number or a sequence of label numbers of
        the objects to be measured.

    Returns
    -------
    sum : ndarray or scalar
        An array of the sums of values of `input` inside the regions defined
        by `labels` with the same shape as `index`. If 'index' is None or scalar,
        a scalar is returned.

    See Also
    --------
    mean, median

    Examples
    --------
    >>> from scipy import ndimage
    >>> input =  [0,1,2,3]
    >>> labels = [1,1,2,2]
    >>> ndimage.sum_labels(input, labels, index=[1,2])
    [1.0, 5.0]
    >>> ndimage.sum_labels(input, labels, index=1)
    1
    >>> ndimage.sum_labels(input, labels)
    6


    """
    count, sum = _stats(input, labels, index)
    return sum


def mean(input, labels=None, index=None):
    """
    Calculate the mean of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array on which to compute the mean of elements over distinct
        regions.
    labels : array_like, optional
        Array of labels of same shape, or broadcastable to the same shape as
        `input`. All elements sharing the same label form one region over
        which the mean of the elements is computed.
    index : int or sequence of ints, optional
        Labels of the objects over which the mean is to be computed.
        Default is None, in which case the mean for all values where label is
        greater than 0 is calculated.

    Returns
    -------
    out : list
        Sequence of same length as `index`, with the mean of the different
        regions labeled by the labels in `index`.

    See Also
    --------
    variance, standard_deviation, minimum, maximum, sum, label

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.arange(25).reshape((5,5))
    >>> labels = np.zeros_like(a)
    >>> labels[3:5,3:5] = 1
    >>> index = np.unique(labels)
    >>> labels
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1],
           [0, 0, 0, 1, 1]])
    >>> index
    array([0, 1])
    >>> ndimage.mean(a, labels=labels, index=index)
    [10.285714285714286, 21.0]

    """

    count, sum = _stats(input, labels, index)
    return sum / np.asanyarray(count).astype(np.float64)


def variance(input, labels=None, index=None):
    """
    Calculate the variance of the values of an N-D image array, optionally at
    specified sub-regions.

    Parameters
    ----------
    input : array_like
        Nd-image data to process.
    labels : array_like, optional
        Labels defining sub-regions in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        `labels` to include in output.  If None (default), all values where
        `labels` is non-zero are used.

    Returns
    -------
    variance : float or ndarray
        Values of variance, for each sub-region if `labels` and `index` are
        specified.

    See Also
    --------
    label, standard_deviation, maximum, minimum, extrema

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from scipy import ndimage
    >>> ndimage.variance(a)
    7.609375

    Features to process can be specified using `labels` and `index`:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.variance(a, lbl, index=np.arange(1, nlbl+1))
    array([ 2.1875,  2.25  ,  9.    ])

    If no index is given, all non-zero `labels` are processed:

    >>> ndimage.variance(a, lbl)
    6.1875

    """
    count, sum, sum_c_sq = _stats(input, labels, index, centered=True)
    return sum_c_sq / np.asanyarray(count).astype(float)


def standard_deviation(input, labels=None, index=None):
    """
    Calculate the standard deviation of the values of an N-D image array,
    optionally at specified sub-regions.

    Parameters
    ----------
    input : array_like
        N-D image data to process.
    labels : array_like, optional
        Labels to identify sub-regions in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        `labels` to include in output. If None (default), all values where
        `labels` is non-zero are used.

    Returns
    -------
    standard_deviation : float or ndarray
        Values of standard deviation, for each sub-region if `labels` and
        `index` are specified.

    See Also
    --------
    label, variance, maximum, minimum, extrema

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from scipy import ndimage
    >>> ndimage.standard_deviation(a)
    2.7585095613392387

    Features to process can be specified using `labels` and `index`:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.standard_deviation(a, lbl, index=np.arange(1, nlbl+1))
    array([ 1.479,  1.5  ,  3.   ])

    If no index is given, non-zero `labels` are processed:

    >>> ndimage.standard_deviation(a, lbl)
    2.4874685927665499

    """
    return np.sqrt(variance(input, labels, index))


def _select(input, labels=None, index=None, find_min=False, find_max=False,
            find_min_positions=False, find_max_positions=False,
            find_median=False):
    """Returns min, max, or both, plus their positions (if requested), and
    median."""

    input = np.asanyarray(input)

    find_positions = find_min_positions or find_max_positions
    positions = None
    if find_positions:
        positions = np.arange(input.size).reshape(input.shape)

    def single_group(vals, positions):
        result = []
        if find_min:
            result += [vals.min()]
        if find_min_positions:
            result += [positions[vals == vals.min()][0]]
        if find_max:
            result += [vals.max()]
        if find_max_positions:
            result += [positions[vals == vals.max()][0]]
        if find_median:
            result += [np.median(vals)]
        return result

    if labels is None:
        return single_group(input, positions)

    # ensure input and labels match sizes
    input, labels = np.broadcast_arrays(input, labels)

    if index is None:
        mask = (labels > 0)
        masked_positions = None
        if find_positions:
            masked_positions = positions[mask]
        return single_group(input[mask], masked_positions)

    if np.isscalar(index):
        mask = (labels == index)
        masked_positions = None
        if find_positions:
            masked_positions = positions[mask]
        return single_group(input[mask], masked_positions)

    index = np.asarray(index)

    # remap labels to unique integers if necessary, or if the largest
    # label is larger than the number of values.
    if (not _safely_castable_to_int(labels.dtype) or
            labels.min() < 0 or labels.max() > labels.size):
        # remap labels, and indexes
        unique_labels, labels = np.unique(labels, return_inverse=True)
        idxs = np.searchsorted(unique_labels, index)

        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = (unique_labels[idxs] == index)
    else:
        # labels are an integer type, and there aren't too many
        idxs = np.asanyarray(index, np.int_).copy()
        found = (idxs >= 0) & (idxs <= labels.max())

    idxs[~ found] = labels.max() + 1

    if find_median:
        order = np.lexsort((input.ravel(), labels.ravel()))
    else:
        order = input.ravel().argsort()
    input = input.ravel()[order]
    labels = labels.ravel()[order]
    if find_positions:
        positions = positions.ravel()[order]

    result = []
    if find_min:
        mins = np.zeros(labels.max() + 2, input.dtype)
        mins[labels[::-1]] = input[::-1]
        result += [mins[idxs]]
    if find_min_positions:
        minpos = np.zeros(labels.max() + 2, int)
        minpos[labels[::-1]] = positions[::-1]
        result += [minpos[idxs]]
    if find_max:
        maxs = np.zeros(labels.max() + 2, input.dtype)
        maxs[labels] = input
        result += [maxs[idxs]]
    if find_max_positions:
        maxpos = np.zeros(labels.max() + 2, int)
        maxpos[labels] = positions
        result += [maxpos[idxs]]
    if find_median:
        locs = np.arange(len(labels))
        lo = np.zeros(labels.max() + 2, np.int_)
        lo[labels[::-1]] = locs[::-1]
        hi = np.zeros(labels.max() + 2, np.int_)
        hi[labels] = locs
        lo = lo[idxs]
        hi = hi[idxs]
        # lo is an index to the lowest value in input for each label,
        # hi is an index to the largest value.
        # move them to be either the same ((hi - lo) % 2 == 0) or next
        # to each other ((hi - lo) % 2 == 1), then average.
        step = (hi - lo) // 2
        lo += step
        hi -= step
        if (np.issubdtype(input.dtype, np.integer)
                or np.issubdtype(input.dtype, np.bool_)):
            # avoid integer overflow or boolean addition (gh-12836)
            result += [(input[lo].astype('d') + input[hi].astype('d')) / 2.0]
        else:
            result += [(input[lo] + input[hi]) / 2.0]

    return result


def minimum(input, labels=None, index=None):
    """
    Calculate the minimum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        minimal values of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        minimum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the minimum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        minima. If index is None, the minimum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    minimum : float or list of floats
        List of minima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the minimal value of `input` if `labels` is None,
        and the minimal value of elements where `labels` is greater than zero
        if `index` is None.

    See Also
    --------
    label, maximum, median, minimum_position, extrema, sum, mean, variance,
    standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]], dtype=int32)
    >>> ndimage.minimum(a, labels=labels, index=np.arange(1, labels_nb + 1))
    [1, 4, 3]
    >>> ndimage.minimum(a)
    0
    >>> ndimage.minimum(a, labels=labels)
    1

    """
    return _select(input, labels, index, find_min=True)[0]


def maximum(input, labels=None, index=None):
    """
    Calculate the maximum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        maximal values of `input` over the region is computed.
    labels : array_like, optional
        An array of integers marking different regions over which the
        maximum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the maximum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        maxima. If index is None, the maximum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    output : float or list of floats
        List of maxima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the maximal value of `input` if `labels` is None,
        and the maximal value of elements where `labels` is greater than zero
        if `index` is None.

    See Also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(16).reshape((4,4))
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> labels = np.zeros_like(a)
    >>> labels[:2,:2] = 1
    >>> labels[2:, 1:3] = 2
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 0],
           [0, 2, 2, 0],
           [0, 2, 2, 0]])
    >>> from scipy import ndimage
    >>> ndimage.maximum(a)
    15
    >>> ndimage.maximum(a, labels=labels, index=[1,2])
    [5, 14]
    >>> ndimage.maximum(a, labels=labels)
    14

    >>> b = np.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(b)
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]], dtype=int32)
    >>> ndimage.maximum(b, labels=labels, index=np.arange(1, labels_nb + 1))
    [5, 7, 9]

    """
    return _select(input, labels, index, find_max=True)[0]


def median(input, labels=None, index=None):
    """
    Calculate the median of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        median value of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        median value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the median
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        medians. If index is None, the median over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    median : float or list of floats
        List of medians of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the median value of `input` if `labels` is None,
        and the median value of elements where `labels` is greater than zero
        if `index` is None.

    See Also
    --------
    label, minimum, maximum, extrema, sum, mean, variance, standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `np.array` to convert the list to an array.

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.array([[1, 2, 0, 1],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 2],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]], dtype=int32)
    >>> ndimage.median(a, labels=labels, index=np.arange(1, labels_nb + 1))
    [2.5, 4.0, 6.0]
    >>> ndimage.median(a)
    1.0
    >>> ndimage.median(a, labels=labels)
    3.0

    """
    return _select(input, labels, index, find_median=True)[0]


def minimum_position(input, labels=None, index=None):
    """
    Find the positions of the minimums of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the minimum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first minimum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the minima. If `index` is None, the ``first`` minimum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : list of tuples of ints
        Tuple of ints or list of tuples of ints that specify the location
        of minima of `input` over the regions determined by `labels` and
        whose index is in `index`.

        If `index` or `labels` are not specified, a tuple of ints is
        returned specifying the location of the first minimal value of `input`.

    See Also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10, 20, 30],
    ...               [40, 80, 100],
    ...               [1, 100, 200]])
    >>> b = np.array([[1, 2, 0, 1],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])

    >>> from scipy import ndimage

    >>> ndimage.minimum_position(a)
    (2, 0)
    >>> ndimage.minimum_position(b)
    (0, 2)

    Features to process can be specified using `labels` and `index`:

    >>> label, pos = ndimage.label(a)
    >>> ndimage.minimum_position(a, label, index=np.arange(1, pos+1))
    [(2, 0)]

    >>> label, pos = ndimage.label(b)
    >>> ndimage.minimum_position(b, label, index=np.arange(1, pos+1))
    [(0, 0), (0, 3), (3, 1)]

    """
    dims = np.array(np.asarray(input).shape)
    # see np.unravel_index to understand this line.
    dim_prod = np.cumprod([1] + list(dims[:0:-1]))[::-1]

    result = _select(input, labels, index, find_min_positions=True)[0]

    if np.isscalar(result):
        return tuple((result // dim_prod) % dims)

    return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]


def maximum_position(input, labels=None, index=None):
    """
    Find the positions of the maximums of the values of an array at labels.

    For each region specified by `labels`, the position of the maximum
    value of `input` within the region is returned.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the maximum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first maximum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the maxima. If `index` is None, the first maximum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : list of tuples of ints
        List of tuples of ints that specify the location of maxima of
        `input` over the regions determined by `labels` and whose index
        is in `index`.

