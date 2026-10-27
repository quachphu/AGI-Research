            self.cache[v] = xval

            return self.cache[v]

    def add_res(self, v, lres, bounds=None):
        v = np.ndarray.tolist(v)
        v = tuple(v)
        self.cache[v].x_l = lres.x
        self.cache[v].lres = lres
        self.cache[v].f_min = lres.fun
        self.cache[v].lbounds = bounds

        # Update cache size
        self.size += 1

        # Cache lists for search queries
        self.v_maps.append(v)
        self.xl_maps.append(lres.x)
        self.xl_maps_set.add(tuple(lres.x))
        self.f_maps.append(lres.fun)
        self.lbound_maps.append(bounds)

    def sort_cache_result(self):
        """
        Sort results and build the global return object
        """
        results = {}
        # Sort results and save
        self.xl_maps = np.array(self.xl_maps)
        self.f_maps = np.array(self.f_maps)

        # Sorted indexes in Func_min
        ind_sorted = np.argsort(self.f_maps)

        # Save ordered list of minima
        results['xl'] = self.xl_maps[ind_sorted]  # Ordered x vals
        self.f_maps = np.array(self.f_maps)
        results['funl'] = self.f_maps[ind_sorted]
        results['funl'] = results['funl'].T

        # Find global of all minimizers
        results['x'] = self.xl_maps[ind_sorted[0]]  # Save global minima
        results['fun'] = self.f_maps[ind_sorted[0]]  # Save global fun value

        self.xl_maps = np.ndarray.tolist(self.xl_maps)
        self.f_maps = np.ndarray.tolist(self.f_maps)
        return results
<|endoftext|>import numpy as np

"""
# 2023 - ported from minpack2.dcsrch, dcstep (Fortran) to Python
c     MINPACK-1 Project. June 1983.
c     Argonne National Laboratory.
c     Jorge J. More' and David J. Thuente.
c
c     MINPACK-2 Project. November 1993.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick, Richard G. Carter, and Jorge J. More'.
"""

# NOTE this file was linted by black on first commit, and can be kept that way.


class DCSRCH:
    """
    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    ftol : float
        A nonnegative tolerance for the sufficient decrease condition.
    gtol : float
        A nonnegative tolerance for the curvature condition.
    xtol : float
        A nonnegative relative tolerance for an acceptable step. The
        subroutine exits with a warning if the relative difference between
        sty and stx is less than xtol.
    stpmin : float
        A nonnegative lower bound for the step.
    stpmax :
        A nonnegative upper bound for the step.

    Notes
    -----

    This subroutine finds a step that satisfies a sufficient
    decrease condition and a curvature condition.

    Each call of the subroutine updates an interval with
    endpoints stx and sty. The interval is initially chosen
    so that it contains a minimizer of the modified function

           psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).

    If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
    interval is chosen so that it contains a minimizer of f.

    The algorithm is designed to find a step that satisfies
    the sufficient decrease condition

           f(stp) <= f(0) + ftol*stp*f'(0),

    and the curvature condition

           abs(f'(stp)) <= gtol*abs(f'(0)).

    If ftol is less than gtol and if, for example, the function
    is bounded below, then there is always a step which satisfies
    both conditions.

    If no step can be found that satisfies both conditions, then
    the algorithm stops with a warning. In this case stp only
    satisfies the sufficient decrease condition.

    A typical invocation of dcsrch has the following outline:

    Evaluate the function at stp = 0.0d0; store in f.
    Evaluate the gradient at stp = 0.0d0; store in g.
    Choose a starting step stp.

    task = 'START'
    10 continue
        call dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,
                   isave,dsave)
        if (task .eq. 'FG') then
           Evaluate the function and the gradient at stp
           go to 10
           end if

    NOTE: The user must not alter work arrays between calls.

    The subroutine statement is

        subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
                         task,isave,dsave)
        where

    stp is a double precision variable.
        On entry stp is the current estimate of a satisfactory
            step. On initial entry, a positive initial estimate
            must be provided.
        On exit stp is the current estimate of a satisfactory step
            if task = 'FG'. If task = 'CONV' then stp satisfies
            the sufficient decrease and curvature condition.

    f is a double precision variable.
        On initial entry f is the value of the function at 0.
        On subsequent entries f is the value of the
            function at stp.
        On exit f is the value of the function at stp.

    g is a double precision variable.
        On initial entry g is the derivative of the function at 0.
        On subsequent entries g is the derivative of the
           function at stp.
        On exit g is the derivative of the function at stp.

    ftol is a double precision variable.
        On entry ftol specifies a nonnegative tolerance for the
           sufficient decrease condition.
        On exit ftol is unchanged.

    gtol is a double precision variable.
        On entry gtol specifies a nonnegative tolerance for the
           curvature condition.
        On exit gtol is unchanged.

    xtol is a double precision variable.
        On entry xtol specifies a nonnegative relative tolerance
          for an acceptable step. The subroutine exits with a
          warning if the relative difference between sty and stx
          is less than xtol.

        On exit xtol is unchanged.

    task is a character variable of length at least 60.
        On initial entry task must be set to 'START'.
        On exit task indicates the required action:

           If task(1:2) = 'FG' then evaluate the function and
           derivative at stp and call dcsrch again.

           If task(1:4) = 'CONV' then the search is successful.

           If task(1:4) = 'WARN' then the subroutine is not able
           to satisfy the convergence conditions. The exit value of
           stp contains the best point found during the search.

          If task(1:5) = 'ERROR' then there is an error in the
          input arguments.

        On exit with convergence, a warning or an error, the
           variable task contains additional information.

    stpmin is a double precision variable.
        On entry stpmin is a nonnegative lower bound for the step.
        On exit stpmin is unchanged.

    stpmax is a double precision variable.
        On entry stpmax is a nonnegative upper bound for the step.
        On exit stpmax is unchanged.

    isave is an integer work array of dimension 2.

    dsave is a double precision work array of dimension 13.

    Subprograms called

      MINPACK-2 ... dcstep
    MINPACK-1 Project. June 1983.
    Argonne National Laboratory.
    Jorge J. More' and David J. Thuente.

    MINPACK-2 Project. November 1993.
    Argonne National Laboratory and University of Minnesota.
    Brett M. Averick, Richard G. Carter, and Jorge J. More'.
    """

    def __init__(self, phi, derphi, ftol, gtol, xtol, stpmin, stpmax):
        self.stage = None
        self.ginit = None
        self.gtest = None
        self.gx = None
        self.gy = None
        self.finit = None
        self.fx = None
        self.fy = None
        self.stx = None
        self.sty = None
        self.stmin = None
        self.stmax = None
        self.width = None
        self.width1 = None

        # leave all assessment of tolerances/limits to the first call of
        # this object
        self.ftol = ftol
        self.gtol = gtol
        self.xtol = xtol
        self.stpmin = stpmin
        self.stpmax = stpmax

        self.phi = phi
        self.derphi = derphi

    def __call__(self, alpha1, phi0=None, derphi0=None, maxiter=100):
        """
        Parameters
        ----------
        alpha1 : float
            alpha1 is the current estimate of a satisfactory
            step. A positive initial estimate must be provided.
        phi0 : float
            the value of `phi` at 0 (if known).
        derphi0 : float
            the derivative of `derphi` at 0 (if known).
        maxiter : int

        Returns
        -------
        alpha : float
            Step size, or None if no suitable step was found.
        phi : float
            Value of `phi` at the new point `alpha`.
        phi0 : float
            Value of `phi` at `alpha=0`.
        task : bytes
            On exit task indicates status information.

           If task[:4] == b'CONV' then the search is successful.

           If task[:4] == b'WARN' then the subroutine is not able
           to satisfy the convergence conditions. The exit value of
           stp contains the best point found during the search.

           If task[:5] == b'ERROR' then there is an error in the
           input arguments.
        """
        if phi0 is None:
            phi0 = self.phi(0.0)
        if derphi0 is None:
            derphi0 = self.derphi(0.0)

        phi1 = phi0
        derphi1 = derphi0

        task = b"START"
        for i in range(maxiter):
            stp, phi1, derphi1, task = self._iterate(
                alpha1, phi1, derphi1, task
            )

            if not np.isfinite(stp):
                task = b"WARN"
                stp = None
                break

            if task[:2] == b"FG":
                alpha1 = stp
                phi1 = self.phi(stp)
                derphi1 = self.derphi(stp)
            else:
                break
        else:
            # maxiter reached, the line search did not converge
            stp = None
            task = b"WARNING: dcsrch did not converge within max iterations"

        if task[:5] == b"ERROR" or task[:4] == b"WARN":
            stp = None  # failed

        return stp, phi1, phi0, task

    def _iterate(self, stp, f, g, task):
        """
        Parameters
        ----------
        stp : float
            The current estimate of a satisfactory step. On initial entry, a
            positive initial estimate must be provided.
        f : float
            On first call f is the value of the function at 0. On subsequent
            entries f should be the value of the function at stp.
        g : float
            On initial entry g is the derivative of the function at 0. On
            subsequent entries g is the derivative of the function at stp.
        task : bytes
            On initial entry task must be set to 'START'.

        On exit with convergence, a warning or an error, the
           variable task contains additional information.


        Returns
        -------
        stp, f, g, task: tuple

            stp : float
                the current estimate of a satisfactory step if task = 'FG'. If
                task = 'CONV' then stp satisfies the sufficient decrease and
                curvature condition.
            f : float
                the value of the function at stp.
            g : float
                the derivative of the function at stp.
            task : bytes
                On exit task indicates the required action:

               If task(1:2) == b'FG' then evaluate the function and
               derivative at stp and call dcsrch again.

               If task(1:4) == b'CONV' then the search is successful.

               If task(1:4) == b'WARN' then the subroutine is not able
               to satisfy the convergence conditions. The exit value of
               stp contains the best point found during the search.

              If task(1:5) == b'ERROR' then there is an error in the
              input arguments.
        """
        p5 = 0.5
        p66 = 0.66
        xtrapl = 1.1
        xtrapu = 4.0

        if task[:5] == b"START":
            if stp < self.stpmin:
                task = b"ERROR: STP .LT. STPMIN"
            if stp > self.stpmax:
                task = b"ERROR: STP .GT. STPMAX"
            if g >= 0:
                task = b"ERROR: INITIAL G .GE. ZERO"
            if self.ftol < 0:
                task = b"ERROR: FTOL .LT. ZERO"
            if self.gtol < 0:
                task = b"ERROR: GTOL .LT. ZERO"
            if self.xtol < 0:
                task = b"ERROR: XTOL .LT. ZERO"
            if self.stpmin < 0:
                task = b"ERROR: STPMIN .LT. ZERO"
            if self.stpmax < self.stpmin:
                task = b"ERROR: STPMAX .LT. STPMIN"

            if task[:5] == b"ERROR":
                return stp, f, g, task

            # Initialize local variables.

            self.brackt = False
            self.stage = 1
            self.finit = f
            self.ginit = g
            self.gtest = self.ftol * self.ginit
            self.width = self.stpmax - self.stpmin
            self.width1 = self.width / p5

            # The variables stx, fx, gx contain the values of the step,
            # function, and derivative at the best step.
            # The variables sty, fy, gy contain the value of the step,
            # function, and derivative at sty.
            # The variables stp, f, g contain the values of the step,
            # function, and derivative at stp.

            self.stx = 0.0
            self.fx = self.finit
            self.gx = self.ginit
            self.sty = 0.0
            self.fy = self.finit
            self.gy = self.ginit
            self.stmin = 0
            self.stmax = stp + xtrapu * stp
            task = b"FG"
            return stp, f, g, task

        # in the original Fortran this was a location to restore variables
        # we don't need to do that because they're attributes.

        # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
        # algorithm enters the second stage.
        ftest = self.finit + stp * self.gtest

        if self.stage == 1 and f <= ftest and g >= 0:
            self.stage = 2

        # test for warnings
        if self.brackt and (stp <= self.stmin or stp >= self.stmax):
            task = b"WARNING: ROUNDING ERRORS PREVENT PROGRESS"
        if self.brackt and self.stmax - self.stmin <= self.xtol * self.stmax:
            task = b"WARNING: XTOL TEST SATISFIED"
        if stp == self.stpmax and f <= ftest and g <= self.gtest:
            task = b"WARNING: STP = STPMAX"
        if stp == self.stpmin and (f > ftest or g >= self.gtest):
            task = b"WARNING: STP = STPMIN"

        # test for convergence
        if f <= ftest and abs(g) <= self.gtol * -self.ginit:
            task = b"CONVERGENCE"

        # test for termination
        if task[:4] == b"WARN" or task[:4] == b"CONV":
            return stp, f, g, task

        # A modified function is used to predict the step during the
        # first stage if a lower function value has been obtained but
        # the decrease is not sufficient.
        if self.stage == 1 and f <= self.fx and f > ftest:
            # Define the modified function and derivative values.
            fm = f - stp * self.gtest
            fxm = self.fx - self.stx * self.gtest
            fym = self.fy - self.sty * self.gtest
            gm = g - self.gtest
            gxm = self.gx - self.gtest
            gym = self.gy - self.gtest

            # Call dcstep to update stx, sty, and to compute the new step.
            # dcstep can have several operations which can produce NaN
            # e.g. inf/inf. Filter these out.
            with np.errstate(invalid="ignore", over="ignore"):
                tup = dcstep(
                    self.stx,
                    fxm,
                    gxm,
                    self.sty,
                    fym,
                    gym,
                    stp,
                    fm,
                    gm,
                    self.brackt,
                    self.stmin,
                    self.stmax,
                )
                self.stx, fxm, gxm, self.sty, fym, gym, stp, self.brackt = tup

            # Reset the function and derivative values for f
            self.fx = fxm + self.stx * self.gtest
            self.fy = fym + self.sty * self.gtest
            self.gx = gxm + self.gtest
            self.gy = gym + self.gtest

        else:
            # Call dcstep to update stx, sty, and to compute the new step.
            # dcstep can have several operations which can produce NaN
            # e.g. inf/inf. Filter these out.

            with np.errstate(invalid="ignore", over="ignore"):
                tup = dcstep(
                    self.stx,
                    self.fx,
                    self.gx,
                    self.sty,
                    self.fy,
                    self.gy,
                    stp,
                    f,
                    g,
                    self.brackt,
                    self.stmin,
                    self.stmax,
                )
            (
                self.stx,
                self.fx,
                self.gx,
                self.sty,
                self.fy,
                self.gy,
                stp,
                self.brackt,
            ) = tup

        # Decide if a bisection step is needed
        if self.brackt:
            if abs(self.sty - self.stx) >= p66 * self.width1:
                stp = self.stx + p5 * (self.sty - self.stx)
            self.width1 = self.width
            self.width = abs(self.sty - self.stx)

        # Set the minimum and maximum steps allowed for stp.
        if self.brackt:
            self.stmin = min(self.stx, self.sty)
            self.stmax = max(self.stx, self.sty)
        else:
            self.stmin = stp + xtrapl * (stp - self.stx)
            self.stmax = stp + xtrapu * (stp - self.stx)

        # Force the step to be within the bounds stpmax and stpmin.
        stp = np.clip(stp, self.stpmin, self.stpmax)

        # If further progress is not possible, let stp be the best
        # point obtained during the search.
        if (
            self.brackt
            and (stp <= self.stmin or stp >= self.stmax)
            or (
                self.brackt
                and self.stmax - self.stmin <= self.xtol * self.stmax
            )
        ):
            stp = self.stx

        # Obtain another function and derivative
        task = b"FG"
        return stp, f, g, task


def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """
    Subroutine dcstep

    This subroutine computes a safeguarded step for a search
    procedure and updates an interval that contains a step that
    satisfies a sufficient decrease and a curvature condition.

    The parameter stx contains the step with the least function
    value. If brackt is set to .true. then a minimizer has
    been bracketed in an interval with endpoints stx and sty.
    The parameter stp contains the current step.
    The subroutine assumes that if brackt is set to .true. then

        min(stx,sty) < stp < max(stx,sty),

    and that the derivative at stx is negative in the direction
    of the step.

    The subroutine statement is

      subroutine dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
                        stpmin,stpmax)

    where

    stx is a double precision variable.
        On entry stx is the best step obtained so far and is an
          endpoint of the interval that contains the minimizer.
        On exit stx is the updated best step.

    fx is a double precision variable.
        On entry fx is the function at stx.
        On exit fx is the function at stx.

    dx is a double precision variable.
        On entry dx is the derivative of the function at
          stx. The derivative must be negative in the direction of
          the step, that is, dx and stp - stx must have opposite
          signs.
        On exit dx is the derivative of the function at stx.

    sty is a double precision variable.
        On entry sty is the second endpoint of the interval that
          contains the minimizer.
        On exit sty is the updated endpoint of the interval that
          contains the minimizer.

    fy is a double precision variable.
        On entry fy is the function at sty.
        On exit fy is the function at sty.

    dy is a double precision variable.
        On entry dy is the derivative of the function at sty.
        On exit dy is the derivative of the function at the exit sty.

    stp is a double precision variable.
        On entry stp is the current step. If brackt is set to .true.
          then on input stp must be between stx and sty.
        On exit stp is a new trial step.

    fp is a double precision variable.
        On entry fp is the function at stp
        On exit fp is unchanged.

    dp is a double precision variable.
        On entry dp is the derivative of the function at stp.
        On exit dp is unchanged.

    brackt is an logical variable.
        On entry brackt specifies if a minimizer has been bracketed.
            Initially brackt must be set to .false.
        On exit brackt specifies if a minimizer has been bracketed.
            When a minimizer is bracketed brackt is set to .true.

    stpmin is a double precision variable.
        On entry stpmin is a lower bound for the step.
        On exit stpmin is unchanged.

    stpmax is a double precision variable.
        On entry stpmax is an upper bound for the step.
        On exit stpmax is unchanged.

    MINPACK-1 Project. June 1983
    Argonne National Laboratory.
    Jorge J. More' and David J. Thuente.

    MINPACK-2 Project. November 1993.
    Argonne National Laboratory and University of Minnesota.
    Brett M. Averick and Jorge J. More'.

    """
    sgn_dp = np.sign(dp)
    sgn_dx = np.sign(dx)

    # sgnd = dp * (dx / abs(dx))
    sgnd = sgn_dp * sgn_dx

    # First case: A higher function value. The minimum is bracketed.
    # If the cubic step is closer to stx than the quadratic step, the
    # cubic step is taken, otherwise the average of the cubic and
    # quadratic steps is taken.
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma *= -1
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        if abs(stpc - stx) <= abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        # Second case: A lower function value and derivatives of opposite
        # sign. The minimum is bracketed. If the cubic step is farther from
        # stp than the secant step, the cubic step is taken, otherwise the
        # secant step is taken.
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma *= -1
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        # Third case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative decreases.

        # The cubic step is computed only if the cubic tends to infinity
        # in the direction of the step or if the minimum of the cubic
        # is beyond stp. Otherwise the cubic step is defined to be the
        # secant step.
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        # The case gamma = 0 only arises if the cubic does not tend
        # to infinity in the direction of the step.
        gamma = s * np.sqrt(max(0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < 0 and gamma != 0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            # A minimizer has been bracketed. If the cubic step is
            # closer to stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            # A minimizer has not been bracketed. If the cubic step is
            # farther from stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = np.clip(stpf, stpmin, stpmax)

    else:
        # Fourth case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative does not decrease. If the
        # minimum is not bracketed, the step is either stpmin or stpmax,
        # otherwise the cubic step is taken.
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval which contains a minimizer.
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    # Compute the new step.
    stp = stpf

    return stx, fx, dx, sty, fy, dy, stp, brackt
<|endoftext|>"""Trust-region optimization."""
import math
import warnings

import numpy as np
import scipy.linalg
from ._optimize import (_check_unknown_options, _status_message,
                        OptimizeResult, _prepare_scalar_function,
                        _call_callback_maybe_halt)
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.optimize._differentiable_functions import FD_METHODS
__all__ = []


def _wrap_function(function, args):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        return function(np.copy(x), *(wrapper_args + args))

    return ncalls, function_wrapper


class BaseQuadraticSubproblem:
    """
    Base/abstract class defining the quadratic model for trust-region
    minimization. Child classes must implement the ``solve`` method.

    Values of the objective function, Jacobian and Hessian (if provided) at
    the current iterate ``x`` are evaluated on demand and then stored as
    attributes ``fun``, ``jac``, ``hess``.
    """

    def __init__(self, x, fun, jac, hess=None, hessp=None):
        self._x = x
        self._f = None
        self._g = None
        self._h = None
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None
        self._fun = fun
        self._jac = jac
        self._hess = hess
        self._hessp = hessp

    def __call__(self, p):
        return self.fun + np.dot(self.jac, p) + 0.5 * np.dot(p, self.hessp(p))

    @property
    def fun(self):
        """Value of objective function at current iteration."""
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def jac(self):
        """Value of Jacobian of objective function at current iteration."""
        if self._g is None:
            self._g = self._jac(self._x)
        return self._g

    @property
    def hess(self):
        """Value of Hessian of objective function at current iteration."""
        if self._h is None:
            self._h = self._hess(self._x)
        return self._h

    def hessp(self, p):
        if self._hessp is not None:
            return self._hessp(self._x, p)
        else:
            return np.dot(self.hess, p)

    @property
    def jac_mag(self):
        """Magnitude of jacobian of objective function at current iteration."""
        if self._g_mag is None:
            self._g_mag = scipy.linalg.norm(self.jac)
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):
        """
        Solve the scalar quadratic equation ``||z + t d|| == trust_radius``.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        a = np.dot(d, d)
        b = 2 * np.dot(z, d)
        c = np.dot(z, z) - trust_radius**2
        sqrt_discriminant = math.sqrt(b*b - 4*a*c)

        # The following calculation is mathematically
        # equivalent to:
        # ta = (-b - sqrt_discriminant) / (2*a)
        # tb = (-b + sqrt_discriminant) / (2*a)
        # but produce smaller round off errors.
        # Look at Matrix Computation p.97
        # for a better justification.
        aux = b + math.copysign(sqrt_discriminant, b)
        ta = -aux / (2*a)
        tb = -2*c / aux
        return sorted([ta, tb])

    def solve(self, trust_radius):
        raise NotImplementedError('The solve method should be implemented by '
                                  'the child class')


def _minimize_trust_region(fun, x0, args=(), jac=None, hess=None, hessp=None,
                           subproblem=None, initial_trust_radius=1.0,
                           max_trust_radius=1000.0, eta=0.15, gtol=1e-4,
                           maxiter=None, disp=False, return_all=False,
                           callback=None, inexact=True, **unknown_options):
    """
    Minimization of scalar function of one or more variables using a
    trust-region algorithm.

    Options for the trust-region algorithm are:
        initial_trust_radius : float
            Initial trust radius.
        max_trust_radius : float
            Never propose steps that are longer than this value.
        eta : float
            Trust region related acceptance stringency for proposed steps.
        gtol : float
            Gradient norm must be less than `gtol`
            before successful termination.
        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.
        inexact : bool
            Accuracy to solve subproblems. If True requires less nonlinear
            iterations, but more vector products. Only effective for method
            trust-krylov.

    This function is called by the `minimize` function.
    It is not supposed to be called directly.
    """
    _check_unknown_options(unknown_options)

    if jac is None:
        raise ValueError('Jacobian is currently required for trust-region '
                         'methods')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is currently required for trust-region methods')
    if subproblem is None:
        raise ValueError('A subproblem solving strategy is required for '
                         'trust-region methods')
    if not (0 <= eta < 0.25):
        raise Exception('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise Exception('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the '
                         'max trust radius')

    # force the initial guess into a nice format
    x0 = np.asarray(x0).flatten()

    # A ScalarFunction representing the problem. This caches calls to fun, jac,
    # hess.
    sf = _prepare_scalar_function(fun, x0, jac=jac, hess=hess, args=args)
    fun = sf.fun
    jac = sf.grad
    if callable(hess):
        hess = sf.hess
    elif callable(hessp):
        # this elif statement must come before examining whether hess
        # is estimated by FD methods or a HessianUpdateStrategy
        pass
    elif (hess in FD_METHODS or isinstance(hess, HessianUpdateStrategy)):
        # If the Hessian is being estimated by finite differences or a
        # Hessian update strategy then ScalarFunction.hess returns a
        # LinearOperator or a HessianUpdateStrategy. This enables the
        # calculation/creation of a hessp. BUT you only want to do this
        # if the user *hasn't* provided a callable(hessp) function.
        hess = None

        def hessp(x, p, *args):
            return sf.hess(x).dot(p)
    else:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is currently required for trust-region methods')

    # ScalarFunction doesn't represent hessp
    nhessp, hessp = _wrap_function(hessp, args)

    # limit the number of iterations
    if maxiter is None:
        maxiter = len(x0)*200

    # init the search status
    warnflag = 0

    # initialize the search
    trust_radius = initial_trust_radius
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)
    k = 0

    # search for the function min
    # do not even start if the gradient is small enough
    while m.jac_mag >= gtol:

        # Solve the sub-problem.
        # This gives us the proposed step relative to the current position
        # and it tells us whether the proposed step
        # has reached the trust region boundary or not.
        try:
            p, hits_boundary = m.solve(trust_radius)
        except np.linalg.LinAlgError:
            warnflag = 3
            break

        # calculate the predicted value at the proposed point
        predicted_value = m(p)

        # define the local approximation at the proposed point
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)

        # evaluate the ratio defined in equation (4.4)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction

        # update the trust radius according to the actual/predicted ratio
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2*trust_radius, max_trust_radius)

        # if the ratio is high enough then accept the proposed step
        if rho > eta:
            x = x_proposed
            m = m_proposed

        # append the best guess, call back, increment the iteration count
        if return_all:
            allvecs.append(np.copy(x))
        k += 1

        intermediate_result = OptimizeResult(x=x, fun=m.fun)
        if _call_callback_maybe_halt(callback, intermediate_result):
            break

        # check if the gradient is small enough to stop
        if m.jac_mag < gtol:
            warnflag = 0
            break

        # check if we have looked at enough iterations
        if k >= maxiter:
            warnflag = 1
            break

    # print some stuff if requested
    status_messages = (
            _status_message['success'],
            _status_message['maxiter'],
            'A bad approximation caused failure to predict improvement.',
            'A linalg error occurred, such as a non-psd Hessian.',
            )
    if disp:
        if warnflag == 0:
            print(status_messages[warnflag])
        else:
            warnings.warn(status_messages[warnflag], RuntimeWarning, stacklevel=3)
        print(f"         Current function value: {m.fun:f}")
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Hessian evaluations: %d" % (sf.nhev + nhessp[0]))

    result = OptimizeResult(x=x, success=(warnflag == 0), status=warnflag,
                            fun=m.fun, jac=m.jac, nfev=sf.nfev, njev=sf.ngev,
                            nhev=sf.nhev + nhessp[0], nit=k,
                            message=status_messages[warnflag])

    if hess is not None:
        result['hess'] = m.hess

    if return_all:
        result['allvecs'] = allvecs

    return result
<|endoftext|>import math
import numpy as np
import scipy._lib._elementwise_iterative_method as eim
from scipy._lib._util import _RichResult
from scipy._lib._array_api import xp_sign, xp_copy, xp_take_along_axis

# TODO:
# - (maybe?) don't use fancy indexing assignment
# - figure out how to replace the new `try`/`except`s


def _chandrupatla(func, a, b, *, args=(), xatol=None, xrtol=None,
                  fatol=None, frtol=0, maxiter=None, callback=None):
    """Find the root of an elementwise function using Chandrupatla's algorithm.

    For each element of the output of `func`, `chandrupatla` seeks the scalar
    root that makes the element 0. This function allows for `a`, `b`, and the
    output of `func` to be of any broadcastable shapes.

    Parameters
    ----------
    func : callable
        The function whose root is desired. The signature must be::

            func(x: ndarray, *args) -> ndarray

         where each element of ``x`` is a finite real and ``args`` is a tuple,
         which may contain an arbitrary number of components of any type(s).
         ``func`` must be an elementwise function: each element ``func(x)[i]``
         must equal ``func(x[i])`` for all indices ``i``. `_chandrupatla`
         seeks an array ``x`` such that ``func(x)`` is an array of zeros.
    a, b : array_like
        The lower and upper bounds of the root of the function. Must be
        broadcastable with one another.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.
    xatol, xrtol, fatol, frtol : float, optional
        Absolute and relative tolerances on the root and function value.
        See Notes for details.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.
        The default is the maximum possible number of bisections within
        the (normal) floating point numbers of the relevant dtype.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is a ``_RichResult``
        similar to that returned by `_chandrupatla` (but containing the current
        iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `_chandrupatla` will return a result.

    Returns
    -------
    res : _RichResult
        An instance of `scipy._lib._util._RichResult` with the following
        attributes. The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.

        x : float
            The root of the function, if the algorithm terminated successfully.
        nfev : int
            The number of times the function was called to find the root.
        nit : int
            The number of iterations of Chandrupatla's algorithm performed.
        status : int
            An integer representing the exit status of the algorithm.
            ``0`` : The algorithm converged to the specified tolerances.
            ``-1`` : The algorithm encountered an invalid bracket.
            ``-2`` : The maximum number of iterations was reached.
            ``-3`` : A non-finite value was encountered.
            ``-4`` : Iteration was terminated by `callback`.
            ``1`` : The algorithm is proceeding normally (in `callback` only).
        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).
        fun : float
            The value of `func` evaluated at `x`.
        xl, xr : float
            The lower and upper ends of the bracket.
        fl, fr : float
            The function value at the lower and upper ends of the bracket.

    Notes
    -----
    Implemented based on Chandrupatla's original paper [1]_.

    If ``xl`` and ``xr`` are the left and right ends of the bracket,
    ``xmin = xl if abs(func(xl)) <= abs(func(xr)) else xr``,
    and ``fmin0 = min(func(a), func(b))``, then the algorithm is considered to
    have converged when ``abs(xr - xl) < xatol + abs(xmin) * xrtol`` or
    ``fun(xmin) <= fatol + abs(fmin0) * frtol``. This is equivalent to the
    termination condition described in [1]_ with ``xrtol = 4e-10``,
    ``xatol = 1e-5``, and ``fatol = frtol = 0``. The default values are
    ``xatol = 4*tiny``, ``xrtol = 4*eps``, ``frtol = 0``, and ``fatol = tiny``,
    where ``eps`` and ``tiny`` are the precision and smallest normal number
    of the result ``dtype`` of function inputs and outputs.

    References
    ----------

    .. [1] Chandrupatla, Tirupathi R.
        "A new hybrid quadratic/bisection algorithm for finding the zero of a
        nonlinear function without using derivatives".
        Advances in Engineering Software, 28(3), 145-149.
        https://doi.org/10.1016/s0965-9978(96)00051-8

    See Also
    --------
    brentq, brenth, ridder, bisect, newton

    Examples
    --------
    >>> from scipy import optimize
    >>> def f(x, c):
    ...     return x**3 - 2*x - c
    >>> c = 5
    >>> res = optimize._chandrupatla._chandrupatla(f, 0, 3, args=(c,))
    >>> res.x
    2.0945514818937463

    >>> c = [3, 4, 5]
    >>> res = optimize._chandrupatla._chandrupatla(f, 0, 3, args=(c,))
    >>> res.x
    array([1.8932892 , 2.        , 2.09455148])

    """
    res = _chandrupatla_iv(func, args, xatol, xrtol,
                           fatol, frtol, maxiter, callback)
    func, args, xatol, xrtol, fatol, frtol, maxiter, callback = res

    # Initialization
    temp = eim._initialize(func, (a, b), args)
    func, xs, fs, args, shape, dtype, xp = temp
    x1, x2 = xs
    f1, f2 = fs
    status = xp.full_like(x1, xp.asarray(eim._EINPROGRESS),
                          dtype=xp.int32)  # in progress
    nit, nfev = 0, 2  # two function evaluations performed above
    finfo = xp.finfo(dtype)
    xatol = 4*finfo.smallest_normal if xatol is None else xatol
    xrtol = 4*finfo.eps if xrtol is None else xrtol
    fatol = finfo.smallest_normal if fatol is None else fatol
    frtol = frtol * xp.minimum(xp.abs(f1), xp.abs(f2))
    maxiter = (math.log2(finfo.max) - math.log2(finfo.smallest_normal)
               if maxiter is None else maxiter)
    work = _RichResult(x1=x1, f1=f1, x2=x2, f2=f2, x3=None, f3=None, t=0.5,
                       xatol=xatol, xrtol=xrtol, fatol=fatol, frtol=frtol,
                       nit=nit, nfev=nfev, status=status)
    res_work_pairs = [('status', 'status'), ('x', 'xmin'), ('fun', 'fmin'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('xl', 'x1'),
                      ('fl', 'f1'), ('xr', 'x2'), ('fr', 'f2')]

    def pre_func_eval(work):
        # [1] Figure 1 (first box)
        x = work.x1 + work.t * (work.x2 - work.x1)
        return x

    def post_func_eval(x, f, work):
        # [1] Figure 1 (first diamond and boxes)
        # Note: y/n are reversed in figure; compare to BASIC in appendix
        work.x3, work.f3 = (xp.asarray(work.x2, copy=True),
                            xp.asarray(work.f2, copy=True))
        j = xp.sign(f) == xp.sign(work.f1)
        nj = ~j
        work.x3[j], work.f3[j] = work.x1[j], work.f1[j]
        work.x2[nj], work.f2[nj] = work.x1[nj], work.f1[nj]
        work.x1, work.f1 = x, f

    def check_termination(work):
        # [1] Figure 1 (second diamond)
        # Check for all terminal conditions and record statuses.

        # See [1] Section 4 (first two sentences)
        i = xp.abs(work.f1) < xp.abs(work.f2)
        work.xmin = xp.where(i, work.x1, work.x2)
        work.fmin = xp.where(i, work.f1, work.f2)
        stop = xp.zeros_like(work.x1, dtype=xp.bool)  # termination condition met

        # If function value tolerance is met, report successful convergence,
        # regardless of other conditions. Note that `frtol` has been redefined
        # as `frtol = frtol * minimum(f1, f2)`, where `f1` and `f2` are the
        # function evaluated at the original ends of the bracket.
        i = xp.abs(work.fmin) <= work.fatol + work.frtol
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        # If the bracket is no longer valid, report failure (unless a function
        # tolerance is met, as detected above).
        i = (xp_sign(work.f1) == xp_sign(work.f2)) & ~stop
        NaN = xp.asarray(xp.nan, dtype=work.xmin.dtype)
        work.xmin[i], work.fmin[i], work.status[i] = NaN, NaN, eim._ESIGNERR
        stop[i] = True

        # If the abscissae are non-finite or either function value is NaN,
        # report failure.
        x_nonfinite = ~(xp.isfinite(work.x1) & xp.isfinite(work.x2))
        f_nan = xp.isnan(work.f1) & xp.isnan(work.f2)
        i = (x_nonfinite | f_nan) & ~stop
        work.xmin[i], work.fmin[i], work.status[i] = NaN, NaN, eim._EVALUEERR
        stop[i] = True

        # This is the convergence criterion used in bisect. Chandrupatla's
        # criterion is equivalent to this except with a factor of 4 on `xrtol`.
        work.dx = xp.abs(work.x2 - work.x1)
        work.tol = xp.abs(work.xmin) * work.xrtol + work.xatol
        i = work.dx < work.tol
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        return stop

    def post_termination_check(work):
        # [1] Figure 1 (third diamond and boxes / Equation 1)
        xi1 = (work.x1 - work.x2) / (work.x3 - work.x2)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi1 = (work.f1 - work.f2) / (work.f3 - work.f2)
        alpha = (work.x3 - work.x1) / (work.x2 - work.x1)
        j = ((1 - xp.sqrt(1 - xi1)) < phi1) & (phi1 < xp.sqrt(xi1))

        f1j, f2j, f3j, alphaj = work.f1[j], work.f2[j], work.f3[j], alpha[j]
        t = xp.full_like(alpha, xp.asarray(0.5))
        t[j] = (f1j / (f1j - f2j) * f3j / (f3j - f2j)
                - alphaj * f1j / (f3j - f1j) * f2j / (f2j - f3j))

        # [1] Figure 1 (last box; see also BASIC in appendix with comment
        # "Adjust T Away from the Interval Boundary")
        tl = 0.5 * work.tol / work.dx
        work.t = xp.clip(t, tl, 1 - tl)

    def customize_result(res, shape):
        xl, xr, fl, fr = res['xl'], res['xr'], res['fl'], res['fr']
        i = res['xl'] < res['xr']
        res['xl'] = xp.where(i, xl, xr)
        res['xr'] = xp.where(i, xr, xl)
        res['fl'] = xp.where(i, fl, fr)
        res['fr'] = xp.where(i, fr, fl)
        return shape

    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval, check_termination,
                     post_termination_check, customize_result, res_work_pairs,
                     xp=xp)


def _chandrupatla_iv(func, args, xatol, xrtol,
                     fatol, frtol, maxiter, callback):
    # Input validation for `_chandrupatla`

    if not callable(func):
        raise ValueError('`func` must be callable.')

    if not np.iterable(args):
        args = (args,)

    # tolerances are floats, not arrays; OK to use NumPy
    tols = np.asarray([xatol if xatol is not None else 1,
                       xrtol if xrtol is not None else 1,
                       fatol if fatol is not None else 1,
                       frtol if frtol is not None else 1])
    if (not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0)
            or np.any(np.isnan(tols)) or tols.shape != (4,)):
        raise ValueError('Tolerances must be non-negative scalars.')

    if maxiter is not None:
        maxiter_int = int(maxiter)
        if maxiter != maxiter_int or maxiter < 0:
            raise ValueError('`maxiter` must be a non-negative integer.')

    if callback is not None and not callable(callback):
        raise ValueError('`callback` must be callable.')

    return func, args, xatol, xrtol, fatol, frtol, maxiter, callback


def _chandrupatla_minimize(func, x1, x2, x3, *, args=(), xatol=None,
                           xrtol=None, fatol=None, frtol=None, maxiter=100,
                           callback=None):
    """Find the minimizer of an elementwise function.

    For each element of the output of `func`, `_chandrupatla_minimize` seeks
    the scalar minimizer that minimizes the element. This function allows for
    `x1`, `x2`, `x3`, and the elements of `args` to be arrays of any
    broadcastable shapes.

    Parameters
    ----------
    func : callable
        The function whose minimizer is desired. The signature must be::

            func(x: ndarray, *args) -> ndarray

         where each element of ``x`` is a finite real and ``args`` is a tuple,
         which may contain an arbitrary number of arrays that are broadcastable
         with `x`. ``func`` must be an elementwise function: each element
         ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
         `_chandrupatla` seeks an array ``x`` such that ``func(x)`` is an array
         of minima.
    x1, x2, x3 : array_like
        The abscissae of a standard scalar minimization bracket. A bracket is
        valid if ``x1 < x2 < x3`` and ``func(x1) > func(x2) <= func(x3)``.
        Must be broadcastable with one another and `args`.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.  Must be arrays
        broadcastable with `x1`, `x2`, and `x3`. If the callable to be
        differentiated requires arguments that are not broadcastable with `x`,
        wrap that callable with `func` such that `func` accepts only `x` and
        broadcastable arrays.
    xatol, xrtol, fatol, frtol : float, optional
        Absolute and relative tolerances on the minimizer and function value.
        See Notes for details.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.
    callback : callable, optional
        An optional user-supplied function to be called before the first
        iteration and after each iteration.
        Called as ``callback(res)``, where ``res`` is a ``_RichResult``
        similar to that returned by `_chandrupatla_minimize` (but containing
        the current iterate's values of all variables). If `callback` raises a
        ``StopIteration``, the algorithm will terminate immediately and
        `_chandrupatla_minimize` will return a result.

    Returns
    -------
    res : _RichResult
        An instance of `scipy._lib._util._RichResult` with the following
        attributes. (The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.)

        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).
        status : int
            An integer representing the exit status of the algorithm.
            ``0`` : The algorithm converged to the specified tolerances.
            ``-1`` : The algorithm encountered an invalid bracket.
            ``-2`` : The maximum number of iterations was reached.
            ``-3`` : A non-finite value was encountered.
            ``-4`` : Iteration was terminated by `callback`.
            ``1`` : The algorithm is proceeding normally (in `callback` only).
        x : float
            The minimizer of the function, if the algorithm terminated
            successfully.
        fun : float
            The value of `func` evaluated at `x`.
        nfev : int
            The number of points at which `func` was evaluated.
        nit : int
            The number of iterations of the algorithm that were performed.
        xl, xm, xr : float
            The final three-point bracket.
        fl, fm, fr : float
            The function value at the bracket points.

    Notes
    -----
    Implemented based on Chandrupatla's original paper [1]_.

    If ``x1 < x2 < x3`` are the points of the bracket and ``f1 > f2 <= f3``
    are the values of ``func`` at those points, then the algorithm is
    considered to have converged when ``x3 - x1 <= abs(x2)*xrtol + xatol``
    or ``(f1 - 2*f2 + f3)/2 <= abs(f2)*frtol + fatol``. Note that first of
    these differs from the termination conditions described in [1]_. The
    default values of `xrtol` is the square root of the precision of the
    appropriate dtype, and ``xatol = fatol = frtol`` is the smallest normal
    number of the appropriate dtype.

    References
    ----------
    .. [1] Chandrupatla, Tirupathi R. (1998).
        "An efficient quadratic fit-sectioning algorithm for minimization
        without derivatives".
        Computer Methods in Applied Mechanics and Engineering, 152 (1-2),
        211-217. https://doi.org/10.1016/S0045-7825(97)00190-4

    See Also
    --------
    golden, brent, bounded

    Examples
    --------
    >>> from scipy.optimize._chandrupatla import _chandrupatla_minimize
    >>> def f(x, args=1):
    ...     return (x - args)**2
    >>> res = _chandrupatla_minimize(f, -5, 0, 5)
    >>> res.x
    1.0
    >>> c = [1, 1.5, 2]
    >>> res = _chandrupatla_minimize(f, -5, 0, 5, args=(c,))
    >>> res.x
    array([1. , 1.5, 2. ])
    """
    res = _chandrupatla_iv(func, args, xatol, xrtol,
                           fatol, frtol, maxiter, callback)
    func, args, xatol, xrtol, fatol, frtol, maxiter, callback = res

    # Initialization
    xs = (x1, x2, x3)
    temp = eim._initialize(func, xs, args)
    func, xs, fs, args, shape, dtype, xp = temp  # line split for PEP8
    x1, x2, x3 = xs
    f1, f2, f3 = fs
    phi = xp.asarray(0.5 + 0.5*5**0.5, dtype=dtype)[()]  # golden ratio
    status = xp.full_like(x1, xp.asarray(eim._EINPROGRESS),
                          dtype=xp.int32)  # in progress
    nit, nfev = 0, 3  # three function evaluations performed above
    fatol = xp.finfo(dtype).smallest_normal if fatol is None else fatol
    frtol = xp.finfo(dtype).smallest_normal if frtol is None else frtol
    xatol = xp.finfo(dtype).smallest_normal if xatol is None else xatol
    xrtol = math.sqrt(xp.finfo(dtype).eps) if xrtol is None else xrtol

    # Ensure that x1 < x2 < x3 initially.
    xs, fs = xp.stack((x1, x2, x3)), xp.stack((f1, f2, f3))
    i = xp.argsort(xs, axis=0)
    x1, x2, x3 = xp_take_along_axis(xs, i, axis=0)  # data-apis/array-api#808
    f1, f2, f3 = xp_take_along_axis(fs, i, axis=0)  # data-apis/array-api#808
    q0 = xp_copy(x3)  # "At the start, q0 is set at x3..." ([1] after (7))

    work = _RichResult(x1=x1, f1=f1, x2=x2, f2=f2, x3=x3, f3=f3, phi=phi,
                       xatol=xatol, xrtol=xrtol, fatol=fatol, frtol=frtol,
                       nit=nit, nfev=nfev, status=status, q0=q0, args=args)
    res_work_pairs = [('status', 'status'),
                      ('x', 'x2'), ('fun', 'f2'),
                      ('nit', 'nit'), ('nfev', 'nfev'),
                      ('xl', 'x1'), ('xm', 'x2'), ('xr', 'x3'),
                      ('fl', 'f1'), ('fm', 'f2'), ('fr', 'f3')]

    def pre_func_eval(work):
        # `_check_termination` is called first -> `x3 - x2 > x2 - x1`
        # But let's calculate a few terms that we'll reuse
        x21 = work.x2 - work.x1
        x32 = work.x3 - work.x2

        # [1] Section 3. "The quadratic minimum point Q1 is calculated using
        # the relations developed in the previous section." [1] Section 2 (5/6)
        A = x21 * (work.f3 - work.f2)
        B = x32 * (work.f1 - work.f2)
        C = A / (A + B)
        # q1 = C * (work.x1 + work.x2) / 2 + (1 - C) * (work.x2 + work.x3) / 2
        q1 = 0.5 * (C*(work.x1 - work.x3) + work.x2 + work.x3)  # much faster
        # this is an array, so multiplying by 0.5 does not change dtype

        # "If Q1 and Q0 are sufficiently close... Q1 is accepted if it is
        # sufficiently away from the inside point x2"
        i = xp.abs(q1 - work.q0) < 0.5 * xp.abs(x21)  # [1] (7)
        xi = q1[i]
        # Later, after (9), "If the point Q1 is in a +/- xtol neighborhood of
        # x2, the new point is chosen in the larger interval at a distance
        # tol away from x2."
        # See also QBASIC code after "Accept Ql adjust if close to X2".
        j = xp.abs(q1[i] - work.x2[i]) <= work.xtol[i]
        xi[j] = work.x2[i][j] + xp_sign(x32[i][j]) * work.xtol[i][j]

        # "If condition (7) is not satisfied, golden sectioning of the larger
        # interval is carried out to introduce the new point."
        # (For simplicity, we go ahead and calculate it for all points, but we
        # change the elements for which the condition was satisfied.)
        x = work.x2 + (2 - work.phi) * x32
        x[i] = xi

        # "We define Q0 as the value of Q1 at the previous iteration."
        work.q0 = q1
        return x

    def post_func_eval(x, f, work):
        # Standard logic for updating a three-point bracket based on a new
        # point. In QBASIC code, see "IF SGN(X-X2) = SGN(X3-X2) THEN...".
        # There is an awful lot of data copying going on here; this would
        # probably benefit from code optimization or implementation in Pythran.
        i = xp_sign(x - work.x2) == xp_sign(work.x3 - work.x2)
        xi, x1i, x2i, x3i = x[i], work.x1[i], work.x2[i], work.x3[i],
        fi, f1i, f2i, f3i = f[i], work.f1[i], work.f2[i], work.f3[i]
        j = fi > f2i
        x3i[j], f3i[j] = xi[j], fi[j]
        j = ~j
        x1i[j], f1i[j], x2i[j], f2i[j] = x2i[j], f2i[j], xi[j], fi[j]

        ni = ~i
        xni, x1ni, x2ni, x3ni = x[ni], work.x1[ni], work.x2[ni], work.x3[ni],
        fni, f1ni, f2ni, f3ni = f[ni], work.f1[ni], work.f2[ni], work.f3[ni]
        j = fni > f2ni
        x1ni[j], f1ni[j] = xni[j], fni[j]
        j = ~j
        x3ni[j], f3ni[j], x2ni[j], f2ni[j] = x2ni[j], f2ni[j], xni[j], fni[j]

        work.x1[i], work.x2[i], work.x3[i] = x1i, x2i, x3i
        work.f1[i], work.f2[i], work.f3[i] = f1i, f2i, f3i
        work.x1[ni], work.x2[ni], work.x3[ni] = x1ni, x2ni, x3ni,
        work.f1[ni], work.f2[ni], work.f3[ni] = f1ni, f2ni, f3ni

    def check_termination(work):
        # Check for all terminal conditions and record statuses.
        stop = xp.zeros_like(work.x1, dtype=bool)  # termination condition met

        # Bracket is invalid; stop and don't return minimizer/minimum
        i = ((work.f2 > work.f1) | (work.f2 > work.f3))
        work.x2[i], work.f2[i] = xp.nan, xp.nan
        stop[i], work.status[i] = True, eim._ESIGNERR

        # Non-finite values; stop and don't return minimizer/minimum
        finite = xp.isfinite(work.x1+work.x2+work.x3+work.f1+work.f2+work.f3)
        i = ~(finite | stop)
        work.x2[i], work.f2[i] = xp.nan, xp.nan
        stop[i], work.status[i] = True, eim._EVALUEERR

        # [1] Section 3 "Points 1 and 3 are interchanged if necessary to make
        # the (x2, x3) the larger interval."
        # Note: I had used np.choose; this is much faster. This would be a good
        # place to save e.g. `work.x3 - work.x2` for reuse, but I tried and
        # didn't notice a speed boost, so let's keep it simple.
        i = xp.abs(work.x3 - work.x2) < xp.abs(work.x2 - work.x1)
        temp = work.x1[i]
        work.x1[i] = work.x3[i]
        work.x3[i] = temp
        temp = work.f1[i]
        work.f1[i] = work.f3[i]
        work.f3[i] = temp

        # [1] Section 3 (bottom of page 212)
        # "We set a tolerance value xtol..."
        work.xtol = xp.abs(work.x2) * work.xrtol + work.xatol  # [1] (8)
        # "The convergence based on interval is achieved when..."
        # Note: Equality allowed in case of `xtol=0`
        i = xp.abs(work.x3 - work.x2) <= 2 * work.xtol  # [1] (9)

        # "We define ftol using..."
        ftol = xp.abs(work.f2) * work.frtol + work.fatol  # [1] (10)
        # "The convergence based on function values is achieved when..."
        # Note 1: modify in place to incorporate tolerance on function value.
        # Note 2: factor of 2 is not in the text; see QBASIC start of DO loop
        i |= (work.f1 - 2 * work.f2 + work.f3) <= 2*ftol  # [1] (11)
        i &= ~stop
        stop[i], work.status[i] = True, eim._ECONVERGED

        return stop

    def post_termination_check(work):
        pass

    def customize_result(res, shape):
        xl, xr, fl, fr = res['xl'], res['xr'], res['fl'], res['fr']
        i = res['xl'] >= res['xr']
        res['xl'] = xp.where(i, xr, xl)
        res['xr'] = xp.where(i, xl, xr)
        res['fl'] = xp.where(i, fr, fl)
        res['fr'] = xp.where(i, fl, fr)
        return shape

    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval, check_termination,
                     post_termination_check, customize_result, res_work_pairs,
                     xp=xp)
<|endoftext|>r"""
Parameters used in test and benchmark methods.

Collections of test cases suitable for testing 1-D root-finders
  'original': The original benchmarking functions.
     Real-valued functions of real-valued inputs on an interval
     with a zero.
     f1, .., f3 are continuous and infinitely differentiable
     f4 has a left- and right- discontinuity at the root
     f5 has a root at 1 replacing a 1st order pole
     f6 is randomly positive on one side of the root,
     randomly negative on the other.
     f4 - f6 are not continuous at the root.

  'aps': The test problems in the 1995 paper
     TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions"
     by Alefeld, Potra and Shi. Real-valued functions of
     real-valued inputs on an interval with a zero.
     Suitable for methods which start with an enclosing interval, and
     derivatives up to 2nd order.

  'complex': Some complex-valued functions of complex-valued inputs.
     No enclosing bracket is provided.
     Suitable for methods which use one or more starting values, and
     derivatives up to 2nd order.

  The test cases are provided as a list of dictionaries. The dictionary
  keys will be a subset of:
  ["f", "fprime", "fprime2", "args", "bracket", "smoothness",
  "a", "b", "x0", "x1", "root", "ID"]
"""

# Sources:
#  [1] Alefeld, G. E. and Potra, F. A. and Shi, Yixun,
#      "Algorithm 748: Enclosing Zeros of Continuous Functions",
#      ACM Trans. Math. Softw. Volume 221(1995)
#       doi = {10.1145/210089.210111},
#  [2] Chandrupatla, Tirupathi R. "A new hybrid quadratic/bisection algorithm
#      for finding the zero of a nonlinear function without using derivatives."
#      Advances in Engineering Software 28.3 (1997): 145-149.

from random import random

import numpy as np

from scipy.optimize import _zeros_py as cc
from scipy._lib._array_api import array_namespace

# "description" refers to the original functions
description = """
f2 is a symmetric parabola, x**2 - 1
f3 is a quartic polynomial with large hump in interval
f4 is step function with a discontinuity at 1
f5 is a hyperbola with vertical asymptote at 1
f6 has random values positive to left of 1, negative to right

Of course, these are not real problems. They just test how the
'good' solvers behave in bad circumstances where bisection is
really the best. A good solver should not be much worse than
bisection in such circumstance, while being faster for smooth
monotone sorts of functions.
"""


def f1(x):
    r"""f1 is a quadratic with roots at 0 and 1"""
    return x * (x - 1.)


def f1_fp(x):
    return 2 * x - 1


def f1_fpp(x):
    return 2


def f2(x):
    r"""f2 is a symmetric parabola, x**2 - 1"""
    return x**2 - 1


def f2_fp(x):
    return 2 * x


def f2_fpp(x):
    return 2


def f3(x):
    r"""A quartic with roots at 0, 1, 2 and 3"""
    return x * (x - 1.) * (x - 2.) * (x - 3.)  # x**4 - 6x**3 + 11x**2 - 6x


def f3_fp(x):
    return 4 * x**3 - 18 * x**2 + 22 * x - 6


def f3_fpp(x):
    return 12 * x**2 - 36 * x + 22


def f4(x):
    r"""Piecewise linear, left- and right- discontinuous at x=1, the root."""
    if x > 1:
        return 1.0 + .1 * x
    if x < 1:
        return -1.0 + .1 * x
    return 0


def f5(x):
    r"""
    Hyperbola with a pole at x=1, but pole replaced with 0. Not continuous at root.
    """
    if x != 1:
        return 1.0 / (1. - x)
    return 0


# f6(x) returns random value. Without memoization, calling twice with the
# same x returns different values, hence a "random value", not a
# "function with random values"
_f6_cache = {}
def f6(x):
    v = _f6_cache.get(x, None)
    if v is None:
        if x > 1:
            v = random()
        elif x < 1:
            v = -random()
        else:
            v = 0
        _f6_cache[x] = v
    return v


# Each Original test case has
# - a function and its two derivatives,
# - additional arguments,
# - a bracket enclosing a root,
# - the order of differentiability (smoothness) on this interval
# - a starting value for methods which don't require a bracket
# - the root (inside the bracket)
# - an Identifier of the test case

_ORIGINAL_TESTS_KEYS = [
    "f", "fprime", "fprime2", "args", "bracket", "smoothness", "x0", "root", "ID"
]
_ORIGINAL_TESTS = [
    [f1, f1_fp, f1_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.01.00"],
    [f2, f2_fp, f2_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.02.00"],
    [f3, f3_fp, f3_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.03.00"],
    [f4, None, None, (), [0.5, np.sqrt(3)], -1, 0.6, 1.0, "original.04.00"],
    [f5, None, None, (), [0.5, np.sqrt(3)], -1, 0.6, 1.0, "original.05.00"],
    [f6, None, None, (), [0.5, np.sqrt(3)], -np.inf, 0.6, 1.0, "original.05.00"]
]

_ORIGINAL_TESTS_DICTS = [
    dict(zip(_ORIGINAL_TESTS_KEYS, testcase)) for testcase in _ORIGINAL_TESTS
]

#   ##################
#   "APS" test cases
#   Functions and test cases that appear in [1]


def aps01_f(x):
    r"""Straightforward sum of trigonometric function and polynomial"""
    return np.sin(x) - x / 2


def aps01_fp(x):
    return np.cos(x) - 1.0 / 2


def aps01_fpp(x):
    return -np.sin(x)


def aps02_f(x):
    r"""poles at x=n**2, 1st and 2nd derivatives at root are also close to 0"""
    ii = np.arange(1, 21)
    return -2 * np.sum((2 * ii - 5)**2 / (x - ii**2)**3)


def aps02_fp(x):
    ii = np.arange(1, 21)
    return 6 * np.sum((2 * ii - 5)**2 / (x - ii**2)**4)


def aps02_fpp(x):
    ii = np.arange(1, 21)
    return 24 * np.sum((2 * ii - 5)**2 / (x - ii**2)**5)


def aps03_f(x, a, b):
    r"""Rapidly changing at the root"""
    return a * x * np.exp(b * x)


def aps03_fp(x, a, b):
    return a * (b * x + 1) * np.exp(b * x)


def aps03_fpp(x, a, b):
    return a * (b * (b * x + 1) + b) * np.exp(b * x)


def aps04_f(x, n, a):
    r"""Medium-degree polynomial"""
    return x**n - a


def aps04_fp(x, n, a):
    return n * x**(n - 1)


def aps04_fpp(x, n, a):
    return n * (n - 1) * x**(n - 2)


def aps05_f(x):
    r"""Simple Trigonometric function"""
    return np.sin(x) - 1.0 / 2


def aps05_fp(x):
    return np.cos(x)


def aps05_fpp(x):
    return -np.sin(x)


def aps06_f(x, n):
    r"""Exponential rapidly changing from -1 to 1 at x=0"""
    return 2 * x * np.exp(-n) - 2 * np.exp(-n * x) + 1


def aps06_fp(x, n):
    return 2 * np.exp(-n) + 2 * n * np.exp(-n * x)


def aps06_fpp(x, n):
    return -2 * n * n * np.exp(-n * x)


def aps07_f(x, n):
    r"""Upside down parabola with parametrizable height"""
    return (1 + (1 - n)**2) * x - (1 - n * x)**2


def aps07_fp(x, n):
    return (1 + (1 - n)**2) + 2 * n * (1 - n * x)


def aps07_fpp(x, n):
    return -2 * n * n


def aps08_f(x, n):
    r"""Degree n polynomial"""
    return x * x - (1 - x)**n


def aps08_fp(x, n):
    return 2 * x + n * (1 - x)**(n - 1)


def aps08_fpp(x, n):
    return 2 - n * (n - 1) * (1 - x)**(n - 2)


def aps09_f(x, n):
    r"""Upside down quartic with parametrizable height"""
    return (1 + (1 - n)**4) * x - (1 - n * x)**4


def aps09_fp(x, n):
    return (1 + (1 - n)**4) + 4 * n * (1 - n * x)**3


def aps09_fpp(x, n):
    return -12 * n * (1 - n * x)**2


def aps10_f(x, n):
    r"""Exponential plus a polynomial"""
    return np.exp(-n * x) * (x - 1) + x**n


def aps10_fp(x, n):
    return np.exp(-n * x) * (-n * (x - 1) + 1) + n * x**(n - 1)


def aps10_fpp(x, n):
    return (np.exp(-n * x) * (-n * (-n * (x - 1) + 1) + -n * x)
            + n * (n - 1) * x**(n - 2))


def aps11_f(x, n):
    r"""Rational function with a zero at x=1/n and a pole at x=0"""
    return (n * x - 1) / ((n - 1) * x)


def aps11_fp(x, n):
    return 1 / (n - 1) / x**2


def aps11_fpp(x, n):
    return -2 / (n - 1) / x**3


def aps12_f(x, n):
    r"""nth root of x, with a zero at x=n"""
    return np.power(x, 1.0 / n) - np.power(n, 1.0 / n)


def aps12_fp(x, n):
    return np.power(x, (1.0 - n) / n) / n


def aps12_fpp(x, n):
    return np.power(x, (1.0 - 2 * n) / n) * (1.0 / n) * (1.0 - n) / n


_MAX_EXPABLE = np.log(np.finfo(float).max)


def aps13_f(x):
    r"""Function with *all* derivatives 0 at the root"""
    if x == 0:
        return 0
    # x2 = 1.0/x**2
    # if x2 > 708:
    #     return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return x / np.exp(y)


def aps13_fp(x):
    if x == 0:
        return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return (1 + 2 / x**2) / np.exp(y)


def aps13_fpp(x):
    if x == 0:
        return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return 2 * (2 - x**2) / x**5 / np.exp(y)


def aps14_f(x, n):
    r"""0 for negative x-values, trigonometric+linear for x positive"""
    if x <= 0:
        return -n / 20.0
    return n / 20.0 * (x / 1.5 + np.sin(x) - 1)


def aps14_fp(x, n):
    if x <= 0:
        return 0
    return n / 20.0 * (1.0 / 1.5 + np.cos(x))


def aps14_fpp(x, n):
    if x <= 0:
        return 0
    return -n / 20.0 * (np.sin(x))


def aps15_f(x, n):
    r"""piecewise linear, constant outside of [0, 0.002/(1+n)]"""
    if x < 0:
        return -0.859
    if x > 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    return np.exp((n + 1) * x / 2 * 1000) - 1.859


def aps15_fp(x, n):
    if not 0 <= x <= 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    return np.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000


def aps15_fpp(x, n):
    if not 0 <= x <= 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    return np.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000 * (n + 1) / 2 * 1000


# Each APS test case has
# - a function and its two derivatives,
# - additional arguments,
# - a bracket enclosing a root,
# - the order of differentiability of the function on this interval
# - a starting value for methods which don't require a bracket
# - the root (inside the bracket)
# - an Identifier of the test case
#
# Algorithm 748 is a bracketing algorithm so a bracketing interval was provided
# in [1] for each test case. Newton and Halley methods need a single
# starting point x0, which was chosen to be near the middle of the interval,
# unless that would have made the problem too easy.

_APS_TESTS_KEYS = [
    "f", "fprime", "fprime2", "args", "bracket", "smoothness", "x0", "root", "ID"
]
_APS_TESTS = [
    [aps01_f, aps01_fp, aps01_fpp, (), [np.pi / 2, np.pi], np.inf,
     3, 1.89549426703398094e+00, "aps.01.00"],
    [aps02_f, aps02_fp, aps02_fpp, (), [1 + 1e-9, 4 - 1e-9], np.inf,
     2, 3.02291534727305677e+00, "aps.02.00"],
    [aps02_f, aps02_fp, aps02_fpp, (), [4 + 1e-9, 9 - 1e-9], np.inf,
     5, 6.68375356080807848e+00, "aps.02.01"],
    [aps02_f, aps02_fp, aps02_fpp, (), [9 + 1e-9, 16 - 1e-9], np.inf,
     10, 1.12387016550022114e+01, "aps.02.02"],
    [aps02_f, aps02_fp, aps02_fpp, (), [16 + 1e-9, 25 - 1e-9], np.inf,
     17, 1.96760000806234103e+01, "aps.02.03"],
    [aps02_f, aps02_fp, aps02_fpp, (), [25 + 1e-9, 36 - 1e-9], np.inf,
     26, 2.98282273265047557e+01, "aps.02.04"],
    [aps02_f, aps02_fp, aps02_fpp, (), [36 + 1e-9, 49 - 1e-9], np.inf,
     37, 4.19061161952894139e+01, "aps.02.05"],
    [aps02_f, aps02_fp, aps02_fpp, (), [49 + 1e-9, 64 - 1e-9], np.inf,
     50, 5.59535958001430913e+01, "aps.02.06"],
    [aps02_f, aps02_fp, aps02_fpp, (), [64 + 1e-9, 81 - 1e-9], np.inf,
     65, 7.19856655865877997e+01, "aps.02.07"],
    [aps02_f, aps02_fp, aps02_fpp, (), [81 + 1e-9, 100 - 1e-9], np.inf,
     82, 9.00088685391666701e+01, "aps.02.08"],
    [aps02_f, aps02_fp, aps02_fpp, (), [100 + 1e-9, 121 - 1e-9], np.inf,
     101, 1.10026532748330197e+02, "aps.02.09"],
    [aps03_f, aps03_fp, aps03_fpp, (-40, -1), [-9, 31], np.inf,
     -2, 0, "aps.03.00"],
    [aps03_f, aps03_fp, aps03_fpp, (-100, -2), [-9, 31], np.inf,
     -2, 0, "aps.03.01"],
    [aps03_f, aps03_fp, aps03_fpp, (-200, -3), [-9, 31], np.inf,
     -2, 0, "aps.03.02"],
    [aps04_f, aps04_fp, aps04_fpp, (4, 0.2), [0, 5], np.inf,
     2.5, 6.68740304976422006e-01, "aps.04.00"],
    [aps04_f, aps04_fp, aps04_fpp, (6, 0.2), [0, 5], np.inf,
     2.5, 7.64724491331730039e-01, "aps.04.01"],
    [aps04_f, aps04_fp, aps04_fpp, (8, 0.2), [0, 5], np.inf,
     2.5, 8.17765433957942545e-01, "aps.04.02"],
    [aps04_f, aps04_fp, aps04_fpp, (10, 0.2), [0, 5], np.inf,
     2.5, 8.51339922520784609e-01, "aps.04.03"],
    [aps04_f, aps04_fp, aps04_fpp, (12, 0.2), [0, 5], np.inf,
     2.5, 8.74485272221167897e-01, "aps.04.04"],
    [aps04_f, aps04_fp, aps04_fpp, (4, 1), [0, 5], np.inf,
     2.5, 1, "aps.04.05"],
    [aps04_f, aps04_fp, aps04_fpp, (6, 1), [0, 5], np.inf,
     2.5, 1, "aps.04.06"],
    [aps04_f, aps04_fp, aps04_fpp, (8, 1), [0, 5], np.inf,
     2.5, 1, "aps.04.07"],
    [aps04_f, aps04_fp, aps04_fpp, (10, 1), [0, 5], np.inf,
     2.5, 1, "aps.04.08"],
    [aps04_f, aps04_fp, aps04_fpp, (12, 1), [0, 5], np.inf,
     2.5, 1, "aps.04.09"],
    [aps04_f, aps04_fp, aps04_fpp, (8, 1), [-0.95, 4.05], np.inf,
     1.5, 1, "aps.04.10"],
    [aps04_f, aps04_fp, aps04_fpp, (10, 1), [-0.95, 4.05], np.inf,
     1.5, 1, "aps.04.11"],
    [aps04_f, aps04_fp, aps04_fpp, (12, 1), [-0.95, 4.05], np.inf,
     1.5, 1, "aps.04.12"],
    [aps04_f, aps04_fp, aps04_fpp, (14, 1), [-0.95, 4.05], np.inf,
     1.5, 1, "aps.04.13"],
    [aps05_f, aps05_fp, aps05_fpp, (), [0, 1.5], np.inf,
     1.3, np.pi / 6, "aps.05.00"],
    [aps06_f, aps06_fp, aps06_fpp, (1,), [0, 1], np.inf,
     0.5, 4.22477709641236709e-01, "aps.06.00"],
    [aps06_f, aps06_fp, aps06_fpp, (2,), [0, 1], np.inf,
     0.5, 3.06699410483203705e-01, "aps.06.01"],
    [aps06_f, aps06_fp, aps06_fpp, (3,), [0, 1], np.inf,
     0.5, 2.23705457654662959e-01, "aps.06.02"],
    [aps06_f, aps06_fp, aps06_fpp, (4,), [0, 1], np.inf,
     0.5, 1.71719147519508369e-01, "aps.06.03"],
    [aps06_f, aps06_fp, aps06_fpp, (5,), [0, 1], np.inf,
     0.4, 1.38257155056824066e-01, "aps.06.04"],
    [aps06_f, aps06_fp, aps06_fpp, (20,), [0, 1], np.inf,
     0.1, 3.46573590208538521e-02, "aps.06.05"],
    [aps06_f, aps06_fp, aps06_fpp, (40,), [0, 1], np.inf,
     5e-02, 1.73286795139986315e-02, "aps.06.06"],
    [aps06_f, aps06_fp, aps06_fpp, (60,), [0, 1], np.inf,
     1.0 / 30, 1.15524530093324210e-02, "aps.06.07"],
    [aps06_f, aps06_fp, aps06_fpp, (80,), [0, 1], np.inf,
     2.5e-02, 8.66433975699931573e-03, "aps.06.08"],
    [aps06_f, aps06_fp, aps06_fpp, (100,), [0, 1], np.inf,
     2e-02, 6.93147180559945415e-03, "aps.06.09"],
    [aps07_f, aps07_fp, aps07_fpp, (5,), [0, 1], np.inf,
     0.4, 3.84025518406218985e-02, "aps.07.00"],
    [aps07_f, aps07_fp, aps07_fpp, (10,), [0, 1], np.inf,
     0.4, 9.90000999800049949e-03, "aps.07.01"],
    [aps07_f, aps07_fp, aps07_fpp, (20,), [0, 1], np.inf,
     0.4, 2.49375003906201174e-03, "aps.07.02"],
    [aps08_f, aps08_fp, aps08_fpp, (2,), [0, 1], np.inf,
     0.9, 0.5, "aps.08.00"],
    [aps08_f, aps08_fp, aps08_fpp, (5,), [0, 1], np.inf,
     0.9, 3.45954815848242059e-01, "aps.08.01"],
    [aps08_f, aps08_fp, aps08_fpp, (10,), [0, 1], np.inf,
     0.9, 2.45122333753307220e-01, "aps.08.02"],
    [aps08_f, aps08_fp, aps08_fpp, (15,), [0, 1], np.inf,
     0.9, 1.95547623536565629e-01, "aps.08.03"],
    [aps08_f, aps08_fp, aps08_fpp, (20,), [0, 1], np.inf,
     0.9, 1.64920957276440960e-01, "aps.08.04"],
    [aps09_f, aps09_fp, aps09_fpp, (1,), [0, 1], np.inf,
     0.5, 2.75508040999484394e-01, "aps.09.00"],
    [aps09_f, aps09_fp, aps09_fpp, (2,), [0, 1], np.inf,
     0.5, 1.37754020499742197e-01, "aps.09.01"],
    [aps09_f, aps09_fp, aps09_fpp, (4,), [0, 1], np.inf,
     0.5, 1.03052837781564422e-02, "aps.09.02"],
    [aps09_f, aps09_fp, aps09_fpp, (5,), [0, 1], np.inf,
     0.5, 3.61710817890406339e-03, "aps.09.03"],
    [aps09_f, aps09_fp, aps09_fpp, (8,), [0, 1], np.inf,
     0.5, 4.10872918496395375e-04, "aps.09.04"],
    [aps09_f, aps09_fp, aps09_fpp, (15,), [0, 1], np.inf,
     0.5, 2.59895758929076292e-05, "aps.09.05"],
    [aps09_f, aps09_fp, aps09_fpp, (20,), [0, 1], np.inf,
     0.5, 7.66859512218533719e-06, "aps.09.06"],
    [aps10_f, aps10_fp, aps10_fpp, (1,), [0, 1], np.inf,
     0.9, 4.01058137541547011e-01, "aps.10.00"],
    [aps10_f, aps10_fp, aps10_fpp, (5,), [0, 1], np.inf,
     0.9, 5.16153518757933583e-01, "aps.10.01"],
    [aps10_f, aps10_fp, aps10_fpp, (10,), [0, 1], np.inf,
     0.9, 5.39522226908415781e-01, "aps.10.02"],
    [aps10_f, aps10_fp, aps10_fpp, (15,), [0, 1], np.inf,
     0.9, 5.48182294340655241e-01, "aps.10.03"],
    [aps10_f, aps10_fp, aps10_fpp, (20,), [0, 1], np.inf,
     0.9, 5.52704666678487833e-01, "aps.10.04"],
    [aps11_f, aps11_fp, aps11_fpp, (2,), [0.01, 1], np.inf,
     1e-02, 1.0 / 2, "aps.11.00"],
    [aps11_f, aps11_fp, aps11_fpp, (5,), [0.01, 1], np.inf,
     1e-02, 1.0 / 5, "aps.11.01"],
    [aps11_f, aps11_fp, aps11_fpp, (15,), [0.01, 1], np.inf,
     1e-02, 1.0 / 15, "aps.11.02"],
    [aps11_f, aps11_fp, aps11_fpp, (20,), [0.01, 1], np.inf,
     1e-02, 1.0 / 20, "aps.11.03"],
    [aps12_f, aps12_fp, aps12_fpp, (2,), [1, 100], np.inf,
     1.1, 2, "aps.12.00"],
    [aps12_f, aps12_fp, aps12_fpp, (3,), [1, 100], np.inf,
     1.1, 3, "aps.12.01"],
    [aps12_f, aps12_fp, aps12_fpp, (4,), [1, 100], np.inf,
     1.1, 4, "aps.12.02"],
    [aps12_f, aps12_fp, aps12_fpp, (5,), [1, 100], np.inf,
     1.1, 5, "aps.12.03"],
    [aps12_f, aps12_fp, aps12_fpp, (6,), [1, 100], np.inf,
     1.1, 6, "aps.12.04"],
    [aps12_f, aps12_fp, aps12_fpp, (7,), [1, 100], np.inf,
     1.1, 7, "aps.12.05"],
    [aps12_f, aps12_fp, aps12_fpp, (9,), [1, 100], np.inf,
     1.1, 9, "aps.12.06"],
    [aps12_f, aps12_fp, aps12_fpp, (11,), [1, 100], np.inf,
     1.1, 11, "aps.12.07"],
    [aps12_f, aps12_fp, aps12_fpp, (13,), [1, 100], np.inf,
     1.1, 13, "aps.12.08"],
    [aps12_f, aps12_fp, aps12_fpp, (15,), [1, 100], np.inf,
     1.1, 15, "aps.12.09"],
    [aps12_f, aps12_fp, aps12_fpp, (17,), [1, 100], np.inf,
     1.1, 17, "aps.12.10"],
    [aps12_f, aps12_fp, aps12_fpp, (19,), [1, 100], np.inf,
     1.1, 19, "aps.12.11"],
    [aps12_f, aps12_fp, aps12_fpp, (21,), [1, 100], np.inf,
     1.1, 21, "aps.12.12"],
    [aps12_f, aps12_fp, aps12_fpp, (23,), [1, 100], np.inf,
     1.1, 23, "aps.12.13"],
    [aps12_f, aps12_fp, aps12_fpp, (25,), [1, 100], np.inf,
     1.1, 25, "aps.12.14"],
    [aps12_f, aps12_fp, aps12_fpp, (27,), [1, 100], np.inf,
     1.1, 27, "aps.12.15"],
    [aps12_f, aps12_fp, aps12_fpp, (29,), [1, 100], np.inf,
     1.1, 29, "aps.12.16"],
    [aps12_f, aps12_fp, aps12_fpp, (31,), [1, 100], np.inf,
     1.1, 31, "aps.12.17"],
    [aps12_f, aps12_fp, aps12_fpp, (33,), [1, 100], np.inf,
     1.1, 33, "aps.12.18"],
    [aps13_f, aps13_fp, aps13_fpp, (), [-1, 4], np.inf,
     1.5, 0, "aps.13.00"],
    [aps14_f, aps14_fp, aps14_fpp, (1,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.00"],
    [aps14_f, aps14_fp, aps14_fpp, (2,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.01"],
    [aps14_f, aps14_fp, aps14_fpp, (3,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.02"],
    [aps14_f, aps14_fp, aps14_fpp, (4,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.03"],
    [aps14_f, aps14_fp, aps14_fpp, (5,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.04"],
    [aps14_f, aps14_fp, aps14_fpp, (6,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.05"],
    [aps14_f, aps14_fp, aps14_fpp, (7,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.06"],
    [aps14_f, aps14_fp, aps14_fpp, (8,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.07"],
    [aps14_f, aps14_fp, aps14_fpp, (9,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.08"],
    [aps14_f, aps14_fp, aps14_fpp, (10,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.09"],
    [aps14_f, aps14_fp, aps14_fpp, (11,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.10"],
    [aps14_f, aps14_fp, aps14_fpp, (12,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.11"],
    [aps14_f, aps14_fp, aps14_fpp, (13,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.12"],
    [aps14_f, aps14_fp, aps14_fpp, (14,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.13"],
    [aps14_f, aps14_fp, aps14_fpp, (15,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.14"],
    [aps14_f, aps14_fp, aps14_fpp, (16,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.15"],
    [aps14_f, aps14_fp, aps14_fpp, (17,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.16"],
    [aps14_f, aps14_fp, aps14_fpp, (18,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.17"],
    [aps14_f, aps14_fp, aps14_fpp, (19,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.18"],
    [aps14_f, aps14_fp, aps14_fpp, (20,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.19"],
    [aps14_f, aps14_fp, aps14_fpp, (21,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.20"],
    [aps14_f, aps14_fp, aps14_fpp, (22,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.21"],
    [aps14_f, aps14_fp, aps14_fpp, (23,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.22"],
    [aps14_f, aps14_fp, aps14_fpp, (24,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.23"],
    [aps14_f, aps14_fp, aps14_fpp, (25,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.24"],
    [aps14_f, aps14_fp, aps14_fpp, (26,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.25"],
    [aps14_f, aps14_fp, aps14_fpp, (27,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.26"],
    [aps14_f, aps14_fp, aps14_fpp, (28,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.27"],
    [aps14_f, aps14_fp, aps14_fpp, (29,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.28"],
    [aps14_f, aps14_fp, aps14_fpp, (30,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.29"],
    [aps14_f, aps14_fp, aps14_fpp, (31,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.30"],
    [aps14_f, aps14_fp, aps14_fpp, (32,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.31"],
    [aps14_f, aps14_fp, aps14_fpp, (33,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.32"],
    [aps14_f, aps14_fp, aps14_fpp, (34,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.33"],
    [aps14_f, aps14_fp, aps14_fpp, (35,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.34"],
    [aps14_f, aps14_fp, aps14_fpp, (36,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.35"],
    [aps14_f, aps14_fp, aps14_fpp, (37,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.36"],
    [aps14_f, aps14_fp, aps14_fpp, (38,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.37"],
    [aps14_f, aps14_fp, aps14_fpp, (39,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.38"],
    [aps14_f, aps14_fp, aps14_fpp, (40,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.39"],
    [aps15_f, aps15_fp, aps15_fpp, (20,), [-1000, 1e-4], 0,
     -2, 5.90513055942197166e-05, "aps.15.00"],
    [aps15_f, aps15_fp, aps15_fpp, (21,), [-1000, 1e-4], 0,
     -2, 5.63671553399369967e-05, "aps.15.01"],
    [aps15_f, aps15_fp, aps15_fpp, (22,), [-1000, 1e-4], 0,
     -2, 5.39164094555919196e-05, "aps.15.02"],
    [aps15_f, aps15_fp, aps15_fpp, (23,), [-1000, 1e-4], 0,
     -2, 5.16698923949422470e-05, "aps.15.03"],
    [aps15_f, aps15_fp, aps15_fpp, (24,), [-1000, 1e-4], 0,
     -2, 4.96030966991445609e-05, "aps.15.04"],
    [aps15_f, aps15_fp, aps15_fpp, (25,), [-1000, 1e-4], 0,
     -2, 4.76952852876389951e-05, "aps.15.05"],
    [aps15_f, aps15_fp, aps15_fpp, (26,), [-1000, 1e-4], 0,
     -2, 4.59287932399486662e-05, "aps.15.06"],
    [aps15_f, aps15_fp, aps15_fpp, (27,), [-1000, 1e-4], 0,
     -2, 4.42884791956647841e-05, "aps.15.07"],
    [aps15_f, aps15_fp, aps15_fpp, (28,), [-1000, 1e-4], 0,
     -2, 4.27612902578832391e-05, "aps.15.08"],
    [aps15_f, aps15_fp, aps15_fpp, (29,), [-1000, 1e-4], 0,
     -2, 4.13359139159538030e-05, "aps.15.09"],
    [aps15_f, aps15_fp, aps15_fpp, (30,), [-1000, 1e-4], 0,
     -2, 4.00024973380198076e-05, "aps.15.10"],
    [aps15_f, aps15_fp, aps15_fpp, (31,), [-1000, 1e-4], 0,
     -2, 3.87524192962066869e-05, "aps.15.11"],
    [aps15_f, aps15_fp, aps15_fpp, (32,), [-1000, 1e-4], 0,
     -2, 3.75781035599579910e-05, "aps.15.12"],
    [aps15_f, aps15_fp, aps15_fpp, (33,), [-1000, 1e-4], 0,
     -2, 3.64728652199592355e-05, "aps.15.13"],
    [aps15_f, aps15_fp, aps15_fpp, (34,), [-1000, 1e-4], 0,
     -2, 3.54307833565318273e-05, "aps.15.14"],
    [aps15_f, aps15_fp, aps15_fpp, (35,), [-1000, 1e-4], 0,
     -2, 3.44465949299614980e-05, "aps.15.15"],
    [aps15_f, aps15_fp, aps15_fpp, (36,), [-1000, 1e-4], 0,
     -2, 3.35156058778003705e-05, "aps.15.16"],
    [aps15_f, aps15_fp, aps15_fpp, (37,), [-1000, 1e-4], 0,
     -2, 3.26336162494372125e-05, "aps.15.17"],
    [aps15_f, aps15_fp, aps15_fpp, (38,), [-1000, 1e-4], 0,
     -2, 3.17968568584260013e-05, "aps.15.18"],
    [aps15_f, aps15_fp, aps15_fpp, (39,), [-1000, 1e-4], 0,
     -2, 3.10019354369653455e-05, "aps.15.19"],
    [aps15_f, aps15_fp, aps15_fpp, (40,), [-1000, 1e-4], 0,
     -2, 3.02457906702100968e-05, "aps.15.20"],
    [aps15_f, aps15_fp, aps15_fpp, (100,), [-1000, 1e-4], 0,
     -2, 1.22779942324615231e-05, "aps.15.21"],
    [aps15_f, aps15_fp, aps15_fpp, (200,), [-1000, 1e-4], 0,
     -2, 6.16953939044086617e-06, "aps.15.22"],
    [aps15_f, aps15_fp, aps15_fpp, (300,), [-1000, 1e-4], 0,
     -2, 4.11985852982928163e-06, "aps.15.23"],
    [aps15_f, aps15_fp, aps15_fpp, (400,), [-1000, 1e-4], 0,
     -2, 3.09246238772721682e-06, "aps.15.24"],
    [aps15_f, aps15_fp, aps15_fpp, (500,), [-1000, 1e-4], 0,
     -2, 2.47520442610501789e-06, "aps.15.25"],
    [aps15_f, aps15_fp, aps15_fpp, (600,), [-1000, 1e-4], 0,
     -2, 2.06335676785127107e-06, "aps.15.26"],
    [aps15_f, aps15_fp, aps15_fpp, (700,), [-1000, 1e-4], 0,
     -2, 1.76901200781542651e-06, "aps.15.27"],
    [aps15_f, aps15_fp, aps15_fpp, (800,), [-1000, 1e-4], 0,
     -2, 1.54816156988591016e-06, "aps.15.28"],
    [aps15_f, aps15_fp, aps15_fpp, (900,), [-1000, 1e-4], 0,
     -2, 1.37633453660223511e-06, "aps.15.29"],
    [aps15_f, aps15_fp, aps15_fpp, (1000,), [-1000, 1e-4], 0,
     -2, 1.23883857889971403e-06, "aps.15.30"]
]

_APS_TESTS_DICTS = [dict(zip(_APS_TESTS_KEYS, testcase)) for testcase in _APS_TESTS]


#   ##################
#   "complex" test cases
#   A few simple, complex-valued, functions, defined on the complex plane.


def cplx01_f(z, n, a):
    r"""z**n-a:  Use to find the nth root of a"""
    return z**n - a


def cplx01_fp(z, n, a):
    return n * z**(n - 1)


def cplx01_fpp(z, n, a):
    return n * (n - 1) * z**(n - 2)


def cplx02_f(z, a):
    r"""e**z - a: Use to find the log of a"""
    return np.exp(z) - a


def cplx02_fp(z, a):
    return np.exp(z)


def cplx02_fpp(z, a):
    return np.exp(z)


# Each "complex" test case has
# - a function and its two derivatives,
# - additional arguments,
# - the order of differentiability of the function on this interval
# - two starting values x0 and x1
# - the root
# - an Identifier of the test case
#
# Algorithm 748 is a bracketing algorithm so a bracketing interval was provided
# in [1] for each test case. Newton and Halley need a single starting point
# x0, which was chosen to be near the middle of the interval, unless that
# would make the problem too easy.


_COMPLEX_TESTS_KEYS = [
    "f", "fprime", "fprime2", "args", "smoothness", "x0", "x1", "root", "ID"
]
_COMPLEX_TESTS = [
    [cplx01_f, cplx01_fp, cplx01_fpp, (2, -1), np.inf,
     (1 + 1j), (0.5 + 0.5j), 1j, "complex.01.00"],
    [cplx01_f, cplx01_fp, cplx01_fpp, (3, 1), np.inf,
     (-1 + 1j), (-0.5 + 2.0j), (-0.5 + np.sqrt(3) / 2 * 1.0j),
     "complex.01.01"],
    [cplx01_f, cplx01_fp, cplx01_fpp, (3, -1), np.inf,
     1j, (0.5 + 0.5j), (0.5 + np.sqrt(3) / 2 * 1.0j),
     "complex.01.02"],
    [cplx01_f, cplx01_fp, cplx01_fpp, (3, 8), np.inf,
     5, 4, 2, "complex.01.03"],
    [cplx02_f, cplx02_fp, cplx02_fpp, (-1,), np.inf,
     (1 + 2j), (0.5 + 0.5j), np.pi * 1.0j, "complex.02.00"],
    [cplx02_f, cplx02_fp, cplx02_fpp, (1j,), np.inf,
     (1 + 2j), (0.5 + 0.5j), np.pi * 0.5j, "complex.02.01"],
]

_COMPLEX_TESTS_DICTS = [
    dict(zip(_COMPLEX_TESTS_KEYS, testcase)) for testcase in _COMPLEX_TESTS
]


def _add_a_b(tests):
    r"""Add "a" and "b" keys to each test from the "bracket" value"""
    for d in tests:
        for k, v in zip(['a', 'b'], d.get('bracket', [])):
            d[k] = v


_add_a_b(_ORIGINAL_TESTS_DICTS)
_add_a_b(_APS_TESTS_DICTS)
_add_a_b(_COMPLEX_TESTS_DICTS)


def get_tests(collection='original', smoothness=None):
    r"""Return the requested collection of test cases, as an array of dicts with subset-specific keys

    Allowed values of collection:
    'original': The original benchmarking functions.
         Real-valued functions of real-valued inputs on an interval with a zero.
         f1, .., f3 are continuous and infinitely differentiable
         f4 has a single discontinuity at the root
         f5 has a root at 1 replacing a 1st order pole
         f6 is randomly positive on one side of the root, randomly negative on the other
    'aps': The test problems in the TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions"
         paper by Alefeld, Potra and Shi. Real-valued functions of
         real-valued inputs on an interval with a zero.
         Suitable for methods which start with an enclosing interval, and
         derivatives up to 2nd order.
    'complex': Some complex-valued functions of complex-valued inputs.
         No enclosing bracket is provided.
         Suitable for methods which use one or more starting values, and
         derivatives up to 2nd order.

    The dictionary keys will be a subset of
    ["f", "fprime", "fprime2", "args", "bracket", "a", b", "smoothness", "x0", "x1", "root", "ID"]
    """  # noqa: E501
    collection = collection or "original"
    subsets = {"aps": _APS_TESTS_DICTS,
               "complex": _COMPLEX_TESTS_DICTS,
               "original": _ORIGINAL_TESTS_DICTS,
               "chandrupatla": _CHANDRUPATLA_TESTS_DICTS}
    tests = subsets.get(collection, [])
    if smoothness is not None:
        tests = [tc for tc in tests if tc['smoothness'] >= smoothness]
    return tests


# Backwards compatibility
methods = [cc.bisect, cc.ridder, cc.brenth, cc.brentq]
mstrings = ['cc.bisect', 'cc.ridder', 'cc.brenth', 'cc.brentq']
functions = [f2, f3, f4, f5, f6]
fstrings = ['f2', 'f3', 'f4', 'f5', 'f6']

#   ##################
#   "Chandrupatla" test cases
#   Functions and test cases that appear in [2]

def fun1(x):
    return x**3 - 2*x - 5
fun1.root = 2.0945514815423265  # additional precision using mpmath.findroot


def fun2(x):
    return 1 - 1/x**2
fun2.root = 1


def fun3(x):
    return (x-3)**3
fun3.root = 3


def fun4(x):
    return 6*(x-2)**5
fun4.root = 2


def fun5(x):
    return x**9
fun5.root = 0


def fun6(x):
    return x**19
fun6.root = 0


def fun7(x):
    xp = array_namespace(x)
    return 0 if xp.abs(x) < 3.8e-4 else x*xp.exp(-x**(-2))
fun7.root = 0


def fun8(x):
    xp = array_namespace(x)
    xi = 0.61489
    return -(3062*(1-xi)*xp.exp(-x))/(xi + (1-xi)*xp.exp(-x)) - 1013 + 1628/x
fun8.root = 1.0375360332870405


def fun9(x):
    xp = array_namespace(x)
    return xp.exp(x) - 2 - 0.01/x**2 + .000002/x**3
fun9.root = 0.7032048403631358

# Each "chandropatla" test case has
# - a function,
# - two starting values x0 and x1
# - the root
# - the number of function evaluations required by Chandrupatla's algorithm
# - an Identifier of the test case
#
# Chandrupatla's is a bracketing algorithm, so a bracketing interval was
# provided in [2] for each test case. No special support for testing with
# secant/Newton/Halley is provided.

_CHANDRUPATLA_TESTS_KEYS = ["f", "bracket", "root", "nfeval", "ID"]
_CHANDRUPATLA_TESTS = [
    [fun1, [2, 3], fun1.root, 7],
    [fun1, [1, 10], fun1.root, 11],
    [fun1, [1, 100], fun1.root, 14],
    [fun1, [-1e4, 1e4], fun1.root, 23],
    [fun1, [-1e10, 1e10], fun1.root, 43],
    [fun2, [0.5, 1.51], fun2.root, 8],
    [fun2, [1e-4, 1e4], fun2.root, 22],
    [fun2, [1e-6, 1e6], fun2.root, 28],
    [fun2, [1e-10, 1e10], fun2.root, 41],
    [fun2, [1e-12, 1e12], fun2.root, 48],
    [fun3, [0, 5], fun3.root, 21],
    [fun3, [-10, 10], fun3.root, 23],
    [fun3, [-1e4, 1e4], fun3.root, 36],
    [fun3, [-1e6, 1e6], fun3.root, 45],
    [fun3, [-1e10, 1e10], fun3.root, 55],
    [fun4, [0, 5], fun4.root, 21],
    [fun4, [-10, 10], fun4.root, 23],
    [fun4, [-1e4, 1e4], fun4.root, 33],
    [fun4, [-1e6, 1e6], fun4.root, 43],
    [fun4, [-1e10, 1e10], fun4.root, 54],
    [fun5, [-1, 4], fun5.root, 21],
    [fun5, [-2, 5], fun5.root, 22],
    [fun5, [-1, 10], fun5.root, 23],
    [fun5, [-5, 50], fun5.root, 25],
    [fun5, [-10, 100], fun5.root, 26],
    [fun6, [-1., 4.], fun6.root, 21],
    [fun6, [-2., 5.], fun6.root, 22],
    [fun6, [-1., 10.], fun6.root, 23],
    [fun6, [-5., 50.], fun6.root, 25],
    [fun6, [-10., 100.], fun6.root, 26],
    [fun7, [-1, 4], fun7.root, 8],
    [fun7, [-2, 5], fun7.root, 8],
    [fun7, [-1, 10], fun7.root, 11],
    [fun7, [-5, 50], fun7.root, 18],
    [fun7, [-10, 100], fun7.root, 19],
    [fun8, [2e-4, 2], fun8.root, 9],
    [fun8, [2e-4, 3], fun8.root, 10],
    [fun8, [2e-4, 9], fun8.root, 11],
    [fun8, [2e-4, 27], fun8.root, 12],
    [fun8, [2e-4, 81], fun8.root, 14],
    [fun9, [2e-4, 1], fun9.root, 7],
    [fun9, [2e-4, 3], fun9.root, 8],
    [fun9, [2e-4, 9], fun9.root, 10],
    [fun9, [2e-4, 27], fun9.root, 11],
    [fun9, [2e-4, 81], fun9.root, 13],
]
_CHANDRUPATLA_TESTS = [test + [f'{test[0].__name__}.{i%5+1}']
                       for i, test in enumerate(_CHANDRUPATLA_TESTS)]

_CHANDRUPATLA_TESTS_DICTS = [dict(zip(_CHANDRUPATLA_TESTS_KEYS, testcase))
                             for testcase in _CHANDRUPATLA_TESTS]
_add_a_b(_CHANDRUPATLA_TESTS_DICTS)
<|endoftext|>"""
This module implements the Sequential Least Squares Programming optimization
algorithm (SLSQP), originally developed by Dieter Kraft.
See http://www.netlib.org/toms/733

Functions
---------
.. autosummary::
   :toctree: generated/

    approx_jacobian
    fmin_slsqp

"""

__all__ = ['approx_jacobian', 'fmin_slsqp']

import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, concatenate, finfo,
                   sqrt, vstack, isfinite, atleast_1d)
from ._optimize import (OptimizeResult, _check_unknown_options,
                        _prepare_scalar_function, _clip_x_for_func,
                        _check_clip_x)
from ._numdiff import approx_derivative
from ._constraints import old_bound_to_new, _arr_to_scalar
from scipy._lib._array_api import array_namespace
from scipy._lib import array_api_extra as xpx


__docformat__ = "restructuredtext en"

_epsilon = sqrt(finfo(float).eps)


def approx_jacobian(x, func, epsilon, *args):
    """
    Approximate the Jacobian matrix of a callable function.

    Parameters
    ----------
    x : array_like
        The state vector at which to compute the Jacobian matrix.
    func : callable f(x,*args)
        The vector-valued function.
    epsilon : float
        The perturbation used to determine the partial derivatives.
    args : sequence
        Additional arguments passed to func.

    Returns
    -------
    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length
    of the outputs of `func`, and ``lenx`` is the number of elements in
    `x`.

    Notes
    -----
    The approximation is done using forward differences.

    """
    # approx_derivative returns (m, n) == (lenf, lenx)
    jac = approx_derivative(func, x, method='2-point', abs_step=epsilon,
                            args=args)
    # if func returns a scalar jac.shape will be (lenx,). Make sure
    # it's at least a 2D array.
    return np.atleast_2d(jac)


def fmin_slsqp(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None,
               bounds=(), fprime=None, fprime_eqcons=None,
               fprime_ieqcons=None, args=(), iter=100, acc=1.0E-6,
               iprint=1, disp=None, full_output=0, epsilon=_epsilon,
               callback=None):
    """
    Minimize a function using Sequential Least Squares Programming

    Python interface function for the SLSQP Optimization subroutine
    originally implemented by Dieter Kraft.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function.  Must return a scalar.
    x0 : 1-D ndarray of float
        Initial guess for the independent variable(s).
    eqcons : list, optional
        A list of functions of length n such that
        eqcons[j](x,*args) == 0.0 in a successfully optimized
        problem.
    f_eqcons : callable f(x,*args), optional
        Returns a 1-D array in which each element must equal 0.0 in a
        successfully optimized problem. If f_eqcons is specified,
        eqcons is ignored.
    ieqcons : list, optional
        A list of functions of length n such that
        ieqcons[j](x,*args) >= 0.0 in a successfully optimized
        problem.
    f_ieqcons : callable f(x,*args), optional
        Returns a 1-D ndarray in which each element must be greater or
        equal to 0.0 in a successfully optimized problem. If
        f_ieqcons is specified, ieqcons is ignored.
    bounds : list, optional
        A list of tuples specifying the lower and upper bound
        for each independent variable [(xl0, xu0),(xl1, xu1),...]
        Infinite values will be interpreted as large floating values.
    fprime : callable ``f(x,*args)``, optional
        A function that evaluates the partial derivatives of func.
    fprime_eqcons : callable ``f(x,*args)``, optional
        A function of the form ``f(x, *args)`` that returns the m by n
        array of equality constraint normals. If not provided,
        the normals will be approximated. The array returned by
        fprime_eqcons should be sized as ( len(eqcons), len(x0) ).
    fprime_ieqcons : callable ``f(x,*args)``, optional
        A function of the form ``f(x, *args)`` that returns the m by n
        array of inequality constraint normals. If not provided,
        the normals will be approximated. The array returned by
        fprime_ieqcons should be sized as ( len(ieqcons), len(x0) ).
    args : sequence, optional
        Additional arguments passed to func and fprime.
    iter : int, optional
        The maximum number of iterations.
    acc : float, optional
        Requested accuracy.
    iprint : int, optional
        The verbosity of fmin_slsqp :

        * iprint <= 0 : Silent operation
        * iprint == 1 : Print summary upon completion (default)
        * iprint >= 2 : Print status of each iterate and summary
    disp : int, optional
        Overrides the iprint interface (preferred).
    full_output : bool, optional
        If False, return only the minimizer of func (default).
        Otherwise, output final objective function and summary
        information.
    epsilon : float, optional
        The step size for finite-difference derivative estimates.
    callback : callable, optional
        Called after each iteration, as ``callback(x)``, where ``x`` is the
        current parameter vector.

    Returns
    -------
    out : ndarray of float
        The final minimizer of func.
    fx : ndarray of float, if full_output is true
        The final value of the objective function.
    its : int, if full_output is true
        The number of iterations.
    imode : int, if full_output is true
        The exit mode from the optimizer (see below).
    smode : string, if full_output is true
        Message describing the exit mode from the optimizer.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'SLSQP' `method` in particular.

    Notes
    -----
    Exit modes are defined as follows:

    - ``-1`` : Gradient evaluation required (g & a)
    - ``0`` : Optimization terminated successfully
    - ``1`` : Function evaluation required (f & c)
    - ``2`` : More equality constraints than independent variables
    - ``3`` : More than 3*n iterations in LSQ subproblem
    - ``4`` : Inequality constraints incompatible
    - ``5`` : Singular matrix E in LSQ subproblem
    - ``6`` : Singular matrix C in LSQ subproblem
    - ``7`` : Rank-deficient equality constraint subproblem HFTI
    - ``8`` : Positive directional derivative for linesearch
    - ``9`` : Iteration limit reached

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-sqlsp>`.

    """
    if disp is not None:
        iprint = disp

    opts = {'maxiter': iter,
            'ftol': acc,
            'iprint': iprint,
            'disp': iprint != 0,
            'eps': epsilon,
            'callback': callback}

    # Build the constraints as a tuple of dictionaries
    cons = ()
    # 1. constraints of the 1st kind (eqcons, ieqcons); no Jacobian; take
    #    the same extra arguments as the objective function.
    cons += tuple({'type': 'eq', 'fun': c, 'args': args} for c in eqcons)
    cons += tuple({'type': 'ineq', 'fun': c, 'args': args} for c in ieqcons)
    # 2. constraints of the 2nd kind (f_eqcons, f_ieqcons) and their Jacobian
    #    (fprime_eqcons, fprime_ieqcons); also take the same extra arguments
    #    as the objective function.
    if f_eqcons:
        cons += ({'type': 'eq', 'fun': f_eqcons, 'jac': fprime_eqcons,
                  'args': args}, )
    if f_ieqcons:
        cons += ({'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons,
                  'args': args}, )

    res = _minimize_slsqp(func, x0, args, jac=fprime, bounds=bounds,
                          constraints=cons, **opts)
    if full_output:
        return res['x'], res['fun'], res['nit'], res['status'], res['message']
    else:
        return res['x']


def _minimize_slsqp(func, x0, args=(), jac=None, bounds=None,
                    constraints=(),
                    maxiter=100, ftol=1.0E-6, iprint=1, disp=False,
                    eps=_epsilon, callback=None, finite_diff_rel_step=None,
                    **unknown_options):
    """
    Minimize a scalar function of one or more variables using Sequential
    Least Squares Programming (SLSQP).

    Options
    -------
    ftol : float
        Precision goal for the value of f in the stopping criterion.
    eps : float
        Step size used for numerical approximation of the Jacobian.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored and set to 0.
    maxiter : int
        Maximum number of iterations.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of `jac`. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    _check_unknown_options(unknown_options)
    iter = maxiter - 1
    acc = ftol
    epsilon = eps

    if not disp:
        iprint = 0

    # Transform x0 into an array.
    xp = array_namespace(x0)
    x0 = xpx.atleast_nd(xp.asarray(x0), ndim=1, xp=xp)
    dtype = xp.float64
    if xp.isdtype(x0.dtype, "real floating"):
        dtype = x0.dtype
    x = xp.reshape(xp.astype(x0, dtype), -1)

    # SLSQP is sent 'old-style' bounds, 'new-style' bounds are required by
    # ScalarFunction
    if bounds is None or len(bounds) == 0:
        new_bounds = (-np.inf, np.inf)
    else:
        new_bounds = old_bound_to_new(bounds)

    # clip the initial guess to bounds, otherwise ScalarFunction doesn't work
    x = np.clip(x, new_bounds[0], new_bounds[1])

    # Constraints are triaged per type into a dictionary of tuples
    if isinstance(constraints, dict):
        constraints = (constraints, )

    cons = {'eq': (), 'ineq': ()}
    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype not in ['eq', 'ineq']:
                raise ValueError(f"Unknown constraint type '{con['type']}'.")

        # check function
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)

        # check Jacobian
        cjac = con.get('jac')
        if cjac is None:
            # approximate Jacobian function. The factory function is needed
            # to keep a reference to `fun`, see gh-4240.
            def cjac_factory(fun):
                def cjac(x, *args):
                    x = _check_clip_x(x, new_bounds)

                    if jac in ['2-point', '3-point', 'cs']:
                        return approx_derivative(fun, x, method=jac, args=args,
                                                 rel_step=finite_diff_rel_step,
                                                 bounds=new_bounds)
                    else:
                        return approx_derivative(fun, x, method='2-point',
                                                 abs_step=epsilon, args=args,
                                                 bounds=new_bounds)

                return cjac
            cjac = cjac_factory(con['fun'])

        # update constraints' dictionary
        cons[ctype] += ({'fun': con['fun'],
                         'jac': cjac,
                         'args': con.get('args', ())}, )

    exit_modes = {-1: "Gradient evaluation required (g & a)",
                   0: "Optimization terminated successfully",
                   1: "Function evaluation required (f & c)",
                   2: "More equality constraints than independent variables",
                   3: "More than 3*n iterations in LSQ subproblem",
                   4: "Inequality constraints incompatible",
                   5: "Singular matrix E in LSQ subproblem",
                   6: "Singular matrix C in LSQ subproblem",
                   7: "Rank-deficient equality constraint subproblem HFTI",
                   8: "Positive directional derivative for linesearch",
                   9: "Iteration limit reached"}

    # Set the parameters that SLSQP will need
    # meq, mieq: number of equality and inequality constraints
    meq = sum(map(len, [atleast_1d(c['fun'](x, *c['args']))
              for c in cons['eq']]))
    mieq = sum(map(len, [atleast_1d(c['fun'](x, *c['args']))
               for c in cons['ineq']]))
    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = array([1, m]).max()
    # n = The number of independent variables
    n = len(x)

    # Define the workspaces for SLSQP
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    len_jw = mineq
    w = zeros(len_w)
    jw = zeros(len_jw)

    # Decompose bounds into xl and xu
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        bnds = array([(_arr_to_scalar(l), _arr_to_scalar(u))
                      for (l, u) in bounds], float)
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not '
                             'compatible with that of x0.')

        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]

        if bnderr.any():
            raise ValueError("SLSQP Error: lb > ub in bounds "
                             f"{', '.join(str(b) for b in bnderr)}.")
        xl, xu = bnds[:, 0], bnds[:, 1]

        # Mark infinite bounds with nans; the Fortran code understands this
        infbnd = ~isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    # ScalarFunction provides function and gradient evaluation
    sf = _prepare_scalar_function(func, x, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  bounds=new_bounds)
    # gh11403 SLSQP sometimes exceeds bounds by 1 or 2 ULP, make sure this
    # doesn't get sent to the func/grad evaluator.
    wrapped_fun = _clip_x_for_func(sf.fun, new_bounds)
    wrapped_grad = _clip_x_for_func(sf.grad, new_bounds)

    # Initialize the iteration counter and the mode value
    mode = array(0, int)
    acc = array(acc, float)
    majiter = array(iter, int)
    majiter_prev = 0

    # Initialize internal SLSQP state variables
    alpha = array(0, float)
    f0 = array(0, float)
    gs = array(0, float)
    h1 = array(0, float)
    h2 = array(0, float)
    h3 = array(0, float)
    h4 = array(0, float)
    t = array(0, float)
    t0 = array(0, float)
    tol = array(0, float)
    iexact = array(0, int)
    incons = array(0, int)
    ireset = array(0, int)
    itermx = array(0, int)
    line = array(0, int)
    n1 = array(0, int)
    n2 = array(0, int)
    n3 = array(0, int)

    # Print the header if iprint >= 2
    if iprint >= 2:
        print("%5s %5s %16s %16s" % ("NIT", "FC", "OBJFUN", "GNORM"))

    # mode is zero on entry, so call objective, constraints and gradients
    # there should be no func evaluations here because it's cached from
    # ScalarFunction
    fx = wrapped_fun(x)
    g = append(wrapped_grad(x), 0.0)
    c = _eval_constraint(x, cons)
    a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

    while 1:
        # Call SLSQP
        slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
              alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
              iexact, incons, ireset, itermx, line,
              n1, n2, n3)

        if mode == 1:  # objective and constraint evaluation required
            fx = wrapped_fun(x)
            c = _eval_constraint(x, cons)

        if mode == -1:  # gradient evaluation required
            g = append(wrapped_grad(x), 0.0)
            a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

        if majiter > majiter_prev:
            # call callback if major iteration has incremented
            if callback is not None:
                callback(np.copy(x))

            # Print the status of the current iterate if iprint > 2
            if iprint >= 2:
                print("%5i %5i % 16.6E % 16.6E" % (majiter, sf.nfev,
                                                   fx, linalg.norm(g)))

        # If exit mode is not -1 or 1, slsqp has completed
        if abs(mode) != 1:
            break

        majiter_prev = int(majiter)

    # Optimization loop complete. Print status if requested
    if iprint >= 1:
        print(exit_modes[int(mode)] + "    (Exit mode " + str(mode) + ')')
        print("            Current function value:", fx)
        print("            Iterations:", majiter)
        print("            Function evaluations:", sf.nfev)
        print("            Gradient evaluations:", sf.ngev)

    return OptimizeResult(x=x, fun=fx, jac=g[:-1], nit=int(majiter),
                          nfev=sf.nfev, njev=sf.ngev, status=int(mode),
                          message=exit_modes[int(mode)], success=(mode == 0))


def _eval_constraint(x, cons):
    # Compute constraints
    if cons['eq']:
        c_eq = concatenate([atleast_1d(con['fun'](x, *con['args']))
                            for con in cons['eq']])
    else:
        c_eq = zeros(0)

    if cons['ineq']:
        c_ieq = concatenate([atleast_1d(con['fun'](x, *con['args']))
                             for con in cons['ineq']])
    else:
        c_ieq = zeros(0)

    # Now combine c_eq and c_ieq into a single matrix
    c = concatenate((c_eq, c_ieq))
    return c


def _eval_con_normals(x, cons, la, n, m, meq, mieq):
    # Compute the normals of the constraints
    if cons['eq']:
        a_eq = vstack([con['jac'](x, *con['args'])
                       for con in cons['eq']])
    else:  # no equality constraint
        a_eq = zeros((meq, n))

    if cons['ineq']:
        a_ieq = vstack([con['jac'](x, *con['args'])
                        for con in cons['ineq']])
    else:  # no inequality constraint
        a_ieq = zeros((mieq, n))

    # Now combine a_eq and a_ieq into a single a matrix
    if m == 0:  # no constraints
        a = zeros((la, n))
    else:
        a = vstack((a_eq, a_ieq))
    a = concatenate((a, zeros([la, 1])), 1)

    return a
<|endoftext|>"""
Unified interfaces to root finding algorithms.

Functions
---------
- root : find a root of a vector function.
"""
__all__ = ['root']

import numpy as np

from warnings import warn

from ._optimize import MemoizeJac, OptimizeResult, _check_unknown_options
from ._minpack_py import _root_hybr, leastsq
from ._spectral import _root_df_sane
from . import _nonlin as nonlin


ROOT_METHODS = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson',
                'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov',
                'df-sane']


def root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None,
         options=None):
    r"""
    Find a root of a vector function.

    Parameters
    ----------
    fun : callable
        A vector function to find a root of.

        Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
        ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
        Rather than passing ``f0`` as the callable, wrap it to accept
        only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
        callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
        gathered before invoking this function.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its Jacobian.
    method : str, optional
        Type of solver. Should be one of

        - 'hybr'             :ref:`(see here) <optimize.root-hybr>`
        - 'lm'               :ref:`(see here) <optimize.root-lm>`
        - 'broyden1'         :ref:`(see here) <optimize.root-broyden1>`
        - 'broyden2'         :ref:`(see here) <optimize.root-broyden2>`
        - 'anderson'         :ref:`(see here) <optimize.root-anderson>`
        - 'linearmixing'     :ref:`(see here) <optimize.root-linearmixing>`
        - 'diagbroyden'      :ref:`(see here) <optimize.root-diagbroyden>`
        - 'excitingmixing'   :ref:`(see here) <optimize.root-excitingmixing>`
        - 'krylov'           :ref:`(see here) <optimize.root-krylov>`
        - 'df-sane'          :ref:`(see here) <optimize.root-dfsane>`

    jac : bool or callable, optional
        If `jac` is a Boolean and is True, `fun` is assumed to return the
        value of Jacobian along with the objective function. If False, the
        Jacobian will be estimated numerically.
        `jac` can also be a callable returning the Jacobian of `fun`. In
        this case, it must accept the same arguments as `fun`.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    callback : function, optional
        Optional callback function. It is called on every iteration as
        ``callback(x, f)`` where `x` is the current solution and `f`
        the corresponding residual. For all methods but 'hybr' and 'lm'.
    options : dict, optional
        A dictionary of solver options. E.g., `xtol` or `maxiter`, see
        :obj:`show_options()` for details.

    Returns
    -------
    sol : OptimizeResult
        The solution represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the algorithm exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    See also
    --------
    show_options : Additional options accepted by the solvers

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *hybr*.

    Method *hybr* uses a modification of the Powell hybrid method as
    implemented in MINPACK [1]_.

    Method *lm* solves the system of nonlinear equations in a least squares
    sense using a modification of the Levenberg-Marquardt algorithm as
    implemented in MINPACK [1]_.

    Method *df-sane* is a derivative-free spectral method. [3]_

    Methods *broyden1*, *broyden2*, *anderson*, *linearmixing*,
    *diagbroyden*, *excitingmixing*, *krylov* are inexact Newton methods,
    with backtracking or full line searches [2]_. Each method corresponds
    to a particular Jacobian approximations.

    - Method *broyden1* uses Broyden's first Jacobian approximation, it is
      known as Broyden's good method.
    - Method *broyden2* uses Broyden's second Jacobian approximation, it
      is known as Broyden's bad method.
    - Method *anderson* uses (extended) Anderson mixing.
    - Method *Krylov* uses Krylov approximation for inverse Jacobian. It
      is suitable for large-scale problem.
    - Method *diagbroyden* uses diagonal Broyden Jacobian approximation.
    - Method *linearmixing* uses a scalar Jacobian approximation.
    - Method *excitingmixing* uses a tuned diagonal Jacobian
      approximation.

    .. warning::

        The algorithms implemented for methods *diagbroyden*,
        *linearmixing* and *excitingmixing* may be useful for specific
        problems, but whether they will work may depend strongly on the
        problem.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] More, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.
       1980. User Guide for MINPACK-1.
    .. [2] C. T. Kelley. 1995. Iterative Methods for Linear and Nonlinear
       Equations. Society for Industrial and Applied Mathematics.
       <https://archive.siam.org/books/kelley/fr16/>
    .. [3] W. La Cruz, J.M. Martinez, M. Raydan. Math. Comp. 75, 1429 (2006).

    Examples
    --------
    The following functions define a system of nonlinear equations and its
    jacobian.

    >>> import numpy as np
    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    >>> def jac(x):
    ...     return np.array([[1 + 1.5 * (x[0] - x[1])**2,
    ...                       -1.5 * (x[0] - x[1])**2],
    ...                      [-1.5 * (x[1] - x[0])**2,
    ...                       1 + 1.5 * (x[1] - x[0])**2]])

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.root(fun, [0, 0], jac=jac, method='hybr')
    >>> sol.x
    array([ 0.8411639,  0.1588361])

    **Large problem**

    Suppose that we needed to solve the following integrodifferential
    equation on the square :math:`[0,1]\times[0,1]`:

    .. math::

       \nabla^2 P = 10 \left(\int_0^1\int_0^1\cosh(P)\,dx\,dy\right)^2

    with :math:`P(x,1) = 1` and :math:`P=0` elsewhere on the boundary of
    the square.

    The solution can be found using the ``method='krylov'`` solver:

    >>> from scipy import optimize
    >>> # parameters
    >>> nx, ny = 75, 75
    >>> hx, hy = 1./(nx-1), 1./(ny-1)

    >>> P_left, P_right = 0, 0
    >>> P_top, P_bottom = 1, 0

    >>> def residual(P):
    ...    d2x = np.zeros_like(P)
    ...    d2y = np.zeros_like(P)
    ...
    ...    d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx
    ...    d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
    ...    d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx
    ...
    ...    d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
    ...    d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
    ...    d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy
    ...
    ...    return d2x + d2y - 10*np.cosh(P).mean()**2

    >>> guess = np.zeros((nx, ny), float)
    >>> sol = optimize.root(residual, guess, method='krylov')
    >>> print('Residual: %g' % abs(residual(sol.x)).max())
    Residual: 5.7972e-06  # may vary

    >>> import matplotlib.pyplot as plt
    >>> x, y = np.mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
    >>> plt.pcolormesh(x, y, sol.x, shading='gouraud')
    >>> plt.colorbar()
    >>> plt.show()

    """
    def _wrapped_fun(*fargs):
        """
        Wrapped `func` to track the number of times
        the function has been called.
        """
        _wrapped_fun.nfev += 1
        return fun(*fargs)

    _wrapped_fun.nfev = 0

    if not isinstance(args, tuple):
        args = (args,)

    meth = method.lower()
    if options is None:
        options = {}

    if callback is not None and meth in ('hybr', 'lm'):
        warn(f'Method {method} does not accept callback.',
             RuntimeWarning, stacklevel=2)

    # fun also returns the Jacobian
    if not callable(jac) and meth in ('hybr', 'lm'):
        if bool(jac):
            fun = MemoizeJac(fun)
            jac = fun.derivative
        else:
            jac = None

    # set default tolerances
    if tol is not None:
        options = dict(options)
        if meth in ('hybr', 'lm'):
            options.setdefault('xtol', tol)
        elif meth in ('df-sane',):
            options.setdefault('ftol', tol)
        elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
                      'diagbroyden', 'excitingmixing', 'krylov'):
            options.setdefault('xtol', tol)
            options.setdefault('xatol', np.inf)
            options.setdefault('ftol', np.inf)
            options.setdefault('fatol', np.inf)

    if meth == 'hybr':
        sol = _root_hybr(_wrapped_fun, x0, args=args, jac=jac, **options)
    elif meth == 'lm':
        sol = _root_leastsq(_wrapped_fun, x0, args=args, jac=jac, **options)
    elif meth == 'df-sane':
        _warn_jac_unused(jac, method)
        sol = _root_df_sane(_wrapped_fun, x0, args=args, callback=callback,
                            **options)
    elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
                  'diagbroyden', 'excitingmixing', 'krylov'):
        _warn_jac_unused(jac, method)
        sol = _root_nonlin_solve(_wrapped_fun, x0, args=args, jac=jac,
                                 _method=meth, _callback=callback,
                                 **options)
    else:
        raise ValueError(f'Unknown solver {method}')

    sol.nfev = _wrapped_fun.nfev
    return sol


def _warn_jac_unused(jac, method):
    if jac is not None:
        warn(f'Method {method} does not use the jacobian (jac).',
             RuntimeWarning, stacklevel=2)


def _root_leastsq(fun, x0, args=(), jac=None,
                  col_deriv=0, xtol=1.49012e-08, ftol=1.49012e-08,
                  gtol=0.0, maxiter=0, eps=0.0, factor=100, diag=None,
                  **unknown_options):
    """
    Solve for least squares with Levenberg-Marquardt

    Options
    -------
    col_deriv : bool
        non-zero to specify that the Jacobian function computes derivatives
        down the columns (faster, because there is no transpose operation).
    ftol : float
        Relative error desired in the sum of squares.
    xtol : float
        Relative error desired in the approximate solution.
    gtol : float
        Orthogonality desired between the function vector and the columns
        of the Jacobian.
    maxiter : int
        The maximum number of calls to the function. If zero, then
        100*(N+1) is the maximum where N is the number of elements in x0.
    eps : float
        A suitable step length for the forward-difference approximation of
        the Jacobian (for Dfun=None). If `eps` is less than the machine
        precision, it is assumed that the relative errors in the functions
        are of the order of the machine precision.
    factor : float
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
    diag : sequence
        N positive entries that serve as a scale factors for the variables.
    """
    nfev = 0
    def _wrapped_fun(*fargs):
        """
        Wrapped `func` to track the number of times
        the function has been called.
        """
        nonlocal nfev
        nfev += 1
        return fun(*fargs)

    _check_unknown_options(unknown_options)
    x, cov_x, info, msg, ier = leastsq(_wrapped_fun, x0, args=args,
                                       Dfun=jac, full_output=True,
                                       col_deriv=col_deriv, xtol=xtol,
                                       ftol=ftol, gtol=gtol,
                                       maxfev=maxiter, epsfcn=eps,
                                       factor=factor, diag=diag)
    sol = OptimizeResult(x=x, message=msg, status=ier,
                         success=ier in (1, 2, 3, 4), cov_x=cov_x,
                         fun=info.pop('fvec'), method="lm")
    sol.update(info)
    sol.nfev = nfev
    return sol


def _root_nonlin_solve(fun, x0, args=(), jac=None,
                       _callback=None, _method=None,
                       nit=None, disp=False, maxiter=None,
                       ftol=None, fatol=None, xtol=None, xatol=None,
                       tol_norm=None, line_search='armijo', jac_options=None,
                       **unknown_options):
    _check_unknown_options(unknown_options)

    f_tol = fatol
    f_rtol = ftol
    x_tol = xatol
    x_rtol = xtol
    verbose = disp
    if jac_options is None:
        jac_options = dict()

    jacobian = {'broyden1': nonlin.BroydenFirst,
                'broyden2': nonlin.BroydenSecond,
                'anderson': nonlin.Anderson,
                'linearmixing': nonlin.LinearMixing,
                'diagbroyden': nonlin.DiagBroyden,
                'excitingmixing': nonlin.ExcitingMixing,
                'krylov': nonlin.KrylovJacobian
                }[_method]

    if args:
        if jac is True:
            def f(x):
                return fun(x, *args)[0]
        else:
            def f(x):
                return fun(x, *args)
    else:
        f = fun

    x, info = nonlin.nonlin_solve(f, x0, jacobian=jacobian(**jac_options),
                                  iter=nit, verbose=verbose,
                                  maxiter=maxiter, f_tol=f_tol,
                                  f_rtol=f_rtol, x_tol=x_tol,
                                  x_rtol=x_rtol, tol_norm=tol_norm,
                                  line_search=line_search,
                                  callback=_callback, full_output=True,
                                  raise_exception=False)
    sol = OptimizeResult(x=x, method=_method)
    sol.update(info)
    return sol

def _root_broyden1_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            Initial guess for the Jacobian is (-1/alpha).
        reduction_method : str or tuple, optional
            Method used in ensuring that the rank of the Broyden
            matrix stays low. Can either be a string giving the
            name of the method, or a tuple of the form ``(method,
            param1, param2, ...)`` that gives the name of the
            method and values for additional parameters.

            Methods available:

            - ``restart``: drop all matrix columns. Has no extra parameters.
            - ``simple``: drop oldest matrix column. Has no extra parameters.
            - ``svd``: keep only the most significant SVD components.
              Takes an extra parameter, ``to_retain``, which determines the
              number of SVD components to retain when rank reduction is done.
              Default is ``max_rank - 2``.

        max_rank : int, optional
            Maximum rank for the Broyden matrix.
            Default is infinity (i.e., no rank reduction).

    Examples
    --------
    >>> def func(x):
    ...     return np.cos(x) + x[::-1] - [1, 2, 3, 4]
    ...
    >>> from scipy import optimize
    >>> res = optimize.root(func, [1, 1, 1, 1], method='broyden1', tol=1e-14)
    >>> x = res.x
    >>> x
    array([4.04674914, 3.91158389, 2.71791677, 1.61756251])
    >>> np.cos(x) + x[::-1]
    array([1., 2., 3., 4.])

    """
    pass


def _root_broyden2_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            Initial guess for the Jacobian is (-1/alpha).
        reduction_method : str or tuple, optional
            Method used in ensuring that the rank of the Broyden
            matrix stays low. Can either be a string giving the
            name of the method, or a tuple of the form ``(method,
            param1, param2, ...)`` that gives the name of the
            method and values for additional parameters.

            Methods available:

            - ``restart``: drop all matrix columns. Has no extra parameters.
            - ``simple``: drop oldest matrix column. Has no extra parameters.
            - ``svd``: keep only the most significant SVD components.
              Takes an extra parameter, ``to_retain``, which determines the
              number of SVD components to retain when rank reduction is done.
              Default is ``max_rank - 2``.

        max_rank : int, optional
            Maximum rank for the Broyden matrix.
            Default is infinity (i.e., no rank reduction).
    """
    pass


def _root_anderson_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            Initial guess for the Jacobian is (-1/alpha).
        M : float, optional
            Number of previous vectors to retain. Defaults to 5.
        w0 : float, optional
            Regularization parameter for numerical stability.
            Compared to unity, good values of the order of 0.01.
    """
    pass

def _root_linearmixing_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            initial guess for the jacobian is (-1/alpha).
    """
    pass

def _root_diagbroyden_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            initial guess for the jacobian is (-1/alpha).
    """
    pass

def _root_excitingmixing_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        alpha : float, optional
            Initial Jacobian approximation is (-1/alpha).
        alphamax : float, optional
            The entries of the diagonal Jacobian are kept in the range
            ``[alpha, alphamax]``.
    """
    pass

def _root_krylov_doc():
    """
    Options
    -------
    nit : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    disp : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make.
    ftol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    fatol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    xtol : float, optional
        Relative minimum step size. If omitted, not used.
    xatol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in
        the direction given by the Jacobian approximation. Defaults to
        'armijo'.
    jac_options : dict, optional
        Options for the respective Jacobian approximation.

        rdiff : float, optional
            Relative step size to use in numerical differentiation.
        method : str or callable, optional
            Krylov method to use to approximate the Jacobian.  Can be a string,
            or a function implementing the same interface as the iterative
            solvers in `scipy.sparse.linalg`. If a string, needs to be one of:
            ``'lgmres'``, ``'gmres'``, ``'bicgstab'``, ``'cgs'``, ``'minres'``,
            ``'tfqmr'``.

            The default is `scipy.sparse.linalg.lgmres`.
        inner_M : LinearOperator or InverseJacobian
            Preconditioner for the inner Krylov iteration.
            Note that you can use also inverse Jacobians as (adaptive)
            preconditioners. For example,

            >>> jac = BroydenFirst()
            >>> kjac = KrylovJacobian(inner_M=jac.inverse).

            If the preconditioner has a method named 'update', it will
            be called as ``update(x, f)`` after each nonlinear step,
            with ``x`` giving the current point, and ``f`` the current
            function value.
        inner_rtol, inner_atol, inner_callback, ...
            Parameters to pass on to the "inner" Krylov solver.

            For a full list of options, see the documentation for the
            solver you are using. By default this is `scipy.sparse.linalg.lgmres`.
            If the solver has been overridden through `method`, see the documentation
            for that solver instead.
            To use an option for that solver, prepend ``inner_`` to it.
            For example, to control the ``rtol`` argument to the solver,
            set the `inner_rtol` option here.

        outer_k : int, optional
            Size of the subspace kept across LGMRES nonlinear
            iterations.

            See `scipy.sparse.linalg.lgmres` for details.
    """
    pass
<|endoftext|>import numpy as np
from threading import Lock

from ._optimize import _check_unknown_options


COBYQA_LOCK = Lock()


def _minimize_cobyqa(fun, x0, args=(), bounds=None, constraints=(),
                     callback=None, disp=False, maxfev=None, maxiter=None,
                     f_target=-np.inf, feasibility_tol=1e-8,
                     initial_tr_radius=1.0, final_tr_radius=1e-6, scale=False,
                     **unknown_options):
    """
    Minimize a scalar function of one or more variables using the
    Constrained Optimization BY Quadratic Approximations (COBYQA) algorithm [1]_.

    .. versionadded:: 1.14.0

    Options
    -------
    disp : bool
        Set to True to print information about the optimization procedure.
        Default is ``False``.
    maxfev : int
        Maximum number of function evaluations. Default is ``500 * n``, where
        ``n`` is the number of variables.
    maxiter : int
        Maximum number of iterations. Default is ``1000 * n``, where ``n`` is
        the number of variables.
    f_target : float
        Target value for the objective function. The optimization procedure is
        terminated when the objective function value of a feasible point (see
        `feasibility_tol` below) is less than or equal to this target. Default
        is ``-numpy.inf``.
    feasibility_tol : float
        Absolute tolerance for the constraint violation. Default is ``1e-8``.
    initial_tr_radius : float
        Initial trust-region radius. Typically, this value should be in the
        order of one tenth of the greatest expected change to the variables.
        Default is ``1.0``.
    final_tr_radius : float
        Final trust-region radius. It should indicate the accuracy required in
        the final values of the variables. If provided, this option overrides
        the value of `tol` in the `minimize` function. Default is ``1e-6``.
    scale : bool
        Set to True to scale the variables according to the bounds. If True and
        if all the lower and upper bounds are finite, the variables are scaled
        to be within the range :math:`[-1, 1]`. If any of the lower or upper
        bounds is infinite, the variables are not scaled. Default is ``False``.

    References
    ----------
    .. [1] COBYQA
           https://www.cobyqa.com/stable/
    """
    from .._lib.cobyqa import minimize  # import here to avoid circular imports

    _check_unknown_options(unknown_options)
    options = {
        'disp': bool(disp),
        'maxfev': int(maxfev) if maxfev is not None else 500 * len(x0),
        'maxiter': int(maxiter) if maxiter is not None else 1000 * len(x0),
        'target': float(f_target),
        'feasibility_tol': float(feasibility_tol),
        'radius_init': float(initial_tr_radius),
        'radius_final': float(final_tr_radius),
        'scale': bool(scale),
    }
    with COBYQA_LOCK:
        return minimize(fun, x0, args, bounds, constraints, callback, options)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'OptimizeResult',
    'fmin_slsqp',
    'slsqp',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="slsqp",
                                   private_modules=["_slsqp_py"], all=__all__,
                                   attribute=name)
<|endoftext|>from warnings import warn

import numpy as np
import scipy.optimize._highspy._core as _h # type: ignore[import-not-found]
from scipy.optimize._highspy import _highs_options as hopt  # type: ignore[attr-defined]
from scipy.optimize import OptimizeWarning


def _highs_wrapper(c, indptr, indices, data, lhs, rhs, lb, ub, integrality, options):
    '''Solve linear programs using HiGHS [1]_.

    Assume problems of the form:

        MIN c.T @ x
        s.t. lhs <= A @ x <= rhs
             lb <= x <= ub

    Parameters
    ----------
    c : 1-D array, (n,)
        Array of objective value coefficients.
    astart : 1-D array
        CSC format index array.
    aindex : 1-D array
        CSC format index array.
    avalue : 1-D array
        Data array of the matrix.
    lhs : 1-D array (or None), (m,)
        Array of left hand side values of the inequality constraints.
        If ``lhs=None``, then an array of ``-inf`` is assumed.
    rhs : 1-D array, (m,)
        Array of right hand side values of the inequality constraints.
    lb : 1-D array (or None), (n,)
        Lower bounds on solution variables x.  If ``lb=None``, then an
        array of all `0` is assumed.
    ub : 1-D array (or None), (n,)
        Upper bounds on solution variables x.  If ``ub=None``, then an
        array of ``inf`` is assumed.
    options : dict
        A dictionary of solver options

    Returns
    -------
    res : dict

        If model_status is one of kOptimal,
        kObjectiveBound, kTimeLimit,
        kIterationLimit:

            - ``status`` : HighsModelStatus
                Model status code.

            - ``message`` : str
                Message corresponding to model status code.

            - ``x`` : list
                Solution variables.

            - ``slack`` : list
                Slack variables.

            - ``lambda`` : list
                Lagrange multipliers associated with the constraints
                Ax = b.

            - ``s`` : list
                Lagrange multipliers associated with the constraints
                x >= 0.

            - ``fun``
                Final objective value.

            - ``simplex_nit`` : int
                Number of iterations accomplished by the simplex
                solver.

            - ``ipm_nit`` : int
                Number of iterations accomplished by the interior-
                point solver.

        If model_status is not one of the above:

            - ``status`` : HighsModelStatus
                Model status code.

            - ``message`` : str
                Message corresponding to model status code.

    Notes
    -----
    If ``options['write_solution_to_file']`` is ``True`` but
    ``options['solution_file']`` is unset or ``''``, then the solution
    will be printed to ``stdout``.

    If any iteration limit is reached, no solution will be
    available.

    ``OptimizeWarning`` will be raised if any option value set by
    the user is found to be incorrect.

    References
    ----------
    .. [1] https://highs.dev/
    .. [2] https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.html
    '''
    numcol = c.size
    numrow = rhs.size
    isMip = integrality is not None and np.sum(integrality) > 0

    # default "null" return values
    res = {
        "x": None,
        "fun": None,
    }

    # Fill up a HighsLp object
    lp = _h.HighsLp()
    lp.num_col_ = numcol
    lp.num_row_ = numrow
    lp.a_matrix_.num_col_ = numcol
    lp.a_matrix_.num_row_ = numrow
    lp.a_matrix_.format_ = _h.MatrixFormat.kColwise
    lp.col_cost_ = c
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = lhs
    lp.row_upper_ = rhs
    lp.a_matrix_.start_ = indptr
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = data
    if integrality.size > 0:
        lp.integrality_ = [_h.HighsVarType(i) for i in integrality]

    # Make a Highs object and pass it everything
    highs = _h._Highs()
    highs_options = _h.HighsOptions()
    hoptmanager = hopt.HighsOptionsManager()
    for key, val in options.items():
        # handle filtering of unsupported and default options
        if val is None or key in ("sense",):
            continue

        # ask for the option type
        opt_type = hoptmanager.get_option_type(key)
        if -1 == opt_type:
            warn(
                f"Unrecognized options detected: {dict({key: val})}",
                OptimizeWarning,
                stacklevel=2,
            )
            continue
        else:
            if key in ("presolve", "parallel"):
                # handle fake bools (require bool -> str conversions)
                if isinstance(val, bool):
                    val = "on" if val else "off"
                else:
                    warn(
                        f'Option f"{key}" is "{val}", but only True or False is '
                        f"allowed. Using default.",
                        OptimizeWarning,
                        stacklevel=2,
                    )
                    continue
            opt_type = _h.HighsOptionType(opt_type)
            status, msg = check_option(highs, key, val)
            if opt_type == _h.HighsOptionType.kBool:
                if not isinstance(val, bool):
                    warn(
                        f'Option f"{key}" is "{val}", but only True or False is '
                        f"allowed. Using default.",
                        OptimizeWarning,
                        stacklevel=2,
                    )
                    continue

            # warn or set option
            if status != 0:
                warn(msg, OptimizeWarning, stacklevel=2)
            else:
                setattr(highs_options, key, val)

    opt_status = highs.passOptions(highs_options)
    if opt_status == _h.HighsStatus.kError:
        res.update(
            {
                "status": highs.getModelStatus(),
                "message": highs.modelStatusToString(highs.getModelStatus()),
            }
        )
        return res

    init_status = highs.passModel(lp)
    if init_status == _h.HighsStatus.kError:
        # if model fails to load, highs.getModelStatus() will be NOT_SET
        err_model_status = _h.HighsModelStatus.kModelError
        res.update(
            {
                "status": err_model_status,
                "message": highs.modelStatusToString(err_model_status),
            }
        )
        return res

    # Solve the LP
    run_status = highs.run()
    if run_status == _h.HighsStatus.kError:
        res.update(
            {
                "status": highs.getModelStatus(),
                "message": highs.modelStatusToString(highs.getModelStatus()),
            }
        )
        return res

    # Extract what we need from the solution
    model_status = highs.getModelStatus()

    # it should always be safe to get the info object
    info = highs.getInfo()

    # Failure modes:
    #     LP: if we have anything other than an Optimal status, it
    #         is unsafe (and unhelpful) to read any results
    #    MIP: has a non-Optimal status or has timed out/reached max iterations
    #             1) If not Optimal/TimedOut/MaxIter status, there is no solution
    #             2) If TimedOut/MaxIter status, there may be a feasible solution.
    #                if the objective function value is not Infinity, then the
    #                current solution is feasible and can be returned.  Else, there
    #                is no solution.
    mipFailCondition = model_status not in (
        _h.HighsModelStatus.kOptimal,
        _h.HighsModelStatus.kTimeLimit,
        _h.HighsModelStatus.kIterationLimit,
        _h.HighsModelStatus.kSolutionLimit,
    ) or (
        model_status
        in {
            _h.HighsModelStatus.kTimeLimit,
            _h.HighsModelStatus.kIterationLimit,
            _h.HighsModelStatus.kSolutionLimit,
        }
        and (info.objective_function_value == _h.kHighsInf)
    )
    lpFailCondition = model_status != _h.HighsModelStatus.kOptimal
    if (isMip and mipFailCondition) or (not isMip and lpFailCondition):
        res.update(
            {
                "status": model_status,
                "message": "model_status is "
                f"{highs.modelStatusToString(model_status)}; "
                "primal_status is "
                f"{highs.solutionStatusToString(info.primal_solution_status)}",
                "simplex_nit": info.simplex_iteration_count,
                "ipm_nit": info.ipm_iteration_count,
                "crossover_nit": info.crossover_iteration_count,
            }
        )
        return res

    # Should be safe to read the solution:
    solution = highs.getSolution()
    basis = highs.getBasis()

    # Lagrangians for bounds based on column statuses
    marg_bnds = np.zeros((2, numcol))
    basis_col_status = basis.col_status
    solution_col_dual = solution.col_dual
    for ii in range(numcol):
        if basis_col_status[ii] == _h.HighsBasisStatus.kLower:
            marg_bnds[0, ii] = solution_col_dual[ii]
        elif basis_col_status[ii] == _h.HighsBasisStatus.kUpper:
            marg_bnds[1, ii] = solution_col_dual[ii]

    res.update(
        {
            "status": model_status,
            "message": highs.modelStatusToString(model_status),
            # Primal solution
            "x": np.array(solution.col_value),
            # Ax + s = b => Ax = b - s
            # Note: this is for all constraints (A_ub and A_eq)
            "slack": rhs - solution.row_value,
            # lambda are the lagrange multipliers associated with Ax=b
            "lambda": np.array(solution.row_dual),
            "marg_bnds": marg_bnds,
            "fun": info.objective_function_value,
            "simplex_nit": info.simplex_iteration_count,
            "ipm_nit": info.ipm_iteration_count,
            "crossover_nit": info.crossover_iteration_count,
        }
    )

    if isMip:
        res.update(
            {
                "mip_node_count": info.mip_node_count,
                "mip_dual_bound": info.mip_dual_bound,
                "mip_gap": info.mip_gap,
            }
        )

    return res


def check_option(highs_inst, option, value):
    status, option_type = highs_inst.getOptionType(option)
    hoptmanager = hopt.HighsOptionsManager()

    if status != _h.HighsStatus.kOk:
        return -1, "Invalid option name."

    valid_types = {
        _h.HighsOptionType.kBool: bool,
        _h.HighsOptionType.kInt: int,
        _h.HighsOptionType.kDouble: float,
        _h.HighsOptionType.kString: str,
    }

    expected_type = valid_types.get(option_type, None)

    if expected_type is str:
        if not hoptmanager.check_string_option(option, value):
            return -1, "Invalid option value."
    if expected_type is float:
        if not hoptmanager.check_double_option(option, value):
            return -1, "Invalid option value."
    if expected_type is int:
        if not hoptmanager.check_int_option(option, value):
            return -1, "Invalid option value."

    if expected_type is None:
        return 3, "Unknown option type."

    status, current_value = highs_inst.getOptionValue(option)
    if status != _h.HighsStatus.kOk:
        return 4, "Failed to validate option value."
    return 0, "Check option succeeded."
<|endoftext|>from ._trlib import TRLIBQuadraticSubproblem

__all__ = ['TRLIBQuadraticSubproblem', 'get_trlib_quadratic_subproblem']


def get_trlib_quadratic_subproblem(tol_rel_i=-2.0, tol_rel_b=-3.0, disp=False):
    def subproblem_factory(x, fun, jac, hess, hessp):
        return TRLIBQuadraticSubproblem(x, fun, jac, hess, hessp,
                                        tol_rel_i=tol_rel_i,
                                        tol_rel_b=tol_rel_b,
                                        disp=disp)
    return subproblem_factory
<|endoftext|>import collections
from abc import ABC, abstractmethod

import numpy as np

from scipy._lib._util import MapWrapper


class VertexBase(ABC):
    """
    Base class for a vertex.
    """
    def __init__(self, x, nn=None, index=None):
        """
        Initiation of a vertex object.

        Parameters
        ----------
        x : tuple or vector
            The geometric location (domain).
        nn : list, optional
            Nearest neighbour list.
        index : int, optional
            Index of vertex.
        """
        self.x = x
        self.hash = hash(self.x)  # Save precomputed hash

        if nn is not None:
            self.nn = set(nn)  # can use .indexupdate to add a new list
        else:
            self.nn = set()

        self.index = index

    def __hash__(self):
        return self.hash

    def __getattr__(self, item):
        if item not in ['x_a']:
            raise AttributeError(f"{type(self)} object has no attribute "
                                 f"'{item}'")
        if item == 'x_a':
            self.x_a = np.array(self.x)
            return self.x_a

    @abstractmethod
    def connect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    @abstractmethod
    def disconnect(self, v):
        raise NotImplementedError("This method is only implemented with an "
                                  "associated child of the base class.")

    def star(self):
        """Returns the star domain ``st(v)`` of the vertex.

        Parameters
        ----------
        v :
            The vertex ``v`` in ``st(v)``

        Returns
        -------
        st : set
            A set containing all the vertices in ``st(v)``
        """
        self.st = self.nn
        self.st.add(self)
        return self.st


class VertexScalarField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R associated with
    the geometry built from the VertexBase class
    """

    def __init__(self, x, field=None, nn=None, index=None, field_args=(),
                 g_cons=None, g_cons_args=()):
        """
        Parameters
        ----------
        x : tuple,
            vector of vertex coordinates
        field : callable, optional
            a scalar field f: R^n --> R associated with the geometry
        nn : list, optional
            list of nearest neighbours
        index : int, optional
            index of the vertex
        field_args : tuple, optional
            additional arguments to be passed to field
        g_cons : callable, optional
            constraints on the vertex
        g_cons_args : tuple, optional
            additional arguments to be passed to g_cons

        """
        super().__init__(x, nn=nn, index=index)

        # Note Vertex is only initiated once for all x so only
        # evaluated once
        # self.feasible = None

        # self.f is externally defined by the cache to allow parallel
        # processing
        # None type that will break arithmetic operations unless defined
        # self.f = None

        self.check_min = True
        self.check_max = True

    def connect(self, v):
        """Connects self to another vertex object v.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

            # Flags for checking homology properties:
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)

            # Flags for checking homology properties:
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def minimiser(self):
        """Check whether this vertex is strictly less than all its
           neighbours"""
        if self.check_min:
            self._min = all(self.f < v.f for v in self.nn)
            self.check_min = False

        return self._min

    def maximiser(self):
        """
        Check whether this vertex is strictly greater than all its
        neighbours.
        """
        if self.check_max:
            self._max = all(self.f > v.f for v in self.nn)
            self.check_max = False

        return self._max


class VertexVectorField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R^m associated with
    the geometry built from the VertexBase class.
    """

    def __init__(self, x, sfield=None, vfield=None, field_args=(),
                 vfield_args=(), g_cons=None,
                 g_cons_args=(), nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

        raise NotImplementedError("This class is still a work in progress")


class VertexCacheBase:
    """Base class for a vertex cache for a simplicial complex."""
    def __init__(self):

        self.cache = collections.OrderedDict()
        self.nfev = 0  # Feasible points
        self.index = -1

    def __iter__(self):
        for v in self.cache:
            yield self.cache[v]
        return

    def size(self):
        """Returns the size of the vertex cache."""
        return self.index + 1

    def print_out(self):
        headlen = len(f"Vertex cache of size: {len(self.cache)}:")
        print('=' * headlen)
        print(f"Vertex cache of size: {len(self.cache)}:")
        print('=' * headlen)
        for v in self.cache:
            self.cache[v].print_out()


class VertexCube(VertexBase):
    """Vertex class to be used for a pure simplicial complex with no associated
    differential geometry (single level domain that exists in R^n)"""
    def __init__(self, x, nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)


class VertexCacheIndex(VertexCacheBase):
    def __init__(self):
        """
        Class for a vertex cache for a simplicial complex without an associated
        field. Useful only for building and visualising a domain complex.

        Parameters
        ----------
        """
        super().__init__()
        self.Vertex = VertexCube

    def __getitem__(self, x, nn=None):
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, index=self.index)
            # logging.info("New generated vertex at x = {}".format(x))
            # NOTE: Surprisingly high performance increase if logging
            # is commented out
            self.cache[x] = xval
            return self.cache[x]


class VertexCacheField(VertexCacheBase):
    def __init__(self, field=None, field_args=(), g_cons=None, g_cons_args=(),
                 workers=1):
        """
        Class for a vertex cache for a simplicial complex with an associated
        field.

        Parameters
        ----------
        field : callable
            Scalar or vector field callable.
        field_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            field function
        g_cons : dict or sequence of dict, optional
            Constraints definition.
            Function(s) ``R**n`` in the form::
        g_cons_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            constraint functions
        workers : int  optional
            Uses `multiprocessing.Pool <multiprocessing>`) to compute the field
             functions in parallel.

        """
        super().__init__()
        self.index = -1
        self.Vertex = VertexScalarField
        self.field = field
        self.field_args = field_args
        self.wfield = FieldWrapper(field, field_args)  # if workers is not 1

        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.wgcons = ConstraintWrapper(g_cons, g_cons_args)
        self.gpool = set()  # A set of tuples to process for feasibility

        # Field processing objects
        self.fpool = set()  # A set of tuples to process for scalar function
        self.sfc_lock = False  # True if self.fpool is non-Empty

        self.workers = workers
        self._mapwrapper = MapWrapper(workers)

        if workers == 1:
            self.process_gpool = self.proc_gpool
            if g_cons is None:
                self.process_fpool = self.proc_fpool_nog
            else:
                self.process_fpool = self.proc_fpool_g
        else:
            self.process_gpool = self.pproc_gpool
            if g_cons is None:
                self.process_fpool = self.pproc_fpool_nog
            else:
                self.process_fpool = self.pproc_fpool_g

    def __getitem__(self, x, nn=None):
        try:
            return self.cache[x]
        except KeyError:
            self.index += 1
            xval = self.Vertex(x, field=self.field, nn=nn, index=self.index,
                               field_args=self.field_args,
                               g_cons=self.g_cons,
                               g_cons_args=self.g_cons_args)

            self.cache[x] = xval  # Define in cache
            self.gpool.add(xval)  # Add to pool for processing feasibility
            self.fpool.add(xval)  # Add to pool for processing field values
            return self.cache[x]

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def process_pools(self):
        if self.g_cons is not None:
            self.process_gpool()
        self.process_fpool()
        self.proc_minimisers()

    def feasibility_check(self, v):
        v.feasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            # constraint may return more than 1 value.
            if np.any(g(v.x_a, *args) < 0.0):
                v.f = np.inf
                v.feasible = False
                break

    def compute_sfield(self, v):
        """Compute the scalar field values of a vertex object `v`.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        try:
            v.f = self.field(v.x_a, *self.field_args)
            self.nfev += 1
        except AttributeError:
            v.f = np.inf
            # logging.warning(f"Field function not found at x = {self.x_a}")
        if np.isnan(v.f):
            v.f = np.inf

    def proc_gpool(self):
        """Process all constraints."""
        if self.g_cons is not None:
            for v in self.gpool:
                self.feasibility_check(v)
        # Clean the pool
        self.gpool = set()

    def pproc_gpool(self):
        """Process all constraints in parallel."""
        gpool_l = []
        for v in self.gpool:
            gpool_l.append(v.x_a)

        G = self._mapwrapper(self.wgcons.gcons, gpool_l)
        for v, g in zip(self.gpool, G):
            v.feasible = g  # set vertex object attribute v.feasible = g (bool)

    def proc_fpool_g(self):
        """Process all field functions with constraints supplied."""
        for v in self.fpool:
            if v.feasible:
                self.compute_sfield(v)
        # Clean the pool
        self.fpool = set()

    def proc_fpool_nog(self):
        """Process all field functions with no constraints supplied."""
        for v in self.fpool:
            self.compute_sfield(v)
        # Clean the pool
        self.fpool = set()

    def pproc_fpool_g(self):
        """
        Process all field functions with constraints supplied in parallel.
        """
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            if v.feasible:
                fpool_l.append(v.x_a)
            else:
                v.f = np.inf
        F = self._mapwrapper(self.wfield.func, fpool_l)
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f  # set vertex object attribute v.f = f
            self.nfev += 1
        # Clean the pool
        self.fpool = set()

    def pproc_fpool_nog(self):
        """
        Process all field functions with no constraints supplied in parallel.
        """
        self.wfield.func
        fpool_l = []
        for v in self.fpool:
            fpool_l.append(v.x_a)
        F = self._mapwrapper(self.wfield.func, fpool_l)
        for va, f in zip(fpool_l, F):
            vt = tuple(va)
            self[vt].f = f  # set vertex object attribute v.f = f
            self.nfev += 1
        # Clean the pool
        self.fpool = set()

    def proc_minimisers(self):
        """Check for minimisers."""
        for v in self:
            v.minimiser()
            v.maximiser()


class ConstraintWrapper:
    """Object to wrap constraints to pass to `multiprocessing.Pool`."""
    def __init__(self, g_cons, g_cons_args):
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args

    def gcons(self, v_x_a):
        vfeasible = True
        for g, args in zip(self.g_cons, self.g_cons_args):
            # constraint may return more than 1 value.
            if np.any(g(v_x_a, *args) < 0.0):
                vfeasible = False
                break
        return vfeasible


class FieldWrapper:
    """Object to wrap field to pass to `multiprocessing.Pool`."""
    def __init__(self, field, field_args):
        self.field = field
        self.field_args = field_args

    def func(self, v_x_a):
        try:
            v_f = self.field(v_x_a, *self.field_args)
        except Exception:
            v_f = np.inf
        if np.isnan(v_f):
            v_f = np.inf

        return v_f
<|endoftext|>"""Base classes for low memory simplicial complex structures."""
import copy
import logging
import itertools
import decimal
from functools import cache

import numpy as np

from ._vertex import (VertexCacheField, VertexCacheIndex)


class Complex:
    """
    Base class for a simplicial complex described as a cache of vertices
    together with their connections.

    Important methods:
        Domain triangulation:
                Complex.triangulate, Complex.split_generation
        Triangulating arbitrary points (must be traingulable,
            may exist outside domain):
                Complex.triangulate(sample_set)
        Converting another simplicial complex structure data type to the
            structure used in Complex (ex. OBJ wavefront)
                Complex.convert(datatype, data)

    Important objects:
        HC.V: The cache of vertices and their connection
        HC.H: Storage structure of all vertex groups

    Parameters
    ----------
    dim : int
        Spatial dimensionality of the complex R^dim
    domain : list of tuples, optional
        The bounds [x_l, x_u]^dim of the hyperrectangle space
        ex. The default domain is the hyperrectangle [0, 1]^dim
        Note: The domain must be convex, non-convex spaces can be cut
              away from this domain using the non-linear
              g_cons functions to define any arbitrary domain
              (these domains may also be disconnected from each other)
    sfield :
        A scalar function defined in the associated domain f: R^dim --> R
    sfield_args : tuple
        Additional arguments to be passed to `sfield`
    vfield :
        A scalar function defined in the associated domain
                       f: R^dim --> R^m
                   (for example a gradient function of the scalar field)
    vfield_args : tuple
        Additional arguments to be passed to vfield
    symmetry : None or list
            Specify if the objective function contains symmetric variables.
            The search space (and therefore performance) is decreased by up to
            O(n!) times in the fully symmetric case.

            E.g.  f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            In this equation x_2 and x_3 are symmetric to x_1, while x_5 and
             x_6 are symmetric to x_4, this can be specified to the solver as:

            symmetry = [0,  # Variable 1
                        0,  # symmetric to variable 1
                        0,  # symmetric to variable 1
                        3,  # Variable 4
                        3,  # symmetric to variable 4
                        3,  # symmetric to variable 4
                        ]

    constraints : dict or sequence of dict, optional
        Constraints definition.
        Function(s) ``R**n`` in the form::

            g(x) <= 0 applied as g : R^n -> R^m
            h(x) == 0 applied as h : R^n -> R^p

        Each constraint is defined in a dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be
        non-negative.constraints : dict or sequence of dict, optional
        Constraints definition.
        Function(s) ``R**n`` in the form::

            g(x) <= 0 applied as g : R^n -> R^m
            h(x) == 0 applied as h : R^n -> R^p

        Each constraint is defined in a dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (unused).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.

    workers : int  optional
        Uses `multiprocessing.Pool <multiprocessing>`) to compute the field
         functions in parallel.
    """
    def __init__(self, dim, domain=None, sfield=None, sfield_args=(),
                 symmetry=None, constraints=None, workers=1):
        self.dim = dim

        # Domains
        self.domain = domain
        if domain is None:
            self.bounds = [(0.0, 1.0), ] * dim
        else:
            self.bounds = domain
        self.symmetry = symmetry
        #      here in init to avoid if checks

        # Field functions
        self.sfield = sfield
        self.sfield_args = sfield_args

        # Process constraints
        # Constraints
        # Process constraint dict sequence:
        if constraints is not None:
            self.min_cons = constraints
            self.g_cons = []
            self.g_args = []
            if not isinstance(constraints, (tuple, list)):
                constraints = (constraints,)

            for cons in constraints:
                if cons['type'] in ('ineq'):
                    self.g_cons.append(cons['fun'])
                    try:
                        self.g_args.append(cons['args'])
                    except KeyError:
                        self.g_args.append(())
            self.g_cons = tuple(self.g_cons)
            self.g_args = tuple(self.g_args)
        else:
            self.g_cons = None
            self.g_args = None

        # Homology properties
        self.gen = 0
        self.perm_cycle = 0

        # Every cell is stored in a list of its generation,
        # ex. the initial cell is stored in self.H[0]
        # 1st get new cells are stored in self.H[1] etc.
        # When a cell is sub-generated it is removed from this list

        self.H = []  # Storage structure of vertex groups

        # Cache of all vertices
        if (sfield is not None) or (self.g_cons is not None):
            # Initiate a vertex cache and an associated field cache, note that
            # the field case is always initiated inside the vertex cache if an
            # associated field scalar field is defined:
            if sfield is not None:
                self.V = VertexCacheField(field=sfield, field_args=sfield_args,
                                          g_cons=self.g_cons,
                                          g_cons_args=self.g_args,
                                          workers=workers)
            elif self.g_cons is not None:
                self.V = VertexCacheField(field=sfield, field_args=sfield_args,
                                          g_cons=self.g_cons,
                                          g_cons_args=self.g_args,
                                          workers=workers)
        else:
            self.V = VertexCacheIndex()

        self.V_non_symm = []  # List of non-symmetric vertices

    def __call__(self):
        return self.H

    # %% Triangulation methods
    def cyclic_product(self, bounds, origin, supremum, centroid=True):
        """Generate initial triangulation using cyclic product"""
        # Define current hyperrectangle
        vot = tuple(origin)
        vut = tuple(supremum)  # Hyperrectangle supremum
        self.V[vot]
        vo = self.V[vot]
        yield vo.x
        self.V[vut].connect(self.V[vot])
        yield vut
        # Cyclic group approach with second x_l --- x_u operation.

        # These containers store the "lower" and "upper" vertices
        # corresponding to the origin or supremum of every C2 group.
        # It has the structure of `dim` times embedded lists each containing
        # these vertices as the entire complex grows. Bounds[0] has to be done
        # outside the loops before we have symmetric containers.
        # NOTE: This means that bounds[0][1] must always exist
        C0x = [[self.V[vot]]]
        a_vo = copy.copy(list(origin))
        a_vo[0] = vut[0]  # Update aN Origin
        a_vo = self.V[tuple(a_vo)]
        # self.V[vot].connect(self.V[tuple(a_vo)])
        self.V[vot].connect(a_vo)
        yield a_vo.x
        C1x = [[a_vo]]
        # C1x = [[self.V[tuple(a_vo)]]]
        ab_C = []  # Container for a + b operations

        # Loop over remaining bounds
        for i, x in enumerate(bounds[1:]):
            # Update lower and upper containers
            C0x.append([])
            C1x.append([])
            # try to access a second bound (if not, C1 is symmetric)
            try:
                # Early try so that we don't have to copy the cache before
                # moving on to next C1/C2: Try to add the operation of a new
                # C2 product by accessing the upper bound
                x[1]
                # Copy lists for iteration
                cC0x = [x[:] for x in C0x[:i + 1]]
                cC1x = [x[:] for x in C1x[:i + 1]]
                for j, (VL, VU) in enumerate(zip(cC0x, cC1x)):
                    for k, (vl, vu) in enumerate(zip(VL, VU)):
                        # Build aN vertices for each lower-upper pair in N:
                        a_vl = list(vl.x)
                        a_vu = list(vu.x)
                        a_vl[i + 1] = vut[i + 1]
                        a_vu[i + 1] = vut[i + 1]
                        a_vl = self.V[tuple(a_vl)]

                        # Connect vertices in N to corresponding vertices
                        # in aN:
                        vl.connect(a_vl)

                        yield a_vl.x

                        a_vu = self.V[tuple(a_vu)]
                        # Connect vertices in N to corresponding vertices
                        # in aN:
                        vu.connect(a_vu)

                        # Connect new vertex pair in aN:
                        a_vl.connect(a_vu)

                        # Connect lower pair to upper (triangulation
                        # operation of a + b (two arbitrary operations):
                        vl.connect(a_vu)
                        ab_C.append((vl, a_vu))

                        # Update the containers
                        C0x[i + 1].append(vl)
                        C0x[i + 1].append(vu)
                        C1x[i + 1].append(a_vl)
                        C1x[i + 1].append(a_vu)

                        # Update old containers
                        C0x[j].append(a_vl)
                        C1x[j].append(a_vu)

                        # Yield new points
                        yield a_vu.x

                # Try to connect aN lower source of previous a + b
                # operation with a aN vertex
                ab_Cc = copy.copy(ab_C)

                for vp in ab_Cc:
                    b_v = list(vp[0].x)
                    ab_v = list(vp[1].x)
                    b_v[i + 1] = vut[i + 1]
                    ab_v[i + 1] = vut[i + 1]
                    b_v = self.V[tuple(b_v)]  # b + vl
                    ab_v = self.V[tuple(ab_v)]  # b + a_vl
                    # Note o---o is already connected
                    vp[0].connect(ab_v)  # o-s
                    b_v.connect(ab_v)  # s-s

                    # Add new list of cross pairs
                    ab_C.append((vp[0], ab_v))
                    ab_C.append((b_v, ab_v))

            except IndexError:
                cC0x = C0x[i]
                cC1x = C1x[i]
                VL, VU = cC0x, cC1x
                for k, (vl, vu) in enumerate(zip(VL, VU)):
                    # Build aN vertices for each lower-upper pair in N:
                    a_vu = list(vu.x)
                    a_vu[i + 1] = vut[i + 1]
                    # Connect vertices in N to corresponding vertices
                    # in aN:
                    a_vu = self.V[tuple(a_vu)]
                    # Connect vertices in N to corresponding vertices
                    # in aN:
                    vu.connect(a_vu)
                    # Connect new vertex pair in aN:
                    # a_vl.connect(a_vu)
                    # Connect lower pair to upper (triangulation
                    # operation of a + b (two arbitrary operations):
                    vl.connect(a_vu)
                    ab_C.append((vl, a_vu))
                    C0x[i + 1].append(vu)
                    C1x[i + 1].append(a_vu)
                    # Yield new points
                    a_vu.connect(self.V[vut])
                    yield a_vu.x
                    ab_Cc = copy.copy(ab_C)
                    for vp in ab_Cc:
                        if vp[1].x[i] == vut[i]:
                            ab_v = list(vp[1].x)
                            ab_v[i + 1] = vut[i + 1]
                            ab_v = self.V[tuple(ab_v)]  # b + a_vl
                            # Note o---o is already connected
                            vp[0].connect(ab_v)  # o-s

                            # Add new list of cross pairs
                            ab_C.append((vp[0], ab_v))

        # Clean class trash
        try:
            del C0x
            del cC0x
            del C1x
            del cC1x
            del ab_C
            del ab_Cc
        except UnboundLocalError:
            pass

        # Extra yield to ensure that the triangulation is completed
        if centroid:
            vo = self.V[vot]
            vs = self.V[vut]
            # Disconnect the origin and supremum
            vo.disconnect(vs)
            # Build centroid
            vc = self.split_edge(vot, vut)
            for v in vo.nn:
                v.connect(vc)
            yield vc.x
            return vc.x
        else:
            yield vut
            return vut

    def triangulate(self, n=None, symmetry=None, centroid=True,
                    printout=False):
        """
        Triangulate the initial domain, if n is not None then a limited number
        of points will be generated

        Parameters
        ----------
        n : int, Number of points to be sampled.
        symmetry :

            Ex. Dictionary/hashtable
            f(x) = (x_1 + x_2 + x_3) + (x_4)**2 + (x_5)**2 + (x_6)**2

            symmetry = symmetry[0]: 0,  # Variable 1
                       symmetry[1]: 0,  # symmetric to variable 1
                       symmetry[2]: 0,  # symmetric to variable 1
                       symmetry[3]: 3,  # Variable 4
                       symmetry[4]: 3,  # symmetric to variable 4
                       symmetry[5]: 3,  # symmetric to variable 4
                        }
        centroid : bool, if True add a central point to the hypercube
        printout : bool, if True print out results

        NOTES:
        ------
        Rather than using the combinatorial algorithm to connect vertices we
        make the following observation:

        The bound pairs are similar a C2 cyclic group and the structure is
        formed using the cartesian product:

        H = C2 x C2 x C2 ... x C2 (dim times)

        So construct any normal subgroup N and consider H/N first, we connect
        all vertices within N (ex. N is C2 (the first dimension), then we move
        to a left coset aN (an operation moving around the defined H/N group by
        for example moving from the lower bound in C2 (dimension 2) to the
        higher bound in C2. During this operation connection all the vertices.
        Now repeat the N connections. Note that these elements can be connected
        in parallel.
        """
        # Inherit class arguments
        if symmetry is None:
            symmetry = self.symmetry
        # Build origin and supremum vectors
        origin = [i[0] for i in self.bounds]
        self.origin = origin
        supremum = [i[1] for i in self.bounds]

        self.supremum = supremum

        if symmetry is None:
            cbounds = self.bounds
        else:
            cbounds = copy.copy(self.bounds)
            for i, j in enumerate(symmetry):
                if i is not j:
                    # pop second entry on second symmetry vars
                    cbounds[i] = [self.bounds[symmetry[i]][0]]
                    # Sole (first) entry is the sup value and there is no
                    # origin:
                    cbounds[i] = [self.bounds[symmetry[i]][1]]
                    if (self.bounds[symmetry[i]] is not
                            self.bounds[symmetry[j]]):
                        logging.warning(f"Variable {i} was specified as "
                                        f"symmetric to variable {j}, however"
                                        f", the bounds {i} ="
                                        f" {self.bounds[symmetry[i]]} and {j}"
                                        f" ="
                                        f" {self.bounds[symmetry[j]]} do not "
                                        f"match, the mismatch was ignored in "
                                        f"the initial triangulation.")
                        cbounds[i] = self.bounds[symmetry[j]]

        if n is None:
            # Build generator
            self.cp = self.cyclic_product(cbounds, origin, supremum, centroid)
            for i in self.cp:
                i

            try:
                self.triangulated_vectors.append((tuple(self.origin),
                                                  tuple(self.supremum)))
            except (AttributeError, KeyError):
                self.triangulated_vectors = [(tuple(self.origin),
                                              tuple(self.supremum))]

        else:
            # Check if generator already exists
            try:
                self.cp
            except (AttributeError, KeyError):
                self.cp = self.cyclic_product(cbounds, origin, supremum,
                                              centroid)

            try:
                while len(self.V.cache) < n:
                    next(self.cp)
            except StopIteration:
                try:
                    self.triangulated_vectors.append((tuple(self.origin),
                                                      tuple(self.supremum)))
                except (AttributeError, KeyError):
                    self.triangulated_vectors = [(tuple(self.origin),
                                                  tuple(self.supremum))]

        if printout:
            # for v in self.C0():
            #   v.print_out()
            for v in self.V.cache:
                self.V[v].print_out()

        return

    def refine(self, n=1):
        if n is None:
            try:
                self.triangulated_vectors
                self.refine_all()
                return
            except AttributeError as ae:
                if str(ae) == "'Complex' object has no attribute " \
                              "'triangulated_vectors'":
                    self.triangulate(symmetry=self.symmetry)
                    return
                else:
                    raise

        nt = len(self.V.cache) + n  # Target number of total vertices
        # In the outer while loop we iterate until we have added an extra `n`
        # vertices to the complex:
        while len(self.V.cache) < nt:  # while loop 1
            try:  # try 1
                # Try to access triangulated_vectors, this should only be
                # defined if an initial triangulation has already been
                # performed:
                self.triangulated_vectors
                # Try a usual iteration of the current generator, if it
                # does not exist or is exhausted then produce a new generator
                try:  # try 2
                    next(self.rls)
                except (AttributeError, StopIteration, KeyError):
                    vp = self.triangulated_vectors[0]
                    self.rls = self.refine_local_space(*vp, bounds=self.bounds)
                    next(self.rls)

            except (AttributeError, KeyError):
                # If an initial triangulation has not been completed, then
                # we start/continue the initial triangulation targeting `nt`
                # vertices, if nt is greater than the initial number of
                # vertices then the `refine` routine will move back to try 1.
                self.triangulate(nt, self.symmetry)
        return

    def refine_all(self, centroids=True):
        """Refine the entire domain of the current complex."""
        try:
            self.triangulated_vectors
            tvs = copy.copy(self.triangulated_vectors)
            for i, vp in enumerate(tvs):
                self.rls = self.refine_local_space(*vp, bounds=self.bounds)
                for i in self.rls:
                    i
        except AttributeError as ae:
            if str(ae) == "'Complex' object has no attribute " \
                          "'triangulated_vectors'":
                self.triangulate(symmetry=self.symmetry, centroid=centroids)
            else:
                raise

        # This adds a centroid to every new sub-domain generated and defined
        # by self.triangulated_vectors, in addition the vertices ! to complete
        # the triangulation
        return

    def refine_local_space(self, origin, supremum, bounds, centroid=1):
        # Copy for later removal
        origin_c = copy.copy(origin)
        supremum_c = copy.copy(supremum)

        # Initiate local variables redefined in later inner `for` loop:
        vl, vu, a_vu = None, None, None

        # Change the vector orientation so that it is only increasing
        s_ov = list(origin)
        s_origin = list(origin)
        s_sv = list(supremum)
        s_supremum = list(supremum)
        for i, vi in enumerate(s_origin):
            if s_ov[i] > s_sv[i]:
                s_origin[i] = s_sv[i]
                s_supremum[i] = s_ov[i]

        vot = tuple(s_origin)
        vut = tuple(s_supremum)  # Hyperrectangle supremum

        vo = self.V[vot]  # initiate if doesn't exist yet
        vs = self.V[vut]
        # Start by finding the old centroid of the new space:
        vco = self.split_edge(vo.x, vs.x)  # Split in case not centroid arg

        # Find set of extreme vertices in current local space
        sup_set = copy.copy(vco.nn)
        # Cyclic group approach with second x_l --- x_u operation.

        # These containers store the "lower" and "upper" vertices
        # corresponding to the origin or supremum of every C2 group.
        # It has the structure of `dim` times embedded lists each containing
        # these vertices as the entire complex grows. Bounds[0] has to be done
        # outside the loops before we have symmetric containers.
        # NOTE: This means that bounds[0][1] must always exist

        a_vl = copy.copy(list(vot))
        a_vl[0] = vut[0]  # Update aN Origin
        if tuple(a_vl) not in self.V.cache:
            vo = self.V[vot]  # initiate if doesn't exist yet
            vs = self.V[vut]
            # Start by finding the old centroid of the new space:
            vco = self.split_edge(vo.x, vs.x)  # Split in case not centroid arg

            # Find set of extreme vertices in current local space
            sup_set = copy.copy(vco.nn)
            a_vl = copy.copy(list(vot))
            a_vl[0] = vut[0]  # Update aN Origin
            a_vl = self.V[tuple(a_vl)]
        else:
            a_vl = self.V[tuple(a_vl)]

        c_v = self.split_edge(vo.x, a_vl.x)
        c_v.connect(vco)
        yield c_v.x
        Cox = [[vo]]
        Ccx = [[c_v]]
        Cux = [[a_vl]]
        ab_C = []  # Container for a + b operations
        s_ab_C = []  # Container for symmetric a + b operations

        # Loop over remaining bounds
        for i, x in enumerate(bounds[1:]):
            # Update lower and upper containers
            Cox.append([])
            Ccx.append([])
            Cux.append([])
            # try to access a second bound (if not, C1 is symmetric)
            try:
                t_a_vl = list(vot)
                t_a_vl[i + 1] = vut[i + 1]

                # New: lists are used anyway, so copy all
                # %%
                # Copy lists for iteration
                cCox = [x[:] for x in Cox[:i + 1]]
                cCcx = [x[:] for x in Ccx[:i + 1]]
                cCux = [x[:] for x in Cux[:i + 1]]
                # Try to connect aN lower source of previous a + b
                # operation with a aN vertex
                ab_Cc = copy.copy(ab_C)  # NOTE: We append ab_C in the
                # (VL, VC, VU) for-loop, but we use the copy of the list in the
                # ab_Cc for-loop.
                s_ab_Cc = copy.copy(s_ab_C)

                # Early try so that we don't have to copy the cache before
                # moving on to next C1/C2: Try to add the operation of a new
                # C2 product by accessing the upper bound
                if tuple(t_a_vl) not in self.V.cache:
                    # Raise error to continue symmetric refine
                    raise IndexError
                t_a_vu = list(vut)
                t_a_vu[i + 1] = vut[i + 1]
                if tuple(t_a_vu) not in self.V.cache:
                    # Raise error to continue symmetric refine:
                    raise IndexError

                for vectors in s_ab_Cc:
                    # s_ab_C.append([c_vc, vl, vu, a_vu])
                    bc_vc = list(vectors[0].x)
                    b_vl = list(vectors[1].x)
                    b_vu = list(vectors[2].x)
                    ba_vu = list(vectors[3].x)

                    bc_vc[i + 1] = vut[i + 1]
                    b_vl[i + 1] = vut[i + 1]
                    b_vu[i + 1] = vut[i + 1]
                    ba_vu[i + 1] = vut[i + 1]

                    bc_vc = self.V[tuple(bc_vc)]
                    bc_vc.connect(vco)  # NOTE: Unneeded?
                    yield bc_vc

                    # Split to centre, call this centre group "d = 0.5*a"
                    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                    d_bc_vc.connect(bc_vc)
                    d_bc_vc.connect(vectors[1])  # Connect all to centroid
                    d_bc_vc.connect(vectors[2])  # Connect all to centroid
                    d_bc_vc.connect(vectors[3])  # Connect all to centroid
                    yield d_bc_vc.x
