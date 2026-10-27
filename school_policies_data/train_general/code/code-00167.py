    constants[Constants.VERY_LOW_RATIO.value] = float(
        constants[Constants.VERY_LOW_RATIO]
    )
    if (
        constants[Constants.VERY_LOW_RATIO] <= 0.0
        or constants[Constants.VERY_LOW_RATIO] >= 1.0
    ):
        raise ValueError(
            "The constant very_low_ratio must be in the interval (0, 1)."
        )
    if (
        Constants.PENALTY_INCREASE_THRESHOLD in constants
        and constants[Constants.PENALTY_INCREASE_THRESHOLD] < 1.0
    ):
        raise ValueError(
            "The constant penalty_increase_threshold must be "
            "greater than or equal to 1."
        )
    if (
        Constants.PENALTY_INCREASE_FACTOR in constants
        and constants[Constants.PENALTY_INCREASE_FACTOR] <= 1.0
    ):
        raise ValueError(
            "The constant penalty_increase_factor must be greater than 1."
        )
    if (
        Constants.PENALTY_INCREASE_THRESHOLD in constants
        and Constants.PENALTY_INCREASE_FACTOR in constants
    ):
        if (
            constants[Constants.PENALTY_INCREASE_FACTOR]
            < constants[Constants.PENALTY_INCREASE_THRESHOLD]
        ):
            raise ValueError(
                "The constant penalty_increase_factor must be "
                "greater than or equal to "
                "penalty_increase_threshold."
            )
    elif Constants.PENALTY_INCREASE_THRESHOLD in constants:
        constants[Constants.PENALTY_INCREASE_FACTOR.value] = np.max(
            [
                DEFAULT_CONSTANTS[Constants.PENALTY_INCREASE_FACTOR],
                constants[Constants.PENALTY_INCREASE_THRESHOLD],
            ]
        )
    elif Constants.PENALTY_INCREASE_FACTOR in constants:
        constants[Constants.PENALTY_INCREASE_THRESHOLD.value] = np.min(
            [
                DEFAULT_CONSTANTS[Constants.PENALTY_INCREASE_THRESHOLD],
                constants[Constants.PENALTY_INCREASE_FACTOR],
            ]
        )
    else:
        constants[Constants.PENALTY_INCREASE_THRESHOLD.value] = (
            DEFAULT_CONSTANTS[Constants.PENALTY_INCREASE_THRESHOLD]
        )
        constants[Constants.PENALTY_INCREASE_FACTOR.value] = DEFAULT_CONSTANTS[
            Constants.PENALTY_INCREASE_FACTOR
        ]
    constants.setdefault(
        Constants.SHORT_STEP_THRESHOLD.value,
        DEFAULT_CONSTANTS[Constants.SHORT_STEP_THRESHOLD],
    )
    constants[Constants.SHORT_STEP_THRESHOLD.value] = float(
        constants[Constants.SHORT_STEP_THRESHOLD]
    )
    if (
        constants[Constants.SHORT_STEP_THRESHOLD] <= 0.0
        or constants[Constants.SHORT_STEP_THRESHOLD] >= 1.0
    ):
        raise ValueError(
            "The constant short_step_threshold must be in the interval (0, 1)."
        )
    constants.setdefault(
        Constants.LOW_RADIUS_FACTOR.value,
        DEFAULT_CONSTANTS[Constants.LOW_RADIUS_FACTOR],
    )
    constants[Constants.LOW_RADIUS_FACTOR.value] = float(
        constants[Constants.LOW_RADIUS_FACTOR]
    )
    if (
        constants[Constants.LOW_RADIUS_FACTOR] <= 0.0
        or constants[Constants.LOW_RADIUS_FACTOR] >= 1.0
    ):
        raise ValueError(
            "The constant low_radius_factor must be in the interval (0, 1)."
        )
    constants.setdefault(
        Constants.BYRD_OMOJOKUN_FACTOR.value,
        DEFAULT_CONSTANTS[Constants.BYRD_OMOJOKUN_FACTOR],
    )
    constants[Constants.BYRD_OMOJOKUN_FACTOR.value] = float(
        constants[Constants.BYRD_OMOJOKUN_FACTOR]
    )
    if (
        constants[Constants.BYRD_OMOJOKUN_FACTOR] <= 0.0
        or constants[Constants.BYRD_OMOJOKUN_FACTOR] >= 1.0
    ):
        raise ValueError(
            "The constant byrd_omojokun_factor must be in the interval (0, 1)."
        )
    constants.setdefault(
        Constants.THRESHOLD_RATIO_CONSTRAINTS.value,
        DEFAULT_CONSTANTS[Constants.THRESHOLD_RATIO_CONSTRAINTS],
    )
    constants[Constants.THRESHOLD_RATIO_CONSTRAINTS.value] = float(
        constants[Constants.THRESHOLD_RATIO_CONSTRAINTS]
    )
    if constants[Constants.THRESHOLD_RATIO_CONSTRAINTS] <= 1.0:
        raise ValueError(
            "The constant threshold_ratio_constraints must be greater than 1."
        )
    constants.setdefault(
        Constants.LARGE_SHIFT_FACTOR.value,
        DEFAULT_CONSTANTS[Constants.LARGE_SHIFT_FACTOR],
    )
    constants[Constants.LARGE_SHIFT_FACTOR.value] = float(
        constants[Constants.LARGE_SHIFT_FACTOR]
    )
    if constants[Constants.LARGE_SHIFT_FACTOR] < 0.0:
        raise ValueError("The constant large_shift_factor must be "
                         "nonnegative.")
    constants.setdefault(
        Constants.LARGE_GRADIENT_FACTOR.value,
        DEFAULT_CONSTANTS[Constants.LARGE_GRADIENT_FACTOR],
    )
    constants[Constants.LARGE_GRADIENT_FACTOR.value] = float(
        constants[Constants.LARGE_GRADIENT_FACTOR]
    )
    if constants[Constants.LARGE_GRADIENT_FACTOR] <= 1.0:
        raise ValueError(
            "The constant large_gradient_factor must be greater than 1."
        )
    constants.setdefault(
        Constants.RESOLUTION_FACTOR.value,
        DEFAULT_CONSTANTS[Constants.RESOLUTION_FACTOR],
    )
    constants[Constants.RESOLUTION_FACTOR.value] = float(
        constants[Constants.RESOLUTION_FACTOR]
    )
    if constants[Constants.RESOLUTION_FACTOR] <= 1.0:
        raise ValueError(
            "The constant resolution_factor must be greater than 1."
        )
    constants.setdefault(
        Constants.IMPROVE_TCG.value,
        DEFAULT_CONSTANTS[Constants.IMPROVE_TCG],
    )
    constants[Constants.IMPROVE_TCG.value] = bool(
        constants[Constants.IMPROVE_TCG]
    )

    # Check whether they are any unknown options.
    for key in kwargs:
        if key not in Constants.__members__.values():
            warnings.warn(f"Unknown constant: {key}.", RuntimeWarning, 3)
    return constants


def _eval(pb, framework, step, options):
    """
    Evaluate the objective and constraint functions.
    """
    if pb.n_eval >= options[Options.MAX_EVAL]:
        raise MaxEvalError
    x_eval = framework.x_best + step
    fun_val, cub_val, ceq_val = pb(x_eval, framework.penalty)
    r_val = pb.maxcv(x_eval, cub_val, ceq_val)
    if (
        fun_val <= options[Options.TARGET]
        and r_val <= options[Options.FEASIBILITY_TOL]
    ):
        raise TargetSuccess
    if pb.is_feasibility and r_val <= options[Options.FEASIBILITY_TOL]:
        raise FeasibleSuccess
    return fun_val, cub_val, ceq_val


def _build_result(pb, penalty, success, status, n_iter, options):
    """
    Build the result of the optimization process.
    """
    # Build the result.
    x, fun, maxcv = pb.best_eval(penalty)
    success = success and np.isfinite(fun) and np.isfinite(maxcv)
    if status not in [ExitStatus.TARGET_SUCCESS, ExitStatus.FEASIBLE_SUCCESS]:
        success = success and maxcv <= options[Options.FEASIBILITY_TOL]
    result = OptimizeResult()
    result.message = {
        ExitStatus.RADIUS_SUCCESS: "The lower bound for the trust-region "
                                   "radius has been reached",
        ExitStatus.TARGET_SUCCESS: "The target objective function value has "
                                   "been reached",
        ExitStatus.FIXED_SUCCESS: "All variables are fixed by the bound "
                                  "constraints",
        ExitStatus.CALLBACK_SUCCESS: "The callback requested to stop the "
                                     "optimization procedure",
        ExitStatus.FEASIBLE_SUCCESS: "The feasibility problem received has "
                                     "been solved successfully",
        ExitStatus.MAX_EVAL_WARNING: "The maximum number of function "
                                     "evaluations has been exceeded",
        ExitStatus.MAX_ITER_WARNING: "The maximum number of iterations has "
                                     "been exceeded",
        ExitStatus.INFEASIBLE_ERROR: "The bound constraints are infeasible",
        ExitStatus.LINALG_ERROR: "A linear algebra error occurred",
    }.get(status, "Unknown exit status")
    result.success = success
    result.status = status.value
    result.x = pb.build_x(x)
    result.fun = fun
    result.maxcv = maxcv
    result.nfev = pb.n_eval
    result.nit = n_iter
    if options[Options.STORE_HISTORY]:
        result.fun_history = pb.fun_history
        result.maxcv_history = pb.maxcv_history

    # Print the result if requested.
    if options[Options.VERBOSE]:
        _print_step(
            result.message,
            pb,
            result.x,
            result.fun,
            result.maxcv,
            result.nfev,
            result.nit,
        )
    return result


def _print_step(message, pb, x, fun_val, r_val, n_eval, n_iter):
    """
    Print information about the current state of the optimization process.
    """
    print()
    print(f"{message}.")
    print(f"Number of function evaluations: {n_eval}.")
    print(f"Number of iterations: {n_iter}.")
    if not pb.is_feasibility:
        print(f"Least value of {pb.fun_name}: {fun_val}.")
    print(f"Maximum constraint violation: {r_val}.")
    with np.printoptions(**PRINT_OPTIONS):
        print(f"Corresponding point: {x}.")
<|endoftext|>import warnings

import numpy as np
from scipy.optimize import lsq_linear

from .models import Models, Quadratic
from .settings import Options, Constants
from .subsolvers import (
    cauchy_geometry,
    spider_geometry,
    normal_byrd_omojokun,
    tangential_byrd_omojokun,
    constrained_tangential_byrd_omojokun,
)
from .subsolvers.optim import qr_tangential_byrd_omojokun
from .utils import get_arrays_tol


TINY = np.finfo(float).tiny
EPS = np.finfo(float).eps


class TrustRegion:
    """
    Trust-region framework.
    """

    def __init__(self, pb, options, constants):
        """
        Initialize the trust-region framework.

        Parameters
        ----------
        pb : `cobyqa.problem.Problem`
            Problem to solve.
        options : dict
            Options of the solver.
        constants : dict
            Constants of the solver.

        Raises
        ------
        `cobyqa.utils.MaxEvalError`
            If the maximum number of evaluations is reached.
        `cobyqa.utils.TargetSuccess`
            If a nearly feasible point has been found with an objective
            function value below the target.
        `cobyqa.utils.FeasibleSuccess`
            If a feasible point has been found for a feasibility problem.
        `numpy.linalg.LinAlgError`
            If the initial interpolation system is ill-defined.
        """
        # Set the initial penalty parameter.
        self._penalty = 0.0

        # Initialize the models.
        self._pb = pb
        self._models = Models(self._pb, options, self.penalty)
        self._constants = constants

        # Set the index of the best interpolation point.
        self._best_index = 0
        self.set_best_index()

        # Set the initial Lagrange multipliers.
        self._lm_linear_ub = np.zeros(self.m_linear_ub)
        self._lm_linear_eq = np.zeros(self.m_linear_eq)
        self._lm_nonlinear_ub = np.zeros(self.m_nonlinear_ub)
        self._lm_nonlinear_eq = np.zeros(self.m_nonlinear_eq)
        self.set_multipliers(self.x_best)

        # Set the initial trust-region radius and the resolution.
        self._resolution = options[Options.RHOBEG]
        self._radius = self.resolution

    @property
    def n(self):
        """
        Number of variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self._pb.n

    @property
    def m_linear_ub(self):
        """
        Number of linear inequality constraints.

        Returns
        -------
        int
            Number of linear inequality constraints.
        """
        return self._pb.m_linear_ub

    @property
    def m_linear_eq(self):
        """
        Number of linear equality constraints.

        Returns
        -------
        int
            Number of linear equality constraints.
        """
        return self._pb.m_linear_eq

    @property
    def m_nonlinear_ub(self):
        """
        Number of nonlinear inequality constraints.

        Returns
        -------
        int
            Number of nonlinear inequality constraints.
        """
        return self._pb.m_nonlinear_ub

    @property
    def m_nonlinear_eq(self):
        """
        Number of nonlinear equality constraints.

        Returns
        -------
        int
            Number of nonlinear equality constraints.
        """
        return self._pb.m_nonlinear_eq

    @property
    def radius(self):
        """
        Trust-region radius.

        Returns
        -------
        float
            Trust-region radius.
        """
        return self._radius

    @radius.setter
    def radius(self, radius):
        """
        Set the trust-region radius.

        Parameters
        ----------
        radius : float
            New trust-region radius.
        """
        self._radius = radius
        if (
            self.radius
            <= self._constants[Constants.DECREASE_RADIUS_THRESHOLD]
            * self.resolution
        ):
            self._radius = self.resolution

    @property
    def resolution(self):
        """
        Resolution of the trust-region framework.

        The resolution is a lower bound on the trust-region radius.

        Returns
        -------
        float
            Resolution of the trust-region framework.
        """
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        """
        Set the resolution of the trust-region framework.

        Parameters
        ----------
        resolution : float
            New resolution of the trust-region framework.
        """
        self._resolution = resolution

    @property
    def penalty(self):
        """
        Penalty parameter.

        Returns
        -------
        float
            Penalty parameter.
        """
        return self._penalty

    @property
    def models(self):
        """
        Models of the objective function and constraints.

        Returns
        -------
        `cobyqa.models.Models`
            Models of the objective function and constraints.
        """
        return self._models

    @property
    def best_index(self):
        """
        Index of the best interpolation point.

        Returns
        -------
        int
            Index of the best interpolation point.
        """
        return self._best_index

    @property
    def x_best(self):
        """
        Best interpolation point.

        Its value is interpreted as relative to the origin, not the base point.

        Returns
        -------
        `numpy.ndarray`
            Best interpolation point.
        """
        return self.models.interpolation.point(self.best_index)

    @property
    def fun_best(self):
        """
        Value of the objective function at `x_best`.

        Returns
        -------
        float
            Value of the objective function at `x_best`.
        """
        return self.models.fun_val[self.best_index]

    @property
    def cub_best(self):
        """
        Values of the nonlinear inequality constraints at `x_best`.

        Returns
        -------
        `numpy.ndarray`, shape (m_nonlinear_ub,)
            Values of the nonlinear inequality constraints at `x_best`.
        """
        return self.models.cub_val[self.best_index, :]

    @property
    def ceq_best(self):
        """
        Values of the nonlinear equality constraints at `x_best`.

        Returns
        -------
        `numpy.ndarray`, shape (m_nonlinear_eq,)
            Values of the nonlinear equality constraints at `x_best`.
        """
        return self.models.ceq_val[self.best_index, :]

    def lag_model(self, x):
        """
        Evaluate the Lagrangian model at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which the Lagrangian model is evaluated.

        Returns
        -------
        float
            Value of the Lagrangian model at `x`.
        """
        return (
            self.models.fun(x)
            + self._lm_linear_ub
            @ (self._pb.linear.a_ub @ x - self._pb.linear.b_ub)
            + self._lm_linear_eq
            @ (self._pb.linear.a_eq @ x - self._pb.linear.b_eq)
            + self._lm_nonlinear_ub @ self.models.cub(x)
            + self._lm_nonlinear_eq @ self.models.ceq(x)
        )

    def lag_model_grad(self, x):
        """
        Evaluate the gradient of the Lagrangian model at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which the gradient of the Lagrangian model is evaluated.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Gradient of the Lagrangian model at `x`.
        """
        return (
            self.models.fun_grad(x)
            + self._lm_linear_ub @ self._pb.linear.a_ub
            + self._lm_linear_eq @ self._pb.linear.a_eq
            + self._lm_nonlinear_ub @ self.models.cub_grad(x)
            + self._lm_nonlinear_eq @ self.models.ceq_grad(x)
        )

    def lag_model_hess(self):
        """
        Evaluate the Hessian matrix of the Lagrangian model at a given point.

        Returns
        -------
        `numpy.ndarray`, shape (n, n)
            Hessian matrix of the Lagrangian model at `x`.
        """
        hess = self.models.fun_hess()
        if self.m_nonlinear_ub > 0:
            hess += self._lm_nonlinear_ub @ self.models.cub_hess()
        if self.m_nonlinear_eq > 0:
            hess += self._lm_nonlinear_eq @ self.models.ceq_hess()
        return hess

    def lag_model_hess_prod(self, v):
        """
        Evaluate the right product of the Hessian matrix of the Lagrangian
        model with a given vector.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Vector with which the Hessian matrix of the Lagrangian model is
            multiplied from the right.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Right product of the Hessian matrix of the Lagrangian model with
            `v`.
        """
        return (
            self.models.fun_hess_prod(v)
            + self._lm_nonlinear_ub @ self.models.cub_hess_prod(v)
            + self._lm_nonlinear_eq @ self.models.ceq_hess_prod(v)
        )

    def lag_model_curv(self, v):
        """
        Evaluate the curvature of the Lagrangian model along a given direction.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Direction along which the curvature of the Lagrangian model is
            evaluated.

        Returns
        -------
        float
            Curvature of the Lagrangian model along `v`.
        """
        return (
            self.models.fun_curv(v)
            + self._lm_nonlinear_ub @ self.models.cub_curv(v)
            + self._lm_nonlinear_eq @ self.models.ceq_curv(v)
        )

    def sqp_fun(self, step):
        """
        Evaluate the objective function of the SQP subproblem.

        Parameters
        ----------
        step : `numpy.ndarray`, shape (n,)
            Step along which the objective function of the SQP subproblem is
            evaluated.

        Returns
        -------
        float
            Value of the objective function of the SQP subproblem along `step`.
        """
        return step @ (
            self.models.fun_grad(self.x_best)
            + 0.5 * self.lag_model_hess_prod(step)
        )

    def sqp_cub(self, step):
        """
        Evaluate the linearization of the nonlinear inequality constraints.

        Parameters
        ----------
        step : `numpy.ndarray`, shape (n,)
            Step along which the linearization of the nonlinear inequality
            constraints is evaluated.

        Returns
        -------
        `numpy.ndarray`, shape (m_nonlinear_ub,)
            Value of the linearization of the nonlinear inequality constraints
            along `step`.
        """
        return (
            self.models.cub(self.x_best)
            + self.models.cub_grad(self.x_best) @ step
        )

    def sqp_ceq(self, step):
        """
        Evaluate the linearization of the nonlinear equality constraints.

        Parameters
        ----------
        step : `numpy.ndarray`, shape (n,)
            Step along which the linearization of the nonlinear equality
            constraints is evaluated.

        Returns
        -------
        `numpy.ndarray`, shape (m_nonlinear_ub,)
            Value of the linearization of the nonlinear equality constraints
            along `step`.
        """
        return (
            self.models.ceq(self.x_best)
            + self.models.ceq_grad(self.x_best) @ step
        )

    def merit(self, x, fun_val=None, cub_val=None, ceq_val=None):
        """
        Evaluate the merit function at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which the merit function is evaluated.
        fun_val : float, optional
            Value of the objective function at `x`. If not provided, the
            objective function is evaluated at `x`.
        cub_val : `numpy.ndarray`, shape (m_nonlinear_ub,), optional
            Values of the nonlinear inequality constraints. If not provided,
            the nonlinear inequality constraints are evaluated at `x`.
        ceq_val : `numpy.ndarray`, shape (m_nonlinear_eq,), optional
            Values of the nonlinear equality constraints. If not provided,
            the nonlinear equality constraints are evaluated at `x`.

        Returns
        -------
        float
            Value of the merit function at `x`.
        """
        if fun_val is None or cub_val is None or ceq_val is None:
            fun_val, cub_val, ceq_val = self._pb(x, self.penalty)
        m_val = fun_val
        if self._penalty > 0.0:
            c_val = self._pb.violation(x, cub_val=cub_val, ceq_val=ceq_val)
            if np.count_nonzero(c_val):
                m_val += self._penalty * np.linalg.norm(c_val)
        return m_val

    def get_constraint_linearizations(self, x):
        """
        Get the linearizations of the constraints at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which the linearizations of the constraints are evaluated.

        Returns
        -------
        `numpy.ndarray`, shape (m_linear_ub + m_nonlinear_ub, n)
            Left-hand side matrix of the linearized inequality constraints.
        `numpy.ndarray`, shape (m_linear_ub + m_nonlinear_ub,)
            Right-hand side vector of the linearized inequality constraints.
        `numpy.ndarray`, shape (m_linear_eq + m_nonlinear_eq, n)
            Left-hand side matrix of the linearized equality constraints.
        `numpy.ndarray`, shape (m_linear_eq + m_nonlinear_eq,)
            Right-hand side vector of the linearized equality constraints.
        """
        aub = np.block(
            [
                [self._pb.linear.a_ub],
                [self.models.cub_grad(x)],
            ]
        )
        bub = np.block(
            [
                self._pb.linear.b_ub - self._pb.linear.a_ub @ x,
                -self.models.cub(x),
            ]
        )
        aeq = np.block(
            [
                [self._pb.linear.a_eq],
                [self.models.ceq_grad(x)],
            ]
        )
        beq = np.block(
            [
                self._pb.linear.b_eq - self._pb.linear.a_eq @ x,
                -self.models.ceq(x),
            ]
        )
        return aub, bub, aeq, beq

    def get_trust_region_step(self, options):
        """
        Get the trust-region step.

        The trust-region step is computed by solving the derivative-free
        trust-region SQP subproblem using a Byrd-Omojokun composite-step
        approach. For more details, see Section 5.2.3 of [1]_.

        Parameters
        ----------
        options : dict
            Options of the solver.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Normal step.
        `numpy.ndarray`, shape (n,)
            Tangential step.

        References
        ----------
        .. [1] T. M. Ragonneau. *Model-Based Derivative-Free Optimization
           Methods and Software*. PhD thesis, Department of Applied
           Mathematics, The Hong Kong Polytechnic University, Hong Kong, China,
           2022. URL: https://theses.lib.polyu.edu.hk/handle/200/12294.
        """
        # Evaluate the linearizations of the constraints.
        aub, bub, aeq, beq = self.get_constraint_linearizations(self.x_best)
        xl = self._pb.bounds.xl - self.x_best
        xu = self._pb.bounds.xu - self.x_best

        # Evaluate the normal step.
        radius = self._constants[Constants.BYRD_OMOJOKUN_FACTOR] * self.radius
        normal_step = normal_byrd_omojokun(
            aub,
            bub,
            aeq,
            beq,
            xl,
            xu,
            radius,
            options[Options.DEBUG],
            **self._constants,
        )
        if options[Options.DEBUG]:
            tol = get_arrays_tol(xl, xu)
            if (np.any(normal_step + tol < xl)
                    or np.any(xu < normal_step - tol)):
                warnings.warn(
                    "the normal step does not respect the bound constraint.",
                    RuntimeWarning,
                    2,
                )
            if np.linalg.norm(normal_step) > 1.1 * radius:
                warnings.warn(
                    "the normal step does not respect the trust-region "
                    "constraint.",
                    RuntimeWarning,
                    2,
                )

        # Evaluate the tangential step.
        radius = np.sqrt(self.radius**2.0 - normal_step @ normal_step)
        xl -= normal_step
        xu -= normal_step
        bub = np.maximum(bub - aub @ normal_step, 0.0)
        g_best = self.models.fun_grad(self.x_best) + self.lag_model_hess_prod(
            normal_step
        )
        if self._pb.type in ["unconstrained", "bound-constrained"]:
            tangential_step = tangential_byrd_omojokun(
                g_best,
                self.lag_model_hess_prod,
                xl,
                xu,
                radius,
                options[Options.DEBUG],
                **self._constants,
            )
        else:
            tangential_step = constrained_tangential_byrd_omojokun(
                g_best,
                self.lag_model_hess_prod,
                xl,
                xu,
                aub,
                bub,
                aeq,
                radius,
                options["debug"],
                **self._constants,
            )
        if options[Options.DEBUG]:
            tol = get_arrays_tol(xl, xu)
            if np.any(tangential_step + tol < xl) or np.any(
                xu < tangential_step - tol
            ):
                warnings.warn(
                    "The tangential step does not respect the bound "
                    "constraints.",
                    RuntimeWarning,
                    2,
                )
            if (
                np.linalg.norm(normal_step + tangential_step)
                > 1.1 * np.sqrt(2.0) * self.radius
            ):
                warnings.warn(
                    "The trial step does not respect the trust-region "
                    "constraint.",
                    RuntimeWarning,
                    2,
                )
        return normal_step, tangential_step

    def get_geometry_step(self, k_new, options):
        """
        Get the geometry-improving step.

        Three different geometry-improving steps are computed and the best one
        is returned. For more details, see Section 5.2.7 of [1]_.

        Parameters
        ----------
        k_new : int
            Index of the interpolation point to be modified.
        options : dict
            Options of the solver.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Geometry-improving step.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the computation of a determinant fails.

        References
        ----------
        .. [1] T. M. Ragonneau. *Model-Based Derivative-Free Optimization
           Methods and Software*. PhD thesis, Department of Applied
           Mathematics, The Hong Kong Polytechnic University, Hong Kong, China,
           2022. URL: https://theses.lib.polyu.edu.hk/handle/200/12294.
        """
        if options[Options.DEBUG]:
            assert (
                k_new != self.best_index
            ), "The index `k_new` must be different from the best index."

        # Build the k_new-th Lagrange polynomial.
        coord_vec = np.squeeze(np.eye(1, self.models.npt, k_new))
        lag = Quadratic(
            self.models.interpolation,
            coord_vec,
            options[Options.DEBUG],
        )
        g_lag = lag.grad(self.x_best, self.models.interpolation)

        # Compute a simple constrained Cauchy step.
        xl = self._pb.bounds.xl - self.x_best
        xu = self._pb.bounds.xu - self.x_best
        step = cauchy_geometry(
            0.0,
            g_lag,
            lambda v: lag.curv(v, self.models.interpolation),
            xl,
            xu,
            self.radius,
            options[Options.DEBUG],
        )
        sigma = self.models.determinants(self.x_best + step, k_new)

        # Compute the solution on the straight lines joining the interpolation
        # points to the k-th one, and choose it if it provides a larger value
        # of the determinant of the interpolation system in absolute value.
        xpt = (
            self.models.interpolation.xpt
            - self.models.interpolation.xpt[:, self.best_index, np.newaxis]
        )
        xpt[:, [0, self.best_index]] = xpt[:, [self.best_index, 0]]
        step_alt = spider_geometry(
            0.0,
            g_lag,
            lambda v: lag.curv(v, self.models.interpolation),
            xpt[:, 1:],
            xl,
            xu,
            self.radius,
            options[Options.DEBUG],
        )
        sigma_alt = self.models.determinants(self.x_best + step_alt, k_new)
        if abs(sigma_alt) > abs(sigma):
            step = step_alt
            sigma = sigma_alt

        # Compute a Cauchy step on the tangent space of the active constraints.
        if self._pb.type in [
            "linearly constrained",
            "nonlinearly constrained",
        ]:
            aub, bub, aeq, beq = (
                self.get_constraint_linearizations(self.x_best))
            tol_bd = get_arrays_tol(xl, xu)
            tol_ub = get_arrays_tol(bub)
            free_xl = xl <= -tol_bd
            free_xu = xu >= tol_bd
            free_ub = bub >= tol_ub

            # Compute the Cauchy step.
            n_act, q = qr_tangential_byrd_omojokun(
                aub,
                aeq,
                free_xl,
                free_xu,
                free_ub,
            )
            g_lag_proj = q[:, n_act:] @ (q[:, n_act:].T @ g_lag)
            norm_g_lag_proj = np.linalg.norm(g_lag_proj)
            if 0 < n_act < self._pb.n and norm_g_lag_proj > TINY * self.radius:
                step_alt = (self.radius / norm_g_lag_proj) * g_lag_proj
                if lag.curv(step_alt, self.models.interpolation) < 0.0:
                    step_alt = -step_alt

                # Evaluate the constraint violation at the Cauchy step.
                cbd = np.block([xl - step_alt, step_alt - xu])
                cub = aub @ step_alt - bub
                ceq = aeq @ step_alt - beq
                maxcv_val = max(
                    np.max(array, initial=0.0)
                    for array in [cbd, cub, np.abs(ceq)]
                )

                # Accept the new step if it is nearly feasible and do not
                # drastically worsen the determinant of the interpolation
                # system in absolute value.
                tol = np.max(np.abs(step_alt[~free_xl]), initial=0.0)
                tol = np.max(np.abs(step_alt[~free_xu]), initial=tol)
                tol = np.max(np.abs(aub[~free_ub, :] @ step_alt), initial=tol)
                tol = min(10.0 * tol, 1e-2 * np.linalg.norm(step_alt))
                if maxcv_val <= tol:
                    sigma_alt = self.models.determinants(
                        self.x_best + step_alt, k_new
                    )
                    if abs(sigma_alt) >= 0.1 * abs(sigma):
                        step = np.clip(step_alt, xl, xu)

        if options[Options.DEBUG]:
            tol = get_arrays_tol(xl, xu)
            if np.any(step + tol < xl) or np.any(xu < step - tol):
                warnings.warn(
                    "The geometry step does not respect the bound "
                    "constraints.",
                    RuntimeWarning,
                    2,
                )
            if np.linalg.norm(step) > 1.1 * self.radius:
                warnings.warn(
                    "The geometry step does not respect the "
                    "trust-region constraint.",
                    RuntimeWarning,
                    2,
                )
        return step

    def get_second_order_correction_step(self, step, options):
        """
        Get the second-order correction step.

        Parameters
        ----------
        step : `numpy.ndarray`, shape (n,)
            Trust-region step.
        options : dict
            Options of the solver.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Second-order correction step.
        """
        # Evaluate the linearizations of the constraints.
        aub, bub, aeq, beq = self.get_constraint_linearizations(self.x_best)
        xl = self._pb.bounds.xl - self.x_best
        xu = self._pb.bounds.xu - self.x_best
        radius = np.linalg.norm(step)
        soc_step = normal_byrd_omojokun(
            aub,
            bub,
            aeq,
            beq,
            xl,
            xu,
            radius,
            options[Options.DEBUG],
            **self._constants,
        )
        if options[Options.DEBUG]:
            tol = get_arrays_tol(xl, xu)
            if np.any(soc_step + tol < xl) or np.any(xu < soc_step - tol):
                warnings.warn(
                    "The second-order correction step does not "
                    "respect the bound constraints.",
                    RuntimeWarning,
                    2,
                )
            if np.linalg.norm(soc_step) > 1.1 * radius:
                warnings.warn(
                    "The second-order correction step does not "
                    "respect the trust-region constraint.",
                    RuntimeWarning,
                    2,
                )
        return soc_step

    def get_reduction_ratio(self, step, fun_val, cub_val, ceq_val):
        """
        Get the reduction ratio.

        Parameters
        ----------
        step : `numpy.ndarray`, shape (n,)
            Trust-region step.
        fun_val : float
            Objective function value at the trial point.
        cub_val : `numpy.ndarray`, shape (m_nonlinear_ub,)
            Nonlinear inequality constraint values at the trial point.
        ceq_val : `numpy.ndarray`, shape (m_nonlinear_eq,)
            Nonlinear equality constraint values at the trial point.

        Returns
        -------
        float
            Reduction ratio.
        """
        merit_old = self.merit(
            self.x_best,
            self.fun_best,
            self.cub_best,
            self.ceq_best,
        )
        merit_new = self.merit(self.x_best + step, fun_val, cub_val, ceq_val)
        merit_model_old = self.merit(
            self.x_best,
            0.0,
            self.models.cub(self.x_best),
            self.models.ceq(self.x_best),
        )
        merit_model_new = self.merit(
            self.x_best + step,
            self.sqp_fun(step),
            self.sqp_cub(step),
            self.sqp_ceq(step),
        )
        if abs(merit_model_old - merit_model_new) > TINY * abs(
            merit_old - merit_new
        ):
            return (merit_old - merit_new) / abs(
                merit_model_old - merit_model_new
            )
        else:
            return -1.0

    def increase_penalty(self, step):
        """
        Increase the penalty parameter.

        Parameters
        ----------
        step : `numpy.ndarray`, shape (n,)
            Trust-region step.
        """
        aub, bub, aeq, beq = self.get_constraint_linearizations(self.x_best)
        viol_diff = max(
            np.linalg.norm(
                np.block(
                    [
                        np.maximum(0.0, -bub),
                        beq,
                    ]
                )
            )
            - np.linalg.norm(
                np.block(
                    [
                        np.maximum(0.0, aub @ step - bub),
                        aeq @ step - beq,
                    ]
                )
            ),
            0.0,
        )
        sqp_val = self.sqp_fun(step)

        threshold = np.linalg.norm(
            np.block(
                [
                    self._lm_linear_ub,
                    self._lm_linear_eq,
                    self._lm_nonlinear_ub,
                    self._lm_nonlinear_eq,
                ]
            )
        )
        if abs(viol_diff) > TINY * abs(sqp_val):
            threshold = max(threshold, sqp_val / viol_diff)
        best_index_save = self.best_index
        if (
            self._penalty
            <= self._constants[Constants.PENALTY_INCREASE_THRESHOLD]
                * threshold
        ):
            self._penalty = max(
                self._constants[Constants.PENALTY_INCREASE_FACTOR] * threshold,
                1.0,
            )
            self.set_best_index()
        return best_index_save == self.best_index

    def decrease_penalty(self):
        """
        Decrease the penalty parameter.
        """
        self._penalty = min(self._penalty, self._get_low_penalty())
        self.set_best_index()

    def set_best_index(self):
        """
        Set the index of the best point.
        """
        best_index = self.best_index
        m_best = self.merit(
            self.x_best,
            self.models.fun_val[best_index],
            self.models.cub_val[best_index, :],
            self.models.ceq_val[best_index, :],
        )
        r_best = self._pb.maxcv(
            self.x_best,
            self.models.cub_val[best_index, :],
            self.models.ceq_val[best_index, :],
        )
        tol = (
            10.0
            * EPS
            * max(self.models.n, self.models.npt)
            * max(abs(m_best), 1.0)
        )
        for k in range(self.models.npt):
            if k != self.best_index:
                x_val = self.models.interpolation.point(k)
                m_val = self.merit(
                    x_val,
                    self.models.fun_val[k],
                    self.models.cub_val[k, :],
                    self.models.ceq_val[k, :],
                )
                r_val = self._pb.maxcv(
                    x_val,
                    self.models.cub_val[k, :],
                    self.models.ceq_val[k, :],
                )
                if m_val < m_best or (m_val < m_best + tol and r_val < r_best):
                    best_index = k
                    m_best = m_val
                    r_best = r_val
        self._best_index = best_index

    def get_index_to_remove(self, x_new=None):
        """
        Get the index of the interpolation point to remove.

        If `x_new` is not provided, the index returned should be used during
        the geometry-improvement phase. Otherwise, the index returned is the
        best index for included `x_new` in the interpolation set.

        Parameters
        ----------
        x_new : `numpy.ndarray`, shape (n,), optional
            New point to be included in the interpolation set.

        Returns
        -------
        int
            Index of the interpolation point to remove.
        float
            Distance between `x_best` and the removed point.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the computation of a determinant fails.
        """
        dist_sq = np.sum(
            (
                self.models.interpolation.xpt
                - self.models.interpolation.xpt[:, self.best_index, np.newaxis]
            )
            ** 2.0,
            axis=0,
        )
        if x_new is None:
            sigma = 1.0
            weights = dist_sq
        else:
            sigma = self.models.determinants(x_new)
            weights = (
                np.maximum(
                    1.0,
                    dist_sq
                    / max(
                        self._constants[Constants.LOW_RADIUS_FACTOR]
                        * self.radius,
                        self.resolution,
                    )
                    ** 2.0,
                )
                ** 3.0
            )
            weights[self.best_index] = -1.0  # do not remove the best point
        k_max = np.argmax(weights * np.abs(sigma))
        return k_max, np.sqrt(dist_sq[k_max])

    def update_radius(self, step, ratio):
        """
        Update the trust-region radius.

        Parameters
        ----------
        step : `numpy.ndarray`, shape (n,)
            Trust-region step.
        ratio : float
            Reduction ratio.
        """
        s_norm = np.linalg.norm(step)
        if ratio <= self._constants[Constants.LOW_RATIO]:
            self.radius *= self._constants[Constants.DECREASE_RADIUS_FACTOR]
        elif ratio <= self._constants[Constants.HIGH_RATIO]:
            self.radius = max(
                self._constants[Constants.DECREASE_RADIUS_FACTOR]
                * self.radius,
                s_norm,
            )
        else:
            self.radius = min(
                self._constants[Constants.INCREASE_RADIUS_FACTOR]
                * self.radius,
                max(
                    self._constants[Constants.DECREASE_RADIUS_FACTOR]
                    * self.radius,
                    self._constants[Constants.INCREASE_RADIUS_THRESHOLD]
                    * s_norm,
                ),
            )

    def enhance_resolution(self, options):
        """
        Enhance the resolution of the trust-region framework.

        Parameters
        ----------
        options : dict
            Options of the solver.
        """
        if (
            self._constants[Constants.LARGE_RESOLUTION_THRESHOLD]
            * options[Options.RHOEND]
            < self.resolution
        ):
            self.resolution *= self._constants[
                Constants.DECREASE_RESOLUTION_FACTOR
            ]
        elif (
            self._constants[Constants.MODERATE_RESOLUTION_THRESHOLD]
            * options[Options.RHOEND]
            < self.resolution
        ):
            self.resolution = np.sqrt(self.resolution
                                      * options[Options.RHOEND])
        else:
            self.resolution = options[Options.RHOEND]

        # Reduce the trust-region radius.
        self._radius = max(
            self._constants[Constants.DECREASE_RADIUS_FACTOR] * self._radius,
            self.resolution,
        )

    def shift_x_base(self, options):
        """
        Shift the base point to `x_best`.

        Parameters
        ----------
        options : dict
            Options of the solver.
        """
        self.models.shift_x_base(np.copy(self.x_best), options)

    def set_multipliers(self, x):
        """
        Set the Lagrange multipliers.

        This method computes and set the Lagrange multipliers of the linear and
        nonlinear constraints to be the QP multipliers.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which the Lagrange multipliers are computed.
        """
        # Build the constraints of the least-squares problem.
        incl_linear_ub = self._pb.linear.a_ub @ x >= self._pb.linear.b_ub
        incl_nonlinear_ub = self.cub_best >= 0.0
        incl_xl = self._pb.bounds.xl >= x
        incl_xu = self._pb.bounds.xu <= x
        m_linear_ub = np.count_nonzero(incl_linear_ub)
        m_nonlinear_ub = np.count_nonzero(incl_nonlinear_ub)
        m_xl = np.count_nonzero(incl_xl)
        m_xu = np.count_nonzero(incl_xu)

        if (
            m_linear_ub + m_nonlinear_ub + self.m_linear_eq
                + self.m_nonlinear_eq > 0
        ):
            identity = np.eye(self._pb.n)
            c_jac = np.r_[
                -identity[incl_xl, :],
                identity[incl_xu, :],
                self._pb.linear.a_ub[incl_linear_ub, :],
                self.models.cub_grad(x, incl_nonlinear_ub),
                self._pb.linear.a_eq,
                self.models.ceq_grad(x),
            ]

            # Solve the least-squares problem.
            g_best = self.models.fun_grad(x)
            xl_lm = np.full(c_jac.shape[0], -np.inf)
            xl_lm[: m_xl + m_xu + m_linear_ub + m_nonlinear_ub] = 0.0
            res = lsq_linear(
                c_jac.T,
                -g_best,
                bounds=(xl_lm, np.inf),
                method="bvls",
            )

            # Extract the Lagrange multipliers.
            self._lm_linear_ub[incl_linear_ub] = res.x[
                m_xl + m_xu:m_xl + m_xu + m_linear_ub
            ]
            self._lm_linear_ub[~incl_linear_ub] = 0.0
            self._lm_nonlinear_ub[incl_nonlinear_ub] = res.x[
                m_xl
                + m_xu
                + m_linear_ub:m_xl
                + m_xu
                + m_linear_ub
                + m_nonlinear_ub
            ]
            self._lm_nonlinear_ub[~incl_nonlinear_ub] = 0.0
            self._lm_linear_eq[:] = res.x[
                m_xl
                + m_xu
                + m_linear_ub
                + m_nonlinear_ub:m_xl
                + m_xu
                + m_linear_ub
                + m_nonlinear_ub
                + self.m_linear_eq
            ]
            self._lm_nonlinear_eq[:] = res.x[
                m_xl + m_xu + m_linear_ub + m_nonlinear_ub + self.m_linear_eq:
            ]

    def _get_low_penalty(self):
        r_val_ub = np.c_[
            (
                self.models.interpolation.x_base[np.newaxis, :]
                + self.models.interpolation.xpt.T
            )
            @ self._pb.linear.a_ub.T
            - self._pb.linear.b_ub[np.newaxis, :],
            self.models.cub_val,
        ]
        r_val_eq = (
            self.models.interpolation.x_base[np.newaxis, :]
            + self.models.interpolation.xpt.T
        ) @ self._pb.linear.a_eq.T - self._pb.linear.b_eq[np.newaxis, :]
        r_val_eq = np.block(
            [
                r_val_eq,
                -r_val_eq,
                self.models.ceq_val,
                -self.models.ceq_val,
            ]
        )
        r_val = np.block([r_val_ub, r_val_eq])
        c_min = np.nanmin(r_val, axis=0)
        c_max = np.nanmax(r_val, axis=0)
        indices = (
            c_min
            < self._constants[Constants.THRESHOLD_RATIO_CONSTRAINTS] * c_max
        )
        if np.any(indices):
            f_min = np.nanmin(self.models.fun_val)
            f_max = np.nanmax(self.models.fun_val)
            c_min_neg = np.minimum(0.0, c_min[indices])
            c_diff = np.min(c_max[indices] - c_min_neg)
            if c_diff > TINY * (f_max - f_min):
                penalty = (f_max - f_min) / c_diff
            else:
                penalty = np.inf
        else:
            penalty = 0.0
        return penalty
<|endoftext|>import warnings

import numpy as np
from scipy.linalg import eigh

from .settings import Options
from .utils import MaxEvalError, TargetSuccess, FeasibleSuccess


EPS = np.finfo(float).eps


class Interpolation:
    """
    Interpolation set.

    This class stores a base point around which the models are expanded and the
    interpolation points. The coordinates of the interpolation points are
    relative to the base point.
    """

    def __init__(self, pb, options):
        """
        Initialize the interpolation set.

        Parameters
        ----------
        pb : `cobyqa.problem.Problem`
            Problem to be solved.
        options : dict
            Options of the solver.
        """
        # Reduce the initial trust-region radius if necessary.
        self._debug = options[Options.DEBUG]
        max_radius = 0.5 * np.min(pb.bounds.xu - pb.bounds.xl)
        if options[Options.RHOBEG] > max_radius:
            options[Options.RHOBEG.value] = max_radius
            options[Options.RHOEND.value] = np.min(
                [
                    options[Options.RHOEND],
                    max_radius,
                ]
            )

        # Set the initial point around which the models are expanded.
        self._x_base = np.copy(pb.x0)
        very_close_xl_idx = (
            self.x_base <= pb.bounds.xl + 0.5 * options[Options.RHOBEG]
        )
        self.x_base[very_close_xl_idx] = pb.bounds.xl[very_close_xl_idx]
        close_xl_idx = (
            pb.bounds.xl + 0.5 * options[Options.RHOBEG] < self.x_base
        ) & (self.x_base <= pb.bounds.xl + options[Options.RHOBEG])
        self.x_base[close_xl_idx] = np.minimum(
            pb.bounds.xl[close_xl_idx] + options[Options.RHOBEG],
            pb.bounds.xu[close_xl_idx],
        )
        very_close_xu_idx = (
            self.x_base >= pb.bounds.xu - 0.5 * options[Options.RHOBEG]
        )
        self.x_base[very_close_xu_idx] = pb.bounds.xu[very_close_xu_idx]
        close_xu_idx = (
            self.x_base < pb.bounds.xu - 0.5 * options[Options.RHOBEG]
        ) & (pb.bounds.xu - options[Options.RHOBEG] <= self.x_base)
        self.x_base[close_xu_idx] = np.maximum(
            pb.bounds.xu[close_xu_idx] - options[Options.RHOBEG],
            pb.bounds.xl[close_xu_idx],
        )

        # Set the initial interpolation set.
        self._xpt = np.zeros((pb.n, options[Options.NPT]))
        for k in range(1, options[Options.NPT]):
            if k <= pb.n:
                if very_close_xu_idx[k - 1]:
                    self.xpt[k - 1, k] = -options[Options.RHOBEG]
                else:
                    self.xpt[k - 1, k] = options[Options.RHOBEG]
            elif k <= 2 * pb.n:
                if very_close_xl_idx[k - pb.n - 1]:
                    self.xpt[k - pb.n - 1, k] = 2.0 * options[Options.RHOBEG]
                elif very_close_xu_idx[k - pb.n - 1]:
                    self.xpt[k - pb.n - 1, k] = -2.0 * options[Options.RHOBEG]
                else:
                    self.xpt[k - pb.n - 1, k] = -options[Options.RHOBEG]
            else:
                spread = (k - pb.n - 1) // pb.n
                k1 = k - (1 + spread) * pb.n - 1
                k2 = (k1 + spread) % pb.n
                self.xpt[k1, k] = self.xpt[k1, k1 + 1]
                self.xpt[k2, k] = self.xpt[k2, k2 + 1]

    @property
    def n(self):
        """
        Number of variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self.xpt.shape[0]

    @property
    def npt(self):
        """
        Number of interpolation points.

        Returns
        -------
        int
            Number of interpolation points.
        """
        return self.xpt.shape[1]

    @property
    def xpt(self):
        """
        Interpolation points.

        Returns
        -------
        `numpy.ndarray`, shape (n, npt)
            Interpolation points.
        """
        return self._xpt

    @xpt.setter
    def xpt(self, xpt):
        """
        Set the interpolation points.

        Parameters
        ----------
        xpt : `numpy.ndarray`, shape (n, npt)
            New interpolation points.
        """
        if self._debug:
            assert xpt.shape == (
                self.n,
                self.npt,
            ), "The shape of `xpt` is not valid."
        self._xpt = xpt

    @property
    def x_base(self):
        """
        Base point around which the models are expanded.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Base point around which the models are expanded.
        """
        return self._x_base

    @x_base.setter
    def x_base(self, x_base):
        """
        Set the base point around which the models are expanded.

        Parameters
        ----------
        x_base : `numpy.ndarray`, shape (n,)
            New base point around which the models are expanded.
        """
        if self._debug:
            assert x_base.shape == (
                self.n,
            ), "The shape of `x_base` is not valid."
        self._x_base = x_base

    def point(self, k):
        """
        Get the `k`-th interpolation point.

        The return point is relative to the origin.

        Parameters
        ----------
        k : int
            Index of the interpolation point.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            `k`-th interpolation point.
        """
        if self._debug:
            assert 0 <= k < self.npt, "The index `k` is not valid."
        return self.x_base + self.xpt[:, k]


_cache = {"xpt": None, "a": None, "right_scaling": None, "eigh": None}


def build_system(interpolation):
    """
    Build the left-hand side matrix of the interpolation system. The
    matrix below stores W * diag(right_scaling),
    where W is the theoretical matrix of the interpolation system. The
    right scaling matrices is chosen to keep the elements in
    the matrix well-balanced.

    Parameters
    ----------
    interpolation : `cobyqa.models.Interpolation`
        Interpolation set.
    """

    # Compute the scaled directions from the base point to the
    # interpolation points. We scale the directions to avoid numerical
    # difficulties.
    if _cache["xpt"] is not None and np.array_equal(
        interpolation.xpt, _cache["xpt"]
    ):
        return _cache["a"], _cache["right_scaling"], _cache["eigh"]

    scale = np.max(np.linalg.norm(interpolation.xpt, axis=0), initial=EPS)
    xpt_scale = interpolation.xpt / scale

    n, npt = xpt_scale.shape
    a = np.zeros((npt + n + 1, npt + n + 1))
    a[:npt, :npt] = 0.5 * (xpt_scale.T @ xpt_scale) ** 2.0
    a[:npt, npt] = 1.0
    a[:npt, npt + 1:] = xpt_scale.T
    a[npt, :npt] = 1.0
    a[npt + 1:, :npt] = xpt_scale

    # Build the left and right scaling diagonal matrices.
    right_scaling = np.empty(npt + n + 1)
    right_scaling[:npt] = 1.0 / scale**2.0
    right_scaling[npt] = scale**2.0
    right_scaling[npt + 1:] = scale

    eig_values, eig_vectors = eigh(a, check_finite=False)

    _cache["xpt"] = np.copy(interpolation.xpt)
    _cache["a"] = np.copy(a)
    _cache["right_scaling"] = np.copy(right_scaling)
    _cache["eigh"] = (eig_values, eig_vectors)

    return a, right_scaling, (eig_values, eig_vectors)


class Quadratic:
    """
    Quadratic model.

    This class stores the Hessian matrix of the quadratic model using the
    implicit/explicit representation designed by Powell for NEWUOA [1]_.

    References
    ----------
    .. [1] M. J. D. Powell. The NEWUOA software for unconstrained optimization
       without derivatives. In G. Di Pillo and M. Roma, editors, *Large-Scale
       Nonlinear Optimization*, volume 83 of Nonconvex Optim. Appl., pages
       255--297. Springer, Boston, MA, USA, 2006. `doi:10.1007/0-387-30065-1_16
       <https://doi.org/10.1007/0-387-30065-1_16>`_.
    """

    def __init__(self, interpolation, values, debug):
        """
        Initialize the quadratic model.

        Parameters
        ----------
        interpolation : `cobyqa.models.Interpolation`
            Interpolation set.
        values : `numpy.ndarray`, shape (npt,)
            Values of the interpolated function at the interpolation points.
        debug : bool
            Whether to make debugging tests during the execution.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the interpolation system is ill-defined.
        """
        self._debug = debug
        if self._debug:
            assert values.shape == (
                interpolation.npt,
            ), "The shape of `values` is not valid."
        if interpolation.npt < interpolation.n + 1:
            raise ValueError(
                f"The number of interpolation points must be at least "
                f"{interpolation.n + 1}."
            )
        self._const, self._grad, self._i_hess, _ = self._get_model(
            interpolation,
            values,
        )
        self._e_hess = np.zeros((self.n, self.n))

    def __call__(self, x, interpolation):
        """
        Evaluate the quadratic model at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which the quadratic model is evaluated.
        interpolation : `cobyqa.models.Interpolation`
            Interpolation set.

        Returns
        -------
        float
            Value of the quadratic model at `x`.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
        x_diff = x - interpolation.x_base
        return (
            self._const
            + self._grad @ x_diff
            + 0.5
            * (
                self._i_hess @ (interpolation.xpt.T @ x_diff) ** 2.0
                + x_diff @ self._e_hess @ x_diff
            )
        )

    @property
    def n(self):
        """
        Number of variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self._grad.size

    @property
    def npt(self):
        """
        Number of interpolation points used to define the quadratic model.

        Returns
        -------
        int
            Number of interpolation points used to define the quadratic model.
        """
        return self._i_hess.size

    def grad(self, x, interpolation):
        """
        Evaluate the gradient of the quadratic model at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which the gradient of the quadratic model is evaluated.
        interpolation : `cobyqa.models.Interpolation`
            Interpolation set.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Gradient of the quadratic model at `x`.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
        x_diff = x - interpolation.x_base
        return self._grad + self.hess_prod(x_diff, interpolation)

    def hess(self, interpolation):
        """
        Evaluate the Hessian matrix of the quadratic model.

        Parameters
        ----------
        interpolation : `cobyqa.models.Interpolation`
            Interpolation set.

        Returns
        -------
        `numpy.ndarray`, shape (n, n)
            Hessian matrix of the quadratic model.
        """
        return self._e_hess + interpolation.xpt @ (
            self._i_hess[:, np.newaxis] * interpolation.xpt.T
        )

    def hess_prod(self, v, interpolation):
        """
        Evaluate the right product of the Hessian matrix of the quadratic model
        with a given vector.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Vector with which the Hessian matrix of the quadratic model is
            multiplied from the right.
        interpolation : `cobyqa.models.Interpolation`
            Interpolation set.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Right product of the Hessian matrix of the quadratic model with
            `v`.
        """
        if self._debug:
            assert v.shape == (self.n,), "The shape of `v` is not valid."
        return self._e_hess @ v + interpolation.xpt @ (
            self._i_hess * (interpolation.xpt.T @ v)
        )

    def curv(self, v, interpolation):
        """
        Evaluate the curvature of the quadratic model along a given direction.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Direction along which the curvature of the quadratic model is
            evaluated.
        interpolation : `cobyqa.models.Interpolation`
            Interpolation set.

        Returns
        -------
        float
            Curvature of the quadratic model along `v`.
        """
        if self._debug:
            assert v.shape == (self.n,), "The shape of `v` is not valid."
        return (
            v @ self._e_hess @ v
            + self._i_hess @ (interpolation.xpt.T @ v) ** 2.0
        )

    def update(self, interpolation, k_new, dir_old, values_diff):
        """
        Update the quadratic model.

        This method applies the derivative-free symmetric Broyden update to the
        quadratic model. The `knew`-th interpolation point must be updated
        before calling this method.

        Parameters
        ----------
        interpolation : `cobyqa.models.Interpolation`
            Updated interpolation set.
        k_new : int
            Index of the updated interpolation point.
        dir_old : `numpy.ndarray`, shape (n,)
            Value of ``interpolation.xpt[:, k_new]`` before the update.
        values_diff : `numpy.ndarray`, shape (npt,)
            Differences between the values of the interpolated nonlinear
            function and the previous quadratic model at the updated
            interpolation points.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the interpolation system is ill-defined.
        """
        if self._debug:
            assert 0 <= k_new < self.npt, "The index `k_new` is not valid."
            assert dir_old.shape == (
                self.n,
            ), "The shape of `dir_old` is not valid."
            assert values_diff.shape == (
                self.npt,
            ), "The shape of `values_diff` is not valid."

        # Forward the k_new-th element of the implicit Hessian matrix to the
        # explicit Hessian matrix. This must be done because the implicit
        # Hessian matrix is related to the interpolation points, and the
        # k_new-th interpolation point is modified.
        self._e_hess += self._i_hess[k_new] * np.outer(dir_old, dir_old)
        self._i_hess[k_new] = 0.0

        # Update the quadratic model.
        const, grad, i_hess, ill_conditioned = self._get_model(
            interpolation,
            values_diff,
        )
        self._const += const
        self._grad += grad
        self._i_hess += i_hess
        return ill_conditioned

    def shift_x_base(self, interpolation, new_x_base):
        """
        Shift the point around which the quadratic model is defined.

        Parameters
        ----------
        interpolation : `cobyqa.models.Interpolation`
            Previous interpolation set.
        new_x_base : `numpy.ndarray`, shape (n,)
            Point that will replace ``interpolation.x_base``.
        """
        if self._debug:
            assert new_x_base.shape == (
                self.n,
            ), "The shape of `new_x_base` is not valid."
        self._const = self(new_x_base, interpolation)
        self._grad = self.grad(new_x_base, interpolation)
        shift = new_x_base - interpolation.x_base
        update = np.outer(
            shift,
            (interpolation.xpt - 0.5 * shift[:, np.newaxis]) @ self._i_hess,
        )
        self._e_hess += update + update.T

    @staticmethod
    def solve_systems(interpolation, rhs):
        """
        Solve the interpolation systems.

        Parameters
        ----------
        interpolation : `cobyqa.models.Interpolation`
            Interpolation set.
        rhs : `numpy.ndarray`, shape (npt + n + 1, m)
            Right-hand side vectors of the ``m`` interpolation systems.

        Returns
        -------
        `numpy.ndarray`, shape (npt + n + 1, m)
            Solutions of the interpolation systems.
        `numpy.ndarray`, shape (m, )
            Whether the interpolation systems are ill-conditioned.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the interpolation systems are ill-defined.
        """
        n, npt = interpolation.xpt.shape
        assert (
            rhs.ndim == 2 and rhs.shape[0] == npt + n + 1
        ), "The shape of `rhs` is not valid."

        # Build the left-hand side matrix of the interpolation system. The
        # matrix below stores diag(left_scaling) * W * diag(right_scaling),
        # where W is the theoretical matrix of the interpolation system. The
        # left and right scaling matrices are chosen to keep the elements in
        # the matrix well-balanced.
        a, right_scaling, eig = build_system(interpolation)

        # Build the solution. After a discussion with Mike Saunders and Alexis
        # Montoison during their visit to the Hong Kong Polytechnic University
        # in 2024, we decided to use the eigendecomposition of the symmetric
        # matrix a. This is more stable than the previously employed LBL
        # decomposition, and allows us to directly detect ill-conditioning of
        # the system and to build the least-squares solution if necessary.
        # Numerical experiments have shown that this strategy improves the
        # performance of the solver.
        rhs_scaled = rhs * right_scaling[:, np.newaxis]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(rhs_scaled))):
            raise np.linalg.LinAlgError(
                "The interpolation system is ill-defined."
            )

        # calculated in build_system
        eig_values, eig_vectors = eig

        large_eig_values = np.abs(eig_values) > EPS
        eig_vectors = eig_vectors[:, large_eig_values]
        inv_eig_values = 1.0 / eig_values[large_eig_values]
        ill_conditioned = ~np.all(large_eig_values, 0)
        left_scaled_solutions = eig_vectors @ (
            (eig_vectors.T @ rhs_scaled) * inv_eig_values[:, np.newaxis]
        )
        return (
            left_scaled_solutions * right_scaling[:, np.newaxis],
            ill_conditioned,
        )

    @staticmethod
    def _get_model(interpolation, values):
        """
        Solve the interpolation system.

        Parameters
        ----------
        interpolation : `cobyqa.models.Interpolation`
            Interpolation set.
        values : `numpy.ndarray`, shape (npt,)
            Values of the interpolated function at the interpolation points.

        Returns
        -------
        float
            Constant term of the quadratic model.
        `numpy.ndarray`, shape (n,)
            Gradient of the quadratic model at ``interpolation.x_base``.
        `numpy.ndarray`, shape (npt,)
            Implicit Hessian matrix of the quadratic model.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the interpolation system is ill-defined.
        """
        assert values.shape == (
            interpolation.npt,
        ), "The shape of `values` is not valid."
        n, npt = interpolation.xpt.shape
        x, ill_conditioned = Quadratic.solve_systems(
            interpolation,
            np.block(
                [
                    [
                        values,
                        np.zeros(n + 1),
                    ]
                ]
            ).T,
        )
        return x[npt, 0], x[npt + 1:, 0], x[:npt, 0], ill_conditioned


class Models:
    """
    Models for a nonlinear optimization problem.
    """

    def __init__(self, pb, options, penalty):
        """
        Initialize the models.

        Parameters
        ----------
        pb : `cobyqa.problem.Problem`
            Problem to be solved.
        options : dict
            Options of the solver.
        penalty : float
            Penalty parameter used to select the point in the filter to forward
            to the callback function.

        Raises
        ------
        `cobyqa.utils.MaxEvalError`
            If the maximum number of evaluations is reached.
        `cobyqa.utils.TargetSuccess`
            If a nearly feasible point has been found with an objective
            function value below the target.
        `cobyqa.utils.FeasibleSuccess`
            If a feasible point has been found for a feasibility problem.
        `numpy.linalg.LinAlgError`
            If the interpolation system is ill-defined.
        """
        # Set the initial interpolation set.
        self._debug = options[Options.DEBUG]
        self._interpolation = Interpolation(pb, options)

        # Evaluate the nonlinear functions at the initial interpolation points.
        x_eval = self.interpolation.point(0)
        fun_init, cub_init, ceq_init = pb(x_eval, penalty)
        self._fun_val = np.full(options[Options.NPT], np.nan)
        self._cub_val = np.full((options[Options.NPT], cub_init.size), np.nan)
        self._ceq_val = np.full((options[Options.NPT], ceq_init.size), np.nan)
        for k in range(options[Options.NPT]):
            if k >= options[Options.MAX_EVAL]:
                raise MaxEvalError
            if k == 0:
                self.fun_val[k] = fun_init
                self.cub_val[k, :] = cub_init
                self.ceq_val[k, :] = ceq_init
            else:
                x_eval = self.interpolation.point(k)
                self.fun_val[k], self.cub_val[k, :], self.ceq_val[k, :] = pb(
                    x_eval,
                    penalty,
                )

            # Stop the iterations if the problem is a feasibility problem and
            # the current interpolation point is feasible.
            if (
                pb.is_feasibility
                and pb.maxcv(
                    self.interpolation.point(k),
                    self.cub_val[k, :],
                    self.ceq_val[k, :],
                )
                <= options[Options.FEASIBILITY_TOL]
            ):
                raise FeasibleSuccess

            # Stop the iterations if the current interpolation point is nearly
            # feasible and has an objective function value below the target.
            if (
                self._fun_val[k] <= options[Options.TARGET]
                and pb.maxcv(
                    self.interpolation.point(k),
                    self.cub_val[k, :],
                    self.ceq_val[k, :],
                )
                <= options[Options.FEASIBILITY_TOL]
            ):
                raise TargetSuccess

        # Build the initial quadratic models.
        self._fun = Quadratic(
            self.interpolation,
            self._fun_val,
            options[Options.DEBUG],
        )
        self._cub = np.empty(self.m_nonlinear_ub, dtype=Quadratic)
        self._ceq = np.empty(self.m_nonlinear_eq, dtype=Quadratic)
        for i in range(self.m_nonlinear_ub):
            self._cub[i] = Quadratic(
                self.interpolation,
                self.cub_val[:, i],
                options[Options.DEBUG],
            )
        for i in range(self.m_nonlinear_eq):
            self._ceq[i] = Quadratic(
                self.interpolation,
                self.ceq_val[:, i],
                options[Options.DEBUG],
            )
        if self._debug:
            self._check_interpolation_conditions()

    @property
    def n(self):
        """
        Dimension of the problem.

        Returns
        -------
        int
            Dimension of the problem.
        """
        return self.interpolation.n

    @property
    def npt(self):
        """
        Number of interpolation points.

        Returns
        -------
        int
            Number of interpolation points.
        """
        return self.interpolation.npt

    @property
    def m_nonlinear_ub(self):
        """
        Number of nonlinear inequality constraints.

        Returns
        -------
        int
            Number of nonlinear inequality constraints.
        """
        return self.cub_val.shape[1]

    @property
    def m_nonlinear_eq(self):
        """
        Number of nonlinear equality constraints.

        Returns
        -------
        int
            Number of nonlinear equality constraints.
        """
        return self.ceq_val.shape[1]

    @property
    def interpolation(self):
        """
        Interpolation set.

        Returns
        -------
        `cobyqa.models.Interpolation`
            Interpolation set.
        """
        return self._interpolation

    @property
    def fun_val(self):
        """
        Values of the objective function at the interpolation points.

        Returns
        -------
        `numpy.ndarray`, shape (npt,)
            Values of the objective function at the interpolation points.
        """
        return self._fun_val

    @property
    def cub_val(self):
        """
        Values of the nonlinear inequality constraint functions at the
        interpolation points.

        Returns
        -------
        `numpy.ndarray`, shape (npt, m_nonlinear_ub)
            Values of the nonlinear inequality constraint functions at the
            interpolation points.
        """
        return self._cub_val

    @property
    def ceq_val(self):
        """
        Values of the nonlinear equality constraint functions at the
        interpolation points.

        Returns
        -------
        `numpy.ndarray`, shape (npt, m_nonlinear_eq)
            Values of the nonlinear equality constraint functions at the
            interpolation points.
        """
        return self._ceq_val

    def fun(self, x):
        """
        Evaluate the quadratic model of the objective function at a given
        point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which to evaluate the quadratic model of the objective
            function.

        Returns
        -------
        float
            Value of the quadratic model of the objective function at `x`.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
        return self._fun(x, self.interpolation)

    def fun_grad(self, x):
        """
        Evaluate the gradient of the quadratic model of the objective function
        at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which to evaluate the gradient of the quadratic model of
            the objective function.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Gradient of the quadratic model of the objective function at `x`.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
        return self._fun.grad(x, self.interpolation)

    def fun_hess(self):
        """
        Evaluate the Hessian matrix of the quadratic model of the objective
        function.

        Returns
        -------
        `numpy.ndarray`, shape (n, n)
            Hessian matrix of the quadratic model of the objective function.
        """
        return self._fun.hess(self.interpolation)

    def fun_hess_prod(self, v):
        """
        Evaluate the right product of the Hessian matrix of the quadratic model
        of the objective function with a given vector.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Vector with which the Hessian matrix of the quadratic model of the
            objective function is multiplied from the right.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Right product of the Hessian matrix of the quadratic model of the
            objective function with `v`.
        """
        if self._debug:
            assert v.shape == (self.n,), "The shape of `v` is not valid."
        return self._fun.hess_prod(v, self.interpolation)

    def fun_curv(self, v):
        """
        Evaluate the curvature of the quadratic model of the objective function
        along a given direction.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Direction along which the curvature of the quadratic model of the
            objective function is evaluated.

        Returns
        -------
        float
            Curvature of the quadratic model of the objective function along
            `v`.
        """
        if self._debug:
            assert v.shape == (self.n,), "The shape of `v` is not valid."
        return self._fun.curv(v, self.interpolation)

    def fun_alt_grad(self, x):
        """
        Evaluate the gradient of the alternative quadratic model of the
        objective function at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which to evaluate the gradient of the alternative
            quadratic model of the objective function.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Gradient of the alternative quadratic model of the objective
            function at `x`.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the interpolation system is ill-defined.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
        model = Quadratic(self.interpolation, self.fun_val, self._debug)
        return model.grad(x, self.interpolation)

    def cub(self, x, mask=None):
        """
        Evaluate the quadratic models of the nonlinear inequality functions at
        a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which to evaluate the quadratic models of the nonlinear
            inequality functions.
        mask : `numpy.ndarray`, shape (m_nonlinear_ub,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Values of the quadratic model of the nonlinear inequality
            functions.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
            assert mask is None or mask.shape == (
                self.m_nonlinear_ub,
            ), "The shape of `mask` is not valid."
        return np.array(
            [model(x, self.interpolation) for model in self._get_cub(mask)]
        )

    def cub_grad(self, x, mask=None):
        """
        Evaluate the gradients of the quadratic models of the nonlinear
        inequality functions at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which to evaluate the gradients of the quadratic models of
            the nonlinear inequality functions.
        mask : `numpy.ndarray`, shape (m_nonlinear_eq,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Gradients of the quadratic model of the nonlinear inequality
            functions.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
            assert mask is None or mask.shape == (
                self.m_nonlinear_ub,
            ), "The shape of `mask` is not valid."
        return np.reshape(
            [model.grad(x, self.interpolation)
             for model in self._get_cub(mask)],
            (-1, self.n),
        )

    def cub_hess(self, mask=None):
        """
        Evaluate the Hessian matrices of the quadratic models of the nonlinear
        inequality functions.

        Parameters
        ----------
        mask : `numpy.ndarray`, shape (m_nonlinear_ub,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Hessian matrices of the quadratic models of the nonlinear
            inequality functions.
        """
        if self._debug:
            assert mask is None or mask.shape == (
                self.m_nonlinear_ub,
            ), "The shape of `mask` is not valid."
        return np.reshape(
            [model.hess(self.interpolation) for model in self._get_cub(mask)],
            (-1, self.n, self.n),
        )

    def cub_hess_prod(self, v, mask=None):
        """
        Evaluate the right product of the Hessian matrices of the quadratic
        models of the nonlinear inequality functions with a given vector.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Vector with which the Hessian matrices of the quadratic models of
            the nonlinear inequality functions are multiplied from the right.
        mask : `numpy.ndarray`, shape (m_nonlinear_ub,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Right products of the Hessian matrices of the quadratic models of
            the nonlinear inequality functions with `v`.
        """
        if self._debug:
            assert v.shape == (self.n,), "The shape of `v` is not valid."
            assert mask is None or mask.shape == (
                self.m_nonlinear_ub,
            ), "The shape of `mask` is not valid."
        return np.reshape(
            [
                model.hess_prod(v, self.interpolation)
                for model in self._get_cub(mask)
            ],
            (-1, self.n),
        )

    def cub_curv(self, v, mask=None):
        """
        Evaluate the curvature of the quadratic models of the nonlinear
        inequality functions along a given direction.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Direction along which the curvature of the quadratic models of the
            nonlinear inequality functions is evaluated.
        mask : `numpy.ndarray`, shape (m_nonlinear_ub,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Curvature of the quadratic models of the nonlinear inequality
            functions along `v`.
        """
        if self._debug:
            assert v.shape == (self.n,), "The shape of `v` is not valid."
            assert mask is None or mask.shape == (
                self.m_nonlinear_ub,
            ), "The shape of `mask` is not valid."
        return np.array(
            [model.curv(v, self.interpolation)
             for model in self._get_cub(mask)]
        )

    def ceq(self, x, mask=None):
        """
        Evaluate the quadratic models of the nonlinear equality functions at a
        given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which to evaluate the quadratic models of the nonlinear
            equality functions.
        mask : `numpy.ndarray`, shape (m_nonlinear_eq,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Values of the quadratic model of the nonlinear equality functions.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
            assert mask is None or mask.shape == (
                self.m_nonlinear_eq,
            ), "The shape of `mask` is not valid."
        return np.array(
            [model(x, self.interpolation) for model in self._get_ceq(mask)]
        )

    def ceq_grad(self, x, mask=None):
        """
        Evaluate the gradients of the quadratic models of the nonlinear
        equality functions at a given point.

        Parameters
        ----------
        x : `numpy.ndarray`, shape (n,)
            Point at which to evaluate the gradients of the quadratic models of
            the nonlinear equality functions.
        mask : `numpy.ndarray`, shape (m_nonlinear_eq,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Gradients of the quadratic model of the nonlinear equality
            functions.
        """
        if self._debug:
            assert x.shape == (self.n,), "The shape of `x` is not valid."
            assert mask is None or mask.shape == (
                self.m_nonlinear_eq,
            ), "The shape of `mask` is not valid."
        return np.reshape(
            [model.grad(x, self.interpolation)
             for model in self._get_ceq(mask)],
            (-1, self.n),
        )

    def ceq_hess(self, mask=None):
        """
        Evaluate the Hessian matrices of the quadratic models of the nonlinear
        equality functions.

        Parameters
        ----------
        mask : `numpy.ndarray`, shape (m_nonlinear_eq,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Hessian matrices of the quadratic models of the nonlinear equality
            functions.
        """
        if self._debug:
            assert mask is None or mask.shape == (
                self.m_nonlinear_eq,
            ), "The shape of `mask` is not valid."
        return np.reshape(
            [model.hess(self.interpolation) for model in self._get_ceq(mask)],
            (-1, self.n, self.n),
        )

    def ceq_hess_prod(self, v, mask=None):
        """
        Evaluate the right product of the Hessian matrices of the quadratic
        models of the nonlinear equality functions with a given vector.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Vector with which the Hessian matrices of the quadratic models of
            the nonlinear equality functions are multiplied from the right.
        mask : `numpy.ndarray`, shape (m_nonlinear_eq,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Right products of the Hessian matrices of the quadratic models of
            the nonlinear equality functions with `v`.
        """
        if self._debug:
            assert v.shape == (self.n,), "The shape of `v` is not valid."
            assert mask is None or mask.shape == (
                self.m_nonlinear_eq,
            ), "The shape of `mask` is not valid."
        return np.reshape(
            [
                model.hess_prod(v, self.interpolation)
                for model in self._get_ceq(mask)
            ],
            (-1, self.n),
        )

    def ceq_curv(self, v, mask=None):
        """
        Evaluate the curvature of the quadratic models of the nonlinear
        equality functions along a given direction.

        Parameters
        ----------
        v : `numpy.ndarray`, shape (n,)
            Direction along which the curvature of the quadratic models of the
            nonlinear equality functions is evaluated.
        mask : `numpy.ndarray`, shape (m_nonlinear_eq,), optional
            Mask of the quadratic models to consider.

        Returns
        -------
        `numpy.ndarray`
            Curvature of the quadratic models of the nonlinear equality
            functions along `v`.
        """
        if self._debug:
            assert v.shape == (self.n,), "The shape of `v` is not valid."
            assert mask is None or mask.shape == (
                self.m_nonlinear_eq,
            ), "The shape of `mask` is not valid."
        return np.array(
            [model.curv(v, self.interpolation)
             for model in self._get_ceq(mask)]
        )

    def reset_models(self):
        """
        Set the quadratic models of the objective function, nonlinear
        inequality constraints, and nonlinear equality constraints to the
        alternative quadratic models.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the interpolation system is ill-defined.
        """
        self._fun = Quadratic(self.interpolation, self.fun_val, self._debug)
        for i in range(self.m_nonlinear_ub):
            self._cub[i] = Quadratic(
                self.interpolation,
                self.cub_val[:, i],
                self._debug,
            )
        for i in range(self.m_nonlinear_eq):
            self._ceq[i] = Quadratic(
                self.interpolation,
                self.ceq_val[:, i],
                self._debug,
            )
        if self._debug:
            self._check_interpolation_conditions()

    def update_interpolation(self, k_new, x_new, fun_val, cub_val, ceq_val):
        """
        Update the interpolation set.

        This method updates the interpolation set by replacing the `knew`-th
        interpolation point with `xnew`. It also updates the function values
        and the quadratic models.

        Parameters
        ----------
        k_new : int
            Index of the updated interpolation point.
        x_new : `numpy.ndarray`, shape (n,)
            New interpolation point. Its value is interpreted as relative to
            the origin, not the base point.
        fun_val : float
            Value of the objective function at `x_new`.
            Objective function value at `x_new`.
        cub_val : `numpy.ndarray`, shape (m_nonlinear_ub,)
            Values of the nonlinear inequality constraints at `x_new`.
        ceq_val : `numpy.ndarray`, shape (m_nonlinear_eq,)
            Values of the nonlinear equality constraints at `x_new`.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the interpolation system is ill-defined.
        """
        if self._debug:
            assert 0 <= k_new < self.npt, "The index `k_new` is not valid."
            assert x_new.shape == (self.n,), \
                "The shape of `x_new` is not valid."
            assert isinstance(fun_val, float), \
                "The function value is not valid."
            assert cub_val.shape == (
                self.m_nonlinear_ub,
            ), "The shape of `cub_val` is not valid."
            assert ceq_val.shape == (
                self.m_nonlinear_eq,
            ), "The shape of `ceq_val` is not valid."

        # Compute the updates in the interpolation conditions.
        fun_diff = np.zeros(self.npt)
        cub_diff = np.zeros(self.cub_val.shape)
        ceq_diff = np.zeros(self.ceq_val.shape)
        fun_diff[k_new] = fun_val - self.fun(x_new)
        cub_diff[k_new, :] = cub_val - self.cub(x_new)
        ceq_diff[k_new, :] = ceq_val - self.ceq(x_new)

        # Update the function values.
        self.fun_val[k_new] = fun_val
        self.cub_val[k_new, :] = cub_val
        self.ceq_val[k_new, :] = ceq_val

        # Update the interpolation set.
        dir_old = np.copy(self.interpolation.xpt[:, k_new])
        self.interpolation.xpt[:, k_new] = x_new - self.interpolation.x_base

        # Update the quadratic models.
        ill_conditioned = self._fun.update(
            self.interpolation,
            k_new,
            dir_old,
            fun_diff,
        )
        for i in range(self.m_nonlinear_ub):
            ill_conditioned = ill_conditioned or self._cub[i].update(
                self.interpolation,
                k_new,
                dir_old,
                cub_diff[:, i],
            )
        for i in range(self.m_nonlinear_eq):
            ill_conditioned = ill_conditioned or self._ceq[i].update(
                self.interpolation,
                k_new,
                dir_old,
                ceq_diff[:, i],
            )
        if self._debug:
            self._check_interpolation_conditions()
        return ill_conditioned

    def determinants(self, x_new, k_new=None):
        """
        Compute the normalized determinants of the new interpolation systems.

        Parameters
        ----------
        x_new : `numpy.ndarray`, shape (n,)
            New interpolation point. Its value is interpreted as relative to
            the origin, not the base point.
        k_new : int, optional
            Index of the updated interpolation point. If `k_new` is not
            specified, all the possible determinants are computed.

        Returns
        -------
        {float, `numpy.ndarray`, shape (npt,)}
            Determinant(s) of the new interpolation system.

        Raises
        ------
        `numpy.linalg.LinAlgError`
            If the interpolation system is ill-defined.

        Notes
        -----
        The determinants are normalized by the determinant of the current
        interpolation system. For stability reasons, the calculations are done
        using the formula (2.12) in [1]_.

        References
        ----------
        .. [1] M. J. D. Powell. On updating the inverse of a KKT matrix.
           Technical Report DAMTP 2004/NA01, Department of Applied Mathematics
           and Theoretical Physics, University of Cambridge, Cambridge, UK,
           2004.
        """
        if self._debug:
            assert x_new.shape == (self.n,), \
                "The shape of `x_new` is not valid."
            assert (
                k_new is None or 0 <= k_new < self.npt
            ), "The index `k_new` is not valid."

        # Compute the values independent of k_new.
        shift = x_new - self.interpolation.x_base
        new_col = np.empty((self.npt + self.n + 1, 1))
        new_col[: self.npt, 0] = (
                0.5 * (self.interpolation.xpt.T @ shift) ** 2.0)
        new_col[self.npt, 0] = 1.0
        new_col[self.npt + 1:, 0] = shift
        inv_new_col = Quadratic.solve_systems(self.interpolation, new_col)[0]
        beta = 0.5 * (shift @ shift) ** 2.0 - new_col[:, 0] @ inv_new_col[:, 0]

        # Compute the values that depend on k.
        if k_new is None:
            coord_vec = np.eye(self.npt + self.n + 1, self.npt)
            alpha = np.diag(
                Quadratic.solve_systems(
                    self.interpolation,
                    coord_vec,
                )[0]
            )
            tau = inv_new_col[: self.npt, 0]
        else:
            coord_vec = np.eye(self.npt + self.n + 1, 1, -k_new)
            alpha = Quadratic.solve_systems(
                self.interpolation,
                coord_vec,
            )[
                0
            ][k_new, 0]
            tau = inv_new_col[k_new, 0]
        return alpha * beta + tau**2.0

    def shift_x_base(self, new_x_base, options):
        """
        Shift the base point without changing the interpolation set.

        Parameters
        ----------
        new_x_base : `numpy.ndarray`, shape (n,)
            New base point.
        options : dict
            Options of the solver.
        """
        if self._debug:
            assert new_x_base.shape == (
                self.n,
            ), "The shape of `new_x_base` is not valid."

        # Update the models.
        self._fun.shift_x_base(self.interpolation, new_x_base)
        for model in self._cub:
            model.shift_x_base(self.interpolation, new_x_base)
        for model in self._ceq:
            model.shift_x_base(self.interpolation, new_x_base)

        # Update the base point and the interpolation points.
        shift = new_x_base - self.interpolation.x_base
        self.interpolation.x_base += shift
        self.interpolation.xpt -= shift[:, np.newaxis]
        if options[Options.DEBUG]:
            self._check_interpolation_conditions()

    def _get_cub(self, mask=None):
        """
        Get the quadratic models of the nonlinear inequality constraints.

        Parameters
        ----------
        mask : `numpy.ndarray`, shape (m_nonlinear_ub,), optional
            Mask of the quadratic models to return.

        Returns
        -------
        `numpy.ndarray`
            Quadratic models of the nonlinear inequality constraints.
        """
        return self._cub if mask is None else self._cub[mask]

    def _get_ceq(self, mask=None):
        """
        Get the quadratic models of the nonlinear equality constraints.

        Parameters
        ----------
        mask : `numpy.ndarray`, shape (m_nonlinear_eq,), optional
            Mask of the quadratic models to return.

        Returns
        -------
        `numpy.ndarray`
            Quadratic models of the nonlinear equality constraints.
        """
        return self._ceq if mask is None else self._ceq[mask]

    def _check_interpolation_conditions(self):
        """
        Check the interpolation conditions of all quadratic models.
        """
        error_fun = 0.0
        error_cub = 0.0
        error_ceq = 0.0
        for k in range(self.npt):
            error_fun = np.max(
                [
                    error_fun,
                    np.abs(
                        self.fun(self.interpolation.point(k)) - self.fun_val[k]
                    ),
                ]
            )
            error_cub = np.max(
                np.abs(
                    self.cub(self.interpolation.point(k)) - self.cub_val[k, :]
                ),
                initial=error_cub,
            )
            error_ceq = np.max(
                np.abs(
                    self.ceq(self.interpolation.point(k)) - self.ceq_val[k, :]
                ),
                initial=error_ceq,
            )
        tol = 10.0 * np.sqrt(EPS) * max(self.n, self.npt)
        if error_fun > tol * np.max(np.abs(self.fun_val), initial=1.0):
            warnings.warn(
                "The interpolation conditions for the objective function are "
                "not satisfied.",
                RuntimeWarning,
                2,
            )
        if error_cub > tol * np.max(np.abs(self.cub_val), initial=1.0):
            warnings.warn(
                "The interpolation conditions for the inequality constraint "
                "function are not satisfied.",
                RuntimeWarning,
                2,
            )
        if error_ceq > tol * np.max(np.abs(self.ceq_val), initial=1.0):
            warnings.warn(
                "The interpolation conditions for the equality constraint "
                "function are not satisfied.",
                RuntimeWarning,
                2,
            )
<|endoftext|>from contextlib import suppress
from inspect import signature
import copy

import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
)
from scipy.optimize._constraints import PreparedConstraint


from .settings import PRINT_OPTIONS, BARRIER
from .utils import CallbackSuccess, get_arrays_tol
from .utils import exact_1d_array


class ObjectiveFunction:
    """
    Real-valued objective function.
    """

    def __init__(self, fun, verbose, debug, *args):
        """
        Initialize the objective function.

        Parameters
        ----------
        fun : {callable, None}
            Function to evaluate, or None.

                ``fun(x, *args) -> float``

            where ``x`` is an array with shape (n,) and `args` is a tuple.
        verbose : bool
            Whether to print the function evaluations.
        debug : bool
            Whether to make debugging tests during the execution.
        *args : tuple
            Additional arguments to be passed to the function.
        """
        if debug:
            assert fun is None or callable(fun)
            assert isinstance(verbose, bool)
            assert isinstance(debug, bool)

        self._fun = fun
        self._verbose = verbose
        self._args = args
        self._n_eval = 0

    def __call__(self, x):
        """
        Evaluate the objective function.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the objective function is evaluated.

        Returns
        -------
        float
            Function value at `x`.
        """
        x = np.array(x, dtype=float)
        if self._fun is None:
            f = 0.0
        else:
            f = float(np.squeeze(self._fun(x, *self._args)))
            self._n_eval += 1
            if self._verbose:
                with np.printoptions(**PRINT_OPTIONS):
                    print(f"{self.name}({x}) = {f}")
        return f

    @property
    def n_eval(self):
        """
        Number of function evaluations.

        Returns
        -------
        int
            Number of function evaluations.
        """
        return self._n_eval

    @property
    def name(self):
        """
        Name of the objective function.

        Returns
        -------
        str
            Name of the objective function.
        """
        name = ""
        if self._fun is not None:
            try:
                name = self._fun.__name__
            except AttributeError:
                name = "fun"
        return name


class BoundConstraints:
    """
    Bound constraints ``xl <= x <= xu``.
    """

    def __init__(self, bounds):
        """
        Initialize the bound constraints.

        Parameters
        ----------
        bounds : scipy.optimize.Bounds
            Bound constraints.
        """
        self._xl = np.array(bounds.lb, float)
        self._xu = np.array(bounds.ub, float)

        # Remove the ill-defined bounds.
        self.xl[np.isnan(self.xl)] = -np.inf
        self.xu[np.isnan(self.xu)] = np.inf

        self.is_feasible = (
            np.all(self.xl <= self.xu)
            and np.all(self.xl < np.inf)
            and np.all(self.xu > -np.inf)
        )
        self.m = np.count_nonzero(self.xl > -np.inf) + np.count_nonzero(
            self.xu < np.inf
        )
        self.pcs = PreparedConstraint(bounds, np.ones(bounds.lb.size))

    @property
    def xl(self):
        """
        Lower bound.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Lower bound.
        """
        return self._xl

    @property
    def xu(self):
        """
        Upper bound.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Upper bound.
        """
        return self._xu

    def maxcv(self, x):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
        """
        x = np.asarray(x, dtype=float)
        return self.violation(x)

    def violation(self, x):
        # shortcut for no bounds
        if self.is_feasible:
            return np.array([0])
        else:
            return self.pcs.violation(x)

    def project(self, x):
        """
        Project a point onto the feasible set.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point to be projected.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Projection of `x` onto the feasible set.
        """
        return np.clip(x, self.xl, self.xu) if self.is_feasible else x


class LinearConstraints:
    """
    Linear constraints ``a_ub @ x <= b_ub`` and ``a_eq @ x == b_eq``.
    """

    def __init__(self, constraints, n, debug):
        """
        Initialize the linear constraints.

        Parameters
        ----------
        constraints : list of LinearConstraint
            Linear constraints.
        n : int
            Number of variables.
        debug : bool
            Whether to make debugging tests during the execution.
        """
        if debug:
            assert isinstance(constraints, list)
            for constraint in constraints:
                assert isinstance(constraint, LinearConstraint)
            assert isinstance(debug, bool)

        self._a_ub = np.empty((0, n))
        self._b_ub = np.empty(0)
        self._a_eq = np.empty((0, n))
        self._b_eq = np.empty(0)
        for constraint in constraints:
            is_equality = np.abs(
                constraint.ub - constraint.lb
            ) <= get_arrays_tol(constraint.lb, constraint.ub)
            if np.any(is_equality):
                self._a_eq = np.vstack((self.a_eq, constraint.A[is_equality]))
                self._b_eq = np.concatenate(
                    (
                        self.b_eq,
                        0.5
                        * (
                            constraint.lb[is_equality]
                            + constraint.ub[is_equality]
                        ),
                    )
                )
            if not np.all(is_equality):
                self._a_ub = np.vstack(
                    (
                        self.a_ub,
                        constraint.A[~is_equality],
                        -constraint.A[~is_equality],
                    )
                )
                self._b_ub = np.concatenate(
                    (
                        self.b_ub,
                        constraint.ub[~is_equality],
                        -constraint.lb[~is_equality],
                    )
                )

        # Remove the ill-defined constraints.
        self.a_ub[np.isnan(self.a_ub)] = 0.0
        self.a_eq[np.isnan(self.a_eq)] = 0.0
        undef_ub = np.isnan(self.b_ub) | np.isinf(self.b_ub)
        undef_eq = np.isnan(self.b_eq)
        self._a_ub = self.a_ub[~undef_ub, :]
        self._b_ub = self.b_ub[~undef_ub]
        self._a_eq = self.a_eq[~undef_eq, :]
        self._b_eq = self.b_eq[~undef_eq]
        self.pcs = [
            PreparedConstraint(c, np.ones(n)) for c in constraints if c.A.size
        ]

    @property
    def a_ub(self):
        """
        Left-hand side matrix of the linear inequality constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m, n)
            Left-hand side matrix of the linear inequality constraints.
        """
        return self._a_ub

    @property
    def b_ub(self):
        """
        Right-hand side vector of the linear inequality constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m, n)
            Right-hand side vector of the linear inequality constraints.
        """
        return self._b_ub

    @property
    def a_eq(self):
        """
        Left-hand side matrix of the linear equality constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m, n)
            Left-hand side matrix of the linear equality constraints.
        """
        return self._a_eq

    @property
    def b_eq(self):
        """
        Right-hand side vector of the linear equality constraints.

        Returns
        -------
        `numpy.ndarray`, shape (m, n)
            Right-hand side vector of the linear equality constraints.
        """
        return self._b_eq

    @property
    def m_ub(self):
        """
        Number of linear inequality constraints.

        Returns
        -------
        int
            Number of linear inequality constraints.
        """
        return self.b_ub.size

    @property
    def m_eq(self):
        """
        Number of linear equality constraints.

        Returns
        -------
        int
            Number of linear equality constraints.
        """
        return self.b_eq.size

    def maxcv(self, x):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
        """
        return np.max(self.violation(x), initial=0.0)

    def violation(self, x):
        if len(self.pcs):
            return np.concatenate([pc.violation(x) for pc in self.pcs])
        return np.array([])


class NonlinearConstraints:
    """
    Nonlinear constraints ``c_ub(x) <= 0`` and ``c_eq(x) == b_eq``.
    """

    def __init__(self, constraints, verbose, debug):
        """
        Initialize the nonlinear constraints.

        Parameters
        ----------
        constraints : list
            Nonlinear constraints.
        verbose : bool
            Whether to print the function evaluations.
        debug : bool
            Whether to make debugging tests during the execution.
        """
        if debug:
            assert isinstance(constraints, list)
            for constraint in constraints:
                assert isinstance(constraint, NonlinearConstraint)
            assert isinstance(verbose, bool)
            assert isinstance(debug, bool)

        self._constraints = constraints
        self.pcs = []
        self._verbose = verbose

        # map of indexes for equality and inequality constraints
        self._map_ub = None
        self._map_eq = None
        self._m_ub = self._m_eq = None

    def __call__(self, x):
        """
        Calculates the residual (slack) for the constraints.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the constraints are evaluated.

        Returns
        -------
        `numpy.ndarray`, shape (m_nonlinear_ub,)
            Nonlinear inequality constraint slack values.
        `numpy.ndarray`, shape (m_nonlinear_eq,)
            Nonlinear equality constraint slack values.
        """
        if not len(self._constraints):
            self._m_eq = self._m_ub = 0
            return np.array([]), np.array([])

        x = np.array(x, dtype=float)
        # first time around the constraints haven't been prepared
        if not len(self.pcs):
            self._map_ub = []
            self._map_eq = []
            self._m_eq = 0
            self._m_ub = 0

            for constraint in self._constraints:
                if not callable(constraint.jac):
                    # having a callable constraint function prevents
                    # constraint.fun from being evaluated when preparing
                    # constraint
                    c = copy.copy(constraint)
                    c.jac = lambda x0: x0
                    c.hess = lambda x0, v: 0.0
                    pc = PreparedConstraint(c, x)
                else:
                    pc = PreparedConstraint(constraint, x)
                # we're going to be using the same x value again immediately
                # after this initialisation
                pc.fun.f_updated = True

                self.pcs.append(pc)
                idx = np.arange(pc.fun.m)

                # figure out equality and inequality maps
                lb, ub = pc.bounds[0], pc.bounds[1]
                arr_tol = get_arrays_tol(lb, ub)
                is_equality = np.abs(ub - lb) <= arr_tol
                self._map_eq.append(idx[is_equality])
                self._map_ub.append(idx[~is_equality])

                # these values will be corrected to their proper values later
                self._m_eq += np.count_nonzero(is_equality)
                self._m_ub += np.count_nonzero(~is_equality)

        c_ub = []
        c_eq = []
        for i, pc in enumerate(self.pcs):
            val = pc.fun.fun(x)
            if self._verbose:
                with np.printoptions(**PRINT_OPTIONS):
                    with suppress(AttributeError):
                        fun_name = self._constraints[i].fun.__name__
                        print(f"{fun_name}({x}) = {val}")

            # separate violations into c_eq and c_ub
            eq_idx = self._map_eq[i]
            ub_idx = self._map_ub[i]

            ub_val = val[ub_idx]
            if len(ub_idx):
                xl = pc.bounds[0][ub_idx]
                xu = pc.bounds[1][ub_idx]

                # calculate slack within lower bound
                finite_xl = xl > -np.inf
                _v = xl[finite_xl] - ub_val[finite_xl]
                c_ub.append(_v)

                # calculate slack within lower bound
                finite_xu = xu < np.inf
                _v = ub_val[finite_xu] - xu[finite_xu]
                c_ub.append(_v)

            # equality constraints taken from midpoint between lb and ub
            eq_val = val[eq_idx]
            if len(eq_idx):
                midpoint = 0.5 * (pc.bounds[1][eq_idx] + pc.bounds[0][eq_idx])
                eq_val -= midpoint
            c_eq.append(eq_val)

        if self._m_eq:
            c_eq = np.concatenate(c_eq)
        else:
            c_eq = np.array([])

        if self._m_ub:
            c_ub = np.concatenate(c_ub)
        else:
            c_ub = np.array([])

        self._m_ub = c_ub.size
        self._m_eq = c_eq.size

        return c_ub, c_eq

    @property
    def m_ub(self):
        """
        Number of nonlinear inequality constraints.

        Returns
        -------
        int
            Number of nonlinear inequality constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear inequality constraints is unknown.
        """
        if self._m_ub is None:
            raise ValueError(
                "The number of nonlinear inequality constraints is unknown."
            )
        else:
            return self._m_ub

    @property
    def m_eq(self):
        """
        Number of nonlinear equality constraints.

        Returns
        -------
        int
            Number of nonlinear equality constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear equality constraints is unknown.
        """
        if self._m_eq is None:
            raise ValueError(
                "The number of nonlinear equality constraints is unknown."
            )
        else:
            return self._m_eq

    @property
    def n_eval(self):
        """
        Number of function evaluations.

        Returns
        -------
        int
            Number of function evaluations.
        """
        if len(self.pcs):
            return self.pcs[0].fun.nfev
        else:
            return 0

    def maxcv(self, x, cub_val=None, ceq_val=None):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.
        cub_val : array_like, shape (m_nonlinear_ub,), optional
            Values of the nonlinear inequality constraints. If not provided,
            the nonlinear inequality constraints are evaluated at `x`.
        ceq_val : array_like, shape (m_nonlinear_eq,), optional
            Values of the nonlinear equality constraints. If not provided,
            the nonlinear equality constraints are evaluated at `x`.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
        """
        return np.max(
            self.violation(x, cub_val=cub_val, ceq_val=ceq_val), initial=0.0
        )

    def violation(self, x, cub_val=None, ceq_val=None):
        return np.concatenate([pc.violation(x) for pc in self.pcs])


class Problem:
    """
    Optimization problem.
    """

    def __init__(
        self,
        obj,
        x0,
        bounds,
        linear,
        nonlinear,
        callback,
        feasibility_tol,
        scale,
        store_history,
        history_size,
        filter_size,
        debug,
    ):
        """
        Initialize the nonlinear problem.

        The problem is preprocessed to remove all the variables that are fixed
        by the bound constraints.

        Parameters
        ----------
        obj : ObjectiveFunction
            Objective function.
        x0 : array_like, shape (n,)
            Initial guess.
        bounds : BoundConstraints
            Bound constraints.
        linear : LinearConstraints
            Linear constraints.
        nonlinear : NonlinearConstraints
            Nonlinear constraints.
        callback : {callable, None}
            Callback function.
        feasibility_tol : float
            Tolerance on the constraint violation.
        scale : bool
            Whether to scale the problem according to the bounds.
        store_history : bool
            Whether to store the function evaluations.
        history_size : int
            Maximum number of function evaluations to store.
        filter_size : int
            Maximum number of points in the filter.
        debug : bool
            Whether to make debugging tests during the execution.
        """
        if debug:
            assert isinstance(obj, ObjectiveFunction)
            assert isinstance(bounds, BoundConstraints)
            assert isinstance(linear, LinearConstraints)
            assert isinstance(nonlinear, NonlinearConstraints)
            assert isinstance(feasibility_tol, float)
            assert isinstance(scale, bool)
            assert isinstance(store_history, bool)
            assert isinstance(history_size, int)
            if store_history:
                assert history_size > 0
            assert isinstance(filter_size, int)
            assert filter_size > 0
            assert isinstance(debug, bool)

        self._obj = obj
        self._linear = linear
        self._nonlinear = nonlinear
        if callback is not None:
            if not callable(callback):
                raise TypeError("The callback must be a callable function.")
        self._callback = callback

        # Check the consistency of the problem.
        x0 = exact_1d_array(x0, "The initial guess must be a vector.")
        n = x0.size
        if bounds.xl.size != n:
            raise ValueError(f"The bounds must have {n} elements.")
        if linear.a_ub.shape[1] != n:
            raise ValueError(
                f"The left-hand side matrices of the linear constraints must "
                f"have {n} columns."
            )

        # Check which variables are fixed.
        tol = get_arrays_tol(bounds.xl, bounds.xu)
        self._fixed_idx = (bounds.xl <= bounds.xu) & (
            np.abs(bounds.xl - bounds.xu) < tol
        )
        self._fixed_val = 0.5 * (
            bounds.xl[self._fixed_idx] + bounds.xu[self._fixed_idx]
        )
        self._fixed_val = np.clip(
            self._fixed_val,
            bounds.xl[self._fixed_idx],
            bounds.xu[self._fixed_idx],
        )

        # Set the bound constraints.
        self._orig_bounds = bounds
        self._bounds = BoundConstraints(
            Bounds(bounds.xl[~self._fixed_idx], bounds.xu[~self._fixed_idx])
        )

        # Set the initial guess.
        self._x0 = self._bounds.project(x0[~self._fixed_idx])

        # Set the linear constraints.
        b_eq = linear.b_eq - linear.a_eq[:, self._fixed_idx] @ self._fixed_val
        self._linear = LinearConstraints(
            [
                LinearConstraint(
                    linear.a_ub[:, ~self._fixed_idx],
                    -np.inf,
                    linear.b_ub
                    - linear.a_ub[:, self._fixed_idx] @ self._fixed_val,
                ),
                LinearConstraint(linear.a_eq[:, ~self._fixed_idx], b_eq, b_eq),
            ],
            self.n,
            debug,
        )

        # Scale the problem if necessary.
        scale = (
            scale
            and self._bounds.is_feasible
            and np.all(np.isfinite(self._bounds.xl))
            and np.all(np.isfinite(self._bounds.xu))
        )
        if scale:
            self._scaling_factor = 0.5 * (self._bounds.xu - self._bounds.xl)
            self._scaling_shift = 0.5 * (self._bounds.xu + self._bounds.xl)
            self._bounds = BoundConstraints(
                Bounds(-np.ones(self.n), np.ones(self.n))
            )
            b_eq = self._linear.b_eq - self._linear.a_eq @ self._scaling_shift
            self._linear = LinearConstraints(
                [
                    LinearConstraint(
                        self._linear.a_ub @ np.diag(self._scaling_factor),
                        -np.inf,
                        self._linear.b_ub
                        - self._linear.a_ub @ self._scaling_shift,
                    ),
                    LinearConstraint(
                        self._linear.a_eq @ np.diag(self._scaling_factor),
                        b_eq,
                        b_eq,
                    ),
                ],
                self.n,
                debug,
            )
            self._x0 = (self._x0 - self._scaling_shift) / self._scaling_factor
        else:
            self._scaling_factor = np.ones(self.n)
            self._scaling_shift = np.zeros(self.n)

        # Set the initial filter.
        self._feasibility_tol = feasibility_tol
        self._filter_size = filter_size
        self._fun_filter = []
        self._maxcv_filter = []
        self._x_filter = []

        # Set the initial history.
        self._store_history = store_history
        self._history_size = history_size
        self._fun_history = []
        self._maxcv_history = []
        self._x_history = []

    def __call__(self, x, penalty=0.0):
        """
        Evaluate the objective and nonlinear constraint functions.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the functions are evaluated.
        penalty : float, optional
            Penalty parameter used to select the point in the filter to forward
            to the callback function.

        Returns
        -------
        float
            Objective function value.
        `numpy.ndarray`, shape (m_nonlinear_ub,)
            Nonlinear inequality constraint function values.
        `numpy.ndarray`, shape (m_nonlinear_eq,)
            Nonlinear equality constraint function values.

        Raises
        ------
        `cobyqa.utils.CallbackSuccess`
            If the callback function raises a ``StopIteration``.
        """
        # Evaluate the objective and nonlinear constraint functions.
        x = np.asarray(x, dtype=float)
        x_full = self.build_x(x)
        fun_val = self._obj(x_full)
        cub_val, ceq_val = self._nonlinear(x_full)
        maxcv_val = self.maxcv(x, cub_val, ceq_val)
        if self._store_history:
            self._fun_history.append(fun_val)
            self._maxcv_history.append(maxcv_val)
            self._x_history.append(x)
            if len(self._fun_history) > self._history_size:
                self._fun_history.pop(0)
                self._maxcv_history.pop(0)
                self._x_history.pop(0)

        # Add the point to the filter if it is not dominated by any point.
        if np.isnan(fun_val) and np.isnan(maxcv_val):
            include_point = len(self._fun_filter) == 0
        elif np.isnan(fun_val):
            include_point = all(
                np.isnan(fun_filter)
                and maxcv_val < maxcv_filter
                or np.isnan(maxcv_filter)
                for fun_filter, maxcv_filter in zip(
                    self._fun_filter,
                    self._maxcv_filter,
                )
            )
        elif np.isnan(maxcv_val):
            include_point = all(
                np.isnan(maxcv_filter)
                and fun_val < fun_filter
                or np.isnan(fun_filter)
                for fun_filter, maxcv_filter in zip(
                    self._fun_filter,
                    self._maxcv_filter,
                )
            )
        else:
            include_point = all(
                fun_val < fun_filter or maxcv_val < maxcv_filter
                for fun_filter, maxcv_filter in zip(
                    self._fun_filter,
                    self._maxcv_filter,
                )
            )
        if include_point:
            self._fun_filter.append(fun_val)
            self._maxcv_filter.append(maxcv_val)
            self._x_filter.append(x)

            # Remove the points in the filter that are dominated by the new
            # point. We must iterate in reverse order to avoid problems when
            # removing elements from the list.
            for k in range(len(self._fun_filter) - 2, -1, -1):
                if np.isnan(fun_val):
                    remove_point = np.isnan(self._fun_filter[k])
                elif np.isnan(maxcv_val):
                    remove_point = np.isnan(self._maxcv_filter[k])
                else:
                    remove_point = (
                        np.isnan(self._fun_filter[k])
                        or np.isnan(self._maxcv_filter[k])
                        or fun_val <= self._fun_filter[k]
                        and maxcv_val <= self._maxcv_filter[k]
                    )
                if remove_point:
                    self._fun_filter.pop(k)
                    self._maxcv_filter.pop(k)
                    self._x_filter.pop(k)

            # Keep only the most recent points in the filter.
            if len(self._fun_filter) > self._filter_size:
                self._fun_filter.pop(0)
                self._maxcv_filter.pop(0)
                self._x_filter.pop(0)

        # Evaluate the callback function after updating the filter to ensure
        # that the current point can be returned by the method.
        if self._callback is not None:
            sig = signature(self._callback)
            try:
                x_best, fun_best, _ = self.best_eval(penalty)
                x_best = self.build_x(x_best)
                if set(sig.parameters) == {"intermediate_result"}:
                    intermediate_result = OptimizeResult(
                        x=x_best,
                        fun=fun_best,
                        # maxcv=maxcv_best,
                    )
                    self._callback(intermediate_result=intermediate_result)
                else:
                    self._callback(x_best)
            except StopIteration as exc:
                raise CallbackSuccess from exc

        # Apply the extreme barriers and return.
        if np.isnan(fun_val):
            fun_val = BARRIER
        cub_val[np.isnan(cub_val)] = BARRIER
        ceq_val[np.isnan(ceq_val)] = BARRIER
        fun_val = max(min(fun_val, BARRIER), -BARRIER)
        cub_val = np.maximum(np.minimum(cub_val, BARRIER), -BARRIER)
        ceq_val = np.maximum(np.minimum(ceq_val, BARRIER), -BARRIER)
        return fun_val, cub_val, ceq_val

    @property
    def n(self):
        """
        Number of variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self.x0.size

    @property
    def n_orig(self):
        """
        Number of variables in the original problem (with fixed variables).

        Returns
        -------
        int
            Number of variables in the original problem (with fixed variables).
        """
        return self._fixed_idx.size

    @property
    def x0(self):
        """
        Initial guess.

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Initial guess.
        """
        return self._x0

    @property
    def n_eval(self):
        """
        Number of function evaluations.

        Returns
        -------
        int
            Number of function evaluations.
        """
        return self._obj.n_eval

    @property
    def fun_name(self):
        """
        Name of the objective function.

        Returns
        -------
        str
            Name of the objective function.
        """
        return self._obj.name

    @property
    def bounds(self):
        """
        Bound constraints.

        Returns
        -------
        BoundConstraints
            Bound constraints.
        """
        return self._bounds

    @property
    def linear(self):
        """
        Linear constraints.

        Returns
        -------
        LinearConstraints
            Linear constraints.
        """
        return self._linear

    @property
    def m_bounds(self):
        """
        Number of bound constraints.

        Returns
        -------
        int
            Number of bound constraints.
        """
        return self.bounds.m

    @property
    def m_linear_ub(self):
        """
        Number of linear inequality constraints.

        Returns
        -------
        int
            Number of linear inequality constraints.
        """
        return self.linear.m_ub

    @property
    def m_linear_eq(self):
        """
        Number of linear equality constraints.

        Returns
        -------
        int
            Number of linear equality constraints.
        """
        return self.linear.m_eq

    @property
    def m_nonlinear_ub(self):
        """
        Number of nonlinear inequality constraints.

        Returns
        -------
        int
            Number of nonlinear inequality constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear inequality constraints is not known.
        """
        return self._nonlinear.m_ub

    @property
    def m_nonlinear_eq(self):
        """
        Number of nonlinear equality constraints.

        Returns
        -------
        int
            Number of nonlinear equality constraints.

        Raises
        ------
        ValueError
            If the number of nonlinear equality constraints is not known.
        """
        return self._nonlinear.m_eq

    @property
    def fun_history(self):
        """
        History of objective function evaluations.

        Returns
        -------
        `numpy.ndarray`, shape (n_eval,)
            History of objective function evaluations.
        """
        return np.array(self._fun_history, dtype=float)

    @property
    def maxcv_history(self):
        """
        History of maximum constraint violations.

        Returns
        -------
        `numpy.ndarray`, shape (n_eval,)
            History of maximum constraint violations.
        """
        return np.array(self._maxcv_history, dtype=float)

    @property
    def type(self):
        """
        Type of the problem.

        The problem can be either 'unconstrained', 'bound-constrained',
        'linearly constrained', or 'nonlinearly constrained'.

        Returns
        -------
        str
            Type of the problem.
        """
        try:
            if self.m_nonlinear_ub > 0 or self.m_nonlinear_eq > 0:
                return "nonlinearly constrained"
            elif self.m_linear_ub > 0 or self.m_linear_eq > 0:
                return "linearly constrained"
            elif self.m_bounds > 0:
                return "bound-constrained"
            else:
                return "unconstrained"
        except ValueError:
            # The number of nonlinear constraints is not known. It may be zero
            # if the user provided a nonlinear inequality and/or equality
            # constraint function that returns an empty array. However, as this
            # is not known before the first call to the function, we assume
            # that the problem is nonlinearly constrained.
            return "nonlinearly constrained"

    @property
    def is_feasibility(self):
        """
        Whether the problem is a feasibility problem.

        Returns
        -------
        bool
            Whether the problem is a feasibility problem.
        """
        return self.fun_name == ""

    def build_x(self, x):
        """
        Build the full vector of variables from the reduced vector.

        Parameters
        ----------
        x : array_like, shape (n,)
            Reduced vector of variables.

        Returns
        -------
        `numpy.ndarray`, shape (n_orig,)
            Full vector of variables.
        """
        x_full = np.empty(self.n_orig)
        x_full[self._fixed_idx] = self._fixed_val
        x_full[~self._fixed_idx] = (x * self._scaling_factor
                                    + self._scaling_shift)
        return self._orig_bounds.project(x_full)

    def maxcv(self, x, cub_val=None, ceq_val=None):
        """
        Evaluate the maximum constraint violation.

        Parameters
        ----------
        x : array_like, shape (n,)
            Point at which the maximum constraint violation is evaluated.
        cub_val : array_like, shape (m_nonlinear_ub,), optional
            Values of the nonlinear inequality constraints. If not provided,
            the nonlinear inequality constraints are evaluated at `x`.
        ceq_val : array_like, shape (m_nonlinear_eq,), optional
            Values of the nonlinear equality constraints. If not provided,
            the nonlinear equality constraints are evaluated at `x`.

        Returns
        -------
        float
            Maximum constraint violation at `x`.
        """
        violation = self.violation(x, cub_val=cub_val, ceq_val=ceq_val)
        if np.count_nonzero(violation):
            return np.max(violation, initial=0.0)
        else:
            return 0.0

    def violation(self, x, cub_val=None, ceq_val=None):
        violation = []
        if not self.bounds.is_feasible:
            b = self.bounds.violation(x)
            violation.append(b)

        if len(self.linear.pcs):
            lc = self.linear.violation(x)
            violation.append(lc)
        if len(self._nonlinear.pcs):
            nlc = self._nonlinear.violation(x, cub_val, ceq_val)
            violation.append(nlc)

        if len(violation):
            return np.concatenate(violation)

    def best_eval(self, penalty):
        """
        Return the best point in the filter and the corresponding objective and
        nonlinear constraint function evaluations.

        Parameters
        ----------
        penalty : float
            Penalty parameter

        Returns
        -------
        `numpy.ndarray`, shape (n,)
            Best point.
        float
            Corresponding objective function value.
        float
            Corresponding maximum constraint violation.
        """
        # If the filter is empty, i.e., if no function evaluation has been
        # performed, we evaluate the objective and nonlinear constraint
        # functions at the initial guess.
        if len(self._fun_filter) == 0:
            self(self.x0)

        # Find the best point in the filter.
        fun_filter = np.array(self._fun_filter)
        maxcv_filter = np.array(self._maxcv_filter)
        x_filter = np.array(self._x_filter)
        finite_idx = np.isfinite(maxcv_filter)
        if np.any(finite_idx):
            # At least one point has a finite maximum constraint violation.
            feasible_idx = maxcv_filter <= self._feasibility_tol
            if np.any(feasible_idx) and not np.all(
                np.isnan(fun_filter[feasible_idx])
            ):
                # At least one point is feasible and has a well-defined
                # objective function value. We select the point with the least
                # objective function value. If there is a tie, we select the
                # point with the least maximum constraint violation. If there
                # is still a tie, we select the most recent point.
                fun_min_idx = feasible_idx & (
                    fun_filter <= np.nanmin(fun_filter[feasible_idx])
                )
                if np.count_nonzero(fun_min_idx) > 1:
                    fun_min_idx &= maxcv_filter <= np.min(
                        maxcv_filter[fun_min_idx]
                    )
                i = np.flatnonzero(fun_min_idx)[-1]
            elif np.any(feasible_idx):
                # At least one point is feasible but no feasible point has a
                # well-defined objective function value. We select the most
                # recent feasible point.
                i = np.flatnonzero(feasible_idx)[-1]
            else:
                # No point is feasible. We first compute the merit function
                # value for each point.
                merit_filter = np.full_like(fun_filter, np.nan)
                merit_filter[finite_idx] = (
                    fun_filter[finite_idx] + penalty * maxcv_filter[finite_idx]
                )
                if np.all(np.isnan(merit_filter)):
                    # No point has a well-defined merit function value. In
                    # other words, among the points with a well-defined maximum
                    # constraint violation, none has a well-defined objective
                    # function value. We select the point with the least
                    # maximum constraint violation. If there is a tie, we
                    # select the most recent point.
                    min_maxcv_idx = maxcv_filter <= np.nanmin(maxcv_filter)
                    i = np.flatnonzero(min_maxcv_idx)[-1]
                else:
                    # At least one point has a well-defined merit function
                    # value. We select the point with the least merit function
                    # value. If there is a tie, we select the point with the
                    # least maximum constraint violation. If there is still a
                    # tie, we select the point with the least objective
                    # function value. If there is still a tie, we select the
                    # most recent point.
                    merit_min_idx = merit_filter <= np.nanmin(merit_filter)
                    if np.count_nonzero(merit_min_idx) > 1:
                        merit_min_idx &= maxcv_filter <= np.min(
                            maxcv_filter[merit_min_idx]
                        )

                    if np.count_nonzero(merit_min_idx) > 1:
                        merit_min_idx &= fun_filter <= np.min(
                            fun_filter[merit_min_idx]
                        )
                    i = np.flatnonzero(merit_min_idx)[-1]
        elif not np.all(np.isnan(fun_filter)):
            # No maximum constraint violation is well-defined but at least one
            # point has a well-defined objective function value. We select the
            # point with the least objective function value. If there is a tie,
            # we select the most recent point.
            fun_min_idx = fun_filter <= np.nanmin(fun_filter)
            i = np.flatnonzero(fun_min_idx)[-1]
        else:
            # No point has a well-defined maximum constraint violation or
            # objective function value. We select the most recent point.
            i = len(fun_filter) - 1
        return (
            self.bounds.project(x_filter[i, :]),
            fun_filter[i],
            maxcv_filter[i],
        )
<|endoftext|>import sys
from enum import Enum

import numpy as np


# Exit status.
class ExitStatus(Enum):
    """
    Exit statuses.
    """

    RADIUS_SUCCESS = 0
    TARGET_SUCCESS = 1
    FIXED_SUCCESS = 2
    CALLBACK_SUCCESS = 3
    FEASIBLE_SUCCESS = 4
    MAX_EVAL_WARNING = 5
    MAX_ITER_WARNING = 6
    INFEASIBLE_ERROR = -1
    LINALG_ERROR = -2


class Options(str, Enum):
    """
    Options.
    """

    DEBUG = "debug"
    FEASIBILITY_TOL = "feasibility_tol"
    FILTER_SIZE = "filter_size"
    HISTORY_SIZE = "history_size"
    MAX_EVAL = "maxfev"
    MAX_ITER = "maxiter"
    NPT = "nb_points"
    RHOBEG = "radius_init"
    RHOEND = "radius_final"
    SCALE = "scale"
    STORE_HISTORY = "store_history"
    TARGET = "target"
    VERBOSE = "disp"


class Constants(str, Enum):
    """
    Constants.
    """

    DECREASE_RADIUS_FACTOR = "decrease_radius_factor"
    INCREASE_RADIUS_FACTOR = "increase_radius_factor"
    INCREASE_RADIUS_THRESHOLD = "increase_radius_threshold"
    DECREASE_RADIUS_THRESHOLD = "decrease_radius_threshold"
    DECREASE_RESOLUTION_FACTOR = "decrease_resolution_factor"
    LARGE_RESOLUTION_THRESHOLD = "large_resolution_threshold"
    MODERATE_RESOLUTION_THRESHOLD = "moderate_resolution_threshold"
    LOW_RATIO = "low_ratio"
    HIGH_RATIO = "high_ratio"
    VERY_LOW_RATIO = "very_low_ratio"
    PENALTY_INCREASE_THRESHOLD = "penalty_increase_threshold"
    PENALTY_INCREASE_FACTOR = "penalty_increase_factor"
    SHORT_STEP_THRESHOLD = "short_step_threshold"
    LOW_RADIUS_FACTOR = "low_radius_factor"
    BYRD_OMOJOKUN_FACTOR = "byrd_omojokun_factor"
    THRESHOLD_RATIO_CONSTRAINTS = "threshold_ratio_constraints"
    LARGE_SHIFT_FACTOR = "large_shift_factor"
    LARGE_GRADIENT_FACTOR = "large_gradient_factor"
    RESOLUTION_FACTOR = "resolution_factor"
    IMPROVE_TCG = "improve_tcg"


# Default options.
DEFAULT_OPTIONS = {
    Options.DEBUG.value: False,
    Options.FEASIBILITY_TOL.value: np.sqrt(np.finfo(float).eps),
    Options.FILTER_SIZE.value: sys.maxsize,
    Options.HISTORY_SIZE.value: sys.maxsize,
    Options.MAX_EVAL.value: lambda n: 500 * n,
    Options.MAX_ITER.value: lambda n: 1000 * n,
    Options.NPT.value: lambda n: 2 * n + 1,
    Options.RHOBEG.value: 1.0,
    Options.RHOEND.value: 1e-6,
    Options.SCALE.value: False,
    Options.STORE_HISTORY.value: False,
    Options.TARGET.value: -np.inf,
    Options.VERBOSE.value: False,
}

# Default constants.
DEFAULT_CONSTANTS = {
    Constants.DECREASE_RADIUS_FACTOR.value: 0.5,
    Constants.INCREASE_RADIUS_FACTOR.value: np.sqrt(2.0),
    Constants.INCREASE_RADIUS_THRESHOLD.value: 2.0,
    Constants.DECREASE_RADIUS_THRESHOLD.value: 1.4,
    Constants.DECREASE_RESOLUTION_FACTOR.value: 0.1,
    Constants.LARGE_RESOLUTION_THRESHOLD.value: 250.0,
    Constants.MODERATE_RESOLUTION_THRESHOLD.value: 16.0,
    Constants.LOW_RATIO.value: 0.1,
    Constants.HIGH_RATIO.value: 0.7,
    Constants.VERY_LOW_RATIO.value: 0.01,
    Constants.PENALTY_INCREASE_THRESHOLD.value: 1.5,
    Constants.PENALTY_INCREASE_FACTOR.value: 2.0,
    Constants.SHORT_STEP_THRESHOLD.value: 0.5,
    Constants.LOW_RADIUS_FACTOR.value: 0.1,
    Constants.BYRD_OMOJOKUN_FACTOR.value: 0.8,
    Constants.THRESHOLD_RATIO_CONSTRAINTS.value: 2.0,
    Constants.LARGE_SHIFT_FACTOR.value: 10.0,
    Constants.LARGE_GRADIENT_FACTOR.value: 10.0,
    Constants.RESOLUTION_FACTOR.value: 2.0,
    Constants.IMPROVE_TCG.value: True,
}

# Printing options.
PRINT_OPTIONS = {
    "threshold": 6,
    "edgeitems": 2,
    "linewidth": sys.maxsize,
    "formatter": {
        "float_kind": lambda x: np.format_float_scientific(
            x,
            precision=3,
            unique=False,
            pad_left=2,
        )
    },
}

# Constants.
BARRIER = 2.0 ** min(
    100,
    np.finfo(float).maxexp // 2,
    -np.finfo(float).minexp // 2,
)
<|endoftext|>import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def _get_sys_info():
    """
    Get useful system information.

    Returns
    -------
    dict
        Useful system information.
    """
    return {
        "python": sys.version.replace(os.linesep, " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }


def _get_deps_info():
    """
    Get the versions of the dependencies.

    Returns
    -------
    dict
        Versions of the dependencies.
    """
    deps = ["cobyqa", "numpy", "scipy", "setuptools", "pip"]
    deps_info = {}
    for module in deps:
        try:
            deps_info[module] = version(module)
        except PackageNotFoundError:
            deps_info[module] = None
    return deps_info


def show_versions():
    """
    Display useful system and dependencies information.

    When reporting issues, please include this information.
    """
    print("System settings")
    print("---------------")
    sys_info = _get_sys_info()
    print(
        "\n".join(
            f"{k:>{max(map(len, sys_info.keys())) + 1}}: {v}"
            for k, v in sys_info.items()
        )
    )

    print()
    print("Python dependencies")
    print("-------------------")
    deps_info = _get_deps_info()
    print(
        "\n".join(
            f"{k:>{max(map(len, deps_info.keys())) + 1}}: {v}"
            for k, v in deps_info.items()
        )
    )
<|endoftext|>from .exceptions import (
    MaxEvalError,
    TargetSuccess,
    CallbackSuccess,
    FeasibleSuccess,
)
from .math import get_arrays_tol, exact_1d_array
from .versions import show_versions

__all__ = [
    "MaxEvalError",
    "TargetSuccess",
    "CallbackSuccess",
    "FeasibleSuccess",
    "get_arrays_tol",
    "exact_1d_array",
    "show_versions",
]
<|endoftext|>import numpy as np


EPS = np.finfo(float).eps


def get_arrays_tol(*arrays):
    """
    Get a relative tolerance for a set of arrays.

    Parameters
    ----------
    *arrays: tuple
        Set of `numpy.ndarray` to get the tolerance for.

    Returns
    -------
    float
        Relative tolerance for the set of arrays.

    Raises
    ------
    ValueError
        If no array is provided.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    size = max(array.size for array in arrays)
    weight = max(
        np.max(np.abs(array[np.isfinite(array)]), initial=1.0)
        for array in arrays
    )
    return 10.0 * EPS * max(size, 1.0) * weight


def exact_1d_array(x, message):
    """
    Preprocess a 1-dimensional array.

    Parameters
    ----------
    x : array_like
        Array to be preprocessed.
    message : str
        Error message if `x` cannot be interpreter as a 1-dimensional array.

    Returns
    -------
    `numpy.ndarray`
        Preprocessed array.
    """
    x = np.atleast_1d(np.squeeze(x)).astype(float)
    if x.ndim != 1:
        raise ValueError(message)
    return x


def exact_2d_array(x, message):
    """
    Preprocess a 2-dimensional array.

    Parameters
    ----------
    x : array_like
        Array to be preprocessed.
    message : str
        Error message if `x` cannot be interpreter as a 2-dimensional array.

    Returns
    -------
    `numpy.ndarray`
        Preprocessed array.
    """
    x = np.atleast_2d(x).astype(float)
    if x.ndim != 2:
        raise ValueError(message)
    return x
<|endoftext|>class MaxEvalError(Exception):
    """
    Exception raised when the maximum number of evaluations is reached.
    """


class TargetSuccess(Exception):
    """
    Exception raised when the target value is reached.
    """


class CallbackSuccess(StopIteration):
    """
    Exception raised when the callback function raises a ``StopIteration``.
    """


class FeasibleSuccess(Exception):
    """
    Exception raised when a feasible point of a feasible problem is found.
    """
<|endoftext|>import inspect

import numpy as np
from scipy.linalg import qr

from ..utils import get_arrays_tol


TINY = np.finfo(float).tiny
EPS = np.finfo(float).eps


def tangential_byrd_omojokun(grad, hess_prod, xl, xu, delta, debug, **kwargs):
    r"""
    Minimize approximately a quadratic function subject to bound constraints in
    a trust region.

    This function solves approximately

    .. math::

        \min_{s \in \mathbb{R}^n} \quad g^{\mathsf{T}} s + \frac{1}{2}
        s^{\mathsf{T}} H s \quad \text{s.t.} \quad
        \left\{ \begin{array}{l}
            l \le s \le u\\
            \lVert s \rVert \le \Delta,
        \end{array} \right.

    using an active-set variation of the truncated conjugate gradient method.

    Parameters
    ----------
    grad : `numpy.ndarray`, shape (n,)
        Gradient :math:`g` as shown above.
    hess_prod : callable
        Product of the Hessian matrix :math:`H` with any vector.

            ``hess_prod(s) -> `numpy.ndarray`, shape (n,)``

        returns the product :math:`H s`.
    xl : `numpy.ndarray`, shape (n,)
        Lower bounds :math:`l` as shown above.
    xu : `numpy.ndarray`, shape (n,)
        Upper bounds :math:`u` as shown above.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.
    debug : bool
        Whether to make debugging tests during the execution.

    Returns
    -------
    `numpy.ndarray`, shape (n,)
        Approximate solution :math:`s`.

    Other Parameters
    ----------------
    improve_tcg : bool, optional
        If True, a solution generated by the truncated conjugate gradient
        method that is on the boundary of the trust region is improved by
        moving around the trust-region boundary on the two-dimensional space
        spanned by the solution and the gradient of the quadratic function at
        the solution (default is True).

    Notes
    -----
    This function implements Algorithm 6.2 of [1]_. It is assumed that the
    origin is feasible with respect to the bound constraints and that `delta`
    is finite and positive.

    References
    ----------
    .. [1] T. M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.
    """
    if debug:
        assert isinstance(grad, np.ndarray) and grad.ndim == 1
        assert inspect.signature(hess_prod).bind(grad)
        assert isinstance(xl, np.ndarray) and xl.shape == grad.shape
        assert isinstance(xu, np.ndarray) and xu.shape == grad.shape
        assert isinstance(delta, float)
        assert isinstance(debug, bool)
        tol = get_arrays_tol(xl, xu)
        assert np.all(xl <= tol)
        assert np.all(xu >= -tol)
        assert np.isfinite(delta) and delta > 0.0
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    # Copy the arrays that may be modified by the code below.
    n = grad.size
    grad = np.copy(grad)
    grad_orig = np.copy(grad)

    # Calculate the initial active set.
    free_bd = ((xl < 0.0) | (grad < 0.0)) & ((xu > 0.0) | (grad > 0.0))

    # Set the initial iterate and the initial search direction.
    step = np.zeros_like(grad)
    sd = np.zeros_like(step)
    sd[free_bd] = -grad[free_bd]

    k = 0
    reduct = 0.0
    boundary_reached = False
    while k < np.count_nonzero(free_bd):
        # Stop the computations if sd is not a descent direction.
        grad_sd = grad @ sd
        if grad_sd >= -10.0 * EPS * n * max(1.0, np.linalg.norm(grad)):
            break

        # Set alpha_tr to the step size for the trust-region constraint.
        try:
            alpha_tr = _alpha_tr(step, sd, delta)
        except ZeroDivisionError:
            break

        # Stop the computations if a step along sd is expected to give a
        # relatively small reduction in the objective function.
        if -alpha_tr * grad_sd <= 1e-8 * reduct:
            break

        # Set alpha_quad to the step size for the minimization problem.
        hess_sd = hess_prod(sd)
        curv_sd = sd @ hess_sd
        if curv_sd > TINY * abs(grad_sd):
            alpha_quad = max(-grad_sd / curv_sd, 0.0)
        else:
            alpha_quad = np.inf

        # Stop the computations if the reduction in the objective function
        # provided by an unconstrained step is small.
        alpha = min(alpha_tr, alpha_quad)
        if -alpha * (grad_sd + 0.5 * alpha * curv_sd) <= 1e-8 * reduct:
            break

        # Set alpha_bd to the step size for the bound constraints.
        i_xl = (xl > -np.inf) & (sd < -TINY * np.abs(xl - step))
        i_xu = (xu < np.inf) & (sd > TINY * np.abs(xu - step))
        all_alpha_xl = np.full_like(step, np.inf)
        all_alpha_xu = np.full_like(step, np.inf)
        all_alpha_xl[i_xl] = np.maximum(
            (xl[i_xl] - step[i_xl]) / sd[i_xl],
            0.0,
        )
        all_alpha_xu[i_xu] = np.maximum(
            (xu[i_xu] - step[i_xu]) / sd[i_xu],
            0.0,
        )
        alpha_xl = np.min(all_alpha_xl)
        alpha_xu = np.min(all_alpha_xu)
        alpha_bd = min(alpha_xl, alpha_xu)

        # Update the iterate.
        alpha = min(alpha, alpha_bd)
        if alpha > 0.0:
            step[free_bd] = np.clip(
                step[free_bd] + alpha * sd[free_bd],
                xl[free_bd],
                xu[free_bd],
            )
            grad += alpha * hess_sd
            reduct -= alpha * (grad_sd + 0.5 * alpha * curv_sd)

        if alpha < min(alpha_tr, alpha_bd):
            # The current iteration is a conjugate gradient iteration. Update
            # the search direction so that it is conjugate (with respect to H)
            # to all the previous search directions.
            beta = (grad[free_bd] @ hess_sd[free_bd]) / curv_sd
            sd[free_bd] = beta * sd[free_bd] - grad[free_bd]
            sd[~free_bd] = 0.0
            k += 1
        elif alpha < alpha_tr:
            # The iterate is restricted by a bound constraint. Add this bound
            # constraint to the active set, and restart the calculations.
            if alpha_xl <= alpha:
                i_new = np.argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
            else:
                i_new = np.argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
            free_bd[i_new] = False
            sd[free_bd] = -grad[free_bd]
            sd[~free_bd] = 0.0
            k = 0
        else:
            # The current iterate is on the trust-region boundary. Add all the
            # active bounds to the working set to prepare for the improvement
            # of the solution, and stop the iterations.
            if alpha_xl <= alpha:
                i_new = _argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_bd[i_new] = False
            if alpha_xu <= alpha:
                i_new = _argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_bd[i_new] = False
            boundary_reached = True
            break

    # Attempt to improve the solution on the trust-region boundary.
    if kwargs.get("improve_tcg", True) and boundary_reached:
        step_base = np.copy(step)
        step_comparator = grad_orig @ step_base + 0.5 * step_base @ hess_prod(
            step_base
        )

        while np.count_nonzero(free_bd) > 0:
            # Check whether a substantial reduction in the objective function
            # is possible, and set the search direction.
            step_sq = step[free_bd] @ step[free_bd]
            grad_sq = grad[free_bd] @ grad[free_bd]
            grad_step = grad[free_bd] @ step[free_bd]
            grad_sd = -np.sqrt(max(step_sq * grad_sq - grad_step**2.0, 0.0))
            sd[free_bd] = grad_step * step[free_bd] - step_sq * grad[free_bd]
            sd[~free_bd] = 0.0
            if grad_sd >= -1e-8 * reduct or np.any(
                grad_sd >= -TINY * np.abs(sd[free_bd])
            ):
                break
            sd[free_bd] /= -grad_sd

            # Calculate an upper bound for the tangent of half the angle theta
            # of this alternative iteration. The step will be updated as:
            # step = cos(theta) * step + sin(theta) * sd.
            temp_xl = np.zeros(n)
            temp_xu = np.zeros(n)
            temp_xl[free_bd] = (
                step[free_bd] ** 2.0 + sd[free_bd] ** 2.0 - xl[free_bd] ** 2.0
            )
            temp_xu[free_bd] = (
                step[free_bd] ** 2.0 + sd[free_bd] ** 2.0 - xu[free_bd] ** 2.0
            )
            temp_xl[temp_xl > 0.0] = (
                np.sqrt(temp_xl[temp_xl > 0.0]) - sd[temp_xl > 0.0]
            )
            temp_xu[temp_xu > 0.0] = (
                np.sqrt(temp_xu[temp_xu > 0.0]) + sd[temp_xu > 0.0]
            )
            dist_xl = np.maximum(step - xl, 0.0)
            dist_xu = np.maximum(xu - step, 0.0)
            i_xl = temp_xl > TINY * dist_xl
            i_xu = temp_xu > TINY * dist_xu
            all_t_xl = np.ones(n)
            all_t_xu = np.ones(n)
            all_t_xl[i_xl] = np.minimum(
                all_t_xl[i_xl],
                dist_xl[i_xl] / temp_xl[i_xl],
            )
            all_t_xu[i_xu] = np.minimum(
                all_t_xu[i_xu],
                dist_xu[i_xu] / temp_xu[i_xu],
            )
            t_xl = np.min(all_t_xl)
            t_xu = np.min(all_t_xu)
            t_bd = min(t_xl, t_xu)

            # Calculate some curvature information.
            hess_step = hess_prod(step)
            hess_sd = hess_prod(sd)
            curv_step = step @ hess_step
            curv_sd = sd @ hess_sd
            curv_step_sd = step @ hess_sd

            # For a range of equally spaced values of tan(0.5 * theta),
            # calculate the reduction in the objective function that would be
            # obtained by accepting the corresponding angle.
            n_samples = 20
            n_samples = int((n_samples - 3) * t_bd + 3)
            t_samples = np.linspace(t_bd / n_samples, t_bd, n_samples)
            sin_values = 2.0 * t_samples / (1.0 + t_samples**2.0)
            all_reduct = sin_values * (
                grad_step * t_samples
                - grad_sd
                - t_samples * curv_step
                + sin_values
                * (t_samples * curv_step_sd - 0.5 * (curv_sd - curv_step))
            )
            if np.all(all_reduct <= 0.0):
                # No reduction in the objective function is obtained.
                break

            # Accept the angle that provides the largest reduction in the
            # objective function, and update the iterate.
            i_max = np.argmax(all_reduct)
            cos_value = (1.0 - t_samples[i_max] ** 2.0) / (
                1.0 + t_samples[i_max] ** 2.0
            )
            step[free_bd] = (
                cos_value * step[free_bd] + sin_values[i_max] * sd[free_bd]
            )
            grad += (cos_value - 1.0) * hess_step + sin_values[i_max] * hess_sd
            reduct += all_reduct[i_max]

            # If the above angle is restricted by bound constraints, add them
            # to the working set, and restart the alternative iteration.
            # Otherwise, the calculations are terminated.
            if t_bd < 1.0 and i_max == n_samples - 1:
                if t_xl <= t_bd:
                    i_new = _argmin(all_t_xl)
                    step[i_new] = xl[i_new]
                    free_bd[i_new] = False
                if t_xu <= t_bd:
                    i_new = _argmin(all_t_xu)
                    step[i_new] = xu[i_new]
                    free_bd[i_new] = False
            else:
                break

        # Ensure that the alternative iteration improves the objective
        # function.
        if grad_orig @ step + 0.5 * step @ hess_prod(step) > step_comparator:
            step = step_base

    if debug:
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.linalg.norm(step) < 1.1 * delta
    return step


def constrained_tangential_byrd_omojokun(
    grad,
    hess_prod,
    xl,
    xu,
    aub,
    bub,
    aeq,
    delta,
    debug,
    **kwargs,
):
    r"""
    Minimize approximately a quadratic function subject to bound and linear
    constraints in a trust region.

    This function solves approximately

    .. math::

        \min_{s \in \mathbb{R}^n} \quad g^{\mathsf{T}} s + \frac{1}{2}
        s^{\mathsf{T}} H s \quad \text{s.t.} \quad
        \left\{ \begin{array}{l}
            l \le s \le u,\\
            A_{\scriptscriptstyle I} s \le b_{\scriptscriptstyle I},\\
            A_{\scriptscriptstyle E} s = 0,\\
            \lVert s \rVert \le \Delta,
        \end{array} \right.

    using an active-set variation of the truncated conjugate gradient method.

    Parameters
    ----------
    grad : `numpy.ndarray`, shape (n,)
        Gradient :math:`g` as shown above.
    hess_prod : callable
        Product of the Hessian matrix :math:`H` with any vector.

            ``hess_prod(s) -> `numpy.ndarray`, shape (n,)``

        returns the product :math:`H s`.
    xl : `numpy.ndarray`, shape (n,)
        Lower bounds :math:`l` as shown above.
    xu : `numpy.ndarray`, shape (n,)
        Upper bounds :math:`u` as shown above.
    aub : `numpy.ndarray`, shape (m_linear_ub, n)
        Coefficient matrix :math:`A_{\scriptscriptstyle I}` as shown above.
    bub : `numpy.ndarray`, shape (m_linear_ub,)
        Right-hand side :math:`b_{\scriptscriptstyle I}` as shown above.
    aeq : `numpy.ndarray`, shape (m_linear_eq, n)
        Coefficient matrix :math:`A_{\scriptscriptstyle E}` as shown above.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.
    debug : bool
        Whether to make debugging tests during the execution.

    Returns
    -------
    `numpy.ndarray`, shape (n,)
        Approximate solution :math:`s`.

    Other Parameters
    ----------------
    improve_tcg : bool, optional
        If True, a solution generated by the truncated conjugate gradient
        method that is on the boundary of the trust region is improved by
        moving around the trust-region boundary on the two-dimensional space
        spanned by the solution and the gradient of the quadratic function at
        the solution (default is True).

    Notes
    -----
    This function implements Algorithm 6.3 of [1]_. It is assumed that the
    origin is feasible with respect to the bound and linear constraints, and
    that `delta` is finite and positive.

    References
    ----------
    .. [1] T. M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.
    """
    if debug:
        assert isinstance(grad, np.ndarray) and grad.ndim == 1
        assert inspect.signature(hess_prod).bind(grad)
        assert isinstance(xl, np.ndarray) and xl.shape == grad.shape
        assert isinstance(xu, np.ndarray) and xu.shape == grad.shape
        assert (
            isinstance(aub, np.ndarray)
            and aub.ndim == 2
            and aub.shape[1] == grad.size
        )
        assert (
            isinstance(bub, np.ndarray)
            and bub.ndim == 1
            and bub.size == aub.shape[0]
        )
        assert (
            isinstance(aeq, np.ndarray)
            and aeq.ndim == 2
            and aeq.shape[1] == grad.size
        )
        assert isinstance(delta, float)
        assert isinstance(debug, bool)
        tol = get_arrays_tol(xl, xu)
        assert np.all(xl <= tol)
        assert np.all(xu >= -tol)
        assert np.all(bub >= -tol)
        assert np.isfinite(delta) and delta > 0.0
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)
    bub = np.maximum(bub, 0.0)

    # Copy the arrays that may be modified by the code below.
    n = grad.size
    grad = np.copy(grad)
    grad_orig = np.copy(grad)

    # Calculate the initial active set.
    free_xl = (xl < 0.0) | (grad < 0.0)
    free_xu = (xu > 0.0) | (grad > 0.0)
    free_ub = (bub > 0.0) | (aub @ grad > 0.0)
    n_act, q = qr_tangential_byrd_omojokun(aub, aeq, free_xl, free_xu, free_ub)

    # Set the initial iterate and the initial search direction.
    step = np.zeros_like(grad)
    sd = -q[:, n_act:] @ (q[:, n_act:].T @ grad)
    resid = np.copy(bub)

    k = 0
    reduct = 0.0
    boundary_reached = False
    while k < n - n_act:
        # Stop the computations if sd is not a descent direction.
        grad_sd = grad @ sd
        if grad_sd >= -10.0 * EPS * n * max(1.0, np.linalg.norm(grad)):
            break

        # Set alpha_tr to the step size for the trust-region constraint.
        try:
            alpha_tr = _alpha_tr(step, sd, delta)
        except ZeroDivisionError:
            break

        # Stop the computations if a step along sd is expected to give a
        # relatively small reduction in the objective function.
        if -alpha_tr * grad_sd <= 1e-8 * reduct:
            break

        # Set alpha_quad to the step size for the minimization problem.
        hess_sd = hess_prod(sd)
        curv_sd = sd @ hess_sd
        if curv_sd > TINY * abs(grad_sd):
            alpha_quad = max(-grad_sd / curv_sd, 0.0)
        else:
            alpha_quad = np.inf

        # Stop the computations if the reduction in the objective function
        # provided by an unconstrained step is small.
        alpha = min(alpha_tr, alpha_quad)
        if -alpha * (grad_sd + 0.5 * alpha * curv_sd) <= 1e-8 * reduct:
            break

        # Set alpha_bd to the step size for the bound constraints.
        i_xl = free_xl & (xl > -np.inf) & (sd < -TINY * np.abs(xl - step))
        i_xu = free_xu & (xu < np.inf) & (sd > TINY * np.abs(xu - step))
        all_alpha_xl = np.full_like(step, np.inf)
        all_alpha_xu = np.full_like(step, np.inf)
        all_alpha_xl[i_xl] = np.maximum(
            (xl[i_xl] - step[i_xl]) / sd[i_xl],
            0.0,
        )
        all_alpha_xu[i_xu] = np.maximum(
            (xu[i_xu] - step[i_xu]) / sd[i_xu],
            0.0,
        )
        alpha_xl = np.min(all_alpha_xl)
        alpha_xu = np.min(all_alpha_xu)
        alpha_bd = min(alpha_xl, alpha_xu)

        # Set alpha_ub to the step size for the linear constraints.
        aub_sd = aub @ sd
        i_ub = free_ub & (aub_sd > TINY * np.abs(resid))
        all_alpha_ub = np.full_like(bub, np.inf)
        all_alpha_ub[i_ub] = resid[i_ub] / aub_sd[i_ub]
        alpha_ub = np.min(all_alpha_ub, initial=np.inf)

        # Update the iterate.
        alpha = min(alpha, alpha_bd, alpha_ub)
        if alpha > 0.0:
            step = np.clip(step + alpha * sd, xl, xu)
            grad += alpha * hess_sd
            resid = np.maximum(0.0, resid - alpha * aub_sd)
            reduct -= alpha * (grad_sd + 0.5 * alpha * curv_sd)

        if alpha < min(alpha_tr, alpha_bd, alpha_ub):
            # The current iteration is a conjugate gradient iteration. Update
            # the search direction so that it is conjugate (with respect to H)
            # to all the previous search directions.
            grad_proj = q[:, n_act:] @ (q[:, n_act:].T @ grad)
            beta = (grad_proj @ hess_sd) / curv_sd
            sd = beta * sd - grad_proj
            k += 1
        elif alpha < alpha_tr:
            # The iterate is restricted by a bound/linear constraint. Add this
            # constraint to the active set, and restart the calculations.
            if alpha_xl <= alpha:
                i_new = np.argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_xl[i_new] = False
            elif alpha_xu <= alpha:
                i_new = np.argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_xu[i_new] = False
            else:
                i_new = np.argmin(all_alpha_ub)
                free_ub[i_new] = False
            n_act, q = qr_tangential_byrd_omojokun(
                aub,
                aeq,
                free_xl,
                free_xu,
                free_ub,
            )
            sd = -q[:, n_act:] @ (q[:, n_act:].T @ grad)
            k = 0
        else:
            # The current iterate is on the trust-region boundary. Add all the
            # active bound/linear constraints to the working set to prepare for
            # the improvement of the solution, and stop the iterations.
            if alpha_xl <= alpha:
                i_new = _argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_xl[i_new] = False
            if alpha_xu <= alpha:
                i_new = _argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_xu[i_new] = False
            if alpha_ub <= alpha:
                i_new = _argmin(all_alpha_ub)
                free_ub[i_new] = False
            n_act, q = qr_tangential_byrd_omojokun(
                aub,
                aeq,
                free_xl,
                free_xu,
                free_ub,
            )
            boundary_reached = True
            break

    # Attempt to improve the solution on the trust-region boundary.
    if kwargs.get("improve_tcg", True) and boundary_reached and n_act < n:
        step_base = np.copy(step)
        while n_act < n:
            # Check whether a substantial reduction in the objective function
            # is possible, and set the search direction.
            step_proj = q[:, n_act:] @ (q[:, n_act:].T @ step)
            grad_proj = q[:, n_act:] @ (q[:, n_act:].T @ grad)
            step_sq = step_proj @ step_proj
            grad_sq = grad_proj @ grad_proj
            grad_step = grad_proj @ step_proj
            grad_sd = -np.sqrt(max(step_sq * grad_sq - grad_step**2.0, 0.0))
            sd = q[:, n_act:] @ (
                q[:, n_act:].T @ (grad_step * step - step_sq * grad)
            )
            if grad_sd >= -1e-8 * reduct or np.any(
                grad_sd >= -TINY * np.abs(sd)
            ):
                break
            sd /= -grad_sd

            # Calculate an upper bound for the tangent of half the angle theta
            # of this alternative iteration for the bound constraints. The step
            # will be updated as:
            # step += (cos(theta) - 1) * step_proj + sin(theta) * sd.
            temp_xl = np.zeros(n)
            temp_xu = np.zeros(n)
            dist_xl = np.maximum(step - xl, 0.0)
            dist_xu = np.maximum(xu - step, 0.0)
            temp_xl[free_xl] = sd[free_xl] ** 2.0 - dist_xl[free_xl] * (
                dist_xl[free_xl] - 2.0 * step_proj[free_xl]
            )
            temp_xu[free_xu] = sd[free_xu] ** 2.0 - dist_xu[free_xu] * (
                dist_xu[free_xu] + 2.0 * step_proj[free_xu]
            )
            temp_xl[temp_xl > 0.0] = (
                np.sqrt(temp_xl[temp_xl > 0.0]) - sd[temp_xl > 0.0]
            )
            temp_xu[temp_xu > 0.0] = (
                np.sqrt(temp_xu[temp_xu > 0.0]) + sd[temp_xu > 0.0]
            )
            i_xl = temp_xl > TINY * dist_xl
            i_xu = temp_xu > TINY * dist_xu
            all_t_xl = np.ones(n)
            all_t_xu = np.ones(n)
            all_t_xl[i_xl] = np.minimum(
                all_t_xl[i_xl],
                dist_xl[i_xl] / temp_xl[i_xl],
            )
            all_t_xu[i_xu] = np.minimum(
                all_t_xu[i_xu],
                dist_xu[i_xu] / temp_xu[i_xu],
            )
            t_xl = np.min(all_t_xl)
            t_xu = np.min(all_t_xu)
            t_bd = min(t_xl, t_xu)

            # Calculate an upper bound for the tangent of half the angle theta
            # of this alternative iteration for the linear constraints.
            temp_ub = np.zeros_like(resid)
            aub_step = aub @ step_proj
            aub_sd = aub @ sd
            temp_ub[free_ub] = aub_sd[free_ub] ** 2.0 - resid[free_ub] * (
                resid[free_ub] + 2.0 * aub_step[free_ub]
            )
            temp_ub[temp_ub > 0.0] = (
                np.sqrt(temp_ub[temp_ub > 0.0]) + aub_sd[temp_ub > 0.0]
            )
            i_ub = temp_ub > TINY * resid
            all_t_ub = np.ones_like(resid)
            all_t_ub[i_ub] = np.minimum(
                all_t_ub[i_ub],
                resid[i_ub] / temp_ub[i_ub],
            )
            t_ub = np.min(all_t_ub, initial=1.0)
            t_min = min(t_bd, t_ub)

            # Calculate some curvature information.
            hess_step = hess_prod(step_proj)
            hess_sd = hess_prod(sd)
            curv_step = step_proj @ hess_step
            curv_sd = sd @ hess_sd
            curv_step_sd = step_proj @ hess_sd

            # For a range of equally spaced values of tan(0.5 * theta),
            # calculate the reduction in the objective function that would be
            # obtained by accepting the corresponding angle.
            n_samples = 20
            n_samples = int((n_samples - 3) * t_min + 3)
            t_samples = np.linspace(t_min / n_samples, t_min, n_samples)
            sin_values = 2.0 * t_samples / (1.0 + t_samples**2.0)
            all_reduct = sin_values * (
                grad_step * t_samples
                - grad_sd
                - sin_values
                * (
                    0.5 * t_samples**2.0 * curv_step
                    - 2.0 * t_samples * curv_step_sd
                    + 0.5 * curv_sd
                )
            )
            if np.all(all_reduct <= 0.0):
                # No reduction in the objective function is obtained.
                break

            # Accept the angle that provides the largest reduction in the
            # objective function, and update the iterate.
            i_max = np.argmax(all_reduct)
            cos_value = (1.0 - t_samples[i_max] ** 2.0) / (
                1.0 + t_samples[i_max] ** 2.0
            )
            step = np.clip(
                step + (cos_value - 1.0) * step_proj + sin_values[i_max] * sd,
                xl,
                xu,
            )
            grad += (cos_value - 1.0) * hess_step + sin_values[i_max] * hess_sd
            resid = np.maximum(
                0.0,
                resid
                - (cos_value - 1.0) * aub_step
                - sin_values[i_max] * aub_sd,
            )
            reduct += all_reduct[i_max]

            # If the above angle is restricted by bound constraints, add them
            # to the working set, and restart the alternative iteration.
            # Otherwise, the calculations are terminated.
            if t_min < 1.0 and i_max == n_samples - 1:
                if t_xl <= t_min:
                    i_new = _argmin(all_t_xl)
                    step[i_new] = xl[i_new]
                    free_xl[i_new] = False
                if t_xu <= t_min:
                    i_new = _argmin(all_t_xu)
                    step[i_new] = xu[i_new]
                    free_xl[i_new] = False
                if t_ub <= t_min:
                    i_new = _argmin(all_t_ub)
                    free_ub[i_new] = False
                n_act, q = qr_tangential_byrd_omojokun(
                    aub,
                    aeq,
                    free_xl,
                    free_xu,
                    free_ub,
                )
            else:
                break

        # Ensure that the alternative iteration improves the objective
        # function.
        if grad_orig @ step + 0.5 * step @ hess_prod(
            step
        ) > grad_orig @ step_base + 0.5 * step_base @ hess_prod(step_base):
            step = step_base

    if debug:
        tol = get_arrays_tol(xl, xu)
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.all(aub @ step <= bub + tol)
        assert np.all(np.abs(aeq @ step) <= tol)
        assert np.linalg.norm(step) < 1.1 * delta
    return step


def normal_byrd_omojokun(aub, bub, aeq, beq, xl, xu, delta, debug, **kwargs):
    r"""
    Minimize approximately a linear constraint violation subject to bound
    constraints in a trust region.

    This function solves approximately

    .. math::

        \min_{s \in \mathbb{R}^n} \quad \frac{1}{2} \big( \lVert \max \{
        A_{\scriptscriptstyle I} s - b_{\scriptscriptstyle I}, 0 \} \rVert^2 +
        \lVert A_{\scriptscriptstyle E} s - b_{\scriptscriptstyle E} \rVert^2
        \big) \quad \text{s.t.}
        \quad
        \left\{ \begin{array}{l}
            l \le s \le u,\\
            \lVert s \rVert \le \Delta,
        \end{array} \right.

    using a variation of the truncated conjugate gradient method.

    Parameters
    ----------
    aub : `numpy.ndarray`, shape (m_linear_ub, n)
        Matrix :math:`A_{\scriptscriptstyle I}` as shown above.
    bub : `numpy.ndarray`, shape (m_linear_ub,)
        Vector :math:`b_{\scriptscriptstyle I}` as shown above.
    aeq : `numpy.ndarray`, shape (m_linear_eq, n)
        Matrix :math:`A_{\scriptscriptstyle E}` as shown above.
    beq : `numpy.ndarray`, shape (m_linear_eq,)
        Vector :math:`b_{\scriptscriptstyle E}` as shown above.
    xl : `numpy.ndarray`, shape (n,)
        Lower bounds :math:`l` as shown above.
    xu : `numpy.ndarray`, shape (n,)
        Upper bounds :math:`u` as shown above.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.
    debug : bool
        Whether to make debugging tests during the execution.

    Returns
    -------
    `numpy.ndarray`, shape (n,)
        Approximate solution :math:`s`.

    Other Parameters
    ----------------
    improve_tcg : bool, optional
        If True, a solution generated by the truncated conjugate gradient
        method that is on the boundary of the trust region is improved by
        moving around the trust-region boundary on the two-dimensional space
        spanned by the solution and the gradient of the quadratic function at
        the solution (default is True).

    Notes
    -----
    This function implements Algorithm 6.4 of [1]_. It is assumed that the
    origin is feasible with respect to the bound constraints and that `delta`
    is finite and positive.

    References
    ----------
    .. [1] T. M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.
    """
    if debug:
        assert isinstance(aub, np.ndarray) and aub.ndim == 2
        assert (
            isinstance(bub, np.ndarray)
            and bub.ndim == 1
            and bub.size == aub.shape[0]
        )
        assert (
            isinstance(aeq, np.ndarray)
            and aeq.ndim == 2
            and aeq.shape[1] == aub.shape[1]
        )
        assert (
            isinstance(beq, np.ndarray)
            and beq.ndim == 1
            and beq.size == aeq.shape[0]
        )
        assert isinstance(xl, np.ndarray) and xl.shape == (aub.shape[1],)
        assert isinstance(xu, np.ndarray) and xu.shape == (aub.shape[1],)
        assert isinstance(delta, float)
        assert isinstance(debug, bool)
        tol = get_arrays_tol(xl, xu)
        assert np.all(xl <= tol)
        assert np.all(xu >= -tol)
        assert np.isfinite(delta) and delta > 0.0
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    # Calculate the initial active set.
    m_linear_ub, n = aub.shape
    grad = np.r_[aeq.T @ -beq, np.maximum(0.0, -bub)]
    free_xl = (xl < 0.0) | (grad[:n] < 0.0)
    free_xu = (xu > 0.0) | (grad[:n] > 0.0)
    free_slack = bub < 0.0
    free_ub = (bub > 0.0) | (aub @ grad[:n] - grad[n:] > 0.0)
    n_act, q = qr_normal_byrd_omojokun(
        aub,
        free_xl,
        free_xu,
        free_slack,
        free_ub,
    )

    # Calculate an upper bound on the norm of the slack variables. It is not
    # used in the original algorithm, but it may prevent undesired behaviors
    # engendered by computer rounding errors.
    delta_slack = np.sqrt(beq @ beq + grad[n:] @ grad[n:])

    # Set the initial iterate and the initial search direction.
    step = np.zeros(n)
    sd = -q[:, n_act:] @ (q[:, n_act:].T @ grad)
    resid = bub + grad[n:]

    k = 0
    reduct = 0.0
    boundary_reached = False
    while k < n + m_linear_ub - n_act:
        # Stop the computations if sd is not a descent direction.
        grad_sd = grad @ sd
        if grad_sd >= -10.0 * EPS * n * max(1.0, np.linalg.norm(grad)):
            break

        # Set alpha_tr to the step size for the trust-region constraint.
        try:
            alpha_tr = _alpha_tr(step, sd[:n], delta)
        except ZeroDivisionError:
            alpha_tr = np.inf

        # Prevent undesired behaviors engendered by computer rounding errors by
        # considering the trust-region constraint on the slack variables.
        try:
            alpha_tr = min(alpha_tr, _alpha_tr(grad[n:], sd[n:], delta_slack))
        except ZeroDivisionError:
            pass

        # Stop the computations if a step along sd is expected to give a
        # relatively small reduction in the objective function.
        if -alpha_tr * grad_sd <= 1e-8 * reduct:
            break

        # Set alpha_quad to the step size for the minimization problem.
        hess_sd = np.r_[aeq.T @ (aeq @ sd[:n]), sd[n:]]
        curv_sd = sd @ hess_sd
        if curv_sd > TINY * abs(grad_sd):
            alpha_quad = max(-grad_sd / curv_sd, 0.0)
        else:
            alpha_quad = np.inf

        # Stop the computations if the reduction in the objective function
        # provided by an unconstrained step is small.
        alpha = min(alpha_tr, alpha_quad)
        if -alpha * (grad_sd + 0.5 * alpha * curv_sd) <= 1e-8 * reduct:
            break

        # Set alpha_bd to the step size for the bound constraints.
        i_xl = free_xl & (xl > -np.inf) & (sd[:n] < -TINY * np.abs(xl - step))
        i_xu = free_xu & (xu < np.inf) & (sd[:n] > TINY * np.abs(xu - step))
        i_slack = free_slack & (sd[n:] < -TINY * np.abs(grad[n:]))
        all_alpha_xl = np.full_like(step, np.inf)
        all_alpha_xu = np.full_like(step, np.inf)
        all_alpha_slack = np.full_like(bub, np.inf)
        all_alpha_xl[i_xl] = np.maximum(
            (xl[i_xl] - step[i_xl]) / sd[:n][i_xl],
            0.0,
        )
        all_alpha_xu[i_xu] = np.maximum(
            (xu[i_xu] - step[i_xu]) / sd[:n][i_xu],
            0.0,
        )
        all_alpha_slack[i_slack] = np.maximum(
            -grad[n:][i_slack] / sd[n:][i_slack],
            0.0,
        )
        alpha_xl = np.min(all_alpha_xl)
        alpha_xu = np.min(all_alpha_xu)
        alpha_slack = np.min(all_alpha_slack, initial=np.inf)
        alpha_bd = min(alpha_xl, alpha_xu, alpha_slack)

        # Set alpha_ub to the step size for the linear constraints.
        aub_sd = aub @ sd[:n] - sd[n:]
        i_ub = free_ub & (aub_sd > TINY * np.abs(resid))
        all_alpha_ub = np.full_like(bub, np.inf)
        all_alpha_ub[i_ub] = resid[i_ub] / aub_sd[i_ub]
        alpha_ub = np.min(all_alpha_ub, initial=np.inf)

        # Update the iterate.
        alpha = min(alpha, alpha_bd, alpha_ub)
        if alpha > 0.0:
            step = np.clip(step + alpha * sd[:n], xl, xu)
            grad += alpha * hess_sd
            resid = np.maximum(0.0, resid - alpha * aub_sd)
            reduct -= alpha * (grad_sd + 0.5 * alpha * curv_sd)

        if alpha < min(alpha_tr, alpha_bd, alpha_ub):
            # The current iteration is a conjugate gradient iteration. Update
            # the search direction so that it is conjugate (with respect to H)
            # to all the previous search directions.
            grad_proj = q[:, n_act:] @ (q[:, n_act:].T @ grad)
            beta = (grad_proj @ hess_sd) / curv_sd
            sd = beta * sd - grad_proj
            k += 1
        elif alpha < alpha_tr:
            # The iterate is restricted by a bound/linear constraint. Add this
            # constraint to the active set, and restart the calculations.
            if alpha_xl <= alpha:
                i_new = np.argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_xl[i_new] = False
            elif alpha_xu <= alpha:
                i_new = np.argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_xu[i_new] = False
            elif alpha_slack <= alpha:
                i_new = np.argmin(all_alpha_slack)
                free_slack[i_new] = False
            else:
                i_new = np.argmin(all_alpha_ub)
                free_ub[i_new] = False
            n_act, q = qr_normal_byrd_omojokun(
                aub, free_xl, free_xu, free_slack, free_ub
            )
            sd = -q[:, n_act:] @ (q[:, n_act:].T @ grad)
            k = 0
        else:
            # The current iterate is on the trust-region boundary. Add all the
            # active bound constraints to the working set to prepare for the
            # improvement of the solution, and stop the iterations.
            if alpha_xl <= alpha:
                i_new = _argmin(all_alpha_xl)
                step[i_new] = xl[i_new]
                free_xl[i_new] = False
            if alpha_xu <= alpha:
                i_new = _argmin(all_alpha_xu)
                step[i_new] = xu[i_new]
                free_xu[i_new] = False
            boundary_reached = True
            break

    # Attempt to improve the solution on the trust-region boundary.
    if kwargs.get("improve_tcg", True) and boundary_reached:
        step_base = np.copy(step)
        free_bd = free_xl & free_xu
        grad = aub.T @ np.maximum(aub @ step - bub, 0.0) + aeq.T @ (
            aeq @ step - beq
        )
        sd = np.zeros(n)
        while np.count_nonzero(free_bd) > 0:
            # Check whether a substantial reduction in the objective function
            # is possible, and set the search direction.
            step_sq = step[free_bd] @ step[free_bd]
            grad_sq = grad[free_bd] @ grad[free_bd]
            grad_step = grad[free_bd] @ step[free_bd]
            grad_sd = -np.sqrt(max(step_sq * grad_sq - grad_step**2.0, 0.0))
            sd[free_bd] = grad_step * step[free_bd] - step_sq * grad[free_bd]
            sd[~free_bd] = 0.0
            if grad_sd >= -1e-8 * reduct or np.any(
                grad_sd >= -TINY * np.abs(sd[free_bd])
            ):
                break
            sd[free_bd] /= -grad_sd

            # Calculate an upper bound for the tangent of half the angle theta
            # of this alternative iteration. The step will be updated as:
            # step = cos(theta) * step + sin(theta) * sd.
            temp_xl = np.zeros(n)
            temp_xu = np.zeros(n)
            temp_xl[free_bd] = (
                step[free_bd] ** 2.0 + sd[free_bd] ** 2.0 - xl[free_bd] ** 2.0
            )
            temp_xu[free_bd] = (
                step[free_bd] ** 2.0 + sd[free_bd] ** 2.0 - xu[free_bd] ** 2.0
            )
            temp_xl[temp_xl > 0.0] = (
                np.sqrt(temp_xl[temp_xl > 0.0]) - sd[temp_xl > 0.0]
            )
            temp_xu[temp_xu > 0.0] = (
                np.sqrt(temp_xu[temp_xu > 0.0]) + sd[temp_xu > 0.0]
            )
            dist_xl = np.maximum(step - xl, 0.0)
            dist_xu = np.maximum(xu - step, 0.0)
            i_xl = temp_xl > TINY * dist_xl
            i_xu = temp_xu > TINY * dist_xu
            all_t_xl = np.ones(n)
            all_t_xu = np.ones(n)
            all_t_xl[i_xl] = np.minimum(
                all_t_xl[i_xl],
                dist_xl[i_xl] / temp_xl[i_xl],
            )
            all_t_xu[i_xu] = np.minimum(
                all_t_xu[i_xu],
                dist_xu[i_xu] / temp_xu[i_xu],
            )
            t_xl = np.min(all_t_xl)
            t_xu = np.min(all_t_xu)
            t_bd = min(t_xl, t_xu)

            # For a range of equally spaced values of tan(0.5 * theta),
            # calculate the reduction in the objective function that would be
            # obtained by accepting the corresponding angle.
            n_samples = 20
            n_samples = int((n_samples - 3) * t_bd + 3)
            t_samples = np.linspace(t_bd / n_samples, t_bd, n_samples)
            resid_ub = np.maximum(aub @ step - bub, 0.0)
            resid_eq = aeq @ step - beq
            step_proj = np.copy(step)
            step_proj[~free_bd] = 0.0
            all_reduct = np.empty(n_samples)
            for i in range(n_samples):
                sin_value = 2.0 * t_samples[i] / (1.0 + t_samples[i] ** 2.0)
                step_alt = np.clip(
                    step + sin_value * (sd - t_samples[i] * step_proj),
                    xl,
                    xu,
                )
                resid_ub_alt = np.maximum(aub @ step_alt - bub, 0.0)
                resid_eq_alt = aeq @ step_alt - beq
                all_reduct[i] = 0.5 * (
                    resid_ub @ resid_ub
                    + resid_eq @ resid_eq
                    - resid_ub_alt @ resid_ub_alt
                    - resid_eq_alt @ resid_eq_alt
                )
            if np.all(all_reduct <= 0.0):
                # No reduction in the objective function is obtained.
                break

            # Accept the angle that provides the largest reduction in the
            # objective function, and update the iterate.
            i_max = np.argmax(all_reduct)
            cos_value = (1.0 - t_samples[i_max] ** 2.0) / (
                1.0 + t_samples[i_max] ** 2.0
            )
            sin_value = (2.0 * t_samples[i_max]
                         / (1.0 + t_samples[i_max] ** 2.0))
            step[free_bd] = cos_value * step[free_bd] + sin_value * sd[free_bd]
            grad = aub.T @ np.maximum(aub @ step - bub, 0.0) + aeq.T @ (
                aeq @ step - beq
            )
            reduct += all_reduct[i_max]

            # If the above angle is restricted by bound constraints, add them
            # to the working set, and restart the alternative iteration.
            # Otherwise, the calculations are terminated.
            if t_bd < 1.0 and i_max == n_samples - 1:
                if t_xl <= t_bd:
                    i_new = _argmin(all_t_xl)
                    step[i_new] = xl[i_new]
                    free_bd[i_new] = False
                if t_xu <= t_bd:
                    i_new = _argmin(all_t_xu)
                    step[i_new] = xu[i_new]
                    free_bd[i_new] = False
            else:
                break

        # Ensure that the alternative iteration improves the objective
        # function.
        resid_ub = np.maximum(aub @ step - bub, 0.0)
        resid_ub_base = np.maximum(aub @ step_base - bub, 0.0)
        resid_eq = aeq @ step - beq
        resid_eq_base = aeq @ step_base - beq
        if (
            resid_ub @ resid_ub + resid_eq @ resid_eq
            > resid_ub_base @ resid_ub_base + resid_eq_base @ resid_eq_base
        ):
            step = step_base

    if debug:
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.linalg.norm(step) < 1.1 * delta
    return step


def qr_tangential_byrd_omojokun(aub, aeq, free_xl, free_xu, free_ub):
    n = free_xl.size
    identity = np.eye(n)
    q, r, _ = qr(
        np.block(
            [
                [aeq],
                [aub[~free_ub, :]],
                [-identity[~free_xl, :]],
                [identity[~free_xu, :]],
            ]
        ).T,
        pivoting=True,
    )
    n_act = np.count_nonzero(
        np.abs(np.diag(r))
        >= 10.0
        * EPS
        * n
        * np.linalg.norm(r[: np.min(r.shape), : np.min(r.shape)], axis=0)
    )
    return n_act, q


def qr_normal_byrd_omojokun(aub, free_xl, free_xu, free_slack, free_ub):
    m_linear_ub, n = aub.shape
    identity_n = np.eye(n)
    identity_m = np.eye(m_linear_ub)
    q, r, _ = qr(
        np.block(
            [
                [
                    aub[~free_ub, :],
                    -identity_m[~free_ub, :],
                ],
                [
                    np.zeros((m_linear_ub - np.count_nonzero(free_slack), n)),
                    -identity_m[~free_slack, :],
                ],
                [
                    -identity_n[~free_xl, :],
                    np.zeros((n - np.count_nonzero(free_xl), m_linear_ub)),
                ],
                [
                    identity_n[~free_xu, :],
                    np.zeros((n - np.count_nonzero(free_xu), m_linear_ub)),
                ],
            ]
        ).T,
        pivoting=True,
    )
    n_act = np.count_nonzero(
        np.abs(np.diag(r))
        >= 10.0
        * EPS
        * (n + m_linear_ub)
        * np.linalg.norm(r[: np.min(r.shape), : np.min(r.shape)], axis=0)
    )
    return n_act, q


def _alpha_tr(step, sd, delta):
    step_sd = step @ sd
    sd_sq = sd @ sd
    dist_tr_sq = delta**2.0 - step @ step
    temp = np.sqrt(max(step_sd**2.0 + sd_sq * dist_tr_sq, 0.0))
    if step_sd <= 0.0 and sd_sq > TINY * abs(temp - step_sd):
        alpha_tr = max((temp - step_sd) / sd_sq, 0.0)
    elif abs(temp + step_sd) > TINY * dist_tr_sq:
        alpha_tr = max(dist_tr_sq / (temp + step_sd), 0.0)
    else:
        raise ZeroDivisionError
    return alpha_tr


def _argmax(x):
    return np.flatnonzero(x >= np.max(x))


def _argmin(x):
    return np.flatnonzero(x <= np.min(x))
<|endoftext|>import inspect

import numpy as np

from ..utils import get_arrays_tol


TINY = np.finfo(float).tiny


def cauchy_geometry(const, grad, curv, xl, xu, delta, debug):
    r"""
    Maximize approximately the absolute value of a quadratic function subject
    to bound constraints in a trust region.

    This function solves approximately

    .. math::

        \max_{s \in \mathbb{R}^n} \quad \bigg\lvert c + g^{\mathsf{T}} s +
        \frac{1}{2} s^{\mathsf{T}} H s \bigg\rvert \quad \text{s.t.} \quad
        \left\{ \begin{array}{l}
            l \le s \le u,\\
            \lVert s \rVert \le \Delta,
        \end{array} \right.

    by maximizing the objective function along the constrained Cauchy
    direction.

    Parameters
    ----------
    const : float
        Constant :math:`c` as shown above.
    grad : `numpy.ndarray`, shape (n,)
        Gradient :math:`g` as shown above.
    curv : callable
        Curvature of :math:`H` along any vector.

            ``curv(s) -> float``

        returns :math:`s^{\mathsf{T}} H s`.
    xl : `numpy.ndarray`, shape (n,)
        Lower bounds :math:`l` as shown above.
    xu : `numpy.ndarray`, shape (n,)
        Upper bounds :math:`u` as shown above.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.
    debug : bool
        Whether to make debugging tests during the execution.

    Returns
    -------
    `numpy.ndarray`, shape (n,)
        Approximate solution :math:`s`.

    Notes
    -----
    This function is described as the first alternative in Section 6.5 of [1]_.
    It is assumed that the origin is feasible with respect to the bound
    constraints and that `delta` is finite and positive.

    References
    ----------
    .. [1] T. M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.
    """
    if debug:
        assert isinstance(const, float)
        assert isinstance(grad, np.ndarray) and grad.ndim == 1
        assert inspect.signature(curv).bind(grad)
        assert isinstance(xl, np.ndarray) and xl.shape == grad.shape
        assert isinstance(xu, np.ndarray) and xu.shape == grad.shape
        assert isinstance(delta, float)
        assert isinstance(debug, bool)
        tol = get_arrays_tol(xl, xu)
        assert np.all(xl <= tol)
        assert np.all(xu >= -tol)
        assert np.isfinite(delta) and delta > 0.0
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    # To maximize the absolute value of a quadratic function, we maximize the
    # function itself or its negative, and we choose the solution that provides
    # the largest function value.
    step1, q_val1 = _cauchy_geom(const, grad, curv, xl, xu, delta, debug)
    step2, q_val2 = _cauchy_geom(
        -const,
        -grad,
        lambda x: -curv(x),
        xl,
        xu,
        delta,
        debug,
    )
    step = step1 if abs(q_val1) >= abs(q_val2) else step2

    if debug:
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.linalg.norm(step) < 1.1 * delta
    return step


def spider_geometry(const, grad, curv, xpt, xl, xu, delta, debug):
    r"""
    Maximize approximately the absolute value of a quadratic function subject
    to bound constraints in a trust region.

    This function solves approximately

    .. math::

        \max_{s \in \mathbb{R}^n} \quad \bigg\lvert c + g^{\mathsf{T}} s +
        \frac{1}{2} s^{\mathsf{T}} H s \bigg\rvert \quad \text{s.t.} \quad
        \left\{ \begin{array}{l}
            l \le s \le u,\\
            \lVert s \rVert \le \Delta,
        \end{array} \right.

    by maximizing the objective function along given straight lines.

    Parameters
    ----------
    const : float
        Constant :math:`c` as shown above.
    grad : `numpy.ndarray`, shape (n,)
        Gradient :math:`g` as shown above.
    curv : callable
        Curvature of :math:`H` along any vector.

            ``curv(s) -> float``

        returns :math:`s^{\mathsf{T}} H s`.
    xpt : `numpy.ndarray`, shape (n, npt)
        Points defining the straight lines. The straight lines considered are
        the ones passing through the origin and the points in `xpt`.
    xl : `numpy.ndarray`, shape (n,)
        Lower bounds :math:`l` as shown above.
    xu : `numpy.ndarray`, shape (n,)
        Upper bounds :math:`u` as shown above.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.
    debug : bool
        Whether to make debugging tests during the execution.

    Returns
    -------
    `numpy.ndarray`, shape (n,)
        Approximate solution :math:`s`.

    Notes
    -----
    This function is described as the second alternative in Section 6.5 of
    [1]_. It is assumed that the origin is feasible with respect to the bound
    constraints and that `delta` is finite and positive.

    References
    ----------
    .. [1] T. M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.
    """
    if debug:
        assert isinstance(const, float)
        assert isinstance(grad, np.ndarray) and grad.ndim == 1
        assert inspect.signature(curv).bind(grad)
        assert (
            isinstance(xpt, np.ndarray)
            and xpt.ndim == 2
            and xpt.shape[0] == grad.size
        )
        assert isinstance(xl, np.ndarray) and xl.shape == grad.shape
        assert isinstance(xu, np.ndarray) and xu.shape == grad.shape
        assert isinstance(delta, float)
        assert isinstance(debug, bool)
        tol = get_arrays_tol(xl, xu)
        assert np.all(xl <= tol)
        assert np.all(xu >= -tol)
        assert np.isfinite(delta) and delta > 0.0
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    # Iterate through the straight lines.
    step = np.zeros_like(grad)
    q_val = const
    s_norm = np.linalg.norm(xpt, axis=0)

    # Set alpha_xl to the step size for the lower-bound constraint and
    # alpha_xu to the step size for the upper-bound constraint.

    # xl.shape = (N,)
    # xpt.shape = (N, M)
    # i_xl_pos.shape = (M, N)
    i_xl_pos = (xl > -np.inf) & (xpt.T > -TINY * xl)
    i_xl_neg = (xl > -np.inf) & (xpt.T < TINY * xl)
    i_xu_pos = (xu < np.inf) & (xpt.T > TINY * xu)
    i_xu_neg = (xu < np.inf) & (xpt.T < -TINY * xu)

    # (M, N)
    alpha_xl_pos = np.atleast_2d(
        np.broadcast_to(xl, i_xl_pos.shape)[i_xl_pos] / xpt.T[i_xl_pos]
    )
    # (M,)
    alpha_xl_pos = np.max(alpha_xl_pos, axis=1, initial=-np.inf)
    # make sure it's (M,)
    alpha_xl_pos = np.broadcast_to(np.atleast_1d(alpha_xl_pos), xpt.shape[1])

    alpha_xl_neg = np.atleast_2d(
        np.broadcast_to(xl, i_xl_neg.shape)[i_xl_neg] / xpt.T[i_xl_neg]
    )
    alpha_xl_neg = np.max(alpha_xl_neg, axis=1, initial=np.inf)
    alpha_xl_neg = np.broadcast_to(np.atleast_1d(alpha_xl_neg), xpt.shape[1])

    alpha_xu_neg = np.atleast_2d(
        np.broadcast_to(xu, i_xu_neg.shape)[i_xu_neg] / xpt.T[i_xu_neg]
    )
    alpha_xu_neg = np.max(alpha_xu_neg, axis=1, initial=-np.inf)
    alpha_xu_neg = np.broadcast_to(np.atleast_1d(alpha_xu_neg), xpt.shape[1])

    alpha_xu_pos = np.atleast_2d(
        np.broadcast_to(xu, i_xu_pos.shape)[i_xu_pos] / xpt.T[i_xu_pos]
    )
    alpha_xu_pos = np.max(alpha_xu_pos, axis=1, initial=np.inf)
