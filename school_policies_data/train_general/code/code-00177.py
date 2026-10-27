        """Reset the engine to base state.

        Returns
        -------
        engine : Sobol
            Engine reset to its base state.

        """
        super().reset()
        self._quasi = self._shift.copy()
        return self

    def fast_forward(self, n: IntNumber) -> "Sobol":
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : Sobol
            The fast-forwarded engine.

        """
        if self.num_generated == 0:
            _fast_forward(
                n=n - 1, num_gen=self.num_generated, dim=self.d,
                sv=self._sv, quasi=self._quasi
            )
        else:
            _fast_forward(
                n=n, num_gen=self.num_generated - 1, dim=self.d,
                sv=self._sv, quasi=self._quasi
            )
        self.num_generated += n
        return self


class PoissonDisk(QMCEngine):
    """Poisson disk sampling.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    radius : float
        Minimal distance to keep between points when sampling new candidates.
    hypersphere : {"volume", "surface"}, optional
        Sampling strategy to generate potential candidates to be added in the
        final sample. Default is "volume".

        * ``volume``: original Bridson algorithm as described in [1]_.
          New candidates are sampled *within* the hypersphere.
        * ``surface``: only sample the surface of the hypersphere.
    ncandidates : int
        Number of candidates to sample per iteration. More candidates result
        in a denser sampling as more candidates can be accepted per iteration.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0

    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `seed` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `seed` keyword will emit warnings. Following a
            deprecation period, the `seed` keyword will be removed.

    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds of target sample data.

    Notes
    -----
    Poisson disk sampling is an iterative sampling strategy. Starting from
    a seed sample, `ncandidates` are sampled in the hypersphere
    surrounding the seed. Candidates below a certain `radius` or outside the
    domain are rejected. New samples are added in a pool of sample seed. The
    process stops when the pool is empty or when the number of required
    samples is reached.

    The maximum number of point that a sample can contain is directly linked
    to the `radius`. As the dimension of the space increases, a higher radius
    spreads the points further and help overcome the curse of dimensionality.
    See the :ref:`quasi monte carlo tutorial <quasi-monte-carlo>` for more
    details.

    .. warning::

       The algorithm is more suitable for low dimensions and sampling size
       due to its iterative nature and memory requirements.
       Selecting a small radius with a high dimension would
       mean that the space could contain more samples than using lower
       dimension or a bigger radius.

    Some code taken from [2]_, written consent given on 31.03.2021
    by the original author, Shamis, for free use in SciPy under
    the 3-clause BSD.

    References
    ----------
    .. [1] Robert Bridson, "Fast Poisson Disk Sampling in Arbitrary
       Dimensions." SIGGRAPH, 2007.
    .. [2] `StackOverflow <https://stackoverflow.com/questions/66047540>`__.

    Examples
    --------
    Generate a 2D sample using a `radius` of 0.2.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.collections import PatchCollection
    >>> from scipy.stats import qmc
    >>>
    >>> rng = np.random.default_rng()
    >>> radius = 0.2
    >>> engine = qmc.PoissonDisk(d=2, radius=radius, rng=rng)
    >>> sample = engine.random(20)

    Visualizing the 2D sample and showing that no points are closer than
    `radius`. ``radius/2`` is used to visualize non-intersecting circles.
    If two samples are exactly at `radius` from each other, then their circle
    of radius ``radius/2`` will touch.

    >>> fig, ax = plt.subplots()
    >>> _ = ax.scatter(sample[:, 0], sample[:, 1])
    >>> circles = [plt.Circle((xi, yi), radius=radius/2, fill=False)
    ...            for xi, yi in sample]
    >>> collection = PatchCollection(circles, match_original=True)
    >>> ax.add_collection(collection)
    >>> _ = ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$',
    ...            xlim=[0, 1], ylim=[0, 1])
    >>> plt.show()

    Such visualization can be seen as circle packing: how many circle can
    we put in the space. It is a np-hard problem. The method `fill_space`
    can be used to add samples until no more samples can be added. This is
    a hard problem and parameters may need to be adjusted manually. Beware of
    the dimension: as the dimensionality increases, the number of samples
    required to fill the space increases exponentially
    (curse-of-dimensionality).

    """

    @_transition_to_rng('seed', replace_doc=False)
    def __init__(
        self,
        d: IntNumber,
        *,
        radius: DecimalNumber = 0.05,
        hypersphere: Literal["volume", "surface"] = "volume",
        ncandidates: IntNumber = 30,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        rng: SeedType = None,
        l_bounds: "npt.ArrayLike | None" = None,
        u_bounds: "npt.ArrayLike | None" = None,
    ) -> None:
        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'radius': radius,
                           'hypersphere': hypersphere,
                           'ncandidates': ncandidates,
                           'optimization': optimization}
        super()._initialize(d=d, optimization=optimization, rng=rng)

        hypersphere_sample = {
            "volume": self._hypersphere_volume_sample,
            "surface": self._hypersphere_surface_sample
        }

        try:
            self.hypersphere_method = hypersphere_sample[hypersphere]
        except KeyError as exc:
            message = (
                f"{hypersphere!r} is not a valid hypersphere sampling"
                f" method. It must be one of {set(hypersphere_sample)!r}")
            raise ValueError(message) from exc

        # size of the sphere from which the samples are drawn relative to the
        # size of a disk (radius)
        # for the surface sampler, all new points are almost exactly 1 radius
        # away from at least one existing sample +eps to avoid rejection
        self.radius_factor = 2 if hypersphere == "volume" else 1.001
        self.radius = radius
        self.radius_squared = self.radius**2

        # sample to generate per iteration in the hypersphere around center
        self.ncandidates = ncandidates
        
        if u_bounds is None:
            u_bounds = np.ones(d)
        if l_bounds is None:
            l_bounds = np.zeros(d)
        self.l_bounds, self.u_bounds = _validate_bounds(
            l_bounds=l_bounds, u_bounds=u_bounds, d=int(d)
        )

        with np.errstate(divide='ignore'):
            self.cell_size = self.radius / np.sqrt(self.d)
            self.grid_size = (
                np.ceil((self.u_bounds - self.l_bounds) / self.cell_size)
            ).astype(int)

        self._initialize_grid_pool()

    def _initialize_grid_pool(self):
        """Sampling pool and sample grid."""
        self.sample_pool = []
        # Positions of cells
        # n-dim value for each grid cell
        self.sample_grid = np.empty(
            np.append(self.grid_size, self.d),
            dtype=np.float32
        )
        # Initialise empty cells with NaNs
        self.sample_grid.fill(np.nan)

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw `n` in the interval ``[l_bounds, u_bounds]``.

        Note that it can return fewer samples if the space is full.
        See the note section of the class.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        if n == 0 or self.d == 0:
            return np.empty((n, self.d))

        def in_limits(sample: np.ndarray) -> bool:
            for i in range(self.d):
                if (sample[i] > self.u_bounds[i] or sample[i] < self.l_bounds[i]):
                    return False
            return True

        def in_neighborhood(candidate: np.ndarray, n: int = 2) -> bool:
            """
            Check if there are samples closer than ``radius_squared`` to the
            `candidate` sample.
            """
            indices = ((candidate - self.l_bounds) / self.cell_size).astype(int)
            ind_min = np.maximum(indices - n, self.l_bounds.astype(int))
            ind_max = np.minimum(indices + n + 1, self.grid_size)

            # Check if the center cell is empty
            if not np.isnan(self.sample_grid[tuple(indices)][0]):
                return True

            a = [slice(ind_min[i], ind_max[i]) for i in range(self.d)]

            # guards against: invalid value encountered in less as we are
            # comparing with nan and returns False. Which is wanted.
            with np.errstate(invalid='ignore'):
                if np.any(
                    np.sum(
                        np.square(candidate - self.sample_grid[tuple(a)]),
                        axis=self.d
                    ) < self.radius_squared
                ):
                    return True

            return False

        def add_sample(candidate: np.ndarray) -> None:
            self.sample_pool.append(candidate)
            indices = ((candidate - self.l_bounds) / self.cell_size).astype(int)
            self.sample_grid[tuple(indices)] = candidate
            curr_sample.append(candidate)

        curr_sample: list[np.ndarray] = []

        if len(self.sample_pool) == 0:
            # the pool is being initialized with a single random sample
            add_sample(self.rng.uniform(self.l_bounds, self.u_bounds))
            num_drawn = 1
        else:
            num_drawn = 0

        # exhaust sample pool to have up to n sample
        while len(self.sample_pool) and num_drawn < n:
            # select a sample from the available pool
            idx_center = rng_integers(self.rng, len(self.sample_pool))
            center = self.sample_pool[idx_center]
            del self.sample_pool[idx_center]

            # generate candidates around the center sample
            candidates = self.hypersphere_method(
                center, self.radius * self.radius_factor, self.ncandidates
            )

            # keep candidates that satisfy some conditions
            for candidate in candidates:
                if in_limits(candidate) and not in_neighborhood(candidate):
                    add_sample(candidate)

                    num_drawn += 1
                    if num_drawn >= n:
                        break

        self.num_generated += num_drawn
        return np.array(curr_sample)

    def fill_space(self) -> np.ndarray:
        """Draw ``n`` samples in the interval ``[l_bounds, u_bounds]``.

        Unlike `random`, this method will try to add points until
        the space is full. Depending on ``candidates`` (and to a lesser extent
        other parameters), some empty areas can still be present in the sample.

        .. warning::

           This can be extremely slow in high dimensions or if the
           ``radius`` is very small-with respect to the dimensionality.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        return self.random(np.inf)  # type: ignore[arg-type]

    def reset(self) -> "PoissonDisk":
        """Reset the engine to base state.

        Returns
        -------
        engine : PoissonDisk
            Engine reset to its base state.

        """
        super().reset()
        self._initialize_grid_pool()
        return self

    def _hypersphere_volume_sample(
        self, center: np.ndarray, radius: DecimalNumber,
        candidates: IntNumber = 1
    ) -> np.ndarray:
        """Uniform sampling within hypersphere."""
        # should remove samples within r/2
        x = self.rng.standard_normal(size=(candidates, self.d))
        ssq = np.sum(x**2, axis=1)
        fr = radius * gammainc(self.d/2, ssq/2)**(1/self.d) / np.sqrt(ssq)
        fr_tiled = np.tile(
            fr.reshape(-1, 1), (1, self.d)  # type: ignore[arg-type]
        )
        p = center + np.multiply(x, fr_tiled)
        return p

    def _hypersphere_surface_sample(
        self, center: np.ndarray, radius: DecimalNumber,
        candidates: IntNumber = 1
    ) -> np.ndarray:
        """Uniform sampling on the hypersphere's surface."""
        vec = self.rng.standard_normal(size=(candidates, self.d))
        vec /= np.linalg.norm(vec, axis=1)[:, None]
        p = center + np.multiply(vec, radius)
        return p


class MultivariateNormalQMC:
    r"""QMC sampling from a multivariate Normal :math:`N(\mu, \Sigma)`.

    Parameters
    ----------
    mean : array_like (d,)
        The mean vector. Where ``d`` is the dimension.
    cov : array_like (d, d), optional
        The covariance matrix. If omitted, use `cov_root` instead.
        If both `cov` and `cov_root` are omitted, use the identity matrix.
    cov_root : array_like (d, d'), optional
        A root decomposition of the covariance matrix, where ``d'`` may be less
        than ``d`` if the covariance is not full rank. If omitted, use `cov`.
    inv_transform : bool, optional
        If True, use inverse transform instead of Box-Muller. Default is True.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `seed` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `seed` keyword will emit warnings. Following a
            deprecation period, the `seed` keyword will be removed.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> dist = qmc.MultivariateNormalQMC(mean=[0, 5], cov=[[1, 0], [0, 1]])
    >>> sample = dist.random(512)
    >>> _ = plt.scatter(sample[:, 0], sample[:, 1])
    >>> plt.show()

    """

    @_transition_to_rng('seed', replace_doc=False)
    def __init__(
            self,
            mean: "npt.ArrayLike",
            cov: "npt.ArrayLike | None" = None,
            *,
            cov_root: "npt.ArrayLike | None" = None,
            inv_transform: bool = True,
            engine: QMCEngine | None = None,
            rng: SeedType = None,
    ) -> None:
        mean = np.asarray(np.atleast_1d(mean))
        d = mean.shape[0]
        if cov is not None:
            # covariance matrix provided
            cov = np.asarray(np.atleast_2d(cov))
            # check for square/symmetric cov matrix and mean vector has the
            # same d
            if not mean.shape[0] == cov.shape[0]:
                raise ValueError("Dimension mismatch between mean and "
                                 "covariance.")
            if not np.allclose(cov, cov.transpose()):
                raise ValueError("Covariance matrix is not symmetric.")
            # compute Cholesky decomp; if it fails, do the eigen decomposition
            try:
                cov_root = np.linalg.cholesky(cov).transpose()
            except np.linalg.LinAlgError:
                eigval, eigvec = np.linalg.eigh(cov)
                if not np.all(eigval >= -1.0e-8):
                    raise ValueError("Covariance matrix not PSD.")
                eigval = np.clip(eigval, 0.0, None)
                cov_root = (eigvec * np.sqrt(eigval)).transpose()
        elif cov_root is not None:
            # root decomposition provided
            cov_root = np.atleast_2d(cov_root)
            if not mean.shape[0] == cov_root.shape[0]:
                raise ValueError("Dimension mismatch between mean and "
                                 "covariance.")
        else:
            # corresponds to identity covariance matrix
            cov_root = None

        self._inv_transform = inv_transform

        if not inv_transform:
            # to apply Box-Muller, we need an even number of dimensions
            engine_dim = 2 * math.ceil(d / 2)
        else:
            engine_dim = d
        if engine is None:
            # Need this during SPEC 7 transition to prevent `RandomState`
            # from being passed via `rng`.
            kwarg = "seed" if isinstance(rng, np.random.RandomState) else "rng"
            kwargs = {kwarg: rng}
            self.engine = Sobol(
                d=engine_dim, scramble=True, bits=30, **kwargs
            )  # type: QMCEngine
        elif isinstance(engine, QMCEngine):
            if engine.d != engine_dim:
                raise ValueError("Dimension of `engine` must be consistent"
                                 " with dimensions of mean and covariance."
                                 " If `inv_transform` is False, it must be"
                                 " an even number.")
            self.engine = engine
        else:
            raise ValueError("`engine` must be an instance of "
                             "`scipy.stats.qmc.QMCEngine` or `None`.")

        self._mean = mean
        self._corr_matrix = cov_root

        self._d = d

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the multivariate Normal.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        base_samples = self._standard_normal_samples(n)
        return self._correlate(base_samples)

    def _correlate(self, base_samples: np.ndarray) -> np.ndarray:
        if self._corr_matrix is not None:
            return base_samples @ self._corr_matrix + self._mean
        else:
            # avoid multiplying with identity here
            return base_samples + self._mean

    def _standard_normal_samples(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the standard Normal :math:`N(0, I_d)`.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        # get base samples
        samples = self.engine.random(n)
        if self._inv_transform:
            # apply inverse transform
            # (values to close to 0/1 result in inf values)
            return stats.norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))  # type: ignore[attr-defined]  # noqa: E501
        else:
            # apply Box-Muller transform (note: indexes starting from 1)
            even = np.arange(0, samples.shape[-1], 2)
            Rs = np.sqrt(-2 * np.log(samples[:, even]))
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = np.cos(thetas)
            sin = np.sin(thetas)
            transf_samples = np.stack([Rs * cos, Rs * sin],
                                      -1).reshape(n, -1)
            # make sure we only return the number of dimension requested
            return transf_samples[:, : self._d]


class MultinomialQMC:
    r"""QMC sampling from a multinomial distribution.

    Parameters
    ----------
    pvals : array_like (k,)
        Vector of probabilities of size ``k``, where ``k`` is the number
        of categories. Elements must be non-negative and sum to 1.
    n_trials : int
        Number of trials.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `seed` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `seed` keyword will emit warnings. Following a
            deprecation period, the `seed` keyword will be removed.

    Examples
    --------
    Let's define 3 categories and for a given sample, the sum of the trials
    of each category is 8. The number of trials per category is determined
    by the `pvals` associated to each category.
    Then, we sample this distribution 64 times.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> dist = qmc.MultinomialQMC(
    ...     pvals=[0.2, 0.4, 0.4], n_trials=10, engine=qmc.Halton(d=1)
    ... )
    >>> sample = dist.random(64)

    We can plot the sample and verify that the median of number of trials
    for each category is following the `pvals`. That would be
    ``pvals * n_trials = [2, 4, 4]``.

    >>> fig, ax = plt.subplots()
    >>> ax.yaxis.get_major_locator().set_params(integer=True)
    >>> _ = ax.boxplot(sample)
    >>> ax.set(xlabel="Categories", ylabel="Trials")
    >>> plt.show()

    """

    @_transition_to_rng('seed', replace_doc=False)
    def __init__(
        self,
        pvals: "npt.ArrayLike",
        n_trials: IntNumber,
        *,
        engine: QMCEngine | None = None,
        rng: SeedType = None,
    ) -> None:
        self.pvals = np.atleast_1d(np.asarray(pvals))
        if np.min(pvals) < 0:
            raise ValueError('Elements of pvals must be non-negative.')
        if not np.isclose(np.sum(pvals), 1):
            raise ValueError('Elements of pvals must sum to 1.')
        self.n_trials = n_trials
        if engine is None:
            # Need this during SPEC 7 transition to prevent `RandomState`
            # from being passed via `rng`.
            kwarg = "seed" if isinstance(rng, np.random.RandomState) else "rng"
            kwargs = {kwarg: rng}
            self.engine = Sobol(
                d=1, scramble=True, bits=30, **kwargs
            )  # type: QMCEngine
        elif isinstance(engine, QMCEngine):
            if engine.d != 1:
                raise ValueError("Dimension of `engine` must be 1.")
            self.engine = engine
        else:
            raise ValueError("`engine` must be an instance of "
                             "`scipy.stats.qmc.QMCEngine` or `None`.")

    def random(self, n: IntNumber = 1) -> np.ndarray:
        """Draw `n` QMC samples from the multinomial distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        samples : array_like (n, pvals)
            Sample.

        """
        sample = np.empty((n, len(self.pvals)))
        for i in range(n):
            base_draws = self.engine.random(self.n_trials).ravel()
            p_cumulative = np.empty_like(self.pvals, dtype=float)
            _fill_p_cumulative(np.array(self.pvals, dtype=float), p_cumulative)
            sample_ = np.zeros_like(self.pvals, dtype=np.intp)
            _categorize(base_draws, p_cumulative, sample_)
            sample[i] = sample_
        return sample


def _select_optimizer(
    optimization: Literal["random-cd", "lloyd"] | None, config: dict
) -> Callable | None:
    """A factory for optimization methods."""
    optimization_method: dict[str, Callable] = {
        "random-cd": _random_cd,
        "lloyd": _lloyd_centroidal_voronoi_tessellation
    }

    optimizer: partial | None
    if optimization is not None:
        try:
            optimization = optimization.lower()  # type: ignore[assignment]
            optimizer_ = optimization_method[optimization]
        except KeyError as exc:
            message = (f"{optimization!r} is not a valid optimization"
                       f" method. It must be one of"
                       f" {set(optimization_method)!r}")
            raise ValueError(message) from exc

        # config
        optimizer = partial(optimizer_, **config)
    else:
        optimizer = None

    return optimizer


def _random_cd(
    best_sample: np.ndarray, n_iters: int, n_nochange: int, rng: GeneratorType,
    **kwargs: dict
) -> np.ndarray:
    """Optimal LHS on CD.

    Create a base LHS and do random permutations of coordinates to
    lower the centered discrepancy.
    Because it starts with a normal LHS, it also works with the
    `scramble` keyword argument.

    Two stopping criterion are used to stop the algorithm: at most,
    `n_iters` iterations are performed; or if there is no improvement
    for `n_nochange` consecutive iterations.
    """
    del kwargs  # only use keywords which are defined, needed by factory

    n, d = best_sample.shape

    if d == 0 or n == 0:
        return np.empty((n, d))

    if d == 1 or n == 1:
        # discrepancy measures are invariant under permuting factors and runs
        return best_sample

    best_disc = discrepancy(best_sample)

    bounds = ([0, d - 1],
              [0, n - 1],
              [0, n - 1])

    n_nochange_ = 0
    n_iters_ = 0
    while n_nochange_ < n_nochange and n_iters_ < n_iters:
        n_iters_ += 1

        col = rng_integers(rng, *bounds[0], endpoint=True)  # type: ignore[misc]
        row_1 = rng_integers(rng, *bounds[1], endpoint=True)  # type: ignore[misc]
        row_2 = rng_integers(rng, *bounds[2], endpoint=True)  # type: ignore[misc]
        disc = _perturb_discrepancy(best_sample,
                                    row_1, row_2, col,
                                    best_disc)
        if disc < best_disc:
            best_sample[row_1, col], best_sample[row_2, col] = (
                best_sample[row_2, col], best_sample[row_1, col])

            best_disc = disc
            n_nochange_ = 0
        else:
            n_nochange_ += 1

    return best_sample


def _l1_norm(sample: np.ndarray) -> float:
    return distance.pdist(sample, 'cityblock').min()


def _lloyd_iteration(
    sample: np.ndarray,
    decay: float,
    qhull_options: str
) -> np.ndarray:
    """Lloyd-Max algorithm iteration.

    Based on the implementation of Stéfan van der Walt:

    https://github.com/stefanv/lloyd

    which is:

        Copyright (c) 2021-04-21 Stéfan van der Walt
        https://github.com/stefanv/lloyd
        MIT License

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to iterate on.
    decay : float
        Relaxation decay. A positive value would move the samples toward
        their centroid, and negative value would move them away.
        1 would move the samples to their centroid.
    qhull_options : str
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qbb Qc Qz Qj Qx" for ndim > 4 and
        "Qbb Qc Qz Qj" otherwise.)

    Returns
    -------
    sample : array_like (n, d)
        The sample after an iteration of Lloyd's algorithm.

    """
    new_sample = np.empty_like(sample)

    voronoi = Voronoi(sample, qhull_options=qhull_options)

    for ii, idx in enumerate(voronoi.point_region):
        # the region is a series of indices into self.voronoi.vertices
        # remove samples at infinity, designated by index -1
        region = [i for i in voronoi.regions[idx] if i != -1]

        # get the vertices for this region
        verts = voronoi.vertices[region]

        # clipping would be wrong, we need to intersect
        # verts = np.clip(verts, 0, 1)

        # move samples towards centroids:
        # Centroid in n-D is the mean for uniformly distributed nodes
        # of a geometry.
        centroid = np.mean(verts, axis=0)
        new_sample[ii] = sample[ii] + (centroid - sample[ii]) * decay

    # only update sample to centroid within the region
    is_valid = np.all(np.logical_and(new_sample >= 0, new_sample <= 1), axis=1)
    sample[is_valid] = new_sample[is_valid]

    return sample


def _lloyd_centroidal_voronoi_tessellation(
    sample: "npt.ArrayLike",
    *,
    tol: DecimalNumber = 1e-5,
    maxiter: IntNumber = 10,
    qhull_options: str | None = None,
    **kwargs: dict
) -> np.ndarray:
    """Approximate Centroidal Voronoi Tessellation.

    Perturb samples in N-dimensions using Lloyd-Max algorithm.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to iterate on. With ``n`` the number of samples and ``d``
        the dimension. Samples must be in :math:`[0, 1]^d`, with ``d>=2``.
    tol : float, optional
        Tolerance for termination. If the min of the L1-norm over the samples
        changes less than `tol`, it stops the algorithm. Default is 1e-5.
    maxiter : int, optional
        Maximum number of iterations. It will stop the algorithm even if
        `tol` is above the threshold.
        Too many iterations tend to cluster the samples as a hypersphere.
        Default is 10.
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qbb Qc Qz Qj Qx" for ndim > 4 and
        "Qbb Qc Qz Qj" otherwise.)

    Returns
    -------
    sample : array_like (n, d)
        The sample after being processed by Lloyd-Max algorithm.

    Notes
    -----
    Lloyd-Max algorithm is an iterative process with the purpose of improving
    the dispersion of samples. For given sample: (i) compute a Voronoi
    Tessellation; (ii) find the centroid of each Voronoi cell; (iii) move the
    samples toward the centroid of their respective cell. See [1]_, [2]_.

    A relaxation factor is used to control how fast samples can move at each
    iteration. This factor is starting at 2 and ending at 1 after `maxiter`
    following an exponential decay.

    The process converges to equally spaced samples. It implies that measures
    like the discrepancy could suffer from too many iterations. On the other
    hand, L1 and L2 distances should improve. This is especially true with
    QMC methods which tend to favor the discrepancy over other criteria.

    .. note::

        The current implementation does not intersect the Voronoi Tessellation
        with the boundaries. This implies that for a low number of samples,
        empirically below 20, no Voronoi cell is touching the boundaries.
        Hence, samples cannot be moved close to the boundaries.

        Further improvements could consider the samples at infinity so that
        all boundaries are segments of some Voronoi cells. This would fix
        the computation of the centroid position.

    .. warning::

       The Voronoi Tessellation step is expensive and quickly becomes
       intractable with dimensions as low as 10 even for a sample
       of size as low as 1000.

    .. versionadded:: 1.9.0

    References
    ----------
    .. [1] Lloyd. "Least Squares Quantization in PCM".
       IEEE Transactions on Information Theory, 1982.
    .. [2] Max J. "Quantizing for minimum distortion".
       IEEE Transactions on Information Theory, 1960.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial import distance
    >>> from scipy.stats._qmc import _lloyd_centroidal_voronoi_tessellation
    >>> rng = np.random.default_rng()
    >>> sample = rng.random((128, 2))

    .. note::

        The samples need to be in :math:`[0, 1]^d`. `scipy.stats.qmc.scale`
        can be used to scale the samples from their
        original bounds to :math:`[0, 1]^d`. And back to their original bounds.

    Compute the quality of the sample using the L1 criterion.

    >>> def l1_norm(sample):
    ...    return distance.pdist(sample, 'cityblock').min()

    >>> l1_norm(sample)
    0.00161...  # random

    Now process the sample using Lloyd's algorithm and check the improvement
    on the L1. The value should increase.

    >>> sample = _lloyd_centroidal_voronoi_tessellation(sample)
    >>> l1_norm(sample)
    0.0278...  # random

    """
    del kwargs  # only use keywords which are defined, needed by factory

    sample = np.asarray(sample).copy()

    if not sample.ndim == 2:
        raise ValueError('`sample` is not a 2D array')

    if not sample.shape[1] >= 2:
        raise ValueError('`sample` dimension is not >= 2')

    # Checking that sample is within the hypercube
    if (sample.max() > 1.) or (sample.min() < 0.):
        raise ValueError('`sample` is not in unit hypercube')

    if qhull_options is None:
        qhull_options = 'Qbb Qc Qz QJ'

        if sample.shape[1] >= 5:
            qhull_options += ' Qx'

    # Fit an exponential to be 2 at 0 and 1 at `maxiter`.
    # The decay is used for relaxation.
    # analytical solution for y=exp(-maxiter/x) - 0.1
    root = -maxiter / np.log(0.1)
    decay = [np.exp(-x / root)+0.9 for x in range(maxiter)]

    l1_old = _l1_norm(sample=sample)
    for i in range(maxiter):
        sample = _lloyd_iteration(
                sample=sample, decay=decay[i],
                qhull_options=qhull_options,
        )

        l1_new = _l1_norm(sample=sample)

        if abs(l1_new - l1_old) < tol:
            break
        else:
            l1_old = l1_new

    return sample


def _validate_workers(workers: IntNumber = 1) -> IntNumber:
    """Validate `workers` based on platform and value.

    Parameters
    ----------
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is
        given all CPU threads are used. Default is 1.

    Returns
    -------
    Workers : int
        Number of CPU used by the algorithm

    """
    workers = int(workers)
    if workers == -1:
        workers = os.cpu_count()  # type: ignore[assignment]
        if workers is None:
            raise NotImplementedError(
                "Cannot determine the number of cpus using os.cpu_count(), "
                "cannot use -1 for the number of workers"
            )
    elif workers <= 0:
        raise ValueError(f"Invalid number of workers: {workers}, must be -1 "
                         "or > 0")

    return workers


def _validate_bounds(
    l_bounds: "npt.ArrayLike", u_bounds: "npt.ArrayLike", d: int
) -> "tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]":
    """Bounds input validation.

    Parameters
    ----------
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds.
    d : int
        Dimension to use for broadcasting.

    Returns
    -------
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds.

    """
    try:
        lower = np.broadcast_to(l_bounds, d)
        upper = np.broadcast_to(u_bounds, d)
    except ValueError as exc:
        msg = ("'l_bounds' and 'u_bounds' must be broadcastable and respect"
               " the sample dimension")
        raise ValueError(msg) from exc

    if not np.all(lower < upper):
        raise ValueError("Bounds are not consistent 'l_bounds' < 'u_bounds'")

    return lower, upper
<|endoftext|># Many scipy.stats functions support `axis` and `nan_policy` parameters.
# When the two are combined, it can be tricky to get all the behavior just
# right. This file contains utility functions useful for scipy.stats functions
# that support `axis` and `nan_policy`, including a decorator that
# automatically adds `axis` and `nan_policy` arguments to a function.

import warnings
import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
from scipy._lib._array_api import array_namespace, is_numpy

import inspect

too_small_1d_not_omit = (
    "One or more sample arguments is too small; all "
    "returned values will be NaN. "
    "See documentation for sample size requirements.")

too_small_1d_omit = (
    "After omitting NaNs, one or more sample arguments "
    "is too small; all returned values will be NaN. "
    "See documentation for sample size requirements.")

too_small_nd_not_omit = (
    "All axis-slices of one or more sample arguments are "
    "too small; all elements of returned arrays will be NaN. "
    "See documentation for sample size requirements.")

too_small_nd_omit = (
    "After omitting NaNs, one or more axis-slices of one "
    "or more sample arguments is too small; corresponding "
    "elements of returned arrays will be NaN. "
    "See documentation for sample size requirements.")

class SmallSampleWarning(RuntimeWarning):
    pass


def _broadcast_arrays(arrays, axis=None, xp=None):
    """
    Broadcast shapes of arrays, ignoring incompatibility of specified axes
    """
    if not arrays:
        return arrays
    xp = array_namespace(*arrays) if xp is None else xp
    arrays = [xp.asarray(arr) for arr in arrays]
    shapes = [arr.shape for arr in arrays]
    new_shapes = _broadcast_shapes(shapes, axis)
    if axis is None:
        new_shapes = [new_shapes]*len(arrays)
    return [xp.broadcast_to(array, new_shape)
            for array, new_shape in zip(arrays, new_shapes)]


def _broadcast_shapes(shapes, axis=None):
    """
    Broadcast shapes, ignoring incompatibility of specified axes
    """
    if not shapes:
        return shapes

    # input validation
    if axis is not None:
        axis = np.atleast_1d(axis)
        message = '`axis` must be an integer, a tuple of integers, or `None`.'
        try:
            with np.errstate(invalid='ignore'):
                axis_int = axis.astype(int)
        except ValueError as e:
            raise AxisError(message) from e
        if not np.array_equal(axis_int, axis):
            raise AxisError(message)
        axis = axis_int

    # First, ensure all shapes have same number of dimensions by prepending 1s.
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for row, shape in zip(new_shapes, shapes):
        row[len(row)-len(shape):] = shape  # can't use negative indices (-0:)

    # Remove the shape elements of the axes to be ignored, but remember them.
    if axis is not None:
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = np.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = (f"`axis` is out of bounds "
                       f"for array of dimension {n_dims}")
            raise AxisError(message)

        if len(np.unique(axis)) != len(axis):
            raise AxisError("`axis` must contain only distinct elements")

        removed_shapes = new_shapes[:, axis]
        new_shapes = np.delete(new_shapes, axis, axis=1)

    # If arrays are broadcastable, shape elements that are 1 may be replaced
    # with a corresponding non-1 shape element. Assuming arrays are
    # broadcastable, that final shape element can be found with:
    new_shape = np.max(new_shapes, axis=0)
    # except in case of an empty array:
    new_shape *= new_shapes.all(axis=0)

    # Among all arrays, there can only be one unique non-1 shape element.
    # Therefore, if any non-1 shape element does not match what we found
    # above, the arrays must not be broadcastable after all.
    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError("Array shapes are incompatible for broadcasting.")

    if axis is not None:
        # Add back the shape elements that were ignored
        new_axis = axis - np.arange(len(axis))
        new_shapes = [tuple(np.insert(new_shape, new_axis, removed_shape))
                      for removed_shape in removed_shapes]
        return new_shapes
    else:
        return tuple(new_shape)


def _broadcast_array_shapes_remove_axis(arrays, axis=None):
    """
    Broadcast shapes of arrays, dropping specified axes

    Given a sequence of arrays `arrays` and an integer or tuple `axis`, find
    the shape of the broadcast result after consuming/dropping `axis`.
    In other words, return output shape of a typical hypothesis test on
    `arrays` vectorized along `axis`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats._axis_nan_policy import _broadcast_array_shapes_remove_axis
    >>> a = np.zeros((5, 2, 1))
    >>> b = np.zeros((9, 3))
    >>> _broadcast_array_shapes_remove_axis((a, b), 1)
    (5, 3)
    """
    # Note that here, `axis=None` means do not consume/drop any axes - _not_
    # ravel arrays before broadcasting.
    shapes = [arr.shape for arr in arrays]
    return _broadcast_shapes_remove_axis(shapes, axis)


def _broadcast_shapes_remove_axis(shapes, axis=None):
    """
    Broadcast shapes, dropping specified axes

    Same as _broadcast_array_shapes_remove_axis, but given a sequence
    of array shapes `shapes` instead of the arrays themselves.
    """
    shapes = _broadcast_shapes(shapes, axis)
    shape = shapes[0]
    if axis is not None:
        shape = np.delete(shape, axis)
    return tuple(shape)


def _broadcast_concatenate(arrays, axis, paired=False):
    """Concatenate arrays along an axis with broadcasting."""
    arrays = _broadcast_arrays(arrays, axis if not paired else None)
    res = np.concatenate(arrays, axis=axis)
    return res


# TODO: add support for `axis` tuples
def _remove_nans(samples, paired):
    "Remove nans from paired or unpaired 1D samples"
    # potential optimization: don't copy arrays that don't contain nans
    if not paired:
        return [sample[~np.isnan(sample)] for sample in samples]

    # for paired samples, we need to remove the whole pair when any part
    # has a nan
    nans = np.isnan(samples[0])
    for sample in samples[1:]:
        nans = nans | np.isnan(sample)
    not_nans = ~nans
    return [sample[not_nans] for sample in samples]


def _remove_sentinel(samples, paired, sentinel):
    "Remove sentinel values from paired or unpaired 1D samples"
    # could consolidate with `_remove_nans`, but it's not quite as simple as
    # passing `sentinel=np.nan` because `(np.nan == np.nan) is False`

    # potential optimization: don't copy arrays that don't contain sentinel
    if not paired:
        return [sample[sample != sentinel] for sample in samples]

    # for paired samples, we need to remove the whole pair when any part
    # has a nan
    sentinels = (samples[0] == sentinel)
    for sample in samples[1:]:
        sentinels = sentinels | (sample == sentinel)
    not_sentinels = ~sentinels
    return [sample[not_sentinels] for sample in samples]


def _masked_arrays_2_sentinel_arrays(samples):
    # masked arrays in `samples` are converted to regular arrays, and values
    # corresponding with masked elements are replaced with a sentinel value

    # return without modifying arrays if none have a mask
    has_mask = False
    for sample in samples:
        mask = getattr(sample, 'mask', False)
        has_mask = has_mask or np.any(mask)
    if not has_mask:
        return samples, None  # None means there is no sentinel value

    # Choose a sentinel value. We can't use `np.nan`, because sentinel (masked)
    # values are always omitted, but there are different nan policies.
    dtype = np.result_type(*samples)
    dtype = dtype if np.issubdtype(dtype, np.number) else np.float64
    for i in range(len(samples)):
        # Things get more complicated if the arrays are of different types.
        # We could have different sentinel values for each array, but
        # the purpose of this code is convenience, not efficiency.
        samples[i] = samples[i].astype(dtype, copy=False)

    inexact = np.issubdtype(dtype, np.inexact)
    info = np.finfo if inexact else np.iinfo
    max_possible, min_possible = info(dtype).max, info(dtype).min
    nextafter = np.nextafter if inexact else (lambda x, _: x - 1)

    sentinel = max_possible
    # For simplicity, min_possible/np.infs are not candidate sentinel values
    while sentinel > min_possible:
        for sample in samples:
            if np.any(sample == sentinel):  # choose a new sentinel value
                sentinel = nextafter(sentinel, -np.inf)
                break
        else:  # when sentinel value is OK, break the while loop
            break
    else:
        message = ("This function replaces masked elements with sentinel "
                   "values, but the data contains all distinct values of this "
                   "data type. Consider promoting the dtype to `np.float64`.")
        raise ValueError(message)

    # replace masked elements with sentinel value
    out_samples = []
    for sample in samples:
        mask = getattr(sample, 'mask', None)
        if mask is not None:  # turn all masked arrays into sentinel arrays
            mask = np.broadcast_to(mask, sample.shape)
            sample = sample.data.copy() if np.any(mask) else sample.data
            sample = np.asarray(sample)  # `sample.data` could be a memoryview?
            sample[mask] = sentinel
        out_samples.append(sample)

    return out_samples, sentinel


def _check_empty_inputs(samples, axis):
    """
    Check for empty sample; return appropriate output for a vectorized hypotest
    """
    # if none of the samples are empty, we need to perform the test
    if not any(sample.size == 0 for sample in samples):
        return None
    # otherwise, the statistic and p-value will be either empty arrays or
    # arrays with NaNs. Produce the appropriate array and return it.
    output_shape = _broadcast_array_shapes_remove_axis(samples, axis)
    output = np.ones(output_shape) * _get_nan(*samples)
    return output


def _add_reduced_axes(res, reduced_axes, keepdims):
    """
    Add reduced axes back to all the arrays in the result object
    if keepdims = True.
    """
    return ([np.expand_dims(output, reduced_axes) 
             if not isinstance(output, int) else output for output in res]
            if keepdims else res)


# Standard docstring / signature entries for `axis`, `nan_policy`, `keepdims`
_name = 'axis'
_desc = (
    """If an int, the axis of the input along which to compute the statistic.
The statistic of each axis-slice (e.g. row) of the input will appear in a
corresponding element of the output.
If ``None``, the input will be raveled before computing the statistic."""
    .split('\n'))


def _get_axis_params(default_axis=0, _name=_name, _desc=_desc):  # bind NOW
    _type = f"int or None, default: {default_axis}"
    _axis_parameter_doc = Parameter(_name, _type, _desc)
    _axis_parameter = inspect.Parameter(_name,
                                        inspect.Parameter.KEYWORD_ONLY,
                                        default=default_axis)
    return _axis_parameter_doc, _axis_parameter


_name = 'nan_policy'
_type = "{'propagate', 'omit', 'raise'}"
_desc = (
    """Defines how to handle input NaNs.

- ``propagate``: if a NaN is present in the axis slice (e.g. row) along
  which the  statistic is computed, the corresponding entry of the output
  will be NaN.
- ``omit``: NaNs will be omitted when performing the calculation.
  If insufficient data remains in the axis slice along which the
  statistic is computed, the corresponding entry of the output will be
  NaN.
- ``raise``: if a NaN is present, a ``ValueError`` will be raised."""
    .split('\n'))
_nan_policy_parameter_doc = Parameter(_name, _type, _desc)
_nan_policy_parameter = inspect.Parameter(_name,
                                          inspect.Parameter.KEYWORD_ONLY,
                                          default='propagate')

_name = 'keepdims'
_type = "bool, default: False"
_desc = (
    """If this is set to True, the axes which are reduced are left
in the result as dimensions with size one. With this option,
the result will broadcast correctly against the input array."""
    .split('\n'))
_keepdims_parameter_doc = Parameter(_name, _type, _desc)
_keepdims_parameter = inspect.Parameter(_name,
                                        inspect.Parameter.KEYWORD_ONLY,
                                        default=False)

_standard_note_addition = (
    """\nBeginning in SciPy 1.9, ``np.matrix`` inputs (not recommended for new
code) are converted to ``np.ndarray`` before the calculation is performed. In
this case, the output will be a scalar or ``np.ndarray`` of appropriate shape
rather than a 2D ``np.matrix``. Similarly, while masked elements of masked
arrays are ignored, the output will be a scalar or ``np.ndarray`` rather than a
masked array with ``mask=False``.""").split('\n')


def _axis_nan_policy_factory(tuple_to_result, default_axis=0,
                             n_samples=1, paired=False,
                             result_to_tuple=None, too_small=0,
                             n_outputs=2, kwd_samples=(), override=None):
    """Factory for a wrapper that adds axis/nan_policy params to a function.

    Parameters
    ----------
    tuple_to_result : callable
        Callable that returns an object of the type returned by the function
        being wrapped (e.g. the namedtuple or dataclass returned by a
        statistical test) provided the separate components (e.g. statistic,
        pvalue).
    default_axis : int, default: 0
        The default value of the axis argument. Standard is 0 except when
        backwards compatibility demands otherwise (e.g. `None`).
    n_samples : int or callable, default: 1
        The number of data samples accepted by the function
        (e.g. `mannwhitneyu`), a callable that accepts a dictionary of
        parameters passed into the function and returns the number of data
        samples (e.g. `wilcoxon`), or `None` to indicate an arbitrary number
        of samples (e.g. `kruskal`).
    paired : {False, True}
        Whether the function being wrapped treats the samples as paired (i.e.
        corresponding elements of each sample should be considered as different
        components of the same sample.)
    result_to_tuple : callable, optional
        Function that unpacks the results of the function being wrapped into
        a tuple. This is essentially the inverse of `tuple_to_result`. Default
        is `None`, which is appropriate for statistical tests that return a
        statistic, pvalue tuple (rather than, e.g., a non-iterable datalass).
    too_small : int or callable, default: 0
        The largest unnacceptably small sample for the function being wrapped.
        For example, some functions require samples of size two or more or they
        raise an error. This argument prevents the error from being raised when
        input is not 1D and instead places a NaN in the corresponding element
        of the result. If callable, it must accept a list of samples, axis,
        and a dictionary of keyword arguments passed to the wrapper function as
        arguments and return a bool indicating weather the samples passed are
        too small.
    n_outputs : int or callable, default: 2
        The number of outputs produced by the function given 1d sample(s). For
        example, hypothesis tests that return a namedtuple or result object
        with attributes ``statistic`` and ``pvalue`` use the default
        ``n_outputs=2``; summary statistics with scalar output use
        ``n_outputs=1``. Alternatively, may be a callable that accepts a
        dictionary of arguments passed into the wrapped function and returns
        the number of outputs corresponding with those arguments.
    kwd_samples : sequence, default: ()
        The names of keyword parameters that should be treated as samples. For
        example, `gmean` accepts as its first argument a sample `a` but
        also `weights` as a fourth, optional keyword argument. In this case, we
        use `n_samples=1` and kwd_samples=['weights'].
    override : dict, default: {'vectorization': False, 'nan_propagation': True}
        Pass a dictionary with ``'vectorization': True`` to ensure that the
        decorator overrides the function's behavior for multimensional input.
        Use ``'nan_propagation': False`` to ensure that the decorator does not
        override the function's behavior for ``nan_policy='propagate'``.
    """
    # Specify which existing behaviors the decorator must override
    temp = override or {}
    override = {'vectorization': False,
                'nan_propagation': True}
    override.update(temp)

    if result_to_tuple is None:
        def result_to_tuple(res):
            return res

    # The only `result_to_tuple` that needs the second argument (number of
    # outputs) is the one for `moment`, and this was realized very late.
    # Rather than changing all `result_to_tuple` definitions, we wrap them
    # here to accept a second argument if they don't already.
    if len(inspect.signature(result_to_tuple).parameters) == 1:
        def result_to_tuple(res, _, f=result_to_tuple):
            return f(res)

    if not callable(too_small):
        def is_too_small(samples, *ts_args, axis=-1, **ts_kwargs):
            for sample in samples:
                if sample.shape[axis] <= too_small:
                    return True
            return False
    else:
        is_too_small = too_small

    def axis_nan_policy_decorator(hypotest_fun_in):
        @wraps(hypotest_fun_in)
        def axis_nan_policy_wrapper(*args, _no_deco=False, **kwds):

            if _no_deco:  # for testing, decorator does nothing
                return hypotest_fun_in(*args, **kwds)

            # For now, skip the decorator entirely if using array API. In the future,
            # we'll probably want to use it for `keepdims`, `axis` tuples, etc.
            if len(args) == 0:  # extract sample from `kwds` if there are no `args`
                used_kwd_samples = list(set(kwds).intersection(set(kwd_samples)))
                temp = used_kwd_samples[:1]
            else:
                temp = args[0]

            if not is_numpy(array_namespace(temp)):
                msg = ("Use of `nan_policy` and `keepdims` "
                       "is incompatible with non-NumPy arrays.")
                if 'nan_policy' in kwds or 'keepdims' in kwds:
                    raise NotImplementedError(msg)
                return hypotest_fun_in(*args, **kwds)

            # We need to be flexible about whether position or keyword
            # arguments are used, but we need to make sure users don't pass
            # both for the same parameter. To complicate matters, some
            # functions accept samples with *args, and some functions already
            # accept `axis` and `nan_policy` as positional arguments.
            # The strategy is to make sure that there is no duplication
            # between `args` and `kwds`, combine the two into `kwds`, then
            # the samples, `nan_policy`, and `axis` from `kwds`, as they are
            # dealt with separately.

            # Check for intersection between positional and keyword args
            params = list(inspect.signature(hypotest_fun_in).parameters)
            if n_samples is None:
                # Give unique names to each positional sample argument
                # Note that *args can't be provided as a keyword argument
                params = [f"arg{i}" for i in range(len(args))] + params[1:]

            # raise if there are too many positional args
            maxarg = (np.inf if inspect.getfullargspec(hypotest_fun_in).varargs
                      else len(inspect.getfullargspec(hypotest_fun_in).args))
            if len(args) > maxarg:  # let the function raise the right error
                hypotest_fun_in(*args, **kwds)

            # raise if multiple values passed for same parameter
            d_args = dict(zip(params, args))
            intersection = set(d_args) & set(kwds)
            if intersection:  # let the function raise the right error
                hypotest_fun_in(*args, **kwds)

            # Consolidate other positional and keyword args into `kwds`
            kwds.update(d_args)

            # rename avoids UnboundLocalError
            if callable(n_samples):
                # Future refactoring idea: no need for callable n_samples.
                # Just replace `n_samples` and `kwd_samples` with a single
                # list of the names of all samples, and treat all of them
                # as `kwd_samples` are treated below.
                n_samp = n_samples(kwds)
            else:
                n_samp = n_samples or len(args)

            # get the number of outputs
            n_out = n_outputs  # rename to avoid UnboundLocalError
            if callable(n_out):
                n_out = n_out(kwds)

            # If necessary, rearrange function signature: accept other samples
            # as positional args right after the first n_samp args
            kwd_samp = [name for name in kwd_samples
                        if kwds.get(name, None) is not None]
            n_kwd_samp = len(kwd_samp)
            if not kwd_samp:
                hypotest_fun_out = hypotest_fun_in
            else:
                def hypotest_fun_out(*samples, **kwds):
                    new_kwds = dict(zip(kwd_samp, samples[n_samp:]))
                    kwds.update(new_kwds)
                    return hypotest_fun_in(*samples[:n_samp], **kwds)

            # Extract the things we need here
            try:  # if something is missing
                samples = [np.atleast_1d(kwds.pop(param))
                           for param in (params[:n_samp] + kwd_samp)]
            except KeyError:  # let the function raise the right error
                # might need to revisit this if required arg is not a "sample"
                hypotest_fun_in(*args, **kwds)
            vectorized = True if 'axis' in params else False
            vectorized = vectorized and not override['vectorization']
            axis = kwds.pop('axis', default_axis)
            nan_policy = kwds.pop('nan_policy', 'propagate')
            keepdims = kwds.pop("keepdims", False)
            del args  # avoid the possibility of passing both `args` and `kwds`

            # convert masked arrays to regular arrays with sentinel values
            samples, sentinel = _masked_arrays_2_sentinel_arrays(samples)

            # standardize to always work along last axis
            reduced_axes = axis
            if axis is None:
                if samples:
                    # when axis=None, take the maximum of all dimensions since
                    # all the dimensions are reduced.
                    n_dims = np.max([sample.ndim for sample in samples])
                    reduced_axes = tuple(range(n_dims))
                samples = [np.asarray(sample.ravel()) for sample in samples]
            else:
                # don't ignore any axes when broadcasting if paired
                samples = _broadcast_arrays(samples, axis=axis if not paired else None)
                axis = np.atleast_1d(axis)
                n_axes = len(axis)
                # move all axes in `axis` to the end to be raveled
                samples = [np.moveaxis(sample, axis, range(-len(axis), 0))
                           for sample in samples]
                shapes = [sample.shape for sample in samples]
                # New shape is unchanged for all axes _not_ in `axis`
                # At the end, we append the product of the shapes of the axes
                # in `axis`. Appending -1 doesn't work for zero-size arrays!
                new_shapes = [shape[:-n_axes] + (np.prod(shape[-n_axes:]),)
                              for shape in shapes]
                samples = [sample.reshape(new_shape)
                           for sample, new_shape in zip(samples, new_shapes)]
            axis = -1  # work over the last axis
            NaN = _get_nan(*samples) if samples else np.nan

            # if axis is not needed, just handle nan_policy and return
            ndims = np.array([sample.ndim for sample in samples])
            if np.all(ndims <= 1):
                # Addresses nan_policy == "raise"
                if nan_policy != 'propagate' or override['nan_propagation']:
                    contains_nan = [_contains_nan(sample, nan_policy)[0]
                                    for sample in samples]
                else:
                    # Behave as though there are no NaNs (even if there are)
                    contains_nan = [False]*len(samples)

                # Addresses nan_policy == "propagate"
                if any(contains_nan) and (nan_policy == 'propagate'
                                          and override['nan_propagation']):
                    res = np.full(n_out, NaN)
                    res = _add_reduced_axes(res, reduced_axes, keepdims)
                    return tuple_to_result(*res)

                # Addresses nan_policy == "omit"
                too_small_msg = too_small_1d_not_omit
                if any(contains_nan) and nan_policy == 'omit':
                    # consider passing in contains_nan
                    samples = _remove_nans(samples, paired)
                    too_small_msg = too_small_1d_omit

                if sentinel:
                    samples = _remove_sentinel(samples, paired, sentinel)

                if is_too_small(samples, kwds):
                    warnings.warn(too_small_msg, SmallSampleWarning, stacklevel=2)
                    res = np.full(n_out, NaN)
                    res = _add_reduced_axes(res, reduced_axes, keepdims)
                    return tuple_to_result(*res)

                res = hypotest_fun_out(*samples, **kwds)
                res = result_to_tuple(res, n_out)
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)

            # check for empty input
            empty_output = _check_empty_inputs(samples, axis)
            # only return empty output if zero sized input is too small.
            if (
                empty_output is not None
                and (is_too_small(samples, kwds) or empty_output.size == 0)
            ):
                if is_too_small(samples, kwds) and empty_output.size != 0:
                    warnings.warn(too_small_nd_not_omit, SmallSampleWarning,
                                  stacklevel=2)
                res = [empty_output.copy() for i in range(n_out)]
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)

            # otherwise, concatenate all samples along axis, remembering where
            # each separate sample begins
            lengths = np.array([sample.shape[axis] for sample in samples])
            split_indices = np.cumsum(lengths)
            x = _broadcast_concatenate(samples, axis, paired=paired)

            # Addresses nan_policy == "raise"
            if nan_policy != 'propagate' or override['nan_propagation']:
                contains_nan, _ = _contains_nan(x, nan_policy)
            else:
                contains_nan = False  # behave like there are no NaNs

            if vectorized and not contains_nan and not sentinel:
                res = hypotest_fun_out(*samples, axis=axis, **kwds)
                res = result_to_tuple(res, n_out)
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)

            # Addresses nan_policy == "omit"
            if contains_nan and nan_policy == 'omit':
                def hypotest_fun(x):
                    samples = np.split(x, split_indices)[:n_samp+n_kwd_samp]
                    samples = _remove_nans(samples, paired)
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples, kwds):
                        warnings.warn(too_small_nd_omit, SmallSampleWarning,
                                      stacklevel=4)
                        return np.full(n_out, NaN)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds), n_out)

            # Addresses nan_policy == "propagate"
            elif (contains_nan and nan_policy == 'propagate'
                  and override['nan_propagation']):
                def hypotest_fun(x):
                    if np.isnan(x).any():
                        return np.full(n_out, NaN)

                    samples = np.split(x, split_indices)[:n_samp+n_kwd_samp]
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples, kwds):
                        return np.full(n_out, NaN)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds), n_out)

            else:
                def hypotest_fun(x):
                    samples = np.split(x, split_indices)[:n_samp+n_kwd_samp]
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples, kwds):
                        return np.full(n_out, NaN)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds), n_out)

            x = np.moveaxis(x, axis, 0)
            res = np.apply_along_axis(hypotest_fun, axis=0, arr=x)
            res = _add_reduced_axes(res, reduced_axes, keepdims)
            return tuple_to_result(*res)

        _axis_parameter_doc, _axis_parameter = _get_axis_params(default_axis)
        doc = FunctionDoc(axis_nan_policy_wrapper)
        parameter_names = [param.name for param in doc['Parameters']]
        if 'axis' in parameter_names:
            doc['Parameters'][parameter_names.index('axis')] = (
                _axis_parameter_doc)
        else:
            doc['Parameters'].append(_axis_parameter_doc)
        if 'nan_policy' in parameter_names:
            doc['Parameters'][parameter_names.index('nan_policy')] = (
                _nan_policy_parameter_doc)
        else:
            doc['Parameters'].append(_nan_policy_parameter_doc)
        if 'keepdims' in parameter_names:
            doc['Parameters'][parameter_names.index('keepdims')] = (
                _keepdims_parameter_doc)
        else:
            doc['Parameters'].append(_keepdims_parameter_doc)
        doc['Notes'] += _standard_note_addition
        doc = str(doc).split("\n", 1)[1]  # remove signature
        axis_nan_policy_wrapper.__doc__ = str(doc)

        sig = inspect.signature(axis_nan_policy_wrapper)
        parameters = sig.parameters
        parameter_list = list(parameters.values())
        if 'axis' not in parameters:
            parameter_list.append(_axis_parameter)
        if 'nan_policy' not in parameters:
            parameter_list.append(_nan_policy_parameter)
        if 'keepdims' not in parameters:
            parameter_list.append(_keepdims_parameter)
        sig = sig.replace(parameters=parameter_list)
        axis_nan_policy_wrapper.__signature__ = sig

        return axis_nan_policy_wrapper
    return axis_nan_policy_decorator
<|endoftext|>import builtins
from warnings import catch_warnings, simplefilter
import numpy as np
from operator import index
from collections import namedtuple

__all__ = ['binned_statistic',
           'binned_statistic_2d',
           'binned_statistic_dd']


BinnedStatisticResult = namedtuple('BinnedStatisticResult',
                                   ('statistic', 'bin_edges', 'binnumber'))


def binned_statistic(x, values, statistic='mean',
                     bins=10, range=None):
    """
    Compute a binned statistic for one or more sets of data.

    This is a generalization of a histogram function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values (or set of values) within each bin.

    Parameters
    ----------
    x : (N,) array_like
        A sequence of values to be binned.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `x`, or a set of sequences - each the same shape as
        `x`.  If `values` is a set of sequences, the statistic will be computed
        on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.

    bins : int or sequence of scalars, optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10 by default).  If `bins` is a sequence, it defines the
        bin edges, including the rightmost edge, allowing for non-uniform bin
        widths.  Values in `x` that are smaller than lowest bin edge are
        assigned to bin number 0, values beyond the highest bin are assigned to
        ``bins[-1]``.  If the bin edges are specified, the number of bins will
        be, (nx = len(bins)-1).
    range : (float, float) or [(float, float)], optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(x.min(), x.max())``.  Values outside the range are
        ignored.

    Returns
    -------
    statistic : array
        The values of the selected statistic in each bin.
    bin_edges : array of dtype float
        Return the bin edges ``(length(statistic)+1)``.
    binnumber: 1-D ndarray of ints
        Indices of the bins (corresponding to `bin_edges`) in which each value
        of `x` belongs.  Same length as `values`.  A binnumber of `i` means the
        corresponding value is between (bin_edges[i-1], bin_edges[i]).

    See Also
    --------
    numpy.digitize, numpy.histogram, binned_statistic_2d, binned_statistic_dd

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words, if
    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,
    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is
    ``[3, 4]``, which *includes* 4.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    First some basic examples:

    Create two evenly spaced bins in the range of the given sample, and sum the
    corresponding values in each of those bins:

    >>> values = [1.0, 1.0, 2.0, 1.5, 3.0]
    >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)
    BinnedStatisticResult(statistic=array([4. , 4.5]),
            bin_edges=array([1., 4., 7.]), binnumber=array([1, 1, 1, 2, 2]))

    Multiple arrays of values can also be passed.  The statistic is calculated
    on each set independently:

    >>> values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]
    >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)
    BinnedStatisticResult(statistic=array([[4. , 4.5],
           [8. , 9. ]]), bin_edges=array([1., 4., 7.]),
           binnumber=array([1, 1, 1, 2, 2]))

    >>> stats.binned_statistic([1, 2, 1, 2, 4], np.arange(5), statistic='mean',
    ...                        bins=3)
    BinnedStatisticResult(statistic=array([1., 2., 4.]),
            bin_edges=array([1., 2., 3., 4.]),
            binnumber=array([1, 2, 1, 2, 3]))

    As a second example, we now generate some random data of sailing boat speed
    as a function of wind speed, and then determine how fast our boat is for
    certain wind speeds:

    >>> rng = np.random.default_rng()
    >>> windspeed = 8 * rng.random(500)
    >>> boatspeed = .3 * windspeed**.5 + .2 * rng.random(500)
    >>> bin_means, bin_edges, binnumber = stats.binned_statistic(windspeed,
    ...                 boatspeed, statistic='median', bins=[1,2,3,4,5,6,7])
    >>> plt.figure()
    >>> plt.plot(windspeed, boatspeed, 'b.', label='raw data')
    >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
    ...            label='binned statistic of data')
    >>> plt.legend()

    Now we can use ``binnumber`` to select all datapoints with a windspeed
    below 1:

    >>> low_boatspeed = boatspeed[binnumber == 0]

    As a final example, we will use ``bin_edges`` and ``binnumber`` to make a
    plot of a distribution that shows the mean and distribution around that
    mean per bin, on top of a regular histogram and the probability
    distribution function:

    >>> x = np.linspace(0, 5, num=500)
    >>> x_pdf = stats.maxwell.pdf(x)
    >>> samples = stats.maxwell.rvs(size=10000)

    >>> bin_means, bin_edges, binnumber = stats.binned_statistic(x, x_pdf,
    ...         statistic='mean', bins=25)
    >>> bin_width = (bin_edges[1] - bin_edges[0])
    >>> bin_centers = bin_edges[1:] - bin_width/2

    >>> plt.figure()
    >>> plt.hist(samples, bins=50, density=True, histtype='stepfilled',
    ...          alpha=0.2, label='histogram of data')
    >>> plt.plot(x, x_pdf, 'r-', label='analytical pdf')
    >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
    ...            label='binned statistic of data')
    >>> plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)
    >>> plt.legend(fontsize=10)
    >>> plt.show()

    """
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1:
        bins = [np.asarray(bins, float)]

    if range is not None:
        if len(range) == 2:
            range = [range]

    medians, edges, binnumbers = binned_statistic_dd(
        [x], values, statistic, bins, range)

    return BinnedStatisticResult(medians, edges[0], binnumbers)


BinnedStatistic2dResult = namedtuple('BinnedStatistic2dResult',
                                     ('statistic', 'x_edge', 'y_edge',
                                      'binnumber'))


def binned_statistic_2d(x, y, values, statistic='mean',
                        bins=10, range=None, expand_binnumbers=False):
    """
    Compute a bidimensional binned statistic for one or more sets of data.

    This is a generalization of a histogram2d function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values (or set of values) within each bin.

    Parameters
    ----------
    x : (N,) array_like
        A sequence of values to be binned along the first dimension.
    y : (N,) array_like
        A sequence of values to be binned along the second dimension.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `x`, or a list of sequences - each with the same
        shape as `x`.  If `values` is such a list, the statistic will be
        computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.

    bins : int or [int, int] or array_like or [array, array], optional
        The bin specification:

          * the number of bins for the two dimensions (nx = ny = bins),
          * the number of bins in each dimension (nx, ny = bins),
          * the bin edges for the two dimensions (x_edge = y_edge = bins),
          * the bin edges in each dimension (x_edge, y_edge = bins).

        If the bin edges are specified, the number of bins will be,
        (nx = len(x_edge)-1, ny = len(y_edge)-1).

    range : (2,2) array_like, optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
        considered outliers and not tallied in the histogram.
    expand_binnumbers : bool, optional
        'False' (default): the returned `binnumber` is a shape (N,) array of
        linearized bin indices.
        'True': the returned `binnumber` is 'unraveled' into a shape (2,N)
        ndarray, where each row gives the bin numbers in the corresponding
        dimension.
        See the `binnumber` returned value, and the `Examples` section.

        .. versionadded:: 0.17.0

    Returns
    -------
    statistic : (nx, ny) ndarray
        The values of the selected statistic in each two-dimensional bin.
    x_edge : (nx + 1) ndarray
        The bin edges along the first dimension.
    y_edge : (ny + 1) ndarray
        The bin edges along the second dimension.
    binnumber : (N,) array of ints or (2,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.


    See Also
    --------
    numpy.digitize, numpy.histogram2d, binned_statistic, binned_statistic_dd

    Notes
    -----
    Binedges:
    All but the last (righthand-most) bin is half-open.  In other words, if
    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,
    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is
    ``[3, 4]``, which *includes* 4.

    `binnumber`:
    This returned argument assigns to each element of `sample` an integer that
    represents the bin in which it belongs.  The representation depends on the
    `expand_binnumbers` argument. If 'False' (default): The returned
    `binnumber` is a shape (N,) array of linearized indices mapping each
    element of `sample` to its corresponding bin (using row-major ordering).
    Note that the returned linearized bin indices are used for an array with
    extra bins on the outer binedges to capture values outside of the defined
    bin bounds.
    If 'True': The returned `binnumber` is a shape (2,N) ndarray where
    each row indicates bin placements for each dimension respectively.  In each
    dimension, a binnumber of `i` means the corresponding value is between
    (D_edge[i-1], D_edge[i]), where 'D' is either 'x' or 'y'.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy import stats

    Calculate the counts with explicit bin-edges:

    >>> x = [0.1, 0.1, 0.1, 0.6]
    >>> y = [2.1, 2.6, 2.1, 2.1]
    >>> binx = [0.0, 0.5, 1.0]
    >>> biny = [2.0, 2.5, 3.0]
    >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])
    >>> ret.statistic
    array([[2., 1.],
           [1., 0.]])

    The bin in which each sample is placed is given by the `binnumber`
    returned parameter.  By default, these are the linearized bin indices:

    >>> ret.binnumber
    array([5, 6, 5, 9])

    The bin indices can also be expanded into separate entries for each
    dimension using the `expand_binnumbers` parameter:

    >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny],
    ...                                 expand_binnumbers=True)
    >>> ret.binnumber
    array([[1, 1, 1, 2],
           [1, 2, 1, 1]])

    Which shows that the first three elements belong in the xbin 1, and the
    fourth into xbin 2; and so on for y.

    """

    # This code is based on np.histogram2d
    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        xedges = yedges = np.asarray(bins, float)
        bins = [xedges, yedges]

    medians, edges, binnumbers = binned_statistic_dd(
        [x, y], values, statistic, bins, range,
        expand_binnumbers=expand_binnumbers)

    return BinnedStatistic2dResult(medians, edges[0], edges[1], binnumbers)


BinnedStatisticddResult = namedtuple('BinnedStatisticddResult',
                                     ('statistic', 'bin_edges',
                                      'binnumber'))


def _bincount(x, weights):
    if np.iscomplexobj(weights):
        a = np.bincount(x, np.real(weights))
        b = np.bincount(x, np.imag(weights))
        z = a + b*1j

    else:
        z = np.bincount(x, weights)
    return z


def binned_statistic_dd(sample, values, statistic='mean',
                        bins=10, range=None, expand_binnumbers=False,
                        binned_statistic_result=None):
    """
    Compute a multidimensional binned statistic for a set of data.

    This is a generalization of a histogramdd function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values within each bin.

    Parameters
    ----------
    sample : array_like
        Data to histogram passed as a sequence of N arrays of length D, or
        as an (N,D) array.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `sample`, or a list of sequences - each with the
        same shape as `sample`.  If `values` is such a list, the statistic
        will be computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0. If the number of values
            within a given bin is 0 or 1, the computed standard deviation value
            will be 0 for the bin.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.

    bins : sequence or positive int, optional
        The bin specification must be in one of the following forms:

          * A sequence of arrays describing the bin edges along each dimension.
          * The number of bins for each dimension (nx, ny, ... = bins).
          * The number of bins for all dimensions (nx = ny = ... = bins).
    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitly in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    expand_binnumbers : bool, optional
        'False' (default): the returned `binnumber` is a shape (N,) array of
        linearized bin indices.
        'True': the returned `binnumber` is 'unraveled' into a shape (D,N)
        ndarray, where each row gives the bin numbers in the corresponding
        dimension.
        See the `binnumber` returned value, and the `Examples` section of
        `binned_statistic_2d`.
    binned_statistic_result : binnedStatisticddResult
        Result of a previous call to the function in order to reuse bin edges
        and bin numbers with new values and/or a different statistic.
        To reuse bin numbers, `expand_binnumbers` must have been set to False
        (the default)

        .. versionadded:: 0.17.0

    Returns
    -------
    statistic : ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin.
    bin_edges : list of ndarrays
        A list of D arrays describing the (nxi + 1) bin edges for each
        dimension.
    binnumber : (N,) array of ints or (D,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.


    See Also
    --------
    numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d

    Notes
    -----
    Binedges:
    All but the last (righthand-most) bin is half-open in each dimension.  In
    other words, if `bins` is ``[1, 2, 3, 4]``, then the first bin is
    ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The
    last bin, however, is ``[3, 4]``, which *includes* 4.

    `binnumber`:
    This returned argument assigns to each element of `sample` an integer that
    represents the bin in which it belongs.  The representation depends on the
    `expand_binnumbers` argument. If 'False' (default): The returned
    `binnumber` is a shape (N,) array of linearized indices mapping each
    element of `sample` to its corresponding bin (using row-major ordering).
    If 'True': The returned `binnumber` is a shape (D,N) ndarray where
    each row indicates bin placements for each dimension respectively.  In each
    dimension, a binnumber of `i` means the corresponding value is between
    (bin_edges[D][i-1], bin_edges[D][i]), for each dimension 'D'.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D

    Take an array of 600 (x, y) coordinates as an example.
    `binned_statistic_dd` can handle arrays of higher dimension `D`. But a plot
    of dimension `D+1` is required.

    >>> mu = np.array([0., 1.])
    >>> sigma = np.array([[1., -0.5],[-0.5, 1.5]])
    >>> multinormal = stats.multivariate_normal(mu, sigma)
    >>> data = multinormal.rvs(size=600, random_state=235412)
    >>> data.shape
    (600, 2)

    Create bins and count how many arrays fall in each bin:

    >>> N = 60
    >>> x = np.linspace(-3, 3, N)
    >>> y = np.linspace(-3, 4, N)
    >>> ret = stats.binned_statistic_dd(data, np.arange(600), bins=[x, y],
    ...                                 statistic='count')
    >>> bincounts = ret.statistic

    Set the volume and the location of bars:

    >>> dx = x[1] - x[0]
    >>> dy = y[1] - y[0]
    >>> x, y = np.meshgrid(x[:-1]+dx/2, y[:-1]+dy/2)
    >>> z = 0

    >>> bincounts = bincounts.ravel()
    >>> x = x.ravel()
    >>> y = y.ravel()

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> with np.errstate(divide='ignore'):   # silence random axes3d warning
    ...     ax.bar3d(x, y, z, dx, dy, bincounts)

    Reuse bin numbers and bin edges with new values:

    >>> ret2 = stats.binned_statistic_dd(data, -np.arange(600),
    ...                                  binned_statistic_result=ret,
    ...                                  statistic='mean')
    """
    known_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max']
    if not callable(statistic) and statistic not in known_stats:
        raise ValueError(f'invalid statistic {statistic!r}')

    try:
        bins = index(bins)
    except TypeError:
        # bins is not an integer
        pass
    # If bins was an integer-like object, now it is an actual Python int.

    # NOTE: for _bin_edges(), see e.g. gh-11365
    if isinstance(bins, int) and not np.isfinite(sample).all():
        raise ValueError(f'{sample!r} contains non-finite values.')

    # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
    # `Dlen` is the length of elements along each dimension.
    # This code is based on np.histogramdd
    try:
        # `sample` is an ND-array.
        Dlen, Ndim = sample.shape
    except (AttributeError, ValueError):
        # `sample` is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        Dlen, Ndim = sample.shape

    # Store initial shape of `values` to preserve it in the output
    values = np.asarray(values)
    input_shape = list(values.shape)
    # Make sure that `values` is 2D to iterate over rows
    values = np.atleast_2d(values)
    Vdim, Vlen = values.shape

    # Make sure `values` match `sample`
    if statistic != 'count' and Vlen != Dlen:
        raise AttributeError('The number of `values` elements must match the '
                             'length of each `sample` dimension.')

    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = Ndim * [bins]

    if binned_statistic_result is None:
        nbin, edges, dedges = _bin_edges(sample, bins, range)
        binnumbers = _bin_numbers(sample, nbin, edges, dedges)
    else:
        edges = binned_statistic_result.bin_edges
        nbin = np.array([len(edges[i]) + 1 for i in builtins.range(Ndim)])
        # +1 for outlier bins
        dedges = [np.diff(edges[i]) for i in builtins.range(Ndim)]
        binnumbers = binned_statistic_result.binnumber

    # Avoid overflow with double precision. Complex `values` -> `complex128`.
    result_type = np.result_type(values, np.float64)
    result = np.empty([Vdim, nbin.prod()], dtype=result_type)

    if statistic in {'mean', np.mean}:
        result.fill(np.nan)
        flatcount = _bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            flatsum = _bincount(binnumbers, values[vv])
            result[vv, a] = flatsum[a] / flatcount[a]
    elif statistic in {'std', np.std}:
        result.fill(np.nan)
        flatcount = _bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            flatsum = _bincount(binnumbers, values[vv])
            delta = values[vv] - flatsum[binnumbers] / flatcount[binnumbers]
            std = np.sqrt(
                _bincount(binnumbers, delta*np.conj(delta))[a] / flatcount[a]
            )
            result[vv, a] = std
        result = np.real(result)
    elif statistic == 'count':
        result = np.empty([Vdim, nbin.prod()], dtype=np.float64)
        result.fill(0)
        flatcount = _bincount(binnumbers, None)
        a = np.arange(len(flatcount))
        result[:, a] = flatcount[np.newaxis, :]
    elif statistic in {'sum', np.sum}:
        result.fill(0)
        for vv in builtins.range(Vdim):
            flatsum = _bincount(binnumbers, values[vv])
            a = np.arange(len(flatsum))
            result[vv, a] = flatsum
    elif statistic in {'median', np.median}:
        result.fill(np.nan)
        for vv in builtins.range(Vdim):
            i = np.lexsort((values[vv], binnumbers))
            _, j, counts = np.unique(binnumbers[i],
                                     return_index=True, return_counts=True)
            mid = j + (counts - 1) / 2
            mid_a = values[vv, i][np.floor(mid).astype(int)]
            mid_b = values[vv, i][np.ceil(mid).astype(int)]
            medians = (mid_a + mid_b) / 2
            result[vv, binnumbers[i][j]] = medians
    elif statistic in {'min', np.min}:
        result.fill(np.nan)
        for vv in builtins.range(Vdim):
            i = np.argsort(values[vv])[::-1]  # Reversed so the min is last
            result[vv, binnumbers[i]] = values[vv, i]
    elif statistic in {'max', np.max}:
        result.fill(np.nan)
        for vv in builtins.range(Vdim):
            i = np.argsort(values[vv])
            result[vv, binnumbers[i]] = values[vv, i]
    elif callable(statistic):
        with np.errstate(invalid='ignore'), catch_warnings():
            simplefilter("ignore", RuntimeWarning)
            try:
                null = statistic([])
            except Exception:
                null = np.nan
        if np.iscomplexobj(null):
            result = result.astype(np.complex128)
        result.fill(null)
        try:
            _calc_binned_statistic(
                Vdim, binnumbers, result, values, statistic
            )
        except ValueError:
            result = result.astype(np.complex128)
            _calc_binned_statistic(
                Vdim, binnumbers, result, values, statistic
            )

    # Shape into a proper matrix
    result = result.reshape(np.append(Vdim, nbin))

    # Remove outliers (indices 0 and -1 for each bin-dimension).
    core = tuple([slice(None)] + Ndim * [slice(1, -1)])
    result = result[core]

    # Unravel binnumbers into an ndarray, each row the bins for each dimension
    if expand_binnumbers and Ndim > 1:
        binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))

    if np.any(result.shape[1:] != nbin - 2):
        raise RuntimeError('Internal Shape Error')

    # Reshape to have output (`result`) match input (`values`) shape
    result = result.reshape(input_shape[:-1] + list(nbin-2))

    return BinnedStatisticddResult(result, edges, binnumbers)


def _calc_binned_statistic(Vdim, bin_numbers, result, values, stat_func):
    unique_bin_numbers = np.unique(bin_numbers)
    for vv in builtins.range(Vdim):
        bin_map = _create_binned_data(bin_numbers, unique_bin_numbers,
                                      values, vv)
        for i in unique_bin_numbers:
            stat = stat_func(np.array(bin_map[i]))
            if np.iscomplexobj(stat) and not np.iscomplexobj(result):
                raise ValueError("The statistic function returns complex ")
            result[vv, i] = stat


def _create_binned_data(bin_numbers, unique_bin_numbers, values, vv):
    """ Create hashmap of bin ids to values in bins
    key: bin number
    value: list of binned data
    """
    bin_map = dict()
    for i in unique_bin_numbers:
        bin_map[i] = []
    for i in builtins.range(len(bin_numbers)):
        bin_map[bin_numbers[i]].append(values[vv, i])
    return bin_map


def _bin_edges(sample, bins=None, range=None):
    """ Create edge arrays
    """
    Dlen, Ndim = sample.shape

    nbin = np.empty(Ndim, int)    # Number of bins in each dimension
    edges = Ndim * [None]         # Bin edges for each dim (will be 2D array)
    dedges = Ndim * [None]        # Spacing between edges (will be 2D array)

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        if len(range) != Ndim:
            raise ValueError(
                f"range given for {len(range)} dimensions; {Ndim} required")
        smin = np.empty(Ndim)
        smax = np.empty(Ndim)
        for i in builtins.range(Ndim):
            if range[i][1] < range[i][0]:
                raise ValueError(
                    f"In {f'dimension {i + 1} of ' if Ndim > 1 else ''}range,"
                    " start must be <= stop")
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in builtins.range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Preserve sample floating point precision in bin edges
    edges_dtype = (sample.dtype if np.issubdtype(sample.dtype, np.floating)
                   else float)

    # Create edge arrays
    for i in builtins.range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1,
                                   dtype=edges_dtype)
        else:
            edges[i] = np.asarray(bins[i], edges_dtype)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    return nbin, edges, dedges


def _bin_numbers(sample, nbin, edges, dedges):
    """Compute the bin number each sample falls into, in each dimension
    """
    Dlen, Ndim = sample.shape

    sampBin = [
        np.digitize(sample[:, i], edges[i])
        for i in range(Ndim)
    ]

    # Using `digitize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in range(Ndim):
        # Find the rounding precision
        dedges_min = dedges[i].min()
        if dedges_min == 0:
            raise ValueError('The smallest edge difference is numerically 0.')
        decimal = int(-np.log10(dedges_min)) + 6
        # Find which points are on the rightmost edge.
        on_edge = np.where((sample[:, i] >= edges[i][-1]) &
                           (np.around(sample[:, i], decimal) ==
                            np.around(edges[i][-1], decimal)))[0]
        # Shift these points one bin to the left.
        sampBin[i][on_edge] -= 1

    # Compute the sample indices in the flattened statistic matrix.
    binnumbers = np.ravel_multi_index(sampBin, nbin)

    return binnumbers
<|endoftext|>"""
======================================================
Random Number Generators (:mod:`scipy.stats.sampling`)
======================================================

.. currentmodule:: scipy.stats.sampling

This module contains a collection of random number generators to sample
from univariate continuous and discrete distributions. It uses the
implementation of a C library called "UNU.RAN". The only exception is
RatioUniforms, which is a pure Python implementation of the
Ratio-of-Uniforms method.

Generators Wrapped
==================

For continuous distributions
----------------------------

.. autosummary::
   :toctree: generated/

   NumericalInverseHermite
   NumericalInversePolynomial
   TransformedDensityRejection
   SimpleRatioUniforms
   RatioUniforms

For discrete distributions
--------------------------

.. autosummary::
   :toctree: generated/

   DiscreteAliasUrn
   DiscreteGuideTable

Warnings / Errors used in :mod:`scipy.stats.sampling`
-----------------------------------------------------

.. autosummary::
   :toctree: generated/

   UNURANError


Generators for pre-defined distributions
========================================

To easily apply the above methods for some of the continuous distributions
in :mod:`scipy.stats`, the following functionality can be used:

.. autosummary::
   :toctree: generated/

   FastGeneratorInversion

"""
from ._sampling import FastGeneratorInversion, RatioUniforms  # noqa: F401
from ._unuran.unuran_wrapper import (  # noqa: F401
   TransformedDensityRejection,
   DiscreteAliasUrn,
   DiscreteGuideTable,
   NumericalInversePolynomial,
   NumericalInverseHermite,
   SimpleRatioUniforms,
   UNURANError
)

__all__ = ["NumericalInverseHermite", "NumericalInversePolynomial",
           "TransformedDensityRejection", "SimpleRatioUniforms",
           "RatioUniforms", "DiscreteAliasUrn", "DiscreteGuideTable",
           "UNURANError", "FastGeneratorInversion"]
<|endoftext|>#

import warnings
from functools import partial

import numpy as np

from scipy import optimize
from scipy import integrate
from scipy.integrate._quadrature import _builtincoeffs
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import scipy.special as sc
from scipy._lib._util import _lazywhere
from .._distn_infrastructure import rv_continuous, _ShapeInfo, rv_continuous_frozen
from .._continuous_distns import uniform, expon, _norm_pdf, _norm_cdf
from .levyst import Nolan
from scipy._lib.doccer import inherit_docstring_from


__all__ = ["levy_stable", "levy_stable_gen", "pdf_from_cf_with_fft"]

# Stable distributions are known for various parameterisations
# some being advantageous for numerical considerations and others
# useful due to their location/scale awareness.
#
# Here we follow [NO] convention (see the references in the docstring
# for levy_stable_gen below).
#
# S0 / Z0 / x0 (aka Zoleterav's M)
# S1 / Z1 / x1
#
# Where S* denotes parameterisation, Z* denotes standardized
# version where gamma = 1, delta = 0 and x* denotes variable.
#
# Scipy's original Stable was a random variate generator. It
# uses S1 and unfortunately is not a location/scale aware.


# default numerical integration tolerance
# used for epsrel in piecewise and both epsrel and epsabs in dni
# (epsabs needed in dni since weighted quad requires epsabs > 0)
_QUAD_EPS = 1.2e-14


def _Phi_Z0(alpha, t):
    return (
        -np.tan(np.pi * alpha / 2) * (np.abs(t) ** (1 - alpha) - 1)
        if alpha != 1
        else -2.0 * np.log(np.abs(t)) / np.pi
    )


def _Phi_Z1(alpha, t):
    return (
        np.tan(np.pi * alpha / 2)
        if alpha != 1
        else -2.0 * np.log(np.abs(t)) / np.pi
    )


def _cf(Phi, t, alpha, beta):
    """Characteristic function."""
    return np.exp(
        -(np.abs(t) ** alpha) * (1 - 1j * beta * np.sign(t) * Phi(alpha, t))
    )


_cf_Z0 = partial(_cf, _Phi_Z0)
_cf_Z1 = partial(_cf, _Phi_Z1)


def _pdf_single_value_cf_integrate(Phi, x, alpha, beta, **kwds):
    """To improve DNI accuracy convert characteristic function in to real
    valued integral using Euler's formula, then exploit cosine symmetry to
    change limits to [0, inf). Finally use cosine addition formula to split
    into two parts that can be handled by weighted quad pack.
    """
    quad_eps = kwds.get("quad_eps", _QUAD_EPS)

    def integrand1(t):
        if t == 0:
            return 0
        return np.exp(-(t ** alpha)) * (
            np.cos(beta * (t ** alpha) * Phi(alpha, t))
        )

    def integrand2(t):
        if t == 0:
            return 0
        return np.exp(-(t ** alpha)) * (
            np.sin(beta * (t ** alpha) * Phi(alpha, t))
        )

    with np.errstate(invalid="ignore"):
        int1, *ret1 = integrate.quad(
            integrand1,
            0,
            np.inf,
            weight="cos",
            wvar=x,
            limit=1000,
            epsabs=quad_eps,
            epsrel=quad_eps,
            full_output=1,
        )

        int2, *ret2 = integrate.quad(
            integrand2,
            0,
            np.inf,
            weight="sin",
            wvar=x,
            limit=1000,
            epsabs=quad_eps,
            epsrel=quad_eps,
            full_output=1,
        )

    return (int1 + int2) / np.pi


_pdf_single_value_cf_integrate_Z0 = partial(
    _pdf_single_value_cf_integrate, _Phi_Z0
)
_pdf_single_value_cf_integrate_Z1 = partial(
    _pdf_single_value_cf_integrate, _Phi_Z1
)


def _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta):
    """Round x close to zeta for Nolan's method in [NO]."""
    #   "8. When |x0-beta*tan(pi*alpha/2)| is small, the
    #   computations of the density and cumulative have numerical problems.
    #   The program works around this by setting
    #   z = beta*tan(pi*alpha/2) when
    #   |z-beta*tan(pi*alpha/2)| < tol(5)*alpha**(1/alpha).
    #   (The bound on the right is ad hoc, to get reasonable behavior
    #   when alpha is small)."
    # where tol(5) = 0.5e-2 by default.
    #
    # We seem to have partially addressed this through re-expression of
    # g(theta) here, but it still needs to be used in some extreme cases.
    # Perhaps tol(5) = 0.5e-2 could be reduced for our implementation.
    if np.abs(x0 - zeta) < x_tol_near_zeta * alpha ** (1 / alpha):
        x0 = zeta
    return x0


def _nolan_round_difficult_input(
    x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one
):
    """Round difficult input values for Nolan's method in [NO]."""

    # following Nolan's STABLE,
    #   "1. When 0 < |alpha-1| < 0.005, the program has numerical problems
    #   evaluating the pdf and cdf.  The current version of the program sets
    #   alpha=1 in these cases. This approximation is not bad in the S0
    #   parameterization."
    if np.abs(alpha - 1) < alpha_tol_near_one:
        alpha = 1.0

    #   "2. When alpha=1 and |beta| < 0.005, the program has numerical
    #   problems.  The current version sets beta=0."
    # We seem to have addressed this through re-expression of g(theta) here

    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)
    return x0, alpha, beta


def _pdf_single_value_piecewise_Z1(x, alpha, beta, **kwds):
    # convert from Nolan's S_1 (aka S) to S_0 (aka Zolaterev M)
    # parameterization

    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    x0 = x + zeta if alpha != 1 else x

    return _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds)


def _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):

    quad_eps = kwds.get("quad_eps", _QUAD_EPS)
    x_tol_near_zeta = kwds.get("piecewise_x_tol_near_zeta", 0.005)
    alpha_tol_near_one = kwds.get("piecewise_alpha_tol_near_one", 0.005)

    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    x0, alpha, beta = _nolan_round_difficult_input(
        x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one
    )

    # some other known distribution pdfs / analytical cases
    # TODO: add more where possible with test coverage,
    # eg https://en.wikipedia.org/wiki/Stable_distribution#Other_analytic_cases
    if alpha == 2.0:
        # normal
        return _norm_pdf(x0 / np.sqrt(2)) / np.sqrt(2)
    elif alpha == 0.5 and beta == 1.0:
        # levy
        # since S(1/2, 1, gamma, delta; <x>) ==
        # S(1/2, 1, gamma, gamma + delta; <x0>).
        _x = x0 + 1
        if _x <= 0:
            return 0

        return 1 / np.sqrt(2 * np.pi * _x) / _x * np.exp(-1 / (2 * _x))
    elif alpha == 0.5 and beta == 0.0 and x0 != 0:
        # analytical solution [HO]
        S, C = sc.fresnel([1 / np.sqrt(2 * np.pi * np.abs(x0))])
        arg = 1 / (4 * np.abs(x0))
        return (
            np.sin(arg) * (0.5 - S[0]) + np.cos(arg) * (0.5 - C[0])
        ) / np.sqrt(2 * np.pi * np.abs(x0) ** 3)
    elif alpha == 1.0 and beta == 0.0:
        # cauchy
        return 1 / (1 + x0 ** 2) / np.pi

    return _pdf_single_value_piecewise_post_rounding_Z0(
        x0, alpha, beta, quad_eps, x_tol_near_zeta
    )


def _pdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps,
                                                 x_tol_near_zeta):
    """Calculate pdf using Nolan's methods as detailed in [NO]."""

    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c2 = _nolan.c2
    g = _nolan.g

    # round x0 to zeta again if needed. zeta was recomputed and may have
    # changed due to floating point differences.
    # See https://github.com/scipy/scipy/pull/18133
    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)
    # handle Nolan's initial case logic
    if x0 == zeta:
        return (
            sc.gamma(1 + 1 / alpha)
            * np.cos(xi)
            / np.pi
            / ((1 + zeta ** 2) ** (1 / alpha / 2))
        )
    elif x0 < zeta:
        return _pdf_single_value_piecewise_post_rounding_Z0(
            -x0, alpha, -beta, quad_eps, x_tol_near_zeta
        )

    # following Nolan, we may now assume
    #   x0 > zeta when alpha != 1
    #   beta != 0 when alpha == 1

    # spare calculating integral on null set
    # use isclose as macos has fp differences
    if np.isclose(-xi, np.pi / 2, rtol=1e-014, atol=1e-014):
        return 0.0

    def integrand(theta):
        # limit any numerical issues leading to g_1 < 0 near theta limits
        g_1 = g(theta)
        if not np.isfinite(g_1) or g_1 < 0:
            g_1 = 0
        return g_1 * np.exp(-g_1)

    with np.errstate(all="ignore"):
        peak = optimize.bisect(
            lambda t: g(t) - 1, -xi, np.pi / 2, xtol=quad_eps
        )

        # this integrand can be very peaked, so we need to force
        # QUADPACK to evaluate the function inside its support
        #

        # lastly, we add additional samples at
        #   ~exp(-100), ~exp(-10), ~exp(-5), ~exp(-1)
        # to improve QUADPACK's detection of rapidly descending tail behavior
        # (this choice is fairly ad hoc)
        tail_points = [
            optimize.bisect(lambda t: g(t) - exp_height, -xi, np.pi / 2)
            for exp_height in [100, 10, 5]
            # exp_height = 1 is handled by peak
        ]
        intg_points = [0, peak] + tail_points
        intg, *ret = integrate.quad(
            integrand,
            -xi,
            np.pi / 2,
            points=intg_points,
            limit=100,
            epsrel=quad_eps,
            epsabs=0,
            full_output=1,
        )

    return c2 * intg


def _cdf_single_value_piecewise_Z1(x, alpha, beta, **kwds):
    # convert from Nolan's S_1 (aka S) to S_0 (aka Zolaterev M)
    # parameterization

    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    x0 = x + zeta if alpha != 1 else x

    return _cdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds)


def _cdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):

    quad_eps = kwds.get("quad_eps", _QUAD_EPS)
    x_tol_near_zeta = kwds.get("piecewise_x_tol_near_zeta", 0.005)
    alpha_tol_near_one = kwds.get("piecewise_alpha_tol_near_one", 0.005)

    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    x0, alpha, beta = _nolan_round_difficult_input(
        x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one
    )

    # some other known distribution cdfs / analytical cases
    # TODO: add more where possible with test coverage,
    # eg https://en.wikipedia.org/wiki/Stable_distribution#Other_analytic_cases
    if alpha == 2.0:
        # normal
        return _norm_cdf(x0 / np.sqrt(2))
    elif alpha == 0.5 and beta == 1.0:
        # levy
        # since S(1/2, 1, gamma, delta; <x>) ==
        # S(1/2, 1, gamma, gamma + delta; <x0>).
        _x = x0 + 1
        if _x <= 0:
            return 0

        return sc.erfc(np.sqrt(0.5 / _x))
    elif alpha == 1.0 and beta == 0.0:
        # cauchy
        return 0.5 + np.arctan(x0) / np.pi

    return _cdf_single_value_piecewise_post_rounding_Z0(
        x0, alpha, beta, quad_eps, x_tol_near_zeta
    )


def _cdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps,
                                                 x_tol_near_zeta):
    """Calculate cdf using Nolan's methods as detailed in [NO]."""
    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c1 = _nolan.c1
    # c2 = _nolan.c2
    c3 = _nolan.c3
    g = _nolan.g
    # round x0 to zeta again if needed. zeta was recomputed and may have
    # changed due to floating point differences.
    # See https://github.com/scipy/scipy/pull/18133
    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)
    # handle Nolan's initial case logic
    if (alpha == 1 and beta < 0) or x0 < zeta:
        # NOTE: Nolan's paper has a typo here!
        # He states F(x) = 1 - F(x, alpha, -beta), but this is clearly
        # incorrect since F(-infty) would be 1.0 in this case
        # Indeed, the alpha != 1, x0 < zeta case is correct here.
        return 1 - _cdf_single_value_piecewise_post_rounding_Z0(
            -x0, alpha, -beta, quad_eps, x_tol_near_zeta
        )
    elif x0 == zeta:
        return 0.5 - xi / np.pi

    # following Nolan, we may now assume
    #   x0 > zeta when alpha != 1
    #   beta > 0 when alpha == 1

    # spare calculating integral on null set
    # use isclose as macos has fp differences
    if np.isclose(-xi, np.pi / 2, rtol=1e-014, atol=1e-014):
        return c1

    def integrand(theta):
        g_1 = g(theta)
        return np.exp(-g_1)

    with np.errstate(all="ignore"):
        # shrink supports where required
        left_support = -xi
        right_support = np.pi / 2
        if alpha > 1:
            # integrand(t) monotonic 0 to 1
            if integrand(-xi) != 0.0:
                res = optimize.minimize(
                    integrand,
                    (-xi,),
                    method="L-BFGS-B",
                    bounds=[(-xi, np.pi / 2)],
                )
                left_support = res.x[0]
        else:
            # integrand(t) monotonic 1 to 0
            if integrand(np.pi / 2) != 0.0:
                res = optimize.minimize(
                    integrand,
                    (np.pi / 2,),
                    method="L-BFGS-B",
                    bounds=[(-xi, np.pi / 2)],
                )
                right_support = res.x[0]

        intg, *ret = integrate.quad(
            integrand,
            left_support,
            right_support,
            points=[left_support, right_support],
            limit=100,
            epsrel=quad_eps,
            epsabs=0,
            full_output=1,
        )

    return c1 + c3 * intg


def _rvs_Z1(alpha, beta, size=None, random_state=None):
    """Simulate random variables using Nolan's methods as detailed in [NO].
    """

    def alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        return (
            2
            / np.pi
            * (
                (np.pi / 2 + bTH) * tanTH
                - beta * np.log((np.pi / 2 * W * cosTH) / (np.pi / 2 + bTH))
            )
        )

    def beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        return (
            W
            / (cosTH / np.tan(aTH) + np.sin(TH))
            * ((np.cos(aTH) + np.sin(aTH) * tanTH) / W) ** (1.0 / alpha)
        )

    def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        # alpha is not 1 and beta is not 0
        val0 = beta * np.tan(np.pi * alpha / 2)
        th0 = np.arctan(val0) / alpha
        val3 = W / (cosTH / np.tan(alpha * (th0 + TH)) + np.sin(TH))
        res3 = val3 * (
            (
                np.cos(aTH)
                + np.sin(aTH) * tanTH
                - val0 * (np.sin(aTH) - np.cos(aTH) * tanTH)
            )
            / W
        ) ** (1.0 / alpha)
        return res3

    def alphanot1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        res = _lazywhere(
            beta == 0,
            (alpha, beta, TH, aTH, bTH, cosTH, tanTH, W),
            beta0func,
            f2=otherwise,
        )
        return res

    alpha = np.broadcast_to(alpha, size)
    beta = np.broadcast_to(beta, size)
    TH = uniform.rvs(
        loc=-np.pi / 2.0, scale=np.pi, size=size, random_state=random_state
    )
    W = expon.rvs(size=size, random_state=random_state)
    aTH = alpha * TH
    bTH = beta * TH
    cosTH = np.cos(TH)
    tanTH = np.tan(TH)
    res = _lazywhere(
        alpha == 1,
        (alpha, beta, TH, aTH, bTH, cosTH, tanTH, W),
        alpha1func,
        f2=alphanot1func,
    )
    return res


def _fitstart_S0(data):
    alpha, beta, delta1, gamma = _fitstart_S1(data)

    # Formulas for mapping parameters in S1 parameterization to
    # those in S0 parameterization can be found in [NO]. Note that
    # only delta changes.
    if alpha != 1:
        delta0 = delta1 + beta * gamma * np.tan(np.pi * alpha / 2.0)
    else:
        delta0 = delta1 + 2 * beta * gamma * np.log(gamma) / np.pi

    return alpha, beta, delta0, gamma


def _fitstart_S1(data):
    # We follow McCullock 1986 method - Simple Consistent Estimators
    # of Stable Distribution Parameters

    # fmt: off
    # Table III and IV
    nu_alpha_range = [2.439, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.5, 4,
                      5, 6, 8, 10, 15, 25]
    nu_beta_range = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]

    # table III - alpha = psi_1(nu_alpha, nu_beta)
    alpha_table = np.array([
        [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
        [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
        [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
        [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
        [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
        [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
        [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
        [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
        [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
        [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
        [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
        [0.896, 0.892, 0.884, 0.883, 0.855, 0.823, 0.769],
        [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
        [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.597],
        [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513]]).T
    # transpose because interpolation with `RectBivariateSpline` is with
    # `nu_beta` as `x` and `nu_alpha` as `y`

    # table IV - beta = psi_2(nu_alpha, nu_beta)
    beta_table = np.array([
        [0, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
        [0, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
        [0, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
        [0, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
        [0, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
        [0, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
        [0, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
        [0, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
        [0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
        [0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
        [0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
        [0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
        [0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
        [0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
        [0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274]]).T

    # Table V and VII
    # These are ordered with decreasing `alpha_range`; so we will need to
    # reverse them as required by RectBivariateSpline.
    alpha_range = [2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
                   1, 0.9, 0.8, 0.7, 0.6, 0.5][::-1]
    beta_range = [0, 0.25, 0.5, 0.75, 1]

    # Table V - nu_c = psi_3(alpha, beta)
    nu_c_table = np.array([
        [1.908, 1.908, 1.908, 1.908, 1.908],
        [1.914, 1.915, 1.916, 1.918, 1.921],
        [1.921, 1.922, 1.927, 1.936, 1.947],
        [1.927, 1.930, 1.943, 1.961, 1.987],
        [1.933, 1.940, 1.962, 1.997, 2.043],
        [1.939, 1.952, 1.988, 2.045, 2.116],
        [1.946, 1.967, 2.022, 2.106, 2.211],
        [1.955, 1.984, 2.067, 2.188, 2.333],
        [1.965, 2.007, 2.125, 2.294, 2.491],
        [1.980, 2.040, 2.205, 2.435, 2.696],
        [2.000, 2.085, 2.311, 2.624, 2.973],
        [2.040, 2.149, 2.461, 2.886, 3.356],
        [2.098, 2.244, 2.676, 3.265, 3.912],
        [2.189, 2.392, 3.004, 3.844, 4.775],
        [2.337, 2.634, 3.542, 4.808, 6.247],
        [2.588, 3.073, 4.534, 6.636, 9.144]])[::-1].T
    # transpose because interpolation with `RectBivariateSpline` is with
    # `beta` as `x` and `alpha` as `y`

    # Table VII - nu_zeta = psi_5(alpha, beta)
    nu_zeta_table = np.array([
        [0, 0.000, 0.000, 0.000, 0.000],
        [0, -0.017, -0.032, -0.049, -0.064],
        [0, -0.030, -0.061, -0.092, -0.123],
        [0, -0.043, -0.088, -0.132, -0.179],
        [0, -0.056, -0.111, -0.170, -0.232],
        [0, -0.066, -0.134, -0.206, -0.283],
        [0, -0.075, -0.154, -0.241, -0.335],
        [0, -0.084, -0.173, -0.276, -0.390],
        [0, -0.090, -0.192, -0.310, -0.447],
        [0, -0.095, -0.208, -0.346, -0.508],
        [0, -0.098, -0.223, -0.380, -0.576],
        [0, -0.099, -0.237, -0.424, -0.652],
        [0, -0.096, -0.250, -0.469, -0.742],
        [0, -0.089, -0.262, -0.520, -0.853],
        [0, -0.078, -0.272, -0.581, -0.997],
        [0, -0.061, -0.279, -0.659, -1.198]])[::-1].T
    # fmt: on

    psi_1 = RectBivariateSpline(nu_beta_range, nu_alpha_range,
                                alpha_table, kx=1, ky=1, s=0)

    def psi_1_1(nu_beta, nu_alpha):
        return psi_1(nu_beta, nu_alpha) \
            if nu_beta > 0 else psi_1(-nu_beta, nu_alpha)

    psi_2 = RectBivariateSpline(nu_beta_range, nu_alpha_range,
                                beta_table, kx=1, ky=1, s=0)

    def psi_2_1(nu_beta, nu_alpha):
        return psi_2(nu_beta, nu_alpha) \
            if nu_beta > 0 else -psi_2(-nu_beta, nu_alpha)

    phi_3 = RectBivariateSpline(beta_range, alpha_range, nu_c_table,
                                kx=1, ky=1, s=0)

    def phi_3_1(beta, alpha):
        return phi_3(beta, alpha) if beta > 0 else phi_3(-beta, alpha)

    phi_5 = RectBivariateSpline(beta_range, alpha_range, nu_zeta_table,
                                kx=1, ky=1, s=0)

    def phi_5_1(beta, alpha):
        return phi_5(beta, alpha) if beta > 0 else -phi_5(-beta, alpha)

    # quantiles
    p05 = np.percentile(data, 5)
    p50 = np.percentile(data, 50)
    p95 = np.percentile(data, 95)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)

    nu_alpha = (p95 - p05) / (p75 - p25)
    nu_beta = (p95 + p05 - 2 * p50) / (p95 - p05)

    if nu_alpha >= 2.439:
        eps = np.finfo(float).eps
        alpha = np.clip(psi_1_1(nu_beta, nu_alpha)[0, 0], eps, 2.)
        beta = np.clip(psi_2_1(nu_beta, nu_alpha)[0, 0], -1.0, 1.0)
    else:
        alpha = 2.0
        beta = np.sign(nu_beta)
    c = (p75 - p25) / phi_3_1(beta, alpha)[0, 0]
    zeta = p50 + c * phi_5_1(beta, alpha)[0, 0]
    delta = zeta-beta*c*np.tan(np.pi*alpha/2.) if alpha != 1. else zeta

    return (alpha, beta, delta, c)


class levy_stable_gen(rv_continuous):
    r"""A Levy-stable continuous random variable.

    %(before_notes)s

    See Also
    --------
    levy, levy_l, cauchy, norm

    Notes
    -----
    The distribution for `levy_stable` has characteristic function:

    .. math::

        \varphi(t, \alpha, \beta, c, \mu) =
        e^{it\mu -|ct|^{\alpha}(1-i\beta\operatorname{sign}(t)\Phi(\alpha, t))}

    where two different parameterizations are supported. The first :math:`S_1`:

    .. math::

        \Phi = \begin{cases}
                \tan \left({\frac {\pi \alpha }{2}}\right)&\alpha \neq 1\\
                -{\frac {2}{\pi }}\log |t|&\alpha =1
                \end{cases}

    The second :math:`S_0`:

    .. math::

        \Phi = \begin{cases}
                -\tan \left({\frac {\pi \alpha }{2}}\right)(|ct|^{1-\alpha}-1)
                &\alpha \neq 1\\
                -{\frac {2}{\pi }}\log |ct|&\alpha =1
                \end{cases}


    The probability density function for `levy_stable` is:

    .. math::

        f(x) = \frac{1}{2\pi}\int_{-\infty}^\infty \varphi(t)e^{-ixt}\,dt

    where :math:`-\infty < t < \infty`. This integral does not have a known
    closed form.

    `levy_stable` generalizes several distributions.  Where possible, they
    should be used instead.  Specifically, when the shape parameters
    assume the values in the table below, the corresponding equivalent
    distribution should be used.

    =========  ========  ===========
    ``alpha``  ``beta``   Equivalent
    =========  ========  ===========
     1/2       -1        `levy_l`
     1/2       1         `levy`
     1         0         `cauchy`
     2         any       `norm` (with ``scale=sqrt(2)``)
    =========  ========  ===========

    Evaluation of the pdf uses Nolan's piecewise integration approach with the
    Zolotarev :math:`M` parameterization by default. There is also the option
    to use direct numerical integration of the standard parameterization of the
    characteristic function or to evaluate by taking the FFT of the
    characteristic function.

    The default method can changed by setting the class variable
    ``levy_stable.pdf_default_method`` to one of 'piecewise' for Nolan's
    approach, 'dni' for direct numerical integration, or 'fft-simpson' for the
    FFT based approach. For the sake of backwards compatibility, the methods
    'best' and 'zolotarev' are equivalent to 'piecewise' and the method
    'quadrature' is equivalent to 'dni'.

    The parameterization can be changed  by setting the class variable
    ``levy_stable.parameterization`` to either 'S0' or 'S1'.
    The default is 'S1'.

    To improve performance of piecewise and direct numerical integration one
    can specify ``levy_stable.quad_eps`` (defaults to 1.2e-14). This is used
    as both the absolute and relative quadrature tolerance for direct numerical
    integration and as the relative quadrature tolerance for the piecewise
    method. One can also specify ``levy_stable.piecewise_x_tol_near_zeta``
    (defaults to 0.005) for how close x is to zeta before it is considered the
    same as x [NO]. The exact check is
    ``abs(x0 - zeta) < piecewise_x_tol_near_zeta*alpha**(1/alpha)``. One can
    also specify ``levy_stable.piecewise_alpha_tol_near_one`` (defaults to
    0.005) for how close alpha is to 1 before being considered equal to 1.

    To increase accuracy of FFT calculation one can specify
    ``levy_stable.pdf_fft_grid_spacing`` (defaults to 0.001) and
    ``pdf_fft_n_points_two_power`` (defaults to None which means a value is
    calculated that sufficiently covers the input range).

    Further control over FFT calculation is available by setting
    ``pdf_fft_interpolation_degree`` (defaults to 3) for spline order and
    ``pdf_fft_interpolation_level`` for determining the number of points to use
    in the Newton-Cotes formula when approximating the characteristic function
    (considered experimental).

    Evaluation of the cdf uses Nolan's piecewise integration approach with the
    Zolatarev :math:`S_0` parameterization by default. There is also the option
    to evaluate through integration of an interpolated spline of the pdf
    calculated by means of the FFT method. The settings affecting FFT
    calculation are the same as for pdf calculation. The default cdf method can
    be changed by setting ``levy_stable.cdf_default_method`` to either
    'piecewise' or 'fft-simpson'.  For cdf calculations the Zolatarev method is
    superior in accuracy, so FFT is disabled by default.

    Fitting estimate uses quantile estimation method in [MC]. MLE estimation of
    parameters in fit method uses this quantile estimate initially. Note that
    MLE doesn't always converge if using FFT for pdf calculations; this will be
    the case if alpha <= 1 where the FFT approach doesn't give good
    approximations.

    Any non-missing value for the attribute
    ``levy_stable.pdf_fft_min_points_threshold`` will set
    ``levy_stable.pdf_default_method`` to 'fft-simpson' if a valid
    default method is not otherwise set.



    .. warning::

        For pdf calculations FFT calculation is considered experimental.

        For cdf calculations FFT calculation is considered experimental. Use
        Zolatarev's method instead (default).

    The probability density above is defined in the "standardized" form. To
    shift and/or scale the distribution use the ``loc`` and ``scale``
    parameters.
    Generally ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` is identically
    equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
    ``y = (x - loc) / scale``, except in the ``S1`` parameterization if
    ``alpha == 1``.  In that case ``%(name)s.pdf(x, %(shapes)s, loc, scale)``
    is identically equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
    ``y = (x - loc - 2 * beta * scale * np.log(scale) / np.pi) / scale``.
    See [NO2]_ Definition 1.8 for more information.
    Note that shifting the location of a distribution
    does not make it a "noncentral" distribution.

    References
    ----------
    .. [MC] McCulloch, J., 1986. Simple consistent estimators of stable
        distribution parameters. Communications in Statistics - Simulation and
        Computation 15, 11091136.
    .. [WZ] Wang, Li and Zhang, Ji-Hong, 2008. Simpson's rule based FFT method
        to compute densities of stable distribution.
    .. [NO] Nolan, J., 1997. Numerical Calculation of Stable Densities and
        distributions Functions.
    .. [NO2] Nolan, J., 2018. Stable Distributions: Models for Heavy Tailed
        Data.
    .. [HO] Hopcraft, K. I., Jakeman, E., Tanner, R. M. J., 1999. Lévy random
        walks with fluctuating step number and multiscale behavior.

    %(example)s

    """
    # Configurable options as class variables
    # (accessible from self by attribute lookup).
    parameterization = "S1"
    pdf_default_method = "piecewise"
    cdf_default_method = "piecewise"
    quad_eps = _QUAD_EPS
    piecewise_x_tol_near_zeta = 0.005
    piecewise_alpha_tol_near_one = 0.005
    pdf_fft_min_points_threshold = None
    pdf_fft_grid_spacing = 0.001
    pdf_fft_n_points_two_power = None
    pdf_fft_interpolation_level = 3
    pdf_fft_interpolation_degree = 3

    def __call__(self, *args, **params):
        dist = levy_stable_frozen(self, *args, **params)
        dist.parameterization = self.parameterization
        return dist

    def _argcheck(self, alpha, beta):
        return (alpha > 0) & (alpha <= 2) & (beta <= 1) & (beta >= -1)

    def _shape_info(self):
        ialpha = _ShapeInfo("alpha", False, (0, 2), (False, True))
        ibeta = _ShapeInfo("beta", False, (-1, 1), (True, True))
        return [ialpha, ibeta]

    def _parameterization(self):
        allowed = ("S0", "S1")
        pz = self.parameterization
        if pz not in allowed:
            raise RuntimeError(
                f"Parameterization '{pz}' in supported list: {allowed}"
            )
        return pz

    @inherit_docstring_from(rv_continuous)
    def rvs(self, *args, **kwds):
        X1 = super().rvs(*args, **kwds)

        kwds.pop("discrete", None)
        kwds.pop("random_state", None)
        (alpha, beta), delta, gamma, size = self._parse_args_rvs(*args, **kwds)

        # shift location for this parameterisation (S1)
        X1 = np.where(
            alpha == 1.0, X1 + 2 * beta * gamma * np.log(gamma) / np.pi, X1
        )

        if self._parameterization() == "S0":
            return np.where(
                alpha == 1.0,
                X1 - (beta * 2 * gamma * np.log(gamma) / np.pi),
                X1 - gamma * beta * np.tan(np.pi * alpha / 2.0),
            )
        elif self._parameterization() == "S1":
            return X1

    def _rvs(self, alpha, beta, size=None, random_state=None):
        return _rvs_Z1(alpha, beta, size, random_state)

    @inherit_docstring_from(rv_continuous)
    def pdf(self, x, *args, **kwds):
        # override base class version to correct
        # location for S1 parameterization
        if self._parameterization() == "S0":
            return super().pdf(x, *args, **kwds)
        elif self._parameterization() == "S1":
            (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)
            if np.all(np.reshape(alpha, (1, -1))[0, :] != 1):
                return super().pdf(x, *args, **kwds)
            else:
                # correct location for this parameterisation
                x = np.reshape(x, (1, -1))[0, :]
                x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

                data_in = np.dstack((x, alpha, beta))[0]
                data_out = np.empty(shape=(len(data_in), 1))
                # group data in unique arrays of alpha, beta pairs
                uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    _delta = (
                        delta + 2 * _beta * gamma * np.log(gamma) / np.pi
                        if _alpha == 1.0
                        else delta
                    )
                    data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
                    _x = data_in[data_mask, 0]
                    data_out[data_mask] = (
                        super()
                        .pdf(_x, _alpha, _beta, loc=_delta, scale=gamma)
                        .reshape(len(_x), 1)
                    )
                output = data_out.T[0]
                if output.shape == (1,):
                    return output[0]
                return output

    def _pdf(self, x, alpha, beta):
        if self._parameterization() == "S0":
            _pdf_single_value_piecewise = _pdf_single_value_piecewise_Z0
            _pdf_single_value_cf_integrate = _pdf_single_value_cf_integrate_Z0
            _cf = _cf_Z0
        elif self._parameterization() == "S1":
            _pdf_single_value_piecewise = _pdf_single_value_piecewise_Z1
            _pdf_single_value_cf_integrate = _pdf_single_value_cf_integrate_Z1
            _cf = _cf_Z1

        x = np.asarray(x).reshape(1, -1)[0, :]

        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

        data_in = np.dstack((x, alpha, beta))[0]
        data_out = np.empty(shape=(len(data_in), 1))

        pdf_default_method_name = self.pdf_default_method
        if pdf_default_method_name in ("piecewise", "best", "zolotarev"):
            pdf_single_value_method = _pdf_single_value_piecewise
        elif pdf_default_method_name in ("dni", "quadrature"):
            pdf_single_value_method = _pdf_single_value_cf_integrate
        elif (
            pdf_default_method_name == "fft-simpson"
            or self.pdf_fft_min_points_threshold is not None
        ):
            pdf_single_value_method = None

        pdf_single_value_kwds = {
            "quad_eps": self.quad_eps,
            "piecewise_x_tol_near_zeta": self.piecewise_x_tol_near_zeta,
            "piecewise_alpha_tol_near_one": self.piecewise_alpha_tol_near_one,
        }

        fft_grid_spacing = self.pdf_fft_grid_spacing
        fft_n_points_two_power = self.pdf_fft_n_points_two_power
        fft_interpolation_level = self.pdf_fft_interpolation_level
        fft_interpolation_degree = self.pdf_fft_interpolation_degree

        # group data in unique arrays of alpha, beta pairs
        uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
        for pair in uniq_param_pairs:
            data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
            data_subset = data_in[data_mask]
            if pdf_single_value_method is not None:
                data_out[data_mask] = np.array(
                    [
                        pdf_single_value_method(
                            _x, _alpha, _beta, **pdf_single_value_kwds
                        )
                        for _x, _alpha, _beta in data_subset
                    ]
                ).reshape(len(data_subset), 1)
            else:
                warnings.warn(
                    "Density calculations experimental for FFT method."
                    + " Use combination of piecewise and dni methods instead.",
                    RuntimeWarning, stacklevel=3,
                )
                _alpha, _beta = pair
                _x = data_subset[:, (0,)]

                if _alpha < 1.0:
                    raise RuntimeError(
                        "FFT method does not work well for alpha less than 1."
                    )

                # need enough points to "cover" _x for interpolation
                if fft_grid_spacing is None and fft_n_points_two_power is None:
                    raise ValueError(
                        "One of fft_grid_spacing or fft_n_points_two_power "
                        + "needs to be set."
                    )
                max_abs_x = np.max(np.abs(_x))
                h = (
                    2 ** (3 - fft_n_points_two_power) * max_abs_x
                    if fft_grid_spacing is None
                    else fft_grid_spacing
                )
                q = (
                    np.ceil(np.log(2 * max_abs_x / h) / np.log(2)) + 2
                    if fft_n_points_two_power is None
                    else int(fft_n_points_two_power)
                )

                # for some parameters, the range of x can be quite
                # large, let's choose an arbitrary cut off (8GB) to save on
                # computer memory.
                MAX_Q = 30
                if q > MAX_Q:
                    raise RuntimeError(
                        "fft_n_points_two_power has a maximum "
                        + f"value of {MAX_Q}"
                    )

                density_x, density = pdf_from_cf_with_fft(
                    lambda t: _cf(t, _alpha, _beta),
                    h=h,
                    q=q,
                    level=fft_interpolation_level,
                )
                f = interpolate.InterpolatedUnivariateSpline(
                    density_x, np.real(density), k=fft_interpolation_degree
                )  # patch FFT to use cubic
                data_out[data_mask] = f(_x)

        return data_out.T[0]

    @inherit_docstring_from(rv_continuous)
    def cdf(self, x, *args, **kwds):
        # override base class version to correct
        # location for S1 parameterization
        # NOTE: this is near identical to pdf() above
        if self._parameterization() == "S0":
            return super().cdf(x, *args, **kwds)
        elif self._parameterization() == "S1":
            (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)
            if np.all(np.reshape(alpha, (1, -1))[0, :] != 1):
                return super().cdf(x, *args, **kwds)
            else:
                # correct location for this parameterisation
                x = np.reshape(x, (1, -1))[0, :]
                x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

                data_in = np.dstack((x, alpha, beta))[0]
                data_out = np.empty(shape=(len(data_in), 1))
                # group data in unique arrays of alpha, beta pairs
                uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    _delta = (
                        delta + 2 * _beta * gamma * np.log(gamma) / np.pi
                        if _alpha == 1.0
                        else delta
                    )
                    data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
                    _x = data_in[data_mask, 0]
                    data_out[data_mask] = (
                        super()
                        .cdf(_x, _alpha, _beta, loc=_delta, scale=gamma)
                        .reshape(len(_x), 1)
                    )
                output = data_out.T[0]
                if output.shape == (1,):
                    return output[0]
                return output

    def _cdf(self, x, alpha, beta):
        if self._parameterization() == "S0":
            _cdf_single_value_piecewise = _cdf_single_value_piecewise_Z0
            _cf = _cf_Z0
        elif self._parameterization() == "S1":
            _cdf_single_value_piecewise = _cdf_single_value_piecewise_Z1
            _cf = _cf_Z1

        x = np.asarray(x).reshape(1, -1)[0, :]

        x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

        data_in = np.dstack((x, alpha, beta))[0]
        data_out = np.empty(shape=(len(data_in), 1))

        cdf_default_method_name = self.cdf_default_method
        if cdf_default_method_name == "piecewise":
            cdf_single_value_method = _cdf_single_value_piecewise
        elif cdf_default_method_name == "fft-simpson":
            cdf_single_value_method = None

        cdf_single_value_kwds = {
            "quad_eps": self.quad_eps,
            "piecewise_x_tol_near_zeta": self.piecewise_x_tol_near_zeta,
            "piecewise_alpha_tol_near_one": self.piecewise_alpha_tol_near_one,
        }

        fft_grid_spacing = self.pdf_fft_grid_spacing
        fft_n_points_two_power = self.pdf_fft_n_points_two_power
        fft_interpolation_level = self.pdf_fft_interpolation_level
        fft_interpolation_degree = self.pdf_fft_interpolation_degree

        # group data in unique arrays of alpha, beta pairs
        uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
        for pair in uniq_param_pairs:
            data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
            data_subset = data_in[data_mask]
            if cdf_single_value_method is not None:
                data_out[data_mask] = np.array(
                    [
                        cdf_single_value_method(
                            _x, _alpha, _beta, **cdf_single_value_kwds
                        )
                        for _x, _alpha, _beta in data_subset
                    ]
                ).reshape(len(data_subset), 1)
            else:
                warnings.warn(
                    "Cumulative density calculations experimental for FFT"
                    + " method. Use piecewise method instead.",
                    RuntimeWarning, stacklevel=3,
                )
                _alpha, _beta = pair
                _x = data_subset[:, (0,)]

                # need enough points to "cover" _x for interpolation
                if fft_grid_spacing is None and fft_n_points_two_power is None:
                    raise ValueError(
                        "One of fft_grid_spacing or fft_n_points_two_power "
                        + "needs to be set."
                    )
                max_abs_x = np.max(np.abs(_x))
                h = (
                    2 ** (3 - fft_n_points_two_power) * max_abs_x
                    if fft_grid_spacing is None
                    else fft_grid_spacing
                )
                q = (
                    np.ceil(np.log(2 * max_abs_x / h) / np.log(2)) + 2
                    if fft_n_points_two_power is None
                    else int(fft_n_points_two_power)
                )

                density_x, density = pdf_from_cf_with_fft(
                    lambda t: _cf(t, _alpha, _beta),
                    h=h,
                    q=q,
                    level=fft_interpolation_level,
                )
                f = interpolate.InterpolatedUnivariateSpline(
                    density_x, np.real(density), k=fft_interpolation_degree
                )
                data_out[data_mask] = np.array(
                    [f.integral(self.a, float(x_1.squeeze())) for x_1 in _x]
                ).reshape(data_out[data_mask].shape)

        return data_out.T[0]

    def _fitstart(self, data):
        if self._parameterization() == "S0":
            _fitstart = _fitstart_S0
        elif self._parameterization() == "S1":
            _fitstart = _fitstart_S1
        return _fitstart(data)

    def _stats(self, alpha, beta):
        mu = 0 if alpha > 1 else np.nan
        mu2 = 2 if alpha == 2 else np.inf
        g1 = 0.0 if alpha == 2.0 else np.nan
        g2 = 0.0 if alpha == 2.0 else np.nan
        return mu, mu2, g1, g2


# cotes numbers - see sequence from http://oeis.org/A100642
Cotes_table = np.array(
    [[], [1]] + [v[2] for v in _builtincoeffs.values()], dtype=object
)
Cotes = np.array(
    [
        np.pad(r, (0, len(Cotes_table) - 1 - len(r)), mode='constant')
        for r in Cotes_table
    ]
)


def pdf_from_cf_with_fft(cf, h=0.01, q=9, level=3):
    """Calculates pdf from characteristic function.

    Uses fast Fourier transform with Newton-Cotes integration following [WZ].
    Defaults to using Simpson's method (3-point Newton-Cotes integration).

    Parameters
    ----------
    cf : callable
        Single argument function from float -> complex expressing a
        characteristic function for some distribution.
    h : Optional[float]
        Step size for Newton-Cotes integration. Default: 0.01
    q : Optional[int]
        Use 2**q steps when performing Newton-Cotes integration.
        The infinite integral in the inverse Fourier transform will then
        be restricted to the interval [-2**q * h / 2, 2**q * h / 2]. Setting
        the number of steps equal to a power of 2 allows the fft to be
        calculated in O(n*log(n)) time rather than O(n**2).
        Default: 9
    level : Optional[int]
        Calculate integral using n-point Newton-Cotes integration for
        n = level. The 3-point Newton-Cotes formula corresponds to Simpson's
        rule. Default: 3

    Returns
    -------
    x_l : ndarray
        Array of points x at which pdf is estimated. 2**q equally spaced
        points from -pi/h up to but not including pi/h.
    density : ndarray
        Estimated values of pdf corresponding to cf at points in x_l.

    References
    ----------
    .. [WZ] Wang, Li and Zhang, Ji-Hong, 2008. Simpson's rule based FFT method
        to compute densities of stable distribution.
    """
    n = level
    N = 2**q
    steps = np.arange(0, N)
    L = N * h / 2
    x_l = np.pi * (steps - N / 2) / L
    if level > 1:
        indices = np.arange(n).reshape(n, 1)
        s1 = np.sum(
            (-1) ** steps * Cotes[n, indices] * np.fft.fft(
                (-1)**steps * cf(-L + h * steps + h * indices / (n - 1))
            ) * np.exp(
                1j * np.pi * indices / (n - 1)
                - 2 * 1j * np.pi * indices * steps /
                (N * (n - 1))
            ),
            axis=0
        )
    else:
        s1 = (-1) ** steps * Cotes[n, 0] * np.fft.fft(
            (-1) ** steps * cf(-L + h * steps)
        )
    density = h * s1 / (2 * np.pi * np.sum(Cotes[n]))
    return (x_l, density)


levy_stable = levy_stable_gen(name="levy_stable")


class levy_stable_frozen(rv_continuous_frozen):
    @property
    def parameterization(self):
        return self.dist.parameterization

    @parameterization.setter
    def parameterization(self, value):
        self.dist.parameterization = value
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = ['multigammaln']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="spfun_stats",
                                   private_modules=["_spfun_stats"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


_polyfuns = ['legendre', 'chebyt', 'chebyu', 'chebyc', 'chebys',
             'jacobi', 'laguerre', 'genlaguerre', 'hermite',
             'hermitenorm', 'gegenbauer', 'sh_legendre', 'sh_chebyt',
             'sh_chebyu', 'sh_jacobi']

# Correspondence between new and old names of root functions
_rootfuns_map = {'roots_legendre': 'p_roots',
               'roots_chebyt': 't_roots',
               'roots_chebyu': 'u_roots',
               'roots_chebyc': 'c_roots',
               'roots_chebys': 's_roots',
               'roots_jacobi': 'j_roots',
               'roots_laguerre': 'l_roots',
               'roots_genlaguerre': 'la_roots',
               'roots_hermite': 'h_roots',
               'roots_hermitenorm': 'he_roots',
               'roots_gegenbauer': 'cg_roots',
               'roots_sh_legendre': 'ps_roots',
               'roots_sh_chebyt': 'ts_roots',
               'roots_sh_chebyu': 'us_roots',
               'roots_sh_jacobi': 'js_roots'}


__all__ = _polyfuns + list(_rootfuns_map.keys()) + [  # noqa: F822
    'airy', 'p_roots', 't_roots', 'u_roots', 'c_roots', 's_roots',
    'j_roots', 'l_roots', 'la_roots', 'h_roots', 'he_roots', 'cg_roots',
    'ps_roots', 'ts_roots', 'us_roots', 'js_roots'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="orthogonal",
                                   private_modules=["_orthogonal"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

# ruff: noqa: F822
__all__ = [
    'clpmn',
    'lpmn',
    'lpn',
    'lqmn',
    'pbdv'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="specfun",
                                   private_modules=["_basic", "_specfun"], all=__all__,
                                   attribute=name)
<|endoftext|>import collections
import numbers
import numpy as np

from ._input_validation import _nonneg_int_or_fail

from ._special_ufuncs import (legendre_p, assoc_legendre_p,
                              sph_legendre_p, sph_harm_y)
from ._gufuncs import (legendre_p_all, assoc_legendre_p_all,
                       sph_legendre_p_all, sph_harm_y_all)

__all__ = [
    "assoc_legendre_p",
    "assoc_legendre_p_all",
    "legendre_p",
    "legendre_p_all",
    "sph_harm_y",
    "sph_harm_y_all",
    "sph_legendre_p",
    "sph_legendre_p_all",
]


class MultiUFunc:
    def __init__(self, ufunc_or_ufuncs, doc=None, *,
                 force_complex_output=False, **default_kwargs):
        if not isinstance(ufunc_or_ufuncs, np.ufunc):
            if isinstance(ufunc_or_ufuncs, collections.abc.Mapping):
                ufuncs_iter = ufunc_or_ufuncs.values()
            elif isinstance(ufunc_or_ufuncs, collections.abc.Iterable):
                ufuncs_iter = ufunc_or_ufuncs
            else:
                raise ValueError("ufunc_or_ufuncs should be a ufunc or a"
                                 " ufunc collection")

            # Perform input validation to ensure all ufuncs in ufuncs are
            # actually ufuncs and all take the same input types.
            seen_input_types = set()
            for ufunc in ufuncs_iter:
                if not isinstance(ufunc, np.ufunc):
                    raise ValueError("All ufuncs must have type `numpy.ufunc`."
                                     f" Received {ufunc_or_ufuncs}")
                seen_input_types.add(frozenset(x.split("->")[0] for x in ufunc.types))
            if len(seen_input_types) > 1:
                raise ValueError("All ufuncs must take the same input types.")

        self._ufunc_or_ufuncs = ufunc_or_ufuncs
        self.__doc = doc
        self.__force_complex_output = force_complex_output
        self._default_kwargs = default_kwargs
        self._resolve_out_shapes = None
        self._finalize_out = None
        self._key = None
        self._ufunc_default_args = lambda *args, **kwargs: ()
        self._ufunc_default_kwargs = lambda *args, **kwargs: {}

    @property
    def __doc__(self):
        return self.__doc

    def _override_key(self, func):
        """Set `key` method by decorating a function.
        """
        self._key = func

    def _override_ufunc_default_args(self, func):
        self._ufunc_default_args = func

    def _override_ufunc_default_kwargs(self, func):
        self._ufunc_default_kwargs = func

    def _override_resolve_out_shapes(self, func):
        """Set `resolve_out_shapes` method by decorating a function."""
        if func.__doc__ is None:
            func.__doc__ = \
                """Resolve to output shapes based on relevant inputs."""
        func.__name__ = "resolve_out_shapes"
        self._resolve_out_shapes = func

    def _override_finalize_out(self, func):
        self._finalize_out = func

    def _resolve_ufunc(self, **kwargs):
        """Resolve to a ufunc based on keyword arguments."""

        if isinstance(self._ufunc_or_ufuncs, np.ufunc):
            return self._ufunc_or_ufuncs

        ufunc_key = self._key(**kwargs)
        return self._ufunc_or_ufuncs[ufunc_key]

    def __call__(self, *args, **kwargs):
        kwargs = self._default_kwargs | kwargs

        args += self._ufunc_default_args(**kwargs)

        ufunc = self._resolve_ufunc(**kwargs)

        # array arguments to be passed to the ufunc
        ufunc_args = [np.asarray(arg) for arg in args[-ufunc.nin:]]

        ufunc_kwargs = self._ufunc_default_kwargs(**kwargs)

        if (self._resolve_out_shapes is not None):
            ufunc_arg_shapes = tuple(np.shape(ufunc_arg) for ufunc_arg in ufunc_args)
            ufunc_out_shapes = self._resolve_out_shapes(*args[:-ufunc.nin],
                                                        *ufunc_arg_shapes, ufunc.nout,
                                                        **kwargs)

            ufunc_arg_dtypes = tuple(ufunc_arg.dtype if hasattr(ufunc_arg, 'dtype')
                                     else np.dtype(type(ufunc_arg))
                                     for ufunc_arg in ufunc_args)

            if hasattr(ufunc, 'resolve_dtypes'):
                ufunc_dtypes = ufunc_arg_dtypes + ufunc.nout * (None,)
                ufunc_dtypes = ufunc.resolve_dtypes(ufunc_dtypes)
                ufunc_out_dtypes = ufunc_dtypes[-ufunc.nout:]
            else:
                ufunc_out_dtype = np.result_type(*ufunc_arg_dtypes)
                if (not np.issubdtype(ufunc_out_dtype, np.inexact)):
                    ufunc_out_dtype = np.float64

                ufunc_out_dtypes = ufunc.nout * (ufunc_out_dtype,)

            if self.__force_complex_output:
                ufunc_out_dtypes = tuple(np.result_type(1j, ufunc_out_dtype)
                                         for ufunc_out_dtype in ufunc_out_dtypes)

            out = tuple(np.empty(ufunc_out_shape, dtype=ufunc_out_dtype)
                        for ufunc_out_shape, ufunc_out_dtype
                        in zip(ufunc_out_shapes, ufunc_out_dtypes))

            ufunc_kwargs['out'] = out

        out = ufunc(*ufunc_args, **ufunc_kwargs)
        if (self._finalize_out is not None):
            out = self._finalize_out(out)

        return out


sph_legendre_p = MultiUFunc(
    sph_legendre_p,
    r"""sph_legendre_p(n, m, theta, *, diff_n=0)

    Spherical Legendre polynomial of the first kind.

    Parameters
    ----------
    n : ArrayLike[int]
        Degree of the spherical Legendre polynomial. Must have ``n >= 0``.
    m : ArrayLike[int]
        Order of the spherical Legendre polynomial.
    theta : ArrayLike[float]
        Input value.
    diff_n : Optional[int]
        A non-negative integer. Compute and return all derivatives up
        to order ``diff_n``. Default is 0.

    Returns
    -------
    p : ndarray or tuple[ndarray]
        Spherical Legendre polynomial with ``diff_n`` derivatives.

    Notes
    -----
    The spherical counterpart of an (unnormalized) associated Legendre polynomial has
    the additional factor

    .. math::

        \sqrt{\frac{(2 n + 1) (n - m)!}{4 \pi (n + m)!}}

    It is the same as the spherical harmonic :math:`Y_{n}^{m}(\theta, \phi)`
    with :math:`\phi = 0`.
    """, diff_n=0
)


@sph_legendre_p._override_key
def _(diff_n):
    diff_n = _nonneg_int_or_fail(diff_n, "diff_n", strict=False)
    if not 0 <= diff_n <= 2:
        raise ValueError(
            "diff_n is currently only implemented for orders 0, 1, and 2,"
            f" received: {diff_n}."
        )
    return diff_n


@sph_legendre_p._override_finalize_out
def _(out):
    return np.moveaxis(out, -1, 0)


sph_legendre_p_all = MultiUFunc(
    sph_legendre_p_all,
    """sph_legendre_p_all(n, m, theta, *, diff_n=0)

    All spherical Legendre polynomials of the first kind up to the
    specified degree ``n`` and order ``m``.

    Output shape is ``(n + 1, 2 * m + 1, ...)``. The entry at ``(j, i)``
    corresponds to degree ``j`` and order ``i`` for all  ``0 <= j <= n``
    and ``-m <= i <= m``.

    See Also
    --------
    sph_legendre_p
    """, diff_n=0
)


@sph_legendre_p_all._override_key
def _(diff_n):
    diff_n = _nonneg_int_or_fail(diff_n, "diff_n", strict=False)
    if not 0 <= diff_n <= 2:
        raise ValueError(
            "diff_n is currently only implemented for orders 0, 1, and 2,"
            f" received: {diff_n}."
        )
    return diff_n


@sph_legendre_p_all._override_ufunc_default_kwargs
def _(diff_n):
    return {'axes': [()] + [(0, 1, -1)]}


@sph_legendre_p_all._override_resolve_out_shapes
def _(n, m, theta_shape, nout, diff_n):
    if not isinstance(n, numbers.Integral) or (n < 0):
        raise ValueError("n must be a non-negative integer.")

    return ((n + 1, 2 * abs(m) + 1) + theta_shape + (diff_n + 1,),)


@sph_legendre_p_all._override_finalize_out
def _(out):
    return np.moveaxis(out, -1, 0)


assoc_legendre_p = MultiUFunc(
    assoc_legendre_p,
    r"""assoc_legendre_p(n, m, z, *, branch_cut=2, norm=False, diff_n=0)

    Associated Legendre polynomial of the first kind.

    Parameters
    ----------
    n : ArrayLike[int]
        Degree of the associated Legendre polynomial. Must have ``n >= 0``.
    m : ArrayLike[int]
        order of the associated Legendre polynomial.
    z : ArrayLike[float | complex]
        Input value.
    branch_cut : Optional[ArrayLike[int]]
        Selects branch cut. Must be 2 (default) or 3.
        2: cut on the real axis ``|z| > 1``
        3: cut on the real axis ``-1 < z < 1``
    norm : Optional[bool]
        If ``True``, compute the normalized associated Legendre polynomial.
        Default is ``False``.
    diff_n : Optional[int]
        A non-negative integer. Compute and return all derivatives up
        to order ``diff_n``. Default is 0.

    Returns
    -------
    p : ndarray or tuple[ndarray]
        Associated Legendre polynomial with ``diff_n`` derivatives.

    Notes
    -----
    The normalized counterpart of an (unnormalized) associated Legendre
    polynomial has the additional factor

    .. math::

        \sqrt{\frac{(2 n + 1) (n - m)!}{2 (n + m)!}}
    """, branch_cut=2, norm=False, diff_n=0
)


@assoc_legendre_p._override_key
def _(branch_cut, norm, diff_n):
    diff_n = _nonneg_int_or_fail(diff_n, "diff_n", strict=False)
    if not 0 <= diff_n <= 2:
        raise ValueError(
            "diff_n is currently only implemented for orders 0, 1, and 2,"
            f" received: {diff_n}."
        )
    return norm, diff_n


@assoc_legendre_p._override_ufunc_default_args
def _(branch_cut, norm, diff_n):
    return branch_cut,


@assoc_legendre_p._override_finalize_out
def _(out):
    return np.moveaxis(out, -1, 0)


assoc_legendre_p_all = MultiUFunc(
    assoc_legendre_p_all,
    """assoc_legendre_p_all(n, m, z, *, branch_cut=2, norm=False, diff_n=0)

    All associated Legendre polynomials of the first kind up to the
    specified degree ``n`` and order ``m``.

    Output shape is ``(n + 1, 2 * m + 1, ...)``. The entry at ``(j, i)``
    corresponds to degree ``j`` and order ``i`` for all  ``0 <= j <= n``
    and ``-m <= i <= m``.

    See Also
    --------
    assoc_legendre_p
    """, branch_cut=2, norm=False, diff_n=0
)


@assoc_legendre_p_all._override_key
def _(branch_cut, norm, diff_n):
    if not ((isinstance(diff_n, numbers.Integral))
            and diff_n >= 0):
        raise ValueError(
            f"diff_n must be a non-negative integer, received: {diff_n}."
        )
    if not 0 <= diff_n <= 2:
        raise ValueError(
            "diff_n is currently only implemented for orders 0, 1, and 2,"
            f" received: {diff_n}."
        )
    return norm, diff_n


@assoc_legendre_p_all._override_ufunc_default_args
def _(branch_cut, norm, diff_n):
    return branch_cut,


@assoc_legendre_p_all._override_ufunc_default_kwargs
def _(branch_cut, norm, diff_n):
    return {'axes': [(), ()] + [(0, 1, -1)]}


@assoc_legendre_p_all._override_resolve_out_shapes
def _(n, m, z_shape, branch_cut_shape, nout, **kwargs):
    diff_n = kwargs['diff_n']

    if not isinstance(n, numbers.Integral) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    if not isinstance(m, numbers.Integral) or (m < 0):
        raise ValueError("m must be a non-negative integer.")

    return ((n + 1, 2 * abs(m) + 1) +
        np.broadcast_shapes(z_shape, branch_cut_shape) + (diff_n + 1,),)


@assoc_legendre_p_all._override_finalize_out
def _(out):
    return np.moveaxis(out, -1, 0)


legendre_p = MultiUFunc(
    legendre_p,
    """legendre_p(n, z, *, diff_n=0)

    Legendre polynomial of the first kind.

    Parameters
    ----------
    n : ArrayLike[int]
        Degree of the Legendre polynomial. Must have ``n >= 0``.
    z : ArrayLike[float]
        Input value.
    diff_n : Optional[int]
        A non-negative integer. Compute and return all derivatives up
        to order ``diff_n``. Default is 0.

    Returns
    -------
    p : ndarray or tuple[ndarray]
        Legendre polynomial with ``diff_n`` derivatives.

    See Also
    --------
    legendre

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    """, diff_n=0
)


@legendre_p._override_key
def _(diff_n):
    if (not isinstance(diff_n, numbers.Integral)) or (diff_n < 0):
        raise ValueError(
            f"diff_n must be a non-negative integer, received: {diff_n}."
        )
    if not 0 <= diff_n <= 2:
        raise NotImplementedError(
            "diff_n is currently only implemented for orders 0, 1, and 2,"
            f" received: {diff_n}."
        )
    return diff_n


@legendre_p._override_finalize_out
def _(out):
    return np.moveaxis(out, -1, 0)


legendre_p_all = MultiUFunc(
    legendre_p_all,
    """legendre_p_all(n, z, *, diff_n=0)

    All Legendre polynomials of the first kind up to the
    specified degree ``n``.

    Output shape is ``(n + 1, ...)``. The entry at ``j``
    corresponds to degree ``j`` for all  ``0 <= j <= n``.

    See Also
    --------
    legendre_p
    """, diff_n=0
)


@legendre_p_all._override_key
def _(diff_n):
    diff_n = _nonneg_int_or_fail(diff_n, "diff_n", strict=False)
    if not 0 <= diff_n <= 2:
        raise ValueError(
            "diff_n is currently only implemented for orders 0, 1, and 2,"
            f" received: {diff_n}."
        )
    return diff_n


@legendre_p_all._override_ufunc_default_kwargs
def _(diff_n):
    return {'axes': [(), (0, -1)]}


@legendre_p_all._override_resolve_out_shapes
def _(n, z_shape, nout, diff_n):
    n = _nonneg_int_or_fail(n, 'n', strict=False)

    return nout * ((n + 1,) + z_shape + (diff_n + 1,),)


@legendre_p_all._override_finalize_out
def _(out):
    return np.moveaxis(out, -1, 0)


sph_harm_y = MultiUFunc(
    sph_harm_y,
    r"""sph_harm_y(n, m, theta, phi, *, diff_n=0)

    Spherical harmonics. They are defined as

    .. math::

        Y_n^m(\theta,\phi) = \sqrt{\frac{2 n + 1}{4 \pi} \frac{(n - m)!}{(n + m)!}}
            P_n^m(\cos(\theta)) e^{i m \phi}

    where :math:`P_n^m` are the (unnormalized) associated Legendre polynomials.

    Parameters
    ----------
    n : ArrayLike[int]
        Degree of the harmonic. Must have ``n >= 0``. This is
        often denoted by ``l`` (lower case L) in descriptions of
        spherical harmonics.
    m : ArrayLike[int]
        Order of the harmonic.
    theta : ArrayLike[float]
        Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
    phi : ArrayLike[float]
        Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.
    diff_n : Optional[int]
        A non-negative integer. Compute and return all derivatives up
        to order ``diff_n``. Default is 0.

    Returns
    -------
    y : ndarray[complex] or tuple[ndarray[complex]]
       Spherical harmonics with ``diff_n`` derivatives.

    Notes
    -----
    There are different conventions for the meanings of the input
    arguments ``theta`` and ``phi``. In SciPy ``theta`` is the
    polar angle and ``phi`` is the azimuthal angle. It is common to
    see the opposite convention, that is, ``theta`` as the azimuthal angle
    and ``phi`` as the polar angle.

    Note that SciPy's spherical harmonics include the Condon-Shortley
    phase [2]_ because it is part of `sph_legendre_p`.

    With SciPy's conventions, the first several spherical harmonics
    are

    .. math::

        Y_0^0(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{1}{\pi}} \\
        Y_1^{-1}(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{3}{2\pi}}
                                    e^{-i\phi} \sin(\theta) \\
        Y_1^0(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{3}{\pi}}
                                 \cos(\theta) \\
        Y_1^1(\theta, \phi) &= -\frac{1}{2} \sqrt{\frac{3}{2\pi}}
                                 e^{i\phi} \sin(\theta).

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 14.30.
           https://dlmf.nist.gov/14.30
    .. [2] https://en.wikipedia.org/wiki/Spherical_harmonics#Condon.E2.80.93Shortley_phase
    """, force_complex_output=True, diff_n=0
)


@sph_harm_y._override_key
def _(diff_n):
    diff_n = _nonneg_int_or_fail(diff_n, "diff_n", strict=False)
    if not 0 <= diff_n <= 2:
        raise ValueError(
            "diff_n is currently only implemented for orders 0, 1, and 2,"
            f" received: {diff_n}."
        )
    return diff_n


@sph_harm_y._override_finalize_out
def _(out):
    if (out.shape[-1] == 1):
        return out[..., 0, 0]

    if (out.shape[-1] == 2):
        return out[..., 0, 0], out[..., [1, 0], [0, 1]]

    if (out.shape[-1] == 3):
        return (out[..., 0, 0], out[..., [1, 0], [0, 1]],
            out[..., [[2, 1], [1, 0]], [[0, 1], [1, 2]]])


sph_harm_y_all = MultiUFunc(
    sph_harm_y_all,
    """sph_harm_y_all(n, m, theta, phi, *, diff_n=0)

    All spherical harmonics up to the specified degree ``n`` and order ``m``.

    Output shape is ``(n + 1, 2 * m + 1, ...)``. The entry at ``(j, i)``
    corresponds to degree ``j`` and order ``i`` for all  ``0 <= j <= n``
    and ``-m <= i <= m``.

    See Also
    --------
    sph_harm_y
    """, force_complex_output=True, diff_n=0
)


@sph_harm_y_all._override_key
def _(diff_n):
    diff_n = _nonneg_int_or_fail(diff_n, "diff_n", strict=False)
    if not 0 <= diff_n <= 2:
        raise ValueError(
            "diff_n is currently only implemented for orders 2,"
            f" received: {diff_n}."
        )
    return diff_n


@sph_harm_y_all._override_ufunc_default_kwargs
def _(diff_n):
    return {'axes': [(), ()] + [(0, 1, -2, -1)]}


@sph_harm_y_all._override_resolve_out_shapes
def _(n, m, theta_shape, phi_shape, nout, **kwargs):
    diff_n = kwargs['diff_n']

    if not isinstance(n, numbers.Integral) or (n < 0):
        raise ValueError("n must be a non-negative integer.")

    return ((n + 1, 2 * abs(m) + 1) + np.broadcast_shapes(theta_shape, phi_shape) +
        (diff_n + 1, diff_n + 1),)


@sph_harm_y_all._override_finalize_out
def _(out):
    if (out.shape[-1] == 1):
        return out[..., 0, 0]

    if (out.shape[-1] == 2):
        return out[..., 0, 0], out[..., [1, 0], [0, 1]]

    if (out.shape[-1] == 3):
        return (out[..., 0, 0], out[..., [1, 0], [0, 1]],
            out[..., [[2, 1], [1, 0]], [[0, 1], [1, 2]]])
<|endoftext|>import math
import operator

def _nonneg_int_or_fail(n, var_name, strict=True):
    try:
        if strict:
            # Raises an exception if float
            n = operator.index(n)
        elif n == math.floor(n):
            n = int(n)
        else:
            raise ValueError()
        if n < 0:
            raise ValueError()
    except (ValueError, TypeError) as err:
        raise err.__class__(f"{var_name} must be a non-negative integer") from err
    return n
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="add_newdocs",
                                   private_modules=["_add_newdocs"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'SpecialFunctionWarning',
    'SpecialFunctionError'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="sf_error",
                                   private_modules=["_sf_error"], all=__all__,
                                   attribute=name)
<|endoftext|>import os
import functools
import operator
from scipy._lib import _pep440

import numpy as np
from numpy.testing import assert_
import pytest

import scipy.special as sc

__all__ = ['with_special_errors', 'assert_func_equal', 'FuncData']


#------------------------------------------------------------------------------
# Check if a module is present to be used in tests
#------------------------------------------------------------------------------

class MissingModule:
    def __init__(self, name):
        self.name = name


def check_version(module, min_ver):
    if type(module) is MissingModule:
        return pytest.mark.skip(reason=f"{module.name} is not installed")
    return pytest.mark.skipif(
        _pep440.parse(module.__version__) < _pep440.Version(min_ver),
        reason=f"{module.__name__} version >= {min_ver} required"
    )


#------------------------------------------------------------------------------
# Enable convergence and loss of precision warnings -- turn off one by one
#------------------------------------------------------------------------------

def with_special_errors(func):
    """
    Enable special function errors (such as underflow, overflow,
    loss of precision, etc.)
    """
    @functools.wraps(func)
    def wrapper(*a, **kw):
        with sc.errstate(all='raise'):
            res = func(*a, **kw)
        return res
    return wrapper


#------------------------------------------------------------------------------
# Comparing function values at many data points at once, with helpful
# error reports
#------------------------------------------------------------------------------

def assert_func_equal(func, results, points, rtol=None, atol=None,
                      param_filter=None, knownfailure=None,
                      vectorized=True, dtype=None, nan_ok=False,
                      ignore_inf_sign=False, distinguish_nan_and_inf=True):
    if hasattr(points, 'next'):
        # it's a generator
        points = list(points)

    points = np.asarray(points)
    if points.ndim == 1:
        points = points[:,None]
    nparams = points.shape[1]

    if hasattr(results, '__name__'):
        # function
        data = points
        result_columns = None
        result_func = results
    else:
        # dataset
        data = np.c_[points, results]
        result_columns = list(range(nparams, data.shape[1]))
        result_func = None

    fdata = FuncData(func, data, list(range(nparams)),
                     result_columns=result_columns, result_func=result_func,
                     rtol=rtol, atol=atol, param_filter=param_filter,
                     knownfailure=knownfailure, nan_ok=nan_ok, vectorized=vectorized,
                     ignore_inf_sign=ignore_inf_sign,
                     distinguish_nan_and_inf=distinguish_nan_and_inf)
    fdata.check()


class FuncData:
    """
    Data set for checking a special function.

    Parameters
    ----------
    func : function
        Function to test
    data : numpy array
        columnar data to use for testing
    param_columns : int or tuple of ints
        Columns indices in which the parameters to `func` lie.
        Can be imaginary integers to indicate that the parameter
        should be cast to complex.
    result_columns : int or tuple of ints, optional
        Column indices for expected results from `func`.
    result_func : callable, optional
        Function to call to obtain results.
    rtol : float, optional
        Required relative tolerance. Default is 5*eps.
    atol : float, optional
        Required absolute tolerance. Default is 5*tiny.
    param_filter : function, or tuple of functions/Nones, optional
        Filter functions to exclude some parameter ranges.
        If omitted, no filtering is done.
    knownfailure : str, optional
        Known failure error message to raise when the test is run.
        If omitted, no exception is raised.
    nan_ok : bool, optional
        If nan is always an accepted result.
    vectorized : bool, optional
        Whether all functions passed in are vectorized.
    ignore_inf_sign : bool, optional
        Whether to ignore signs of infinities.
        (Doesn't matter for complex-valued functions.)
    distinguish_nan_and_inf : bool, optional
        If True, treat numbers which contain nans or infs as
        equal. Sets ignore_inf_sign to be True.

    """

    def __init__(self, func, data, param_columns, result_columns=None,
                 result_func=None, rtol=None, atol=None, param_filter=None,
                 knownfailure=None, dataname=None, nan_ok=False, vectorized=True,
                 ignore_inf_sign=False, distinguish_nan_and_inf=True):
        self.func = func
        self.data = data
        self.dataname = dataname
        if not hasattr(param_columns, '__len__'):
            param_columns = (param_columns,)
        self.param_columns = tuple(param_columns)
        if result_columns is not None:
            if not hasattr(result_columns, '__len__'):
                result_columns = (result_columns,)
            self.result_columns = tuple(result_columns)
            if result_func is not None:
                message = "Only result_func or result_columns should be provided"
                raise ValueError(message)
        elif result_func is not None:
            self.result_columns = None
        else:
            raise ValueError("Either result_func or result_columns should be provided")
        self.result_func = result_func
        self.rtol = rtol
        self.atol = atol
        if not hasattr(param_filter, '__len__'):
            param_filter = (param_filter,)
        self.param_filter = param_filter
        self.knownfailure = knownfailure
        self.nan_ok = nan_ok
        self.vectorized = vectorized
        self.ignore_inf_sign = ignore_inf_sign
        self.distinguish_nan_and_inf = distinguish_nan_and_inf
        if not self.distinguish_nan_and_inf:
            self.ignore_inf_sign = True

    def get_tolerances(self, dtype):
        if not np.issubdtype(dtype, np.inexact):
            dtype = np.dtype(float)
        info = np.finfo(dtype)
        rtol, atol = self.rtol, self.atol
        if rtol is None:
            rtol = 5*info.eps
        if atol is None:
            atol = 5*info.tiny
        return rtol, atol

    def check(self, data=None, dtype=None, dtypes=None):
        """Check the special function against the data."""
        __tracebackhide__ = operator.methodcaller(
            'errisinstance', AssertionError
        )

        if self.knownfailure:
            pytest.xfail(reason=self.knownfailure)

        if data is None:
            data = self.data

        if dtype is None:
            dtype = data.dtype
        else:
            data = data.astype(dtype)

        rtol, atol = self.get_tolerances(dtype)

        # Apply given filter functions
        if self.param_filter:
            param_mask = np.ones((data.shape[0],), np.bool_)
            for j, filter in zip(self.param_columns, self.param_filter):
                if filter:
                    param_mask &= list(filter(data[:,j]))
            data = data[param_mask]

        # Pick parameters from the correct columns
        params = []
        for idx, j in enumerate(self.param_columns):
            if np.iscomplexobj(j):
                j = int(j.imag)
                params.append(data[:,j].astype(complex))
            elif dtypes and idx < len(dtypes):
                params.append(data[:, j].astype(dtypes[idx]))
            else:
                params.append(data[:,j])

        # Helper for evaluating results
        def eval_func_at_params(func, skip_mask=None):
            if self.vectorized:
                got = func(*params)
            else:
                got = []
                for j in range(len(params[0])):
                    if skip_mask is not None and skip_mask[j]:
                        got.append(np.nan)
                        continue
                    got.append(func(*tuple([params[i][j] for i in range(len(params))])))
                got = np.asarray(got)
            if not isinstance(got, tuple):
                got = (got,)
            return got

        # Evaluate function to be tested
        got = eval_func_at_params(self.func)

        # Grab the correct results
        if self.result_columns is not None:
            # Correct results passed in with the data
            wanted = tuple([data[:,icol] for icol in self.result_columns])
        else:
            # Function producing correct results passed in
            skip_mask = None
            if self.nan_ok and len(got) == 1:
                # Don't spend time evaluating what doesn't need to be evaluated
                skip_mask = np.isnan(got[0])
            wanted = eval_func_at_params(self.result_func, skip_mask=skip_mask)

        # Check the validity of each output returned
        assert_(len(got) == len(wanted))

        for output_num, (x, y) in enumerate(zip(got, wanted)):
            if np.issubdtype(x.dtype, np.complexfloating) or self.ignore_inf_sign:
                pinf_x = np.isinf(x)
                pinf_y = np.isinf(y)
                minf_x = np.isinf(x)
                minf_y = np.isinf(y)
            else:
                pinf_x = np.isposinf(x)
                pinf_y = np.isposinf(y)
                minf_x = np.isneginf(x)
                minf_y = np.isneginf(y)
            nan_x = np.isnan(x)
            nan_y = np.isnan(y)

            with np.errstate(all='ignore'):
                abs_y = np.absolute(y)
                abs_y[~np.isfinite(abs_y)] = 0
                diff = np.absolute(x - y)
                diff[~np.isfinite(diff)] = 0

                rdiff = diff / np.absolute(y)
                rdiff[~np.isfinite(rdiff)] = 0

            tol_mask = (diff <= atol + rtol*abs_y)
            pinf_mask = (pinf_x == pinf_y)
            minf_mask = (minf_x == minf_y)

            nan_mask = (nan_x == nan_y)

            bad_j = ~(tol_mask & pinf_mask & minf_mask & nan_mask)

            point_count = bad_j.size
            if self.nan_ok:
                bad_j &= ~nan_x
                bad_j &= ~nan_y
                point_count -= (nan_x | nan_y).sum()

            if not self.distinguish_nan_and_inf and not self.nan_ok:
                # If nan's are okay we've already covered all these cases
                inf_x = np.isinf(x)
                inf_y = np.isinf(y)
                both_nonfinite = (inf_x & nan_y) | (nan_x & inf_y)
                bad_j &= ~both_nonfinite
                point_count -= both_nonfinite.sum()

            if np.any(bad_j):
                # Some bad results: inform what, where, and how bad
                msg = [""]
                msg.append(f"Max |adiff|: {diff[bad_j].max():g}")
                msg.append(f"Max |rdiff|: {rdiff[bad_j].max():g}")
                msg.append("Bad results (%d out of %d) for the following points "
                           "(in output %d):"
                           % (np.sum(bad_j), point_count, output_num,))
                for j in np.nonzero(bad_j)[0]:
                    j = int(j)
                    def fmt(x):
                        return '%30s' % np.array2string(x[j], precision=18)
                    a = "  ".join(map(fmt, params))
                    b = "  ".join(map(fmt, got))
                    c = "  ".join(map(fmt, wanted))
                    d = fmt(rdiff)
                    msg.append(f"{a} => {b} != {c}  (rdiff {d})")
                assert_(False, "\n".join(msg))

    def __repr__(self):
        """Pretty-printing"""
        if np.any(list(map(np.iscomplexobj, self.param_columns))):
            is_complex = " (complex)"
        else:
            is_complex = ""
        if self.dataname:
            return (f"<Data for {self.func.__name__}{is_complex}: "
                    f"{os.path.basename(self.dataname)}>")
        else:
            return f"<Data for {self.func.__name__}{is_complex}>"
<|endoftext|>import numpy as np

from ._ufuncs import _ellip_harm
from ._ellip_harm_2 import _ellipsoid, _ellipsoid_norm


def ellip_harm(h2, k2, n, p, s, signm=1, signn=1):
    r"""
    Ellipsoidal harmonic functions E^p_n(l)

    These are also known as Lame functions of the first kind, and are
    solutions to the Lame equation:

    .. math:: (s^2 - h^2)(s^2 - k^2)E''(s)
              + s(2s^2 - h^2 - k^2)E'(s) + (a - q s^2)E(s) = 0

    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not
    returned) corresponding to the solutions.

    Parameters
    ----------
    h2 : float
        ``h**2``
    k2 : float
        ``k**2``; should be larger than ``h**2``
    n : int
        Degree
    s : float
        Coordinate
    p : int
        Order, can range between [1,2n+1]
    signm : {1, -1}, optional
        Sign of prefactor of functions. Can be +/-1. See Notes.
    signn : {1, -1}, optional
        Sign of prefactor of functions. Can be +/-1. See Notes.

    Returns
    -------
    E : float
        the harmonic :math:`E^p_n(s)`

    See Also
    --------
    ellip_harm_2, ellip_normal

    Notes
    -----
    The geometric interpretation of the ellipsoidal functions is
    explained in [2]_, [3]_, [4]_. The `signm` and `signn` arguments control the
    sign of prefactors for functions according to their type::

        K : +1
        L : signm
        M : signn
        N : signm*signn

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Digital Library of Mathematical Functions 29.12
       https://dlmf.nist.gov/29.12
    .. [2] Bardhan and Knepley, "Computational science and
       re-discovery: open-source implementations of
       ellipsoidal harmonics for problems in potential theory",
       Comput. Sci. Disc. 5, 014006 (2012)
       :doi:`10.1088/1749-4699/5/1/014006`.
    .. [3] David J.and Dechambre P, "Computation of Ellipsoidal
       Gravity Field Harmonics for small solar system bodies"
       pp. 30-36, 2000
    .. [4] George Dassios, "Ellipsoidal Harmonics: Theory and Applications"
       pp. 418, 2012

    Examples
    --------
    >>> from scipy.special import ellip_harm
    >>> w = ellip_harm(5,8,1,1,2.5)
    >>> w
    2.5

    Check that the functions indeed are solutions to the Lame equation:

    >>> import numpy as np
    >>> from scipy.interpolate import UnivariateSpline
    >>> def eigenvalue(f, df, ddf):
    ...     r = (((s**2 - h**2) * (s**2 - k**2) * ddf
    ...           + s * (2*s**2 - h**2 - k**2) * df
    ...           - n * (n + 1)*s**2*f) / f)
    ...     return -r.mean(), r.std()
    >>> s = np.linspace(0.1, 10, 200)
    >>> k, h, n, p = 8.0, 2.2, 3, 2
    >>> E = ellip_harm(h**2, k**2, n, p, s)
    >>> E_spl = UnivariateSpline(s, E)
    >>> a, a_err = eigenvalue(E_spl(s), E_spl(s,1), E_spl(s,2))
    >>> a, a_err
    (583.44366156701483, 6.4580890640310646e-11)

    """  # noqa: E501
    return _ellip_harm(h2, k2, n, p, s, signm, signn)


_ellip_harm_2_vec = np.vectorize(_ellipsoid, otypes='d')


def ellip_harm_2(h2, k2, n, p, s):
    r"""
    Ellipsoidal harmonic functions F^p_n(l)

    These are also known as Lame functions of the second kind, and are
    solutions to the Lame equation:

    .. math:: (s^2 - h^2)(s^2 - k^2)F''(s)
              + s(2s^2 - h^2 - k^2)F'(s) + (a - q s^2)F(s) = 0

    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not
    returned) corresponding to the solutions.

    Parameters
    ----------
    h2 : float
        ``h**2``
    k2 : float
        ``k**2``; should be larger than ``h**2``
    n : int
        Degree.
    p : int
        Order, can range between [1,2n+1].
    s : float
        Coordinate

    Returns
    -------
    F : float
        The harmonic :math:`F^p_n(s)`

    See Also
    --------
    ellip_harm, ellip_normal

    Notes
    -----
    Lame functions of the second kind are related to the functions of the first kind:

    .. math::

       F^p_n(s)=(2n + 1)E^p_n(s)\int_{0}^{1/s}
       \frac{du}{(E^p_n(1/u))^2\sqrt{(1-u^2k^2)(1-u^2h^2)}}

    .. versionadded:: 0.15.0

    Examples
    --------
    >>> from scipy.special import ellip_harm_2
    >>> w = ellip_harm_2(5,8,2,1,10)
    >>> w
    0.00108056853382

    """
    with np.errstate(all='ignore'):
        return _ellip_harm_2_vec(h2, k2, n, p, s)


def _ellip_normal_vec(h2, k2, n, p):
    return _ellipsoid_norm(h2, k2, n, p)


_ellip_normal_vec = np.vectorize(_ellip_normal_vec, otypes='d')


def ellip_normal(h2, k2, n, p):
    r"""
    Ellipsoidal harmonic normalization constants gamma^p_n

    The normalization constant is defined as

    .. math::

       \gamma^p_n=8\int_{0}^{h}dx\int_{h}^{k}dy
       \frac{(y^2-x^2)(E^p_n(y)E^p_n(x))^2}{\sqrt((k^2-y^2)(y^2-h^2)(h^2-x^2)(k^2-x^2)}

    Parameters
    ----------
    h2 : float
        ``h**2``
    k2 : float
        ``k**2``; should be larger than ``h**2``
    n : int
        Degree.
    p : int
        Order, can range between [1,2n+1].

    Returns
    -------
    gamma : float
        The normalization constant :math:`\gamma^p_n`

    See Also
    --------
    ellip_harm, ellip_harm_2

    Notes
    -----
    .. versionadded:: 0.15.0

    Examples
    --------
    >>> from scipy.special import ellip_normal
    >>> w = ellip_normal(5,8,3,7)
    >>> w
    1723.38796997

    """
    with np.errstate(all='ignore'):
        return _ellip_normal_vec(h2, k2, n, p)
<|endoftext|>from ._ufuncs import _lambertw

import numpy as np


def lambertw(z, k=0, tol=1e-8):
    r"""
    lambertw(z, k=0, tol=1e-8)

    Lambert W function.

    The Lambert W function `W(z)` is defined as the inverse function
    of ``w * exp(w)``. In other words, the value of ``W(z)`` is
    such that ``z = W(z) * exp(W(z))`` for any complex number
    ``z``.

    The Lambert W function is a multivalued function with infinitely
    many branches. Each branch gives a separate solution of the
    equation ``z = w exp(w)``. Here, the branches are indexed by the
    integer `k`.

    Parameters
    ----------
    z : array_like
        Input argument.
    k : int, optional
        Branch index.
    tol : float, optional
        Evaluation tolerance.

    Returns
    -------
    w : array
        `w` will have the same shape as `z`.

    See Also
    --------
    wrightomega : the Wright Omega function

    Notes
    -----
    All branches are supported by `lambertw`:

    * ``lambertw(z)`` gives the principal solution (branch 0)
    * ``lambertw(z, k)`` gives the solution on branch `k`

    The Lambert W function has two partially real branches: the
    principal branch (`k = 0`) is real for real ``z > -1/e``, and the
    ``k = -1`` branch is real for ``-1/e < z < 0``. All branches except
    ``k = 0`` have a logarithmic singularity at ``z = 0``.

    **Possible issues**

    The evaluation can become inaccurate very close to the branch point
    at ``-1/e``. In some corner cases, `lambertw` might currently
    fail to converge, or can end up on the wrong branch.

    **Algorithm**

    Halley's iteration is used to invert ``w * exp(w)``, using a first-order
    asymptotic approximation (O(log(w)) or `O(w)`) as the initial estimate.

    The definition, implementation and choice of branches is based on [2]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Lambert_W_function
    .. [2] Corless et al, "On the Lambert W function", Adv. Comp. Math. 5
       (1996) 329-359.
       https://cs.uwaterloo.ca/research/tr/1993/03/W.pdf

    Examples
    --------
    The Lambert W function is the inverse of ``w exp(w)``:

    >>> import numpy as np
    >>> from scipy.special import lambertw
    >>> w = lambertw(1)
    >>> w
    (0.56714329040978384+0j)
    >>> w * np.exp(w)
    (1.0+0j)

    Any branch gives a valid inverse:

    >>> w = lambertw(1, k=3)
    >>> w
    (-2.8535817554090377+17.113535539412148j)
    >>> w*np.exp(w)
    (1.0000000000000002+1.609823385706477e-15j)

    **Applications to equation-solving**

    The Lambert W function may be used to solve various kinds of
    equations.  We give two examples here.

    First, the function can be used to solve implicit equations of the
    form

        :math:`x = a + b e^{c x}`

    for :math:`x`.  We assume :math:`c` is not zero.  After a little
    algebra, the equation may be written

        :math:`z e^z = -b c e^{a c}`

    where :math:`z = c (a - x)`.  :math:`z` may then be expressed using
    the Lambert W function

        :math:`z = W(-b c e^{a c})`

    giving

        :math:`x = a - W(-b c e^{a c})/c`

    For example,

    >>> a = 3
    >>> b = 2
    >>> c = -0.5

    The solution to :math:`x = a + b e^{c x}` is:

    >>> x = a - lambertw(-b*c*np.exp(a*c))/c
    >>> x
    (3.3707498368978794+0j)

    Verify that it solves the equation:

    >>> a + b*np.exp(c*x)
    (3.37074983689788+0j)

    The Lambert W function may also be used find the value of the infinite
    power tower :math:`z^{z^{z^{\ldots}}}`:

    >>> def tower(z, n):
    ...     if n == 0:
    ...         return z
    ...     return z ** tower(z, n-1)
    ...
    >>> tower(0.5, 100)
    0.641185744504986
    >>> -lambertw(-np.log(0.5)) / np.log(0.5)
    (0.64118574450498589+0j)
    """
    # TODO: special expert should inspect this
    # interception; better place to do it?
    k = np.asarray(k, dtype=np.dtype("long"))
    return _lambertw(z, k, tol)
<|endoftext|>from functools import wraps
from scipy._lib._util import _lazywhere
import numpy as np
from ._ufuncs import (_spherical_jn, _spherical_yn, _spherical_in,
                      _spherical_kn, _spherical_jn_d, _spherical_yn_d,
                      _spherical_in_d, _spherical_kn_d)


def use_reflection(sign_n_even=None, reflection_fun=None):
    # - If reflection_fun is not specified, reflects negative `z` and multiplies
    #   output by appropriate sign (indicated by `sign_n_even`).
    # - If reflection_fun is specified, calls `reflection_fun` instead of `fun`.
    # See DLMF 10.47(v) https://dlmf.nist.gov/10.47
    def decorator(fun):
        def standard_reflection(n, z, derivative):
            # sign_n_even indicates the sign when the order `n` is even
            sign = np.where(n % 2 == 0, sign_n_even, -sign_n_even)
            # By the chain rule, differentiation at `-z` adds a minus sign
            sign = -sign if derivative else sign
            # Evaluate at positive z (minus negative z) and adjust the sign
            return fun(n, -z, derivative) * sign

        @wraps(fun)
        def wrapper(n, z, derivative=False):
            z = np.asarray(z)

            if np.issubdtype(z.dtype, np.complexfloating):
                return fun(n, z, derivative)  # complex dtype just works

            f2 = standard_reflection if reflection_fun is None else reflection_fun
            return _lazywhere(z.real >= 0, (n, z),
                              f=lambda n, z: fun(n, z, derivative),
                              f2=lambda n, z: f2(n, z, derivative))[()]
        return wrapper
    return decorator


@use_reflection(+1)  # See DLMF 10.47(v) https://dlmf.nist.gov/10.47
def spherical_jn(n, z, derivative=False):
    r"""Spherical Bessel function of the first kind or its derivative.

    Defined as [1]_,

    .. math:: j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n + 1/2}(z),

    where :math:`J_n` is the Bessel function of the first kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    jn : ndarray

    Notes
    -----
    For real arguments greater than the order, the function is computed
    using the ascending recurrence [2]_. For small real or complex
    arguments, the definitional relation to the cylindrical Bessel function
    of the first kind is used.

    The derivative is computed using the relations [3]_,

    .. math::
        j_n'(z) = j_{n-1}(z) - \frac{n + 1}{z} j_n(z).

        j_0'(z) = -j_1(z)


    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E3
    .. [2] https://dlmf.nist.gov/10.51.E1
    .. [3] https://dlmf.nist.gov/10.51.E2
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The spherical Bessel functions of the first kind :math:`j_n` accept
    both real and complex second argument. They can return a complex type:

    >>> from scipy.special import spherical_jn
    >>> spherical_jn(0, 3+5j)
    (-9.878987731663194-8.021894345786002j)
    >>> type(spherical_jn(0, 3+5j))
    <class 'numpy.complex128'>

    We can verify the relation for the derivative from the Notes
    for :math:`n=3` in the interval :math:`[1, 2]`:

    >>> import numpy as np
    >>> x = np.arange(1.0, 2.0, 0.01)
    >>> np.allclose(spherical_jn(3, x, True),
    ...             spherical_jn(2, x) - 4/x * spherical_jn(3, x))
    True

    The first few :math:`j_n` with real argument:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0.0, 10.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-0.5, 1.5)
    >>> ax.set_title(r'Spherical Bessel functions $j_n$')
    >>> for n in np.arange(0, 4):
    ...     ax.plot(x, spherical_jn(n, x), label=rf'$j_{n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    n = np.asarray(n, dtype=np.dtype("long"))
    if derivative:
        return _spherical_jn_d(n, z)
    else:
        return _spherical_jn(n, z)


@use_reflection(-1)  # See DLMF 10.47(v) https://dlmf.nist.gov/10.47
def spherical_yn(n, z, derivative=False):
    r"""Spherical Bessel function of the second kind or its derivative.

    Defined as [1]_,

    .. math:: y_n(z) = \sqrt{\frac{\pi}{2z}} Y_{n + 1/2}(z),

    where :math:`Y_n` is the Bessel function of the second kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    yn : ndarray

    Notes
    -----
    For real arguments, the function is computed using the ascending
    recurrence [2]_.  For complex arguments, the definitional relation to
    the cylindrical Bessel function of the second kind is used.

    The derivative is computed using the relations [3]_,

    .. math::
        y_n' = y_{n-1} - \frac{n + 1}{z} y_n.

        y_0' = -y_1


