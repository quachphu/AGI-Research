    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Cauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Cauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return torch.full(
            self._extended_shape(), nan, dtype=self.loc.dtype, device=self.loc.device
        )

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return torch.full(
            self._extended_shape(), inf, dtype=self.loc.dtype, device=self.loc.device
        )

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).cauchy_()
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            -math.log(math.pi)
            - self.scale.log()
            - (((value - self.loc) / self.scale) ** 2).log1p()
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.atan((value - self.loc) / self.scale) / math.pi + 0.5

    def icdf(self, value):
        return torch.tan(math.pi * (value - 0.5)) * self.scale + self.loc

    def entropy(self):
        return math.log(4 * math.pi) + self.scale.log()
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.gumbel import euler_constant
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all


__all__ = ["Weibull"]


class Weibull(TransformedDistribution):
    r"""
    Samples from a two-parameter Weibull distribution.

    Example:

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
        tensor([ 0.4784])

    Args:
        scale (float or Tensor): Scale parameter of distribution (lambda).
        concentration (float or Tensor): Concentration parameter of distribution (k/shape).
        validate_args (bool, optional): Whether to validate arguments. Default: None.
    """

    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.positive

    def __init__(
        self,
        scale: Tensor | float,
        concentration: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.scale, self.concentration = broadcast_all(scale, concentration)
        self.concentration_reciprocal = self.concentration.reciprocal()
        base_dist = Exponential(
            torch.ones_like(self.scale), validate_args=validate_args
        )
        transforms = [
            PowerTransform(exponent=self.concentration_reciprocal),
            AffineTransform(loc=0, scale=self.scale),
        ]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Weibull, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.concentration_reciprocal = new.concentration.reciprocal()
        base_dist = self.base_dist.expand(batch_shape)
        transforms = [
            PowerTransform(exponent=new.concentration_reciprocal),
            AffineTransform(loc=0, scale=new.scale),
        ]
        super(Weibull, new).__init__(base_dist, transforms, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.scale * torch.exp(torch.lgamma(1 + self.concentration_reciprocal))

    @property
    def mode(self) -> Tensor:
        return (
            self.scale
            * ((self.concentration - 1) / self.concentration)
            ** self.concentration.reciprocal()
        )

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) * (
            torch.exp(torch.lgamma(1 + 2 * self.concentration_reciprocal))
            - torch.exp(2 * torch.lgamma(1 + self.concentration_reciprocal))
        )

    def entropy(self):
        return (
            euler_constant * (1 - self.concentration_reciprocal)
            + torch.log(self.scale * self.concentration_reciprocal)
            + 1
        )
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import Categorical, constraints
from torch.distributions.constraints import MixtureSameFamilyConstraint
from torch.distributions.distribution import Distribution


__all__ = ["MixtureSameFamily"]


class MixtureSameFamily(Distribution):
    r"""
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all components are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` components) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.

    Examples::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        >>> # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)

        >>> # Construct Gaussian Mixture Model in 2D consisting of 5 equally
        >>> # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
        ...          torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

        >>> # Construct a batch of 3 Gaussian Mixture Models in 2D each
        >>> # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
        ...         torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)

    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting components.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    """

    arg_constraints: dict[str, constraints.Constraint] = {}
    has_rsample = False

    def __init__(
        self,
        mixture_distribution: Categorical,
        component_distribution: Distribution,
        validate_args: bool | None = None,
    ) -> None:
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution

        if not isinstance(self._mixture_distribution, Categorical):
            raise ValueError(
                " The Mixture distribution needs to be an "
                " instance of torch.distributions.Categorical"
            )

        if not isinstance(self._component_distribution, Distribution):
            raise ValueError(
                "The Component distribution needs to be an "
                "instance of torch.distributions.Distribution"
            )

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError(
                    f"`mixture_distribution.batch_shape` ({mdbs}) is not "
                    "compatible with `component_distribution."
                    f"batch_shape`({cdbs})"
                )

        # Check that the number of mixture components matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError(
                f"`mixture_distribution component` ({km}) does not"
                " equal `component_distribution.batch_shape[-1]`"
                f" ({kc})"
            )
        self._num_component = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super().__init__(
            # pyrefly: ignore [bad-argument-type]
            batch_shape=cdbs,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        batch_shape_comp = batch_shape + (self._num_component,)
        new = self._get_checked_instance(MixtureSameFamily, _instance)
        new._component_distribution = self._component_distribution.expand(
            batch_shape_comp
        )
        new._mixture_distribution = self._mixture_distribution.expand(batch_shape)
        new._num_component = self._num_component
        new._event_ndims = self._event_ndims
        event_shape = new._component_distribution.event_shape
        super(MixtureSameFamily, new).__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def support(self):
        return MixtureSameFamilyConstraint(self._component_distribution.support)

    @property
    def mixture_distribution(self) -> Categorical:
        return self._mixture_distribution

    @property
    def component_distribution(self) -> Distribution:
        return self._component_distribution

    @property
    def mean(self) -> Tensor:
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(
            probs * self.component_distribution.mean, dim=-1 - self._event_ndims
        )  # [B, E]

    @property
    def variance(self) -> Tensor:
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(
            probs * self.component_distribution.variance, dim=-1 - self._event_ndims
        )
        var_cond_mean = torch.sum(
            probs * (self.component_distribution.mean - self._pad(self.mean)).pow(2.0),
            dim=-1 - self._event_ndims,
        )
        return mean_cond_var + var_cond_mean

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs

        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(
            self.mixture_distribution.logits, dim=-1
        )  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1))
            )
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es
            )

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = len(self.batch_shape)
        cat_batch_ndims = len(self.mixture_distribution.batch_shape)
        pad_ndims = 0 if cat_batch_ndims == 1 else dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(
            xs[:-1]
            + torch.Size(pad_ndims * [1])
            + xs[-1:]
            + torch.Size(self._event_ndims * [1])
        )
        return x

    def __repr__(self):
        args_string = (
            f"\n  {self.mixture_distribution},\n  {self.component_distribution}"
        )
        return "MixtureSameFamily" + "(" + args_string + ")"
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import inf, Tensor
from torch.distributions import Categorical, constraints
from torch.distributions.binomial import Binomial
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["Multinomial"]


class Multinomial(Distribution):
    r"""
    Creates a Multinomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). The innermost dimension of
    :attr:`probs` indexes over categories. All other dimensions index over batches.

    Note that :attr:`total_count` need not be specified if only :meth:`log_prob` is
    called (see example below)

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    -   :meth:`sample` requires a single shared `total_count` for all
        parameters and samples.
    -   :meth:`log_prob` allows different `total_count` for each parameter and
        sample.

    Example::

        >>> # xdoctest: +SKIP("FIXME: found invalid values")
        >>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
        >>> x = m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 21.,  24.,  30.,  25.])

        >>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
        tensor([-4.1338])

    Args:
        total_count (int): number of trials
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    total_count: int

    @property
    def mean(self) -> Tensor:
        return self.probs * self.total_count

    @property
    def variance(self) -> Tensor:
        return self.total_count * self.probs * (1 - self.probs)

    def __init__(
        self,
        total_count: int = 1,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if not isinstance(total_count, int):
            raise NotImplementedError("inhomogeneous total_count is not supported")
        self.total_count = total_count
        self._categorical = Categorical(probs=probs, logits=logits)
        self._binomial = Binomial(total_count=total_count, probs=self.probs)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Multinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count
        new._categorical = self._categorical.expand(batch_shape)
        super(Multinomial, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    # pyrefly: ignore [bad-override]
    def support(self):
        return constraints.multinomial(self.total_count)

    @property
    def logits(self) -> Tensor:
        return self._categorical.logits

    @property
    def probs(self) -> Tensor:
        return self._categorical.probs

    @property
    def param_shape(self) -> torch.Size:
        return self._categorical.param_shape

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        samples = self._categorical.sample(
            torch.Size((self.total_count,)) + sample_shape
        )
        # samples.shape is (total_count, sample_shape, batch_shape), need to change it to
        # (sample_shape, batch_shape, total_count)
        shifted_idx = list(range(samples.dim()))
        shifted_idx.append(shifted_idx.pop(0))
        samples = samples.permute(*shifted_idx)
        counts = samples.new(self._extended_shape(sample_shape)).zero_()
        counts.scatter_add_(-1, samples, torch.ones_like(samples))
        return counts.type_as(self.probs)

    def entropy(self):
        n = torch.tensor(self.total_count)

        cat_entropy = self._categorical.entropy()
        term1 = n * cat_entropy - torch.lgamma(n + 1)

        support = self._binomial.enumerate_support(expand=False)[1:]
        binomial_probs = torch.exp(self._binomial.log_prob(support))
        weights = torch.lgamma(support + 1)
        term2 = (binomial_probs * weights).sum([0, -1])

        return term1 + term2

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        logits = logits.clone(memory_format=torch.contiguous_format)
        log_factorial_n = torch.lgamma(value.sum(-1) + 1)
        log_factorial_xs = torch.lgamma(value + 1).sum(-1)
        logits[(value == 0) & (logits == -inf)] = 0
        log_powers = (logits * value).sum(-1)
        return log_factorial_n - log_factorial_xs + log_powers
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform
from torch.distributions.utils import broadcast_all, clamp_probs
from torch.types import _size


__all__ = ["ExpRelaxedCategorical", "RelaxedOneHotCategorical"]


class ExpRelaxedCategorical(Distribution):
    r"""
    Creates a ExpRelaxedCategorical parameterized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both).
    Returns the log of a point in the simplex. Based on the interface to
    :class:`OneHotCategorical`.

    Implementation based on [1].

    See also: :func:`torch.distributions.OneHotCategorical`

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al., 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al., 2017)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = (
        constraints.real_vector
    )  # The true support is actually a submanifold of this.
    has_rsample = True

    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        self._categorical = Categorical(probs, logits)
        self.temperature = temperature
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExpRelaxedCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        new._categorical = self._categorical.expand(batch_shape)
        super(ExpRelaxedCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def param_shape(self) -> torch.Size:
        return self._categorical.param_shape

    @property
    def logits(self) -> Tensor:
        return self._categorical.logits

    @property
    def probs(self) -> Tensor:
        return self._categorical.probs

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(
            torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        )
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels) / self.temperature
        return scores - scores.logsumexp(dim=-1, keepdim=True)

    def log_prob(self, value):
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        log_scale = torch.full_like(
            self.temperature, float(K)
        ).lgamma() - self.temperature.log().mul(-(K - 1))
        score = logits - value.mul(self.temperature)
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score + log_scale


class RelaxedOneHotCategorical(TransformedDistribution):
    r"""
    Creates a RelaxedOneHotCategorical distribution parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`.
    This is a relaxed version of the :class:`OneHotCategorical` distribution, so
    its samples are on simplex, and are reparametrizable.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedOneHotCategorical(torch.tensor([2.2]),
        ...                              torch.tensor([0.1, 0.2, 0.3, 0.4]))
        >>> m.sample()
        tensor([ 0.1294,  0.2324,  0.3859,  0.2523])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event
    """

    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    # pyrefly: ignore [bad-override]
    support = constraints.simplex
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: ExpRelaxedCategorical

    def __init__(
        self,
        temperature: Tensor,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = ExpRelaxedCategorical(
            temperature, probs, logits, validate_args=validate_args
        )
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedOneHotCategorical, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def temperature(self) -> Tensor:
        return self.base_dist.temperature

    @property
    def logits(self) -> Tensor:
        return self.base_dist.logits

    @property
    def probs(self) -> Tensor:
        return self.base_dist.probs
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all, euler_constant


__all__ = ["Kumaraswamy"]


def _moments(a, b, n):
    """
    Computes nth moment of Kumaraswamy using torch.lgamma
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)


class Kumaraswamy(TransformedDistribution):
    r"""
    Samples from a Kumaraswamy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution with concentration alpha=1 and beta=1
        tensor([ 0.1729])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Tensor | float,
        concentration0: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration1, self.concentration0 = broadcast_all(
            concentration1, concentration0
        )
        base_dist = Uniform(
            torch.full_like(self.concentration0, 0),
            torch.full_like(self.concentration0, 1),
            validate_args=validate_args,
        )
        transforms = [
            PowerTransform(exponent=self.concentration0.reciprocal()),
            AffineTransform(loc=1.0, scale=-1.0),
            PowerTransform(exponent=self.concentration1.reciprocal()),
        ]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Kumaraswamy, _instance)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self) -> Tensor:
        return _moments(self.concentration1, self.concentration0, 1)

    @property
    def mode(self) -> Tensor:
        # Evaluate in log-space for numerical stability.
        log_mode = (
            self.concentration0.reciprocal() * (-self.concentration0).log1p()
            - (-self.concentration0 * self.concentration1).log1p()
        )
        log_mode[(self.concentration0 < 1) | (self.concentration1 < 1)] = nan
        return log_mode.exp()

    @property
    def variance(self) -> Tensor:
        return _moments(self.concentration1, self.concentration0, 2) - torch.pow(
            self.mean, 2
        )

    def entropy(self):
        t1 = 1 - self.concentration1.reciprocal()
        t0 = 1 - self.concentration0.reciprocal()
        H0 = torch.digamma(self.concentration0 + 1) + euler_constant
        return (
            t0
            + t1 * H0
            - torch.log(self.concentration1)
            - torch.log(self.concentration0)
        )
<|endoftext|># mypy: allow-untyped-defs

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform


__all__ = ["LogNormal"]


class LogNormal(TransformedDistribution):
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.positive
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Normal

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self) -> Tensor:
        return self.base_dist.loc

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def mode(self) -> Tensor:
        return (self.loc - self.scale.square()).exp()

    @property
    def variance(self) -> Tensor:
        scale_sq = self.scale.pow(2)
        return scale_sq.expm1() * (2 * self.loc + scale_sq).exp()

    def entropy(self):
        return self.base_dist.entropy() + self.loc
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all, euler_constant
from torch.types import _Number


__all__ = ["Gumbel"]


class Gumbel(TransformedDistribution):
    r"""
    Samples from a Gumbel Distribution.

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
        tensor([ 1.0124])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.real

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = torch.finfo(self.loc.dtype)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps, validate_args=validate_args)
        else:
            base_dist = Uniform(
                torch.full_like(self.loc, finfo.tiny),
                torch.full_like(self.loc, 1 - finfo.eps),
                validate_args=validate_args,
            )
        transforms = [
            ExpTransform().inv,
            AffineTransform(loc=0, scale=-torch.ones_like(self.scale)),
            ExpTransform().inv,
            AffineTransform(loc=loc, scale=-self.scale),
        ]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gumbel, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    # Explicitly defining the log probability function for Gumbel due to precision issues
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (self.loc - value) / self.scale
        return (y - y.exp()) - self.scale.log()

    @property
    def mean(self) -> Tensor:
        return self.loc + self.scale * euler_constant

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def stddev(self) -> Tensor:
        return (math.pi / math.sqrt(6)) * self.scale

    @property
    def variance(self) -> Tensor:
        return self.stddev.pow(2)

    def entropy(self):
        return self.scale.log() + (1 + euler_constant)
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Laplace"]


class Laplace(Distribution):
    r"""
    Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # Laplace distributed with loc=0, scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return 2 * self.scale.pow(2)

    @property
    def stddev(self) -> Tensor:
        return (2**0.5) * self.scale

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Laplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Laplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        finfo = torch.finfo(self.loc.dtype)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] lack of support for .uniform_()
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device) * 2 - 1
            return self.loc - self.scale * u.sign() * torch.log1p(
                -u.abs().clamp(min=finfo.tiny)
            )
        u = self.loc.new(shape).uniform_(finfo.eps - 1, 1)
        # TODO: If we ever implement tensor.nextafter, below is what we want ideally.
        # u = self.loc.new(shape).uniform_(self.loc.nextafter(-.5, 0), .5)
        return self.loc - self.scale * u.sign() * torch.log1p(-u.abs())

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -torch.log(2 * self.scale) - torch.abs(value - self.loc) / self.scale

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 - 0.5 * (value - self.loc).sign() * torch.expm1(
            -(value - self.loc).abs() / self.scale
        )

    def icdf(self, value):
        term = value - 0.5
        return self.loc - self.scale * (term).sign() * torch.log1p(-2 * term.abs())

    def entropy(self):
        return 1 + torch.log(2 * self.scale)
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property
from torch.types import _size


__all__ = ["MultivariateNormal"]


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to be broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = (
        torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
    )  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    Id = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    L = torch.linalg.solve_triangular(L_inv, Id, upper=False)
    return L


class MultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\mathbf{\Sigma}`
    or a positive definite precision matrix :math:`\mathbf{\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal

    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.

        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc: Tensor,
        covariance_matrix: Tensor | None = None,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
            # pyrefly: ignore [read-only]
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                covariance_matrix.shape[:-2], loc.shape[:-1]
            )
            # pyrefly: ignore [read-only]
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
        else:
            if precision_matrix is None:
                raise AssertionError("precision_matrix is unexpectedly None")
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            batch_shape = torch.broadcast_shapes(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            # pyrefly: ignore [read-only]
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
        self.loc = loc.expand(batch_shape + (-1,))

        event_shape = self.loc.shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(MultivariateNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self) -> Tensor:
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self) -> Tensor:
        return torch.matmul(
            self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self) -> Tensor:
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return (
            self._unbroadcasted_scale_tril.pow(2)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.types import _size


__all__ = ["Dirichlet"]


# This helper is exposed for testing.
def _Dirichlet_backward(x, concentration, grad_output):
    total = concentration.sum(-1, True).expand_as(concentration)
    grad = torch._dirichlet_grad(x, concentration, total)
    return grad * (grad_output - (x * grad_output).sum(-1, True))


class _Dirichlet(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, concentration):
        x = torch._sample_dirichlet(concentration)
        ctx.save_for_backward(x, concentration)
        return x

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        x, concentration = ctx.saved_tensors
        return _Dirichlet_backward(x, concentration, grad_output)


class Dirichlet(ExponentialFamily):
    r"""
    Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]
        tensor([ 0.1046,  0.8954])

    Args:
        concentration (Tensor): concentration parameter of the distribution
            (often referred to as alpha)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    support = constraints.simplex
    has_rsample = True

    def __init__(
        self,
        concentration: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` parameter must be at least one-dimensional."
            )
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        super(Dirichlet, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        return _Dirichlet.apply(concentration)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration - 1.0, value).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.concentration.sum(-1, True)

    @property
    def mode(self) -> Tensor:
        concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        mode = concentrationm1 / concentrationm1.sum(-1, True)
        mask = (self.concentration < 1).all(dim=-1)
        mode[mask] = torch.nn.functional.one_hot(
            mode[mask].argmax(dim=-1), concentrationm1.shape[-1]
        ).to(mode)
        return mode

    @property
    def variance(self) -> Tensor:
        con0 = self.concentration.sum(-1, True)
        return (
            self.concentration
            * (con0 - self.concentration)
            / (con0.pow(2) * (con0 + 1))
        )

    def entropy(self):
        k = self.concentration.size(-1)
        a0 = self.concentration.sum(-1)
        return (
            torch.lgamma(self.concentration).sum(-1)
            - torch.lgamma(a0)
            - (k - a0) * torch.digamma(a0)
            - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)
        )

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (self.concentration,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
import torch.jit
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property


__all__ = ["VonMises"]


def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_I0_COEF_SMALL = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.360768e-1,
    0.45813e-2,
]
_I0_COEF_LARGE = [
    0.39894228,
    0.1328592e-1,
    0.225319e-2,
    -0.157565e-2,
    0.916281e-2,
    -0.2057706e-1,
    0.2635537e-1,
    -0.1647633e-1,
    0.392377e-2,
]
_I1_COEF_SMALL = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.2658733e-1,
    0.301532e-2,
    0.32411e-3,
]
_I1_COEF_LARGE = [
    0.39894228,
    -0.3988024e-1,
    -0.362018e-2,
    0.163801e-2,
    -0.1031555e-1,
    0.2282967e-1,
    -0.2895312e-1,
    0.1787654e-1,
    -0.420059e-2,
]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    """
    Returns ``log(I_order(x))`` for ``x > 0``,
    where `order` is either 0 or 1.
    """
    if order != 0 and order != 1:
        raise AssertionError(f"order must be 0 or 1, got {order}")

    # compute small solution
    y = x / 3.75
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    result = torch.where(x < 3.75, small, large)
    return result


@torch.jit.script_if_tracing
def _rejection_sample(loc, concentration, proposal_r, x):
    done = torch.zeros(x.shape, dtype=torch.bool, device=loc.device)
    # pyrefly: ignore [bad-assignment, missing-attribute]
    while not done.all():
        u = torch.rand((3,) + x.shape, dtype=loc.dtype, device=loc.device)
        u1, u2, u3 = u.unbind()
        z = torch.cos(math.pi * u1)
        f = (1 + proposal_r * z) / (proposal_r + z)
        c = concentration * (proposal_r - f)
        accept = ((c * (2 - c) - u2) > 0) | ((c / u2).log() + 1 - c >= 0)
        if accept.any():
            # pyrefly: ignore [no-matching-overload]
            x = torch.where(accept, (u3 - 0.5).sign() * f.acos(), x)
            done = done | accept
    return (x + math.pi + loc) % (2 * math.pi) - math.pi


class VonMises(Distribution):
    """
    A circular von Mises distribution.

    This implementation uses polar coordinates. The ``loc`` and ``value`` args
    can be any real number (to facilitate unconstrained optimization), but are
    interpreted as angles modulo 2 pi.

    Example::
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = VonMises(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # von Mises distributed with loc=1 and concentration=1
        tensor([1.9777])

    :param torch.Tensor loc: an angle in radians.
    :param torch.Tensor concentration: concentration parameter
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "concentration": constraints.positive}
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        loc: Tensor,
        concentration: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.concentration = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.concentration * torch.cos(value - self.loc)
        log_prob = (
            log_prob
            - math.log(2 * math.pi)
            - _log_modified_bessel_fn(self.concentration, order=0)
        )
        return log_prob

    @lazy_property
    def _loc(self) -> Tensor:
        return self.loc.to(torch.double)

    @lazy_property
    def _concentration(self) -> Tensor:
        return self.concentration.to(torch.double)

    @lazy_property
    def _proposal_r(self) -> Tensor:
        kappa = self._concentration
        tau = 1 + (1 + 4 * kappa**2).sqrt()
        rho = (tau - (2 * tau).sqrt()) / (2 * kappa)
        _proposal_r = (1 + rho**2) / (2 * rho)
        # second order Taylor expansion around 0 for small kappa
        _proposal_r_taylor = 1 / kappa + kappa
        return torch.where(kappa < 1e-5, _proposal_r_taylor, _proposal_r)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        """
        The sampling algorithm for the von Mises distribution is based on the
        following paper: D.J. Best and N.I. Fisher, "Efficient simulation of the
        von Mises distribution." Applied Statistics (1979): 152-157.

        Sampling is always done in double precision internally to avoid a hang
        in _rejection_sample() for small values of the concentration, which
        starts to happen for single precision around 1e-4 (see issue #88443).
        """
        shape = self._extended_shape(sample_shape)
        x = torch.empty(shape, dtype=self._loc.dtype, device=self.loc.device)
        return _rejection_sample(
            self._loc, self._concentration, self._proposal_r, x
        ).to(self.loc.dtype)

    def expand(self, batch_shape, _instance=None):
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get("_validate_args")
            loc = self.loc.expand(batch_shape)
            concentration = self.concentration.expand(batch_shape)
            return type(self)(loc, concentration, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """
        The provided mean is the circular one.
        """
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @lazy_property
    def variance(self) -> Tensor:  # type: ignore[override]
        """
        The provided variance is the circular one.
        """
        return (
            1
            - (
                _log_modified_bessel_fn(self.concentration, order=1)
                - _log_modified_bessel_fn(self.concentration, order=0)
            ).exp()
        )
<|endoftext|># mypy: allow-untyped-defs
r"""
PyTorch provides two global :class:`ConstraintRegistry` objects that link
:class:`~torch.distributions.constraints.Constraint` objects to
:class:`~torch.distributions.transforms.Transform` objects. These objects both
input constraints and return transforms, but they have different guarantees on
bijectivity.

1. ``biject_to(constraint)`` looks up a bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is guaranteed to have
   ``.bijective = True`` and should implement ``.log_abs_det_jacobian()``.
2. ``transform_to(constraint)`` looks up a not-necessarily bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is not guaranteed to
   implement ``.log_abs_det_jacobian()``.

The ``transform_to()`` registry is useful for performing unconstrained
optimization on constrained parameters of probability distributions, which are
indicated by each distribution's ``.arg_constraints`` dict. These transforms often
overparameterize a space in order to avoid rotation; they are thus more
suitable for coordinate-wise optimization algorithms like Adam::

    loc = torch.zeros(100, requires_grad=True)
    unconstrained = torch.zeros(100, requires_grad=True)
    scale = transform_to(Normal.arg_constraints["scale"])(unconstrained)
    loss = -Normal(loc, scale).log_prob(data).sum()

The ``biject_to()`` registry is useful for Hamiltonian Monte Carlo, where
samples from a probability distribution with constrained ``.support`` are
propagated in an unconstrained space, and algorithms are typically rotation
invariant.::

    dist = Exponential(rate)
    unconstrained = torch.zeros(100, requires_grad=True)
    sample = biject_to(dist.support)(unconstrained)
    potential_energy = -dist.log_prob(sample).sum()

.. note::

    An example where ``transform_to`` and ``biject_to`` differ is
    ``constraints.simplex``: ``transform_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.SoftmaxTransform` that simply
    exponentiates and normalizes its inputs; this is a cheap and mostly
    coordinate-wise operation appropriate for algorithms like SVI. In
    contrast, ``biject_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.StickBreakingTransform` that
    bijects its input down to a one-fewer-dimensional space; this a more
    expensive less numerically stable transform but is needed for algorithms
    like HMC.

The ``biject_to`` and ``transform_to`` objects can be extended by user-defined
constraints and transforms using their ``.register()`` method either as a
function on singleton constraints::

    transform_to.register(my_constraint, my_transform)

or as a decorator on parameterized constraints::

    @transform_to.register(MyConstraintClass)
    def my_factory(constraint):
        assert isinstance(constraint, MyConstraintClass)
        return MyTransform(constraint.param1, constraint.param2)

You can create your own registry by creating a new :class:`ConstraintRegistry`
object.
"""

from torch.distributions import constraints, transforms
from torch.types import _Number


__all__ = [
    "ConstraintRegistry",
    "biject_to",
    "transform_to",
]


class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        self._registry = {}
        super().__init__()

    def register(self, constraint, factory=None):
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
        # Support use as decorator.
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        # Support calling on singleton instances.
        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)

        if not isinstance(constraint, type) or not issubclass(
            constraint, constraints.Constraint
        ):
            raise TypeError(
                f"Expected constraint to be either a Constraint subclass or instance, but got {constraint}"
            )

        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint):
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints["scale"]
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)  # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        # Look up by Constraint subclass.
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                f"Cannot transform {type(constraint).__name__} constraints"
            ) from None
        return factory(constraint)


biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()


################################################################################
# Registration Table
################################################################################


@biject_to.register(constraints.real)
@transform_to.register(constraints.real)
def _transform_to_real(constraint):
    return transforms.identity_transform


@biject_to.register(constraints.independent)
def _biject_to_independent(constraint):
    base_transform = biject_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )


@transform_to.register(constraints.independent)
def _transform_to_independent(constraint):
    base_transform = transform_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )


@biject_to.register(constraints.positive)
@biject_to.register(constraints.nonnegative)
@transform_to.register(constraints.positive)
@transform_to.register(constraints.nonnegative)
def _transform_to_positive(constraint):
    return transforms.ExpTransform()


@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
@transform_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(constraint):
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.lower_bound, 1),
        ]
    )


@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint):
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.upper_bound, -1),
        ]
    )


@biject_to.register(constraints.interval)
@biject_to.register(constraints.half_open_interval)
@transform_to.register(constraints.interval)
@transform_to.register(constraints.half_open_interval)
def _transform_to_interval(constraint):
    # Handle the special case of the unit interval.
    lower_is_0 = (
        isinstance(constraint.lower_bound, _Number) and constraint.lower_bound == 0
    )
    upper_is_1 = (
        isinstance(constraint.upper_bound, _Number) and constraint.upper_bound == 1
    )
    if lower_is_0 and upper_is_1:
        return transforms.SigmoidTransform()

    loc = constraint.lower_bound
    scale = constraint.upper_bound - constraint.lower_bound
    return transforms.ComposeTransform(
        [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
    )


@biject_to.register(constraints.simplex)
def _biject_to_simplex(constraint):
    return transforms.StickBreakingTransform()


@transform_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return transforms.SoftmaxTransform()


# TODO define a bijection for LowerCholeskyTransform
@transform_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint):
    return transforms.LowerCholeskyTransform()


@transform_to.register(constraints.positive_definite)
@transform_to.register(constraints.positive_semidefinite)
def _transform_to_positive_definite(constraint):
    return transforms.PositiveDefiniteTransform()


@biject_to.register(constraints.corr_cholesky)
@transform_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    return transforms.CorrCholeskyTransform()


@biject_to.register(constraints.cat)
def _biject_to_cat(constraint):
    return transforms.CatTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


@transform_to.register(constraints.cat)
def _transform_to_cat(constraint):
    return transforms.CatTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


@biject_to.register(constraints.stack)
def _biject_to_stack(constraint):
    return transforms.StackTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim
    )


@transform_to.register(constraints.stack)
def _transform_to_stack(constraint):
    return transforms.StackTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim
    )
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)


__all__ = ["NegativeBinomial"]


class NegativeBinomial(Distribution):
    r"""
    Creates a Negative Binomial distribution, i.e. distribution
    of the number of successful independent and identical Bernoulli trials
    before :attr:`total_count` failures are achieved. The probability
    of success of each Bernoulli trial is :attr:`probs`.

    Args:
        total_count (float or Tensor): non-negative number of negative Bernoulli
            trials to stop, although the distribution is still valid for real
            valued count
        probs (Tensor): Event probabilities of success in the half open interval [0, 1)
        logits (Tensor): Event log-odds for probabilities of success
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "total_count": constraints.greater_than_eq(0),
        "probs": constraints.half_open_interval(0.0, 1.0),
        "logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Tensor | float,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            (
                self.total_count,
                # pyrefly: ignore [read-only]
                self.probs,
            ) = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.probs)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            (
                self.total_count,
                # pyrefly: ignore [read-only]
                self.logits,
            ) = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)

        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(NegativeBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self) -> Tensor:
        return self.total_count * torch.exp(self.logits)

    @property
    def mode(self) -> Tensor:
        return ((self.total_count - 1) * self.logits.exp()).floor().clamp(min=0.0)

    @property
    def variance(self) -> Tensor:
        return self.mean / torch.sigmoid(-self.logits)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        return self._param.size()

    @lazy_property
    def _gamma(self) -> Gamma:
        # Note we avoid validating because self.total_count can be zero.
        return Gamma(
            concentration=self.total_count,
            rate=torch.exp(-self.logits),
            validate_args=False,
        )

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_unnormalized_prob = self.total_count * F.logsigmoid(
            -self.logits
        ) + value * F.logsigmoid(self.logits)

        log_normalization = (
            -torch.lgamma(self.total_count + value)
            + torch.lgamma(1.0 + value)
            + torch.lgamma(self.total_count)
        )
        # The case self.total_count == 0 and value == 0 has probability 1 but
        # lgamma(0) is infinite. Handle this case separately using a function
        # that does not modify tensors in place to allow Jit compilation.
        log_normalization = log_normalization.masked_fill(
            self.total_count + value == 0.0, 0.0
        )

        return log_unnormalized_prob - log_normalization
<|endoftext|>from typing import Optional

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.utils import broadcast_all
from torch.types import _size


__all__ = ["Pareto"]


class Pareto(TransformedDistribution):
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    """

    arg_constraints = {"alpha": constraints.positive, "scale": constraints.positive}

    def __init__(
        self,
        scale: Tensor | float,
        alpha: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.scale, self.alpha = broadcast_all(scale, alpha)
        base_dist = Exponential(self.alpha, validate_args=validate_args)
        transforms = [ExpTransform(), AffineTransform(loc=0, scale=self.scale)]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(
        self, batch_shape: _size, _instance: Optional["Pareto"] = None
    ) -> "Pareto":
        new = self._get_checked_instance(Pareto, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self) -> Tensor:
        # mean is inf for alpha <= 1
        a = self.alpha.clamp(min=1)
        return a * self.scale / (a - 1)

    @property
    def mode(self) -> Tensor:
        return self.scale

    @property
    def variance(self) -> Tensor:
        # var is inf for alpha <= 2
        a = self.alpha.clamp(min=2)
        return self.scale.pow(2) * a / ((a - 1).pow(2) * (a - 2))

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        return constraints.greater_than_eq(self.scale)

    def entropy(self) -> Tensor:
        return (self.scale / self.alpha).log() + (1 + self.alpha.reciprocal())
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform


__all__ = ["HalfNormal"]


class HalfNormal(TransformedDistribution):
    r"""
    Creates a half-normal distribution parameterized by `scale` where::

        X ~ Normal(0, scale)
        Y = |X| ~ HalfNormal(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfNormal(torch.tensor([1.0]))
        >>> m.sample()  # half-normal distributed with scale=1
        tensor([ 0.1046])

    Args:
        scale (float or Tensor): scale of the full Normal distribution
    """

    arg_constraints = {"scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.nonnegative
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Normal

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = Normal(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.scale)

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) * (1 - 2 / math.pi)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        return self.base_dist.entropy() - math.log(2)
<|endoftext|>from collections.abc import Callable, Sequence
from functools import update_wrapper
from typing import Any, Final, Generic, overload, TypeVar

import torch
import torch.nn.functional as F
from torch import SymInt, Tensor
from torch.overrides import is_tensor_like
from torch.types import _dtype, _Number, Device, Number


euler_constant: Final[float] = 0.57721566490153286060  # Euler Mascheroni Constant

__all__ = [
    "broadcast_all",
    "logits_to_probs",
    "clamp_probs",
    "probs_to_logits",
    "lazy_property",
    "tril_matrix_to_vec",
    "vec_to_tril_matrix",
]


# FIXME: Use (*values: *Ts) -> tuple[Tensor for T in Ts] if Mapping-Type is ever added.
#   See https://github.com/python/typing/issues/1216#issuecomment-2126153831
def broadcast_all(*values: Tensor | Number) -> tuple[Tensor, ...]:
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:

    - `torch.*Tensor` instances are broadcasted as per :ref:`broadcasting-semantics`.
    - Number instances (scalars) are upcast to tensors having
      the same size and type as the first tensor passed to `values`.  If all the
      values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `Number`, `torch.*Tensor` or objects implementing __torch_function__)

    Raises:
        ValueError: if any of the values is not a `Number` instance,
            a `torch.*Tensor` instance, or an instance implementing __torch_function__
    """
    if not all(is_tensor_like(v) or isinstance(v, _Number) for v in values):
        raise ValueError(
            "Input arguments must all be instances of Number, "
            "torch.Tensor or objects implementing __torch_function__."
        )
    if not all(is_tensor_like(v) for v in values):
        options: dict[str, Any] = dict(dtype=torch.get_default_dtype())
        for value in values:
            if isinstance(value, torch.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [
            v if is_tensor_like(v) else torch.tensor(v, **options) for v in values
        ]
        return torch.broadcast_tensors(*new_values)
    return torch.broadcast_tensors(*values)


def _standard_normal(
    shape: Sequence[int | SymInt],
    dtype: _dtype | None,
    device: Device | None,
) -> Tensor:
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(
            torch.zeros(shape, dtype=dtype, device=device),
            torch.ones(shape, dtype=dtype, device=device),
        )
    return torch.empty(shape, dtype=dtype, device=device).normal_()


def _sum_rightmost(value: Tensor, dim: int) -> Tensor:
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits: Tensor, is_binary: bool = False) -> Tensor:
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def clamp_probs(probs: Tensor) -> Tensor:
    """Clamps the probabilities to be in the open interval `(0, 1)`.

    The probabilities would be clamped between `eps` and `1 - eps`,
    and `eps` would be the smallest representable positive number for the input data type.

    Args:
        probs (Tensor): A tensor of probabilities.

    Returns:
        Tensor: The clamped probabilities.

    Examples:
        >>> probs = torch.tensor([0.0, 0.5, 1.0])
        >>> clamp_probs(probs)
        tensor([1.1921e-07, 5.0000e-01, 1.0000e+00])

        >>> probs = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        >>> clamp_probs(probs)
        tensor([2.2204e-16, 5.0000e-01, 1.0000e+00], dtype=torch.float64)

    """
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def probs_to_logits(probs: Tensor, is_binary: bool = False) -> Tensor:
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)


T = TypeVar("T", contravariant=True)
R = TypeVar("R", covariant=True)


class lazy_property(Generic[T, R]):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """

    def __init__(self, wrapped: Callable[[T], R]) -> None:
        self.wrapped: Callable[[T], R] = wrapped
        update_wrapper(self, wrapped)  # type:ignore[arg-type]

    @overload
    def __get__(
        self, instance: None, obj_type: Any = None
    ) -> "_lazy_property_and_property[T, R]": ...

    @overload
    def __get__(self, instance: T, obj_type: Any = None) -> R: ...

    def __get__(
        self, instance: T | None, obj_type: Any = None
    ) -> "R | _lazy_property_and_property[T, R]":
        if instance is None:
            return _lazy_property_and_property(self.wrapped)
        with torch.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


class _lazy_property_and_property(lazy_property[T, R], property):
    """We want lazy properties to look like multiple things.

    * property when Sphinx autodoc looks
    * lazy_property when Distribution validate_args looks
    """

    def __init__(self, wrapped: Callable[[T], R]) -> None:
        property.__init__(self, wrapped)


def tril_matrix_to_vec(mat: Tensor, diag: int = 0) -> Tensor:
    r"""
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    n = mat.shape[-1]
    if not torch._C._get_tracing_state() and (diag < -n or diag >= n):
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n - 1}].")
    arange = torch.arange(n, device=mat.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    vec = mat[..., tril_mask]
    return vec


def vec_to_tril_matrix(vec: Tensor, diag: int = 0) -> Tensor:
    r"""
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # +ve root of D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0
    n = (
        -(1 + 2 * diag)
        + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5
    ) / 2
    eps = torch.finfo(vec.dtype).eps
    if not torch._C._get_tracing_state() and (round(n) - n > eps):
        raise ValueError(
            f"The size of last dimension is {vec.shape[-1]} which cannot be expressed as "
            + "the lower triangular part of a square D x D matrix."
        )
    n = round(n.item()) if isinstance(n, torch.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    arange = torch.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_mv
from torch.distributions.utils import _standard_normal, lazy_property
from torch.types import _size


__all__ = ["LowRankMultivariateNormal"]


def _batch_capacitance_tril(W, D):
    r"""
    Computes Cholesky of :math:`I + W.T @ inv(D) @ W` for a batch of matrices :math:`W`
    and a batch of vectors :math:`D`.
    """
    m = W.size(-1)
    Wt_Dinv = W.mT / D.unsqueeze(-2)
    K = torch.matmul(Wt_Dinv, W).contiguous()
    K.view(-1, m * m)[:, :: m + 1] += 1  # add identity matrix to K
    return torch.linalg.cholesky(K)


def _batch_lowrank_logdet(W, D, capacitance_tril):
    r"""
    Uses "matrix determinant lemma"::
        log|W @ W.T + D| = log|C| + log|D|,
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute
    the log determinant.
    """
    return 2 * capacitance_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) + D.log().sum(
        -1
    )


def _batch_lowrank_mahalanobis(W, D, x, capacitance_tril):
    r"""
    Uses "Woodbury matrix identity"::
        inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D),
    where :math:`C` is the capacitance matrix :math:`I + W.T @ inv(D) @ W`, to compute the squared
    Mahalanobis distance :math:`x.T @ inv(W @ W.T + D) @ x`.
    """
    Wt_Dinv = W.mT / D.unsqueeze(-2)
    Wt_Dinv_x = _batch_mv(Wt_Dinv, x)
    mahalanobis_term1 = (x.pow(2) / D).sum(-1)
    mahalanobis_term2 = _batch_mahalanobis(capacitance_tril, Wt_Dinv_x)
    return mahalanobis_term1 - mahalanobis_term2


class LowRankMultivariateNormal(Distribution):
    r"""
    Creates a multivariate normal distribution with covariance matrix having a low-rank form
    parameterized by :attr:`cov_factor` and :attr:`cov_diag`::

        covariance_matrix = cov_factor @ cov_factor.T + cov_diag

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LowRankMultivariateNormal(
        ...     torch.zeros(2), torch.tensor([[1.0], [0.0]]), torch.ones(2)
        ... )
        >>> m.sample()  # normally distributed with mean=`[0,0]`, cov_factor=`[[1],[0]]`, cov_diag=`[1,1]`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution with shape `batch_shape + event_shape`
        cov_factor (Tensor): factor part of low-rank form of covariance matrix with shape
            `batch_shape + event_shape + (rank,)`
        cov_diag (Tensor): diagonal part of low-rank form of covariance matrix with shape
            `batch_shape + event_shape`

    Note:
        The computation for determinant and inverse of covariance matrix is avoided when
        `cov_factor.shape[1] << cov_factor.shape[0]` thanks to `Woodbury matrix identity
        <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_ and
        `matrix determinant lemma <https://en.wikipedia.org/wiki/Matrix_determinant_lemma>`_.
        Thanks to these formulas, we just need to compute the determinant and inverse of
        the small size "capacitance" matrix::

            capacitance = I + cov_factor.T @ inv(cov_diag) @ cov_factor
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.independent(constraints.real, 2),
        "cov_diag": constraints.independent(constraints.positive, 1),
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc: Tensor,
        cov_factor: Tensor,
        cov_diag: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        event_shape = loc.shape[-1:]
        if cov_factor.dim() < 2:
            raise ValueError(
                "cov_factor must be at least two-dimensional, "
                "with optional leading batch dimensions"
            )
        if cov_factor.shape[-2:-1] != event_shape:
            raise ValueError(
                f"cov_factor must be a batch of matrices with shape {event_shape[0]} x m"
            )
        if cov_diag.shape[-1:] != event_shape:
            raise ValueError(
                f"cov_diag must be a batch of vectors with shape {event_shape}"
            )

        loc_ = loc.unsqueeze(-1)
        cov_diag_ = cov_diag.unsqueeze(-1)
        try:
            loc_, self.cov_factor, cov_diag_ = torch.broadcast_tensors(
                loc_, cov_factor, cov_diag_
            )
        except RuntimeError as e:
            raise ValueError(
                f"Incompatible batch shapes: loc {loc.shape}, cov_factor {cov_factor.shape}, cov_diag {cov_diag.shape}"
            ) from e
        self.loc = loc_[..., 0]
        self.cov_diag = cov_diag_[..., 0]
        batch_shape = self.loc.shape[:-1]

        self._unbroadcasted_cov_factor = cov_factor
        self._unbroadcasted_cov_diag = cov_diag
        self._capacitance_tril = _batch_capacitance_tril(cov_factor, cov_diag)
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LowRankMultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new.cov_diag = self.cov_diag.expand(loc_shape)
        new.cov_factor = self.cov_factor.expand(loc_shape + self.cov_factor.shape[-1:])
        new._unbroadcasted_cov_factor = self._unbroadcasted_cov_factor
        new._unbroadcasted_cov_diag = self._unbroadcasted_cov_diag
        new._capacitance_tril = self._capacitance_tril
        super(LowRankMultivariateNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @lazy_property
    def variance(self) -> Tensor:  # type: ignore[override]
        return (
            self._unbroadcasted_cov_factor.pow(2).sum(-1) + self._unbroadcasted_cov_diag
        ).expand(self._batch_shape + self._event_shape)

    @lazy_property
    def scale_tril(self) -> Tensor:
        # The following identity is used to increase the numerical computation stability
        # for Cholesky decomposition (see http://www.gaussianprocess.org/gpml/, Section 3.4.3):
        #     W @ W.T + D = D1/2 @ (I + D-1/2 @ W @ W.T @ D-1/2) @ D1/2
        # The matrix "I + D-1/2 @ W @ W.T @ D-1/2" has eigenvalues bounded from below by 1,
        # hence it is well-conditioned and safe to take Cholesky decomposition.
        n = self._event_shape[0]
        cov_diag_sqrt_unsqueeze = self._unbroadcasted_cov_diag.sqrt().unsqueeze(-1)
        Dinvsqrt_W = self._unbroadcasted_cov_factor / cov_diag_sqrt_unsqueeze
        K = torch.matmul(Dinvsqrt_W, Dinvsqrt_W.mT).contiguous()
        K.view(-1, n * n)[:, :: n + 1] += 1  # add identity matrix to K
        scale_tril = cov_diag_sqrt_unsqueeze * torch.linalg.cholesky(K)
        return scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self) -> Tensor:
        covariance_matrix = torch.matmul(
            self._unbroadcasted_cov_factor, self._unbroadcasted_cov_factor.mT
        ) + torch.diag_embed(self._unbroadcasted_cov_diag)
        return covariance_matrix.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def precision_matrix(self) -> Tensor:
        # We use "Woodbury matrix identity" to take advantage of low rank form::
        #     inv(W @ W.T + D) = inv(D) - inv(D) @ W @ inv(C) @ W.T @ inv(D)
        # where :math:`C` is the capacitance matrix.
        Wt_Dinv = (
            self._unbroadcasted_cov_factor.mT
            / self._unbroadcasted_cov_diag.unsqueeze(-2)
        )
        A = torch.linalg.solve_triangular(self._capacitance_tril, Wt_Dinv, upper=False)
        precision_matrix = (
            torch.diag_embed(self._unbroadcasted_cov_diag.reciprocal()) - A.mT @ A
        )
        return precision_matrix.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        W_shape = shape[:-1] + self.cov_factor.shape[-1:]
        eps_W = _standard_normal(W_shape, dtype=self.loc.dtype, device=self.loc.device)
        eps_D = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return (
            self.loc
            + _batch_mv(self._unbroadcasted_cov_factor, eps_W)
            + self._unbroadcasted_cov_diag.sqrt() * eps_D
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_lowrank_mahalanobis(
            self._unbroadcasted_cov_factor,
            self._unbroadcasted_cov_diag,
            diff,
            self._capacitance_tril,
        )
        log_det = _batch_lowrank_logdet(
            self._unbroadcasted_cov_factor,
            self._unbroadcasted_cov_diag,
            self._capacitance_tril,
        )
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + log_det + M)

    def entropy(self):
        log_det = _batch_lowrank_logdet(
            self._unbroadcasted_cov_factor,
            self._unbroadcasted_cov_diag,
            self._capacitance_tril,
        )
        H = 0.5 * (self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + log_det)
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _Number, Number


__all__ = ["Bernoulli"]


class Bernoulli(ExponentialFamily):
    r"""
    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Bernoulli(torch.tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
        validate_args (bool, optional): whether to validate arguments, None by default
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.boolean
    has_enumerate_support = True
    _mean_carrier_measure = 0

    def __init__(
        self,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, _Number)
            # pyrefly: ignore [read-only]
            (self.probs,) = broadcast_all(probs)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            is_scalar = isinstance(logits, _Number)
            # pyrefly: ignore [read-only]
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Bernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Bernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self) -> Tensor:
        return self.probs

    @property
    def mode(self) -> Tensor:
        mode = (self.probs >= 0.5).to(self.probs)
        mode[self.probs == 0.5] = nan
        return mode

    @property
    def variance(self) -> Tensor:
        return self.probs * (1 - self.probs)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.bernoulli(self.probs.expand(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return -binary_cross_entropy_with_logits(logits, value, reduction="none")

    def entropy(self):
        return binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )

    def enumerate_support(self, expand=True):
        values = torch.arange(2, dtype=self._param.dtype, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (torch.logit(self.probs),)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        return torch.log1p(torch.exp(x))
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["FisherSnedecor"]


class FisherSnedecor(Distribution):
    r"""
    Creates a Fisher-Snedecor distribution parameterized by :attr:`df1` and :attr:`df2`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
        tensor([ 0.2453])

    Args:
        df1 (float or Tensor): degrees of freedom parameter 1
        df2 (float or Tensor): degrees of freedom parameter 2
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"df1": constraints.positive, "df2": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        df1: Tensor | float,
        df2: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.df1, self.df2 = broadcast_all(df1, df2)
        self._gamma1 = Gamma(self.df1 * 0.5, self.df1)
        self._gamma2 = Gamma(self.df2 * 0.5, self.df2)

        if isinstance(df1, _Number) and isinstance(df2, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.df1.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(FisherSnedecor, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df1 = self.df1.expand(batch_shape)
        new.df2 = self.df2.expand(batch_shape)
        new._gamma1 = self._gamma1.expand(batch_shape)
        new._gamma2 = self._gamma2.expand(batch_shape)
        super(FisherSnedecor, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        df2 = self.df2.clone(memory_format=torch.contiguous_format)
        df2[df2 <= 2] = nan
        return df2 / (df2 - 2)

    @property
    def mode(self) -> Tensor:
        mode = (self.df1 - 2) / self.df1 * self.df2 / (self.df2 + 2)
        mode[self.df1 <= 2] = nan
        return mode

    @property
    def variance(self) -> Tensor:
        df2 = self.df2.clone(memory_format=torch.contiguous_format)
        df2[df2 <= 4] = nan
        return (
            2
            * df2.pow(2)
            * (self.df1 + df2 - 2)
            / (self.df1 * (df2 - 2).pow(2) * (df2 - 4))
        )

    def rsample(self, sample_shape: _size = torch.Size(())) -> Tensor:
        shape = self._extended_shape(sample_shape)
        #   X1 ~ Gamma(df1 / 2, 1 / df1), X2 ~ Gamma(df2 / 2, 1 / df2)
        #   Y = df2 * df1 * X1 / (df1 * df2 * X2) = X1 / X2 ~ F(df1, df2)
        X1 = self._gamma1.rsample(sample_shape).view(shape)
        X2 = self._gamma2.rsample(sample_shape).view(shape)
        tiny = torch.finfo(X2.dtype).tiny
        X2.clamp_(min=tiny)
        Y = X1 / X2
        Y.clamp_(min=tiny)
        return Y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        ct1 = self.df1 * 0.5
        ct2 = self.df2 * 0.5
        ct3 = self.df1 / self.df2
        t1 = (ct1 + ct2).lgamma() - ct1.lgamma() - ct2.lgamma()
        t2 = ct1 * ct3.log() + (ct1 - 1) * torch.log(value)
        t3 = (ct1 + ct2) * torch.log1p(ct3 * value)
        return t1 + t2 - t3
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Gamma"]


def _standard_gamma(concentration, generator=None):
    return torch._standard_gamma(concentration, generator=generator)


class Gamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate parameter of the distribution
            (often referred to as beta), rate = 1 / scale
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.rate

    @property
    def mode(self) -> Tensor:
        return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self) -> Tensor:
        return self.concentration / self.rate.pow(2)

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, _Number) and isinstance(rate, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Gamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(
        self,
        sample_shape: _size = torch.Size(),
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape, generator=generator)

    def rsample(
        self,
        sample_shape: _size = torch.Size(),
        generator: torch.Generator | None = None,
    ) -> Tensor:
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(
            self.concentration.expand(shape), generator=generator
        ) / self.rate.expand(shape)
        value.detach().clamp_(
            min=torch.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration, self.rate)
            + torch.xlogy(self.concentration - 1, value)
            - self.rate * value
            - torch.lgamma(self.concentration)
        )

    def entropy(self):
        return (
            self.concentration
            - torch.log(self.rate)
            + torch.lgamma(self.concentration)
            + (1.0 - self.concentration) * torch.digamma(self.concentration)
        )

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.concentration - 1, -self.rate)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.special.gammainc(self.concentration, self.rate * value)
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.transforms import ComposeTransform, Transform
from torch.distributions.utils import _sum_rightmost
from torch.types import _size


__all__ = ["TransformedDistribution"]


class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
        log p(Y) = log p(X) + log |det (dX/dY)|

    Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
    maximum shape of its base distribution and its transforms, since transforms
    can introduce correlations among events.

    An example for the usage of :class:`TransformedDistribution` would be::

        # Building a Logistic Distribution
        # X ~ Uniform(0, 1)
        # f = a + b * logit(X)
        # Y ~ f(X) ~ Logistic(a, b)
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
        logistic = TransformedDistribution(base_distribution, transforms)

    For more examples, please look at the implementations of
    :class:`~torch.distributions.gumbel.Gumbel`,
    :class:`~torch.distributions.half_cauchy.HalfCauchy`,
    :class:`~torch.distributions.half_normal.HalfNormal`,
    :class:`~torch.distributions.log_normal.LogNormal`,
    :class:`~torch.distributions.pareto.Pareto`,
    :class:`~torch.distributions.weibull.Weibull`,
    :class:`~torch.distributions.relaxed_bernoulli.RelaxedBernoulli` and
    :class:`~torch.distributions.relaxed_categorical.RelaxedOneHotCategorical`
    """

    arg_constraints: dict[str, constraints.Constraint] = {}

    def __init__(
        self,
        base_distribution: Distribution,
        transforms: Transform | list[Transform],
        validate_args: bool | None = None,
    ) -> None:
        if isinstance(transforms, Transform):
            self.transforms = [
                transforms,
            ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError(
                    "transforms must be a Transform or a list of Transforms"
                )
            self.transforms = transforms
        else:
            raise ValueError(
                f"transforms must be a Transform or list, but was {transforms}"
            )

        # Reshape base_distribution according to transforms.
        base_shape = base_distribution.batch_shape + base_distribution.event_shape
        base_event_dim = len(base_distribution.event_shape)
        transform = ComposeTransform(self.transforms)
        if len(base_shape) < transform.domain.event_dim:
            raise ValueError(
                f"base_distribution needs to have shape with size at least {transform.domain.event_dim}, but got {base_shape}."
            )
        forward_shape = transform.forward_shape(base_shape)
        expanded_base_shape = transform.inverse_shape(forward_shape)
        if base_shape != expanded_base_shape:
            base_batch_shape = expanded_base_shape[
                : len(expanded_base_shape) - base_event_dim
            ]
            base_distribution = base_distribution.expand(base_batch_shape)
        reinterpreted_batch_ndims = transform.domain.event_dim - base_event_dim
        if reinterpreted_batch_ndims > 0:
            base_distribution = Independent(
                base_distribution, reinterpreted_batch_ndims
            )
        self.base_dist = base_distribution

        # Compute shapes.
        transform_change_in_event_dim = (
            transform.codomain.event_dim - transform.domain.event_dim
        )
        event_dim = max(
            transform.codomain.event_dim,  # the transform is coupled
            base_event_dim + transform_change_in_event_dim,  # the base dist is coupled
        )
        if len(forward_shape) < event_dim:
            raise AssertionError(
                f"forward_shape length {len(forward_shape)} must be >= event_dim {event_dim}"
            )
        cut = len(forward_shape) - event_dim
        batch_shape = forward_shape[:cut]
        event_shape = forward_shape[cut:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TransformedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        shape = batch_shape + self.event_shape
        for t in reversed(self.transforms):
            shape = t.inverse_shape(shape)
        base_batch_shape = shape[: len(shape) - len(self.base_dist.event_shape)]
        new.base_dist = self.base_dist.expand(base_batch_shape)
        new.transforms = self.transforms
        super(TransformedDistribution, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def support(self):
        if not self.transforms:
            return self.base_dist.support
        support = self.transforms[-1].codomain
        if len(self.event_shape) > support.event_dim:
            support = constraints.independent(
                support, len(self.event_shape) - support.event_dim
            )
        return support

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        return self.base_dist.has_rsample

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob: Tensor | float = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            y = x

        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(y), event_dim - len(self.base_dist.event_shape)
        )
        return log_prob

    def _monotonize_cdf(self, value):
        """
        This conditionally flips ``value -> 1-value`` to ensure :meth:`cdf` is
        monotone increasing.
        """
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        if isinstance(sign, int) and sign == 1:
            return value
        return sign * (value - 0.5) + 0.5

    def cdf(self, value):
        """
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        """
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.cdf(value)
        value = self._monotonize_cdf(value)
        return value

    def icdf(self, value):
        """
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        """
        value = self._monotonize_cdf(value)
        value = self.base_dist.icdf(value)
        for transform in self.transforms:
            value = transform(value)
        return value
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Exponential"]


class Exponential(ExponentialFamily):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"rate": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.rate.reciprocal()

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.rate)

    @property
    def stddev(self) -> Tensor:
        return self.rate.reciprocal()

    @property
    def variance(self) -> Tensor:
        return self.rate.pow(-2)

    def __init__(
        self,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        (self.rate,) = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, _Number) else self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Exponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Exponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        return -torch.log1p(-value) / self.rate

    def entropy(self):
        return 1.0 - torch.log(self.rate)

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (-self.rate,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        return -torch.log(-x)
<|endoftext|>r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions. This allows the construction of stochastic computation
graphs and stochastic gradient estimators for optimization. This package
generally follows the design of the `TensorFlow Distributions`_ package.

.. _`TensorFlow Distributions`:
    https://arxiv.org/abs/1711.10604

It is not possible to directly backpropagate through random samples. However,
there are two main methods for creating surrogate functions that can be
backpropagated through. These are the score function estimator/likelihood ratio
estimator/REINFORCE and the pathwise derivative estimator. REINFORCE is commonly
seen as the basis for policy gradient methods in reinforcement learning, and the
pathwise derivative estimator is commonly seen in the reparameterization trick
in variational autoencoders. Whilst the score function only requires the value
of samples :math:`f(x)`, the pathwise derivative requires the derivative
:math:`f'(x)`. The next sections discuss these two in a reinforcement learning
example. For more details see
`Gradient Estimation Using Stochastic Computation Graphs`_ .

.. _`Gradient Estimation Using Stochastic Computation Graphs`:
     https://arxiv.org/abs/1506.05254

Score function
^^^^^^^^^^^^^^

When the probability density function is differentiable with respect to its
parameters, we only need :meth:`~torch.distributions.Distribution.sample` and
:meth:`~torch.distributions.Distribution.log_prob` to implement REINFORCE:

.. math::

    \Delta\theta  = \alpha r \nabla_\theta \log \pi_\theta(a \mid s)

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`\pi_\theta(a \mid s)` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi_\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimizers use gradient
descent, whilst the rule above assumes gradient ascent. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()

Pathwise derivative
^^^^^^^^^^^^^^^^^^^

The other way to implement these stochastic/policy gradients would be to use the
reparameterization trick from the
:meth:`~torch.distributions.Distribution.rsample` method, where the
parameterized random variable can be constructed via a parameterized
deterministic function of a parameter-free random variable. The reparameterized
sample therefore becomes differentiable. The code for implementing the pathwise
derivative would be as follows::

    params = policy_network(state)
    m = Normal(*params)
    # Any distribution with .has_rsample == True could work based on the application
    action = m.rsample()
    next_state, reward = env.step(action)  # Assuming that reward is differentiable
    loss = -reward
    loss.backward()
"""

from . import transforms
from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .chi2 import Chi2
from .constraint_registry import biject_to, transform_to
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .fishersnedecor import FisherSnedecor
from .gamma import Gamma
from .generalized_pareto import GeneralizedPareto
from .geometric import Geometric
from .gumbel import Gumbel
from .half_cauchy import HalfCauchy
from .half_normal import HalfNormal
from .independent import Independent
from .inverse_gamma import InverseGamma
from .kl import _add_kl_info, kl_divergence, register_kl
from .kumaraswamy import Kumaraswamy
from .laplace import Laplace
from .lkj_cholesky import LKJCholesky
from .log_normal import LogNormal
from .logistic_normal import LogisticNormal
from .lowrank_multivariate_normal import LowRankMultivariateNormal
from .mixture_same_family import MixtureSameFamily
from .multinomial import Multinomial
from .multivariate_normal import MultivariateNormal
from .negative_binomial import NegativeBinomial
from .normal import Normal
from .one_hot_categorical import OneHotCategorical, OneHotCategoricalStraightThrough
from .pareto import Pareto
from .poisson import Poisson
from .relaxed_bernoulli import RelaxedBernoulli
from .relaxed_categorical import RelaxedOneHotCategorical
from .studentT import StudentT
from .transformed_distribution import TransformedDistribution
from .transforms import *  # noqa: F403
from .uniform import Uniform
from .von_mises import VonMises
from .weibull import Weibull
from .wishart import Wishart


_add_kl_info()
del _add_kl_info

__all__ = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "Dirichlet",
    "Distribution",
    "Exponential",
    "ExponentialFamily",
    "FisherSnedecor",
    "Gamma",
    "GeneralizedPareto",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Independent",
    "InverseGamma",
    "Kumaraswamy",
    "LKJCholesky",
    "Laplace",
    "LogNormal",
    "LogisticNormal",
    "LowRankMultivariateNormal",
    "MixtureSameFamily",
    "Multinomial",
    "MultivariateNormal",
    "NegativeBinomial",
    "Normal",
    "OneHotCategorical",
    "OneHotCategoricalStraightThrough",
    "Pareto",
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    "StudentT",
    "Poisson",
    "Uniform",
    "VonMises",
    "Weibull",
    "Wishart",
    "TransformedDistribution",
    "biject_to",
    "kl_divergence",
    "register_kl",
    "transform_to",
]
__all__.extend(transforms.__all__)
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform


__all__ = ["HalfCauchy"]


class HalfCauchy(TransformedDistribution):
    r"""
    Creates a half-Cauchy distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    """

    arg_constraints = {"scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.nonnegative
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Cauchy

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = Cauchy(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfCauchy, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        return torch.full(
            self._extended_shape(),
            math.inf,
            dtype=self.scale.dtype,
            device=self.scale.device,
        )

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.scale)

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = torch.as_tensor(
            value, dtype=self.base_dist.scale.dtype, device=self.base_dist.scale.device
        )
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        return self.base_dist.entropy() - math.log(2)
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.types import _Number, _size


__all__ = ["Normal"]


class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def stddev(self) -> Tensor:
        return self.scale

    @property
    def variance(self) -> Tensor:
        return self.stddev.pow(2)

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        log_scale = (
            math.log(self.scale)
            if isinstance(self.scale, _Number)
            else self.scale.log()
        )
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (
            1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2))
        )

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)
<|endoftext|># mypy: allow-untyped-defs
import math
import warnings

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.distributions.utils import lazy_property
from torch.types import _Number, _size, Number


__all__ = ["Wishart"]

_log_2 = math.log(2)


def _mvdigamma(x: Tensor, p: int) -> Tensor:
    if not x.gt((p - 1) / 2).all():
        raise AssertionError("Wrong domain for multivariate digamma function.")
    return torch.digamma(
        x.unsqueeze(-1)
        - torch.arange(p, dtype=x.dtype, device=x.device).div(2).expand(x.shape + (-1,))
    ).sum(-1)


def _clamp_above_eps(x: Tensor) -> Tensor:
    # We assume positive input for this function
    return x.clamp(min=torch.finfo(x.dtype).eps)


class Wishart(ExponentialFamily):
    r"""
    Creates a Wishart distribution parameterized by a symmetric positive definite matrix :math:`\Sigma`,
    or its Cholesky decomposition :math:`\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top`

    Example:
        >>> # xdoctest: +SKIP("FIXME: scale_tril must be at least two-dimensional")
        >>> m = Wishart(torch.Tensor([2]), covariance_matrix=torch.eye(2))
        >>> m.sample()  # Wishart distributed with mean=`df * I` and
        >>> # variance(x_ij)=`df` for i != j and variance(x_ij)=`2 * df` for i == j

    Args:
        df (float or Tensor): real-valued parameter larger than the (dimension of Square matrix) - 1
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal
    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.
        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
        'torch.distributions.LKJCholesky' is a restricted Wishart distribution.[1]

    **References**

    [1] Wang, Z., Wu, Y. and Chu, H., 2018. `On equivalence of the LKJ distribution and the restricted Wishart distribution`.
    [2] Sawyer, S., 2007. `Wishart Distributions and Inverse-Wishart Sampling`.
    [3] Anderson, T. W., 2003. `An Introduction to Multivariate Statistical Analysis (3rd ed.)`.
    [4] Odell, P. L. & Feiveson, A. H., 1966. `A Numerical Procedure to Generate a Sample Covariance Matrix`. JASA, 61(313):199-203.
    [5] Ku, Y.-C. & Bloomfield, P., 2010. `Generating Random Wishart Matrices with Fractional Degrees of Freedom in OX`.
    """

    support = constraints.positive_definite
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def arg_constraints(self):
        return {
            "covariance_matrix": constraints.positive_definite,
            "precision_matrix": constraints.positive_definite,
            "scale_tril": constraints.lower_cholesky,
            "df": constraints.greater_than(self.event_shape[-1] - 1),
        }

    def __init__(
        self,
        df: Tensor | Number,
        covariance_matrix: Tensor | None = None,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (
            (covariance_matrix is not None)
            + (scale_tril is not None)
            + (precision_matrix is not None)
        ) != 1:
            raise AssertionError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        param = next(
            p
            for p in (covariance_matrix, precision_matrix, scale_tril)
            if p is not None
        )

        if param.dim() < 2:
            raise ValueError(
                "scale_tril must be at least two-dimensional, with optional leading batch dimensions"
            )

        if isinstance(df, _Number):
            batch_shape = torch.Size(param.shape[:-2])
            self.df = torch.tensor(df, dtype=param.dtype, device=param.device)
        else:
            batch_shape = torch.broadcast_shapes(param.shape[:-2], df.shape)
            self.df = df.expand(batch_shape)
        event_shape = param.shape[-2:]

        if self.df.le(event_shape[-1] - 1).any():
            raise ValueError(
                f"Value of df={df} expected to be greater than ndim - 1 = {event_shape[-1] - 1}."
            )

        if scale_tril is not None:
            # pyrefly: ignore [read-only]
            self.scale_tril = param.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            # pyrefly: ignore [read-only]
            self.covariance_matrix = param.expand(batch_shape + (-1, -1))
        elif precision_matrix is not None:
            # pyrefly: ignore [read-only]
            self.precision_matrix = param.expand(batch_shape + (-1, -1))

        if self.df.lt(event_shape[-1]).any():
            warnings.warn(
                "Low df values detected. Singular samples are highly likely to occur for ndim - 1 < df < ndim.",
                stacklevel=2,
            )

        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        self._batch_dims = [-(x + 1) for x in range(len(self._batch_shape))]

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

        # Chi2 distribution is needed for Bartlett decomposition sampling
        self._dist_chi2 = torch.distributions.chi2.Chi2(
            df=(
                self.df.unsqueeze(-1)
                - torch.arange(
                    self._event_shape[-1],
                    dtype=self._unbroadcasted_scale_tril.dtype,
                    device=self._unbroadcasted_scale_tril.device,
                ).expand(batch_shape + (-1,))
            )
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Wishart, _instance)
        batch_shape = torch.Size(batch_shape)
        cov_shape = batch_shape + self.event_shape
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril.expand(cov_shape)
        new.df = self.df.expand(batch_shape)

        new._batch_dims = [-(x + 1) for x in range(len(batch_shape))]

        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)

        # Chi2 distribution is needed for Bartlett decomposition sampling
        new._dist_chi2 = torch.distributions.chi2.Chi2(
            df=(
                new.df.unsqueeze(-1)
                - torch.arange(
                    self.event_shape[-1],
                    dtype=new._unbroadcasted_scale_tril.dtype,
                    device=new._unbroadcasted_scale_tril.device,
                ).expand(batch_shape + (-1,))
            )
        )

        super(Wishart, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self) -> Tensor:
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self) -> Tensor:
        return (
            self._unbroadcasted_scale_tril
            @ self._unbroadcasted_scale_tril.transpose(-2, -1)
        ).expand(self._batch_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self) -> Tensor:
        identity = torch.eye(
            self._event_shape[-1],
            device=self._unbroadcasted_scale_tril.device,
            dtype=self._unbroadcasted_scale_tril.dtype,
        )
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape
        )

    @property
    def mean(self) -> Tensor:
        return self.df.view(self._batch_shape + (1, 1)) * self.covariance_matrix

    @property
    def mode(self) -> Tensor:
        factor = self.df - self.covariance_matrix.shape[-1] - 1
        factor[factor <= 0] = nan
        return factor.view(self._batch_shape + (1, 1)) * self.covariance_matrix

    @property
    def variance(self) -> Tensor:
        V = self.covariance_matrix  # has shape (batch_shape x event_shape)
        diag_V = V.diagonal(dim1=-2, dim2=-1)
        return self.df.view(self._batch_shape + (1, 1)) * (
            V.pow(2) + torch.einsum("...i,...j->...ij", diag_V, diag_V)
        )

    def _bartlett_sampling(self, sample_shape=torch.Size()):
        p = self._event_shape[-1]  # has singleton shape

        # Implemented Sampling using Bartlett decomposition
        noise = _clamp_above_eps(
            self._dist_chi2.rsample(sample_shape).sqrt()
        ).diag_embed(dim1=-2, dim2=-1)

        i, j = torch.tril_indices(p, p, offset=-1)
        noise[..., i, j] = torch.randn(
            torch.Size(sample_shape) + self._batch_shape + (int(p * (p - 1) / 2),),
            dtype=noise.dtype,
            device=noise.device,
        )
        chol = self._unbroadcasted_scale_tril @ noise
        return chol @ chol.transpose(-2, -1)

    def rsample(
        self, sample_shape: _size = torch.Size(), max_try_correction=None
    ) -> Tensor:
        r"""
        .. warning::
            In some cases, sampling algorithm based on Bartlett decomposition may return singular matrix samples.
            Several tries to correct singular samples are performed by default, but it may end up returning
            singular matrix samples. Singular samples may return `-inf` values in `.log_prob()`.
            In those cases, the user should validate the samples and either fix the value of `df`
            or adjust `max_try_correction` value for argument in `.rsample` accordingly.
        """

        if max_try_correction is None:
            max_try_correction = 3 if torch._C._get_tracing_state() else 10

        sample_shape = torch.Size(sample_shape)
        sample = self._bartlett_sampling(sample_shape)

        # Below part is to improve numerical stability temporarily and should be removed in the future
        is_singular = self.support.check(sample)
        if self._batch_shape:
            is_singular = is_singular.amax(self._batch_dims)

        if torch._C._get_tracing_state():
            # Less optimized version for JIT
            for _ in range(max_try_correction):
                sample_new = self._bartlett_sampling(sample_shape)
                sample = torch.where(is_singular, sample_new, sample)

                is_singular = ~self.support.check(sample)
                if self._batch_shape:
                    is_singular = is_singular.amax(self._batch_dims)

        else:
            # More optimized version with data-dependent control flow.
            if is_singular.any():
                warnings.warn("Singular sample detected.", stacklevel=2)

                for _ in range(max_try_correction):
                    sample_new = self._bartlett_sampling(is_singular[is_singular].shape)
                    sample[is_singular] = sample_new

                    is_singular_new = ~self.support.check(sample_new)
                    if self._batch_shape:
                        is_singular_new = is_singular_new.amax(self._batch_dims)
                    is_singular[is_singular.clone()] = is_singular_new

                    if not is_singular.any():
                        break

        return sample

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        return (
            -nu
            * (
                p * _log_2 / 2
                + self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1)
                .log()
                .sum(-1)
            )
            - torch.mvlgamma(nu / 2, p=p)
            + (nu - p - 1) / 2 * torch.linalg.slogdet(value).logabsdet
            - torch.cholesky_solve(value, self._unbroadcasted_scale_tril)
            .diagonal(dim1=-2, dim2=-1)
            .sum(dim=-1)
            / 2
        )

    def entropy(self):
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        return (
            (p + 1)
            * (
                p * _log_2 / 2
                + self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1)
                .log()
                .sum(-1)
            )
            + torch.mvlgamma(nu / 2, p=p)
            - (nu - p - 1) / 2 * _mvdigamma(nu / 2, p=p)
            + nu * p / 2
        )

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        return -self.precision_matrix / 2, (nu - p - 1) / 2

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        p = self._event_shape[-1]
        return (y + (p + 1) / 2) * (
            -torch.linalg.slogdet(-2 * x).logabsdet + _log_2 * p
        ) + torch.mvlgamma(y + (p + 1) / 2, p=p)
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Uniform"]


class Uniform(Distribution):
    r"""
    Generates uniformly distributed random samples from the half-open interval
    ``[low, high)``.

    Example::

        >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
        >>> # xdoctest: +SKIP
        tensor([ 2.3418])

    Args:
        low (float or Tensor): lower range (inclusive).
        high (float or Tensor): upper range (exclusive).
    """

    has_rsample = True

    @property
    def arg_constraints(self):
        # TODO allow (loc,scale) parameterization to allow independent constraints.
        return {
            "low": constraints.less_than(self.high),
            "high": constraints.greater_than(self.low),
        }

    @property
    def mean(self) -> Tensor:
        return (self.high + self.low) / 2

    @property
    def mode(self) -> Tensor:
        return nan * self.high

    @property
    def stddev(self) -> Tensor:
        return (self.high - self.low) / 12**0.5

    @property
    def variance(self) -> Tensor:
        return (self.high - self.low).pow(2) / 12

    def __init__(
        self,
        low: Tensor | float,
        high: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.low, self.high = broadcast_all(low, high)

        if isinstance(low, _Number) and isinstance(high, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Uniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(Uniform, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        return constraints.interval(self.low, self.high)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        lb = self.low.le(value).type_as(self.low)
        ub = self.high.gt(value).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        result = (value - self.low) / (self.high - self.low)
        return result.clamp(min=0, max=1)

    def icdf(self, value):
        result = value * (self.high - self.low) + self.low
        return result

    def entropy(self):
        return torch.log(self.high - self.low)
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution


__all__ = ["ExponentialFamily"]


class ExponentialFamily(Distribution):
    r"""
    ExponentialFamily is the abstract base class for probability distributions belonging to an
    exponential family, whose probability mass/density function is defined below

    .. math::

        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))

    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,
    :math:`F(\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier
    measure.

    Note:
        This class is an intermediary between the `Distribution` class and distributions which belong
        to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL
        divergence methods. We use this class to compute the entropy and KL divergence using the AD
        framework and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and
        Cross-entropies of Exponential Families).
    """

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        """
        Abstract method for natural parameters. Returns a tuple of Tensors based
        on the distribution
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        """
        Abstract method for log normalizer function. Returns a log normalizer based on
        the distribution and input
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self) -> float:
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
        raise NotImplementedError

    def entropy(self):
        """
        Method to compute the entropy using Bregman divergence of the log normalizer.
        """
        result: Tensor | float = -self._mean_carrier_measure
        nparams = [p.detach().requires_grad_() for p in self._natural_params]
        lg_normal = self._log_normalizer(*nparams)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        result += lg_normal
        for np, g in zip(nparams, gradients):
            result -= (np * g).reshape(self._batch_shape + (-1,)).sum(-1)
        return result
<|endoftext|># mypy: allow-untyped-defs

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.gamma import Gamma


__all__ = ["Chi2"]


class Chi2(Gamma):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi2(torch.tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    """

    arg_constraints = {"df": constraints.positive}

    def __init__(
        self,
        df: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        super().__init__(0.5 * df, 0.5, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Chi2, _instance)
        return super().expand(batch_shape, new)

    @property
    def df(self) -> Tensor:
        return self.concentration * 2
<|endoftext|># mypy: allow-untyped-defs
from typing import Generic, TypeVar

import torch
from torch import Size, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _sum_rightmost
from torch.types import _size


__all__ = ["Independent"]


D = TypeVar("D", bound=Distribution)


class Independent(Distribution, Generic[D]):
    r"""
    Reinterprets some of the batch dims of a distribution as event dims.

    This is mainly useful for changing the shape of the result of
    :meth:`log_prob`. For example to create a diagonal Normal distribution with
    the same shape as a Multivariate Normal distribution (so they are
    interchangeable), you can::

        >>> from torch.distributions.multivariate_normal import MultivariateNormal
        >>> from torch.distributions.normal import Normal
        >>> loc = torch.zeros(3)
        >>> scale = torch.ones(3)
        >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        >>> [mvn.batch_shape, mvn.event_shape]
        [torch.Size([]), torch.Size([3])]
        >>> normal = Normal(loc, scale)
        >>> [normal.batch_shape, normal.event_shape]
        [torch.Size([3]), torch.Size([])]
        >>> diagn = Independent(normal, 1)
        >>> [diagn.batch_shape, diagn.event_shape]
        [torch.Size([]), torch.Size([3])]

    Args:
        base_distribution (torch.distributions.distribution.Distribution): a
            base distribution
        reinterpreted_batch_ndims (int): the number of batch dims to
            reinterpret as event dims
    """

    arg_constraints: dict[str, constraints.Constraint] = {}
    base_dist: D

    def __init__(
        self,
        base_distribution: D,
        reinterpreted_batch_ndims: int,
        validate_args: bool | None = None,
    ) -> None:
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                "Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                f"actual {reinterpreted_batch_ndims} vs {len(base_distribution.batch_shape)}"
            )
        shape: Size = base_distribution.batch_shape + base_distribution.event_shape
        event_dim: int = reinterpreted_batch_ndims + len(base_distribution.event_shape)
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Independent, _instance)
        batch_shape = torch.Size(batch_shape)
        new.base_dist = self.base_dist.expand(
            batch_shape + self.event_shape[: self.reinterpreted_batch_ndims]
        )
        new.reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        super(Independent, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self) -> bool:  # type: ignore[override]
        if self.reinterpreted_batch_ndims > 0:
            return False
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def support(self):
        result = self.base_dist.support
        if self.reinterpreted_batch_ndims:
            result = constraints.independent(result, self.reinterpreted_batch_ndims)
        return result

    @property
    def mean(self) -> Tensor:
        return self.base_dist.mean

    @property
    def mode(self) -> Tensor:
        return self.base_dist.mode

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        return _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return _sum_rightmost(entropy, self.reinterpreted_batch_ndims)

    def enumerate_support(self, expand=True):
        if self.reinterpreted_batch_ndims > 0:
            raise NotImplementedError(
                "Enumeration over cartesian product is not implemented"
            )
        return self.base_dist.enumerate_support(expand=expand)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"({self.base_dist}, {self.reinterpreted_batch_ndims})"
        )
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _Number, _size, Number


__all__ = ["ContinuousBernoulli"]


class ContinuousBernoulli(ExponentialFamily):
    r"""
    Creates a continuous Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    The distribution is supported in [0, 1] and parameterized by 'probs' (in
    (0,1)) or 'logits' (real-valued). Note that, unlike the Bernoulli, 'probs'
    does not correspond to a probability and 'logits' does not correspond to
    log-odds, but the same names are used due to the similarity with the
    Bernoulli. See [1] for more details.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = ContinuousBernoulli(torch.tensor([0.3]))
        >>> m.sample()
        tensor([ 0.2538])

    Args:
        probs (Number, Tensor): (0,1) valued parameters
        logits (Number, Tensor): real valued parameters whose sigmoid matches 'probs'

    [1] The continuous Bernoulli: fixing a pervasive error in variational
    autoencoders, Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    https://arxiv.org/abs/1907.06845
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.unit_interval
    _mean_carrier_measure = 0
    has_rsample = True

    def __init__(
        self,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        lims: tuple[float, float] = (0.499, 0.501),
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, _Number)
            # pyrefly: ignore [read-only]
            (self.probs,) = broadcast_all(probs)
            # validate 'probs' here if necessary as it is later clamped for numerical stability
            # close to 0 and 1, later on; otherwise the clamped 'probs' would always pass
            if validate_args is not None:
                if not self.arg_constraints["probs"].check(self.probs).all():
                    raise ValueError("The parameter probs has invalid values")
            # pyrefly: ignore [read-only]
            self.probs = clamp_probs(self.probs)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            is_scalar = isinstance(logits, _Number)
            # pyrefly: ignore [read-only]
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        self._lims = lims
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousBernoulli, _instance)
        new._lims = self._lims
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(ContinuousBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    def _outside_unstable_region(self):
        return torch.max(
            torch.le(self.probs, self._lims[0]), torch.gt(self.probs, self._lims[1])
        )

    def _cut_probs(self):
        return torch.where(
            self._outside_unstable_region(),
            self.probs,
            self._lims[0] * torch.ones_like(self.probs),
        )

    def _cont_bern_log_norm(self):
        """computes the log normalizing constant as a function of the 'probs' parameter"""
        cut_probs = self._cut_probs()
        cut_probs_below_half = torch.where(
            torch.le(cut_probs, 0.5), cut_probs, torch.zeros_like(cut_probs)
        )
        cut_probs_above_half = torch.where(
            torch.ge(cut_probs, 0.5), cut_probs, torch.ones_like(cut_probs)
        )
        log_norm = torch.log(
            torch.abs(torch.log1p(-cut_probs) - torch.log(cut_probs))
        ) - torch.where(
            torch.le(cut_probs, 0.5),
            torch.log1p(-2.0 * cut_probs_below_half),
            torch.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = torch.pow(self.probs - 0.5, 2)
        taylor = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return torch.where(self._outside_unstable_region(), log_norm, taylor)

    @property
    def mean(self) -> Tensor:
        cut_probs = self._cut_probs()
        mus = cut_probs / (2.0 * cut_probs - 1.0) + 1.0 / (
            torch.log1p(-cut_probs) - torch.log(cut_probs)
        )
        x = self.probs - 0.5
        taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * torch.pow(x, 2)) * x
        return torch.where(self._outside_unstable_region(), mus, taylor)

    @property
    def stddev(self) -> Tensor:
        return torch.sqrt(self.variance)

    @property
    def variance(self) -> Tensor:
        cut_probs = self._cut_probs()
        vars = cut_probs * (cut_probs - 1.0) / torch.pow(
            1.0 - 2.0 * cut_probs, 2
        ) + 1.0 / torch.pow(torch.log1p(-cut_probs) - torch.log(cut_probs), 2)
        x = torch.pow(self.probs - 0.5, 2)
        taylor = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        return torch.where(self._outside_unstable_region(), vars, taylor)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return clamp_probs(logits_to_probs(self.logits, is_binary=True))

    @property
    def param_shape(self) -> torch.Size:
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        with torch.no_grad():
            return self.icdf(u)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return (
            -binary_cross_entropy_with_logits(logits, value, reduction="none")
            + self._cont_bern_log_norm()
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        cut_probs = self._cut_probs()
        cdfs = (
            torch.pow(cut_probs, value) * torch.pow(1.0 - cut_probs, 1.0 - value)
            + cut_probs
            - 1.0
        ) / (2.0 * cut_probs - 1.0)
        unbounded_cdfs = torch.where(self._outside_unstable_region(), cdfs, value)
        return torch.where(
            torch.le(value, 0.0),
            torch.zeros_like(value),
            torch.where(torch.ge(value, 1.0), torch.ones_like(value), unbounded_cdfs),
        )

    def icdf(self, value):
        cut_probs = self._cut_probs()
        return torch.where(
            self._outside_unstable_region(),
            (
                torch.log1p(-cut_probs + value * (2.0 * cut_probs - 1.0))
                - torch.log1p(-cut_probs)
            )
            / (torch.log(cut_probs) - torch.log1p(-cut_probs)),
            value,
        )

    def entropy(self):
        log_probs0 = torch.log1p(-self.probs)
        log_probs1 = torch.log(self.probs)
        return (
            self.mean * (log_probs0 - log_probs1)
            - self._cont_bern_log_norm()
            - log_probs0
        )

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (self.logits,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        """computes the log normalizing constant as a function of the natural parameter"""
        out_unst_reg = torch.max(
            torch.le(x, self._lims[0] - 0.5), torch.gt(x, self._lims[1] - 0.5)
        )
        cut_nat_params = torch.where(
            out_unst_reg, x, (self._lims[0] - 0.5) * torch.ones_like(x)
        )
        log_norm = torch.log(
            torch.abs(torch.special.expm1(cut_nat_params))
        ) - torch.log(torch.abs(cut_nat_params))
        taylor = 0.5 * x + torch.pow(x, 2) / 24.0 - torch.pow(x, 4) / 2880.0
        return torch.where(out_unst_reg, log_norm, taylor)
<|endoftext|># mypy: allow-untyped-defs

from collections.abc import Callable
from typing import Any


r"""
The following constraints are implemented:

- ``constraints.boolean``
- ``constraints.cat``
- ``constraints.corr_cholesky``
- ``constraints.dependent``
- ``constraints.greater_than(lower_bound)``
- ``constraints.greater_than_eq(lower_bound)``
- ``constraints.independent(constraint, reinterpreted_batch_ndims)``
- ``constraints.integer_interval(lower_bound, upper_bound)``
- ``constraints.interval(lower_bound, upper_bound)``
- ``constraints.less_than(upper_bound)``
- ``constraints.lower_cholesky``
- ``constraints.lower_triangular``
- ``constraints.MixtureSameFamilyConstraint(base_constraint)``
- ``constraints.multinomial``
- ``constraints.nonnegative``
- ``constraints.nonnegative_integer``
- ``constraints.one_hot``
- ``constraints.positive_integer``
- ``constraints.positive``
- ``constraints.positive_semidefinite``
- ``constraints.positive_definite``
- ``constraints.real_vector``
- ``constraints.real``
- ``constraints.simplex``
- ``constraints.symmetric``
- ``constraints.stack``
- ``constraints.square``
- ``constraints.symmetric``
- ``constraints.unit_interval``
"""

import torch


__all__ = [
    "Constraint",
    "boolean",
    "cat",
    "corr_cholesky",
    "dependent",
    "dependent_property",
    "greater_than",
    "greater_than_eq",
    "independent",
    "integer_interval",
    "interval",
    "half_open_interval",
    "is_dependent",
    "less_than",
    "lower_cholesky",
    "lower_triangular",
    "MixtureSameFamilyConstraint",
    "multinomial",
    "nonnegative",
    "nonnegative_integer",
    "one_hot",
    "positive",
    "positive_semidefinite",
    "positive_definite",
    "positive_integer",
    "real",
    "real_vector",
    "simplex",
    "square",
    "stack",
    "symmetric",
    "unit_interval",
]


class Constraint:
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.

    Attributes:
        is_discrete (bool): Whether constrained space is discrete.
            Defaults to False.
        event_dim (int): Number of rightmost dimensions that together define
            an event. The :meth:`check` method will remove this many dimensions
            when computing validity.
    """

    is_discrete = False  # Default to continuous.
    event_dim = 0  # Default to univariate.

    def check(self, value):
        """
        Returns a byte tensor of ``sample_shape + batch_shape`` indicating
        whether each event in value satisfies this constraint.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__[1:] + "()"


class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.

    Args:
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """

    def __init__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        self._is_discrete = is_discrete
        self._event_dim = event_dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        if self._is_discrete is NotImplemented:
            raise NotImplementedError(".is_discrete cannot be determined statically")
        return self._is_discrete

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        if self._event_dim is NotImplemented:
            raise NotImplementedError(".event_dim cannot be determined statically")
        return self._event_dim

    def __call__(self, *, is_discrete=NotImplemented, event_dim=NotImplemented):
        """
        Support for syntax to customize static attributes::

            constraints.dependent(is_discrete=True, event_dim=1)
        """
        if is_discrete is NotImplemented:
            is_discrete = self._is_discrete
        if event_dim is NotImplemented:
            event_dim = self._event_dim
        return _Dependent(is_discrete=is_discrete, event_dim=event_dim)

    def check(self, x):
        raise ValueError("Cannot determine validity of dependent constraint")


def is_dependent(constraint):
    """
    Checks if ``constraint`` is a ``_Dependent`` object.

    Args:
        constraint : A ``Constraint`` object.

    Returns:
        ``bool``: True if ``constraint`` can be refined to the type ``_Dependent``, False otherwise.

    Examples:
        >>> import torch
        >>> from torch.distributions import Bernoulli
        >>> from torch.distributions.constraints import is_dependent

        >>> dist = Bernoulli(probs=torch.tensor([0.6], requires_grad=True))
        >>> constraint1 = dist.arg_constraints["probs"]
        >>> constraint2 = dist.arg_constraints["logits"]

        >>> for constraint in [constraint1, constraint2]:
        >>>     if is_dependent(constraint):
        >>>         continue
    """
    return isinstance(constraint, _Dependent)


class _DependentProperty(property, _Dependent):
    """
    Decorator that extends @property to act like a `Dependent` constraint when
    called on a class and act like a property when called on an object.

    Example::

        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high

            @constraints.dependent_property(is_discrete=False, event_dim=0)
            def support(self):
                return constraints.interval(self.low, self.high)

    Args:
        fn (Callable): The function to be decorated.
        is_discrete (bool): Optional value of ``.is_discrete`` in case this
            can be computed statically. If not provided, access to the
            ``.is_discrete`` attribute will raise a NotImplementedError.
        event_dim (int): Optional value of ``.event_dim`` in case this
            can be computed statically. If not provided, access to the
            ``.event_dim`` attribute will raise a NotImplementedError.
    """

    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        is_discrete: bool | None = NotImplemented,
        event_dim: int | None = NotImplemented,
    ) -> None:
        super().__init__(fn)
        self._is_discrete = is_discrete
        self._event_dim = event_dim

    def __call__(self, fn: Callable[..., Any]) -> "_DependentProperty":  # type: ignore[override]
        """
        Support for syntax to customize static attributes::

            @constraints.dependent_property(is_discrete=True, event_dim=1)
            def support(self): ...
        """
        return _DependentProperty(
            fn, is_discrete=self._is_discrete, event_dim=self._event_dim
        )


class _IndependentConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        if not isinstance(base_constraint, Constraint):
            raise AssertionError(
                f"base_constraint must be a Constraint, got {type(base_constraint).__name__}"
            )
        if not isinstance(reinterpreted_batch_ndims, int):
            raise AssertionError(
                f"reinterpreted_batch_ndims must be an int, got {type(reinterpreted_batch_ndims).__name__}"
            )
        if reinterpreted_batch_ndims < 0:
            raise AssertionError(
                f"reinterpreted_batch_ndims must be >= 0, got {reinterpreted_batch_ndims}"
            )
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        return self.base_constraint.is_discrete

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        return self.base_constraint.event_dim + self.reinterpreted_batch_ndims

    def check(self, value):
        result = self.base_constraint.check(value)
        if result.dim() < self.reinterpreted_batch_ndims:
            expected = self.base_constraint.event_dim + self.reinterpreted_batch_ndims
            raise ValueError(
                f"Expected value.dim() >= {expected} but got {value.dim()}"
            )
        result = result.reshape(
            result.shape[: result.dim() - self.reinterpreted_batch_ndims] + (-1,)
        )
        result = result.all(-1)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__[1:]}({repr(self.base_constraint)}, {self.reinterpreted_batch_ndims})"


class MixtureSameFamilyConstraint(Constraint):
    """
    Constraint for the :class:`~torch.distributions.MixtureSameFamily`
    distribution that adds back the rightmost batch dimension before
    performing the validity check with the component distribution
    constraint.

    Args:
        base_constraint: The ``Constraint`` object of
            the component distribution of
            the :class:`~torch.distributions.MixtureSameFamily` distribution.
    """

    def __init__(self, base_constraint):
        if not isinstance(base_constraint, Constraint):
            raise AssertionError(
                f"base_constraint must be a Constraint, got {type(base_constraint).__name__}"
            )
        self.base_constraint = base_constraint
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        return self.base_constraint.is_discrete

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        return self.base_constraint.event_dim

    def check(self, value):
        """
        Check validity of ``value`` as a possible outcome of sampling
        the :class:`~torch.distributions.MixtureSameFamily` distribution.
        """
        unsqueezed_value = value.unsqueeze(-1 - self.event_dim)
        result = self.base_constraint.check(unsqueezed_value)
        if value.dim() < self.event_dim:
            raise ValueError(
                f"Expected value.dim() >= {self.event_dim} but got {value.dim()}"
            )
        num_dim_to_keep = value.dim() - self.event_dim
        result = result.reshape(result.shape[:num_dim_to_keep] + (-1,))
        result = result.all(-1)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.base_constraint)})"


class _Boolean(Constraint):
    """
    Constrain to the two values `{0, 1}`.
    """

    is_discrete = True

    def check(self, value):
        return (value == 0) | (value == 1)


class _OneHot(Constraint):
    """
    Constrain to one-hot vectors.
    """

    is_discrete = True
    event_dim = 1

    def check(self, value):
        is_boolean = (value == 0) | (value == 1)
        is_normalized = value.sum(-1).eq(1)
        return is_boolean.all(-1) & is_normalized


class _IntegerInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound]`.
    """

    is_discrete = True

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (
            (value % 1 == 0) & (self.lower_bound <= value) & (value <= self.upper_bound)
        )

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )
        return fmt_string


class _IntegerLessThan(Constraint):
    """
    Constrain to an integer interval `(-inf, upper_bound]`.
    """

    is_discrete = True

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (value % 1 == 0) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(upper_bound={self.upper_bound})"
        return fmt_string


class _IntegerGreaterThan(Constraint):
    """
    Constrain to an integer interval `[lower_bound, inf)`.
    """

    is_discrete = True

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        return (value % 1 == 0) & (value >= self.lower_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound})"
        return fmt_string


class _Real(Constraint):
    """
    Trivially constrain to the extended real line `[-inf, inf]`.
    """

    def check(self, value):
        return value == value  # False for NANs.


class _GreaterThan(Constraint):
    """
    Constrain to a real half line `(lower_bound, inf]`.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        return self.lower_bound < value

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound})"
        return fmt_string


class _GreaterThanEq(Constraint):
    """
    Constrain to a real half line `[lower_bound, inf)`.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        return self.lower_bound <= value

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound})"
        return fmt_string


class _LessThan(Constraint):
    """
    Constrain to a real half line `[-inf, upper_bound)`.
    """

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return value < self.upper_bound

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(upper_bound={self.upper_bound})"
        return fmt_string


class _Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`.
    """

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (self.lower_bound <= value) & (value <= self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )
        return fmt_string


class _HalfOpenInterval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound)`.
    """

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (self.lower_bound <= value) & (value < self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += (
            f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        )
        return fmt_string


class _Simplex(Constraint):
    """
    Constrain to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """

    event_dim = 1

    def check(self, value):
        return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)


class _Multinomial(Constraint):
    """
    Constrain to nonnegative integer values summing to at most an upper bound.

    Note due to limitations of the Multinomial distribution, this currently
    checks the weaker condition ``value.sum(-1) <= upper_bound``. In the future
    this may be strengthened to ``value.sum(-1) == upper_bound``.
    """

    is_discrete = True
    event_dim = 1

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound

    def check(self, x):
        return (x >= 0).all(dim=-1) & (x.sum(dim=-1) <= self.upper_bound)


class _LowerTriangular(Constraint):
    """
    Constrain to lower-triangular square matrices.
    """

    event_dim = 2

    def check(self, value):
        value_tril = value.tril()
        return (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]


class _LowerCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals.
    """

    event_dim = 2

    def check(self, value):
        value_tril = value.tril()
        lower_triangular = (
            (value_tril == value).view(value.shape[:-2] + (-1,)).min(-1)[0]
        )

        positive_diagonal = (value.diagonal(dim1=-2, dim2=-1) > 0).min(-1)[0]
        return lower_triangular & positive_diagonal


class _CorrCholesky(Constraint):
    """
    Constrain to lower-triangular square matrices with positive diagonals and each
    row vector being of unit length.
    """

    event_dim = 2

    def check(self, value):
        tol = (
            torch.finfo(value.dtype).eps * value.size(-1) * 10
        )  # 10 is an adjustable fudge factor
        row_norm = torch.linalg.norm(value.detach(), dim=-1)
        unit_row_norm = (row_norm - 1.0).abs().le(tol).all(dim=-1)
        return _LowerCholesky().check(value) & unit_row_norm


class _Square(Constraint):
    """
    Constrain to square matrices.
    """

    event_dim = 2

    def check(self, value):
        return torch.full(
            size=value.shape[:-2],
            fill_value=(value.shape[-2] == value.shape[-1]),
            dtype=torch.bool,
            device=value.device,
        )


class _Symmetric(_Square):
    """
    Constrain to Symmetric square matrices.
    """

    def check(self, value):
        square_check = super().check(value)
        if not square_check.all():
            return square_check
        return torch.isclose(value, value.mT, atol=1e-6).all(-2).all(-1)


class _PositiveSemidefinite(_Symmetric):
    """
    Constrain to positive-semidefinite matrices.
    """

    def check(self, value):
        sym_check = super().check(value)
        if not sym_check.all():
            return sym_check
        return torch.linalg.eigvalsh(value).ge(0).all(-1)


class _PositiveDefinite(_Symmetric):
    """
    Constrain to positive-definite matrices.
    """

    def check(self, value):
        sym_check = super().check(value)
        if not sym_check.all():
            return sym_check
        return torch.linalg.cholesky_ex(value).info.eq(0)


class _Cat(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    each of size `lengths[dim]`, in a way compatible with :func:`torch.cat`.
    """

    def __init__(self, cseq, dim=0, lengths=None):
        if not all(isinstance(c, Constraint) for c in cseq):
            raise AssertionError("All elements of cseq must be Constraint instances")
        self.cseq = list(cseq)
        if lengths is None:
            lengths = [1] * len(self.cseq)
        self.lengths = list(lengths)
        if len(self.lengths) != len(self.cseq):
            raise AssertionError(
                f"lengths ({len(self.lengths)}) must match cseq ({len(self.cseq)})"
            )
        self.dim = dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        return any(c.is_discrete for c in self.cseq)

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        return max(c.event_dim for c in self.cseq)

    def check(self, value):
        if not (-value.dim() <= self.dim < value.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for value with {value.dim()} dimensions"
            )
        checks = []
        start = 0
        for constr, length in zip(self.cseq, self.lengths):
            v = value.narrow(self.dim, start, length)
            checks.append(constr.check(v))
            start = start + length  # avoid += for jit compat
        return torch.cat(checks, self.dim)


class _Stack(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `cseq` at the submatrices at dimension `dim`,
    in a way compatible with :func:`torch.stack`.
    """

    def __init__(self, cseq, dim=0):
        if not all(isinstance(c, Constraint) for c in cseq):
            raise AssertionError("All elements of cseq must be Constraint instances")
        self.cseq = list(cseq)
        self.dim = dim
        super().__init__()

    @property
    def is_discrete(self) -> bool:  # type: ignore[override]
        return any(c.is_discrete for c in self.cseq)

    @property
    def event_dim(self) -> int:  # type: ignore[override]
        dim = max(c.event_dim for c in self.cseq)
        if self.dim + dim < 0:
            dim += 1
        return dim

    def check(self, value):
        if not (-value.dim() <= self.dim < value.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for value with {value.dim()} dimensions"
            )
        vs = [value.select(self.dim, i) for i in range(value.size(self.dim))]
        return torch.stack(
            [constr.check(v) for v, constr in zip(vs, self.cseq)], self.dim
        )


# Public interface.
dependent = _Dependent()
dependent_property = _DependentProperty
independent = _IndependentConstraint
boolean = _Boolean()
one_hot = _OneHot()
nonnegative_integer = _IntegerGreaterThan(0)
positive_integer = _IntegerGreaterThan(1)
integer_interval = _IntegerInterval
real = _Real()
real_vector = independent(real, 1)
positive = _GreaterThan(0.0)
nonnegative = _GreaterThanEq(0.0)
greater_than = _GreaterThan
greater_than_eq = _GreaterThanEq
less_than = _LessThan
multinomial = _Multinomial
unit_interval = _Interval(0.0, 1.0)
interval = _Interval
half_open_interval = _HalfOpenInterval
simplex = _Simplex()
lower_triangular = _LowerTriangular()
lower_cholesky = _LowerCholesky()
corr_cholesky = _CorrCholesky()
square = _Square()
symmetric = _Symmetric()
positive_semidefinite = _PositiveSemidefinite()
positive_definite = _PositiveDefinite()
cat = _Cat
stack = _Stack
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from torch.types import _size


__all__ = ["OneHotCategorical", "OneHotCategoricalStraightThrough"]


class OneHotCategorical(Distribution):
    r"""
    Creates a one-hot categorical distribution parameterized by :attr:`probs` or
    :attr:`logits`.

    Samples are one-hot coded vectors of size ``probs.size(-1)``.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.distributions.Categorical` for specifications of
    :attr:`probs` and :attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 0.,  0.,  0.,  1.])

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = constraints.one_hot
    has_enumerate_support = True

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        self._categorical = Categorical(probs, logits)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(OneHotCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        new._categorical = self._categorical.expand(batch_shape)
        super(OneHotCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def _param(self) -> Tensor:
        return self._categorical._param

    @property
    def probs(self) -> Tensor:
        return self._categorical.probs

    @property
    def logits(self) -> Tensor:
        return self._categorical.logits

    @property
    def mean(self) -> Tensor:
        return self._categorical.probs

    @property
    def mode(self) -> Tensor:
        probs = self._categorical.probs
        mode = probs.argmax(dim=-1)
        return torch.nn.functional.one_hot(mode, num_classes=probs.shape[-1]).to(probs)

    @property
    def variance(self) -> Tensor:
        return self._categorical.probs * (1 - self._categorical.probs)

    @property
    def param_shape(self) -> torch.Size:
        return self._categorical.param_shape

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        probs = self._categorical.probs
        num_events = self._categorical._num_events
        indices = self._categorical.sample(sample_shape)
        return torch.nn.functional.one_hot(indices, num_events).to(probs)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        indices = value.max(-1)[1]
        return self._categorical.log_prob(indices)

    def entropy(self):
        return self._categorical.entropy()

    def enumerate_support(self, expand=True):
        n = self.event_shape[0]
        values = torch.eye(n, dtype=self._param.dtype, device=self._param.device)
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        if expand:
            values = values.expand((n,) + self.batch_shape + (n,))
        return values


class OneHotCategoricalStraightThrough(OneHotCategorical):
    r"""
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al., 2013)
    """

    has_rsample = True

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        samples = self.sample(sample_shape)
        probs = self._categorical.probs  # cached via @lazy_property
        return samples + (probs - probs.detach())
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits


__all__ = ["Categorical"]


