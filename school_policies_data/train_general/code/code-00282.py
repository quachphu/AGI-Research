    cls_to_become = BatchNorm2d  # type: ignore[assignment]

    def _check_input_dim(self, input) -> None:
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class BatchNorm3d(_BatchNorm):
    r"""Applies Batch Normalization over a 5D input.

    5D is a mini-batch of 3D inputs with additional channel dimension as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, correction=0)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased estimator, equivalent to
    ``torch.var(input, correction=1)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _check_input_dim(self, input) -> None:
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")


class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
    r"""A :class:`torch.nn.BatchNorm3d` module with lazy initialization.

    Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm3d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``
    """

    cls_to_become = BatchNorm3d  # type: ignore[assignment]

    def _check_input_dim(self, input) -> None:
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")


class SyncBatchNorm(_BatchNorm):
    r"""Applies Batch Normalization over a N-Dimensional input.

    The N-D input is a mini-batch of [N-2]D inputs with additional channel dimension as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, correction=0)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done for each channel in the ``C`` dimension, computing
    statistics on ``(N, +)`` slices, it's common terminology to call this Volumetric Batch
    Normalization or Spatio-temporal Batch Normalization.

    Currently :class:`SyncBatchNorm` only supports
    :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`torch.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm*D` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)

    .. note::
        Synchronization of batchnorm statistics occurs only while training, i.e.
        synchronization is disabled when ``model.eval()`` is set or if
        ``self.training`` is otherwise ``False``.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With Learnable Parameters
        >>> m = nn.SyncBatchNorm(100)
        >>> # creating process group (optional)
        >>> # ranks is a list of int identifying rank ids.
        >>> ranks = list(range(8))
        >>> r1, r2 = ranks[:4], ranks[4:]
        >>> # Note: every rank calls into new_group for every
        >>> # process group created, even if that rank is not
        >>> # part of the group.
        >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
        >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

        >>> # network is nn.BatchNorm layer
        >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
        >>> # only single gpu per process is currently supported
        >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        >>>                         sync_bn_network,
        >>>                         device_ids=[args.local_rank],
        >>>                         output_device=args.local_rank)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Any | None = None,
        device=None,
        dtype=None,
        *,
        bias: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            **factory_kwargs,
            bias=bias,
        )
        self.process_group = process_group

    def _check_input_dim(self, input) -> None:
        if input.dim() < 2:
            raise ValueError(f"expected at least 2D input (got {input.dim()}D input)")

    def _check_non_zero_input_channels(self, input) -> None:
        if input.size(1) == 0:
            raise ValueError(
                "SyncBatchNorm number of input channels should be non-zero"
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is None:
                raise AssertionError("num_batches_tracked must not be None")
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (
            bn_training
            and self.training
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if need_sync:
            # currently only GPU/PrivateUse1 input is supported
            if input.device.type not in [
                "cuda",
                "hpu",
                "xpu",
                torch._C._get_privateuse1_backend_name(),
            ]:
                raise ValueError(
                    "SyncBatchNorm expected input tensor to be on GPU or XPU or "
                    f"{torch._C._get_privateuse1_backend_name()}"
                )

            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            if not bn_training:
                raise AssertionError("bn_training must be True")
            return sync_batch_norm.apply(
                input,
                self.weight,
                self.bias,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,  # type: ignore[possibly-undefined]
                world_size,  # type: ignore[possibly-undefined]
            )

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Converts all :attr:`BatchNorm*D` layers in the model to :class:`torch.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> # xdoctest: +SKIP("distributed")
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
                bias=module.bias is not None,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            module_output.training = module.training
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output
<|endoftext|># mypy: allow-untyped-defs
import itertools
from typing import Any, Protocol

import torch
from torch.nn.parameter import is_lazy


__all__ = ["LazyModuleMixin"]


class _LazyProtocol(Protocol):
    """This class is used to avoid errors with mypy checks for the attributes in a mixin.

    https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes
    """

    def _register_load_state_dict_pre_hook(self, hook): ...

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False): ...

    def _lazy_load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ): ...

    def _get_name(self): ...

    def _infer_parameters(self, module, input): ...

    @property
    def _parameters(self): ...

    @property
    def _buffers(self): ...

    @property
    def _non_persistent_buffers_set(self): ...

    @property
    def _load_hook(self): ...

    @property
    def _initialize_hook(self): ...


class LazyModuleMixin:
    r"""A mixin for modules that lazily initialize parameters, also known as "lazy modules".

    .. warning::
        Lazy modules are an experimental new feature under active development,
        and their API is likely to change.

    Modules that lazily initialize parameters, or "lazy modules",
    derive the shapes of their parameters from the first input(s)
    to their forward method. Until that first forward they contain
    :class:`torch.nn.UninitializedParameter` s that should not be accessed
    or used, and afterward they contain regular :class:`torch.nn.Parameter` s.
    Lazy modules are convenient since they don't require computing some
    module arguments, like the :attr:`in_features` argument of a
    typical :class:`torch.nn.Linear`.

    After construction, networks with lazy modules should first
    be converted to the desired dtype and placed on the expected device.
    This is because lazy modules only perform shape inference so the usual dtype
    and device placement behavior applies.
    The lazy modules should then perform "dry runs" to initialize all the components in the module.
    These "dry runs" send inputs of the correct size, dtype, and device through
    the network and to each one of its lazy modules. After this the network can be used as usual.

    >>> # xdoctest: +SKIP
    >>> class LazyMLP(torch.nn.Module):
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.fc1 = torch.nn.LazyLinear(10)
    ...         self.relu1 = torch.nn.ReLU()
    ...         self.fc2 = torch.nn.LazyLinear(1)
    ...         self.relu2 = torch.nn.ReLU()
    ...
    ...     def forward(self, input):
    ...         x = self.relu1(self.fc1(input))
    ...         y = self.relu2(self.fc2(x))
    ...         return y
    >>> # constructs a network with lazy modules
    >>> lazy_mlp = LazyMLP()
    >>> # transforms the network's device and dtype
    >>> # NOTE: these transforms can and should be applied after construction and before any 'dry runs'
    >>> lazy_mlp = lazy_mlp.cuda()
    >>> lazy_mlp
    LazyMLP( (fc1): LazyLinear(in_features=0, out_features=10, bias=True)
      (relu1): ReLU()
      (fc2): LazyLinear(in_features=0, out_features=1, bias=True)
      (relu2): ReLU()
    )
    >>> # performs a dry run to initialize the network's lazy modules
    >>> lazy_mlp(torch.ones(10, 10).cuda())
    >>> # after initialization, LazyLinear modules become regular Linear modules
    >>> lazy_mlp
    LazyMLP(
      (fc1): Linear(in_features=10, out_features=10, bias=True)
      (relu1): ReLU()
      (fc2): Linear(in_features=10, out_features=1, bias=True)
      (relu2): ReLU()
    )
    >>> # attaches an optimizer, since parameters can now be used as usual
    >>> optim = torch.optim.SGD(lazy_mlp.parameters(), lr=0.01)

    A final caveat when using lazy modules is that the order of initialization of a network's
    parameters may change, since the lazy modules are always initialized after other modules.
    For example, if the LazyMLP class defined above had a :class:`torch.nn.LazyLinear` module
    first and then a regular :class:`torch.nn.Linear` second, the second module would be
    initialized on construction and the first module would be initialized during the first dry run.
    This can cause the parameters of a network using lazy modules to be initialized differently
    than the parameters of a network without lazy modules as the order of parameter initializations,
    which often depends on a stateful random number generator, is different.
    Check :doc:`/notes/randomness` for more details.

    Lazy modules can be serialized with a state dict like other modules. For example:

    >>> lazy_mlp = LazyMLP()
    >>> # The state dict shows the uninitialized parameters
    >>> lazy_mlp.state_dict()
    OrderedDict({'fc1.weight': <UninitializedParameter>,
                 'fc1.bias': <UninitializedParameter>,
                 'fc2.weight': <UninitializedParameter>,
                 'fc2.bias': <UninitializedParameter>})

    Lazy modules can load regular :class:`torch.nn.Parameter` s (i.e. you can serialize/deserialize
    initialized LazyModules and they will remain initialized)


    >>> full_mlp = LazyMLP()
    >>> # Dry run to initialize another module
    >>> full_mlp.forward(torch.ones(10, 1))
    >>> # Load an initialized state into a lazy module
    >>> lazy_mlp.load_state_dict(full_mlp.state_dict())
    >>> # The state dict now holds valid values
    >>> lazy_mlp.state_dict()
    OrderedDict([('fc1.weight',
                  tensor([[-0.3837],
                          [ 0.0907],
                          [ 0.6708],
                          [-0.5223],
                          [-0.9028],
                          [ 0.2851],
                          [-0.4537],
                          [ 0.6813],
                          [ 0.5766],
                          [-0.8678]])),
                 ('fc1.bias',
                  tensor([-1.8832e+25,  4.5636e-41, -1.8832e+25,  4.5636e-41, -6.1598e-30,
                           4.5637e-41, -1.8788e+22,  4.5636e-41, -2.0042e-31,  4.5637e-41])),
                 ('fc2.weight',
                  tensor([[ 0.1320,  0.2938,  0.0679,  0.2793,  0.1088, -0.1795, -0.2301,  0.2807,
                            0.2479,  0.1091]])),
                 ('fc2.bias', tensor([0.0019]))])

    Note, however, that the loaded parameters will not be replaced when doing a "dry run" if they are initialized
    when the state is loaded. This prevents using initialized modules in different contexts.
    """

    # modules inheriting from this will change their __class__ to the specified
    # one after they are fully initialized
    cls_to_become: type[Any] | None = None

    def __init__(self: _LazyProtocol, *args, **kwargs):
        # Mypy doesn't like this super call in a mixin
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        # pyrefly: ignore [read-only]
        self._load_hook = self._register_load_state_dict_pre_hook(self._lazy_load_hook)
        # pyrefly: ignore [read-only]
        self._initialize_hook = self.register_forward_pre_hook(
            self._infer_parameters, with_kwargs=True
        )

    def _save_to_state_dict(self: _LazyProtocol, destination, prefix, keep_vars):
        # This should be ideally implemented as a hook,
        # but we should override `detach` in the UninitializedParameter to return itself
        # which is not clean
        for name, param in self._parameters.items():
            if param is not None:
                if not (is_lazy(param) or keep_vars):
                    param = param.detach()
                destination[prefix + name] = param
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                if not (is_lazy(buf) or keep_vars):
                    buf = buf.detach()
                destination[prefix + name] = buf

    def _lazy_load_hook(
        self: _LazyProtocol,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load_state_dict pre-hook function for lazy buffers and parameters.

        The purpose of this hook is to adjust the current state and/or
        ``state_dict`` being loaded so that a module instance serialized in
        both un/initialized state can be deserialized onto both un/initialized
        module instance.
        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
        """
        for name, param in itertools.chain(
            self._parameters.items(), self._buffers.items()
        ):
            key = prefix + name
            if key in state_dict and param is not None:
                input_param = state_dict[key]
                if is_lazy(param):
                    # The current parameter is not initialized but the one being loaded is
                    # create a new parameter based on the uninitialized one
                    if not is_lazy(input_param):
                        with torch.no_grad():
                            param.materialize(input_param.shape)

    def initialize_parameters(self: _LazyProtocol, *args, **kwargs):
        r"""Initialize parameters according to the input batch properties.

        This adds an interface to isolate parameter initialization from the
        forward pass when doing parameter shape inference.
        """
        raise NotImplementedError(
            f"initialize_parameters is not implemented for {self.__class__.__name__}"
        )

    def has_uninitialized_params(self: _LazyProtocol):
        r"""Check if a module has parameters that are not initialized."""
        # This is to avoid the JIT to track this parameter and force
        # custom modules __setstate__ to add it
        params = self._parameters.values()
        buffers = self._buffers.values()
        for param in itertools.chain(params, buffers):
            if is_lazy(param):
                return True
        return False

    # torchrec tests the code consistency with the following code
    # fmt: off
    def _infer_parameters(self: _LazyProtocol, module, args, kwargs=None):
        r"""Infers the size and initializes the parameters according to the provided input batch.

        Given a module that contains parameters that were declared inferable
        using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
        in the complete module using the provided input to initialize all the parameters
        as needed.
        The module is set into evaluation mode before running the forward pass in order
        to avoid saving statistics or calculating gradients
        """
        kwargs = kwargs if kwargs else {}
        module.initialize_parameters(*args, **kwargs)
        if module.has_uninitialized_params():
            raise RuntimeError(f'module {self._get_name()} has not been fully initialized')
        module._initialize_hook.remove()
        module._load_hook.remove()
        delattr(module, '_initialize_hook')
        delattr(module, '_load_hook')
        if module.cls_to_become is not None:
            module.__class__ = module.cls_to_become
    # fmt: on

    def _replicate_for_data_parallel(self: _LazyProtocol):
        raise RuntimeError(
            "Modules with uninitialized parameters can't be used with `DataParallel`. "
            "Run a dummy forward pass to correctly initialize the modules"
        )
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.distributed as dist
from torch.autograd.function import Function


class SyncBatchNorm(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        self,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        momentum,
        process_group,
        world_size,
    ):
        if not (
            input.is_contiguous(memory_format=torch.channels_last)
            or input.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        size = int(input.numel() // input.size(1))
        if size == 1 and world_size < 2:
            raise ValueError(
                f"Expected more than 1 value per channel when training, got input size {size}"
            )

        num_channels = input.shape[1]
        if input.numel() > 0:
            # calculate mean/invstd for input.
            mean, invstd = torch.batch_norm_stats(input, eps)

            count = torch.full(
                (1,),
                input.numel() // input.size(1),
                dtype=mean.dtype,
                device=mean.device,
            )

            # C, C, 1 -> (2C + 1)
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            # for empty input, set stats and the count to zero. The stats with
            # zero count will be filtered out later when computing global mean
            # & invstd, but they still needs to participate the all_gather
            # collective communication to unblock other peer processes.
            combined = torch.zeros(
                2 * num_channels + 1, dtype=input.dtype, device=input.device
            )

        # Use allgather instead of allreduce because count could be different across
        # ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
        # all gathered mean, invstd and count.
        # for nccl backend, use the optimized version of all gather.
        # The Gloo backend does not support `all_gather_into_tensor`.
        if process_group._get_backend_name() != "gloo":
            # world_size * (2C + 1)
            combined_size = combined.numel()
            combined_flat = torch.empty(
                1,
                combined_size * world_size,
                dtype=combined.dtype,
                device=combined.device,
            )
            dist.all_gather_single(
                combined_flat, combined, process_group, async_op=False
            )
            combined = torch.reshape(combined_flat, (world_size, combined_size))
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
        else:
            # world_size * (2C + 1)
            combined_list = [torch.empty_like(combined) for _ in range(world_size)]
            dist.all_gather(combined_list, combined, process_group, async_op=False)
            combined = torch.stack(combined_list, dim=0)
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            # The lines below force a synchronization between CUDA and CPU, because
            # the shape of the result count_all depends on the values in mask tensor.
            # Such synchronizations break CUDA Graph capturing.
            # See https://github.com/pytorch/pytorch/issues/78549
            # FIXME: https://github.com/pytorch/pytorch/issues/78656 describes
            # a better longer-term solution.

            # remove stats from empty inputs
            mask = count_all.squeeze(-1) >= 1
            count_all = count_all[mask]
            mean_all = mean_all[mask]
            invstd_all = invstd_all[mask]

        # calculate global mean & invstd
        counts = count_all.view(-1)
        if running_mean is not None and counts.dtype != running_mean.dtype:
            counts = counts.to(running_mean.dtype)
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            counts,
        )

        self.save_for_backward(input, weight, mean, invstd, count_all.to(torch.int32))
        self.process_group = process_group

        # apply element-wise normalization
        if input.numel() > 0:
            return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        else:
            return torch.empty_like(input)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(self, grad_output):
        if not (
            grad_output.is_contiguous(memory_format=torch.channels_last)
            or grad_output.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        if saved_input.numel() > 0:
            # calculate local stats as well as grad_weight / grad_bias
            (
                sum_dy,
                sum_dy_xmu,
                grad_weight,
                grad_bias,
            ) = torch.batch_norm_backward_reduce(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                self.needs_input_grad[0],
                self.needs_input_grad[1],
                self.needs_input_grad[2],
            )

            if self.needs_input_grad[0]:
                # synchronizing stats used to calculate input gradient.
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(
                    combined,
                    torch.distributed.ReduceOp.SUM,
                    process_group,
                    async_op=False,
                )
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

                # backward pass for gradient calculation
                if weight is not None and weight.dtype != mean.dtype:
                    weight = weight.to(mean.dtype)
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    sum_dy,
                    sum_dy_xmu,
                    count_tensor,
                )
            # synchronizing of grad_weight / grad_bias is not needed as distributed
            # training would handle all reduce.
            if weight is None or not self.needs_input_grad[1]:
                grad_weight = None

            if weight is None or not self.needs_input_grad[2]:
                grad_bias = None
        else:
            # This process got an empty input tensor in the forward pass.
            # Although this process can directly set grad_input as an empty
            # tensor of zeros, it still needs to participate in the collective
            # communication to unblock its peers, as other peer processes might
            # have received non-empty inputs.
            num_channels = saved_input.shape[1]
            if self.needs_input_grad[0]:
                # launch all_reduce to unblock other peer processes
                combined = torch.zeros(
                    2 * num_channels, dtype=saved_input.dtype, device=saved_input.device
                )
                torch.distributed.all_reduce(
                    combined,
                    torch.distributed.ReduceOp.SUM,
                    process_group,
                    async_op=False,
                )

            # Leave grad_input, grad_weight and grad_bias as None, which will be
            # interpreted by the autograd engine as Tensors full of zeros.

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class CrossMapLRN2d(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input, size, alpha=1e-4, beta=0.75, k=1):
        ctx.size = size
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.k = k
        ctx.scale = None

        if input.dim() != 4:
            raise ValueError(
                f"CrossMapLRN2d: Expected input to be 4D, got {input.dim()}D instead."
            )

        ctx.scale = ctx.scale or input.new()
        output = input.new()
        channels = input.size(1)

        output.resize_as_(input)
        ctx.scale.resize_as_(input)

        # use output storage as temporary buffer
        input_square = output
        torch.pow(input, 2, out=input_square)

        pre_pad = int((ctx.size - 1) / 2 + 1)
        pre_pad_crop = min(pre_pad, channels)

        scale_first = ctx.scale.select(1, 0)
        scale_first.zero_()
        # compute first feature map normalization
        for c in range(pre_pad_crop):
            scale_first.add_(input_square.select(1, c))

        # reuse computations for next feature maps normalization
        # by adding the next feature map and removing the previous
        for c in range(1, channels):
            scale_previous = ctx.scale.select(1, c - 1)
            scale_current = ctx.scale.select(1, c)
            scale_current.copy_(scale_previous)
            if c < channels - pre_pad + 1:
                square_next = input_square.select(1, c + pre_pad - 1)
                scale_current.add_(square_next, alpha=1)

            if c > pre_pad:
                square_previous = input_square.select(1, c - pre_pad)
                scale_current.add_(square_previous, alpha=-1)

        ctx.scale.mul_(ctx.alpha / ctx.size).add_(ctx.k)

        torch.pow(ctx.scale, -ctx.beta, out=output)
        output.mul_(input)

        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        grad_input = grad_output.new()

        batch_size = input.size(0)
        channels = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        paddded_ratio = input.new(channels + ctx.size - 1, input_height, input_width)
        accum_ratio = input.new(input_height, input_width)

        cache_ratio_value = 2 * ctx.alpha * ctx.beta / ctx.size
        inversePrePad = int(ctx.size - (ctx.size - 1) / 2)

        grad_input.resize_as_(input)
        torch.pow(ctx.scale, -ctx.beta, out=grad_input).mul_(grad_output)

        paddded_ratio.zero_()
        padded_ratio_center = paddded_ratio.narrow(0, inversePrePad, channels)
        for n in range(batch_size):
            torch.mul(grad_output[n], output[n], out=padded_ratio_center)
            padded_ratio_center.div_(ctx.scale[n])
            torch.sum(
                paddded_ratio.narrow(0, 0, ctx.size - 1),
                0,
                keepdim=False,
                out=accum_ratio,
            )
            for c in range(channels):
                accum_ratio.add_(paddded_ratio[c + ctx.size - 1])
                grad_input[n][c].addcmul_(
                    input[n][c], accum_ratio, value=-cache_ratio_value
                )
                accum_ratio.add_(paddded_ratio[c], alpha=-1)

        return grad_input, None, None, None, None


class BackwardHookFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, *args):
        ctx.mark_non_differentiable(*[arg for arg in args if not arg.requires_grad])
        return args

    @staticmethod
    def backward(ctx, *args):
        return args
<|endoftext|># mypy: allow-untyped-defs

import warnings

import torch.nn.functional as F
from torch import Tensor

from .batchnorm import _LazyNormBase, _NormBase


__all__ = [
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
]


class _InstanceNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
        *,
        bias: bool = True,  # for backward compatibility
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            **factory_kwargs,
            bias=bias,
        )

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _get_no_batch_dim(self):
        raise NotImplementedError

    def _handle_no_batch_input(self, input):
        return self._apply_instance_norm(input.unsqueeze(0)).squeeze(0)

    def _apply_instance_norm(self, input):
        return F.instance_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum if self.momentum is not None else 0.0,
            self.eps,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        version = local_metadata.get("version", None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ("running_mean", "running_var"):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    "Unexpected running stats buffer(s) {names} for {klass} "
                    "with track_running_stats=False. If state_dict is a "
                    "checkpoint saved before 0.4.0, this may be expected "
                    "because {klass} does not track running stats by default "
                    "since 0.4.0. Please remove these keys from state_dict. If "
                    "the running stats are actually needed, instead set "
                    "track_running_stats=True in {klass} to enable them. See "
                    "the documentation of {klass} for details.".format(
                        names=" and ".join(f'"{k}"' for k in running_stats_keys),
                        klass=self.__class__.__name__,
                    )
                )
                for key in running_stats_keys:
                    state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        feature_dim = input.dim() - self._get_no_batch_dim()
        if input.size(feature_dim) != self.num_features:
            if self.affine:
                raise ValueError(
                    f"expected input's size at dim={feature_dim} to match num_features"
                    f" ({self.num_features}), but got: {input.size(feature_dim)}."
                )
            else:
                warnings.warn(
                    f"input's size at dim={feature_dim} does not match num_features. "
                    "You can silence this warning by not passing in num_features, "
                    "which is not used because affine=False",
                    stacklevel=2,
                )

        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return self._apply_instance_norm(input)


class InstanceNorm1d(_InstanceNorm):
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 2D (unbatched) or 3D (batched) input as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input) if :attr:`affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    `torch.var(input, correction=0)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm1d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm1d` is applied
        on each channel of channeled data like multidimensional time series, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm1d` usually doesn't apply affine
        transform.

    Args:
        num_features: number of features or channels :math:`C` of the input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm1d(100, affine=True)
        >>> input = torch.randn(20, 100, 40)
        >>> output = m(input)
    """

    def _get_no_batch_dim(self) -> int:
        return 2

    def _check_input_dim(self, input) -> None:
        if input.dim() not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class LazyInstanceNorm1d(_LazyNormBase, _InstanceNorm):
    r"""A :class:`torch.nn.InstanceNorm1d` module with lazy initialization of the ``num_features`` argument.

    The ``num_features`` argument of the :class:`InstanceNorm1d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`, `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`(C, L)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input)
    """

    cls_to_become = InstanceNorm1d  # type: ignore[assignment]

    def _get_no_batch_dim(self) -> int:
        return 2

    def _check_input_dim(self, input) -> None:
        if input.dim() not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class InstanceNorm2d(_InstanceNorm):
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, correction=0)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm2d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm2d` is applied
        on each channel of channeled data like RGB images, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm2d` usually doesn't apply affine
        transform.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)` or :math:`(C, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm2d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm2d(100, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _get_no_batch_dim(self) -> int:
        return 3

    def _check_input_dim(self, input) -> None:
        if input.dim() not in (3, 4):
            raise ValueError(f"expected 3D or 4D input (got {input.dim()}D input)")


class LazyInstanceNorm2d(_LazyNormBase, _InstanceNorm):
    r"""A :class:`torch.nn.InstanceNorm2d` module with lazy initialization of the ``num_features`` argument.

    The ``num_features`` argument of the :class:`InstanceNorm2d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)` or :math:`(C, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)
    """

    cls_to_become = InstanceNorm2d  # type: ignore[assignment]

    def _get_no_batch_dim(self) -> int:
        return 3

    def _check_input_dim(self, input) -> None:
        if input.dim() not in (3, 4):
            raise ValueError(f"expected 3D or 4D input (got {input.dim()}D input)")


class InstanceNorm3d(_InstanceNorm):
    r"""Applies Instance Normalization.

    This operation applies Instance Normalization
    over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`__.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size C (where C is the input size) if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, correction=0)`.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm3d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm3d` is applied
        on each channel of channeled data like 3D models with RGB color, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm3d` usually doesn't apply affine
        transform.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm3d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm3d(100, affine=True)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _get_no_batch_dim(self) -> int:
        return 4

    def _check_input_dim(self, input) -> None:
        if input.dim() not in (4, 5):
            raise ValueError(f"expected 4D or 5D input (got {input.dim()}D input)")


class LazyInstanceNorm3d(_LazyNormBase, _InstanceNorm):
    r"""A :class:`torch.nn.InstanceNorm3d` module with lazy initialization of the ``num_features`` argument.

    The ``num_features`` argument of the :class:`InstanceNorm3d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input)
    """

    cls_to_become = InstanceNorm3d  # type: ignore[assignment]

    def _get_no_batch_dim(self) -> int:
        return 4

    def _check_input_dim(self, input) -> None:
        if input.dim() not in (4, 5):
            raise ValueError(f"expected 4D or 5D input (got {input.dim()}D input)")
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter

from .module import Module


__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros,
                                     but can be updated to another value to be used as the padding vector.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`

    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)

    .. note::
        When :attr:`max_norm` is not ``None``, :class:`Embedding`'s forward method will modify the
        :attr:`weight` tensor in-place. Since tensors needed for gradient computations cannot be
        modified in-place, performing a differentiable operation on ``Embedding.weight`` before
        calling :class:`Embedding`'s forward method requires cloning ``Embedding.weight`` when
        :attr:`max_norm` is not ``None``. For example::

            n, d, m = 3, 5, 7
            embedding = nn.Embedding(n, d, max_norm=1.0)
            W = torch.randn((m, d), requires_grad=True)
            idx = torch.tensor([1, 2])
            a = (
                embedding.weight.clone() @ W.t()
            )  # weight must be cloned for this to be differentiable
            b = embedding(idx) @ W.t()  # modifies weight in-place
            out = a.unsqueeze(0) + b.unsqueeze(1)
            loss = out.sigmoid().prod()
            loss.backward()

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])


        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0, 2, 0, 5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])

        >>> # example of changing `pad` vector
        >>> padding_idx = 0
        >>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
        >>> embedding.weight
        Parameter containing:
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7895, -0.7089, -0.0364],
                [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
        >>> with torch.no_grad():
        ...     embedding.weight[padding_idx] = torch.ones(3)
        >>> embedding.weight
        Parameter containing:
        tensor([[ 1.0000,  1.0000,  1.0000],
                [-0.7895, -0.7089, -0.0364],
                [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: int | None
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                if padding_idx >= self.num_embeddings:
                    raise AssertionError("Padding_idx must be within num_embeddings")
            elif padding_idx < 0:
                if padding_idx < -self.num_embeddings:
                    raise AssertionError("Padding_idx must be within num_embeddings")
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                requires_grad=not _freeze,
            )
            self.reset_parameters()
        else:
            if list(_weight.shape) != [num_embeddings, embedding_dim]:
                raise AssertionError(
                    "Shape of weight does not match num_embeddings and embedding_dim"
                )
            self.weight = Parameter(_weight, requires_grad=not _freeze)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        r"""Create Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                         therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                         i.e. it remains as a fixed "pad".
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        if embeddings.dim() != 2:
            raise AssertionError("Embeddings parameter is expected to be 2-dimensional")
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        return embedding


class EmbeddingBag(Module):
    r"""Compute sums or means of 'bags' of embeddings, without instantiating the intermediate embeddings.

    For bags of constant length, no :attr:`per_sample_weights`, no indices equal to :attr:`padding_idx`,
    and with 2D inputs, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=1)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=1)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=1)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory efficient than using a chain of these
    operations.

    EmbeddingBag also supports per-sample weights as an argument to the forward
    pass. This scales the output of the Embedding before performing a weighted
    reduction as specified by ``mode``. If :attr:`per_sample_weights` is passed, the
    only supported ``mode`` is ``"sum"``, which computes a weighted sum according to
    :attr:`per_sample_weights`.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (str, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.
        include_last_offset (bool, optional): if ``True``, the size of offsets is equal to the number of bags + 1.
                                              The last element is the size of the input, or the ending index position
                                              of the last bag (sequence). This matches the CSR format. Ignored when
                                              input is 2D. Default ``False``.
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the
                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated
                                     during training, i.e. it remains as a fixed "pad". For a newly constructed
                                     EmbeddingBag, the embedding vector at :attr:`padding_idx` will default to all
                                     zeros, but can be updated to another value to be used as the padding vector.
                                     Note that the embedding vector at :attr:`padding_idx` is excluded from the
                                     reduction.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape `(num_embeddings, embedding_dim)`
                         initialized from :math:`\mathcal{N}(0, 1)`.

    Examples::

        >>> # an EmbeddingBag module containing 10 tensors of size 3
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
        >>> offsets = torch.tensor([0, 4], dtype=torch.long)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> embedding_sum(input, offsets)
        tensor([[-0.8861, -5.4350, -0.0523],
                [ 1.1306, -2.5798, -1.0044]])

        >>> # Example with padding_idx
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum', padding_idx=2)
        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9], dtype=torch.long)
        >>> offsets = torch.tensor([0, 4], dtype=torch.long)
        >>> embedding_sum(input, offsets)
        tensor([[ 0.0000,  0.0000,  0.0000],
                [-0.7082,  3.2145, -2.6251]])

        >>> # An EmbeddingBag can be loaded from an Embedding like so
        >>> embedding = nn.Embedding(10, 3, padding_idx=2)
        >>> embedding_sum = nn.EmbeddingBag.from_pretrained(
                embedding.weight,
                padding_idx=embedding.padding_idx,
                mode='sum')
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
        "include_last_offset",
        "padding_idx",
    ]

    num_embeddings: int
    embedding_dim: int
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    mode: str
    sparse: bool
    include_last_offset: bool
    padding_idx: int | None

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if padding_idx is not None:
            if padding_idx > 0:
                if padding_idx >= self.num_embeddings:
                    raise AssertionError("padding_idx must be within num_embeddings")
            elif padding_idx < 0:
                if padding_idx < -self.num_embeddings:
                    raise AssertionError("padding_idx must be within num_embeddings")
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        if _weight is None:
            self.weight = Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
            )
            self.reset_parameters()
        else:
            if list(_weight.shape) != [num_embeddings, embedding_dim]:
                raise AssertionError(
                    "Shape of weight does not match num_embeddings and embedding_dim"
                )
            self.weight = Parameter(_weight)
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(
        self,
        input: Tensor,
        offsets: Tensor | None = None,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of EmbeddingBag.

        Args:
            input (Tensor): Tensor containing bags of indices into the embedding matrix.
            offsets (Tensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                the starting index position of each bag (sequence) in :attr:`input`.
            per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
                to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
                must have exactly the same shape as input and is treated as having the same
                :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.

        Returns:
            Tensor output shape of `(B, embedding_dim)`.

        .. note::

            A few notes about ``input`` and ``offsets``:

            - :attr:`input` and :attr:`offsets` have to be of the same type, either int or long

            - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)
              each of fixed length ``N``, and this will return ``B`` values aggregated in a way
              depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.

            - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of
              multiple bags (sequences).  :attr:`offsets` is required to be a 1D tensor containing the
              starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets` of shape `(B)`,
              :attr:`input` will be viewed as having ``B`` bags. Empty bags (i.e., having 0-length) will have
              returned vectors filled by zeros.
        """
        return F.embedding_bag(
            input,
            self.weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        s += ", mode={mode}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**{k: repr(v) for k, v in self.__dict__.items()})

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
    ) -> "EmbeddingBag":
        r"""Create EmbeddingBag instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the EmbeddingBag.
                First dimension is being passed to EmbeddingBag as 'num_embeddings', second as 'embedding_dim'.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embeddingbag.weight.requires_grad = False``. Default: ``True``
            max_norm (float, optional): See module initialization documentation. Default: ``None``
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            mode (str, optional): See module initialization documentation. Default: ``"mean"``
            sparse (bool, optional): See module initialization documentation. Default: ``False``.
            include_last_offset (bool, optional): See module initialization documentation. Default: ``False``.
            padding_idx (int, optional): See module initialization documentation. Default: ``None``.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embeddingbag = nn.EmbeddingBag.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([[1, 0]])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embeddingbag(input)
            tensor([[ 2.5000,  3.7000,  4.6500]])
        """
        if embeddings.dim() != 2:
            raise AssertionError("Embeddings parameter is expected to be 2-dimensional")
        rows, cols = embeddings.shape
        embeddingbag = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            mode=mode,
            sparse=sparse,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )
        embeddingbag.weight.requires_grad = not freeze
        return embeddingbag
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t

from .module import Module
from .utils import _ntuple, _pair, _quadruple


# TODO: grad_output size asserts in THNN

__all__ = [
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]


class _CircularPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        return F.pad(input, self.padding, "circular")

    def extra_repr(self) -> str:
        return f"{self.padding}"


class CircularPad1d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.CircularPad1d(2)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[2., 3., 0., 1., 2., 3., 0., 1.],
                 [6., 7., 4., 5., 6., 7., 4., 5.]]])
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad1d((3, 1))
        >>> m(input)
        tensor([[[1., 2., 3., 0., 1., 2., 3., 0.],
                 [5., 6., 7., 4., 5., 6., 7., 4.]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__()
        self.padding = _pair(padding)

    def _check_input_dim(self, input) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class CircularPad2d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.CircularPad2d(2)
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[4., 5., 3., 4., 5., 3., 4.],
                  [7., 8., 6., 7., 8., 6., 7.],
                  [1., 2., 0., 1., 2., 0., 1.],
                  [4., 5., 3., 4., 5., 3., 4.],
                  [7., 8., 6., 7., 8., 6., 7.],
                  [1., 2., 0., 1., 2., 0., 1.],
                  [4., 5., 3., 4., 5., 3., 4.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[5., 3., 4., 5., 3.],
                  [8., 6., 7., 8., 6.],
                  [2., 0., 1., 2., 0.],
                  [5., 3., 4., 5., 3.],
                  [8., 6., 7., 8., 6.]]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        self.padding = _quadruple(padding)

    def _check_input_dim(self, input) -> None:
        if input.dim() != 3 and input.dim() != 4:
            raise ValueError(f"expected 3D or 4D input (got {input.dim()}D input)")


class CircularPad3d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.CircularPad3d(3)
        >>> input = torch.randn(16, 3, 8, 320, 480)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()
        self.padding = _ntuple(6)(padding)

    def _check_input_dim(self, input) -> None:
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError(f"expected 4D or 5D input (got {input.dim()}D input)")


class _ConstantPadNd(Module):
    __constants__ = ["padding", "value"]
    value: float
    padding: Sequence[int]

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "constant", self.value)

    def extra_repr(self) -> str:
        return f"padding={self.padding}, value={self.value}"


class ConstantPad1d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000, -1.0491, -0.7152, -0.0749,  0.8530,  3.5000,
                   3.5000],
                 [ 3.5000,  3.5000, -1.3287,  1.8966,  0.1466, -0.2771,  3.5000,
                   3.5000]]])
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad1d((3, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000],
                 [ 3.5000,  3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t, value: float) -> None:
        super().__init__(value)
        self.padding = _pair(padding)


class ConstantPad2d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ConstantPad2d(2, 3.5)
        >>> input = torch.randn(1, 2, 2)
        >>> input
        tensor([[[ 1.6585,  0.4320],
                 [-0.8701, -0.4649]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  1.6585,  0.4320,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -0.8701, -0.4649,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  1.6585,  0.4320],
                 [ 3.5000,  3.5000,  3.5000, -0.8701, -0.4649],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
    """

    __constants__ = ["padding", "value"]
    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t, value: float) -> None:
        super().__init__(value)
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ConstantPad3d(3, 3.5)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
        >>> output = m(input)
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t, value: float) -> None:
        super().__init__(value)
        self.padding = _ntuple(6)(padding)


class _ReflectionPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "reflect")

    def extra_repr(self) -> str:
        return f"{self.padding}"


class ReflectionPad1d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ReflectionPad1d(2)
        >>> # xdoctest: +IGNORE_WANT("other tests seem to modify printing styles")
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[2., 1., 0., 1., 2., 3., 2., 1.],
                 [6., 5., 4., 5., 6., 7., 6., 5.]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReflectionPad1d((3, 1))
        >>> m(input)
        tensor([[[3., 2., 1., 0., 1., 2., 3., 2.],
                 [7., 6., 5., 4., 5., 6., 7., 6.]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__()
        self.padding = _pair(padding)


class ReflectionPad2d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})` where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReflectionPad2d(2)
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[8., 7., 6., 7., 8., 7., 6.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [2., 1., 0., 1., 2., 1., 0.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [8., 7., 6., 7., 8., 7., 6.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [2., 1., 0., 1., 2., 1., 0.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReflectionPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[7., 6., 7., 8., 7.],
                  [4., 3., 4., 5., 4.],
                  [1., 0., 1., 2., 1.],
                  [4., 3., 4., 5., 4.],
                  [7., 6., 7., 8., 7.]]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        self.padding = _quadruple(padding)


class ReflectionPad3d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReflectionPad3d(1)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2)
        >>> m(input)
        tensor([[[[[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]],
                  [[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]]]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()
        self.padding = _ntuple(6)(padding)


class _ReplicationPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "replicate")

    def extra_repr(self) -> str:
        return f"{self.padding}"


class ReplicationPad1d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReplicationPad1d(2)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[0., 0., 0., 1., 2., 3., 3., 3.],
                 [4., 4., 4., 5., 6., 7., 7., 7.]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad1d((3, 1))
        >>> m(input)
        tensor([[[0., 0., 0., 0., 1., 2., 3., 3.],
                 [4., 4., 4., 4., 5., 6., 7., 7.]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__()
        self.padding = _pair(padding)


class ReplicationPad2d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ReplicationPad2d(2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[0., 0., 0., 1., 2., 2., 2.],
                  [0., 0., 0., 1., 2., 2., 2.],
                  [0., 0., 0., 1., 2., 2., 2.],
                  [3., 3., 3., 4., 5., 5., 5.],
                  [6., 6., 6., 7., 8., 8., 8.],
                  [6., 6., 6., 7., 8., 8., 8.],
                  [6., 6., 6., 7., 8., 8., 8.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[0., 0., 1., 2., 2.],
                  [0., 0., 1., 2., 2.],
                  [0., 0., 1., 2., 2.],
                  [3., 3., 4., 5., 5.],
                  [6., 6., 7., 8., 8.]]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ReplicationPad3d(3)
        >>> input = torch.randn(16, 3, 8, 320, 480)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()
        self.padding = _ntuple(6)(padding)


class ZeroPad1d(ConstantPad1d):
    r"""Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ZeroPad1d(2)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 0.0000,  0.0000, -1.0491, -0.7152, -0.0749,  0.8530,  0.0000,
                   0.0000],
                 [ 0.0000,  0.0000, -1.3287,  1.8966,  0.1466, -0.2771,  0.0000,
                   0.0000]]])
        >>> m = nn.ZeroPad1d(2)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000,  0.0000],
                 [ 0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000,  0.0000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad1d((3, 1))
        >>> m(input)
        tensor([[[ 0.0000,  0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000],
                 [ 0.0000,  0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000]]])
    """

    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__(padding, 0.0)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"{self.padding}"


class ZeroPad2d(ConstantPad2d):
    r"""Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ZeroPad2d(2)
        >>> input = torch.randn(1, 1, 3, 3)
        >>> input
        tensor([[[[-0.1678, -0.4418,  1.9466],
                  [ 0.9604, -0.4219, -0.5241],
                  [-0.9162, -0.5436, -0.6446]]]])
        >>> m(input)
        tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000, -0.1678, -0.4418,  1.9466,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.9604, -0.4219, -0.5241,  0.0000,  0.0000],
                  [ 0.0000,  0.0000, -0.9162, -0.5436, -0.6446,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000, -0.1678, -0.4418,  1.9466,  0.0000],
                  [ 0.0000,  0.9604, -0.4219, -0.5241,  0.0000],
                  [ 0.0000, -0.9162, -0.5436, -0.6446,  0.0000]]]])
    """

    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__(padding, 0.0)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"{self.padding}"


class ZeroPad3d(ConstantPad3d):
    r"""Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If it is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ZeroPad3d(3)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))
        >>> output = m(input)
    """

    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__(padding, 0.0)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"{self.padding}"
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math
import numbers
import warnings
import weakref
from typing import overload
from typing_extensions import deprecated

import torch
from torch import _VF, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence

from .module import Module


__all__ = [
    "RNNBase",
    "RNN",
    "LSTM",
    "GRU",
    "RNNCellBase",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
]

_rnn_impls = {
    "RNN_TANH": _VF.rnn_tanh,
    "RNN_RELU": _VF.rnn_relu,
}


def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


@deprecated(
    "`apply_permutation` is deprecated, please use `tensor.index_select(dim, permutation)` instead",
    category=FutureWarning,
)
def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return _apply_permutation(tensor, permutation, dim)


class RNNBase(Module):
    r"""Base class for RNN modules (RNN, LSTM, GRU).

    Implements aspects of RNNs shared by the RNN, LSTM, and GRU classes, such as module initialization
    and utility methods for parameter storage management.

    .. note::
        The forward method is not implemented by the RNNBase class.

    .. note::
        LSTM and GRU classes override some methods implemented by RNNBase.
    """

    __constants__ = [
        "mode",
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
        "proj_size",
    ]
    __jit_unused_properties__ = ["all_weights"]

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self._flat_weight_refs: list[weakref.ReferenceType[Parameter] | None] = []
        num_directions = 2 if bidirectional else 1

        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                f"num_layers greater than 1, but got dropout={dropout} and "
                f"num_layers={num_layers}",
                stacklevel=2,
            )

        if not isinstance(bias, bool):
            raise TypeError(f"bias should be of type bool, got: {type(bias).__name__}")
        if not isinstance(batch_first, bool):
            raise TypeError(
                f"batch_first should be of type bool, got: {type(batch_first).__name__}"
            )
        if not isinstance(input_size, int):
            raise TypeError(
                f"input_size should be of type int, got: {type(input_size).__name__}"
            )
        if input_size <= 0:
            raise ValueError("input_size must be greater than zero")
        if not isinstance(hidden_size, int):
            raise TypeError(
                f"hidden_size should be of type int, got: {type(hidden_size).__name__}"
            )
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        if proj_size < 0:
            raise ValueError(
                "proj_size should be a positive integer or zero to disable projections"
            )
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        elif mode == "RNN_TANH":
            gate_size = hidden_size
        elif mode == "RNN_RELU":
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                layer_input_size = (
                    input_size if layer == 0 else real_hidden_size * num_directions
                )

                w_ih = Parameter(
                    torch.empty((gate_size, layer_input_size), **factory_kwargs)
                )
                w_hh = Parameter(
                    torch.empty((gate_size, real_hidden_size), **factory_kwargs)
                )
                b_ih = Parameter(torch.empty(gate_size, **factory_kwargs))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.empty(gate_size, **factory_kwargs))
                layer_params: tuple[Tensor, ...] = ()
                if self.proj_size == 0:
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)
                else:
                    w_hr = Parameter(
                        torch.empty((proj_size, hidden_size), **factory_kwargs)
                    )
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
                    else:
                        layer_params = (w_ih, w_hh, w_hr)

                suffix = "_reverse" if direction == 1 else ""
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                if self.proj_size > 0:
                    param_names += ["weight_hr_l{}{}"]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params, strict=True):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._init_flat_weights()

        self.reset_parameters()

    def _init_flat_weights(self) -> None:
        self._flat_weights = [
            getattr(self, wn) if hasattr(self, wn) else None
            for wn in self._flat_weights_names
        ]
        self._flat_weight_refs = [
            weakref.ref(w) if w is not None else None for w in self._flat_weights
        ]
        self.flatten_parameters()

    def __setattr__(self, attr, value) -> None:
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super().__setattr__(attr, value)

    def flatten_parameters(self) -> None:
        """Reset parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = self._flat_weights[0]  # type: ignore[union-attr]
        dtype = first_fw.dtype  # type: ignore[union-attr]
        for fw in self._flat_weights:
            if (
                not isinstance(fw, Tensor)
                or fw.dtype != dtype
                or not fw.is_cuda
                or not torch.backends.cudnn.is_acceptable(fw)
            ):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = {
            p.data_ptr()  # type: ignore[union-attr]
            for p in self._flat_weights
        }
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights,  # type: ignore[arg-type]
                        num_weights,
                        self.input_size,
                        rnn.get_cudnn_mode(self.mode),
                        self.hidden_size,
                        self.proj_size,
                        self.num_layers,
                        self.batch_first,
                        bool(self.bidirectional),
                    )

    def _apply(self, fn, recurse=True):
        self._flat_weight_refs = []
        ret = super()._apply(fn, recurse)

        # Resets _flat_weights
        # Note: be v. careful before removing this, as 3rd party device types
        # likely rely on this behavior to properly .to() modules like LSTM.
        self._init_flat_weights()

        return ret

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def check_input(self, input: Tensor, batch_sizes: Tensor | None) -> None:
        if not torch.jit.is_scripting():
            if (
                input.dtype != self._flat_weights[0].dtype  # type: ignore[union-attr]
                and not torch._C._is_any_autocast_enabled()
            ):
                raise ValueError(
                    f"RNN input dtype ({input.dtype}) does not match weight dtype ({self._flat_weights[0].dtype}). "  # type: ignore[union-attr]
                    f"Convert input: input.to({self._flat_weights[0].dtype}), or convert model: model.to({input.dtype})"  # type: ignore[union-attr]
                )
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                f"input must have {expected_input_dim} dimensions, got {input.dim()}"
            )
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f"input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}"
            )

    def get_expected_hidden_size(
        self, input: Tensor, batch_sizes: Tensor | None
    ) -> tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (
                self.num_layers * num_directions,
                mini_batch,
                self.proj_size,
            )
        else:
            expected_hidden_size = (
                self.num_layers * num_directions,
                mini_batch,
                self.hidden_size,
            )
        return expected_hidden_size

    def check_hidden_size(
        self,
        hx: Tensor,
        expected_hidden_size: tuple[int, int, int],
        msg: str = "Expected hidden size {}, got {}",
    ) -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def _weights_have_changed(self):
        # Returns True if the weight tensors have changed since the last forward pass.
        # This is the case when used with torch.func.functional_call(), for example.
        weights_changed = False
        for ref, name in zip(
            self._flat_weight_refs, self._flat_weights_names, strict=True
        ):
            weight = getattr(self, name) if hasattr(self, name) else None
            if weight is not None and ref is not None and ref() is not weight:
                weights_changed = True
                break
        return weights_changed

    def check_forward_args(
        self, input: Tensor, hidden: Tensor, batch_sizes: Tensor | None
    ) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Tensor | None):
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if self.proj_size != 0:
            s += ", proj_size={proj_size}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    def _update_flat_weights(self) -> None:
        if not torch.jit.is_scripting():
            if self._weights_have_changed():
                self._init_flat_weights()

    def __getstate__(self):
        # If weights have been changed, update the _flat_weights in __getstate__ here.
        self._update_flat_weights()
        # Don't serialize the weight references.
        state = self.__dict__.copy()
        del state["_flat_weight_refs"]
        return state

    def __setstate__(self, d):
        super().__setstate__(d)
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        # In PyTorch 1.8 we added a proj_size member variable to LSTM.
        # LSTMs that were serialized via torch.save(module) before PyTorch 1.8
        # don't have it, so to preserve compatibility we set proj_size here.
        if "proj_size" not in d:
            self.proj_size = 0

        if not isinstance(self._all_weights[0][0], str):
            num_layers = self.num_layers
            num_directions = 2 if self.bidirectional else 1
            self._flat_weights_names = []
            self._all_weights = []
            for layer in range(num_layers):
                for direction in range(num_directions):
                    suffix = "_reverse" if direction == 1 else ""
                    weights = [
                        "weight_ih_l{}{}",
                        "weight_hh_l{}{}",
                        "bias_ih_l{}{}",
                        "bias_hh_l{}{}",
                        "weight_hr_l{}{}",
                    ]
                    weights = [x.format(layer, suffix) for x in weights]
                    if self.bias:
                        if self.proj_size > 0:
                            self._all_weights += [weights]
                            self._flat_weights_names.extend(weights)
                        else:
                            self._all_weights += [weights[:4]]
                            self._flat_weights_names.extend(weights[:4])
                    else:
                        if self.proj_size > 0:
                            self._all_weights += [weights[:2]] + [weights[-1:]]
                            self._flat_weights_names.extend(
                                weights[:2] + [weights[-1:]]
                            )
                        else:
                            self._all_weights += [weights[:2]]
                            self._flat_weights_names.extend(weights[:2])
            self._flat_weights = [
                getattr(self, wn) if hasattr(self, wn) else None
                for wn in self._flat_weights_names
            ]

        self._flat_weight_refs = [
            weakref.ref(w) if w is not None else None for w in self._flat_weights
        ]

    @property
    def all_weights(self) -> list[list[Parameter]]:
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        # Need to copy these caches, otherwise the replica will share the same
        # flat weights list.
        replica._flat_weights = replica._flat_weights[:]
        replica._flat_weights_names = replica._flat_weights_names[:]
        return replica


class RNN(RNNBase):
    r"""__init__(input_size,hidden_size,num_layers=1,nonlinearity='tanh',bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None)

    Apply a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity to an input sequence. For each element in the input sequence,
    each layer computes the following function:

    .. math::
        h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    .. code-block:: python

        # Efficient implementation equivalent to the following with bidirectional=False
        rnn = nn.RNN(input_size, hidden_size, num_layers)
        params = dict(rnn.named_parameters())
        def forward(x, hx=None, batch_first=False):
            if batch_first:
                x = x.transpose(0, 1)
            seq_len, batch_size, _ = x.size()
            if hx is None:
                hx = torch.zeros(rnn.num_layers, batch_size, rnn.hidden_size)
            h_t_minus_1 = hx.clone()
            h_t = hx.clone()
            output = []
            for t in range(seq_len):
                for layer in range(rnn.num_layers):
                    input_t = x[t] if layer == 0 else h_t[layer - 1]
                    h_t[layer] = torch.tanh(
                        input_t @ params[f"weight_ih_l{layer}"].T
                        + h_t_minus_1[layer] @ params[f"weight_hh_l{layer}"].T
                        + params[f"bias_hh_l{layer}"]
                        + params[f"bias_ih_l{layer}"]
                    )
                output.append(h_t[-1].clone())
                h_t_minus_1 = h_t.clone()
            output = torch.stack(output)
            if batch_first:
                output = output.transpose(0, 1)
            return output, h_t

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, hx
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **hx**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
          state for the input sequence batch. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. include:: ../cudnn_rnn_determinism.md
        :parser: myst_parser.sphinx_

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None: ...

    @overload
    def __init__(self, *args, **kwargs) -> None: ...

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        if len(args) > 3:
            self.nonlinearity = args[3]
            args = args[:3] + args[4:]
        else:
            self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if self.nonlinearity == "tanh":
            mode = "RNN_TANH"
        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(
                f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'."
            )
        super().__init__(mode, *args, **kwargs)

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: Tensor,
        hx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: PackedSequence,
        hx: Tensor | None = None,
    ) -> tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):
        """
        Runs the forward pass.
        """
        self._update_flat_weights()

        num_directions = 2 if self.bidirectional else 1
        orig_input = input

        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            # script() is unhappy when max_batch_size is different type in cond branches, so we duplicate
            if hx is None:
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"RNN: Expected input to be 2D or 3D, got {input.dim()}D tensor instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

        if hx is None:
            raise AssertionError("hx must not be None")
        self.check_forward_args(input, hx, batch_sizes)
        if self.mode != "RNN_TANH" and self.mode != "RNN_RELU":
            raise AssertionError(f"mode must be RNN_TANH or RNN_RELU, got {self.mode}")
        if batch_sizes is None:
            if self.mode == "RNN_TANH":
                # pyrefly: ignore [no-matching-overload]
                result = _VF.rnn_tanh(
                    input,
                    hx,
                    self._flat_weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                    self.batch_first,
                )
            else:
                # pyrefly: ignore [no-matching-overload]
                result = _VF.rnn_relu(
                    input,
                    hx,
                    self._flat_weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                    self.batch_first,
                )
        else:
            if self.mode == "RNN_TANH":
                # pyrefly: ignore [no-matching-overload]
                result = _VF.rnn_tanh(
                    input,
                    batch_sizes,
                    hx,
                    self._flat_weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                )
            else:
                # pyrefly: ignore [no-matching-overload]
                result = _VF.rnn_relu(
                    input,
                    batch_sizes,
                    hx,
                    self._flat_weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                )

        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)

        if not is_batched:  # type: ignore[possibly-undefined]
            output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
            hidden = hidden.squeeze(1)

        return output, self.permute_hidden(hidden, unsorted_indices)


# XXX: LSTM and GRU implementation is different from RNNBase, this is because:
# 1. we want to support nn.LSTM and nn.GRU in TorchScript and TorchScript in
#    its current state could not support the python Union Type or Any Type
# 2. TorchScript static typing does not allow a Function or Callable type in
#    Dict values, so we have to separately call _VF instead of using _rnn_impls
# 3. This is temporary only and in the transition state that we want to make it
#    on time for the release
#
# More discussion details in https://github.com/pytorch/pytorch/pull/23266
#
# TODO: remove the overriding implementations for LSTM and GRU when TorchScript
# support expressing these two modules generally.


class LSTM(RNNBase):
    r"""__init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,proj_size=0,device=None,dtype=None)

    Apply a multi-layer long short-term memory (LSTM) RNN to an input sequence.
    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state of each layer will be multiplied by a learnable projection
    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output
    of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact
    dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

    Inputs: input, (h_0, c_0)
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          initial hidden state for each element in the input sequence.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **c_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          initial cell state for each element in the input sequence.
          Defaults to zeros if (h_0, c_0) is not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{cell} ={} & \text{hidden\_size} \\
                H_{out} ={} & \text{proj\_size if } \text{proj\_size}>0 \text{ otherwise hidden\_size} \\
            \end{aligned}

    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence. When ``bidirectional=True``, `output` will contain
          a concatenation of the forward and reverse hidden states at each time step in the sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          final hidden state for each element in the sequence. When ``bidirectional=True``,
          `h_n` will contain a concatenation of the final forward and reverse hidden states, respectively.
        * **c_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          final cell state for each element in the sequence. When ``bidirectional=True``,
          `c_n` will contain a concatenation of the final forward and reverse cell states, respectively.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`. If
            ``proj_size > 0`` was specified, the shape will be
            `(4*hidden_size, num_directions * proj_size)` for `k > 0`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``
            was specified, the shape will be `(4*hidden_size, proj_size)`.
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
        weight_hr_l[k] : the learnable projection weights of the :math:`\text{k}^{th}` layer
            of shape `(proj_size, hidden_size)`. Only present when ``proj_size > 0`` was
            specified.
        weight_ih_l[k]_reverse: Analogous to `weight_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hh_l[k]_reverse:  Analogous to `weight_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_ih_l[k]_reverse:  Analogous to `bias_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_hh_l[k]_reverse:  Analogous to `bias_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hr_l[k]_reverse:  Analogous to `weight_hr_l[k]` for the reverse direction.
            Only present when ``bidirectional=True`` and ``proj_size > 0`` was specified.

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        For bidirectional LSTMs, `h_n` is not equivalent to the last element of `output`; the
        former contains the final forward and reverse hidden states, while the latter contains the
        final forward hidden state and the initial reverse hidden state.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. note::
        ``proj_size`` should be smaller than ``hidden_size``.

    .. include:: ../cudnn_rnn_determinism.md
        :parser: myst_parser.sphinx_

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None: ...

    @overload
    def __init__(self, *args, **kwargs) -> None: ...

    def __init__(self, *args, **kwargs):
        super().__init__("LSTM", *args, **kwargs)

    def get_expected_cell_size(
        self, input: Tensor, batch_sizes: Tensor | None
    ) -> tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions,
            mini_batch,
            self.hidden_size,
        )
        return expected_hidden_size

    # In the future, we should prevent mypy from applying contravariance rules here.
    # See torch/nn/modules/module.py::_forward_unimplemented
    # pyrefly: ignore [bad-override]
    def check_forward_args(
        self,
        input: Tensor,
        hidden: tuple[Tensor, Tensor],  # type: ignore[override]
        batch_sizes: Tensor | None,
    ) -> None:
        self.check_input(input, batch_sizes)
        self.check_hidden_size(
            hidden[0],
            self.get_expected_hidden_size(input, batch_sizes),
            "Expected hidden[0] size {}, got {}",
        )
        self.check_hidden_size(
            hidden[1],
            self.get_expected_cell_size(input, batch_sizes),
            "Expected hidden[1] size {}, got {}",
        )

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    def permute_hidden(  # type: ignore[override]
        self,
        hx: tuple[Tensor, Tensor],
        permutation: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(
            hx[1], permutation
        )

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    @overload  # type: ignore[override]
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        pass

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: PackedSequence,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[PackedSequence, tuple[Tensor, Tensor]]:
        pass

    def forward(self, input, hx=None):
        self._update_flat_weights()

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            if hx is None:
                h_zeros = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    real_hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                c_zeros = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                hx = (h_zeros, c_zeros)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"LSTM: Expected input to be 2D or 3D, got {input.dim()}D instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                h_zeros = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    real_hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                c_zeros = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                hx = (h_zeros, c_zeros)
                self.check_forward_args(input, hx, batch_sizes)
            else:
                if is_batched:
                    if hx[0].dim() != 3 or hx[1].dim() != 3:
                        msg = (
                            "For batched 3-D input, hx and cx should "
                            f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = (
                            "For unbatched 2-D input, hx and cx should "
                            f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                self.check_forward_args(input, hx, batch_sizes)
                hx = self.permute_hidden(hx, sorted_indices)

        if batch_sizes is None:
            # pyrefly: ignore [no-matching-overload]
            result = _VF.lstm(
                input,
                hx,
                self._flat_weights,  # type: ignore[arg-type]
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            # pyrefly: ignore [no-matching-overload]
            result = _VF.lstm(
                input,
                batch_sizes,
                hx,
                self._flat_weights,  # type: ignore[arg-type]
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:  # type: ignore[possibly-undefined]
                output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)


class GRU(RNNBase):
    r"""__init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None)

    Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` or
          :math:`(D * \text{num\_layers}, N, H_{out})`
          containing the initial hidden state for the input sequence. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the GRU, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for the input sequence.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. note::
        The calculation of new gate :math:`n_t` subtly differs from the original paper and other frameworks.
        In the original implementation, the Hadamard product :math:`(\odot)` between :math:`r_t` and the
        previous hidden state :math:`h_{(t-1)}` is done before the multiplication with the weight matrix
        `W` and addition of bias:

        .. math::
            \begin{aligned}
                n_t = \tanh(W_{in} x_t + b_{in} + W_{hn} ( r_t \odot h_{(t-1)} ) + b_{hn})
            \end{aligned}

        This is in contrast to PyTorch implementation, which is done after :math:`W_{hn} h_{(t-1)}`

        .. math::
            \begin{aligned}
                n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn}))
            \end{aligned}

        This implementation differs on purpose for efficiency.

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None: ...

    @overload
    def __init__(self, *args, **kwargs) -> None: ...

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        super().__init__("GRU", *args, **kwargs)

    @overload  # type: ignore[override]
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: Tensor,
        hx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: PackedSequence,
        hx: Tensor | None = None,
    ) -> tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):
        self._update_flat_weights()

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            if hx is None:
                num_directions = 2 if self.bidirectional else 1
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"GRU: Expected input to be 2D or 3D, got {input.dim()}D instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                num_directions = 2 if self.bidirectional else 1
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            # pyrefly: ignore [no-matching-overload]
            result = _VF.gru(
                input,
                hx,
                self._flat_weights,  # type: ignore[arg-type]
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            # pyrefly: ignore [no-matching-overload]
            result = _VF.gru(
                input,
                batch_sizes,
                hx,
                self._flat_weights,  # type: ignore[arg-type]
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:  # type: ignore[possibly-undefined]
                output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
                hidden = hidden.squeeze(1)

            return output, self.permute_hidden(hidden, unsorted_indices)


class RNNCellBase(Module):
    __constants__ = ["input_size", "hidden_size", "bias"]

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(
            torch.empty((num_chunks * hidden_size, input_size), **factory_kwargs)
        )
        self.weight_hh = Parameter(
            torch.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs)
        )
        if bias:
            self.bias_ih = Parameter(
                torch.empty(num_chunks * hidden_size, **factory_kwargs)
            )
            self.bias_hh = Parameter(
                torch.empty(num_chunks * hidden_size, **factory_kwargs)
            )
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class RNNCell(RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

    If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``

    Inputs: input, hidden
        - **input**: tensor containing input features
        - **hidden**: tensor containing the initial hidden state
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "nonlinearity"]
    nonlinearity: str

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=1, **factory_kwargs)
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"RNNCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"RNNCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        if self.nonlinearity == "tanh":
            ret = _VF.rnn_tanh_cell(
                input,
                hx,
                self.weight_ih,
                self.weight_hh,
                self.bias_ih,
                self.bias_hh,
            )
        elif self.nonlinearity == "relu":
            ret = _VF.rnn_relu_cell(
                input,
                hx,
                self.weight_ih,
                self.weight_hh,
                self.bias_ih,
                self.bias_hh,
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(f"Unknown nonlinearity: {self.nonlinearity}")

        if not is_batched:
            ret = ret.squeeze(0)

        return ret


class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f \odot c + i \odot g \\
        h' = o \odot \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)` or `(input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial hidden state
        - **c_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial cell state

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next hidden state
        - **c_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next cell state

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
        >>> input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
        >>> hx = torch.randn(3, 20)  # (batch, hidden_size)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(input.size()[0]):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
        >>> output = torch.stack(output, dim=0)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"LSTMCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None:
            for idx, value in enumerate(hx):
                if value.dim() not in (1, 2):
                    raise ValueError(
                        f"LSTMCell: Expected hx[{idx}] to be 1D or 2D, got {value.dim()}D instead"
                    )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = _VF.lstm_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret


class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r \odot (W_{hn} h + b_{hn})) \\
        h' = (1 - z) \odot n + z \odot h
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, hidden
        - **input** : tensor containing input features
        - **hidden** : tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** : tensor containing the next hidden state
          for each element in the batch

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        ret = _VF.gru_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            ret = ret.squeeze(0)

        return ret
<|endoftext|># mypy: allow-untyped-defs
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from .lazy import LazyModuleMixin
from .module import Module


__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]


class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return input


class Linear(Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
