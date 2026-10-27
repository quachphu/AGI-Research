        >>>      if i > swa_start:
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    """

    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float,
        anneal_epochs=10,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        last_epoch=-1,
    ) -> None:
        swa_lrs = _format_param("swa_lr", optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups, strict=True):
            group["swa_lr"] = swa_lr
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                "anneal_strategy must be one of 'cos' or 'linear', "
                f"instead got {anneal_strategy}"
            )
        self._set_anneal_func(anneal_strategy)
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(
                f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}"
            )
        self.anneal_epochs = anneal_epochs
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _linear_anneal(t):
        return t

    @staticmethod
    def _cosine_anneal(t):
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    @override
    def get_lr(self):
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Uses :attr:`anneal_func` to interpolate between each group's
        ``group["lr"]`` and ``group["swa_lr"]`` over :attr:`anneal_epochs`
        epochs. Once :attr:`anneal_epochs` is reached, keeps the learning rate
        fixed at ``group["swa_lr"]``.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        # `_get_lr_called_within_step` is only available `_enable_get_lr_call`,
        # so we ignore the type error here. See `LRScheduler.step()` for more details.
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )
        # Set in `LRScheduler._initial_step()`
        step = self._step_count - 1
        if self.anneal_epochs == 0:
            step = max(1, step)
        # pyrefly: ignore [no-matching-overload]
        prev_t = max(0, min(1, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [
            self._get_initial_lr(group["lr"], group["swa_lr"], prev_alpha)
            for group in self.optimizer.param_groups
        ]
        # pyrefly: ignore [no-matching-overload]
        t = max(0, min(1, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        return [
            group["swa_lr"] * alpha + lr * (1 - alpha)
            for group, lr in zip(self.optimizer.param_groups, prev_lrs, strict=True)
        ]

    def _set_anneal_func(self, anneal_strategy: Literal["cos", "linear"]) -> None:
        self._anneal_strategy = anneal_strategy
        if anneal_strategy == "cos":
            self.anneal_func = self._cosine_anneal
        else:
            self.anneal_func = self._linear_anneal

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer or anneal_func.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "anneal_func")
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        self._set_anneal_func(self._anneal_strategy)
<|endoftext|># mypy: allow-untyped-defs
r"""Implementation for the NAdam algorithm."""

from typing import cast

import torch
from torch import Tensor

from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["NAdam", "nadam"]


class NAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum_decay:
            raise ValueError(f"Invalid momentum_decay value: {momentum_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum_decay": momentum_decay,
            "decoupled_weight_decay": decoupled_weight_decay,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0:
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = (
                            torch.tensor(
                                step_val, dtype=_get_scalar_dtype(), device=p.device
                            )
                            if group["capturable"]
                            else torch.tensor(step_val, dtype=_get_scalar_dtype())
                        )
                    if not torch.is_tensor(p_state["mu_product"]):
                        mu_prod_val = p_state["mu_product"]
                        p_state["mu_product"] = (
                            torch.tensor(
                                mu_prod_val, dtype=_get_scalar_dtype(), device=p.device
                            )
                            if group["capturable"]
                            else torch.tensor(mu_prod_val, dtype=_get_scalar_dtype())
                        )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` and `mu_product` on CPU if capturable is False.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["mu_product"] = (
                        torch.ones((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(1.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                mu_products.append(state["mu_product"])
                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            mu_products: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = cast(tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
            )

            nadam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum_decay=group["momentum_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )

        return loss


NAdam.__doc__ = (
    r"""Implements NAdam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma_t \text{ (lr)}, \: \beta_1,\beta_2 \text{ (betas)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm} \: \lambda \text{ (weight decay)}, \:\psi \text{ (momentum decay)}    \\
            &\hspace{13mm} \: \textit{decoupled\_weight\_decay}, \:\textit{maximize}             \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1}                                       \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm}\textbf{if} \: \textit{decoupled\_weight\_decay}                       \\
            &\hspace{15mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}                    \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm} \mu_t \leftarrow \beta_1 \big(1 - \frac{1}{2}  0.96^{t \psi} \big)     \\
            &\hspace{5mm} \mu_{t+1} \leftarrow \beta_1 \big(1 - \frac{1}{2} 0.96^{(t+1)\psi}\big)\\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow \mu_{t+1} m_t/(1-\prod_{i=1}^{t+1}\mu_i)\\[-1.ex]
            & \hspace{11mm} + (1-\mu_t) g_t /(1-\prod_{i=1}^{t} \mu_{i})                         \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Incorporating Nesterov Momentum into Adam`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum_decay (float, optional): momentum decay (default: 4e-3)
        decoupled_weight_decay (bool, optional): whether to decouple the weight
            decay as in AdamW to obtain NAdamW. If True, the algorithm does not
            accumulate weight decay in the momentum nor variance. (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}

    .. _Incorporating Nesterov Momentum into Adam:
        https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    """
)


def _single_tensor_nadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    mu_products: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
) -> None:
    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu_product = mu_products[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            if not (
                param.device.type == mu_product.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ):
                raise AssertionError(
                    f"If capturable=True, params, mu_products and state_steps must be "
                    f"on supported devices: {capturable_supported_devices}."
                )

        # update step
        step_t += 1

        if capturable:
            step = step_t
        else:
            step = _get_value(step_t)

        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform step weight decay
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        # calculate the momentum cache \mu^{t} and \mu^{t+1}
        mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
        mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))

        # update mu_product
        mu_product *= mu

        # decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.div(bias_correction2).sqrt()

        if differentiable or capturable:
            denom = denom.add(eps)
            # Make autograd track the operations
            # by updating the grad and exp_avg directly and not using the
            # scalar "value" argument of addcdiv.
            mu_product_next = mu_product * mu_next
            grad = grad * (-lr * (1.0 - mu) / (1.0 - mu_product))
            exp_avg = exp_avg * (-lr * mu_next / (1.0 - mu_product_next))
            param.addcdiv_(grad, denom)
            param.addcdiv_(exp_avg, denom)
        else:
            mu_product_next = _get_value(mu_product) * mu_next
            denom.add_(eps)
            param.addcdiv_(
                grad,
                denom,
                value=(-lr * (1.0 - mu) / (1.0 - _get_value(mu_product))),  # type: ignore[arg-type]
            )
            param.addcdiv_(
                exp_avg,
                denom,
                value=cast(float, (-lr * mu_next) / (1.0 - mu_product_next)),
            )


def _multi_tensor_nadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    mu_products: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
) -> None:
    if len(params) == 0:
        return

    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        if not all(
            p.device.type == mp.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, mp, step in zip(params, mu_products, state_steps, strict=True)
        ):
            raise AssertionError(
                "If capturable=True, "
                "params, mu_products, and state_steps must be on supported devices: "
                f"{capturable_supported_devices}."
            )

    lr = _to_scalar(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps]  # type: ignore[list-item]
    )
    for (
        grouped_params_,
        grouped_grads_,
        grouped_exp_avgs_,
        grouped_exp_avg_sqs_,
        grouped_mu_products_,
        grouped_state_steps_,
    ), _ in grouped_tensors.values():
        grouped_params = cast(list[Tensor], grouped_params_)
        grouped_grads = cast(list[Tensor], grouped_grads_)
        grouped_exp_avgs = cast(list[Tensor], grouped_exp_avgs_)
        grouped_exp_avg_sqs = cast(list[Tensor], grouped_exp_avg_sqs_)
        grouped_mu_products = cast(list[Tensor], grouped_mu_products_)
        grouped_state_steps = cast(list[Tensor], grouped_state_steps_)

        # handle complex
        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs
            )

        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and grouped_state_steps[0].is_cpu:
            torch._foreach_add_(
                grouped_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(grouped_state_steps, 1)

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform step weight decay
                torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            else:
                # Reuse the intermediate memory (grouped_grads) already allocated for maximize
                if maximize:
                    torch._foreach_add_(
                        grouped_grads, grouped_params, alpha=weight_decay
                    )
                else:
                    grouped_grads = torch._foreach_add(  # type: ignore[assignment]
                        grouped_grads, grouped_params, alpha=weight_decay
                    )

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)

        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2
        )

        exp_avg_sq_sqrt = torch._foreach_sqrt(grouped_exp_avg_sqs)

        bias_correction_sqrt: tuple[Tensor, ...] | list[Tensor]
        mus: tuple[Tensor, ...] | list[Tensor]
        mu_nexts: tuple[Tensor, ...] | list[Tensor]
        if capturable:
            # mus will be beta1 * (1 - 0.5 * 0.96 ** (step * momentum_decay))
            exponent = torch._foreach_mul(grouped_state_steps, momentum_decay)
            mus = torch._foreach_pow(0.96, exponent)
            torch._foreach_mul_(mus, -0.5)
            torch._foreach_add_(mus, 1.0)
            torch._foreach_mul_(mus, beta1)

            # mu_nexts will be beta1 * (1 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
            torch._foreach_add_(exponent, momentum_decay)
            mu_nexts = torch._foreach_pow(0.96, exponent)
            torch._foreach_mul_(mu_nexts, -0.5)
            torch._foreach_add_(mu_nexts, 1.0)
            torch._foreach_mul_(mu_nexts, beta1)

            # save peak memory as we don't need exponent anymore
            del exponent

            bias_correction_sqrt = torch._foreach_pow(beta2, grouped_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction_sqrt, 1.0)
            torch._foreach_neg_(bias_correction_sqrt)
            torch._foreach_sqrt_(bias_correction_sqrt)
        else:
            bias_correction_sqrt = [
                (1 - beta2 ** _get_value(step)) ** 0.5 for step in grouped_state_steps
            ]
            mus = [
                beta1 * (1.0 - 0.5 * (0.96 ** (_get_value(step) * momentum_decay)))
                for step in grouped_state_steps
            ]
            mu_nexts = [
                beta1
                * (1.0 - 0.5 * (0.96 ** ((_get_value(step) + 1) * momentum_decay)))
                for step in grouped_state_steps
            ]

        # update mu_products
        torch._foreach_mul_(grouped_mu_products, mus)

        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)

        # explicitly delete bias_correction refs to save memory
        del bias_correction_sqrt

        if capturable:
            # Build up the step_size multiplier for grad, reusing mus' memory
            torch._foreach_sub_(mus, 1.0)
            torch._foreach_mul_(mus, lr)
            # foreach_sub doesn't allow a scalar as the first arg
            denom = torch._foreach_sub(grouped_mu_products, 1.0)
            torch._foreach_neg_(denom)
            torch._foreach_div_(mus, denom)
            # - lr * (1 - mu) / (1 - mu_product)
            step_size_grads = mus
            # explicitly delete denom to save memory
            del denom

            # Build up the step_size multiplier for exp_avg, reusing mu_nexts' memory
            denom = torch._foreach_mul(grouped_mu_products, mu_nexts)
            torch._foreach_mul_(mu_nexts, lr)
            # foreach_sub doesn't allow a scalar as the first arg, but it's okay because
            # we need a negative here anyway
            torch._foreach_sub_(denom, 1.0)
            torch._foreach_div_(mu_nexts, denom)
            # - lr * mu_next / (1 - mu_product * mu_next)
            step_size_expavg = mu_nexts
            # explicitly delete denom to save memory
            del denom

            # we cannot inplace into step_size_grads cuz it is a list of ScalarTensors
            # and mul'ing with grouped_grads will result in a list of bigger Tensors
            numerator = torch._foreach_mul(step_size_grads, grouped_grads)
            torch._foreach_addcmul_(numerator, step_size_expavg, grouped_exp_avgs)

            # finally, update params
            torch._foreach_addcdiv_(grouped_params, numerator, exp_avg_sq_sqrt)
        else:
            step_size_grads = _stack_if_compiling(
                [
                    (_get_value(lr) * (1.0 - mu) / (1.0 - _get_value(mu_product))) * -1
                    for mu_product, mu in zip(grouped_mu_products, mus, strict=True)
                ]
            )
            step_size_expavg = _stack_if_compiling(
                [
                    (
                        _get_value(lr)
                        * mu_next
                        / (1.0 - _get_value(mu_product) * mu_next)
                    )
                    * -1
                    for mu_product, mu_next in zip(
                        grouped_mu_products, mu_nexts, strict=True
                    )
                ]
            )

            torch._foreach_addcdiv_(  # type: ignore[arg-type]
                grouped_params,
                grouped_grads,
                exp_avg_sq_sqrt,
                step_size_grads,
            )
            torch._foreach_addcdiv_(  # type: ignore[arg-type]
                grouped_params,
                grouped_exp_avgs,
                exp_avg_sq_sqrt,
                step_size_expavg,
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_nadam)
def nadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    mu_products: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    decoupled_weight_decay: bool = False,
    foreach: bool | None = None,
    capturable: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    maximize: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
) -> None:
    r"""Functional API that performs NAdam algorithm computation.

    See :class:`~torch.optim.NAdam` for details.
    """
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if not all(isinstance(t, torch.Tensor) for t in mu_products):
        raise RuntimeError(
            "API has changed, `mu_products` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_nadam
    else:
        func = _single_tensor_nadam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        momentum_decay=momentum_decay,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
        eps=eps,
        capturable=capturable,
        differentiable=differentiable,
        has_complex=has_complex,
    )
<|endoftext|># mypy: allow-untyped-defs
from typing import cast

import torch
from torch import Tensor

from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict,
    DeviceDtypeDict,
    Optimizer,
    ParamsT,
)


__all__ = ["Adam", "adam"]


class Adam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        decoupled_weight_decay: bool = False,
    ) -> None:
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float))
            or (isinstance(betas[0], Tensor) and isinstance(betas[1], Tensor))
        ):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(betas[0], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[0] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[0].numel() != 1:
                raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[1] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[1].numel() != 1:
                raise ValueError("Tensor betas[1] must be 1-element")
        betas = (_to_scalar(betas[0]), _to_scalar(betas[1]))

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": fused,
            "decoupled_weight_decay": decoupled_weight_decay,
        }
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Support AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    if group["fused"]:
                        _device_dtype_check_for_fused(p)
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                # Foreach without capturable does not support a tensor lr
                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )

        return loss


Adam.__doc__ = (
    r"""Implements Adam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize},  \: \epsilon \text{ (epsilon)}                              \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: v_0^{max}\leftarrow 0          \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm} v_t^{max} \leftarrow \mathrm{max}(v_{t-1}^{max},v_t)                  \\
            &\hspace{10mm}\widehat{v_t} \leftarrow v_t^{max}/\big(1-\beta_2^t \big)              \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                  \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying fused=True or capturable=True.
        betas (tuple[float | Tensor, float | Tensor], optional):
            coefficients used for computing running averages of gradient and
            its square. If a tensor is provided, must be 1-element. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): if True, this optimizer is
            equivalent to AdamW and the algorithm will not accumulate weight
            decay in the momentum nor variance. (default: False)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}
        {_fused_doc}
    .. Note::
        A prototype implementation of Adam and AdamW for MPS supports `torch.float32` and `torch.float16`.
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """
)


def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: float | Tensor,
    beta2: float | Tensor,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
) -> None:
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        if not isinstance(lr, float):
            raise AssertionError(f"Expected lr to be a float, but got {type(lr)}")
        if not isinstance(beta1, float):
            raise AssertionError(f"Expected beta1 to be a float, but got {type(beta1)}")
        if not isinstance(beta2, float):
            raise AssertionError(f"Expected beta2 to be a float, but got {type(beta2)}")
    else:
        lr = _to_scalar(lr)
        beta1 = _to_scalar(beta1)
        beta2 = _to_scalar(beta2)

    # We only shuffle around the beta when it is a Tensor, otherwise, we prefer
    # treating it as a scalar.
    # Note: ensure type declaration is under conditional check for isinstance
    # or else torchscript will get cranky about the DeviceDict type.
    if isinstance(beta1, Tensor):
        beta1_dict: DeviceDtypeDict | None = {(beta1.device, beta1.dtype): beta1}
    else:
        beta1_dict = None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            if not (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ):
                raise AssertionError(
                    f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
                )

        # update step
        step_t += 1

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform step weight decay
                param.mul_(1 - lr * weight_decay)
            else:
                # Nested if is necessary to bypass jitscript rules
                if differentiable and isinstance(weight_decay, Tensor):
                    if weight_decay.requires_grad:
                        grad = grad.addcmul_(param.clone(), weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        device = param.device

        if beta1_dict is not None:
            dtype = param.dtype  # type: ignore[union-attr]

            # cast to workaround https://github.com/pytorch/pytorch/issues/140601
            key = (device, dtype)
            if key not in beta1_dict:
                beta1_dict[key] = beta1.to(  # type: ignore[union-attr]
                    device=device, dtype=dtype, non_blocking=True
                )

            device_beta1: float | Tensor = beta1_dict[key]
        else:
            device_beta1 = beta1

        # Decay the first and second moment running average coefficient

        exp_avg.lerp_(grad, 1 - device_beta1)

        # Nested if is necessary to bypass jitscript rules
        if differentiable and isinstance(beta2, Tensor):
            if beta2.requires_grad:
                # Using lerp to only use 2 operations bc addcmul's value cannot be a tensor
                # Showing equivalence of differentiable path and nondifferentiable path
                # expavg * b2 + grad^2 * (1-b2)
                #           add expavg * (1-b2) - expavg * (1-b2) = 0
                # expavg * b2 + expavg * (1-b2) - expavg * (1-b2) + grad^2 * (1-b2)
                # expavg - expavg * (1-b2) + grad^2 * (1-b2)
                # expavg + (grad^2 - expavg) * (1-b2)
                # expavg.lerp(grad^2, 1-beta2)
                exp_avg_sq.lerp_(torch.square(grad), weight=1 - beta2)
            else:
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=cast(float, 1 - beta2)
                )
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # type: ignore[arg-type]

        if capturable or differentiable:
            step = step_t

            # Nested if is necessary to bypass jitscript rules
            if differentiable and isinstance(beta1, Tensor):
                if beta1.requires_grad:
                    bias_correction1 = 1 - beta1 ** step.clone()
                else:
                    bias_correction1 = 1 - beta1**step
            else:
                bias_correction1 = 1 - beta1**step

            # Nested if is necessary to bypass jitscript rules
            if differentiable and isinstance(beta2, Tensor):
                if beta2.requires_grad:
                    bias_correction2 = 1 - beta2 ** step.clone()
                else:
                    bias_correction2 = 1 - beta2**step
            else:
                bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            if differentiable:
                param.addcdiv_(exp_avg.clone(), denom)
            else:
                param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = bias_correction2**0.5

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)  # type: ignore[arg-type]

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: float | Tensor,
    beta2: float | Tensor,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
) -> None:
    if len(params) == 0:
        return

    if isinstance(lr, Tensor):
        if not capturable:
            raise RuntimeError(
                "lr as a Tensor is not supported for capturable=False and foreach=True"
            )
        if lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")

    if isinstance(beta1, Tensor):
        if not capturable:
            raise ValueError(
                "beta1 as a Tensor is not supported for capturable=False and foreach=True"
            )
        if beta1.numel() != 1:
            raise ValueError("Tensor beta1 must be 1-element")

    if isinstance(beta2, Tensor):
        if not capturable:
            raise ValueError(
                "beta2 as a Tensor is not supported for capturable=False and foreach=True"
            )
        if beta2.numel() != 1:
            raise ValueError("Tensor beta2 must be 1-element")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        if not all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps, strict=True)
        ):
            raise AssertionError(
                f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
            )

    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")

    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    lr = _to_scalar(lr)
    beta1 = _to_scalar(beta1)
    beta2 = _to_scalar(beta2)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )

    # We only shuffle around the beta when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    beta1_dict: DeviceDict | None = (  # type: ignore[attr-defined]
        {beta1.device: beta1}
        if isinstance(beta1, Tensor) and str(beta1.device) != "cpu"
        else None
    )

    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_max_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_exp_avgs = cast(list[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(list[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device = device_params[0].device
        if beta1_dict is not None and device not in beta1_dict:
            beta1_dict[device] = beta1.to(device=device, non_blocking=True)  # type: ignore[union-attr, attr-defined]

        device_beta1 = beta1_dict[device] if beta1_dict else beta1

        # Handle complex parameters
        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                _view_as_real(
                    device_params,
                    device_grads,
                    device_exp_avgs,
                    device_exp_avg_sqs,
                    device_max_exp_avg_sqs,
                )
            else:
                _view_as_real(
                    device_params, device_grads, device_exp_avgs, device_exp_avg_sqs
                )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform step weight decay
                torch._foreach_mul_(device_params, 1 - lr * weight_decay)
            else:
                # Reuse the intermediate memory (device_grads) already allocated for maximize
                if maximize:
                    torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
                else:
                    device_grads = torch._foreach_add(  # type: ignore[assignment]
                        device_grads, device_params, alpha=weight_decay
                    )

        # Decay the first and second moment running average coefficient
        # Use device beta1 if beta1 is a tensor to ensure all
        # tensors are on the same device
        torch._foreach_lerp_(
            device_exp_avgs, device_grads, cast(float, 1 - device_beta1)
        )

        torch._foreach_mul_(device_exp_avg_sqs, beta2)

        # Due to the strictness of the _foreach_addcmul API, we can't have a single
        # tensor scalar as the scalar arg (only python number is supported there)
        # as a result, separate out the value mul
        # Filed https://github.com/pytorch/pytorch/issues/139795
        if isinstance(beta2, torch.Tensor):
            scaled_device_grads = torch._foreach_mul(device_grads, 1 - beta2)  # type: ignore[assignment]
            value = 1.0
        else:
            scaled_device_grads = device_grads  # type: ignore[assignment]
            value = 1 - beta2

        torch._foreach_addcmul_(
            device_exp_avg_sqs, scaled_device_grads, device_grads, value
        )

        # Delete the local intermediate(s) since they won't be used anymore to save on peak memory
        del device_grads
        del scaled_device_grads

        bias_correction1: tuple[Tensor, ...] | list[Tensor]
        bias_correction2: tuple[Tensor, ...] | list[Tensor]
        bias_correction2_sqrt: tuple[Tensor, ...] | list[Tensor]

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)  # type: ignore[arg-type]
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)  # type: ignore[arg-type]
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)  # type: ignore[assignment]

                # Set intermediate to the max. for normalizing running avg. of gradient when amsgrad
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [
                1 - beta1 ** _get_value(step) for step in device_state_steps
            ]
            bias_correction2 = [
                1 - beta2 ** _get_value(step) for step in device_state_steps
            ]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [bc**0.5 for bc in bias_correction2]  # type: ignore[arg-type]

            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(  # type: ignore[arg-type]
                device_params,
                device_exp_avgs,
                exp_avg_sq_sqrt,
                step_size,
            )


def _fused_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    amsgrad: bool,
    has_complex: bool,  # Needed for consistency.
    beta1: float | Tensor,
    beta2: float | Tensor,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,  # Needed for consistency.
    differentiable: bool,
    decoupled_weight_decay: bool,
) -> None:
    if not params:
        return
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    beta1 = _to_scalar(beta1)
    beta2 = _to_scalar(beta2)

    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    # We only shuffle around the lr when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    lr_dict: DeviceDict | None = (
        {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None
    )
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (device, _), (
        (
            device_params_,
            device_grads_,
            device_exp_avgs_,
            device_exp_avg_sqs_,
            device_max_exp_avg_sqs,
            device_state_steps_,
        ),
        _,
    ) in grouped_tensors.items():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_exp_avgs = cast(list[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(list[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device, non_blocking=True)
            )
        if found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(
                device, found_inf.to(device, non_blocking=True)
            )
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)  # type: ignore[union-attr]
            lr = lr_dict[device]
        torch._foreach_add_(device_state_steps, 1)
        func = torch._fused_adam_ if not decoupled_weight_decay else torch._fused_adamw_
        # pyrefly: ignore [no-matching-overload]
        func(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,  # type: ignore[arg-type]
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,  # type: ignore[arg-type]
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(
                device_state_steps, [device_found_inf] * len(device_state_steps)
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adam)
def adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: bool | None = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: bool | None = None,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
    has_complex: bool = False,
    decoupled_weight_decay: bool = False,
    *,
    amsgrad: bool,
    beta1: float | Tensor,
    beta2: float | Tensor,
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
) -> None:
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adam
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        decoupled_weight_decay=decoupled_weight_decay,
    )
<|endoftext|># mypy: allow-untyped-defs
r"""Implementation for the RAdam algorithm."""

from typing import cast

import torch
from torch import Tensor

from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["RAdam", "radam"]


class RAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        decoupled_weight_decay: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "decoupled_weight_decay": decoupled_weight_decay,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = cast(tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
            )

            radam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                has_complex=has_complex,
            )

        return loss


RAdam.__doc__ = (
    r"""Implements RAdam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \beta_1, \beta_2
                \text{ (betas)}, \: \theta_0 \text{ (params)}, \:f(\theta) \text{ (objective)}, \:
                \lambda \text{ (weight decay)}, \:\textit{maximize}                               \\
            &\hspace{13mm} \epsilon \text{ (epsilon)}, \textit{decoupled\_weight\_decay}         \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)},                                       \\
            &\hspace{18mm} \rho_{\infty} \leftarrow 2/(1-\beta_2) -1                      \\[-1.ex]
            &\rule{110mm}{0.4pt}  \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{6mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{12mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{6mm} \theta_t \leftarrow \theta_{t-1}                                       \\
            &\hspace{6mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{12mm}\textbf{if} \: \textit{decoupled\_weight\_decay}                       \\
            &\hspace{18mm} \theta_t \leftarrow \theta_{t} - \gamma \lambda \theta_{t}            \\
            &\hspace{12mm}\textbf{else}                                                          \\
            &\hspace{18mm} g_t \leftarrow g_t + \lambda \theta_{t}                               \\
            &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{6mm}\rho_t \leftarrow \rho_{\infty} -
                2 t \beta^t_2 /\big(1-\beta_2^t \big)                                    \\[0.1.ex]
            &\hspace{6mm}\textbf{if} \: \rho_t > 5                                               \\
            &\hspace{12mm} l_t \leftarrow \frac{\sqrt{ (1-\beta^t_2) }}{ \sqrt{v_t} +\epsilon  } \\
            &\hspace{12mm} r_t \leftarrow
      \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_{\infty}}{(\rho_{\infty}-4)(\rho_{\infty}-2) \rho_t}} \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t} r_t l_t        \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `On the variance of the adaptive learning rate and beyond`_.

    This implementation provides an option to use either the original weight_decay implementation as in Adam
    (where the weight_decay is applied to the gradient) or the one from AdamW (where weight_decay is applied
    to the weight) through the decoupled_weight_decay option. When decoupled_weight_decay is set to False
    (default), it uses the original Adam style weight decay, otherwise, it uses the AdamW style which
    corresponds more closely to the `author's implementation`_ in the RAdam paper. Further information
    about decoupled weight decay can be found in `Decoupled Weight Decay Regularization`_.

    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): whether to decouple the weight
            decay as in AdamW to obtain RAdamW. If True, the algorithm does not
            accumulate weight decay in the momentum nor variance. (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}

    .. _On the variance of the adaptive learning rate and beyond:
        https://arxiv.org/abs/1908.03265
    .. _author's implementation:
        https://github.com/LiyuanLucasLiu/RAdam
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    """
)


def _single_tensor_radam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    differentiable: bool,
    maximize: bool,
    capturable: bool,
    has_complex: bool,
) -> None:
    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            if not (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ):
                raise AssertionError(
                    f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
                )

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)

        # update step
        step_t += 1
        step = step_t if capturable else _get_value(step_t)

        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # correcting bias for the first moving moment
        bias_corrected_exp_avg = exp_avg / bias_correction1

        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2

        def _compute_rect():
            return (
                (rho_t - 4)
                * (rho_t - 2)
                * rho_inf
                / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
            ) ** 0.5

        def _compute_adaptive_lr():
            exp_avg_sq_sqrt = exp_avg_sq.sqrt()
            if differentiable:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add(eps)
            else:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)

            return (bias_correction2**0.5) / exp_avg_sq_sqrt

        # Compute the variance rectification term and update parameters accordingly
        if capturable:
            update = torch.where(
                rho_t > 5.0, _compute_rect() * _compute_adaptive_lr(), 1.0
            )
            param.add_(bias_corrected_exp_avg * lr * update, alpha=-1.0)
        else:
            if rho_t > 5.0:
                param.add_(
                    bias_corrected_exp_avg
                    * lr
                    * _compute_adaptive_lr()
                    * _compute_rect(),
                    alpha=-1.0,
                )
            else:
                param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)


def _multi_tensor_radam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    differentiable: bool,
    maximize: bool,
    capturable: bool,
    has_complex: bool,
) -> None:
    if len(params) == 0:
        return

    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        if not all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps, strict=True)
        ):
            raise AssertionError(
                f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
            )

    lr = _to_scalar(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (
        grouped_params_,
        grouped_grads_,
        grouped_exp_avgs_,
        grouped_exp_avg_sqs_,
        grouped_state_steps_,
    ), _ in grouped_tensors.values():
        grouped_params = cast(list[Tensor], grouped_params_)
        grouped_grads = cast(list[Tensor], grouped_grads_)
        grouped_exp_avgs = cast(list[Tensor], grouped_exp_avgs_)
        grouped_exp_avg_sqs = cast(list[Tensor], grouped_exp_avg_sqs_)
        grouped_state_steps = cast(list[Tensor], grouped_state_steps_)

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and grouped_state_steps[0].is_cpu:
            torch._foreach_add_(
                grouped_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(grouped_state_steps, 1)

        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs
            )

        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)  # type: ignore[assignment]

        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        bias_correction1: tuple[Tensor, ...] | list[Tensor]
        bias_correction2: tuple[Tensor, ...] | list[Tensor]
        rho_t_list: tuple[Tensor, ...] | list[Tensor]
        if capturable:
            bias_correction1 = torch._foreach_pow(beta2, grouped_state_steps)
            torch._foreach_neg_(bias_correction1)
            torch._foreach_add_(bias_correction1, 1)
            bias_correction2 = torch._foreach_pow(beta2, grouped_state_steps)
            torch._foreach_mul_(bias_correction2, grouped_state_steps)
            torch._foreach_mul_(bias_correction2, 2)
            torch._foreach_div_(bias_correction2, bias_correction1)
            torch._foreach_neg_(bias_correction2)
            torch._foreach_add_(bias_correction2, rho_inf)
            rho_t_list = bias_correction2
        else:
            rho_t_list = [
                rho_inf
                - 2
                * _get_value(step)
                * (beta2 ** _get_value(step))
                / (1 - beta2 ** _get_value(step))
                for step in grouped_state_steps
            ]

        if weight_decay != 0:
            if decoupled_weight_decay:
                torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            else:
                # Reuse the intermediate memory (grouped_grads) already allocated for maximize
                if maximize:
                    torch._foreach_add_(
                        grouped_grads, grouped_params, alpha=weight_decay
                    )
                else:
                    grouped_grads = torch._foreach_add(  # type: ignore[assignment]
                        grouped_grads, grouped_params, alpha=weight_decay
                    )

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)

        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2
        )

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del grouped_grads

        if capturable:
            num = torch._foreach_sub(rho_t_list, 4)
            sub2 = torch._foreach_sub(rho_t_list, 2)
            torch._foreach_mul_(num, sub2)
            del sub2
            torch._foreach_mul_(num, rho_inf)
            rho_inf = (rho_inf - 4) * (rho_inf - 2)
            denom = torch._foreach_mul(rho_t_list, rho_inf)
            torch._foreach_div_(num, denom)
            del denom
            torch._foreach_sqrt_(num)

            # TODO(mlazos): we should try and get a foreach_where op https://github.com/pytorch/pytorch/issues/117884
            rect = [
                torch.where(rho_t > 5.0, n, 0.0)
                for n, rho_t in zip(num, rho_t_list, strict=True)
            ]
            del num
            del rho_t_list
            unrect_step_size = [torch.where(rect > 0, 0.0, 1.0) for rect in rect]
            torch._foreach_mul_(unrect_step_size, lr)

            bias_correction1 = torch._foreach_pow(beta1, grouped_state_steps)
            torch._foreach_neg_(bias_correction1)
            torch._foreach_add_(bias_correction1, 1)

            torch._foreach_div_(unrect_step_size, bias_correction1)
            torch._foreach_neg_(unrect_step_size)

            bias_correction2 = torch._foreach_pow(beta2, grouped_state_steps)
            torch._foreach_neg_(bias_correction2)
            torch._foreach_add_(bias_correction2, 1)
            torch._foreach_sqrt_(bias_correction2)
            torch._foreach_mul_(bias_correction2, lr)
            torch._foreach_mul_(bias_correction2, rect)
            del rect
            torch._foreach_neg_(bias_correction2)
            torch._foreach_div_(bias_correction2, bias_correction1)
            del bias_correction1
        else:
            rect = [
                (  # type: ignore[misc]
                    (rho_t - 4)  # type: ignore[arg-type]
                    * (rho_t - 2)
                    * rho_inf
                    / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                )
                ** 0.5
                if rho_t > 5
                else 0
                for rho_t in rho_t_list
            ]
            unrectified = [0 if rect > 0 else 1.0 for rect in rect]

            bias_correction1 = [
                1 - beta1 ** _get_value(step) for step in grouped_state_steps
            ]
            unrect_step_size = [
                (lr * rect / bc) * -1
                for rect, bc in zip(unrectified, bias_correction1, strict=True)
            ]
            bias_correction2 = [
                ((1 - beta2 ** _get_value(step)) ** 0.5) * (lr * rect / bc) * -1
                for step, rect, bc in zip(
                    grouped_state_steps, rect, bias_correction1, strict=True
                )
            ]

        buffer = torch._foreach_sqrt(grouped_exp_avg_sqs)
        torch._foreach_add_(buffer, eps)
        torch._foreach_div_(buffer, bias_correction2)
        torch._foreach_reciprocal_(buffer)
        torch._foreach_add_(buffer, unrect_step_size)

        # Here, buffer = sqrt(1 - beta2^t) * rect_step_size / (sqrt(v) + eps) + unrect_step_size
        torch._foreach_addcmul_(grouped_params, grouped_exp_avgs, buffer)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_radam)
def radam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    decoupled_weight_decay: bool = False,
    foreach: bool | None = None,
    differentiable: bool = False,
    capturable: bool = False,
    has_complex: bool = False,
    maximize: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
) -> None:
    r"""Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_radam
    else:
        func = _single_tensor_radam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
<|endoftext|># mypy: allow-untyped-defs
from typing import cast

import torch
from torch import Tensor

from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["Adamax", "adamax"]


class Adamax(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        foreach: bool | None = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "foreach": foreach,
            "maximize": maximize,
            "differentiable": differentiable,
            "capturable": capturable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_infs, state_steps
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adamax does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_inf"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            exp_avgs.append(state["exp_avg"])
            exp_infs.append(state["exp_inf"])
            state_steps.append(state["step"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_infs: list[Tensor] = []
            state_steps: list[Tensor] = []

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            foreach = group["foreach"]
            maximize = group["maximize"]
            differentiable = group["differentiable"]
            capturable = group["capturable"]

            has_complex = self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_infs, state_steps
            )

            adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=eps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
                capturable=capturable,
                has_complex=has_complex,
            )

        return loss


Adamax.__doc__ = (
    r"""Implements Adamax algorithm (a variant of Adam based on infinity norm).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)},
                \: \lambda \text{ (weight decay)},                                                \\
            &\hspace{13mm}    \epsilon \text{ (epsilon)}                                          \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                u_0 \leftarrow 0 \text{ ( infinity norm)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t      \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t               \\
            &\hspace{5mm}u_t      \leftarrow   \mathrm{max}(\beta_2 u_{t-1}, |g_{t}|+\epsilon)   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \frac{\gamma m_t}{(1-\beta^t_1) u_t} \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        {_capturable_doc}

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    """
)


def _single_tensor_adamax(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_infs: list[Tensor],
    state_steps: list[Tensor],
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
) -> None:
    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        exp_avg = exp_avgs[i]
        exp_inf = exp_infs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            if not (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ):
                raise AssertionError(
                    f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
                )

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_inf = torch.view_as_real(exp_inf)

        # Update biased first moment estimate.
        exp_avg.lerp_(grad, 1 - beta1)
        # Update the exponentially weighted infinity norm.
        if not differentiable:
            torch.maximum(
                exp_inf.mul_(beta2),
                grad.abs().add_(eps),
                out=exp_inf,
            )
        else:
            norm_buf = torch.cat(
                [exp_inf.mul_(beta2).unsqueeze(0), grad.abs().add_(eps).unsqueeze_(0)],
                0,
            )
            exp_inf.copy_(torch.amax(norm_buf, 0, keepdim=False))

        if capturable:
            # why jump through extra hoops and negate bias_correction? check out #121238
            # once fixed, we should use bias_correction with addcdiv value=-1 for readability
            neg_bias_correction = beta1**step_t - 1
            neg_bias_correction.div_(lr)
            denom = exp_inf * neg_bias_correction
            param.addcdiv_(exp_avg, denom)
        else:
            bias_correction = 1 - beta1 ** _get_value(step_t)
            clr = lr / bias_correction

            param.addcdiv_(exp_avg, exp_inf, value=-clr)  # type: ignore[arg-type]


def _multi_tensor_adamax(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_infs: list[Tensor],
    state_steps: list[Tensor],
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
) -> None:
    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    if len(params) == 0:
        return

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        if not all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps, strict=True)
        ):
            raise AssertionError(
                f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
            )

    lr = _to_scalar(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_infs, state_steps]  # type: ignore[list-item]
    )
    for (
        grouped_params_,
        grouped_grads_,
        grouped_exp_avgs_,
        grouped_exp_infs_,
        grouped_state_steps_,
    ), _ in grouped_tensors.values():
        grouped_params = cast(list[Tensor], grouped_params_)
        grouped_grads = cast(list[Tensor], grouped_grads_)
        grouped_exp_avgs = cast(list[Tensor], grouped_exp_avgs_)
        grouped_exp_infs = cast(list[Tensor], grouped_exp_infs_)
        grouped_state_steps = cast(list[Tensor], grouped_state_steps_)

        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_infs
            )

        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and grouped_state_steps[0].is_cpu:
            torch._foreach_add_(
                grouped_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(grouped_state_steps, 1)

        if weight_decay != 0:
            if maximize:
                # Reuse the intermediate memory (grouped_grads) already allocated for maximize
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
            else:
                grouped_grads = torch._foreach_add(  # type: ignore[assignment]
                    grouped_grads, grouped_params, alpha=weight_decay
                )

        # Update biased first moment estimate.
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)

        # Update the exponentially weighted infinity norm.
        torch._foreach_mul_(grouped_exp_infs, beta2)

        # in this case, we need to introduce a copy of the grads
        # since one has not been introduced previously
        if not maximize and weight_decay == 0:
            grouped_grads = torch._foreach_abs(grouped_grads)  # type: ignore[assignment]
        else:
            torch._foreach_abs_(grouped_grads)

        torch._foreach_add_(grouped_grads, eps)
        torch._foreach_maximum_(grouped_exp_infs, grouped_grads)

        bias_corrections: tuple[Tensor, ...] | list[Tensor]
        if capturable:
            bias_corrections = torch._foreach_pow(beta1, grouped_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_corrections, 1)
            torch._foreach_div_(bias_corrections, lr)

            denom = torch._foreach_mul(grouped_exp_infs, bias_corrections)
            torch._foreach_addcdiv_(grouped_params, grouped_exp_avgs, denom)
        else:
            bias_corrections = [
                1 - beta1 ** _get_value(step) for step in grouped_state_steps
            ]
            step_size = [(_get_value(lr) / bc) * -1 for bc in bias_corrections]
            torch._foreach_addcdiv_(  # type: ignore[arg-type]
                grouped_params, grouped_exp_avgs, grouped_exp_infs, step_size
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adamax)
def adamax(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_infs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: bool | None = None,
    maximize: bool = False,
    differentiable: bool = False,
    capturable: bool = False,
    has_complex: bool = False,
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    weight_decay: float,
) -> None:
    r"""Functional API that performs adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """

    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamax
    else:
        func = _single_tensor_adamax

    func(
        params,
        grads,
        exp_avgs,
        exp_infs,
        state_steps,
        eps=eps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
        capturable=capturable,
    )
<|endoftext|>"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from functools import partialmethod

from torch import optim


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


Adam = partialclass(optim.Adam, foreach=True)
AdamW = partialclass(optim.AdamW, foreach=True)
NAdam = partialclass(optim.NAdam, foreach=True)
SGD = partialclass(optim.SGD, foreach=True)
RAdam = partialclass(optim.RAdam, foreach=True)
RMSprop = partialclass(optim.RMSprop, foreach=True)
Rprop = partialclass(optim.Rprop, foreach=True)
ASGD = partialclass(optim.ASGD, foreach=True)
Adamax = partialclass(optim.Adamax, foreach=True)
Adadelta = partialclass(optim.Adadelta, foreach=True)
Adagrad = partialclass(optim.Adagrad, foreach=True)
<|endoftext|># mypy: allow-untyped-defs
import time
from collections import defaultdict
from functools import partial

import torch


# Unfortunately it doesn't seem as if there was any way to get TensorBoard to do
# anything without having TF installed, and so this file has a hard dependency on it
# as well. It really is a debugging tool, so it doesn't matter.
try:
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.util import event_pb2
    from tensorflow.python.summary.writer.writer import FileWriter
except ImportError:
    raise ImportError(
        "TensorBoard visualization of GraphExecutors requires having "
        "TensorFlow installed"
    ) from None


def dump_tensorboard_summary(graph_executor, logdir):
    with FileWriter(logdir) as w:
        pb_graph = visualize(graph_executor)
        evt = event_pb2.Event(
            wall_time=time.time(), graph_def=pb_graph.SerializeToString()
        )
        w.add_event(evt)


def visualize(graph, name_prefix="", pb_graph=None, executors_it=None):
    """Visualizes an independent graph, or a graph executor."""
    value_map = {}
    pb_graph = pb_graph or graph_pb2.GraphDef()

    if isinstance(graph, torch._C.GraphExecutorState):
        visualize_graph_executor(
            graph, name_prefix, pb_graph, partial(visualize, pb_graph=pb_graph)
        )
        return pb_graph

    # Set up an input node
    pb_graph.node.add(op="input", name=name_prefix + "input")
    for i, value in enumerate(graph.param_node().outputs()):
        value_map[value.unique()] = name_prefix + "input:" + str(i)

    visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it)

    # Gather all outputs
    return_node = pb_graph.node.add(op="output", name=name_prefix + "output")
    for value in graph.return_node().inputs():
        return_node.input.append(value_map[value.unique()])

    return pb_graph


def visualize_graph_executor(state, name_prefix, pb_graph, inline_graph):
    """Append the state of a given GraphExecutor to the graph protobuf.

    Args:
        state (GraphExecutor or GraphExecutorState): GraphExecutor to display.
        name_prefix (str): Name prefix of the containing subgraph.
        pb_graph (GraphDef): graph to append to.
        inline_graph (Callable): a function that handles setting up a value_map,
            so that some graphs in here can be inlined. This is necessary, because
            this will simply be `visualize` for the top-level GraphExecutor,
            or `inline_graph` for all nested ones.

            The signature should look like (Graph, name_prefix) -> ().
            It will be called exactly once.

    The strategy is to embed all different configurations as independent subgraphs,
    while inlining the original graph as the one that actually produces the values.
    """
    if state.autograd_fallback_graph is not None:
        visualize(
            graph=state.autograd_fallback_graph,
            name_prefix=name_prefix + "autograd_fallback/",
            pb_graph=pb_graph,
            executors_it=iter(state.autograd_fallback.executors()),
        )

    for i, (arg_spec, plan) in enumerate(state.execution_plans.items()):
        subgraph_name = name_prefix + f"plan{i}/"

        # Create a disconnected node that will keep information regarding the input
        # types of this trace. This is unfortunately a bit too verbose to be included
        # in the subgraph name.
        input_kinds = pb_graph.node.add(op="INPUT_KIND", name=subgraph_name)
        input_kinds.attr["inputs"].s = repr(arg_spec).encode("ascii")

        visualize(plan.graph, subgraph_name, pb_graph, iter(plan.code.executors()))

        # Show gradient as an independent subgraph of this plan
        if plan.grad_executor is not None:
            grad_subgraph_name = subgraph_name + "grad/"
            visualize(plan.grad_executor, grad_subgraph_name, pb_graph)

    return inline_graph(state.graph, name_prefix + "original/")


def visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it=None):
    """Recursive part of visualize (basically skips setting up the input and output nodes)."""

    def inline_graph(subgraph, name, node):
        rec_value_map = {
            inp.unique(): value_map[val.unique()]
            for inp, val in zip(subgraph.inputs(), node.inputs())
        }
        visualize_rec(
            graph=subgraph, value_map=rec_value_map, name_prefix=name, pb_graph=pb_graph
        )
        for out, val in zip(subgraph.outputs(), node.outputs()):
            value_map[val.unique()] = rec_value_map[out.unique()]

    op_id_counter: defaultdict[str, int] = defaultdict(int)

    def name_for(node):
        kind = node.kind()[node.kind().index("::") + 2 :]
        op_id_counter[kind] += 1
        return kind, name_prefix + kind + "_" + str(op_id_counter[kind])

    def add_fusion_group(node):
        op, name = name_for(node)
        inline_graph(node.g("Subgraph"), name + "/", node)

    def add_graph_executor(node):
        op, name = name_for(node)
        if executors_it is None:
            add_node(node)
        else:
            ge = next(executors_it)
            visualize_graph_executor(
                ge, name + "/", pb_graph, partial(inline_graph, node=node)
            )

    def add_node(node):
        if node.kind() == "prim::FusionGroup":
            return add_fusion_group(node)
        elif node.kind() == "prim::GraphExecutor":
            return add_graph_executor(node)
        op, name = name_for(node)
        pb_node = pb_graph.node.add(op=op, name=name)
        for value in node.inputs():
            pb_node.input.append(value_map[value.unique()])
        # TODO: handle attrs
        for i, value in enumerate(node.outputs()):
            value_map[value.unique()] = name + ":" + str(i)

    for node in graph.nodes():
        add_node(node)
<|endoftext|># mypy: allow-untyped-defs

import abc

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.auto_functionalize import FunctionalCallableWithEpilogue
from torch._higher_order_ops.utils import (
    check_input_alias_and_mutation_return_outputs,
    HopInstance,
    materialize_as_graph,
    reenter_make_fx,
    register_fake,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class BaseHOP(HigherOrderOperator, abc.ABC):
    """
    This is the "Base" HOP implementation for a HOP that looks like:

        call_subgraph_hop(subgraph, *operands, **kwargs)

    That is:
    1) the HOP stays alive until Inductor
    2) the HOP's semantics are subgraph(*operands)
    3) kwargs may be some config options but aren't passed directly to the subgraph.

    To use this, please subclass this class and override methods as necessary:
    ```
    class InvokeQuant(BaseHOP):
        def __init__(self):
            return super().__init__("invoke_quant")


    invoke_quant = InvokeQuant()


    def g(x):
        return x.sin().cos()


    @torch.compile(backend="aot_eager")
    def f(x):
        return invoke_quant(g, x, scheme="nf4")
    ```

    NOTE: don't subclass BaseHOP out of tree! That is not allowed. All
    usages must be in tree.
    """

    def __init__(self, hop_name) -> None:
        super().__init__(hop_name)

        # Set up the registrations
        # If you want to override any of these, override them in your subclass.
        self.py_autograd_impl(self._call_Autograd)
        self.py_functionalize_impl(self._call_Functionalize)
        self.py_impl(ProxyTorchDispatchMode)(self._call_ProxyTorchDispatchMode)
        register_fake(self, self._fake_impl, skip_cache=True)
        self.py_impl(DispatchKey.CompositeExplicitAutograd)(
            self._call_CompositeExplicitAutograd
        )

    def __call__(self, subgraph, *operands, **kwargs):
        if not isinstance(
            subgraph,
            (
                torch.fx.GraphModule,
                FunctionWithNoFreeVars,
                FunctionalCallableWithEpilogue,
            ),
        ):
            raise RuntimeError(
                f"{self._name}: when calling this API without torch.compile, "
                f"we require that the subgraph be a torch.fx.GraphModule (or "
                f"a function we know doesn't have free variables)."
            )
        # pyrefly: ignore [missing-attribute]
        return super().__call__(subgraph, *operands, **kwargs)

    def _call_Autograd(self, subgraph, *operands, **kwargs):
        if isinstance(subgraph, torch.fx.GraphModule):
            pass

        # We assume the subgraph doesn't mutate inputs and there is no aliasing.
        # In the PT2 stack, this is Dynamo's responsibility to figure out.
        return BaseHOPFunction.apply(self, subgraph, kwargs, *operands)

    def _call_CompositeExplicitAutograd(self, subgraph, *operands, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        mode = _get_current_dispatch_mode()
        if mode is not None:
            raise AssertionError("Mode should never be enabled for CPU/CUDA key")
        return subgraph(*operands)

    def _call_ProxyTorchDispatchMode(self, proxy_mode, subgraph, *operands, **kwargs):
        traced_graph = reenter_make_fx(subgraph)(*operands)
        if not isinstance(proxy_mode.tracer, torch.fx.Tracer):
            raise AssertionError(
                f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(proxy_mode.tracer)}"
            )
        qualname = proxy_mode.tracer.get_fresh_qualname("subgraph")
        proxy_mode.tracer.root.register_module(qualname, traced_graph)

        node_args = (traced_graph, *operands)
        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)  # type: ignore[attr-defined]
        proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, kwargs)  # type: ignore[attr-defined]
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", self, proxy_args, proxy_kwargs
        )

        out = self(subgraph, *operands, **kwargs)
        return track_tensor_tree(
            out,
            out_proxy,
            constant=None,
            tracer=proxy_mode.tracer,  # type: ignore[arg-type]
        )

    def _fake_impl(self, subgraph, *operands, **kwargs):
        return subgraph(*operands)

    # NOTE [Support input mutation of hops]
    # To support input mutation, hop's subgraph must be functionalized because many inductor passes are
    #   applied to subgraph recursively and only work on functional graph. However, we could inline an
    #   epilogue graph (i.e. the copy_) into the subgraph because this is how input mutation
    #   is implemented in the top-level graph when no hop is presented. All passes must have been and will be
    #   aware of the epilogue graph.
    #
    # Since we've supported input mutation for custom op with auto_functionalized, we share the infra for hops
    # The plan is:
    #   1. In hop's Functionalization key, it calls do_auto_functionalize_v2 if subgraph mutates input
    #   2. In do_auto_functionalize_v2:
    #       a. we functionalize the callables in hop's argument. This is to make the subgraphs functional so we
    #          could recursively run passes on them. Also the epilogue graph is inlined at the end.
    #       b. we call auto_functionalized_v2 and pass in an additional schema in order to properly invoke
    #          the hop with normalized kwargs.
    #   3. In inductor, we decompose the auto_functionalized hop by calling into the dense implementation, which
    #      copies the mutated inputs to the hop if necessary and call the hop.
    # After these steps, the rest of the inductor stack knows how to fuse the copy_ in subgraph with other ops.
    def _call_Functionalize(self, ctx, subgraph, *operands, **kwargs):
        from torch._higher_order_ops.auto_functionalize import (
            can_auto_functionalize,
            do_auto_functionalize_v2,
        )

        # invoke_quant has non-proxable argument of type InvokeQuant that
        # we cannot generate schema for.
        if self is not torch.ops.higher_order.invoke_quant_packed:
            hop_instance = HopInstance.create(self, subgraph, *operands, **kwargs)
            if can_auto_functionalize(hop_instance):
                return do_auto_functionalize_v2(
                    ctx.mode, hop_instance, (subgraph, *operands), kwargs
                )

        unwrapped_operands = ctx.unwrap_tensors(operands)
        with ctx.redispatch_to_next():
            # We assume the subgraph doesn't mutate inputs and there is no aliasing.
            # In the PT2 stack, this is Dynamo's responsibility to figure out.
            functionalized_subgraph = FunctionWithNoFreeVars(
                ctx.functionalize(subgraph)
            )
            out = self(functionalized_subgraph, *unwrapped_operands, **kwargs)
        return ctx.wrap_tensors(out)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, subgraph, *operands, **kwargs):
        from .schema import HopSchemaGenerator

        subgraph = materialize_as_graph(subgraph, operands)
        (
            inp_inp_alias,
            inp_out_alias,
            out_out_alias,
            mutated_inp_idx,
            output,
        ) = check_input_alias_and_mutation_return_outputs(subgraph)

        if not (
            len(inp_inp_alias) == 0
            and len(inp_out_alias) == 0
            and len(out_out_alias) == 0
        ):
            # TODO: turn this into an error.
            # test_foreach_map_backward_binary_foreach_map_addrecip_op fails the alias test.
            import warnings

            warnings.warn(
                "Aliasing is not supported for HOP subgraph.\n"
                f"{subgraph.print_readable(print_output=False)}\n"
                f"Alias info: inp-inp alias: {inp_inp_alias}, inp-out alias: {inp_out_alias}, out-out alias{out_out_alias}"
                f"This may lead to silent incorrectness.",
                stacklevel=2,
            )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("subgraph", subgraph)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"arg{idx}", arg, is_mutated=idx in mutated_inp_idx)

        for name, arg in kwargs.items():
            schema_gen.add_arg(name, arg, default_value=arg, kw_only=True)

        for out in output:
            schema_gen.add_output(out)

        return schema_gen.gen_schema()


class BaseHOPFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, hop, subgraph, kwargs, *operands):
        ctx.hop = hop
        ctx.operands = operands
        ctx.subgraph = subgraph
        ctx.kwargs = kwargs

        with torch._C._AutoDispatchBelowAutograd():
            return hop(subgraph, *operands, **kwargs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        subgraph = ctx.subgraph
        operands = ctx.operands
        kwargs = ctx.kwargs

        # TODO: Something special needs to happen with min cut partitioner
        with (
            suspend_functionalization(),
            disable_functional_mode(),
            torch.enable_grad(),
        ):
            with disable_proxy_modes_tracing():
                from .invoke_subgraph import create_fw_bw_graph
                from .utils import _from_fun

                fw_inputs = pytree.tree_map(_from_fun, operands)
                (
                    _,
                    joint_graph,
                    _,
                ) = create_fw_bw_graph(subgraph, fw_inputs, grad_outputs)

        # The joint graph returns (*grad_inputs, *fwd_outputs).
        # We only need the grad_inputs.
        def bwd_fn(*args):
            operands = args[: -len(grad_outputs)]
            grad_outs = args[-len(grad_outputs) :]
            result = joint_graph(*operands, *grad_outs)
            grad_inputs = result[: -len(grad_outputs)]
            return grad_inputs

        return (
            None,
            None,
            None,
            *ctx.hop(
                FunctionWithNoFreeVars(bwd_fn), *operands, *grad_outputs, **kwargs
            ),
        )


class FunctionWithNoFreeVars:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any

from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars


class ForeachMap(BaseHOP):
    def __init__(self):
        super().__init__("foreach_map")

    def __call__(self, fn, *operands, **kwargs):  # type: ignore[override]
        fn = FunctionWithNoFreeVars(fn)
        return super().__call__(fn, *operands, **kwargs)


_foreach_map = ForeachMap()


def foreach_map(op: Callable, *operands: Any, **kwargs: dict[str, Any]):
    from torch._dynamo.polyfills import foreach_map_fn

    return _foreach_map(foreach_map_fn, op, *operands, **kwargs)
<|endoftext|>import contextlib
import functools
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet
from torch._custom_class_base import CustomClassBase
from torch._higher_order_ops.utils import register_fake
from torch._library.opaque_object import register_custom_class
from torch._ops import HigherOrderOperator
from torch.autograd.graph import get_gradient_edge
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.nn.utils.stateless import _reparametrize_module


_leaf_function_module_retriever: Callable[[int], Any] | None = None

# Separate storage for the make_fx / _invoke_leaf_function_python path.
# We can't use Dynamo's register_user_object because it requires a Source
# (for bytecode generation) and adds entries to index_to_bytecode_constructor.
# In the make_fx path we have neither a Source nor bytecode generation, so we
# maintain our own dict keyed by negative indices to avoid collisions with
# Dynamo's non-negative indices.
_makefx_module_storage: dict[int, torch.nn.Module] = {}
_makefx_next_index = 0


def store_makefx_modules(modules: list[torch.nn.Module]) -> tuple[int, ...]:
    """Store modules for the make_fx path and return their assigned indices.

    Uses negative indices to avoid collisions with Dynamo's register_user_object
    which uses non-negative indices.
    """
    global _makefx_next_index
    indices = []
    for mod in modules:
        _makefx_next_index -= 1
        _makefx_module_storage[_makefx_next_index] = mod
        indices.append(_makefx_next_index)
    return tuple(indices)


def reset_makefx_module_storage() -> None:
    global _makefx_next_index
    _makefx_next_index = 0
    _makefx_module_storage.clear()


class _LeafCallable(CustomClassBase):
    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


register_custom_class(_LeafCallable, typ="symbolic")


def set_leaf_function_module_retriever(retriever: Callable[[int], Any]) -> None:
    global _leaf_function_module_retriever
    _leaf_function_module_retriever = retriever


class LeafModuleState(NamedTuple):
    """
    In dynamo, nn.Module arguments to leaf functions are converted to this
    pytree format (index, parameters, buffers). This structure is then
    flattened to produce the actual inputs to invoke_leaf_function.

    At runtime, the original module is reconstructed from it.
    """

    nn_module_index: int
    named_parameters: dict[str, torch.nn.Parameter]
    named_buffers: dict[str, torch.Tensor]


def convert_modules_to_states(values: Any, module_to_index: dict[int, int]) -> Any:
    """Replace nn.Module instances in a pytree with LeafModuleState objects.

    Args:
        values: A pytree of values that may contain nn.Module instances.
        module_to_index: Mapping from id(module) to its integer index.
    """

    def module_to_state(val: Any) -> Any:
        if isinstance(val, torch.nn.Module):
            return LeafModuleState(
                nn_module_index=module_to_index[id(val)],
                named_parameters=dict(val.named_parameters()),
                named_buffers=dict(val.named_buffers()),
            )
        return val

    return pytree.tree_map(module_to_state, values)


def _resolve_mutated_flat_indices(
    fn: Callable,
    mutates_args: frozenset[str],
    num_flat_args: int,
    input_spec: pytree.TreeSpec,
) -> str:
    """Resolve mutates_args expressions to a comma-separated string of flat-arg indices.

    Each expression in mutates_args (e.g. "x", "model.running_mean") is evaluated
    against sentinel values to determine which flat-arg positions are mutated.

    Example: for ``def fn(x, model)`` where model is an nn.Module with parameters
    ``weight`` and ``bias``, the flat args are ``[x, nn_module_index, weight, bias]``.
    Given ``mutates_args={"model.weight"}``, this assigns sentinels ``[0, 1, 2, 3]``
    to the flat args, evaluates ``model.weight`` to ``2``, and returns ``"2"``.
    """
    import inspect

    class _AttrDict:
        pass

    def _set_nested_attr(obj: _AttrDict, fqn: str, value: Any) -> None:
        parts = fqn.split(".")
        for part in parts[:-1]:
            if not hasattr(obj, part):
                setattr(obj, part, _AttrDict())
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _lms_to_attr_dict(val: Any) -> Any:
        if isinstance(val, LeafModuleState):
            target = _AttrDict()
            for fqn, sentinel in val.named_parameters.items():
                _set_nested_attr(target, fqn, sentinel)
            for fqn, sentinel in val.named_buffers.items():
                _set_nested_attr(target, fqn, sentinel)
            return target
        return val

    sig = inspect.signature(fn)
    sentinels = list(range(num_flat_args))
    args_struct, kwargs_struct = pytree.tree_unflatten(sentinels, input_spec)
    args_eval, kwargs_eval = pytree.tree_map(
        _lms_to_attr_dict,
        (args_struct, kwargs_struct),
        is_leaf=lambda x: isinstance(x, LeafModuleState),
    )
    namespace = dict(sig.bind(*args_eval, **kwargs_eval).arguments)

    indices: list[int] = []
    for expr in mutates_args:
        # Empty __builtins__ prevents access to builtins like __import__, open, exec.
        result = eval(expr, {"__builtins__": {}}, namespace)
        leaves = pytree.tree_leaves(result)
        for sentinel in leaves:
            if not isinstance(sentinel, int):
                raise ValueError(
                    f"mutates_args expression '{expr}' resolved to a non-leaf value "
                    f"of type {type(sentinel).__name__}. Expressions must resolve to "
                    f"individual tensor positions, e.g. 'model.weight' not 'model'."
                )
            indices.append(sentinel)
    indices.sort()
    return ",".join(str(i) for i in indices)


@dataclass
class GradientInfo:
    """
    We need the gradient edge to trigger the autograd engine backward.

    We need the tensor metadata (size, stride, dtype, device) to create zeros when
    gradient is None at runtime but required by the backward graph (because the graph
    is traced with fake implementation).
    """

    edge: torch.autograd.graph.GradientEdge
    size: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def _retrieve_module_by_index(nn_module_index: int) -> torch.nn.Module:
    # Check make_fx storage first (used by _invoke_leaf_function_python).
    # Fall back to the Dynamo retriever (used by the compiled path).
    if nn_module_index in _makefx_module_storage:
        if nn_module_index >= 0:
            raise RuntimeError(
                f"Expected negative nn_module_index for non-strict trace over leaf_function, but got {nn_module_index}."
            )
        return _makefx_module_storage[nn_module_index]

    if _leaf_function_module_retriever is None:
        raise RuntimeError("Leaf function module retriever not set.")

    mod = _leaf_function_module_retriever(nn_module_index)
    if not isinstance(mod, torch.nn.Module):
        raise TypeError(
            f"Expected nn.Module at index {nn_module_index} for leaf function invocation, "
            f"but got {type(mod).__name__}."
        )
    return mod


def check_escaped_gradients(
    outputs: Any,
    inputs: Sequence[Any],
    requires_grad_indices: set[int],
) -> None:
    """
    Check if autograd graph depends on tensors that are not passed as explicit inputs.

    Controlled by torch._dynamo.config.leaf_function_check_escaped_gradients.
    """
    if not requires_grad_indices:
        return

    import torch._dynamo.config as config

    if not config.leaf_function_check_escaped_gradients:
        return

    input_nodes: set[torch.autograd.graph.Node] = set()
    for i, inp in enumerate(inputs):
        if (
            isinstance(inp, torch.Tensor)
            and i in requires_grad_indices
            and inp.requires_grad
        ):
            edge = get_gradient_edge(inp)
            if edge.node is not None:
                input_nodes.add(edge.node)

    flat_outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    start_nodes: set[torch.autograd.graph.Node] = {
        out.grad_fn
        for out in flat_outputs
        if isinstance(out, torch.Tensor)
        and out.requires_grad
        and out.grad_fn is not None
    }
    if not start_nodes:
        return

    escaped: set[torch.autograd.graph.Node] = set()
    visited: set[torch.autograd.graph.Node] = set()
    stack = list(start_nodes)

    while stack:
        node = stack.pop()
        if node in visited or node in input_nodes:
            continue
        visited.add(node)

        for next_node, _ in node.next_functions:
            if next_node is None or next_node in input_nodes:
                continue
            if not next_node.next_functions:
                escaped.add(next_node)
            else:
                stack.append(next_node)

    if escaped:
        tensor_info = []
        for node in escaped:
            if hasattr(node, "variable"):
                t = node.variable
                tensor_info.append(
                    f"  - Tensor(shape={list(t.shape)}, dtype={t.dtype})"
                )
        tensor_details = (
            "\n".join(tensor_info) if tensor_info else "  (tensor details unavailable)"
        )

        raise RuntimeError(
            f"@leaf_function detected {len(escaped)} tensor(s) with requires_grad=True "
            f"that are not passed as explicit inputs:\n{tensor_details}\n"
            f"Gradients will not flow back to closure-captured or global tensors. "
            f"Pass them as explicit arguments to the leaf function."
        )


@contextlib.contextmanager
def unflatten_args_with_modules(
    flat_args: tuple[Any, ...], input_spec: pytree.TreeSpec
) -> Generator[tuple[list[Any] | tuple[Any, ...], dict[str, Any]], None, None]:
    args, kwargs = pytree.tree_unflatten(flat_args, input_spec)

    with contextlib.ExitStack() as stack:

        def process_module_state(state: LeafModuleState) -> torch.nn.Module:
            orig_module = _retrieve_module_by_index(state.nn_module_index)
            stack.enter_context(
                _reparametrize_module(
                    orig_module,
                    {**state.named_parameters, **state.named_buffers},
                )
            )
            return orig_module

        new_args, new_kwargs = pytree.tree_map_only(
            LeafModuleState,
            process_module_state,
            (args, kwargs),
            is_leaf=lambda x: isinstance(x, LeafModuleState),
        )
        yield new_args, new_kwargs


def flatten_args_with_modules(
    args_kwargs: tuple[Any, ...],
) -> list[Any]:
    def expand_module(x: Any) -> Any:
        if isinstance(x, torch.nn.Module):
            return LeafModuleState(
                nn_module_index=-1,
                named_parameters=dict(x.named_parameters()),
                named_buffers=dict(x.named_buffers()),
            )
        return x

    expanded = pytree.tree_map(expand_module, args_kwargs)
    return pytree.tree_leaves(expanded)


def make_leaf_function_wrappers(
    real_fn: Callable[..., Any],
    fake_fn: Callable[..., Any],
    captured_out_spec: list[pytree.TreeSpec | None],
) -> tuple[Callable[..., tuple[Any, ...]], Callable[..., tuple[Any, ...]]]:
    """Wrap real_fn and fake_fn to flatten outputs and capture the output TreeSpec.

    Both wrappers share the same captured output spec: the first call (typically
    fake_fn during tracing) records it, and subsequent calls verify consistency.
    The caller passes in a single-element list and reads captured_out_spec[0]
    after the wrappers have been called.

    Used by both the Dynamo path (_call_leaf_function in torch.py) and the
    make_fx path (_invoke_leaf_function_python in decorators.py).
    """

    def _wrap(fn: Callable[..., Any]) -> Callable[..., tuple[Any, ...]]:
        if len(captured_out_spec) != 1:
            raise RuntimeError(
                f"captured_out_spec must be a single-element list, got length {len(captured_out_spec)}"
            )

        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            out = fn(*args, **kwargs)

            flat_out, out_spec = pytree.tree_flatten(out)
            if captured_out_spec[0] is None:
                captured_out_spec[0] = out_spec
            elif captured_out_spec[0] != out_spec:
                raise AssertionError(
                    f"leaf_function output structure mismatch: "
                    f"expected {captured_out_spec[0]}, got {out_spec}. "
                    f"This can happen if the real function and fake function return "
                    f"different pytree structures (e.g., dict vs tuple, different number "
                    f"of elements). Ensure both functions return the same structure."
                )
            return tuple(flat_out)

        return wrapper

    return _wrap(real_fn), _wrap(fake_fn)


def autograd_grad_with_gradient_info(
    output_infos: Sequence[GradientInfo | None],
    input_infos: Sequence[GradientInfo | None],
    grad_outputs: Sequence[Any] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """
    Compute gradients using GradientInfo, it additionally handles the case
    where input and output infos are None.

    Args:
        output_infos: GradientInfo for each output (None if no grad_fn),
        input_infos: GradientInfo for each input (None if not requires_grad)
        grad_outputs: Gradients w.r.t. outputs

    Returns a tuple of gradients with None for inputs that don't require grad.
    """
    filtered_output_edges = []
    filtered_grad_outputs = []
    for i, info in enumerate(output_infos):
        if info is not None and info.edge.node is not None:
            filtered_output_edges.append(info.edge)
            if grad_outputs is not None:
                filtered_grad_outputs.append(grad_outputs[i])

    filtered_input_edges = []
    filtered_input_infos = []
    input_indices = []
    for i, info in enumerate(input_infos):
        if info is not None:
            filtered_input_edges.append(info.edge)
            filtered_input_infos.append(info)
            input_indices.append(i)

    if not filtered_output_edges or not filtered_input_edges:
        return tuple(None for _ in input_infos)

    grads = torch.autograd.grad(
        outputs=filtered_output_edges,
        inputs=filtered_input_edges,
        grad_outputs=filtered_grad_outputs if grad_outputs is not None else None,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )

    # Reconstruct full gradient tuple with Nones at proper positions.
    #
    # NB: For unused inputs that require grad, we return zeros instead of None.
    # This is necessary because during AOT tracing, we use fake_impl to determine
    # the backward graph structure. fake_impl may not accurately reflect which
    # inputs are truly used for gradients in real_impl. For example, users often
    # write fake_impl by returning torch.empty_like(input) to get the right shape
    # without actually using the input in a differentiable computation. So we must
    # be permissive and assume all inputs with requires_grad=True could need
    # gradients. When multiple invoke_leaf_function calls share an input (e.g.,
    # a module's forward + hook both receiving the same tensor), the traced
    # backward graph generates explicit add operations to accumulate their
    # gradients. At runtime, if one leaf function doesn't actually use the input,
    # autograd.grad returns None for it, and the traced add(grad1, grad2) fails
    # because one operand is None. Returning zeros ensures the add always works.
    result: list[torch.Tensor | None] = [None] * len(input_infos)
    for filtered_idx, original_idx in enumerate(input_indices):
        grad = grads[filtered_idx]
        if grad is None:
            info = filtered_input_infos[filtered_idx]
            grad = torch.empty_strided(
                info.size, info.stride, dtype=info.dtype, device=info.device
            ).zero_()
        result[original_idx] = grad

    return tuple(result)


def _make_forward(
    fn: Callable,
    include_keys: DispatchKeySet,
    exclude_keys: DispatchKeySet,
) -> tuple[Callable, dict[str, Any]]:
    state: dict[str, Any] = {"inputs": None, "outputs": None}

    @functools.wraps(fn)
    def forward(*args, **kwargs):
        effective_keys = include_keys
        if include_keys.has(DispatchKey.PythonDispatcher):
            effective_keys = effective_keys.remove(DispatchKey.PythonDispatcher)
        if effective_keys.has(DispatchKey.Python):
            effective_keys = effective_keys.remove(DispatchKey.Python)
        with torch._C._ForceDispatchKeyGuard(effective_keys, exclude_keys):
            with torch.enable_grad():
                outputs = fn(*args, **kwargs)

                flat_inputs = flatten_args_with_modules((args, kwargs))
                requires_grad_indices = {
                    i
                    for i, inp in enumerate(flat_inputs)
                    if isinstance(inp, torch.Tensor) and inp.requires_grad
                }
                check_escaped_gradients(outputs, flat_inputs, requires_grad_indices)

                state["inputs"] = tuple(
                    GradientInfo(
                        edge=get_gradient_edge(inp),
                        size=inp.size(),
                        stride=inp.stride(),
                        dtype=inp.dtype,
                        device=inp.device,
                    )
                    if isinstance(inp, torch.Tensor) and inp.requires_grad
                    else None
                    for inp in flat_inputs
                )

                if outputs is None:
                    state["outputs"] = ()
                else:
                    state["outputs"] = tuple(
                        GradientInfo(
                            edge=get_gradient_edge(out),
                            size=out.size(),
                            stride=out.stride(),
                            dtype=out.dtype,
                            device=out.device,
                        )
                        if isinstance(out, torch.Tensor)
                        and out.requires_grad
                        and out.grad_fn is not None
                        else None
                        for out in outputs
                    )

        return pytree.tree_map_only(
            torch.Tensor,
            lambda t: t.detach().requires_grad_(t.requires_grad),
            outputs,
        )

    return forward, state


class InvokeLeafFunction(HigherOrderOperator):
    def __init__(self):
        super().__init__("invoke_leaf_function", supports_training_input_mutation=True)

    def __call__(
        self,
        real_fn_callable,
        fake_fn_callable,
        input_spec,
        mutated_arg_indices,
        *flat_args,
        requires_grad_indices="",
    ):
        """
        real_fn_callable: _LeafCallable wrapping the real function
        fake_fn_callable: _LeafCallable wrapping the fake function
        input_spec: pytree.TreeSpec for unflattening flat_args back to (args, kwargs)
        mutated_arg_indices: comma-separated string of flat-arg indices that are
            declared as mutated (e.g. "1,2"), or "" for no mutations. Encoded as a
            string so it is a pytree leaf for the HOP schema infrastructure.
        requires_grad_indices: comma-separated string of flat-arg indices that
            require grad (e.g. "0,1"), or "" for none.
        """
        return super().__call__(  # type: ignore[attr-defined]
            real_fn_callable,
            fake_fn_callable,
            input_spec,
            mutated_arg_indices,
            *flat_args,
            requires_grad_indices=requires_grad_indices,
        )

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self,
        real_fn_callable,
        fake_fn_callable,
        input_spec,
        mutated_arg_indices,
        *flat_args,
        requires_grad_indices="",
    ):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import _maybe_fake_prop_ignore_unbacked
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

        mutated_set = _parse_mutated_arg_indices(mutated_arg_indices)

        with disable_proxy_modes_tracing():
            if mutated_set:
                schema_flat_args = tuple(
                    arg.detach().clone()
                    if isinstance(arg, torch.Tensor) and i in mutated_set
                    else arg
                    for i, arg in enumerate(flat_args)
                )
            else:
                schema_flat_args = flat_args

            def run_fake(*unfunc_flat_args):
                with unflatten_args_with_modules(unfunc_flat_args, input_spec) as (
                    args,
                    kwargs,
                ):
                    return fake_fn_callable(*args, **kwargs)

            fake_outputs = _maybe_fake_prop_ignore_unbacked(run_fake, schema_flat_args)

        gen = HopSchemaGenerator(self)
        gen.add_arg("real_fn_callable", real_fn_callable)
        gen.add_arg("fake_fn_callable", fake_fn_callable)
        gen.add_arg("input_spec", input_spec)
        gen.add_arg("mutated_arg_indices", mutated_arg_indices)
        for i, arg in enumerate(flat_args):
            gen.add_arg(f"arg{i}", arg, is_mutated=i in mutated_set)
        gen.add_arg(
            "requires_grad_indices",
            requires_grad_indices,
            default_value="",
            kw_only=True,
        )

        if isinstance(fake_outputs, tuple):
            for out in fake_outputs:
                gen.add_output(out)
        else:
            if fake_outputs is not None:
                raise AssertionError(
                    f"Expected fake_outputs to be a tuple or None, got {type(fake_outputs)}"
                )
            gen.add_output(fake_outputs)

        return gen.gen_schema()


invoke_leaf_function = InvokeLeafFunction()


# NOTE: [Autograd support for invoke_leaf_function]
#
# The overall idea is that when the real forward executes, we are going to build an autograd graph
# and save it.  When the real backward executes, we are going to invoke the autograd graph.
# We need to build these "real_forward" and "real_backward" functions from the "real_fn".
#
# Inputs:
# real_fn_callable/fake_fn_callable are _LeafCallable objects that wrap real_fn and fake_fn.
# These functions were created in dynamo by wrapping the user's original leaf function and fake function:
#   - They accept *flat_args (flattened LeafModuleState objects + other args)
#   - They unflatten flat_args and convert LeafModuleState back to nn.Modules
#   - They call the user's original leaf function with the reconstructed args
#   - They return the user function's outputs
#
# We wrap real_fn (via _make_forward) to handle autograd properly:
# 1. Detach inputs and outputs to isolate the leaf function's autograd graph from the
#    outer graph (gradients flow through our backward, not internal ops)
# 2. Real function is invoked at backend dispatch keys (i.e. CompositeExplicitAutograd)
#    where autograd is disabled. We re-enable grad and restore dispatch keys for the autograd engine to work.
# 3. Store GradientInfo (gradient edges + tensor metadata) instead of full tensors for backward,
#    which is sufficient for autograd.grad and avoids keeping tensors alive.
#
# We automatically generate backward functions:
# - real_backward uses the stored input/output GradientInfo and invokes the autograd engine
# - fake_backward creates empty gradients for inputs that requires grad


class InvokeLeafFunctionAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        real_fn_callable,
        fake_fn_callable,
        input_spec,
        mutated_arg_indices,
        *flat_args,
    ):
        include_keys = torch._C._dispatch_tls_local_include_set()
        exclude_keys = torch._C._dispatch_tls_local_exclude_set()

        requires_grad_indices = ",".join(
            str(i)
            for i, arg in enumerate(flat_args)
            if isinstance(arg, torch.Tensor) and arg.requires_grad
        )

        real_forward, real_state = _make_forward(
            real_fn_callable, include_keys, exclude_keys
        )

        def real_backward(*grads):
            if real_state["inputs"] is None or real_state["outputs"] is None:
                raise RuntimeError(
                    "invoke_leaf_function backward expects inputs/outputs to be set in forward."
                )
            return autograd_grad_with_gradient_info(
                output_infos=real_state["outputs"],
                input_infos=real_state["inputs"],
                grad_outputs=grads,
                allow_unused=True,
            )

        input_infos_for_fake = tuple(
            GradientInfo(
                edge=None,  # type: ignore[arg-type]
                size=arg.size(),
                stride=arg.stride(),
                dtype=arg.dtype,
                device=arg.device,
            )
            if isinstance(arg, torch.Tensor) and arg.requires_grad
            else None
            for arg in flat_args
        )

        def fake_backward(*grads):
            return tuple(
                torch.empty_strided(
                    info.size, info.stride, dtype=info.dtype, device=info.device
                )
                if info is not None
                else None
                for info in input_infos_for_fake
            )

        new_real_fn_callable = _LeafCallable(real_forward)

        with torch._C._AutoDispatchBelowAutograd():
            fw_outputs = invoke_leaf_function(
                new_real_fn_callable,
                fake_fn_callable,
                input_spec,
                mutated_arg_indices,
                *flat_args,
                requires_grad_indices=requires_grad_indices,
            )

        hook_real = getattr(real_fn_callable, "_leaf_hook_real_fn", None)
        hook_fake = getattr(real_fn_callable, "_leaf_hook_fake_fn", None)
        if hook_real is not None:
            if hook_fake is None:
                raise AssertionError("expected hook_fake to be not None")
            hook_captured_out_spec: list[pytree.TreeSpec | None] = [None]
            wrapped_hook_real, wrapped_hook_fake = make_leaf_function_wrappers(
                hook_real, hook_fake, hook_captured_out_spec
            )
            hook_real_callable = _LeafCallable(wrapped_hook_real)
            hook_fake_callable = _LeafCallable(wrapped_hook_fake)

            grad_tensors = [
                arg
                for arg in flat_args
                if isinstance(arg, torch.Tensor) and arg.requires_grad
            ]
            if grad_tensors:

                @torch._dynamo.disable
                def _multi_grad_callback(
                    grads: Sequence[torch.Tensor],
                ) -> None:
                    _, hook_spec = pytree.tree_flatten((tuple(grads), {}))
                    invoke_leaf_function(
                        hook_real_callable,
                        hook_fake_callable,
                        hook_spec,
                        "",
                        *grads,
                    )

                torch.autograd.graph.register_multi_grad_hook(
                    grad_tensors, _multi_grad_callback
                )

        ctx.real_backward = real_backward
        ctx.fake_backward = fake_backward

        return fw_outputs

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, *grads):
        real_bw_callable = _LeafCallable(ctx.real_backward)
        fake_bw_callable = _LeafCallable(ctx.fake_backward)
        _, bw_input_spec = pytree.tree_flatten((grads, {}))
        fw_grads = invoke_leaf_function(
            real_bw_callable, fake_bw_callable, bw_input_spec, "", *grads
        )
        return None, None, None, None, *fw_grads


@invoke_leaf_function.py_autograd_impl
def invoke_leaf_function_autograd(
    real_fn_callable,
    fake_fn_callable,
    input_spec,
    mutated_arg_indices,
    *flat_args,
    requires_grad_indices="",
):
    return InvokeLeafFunctionAutogradOp.apply(
        real_fn_callable, fake_fn_callable, input_spec, mutated_arg_indices, *flat_args
    )


# TODO: aliasing is not allowed
@invoke_leaf_function.py_functionalize_impl
def invoke_leaf_function_functionalization(ctx, *all_args, **kwargs):
    from torch._higher_order_ops.auto_functionalize import (
        can_auto_functionalize,
        do_auto_functionalize_v2,
    )
    from torch._higher_order_ops.utils import HopInstance

    unwrapped_args = ctx.unwrap_tensors(all_args)
    hop_instance = HopInstance.create(invoke_leaf_function, *unwrapped_args, **kwargs)
    if can_auto_functionalize(hop_instance):
        return do_auto_functionalize_v2(ctx.mode, hop_instance, all_args, kwargs)

    from torch._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery,
        ctx.mode._tokens,
        invoke_leaf_function,
        all_args,
        kwargs,
    )


@invoke_leaf_function.py_impl(ProxyTorchDispatchMode)
def invoke_leaf_function_proxy_mode(proxy_mode, *all_args, **kwargs):
    out = invoke_leaf_function(*all_args, **kwargs)
    proxies = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, all_args)
    proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_leaf_function, proxies, kwargs
    )
    return track_tensor_tree(out, proxy, constant=None, tracer=proxy_mode.tracer)


def _validate_outputs_match(
    fake_output: Any,
    real_output: Any,
) -> None:
    """
    Validate that fake_fn and real_fn outputs have matching pytree structure,
    shapes, and dtypes.

    Raises:
        RuntimeError: If outputs don't match with detailed error message.
    """
    fake_flat, fake_spec = pytree.tree_flatten(fake_output)
    real_flat, real_spec = pytree.tree_flatten(real_output)

    if fake_spec != real_spec:
        raise RuntimeError(
            f"Output structure mismatch in @leaf_function decorator.\n"
            f"fake_impl returned structure: {fake_spec}\n"
            f"real_impl returned structure: {real_spec}\n"
            f"The fake_impl must return outputs with the same pytree structure as real_impl."
        )

    if len(fake_flat) != len(real_flat):
        raise RuntimeError(
            f"Output count mismatch in @leaf_function decorator.\n"
            f"fake_impl returned {len(fake_flat)} values\n"
            f"real_impl returned {len(real_flat)} values"
        )

    for i, (fake_val, real_val) in enumerate(zip(fake_flat, real_flat)):
        fake_is_tensor = isinstance(fake_val, torch.Tensor)
        real_is_tensor = isinstance(real_val, torch.Tensor)

        if fake_is_tensor != real_is_tensor:
            raise RuntimeError(
                f"Output type mismatch at position {i} in @leaf_function decorator.\n"
                f"fake_impl returned: {type(fake_val).__name__}\n"
                f"real_impl returned: {type(real_val).__name__}"
            )

        if fake_is_tensor:
            if fake_val.shape != real_val.shape:
                raise RuntimeError(
                    f"Shape mismatch at output position {i} in @leaf_function decorator.\n"
                    f"fake_impl output shape: {list(fake_val.shape)}\n"
                    f"real_impl output shape: {list(real_val.shape)}\n"
                    f"The fake_impl must produce tensors with the same shapes as real_impl."
                )

            if fake_val.dtype != real_val.dtype:
                raise RuntimeError(
                    f"Dtype mismatch at output position {i} in @leaf_function decorator.\n"
                    f"fake_impl output dtype: {fake_val.dtype}\n"
                    f"real_impl output dtype: {real_val.dtype}\n"
                    f"The fake_impl must produce tensors with the same dtypes as real_impl."
                )


def _parse_mutated_arg_indices(s: str) -> set[int]:
    return {int(x) for x in s.split(",") if x}


def _check_no_input_mutation(
    flat_args: tuple[Any, ...],
    version_before: list[int],
    mutated_arg_indices: str = "",
) -> None:
    mutated_set = _parse_mutated_arg_indices(mutated_arg_indices)
    for i, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor) and arg._version != version_before[i]:
            if i not in mutated_set:
                raise RuntimeError(
                    f"Undeclared in-place mutation on input tensor at position {i}. "
                    f"Declare it in @leaf_function(mutates_args=...) or avoid mutating inputs."
                )


@register_fake(invoke_leaf_function)
def invoke_leaf_function_fake(
    real_fn_callable,
    fake_fn_callable,
    input_spec,
    mutated_arg_indices,
    *flat_args,
    requires_grad_indices="",
):
    with unflatten_args_with_modules(flat_args, input_spec) as (args, kwargs):
        return fake_fn_callable(*args, **kwargs)


def _detach_with_grad_dtype(arg):
    # Tensor.detach() resets grad_dtype to the tensor's own dtype, dropping
    # any caller-set grad_dtype (e.g. param.grad_dtype = torch.float32 on a
    # bf16 weight). Propagate it back so the inner autograd graph accumulates
    # the kernel's gradient at the requested dtype. grad_dtype is only
    # meaningful on leaf tensors.
    if not isinstance(arg, torch.Tensor):
        return arg
    detached = arg.detach()
    if arg is not detached and arg.is_leaf and arg.grad_dtype != detached.grad_dtype:
        detached.grad_dtype = arg.grad_dtype
    return detached


@invoke_leaf_function.py_impl(DispatchKey.CompositeExplicitAutograd)
def invoke_leaf_function_dense(
    real_fn_callable,
    fake_fn_callable,
    input_spec,
    mutated_arg_indices,
    *flat_args,
    requires_grad_indices="",
):
    from torch._dynamo import config as dynamo_config

    version_before = [
        arg._version if isinstance(arg, torch.Tensor) else 0 for arg in flat_args
    ]

    flat_args = tuple(_detach_with_grad_dtype(arg) for arg in flat_args)
    requires_grad_indices_set = _parse_mutated_arg_indices(requires_grad_indices)
    flat_args = tuple(
        arg.requires_grad_(True) if idx in requires_grad_indices_set else arg
        for idx, arg in enumerate(flat_args)
    )

    with unflatten_args_with_modules(flat_args, input_spec) as (args, kwargs):
        real_output = real_fn_callable(*args, **kwargs)

        _check_no_input_mutation(flat_args, version_before, mutated_arg_indices)

        if dynamo_config.leaf_function_validate_outputs:
            fake_output = fake_fn_callable(*args, **kwargs)
            _validate_outputs_match(fake_output, real_output)

    return real_output
<|endoftext|># mypy: allow-untyped-defs

import contextlib
import copy
import functools
import threading
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _from_fun,
    _maybe_reenter_make_fx,
    clone_outputs_aliasing_inputs,
    FunctionalizeCtxWrapper,
    get_dummy_aot_autograd_config,
    HopInstance,
    prepare_fw_with_masks,
    redirect_to_mode,
    reenter_make_fx,
    register_fake,
    save_values_for_backward,
    saved_values,
)
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class
from torch._ops import HigherOrderOperator
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.utils._debug_mode import DebugMode
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


invoke_subgraph_counter = 0


# During the tracing of the joint graph, we construct this information. This is
# used to filter out grad_outs/tangents in the `backward` method of
# InvokeSubgraphAutogradOp.
@dataclass
class OutputMetadata:
    num_fw_outs: int | None = None
    indexes_with_symint: set[int] = field(default_factory=set)
    indexes_with_no_grad: set[int] = field(default_factory=set)


# This config will be stored in invoke_subgraph HOP node.meta["custom"]["nested_region_config"]
# as well as the subgraph's gm.meta["nested_region_config"].
@dataclass
class NestedCompileRegionOptions:
    # A Callable that takes (gm, example_inputs, decompositions=None, **kwargs) as inputs.
    # Returns AOTCompiledArtifact
    fw_compiler: Callable | None = None
    bw_compiler: Callable | None = None

    # Note: [InvokeSubgraphHOP Partitioner]
    # If not None, add "partitioner" to HOP node meta.
    # If Callable, directly assign the callable, but the callable cannot be pickled
    # If str, the options are "default_partition" and "min_cut_rematerialization_partition".
    # The HOP joint graph will be partitioned using the corresponding functions in
    # torch/_functorch/partitioners.py
    partitioner: Callable | str | None = None

    # If it's None, we'll inherit the parent call's decompositions.
    # Otherwise, the nested region will use this decompositions.
    decompositions: dict[str, Any] | None = None

    # Inductor config patches to apply while compiling this nested region through
    # Inductor's normal invoke_subgraph lowering path. Also used for the backward
    # unless bw_inductor_config_patches replaces it.
    inductor_config_patches: dict[str, Any] | None = None

    # If set, the full inductor config for the backward subgraph, used instead of
    # inductor_config_patches (a replacement, not merged with it), mirroring
    # aot_autograd's separate fw_compiler/bw_compiler. If None, the backward
    # reuses the forward config.
    bw_inductor_config_patches: dict[str, Any] | None = None


def _extract_nested_region_config(fn):
    """
    Extract the NestedCompileRegionOptions from the HOP subgraph gm.meta["nested_region_config"]
    """
    gm_to_compile = None
    if isinstance(fn, torch.fx.GraphModule):
        gm_to_compile = fn
    elif isinstance(fn, FunctionalizeCtxWrapper):
        gm_to_compile = fn.subgraph

    if (
        isinstance(gm_to_compile, torch.fx.GraphModule)
        and hasattr(gm_to_compile, "meta")
        and "nested_region_config" in gm_to_compile.meta
    ):
        if isinstance(
            gm_to_compile.meta["nested_region_config"], NestedCompileRegionOptions
        ):
            return gm_to_compile.meta["nested_region_config"].decompositions
    return None


def get_backward_nested_region_config(
    fw_config: NestedCompileRegionOptions | None,
) -> NestedCompileRegionOptions | None:
    """Region config for compiling the backward subgraph.

    When the region sets bw_inductor_config_patches, the backward compiles under
    it (a replacement for inductor_config_patches, not merged with it). Otherwise
    the forward config is reused unchanged, so the returned object is identical
    (callers rely on this identity to detect a distinct backward config).
    """
    if (
        isinstance(fw_config, NestedCompileRegionOptions)
        and fw_config.bw_inductor_config_patches is not None
    ):
        return replace(
            fw_config,
            inductor_config_patches=fw_config.bw_inductor_config_patches,
            bw_inductor_config_patches=None,
        )
    return fw_config


# Per-call id used by downstream graph passes to pair fw and bw
# invoke_subgraph HOP nodes (e.g. run_joint_graph_passes_on_hops). The
# autograd Function pushes the id around its fw and bw dispatches; the
# proxy-mode handler stamps it on the resulting FX nodes via
# meta["custom"]["call_id"].
_invoke_subgraph_call_state = threading.local()


def _next_invoke_subgraph_call_id() -> int:
    counter = getattr(_invoke_subgraph_call_state, "counter", 0)
    _invoke_subgraph_call_state.counter = counter + 1
    return counter


def _current_invoke_subgraph_call_id() -> int | None:
    return getattr(_invoke_subgraph_call_state, "current", None)


@contextlib.contextmanager
def _set_invoke_subgraph_call_id(call_id: int):
    prev = getattr(_invoke_subgraph_call_state, "current", None)
    _invoke_subgraph_call_state.current = call_id
    try:
        yield
    finally:
        _invoke_subgraph_call_state.current = prev


def warn_and_trace_duplicate_backward(
    identifier: str | None,
    suffix: int,
    bw_graph: torch.fx.GraphModule,
    num_primals: int,
    tangents: tuple[Any, ...],
) -> None:
    if suffix == 0:
        return

    from torch._dynamo.utils import warn_once
    from torch._logging import trace_structured

    placeholders = [node for node in bw_graph.graph.nodes if node.op == "placeholder"]
    tangent_summaries: list[dict[str, Any]] = []
    for idx, tangent in enumerate(tangents):
        summary: dict[str, Any] = {}
        backward_input_idx = num_primals + idx
        if backward_input_idx < len(placeholders):
            summary["backward_input_name"] = placeholders[backward_input_idx].name

        if not isinstance(tangent, torch.Tensor):
            summary["type"] = "non-Tensor"
            tangent_summaries.append(summary)
            continue

        summary.update(
            {
                "shape": [
                    int(s) if isinstance(s, int) else str(s) for s in tangent.shape
                ],
                "stride": [
                    int(s) if isinstance(s, int) else str(s) for s in tangent.stride()
                ],
                "dtype": str(tangent.dtype),
                "device": str(tangent.device),
            }
        )
        tangent_summaries.append(summary)

    warn_once(
        "invoke_subgraph traced multiple backward graphs for the same forward "
        f"identifier {identifier!r}. This usually means the incoming backward "
        "tangent metadata differs across backward invocations. This can happen "
        "when upstream backward formulas produce gradients with different "
        "layouts, aliases, or other tensor metadata. This can increase compile "
        "time and code size. Use tlparse and look for the "
        "invoke_subgraph_backward_duplicate artifact for tangent details."
    )
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "invoke_subgraph_backward_duplicate",
            "encoding": "json",
        },
        payload_fn=lambda: {
            "forward_identifier": identifier,
            "backward_identifier": f"bw_{identifier}_{suffix}",
            "new_variant_suffix": suffix,
            "num_variants_for_identifier": suffix + 1,
            "new_variant": {
                "backward_identifier": f"bw_{identifier}_{suffix}",
                "tangents": tangent_summaries,
            },
            "explanation": (
                "invoke_subgraph caches backward graphs by both the forward "
                "identifier and tangent metadata. Multiple backward variants "
                "for one forward identifier usually mean the backward tangents "
                "have different tensor metadata, commonly stride."
            ),
            "common_cause": (
                "This can happen when equivalent forward invocations receive backward "
                "tangents from different upstream formulas, or from the same "
                "formula with different input metadata. For example, an "
                "upstream backward may return a view, alias, slice, expanded "
                "tensor, or otherwise non-contiguous gradient whose metadata "
                "does not match the tangent seen by another use."
            ),
            "suggested_action": (
                "If the duplicate backward graph is undesirable, materialize a "
                "consistent tangent layout before the nested region backward. "
                "One model-level workaround is a custom autograd.Function whose "
                "forward returns the input and whose backward returns "
                "grad.contiguous()."
            ),
        },
    )


class InvokeSubgraphHOP(HigherOrderOperator):
    def __init__(self) -> None:
        # Invoke subgraph does not have any state, it is just a wrapper over a
        # subgraph, so we can safely cache the HOP.
        super().__init__("invoke_subgraph", cacheable=True)
        # This is used by the fake tensor cache key validator to extract the
        # subgraph and iterate over the nodes to find if all nodes are fake
        # tensor cacheable.
        self.subgraph_indexes = [
            0,
        ]

    # identifier is setup by upper part of the stack. This helps us in
    # identifying two invoke_subgraph calls have same subgraph.
    def __call__(
        self,
        subgraph: GraphModule | FunctionalizeCtxWrapper,
        identifier: str | None,
        *operands,
    ):
        if identifier is not None and not isinstance(identifier, str):
            raise AssertionError(
                f"identifier must be None or a string, got {type(identifier)}"
            )

        if not all(
            isinstance(
                o, (torch.Tensor, int, torch.SymInt, torch.Generator, FakeScriptObject)
            )
            or is_custom_class(type(o))
            for o in operands
            if o is not None
        ):
            raise AssertionError(
                f"invoke_subgraph operands must be a list of tensors/ints/SymInts/Generator, got {operands}"
            )

        # pyrefly: ignore [missing-attribute]
        return super().__call__(subgraph, identifier, *operands)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, subgraph, identifier, *operands):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import (
            check_input_alias_and_mutation_return_outputs,
            materialize_as_graph,
        )
        from torch._subclasses.functional_tensor import FunctionalTensor

        subgraph_decomp_table = _extract_nested_region_config(subgraph)
        # When the subgraph is a GraphModule whose captured buffers are
        # already FunctionalTensors (produced by an enclosing
        # FunctionalTensorMode, e.g. the non-strict export path or
        # ``run_decompositions`` on an ExportedProgram that preserved an
        # ``invoke_subgraph`` HOP), re-tracing through
        # ``materialize_as_graph`` would pop that mode via
        # ``_disable_current_modes`` and trigger "FunctionalTensor on its
        # own" when the inner trace touches the captured constants. In that
        # case the existing graph has already been functionalized by the
        # outer pass, so reuse it for schema / mutation analysis.
        gm: torch.fx.GraphModule | None = None
        if isinstance(subgraph, FunctionalizeCtxWrapper):
            candidate = subgraph.subgraph
        elif isinstance(subgraph, torch.fx.GraphModule):
            candidate = subgraph
        else:
            candidate = None
        bufs = (
            list(candidate.buffers())
            if isinstance(candidate, torch.fx.GraphModule)
            else []
        )
        if bufs and all(isinstance(buf, FunctionalTensor) for buf in bufs):
            gm = candidate
        if gm is None:
            gm = materialize_as_graph(
                subgraph, operands, subgraph_decomp_table=subgraph_decomp_table
            )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("subgraph", gm)
        schema_gen.add_arg("identifier", identifier)
        (
            _,
            _,
            _,
            mutated_inputs,
            outputs,
        ) = check_input_alias_and_mutation_return_outputs(gm)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"arg{idx}", arg, is_mutated=idx in mutated_inputs)
        for out in outputs:
            schema_gen.add_output(out)

        return schema_gen.gen_schema()


invoke_subgraph = InvokeSubgraphHOP()


def invoke_subgraph_infer(
    subgraph: GraphModule | FunctionalizeCtxWrapper,
    *operands,
):
    """Inference-only entrypoint for invoke_subgraph that auto-generates identifier.

    This is intended for use cases where we are building an inference graph and
    don't need the forward/backward caching that requires a stable identifier.
    The identifier is automatically computed based on the current proxy mode's
    tracer state.

    If no proxy mode is active, the subgraph is called directly.
    """
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    proxy_mode = get_proxy_mode()
    if proxy_mode is None:
        # No tracing active, just call the subgraph directly
        if getattr(subgraph, "_boxed_call", False):
            return subgraph(list(operands))
        else:
            return subgraph(*operands)

    from torch._dynamo.utils import get_unique_name_wrt

    # How exactly should we allocate names for the HOP invoke_subgraph we
    # are going to put into the graph?  This is a bit tricky.  In the
    # original design of invoke_subgraph, this HOP never shows up in the
    # wild: it is only generated Dynamo, so Dynamo can take sure of
    # ensuring it picks unique names in the context of the particular
    # Dynamo compilation.  However, these invoke_subgraph are different:
    # they live as Dynamo compiled code that can potentially get traced
    # multiple times!  If they get retraced several times in the same
    # trace, deduplication occurs; but if I make_fx a function f once,
    # and then do a separate new trace, there's no relationship between
    # these.  Additionally, we also want the name we put in the graph to
    # be deterministic, and for it to be indifferent to how many
    # unrelated invoke_subgraphs/make_fxs we've done, prior to THIS
    # particular make_fx.
    #
    # To satisfy all of these constraints, it's impossible to preallocate
    # a name before tracing actually goes through us (since those names
    # would have to all be unique even if a subgraph never gets used.)
    # So we allocate the subgraph a fresh name PER proxy mode, and then
    # consistently reuse it if it hits again.
    #
    # Note we do NOT do equality comparison subgraph, since it has
    # reference equality semantics.

    if subgraph in proxy_mode._invoke_subgraph_cache:
        name = proxy_mode._invoke_subgraph_cache[subgraph]
    else:
        name = get_unique_name_wrt(
            "invoke_subgraph",
            proxy_mode._invoke_subgraph_names,
            requires_suffix=True,
        )
        proxy_mode._invoke_subgraph_names.add(name)
        proxy_mode._invoke_subgraph_cache[subgraph] = name

    return invoke_subgraph(subgraph, name, *operands)


# Registers dispatches for SAC
redirect_to_mode(invoke_subgraph, _CachingTorchDispatchMode)
redirect_to_mode(invoke_subgraph, _CachedTorchDispatchMode)


def invoke_subgraph_placeholder(func, *args, **kwargs):
    if torch.compiler.is_dynamo_compiling():
        # This is just a placeholder for Dynamo to replace with invoke_subgraph
        raise RuntimeError("invoke_subgraph should not be called directly in Dynamo")

    if torch.compiler.is_exporting():
        # For non-strict export tracing, we still want to go through Dynamo

        def _invoke_subgraph_placeholder_wrapper(func, args, kwargs):
            return invoke_subgraph_placeholder(func, *args, **kwargs)

        from torch._higher_order_ops.utils import _hop_compile_and_call

        return _hop_compile_and_call(
            _invoke_subgraph_placeholder_wrapper, (func, args, kwargs)
        )

    return func(*args, **kwargs)


def mark_compile_region(
    fn=None,
    options: NestedCompileRegionOptions | None = None,
    max_reuse_entries: int = 8,
    reuse_hash_fn=None,
):
    """
    This wrapper instructs torch.compile to compile the wrapped region once and
    reuse the compiled artifact, instead of the usual way of aggressively
    inlining the function.

    Under the hood, it tells TorchDynamo to use InvokeSubgraph HOP for the
    region. For PyTorch eager, this is a no-op.

    Args:
        fn: The function to wrap
        options: Optional config to use for compiling the subgraph.
            Warning: this is an experimental feature under development and
            not ready for use yet.
        max_reuse_entries: Maximum number of reuse cache entries per function
            before raising an error. If this limit is hit, guards keep failing
            across invocations and hierarchical compilation is not effective.
    """

    def wrap(func):
        def inner(*args, **kwargs):
            # Get the innermost function to avoid nested compile regions
            inner_func = func
            while hasattr(inner_func, "__marked_compile_region_fn__"):
                inner_func = inner_func.__marked_compile_region_fn__
            return invoke_subgraph_placeholder(inner_func, *args, **kwargs)

        inner.__marked_compile_region_fn__ = func  # type: ignore[attr-defined]
        func.__marked_compile_region_config__ = options  # type: ignore[attr-defined]
        func.__marked_compile_region_max_reuse_entries__ = max_reuse_entries  # type: ignore[attr-defined]
        func.__marked_compile_region_reuse_hash_fn__ = reuse_hash_fn  # type: ignore[attr-defined]

        return inner

    if fn:
        return wrap(fn)
    else:
        return wrap


def get_invoke_subgraph_cache():
    cache = None
    if tracing_ctx := torch._guards.TracingContext.try_get():
        cache = tracing_ctx.hop_dispatch_set_cache.get_cache(invoke_subgraph)
    return cache


# TODO (@anijain2305) - Delete this function when base_hop uses invoke_subgraph infra
def trace_joint_graph(fn, fw_inputs, fw_outputs):
    """
    Naively trace out a joint graph. This simplifies the reconstruction of joint
    graph in the min-cut partitioner later on.
    """
    from torch._functorch.aot_autograd import create_joint

    dummy_aot_config = get_dummy_aot_autograd_config()

    # This joint_fn is inserted as the backward graph as is. This simplifies the
    # min-cut partitioner work later on.
    #   Input signature - (*primals, *tangents)
    #   Output signature - (*grads, *fw_outs)
    # The output signature is deliberately kept grads first and fw_outs second.
    # Having grads first makes the min-cut partitioner HOP graph stitching
    # easier.
    def joint_fn(*primals_and_tangents):
        primals = primals_and_tangents[: len(fw_inputs)]
        tangents = primals_and_tangents[len(fw_inputs) :]

        fw_outs, grads = create_joint(
            prepare_fw_with_masks(fn), aot_config=dummy_aot_config
        )(primals, tangents)

        maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)

        # return signature is deliberately kept (*grads, *fw_outs). This
        # simplifies partitioning work later on.
        return pytree.tree_map(maybe_clone, tuple(grads + list(fw_outs)))

    primals = list(fw_inputs)
    # This assumes that the tangent strides match fw_outputs strides. Check the
    # InvokeSubgraphAutogradOp backward op for the contiguous call.
    tangents = [_from_fun(out) for out in fw_outputs]

    joint_operands = primals + tangents

    return _maybe_reenter_make_fx(joint_fn)(*joint_operands)


# TODO (@anijain2305) - Delete this function when base_hop uses invoke_subgraph infra
def create_fw_bw_graph(subgraph, operands, grad_outputs=None):
    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # args are functional tensors, generate some example tensors
            fw_inputs = pytree.tree_map(_from_fun, operands)

            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(fw_inputs)
            context = (
                nullcontext()
                if fake_mode is None or fake_mode.shape_env is None
                else fake_mode.shape_env.ignore_fresh_unbacked_symbols()
            )

            with context:
                fw_outs = pytree.tree_map(_from_fun, subgraph(*fw_inputs))

            num_fw_outs = len(fw_outs)

            # Collect the indexes of none in the output to check that the grad
            # is None at the corresponding index in the backward. This check is
            # performed in the autograd.Function - InvokeSubgraphAutogradOp.
            # Also collect the indexes of no_grad in the output to filter out
            # the grad_outs in the `backward` method.
            output_metadata = OutputMetadata()

            output_metadata.num_fw_outs = num_fw_outs
            for idx, fw_out in enumerate(fw_outs):
                if isinstance(fw_out, torch.SymInt):
                    output_metadata.indexes_with_symint.add(idx)
                elif not fw_out.requires_grad:
                    output_metadata.indexes_with_no_grad.add(idx)

            if grad_outputs is None:
                # Infer grad_outputs to be the same properties as the fw_outputs
                # if they're not passed in
                # Although fw_outs are equivalent to grad_outputs for tracing
                # purposes, we have to carefully handle the None and fw_out that do
                # not have require_grad. At those indexes, we will have None in the
                # backward graph.
                grad_outputs = fw_outs
                grad_outputs = [grad for grad in grad_outputs if grad is not None]
                grad_outputs = [grad for grad in grad_outputs if grad.requires_grad]

                # Force grad_out to be contiguous. This is because at runtime,
                # grad_out could have different strides than fw_outs. So, we
                # force the grad_outs to be contiguous for both tracing and
                # runtime.
                grad_outputs = [grad.contiguous() for grad in grad_outputs]

            if any(
                not isinstance(out, torch.Tensor)
                for out in grad_outputs
                if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of invoke_subgraph to only contains tensors or None. "
                    f"Got types {[type(out) for out in grad_outputs]}."
                )

            # Trace the forward subgraph
            fw_graph = _maybe_reenter_make_fx(subgraph)(*fw_inputs)

            # Trace the joint graph and assign it to the bwd graph
            bw_graph = trace_joint_graph(
                subgraph,
                fw_inputs,
                grad_outputs,
            )
            return fw_graph, bw_graph, output_metadata


def get_output_metadata(subgraph, *operands):
    """
    Extract metadata about the subgraph outputs WITHOUT executing the subgraph.
    This avoids running side-effectful operations twice (once here, once in forward).
    We analyze the graph structure statically to extract metadata.
    """
    # Unwrap FunctionalizeCtxWrapper if present
    if isinstance(subgraph, FunctionalizeCtxWrapper):
        subgraph = subgraph.subgraph

    # If not a GraphModule, fall back to execution-based metadata extraction
    if not isinstance(subgraph, torch.fx.GraphModule):
        return _get_output_metadata_by_execution(subgraph, *operands)

    output_metadata = OutputMetadata()

    # Extract output arguments from the output node
    # The output node has args=(output_values,) where output_values is a tuple/list
    output_node = next(reversed(subgraph.graph.find_nodes(op="output")))
    output_metadata.num_fw_outs = len(output_node.args[0])

    for idx, output_arg in enumerate(output_node.args[0]):
        if not isinstance(output_arg, torch.fx.Node):
            if isinstance(output_arg, int):
                output_metadata.indexes_with_symint.add(idx)
            output_metadata.indexes_with_no_grad.add(idx)
            continue

        # Check node metadata for type information
        if output_arg.meta.get("val") is None:
            # If we don't have complete metadata for all outputs, fall back to execution
            # This is important for correctness (e.g., detecting SymInts) even though it
            # runs side-effectful operations
            return _get_output_metadata_by_execution(subgraph, *operands)

        val = output_arg.meta["val"]
        if isinstance(val, torch.SymInt):
            output_metadata.indexes_with_symint.add(idx)
            output_metadata.indexes_with_no_grad.add(idx)
        elif isinstance(val, torch.Tensor):
            # Check if tensor requires grad from metadata
            if hasattr(val, "requires_grad") and not val.requires_grad:
                output_metadata.indexes_with_no_grad.add(idx)
        else:
            # Non-tensor, non-symint (shouldn't happen but be safe)
            output_metadata.indexes_with_no_grad.add(idx)

    return output_metadata


def _get_output_metadata_by_execution(subgraph, *operands):
    """
    Fallback: Extract metadata by executing the subgraph.
    This should only be used when static analysis fails.
    WARNING: This will run side-effectful operations!
    """

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # args are functional tensors, generate some example tensors
            fw_inputs = pytree.tree_map(_from_fun, operands)

            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(fw_inputs)
            context = (
                nullcontext()
                if fake_mode is None or fake_mode.shape_env is None
                else fake_mode.shape_env.ignore_fresh_unbacked_symbols()
            )

            with context:
                fw_outs = pytree.tree_map(_from_fun, subgraph(*fw_inputs))

            num_fw_outs = len(fw_outs)

            output_metadata = OutputMetadata()
            output_metadata.num_fw_outs = num_fw_outs

            for idx, fw_out in enumerate(fw_outs):
                if isinstance(fw_out, torch.SymInt):
                    output_metadata.indexes_with_symint.add(idx)
                elif not fw_out.requires_grad:
                    output_metadata.indexes_with_no_grad.add(idx)

            return output_metadata


def trace_joint_graph_as_bwd(
    subgraph, num_primals, joint_operands, include_key_set, exclude_key_set
):
    """
    Naively trace out a joint graph. This simplifies the reconstruction of joint
    graph in the min-cut partitioner later on.
    """
    from torch._functorch.aot_autograd import create_joint

    dummy_aot_config = get_dummy_aot_autograd_config()

    if isinstance(subgraph, torch.fx.GraphModule):

        def graph_with_interpreter(*args):
            # Running graph with interpreter is needed for propagating the stack_trace
            with torch.fx.traceback.preserve_node_meta():
                return torch.fx.Interpreter(subgraph).run(*args)

        fn = graph_with_interpreter
    else:
        fn = subgraph

    # This joint_fn is inserted as the backward graph as is. This simplifies the
    # min-cut partitioner work later on.
    #   Input signature - (*primals, *tangents)
    #   Output signature - (*grads, *fw_outs)
    # The output signature is deliberately kept grads first and fw_outs second.
    # Having grads first makes the min-cut partitioner HOP graph stitching
    # easier.
    def joint_fn(*primals_and_tangents):
        primals = primals_and_tangents[:num_primals]
        tangents = primals_and_tangents[num_primals:]

        fw_outs, grads = create_joint(
            prepare_fw_with_masks(fn), aot_config=dummy_aot_config
        )(primals, tangents)

        maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)

        # return signature is deliberately kept (*grads, *fw_outs). This
        # simplifies partitioning work later on.
        return pytree.tree_map(maybe_clone, tuple(grads + list(fw_outs)))

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            joint_operands = [_from_fun(arg) for arg in joint_operands]
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    torch._C._ForceDispatchKeyGuard(include_key_set, exclude_key_set),
                )
                subgraph_decomp_table = _extract_nested_region_config(subgraph)
                with torch.enable_grad():
                    return _maybe_reenter_make_fx(
                        joint_fn, subgraph_decomp_table=subgraph_decomp_table
                    )(*joint_operands)


def get_non_differentiable_indices(subgraph) -> list[int]:
    """Inspect a subgraph's output node metadata to find outputs that
    were non-differentiable (requires_grad=False) during Dynamo tracing.

    This uses the example_value stored on each output node, which reflects
    the correct requires_grad from the original tracing (where inner
    autograd.Functions DID run and mark_non_differentiable was called).

    InvokeSubgraphAutogradOp runs the subgraph under
    _AutoDispatchBelowAutograd, which suppresses the autograd key.  Inner
    autograd.Functions (e.g. FlexAttentionAutogradOp) therefore never call
    mark_non_differentiable.  The outer autograd engine then sets
    requires_grad=True on ALL outputs because the inputs require grad.
    We use the Dynamo-traced example_value to restore the correct
    non-differentiable status at the outer level.
    """
    if not isinstance(subgraph, torch.fx.GraphModule):
        return []

    output_nodes = [n for n in subgraph.graph.nodes if n.op == "output"]
    if not output_nodes:
        return []

    output_args = output_nodes[0].args[0]
    if not isinstance(output_args, (tuple, list)):
        output_args = [output_args]

    non_differentiable: list[int] = []
    for out_idx, arg in enumerate(output_args):
        if not isinstance(arg, torch.fx.Node):
            continue
        example = arg.meta.get("example_value")
        if isinstance(example, torch.Tensor) and not example.requires_grad:
            non_differentiable.append(out_idx)

    return non_differentiable


class InvokeSubgraphAutogradOp(torch.autograd.Function):
    """
