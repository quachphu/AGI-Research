                        f"mark_unbacked declared max={_hi} (invariant 3)."
                    )
    pb, _names = _extract_param_buffers(mods)
    _check_structure(pb, _names)
    try:
        out = list(call([*pb, *user_flat]))
    except AssertionError as _e:
        # Only relabel inductor's own assert_size_stride failure (a stride/memory-format
        # mismatch, or a size mismatch on an unbacked dim the static check above cannot
        # pre-validate; invariants 3 and 6). assert_size_stride raises one of two messages
        # -- "expected size A==B, stride C==D at dim=N" or "wrong number of dimensions" --
        # so match those. Any OTHER AssertionError (a user torch._assert, an internal
        # inductor invariant) is re-raised unchanged so its real message is not mislabeled.
        _m = str(_e)
        if not (
            ("expected size" in _m and "stride" in _m)
            or "wrong number of dimensions" in _m
        ):
            raise
        # When the artifact has dynamic (None) user-input dims, an "expected size"
        # assert_size_stride failure on a dynamic dim most likely means two inputs that
        # share a mark_unbacked shape_id (bound to ONE symbol, hence equal by
        # construction) were called with mismatched sizes. Call that out so the message
        # is not misleadingly only about memory format.
        _has_dynamic = any(
            _s is not None and any(_d is None for _d in _s) for _s in USER_INPUT_SHAPES
        )
        _shape_id_note = ""
        if _has_dynamic and "expected size" in _m:
            _shape_id_note = (
                " If two inputs share a mark_unbacked shape_id, their marked dims are "
                "bound to one symbol and so MUST have equal sizes at runtime; this can "
                "also be a shape_id equality violation."
            )
        _fail(
            f"precompile: a runtime tensor's shape or memory format differs from the "
            f"traced example; the inductor backend specializes on input shape and memory "
            f"format (invariants 3 and 6). The mismatch can be a user INPUT or a model "
            f"PARAMETER/BUFFER whose layout (memory format) differs from the example "
            f"weight, since the inductor backend also bakes each param/buffer's layout. "
            f"Pass the model/inputs in the example's shape and layout (.contiguous() to "
            f"match a contiguous example, or match the example weight's layout), or use "
            f"backend='eager'.{_shape_id_note} Underlying: {_e}"
        )
        # Unreachable: _fail always raises. The bare re-raise keeps `out` provably bound
        # for static tools (which do not model _fail as NoReturn) and re-raises the
        # original assert if _fail were ever changed to return.
        raise
    if GRAD_PARAM_INDICES:
        n = len(GRAD_PARAM_INDICES)
        grads = out[len(out) - n :]
        out = out[: len(out) - n]
        for idx, g in zip(GRAD_PARAM_INDICES, grads):
            p = pb[idx]
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)
    return _pytree.tree_unflatten(out, _pytree.treespec_loads(OUT_SPEC))
<|endoftext|># mypy: allow-untyped-defs
r"""
This package implements abstractions found in ``torch.cuda``
to facilitate writing device-agnostic code.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import torch

from .. import device as _device
from . import amp


__all__ = [
    "is_available",
    "is_initialized",
    "synchronize",
    "current_device",
    "current_stream",
    "stream",
    "set_device",
    "device_count",
    "Stream",
    "StreamContext",
    "Event",
    "get_capabilities",
]


@lru_cache(None)
def get_capabilities() -> Mapping[str, Any]:
    """
    Returns an immutable mapping of CPU capabilities detected at runtime.

    This function queries the CPU for supported instruction sets and features
    using cpuinfo. The result is cached after the first call for efficiency.

    The returned mapping contains architecture-specific capabilities:

    For x86/x86_64:
        - SSE family: sse, sse2, sse3, ssse3, sse4_1, sse4_2, sse4a
        - AVX family: avx, avx2, avx_vnni
        - AVX-512 family: avx512_f, avx512_cd, avx512_dq, avx512_bw, avx512_vl,
          avx512_ifma, avx512_vbmi, avx512_vbmi2, avx512_bitalg, avx512_vpopcntdq,
          avx512_vnni, avx512_bf16, avx512_fp16, avx512_vp2intersect,
          avx512_4vnniw, avx512_4fmaps
        - AVX10 family: avx10_1, avx10_2
        - AVX-VNNI-INT: avx_vnni_int8, avx_vnni_int16, avx_ne_convert
        - AMX: amx_bf16, amx_tile, amx_int8, amx_fp16
        - FMA: fma3, fma4
        - Other: f16c, bmi, bmi2, popcnt, lzcnt, aes, sha, clflush, clflushopt, clwb

    For ARM64:
        - SIMD: neon, fp16_arith, bf16, i8mm, dot
        - SVE: sve, sve2, sve_bf16, sve_max_length (when supported)
        - SME: sme, sme2, sme_max_length (when supported)
        - Other: atomics, fhm, rdm, crc32, aes, sha1, sha2, pmull

    Common to all architectures:
        - architecture: string identifying the CPU architecture

    Returns:
        MappingProxyType: An immutable mapping where keys are capability names
        (e.g., 'avx2', 'sve') and values are booleans indicating
        support, or integers for properties like vector lengths.

    Example:
        >>> caps = torch.cpu.get_capabilities()
        >>> if caps.get("avx2", False):
        ...     print("AVX2 is supported")
        >>> print(f"Architecture: {caps['architecture']}")
    """
    return MappingProxyType(torch._C._cpu._get_cpu_capability())


def _is_avx2_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX2."""
    return get_capabilities().get("avx2", False)


def _is_avx512_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512."""
    return get_capabilities().get("avx512_f", False)


def _is_avx512_bf16_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512_BF16."""
    return get_capabilities().get("avx512_bf16", False)


def _is_vnni_supported() -> bool:
    r"""Returns a bool indicating if CPU supports VNNI."""
    # Note: Currently, it only checks avx512_vnni, will add the support of avx2_vnni later.
    return get_capabilities().get("avx512_vnni", False)


def _is_amx_tile_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AMX_TILE."""
    return get_capabilities().get("amx_tile", False)


def _is_amx_fp16_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AMX FP16."""
    return get_capabilities().get("amx_fp16", False)


def _init_amx() -> bool:
    r"""Initializes AMX instructions."""
    return torch._C._cpu._init_amx()


def is_available() -> bool:
    r"""Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return True


def synchronize(device: torch.types.Device = None) -> None:
    r"""Waits for all kernels in all streams on the CPU device to complete.

    Args:
        device (torch.device or int, optional): ignored, there's only one CPU device.

    N.B. This function only exists to facilitate device-agnostic code.
    """


class Stream:
    """
    N.B. This class only exists to facilitate device-agnostic code
    """

    def __init__(self, priority: int = -1) -> None:
        pass

    def wait_stream(self, stream) -> None:
        pass

    def record_event(self) -> None:
        pass

    def wait_event(self, event) -> None:
        pass


class Event:
    def query(self) -> bool:
        return True

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass


_default_cpu_stream = Stream()
_current_stream = _default_cpu_stream


def current_stream(device: torch.types.Device = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): Ignored.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return _current_stream


class StreamContext(AbstractContextManager):
    r"""Context-manager that selects a given stream.

    N.B. This class only exists to facilitate device-agnostic code

    """

    cur_stream: Stream | None

    def __init__(self, stream):
        self.stream = stream
        self.prev_stream = _default_cpu_stream

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        self.prev_stream = _current_stream
        _current_stream = cur_stream

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        _current_stream = self.prev_stream


def stream(stream: Stream) -> AbstractContextManager:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return StreamContext(stream)


def device_count() -> int:
    r"""Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return 1


def set_device(device: torch.types.Device) -> None:
    r"""Sets the current device, in CPU we do nothing.

    N.B. This function only exists to facilitate device-agnostic code
    """


def current_device() -> str:
    r"""Returns current device for cpu. Always 'cpu'.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return "cpu"


def is_initialized() -> bool:
    r"""Returns True if the CPU is initialized. Always True.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return True
<|endoftext|># mypy: allow-untyped-defs
import sys
from typing import Any
from typing_extensions import deprecated

import torch


__all__ = ["autocast"]


@deprecated(
    "`torch.cpu.amp.autocast(args...)` is deprecated. "
    "Please use `torch.amp.autocast('cpu', args...)` instead.",
    category=FutureWarning,
)
class autocast(torch.amp.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.cpu.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cpu", args...)`` instead.
    """

    # TODO: remove this conditional once we stop supporting Python < 3.13
    # Prior to Python 3.13, inspect.signature could not retrieve the correct
    # signature information for classes decorated with @deprecated (unless
    # the __new__ static method was explicitly defined);
    #
    # However, this issue has been fixed in Python 3.13 and later versions.
    if sys.version_info < (3, 13):

        def __new__(
            cls,
            enabled: bool = True,
            dtype: torch.dtype = torch.bfloat16,
            cache_enabled: bool = True,
        ):
            return super().__new__(cls)

        def __init_subclass__(cls):
            pass

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cpu"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)
<|endoftext|>from typing_extensions import deprecated

import torch


__all__ = ["GradScaler"]


class GradScaler(torch.amp.GradScaler):
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.cpu.amp.GradScaler(args...)`` is deprecated. Please use ``torch.amp.GradScaler("cpu", args...)`` instead.
    """

    @deprecated(
        "`torch.cpu.amp.GradScaler(args...)` is deprecated. "
        "Please use `torch.amp.GradScaler('cpu', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            "cpu",
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
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


__all__ = ["ASGD", "asgd"]


class ASGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "lambd": lambd,
            "alpha": alpha,
            "t0": t0,
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
                if len(p_state) != 0:
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                    if not torch.is_tensor(p_state["eta"]):
                        p_state["eta"] = torch.tensor(
                            p_state["eta"], dtype=_get_scalar_dtype(), device=p.device
                        )
                    if not torch.is_tensor(p_state["mu"]):
                        p_state["mu"] = torch.tensor(
                            p_state["mu"], dtype=_get_scalar_dtype(), device=p.device
                        )

    def _init_group(self, group, params_with_grad, grads, mus, axs, etas, state_steps):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("ASGD does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.zeros(
                        (), device=p.device, dtype=_get_scalar_dtype()
                    )
                    state["eta"] = (
                        torch.as_tensor(
                            _to_scalar(group["lr"]),
                            device=p.device,
                            dtype=_get_scalar_dtype(),
                        )
                        .clone()
                        .detach()
                    )
                    state["mu"] = torch.ones(
                        (), device=p.device, dtype=_get_scalar_dtype()
                    )
                    state["ax"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                mus.append(state["mu"])
                axs.append(state["ax"])
                etas.append(state["eta"])
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
            mus: list[Tensor] = []
            axs: list[Tensor] = []
            etas: list[Tensor] = []
            state_steps: list[Tensor] = []

            has_complex = self._init_group(
                group, params_with_grad, grads, mus, axs, etas, state_steps
            )

            asgd(
                params_with_grad,
                grads,
                axs,
                mus,
                etas,
                state_steps,
                lambd=group["lambd"],
                lr=group["lr"],
                t0=group["t0"],
                alpha=group["alpha"],
                weight_decay=group["weight_decay"],
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )

        return loss


ASGD.__doc__ = rf"""Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        {_capturable_doc}

    .. _Acceleration of stochastic approximation by averaging:
        https://doi.org/10.1137/0330046

    """


def _single_tensor_asgd(
    params: list[Tensor],
    grads: list[Tensor],
    axs: list[Tensor],
    mus: list[Tensor],
    etas: list[Tensor],
    state_steps: list[Tensor],
    *,
    lambd: float,
    lr: float | Tensor,
    t0: float,
    alpha: float,
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
        mu = mus[i]
        ax = axs[i]
        eta = etas[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            if not (
                param.device.type
                == mu.device.type
                == eta.device.type
                == step_t.device.type
                and param.device.type in capturable_supported_devices
            ):
                raise AssertionError(
                    f"If capturable=True, params, mus, etas, and state_steps must be "
                    f"on supported devices: {capturable_supported_devices}."
                )

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            param = torch.view_as_real(param)
            ax = torch.view_as_real(ax)

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if capturable:
            param.mul_(1 - lambd * eta)
            param.addcmul_(grad, eta, value=-1)  # update parameter
        else:
            eta_value = _get_value(eta)
            param.mul_(1 - lambd * eta_value)  # decay term
            param.add_(grad, alpha=-eta_value)  # type: ignore[arg-type]  # update parameter

        # averaging
        if capturable or mu.item() != 1:
            ax.add_(param.sub(ax).mul_(mu))
        else:
            ax.copy_(param)

        if capturable:
            eta.copy_(lr / ((1 + lambd * lr * step_t) ** alpha))
            mu.copy_(1 / torch.maximum(step_t - t0, torch.ones_like(step_t)))
        else:
            step = _get_value(step_t)
            new_eta = torch.as_tensor(lr / ((1 + lambd * lr * step) ** alpha))
            eta.copy_(new_eta)
            new_mu = torch.as_tensor(1 / max(1, step - t0))
            mu.copy_(new_mu)


def _multi_tensor_asgd(
    params: list[Tensor],
    grads: list[Tensor],
    axs: list[Tensor],
    mus: list[Tensor],
    etas: list[Tensor],
    state_steps: list[Tensor],
    *,
    lambd: float,
    lr: float | Tensor,
    t0: float,
    alpha: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
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
            p.device.type == mu.device.type == eta.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, mu, eta, step in zip(params, mus, etas, state_steps, strict=True)
        ):
            raise AssertionError(
                f"If capturable=True, params, mus, etas, and state_steps must be on "
                f"supported devices: {capturable_supported_devices}."
            )

    lr = _to_scalar(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, axs, mus, etas, state_steps]  # type: ignore[list-item]
    )
    for (device, _), (
        (
            grouped_params_,
            grouped_grads_,
            grouped_axs_,
            grouped_mus_,
            grouped_etas_,
            grouped_state_steps_,
        ),
        _,
    ) in grouped_tensors.items():
        grouped_params = cast(list[Tensor], grouped_params_)
        grouped_grads = cast(list[Tensor], grouped_grads_)
        grouped_axs = cast(list[Tensor], grouped_axs_)
        grouped_mus = cast(list[Tensor], grouped_mus_)
        grouped_etas = cast(list[Tensor], grouped_etas_)
        grouped_state_steps = cast(list[Tensor], grouped_state_steps_)

        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_axs)

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

        # intermediate = grad + param * lambd
        intermediate: tuple[Tensor, ...] | list[Tensor]
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
                intermediate = grouped_grads
            else:
                intermediate = torch._foreach_add(
                    grouped_grads, grouped_params, alpha=weight_decay
                )

            torch._foreach_add_(intermediate, grouped_params, alpha=lambd)
        else:
            intermediate = torch._foreach_add(
                grouped_grads, grouped_params, alpha=lambd
            )

        # update param
        # param * (1 - lambd * eta) - eta * grad
        # => param - param * lambd * eta - eta * grad
        # => param - eta * intermediate
        torch._foreach_addcmul_(grouped_params, intermediate, grouped_etas, value=-1)
        del intermediate

        # update grouped_axs
        # averaging: ax = ax + mu * (param - ax)
        # Note (mlazos): We can't use lerp here since it requires weight to be float64
        # and our grouping code requires dtypes to match for all tensors in a group (and it should, since
        # we use the mus in other places)
        # all dtypes need to match, so we could introduce a cast in a loop
        # but since this only adds one additional kernel launch, this looks like the cleaner
        # and faster solution
        intermediate = torch._foreach_sub(grouped_params, grouped_axs)
        torch._foreach_addcmul_(grouped_axs, intermediate, grouped_mus)
        del intermediate

        new_etas: tuple[Tensor, ...] | list[Tensor]
        new_mus: tuple[Tensor, ...] | list[Tensor]
        if capturable:
            # update grouped_mus
            new_mus = torch._foreach_sub(grouped_state_steps, t0)
            torch._foreach_maximum_(new_mus, 1.0)
            torch._foreach_reciprocal_(new_mus)
            torch._foreach_copy_(grouped_mus, new_mus)
            del new_mus

            # update eta = lr / ((1 + lambd * lr * step)^alpha)
            new_etas = torch._foreach_mul(grouped_state_steps, lambd)
            torch._foreach_mul_(new_etas, lr)
            torch._foreach_add_(new_etas, 1)
            torch._foreach_pow_(new_etas, alpha)
            torch._foreach_reciprocal_(new_etas)
            torch._foreach_mul_(new_etas, lr)
            torch._foreach_copy_(grouped_etas, new_etas)
        else:
            new_etas = [
                torch.as_tensor(lr / ((1 + lambd * lr * step) ** alpha), device=device)
                for step in grouped_state_steps
            ]
            new_mus = [
                torch.as_tensor(1 / max(1, _get_value(step) - t0), device=device)
                for step in grouped_state_steps
            ]
            torch._foreach_copy_(grouped_etas, new_etas)
            torch._foreach_copy_(grouped_mus, new_mus)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_asgd)
def asgd(
    params: list[Tensor],
    grads: list[Tensor],
    axs: list[Tensor],
    mus: list[Tensor],
    etas: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: bool | None = None,
    maximize: bool = False,
    differentiable: bool = False,
    capturable: bool = False,
    has_complex: bool = False,
    *,
    lambd: float,
    lr: float | Tensor,
    t0: float,
    alpha: float,
    weight_decay: float,
) -> None:
    r"""Functional API that performs asgd algorithm computation.

    See :class:`~torch.optim.ASGD` for details.
    """
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_asgd
    else:
        func = _single_tensor_asgd

    func(
        params,
        grads,
        axs,
        mus,
        etas,
        state_steps,
        lambd=lambd,
        lr=lr,
        t0=t0,
        alpha=alpha,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
<|endoftext|># mypy: allow-untyped-defs
import contextlib
from collections import defaultdict
from typing import Any, NamedTuple

import torch
from torch import Tensor


class _GroupSwapinInfo(NamedTuple):
    """Per-group bookkeeping for ``swap_in_optimizer_params_and_state``.

    Fields:
        live_group: live ``optimizer.param_groups[i]`` dict, mutated in place.
        swapin_group: per-group hyperparameter dict from the input state_dict
            (e.g. lr, betas, ...) plus packed ``params`` ids.
        swapin_params: replacement parameter tensors for ``live_group["params"]``.
            Same order as ``swapin_group["params"]`` (i.e. the i-th tensor
            corresponds to the i-th packed id), so the two are zippable.
    """

    live_group: dict[str, Any]
    swapin_group: dict[str, Any]
    swapin_params: list[Tensor]


def _validate_state_field(state: Any) -> dict[Any, Any]:
    """Check ``state`` is a dict mapping packed parameter ids to per-param state dicts."""
    if not isinstance(state, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires swapin_optim_state['state'] to "
            "be a dict mapping packed parameter ids to per-param state dicts, "
            f"got {type(state).__name__}."
        )
    if any(isinstance(k, torch.Tensor) for k in state):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "state keyed by packed parameter ids."
        )
    if any(isinstance(k, int) and not isinstance(v, dict) for k, v in state.items()):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires per-parameter optimizer "
            "state entries to be dictionaries."
        )
    return state


def _validate_param_groups_field(
    optimizer: "torch.optim.Optimizer", param_groups: Any
) -> list[dict[str, Any]]:
    """``param_groups`` must be a list whose length matches the live optimizer's param groups."""
    if not isinstance(param_groups, list):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires swapin_optim_state['param_groups'] "
            f"to be a list of param-group dicts, got {type(param_groups).__name__}."
        )
    if len(optimizer.param_groups) != len(param_groups):
        raise RuntimeError(
            "swapin_optim_state has a different number of parameter groups than "
            "the live optimizer."
        )
    return param_groups


def _validate_group_against_live(
    idx: int,
    group: dict[str, Any],
    swapin_group: Any,
) -> list[int]:
    """Validate a single swap-in param group against its live counterpart and
    return its packed parameter ids.

    Asserts:
    1. swap-in group is a dict.
    2. ``group['params']`` is a list of ints, and its length matches the params of the live optim group.
    3. Keys inside swapin group should match the keys in the live optim group.
    """
    if not isinstance(swapin_group, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires each optimizer param group "
            "to be a dictionary."
        )
    swapin_param_ids = swapin_group.get("params")
    if not isinstance(swapin_param_ids, list) or not all(
        isinstance(pid, int) for pid in swapin_param_ids
    ):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "param_groups[*]['params'] entries keyed by packed parameter ids."
        )
    if len(group["params"]) != len(swapin_param_ids):
        raise RuntimeError(
            f"swapin_optim_state param group {idx} has a different number of "
            "params than the live optimizer param group."
        )
    swapin_only = [k for k in swapin_group if k not in group]
    live_only = [k for k in group if k not in swapin_group]
    if swapin_only or live_only:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "param group keys to exactly match the live optimizer group keys for "
            f"group {idx}. "
            f"Keys only in swap-in: {swapin_only}. Keys only in live: {live_only}."
        )
    return swapin_param_ids


def _prepare_swap_in(
    optimizer: "torch.optim.Optimizer",
    swapin_parameters: dict[str, Tensor],
    swapin_optim_state: dict[str, Any],
) -> tuple[dict[Any, Any], list[_GroupSwapinInfo]]:
    """
    Validate and normalize optimizer state for ``swap_in_optimizer_params_and_state``.

    This follows the same structural assumptions as DCP-compatible optimizers,
    but consumes the raw ``optimizer.state_dict()`` format:
    ``state`` is keyed by packed parameter ids and each param group contains
    the live optimizer group fields plus a packed ``params`` list whose order
    matches ``optimizer.param_groups``.
    """
    if not optimizer.state:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires initialized optimizer state."
        )
    if not isinstance(swapin_optim_state, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires a DCP-style optimizer state_dict."
        )
    swapin_state = _validate_state_field(swapin_optim_state.get("state"))
    swapin_param_groups = _validate_param_groups_field(
        optimizer, swapin_optim_state.get("param_groups")
    )

    # Raw optimizer state_dicts address parameters by packed integer ids, so we
    # align explicit parameter tensors with optimizer.param_groups by order.
    # Example: if param_groups[*]["params"] is [[0, 1], [2]] and
    # swapin_parameters.values() is [fake_p0, fake_p1, fake_p2], then the
    # first optimizer group is swapped onto [fake_p0, fake_p1] and the
    # second onto [fake_p2].
    flat_parameters = list(swapin_parameters.values())
    flat_param_offset = 0
    seen_param_ids: set[int] = set()
    group_swapin_infos = []
    for idx, (group, swapin_group) in enumerate(
        zip(optimizer.param_groups, swapin_param_groups, strict=True)
    ):
        swapin_param_ids = _validate_group_against_live(idx, group, swapin_group)
        seen_param_ids.update(swapin_param_ids)

        next_offset = flat_param_offset + len(swapin_param_ids)
        if next_offset > len(flat_parameters):
            raise RuntimeError(
                "swap_in_optimizer_params_and_state requires the explicit parameter state to "
                "match optimizer.param_groups ordering."
            )
        swapin_params = flat_parameters[flat_param_offset:next_offset]
        flat_param_offset = next_offset

        # ``swapin_group`` is the per-group hyperparameter dict from the
        # input state_dict (e.g. lr, betas, eps, weight_decay, amsgrad,
        # maximize, foreach, capturable, differentiable, fused, ...) plus
        # the packed ``params`` ids; installed onto the live group for the
        # duration of the context. Example for a param-group for AdamW:
        #   {
        #     'lr': 0.1, 'weight_decay': 0.01,
        #     'betas': (0.9, 0.999), 'eps': 1e-08,
        #     'amsgrad': False, 'maximize': False,
        #     'foreach': None, 'capturable': False,
        #     'differentiable': False, 'fused': None,
        #     'decoupled_weight_decay': True,
        #     'params': [0, 1],
        #   }
        group_swapin_infos.append(
            _GroupSwapinInfo(
                live_group=group,
                swapin_group=swapin_group,
                swapin_params=swapin_params,
            )
        )

    extra_keys = [k for k in swapin_state if k not in seen_param_ids]
    if extra_keys:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires swapin_optim_state['state'] to "
            "be keyed only by packed parameter ids from "
            f"param_groups[*]['params']; got extra keys {extra_keys!r}."
        )
    return swapin_state, group_swapin_infos


@contextlib.contextmanager
def swap_in_optimizer_params_and_state(
    optimizer: "torch.optim.Optimizer",
    swapin_parameters: dict[str, Tensor],
    swapin_optim_state: dict[str, Any],
):
    """Temporarily replace an optimizer's parameters and state with the
    supplied params and optim states, then restore them on exit.

    For the duration of the context, all optimizer APIs (including
    user hooks) see the swap-in values; the live optimizer is restored on
    exit.

    The difference between this API and ``optimizer.load_state_dict`` is
    that ``optimizer.load_state_dict`` only updates the optimizer's state
    and leaves the parameters in ``param_groups`` untouched. This API also
    swaps in the parameters, so that ``optimizer.step()`` acts on the
    swap-in parameter tensors you supply.

    Args:
        optimizer: the live optimizer; its state must already be
            initialized.
        swapin_parameters: tensors to use as parameters during the context,
            provided in the same order as the existing input parameters to
            the optimizer (most commonly in ``model.named_parameters()``
            order).
        swapin_optim_state: an ``optimizer.state_dict()``-shaped dict
            (``{"state": ..., "param_groups": ...}``) holding the state to
            install. ``"state"`` is keyed by packed integer parameter ids
            and ``"param_groups"`` mirrors ``optimizer.param_groups``,
            with each ``"params"`` entry as a list of those packed ids
            and the remaining keys carrying per-group hyperparameters
            (``lr``, ``betas``, ``foreach``, ``capturable``, ...).
            Only in-place tensor edits propagate back to the user supplied
            ``swapin_optim_state``; all other side-effects (e.g.,
            assigning a new tensor to the optim state)
            are ignored.

    Example:

        One use of this API is to run ``optimizer.step()`` against
        ``FakeTensor`` versions of the parameters and state for nonstrict
        tracing — capturing an FX graph of the step without touching the
        live optimizer::

            from torch.fx.experimental.proxy_tensor import make_fx
            from torch._subclasses import FakeTensorMode
            from torch.utils import _pytree as pytree

            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            with fake_mode:
                fake_params = {
                    n: fake_mode.from_tensor(p) for n, p in model.named_parameters()
                }
                fake_osd = pytree.tree_map_only(
                    torch.Tensor, fake_mode.from_tensor, optimizer.state_dict()
                )


            def step_fn(params, osd):
                with swap_in_optimizer_params_and_state(optimizer, params, osd):
                    optimizer.step()
                return params, osd


            gm = make_fx(step_fn)(fake_params, fake_osd)
    """
    state, group_swapin_infos = _prepare_swap_in(
        optimizer, swapin_parameters, swapin_optim_state
    )

    original_state = optimizer.state
    # Shallow-copy each group dict so we can restore the full pre-swap
    # contents (params list, lr, betas, ...) on exit. Shallow
    # copy preserves the identity of the original ``params`` list, which
    # callers may compare with ``is``.
    original_group_snapshots = [dict(g) for g in optimizer.param_groups]

    try:
        swapin_state: defaultdict[Tensor, Any] = defaultdict(dict)

        for info in group_swapin_infos:
            # Swap the explicit tensors and saved group metadata onto the
            # live optimizer group.
            info.live_group["params"] = info.swapin_params
            for key, value in info.swapin_group.items():
                if key == "params":
                    continue
                info.live_group[key] = value

            for swapin_param, param_id in zip(
                info.swapin_params, info.swapin_group["params"], strict=True
            ):
                # Re-key per-parameter optimizer state from packed ids to the
                # swapped-in parameter tensors. Shallow-copy the per-param dict so
                # inplace tensor updates are propagated but structural mutations
                # to the state_dict during the context stay local.
                swapin_state[swapin_param] = dict(state.get(param_id, {}))

        optimizer.state = swapin_state
        yield
    finally:
        # Restore each live group dict to its pre-swap contents.
        for group, snapshot in zip(
            optimizer.param_groups, original_group_snapshots, strict=True
        ):
            group.clear()
            group.update(snapshot)
        optimizer.state = original_state
<|endoftext|># mypy: allow-untyped-defs
from typing import Any, cast

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
    _maximize_doc,
    _params_doc,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["Adadelta", "adadelta"]


class Adadelta(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        foreach: bool | None = None,
        *,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "rho": rho,
            "eps": eps,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "capturable": capturable,
            "foreach": foreach,
            "differentiable": differentiable,
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
        self,
        group: dict[str, Any],
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        square_avgs: list[Tensor],
        acc_deltas: list[Tensor],
        state_steps: list[Tensor],
    ):
        has_complex = False
        p: Tensor
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adadelta does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # Lazy state initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.zeros((), dtype=_get_scalar_dtype())
                )

                state["square_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["acc_delta"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            square_avgs.append(state["square_avg"])
            acc_deltas.append(state["acc_delta"])
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
            square_avgs: list[Tensor] = []
            acc_deltas: list[Tensor] = []
            state_steps: list[Tensor] = []
            (
                lr,
                rho,
                eps,
                weight_decay,
                foreach,
                maximize,
                differentiable,
                capturable,
            ) = (
                group["lr"],
                group["rho"],
                group["eps"],
                group["weight_decay"],
                group["foreach"],
                group["maximize"],
                group["differentiable"],
                group["capturable"],
            )

            has_complex = self._init_group(
                group, params_with_grad, grads, square_avgs, acc_deltas, state_steps
            )

            adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                state_steps,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
                capturable=capturable,
                has_complex=has_complex,
            )

        return loss


Adadelta.__doc__ = (
    r"""Implements Adadelta algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `ADADELTA: An Adaptive Learning Rate Method`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9). A higher value of `rho` will
            result in a slower average, which can be helpful for preventing
            oscillations in the learning process.
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_capturable_doc}
        {_maximize_doc}
        {_differentiable_doc}

    .. _ADADELTA\: An Adaptive Learning Rate Method:
        https://arxiv.org/abs/1212.5701

    """
)


def _single_tensor_adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float | Tensor,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
) -> None:
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

    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for param, grad, square_avg, acc_delta, step in zip(
        params, grads, square_avgs, acc_deltas, state_steps, strict=True
    ):
        step += 1
        grad = grad if not maximize else -grad

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            square_avg = torch.view_as_real(square_avg)
            acc_delta = torch.view_as_real(acc_delta)
            grad = torch.view_as_real(grad)

        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        std = square_avg.add(eps).sqrt_()
        delta = acc_delta.add(eps).sqrt_()
        if differentiable:
            delta = delta.clone()
        delta.div_(std).mul_(grad)
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

        if torch.is_complex(param):
            delta = torch.view_as_complex(delta)
        param.add_(delta, alpha=-lr)  # type: ignore[arg-type]


def _multi_tensor_adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float | Tensor,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
) -> None:
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

    if len(params) == 0:
        return

    lr = _to_scalar(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, square_avgs, acc_deltas, state_steps]  # type: ignore[list-item]
    )
    for (
        device_params_,
        device_grads_,
        device_square_avgs_,
        device_acc_deltas_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_square_avgs = cast(list[Tensor], device_square_avgs_)
        device_acc_deltas = cast(list[Tensor], device_acc_deltas_)
        device_state_steps = cast(list[Tensor], device_state_steps_)
        if has_complex:
            _view_as_real(
                device_params, device_grads, device_square_avgs, device_acc_deltas
            )

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

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0:
            # Reuse the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        torch._foreach_mul_(device_square_avgs, rho)
        torch._foreach_addcmul_(
            device_square_avgs, device_grads, device_grads, value=1 - rho
        )

        std = torch._foreach_add(device_square_avgs, eps)
        torch._foreach_sqrt_(std)

        deltas = torch._foreach_add(device_acc_deltas, eps)
        torch._foreach_sqrt_(deltas)
        torch._foreach_div_(deltas, std)
        torch._foreach_mul_(deltas, device_grads)

        torch._foreach_mul_(device_acc_deltas, rho)
        torch._foreach_addcmul_(device_acc_deltas, deltas, deltas, value=1 - rho)

        # If LR is a tensor, the else branch will internally call item()
        # which will cause silent incorrectness if we are capturing
        if capturable and isinstance(lr, torch.Tensor):
            torch._foreach_mul_(deltas, -lr)
            torch._foreach_add_(device_params, deltas)
        else:
            torch._foreach_add_(device_params, deltas, alpha=-lr)  # type: ignore[arg-type]


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adadelta)
def adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    capturable: bool = False,
    foreach: bool | None = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    lr: float | Tensor,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    r"""Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # We still respect when the user inputs False for foreach.
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adadelta
    else:
        func = _single_tensor_adadelta

    func(
        params,
        grads,
        square_avgs,
        acc_deltas,
        state_steps,
        lr=lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
<|endoftext|>"""Base optimizer."""

import functools
import warnings
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from itertools import chain
from typing import Any, cast, overload, TypeAlias, TypeVar
from typing_extensions import deprecated, ParamSpec, Self

import torch
import torch.utils.hooks as hooks
from torch.utils._foreach_utils import (
    _get_foreach_kernels_supported_devices,
    _get_fused_kernels_supported_devices,
    _group_tensors_by_device_and_dtype,
    Indices,
    TensorListList,
)
from torch.utils.hooks import RemovableHandle


_T = TypeVar("_T")
_P = ParamSpec("_P")

Args: TypeAlias = tuple[Any, ...]
Kwargs: TypeAlias = dict[str, Any]
StateDict: TypeAlias = dict[str, Any]
DeviceDict: TypeAlias = dict[torch.device | None, torch.Tensor]
DeviceDtypeDict: TypeAlias = dict[tuple[torch.device, torch.dtype] | None, torch.Tensor]

GlobalOptimizerPreHook: TypeAlias = Callable[
    ["Optimizer", Args, Kwargs], tuple[Args, Kwargs] | None
]
GlobalOptimizerPostHook: TypeAlias = Callable[["Optimizer", Args, Kwargs], None]

__all__ = [
    "Optimizer",
    "register_optimizer_step_pre_hook",
    "register_optimizer_step_post_hook",
]
_global_optimizer_pre_hooks: dict[int, GlobalOptimizerPreHook] = OrderedDict()
_global_optimizer_post_hooks: dict[int, GlobalOptimizerPostHook] = OrderedDict()
_foreach_supported_types = [torch.Tensor, torch.nn.parameter.Parameter]


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self) -> str:
        return "<required parameter>"


required = _RequiredParameter()


def _use_grad_for_differentiable(func: Callable[_P, _T]) -> Callable[_P, _T]:
    def _use_grad(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        import torch._dynamo

        self = cast(Optimizer, args[0])  # assume first positional arg is `self`
        prev_grad = torch.is_grad_enabled()
        try:
            # Note on graph break below:
            # we need to graph break to ensure that aot respects the no_grad annotation.
            # This is important for perf because without this, functionalization will generate an epilogue
            # which updates the mutated parameters of the optimizer which is *not* visible to inductor, as a result,
            # inductor will allocate for every parameter in the model, which is horrible.
            # With this, aot correctly sees that this is an inference graph, and functionalization will generate
            # an epilogue which is appended to the graph, which *is* visible to inductor, as a result, inductor sees that
            # step is in place and is able to avoid the extra allocation.
            # In the future, we will either 1) continue to graph break on backward, so this graph break does not matter
            # or 2) have a fully fused forward and backward graph, which will have no_grad by default, and we can remove this
            # graph break to allow the fully fused fwd-bwd-optimizer graph to be compiled.
            # see https://github.com/pytorch/pytorch/issues/104053
            torch.set_grad_enabled(self.defaults["differentiable"])
            torch._dynamo.graph_break()
            ret = func(*args, **kwargs)
        finally:
            torch._dynamo.graph_break()
            torch.set_grad_enabled(prev_grad)
        return ret

    functools.update_wrapper(_use_grad, func)
    return _use_grad


@overload
def _get_value(x: torch.Tensor) -> torch.Tensor: ...


@overload
def _get_value(x: float) -> float: ...


def _get_value(x: torch.Tensor | float) -> torch.Tensor | float:
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        return x
    else:
        return x.item() if isinstance(x, torch.Tensor) else x


def _stack_if_compiling(
    x: Sequence[torch.Tensor | float],
) -> torch.Tensor | Sequence[torch.Tensor | float]:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        return torch.stack(cast(list[torch.Tensor], x))
    else:
        return x


def _disable_dynamo_if_unsupported(
    single_tensor_fn: Callable[..., object] | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # workaround for torchscript BC
    # it requires all called functions to be in the
    # global environment at the site at which the
    # maybe_fallback closure is created
    if single_tensor_fn:
        globals()[single_tensor_fn.__name__] = single_tensor_fn

    def wrapper(func: Callable[_P, _T]) -> Callable[_P, _T]:
        import inspect

        disabled_func = torch._disable_dynamo(func)
        ps = inspect.signature(func).parameters
        has_state_steps = True
        try:
            state_steps_ind = list(ps.keys()).index("state_steps")
        except ValueError:
            has_state_steps = False

        # Today, there are cases where we stack state steps
        # and pass them as the value arg of foreach ops.
        # Having state steps on cuda as the value arg is not supported in eager,
        # but this only occurs in the rare case that the user explicitly deletes
        # the capturable flag. If capturable=True, this is not a problem.
        @functools.wraps(func)
        def maybe_fallback(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if torch.compiler.is_compiling() and (
                not kwargs.get("capturable", False)
                and has_state_steps
                and (arg := args[state_steps_ind])
                and isinstance(arg, Sequence)
                and arg[0].device.type in {"cuda", "xpu"}
                or (
                    "state_steps" in kwargs
                    and (kwarg := kwargs["state_steps"])
                    and isinstance(kwarg, Sequence)
                    and kwarg[0].device.type in {"cuda", "xpu"}
                )
            ):
                return disabled_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return maybe_fallback

    return wrapper


# For any optimizer with a faster implementation, we attempt to default to the
# fastest + stablest whenever possible. For foreach, the requirements are to have
# native params all on CUDA. For fused, there's currently the additional requirement
# that the tensors' dtypes must be floating point. Neither alternative supports
# torch.jit.script nor differentiable, so we fall back to the single tensor
# implementation in those cases.
def _default_to_fused_or_foreach(
    params: list[torch.Tensor], differentiable: bool, use_fused: bool = False
) -> tuple[bool, bool]:
    if torch.jit.is_scripting() or differentiable:
        return False, False

    fused_supported_devices = _get_fused_kernels_supported_devices()
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    fused = use_fused and all(
        p is None
        or (
            type(p) in _foreach_supported_types
            and p.device.type in fused_supported_devices
            and torch.is_floating_point(p)
        )
        for p in params
    )
    foreach = not fused and all(
        p is None
        or (
            type(p) in _foreach_supported_types
            and p.device.type in foreach_supported_devices
        )
        for p in params
    )
    return fused, foreach


def _device_dtype_check_for_fused(
    p: torch.Tensor, cuda_unsupported: bool = False
) -> None:
    fused_supported_devices = _get_fused_kernels_supported_devices()
    if cuda_unsupported:
        fused_supported_devices.remove("cuda")
    if not (p.device.type in fused_supported_devices and torch.is_floating_point(p)):
        raise RuntimeError(
            "`fused=True` requires all the params to be floating point Tensors of "
            f"supported devices: {fused_supported_devices} but {p.dtype} and {p.device.type}"
        )


def _view_as_real(
    params: list[torch.Tensor], *state_and_grads: list[torch.Tensor]
) -> None:
    for i, p in enumerate(params):
        if torch.is_complex(p):
            params[i] = torch.view_as_real(params[i])
            for s in state_and_grads:
                s[i] = torch.view_as_real(s[i])


def _get_scalar_dtype(is_fused: bool | None = None) -> torch.dtype:
    if is_fused:
        return torch.float32
    return (
        torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    )


def _get_capturable_supported_devices(supports_xla: bool = True) -> list[str]:
    r"""Return the device type list that supports capturable optimizer."""
    capturable_supported_devices = ["cuda", "xpu", "hpu"]
    if not torch.jit.is_scripting():
        capturable_supported_devices.append(torch._C._get_privateuse1_backend_name())
    if supports_xla:
        capturable_supported_devices.append("xla")
    return capturable_supported_devices


@overload
def _to_scalar(x: torch.Tensor) -> torch.Tensor: ...


@overload
def _to_scalar(x: float) -> float: ...


def _to_scalar(x: torch.Tensor | float) -> torch.Tensor | float:
    r"""This function converts a hyperparameter to a 0-dimension (scalar) tensor
    if it is a nonzero-dimensions 1-element tensor. If it is not a tensor, it is
    kept as is.

    Args:
        x (float or Tensor): A hyperparameter of the optimizer.
            If it is Tensor, it is needed to be 1-element.

    Returns:
        float or Tensor:
            a scalar tensor if x is Tensor otherwise Python scalar (float) value.
    """
    if isinstance(x, torch.Tensor) and x.dim() != 0:
        return x.squeeze()
    else:
        return x


# Common doc strings among optimizers
_params_doc = r"""params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named"""

_foreach_doc = r"""foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)"""

_fused_doc = r"""fused (bool, optional): whether the fused implementation is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None)

    .. note:: The foreach and fused implementations are typically faster than the for-loop,
              single-tensor implementation, with fused being theoretically fastest with both
              vertical and horizontal fusion. As such, if the user has not specified either
              flag (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
              implementation when the tensors are all on CUDA. Why not fused? Since the fused
              implementation is relatively new, we want to give it sufficient bake-in time.
              To specify fused, pass True for fused. To force running the for-loop
              implementation, pass False for either foreach or fused. """

_capturable_doc = r"""capturable (bool, optional): whether this instance is safe to
            capture in a graph, whether for CUDA graphs or for torch.compile support.
            Tensors are only capturable when on supported :ref:`accelerators<accelerators>`.
            Passing True can impair ungraphed performance, so if you don't intend to graph
            capture this instance, leave it False (default: False)"""

_differentiable_doc = r"""differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)"""

_maximize_doc = r"""maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)"""


def register_optimizer_step_pre_hook(hook: GlobalOptimizerPreHook) -> RemovableHandle:
    r"""Register a pre hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle


def register_optimizer_step_post_hook(hook: GlobalOptimizerPostHook) -> RemovableHandle:
    r"""Register a post hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_post_hooks)
    _global_optimizer_post_hooks[handle.id] = hook
    return handle


ParamsT: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, torch.Tensor]]
)

R = TypeVar("R")
T = TypeVar("T")


class Optimizer:
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    OptimizerPreHook: TypeAlias = Callable[
        [Self, Args, Kwargs],  # type: ignore[misc]
        tuple[Args, Kwargs] | None,
    ]
    OptimizerPostHook: TypeAlias = Callable[[Self, Args, Kwargs], None]  # type: ignore[misc]

    _optimizer_step_pre_hooks: dict[int, OptimizerPreHook]
    _optimizer_step_post_hooks: dict[int, OptimizerPostHook]
    # pyrefly: ignore [not-a-type]
    _optimizer_state_dict_pre_hooks: 'OrderedDict[int, Callable[["Optimizer"], None]]'
    _optimizer_state_dict_post_hooks: (
        # pyrefly: ignore [not-a-type]
        'OrderedDict[int, Callable[["Optimizer", StateDict], StateDict | None]]'
    )
    _optimizer_load_state_dict_pre_hooks: (
        # pyrefly: ignore [not-a-type]
        'OrderedDict[int, Callable[["Optimizer", StateDict], StateDict | None]]'
    )
    _optimizer_load_state_dict_post_hooks: (
        # pyrefly: ignore [not-a-type]
        'OrderedDict[int, Callable[["Optimizer"], None]]'
    )

    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None:
        torch._C._log_api_usage_once("python.optimizer")
        self.defaults = defaults
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()

        self._patch_step_function()

        if isinstance(params, torch.Tensor):
            raise TypeError(
                "params argument given to the optimizer should be "
                "an iterable of Tensors or dicts, but got " + torch.typename(params)
            )

        self.state: defaultdict[torch.Tensor, Any] = defaultdict(dict)
        self.param_groups: list[dict[str, Any]] = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.add_param_group(cast(dict[str, Any], param_group))

        # Allows _accelerator_graph_capture_health_check to rig a poor man's TORCH_WARN_ONCE in python,
        # which I don't think exists
        # https://github.com/pytorch/pytorch/issues/72948
        self._warned_capturable_if_run_uncaptured = True

    def __getstate__(self) -> dict[str, Any]:
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "_optimizer_step_pre_hooks" not in self.__dict__:
            self._optimizer_step_pre_hooks = OrderedDict()
        if "_optimizer_step_post_hooks" not in self.__dict__:
            self._optimizer_step_post_hooks = OrderedDict()
        if "_optimizer_state_dict_pre_hooks" not in self.__dict__:
            self._optimizer_state_dict_pre_hooks = OrderedDict()
        if "_optimizer_state_dict_post_hooks" not in self.__dict__:
            self._optimizer_state_dict_post_hooks = OrderedDict()
        if "_optimizer_load_state_dict_pre_hooks" not in self.__dict__:
            self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        if "_optimizer_load_state_dict_post_hooks" not in self.__dict__:
            self._optimizer_load_state_dict_post_hooks = OrderedDict()
        self._patch_step_function()  # To support multiprocessing pickle/unpickle
        self.defaults.setdefault("differentiable", False)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += f"Parameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string

    # Currently needed by Adam and AdamW
    def _accelerator_graph_capture_health_check(self) -> None:
        # Note [torch.compile x capturable]
        # If we are compiling, we try to take the capturable path automatically by
        # setting the flag to True during tracing. Due to this, we skip all the checks
        # normally required for determining whether we can use CUDA/XPU graphs and
        # shunt the responsibility to torch.inductor. This saves time during tracing
        # since the checks are slow without sacrificing UX since inductor will warn
        # later if CUDA/XPU graphs cannot be enabled, e.g.,
        # https://github.com/pytorch/pytorch/blob/d3ba8901d8640eb16f88b2bfef9df7fa383d4b47/torch/_inductor/compile_fx.py#L390.
        # Thus, when compiling, inductor will determine if cudagraphs
        # can be enabled based on whether there is input mutation or CPU tensors.
        if torch.compiler.is_compiling():
            return

        # Determine available accelerator device
        accelerator = torch.accelerator.current_accelerator(check_available=True)

        if accelerator and accelerator.type in {"cuda", "xpu"}:
            capturing = torch.accelerator.current_stream().is_capturing()

            if capturing and not all(
                group["capturable"] for group in self.param_groups
            ):
                raise RuntimeError(
                    f"Attempting {accelerator.type.upper()} graph capture of step() for an instance of "
                    + self.__class__.__name__
                    + " but param_groups' capturable is False."
                )

            if (
                (not getattr(self, "_warned_capturable_if_run_uncaptured", False))
                and all(group["capturable"] for group in self.param_groups)
                and (not capturing)
            ):
                warnings.warn(
                    "This instance was constructed with capturable=True or some of all the param_groups came with capturable=True, "
                    f"but step() is running without {accelerator.type.upper()} graph capture. If you never intend to graph-capture this "
                    "instance, capturable=True can impair performance, and you should set capturable=False.",
                    stacklevel=2,
                )
                self._warned_capturable_if_run_uncaptured = True

    # Backward compatibility alias for internal callers still using the old name.
    _cuda_graph_capture_health_check = deprecated(
        "Use _accelerator_graph_capture_health_check instead",
    )(_accelerator_graph_capture_health_check)

    def _optimizer_step_code(self) -> None:
        """Entry point for `torch.profile.profiler`.

        When python tracing is enabled the profiler will hook into this
        function at the CPython level to inspect the optimizer's parameters and
        param groups. It is called after `step()` since many optimizers
        lazily initialize state.

        This is a workaround due to lack of a proper step hook on the optimizer,
        and will be removed if it exists.
        """

    @staticmethod
    def profile_hook_step(func: Callable[_P, R]) -> Callable[_P, R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> R:
            self, *_ = args
            self = cast(Optimizer, self)
            profile_name = f"Optimizer.step#{self.__class__.__name__}.step"
            with torch.autograd.profiler.record_function(profile_name):
                # call optimizer step pre hooks
                for pre_hook in chain(
                    _global_optimizer_pre_hooks.values(),
                    self._optimizer_step_pre_hooks.values(),
                ):
                    result = pre_hook(self, args, kwargs)
                    if result is not None:
                        if isinstance(result, tuple) and len(result) == 2:
                            args, kwargs = result  # type: ignore[assignment]
                        else:
                            raise RuntimeError(
                                f"{func} must return None or a tuple of (new_args, new_kwargs), but got {result}."
                            )

                # pyrefly: ignore [invalid-param-spec]
                out = func(*args, **kwargs)
                self._optimizer_step_code()

                # call optimizer step post hooks
                for post_hook in chain(
                    self._optimizer_step_post_hooks.values(),
                    _global_optimizer_post_hooks.values(),
                ):
                    post_hook(self, args, kwargs)

                return out

        return wrapper

    @staticmethod
    def _group_tensors_by_device_and_dtype(
        tensorlistlist: TensorListList,
        with_indices: bool = False,
    ) -> (
        dict[tuple[None, None], tuple[TensorListList, Indices]]
        | dict[tuple[torch.device, torch.dtype], tuple[TensorListList, Indices]]
    ):
        """Group a list of lists of tensors by device and dtype.

        Skips this step if we are compiling since this will occur during inductor lowering.
        """
        if torch.compiler.is_compiling():
            return {(None, None): (tensorlistlist, list(range(len(tensorlistlist[0]))))}
        else:
            return _group_tensors_by_device_and_dtype(tensorlistlist, with_indices)  # type: ignore[return-value, arg-type]

    def _patch_step_function(self) -> None:
        self._zero_grad_profile_name = (
            f"Optimizer.zero_grad#{self.__class__.__name__}.zero_grad"
        )
        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = self.profile_hook_step(self.__class__.step)  # type: ignore[assignment]
            self.__class__.step.hooked = True  # type: ignore[attr-defined]

    def register_step_pre_hook(self, hook: OptimizerPreHook) -> RemovableHandle:
        r"""Register an optimizer step pre hook which will be called before optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_step_pre_hooks)
        self._optimizer_step_pre_hooks[handle.id] = hook
        return handle

    def register_step_post_hook(self, hook: OptimizerPostHook) -> RemovableHandle:
        r"""Register an optimizer step post hook which will be called after optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_step_post_hooks)
        self._optimizer_step_post_hooks[handle.id] = hook
        return handle

    def register_state_dict_pre_hook(
        self, hook: Callable[["Optimizer"], None], prepend: bool = False
    ) -> RemovableHandle:
        r"""Register a state dict pre-hook which will be called before :meth:`~torch.optim.Optimizer.state_dict` is called.

        It should have the following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.
        The hook will be called with argument ``self`` before calling ``state_dict`` on ``self``.
        The registered hook can be used to perform pre-processing before the ``state_dict``
        call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_state_dict_pre_hooks)
        self._optimizer_state_dict_pre_hooks[handle.id] = hook
        if prepend:
            self._optimizer_state_dict_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_state_dict_post_hook(
        self,
        hook: Callable[["Optimizer", StateDict], StateDict | None],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a state dict post-hook which will be called after :meth:`~torch.optim.Optimizer.state_dict` is called.

        It should have the following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The hook will be called with arguments ``self`` and ``state_dict`` after generating
        a ``state_dict`` on ``self``. The hook may modify the state_dict inplace or optionally
        return a new one. The registered hook can be used to perform post-processing
        on the ``state_dict`` before it is returned.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_state_dict_post_hooks)
        self._optimizer_state_dict_post_hooks[handle.id] = hook
        if prepend:
            self._optimizer_state_dict_post_hooks.move_to_end(handle.id, last=False)
        return handle

    @torch._disable_dynamo
    def state_dict(self) -> StateDict:
        r"""Return the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * ``state``: a Dict holding current optimization state. Its content
            differs between optimizer classes, but some common characteristics
            hold. For example, state is saved per parameter, and the parameter
            itself is NOT saved. ``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.
        * ``param_groups``: a List containing all parameter groups where each
            parameter group is a Dict. Each parameter group contains metadata
            specific to the optimizer, such as learning rate and weight decay,
            as well as a List of parameter IDs of the parameters in the group.
            If a param group was initialized with ``named_parameters()`` the names
            content will also be saved in the state dict.

        NOTE: The parameter IDs may look like indices but they are just IDs
        associating state with param_group. When loading from a state_dict,
        the optimizer will zip the param_group ``params`` (int IDs) and the
        optimizer ``param_groups`` (actual ``nn.Parameter`` s) in order to
        match state WITHOUT additional verification.

        A returned state dict might look something like:

        .. code-block:: text

            {
                'state': {
                    0: {'momentum_buffer': tensor(...), ...},
                    1: {'momentum_buffer': tensor(...), ...},
                    2: {'momentum_buffer': tensor(...), ...},
                    3: {'momentum_buffer': tensor(...), ...}
                },
                'param_groups': [
                    {
                        'lr': 0.01,
                        'weight_decay': 0,
                        ...
                        'params': [0]
                        'param_names' ['param0']  (optional)
                    },
                    {
                        'lr': 0.001,
                        'weight_decay': 0.5,
                        ...
                        'params': [1, 2, 3]
                        'param_names': ['param1', 'layer.weight', 'layer.bias'] (optional)
                    }
                ]
            }

        """
        for pre_hook in self._optimizer_state_dict_pre_hooks.values():
            pre_hook(self)

        # Save order indices instead of Tensors
        param_mappings: dict[int, int] = {}
        start_index = 0

        def pack_group(group: dict[str, Any]) -> dict[str, Any]:
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(group["params"], start_index)
                    if id(p) not in param_mappings
                }
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }

        state_dict = {
            "state": packed_state,
            "param_groups": param_groups,
        }

        for post_hook in self._optimizer_state_dict_post_hooks.values():
            hook_result = post_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        return state_dict

    @staticmethod
    def _process_value_according_to_param_policy(
        param: torch.Tensor,
        value: torch.Tensor,
        param_id: int,
        param_groups: list[dict[str, Any]],
        key: str | None = None,
    ) -> torch.Tensor:
        # Floating-point types are a bit special here. They are the only ones
        # that are assumed to always match the type of params.
        # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
        # UNLESS fused or capturable, see note [special device hosting for step]
        fused = False
        capturable = False
        if param_groups is None:
            raise AssertionError("Expected param_groups to be set")
        for pg in param_groups:
            if param_id in pg["params"]:
                fused = pg.get("fused", False)
                capturable = pg.get("capturable", False)
                break
        if key == "step":
            if capturable or fused:
                return value.to(dtype=torch.float32, device=param.device)
            else:
                return value
        else:
            if param.is_floating_point():
                return value.to(dtype=param.dtype, device=param.device)
            else:
                return value.to(device=param.device)

    def register_load_state_dict_pre_hook(
        self,
        hook: Callable[["Optimizer", StateDict], StateDict | None],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a load_state_dict pre-hook which will be called before
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The ``optimizer`` argument is the optimizer instance being used and the
        ``state_dict`` argument is a shallow copy of the ``state_dict`` the user
        passed in to ``load_state_dict``. The hook may modify the state_dict inplace
        or optionally return a new one. If a state_dict is returned, it will be used
        to be loaded into the optimizer.

        The hook will be called with argument ``self`` and ``state_dict`` before
        calling ``load_state_dict`` on ``self``. The registered hook can be used to
        perform pre-processing before the ``load_state_dict`` call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_load_state_dict_pre_hooks)
        self._optimizer_load_state_dict_pre_hooks[handle.id] = hook
        if prepend:
            self._optimizer_load_state_dict_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_load_state_dict_post_hook(
        self, hook: Callable[["Optimizer"], None], prepend: bool = False
    ) -> RemovableHandle:
        r"""Register a load_state_dict post-hook which will be called after
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        The hook will be called with argument ``self`` after calling
        ``load_state_dict`` on ``self``. The registered hook can be used to
        perform post-processing after ``load_state_dict`` has loaded the
        ``state_dict``.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_load_state_dict_post_hooks)
        self._optimizer_load_state_dict_post_hooks[handle.id] = hook
        if prepend:
            self._optimizer_load_state_dict_post_hooks.move_to_end(
                handle.id, last=False
            )  # type: ignore[attr-defined]
        return handle

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        r"""Load the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        .. warning::
            Make sure this method is called after initializing :class:`torch.optim.lr_scheduler.LRScheduler`,
            as calling it beforehand will overwrite the loaded learning rates.

        .. note::
            The names of the parameters (if they exist under the "param_names" key of each param group
            in :meth:`state_dict`) will not affect the loading process.
            To use the parameters' names for custom cases (such as when the parameters in the loaded state dict
            differ from those initialized in the optimizer),
            a custom ``register_load_state_dict_pre_hook`` should be implemented to adapt the loaded dict
            accordingly.
            If ``param_names`` exist in loaded state dict ``param_groups`` they will be saved and override
            the current names, if present, in the optimizer state. If they do not exist in loaded state dict,
            the optimizer ``param_names`` will remain unchanged.

        Example:
            >>> # xdoctest: +SKIP
            >>> optimizer = ...  # initialized optimizer matching the saved state
            >>> scheduler1 = torch.optim.lr_scheduler.LinearLR(
            ...     optimizer,
            ...     start_factor=0.1,
            ...     end_factor=1,
            ...     total_iters=20,
            ... )
            >>> scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            ...     optimizer,
            ...     T_max=80,
            ...     eta_min=3e-5,
            ... )
            >>> lr = torch.optim.lr_scheduler.SequentialLR(
            ...     optimizer,
            ...     schedulers=[scheduler1, scheduler2],
            ...     milestones=[20],
            ... )
            >>> lr.load_state_dict(torch.load("./save_seq.pt"))
            >>> # now load the optimizer checkpoint after loading the LRScheduler
            >>> optimizer.load_state_dict(torch.load("./save_optim.pt"))

        """
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Validate the state_dict
        groups = self.param_groups

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(
            p_len != s_len for p_len, s_len in zip(param_lens, saved_lens, strict=True)
        ):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
                strict=True,
            )
        )

        def _cast(
            param: torch.Tensor,
            value: _T,
            param_id: int,
            param_groups: list[dict[str, Any]],
            key: str | None = None,
        ) -> _T:
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                return cast(
                    _T,
                    Optimizer._process_value_according_to_param_policy(
                        param, value, param_id, param_groups, key
                    ),
                )
            elif isinstance(value, dict):
                casted = {
                    k: _cast(
                        param, v, param_id=param_id, param_groups=param_groups, key=k
                    )
                    for k, v in value.items()
                }
                return cast(_T, casted)
            elif isinstance(value, Iterable):
                # pyrefly: ignore [bad-instantiation]
                rebuilt = type(value)(
                    # pyrefly: ignore [bad-argument-count]
                    _cast(param, v, param_id=param_id, param_groups=param_groups)
                    for v in value
                )  # type: ignore[call-arg]
                return cast(_T, rebuilt)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state: defaultdict[torch.Tensor, dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(
                    param, v, param_id=k, param_groups=state_dict["param_groups"]
                )
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(
            group: dict[str, Any], new_group: dict[str, Any]
        ) -> dict[str, Any]:
            new_group["params"] = group["params"]
            if "param_names" in group and "param_names" not in new_group:
                new_group["param_names"] = group["param_names"]
            return new_group

        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups, strict=True)
        ]
        self.__setstate__({"state": state, "param_groups": param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Reset the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool, optional): Instead of setting to zero, set the grads to None. Default: ``True``

                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:

                1. When the user tries to access a gradient and perform manual ops on it,
                   a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                   are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                   (in one case it does the step with a gradient of 0 and in the other it skips
                   the step altogether).
        """
        foreach = self.defaults.get("foreach", False) or self.defaults.get(
            "fused", False
        )

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()

        per_device_and_dtype_grads: (
            defaultdict[torch.device, defaultdict[torch.dtype, list[torch.Tensor]]]
            | None
        )
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        else:
            per_device_and_dtype_grads = None

        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if not foreach or p.grad.is_sparse:
                                p.grad.zero_()
                            else:
                                if per_device_and_dtype_grads is None:
                                    raise AssertionError(
                                        "Expected per_device_and_dtype_grads to be set"
                                    )
                                per_device_and_dtype_grads[p.grad.device][
                                    p.grad.dtype
                                ].append(p.grad)
            if foreach:
                if per_device_and_dtype_grads is None:
                    raise AssertionError(
                        "Expected per_device_and_dtype_grads to be set"
                    )
                for per_dtype_grads in per_device_and_dtype_grads.values():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        r"""Perform a single optimization step to update parameter.

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError

    @torch._disable_dynamo
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            param_group["params"] = list(params)

        extracted_param_tensors: list[torch.Tensor] = []
        extracted_param_names: list[str] = []
        for param in param_group["params"]:
            if isinstance(param, tuple):
                param_name = param[0]
                extracted_param_names.append(param_name)
                extracted_param_tensors.append(param[1])
            else:
                extracted_param_tensors.append(param)

        param_group["params"] = extracted_param_tensors
        if len(extracted_param_names) != 0:
            if len(extracted_param_names) == len(extracted_param_tensors):
                param_group["param_names"] = extracted_param_names
            else:
                raise ValueError(
                    "all optimizer params should be with/without names. Some param names are missing"
                )

        for param in param_group["params"]:
            if not isinstance(param, torch.Tensor):
                raise TypeError(
                    "optimizer can only optimize Tensors, "
                    "but one of the params is " + torch.typename(param)
                )
            if not self.defaults.get("differentiable", None) and not (
                param.is_leaf or param.retains_grad
            ):
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    f"parameter group didn't specify a value of required optimization parameter {name}"
                )
            else:
                param_group.setdefault(name, default)

        params = param_group["params"]
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; "
                "see github.com/pytorch/pytorch/issues/40967 for more information",
                stacklevel=3,
            )

        param_set: set[torch.Tensor] = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))
            if ("param_names" in param_group) != ("param_names" in group):
                current_group_txt = (
                    "with names" if "param_names" in param_group else "without names"
                )
                raise ValueError(
                    "all optimizer param groups should be with/without names. "
                    f"cannot add param group {current_group_txt} to the optimizer"
                )

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
<|endoftext|># mypy: allow-untyped-defs
r"""Implementation for the RMSprop algorithm."""

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
    _maximize_doc,
    _params_doc,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["RMSprop", "rmsprop"]


class RMSprop(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        capturable: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "alpha": alpha,
            "eps": eps,
            "centered": centered,
            "weight_decay": weight_decay,
            "capturable": capturable,
            "foreach": foreach,
            "maximize": maximize,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)
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
        self,
        group,
        params_with_grad,
        grads,
        square_avgs,
        momentum_buffer_list,
        grad_avgs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)

            if p.grad.is_sparse:
                raise RuntimeError("RMSprop does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.zeros((), dtype=_get_scalar_dtype())
                )
                state["square_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if group["momentum"] > 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                if group["centered"]:
                    state["grad_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
            square_avgs.append(state["square_avg"])
            state_steps.append(state["step"])

            if group["momentum"] > 0:
                momentum_buffer_list.append(state["momentum_buffer"])
            if group["centered"]:
                grad_avgs.append(state["grad_avg"])

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
            square_avgs: list[Tensor] = []
            grad_avgs: list[Tensor] = []
            momentum_buffer_list: list[Tensor] = []
            state_steps: list[Tensor] = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                square_avgs,
                momentum_buffer_list,
                grad_avgs,
                state_steps,
            )

            rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=group["lr"],
                alpha=group["alpha"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                centered=group["centered"],
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )

        return loss


RMSprop.__doc__ = (
    r"""Implements RMSprop algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \alpha \text{ (alpha)}, \: \gamma \text{ (lr)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm}   \lambda \text{ (weight decay)},\: \mu \text{ (momentum)},
                \: centered, \: \epsilon \text{ (epsilon)}                                       \\
            &\textbf{initialize} : v_0 \leftarrow 0 \text{ (square average)}, \:
                \textbf{b}_0 \leftarrow 0 \text{ (buffer)}, \: g^{ave}_0 \leftarrow 0     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}v_t           \leftarrow   \alpha v_{t-1} + (1 - \alpha) g^2_t
                \hspace{8mm}                                                                     \\
            &\hspace{5mm} \tilde{v_t} \leftarrow v_t                                             \\
            &\hspace{5mm}if \: centered                                                          \\
            &\hspace{10mm} g^{ave}_t \leftarrow g^{ave}_{t-1} \alpha + (1-\alpha) g_t            \\
            &\hspace{10mm} \tilde{v_t} \leftarrow \tilde{v_t} -  \big(g^{ave}_{t} \big)^2        \\
            &\hspace{5mm}if \: \mu > 0                                                           \\
            &\hspace{10mm} \textbf{b}_t\leftarrow \mu \textbf{b}_{t-1} +
                g_t/ \big(\sqrt{\tilde{v_t}} +  \epsilon \big)                                   \\
            &\hspace{10mm} \theta_t \leftarrow \theta_{t-1} - \gamma \textbf{b}_t                \\
            &\hspace{5mm} else                                                                   \\
            &\hspace{10mm}\theta_t      \leftarrow   \theta_{t-1} -
                \gamma  g_t/ \big(\sqrt{\tilde{v_t}} + \epsilon \big)  \hspace{3mm}              \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to
    `lecture notes <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ by G. Hinton.
    and centered version `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\gamma/(\sqrt{v} + \epsilon)` where :math:`\gamma`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum factor (default: 0)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        {_capturable_doc}
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}

    """
)


def _single_tensor_rmsprop(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    grad_avgs: list[Tensor],
    momentum_buffer_list: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float | Tensor,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
) -> None:
    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        step = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            if not (
                param.device.type == step.device.type
                and param.device.type in capturable_supported_devices
            ):
                raise AssertionError(
                    f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
                )

        grad = grads[i]
        grad = grad if not maximize else -grad
        square_avg = square_avgs[i]

        step += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        is_complex_param = torch.is_complex(param)
        if is_complex_param:
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            square_avg = torch.view_as_real(square_avg)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if centered:
            grad_avg = grad_avgs[i]
            if is_complex_param:
                grad_avg = torch.view_as_real(grad_avg)
            grad_avg.lerp_(grad, 1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_()
        else:
            avg = square_avg.sqrt()

        if differentiable:
            avg = avg.add(eps)
        else:
            avg = avg.add_(eps)

        if momentum > 0:
            buf = momentum_buffer_list[i]
            if is_complex_param:
                buf = torch.view_as_real(buf)
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr)  # type: ignore[arg-type]
        else:
            param.addcdiv_(grad, avg, value=-lr)  # type: ignore[arg-type]


def _multi_tensor_rmsprop(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    grad_avgs: list[Tensor],
    momentum_buffer_list: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float | Tensor,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
) -> None:
    if len(params) == 0:
        return

    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices()
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
        [params, grads, square_avgs, grad_avgs, momentum_buffer_list, state_steps]  # type: ignore[list-item]
    )
    for (
        (
            grouped_params_,
            grouped_grads_,
            grouped_square_avgs_,
            grouped_grad_avgs_,
            grouped_momentum_buffer_list_,
            grouped_state_steps_,
        )
    ), _ in grouped_tensors.values():
        grouped_params = cast(list[Tensor], grouped_params_)
        grouped_grads = cast(list[Tensor], grouped_grads_)
        grouped_square_avgs = cast(list[Tensor], grouped_square_avgs_)
        grouped_state_steps = cast(list[Tensor], grouped_state_steps_)

        if has_complex:
            state_and_grads = [grouped_grads, grouped_square_avgs]
            if momentum > 0:
                grouped_momentum_buffer_list = cast(
                    list[Tensor], grouped_momentum_buffer_list_
                )
                state_and_grads.append(grouped_momentum_buffer_list)
            if centered:
                grouped_grad_avgs = cast(list[Tensor], grouped_grad_avgs_)
                state_and_grads.append(grouped_grad_avgs)
            _view_as_real(grouped_params, *state_and_grads)

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
            # Reuse the intermediate memory (grouped_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
            else:
                grouped_grads = torch._foreach_add(  # type: ignore[assignment]
                    grouped_grads, grouped_params, alpha=weight_decay
                )

        torch._foreach_mul_(grouped_square_avgs, alpha)
        torch._foreach_addcmul_(
            grouped_square_avgs, grouped_grads, grouped_grads, value=1 - alpha
        )

        if centered:
            grouped_grad_avgs = cast(list[Tensor], grouped_grad_avgs_)
            torch._foreach_lerp_(grouped_grad_avgs, grouped_grads, 1 - alpha)
            avg = torch._foreach_addcmul(
                grouped_square_avgs, grouped_grad_avgs, grouped_grad_avgs, value=-1
            )
            torch._foreach_sqrt_(avg)
            torch._foreach_add_(avg, eps)
        else:
            avg = torch._foreach_sqrt(grouped_square_avgs)
            torch._foreach_add_(avg, eps)

        if momentum > 0:
            grouped_momentum_buffer_list = cast(
                list[Tensor], grouped_momentum_buffer_list_
            )
            torch._foreach_mul_(grouped_momentum_buffer_list, momentum)
            torch._foreach_addcdiv_(grouped_momentum_buffer_list, grouped_grads, avg)
            # If LR is a tensor, the else branch will internally call item()
            # which will cause silent incorrectness if we are capturing
            if capturable and isinstance(lr, torch.Tensor):
                momentum_lr = torch._foreach_mul(grouped_momentum_buffer_list, -lr)
                torch._foreach_add_(grouped_params, momentum_lr)
            else:
                torch._foreach_add_(  # type: ignore[arg-type]
                    grouped_params, grouped_momentum_buffer_list, alpha=-lr
                )
        else:
            # If LR is a tensor, the else branch will internally call item()
            # which will cause silent incorrectness if we are capturing
            if capturable and isinstance(lr, torch.Tensor):
                torch._foreach_div_(avg, -lr)
                torch._foreach_addcdiv_(grouped_params, grouped_grads, avg)
            else:
                torch._foreach_addcdiv_(  # type: ignore[arg-type]
                    grouped_params, grouped_grads, avg, value=-lr
                )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rmsprop)
def rmsprop(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    grad_avgs: list[Tensor],
    momentum_buffer_list: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: bool | None = None,
    maximize: bool = False,
    differentiable: bool = False,
    capturable: bool = False,
    has_complex: bool = False,
    *,
    lr: float | Tensor,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
) -> None:
    r"""Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """
    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
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
        func = _multi_tensor_rmsprop
    else:
        func = _single_tensor_rmsprop

    func(
        params,
        grads,
        square_avgs,
        grad_avgs,
        momentum_buffer_list,
        state_steps,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        has_complex=has_complex,
    )
<|endoftext|># mypy: allow-untyped-defs
r"""Implementation for Stochastic Gradient Descent optimizer."""

from typing import cast

import torch
from torch import Tensor

from .optimizer import (
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
    _params_doc,
    _to_scalar,
    _use_grad_for_differentiable,
    DeviceDict,
    Optimizer,
    ParamsT,
)


__all__ = ["SGD", "sgd"]


class SGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float | Tensor = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
            "foreach": foreach,
            "differentiable": differentiable,
            "fused": fused,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            self._need_device_dtype_check_for_fused = True
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                if group["fused"] and getattr(
                    self, "_need_device_dtype_check_for_fused", True
                ):
                    _device_dtype_check_for_fused(p)
                    self._need_device_dtype_check_for_fused = False
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_buffer_list: list[Tensor | None] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(
                    params, momentum_buffer_list, strict=True
                ):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


SGD.__doc__ = (
    r"""Implements stochastic gradient descent (optionally with momentum).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
            \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
            &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                               \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                    \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum. Only applicable
            when momentum is non-zero. (default: False)
        {_maximize_doc}
        {_foreach_doc}
        {_differentiable_doc}
        {_fused_doc}
    """
    + r"""

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.

        Moreover, the initial value of the momentum buffer is set to the
        gradient value at the first step. This is in contrast to some other
        frameworks that initialize it to all zeros. One notable side effect
        of this decision is that the first momentum value will not be scaled
        by dampening. Dampening will be applied starting at the second step.

    """
)


def sgd(
    params: list[Tensor],
    d_p_list: list[Tensor],
    momentum_buffer_list: list[Tensor | None],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: bool | None = None,
    fused: bool | None = None,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float | Tensor,
    dampening: float,
    nesterov: bool,
    maximize: bool,
) -> None:
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if foreach is None and fused is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            fused, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False
            fused = False
    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        func = _fused_sgd
    else:
        func = _single_tensor_sgd

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_sgd(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_buffer_list: list[Tensor | None],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float | Tensor,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")

    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            # Nested if is necessary to bypass jitscript rules
            if isinstance(weight_decay, Tensor):
                if weight_decay.requires_grad:
                    # usually this is the differentiable path, which is why the param.clone() is needed
                    grad = grad.addcmul_(param.clone(), weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = grad.detach().clone()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        # Nested if is necessary to bypass jitscript rules
        if isinstance(lr, Tensor):
            if lr.requires_grad:
                param.addcmul_(grad, lr, value=-1)
            else:
                # pyrefly: ignore [bad-argument-type]
                param.add_(grad, alpha=-lr)
        else:
            param.add_(grad, alpha=-lr)


def _multi_tensor_sgd(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_buffer_list: list[Tensor | None],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float | Tensor,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")

    if len(params) == 0:
        return

    lr = _to_scalar(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list],  # type: ignore[list-item]
        with_indices=True,
    )
    for (
        device_params_,
        device_grads_,
        device_momentum_buffer_list,
    ), indices in grouped_tensors.values():
        device_params: list[Tensor] = cast(list[Tensor], device_params_)
        device_grads: list[Tensor] = cast(list[Tensor], device_grads_)

        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0:
            # Reuse the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        if momentum != 0:
            bufs: list[Tensor] = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(cast(Tensor, device_momentum_buffer_list[i]))

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []

                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[
                            indices[i]
                        ] = device_grads[i].detach().clone()
                    else:
                        buf = cast(Tensor, device_momentum_buffer_list[i])
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs

        if not device_has_sparse_grad:
            # handle internal item() call if lr is a tensor
            if isinstance(lr, torch.Tensor) and torch.compiler.is_compiling():
                grads_x_lr = torch._foreach_mul(device_grads, -lr)
                torch._foreach_add_(device_params, grads_x_lr)
            else:
                torch._foreach_add_(device_params, device_grads, alpha=-lr)  # type: ignore[arg-type]
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)  # type: ignore[arg-type]


def _fused_sgd(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_buffer_list: list[Tensor | None],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float | Tensor,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    if not params:
        return
    if has_sparse_grad:
        raise RuntimeError("`_fused_sgd` does not support sparse gradients")
    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    no_momentum_buffer = momentum == 0
    is_first_step = (
        all(t is None for t in momentum_buffer_list) and not no_momentum_buffer
    )
    if is_first_step:
        for i, g in enumerate(grads):
            momentum_buffer_list[i] = torch.empty_like(g)
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list],  # type: ignore[list-item]
        with_indices=False,
    )
    for (device, _), (
        (device_params_, device_grads_, device_momentum_buffer_list),
        _,
    ) in grouped_tensors.items():
        device_params: list[Tensor] = cast(list[Tensor], device_params_)
        device_grads: list[Tensor] = cast(list[Tensor], device_grads_)
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device)
            )
        if found_inf_dict is not None and found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(device, found_inf.to(device))
        torch._fused_sgd_(
            device_params,
            device_grads,
            []
            if no_momentum_buffer
            else cast(list[Tensor], device_momentum_buffer_list),
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            is_first_step=is_first_step,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
<|endoftext|># mypy: allow-untyped-defs
r"""Learning Rate Scheduler."""

from __future__ import annotations

import math
import types
import warnings
from bisect import bisect_right
from collections import Counter
from functools import partial, wraps
from typing import Any, cast, Literal, SupportsFloat, TYPE_CHECKING, TypedDict
from typing_extensions import override, Self
from weakref import ref

from torch import inf, Tensor

from .optimizer import _to_scalar, Optimizer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


__all__ = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "SequentialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "ReduceLROnPlateau",
    "CyclicLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialLR",
    "LRScheduler",
]

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


def _format_param(name: str, optimizer: Optimizer, param):
    """Return correctly formatted lr/momentum for each param group."""

    def _copy(_param):
        return _param.clone() if isinstance(_param, Tensor) else _param

    if isinstance(param, (list, tuple)):
        if len(param) != len(optimizer.param_groups):
            raise ValueError(
                f"{name} must have the same length as optimizer.param_groups. "
                f"{name} has {len(param)} values, param_groups has {len(optimizer.param_groups)}."
            )
    else:
        param = [param] * len(optimizer.param_groups)

    return list(map(_copy, param))


def _param_groups_val_list(optimizer: Optimizer, key: str) -> list[Any]:
    """Create a list containing group[key] for each optimizer param_group.
    Prevents aliasing when group[key] could be a Tensor.
    Raises a KeyError when group[key] does not exist.
    """
    return [
        group[key].clone() if isinstance(group[key], Tensor) else group[key]
        for group in optimizer.param_groups
    ]


def _update_param_group_val(
    param_group: dict[str, Any], key: str, val: float | Tensor
) -> None:
    """Set param_group[key] to val without aliasing or assignment when they're
    both tensors. Raises a KeyError if param_group[key] does not exist.
    """
    if isinstance(param_group[key], Tensor):
        param_group[key].fill_(_to_scalar(val))
    else:
        param_group[key] = val


class LRScheduler:
    r"""Base class for all learning rate schedulers.

    Subclasses implement :meth:`get_lr` and optionally override :meth:`step` to
    define scheduling behavior.

    Args:
        optimizer (Optimizer): The optimizer this scheduler will adjust the
            learning rates of.
        last_epoch (int): Index of the last epoch seen by the scheduler. Use
            ``-1`` (default) to initialize the scheduler. Only use a non-default
            value when restoring this scheduler from a saved checkpoint.

    .. warning::
        Initializing a scheduler overwrites its optimizer's
        ``param_group["lr"]``\s. When restoring a checkpoint, initialize the
        scheduler **before** calling your optimizer's
        :meth:`~torch.optim.Optimizer.load_state_dict` to avoid overwriting the
        loaded learning rates.
    """

    _get_lr_called_within_step: bool = False
    _is_initial: bool = False

    def __init__(
        self,
        optimizer: Optimizer,
        last_epoch: int = -1,
    ) -> None:
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                initial_lr = group["lr"]
                if isinstance(initial_lr, Tensor):
                    initial_lr = initial_lr.clone()
                group.setdefault("initial_lr", initial_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        f"param 'initial_lr' is not specified in param_groups[{i}] when resuming scheduler with last_epoch >= 0.\n"
                        "This typically happens when:\n"
                        "1. You're trying to resume training from a checkpoint but haven't properly loaded the optimizer state\n"
                        "2. You're using last_epoch >= 0 for a fresh training run (not recommended)"
                    )
        self.base_lrs: list[float | Tensor] = _param_groups_val_list(
            optimizer, "initial_lr"
        )
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def patch_track_step_called(opt: Optimizer):
            if hasattr(opt.step, "_wrapped_by_lr_sched"):
                # we've already patched
                return opt.step

            def wrap_step(step_fn):
                opt_ref = ref(self.optimizer)
                func = step_fn.__func__

                @wraps(func)
                def wrapper(*args, **kwargs):
                    opt = opt_ref()
                    opt._opt_called = True  # type: ignore[union-attr]
                    return func.__get__(opt, opt.__class__)(*args, **kwargs)

                wrapper._wrapped_by_lr_sched = True  # type: ignore[attr-defined]
                return wrapper

            opt.step = wrap_step(opt.step)  # type: ignore[method-assign]

        patch_track_step_called(self.optimizer)
        self._initial_step()

    def _initial_step(self) -> None:
        """Initialize step counts and perform a step."""
        self._step_count = 0
        with _initial_mode(self):
            self.step()

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> list[float | Tensor]:
        r"""Get the most recent learning rates computed by this scheduler.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates with entries
            for each of the optimizer's
            :attr:`~torch.optim.Optimizer.param_groups`, with the same types as
            their ``group["lr"]``\s.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        # We always update self._last_lr with _param_groups_val_list, so it's a
        # .clone() of the group["lr"]s. If we didn't do this, the user could
        # corrupt their learning rates by modifying the outputs in place.
        return self._last_lr

    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

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
        raise NotImplementedError

    def step(self, epoch: int | None = None) -> None:
        """Step the scheduler.

        Args:
            epoch (int, optional):
                .. deprecated:: 1.4
                    If provided, sets :attr:`last_epoch` to ``epoch`` and uses
                    :meth:`_get_closed_form_lr` if it is available. This is not
                    universally supported. Use :meth:`step` without arguments
                    instead.

        .. note::
            Call this method after calling the optimizer's
            :meth:`~torch.optim.Optimizer.step`.
        """
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                    stacklevel=2,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif not getattr(self.optimizer, "_opt_called", False):
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                    stacklevel=2,
                )

        self._step_count += 1
        if epoch is not None:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning, stacklevel=2)
        self._update_lr(epoch)

    def _update_lr(self, epoch: int | None = None) -> None:
        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = cast(list[float | Tensor], self._get_closed_form_lr())
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values, strict=True):
            _update_param_group_val(param_group, "lr", lr)

        self._last_lr: list[float | Tensor] = _param_groups_val_list(
            self.optimizer, "lr"
        )


def _warn_get_lr_called_within_step(lr_scheduler: LRScheduler) -> None:
    if not lr_scheduler._get_lr_called_within_step:
        warnings.warn(
            "To get the last learning rate computed by the scheduler, "
            "please use `get_last_lr()`.",
            UserWarning,
            stacklevel=2,
        )


# Including _LRScheduler for backwards compatibility
# Subclass instead of assign because we want __name__ of _LRScheduler to be _LRScheduler (assigning would make it LRScheduler).
class _LRScheduler(LRScheduler):
    pass


class _enable_get_lr_call:
    def __init__(self, o: LRScheduler) -> None:
        self.o = o

    def __enter__(self) -> Self:
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.o._get_lr_called_within_step = False


class _initial_mode:
    def __init__(self, o: LRScheduler) -> None:
        self.o = o

    def __enter__(self):
        self.o._is_initial = True

    def __exit__(self, type, value, traceback):
        self.o._is_initial = False


class LambdaLR(LRScheduler):
    """Sets the initial learning rate.

    The learning rate of each parameter group is set to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer has two groups.
        >>> num_epochs = 100
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95**epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
        >>>
        >>> # Alternatively, you can use a single lambda function for all groups.
        >>> scheduler = LambdaLR(opt, lr_lambda=lambda epoch: epoch // 30)
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/LambdaLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer

        self.lr_lambdas: list[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the :attr:`base_lrs` by the outputs of the :attr:`lr_lambdas` at
        :attr:`last_epoch`.

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
        _warn_get_lr_called_within_step(self)

        return [
            base_lr * lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs, strict=True)
        ]


class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given in the specified function.

    When last_epoch=-1, set initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> lmbda = lambda epoch: 0.95
        >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/MultiplicativeLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer

        self.lr_lambdas: list[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        for lr_lambda in self.lr_lambdas:
            if not callable(lr_lambda):
                raise TypeError(
                    f"lr_lambda should be a function, but got {type(lr_lambda).__name__}"
                )
        super().__init__(optimizer, last_epoch)

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the current ``group["lr"]``\s in each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by the outputs of the
        :attr:`lr_lambdas` at :attr:`last_epoch`.

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
        _warn_get_lr_called_within_step(self)

        if not self._is_initial:
            return [
                group["lr"] * lmbda(self.last_epoch)
                for lmbda, group in zip(
                    self.lr_lambdas, self.optimizer.param_groups, strict=True
                )
            ]
        else:
            return _param_groups_val_list(self.optimizer, "lr")


class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/StepLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        If the current epoch is a non-zero multiple of :attr:`step_size`, we
        scale the current ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

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
        _warn_get_lr_called_within_step(self)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return _param_groups_val_list(self.optimizer, "lr")
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/MultiStepLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Iterable[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        If the current epoch is in :attr:`milestones`, decays the
        ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

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

        .. note::
            If the current epoch appears in :attr:`milestones` ``n`` times, we
            scale by :attr:`gamma` to the power of ``n``
        """
        _warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.milestones:
            return _param_groups_val_list(self.optimizer, "lr")
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        milestones = sorted(self.milestones.elements())
        return [
            base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class ConstantLR(LRScheduler):
    """Multiply the learning rate of each parameter group by a small constant factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such multiplication of the small constant factor can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler multiplies the learning rate by the factor.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # ...
        >>> # lr = 0.05    if epoch >= 40
        >>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=40)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ConstantLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        if factor > 1.0 or factor < 0:
            raise ValueError(
                "Constant multiplicative factor expected to be between 0 and 1."
            )

        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        When :attr:`last_epoch` is 0, this method scales the ``group["lr"]``\s
        in each of the optimizer's :attr:`~torch.optim.Optimizer.param_groups`
        by :attr:`factor`. Once :attr:`total_iters` is reached, it undoes this,
        scaling by ``1 / factor``.

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
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch != self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        return [
            group["lr"] * (1.0 / self.factor) for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr
            * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for base_lr in self.base_lrs
        ]


class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small multiplicative factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.003687  if epoch == 0
        >>> # lr = 0.004875  if epoch == 1
        >>> # lr = 0.006062  if epoch == 2
        >>> # lr = 0.00725   if epoch == 3
        >>> # ...
        >>> # lr = 0.05      if epoch >= 40
        >>> scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=40)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/LinearLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` such that successive steps
        interpolate linearly from :attr:`start_factor` up to :attr:`end_factor`
        across :attr:`total_iters` steps.

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
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        if self._is_initial or self.last_epoch > self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_lr in self.base_lrs
        ]


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ExponentialLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
    ) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Multiplies the current ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

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
        _warn_get_lr_called_within_step(self)

        # when loading from a checkpoint, we don't want _initial_step (called from the constructor)
        # to update the lr one more step ahead of itself.
        if self._is_initial:
            return _param_groups_val_list(self.optimizer, "lr")
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


class SequentialLR(LRScheduler):
    """Contains a list of schedulers expected to be called sequentially during the optimization process.

    Specifically, the schedulers will be called according to the milestone points, which should provide exact
    intervals by which each scheduler should be called at a given epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.005     if epoch == 0
        >>> # lr = 0.005     if epoch == 1
        >>> # lr = 0.005     if epoch == 2
        >>> # ...
        >>> # lr = 0.05      if epoch == 20
        >>> # lr = 0.045     if epoch == 21
        >>> # lr = 0.0405    if epoch == 22
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = SequentialLR(
        ...     optimizer,
        ...     schedulers=[scheduler1, scheduler2],
        ...     milestones=[20],
        ... )
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/SequentialLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[LRScheduler],
        milestones: list[int],
        last_epoch: int = -1,
    ) -> None:
        if len(schedulers) < 1:
            raise ValueError(
                f"{self.__class__.__name__} expects at least one scheduler, but got no scheduler."
            )

        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(
                    f"{self.__class__.__name__} at index {scheduler_idx} should have `optimizer` as its attribute."
                )
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(
                    f"{self.__class__.__name__} does not support `ReduceLROnPlateau` scheduler as it "
                    "requires additional kwargs to be specified when calling `step`, "
                    f"but got one at index {scheduler_idx} in the given schedulers sequence."
                )
            if optimizer != scheduler.optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler {scheduler.__class__.__name__} at index {scheduler_idx} has {scheduler.optimizer}, "
                    f"which is different from {optimizer.__class__.__name__}."
                )

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                f"than the number of milestone points, but got number of schedulers {len(schedulers)} and the "
                f"number of milestones to be equal to {len(milestones)}"
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

        # Reset learning rates back to initial values
        for group in self.optimizer.param_groups:
            _update_param_group_val(group, "lr", group["initial_lr"])

        # "Undo" the step performed by other schedulers
        self.recursive_undo()

        # Perform the initial step for only the scheduler meant to run at step 0.
        idx = bisect_right(self._milestones, 0)
        self._schedulers[idx]._initial_step()

        self._last_lr = schedulers[idx].get_last_lr()

    def recursive_undo(self, sched=None) -> None:
        """
        Recursively undo any step performed by the initialization of
        schedulers.
        """
        scheds = self if sched is None else sched

        if hasattr(scheds, "_schedulers"):
            for s in scheds._schedulers:
                self.recursive_undo(s)
        elif hasattr(scheds, "last_epoch"):
            scheds.last_epoch -= 1

    def step(self) -> None:  # type: ignore[override]
        """Perform a step."""
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler._update_lr(0)
        else:
            scheduler.step()

        self._last_lr = scheduler.get_last_lr()

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            # pyrefly: ignore [unsupported-operation]
            state_dict["_schedulers"][idx] = s.state_dict()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class PolynomialLR(LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function in the given total_iters.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (float): The power of the polynomial. Default: 1.0.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.0490   if epoch == 0
        >>> # lr = 0.0481   if epoch == 1
        >>> # lr = 0.0472   if epoch == 2
        >>> # ...
        >>> # lr = 0.0      if epoch >= 50
        >>> scheduler = PolynomialLR(optimizer, total_iters=50, power=0.9)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

