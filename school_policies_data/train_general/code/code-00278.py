        Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.

        Example::
            >>> # xdoctest: +SKIP
            >>> g1 = torch.accelerator.Graph()
            >>> g1.capture_begin()
            >>> # ... operations ...
            >>> g1.capture_end()

            >>> # Share g1's memory pool with a new graph
            >>> pool_id = g1.pool()
            >>> g2 = torch.accelerator.Graph(pool=pool_id)
        """
        return super().pool()

    def enable_debug_mode(self) -> None:
        r"""Enable debugging mode for ``debug_dump``."""
        return super().enable_debug_mode()

    def debug_dump(self, path: str) -> None:
        r"""
        Dump the captured graph to a file for debugging purposes if the debugging is
        enabled via ``enable_debug_mode``.

        Arguments:
            path (str): Path to dump the graph to.

        Example::
            >>> # xdoctest: +SKIP
            >>> s = torch.Stream()
            >>> g = torch.accelerator.Graph()
            >>> g.enable_debug_mode()

            >>> with s, g:
            >>> # ... operations ...

            >>> # Dump captured graph to a file "graph_dump.dot"
            >>> g.debug_dump("graph_dump.dot")
        """
        return super().debug_dump(path)

    def __enter__(self) -> None:
        torch.accelerator.synchronize()
        if torch.compiler.config.force_cudagraph_gc:
            # We previously always ran garbage collection here. While this can help
            # reclaim accelerator device memory held by dead Python cycles, it is
            # very expensive, especially when performing multiple graph captures in sequence.
            gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.empty_host_cache()
        self.capture_begin()

    def __exit__(self, *exc_info: object) -> None:
        self.capture_end()


__all__ = ["Graph"]
<|endoftext|>from collections.abc import Sequence

import torch
import torch._C._lazy


def render_ir_graph(tensors: Sequence[torch.Tensor]) -> str:
    """Return a text dump of the LTC IR graph in dot format for the tensors.
    The text can be processed by tools like dot to be rendered in pdf,png etc."""
    return torch._C._lazy._get_tensors_dot(list(tensors))


def dump_ir(tensors: Sequence[torch.Tensor], ir_format: str) -> str:
    """Return a dump of the tensors in the specified format.
    Valid format are
    - text: for LTC IR
    - backend: for the active backend IR
    """
    if ir_format == "text":
        return torch._C._lazy._get_tensors_text(list(tensors))
    elif ir_format == "backend":
        return torch._C._lazy._get_tensors_backend(list(tensors))
    else:
        raise RuntimeError(f"Unrecognized IR format: {ir_format}")
<|endoftext|>from collections.abc import Sequence

import torch
import torch._C._lazy
import torch._C._lazy_ts_backend


def get_tensors_ts_device_data_node(
    tensors: Sequence[torch.Tensor],
) -> tuple[list[int], list[object]]:
    """Return tensor ids and eager tensors for DeviceData nodes in the
    IR for the passed in lazy tensors.

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    return torch._C._lazy_ts_backend._get_tensors_ts_device_data_node(list(tensors))


def get_graph_hash(tensors: Sequence[torch.Tensor]) -> str:
    """Return the graph hash for the passed in lazy tensors"""
    return torch._C._lazy._get_graph_hash(list(tensors))


def run_cached_graph(
    hash_str: str, graph_inputs: Sequence[object]
) -> list[torch.Tensor]:
    """Running the cached computation graph with the given inputs

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    return torch._C._lazy_ts_backend._run_cached_graph(hash_str, list(graph_inputs))
<|endoftext|>import torch._C._lazy


def dump(dot_file_name: str) -> None:
    """Dump TrieCache in the dot format"""
    return torch._C._lazy._dump_ir_cache(dot_file_name)


def reset() -> None:
    """Clear TrieCache. This is needed in testing to avoid
    node reusing between different tests.
    """
    return torch._C._lazy._clear_ir_cache()
<|endoftext|>import copy
import dataclasses
import itertools
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import cast, Protocol, TypeAlias, TypeVar

import torch
import torch._lazy as lazy
import torch._lazy.metrics as metrics
from torch import fx
from torch._lazy import computation, debug as lazy_debug
from torch._lazy.tensor_factory_functions import tensor_factory_functions
from torch.fx.node import Argument


debug = os.environ.get("debug_extract_compiled_graph") is not None

_T = TypeVar("_T")

# A TorchScript graph-input IValue: concretely a const tensor, a device-data
# node, or a runtime argument, none of which share a useful static type here.
_IValue: TypeAlias = object


class _CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> Sequence[torch.Tensor]: ...


@dataclasses.dataclass
class GraphInputMatcher:
    """
    The GraphInputMatcher class setup the graph inputs for future calls after lazy tracing.
    Specifically, those graph inputs corresponding to method parameters should be replaced with the
    arguments for the current call.

    tensor_id_to_arg_idx maps the tensor id to the parameter index.
    graph_input_tensor_ids, graph_input_ivalues list the tensor_id and ivalue for each of the
    TS/XLA graph inputs.
    """

    tensor_id_to_arg_idx: dict[int, int]
    graph_input_tensor_ids: list[int]
    # there are 2 categories of graph_input_tensors.
    # Category 1: those whose id are not found in tensor_id_to_arg_idx. These are
    # most likely const tensors and we can get its content from graph_input_tensors
    # Category 2: those whose id are found in tensor_id_to_arg_idx. We should get
    #  the tensor from method arguments
    graph_input_ivalues: list[_IValue]

    # get the real graph input tensors
    def __call__(self, args: Sequence[object]) -> list[_IValue]:
        real_input: list[_IValue] = []
        for tensor_id, traced_ivalue in zip(
            self.graph_input_tensor_ids, self.graph_input_ivalues
        ):
            arg_idx = self.tensor_id_to_arg_idx.get(tensor_id, None)
            if arg_idx is None:
                inp = traced_ivalue
            else:
                inp = args[arg_idx]
            real_input.append(inp)
        return real_input


class ReturnValueHandler:
    r"""
    When ltc_sync_multi is called on multi tensors, the compiled graph
    will contain output only for unique tensors - if a tensor appears multiple
    times in the input to _ltc_sync_multi, only the first occurrence matters.

    However from python level, we still expect multi tensors returned with duplication
    even if the TS graph dedup the output. e.g. for method:

      def forward(self, a):
        return a, a

    the TS graph captured by LTC will return a single tensor, but Python method expects 2.

    This class dedup the lazy tensors first to get the index that will be used
    to duplicate the eager tensors later.
    """

    def __init__(self, lazy_out_list: Sequence[object]) -> None:
        self.index: list[list[int]] = []
        self.total_count = len(lazy_out_list)

        tensor_id_to_idx: dict[int, int] = {}
        for dup_idx, lazy_tensor in enumerate(lazy_out_list):
            uniq_idx = tensor_id_to_idx.get(id(lazy_tensor))
            if uniq_idx is not None:
                self.index[uniq_idx].append(dup_idx)
            else:
                uniq_idx = len(self.index)
                self.index.append([dup_idx])
                tensor_id_to_idx[id(lazy_tensor)] = uniq_idx

    def duplicate_eager_tensors(self, eager_tensor_list: Sequence[_T]) -> list[_T]:
        duplicated_list: list[_T | None] = [None] * self.total_count
        if len(eager_tensor_list) != len(self.index):
            raise AssertionError(
                f"eager_tensor_list length {len(eager_tensor_list)} != index length {len(self.index)}"
            )

        for uniq_idx, eager_tensor in enumerate(eager_tensor_list):
            for dup_idx in self.index[uniq_idx]:
                duplicated_list[dup_idx] = eager_tensor
        # every slot is filled: self.index partitions range(total_count).
        return cast(list[_T], duplicated_list)


def force_lazy_device(model: fx.GraphModule) -> None:
    """
    Factory methods in a Fx graph may create tensors for a specific eager devices.
    If we take no actions, those eager tensors will be mixed with lazy tensors and
    cause crash. This method overwrite those eager device to lazy device.
    """

    def tolazydevice(dev: _T) -> _T:
        if isinstance(dev, torch.device):
            return cast(_T, torch.device("lazy", index=dev.index))
        return dev

    def hasDeviceArg(args: Iterable[Argument], kwargs: Mapping[str, Argument]) -> bool:
        return any(
            isinstance(arg, torch.device)
            for arg in itertools.chain(args, kwargs.values())
        )

    for nd in model.graph.nodes:
        nd.args = tuple(tolazydevice(arg) for arg in nd.args)
        nd.kwargs = {k: tolazydevice(v) for k, v in nd.kwargs.items()}

        # For torchbench like yolov3, hf_Bart, dynamo generates Fx graph that return
        # eager tensors on the default device
        # (check https://gist.github.com/shunting314/eabdf6c769c59bc384469717b8f9bb7f for yolove,
        # and https://gist.github.com/shunting314/8d5e2d9348a3258959d3954186c48814 for hf_Bart).
        # To force those tensors on the lazy device, we can not simply override
        # the device argument since there is no explicit device argument.
        # What we are doing here is, for the list of covered tensor factory methods
        # we add a lazy device argument explicitly.
        #
        # TODO: This solution is not ideal since we may miss some factory methods. In future
        # when we support lazy mode, this method can be replaced by that.
        if nd.target in tensor_factory_functions and not hasDeviceArg(
            nd.args, nd.kwargs
        ):
            kwargs = dict(nd.kwargs)  # nd.kwargs is immutable. make a mutable copy.
            kwargs["device"] = torch.device("lazy")
            nd.kwargs = kwargs

    model.recompile()


def get_fallback_ops() -> list[str]:
    fallback_ops = []
    for opname in metrics.counter_names():
        if "aten::" not in opname:
            continue
        val = int(metrics.counter_value(opname))
        if val > 0:
            fallback_ops.append(f"{opname}={val}")

    return fallback_ops


def extract_compiled_graph(
    model: fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> _CompiledFn:
    """
    Optimize an eager model with LTC and returns a wrapper to execute the
    compiled graph directly without retracing. It depends on other mechanisms
    like TorchDynamo guards to guarantee the returned wrapper is only called
    when it's safe.
    """
    lazy_args = [arg.to(device="lazy") for arg in example_inputs]
    args_tensor_ids = [lazy.get_tensor_id(lazy_arg) for lazy_arg in lazy_args]
    tensor_id_to_arg_idx = {tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)}
    lazy_model = copy.deepcopy(model).to(device=torch.device("lazy"))
    force_lazy_device(lazy_model)

    # This line executes lazy tracing and enable us extracting compiled graph later
    metrics.reset()
    lazy_out = lazy_model(*lazy_args)
    fallback_ops = get_fallback_ops()
    metrics.reset()

    if len(fallback_ops) > 0:
        raise RuntimeError(
            f"Fail to extract the compiled graph because of fallback: {','.join(fallback_ops)}"
        )

    if not isinstance(lazy_out, (tuple, list)):
        lazy_out = (lazy_out,)

    args_and_out = tuple(lazy_args) + tuple(lazy_out)
    return_value_handler = ReturnValueHandler(args_and_out)
    if debug:
        print("Fx code:\n", model.code)
        print("LTC IR:", lazy_debug.dump_ir(args_and_out, "text"))

    # TODO: this part is TS backend specific for now and will be generalized to
    # support XLA
    (
        graph_input_tensor_ids,
        graph_input_ivalues,
    ) = computation.get_tensors_ts_device_data_node(args_and_out)
    if len(graph_input_tensor_ids) != len(graph_input_ivalues):
        raise AssertionError(
            f"graph_input_tensor_ids length {len(graph_input_tensor_ids)} "
            f"!= graph_input_ivalues length {len(graph_input_ivalues)}"
        )
    graph_input_matcher = GraphInputMatcher(
        tensor_id_to_arg_idx, graph_input_tensor_ids, graph_input_ivalues
    )

    graph_hash = computation.get_graph_hash(args_and_out)

    if debug:
        print("graph_hash", graph_hash)
        print(f"args_tensor_ids {args_tensor_ids}")
        print("tensor ids from device data:", graph_input_tensor_ids)

    # sync the list of output tensors so the computation graph for these
    # tensors will be cached. Those computation graphs can be retrieved
    # by graph hash later.
    lazy.sync_multi(args_and_out, [])

    def optimized_mod(*args: torch.Tensor) -> Sequence[torch.Tensor]:
        if len(args_and_out) == 0:
            return ()
        graph_input = graph_input_matcher(args)
        res = return_value_handler.duplicate_eager_tensors(
            computation.run_cached_graph(graph_hash, graph_input)
        )

        if len(res) != len(args_and_out):
            raise AssertionError(
                f"result length {len(res)} != args_and_out length {len(args_and_out)}"
            )
        for i, arg in enumerate(args):
            # only copy those tensors that get inplace updated
            if arg is not res[i]:
                arg.copy_(res[i])

        # skip the args
        return res[len(args) :]

    return optimized_mod
<|endoftext|>import torch._C._lazy


def get_force_fallback() -> str:
    """Get the config used to force LTC fallback"""
    return torch._C._lazy._get_force_fallback()


def set_force_fallback(configval: str) -> None:
    """Set the config used to force LTC fallback"""
    torch._C._lazy._set_force_fallback(configval)


def set_reuse_ir(val: bool) -> None:
    """Set the config to reuse IR nodes for faster tracing"""
    torch._C._lazy._set_reuse_ir(val)
<|endoftext|>import threading

import torch._C._lazy


class DeviceContext:
    _CONTEXTS: dict[str, "DeviceContext"] = {}
    _CONTEXTS_LOCK = threading.Lock()

    def __init__(self, device: str) -> None:
        self.device = device


def get_device_context(device: str | None = None) -> DeviceContext:
    if device is None:
        device = torch._C._lazy._get_default_device_type()
    else:
        device = str(device)
    with DeviceContext._CONTEXTS_LOCK:
        devctx = DeviceContext._CONTEXTS.get(device, None)
        if devctx is None:
            devctx = DeviceContext(device)
            DeviceContext._CONTEXTS[device] = devctx
        return devctx
<|endoftext|>import pickle
from collections.abc import Sequence
from types import ModuleType
from typing import cast, TypeVar

import torch
import torch._C._lazy
from torch.serialization import DEFAULT_PROTOCOL
from torch.types import FileLike
from torch.utils._pytree import tree_flatten, tree_unflatten

from .closure import add_step_closure, run_step_closures


_T = TypeVar("_T")


def mark_step(device: str = "", wait: bool = False) -> None:
    """Triggers a mark step, which amounts to
    - collecting a group of 'live' lazy tensors to index into the compilation cache
      (lowering/compiling their IR graphs if not cached)
    - kicking off execution of the compiled function
    - (optionally, wait=True) waiting for cpu-side execution to complete (does not sync the accelerator)
    """
    # TODO(whc) expand this to include backend hooks and align with XLA backend needs
    torch._C._lazy._mark_step(device, [], wait=wait)

    run_step_closures()


def wait_device_ops(devices: list[str] | None = None) -> None:
    """Waits for all the async operations on the given devices to complete.
    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
    if devices is None:
        devices = []
    torch._C._lazy._wait_device_ops(devices=devices)


def sync_multi(tensors: Sequence[torch.Tensor], devices: list[str]) -> None:
    """
    Sync the list of lazy tensors so there IR get lowered for the activate backend
    and the compiled computation graph get cached.
    """
    torch._C._lazy._sync_multi(list(tensors), devices)


def get_tensor_id(tensor: torch.Tensor) -> int:
    """Return a unique id of the lazy tensor maintained by LTC"""
    return torch._C._lazy._get_tensor_id(tensor)


def to_cpu(tensors: _T, devices: list[str] | None = None) -> _T:
    devices = devices or ["lazy"]

    flattened, spec = tree_flatten(tensors)
    sync_multi(flattened, devices)
    return cast(_T, tree_unflatten([t.to("cpu") for t in flattened], spec))


def save(
    tensors: object,
    f: FileLike,
    pickle_module: ModuleType = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    torch.save(
        to_cpu(tensors),
        f,
        pickle_module=pickle_module,
        pickle_protocol=pickle_protocol,
        _use_new_zipfile_serialization=_use_new_zipfile_serialization,
        _disable_byteorder_record=_disable_byteorder_record,
    )
<|endoftext|>import os
import threading
from collections.abc import Callable, Iterable
from queue import Empty as EmptyQueue, Queue
from typing import overload
from typing_extensions import TypeVarTuple, Unpack

from torch._lazy.device_context import DeviceContext, get_device_context


_Ts = TypeVarTuple("_Ts")


class ClosureHandler:
    def __init__(self) -> None:
        pass

    def run(self, closure: Callable[[], object]) -> None:
        """Run closure function

        Args:
        closure: callable function to run
        """
        closure()

    def __call__(self, closures: Iterable[Callable[[], object]]) -> None:
        for closure in closures:
            self.run(closure)


class AsyncClosureHandler(ClosureHandler):
    """Handler for Asynchronous Step Closures
    Args:
        max_queue_size: The maximum length of the closure queue after which
        the training loop will block until closures are evaluated.
        By default, a reasonable limit of a maximum of 100 on the queue.
        This value can be set using the `XLA_MAX_ASYNC_QUEUE` environment
        variable.
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        super().__init__()
        self._closure_queue: Queue[Callable[[], object]] = Queue(
            int(os.environ.get("LTC_MAX_ASYNC_QUEUE", max_queue_size))
        )
        self._closure_exception: Queue[Exception] = Queue()
        self._closure_lock = threading.Lock()
        self._closure_event_loop_finished = threading.Event()
        self._closure_event_loop: threading.Thread | None = None

    def start_event_loop(self) -> None:
        """Start closure event loop if not started"""
        if self._closure_event_loop is None:

            def event_loop() -> None:
                # Run loop until closure event is set and closure queue is empty
                while True:
                    try:
                        closure = self._closure_queue.get(block=True, timeout=3)
                        closure()
                        self._closure_queue.task_done()
                    except EmptyQueue:
                        with self._closure_lock:
                            if self._closure_queue.empty():
                                self._closure_event_loop_finished.set()
                                return
                    except Exception as e:
                        self._closure_exception.put(e)
                        return

            self._closure_event_loop = threading.Thread(target=event_loop)
            self._closure_event_loop.start()

    def run(self, closure: Callable[[], object]) -> None:
        with self._closure_lock:
            self._closure_queue.put(closure, block=True)
            if (
                self._closure_event_loop is None
                or not self._closure_event_loop.is_alive()
            ):
                try:
                    e = self._closure_exception.get(block=False)
                    raise RuntimeError(
                        "Cannot run asynchronous closure due to previously raised exception"
                    ) from e
                except EmptyQueue:
                    self._closure_event_loop = None
                    self.start_event_loop()


@overload
def add_step_closure(
    closure: Callable[[], object],
    args: tuple[()] = (),
    run_async: bool = False,
) -> None: ...
@overload
def add_step_closure(
    closure: Callable[[Unpack[_Ts]], object],
    args: tuple[Unpack[_Ts]],
    run_async: bool = False,
) -> None: ...
def add_step_closure(
    closure: Callable[..., object],
    args: tuple[object, ...] = (),
    run_async: bool = False,
) -> None:
    """Adds a closure to the list of the ones to be run at the end of the step.
    Many times during model training there is the need to print/report (print to
    console, post to tensorboard, etc...) information which require the content of
    intermediary tensors to be inspected.
    Inspecting different tensors content in different points of the model code
    requires many executions and typically causes performance issues.
    Adding a step closure will ensure that it will be run after the barrier, when
    all the live tensors will be already materialized to device data.
    Live tensors which will include the ones captured by the closure arguments.
    So using `add_step_closure()` will ensure a single execution will be
    performed, even when multiple closures are queued, requiring multiple tensors
    to be inspected.
    Step closures will be run sequentially in the order they have been queued.
    Note that even though using this API the execution will be optimized, it is
    advised to throttle the printing/reporting events once every N steps.
    Args:
      closure (callable): The function to be called.
      args (tuple): The arguments to be passed to the closure.
      run_async: If True, run the closure asynchronously.
    """
    devctx = get_device_context()
    closures_type = "async_step_closures" if run_async else "step_closures"
    step_closures: list[Callable[[], object]] | None = getattr(
        devctx, closures_type, None
    )
    if step_closures is None:
        step_closures = []
        setattr(devctx, closures_type, step_closures)
    step_closures.append(lambda a=args: closure(*a))


def run_step_closures() -> DeviceContext:
    devctx = get_device_context()
    async_step_closures = getattr(devctx, "async_step_closures", None)
    if async_step_closures is not None:
        devctx.async_step_closures = []  # type: ignore[attr-defined]
        async_closure_handler = getattr(devctx, "async_closure_handler", None)
        if async_closure_handler is None:
            async_closure_handler = AsyncClosureHandler()
            devctx.async_closure_handler = async_closure_handler  # type: ignore[attr-defined]
        async_closure_handler(async_step_closures)

    step_closures = getattr(devctx, "step_closures", None)
    if step_closures is not None:
        devctx.step_closures = []  # type: ignore[attr-defined]
        closure_handler = getattr(devctx, "closure_handler", None)
        if closure_handler is None:
            closure_handler = ClosureHandler()
            devctx.closure_handler = closure_handler  # type: ignore[attr-defined]
        closure_handler(step_closures)
    return devctx
<|endoftext|>import torch._C._lazy


def reset() -> None:
    """Resets all metric counters."""
    torch._C._lazy._reset_metrics()


def counter_names() -> list[str]:
    """Retrieves all the currently active counter names."""
    return torch._C._lazy._counter_names()


def counter_value(name: str) -> int:
    """Return the value of the counter with the specified name"""
    return torch._C._lazy._counter_value(name)


def metrics_report() -> str:
    """Return the combined (lazy core and backend) metric report"""
    return torch._C._lazy._metrics_report()
<|endoftext|>"""This file contains utilities for initializing neural network parameters."""

import math
import warnings
from collections.abc import Callable
from typing import Literal, TypeVar
from typing_extensions import ParamSpec

import torch
from torch import Tensor


__all__ = [
    "calculate_gain",
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "ones_",
    "zeros_",
    "eye_",
    "dirac_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
    "sparse_",
    # Deprecated aliases (for backward compatibility)
    "uniform",
    "normal",
    "constant",
    "eye",
    "dirac",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "sparse",
]


_R = TypeVar("_R")
_P = ParamSpec("_P")

_NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]

_FanMode = Literal["fan_in", "fan_out"]


# These no_grad_* functions are necessary as wrappers around the parts of these
# functions that use `with torch.no_grad()`. The JIT doesn't support context
# managers, so these need to be implemented as builtins. Using these wrappers
# lets us keep those builtins small and reusable.
def _no_grad_uniform_(
    tensor: Tensor, a: float, b: float, generator: torch.Generator | None = None
) -> Tensor:
    with torch.no_grad():
        return tensor.uniform_(a, b, generator=generator)


def _no_grad_normal_(
    tensor: Tensor,
    mean: float,
    std: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)


def _no_grad_trunc_normal_(
    tensor: Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    # Meta tensors have no storage, so sampling is a no-op.
    if tensor.is_meta:
        return tensor

    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        p = norm_cdf((b - mean) / std) - norm_cdf((a - mean) / std)

        if p > 0.3:
            # Cast bounds to tensor dtype so the rejection mask is consistent
            # with the tensor's representable values.
            lo = tensor.new_tensor(a, device="cpu").item()
            hi = tensor.new_tensor(b, device="cpu").item()
            result = tensor.normal_(mean, std, generator=generator)
            while True:
                mask = (result < lo) | (result > hi)
                if not mask.any():
                    break
                result = torch.where(
                    mask,
                    torch.empty_like(result).normal_(mean, std, generator=generator),
                    result,
                )
            if tensor is not result:
                tensor.copy_(result)
        else:
            mode = max(a, min(mean, b))
            log_peak = -0.5 * ((mode - mean) / std) ** 2

            candidates = torch.empty_like(tensor)
            accept_buf = torch.empty_like(tensor)

            # First iteration: sample directly into tensor to avoid
            # a where() + copy_() if all samples are accepted.
            tensor.uniform_(a, b, generator=generator)
            candidates.copy_(tensor)
            # log_pdf = -0.5 * ((candidates - mean) / std) ** 2
            candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)
            pending = accept_buf.uniform_(generator=generator).log_().gt(candidates)
            if not pending.any():
                pass
            else:
                result = tensor
                while True:
                    candidates.uniform_(a, b, generator=generator)
                    result = torch.where(pending, candidates, result)
                    # log_pdf = -0.5 * ((candidates - mean) / std) ** 2
                    candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)
                    pending = torch.where(
                        pending,
                        accept_buf.uniform_(generator=generator).log_().gt(candidates),
                        pending,
                    )
                    if not pending.any():
                        break
                tensor.copy_(result)

        return tensor


def _no_grad_fill_(tensor: Tensor, val: float) -> Tensor:
    with torch.no_grad():
        return tensor.fill_(val)


def _no_grad_zero_(tensor: Tensor) -> Tensor:
    with torch.no_grad():
        return tensor.zero_()


def calculate_gain(
    nonlinearity: _NonlinearityType, param: int | float | None = None
) -> float:
    r"""Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks
        <https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html>`__,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalization
        effect for more stable gradient flow in rectangular layers.

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain(
        ...     "leaky_relu", 0.2
        ... )  # leaky_relu with negative_slope=0.2

    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def uniform_(
    tensor: Tensor,
    a: float = 0.0,
    b: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the input Tensor with values drawn from the uniform distribution.

    :math:`\mathcal{U}(a, b)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            uniform_, (tensor,), tensor=tensor, a=a, b=b, generator=generator
        )
    return _no_grad_uniform_(tensor, a, b, generator)


def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the input Tensor with values drawn from the normal distribution.

    :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            normal_, (tensor,), tensor=tensor, mean=mean, std=std, generator=generator
        )
    return _no_grad_normal_(tensor, mean, std, generator)


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the input Tensor with values drawn from a truncated normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    For reduced-precision types (``torch.float16`` and ``torch.bfloat16``),
    sampling quality depends on the underlying ``normal_()`` and ``uniform_()``
    implementations which operate at higher internal precision to avoid
    quantization artifacts.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b, generator=generator)


def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fill the input Tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            constant_, (tensor,), tensor=tensor, val=val
        )
    return _no_grad_fill_(tensor, val)


def ones_(tensor: Tensor) -> Tensor:
    r"""Fill the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    return _no_grad_fill_(tensor, 1.0)


def zeros_(tensor: Tensor) -> Tensor:
    r"""Fill the input Tensor with the scalar value `0`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
    return _no_grad_zero_(tensor)


def eye_(tensor: Tensor) -> Tensor:
    r"""Fill the 2-dimensional input `Tensor` with the identity matrix.

    Preserves the identity of the inputs in `Linear` layers, where as
    many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.eye_(w)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    with torch.no_grad():
        torch.eye(*tensor.shape, out=tensor, requires_grad=tensor.requires_grad)
    return tensor


def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    r"""Fill the {3, 4, 5}-dimensional input `Tensor` with the Dirac delta function.

    Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`
        groups (int, optional): number of groups in the conv layer (default: 1)
    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
        >>> w = torch.empty(3, 24, 5, 5)
        >>> nn.init.dirac_(w, 3)
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    sizes = tensor.size()

    if sizes[0] % groups != 0:
        raise ValueError("dim 0 must be divisible by groups")

    if tensor.is_meta:
        return tensor

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    with torch.no_grad():
        tensor.zero_()

        for g in range(groups):
            for d in range(min_dim):
                if dimensions == 3:  # Temporal convolution
                    tensor[g * out_chans_per_grp + d, d, tensor.size(2) // 2] = 1
                elif dimensions == 4:  # Spatial convolution
                    tensor[
                        g * out_chans_per_grp + d,
                        d,
                        tensor.size(2) // 2,
                        tensor.size(3) // 2,
                    ] = 1
                else:  # Volumetric convolution
                    tensor[
                        g * out_chans_per_grp + d,
                        d,
                        tensor.size(2) // 2,
                        tensor.size(3) // 2,
                        tensor.size(4) // 2,
                    ] = 1
    return tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier uniform distribution.

    The method is described in `Understanding the difficulty of training
    deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("relu"))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a, generator)


def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Xavier normal distribution.

    The method is described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010). The resulting tensor
    will have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0.0, std, generator)


def _calculate_correct_fan(tensor: Tensor, mode: _FanMode) -> int:
    # pyrefly: ignore [bad-assignment]
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Kaiming uniform distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode="fan_in", nonlinearity="relu")

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.kaiming_uniform_(w.T, ...)``.
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            kaiming_uniform_,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
            generator=generator,
        )

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the input `Tensor` with values using a Kaiming normal distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")

    Note:
        Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
        that the weight matrix is used in a transposed manner,
        (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
        This is important for correct initialization.
        If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
        pass in a transposed weight matrix, i.e. ``nn.init.kaiming_normal_(w.T, ...)``.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)


def orthogonal_(
    tensor: Tensor,
    gain: float = 1,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the input `Tensor` with a (semi) orthogonal matrix.

    Described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    if tensor.numel() == 0 or tensor.is_meta:
        # no-op
        return tensor
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new_empty((rows, cols)).normal_(0, 1, generator=generator)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def sparse_(
    tensor: Tensor,
    sparsity: float,
    std: float = 0.01,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Fill the 2D input `Tensor` as a sparse matrix.

    The non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    if tensor.is_meta:
        return tensor

    rows, cols = tensor.shape
    num_zeros = math.ceil(sparsity * rows)

    with torch.no_grad():
        tensor.normal_(0, std, generator=generator)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


# for backward compatibility
def _make_deprecate(meth: Callable[_P, _R]) -> Callable[_P, _R]:
    new_name = meth.__name__
    old_name = new_name[:-1]

    def deprecated_init(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        warnings.warn(
            f"`nn.init.{old_name}` is now deprecated in favor of `nn.init.{new_name}`.",
            FutureWarning,
            stacklevel=2,
        )
        return meth(*args, **kwargs)

    deprecated_init.__doc__ = rf"""
    {old_name}(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.{new_name}`.

    See :func:`~torch.nn.init.{new_name}` for details."""
    deprecated_init.__name__ = old_name
    return deprecated_init


uniform = _make_deprecate(uniform_)
normal = _make_deprecate(normal_)
constant = _make_deprecate(constant_)
eye = _make_deprecate(eye_)
dirac = _make_deprecate(dirac_)
xavier_uniform = _make_deprecate(xavier_uniform_)
xavier_normal = _make_deprecate(xavier_normal_)
kaiming_uniform = _make_deprecate(kaiming_uniform_)
kaiming_normal = _make_deprecate(kaiming_normal_)
orthogonal = _make_deprecate(orthogonal_)
sparse = _make_deprecate(sparse_)
<|endoftext|>from collections import OrderedDict
from typing import Any

import torch
from torch._C import _disabled_torch_function_impl


__all__ = [
    "Parameter",
    "UninitializedParameter",
    "is_lazy",
    "Buffer",
    "UninitializedBuffer",
    "UninitializedTensorMixin",
]


# Metaclass to combine _TensorMeta and the instance check override for Parameter.
class _ParameterMeta(torch._C._TensorMeta):
    # Make `isinstance(t, Parameter)` return True for custom tensor instances that have the _is_param flag.
    def __instancecheck__(self, instance) -> bool:
        if self is Parameter:
            if isinstance(instance, torch.Tensor) and getattr(
                instance, "_is_param", False
            ):
                return True
        return super().__instancecheck__(instance)


class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a Parameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead. To use the type as a "
                "Parameter, please correct the detach() semantics defined by "
                "its __torch_dispatch__() implementation."
            )
        t._is_param = True
        return t

    # Note: the 3 methods below only apply to standard Tensor. Parameters of custom tensor types
    # are still considered that custom tensor type and these methods will not be called for them.
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format), self.requires_grad
            )
            memo[id(self)] = result
            return result

    # pyrefly: ignore [bad-override]
    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()

    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # See Note [Don't serialize hooks]
        hooks = OrderedDict()
        if not state:
            return (
                torch._utils._rebuild_parameter,
                (self.data, self.requires_grad, hooks),
            )

        return (
            torch._utils._rebuild_parameter_with_state,
            (self.data, self.requires_grad, hooks, state),
        )

    # pyrefly: ignore [bad-override]
    __torch_function__ = _disabled_torch_function_impl


class UninitializedTensorMixin:
    _allowed_methods = [
        torch.Tensor.__hash__,
        torch.Tensor.size,
        torch.Tensor.copy_,
        torch.Tensor.is_complex,
        torch.Tensor.is_floating_point,
        torch.Tensor.half,
        torch.Tensor.float,
        torch.Tensor.double,
        torch.Tensor.char,
        torch.Tensor.short,
        torch.Tensor.int,
        torch.Tensor.long,
        torch.Tensor.cuda,
        torch.Tensor.cpu,
        torch.Tensor.to,
        torch.Tensor.get_device,
        torch._has_compatible_shallow_copy_type,
    ]

    def materialize(self, shape, device=None, dtype=None) -> None:
        r"""Create a Parameter or Tensor with the same properties of the uninitialized one.

        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
        if device is None:
            device = self.data.device
        if dtype is None:
            dtype = self.data.dtype
        self.data = torch.empty(shape, device=device, dtype=dtype)
        # pyrefly: ignore [missing-attribute]
        self.__class__ = self.cls_to_become

    @property
    def shape(self):
        raise RuntimeError(
            "Can't access the shape of an uninitialized parameter or buffer. "
            "This error usually happens in `load_state_dict` when trying to load "
            "an uninitialized parameter into an initialized one. "
            "Call `forward` to initialize the parameters before accessing their attributes."
        )

    def share_memory_(self):
        raise RuntimeError(
            "Can't share memory on an uninitialized parameter or buffer. "
            "Call `forward` to initialize the parameters before calling "
            "`module.share_memory()`."
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        # pyrefly: ignore [missing-attribute]
        return (self.__class__, (self.requires_grad,))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # method-wrapper is to detect access to Tensor properties that are
        # wrapped in descriptors
        if func in cls._allowed_methods or func.__class__.__name__ == "method-wrapper":
            if kwargs is None:
                kwargs = {}
            # pyrefly: ignore [missing-attribute]
            return super().__torch_function__(func, types, args, kwargs)
        raise ValueError(
            f"Attempted to use an uninitialized parameter in {func}. "
            "This error happens when you are using a `LazyModule` or "
            f"explicitly manipulating `torch.nn.parameter.{cls.__name__}` "
            "objects. When using LazyModules Call `forward` with a dummy batch "
            "to initialize the parameters before calling torch functions"
        )


def is_lazy(param: Any) -> bool:
    """
    Returns whether ``param`` is an ``UninitializedParameter`` or ``UninitializedBuffer``.

    Args:
        param (Any): the input to check.
    """
    return isinstance(param, UninitializedTensorMixin)


# pyrefly: ignore [inconsistent-inheritance]
class UninitializedParameter(UninitializedTensorMixin, Parameter):
    r"""A parameter that is not initialized.

    Uninitialized Parameters are a special case of :class:`torch.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on an uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = Parameter

    def __new__(cls, requires_grad=True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        data = torch.empty(0, **factory_kwargs)
        # pyrefly: ignore [bad-return]
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.requires_grad, self.data.device, self.data.dtype)
            memo[id(self)] = result
            return result


# Metaclass to combine _TensorMeta and the instance check override for Buffer.
class _BufferMeta(torch._C._TensorMeta):
    # Make `isinstance(t, Buffer)` return True for custom tensor instances that have the _is_buffer flag.
    def __instancecheck__(self, instance) -> bool:
        if self is Buffer:
            if isinstance(instance, torch.Tensor) and getattr(
                instance, "_is_buffer", False
            ):
                return True
        return super().__instancecheck__(instance)


class Buffer(torch.Tensor, metaclass=_BufferMeta):
    r"""A kind of Tensor that should not be considered a model
    parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state.

    Buffers are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s -- when they're
    assigned as Module attributes they are automatically added to the list of
    its buffers, and will appear e.g. in :meth:`~torch.nn.Module.buffers` iterator.
    Assigning a Tensor doesn't have such effect. One can still assign a Tensor as explicitly by using
    the :meth:`~torch.nn.Module.register_buffer` function.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is part of the module's
            :attr:`state_dict`. Default: ``True``
    """

    def __new__(cls, data=None, *, persistent=True):
        if data is None:
            data = torch.empty(0)

        t = data.detach().requires_grad_(data.requires_grad)
        # pyrefly: ignore [missing-attribute]
        t.persistent = persistent
        # pyrefly: ignore [missing-attribute]
        t._is_buffer = True
        return t

    # pyrefly: ignore [bad-override]
    __torch_function__ = _disabled_torch_function_impl


class UninitializedBuffer(UninitializedTensorMixin, torch.Tensor):
    r"""A buffer that is not initialized.

    Uninitialized Buffer is a special case of :class:`torch.Tensor`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.Tensor`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on an uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.Tensor`.

    The default device or dtype to use when the buffer is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = torch.Tensor

    def __new__(
        cls, requires_grad=False, device=None, dtype=None, persistent=True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        data = torch.empty(0, **factory_kwargs)
        ret = torch.Tensor._make_subclass(cls, data, requires_grad)
        # pyrefly: ignore [missing-attribute]
        ret.persistent = persistent
        # pyrefly: ignore [missing-attribute]
        ret._is_buffer = True
        # pyrefly: ignore [bad-return]
        return ret
<|endoftext|># mypy: allow-untyped-defs
"""Functionality for Python <-> C++ frontend inter-op."""

from torch import nn


class OrderedDictWrapper:
    """A wrapper around a C++ OrderedDict.

    It dynamically evaluates the OrderedDict getter on a bound C++ module, such
    that new changes on the C++ side are picked up. Otherwise accessing e.g.
    ``cpp_module._parameters`` just once would get a frozen copy of the parameters
    at the time of access. ``torch.nn.Module`` accesses ``_parameters`` et al. via ``self.__dict__``
    so using properties does not work.
    """

    def __init__(self, cpp_module, attr) -> None:
        self.cpp_module = cpp_module
        self.attr = attr

    @property
    def cpp_dict(self):
        return getattr(self.cpp_module, self.attr)

    # Magic methods cannot be assigned dynamically and bypass ``getattr``, so we
    # must manually override them.

    def items(self):
        return self.cpp_dict.items()

    def keys(self):
        return self.cpp_dict.keys()

    def values(self):
        return self.cpp_dict.values()

    def __iter__(self):
        return self.cpp_dict.__iter__()

    def __len__(self) -> int:
        return self.cpp_dict.__len__()

    def __contains__(self, key) -> bool:
        return self.cpp_dict.__contains__(key)

    def __getitem__(self, key):
        return self.cpp_dict.__getitem__(key)


class ModuleWrapper(nn.Module):
    """A subclass of ``torch.nn.Module`` that wraps a C++ frontend module and delegates all access."""

    def __init__(self, cpp_module) -> None:
        # Assign before the super class constructor so ``self.training`` can be
        # assigned to in the super class constructor.
        self.cpp_module = cpp_module
        super().__init__()
        self._parameters = OrderedDictWrapper(cpp_module, "_parameters")  # type: ignore[assignment]
        self._buffers: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_buffers")  # type: ignore[assignment]
        self._modules: OrderedDictWrapper = OrderedDictWrapper(cpp_module, "_modules")  # type: ignore[assignment]
        for attr in dir(cpp_module):
            # Skip magic methods and the three attributes above.
            if not attr.startswith("_"):
                setattr(self, attr, getattr(self.cpp_module, attr))

    def _apply(self, fn, recurse=True):
        for param in self.parameters():
            # Tensors stored in modules are graph leaves, and we don't
            # want to create copy nodes, so we have to unpack the data.
            param.data = fn(param.data)
            if param._grad is not None:
                param._grad.data = fn(param._grad.data)

        for buf in self.buffers():
            buf.data = fn(buf.data)

        return self

    # nn.Module defines training as a boolean
    @property  # type: ignore[override]
    # pyrefly: ignore [bad-override]
    def training(self):
        return self.cpp_module.training

    @training.setter
    def training(self, mode) -> None:
        self.cpp_module.train(mode)

    def __repr__(self) -> str:
        return self.cpp_module.__repr__()
<|endoftext|># mypy: allow-untyped-defs
"""Gradient interface."""

import torch
from torch.nn.modules.utils import _pair, _single, _triple


def conv1d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv1d with respect to the input of the convolution.

    This is same as the 1D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv1d_input(input.shape, weight, grad_output)

    """
    input = grad_output.new_empty(1).expand(input_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _single(stride),
        _single(padding),
        _single(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


def conv1d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv1d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> # xdoctest: +SKIP
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv1d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _single(stride),
        _single(padding),
        _single(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]


def conv2d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv2d with respect to the input of the convolution.

    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv2d_input(input.shape, weight, grad_output)

    """
    input = grad_output.new_empty(1).expand(input_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


def conv2d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv2d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> # xdoctest: +SKIP
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]


def conv3d_input(
    input_size,
    weight,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv3d with respect to the input of the convolution.

    This is same as the 3D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weights tensor (out_channels x in_channels/groups x kT x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv3d_input(input.shape, weight, grad_output)

    """
    input = grad_output.new_empty(1).expand(input_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _triple(stride),
        _triple(padding),
        _triple(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


def conv3d_weight(
    input,
    weight_size,
    grad_output,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    r"""Compute the gradient of conv3d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, weight, grad_output)
        >>> F.grad.conv3d_weight(input, weight.shape, grad_output)

    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _triple(stride),
        _triple(padding),
        _triple(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]
<|endoftext|># mypy: allow-untyped-defs
from torch.nn.parameter import (  # usort: skip
    Buffer as Buffer,
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)
from torch.nn.modules import *  # usort: skip # noqa: F403
from torch.nn import (
    attention as attention,
    functional as functional,
    init as init,
    modules as modules,
    parallel as parallel,
    parameter as parameter,
    utils as utils,
)
from torch.nn.parallel import DataParallel as DataParallel


def factory_kwargs(kwargs):
    r"""Return a canonicalized dict of factory kwargs.

    Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed
    to factory functions like torch.empty, or errors if unrecognized kwargs are present.

    This function makes it simple to write code like this::

        class MyModule(nn.Module):
            def __init__(self, **kwargs):
                factory_kwargs = torch.nn.factory_kwargs(kwargs)
                self.weight = Parameter(torch.empty(10, **factory_kwargs))

    Why should you use this function instead of just passing `kwargs` along directly?

    1. This function does error validation, so if there are unexpected kwargs we will
    immediately report an error, instead of deferring it to the factory call
    2. This function supports a special `factory_kwargs` argument, which can be used to
    explicitly specify a kwarg to be used for factory functions, in the event one of the
    factory kwargs conflicts with an already existing argument in the signature (e.g.
    in the signature ``def f(dtype, **kwargs)``, you can specify ``dtype`` for factory
    functions, as distinct from the dtype argument, by saying
    ``f(dtype1, factory_kwargs={"dtype": dtype2})``)
    """
    if kwargs is None:
        return {}
    simple_keys = {"device", "dtype", "memory_format"}
    expected_keys = simple_keys | {"factory_kwargs"}
    if not kwargs.keys() <= expected_keys:
        raise TypeError(f"unexpected kwargs {kwargs.keys() - expected_keys}")

    # guarantee no input kwargs is untouched
    r = dict(kwargs.get("factory_kwargs", {}))
    for k in simple_keys:
        if k in kwargs:
            if k in r:
                raise TypeError(
                    f"{k} specified twice, in **kwargs and in factory_kwargs"
                )
            r[k] = kwargs[k]

    return r
<|endoftext|>import warnings


# NB: Keep this file in sync with enums in aten/src/ATen/core/Reduction.h


def get_enum(reduction: str) -> int:
    if reduction == "none":
        ret = 0
    elif reduction == "mean":
        ret = 1
    elif reduction == "elementwise_mean":
        warnings.warn(
            "reduction='elementwise_mean' is deprecated. "
            "Please use reduction='mean' instead.",
            stacklevel=2,
        )
        ret = 1
    elif reduction == "sum":
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return ret


# In order to support previous versions, accept boolean size_average and reduce
# and convert them into the new constants for now


# We use these functions in torch/legacy as well, in which case we'll silence the warning
def legacy_get_string(
    size_average: bool | None,
    reduce: bool | None,
    emit_warning: bool = True,
) -> str:
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    if emit_warning:
        warnings.warn(warning.format(ret), stacklevel=2)
    return ret


def legacy_get_enum(
    size_average: bool | None,
    reduce: bool | None,
    emit_warning: bool = True,
) -> int:
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))
<|endoftext|>from torch.nn.quantized import dynamic, functional, modules
from torch.nn.quantized.modules import *  # noqa: F403
from torch.nn.quantized.modules import MaxPool2d


__all__ = [
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "Dropout",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LSTM",
    "MultiheadAttention",
    "PReLU",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    # Wrapper modules
    "FloatFunctional",
    "FXFloatFunctional",
    "QFunctional",
]
<|endoftext|>r"""Quantized Modules.

Note::
    The `torch.nn.quantized` namespace is in the process of being deprecated.
    Please, use `torch.ao.nn.quantized` instead.
"""

# The following imports are needed in case the user decides
# to import the files directly,
# s.a. `from torch.nn.quantized.modules.conv import ...`.
# No need to add them to the `__all__`.
from torch.ao.nn.quantized.modules import (
    activation,
    batchnorm,
    conv,
    DeQuantize,
    dropout,
    embedding_ops,
    functional_modules,
    linear,
    MaxPool2d,
    normalization,
    Quantize,
    rnn,
    utils,
)
from torch.ao.nn.quantized.modules.activation import (
    ELU,
    Hardswish,
    LeakyReLU,
    MultiheadAttention,
    PReLU,
    ReLU6,
    Sigmoid,
    Softmax,
)
from torch.ao.nn.quantized.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.ao.nn.quantized.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from torch.ao.nn.quantized.modules.dropout import Dropout
from torch.ao.nn.quantized.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.quantized.modules.functional_modules import (
    FloatFunctional,
    FXFloatFunctional,
    QFunctional,
)
from torch.ao.nn.quantized.modules.linear import Linear
from torch.ao.nn.quantized.modules.normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from torch.ao.nn.quantized.modules.rnn import LSTM


__all__ = [
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LSTM",
    "MultiheadAttention",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Dropout",
    "PReLU",
    # Wrapper modules
    "FloatFunctional",
    "FXFloatFunctional",
    "QFunctional",
]
<|endoftext|>r"""Quantized Reference Modules.

This module is in the process of migration to
`torch/ao/nn/quantized/reference`, and is kept here for
compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/reference`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.reference.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from torch.ao.nn.quantized.reference.modules.linear import Linear
from torch.ao.nn.quantized.reference.modules.rnn import GRUCell, LSTM, LSTMCell, RNNCell
from torch.ao.nn.quantized.reference.modules.sparse import Embedding, EmbeddingBag


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "LSTM",
    "Embedding",
    "EmbeddingBag",
]
<|endoftext|>r"""Quantized Dynamic Modules.

This file is in the process of migration to `torch/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/dynamic`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.dynamic.modules import conv, linear, rnn
from torch.ao.nn.quantized.dynamic.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from torch.ao.nn.quantized.dynamic.modules.linear import Linear
from torch.ao.nn.quantized.dynamic.modules.rnn import (
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
    RNNCell,
)


__all__ = [
    "Linear",
    "LSTM",
    "GRU",
    "LSTMCell",
    "RNNCell",
    "GRUCell",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]
<|endoftext|>import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = ["PairwiseDistance", "CosineSimilarity"]


class PairwiseDistance(Module):
    r"""
    Computes the pairwise distance between input vectors, or between columns of input matrices.

    Distances are computed using ``p``-norm, with constant ``eps`` added to avoid division by zero
    if ``p`` is negative, i.e.:

    .. math ::
        \mathrm{dist}\left(x, y\right) = \left\Vert x-y + \epsilon e \right\Vert_p,

    where :math:`e` is the vector of ones and the ``p``-norm is given by.

    .. math ::
        \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.

    Args:
        p (real, optional): the norm degree. Can be negative. Default: 2
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-6
        keepdim (bool, optional): Determines whether or not to keep the vector dimension.
            Default: False
    Shape:
        - Input1: :math:`(N, D)` or :math:`(D)` where `N = batch dimension` and `D = vector dimension`
        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as the Input1
        - Output: :math:`(N)` or :math:`()` based on input dimension.
          If :attr:`keepdim` is ``True``, then :math:`(N, 1)` or :math:`(1)` based on input dimension.

    Examples:
        >>> pdist = nn.PairwiseDistance(p=2)
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> output = pdist(input1, input2)
    """

    __constants__ = ["norm", "eps", "keepdim"]
    norm: float
    eps: float
    keepdim: bool

    def __init__(
        self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False
    ) -> None:
        super().__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.pairwise_distance(x1, x2, self.norm, self.eps, self.keepdim)


class CosineSimilarity(Module):
    r"""Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same number of dimensions as x1, matching x1 size at dimension `dim`,
          and broadcastable with x1 at other dimensions.
        - Output: :math:`(\ast_1, \ast_2)`

    Examples:
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """

    __constants__ = ["dim", "eps"]
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.cosine_similarity(x1, x2, self.dim, self.eps)
<|endoftext|># mypy: allow-untyped-defs

import itertools
from collections import namedtuple
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .container import ModuleList, Sequential
from .linear import Linear
from .module import Module


__all__ = ["AdaptiveLogSoftmaxWithLoss"]

_ASMoutput = namedtuple("_ASMoutput", ["output", "loss"])


class AdaptiveLogSoftmaxWithLoss(Module):
    (
        """Efficient softmax approximation.

    As described in
    `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
    Moustapha Ciss\u00e9, David Grangier, and Herv\u00e9 J\u00e9gou
    <https://arxiv.org/abs/1609.04309>`__.
"""
        r"""
    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containing less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are evaluated.

    The idea is that the clusters which are accessed frequently
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned labels.

    We highly recommend taking a look at the original paper for more details.

    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\left\lfloor\frac{\texttt{in\_features}}{\texttt{div\_value}^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      adaptive softmax. See paper for details. Set to False in the official
      implementation.

    .. warning::
        Labels passed as inputs to this module should be sorted according to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.

    .. note::
        This module returns a ``NamedTuple`` with ``output``
        and ``loss`` fields. See further documentation for details.

    .. note::
        To compute log-probabilities for all classes, the ``log_prob``
        method can be used.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            adaptive softmax. Default: ``False``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    Shape:
        - input: :math:`(N, \texttt{in\_features})` or :math:`(\texttt{in\_features})`
        - target: :math:`(N)` or :math:`()` where each value satisfies :math:`0 <= \texttt{target[i]} <= \texttt{n\_classes}`
        - output1: :math:`(N)` or :math:`()`
        - output2: ``Scalar``

    .. _Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law
    """
    )

    in_features: int
    n_classes: int
    cutoffs: list[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = 4.0,
        head_bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        cutoffs = list(cutoffs)

        if len(cutoffs) == 0:
            raise ValueError("cutoffs should be a sequence of length larger than 0")

        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) > (n_classes - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any(int(c) != c for c in cutoffs)
        ):
            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1"
            )

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(
            self.in_features, self.head_size, bias=self.head_bias, **factory_kwargs
        )
        self.tail = ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False, **factory_kwargs),
                Linear(hsz, osz, bias=False, **factory_kwargs),
            )

            self.tail.append(projection)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        self.head.reset_parameters()
        for i2h, h2o in self.tail:  # type: ignore[misc]
            i2h.reset_parameters()  # type: ignore[has-type]
            h2o.reset_parameters()  # type: ignore[has-type]

    def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput:
        """
        Runs the forward pass.
        """
        targ_dim = target_.dim()

        if targ_dim == 1:
            if input_.size(0) != target_.size(0):
                raise RuntimeError(
                    "Input and target should have the same size in the batch dimension."
                )
            if input_.dim() != 2:
                raise RuntimeError(
                    f"1D target tensor expects 2D input tensors, "
                    f"but found inputs with size {input_.size()}"
                )
        elif targ_dim == 0:
            if input_.dim() != 1:
                raise RuntimeError(
                    f"0D target tensor expects 1D input tensors, "
                    f"but found inputs with size {input_.size()}"
                )
        else:
            raise RuntimeError(
                "0D or 1D target tensor expected, multi-target not supported"
            )

        is_batched = targ_dim > 0
        input = input_ if is_batched else input_.unsqueeze(0)
        target = target_ if is_batched else target_.unsqueeze(0)

        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)
                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError(
                f"Target values should be in [0, {self.n_classes - 1}], "
                f"but values in range [{target.min().item()}, {target.max().item()}] "
                "were found. "
            )

        head_output = self.head(input)
        head_logprob = F.log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean()

        if not is_batched:
            output = output.squeeze(0)

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, input, head_output):
        """Given input tensor, and output of ``self.head``, compute the log of the full distribution."""
        out = input.new_empty((head_output.size(0), self.n_classes))
        head_logprob = F.log_softmax(head_output, dim=1)

        out[:, : self.shortlist_size] = head_logprob[:, : self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(itertools.pairwise(self.cutoffs)):
            cluster_output = self.tail[i](input)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + head_logprob[
                :, self.shortlist_size + i
            ].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        r"""Compute log probabilities for all :math:`\texttt{n\_classes}`.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """
        head_output = self.head(input)
        return self._get_full_log_prob(input, head_output)

    def predict(self, input: Tensor) -> Tensor:
        r"""Return the class with the highest probability for each example in the input minibatch.

        This is equivalent to ``self.log_prob(input).argmax(dim=1)``, but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """
        head_output = self.head(input)
        output = torch.argmax(head_output, dim=1)
        not_in_shortlist = output >= self.shortlist_size
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            return torch.argmax(log_prob, dim=1)

        else:
            log_prob = self._get_full_log_prob(
                input[not_in_shortlist], head_output[not_in_shortlist]
            )
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return output
<|endoftext|>import dataclasses
from functools import cached_property

import torch


__all__ = []


def _make_empty(shape, dtype, device, when=True):
    # when=False returns a rank-matching empty (shape (0,)*len(shape)).
    return torch.empty(
        shape if when else (0,) * len(shape),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )


def _make_zeros(shape, dtype, device, when=True):
    return torch.zeros(
        shape if when else (0,) * len(shape),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )


def _linear_cross_entropy_batch_chunked_setup_context(ctx, inputs, output):
    (
        *_,
        allow_retain_graph,
        compute_input_grad,
        compute_linear_weight_grad,
        compute_linear_bias_grad,
    ) = inputs
    # Without this, backward through the loss materializes zero-filled
    # gradients for the three unused grad outputs -- (N, F) + (C, F) + (C,)
    # at input dtype -- silently adding an input+weight-sized transient to
    # every backward peak. The backward only consumes the loss gradient.
    ctx.set_materialize_grads(False)
    ctx.allow_retain_graph = allow_retain_graph
    ctx.compute_input_grad = compute_input_grad
    ctx.compute_linear_weight_grad = compute_linear_weight_grad
    ctx.compute_linear_bias_grad = compute_linear_bias_grad
    _, grad_input, grad_linear_weight, grad_linear_bias = output
    # ``.detach()`` the stashed grads: they are forward OUTPUTS, so they
    # carry ``grad_fn`` back to this custom-op node. Holding them as raw
    # ctx attributes would form a reference cycle (node -> ctx -> grad ->
    # grad_fn -> node) that refcounting can't break, leaking the tensors
    # until a GC pass -- which the CUDA mem-leak check flags. Detaching
    # drops the grad_fn link so the tensors free promptly when the loss's
    # graph is. Backward only multiplies them by the upstream grad, so the
    # lost autograd history is irrelevant (double-backward is unsupported).
    ctx._gi = grad_input.detach() if compute_input_grad else None
    ctx._gw = grad_linear_weight.detach() if compute_linear_weight_grad else None
    ctx._gb = grad_linear_bias.detach() if compute_linear_bias_grad else None


@dataclasses.dataclass
class _ChunkViews:
    """Per-iteration tensor views; each property picks the right operand
    or scratch from ctx dispatch flags so call sites read raw.
    """

    ctx: "_ChunkContext"
    bchunk_start: int
    bchunk_size: int
    input_chunk: torch.Tensor
    target_chunk: torch.Tensor
    weight_chunk: torch.Tensor | None  # None on the prob path (see prob_wt)
    logits: torch.Tensor
    input_chunk_acc: torch.Tensor

    @cached_property
    def prob_wt(self) -> torch.Tensor:
        # Prob path: (B, V) weight*target at the logits dtype -- the dense
        # analogue of the index path's per-row ``weight_chunk``. A view of
        # ``target_chunk`` when no scratch is needed.
        ctx = self.ctx
        if not ctx.alloc_prob_target_buf:
            return self.target_chunk
        out = ctx.prob_target_buf.narrow(0, 0, self.bchunk_size)
        if ctx.weight_row is not None:
            torch.mul(self.target_chunk, ctx.weight_row, out=out)
        else:
            out.copy_(self.target_chunk)
        return out

    @property
    def input(self) -> torch.Tensor:
        return (
            self.input_chunk_acc
            if self.ctx.forward_uses_acc_input
            else self.input_chunk
        )

    @property
    def linear_weight(self) -> torch.Tensor:
        ctx = self.ctx
        return (
            ctx.linear_weight
            if ctx.forward_uses_cuda_out_dtype
            else ctx.linear_weight_cast
        )

    @property
    def input_grad_logits(self) -> torch.Tensor:
        if self.ctx.input_grad_uses_logits_lw:
            return self.logits_downcast
        return self.logits

    @property
    def input_grad_linear_weight(self) -> torch.Tensor:
        ctx = self.ctx
        return (
            ctx.linear_weight
            if ctx.input_grad_uses_logits_lw
            else ctx.linear_weight_cast
        )

    @property
    def weight_grad_input(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.is_cuda and not ctx.weight_grad_mm_same_dtype:
            return self.input_chunk
        return self.input_chunk_acc

    @property
    def logits_upcast(self) -> torch.Tensor:
        # Non-CUDA mixed-dtype: copy into logits_acc_buf for matching-dtype mm.
        ctx = self.ctx
        if ctx.is_cuda or ctx.weight_grad_mm_same_dtype:
            return self.logits
        return ctx.logits_acc_buf.narrow(0, 0, self.bchunk_size).copy_(self.logits)

    @cached_property
    def grad_input_chunk(self) -> torch.Tensor:
        # Buf-and-copy commit happens in ctx.chunks() post-yield.
        ctx = self.ctx
        if ctx.alloc_input_grad_acc_buf:
            return ctx.input_grad_acc_buf.narrow(0, 0, self.bchunk_size)
        return ctx.grad_input.narrow(0, self.bchunk_start, self.bchunk_size)

    @cached_property
    def bias_grad_acc(self) -> torch.Tensor:
        # ``ctx.linear_bias_grad_chunk`` is an acc_dtype scratch that
        # accumulates the per-chunk sum + index_add at acc precision,
        # then commits to ``grad_linear_bias`` (input dtype) via the
        # ``ctx.chunks()`` post-yield hook -- mirrors
        # ``weight_grad_chunk``'s bulk-+-correction pattern for
        # ``grad_linear_weight``. On the non-acc path, accumulation
        # goes directly into ``grad_linear_bias`` (no commit, no
        # post-yield action).
        ctx = self.ctx
        if ctx.alloc_linear_bias_grad_chunk:
            return ctx.linear_bias_grad_chunk
        return ctx.grad_linear_bias

    @cached_property
    def logits_downcast(self) -> torch.Tensor:
        # Lazy: first access must come AFTER in-place mods to self.logits.
        ctx = self.ctx
        if ctx.loop_caches_logits_downcast:
            return self.logits.to(ctx.linear_weight.dtype)
        return self.logits


@dataclasses.dataclass
class _ChunkContext:
    """Per-call state for the chunked loop, built once via ``build``.
    Methods hide dtype/device/acc_policy dispatch behind single math ops;
    dispatch-free per-iter math is inlined into the loop body. Buffers
    dispatch decided are not needed are rank-matching empty tensors
    (``when=False`` in ``_make_*``) so the dataclass surface stays uniform.
    """

    dtype: torch.dtype
    num_batches: int
    in_features: int
    num_classes: int
    is_cuda: bool
    use_acc_dtype: bool

    acc_dtype: torch.dtype
    weight_chunk_dtype: torch.dtype
    grad_input_dtype: torch.dtype
    linear_weight_cast_dtype: torch.dtype

    input: torch.Tensor
    target: torch.Tensor  # uncorrected; per-iter slice is from corrected_target
    weight: torch.Tensor | None
    ignore_index: int
    reduction: str
    # Probability (soft-label) target: ``target`` is a floating ``(N, V)``
    # tensor of per-class probabilities. ``loss_scale`` is the positive
    # scalar reduction factor (1/N for "mean" -- the prob-target mean
    # divisor is the row count, not the weight mass -- and 1 for "sum").
    is_prob_target: bool
    loss_scale: float
    linear_weight: torch.Tensor
    linear_bias: torch.Tensor | None
    # reduction='none' backward only: per-sample upstream grad (N,). When
    # set, neg_weight_target folds it in so the shared grad loop produces
    # the grad_output-weighted VJP. None on every other path.
    loss_grad_output: torch.Tensor | None

    # The optional "when=" buffers (weight_grad_chunk, logits_acc_buf,
    # input_grad_acc_buf, input_chunk_acc_buf, grad_input,
    # grad_linear_weight, grad_linear_bias) are cached_properties below.
    logits_buf: torch.Tensor
    tmp: torch.Tensor
    output: torch.Tensor

    compute_input_grad: bool
    compute_linear_weight_grad: bool
    compute_linear_bias_grad: bool
    alloc_prob_target_buf: bool
    alloc_weight_grad_chunk: bool
    alloc_linear_bias_grad_chunk: bool
    alloc_input_grad_acc_buf: bool
    alloc_input_chunk_acc_buf: bool
    alloc_logits_acc_buf: bool
    forward_uses_acc_input: bool
    forward_uses_cuda_out_dtype: bool
    weight_grad_mm_same_dtype: bool
    input_grad_uses_logits_lw: bool
    loop_caches_logits_downcast: bool
    weight_grad_uses_logits_buf_temp: bool

    @cached_property
    def _mask(self) -> torch.Tensor:
        return self.target == self.ignore_index

    @cached_property
    def corrected_target(self) -> torch.Tensor:
        # Replace out-of-range ignore_index values with 0 lazily so
        # downstream index_select / index_add_ stay in bounds.
        if self.is_prob_target:
            # ignore_index does not apply to probability targets.
            return self.target
        if self.ignore_index < 0 or self.ignore_index >= self.num_classes:
            return torch.where(self._mask, 0, self.target)
        return self.target

    @cached_property
    def neg_weight_target(self) -> torch.Tensor:
        # Per-row weighting: -(weight[target] if weight else 1) / d on
        # unmasked positions, 0 on masked positions. d = sum of unmasked
        # weights for "mean", 1 for "sum".
        mask = self._mask
        target = self.corrected_target
        weight = self.weight
        weight_chunk_dtype = self.weight_chunk_dtype
        if weight is None:
            neg_weight_target = (~mask).to(weight_chunk_dtype)
        elif target.numel() > weight.numel():
            neg_weight_target = torch.where(
                mask, 0, weight.to(weight_chunk_dtype).index_select(0, target)
            )
        else:
            neg_weight_target = torch.where(
                mask, 0, weight.index_select(0, target).to(weight_chunk_dtype)
            )
        if self.reduction == "mean":
            d = neg_weight_target.sum()
            neg_weight_target.div_(torch.where(d == 0, torch.nan, -d))
        elif self.reduction == "sum":
            neg_weight_target.neg_()
        else:  # "none"
            # Forward (loss_grad_output is None) intentionally leaves the
            # bare +W[T] unsigned: the loss-only branch needs the positive
            # per-row loss weight. Only the backward (upstream grad set)
            # negates and folds in grad_output[n] for the VJP -- do NOT add
            # a .neg_() here to mirror the "sum" case.
            if self.loss_grad_output is not None:
                neg_weight_target.neg_().mul_(
                    self.loss_grad_output.to(neg_weight_target.dtype)
                )
        return neg_weight_target

    @cached_property
    def linear_weight_cast(self) -> torch.Tensor:
        if self.linear_weight_cast_dtype != self.dtype:
            return self.linear_weight.to(self.linear_weight_cast_dtype)
        return self.linear_weight

    @cached_property
    def weight_row(self) -> torch.Tensor | None:
        # (1, V) class-weight row at the logits dtype, broadcast against
        # each (B, V) target chunk on the prob path. None when no class
        # weight was passed.
        if self.weight is None:
            return None
        weight = self.weight
        if weight.dtype != self.logits_buf.dtype:
            weight = weight.to(self.logits_buf.dtype)
        return weight.unsqueeze(0)

    @cached_property
    def prob_target_buf(self) -> torch.Tensor:
        # (B, V) scratch at the logits dtype holding weight*target per
        # chunk on the prob path; rank-empty when target_chunk is usable
        # directly (no class weight, matching dtype).
        return _make_empty(
            self.logits_buf.shape,
            self.logits_buf.dtype,
            self.input.device,
            when=self.alloc_prob_target_buf,
        )

    @cached_property
    def linear_bias_cast(self) -> torch.Tensor | None:
        # ``linear_bias`` is added to per-chunk logits, so its dtype
        # must match the logits buffer (which may be acc_dtype on the
        # upcast path). ``None`` when the user passed no bias -- the
        # loop skips the add.
        if self.linear_bias is None:
            return None
        if self.linear_bias.dtype != self.logits_buf.dtype:
            return self.linear_bias.to(self.logits_buf.dtype)
        return self.linear_bias

    @cached_property
    def weight_grad_chunk(self) -> torch.Tensor:
        # (V, F) acc_dtype accumulator for keep-path weight-grad mm;
        # rank-empty on the direct (compact) path.
        return _make_empty(
            (self.num_classes, self.in_features),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_weight_grad_chunk,
        )

    @cached_property
    def logits_acc_buf(self) -> torch.Tensor:
        # (B, V) scratch for non-CUDA mixed-dtype logits_upcast.
        return _make_empty(
            (self.logits_buf.shape[0], self.num_classes),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_logits_acc_buf,
        )

    @cached_property
    def input_grad_acc_buf(self) -> torch.Tensor:
        # (B, F) scratch for input-grad addmm on the buf-and-copy path
        # (CPU mixed-dtype, MPS).
        return _make_empty(
            (self.logits_buf.shape[0], self.in_features),
            self.linear_weight_cast_dtype,
            self.input.device,
            when=self.alloc_input_grad_acc_buf,
        )

    @cached_property
    def input_chunk_acc_buf(self) -> torch.Tensor:
        return _make_empty(
            (self.logits_buf.shape[0], self.in_features),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_input_chunk_acc_buf,
        )

    @cached_property
    def grad_input(self) -> torch.Tensor:
        return _make_empty(
            self.input.shape,
            self.grad_input_dtype,
            self.input.device,
            when=self.compute_input_grad,
        )

    @cached_property
    def grad_linear_weight(self) -> torch.Tensor:
        return _make_zeros(
            self.linear_weight.shape,
            self.dtype,
            self.input.device,
            when=self.compute_linear_weight_grad,
        )

    @cached_property
    def grad_linear_bias(self) -> torch.Tensor:
        # Same shape invariant as ``linear_bias``:
        # ``linear_bias.shape == linear_weight.shape[:-1]``. Final
        # accumulator in input dtype to match ``grad_linear_weight``.
        return _make_zeros(
            self.linear_weight.shape[:-1],
            self.dtype,
            self.input.device,
            when=self.compute_linear_bias_grad,
        )

    @cached_property
    def linear_bias_grad_chunk(self) -> torch.Tensor:
        # ``linear_weight.shape[:-1]`` acc_dtype per-chunk scratch,
        # mirroring ``weight_grad_chunk``'s role for
        # ``grad_linear_weight``: the sub_ + index_add_ pair
        # accumulates per-chunk in acc_dtype, then a single downcast
        # commit (in ``chunks()`` post-yield) lands in
        # ``grad_linear_bias``. ``_make_zeros`` so the first chunk's
        # accumulators start at zero; the post-yield hook zeroes it
        # again for subsequent chunks.
        return _make_zeros(
            self.linear_weight.shape[:-1],
            self.acc_dtype,
            self.input.device,
            when=self.alloc_linear_bias_grad_chunk,
        )

    @classmethod
    def build(
        cls,
        input: torch.Tensor,
        linear_weight: torch.Tensor,
        target: torch.Tensor,
        linear_bias: torch.Tensor | None,
        weight: torch.Tensor | None,
        reduction: str,
        ignore_index: int,
        label_smoothing: float,
        batch_chunk_size: int,
        acc_policy: str,
        acc_dtype: torch.dtype,
        compute_input_grad: bool,
        compute_linear_weight_grad: bool,
        compute_linear_bias_grad: bool,
        loss_grad_output: torch.Tensor | None = None,
    ) -> "_ChunkContext":
        # ===== Validation =====
        is_prob_target = target.dtype.is_floating_point
        if is_prob_target:
            if target.dtype != input.dtype:
                raise TypeError(
                    "linear_cross_entropy: probability target dtype must match"
                    f" input dtype {input.dtype}, got {target.dtype}."
                )
            if target.shape != (input.shape[0], linear_weight.shape[0]):
                raise RuntimeError(
                    "linear_cross_entropy: probability target shape must be "
                    f"{(input.shape[0], linear_weight.shape[0])}, got {tuple(target.shape)}."
                )
            if reduction not in {"mean", "sum"}:
                raise NotImplementedError(
                    "linear_cross_entropy: probability target requires"
                    f" reduction in {{'mean', 'sum'}}, got {reduction!r}"
                )
        elif target.dtype != torch.int64:
            raise TypeError(
                f"linear_cross_entropy: target dtype must be torch.int64, got {target.dtype}."
            )
        if label_smoothing > 0.0:
            raise NotImplementedError(
                "linear_cross_entropy does not support label smoothing"
            )
        if reduction not in {"mean", "sum", "none"}:
            raise NotImplementedError(
                f"linear_cross_entropy does not support {reduction=}"
            )
        if linear_bias is not None and linear_bias.shape != linear_weight.shape[:-1]:
            raise RuntimeError(
                "linear_cross_entropy: expected linear_bias shape "
                f"{tuple(linear_weight.shape[:-1])}, got {tuple(linear_bias.shape)}."
            )

        device = input.device
        dtype = input.dtype
        num_batches, in_features = input.shape
        num_classes, _ = linear_weight.shape
        # CUDA gates the out_dtype= mm fast path; the non-CUDA path
        # (CPU, MPS, XPU, ...) routes mixed-dtype mm through explicit
        # casts until out_dtype= is validated on those backends.
        is_cuda = device.type == "cuda"
        is_mps = device.type == "mps"

        if dtype != acc_dtype and not (
            dtype in {torch.float16, torch.bfloat16} and acc_dtype == torch.float32
        ):
            raise RuntimeError(
                "linear_cross_entropy supports float32 acc_dtype with"
                f" float16/bfloat16 inputs, but got {acc_dtype} acc_dtype and {dtype} inputs."
            )
        use_acc_dtype = dtype != acc_dtype

        # Internal dtype layout shared by the memory-like policy. ``compact``'s
        # savings come from skipping the weight_grad_chunk scratch, not dtype.
        is_memory_like = acc_policy == "compact"
        if use_acc_dtype:
            output_dtype = acc_dtype if dtype == torch.float16 else dtype
            grad_input_dtype = dtype if is_memory_like else acc_dtype
            # fp16 + memory-like keeps logits at fp16 (2BC vs 4BC);
            # everything else upcasts for softmax stability.
            logits_buf_dtype = (
                dtype if dtype == torch.float16 and is_memory_like else acc_dtype
            )
            # acc_dtype mandatory: 1/N for reduction="mean" is subnormal
            # in fp16 at N >= 65536.
            weight_chunk_dtype = acc_dtype
        else:
            output_dtype = grad_input_dtype = logits_buf_dtype = weight_chunk_dtype = (
                dtype
            )

        # ===== Dispatch flags =====
        # CUDA + compact uses cuBLAS out_dtype= directly; no cast.
        needs_linear_weight_cast = use_acc_dtype and (
            not is_cuda or (compute_input_grad and grad_input_dtype == logits_buf_dtype)
        )
        linear_weight_cast_dtype = (
            logits_buf_dtype if needs_linear_weight_cast else dtype
        )
        alloc_weight_grad_chunk = compute_linear_weight_grad and not (
            acc_policy == "compact" and (is_cuda or logits_buf_dtype == dtype)
        )
        # Per-chunk acc_dtype scratch for grad_linear_bias; same
        # bulk-+-correction precision rationale as
        # ``alloc_weight_grad_chunk``. Fires only when there is
        # precision to gain: ``use_acc_dtype`` and the user asked for
        # the bias gradient.
        alloc_linear_bias_grad_chunk = compute_linear_bias_grad and use_acc_dtype
        alloc_input_grad_acc_buf = (
            compute_input_grad
            and not is_cuda
            and (grad_input_dtype != linear_weight_cast_dtype or is_mps)
        )
        alloc_input_chunk_acc_buf = use_acc_dtype and (
            compute_linear_weight_grad or (not is_cuda and dtype != logits_buf_dtype)
        )
        forward_uses_acc_input = (
            use_acc_dtype and not is_cuda and dtype != logits_buf_dtype
        )
        forward_uses_cuda_out_dtype = is_cuda and use_acc_dtype
        weight_grad_mm_same_dtype = logits_buf_dtype == acc_dtype
        # Storage-trick fires when logits' dtype differs from the input-
        # grad accumulator dtype.
        input_grad_uses_logits_lw = (
            is_cuda and compute_input_grad and logits_buf_dtype != grad_input_dtype
        )
        # Per-iter ``logits.to(linear_weight.dtype)`` shared between the
        # inlined input-grad addmm and the inlined direct weight-grad addmm_.
        loop_caches_logits_downcast = input_grad_uses_logits_lw or (
            compute_linear_weight_grad
            and not alloc_weight_grad_chunk
            and logits_buf_dtype != dtype
        )
        alloc_logits_acc_buf = (
            alloc_weight_grad_chunk and not is_cuda and logits_buf_dtype != acc_dtype
        )
        # Direct weight-grad path: reuse logits_buf storage for the
        # acc_dtype->dtype cast when its bytes/row fits the (B, F) cast.
        weight_grad_uses_logits_buf_temp = (
            not alloc_weight_grad_chunk
            and use_acc_dtype
            and num_classes * logits_buf_dtype.itemsize >= in_features * dtype.itemsize
        )
        # Prob path: per-chunk (B, V) scratch for weight*target at the
        # logits dtype. Skipped when target_chunk can be used directly
        # (no class weight and dtype already matches).
        alloc_prob_target_buf = is_prob_target and (
            weight is not None or dtype != logits_buf_dtype
        )
        if is_prob_target and reduction == "mean":
            loss_scale = 1.0 / num_batches if num_batches else float("nan")
        else:
            loss_scale = 1.0

        return cls(
            dtype=dtype,
            num_batches=num_batches,
            in_features=in_features,
            num_classes=num_classes,
            is_cuda=is_cuda,
            use_acc_dtype=use_acc_dtype,
            acc_dtype=acc_dtype,
            weight_chunk_dtype=weight_chunk_dtype,
            grad_input_dtype=grad_input_dtype,
            linear_weight_cast_dtype=linear_weight_cast_dtype,
            input=input,
            target=target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            is_prob_target=is_prob_target,
            loss_scale=loss_scale,
            linear_weight=linear_weight,
            linear_bias=linear_bias,
            loss_grad_output=loss_grad_output,
            logits_buf=torch.empty(
                (batch_chunk_size, num_classes),
                dtype=logits_buf_dtype,
                device=device,
                requires_grad=False,
            ),
            tmp=torch.empty(
                (batch_chunk_size,),
                dtype=logits_buf_dtype,
                device=device,
                requires_grad=False,
            ),
            output=_make_zeros((), output_dtype, device),
            compute_input_grad=compute_input_grad,
            compute_linear_weight_grad=compute_linear_weight_grad,
            compute_linear_bias_grad=compute_linear_bias_grad,
            alloc_prob_target_buf=alloc_prob_target_buf,
            alloc_weight_grad_chunk=alloc_weight_grad_chunk,
            alloc_linear_bias_grad_chunk=alloc_linear_bias_grad_chunk,
            alloc_input_grad_acc_buf=alloc_input_grad_acc_buf,
            alloc_input_chunk_acc_buf=alloc_input_chunk_acc_buf,
            alloc_logits_acc_buf=alloc_logits_acc_buf,
            forward_uses_acc_input=forward_uses_acc_input,
            forward_uses_cuda_out_dtype=forward_uses_cuda_out_dtype,
            weight_grad_mm_same_dtype=weight_grad_mm_same_dtype,
            input_grad_uses_logits_lw=input_grad_uses_logits_lw,
            loop_caches_logits_downcast=loop_caches_logits_downcast,
            weight_grad_uses_logits_buf_temp=weight_grad_uses_logits_buf_temp,
        )

    def chunks(self):
        """Yield a ``_ChunkViews`` per iter. Post-yield, run the
        scratch-to-final commits that the loop body deferred: the
        buf-and-copy input-grad slice into ``grad_input`` (skipped on
        the fast path where ``grad_input_chunk`` aliases
        ``grad_input``), and the acc_dtype linear_bias scratch into
        ``grad_linear_bias`` (skipped when accumulation went straight
        into ``grad_linear_bias`` in input dtype).
        """
        batch_chunk_size = self.logits_buf.shape[0]
        for bchunk_start in range(0, self.num_batches, batch_chunk_size):
            bchunk_size = min(batch_chunk_size, self.num_batches - bchunk_start)
            chunk = self.bind_chunk(bchunk_start, bchunk_size)
            yield chunk
            if self.alloc_input_grad_acc_buf:
                self.grad_input.narrow(0, bchunk_start, bchunk_size).copy_(
                    chunk.grad_input_chunk
                )
            if self.alloc_linear_bias_grad_chunk:
                # ``chunk.bias_grad_acc`` aliases ``linear_bias_grad_chunk``
                # here; downcast-commit, then zero for the next chunk.
                self.grad_linear_bias.add_(
                    chunk.bias_grad_acc.to(self.grad_linear_bias.dtype)
                )
                chunk.bias_grad_acc.zero_()

    def bind_chunk(self, bchunk_start: int, bchunk_size: int) -> _ChunkViews:
        input_chunk = self.input.narrow(0, bchunk_start, bchunk_size)
        target_chunk = self.corrected_target.narrow(0, bchunk_start, bchunk_size)
        # The prob loop derives its per-chunk row weighting from
        # ``prob_wt`` instead of the per-row ``neg_weight_target``.
        weight_chunk = (
            None
            if self.is_prob_target
            else self.neg_weight_target.narrow(0, bchunk_start, bchunk_size)
        )
        logits = self.logits_buf.narrow(0, 0, bchunk_size)
        input_chunk_acc = (
            self.input_chunk_acc_buf.narrow(0, 0, bchunk_size).copy_(input_chunk)
            if self.alloc_input_chunk_acc_buf
            else input_chunk
        )
        return _ChunkViews(
            ctx=self,
            bchunk_start=bchunk_start,
            bchunk_size=bchunk_size,
            input_chunk=input_chunk,
            target_chunk=target_chunk,
            weight_chunk=weight_chunk,
            logits=logits,
            input_chunk_acc=input_chunk_acc,
        )

    def mm(self, mat1: torch.Tensor, mat2: torch.Tensor, *, out: torch.Tensor) -> None:
        # Mismatched dtype: cuBLAS ``out_dtype=`` upcasts inside the matmul.
        if mat1.dtype == out.dtype:
            torch.mm(mat1, mat2, out=out)
        else:
            torch.mm(mat1, mat2, out_dtype=out.dtype, out=out)

    def shifted_logits(self, chunk: _ChunkViews) -> torch.Tensor:
        # Per-chunk forward prelude shared by all three chunk loops (none /
        # prob / index): logits = input @ weight.T (+ bias), shifted by the
        # per-row max for softmax stability. Writes into ``chunk.logits``.
        logits = chunk.logits
        self.mm(chunk.input, chunk.linear_weight.T, out=logits)
        if self.linear_bias_cast is not None:
            logits.add_(self.linear_bias_cast)
        # Subtract the per-row max in place; ``self.tmp`` is reused scratch.
        row_max = self.tmp.narrow(0, 0, logits.shape[0]).unsqueeze(1)
        torch.amax(logits, dim=1, keepdim=True, out=row_max)
        logits.sub_(row_max)
        return logits

    def dotgather(
        self, weight: torch.Tensor, x: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        return weight.dot(x.gather(1, indices.unsqueeze(1)).squeeze(1).to(weight.dtype))

    def sumexp_(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # The denom sums num_classes terms in (0, 1] (shifted logits <= 0).
        # fp16: accumulate in fp32 -- the sum overflows fp16's 65504 for a large
        # vocabulary (-> inf). bf16 stays in acc_dtype: it shares fp32's exponent
        # range so it cannot overflow, and computing the denom in fp32 vs bf16
        # was measured to move the bf16-stored log(denom) / (1/denom) by <=1 ULP,
        # so there is no gain to justify it (and it would re-calibrate bf16 caps).
        reduce_dtype = torch.float32 if self.dtype == torch.float16 else self.acc_dtype
        return x.exp_().sum(dim, dtype=reduce_dtype)

    def sum(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # Reduce ``x`` along ``dim`` at acc_dtype on mixed-precision
        # paths and at input dtype otherwise -- the precision
        # contract callers see is "as accurate as the active policy
        # allows". Currently used by the bias-grad path so the per-
        # chunk sum carries full precision before the post-yield
        # downcast commit to grad_linear_bias.
        target_dtype = self.acc_dtype if self.use_acc_dtype else self.dtype
        return x.sum(dim, dtype=target_dtype)

    def div(self, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        factor = self.tmp.narrow(0, 0, num.shape[0])
        # MPS lacks fused cross-dtype torch.div(out=); copy_ fallback.
        if self.input.device.type == "mps" and (
            num.dtype != factor.dtype or den.dtype != factor.dtype
        ):
            factor.copy_(num / den)
        else:
            torch.div(num, den, out=factor)
        return factor

    def mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Pick a destination from in-flight scratch when possible.
        if self.use_acc_dtype:
            return x.mul_(w.unsqueeze(1))
        if self.num_classes >= self.in_features:
            return torch.mul(
                x,
                w.unsqueeze(1),
                out=self.logits_buf.narrow(0, 0, x.shape[0]).narrow(
                    1, 0, self.in_features
                ),
            )
        return x * w.unsqueeze(1)

    def to(self, x: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        # Reuse logits_buf storage as cast destination when its bytes/row fits.
        if x.dtype == dtype:
            return x
        if self.weight_grad_uses_logits_buf_temp:
            x = (
                self.logits_buf.view(dtype)
                .narrow(0, 0, x.shape[0])
                .narrow(1, 0, x.shape[1])
                .copy_(x)
            )
            return x
        return x.to(dtype)


def _linear_cross_entropy_batch_chunked_accumulator(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
    compute_linear_bias_grad: bool,
    loss_grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked loop shared by all three custom-op entry points; returns
    ``(loss, grad_input, grad_linear_weight, grad_linear_bias)``.

    Caller -> branch:
      scalar-op forward (reduction in {mean,sum})  -> main grad loop
      scalar-op forward, probability target        -> prob loop
      no_reduction forward (none, loss_grad_output None) -> loss-only branch
      no_reduction backward (none, loss_grad_output set) -> main grad loop

    - ``_linear_cross_entropy_batch_chunked`` (scalar reduction): the
      forward precomputes the scalar loss and all gradients in one pass;
      backward just scales the stashed grads by the upstream gradient.
    - ``_linear_cross_entropy_batch_chunked_no_reduction`` (forward):
      ``reduction='none'`` with ``loss_grad_output is None`` takes the
      loss-only branch, returning the per-sample loss ``(N,)`` and no
      gradients (they are recomputed in its backward instead).
    - ``_linear_cross_entropy_batch_chunked_no_reduction`` (backward):
      ``loss_grad_output`` set folds the per-sample upstream grad into
      ``neg_weight_target``, so this same loop recomputes the chunked
      grads as the grad_output-weighted VJP (the scalar loss it also
      accumulates is then meaningless and ignored).

    Dispatch contract
    -----------------
    The chunked loop body is the mathematical description of the
    algorithm. All policy / device / dtype dispatch lives in
    ``_ChunkContext`` (and its ``_ChunkViews`` peers):

    - **Buffer choices** -- whether a per-chunk acc_dtype scratch is
      used or accumulation runs directly into the final tensor -- live
      in ``alloc_*`` flags computed in ``_ChunkContext.build`` and
      surfaced as ``_ChunkViews`` properties (e.g.
      ``grad_input_chunk``, ``bias_grad_acc``).
    - **Scratch-to-final commits** for those buffers live in
      ``_ChunkContext.chunks()`` post-yield hooks (e.g. the
      ``grad_input`` buf-and-copy commit, the ``grad_linear_bias``
      downcast-and-zero commit).
    - **Dtype dispatch on individual math operations** lives in
      ``_ChunkContext`` methods that mirror PyTorch ops --
      ``mm``, ``amax``, ``dotgather``, ``sumexp_``, ``sum``, ``div``,
      ``mul``, ``to`` -- each one hiding ``out_dtype=`` / acc-dtype /
      buffer-reuse decisions behind a single math call.

    The function body therefore should not introduce inline
    ``if ctx.use_acc_dtype`` / ``if ctx.is_cuda`` / etc. branches; new
    policy-aware behaviour belongs in one of the three locations
    above. The two existing inline branches (``if compute_input_grad``,
    ``if compute_linear_weight_grad``, etc.) gate optional outputs,
    not dispatch.
    """
    # Direct callers must resolve "auto" / None via _adjust first.
    if acc_policy == "auto" or acc_dtype is None:
        raise RuntimeError(
            f"unresolved acc_policy={acc_policy!r} or acc_dtype={acc_dtype!r};"
            " use F.linear_cross_entropy."
        )
    ctx = _ChunkContext.build(
        input,
        linear_weight,
        target,
        linear_bias,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad,
        compute_linear_weight_grad,
        compute_linear_bias_grad,
        loss_grad_output=loss_grad_output,
    )
    dtype = ctx.dtype
    # reduction='none' forward (no upstream grad yet): per-sample loss
    # into an (N,) output, no gradient precompute. ``chunk.weight_chunk``
    # is the unsigned masked class weight here (neg_weight_target's "none"
    # forward form), so each row's loss is W[T[n]] * (log denom -
    # shifted_logit[T[n]]) = W[T[n]] * (-log_softmax). The "none" backward
    # (loss_grad_output set) routes through the grad loop below instead.
    if reduction == "none" and loss_grad_output is None:
        out = torch.empty(ctx.num_batches, dtype=dtype, device=ctx.input.device)
        for chunk in ctx.chunks():
            logits = ctx.shifted_logits(chunk)
            # Read the target logit BEFORE ``sumexp_`` -- it does ``exp_()``
            # in place, overwriting logits with exp(shifted).
            ls_target = logits.gather(1, chunk.target_chunk.unsqueeze(1)).squeeze(1)
            softmax_denom = ctx.sumexp_(logits, dim=1)
            loss_chunk = softmax_denom.log_().sub_(ls_target.to(softmax_denom.dtype))
            loss_chunk.mul_(chunk.weight_chunk.to(softmax_denom.dtype))
            out.narrow(0, chunk.bchunk_start, chunk.bchunk_size).copy_(loss_chunk)
        return (
            out,
            ctx.grad_input.to(dtype),
            ctx.grad_linear_weight,
            ctx.grad_linear_bias,
        )
    output = ctx.output
    grad_input = ctx.grad_input
    grad_linear_weight = ctx.grad_linear_weight
    grad_linear_bias = ctx.grad_linear_bias
    weight_grad_chunk = ctx.weight_grad_chunk
    linear_weight_cast = ctx.linear_weight_cast

    if reduction == "mean" and ctx.num_batches == 0:
        output.fill_(torch.nan)

    compute_grads = (
        compute_input_grad or compute_linear_weight_grad or compute_linear_bias_grad
    )

    if ctx.is_prob_target:
        # Probability-target loop. The per-row weighting is the dense
        # (B, V) ``prob_wt`` (= class_weight * target) instead of the
        # index path's gathered per-row scalar: the loss target term is
        # a flat dot, the softmax scaling uses the per-chunk row mass
        # s_n = sum_c (w*t)[n, c], and the gradient correction is a
        # dense subtraction -- no gather / index_add_ / index_select.
        # Signs are direct (positive ``loss_scale``), unlike the index
        # loop's negated ``neg_weight_target`` convention.
        loss_scale = ctx.loss_scale
        for chunk in ctx.chunks():
            logits = ctx.shifted_logits(chunk)
            wt = chunk.prob_wt
            # Loss target term sum_{n,c} (w*t)*x_shifted, read BEFORE
            # ``sumexp_`` (it ``exp_``s logits in place). torch.dot returns its
            # scalar at the operand dtype, so when the logits buffer is fp16
            # (CUDA compact) the sum overflows fp16 (>65504) for a
            # large batch of max-shifted (<=0) logits; a per-row einsum summed
            # across rows in fp32 stays finite (same cost as the dot). bf16/fp32
            # operands return a wide-enough scalar, so keep the flat dot.
            if logits.dtype == torch.float16:
                per_row = torch.einsum("bv,bv->b", wt, logits)
                loss_term = per_row.sum(dtype=torch.float32)
            else:
                loss_term = torch.dot(wt.reshape(-1), logits.reshape(-1))
            # ``output`` is fp16 when acc_dtype is fp16 (e.g. MPS). MPS narrows
            # the large fp32 loss_term to fp16 (-> inf) BEFORE applying alpha,
            # unlike the documented type-promoted in-place subtract other
            # backends do; scale first there so the operand stays in fp16 range.
            if output.device.type == "mps":
                output.sub_(loss_term * loss_scale)
            else:
                output.sub_(loss_term, alpha=loss_scale)
            # Per-row weight mass with the reduction factor folded in.
            s = wt.sum(1, dtype=ctx.weight_chunk_dtype).mul_(loss_scale)
            softmax_denom = ctx.sumexp_(logits, dim=1)
            if compute_grads:
                # logits *= s/denom      (= loss_scale * softmax(x) * s_n)
                torch.mul(
                    logits,
                    ctx.div(s, softmax_denom).unsqueeze(1),
                    out=logits,
                )
            # output += <s, log denom>   (loss_scale already in s)
            output.add_(s.to(softmax_denom.dtype).dot(softmax_denom.log_()))
            if compute_grads:
                # grad_logits = loss_scale * (softmax*s_n - w*t); from
                # here every gradient is a single dense op.
                logits.sub_(wt, alpha=loss_scale)
                if compute_linear_bias_grad:
                    chunk.bias_grad_acc.add_(ctx.sum(logits, dim=0))
                if compute_input_grad:
                    ctx.mm(
                        chunk.input_grad_logits,
                        chunk.input_grad_linear_weight,
                        out=chunk.grad_input_chunk,
                    )
                if compute_linear_weight_grad:
                    if ctx.alloc_weight_grad_chunk:
                        ctx.mm(
                            chunk.logits_upcast.T,
                            chunk.weight_grad_input,
                            out=weight_grad_chunk,
                        )
                        grad_linear_weight.add_(weight_grad_chunk)
                    else:
                        grad_linear_weight.addmm_(
                            chunk.logits_downcast.T, chunk.input_chunk, alpha=1
                        )
        return (
            output.to(dtype),
            grad_input.to(dtype),
            grad_linear_weight,
            grad_linear_bias,
        )

    # Do not ``break`` from this loop -- ``ctx.chunks()`` runs a
    # post-yield grad_input commit that the in-flight chunk needs.
    for chunk in ctx.chunks():
        logits = ctx.shifted_logits(chunk)
        weight_chunk = chunk.weight_chunk
        target_chunk = chunk.target_chunk
        # output += <weight_chunk, logits[:, target_chunk]>
        output.add_(ctx.dotgather(weight_chunk, logits, target_chunk))
        softmax_denom = ctx.sumexp_(logits, dim=1)
        if compute_grads:
            # logits *= weight_chunk / softmax_denom         (= softmax(logits) * w)
            # MPS: in-place mul_ on a narrow view of logits_buf doesn't
            # propagate to parent storage; the out= form does.
            torch.mul(
                logits,
                ctx.div(weight_chunk, softmax_denom).unsqueeze(1),
                out=logits,
            )
        # output -= <weight_chunk, log(softmax_denom)>
        # softmax_denom is in acc_dtype (see ``sumexp_``); promote weight_chunk
        # for the dot rather than rounding the wider denominator down.
        output.sub_(weight_chunk.to(softmax_denom.dtype).dot(softmax_denom.log_()))

        if compute_grads:
            if compute_linear_bias_grad:
                # Must run BEFORE the linear-weight-grad branch below.
                # The ordering invariant is defensive across all
                # ``ctx.mul`` paths: only the (not use_acc_dtype &&
                # num_classes >= in_features) branch overwrites the
                # first ``in_features`` columns of ``logits_buf`` via a
                # narrow ``out=`` view today, but ``ctx.mul``'s contract
                # does not promise to leave ``logits_buf`` untouched on
                # other paths -- placing the bias-grad read here keeps
                # the loop correct regardless of which ``ctx.mul``
                # variant fires.
                #
                # The inline form expresses
                #   dL/db_j = sum_n W[T[n]] * (softmax(X)[n, j] - delta(T[n], j))
                # as ``-logits.sum(dim=0)`` (the softmax-weighted term)
                # plus ``index_add_(0, target_chunk, weight_chunk)``
                # (the one-hot-weighted correction). ``logits`` is
                # ``softmax(X) * weight_chunk[n]`` here; ``weight_chunk``
                # already carries the sign and the 1/d mean
                # normalization (``W`` here is the CE class weight, not
                # ``linear_weight``), so no extra scaling is needed.
                bias_grad_acc = chunk.bias_grad_acc
                bias_grad_acc.sub_(ctx.sum(logits, dim=0))
                bias_grad_acc.index_add_(0, target_chunk, weight_chunk)
            if compute_input_grad:
                grad_input_chunk = chunk.grad_input_chunk
                input_grad_logits = chunk.input_grad_logits
                input_grad_linear_weight = chunk.input_grad_linear_weight

                # grad_input_chunk = linear_weight_cast[target_chunk] * weight_chunk
                torch.mul(
                    torch.index_select(linear_weight_cast, 0, target_chunk),
                    weight_chunk.to(grad_input_chunk.dtype).unsqueeze(1),
                    out=grad_input_chunk,
                )
                # grad_input_chunk -= input_grad_logits @ input_grad_linear_weight
                torch.addmm(
                    grad_input_chunk,
                    input_grad_logits,
                    input_grad_linear_weight,
                    alpha=-1,
                    out=grad_input_chunk,
                )
            if compute_linear_weight_grad:
                input_chunk_acc = chunk.input_chunk_acc

                # grad_linear_weight += per-chunk weight grad
                # (= -logits.T @ input + input*weight scattered at target rows)
                if ctx.alloc_weight_grad_chunk:
                    # Stage per-chunk weight grad in acc_dtype scratch;
                    # one sub_ commit keeps bulk-minus-correction in fp32.
                    logits_upcast = chunk.logits_upcast
                    weight_grad_input = chunk.weight_grad_input
                    ctx.mm(
                        logits_upcast.T,
                        weight_grad_input,
                        out=weight_grad_chunk,
                    )
                    temp = ctx.mul(input_chunk_acc, weight_chunk)
                    weight_grad_chunk.index_add_(0, target_chunk, temp, alpha=-1)
                    grad_linear_weight.sub_(weight_grad_chunk)
                else:
                    # Stream bulk + correction directly into grad_lw without
                    # a (V, F) scratch (the compact path).
                    grad_linear_weight.addmm_(
                        chunk.logits_downcast.T, chunk.input_chunk, alpha=-1
                    )
                    temp = ctx.to(
                        ctx.mul(input_chunk_acc, weight_chunk),
                        dtype=grad_linear_weight.dtype,
                    )
                    grad_linear_weight.index_add_(0, target_chunk, temp, alpha=1)

    return (
        output.to(dtype),
        grad_input.to(dtype),
        grad_linear_weight,
        grad_linear_bias,
    )


# Scalar-reduction (mean/sum) op. Returns
# ``(loss, grad_input, grad_linear_weight, grad_linear_bias)`` with grads
# precomputed in forward (the accumulator runs the full grad loop) and
# stashed on ctx by ``setup_context``; backward mutates them in place via
# ``.mul_()`` by the upstream gradient. ``torch.library.register_autograd``
# has no ``save_for_backward``-style state, hence the return-tuple +
# ctx-stash dance. ``mutates_args=()`` reflects that inputs are not mutated.
# Shape invariants: grad_input.shape == input.shape,
# grad_linear_weight.shape == linear_weight.shape, grad_linear_bias.shape ==
# linear_weight.shape[:-1] (here (C,), since the upstream gate requires
# out_features == ()).
@torch.library.custom_op(
    "torch_nn::_linear_cross_entropy_batch_chunked", mutates_args=()
)
def _linear_cross_entropy_batch_chunked(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
    allow_retain_graph: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
    compute_linear_bias_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalar-reduction (mean/sum) chunked op: precomputes the loss and
    gradients in forward, backward scales by the upstream gradient. The
    chunked math lives in ``_linear_cross_entropy_batch_chunked_accumulator``.
    """
    # AOTAutograd/AOTInductor bake compute_*_grad at trace time; catch
    # the silent-corruption case (False at trace, but grad-enabled at
    # runtime with a requires_grad leaf). These guards are specific to
    # the precompute path -- the reduction='none' op recomputes grads in
    # its own backward, so it does not route through here.
    grad_enabled = torch.is_grad_enabled()
    if not compute_input_grad and input.requires_grad and grad_enabled:
        raise RuntimeError(
            "linear_cross_entropy chunked op: compute_input_grad was False at "
            "trace time but input.requires_grad is True at runtime; recompile "
            "the graph with the desired requires_grad."
        )
    if not compute_linear_weight_grad and linear_weight.requires_grad and grad_enabled:
        raise RuntimeError(
            "linear_cross_entropy chunked op: compute_linear_weight_grad was "
            "False at trace time but linear_weight.requires_grad is True at "
            "runtime; recompile the graph with the desired requires_grad."
        )
    if (
        not compute_linear_bias_grad
        and linear_bias is not None
        and linear_bias.requires_grad
        and grad_enabled
    ):
        raise RuntimeError(
            "linear_cross_entropy chunked op: compute_linear_bias_grad was False at "
            "trace time but linear_bias.requires_grad is True at runtime; recompile "
            "the graph with the desired requires_grad."
        )
    if target.dtype.is_floating_point and target.requires_grad and grad_enabled:
        # No gradient slot is produced for target; without this guard a
        # probability target requiring grad would silently get none.
        raise RuntimeError(
            "linear_cross_entropy chunked op: gradients w.r.t. a probability "
            "target are not supported on the chunked path; use options=None."
        )
    return _linear_cross_entropy_batch_chunked_accumulator(
        input,
        linear_weight,
        target,
        linear_bias,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad,
        compute_linear_weight_grad,
        compute_linear_bias_grad,
    )


# Op arg count; backward returns this many slots. Hand-maintained
# because CustomOpDef has no public arg-introspection. Update on
# signature change.
_NUM_OP_INPUTS = 15


@_linear_cross_entropy_batch_chunked.register_fake
def _(
    input,
    linear_weight,
    target,
    linear_bias,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
    batch_chunk_size,
    acc_policy: str,
    acc_dtype,
    allow_retain_graph,
    compute_input_grad,
    compute_linear_weight_grad,
    compute_linear_bias_grad,
):
    if reduction in {"mean", "sum"}:
        result = torch.empty((), dtype=input.dtype, device=input.device)
    else:
        raise NotImplementedError(f"linear_cross_entropy does not support {reduction=}")
    grad_input_shape = input.shape if compute_input_grad else (0, 0)
    grad_linear_weight_shape = (
        linear_weight.shape if compute_linear_weight_grad else (0, 0)
    )
    # ``linear_bias.shape == linear_weight.shape[:-1]`` (the (C, *out_features)
    # invariant); on this code path ``out_features == ()`` so it reduces
    # to (C,), but expressing it this way keeps the relationship visible.
    grad_linear_bias_shape = (
        linear_weight.shape[:-1] if compute_linear_bias_grad else (0,)
    )

    grad_input = torch.empty(
        grad_input_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=False,
    )
    # The real implementation allocates ``grad_linear_weight`` and
    # ``grad_linear_bias`` in ``self.dtype = input.dtype`` (chunked op
    # supports mixed-precision: ``input.dtype`` may differ from
    # ``linear_weight.dtype``). Mirror that here so the fake-tensor
    # shape/dtype matches what eager produces.
    grad_linear_weight = torch.empty(
        grad_linear_weight_shape,
        dtype=input.dtype,
        device=linear_weight.device,
        requires_grad=False,
    )
    grad_linear_bias = torch.empty(
        grad_linear_bias_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=False,
    )
    return result, grad_input, grad_linear_weight, grad_linear_bias


@_linear_cross_entropy_batch_chunked.register_vmap
def _vmap(info, in_dims, *args):
    """vmap rule (slow path: per-sample Python loop). A fold-into-num_batches
    fast path is now possible for both ops via the reduction='none' chunked op
    (``_linear_cross_entropy_batch_chunked_no_reduction``); left as a follow-up.
    """
    batch_size = info.batch_size

    moved_args = [
        arg.movedim(in_dim, 0)
        if in_dim is not None and isinstance(arg, torch.Tensor)
        else arg
        for arg, in_dim in zip(args, in_dims)
    ]

    outputs = [
        _linear_cross_entropy_batch_chunked(
            *[
                arg[i] if in_dim is not None and isinstance(arg, torch.Tensor) else arg
                for arg, in_dim in zip(moved_args, in_dims)
            ]
        )
        for i in range(batch_size)
    ]
    losses = torch.stack([o[0] for o in outputs])
    grad_inputs = torch.stack([o[1] for o in outputs])
    grad_linear_weights = torch.stack([o[2] for o in outputs])
    grad_linear_biases = torch.stack([o[3] for o in outputs])
    return (losses, grad_inputs, grad_linear_weights, grad_linear_biases), (0, 0, 0, 0)


def _linear_cross_entropy_batch_chunked_backward(
    ctx, grad_output, _gi_grad, _gw_grad, _gb_grad
):
    result = [None] * _NUM_OP_INPUTS
    # set_materialize_grads(False): grad_output is None when backward is
    # driven only through the (unsupported) grad outputs -- no gradient
    # flows through the loss, so none flow to the inputs.
    if grad_output is None:
        return tuple(result)
    if ctx.compute_input_grad:
        if ctx.allow_retain_graph:
            result[0] = ctx._gi * grad_output
        else:
            gi, ctx._gi = ctx._gi, None
            if gi is None:
                raise RuntimeError(
                    "linear_cross_entropy chunked backward called twice; "
                    "retain_graph=True / double backward are not supported."
                )
            result[0] = gi.mul_(grad_output)

    if ctx.compute_linear_weight_grad:
        if ctx.allow_retain_graph:
            result[1] = ctx._gw * grad_output
        else:
            gw, ctx._gw = ctx._gw, None
            if gw is None:
                raise RuntimeError(
                    "linear_cross_entropy chunked backward called twice; "
                    "retain_graph=True / double backward are not supported."
                )
            result[1] = gw.mul_(grad_output)

    # ``linear_bias`` is forward arg position 3 (after input, linear_weight, target).
    if ctx.compute_linear_bias_grad:
        if ctx.allow_retain_graph:
            result[3] = ctx._gb * grad_output
        else:
            gb, ctx._gb = ctx._gb, None
            if gb is None:
                raise RuntimeError(
                    "linear_cross_entropy chunked backward called twice; "
                    "retain_graph=True / double backward are not supported."
                )
            result[3] = gb.mul_(grad_output)
    return tuple(result)


_linear_cross_entropy_batch_chunked.register_autograd(
    _linear_cross_entropy_batch_chunked_backward,
    setup_context=_linear_cross_entropy_batch_chunked_setup_context,
)


# reduction='none' op. Separate from the scalar-reduction op because the
# autograd model differs: per-sample loss makes ``grad_output`` a vector,
# so grad_linear_weight / grad_linear_bias cannot be precomputed in forward
# (they need the per-sample upstream grad before summing over the batch, and
# storing the per-sample (N, V) softmax would defeat chunking). Instead the
# forward returns only the loss and saves its inputs; backward recomputes the
# chunked grads with ``grad_output`` folded into ``weight_chunk`` (both share
# ``_linear_cross_entropy_batch_chunked_accumulator``). Peak memory stays bounded to
# (batch_chunk_size, num_classes); the cost is one recomputed logits matmul
# per chunk in backward. Recompute-from-saved-inputs supports retain_graph
# naturally (no consumed buffers), so no allow_retain_graph flag is needed.
@torch.library.custom_op(
    "torch_nn::_linear_cross_entropy_batch_chunked_no_reduction", mutates_args=()
)
def _linear_cross_entropy_batch_chunked_no_reduction(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    ignore_index: int,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
) -> torch.Tensor:
    """reduction='none' chunked forward: returns per-sample loss ``(N,)``.

    Gradients are not precomputed (the per-sample upstream grad is
    unknown at forward); the accumulator's loss-only branch runs here,
    and the backward recomputes the chunked grads. See
    ``_linear_cross_entropy_batch_chunked_accumulator``.
    """
    # loss_grad_output=None -> the accumulator's reduction='none' forward
    # (loss-only) branch; compute_*=False since nothing is precomputed.
    return _linear_cross_entropy_batch_chunked_accumulator(
        input,
        linear_weight,
        target,
        linear_bias,
        weight,
        "none",
        ignore_index,
        0.0,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad=False,
        compute_linear_weight_grad=False,
        compute_linear_bias_grad=False,
    )[0]


def _linear_cross_entropy_batch_chunked_no_reduction_setup_context(ctx, inputs, output):
    (
        input,
        linear_weight,
        target,
        linear_bias,
        weight,
        ignore_index,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
    ) = inputs
    # Detach before saving: the backward recomputes the grads analytically
    # from these values (it does not differentiate through them), and saving
    # ``linear_weight`` -- a differentiable ``reshape`` view of the
    # ``nn.Linear`` weight -- would make its SavedVariable retain the base
    # parameter, inflating its use_count past backward and breaking
    # swap-based ``module.to()`` (the custom-op node keeps saved tensors
    # alive while the loss output is still referenced).
    ctx.save_for_backward(
        input.detach(),
        linear_weight.detach(),
        target,
        linear_bias.detach() if linear_bias is not None else None,
        weight.detach() if weight is not None else None,
    )
    ctx._ignore_index = ignore_index
    ctx._batch_chunk_size = batch_chunk_size
    ctx._acc_policy = acc_policy
    ctx._acc_dtype = acc_dtype


@_linear_cross_entropy_batch_chunked_no_reduction.register_fake
def _(
    input,
    linear_weight,
    target,
    linear_bias,
    weight,
    ignore_index,
    batch_chunk_size,
    acc_policy: str,
    acc_dtype,
):
    return torch.empty(
        input.shape[:1], dtype=input.dtype, device=input.device, requires_grad=False
    )


@_linear_cross_entropy_batch_chunked_no_reduction.register_vmap
def _vmap_no_reduction(info, in_dims, *args):
    """vmap rule (slow path: per-sample Python loop, mirrors the
    scalar-reduction op). Stacks the per-sample ``(N,)`` losses.
    """
    batch_size = info.batch_size
    moved_args = [
        arg.movedim(in_dim, 0)
        if in_dim is not None and isinstance(arg, torch.Tensor)
        else arg
        for arg, in_dim in zip(args, in_dims)
    ]
    outputs = [
        _linear_cross_entropy_batch_chunked_no_reduction(
            *[
                arg[i] if in_dim is not None and isinstance(arg, torch.Tensor) else arg
                for arg, in_dim in zip(moved_args, in_dims)
            ]
        )
        for i in range(batch_size)
    ]
    return torch.stack(outputs), 0


def _linear_cross_entropy_batch_chunked_no_reduction_backward(ctx, grad_output):
    # grad_output is the per-sample upstream grad (N,). Forward arg order:
    # (input, linear_weight, target, linear_bias, weight, ignore_index,
    #  batch_chunk_size, acc_policy, acc_dtype).
    input, linear_weight, target, linear_bias, weight = ctx.saved_tensors
    needs = ctx.needs_input_grad
    compute_input_grad = needs[0]
    compute_linear_weight_grad = needs[1]
    compute_linear_bias_grad = linear_bias is not None and needs[3]
    result: list[torch.Tensor | None] = [None] * 9
    if not (
        compute_input_grad or compute_linear_weight_grad or compute_linear_bias_grad
    ):
        return tuple(result)
    # loss_grad_output=grad_output -> the accumulator's grad loop with the
    # per-sample upstream grad folded into neg_weight_target (the "none"
    # backward path), recomputing the chunked grads as the VJP. The loss
    # (result[0]) it also returns is the meaningless scalar and dropped.
    _, grad_input, grad_linear_weight, grad_linear_bias = (
        _linear_cross_entropy_batch_chunked_accumulator(
            input,
            linear_weight,
            target,
            linear_bias,
            weight,
            "none",
            ctx._ignore_index,
            0.0,  # label_smoothing
            ctx._batch_chunk_size,
            ctx._acc_policy,
            ctx._acc_dtype,
            compute_input_grad,
            compute_linear_weight_grad,
            compute_linear_bias_grad,
            loss_grad_output=grad_output,
        )
    )
    if compute_input_grad:
        result[0] = grad_input
    if compute_linear_weight_grad:
        result[1] = grad_linear_weight
    if compute_linear_bias_grad:
        result[3] = grad_linear_bias
    return tuple(result)


_linear_cross_entropy_batch_chunked_no_reduction.register_autograd(
    _linear_cross_entropy_batch_chunked_no_reduction_backward,
    setup_context=_linear_cross_entropy_batch_chunked_no_reduction_setup_context,
)
<|endoftext|># mypy: allow-untyped-defs
import numbers

import torch
from torch import Size, Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter

from ._functions import CrossMapLRN2d as _cross_map_lrn2d
from .module import Module


__all__ = ["LocalResponseNorm", "CrossMapLRN2d", "LayerNorm", "GroupNorm", "RMSNorm"]


class LocalResponseNorm(Module):
    r"""Applies local response normalization over an input signal.

    The input signal is composed of several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    .. math::
        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    Args:
        size: amount of neighbouring channels used for normalization
        alpha: multiplicative factor. Default: 0.0001
        beta: exponent. Default: 0.75
        k: additive factor. Default: 1

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> lrn = nn.LocalResponseNorm(2)
        >>> signal_2d = torch.randn(32, 5, 24, 24)
        >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
        >>> output_2d = lrn(signal_2d)
        >>> output_4d = lrn(signal_4d)

    """

    __constants__ = ["size", "alpha", "beta", "k"]
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.local_response_norm(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self):
        """
        Return the extra representation of the module.
        """
        return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)


class CrossMapLRN2d(Module):
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(
        self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return _cross_map_lrn2d.apply(input, self.size, self.alpha, self.beta, self.k)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return "{size}, alpha={alpha}, beta={beta}, k={k}".format(**self.__dict__)


_shape_t = int | list[int] | Size


class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    `torch.var(input, correction=0)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    .. image:: ../_static/img/nn/layer_norm.jpg
        :scale: 50 %

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, "
            "bias={use_bias}".format(**self.__dict__, use_bias=self.bias is not None)
        )


class GroupNorm(Module):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    `torch.var(input, correction=0)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
        *,
        bias: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (
            "{num_groups}, {num_channels}, eps={eps}, affine={affine}, "
            "bias={use_bias}".format(**self.__dict__, use_bias=self.bias is not None)
        )


class RMSNorm(Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
        \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}

    The RMS is taken over the last ``D`` dimensions, where ``D``
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the RMS is computed over
    the last 2 dimensions of the input.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float, optional): a value added to the denominator for numerical stability.
            If not specified, uses the machine epsilon of the computation (opmath) type:
            fp16/bf16 and fp32 inputs use ``torch.finfo(torch.float32).eps``, while fp64
            inputs use ``torch.finfo(torch.float64).eps``. Default: ``None``
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> rms_norm = nn.RMSNorm([2, 3])
        >>> input = torch.randn(2, 2, 3)
        >>> rms_norm(input)

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: float | None
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


# TODO: ContrastiveNorm2d
# TODO: DivisiveNorm2d
# TODO: SubtractiveNorm2d
<|endoftext|># mypy: allow-untyped-defs

import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _ratio_2_t, _ratio_any_t, _size_2_t, _size_any_t

from .module import Module


__all__ = ["Upsample", "UpsamplingNearest2d", "UpsamplingBilinear2d"]


class Upsample(Module):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation. If `recompute_scale_factor` is ``True``, then
            `scale_factor` must be passed in and `scale_factor` is used to compute the
            output `size`. The computed output `size` will be used to infer new scales for
            the interpolation. Note that when `scale_factor` is floating-point, it may differ
            from the recomputed `scale_factor` due to rounding and precision issues.
            If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
            be used directly for interpolation.

    Shape:
        - Input: :math:`(N, C, W_{in})`, :math:`(N, C, H_{in}, W_{in})` or :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, W_{out})`, :math:`(N, C, H_{out}, W_{out})`
          or :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

    .. math::
        D_{out} = \left\lfloor D_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, `bicubic`, and `trilinear`) don't proportionally
        align the output and input pixels, and thus the output values can depend
        on the input size. This was the default behavior for these modes up to
        version 0.3.1. Since then, the default behavior is
        ``align_corners = False``. See below for concrete examples on how this
        affects the outputs.

    .. note::
        If you want downsampling/general resizing, you should use :func:`~nn.functional.interpolate`.

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[1., 2.],
                  [3., 4.]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='nearest')
        >>> m(input)
        tensor([[[[1., 1., 2., 2.],
                  [1., 1., 2., 2.],
                  [3., 3., 4., 4.],
                  [3., 3., 4., 4.]]]])

        >>> # xdoctest: +IGNORE_WANT("other tests seem to modify printing styles")
        >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
        >>> m(input)
        tensor([[[[1.0000, 1.2500, 1.7500, 2.0000],
                  [1.5000, 1.7500, 2.2500, 2.5000],
                  [2.5000, 2.7500, 3.2500, 3.5000],
                  [3.0000, 3.2500, 3.7500, 4.0000]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        >>> m(input)
        tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
                  [1.6667, 2.0000, 2.3333, 2.6667],
                  [2.3333, 2.6667, 3.0000, 3.3333],
                  [3.0000, 3.3333, 3.6667, 4.0000]]]])

        >>> # Try scaling the same data in a larger tensor
        >>> input_3x3 = torch.zeros(3, 3).view(1, 1, 3, 3)
        >>> input_3x3[:, :, :2, :2].copy_(input)
        tensor([[[[1., 2.],
                  [3., 4.]]]])
        >>> input_3x3
        tensor([[[[1., 2., 0.],
                  [3., 4., 0.],
                  [0., 0., 0.]]]])

        >>> # xdoctest: +IGNORE_WANT("seems to fail when other tests are run in the same session")
        >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
        >>> # Notice that values in top left corner are the same with the small input (except at boundary)
        >>> m(input_3x3)
        tensor([[[[1.0000, 1.2500, 1.7500, 1.5000, 0.5000, 0.0000],
                  [1.5000, 1.7500, 2.2500, 1.8750, 0.6250, 0.0000],
                  [2.5000, 2.7500, 3.2500, 2.6250, 0.8750, 0.0000],
                  [2.2500, 2.4375, 2.8125, 2.2500, 0.7500, 0.0000],
                  [0.7500, 0.8125, 0.9375, 0.7500, 0.2500, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        >>> # Notice that values in top left corner are now changed
        >>> m(input_3x3)
        tensor([[[[1.0000, 1.4000, 1.8000, 1.6000, 0.8000, 0.0000],
                  [1.8000, 2.2000, 2.6000, 2.2400, 1.1200, 0.0000],
                  [2.6000, 3.0000, 3.4000, 2.8800, 1.4400, 0.0000],
                  [2.4000, 2.7200, 3.0400, 2.5600, 1.2800, 0.0000],
                  [1.2000, 1.3600, 1.5200, 1.2800, 0.6400, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
    """

    __constants__ = [
        "size",
        "scale_factor",
        "mode",
        "align_corners",
        "name",
        "recompute_scale_factor",
    ]
    name: str
    size: _size_any_t | None
    scale_factor: _ratio_any_t | None
    mode: str
    align_corners: bool | None
    recompute_scale_factor: bool | None

    def __init__(
        self,
        size: _size_any_t | None = None,
        scale_factor: _ratio_any_t | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
    ) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

    def __setstate__(self, state):
        if "recompute_scale_factor" not in state:
            state["recompute_scale_factor"] = True

        super().__setstate__(state)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        if self.scale_factor is not None:
            info = "scale_factor=" + repr(self.scale_factor)
        else:
            info = "size=" + repr(self.size)
        info += ", mode=" + repr(self.mode)
        return info


class UpsamplingNearest2d(Upsample):
    r"""Applies a 2D nearest neighbor upsampling to an input signal composed of several input channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as its constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    .. warning::
        This class is deprecated in favor of :func:`~nn.functional.interpolate`.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    .. math::
          H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
          W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[1., 2.],
                  [3., 4.]]]])

        >>> m = nn.UpsamplingNearest2d(scale_factor=2)
        >>> m(input)
        tensor([[[[1., 1., 2., 2.],
                  [1., 1., 2., 2.],
                  [3., 3., 4., 4.],
                  [3., 3., 4., 4.]]]])
    """

    def __init__(
        self,
        size: _size_2_t | None = None,
        scale_factor: _ratio_2_t | None = None,
    ) -> None:
        super().__init__(size, scale_factor, mode="nearest")


class UpsamplingBilinear2d(Upsample):
    r"""Applies a 2D bilinear upsampling to an input signal composed of several input channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as its constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    .. warning::
        This class is deprecated in favor of :func:`~nn.functional.interpolate`. It is
        equivalent to ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale\_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale\_factor} \right\rfloor

    Examples::

        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
        >>> input
        tensor([[[[1., 2.],
                  [3., 4.]]]])

        >>> # xdoctest: +IGNORE_WANT("do other tests modify the global state?")
        >>> m = nn.UpsamplingBilinear2d(scale_factor=2)
        >>> m(input)
        tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
                  [1.6667, 2.0000, 2.3333, 2.6667],
                  [2.3333, 2.6667, 3.0000, 3.3333],
                  [3.0000, 3.3333, 3.6667, 4.0000]]]])
    """

    def __init__(
        self,
        size: _size_2_t | None = None,
        scale_factor: _ratio_2_t | None = None,
    ) -> None:
        super().__init__(size, scale_factor, mode="bilinear", align_corners=True)
<|endoftext|>import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = ["ChannelShuffle"]


class ChannelShuffle(Module):
    r"""Divides and rearranges the channels in a tensor.

    This operation divides the channels in a tensor of shape :math:`(N, C, *)`
    into g groups as :math:`(N, \frac{C}{g}, g, *)` and shuffles them,
    while retaining the original tensor shape in the final output.

    Args:
        groups (int): number of groups to divide channels in.

    Examples::

        >>> channel_shuffle = nn.ChannelShuffle(2)
        >>> input = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 2, 2)
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[13., 14.],
                  [15., 16.]]]])
        >>> output = channel_shuffle(input)
        >>> output
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[13., 14.],
                  [15., 16.]]]])
    """

    __constants__ = ["groups"]
    groups: int

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.channel_shuffle(input, self.groups)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"groups={self.groups}"
<|endoftext|>import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import (
    _ratio_2_t,
    _ratio_3_t,
    _size_1_t,
    _size_2_opt_t,
    _size_2_t,
    _size_3_opt_t,
    _size_3_t,
    _size_any_opt_t,
    _size_any_t,
)

from .module import Module
from .utils import _pair, _single, _triple


__all__ = [
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
]


class _MaxPoolNd(Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "return_indices",
        "ceil_mode",
    ]
    return_indices: bool
    ceil_mode: bool

    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: _size_any_t | None = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", dilation={dilation}, ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class MaxPool1d(_MaxPoolNd):
    r"""Applies a 1D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the
    sliding window. This `link`_ has a nice visualization of the pooling parameters.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    Args:
        kernel_size: The size of the sliding window, must be > 0.
        stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.

    Shape:
        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.
        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`,

          where ``ceil_mode = False``

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                   \times (\text{kernel\_size} - 1) - 1}{\text{stride}}\right\rfloor + 1

          where ``ceil_mode = True``

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1 + (\text{stride} - 1)}{\text{stride}}\right\rfloor + 1

        - Ensure that the last pooling starts inside the image, make :math:`L_{out} = L_{out} - 1`
          when :math:`(L_{out} - 1) * \text{stride} >= L_{in} + \text{padding}`.

    Examples::

        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t

    def forward(self, input: Tensor):
        """Runs the forward pass."""
        return F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MaxPool2d(_MaxPoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
