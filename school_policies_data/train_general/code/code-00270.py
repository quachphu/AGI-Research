                    f"Stage {self.stage_index}: output_grads metadata is required for backward inference."
                )
            stage_output_grad_metas = self._stage_meta.output_grads

            filtered_fwd_outputs: list[torch.Tensor] = []
            filtered_output_grads: list[torch.Tensor | None] = []

            for idx, (fwd_out, grad_meta) in enumerate(
                zip(fwd_outputs, stage_output_grad_metas, strict=True)
            ):
                # Match _backward.py behavior: skip if output doesn't require grad AND has no grad_fn
                if not fwd_out.requires_grad:
                    if grad_meta is not None:
                        raise PipeliningMetadataError(
                            f"Stage {self.stage_index}: output {idx} requires_grad=False, "
                            f"but output_grads metadata is provided: {grad_meta}."
                        )
                    continue
                filtered_fwd_outputs.append(fwd_out)
                # For outputs that require grad, include them even if grad_meta is None
                # (runtime passes None grad_outputs to autograd.backward in this case)
                filtered_output_grads.append(
                    self._ones_from_metadata(grad_meta) if grad_meta else None
                )

            if filtered_fwd_outputs:
                all_input_grads = self._compute_input_grads(
                    filtered_fwd_outputs, all_fwd_inputs, filtered_output_grads
                )
                # Free intermediate references early
                filtered_fwd_outputs.clear()
                filtered_output_grads.clear()
                all_fwd_inputs.clear()
                # Only positional input grads flow to previous stage
            else:
                all_input_grads = tuple(None for _ in range(len(all_fwd_inputs)))

        input_grads = all_input_grads[: len(fwd_inputs)]
        processed_metas: list[TensorMeta | None] = []
        for i, g in enumerate(input_grads):
            if isinstance(g, torch.Tensor):
                processed_metas.append(extract_tensor_meta(g))
            elif fwd_inputs[i].requires_grad:
                processed_metas.append(
                    _derive_grad_metas((extract_tensor_meta(fwd_inputs[i]),))[0]
                )
            else:
                processed_metas.append(None)
        self._stage_meta.input_grads = tuple(processed_metas)

        # === SEND: Pass input grad metadata to previous stage ===
        bwd_meta = _StageBackwardMeta(backward_metas=self._stage_meta.input_grads)

        if self.is_first or self._is_same_rank(self.stage_index - 1):
            # First rank or Same-rank: return for caller to pass
            return bwd_meta
        else:
            # Cross-rank: send via P2P
            self._send_meta(bwd_meta, self.stage_index - 1)
            return None

    def _post_metadata_inference_cleanup(self) -> None:
        """Restore state and clean up side effects from metadata inference.

        Buffer restore is intentionally delayed until after both forward and
        backward metadata inference. Restoring immediately after forward can
        bump autograd version counters for buffers saved by the forward graph.
        """
        buffer_backup = self._metadata_inference_buffer_backup
        if buffer_backup is not None:
            with torch.no_grad():
                for buffer, saved_buffer in buffer_backup:
                    buffer.copy_(saved_buffer)
            self._metadata_inference_buffer_backup = None

        # Clear stored inference tensors (frees autograd graph + activations)
        self._fwd_outputs_for_bwd_meta = None
        self._fwd_inputs_for_bwd_meta = None
        self._fwd_kwargs_tensors_for_bwd_meta = None

        # Metadata inference runs real fwd/bwd, which unshards FSDP params and
        # accumulates grads.  Reshard to free memory and clear stale grads.
        for module in self.submod.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
                for param in module.parameters():
                    param.grad = None

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: "_StageBackwardMeta | None" = None,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> "_StageBackwardMeta | None":
        """Run backward metadata inference and prepare backward infrastructure.

        Returns:
            ``_StageBackwardMeta`` for the previous same-rank stage, or ``None``.
        """
        grad_meta_result: _StageBackwardMeta | None = None
        if self._inference_mode == InferenceMode.DYNAMIC:
            # DYNAMIC mode: run backward metadata inference
            # received_grad_meta is used for same-rank V-schedule stages
            grad_meta_result = self._backward_metadata_inference(
                loss_fn=loss_fn,
                target=target,
                received_grad_meta=received_grad_meta,
                loss_kwargs=loss_kwargs,
            )
            # Validate dynamically inferred metadata against user-provided metadata
            self._validate_inferred_metadata()
        else:
            # STATIC mode: metadata comes from user inputs, no validation needed
            self._stage_meta.input_grads = self._user_meta.input_grads
            self._stage_meta.output_grads = self._user_meta.output_grads
            # For STATIC mode with plain tensors, if output_grads is not set but
            # we have outputs, derive output_grads from outputs.
            # (gradient shape == output shape, but requires_grad=False for gradients)
            if self._stage_meta.output_grads is None:
                if self._stage_meta.outputs is None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: output metadata is required for backward inference."
                    )
                self._stage_meta.output_grads = _derive_grad_metas(
                    self._stage_meta.outputs
                )
            # Similarly, derive input_grads from inputs if not provided
            if self._stage_meta.input_grads is None:
                if self._stage_meta.inputs is None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: input metadata is required for backward inference."
                    )
                self._stage_meta.input_grads = _derive_grad_metas(
                    self._stage_meta.inputs
                )

        # Note: grad_send_info is created lazily in get_bwd_send_ops() since
        # it mirrors args_recv_info (already populated during forward).
        self._setup_backward_recv_info(num_microbatches)
        return grad_meta_result

    def _validate_inferred_metadata(self) -> None:
        """Validate dynamically inferred metadata against user-provided metadata."""
        pairs = [
            (self._user_meta.inputs, self._stage_meta.inputs, "input"),
            (self._user_meta.outputs, self._stage_meta.outputs, "output"),
            (self._user_meta.input_grads, self._stage_meta.input_grads, "input_grad"),
            (
                self._user_meta.output_grads,
                self._stage_meta.output_grads,
                "output_grad",
            ),
        ]
        for user_val, stage_val, label in pairs:
            if user_val and stage_val:
                validate_tensors_metadata(
                    f"Stage {self.stage_index} {label}",
                    user_val,
                    stage_val,
                    warn_on_mismatch=True,
                )

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        """Prepare the stage infrastructure for forward pass.

        Returns:
            ``_StageForwardMeta`` for next stage (same-rank), or ``None`` if sent via P2P.
        """
        if self._inference_mode is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: inference mode not set. "
                f"Run warmup vote protocol first."
            )

        fwd_meta_output: _StageForwardMeta | None = None

        if self._inference_mode == InferenceMode.DYNAMIC:
            # DYNAMIC mode: run forward metadata inference
            # args may be _StageForwardMeta for same-rank V-schedule stages
            fwd_meta_output = self._forward_metadata_inference(
                args, kwargs, has_backward
            )
            # Validate dynamically inferred metadata against user-provided metadata
            self._validate_inferred_metadata()
        # STATIC mode: metadata comes from user inputs, no validation needed
        else:
            self._stage_meta.inputs = self._user_meta.inputs
            self._stage_meta.outputs = self._user_meta.outputs

        # Setup recv and send info
        self._setup_forward_recv_info(num_microbatches, has_backward)
        self._setup_forward_send_info()

        return fwd_meta_output

    def _setup_forward_recv_info(
        self, num_microbatches: int, has_backward: bool
    ) -> None:
        """Setup receive info for forward pass."""
        if self._stage_meta.inputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: inputs metadata required for recv info."
            )
        for chunk_id in range(num_microbatches):
            if self.is_first:
                # First stage: all inputs are root arguments (no recv needed)
                self.args_recv_info[chunk_id] = tuple(
                    _RecvInfo(
                        input_name=f"root_input_{idx}",
                        source=None,
                        buffer=None,
                        tensor_meta=meta,
                        is_root_arg=True,
                    )
                    for idx, meta in enumerate(self._stage_meta.inputs)
                )
            else:
                # Non-first stages: receive from previous stage
                self.args_recv_info[chunk_id] = tuple(
                    _RecvInfo(
                        input_name=f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                        source=self.stage_index - 1,
                        buffer=_make_tensor_from_meta(meta, self.device),
                        tensor_meta=meta,
                    )
                    for meta in self._stage_meta.inputs
                )

    def _setup_forward_send_info(self) -> None:
        """Setup send info for forward pass."""
        self.act_send_info: dict[int, list] = {}
        if self._stage_meta.outputs is None:
            raise PipeliningMetadataError(
                f"Stage {self.stage_index}: outputs metadata required for recv info."
            )
        for idx in range(len(self._stage_meta.outputs)):
            self.act_send_info[idx] = [self.stage_index + 1] if not self.is_last else []

    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        grad_recv_infos: list[_RecvInfo] = []
        if not self.is_last:
            # Ensure output_grads metadata is available
            if self._stage_meta.output_grads is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: output_grads metadata is required for "
                    f"creating grad recv info. Ensure backward metadata is populated."
                )

            # Receiving gradients from multiple sources is not supported
            # hence we only take the first destination
            # Use a helper function to safely extract the metadata
            output_grads = self._stage_meta.output_grads
            for idx, dst_list in act_send_info.items():
                if dst_list is None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: output {idx} is not sent to any stage."
                    )
                src = dst_list[0]
                grad_meta = output_grads[idx]
                grad_recv_infos.append(
                    _RecvInfo(
                        input_name=f"recv_grad_for_{self.stage_index}_from_{src}",
                        source=src,
                        buffer=_make_tensor_from_meta(grad_meta, self.device)
                        if grad_meta
                        else None,
                        tensor_meta=grad_meta,
                    )
                )
        return tuple(grad_recv_infos)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import logging
import operator
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from inspect import Parameter, Signature, signature
from types import MethodType
from typing import Any

import torch
import torch.fx as fx
from torch.distributed import ProcessGroup
from torch.export import ExportedProgram
from torch.export.unflatten import (
    _assign_attr,
    _AttrKind,
    _sink_params,
    InterpreterModule,
)
from torch.fx.node import map_aggregate
from torch.fx.passes.split_module import split_module

from ._backward import _null_coalesce_accumulate, stage_backward
from ._unflatten import _outline_submodules
from ._utils import PipeInfo
from .stage import _PipelineStage


logger = logging.getLogger(__name__)

# TODO:
# 1. investigate gradient sync for shared parameters. how does DDP do it?
# 2. Add parameter movement to split_module


PP_SUBMOD_PREFIX = "submod_pp"


def get_submod_name(stage_idx: int):
    """Returns the name of the submod for a given stage index.
    For example, "submod_pp_0", "submod_pp_1", etc.
    """
    return "_".join([PP_SUBMOD_PREFIX, str(stage_idx)])


def _find_loss_from_output_and_spec(output_val, spec_val):
    if spec_val is False:
        return None
    if spec_val is True:
        if not isinstance(output_val, fx.Node):
            raise RuntimeError(
                f"Loss spec must specify a dynamic value but got {output_val}"
            )
        return output_val

    if isinstance(spec_val, (tuple, list)):
        if not isinstance(output_val, (tuple, list)):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        if len(output_val) != len(spec_val):
            raise RuntimeError(
                f"Output value {output_val} must match length of loss specification "
                f"{spec_val}"
            )
        for out, spec in zip(output_val, spec_val):
            loss_val = _find_loss_from_output_and_spec(out, spec)
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    if isinstance(spec_val, dict):
        if not isinstance(output_val, dict):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        if set(output_val.keys()) != set(spec_val.keys()):
            raise RuntimeError(
                f"Output value {output_val} must match keys of loss specification "
                f"{spec_val}"
            )
        for k in spec_val:
            loss_val = _find_loss_from_output_and_spec(output_val[k], spec_val[k])
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    raise RuntimeError(f"Unsupported type {type(spec_val)} in loss specification")


def _find_loss_output(mod: torch.nn.Module, g: fx.Graph, output_loss_value_spec):
    output_nodes = [n for n in g.nodes if n.op == "output"]
    if not len(output_nodes) == 1:
        raise AssertionError(f"Expected 1 output node, got {len(output_nodes)}")
    output_node = output_nodes[0]
    output_val = output_node.args[0]
    generated_spec: Any = None

    if isinstance(mod, TrivialLossWrapper):
        # TrivialLossWrapper is pre-defined by PiPPy.
        # It has loss as the only output so we can safely assume the first output arg is the loss.
        if not len(output_node.args) == 1:
            raise AssertionError(f"Expected 1 output arg, got {len(output_node.args)}")
        loss_node = output_val
        generated_spec = TrivialLossWrapper.loss_spec
    elif output_loss_value_spec is None:
        # Use default spec, i.e. search for "loss" in output values
        if isinstance(output_val, dict) and "loss" in output_val:
            loss_node = output_val["loss"]
            generated_spec = {k: k == "loss" for k in output_val}
        else:
            loss_node = None
            generated_spec = None
    else:
        loss_node = _find_loss_from_output_and_spec(output_val, output_loss_value_spec)
        generated_spec = output_loss_value_spec

    return loss_node, output_node, generated_spec


def _insert_stage_symbolic_backward(
    g: fx.Graph,
    loss_node: fx.Node,
    output_node: fx.Node,
):
    # Collect metadata about tuple output values. TODO: move this to split_module or FX IR
    tuples: dict[fx.Node, tuple] = {}
    for node in reversed(g.nodes):
        if node.op == "call_function":
            # In the forward pass, only emit placeholder, module calls, and
            # getitem calls. If we have a target other than getitem in this
            # (forward-only) code, there is a bug.
            if not node.target == operator.getitem:
                raise AssertionError(
                    "Found non-getitem call in forward pass. Please report a bug to PiPPy"
                )
            if not len(node.args) == 2:
                raise AssertionError(
                    "Found malformed getitem call. Please report a bug to PiPPy"
                )
            indexed_value, node_idx = tuple(node.args)

            # indexed_value is a collection that we are indexing into. It could
            # exist in the tuples map if we've processed another `getitem`
            # already.
            existing_list_size = (
                len(tuples[indexed_value]) if indexed_value in tuples else -1
            )
            new_list_size = max(node_idx + 1, existing_list_size)

            reconstructed_list = [None for _ in range(new_list_size)]

            # Copy over existing elements if present
            if indexed_value in tuples:
                for i, val in enumerate(tuples[indexed_value]):
                    reconstructed_list[i] = val

            # Populate value represented by this node
            reconstructed_list[node_idx] = node

            tuples[indexed_value] = tuple(reconstructed_list)

    # Keep track of nodes that dominate the loss node.
    # We will only emit backward operations for nodes that can contribute
    # to the specified loss value.
    live_nodes = {loss_node: None}
    val_to_grad: dict[fx.Node, fx.Node | None] = {loss_node: None}

    def assign_or_accumulate_grad(forward_node, grad_value):
        if forward_node in val_to_grad and forward_node.op != "placeholder":
            grad_value = g.call_function(
                _null_coalesce_accumulate,
                (val_to_grad[forward_node], grad_value),
            )
        val_to_grad[forward_node] = grad_value

    with g.inserting_before(output_node):
        for node in reversed(g.nodes):
            if node not in live_nodes:
                continue

            def add_to_live_nodes(n):
                live_nodes.setdefault(n, None)

            fx.node.map_arg(node.args, add_to_live_nodes)
            fx.node.map_arg(node.kwargs, add_to_live_nodes)
            if node.op == "call_module":
                output_grads: tuple[fx.Node | None, ...] | fx.Node | None
                if node in tuples:
                    stage_output = tuples[node]
                    output_grads = tuple(val_to_grad.get(n) for n in tuples[node])
                    outputs_with_grads_idxs = [
                        i for i, n in enumerate(tuples[node]) if n in live_nodes
                    ]
                else:
                    stage_output = (node,)
                    output_grads = val_to_grad[node]
                    outputs_with_grads_idxs = [0]

                output_grads = (
                    (output_grads,)
                    if not isinstance(output_grads, tuple)
                    else output_grads
                )

                grad_call = g.call_function(
                    stage_backward,
                    kwargs={
                        "stage_output": stage_output,
                        "output_grads": output_grads,
                        "input_values": list(node.all_input_nodes),
                        "outputs_with_grads_idxs": outputs_with_grads_idxs,
                    },
                )
                # Insert backward stage debug info
                kwargs_copy = dict(grad_call.kwargs)
                grad_call.kwargs = kwargs_copy

                grad_call_proxy = fx.Proxy(grad_call)
                grads = grad_call_proxy.node

                input_nodes = list(node.all_input_nodes)
                grads_proxy = fx.Proxy(grads)
                for i, input_node in enumerate(input_nodes):
                    assign_or_accumulate_grad(input_node, grads_proxy[i].node)  # type: ignore[index]

    return g


def _move_placeholders_to_front(graph: fx.Graph):
    """Ensure all placeholder nodes form a contiguous prefix of the graph.
    split_module can interleave get_attr nodes with placeholders."""
    last_ph = None
    for node in graph.nodes:
        if node.op == "placeholder":
            last_ph = node
    if last_ph is None:
        return
    nodes_to_move = []
    for node in graph.nodes:
        if node is last_ph:
            break
        if node.op != "placeholder":
            nodes_to_move.append(node)
    insert_point = last_ph
    for node in nodes_to_move:
        insert_point.append(node)
        insert_point = node


class PipeSequential(torch.nn.Sequential):
    @staticmethod
    def from_sequential(sequential_instance: torch.nn.Sequential):
        return PipeSequential(*[copy.copy(m) for m in sequential_instance])

    def forward(self, input):
        for i, module in enumerate(self):
            input = module(input)
            if i != len(self) - 1:
                pipe_split()
        return input


class LossWrapper(torch.nn.Module):
    """
    LossWrapper is a convenient abstract class that allows you to wrap up both
    your model as well as its loss function and specify the connectivity between
    the inputs, model, loss function, and output value. Example::

        class MyModelWrapper(LossWrapper):
            def forward(self, x, targets):
                model_out = self.module(x)
                loss_value = self.loss_fn(model_out, targets)
                return loss_value

    The above example defines a connectivity where we expect the forward/loss/backward
    training procedure to take two arguments (x and targets), pass x into the module
    to get the output of the feedforward computation, pass the model output and the
    targets value into the loss function, and get and return the loss value, which will
    be backpropagated by PiPPy. The above class would then be instantiated like::

        model = ...  # instantiate the model
        loss_fn = torch.nn.MSELoss()  # for the sake of demonstration

        wrapper = MyModelWrapper(model, loss_fn)
        pipe = Pipe.from_tracing(wrapper, ...)

    """

    def __init__(self, module, loss_fn):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This instance of LossWrapper does not have an overridden"
            "forward(). Please implement forward() to specify the arguments, "
            "connection between the module and loss, and loss output "
            "value."
        )


class TrivialLossWrapper(LossWrapper):
    # pyrefly: ignore [bad-override]
    def forward(self, x, targets):
        model_out = self.module(x)
        return self.loss_fn(model_out, targets)

    loss_spec = True


# Pipe model representation
#
# Pipe can be thought of as an `nn.Sequential++`. That is to say: it specifies
# a single topological ordering of pipeline "stages" that, when run in series,
# constitutes all of the operations of the program. However, unlike `nn.Sequential`,
# Pipe allows non-local usages of values, so long as those uses still respect
# topological ordering. In particular:
#
# 1. Non-local activations. This type of usage can appear in, for example, skip
#    connections. These values will be directly transmitted from the "def" stage
#    to all stages that use them skipping intermediate stages. During autograd,
#    gradients will be propagated back through this skip connection reverse
#    to how activations propagated in the forward pass.
# 2. Non-local parameter/module invocations. This occurs when a parameter is used
#    in a stage downstream of where it is resident. These values can be carried
#    forward similarly to (1), but in addition one might want to replicate the
#    value on multiple stages. Gradients for these shared parameters will be
#    accumulated separately on each stage, but there will be an additional
#    gradient accumulation before the optimizer step.


# Register `_pipe_split()` as an ATen operator. This is required for Export to
# preserve this marker in the graph.
torch.library.define("pippy::_pipe_split", "() -> ()")


@torch.library.impl("pippy::_pipe_split", "BackendSelect")
def _pipe_split():
    return None


@torch.library.register_fake("pippy::_pipe_split")  # type: ignore[no-redef]
def _pipe_split():
    return None


# Add an alias for convenience
aten_pipe_split_alias = torch.ops.pippy._pipe_split.default

# Ask Export to preserve the `_pipe_split` op.
# See examples in pytorch/torch/fx/node.py
fx.node._side_effectful_functions.add(aten_pipe_split_alias)


# User facing API
def pipe_split():
    """
    pipe_split is a special operator that is used to mark the boundary between
    stages in a module. It is used to split the module into stages. It is a
    no-op if your annotated module is run eagerly.

    Example:
        >>> # xdoctest: +SKIP
        >>> def forward(self, x):
        >>>     x = torch.mm(x, self.mm_param)
        >>>     x = torch.relu(x)
        >>>     pipe_split()
        >>>     x = self.lin(x)
        >>>     return x

    The above example will be split into two stages.
    """
    return torch.ops.pippy._pipe_split()


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2


MultiUseParamSpec = MultiUseParameterConfig | dict[str, MultiUseParameterConfig]


class DetachExecutor(fx.Interpreter):
    """
    Special interpreter to run the split_gm in testing that detaches all inputs to
    a module invocation. This is needed so that the values at the boundary are
    leaf modules in autograd execution.
    """

    def __init__(self, module, garbage_collect_values=True):
        garbage_collect_values = False
        super().__init__(module, garbage_collect_values)
        self.value_remap = {}

    def run(self, *args, initial_env=None):  # type: ignore[override]
        self.value_remap = {}
        return super().run(*args, initial_env=initial_env)

    def call_module(self, target, args, kwargs):
        def detach_tensors(a):
            if isinstance(a, torch.Tensor) and a.requires_grad:
                if a not in self.value_remap:
                    new_val = a.detach().requires_grad_(True)
                    self.value_remap[a] = new_val
                return self.value_remap[a]
            else:
                return a

        """
        def dont_traverse_size(a):
            return type(a) is not torch.Size
        """

        args = map_aggregate(
            args,
            detach_tensors,  # dont_traverse_size
        )
        kwargs = map_aggregate(
            kwargs,
            detach_tensors,  # dont_traverse_size
        )

        return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        # HACK to reroute saved input tensors to point to the detach()ed version
        if target is stage_backward:
            kwargs = dict(kwargs)
            kwargs["input_values"] = [
                self.value_remap.get(v, v) for v in kwargs["input_values"]
            ]
        return super().call_function(target, args, kwargs)


class _NodeReference:
    def __init__(self, name):
        self.name = name

    name: str


class _LinearNodeList:
    def __init__(self, node_list):
        self.serialize_node_list = []
        for node in node_list:
            node_args = fx.node.map_arg(node.args, lambda n: _NodeReference(n.name))  # type: ignore[arg-type,return-value]
            node_kwargs = fx.node.map_arg(node.kwargs, lambda n: _NodeReference(n.name))  # type: ignore[arg-type,return-value]
            serialize_node = fx.Node(
                graph=None,  # type: ignore[arg-type]
                name=node.name,
                op=node.op,
                target=node.target,
                args=node_args,  # type: ignore[arg-type]
                kwargs=node_kwargs,  # type: ignore[arg-type]
                return_type=node.type,
            )
            serialize_node.meta = copy.copy(node.meta)
            self.serialize_node_list.append(serialize_node)

    def to_graph(self):
        graph = fx.Graph()

        ref_str_to_node: dict[str, fx.Node] = {}

        def ref_to_node(arg):
            if isinstance(arg, _NodeReference):
                return ref_str_to_node[arg.name]
            else:
                return arg

        for node in self.serialize_node_list:
            node_args = map_aggregate(node.args, ref_to_node)
            node_kwargs = map_aggregate(node.kwargs, ref_to_node)
            deser_node = graph.create_node(
                op=node.op,
                target=node.target,
                args=node_args,  # type: ignore[arg-type]
                kwargs=node_kwargs,  # type: ignore[arg-type]
                name=node.name,
                type_expr=node.type,
            )
            ref_str_to_node[node.name] = deser_node

        return graph


def _direct_serialization_deserialize(body, nodes):
    """
    Custom `__reduce__` method for serialization.
    DO AS I SAY -- NOT AS I DO. This violates the principle that
    GraphModules serialize via code export & re-tracing. We allow
    for this here because **PIPE STAGES SHOULD NOT BE PERSISTED
    TO DISK -- THIS IS ONLY FOR TRANSMISSION VIA RPC**. Persisting
    these instances to disk will expose internal implementation
    details of `fx.Graph` and related data structures and is
    NOT advised.
    """

    class DummyModule(torch.nn.Module):
        def __init__(self, body):
            super().__init__()
            self.__dict__.update(body)

    dummy = DummyModule(body)

    return fx.GraphModule(dummy, nodes.to_graph())


def _direct_serialization_reduce(self):
    serialization_dict = dict(self.__dict__)
    serialization_dict.pop("_graph")
    return (
        _direct_serialization_deserialize,
        (serialization_dict, _LinearNodeList(self.graph.nodes)),
    )


def _modify_graph_op_device(
    gm: torch.fx.GraphModule,
    new_device: torch.device,
):
    """
    Modify the device argument of all "call_function" nodes in the graph.  This
    is useful for moving the graph to a different device. In particular for
    generator ops, like torch.ones.
    """
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if "device" in node.kwargs and node.kwargs["device"] != new_device:
                logger.debug(
                    f"Changing device of Node {node.name} from {node.kwargs['device']} to {new_device}"  # noqa: G004
                )
                node.update_kwarg("device", new_device)
                modified = True
        elif node.op == "call_module":
            # Recursively modify "device" in submodules
            submod = gm.get_submodule(node.target)
            if isinstance(submod, torch.fx.GraphModule):
                _modify_graph_op_device(submod, new_device)
            elif isinstance(submod, InterpreterModule):
                # If unflattening has been performed, we need to access its graph module by `.graph_module`
                _modify_graph_op_device(submod.graph_module, new_device)  # type: ignore[arg-type]
            else:
                logger.warning(
                    f"Skipping device modification for submodule {node.target} because it is a {type(submod)}"  # noqa: G004
                )

    if modified:
        gm.recompile()


class Pipe(torch.nn.Module):
    def __init__(
        self,
        split_gm: fx.GraphModule,
        num_stages: int,
        has_loss_and_backward: bool,
        loss_spec,
    ):
        # TODO: is there a way not to hard wire init?
        torch.nn.Module.__init__(self)
        self.split_gm: fx.GraphModule = split_gm
        self.executor: DetachExecutor = DetachExecutor(self.split_gm)
        self.num_stages: int = num_stages
        self.has_loss_and_backward = has_loss_and_backward
        self.loss_spec = loss_spec

        for node in split_gm.graph.nodes:
            if not (
                node.op in {"call_module", "placeholder", "output"}
                or (node.op, node.target) == ("call_function", operator.getitem)
                or (node.op, node.target) == ("call_method", "backward")
                or (node.op, node.target) == ("call_function", stage_backward)
                or (node.op, node.target)
                == ("call_function", _null_coalesce_accumulate)
            ):
                raise AssertionError(f"Unexpected node: {node}")

        # Detect replicated parameters so we know that we have to do an additional allreduce
        # before applying the optimizer
        #
        # Note that this also handles the case where there were multiple calls to a single
        # module from different stages, regardless of whether that module invocation
        # was handled by the logic above.

        # Map parameter value to a dictionary that maps the user pipeline module
        # to the local qualname within that module
        params_to_users: dict[torch.nn.Parameter, dict[str, str]] = {}

        for m_qualname, mod in self.split_gm.named_children():
            for p_qualname, param in mod.named_parameters():
                params_to_users.setdefault(param, {})
                params_to_users[param][m_qualname] = p_qualname

        self.replicated_params: list[dict[str, str]] = [
            use_mapping
            for use_mapping in params_to_users.values()
            if len(use_mapping) > 1
        ]

        # We must break the aliasing relationship between the replicated parameters for correct
        # numerics in reference runs. If we do not do this, the autograd tape in separate stages
        # will have a reference to the same tensor value and will erroneously apply gradient
        # updates multiple times. Therefore, for each replicated parameter set, we deepcopy the
        # values so that we have separate instances.
        for param_mapping in self.replicated_params:
            for submod_name, param_qualname in param_mapping.items():
                submod = getattr(self.split_gm, submod_name)
                atoms = param_qualname.split(".")
                for atom in atoms[:-1]:
                    submod = getattr(submod, atom)
                setattr(submod, atoms[-1], copy.deepcopy(getattr(submod, atoms[-1])))

        def throw(self, *args, **kwargs):
            raise RuntimeError(
                "To run pipeline locally, invoke the Pipe object directly, not `split_gm`"
            )

        self.split_gm.forward = throw

        # Make submodules use custom direct-serialized GraphModule
        i = 0
        while True:
            try:
                name = get_submod_name(i)
                submod = getattr(self.split_gm, name)
                submod.__class__.__reduce__ = _direct_serialization_reduce
                i += 1
            except AttributeError:
                break

    def forward(self, *args, **kwargs):
        executor_args = args
        if len(kwargs) > 0:
            parameters = []
            for node in self.split_gm.graph.nodes:
                if node.op == "placeholder":
                    if node.args and len(node.args) > 0:
                        parameters.append(
                            Parameter(
                                node.target,
                                Parameter.POSITIONAL_OR_KEYWORD,
                                default=node.args[0],
                            )
                        )
                    else:
                        parameter_kind = Parameter.POSITIONAL_OR_KEYWORD
                        param_name = node.target
                        if node.target.startswith("**"):
                            parameter_kind = Parameter.VAR_KEYWORD  # type: ignore[assignment]
                            param_name = param_name[2:]
                        elif node.target.startswith("*"):
                            parameter_kind = Parameter.VAR_POSITIONAL  # type: ignore[assignment]
                            param_name = param_name[1:]
                        parameters.append(Parameter(param_name, parameter_kind))
            signature = Signature(parameters)
            ba = signature.bind(*args, **kwargs)
            ba.apply_defaults()
            executor_args = ba.arguments.values()  # type: ignore[assignment]

        res = self.executor.run(*executor_args)

        return res

    def get_stage_module(self, stage_idx: int) -> torch.nn.Module:
        """
        Return a stage module corresponding to `stage_idx` of the `pipe`.
        """
        if stage_idx < 0 or stage_idx >= self.num_stages:
            raise ValueError(f"Invalid stage index {stage_idx}!")

        submod_name = get_submod_name(stage_idx)
        return getattr(self.split_gm, submod_name)

    @staticmethod
    def _number_and_count_forward_stages(gm: fx.GraphModule):
        num_stages = 0
        found_idxs: dict[int, None] = {}
        for node in gm.graph.nodes:
            if node.op == "call_module" and node.target.startswith(PP_SUBMOD_PREFIX):
                node.meta["stage_idx"] = int(node.target[len(PP_SUBMOD_PREFIX) + 1 :])
                found_idxs.setdefault(node.meta["stage_idx"])
                num_stages += 1

        # this assert will fail if a split point is inserted before the first layer, which creates empty first submodule
        # Update: the following assert may fail against some torch versions >=
        # 2.2.0, as:
        # submod_0, submod_1, submod_2, ...
        # may be named as
        # submod_0, submod_2, submod_4, ...
        # TODO: investigate
        # assert all(i in found_idxs for i in range(num_stages))

        return num_stages

    @staticmethod
    def _from_traced(
        mod: torch.nn.Module,
        exported_program: ExportedProgram,
        multi_use_param_spec: MultiUseParamSpec | None = None,
        output_loss_value_spec=None,
        split_policy: Callable[[torch.fx.GraphModule], torch.fx.GraphModule]
        | None = None,
    ):
        """
        Additionally, the ``output_loss_value_spec`` value can be specified to disambiguate
        which value in the output of `forward` is the loss value on which PiPPy should apply
        backpropagation. For example, if your ``forward`` returns a tuple ``(loss, model_out)``,
        you can specify ``output_loss_value_spec=(True, False)``. Or, if your ``forward`` returns
        a dict ``{'loss': loss_value, 'model_out': model_out}``, you can specify
        ``output_loss_value_spec={'loss': True, 'model_out': False}``
        """

        traced = exported_program.module(check_guards=False)

        if split_policy is not None:
            logger.info("Auto-splitting model")
            traced = split_policy(traced)  # type: ignore[arg-type]

        logger.debug(traced.print_readable(print_output=False))  # type: ignore[operator]

        # Deduplicate `get_attr` nodes that refer to the same parameter . Downstream code for moving
        # parameters relies on the invariant that parameter accesses happen once. This is not necessarily
        # the case (especially with custom tracers), so fix that up here.
        get_attr_nodes: dict[str, fx.Node] = {}
        for node in traced.graph.nodes:  # type: ignore[union-attr]
            if node.op == "get_attr":
                get_attr_nodes.setdefault(node.target, node)

                if get_attr_nodes[node.target] != node:
                    node.replace_all_uses_with(get_attr_nodes[node.target])
                    traced.graph.erase_node(node)  # type: ignore[operator, union-attr]

        # avoid looking at next node by keeping track of previous pipe_split
        prev_pipe_split_idx = -1
        pipe_split_nodes_to_erase = set()
        for i, node in enumerate(traced.graph.nodes):  # type: ignore[arg-type, union-attr]
            if (node.op, node.target) == ("call_function", pipe_split):
                if prev_pipe_split_idx == i - 1:
                    pipe_split_nodes_to_erase.add(node)
                prev_pipe_split_idx = i

        for node in pipe_split_nodes_to_erase:
            traced.graph.erase_node(node)  # type: ignore[operator, union-attr]

        traced.recompile()  # type: ignore[operator]

        part_idx = 0

        def split_callback(n: fx.Node):
            nonlocal part_idx
            if (n.op, n.target) == (
                "call_function",
                aten_pipe_split_alias,
            ):
                logger.debug(f"Found pipe_split {part_idx}")  # noqa: G004
                part_idx += 1
            return part_idx

        # TODO: what does split do with module invocations? does it move the modules
        # into the submodules?
        split = split_module(traced, mod, split_callback, partition_affix="pp")  # type: ignore[arg-type]
        # a (custom) tracer can produce dead code like orphan get_attr nodes
        split.graph.eliminate_dead_code()

        # peephole to remove pipe_split
        for submodule in split.modules():
            if isinstance(submodule, fx.GraphModule):
                for node in submodule.graph.nodes:
                    if (node.op, node.target) == (
                        "call_function",
                        aten_pipe_split_alias,
                    ):
                        submodule.graph.erase_node(node)
                submodule.recompile()

        for name, submodule in split.named_children():
            if isinstance(submodule, fx.GraphModule):
                _move_placeholders_to_front(submodule.graph)
                new_submod = _outline_submodules(submodule.graph)
                # Replace old submod
                split.register_module(name, new_submod)

        # TODO: backport this into split_module
        def delete_user_reference(node, user):
            """
            Delete reference of `node` from `user`'s arg list.
            Args:
                - node: a `get_attr` node at root.
                - user: a submodule node that uses `node`.
            """
            if not len(user.kwargs) == 0:
                raise AssertionError(
                    f"Expected user.kwargs to be empty, got {len(user.kwargs)}"
                )
            use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
            if not len(use_idxs) == 1:
                raise AssertionError(f"Expected 1 use index, got {len(use_idxs)}")
            args_copy = list(user.args)
            args_copy.pop(use_idxs[0])
            user.args = tuple(args_copy)
            logger.debug(
                f"Deleted {node} from user {user}, arg index = {use_idxs[0]}"  # noqa: G004
            )

        # A list of param referrals for deferred deletion.
        # To be accumulated in `move_param_to_callee`.
        to_delete = []

        def _recursive_getattr_with_parent(mod, fqn):
            # Returns getattr call given a nested FQN, and the last parent
            atoms = fqn.split(".")
            for atom in atoms[:-1]:
                if not hasattr(mod, atom):
                    return None, None
                mod = getattr(mod, atom)
            if not hasattr(mod, atoms[-1]):
                return mod, None
            attr = getattr(mod, atoms[-1])
            return mod, attr

        def move_param_to_callee(
            root,
            callee_name,
            param_fqn,
        ):
            """
            Move a parameter from the root module to a submodule.
            Args:
                root: The root module.
                callee_name: The name of the submodule to move the parameter to.
                param_fqn: The fully qualified name of the parameter to move.
            """
            # `atoms` is a list of strings representing the path to the
            # parameter in the original model
            atoms = param_fqn.split(".")
            mod_itr, param_val = _recursive_getattr_with_parent(split, param_fqn)
            # Check whether the parameter is a buffer or a parameter
            is_buffer = atoms[-1] in mod_itr._buffers

            # Check whether the parameter is a tensor
            if not isinstance(param_val, torch.Tensor):
                raise AssertionError(
                    f"Expected '{param_fqn}' to be {torch.Tensor} but got {type(param_val)}."
                    + (
                        f" It might happen if module '{param_fqn}' was passed to some 'leaf function'"
                        f"(see https://pytorch.org/docs/stable/fx.html#fx.wrap). Please inspect "
                        f"usages of '{param_fqn}' in the traced graph."
                        if isinstance(param_val, torch.nn.Module)
                        else ""
                    )
                )

            # Get submodule
            callee = root.get_submodule(callee_name)
            if hasattr(callee, param_fqn):
                raise AssertionError(
                    f"Module {callee_name} already has a parameter named {param_fqn}"
                )

            # Assign the parameter to the submodule
            if is_buffer:
                _assign_attr(
                    param_val,
                    callee,
                    param_fqn,
                    attr_kind=_AttrKind.BUFFER,
                    persistent=True,  # TODO: handle non-persistent buffer
                )
            else:
                _assign_attr(
                    param_val,
                    callee,
                    param_fqn,
                    attr_kind=_AttrKind.PARAMETER,
                )
            logger.debug(f"Moved parameter {param_fqn} to {callee_name}")  # noqa: G004

            # Next step is to replace placeholder of submodule with a get_attr.
            # Those placeholders are created by `split_module` inside each
            # submodule.
            # Update: this step is now moved to `_sink_params` because
            # `_sink_params` can do it recursively (i.e. for modules inside
            # submodule)

            to_delete.append((mod_itr, atoms[-1]))

        # Get the list of all parameters in the root module
        attr_nodes = list(filter(lambda n: n.op == "get_attr", split.graph.nodes))
        for node in attr_nodes:
            # Check whether the parameter is used in only one submodule
            if len(node.users) > 1:
                logger.info(
                    f"Parameter {node.target} used in multiple stages: {node.users}."  # noqa: G004
                )
            for user in node.users:
                if not user.op == "call_module":
                    raise AssertionError(
                        f"Expected user.op to be 'call_module', got {user.op}"
                    )
                # Move parameter into submodule
                move_param_to_callee(
                    split,
                    user.target,
                    node.target,
                )

        # [aliasing] store tensor id -> list of FQNs, built from state dict
        # Also assign non-persistent buffers
        id_to_fqns: dict[int, set[str]] = defaultdict(set)
        for fqn, tensor in mod.state_dict(keep_vars=True).items():
            id_to_fqns[id(tensor)].add(fqn)
        for fqn, tensor in mod.named_buffers():
            id_to_fqns[id(tensor)].add(fqn)

        # After moving the params to their corresponding hierarchies, we also
        # need to move the `get_attr` nodes from the root of the graph to those
        # hierarchies.
        # [aliasing] use id -> fqn mapping to list out all valid FQNs
        inputs_to_state: dict[str, list[str]] = {}
        for attr in attr_nodes:
            _, tensor = _recursive_getattr_with_parent(mod, attr.target)
            fqns = list(id_to_fqns[id(tensor)])
            if fqns:
                inputs_to_state[attr.name] = fqns
            elif attr.target in exported_program.constants:  # lifted constants
                inputs_to_state[attr.name] = [attr.target]

        # [aliasing] for each submodule split, assign attributes on FQNs that may be used.
        # We determine this based on whether or not the FQN attribute parent exists.
        # i.e. if the last submodule exists, assign the attribute.
        added_attributes: dict[str, list[str]] = defaultdict(list)
        for fqn, tensor in mod.state_dict(keep_vars=True).items():
            for name, submod in split.named_children():
                if isinstance(submod, fx.GraphModule):
                    parent, child = _recursive_getattr_with_parent(submod, fqn)
                    if (
                        parent and child is None
                    ):  # parent exists, attribute doesn't -> assign
                        added_attributes[name].append(fqn)
                        setattr(parent, fqn.split(".")[-1], tensor)

        # Deferral deletion: Remove the original attributes (to params) from the
        # root GraphModule
        for mod_itr, last_atom in to_delete:
            try:
                delattr(mod_itr, last_atom)
            except AttributeError:
                # This is expected if the parameter is used in multiple stages
                pass

        # This is done by (1) `_sink_params` at each submodule;
        for submod in split.children():
            if isinstance(submod, fx.GraphModule):
                _sink_params(submod, inputs_to_state, [])
                submod.graph.lint()
                submod.recompile()

        # [aliasing] This step is not super necessary, but helps reduce parameter usage/memory.
        # After _sink_params() routine has run, clean up unused attributes that we previously added.
        # Determine this based on the get_attr nodes - if not used, remove it.
        for name, attributes in added_attributes.items():
            submod = getattr(split, name)
            unused_attributes = set(attributes)
            # track used attributes in the submodule, running DFS on subgraph hierarchy
            stack = [("", submod)]  # (scope, submodule)
            while stack:
                scope, _mod = stack.pop()
                if isinstance(_mod, (fx.GraphModule, InterpreterModule)):
                    for node in _mod.graph.nodes:
                        if node.op == "get_attr":
                            # get_attr might get access deeper level attribute
                            fqn = scope + "." + node.target if scope else node.target
                            unused_attributes.discard(fqn)
                for _name, _submod in _mod.named_children():
                    stack.append((scope + "." + _name if scope else _name, _submod))
            # delete unused attributes
            for attr in unused_attributes:
                mod_itr, atoms = submod, attr.split(".")
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)
                delattr(mod_itr, atoms[-1])

        for node in attr_nodes:
            # And (2): remove `get_attr` node from submod's arg list
            for user in copy.copy(node.users):
                if not user.op == "call_module":
                    raise AssertionError(
                        f"Expected user.op to be 'call_module', got {user.op}"
                    )
                delete_user_reference(node, user)
            # And (3): remove the `get_attr` node from the root graph.
            split.graph.erase_node(node)

        split.delete_all_unused_submodules()
        split.graph.lint()
        split.recompile()

        num_stages = Pipe._number_and_count_forward_stages(split)

        has_loss_and_backward = False
        generated_loss_spec = output_loss_value_spec

        if output_loss_value_spec is not None:
            loss_node, output_node, generated_loss_spec = _find_loss_output(
                mod, split.graph, output_loss_value_spec
            )
            if loss_node is not None:
                _insert_stage_symbolic_backward(
                    split.graph,
                    loss_node,
                    output_node,
                )
                split.recompile()
                has_loss_and_backward = True
                logger.debug("Pipeline is in training mode, backward pass generated")
            else:
                raise RuntimeError(
                    f"Did not find any loss value according to {output_loss_value_spec=}"
                )
        else:
            logger.debug("Pipeline is in inference mode, backward pass not generated")

        logger.debug(f"Full pipe model:\n{split}")  # noqa: G004

        return Pipe(
            split,
            num_stages,
            has_loss_and_backward,
            generated_loss_spec,
        )

    def print_readable(self):
        """
        Print the pipe in a human-readable format.
        This will print both the root pipe and each stage module.
        """
        self.split_gm.print_readable()

    @staticmethod
    def _trace_with_export(
        mod: torch.nn.Module,
        example_args: tuple[Any, ...],
        example_kwargs: dict[str, Any] | None = None,
    ) -> ExportedProgram:
        logger.info("Tracing model ...")
        try:
            ep = torch.export.export(mod, example_args, example_kwargs)
        except Exception as e:
            raise RuntimeError(
                "It seems that we cannot capture your model as a full graph. "
                "Typical reasons include graph breaks, data/shape-dependent "
                "control flow, or missing meta kernels for custom operators. "
                "You can use our manual pipeline interfaces, or try to fix the "
                "graph breaks, see https://pytorch.org/docs/stable/export.html"
            ) from e

        return ep

    @staticmethod
    def from_tracing(
        mod: torch.nn.Module,
        example_args: tuple[Any, ...],
        example_kwargs: dict[str, Any] | None = None,
        split_policy: Callable[[fx.GraphModule], fx.GraphModule] | None = None,
    ):
        # If a param will be used in multiple pipeline stages, we default the strategy to REPLICATE'ing the param across
        # stages instead of TRANSMIT'ting it
        multi_use_param_spec = MultiUseParameterConfig.REPLICATE

        # Figure out which output is loss from output_chunk_spec
        output_loss_value_spec: Any = None
        # Deprecated
        """
        if output_chunk_spec is not None:
            output_loss_value_spec = map_aggregate(
                output_chunk_spec, lambda v: isinstance(v, _LossReducer)
            )
        """

        # Trace with export
        exported_program = Pipe._trace_with_export(
            mod,
            example_args,
            example_kwargs,
        )

        pipe = Pipe._from_traced(
            mod,
            exported_program,
            multi_use_param_spec,
            output_loss_value_spec=output_loss_value_spec,
            split_policy=split_policy,
        )

        # Users want the first pipeline stage to accept kwargs if the original
        # program does. This is controlled by the `_codegen` field of the graph,
        # so we make a copy here. Note: we only want the input spec and not the
        # output spec, because the output spec is for the last stage. Maybe a
        # TODO? Not sure yet.
        split = pipe.split_gm
        traced = exported_program.module()
        submod0 = next(iter(split.children()))
        submod0_sign = signature(submod0.forward)
        model_sign = signature(traced.forward)
        if len(model_sign.parameters) != len(submod0_sign.parameters):
            # We don't change the signature of the first stage if it takes
            # different number of args than original model
            logger.info(
                f"Original model takes {len(model_sign.parameters)} args but the "  # noqa: G004
                f"first pipeline stage takes {len(submod0_sign.parameters)}. "
                "Please provide args to respective pipeline stages."
            )
        else:
            # Support kwargs for the first stage
            submod0.graph._codegen = copy.deepcopy(traced.graph._codegen)  # type: ignore[union-attr]
            # `_replace` is actually not "private" or internal. based on this doc:
            # To prevent conflicts with field names, the method and attribute names
            # start with an underscore
            submod0.graph._codegen.pytree_info = (  # type: ignore[union-attr]
                submod0.graph._codegen.pytree_info._replace(out_spec=None)  # type: ignore[operator, union-attr]
            )
            submod0.recompile()

        return pipe

    def __str__(self):
        return self.split_gm.__str__()

    def __repr__(self):
        return self.split_gm.__repr__()

    def info(self) -> PipeInfo:
        """
        Get information about the pipe.

        Returns
        -------
        PipeInfo
            A dataclass containing information about the pipe.
        """
        return PipeInfo(
            graph=self.split_gm.graph,
            num_stages=self.num_stages,
            has_loss_and_backward=self.has_loss_and_backward,
        )

    def build_stage(
        self,
        stage_index: int,
        device: torch.device,
        group: ProcessGroup | None = None,
    ) -> _PipelineStage:
        """
        Create a `PipelineStage` given a stage index and distributed group.
        The `PipelineStage` can run with `PipelineSchedule`s.
        """
        # Find stage module
        stage_module = self.get_stage_module(stage_index)

        # Move ops argument to device
        # Today PT2 tracer does not treat `x.device` as a symbolic device;
        # instead, the device of tracing time got burned into the generated
        # code.  Here we provide a workaround for users to manually modify the
        # "device" kwarg of operations. Such operation may include:
        # `torch.ones`, `torch.zeros`, `torch.rand`, etc.
        if isinstance(stage_module, torch.fx.GraphModule):
            _modify_graph_op_device(stage_module, device)
        else:
            logger.warning(
                f"Expected a `torch.fx.GraphModule` but got {type(stage_module)}"  # noqa: G004
            )

        # Detach pipe info
        # Note: be careful what's included in `pipe_info`. We don't want to keep
        # a reference to `Pipe` or `Pipe.split_gm` which stops python from
        # recycling them. When python recycles them, other stage modules (which
        # are irrelevant to current rank) can be automatically freed.
        pipe_info = self.info()
        return _PipelineStage(stage_module, stage_index, pipe_info, device, group)


class SplitPoint(Enum):
    """
    Enum representing the points at which a split can occur in the execution of a submodule.
    Attributes:
        BEGINNING: Represents adding a split point *before* the execution of a certain submodule in the `forward` function.
        END: Represents adding a split point *after* the execution of a certain submodule in the `forward` function.
    """

    BEGINNING = 1
    END = 2


# For backward compatibility, we kept the PipeSplitWrapper class because `class
# SplitPoint` used to be defined in this class.
class PipeSplitWrapper:
    # Create a class alias for BC
    SplitPoint = SplitPoint


def _split_before_forward(self, *args, **kwargs):
    pipe_split()
    return self._orig_forward(*args, **kwargs)


def _split_after_forward(self, *args, **kwargs):
    try:
        return self._orig_forward(*args, **kwargs)
    finally:
        pipe_split()


def annotate_split_points(mod: torch.nn.Module, spec: dict[str, SplitPoint]):
    # TODO: make this implementation out-of-place?
    for qualname, split_type in spec.items():
        atoms = qualname.split(".")
        predecessor_module = mod
        for i, atom in enumerate(atoms[:-1]):
            try:
                predecessor_module = getattr(predecessor_module, atom)
            except AttributeError as e:
                raise AttributeError(
                    f"Specified target {qualname} referenced "
                    f"nonexistent module {'.'.join(atoms[: i + 1])}"
                ) from e

        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        mod_to_wrap._orig_forward = mod_to_wrap.forward
        if split_type == SplitPoint.BEGINNING:
            mod_to_wrap.forward = MethodType(_split_before_forward, mod_to_wrap)
        elif split_type == SplitPoint.END:
            mod_to_wrap.forward = MethodType(_split_after_forward, mod_to_wrap)
        else:
            raise ValueError("Unknown split point type.")


def pipeline(
    module: torch.nn.Module,
    mb_args: tuple[Any, ...],
    mb_kwargs: dict[str, Any] | None = None,
    split_spec: dict[str, SplitPoint] | None = None,
    split_policy: Callable[[fx.GraphModule], fx.GraphModule] | None = None,
) -> Pipe:
    """
    Split a module based on a specification.

    See `Pipe` for more details.

    Arguments
    ---------
    module:
        The module to be split.
    mb_args:
        Example positional inputs, in micro-batch form.
    mb_kwargs:
        Example keyword inputs, in micro-batch form. (default: `None`)
    split_spec:
        A dictionary using submodule names as split marker. (default: `None`)
    split_policy:
        The policy to use for splitting the module. (default: `None`)

    Returns
    -------
    A pipeline representation of class `Pipe`.
    """
    if split_spec is not None and split_policy is not None:
        raise ValueError(
            "Cannot specify both `split_spec` and `split_policy`. Please use only one of them."
        )

    if split_spec is not None:
        # Annotate split points in the module based on user spec
        annotate_split_points(module, split_spec)
        return Pipe.from_tracing(
            mod=module,
            example_args=mb_args,
            example_kwargs=mb_kwargs,
        )
    else:
        # Use split policy
        return Pipe.from_tracing(
            mod=module,
            example_args=mb_args,
            example_kwargs=mb_kwargs,
            split_policy=split_policy,
        )
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

"""
This visualizer requires matplotlib to be installed.

Example usage:

ops = get_schedule_ops("InterleavedZeroBubble", 4, 8)
visualize_schedule(ops, "test.png")
"""

import collections
from typing import NamedTuple
from unittest import mock

from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import PipelineStage


class OpKey(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: int


def get_schedule_ops(
    schedule: str | type[_PipelineSchedule],
    pp_degree: int,
    num_microbatches: int,
    num_stages_per_rank: int | None = None,
    add_spacing: bool = False,
    with_comms: bool = False,
    defer_pp_recv: bool = False,
) -> list[list[_Action | None]]:
    """
    Get all actions for a given schedule, pp_degree, and num_microbatches. The actions are returned in a list of lists
    where each inner list represents a rank and each element in the inner list represents an action.

    The schedule can be specified as a string which is passed into get_schedule_class() or a _PipelineSchedule instance.

    Args:
        defer_pp_recv: If True, RECV ops are deferred to right before the consuming
            compute op. If False (default), RECV ops are placed as early as possible to
            overlap P2P communication with computation. Only takes effect when with_comms=True.
    """
    if add_spacing and with_comms:
        raise ValueError("Cannot add spacing and view comms at the same time")

    if isinstance(schedule, str):
        schedule_class = get_schedule_class(schedule)
    elif issubclass(schedule, _PipelineSchedule):
        schedule_class = schedule
    else:
        raise ValueError(f"Invalid schedule: {schedule}")

    # Create a mock of the PipelineStage class
    mock_pipeline_stage = mock.create_autospec(PipelineStage, instance=True)
    # Set the return values for group_rank and group_size methods
    mock_pipeline_stage.group_rank = 0
    mock_pipeline_stage.group_size = pp_degree
    mock_pipeline_stage.submod = None

    # Check num_stages_per_rank is valid
    if issubclass(schedule_class, PipelineScheduleSingle):
        if num_stages_per_rank is None:
            num_stages_per_rank = 1
        if not num_stages_per_rank == 1:
            raise AssertionError(
                f"Expected num_stages_per_rank to be 1, got {num_stages_per_rank}"
            )
        stages = mock_pipeline_stage
        stages.num_stages = num_stages_per_rank * pp_degree
    elif issubclass(schedule_class, PipelineScheduleMulti):
        if num_stages_per_rank is None:
            num_stages_per_rank = 2
        if not num_stages_per_rank >= 2:
            raise AssertionError(
                f"Expected num_stages_per_rank >= 2, got {num_stages_per_rank}"
            )
        stages = [mock_pipeline_stage for _ in range(num_stages_per_rank)]
        for stage in stages:
            stage.num_stages = num_stages_per_rank * pp_degree

    else:
        raise ValueError(f"Invalid schedule: {schedule_class}")

    # Instantiate the schedule class
    # pyrefly: ignore [bad-argument-type]
    schedule_instance = schedule_class(stages, num_microbatches)
    if schedule_instance.pipeline_order is None:
        raise AssertionError("Expected pipeline_order to not be None")

    # Convert to List[List[_Action]]
    all_actions: list[list[_Action | None]] = []
    if with_comms:
        runtime = _PipelineScheduleRuntime(
            stages, num_microbatches, defer_pp_recv=defer_pp_recv
        )
        runtime._prepare_schedule_with_comms(schedule_instance.pipeline_order)
        for rank in range(pp_degree):
            all_actions.append(list(runtime.pipeline_order_with_comms[rank]))
    else:
        for rank in range(pp_degree):
            all_actions.append(schedule_instance.pipeline_order[rank])

    # Add spacing
    if add_spacing:
        # remove all Nones, then respace
        # TODO: later we can change this at the schedule creation level to not use Nones
        all_actions = [
            [action for action in rank if action is not None] for rank in all_actions
        ]
        all_actions = add_schedule_op_spacing(all_actions)

    # Return the pipeline order
    return all_actions


class _ComputationTypeVisual:
    def __init__(
        self,
        color: str,
        text: str = "",
        width: int = 1,
    ):
        self.color = color
        self.width = width
        self.text = text


# Update the mapping to use _ComputationTypeVisual instances
action_type_to_color_mapping = {
    _ComputationType.FORWARD: _ComputationTypeVisual("blue", "Forward"),
    _ComputationType.BACKWARD_INPUT: _ComputationTypeVisual("teal", "Backward Input"),
    _ComputationType.BACKWARD_WEIGHT: _ComputationTypeVisual(
        "green", "Backward Weight"
    ),
    _ComputationType.FULL_BACKWARD: _ComputationTypeVisual(
        "orange", "Full Backward", 2
    ),
    _ComputationType.OVERLAP_F_B: _ComputationTypeVisual("purple", "Overlap F+B", 3),
}


def add_schedule_op_spacing(
    schedule: list[list[_Action | None]],
) -> list[list[_Action | None]]:
    """
    Add spacing to the schedule based on dependencies between ranks.

    Before adding an operation to the list, this function checks if there are
    dependencies from other ranks. If there are dependencies (other ranks have
    not finished processing the required microbatch), it adds None instead.

    For example, Forward microbatch 0 on rank 1 depends on rank 0 processing
    Forward microbatch 0 first.

    Args:
        schedule: The original schedule as a list of lists where each inner list
                 represents a rank and each element represents an action.

    Returns:
        A new schedule with proper spacing based on dependencies.
    """
    if not schedule:
        return schedule

    num_stages = (
        max(
            action.stage_index
            for rank_actions in schedule
            for action in rank_actions
            if action is not None
        )
        + 1
    )

    num_ranks = len(schedule)
    spaced_schedule: list[list[_Action | None]] = [[] for _ in range(num_ranks)]
    rank_ops = [collections.deque(ops) for ops in schedule]

    # Track completion times: (stage_index, action_type, microbatch_index) -> completion_time
    scheduled_ops: dict[OpKey, int] = {}

    def is_dependency_ready(dependency_key: OpKey, timestep: int) -> bool:
        """Check if a dependency operation has completed by the given timestep."""
        return (
            dependency_key in scheduled_ops
            and timestep >= scheduled_ops[dependency_key]
        )

    def get_dependencies(action: _Action) -> list[OpKey]:
        """Get the list of dependencies for an action."""
        stage_idx = action.stage_index
        comp_type = action.computation_type
        mb_idx = action.microbatch_index

        # Ensure mb_idx is not None for dependency tracking
        if mb_idx is None:
            raise AssertionError(f"Action {action} has None microbatch_index")

        # First stage forward has no dependencies
        if stage_idx == 0 and comp_type == _ComputationType.FORWARD:
            return []

        # Last stage backward depends on forward from previous stage
        if stage_idx == num_stages - 1 and comp_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
        ):
            return [OpKey(stage_idx - 1, _ComputationType.FORWARD, mb_idx)]

        # Forward depends on previous stage forward
        if comp_type == _ComputationType.FORWARD:
            return [OpKey(stage_idx - 1, _ComputationType.FORWARD, mb_idx)]

        # Backward depends on next stage backward
        if comp_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
        ):
            return [
                OpKey(stage_idx + 1, _ComputationType.FULL_BACKWARD, mb_idx),
                OpKey(stage_idx + 1, _ComputationType.BACKWARD_INPUT, mb_idx),
            ]

        # Weight backward depends on input backward
        if comp_type == _ComputationType.BACKWARD_WEIGHT:
            return [OpKey(stage_idx, _ComputationType.BACKWARD_INPUT, mb_idx)]

        raise RuntimeError(f"Unknown computation type: {comp_type}")

    def is_action_ready(action: _Action, timestep: int) -> bool:
        """Check if an action is ready to be scheduled at the given timestep."""
        # For OR dependencies (like backward), check if any dependency is satisfied
        if action.computation_type in (
            _ComputationType.FULL_BACKWARD,
            _ComputationType.BACKWARD_INPUT,
            _ComputationType.BACKWARD_WEIGHT,
        ):
            dependencies = get_dependencies(action)
            return any(is_dependency_ready(dep, timestep) for dep in dependencies)
        # For AND dependencies, all must be satisfied
        elif action.computation_type == _ComputationType.FORWARD:
            dependencies = get_dependencies(action)
            return all(is_dependency_ready(dep, timestep) for dep in dependencies)
        elif action.computation_type == _ComputationType.OVERLAP_F_B:
            if action.sub_actions is None:
                raise AssertionError(
                    f"OVERLAP_F_B action {action} has None sub_actions"
                )
            dep_list: list[bool] = []
            for sub_action in action.sub_actions:
                dep_list.append(is_action_ready(sub_action, timestep))
            return all(dep_list)
        else:
            raise RuntimeError(f"Unknown computation type: {action.computation_type}")

    def schedule_action(action: _Action, rank: int, timestep: int) -> int:
        """Schedule an action and return completion time."""
        spaced_schedule[rank].append(action)
        comp_type = action.computation_type
        comp_time = action_type_to_color_mapping[comp_type].width
        completion_time = timestep + comp_time

        if comp_type == _ComputationType.OVERLAP_F_B:
            # For overlap actions, schedule each sub-action with cumulative timing
            if action.sub_actions is None:
                raise AssertionError(
                    f"OVERLAP_F_B action {action} has None sub_actions"
                )
            cumulative_time = 0
            for sub_action in action.sub_actions:
                if sub_action.microbatch_index is None:
                    raise AssertionError(
                        f"Sub-action {sub_action} has None microbatch_index"
                    )
                sub_comp_time = action_type_to_color_mapping[
                    sub_action.computation_type
                ].width
                cumulative_time += sub_comp_time
                scheduled_ops[
                    OpKey(
                        sub_action.stage_index,
                        sub_action.computation_type,
                        sub_action.microbatch_index,
                    )
                ] = timestep + cumulative_time
        else:
            if action.microbatch_index is None:
                raise AssertionError(f"Action {action} has None microbatch_index")
            scheduled_ops[
                OpKey(action.stage_index, comp_type, action.microbatch_index)
            ] = completion_time

        return completion_time

    # Main scheduling loop
    current_timestep = 0
    timesteps_without_progress = 0
    rank_completion_times = dict.fromkeys(range(num_ranks), 0)
    while rank_ops:
        print(f"Current timestep: {current_timestep}")
        # Process all operations during timestep until we run out of ready operations
        for rank, op_queue in enumerate(rank_ops):
            if not op_queue:
                continue

            op_queue = rank_ops[rank]
            action = op_queue[0]
            print(f"Rank: {rank}, {action=}")
            if action is None:
                spaced_schedule[rank].append(None)
                op_queue.popleft()
                timesteps_without_progress = 0
            elif current_timestep >= rank_completion_times[rank] and is_action_ready(
                action, current_timestep
            ):
                rank_completion_times[rank] = schedule_action(
                    action, rank, current_timestep
                )
                op_queue.popleft()
                timesteps_without_progress = 0

        # Add None for ranks that are waiting
        for rank in range(num_ranks):
            if current_timestep >= rank_completion_times[rank]:
                spaced_schedule[rank].append(None)

        # Remove empty queues and advance timestep
        rank_ops = [op_queue for op_queue in rank_ops if op_queue]
        current_timestep += 1
        timesteps_without_progress += 1

        if timesteps_without_progress > max(
            visual.width for visual in action_type_to_color_mapping.values()
        ):
            raise RuntimeError("No progress made in scheduling - possible deadlock")

    return spaced_schedule


def visualize_schedule(
    schedule: list[list[_Action | None]],
    filename: str | None = None,
) -> None:
    """
    Visualize the schedule using matplotlib.
    The schedule is a list of lists where each inner list represents a rank and each element in the inner list represents an action.
    The actions are represented as rectangles with different colors based on their computation type.
    The filename is optional and if provided, the plot will be saved to that file.

    Args:
        schedule: The schedule to visualize.
        filename: The filename to save the plot to. If not provided, the plot will be displayed.
        add_schedule_spacing: If True, add spacing to the schedule based on dependencies between ranks.

    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.family"] = (
        "DejaVu Sans"  # or any other font available on your system
    )
    num_ranks = len(schedule)
    max_actions = max(len(rank) for rank in schedule)

    # Increase the figure size to provide more space for the legend
    fig, ax = plt.subplots(figsize=(max_actions + 2, num_ranks + 2))
    max_draw_position = -1
    # Calculate dynamic font size based on figure size
    font_size = min(max_actions, num_ranks) + 4
    used_computation = set()
    for rank_idx, actions in enumerate(schedule):
        draw_position = 0  # Initialize drawing position for each rank
        for action in actions:
            if action is not None:
                comp_type_color = action_type_to_color_mapping.get(
                    action.computation_type, _ComputationTypeVisual("black")
                )
                used_computation.add(action.computation_type)
                color = comp_type_color.color
                width = comp_type_color.width

                # Check if action has sub_actions to determine styling
                if action.sub_actions is not None:
                    linewidth = 2  # Thicker border for compound actions
                    text_weight = "normal"  # Bold text for compound actions
                else:
                    linewidth = 1  # Default linewidth for regular actions
                    text_weight = "normal"  # Default text weight

                # Draw the rectangle to represent the action duration
                rect = Rectangle(
                    (draw_position, num_ranks - rank_idx - 1),
                    width,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=linewidth,
                )
                ax.add_patch(rect)

                # Draw the text centered within the rectangle
                ax.text(
                    draw_position + width / 2,
                    num_ranks - rank_idx - 1 + 0.5,
                    str(action),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="white",
                    weight=text_weight,
                )

                draw_position += width
            else:
                draw_position += 1  # Move to the next
            max_draw_position = max(max_draw_position, draw_position)
    ax.set_xlim(-0.5, max_draw_position + 1)
    ax.set_ylim(-0.5, num_ranks + 0.5)  # Add extra space at the top
    # Set y-ticks to be in the middle of each rank's row
    ax.set_yticks([num_ranks - rank_idx - 0.5 for rank_idx in range(num_ranks)])
    ax.set_yticklabels([f"Rank {i}" for i in range(num_ranks)], fontsize=font_size)
    ax.set_xticklabels([])

    # Remove grid lines and ticks
    ax.grid(False)
    # Add legend with larger font size
    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=action_type_to_color_mapping[comp_type].color,
            edgecolor="black",
            label=action_type_to_color_mapping[comp_type].text,
        )
        for comp_type in used_computation
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=font_size)
    # Save to file if filename is provided, otherwise display the plot
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
from collections import defaultdict

import torch
from torch.export.unflatten import _ModuleFrame, _SubmoduleEntry


def _outline_submodules(orig_graph: torch.fx.Graph) -> torch.fx.GraphModule:
    # Create an empty GraphModule to hold the outlined modules
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    seen_nodes: dict[str, torch.fx.Node] = {}
    seen_modules: dict[int, list[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: dict[str, set[str]] = defaultdict(set)
    created_modules: dict[str, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", None, 0)],
        "",
        {},
        module=new_module,
    ).run_outer()
    new_module.graph.lint()
    new_module.recompile()
    return new_module
<|endoftext|># Keep old package for BC purposes, this file should be removed once
# everything moves to the `torch.distributed._shard` package.
import sys
import warnings

import torch
from torch.distributed._shard.sharded_tensor import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`torch.distributed._sharded_tensor` will be deprecated, "
        "use `torch.distributed._shard.sharded_tensor` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["torch.distributed._sharded_tensor"] = (
    torch.distributed._shard.sharded_tensor
)
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import contextlib
import inspect
import os
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import cast, TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch.distributed as dist
from torch.distributed._state_dict_utils import STATE_DICT_TYPE
from torch.distributed.checkpoint._async_process_executor import (
    _ProcessBasedAsyncCheckpointExecutor,
)
from torch.distributed.checkpoint._async_thread_executor import (
    _ThreadBasedAsyncCheckpointExecutor,
)
from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.staging import (
    AsyncStager,
    DefaultStager,
    StagingOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.storage import StorageWriter, WriteResult
from torch.distributed.distributed_c10d import _get_default_group

from .utils import _api_bc_check, _DistWrapper, _profile


if TYPE_CHECKING:
    from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor


__all__ = [
    "save_state_dict",
    "save",
    "async_save",
    "AsyncCheckpointerType",
    "AsyncSaveResponse",
]


class AsyncCheckpointerType(Enum):
    """Enum for async checkpointer type."""

    THREAD = "thread"
    PROCESS = "process"


@deprecated(
    "`save_state_dict` is deprecated and will be removed in future versions."
    "Please use `save` instead.",
    category=FutureWarning,
)
def save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: dist.ProcessGroup | None = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: SavePlanner | None = None,
) -> Metadata:
    """This method is deprecated. Please switch to 'save'."""
    storage_writer.reset()

    # TODO: test returning `save` here instead.
    with _profile():
        return _save_state_dict(
            state_dict,
            storage_writer,
            process_group,
            coordinator_rank,
            no_dist,
            planner,
        )


@_dcp_method_logger(log_exceptions=True)  # type: ignore[arg-type]
@_api_bc_check
def save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_writer: StorageWriter | None = None,
    planner: SavePlanner | None = None,
    process_group: dist.ProcessGroup | None = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Metadata:
    """
    Save a distributed model in SPMD style.

    This function is different from ``torch.save()`` as it handles
    ``ShardedTensor`` , and ``DTensor`` by having each rank only save their local shards.

    For each ``Stateful`` object (having both a ``state_dict`` and a ``load_state_dict``),
    save will call ``state_dict`` before serialization.

    .. warning::
        There are no guarantees of Backwards Compatibility across PyTorch versions
        for saved state_dicts.

    .. warning::
        If using the `process_group` argument, make sure that only its ranks
        call `save_state_dict` and that all data in state_dict belong to it.

    .. note::
        When saving checkpoint for FSDP's `ShardingStrategy.HYBRID_SHARD`, only one of
        the shard_group should be calling `save_state_dict` and the corresponding process
        group needs to be passed in.

    .. note::
        If no process group is available, this function assumes the intention is to save the
         state_dict in the local process.

    .. note:
        Rank 0 is assumed to be the coordinator rank.


    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_writer (Optional[StorageWriter]):
            Instance of StorageWriter used to perform writes. If this is not
            specified, DCP will automatically infer the writer based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[SavePlanner]):
            Instance of SavePlanner. If this is not specified, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        no_dist (bool):
            If ``True``, this function will assume the intent is to load
            a checkpoint on a single rank/process.
            (Default: ``False``)
        use_collectives (bool): If ``False``, this function will assume the intent is to save
            a checkpoint without using cross-rank synchronization.
            (Default: ``True``)
            This configuration is experimental and should be used with caution.
            It will change the format of the saved checkpoint and may not be backward compatible.

    Returns:
        Metadata: Metadata object for the saved checkpoint.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> state_dict = {"model": my_model}

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(
        ...     "/checkpoint/1"
        ... )
        >>> torch.distributed.checkpoint.save(
        >>>     state_dict=state_dict,
        >>>     storage_writer=fs_storage_writer,
        >>> )

    .. note::
        save_state_dict uses collectives to coordinate writes across ranks.
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication takes place.
        In this case, the device used is given by ``torch.cuda.current_device()``
        and it is the user's responsibility to ensure that this is set so that
        each rank has an individual GPU, via ``torch.cuda.set_device()``.
    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save")

    no_dist = no_dist or (not dist.is_available()) or (not dist.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process.",
            stacklevel=2,
        )

    with _profile():
        storage_writer = cast(
            StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False)
        )

        return _save_state_dict(
            state_dict=_stateful_to_state_dict(state_dict),
            storage_writer=storage_writer,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
            use_collectives=use_collectives,
        )


@dataclass
class AsyncSaveResponse:
    """This class contains futures for staging and upload completion.
    It is returned by async_save().
    staging_completion is a future that indicates when local copy
    of state_dict is complete.
    upload_completion is a future that indicates when a checkpoint
    completed saving.
    """

    staging_completion: Future[None]
    upload_completion: Future[None]


@_dcp_method_logger(log_exceptions=True)
def async_save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_writer: StorageWriter | None = None,
    planner: SavePlanner | None = None,
    process_group: dist.ProcessGroup | None = None,
    async_checkpointer_type: AsyncCheckpointerType = AsyncCheckpointerType.THREAD,
    async_stager: AsyncStager | None = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Future | AsyncSaveResponse:
    """Asynchronous version of ``save``. This code first de-stages the state_dict on to the
    staging storage (defaults to CPU memory), and then calls the `save` in a separate thread.

    .. warning::
        This feature is experimental and subject to change.
        If you provide an ``async_stager``, call ``close()`` after the last
        checkpoint is saved. Internally created default stagers are closed
        automatically.

    Args:
        state_dict (Dict[str, Any]): The state_dict to save.
        checkpoint_id (Union[str, os.PathLike, None]):
            The ID of this checkpoint instance. The meaning of the checkpoint_id
            depends on the storage. It can be a path to a folder or to a file.
            It can also be a key if the storage is a key-value store.
            (Default: ``None``)
        storage_writer (Optional[StorageWriter]):
            Instance of StorageWriter used to perform 'stage' and  'save'. If
            this is not specified, DCP will automatically infer the writer based on the
            checkpoint_id. If checkpoint_id is also None, an exception will
            be raised. (Default: ``None``)
        planner (Optional[SavePlanner]):
            Instance of SavePlanner. If this is not specified, the default
            planner will be used. (Default: ``None``)
        process_group (Optional[ProcessGroup]):
            ProcessGroup to be used for cross-rank synchronization.
            (Default: ``None``)
        async_checkpointer_type (AsyncCheckpointerType):
            whether to do checkpoint in separate thread or process
            (Default: ``AsyncCheckpointerType.THREAD``)
        async_stager (AsyncStager):
            provides staging implementation. If ``storage_writer`` implements
            ``AsyncStager`` and ``async_stager`` is not provided, ``storage_writer``
            will be used for staging. User-provided stagers remain owned by the
            caller and should be closed after the last checkpoint is saved.
        no_dist (bool):
            If ``True``, this function will assume the intent is to save
            a checkpoint on a single rank/process.
            (Default: ``False``)
        use_collectives: If False, Save the checkpoint without rank coordination. (Default: ``True``)
            This configuration is experimental and should be used with caution.
            It will change the format of the saved checkpoint and may not be backward compatible.

    Returns:
        Future: A future holding the resultant Metadata object from `save`.

    Example:
        >>> # xdoctest: +SKIP
        >>> my_model = MyModule()

        >>> state_dict = {"model": my_model}

        >>> fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(
        ...     "/checkpoint/1"
        ... )
        >>> checkpoint_future = torch.distributed.checkpoint.async_save(
        >>>     state_dict=state_dict,
        >>>     storage_writer=fs_storage_writer,
        >>> )
        >>>
        >>> # ... do some work ...
        >>>
        >>> checkpoint_future.result()

    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint.async_save")

    if dist.is_available() and dist.is_initialized():
        pg = process_group or _get_default_group()
        if torch.device("cpu") not in pg._device_types:
            raise AssertionError(
                "A CPU backend must be enabled for async save; try initializing process group with 'cpu:gloo,cuda:nccl'"
            )

    owned_async_stager: AsyncStager | None = None
    if async_stager is None:
        if storage_writer is not None and isinstance(storage_writer, AsyncStager):
            # bwc with old storage_writers
            async_stager = storage_writer
        else:
            async_stager = DefaultStager(
                StagingOptions(
                    False,
                    False,
                    False,
                    False,
                )
            )
            owned_async_stager = async_stager

    state_dict = _stateful_to_state_dict(state_dict)

    owned_async_stager_closed = False

    def maybe_close_owned_async_stager() -> None:
        nonlocal owned_async_stager_closed
        if owned_async_stager is not None and not owned_async_stager_closed:
            owned_async_stager_closed = True
            owned_async_stager.close()

    @_dcp_method_logger(log_exceptions=True)
    def stage_state_dict() -> Future[STATE_DICT_TYPE] | STATE_DICT_TYPE:
        return async_stager.stage(state_dict)

    with contextlib.ExitStack() as stack:
        if owned_async_stager is not None:
            stack.callback(maybe_close_owned_async_stager)

        staging_future_or_state_dict = stage_state_dict()

        upload_executor: _AsyncCheckpointExecutor = (
            _ProcessBasedAsyncCheckpointExecutor()
            if async_checkpointer_type == AsyncCheckpointerType.PROCESS
            else _ThreadBasedAsyncCheckpointExecutor()
        )

        upload_future: Future = upload_executor.execute_save(
            staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            process_group=process_group,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )

        if owned_async_stager is not None:
            upload_future.add_done_callback(lambda _: maybe_close_owned_async_stager())
            # in the success path transfer cleanup ownership to the upload future
            stack.pop_all()

    if isinstance(staging_future_or_state_dict, Future):
        staging_future = staging_future_or_state_dict
        return_staging_future: Future[None] = Future()

        def callback(
            original_staging_future: Future[STATE_DICT_TYPE],
            return_staging_future: Future[None] = return_staging_future,
        ):
            try:
                original_staging_future.result()
                return_staging_future.set_result(None)
            except Exception as e:
                return_staging_future.set_exception(e)

        if not staging_future.done():
            staging_future.add_done_callback(callback)
        else:
            return_staging_future.set_result(None)

        # return new AsyncSaveResponse for users using new ZOC implementation
        return AsyncSaveResponse(
            staging_completion=return_staging_future, upload_completion=upload_future
        )
    else:

        @_dcp_method_logger(log_exceptions=True)
        def maybe_synchronize_staging():
            if async_stager.should_synchronize_after_execute:
                async_stager.synchronize_staging()

        maybe_synchronize_staging()
        return upload_future


@_dcp_method_logger(log_exceptions=True)
def _stateful_to_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Creates a shallow copy of `state_dict` where `state_dict` is called for each Stateful object."""
    stateful_state_dict = {}
    for key, elem in state_dict.items():
        # Apply _dcp_method_logger to each state_dict() call
        def _elem_to_state_dict(elem):
            return elem.state_dict() if isinstance(elem, Stateful) else elem

        _elem_to_state_dict.__name__ = f"_stateful_to_state_dict.{key}"

        stateful_state_dict[key] = _dcp_method_logger(log_exceptions=True)(
            _elem_to_state_dict
        )(elem)
    return stateful_state_dict


def _save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: dist.ProcessGroup | None = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: SavePlanner | None = None,
    use_collectives: bool = True,
) -> Metadata:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    if planner is None:
        raise AssertionError("planner is None")

    global_metadata = None

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        if planner is None:
            raise AssertionError("planner is None")
        storage_meta = storage_writer.storage_meta()
        if "storage_meta" not in inspect.signature(planner.set_up_planner).parameters:
            warnings.warn(
                "The function definition for SavePlanner.set_up_planner has been updated"
                " to include the storage_meta argument. Please update your implementation"
                " to include this parameter.",
                stacklevel=2,
            )
            planner.set_up_planner(state_dict, distW.is_coordinator)  # type: ignore[call-arg, arg-type]
        else:
            planner.set_up_planner(
                state_dict=state_dict,
                storage_meta=storage_meta,
                is_coordinator=distW.is_coordinator,
            )

        if (
            "kwargs"
            in inspect.signature(storage_writer.set_up_storage_writer).parameters
        ):
            storage_writer.set_up_storage_writer(
                distW.is_coordinator,
                rank=distW.rank,
                use_collectives=use_collectives,
            )
        else:
            storage_writer.set_up_storage_writer(distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        nonlocal global_metadata

        if planner is None:
            raise AssertionError("planner is None")
        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: SavePlan | None = None
    if use_collectives:
        central_plan = distW.reduce_scatter("plan", local_step, global_step)
    else:
        local_plan: SavePlan = local_step()
        global_plan: list[SavePlan] = global_step([local_plan])
        central_plan = global_plan[0]

    @_dcp_method_logger(**ckpt_kwargs)
    def write_data():
        if planner is None:
            raise AssertionError("planner is None")
        if central_plan is None:
            raise AssertionError("central_plan is None")
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)

        all_writes.wait()
        return all_writes.value()

    @_dcp_method_logger(**ckpt_kwargs)
    def finish_checkpoint(all_results):
        if global_metadata is None:
            raise AssertionError("global_metadata is None")
        storage_writer.finish(metadata=global_metadata, results=all_results)
        return global_metadata

    if use_collectives:
        metadata = distW.all_reduce("write", write_data, finish_checkpoint)
    else:
        write_results: list[WriteResult] = write_data()
        metadata = finish_checkpoint([write_results])
        distW.barrier()

    return metadata
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
# ruff: noqa: F821
# flake8: noqa: F821
from collections.abc import Callable, Collection, Mapping, MutableMapping
from typing import cast, TypeVar
from typing_extensions import TypeIs

import torch
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.tensor import DTensor


PATH_ITEM = str | int
OBJ_PATH = tuple[PATH_ITEM, ...]
T = TypeVar("T")

STATE_DICT_ITEM = object
CONTAINER_TYPE = MutableMapping[PATH_ITEM, STATE_DICT_ITEM]

__all__ = ["traverse_state_dict", "set_element", "get_element", "print_tensor"]


def _keep_visiting_tensors(value: STATE_DICT_ITEM) -> TypeIs[torch.Tensor]:
    return isinstance(value, torch.Tensor)


# TODO: update docstring for traverse.py
def traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping will be traversed and ``visitor`` will be applied to the leaf elements.
    ``visitor`` will only be applied to elements in a list or a tuple, if the
    container contains tensors or mappings.
    """

    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            return False
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)

    # release reference cycle to prevent memory leaks in async_save
    del _traverse_obj, _is_terminal


def traverse_state_dict_v_2_3(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    """
    Traversal is short-circuited when it finds a collection for which ``keep_visiting_tensors`` evaluates
    to false for all elements.
    By default, all collections with at least one ``torch.Tensor`` element are traversed.
    Visitor takes a path argument that is a tuple of the keys used to reach it.
    """

    # a value is terminal if it has no other containers values inside it
    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)

    # release reference cycle to prevent memory leaks in async_save
    del _traverse_obj, _is_terminal


def set_element(
    root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: STATE_DICT_ITEM
) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    cur_container = cast(CONTAINER_TYPE, root_dict)

    def extend_list(lst: list[STATE_DICT_ITEM], idx: int) -> None:
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        def_val = cast(STATE_DICT_ITEM, {} if type(key) is str else [])

        if isinstance(cur_container, Mapping):
            cur_container = cast(
                CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val)
            )
        else:
            # pyrefly: ignore [bad-argument-type]
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    key = path[-1]
    if type(key) is int:
        extend_list(cast(list[STATE_DICT_ITEM], cur_container), key)

    cur_container[key] = value


def get_element(
    root_dict: STATE_DICT_TYPE,
    path: OBJ_PATH,
    default_value: T | None = None,
) -> T | None:
    """Retrieve the value at ``path``from ``root_dict``, returning ``default_value`` if not found."""
    cur_value = cast(CONTAINER_TYPE, root_dict)
    for part in path:
        if type(part) is int:
            if not isinstance(cur_value, list) or len(cur_value) < part:
                return default_value
        elif not isinstance(cur_value, Mapping) or part not in cur_value:
            return default_value

        cur_value = cast(CONTAINER_TYPE, cur_value[part])
    return cast(T | None, cur_value)


def _print_nested(
    value: STATE_DICT_ITEM,
    prefix: str = "",
    print_fun: Callable[[str], None] = print,
) -> None:
    if type(value) is ShardedTensor:
        print_fun(f"{prefix} ShardedTensor size: {value.size()}")
        for shard in value.local_shards():
            _print_nested(
                shard.tensor,
                f"{shard.metadata.shard_offsets} ",
                print_fun=print_fun,
            )
    elif type(value) is (DTensor):
        print_fun(f"{prefix} DistributedTensor size: {value.size()}")
        # TODO: add local offset for _local_tensor in print_nested.
        _print_nested(
            value._local_tensor,
            print_fun=print_fun,
        )
    elif isinstance(value, torch.Tensor):
        print_fun(f"{prefix} Tensor size: {value.size()}")
    else:
        print_fun(f"{prefix} Type: {type(value)}")


def print_tensor(
    path: OBJ_PATH,
    value: STATE_DICT_ITEM,
    print_fun: Callable[[str], None] = print,
) -> None:
    """
    Use this callback with traverse_state_dict to print its content.

    By default the content is printed using the builtin ``print`` but this can
    be changed by passing a different ``print_fun`` callable.
    """
    _print_nested(value, prefix=str(path), print_fun=print_fun)
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE

from . import _version
from ._traverse import (
    OBJ_PATH,
    set_element,
    STATE_DICT_ITEM,
    traverse_state_dict,
    traverse_state_dict_v_2_3,
)


"""
TODO:
Need to add ability to handle tuple, OrderedDict, NamedTuple.
Update mappings from dict to a class.
Change set_element to recreate the right type for tuple, OrderedDict, and NamedTuple.
"""


FLATTEN_MAPPING = dict[str, OBJ_PATH]


# TODO: Update Docstring for nested_dict.py
def flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    # We started to flatten dictionary since v2.4. But in order to not break
    # the checkpoints that were saved before v2.4, we need to keep the old
    # traversal so that we can reconstruct those checkpoints.
    use_v_2_3 = (
        _version._derived_version is not None and _version._derived_version == "2_3"
    )
    if use_v_2_3:
        traverse_state_dict_v_2_3(state_dict, flat_copy)
    else:
        traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings


def unflatten_state_dict(
    state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING
) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    nested: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        set_element(nested, mapping[key], value)
    return nested
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import json
import logging
import queue
import threading
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files,
)
from torch.distributed.checkpoint._hf_utils import (
    _gen_file_name,
    _HFStorageInfo,
    _metadata_fn,
    CUSTOM_METADATA_KEY,
    SAVED_OFFSETS_KEY,
    SHARDED_DIR_NAME,
    SUFFIX,
)
from torch.distributed.checkpoint.filesystem import SerializationFormat
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future


logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["HuggingFaceStorageWriter", "HuggingFaceStorageReader"]


class HuggingFaceStorageWriter(FileSystemWriter):
    """
    A writer that writes to storage in the huggingface safetensors format.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: dict[str, int] | None = None,
        thread_count: int = 1,
        save_distributed: bool = False,
        enable_consolidation: bool = False,
        thread_count_consolidation: int = 1,
    ) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: directory where the checkpoint will be read from.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the tensor should be written to.
                              Indices are from 1 to N, where N is the number of files. If not provided,
                              the tensors will be written to a single file. If none, then all the tensors on the
                              same rank will be written to the same file.
            thread_count: Number of threads to use to write distributed checkpoint. Default to 1.
            save_distributed: If True, save the checkpoint using distributed APIs where every rank saves its own shard.
                        Default is False which assumes rank-0 checkpointing of the full state_dict.
            enable_consolidation: If True, consolidate the sharded checkpoint after saving. The sharded tensors will be
                                saved to path/sharded and the full tensors will be saved to path. Default to False.
            thread_count_consolidation: Number of threads to use for parallel processing of saving data
                                to consolidated output files. Default to 1.
        """

        super().__init__(
            path=path,
            serialization_format=SerializationFormat.SAFETENSORS,
            thread_count=thread_count,
        )
        self.fqn_to_index_mapping: dict[str, int] | None = fqn_to_index_mapping
        self.save_distributed: bool = save_distributed
        self.enable_consolidation: bool = enable_consolidation
        self.consolidated_output_path: str | None = None
        if self.enable_consolidation:
            self.consolidated_output_path = str(self.path)
            self.path = self.fs.concat_path(self.path, SHARDED_DIR_NAME)
        self.thread_count_consolidation = thread_count_consolidation

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        new_plans = []
        for i, plan in enumerate(plans, start=1):
            storage_data: dict[str, Any] = {}
            if self.fqn_to_index_mapping is not None:
                storage_data["fqn_to_index_mapping"] = self.fqn_to_index_mapping
            if self.save_distributed:
                storage_data["shard_index"] = i

            new_plans.append(dataclasses.replace(plan, storage_data=storage_data))

        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        if len(plan.items) == 0:
            fut: Future = Future()
            fut.set_result([])
            return fut

        # storage_plan is a map from key to file index
        storage_data: dict[str, Any] = plan.storage_data
        storage_plan: dict[str, int] | None = None
        shard_index: int | None = None
        if "fqn_to_index_mapping" in storage_data:
            storage_plan = storage_data["fqn_to_index_mapping"]
        if "shard_index" in storage_data:
            shard_index = storage_data["shard_index"]

        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(storage_plan.values()) if storage_plan is not None else 1

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            file_name = _gen_file_name(file_index, highest_index, shard_index)
            file_queue.put(
                (self.fs.concat_path(self.path, file_name), file_name, write_items)
            )

        return super()._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        if self.save_distributed and not self.enable_consolidation:
            # if we are saving distributed, without consolidating,
            # then we have no metadata to write because a metadata
            # file with fqn to file mapping doesn't make sense
            # in this case, because fqns will be in multiple files
            logger.info("Not consolidating sharded checkpoint in finish step.")
            return
        if self.save_distributed:
            fqn_to_index_mapping: dict[str, int] = (
                self.fqn_to_index_mapping
                if self.fqn_to_index_mapping is not None
                else dict.fromkeys(metadata.state_dict_metadata.keys(), 1)
            )

            return consolidate_safetensors_files(
                input_dir=str(self.path),
                output_dir=self.consolidated_output_path,  # type: ignore[arg-type]
                num_threads=self.thread_count_consolidation,
                fqn_to_index_mapping=fqn_to_index_mapping,
            )

        # writing a model.index.safetensors.json file with fqn to file mapping
        # for the rank-0 checkpointing case
        metadata_to_write = {}
        storage_md = {}
        total_size = 0
        for wr_list in results:
            storage_md.update(
                {wr.index.fqn: wr.storage_data.relative_path for wr in wr_list}
            )
            total_size += sum([wr.storage_data.length for wr in wr_list])
        metadata_to_write["metadata"] = {"total_size": total_size}
        metadata_to_write["weight_map"] = storage_md

        metadata_path = self.fs.concat_path(self.path, f"{_metadata_fn}")
        with self.fs.create_stream(metadata_path, "w") as metadata_file:
            json.dump(metadata_to_write, metadata_file, indent=2)

    def _split_by_storage_plan(
        self, storage_plan: dict[str, int] | None, items: list[WriteItem]
    ) -> dict[int, list[WriteItem]]:
        # storage_plan is a map from key to index
        if storage_plan is None:
            return {1: items}

        buckets = {}
        for item in items:
            key = item.index.fqn

            idx = storage_plan[key]
            if idx not in buckets:
                buckets[idx] = [item]
            else:
                buckets[idx].append(item)

        return buckets

    @property
    def metadata_path(self) -> str:
        return _metadata_fn


class HuggingFaceStorageReader(FileSystemReader):
    """
    A reader that reads a checkpoint in the huggingface safetensors format.
    """

    def __init__(self, path: str, thread_count: int = 1) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: directory where the checkpoint will be read from.
            thread_count: Number of threads to use to read distributed checkpoint. Default to 1.
        """

        super().__init__(path=path)
        self.thread_count = thread_count

    def _process_read_request(self, f, req: ReadItem, planner: LoadPlanner) -> None:
        """Helper function to process a single read request."""
        # Create slices for each dimension based on offsets and lengths
        slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(req.storage_offsets, req.lengths)
        )
        tensor = f.get_slice(req.storage_index.fqn)[slices]
        target_tensor = planner.resolve_tensor(req).detach()

        if target_tensor.size() != tensor.size():
            raise AssertionError(
                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
            )

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)

    def _read_files_from_queue(
        self,
        file_queue: queue.Queue,
        result_queue: queue.Queue,
        planner: LoadPlanner,
    ) -> None:
        from safetensors import safe_open  # type: ignore[import]

        try:
            while True:
                file_name, reqs = file_queue.get_nowait()
                with safe_open(filename=file_name, framework="pt") as f:
                    for req in reqs:
                        self._process_read_request(f, req, planner)
                result_queue.put(True)  # Signal that this file has been processed
        except queue.Empty:
            pass

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        from safetensors import safe_open  # type: ignore[import]

        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        if self.thread_count <= 1 or len(per_file) <= 1:
            for file_name, reqs in per_file.items():
                with safe_open(filename=file_name, framework="pt") as f:
                    for req in reqs:
                        self._process_read_request(f, req, planner)
        else:
            # Use parallel implementation with thread pool
            file_queue: queue.Queue = queue.Queue()
            result_queue: queue.Queue = queue.Queue()

            # Fill the queue with files to process
            for file_name, reqs in per_file.items():
                file_queue.put((file_name, reqs))

            # Create and start worker threads
            threads = []
            num_threads = min(self.thread_count, len(per_file))
            for _ in range(num_threads):
                t = threading.Thread(
                    target=self._read_files_from_queue,
                    args=(file_queue, result_queue, planner),
                )
                t.start()
                threads.append(t)

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Check if all files were processed
            processed_count = 0
            try:
                while True:
                    result_queue.get_nowait()
                    processed_count += 1
            except queue.Empty:
                pass

            if processed_count != len(per_file):
                raise AssertionError(
                    f"Not all files were processed: {processed_count} out of {len(per_file)}"
                )

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # pyrefly: ignore [bad-override]
    def read_metadata(self) -> Metadata:
        from safetensors import safe_open  # type: ignore[import]
        from safetensors.torch import _getdtype  # type: ignore[import]

        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        safetensors_files = []
        for file in self.fs.ls(self.path):
            if file.endswith(SUFFIX):
                safetensors_files.append(file)

        for safetensor_file in safetensors_files:
            with safe_open(safetensor_file, framework="pt") as f:
                keys = f.keys()
                extra_metadata = f.metadata()

                dcp_sharding_info = None
                if extra_metadata and extra_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(
                        extra_metadata.get(CUSTOM_METADATA_KEY)
                    )

                for key in keys:
                    shape = f.get_slice(key).get_shape()
                    dtype = f.get_slice(key).get_dtype()
                    # construct state_dict_metadata
                    if dcp_sharding_info is not None:
                        offset = dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                    else:
                        offset = [0] * len(shape)

                    if key not in state_dict_metadata:
                        state_dict_metadata[key] = TensorStorageMetadata(
                            properties=TensorProperties(dtype=_getdtype(dtype)),
                            size=torch.Size(
                                [saved + offset for saved, offset in zip(shape, offset)]
                            ),
                            chunks=[
                                ChunkStorageMetadata(
                                    offsets=torch.Size(offset),
                                    sizes=torch.Size(shape),
                                )
                            ],
                        )
                    else:
                        state_dict_metadata[key].chunks.append(
                            ChunkStorageMetadata(
                                torch.Size(offset), sizes=torch.Size(shape)
                            )
                        )
                        size = list(state_dict_metadata[key].size)
                        for i in range(len(size)):
                            size[i] = max(size[i], shape[i] + offset[i])
                        state_dict_metadata[key].size = torch.Size(size)

                    # construct storage data
                    if dcp_sharding_info is not None:
                        metadata_index = MetadataIndex(
                            fqn=key, offset=dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                        )
                    else:
                        metadata_index = MetadataIndex(fqn=key, offset=[0] * len(shape))
                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=safetensor_file,
                        shape=torch.Size(shape),
                        dtype=_getdtype(dtype),
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,  # type: ignore[arg-type]
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id  # type: ignore[union-attr]

        return metadata
<|endoftext|># Mypy will not try inferring the types of any 3rd party libraries installed.
# mypy: ignore-errors

import io
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fsspec.core import url_to_fs

from torch.distributed.checkpoint._extension import StreamTransformExtension
from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    FileSystemWriter,
    SerializationFormat,
)


if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


__all__ = [
    "FsspecWriter",
    "FsspecReader",
]


class FileSystem(FileSystemBase):
    def __init__(self) -> None:
        self.fs: AbstractFileSystem | None = None

    @contextmanager
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]:
        if self.fs is None:
            raise AssertionError("fs should not be None")
        path = os.fspath(path)

        # fsspec does not support concurrent transactions, and not all
        # AbstractFileSystem have working rollback implementations, so
        # just manually delete the file if necessary on errors.
        with self.fs.open(path, mode) as stream:
            try:
                yield stream
            except:
                if any(ch in mode for ch in "w+a"):  # cleanup file if not read-only
                    try:
                        self.rm_file(path)
                    except:  # noqa: E722
                        pass
                raise

    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike:
        return os.path.join(path, suffix)

    def init_path(self, path: str | os.PathLike, **kwargs) -> str | os.PathLike:
        self.fs, _ = url_to_fs(path, **kwargs)
        return path

    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None:
        self.fs.rename(path, new_path)

    def mkdir(self, path: str | os.PathLike) -> None:
        self.fs.makedirs(path, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        if isinstance(checkpoint_id, Path):
            return False

        try:
            url_to_fs(checkpoint_id)
        except ValueError:
            return False

        return True

    def exists(self, path: str | os.PathLike) -> bool:
        return self.fs.exists(path)

    def rm_file(self, path: str | os.PathLike) -> None:
        self.fs.rm(path)

    def ls(self, path: str | os.PathLike) -> list[str]:
        # setting detail to False explicitly to keep the list[str] return type,
        # instead of the list[Dict] return type when detail=True
        return self.fs.ls(path, detail=False)


# TODO: add the dcp.async_save mixin
class FsspecWriter(FileSystemWriter):
    """
    Basic implementation of StorageWriter using fsspec.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: str | os.PathLike,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
        **kwargs,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving them. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__(
            path,
            single_file_per_rank,
            sync_files,
            thread_count,
            per_thread_copy_ahead,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class FsspecReader(FileSystemReader):
    def __init__(self, path: str | os.PathLike, **kwargs) -> None:
        super().__init__(path)
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
<|endoftext|>import os

from .filesystem import FileSystemReader, FileSystemWriter
from .storage import StorageReader, StorageWriter


def _storage_setup(
    storage: StorageReader | StorageWriter | None,
    checkpoint_id: str | os.PathLike | None,
    reader: bool = False,
) -> StorageReader | StorageWriter | None:
    if storage:
        if checkpoint_id is not None:
            storage.reset(checkpoint_id)
        return storage

    if not checkpoint_id:
        raise RuntimeError(
            "`checkpoint_id` must be specified if "
            "storage_reader/storage_writer is None."
        )

    targets: list[type[StorageReader | StorageWriter]] = []
    if reader:
        targets = [
            FileSystemReader,
        ]
    else:
        targets = [
            FileSystemWriter,
        ]
    try:
        from ._fsspec_filesystem import FsspecReader, FsspecWriter

        targets.append(FsspecReader if reader else FsspecWriter)
    except Exception:
        pass

    for target in targets:
        if target.validate_checkpoint_id(checkpoint_id):
            storage = target(checkpoint_id)  # type: ignore[call-arg]
            storage.reset(checkpoint_id)
            return storage

    raise RuntimeError(
        "Cannot detect which StorageReader or StorageWriter to use. "
        "Please specify the storage_reader/storage_writer."
    )
<|endoftext|># mypy: allow-untyped-defs
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec
from uuid import uuid4

import torch.distributed.c10d_logger as c10d_logger
from torch.distributed.checkpoint.logging_handlers import DCP_LOGGER_NAME


logger = logging.getLogger()


__all__: list[str] = []

# pyrefly: ignore [unknown-name]
global _dcp_logger
_dcp_logger = c10d_logger._get_or_create_logger(DCP_LOGGER_NAME)

_T = TypeVar("_T")
_P = ParamSpec("_P")


def _msg_dict_from_dcp_method_args(*args, **kwargs) -> dict[str, Any]:
    """
    Extracts log data from dcp method args
    """
    msg_dict = {}

    # checkpoint ID can be passed in through the serializer or through the checkpoint id directly
    storage_writer = kwargs.get("storage_writer")
    storage_reader = kwargs.get("storage_reader")
    planner = kwargs.get("planner")

    checkpoint_id = kwargs.get("checkpoint_id")
    if not checkpoint_id and (serializer := storage_writer or storage_reader):
        checkpoint_id = getattr(serializer, "checkpoint_id", None)

    msg_dict["checkpoint_id"] = (
        # pyrefly: ignore [unsupported-operation]
        str(checkpoint_id) if checkpoint_id is not None else checkpoint_id
    )

    # Uniquely identify a _dcp_method_logger wrapped function call.
    msg_dict["uuid"] = str(uuid4().int)

    if storage_writer:
        msg_dict["storage_writer"] = storage_writer.__class__.__name__

    if storage_reader:
        msg_dict["storage_reader"] = storage_reader.__class__.__name__

    if planner:
        msg_dict["planner"] = planner.__class__.__name__

    return msg_dict


def _get_msg_dict(func_name, *args, **kwargs) -> dict[str, Any]:
    msg_dict = _msg_dict_from_dcp_method_args(*args, **kwargs)
    msg_dict.update(c10d_logger._get_msg_dict(func_name, *args, **kwargs))

    return msg_dict


def _dcp_method_logger(
    log_exceptions: bool = False, **wrapper_kwargs: Any
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:  # pyre-ignore
    """This method decorator logs the start, end, and exception of wrapped events."""

    def decorator(func: Callable[_P, _T]):
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            msg_dict = _get_msg_dict(
                func.__name__, *args, **{**wrapper_kwargs, **kwargs}
            )

            # log start event
            msg_dict["event"] = "start"
            t0 = time.time_ns()
            msg_dict["time"] = t0
            msg_dict["log_exceptions"] = log_exceptions
            _dcp_logger.debug(msg_dict)

            # exceptions
            try:
                result = func(*args, **kwargs)
            except BaseException as error:
                if log_exceptions:
                    msg_dict["event"] = "exception"
                    msg_dict["error"] = f"{error}"
                    msg_dict["time"] = time.time_ns()
                    _dcp_logger.error(msg_dict)
                raise

            # end event
            msg_dict["event"] = "end"
            t1 = time.time_ns()
            msg_dict["time"] = time.time_ns()
            msg_dict["times_spent"] = t1 - t0
            _dcp_logger.debug(msg_dict)

            return result

        return wrapper

    return decorator


def _init_logger(rank: int):
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f"[{rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING

from torch.distributed.checkpoint.planner import SavePlan, WriteItem


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import MetadataIndex

__all__ = ["dedup_save_plans"]


def dedup_save_plans(
    all_plans: list[SavePlan],
    save_to_lowest_rank: bool = False,
) -> list[SavePlan]:
    """
    Removes duplicate entries from appearing on multiple SavePlans. For each duplicate across
    a set of SavePlans, only the smallest SavePlan in terms of planned storage keeps the entry.

    Please note that this function does not modify the original SavePlans, but rather returns
    """

    # Map to query the plan indices that a write item is duplicated in
    write_item_to_plan_indices: dict[MetadataIndex, set[int]] = defaultdict(set)
    # Map to query the write item from its index
    write_item_idx_to_write_item: dict[MetadataIndex, WriteItem] = {}
    # Set of write item indices that are present in each plan
    # After deduplication, this will be the set of write item indices that are present in the final plans
    plan_to_item_indices: list[set[MetadataIndex]] = [
        {item.index for item in plan.items} for plan in all_plans
    ]

    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            # map each write item to its plan
            write_item_to_plan_indices[write_item.index].add(plan_idx)
            write_item_idx_to_write_item[write_item.index] = write_item
    plan_to_size = [0] * len(all_plans)
    for write_item_idx, plan_indices in write_item_to_plan_indices.items():
        if save_to_lowest_rank:
            select_plan_idx = min(plan_indices)
        else:
            select_plan_idx = min(
                plan_indices, key=lambda plan_idx: plan_to_size[plan_idx]
            )

        write_item = write_item_idx_to_write_item[write_item_idx]
        # Ignore the storage size of anything that is not a tensor, since
        # we don't know how much storage they represent
        plan_to_size[select_plan_idx] += write_item.tensor_storage_size() or 1
        for plan_idx in plan_indices - {select_plan_idx}:
            plan_to_item_indices[plan_idx].discard(write_item_idx)
    # Sanity check
    if len(all_plans) != len(plan_to_item_indices):
        raise AssertionError("len(all_plans) != len(plan_to_item_indices)")
    # Create new plans with the updated write items post deduplication
    return [
        dataclasses.replace(
            plan, items=[item for item in plan.items if item.index in item_indexes]
        )
        for plan, item_indexes in zip(all_plans, plan_to_item_indices)
    ]
<|endoftext|># pyre-strict

import concurrent.futures
import glob
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import distributed as dist
from torch.distributed.checkpoint._hf_utils import (
    _gen_file_name,
    _get_dcp_custom_metadata,
    _get_safetensors_file_metadata,
    _metadata_fn,
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SUFFIX,
)


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class _FqnData:
    """
    Dataclass to store information about a tensor (identified by its fully qualified name).

    Attributes:
        offset_in_file: Byte offset where this tensor's data begins in the output file
        shape_in_file: Shape of the tensor in the output file
        dtype_size: Size of the tensor's data type in bytes
        dtype_str: String representation of the tensor's data type
    """

    offset_in_file: int = 0
    shape_in_file: list[int] = field(default_factory=list)
    dtype_size: int = 0
    dtype_str: str = ""


@dataclass
class _OutputFileData:
    """
    Dataclass to store information about an output safetensors file.

    Attributes:
        metadata_size: Size of the metadata section in bytes
        fqn_data: Dictionary mapping tensor names to their metadata
    """

    metadata_size: int = 0
    fqn_data: dict[str, _FqnData] = field(default_factory=dict)


@dataclass
class _InputFileData:
    """
    Dataclass to store information about an input safetensors file.

    Attributes:
        metadata_size: Size of the metadata section in bytes
        metadata: Json metadata from the safetensors file
    """

    metadata_size: int = 0
    metadata: Any = None


def _parse_input_metadata(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Parse metadata from input safetensors files to determine the full tensor shapes and types.

    This function analyzes the metadata from all input files to determine the complete shape
    of each tensor after consolidation. It updates the output_files_data with this information.

    Args:
        input_files_data: dict of metadata from input safetensors files
        output_files_data: Dictionary mapping output file paths to their metadata

    Raises:
        ValueError: If no DCP custom metadata is found in a safetensors file
    """

    from safetensors.torch import _getdtype  # type: ignore[import]

    # Dictionary to track the full size of each tensor across all shards
    fqn_to_size_mapping: dict[str, tuple[list[int], str]] = {}

    for file_data in input_files_data.values():
        safetensors_metadata = file_data.metadata
        dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)
        if not dcp_sharding_info:
            raise ValueError(
                "No DCP custom metadata found in safetensors file. The file must be saved with DCP to be consolidated."
            )

        for key, val in safetensors_metadata.items():
            if key == DEFAULT_EXTRA_METADATA_KEY:
                continue

            # Get the shape of this tensor shard and its offset in the full tensor
            sizes = val[SHAPE_KEY]
            offsets = dcp_sharding_info[key][SAVED_OFFSETS_KEY]

            if key not in fqn_to_size_mapping:
                # First time seeing this tensor - calculate its full size by adding offsets to dimensions
                cur_size = [size + offset for size, offset in zip(sizes, offsets)]
                fqn_to_size_mapping[key] = (cur_size, val[DTYPE_KEY])
            else:
                # We've seen this tensor before - update its size if this shard extends beyond current known dimensions
                cur_size = fqn_to_size_mapping[key][0]
                for i in range(len(sizes)):
                    cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])

    # Now that we know the full size of each tensor, populate the output file data
    for fqn, tensor_info in fqn_to_size_mapping.items():
        tensor_size = tensor_info[0]
        dtype_str = tensor_info[1]
        for output_data in output_files_data.values():
            # Add this tensor to the output file if it's already assigned there
            if fqn in output_data.fqn_data:
                output_data.fqn_data[fqn] = _FqnData(
                    shape_in_file=tensor_size,
                    dtype_size=torch.finfo(_getdtype(dtype_str)).bits
                    // 8,  # Convert bits to bytes
                    dtype_str=dtype_str,
                )


def _write_metadata(
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Write metadata to the beginning of each output safetensors file.

    This function writes the metadata section to each output file, including information
    about tensor shapes, data types, and offsets. It also updates the offset_in_file
    field for each tensor in the output_files_data.

    Args:
        output_files_data: Dictionary mapping output file paths to their metadata
    """
    # Process each output file
    for file_path, output_data in output_files_data.items():
        with open(file_path, "wb") as f:
            metadata = {}
            curr_offset = 0

            # Calculate offsets for each tensor in the file
            for fqn, fqn_data in output_data.fqn_data.items():
                # Calculate the end offset by multiplying all dimensions and the data type size
                end_offset = (
                    curr_offset
                    + math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
                )

                # Store metadata for this tensor
                metadata[fqn] = {
                    SHAPE_KEY: fqn_data.shape_in_file,
                    DTYPE_KEY: fqn_data.dtype_str,
                    DATA_OFFSETS_KEY: [
                        curr_offset,
                        end_offset,
                    ],  # Start and end byte offsets
                }
                # Store the offset for later use when writing the actual tensor data
                fqn_data.offset_in_file = curr_offset

                # Update current offset for the next tensor
                curr_offset = end_offset

            # Convert metadata to JSON and encode as bytes
            json_metadata = json.dumps(metadata)
            json_bytes = json_metadata.encode("utf-8")

            # Write the metadata size as an 8-byte unsigned integer (little-endian)
            size_in_bytes = len(json_bytes)
            header_len = struct.pack("<Q", size_in_bytes)

            # Write the header length and metadata to the file
            f.write(header_len)
            f.write(json_bytes)

            # Store the total metadata size (header + JSON) for later use
            output_data.metadata_size = f.tell()


def _read_tensor_data(
    f,
    start_offset: int,
    end_offset: int,
    metadata_size: int,
) -> bytes:
    """
    Read a specific byte range of tensor data from an open safetensors file.

    Args:
        f: An open file object (handle) for the safetensors file
        start_offset: Start offset of tensor data within the data section
        end_offset: End offset of tensor data within the data section
        metadata_size: Size of the metadata header

    Returns:
        Raw tensor data as bytes
    """
    absolute_start = metadata_size + start_offset
    length = end_offset - start_offset

    f.seek(absolute_start)
    return f.read(length)


def _process_output_file(
    output_file: str,
    output_data: _OutputFileData,
    input_files_data: dict[str, _InputFileData],
) -> None:
    """
    Process a single output file by writing tensor data from input files using direct reads.

    This function is designed to be run in parallel for different output files.

    Args:
        output_file: Path to the output file
        output_data: Metadata for the output file
        input_files_data: Dictionary mapping input file paths to their metadata
    """

    sorted_tensors = sorted(
        output_data.fqn_data.items(), key=lambda x: x[1].offset_in_file
    )

    file_handles = {}
    dcp_metadata = {}
    for safetensors_file, file_data in input_files_data.items():
        dcp_metadata[safetensors_file] = _get_dcp_custom_metadata(file_data.metadata)

    try:
        # Open all input files for reading
        for safetensors_file in input_files_data:
            file_handles[safetensors_file] = open(safetensors_file, "rb")  # noqa: SIM115

        with open(output_file, "r+b") as output_stream:
            output_stream.seek(0, os.SEEK_END)
            # Process each tensor in sequential output order
            for tensor_fqn, tensor_fqn_data in sorted_tensors:
                full_tensor_mv = memoryview(
                    bytearray(
                        math.prod(tensor_fqn_data.shape_in_file)
                        * tensor_fqn_data.dtype_size
                    )
                )

                # Process each input safetensors file
                for safetensors_file in input_files_data:
                    file_metadata = input_files_data[safetensors_file].metadata
                    input_metadata_size = input_files_data[
                        safetensors_file
                    ].metadata_size

                    if tensor_fqn not in file_metadata:
                        continue

                    metadata = file_metadata[tensor_fqn]

                    data_offsets = metadata[DATA_OFFSETS_KEY]

                    # Use explicit reads to fetch tensor data efficiently
                    data_to_write = _read_tensor_data(
                        file_handles[safetensors_file],
                        data_offsets[0],
                        data_offsets[1],
                        input_metadata_size,
                    )

                    # Get the offsets of this tensor shard within the full tensor
                    # pyrefly: ignore [unsupported-operation]
                    fqn_custom_metadata = dcp_metadata[safetensors_file][tensor_fqn]  # type: ignore[index]
                    offsets_of_tensor_being_read = fqn_custom_metadata[
                        SAVED_OFFSETS_KEY
                    ]  # type: ignore[index]

                    # Write this tensor shard to the appropriate position in the output file
                    _write_sub_tensor_to_file_optimized(
                        full_tensor_mv,
                        data_to_write,
                        tensor_fqn_data.dtype_size,  # Size of each element in bytes
                        tensor_fqn_data.shape_in_file,  # Full tensor shape
                        offsets_of_tensor_being_read,  # Where this shard belongs in the full tensor
                        metadata[SHAPE_KEY],  # Shape of this shard
                    )

                output_stream.write(full_tensor_mv)

    finally:
        for f in file_handles.values():
            f.close()


def _write_data(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
    num_threads: int = 1,
) -> None:
    """
    Write tensor data from input files to the output files using memory mapping.

    This function reads tensor data from each input file and writes it to the appropriate
    position in the output files based on the tensor's offsets. When num_threads > 1,
    the work is split across threads with each thread handling a different output file.

    Args:
        input_files_data: Dictionary mapping input file paths to their metadata
        output_files_data: Dictionary mapping output file paths to their metadata
        num_threads: Number of threads to use for parallel processing
    """
    if num_threads <= 1 or len(output_files_data) <= 1:
        # Sequential processing
        for output_file, output_data in output_files_data.items():
            _process_output_file(output_file, output_data, input_files_data)
    else:
        # Parallel processing with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(num_threads, len(output_files_data))
        ) as executor:
            futures = []
            for output_file, output_data in output_files_data.items():
                futures.append(
                    executor.submit(
                        _process_output_file,
                        output_file,
                        output_data,
                        input_files_data,
                    )
                )

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                # Handle any exceptions that might have occurred
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing output file: {e}")
                    raise


def _write_sub_tensor_to_file_optimized(
    full_tensor_mv: memoryview,
    sub_tensor_bytes: bytes,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
) -> None:
    """
    Optimized version that writes the maximum number of contiguous bytes possible.

    Uses a unified algorithm that calculates the maximum contiguous bytes that can be
    written in each iteration and continues until the entire subtensor is written.
    Handles all sharding patterns efficiently:
    - Full sub-tensor at once for row-wise sharding
    - Row-by-row for column-wise sharding
    - Optimized chunks for other patterns

    Args:
        full_tensor_mv: Buffer to write the full tensor to
        sub_tensor_bytes: Raw tensor data as bytes
        element_size: Size of each element in bytes
        tensor_shape: Shape of the full tensor
        sub_tensor_offsets: Starting offsets of the sub-tensor within the full tensor
        sub_tensor_shape: Shape of the sub-tensor
    """
    # Handle empty tensors
    if not tensor_shape or not sub_tensor_shape:
        return

    # Calculate tensor strides for efficient indexing
    tensor_strides = [1]
    for i in range(len(tensor_shape) - 1, 0, -1):
        tensor_strides.insert(0, tensor_strides[0] * tensor_shape[i])

    sub_tensor_strides = [1]
    for i in range(len(sub_tensor_shape) - 1, 0, -1):
        sub_tensor_strides.insert(0, sub_tensor_strides[0] * sub_tensor_shape[i])

    total_elements = math.prod(sub_tensor_shape)

    elements_written = 0
    while elements_written < total_elements:
        # Convert linear index to multi-dimensional indices
        temp_idx = elements_written
        indices = []
        for dim_size in reversed(sub_tensor_shape):
            indices.append(temp_idx % dim_size)
            temp_idx //= dim_size
        indices.reverse()

        # Calculate maximum contiguous elements we can write from this position
        max_contiguous = _calculate_max_contiguous_elements(
            indices, sub_tensor_shape, tensor_shape
        )

        # Calculate source position in bytes
        src_pos = sum(idx * stride for idx, stride in zip(indices, sub_tensor_strides))
        src_byte_offset = src_pos * element_size

        # Calculate destination position in bytes
        dest_indices = [
            idx + offset for idx, offset in zip(indices, sub_tensor_offsets)
        ]
        dest_pos = sum(
            idx * stride for idx, stride in zip(dest_indices, tensor_strides)
        )
        dest_byte_offset = dest_pos * element_size

        # Write the contiguous chunk
        bytes_to_write = max_contiguous * element_size
        chunk_data = sub_tensor_bytes[
            src_byte_offset : src_byte_offset + bytes_to_write
        ]
        full_tensor_mv[dest_byte_offset : dest_byte_offset + bytes_to_write] = (
            chunk_data
        )

        elements_written += max_contiguous


def _calculate_max_contiguous_elements(
    indices: list[int],
    sub_tensor_shape: list[int],
    tensor_shape: list[int],
) -> int:
    """
    Calculate the maximum number of contiguous elements that can be written from current position.

    This determines the largest chunk by checking how elements are laid out in memory
    and finding natural boundaries where contiguity breaks.

    Args:
        indices: Current position indices in the sub-tensor
        sub_tensor_shape: Shape of the sub-tensor being written
        tensor_shape: Shape of the full tensor

    Raises:
        ValueError: If input lists are empty, have mismatched lengths, or contain invalid values
    """
    # Validate input lists are not empty
    if not indices or not sub_tensor_shape or not tensor_shape:
        raise ValueError("Input lists cannot be empty")

    # Validate all lists have the same length (same number of dimensions)
    if not (len(indices) == len(sub_tensor_shape) == len(tensor_shape)):
        raise ValueError(
            f"All input lists must have the same length. Got indices: {len(indices)}, "
            f"sub_tensor_shape: {len(sub_tensor_shape)}, tensor_shape: {len(tensor_shape)}"
        )

    # Validate indices are within bounds of sub_tensor_shape
    for i, (idx, sub_dim) in enumerate(zip(indices, sub_tensor_shape)):
        if idx >= sub_dim:
            raise ValueError(
                f"Index {idx} at dimension {i} is out of bounds for sub-tensor shape {sub_tensor_shape}"
            )

    # Validate sub_tensor dimensions don't exceed tensor dimensions
    for i, (sub_dim, tensor_dim) in enumerate(zip(sub_tensor_shape, tensor_shape)):
        if sub_dim > tensor_dim:
            raise ValueError(
                f"Sub-tensor dimension {sub_dim} at position {i} exceeds tensor dimension {tensor_dim}"
            )

    # Start with elements remaining in the last dimension
    max_contiguous = sub_tensor_shape[-1] - indices[-1]

    # Check if we can extend across multiple dimensions
    # We can write across dimension boundaries if we're writing complete "rows"
    # and the layout in destination tensor maintains contiguity

    # For 2D case: check if we can write multiple complete rows
    if len(sub_tensor_shape) >= 2:
        # If we're at the start of a row and can write complete rows
        if indices[-1] == 0:  # At start of last dimension (column)
            rows_remaining = sub_tensor_shape[-2] - indices[-2]  # Rows left to write

            # Check if writing complete rows maintains contiguity in destination
            # This is true for row-wise sharding or when sub-tensor spans full width
            if sub_tensor_shape[-1] == tensor_shape[-1]:  # Full width
                max_contiguous = rows_remaining * sub_tensor_shape[-1]

            # For higher dimensions, check if we can extend further
            if len(sub_tensor_shape) >= 3 and indices[-2] == 0:
                # Check if we can write complete 2D slices
                remaining_in_dim = sub_tensor_shape[-3] - indices[-3]
                if (
                    sub_tensor_shape[-1] == tensor_shape[-1]
                    and sub_tensor_shape[-2] == tensor_shape[-2]
                ):
                    max_contiguous = (
                        remaining_in_dim * sub_tensor_shape[-2] * sub_tensor_shape[-1]
                    )

    return max_contiguous


def _write_overall_metadata_file(
    output_dir: str,
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Write the overall metadata file that maps tensor names to their file locations.

    This creates a model.safetensors.index.json file that HuggingFace models use
    to locate tensors across multiple files.

    Args:
        output_dir: Directory where the metadata file will be written
        output_files_data: Dictionary mapping output file paths to their metadata
    """
    total_size = 0
    weight_map = {}
    for output_path, value in output_files_data.items():
        for fqn, fqn_data in value.fqn_data.items():
            total_size += math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
            weight_map[fqn] = os.path.basename(output_path)

    metadata_to_write: dict[str, Any] = {}
    metadata_to_write["metadata"] = {"total_size": total_size}
    metadata_to_write["weight_map"] = weight_map

    metadata_path = os.path.join(output_dir, f"{_metadata_fn}")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata_to_write, metadata_file, indent=2)


def _consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_file_mapping: dict[str, str],
    num_threads: int,
) -> dict[str, _OutputFileData]:
    output_files_data: dict[str, _OutputFileData] = {}
    # Create multiple output files based on the provided mapping
    for fqn, filename in fqn_to_file_mapping.items():
        output_path = os.path.join(output_dir, filename)

        if output_path not in output_files_data:
            output_files_data[output_path] = _OutputFileData(fqn_data={fqn: _FqnData()})
        else:
            output_files_data[output_path].fqn_data[fqn] = _FqnData()

    # Find all safetensors files in the input directory
    safetensors_files = glob.glob(os.path.join(input_dir, f"*{SUFFIX}"))

    # Read metadata from all input files
    input_files_data: dict[str, _InputFileData] = {}
    for safetensor_file in safetensors_files:
        with open(safetensor_file, "rb") as f:
            metadata, size = _get_safetensors_file_metadata(f)
            input_files_data[safetensor_file] = _InputFileData(
                metadata_size=size, metadata=metadata
            )
    # Step 1: Parse metadata to determine tensor shapes and types
    _parse_input_metadata(input_files_data, output_files_data)

    # Step 2: Write metadata headers to output files
    _write_metadata(output_files_data)
    # Step 3: Write actual tensor data from input files to output files
    _write_data(input_files_data, output_files_data, num_threads)

    return output_files_data


def consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
) -> None:
    """
    Main function to consolidate sharded safetensors files into one or more output files.

    This function orchestrates the entire consolidation process:
    1. Sets up the output file structure based on the fqn_to_index_mapping
    2. Finds all safetensors files in the input directory
    3. Parses metadata from all input files
    4. Writes metadata to the output files
    5. Writes tensor data from input files to output files
    6. Writes overall model.index.safetensors.json file with weight map

    Args:
        input_dir: Directory containing sharded safetensors files
        output_dir: Directory where consolidated files will be written
        fqn_to_index_mapping: Optional mapping of tensor names to output file indices.
                             If None, all tensors will be consolidated into a single file.
        num_threads: Number of threads to use for parallel processing of saving data to output files.
    """
    start_time = time.time()
    logger.info(
        "Consolidating safetensors files from %s to %s. Beginning at time %f",
        input_dir,
        output_dir,
        start_time,
    )

    max_index = max(fqn_to_index_mapping.values())
    fqn_to_file_mapping = {
        fqn: _gen_file_name(idx, max_index) for fqn, idx in fqn_to_index_mapping.items()
    }

    output_files_data = _consolidate_safetensors_files(
        input_dir, output_dir, fqn_to_file_mapping, num_threads
    )

    # Step 4: Write overall model.index.safetensors.json file with weight map
    _write_overall_metadata_file(output_dir, output_files_data)

    logger.info("Done consolidating. Took %.2f secs.", time.time() - start_time)


def consolidate_safetensors_files_on_every_rank(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """
    Consolidate sharded safetensors files across multiple ranks, with each rank handling a subset of output files.

    This function distributes the consolidation work by assigning output files to different ranks.
    All tensors with the same index in fqn_to_index_mapping are processed by the same rank,
    as they belong to the same output file.

    If process_group is provided, rank and world_size will be derived from it. Otherwise,
    they will be automatically detected from the distributed environment if available.

    Args:
        input_dir: Directory containing sharded safetensors files
        output_dir: Directory where consolidated files will be written
        fqn_to_index_mapping: Mapping of tensor names to output file indices
        num_threads: Number of threads to use for parallel processing on each rank
        process_group: PyTorch distributed process group (default: None, will use default group)
    """

    start_time = time.time()
    # Derive rank and world_size from process_group or default distributed environment
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
    else:
        # Default to single process mode if distributed is not initialized
        rank = 0
        world_size = 1
        logger.warning(
            "Distributed environment not initialized. Running in single process mode."
        )
    logger.info(
        "Rank %d/%d: Consolidating safetensors files from %s to %s",
        rank,
        world_size,
        input_dir,
        output_dir,
    )

    # Find all unique indices in the mapping
    unique_indices = set(fqn_to_index_mapping.values())

    # Distribute indices across ranks
    indices_for_this_rank = []
    for idx in unique_indices:
        # Simple distribution: index % world_size == rank
        if idx % world_size == rank:
            indices_for_this_rank.append(idx)

    logger.info(
        "Rank %d: Assigned %d output files out of %d total files",
        rank,
        len(indices_for_this_rank),
        len(unique_indices),
    )

    # Filter the fqn_to_index_mapping to only include tensors for this rank
    filtered_mapping = {
        fqn: idx
        for fqn, idx in fqn_to_index_mapping.items()
        if idx in indices_for_this_rank
    }

    output_files_data: dict[str, _OutputFileData] = {}
    if filtered_mapping:
        # Convert index mapping to filename mapping
        max_index = max(unique_indices)
        filtered_filename_mapping = {}
        for fqn, idx in filtered_mapping.items():
            filename = _gen_file_name(idx, max_index)
            filtered_filename_mapping[fqn] = filename

        # Call the existing consolidation function with the filtered mapping
        output_files_data = _consolidate_safetensors_files(
            input_dir=input_dir,
            output_dir=output_dir,
            fqn_to_file_mapping=filtered_filename_mapping,
            num_threads=num_threads,
        )

    logger.info(
        "Rank %d: Done consolidating. Processed %d unique indices in %.2f secs.",
        rank,
        len(indices_for_this_rank),
        time.time() - start_time,
    )

    # Wait for all ranks to complete and gather output_files_data on rank 0
    if dist.is_available() and dist.is_initialized():
        gathered_output_files_data: list[dict[str, _OutputFileData]] | None = (
            [{} for _ in range(world_size)] if rank == 0 else None
        )
        dist.gather_object(
            output_files_data,
            gathered_output_files_data,
            dst=0,
            group=process_group,
        )

        if rank == 0:
            # Merge all output_files_data from all ranks
            all_output_files_data: dict[str, _OutputFileData] = {}
            if gathered_output_files_data is None:
                raise AssertionError
            for rank_data in gathered_output_files_data:
                all_output_files_data.update(rank_data)

            _write_overall_metadata_file(output_dir, all_output_files_data)
            logger.info("Rank 0: Wrote overall metadata file.")
            logger.info("Total time taken: %.2f secs.", time.time() - start_time)
        dist.barrier(group=process_group)
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

import dataclasses
from collections.abc import Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import (
    TensorProperties as ShardTensorProperties,
)
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    create_read_items_for_chunk_list,
)

# pyrefly: ignore [deprecated]
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.checkpoint.utils import (
    _element_wise_add,
    _element_wise_sub,
    _normalize_device_info,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor import DTensor


STATE_DICT_2D_LAYOUT = dict[str, tuple[Sequence[int] | None, Sequence[int]]]


# TODO: Update docstrings for optimizer.py
__all__ = [
    "load_sharded_optimizer_state_dict",
]


def _gen_rank_device(global_rank: int, device_type: str = "cuda") -> str:
    if device_type == "cpu":
        return "cpu"
    device_module = _get_device_module(device_type)
    if device_module.is_available():
        return _normalize_device_info(
            device_type, global_rank % device_module.device_count()
        )
    return "cpu"


def _create_colwise_spec(
    pg: dist.ProcessGroup | None = None,
) -> ChunkShardingSpec:
    pg_device_type = dist.distributed_c10d._get_pg_default_device(pg).type
    if pg is None:
        placements = [
            f"rank:{idx}/{_gen_rank_device(idx, pg_device_type)}"
            for idx in range(dist.get_world_size())
        ]
    else:
        placements = [
            f"rank:{idx}/{_gen_rank_device(dist.get_global_rank(pg, idx), pg_device_type)}"
            for idx in range(pg.size())
        ]
    return ChunkShardingSpec(
        dim=0,
        placements=cast(list[_remote_device | str], placements),
    )


def _is_nested_tensor(val: torch.Tensor) -> bool:
    if type(val) is ShardedTensor:
        if len(val.local_shards()) == 0:
            return False
        if type(val.local_shards()[0].tensor) is ShardedTensor:
            return True
        if type(val.local_shards()[0].tensor) is DTensor:
            raise ValueError("Cannot handle DTensor nested inside ShardedTensor")
    elif type(val) is DTensor and (
        type(val._local_tensor) is DTensor or type(val._local_tensor) is ShardedTensor
    ):
        raise ValueError("Cannot handle nested DTensor")
    return False


def _alloc_tensor(
    props: TensorProperties, size: Sequence[int], device_type: str = "cuda"
) -> torch.Tensor:
    if device_type == "cpu":
        device = cast(torch.device, _get_device_module(device_type).current_device())
    else:
        device = torch.device(
            device_type, _get_device_module(device_type).current_device()
        )

    return torch.empty(
        size=size,
        dtype=props.dtype,
        layout=props.layout,
        requires_grad=props.requires_grad,
        pin_memory=props.pin_memory,
        device=device,
    )


def _get_state_dict_2d_layout(
    state_dict: STATE_DICT_TYPE,
) -> tuple[STATE_DICT_2D_LAYOUT, dist.ProcessGroup | None]:
    """
    Load the right TP slice of the optimizer state.

    This is not easy since the per-tensor slicing can't be inferred from checkpoint metadata.
    We take advantage of the model state_dict producing a sliced ST to figure out what we need to load.
    This is pretty fragile and it might be easier for FSDP to compute this info for us.
    Returns a dictionary where keys are the same of the state_dict and the value is a tuple of
    (offset, size) for the current rank TP slice.
    N.B. The state_dict *MUST* come from FSDP.sharded_state_dict.
    """
    specs: STATE_DICT_2D_LAYOUT = {}
    dp_pg: dist.ProcessGroup | None = None
    for key, value in state_dict.items():
        specs[key] = (None, value.size())
        if _is_nested_tensor(value):
            if not len(value.local_shards()) == 1:
                raise AssertionError("Cannot handle ST with multiple shards")
            if not isinstance(value, ShardedTensor):
                raise AssertionError("Can only handle nested ShardedTensor")
            shard = value.local_shards()[0]
            specs[key] = (
                shard.metadata.shard_offsets,
                shard.metadata.shard_sizes,
            )
            dp_pg = shard.tensor._process_group  # type: ignore[attr-defined]

    return (
        specs,
        dp_pg,
    )


class _ReaderWithOffset(DefaultLoadPlanner):
    translation: dict[MetadataIndex, MetadataIndex]
    state_dict: STATE_DICT_TYPE
    # pyrefly: ignore [bad-override]
    metadata: Metadata

    def __init__(self, fqn_to_offset: dict[str, Sequence[int]]) -> None:
        super().__init__()
        self.fqn_to_offset = fqn_to_offset
        self.metadata = Metadata({})
        self.state_dict = {}
        self.translation = {}

    def create_local_plan(self) -> LoadPlan:
        requests = []
        self.translation = {}
        for fqn, obj in self.state_dict.items():
            md = self.metadata.state_dict_metadata[fqn]
            if not isinstance(obj, ShardedTensor):
                requests += _create_read_items(fqn, md, obj)
                continue

            if fqn not in self.fqn_to_offset:
                requests += _create_read_items(fqn, md, obj)
                continue

            offset = self.fqn_to_offset[fqn]

            if not len(obj.local_shards()) == 1:
                raise AssertionError("Expected exactly one local shard")
            original_shard = obj.local_shards()[0]
            local_chunks = [
                ChunkStorageMetadata(
                    offsets=torch.Size(
                        _element_wise_add(original_shard.metadata.shard_offsets, offset)
                    ),
                    sizes=torch.Size(original_shard.metadata.shard_sizes),
                )
            ]

            reqs = create_read_items_for_chunk_list(
                fqn, cast(TensorStorageMetadata, md), local_chunks
            )
            # TODO: The ReadItems will have a displaced MetadataIndex, fix it.
            # TODO: we should change _create_sharded_read_items to have more ergonomic API
            for ri in reqs:
                if ri.dest_index.offset is None:
                    raise AssertionError("dest_index.offset must not be None")
                original_offset = _element_wise_sub(ri.dest_index.offset, offset)
                original_index = dataclasses.replace(
                    ri.dest_index, offset=torch.Size(original_offset)
                )
                self.translation[ri.dest_index] = original_index

            requests += reqs
        return LoadPlan(requests)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        return super().lookup_tensor(self.translation.get(index, index))


def load_sharded_optimizer_state_dict(
    model_state_dict: STATE_DICT_TYPE,
    optimizer_key: str,
    storage_reader: StorageReader,
    planner: LoadPlanner | None = None,
) -> STATE_DICT_TYPE:
    """
    Load a state_dict in conjunction with FSDP sharded optimizer state.

    This is the current recommended way to checkpoint FSDP.

    Examples::

    >>> # xdoctest: +SKIP
    >>> import torch.distributed.checkpoint as dist_cp
    >>> # Save
    >>> model: torch.nn.Model
    >>> optim_params = model.parameters()
    >>> optim = torch.optim.SGD(optim_params, lr=0.01)
    >>> # Save
    >>> with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    >>>     state_dict = {
    >>>         "optimizer": FSDP.optim_state_dict(model, optim),
    >>>         "model": model.state_dict()
    >>>     }
    >>>     dist_cp.save_state_dict(
    >>>         state_dict=optim_state,
    >>>         storage_writer=dist_cp.FileSystemWriter("checkpoint"),
    >>>         planner=dist_cp.DefaultSavePlanner(),
    >>>     )
    >>>
    >>> # Load
    >>> with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
    >>>     model_state_dict = model_tp.state_dict()
    >>>     checkpoint = {
    >>>         "model": model_state_dict
    >>>     }
    >>>     dist_cp.load_state_dict(
    >>>         state_dict=checkpoint,
    >>>         storage_reader=dist_cp.FileSystemReader(checkpoint_file),
    >>>         planner=dist_cp.DefaultLoadPlanner(),
    >>>     )
    >>>     model.load_state_dict(checkpoint["model_state"])
    >>>
    >>>     optim_state = dist_cp.load_sharded_optimizer_state_dict(
    >>>         model_state_dict,
    >>>         optimizer_key="optimizer",
    >>>         storage_reader=dist_cp.FileSystemReader("checkpoint"),
    >>>     )
    >>>
    >>>     flattened_osd = FSDP.optim_state_dict_to_load(
    >>>        model, optim, optim_state["optimizer"]
    >>>     )
    >>>
    >>>     optim.load_state_dict(flattened_osd)
    """
    metadata = storage_reader.read_metadata()

    layout_specs, dp_pg = _get_state_dict_2d_layout(model_state_dict)
    dp_pg_device_type = dist.distributed_c10d._get_pg_default_device(dp_pg).type
    device_module = _get_device_module(dp_pg_device_type)

    if dp_pg is None:
        placements = []
        for i in range(dist.get_world_size()):
            device_info = _normalize_device_info(
                dp_pg_device_type, i % device_module.device_count()
            )
            placements.append(f"rank:{i}/{device_info}")
        sharding_spec = ChunkShardingSpec(dim=0, placements=placements)  # type: ignore[arg-type]
    else:
        sharding_spec = _create_colwise_spec(dp_pg)

    # Create a state_dict for optimizer state
    state_dict: STATE_DICT_TYPE = {}

    fqn_to_offset: dict[str, Sequence[int]] = {}
    for key, value in metadata.state_dict_metadata.items():
        key_path = metadata.planner_data[key]
        if key_path[0] != optimizer_key:
            continue

        if isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
            continue

        # value: TensorStorageMetadata
        if value.size.numel() == 1:
            state_dict[key] = _alloc_tensor(
                value.properties, value.size, dp_pg_device_type
            )
        elif dp_pg is None:
            state_dict[key] = _create_chunk_sharded_tensor(
                _alloc_tensor(value.properties, value.size, dp_pg_device_type),
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                num_devices_per_node=device_module.device_count(),
                pg=_get_default_group(),
            )
        else:
            spec_key = key_path[2]
            alloc_size = layout_specs.get(spec_key, (None, value.size))[1]

            properties = ShardTensorProperties(
                dtype=value.properties.dtype,
                layout=value.properties.layout,
                requires_grad=value.properties.requires_grad,
                memory_format=value.properties.memory_format,
                pin_memory=value.properties.pin_memory,
            )

            st_md = sharding_spec.build_metadata(torch.Size(alloc_size), properties)
            local_shards = []
            current_rank = dist.get_rank(dp_pg)
            for shard_md in st_md.shards_metadata:
                if cast(_remote_device, shard_md.placement).rank() != current_rank:
                    continue
                local_shards.append(
                    Shard(
                        tensor=_alloc_tensor(
                            value.properties, shard_md.shard_sizes, dp_pg_device_type
                        ),
                        metadata=shard_md,
                    )
                )

            st = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards, st_md, process_group=dp_pg
            )

            if spec_key in layout_specs and layout_specs[spec_key][0] is not None:
                fqn_to_offset[key] = cast(Sequence[int], layout_specs[spec_key][0])

            state_dict[key] = st

    # Whether we unflatten before or after doesn't matter
    load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        # FIXME the type of planner is wrong in load_state_dict
        planner=_ReaderWithOffset(fqn_to_offset) if dp_pg is not None else planner,
    )

    state_dict = unflatten_state_dict(state_dict, metadata.planner_data)

    return state_dict
<|endoftext|># pyre-strict
# mypy: allow-untyped-defs
import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch.distributed as dist
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter


def save_wrapper(
    staging_future_or_state_dict: Future[STATE_DICT_TYPE] | STATE_DICT_TYPE,
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_writer: StorageWriter | None = None,
    planner: SavePlanner | None = None,
    process_group: dist.ProcessGroup | None = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Future:
    from torch.distributed.checkpoint.state_dict_saver import save

    staged_dict = (
        staging_future_or_state_dict.result()
        if isinstance(staging_future_or_state_dict, Future)
        else staging_future_or_state_dict
    )
    return save(
        staged_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )


class _ThreadBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="AsyncCheckpointExecutor"
        )

    def execute_save(
        self,
        staging_future_or_state_dict: Future[STATE_DICT_TYPE] | STATE_DICT_TYPE,
        *,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        process_group: dist.ProcessGroup | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future:
        f: Future = self._executor.submit(
            save_wrapper,
            staging_future_or_state_dict=staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            process_group=process_group,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        f.add_done_callback(lambda f: self._executor.shutdown(wait=False))

        return f
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import functools
import gc
import warnings
from collections.abc import Callable, Generator, Iterable
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Any, cast, no_type_check

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._state_dict_utils import (
    _broadcast_state_dict,
    _distribute_state_dict,
    _flatten_state_dict,
    _gather_state_dict,
    _offload_state_dict_to_cpu,
    _unflatten_state_dict,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
)
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state_if_fully_sharded_module,
    FSDP_WRAPPED_MODULE,
)
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils._pytree import tree_map_only


__all__ = [
    "FQNS_T",
    "PrimitiveType",
    "ValueType",
    "DictValueType",
    "ListDictValueType",
    "OptimizerStateType",
    "StateDictOptions",
    "get_model_state_dict",
    "get_optimizer_state_dict",
    "get_state_dict",
    "set_model_state_dict",
    "set_optimizer_state_dict",
    "set_state_dict",
]


_FLAT_PARAM = "_flat_param"
_PG = "param_groups"
_PARAMS = "params"
_STATE = "state"

FQNS_T = set[str]
PrimitiveType = DTensor | ShardedTensor | torch.Tensor | int | float | str
ValueType = (
    PrimitiveType | list[PrimitiveType] | tuple[PrimitiveType] | dict[str, "ValueType"]
)
DictValueType = dict[str, ValueType]
ListDictValueType = list[DictValueType]
OptimizerStateType = dict[str, DictValueType | ListDictValueType]


_patched_state_dict: set[Callable] = set()


@contextlib.contextmanager
def _gc_context():
    is_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if is_enabled:
            gc.enable()


@dataclass
class StateDictOptions:
    """
    This dataclass specifies how get_state_dict/set_state_dict will work.

    - ``full_state_dict``: if this is set to True, all the tensors in the
      returned state_dict will be gathered. No ShardedTensor and DTensor
      will be in the returned state_dict.

    - ``cpu_offload``: offload all the tensors to cpu. To prevent CPU OOM, if
      ``full_state_dict`` is also true, then only the rank0 will get the
      state_dict and all other ranks will get empty state_dict.

    - ``ignore_frozen_params``: if the value is True, the returned state_dict
      won't contain any frozen parameters -- the ``requires_grad`` is False.
      The default value is False.

    - ``keep_submodule_prefixes`` (deprecated): when ``submodules`` is not None, this option
      indicates whether to keep the submodule prefixes from the state_dict keys.
      or example, if the submodule is ``module.pretrain`` and the full FQN of
      the parameter is ``pretrain.layer1.weight`` of the param. When this option
      is True, the parameter's key in the returned state_dict will be
      ``pretrain.layer1.weight``. If the options is False, the key will be
      ``layer1.weight``.
      Note that if ``keep_submodule_prefixes`` is False, there may be conflicted
      FQNs, hence there should be only one submodule in ``submodules``.

    - ``strict``: the ``strict`` option when ``set_state_dict`` calls
      model.load_state_dict().

    - ``broadcast_from_rank0``: when the option is True, rank0 should receive a
       full state_dict and will broadcast the tensors in the state_dict/
       optim_state_dict one by one to other ranks. Other ranks will receive
       the tensors and shard according to the local shards in the model and
       optimizer. ``full_state_dict`` must be set to True when using this option.
       This option currently only supports DTensor, not the legacy ShardedTensor.
    """

    full_state_dict: bool = False
    cpu_offload: bool = False
    ignore_frozen_params: bool = False
    keep_submodule_prefixes: bool = True
    strict: bool = True
    broadcast_from_rank0: bool = False
    flatten_optimizer_state_dict: bool = False
    dsd_fqn_modifiers: str = "_fqn_modifiers"


@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: dict[
        str | torch.Tensor,
        FQNS_T | torch.Tensor,
    ] = field(default_factory=dict)
    shared_params_mapping: dict[
        str | torch.Tensor,
        FQNS_T | torch.Tensor,
    ] = field(default_factory=dict)
    submodule_prefixes: set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable = contextlib.nullcontext
    fsdp_modules: list[nn.Module] = field(default_factory=list)


def _get_fqns(
    model: nn.Module,
    name: str,
    dsd_fqn_modifiers: str = "_fqn_modifiers",
    skip_ddp_prefix: bool = True,
    skip_compiler_prefix: bool = True,
) -> FQNS_T:
    """
    This API is used to convert the name of a parameter to the FQNs. For FSDP
    without `use_orig_params`, the name of FlatParameter can be mapped to
    multiple original parameters. As a result, the return type of this function
    is `set[str]`.

    Args:
        module (nn.Module): the root model.
        name (str): the name
        skip_ddp_prefix (bool): whether to skip DDP's `module` prefix

    Returns:
        The canonical FQNs based on the model traversal.
    """

    # Remove the checkpoint prefix, if it exists.
    name = name.replace(_CHECKPOINT_PREFIX, "")
    if "." not in name:
        return {name}

    obj_names = name.split(".")
    fqn_obj_names = []
    curr_obj = model
    for i, curr_obj_name in enumerate(obj_names):
        if isinstance(curr_obj, DDP):
            if curr_obj_name != "module":
                raise AssertionError(f"Expected 'module', got '{curr_obj_name}'")
            curr_obj = curr_obj.module
            if not skip_ddp_prefix:
                fqn_obj_names.append(curr_obj_name)
        elif isinstance(curr_obj, FSDP):
            if i < len(obj_names) - 1 and obj_names[i + 1] == _FLAT_PARAM:
                prefix = ".".join(fqn_obj_names)
                flat_param = getattr(curr_obj, _FLAT_PARAM)
                if prefix:
                    prefix = f"{prefix}."
                return {f"{prefix}{fqn}" for fqn in flat_param._fqns}
            curr_obj = getattr(curr_obj, FSDP_WRAPPED_MODULE)
            if curr_obj_name != FSDP_WRAPPED_MODULE:
                fqn_obj_names.append(curr_obj_name)
                curr_obj = getattr(curr_obj, curr_obj_name)
        elif isinstance(curr_obj, torch._dynamo.eval_frame.OptimizedModule):
            if curr_obj_name != "_orig_mod":
                raise AssertionError(f"Expected '_orig_mod', got '{curr_obj_name}'")
            curr_obj = curr_obj._orig_mod
            if not skip_compiler_prefix:
                fqn_obj_names.append(curr_obj_name)
        else:
            # In some modules, _fqn_modifiers would not shown in the state_dict keys,
            # skip them in the fqn to ensure load stat dict successfully for them.
            if hasattr(curr_obj, dsd_fqn_modifiers):
                if removed_fqn := getattr(curr_obj, dsd_fqn_modifiers)().get(
                    curr_obj_name
                ):
                    if hasattr(curr_obj, removed_fqn):
                        curr_obj = getattr(curr_obj, removed_fqn)
            fqn_obj_names.append(curr_obj_name)
            if curr_obj_name == nn.modules.module._EXTRA_STATE_KEY_SUFFIX:
                if i != len(obj_names) - 1:
                    raise RuntimeError("Expect `_extra_state` to be the last obj name")
            else:
                curr_obj = getattr(curr_obj, curr_obj_name)

    return {".".join(fqn_obj_names).replace(_CHECKPOINT_PREFIX, "")}


class _EXTRA_STATE:
    pass


def _iterate_valid_model_state(model, dsd_fqn_modifiers="_fqn_modifiers"):
    visited_modules: set[nn.Module] = set()

    def recurse(module: nn.Module, curr_fqn: str) -> Generator:
        visited_modules.add(module)

        curr_fqn = f"{curr_fqn}." if curr_fqn else ""
        for name, submodule in module.named_children():
            if submodule in visited_modules:
                continue
            # if user have state_dict_hooks in their model, they can add the state_dict key changes
            # at dsd_fqn_modifiers in input to align with the function of state_dict_hook
            if (
                hasattr(module, dsd_fqn_modifiers)
                and name in getattr(module, dsd_fqn_modifiers)().values()
            ):
                # skip _fqn_modifiers here thus remove the last `.` added
                new_fqn = curr_fqn[:-1]
            else:
                new_fqn = f"{curr_fqn}{name}"
            yield from recurse(submodule, new_fqn)

        for name, obj in chain(
            module.named_buffers(recurse=False), module.named_parameters(recurse=False)
        ):
            if name in module._non_persistent_buffers_set:
                continue
            new_fqn = f"{curr_fqn}{name}"
            yield new_fqn, obj

        if (
            getattr(module.__class__, "get_extra_state", nn.Module.get_extra_state)
            != nn.Module.get_extra_state
        ):
            new_fqn = f"{curr_fqn}{nn.modules.module._EXTRA_STATE_KEY_SUFFIX}"
            yield new_fqn, _EXTRA_STATE()

    yield from recurse(model, "")


def _verify_options(
    model: nn.Module,
    optims: tuple[torch.optim.Optimizer, ...],
    optim_only: bool,
    *,
    submodules: set[nn.Module] | None = None,
    options: StateDictOptions | None = None,
) -> _StateDictInfo:
    """
    Verify the model and options passed by the user and generates _StateDictInfo.
    """
    if submodules:
        warnings.warn(
            "Getting submodules only model/optim state_dict is deprecated and "
            "will be removed in 2.5. This feature can be achieved by manually "
            "filtering out the state_dict returned from get_state_dict.",
            FutureWarning,
            stacklevel=2,
        )
    if optim_only and not optims:
        raise RuntimeError(
            "Optimizers are not passed in but optim_only is set to True."
        )

    options = options or StateDictOptions()

    fqn_param_mapping: dict[str | torch.Tensor, set[str] | torch.Tensor] = {}
    shared_params_mapping: dict[str | torch.Tensor, set[str] | torch.Tensor] = {}
    for name, param in _iterate_valid_model_state(model):
        if isinstance(param, _EXTRA_STATE):
            continue

        fqns = _get_fqns(model, name)
        fqn = fqn_param_mapping.get(param)
        if fqn is not None:
            cast(set[str], fqn_param_mapping[param]).update(fqns)
            shared_params_mapping[param] = fqn_param_mapping[param]
        else:
            # We need to do copy as _get_fqns is lru_cached
            fqn_param_mapping[param] = fqns.copy()
        for fqn in fqns:
            if not isinstance(param, _EXTRA_STATE):
                fqn_param_mapping[fqn] = param

    for param_, fqns_ in list(shared_params_mapping.items()):
        for fqn in fqns_:
            shared_params_mapping[fqn] = cast(torch.Tensor, param_)

    submodule_prefixes: set[str] = set()
    if submodules:
        submodules = set(submodules)
        for name, module in model.named_modules():
            if module not in submodules:
                continue
            fqns = _get_fqns(model, name)
            if len(fqns) != 1:
                raise AssertionError("Submodule FQN should only have 1 instance")
            submodule_prefixes.update(f"{fqn}." for fqn in fqns)

    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError(
            "full_state_dict must be True when broadcast_from_rank0 is True."
        )
    fsdp_modules = FSDP.fsdp_modules(model)
    state_dict_config: StateDictConfig
    optim_state_dict_config: OptimStateDictConfig
    fsdp_context: Callable
    if fsdp_modules:
        # FSDP API only work if at least one FSDP instance exists.
        if options.full_state_dict:
            state_dict_config = FullStateDictConfig(
                offload_to_cpu=options.cpu_offload, rank0_only=options.cpu_offload
            )
            optim_state_dict_config = FullOptimStateDictConfig(
                offload_to_cpu=options.cpu_offload,
                rank0_only=(options.cpu_offload or options.broadcast_from_rank0),
            )
            state_dict_type = StateDictType.FULL_STATE_DICT
        else:
            state_dict_config = ShardedStateDictConfig(
                offload_to_cpu=options.cpu_offload,
            )
            optim_state_dict_config = ShardedOptimStateDictConfig(
                offload_to_cpu=options.cpu_offload,
            )
            state_dict_type = StateDictType.SHARDED_STATE_DICT

        @contextlib.contextmanager
        def fsdp_state_dict_type_without_warning(
            module,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="FSDP.state_dict_type", category=FutureWarning
                )
                with FSDP.state_dict_type(
                    module=module,
                    state_dict_type=state_dict_type,
                    state_dict_config=state_dict_config,
                    optim_state_dict_config=optim_state_dict_config,
                ):
                    yield

        fsdp_context = functools.partial(
            fsdp_state_dict_type_without_warning,
            module=model,
            state_dict_type=state_dict_type,
            state_dict_config=state_dict_config,
            optim_state_dict_config=optim_state_dict_config,
        )
    else:
        fsdp_context = contextlib.nullcontext

    return _StateDictInfo(
        **asdict(options),
        fqn_param_mapping=fqn_param_mapping,
        shared_params_mapping=shared_params_mapping,
        submodule_prefixes=submodule_prefixes,
        fsdp_context=fsdp_context,
        fsdp_modules=cast(list[nn.Module], fsdp_modules),
        handle_model=not optim_only,
        handle_optim=(len(optims) > 0),
    )


def _verify_state_dict(
    model_state_dict: dict[str, ValueType],
    optim_state_dict: OptimizerStateType,
