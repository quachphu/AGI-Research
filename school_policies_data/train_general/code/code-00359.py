                torch.ops.aten.split_with_sizes.default, users=MULTIPLE
            ),
            Ignored(),
            _users=MULTIPLE,
        ),
        Arg(),
        _users=MULTIPLE,
    ),
    partial=True,
)


@register_graph_pattern(
    CallFunction(
        torch.ops.aten.cat.default,
        view_getitem_split_aten,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=construct_pattern_matcher_pass("move_view_after_cat_aten_pass"),
)
def move_view_after_cat(match: Match, *args, **kwargs):
    split_node = next(
        node
        for node in match.nodes
        if node.target is torch.ops.aten.split_with_sizes.default
    )
    split_input, split_section, split_dim = _get_split_args_default(split_node)
    split_users = list(split_node.users.keys())
    getitem_indices = [
        getitem.args[1] for getitem in split_users if getitem.target is operator.getitem
    ]
    if not is_sorted_and_consecutive(getitem_indices):  # type: ignore[arg-type]
        return
    cat_nodes = [
        node for node in match.nodes if node.target is torch.ops.aten.cat.default
    ]
    graph = match.graph
    for cat_node in cat_nodes:
        if not is_node_meta_valid(cat_node):
            log.debug("example value absent for node: %s", cat_node)
            continue
        cat_dim = _get_dim(cat_node)
        cat_inputs = get_arg_value(cat_node, 0, "tensors")  # type: ignore[union-attr]
        # we only consider the following special case
        if len(cat_inputs) != len(split_section):
            continue
        # check if the cat inputs are all the view nodes
        if not all(
            view_node.target is torch.ops.aten.reshape.default
            for view_node in cat_inputs
        ):
            continue
        # check if the view nodes are all from getitem nodes
        if not all(
            view_node.args[0].target is operator.getitem for view_node in cat_inputs
        ):
            continue
        view_indices = [view.args[0].args[1] for view in cat_inputs]
        if not is_sorted_and_consecutive(view_indices):  # type: ignore[arg-type]
            continue
        if cat_dim != split_dim:
            # construct permute node
            permute_list = list(range(len(cat_node.meta["val"].shape) + 1))
            permute_list[split_dim], permute_list[cat_dim] = (
                permute_list[cat_dim],
                permute_list[split_dim],
            )
            permute_node = graph.call_function(
                torch.ops.aten.permute.default,
                args=(split_input, permute_list),
            )
        else:
            permute_node = split_input

        with graph.inserting_before(cat_node):
            view_node = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(permute_node, list(cat_node.meta["val"].shape)),
            )
            cat_node.replace_all_uses_with(view_node)
            view_node.meta.update(cat_node.meta)
            graph.erase_node(cat_node)
        counters[backend]["move_view_after_cat_aten_pass"] += 1


def match_einsum_strings(s: str) -> bool:
    """
    This function takes a string s as input, where s is in the format "3 letter string,
    4 letter string -> 3 letter string".
    It checks if the strings match the rule and returns True if they do, False otherwise.

    The rule is:
    - The three strings have the same first two characters.
    - The first two strings have the same third character.
    - The second and third strings have the same last character.
    """

    # Split the input string into parts
    parts = s.replace("->", ",").split(",")

    # Strip leading/trailing whitespaces from each part
    parts = [part.strip() for part in parts]

    # Check if we have exactly three parts
    if len(parts) != 3:
        return False

    # Extract the strings
    s1, s2, s3 = parts

    # Check if the strings have the correct lengths
    if len(s1) != 3 or len(s2) != 4 or len(s3) != 3:
        return False

    # Check the rule
    return s1[:2] == s2[:2] == s3[:2] and s1[2] == s2[2] and s2[3] == s3[2]


@register_graph_pattern(
    CallFunctionVarArgs(torch.functional.einsum, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("einsum_to_pointwise_pass"),
)
def replace_einsum_to_pointwise(match: Match, *args, **kwargs):
    def repl(input, weights):
        return (input.unsqueeze(-1) * weights).sum(-2)

    def should_replace_einsum(einsum_node) -> bool:
        equation = get_arg_value(einsum_node, 0)
        users = einsum_node.users.keys()
        # for now, we only consider the case of two operands
        return (
            len(einsum_node.args) == 3
            and is_node_meta_valid(input)
            and is_node_meta_valid(weights)
            and any(
                user.target == "add" or user.target is operator.add for user in users
            )
            and match_einsum_strings(equation)
        )

    einsum_node = match.nodes[0]
    input, weights = get_arg_value(einsum_node, 1), get_arg_value(einsum_node, 2)
    if should_replace_einsum(einsum_node):
        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(repl, [input, weights])
        counters[backend]["einsum_to_pointwise_pass"] += 1
<|endoftext|># mypy: allow-untyped-defs
import copy
import functools
import itertools
import logging
import types
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch._dynamo.utils import counters, detect_fake_mode
from torch._logging import trace_structured
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.fx.passes.graph_transform_observer import (
    GraphTransformObserver as GraphTransformObserverBase,
)
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..custom_graph_pass import get_custom_graph_passes
from ..fx_utils import matches_module_function_pattern
from ..pattern_matcher import (
    init_once_fakemode,
    PatternMatcherPass as PatternMatcherPassBase,
    stable_topological_sort,
)
from ..utils import is_cpu_device, pass_execution_and_save
from .group_batch_fusion import group_batch_fusion_passes, PRE_GRAD_FUSIONS
from .misc_patterns import numpy_compat_normalization
from .split_cat import PRE_GRAD_PATTERNS


PatternMatcherPass = functools.partial(
    PatternMatcherPassBase, subsystem="pre_grad_passes"
)
GraphTransformObserver = functools.partial(
    GraphTransformObserverBase, subsystem="pre_grad_passes"
)

log = logging.getLogger(__name__)

apply_gumbel_max_trick_pass = PatternMatcherPass(
    pass_name="apply_gumbel_max_trick_pass"
)

efficient_conv_bn_eval_pass = PatternMatcherPass(
    pass_name="efficient_conv_bn_eval_pass"
)

fuse_split_linear_add_pass = PatternMatcherPass(
    pass_name="fuse_split_linear_add_pass",
)
fuse_chunk_squeeze_cat_pass = PatternMatcherPass(
    pass_name="fuse_chunk_squeeze_cat_pass",
)
remove_reshape_pass = PatternMatcherPass(
    pass_name="remove_reshape_pass",
)

# based on predispatch aten IR
normalization_pass_aten = PatternMatcherPass(pass_name="normalization_pass_aten")
merge_splits_pass_aten = PatternMatcherPass(pass_name="merge_splits_pass_aten")
split_cat_pass_aten = PatternMatcherPass(pass_name="split_cat_pass_aten")
unbind_stack_pass_aten = PatternMatcherPass(pass_name="unbind_stack_pass_aten")
merge_getitem_cat_pass_aten = PatternMatcherPass(
    pass_name="merge_getitem_cat_pass_aten"
)
merge_stack_tahn_unbind_pass_aten = PatternMatcherPass(
    pass_name="merge_stack_tahn_unbind_pass_aten"
)
mutate_cat_pass_aten = PatternMatcherPass(pass_name="mutate_cat_pass_aten")
remove_split_with_size_one_pass_aten = PatternMatcherPass(
    pass_name="remove_split_with_size_one_pass_aten"
)


def save_inductor_dict(pass_to_compare=None):
    if not pass_to_compare:
        pass_to_compare = list(config.pre_grad_fusion_options.keys()) + list(
            config.post_grad_fusion_options.keys()
        )
    return {p: dict(counters["inductor"]).get(p, 0) for p in pass_to_compare}


def is_same_dict(inductor_dict, optimus_dict):
    for pass_name, count in optimus_dict.items():
        if count != dict(inductor_dict).get(pass_name, 0):
            return False
    return True


def shape_prop(mod) -> None:
    return None


def normalize_node_kwargs_pass(graph):
    return None


def fuse_parallel_linear_pass(graph):
    return None


def remove_split_ops(graph, shape_prop):
    return None


def remove_split_ops_pass(graph):
    remove_split_ops(graph.owning_module, shape_prop)


def fuse_chunk_reshape_unsqueeze_concat_pass(graph):
    return None


def fuse_chunk_reshape_concat_pass(graph):
    return None


def remove_noop_pass(graph):
    return None


def stack_to_unsqueeze_pass(graph):
    return None


def merge_concats_pass(graph):
    return None


def relu_nan_to_num(graph):
    return None


def fuse_split_getitem_squeeze_cat(graph):
    return None


def use_triton_dot_compress(graph):
    return None


def use_triton_lce_replace_simple_LCE_helper(gm, shape_prop):
    return None


def use_triton_lce_replace_simple_LCE(graph):
    return use_triton_lce_replace_simple_LCE_helper(graph.owning_module, shape_prop)


def use_triton_lce_replace_normal_LCE_helper(gm, shape_prop):
    return None


def use_triton_lce_replace_normal_LCE(graph):
    return use_triton_lce_replace_simple_LCE_helper(graph.owning_module, shape_prop)


def use_matmul_lce_replace_normal_LCE(graph):
    return None


def use_matmul_fuse_lce_replace_first_LCE(graph):
    return None


@init_once_fakemode
def lazy_init(input_device: torch.device | None = None):
    from . import (  # noqa: F401  # noqa: F401
        apply_gumbel_max_trick,
        efficient_conv_bn_eval,
        split_cat,
    )

    if config.is_fbcode():
        from . import fb  # type: ignore[attr-defined]  # noqa: F401


def _get_pass_name_func(p):
    if isinstance(p, PatternMatcherPassBase):
        pass_name = p.pass_name
        pass_func = p.apply
    elif isinstance(p, types.FunctionType):
        pass_name = p.__name__.lstrip("_")
        pass_func = p
    else:
        pass_name = None
        pass_func = None

    return pass_name, pass_func


def _run_pre_dispatch_passes(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[object] = (),
    add_passes: str | None = None,
    remove_passes: str | None = None,
) -> None:
    # Wrapper that resolves pre_grad_fusion_options by device before
    # calling group_batch_fusion_passes, so the predispatch path
    # gets the same device filtering as the non-predispatch path.
    def _group_batch_fusion_passes_wrapper(graph: torch.fx.Graph) -> None:
        group_batch_fusion_passes(
            graph,
            pre_grad=True,
            fusion_options=_resolve_pre_grad_fusion_options(
                config.pre_grad_fusion_options, example_inputs
            ),
        )

    _group_batch_fusion_passes_wrapper.__name__ = "group_batch_fusion_passes"

    # order matters
    default_pass_list = [
        # normalize passes, must be called as the first passes
        normalization_pass_aten,
        normalize_node_kwargs_pass,
        remove_noop_pass,
        relu_nan_to_num,
        fuse_chunk_reshape_concat_pass,
        _group_batch_fusion_passes_wrapper,
        normalize_node_kwargs_pass,
        fuse_chunk_squeeze_cat_pass,
        merge_concats_pass,
        fuse_split_linear_add_pass,
        remove_reshape_pass,
        fuse_parallel_linear_pass,
        remove_split_ops_pass,
        stack_to_unsqueeze_pass,  # run before fuse_chunk_reshape_unsqueeze_concat_pass
        fuse_chunk_reshape_unsqueeze_concat_pass,
    ]

    full_pass_list = default_pass_list + [
        fuse_split_getitem_squeeze_cat,
        use_triton_dot_compress,
        use_triton_lce_replace_simple_LCE,
        use_triton_lce_replace_normal_LCE,
        use_matmul_fuse_lce_replace_first_LCE,
        use_matmul_lce_replace_normal_LCE,
    ]

    log.info(
        f"pre_grad_passes: add_passes: {add_passes}, remove_pass: {remove_passes}"  # noqa: G004
    )
    add_passes_list = []
    remove_passes_list = []
    if add_passes:
        add_passes_list = add_passes.split(",")
    if remove_passes:
        remove_passes_list = remove_passes.split(",")

    shape_prop = lambda mod: ShapeProp(  # noqa: E731
        gm=mod,
        # pyre-fixme[16]: Module `torch._dynamo.utils` has no attribute `detect_fake_mode`
        fake_mode=detect_fake_mode(example_inputs),
    ).propagate(*tuple(example_inputs))

    for p in default_pass_list:
        pass_name, pass_func = _get_pass_name_func(p)
        # should not happen
        if pass_name is None or pass_func is None:
            continue
        if pass_name in remove_passes_list:
            continue
        pass_execution_and_save(
            pass_func,
            gm,
            example_inputs,
            f"[Pre grad(predispatch IR)] Apply {pass_name} pass",
        )

    for p in full_pass_list:
        pass_name, pass_func = _get_pass_name_func(p)
        if pass_name is None or pass_func is None:
            continue
        if pass_name in add_passes_list:
            pass_execution_and_save(
                pass_func,
                gm,
                example_inputs,
                f"[Pre grad(predispatch IR)] Apply {pass_name} pass",
            )

    if "remove_noop" not in remove_passes_list:
        # Remove noops at the end, which may be generated by other passes.
        pass_execution_and_save(
            remove_noop_pass,
            gm,
            example_inputs,
            "[Pre grad(predispatch IR)]Apply remove_noop pass",
        )
    shape_prop(gm)


def _resolve_pre_grad_fusion_options(
    fusion_options: dict[str, dict[str, object]],
    example_inputs: Sequence[object] | None,
) -> dict[str, dict[str, object]]:
    """Filter pre_grad_fusion_options by device compatibility.

    If an option has a "devices" key, only include it when at least one
    example input tensor's device type matches. Fusions without a "devices"
    key always pass through (user explicitly opted in). The "devices" key
    is stripped from the returned options.
    """
    if example_inputs is None:
        resolved: dict[str, dict[str, object]] = {}
        for pass_name, options in fusion_options.items():
            resolved[pass_name] = {k: v for k, v in options.items() if k != "devices"}
        return resolved

    example_device_types = OrderedSet(
        t.device.type for t in example_inputs if isinstance(t, torch.Tensor)
    )

    resolved = {}
    for pass_name, options in fusion_options.items():
        devices = options.get("devices")
        if (
            devices is not None
            and isinstance(devices, (list, tuple, OrderedSet))
            and example_device_types.isdisjoint(OrderedSet(devices))
        ):
            continue
        resolved[pass_name] = {k: v for k, v in options.items() if k != "devices"}
    return resolved


def pre_grad_passes(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[object] = (),
    add_passes: str | None = None,
    remove_passes: str | None = None,
) -> torch.fx.GraphModule:
    """
    Apply passes on the input FX graph using Torch IR.

    WARNING:
    The IR before grad is not functional or normalized, so it is harder
    to write passes on this IR.  Passes must be safe with respect to
    aliasing and mutation and need to handle all possible arg schemas.

    Consider adding a new pass to post_grad.py or joint_graph.py which
    are after functionalization and normalization.
    """
    if config.pattern_matcher:
        lazy_init()
        if hasattr(
            config, "fx_passes_numeric_check"
        ) and config.fx_passes_numeric_check.get("pre_grad", False):
            gm_before_fx_passes = gm.__copy__()
        # explicitly run with predispatch atenIR based passes
        if config.is_predispatch:
            _run_pre_dispatch_passes(gm, example_inputs, add_passes, remove_passes)
        else:
            # We only log the graph with changes to avoid the excessive compilation time
            # https://fb.workplace.com/groups/257735836456307/permalink/633533465543207/
            numpy_compat_normalization(gm.graph)
            if example_inputs is not None:
                gm = fuse_fx(gm, example_inputs)
            # Resolve pre_grad_fusion_options by filtering on device type
            # declared in each fusion's "devices" config key. Fusions without
            # a "devices" key always pass through. Uses a local copy to avoid
            # mutating global config.
            fusion_options = _resolve_pre_grad_fusion_options(
                config.pre_grad_fusion_options, example_inputs
            )
            # We should always do the normalization_pass first
            if "normalization_pass" in fusion_options:
                pattern_matcher_pass = PRE_GRAD_PATTERNS["normalization_pass"]
                pattern_matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
            GraphTransformObserver(gm, "group_batch_fusion_passes").apply_graph_pass(
                lambda graph: group_batch_fusion_passes(
                    graph, pre_grad=True, fusion_options=fusion_options
                )
            )
            for pass_name in fusion_options:
                # skip all patterns for group batch fusions
                if pass_name in PRE_GRAD_FUSIONS or pass_name == "normalization_pass":
                    continue
                # skip auto-enabled fusion options that are not registered
                # as pattern matcher passes (e.g., batch_linear_lhs when
                # PRE_GRAD_FUSIONS is mocked by tests)
                if pass_name not in PRE_GRAD_PATTERNS:
                    continue
                pattern_matcher_pass = PRE_GRAD_PATTERNS[pass_name]
                inductor_before_change = save_inductor_dict(
                    [pattern_matcher_pass.pass_name]
                )
                # we support run same pattern multiple times, the default is to run only once
                counter_raw = fusion_options[pass_name].get("counter", 1)
                if not isinstance(counter_raw, int):
                    raise AssertionError(f"expected int, got {type(counter_raw)}")
                for _ in range(counter_raw):
                    pattern_matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
                if not is_same_dict(counters["inductor"], inductor_before_change):
                    trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": f"{pattern_matcher_pass.pass_name}_pre_grad",
                            "encoding": "string",
                        },
                        payload_fn=lambda: gm.print_readable(
                            print_output=False, include_stride=True, include_device=True
                        ),
                    )
            # TODO: move efficient_conv_bn_eval_pass to the fusions dict too.
            GraphTransformObserver(gm, "efficient_conv_bn_eval_pass").apply_graph_pass(
                efficient_conv_bn_eval_pass.apply
            )
            GraphTransformObserver(gm, "apply_gumbel_max_trick_pass").apply_graph_pass(
                apply_gumbel_max_trick_pass.apply
            )

    for pre_grad_custom_pass in get_custom_graph_passes(config.pre_grad_custom_pass):
        GraphTransformObserver(gm, "pre_grad_custom_pass").apply_graph_pass(
            pre_grad_custom_pass
        )

    stable_topological_sort(gm.graph)

    from .quantization import quant_lift_up

    quant_lift_up(gm)

    gm.graph.lint()
    gm.recompile()

    if (
        config.pattern_matcher
        and hasattr(config, "fx_passes_numeric_check")
        and config.fx_passes_numeric_check.get("pre_grad", False)
        and example_inputs is not None
    ):
        from .numeric_utils import numeric_check_if_enabled

        gm_after_fx_passes = gm.__copy__()
        numeric_check_if_enabled(
            gm_before_fx_passes,  # type: ignore[possibly-undefined]
            gm_after_fx_passes,
            example_inputs,
            config.fx_passes_numeric_check.get("num_iterations", 1),
            config.fx_passes_numeric_check.get("precision", 1e-4),
        )

    return gm


def fuse_fx(gm: torch.fx.GraphModule, example_inputs) -> torch.fx.GraphModule:
    is_cpu = is_cpu_device(example_inputs)
    # pyre-fixme[16]: Module `torch._dynamo.utils` has no attribute `detect_fake_mode`
    fake_mode = detect_fake_mode(example_inputs)

    gm = sink_cat_after_pointwise(gm)
    if config.permute_fusion and not is_cpu:
        # For linear permute fusion, we need to check input info to identify
        # and perform proper permutation/transpose
        ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
        with GraphTransformObserver(gm, "linear_permute_fusion"):
            gm = linear_permute_fusion(gm)
        with GraphTransformObserver(gm, "permute_linear_fusion"):
            gm = permute_linear_fusion(gm)
        with GraphTransformObserver(gm, "permute_matmul_fusion"):
            gm = permute_matmul_fusion(gm)

    # make sure the autograd is disabled.
    if torch.is_grad_enabled() or not is_cpu:
        return gm
    if config.freezing:
        with GraphTransformObserver(gm, "remove_identity"):
            gm = remove_identity(gm)
        with GraphTransformObserver(gm, "fuse_conv_bn"):
            gm = fuse_conv_bn(gm)
    return gm


def fetch_attr(target: str, mod):
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def remove_identity(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Removes all identity layers from the module.
    """
    graph = gm.graph
    work_done = False
    for module_name, module in gm.named_modules():
        if type(module) is nn.Identity:
            for node in list(graph.find_nodes(op="call_module", target=module_name)):
                if len(node.args) != 1:
                    raise AssertionError(
                        f"expected 1 arg for identity node, got {len(node.args)}"
                    )
                input_node = node.args[0]
                node.replace_all_uses_with(input_node)
                graph.erase_node(node)
                work_done = True
    if work_done:
        graph.lint()
        gm.recompile()
    return gm


def fuse_conv_bn(gm: torch.fx.GraphModule, inplace=False) -> torch.fx.GraphModule:
    """
    Fuses Convolution/BN layers for inference purposes.
    """
    modules_patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    module_function_patterns = [
        (torch.nn.Conv1d, F.batch_norm),
        (torch.nn.Conv2d, F.batch_norm),
        (torch.nn.Conv3d, F.batch_norm),
    ]
    modules = dict(gm.named_modules())

    class ConvBNFusion:
        def __init__(
            self,
            bn_node,
            conv_module,
            bn_module=None,  # For BN Module
            bn_running_mean=None,  # For Functional BN
            bn_running_var=None,
            bn_eps=None,
            bn_weight=None,
            bn_bias=None,
        ) -> None:
            self.bn_nodes = [
                bn_node,
            ]
            self.conv_module = conv_module
            self.bn_module = bn_module
            self.bn_running_mean = bn_running_mean
            self.bn_running_var = bn_running_var
            self.bn_eps = bn_eps
            self.bn_weight = bn_weight
            self.bn_bias = bn_bias
            self.fusion_enabled = True

        def add_bn_node(self, bn_node):
            self.bn_nodes.append(bn_node)

        def disable_fusion(self):
            self.fusion_enabled = False

        def is_fusion_enabled(self):
            return self.fusion_enabled

    conv_bn_to_fuse: dict[int, ConvBNFusion] = {}
    for pattern in modules_patterns:
        conv_bn_to_fuse.clear()
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                eval_mode = all(not n.training for n in [conv, bn])
                if not eval_mode:
                    continue
                if not bn.track_running_stats:
                    continue

                # Do hash based on the module name of conv
                hash_id = hash(node.args[0].target)
                if hash_id not in conv_bn_to_fuse:
                    conv_bn_to_fuse[hash_id] = ConvBNFusion(node, conv, bn)
                else:
                    if bn == conv_bn_to_fuse[hash_id].bn_module:
                        # Do fusion if same bn module
                        conv_bn_to_fuse[hash_id].add_bn_node(node)
                    else:
                        # Disable the conv bn folding if conv shared by different bn
                        conv_bn_to_fuse[hash_id].disable_fusion()

        for conv_bn_fusion in conv_bn_to_fuse.values():
            if conv_bn_fusion.is_fusion_enabled():
                bn_nodes = conv_bn_fusion.bn_nodes
                conv = conv_bn_fusion.conv_module
                bn = conv_bn_fusion.bn_module

                # pyrefly: ignore [bad-argument-type]
                fused_conv = fuse_conv_bn_eval(conv, bn)
                for bn_node in bn_nodes:
                    replace_node_module(bn_node.args[0], modules, fused_conv)
                    bn_node.replace_all_uses_with(bn_node.args[0])
                    gm.graph.erase_node(bn_node)

    gm.graph.lint()
    for pattern in module_function_patterns:
        conv_bn_to_fuse.clear()
        for node in gm.graph.nodes:
            if matches_module_function_pattern(pattern, node, modules):
                # TODO: support kwargs.
                if len(node.args) != 8:
                    continue
                conv = modules[node.args[0].target]
                bn_training = node.args[5]
                bn_eps = node.args[7]
                if conv.training or bn_training:
                    continue
                if type(bn_eps) is not float:
                    continue

                def _used_by_same_conv_module(users):
                    conv_module_name = users[0].args[0].target
                    return all(
                        conv_module_name == user.args[0].target for user in users
                    )

                bn_args_is_constant = all(
                    n.op == "get_attr"
                    and (len(n.users) == 1 or _used_by_same_conv_module(list(n.users)))
                    for n in node.args[1:5]
                )
                if not bn_args_is_constant:
                    continue
                bn_running_mean = fetch_attr(node.args[1].target, gm)
                bn_running_var = fetch_attr(node.args[2].target, gm)
                bn_weight = fetch_attr(node.args[3].target, gm)
                bn_bias = fetch_attr(node.args[4].target, gm)
                if bn_running_mean is None or bn_running_var is None:
                    continue

                # Do hash based on the module name of conv
                hash_id = hash(node.args[0].target)
                if hash_id not in conv_bn_to_fuse:
                    conv_bn_to_fuse[hash_id] = ConvBNFusion(
                        node,
                        conv,
                        bn_running_mean=bn_running_mean,
                        bn_running_var=bn_running_var,
                        bn_eps=bn_eps,
                        bn_weight=bn_weight,
                        bn_bias=bn_bias,
                    )
                else:
                    if (
                        hash(bn_running_mean)
                        == hash(conv_bn_to_fuse[hash_id].bn_running_mean)
                        and hash(bn_running_var)
                        == hash(conv_bn_to_fuse[hash_id].bn_running_var)
                        and torch.allclose(
                            torch.tensor(bn_eps),
                            torch.tensor(conv_bn_to_fuse[hash_id].bn_eps),
                        )
                        and hash(bn_weight) == hash(conv_bn_to_fuse[hash_id].bn_weight)
                        and hash(bn_bias) == hash(conv_bn_to_fuse[hash_id].bn_bias)
                    ):
                        # Do fusion if same functional bn
                        conv_bn_to_fuse[hash_id].add_bn_node(node)
                    else:
                        # Disable the conv bn folding if conv shared by different bn
                        conv_bn_to_fuse[hash_id].disable_fusion()

        for conv_bn_fusion in conv_bn_to_fuse.values():
            if conv_bn_fusion.is_fusion_enabled():
                bn_nodes = conv_bn_fusion.bn_nodes
                conv = conv_bn_fusion.conv_module
                bn_running_mean = conv_bn_fusion.bn_running_mean
                bn_running_var = conv_bn_fusion.bn_running_var
                bn_eps = conv_bn_fusion.bn_eps
                bn_weight = conv_bn_fusion.bn_weight
                bn_bias = conv_bn_fusion.bn_bias

                fused_conv = copy.deepcopy(conv)
                fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
                    fused_conv.weight,
                    fused_conv.bias,
                    # pyrefly: ignore [bad-argument-type]
                    bn_running_mean,
                    # pyrefly: ignore [bad-argument-type]
                    bn_running_var,
                    # pyrefly: ignore [bad-argument-type]
                    bn_eps,
                    bn_weight,
                    bn_bias,
                )
                for bn_node in bn_nodes:
                    replace_node_module(bn_node.args[0], modules, fused_conv)
                    bn_node.replace_all_uses_with(bn_node.args[0])
                    gm.graph.erase_node(bn_node)
    gm.graph.lint()
    gm.recompile()

    return gm


class NormalizedLinearNode:
    def __init__(self, node: torch.fx.Node) -> None:
        if node.op != "call_function":
            raise AssertionError(f"expected call_function node, got {node.op}")
        if node.target is not torch.nn.functional.linear:
            raise AssertionError(
                f"expected torch.nn.functional.linear target, got {node.target}"
            )
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]  # type: ignore[return-value]
        else:
            return self.node.kwargs["input"]  # type: ignore[return-value]

    def get_weight(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]  # type: ignore[return-value]
        else:
            return self.node.kwargs["weight"]  # type: ignore[return-value]

    def get_bias(self) -> torch.fx.Node:
        if len(self.node.args) > 2:
            return self.node.args[2]  # type: ignore[return-value]
        else:
            return self.node.kwargs.get("bias", None)  # type: ignore[return-value]


class NormalizedMatmulNode:
    def __init__(self, node: torch.fx.Node) -> None:
        if node.op != "call_function":
            raise AssertionError(f"expected call_function node, got {node.op}")
        if node.target not in [torch.bmm, torch.matmul]:
            raise AssertionError(
                f"expected torch.bmm or torch.matmul target, got {node.target}"
            )
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]  # type: ignore[return-value]
        else:
            return self.node.kwargs["input"]  # type: ignore[return-value]

    def get_other(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]  # type: ignore[return-value]
        else:
            return self.node.kwargs["other"]  # type: ignore[return-value]


def check_permute(node: torch.fx.Node) -> bool:
    ranks = len(node.meta["tensor_meta"].shape)
    if len(node.args) > 3:
        permutation = [node.args[i] % ranks for i in range(1, ranks + 1)]  # type: ignore[operator]
    elif (
        "permutation" in node.kwargs
        and node.kwargs["permutation"] is not None
        and len(node.kwargs["permutation"]) > 2  # type: ignore[arg-type]
    ):
        permutation = [i % ranks for i in node.kwargs["permutation"]]  # type: ignore[operator, union-attr]
    else:
        return False
    allowed_permutation = list(range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


def sink_cat_after_pointwise(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    def one_user(node):
        users = list(node.users)
        return users[0] if len(users) == 1 else None

    def is_view(node):
        return node.op == "call_method" and node.target == "view"

    def is_pointwise_unary(node):
        ops = "call_function", "call_method"
        pointwise = torch.relu, torch.tanh, "relu", "tanh"
        return node.op in ops and node.target in pointwise

    g = module.graph
    for node in g.nodes:
        if node.op != "call_function" or node.target != torch.cat:
            continue

        cat_or_view = node
        while True:
            user = one_user(cat_or_view)
            if not user or not is_view(user):
                break
            cat_or_view = user

        if user and is_pointwise_unary(user):
            with g.inserting_before(node):
                arg_not_provided = object()

                def cat_args(
                    tensors,
                    dim=arg_not_provided,
                    axis=arg_not_provided,
                    out=None,
                ):
                    if dim is arg_not_provided:
                        dim = 0
                    if axis is not arg_not_provided:
                        dim = axis
                    return tensors, dim, out

                tensors, dim, out = cat_args(*node.args, **node.kwargs)
                if out is not None or dim is None:
                    continue
                new_kwargs = {
                    name: val for name, val in user.kwargs.items() if name != "input"
                }
                new_tensors = [
                    g.create_node(user.op, user.target, args=(arg,), kwargs=new_kwargs)
                    for arg in tensors
                ]
                new_cat = g.create_node(
                    "call_function", torch.cat, args=(new_tensors, dim)
                )
                user.replace_all_uses_with(cat_or_view)
                node.replace_all_uses_with(new_cat)
                g.erase_node(user)
                g.erase_node(node)
    g.lint()
    module.recompile()
    return module


def linear_permute_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.find_nodes(op="call_method", target="permute"):
        if check_permute(node):
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            if (
                input_node.op == "call_function"
                and input_node.target is torch.nn.functional.linear
            ):
                normalized = NormalizedLinearNode(input_node)
                input = normalized.get_input()
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        linear_transpose, args=(input, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


# Y1 = X * W^T + bias
# Y2 = Y1.permute(0, 2, 1)
# ---->
# Y2 = (W * X^T + bias.unsqueeze(-1))^T
def linear_transpose(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(weight, input.transpose(-1, -2))
    return torch.matmul(weight, input.transpose(-1, -2)) + bias.unsqueeze(-1)


def permute_linear_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.find_nodes(
        op="call_function", target=torch.nn.functional.linear
    ):
        if len(node.args) > 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs["input"]
        if (
            input_node.op == "call_method"
            and input_node.target == "permute"
            and check_permute(input_node)
        ):
            normalized = NormalizedLinearNode(node)
            if len(input_node.args) > 0:
                input = input_node.args[0]
            else:
                input = input_node.kwargs["input"]
            weight = normalized.get_weight()
            bias = normalized.get_bias()
            with module.graph.inserting_before(node):
                fused_node = module.graph.call_function(
                    transpose_linear, args=(input, weight, bias)
                )
                node.replace_all_uses_with(fused_node)
                module.graph.erase_node(node)
                if len(input_node.users) == 0:
                    module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


def permute_matmul_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in itertools.chain(
        module.graph.find_nodes(op="call_function", target=torch.bmm),
        module.graph.find_nodes(op="call_function", target=torch.matmul),
    ):
        normalized = NormalizedMatmulNode(node)
        input_A_node = normalized.get_input()
        input_B_node = normalized.get_other()
        input_A = input_A_node
        input_B = input_B_node
        Atrans = Btrans = False
        if (
            input_A_node.op == "call_method"
            and input_A_node.target == "permute"
            and check_permute(input_A_node)
        ):
            Atrans = True
            if len(input_A_node.args) > 0:
                input_A = input_A_node.args[0]  # type: ignore[assignment]
            else:
                input_A = input_A_node.kwargs["input"]  # type: ignore[assignment]

        if (
            input_B_node.op == "call_method"
            and input_B_node.target == "permute"
            and check_permute(input_B_node)
        ):
            Btrans = True
            if len(input_B_node.args) > 0:
                input_B = input_B_node.args[0]  # type: ignore[assignment]
            else:
                input_B = input_B_node.kwargs["input"]  # type: ignore[assignment]

        if Atrans or Btrans:
            with module.graph.inserting_before(node):
                fused_node = module.graph.call_function(
                    transpose_matmul,
                    args=(input_A, input_B, Atrans, Btrans),
                )
            node.replace_all_uses_with(fused_node)
            module.graph.erase_node(node)
            if Atrans and len(input_A_node.users) == 0:
                module.graph.erase_node(input_A_node)
            if Btrans and len(input_B_node.users) == 0:
                module.graph.erase_node(input_B_node)

    module.graph.lint()
    module.recompile()
    return module


# X1 = X.permute(0, 2, 1)
# Y1 = X1 * W1^T + bias1
# ---->
# Y2 = X1.transpose(-1, -2) * W1^T + bias1
def transpose_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(input.transpose(-1, -2), weight.t())
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def transpose_matmul(
    A: torch.Tensor, B: torch.Tensor, Atrans: bool, Btrans: bool
) -> torch.Tensor:
    if Atrans:
        A = A.transpose(-1, -2)
    if Btrans:
        B = B.transpose(-1, -2)
    return torch.matmul(A, B)
<|endoftext|># mypy: allow-untyped-defs
import functools
import operator
from functools import reduce
from typing import Any, TYPE_CHECKING

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols, optimization_hint
from torch.utils._ordered_set import OrderedSet

from .. import ir, mkldnn_ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
    Arg,
    CallFunction,
    filter_nodes,
    get_arg_value,
    KeywordArg,
    MULTIPLE,
)
from ..utils import (
    is_mkldnn_bf16_supported,
    is_mkldnn_fp16_supported,
    SUPPORTED_MKLDNN_DEVICES,
)
from ..virtualized import ops, V
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
    _register_int8_woq_concat_linear_pattern,
    _register_quantization_lowerings,
    _register_woq_lowerings,
)


if TYPE_CHECKING:
    from collections.abc import Callable


if torch._C._has_mkldnn:
    aten = torch.ops.aten
    mkldnn = torch.ops.mkldnn
    prims = torch.ops.prims

    _conv_args = [Arg() for _ in range(10)]
    _linear_args = [Arg() for _ in range(6)]
    _conv_transpose_args = [Arg() for _ in range(11)]

    class MkldnnDeviceOpBase:
        def get_linear_transpose_weight(self, weight_node):
            raise NotImplementedError

        def pack_conv_weight(
            self,
            graph,
            is_transposed,
            weight,
            constant_args,
            input_size,
        ):
            raise NotImplementedError

        def pack_linear_weight(
            self, graph, is_lp_weight, transpose_weight_node, batch_size
        ):
            raise NotImplementedError

        def pack_linear(
            self, graph, is_lp_weight, batch_size, input, packed_weight_node, bias
        ):
            raise NotImplementedError

    class CpuMkldnnDeviceOp(MkldnnDeviceOpBase):
        def get_linear_transpose_weight(self, weight_node):
            packed_weight_node = weight_node
            if packed_weight_node.target != mkldnn._reorder_linear_weight:
                raise AssertionError(
                    "expected packed_weight_node.target to be "
                    "mkldnn._reorder_linear_weight, got "
                    f"{packed_weight_node.target}"
                )
            transpose_weight_node = packed_weight_node.args[0]
            if transpose_weight_node.target is not aten.permute.default:
                raise AssertionError(
                    "expected transpose_weight_node.target to be "
                    f"aten.permute.default, got {transpose_weight_node.target}"
                )
            return transpose_weight_node

        def pack_conv_weight(
            self,
            graph,
            is_transposed,
            weight,
            constant_args,
            input_size,
        ):
            packed_weight_op = mkldnn._reorder_convolution_weight
            if is_transposed:
                packed_weight_op = mkldnn._reorder_convolution_transpose_weight

            # mkldnn_reorder_conv_weight(self, padding, stride, dilation, groups, input_size)
            packed_weight_inputs = (weight,) + tuple(constant_args) + (input_size,)
            return graph.create_node(
                "call_function", packed_weight_op, args=packed_weight_inputs
            )

        def pack_linear_weight(
            self, graph, is_lp_weight, transpose_weight_node, batch_size
        ):
            # For bfloat16 dynamic shape path, using input size hint to pack weight for a better performance.
            packed_weight_inputs = (
                transpose_weight_node,
                optimization_hint(batch_size)
                if has_free_symbols(batch_size)
                else batch_size,
            )

            # MKL packed matrix can't be copied to a different address because the internal implementation
            # depends on the alignment of internally-stored metadata.
            # In aot mode, we need to firstly save the packed weight, when loading it,
            # it will be in a different address which doesn't work.
            # Disable MKL prepack linear in AOT mode.
            # Disable MKL prepack linear when batch_size has free symbols.
            packed_weight_op = (
                mkldnn._reorder_linear_weight
                if (
                    is_lp_weight
                    or mkldnn._is_mkldnn_acl_supported()
                    or V.aot_compilation
                    or has_free_symbols(batch_size)
                )
                else torch.ops.mkl._mkl_reorder_linear_weight
            )
            return graph.create_node(
                "call_function", packed_weight_op, args=packed_weight_inputs
            )

        def pack_linear(
            self, graph, is_lp_weight, batch_size, input, packed_weight_node, bias
        ):
            packed_linear_inputs: tuple[Any, ...] = (input, packed_weight_node)
            transpose_weight_node = packed_weight_node.args[0]
            if (
                is_lp_weight
                or mkldnn._is_mkldnn_acl_supported()
                or V.aot_compilation
                or has_free_symbols(batch_size)
            ):
                packed_linear_inputs += (bias, "none", [], "")
                packed_linear_op: Callable[..., Any] = mkldnn._linear_pointwise.default
            else:
                packed_linear_inputs += (transpose_weight_node, bias, batch_size)
                packed_linear_op = torch.ops.mkl._mkl_linear

            return graph.create_node(
                "call_function", packed_linear_op, packed_linear_inputs
            )

    class XpuMkldnnDeviceOp(MkldnnDeviceOpBase):
        def pack_conv_weight(
            self,
            graph,
            is_transposed,
            weight,
            constant_args,
            input_size,
        ):
            if is_transposed:
                raise AssertionError(
                    "'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device."
                )
            return weight

    def _get_mkldnn_device_op(device_type: str) -> MkldnnDeviceOpBase:
        """
        Returns the MKLDNN device operation class based on the current device type.
        """
        if device_type == "cpu":
            return CpuMkldnnDeviceOp()
        elif device_type == "xpu":
            return XpuMkldnnDeviceOp()
        else:
            raise RuntimeError(f"MKLDNN is not supported on {device_type} device.")

    def _is_valid_grouped_gemm_fusion(computation_nodes):
        """
        Here we check:
        1. More than 1 GEMM nodes has been found.
        2. All the GEMM nodes share the same activation.
        3. All the GEMM nodes have same weight size but different wgt node.
        """
        computation_op = mkldnn._linear_pointwise.default
        act = computation_nodes[0].args[0]
        wgt = computation_nodes[0].args[1]
        wgt_size = wgt.meta.get("val").size()  # type: ignore[union-attr]
        return len(computation_nodes) >= 2 and all(
            (
                node.target == computation_op
                and node.args[0] == act
                and (node.args[1].meta.get("val").size() == wgt_size)
                and (node.args[1] != wgt or gemm_idx == 0)
            )
            for gemm_idx, node in enumerate(computation_nodes)
        )

    def grouped_gemm_pass(graph: torch.fx.Graph):
        """
        Group GEMM has multi output nodes which is complicated to define a Pattern.
        Use below way to connect the pattern to the lowering.
        TODO: Use MultiOutputPattern, current limitation is the pattern requires
        fixed number of output nodes. Extend to support Group GEMM for pattern matcher.
        """
        computation_op = mkldnn._linear_pointwise.default
        from ..mkldnn_lowerings import grouped_gemm_lowering

        for node in graph.find_nodes(op="call_function", target=computation_op):
            if (
                not node._erased
                and isinstance(node.meta.get("val"), torch.Tensor)
                and node.meta["val"].device.type == "cpu"
            ):
                act = node.args[0]
                users = list(act.users)
                if _is_valid_grouped_gemm_fusion(users):
                    with graph.inserting_before(node):
                        grouped_gemm_node = graph.create_node(
                            "call_function",
                            grouped_gemm_lowering,
                            (
                                act,
                                [user.args[1] for user in users],
                                [user.args[2] for user in users],
                            ),
                        )
                        grouped_gemm_node.meta["val"] = [
                            user.meta["val"] for user in users
                        ]
                        with graph.inserting_after(grouped_gemm_node):
                            for gemm_idx, user in enumerate(users):
                                if user.target != computation_op:
                                    raise AssertionError(
                                        "expected user.target to be "
                                        f"{computation_op}, got {user.target}"
                                    )
                                get_item = graph.create_node(
                                    "call_function",
                                    operator.getitem,
                                    (
                                        grouped_gemm_node,
                                        gemm_idx,
                                    ),
                                )
                                user.replace_all_uses_with(get_item)
                                graph.erase_node(user)
        return

    def _conv_call(users=1):
        return CallFunction(
            mkldnn._convolution_pointwise.default, *_conv_args, _users=users
        )

    def _linear_call(users=1):
        return CallFunction(
            mkldnn._linear_pointwise.default, *_linear_args, _users=users
        )

    def _conv_transpose_call(users=1):
        return CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=users,
        )

    def _to_float(input_call, users=1):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_float"),
            _users=users,
        )

    def _to_bf16(input_call):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_bf16"),
            _users=1,
        )

    def _to_fp16(input_call):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_fp16"),
            _users=1,
        )

    def _unary_fusion_pattern(unary_fusion, call_fn, users, lowp_dtype):
        # only insert to_dtype if lowp_dtype is True
        computation_call = (
            _to_float(call_fn(), users=users) if lowp_dtype else call_fn(users=users)
        )
        out = unary_fusion(computation_call)
        if lowp_dtype == torch.bfloat16:
            return _to_bf16(out)
        elif lowp_dtype == torch.float16:
            return _to_fp16(out)
        else:
            return out

    def _gelu_fusion_1(computation_call):
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.erf,
                    CallFunction(aten.mul, computation_call, 0.7071067811865476),
                ),
                1,
            ),
        )

    def _gelu_fusion_2(computation_call):
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.tanh,
                    CallFunction(
                        aten.mul,
                        CallFunction(
                            aten.add,
                            computation_call,
                            CallFunction(
                                aten.mul,
                                CallFunction(
                                    aten.mul,
                                    CallFunction(
                                        aten.mul, computation_call, computation_call
                                    ),
                                    computation_call,
                                ),
                                0.044715,
                            ),
                        ),
                        0.7978845608028654,
                    ),
                ),
                1,
            ),
        )

    def _hardswish_fusion(computation_call):
        return CallFunction(
            aten.div,
            CallFunction(
                aten.mul,
                computation_call,
                CallFunction(
                    aten.clamp_max,
                    CallFunction(
                        aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                    ),
                    6,
                ),
            ),
            6,
        )

    def _silu_fusion(computation_call):
        # Match: x / (exp(-x) + 1) pattern used by inductor's silu decomposition
        return CallFunction(
            aten.div,
            computation_call,
            CallFunction(
                aten.add,
                CallFunction(aten.exp, CallFunction(aten.neg, computation_call)),
                1,
            ),
        )

    def _hardsigmoid_fusion(computation_call):
        return CallFunction(
            aten.div,
            CallFunction(
                aten.clamp_max,
                CallFunction(
                    aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                ),
                6,
            ),
            6,
        )

    def _leaky_relu_fusion(computation_call):
        return CallFunction(
            aten.where,
            CallFunction(aten.gt, computation_call, 0),
            computation_call,
            CallFunction(aten.mul, computation_call, KeywordArg("negative_slope")),
        )

    def _hardtanh_fusion(computation_call):
        return CallFunction(
            aten.clamp_max,
            CallFunction(aten.clamp_min, computation_call, KeywordArg("min_value")),
            KeywordArg("max_value"),
        )

    def _combined_fusion(computation_call, elementwise_op):
        return CallFunction(elementwise_op, computation_call)

    # binary_op(other, computation_op)
    def _binary_fusion_v1(computation_call, binary_fn):
        return CallFunction(binary_fn, KeywordArg("other"), computation_call)

    # binary_op(computation_op, other)
    def _binary_fusion_v2(computation_call, binary_fn):
        return CallFunction(binary_fn, computation_call, KeywordArg("other"))

    def _is_single_computation_op(computation_op, lowp_dtype=None):
        def fn(match):
            computation_nodes = filter_nodes(match.nodes, computation_op)

            if lowp_dtype:
                output_node_meta = match.output_node().meta.get("val")
                if output_node_meta.dtype != lowp_dtype:
                    return False

            if len(computation_nodes) < 1:
                return False
            if any(n.args[-3] != "none" for n in computation_nodes):
                return False
            return True

        return fn

    def _is_valid_computation_unary_fusion(computation_op, lowp_dtype=None):
        def fn(match):
            matched = _is_single_computation_op(computation_op, lowp_dtype)(match)
            computation_node = filter_nodes(match.nodes, computation_op)[0]
            if lowp_dtype:
                conversion_dtype_nodes = filter_nodes(
                    match.nodes, prims.convert_element_type.default
                )
                if len(conversion_dtype_nodes) != 2:
                    return False
                # fusion pattern is always in the form of computation_op + to_float32 + unary_op + to_bfloat16
                if computation_node == conversion_dtype_nodes[0].args[0]:
                    to_float = conversion_dtype_nodes[0].args[1]
                    to_lp = conversion_dtype_nodes[1].args[1]
                else:
                    to_float = conversion_dtype_nodes[1].args[1]
                    to_lp = conversion_dtype_nodes[0].args[1]
                matched = matched and to_float == torch.float and to_lp == lowp_dtype
            return matched

        return fn

    def _register_unary_fusion_lowering(
        pattern, unary_attr, computation_op, lowp_dtype=None
    ):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_unary_fusion(computation_op, lowp_dtype),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            computation_args = list(args)[:-3] + [
                unary_attr.op_name,
                unary_attr.scalars_attr,
                unary_attr.algorithm_attr,
            ]
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            return L[computation_op](*computation_args)

        return fn

    def _register_leaky_relu_fusion_lowering(pattern, computation_op, lowp_dtype=None):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_single_computation_op(computation_op, lowp_dtype),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            negative_slope = kwargs.get("negative_slope")
            if isinstance(negative_slope, ir.TensorBox):
                matched = False
            else:  # inp is a Number
                matched = True
            if lowp_dtype:
                dtype1 = kwargs.get("to_float")
                dtype2 = (
                    kwargs.get("to_bf16")
                    if lowp_dtype == torch.bfloat16
                    else kwargs.get("to_fp16")
                )
                matched = matched and dtype1 == torch.float and dtype2 == lowp_dtype
            computation_args = list(args)
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            if matched:
                computation_args = computation_args[:-3] + [
                    "leaky_relu",
                    [negative_slope],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                # computation_args += ["none", [], ""]
                out = L[computation_op](*computation_args)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                out = L[aten.where](
                    L[aten.gt](out, 0),
                    out,
                    L[aten.mul](out, negative_slope),
                )
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=dtype2)  # type: ignore[possibly-undefined]
                return out

        return fn

    def _register_hardtanh_fusion_lowering(pattern, computation_op, lowp_dtype=None):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_single_computation_op(computation_op, lowp_dtype),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")
            if isinstance(min_value, ir.TensorBox) or isinstance(
                max_value, ir.TensorBox
            ):
                matched = False
            else:  # inp is a Number
                if max_value is None:
                    raise AssertionError("expected max_value to not be None")
                matched = min_value <= max_value
            if lowp_dtype:
                dtype1 = kwargs.get("to_float")
                dtype2 = (
                    kwargs.get("to_bf16")
                    if lowp_dtype == torch.bfloat16
                    else kwargs.get("to_fp16")
                )
                matched = matched and dtype1 == torch.float and dtype2 == lowp_dtype
            computation_args = list(args)
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            if matched:
                computation_args = computation_args[:-3] + [
                    "hardtanh",
                    [min_value, max_value],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                out = L[computation_op](*computation_args)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                out = L[aten.clamp_max](L[aten.clamp_min](out, min_value), max_value)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=dtype2)  # type: ignore[possibly-undefined]
                return out

        return fn

    _binary_attr = {
        aten.add: "add",
        ops.add: "add",
        aten.sub: "sub",
        ops.sub: "sub",
    }

    def _is_valid_binary(match, computation_op, binary_op):
        binary_nodes = filter_nodes(match.nodes, binary_op)
        if len(binary_nodes) < 1:
            return False

        def get_meta_value(argument: torch.fx.node.Argument):
            # Only torch.fx.Node is expected to have meta.
            if isinstance(argument, torch.fx.Node):
                return argument.meta.get("val", None)
            return None

        if any(
            not isinstance(get_meta_value(n.args[0]), torch.Tensor)
            or not isinstance(get_meta_value(n.args[1]), torch.Tensor)
            for n in binary_nodes
        ):
            return False
        # check alpha is one.
        if any(
            get_arg_value(n, 2, kwarg_name="alpha") != 1.0
            and get_arg_value(n, 2, kwarg_name="alpha") is not None
            for n in binary_nodes
        ):
            return False

        def _check_input_sizes(n, computation_op):
            # Check if the tensor shape of the 'other' node is the same as or
            # can be broadcasted to the tensor shape of the computation node.
            computation_node = (
                n.args[0] if n.args[1] is match.kwargs["other"] else n.args[1]
            )
            if computation_node.target != computation_op:
                raise AssertionError(
                    "expected computation_node.target to be "
                    f"{computation_op}, got {computation_node.target}"
                )
            computation_node_size = get_meta_value(computation_node).size()
            if computation_op is mkldnn._linear_pointwise.default:
                broadcast_sizes = []
                if len(computation_node_size) >= 2:
                    broadcast_sizes = [
                        torch.Size(
                            [1 for _ in range(len(computation_node_size) - 1)]
                            + [computation_node_size[-1]]
                        ),
                    ]
            else:
                if not len(computation_node_size) > 2:
                    raise AssertionError(
                        "expected len(computation_node_size) > 2, got "
                        f"{len(computation_node_size)}"
                    )
                broadcast_sizes = [
                    torch.Size(
                        [computation_node_size[0], computation_node_size[1]]
                        + [1 for _ in range(len(computation_node_size) - 2)]
                    ),
                    torch.Size(
                        [1, computation_node_size[1]]
                        + [1 for _ in range(len(computation_node_size) - 2)]
                    ),
                    torch.Size([1 for _ in range(len(computation_node_size))]),
                ]
            return (
                get_meta_value(match.kwargs["other"]).size()
                in [
                    computation_node_size,
                ]
                + broadcast_sizes
            )

        if any(
            not _check_input_sizes(n, computation_op)
            or get_meta_value(n.args[0]).device != get_meta_value(n.args[1]).device
            or get_meta_value(n.args[0]).dtype != get_meta_value(n.args[1]).dtype
            for n in binary_nodes
        ):
            return False
        # check args[0] and args[1] is not same
        if any(n.args[0] == n.args[1] for n in binary_nodes):
            return False
        return True

    def _is_valid_computation_binary(computation_op, binary_op, other_index=None):
        def fn(match):
            if not _is_single_computation_op(computation_op)(match):
                return False
            if not _is_valid_binary(match, computation_op, binary_op):
                return False
            return True

        return fn

    def _get_remaining_users(extra_input_node, compute_node):
        # Think about this pattern:
        #      ReLU
        #     /   \
        #  Conv1
        #   /      \
        # Conv2
        #   \      /
        #      Add
        # Although, the extra input node (ReLU) has more than 1 users: Conv1 and Add.
        # The Conv1 is the ancestor node of the current compute node (Conv2).
        # This indicates that the buffer of ReLU has completed all its usage,
        # So we can safely make changes to it now by doing Conv2->Add inplace fusion.
        # Take above case as example:
        # * extra_input_node: ReLU
        # * compute_node: Conv2
        # _get_remaining_users will return the users of extra_input_node which are not
        # ancestor node of compute_node.
        def _is_ancestor_node(_current_node, _ancestor_node):
            # Check whether _ancestor_node is the ancestor node of _current_node
            _node_list = [_current_node]
            _visited_nodes = OrderedSet[torch.fx.Node]()
            while len(_node_list) != 0:
                _current_node = _node_list.pop(0)
                if _current_node not in _visited_nodes:
                    _visited_nodes.add(_current_node)
                    if _current_node == _ancestor_node:
                        return True
                    elif isinstance(
                        _current_node, torch.fx.Node
                    ) and _current_node.op not in ["placeholder", "output", "get_attr"]:
                        for input in _current_node.all_input_nodes:
                            _node_list.append(input)  # noqa: PERF402
            return False

        return [
            user
            for user in list(extra_input_node.users)
            if not _is_ancestor_node(compute_node, user)
        ]

    def _is_valid_computation_binary_inplace(computation_op, binary_op, other_index):
        def fn(match):
            if not _is_valid_computation_binary(computation_op, binary_op)(match):
                return False
            binary_nodes = filter_nodes(match.nodes, binary_op)

            def _get_compute_node(_binary_node, _other_index):
                if len(_binary_node.all_input_nodes) != 2:
                    raise AssertionError("Binary node should have 2 input nodes.")
                _compute_index = 1 if (_other_index == 0) else 0
                return _binary_node.args[_compute_index]

            def _other_input_not_inplaceable(_binary_node, _other_index):
                _compute_node = _get_compute_node(_binary_node, _other_index)
                return (
                    len(
                        _get_remaining_users(
                            _binary_node.args[_other_index], _compute_node
                        )
                    )
                    > 1
                    or _binary_node.args[_other_index] == _compute_node.args[0]
                )

            if any(_other_input_not_inplaceable(n, other_index) for n in binary_nodes):
                return False
            if any(
                n.args[other_index].op in ["placeholder", "output"]
                for n in binary_nodes
            ):
                return False
            return True

        return fn

    def _register_binary_unary_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        fusion_op,
        unary_attr=None,
    ):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_binary(computation_op, binary_op),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")
            if not isinstance(other, ir.TensorBox):
                raise AssertionError(
                    f"expected other to be ir.TensorBox, got {type(other)}"
                )
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_nodes"] += (
                len(match.nodes)
            )
            return L[fusion_op](*computation_args)

        return fn

    def _can_be_inplace(_other):
        return not (
            isinstance(_other.data, ir.BaseView)
            or len(_other.get_inputs_that_alias_output()) > 0
        )

    def _qlinear_binary_can_be_inplace(_other):
        if isinstance(_other.data, ir.BaseView):

            def unwrap_buffer(data):
                if isinstance(data, ir.StorageBox):
                    return data.data
                return data

            data = _other.data.unwrap_view()
            if isinstance(unwrap_buffer(data), ir.CppTemplateBuffer):
                # It can be inplaced when _other is the 2D to 3D view of
                # a CppTemplateBuffer because if there is a view of CppTemplateBuffer,
                # CppTemplateBuffer will not be used directly but the view.
                return True
            else:
                # The case of QLinearPointwiseBinaryPT2E(sum) -> QLinearPointwiseBinaryPT2E(sum)
                # is similar to CppTemplateBuffer above.
                # The output of previous QLinearPointwiseBinaryPT2E is
                # the input x2 of current QLinearPointwiseBinaryPT2E.
                # Use V.graph.operations to check if _other is a view of the output
                # of previous QLinearPointwiseBinaryPT2E (the inputs[6]).
                for op in V.graph.operations:
                    if (
                        isinstance(op, mkldnn_ir.QLinearPointwiseBinaryPT2E)
                        and unwrap_buffer(data) == op.inputs[6]  # type: ignore[attr-defined]
                    ):
                        return True
            return False
        elif len(_other.get_inputs_that_alias_output()) > 0:
            return False
        else:
            return True

    def _register_binary_unary_maybe_inplace_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        inplace_fusion_op,
        outplace_fusion_op,
        unary_attr=None,
        other_index=None,
    ):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_binary_inplace(
                computation_op, binary_op, other_index
            ),
            output_metadata_ignores_input_storage=True,
        )
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")
            if not isinstance(other, ir.TensorBox):
                raise AssertionError(
                    f"expected other to be ir.TensorBox, got {type(other)}"
                )
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_nodes"] += (
                len(match.nodes)
            )
            # Make sure the other is not an alias or mutation (fx side doesn't have such info).
            other.realize()
            if not _can_be_inplace(other) or other.data.shape != list(
                match.nodes[0].meta["val"].size()
            ):
                return L[outplace_fusion_op](*computation_args)
            return L[inplace_fusion_op](*computation_args)

        return fn

    computation_ops = [
        mkldnn._convolution_pointwise.default,
        mkldnn._linear_pointwise.default,
        mkldnn._convolution_transpose_pointwise.default,
    ]

    class UnaryAttr:
        def __init__(
            self, op_name: str, scalars_attr=None, algorithm_attr=None
        ) -> None:
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    def _register_unary_fusion():
        computation_call_fns = [_conv_call, _linear_call, _conv_transpose_call]

        def _unary_fusion_patterns(lowp_dtype):
            replacement_unary_fusion_patterns = {
                UnaryAttr("gelu", algorithm_attr="tanh"): [
                    _unary_fusion_pattern(_gelu_fusion_2, call_fn, 4, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("gelu", algorithm_attr="none"): [
                    _unary_fusion_pattern(_gelu_fusion_1, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("hardswish"): [
                    _unary_fusion_pattern(_hardswish_fusion, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("hardsigmoid"): [
                    _unary_fusion_pattern(_hardsigmoid_fusion, call_fn, 1, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("swish"): [
                    _unary_fusion_pattern(_silu_fusion, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
            }
            if not lowp_dtype:
                call_user1 = [call_fn(users=1) for call_fn in computation_call_fns]
                replacement_unary_fusion_patterns.update(
                    {
                        UnaryAttr("relu"): [
                            _combined_fusion(u, aten.relu) for u in call_user1
                        ],
                        UnaryAttr("sigmoid"): [
                            _combined_fusion(u, aten.sigmoid) for u in call_user1
                        ],
                        UnaryAttr("tanh"): [
                            _combined_fusion(u, aten.tanh) for u in call_user1
                        ],
                    }
                )

            return replacement_unary_fusion_patterns

        for lowp_dtype in [torch.bfloat16, torch.float16, None]:
            replace_patterns = _unary_fusion_patterns(lowp_dtype)
            for unary_attr, patterns in replace_patterns.items():
                _register_unary_fusion_lowering(
                    patterns[0], unary_attr, computation_ops[0], lowp_dtype
                )
                _register_unary_fusion_lowering(
                    patterns[1], unary_attr, computation_ops[1], lowp_dtype
                )
                _register_unary_fusion_lowering(
                    patterns[2], unary_attr, computation_ops[2], lowp_dtype
                )
            _leaky_relu_patterns = [
                _unary_fusion_pattern(_leaky_relu_fusion, call_fn, 3, lowp_dtype)
                for call_fn in computation_call_fns
            ]
            for pattern, computation_op in zip(_leaky_relu_patterns, computation_ops):
                _register_leaky_relu_fusion_lowering(
                    pattern, computation_op, lowp_dtype
                )
            hardtanh_patterns = [
                _unary_fusion_pattern(_hardtanh_fusion, call_fn, 1, lowp_dtype)
                for call_fn in computation_call_fns
            ]
            for pattern, computation_op in zip(hardtanh_patterns, computation_ops):
                _register_hardtanh_fusion_lowering(pattern, computation_op, lowp_dtype)

    def _register_inplace_fusion():
        binary_ops = [aten.add, ops.add]
        inplace_fusion_op = mkldnn._convolution_pointwise_.binary
        outplace_fusion_op = mkldnn._convolution_pointwise.binary
        conv_call = _conv_call(users=1)
        conv_op = computation_ops[0]
        for binary_op in binary_ops:
            binary_v1 = _binary_fusion_v1(conv_call, binary_op)
            binary_unary_v1 = _combined_fusion(binary_v1, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
            )
            binary_v2 = _binary_fusion_v2(conv_call, binary_op)
            binary_unary_v2 = _combined_fusion(binary_v2, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
            )

    def _register_binary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [
            mkldnn._convolution_pointwise.binary,
            mkldnn._linear_pointwise.binary,
        ]
        _computation_user_1 = [_conv_call(users=1), _linear_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern = _binary_fusion_v2(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

            for binary_op in [aten.add, ops.add]:
                pattern = _binary_fusion_v1(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

    def _register_binary_unary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [mkldnn._convolution_pointwise.binary]
        _computation_user_1 = [_conv_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern_v1 = _combined_fusion(
                    _binary_fusion_v2(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v1,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )
            for binary_op in [aten.add, ops.add]:
                pattern_v2 = _combined_fusion(
                    _binary_fusion_v1(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v2,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )

    def _recover_linear():
        # convert reshape+linear+reshape to a single linear for applying fusion path.
        # concat_linear (pass_number=0) -> mkldnn_linear_pack (pass_number=1) -> _recover_linear(pass_number=2)
        @register_freezing_graph_pattern(
            CallFunction(
                aten.reshape.default,
                CallFunction(
                    mkldnn._linear_pointwise.default,
                    CallFunction(
                        aten.reshape.default,
                        Arg(),
                        KeywordArg("reshape_1"),
                        _users=MULTIPLE,
                    ),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                ),
                KeywordArg("reshape_2"),
            ),
            pass_number=2,
        )
        def reshape_linear_reshape_pattern(match, *args, **kwargs):
            def get_val(val):
                return val if isinstance(val, int) else val.meta.get("val")

            reshape_1 = kwargs.get("reshape_1")
            reshape_2 = kwargs.get("reshape_2")
            if not isinstance(reshape_1, list):
                raise AssertionError(
                    f"expected reshape_1 to be list, got {type(reshape_1)}"
                )
            if not isinstance(reshape_2, list):
                raise AssertionError(
                    f"expected reshape_2 to be list, got {type(reshape_2)}"
                )
            if len(reshape_1) != 2:
                raise AssertionError(
                    f"expected len(reshape_1) == 2, got {len(reshape_1)}"
                )

            graph = match.graph
            reshape_2_node = match.output_node()
            linear_input_node = reshape_2_node.args[0].args[0].args[0]
            # check linear's input's shape[:-1] == reshape_2[:-1]
            # and check product(reshape_2[:-1]) == reshape_1[0]
            can_remove_reshape = linear_input_node.meta.get("val").shape[
                :-1
            ] == torch.Size([get_val(val) for val in reshape_2[:-1]])
            can_remove_reshape = can_remove_reshape and (
                reduce(
                    operator.mul,
                    [get_val(val) for val in reshape_2[:-1]],
                )
                == get_val(reshape_1[0])
            )

            if can_remove_reshape:
                repl = graph.call_function(mkldnn._linear_pointwise.default, args)
                repl.meta.update(reshape_2_node.meta)
                reshape_2_node.replace_all_uses_with(repl)
                old_linear_node = reshape_2_node.args[0]
                reshape_1_node = old_linear_node.args[0]
                graph.erase_node(reshape_2_node)
                graph.erase_node(old_linear_node)
                if len(reshape_1_node.users) == 0:
                    graph.erase_node(reshape_1_node)
            counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_count"] += 1
            counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"] += len(
                match.nodes
            )

        def is_linear_add_bias(match):
            add_node = match.output_node()
            linear_node = add_node.args[0]
            device_type = add_node.meta.get("val").device.type
            mkldnn_device_op = _get_mkldnn_device_op(device_type)
            transpose_weight_node = mkldnn_device_op.get_linear_transpose_weight(
                linear_node.args[1]
            )
            weight_meta = transpose_weight_node.args[0].meta.get("val")
            bias_node = add_node.args[1]
            if not isinstance(bias_node, torch.fx.Node):
                return False
            bias_meta = bias_node.meta.get("val")
            if weight_meta is None or bias_meta is None:
                return False

            if bias_meta.dtype != weight_meta.dtype:
                return False
            return (
                linear_node.args[2] is None
                and bias_meta.dim() == 1
                and bias_meta.size(0) == weight_meta.size(1)
            )

        # convert linear+bias to a single linear for applying fusion path.
        @register_freezing_graph_pattern(
            CallFunction(
                aten.add.Tensor,
                CallFunction(mkldnn._linear_pointwise.default, *_linear_args),
                Arg(),
            ),
            pass_number=2,
            extra_check=is_linear_add_bias,
        )
        def linear_bias_pattern(match, *args):
            graph = match.graph
            add_node = match.output_node()
            linear_node = add_node.args[0]
            new_args = list(linear_node.args)
            new_args[2] = add_node.args[1]
            repl = graph.call_function(
                mkldnn._linear_pointwise.default, tuple(new_args)
            )
            repl.meta.update(add_node.meta)
            add_node.replace_all_uses_with(repl)
            match.erase_nodes()
            counters["inductor"]["mkldnn_linear_bias_matcher_count"] += 1
            counters["inductor"]["mkldnn_linear_bias_matcher_nodes"] += len(match.nodes)

    def _is_packable_mkldnn_rnn_layer(match):
        lstm_node = match.output_node()
        POS_WEIGHTS = [1, 2]
        POS_INPUTS = [0, 5, 6]
        POS_ARGS = POS_WEIGHTS + POS_INPUTS
        # Weights should be Constant
        if any(
            lstm_node.args[POS_WEIGHT].op != "get_attr" for POS_WEIGHT in POS_WEIGHTS
        ):
            return False

        # Meta info for weights and inputs should be available
        if any(lstm_node.args[POS_ARG].meta.get("val") is None for POS_ARG in POS_ARGS):
            return False

        # Check device
        if any(
            lstm_node.args[POS_ARG].meta.get("val").device.type != "cpu"
            for POS_ARG in POS_ARGS
        ):
            return False

        # Check dtype
        if any(
            lstm_node.args[POS_ARG].meta.get("val").dtype == torch.bfloat16
            and not is_mkldnn_bf16_supported("cpu")
            for POS_ARG in POS_ARGS
        ):
            return False
        if any(
            lstm_node.args[POS_ARG].meta.get("val").dtype == torch.float16
            and not is_mkldnn_fp16_supported("cpu")
            for POS_ARG in POS_ARGS
        ):
            return False

        return True

    def _is_packable_convolution(match):
        """
        Check if the node is supported for MKLDNN convolution.
        """
        conv_node = match.output_node()
        device_type = conv_node.meta.get("val").device.type
        # The operator 'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device.
        if match.kwargs["is_transposed"] and device_type == "xpu":
            return False

        input_meta_value = conv_node.args[0].meta.get("val")
        weight_meta_value = conv_node.args[1].meta.get("val")
        if input_meta_value is None or weight_meta_value is None:
            return False
        input_size = input_meta_value.shape
        if conv_node.args[1].op != "get_attr":
            return False
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type not in SUPPORTED_MKLDNN_DEVICES
                or (meta_value.dim() != 4 and meta_value.dim() != 5)
            ):
                return False

        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not is_mkldnn_bf16_supported(device_type):
                return False
        if (
            input_meta_value.dtype == torch.float16
            or weight_meta_value.dtype == torch.float16
        ):
            if not is_mkldnn_fp16_supported(device_type):
                return False
        is_transposed = conv_node.args[-3]
        if is_transposed:
            # TODO: Support dynamic shape case for MKLDNN conv transpose.
            if has_free_symbols(input_size):
                return False
            groups = conv_node.args[-1]
            in_channels = weight_meta_value.size(0)
            # doesn't support group_depthwise_conv_transpose.
            if groups > 1 and groups == in_channels:
                return False
            # Port from: aten/src/ATen/native/Convolution.cpp:is_output_padding_big
            output_paddings = conv_node.args[-2]
            strides = conv_node.args[3]
            if any(
                output_padding >= stride
                for output_padding, stride in zip(output_paddings, strides)
            ):
                return False
        return True

    def _is_packable_linear(match):
        """
        Check if the node is supported for MKLDNN linear.
        """

        def is_const_or_cat_by_const(weight):
            if weight.op == "get_attr":
                return True
            if weight.target != aten.cat.default:
                return False
            return all(arg.op == "get_attr" for arg in weight.args[0])

        linear_node = match.output_node()
        # mkldnn linear only supports beta=1 or 0 and alpha=1
        if linear_node.target is aten.addmm.default:
            alpha = linear_node.kwargs.get("alpha", 1.0)
            beta = linear_node.kwargs.get("beta", 1.0)
            if (beta != 0.0 and beta != 1.0) or alpha != 1.0:
                return False
        # weight_idx is 1 for aten.mm and is 2 for aten.addmm
        weight_idx = 2 if linear_node.target is aten.addmm.default else 1
        if not is_const_or_cat_by_const(linear_node.args[weight_idx]):
            return False
        input_meta_value = linear_node.args[weight_idx - 1].meta.get("val")
        weight_meta_value = linear_node.args[weight_idx].meta.get("val")
        if input_meta_value is None or weight_meta_value is None:
            return False
        if (
            input_meta_value.dtype == torch.float64
            or weight_meta_value.dtype == torch.float64
        ):
            return False
        is_lp_weight = weight_meta_value.dtype in (
            torch.bfloat16,
            torch.float16,
        )
        reduced_f32_matmul_enabled = torch.backends.mkldnn.matmul.fp32_precision in [  # type: ignore[attr-defined]
            "bf16",
            "tf32",
        ]
        use_reduced_f32_for_fp32_weight = (
            reduced_f32_matmul_enabled and weight_meta_value.dtype == torch.float32
        )
        compute_with_lp = is_lp_weight or use_reduced_f32_for_fp32_weight
        # on x86, for fp32, mkl should be enabled.
        # on aarch64, use mkldnn op for fp32 as well if acl is enabled
        if (
            not compute_with_lp
            and not mkldnn._is_mkldnn_acl_supported()
            and not torch._C.has_mkl
        ):
            return False
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or meta_value.dim() != 2
            ):
                return False
        if weight_idx == 2:
            bias_meta_value = linear_node.args[0].meta.get("val")
            if (
                bias_meta_value is None
                or meta_value.device.type != "cpu"
                or bias_meta_value.dim() != 1
                or bias_meta_value.size(0) != weight_meta_value.size(1)
            ):
                return False

        device_type = input_meta_value.device.type
        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not is_mkldnn_bf16_supported(device_type):
                return False
        if (
            input_meta_value.dtype == torch.float16
            or weight_meta_value.dtype == torch.float16
        ):
            if not is_mkldnn_fp16_supported(device_type):
                return False
        return True

    _aten_conv_args = (
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        KeywordArg("is_transposed"),
        Arg(),
        Arg(),
    )

    _aten_mkldnn_rnn_layer_args = (
        Arg(),  # input
        Arg(),  # weight0
        Arg(),  # weight1
        Arg(),  # weight2
        Arg(),  # weight3
        Arg(),  # hx_
        Arg(),  # cx_
        KeywordArg("reverse"),  # reverse
        Arg(),  # batch_sizes
        Arg(),  # mode
        Arg(),  # hidden_size
        Arg(),  # num_layers
        Arg(),  # has_biases
        Arg(),  # bidirectional
        Arg(),  # batch_first
        Arg(),  # train
    )

    def _register_weight_pack_pass():
        @register_freezing_graph_pattern(
            CallFunction(aten.convolution.default, *_aten_conv_args),
            extra_check=_is_packable_convolution,
        )
        def convolution(match, *args, **kwargs):
            is_transposed = kwargs.get("is_transposed")
            if not isinstance(is_transposed, bool):
                raise AssertionError(
                    f"expected is_transposed to be bool, got {type(is_transposed)}"
                )
            graph = match.graph
            conv_node = match.output_node()
            device_type = conv_node.args[0].meta.get("val").device.type
            mkldnn_device_op = _get_mkldnn_device_op(device_type)
            input_size = conv_node.args[0].meta.get("val").shape
            with graph.inserting_before(conv_node):
                constant_args = [args[4], args[3], args[5], args[-1]]
                packed_conv_op = mkldnn._convolution_pointwise.default
                if is_transposed:
                    constant_args.insert(1, args[-2])  # output_padding
                    packed_conv_op = mkldnn._convolution_transpose_pointwise.default

                if not has_free_symbols(input_size):
                    packed_weight_node = mkldnn_device_op.pack_conv_weight(
                        graph,
                        is_transposed,
                        args[1],
                        constant_args,
                        input_size,
                    )
                else:
                    if is_transposed:
                        raise AssertionError("expected is_transposed to be False")
                    # For dynamic shape case, we need to pack weight in runtime.
                    packed_weight_node = args[1]

                packed_conv_inputs = (
                    (args[0], packed_weight_node, args[2])
                    + tuple(constant_args)
                    + ("none", [], "")
                )
                packed_conv_node = graph.create_node(
                    "call_function", packed_conv_op, tuple(packed_conv_inputs)
                )
                conv_node.replace_all_uses_with(packed_conv_node)
                packed_conv_node.meta.update(conv_node.meta)
                graph.erase_node(conv_node)
            counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

        @register_freezing_graph_pattern(
            CallFunction(aten.mkldnn_rnn_layer.default, *_aten_mkldnn_rnn_layer_args),
            extra_check=_is_packable_mkldnn_rnn_layer,
        )
        def mkldnn_rnn_layer(match, *args, **kwargs):
            def get_item(graph, node, index):
                return graph.call_function(operator.getitem, (node, index))

            graph = match.graph
            lstm_node = match.output_node()
            weight0, weight1 = args[1:3]
            reverse = kwargs.get("reverse")
            packed_lstm_op = aten.mkldnn_rnn_layer.default
            hidden_size = args[9]
            has_biases = args[11]
            batch_first = args[13]
            with graph.inserting_before(lstm_node):
                packed_weight_op = mkldnn._reorder_mkldnn_rnn_layer_weight.default
                packed_weight_inputs = (
                    weight0,
                    weight1,
                    hidden_size,
                    reverse,
                    has_biases,
                    batch_first,
                )
                packed_weight_node = graph.create_node(
                    "call_function", packed_weight_op, packed_weight_inputs, {}, "name"
                )
                packed_weight_items = [
                    get_item(graph, packed_weight_node, i) for i in range(2)
                ]
                pack_lstm_inputs = (
                    args[0],
                    *packed_weight_items,
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                    reverse,
                    *args[7:],
                )

                packed_lstm_node = graph.create_node(
                    "call_function", packed_lstm_op, args=pack_lstm_inputs
                )
                lstm_node.replace_all_uses_with(packed_lstm_node)
                packed_lstm_node.meta.update(lstm_node.meta)
                graph.erase_node(lstm_node)
            counters["inductor"]["mkldnn_rnn_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_rnn_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

        @register_freezing_graph_pattern(
            CallFunction(
                aten.addmm.default,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            extra_check=_is_packable_linear,
            pass_number=1,
        )
        @register_freezing_graph_pattern(
            CallFunction(aten.mm.default, Arg(), Arg()),
            extra_check=_is_packable_linear,
            pass_number=1,
        )
        def linear(match, *args, **kwargs):
            graph = match.graph
            linear_node = match.output_node()
            input = args[0] if linear_node.target is aten.mm.default else args[1]
            bias = (
                None
                if linear_node.target is aten.mm.default
                or (
                    linear_node.target is aten.addmm.default
                    and linear_node.kwargs.get("beta", 1.0) == 0.0
                )
                else args[0]
            )
            weight = args[1] if linear_node.target is aten.mm.default else args[2]
            device_type = input.meta.get("val").device.type
            mkldnn_device_op = _get_mkldnn_device_op(device_type)
            with graph.inserting_before(linear_node):
                transpose_weight_node = graph.create_node(
                    "call_function", aten.permute.default, (weight, (1, 0))
                )
                weight_dtype = weight.meta.get("val").dtype
                is_lp_weight = weight_dtype in (
                    torch.bfloat16,
                    torch.float16,
                )
                reduced_f32_matmul_enabled = (
                    torch.backends.mkldnn.matmul.fp32_precision in ["bf16", "tf32"]  # type: ignore[attr-defined]
                )
                use_reduced_f32_for_fp32_weight = (
                    reduced_f32_matmul_enabled and weight_dtype == torch.float32
                )
                compute_with_lp = is_lp_weight or use_reduced_f32_for_fp32_weight
                batch_size = input.meta.get("val").shape[0]
                packed_weight_node = mkldnn_device_op.pack_linear_weight(
                    graph, compute_with_lp, transpose_weight_node, batch_size
                )
                packed_linear_node = mkldnn_device_op.pack_linear(
                    graph, compute_with_lp, batch_size, input, packed_weight_node, bias
                )

                linear_node.replace_all_uses_with(packed_linear_node)
                packed_linear_node.meta.update(linear_node.meta)
                graph.erase_node(linear_node)
            counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_linear_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

    def _eliminate_duplicate_packed_nodes(gm):
        """
        Combine packed weight nodes with the same inputs to reduce memory usage.
        for example:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(32, 32, bias=True)

            def forward(self, x):
                return self.linear(self.linear(x))

        the above's packed weight nodes are duplicate if two linear calls have same input size.
        """
        if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
            return gm

        packed_weight_ops = [
            torch._C._nn.mkldnn_reorder_conv2d_weight,
            torch._C._nn.mkldnn_reorder_conv3d_weight,
            mkldnn._reorder_convolution_transpose_weight,
            mkldnn._reorder_linear_weight,
            mkldnn._reorder_mkldnn_rnn_layer_weight,
        ]
        if torch._C.has_mkl:
            packed_weight_ops.append(torch.ops.mkl._mkl_reorder_linear_weight)

        for node in gm.graph.nodes:
            if node.target in packed_weight_ops and len(node.args[0].users) > 1:
                for user_node in list(node.args[0].users.keys()):
                    if (
                        user_node.target == node.target
                        and user_node != node
                        and user_node.args == node.args
                    ):
                        user_node.replace_all_uses_with(node)
                        gm.graph.erase_node(user_node)

    @functools.cache
    def _mkldnn_fusion_init():
        # TODO: aarch64: enable op fusion for acl once it supports fused operators. Disabling it for now.
        # Otherwise even the matmul or innerproduct can not be accelerated with acl
        if (
            not torch.backends.mkldnn.enabled
            or not torch.backends.mkldnn.is_available()
        ):
            return

        if not torch.ops.mkldnn._is_mkldnn_acl_supported():
            _register_unary_fusion()
            _register_inplace_fusion()
            _register_binary_unary_fusion()
            _register_binary_fusion()
            _register_quantization_lowerings()

        _register_woq_lowerings()

    @functools.cache
    def _mkldnn_weight_pack_init():
        if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
            _register_weight_pack_pass()
            _recover_linear()
            _register_int8_woq_concat_linear_pattern()
<|endoftext|># mypy: allow-untyped-defs
import gc
import logging
import os
import random
import traceback

import numpy

import torch
import torch.optim as optim
from torch.utils._ordered_set import OrderedSet

from .. import config


logger: logging.Logger = logging.getLogger(__name__)

MAIN_RANDOM_SEED = 1337

# Set the CUBLAS_WORKSPACE_CONFIG environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# If the two forward functions involve any non-deterministic operations,
# such as certain types of parallelism or asynchronous execution,
# this can also lead to different outputs.
def set_deterministic() -> None:
    """Make torch manual seed deterministic."""

    torch.manual_seed(MAIN_RANDOM_SEED)
    random.seed(MAIN_RANDOM_SEED)
    numpy.random.seed(MAIN_RANDOM_SEED)
    torch.use_deterministic_algorithms(True)


def clean_memory() -> None:
    """Clean memory to avoid OOM."""
    gc.collect()
    torch.cuda.empty_cache()


# We compare the numerical results before and after pre/post grad fx passes
# transformation to make sure the numerical results are the same.
def compare_dict_tensors(dict_base, dict_control, precision):
    if len(OrderedSet(dict_base.keys())) != len(OrderedSet(dict_control.keys())):
        logger.warning("Mismatch keys found before and after pre/post grad fx passes.")
        logger.debug("keys before pre/post grad fx passes %s", dict_base.keys())
        logger.debug("keys after pre/post grad fx passes %s", dict_control.keys())
        return False
    is_allclose = True
    for key in dict_base:
        if key not in dict_control:
            logger.warning(
                "Mismatch parameter name %s does not exist after pre/post grad fx passes",
                key,
            )
        # Some parameters have `None`, and not every param has a valid .grad field, we skip them
        if dict_base[key] is None or dict_control[key] is None:
            continue
        if not torch.allclose(
            dict_base[key],
            dict_control[key],
            rtol=precision,
            atol=precision,
            equal_nan=True,
        ):
            logger.warning(
                "Mismatch parameter values found before and after pre/post grad fx passes."
            )
            logger.debug("value before pre/post grad fx passes %s", dict_base[key])
            logger.debug("value after pre/post grad fx passes %s", dict_control[key])
            is_allclose = False
    return is_allclose


def compare_tuple_tensors(tuple_base, tuple_control, precision):
    if len(tuple_base) != len(tuple_control):
        logger.warning(
            "Mismatch fw output length. before transformation: %s, after transformation: %s",
            len(tuple_base),
            len(tuple_control),
        )
        return False
    is_allclose = True
    for i in range(len(tuple_base)):
        # Some parameters have `None`, we skip them
        if tuple_base[i] is None or tuple_control[i] is None:
            continue
        if not torch.allclose(
            tuple_base[i],
            tuple_control[i],
            rtol=precision,
            atol=precision,
            equal_nan=True,
        ):
            logger.debug(
                "forward output before pre/post grad fx passes %s", tuple_base[i]
            )
            logger.debug(
                "forward output after pre/post grad fx passes %s", tuple_control[i]
            )
            is_allclose = False
    return is_allclose


def compare_parameters(model_base, model_control, precision):
    return compare_dict_tensors(
        dict(model_base.named_parameters()),
        dict(model_control.named_parameters()),
        precision,
    )


def compare_forward_output(pred_base, pred_control, precision):
    return compare_tuple_tensors(
        pred_base,
        pred_control,
        precision,
    )


def compare_gradients(model_base, model_control, precision):
    grad_base = {key: param.grad for key, param in model_base.named_parameters()}
    grad_pt2 = {key: param.grad for key, param in model_control.named_parameters()}
    return compare_dict_tensors(
        grad_base,
        grad_pt2,
        precision,
    )


def run_model(
    model_base, model_control, model_input, num_iterations=10, precision=1e-4
):
    clean_memory()
    for i in range(num_iterations):
        logger.info("start %s iteration", i)
        set_deterministic()
        pred_base = model_base(*model_input)
        set_deterministic()
        pred_control = model_control(*model_input)

        res = compare_parameters(model_base, model_control, precision)
        logger.info("compare parameters. Numerical result : %s", res)

        res = compare_forward_output(pred_base, pred_control, precision)
        logger.info("compare loss/predict. Numerical result : %s", res)
        # tensor may not have a grad_fn
        try:
            _ = pred_base[0].sum().backward(retain_graph=True)
            _ = pred_control[0].sum().backward(retain_graph=True)
            res = compare_gradients(model_base, model_control, precision)
            logger.info("compare param grad. Numerical result : %s", res)
        except Exception:
            logger.exception("Exception when comparing gradients")
            traceback.print_exc()

        if config.fx_passes_numeric_check["requires_optimizer"]:
            try:
                optimizer_base = optim.SGD(
                    [param for name, param in model_base.named_parameters()], lr=0.01
                )
                optimizer_base.step()

                optimizer_control = optim.SGD(
                    [param for name, param in model_control.named_parameters()], lr=0.01
                )
                optimizer_control.step()

                res = compare_parameters(model_base, model_control, precision)
                logger.info(
                    "compare parameters with optimizer added. Numerical result : %s",
                    res,
                )
            except Exception:
                logger.exception(
                    "Exception when optimizer is added to check parameter names"
                )
                traceback.print_exc()
        else:
            logger.warning(
                "no parameter with optimizer to compare with length %s before transformation"
                " and the length %s after transformation",
                len(dict(model_base.named_parameters())),
                len(dict(model_control.named_parameters())),
            )


def numeric_check_if_enabled(
    gm_before_fx_passes,
    gm_after_fx_passes,
    example_inputs,
    num_iterations,
    precision,
):
    # need to topo-sort graphmodule before we run the model,
    # otherwise it may fail as refer before def
    # fail silently in order not to block the model run
    try:
        with torch.autograd.set_detect_anomaly(True):
            run_model(
                gm_before_fx_passes,
                gm_after_fx_passes,
                example_inputs,
                num_iterations=num_iterations,
                precision=precision,
            )
    except Exception as e:
        logger.warning(
            "Runtime numeric check failed in pre grad fx passes with error: %s", e
        )
        traceback.print_exc()
<|endoftext|>"""SPMD graph verification for overlap scheduling.

Verifies all ranks have identical FX graph structure before collective
reordering passes. Non-SPMD graphs cause NCCL collective ordering
mismatches and hangs.
"""

import hashlib
import logging
from collections import Counter

import torch
from torch._inductor import config
from torch._logging import trace_structured


log = logging.getLogger(__name__)


def _compute_hash(gm: torch.fx.GraphModule) -> int | None:
    """Compute a structural hash of the graph including tensor metadata.

    Uses FxGraphCachePickler(device_id_agnostic=True) to serialize
    (target, val) per call_function node, capturing op targets and
    FakeTensor metadata (dtype, shape, stride, etc.) with device indices
    normalized to 0.

    Returns None if the graph contains unpicklable objects.
    """
    from torch._inductor.codecache import BypassFxGraphCache, FxGraphCachePickler

    try:
        pickler = FxGraphCachePickler(gm, device_id_agnostic=True)
        data = pickler.dumps(
            tuple(
                (str(n.target), n.meta.get("val"))
                for n in gm.graph.nodes
                if n.op == "call_function"
            )
        )
        digest = hashlib.blake2b(data, digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=True)
    except BypassFxGraphCache:
        # FxGraphCachePickler can't serialize certain objects:
        # mkldnn tensors, BackwardState, torchbind objects, or general
        # pickle failures. Skip the SPMD check gracefully.
        log.warning("SPMD check: skipping, unpicklable graph objects", exc_info=True)
        return None


def _build_diag_fingerprint(
    gm: torch.fx.GraphModule,
) -> tuple[tuple[str, str | None], ...]:
    """Build human-readable fingerprint for mismatch diagnostics.

    Only called on the rare mismatch path.
    """
    from torch._inductor.codecache import extract_tensor_metadata_for_cache_key

    entries: list[tuple[str, str | None]] = []
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        target_str = str(n.target)
        val = n.meta.get("val")
        entries.append(
            (
                target_str,
                _format_val_metadata(val, extract_tensor_metadata_for_cache_key),
            )
        )
    return tuple(entries)


def _format_val_metadata(val: object, extract_fn: object) -> str | None:
    """Format node val metadata for human-readable diagnostics."""
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return str(extract_fn(val))  # type: ignore[operator]
    if isinstance(val, (tuple, list)):
        parts = []
        for v in val:
            if isinstance(v, torch.Tensor):
                parts.append(str(extract_fn(v)))  # type: ignore[operator]
            else:
                parts.append(str(type(v).__name__))
        return f"({', '.join(parts)})"
    return str(type(val).__name__)


def spmd_check(gm: torch.fx.GraphModule) -> bool:
    """Verify all ranks have identical FX graph structure (SPMD).

    Computes a structural hash (op targets + tensor metadata including
    shapes, dtypes, strides) and compares across ranks.
    On mismatch, emits a diagnostic report to stdout, logging, and
    trace_structured.

    Returns True if graphs match (SPMD), False on mismatch.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return True

    structure_hash = _compute_hash(gm)
    if structure_hash is None:
        return True

    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.distributed.distributed_c10d import _get_default_group

    pg = _get_default_group()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    with unset_fake_temporarily():
        all_hashes: list[int] = [0] * world_size
        dist.all_gather_object(all_hashes, structure_hash, pg)

    if all(h == all_hashes[0] for h in all_hashes):
        return True

    # Mismatch detected — build and gather diagnostic fingerprints
    fingerprint = _build_diag_fingerprint(gm)
    with unset_fake_temporarily():
        all_fingerprints: list[tuple[object, ...]] = [() for _ in range(world_size)]
        dist.all_gather_object(all_fingerprints, fingerprint, pg)

    report = _build_mismatch_report(all_fingerprints, rank, world_size)

    print(report, flush=True)
    log.warning("\n%s", report)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "inductor_spmd_graph_mismatch",
            "encoding": "string",
        },
        payload_fn=lambda: report,
    )

    if config.aten_distributed_optimizations.spmd_mismatch == "error":
        raise RuntimeError(
            "SPMD graph verification failed. "
            'Set aten_distributed_optimizations.spmd_mismatch="warn" '
            "to warn instead of fail.\n" + report
        )

    return False


def _entry_target(entry: object) -> str:
    """Extract the target string from a fingerprint entry."""
    if isinstance(entry, tuple):
        return str(entry[0])
    return str(entry)


def _entry_metadata(entry: object) -> str:
    """Format metadata from a fingerprint entry, if present."""
    if isinstance(entry, tuple) and len(entry) >= 2:
        meta = entry[1]
        if meta is not None:
            return f" meta={meta}"
    return ""


def _build_mismatch_report(
    all_fingerprints: list[tuple[object, ...]],
    rank: int,
    world_size: int,
) -> str:
    """Build diagnostic report for SPMD graph mismatch."""
    lines = [
        "=" * 80,
        f"SPMD GRAPH MISMATCH — rank {rank}, world_size={world_size}",
        "=" * 80,
    ]

    # Node count per rank
    counts = [len(t) for t in all_fingerprints]
    lines.append("NODE COUNTS PER RANK:")
    for r in range(world_size):
        marker = " <--" if counts[r] != counts[0] else ""
        lines.append(f"  rank {r}: {counts[r]} call_function nodes{marker}")
    lines.append("")

    # Find entries that differ
    ref = all_fingerprints[0]
    for r in range(1, world_size):
        other = all_fingerprints[r]
        if other == ref:
            continue
        lines.append(f"DIFFS rank 0 vs rank {r}:")

        # Show first few positional differences
        max_diffs = 10
        shown = 0
        for i, (a, b) in enumerate(zip(ref, other)):
            if a != b and shown < max_diffs:
                lines.append(f"  node {i}:")
                lines.append(f"    rank 0: {_entry_target(a)}{_entry_metadata(a)}")
                lines.append(f"    rank {r}: {_entry_target(b)}{_entry_metadata(b)}")
                shown += 1

        # Also show count-based diffs for op targets
        ref_targets = [_entry_target(e) for e in ref]
        other_targets = [_entry_target(e) for e in other]

        ref_counts = Counter(ref_targets)
        other_counts = Counter(other_targets)
        only_ref = ref_counts - other_counts
        only_other = other_counts - ref_counts
        if only_ref:
            lines.append("  Only on rank 0:")
            for op, cnt in only_ref.most_common(10):
                lines.append(f"    {op} (x{cnt})")
        if only_other:
            lines.append(f"  Only on rank {r}:")
            for op, cnt in only_other.most_common(10):
                lines.append(f"    {op} (x{cnt})")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)
<|endoftext|>"""
Profile-Guided Estimation (PGE) for overlap scheduling.

Parses a Chrome Trace JSON (from torch.profiler) and builds lookup tables
for kernel runtimes (collectives, matmuls, attention, custom ops, etc.).
Used as a custom_runtime_estimation hook in the overlap scheduler.

When the same profile is loaded on all ranks, estimates are deterministic
and no cross-rank synchronization is needed.
"""

from __future__ import annotations

import functools
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch._inductor.analysis.profile_analysis import (
    _create_extern_mapping,
    _dtype_map,
    _get_size_from_string,
    ParseException,
)
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_all_to_all_tensor,
    is_reduce_scatter_tensor,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def _rank_stride(ranks: tuple[int, ...]) -> int | None:
    """Compute the stride of a sorted rank tuple, or None if non-uniform.

    Examples:
        (0, 2, 4, 6) → stride 2
        (0, 1)       → stride 1
        (1, 3, 5, 7) → stride 2
        (0, 1, 4, 5) → None (non-uniform)
    """
    if len(ranks) <= 1:
        return None
    stride = ranks[1] - ranks[0]
    if stride <= 0:
        return None
    for i in range(2, len(ranks)):
        if ranks[i] - ranks[i - 1] != stride:
            return None
    return stride


@dataclass
class CollectiveRecord:
    """A single collective kernel observation from the profile."""

    collective_name: str  # "all_gather_into_tensor", "reduce_scatter_tensor", etc.
    pg_ranks: tuple[int, ...]  # sorted rank tuple
    group_size: int
    in_nelems: int  # "In msg nelems" from profile
    out_nelems: int  # "Out msg nelems" from profile
    dtype: str  # "Float", "BFloat16", etc.
    duration_us: float


@dataclass
class OpRecord:
    """A single op observation from the profile (any CPU op with GPU kernels)."""

    op_name: str  # normalized name, e.g. "aten::mm", "mylib::my_custom_op"
    input_shapes: tuple[tuple[int, ...], ...]
    input_strides: tuple[tuple[int, ...], ...]
    dtype: torch.dtype | None
    duration_us: float  # sum of all GPU kernels for this CPU op


def _to_nested_tuple(x: Any) -> Any:
    """Recursively convert nested lists to tuples for hashability."""
    if isinstance(x, (list, tuple)):
        return tuple(_to_nested_tuple(i) for i in x)
    return x


@dataclass
class ProfileData:
    """Parse Chrome Trace JSON and build lookup tables for kernel runtimes."""

    collectives: list[CollectiveRecord] = field(default_factory=list)
    ops: list[OpRecord] = field(default_factory=list)
    pg_configs: dict[str, tuple[int, ...]] = field(default_factory=dict)

    # Lookup indices built after loading
    _collective_index: dict[
        tuple[str, tuple[int, ...], str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Fallback index by mesh dimension (name, stride, group_size, dtype).
    # Matches PGs belonging to the same mesh dimension regardless of specific ranks.
    # E.g. (0,2,4,6) and (1,3,5,7) both have stride=2, size=4 → same mesh dim.
    _collective_index_by_mesh_dim: dict[
        tuple[str, int, int, str], list[tuple[int, float]]
    ] = field(default_factory=dict)
    # Count of distinct PGs per mesh dimension (stride, group_size) — used for
    # ambiguity check (skip fallback if multiple PGs share the same mesh dim).
    _pg_count_by_mesh_dim: dict[tuple[int, int], int] = field(default_factory=dict)
    # Generic op index: (op_name, input_shapes, input_strides, dtype) -> avg_dur_us
    _op_index: dict[
        tuple[
            str,
            tuple[tuple[int, ...], ...],
            tuple[tuple[int, ...], ...],
            torch.dtype | None,
        ],
        float,
    ] = field(default_factory=dict)
    # Peak observed bandwidth per PG (GB/s), computed from largest messages
    _pg_peak_bw: dict[tuple[int, ...], float] = field(default_factory=dict)
    # Mesh-dimension fallback: (stride, group_size) -> peak BW (GB/s)
    _mesh_dim_peak_bw: dict[tuple[int, int], float] = field(default_factory=dict)

    def load(self, trace_path: str) -> None:
        """Load and parse a Chrome Trace JSON file."""
        import os

        if not os.path.isfile(trace_path):
            raise FileNotFoundError(
                f"PGE trace file not found: {trace_path}. "
                f"Check config.aten_distributed_optimizations.profile_guided_estimations_profile_path"
            )
        with open(trace_path) as f:
            data = json.load(f)

        self._parse_pg_configs(data)
        self._parse_events(data)
        self._build_indices()

        log.info(
            "PGE loaded: %d collectives, %d op records (%d distinct shapes), %d PGs",
            len(self.collectives),
            len(self.ops),
            len(self._op_index),
            len(self.pg_configs),
        )

    def _parse_pg_configs(self, data: dict[str, Any]) -> None:
        dist_info = data.get("distributedInfo", {})
        pg_config = dist_info.get("pg_config", {})
        # pg_config can be a list of dicts or a dict of dicts
        if isinstance(pg_config, list):
            for pg_info in pg_config:
                pg_name = str(pg_info.get("pg_name", ""))
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))
        elif isinstance(pg_config, dict):
            for pg_name, pg_info in pg_config.items():
                ranks = pg_info.get("ranks", [])
                if ranks:
                    self.pg_configs[pg_name] = tuple(sorted(ranks))

    def _parse_events(self, data: dict[str, Any]) -> None:
        events = data.get("traceEvents", [])
        # Reuse profile_analysis's External id -> CPU op mapping
        try:
            extern_mapping = _create_extern_mapping(data)
        except (ParseException, KeyError):
            # Malformed trace (e.g. duplicate External ids, missing traceEvents)
            extern_mapping = defaultdict(list)
            for ev in events:
                if (
                    isinstance(ev, dict)
                    and ev.get("cat") == "cpu_op"
                    and "args" in ev
                    and "External id" in ev["args"]
                ):
                    extern_mapping[ev["args"]["External id"]].append(ev)

        # Build External id -> total GPU kernel duration
        gpu_dur: dict[int, float] = defaultdict(float)
        for ev in events:
            if not isinstance(ev, dict) or ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            eid = args.get("External id")
            dur = ev.get("dur", 0.0)
            if eid is not None and dur > 0:
                gpu_dur[eid] += dur

        # Parse collectives from GPU kernel events directly
        # (NCCL kernels carry collective metadata in args)
        for ev in events:
            if not isinstance(ev, dict) or ev.get("cat") != "kernel":
                continue
            args = ev.get("args", {})
            coll_name = args.get("Collective name")
            if coll_name is None:
                continue
            pg_name = args.get("Process Group Name", "")
            pg_ranks_str = args.get("Process Group Ranks", "")
            group_size = args.get("Group size", 0)
            in_nelems = args.get("In msg nelems", 0)
            out_nelems = args.get("Out msg nelems", 0)
            dtype = args.get("dtype", "")
            dur = ev.get("dur", 0.0)
            if dur <= 0:
                continue

            pg_ranks = self._parse_ranks(pg_ranks_str, pg_name)

            self.collectives.append(
                CollectiveRecord(
                    collective_name=coll_name,
                    pg_ranks=pg_ranks,
                    group_size=group_size,
                    in_nelems=in_nelems,
                    out_nelems=out_nelems,
                    dtype=dtype,
                    duration_us=dur,
                )
            )

        # Parse all CPU ops that have associated GPU kernels
        for eid, cpu_evs in extern_mapping.items():
            if not cpu_evs:
                continue
            total_dur = gpu_dur.get(eid, 0.0)
            if total_dur <= 0:
                continue
            cpu_ev = cpu_evs[0]
            self._parse_op(cpu_ev.get("name", ""), cpu_ev.get("args", {}), total_dur)

    def _parse_ranks(self, ranks_str: str, pg_name: str) -> tuple[int, ...]:
        """Parse rank list from profile string or fall back to pg_configs."""
        if isinstance(ranks_str, str) and ranks_str.startswith("["):
            try:
                ranks = json.loads(ranks_str)
                return tuple(sorted(ranks))
            except (json.JSONDecodeError, TypeError):
                pass
        # Fall back to pg_configs
        if pg_name in self.pg_configs:
            return self.pg_configs[pg_name]
        return ()

    def _parse_op(self, name: str, args: dict[str, Any], total_dur: float) -> None:
        """Parse any CPU op into a generic OpRecord."""
        input_dims = args.get("Input Dims", [])
        input_strides = args.get("Input Strides", [])
        input_types = args.get("Input type", [])
        if not input_dims:
            return
        dtype_str = input_types[0] if input_types else ""
        dtype = _dtype_map.get(dtype_str)
        # Skip empty entries (non-tensor args like scalars/None) so the
        # tuples match what _get_node_input_shapes/strides extract from FX nodes.
        shapes = tuple(
            _to_nested_tuple(d)
            for d in input_dims
            if isinstance(d, (list, tuple)) and d
        )
        strides = tuple(
            _to_nested_tuple(d)
            for d in input_strides
            if isinstance(d, (list, tuple)) and d
        )
        if not shapes:
            return
        self.ops.append(
            OpRecord(
                op_name=name,
                input_shapes=shapes,
                input_strides=strides,
                dtype=dtype,
                duration_us=total_dur,
            )
        )

    def _build_indices(self) -> None:
        """Build lookup indices from parsed records."""
        coll_idx: dict[tuple[str, tuple[int, ...], str], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        coll_idx_by_mesh_dim: dict[
            tuple[str, int, int, str], list[tuple[int, float]]
        ] = defaultdict(list)
        # Track distinct PG rank sets per mesh dimension for ambiguity check
        pg_sets_by_mesh_dim: dict[tuple[int, int], OrderedSet[tuple[int, ...]]] = (
            defaultdict(OrderedSet)
        )
        for rec in self.collectives:
            norm_name = self._normalize_collective_name(rec.collective_name)
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            coll_idx[(norm_name, rec.pg_ranks, rec.dtype)].append(
                (rec.out_nelems, rec.duration_us)
            )
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                coll_idx_by_mesh_dim[(norm_name, stride, gs, rec.dtype)].append(
                    (rec.out_nelems, rec.duration_us)
                )
                pg_sets_by_mesh_dim[(stride, gs)].add(rec.pg_ranks)
        # Sort by nelems for interpolation
        self._collective_index = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx.items()
        }
        self._collective_index_by_mesh_dim = {
            k: sorted(v, key=lambda x: x[0]) for k, v in coll_idx_by_mesh_dim.items()
        }
        self._pg_count_by_mesh_dim = {
            k: len(pgs) for k, pgs in pg_sets_by_mesh_dim.items()
        }

        op_groups: defaultdict[
            tuple[
                str,
                tuple[tuple[int, ...], ...],
                tuple[tuple[int, ...], ...],
                torch.dtype | None,
            ],
            list[float],
        ] = defaultdict(list)
        for rec in self.ops:
            key = (rec.op_name, rec.input_shapes, rec.input_strides, rec.dtype)
            op_groups[key].append(rec.duration_us)
        self._op_index = {k: sum(v) / len(v) for k, v in op_groups.items()}

        # Per-PG peak bandwidth: compute bytes/us for each collective observation,
        # then take the max from the top-N largest messages per PG (where bandwidth
        # is most representative of hardware speed, not dominated by startup latency).
        # Uses output-convention bytes (matching _estimate_with_pg_bandwidth).
        _TOP_N = 5  # consider top N largest messages for peak BW
        pg_bw_samples: dict[tuple[int, ...], list[tuple[int, float]]] = defaultdict(
            list
        )
        mesh_dim_bw_samples: dict[tuple[int, int], list[tuple[int, float]]] = (
            defaultdict(list)
        )
        for rec in self.collectives:
            if rec.out_nelems <= 0 or rec.duration_us <= 0:
                continue
            gs = len(rec.pg_ranks) if rec.pg_ranks else rec.group_size
            elem_bytes = self._dtype_elem_bytes(rec.dtype)
            total_bytes = rec.out_nelems * elem_bytes
            bw_gbps = total_bytes / (rec.duration_us * 1e-6) / 1e9  # GB/s
            pg_bw_samples[rec.pg_ranks].append((total_bytes, bw_gbps))
            stride = _rank_stride(rec.pg_ranks)
            if stride is not None:
                mesh_dim_bw_samples[(stride, gs)].append((total_bytes, bw_gbps))

        def _peak_bw_from_samples(
            samples: list[tuple[int, float]],
        ) -> float:
            """Get peak BW from the top-N largest messages."""
            # Sort by message size descending, take top N, return max BW
            sorted_samples = sorted(samples, key=lambda x: x[0], reverse=True)
            top = sorted_samples[:_TOP_N]
            return max(bw for _, bw in top) if top else 0.0

        self._pg_peak_bw = {
            pg: _peak_bw_from_samples(samples)
            for pg, samples in pg_bw_samples.items()
            if samples
        }
        self._mesh_dim_peak_bw = {
            key: _peak_bw_from_samples(samples)
            for key, samples in mesh_dim_bw_samples.items()
            if samples
        }

    def get_collective_keys(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Return the collective index keys: (name, pg_ranks, dtype)."""
        return list(self._collective_index.keys())

    @property
    def op_count(self) -> int:
        """Number of distinct op shapes in the index."""
        return len(self._op_index)

    def get_op_names(self) -> list[str]:
        """Return distinct op names in the op index."""
        return list(OrderedSet(name for name, _, _, _ in self._op_index))

    @staticmethod
    def _dtype_elem_bytes(dtype: str) -> int:
        """Return bytes per element for a dtype string (NCCL CamelCase or TypeMeta)."""
        return _get_size_from_string(dtype.lower())

    @staticmethod
    def _normalize_collective_name(name: str) -> str:
        """Normalize collective name between profile and FX conventions.

        Profile uses: _allgather_base, allreduce, reduce_scatter_tensor_coalesced
        FX uses: all_gather_into_tensor, all_reduce, reduce_scatter_tensor
        """
        n = name.lower()
        if "allgather" in n or "all_gather" in n:
            return "all_gather"
        if "reduce_scatter" in n:
            return "reduce_scatter"
        if "allreduce" in n or "all_reduce" in n:
            return "all_reduce"
        if "all_to_all" in n or "alltoall" in n:
            return "all_to_all"
        return name

    # Maximum ratio of target_nelems / max_observed before switching from
    # log-log extrapolation to bandwidth-based estimation.
    EXTRAPOLATION_CAP = 2.0

    def _estimate_with_pg_bandwidth(
        self,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> float | None:
        """Estimate collective duration using peak observed bandwidth for this PG.

        Used when the target size exceeds the extrapolation cap. Returns ms or None.
        """
        bw_gbps = self._pg_peak_bw.get(pg_ranks)
        if bw_gbps is None or bw_gbps <= 0:
            # Try mesh-dimension fallback
            stride = _rank_stride(pg_ranks)
            gs = len(pg_ranks)
            if stride is not None:
                bw_gbps = self._mesh_dim_peak_bw.get((stride, gs))
        if bw_gbps is None or bw_gbps <= 0:
            return None  # fall through to analytical
        elem_bytes = self._dtype_elem_bytes(dtype)
        total_bytes = nelems * elem_bytes
        dur_ms = total_bytes / (bw_gbps * 1e6)  # GB/s → bytes/ms = 1e6
        return dur_ms

    def lookup_collective(
        self,
        collective_name: str,
        pg_ranks: tuple[int, ...],
        nelems: int,
        dtype: str,
    ) -> tuple[float, str] | None:
        """Look up collective duration in ms. Returns (duration_ms, source) or None.

        ``source`` is ``"profile"`` for exact/interpolated matches, or
        ``"pg_bandwidth"`` when bandwidth-based extrapolation was used.

        Tries exact rank match first, then falls back to mesh-dimension match
        (e.g. (0,2,4,6) and (1,3,5,7) both have stride=2, size=4 → same mesh dim).

        When the target size exceeds EXTRAPOLATION_CAP * max_observed, uses
        bandwidth-based estimation from peak observed bandwidth instead of
        linear extrapolation (which overestimates for large messages).
        """
        norm_name = self._normalize_collective_name(collective_name)
        # Try exact rank match first
        key = (norm_name, pg_ranks, dtype)
        entries = self._collective_index.get(key)
        if not entries:
            # Fall back to mesh-dimension match
            gs = len(pg_ranks)
            stride = _rank_stride(pg_ranks)
            if (
                stride is not None
                and self._pg_count_by_mesh_dim.get((stride, gs), 0) == 1
            ):
                mesh_dim_key = (norm_name, stride, gs, dtype)
                entries = self._collective_index_by_mesh_dim.get(mesh_dim_key)
            if not entries:
                return None

        # Exact match
        for n, dur in entries:
            if n == nelems:
                return (dur / 1e3, "profile")  # us -> ms

        # Check extrapolation distance: if target is far beyond observed range,
        # use bandwidth-based model instead of log-log extrapolation
        max_observed = max((n for n, _ in entries if n > 0), default=0)
        if max_observed > 0 and nelems > max_observed * self.EXTRAPOLATION_CAP:
            est = self._estimate_with_pg_bandwidth(pg_ranks, nelems, dtype)
            if est is not None:
                return (est, "pg_bandwidth")
            # Fall through to log-log if no BW data available

        # Interpolation in log-log space
        result = self._interpolate_log_log(entries, nelems)
        if result is not None:
            return (result, "profile")
        return None

    def _interpolate_log_log(
        self, entries: list[tuple[int, float]], target_nelems: int
    ) -> float | None:
        """Interpolate duration in log-log space (log(nelems) vs log(dur))."""
        if not entries or target_nelems <= 0:
            return None

        log_target = math.log(target_nelems)

        # Find bracketing entries
        lower: tuple[int, float] | None = None
        upper: tuple[int, float] | None = None
        for n, dur in entries:
            if n <= 0 or dur <= 0:
                continue
            if n <= target_nelems:
                lower = (n, dur)
            if n >= target_nelems and upper is None:
                upper = (n, dur)

        if lower is not None and upper is not None:
            log_n0, log_d0 = math.log(lower[0]), math.log(lower[1])
            log_n1, log_d1 = math.log(upper[0]), math.log(upper[1])
            if log_n1 == log_n0:
                return lower[1] / 1e3
            t = (log_target - log_n0) / (log_n1 - log_n0)
            log_dur = log_d0 + t * (log_d1 - log_d0)
            return math.exp(log_dur) / 1e3  # us -> ms
        elif lower is not None:
            # Linear extrapolation (not log-log) from nearest lower;
            # EXTRAPOLATION_CAP in lookup_collective limits how far this reaches.
            return (lower[1] * target_nelems / lower[0]) / 1e3
        elif upper is not None:
            # Linear extrapolation from nearest upper
            return (upper[1] * target_nelems / upper[0]) / 1e3

        return None

    def lookup_op(
        self,
        op_name: str,
        input_shapes: tuple[tuple[int, ...], ...],
        input_strides: tuple[tuple[int, ...], ...],
        dtype: torch.dtype | None,
    ) -> float | None:
        """Look up op duration in ms by exact shape+stride match. Returns None on miss."""
        key = (op_name, input_shapes, input_strides, dtype)
        dur_us = self._op_index.get(key)
        if dur_us is not None:
            return dur_us / 1e3  # us -> ms
        return None


@functools.cache
def _dtype_to_nccl_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to NCCL/ScalarType name (for collective matching).

    Derives the name from torch.Tensor.type() which returns e.g.
    "torch.BFloat16Tensor" -> "BFloat16".
    """
    return (
        torch.tensor([], dtype=dtype)
        .type()
        .removeprefix("torch.")
        .removesuffix("Tensor")
    )


def _get_node_dtype(node: fx.Node) -> torch.dtype | None:
    """Extract dtype from FX node metadata."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.dtype
    if isinstance(val, (list, tuple)) and val:
        first = val[0]
        if isinstance(first, torch.Tensor):
            return first.dtype
    return None


def _fx_target_to_profile_name(node: fx.Node) -> str | None:
    """Convert FX node target to the profile op name format.

    FX: torch.ops.aten.mm.default → "aten::mm"
    FX: torch.ops.deepep.dispatch.default → "deepep::dispatch"
    """
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        # e.g. "aten::mm" from torch.ops.aten.mm.default
        ns = target.namespace
        op_name = target._schema.name.split("::")[-1]
        return f"{ns}::{op_name}"
    if hasattr(target, "__name__"):
        return target.__name__
    return None


def _get_node_input_shapes_and_strides(
    node: fx.Node,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]] | None:
    """Extract input shapes and strides from FX node tensor args.

    Returns (shapes, strides) or None if no tensor args or symbolic dims.
    """
    from torch._inductor.fx_passes.node_runtime_estimation import get_hint

    shapes: list[tuple[int, ...]] = []
    strides: list[tuple[int, ...]] = []
    for arg in node.args:
        if not isinstance(arg, fx.Node):
            continue
        val = arg.meta.get("val")
        if isinstance(val, torch.Tensor):
            resolved_shape = []
            for s in val.shape:
                h = get_hint(s)
                if h is None:
                    return None
                resolved_shape.append(h)
            resolved_stride = []
            for s in val.stride():
                h = get_hint(s)
                if h is None:
                    return None
                resolved_stride.append(h)
            shapes.append(tuple(resolved_shape))
            strides.append(tuple(resolved_stride))
    if not shapes:
        return None
    return tuple(shapes), tuple(strides)


def _is_collective_node(node: fx.Node) -> bool:
    """Check if node is a collective communication op."""
    return (
        is_all_gather_into_tensor(node)
        or is_reduce_scatter_tensor(node)
        or is_all_reduce_tensor(node)
        or is_all_to_all_tensor(node)
    )


def _get_collective_info(
    node: fx.Node,
) -> tuple[str, tuple[int, ...], int, str] | None:
    """Extract (collective_name, pg_ranks, nelems, dtype) from collective node."""
    import torch.distributed as c10d
    from torch.fx.operator_schemas import normalize_function

    if not c10d.is_initialized():
        return None

    target = node.target
    if not isinstance(target, torch._ops.OpOverload):
        return None
    collective_name = target.name().split("::")[-1].split(".")[0]

    opt = normalize_function(
        target,
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt is None:
        return None
    _, kwargs = opt
    group_name = kwargs.get("group_name", "")

    try:
        from torch.distributed.distributed_c10d import (
            _resolve_process_group,
            get_process_group_ranks,
        )

        pg = _resolve_process_group(group_name)
        pg_ranks = tuple(sorted(get_process_group_ranks(pg)))
    except (RuntimeError, KeyError, ValueError):
        log.debug(
            "PGE: failed to resolve process group for %s", node.name, exc_info=True
        )
        return None

    # Get nelems from input tensor
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        nelems = 1
        for s in val.shape:
            nelems *= int(s)
        dtype = _dtype_to_nccl_str(val.dtype)
    else:
        # Try first arg
        if node.args and isinstance(node.args[0], fx.Node):
            inp_val = node.args[0].meta.get("val")
            if isinstance(inp_val, torch.Tensor):
                nelems = 1
                for s in inp_val.shape:
                    nelems *= int(s)
                dtype = _dtype_to_nccl_str(inp_val.dtype)
            else:
                return None
        else:
            return None

    return (collective_name, pg_ranks, nelems, dtype)


class ProfileGuidedEstimator:
    """Profile-guided runtime estimator for FX nodes.

    Implements the ``custom_runtime_estimation`` interface:
    ``(fx.Node, int | None) -> float | None`` (returns ms or None for fallback).

    Handles collectives via interpolation (latency + bandwidth model) and all
    other ops (matmul, SDPA, custom ops, etc.) via exact shape match from the
    profile trace.
    """

    def __init__(
        self,
        trace_path: str,
        diagnostics_gm: torch.fx.GraphModule | None = None,
    ) -> None:
        self.profile = ProfileData()
        self.profile.load(trace_path)
        self._log_profile_vs_analytical_comparison(diagnostics_gm)

    def _log_profile_vs_analytical_comparison(
        self, diagnostics_gm: torch.fx.GraphModule | None
    ) -> None:
        """Log profile data and PGE vs analytical comparison to trace_structured.

        Logs all profile entries (collectives, ops with durations).
        If diagnostics_gm is provided, walks the graph and compares PGE
        estimates with analytical (roofline / NCCL) for each matched node.
        """
        profile = self.profile
        op_entries = [
            {
                "op": op_name,
                "shapes": [list(s) for s in shapes],
                "strides": [list(s) for s in strides],
                "dtype": str(dtype) if dtype is not None else "",
                "profile_ms": dur_us / 1e3,
            }
            for (op_name, shapes, strides, dtype), dur_us in profile._op_index.items()
        ]

        diagnostics: list[dict[str, Any]] = []
        if diagnostics_gm is not None:
            from torch._inductor.fx_passes.overlap_scheduling import (
                estimate_roofline_runtime_ms,
            )

            for node in diagnostics_gm.graph.nodes:
                pge_est = self(node)
                if pge_est is None:
                    continue
                entry: dict[str, Any] = {
                    "node": node.name,
                    "op": str(node.target),
                    "pge_ms": pge_est,
                }
                if _is_collective_node(node):
                    try:
                        entry["analytical_ms"] = (
                            torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
                                node
                            )
                        )
                    except (RuntimeError, ValueError, TypeError):
                        pass
                else:
                    analytical = estimate_roofline_runtime_ms(node)
                    if analytical is not None and analytical > 0:
                        entry["analytical_ms"] = analytical
                diagnostics.append(entry)

        payload: dict[str, Any] = {
            "collective_count": len(profile.collectives),
            "op_count": profile.op_count,
            "op_entries": op_entries,
        }
        if diagnostics:
            payload["diagnostics"] = diagnostics

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pge_profile_vs_analytical",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(payload),
        )

    def __call__(self, node: fx.Node, override_size: int | None = None) -> float | None:
        if _is_collective_node(node):
            return self._estimate_collective(node, override_size)
        return self._estimate_op(node)

    def _estimate_collective(
        self, node: fx.Node, override_size: int | None
    ) -> float | None:
        info = _get_collective_info(node)
        if info is None:
            return None
        coll_name, pg_ranks, nelems, dtype = info
        val = node.meta.get("val")
        if override_size is not None:
            if override_size == 0:
                return None
            if isinstance(val, torch.Tensor):
                elem_size = val.element_size()
                if elem_size > 0:
                    nelems = override_size // elem_size
        result = self.profile.lookup_collective(coll_name, pg_ranks, nelems, dtype)
        if result is not None:
            return result[0]
        return None

    def _estimate_op(self, node: fx.Node) -> float | None:
        """Estimate any non-collective op via exact shape+stride match in profile."""
        profile_name = _fx_target_to_profile_name(node)
        if profile_name is None:
            return None
        result = _get_node_input_shapes_and_strides(node)
        if result is None:
            return None
        input_shapes, input_strides = result
        dtype = _get_node_dtype(node)
        return self.profile.lookup_op(profile_name, input_shapes, input_strides, dtype)
<|endoftext|># Owner(s): ["oncall: distributed"]
import collections
import inspect
import logging
import math
import operator
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from ..fx_utils import get_fake_args_kwargs
from ..virtualized import V


aten = torch.ops.aten
logger: logging.Logger = logging.getLogger("comm_fusion")


def move_block_after(block: list[fx.Node], target_node: fx.Node) -> None:
    for node in block:
        target_node.append(node)
        target_node = node


def move_block_before(block: list[fx.Node], target_node: fx.Node) -> None:
    for node in block:
        target_node.prepend(node)
        target_node = node


def call_function(
    graph: fx.Graph,
    target: str | Callable[..., Any],
    args: tuple[fx.node.Argument, ...] | None = None,
    kwargs: dict[str, fx.node.Argument] | None = None,
) -> fx.Node:
    # We accept target as a str to avoid typing error as the type of
    # a node.target is str | Callable[..., Any].
    # This also allows us to avoid writing check for every call.
    if isinstance(target, str):
        raise RuntimeError(f"Call function should not get a str target {target=}")
    node = graph.call_function(target, args, kwargs)
    _, args, kwargs = get_fake_args_kwargs(node)
    with V.fake_mode:
        node.meta["val"] = target(*args, **kwargs)
        # node.meta["val"] may be a container. So we use tree_map here
        # to recursively extract the tensor metadata.
        node.meta["tensor_meta"] = tree_map(
            _extract_tensor_metadata, (node.meta["val"],)
        )[0]
    return node


@dataclass(unsafe_hash=True)
class CommBlock:
    shape: torch.Size | list[torch.Size]
    node_list: list[fx.Node]
    inputs: list[fx.Node]
    wait_nodes: list[fx.Node]
    comm_node: fx.Node
    outputs: OrderedSet[fx.Node]


def get_comm_block(comm_node: fx.Node) -> CommBlock | None:
    """
    Given a collective node (e.g., allreduce), find out all the nodes belong to
    this communication.

    Args:
        comm_node(fx.Node): The target communication/collective node.
    Returns:
        The CommBlock that encapsulates the related nodes (e.g., wait_node) of
        the given comm_node.
    """
    node_list = []
    wait_nodes = []
    inputs, _ = tree_flatten((comm_node.args, comm_node.kwargs))
    input_nodes = [inp for inp in inputs if isinstance(inp, fx.Node)]
    # If the users of the wait node are following items, we consider them
    # to be a part of the output.
    intermediate_outputs = ("split", "reshape", "getitem", "detach", "alias")

    first_user = next(iter(comm_node.users))
    if (
        len(comm_node.users) == 1
        and first_user.target is torch.ops._c10d_functional.wait_tensor.default
    ):
        # Collective with only one output
        node_list = [comm_node, first_user]
        wait_nodes.append(first_user)
    elif len(comm_node.users) > 1 and first_user.target is operator.getitem:
        # Collective with only more than one output
        node_list.append(comm_node)
        for user in comm_node.users:
            if user.target != operator.getitem:
                return None
            if len(user.users) != 1:
                return None
            wait_node = next(iter(user.users))
            if wait_node.target != torch.ops._c10d_functional.wait_tensor.default:
                return None
            wait_nodes.append(wait_node)
            node_list.append(user)
        node_list.extend(wait_nodes)
    else:
        return None

    # Identify all the outputs of this collective block.
    outputs = OrderedSet[fx.Node]()
    nodes = collections.deque(wait_nodes)
    while nodes:
        node = nodes.popleft()
        for user in node.users:
            if isinstance(user, fx.Node) and user.name.startswith(intermediate_outputs):
                nodes.append(user)
                node_list.append(user)
            else:
                outputs.add(node)
                break

    tensor_meta = input_nodes[0].meta["tensor_meta"]
    shape: torch.Size | list[torch.Size]
    if isinstance(tensor_meta, TensorMetadata):
        shape = tensor_meta.shape
    elif isinstance(tensor_meta, (list, tuple)):
        shape = [tm.shape for tm in tensor_meta]
    else:
        logger.warning("Unexpected type of tensor_meta %s", type(tensor_meta))
        return None

    return CommBlock(
        shape=shape,
        node_list=node_list,
        wait_nodes=wait_nodes,
        comm_node=comm_node,
        inputs=input_nodes,
        outputs=outputs,
    )


def get_all_comm_blocks(
    graph: fx.Graph,
    comm_ops: tuple[torch._ops.OpOverload, ...],
    comm_filter: Callable[..., bool] | None = None,
) -> list[CommBlock]:
    if comm_filter is None:

        def always_true(comm_block: CommBlock) -> bool:
            return True

        comm_filter = always_true

    blocks = []
    for node in graph.nodes:
        if node.target not in comm_ops:
            continue
        comm_block = get_comm_block(node)
        if comm_block is not None and comm_filter(comm_block):
            blocks.append(comm_block)
    return blocks


def _fuse_allreduce_by_concat(
    graph: fx.Graph,
    last_input_node: fx.Node,
    all_input_nodes: list[fx.Node],
    last_comm_block: CommBlock,
) -> CommBlock:
    """Given a list of inputs in order, create a fused allreduce using concat."""
    # Flatten all the inputs to the all_reduce nodes.
    with graph.inserting_after(last_input_node):
        cat_inputs = []
        for input_node in all_input_nodes:
            if not isinstance(input_node.args[0], fx.Node):
                raise AssertionError(
                    f"expected input_node.args[0] to be an fx.Node, got "
                    f"{type(input_node.args[0])}"
                )
            input_node = input_node.args[0]
            cat_inputs.append(
                call_function(graph, aten.flatten.using_ints, (input_node,))
            )

    # Concat all the flattened nodes.
    with graph.inserting_after(cat_inputs[0]):
        cat_node = call_function(graph, aten.cat, (cat_inputs,))

    # Insert the fused div node and remove the input div nodes.
    # This is an optimization and is not mandatory for fusion.
    divisors = [div.args[1] for div in all_input_nodes]
    if not all(divisor == divisors[0] for divisor in divisors):
        raise AssertionError("expected all divisors to be equal")
    with graph.inserting_after(cat_node):
        div_node = call_function(graph, last_input_node.target, (cat_node, divisors[0]))

    # Create a new Comm/all_reduce node.
    last_comm_node = last_comm_block.comm_node
    last_wait_node = last_comm_block.wait_nodes[0]
    with graph.inserting_after(div_node):
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = div_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = call_function(graph, last_comm_node.target, args, kwargs)

    # Create a new Wait node.
    with graph.inserting_after(fused_comm_node):
        flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
        flatten_args[0] = fused_comm_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_wait_node = call_function(graph, last_wait_node.target, args, kwargs)

    # Move the fused all_reduce and its args to right after the input node
    nodes_to_move = cat_inputs + [cat_node, div_node, fused_comm_node, fused_wait_node]
    # pyrefly: ignore [bad-argument-type]
    move_block_after(nodes_to_move, last_input_node)

    return CommBlock(
        shape=cast(TensorMetadata, cat_node.meta.get("tensor_meta")).shape,
        node_list=[fused_comm_node, fused_wait_node],
        wait_nodes=[fused_wait_node],
        comm_node=fused_comm_node,
        inputs=[div_node],
        outputs=OrderedSet([fused_wait_node]),
    )


def _fuse_with_coalesced_op(
    graph: fx.Graph,
    last_input_node: fx.Node,
    all_input_nodes: list[fx.Node],
    last_comm_block: CommBlock,
) -> CommBlock:
    """Given a list of inputs in order, create a fused allreduce by coalesced."""
    last_comm_node = last_comm_block.comm_node
    last_wait_node = last_comm_block.wait_nodes[0]

    # Insert the fused div node and remove the input div nodes.
    # This is an optimization and is not mandatory for fusion.
    dividends = [div.args[0] for div in all_input_nodes]
    divisors = [div.args[1] for div in all_input_nodes]
    if not all(divisor == divisors[0] for divisor in divisors):
        raise AssertionError("expected all divisors to be equal")
    with graph.inserting_before(last_input_node):
        last_input_node = call_function(
            graph, aten._foreach_div.Scalar, (dividends, divisors[0])
        )
    input_node = last_input_node

    # Create a new Comm/all_reduce_coalesced node.
    with graph.inserting_after(last_comm_node):
        flatten_args, spec = tree_flatten((last_comm_node.args, last_comm_node.kwargs))
        flatten_args[0] = input_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        fused_comm_node = call_function(
            graph, torch.ops._c10d_functional.all_reduce_coalesced.default, args, kwargs
        )

    # Create a new wait node.
    getitem_nodes = []
    wait_nodes = []
    flatten_args, spec = tree_flatten((last_wait_node.args, last_wait_node.kwargs))
    for idx in range(len(all_input_nodes)):
        with graph.inserting_after(fused_comm_node):
            gi_node = call_function(graph, operator.getitem, (fused_comm_node, idx))
        getitem_nodes.append(gi_node)
        flatten_args[0] = gi_node
        args, kwargs = tree_unflatten(flatten_args, spec)
        with graph.inserting_after(gi_node):
            wait_nodes.append(call_function(graph, last_wait_node.target, args, kwargs))

    # Move the new all_reduce_coalesced and its args to right after the input node
    nodes_to_move = [fused_comm_node] + getitem_nodes + wait_nodes
    move_block_after(nodes_to_move, last_input_node)

    return CommBlock(
        shape=[
            tm.shape
            for tm in cast(
                list[TensorMetadata], fused_comm_node.meta.get("tensor_meta")
            )
        ],
        node_list=[fused_comm_node] + getitem_nodes + wait_nodes,
        wait_nodes=wait_nodes,
        comm_node=fused_comm_node,
        inputs=[input_node],
        outputs=OrderedSet(wait_nodes),
    )


def _scatter_fused_allreduce_waits(
    graph: fx.Graph,
    fused_comm_block: CommBlock,
    orig_comm_blocks: list[CommBlock],
    node_indices: dict[fx.Node, int],
    split_and_reshape: bool = True,
) -> None:
    """
    Scatters the result of the fused communication node to the original users.
    If the fused method is concat splitting the output and reshape will be inserted,
    before inserting getitem. Otherwise getitem will be used as the users of the
    wait node.
    """

    # Before we mass up the order, we need to get the index of the last wait node
    # in orig_comm_blocks. This index will be later used to determine what users
    # nodes need to be move to maintain a correct topological sort order.
    last_wait_node_idx = 0

    for node in graph.nodes:
        last_wait_node_idx = max(
            node_indices.get(node, last_wait_node_idx), last_wait_node_idx
        )
        if node == orig_comm_blocks[-1].wait_nodes[0]:
            break

    if split_and_reshape:
        fused_wait_node = fused_comm_block.wait_nodes[0]
        with graph.inserting_after(fused_wait_node):
            split_node = call_function(
                graph,
                aten.split,
                (
                    fused_wait_node,
                    [math.prod(cast(list[int], cb.shape)) for cb in orig_comm_blocks],
                ),
            )
        with graph.inserting_after(split_node):
            fused_outputs = []
            for idx, comm_block in enumerate(orig_comm_blocks):
                split_idx_node = call_function(
                    graph, operator.getitem, (split_node, idx)
                )
                with graph.inserting_after(split_idx_node):
                    fused_outputs.append(
                        call_function(
                            graph, aten.reshape, (split_idx_node, comm_block.shape)
                        )
                    )
    else:
        fused_outputs = fused_comm_block.wait_nodes

    # Scatter the fused outputs.
    incorrect_order_nodes = []
    for comm_block, fused_output in zip(orig_comm_blocks, fused_outputs):
        # Some descendant users of the orig_comm_blocks may be scheduled before
        # the fused all_reduce. For example, the user nodes of the very first
        # all_reduce may be scheduled before the second all_reduce. Since the
        # fused all_reduce is inserted right after the last all_reduce, the
        # order can be wrong.
        # `incorrect_order_nodes` records these nodes.

        orig_wait = comm_block.wait_nodes[0]
        nodes = collections.deque(list(orig_wait.users))
        while nodes:
            user_node = nodes.popleft()
            if not isinstance(user_node, fx.Node):
                continue

            if node_indices[user_node] < last_wait_node_idx:
                incorrect_order_nodes.append(user_node)
                nodes.extend(list(user_node.users))

        orig_wait.replace_all_uses_with(fused_output)

    last_fused_result = fused_outputs[0]
    fused_outputs_set = OrderedSet(fused_outputs)
    for node in graph.nodes:
        if node in fused_outputs_set:
            last_fused_result = node

    # Move the incorrect_order_nodes to right after the last fused_result.
    incorrect_order_nodes = sorted(
        incorrect_order_nodes, key=lambda node: node_indices[node]
    )
    move_block_after(incorrect_order_nodes, last_fused_result)


def _fuse_allreduce(
    graph: fx.Graph,
    comm_blocks: list[CommBlock],
    node_indices: dict[fx.Node, int],
    use_concat: bool,
) -> CommBlock:
    """Given a list of allreduce CommBlock, fuse the CommBlocks into one CommBlock."""

    if len(comm_blocks) == 1:
        return comm_blocks[0]

    # Find the last input node of all the CommBlocks. This node will be served
    # as the inserting point of the new collective op.
    last_input_node = comm_blocks[0].inputs[0]
    last_input_index = -1
    all_input_nodes = []
    for comm_block in comm_blocks:
        input_node = comm_block.inputs[0]
        all_input_nodes.append(input_node)
        index = node_indices[input_node]
        if index >= last_input_index:
            if index == last_input_index:
                raise AssertionError(f"expected index != last_input_index, got {index}")
            last_input_node = input_node
            last_input_index = index

    if use_concat:
        fused_comm_block = _fuse_allreduce_by_concat(
            graph, last_input_node, all_input_nodes, comm_blocks[-1]
        )
    else:
        fused_comm_block = _fuse_with_coalesced_op(
            graph, last_input_node, all_input_nodes, comm_blocks[-1]
        )

    _scatter_fused_allreduce_waits(
        graph, fused_comm_block, comm_blocks, node_indices, split_and_reshape=use_concat
    )

    for comm_block in comm_blocks:
        for wait in comm_block.wait_nodes:
            graph.erase_node(wait)
        graph.erase_node(comm_block.comm_node)
    graph.eliminate_dead_code()

    return fused_comm_block


def _bucket_size_fusion(
    graph: fx.Graph, comm_blocks: list[CommBlock], bucket_size_mb: int
) -> Generator[list[CommBlock], None, None]:
    MB = 1024**2
    bucket_size = 1 * MB
    bucket_cap_size = bucket_size_mb * MB
    curr_size = 0
    curr_blocks = []

    count = 0
    fuse_count = 0
    for i, block in enumerate(comm_blocks):
        curr_blocks.append(block)
        itemsize = block.comm_node.meta["tensor_meta"].dtype.itemsize
        curr_size += cast(torch.Size, block.shape).numel() * itemsize
        count += 1
        if curr_size < bucket_size and i != len(comm_blocks) - 1:
            continue

        fuse_count += 1
        if torch.distributed.get_rank() == 0:
            logger.info(
                "DDP bucketing: block%d, count=%d, curr_size=%d, bucket_size=%d",
                fuse_count,
                count,
                curr_size,
                bucket_size,
            )

        # Set the debug counters
        counters["inductor"]["ddp_buckets"] = fuse_count
        yield curr_blocks

        bucket_size = bucket_cap_size
        curr_blocks = []
        curr_size = 0
        count = 0


def _fuse_ddp_communication(
    graph: fx.Graph, algorithm_fn: Callable[..., Any], fusion_fn: Callable[..., Any]
) -> None:
    for output in reversed(graph.nodes):
        if output.op == "output":
            break

    def ddp_reducer_filter(block: CommBlock) -> bool:
        if (
            not isinstance(block.comm_node.args[0], fx.Node)
            or block.comm_node.args[0].target != aten.div.Tensor
        ):
            return False

        if len(block.wait_nodes[0].users) != 1:
            # gradient/wait node should only be used by one user
            return False

        # Two cases:
        # 1. gradient/wait node should be directly used by the output
        # if gradient is None before bwd.
        # 2. gradient/wait node should be directly used by copy_.
        if (
            output not in block.wait_nodes[0].users
            and next(iter(block.wait_nodes[0].users)).target != aten.copy_.default
        ):
            return False

        return True

    ops = (
        torch.ops._c10d_functional.all_reduce_.default,
        torch.ops._c10d_functional.all_reduce.default,
    )
    comm_blocks = get_all_comm_blocks(graph, ops, comm_filter=ddp_reducer_filter)
    node_indices = {node: i for i, node in enumerate(graph.nodes)}

    for block in algorithm_fn(graph, comm_blocks):
        fusion_fn(graph, block, node_indices)


def fuse_ddp_with_coalesced_op(graph: fx.Graph, bucket_size_mb: int) -> None:
    _fuse_ddp_communication(
        graph,
        partial(_bucket_size_fusion, bucket_size_mb=bucket_size_mb),
        partial(_fuse_allreduce, use_concat=False),
    )


def fuse_ddp_with_concat_op(graph: fx.Graph, bucket_size_mb: int) -> None:
    _fuse_ddp_communication(
        graph,
        partial(_bucket_size_fusion, bucket_size_mb=bucket_size_mb),
        partial(_fuse_allreduce, use_concat=True),
    )


def schedule_comm_wait(graph: fx.Graph) -> None:
    """
    Delay the execution of wait tensors of allreduce until its first user.

    This algorithm considers the intermediate users, like split, getitem,
    of the wait node and schedule those intermediate users as well.
    This will result in a better overlapping result.
    """
    ops = (
        torch.ops._c10d_functional.all_reduce_.default,
        torch.ops._c10d_functional.all_reduce.default,
        torch.ops._c10d_functional.all_reduce_coalesced.default,
        torch.ops._c10d_functional.all_reduce_coalesced_.default,
    )
    comm_blocks = get_all_comm_blocks(graph, ops)
    if not comm_blocks:
        return

    # Find all the end users.
    allreduce_users = OrderedSet[fx.Node]()
    for allreduce in comm_blocks:
        for output in allreduce.outputs:
            allreduce_users.update(output.users)

    node_indices = {node: i for i, node in enumerate(graph.nodes)}
    for allreduce in comm_blocks:
        # Find the earliest/first user -- target_node.
        if len(allreduce.outputs) < 1:
            raise AssertionError(
                f"Found an allreduce that has zero outputs/users -- {allreduce}."
            )
        # Initialize the target node to avoid typing issues.
        target_node = next(iter(next(iter(allreduce.outputs)).users))
        target_node_index = 2**31
        for user in (user for output in allreduce.outputs for user in output.users):
            index = node_indices[user]
            if index < target_node_index:
                target_node = user
                target_node_index = index

        # Move wait nodes and all the subsequent nodes in the comm_block to
        # before the first user -- target_node.
        wait_idx = -1
        for wait_idx, node in enumerate(allreduce.node_list):
            if node == allreduce.wait_nodes[0]:
                break
        if wait_idx < 0:
            raise AssertionError(f"expected wait_idx >= 0, got {wait_idx}")
        move_block_before(allreduce.node_list[wait_idx:], target_node)


def fuse_ddp_communication(
    graph: fx.Graph, passes: list[Callable[..., None] | str], bucket_size_mb: int
) -> None:
    for i, pa in enumerate(passes):
        with GraphTransformObserver(
            graph.owning_module,  # pyrefly: ignore[bad-argument-type]
            f"fuse_ddp_communication_pass_{i}",
        ):
            if isinstance(pa, str):
                func = globals()[pa]
            else:
                func = pa
            if "bucket_size_mb" in OrderedSet(
                v.name for v in inspect.signature(func).parameters.values()
            ):
                func(graph, bucket_size_mb=bucket_size_mb)
            else:
                func(graph)
<|endoftext|>import functools
import heapq
import itertools
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.fx as fx
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor import config
from torch._inductor.comm_analysis import estimate_fx_collective_memory_footprint
from torch._inductor.fx_passes.bucketing import (
    _default_bucket_mode,
    _get_collective_node_from_wait,
    _resolve_group_name,
    _schedulable_wait_node,
    bucket_key,
    BucketMode,
    get_full_bucket_key,
    is_wait_tensor,
)
from torch._inductor.fx_passes.memory_estimator import MemoryTracker
from torch._inductor.fx_passes.utils import BitsetAncestors
from torch._logging import trace_structured
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import _disable_current_modes


log = logging.getLogger(__name__)

from torch._inductor.pattern_matcher import stable_topological_sort


def make_all_device_put_sync(gm: torch.fx.GraphModule) -> int:
    """
    Convert all non_blocking=True device_put operations to non_blocking=False.

    Only performs the conversion if at least one non_blocking=True device_put
    exists in the graph.

    Returns:
        The number of device_put operations converted to sync.
    """
    g = gm.graph
    device_put_nodes = list(
        g.find_nodes(op="call_function", target=torch.ops.prims.device_put.default)
    )

    # Check if any non_blocking=True device_put exists
    has_async_device_put = False
    for n in device_put_nodes:
        opt_args_kwargs = normalize_function(
            n.target,
            args=n.args,
            kwargs=n.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        if opt_args_kwargs is not None:
            _, kwargs = opt_args_kwargs
            if kwargs.get("non_blocking", False):
                has_async_device_put = True
                break

    if not has_async_device_put:
        return 0

    # Convert all non_blocking=True to non_blocking=False
    count = 0
    for n in device_put_nodes:
        opt_args_kwargs = normalize_function(
            n.target,
            args=n.args,
            kwargs=n.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        if opt_args_kwargs is not None:
            _, kwargs = opt_args_kwargs
            if kwargs.get("non_blocking", False):
                kwargs["non_blocking"] = False
                n.args = n.args[0], kwargs["device"], kwargs["non_blocking"]
                n.kwargs = {}
                count += 1

    return count


@dataclass
class WhyNoOverlap:
    """Track reasons why a collective cannot overlap with compute."""

    compute_name: str
    collective_name: str

    def __init__(self, compute_node: fx.Node, collective_node: fx.Node) -> None:
        self.compute_name = compute_node.name
        self.collective_name = collective_node.name

    def __call__(self, reason: str, *args: Any) -> None:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "cannot overlap %s with %s: " + reason,  # noqa: G003
                self.collective_name,
                self.compute_name,
                *args,
            )


@functools.cache
def get_group_name(n: fx.Node) -> str:
    """Extract the group name from a collective operation node."""
    opt_args_kwargs = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt_args_kwargs is None:
        raise AssertionError("failed to normalize args/kwargs for node")
    _, kwargs = opt_args_kwargs
    return _resolve_group_name(kwargs["group_name"])


def get_custom_estimation(
    n: fx.Node,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
    override_size: int | None = None,
) -> float | None:
    if custom_runtime_estimation is None:
        return None

    return custom_runtime_estimation(n, override_size)


def estimate_collective_time(
    n: fx.Node,
    override_size: int | None = None,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
    collective_estimator: Literal["analytical", "benchmark"] = "analytical",
) -> float:
    """Estimate the runtime of a collective operation, optionally with an overridden size."""
    if (
        est := get_custom_estimation(n, custom_runtime_estimation, override_size)
    ) is not None:
        return est

    if collective_estimator == "benchmark":
        from torch._inductor.fx_passes.node_runtime_estimation import (
            benchmark_collective_with_cuda_events,
        )

        cuda_val, _ = benchmark_collective_with_cuda_events(n, nruns=5)
        if cuda_val is not None:
            return cuda_val

    # Analytical model (also fallback when benchmark returns None)
    return torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
        n, override_size
    )


def _get_flop_registry_key(target: Any) -> Any | None:
    """Return the flop_registry key for a node target, or None if not registered."""
    key = getattr(target, "overloadpacket", None)
    if key in torch.utils.flop_counter.flop_registry:
        return key
    if target in torch.utils.flop_counter.flop_registry:
        return target
    return None


def is_compute_node(n: fx.Node) -> bool:
    """
    Should we consider this node computationally expensive ?
    Currently uses flop registration, but we could expand more generally.
    """
    return _get_flop_registry_key(n.target) is not None


def estimate_roofline_runtime_ms(node: fx.Node) -> float:
    """Estimate runtime using roofline model (max of compute and memory bound).

    Uses FLOPs for compute-bound estimate if op is in flop_registry,
    and memory bandwidth for memory-bound estimate.
    Returns 0 for view nodes (no cost).
    """
    from torch._inductor.fx_passes.fusion_regions import is_view_node
    from torch.utils._pytree import tree_flatten, tree_map
    from torch.utils._runtime_estimation import get_compute_time, get_transfer_time

    if is_view_node(node):
        return 0.0

    def _get_val(n: Any) -> Any:
        if isinstance(n, fx.Node):
            return n.meta.get("val")
        return n

    args = tree_map(_get_val, node.args)
    kwargs = tree_map(_get_val, node.kwargs)
    out = _get_val(node)

    if out is None:
        return 0.0

    flat_args_kwargs, _ = tree_flatten((args, kwargs))
    flat_outs, _ = tree_flatten(out)
    out_dtypes = OrderedSet([t.dtype for t in flat_outs if isinstance(t, torch.Tensor)])

    # Compute time (FLOPs-based, only if op is in flop_registry)
    compute_ns: float = 0.0
    flop_key = _get_flop_registry_key(node.target)
    if flop_key is not None and len(out_dtypes) >= 1:
        # Use a single dtype for peak-flops lookup. For mixed-dtype outputs
        # (e.g. flex_attention returns bf16 out + fp32 logsumexp), use the
        # first output tensor's dtype as the compute dtype.
        compute_dtypes = (
            OrderedSet(out_dtypes)
            if len(out_dtypes) == 1
            else OrderedSet(
                [next(t.dtype for t in flat_outs if isinstance(t, torch.Tensor))]
            )
        )
        compute_ns = get_compute_time(
            flop_key, args, kwargs, out, compute_dtypes, node_meta=node.meta
        )
        if isinstance(compute_ns, (torch.SymInt, torch.SymFloat)):
            compute_ns = hint if (hint := compute_ns.hint) is not None else 0.0

    # Transfer time (memory bandwidth-based, uses size_hint internally)
    transfer_ns = get_transfer_time(flat_args_kwargs, flat_outs)

    # Roofline: max of compute and transfer, convert ns to ms
    return max(float(compute_ns), float(transfer_ns)) / 1e6


def get_hint(x: int | torch.SymInt) -> int | None:
    if isinstance(x, int):
        return x
    return x.hint


def get_collective_do_bench() -> Callable[[Callable[[], Any]], float]:
    with dynamo_timed("collective_compute_do_bench"):
        return functools.partial(
            # pyrefly: ignore [bad-argument-type]
            torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
            warmup=5,
        )


def benchmark_node_with_cache_key(
    n: fx.Node,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
) -> tuple[float, str | None]:
    """Benchmark a compute node and return (runtime, cache_key)."""
    if not is_compute_node(n):
        raise AssertionError(f"expected a compute node, got {n}")

    # HOPs can't be benchmarked standalone (args include subgraphs) — use analytical
    from torch._ops import HigherOrderOperator

    if isinstance(n.target, HigherOrderOperator):
        return estimate_roofline_runtime_ms(n), None

    from torch._dynamo.testing import rand_strided

    # todo - skip unbacked, symbolic
    success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(n)

    if not success:
        return 0, None

    unbacked_tensor = False

    key = f"{str(n.target)}: "

    def to_real(t: torch.Tensor) -> torch.Tensor | None:
        shape = [get_hint(dim) for dim in t.shape]
        stride = [get_hint(s) for s in t.stride()]

        if any(s is None for s in itertools.chain(shape, stride)):
            nonlocal unbacked_tensor
            unbacked_tensor = True
            return None

        nonlocal key
        key += f"T: {shape, stride, t.dtype} "
        return rand_strided(shape, stride, device=t.device, dtype=t.dtype)  # type: ignore[arg-type]

    with _disable_current_modes():
        args, kwargs = torch.utils._pytree.tree_map_only(
            torch.Tensor,
            lambda t: to_real(t),
            (args, kwargs),
        )

        args, kwargs = torch.utils._pytree.tree_map_only(
            torch.SymInt,
            lambda s: optimization_hint(s, fallback=config.unbacked_symint_fallback),
            (args, kwargs),
        )

        if val := get_cached_node_time(key):
            return val, key

        if unbacked_tensor:
            return 0, key

        if (
            est := get_custom_estimation(n, custom_runtime_estimation, None)
        ) is not None:
            set_cached_node_time(key, est)
            return est, key

        bench = get_collective_do_bench()
        out = bench(lambda: n.target(*args, **kwargs))  # type: ignore[operator]
        set_cached_node_time(key, out)
        return out, key


def benchmark_node(
    n: fx.Node,
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
) -> float:
    return benchmark_node_with_cache_key(n, custom_runtime_estimation)[0]


@functools.cache
def get_benchmark_cache() -> torch._inductor.codecache.LocalCache:
    return torch._inductor.codecache.LocalCache()


def get_cached_node_time(key: str) -> float:
    return get_benchmark_cache().lookup(key)  # type: ignore[return-value]


def set_cached_node_time(key: str, value: float) -> None:
    return get_benchmark_cache().set_value(key, value=value)


@dataclass
class CollectiveInfo:
    """Track info about a collective operation"""

    start_node: fx.Node
    wait_node: fx.Node
    size_bytes: int
    estimated_time_ms: float
    exposed_time_ms: float  # How much of this collective is still exposed
    hiding_nodes: OrderedSet[fx.Node] = field(default_factory=OrderedSet)

    @property
    def is_exposed(self) -> bool:
        return self.exposed_time_ms != 0


@dataclass
class CollBucket:
    """Track information about a bucket of collectives."""

    collectives: list[fx.Node]  # Original collective starts
    bucketed_start: fx.Node | None = None  # After bucketing
    bucketed_wait: fx.Node | None = None  # After bucketing
    total_bytes: int = 0


def gb_to_bytes(gb: float) -> int:
    """Convert gigabytes to bytes."""
    return int(gb * 1024 * 1024 * 1024)


class OverlapScheduler:
    """
    Scheduler that reorders operations to maximize compute-collective overlap.

    The reordering is done as a scheduling pass. We maintain a priority queue of
    schedulable nodes. The nodes are ranked by:

    1) the compute node index they dominate. this allows reordering locally, such as with
    parallel mms, and also allows overlapping reduce scatter nodes outputs in the backward
    with compute by deferring their waits.

    2) whether the current node is a collective or wait that is currently exposed but has a compute
    node which it could be overlapped with.

    3) original order in the graph for stability.

    When we schedule compute nodes, we first overlap exposed in-flight collectives, then look for unscheduled
    collectives that can be scheduled concurrently.

    TODO:
        - experiment with other priority scores / allow other mechanisms of reorder / more strict adherence to original graph
        - memory limit for deferred scheduling of reduce_scatter nodes.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        max_in_flight_gb: float,
        max_compute_pre_fetch: int,
        collective_bucketing: bool,
        insert_overlap_deps: bool,
        compute_overlap_multipler: float,
        max_coll_distance: int,
        custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
        | None = None,
        collective_estimator: Literal["analytical", "benchmark"] = "analytical",
        compute_estimator: Literal["analytical", "benchmark"] = "benchmark",
        max_memory_increase_gb: float | None = 1.0,
        max_memory_increase_ratio: float | None = 0.05,
        log_final_collectives_estimations: bool = False,
        bucket_exposed_first: bool | None = None,
        enable_fusion_regions: bool = False,
        bucket_only_internode_comms: bool = False,
        bucket_mode: BucketMode | None = None,
        max_off_bucket_gb: float | None = 0.5,
        prioritize_bucketing_during_scheduling: bool = True,
        pge_profile_path: str | None = None,
    ):
        self.gm = gm
        self.graph = gm.graph
        self.compute_overlap_multipler = compute_overlap_multipler
        self.max_node_distance = max_coll_distance
        self.max_in_flight_bytes: int = gb_to_bytes(max_in_flight_gb)

        # Profile-guided estimation: create estimator from profile path
        if pge_profile_path and custom_runtime_estimation is None:
            from torch._inductor.fx_passes.profile_guided_estimation import (
                ProfileGuidedEstimator,
            )

            custom_runtime_estimation = ProfileGuidedEstimator(
                pge_profile_path, diagnostics_gm=gm
            )

        self.custom_runtime_estimation = custom_runtime_estimation
        self.collective_bucketing = collective_bucketing
        self.insert_overlap_deps = insert_overlap_deps
        self.max_compute_pre_fetch = max_compute_pre_fetch
        # In deterministic mode, force analytical estimation to avoid GPU sync
        if config.deterministic:
            self.collective_estimator = "analytical"
            self.compute_estimator = "analytical"
        else:
            self.collective_estimator = collective_estimator
            self.compute_estimator = compute_estimator
        self.log_final_collectives_estimations = log_final_collectives_estimations
        self.bucket_exposed_first = bucket_exposed_first
        self.bucket_only_internode_comms = bucket_only_internode_comms
        self.bucket_mode = bucket_mode or _default_bucket_mode()
        self.max_off_bucket_bytes: int | None = (
            gb_to_bytes(max_off_bucket_gb) if max_off_bucket_gb is not None else None
        )
        self.prioritize_bucketing_during_scheduling = (
            prioritize_bucketing_during_scheduling
        )

        # Make all to(device) non_blocking=False,
        # They can be implicitly depending by user logic on other to(device) non_blocking=True.
        # OverlapScheduler can put reads of non_blocking device_put before blocking one.
        # This results in dirty reads.
        num_device_put_converted = make_all_device_put_sync(gm)
        if num_device_put_converted > 0:
            log.warning(
                "overlap_scheduling converted %d device_put operations from "
                "non_blocking=True to non_blocking=False. This may affect performance.",
                num_device_put_converted,
            )

        # Compute initial node runtime estimates. Compute nodes use roofline model
        # here; the alignment step in run() replaces them with benchmarked +
        # cross-rank-aligned values.
        self.node_estimations = gather_node_runtime_estimations(
            gm,
            custom_runtime_estimation,
            enable_fusion_regions=enable_fusion_regions,
            log_estimations=True,
        )
        # Build structures
        stable_topological_sort(self.graph)
        self.nodes = list(self.graph.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self._parent_lists: list[list[fx.Node]] = [
            list(n._input_nodes) for n in self.nodes
        ]
        self.node_ancestors: BitsetAncestors = self._collect_node_ancestors()

        # Identify collectives and compute nodes
        self.collective_info: dict[fx.Node, CollectiveInfo] = {}
        self.unscheduled_collectives: OrderedSet[fx.Node] = OrderedSet()

        # Identify compute nodes early (needed for baseline memory computation)
        self.compute_nodes = [n for n in self.nodes if is_compute_node(n)]
        self.current_compute_index = 0

        self.original_mem_before_compute_index: list[int] = []
        needs_memory_tracking = (
            max_memory_increase_gb is not None or max_memory_increase_ratio is not None
        )
        if needs_memory_tracking:
            self.original_peak_memory = self._compute_baseline_memory()
            memory_increase_bytes = None
            if max_memory_increase_gb is not None:
                memory_increase_bytes = gb_to_bytes(max_memory_increase_gb)
            if max_memory_increase_ratio is not None:
                ratio_increase = int(
                    self.original_peak_memory * max_memory_increase_ratio
                )
                memory_increase_bytes = (
                    max(memory_increase_bytes, ratio_increase)
                    if memory_increase_bytes is not None
                    else ratio_increase
                )
            self.allowed_peak_memory_bytes = self.original_peak_memory + (
                memory_increase_bytes or 0
            )
            self.memory_tracker = MemoryTracker(self.graph)
        else:
            self.original_peak_memory = 0
            self.allowed_peak_memory_bytes = sys.maxsize
            self.memory_tracker = None  # type: ignore[assignment]

        self.cumulative_prefetch_mem_by_compute_index: list[int] = [
            0 for _ in range(len(self.compute_nodes))
        ]

        self.wait_to_start: dict[fx.Node, fx.Node] = {}
        self._identify_collectives()
        self.wasted_compute = 0.0

        # Bitset masks for O(N/64) pre-filtering in _find_schedulable_path.
        #
        # _scheduled_bits: bit i is set when node i has been scheduled.
        #   Updated incrementally in _schedule() via: _scheduled_bits |= 1 << idx
        #
        # _compute_bits: bit i is set for every compute node (fixed at init).
        #   Used to detect "is any unscheduled ancestor a compute node?" in O(N/64)
        #   instead of doing a full BFS + per-node is_compute_node() check.
        self._scheduled_bits: int = 0
        self._compute_bits: int = 0
        for n in self.compute_nodes:
            self._compute_bits |= self.node_ancestors.node_bit(n)

        # Calculate domination indices for both compute and reduce_scatter nodes
        self.reduce_scatter_nodes = self.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
        )
        self.compute_index_domination = self._calculate_domination_index(
            self.compute_nodes
        )
        self.reduce_scatter_domination = self._calculate_domination_index(
            self.reduce_scatter_nodes
        )

        self.in_degree = Counter(user for node in self.nodes for user in node.users)

        # Two separate queues: on-path (domination-based) and off-path (node_idx-based)
        self.on_path_ready: list[tuple[object, fx.Node]] = []
        self.off_path_ready: list[tuple[object, fx.Node]] = []
        # Track potential bucket sizes for off-path collectives (for batch scheduling)
        self.off_path_ready_potential_buckets: dict[object, int] = defaultdict(int)

        for node in self.nodes:
            if self.in_degree[node] == 0:
                self._add_to_ready_queue(node)

        self.in_flight: dict[fx.Node, CollectiveInfo] = {}  # start -> info
        self.in_flight_bytes = 0
        self.scheduled: OrderedSet[fx.Node] = OrderedSet()
        self.max_compute_pre_fetch = max_compute_pre_fetch

        self.last_on_path_node_idx = -1

    def _add_to_ready_queue(self, node: fx.Node) -> None:
        if self.off_compute_path(node):
            score = self._compute_off_path_score(node)
            heapq.heappush(self.off_path_ready, (score, node))
            # Track potential bucket sizes for off-path ready collectives
            if node in self.collective_info:
                bucket_key = get_full_bucket_key(node, self.bucket_mode)
                node_bytes = self.collective_info[node].size_bytes
                self.off_path_ready_potential_buckets[bucket_key] += node_bytes
        else:
            score = self._compute_on_path_score(node)
            heapq.heappush(self.on_path_ready, (score, node))

    def _collect_node_ancestors(self) -> BitsetAncestors:
        """Collect all ancestors for each node using int-bitset representation."""
        return BitsetAncestors(self.nodes)

    def _compute_baseline_memory(self) -> int:
        """
        Simulate the original schedule to compute baseline memory profile.
        Returns the peak memory observed during simulation.
        """
        baseline_tracker = MemoryTracker(self.graph)

        last_compute_max_memory = 0
        peak_memory = 0

        for node in self.nodes:
            baseline_tracker.schedule_node(node)
            current_mem = baseline_tracker.current_memory_bytes

            # Record the max memory between this and previous compute node
            last_compute_max_memory = max(last_compute_max_memory, current_mem)

            if is_compute_node(node):
                self.original_mem_before_compute_index.append(last_compute_max_memory)
                last_compute_max_memory = current_mem

            peak_memory = max(peak_memory, current_mem)

        return peak_memory

    def _prefetch_would_exceed_memory_budget(self, start_node: fx.Node) -> bool:
        """
        Check if prefetching this collective would exceed memory budget at ANY compute node
        between now and when it's used.
        """
        info = self.collective_info[start_node]
        size = info.size_bytes

        domination_index = self.compute_index_domination[start_node]

        # If off-path, assume it doesn't increase memory
        if domination_index == sys.maxsize:
            return False

        # check current mem
        if (
            self.memory_tracker.current_memory_bytes + size
            > self.allowed_peak_memory_bytes
        ):
            return True

        start_index = self.current_compute_index

        # then, check future mem
        for compute_idx in range(start_index, domination_index):
            cumulative_prefetch = self.cumulative_prefetch_mem_by_compute_index[
                compute_idx
            ]

            # Check 1: Would cumulative prefetch exceed in-flight limit?
            if (cumulative_prefetch + size) > self.max_in_flight_bytes:
                return True

            # Check 2: Would total memory (baseline + cumulative prefetch) exceed budget?
            baseline_mem = self.original_mem_before_compute_index[compute_idx]
            projected = baseline_mem + cumulative_prefetch + size

            if projected > self.allowed_peak_memory_bytes:
                return True

        return False

    def _update_cumulative_prefetch_memory(
        self, collective: fx.Node, info: CollectiveInfo
    ) -> None:
        """
        Update cumulative prefetch memory for all compute indices this collective will be live.
        """
        domination_index = self.compute_index_domination[collective]
        if domination_index == sys.maxsize:
            return

        for compute_idx in range(self.current_compute_index, domination_index):
            self.cumulative_prefetch_mem_by_compute_index[compute_idx] += (
                info.size_bytes
            )

    def off_compute_path(self, n: fx.Node) -> bool:
        """Check if a node is off the compute path (doesn't block any compute)."""
        return self.compute_index_domination[n] == sys.maxsize

    def dominates_reduce_scatter(self, n: fx.Node) -> bool:
        """Check if a node dominates (blocks) any reduce_scatter."""
        return self.reduce_scatter_domination[n] != sys.maxsize

    def _identify_collectives(self) -> None:
        """Identify all collective operations and process groups."""
        self.all_pgs: OrderedSet[str] = OrderedSet()

        for node in self.nodes:
            if _schedulable_wait_node(node):
                start = _get_collective_node_from_wait(node)
                if start is None:
                    raise AssertionError("wait node has no associated collective")
                if start not in self.node_estimations:
                    raise AssertionError(
                        f"Missing estimation for collective {start.name}. "
                        f"Ensure custom_runtime_estimation returns a value for this node."
                    )
                self.wait_to_start[node] = start
                # For coalesced collectives, multiple waits share the same
                # start node. Only register the first wait as the representative.
                if start in self.collective_info:
                    continue
                coll_time_ms = self.node_estimations[start]

                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
