            conf.set_module_name_regex(module_name_regex, qconfig)
        for module_name, qconfig in qconfig_dict.get(_MODULE_NAME_DICT_KEY, []):
            conf.set_module_name(module_name, qconfig)
        for module_name, object_type, index, qconfig in qconfig_dict.get(
            _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY, []
        ):
            conf.set_module_name_object_type_order(
                module_name, object_type, index, qconfig
            )
        return conf
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module


__all__ = [
    "script_qconfig",
    "script_qconfig_dict",
    "fuse_conv_bn_jit",
    "prepare_jit",
    "prepare_dynamic_jit",
    "convert_jit",
    "convert_dynamic_jit",
    "quantize_jit",
    "quantize_dynamic_jit",
]


def _check_is_script_module(model):
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError("input must be a script module, got: " + str(type(model)))


def _check_forward_method(model):
    if not model._c._has_method("forward"):
        raise ValueError("input script module does not have forward method")


def script_qconfig(qconfig):
    r"""Instantiate the activation and weight observer modules and script
    them, these observer module instances will be deepcopied during
    prepare_jit step.
    """
    return QConfig(
        activation=torch.jit.script(qconfig.activation())._c,
        weight=torch.jit.script(qconfig.weight())._c,
    )


def script_qconfig_dict(qconfig_dict):
    r"""Helper function used by `prepare_jit`.
    Apply `script_qconfig` for all entries in `qconfig_dict` that is
    not None.
    """
    return {k: script_qconfig(v) if v else None for k, v in qconfig_dict.items()}


def fuse_conv_bn_jit(model, inplace=False):
    r"""Fuse conv - bn module
    Works for eval model only.

    Args:
        model: TorchScript model from scripting or tracing
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.fuse_conv_bn_jit")
    model_c = model._c
    model_c = torch._C._jit_pass_fold_convbn(model_c)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model


def _prepare_jit(model, qconfig_dict, inplace=False, quant_type=QuantType.STATIC):
    _check_is_script_module(model)
    _check_forward_method(model)
    if not all(isinstance(x, str) for x in qconfig_dict):
        raise ValueError("qconfig_dict should only contain names(str) as keys.")
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observers(
        model._c, "forward", scripted_qconfig_dict, inplace, quant_type
    )
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model


def _prepare_ondevice_jit(
    model,
    qconfig_dict,
    method_name="forward",
    inplace=False,
    quant_type=QuantType.STATIC,
):
    _check_is_script_module(model)
    if not all(isinstance(x, str) for x in qconfig_dict):
        raise ValueError("qconfig_dict should only contain names(str) as keys.")
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    method_graph = model._c._get_method(method_name).graph
    torch._C._jit_pass_inline(method_graph)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observer_method_for_ondevice_ptq(
        model._c, method_name, scripted_qconfig_dict, inplace, quant_type
    )
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model


def prepare_jit(model, qconfig_dict, inplace=False):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.prepare_jit")
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.STATIC)


def prepare_dynamic_jit(model, qconfig_dict, inplace=False):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.prepare_dynamic_jit")
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.DYNAMIC)


def _prepare_ondevice_dynamic_jit(
    model, qconfig_dict, method_name="forward", inplace=False
):
    return _prepare_ondevice_jit(
        model, qconfig_dict, method_name, inplace, quant_type=QuantType.DYNAMIC
    )


def _convert_jit(
    model, inplace=False, debug=False, quant_type=QuantType.STATIC, preserved_attrs=None
):
    _check_is_script_module(model)
    model.eval()
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant(
        model_c, "forward", inplace, debug, quant_type
    )
    if not debug:
        is_xpu = all(p.device.type == "xpu" for p in model.parameters())
        if not is_xpu:
            # Moving model parameters to CPU since quantized operators
            # are only supported on CPU and XPU right now
            model.cpu()
        if preserved_attrs is None:
            preserved_attrs = []
        model_c = torch._C._jit_pass_quant_finalize(
            model_c, quant_type, preserved_attrs
        )
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model


def _convert_ondevice_jit(
    model, method_name, inplace=False, debug=False, quant_type=QuantType.STATIC
):
    _check_is_script_module(model)
    if quant_type != QuantType.DYNAMIC:
        raise AssertionError(
            "This API, while should work for static quant, is only tested for dynamic quant."
        )
    if method_name.startswith("observe_"):
        raise AssertionError("Pass in valid method to be quantized, e.g. forward")
    observe_method_name = "observe_" + method_name
    quantize_method_name = "quantize_" + method_name
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant_for_ondevice_ptq(
        model._c, observe_method_name, inplace, debug, QuantType.DYNAMIC
    )
    model_c = torch._C._jit_pass_quant_finalize_for_ondevice_ptq(
        model_c, QuantType.DYNAMIC, quantize_method_name
    )
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model


def convert_jit(model, inplace=False, debug=False, preserved_attrs=None):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.convert_jit")
    return _convert_jit(
        model,
        inplace,
        debug,
        quant_type=QuantType.STATIC,
        preserved_attrs=preserved_attrs,
    )


def convert_dynamic_jit(model, inplace=False, debug=False, preserved_attrs=None):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.convert_dynamic_jit")
    return _convert_jit(
        model,
        inplace,
        debug,
        quant_type=QuantType.DYNAMIC,
        preserved_attrs=preserved_attrs,
    )


def _convert_ondevice_dynamic_jit(model, method_name, inplace=False, debug=False):
    return _convert_ondevice_jit(
        model, method_name, inplace, debug, quant_type=QuantType.DYNAMIC
    )


def _quantize_ondevice_dynamic_jit_impl(
    model, qconfig_dict, method_name, inplace=False
):
    model = _prepare_ondevice_dynamic_jit(model, qconfig_dict, method_name, inplace)
    model = _convert_ondevice_dynamic_jit(model, method_name, inplace)
    return model


def _quantize_jit(
    model,
    qconfig_dict,
    run_fn=None,
    run_args=None,
    inplace=False,
    debug=False,
    quant_type=QuantType.STATIC,
):
    # Always do inplace convert because the Tensor is already
    # copied in prepare_jit when inplace is False
    if quant_type == QuantType.DYNAMIC:
        model = prepare_dynamic_jit(model, qconfig_dict, inplace)
        model = convert_dynamic_jit(model, True, debug)
    else:
        if not run_fn:
            raise AssertionError(
                "Must provide calibration function for post training static quantization"
            )
        if not run_args:
            raise AssertionError(
                "Must provide calibration dataset for post training static quantization"
            )
        model = prepare_jit(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        model = convert_jit(model, True, debug)

    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model


def quantize_jit(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    r"""Quantize the input float TorchScript model with
    post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, empty key means the qconfig will be applied
        to whole model unless it's overwritten by more specific configurations, the
        qconfig for each module is either found in the dictionary or fallback to
         the qconfig of parent module.

        Right now qconfig_dict is the only way to configure how the model is quantized,
        and it is done in the granularity of module, that is, we only support one type
        of qconfig for each torch.nn.Module, and the qconfig for sub module will
        override the qconfig for parent module, empty string means global configuration.
        `run_fn`: a calibration function for calibrating the prepared model
        `run_args`: positional arguments for `run_fn`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchScript model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization import quantize_jit

    ts_model = torch.jit.script(
        float_model.eval()
    )  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig("fbgemm")


    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)


    quantized_model = quantize_jit(
        ts_model, {"": qconfig}, calibrate, [data_loader_test]
    )
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.quantize_jit")
    return _quantize_jit(
        model,
        qconfig_dict,
        run_fn,
        run_args,
        inplace,
        debug,
        quant_type=QuantType.STATIC,
    )


def quantize_dynamic_jit(model, qconfig_dict, inplace=False, debug=False):
    r"""Quantize the input float TorchScript model with
    post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        descriptions in :func:`~torch.ao.quantization.quantize_jit`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchScript model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization import quantize_dynamic_jit

    ts_model = torch.jit.script(
        float_model.eval()
    )  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig("fbgemm")


    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)


    quantized_model = quantize_dynamic_jit(
        ts_model, {"": qconfig}, calibrate, [data_loader_test]
    )
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.quantize_dynamic_jit")
    return _quantize_jit(
        model, qconfig_dict, inplace=inplace, debug=debug, quant_type=QuantType.DYNAMIC
    )


def _quantize_ondevice_dynamic_jit(
    model, qconfig_dict, method_name="forward", inplace=False
):
    r"""Prepares the input float TorchScript model with
    *on-device* post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        `method_name`: Name of the method within the model, to be prepared for quantization
        descriptions in :func:`~torch.ao.quantization.quantize_jit`
        `inplace`: carry out model transformations in-place, the original module is
        mutated

    Return:
        TorchScript model that is ready for on device quantization.
        This means that the returned
        model has:
        - Method is inlined.
        - Model has observer modules inserted in the model.
        - Model has packed params inserted in the model. However they are empty as in they don't
          contain valid quantized weights.
        - observe_<method_name> is added that observe the values to be quantized.
        - reset_observers_<method_name> to reset observers.
        - quantize_<method_name> is added to the model.
          - This method extract scale, zero points.
          - Quantizes observed weights.
          - Creates packed params from it and update the attribute of the model with the new values
            for the packed params.
          - Reset the original fp32 weights with empty tensor using SetAttr.
        - quantized_<method_name> is added to the model.
          - This method uses quantized weights and quantized linear ops instead of fp32 op.
          - This method should be used for inference post PTQ.
        - Note that all method's signatures should be the same as method_name.

        Later on device:
        - Run reset_observers_<method_name>
        - Run observe_<method_name>
        - Run quantize_<method_name>
        - Now model can be saved and loaded later.
        - Run model with quantized_<method_name>

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization.quantize_jit import _quantize_ondevice_dynamic_jit

    ts_model = torch.jit.script(
        float_model.eval()
    )  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig("fbgemm")
    quant_ready_model = _quantize_ondevice_dynamic_jit(
        ts_model, {"": qconfig}, "forward", True
    )
    ```
    """
    return _quantize_ondevice_dynamic_jit_impl(
        model, qconfig_dict, method_name, inplace=inplace
    )
<|endoftext|># mypy: allow-untyped-defs
import copy
from itertools import chain
from typing import Any

import torch


__all__ = [
    "set_module_weight",
    "set_module_bias",
    "has_bias",
    "get_module_weight",
    "get_module_bias",
    "max_over_ndim",
    "min_over_ndim",
    "channel_range",
    "get_name_by_module",
    "cross_layer_equalization",
    "process_paired_modules_list_to_name",
    "expand_groups_in_paired_modules_list",
    "equalize",
    "converged",
]

_supported_types = {torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d}
_supported_intrinsic_types = {
    torch.ao.nn.intrinsic.ConvReLU2d,
    torch.ao.nn.intrinsic.LinearReLU,
    torch.ao.nn.intrinsic.ConvReLU1d,
}
_all_supported_types = _supported_types.union(_supported_intrinsic_types)


def set_module_weight(module, weight) -> None:
    if type(module) in _supported_types:
        module.weight = torch.nn.Parameter(weight)
    else:
        module[0].weight = torch.nn.Parameter(weight)


def set_module_bias(module, bias) -> None:
    if type(module) in _supported_types:
        module.bias = torch.nn.Parameter(bias)
    else:
        module[0].bias = torch.nn.Parameter(bias)


def has_bias(module) -> bool:
    if type(module) in _supported_types:
        return module.bias is not None
    else:
        return module[0].bias is not None


def get_module_weight(module):
    if type(module) in _supported_types:
        return module.weight
    else:
        return module[0].weight


def get_module_bias(module):
    if type(module) in _supported_types:
        return module.bias
    else:
        return module[0].bias


def max_over_ndim(input, axis_list, keepdim=False):
    """Apply 'torch.max' over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.max(axis, keepdim)
    return input


def min_over_ndim(input, axis_list, keepdim=False):
    """Apply 'torch.min' over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.min(axis, keepdim)
    return input


def channel_range(input, axis=0):
    """Find the range of weights associated with a specific channel."""
    size_of_tensor_dim = input.ndim
    axis_list = list(range(size_of_tensor_dim))
    axis_list.remove(axis)

    mins = min_over_ndim(input, axis_list)
    maxs = max_over_ndim(input, axis_list)

    if mins.size(0) != input.size(axis):
        raise AssertionError(
            "Dimensions of resultant channel range does not match size of requested axis"
        )
    return maxs - mins


def get_name_by_module(model, module):
    """Get the name of a module within a model.

    Args:
        model: a model (nn.module) that equalization is to be applied on
        module: a module within the model

    Returns:
        name: the name of the module within the model
    """
    for name, m in model.named_modules():
        if m is module:
            return name
    raise ValueError("module is not in the model")


def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
    """Scale the range of Tensor1.output to equal Tensor2.input.

    Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    """
    if (
        type(module1) not in _all_supported_types
        or type(module2) not in _all_supported_types
    ):
        raise ValueError(
            "module type not supported:", type(module1), " ", type(module2)
        )

    bias = get_module_bias(module1) if has_bias(module1) else None

    weight1 = get_module_weight(module1)
    weight2 = get_module_weight(module2)

    if weight1.size(output_axis) != weight2.size(input_axis):
        raise TypeError(
            "Number of output channels of first arg do not match \
        number input channels of second arg"
        )

    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)

    # producing scaling factors to applied
    weight2_range += 1e-9
    scaling_factors = torch.sqrt(weight1_range / weight2_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    if bias is not None:
        bias = bias * inverse_scaling_factors

    # formatting the scaling (1D) tensors to be applied on the given argument tensors
    # pads axis to (1D) tensors to then be broadcasted
    size1 = [1] * weight1.ndim
    size1[output_axis] = weight1.size(output_axis)
    size2 = [1] * weight2.ndim
    size2[input_axis] = weight2.size(input_axis)

    scaling_factors = torch.reshape(scaling_factors, size2)
    inverse_scaling_factors = torch.reshape(inverse_scaling_factors, size1)

    weight1 = weight1 * inverse_scaling_factors
    weight2 = weight2 * scaling_factors

    set_module_weight(module1, weight1)
    if bias is not None:
        set_module_bias(module1, bias)
    set_module_weight(module2, weight2)


def process_paired_modules_list_to_name(model, paired_modules_list):
    """Processes a list of paired modules to a list of names of paired modules."""

    for group in paired_modules_list:
        for i, item in enumerate(group):
            if isinstance(item, torch.nn.Module):
                group[i] = get_name_by_module(model, item)
            elif not isinstance(item, str):
                raise TypeError("item must be a nn.Module or a string")
    return paired_modules_list


def expand_groups_in_paired_modules_list(paired_modules_list):
    """Expands module pair groups larger than two into groups of two modules."""
    new_list = []

    for group in paired_modules_list:
        if len(group) == 1:
            raise ValueError("Group must have at least two modules")
        elif len(group) == 2:
            new_list.append(group)
        elif len(group) > 2:
            new_list.extend([group[i], group[i + 1]] for i in range(len(group) - 1))

    return new_list


def equalize(model, paired_modules_list, threshold=1e-4, inplace=True):
    """Equalize modules until convergence is achieved.

    Given a list of adjacent modules within a model, equalization will
    be applied between each pair, this will repeated until convergence is achieved

    Keeps a copy of the changing modules from the previous iteration, if the copies
    are not that different than the current modules (determined by converged_test),
    then the modules have converged enough that further equalizing is not necessary

    Reference is section 4.1 of this paper https://arxiv.org/pdf/1906.04721.pdf

    Args:
        model: a model (nn.Module) that equalization is to be applied on
            paired_modules_list (List(List[nn.module || str])): a list of lists
            where each sublist is a pair of two submodules found in the model,
            for each pair the two modules have to be adjacent in the model,
            with only piece-wise-linear functions like a (P)ReLU or LeakyReLU in between
            to get expected results.
            The list can contain either modules, or names of modules in the model.
            If you pass multiple modules in the same list, they will all be equalized together.
            threshold (float): a number used by the converged function to determine what degree
            of similarity between models is necessary for them to be called equivalent
        inplace (bool): determines if function is inplace or not
    """

    paired_modules_list = process_paired_modules_list_to_name(
        model, paired_modules_list
    )

    if not inplace:
        model = copy.deepcopy(model)

    paired_modules_list = expand_groups_in_paired_modules_list(paired_modules_list)

    name_to_module: dict[str, torch.nn.Module] = {}
    previous_name_to_module: dict[str, Any] = {}
    name_set = set(chain.from_iterable(paired_modules_list))

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
            previous_name_to_module[name] = None
    while not converged(name_to_module, previous_name_to_module, threshold):
        for pair in paired_modules_list:
            previous_name_to_module[pair[0]] = copy.deepcopy(name_to_module[pair[0]])
            previous_name_to_module[pair[1]] = copy.deepcopy(name_to_module[pair[1]])

            cross_layer_equalization(name_to_module[pair[0]], name_to_module[pair[1]])

    return model


def converged(curr_modules, prev_modules, threshold=1e-4):
    """Test whether modules are converged to a specified threshold.

    Tests for the summed norm of the differences between each set of modules
    being less than the given threshold

    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the difference
    between the associated modules in each dictionary

    """
    if curr_modules.keys() != prev_modules.keys():
        raise ValueError(
            "The keys to the given mappings must have the same set of names of modules"
        )

    summed_norms = torch.tensor(0.0)
    if None in prev_modules.values():
        return False
    for name in curr_modules:
        curr_weight = get_module_weight(curr_modules[name])
        prev_weight = get_module_weight(prev_modules[name])

        difference = curr_weight.sub(prev_weight)
        summed_norms += torch.linalg.vector_norm(difference)
    return bool(summed_norms < threshold)
<|endoftext|># mypy: allow-untyped-defs
import copy
import warnings
from collections import namedtuple
from typing import Any
from typing_extensions import deprecated, TypeAliasType

import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
    default_dynamic_fake_quant,
    default_embedding_fake_quant,
    default_embedding_fake_quant_4bit,
    default_fake_quant,
    default_fused_act_fake_quant,
    default_fused_per_channel_wt_fake_quant,
    default_fused_wt_fake_quant,
    default_per_channel_weight_fake_quant,
    default_weight_fake_quant,
    FakeQuantize,
    FakeQuantizeBase,
    fused_per_channel_wt_fake_quant_range_neg_127_to_127,
    fused_wt_fake_quant_range_neg_127_to_127,
    FusedMovingAvgObsFakeQuantize,
)

from .observer import (
    _PartialWrapper,
    default_debug_observer,
    default_dynamic_quant_observer,
    default_float_qparams_observer,
    default_float_qparams_observer_4bit,
    default_observer,
    default_per_channel_weight_observer,
    default_placeholder_observer,
    default_reuse_input_observer,
    default_weight_observer,
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    NoopObserver,
    ObserverBase,
    per_channel_weight_observer_range_neg_127_to_127,
    PlaceholderObserver,
    ReuseInputObserver,
    weight_observer_range_neg_127_to_127,
)


__all__ = [
    "QConfig",
    # TODO: deprecated, remove
    "QConfigDynamic",
    "default_qconfig",
    "default_debug_qconfig",
    "default_per_channel_qconfig",
    "default_dynamic_qconfig",
    "float16_dynamic_qconfig",
    "float16_static_qconfig",
    "per_channel_dynamic_qconfig",
    "float_qparams_weight_only_qconfig",
    "float_qparams_weight_only_qconfig_4bit",
    "default_quint8_weight_qconfig",
    "default_qat_qconfig",
    "default_dynamic_qat_qconfig",
    "default_weight_only_qconfig",
    "default_activation_only_qconfig",
    "default_qat_qconfig_v2",
    "default_reuse_input_qconfig",
    "default_symmetric_qnnpack_qconfig",
    "default_per_channel_symmetric_qnnpack_qconfig",
    "default_symmetric_qnnpack_qat_qconfig",
    "default_per_channel_symmetric_qnnpack_qat_qconfig",
    "default_embedding_qat_qconfig",
    "default_embedding_qat_qconfig_4bit",
    "get_default_qconfig",
    "get_default_qat_qconfig",
    "get_default_qconfig_dict",
    "get_default_qat_qconfig_dict",
    "QConfigAny",
    "qconfig_equals",
]


# pyrefly: ignore [invalid-inheritance]
class QConfig(namedtuple("QConfig", ["activation", "weight"])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8),
      )

    """

    __slots__ = ()

    def __new__(cls, activation, weight):
        # catch common mistakes
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError(
                "QConfig received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super().__new__(cls, activation, weight)


@deprecated(
    "`QConfigDynamic` is going to be deprecated in PyTorch 1.12, please use `QConfig` instead",
    category=FutureWarning,
)
# pyrefly: ignore [invalid-inheritance]
class QConfigDynamic(namedtuple("QConfigDynamic", ["activation", "weight"])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """

    __slots__ = ()

    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity):
        # catch common mistakes
        if isinstance(weight, nn.Module):
            raise ValueError(
                "QConfigDynamic received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super().__new__(cls, activation, weight)


default_qconfig = QConfig(activation=default_observer, weight=default_weight_observer)
"""
Default qconfig configuration.
"""

default_debug_qconfig = QConfig(
    weight=default_weight_observer, activation=default_debug_observer
)
"""
Default qconfig configuration for debugging.
"""

default_per_channel_qconfig = QConfig(
    activation=default_observer, weight=default_per_channel_weight_observer
)
"""
Default qconfig configuration for per channel weight quantization.
"""

default_dynamic_qconfig = QConfig(
    activation=default_dynamic_quant_observer, weight=default_weight_observer
)
"""
Default dynamic qconfig.
"""

float16_dynamic_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.float16, is_dynamic=True),
    weight=PlaceholderObserver.with_args(dtype=torch.float16),
)
"""
Dynamic qconfig with weights quantized to `torch.float16`.
"""

float16_static_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.float16),
    weight=PlaceholderObserver.with_args(dtype=torch.float16),
)
"""
Dynamic qconfig with both activations and weights quantized to `torch.float16`.
"""

per_channel_dynamic_qconfig = QConfig(
    activation=default_dynamic_quant_observer,
    weight=default_per_channel_weight_observer,
)
"""
Dynamic qconfig with weights quantized per channel.
"""

float_qparams_weight_only_qconfig = QConfig(
    activation=default_placeholder_observer, weight=default_float_qparams_observer
)
"""
Dynamic qconfig with weights quantized with a floating point zero_point.
"""

float_qparams_weight_only_qconfig_4bit = QConfig(
    activation=default_placeholder_observer, weight=default_float_qparams_observer_4bit
)

default_qat_qconfig = QConfig(
    activation=default_fake_quant, weight=default_weight_fake_quant
)
"""
Default qconfig for QAT.
"""

default_dynamic_qat_qconfig = QConfig(
    activation=default_dynamic_fake_quant, weight=default_weight_fake_quant
)
"""
Default qconfig for dynamic QAT.
"""

default_weight_only_qconfig = QConfig(
    activation=torch.nn.Identity, weight=default_weight_fake_quant
)
"""
Default qconfig for quantizing weights only.
"""

default_activation_only_qconfig = QConfig(
    activation=default_fake_quant, weight=torch.nn.Identity
)
"""
Default qconfig for quantizing activations only.
"""

# QAT config that uses a fused observer + fake quant modules for optimized training performance.
# to modify the activation/weight observers, the default entries in fake_quantize.py can be modified.
default_qat_qconfig_v2 = QConfig(
    activation=default_fused_act_fake_quant, weight=default_fused_wt_fake_quant
)
"""
Fused version of `default_qat_config`, has performance benefits.
"""

default_reuse_input_qconfig = QConfig(
    activation=default_reuse_input_observer, weight=NoopObserver
)
"""
Default qconfig for operators that reuse the observers from input Tensor, e.g. reshape
"""


def get_default_qconfig(backend="x86", version=0):
    """
    Returns the default PTQ qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    if backend not in supported_backends:
        raise AssertionError(
            "backend: "
            + str(backend)
            + f" not supported. backend must be one of {supported_backends}"
        )

    if version == 0:
        if backend == "fbgemm":
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=True),
                weight=default_per_channel_weight_observer,
            )
        elif backend == "qnnpack":
            # TODO: make this compatible with xnnpack constraints
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=False),
                weight=default_weight_observer,
            )
        elif backend == "onednn":
            if not torch.cpu._is_vnni_supported():
                warnings.warn(
                    "Default qconfig of oneDNN backend with reduce_range of false may have accuracy issues "
                    "on CPU without Vector Neural Network Instruction support.",
                    stacklevel=2,
                )
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=False),
                weight=default_per_channel_weight_observer,
            )
        elif backend == "x86":
            qconfig = QConfig(
                activation=HistogramObserver.with_args(reduce_range=True),
                weight=default_per_channel_weight_observer,
            )
        else:
            # won't reach
            qconfig = default_qconfig
    else:
        raise AssertionError(
            "Version number: "
            + str(version)
            + " in get_default_qconfig is not supported. Version number must be 0"
        )

    return qconfig


"""
Default, symmetric PTQ qconfig for the specified backend. And a per_channel
variant of the same.

Symmetric here applies to signed weights with zero point = 0, and additional
value restrictions. The activations are also signed 8-bit integers with this
qconfig.

    * Once this change is merged [as of 3/17/22], with backend or qengine =
    'qnnpack', some quantized operators with this symmetric qconfig may use
    operators from xnnpack library.

        ** Support to use xnnpack ops with `qnnpack` backed for asymmetric
        qconfig (returned by get_default_qconfig()) is not available yet.

    * This qconfig uses signed activations and weights. Weights have added
    restrictions such as zero point is forced to be 0, making the weights
    symmetric, hence the name. And the 8-bit quantized values are
    restricting to [-127, +127], excluding -128.

    * xnnpack has a requantization scale value restriction, 0x1p-32 <=
    requantization_scale < 256.0 where, `requantization_scale = (input_scale
    * kernel_scale) / (output_scale)`. Using this eps (w/ assumed max value
    of 256) is to prevent requantization_scale to go below xnnpack lower
    threshold.
"""
default_symmetric_qnnpack_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=torch.qint8, reduce_range=False, eps=2**-12
    ),
    weight=weight_observer_range_neg_127_to_127,
)

default_per_channel_symmetric_qnnpack_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=torch.qint8, reduce_range=False, eps=2**-12
    ),
    weight=per_channel_weight_observer_range_neg_127_to_127,
)

default_embedding_qat_qconfig = QConfig(
    activation=NoopObserver.with_args(dtype=torch.float32),
    weight=default_embedding_fake_quant,
)

default_embedding_qat_qconfig_4bit = QConfig(
    activation=NoopObserver.with_args(dtype=torch.float32),
    weight=default_embedding_fake_quant_4bit,
)

default_quint8_weight_qconfig = QConfig(
    activation=HistogramObserver, weight=MinMaxObserver
)


def get_default_qat_qconfig(backend="x86", version=1):
    """
    Returns the default QAT qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.
      * `version`: version, for backwards compatibility. Can be `None` or `1`.

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    if backend not in supported_backends:
        raise AssertionError(
            "backend: "
            + str(backend)
            + f" not supported. backend must be one of {supported_backends}"
        )

    # Histogram observer is too slow for quantization aware training
    if version == 0:
        if backend == "fbgemm":
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_per_channel_weight_fake_quant,
            )
        elif backend == "qnnpack":
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=False,
                ),
                weight=default_weight_fake_quant,
            )
        elif backend == "onednn":
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255
                ),
                weight=default_per_channel_weight_fake_quant,
            )
        elif backend == "x86":
            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_per_channel_weight_fake_quant,
            )
        else:
            qconfig = default_qat_qconfig
    # Use the fused observe + fake_quant modules for doing QAT.
    elif version == 1:
        if backend == "fbgemm":
            qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_fused_per_channel_wt_fake_quant,
            )
        elif backend == "qnnpack":
            # TODO: make this compatible with xnnpack constraints
            qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=False,
                ),
                weight=default_fused_wt_fake_quant,
            )
        elif backend == "onednn":
            qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255
                ),
                weight=default_fused_per_channel_wt_fake_quant,
            )
        elif backend == "x86":
            qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=True,
                ),
                weight=default_fused_per_channel_wt_fake_quant,
            )
        else:
            qconfig = default_qat_qconfig_v2
    else:
        raise AssertionError(
            "Version number: "
            + str(version)
            + "in get_default_qat_qconfig is not supported. Version number must be 0 or 1"
        )

    return qconfig


"""
Default symmetric QAT qconfig for qnnpack. And its per channel weight variant.
"""
default_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        reduce_range=False,
        eps=2**-12,
    ),
    weight=fused_wt_fake_quant_range_neg_127_to_127,
)

default_per_channel_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        reduce_range=False,
        eps=2**-12,
    ),
    weight=fused_per_channel_wt_fake_quant_range_neg_127_to_127,
)

_default_fp32_placeholder_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.float32),
    weight=PlaceholderObserver.with_args(dtype=torch.float32),
)

_default_quint8_placeholder_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=torch.quint8),
    # operators using this qconfig doesn't have weights
    weight=None,
)


@deprecated(
    "`torch.ao.quantization.get_default_qconfig_dict` is deprecated and will be removed in "
    "a future version. Please use `torch.ao.quantization.get_default_qconfig_mapping` instead.",
    category=FutureWarning,
)
def get_default_qconfig_dict(backend="x86", version=0):
    return torch.ao.quantization.get_default_qconfig_mapping(backend, version).to_dict()


@deprecated(
    "`torch.ao.quantization.get_default_qat_qconfig_dict` is deprecated and will be removed in "
    "a future version. Please use `torch.ao.quantization.get_default_qat_qconfig_mapping` instead.",
    category=FutureWarning,
)
def get_default_qat_qconfig_dict(backend="x86", version=1):
    return torch.ao.quantization.get_default_qat_qconfig_mapping(
        backend, version
    ).to_dict()


def _assert_valid_qconfig(qconfig: QConfig | None, mod: torch.nn.Module) -> None:
    """
    Verifies that this `qconfig` is valid.
    """
    if qconfig is None:
        return
    is_conv_transpose_mod = isinstance(
        mod,
        (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d),
    )
    if is_conv_transpose_mod:
        if qconfig.weight is None:
            # for now, we assume that any qconfig for ConvTranspose without a weight is valid
            return
        example_observer = qconfig.weight()
        is_per_channel = isinstance(
            example_observer,
            (
                torch.ao.quantization.PerChannelMinMaxObserver,
                torch.ao.quantization.MovingAveragePerChannelMinMaxObserver,
            ),
        )
        if is_per_channel:
            raise AssertionError(
                "Per channel weight observer is not supported yet for ConvTranspose{n}d."
            )


QConfigAny = TypeAliasType("QConfigAny", QConfig | None)


def _add_module_to_qconfig_obs_ctr(
    qconfig: QConfigAny, module: nn.Module | None
) -> Any:
    r"""This is a helper function for use in quantization prepare that updates a qconfig so that
    the constructors stored in the qconfig will create observers on the same device that
    'module' is on. This is intended to be used when the qconfigs are propagated to each
    module in order to avoid potential device alignment issues.

    Args:
        qconfig: QConfig with obs constructors stored in activation and weight
        module: module which the qconfig is related to

    Return:
        qconfig: configured so that obs constructors set to construct on the same device as module
    """

    if module is None or qconfig is None or qconfig._fields != ("activation", "weight"):
        return qconfig

    def get_factory_kwargs_based_on_module_device():
        if not isinstance(module, torch.nn.Module):
            raise AssertionError("module must be an instance of torch.nn.Module")
        devices = {p.device for p in module.parameters()} | {
            p.device for p in module.buffers()
        }
        device = next(iter(devices)) if len(devices) > 0 else None
        return None if device is None else {"device": device}

    def configure_constructor_to_put_obs_on_module_device(original_constructor):
        try:
            # check if constructor can accept factory_kwargs
            check = original_constructor.with_args(factory_kwargs=None)
            check()
            return original_constructor.with_callable_args(
                factory_kwargs=get_factory_kwargs_based_on_module_device
            )
        except AttributeError:  # qconfig doesn't have activation or weight
            return original_constructor
        except TypeError:  # the class doesn't accept factory_kwargs argument
            return original_constructor

    activation = configure_constructor_to_put_obs_on_module_device(qconfig.activation)
    weight = configure_constructor_to_put_obs_on_module_device(qconfig.weight)

    return QConfig(activation, weight)


_ObserverOrFakeQuantizeConstructor = (
    _PartialWrapper | type[ObserverBase] | type[FakeQuantizeBase]
)


def _obs_or_fq_ctr_equals(
    obs_or_fq1: _ObserverOrFakeQuantizeConstructor,
    obs_or_fq2: _ObserverOrFakeQuantizeConstructor,
):
    if isinstance(obs_or_fq1, _PartialWrapper) and isinstance(
        obs_or_fq2, _PartialWrapper
    ):
        return _partial_wrapper_equals(obs_or_fq1, obs_or_fq2)
    return obs_or_fq1 == obs_or_fq2


def _partial_wrapper_equals(obs_or_fq1: _PartialWrapper, obs_or_fq2: _PartialWrapper):
    """
    Return whether the two partial wrappers are equal,
    """
    # functools.partial has no __eq__ operator defined so '==' defaults to 'is'
    obs_or_fq1_keywords = copy.copy(obs_or_fq1.p.keywords)
    obs_or_fq2_keywords = copy.copy(obs_or_fq2.p.keywords)
    keywords_equal = True
    # compare observer constructor with _obs_or_fq_ctr_equals since direct compare would fail
    if "observer" in obs_or_fq1_keywords and "observer" in obs_or_fq2_keywords:
        keywords_equal = keywords_equal and _obs_or_fq_ctr_equals(
            obs_or_fq1_keywords["observer"], obs_or_fq2_keywords["observer"]
        )
        obs_or_fq1_keywords.pop("observer")
        obs_or_fq2_keywords.pop("observer")
    keywords_equal = keywords_equal and obs_or_fq1_keywords == obs_or_fq2_keywords
    return (
        obs_or_fq1.p.func == obs_or_fq2.p.func
        and obs_or_fq1.p.args == obs_or_fq2.p.args
        and keywords_equal
    )


def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
    """
    Returns `True` if `q1` equals `q2`, and `False` otherwise.
    """
    if q1 is None or q2 is None:
        return q1 == q2
    else:
        if q1 is None or q2 is None:
            raise AssertionError(
                "Both q1 and q2 must be non-None for qconfig comparison"
            )
        try:
            # Qconfig weight and activation can be either a partial wrapper,
            # or an observer class. Special handling is required (above) for
            # comparing partial wrappers.
            activation_same = _obs_or_fq_ctr_equals(q1.activation, q2.activation)
            weight_same = _obs_or_fq_ctr_equals(q1.weight, q2.weight)
            return activation_same and weight_same
        except AttributeError:
            return q1 == q2


def _activation_is_memoryless(qconfig: QConfig):
    """
    Return whether the observer for activations defined in the given QConfig is memoryless.
    This means a MovingAverage observer with averaging constant equal to 1.
    """

    def _is_memoryless(observer):
        return (
            hasattr(observer, "averaging_constant") and observer.averaging_constant == 1
        )

    act = qconfig.activation()
    if isinstance(act, FakeQuantizeBase) and hasattr(act, "activation_post_process"):
        return _is_memoryless(act.activation_post_process)
    else:
        return _is_memoryless(act)


def _is_reuse_input_qconfig(qconfig: QConfig | None):
    return (
        qconfig is not None
        and isinstance(qconfig.activation(), ReuseInputObserver)
        and isinstance(qconfig.weight(), NoopObserver)
    )
<|endoftext|># mypy: allow-untyped-defs
"""
Utils shared by different modes of quantization (eager/graph)
"""

import functools
import warnings
from collections import OrderedDict
from collections.abc import Callable
from inspect import getfullargspec, signature
from typing import Any
from typing_extensions import TypeAliasType

import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized


NodePattern = TypeAliasType(
    "NodePattern", tuple[Node, Node] | tuple[Node, tuple[Node, Node]] | Any
)


# This is the Quantizer class instance from torch/quantization/fx/quantize.py.
# Define separately to prevent circular imports.
# TODO(future PR): improve this.
# make this public once fixed (can't be public as is because setting the module directly
# doesn't work)
QuantizerCls = Any

# Type for fusion patterns, it can be more complicated than the following actually,
# see pattern.md for docs
# TODO: not sure if typing supports recursive data types


Pattern = TypeAliasType(
    "Pattern",
    Callable
    | tuple[Callable, Callable]
    | tuple[Callable, tuple[Callable, Callable]]
    | Any,
)


# TODO: maybe rename this to MatchInputNode
class MatchAllNode:
    """A node pattern that matches all nodes, used in defining
    fusion patterns in FX Graph Mode Quantization
    """


module_type_list = {
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AdaptiveAvgPool3d,
    torch.nn.AvgPool1d,
    torch.nn.AvgPool2d,
    torch.nn.AvgPool3d,
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
    torch.nn.MaxPool3d,
    torch.nn.Identity,
    torch.nn.Hardsigmoid,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
}
func_list = {
    torch.nn.functional.adaptive_avg_pool1d,
    torch.nn.functional.adaptive_avg_pool2d,
    torch.nn.functional.adaptive_avg_pool3d,
    torch.nn.functional.elu,
    torch.nn.functional.hardswish,
    torch.nn.functional.instance_norm,
    torch.nn.functional.layer_norm,
    torch.nn.functional.leaky_relu,
    torch.nn.functional.silu,
    torch.nn.functional.mish,
    torch.nn.functional.dropout,
    torch.nn.functional.max_pool1d,
    torch.nn.functional.max_pool2d,
    torch.nn.functional.max_pool3d,
    torch.nn.functional.relu,
    torch.nn.functional.hardtanh,
    torch.nn.functional.hardtanh_,
    torch.nn.functional.hardsigmoid,
    torch.nn.functional.sigmoid,
    torch.transpose,
    torch.repeat_interleave,
    torch.sigmoid,
    torch.squeeze,
    torch.stack,
    torch.sum,
    torch.tanh,
    torch.unsqueeze,
    torch.cat,
}
method_list = {
    torch.mean,
    "relu",
    "relu_",
    "contiguous",
    "detach",
    "detach_",
    "hardsigmoid",
    "hardsigmoid_",
    "permute",
    "repeat",
    "repeat_interleave",
    "reshape",
    "resize_",
    "shape",
    "sigmoid",
    "sigmoid_",
    "size",
    "squeeze",
    "squeeze_",
    "tanh",
    "tanh_",
    "transpose",
    "unsqueeze",
    "unsqueeze_",
    "view",
}


# TODO: not used now, remove
def check_node(node, modules):
    # TODO: reuse is_fixed_qparam_node after we move this function to _lower_to_native_backend.py
    is_call_function = node.op == "call_function" and node.target in func_list
    is_call_method = node.op == "call_method" and node.target in method_list
    is_call_module = (
        node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    )
    return is_call_function, is_call_method, is_call_module


def get_combined_dict(default_dict, additional_dict):
    """
    Combines two dictionaries.

    This function takes two dictionaries as input and returns a new dictionary
    that contains all the key-value pairs from both input dictionaries.
    If there are any duplicate keys in the `additional_dict`, the values
    from the `additional_dict` will overwrite those in the `default_dict`.
    Args:
        default_dict (dict): The main dictionary that will be used as the base
        additional_dict (dict): The dictionary used to update `default_dict`

    Returns:
        dict: The resulting dictionary
    Example:
        >>> x = dict(a=1, b=1)
        >>> y = dict(b=2, c=3)
        >>> get_combined_dict(x, y)
        {'a': 1, 'b': 2, 'c': 3}
    """
    d = default_dict.copy()
    d.update(additional_dict)
    return d


def is_per_tensor(qscheme):
    return qscheme == torch.per_tensor_affine or qscheme == torch.per_tensor_symmetric


def is_per_channel(qscheme):
    return qscheme in [
        torch.per_channel_affine,
        torch.per_channel_affine_float_qparams,
        torch.per_channel_symmetric,
    ]


def getattr_from_fqn(obj: Any, fqn: str) -> Any:
    """
    Given an obj and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    return functools.reduce(getattr, fqn.split("."), obj)


def to_underlying_dtype(qdtype):
    DTYPE_MAPPING = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
        torch.quint4x2: torch.uint8,
        torch.quint2x4: torch.uint8,
        torch.uint8: torch.uint8,
        torch.int8: torch.int8,
        torch.uint16: torch.uint16,
        torch.int16: torch.int16,
        torch.int32: torch.int32,
        torch.float8_e5m2: torch.float8_e5m2,
        torch.float8_e4m3fn: torch.float8_e4m3fn,
    }
    if qdtype not in DTYPE_MAPPING:
        raise AssertionError("Unsupported dtype: " + str(qdtype))
    return DTYPE_MAPPING[qdtype]


def get_qparam_dict(observer_or_fake_quant):
    from torch.ao.quantization.observer import PlaceholderObserver

    qscheme = getattr(observer_or_fake_quant, "qscheme", None)
    dtype = observer_or_fake_quant.dtype
    qparams = {"qscheme": qscheme, "dtype": dtype}

    if not qscheme or isinstance(observer_or_fake_quant, PlaceholderObserver):
        return {"qscheme": None, "dtype": dtype}

    if is_per_tensor(qscheme):
        qscheme = torch.per_tensor_affine
    elif is_per_channel(qscheme):
        # change symmetric to affine since we do not have symmetric
        # quantized Tensor
        if qscheme == torch.per_channel_symmetric:
            qscheme = torch.per_channel_affine
        qparams["axis"] = observer_or_fake_quant.ch_axis
    else:
        raise RuntimeError(f"Unrecognized qscheme: {qscheme}")
    # update qscheme, since we don't have symmetric quant qscheme
    # in quantized Tensor
    qparams["qscheme"] = qscheme

    scale, zero_point = observer_or_fake_quant.calculate_qparams()
    qparams["scale"] = scale
    qparams["zero_point"] = zero_point

    if hasattr(observer_or_fake_quant, "quant_min"):
        qparams["quant_min"] = observer_or_fake_quant.quant_min
    if hasattr(observer_or_fake_quant, "quant_max"):
        qparams["quant_max"] = observer_or_fake_quant.quant_max

    return qparams


def get_swapped_custom_module_class(
    custom_module, custom_module_class_mapping, qconfig
):
    """Get the observed/quantized custom module class that we need
    to swap ``custom_module`` to.

    Input:

    - custom_module: input, can be an instance of either a float or observed custom module
    - custom_module_class_mapping: the float to observed or observed to quantized custom module class mapping
    - qconfig: qconfig configured for the custom module

    Output:

    Corresponding observed/quantized custom module class for input custom module instance.
    """
    quant_type = get_quant_type(qconfig)
    class_mapping = custom_module_class_mapping.get(quant_type, {})
    if type(custom_module) not in class_mapping:
        raise AssertionError(
            "did not find corresponding observed "
            f"module class for {type(custom_module)} in mapping: {class_mapping}"
        )
    return class_mapping[type(custom_module)]


def activation_dtype(qconfig):
    if qconfig is None:
        raise AssertionError("qconfig must be provided to determine activation dtype")
    activation = qconfig.activation()
    return activation.dtype


def weight_dtype(qconfig):
    if qconfig is None:
        raise AssertionError("qconfig must be provided to determine weight dtype")
    weight = qconfig.weight()
    return weight.dtype


def activation_is_statically_quantized(qconfig):
    """Given a qconfig, decide if the activation needs to be
    quantized or not, this includes quantizing to quint8, qint8 and qint32 and float16
    """
    return activation_dtype(qconfig) in [
        torch.quint8,
        torch.qint8,
        torch.qint32,
        torch.float16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ] and (not activation_is_dynamically_quantized(qconfig))


def activation_is_dynamically_quantized(qconfig):
    """Given a qconfig, decide if the activation needs to be
    dynamically quantized or not, this includes dynamically quantizing to
    quint8, qint8 and float16
    """
    _activation_dtype, _, activation_is_dynamic = get_qconfig_dtypes(qconfig)
    return activation_is_dynamic


def activation_is_int8_quantized(qconfig):
    """Given a qconfig, decide if the activation needs to be
    quantized to int8 or not, this includes quantizing to quint8, qint8
    """
    return activation_dtype(qconfig) in [
        torch.quint8,
        torch.qint8,
        torch.uint8,
        torch.int8,
    ]


def activation_is_int32_quantized(qconfig):
    """Given a qconfig, decide if the activation needs to be
    quantized to int32 or not
    """
    return activation_dtype(qconfig) in [torch.qint32, torch.int32]


def weight_is_quantized(qconfig):
    """Given a qconfig, decide if the weight needs to be
    quantized or not
    """
    return weight_dtype(qconfig) in [
        torch.quint8,
        torch.qint8,
        torch.float16,
        torch.quint4x2,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ]


def weight_is_statically_quantized(qconfig):
    """Given a qconfig, decide if the weight needs to be statically
    quantized or not
    """
    return weight_dtype(qconfig) in [torch.quint8, torch.qint8, torch.uint8, torch.int8]


def op_is_int8_dynamically_quantized(qconfig) -> bool:
    """Given a qconfig, returns True if this op is using int8 dynamic
    quantization
    """
    activation_dtype, weight_dtype, activation_is_dynamic = get_qconfig_dtypes(qconfig)
    return (
        activation_dtype in [torch.quint8, torch.uint8]
        and
        # for now, the lines below assume fbgemm or qnnpack
        weight_dtype in [torch.qint8, torch.int8]
        and activation_is_dynamic
    )


def get_qconfig_dtypes(qconfig):
    r"""returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_is_dynamic)
    """
    if qconfig is None:
        raise AssertionError("qconfig must be provided to extract dtypes")
    activation = qconfig.activation()
    weight = qconfig.weight()
    act_is_dynamic = getattr(activation, "is_dynamic", False)
    return (activation.dtype, weight.dtype, act_is_dynamic)


def get_quant_type(qconfig):
    if qconfig is None:
        raise AssertionError("qconfig must be provided to determine quant type")
    activation = qconfig.activation()
    weight = qconfig.weight()
    static_dtypes = [
        torch.quint8,
        torch.qint8,
        torch.quint4x2,
        torch.qint32,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ]
    if weight.dtype in static_dtypes:
        if hasattr(activation, "is_dynamic") and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype in static_dtypes:
            return QuantType.STATIC
        else:
            return QuantType.WEIGHT_ONLY

    if weight.dtype == torch.float16:
        if hasattr(activation, "is_dynamic") and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype == torch.float16:
            return QuantType.STATIC

    raise Exception(  # noqa: TRY002
        f"Unrecognized dtype combination in get_quant_type: activation({activation.dtype}),"
        f"weight({weight.dtype})"
    )


def check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
    """Checks if the given minimum and maximum values are valid, meaning that
    they exist and the min value is less than the max value.
    """
    if min_val.numel() == 0 or max_val.numel() == 0:
        warnings.warn(
            "must run observer before calling calculate_qparams. "
            + "Returning default values.",
            stacklevel=2,
        )
        return False

    if min_val.dim() == 0 or max_val.dim() == 0:
        if min_val == float("inf") and max_val == float("-inf"):
            warnings.warn(
                "must run observer before calling calculate_qparams. "
                + "Returning default values.",
                stacklevel=2,
            )

            return False

        if min_val > max_val:
            raise AssertionError(f"min {min_val} should be less than max {max_val}")
    else:
        if torch.any(min_val > max_val):
            raise AssertionError(f"min {min_val} should be less than max {max_val}")

    return True


def calculate_qmin_qmax(
    quant_min: int,
    quant_max: int,
    has_customized_qrange: bool,
    dtype: torch.dtype,
    reduce_range: bool,
) -> tuple[int, int]:
    r"""Calculates actual qmin and qmax based on the quantization range,
    observer datatype and if range is reduced.
    """
    # TODO(jerryzh): Figure out why custom quant_min/quant_max are still adjusted.
    if has_customized_qrange:
        # This initialization here is to be resolve TorchScript compilation issues and allow
        # using of refinement to decouple initial_qmin and initial_qmax from quantization range.
        # The actual values of initial_qmin and initial_qmax will be reset below.
        if dtype in [torch.qint32, torch.int32]:
            initial_quant_min, initial_quant_max = 0, 2**32 - 1
        else:
            initial_quant_min, initial_quant_max = 0, 255
        # The following assignment of self.qmin and self.qmax to the local variables and the if check refine the
        # attribute from Optional valid integers for use, based on TorchScript's requirements.
        custom_quant_min, custom_quant_max = quant_min, quant_max
        if custom_quant_min is not None and custom_quant_max is not None:
            initial_quant_min, initial_quant_max = (
                custom_quant_min,
                custom_quant_max,
            )

        qrange_len = initial_quant_max - initial_quant_min + 1
        if dtype in [torch.qint8, torch.int8]:
            if not (0 < qrange_len <= 256):
                raise AssertionError(
                    "quantization range should be positive and not exceed the maximum bit range (=256)."
                )
        elif dtype in [torch.qint32, torch.int32]:
            if not (0 < qrange_len <= 2**32):
                raise AssertionError(
                    "quantization range should be positive and not exceed the maximum bit range (=4294967296)."
                )
        if reduce_range:
            quant_min, quant_max = quant_min // 2, quant_max // 2
    else:
        # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
        if dtype in [torch.qint8, torch.int8]:
            if reduce_range:
                quant_min, quant_max = -64, 63
            else:
                quant_min, quant_max = -128, 127
        elif dtype in [torch.quint8, torch.uint8]:
            if reduce_range:
                quant_min, quant_max = 0, 127
            else:
                quant_min, quant_max = 0, 255
        elif dtype in [torch.qint32, torch.int32]:
            quant_min, quant_max = -1 * (2**31), (2**31) - 1
        elif dtype == torch.uint16:
            quant_min, quant_max = 0, 2**16 - 1
        elif dtype == torch.int16:
            quant_min, quant_max = -(2**15), 2**15 - 1
        else:
            quant_min, quant_max = 0, 15
    return quant_min, quant_max


def _parent_name(target):
    """
    Turn 'foo.bar' into ['foo', 'bar']
    """
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


def has_no_children_ignoring_parametrizations(module):
    """
    Checks if module._modules is empty or
    if module is a parametrization, checks that module._modules only has
    the 'parametrizations' module
    """
    if len(module._modules) == 0:
        return True
    elif is_parametrized(module):
        return len(module._modules) == 1 and "parametrizations" in module._modules
    else:
        return False


def _get_path_of_module(
    root: torch.nn.Module, submodule: torch.nn.Module
) -> str | None:
    """Get the path (fully qualified name) of a submodule

    Example::

    >> class M(torch.nn.Module):
           def __init__(self) -> None:
               self.linear = torch.nn.Linear(5, 5)
           def forward(self, x):
               return self.linear(x)

    >> m = M()
    >> l = m.linear
    >> _get_path_of_module(m, l)
    "linear"
    """
    for n, p in root.named_modules():
        if submodule is p:
            return n
    return None


def _get_signature_locals(f: Callable, loc: dict[str, Any]) -> dict[str, Any]:
    """Get local keyword arguments

    Example::

    >> def f(self, a, b=9):
           pass
    >> loc = {"a": 6, "c": 7}
    >> _get_signature_locals(f, loc)
    {"a": 6}
    """
    return {k: v for k, v in loc.items() if k in signature(f).parameters}


def _get_default_kwargs(f: Callable) -> "OrderedDict[str, Any]":
    """Get all default keyword arguments from function signature

    Example::

    >> def f(self, a, b=9):
           pass
    >> _get_default_kwargs(f)
    {"b": 9}
    """
    kwargs = {}
    for name, param in signature(f).parameters.items():
        if param.default is not param.empty:
            kwargs[name] = param.default
        elif param.kind is param.VAR_POSITIONAL:
            kwargs[name] = ()
        elif param.kind is param.VAR_KEYWORD:
            kwargs[name] = {}
    return OrderedDict(kwargs)


def _normalize_kwargs(func: Callable, loc: dict[str, Any]) -> "OrderedDict[str, Any]":
    """Given a function and local function arguments, normalize the keyword
    arguments by filling in default arguments from function signature

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> loc = {"key2": 6}
    >> _normalize_kwargs(f, loc)
    {"key1": 3, "key2": 6}
    """
    default_kwargs = _get_default_kwargs(func)
    local_kwargs = _get_signature_locals(func, loc)
    normalized_kwargs = default_kwargs.copy()
    for attr, val in local_kwargs.items():
        if attr in normalized_kwargs:
            # override the default keyword arguments
            normalized_kwargs[attr] = val
    return normalized_kwargs


def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    r"""Validates that the user-specified quantization range is properly initialized
    and within the given bound supported by the observer dtype.

    To accommodate lower-bit quantization with respect to the existing torch.qint8 and
    torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
    in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
    values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
    fake quantization. These estimates are compared against parameters learned through backpropagation.
    The related literature for scale and zero point via backpropagation are as follows:

    Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
    Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
    """
    # The variable names are prefixed with "initial" because their values (qmin and qmax) might be adjusted
    # based on whether quantization range is reduced and the datatype (signed/unsigned) used by the observer.
    if not (quant_min <= 0 <= quant_max):
        raise AssertionError("Used-specified quantization range must include 0.")
    if quant_min >= quant_max:
        raise AssertionError(
            "qmin must be strictly less than qmax for user-specified quantization range."
        )


# Functionally equivalent to '_calculate_qparams' in observer.py. Observers must be torchscriptable however and qscheme
# as far as I can tell is not allowed to passed as a parameter in torchscript functions. This makes refactoring observer
# to use this utility a massive pain and very gross. For now Im opting just to duplicate as this code seems unlikely to change
# (last update over 1 year ago) and when torchscript is fully deprecated we can refactor. TODO(jakeszwe, jerryzh168)
def determine_qparams(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    eps: torch.Tensor,
    has_customized_qrange: bool,
    qscheme: torch.qscheme = torch.per_tensor_affine,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates the quantization parameters, given min and max
    value tensors. Works for both per tensor and per channel cases

    Args:
        min_val: Minimum values per channel
        max_val: Maximum values per channel

    Returns:
        scales: Scales tensor of shape (#channels,)
        zero_points: Zero points tensor of shape (#channels,)
    """
    if not check_min_max_valid(min_val, max_val):
        return torch.tensor([1.0], device=min_val.device.type), torch.tensor(
            [0], device=min_val.device.type
        )

    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    device = min_val_neg.device
    scale = torch.ones(min_val_neg.size(), dtype=torch.double, device=device)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)
    eps = eps.to(device)

    if qscheme == torch.per_tensor_symmetric or qscheme == torch.per_channel_symmetric:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        scale = torch.max(scale, eps)
        if dtype in [torch.uint8, torch.quint8]:
            if has_customized_qrange:
                # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                zero_point = zero_point.new_full(
                    zero_point.size(), (quant_min + quant_max) // 2
                )
            else:
                zero_point = zero_point.new_full(zero_point.size(), 128)
    elif qscheme == torch.per_channel_affine_float_qparams:
        scale = (max_val - min_val) / float(quant_max - quant_min)
        scale = torch.where(scale > eps, scale, torch.ones_like(scale))
        # We use the quantize function
        # xq = Round(Xf * inv_scale + zero_point),
        # setting zero_point to (-1 * min *inv_scale) we get
        # Xq = Round((Xf - min) * inv_scale)
        zero_point = -1 * min_val / scale
    else:
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, eps)
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

    # For scalar values, cast them to Tensors of size 1 to keep the shape
    # consistent with default values in FakeQuantize.
    if len(scale.shape) == 0:
        # TODO: switch to scale.item() after adding JIT support
        scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
    if len(zero_point.shape) == 0:
        # TODO: switch to zero_point.item() after adding JIT support
        zero_point = torch.tensor(
            [int(zero_point)], dtype=zero_point.dtype, device=device
        )
        if qscheme == torch.per_channel_affine_float_qparams:
            zero_point = torch.tensor(
                [float(zero_point)], dtype=zero_point.dtype, device=device
            )

    return scale.to(torch.double), zero_point.to(torch.int64)


def _get_num_pos_args(f: Callable) -> int:
    """Get number of positional args for a function

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> _get_num_pos_args(f)
    3
    """
    return len(getfullargspec(f).args)


def get_fqn_to_example_inputs(
    model: torch.nn.Module, example_inputs: tuple[Any, ...]
) -> dict[str, tuple[Any, ...]]:
    """Given a model and its example inputs, return a dictionary from
    fully qualified name of submodules to example_inputs for that submodule,
    e.g. ``{"linear1": (tensor1,), "linear2": (tensor2,), "sub": (tensor3,),
    "sub.linear1": (tensor4,), ...}``

    Used to make quantizing submodules easier now that FX Graph Mode Quantization requires
    example inputs.

    Also works for keyword arguments with default values, we would flatten keyword
    arguments as positional arguments and fill in the missing keyword args with default
    values, e.g. if we have a forward function::

        def forward(self, x, key1=3, key2=3): ...

    and we call it with ``self.submodule(x, key2=6)``
    we'll get ``example_inputs: (x, 3, 6)``

    user can also override ``key1`` with positional arguments as well:
    for ``self.submodule(x, 5, key2=6)``
    we'll get: ``(x, 5, 6)``

    variable positional arguments and variable positional keyword arguments in forward
    function are not supported currently, so please make sure no submodules is using
    them.
    """
    root = model
    fqn_to_example_inputs = {}

    def _patched_module_call(self, *args, **kwargs):
        submodule_example_inputs = list(args).copy()
        normalized_kwargs = _normalize_kwargs(self.forward, kwargs)
        # minus 1 to skipping counting `self`
        num_args = _get_num_pos_args(self.forward) - 1
        num_to_pop = num_args - len(submodule_example_inputs)
        while num_to_pop and normalized_kwargs:
            normalized_kwargs.popitem(last=False)
            num_to_pop -= 1
        submodule_example_inputs.extend(normalized_kwargs.values())
        submodule_example_inputs_tuple = tuple(submodule_example_inputs)
        fqn = _get_path_of_module(root, self)
        if fqn is not None:
            fqn_to_example_inputs[fqn] = submodule_example_inputs_tuple
        return orig_module_call(self, *args, **kwargs)

    orig_module_call = torch.nn.Module.__call__
    torch.nn.Module.__call__ = _patched_module_call  # type: ignore[method-assign]
    try:
        model(*example_inputs)
    finally:
        # restore the module call even if there is an exception
        torch.nn.Module.__call__ = orig_module_call  # type: ignore[method-assign]
    return fqn_to_example_inputs


def _assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    devices = {p.device for p in module.parameters()} | {
        p.device for p in module.buffers()
    }
    """
    As a temp workaround for AIMP HHC publish we added CPU check.remove it later. T163614564
    """
    if {torch.device("cpu"), torch.device("meta")} == devices:
        warnings.warn(
            "Both 'meta' and 'cpu' are present in the list of devices. Module can have one device. We Select 'cpu'.",
            stacklevel=2,
        )
        devices = {torch.device("cpu")}
    ""
    if len(devices) > 1:
        raise AssertionError(
            "prepare only works with cpu or single-device CUDA modules, "
            f"but got devices {devices}"
        )
    device = next(iter(devices)) if len(devices) > 0 else None
    return device


DEPRECATION_WARNING = (
    "torch.ao.quantization is deprecated and will be removed in 2.10. \n"
    "For migrations of users: \n"
    "1. Eager mode quantization (torch.ao.quantization.quantize, "
    "torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode "
    "quantize_ API instead \n"
    "2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx,"
    "torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization "
    "API instead (prepare_pt2e, convert_pt2e) \n"
    "3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e) \n"
    "see https://github.com/pytorch/ao/issues/2259 for more details"
)


__all__ = [
    "NodePattern",
    "Pattern",
    "MatchAllNode",
    "check_node",
    "get_combined_dict",
    "is_per_tensor",
    "is_per_channel",
    "getattr_from_fqn",
    "get_qparam_dict",
    "get_swapped_custom_module_class",
    "activation_dtype",
    "weight_dtype",
    "activation_is_statically_quantized",
    "activation_is_dynamically_quantized",
    "activation_is_int8_quantized",
    "activation_is_int32_quantized",
    "weight_is_quantized",
    "weight_is_statically_quantized",
    "op_is_int8_dynamically_quantized",
    "get_qconfig_dtypes",
    "get_quant_type",
    "check_min_max_valid",
    "calculate_qmin_qmax",
    "has_no_children_ignoring_parametrizations",
    "get_fqn_to_example_inputs",
    "to_underlying_dtype",
    "determine_qparams",
    "validate_qmin_qmax",
    "DEPRECATION_WARNING",
]
<|endoftext|># mypy: allow-untyped-defs
import copy

import torch.nn as nn

# for backward compatibility
from torch.ao.quantization.fuser_method_mappings import (  # noqa: F401  # noqa: F401
    fuse_conv_bn,
    fuse_conv_bn_relu,
    get_fuser_method,
)
from torch.nn.utils.parametrize import type_before_parametrizations


__all__ = [
    "fuse_known_modules",
    "fuse_modules",
    "fuse_modules_qat",
]


# Generalization of getattr
def _get_module(model, submodule_key):
    tokens = submodule_key.split(".")
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


# Generalization of setattr
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)


def fuse_known_modules(mod_list, is_qat, additional_fuser_method_mapping=None):
    r"""Return a list of known fuse modules.

    Returns a list of modules that fuses the operations specified
     in the input module list.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, bn
    linear, relu
    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """
    types = tuple(type_before_parametrizations(m) for m in mod_list)
    fuser_method = get_fuser_method(types, additional_fuser_method_mapping)
    if fuser_method is None:
        raise NotImplementedError(f"Cannot fuse modules: {types}")
    new_mod: list[nn.Module | None] = [None] * len(mod_list)
    fused = fuser_method(is_qat, *mod_list)
    # NOTE: forward hooks not processed in the two following for loops will be lost after the fusion
    # Move pre forward hooks of the base module to resulting fused module
    for pre_hook_fn in mod_list[0]._forward_pre_hooks.values():
        fused.register_forward_pre_hook(pre_hook_fn)
    mod_list[0]._forward_pre_hooks.clear()
    # Move post forward hooks of the last module to resulting fused module
    for hook_fn in mod_list[-1]._forward_hooks.values():
        fused.register_forward_hook(hook_fn)
    mod_list[-1]._forward_hooks.clear()
    new_mod[0] = fused

    for i in range(1, len(mod_list)):
        identity = nn.Identity()
        identity.training = mod_list[0].training
        new_mod[i] = identity

    return new_mod


def _fuse_modules_helper(
    model,
    modules_to_fuse,
    is_qat,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    if fuse_custom_config_dict is None:
        fuse_custom_config_dict = {}
    additional_fuser_method_mapping = fuse_custom_config_dict.get(
        "additional_fuser_method_mapping", {}
    )
    mod_list = [_get_module(model, item) for item in modules_to_fuse]

    # Fuse list of modules
    new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)

    # Replace original module list with fused module list
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])


def _fuse_modules(
    model,
    modules_to_fuse,
    is_qat,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    if not inplace:
        model = copy.deepcopy(model)

    if all(isinstance(module_element, str) for module_element in modules_to_fuse):
        # Handle case of modules_to_fuse being a list
        _fuse_modules_helper(
            model, modules_to_fuse, is_qat, fuser_func, fuse_custom_config_dict
        )
    else:
        # Handle case of modules_to_fuse being a list of lists
        for module_list in modules_to_fuse:
            _fuse_modules_helper(
                model, module_list, is_qat, fuser_func, fuse_custom_config_dict
            )
    return model


def fuse_modules(
    model,
    modules_to_fuse,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    r"""Fuse a list of modules into a single module.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    bn, relu
    All other sequences are left unchanged.
    For these sequences, replaces the first item in the list
    with the fused module, replacing the rest of the modules
    with identity.

    Args:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse. Can also be a list
                         of strings if there is only a single list of modules to fuse.
        inplace: bool specifying if fusion happens in place on the model, by default
                 a new model is returned
        fuser_func: Function that takes in a list of modules and outputs a list of fused modules
                    of the same length. For example,
                    fuser_func([convModule, BNModule]) returns the list [ConvBNModule, nn.Identity()]
                    Defaults to torch.ao.quantization.fuse_known_modules
        `fuse_custom_config_dict`: custom configuration for fusion

    .. code-block:: python

       # Example of fuse_custom_config_dict
       fuse_custom_config_dict = {
           # Additional fuser_method mapping
           "additional_fuser_method_mapping": {
               (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
           },
       }

    Returns:
        model with fused modules. A new copy is created if inplace=True.

    Examples::

            >>> # xdoctest: +SKIP
            >>> m = M().eval()
            >>> # m is a module containing the sub-modules below
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

            >>> m = M().eval()
            >>> # Alternately provide a single list of modules to fuse
            >>> modules_to_fuse = ['conv1', 'bn1', 'relu1']
            >>> fused_m = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

    """
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=False,
        inplace=inplace,
        fuser_func=fuser_func,
        fuse_custom_config_dict=fuse_custom_config_dict,
    )


def fuse_modules_qat(
    model,
    modules_to_fuse,
    inplace=False,
    fuser_func=fuse_known_modules,
    fuse_custom_config_dict=None,
):
    """QAT version for `fuse_modules`."""
    return _fuse_modules(
        model,
        modules_to_fuse,
        is_qat=True,
        inplace=inplace,
        fuser_func=fuser_func,
        fuse_custom_config_dict=fuse_custom_config_dict,
    )
<|endoftext|># mypy: allow-untyped-defs

import sys
from collections.abc import Callable
from typing_extensions import TypeAliasType

import torch
from torch import Tensor

from .fake_quantize import *  # noqa: F403
from .fuse_modules import fuse_modules, fuse_modules_qat
from .fuser_method_mappings import *  # noqa: F403
from .observer import *  # noqa: F403
from .qconfig import *  # noqa: F403
from .qconfig_mapping import *  # noqa: F403
from .quant_type import *  # noqa: F403
from .quantization_mappings import *  # noqa: F403 # type: ignore[no-redef]
from .quantize import *  # noqa: F403
from .quantize_jit import *  # noqa: F403
from .stubs import *  # noqa: F403


# ensure __module__ is set correctly for public APIs
ObserverOrFakeQuantize = TypeAliasType(
    "ObserverOrFakeQuantize", ObserverBase | FakeQuantizeBase
)


__all__ = [
    "DeQuantStub",
    "FakeQuantize",
    "FakeQuantizeBase",
    "FixedQParamsFakeQuantize",
    "FixedQParamsObserver",
    "FusedMovingAvgObsFakeQuantize",
    "HistogramObserver",
    # pyrefly: ignore [bad-dunder-all]
    "MatchAllNode",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "ObserverOrFakeQuantize",
    # pyrefly: ignore [bad-dunder-all]
    "Pattern",
    "PerChannelMinMaxObserver",
    "PlaceholderObserver",
    "QConfig",
    "QConfigAny",
    "QConfigDynamic",
    "QConfigMapping",
    "QuantStub",
    "QuantType",
    "QuantWrapper",
    "RecordingObserver",
    "ReuseInputObserver",
    "UniformQuantizationObserverBase",
    "add_quant_dequant",
    "convert",
    "convert_dynamic_jit",
    "convert_jit",
    "default_affine_fixed_qparams_fake_quant",
    "default_affine_fixed_qparams_observer",
    "default_debug_observer",
    "default_dynamic_fake_quant",
    "default_dynamic_quant_observer",
    "default_embedding_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_eval_fn",
    "default_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_fixed_qparams_range_0to1_observer",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_neg1to1_observer",
    "default_float_qparams_observer",
    "default_float_qparams_observer_4bit",
    "default_fused_act_fake_quant",
    "default_fused_per_channel_wt_fake_quant",
    "default_fused_wt_fake_quant",
    "default_histogram_fake_quant",
    "default_histogram_observer",
    "default_observer",
    "default_per_channel_weight_fake_quant",
    "default_per_channel_weight_observer",
    "default_placeholder_observer",
    "default_reuse_input_observer",
    "default_symmetric_fixed_qparams_fake_quant",
    "default_symmetric_fixed_qparams_observer",
    "default_weight_fake_quant",
    "default_weight_observer",
    "disable_fake_quant",
    "disable_observer",
    "enable_fake_quant",
    "enable_observer",
    "fuse_conv_bn",
    "fuse_conv_bn_jit",
    "fuse_conv_bn_relu",
    "fuse_convtranspose_bn",
    "fuse_linear_bn",
    "fuse_modules",
    "fuse_modules_qat",
    "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
    "fused_wt_fake_quant_range_neg_127_to_127",
    # pyrefly: ignore [bad-dunder-all]
    "get_combined_dict",
    "get_default_compare_output_module_list",
    "get_default_custom_config_dict",
    "get_default_dynamic_quant_module_mappings",
    "get_default_dynamic_sparse_quant_module_mappings",
    "get_default_float_to_quantized_operator_mappings",
    "get_default_qat_module_mappings",
    "get_default_qat_qconfig",
    "get_default_qat_qconfig_dict",
    "get_default_qat_qconfig_mapping",
    "get_default_qconfig",
    "get_default_qconfig_dict",
    "get_default_qconfig_mapping",
    "get_default_qconfig_propagation_list",
    "get_default_static_quant_module_mappings",
    "get_default_static_quant_reference_module_mappings",
    "get_default_static_sparse_quant_module_mappings",
    "get_dynamic_quant_module_class",
    "get_embedding_qat_module_mappings",
    "get_embedding_static_quant_module_mappings",
    "get_fuser_method",
    "get_fuser_method_new",
    "get_observer_state_dict",
    "get_quantized_operator",
    "get_static_quant_module_class",
    "load_observer_state_dict",
    "no_observer_set",
    "per_channel_weight_observer_range_neg_127_to_127",
    "prepare",
    "prepare_dynamic_jit",
    "prepare_jit",
    "prepare_qat",
    "propagate_qconfig_",
    "qconfig_equals",
    "quantize",
    "quantize_dynamic",
    "quantize_dynamic_jit",
    "quantize_jit",
    "quantize_qat",
    "script_qconfig",
    "script_qconfig_dict",
    "swap_module",
    "weight_observer_range_neg_127_to_127",
    # from torchao, should be merged with torchao
    # in the future
    "AffineQuantizedObserverBase",
    "Granularity",
    "MappingType",
    "PerAxis",
    "PerBlock",
    "PerGroup",
    "PerRow",
    "PerTensor",
    "PerToken",
    "TorchAODType",
    "ZeroPointDomain",
    "get_block_size",
]


def default_eval_fn(model, calib_data):
    r"""Define the default evaluation function.

    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, _target in calib_data:
        model(data)


class _DerivedObserverOrFakeQuantize(ObserverBase):
    r"""This observer is used to describe an observer whose quantization parameters
    are derived from other observers
    """

    def __init__(
        self,
        dtype: torch.dtype,
        obs_or_fqs: list[ObserverOrFakeQuantize],
        derive_qparams_fn: Callable[
            [list[ObserverOrFakeQuantize]], tuple[Tensor, Tensor]
        ],
        quant_min: int | None = None,
        quant_max: int | None = None,
        qscheme: torch.qscheme | None = None,
        ch_axis: int | None = None,
    ):
        super().__init__(dtype)
        self.obs_or_fqs = obs_or_fqs
        self.derive_qparams_fn = derive_qparams_fn
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.qscheme = qscheme
        self.ch_axis = ch_axis

        from .utils import is_per_channel

        if is_per_channel(self.qscheme):
            if self.ch_axis is None:
                raise AssertionError(
                    "Must provide a valid ch_axis if qscheme is per channel"
                )

    def forward(self, x: Tensor) -> Tensor:
        return x

    def calculate_qparams(self):  # type:ignore[override]
        return self.derive_qparams_fn(self.obs_or_fqs)
<|endoftext|># mypy: allow-untyped-defs
import itertools
from collections.abc import Callable
from typing import Any

import torch.ao.nn.intrinsic as nni
import torch.nn as nn
from torch.ao.quantization.utils import get_combined_dict, MatchAllNode, Pattern


__all__ = [
    "fuse_conv_bn",
    "fuse_conv_bn_relu",
    "fuse_linear_bn",
    "fuse_convtranspose_bn",
    "get_fuser_method",
    "get_fuser_method_new",
]


def fuse_conv_bn(is_qat, conv, bn):
    r"""Return the fused conv and bn modules.
    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    if conv.training != bn.training:
        raise AssertionError(
            "Conv and BN both must be in the same mode (train or eval)."
        )

    fused_module_class_map = {
        nn.Conv1d: nni.ConvBn1d,
        nn.Conv2d: nni.ConvBn2d,
        nn.Conv3d: nni.ConvBn3d,
    }

    if is_qat:
        if bn.num_features != conv.out_channels:
            raise AssertionError(
                "Output channel of Conv2d must match num_features of BatchNorm2d."
            )
        if not bn.affine:
            raise AssertionError(
                "Only support fusing BatchNorm2d with affine set to True"
            )
        if not bn.track_running_stats:
            raise AssertionError(
                "Only support fusing BatchNorm2d with tracking_running_stats set to True"
            )
        fused_module_class = fused_module_class_map.get(type(conv))
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn)}")
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)


def fuse_conv_bn_relu(is_qat, conv, bn, relu):
    r"""Return the fused conv and bn modules.

    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> r1 = nn.ReLU(inplace=False)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn_relu(m1, b1, r1)
    """
    if not (conv.training == bn.training == relu.training):
        raise AssertionError(
            "Conv and BN both must be in the same mode (train or eval)."
        )
    fused_module: type[nn.Sequential] | None = None
    if is_qat:
        map_to_fused_module_train = {
            nn.Conv1d: nni.ConvBnReLU1d,
            nn.Conv2d: nni.ConvBnReLU2d,
            nn.Conv3d: nni.ConvBnReLU3d,
        }
        if bn.num_features != conv.out_channels:
            raise AssertionError(
                "Output channel of Conv2d must match num_features of BatchNorm2d"
            )
        if not bn.affine:
            raise AssertionError(
                "Only support fusing BatchNorm2d with affine set to True"
            )
        if not bn.track_running_stats:
            raise AssertionError(
                "Only support fusing BatchNorm2d with tracking_running_stats set to True"
            )
        fused_module = map_to_fused_module_train.get(type(conv))
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, relu)}")
    else:
        map_to_fused_module_eval = {
            nn.Conv1d: nni.ConvReLU1d,
            nn.Conv2d: nni.ConvReLU2d,
            nn.Conv3d: nni.ConvReLU3d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv))
        if fused_module is not None:
            fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        else:
            raise NotImplementedError(f"Cannot fuse eval modules: {(conv, bn, relu)}")


def fuse_linear_bn(is_qat, linear, bn):
    r"""Return the fused linear and bn modules.
    Given the linear and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer

    Examples::

        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_linear_bn(m1, b1)
    """
    if linear.training != bn.training:
        raise AssertionError(
            "Linear and BN both must be in the same mode (train or eval)."
        )

    if is_qat:
        if bn.num_features != linear.out_features:
            raise AssertionError(
                "Output features of Linear must match num_features of BatchNorm1d"
            )
        if not bn.affine:
            raise AssertionError(
                "Only support fusing BatchNorm1d with affine set to True"
            )
        if not bn.track_running_stats:
            raise AssertionError(
                "Only support fusing BatchNorm1d with tracking_running_stats set to True"
            )
        return nni.LinearBn1d(linear, bn)
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)


def fuse_convtranspose_bn(is_qat, convt, bn):
    r"""Return the fused ConvTranspose and bn modules.
    Given ConvTranspose and bn modules, fuses them and returns the fused module

    Args:
        convt: Module instance of type ConvTransposeNd
        bn: BatchNormNd instance that needs to be fused with the linear layer.
            batch norm N should match the ConvTranspose N

    Examples::

        >>> m1 = nn.ConvTranspose2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_convtranspose_bn(m1, b1)
    """
    if convt.training != bn.training:
        raise AssertionError(
            "ConvTranspose and BN both must be in the same mode (train or eval)."
        )

    if is_qat:
        raise Exception(  # noqa: TRY002
            "Fusing ConvTranspose+BatchNorm not yet supported in QAT."
        )
    else:
        return nn.utils.fusion.fuse_conv_bn_eval(convt, bn, transpose=True)


def _sequential_wrapper2(sequential):
    """Return a sequential wrapped that for is_qat and two modules.
    Given a sequential class for two modules, return a function that takes
    is_qat, and then two modules as argument, that ignores the is_qat flag
    and always returns the sequential that combines the two input modules
    """

    def fuser_method(is_qat, m1, m2):
        return sequential(m1, m2)

    return fuser_method


_DEFAULT_OP_LIST_TO_FUSER_METHOD: dict[tuple, nn.Sequential | Callable] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU1d),
    (nn.Conv2d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU2d),
    (nn.Conv3d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU3d),
    (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
    (nn.Linear, nn.ReLU): _sequential_wrapper2(nni.LinearReLU),
    (nn.BatchNorm2d, nn.ReLU): _sequential_wrapper2(nni.BNReLU2d),
    (nn.BatchNorm3d, nn.ReLU): _sequential_wrapper2(nni.BNReLU3d),
    (nn.ConvTranspose1d, nn.BatchNorm1d): fuse_convtranspose_bn,
    (nn.ConvTranspose2d, nn.BatchNorm2d): fuse_convtranspose_bn,
    (nn.ConvTranspose3d, nn.BatchNorm3d): fuse_convtranspose_bn,
}


def get_fuser_method(op_list, additional_fuser_method_mapping=None):
    """Get fuser method for the given list of module types.

    Get fuser method for the given list of module types,
    return None if fuser method does not exist
    """
    if additional_fuser_method_mapping is None:
        additional_fuser_method_mapping = {}
    all_mappings = get_combined_dict(
        _DEFAULT_OP_LIST_TO_FUSER_METHOD, additional_fuser_method_mapping
    )
    fuser_method = all_mappings.get(op_list, None)
    if fuser_method is None:
        raise AssertionError(f"did not find fuser method for: {op_list} ")
    return fuser_method


def _reverse2(f):
    def reversed(is_qat, x, y):
        return f(is_qat, y, x)

    return reversed


def _reverse3(f):
    def reversed(is_qat, x, w):
        y, z = w
        return f(is_qat, z, y, x)

    return reversed


def _get_valid_patterns(op_pattern):
    """Return a list of valid patterns generated from the op_pattern.

    Returns a list of valid patterns generated from the op_pattern,
    since MatchAllNode can match all types of nodes,
    e.g. pattern (torch.nn.Conv2d, torch.add) should also be able to match keys like
    (MatchAllNode, torch.add) and (torch.nn.Conv2d, MatchAllNode)

    Example Input:
    (torch.add, (torch.nn.ReLU, torch.nn.Conv2d))

    Example Output:
    [(torch.add, (torch.nn.ReLU, torch.nn.Conv2d)),
     (torch.add, (torch.nn.ReLU, MatchAllNode)),
     (torch.add, (MatchAllNode, torch.nn.Conv2d)),
     (torch.add, (MatchAllNode, MatchAllNode)),
     (MatchAllNode, (torch.nn.ReLU, torch.nn.Conv2d)),
     (MatchAllNode, (torch.nn.ReLU, MatchAllNode)),
     (MatchAllNode, (MatchAllNode, torch.nn.Conv2d)),
     (MatchAllNode, (MatchAllNode, MatchAllNode)),
    ]
    """
    result: list[Any]
    if isinstance(op_pattern, (tuple, list)):
        sub_combs = [_get_valid_patterns(sub_pattern) for sub_pattern in op_pattern]
        result = list(itertools.product(*sub_combs))
    else:
        result = [op_pattern, MatchAllNode]
    return result


def get_fuser_method_new(
    op_pattern: Pattern,
    fuser_method_mapping: dict[Pattern, nn.Sequential | Callable],
):
    """Get fuser method.

    This will be made default after we deprecate the get_fuser_method
    Would like to implement this first and have a separate PR for deprecation
    """
    op_patterns = _get_valid_patterns(op_pattern)
    fuser_method = None
    for op_pattern in op_patterns:
        fuser_method = fuser_method_mapping.get(op_pattern)
        if fuser_method is not None:
            break
    if fuser_method is None:
        raise AssertionError(f"did not find fuser method for: {op_pattern} ")
    return fuser_method
<|endoftext|>import enum


__all__ = [
    "QuantType",
]


# Quantization type (dynamic quantization, static quantization).
# Should match the c++ enum in quantization_type.h
class QuantType(enum.IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    WEIGHT_ONLY = 3


_quant_type_to_str = {
    QuantType.STATIC: "static",
    QuantType.DYNAMIC: "dynamic",
    QuantType.QAT: "qat",
    QuantType.WEIGHT_ONLY: "weight_only",
}


# TODO: make this private
def _get_quant_type_to_str(quant_type: QuantType) -> str:
    return _quant_type_to_str[quant_type]


def _quant_type_from_str(name: str) -> QuantType:
    for quant_type, s in _quant_type_to_str.items():
        if name == s:
            return quant_type
    raise ValueError(f"Unknown QuantType name '{name}'")
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.ao.nn.quantized as nnq
import torch.ao.ns._numeric_suite as ns
import torch.ao.quantization
import torch.nn as nn


__all__ = [
    "get_module",
    "parent_child_names",
    "get_param",
    "MeanShadowLogger",
    "bias_correction",
]

_supported_modules = {nn.Linear, nn.Conv2d}
_supported_modules_quantized = {nnq.Linear, nnq.Conv2d}


def get_module(model, name):
    """Given name of submodule, this function grabs the submodule from given model."""
    return dict(model.named_modules())[name]


def parent_child_names(name):
    """Split full name of submodule into parent submodule's full name and submodule's name."""
    split_name = name.rsplit(".", 1)
    if len(split_name) == 1:
        return "", split_name[0]
    else:
        return split_name[0], split_name[1]


def get_param(module, attr):
    """Get the parameter given a module and attribute.

    Sometimes the weights/bias attribute gives you the raw tensor, but sometimes
    gives a function that will give you the raw tensor, this function takes care of that logic
    """
    param = getattr(module, attr, None)
    if callable(param):
        return param()
    else:
        return param


class MeanShadowLogger(ns.Logger):
    """Mean Logger for a Shadow module.

    A logger for a Shadow module whose purpose is to record the rolling mean
    of the data passed to the floating point and quantized models
    """

    def __init__(self):
        """Set up initial values for float and quantized stats, count, float sum, and quant sum."""
        super().__init__()
        self.stats["float"] = None
        self.stats["quantized"] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None

    def forward(self, x, y):  # type: ignore[override]
        """Compute the average of quantized and floating-point data from modules.

        The inputs x,y are output data from the quantized and floating-point modules.
        x is for the quantized module, y is for the floating point module
        """
        if x.is_quantized:
            x = x.dequantize()

        self.count += 1
        if self.stats["quantized"] is None:
            self.stats["quantized"] = x
            self.quant_sum = x
        else:
            self.quant_sum += x
            self.stats["quantized"] = self.quant_sum / self.count

        if self.stats["float"] is None:
            self.stats["float"] = y
            self.float_sum = y
        else:
            self.float_sum += y
            self.stats["float"] = self.float_sum / self.count

    def clear(self):
        self.stats["float"] = None
        self.stats["quantized"] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None


def bias_correction(
    float_model,
    quantized_model,
    img_data,
    target_modules=_supported_modules_quantized,
    neval_batches=None,
):
    """Perform bias correction on a module.

    Using numeric suite shadow module, the expected output of the floating point and quantized modules
    is recorded. Using that data the bias of supported modules is shifted to compensate for the drift caused
    by quantization
    Paper reference: https://arxiv.org/pdf/1906.04721.pdf (Section 4.2)

    Args:
        float_model: a trained model that serves as a reference to what bias correction should aim for
        quantized_model: quantized form of float_model that bias correction is to applied to
        img_data: calibration data to estimate the expected output (used to find quantization error)
        target_modules: specifies what submodules in quantized_model need bias correction (can be extended to
                unquantized submodules)
        neval_batches: a cap to the number of batches you want to be used for estimating the expected output
    """
    ns.prepare_model_with_stubs(
        float_model, quantized_model, _supported_modules, MeanShadowLogger
    )

    uncorrected_modules = {
        name: submodule
        for name, submodule in quantized_model.named_modules()
        if type(submodule) in target_modules
    }

    for uncorrected_module in uncorrected_modules:
        quantized_submodule = get_module(quantized_model, uncorrected_module)
        bias = get_param(quantized_submodule, "bias")
        if bias is not None:
            for count, data in enumerate(img_data, start=1):
                quantized_model(data[0])
                if count == neval_batches:
                    break
            ob_dict = ns.get_logger_dict(quantized_model)
            parent_name, _ = parent_child_names(uncorrected_module)

            float_data = ob_dict[parent_name + ".stats"]["float"]
            quant_data = ob_dict[parent_name + ".stats"]["quantized"]

            # math for expected_error
            quantization_error = quant_data - float_data
            dims = list(range(quantization_error.dim()))
            # Note: we don't want to take the mean over the output channel dimension
            dims.remove(1)
            expected_error = torch.mean(quantization_error, dims)

            updated_bias = bias.data - expected_error

            bias.data = updated_bias

            # Resets the data contained in the loggers
            for submodule in quantized_model.modules():
                if isinstance(submodule, MeanShadowLogger):
                    submodule.clear()
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# temporarily skip RUF for this file for now, we can re-enable
# after move the affine quantization related things to torchao
# noqa: RUF
"""
This module implements observers which are used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import operator
import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
    calculate_qmin_qmax,
    check_min_max_valid,
    is_per_channel,
    is_per_tensor,
    validate_qmin_qmax,
)
from torch.fx import Node


__all__ = [
    "default_affine_fixed_qparams_observer",
    "default_debug_observer",
    "default_dynamic_quant_observer",
    "default_fixed_qparams_range_0to1_observer",
    "default_fixed_qparams_range_neg1to1_observer",
    "default_float_qparams_observer",
    "default_float_qparams_observer_4bit",
    "default_histogram_observer",
    "default_observer",
    "default_per_channel_weight_observer",
    "default_placeholder_observer",
    "default_reuse_input_observer",
    "default_symmetric_fixed_qparams_observer",
    "default_weight_observer",
    "get_observer_state_dict",
    "load_observer_state_dict",
    "per_channel_weight_observer_range_neg_127_to_127",
    "weight_observer_range_neg_127_to_127",
    "FixedQParamsObserver",
    "HistogramObserver",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "PerChannelMinMaxObserver",
    "PlaceholderObserver",
    "RecordingObserver",
    "ReuseInputObserver",
    "UniformQuantizationObserverBase",
    "AffineQuantizedObserverBase",
    "Granularity",
    "MappingType",
    "PerAxis",
    "PerBlock",
    "PerGroup",
    "PerRow",
    "PerTensor",
    "PerToken",
    "TorchAODType",
    "ZeroPointDomain",
    "get_block_size",
]


class _PartialWrapper:
    def __init__(self, p):
        self.p = p
        self.callable_args = {}

    def __call__(self, *args, **keywords):
        # call each arg in callable_args and add them partial, then run with keywords
        # skip if arg_name in keywords so its possible to overwrite
        for arg_name in self.callable_args:
            if arg_name not in keywords:
                keywords = {**keywords, arg_name: self.callable_args[arg_name]()}
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__() + self.callable_args.__repr__()

    def with_args(self, **kwargs):
        return _with_args(self, **kwargs)

    def with_callable_args(self, **kwargs):
        result = _PartialWrapper(p=self.p)
        result.callable_args = {**self.callable_args, **kwargs}
        return result


def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances. Can be used in conjunction with
    _callable_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


def _with_callable_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories args that need to be
    called at construction time.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances and those arguments should only
    be calculated at construction time. Can be used in conjunction with _with_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_callable_args = classmethod(_with_callable_args)
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_callable_args(cur_time=get_time_func).with_args(name="dan")
        >>> foo_instance1 = foo_builder()
        >>> # wait 50
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1.creation_time) == id(foo_instance2.creation_time)
        False
    """
    r = _PartialWrapper(partial(cls_or_self))
    return r.with_callable_args(**kwargs)


ABC: Any = ABCMeta("ABC", (object,), {})


class ObserverBase(ABC, nn.Module):
    r"""Base observer Module.
    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        is_dynamic: indicator for whether the observer is a placeholder for dynamic quantization
        or static quantization
    """

    def __init__(self, dtype, is_dynamic: bool = False):
        super().__init__()
        self.dtype = dtype
        self.is_dynamic = is_dynamic

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    with_args = classmethod(_with_args)
    with_callable_args = classmethod(_with_callable_args)


class UniformQuantizationObserverBase(ObserverBase):
    r"""Common base for all observers using uniform quantization to calculate
    scale and zero_point.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.
               or `torch.int8` or `torch.uint8`

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """

    # Note: the version is shared by all observer types
    #
    # Version 1/None
    #   self
    #
    # Version 2 (base class only, does not include child class buffers)
    #   self
    #   |--- eps : Tensor
    #
    # Version 3
    #   for HistogramObserver only, changed the shape of uninitialized
    #   min_val and max_val buffers from torch.Size([0]) to torch.Size([])
    #   for PerChannelObservers, changed the name of the buffers from min_vals
    #   to min_val and from max_vals to max_val.
    _version = 3

    eps: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.qscheme = qscheme
        if reduce_range:
            warnings.warn(
                "Please use quant_min and quant_max to specify the range for observers. \
                    reduce_range will be deprecated in a future release of PyTorch.",
                stacklevel=2,
            )
        self.reduce_range = reduce_range
        self.register_buffer("eps", torch.tensor([eps], **factory_kwargs))
        if self.qscheme not in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ):
            raise AssertionError(
                "Default Observer only works for per_tensor_affine, per_tensor_symmetric, "
                "per_channel_affine, per_channel_symmetric and per_channel_float_qparams quantization scheme"
            )

        _ALLOWED_DTYPES = (
            torch.qint8,
            torch.quint8,
            torch.quint4x2,
            torch.qint32,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.uint16,
        )

        if self.dtype not in _ALLOWED_DTYPES:
            raise AssertionError(
                f"Default Observer only works for {_ALLOWED_DTYPES} data type"
            )
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            # pyrefly: ignore [bad-argument-type]
            validate_qmin_qmax(quant_min, quant_max)
        self.quant_min, self.quant_max = calculate_qmin_qmax(
            # pyrefly: ignore [bad-argument-type]
            quant_min,
            # pyrefly: ignore [bad-argument-type]
            quant_max,
            self.has_customized_qrange,
            self.dtype,
            self.reduce_range,
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
    ):
        version = local_metadata.get("version", None)

        if version is None or version == 1:
            # eps was moved to a buffer in version 2
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + "eps"] = eps

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min: int, quant_max: int) -> None:
        r"""Validates that the user-specified quantization range is properly initialized
        and within the given bound supported by the observer dtype.

        To accommodate lower-bit quantization with respect to the existing torch.qint8 and
        torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
        in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
        values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
        fake quantization. These estimates are compared against parameters learned through backpropagation.
        The related literatures for scale and zero point via backpropagation are as follows:

        Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
        Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
        """
        # The variable names are prefixed with "initial" because their values (qmin and qmax) might be adjusted
        # based on whether quantization range is reduced and the datatype (signed/unsigned) used by the observer.
        if not quant_min <= 0 <= quant_max:
            raise AssertionError("Used-specified quantization range must include 0.")
        if quant_min >= quant_max:
            raise AssertionError(
                "qmin must be strictly less than qmax for user-specified quantization range."
            )

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        # Functionally equivalent to 'determine_qparams' in utils.py. Observers must be torchscriptable however and qscheme
        # as far as I can tell is not allowed to passed as a parameter in torchscript functions. This makes refactoring observer
        # to use this utility a massive pain and very gross. For now Im opting just to duplicate as this code
        # seems unlikely to change (last update over 1 year ago) and when torchscript is fully deprecated we can refactor.
        # TODO(jakeszwe, jerryzh168)
        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor(
                [0], device=min_val.device.type
            )

        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if (
            self.qscheme == torch.per_tensor_symmetric
            or self.qscheme == torch.per_channel_symmetric
        ):
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            if self.dtype in [torch.quint8, torch.uint8]:
                if self.has_customized_qrange:
                    # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                    zero_point = zero_point.new_full(
                        zero_point.size(), (quant_min + quant_max) // 2
                    )
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
            elif self.dtype == torch.uint16:
                zero_point = zero_point.new_full(zero_point.size(), 2**15)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (max_val - min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            # We use the quantize function
            # xq = Round(Xf * inv_scale + zero_point),
            # setting zero_point to (-1 * min *inv_scale) we get
            # Xq = Round((Xf - min) * inv_scale)
            zero_point = -1 * min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor(
                [int(zero_point)], dtype=zero_point.dtype, device=device
            )
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor(
                    [float(zero_point)], dtype=zero_point.dtype, device=device
                )

        return scale, zero_point

    @torch.jit.export
    def reset_min_max_vals(self):
        raise NotImplementedError("Cannot reset min/max values in the given observer.")


# Originally, this class was called `_ObserverBase`.  Keeping the old name around
# for backwards compatibility.
# TODO(after v1.13): delete this
_ObserverBase = UniformQuantizationObserverBase


class MinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.

    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    Given running min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    The running minimum/maximum :math:`x_\text{min/max}` is computed as:

    .. math::

        \begin{array}{ll}
        x_\text{min} &= \begin{cases}
            \min(X) & \text{if~}x_\text{min} = \text{None} \\
            \min\left(x_\text{min}, \min(X)\right) & \text{otherwise}
        \end{cases}\\
        x_\text{max} &= \begin{cases}
            \max(X) & \text{if~}x_\text{max} = \text{None} \\
            \max\left(x_\text{max}, \max(X)\right) & \text{otherwise}
        \end{cases}\\
        \end{array}

    where :math:`X` is the observed tensor.

    The scale :math:`s` and zero point :math:`z` are then computed as:

    .. math::

        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}

    where :math:`Q_\text{min}` and :math:`Q_\text{max}` are the minimum and
    maximum of the quantized data type.

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "MinMaxObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        # TODO: MinMaxObserver by itself doesn't support dynamic quantization, but
        # if it's inherited by MovingAverageObserver, and averaging_constant is 1, it
        # supports dynamic quantization, we may need to better error checking here

        # For x86 quantized kernels, we need to ensure that the vpmaddubsw
        # instruction does not overflow. We allow for a reduce_range argument to
        # observers that reduces the quantized range to (0,127) or (-64, 63).
        # For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
        # This is not an optimal choice for non x86 backends as it loses a bit
        # of precision for activations.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        if (
            self.qscheme == torch.per_tensor_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric \
                                       quantization for quint8"
            )

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))


class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The moving average min/max is computed as follows

    .. math::

        \begin{array}{ll}
                x_\text{min} = \begin{cases}
                    \min(X) & \text{if~}x_\text{min} = \text{None} \\
                    (1 - c) x_\text{min} + c \min(X) & \text{otherwise}
                \end{cases}\\
                x_\text{max} = \begin{cases}
                    \max(X) & \text{if~}x_\text{max} = \text{None} \\
                    (1 - c) x_\text{max} + c \max(X) & \text{otherwise}
                \end{cases}\\
        \end{array}

    where :math:`x_\text{min/max}` is the running average min/max, :math:`X`
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                f"MovingAverageMinMaxObserver's qscheme only support \
                torch.per_tensor_symmetric and torch.per_tensor_affine. \
                but got: {qscheme}"
            )
        self.averaging_constant = averaging_constant
        if is_dynamic and self.averaging_constant != 1:
            raise NotImplementedError(
                "MovingAverageMinMaxObserver doesn't support dynamic quantization for "
                f"averaging constant of {self.averaging_constant}"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float("inf") and max_val == float("-inf"):
            min_val, max_val = torch.aminmax(x)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "PerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        if is_dynamic:
            raise NotImplementedError(
                "PerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self._calculate_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, torch.Tensor],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        version = local_metadata.get("version")
        if version is not None and version < 3:
            local_state = ["min_vals", "max_vals"]
            expected_min_name = "min_vals"
            expected_max_name = "max_vals"
        else:
            local_state = ["min_val", "max_val"]
            expected_min_name = "min_val"
            expected_max_name = "max_val"
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == expected_min_name:
                    self.min_val.resize_(val.shape)
                elif name == expected_max_name:
                    self.max_val.resize_(val.shape)
                else:
                    warnings.warn(
                        f"Observer load_from_state_dict got unexpected name {name}",
                        stacklevel=2,
                    )
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == expected_min_name:
                        self.min_val.copy_(val)
                    elif name == expected_max_name:
                        self.max_val.copy_(val)
                    else:
                        warnings.warn(
                            f"Observer load_from_state_dict got unexpected name {name}",
                            stacklevel=2,
                        )
            elif strict:
                missing_keys.append(key)

        if not torch.jit.is_scripting():
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def _load_from_state_dict_script(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, torch.Tensor],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        self._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        # This used to be torch.ones but that does not work because
        # JIT compiler can optimize it via common subexpression elimination
        # in which case both min_val and max_val point to the same tensor.
        self.min_val = torch.rand(
            0,
        )
        self.max_val = torch.rand(
            0,
        )


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MovingAverageMinMaxObserver`, with the
    difference that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        if is_dynamic:
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class HistogramObserver(UniformQuantizationObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.ao.quantization.MinMaxObserver`
    """

    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        bins: int = 2048,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "HistogramObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        if is_dynamic:
            raise NotImplementedError(
                "HistogramObserver doesn't support dynamic quantization"
            )
        # bins: The number of bins used for histogram calculation.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer("histogram", torch.zeros(self.bins, **factory_kwargs))
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
        self.upsample_rate = (
            16  # used to reduce quantization errors when upscaling histogram
        )

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(
            delta_begin,
            torch.ones(self.bins, device=self.histogram.device) * delta_end,
            density,
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        if self.histogram.size()[0] != self.bins:
            raise AssertionError("bins mismatch")
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _upscale_histogram(
        self,
        histogram: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
    ):
        # this turns the histogram into a more fine-coarsed histogram to reduce
        # bin quantization errors
        histogram = histogram.repeat_interleave(self.upsample_rate) / self.upsample_rate
        bin_size = (orig_max - orig_min) / (self.bins * self.upsample_rate)
        mid_points_histogram = (
            torch.linspace(
                orig_min,
                orig_max,
                self.bins * self.upsample_rate + 1,
                device=orig_min.device,
            )[:-1].to(histogram.device)
            + 0.5 * bin_size
        )
        boundaries_new_histogram = torch.linspace(
            update_min, update_max, self.bins + 1, device=update_min.device
        ).to(histogram.device)
        # this maps the mid-points of the histogram to the new histogram's space
        bucket_assignments = (
            torch.bucketize(mid_points_histogram, boundaries_new_histogram, right=True)
            - 1
        )
        # this then maps the histogram mid-points in the new space, weighted by the original histogram's values
        # this is just the old histogram in the new histogram's space

        # In case due to numerical issues the values land higher/lower than the maximum/minimum
        bucket_assignments[bucket_assignments >= self.bins] = self.bins - 1
        bucket_assignments[bucket_assignments < 0] = 0

        update_histogram = torch.bincount(
            bucket_assignments, weights=histogram, minlength=self.bins
        )
        return update_histogram

    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_hist: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
    ) -> torch.Tensor:
        # If the new min and max are the same as the current min and max,
        # we can just add the new histogram to the original histogram
        if update_min == orig_min and update_max == orig_max:
            return orig_hist + update_hist

        # If the orig hist only has one value (i.e., the min and max are the same)
        # we can just add it into new histogram
        if orig_min == orig_max:
            bin_value = torch.sum(orig_hist)
            transformed_orig_hist = (
                torch.histc(orig_min, bins=self.bins, min=update_min, max=update_max)  # type: ignore[arg-type]
                * bin_value
            )
            return transformed_orig_hist + update_hist

        # We assume the update_hist is already in the target range, we will map the orig_max to it
        if update_min > orig_min:
            raise AssertionError("update_min must be <= orig_min")
        if update_max < orig_max:
            raise AssertionError("update_max must be >= orig_max")

        # Now we need to turn the old_histogram, into the range of the new histogram
        transformed_orig_hist = self._upscale_histogram(
            orig_hist,
            orig_min,
            orig_max,
            update_min,
            update_max,
        )

        return update_hist + transformed_orig_hist

    def reset_histogram(
        self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> None:
        self.min_val.resize_(min_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)
        if min_val.numel() != 1 or max_val.numel() != 1:
            raise AssertionError("histogram min/max values must be scalar.")
        new_histogram = torch.histc(x, self.bins, min=min_val, max=max_val)  # type: ignore[arg-type]
        self.histogram.detach_().resize_(new_histogram.shape)
        self.histogram.copy_(new_histogram)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:  # pyre-ignore[14]
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x_min, x_max = torch.aminmax(x)
        # want to ignore torch.inf since we don't actually
        # want to make our quantization range infinite
        # and in practice those values will be clamped
        if x_min == -torch.inf or x_max == torch.inf:
            warnings.warn(
                "torch.inf detected in input tensor, ignoring input", stacklevel=2
            )
            x = x[x.abs() != torch.inf]
            if x.numel() == 0:
                return x_orig
            x_min, x_max = torch.aminmax(x)

        current_min = self.min_val
        current_max = self.max_val

        is_uninitialized = self.min_val == float("inf") or self.max_val == float("-inf")
        if is_uninitialized:
            self.reset_histogram(x, x_min, x_max)
        else:
            update_min, update_max = x_min, x_max
            new_min = torch.min(current_min, update_min)
            new_max = torch.max(current_max, update_max)

            # TODO: For some reason, this is required for it to pass torchscript test
            # new_min and new_max should already have requires_grad set to False
            new_min, new_max = new_min.detach(), new_max.detach()
            update_histogram = torch.histc(
                x,
                self.bins,
                min=new_min,  # type: ignore[arg-type]
                max=new_max,  # type: ignore[arg-type]
            ).to(self.histogram.device)
            if new_min == current_min and new_max == current_max:
                combined_histogram = self.histogram + update_histogram
                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
            else:
                combined_histogram = self._combine_histograms(
                    self.histogram,
                    current_min,
                    current_max,
                    update_histogram,
                    new_min,
                    new_max,
                )
                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
                self.min_val.detach_().resize_(new_min.shape)
                self.min_val.copy_(new_min)
                self.max_val.detach_().resize_(new_max.shape)
                self.max_val.copy_(new_max)

        return x_orig

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point ",
                stacklevel=2,
            )
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor(
                [0], device=self.min_val.device.type
            )
        if self.bins != len(self.histogram):
            raise AssertionError(
                "The number of bins in histogram should be equal to the number of bins "
                "supplied while making this observer"
            )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "min_val"] = self.min_val
        destination[prefix + "max_val"] = self.max_val

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float("inf"))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float("-inf"))

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"


class FixedQParamsObserver(ObserverBase):
    r"""
    Observer that simulates quantize and dequantize with fixed
    quantization parameters in training time. Only per tensor
    quantization is supported.

    Args:
        `scale` (float): fixed scale for the observer
        `zero_point` (int): fixed zero point for the observer
        `dtype`, `qscheme`, `quant_min`, `quant_max`
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        scale,
        zero_point,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
        is_dynamic=False,
        **kwargs,
    ):
        if is_dynamic:
            raise NotImplementedError(
                "FixedQParamsObserver doesn't support dynamic quantization"
            )
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.register_buffer("scale", torch.tensor([scale], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([zero_point], dtype=torch.int))
        self.dtype = dtype
        self.qscheme = qscheme

    def forward(self, X):
        return X

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        return self.scale, self.zero_point


class PlaceholderObserver(ObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Can be used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        quant_min: minimum value in quantized domain (TODO: align behavior with other observers)
        quant_max: maximum value in quantized domain
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
        compute_dtype (deprecated): if set, marks the future quantize function to use
                       dynamic quantization instead of static quantization.
                       This field is deprecated, use `is_dynamic=True` instead.
        is_dynamic: if True, the `quantize` function in the reference model
                    representation taking stats from this observer instance will
                    use dynamic quantization.
    """

    def __init__(
        self,
        dtype=torch.float32,
        custom_op_name="",
        compute_dtype=None,
        quant_min=None,
        quant_max=None,
        qscheme=None,
        eps=None,
        is_dynamic=False,
    ) -> None:
        super().__init__(dtype=dtype, is_dynamic=is_dynamic)
        if qscheme is None:
            qscheme = torch.per_tensor_affine
        if eps is None:
            eps = torch.finfo(torch.float32).eps

        # dtype of input of the target operator, e.g. for dynamic quantization
        # ops, the dtype will be float32
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.custom_op = custom_op_name
        # used for configuration of computation type for dynamic quantization
        if compute_dtype:
            is_dynamic = True
            warnings.warn(
                "Please use `is_dynamic` instead of `compute_dtype`. \
                    `compute_dtype` will be deprecated in a future release \
                    of PyTorch.",
                stacklevel=2,
            )

    def forward(self, x):
        return x

    @torch.jit.export
    def extra_repr(self):
        return f"dtype={self.dtype}, is_dynamic={self.is_dynamic}"

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for PlaceholderObserver"
        )


class RecordingObserver(ObserverBase):
    r"""
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    """

    __annotations__ = {"tensor_val": list[torch.Tensor | None]}

    def __init__(self, dtype=torch.quint8):
        super().__init__(dtype=dtype, is_dynamic=False)
        self.tensor_val = []

    def forward(self, x):
        self.tensor_val.append(x.clone())
        return x

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for RecordingObserver"
        )

    @torch.jit.export
    def get_tensor_value(self):
        return self.tensor_val


class NoopObserver(ObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Primarily used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: Quantized data type
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
    """

    def __init__(self, dtype=torch.float16, custom_op_name="") -> None:
        super().__init__(dtype=dtype, is_dynamic=False)
        self.dtype = dtype
        self.custom_op = custom_op_name

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for NoopObserver"
        )


class ReuseInputObserver(ObserverBase):
    r"""This observer is used when we want to reuse the observer from the operator
    that produces the input Tensor, typically used for operators like reshape, e.g.
    ```
    x0 = ...
    x1 = x0.reshape()
    ```
    if we configure x0 to be observed by some observer, let's say MinMaxObserver,
    and reshape is configured with ReuseInputObserver, we'll reuse the observer instance
    for x0 for x1 (output of reshape). If x0 is not observed, we also won't observe x1.

    Note: this is only enabled in FX Graph Mode Quantization
    """

    def __init__(self) -> None:
        super().__init__(torch.quint8, is_dynamic=False)

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for ReuseInputObserver"
        )


"""
# Experimental Affine Quantization Feature START
We plan to merge the following with torchao repo after we move pt2e flow to torchao
copied from https://github.com/pytorch/ao/blob/main/torchao/quantization/observer.py
"""
from dataclasses import dataclass
from enum import auto, Enum


class MappingType(Enum):
    """How floating point number is mapped to integer number

    symmetric mapping means floating point range is symmetrically mapped to integer range
    let's say we have floating point range (-3.5, 10.2) and integer range (-8, 7) (int4)
    we'll use (-10.2, 10.2) as the range for floating point and map that to (-8, 7)
    e.g. scale = (10.2 - (-10.2)) / (7 - (-8))

    SYMMETRIC_NO_CLIPPING_ERR is a variant of symmetric mapping, where the scale is the max of smin
    and smax, where smin = min_val_neg / quant_min, and smax = max_val_pos / quant_max. By calculating
    smin and smax individually, there can be less round error on negative values, and no out-of-range
    of all floating point values.

    asymmetric mapping means we just directly map the floating point range to integer range,
    for the above example, we will map (-3.5, 10.2) to (-8, 7) and calculate quantization parameter
    based on this mapping
    e.g. scale = (10.2 - (-3.5)) / (7 - (-8))
    """

    SYMMETRIC = auto()
    SYMMETRIC_NO_CLIPPING_ERR = auto()
    ASYMMETRIC = auto()


class ZeroPointDomain(Enum):
    """Enum that indicate whether zero_point is in integer domain or floating point domain

    integer domain: quantized_val = (float_val / scale) (integer) + zero_point (integer)
    float domain: quantized_val = (float_val - (zero_point (float) - scale * mid_point)) / scale
    none domain: quantized_val = (float_val / scale)
    """

    INT = auto()
    FLOAT = auto()
    NONE = auto()


class TorchAODType(Enum):
    """
    Placeholder for dtypes that do not exist in PyTorch core yet.
    """

    # torch.int1 to torch.int7 will be added to PyTorch 2.6
    # These will remain here for BC with older PyTorch versions
    INT1 = auto()
    INT2 = auto()
    INT3 = auto()
    INT4 = auto()
    INT5 = auto()
    INT6 = auto()
    INT7 = auto()


@dataclass(frozen=True)
class Granularity:
    """
    Base class for representing the granularity of quantization.

    This class serves as a parent for specific granularity types used in
    quantization operations, such as per-tensor or per-axis quantization.
    """


@dataclass(frozen=True)
class PerBlock(Granularity):
    """
    Represents per-block granularity in quantization. See
    :func:`~torchao.quantization.quant_primitives.quantize_affine` for docs for
    `block_size`

    Attributes:
        block_size (Tuple[int, ...]): The size of each quantization group
    """

    block_size: tuple[int, ...]


@dataclass(frozen=True)
class PerTensor(Granularity):
    """
    Represents per-tensor granularity in quantization.

    This granularity type calculates the quantization parameters
    based off the entire tensor.

    """


@dataclass(frozen=True)
class PerAxis(Granularity):
    """
    Represents per-axis granularity in quantization.

    This granularity type calculates different quantization parameters
    along a specified axis of the tensor.

    For example if the input tensor is shape [8, 16] and axis=0, then
    the quantization parameters are calculated for each row of the tensor.
    Giving a total of 8 quantization parameters.

    Attributes:
        axis (int): The axis along which reduction is performed.
    """

    axis: int


@dataclass(frozen=True)
class PerGroup(Granularity):
    """
    Represents per-channel group granularity in quantization.

    This granularity type calculates different quantization parameters
    for each group of <group_size> elements.

    For example if the input tensor is shape [8, 16], and the group size is 4, then
    the input tensor is reshaped to [64, 4]
    quantization parameters are calculated for each group of 4 elements,
    giving a total of 64 quantization parameters.

    Attributes:
        group_size (int): The size of each quantization group

    """

    group_size: int


class PerRow(Granularity):
    """
    Represents row-wise granularity in quantization.

    This is a special case of per-axis quantization and is unique to Float8 matmuls
    where the input is quantized with a block_size of (1, ..., input.shape[-1]). And the weight
    is quantized with a block_size of (1, weight.shape[1]).
    """


class PerToken(Granularity):
    """
    Represents per-token granularity in quantization.

    This granularity type calculates a different set of quantization parameters
    for each token, which is represented as the last dimension of the tensor.

    For example, if the input tensor has shape [2, 3, 4], then there are 6 tokens
    with 4 elements each, and we will calculate 6 sets of quantization parameters,
    one for each token.

    If the input tensor has only two dimensions, e.g. [8, 16], then this is
    equivalent to `PerAxis(axis=0)`, which yields 8 sets of quantization parameters.
    """


def get_block_size(
    input_shape: tuple[int, ...], granularity: Granularity
) -> tuple[int, ...]:
    """Get the block size based on the input shape and granularity type.

    Args:
        input_shape: The input tensor shape possibly more than 2 dimensions
        granularity: The granularity type of the quantization
    """
    if not isinstance(granularity, Granularity):
        raise AssertionError(
            "Please provide an instance of Granularity, not subclass of it"
        )
    if isinstance(granularity, PerTensor):
        return input_shape
    elif isinstance(granularity, PerAxis):
        block_size = list(input_shape)
        block_size[granularity.axis] = 1
        return tuple(block_size)
    elif isinstance(granularity, PerRow):
        return (1,) * (len(input_shape) - 1) + (input_shape[-1],)
    elif isinstance(granularity, PerGroup):
        if len(input_shape) != 2:
            raise AssertionError(
                f"Expecting input shape dim to be 2 for per group quantization, gotinput shape: {input_shape}"
            )
        return (1, granularity.group_size)
    elif isinstance(granularity, PerToken):
        block_size = [1] * len(input_shape)
        block_size[-1] = input_shape[-1]
        return tuple(block_size)
    raise ValueError(f"Unsupported Granularity: {granularity}")


class AffineQuantizedObserverBase(ABC, torch.nn.Module):
    """Observer module for affine quantization (https://github.com/pytorch/ao/tree/main/torchao/quantization#affine-quantization)

    Args:
      `granularity` and `block_size`: The granularity of the quantization,
        must specify at least one, if both are specified `block_size` takes precedence
        Current supported granularity type are `PerTensor` and `PerAxis`
      other args: please see `:class:torchao.dtypes.AffineQuantizedTensor`
    """

    with_args = classmethod(_with_args)

    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: int | None = None,
        quant_max: int | None = None,
        eps: float | None = None,
        scale_dtype: torch.dtype | None = None,
        zero_point_dtype: torch.dtype | None = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain | None = ZeroPointDomain.INT,
        # there could be some extra args that's ignored
        **kwargs,
    ):
        super().__init__()
        if granularity is None:
            raise AssertionError("granularity is None")
        self.mapping_type = mapping_type
        self.target_dtype = target_dtype
        self.granularity = granularity
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.scale_dtype = scale_dtype
        self.zero_point_dtype = zero_point_dtype
        self.preserve_zero = preserve_zero
        self.zero_point_domain = zero_point_domain
        # populated during forward
        self.block_size = None
        self.original_dtype = None

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward function should take the input tensor
        and updates internal stats and return the original input Tensor
        """

    @abstractmethod
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantization parameter based on the stats attached to the observer module
        and returns a tuple of scale and zero_point Tensor
        """

    def convert(self, model: torch.fx.GraphModule, observer_node: Node):
        """
        Converts the observer node in the graph into its quantized representation

        Args:
            model: graph module to convert the observer node in
            observer_node: the observer node to convert
        """
        from torch.ao.quantization.fx.utils import create_getattr_from_value

        with model.graph.inserting_before(observer_node):
            if self.block_size is None:
                raise AssertionError("Expecting block_size to be populated")
            if self.original_dtype is None:
                raise AssertionError("Expecting original_dtype to be populated")
            if hasattr(self, "is_dynamic") and self.is_dynamic:
                choose_qparams_affine = model.graph.call_function(
                    torch.ops.pt2e_quant.choose_qparams_affine,
                    (
                        observer_node.args[0],
                        self.mapping_type.name,
                        self.block_size,
                        self.target_dtype,
                        self.quant_min,
                        self.quant_max,
                        self.eps,
                        self.scale_dtype,
                        self.zero_point_dtype,
                        self.preserve_zero,
                        # pyrefly: ignore [missing-attribute]
                        self.zero_point_domain.name,
                    ),
                )
                scale_node = model.graph.call_function(
                    operator.getitem, (choose_qparams_affine, 0)
                )
                zero_point_node = model.graph.call_function(
                    operator.getitem, (choose_qparams_affine, 1)
                )
            else:
                scale, zero_point = self.calculate_qparams()
                scale_node = create_getattr_from_value(
                    model,
                    model.graph,
                    "_scale",
                    scale,
                    scale.device if isinstance(scale, torch.Tensor) else None,
                )
                zero_point_node = create_getattr_from_value(
                    model,
                    model.graph,
                    "_zero_point",
                    zero_point,
                    zero_point.device if isinstance(zero_point, torch.Tensor) else None,
                )

            q_node = model.graph.call_function(
                torch.ops.pt2e_quant.quantize_affine,
                (
                    observer_node.args[0],
                    self.block_size,
                    scale_node,
                    zero_point_node,
                    self.target_dtype,
                    self.quant_min,
                    self.quant_max,
                    # pyrefly: ignore [missing-attribute]
                    self.zero_point_domain.name,
                ),
                {},
            )
            dq_node = model.graph.call_function(
                torch.ops.pt2e_quant.dequantize_affine,
                (
                    q_node,
                    self.block_size,
                    scale_node,
                    zero_point_node,
                    self.target_dtype,
                    self.quant_min,
                    self.quant_max,
                    # pyrefly: ignore [missing-attribute]
                    self.zero_point_domain.name,
                ),
                {"output_dtype": self.original_dtype},
            )
            observer_node.replace_all_uses_with(dq_node)
            model.graph.erase_node(observer_node)


def _is_observer_script_module(mod, obs_type_name):
    """Returns true if given mod is an instance of Observer script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.observer.___torch_mangle_2.MinMaxObserver'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return obs_type_name in name
    return False


# Experimental Affine Quantization Feature END


def _is_activation_post_process(module):
    return isinstance(
        module,
        (
            torch.ao.quantization.ObserverBase,
            torch.ao.quantization.FakeQuantizeBase,
            AffineQuantizedObserverBase,
        ),
    ) or _is_observer_script_module(module, "quantization.observer")


def _is_per_channel_script_obs_instance(module):
    if isinstance(module, torch.jit.RecursiveScriptModule):
        return _is_observer_script_module(
            module, "quantization.observer.PerChannelMinMaxObserver"
        ) or _is_observer_script_module(
            module, "quantization.observer.MovingAveragePerChannelMinMaxObserver"
        )
    return False


def get_observer_state_dict(mod):
    r"""
    Returns the state dict corresponding to the observer stats.
    Traverse the model state_dict and extract out the stats.
    """
    od = OrderedDict()
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        for k, v in mod.state_dict().items():
            if "observer" in k:
                od[k] = v
    else:
        # path for GraphModule and nn.Module (eager mode)
        for k, v in mod.state_dict().items():
            if "activation_post_process" in k:
                od[k] = v
    od._metadata = mod.state_dict()._metadata  # type: ignore[attr-defined]
    return od


def load_observer_state_dict(mod, obs_dict):
    r"""
    Given input model and a state_dict containing model observer stats,
    load the stats back into the model. The observer state_dict can be saved
    using torch.ao.quantization.get_observer_state_dict
    """
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    for name, module in mod.named_modules():
        prefix = name + "."
        if _is_activation_post_process(module):
            if _is_per_channel_script_obs_instance(module):
                # For per-channel observers we need to call a custom load_from_state_dict to resize the tensor.
                # However this is not called when the module is scripted and we end up calling the default one in module.py
                module._load_from_state_dict_script(
                    obs_dict, prefix, {}, True, missing_keys, unexpected_keys, []
                )
            else:
                module._load_from_state_dict(
                    obs_dict, prefix, {}, False, missing_keys, unexpected_keys, []
                )
    for k in missing_keys:
        if "observer" in k or "activation_post_process" in k:
            raise Exception(  # noqa: TRY002
                f"Missing keys for observer {k} in state_dict"
            )
    for k in unexpected_keys:
        if "observer" in k or "activation_post_process" in k:
            raise Exception(  # noqa: TRY002
                f"Unexpected keys for observer {k} in state_dict"
            )


# Restrict activations to be in the range (0,127)
default_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127)
"""
Default observer for static quantization, usually used for debugging.
"""

default_placeholder_observer = PlaceholderObserver
"""
Default placeholder observer, usually used for quantization to torch.float16.
"""

default_debug_observer = RecordingObserver
"""
Default debug-only observer.
"""

default_weight_observer = MinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
)
"""
Default weight observer.
"""

weight_observer_range_neg_127_to_127 = MinMaxObserver.with_args(
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    quant_min=-127,
    quant_max=127,
    eps=2**-12,
)
"""
Symmetric weight observer with the 8-bit values restricted to [-127, +127], excluding -128.
"""

default_histogram_observer = HistogramObserver.with_args(quant_min=0, quant_max=127)
"""
Default histogram observer, usually used for PTQ.
"""

default_per_channel_weight_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)
"""
Default per-channel weight observer, usually used on backends where per-channel
weight quantization is supported, such as `fbgemm`.
"""

per_channel_weight_observer_range_neg_127_to_127 = PerChannelMinMaxObserver.with_args(
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    quant_min=-127,
    quant_max=127,
    eps=2**-12,
)
"""
Per-channel, symmetric weight observer with the 8-bit values restricted to [-127, +127], excluding -128.
"""

default_dynamic_quant_observer = PlaceholderObserver.with_args(
    dtype=torch.quint8,
    quant_min=0,
    quant_max=255,
    is_dynamic=True,
)
"""
Default observer for dynamic quantization.
"""

default_float_qparams_observer = PerChannelMinMaxObserver.with_args(
    dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
)
"""
Default observer for a floating point zero-point.
"""

default_float_qparams_observer_4bit = PerChannelMinMaxObserver.with_args(
    dtype=torch.quint4x2, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
)
"""
Default observer for a floating point zero-point and 4 bit activations.
"""

# TODO(future PR): remove these defaults and enforce activation functions
# to explicitly specify their output range
default_fixed_qparams_range_neg1to1_observer = FixedQParamsObserver.with_args(
    scale=2.0 / 256.0, zero_point=128, dtype=torch.quint8, quant_min=0, quant_max=255
)
default_fixed_qparams_range_0to1_observer = FixedQParamsObserver.with_args(
    scale=1.0 / 256.0, zero_point=0, dtype=torch.quint8, quant_min=0, quant_max=255
)
# TODO: the following 2 variables are kept for backwards compatibility; remove after a few releases
default_symmetric_fixed_qparams_observer = default_fixed_qparams_range_neg1to1_observer
default_affine_fixed_qparams_observer = default_fixed_qparams_range_0to1_observer

"""
Default observers for fixed qparams operations.
"""

default_reuse_input_observer = ReuseInputObserver
"""
Default observer for operators like reshape that reuses the observer of input to
the operator
"""
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.ao.quantization.utils import Pattern


__all__ = [
    "BackendConfig",
    "BackendPatternConfig",
    "DTypeConfig",
    "DTypeWithConstraints",
    "ObservationType",
]


# DTypeConfig dict keys
INPUT_DTYPE_DICT_KEY = "input_dtype"
OUTPUT_DTYPE_DICT_KEY = "output_dtype"
WEIGHT_DTYPE_DICT_KEY = "weight_dtype"
BIAS_DTYPE_DICT_KEY = "bias_dtype"
IS_DYNAMIC_DICT_KEY = "is_dynamic"

# BackendConfig dict keys
NAME_DICT_KEY = "name"
CONFIGS_DICT_KEY = "configs"

# BackendPatternConfig dict keys
PATTERN_DICT_KEY = "pattern"
PATTERN_COMPLEX_FORMAT_DICT_KEY = "pattern_complex_format"
OBSERVATION_TYPE_DICT_KEY = "observation_type"
DTYPE_CONFIGS_DICT_KEY = "dtype_configs"
ROOT_MODULE_DICT_KEY = "root_module"
QAT_MODULE_DICT_KEY = "qat_module"
REFERENCE_QUANTIZED_MODULE_DICT_KEY = "reference_quantized_module_for_root"
FUSED_MODULE_DICT_KEY = "fused_module"
FUSER_METHOD_DICT_KEY = "fuser_method"
ROOT_NODE_GETTER_DICT_KEY = "root_node_getter"
EXTRA_INPUTS_GETTER_DICT_KEY = "extra_inputs_getter"
NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY = "num_tensor_args_to_observation_type"
INPUT_TYPE_TO_INDEX_DICT_KEY = "input_type_to_index"


# TODO: maybe rename this to something that's not related to observer
# e.g. QParamsType
class ObservationType(Enum):
    """An enum that represents different ways of how an operator/operator pattern
    should be observed
    """

    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    """this means input and output are observed with different observers, based
    on qconfig.activation
    example: conv, linear, softmax
    """

    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
    """this means the output will use the same observer instance as input, based
    on qconfig.activation
    example: torch.cat, maxpool
    """

    INPUT_OUTPUT_NOT_OBSERVED = 2
    """this means the input and output are never observed
    example: x.shape, x.size
    """


@dataclass
class DTypeWithConstraints:
    """
    Config for specifying additional constraints for a given dtype, such as quantization
    value ranges, scale value ranges, and fixed quantization params, to be used in
    :class:`~torch.ao.quantization.backend_config.DTypeConfig`.

    The constraints currently supported are:

    * `quant_min_lower_bound` and `quant_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum quantized values respectively. If
      the QConfig's `quant_min` and `quant_max` fall outside this range,
      then the QConfig will be ignored.

    * `scale_min_lower_bound` and `scale_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum scale values respectively. If the
      QConfig's minimum scale value (currently exposed as `eps`) falls below
      the lower bound, then the QConfig will be ignored. Note that the upper
      bound is currently not enforced.

    * `scale_exact_match` and `zero_point_exact_match`: Exact match requirements
      for scale and zero point, to be used for operators with fixed quantization
      parameters such as sigmoid and tanh. If the observer specified in the QConfig
      is neither `FixedQParamsObserver` nor `FixedQParamsFakeQuantize`, or if
      the quantization parameters don't match, then the QConfig will be ignored.
    """

    dtype: torch.dtype | None = None
    quant_min_lower_bound: int | float | None = None
    quant_max_upper_bound: int | float | None = None
    scale_min_lower_bound: int | float | None = None
    scale_max_upper_bound: int | float | None = None
    scale_exact_match: float | None = None
    zero_point_exact_match: int | None = None


@dataclass
class DTypeConfig:
    """
    Config object that specifies the supported data types passed as arguments to
    quantize ops in the reference model spec, for input and output activations,
    weights, and biases.

    For example, consider the following reference model:

      quant1 - [dequant1 - fp32_linear - quant2] - dequant2

    The pattern in the square brackets refers to the reference pattern of
    statically quantized linear. Setting the input dtype as `torch.quint8`
    in the DTypeConfig means we pass in `torch.quint8` as the dtype argument
    to the first quantize op (quant1). Similarly, setting the output dtype as
    `torch.quint8` means we pass in `torch.quint8` as the dtype argument to
    the second quantize op (quant2).

    Note that the dtype here does not refer to the interface dtypes of the
    op. For example, the "input dtype" here is not the dtype of the input
    tensor passed to the quantized linear op. Though it can still be the
    same as the interface dtype, this is not always the case, e.g. the
    interface dtype is fp32 in dynamic quantization but the "input dtype"
    specified in the DTypeConfig would still be quint8. The semantics of
    dtypes here are the same as the semantics of the dtypes specified in
    the observers.

    These dtypes are matched against the ones specified in the user's
    QConfig. If there is a match, and the QConfig satisfies the constraints
    specified in the DTypeConfig (if any), then we will quantize the given
    pattern using this DTypeConfig. Otherwise, the QConfig is ignored and
    the pattern will not be quantized.

    Example usage::

        >>> # xdoctest: +SKIP(failing)
        >>> dtype_config1 = DTypeConfig(
        ...     input_dtype=torch.quint8,
        ...     output_dtype=torch.quint8,
        ...     weight_dtype=torch.qint8,
        ...     bias_dtype=torch.float)

        >>> dtype_config2 = DTypeConfig(
        ...     input_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     output_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     weight_dtype=DTypeWithConstraints(
        ...         dtype=torch.qint8,
        ...         quant_min_lower_bound=-128,
        ...         quant_max_upper_bound=127,
        ...     ),
        ...     bias_dtype=torch.float)

        >>> dtype_config1.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype_with_constraints
        DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, \
scale_min_lower_bound=None, scale_max_upper_bound=None)
    """

    input_dtype_with_constraints: DTypeWithConstraints
    output_dtype_with_constraints: DTypeWithConstraints
    weight_dtype_with_constraints: DTypeWithConstraints
    bias_dtype: torch.dtype | None
    is_dynamic: bool | None

    def __init__(
        self,
        input_dtype: torch.dtype | DTypeWithConstraints | None = None,
        output_dtype: torch.dtype | DTypeWithConstraints | None = None,
        weight_dtype: torch.dtype | DTypeWithConstraints | None = None,
        bias_dtype: torch.dtype | None = None,
        is_dynamic: bool | None = None,
    ):
        if isinstance(input_dtype, DTypeWithConstraints):
            self.input_dtype_with_constraints = input_dtype
        else:
            self.input_dtype_with_constraints = DTypeWithConstraints(dtype=input_dtype)

        if isinstance(output_dtype, DTypeWithConstraints):
            self.output_dtype_with_constraints = output_dtype
        else:
            self.output_dtype_with_constraints = DTypeWithConstraints(
                dtype=output_dtype
            )

        if isinstance(weight_dtype, DTypeWithConstraints):
            self.weight_dtype_with_constraints = weight_dtype
        else:
            self.weight_dtype_with_constraints = DTypeWithConstraints(
                dtype=weight_dtype
            )

        self.bias_dtype = bias_dtype
        self.is_dynamic = is_dynamic

    @property
    def input_dtype(self) -> torch.dtype | None:
        return self.input_dtype_with_constraints.dtype

    @property
    def output_dtype(self) -> torch.dtype | None:
        return self.output_dtype_with_constraints.dtype

    @property
    def weight_dtype(self) -> torch.dtype | None:
        return self.weight_dtype_with_constraints.dtype

    @classmethod
    def from_dict(cls, dtype_config_dict: dict[str, Any]) -> DTypeConfig:
        """
        Create a ``DTypeConfig`` from a dictionary with the following items (all optional):
            "input_dtype": torch.dtype or ``DTypeWithConstraints``
            "output_dtype": torch.dtype or ``DTypeWithConstraints``
            "weight_dtype": torch.dtype or ``DTypeWithConstraints``
            "bias_type": torch.dtype
            "is_dynamic": bool
        """
        input_dtype = dtype_config_dict.get(INPUT_DTYPE_DICT_KEY)
        if input_dtype is not None and not isinstance(
            input_dtype, (torch.dtype, DTypeWithConstraints)
        ):
            raise ValueError(
                "Expected input_dtype to be a torch.dtype or DTypeWithConstraints"
            )
        output_dtype = dtype_config_dict.get(OUTPUT_DTYPE_DICT_KEY)
        if output_dtype is not None and not isinstance(
            output_dtype, (torch.dtype, DTypeWithConstraints)
        ):
            raise ValueError(
                "Expected output_dtype to be a torch.dtype or DTypeWithConstraints"
            )
        weight_dtype = dtype_config_dict.get(WEIGHT_DTYPE_DICT_KEY)
        if weight_dtype is not None and not isinstance(
            weight_dtype, (torch.dtype, DTypeWithConstraints)
        ):
            raise ValueError(
                "Expected weight_dtype to be a torch.dtype or DTypeWithConstraints"
            )
        bias_dtype = dtype_config_dict.get(BIAS_DTYPE_DICT_KEY)
