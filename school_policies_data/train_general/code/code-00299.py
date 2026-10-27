            Throws exception if attempt to get visualizers without generating report
        """
        # check if user has generated reports at least once
        if len(self._generated_reports) == 0:
            raise Exception(  # noqa: TRY002
                "Unable to generate visualizers without first generating reports"
            )

        # get the ordered dict mapping modules to their full set of collected features / stats
        module_fqns_to_features: OrderedDict = self._reformat_reports_for_visualizer()

        # create and return ModelReportVisualizer instance
        visualizer: ModelReportVisualizer = ModelReportVisualizer(
            module_fqns_to_features
        )

        return visualizer

    def _generate_qconfig_mapping_helper(
        self,
        detector_qconfig_info_combined: dict[str, DetectorQConfigInfo],
        generation_function: Callable,
    ) -> QConfigMapping:
        r"""
        This helper takes in the compiled detector qconfig info that
        has been compiled together and merges it into a QConfigMapping
        """
        # keep track of the qconfigmapping
        qconfig_mapping = QConfigMapping()

        # loop through each module / fqn and attempt to create QConfigMapping
        for fqn, module in self._model.named_modules():
            # if we have a qconfig info for this module
            if fqn in detector_qconfig_info_combined:
                qconfig_info_compiled = detector_qconfig_info_combined[fqn]

                # now generate the qconfig and add it to the mapping
                generated_qconfig = generation_function(qconfig_info_compiled, module)

                # add to our config
                qconfig_mapping.set_module_name(fqn, generated_qconfig)

        # return compiled mapping
        return qconfig_mapping

    def _update_detector_quantizaiton_qconfig_info(
        self, combined_info: DetectorQConfigInfo, new_info: DetectorQConfigInfo
    ):
        r"""
        Takes in the old and new information and updates the combined information.

        Args:
            combined_info (DetectorQConfigInfo): The DetectorQConfigInfo we are compiling all of the information in
            new_info (DetectorQConfigInfo): The DetectorQConfigInfo with the information we are trying to merge the new info
                into it
        """
        combined_info.is_activation_dynamic = (
            combined_info.is_activation_dynamic or new_info.is_activation_dynamic
        )
        combined_info.is_weight_per_channel = (
            combined_info.is_weight_per_channel or new_info.is_weight_per_channel
        )

    def _update_detector_equalization_qconfig_info(
        self, combined_info: DetectorQConfigInfo, new_info: DetectorQConfigInfo
    ):
        r"""
        Takes in the old and new information and updates the combined information.

        Args:
            combined_info (DetectorQConfigInfo): The DetectorQConfigInfo we are compiling all of the information in
            new_info (DetectorQConfigInfo): The DetectorQConfigInfo with the information we are trying to merge the new info
                into it
        """
        is_equalization_recommended = (
            combined_info.is_equalization_recommended
            or new_info.is_equalization_recommended
        )
        combined_info.is_equalization_recommended = is_equalization_recommended

    def _generate_module_fqn_to_detector_info_mapping(
        self, update_qconfig_info_function: Callable
    ) -> dict[str, DetectorQConfigInfo]:
        r"""
        Generates a QConfigMapping based on the suggestions of the
        ModelReport API. The generated mapping encompasses all the
        different types of feedback from the different detectors
        all into one place.

        These configs are based on the suggestions provided by the ModelReport API
        and can only be generated once the reports have been generated.

        Args:
            update_qconfig_info_function (Callable) takes in a function that takes in two DetectorQConfigInfo
            and updates the one that is being compiled

        Returns a Dict mapping module_fqns to DetectorQConfigInfo objects

        Note:
            Throws exception if we try to generate mapping on model we already removed observers from
            Throws exception if we try to generate mapping without preparing for calibration
        """
        # if we haven't prepped model for calibration, then we shouldn't generate mapping yet
        if not self._prepared_flag:
            raise Exception(  # noqa: TRY002
                "Cannot generate report without preparing model for calibration"
            )

        # if we already removed the observers, we cannot mapping
        if self._removed_observers:
            raise Exception(  # noqa: TRY002
                "Cannot generate report on model you already removed observers from"
            )

        # keep track of qconfig info for each module across detectors
        detector_qconfig_info_combined: dict[str, DetectorQConfigInfo] = {}

        for detector in self._desired_report_detectors:
            # get the info from the detector
            detector_info: dict[str, DetectorQConfigInfo] = detector.get_qconfig_info(
                self._model
            )

            # we go through the modules
            for module_fqn in detector_info:
                # see if we already have info on it
                if module_fqn in detector_qconfig_info_combined:
                    # we combine the current options with what is there
                    current_options = detector_qconfig_info_combined[module_fqn]
                    detector_options = detector_info[module_fqn]

                    update_qconfig_info_function(current_options, detector_options)
                else:
                    # we just use this for now
                    detector_qconfig_info_combined[module_fqn] = detector_info[
                        module_fqn
                    ]

        return detector_qconfig_info_combined

    def generate_qconfig_mapping(self) -> QConfigMapping:
        r"""
        Generates a QConfigMapping based on the suggestions of the
        ModelReport API. The generated mapping encompasses all the
        different types of feedback from the different detectors
        all into one place.

        These configs are based on the suggestions provided by the ModelReport API
        and can only be generated once the reports have been generated.

        Returns a QConfigMapping for the quantization configuration

        Note:
            Throws exception if we try to generate mapping on model we already removed observers from
            Throws exception if we try to generate mapping without preparing for calibration
        """
        # get the mapping info
        detector_qconfig_info_combined = (
            self._generate_module_fqn_to_detector_info_mapping(
                self._update_detector_quantizaiton_qconfig_info
            )
        )

        # we will do a bit of processing and remove fqns that don't have input weight recommended

        # now we generate the QConfig for each of the options
        mapping: QConfigMapping = self._generate_qconfig_mapping_helper(
            detector_qconfig_info_combined, self._quantization_config_generator
        )

        # return the generated mapping
        return mapping

    def _quantization_config_generator(
        self, detector_qconfig_info: DetectorQConfigInfo, module: torch.nn.Module
    ) -> QConfig:
        r"""
        Returns the quantization configuration generated by the DetectorQConfigInfo object
        """
        return detector_qconfig_info.generate_quantization_qconfig(module)

    def _equalization_config_generator(
        self, detector_qconfig_info: DetectorQConfigInfo, module: torch.nn.Module
    ) -> EqualizationQConfig:
        r"""
        We ignore the module argument here, and only focus on thedetector_qconfig_info

        Returns the equalization configuration generated by the DetectorQConfigInfo object
        """
        return detector_qconfig_info.generate_equalization_qconfig()

    def generate_equalization_mapping(self) -> QConfigMapping:
        r"""
        Generates a QConfigMapping based on the suggestions of the
        ModelReport API for equalization. The generated mapping encompasses all the
        different types of feedback from the input-weight equalization detector.

        These configs are based on the suggestions provided by the ModelReport API
        and can only be generated once the reports have been generated.

        Returns a QConfigMapping for the equalization configuration
        """
        # get the mapping info
        detector_qconfig_info_combined = (
            self._generate_module_fqn_to_detector_info_mapping(
                self._update_detector_equalization_qconfig_info
            )
        )

        # now we generate the QConfig for each of the options
        mapping: QConfigMapping = self._generate_qconfig_mapping_helper(
            detector_qconfig_info_combined, self._equalization_config_generator
        )

        # return the generated mapping
        return mapping
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch.ao.quantization.observer import ObserverBase


class ModelReportObserver(ObserverBase):
    r"""This observer is used to record additional information regarding keeping track
    of S = average_batch_activation_range/epoch_activation_range.

    The purpose of this information is to prepare a report to present to users on whether
    Dynamic or Static Quantization is more appropriate for their model given the general
    distributions of their data.

    Args:
        ch_axis (int, optional): The channel axis for which the range and outlier stats are computed
            Default: 1
        comp_percentile (float, optional): The percentile to compare against 100 percentile to find outliers
            Should be between 0 and 1 exclusive
            Default: 0.9

    * :attr:`num_batches_tracked` specifies number of batches passed through the observer

    * :attr:`average_batch_activation_range` defines average across the ranges of each batch passed through

    * :attr:`epoch_activation_min` defines the minimum value passed through the observer

    * :attr:`epoch_activation_max` defines the maximum value passed through the observer

    * :attr:`ch_axis` defines the channel being used to compute per channel min max stats

    * :attr:`min_val` defines the per channel minimum values passed through

    * :attr:`max_val` defines the per channel maximum values passed through

    * :attr:`comp_percentile` defines comparison percentile to find outliers

    * :attr:`average_percentile_ratio` defines the per channel average percentile ratios

    * :attr:`percentile_batches_tracked` defines the number of percentile batches tracked for each channel

    * :attr:`constant_channels` defines the number of batches that aren't constant channels per channel

    Note: this tool is meant for FX Graph Mode Quantization
    """

    epoch_activation_min: torch.Tensor
    epoch_activation_max: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor
    comp_percentile: torch.Tensor
    average_percentile_ratio: torch.Tensor
    percentile_batches_tracked: torch.Tensor
    constant_channels: torch.Tensor

    def __init__(self, ch_axis: int = 1, comp_percentile: float = 0.9):
        super().__init__(torch.qint8)
        self.num_batches_tracked = 0

        # keep track of the min and mix of the range for average batch and epoch as a whole
        self.average_batch_activation_range: torch.Tensor = torch.tensor(float(0))
        self.register_buffer("epoch_activation_min", torch.tensor(float("inf")))
        self.register_buffer("epoch_activation_max", torch.tensor(float("-inf")))

        # keep track of per channel min max information using the given channel
        self.ch_axis: int = ch_axis
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))

        # keep track of percentile ratio information per channel
        self.register_buffer("comp_percentile", torch.tensor([comp_percentile]))
        self.register_buffer("average_percentile_ratio", torch.tensor([]))
        self.register_buffer("percentile_batches_tracked", torch.tensor([]))
        self.register_buffer("constant_channels", torch.tensor([]))

    def forward(self, x):
        x_copy = x.detach()  # avoid keeping autograd tape
        x_copy = x_copy.to(self.epoch_activation_min.dtype)

        x_copy = self._calculate_range_stats(x_copy)
        x_copy = self._calculate_min_max_stats(x_copy)
        x_copy = self._calculate_percentile_stats(x_copy)

        # return the passed in the value
        return x

    def _calculate_range_stats(self, x_copy):
        r"""Calculates and stores range stats with forward values.

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
        # get the min, max values of the data
        min_val_cur, max_val_cur = torch.aminmax(x_copy)

        # calculate new epoch range values
        epoch_min_val = torch.min(self.epoch_activation_min, min_val_cur)
        epoch_max_val = torch.max(self.epoch_activation_max, max_val_cur)

        self.epoch_activation_min.copy_(epoch_min_val)
        self.epoch_activation_max.copy_(epoch_max_val)

        # calculate the average batch activation range
        current_batch_range = max_val_cur - min_val_cur
        new_range = (
            self.average_batch_activation_range * self.num_batches_tracked
            + current_batch_range
        ) / (self.num_batches_tracked + 1)

        self.average_batch_activation_range = new_range
        self.num_batches_tracked += 1  # new batch was processed

        return x_copy

    def _calculate_min_max_stats(self, x_copy):
        r"""Calculates and stores the per_channel min, max stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
        # get the current min and max vals
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x_copy.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x_copy.permute(new_axis_list)
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

        return x_copy

    def _calculate_percentile_stats(self, x_copy):
        r"""Calculates and stores the per_channel percentile stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
        # get the dimension of the copy
        x_dim = x_copy.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x_copy.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        y = y.to(dtype=self.min_val.dtype, device="cpu")

        # find the percentile values along the axis
        # we want both 100th percentile and comp_percentile
        # we also want to find 0th quartile to see if we have constant channel
        quantiles_list = [0, self.comp_percentile, 1.00]
        quantiles_to_find = torch.tensor(quantiles_list, dtype=self.min_val.dtype)

        # find the quantiles
        desired_quantiles = torch.quantile(
            y, quantiles_to_find, dim=self.ch_axis, interpolation="lower"
        )
        zero_quantile = desired_quantiles[0]
        comp_quantile = desired_quantiles[1]
        hundreth_quartile = desired_quantiles[2]

        # if any of the channels have 0s, we ignore that channel for this calculation
        any_non_zero_quantile_value: torch.Tensor = (
            comp_quantile != torch.tensor([0])
        ) | (hundreth_quartile != torch.tensor([0]))
        any_non_zero_quantile_value = (
            any_non_zero_quantile_value.int()
        )  # transform boolean values to int values

        # we also check if we have a constant channel
        any_constant_channels: torch.Tensor = (
            hundreth_quartile - zero_quantile
        ) == torch.tensor([0])
        any_constant_channels = (
            any_constant_channels.int()
        )  # transform boolean values to int values

        # possibilities to get nan as an answer
        #   will ignore any of these three cases with 0s and just not deal with them for now
        # case (1) 0 in numerator: issue if 0 is largest, all negative, and rest are really negative
        # case (2) 0 in denominator: is possible unless case 3, we just ignore
        # case (3) 0 in both: not outlier, channel just kinda useless, ignore

        # get the ratio and get rid of nan values
        quantile_ratios = hundreth_quartile / comp_quantile
        quantile_ratios = torch.nan_to_num(quantile_ratios)
        # update averages, remembering to only update if didn't have zeros
        ratio_if_not_zero = any_non_zero_quantile_value * quantile_ratios

        # if num_batches and average_ratio are not initialized, we want to initialize them
        if (
            self.percentile_batches_tracked.shape[0] == 0
            or self.average_percentile_ratio.shape[0] == 0
        ):
            self.percentile_batches_tracked = torch.zeros_like(
                any_non_zero_quantile_value
            )
            self.average_percentile_ratio = torch.zeros_like(ratio_if_not_zero)

        # also initialize the constant channel var if that is not initialized separately
        if self.constant_channels.shape[0] == 0:
            self.constant_channels = torch.zeros_like(any_constant_channels)

        # get current num batches and average ratio
        num_batches = self.percentile_batches_tracked
        average_ratio = self.average_percentile_ratio

        # calculate new_number of batches, new_ratios, and get rid of nans because of 0 size batches
        new_number_of_batches: torch.Tensor = num_batches + any_non_zero_quantile_value
        new_ratios: torch.Tensor = (
            (average_ratio * num_batches) + ratio_if_not_zero
        ) / new_number_of_batches
        new_ratios = torch.nan_to_num(new_ratios)

        # update the number of non-constant channels
        new_constant_count: torch.Tensor = (
            self.constant_channels + any_constant_channels
        )

        # update the values locally
        self.percentile_batches_tracked.copy_(new_number_of_batches)
        self.average_percentile_ratio.copy_(new_ratios)
        self.constant_channels.copy_(new_constant_count)

        return x_copy

    @torch.jit.export
    def get_batch_to_epoch_ratio(self):
        epoch_activation_range = self.epoch_activation_max - self.epoch_activation_min

        if epoch_activation_range == torch.tensor(float(0)):
            raise ValueError("Range for Epoch is 0")
        elif epoch_activation_range == torch.tensor(float("inf")):
            raise ValueError(
                "No data has been run through observer or infinity value present"
            )
        else:
            return self.average_batch_activation_range / epoch_activation_range

    @torch.jit.export
    def reset_batch_and_epoch_values(self):
        # set all the values back to their original defaults for a new epoch
        # keep device
        device = self.max_val.device
        self.num_batches_tracked = 0
        self.average_batch_activation_range = torch.tensor(float(0), device=device)
        self.epoch_activation_min = torch.tensor(float("inf"), device=device)
        self.epoch_activation_max = torch.tensor(float("-inf"), device=device)
        self.min_val = torch.tensor([], device=device)
        self.max_val = torch.tensor([], device=device)
        self.average_percentile_ratio = torch.tensor([], device=device)
        self.percentile_batches_tracked = torch.tensor([], device=device)
        self.constant_channels = torch.tensor([], device=device)

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for ModelReportObserver"
        )
<|endoftext|># mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.ao.nn.qat as nnqat
import torch.nn as nn
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx._equalize import (
    default_equalization_qconfig,
    EqualizationQConfig,
)
from torch.ao.quantization.fx._model_report.model_report_observer import (
    ModelReportObserver,
)
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import (
    _is_activation_post_process,
    default_dynamic_quant_observer,
    default_observer,
    default_per_channel_weight_observer,
    default_weight_observer,
    ObserverBase,
)
from torch.ao.quantization.qconfig import (
    _assert_valid_qconfig,
    default_qconfig,
    QConfig,
)


# Names for observer insert keys
DETECTOR_TARGET_NODE_KEY = "target_node"
DETECTOR_OBS_TO_INSERT_KEY = "observer_to_insert"
DETECTOR_IS_POST_OBS_KEY = "is_post_observer"
DETECTOR_OBS_ARGS_KEY = "observer_args"


# Mapping related code
class DetectorQConfigInfo:
    r"""
    This class contains the QConfig information for a single module.
    The list of variables / values this contains can grow depending on the
    extensibility of the qconfig mapping feature set but this currently includes:
    - if activation observer is dynamic
    - if weight observer is per channel


    Args:
        module_fqn (str): The fully qualified name (fqn) of the module that this
            information contains info relevant to qconfig for
    """

    def __init__(self, module_fqn: str):
        super().__init__()
        self.module_fqn = module_fqn

        # populate this section with all the variables we might find important
        # change from none if your detector is actually using this
        self.is_activation_dynamic = False
        self.is_weight_per_channel = False

        # equalization related options
        self.is_equalization_recommended = False

    def generate_quantization_qconfig(self, module: torch.nn.Module) -> QConfig:
        r"""
        Args:
            module (torch.nn.Module) The module we are generating
            the qconfig for

        Returns the generated quantization QConfig according to what a valid configuration is
        """
        # Apply suggestions to new qconfig
        module_qconfig = default_qconfig

        # keep track of dynamic and per_channel recommendations
        recommendations_list = []
        # append as if a list of combinations
        recommendations_list.append(
            (self.is_activation_dynamic, self.is_weight_per_channel)
        )
        recommendations_list.append(
            (self.is_activation_dynamic, False)
        )  # only trying dynamic rec
        recommendations_list.append(
            (False, self.is_weight_per_channel)
        )  # only trying dynamic

        # now we try each of the combinations
        for rec in recommendations_list:
            # rec[0] -> dynamic recommended
            # rec[1] -> per channel recommended
            activation = default_dynamic_quant_observer if rec[0] else default_observer
            weight = (
                default_per_channel_weight_observer
                if rec[1]
                else default_weight_observer
            )
            test_config = QConfig(activation, weight)
            try:
                _assert_valid_qconfig(test_config, module)
                module_qconfig = test_config
                break
            except AssertionError:
                # if not a valid configuration, we move on to the next one in priority
                continue

        # return the QConfig chosen
        return module_qconfig

    def generate_equalization_qconfig(self) -> EqualizationQConfig:
        r"""
        This returns the equalization configuration for a module.

        For now, it just returns the default, but as more equalization options become
        possible, this method can get more fleshed out with more nuanced granularity.


        Returns the generated equalization QConfig according to what a valid configuration is
        """
        # in this case, we just return default equalization config
        # we know this is valid because only valid modules would even
        # have this option
        return default_equalization_qconfig


# Adding base class for detectors
class DetectorBase(ABC):
    r"""Base Detector Module
    Any detector class should derive from this class.

    Concrete detectors should follow the same general API, which includes:
    - A method to calculate and return observer insertion points
        - Should return both the fqns and the Observer class to insert
    - A method to return a report based on the detector
        - Should return a str-based report and dict info in Tuple[str,Dict] format
    """

    def __init__(self) -> None:
        super().__init__()
        self.detector_config_info = None

    @abstractmethod
    def determine_observer_insert_points(self, model) -> dict:
        r"""
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict.
            This dict maps string keys to detector specific information
        """

    @abstractmethod
    def get_detector_name(self) -> str:
        r"""Returns the name of the current detector"""

    @abstractmethod
    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """

    def _get_targeting_node(
        self, prepared_fx_model: GraphModule, target_fqn: str
    ) -> torch.fx.node.Node:
        r"""
        Takes in a GraphModule and the target_fqn and finds the node whose target is this fqn.

        If it's not found, it means it is most likely inside a fused layer
            We just go one layer up in terms of the fqn we are searching for until we find parent node
            If we get to empty string, then we know that it doesn't exist

        The reason for the recursion is that if the model that we are looking for got fused,
        we will have module fqn as e.g. x.linear.0 but the graph will only have a node for the fused module,
        which would have fqn as x.linear so they will not match.
        To handle this, if we don't match, we then take off the last bit of the fqn e.g. x.linear.0 -> x.linear,
        or more generally foo.bar.baz -> foo.bar and search again, this will allow us to locate the correct module
        even in cases with fusion

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            target_fqn (str): The fqn of the layer we are trying to target

        Returns the node object we are trying to add observers around
        """
        for node in prepared_fx_model.graph.nodes:
            # if the node's target is our target, return it
            if node.target == target_fqn:
                return node

        # getting here means node not found
        # if no "." we are already at base and failed
        parent_fqn_sep_index = target_fqn.rfind(".")
        if parent_fqn_sep_index == -1:
            raise ValueError("passed in target_fqn not found in graph's targets.")
        else:
            # recursively call it with parent fqn
            return self._get_targeting_node(
                prepared_fx_model, target_fqn[:parent_fqn_sep_index]
            )

    @abstractmethod
    def generate_detector_report(self, model) -> tuple[str, dict[str, Any]]:
        r"""
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Tuple of two elements:
            Str: string report of the suggested improvements
            Dict: contains useful data collected by the observer pertinent to this report
        """


class PerChannelDetector(DetectorBase):
    r"""This class is used to detect if any Linear or Conv layers in a model utilize per_channel quantization.
    Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

    per_channel quantization can lead to major benefits in the form of accuracy.
    Therefore, if the backend used by the user supports it, it is recommended to use

    Args:
        backend (str, optional): the backend the user wishes to use in production
            Default value is current torch.backends.quantized.engine
    """

    # Keys for return dictionary
    BACKEND_KEY = "backend"
    PER_CHAN_SUPPORTED_KEY = "per_channel_quantization_supported"
    PER_CHAN_USED_KEY = "per_channel_quantization_used"

    # Default map for representing supported per channel quantization modules for different backends
    DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES: dict[str, set[Any]] = {
        "fbgemm": {
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nnqat.Linear,
            nnqat.Conv1d,
            nnqat.Conv2d,
            nnqat.Conv3d,
        },
        "qnnpack": {
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nnqat.Linear,
            nnqat.Conv1d,
            nnqat.Conv2d,
            nnqat.Conv3d,
        },
        "onednn": {
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nnqat.Linear,
            nnqat.Conv1d,
            nnqat.Conv2d,
            nnqat.Conv3d,
        },
        "x86": {
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nnqat.Linear,
            nnqat.Conv1d,
            nnqat.Conv2d,
            nnqat.Conv3d,
        },
    }

    def __init__(self, backend: str = torch.backends.quantized.engine):
        super().__init__()

        # store the backend information
        self.backend_chosen = backend
        self.supported_modules = set()
        if self.backend_chosen in self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES:
            self.supported_modules = self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES[
                self.backend_chosen
            ]
        else:
            raise ValueError(
                f"Not configured to work with {self.backend_chosen}. Try a different default backend"
            )

    def get_detector_name(self) -> str:
        r"""returns the string name of this detector"""
        return "per_channel_detector"

    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # run the helper function to populate the dictionary
        per_channel_info = self._detect_per_channel_helper(model)

        # we actually have a qconfig info object we are populating
        module_fqn_to_detector_qconfig_info = {}

        for module_fqn in per_channel_info:
            # create a detector info instance
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)

            # see if per channel quantization is supported
            per_chan_supported: bool = per_channel_info[module_fqn][
                self.PER_CHAN_SUPPORTED_KEY
            ]
            detector_qconfig_info.is_weight_per_channel = per_chan_supported
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info

        return module_fqn_to_detector_qconfig_info

    def determine_observer_insert_points(self, model: nn.Module) -> dict:
        r"""
        There is no observers inserted for the PerChannelDetector.

        Returns an empty dictionary since no observers are added or needed
        """
        return {}

    def _detect_per_channel_helper(self, model: nn.Module):
        r"""
        determines if per_channel quantization is supported in modules and submodules.

        Returns a dictionary in the higher level _detect_per_channel function.
        Each entry maps the fully-qualified-name to information on whether per_channel quantization.

        Args:
            model: The current module that is being checked to see if it is per_channel quantizable

        Returns dictionary mapping fqns to if per_channel quantization is possible
        """
        # create dict we will return
        per_channel_info: dict = {}

        # get the fully qualified name and check if in list of modules to include and list of modules to ignore
        for fqn, module in model.named_modules():
            is_in_include_list = any(
                isinstance(module, x) for x in self.supported_modules
            )

            # check if the module per_channel is supported
            # based on backend
            per_channel_supported = False

            if is_in_include_list:
                per_channel_supported = True

                # assert statement for MyPy
                q_config_file = module.qconfig
                if not isinstance(q_config_file, QConfig):
                    raise AssertionError("module.qconfig must be a QConfig")

                # this object should either be fake quant or observer
                q_or_s_obj = module.qconfig.weight.p.func()
                if not isinstance(q_or_s_obj, (FakeQuantize, ObserverBase)):
                    raise AssertionError(
                        "module.qconfig.weight must be a FakeQuantize or ObserverBase"
                    )

                per_channel_used = False  # will be true if found in qconfig

                if hasattr(
                    q_or_s_obj, "ch_axis"
                ):  # then we know that per_channel quantization used
                    # all fake quants have channel axis so need to check is_per_channel
                    if isinstance(q_or_s_obj, FakeQuantize):
                        if (
                            hasattr(q_or_s_obj, "is_per_channel")
                            and q_or_s_obj.is_per_channel
                        ):
                            per_channel_used = True
                    elif isinstance(q_or_s_obj, ObserverBase):
                        # should be an observer otherwise
                        per_channel_used = True
                    else:
                        raise ValueError("Should be either observer or fake quant")

                per_channel_info[fqn] = {
                    self.PER_CHAN_SUPPORTED_KEY: per_channel_supported,
                    self.PER_CHAN_USED_KEY: per_channel_used,
                    self.BACKEND_KEY: self.backend_chosen,
                }

        return per_channel_info

    def generate_detector_report(self, model: nn.Module) -> tuple[str, dict[str, Any]]:
        r"""Checks if any Linear or Conv layers in the model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        Looks at q_config format and backend to determine if per_channel can be utilized.
        Uses the DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES structure to determine support

        Args:
            model: The prepared and calibrated model we want to check if using per_channel

        Returns a tuple with two elements:
            String report of potential actions to improve model (if per_channel quantization is available in backend)
            Dictionary mapping per_channel quantizable elements to:
                whether per_channel quantization is supported by the backend
                if it is being utilized in the current model
        """

        # run the helper function to populate the dictionary
        per_channel_info = self._detect_per_channel_helper(model)

        # String to let the user know of further optimizations
        further_optims_str = (
            f"Further Optimizations for backend {self.backend_chosen}: \n"
        )

        optimizations_possible = False
        for fqn in per_channel_info:
            fqn_dict = per_channel_info[fqn]
            if (
                fqn_dict[self.PER_CHAN_SUPPORTED_KEY]
                and not fqn_dict[self.PER_CHAN_USED_KEY]
            ):
                optimizations_possible = True
                further_optims_str += (
                    f"Module {fqn} can be configured to use per_channel quantization.\n"
                )

        if optimizations_possible:
            further_optims_str += "To use per_channel quantization, make sure the qconfig has a per_channel weight observer."
        else:
            further_optims_str += "No further per_channel optimizations possible."

        # return the string and the dictionary form of same information
        return (further_optims_str, per_channel_info)


class DynamicStaticDetector(DetectorBase):
    r"""
    Determines whether dynamic or static quantization is more appropriate for a given module.

    Takes advantage of the ModelReportObserver that records range information.
    Stationary distribution of data are strictly above tolerance level for the comparison statistic:

        S = average_batch_activation_range/epoch_activation_range

    Nonstationary distributions are below or at the tolerance level for this metric.

    If the distribution of data right after the module is non-stationary, recommend dynamic quantization
        Otherwise recommend static quantization

    Args:
        tolerance (float, optional): The threshold where S metric is stationary above and non-stationary otherwise. Default: 0.5
    """

    # names for the pre and post observers that are inserted
    DEFAULT_PRE_OBSERVER_NAME = "model_report_pre_observer"
    DEFAULT_POST_OBSERVER_NAME = "model_report_post_observer"

    # naming conventions for stationary vs non-stationary data
    STATIONARY_STR = "stationary"
    NON_STATIONARY_STR = "non-stationary"

    # naming for activation
    INPUT_ACTIVATION_PREFIX = "input_activation_"
    OUTPUT_ACTIVATION_PREFIX = "output_activation_"

    # naming conventions for the keys of the return module info
    TOLERANCE_KEY = "dynamic_static_tolerance"
    DEFAULT_DYNAMIC_REC_KEY = "dynamic_recommended"
    PRE_OBS_COMP_STAT_KEY = INPUT_ACTIVATION_PREFIX + "dynamic_static_comp_stat"
    POST_OBS_COMP_STAT_KEY = OUTPUT_ACTIVATION_PREFIX + "dynamic_static_comp_stat"
    PRE_OBS_DATA_DIST_KEY = (
        INPUT_ACTIVATION_PREFIX + "dynamic_static_data_classification"
    )
    POST_OBS_DATA_DIST_KEY = (
        OUTPUT_ACTIVATION_PREFIX + "dynamic_static_data_classification"
    )
    IS_CURRENTLY_SUPPORTED_KEY = "is_dynamic_supported"

    # modules that are supported both dynamic and static for this report function
    DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED = {nn.Linear}

    # modules that will be supported soon for both
    DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED = {nn.Conv1d, nn.Conv2d, nn.Conv3d}

    def __init__(self, tolerance=0.5):
        super().__init__()

        # set tolerance level and initialize a set to keep track of useful fqn locations
        self.tolerance = tolerance
        self.useful_observer_fqns: set[str] = set()

    def determine_observer_insert_points(
        self, prepared_fx_model: GraphModule
    ) -> dict[str, dict[str, Any]]:
        r"""
        Determines where observers need to be inserted for the Dynamic vs Static detector.
        For this detector, we want to place observers on either side of linear layers in the model.

        Currently inserts observers for:
            linear layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """

        # observer for this detector is ModelReportObserver
        obs_ctr = ModelReportObserver

        # return dict
        obs_fqn_to_info: dict[str, dict[str, Any]] = {}

        for fqn, module in prepared_fx_model.named_modules():
            # make sure module is supported
            if self._is_supported(module, insert=True):
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(),
                    DETECTOR_IS_POST_OBS_KEY: False,
                    DETECTOR_OBS_ARGS_KEY: targeted_node.args,
                }

                # add entry for post-observer
                post_obs_fqn = fqn + "." + self.DEFAULT_POST_OBSERVER_NAME

                obs_fqn_to_info[post_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(),
                    DETECTOR_IS_POST_OBS_KEY: True,
                    DETECTOR_OBS_ARGS_KEY: (targeted_node,),
                }

        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        r"""returns the string name of this detector"""
        return "dynamic_vs_static_detector"

    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # run the helper function to populate the dictionary
        dynamic_static_info = self._generate_dict_info(model)

        # we actually have a qconfig info object we are populating
        module_fqn_to_detector_qconfig_info = {}

        for module_fqn in dynamic_static_info:
            # create a detector info instance
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)

            # see if per channel quantization is supported
            dynamic_static_recommended: bool = dynamic_static_info[module_fqn][
                self.DEFAULT_DYNAMIC_REC_KEY
            ]
            detector_qconfig_info.is_activation_dynamic = dynamic_static_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info

        return module_fqn_to_detector_qconfig_info

    def _is_supported(self, module: nn.Module, insert: bool = False) -> bool:
        r"""Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
        # check to see if module is of a supported type
        is_supported_type = any(
            isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED
        )

        # check if it will be supported
        future_supported_type = any(
            isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED
        )

        # supported
        supported = is_supported_type or future_supported_type

        # this is check for observer insertion
        if insert:
            return supported
        else:
            # this is for report gen and we also need to check if it contains observers
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME) and hasattr(
                module, self.DEFAULT_POST_OBSERVER_NAME
            )
            return supported and has_obs

    def _generate_dict_info(self, model: GraphModule) -> dict[str, Any]:
        r"""
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a Dictionary mapping modules with ModelReportObservers around them to:
                whether dynamic quantization is recommended
                their S metric of input to module
                whether input to module is stationary or non-stationary
                their S metric of output of module
                whether output of module is stationary or non-stationary
                the tolerance level to decided whether input/output is stationary or non-stationary
                whether it is currently supported or planned for the future
        """
        # store modules dynamic vs static information
        module_dynamic_static_info = {}

        # This for loop goes through the modules, and extracts all relevant information into module_dynamic_static_info
        #   This information primary includes whether the data distributions around a supported module is stationary or not
        #   Based on this, it is recorded whether dynamic or static quantization is recommended

        # loop through all submodules included nested ones
        for fqn, module in model.named_modules():
            # if module is Linear has the ModelReportObserver attached to it
            if self._is_supported(module):
                # get pre and post observers for the module
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                post_obs = getattr(module, self.DEFAULT_POST_OBSERVER_NAME)

                # get the statistics for each module
                pre_stat = pre_obs.get_batch_to_epoch_ratio()
                post_stat = post_obs.get_batch_to_epoch_ratio()

                # record module, pre and post stat, and whether to do dynamic or static based off it
                # true if post observer data distribution is non-stationary, false if it's stationary
                dynamic_recommended = post_stat <= self.tolerance

                # specify the classifications for whether data distributions considered stationary or non-stationary
                pre_obs_dist_classif = (
                    self.STATIONARY_STR
                    if pre_stat > self.tolerance
                    else self.NON_STATIONARY_STR
                )
                post_obs_dist_classif = (
                    self.STATIONARY_STR
                    if post_stat > self.tolerance
                    else self.NON_STATIONARY_STR
                )

                # check if current support or future support
                is_supported_type = any(
                    isinstance(module, x)
                    for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED
                )

                # store the set of important information for this module
                module_info = {
                    self.TOLERANCE_KEY: self.tolerance,
                    self.DEFAULT_DYNAMIC_REC_KEY: dynamic_recommended,
                    self.PRE_OBS_COMP_STAT_KEY: pre_stat,
                    self.PRE_OBS_DATA_DIST_KEY: pre_obs_dist_classif,
                    self.POST_OBS_COMP_STAT_KEY: post_stat,
                    self.POST_OBS_DATA_DIST_KEY: post_obs_dist_classif,
                    self.IS_CURRENTLY_SUPPORTED_KEY: is_supported_type,
                }

                module_dynamic_static_info[fqn] = module_info

        return module_dynamic_static_info

    def generate_detector_report(
        self, model: GraphModule
    ) -> tuple[str, dict[str, Any]]:
        r"""
        Determines whether dynamic or static quantization is more appropriate for a given module.

        Takes advantage of the ModelReportObserver that records range information.
        Stationary distribution of data are strictly above tolerance level for the comparison statistic:

            S = average_batch_activation_range/epoch_activation_range

        Nonstationary distributions are below or at the tolerance level for this metric.

        If the distribution of data right after the module is non-stationary, recommend dynamic quantization
            Otherwise recommend static quantization

        This will then generate suggestions for dynamic vs static quantization focused around Linear.

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a tuple with two elements:
            String report of whether dynamic or static quantization is recommended for certain modules
            Dictionary mapping modules with ModelReportObservers around them to:
                whether dynamic quantization is recommended
                their S metric of input to module
                whether input to module is stationary or non-stationary
                their S metric of output of module
                whether output of module is stationary or non-stationary
                the tolerance level to decided whether input/output is stationary or non-stationary
                whether it is currently supported or planned for the future
        """

        # get the dictionary of the information to format the string report
        module_dynamic_static_info = self._generate_dict_info(model)

        dynamic_vs_static_string = "Dynamic vs. Static Quantization suggestions: \n"

        modules_added: bool = False  # check to make sure at least 1 module added.

        dynamic_benefit = (
            " You will get more accurate results if you use dynamic quantization"
        )
        static_benefit = (
            " You can increase model efficiency if you use static quantization"
        )
        future_support_str = (
            ". This layer is not yet supported for dynamic quantization"
        )
        # This for loop goes through the information collected in module_dynamic_static_info and:
        #   Populates the string based report with the information from module_dynamic_static_info
        #   Compiles the complete report by appending relevant formatted strings

        for module_fqn in module_dynamic_static_info:
            # there is at least 1 module for suggestion
            modules_added = True
            module_info = module_dynamic_static_info[module_fqn]
            suggestion_string_template = (
                "For module {} it is suggested to use {} quantization because {}.\n"
            )

            # decide what string formatting values will be
            quantization_type = ""
            quantization_reasoning = "the distribution of data before {} is {} and the distribution after is {}."

            benefit_str = ""

            # strings for if dynamic quantized per tensor is needed
            recommend_per_tensor = (
                ". We recommend to add a {} before this module if it is static."
            )
            rec_lay_to_add = "dynamic quantize per tensor layer"
            dynamic_per_tensor_string = recommend_per_tensor.format(rec_lay_to_add)
            dynamic_per_tensor_reasoning_string = " This is because the input to this module has a non-stationary distribution"

            # start composing explanation
            if module_info[self.DEFAULT_DYNAMIC_REC_KEY]:
                quantization_type = "dynamic"
                # check if currently supported or future supported
                benefit_str = dynamic_benefit
                if not module_info[self.IS_CURRENTLY_SUPPORTED_KEY]:
                    benefit_str += future_support_str
            else:
                quantization_type = "static"
                benefit_str = static_benefit

            # now set the quantization explanation string
            quantization_reasoning = (
                quantization_reasoning.format(
                    module_fqn,
                    module_info[self.PRE_OBS_DATA_DIST_KEY],
                    module_info[self.POST_OBS_DATA_DIST_KEY],
                )
                + benefit_str
            )

            # if we have a non-stationary input -> linear -> stationary we suggested static
            # however, we want to also recommend they add a dynamic quantize per tensor right if this change is made
            if (
                module_info[self.PRE_OBS_DATA_DIST_KEY] == self.NON_STATIONARY_STR
                and module_info[self.POST_OBS_DATA_DIST_KEY] == self.STATIONARY_STR
            ):
                quantization_reasoning = (
                    quantization_reasoning
                    + dynamic_per_tensor_string
                    + dynamic_per_tensor_reasoning_string
                )

            # format the overall suggestion string with the specific inputs
            module_suggestion_string = suggestion_string_template.format(
                module_fqn, quantization_type, quantization_reasoning
            )

            # append to overall suggestion
            dynamic_vs_static_string += module_suggestion_string

        if not modules_added:
            dynamic_vs_static_string += "No applicable layers for suggestions. Only linear and conv are valid.\n"

        # return the string as well as the dictionary of information
        return (dynamic_vs_static_string, module_dynamic_static_info)


class InputWeightEqualizationDetector(DetectorBase):
    r"""
    Determines whether input-weight equalization can help improve quantization for certain modules.

    Specifically, this list of modules includes:
        linear
        conv

    Determines whether input-weight equalization is recommended based on the comp stat:
        s_c = sqrt(w_c/W)/sqrt(i_c/I)
        where:
            w_c is range of weight for channel c, W is range of weight over all channels
            i_c is range of input for channel c, I is range of input over all channels

        if s_c >= threshold or <= 1 / threshold, recommends input-weight equalization

    Args:
        ratio_threshold (float): The threshold for s_c to determine if input-weight equalization is suggested
            Should be between 0 and 1 (both non-inclusive)
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for s_c to determine if input-weight equalization is suggested
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine input weight equalization

    * :attr:`SUPPORTED_MODULES`: This specifies the modules that are supported for input-weight equalization

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """

    SUPPORTED_MODULES: set[Callable] = {
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nnqat.Linear,
        nnqat.Conv1d,
        nnqat.Conv2d,
        nnqat.Conv3d,
    }

    # names for the pre and post observers that are inserted
    DEFAULT_PRE_OBSERVER_NAME: str = "model_report_pre_observer"

    # weight / activation prefix for each of the below info
    WEIGHT_PREFIX = "weight_"
    ACTIVATION_PREFIX = "input_activation_"

    # string names for keys of info dictionaries
    PER_CHANNEL_MAX_KEY = "per_channel_max"
    PER_CHANNEL_MIN_KEY = "per_channel_min"
    GLOBAL_MAX_KEY = "global_max"
    GLOBAL_MIN_KEY = "global_min"

    # keys for return dict of recommendations
    RECOMMENDED_KEY = "input_weight_equalization_recommended"
    COMP_METRIC_KEY = "input_weight_channel_comparison_metrics"
    THRESHOLD_KEY = "input_weight_threshold"
    CHANNEL_KEY = "input_weight_channel_axis"

    # default weight and info strings
    WEIGHT_STR = "weight"
    INPUT_STR = "input"

    # default for what ratio we recommend input weight
    DEFAULT_RECOMMEND_INPUT_WEIGHT_CHANNEL_RATIO = 0.4

    def __init__(self, ratio_threshold: float, ch_axis: int = 1):
        # ensure passed in inputs are valid
        if ratio_threshold <= 0 or ratio_threshold >= 1:
            raise ValueError("Make sure threshold is > 0 and < 1")

        # initialize attributes based on args
        self.ratio_threshold: float = ratio_threshold
        self.ch_axis: int = ch_axis

    def _is_supported(self, module: nn.Module, insert: bool = False) -> bool:
        r"""Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
        # check to see if module is of a supported type
        is_supported_type = any(type(module) is x for x in self.SUPPORTED_MODULES)

        # this is check for observer insertion
        if insert:
            return is_supported_type
        else:
            # this is for report gen and we also need to check if it contains observers
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
            return is_supported_type and has_obs

    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # run the helper function to populate the dictionary
        # find the range of inputs
        input_values: dict[str, dict] = self._extract_input_info(model)

        # find the range of weights
        weight_values: dict[str, dict] = self._extract_weight_info(model)

        # calculate per_channel comparison statistic s_c
        comp_stats: dict[str, torch.Tensor] = self._generate_comparison_values(
            input_values, weight_values
        )

        # generate the return dictionary
        input_weight_equalization_info: dict[str, dict] = self._generate_dict_info(
            input_values, weight_values, comp_stats
        )

        # we actually have a qconfig info object we are populating
        module_fqn_to_detector_qconfig_info = {}

        for module_fqn in input_weight_equalization_info:
            # create a detector info instance
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)

            # see if per channel quantization is supported
            input_weight_recommended: bool = input_weight_equalization_info[module_fqn][
                self.RECOMMENDED_KEY
            ]
            detector_qconfig_info.is_equalization_recommended = input_weight_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info

        return module_fqn_to_detector_qconfig_info

    def determine_observer_insert_points(
        self, prepared_fx_model: GraphModule
    ) -> dict[str, dict[str, Any]]:
        r"""Determines where observers need to be inserted for the Input Weight Equalization Detector.
        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            linear layers
            conv layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """

        # observer for this detector is ModelReportObserver
        obs_ctr = ModelReportObserver

        # return dict
        obs_fqn_to_info: dict[str, dict[str, Any]] = {}

        for fqn, module in prepared_fx_model.named_modules():
            # check to see if module is of a supported type
            if self._is_supported(module, insert=True):
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis),
                    DETECTOR_IS_POST_OBS_KEY: False,
                    DETECTOR_OBS_ARGS_KEY: targeted_node.args,
                }

        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        r"""Returns the name of this detector"""
        return "input_weight_equalization_detector"

    def _extract_input_info(self, model: GraphModule) -> dict[str, dict]:
        r"""
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the input information for each observer returns it

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping relevant module fqns (str) to a dict with keys:
            "input_activation_per_channel_max" : maps to the per_channel max values
            "input_activation_per_channel_min" : maps to the per_channel min values
            "input_activation_global_max" : maps to the global max recorded
            "input_activation_global_min" : maps to the global min recorded
        """

        # return dictionary mapping observer fqns to desired info
        input_info: dict[str, dict] = {}

        for fqn, module in model.named_modules():
            # if module is supported and it has a pre-observer
            if self._is_supported(module):
                # get pre observer for the module
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)

                input_info[fqn] = {
                    self.ACTIVATION_PREFIX + self.PER_CHANNEL_MAX_KEY: pre_obs.max_val,
                    self.ACTIVATION_PREFIX + self.PER_CHANNEL_MIN_KEY: pre_obs.min_val,
                    self.ACTIVATION_PREFIX + self.GLOBAL_MAX_KEY: max(pre_obs.max_val),
                    self.ACTIVATION_PREFIX + self.GLOBAL_MIN_KEY: min(pre_obs.min_val),
                }

        return input_info

    def _extract_weight_info(self, model: GraphModule) -> dict[str, dict]:
        r"""
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the weight information for each layer an observer is attached to.

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping module fqns (str) to a dict with keys:
            "per_channel_max" : maps to the per_channel max values
            "per_channel_min" : maps to the per_channel min values
            "global_max" : maps to the global max recorded
            "global_min" : maps to the global min recorded
        """
        # return dictionary mapping observer fqns to desired info
        weight_info: dict[str, dict] = {}

        for fqn, module in model.named_modules():
            # if module is supported and it has a pre-observer
            if self._is_supported(module):
                # we don't need actual observer, just the module weights
                # calculate min and max vals
                device = module.weight.device
                min_val: torch.Tensor = torch.tensor([float("inf")], device=device)
                max_val: torch.Tensor = torch.tensor([float("-inf")], device=device)
                x_copy = module.weight
                x_dim = x_copy.size()

                new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
                new_axis_list[self.ch_axis] = 0
                new_axis_list[0] = self.ch_axis
                y = x_copy.permute(new_axis_list)

                # Need to match dtype of min/max because the updates to buffers
                # are done in place and types need to match for comparisons
                y = y.to(min_val.dtype)
                y = torch.flatten(y, start_dim=1)
                if min_val.numel() == 0 or max_val.numel() == 0:
                    min_val, max_val = torch.aminmax(y, dim=1)
                else:
                    min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
                    min_val = torch.min(min_val_cur, min_val)
                    max_val = torch.max(max_val_cur, max_val)

                weight_info[fqn] = {
                    self.WEIGHT_PREFIX + self.PER_CHANNEL_MAX_KEY: max_val,
                    self.WEIGHT_PREFIX + self.PER_CHANNEL_MIN_KEY: min_val,
                    self.WEIGHT_PREFIX + self.GLOBAL_MAX_KEY: max(max_val),
                    self.WEIGHT_PREFIX + self.GLOBAL_MIN_KEY: min(min_val),
                }

        return weight_info

    def _calculate_range_ratio(
        self, info_dict: dict, info_str: str, module_fqn: str
    ) -> torch.Tensor:
        r"""
        Takes in an info dict and calculates the s_c matrix.

        Args:
            info_dict (dict): A dictionary of either input or weight range info
            info_str (str): A str describing whether currently looking at weight or input info
                Either "weight" or "input"
            module_fqn (str): The fqn of the module we are looking at

        Returns a tensor of values, where each value is the s_c stat for a different channel
        """
        # calculate the ratios of the info
        # get the prefix str
        prefix_str = (
            self.ACTIVATION_PREFIX if info_str == self.INPUT_STR else self.WEIGHT_PREFIX
        )

        per_channel_range = (
            info_dict[prefix_str + self.PER_CHANNEL_MAX_KEY]
            - info_dict[prefix_str + self.PER_CHANNEL_MIN_KEY]
        )
        global_range = (
            info_dict[prefix_str + self.GLOBAL_MAX_KEY]
            - info_dict[prefix_str + self.GLOBAL_MIN_KEY]
        )

        if global_range == 0:
            range_zero_explanation = "We recommend removing this channel as it doesn't provide any useful information."
            raise ValueError(
                f"The range of the {info_str} data for module {module_fqn} is 0, "
                f"which means you have a constant value channel. {range_zero_explanation}"
            )

        ratio = per_channel_range / global_range

        return ratio

    def _generate_comparison_values(
        self, input_info: dict, weight_info: dict
    ) -> dict[str, torch.Tensor]:
        r"""
        Takes in the information on the min and max values of the inputs and weights and:
            Calculates the comp stat for each channel: s_c = sqrt(w_c/W)/sqrt(i_c/I)

        Args:
            input_info (dict): A dict mapping each observer to input range information
            weight_info (dict): A dict mapping each observer to weight range information

        Returns a dict mapping relevant observer fqns (str) to a 1-D tensor.
            Each value is a different s_c value for a different channel
        """
        # create return dictionary for each observer
        module_fqn_to_channel: dict[str, torch.Tensor] = {}

        # for each module (both passed in dicts should have same keys)
        for module_fqn in input_info:
            # raise error if not in weight info
            if module_fqn not in weight_info:
                raise KeyError(
                    f"Unable to find weight range stats for module {module_fqn}"
                )

            # calculate the ratios of the weight info and input info
            weight_ratio = self._calculate_range_ratio(
                weight_info[module_fqn], self.WEIGHT_STR, module_fqn
            )
            input_ratio = self._calculate_range_ratio(
                input_info[module_fqn], self.INPUT_STR, module_fqn
            )

            # if mismatched size, because of grouping, we want to replicate weight enough times
            weight_channels = len(weight_ratio)
            input_channels = len(input_ratio)
            if weight_channels != input_channels:
                # we try to replicate
                if input_channels % weight_channels != 0:
                    raise AssertionError(
                        "input channels should be divisible by weight channels."
                    )
                # get replication factor
                rep_factor: int = input_channels // weight_channels

                # weight ratio is (n,), input ratio is (k,), we just repeat weight ratio k // n
                weight_ratio = weight_ratio.repeat(rep_factor)

            # calculate the s metric per channel
            s = torch.sqrt(weight_ratio) / torch.sqrt(input_ratio)
            module_fqn_to_channel[module_fqn] = s

        # return compiled observer ratios
        return module_fqn_to_channel

    def _generate_dict_info(
        self, input_info: dict, weight_info: dict, comp_stats: dict
    ) -> dict[str, dict]:
        r"""
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            input_info (dict): A dict mapping each module to input range information
            weight_info (dict): A dict mapping each module to weight range information
            comp_stats (dict): A dict mapping each module to its corresponding comp stat

        Returns a dictionary mapping each module with relevant ModelReportObservers around them to:
            whether input weight equalization is recommended
            their s_c metric compared to the threshold
            the threshold used to make the recommendation
            the channel used for recording data
            the input channel range info
            the weight channel range info
        """
        # store modules input weight equalization info
        input_weight_equalization_info: dict[str, dict] = {}

        # for each module we add separate set of suggestions
        for module_fqn in input_info:
            # get relevant info for this module
            mod_input_info: dict = input_info[module_fqn]
            mod_weight_info: dict = weight_info[module_fqn]
            mod_comp_stat: dict = comp_stats[module_fqn]

            # decide if each channel should have input weight equalization or not
            channel_rec_vals: list = []

            for val in mod_comp_stat:
                float_rep: float = val.item()

                # decide if recommending input weight equalization
                recommended: bool = (
                    float_rep >= self.ratio_threshold
                    and float_rep <= 1 / self.ratio_threshold
                )
                channel_rec_vals.append(recommended)

            # build the return dict input
            # also unpack input and weight dicts into it
            input_weight_equalization_info[module_fqn] = {
                self.RECOMMENDED_KEY: channel_rec_vals,
                self.COMP_METRIC_KEY: mod_comp_stat,
                self.THRESHOLD_KEY: self.ratio_threshold,
                self.CHANNEL_KEY: self.ch_axis,
                **mod_input_info,
                **mod_weight_info,
            }

        # return our compiled info for each module
        return input_weight_equalization_info

    def generate_detector_report(
        self, model: GraphModule
    ) -> tuple[str, dict[str, Any]]:
        r"""
        Determines whether input weight equalization is appropriate for a given module.

        Takes advantage of the ModelReport Observer which records per channel information of input range
        It then uses the passed in weight info in conjunction to compute the desired ratio
        Finally, it gives suggestions based on this information for each module of interest

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a tuple with two elements:
            String report of whether input weight equalization is recommended for certain modules
            Dictionary mapping modules of interest to:
                whether input weight equalization is recommended
                their s_c metric compared to the threshold
                the threshold used to make the recommendation
                the channel used for recording data
                the input channel range info
                the weight channel range info
        """

        # find the range of inputs
        input_values: dict[str, dict] = self._extract_input_info(model)

        # find the range of weights
        weight_values: dict[str, dict] = self._extract_weight_info(model)

        # calculate per_channel comparison statistic s_c
        comp_stats: dict[str, torch.Tensor] = self._generate_comparison_values(
            input_values, weight_values
        )

        # generate the return dictionary
        input_weight_equalization_info: dict[str, dict] = self._generate_dict_info(
            input_values, weight_values, comp_stats
        )

        # now we can generate report based on this information
        input_weight_string = "Input-Weight Equalization suggestions: \n"

        # some strings to be formatted depending on module we are adding
        module_suggestion_str = "For Module {} looked at with axis {}: \n"
        channel_suggestion_str = (
            "\tWe suggest {} input weight equalization because {}\n"
        )
        use_str = "to use"
        no_use_str = "to not use"
        input_weight_benefit_str = "{}/{} channels would benefit and we expect significant reduction in quantization error."
        input_weight_non_benefit_reasoning = (
            "{}/{} channels benefitting from input-weight equalization being applied."
        )
        input_weight_non_benefit_str = "we don't expect much improvement from input-weight equalization based on {}"

        # added module check
        added_module: bool = False

        # compile the suggestion string
        for module_fqn in input_weight_equalization_info:
            # we added at least 1 module
            added_module = True
            # add the module level description
            input_weight_string += module_suggestion_str.format(
                module_fqn, self.ch_axis
            )

            mod_info: dict[str, Any] = input_weight_equalization_info[module_fqn]

            # gather info on how many channels would benefit from input weight and
            recommendation_per_channel: torch.Tensor = mod_info[self.RECOMMENDED_KEY]
            num_recs = sum(recommendation_per_channel)

            if (
                num_recs / len(recommendation_per_channel)
                >= self.DEFAULT_RECOMMEND_INPUT_WEIGHT_CHANNEL_RATIO
            ):
                input_benefit_formatted = input_weight_benefit_str.format(
                    num_recs, len(recommendation_per_channel)
                )
                channel_str = channel_suggestion_str.format(
                    use_str, input_benefit_formatted
                )
                input_weight_string += channel_str
            else:
                non_benefit_reason_formatted = (
                    input_weight_non_benefit_reasoning.format(
                        num_recs, len(recommendation_per_channel)
                    )
                )
                non_benefit_str = input_weight_non_benefit_str.format(
                    non_benefit_reason_formatted
                )
                channel_str = channel_suggestion_str.format(no_use_str, non_benefit_str)
                input_weight_string += channel_str

        # if no modules looked at, amend return string
        if not added_module:
            input_weight_string += (
                "No applicable layers for suggestions. Only linear and conv valid.\n"
            )

        # return a tuple with the string explanation and the compiled dict info
        return (input_weight_string, input_weight_equalization_info)


class OutlierDetector(DetectorBase):
    r"""
    Determines whether there are significant outliers in activation data around a certain layer.

    This is ideally used in conjunction with information on stationary vs. non-stationary distribution:
        If the data is stationary, and there are significant outliers, then we want to flag them
        We want to do this on a per channel basis for detecting outliers

    Determines whether activation data is flagged as outlier based on if data is stationary and:
        p_r = avg(100th percentile / "reference_percentile"th percentile)
        where:
            p_r is average percentile ratio across all batches in the epoch
            reference_percentile is a percentile values between 0 and 100 exclusive

        if p_r is above some threshold, then we consider the activations to have significant outliers

    Args:
        ratio_threshold (float, optional): The threshold for p_r to determine if there are outliers in activations
            Should be >= 1
            Default: 3.5
        reference_percentile (float, optional): The denominator to find the relative scale of the 100th percentile
            Should be between 0 and 1
            Default: 0.975
        fraction_batches_used_threshold (float, optional): Threshold of fraction of batches per channel to determine outlier
            If fraction is below this, we deem number of samples used to calculate outliers as insignificant and alert user
            regardless of whether we detected outliers or not in channel to take a closer look at channel results
            Should be between 0 and 1
            Default: 0.95
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for p_r to determine if there are outliers in activations
        The p_r value (average ratio of 100th percentile/reference_percentile) is compared to ratio_threshold
        If it is significantly greater, then we consider it an outlier
        This threshold was calculated based on the ratio of the percentiles in a normal distribution
        The calculations behind value choice: https://drive.google.com/file/d/1N2wdtXWI-kOH8S7HH4-PYB_NmqzZil4p/view?usp=sharing

    * :attr:`reference_percentile`: The denominator of the top fraction to find the relative scale of the 100th percentile
        Should be between 0 and 1
        The calculations behind value choice: https://drive.google.com/file/d/1N2wdtXWI-kOH8S7HH4-PYB_NmqzZil4p/view?usp=sharing

    * :attr:`fraction_batches_used_threshold`: The fraction of batches to determine outliers for each channel should be above this
        Some batches may not be used because of 0-based errors, so this is to ensure a good amount of the total batches are used
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine outliers

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """

    # names for the pre observers that are inserted
    DEFAULT_PRE_OBSERVER_NAME: str = "model_report_pre_observer"

    # pre activation prefix
    INPUT_ACTIVATION_PREFIX = "input_activation_"

    # names for dict keys
    OUTLIER_KEY = "outliers_detected"
    NUM_BATCHES_KEY = "outlier_detection_batches_used"
    IS_SUFFICIENT_BATCHES_KEY = "outlier_detection_is_sufficient_batches"
    COMP_METRIC_KEY = "outlier_detection_percentile_ratios"
    RATIO_THRES_KEY = "outlier_detection_ratio_threshold"
    REF_PERCENTILE_KEY = "outlier_detection_reference_percentile"
    CHANNEL_AXIS_KEY = "outlier_detection_channel_axis"
    MAX_VALS_KEY = INPUT_ACTIVATION_PREFIX + "per_channel_max"
    CONSTANT_COUNTS_KEY = "constant_batch_counts"

    def __init__(
        self,
        ratio_threshold: float = 3.5,
        reference_percentile: float = 0.975,
        fraction_batches_used_threshold: float = 0.95,
        ch_axis: int = 1,
    ):
        # initialize the variables of interest
        self.ratio_threshold = ratio_threshold

        # make sure passed in percentile is valid
        if reference_percentile < 0 or reference_percentile > 1:
            raise AssertionError("reference_percentile must be between 0 and 1")
        if not (
            fraction_batches_used_threshold >= 0
            and fraction_batches_used_threshold <= 1
        ):
            raise AssertionError(
                "fraction_batches_used_threshold must be between 0 and 1"
            )
        self.reference_percentile = reference_percentile
        self.fraction_batches_used_threshold = fraction_batches_used_threshold
        self.ch_axis = ch_axis

    def get_detector_name(self) -> str:
        r"""Returns the name of this detector"""
        return "outlier_detector"

    def _supports_insertion(self, module: nn.Module) -> bool:
        r"""Returns whether the given module is supported for observers insertion

        Any module that doesn't have children and isn't an observer itself is supported

        Args
            module: The module to check and ensure is supported

        Returns True if the module is supported by observer, False otherwise
        """
        # case for insertion of module
        # check if the module has any children and isn't observer
        num_children = len(list(module.children()))
        return num_children == 0 and not _is_activation_post_process(module)

    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # currently doesn't do anything for outlier detector
        return {}

    def _supports_report_gen(self, module: nn.Module) -> bool:
        r"""Returns whether the given module is supported for report generation

        Any module that has a model report pre-observer is supported

        Args
            module: The module to check and ensure is supported

        Returns True if the module is supported by observer, False otherwise
        """
        return hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)

    def determine_observer_insert_points(
        self, prepared_fx_model: GraphModule
    ) -> dict[str, dict[str, Any]]:
        r"""Determines where observers need to be inserted for the Outlier Detector.

        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            all layers that do not have children (leaf level layers)

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """
        # observer for this detector is ModelReportObserver
        obs_ctr = ModelReportObserver

        # return dict
        obs_fqn_to_info: dict[str, dict[str, Any]] = {}

        for fqn, module in prepared_fx_model.named_modules():
            # check to see if module is of a supported type
            if self._supports_insertion(module):
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(
                        ch_axis=self.ch_axis, comp_percentile=self.reference_percentile
                    ),
                    DETECTOR_IS_POST_OBS_KEY: False,
                    DETECTOR_OBS_ARGS_KEY: targeted_node.args,
                }

        return obs_fqn_to_info

    def _calculate_outlier_info(
        self,
        percentile_ratios: torch.Tensor,
        counted_batches: torch.Tensor,
        total_batches: int,
    ) -> dict[str, list[bool]]:
        r"""
        Gives info on whether the percentile ratios calculated would be considered outliers
        Also gives information on whether the collected data is statistically significant to make this claim

        Args:
            percentile_ratios (torch.Tensor): The average percentile_ratios per channel calculated by the observer
            counted_batches (torch.Tensor): The number of batches used for average calculation per tensor
            total_batches (int): The total number of batches that passed through observer in this epoch

        Returns a dictionary mapping:
            "outliers_detected" : list of bools per channel that are true if it is considered an outlier
            "is_sufficient_batches": if o_r was >= fraction_batches_used_threshold:
                where o_r = counted_batches / total_batches
        """
        outlier_dict: dict[str, list[bool]] = {
            self.OUTLIER_KEY: [],
            self.IS_SUFFICIENT_BATCHES_KEY: [],
        }

        # get both as flattened lists for easy mapping
        ratios_list: list = percentile_ratios.tolist()
        num_batches_list: list = counted_batches.tolist()

        # calculate whether channels were statistically significant
        significant_size = [
            batch_size / total_batches >= self.fraction_batches_used_threshold
            for batch_size in num_batches_list
        ]
        outlier_dict[self.IS_SUFFICIENT_BATCHES_KEY] = significant_size

        # calculate for each channel whether it's an outlier or not based on ratio
        outlier_detected = [ratio > self.ratio_threshold for ratio in ratios_list]
        outlier_dict[self.OUTLIER_KEY] = outlier_detected

        # return the dictionary with the two lists
        return outlier_dict

    def _generate_info_dict(self, model: GraphModule) -> dict[str, dict]:
        r"""
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping relevant module fqns to:
            whether there were outliers found in activation before
            the number of batches used for each channel
            whether fraction of applicable batches used is above fraction_batches_used_threshold
            their p_r metric compared to the threshold
            the threshold used to make the recommendation
            the reference_percentile used to make the recommendation
            the channel axis used to determine individual channels
            the constant batch counts per channel
            the per channel max values
        """
        # return dictionary mapping observer fqns to desired info
        info_dict: dict[str, dict] = {}

        for fqn, module in model.named_modules():
            # if module is supported and it has a pre-observer
            if self._supports_report_gen(module):
                # get pre observer for the module
                pre_obs: ModelReportObserver = getattr(
                    module, self.DEFAULT_PRE_OBSERVER_NAME
                )

                # get the number of batches and calculated ratio thresholds
                num_batches: torch.Tensor = pre_obs.percentile_batches_tracked
                average_ratios: torch.Tensor = pre_obs.average_percentile_ratio
                channel_batch_cnts: torch.Tensor = pre_obs.constant_channels
                total_batches: int = pre_obs.num_batches_tracked

                # also get the max values
                max_vals: torch.Tensor = pre_obs.max_val

                # we have to specifically modify how we are recording negative ratio for pre-relu layers
                for index, ratio_val in enumerate(average_ratios):
                    # check if we have a negative ratio
                    # a ratio might be negative if we have a situation where the 100th percentile is
                    # > 0 while the nth percentile is < 0, in which case this would not be detected
                    # as an outlier. Since we care more about magnitude, we make it positive.
                    if ratio_val.item() < 0:
                        # first make it positive
                        average_ratios[index] = -ratio_val

                    if ratio_val.item() < 1:
                        # if it's less than 1 we have the flip it as well
                        average_ratios[index] = 1 / ratio_val

                outlier_calcs = self._calculate_outlier_info(
                    average_ratios, num_batches, total_batches
                )

                # calculate whether ratios were outliers
                info_dict[fqn] = {
                    self.CHANNEL_AXIS_KEY: self.ch_axis,
                    self.REF_PERCENTILE_KEY: self.reference_percentile,
                    self.RATIO_THRES_KEY: self.ratio_threshold,
                    self.COMP_METRIC_KEY: average_ratios,
                    self.NUM_BATCHES_KEY: num_batches,
                    self.OUTLIER_KEY: outlier_calcs[self.OUTLIER_KEY],
                    self.IS_SUFFICIENT_BATCHES_KEY: outlier_calcs[
                        self.IS_SUFFICIENT_BATCHES_KEY
                    ],
                    self.CONSTANT_COUNTS_KEY: channel_batch_cnts,
                    self.MAX_VALS_KEY: max_vals,
                }

        return info_dict

    def generate_detector_report(
        self, model: GraphModule
    ) -> tuple[str, dict[str, Any]]:
        r"""
        Determines whether input weight equalization is appropriate for a given module.

        Takes advantage of the ModelReport Observer which records the relevant percentile information

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a tuple with two elements:
            String report of whether there are outliers in the activations around certain modules
            Dictionary mapping modules of interest to:
                whether there were outliers found in activation before
                the number of batches used for each channel
                whether fraction of applicable batches used is above fraction_batches_used_threshold
                their p_r metric compared to the threshold
                the threshold used to make the recommendation
                the reference_percentile used to make the recommendation
                the channel axis used to determine individual channels
                the constant batch counts per channel
                the per channel max values
        """
        # generate the information dictionary of outlier information
        info_dict = self._generate_info_dict(model)

        # now we can generate report based on this information
        outlier_string = "Outlier detection report: \n"

        # added module check
        added_module: bool = False

        # some strings to be formatted depending on module we are adding
        module_suggestion_str = "For Module {} looked at with axis {}: \n"
        channel_suggestion_str = "\tFor channel {}, we found outliers in the preceding activation data with {}.\n"
        channel_max_value_str = "a max value across all batches of {}"
        note_string = "Note: outlier detection is only reliable for {}. We recommend {} to ensure the most accurate results."
        note_distribution = "stationary distributions"
        note_rec = "running the static vs. dynamic detector to ensure activation data before modules above is stationary"

        # suggestion for constant batch check since that can make it no outliers
        constant_str = "\tFor channel {}, we found {} constant value batches. {}\n"
        constant_suggestion = "We recommend taking a look at the dict and data to see how frequent this occurred and why."

        # compile the suggestion string
        for module_fqn in info_dict:
            # get module specific info
            mod_info: dict[str, Any] = info_dict[module_fqn]
            # check to see if we already added high level model desc
            added_model_desc = False
            # look at each individual channel and add a suggestion
            for index, outlier_detected in enumerate(mod_info[self.OUTLIER_KEY]):
                if outlier_detected:
                    # we found at least 1 outlier
                    if not added_model_desc:
                        # add the module level description
                        outlier_string += module_suggestion_str.format(
                            module_fqn, self.ch_axis
                        )
                        added_model_desc = True

                    # we mark that we found at least one outlier
                    added_module = True
                    max_value_found_str = channel_max_value_str.format(
                        mod_info[self.MAX_VALS_KEY][index]
                    )
                    channel_str = channel_suggestion_str.format(
                        index, max_value_found_str
                    )
                    outlier_string += channel_str

                # also check if we found constant batch
                if mod_info[self.CONSTANT_COUNTS_KEY][index] != 0:
                    # make sure we add a module level highlight.
                    if not added_model_desc:
                        # add the module level description
                        outlier_string += module_suggestion_str.format(
                            module_fqn, self.ch_axis
                        )
                        added_model_desc = True

                    constant_values_for_channel = mod_info[self.CONSTANT_COUNTS_KEY][
                        index
                    ]
                    formatted_str = constant_str.format(
                        index, constant_values_for_channel, constant_suggestion
                    )
                    outlier_string += formatted_str
                    # we also added at least one thing to description
                    added_module = True

        # if found outlier, give suggestion, else give default response
        if added_module:
            # compose the note string
            note_composed = note_string.format(note_distribution, note_rec)
            outlier_string += note_composed
        else:
            outlier_string += "There were no outliers found in the activations.\n"

        return (outlier_string, info_dict)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any

import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn as nn
from torch.ao.quantization import prepare
from torch.ao.quantization.quantization_mappings import (
    get_default_compare_output_module_list,
)


NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST = {
    nnqd.Linear,
    nnq.Linear,
    nnqd.LSTM,
    nn.LSTM,
}


def _find_match(
    str_list: dict[str, Any] | list[str],
    key_str: str,
    postfix: str,
) -> str | None:
    split_str = key_str.split(".")
    if split_str[-1] == postfix:
        match_string = "".join(key_str.split(".")[0:-1])
        for s2 in str_list:
            pattern1 = "".join(s2.split(".")[0:-1])
            pattern2 = "".join(s2.split(".")[0:-2])
            if match_string == pattern1:
                return s2
            if match_string == pattern2:
                return s2

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        if postfix == "_packed_params":
            match_string = "".join(key_str.split(".")[0:-2])
            if len(match_string) == 0:
                return None
            for s2 in str_list:
                pattern1 = "".join(s2.split(".")[0:-1])
                pattern2 = "".join(s2.split(".")[0:-2])
                if match_string == pattern1:
                    return s2
                if match_string == pattern2:
                    return s2
        return None
    else:
        return None


def compare_weights(
    float_dict: dict[str, Any], quantized_dict: dict[str, Any]
) -> dict[str, dict[str, torch.Tensor]]:
    r"""Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage::

        wt_compare_dict = compare_weights(float_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(
                key,
                compute_error(
                    wt_compare_dict[key]["float"],
                    wt_compare_dict[key]["quantized"].dequantize(),
                ),
            )

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_weights")
    weight_dict: dict[str, dict] = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, "weight")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key]
            continue

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        match_key = _find_match(float_dict, key, "_packed_params")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key][0]

        # For LSTM
        split_str = key.split(".")
        if split_str[-1] == "param" and split_str[-3] == "_all_weight_values":
            layer = split_str[-2]
            module_name = ".".join(split_str[:-3])
            float_weight_ih_key = module_name + ".weight_ih_l" + layer
            float_weight_hh_key = module_name + ".weight_hh_l" + layer
            if float_weight_ih_key in float_dict and float_weight_hh_key in float_dict:
                weight_dict[key] = {}
                weight_dict[key]["float"] = float_dict[float_weight_ih_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][0].__getstate__()[0][0]
                )
                weight_dict[key]["float"] = float_dict[float_weight_hh_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][1].__getstate__()[0][0]
                )

    return weight_dict


def _get_logger_dict_helper(
    mod: nn.Module,
    target_dict: dict[str, Any],
    prefix: str = "",
) -> None:
    r"""This is the helper function for get_logger_dict

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module
        target_dict: the dictionary used to save all logger stats
    """

    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + "."

    for child in mod.children():
        if isinstance(child, Logger):
            target_dict[get_prefix(prefix) + "stats"] = child.stats
            break

    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_logger_dict_helper(child, target_dict, module_prefix)


def get_logger_dict(mod: nn.Module, prefix: str = "") -> dict[str, dict]:
    r"""Traverse the modules and save all logger stats into target dict.
    This is mainly used for quantization accuracy debug.

    Type of loggers supported:
        ShadowLogger: used to log the outputs of the quantized module and its matching float shadow module,
        OutputLogger: used to log the outputs of the modules

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module

    Return:
        target_dict: the dictionary used to save all logger stats

    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.get_logger_dict")

    target_dict: dict[str, dict] = {}
    _get_logger_dict_helper(mod, target_dict, prefix)
    return target_dict


class Logger(nn.Module):
    r"""Base class for stats logging"""

    def __init__(self):
        super().__init__()
        self.stats = {}
        # We only insert observer if the op is quantized with static quantization,
        # which is identified by activation_observer.dtype == quint8.  This is needed
        # when attaching Logger as observer for FX mode
        self.dtype = torch.quint8

    def forward(self, x):
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on


class ShadowLogger(Logger):
    r"""Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """

    def __init__(self):
        super().__init__()
        self.stats["float"] = []
        self.stats["quantized"] = []

    def forward(self, x, y):  # type: ignore[override]
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        self.stats["quantized"].append(x.detach())
        self.stats["float"].append(y.detach())


class OutputLogger(Logger):
    r"""Class used to log the outputs of the module"""

    def __init__(self):
        super().__init__()
        self.stats["tensor_val"] = []

    def forward(self, x):
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        self.stats["tensor_val"].append(x)
        return x


def _convert_tuple_to_list(t: Any) -> Any:
    return [_convert_tuple_to_list(x) for x in t] if type(t) is tuple else t


def _dequantize_tensor_list(t: Any) -> Any:
    return (
        [_dequantize_tensor_list(x) for x in t]
        if type(t) is list
        else t.dequantize()
        if t.is_quantized
        else t
    )


class Shadow(nn.Module):
    r"""Shadow module attaches the float module to its matching quantized module
    as the shadow. Then it uses Logger module to process the outputs of both
    modules.

    Args:
        q_module: module quantized from float_module that we want to shadow
        float_module: float module used to shadow q_module
        logger_cls: type of logger used to process the outputs of q_module and
            float_module. ShadowLogger or custom loggers can be used.
    """

    def __init__(self, q_module, float_module, logger_cls):
        super().__init__()
        self.orig_module = q_module
        self.shadow_module = float_module
        self.dequant = nnq.DeQuantize()
        self.logger = logger_cls()

    def forward(self, *x) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        xl = _convert_tuple_to_list(x)
        output = self.orig_module(*xl)
        xl_float = _dequantize_tensor_list(xl)
        shadow_output = self.shadow_module(*xl_float)
        self.logger(output, shadow_output)
        return output

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add(x, y)
        self.logger(output, shadow_output)
        return output

    def add_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.add_scalar(x, y)
        self.logger(output, shadow_output)
        return output

    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.mul(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.mul(x, y)
        self.logger(output, shadow_output)
        return output

    def mul_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.mul_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.mul_scalar(x, y)
        self.logger(output, shadow_output)
        return output

    def cat(self, x: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.cat(x, dim)
        x = [y.dequantize() for y in x]
        shadow_output = self.shadow_module.cat(x, dim)
        self.logger(output, shadow_output)
        return output

    def add_relu(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        output = self.orig_module.add_relu(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add_relu(x, y)
        self.logger(output, shadow_output)
        return output


def prepare_model_with_stubs(
    float_module: nn.Module,
    q_module: nn.Module,
    module_swap_list: set[type],
    logger_cls: Callable,
) -> None:
    r"""Prepare the model by attaching the float module to its matching quantized
    module as the shadow if the float module type is in module_swap_list.

    Example usage::

        prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)
        q_model(data)
        ob_dict = get_logger_dict(q_model)

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        module_swap_list: list of float module types to attach the shadow
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.prepare_model_with_stubs"
    )

    float_module_children = dict(float_module.named_children())

    reassign = {}
    for name, mod in q_module.named_children():
        if name not in float_module_children:
            continue

        float_mod = float_module_children[name]

        if type(float_mod) not in module_swap_list:
            prepare_model_with_stubs(float_mod, mod, module_swap_list, logger_cls)

        # Insert shadow module only if the module is not of the same type as
        # the floating point module
        if type(float_mod) in module_swap_list and not _is_identical_module_type(
            mod, float_mod
        ):
            reassign[name] = Shadow(mod, float_mod, logger_cls)

    for key, value in reassign.items():
        q_module._modules[key] = value


def _is_identical_module_type(mod1, mod2):
    # Compare if two modules have the same dtype
    mod1_module_types = [type(mod) for mod in mod1.modules()]
    mod2_module_types = [type(mod) for mod in mod2.modules()]
    return mod1_module_types == mod2_module_types


def compare_model_stub(
    float_model: nn.Module,
    q_model: nn.Module,
    module_swap_list: set[type],
    *data,
    logger_cls=ShadowLogger,
) -> dict[str, dict]:
    r"""Compare quantized module in a model with its floating point counterpart,
    feeding both of them the same input. Return a dict with key corresponding to
    module names and each entry being a dictionary with two keys 'float' and
    'quantized', containing the output tensors of quantized and its matching
    float shadow module. This dict can be used to compare and compute the module
    level quantization error.

    This function first call prepare_model_with_stubs() to swap the quantized
    module that we want to compare with the Shadow module, which takes quantized
    module, corresponding float module and logger as input, and creates a forward
    path inside to make the float module to shadow quantized module sharing the
    same input. The logger can be customizable, default logger is ShadowLogger
    and it will save the outputs of the quantized module and float module that
    can be used to compute the module level quantization error.

    Example usage::

        module_swap_list = [
            torchvision.models.quantization.resnet.QuantizableBasicBlock
        ]
        ob_dict = compare_model_stub(float_model, qmodel, module_swap_list, data)
        for key in ob_dict:
            print(
                key,
                compute_error(
                    ob_dict[key]["float"], ob_dict[key]["quantized"].dequantize()
                ),
            )

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        module_swap_list: list of float module types at which shadow modules will
            be attached.
        data: input data used to run the prepared q_model
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_model_stub")
    prepare_model_with_stubs(float_model, q_model, module_swap_list, logger_cls)
    q_model(*data)
    ob_dict = get_logger_dict(q_model)
    return ob_dict


def get_matching_activations(
    float_module: nn.Module,
    q_module: nn.Module,
) -> dict[str, dict[str, torch.Tensor]]:
    r"""Find the matching activation between float and quantized modules.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module

    Return:
        act_dict: dict with key corresponding to quantized module names and each
        entry being a dictionary with two keys 'float' and 'quantized', containing
        the matching float and quantized activations
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.get_matching_activations"
    )
    float_dict = get_logger_dict(float_module)
    quantized_dict = get_logger_dict(q_module)
    act_dict: dict[str, dict] = {}
    for key in quantized_dict:
        if len(quantized_dict[key]["tensor_val"]) == 0:
            continue
        match_key = _find_match(sorted(float_dict, reverse=True), key, "stats")
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]["float"] = float_dict[match_key]["tensor_val"]
            act_dict[key]["quantized"] = quantized_dict[key]["tensor_val"]
    return act_dict


def prepare_model_outputs(
    float_module: nn.Module,
    q_module: nn.Module,
    logger_cls=OutputLogger,
    allow_list=None,
) -> None:
    r"""Prepare the model by attaching the logger to both float module
    and quantized module if they are in the allow_list.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.prepare_model_outputs"
    )
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()

    qconfig_debug = torch.ao.quantization.QConfig(activation=logger_cls, weight=None)
    float_module.qconfig = qconfig_debug  # type: ignore[assignment]
    prepare(
        float_module, inplace=True, allow_list=allow_list, prepare_custom_config_dict={}
    )
    q_module.qconfig = qconfig_debug  # type: ignore[assignment]
    prepare(
        q_module,
        inplace=True,
        allow_list=allow_list,
        observer_non_leaf_module_list=NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST,
        prepare_custom_config_dict={},
    )


def compare_model_outputs(
    float_model: nn.Module,
    q_model: nn.Module,
    *data,
    logger_cls=OutputLogger,
    allow_list=None,
) -> dict[str, dict[str, torch.Tensor]]:
    r"""Compare output activations between float and quantized models at
    corresponding locations for the same input. Return a dict with key corresponding
    to quantized module names and each entry being a dictionary with two keys
    'float' and 'quantized', containing the activations of quantized model and
    float model at matching locations. This dict can be used to compare and
    compute the propagation quantization error.

    Example usage::

        act_compare_dict = compare_model_outputs(float_model, qmodel, data)
        for key in act_compare_dict:
            print(
                key,
                compute_error(
                    act_compare_dict[key]["float"],
                    act_compare_dict[key]["quantized"].dequantize(),
                ),
            )

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        data: input data used to run the prepared float_model and q_model
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger

    Return:
        act_compare_dict: dict with key corresponding to quantized module names
        and each entry being a dictionary with two keys 'float' and 'quantized',
        containing the matching float and quantized activations
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite.compare_model_outputs"
    )
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()
    prepare_model_outputs(float_model, q_model, logger_cls, allow_list)
    float_model(*data)
    q_model(*data)
    act_compare_dict = get_matching_activations(float_model, q_model)
    return act_compare_dict
<|endoftext|># mypy: allow-untyped-defs
"""
This module contains tooling to compare weights and activations
across models. Example usage::

    import copy
    import torch
    import torch.ao.quantization.quantize_fx as quantize_fx
    import torch.ao.ns._numeric_suite_fx as ns

    m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1)).eval()
    mp = quantize_fx.prepare_fx(m, {"": torch.ao.quantization.default_qconfig})
    # We convert a copy because we need the original prepared model
    # to be available for comparisons, and `quantize_fx.convert_fx` is inplace.
    mq = quantize_fx.convert_fx(copy.deepcopy(mp))

    #
    # Comparing weights
    #

    # extract weight pairs
    weight_comparison = ns.extract_weights("a", mp, "b", mq)

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        weight_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # weight_comparison contains the weights from `mp` and `mq` stored
    # in pairs, and can be used for further analysis.


    #
    # Comparing activations, with error propagation
    #

    # add loggers
    mp_ns, mq_ns = ns.add_loggers(
        "a", copy.deepcopy(mp), "b", copy.deepcopy(mq), ns.OutputLogger
    )

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_ns(datum)
    mq_ns(datum)

    # extract intermediate activations
    act_comparison = ns.extract_logger_info(mp_ns, mq_ns, ns.OutputLogger, "b")

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        act_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

    #
    # Comparing activations, without error propagation
    #

    # create shadow model
    mp_shadows_mq = ns.add_shadow_loggers(
        "a", copy.deepcopy(mp), "b", copy.deepcopy(mq), ns.OutputLogger
    )

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_shadows_mq(datum)

    # extract intermediate activations
    shadow_act_comparison = ns.extract_shadow_logger_info(
        mp_shadows_mq, ns.OutputLogger, "b"
    )

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        shadow_act_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # shadow_act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

"""

import collections
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn as nn
from torch.ao.ns.fx.graph_matcher import get_matching_subgraph_pairs
from torch.ao.ns.fx.mappings import get_base_name_to_sets_of_related_ops
from torch.ao.ns.fx.n_shadows_utils import (
    _get_dedup_subgraphs,
    create_add_loggers_graph,
    create_n_transformed_and_logged_copies_of_subgraph,
    create_results_comparison,
    extract_weight_comparison,
    group_results_by_subgraph,
    OutputProp,
    print_n_shadows_summary,
    SHADOW_WRAPPER_NODE_NAME_PREFIX,
)
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.backend_config.utils import (
    get_fusion_pattern_to_root_node_getter,
)
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from torch.ao.quantization.fx.match_utils import _find_matches
from torch.ao.quantization.fx.qconfig_mapping_utils import (
    _generate_node_name_to_qconfig,
)
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.fx import GraphModule
from torch.fx.graph import Node

from .fx.graph_passes import add_loggers_to_model, create_a_shadows_b
from .fx.ns_types import NSNodeTargetType, NSResultsType, NSSingleResultValuesType
from .fx.utils import (
    get_target_type_str,
    maybe_add_missing_fqns,
    rekey_logger_info_on_node_name_of_model,
)
from .fx.weight_utils import extract_weight_from_node


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfigAny

RNNReturnType = tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


class OutputLogger(nn.Module):
    """
    Base class for capturing intermediate values.
    """

    stats: list[torch.Tensor]
    stats_rnn: list[RNNReturnType]

    # Mark as impure so that calls to it will not be removed during DCE.
    _is_impure = True

    def __init__(
        self,
        ref_node_name: str,
        prev_node_name: str,
        model_name: str,
        ref_name: str,
        prev_node_target_type: str,
        ref_node_target_type: str,
        results_type: str,
        index_within_arg: int,
        index_of_arg: int,
        fqn: str | None,
        qconfig_str: str | None = "",
    ):
        super().__init__()
        self.stats: list[torch.Tensor] = []
        self.stats_rnn: list[RNNReturnType] = []

        # name of the node which was responsible for adding this logger
        # Note:
        # - if we are logging node outputs, this is the same as prev_node_name
        # - if we are logging node inputs, this is the name of the node
        #   whose input this logger is logging.
        #
        # example, where logger1 is logging input of op1 and logger2 is logging
        #    the output of op1:
        #
        #  x1 -> logger1 -> op1 -> logger2 -> x2
        #
        # in this example,
        #   - logger1's prev_node_name is x1 and ref_node_name is op1
        #   - logger2's prev_node_name is op1 and ref_node_name is op1
        self.ref_node_name = ref_node_name
        # name of the node whose output this Logger is capturing
        self.prev_node_name = prev_node_name

        # name of the model from which the node originated from
        self.model_name = model_name
        # reference name, used to match loggers from separate models
        # to each other
        self.ref_name = ref_name
        # type of the target of the node whose output this logger is logging
        self.prev_node_target_type = prev_node_target_type
        # type of the target of the node which was responsible for adding this
        # logger
        self.ref_node_target_type = ref_node_target_type
        # what kind of values are inside of stats
        self.results_type = results_type
        # index of this node within the arg of the input/output node
        # for example, in cat([x1, x2, x3], dim=0), x2 would have index_within_arg == 1
        self.index_within_arg = index_within_arg
        # index of this node within the args of the input/output node
        # for example, in add(x1, x2), x2 would have index_of_arg == 1
        self.index_of_arg = index_of_arg
        # fully qualified name
        self.fqn = fqn
        # if loggers are added before prepare_fx, but we do not want
        # collect results of calibration, only results after convert_fx
        # so, we add a flag to control whether this logger collects data
        self.enabled = True
        # string representation of qconfig
        self.qconfig_str = qconfig_str
        # this can be turned off to reduce memory usage during calibration
        self.save_activations = True

    # Note: cannot annotate the type of x because TorchScript does not support
    #   the Union type.
    def forward(self, x):
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        # TODO(future PR): consider designing this better, as the difference
        # between these two flags is subtle and not obvious.
        if not self.enabled:
            return x
        if not self.save_activations:
            return x
        # TODO(future PR): consider refactoring this to better reuse the parent
        # class
        if isinstance(x, torch.Tensor):
            self.stats.append(x.detach())
        elif isinstance(x, tuple) and len(x) == 2 and len(x[1]) == 2:
            new_res = (x[0].detach(), (x[1][0].detach(), x[1][1].detach()))
            self.stats_rnn.append(new_res)
        return x

    def __repr__(self):
        clean_dict = {
            k: v
            for k, v in self.__dict__.items()
            # skip nn.Module keys
            if (k != "training") and not k.startswith("_")
        }
        return f"OutputLogger({clean_dict})"


class OutputComparisonLogger(OutputLogger):
    """
    Same as OutputLogger, but also requires the original activation
    in order to calculate the comparison at calibration time
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO(future PR): make the comparison function configurable
        self.comparison_fn = torch.ao.ns.fx.utils.compute_sqnr
        self.comparison_fn_name = "sqnr"
        # precalculated comparisons of logger output versus reference
        self.comparisons = []
        # precalculated comparisons function

    def forward(self, x, x_ref):  # type: ignore[override]
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        if not self.enabled:
            return x
        if not isinstance(x, torch.Tensor):
            raise AssertionError("non-tensor inputs not yet supported")
        if self.save_activations:
            # save the activation, for debugging
            self.stats.append(x.detach())
        # save the comparison
        self.comparisons.append(self.comparison_fn(x, x_ref))
        return x

    def __repr__(self):
        clean_dict = {
            k: v
            for k, v in self.__dict__.items()
            # skip nn.Module keys
            if (k != "training") and not k.startswith("_")
        }
        return f"OutputComparisonLogger({clean_dict})"


class NSTracer(quantize_fx.QuantizationTracer):
    """
    Just like a regular FX quantization tracer, but treats observers and fake_quantize
    modules as leaf modules.
    """

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # fmt: off
        """
        """  # blank docblock to make autodoc happy
        # fmt: on
        if isinstance(m, torch.ao.quantization.ObserverBase):
            return True
        elif isinstance(m, torch.ao.quantization.FakeQuantizeBase):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def _extract_weights_one_model(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument: list[tuple[Node, str]],
    results: NSResultsType,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = None,
) -> None:
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._extract_weights_one_model"
    )
    for node, ref_name in nodes_and_names_to_instrument:
        res_type = NSSingleResultValuesType.WEIGHT.value
        extracted_weight = extract_weight_from_node(
            node, model, op_to_type_to_weight_extraction_fn
        )
        if extracted_weight:
            if ref_name not in results:
                results[ref_name] = {res_type: {}}
            results[ref_name][res_type][model_name] = [extracted_weight]


def _extract_weights_impl(
    model_name_a: str,
    gm_a: GraphModule,
    model_name_b: str,
    gm_b: GraphModule,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = None,
) -> NSResultsType:
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._extract_weights_impl"
    )
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map
    )

    # split the subgraph pairs into one data structure for each model
    nodes_and_names_to_instrument_a: list[tuple[Node, str]] = []
    nodes_and_names_to_instrument_b: list[tuple[Node, str]] = []
    for match_name, match in matched_subgraph_pairs.items():
        subgraph_a, subgraph_b = match
        nodes_and_names_to_instrument_a.append((subgraph_a.base_op_node, match_name))
        nodes_and_names_to_instrument_b.append((subgraph_b.base_op_node, match_name))

    # populate the results, one model at a time
    results: NSResultsType = {}
    _extract_weights_one_model(
        model_name_a,
        gm_a,
        nodes_and_names_to_instrument_a,
        results,
        op_to_type_to_weight_extraction_fn,
    )
    _extract_weights_one_model(
        model_name_b,
        gm_b,
        nodes_and_names_to_instrument_b,
        results,
        op_to_type_to_weight_extraction_fn,
    )

    # fill in missing fqn entries
    maybe_add_missing_fqns(results)

    # rekey on names of nodes in gm_b
    results = rekey_logger_info_on_node_name_of_model(results, model_name_b)

    return results


def extract_weights(
    model_name_a: str,
    model_a: nn.Module,
    model_name_b: str,
    model_b: nn.Module,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = None,
) -> NSResultsType:
    """
    Extract weights from model A and model B, and return a comparison.

    Args:
        model_name_a: string name of model A to use in results
        model_a: model A
        model_name_b: string name of model B to use in results
        model_b: model B
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change
        op_to_type_to_weight_extraction_fn: optional override of function which extracts weight
            from a type, subject to change

    Return:
        NSResultsType, containing the weight comparisons
    """

    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx.extract_weights")
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()

    # TODO(future PR): expose these
    skipped_module_names: list[str] = []
    skipped_module_classes: list[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(
        model_a, "node_name_to_scope"
    )
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(
        model_b, "node_name_to_scope"
    )
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _extract_weights_impl(
        model_name_a,
        gm_a,
        model_name_b,
        gm_b,
        base_name_to_sets_of_related_ops,
        unmatchable_types_map,
        op_to_type_to_weight_extraction_fn,
    )


def _add_loggers_one_model(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument_inputs: list[tuple[Node, str, str]],
    nodes_and_names_to_instrument_outputs: list[tuple[Node, str, str]],
    logger_cls: Callable,
) -> nn.Module:
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._add_loggers_one_model"
    )

    # TODO(future PR): do not observe nodes we do not care
    #   about (both fp32, denylist, etc)
    node_to_instrument_inputs_to_ref_name: dict[Node, tuple[str, str]] = {}
    node_to_instrument_outputs_to_ref_name: dict[Node, tuple[str, str]] = {}
    for node, ref_name, ref_node_type in nodes_and_names_to_instrument_inputs:
        node_to_instrument_inputs_to_ref_name[node] = (ref_name, ref_node_type)
    for node, ref_name, ref_node_type in nodes_and_names_to_instrument_outputs:
        node_to_instrument_outputs_to_ref_name[node] = (ref_name, ref_node_type)

    model = add_loggers_to_model(
        model,
        node_to_instrument_inputs_to_ref_name,
        node_to_instrument_outputs_to_ref_name,
        logger_cls,
        model_name,
    )
    return model


def _add_loggers_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> tuple[nn.Module, nn.Module]:
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._add_loggers_impl")
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map
    )
    nodes_and_names_to_instrument_inputs_a = []
    nodes_and_names_to_instrument_inputs_b = []
    nodes_and_names_to_instrument_outputs_a = []
    nodes_and_names_to_instrument_outputs_b = []
    for match_name, (subgraph_a, subgraph_b) in matched_subgraph_pairs.items():
        ref_node_type_a = get_target_type_str(subgraph_a.base_op_node, gm_a)
        ref_node_type_b = get_target_type_str(subgraph_b.base_op_node, gm_b)
        # Note: for matching inputs we use start_node, such as observing
        # the input of linear in linear-relu
        if should_log_inputs:
            nodes_and_names_to_instrument_inputs_a.append(
                (subgraph_a.start_node, match_name, ref_node_type_a)
            )
            nodes_and_names_to_instrument_inputs_b.append(
                (subgraph_b.start_node, match_name, ref_node_type_b)
            )
        # Note: for matching activations we always use end_node,
        # such as observing the output of relu in linear-relu
        nodes_and_names_to_instrument_outputs_a.append(
            (subgraph_a.end_node, match_name, ref_node_type_a)
        )
        nodes_and_names_to_instrument_outputs_b.append(
            (subgraph_b.end_node, match_name, ref_node_type_b)
        )

    new_model_a = _add_loggers_one_model(
        name_a,
        gm_a,
        nodes_and_names_to_instrument_inputs_a,
        nodes_and_names_to_instrument_outputs_a,
        logger_cls,
    )
    new_model_b = _add_loggers_one_model(
        name_b,
        gm_b,
        nodes_and_names_to_instrument_inputs_b,
        nodes_and_names_to_instrument_outputs_b,
        logger_cls,
    )
    return (new_model_a, new_model_b)


def add_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = False,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> tuple[nn.Module, nn.Module]:
    """
    Instrument model A and model B with loggers.

    Args:
        name_a: string name of model A to use in results
        model_a: model A
        name_b: string name of model B to use in results
        model_b: model B
        logger_cls: class of Logger to use
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change

    Return:
        Returns a tuple of (model_a_with_loggers, model_b_with_loggers).  Modifies both models inplace.
    """

    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx.add_loggers")
    # TODO(future PR): expose these
    skipped_module_names: list[str] = []
    skipped_module_classes: list[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(
        model_a, "node_name_to_scope"
    )
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(
        model_b, "node_name_to_scope"
    )
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _add_loggers_impl(
        name_a,
        gm_a,
        name_b,
        gm_b,
        logger_cls,
        should_log_inputs=should_log_inputs,
        base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
        unmatchable_types_map=unmatchable_types_map,
    )


def _extract_logger_info_one_model(
    model: nn.Module,
    results: NSResultsType,
    logger_cls: Callable,
) -> None:
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._extract_logger_info_one_model"
    )
    for _gm_name, mod in model.named_modules():
        # TODO(future PR): better check when scripted
        is_logger = isinstance(mod, logger_cls) or (  # type: ignore[arg-type]
            isinstance(mod, torch.jit.RecursiveScriptModule)
            and mod.original_name == "OutputLogger"
        )
        if is_logger:
            key = mod.ref_name
            if key not in results:
                results[key] = {}
            if mod.model_name in results[key]:
                raise AssertionError(f"{mod.model_name} is already present in results")
            if mod.results_type not in results[key]:
                results[key][mod.results_type] = {}
            if mod.model_name not in results[key][mod.results_type]:
                results[key][mod.results_type][mod.model_name] = []
            stats_to_use = mod.stats
            if len(mod.stats_rnn) > 0:
                stats_to_use = mod.stats_rnn
            data = {
                "type": mod.results_type,
                "values": stats_to_use,
                "ref_node_name": mod.ref_node_name,
                "ref_node_target_type": mod.ref_node_target_type,
                "prev_node_name": mod.prev_node_name,
                "prev_node_target_type": mod.prev_node_target_type,
                "index_within_arg": mod.index_within_arg,
                "index_of_arg": mod.index_of_arg,
                "fqn": mod.fqn,
                "qconfig_str": mod.qconfig_str,
            }
            if hasattr(mod, "comparisons"):
                data["comparisons"] = mod.comparisons
                data["comparison_fn_name"] = mod.comparison_fn_name
            else:
                data["comparisons"] = []
                data["comparison_fn_name"] = ""
            results[key][mod.results_type][mod.model_name].append(data)
            # ensure the list stays sorted
            results[key][mod.results_type][mod.model_name].sort(
                key=lambda res: f"{res['index_of_arg']}:{res['index_within_arg']}"
            )


# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_outputs`
def extract_logger_info(
    model_a: nn.Module,
    model_b: nn.Module,
    logger_cls: Callable,
    model_name_to_use_for_layer_names: str,
) -> NSResultsType:
    """
    Traverse all loggers in `model_a` and `model_b`, and extract the logged
    information.

    Args:
        model_a: model A
        model_b: model B
        logger_cls: class of Logger to use
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output

    Return:
        NSResultsType, containing the logged comparisons
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx.extract_logger_info"
    )
    results: NSResultsType = {}
    for model in (model_a, model_b):
        _extract_logger_info_one_model(model, results, logger_cls)
    # fill in missing fqn entries
    maybe_add_missing_fqns(results)
    # rekey on the name of model b
    results = rekey_logger_info_on_node_name_of_model(
        results, model_name_to_use_for_layer_names
    )
    return results


def _add_shadow_loggers_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> nn.Module:
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx._add_shadow_loggers_impl"
    )
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map
    )
    gm_a_shadows_b = create_a_shadows_b(
        name_a,
        gm_a,
        name_b,
        gm_b,
        matched_subgraph_pairs,
        logger_cls,
        should_log_inputs=should_log_inputs,
        node_type_to_io_type_map=node_type_to_io_type_map,
    )
    return gm_a_shadows_b


def add_shadow_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = False,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> nn.Module:
    """
    Instrument model A and model B with shadow loggers.

    Args:
        name_a: string name of model A to use in results
        model_a: model A
        name_b: string name of model B to use in results
        model_b: model B
        logger_cls: class of Logger to use
        should_log_inputs: whether to log inputs
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx.add_shadow_loggers"
    )
    # TODO(future PR): expose these
    skipped_module_names: list[str] = []
    skipped_module_classes: list[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(
        model_a, "node_name_to_scope"
    )
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(
        model_b, "node_name_to_scope"
    )
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _add_shadow_loggers_impl(
        name_a,
        gm_a,
        name_b,
        gm_b,
        logger_cls,
        should_log_inputs=should_log_inputs,
        base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
        node_type_to_io_type_map=node_type_to_io_type_map,
        unmatchable_types_map=unmatchable_types_map,
    )


def extract_shadow_logger_info(
    model_a_shadows_b: nn.Module,
    logger_cls: Callable,
    model_name_to_use_for_layer_names: str,
) -> NSResultsType:
    """
    Traverse all loggers in a shadow model, and extract the logged
    information.

    Args:
        model_a_shadows_b: shadow model
        logger_cls: class of Logger to use
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output

    Return:
        NSResultsType, containing the logged comparisons
    """
    torch._C._log_api_usage_once(
        "quantization_api._numeric_suite_fx.extract_shadow_logger_info"
    )
    results: NSResultsType = collections.defaultdict(dict)
    _extract_logger_info_one_model(model_a_shadows_b, results, logger_cls)
    # fill in missing fqn entries
    maybe_add_missing_fqns(results)
    # rekey on the name of model b
    results = rekey_logger_info_on_node_name_of_model(
        results, model_name_to_use_for_layer_names
    )
    return dict(results)


def extend_logger_results_with_comparison(
    results: NSResultsType,
    model_name_1: str,
    model_name_2: str,
    comparison_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    comparison_name: str,
) -> None:
    """
    Compares the logged values from `model_name_2` against the corresponding
    values in `model_name_1`, using `comparison_fn`. Records the result
    in `model_name_2`'s results under `comparison_name`. Modifies `results` inplace.

    Args:
        results: the result data structure from `extract_logger_info` or
          `extract_shadow_logger_info`.
        model_name_1: string name of model 1
        model_name_2: string name of model 2
        comparison_fn: function to compare two Tensors
        comparison_name: string name of model to use for
          layer names in the output
    """
    for results_type_to_results in results.values():
        for model_name_to_results in results_type_to_results.values():
            if model_name_1 not in model_name_to_results:
                raise AssertionError(f"{model_name_1} not found in results")
            if model_name_2 not in model_name_to_results:
                raise AssertionError(f"{model_name_2} not found in results")

            results_1 = model_name_to_results[model_name_1]
            results_2 = model_name_to_results[model_name_2]

            for result_2 in results_2:
                index_within_arg_2 = result_2["index_within_arg"]
                index_of_arg_2 = result_2["index_of_arg"]
                # find corresponding result_1
                result_1 = None
                for cur_result_1 in results_1:
                    index_within_arg_1 = cur_result_1["index_within_arg"]
                    index_of_arg_1 = cur_result_1["index_of_arg"]
                    if (index_within_arg_1 == index_within_arg_2) and (
                        index_of_arg_1 == index_of_arg_2
                    ):
                        result_1 = cur_result_1
                        break
                if result_1 is None:
                    raise AssertionError("Expected result_1 to be not None")

                values_1 = result_1["values"]
                values_2 = result_2["values"]
                result_2[comparison_name] = []
                for value_1, value_2 in zip(values_1, values_2):
                    comparison_result = comparison_fn(value_1, value_2)
                    result_2[comparison_name].append(comparison_result)


def prepare_n_shadows_model(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_multi_mapping: QConfigMultiMapping,
    backend_config: BackendConfig,
    custom_prepare_fn: Callable | None = None,
    custom_prepare_kwargs: dict[str, Any] | None = None,
    custom_tracer: Any = None,
) -> GraphModule:
    """
    Given a model with a graph with M ops such as


      args_kwargs_m -> op_m -> output_m


    And a set of N qconfigs for each op, creates a new model, with
    each of the subgraph of `op_m` transformed into

    .. code::

           |---------> op_m_n -> log_m_n
           |                     /
      args_kwargs_m ---------> op_m -> log_m_0

    Where op_m_n is op_m wrapped in a submodule and transformed with
    qconfig_n, and its inner graph looks like

    .. code::

      args_m -------- op_m_prepared_with_qconfig_n -> out_m_n
                  /
      kwargs_m ---

    This is useful for testing different quantization of multiple layers in
    a single pass through the model.

    High level TODOs for future PRs:
    * figure out a better way to name the output structure
    * return a results data structure instead of printing it out
    * add examples to docblocks
    """

    if custom_tracer is None:
        tracer = quantize_fx.QuantizationTracer([], [])
    else:
        tracer = custom_tracer
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    # this is necessary to ensure logger FQNs get populated
    mt._node_name_to_scope = tracer.node_name_to_scope  # type: ignore[assignment]

    # run example input propagation, we need this to call prepare_fx on
    # individual subgraphs
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)

    # Find the set of subgraphs in the original graph which we need to
    # consider.
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: list[str] = []
    standalone_module_classes: list[type] = []
    custom_module_classes: list[type] = []
    matches = _find_matches(
        mt.graph,
        modules,
        patterns,
        root_node_getter_mapping,
        standalone_module_names,
        standalone_module_classes,
        custom_module_classes,
    )
    subgraphs_dedup: dict[str, list[Node]] = _get_dedup_subgraphs(matches)

    # generate node to qconfig for each subgraph
    # TODO(future PR): deduplicate repeating entries
    list_of_node_name_to_qconfig: list[dict[str, QConfigAny]] = []
    for qconfig_mapping in qconfig_multi_mapping.qconfig_mappings_list:
        node_name_to_qconfig = _generate_node_name_to_qconfig(
            mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope
        )
        list_of_node_name_to_qconfig.append(node_name_to_qconfig)

    # For each region in the model, do the following:
    #   For each qconfig for that region, do the following:
    #     1. create a copy of the region wrapped in a module
    #     2. pass original args, original kwargs, and expected output to module
    #     3. add an output comparison logger and hook it up to compare
    #        actual output to expected output
    #     4. run `prepare_fx` on the module
    for subgraph_idx, (match_name, nodes_in_this_subgraph) in enumerate(
        subgraphs_dedup.items()
    ):
        create_n_transformed_and_logged_copies_of_subgraph(
            mt,
            subgraph_idx,
            match_name,
            nodes_in_this_subgraph,
            qconfig_multi_mapping.qconfig_mappings_list,
            list_of_node_name_to_qconfig,
            custom_prepare_fn,
            custom_prepare_kwargs,  # type: ignore[arg-type]
        )

    return mt


# TODO(future PR): we should rethink the names of all the PNP APIs
def _prepare_n_shadows_add_loggers_model(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_mapping: QConfigMapping,
    backend_config: BackendConfig,
) -> torch.nn.Module:
    r"""
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.

    This creates a model which provides logging for the following
    problem: if we quantize `model` with `qconfig_mapping` and feed
    the same input through both models, log the comparisons of
    corresponding intermediate layers.

    The problem is solved with a single model.  Specifically, we
    partition `model` into N subgraphs, create a copy of each relevant
    subgraph, wrap it in a module, apply the quantization API to that
    module, and hook up loggers to measure the comparisons.

    Example starting graph:

      x0 -> op0 -> x1 -> op1 -> x2

    Example config: quantize op0 to int8, do nothing to op1.
    The following graph will be created:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \                        \                           \       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_0 -> x2_1 ----> clog

    Where op0_0 is op0, op0_1 is op0 wrapped in a submodule and quantized
    to int8, op1_0 is op1 (appearing in the graph twice), log is a logger,
    and clog is a comparison logger.
    """

    tracer = quantize_fx.QuantizationTracer([], [])
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    # this is necessary to ensure logger FQNs get populated
    mt._node_name_to_scope = tracer.node_name_to_scope  # type: ignore[assignment]

    # run example input propagation, we need this to call prepare_fx on
    # individual subgraphs
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)

    # Find the set of subgraphs in the original graph which we need to
    # consider.
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: list[str] = []
    standalone_module_classes: list[type] = []
    custom_module_classes: list[type] = []
    matches = _find_matches(
        mt.graph,
        modules,
        patterns,
        root_node_getter_mapping,
        standalone_module_names,
        standalone_module_classes,
        custom_module_classes,
    )
    subgraphs_dedup: dict[str, list[Node]] = _get_dedup_subgraphs(matches)

    # generate node to qconfig for each subgraph
    node_name_to_qconfig = _generate_node_name_to_qconfig(
        mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope
    )

    # Now, mutate the graph to be the add_loggers graph with propagation
    # error.
    create_add_loggers_graph(mt, subgraphs_dedup, qconfig_mapping, node_name_to_qconfig)

    return mt


# TODO(future PR): we should rethink the names of all the PNP APIs
def _n_shadows_compare_weights(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_mapping: QConfigMapping,
    backend_config: BackendConfig,
) -> NSResultsType:
    """
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.
    """
    qconfig_multi_mapping = QConfigMultiMapping.from_list_qconfig_mapping(
        [qconfig_mapping]
    )
    mp = prepare_n_shadows_model(
        model, example_inputs, qconfig_multi_mapping, backend_config
    )
    # passing inputs through the model is necessary to populate
    # observers which observe weights with real values
    mp(*example_inputs)
    mq = convert_n_shadows_model(mp)
    weight_comparison = extract_weight_comparison(mq)
    return weight_comparison


# TODO(future PR): consider aligning API signature with other similar quantization
# functions (enable_fake_quant, etc)
def loggers_set_enabled(model: torch.nn.Module, enabled: bool) -> None:
    """
    Sets the `enabled` setting on a `model`'s loggers
    """
    for _, child in model.named_modules():
        if isinstance(child, OutputLogger):
            child.enabled = enabled


# TODO(future PR): consider aligning API signature with other similar quantization
# functions (enable_fake_quant, etc)
def loggers_set_save_activations(
    model: torch.nn.Module,
    save_activations: bool,
) -> None:
    """
    Sets the `save_activations` setting on a `model`'s loggers
    """
    for _name, child in model.named_modules():
        if isinstance(child, OutputLogger):
            child.save_activations = save_activations


def convert_n_shadows_model(
    model: GraphModule,
    custom_convert_fn: Callable | None = None,
    custom_convert_kwargs: dict[str, Any] | None = None,
) -> GraphModule:
    """
    Given a model from `prepare_n_shadows_model`, runs `convert_fx`
    on each shadow submodule.
    """
    for node in model.graph.nodes:
        # TODO(future PR): consider matching in a safer way than
        # node name string match
        if node.name.startswith(SHADOW_WRAPPER_NODE_NAME_PREFIX):
            orig_mod = getattr(model, node.name)
            if custom_convert_fn is None:
                converted_mod = torch.ao.quantization.quantize_fx.convert_fx(orig_mod)
            else:
                if custom_convert_kwargs is None:
                    custom_convert_kwargs = {}
                converted_mod = custom_convert_fn(orig_mod, **custom_convert_kwargs)
            setattr(model, node.name, converted_mod)

    return model


def extract_results_n_shadows_model(model: torch.nn.Module) -> NSResultsType:
    """
    Extracts logger results from `model`.
    """
    results: NSResultsType = {}
    _extract_logger_info_one_model(model, results, OutputLogger)
    return results


def print_comparisons_n_shadows_model(results: NSResultsType) -> None:
    """
    Prints a summary of extracted `results`.
    """
    results_grouped = group_results_by_subgraph(results)
    results_comparison = create_results_comparison(results_grouped)
    print_n_shadows_summary(results_comparison)
<|endoftext|>import operator
from typing import TYPE_CHECKING

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.quantization.fx._lower_to_native_backend as _lower_to_native_backend
import torch.ao.quantization.quantization_mappings as quantization_mappings
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.backend_config import get_native_backend_config

from .ns_types import NSNodeTargetType


if TYPE_CHECKING:
    from collections.abc import Callable


toq = torch.ops.quantized


def get_base_name_to_sets_of_related_ops() -> dict[str, set[NSNodeTargetType]]:
    # note: this set is modified below by items from backend_config
    sets_of_related_ops: list[set[NSNodeTargetType]] = [
        # conv modules
        {
            nn.Conv1d,
        },
        {
            nn.Conv2d,
        },
        {
            nn.Conv3d,
        },
        # conv functionals
        {
            F.conv1d,
        },
        {
            F.conv2d,
        },
        {
            F.conv3d,
        },
        # linear modules
        {
            nn.Linear,
        },
        # linear functionals
        {
            F.linear,
        },
        # average pool
        {
            nn.AvgPool1d,
            torch.avg_pool1d,
        },
        {
            nn.AvgPool2d,
            torch._C._nn.avg_pool2d,
        },
        {
            nn.AvgPool3d,
            torch._C._nn.avg_pool3d,
        },
        # adaptive average pool
        {
            nn.AdaptiveAvgPool1d,
            F.adaptive_avg_pool1d,
        },
        {
            nn.AdaptiveAvgPool2d,
            F.adaptive_avg_pool2d,
        },
        {
            nn.AdaptiveAvgPool3d,
            F.adaptive_avg_pool3d,
        },
        # LSTM
        {
            nn.LSTM,
        },
        # add
        {
            torch.add,
            operator.add,  # x + y
        },
        # cat
        {
            torch.cat,
        },
        # mul
        {
            torch.mul,
            operator.mul,
        },
        # relu
        {
            F.relu,
            nn.ReLU,
            "relu",
            "relu_",
            torch.relu,
        },
        # maxpool
        {
            nn.MaxPool1d,
            F.max_pool1d,
        },
        {
            nn.MaxPool2d,
            F.max_pool2d,
        },
        {
            nn.MaxPool3d,
            F.max_pool3d,
        },
        # sigmoid
        {
            torch.sigmoid,
            "sigmoid",
            "sigmoid_",
            nn.Sigmoid,
            F.sigmoid,
        },
        # BatchNorm
        {
            nn.BatchNorm2d,
        },
        {
            nn.BatchNorm3d,
        },
        # ConvTranspose
        {
            nn.ConvTranspose1d,
        },
        {
            nn.ConvTranspose2d,
        },
        {
            nn.ConvTranspose3d,
        },
        # functional transposed conv
        {
            F.conv_transpose1d,
        },
        {
            F.conv_transpose2d,
        },
        {
            F.conv_transpose3d,
        },
        # ELU
        {
            nn.ELU,
        },
        # Embedding
        {
            nn.Embedding,
        },
        # EmbeddingBag
        {
            nn.EmbeddingBag,
        },
        # GroupNorm
        {
            nn.GroupNorm,
        },
        # Hardswish
        {
            nn.Hardswish,
        },
        # InstanceNorm
        {
            nn.InstanceNorm1d,
        },
        {
            nn.InstanceNorm2d,
        },
        {
            nn.InstanceNorm3d,
        },
        # LayerNorm
        {
            nn.LayerNorm,
        },
        # LeakyReLU
        {
            nn.LeakyReLU,
        },
        # ReLU6
        {
            nn.ReLU6,
            F.relu6,
        },
        # F.elu
        {
            F.elu,
        },
        # F.hardswish
        {
            F.hardswish,
        },
        # F.group_norm
        {
            F.group_norm,
        },
        # F.instance_norm
        {
            F.instance_norm,
        },
        # F.layer_norm
        {
            F.layer_norm,
        },
        # F.leaky_relu
        {
            F.leaky_relu,
        },
        # F.silu
        {
            nn.SiLU,
            F.silu,
        },
        # F.mish
        {
            nn.Mish,
            F.mish,
        },
        # F.tanh
        {
            nn.Tanh,
            F.tanh,
            torch.tanh,
            "tanh_",
            "tanh",
        },
        # F.hardsigmoid
        {
            "hardsigmoid_",
            "hardsigmoid",
            F.hardsigmoid,
            nn.Hardsigmoid,
        },
        # F.hardtanh
        {
            nn.Hardtanh,
            F.hardtanh,
            F.hardtanh_,
        },
        # floordiv
        {
            operator.floordiv,
        },
        # unsqueeze
        {
            torch.unsqueeze,
        },
        # stack
        {
            torch.stack,
        },
        # squeeze
        {
            torch.squeeze,
        },
        # sort
        {
            torch.sort,
        },
        # repeat_interleave
        {
            torch.repeat_interleave,
        },
        # min
        {
            torch.min,
        },
        # mean
        {
            torch.mean,
        },
        # max
        {
            torch.max,
        },
        # transpose
        {
            torch.transpose,
        },
        # flatten
        {
            torch.flatten,
        },
        # clamp
        {
            torch.clamp,
        },
        # chunk
        {
            torch.chunk,
        },
        # interpolate
        {
            torch.nn.functional.interpolate,
        },
        # dropout
        {
            nn.Dropout,
        },
        # F.dropout
        {
            F.dropout,
        },
        # matmul
        {
            torch.matmul,
        },
        # Softmax
        {
            nn.Softmax,
        },
        # PReLU
        {
            nn.PReLU,
            nnq.PReLU,
        },
        # F.prelu
        {
            F.prelu,
            toq.prelu,
        },
        # pixel shuffle
        {
            nn.PixelShuffle,
        },
        {
            F.pixel_shuffle,
        },
        # pixel unshuffle
        {
            nn.PixelUnshuffle,
        },
        {
            F.pixel_unshuffle,
        },
        # narrow
        {
            torch.narrow,
        },
    ]

    # for each floating point op, add versions of the op added by
    # backend_config
    backend_config = get_native_backend_config()

    new_connections: list[tuple[Callable, Callable]] = [
        # technical debt edge case
        (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear),
    ]

    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        # pattern format: (c, (b, a))
        first_element = pattern
        # look from the end, because pattern is in reverse order
        while isinstance(first_element, (list, tuple)):
            first_element = first_element[-1]

        if config.fused_module is not None:
            # case 1: pattern fuses a pattern of ops into an op
            # example: nn.Conv1d, nn.ReLU fused into nni.ConvReLU1d
            new_connections.append((first_element, config.fused_module))

        if config.qat_module is not None:
            # case 2: pattern swaps a module into a QAT module
            # example: nni.ConvReLU1d swapped into nniqat.ConvReLU1d
            new_connections.append((first_element, config.qat_module))

        if config.reference_quantized_module is not None:
            # case 3: reference version of floating point module, such as
            # nn.Conv2d and nnqr.Conv2d
            new_connections.append((first_element, config.reference_quantized_module))

    #
    # Add reference module swaps from default lowering path
    #

    for source_to_target in (
        _lower_to_native_backend.STATIC_LOWER_MODULE_MAP,
        _lower_to_native_backend.DYNAMIC_LOWER_MODULE_MAP,
        _lower_to_native_backend.WEIGHT_ONLY_LOWER_MODULE_MAP,
        _lower_to_native_backend.SPECIAL_PATTERN_LOWER_MODULE_MAP,
    ):
        for source, target in source_to_target.items():  # type: ignore[attr-defined]
            new_connections.append((source, target))

    for source_to_double_target in (
        _lower_to_native_backend.STATIC_LOWER_FUSED_MODULE_MAP,
        _lower_to_native_backend.STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP,
        _lower_to_native_backend.DYNAMIC_LOWER_FUSED_MODULE_MAP,
    ):
        for source, (target1, target2) in source_to_double_target.items():  # type: ignore[attr-defined]
            new_connections.append((source, target1))
            new_connections.append((source, target2))

    #
    # Add function swaps from default lowering path
    #

    for source, (  # type:ignore[assignment]
        target1,
        target2,
    ) in _lower_to_native_backend.STATIC_LOWER_FUNCTIONAL_MAP.items():
        new_connections.append((source, target1))
        # pyrefly: ignore [bad-argument-type]
        new_connections.append((source, target2))

    for source_to_target in (
        _lower_to_native_backend.QBIN_OP_MAPPING,
        _lower_to_native_backend.QBIN_RELU_OP_MAPPING,
        quantization_mappings.DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS,
    ):
        for source, target in source_to_target.items():  # type:ignore[assignment]
            # pyrefly: ignore [bad-argument-type]
            new_connections.append((source, target))

    #
    # Add other swaps, ideally in the future this could be removed
    # after the lowering code stops using these.
    #
    for source_to_target in (
        quantization_mappings.DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,
    ):
        for source, target in source_to_target.items():  # type:ignore[assignment]
            new_connections.append((source, target))

    # add the new connections from backend_config
    for item1, item2 in new_connections:
        for set_of_related_ops in sets_of_related_ops:
            if item1 in set_of_related_ops or item2 in set_of_related_ops:
                set_of_related_ops.add(item1)
                set_of_related_ops.add(item2)
                break

    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] = {}

    for counter, set_of_related_ops in enumerate(sets_of_related_ops):
        base_name = str(counter)
        base_name_to_sets_of_related_ops[base_name] = set_of_related_ops

    return base_name_to_sets_of_related_ops


def get_base_name_for_op(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
    op: NSNodeTargetType,
) -> str | None:
    for base_name, set_of_related_ops in base_name_to_sets_of_related_ops.items():
        if op in set_of_related_ops:
            return base_name
    return None


def add_op_to_sets_of_related_ops(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
    op: NSNodeTargetType,
    related_op: NSNodeTargetType | None,
) -> None:
    if related_op is not None:
        for set_of_related_ops in base_name_to_sets_of_related_ops.values():
            if related_op in set_of_related_ops:
                set_of_related_ops.add(op)
                return
        # if we got here, related_op was not found
        raise AssertionError(f"{related_op} was not found")
    else:
        counter = 0
        while str(counter) in base_name_to_sets_of_related_ops:
            counter += 1
        base_name_to_sets_of_related_ops[str(counter)] = {op}


# TODO(future PR): clean this up
def get_node_type_to_io_type_map() -> dict[str, set[NSNodeTargetType]]:
    FUNS_IO_TYPE_FP32: set[NSNodeTargetType] = {
        F.linear,
        F.conv1d,
        F.conv2d,
        F.conv3d,
        torch.cat,
        F.elu,
        F.hardswish,
        F.instance_norm,
        F.layer_norm,
        F.leaky_relu,
        F.dropout,
        F.silu,
        F.mish,
        operator.add,
        torch.add,
        operator.mul,
        torch.mul,
        torch.sum,
        F.prelu,
    }

    FUNS_IO_TYPE_FP16: set[NSNodeTargetType] = set()

    FUNS_IO_TYPE_INT8: set[NSNodeTargetType] = {
        toq.linear,
        toq.linear_relu,
        toq.conv1d,
        toq.conv1d_relu,
        toq.conv2d,
        toq.conv2d_relu,
        toq.conv3d,
        toq.conv3d_relu,
        toq.cat,
        toq.elu,
        toq.hardswish,
        toq.instance_norm,
        toq.layer_norm,
        toq.leaky_relu,
        toq.dropout,
        toq.prelu,
        # TODO(future PR): implement shadowing for binary ops and
        # uncomment below
        # toq.add,
        # toq.mul,
    }

    FUNS_IO_TYPE_FP32_OR_INT8: set[NSNodeTargetType] = {
        F.relu,
        F.tanh,
        torch.tanh,
        F.sigmoid,
        torch.sigmoid,
        F.hardsigmoid,
        operator.floordiv,
        torch.adaptive_avg_pool1d,
        F.adaptive_avg_pool2d,
        F.adaptive_avg_pool3d,
        F.dropout,
        F.hardtanh,
        F.hardtanh_,
        F.interpolate,
        F.max_pool1d,
        F.max_pool2d,
        F.max_pool3d,
        F.relu6,
        F.pixel_shuffle,
        F.pixel_unshuffle,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.cat,
        torch.chunk,
        torch.clamp,
        torch.flatten,
        torch.transpose,
        torch.max,
        torch.mean,
        torch.min,
        torch.narrow,
        torch.repeat_interleave,
        torch.sort,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        operator.add,
    }

    MODS_IO_TYPE_FP32: set[NSNodeTargetType] = {
        nn.Linear,
        nnqat.Linear,
        nnqatd.Linear,
        nnqd.Linear,
        torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nnqat.Conv1d,
        nnqat.Conv2d,
        nnqat.Conv3d,
        nnqat.Embedding,
        nnqat.EmbeddingBag,
        nn.LSTM,
        # note: nnqd.Linear is an instance of nnq.Linear, so this
        # check has to happen before the int8 module check
        nnqd.LSTM,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.Dropout,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.ELU,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.Hardswish,
        nn.LeakyReLU,
        nn.ReLU6,
        nn.SiLU,
        nn.Mish,
        nn.Softmax,
        nn.PReLU,
        nni.BNReLU2d,
        nni.BNReLU3d,
        nni.ConvReLU1d,
        nni.ConvReLU2d,
        nni.ConvReLU3d,
        nni.LinearReLU,
        nni.LinearBn1d,
        nni.ConvBn1d,
        nni.ConvBn2d,
        nni.ConvBn3d,
        nniqat.ConvBn1d,
        nniqat.ConvBn2d,
        nniqat.ConvBn3d,
        nniqat.ConvBnReLU1d,
        nniqat.ConvBnReLU2d,
        nniqat.ConvBnReLU3d,
        nniqat.ConvReLU1d,
        nniqat.ConvReLU2d,
        nniqat.ConvReLU3d,
        nniqat.LinearReLU,
        nniqat.LinearBn1d,
        nniqd.LinearReLU,
        nni.LinearLeakyReLU,
        nni.LinearTanh,
        nni.ConvAdd2d,
        nni.ConvAddReLU2d,
    }

    MODS_IO_TYPE_INT8: set[NSNodeTargetType] = {
        nnq.Linear,
        nnq.Conv1d,
        nnq.Conv2d,
        nnq.Conv3d,
        nnq.BatchNorm2d,
        nnq.BatchNorm3d,
        nnq.Dropout,
        nnq.ConvTranspose1d,
        nnq.ConvTranspose2d,
        nnq.ELU,
        nnq.InstanceNorm1d,
        nnq.InstanceNorm2d,
        nnq.InstanceNorm3d,
        nnq.LayerNorm,
        nnq.Hardswish,
        nnq.LeakyReLU,
        nnq.Embedding,
        nnq.EmbeddingBag,
        nnq.Dropout,
        nnq.Softmax,
        nnq.PReLU,
        nniq.BNReLU2d,
        nniq.BNReLU3d,
        nniq.ConvReLU1d,
        nniq.ConvReLU2d,
        nniq.ConvReLU3d,
        nniq.LinearReLU,
        nniq.LinearLeakyReLU,
        nniq.LinearTanh,
        nniq.ConvAdd2d,
        nniq.ConvAddReLU2d,
    }

    MODS_IO_TYPE_FP32_OR_INT8: set[NSNodeTargetType] = {
        nn.ReLU,
        nn.Tanh,
        nn.Sigmoid,
        nn.Hardsigmoid,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.Dropout,
        nn.Hardtanh,
        nn.Identity,
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.PixelShuffle,
        nn.PixelUnshuffle,
        nn.ReLU6,
    }

    METHS_IO_TYPE_FP32_OR_INT8: set[NSNodeTargetType] = {
        "sigmoid_",
        "sigmoid",
        "tanh_",
        "tanh",
        "hardsigmoid_",
        "hardsigmoid",
        "relu_",
        "relu",
    }

    return {
        "funs_io_type_fp32": FUNS_IO_TYPE_FP32,
        "funs_io_type_fp16": FUNS_IO_TYPE_FP16,
        "funs_io_type_int8": FUNS_IO_TYPE_INT8,
        "funs_io_type_fp32_or_int8": FUNS_IO_TYPE_FP32_OR_INT8,
        "mods_io_type_fp32": MODS_IO_TYPE_FP32,
        "mods_io_type_int8": MODS_IO_TYPE_INT8,
        "mods_io_type_fp32_or_int8": MODS_IO_TYPE_FP32_OR_INT8,
        "meths_io_type_fp32_or_int8": METHS_IO_TYPE_FP32_OR_INT8,
    }


def get_unmatchable_types_map() -> dict[str, set[NSNodeTargetType]]:
    FUNS_UNMATCHABLE: set[NSNodeTargetType] = {
        torch.quantize_per_tensor,
        operator.getitem,
    }

    MODS_UNMATCHABLE: set[NSNodeTargetType] = {
        nn.Identity,
    }

    METHS_UNMATCHABLE: set[NSNodeTargetType] = {
        "to",
        "dequantize",
        "reshape",
        "view",
        "unsqueeze_",
        "unsqueeze",
        "transpose",
        "squeeze_",
        "squeeze",
        "size",
        "shape",
        "resize_",
        "repeat_interleave",
        "repeat",
        "permute",
        "numel",
        "mean",
        "detach_",
        "detach",
        "contiguous",
        "clamp",
        "chunk",
    }

    return {
        "funs_unmatchable": FUNS_UNMATCHABLE,
        "mods_unmatchable": MODS_UNMATCHABLE,
        "meths_unmatchable": METHS_UNMATCHABLE,
    }
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any

import torch
from torch.ao.ns.fx.mappings import get_node_type_to_io_type_map
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.observer import _is_activation_post_process
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node

from .ns_types import NSNodeTargetType, NSSingleResultValuesType, NSSubgraph
from .utils import (
    get_arg_indices_of_inputs_to_log,
    get_node_first_input_and_output_type,
    get_node_input_qparams,
    get_normalized_nth_input,
    get_number_of_non_param_args,
    get_target_type_str,
    getattr_from_fqn,
    NodeInputOrOutputType,
    op_type_supports_shadowing,
    return_first_non_observer_node,
)


def _maybe_get_fqn(node: Node, gm: GraphModule) -> str | None:
    fqn = None
    if hasattr(gm, "_node_name_to_scope"):
        # fqn on observers is not present, because they do not
        # exist when the fqns are created during tracing. If this is
        # an observer, get the fqn of the node being observed.
        node_to_use_for_fqn = node
        if node.op == "call_module":
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str, got {type(node.target)}")
            module = getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(module):
                node_to_use_for_fqn = get_normalized_nth_input(node, gm, 0)
        fqn = gm._node_name_to_scope[node_to_use_for_fqn.name][0]  # type: ignore[index]
    return fqn  # type: ignore[return-value]


def _insert_logger_after_node(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    logger_node_name_suffix: str,
    ref_node_name: str,
    model_name: str,
    ref_name: str,
    ref_node_target_type: str,
    results_type: str,
    index_within_arg: int,
    index_of_arg: int,
    fqn: str | None,
) -> Node:
    """
    Given a starting graph of

    prev_node -> node -> next_node

    This function creates a new logger_cls obj and adds it
    after node, resulting in

    prev_node -> node -> logger_obj -> next_node
    """
    # create new name
    logger_node_name = get_new_attr_name_with_prefix(
        node.name + logger_node_name_suffix
    )(gm)
    target_type = get_target_type_str(node, gm)
    # create the logger object
    logger_obj = logger_cls(
        ref_node_name,
        node.name,
        model_name,
        ref_name,
        target_type,
        ref_node_target_type,
        results_type,
        index_within_arg,
        index_of_arg,
        fqn,
    )
    # attach the logger object to the parent module
    setattr(gm, logger_node_name, logger_obj)
    logger_node = node.graph.create_node("call_module", logger_node_name, (node,), {})
    return logger_node


def add_loggers_to_model(
    gm: GraphModule,
    node_to_instrument_inputs_to_ref_node_name: dict[Node, tuple[str, str]],
    node_to_instrument_outputs_to_ref_node_name: dict[Node, tuple[str, str]],
    logger_cls: Callable,
    model_name: str,
) -> GraphModule:
    """
    Takes the graph of gm, adds loggers to the output
    of each node in nodes_to_instrument. Returns a GraphModule with the new
    graph.
    """

    new_graph = Graph()
    env: dict[str, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    for node in gm.graph.nodes:
        if node.op == "output":
            new_graph.output(map_arg(get_normalized_nth_input(node, gm, 0), load_arg))
            continue

        if (node in node_to_instrument_inputs_to_ref_node_name) or (
            node in node_to_instrument_outputs_to_ref_node_name
        ):
            fqn = _maybe_get_fqn(node, gm)

            if node in node_to_instrument_inputs_to_ref_node_name:
                ref_name, ref_node_type = node_to_instrument_inputs_to_ref_node_name[
                    node
                ]
                # Ops such as add and mul are special because either
                # one or two of the first two arguments can be tensors,
                # and if one argument is a tensor it can be first or
                # second (x + 1 versus 1 + x).
                arg_indices_to_log = get_arg_indices_of_inputs_to_log(node)
                for node_arg_idx in arg_indices_to_log:
                    node_arg = get_normalized_nth_input(node, gm, node_arg_idx)
                    if type(node_arg) is Node:
                        # create a single input logger
                        prev_node = env[node_arg.name]
                        env[node_arg.name] = _insert_logger_after_node(
                            prev_node,
                            gm,
                            logger_cls,
                            "_ns_logger_",
                            node.name,
                            model_name,
                            ref_name,
                            ref_node_type,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0,
                            index_of_arg=node_arg_idx,
                            fqn=fqn,
                        )
                    elif (
                        type(node_arg) is torch.fx.immutable_collections.immutable_list
                    ):
                        # create N input loggers, one for each node
                        for arg_idx, arg in enumerate(node_arg):  # type: ignore[var-annotated, arg-type]
                            prev_node = env[arg.name]
                            env[prev_node.name] = _insert_logger_after_node(
                                prev_node,
                                gm,
                                logger_cls,
                                "_ns_logger_",
                                node.name,
                                model_name,
                                ref_name,
                                ref_node_type,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=arg_idx,
                                index_of_arg=node_arg_idx,
                                fqn=fqn,
                            )

            # ensure env is populated with base node
            # Note: runs for both inputs and outputs
            env[node.name] = new_graph.node_copy(node, load_arg)

            if node in node_to_instrument_outputs_to_ref_node_name:
                ref_name, ref_node_type = node_to_instrument_outputs_to_ref_node_name[
                    node
                ]
                # add the logger after the base node
                env[node.name] = _insert_logger_after_node(
                    env[node.name],
                    gm,
                    logger_cls,
                    "_ns_logger_",
                    node.name,
                    model_name,
                    ref_name,
                    ref_node_type,
                    NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0,
                    index_of_arg=0,
                    fqn=fqn,
                )

        else:
            env[node.name] = new_graph.node_copy(node, load_arg)

    new_gm = GraphModule(gm, new_graph)
    return new_gm


def _insert_quantize_per_tensor_node(
    prev_node_c: Node,
    node_a: Node,
    gm_b: GraphModule,
    graph_c: Graph,
    scale: torch.Tensor | float,
    zero_point: torch.Tensor | int,
    dtype_cast_name: str,
) -> Node:
    # copy scale
    scale_node_name = get_new_attr_name_with_prefix(node_a.name + "_input_scale_")(gm_b)
    setattr(gm_b, scale_node_name, scale)
    scale_node = graph_c.create_node(
        "get_attr", scale_node_name, (), {}, scale_node_name
    )
    # copy zero_point
    zero_point_node_name = get_new_attr_name_with_prefix(
        node_a.name + "_input_zero_point_"
    )(gm_b)
    setattr(gm_b, zero_point_node_name, zero_point)
    zero_point_node = graph_c.create_node(
        "get_attr", zero_point_node_name, (), {}, zero_point_node_name
    )
    # create the quantize_per_tensor call
    return graph_c.create_node(
        "call_function",
        torch.quantize_per_tensor,
        (prev_node_c, scale_node, zero_point_node, torch.quint8),
        {},
        dtype_cast_name,
    )


def _insert_dtype_cast_after_node(
    node_a: Node,
    node_c: Node,
    prev_node_c: Node | list[Node],
    gm_a: GraphModule,
    gm_b: GraphModule,
    graph_c: Graph,
    node_name_prefix: str,
    logger_cls: Callable,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]],
) -> Node | list[Node]:
    """
    Given a starting graph C (derived from graph B) of

    ... -> prev_node_c -> node_c -> ...

    And a corresponding related node_a, inserts the correct dtype
    cast node after prev_node_c to cast into the dtype expected
    by node_a, resulting in:

                          dtype_cast
                        /
    ... -> prev_node_c -> node_c -> ...

    For example, if node_c is an int8 op and node_a is an fp32 op, this function
    will insert a dequant.
    """
    dtype_cast_op = None
    dtype_cast_mod_cls = None
    dtype_cast_method = None
    dtype_cast_method_dtype = None
    dtype_cast_scale = None
    dtype_cast_zero_point = None
    node_input_type_a, _node_output_type_a = get_node_first_input_and_output_type(
        node_a, gm_a, logger_cls, node_type_to_io_type_map
    )
    node_input_type_c, _node_output_type_c = get_node_first_input_and_output_type(
        node_c, gm_b, logger_cls, node_type_to_io_type_map
    )

    if (
        (
            node_input_type_a == NodeInputOrOutputType.FP32
            and node_input_type_c == NodeInputOrOutputType.INT8
        )
        or (
            node_input_type_a == NodeInputOrOutputType.FP32
            and node_input_type_c == NodeInputOrOutputType.FP16
        )
        or
        # TODO(future PR): determine the actual dtype of node_c,
        # the current code only works because dequantize works with
        # multiple input dtypes.
        (
            node_input_type_a == NodeInputOrOutputType.FP32
            and node_input_type_c == NodeInputOrOutputType.FP32_OR_INT8
        )
    ):
        dtype_cast_op = torch.dequantize
    elif (
        node_input_type_a == node_input_type_c
        and node_input_type_a != NodeInputOrOutputType.UNKNOWN
    ):
        dtype_cast_mod_cls = torch.nn.Identity
    elif (
        node_input_type_a == NodeInputOrOutputType.INT8
        and node_input_type_c == NodeInputOrOutputType.FP32
    ):
        # int8 shadows fp32, the dtype cast needs to quantize to int8
        # with the right qparams.
        node_a_input_qparams = get_node_input_qparams(
            node_a, gm_a, node_type_to_io_type_map
        )
        if node_a_input_qparams is not None:
            dtype_cast_op = torch.quantize_per_tensor  # type: ignore[assignment]
            dtype_cast_scale, dtype_cast_zero_point = node_a_input_qparams
    elif (
        node_input_type_a == NodeInputOrOutputType.FP16
        and node_input_type_c == NodeInputOrOutputType.FP32
    ):
        dtype_cast_method = "to"
        dtype_cast_method_dtype = torch.float16
    else:
        raise AssertionError(
            f"dtype cast from {node_input_type_c} {node_c.format_node()} to "
            + f"{node_input_type_a} {node_a.format_node()} needs to be implemented"
        )

    if isinstance(prev_node_c, Node):
        new_dtype_cast_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        if dtype_cast_op:
            if dtype_cast_scale is not None and dtype_cast_zero_point is not None:
                return _insert_quantize_per_tensor_node(
                    prev_node_c,
                    node_a,
                    gm_b,
                    graph_c,
                    dtype_cast_scale,
                    dtype_cast_zero_point,
                    new_dtype_cast_name,
                )
            else:
                return graph_c.create_node(
                    "call_function",
                    dtype_cast_op,
                    (prev_node_c,),
                    {},
                    new_dtype_cast_name,
                )
        elif dtype_cast_method:
            return graph_c.create_node(
                "call_method",
                dtype_cast_method,
                (prev_node_c, dtype_cast_method_dtype),
                {},
                new_dtype_cast_name,
            )
        else:
            if not dtype_cast_mod_cls:
                raise AssertionError("Expected dtype_cast_mod_cls to be not None")
            dtype_cast_mod = dtype_cast_mod_cls()
            setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
            return graph_c.create_node(
                "call_module",
                new_dtype_cast_name,
                (prev_node_c,),
                {},
                new_dtype_cast_name,
            )
    elif isinstance(prev_node_c, list):
        results = []
        for prev_node_c_inner in prev_node_c:
            new_dtype_cast_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
            if dtype_cast_op:
                # TODO(future PR): add handling for quantize_per_tensor
                new_dtype_cast_node = graph_c.create_node(
                    "call_function",
                    dtype_cast_op,
                    (prev_node_c_inner,),
                    {},
                    new_dtype_cast_name,
                )
                results.append(new_dtype_cast_node)
            else:
                if not dtype_cast_mod_cls:
                    raise AssertionError("Expected dtype_cast_mod_cls to be not None")
                dtype_cast_mod = dtype_cast_mod_cls()
                setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
                new_dtype_cast_node = graph_c.create_node(
                    "call_module",
                    new_dtype_cast_name,
                    (prev_node_c_inner,),
                    {},
                    new_dtype_cast_name,
                )
                results.append(new_dtype_cast_node)
        return results
    else:
        raise AssertionError(f"type f{type(prev_node_c)} is not handled")


# TODO(future PR): look into using copy_node API instead
def _copy_node_from_a_to_c(
    node_a: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    graph_c: Graph,
) -> Node:
    """
    Simple copy of node_a to graph_c.
    """
    if node_a.op == "get_attr":
        node_a_copy_name = get_new_attr_name_with_prefix(node_a.name + "_shadow_copy_")(
            gm_b
        )
        node_a_obj = getattr_from_fqn(gm_a, node_a.target)  # type: ignore[arg-type]
        if torch.is_tensor(node_a_obj):
            node_a_obj = node_a_obj.detach()
        setattr(gm_b, node_a_copy_name, node_a_obj)
        node_a_copy = graph_c.create_node(
            node_a.op, node_a_copy_name, (), {}, node_a_copy_name
        )
        return node_a_copy
    elif node_a.op == "call_method":
        if node_a.target not in ("dequantize", "to"):
            raise AssertionError(f"target {node_a.target} is not implemented")
        if node_a.target == "dequantize":
            arg_copy = _copy_node_from_a_to_c(
                get_normalized_nth_input(node_a, gm_a, 0), gm_a, gm_b, graph_c
            )  # type: ignore[arg-type]
            node_a_copy_name = get_new_attr_name_with_prefix(
                node_a.name + "_shadow_copy_"
            )(gm_b)
            node_a_copy = graph_c.create_node(
                node_a.op, node_a.target, (arg_copy,), {}, node_a_copy_name
            )
            return node_a_copy
        else:  # to
            arg_copy = _copy_node_from_a_to_c(
                get_normalized_nth_input(node_a, gm_a, 0), gm_a, gm_b, graph_c
            )  # type: ignore[arg-type]
            node_a_copy_name = get_new_attr_name_with_prefix(
                node_a.name + "_shadow_copy_"
            )(gm_b)
            node_a_copy = graph_c.create_node(
                node_a.op,
                node_a.target,
                (arg_copy, get_normalized_nth_input(node_a, gm_a, 1)),
                {},
                node_a_copy_name,
            )
            return node_a_copy

    else:
        raise AssertionError(
            f"handling of node {node_a.format_node()} with op {node_a.op} is not implemented"
        )


def _can_insert_copy_of_subgraph_a(
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    num_non_param_args_node_a: int,
) -> bool:
    """
    This function returns `False` if the input subgraph cannot be copied by
    `_insert_copy_of_subgraph_a_after_input_node_c`. This usually means
    that there is a corner case logic for which copy is not yet implemented.
    """
    # populate the list of nodes we need to check
    nodes = []
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        nodes.append(cur_node)
        cur_node = get_normalized_nth_input(cur_node, gm_a, 0)  # type: ignore[assignment]
    nodes.append(cur_node)
    nodes.reverse()

    def _can_insert(node_a_arg, gm_a):
        if isinstance(node_a_arg, Node):
            arg_a = return_first_non_observer_node(node_a_arg, gm_a)
            if arg_a.op == "call_method":
                return arg_a.target in ("dequantize", "to")
            elif arg_a.op == "get_attr":
                return True
            else:
                return False
        elif isinstance(node_a_arg, (list, tuple)):
            for el in node_a_arg:
                if not isinstance(el, Node):
                    return False
        return True

    # For each node, check if we handle the copy behavior. This follows the
    # logic in `_insert_copy_of_subgraph_a_after_input_node_c`.
    for node_a in nodes:
        local_num_non_param_args_node_a = (
            num_non_param_args_node_a if node_a is nodes[0] else 1
        )

        norm_args_kwargs = node_a.normalized_arguments(
            gm_a, normalize_to_only_use_kwargs=True
        )
        if norm_args_kwargs is not None:
            norm_args, norm_kwargs = norm_args_kwargs
        else:
            norm_args, norm_kwargs = node_a.args, node_a.kwargs

        cur_idx = 0

        while cur_idx < len(norm_args):
            if cur_idx == 0:
                pass
            elif cur_idx == 1 and local_num_non_param_args_node_a == 2:
                pass
            else:
                if not _can_insert(norm_args[cur_idx], gm_a):
                    return False
            cur_idx += 1

        for kwarg_val in norm_kwargs.values():
            # stitch the inputs from base graph
            if cur_idx == 0:
                pass
            elif cur_idx == 1 and local_num_non_param_args_node_a == 2:
                pass
            else:
                if not _can_insert(kwarg_val, gm_a):
                    return False
            cur_idx += 1

    return True


def _insert_copy_of_subgraph_a_after_input_node_c(
    input_node_c: Node | list[Node],
    input_node_c_2: Node | list[Node] | None,
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    TODO(before land): real docblock
    """
    if not isinstance(input_node_c, (Node, list)):
        raise AssertionError(f"Expected Node or list, got {type(input_node_c)}")

    # create a sequential list of the subgraphs' nodes from start to end,
    # because we need to add the nodes to graph C in non-reverse order
    nodes_of_a = [subgraph_a.end_node]
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        cur_node = get_normalized_nth_input(cur_node, gm_a, 0)  # type: ignore[assignment]
        nodes_of_a.insert(0, cur_node)

    # go through nodes of a in order, and insert them into the graph of c
    # sequentially
    cur_node_a = nodes_of_a[0]
    cur_node_c = _insert_copy_of_node_a_after_input_node_c(
        input_node_c, input_node_c_2, cur_node_a, gm_a, gm_b, node_name_prefix
    )
    for cur_idx_a in range(1, len(nodes_of_a)):
        cur_node_a = nodes_of_a[cur_idx_a]
        prev_node_c = cur_node_c  # previous added node is the input to next node
        cur_node_c = _insert_copy_of_node_a_after_input_node_c(
            prev_node_c,
            # TODO(future PR): enable multiple inputs for nodes which are not at start of subgraph
            None,
            cur_node_a,
            gm_a,
            gm_b,
            node_name_prefix,
        )
    # return the last inserted node
    return cur_node_c


def _insert_copy_of_node_a_after_input_node_c(
    input_node_c: Node | list[Node],
    input_node_c_2: Node | list[Node] | None,
    node_a: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    Assume that node_a from graph_a has
      args (input, (input2)?, arg1, ...), and
      kwargs {kw0: kwarg0, ...}

    Note: input2 is optional. If it equals to None, we assume that the op
    has a single non-param input.  If it is specified, we assume that the op
    has two non-param inputs.

    Copies the underlying values of arg1..argn and kwarg0..kwargn into gm_b,
    and creates the corresponding nodes in graph_c. Note: observers are ignored,
    so if an arg is an observer we navigate up until we find a non-observer parent.

    If node_a is a call_module, points the module pointed to by node_a to gm_b.

    Creates the copy of node_a in graph_c, with input as the first arg,
    and all other args and kwargs pointing to the copies of the objects
    in gm_b created above.

    An example in pictures:

    graph A:
    ========

    input -------------> node_a
