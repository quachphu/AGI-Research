        is_dynamic = dtype_config_dict.get(IS_DYNAMIC_DICT_KEY)
        return cls(input_dtype, output_dtype, weight_dtype, bias_dtype, is_dynamic)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``DTypeConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.DTypeConfig.from_dict`.
        """
        dtype_config_dict: dict[str, Any] = {}
        if self.input_dtype is not None:
            dtype_config_dict[INPUT_DTYPE_DICT_KEY] = self.input_dtype_with_constraints
        if self.output_dtype is not None:
            dtype_config_dict[OUTPUT_DTYPE_DICT_KEY] = (
                self.output_dtype_with_constraints
            )
        if self.weight_dtype is not None:
            dtype_config_dict[WEIGHT_DTYPE_DICT_KEY] = (
                self.weight_dtype_with_constraints
            )
        if self.bias_dtype is not None:
            dtype_config_dict[BIAS_DTYPE_DICT_KEY] = self.bias_dtype
        if self.is_dynamic is not None:
            dtype_config_dict[IS_DYNAMIC_DICT_KEY] = self.is_dynamic
        return dtype_config_dict


class BackendConfig:
    # TODO: refer to NativeBackendConfig once that is implemented
    """Config that defines the set of patterns that can be quantized on a given backend, and how reference
    quantized models can be produced from these patterns.

    A pattern in this context refers to a module, a functional, an operator, or a directed acyclic graph
    of the above. Each pattern supported on the target backend can be individually configured through
    :class:`~torch.ao.quantization.backend_config.BackendPatternConfig` in terms of:

    (1) The supported input/output activation, weight, and bias data types

    (2) How observers and quant/dequant ops are inserted in order to construct the reference pattern, and

    (3) (Optionally) Fusion, QAT, and reference module mappings.

    The format of the patterns is described in:
    https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md

    Example usage::

        import torch
        from torch.ao.quantization.backend_config import (
            BackendConfig,
            BackendPatternConfig,
            DTypeConfig,
            ObservationType,
        )

        weighted_int8_dtype_config = DTypeConfig(
            input_dtype=torch.quint8,
            output_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            bias_dtype=torch.float)

        def fuse_conv2d_relu(is_qat, conv, relu):
            return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)

        # For quantizing Linear
        linear_config = BackendPatternConfig(torch.nn.Linear) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .add_dtype_config(weighted_int8_dtype_config) \
            .set_root_module(torch.nn.Linear) \
            .set_qat_module(torch.ao.nn.qat.Linear) \
            .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

        # For fusing Conv2d + ReLU into ConvReLU2d
        conv_relu_config = BackendPatternConfig((torch.nn.Conv2d, torch.nn.ReLU)) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .add_dtype_config(weighted_int8_dtype_config) \
            .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
            .set_fuser_method(fuse_conv2d_relu)

        # For quantizing ConvReLU2d
        fused_conv_relu_config = BackendPatternConfig(torch.ao.nn.intrinsic.ConvReLU2d) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .add_dtype_config(weighted_int8_dtype_config) \
            .set_root_module(torch.nn.Conv2d) \
            .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d) \
            .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)

        backend_config = BackendConfig("my_backend") \
            .set_backend_pattern_config(linear_config) \
            .set_backend_pattern_config(conv_relu_config) \
            .set_backend_pattern_config(fused_conv_relu_config)

    """

    def __init__(self, name: str = ""):
        self.name = name
        # Store all BackendPatternConfigs in a map to handle duplicates
        # Note: the key in this map uses the complex reversed tuple format.
        # This is intended only for internal use; users who wish to access
        # the original patterns should go through `self.configs` instead.
        self._pattern_complex_format_to_config: dict[Pattern, BackendPatternConfig] = {}

    def __repr__(self):
        return f"BackendConfig({self.__dict__})"

    def set_name(self, name: str) -> BackendConfig:
        """
        Set the name of the target backend.
        """
        self.name = name
        return self

    def set_backend_pattern_config(self, config: BackendPatternConfig) -> BackendConfig:
        """
        Set the config for an pattern that can be run on the target backend.
        This overrides any existing config for the given pattern.
        """
        # Avoid circular dependencies
        pattern_complex_format = torch.ao.quantization.backend_config.utils._get_pattern_in_reversed_nested_tuple_format(
            config
        )  # type: ignore[attr-defined]
        self._pattern_complex_format_to_config[pattern_complex_format] = config
        return self

    def set_backend_pattern_configs(
        self, configs: list[BackendPatternConfig]
    ) -> BackendConfig:
        """
        Set the configs for patterns that can be run on the target backend.
        This overrides any existing config for a given pattern if it was previously registered already.
        """
        for conf in configs:
            self.set_backend_pattern_config(conf)
        return self

    @property
    def configs(self) -> list[BackendPatternConfig]:
        """
        Return a copy of the list of configs set in this `BackendConfig`.
        """
        return list(self._pattern_complex_format_to_config.values())

    @classmethod
    def from_dict(cls, backend_config_dict: dict[str, Any]) -> BackendConfig:
        """
        Create a ``BackendConfig`` from a dictionary with the following items:

            "name": the name of the target backend

            "configs": a list of dictionaries that each represents a `BackendPatternConfig`

        """
        conf = cls(backend_config_dict.get(NAME_DICT_KEY, ""))
        for d in backend_config_dict.get(CONFIGS_DICT_KEY, []):
            if isinstance(d, BackendPatternConfig):
                conf.set_backend_pattern_config(d)
            elif isinstance(d, dict):
                conf.set_backend_pattern_config(BackendPatternConfig.from_dict(d))
            else:
                raise ValueError(
                    f"Expected backend_config_dict['{CONFIGS_DICT_KEY}'] to be a dictionary"
                )
        return conf

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``BackendConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendConfig.from_dict`.
        """
        return {
            NAME_DICT_KEY: self.name,
            CONFIGS_DICT_KEY: [c.to_dict() for c in self.configs],
        }


class BackendPatternConfig:
    """
    Config object that specifies quantization behavior for a given operator pattern.
    For a detailed example usage, see :class:`~torch.ao.quantization.backend_config.BackendConfig`.
    """

    def __init__(self, pattern: Pattern | None = None):
        self.pattern: Pattern | None = pattern
        self.observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        self.dtype_configs: list[DTypeConfig] = []
        self.root_module: type[torch.nn.Module] | None = None
        self.qat_module: type[torch.nn.Module] | None = None
        self.reference_quantized_module: type[torch.nn.Module] | None = None
        self.fused_module: type[torch.nn.Module] | None = None
        self.fuser_method: Callable | None = None

        # Temporary/internal configs
        self._root_node_getter: Callable | None = None
        self._extra_inputs_getter: Callable | None = None
        self._num_tensor_args_to_observation_type: dict[int, ObservationType] = {}
        self._input_type_to_index: dict[str, int] = {}
        self._pattern_complex_format: Pattern | None = None

    def __repr__(self):
        dict_nonempty = {
            k: v
            for k, v in self.__dict__.items()
            if (
                (not isinstance(v, (list, dict)) and v is not None)
                or (isinstance(v, (list, dict)) and len(v) > 0)
            )
        }
        return f"BackendPatternConfig({dict_nonempty})"

    def set_pattern(self, pattern: Pattern) -> BackendPatternConfig:
        """
        Set the pattern to configure.

        The pattern can be a float module, functional operator, pytorch operator, or a tuple
        combination of the above. Tuple patterns are treated as sequential patterns, and
        currently only tuples of 2 or 3 elements are supported.
        """
        if self._pattern_complex_format is not None:
            raise ValueError(
                "Only one of 'pattern' or 'pattern_complex_format' can be set"
            )
        self.pattern = pattern
        return self

    def set_observation_type(
        self, observation_type: ObservationType
    ) -> BackendPatternConfig:
        """
        Set how observers should be inserted in the graph for this pattern.

        Observation type here refers to how observers (or quant-dequant ops) will be placed
        in the graph. This is used to produce the desired reference patterns understood by
        the backend. Weighted ops such as linear and conv require different observers
        (or quantization parameters passed to quantize ops in the reference model) for the
        input and the output.

        There are two observation types:

            `OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT` (default): the output observer instance
            will be different from the input. This is the most common observation type.

            `OUTPUT_SHARE_OBSERVER_WITH_INPUT`: the output observer instance will be the
            same as the input. This is useful for operators like `cat`.

        Note: This will be renamed in the near future, since we will soon insert QuantDeQuantStubs
        with observers (and fake quantizes) attached instead of observers themselves.
        """
        self.observation_type = observation_type
        return self

    def add_dtype_config(self, dtype_config: DTypeConfig) -> BackendPatternConfig:
        """
        Add a set of supported data types passed as arguments to quantize ops in the
        reference model spec.
        """
        self.dtype_configs.append(dtype_config)
        return self

    def set_dtype_configs(
        self, dtype_configs: list[DTypeConfig]
    ) -> BackendPatternConfig:
        """
        Set the supported data types passed as arguments to quantize ops in the
        reference model spec, overriding all previously registered data types.
        """
        self.dtype_configs = dtype_configs
        return self

    def set_root_module(
        self, root_module: type[torch.nn.Module]
    ) -> BackendPatternConfig:
        """
        Set the module that represents the root for this pattern.

        When we construct the reference quantized model during the convert phase,
        the root modules (e.g. torch.nn.Linear for torch.ao.nn.intrinsic.LinearReLU)
        will be swapped to the corresponding reference quantized modules (e.g.
        torch.ao.nn.reference.quantized.Linear). This allows custom backends to
        specify custom reference quantized module implementations to match the
        numerics of their lowered operators. Since this is a one-to-one mapping,
        both the root module and the reference quantized module must be specified
        in the same BackendPatternConfig in order for the conversion to take place.
        """
        self.root_module = root_module
        return self

    def set_qat_module(self, qat_module: type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the QAT implementation for this pattern.
        """
        self.qat_module = qat_module
        return self

    def set_reference_quantized_module(
        self, reference_quantized_module: type[torch.nn.Module]
    ) -> BackendPatternConfig:
        """
        Set the module that represents the reference quantized implementation for
        this pattern's root module.

        For more detail, see :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module`.
        """
        self.reference_quantized_module = reference_quantized_module
        return self

    def set_fused_module(
        self, fused_module: type[torch.nn.Module]
    ) -> BackendPatternConfig:
        """
        Set the module that represents the fused implementation for this pattern.
        """
        self.fused_module = fused_module
        return self

    def set_fuser_method(self, fuser_method: Callable) -> BackendPatternConfig:
        """
        Set the function that specifies how to fuse this BackendPatternConfig's pattern.

        The first argument of this function should be `is_qat`, and the rest of the arguments
        should be the items in the tuple pattern. The return value of this function should be
        the resulting fused module.

        For example, the fuser method for the pattern `(torch.nn.Linear, torch.nn.ReLU)` can be:

            def fuse_linear_relu(is_qat, linear, relu):
                return torch.ao.nn.intrinsic.LinearReLU(linear, relu)

        For a more complicated example, see https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6.
        """
        self.fuser_method = fuser_method
        return self

    def _set_root_node_getter(self, root_node_getter: Callable) -> BackendPatternConfig:
        self._root_node_getter = root_node_getter
        return self

    def _set_extra_inputs_getter(
        self, extra_inputs_getter: Callable
    ) -> BackendPatternConfig:
        self._extra_inputs_getter = extra_inputs_getter
        return self

    def _set_num_tensor_args_to_observation_type(
        self, num_tensor_args_to_observation_type: dict[int, ObservationType]
    ) -> BackendPatternConfig:
        self._num_tensor_args_to_observation_type = num_tensor_args_to_observation_type
        return self

    def _set_input_type_to_index(
        self, input_type_to_index: dict[str, int]
    ) -> BackendPatternConfig:
        self._input_type_to_index = input_type_to_index
        return self

    def _set_pattern_complex_format(self, pattern: Pattern) -> BackendPatternConfig:
        """
        Set the pattern to configure, using the reversed nested tuple format.

        See the BackendConfig README for more detail:
        https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md#advanced-pattern-specification
        """
        if self.pattern is not None:
            raise ValueError(
                "Only one of 'pattern' or 'pattern_complex_format' can be set"
            )
        self._pattern_complex_format = pattern
        return self

    @classmethod
    def from_dict(
        cls, backend_pattern_config_dict: dict[str, Any]
    ) -> BackendPatternConfig:
        """
        Create a ``BackendPatternConfig`` from a dictionary with the following items:

            "pattern": the pattern being configured
            "observation_type": the :class:`~torch.ao.quantization.backend_config.ObservationType` that specifies how
            observers should be inserted for this pattern
            "dtype_configs": a list of dictionaries that represents :class:`~torch.ao.quantization.backend_config.DTypeConfig` s
            "root_module": a :class:`torch.nn.Module` that represents the root for this pattern
            "qat_module": a :class:`torch.nn.Module` that represents the QAT implementation for this pattern
            "reference_quantized_module": a :class:`torch.nn.Module` that represents the reference quantized
            implementation for this pattern's root module.
            "fused_module": a :class:`torch.nn.Module` that represents the fused implementation for this pattern
            "fuser_method": a function that specifies how to fuse the pattern for this pattern
            "pattern_complex_format": the pattern specified in the reversed nested tuple format (deprecated)

        """

        def _get_dtype_config(obj: Any) -> DTypeConfig:
            """
            Convert the given object into a ``DTypeConfig`` if possible, else throw an exception.
            """
            if isinstance(obj, DTypeConfig):
                return obj
            if isinstance(obj, dict):
                return DTypeConfig.from_dict(obj)
            raise ValueError(
                f"Expected a list of DTypeConfigs in "
                f"backend_pattern_config_dict[\"{DTYPE_CONFIGS_DICT_KEY}\"], got '{type(obj)}'"
            )

        conf = cls()
        if PATTERN_DICT_KEY in backend_pattern_config_dict:
            conf.set_pattern(backend_pattern_config_dict[PATTERN_DICT_KEY])
        if OBSERVATION_TYPE_DICT_KEY in backend_pattern_config_dict:
            conf.set_observation_type(
                backend_pattern_config_dict[OBSERVATION_TYPE_DICT_KEY]
            )
        for d in backend_pattern_config_dict.get(DTYPE_CONFIGS_DICT_KEY, []):
            conf.add_dtype_config(_get_dtype_config(d))
        conf.set_root_module(
            backend_pattern_config_dict.get(ROOT_MODULE_DICT_KEY)  # type: ignore[arg-type]
        )
        conf.set_qat_module(backend_pattern_config_dict.get(QAT_MODULE_DICT_KEY))  # type: ignore[arg-type]
        conf.set_reference_quantized_module(
            backend_pattern_config_dict.get(REFERENCE_QUANTIZED_MODULE_DICT_KEY)  # type: ignore[arg-type]
        )
        conf.set_fused_module(
            backend_pattern_config_dict.get(FUSED_MODULE_DICT_KEY)  # type: ignore[arg-type]
        )
        conf.set_fuser_method(
            backend_pattern_config_dict.get(FUSER_METHOD_DICT_KEY)  # type: ignore[arg-type]
        )
        conf._set_root_node_getter(
            backend_pattern_config_dict.get(ROOT_NODE_GETTER_DICT_KEY)  # type: ignore[arg-type]
        )
        conf._set_extra_inputs_getter(
            backend_pattern_config_dict.get(EXTRA_INPUTS_GETTER_DICT_KEY)  # type: ignore[arg-type]
        )
        conf._set_num_tensor_args_to_observation_type(
            backend_pattern_config_dict.get(
                NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY, {}
            )
        )
        conf._set_input_type_to_index(
            backend_pattern_config_dict.get(INPUT_TYPE_TO_INDEX_DICT_KEY, {})
        )
        if PATTERN_COMPLEX_FORMAT_DICT_KEY in backend_pattern_config_dict:
            conf._set_pattern_complex_format(
                backend_pattern_config_dict[PATTERN_COMPLEX_FORMAT_DICT_KEY]
            )
        return conf

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this ``BackendPatternConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.from_dict`.
        """
        backend_pattern_config_dict: dict[str, Any] = {
            OBSERVATION_TYPE_DICT_KEY: self.observation_type,
            DTYPE_CONFIGS_DICT_KEY: [c.to_dict() for c in self.dtype_configs],
        }
        if self.pattern is not None:
            backend_pattern_config_dict[PATTERN_DICT_KEY] = self.pattern
        if self.root_module is not None:
            backend_pattern_config_dict[ROOT_MODULE_DICT_KEY] = self.root_module
        if self.qat_module is not None:
            backend_pattern_config_dict[QAT_MODULE_DICT_KEY] = self.qat_module
        if self.reference_quantized_module is not None:
            backend_pattern_config_dict[REFERENCE_QUANTIZED_MODULE_DICT_KEY] = (
                self.reference_quantized_module
            )
        if self.fused_module is not None:
            backend_pattern_config_dict[FUSED_MODULE_DICT_KEY] = self.fused_module
        if self.fuser_method is not None:
            backend_pattern_config_dict[FUSER_METHOD_DICT_KEY] = self.fuser_method
        if self._root_node_getter is not None:
            backend_pattern_config_dict[ROOT_NODE_GETTER_DICT_KEY] = (
                self._root_node_getter
            )
        if self._extra_inputs_getter is not None:
            backend_pattern_config_dict[EXTRA_INPUTS_GETTER_DICT_KEY] = (
                self._extra_inputs_getter
            )
        if len(self._num_tensor_args_to_observation_type) > 0:
            backend_pattern_config_dict[
                NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY
            ] = self._num_tensor_args_to_observation_type
        if len(self._input_type_to_index) > 0:
            backend_pattern_config_dict[INPUT_TYPE_TO_INDEX_DICT_KEY] = (
                self._input_type_to_index
            )
        if self._pattern_complex_format is not None:
            backend_pattern_config_dict[PATTERN_COMPLEX_FORMAT_DICT_KEY] = (
                self._pattern_complex_format
            )
        return backend_pattern_config_dict
<|endoftext|>import torch

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
)
from .backend_config import BackendConfig, DTypeConfig, DTypeWithConstraints


__all__ = [
    "get_qnnpack_backend_config",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

qnnpack_weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

qnnpack_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

qnnpack_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

qnnpack_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

qnnpack_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

qnnpack_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

qnnpack_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)

# xnnpack compatible dtype configs

# We restrict scale values to be 2 ** -12 to ensure the
# requantization scale never falls below the xnnpack lower
# threshold. Additionally, for qint8 weight, we restrict
# the quantization values to [-127, +127], excluding -128.
# For more detail, refer to the description of
# `default_symmetric_qnnpack_qconfig`.

# TODO: add additional restriction on qscheme to ensure it
# is either per_tensor_symmetric or per_channel_symmetric

qnnpack_act_qint8_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    scale_min_lower_bound=2**-12,
)

qnnpack_weight_qint8_neg_127_to_127_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    quant_min_lower_bound=-127,
    quant_max_upper_bound=127,
    scale_min_lower_bound=2**-12,
)

qnnpack_weighted_op_qint8_symmetric_dtype_config = DTypeConfig(
    input_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    output_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    weight_dtype=qnnpack_weight_qint8_neg_127_to_127_scale_min_2_neg_12,
    bias_dtype=torch.float,
)

qnnpack_default_op_qint8_symmetric_dtype_config = DTypeConfig(
    input_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    output_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_qnnpack_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native QNNPACK backend.
    """
    conv_dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
    ]
    linear_dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
        qnnpack_default_dynamic_int8_dtype_config,
        qnnpack_default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    default_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    fixed_qparams_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    rnn_op_dtype_configs = [
        qnnpack_default_dynamic_int8_dtype_config,
        qnnpack_default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        qnnpack_weight_only_quint8_dtype_config,
        qnnpack_weight_only_quint4x2_dtype_config,
    ]
    return (
        BackendConfig("qnnpack")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )
<|endoftext|># TODO: rename executorch to qnnpack_executorch since executorch is a general runtime
# not a specific backend

import operator

import torch
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fuser_method_mappings import (
    _sequential_wrapper2,
    fuse_conv_bn,
    fuse_conv_bn_relu,
)

from ._common_operator_config_utils import _Conv2dMetadata
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from .qnnpack import (
    qnnpack_default_op_qint8_symmetric_dtype_config,
    qnnpack_weighted_op_qint8_symmetric_dtype_config,
)


__all__ = [
    "get_executorch_backend_config",
]


# ===================
# |  DTYPE CONFIGS  |
# ===================

executorch_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

executorch_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

executorch_default_dynamic_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

executorch_act_qint8_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    scale_min_lower_bound=2**-12,
)

executorch_weight_qint8_neg_127_to_127_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    quant_min_lower_bound=-127,
    quant_max_upper_bound=127,
    scale_min_lower_bound=2**-12,
)

executorch_default_dynamic_qint8_dtype_config = DTypeConfig(
    input_dtype=executorch_act_qint8_scale_min_2_neg_12,
    output_dtype=torch.float,
    weight_dtype=executorch_weight_qint8_neg_127_to_127_scale_min_2_neg_12,
    bias_dtype=torch.float,
    is_dynamic=True,
)

executorch_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

executorch_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)


# =============================
# |  BACKEND PATTERN CONFIGS  |
# =============================


def _get_linear_configs() -> list[BackendPatternConfig]:
    """
    Return all configs related to linear modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        executorch_weighted_op_int8_dtype_config,
        executorch_default_dynamic_quint8_dtype_config,
        executorch_default_dynamic_qint8_dtype_config,
        executorch_default_dynamic_float16_dtype_config,
    ]
    linear_configs: list[BackendPatternConfig] = []
    # linear module
    linear_configs.append(
        BackendPatternConfig(torch.nn.Linear)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
        .set_qat_module(nnqat.Linear)
    )
    # linear qat module
    linear_configs.append(
        BackendPatternConfig(nnqat.Linear)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
    )
    # functional linear
    linear_configs.append(
        BackendPatternConfig(torch.nn.functional.linear)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    return linear_configs


def _get_conv_configs() -> list[BackendPatternConfig]:
    """
    Return all configs related to conv modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        executorch_weighted_op_int8_dtype_config,
    ]
    conv_configs = []
    for convs in [_Conv2dMetadata]:
        # (1) Single conv modules/functions
        # -----------------------------------
        # conv module
        conv_configs.append(
            BackendPatternConfig(convs.root)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
            .set_qat_module(convs.qat)
        )
        # conv qat module
        conv_configs.append(
            BackendPatternConfig(convs.qat)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # functional conv
        conv_configs.append(
            BackendPatternConfig(convs.func)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2})
        )

        # (2) Conv + relu
        # -----------------------------------
        # conv module + relu module
        conv_configs.append(
            BackendPatternConfig((convs.root, nn.ReLU))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu))
            .set_fused_module(convs.fused_conv_relu)
        )
        # conv module + functional relu
        conv_configs.append(
            BackendPatternConfig((convs.root, F.relu))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu))
            .set_fused_module(convs.fused_conv_relu)
        )
        # fused conv relu module
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
            .set_qat_module(convs.relu_qat)
        )
        # conv relu, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.relu_qat)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # functional conv + relu module
        conv_configs.append(
            BackendPatternConfig((convs.func, nn.ReLU))
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
        )
        # functional conv + functional relu
        conv_configs.append(
            BackendPatternConfig((convs.func, F.relu))
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
        )
        # fused conv relu
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
            .set_dtype_configs(dtype_configs)
            .set_qat_module(convs.relu_qat)
        )

        conv_configs.append(
            BackendPatternConfig(convs.relu_qat)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )

        # (3) Conv + batchnorm (+ relu)
        # -------------------------------
        # conv + batchnorm (+ relu)
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(fuse_conv_bn)
            .set_fused_module(convs.fused_conv_bn)
        )
        # conv + bn + relu module fusion
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn, nn.ReLU))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(fuse_conv_bn_relu)
            .set_fused_module(convs.fused_conv_bn_relu)
        )
        # conv + bn + relu functional fusion
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn, F.relu))
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_fuser_method(fuse_conv_bn_relu)
            .set_fused_module(convs.fused_conv_bn_relu)
        )
        # TODO: we can add fusion for torch.relu as well
        # 3.2 conv + bn (+ relu) fused module configs
        # fused conv bn
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_bn)
            .set_dtype_configs(dtype_configs)
            .set_qat_module(convs.bn_qat)
        )

        # fused conv bn relu
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_bn_relu)
            .set_dtype_configs(dtype_configs)
            .set_qat_module(convs.bn_relu_qat)
        )

        # conv bn, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.bn_qat)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # conv bn relu, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.bn_relu_qat)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
    return conv_configs


def _get_binary_ops_configs() -> list[BackendPatternConfig]:
    """
    Return all configs related to binary ops.
    """
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        executorch_weighted_op_int8_dtype_config,
    ]
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    binary_op_configs: list[BackendPatternConfig] = []
    for op in [
        operator.add,
        torch.add,
        operator.sub,
        torch.sub,
        operator.mul,
        torch.mul,
    ]:
        bop_patterns = [
            (op, torch.nn.ReLU),
            (op, torch.nn.functional.relu),
            (op, torch.relu),
            op,
        ]
        binary_op_configs.extend(
            BackendPatternConfig(bop_pattern)
            .set_dtype_configs(dtype_configs)
            ._set_num_tensor_args_to_observation_type(
                num_tensor_args_to_observation_type_mapping
            )
            for bop_pattern in bop_patterns
        )
    return binary_op_configs


def _get_share_qparams_ops_configs() -> list[BackendPatternConfig]:
    """
    Return the operator configs for the operators that works for both float and quantized
    input if input is quantized, the output Tensor shares the same quantization parameter
    with input.

    Example operator: avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - avgpool2d - observer_0 (same observer instance as input)
    """
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        executorch_default_op_quint8_dtype_config,
    ]
    share_qparams_ops = [
        torch.nn.Flatten,
        F.adaptive_avg_pool2d,
        F.elu,
        F.hardtanh,
        F.max_pool2d,
        F.pad,
        F.relu,
        F.relu6,
        F.leaky_relu,
        F.leaky_relu_,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.ConstantPad2d,
        torch.nn.ELU,
        torch.nn.MaxPool2d,
        torch.nn.ReLU6,
        torch.nn.Hardtanh,
        torch.nn.LeakyReLU,
        torch.clamp,
        torch.flatten,
        torch.mean,
        torch.permute,
        torch.permute_copy,
        torch.squeeze,
        "clamp",
        "mean",
        "permute",
        "reshape",
        "relu",
        "relu_",
        "squeeze",
        "squeeze_",
        "leaky_relu",
    ]
    share_qparams_op_configs: list[BackendPatternConfig] = [
        BackendPatternConfig(op)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        for op in share_qparams_ops
    ]
    return share_qparams_op_configs


def _get_bn_configs() -> list[BackendPatternConfig]:
    """
    Return all configs related to batchnorm.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        executorch_default_op_quint8_dtype_config,
    ]
    bn_configs = []
    bn_configs.append(
        BackendPatternConfig(nn.BatchNorm2d)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
    )
    return bn_configs


def _get_cat_configs() -> list[BackendPatternConfig]:
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        executorch_default_op_quint8_dtype_config,
    ]
    cat_configs = []
    cat_configs.append(
        BackendPatternConfig(torch.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    cat_configs.append(
        BackendPatternConfig(torch.concat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    cat_configs.append(
        BackendPatternConfig(torch.concatenate)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    return cat_configs


def _get_embedding_op_configs() -> list[BackendPatternConfig]:
    dtype_configs = [
        executorch_weight_only_quint8_dtype_config,
    ]
    embedding_op_configs = []
    for embedding_op, qat_embedding_op, ref_embedding_op in [
        (nn.Embedding, nnqat.Embedding, nnqr.Embedding),
        (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag),
    ]:
        embedding_op_configs.append(
            BackendPatternConfig(embedding_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(dtype_configs)
            .set_qat_module(qat_embedding_op)
            .set_root_module(embedding_op)
            .set_reference_quantized_module(ref_embedding_op)
        )
        # config for qat op
        embedding_op_configs.append(
            BackendPatternConfig(qat_embedding_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(dtype_configs)
            .set_root_module(embedding_op)
            .set_reference_quantized_module(ref_embedding_op)
        )

        # config for functional embedding
        embedding_op_configs.append(
            BackendPatternConfig(torch.nn.functional.embedding)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1})
        )
    return embedding_op_configs


# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_executorch_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for backends PyTorch lowers to through the Executorch stack.
    """
    return (
        BackendConfig("executorch")
        .set_backend_pattern_configs(_get_linear_configs())
        .set_backend_pattern_configs(_get_conv_configs())
        .set_backend_pattern_configs(_get_binary_ops_configs())
        .set_backend_pattern_configs(_get_share_qparams_ops_configs())
        .set_backend_pattern_configs(_get_bn_configs())
        .set_backend_pattern_configs(_get_cat_configs())
        .set_backend_pattern_configs(_get_embedding_op_configs())
    )
<|endoftext|># mypy: allow-untyped-defs
import operator

import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)


weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)


def get_linear_configs():
    linear_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    # TODO: need to fix the way we insert observers for this pattern
    # should be solved in the new fusion API
    # reason that this doesn't work: the pattern is a bit complicated and we don't
    # have a way to specify which input of the pattern we would like to observe
    # pattern:
    # bias input weight
    # \     |    /
    #  \    |   t
    #   \   |  /
    #    addmm
    # we want to observe "weight" as weight, but there is not way to convey this
    # information with current pattern language
    #
    # right now:
    # original:
    #         weight - t \
    #         input  - addmm
    # observed (no hack):
    #      weight - t - observer \
    #       input - observer - addmm
    # target:
    #      weight - observer - t \
    #        input - observer - addmm

    # def root_node_getter(node_pattern):
    #     addmm, bias, act, weight = node_pattern
    #     return addmm

    # linear_configs.append(
    #     BackendPatternConfig((torch.ops.aten.addmm.default, MatchAllNode, MatchAllNode, torch.ops.aten.t.default))
    #     .set_observation_type(observation_type)
    #     .set_dtype_configs(dtype_configs)
    #     ._set_root_node_getter(root_node_getter))

    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.addmm.default)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 0})
    )
    # linear is decomposed to `t - mm` if bias is not present
    linear_configs.append(
        BackendPatternConfig(torch.ops.aten.mm.default)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1})
    )
    return linear_configs


def get_conv_configs():
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    conv_configs.append(
        BackendPatternConfig(torch.ops.aten.convolution.default)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    conv_configs.append(
        BackendPatternConfig(
            (torch.ops.aten.convolution.default, torch.ops.aten.relu.default)
        )
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    # TODO: remove when functionalization is supported in PT2 mode
    conv_configs.append(
        BackendPatternConfig(
            (torch.ops.aten.convolution.default, torch.ops.aten.relu_.default)
        )
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    return conv_configs


def get_pooling_configs():
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    def root_node_getter(node_pattern):
        _getitem, maxpool, _index = node_pattern
        return maxpool

    backend_pattern_configs.append(
        BackendPatternConfig()
        ._set_pattern_complex_format(
            (operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0)
        )
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_root_node_getter(root_node_getter)
    )

    return backend_pattern_configs


def get_relu_configs():
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    backend_pattern_configs.append(
        BackendPatternConfig(torch.ops.aten.relu.default)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
    )
    return backend_pattern_configs


def get_binary_op_configs():
    binary_op_configs: list[BackendPatternConfig] = []
    dtype_configs = [weighted_op_quint8_dtype_config]
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    for op_with_quantized_bop_scalar_variant in [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    ]:
        bop_patterns = [
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu.default),
            op_with_quantized_bop_scalar_variant,
            # TODO: remove when functionalization is supported in pt2_mode
            (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu_.default),
        ]
        binary_op_configs.extend(
            BackendPatternConfig(bop_pattern)
            .set_dtype_configs(dtype_configs)
            ._set_num_tensor_args_to_observation_type(
                num_tensor_args_to_observation_type_mapping
            )
            for bop_pattern in bop_patterns
        )

    return binary_op_configs


def get_qnnpack_pt2e_backend_config():
    return (
        BackendConfig("qnnpack_pytorch_2.0_export")
        .set_backend_pattern_configs(get_linear_configs())
        .set_backend_pattern_configs(get_binary_op_configs())
        .set_backend_pattern_configs(get_conv_configs())
        .set_backend_pattern_configs(get_pooling_configs())
        .set_backend_pattern_configs(get_relu_configs())
    )
<|endoftext|># mypy: allow-untyped-defs
import torch

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_conv_configs,
    _get_linear_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)


__all__ = [
    "get_tensorrt_backend_config",
    "get_tensorrt_backend_config_dict",
]


def get_tensorrt_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for the TensorRT backend.
    NOTE: Current api will change in the future, it's just to unblock experimentation for
    new backends, please don't use it right now.
    TODO: add a README when it's more stable
    """
    # dtype configs
    weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    non_weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
    )

    addmm_config = (
        BackendPatternConfig(torch.addmm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .add_dtype_config(weighted_op_qint8_dtype_config)
        ._set_input_type_to_index(
            {
                "bias": 0,
                "input": 1,
                "weight": 2,
            }
        )
    )
    cat_config = (
        BackendPatternConfig(torch.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .add_dtype_config(non_weighted_op_qint8_dtype_config)
    )
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    tensor_info_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    # there might be things not supported in fx2trt, but it will error out
    # during fx2trt conversion and can support them after that
    return (
        BackendConfig("tensorrt")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_config(addmm_config)
        .set_backend_pattern_config(cat_config)
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
    )


def get_tensorrt_backend_config_dict():
    """
    Return the `BackendConfig` for the TensorRT backend in dictionary form.
    """
    return get_tensorrt_backend_config().to_dict()
<|endoftext|># mypy: allow-untyped-defs
import torch

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_ln_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
from .backend_config import BackendConfig, DTypeConfig


__all__ = [
    "get_test_only_legacy_native_backend_config",
    "default_op_quint8_dtype_config",
    "default_op_fp16_dtype_config",
    "default_dynamic_int8_dtype_config",
    "default_dynamic_float16_dtype_config",
    "input_output_only_quint8_dtype_config",
    "weight_only_quint8_dtype_config",
    "weight_only_quint4x2_dtype_config",
    "get_native_backend_config",
    "get_native_backend_config_dict",
    "get_test_only_legacy_native_backend_config_dict",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# weighted op int8 dtype config
# this is config for ops that has quantized weights, like linear, conv
weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    is_dynamic=True,
)

default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    is_dynamic=True,
)

# Needed for LayerNorm and f.layer_norm, since currently the kernel only supports float weights
input_output_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.float,
    bias_dtype=torch.float,
)

weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_test_only_legacy_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional fp16 ops.
    """
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    linear_dtype_configs = [
        weighted_op_quint8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
        default_op_fp16_dtype_config,
    ]
    binary_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    tensor_info_op_dtype_configs = [
        default_op_quint8_dtype_config,
    ]
    rnn_op_dtype_configs = [
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        weight_only_quint8_dtype_config,
        weight_only_quint4x2_dtype_config,
    ]
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    return (
        BackendConfig("_native_and_fp16")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )


def get_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack).
    """
    # TODO: express this BackendConfig as a union of the FBGEMM and QNNPACK BackendConfigs
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    linear_dtype_configs = [
        weighted_op_quint8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [default_op_quint8_dtype_config]
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [default_op_quint8_dtype_config]
    share_qparams_op_dtype_configs = [default_op_quint8_dtype_config]
    tensor_info_op_dtype_configs = [default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        weight_only_quint8_dtype_config,
        weight_only_quint4x2_dtype_config,
    ]
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    return (
        BackendConfig("native")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )


def get_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) in dictionary form.
    """
    return get_native_backend_config().to_dict()


def get_test_only_legacy_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional
    fp16 ops in dictionary form.
    """
    return get_test_only_legacy_native_backend_config().to_dict()
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fuser_method_mappings import _reverse2, _reverse3
from torch.ao.quantization.utils import Pattern

from .backend_config import BackendConfig, BackendPatternConfig, DTypeConfig


__all__ = [
    "get_pattern_to_dtype_configs",
    "get_qat_module_classes",
    "get_fused_module_classes",
    "get_pattern_to_input_type_to_index",
    "get_root_module_to_quantized_reference_module",
    "get_fuser_method_mapping",
    "get_module_to_qat_module",
    "get_fusion_pattern_to_root_node_getter",
    "get_fusion_pattern_to_extra_inputs_getter",
    "remove_boolean_dispatch_from_name",
    "pattern_to_human_readable",
    "entry_to_pretty_str",
]


def get_pattern_to_dtype_configs(
    backend_config: BackendConfig,
) -> dict[Pattern, list[DTypeConfig]]:
    pattern_to_dtype_configs: dict[Pattern, list[DTypeConfig]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        pattern_to_dtype_configs[pattern] = config.dtype_configs
    return pattern_to_dtype_configs


def get_qat_module_classes(backend_config: BackendConfig) -> tuple[type, ...]:
    qat_module_classes = [
        config.qat_module
        for config in backend_config.configs
        if config.qat_module is not None
    ]
    return tuple(set(qat_module_classes))


def get_fused_module_classes(backend_config: BackendConfig) -> tuple[type, ...]:
    fused_module_classes = [
        config.fused_module
        for config in backend_config.configs
        if config.fused_module is not None
    ]
    return tuple(set(fused_module_classes))


def get_pattern_to_input_type_to_index(
    backend_config: BackendConfig,
) -> dict[Pattern, dict[str, int]]:
    pattern_to_input_type_to_index: dict[Pattern, dict[str, int]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        pattern_to_input_type_to_index[pattern] = config._input_type_to_index
    return pattern_to_input_type_to_index


def get_root_module_to_quantized_reference_module(
    backend_config: BackendConfig,
) -> dict[type[torch.nn.Module], type[torch.nn.Module]]:
    mapping: dict[type[torch.nn.Module], type[torch.nn.Module]] = {}
    for config in backend_config.configs:
        if (
            config.root_module is not None
            and config.reference_quantized_module is not None
        ):
            mapping[config.root_module] = config.reference_quantized_module
    return mapping


def get_fuser_method_mapping(
    backend_config: BackendConfig,
) -> dict[Pattern, nn.Sequential | Callable]:
    fuser_method_mapping: dict[Pattern, nn.Sequential | Callable] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.fuser_method is not None:
            # Note: both the fuser method and the pattern are specified in forward order in the
            # BackendConfig, but the internal pattern matching code uses the reversed nested tuple
            # format, so we need to convert both to the internal format
            fuser_method = _get_fuser_method_in_reversed_nested_tuple_format(config)
            fuser_method_mapping[pattern] = fuser_method
    return fuser_method_mapping


def get_module_to_qat_module(
    backend_config: BackendConfig,
) -> dict[Pattern, type[torch.nn.Module]]:
    module_to_qat_module: dict[Pattern, type[torch.nn.Module]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.qat_module is not None:
            module_to_qat_module[pattern] = config.qat_module
    return module_to_qat_module


def get_fusion_pattern_to_root_node_getter(
    backend_config: BackendConfig,
) -> dict[Pattern, Callable]:
    """Get a map from fusion pattern to a function that returns the root node
    from the fusion pattern, e.g. the most common one is::

        def get_root_node(node_pattern):
            while not isinstance(node_pattern[-1], Node):
                node_pattern = node_pattern[-1]
            return node_pattern[-1]

    This can work for all patterns whose root node is the "last node" in the pattern,
    e.g. ``(torch.add, MatchAllNode, (torch.ReLU, torch.Conv2d))``.
    """
    root_node_getter_mapping: dict[Pattern, Callable] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config._root_node_getter is not None:
            root_node_getter_mapping[pattern] = config._root_node_getter
    return root_node_getter_mapping


def get_fusion_pattern_to_extra_inputs_getter(
    backend_config: BackendConfig,
) -> dict[Pattern, Callable]:
    """Get a map from fusion pattern to a function that returns extra input nodes
    from the fusion pattern, in the order required by the root node. This is optional,
    if not specified, we will not copy over any extra inputs for the root node.

    Example::

        # Let's say we have the pattern (torch.add, MatchAllNode, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
        # and root node is torch.nn.Conv2d, and the node in MatchAllNode would be an extra
        # argument to the fused module, we can unpack the pattern and return the node at
        # MatchAllNode here
        # we can implement extra_inputs_getter as follows:
        def extra_inputs_getter(pattern) -> List[Any]:
            add, extra_input, conv_pattern = pattern
            return [extra_input]
    """
    extra_inputs_getter_mapping: dict[Pattern, Callable] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config._extra_inputs_getter is not None:
            extra_inputs_getter_mapping[pattern] = config._extra_inputs_getter
    return extra_inputs_getter_mapping


def remove_boolean_dispatch_from_name(p) -> Any:
    """
    Some ops have a default string representation such as
    '<function boolean_dispatch.<locals>.fn at 0x7ff1106bf280>',
    this function replaces them with the hardcoded function names.
    """
    if p is F.fractional_max_pool2d:
        return "torch.nn.functional.fractional_max_pool2d"
    elif p is F.fractional_max_pool3d:
        return "torch.nn.functional.fractional_max_pool3d"
    elif p is F.max_pool1d:
        return "torch.nn.functional.max_pool1d"
    elif p is F.max_pool2d:
        return "torch.nn.functional.max_pool2d"
    elif p is F.max_pool3d:
        return "torch.nn.functional.max_pool3d"
    elif p is F.adaptive_max_pool1d:
        return "torch.nn.functional.adaptive_max_pool1d"
    elif p is F.adaptive_max_pool2d:
        return "torch.nn.functional.adaptive_max_pool2d"
    elif p is F.adaptive_max_pool3d:
        return "torch.nn.functional.adaptive_max_pool3d"
    if "boolean_dispatch" in str(p):
        raise AssertionError(
            f"{p} does not have a human readable representation in "
            + "quantization documentation"
        )
    return p


def pattern_to_human_readable(p) -> Any:
    if isinstance(p, tuple):
        # nested patterns, recurse
        return tuple(pattern_to_human_readable(inner_p) for inner_p in p)
    elif isinstance(p, str):
        # method names are already human readable
        return p
    else:
        p = remove_boolean_dispatch_from_name(p)
        return p


# TODO(future PR): move backend_config_dict to use dataclass and move this logic to
# the corresponding __str__ function
def entry_to_pretty_str(entry) -> str:
    """
    Given a backend_config_dict entry, returns a string with the human readable
    representation of it.
    """
    s = "{\n"

    # always output the pattern first
    if "pattern" in entry:
        pattern_str = pattern_to_human_readable(entry["pattern"])

        s += f"  'pattern': {pattern_str},\n"

    # custom output for dtype_configs to make it look nice
    if "dtype_configs" in entry:
        s += "  'dtype_configs': [\n"
        for dtype_config in entry["dtype_configs"]:
            s += "    {\n"
            for k, v in dtype_config.items():
                s += f"      '{k}': {v},\n"
            s += "    },\n"
        s += "  ],\n"

    # custom output for num_tensor_args_to_observation_type to make it look nice
    if "num_tensor_args_to_observation_type" in entry:
        s += "  'num_tensor_args_to_observation_type': {\n"
        for k, v in entry["num_tensor_args_to_observation_type"].items():
            s += f"    {k}: {v},\n"
        s += "  },\n"

    # output all the other fields
    custom_handled_fields = [
        "pattern",
        "dtype_configs",
        "num_tensor_args_to_observation_type",
    ]
    for field_name in entry:
        if field_name in custom_handled_fields:
            continue
        s += f"  '{field_name}': {entry[field_name]},\n"

    s += "}"
    return s


def _get_pattern_in_reversed_nested_tuple_format(
    config: BackendPatternConfig,
) -> Pattern:
    """
    Return the pattern specified in the given config in the reversed nested tuple format
    used internally in the quantization pattern matching code.

    If the pattern is not a tuple, or the pattern is already specified in the reversed
    nested tuple format, return the pattern as is. Otherwise:

    For 2-tuples (a, b), return (b, a).
    For 3-tuples (a, b, c), return (c, (b, a)).

    For example:
        * Given nn.Linear, return nn.Linear
        * Given (nn.Linear, nn.ReLU), return (nn.ReLU, nn.Linear)
        * Given (nn.Conv2d, nn.BatchNorm2d, nn.ReLU), return
          (nn.ReLU, (nn.BatchNorm2d, nn.Conv2d))

    For context, the reason why this is needed is the user-facing BackendConfig
    API accepts the flat 2-or-3-tuple format in forward order. While this simple
    format handles the vast majority of use cases, it does not handle the more
    complex ones, and so the internal pattern matching code for quantization uses
    the following, more general reversed nested tuple format instead:

        operator = module_type | functional | torch op | native op | MatchAllNode
        Pattern = (operator, Pattern, Pattern, ...) | operator

    In the future, we expect to replace the above complex format with the one used
    by the subgraph rewriter in torch.fx, so we don't have to maintain our own
    complex pattern matching code. Then we won't need this helper function anymore.
    """
    if config._pattern_complex_format is not None:
        return config._pattern_complex_format
    if config.pattern is None:
        raise ValueError(
            "Either 'pattern' or 'pattern_complex_format' must be specified"
        )
    if not isinstance(config.pattern, tuple):
        return config.pattern

    # Pattern is specified in the simple tuple format, need to convert
    if len(config.pattern) == 2:
        (a, b) = config.pattern
        return (b, a)
    elif len(config.pattern) == 3:
        (a, b, c) = config.pattern
        return (c, (b, a))
    else:
        raise ValueError(
            f"Expected a tuple with 2 or 3 elements, got: {config.pattern}"
        )


def _get_fuser_method_in_reversed_nested_tuple_format(
    config: BackendPatternConfig,
) -> Callable:
    """
    Return the fuser method specified in the given config in the reversed nested
    tuple format used internally in the quantization pattern matching code.

    If pattern is specified in the reversed nested tuple format, we assume the
    fuser method is also specified in this format and simply return it as is.
    Otherwise, we convert the fuser method as follows:

        * Given f(is_qat, conv, relu), return f'(is_qat, relu, conv)
        * Given f(is_qat, conv, bn, relu), return f'(is_qat, relu, bn_conv),
          where bn_conv is a 2-tuple (bn, conv)

    The first argument of a fuser method is always `is_qat` and is not affected
    in the conversion. We currently only support functions with 3 or 4 arguments.
    """
    if config.fuser_method is None:
        raise AssertionError("config.fuser_method must be provided")
    if config._pattern_complex_format is not None:
        return config.fuser_method
    if not isinstance(config.pattern, tuple):
        raise ValueError(f"Expected pattern to be a tuple, got: {config.pattern}")

    # Pattern is specified in the simple tuple format, need to convert
    if len(config.pattern) == 2:
        return _reverse2(config.fuser_method)
    elif len(config.pattern) == 3:
        return _reverse3(config.fuser_method)
    else:
        raise ValueError(
            f"Expected a tuple with 2 or 3 elements, got: {config.pattern}"
        )
<|endoftext|>from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from .executorch import get_executorch_backend_config
from .fbgemm import get_fbgemm_backend_config
from .native import get_native_backend_config, get_native_backend_config_dict
from .onednn import get_onednn_backend_config
from .qnnpack import get_qnnpack_backend_config
from .tensorrt import get_tensorrt_backend_config, get_tensorrt_backend_config_dict


__all__ = [
    "get_fbgemm_backend_config",
    "get_native_backend_config",
    "get_native_backend_config_dict",
    "get_qnnpack_backend_config",
    "get_tensorrt_backend_config",
    "get_tensorrt_backend_config_dict",
    "get_executorch_backend_config",
    "BackendConfig",
    "BackendPatternConfig",
    "DTypeConfig",
    "DTypeWithConstraints",
    "ObservationType",
    "get_onednn_backend_config",
]
<|endoftext|># mypy: allow-untyped-defs
import itertools
import operator

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fuser_method_mappings import _sequential_wrapper2
from torch.ao.quantization.utils import MatchAllNode

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_ln_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
)
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)


# ===================
# |  DTYPE CONFIGS  |
# ===================

onednn_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

onednn_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

onednn_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

onednn_weight_only_qint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
)

onednn_input_output_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.float,
    bias_dtype=torch.float,
)

# ===================
# |  FUSER METHODS  |
# ===================


def _fuse_linear_bn_leaky_relu(is_qat, linear, bn, leaky_relu):
    r"""Given the linear, bn and leaky_relu modules, fuses them and returns the fused module
    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
                or post training quantization fusion
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer
        leaky_relu: LeakyReLU instance that needs to be fused with the linear layer
    Examples::
        >>> # xdoctest: +SKIP(failing)
        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> lr = nn.LeakyReLU(0.01)
        >>> m2 = _fuse_linear_bn_leaky_relu(m1, b1, lr)
    """
    if linear.training != bn.training or bn.training != leaky_relu.training:
        raise AssertionError(
            "Linear, BN and LeakyReLU all must be in the same mode (train or eval)."
        )

    if is_qat:
        raise NotImplementedError(
            f"Cannot fuse train modules: {(linear, bn, leaky_relu)}"
        )
    else:
        map_to_fused_module_eval = {
            nn.Linear: nni.LinearLeakyReLU,
        }
        fused_module = map_to_fused_module_eval.get(type(linear))
        if fused_module is not None:
            fused_linear = nn.utils.fusion.fuse_linear_bn_eval(linear, bn)
            fm = fused_module(fused_linear, leaky_relu)
            return fm
        else:
            raise NotImplementedError(
                f"Cannot fuse eval modules: {(linear, bn, leaky_relu)}"
            )


# ======================
# |  CONFIGS FOR CONV  |
# ======================
observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT

conv_dtype_configs = [onednn_weighted_op_int8_dtype_config]
conv_configs = _get_conv_configs(conv_dtype_configs)

# (1) Conv2d + Add

# conv2d   Y
#   \   /
#    add

# include:
# conv2d conv2d
#   \   /
#    add


def _fuse_conv_add_left(is_qat, add, conv, _):
    return nni.ConvAdd2d(conv, add)


def _conv_add_root_node_getter_left(pattern):
    _, conv, _ = pattern
    return conv


def _conv_add_extra_inputs_getter_left(pattern):
    """get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _, _conv, extra_input = pattern
    return [extra_input]


# conv2d
#  \
#  bn   Y
#   \   /
#    add


def _fuse_conv_bn_add_left(is_qat, add, bn_conv, _):
    bn, conv = bn_conv
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, add)}")
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAdd2d(fused_conv, add)


def _conv_bn_add_root_node_getter_left(add_pattern):
    _, bn_conv, _ = add_pattern
    _bn, conv = bn_conv
    return conv


def _conv_bn_add_extra_inputs_getter_left(add_pattern):
    """get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _, _bn_conv, extra_input = add_pattern
    return [extra_input]


conv_add_left_optioins = itertools.product(
    [True, False],  # with_bn
    [torch.add, operator.add],  # add_op
)

for with_bn, add_op in conv_add_left_optioins:
    if with_bn:
        conv_configs.append(
            BackendPatternConfig()
            ._set_pattern_complex_format(
                (add_op, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode)
            )
            .set_observation_type(observation_type)
            .set_dtype_configs(conv_dtype_configs)
            .set_fuser_method(_fuse_conv_bn_add_left)
            ._set_root_node_getter(_conv_bn_add_root_node_getter_left)
            ._set_extra_inputs_getter(_conv_bn_add_extra_inputs_getter_left)
            .set_fused_module(nni.ConvAdd2d)
        )
    else:
        conv_configs.append(
            BackendPatternConfig()
            ._set_pattern_complex_format((add_op, nn.Conv2d, MatchAllNode))
            .set_observation_type(observation_type)
            .set_dtype_configs(conv_dtype_configs)
            .set_fuser_method(_fuse_conv_add_left)
            ._set_root_node_getter(_conv_add_root_node_getter_left)
            ._set_extra_inputs_getter(_conv_add_extra_inputs_getter_left)
            .set_fused_module(nni.ConvAdd2d)
        )

#  Y   conv2d
#   \   /
#    add


def _fuse_conv_add_right(is_qat, add, _, conv):
    return nni.ConvAdd2d(conv, add)


def _conv_add_root_node_getter_right(pattern):
    _add, _, conv = pattern
    return conv


def _conv_add_extra_inputs_getter_right(pattern):
    """get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _, extra_input, _conv = pattern
    return [extra_input]


#      conv2d
#        /
#  Y    bn
#   \   /
#    add


def _fuse_conv_bn_add_right(is_qat, add, _, bn_conv):
    bn, conv = bn_conv
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, add)}")
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAdd2d(fused_conv, add)


def _conv_bn_add_root_node_getter_right(pattern):
    _add, _, bn_conv = pattern
    _bn, conv = bn_conv
    return conv


def _conv_bn_add_extra_inputs_getter_right(pattern):
    """get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _, extra_input, _bn_conv = pattern
    return [extra_input]


conv_add_optioins = itertools.product(
    [True, False],  # with_bn
    [torch.add, operator.add],  # add_op
)

for with_bn, add_op in conv_add_optioins:
    if with_bn:
        conv_configs.append(
            BackendPatternConfig()
            ._set_pattern_complex_format(
                (add_op, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d))
            )
            .set_observation_type(observation_type)
            .set_dtype_configs(conv_dtype_configs)
            .set_fuser_method(_fuse_conv_bn_add_right)
            ._set_root_node_getter(_conv_bn_add_root_node_getter_right)
            ._set_extra_inputs_getter(_conv_bn_add_extra_inputs_getter_right)
            .set_fused_module(nni.ConvAdd2d)
        )
    else:
        conv_configs.append(
            BackendPatternConfig()
            ._set_pattern_complex_format((add_op, MatchAllNode, nn.Conv2d))
            .set_observation_type(observation_type)
            .set_dtype_configs(conv_dtype_configs)
            .set_fuser_method(_fuse_conv_add_right)
            ._set_root_node_getter(_conv_add_root_node_getter_right)
            ._set_extra_inputs_getter(_conv_add_extra_inputs_getter_right)
            .set_fused_module(nni.ConvAdd2d)
        )

conv_configs.append(
    BackendPatternConfig(nni.ConvAdd2d)
    .set_observation_type(observation_type)
    .set_dtype_configs(conv_dtype_configs)
    .set_root_module(nn.Conv2d)
    .set_reference_quantized_module(nnqr.Conv2d)
)

# (2) Conv2d + Add + Relu

# conv2d Y
#   \   /
#    add
#     \
#     relu


def _fuse_conv_add_relu_left(is_qat, relu, add_pattern):
    add, conv, _ = add_pattern
    return nni.ConvAddReLU2d(conv, add, relu)


def _conv_add_relu_root_node_getter_left(pattern):
    _relu, add_pattern = pattern
    _, conv, _ = add_pattern
    return conv


def _conv_add_relu_extra_inputs_getter_left(pattern):
    """get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _relu, add_pattern = pattern
    _, _conv, extra_input = add_pattern
    return [extra_input]


# conv2d
#  \
#  bn   Y
#   \   /
#    add
#     \
#     relu


def _fuse_conv_bn_add_relu_left(is_qat, relu, add_pattern):
    add, bn_conv, _ = add_pattern
    bn, conv = bn_conv
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, add, relu)}")
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAddReLU2d(fused_conv, add, relu)


def _conv_bn_add_relu_root_node_getter_left(pattern):
    _relu, add_pattern = pattern
    _, bn_conv, _ = add_pattern
    _bn, conv = bn_conv
    return conv


def _conv_bn_add_relu_extra_inputs_getter_left(pattern):
    """get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _relu, add_pattern = pattern
    _, _bn_conv, extra_input = add_pattern
    return [extra_input]


conv_add_relu_left_optioins = itertools.product(
    [True, False],  # with_bn
    [torch.add, operator.add],  # add_op
)

for with_bn, add_op in conv_add_relu_left_optioins:
    if with_bn:
        conv_configs.append(
            BackendPatternConfig()
            ._set_pattern_complex_format(
                (nn.ReLU, (add_op, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))
            )
            .set_observation_type(observation_type)
            .set_dtype_configs(conv_dtype_configs)
            .set_fuser_method(_fuse_conv_bn_add_relu_left)
            ._set_root_node_getter(_conv_bn_add_relu_root_node_getter_left)
            ._set_extra_inputs_getter(_conv_bn_add_relu_extra_inputs_getter_left)
            .set_fused_module(nni.ConvAddReLU2d)
        )
    else:
        conv_configs.append(
            BackendPatternConfig()
            ._set_pattern_complex_format((nn.ReLU, (add_op, nn.Conv2d, MatchAllNode)))
            .set_observation_type(observation_type)
            .set_dtype_configs(conv_dtype_configs)
            .set_fuser_method(_fuse_conv_add_relu_left)
            ._set_root_node_getter(_conv_add_relu_root_node_getter_left)
            ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_left)
            .set_fused_module(nni.ConvAddReLU2d)
        )

#  Y   conv2d
#   \   /
#    add
#     \
#     relu


def _fuse_conv_add_relu_right(is_qat, relu, add_pattern):
    add, _, conv = add_pattern
    return nni.ConvAddReLU2d(conv, add, relu)


def _conv_add_relu_root_node_getter_right(pattern):
    _relu, add_pattern = pattern
    _, _extra_input, conv = add_pattern
    return conv


def _conv_add_relu_extra_inputs_getter_right(pattern):
    """get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _relu, add_pattern = pattern
    _, extra_input, _conv = add_pattern
    return [extra_input]


#      conv2d
#        /
#  Y    bn
#   \   /
#    add
#     \
#     relu


def _fuse_conv_bn_add_relu_right(is_qat, relu, add_pattern):
    add, _, bn_conv = add_pattern
    bn, conv = bn_conv
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, add, relu)}")
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAddReLU2d(fused_conv, add, relu)


def _conv_bn_add_relu_root_node_getter_right(pattern):
    _relu, add_pattern = pattern
    _, _, bn_conv = add_pattern
    _bn, conv = bn_conv
    return conv


def _conv_bn_add_relu_extra_inputs_getter_right(pattern):
    """get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _relu, add_pattern = pattern
    _, extra_input, _bn_conv = add_pattern
    return [extra_input]


conv_add_relu_left_optioins = itertools.product(
    [True, False],  # with_bn
    [torch.add, operator.add],  # add_op
)

for with_bn, add_op in conv_add_relu_left_optioins:
    if with_bn:
        conv_configs.append(
            BackendPatternConfig()
            ._set_pattern_complex_format(
                (nn.ReLU, (add_op, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))
            )
            .set_observation_type(observation_type)
            .set_dtype_configs(conv_dtype_configs)
            .set_fuser_method(_fuse_conv_bn_add_relu_right)
            ._set_root_node_getter(_conv_bn_add_relu_root_node_getter_right)
            ._set_extra_inputs_getter(_conv_bn_add_relu_extra_inputs_getter_right)
            .set_fused_module(nni.ConvAddReLU2d)
        )
    else:
        conv_configs.append(
            BackendPatternConfig()
            ._set_pattern_complex_format((nn.ReLU, (add_op, MatchAllNode, nn.Conv2d)))
            .set_observation_type(observation_type)
            .set_dtype_configs(conv_dtype_configs)
            .set_fuser_method(_fuse_conv_add_relu_right)
            ._set_root_node_getter(_conv_add_relu_root_node_getter_right)
            ._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_right)
            .set_fused_module(nni.ConvAddReLU2d)
        )

conv_configs.append(
    BackendPatternConfig(nni.ConvAddReLU2d)
    .set_observation_type(observation_type)
    .set_dtype_configs(conv_dtype_configs)
    .set_root_module(nn.Conv2d)
    .set_reference_quantized_module(nnqr.Conv2d)
)

# ========================
# |  CONFIGS FOR LINEAR  |
# ========================

linear_dtype_configs = [
    onednn_weighted_op_int8_dtype_config,
    onednn_dynamic_int8_dtype_config,
]
linear_configs = _get_linear_configs(linear_dtype_configs)


def _add_eltwise_fusion_configs(
    configs,
    root_module,
    root_op,
    post_module,
    post_op,
    dtype_configs,
    fuser_method,
    fused_module,
    observation_type,
    ref_quant_module,
):
    # 1 base module + op module fusion config
    configs.append(
        BackendPatternConfig((root_module, post_module))
        .set_dtype_configs(dtype_configs)
        .set_fuser_method(fuser_method)
        .set_fused_module(fused_module)
    )
    # base module + functional post op
    configs.append(
        BackendPatternConfig((root_module, post_op))
        .set_dtype_configs(dtype_configs)
        .set_fuser_method(fuser_method)
        .set_fused_module(fused_module)
    )

    # 2 fused module configs
    configs.append(
        BackendPatternConfig(fused_module)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(root_module)
        .set_reference_quantized_module(ref_quant_module)
    )

    # 3 functional base op + post op configs
    configs.append(
        BackendPatternConfig((root_op, post_module))
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
    )
    configs.append(
        BackendPatternConfig((root_op, post_op))
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
    )


# Configs for linear + leaky_relu fusion
_add_eltwise_fusion_configs(
    linear_configs,
    nn.Linear,
    F.linear,
    nn.LeakyReLU,
    F.leaky_relu,
    linear_dtype_configs,
    _sequential_wrapper2(nni.LinearLeakyReLU),
    nni.LinearLeakyReLU,
    observation_type,
    nnqr.Linear,
)

# Configs for linear module + batchnorm + leaky_relu
linear_configs.append(
    BackendPatternConfig((nn.Linear, nn.BatchNorm1d, nn.LeakyReLU))
    .set_dtype_configs(linear_dtype_configs)
    .set_fuser_method(_fuse_linear_bn_leaky_relu)
    .set_fused_module(nni.LinearLeakyReLU)
)

# Configs for linear + tanh fusion
_add_eltwise_fusion_configs(
    linear_configs,
    nn.Linear,
    F.linear,
    nn.Tanh,
    torch.tanh,
    linear_dtype_configs,
    _sequential_wrapper2(nni.LinearTanh),
    nni.LinearTanh,
    observation_type,
    nnqr.Linear,
)

# ===========================
# |  CONFIGS FOR OTHER OPS  |
# ===========================

binary_op_dtype_configs = [onednn_op_quint8_dtype_config]
default_op_dtype_configs = [onednn_op_quint8_dtype_config]
fixed_qparams_op_dtype_configs = [onednn_op_quint8_dtype_config]
share_qparams_op_dtype_configs = [onednn_op_quint8_dtype_config]
rnn_op_dtype_configs = [onednn_dynamic_int8_dtype_config]
embedding_op_dtype_configs = [onednn_weight_only_qint8_dtype_config]
layer_norm_op_dtype_configs = [onednn_input_output_only_quint8_dtype_config]

# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_onednn_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native ONEDNN backend.
    """
    return (
        BackendConfig("onednn")
        .set_backend_pattern_configs(conv_configs)
        .set_backend_pattern_configs(linear_configs)
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )


__all__ = [
    "get_onednn_backend_config",
]
<|endoftext|># mypy: allow-untyped-defs
import copy
import operator
from collections import namedtuple
from collections.abc import Callable

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fuser_method_mappings import (
    _sequential_wrapper2,
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_convtranspose_bn,
    fuse_linear_bn,
)

from .backend_config import (
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)


__all__: list[str] = []

# TODO: rename to be more explicit, e.g. qat_conv_relu
_ConvMetadata = namedtuple(
    "_ConvMetadata",
    [
        "root",
        "transpose",
        "bn",
        "reference",
        "transpose_reference",
        "fused_conv_relu",
        "fused_conv_bn",
        "fused_conv_bn_relu",
        "qat",
        "relu_qat",
        "bn_qat",
        "bn_relu_qat",
        "func",
        "func_transpose",
    ],
)
_Conv1dMetadata = _ConvMetadata(
    nn.Conv1d,
    nn.ConvTranspose1d,
    nn.BatchNorm1d,
    nnqr.Conv1d,
    nnqr.ConvTranspose1d,
    nni.ConvReLU1d,
    nni.ConvBn1d,
    nni.ConvBnReLU1d,
    nnqat.Conv1d,
    nniqat.ConvReLU1d,
    nniqat.ConvBn1d,
    nniqat.ConvBnReLU1d,
    F.conv1d,
    F.conv_transpose1d,
)
_Conv2dMetadata = _ConvMetadata(
    nn.Conv2d,
    nn.ConvTranspose2d,
    nn.BatchNorm2d,
    nnqr.Conv2d,
    nnqr.ConvTranspose2d,
    nni.ConvReLU2d,
    nni.ConvBn2d,
    nni.ConvBnReLU2d,
    nnqat.Conv2d,
    nniqat.ConvReLU2d,
    nniqat.ConvBn2d,
    nniqat.ConvBnReLU2d,
    F.conv2d,
    F.conv_transpose2d,
)
_Conv3dMetadata = _ConvMetadata(
    nn.Conv3d,
    nn.ConvTranspose3d,
    nn.BatchNorm3d,
    nnqr.Conv3d,
    nnqr.ConvTranspose3d,
    nni.ConvReLU3d,
    nni.ConvBn3d,
    nni.ConvBnReLU3d,
    nnqat.Conv3d,
    nniqat.ConvReLU3d,
    nniqat.ConvBn3d,
    nniqat.ConvBnReLU3d,
    F.conv3d,
    F.conv_transpose3d,
)

# Add constraints for fixed qparams ops like sigmoid and tanh to ensure values
# fall within the proper ranges, e.g. [0, 1] for sigmoid, [-1, 1] for tanh
_FIXED_QPARAM_OP_0TO1_CONSTRAINTS = DTypeWithConstraints(
    dtype=torch.quint8,
    quant_min_lower_bound=0,
    quant_max_upper_bound=255,
    scale_exact_match=1.0 / 256.0,
    zero_point_exact_match=0,
)
_FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS = DTypeWithConstraints(
    dtype=torch.quint8,
    quant_min_lower_bound=0,
    quant_max_upper_bound=255,
    scale_exact_match=2.0 / 256.0,
    zero_point_exact_match=128,
)
_FIXED_QPARAMS_OP_TO_CONSTRAINTS: dict[Callable | str, DTypeWithConstraints] = {
    torch.nn.Hardsigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.nn.functional.hardsigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "hardsigmoid": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "hardsigmoid_": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.nn.Sigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.sigmoid: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "sigmoid": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    "sigmoid_": _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.nn.Softmax: _FIXED_QPARAM_OP_0TO1_CONSTRAINTS,
    torch.nn.Tanh: _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    torch.tanh: _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    "tanh": _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
    "tanh_": _FIXED_QPARAM_OP_NEG1TO1_CONSTRAINTS,
}


def _get_binary_op_configs(
    dtype_configs: list[DTypeConfig],
) -> list[BackendPatternConfig]:
    binary_op_configs: list[BackendPatternConfig] = []
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    for op_with_quantized_bop_scalar_variant in [
        operator.add,
        torch.add,
        operator.mul,
        torch.mul,
    ]:
        bop_patterns = [
            (op_with_quantized_bop_scalar_variant, nn.ReLU),
            (op_with_quantized_bop_scalar_variant, F.relu),
            (op_with_quantized_bop_scalar_variant, torch.relu),
            op_with_quantized_bop_scalar_variant,
        ]
        binary_op_configs.extend(
            BackendPatternConfig(bop_pattern)
            .set_dtype_configs(dtype_configs)
            ._set_num_tensor_args_to_observation_type(
                num_tensor_args_to_observation_type_mapping
            )
            for bop_pattern in bop_patterns
        )
    # matmul
    binary_op_configs.append(
        BackendPatternConfig(torch.matmul).set_dtype_configs(dtype_configs)
    )
    return binary_op_configs


def _get_linear_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    """
    Return all configs related to linear modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    linear_configs: list[BackendPatternConfig] = []

    # (1) Single linear modules/functions
    # -------------------------------------
    # linear module
    linear_configs.append(
        BackendPatternConfig(torch.nn.Linear)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
        .set_qat_module(nnqat.Linear)
    )
    # linear qat module
    linear_configs.append(
        BackendPatternConfig(nnqat.Linear)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
    )
    # functional linear
    linear_configs.append(
        BackendPatternConfig(torch.nn.functional.linear)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )

    # (2) Linear + relu
    # -------------------
    # 2.1 linear module + relu fusion config
    # linear relu, linear module + relu module
    linear_configs.append(
        BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
        .set_dtype_configs(dtype_configs)
        .set_fuser_method(_sequential_wrapper2(nni.LinearReLU))
        .set_fused_module(nni.LinearReLU)
    )
    # linear relu, linear module + functional relu
    linear_configs.append(
        BackendPatternConfig((torch.nn.Linear, torch.nn.functional.relu))
        .set_dtype_configs(dtype_configs)
        .set_fuser_method(_sequential_wrapper2(nni.LinearReLU))
        .set_fused_module(nni.LinearReLU)
    )

    # 2.2 linear module + relu, fused module configs
    # linear relu, fused module
    linear_configs.append(
        BackendPatternConfig(nni.LinearReLU)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
        .set_qat_module(nniqat.LinearReLU)
    )
    # linear relu, qat fused module
    linear_configs.append(
        BackendPatternConfig(nniqat.LinearReLU)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
    )
    # 2.3 functional linear + relu configs
    # linear relu, functional linear + relu module
    linear_configs.append(
        BackendPatternConfig((F.linear, torch.nn.ReLU))
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
    )
    # linear relu, functional linear + functional relu
    linear_configs.append(
        BackendPatternConfig((F.linear, F.relu))
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
    )

    # (3) Linear + batchnorm
    # ------------------------
    # 3.1 linear bn fusion
    linear_configs.append(
        BackendPatternConfig((nn.Linear, nn.BatchNorm1d))
        .set_dtype_configs(dtype_configs)
        .set_fuser_method(fuse_linear_bn)
        .set_fused_module(nni.LinearBn1d)
    )

    # 3.2 linear bn fused
    # linear bn, fused module
    linear_configs.append(
        BackendPatternConfig(nni.LinearBn1d)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
        .set_qat_module(nniqat.LinearBn1d)
    )
    # linear bn, qat fused module
    linear_configs.append(
        BackendPatternConfig(nniqat.LinearBn1d)
        .set_observation_type(observation_type)
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
    )
    return linear_configs


def _get_conv_configs(dtype_configs):
    """
    Return all configs related to conv modules and ops.
    """
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    for convs in [_Conv1dMetadata, _Conv2dMetadata, _Conv3dMetadata]:
        # (1) Single conv modules/functions
        # -----------------------------------
        # conv module
        conv_configs.append(
            BackendPatternConfig(convs.root)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
            .set_qat_module(convs.qat)
        )
        # conv qat module
        conv_configs.append(
            BackendPatternConfig(convs.qat)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # functional conv
        conv_configs.append(
            BackendPatternConfig(convs.func)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2})
        )

        # (2) Conv + relu
        # -----------------
        # 2.1 conv module + relu fusion configs
        # conv relu fusion, conv module + relu module
        conv_configs.append(
            BackendPatternConfig((convs.root, torch.nn.ReLU))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu))
            .set_fused_module(convs.fused_conv_relu)
        )
        # conv relu fusion, conv module + functional relu
        conv_configs.append(
            BackendPatternConfig((convs.root, F.relu))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu))
            .set_fused_module(convs.fused_conv_relu)
        )
        # 2.2 conv module + relu fused module configs
        # conv relu, fused module
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
            .set_qat_module(convs.relu_qat)
        )
        # conv relu, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.relu_qat)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # 2.3 functional conv + relu configs
        # conv relu, functional conv + relu module
        conv_configs.append(
            BackendPatternConfig((convs.func, torch.nn.ReLU))
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
        )
        # conv relu, functional conv + functional relu
        conv_configs.append(
            BackendPatternConfig((convs.func, F.relu))
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
        )

        # fused conv relu
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
            .set_dtype_configs(dtype_configs)
            .set_qat_module(convs.relu_qat)
        )

        conv_configs.append(
            BackendPatternConfig(convs.relu_qat)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )

        # (3) Conv + batchnorm (+ relu)
        # -------------------------------
        # 3.1 conv bn fusion configs
        # conv + bn fusion
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(fuse_conv_bn)
            .set_fused_module(convs.fused_conv_bn)
        )
        # conv + bn + relu module fusion
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn, nn.ReLU))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(fuse_conv_bn_relu)
            .set_fused_module(convs.fused_conv_bn_relu)
        )
        # conv + bn + relu functional fusion
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn, F.relu))
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_fuser_method(fuse_conv_bn_relu)
            .set_fused_module(convs.fused_conv_bn_relu)
        )
        # TODO: we can add fusion for torch.relu as well

        # 3.2 conv + bn (+ relu) fused module configs
        # fused conv bn
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_bn)
            .set_dtype_configs(dtype_configs)
            .set_qat_module(convs.bn_qat)
        )

        # fused conv bn relu
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_bn_relu)
            .set_dtype_configs(dtype_configs)
            .set_qat_module(convs.bn_relu_qat)
        )

        # conv bn, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.bn_qat)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # conv bn relu, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.bn_relu_qat)
            .set_observation_type(observation_type)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )

        # (4) conv transpose and its fusion
        # 4.1 conv transpose config
        conv_configs.append(
            BackendPatternConfig(convs.transpose)
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.transpose)
            .set_reference_quantized_module(convs.transpose_reference)
        )

        # 4.2 conv transpose + bn fusion
        conv_configs.append(
            BackendPatternConfig((convs.transpose, convs.bn))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(fuse_convtranspose_bn)
            .set_root_module(convs.transpose)
            .set_reference_quantized_module(convs.transpose_reference)
        )

        # 4.3 functional conv transpose
        conv_configs.append(
            BackendPatternConfig(convs.func_transpose)
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2})
        )

    return conv_configs


def _get_cat_config(dtype_configs: list[DTypeConfig]) -> BackendPatternConfig:
    return (
        BackendPatternConfig(torch.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(dtype_configs)
    )


def _get_ln_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    ln_configs = []
    ln_configs.append(
        BackendPatternConfig(torch.nn.LayerNorm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    ln_configs.append(
        BackendPatternConfig(torch.nn.functional.layer_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 3})
    )
    return ln_configs


def _get_default_op_configs(
    dtype_configs: list[DTypeConfig],
) -> list[BackendPatternConfig]:
    default_ops = [
        torch.nn.ELU,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.Dropout,
        torch.nn.PReLU,
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
    ]
    configs = [
        BackendPatternConfig(op)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        for op in default_ops
    ]

    configs.append(
        BackendPatternConfig(torch.nn.functional.group_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 3})
    )

    configs.append(
        BackendPatternConfig(torch.nn.functional.instance_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 3, "bias": 4})
    )
    return configs


def _add_fixed_qparams_to_dtype_configs(
    dtype_configs: list[DTypeConfig],
    constraints: DTypeWithConstraints,
) -> list[DTypeConfig]:
    """
    Return a copy of the list of DTypeConfigs where activations are subject to the specified
    constraints required for fixed qparams ops.

    If the data type doesn't match the one in the constraints, simply leave the corresponding
    DTypeConfig unchanged.

    If `scale_min_lower_bound` or `scale_max_upper_bound` is specified in the activations,
    throw an exception since these settings are incompatible with fixed qparams ops.
    """
    new_dtype_configs = []
    for dtype_config in dtype_configs:
        dc = copy.deepcopy(dtype_config)
        for orig_constraints in [
            dc.input_dtype_with_constraints,
            dc.output_dtype_with_constraints,
        ]:
            if orig_constraints.dtype != constraints.dtype:
                continue
            if orig_constraints.scale_min_lower_bound is not None:
                raise ValueError(
                    f"scale_min_lower_bound is invalid for fixed qparams ops: {dtype_config}"
                )
            if orig_constraints.scale_max_upper_bound is not None:
                raise ValueError(
                    f"scale_max_upper_bound is invalid for fixed qparams ops: {dtype_config}"
                )
            orig_constraints.quant_min_lower_bound = constraints.quant_min_lower_bound
            orig_constraints.quant_max_upper_bound = constraints.quant_max_upper_bound
            orig_constraints.scale_exact_match = constraints.scale_exact_match
            orig_constraints.zero_point_exact_match = constraints.zero_point_exact_match
        new_dtype_configs.append(dc)
    return new_dtype_configs


def _get_fixed_qparams_op_configs(
    dtype_configs: list[DTypeConfig],
) -> list[BackendPatternConfig]:
    fixed_qparams_op_configs = []
    for fixed_qparam_op, constraints in _FIXED_QPARAMS_OP_TO_CONSTRAINTS.items():
        new_dtype_configs = _add_fixed_qparams_to_dtype_configs(
            dtype_configs, constraints
        )
        fixed_qparams_op_configs.append(
            BackendPatternConfig(fixed_qparam_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(new_dtype_configs)
        )
    return fixed_qparams_op_configs


def _get_share_qparams_op_configs(dtype_configs):
    """Get the operator config for the operators that works for both float and quantized input
    if input is quantized, the output Tensor shares the same quantization parameter
    with input.
    Example operator: avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - avgpool2d - observer_0 (same observer instance as input)
    """

    def _get_share_qprams_op_backend_config(op):
        return (
            BackendPatternConfig(op)
            .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
            .set_dtype_configs(dtype_configs)
        )

    share_qparams_ops = [
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.Hardtanh,
        torch.nn.Identity,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.PixelShuffle,
        torch.nn.PixelUnshuffle,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hardtanh_,
        torch.nn.functional.interpolate,
        torch.nn.functional.max_pool1d,
        torch.nn.functional.max_pool2d,
        torch.nn.functional.max_pool3d,
        torch.nn.functional.pixel_shuffle,
        torch.nn.functional.pixel_unshuffle,
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.clamp,
        torch.flatten,
        torch.mean,
        torch.narrow,
        torch.repeat_interleave,
        torch.transpose,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        operator.floordiv,
        "contiguous",
        "clamp",
        "detach",
        "detach_",
        "mean",
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "resize_",
        "relu",
        "relu_",
        "squeeze",
        "squeeze_",
        "transpose",
        "unsqueeze",
        "unsqueeze_",
        "view",
    ]
    return [_get_share_qprams_op_backend_config(op) for op in share_qparams_ops]


def _get_bn_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    """Get configs related to batchnorm."""
    bn_configs = []
    bn_to_fused_bn = {
        torch.nn.BatchNorm2d: nni.BNReLU2d,
        torch.nn.BatchNorm3d: nni.BNReLU3d,
    }
    for bn in bn_to_fused_bn:
        fused_bn = bn_to_fused_bn[bn]
        # bn module + relu module fusion config
        bn_configs.append(
            BackendPatternConfig((bn, nn.ReLU))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(_sequential_wrapper2(fused_bn))
            .set_fused_module(fused_bn)
        )
        # bn module + F.relu fusion config
        bn_configs.append(
            BackendPatternConfig((bn, F.relu))
            .set_dtype_configs(dtype_configs)
            .set_fuser_method(_sequential_wrapper2(fused_bn))
            .set_fused_module(fused_bn)
        )
        bn_configs.append(
            BackendPatternConfig(bn)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(dtype_configs)
        )

    # fused bn configs
    for fused_bn in bn_to_fused_bn.values():
        bn_configs.append(
            BackendPatternConfig(fused_bn)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(dtype_configs)
        )
    return bn_configs


def _get_rnn_op_configs(dtype_configs: list[DTypeConfig]) -> list[BackendPatternConfig]:
    rnn_op_configs = []
    for rnn_op, ref_rnn_op in [
        (nn.GRUCell, nnqr.GRUCell),
        (nn.LSTMCell, nnqr.LSTMCell),
        (nn.RNNCell, nnqr.RNNCell),
        (nn.LSTM, nnqr.LSTM),
        (nn.GRU, nnqr.GRU),
    ]:
        rnn_op_configs.append(
            BackendPatternConfig(rnn_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(dtype_configs)
            .set_root_module(rnn_op)
            .set_reference_quantized_module(ref_rnn_op)
        )
    return rnn_op_configs


def _get_embedding_op_configs(
    dtype_configs: list[DTypeConfig],
) -> list[BackendPatternConfig]:
    embedding_op_configs = []
    for embedding_op, qat_embedding_op, ref_embedding_op in [
        (nn.Embedding, nnqat.Embedding, nnqr.Embedding),
        (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag),
    ]:
        embedding_op_configs.append(
            BackendPatternConfig(embedding_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(dtype_configs)
            .set_qat_module(qat_embedding_op)
            .set_root_module(embedding_op)
            .set_reference_quantized_module(ref_embedding_op)
        )

        # config for qat op
        embedding_op_configs.append(
            BackendPatternConfig(qat_embedding_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )
            .set_dtype_configs(dtype_configs)
            .set_root_module(embedding_op)
            .set_reference_quantized_module(ref_embedding_op)
        )
    return embedding_op_configs


def _get_tensor_info_op_configs(dtype_configs):
    """
    These ops work on tensors of different dtypes but return non-tensors
    containing information about the input tensor.
    """

    def _get_config(op):
        return (
            BackendPatternConfig(op)
            .set_observation_type(ObservationType.INPUT_OUTPUT_NOT_OBSERVED)
            .set_dtype_configs(dtype_configs)
        )

    return [_get_config(op) for op in ("shape", "size")]
<|endoftext|>import torch

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
from .backend_config import BackendConfig, DTypeConfig


__all__ = [
    "get_x86_backend_config",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# X86 aligns with FBGEMM for now

x86_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

x86_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

x86_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

x86_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

x86_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

x86_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

x86_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_x86_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native x86 backend.
    """
    conv_dtype_configs = [x86_weighted_op_int8_dtype_config]
    linear_dtype_configs = [
        x86_weighted_op_int8_dtype_config,
        x86_default_dynamic_int8_dtype_config,
        x86_default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [x86_weighted_op_int8_dtype_config]
    default_op_dtype_configs = [x86_default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [x86_weighted_op_int8_dtype_config]
    share_qparams_op_dtype_configs = [x86_default_op_quint8_dtype_config]
    tensor_info_op_dtype_configs = [x86_default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        x86_default_dynamic_int8_dtype_config,
        x86_default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        x86_weight_only_quint8_dtype_config,
        x86_weight_only_quint4x2_dtype_config,
    ]
    return (
        BackendConfig("x86")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )
<|endoftext|>import torch

from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)
from .backend_config import BackendConfig, DTypeConfig


__all__ = [
    "get_fbgemm_backend_config",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

# TODO: For now, these DTypeConfigs are identical to the ones defined in native.py
# In the future, once we support specifying quant_min/quant_max and scale_min/scale_max,
# these will diverge. In particular, for FBGEMM, we will restrict the activation quantized
# values to within [0, 127].

fbgemm_weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

fbgemm_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

fbgemm_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

fbgemm_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

fbgemm_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

fbgemm_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

fbgemm_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_fbgemm_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native FBGEMM backend.
    """
    conv_dtype_configs = [fbgemm_weighted_op_quint8_dtype_config]
    linear_dtype_configs = [
        fbgemm_weighted_op_quint8_dtype_config,
        fbgemm_default_dynamic_int8_dtype_config,
        fbgemm_default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    default_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    share_qparams_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    tensor_info_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        fbgemm_default_dynamic_int8_dtype_config,
        fbgemm_default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        fbgemm_weight_only_quint8_dtype_config,
        fbgemm_weight_only_quint4x2_dtype_config,
    ]
    return (
        BackendConfig("fbgemm")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_tensor_info_op_configs(tensor_info_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )
<|endoftext|># mypy: allow-untyped-defs
import warnings
from collections.abc import Callable
from typing import Any

from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
)
from torch.ao.quantization.backend_config.utils import (
    get_fuser_method_mapping,
    get_fusion_pattern_to_extra_inputs_getter,
    get_fusion_pattern_to_root_node_getter,
)
from torch.ao.quantization.utils import NodePattern, Pattern
from torch.fx import GraphModule, map_arg, Node
from torch.fx.graph import Graph

from .custom_config import FuseCustomConfig
from .fuse_handler import _get_fusion_pattern_to_fuse_handler_cls, FuseHandler
from .match_utils import _is_match, MatchAllNode
from .pattern_utils import _sorted_patterns_dict


__all__ = [
    "fuse",
    # TODO: We should make this private in the future
    # This is currently needed for test_public_bindings for some reason
    "FuseHandler",
]


def fuse(
    model: GraphModule,
    is_qat: bool,
    fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()

    if isinstance(fuse_custom_config, dict):
        warnings.warn(
            "Passing a fuse_custom_config_dict to fuse is deprecated and will not be supported "
            "in a future version. Please pass in a FuseCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)

    if isinstance(backend_config, dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        backend_config = BackendConfig.from_dict(backend_config)

    named_modules = dict(model.named_modules())

    if backend_config is None:
        backend_config = get_native_backend_config()

    fusion_pattern_to_fuse_handler_cls = _sorted_patterns_dict(
        _get_fusion_pattern_to_fuse_handler_cls(backend_config)
    )
    fuser_method_mapping = get_fuser_method_mapping(backend_config)
    fusion_pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(
        backend_config
    )
    fusion_pattern_to_extra_inputs_getter = get_fusion_pattern_to_extra_inputs_getter(
        backend_config
    )

    # find fusion
    fusion_pairs = _find_matches(model, model.graph, fusion_pattern_to_fuse_handler_cls)
    # TODO: change this to inplace changes to graph, since we no longer construct
    # new GraphModule anymore
    fused_graph = Graph()
    env: dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    def default_root_node_getter(node_pattern):
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]

    for node in model.graph.nodes:
        (
            maybe_last_node,
            pattern,
            matched_node_pattern,
            obj,
            node_to_subpattern,
        ) = fusion_pairs.get(node.name, (None, None, None, None, None))
        # get the corresponding subpattern for the current node
        if node_to_subpattern is not None:
            node_subpattern = node_to_subpattern.get(node, None)
        else:
            node_subpattern = None
        if maybe_last_node is node:
            if obj is None:
                raise AssertionError(
                    "fuse handler object must not be None for matched root node"
                )
            root_node_getter = fusion_pattern_to_root_node_getter.get(
                pattern, default_root_node_getter
            )
            root_node = root_node_getter(matched_node_pattern)  # type: ignore[index]
            extra_inputs_getter = fusion_pattern_to_extra_inputs_getter.get(
                pattern, None
            )
            extra_inputs = []
            if extra_inputs_getter is not None:
                extra_inputs = extra_inputs_getter(matched_node_pattern)
            # TODO: add validation that root_node is a module and has the same type
            # as the root_module in the configuration
            env[node.name] = obj.fuse(
                load_arg,
                named_modules,
                fused_graph,
                root_node,
                extra_inputs,
                matched_node_pattern,  # type: ignore[arg-type]
                fuse_custom_config,
                fuser_method_mapping,
                is_qat,
            )
        elif maybe_last_node is None or node_subpattern is MatchAllNode:
            env[node.name] = fused_graph.node_copy(node, load_arg)
        # node matched in patterns and is not root is removed here

    model = GraphModule(model, fused_graph)
    return model


def _find_matches(
    root: GraphModule,
    graph: Graph,
    pattern_to_fuse_handler_cls: dict[Pattern, Callable],
) -> dict[str, tuple[Node, Pattern, NodePattern, FuseHandler, dict[Node, Any]]]:
    modules = dict(root.named_modules())
    # node name -> (root_node, match_value)
    match_map: dict[
        str, tuple[Node, Pattern, NodePattern, FuseHandler, dict[Node, Any]]
    ] = {}
    # a map from node to the matched subpattern
    node_to_subpattern: dict[Node, Any] = {}

    # TODO: dedup with quantization matching function in match_utils.py
    def apply_match(pattern, node, match, matched_node_pattern, node_to_subpattern):
        if isinstance(pattern, tuple):
            s, *args = pattern
            current_node_pattern: list[Node] = []
            apply_match(s, node, match, current_node_pattern, node_to_subpattern)
            for subpattern, arg in zip(args, node.args):
                apply_match(
                    subpattern, arg, match, current_node_pattern, node_to_subpattern
                )
            matched_node_pattern.append(tuple(current_node_pattern))
        else:
            # the first pattern matches will take precedence
            if node.name not in match_map:
                matched_node_pattern.append(node)
                # MatchAllNode here is actually MatchAllInputNode which should not
                # be added to match_map
                if pattern is not MatchAllNode:
                    node_to_subpattern[node] = pattern
                    root_node, pattern, handler = match
                    match_map[node.name] = (
                        root_node,
                        pattern,
                        matched_node_pattern,
                        handler,
                        node_to_subpattern,
                    )

    for node in reversed(graph.nodes):
        if node.name not in match_map:
            for pattern, fuse_handler_cls in pattern_to_fuse_handler_cls.items():
                matched_node_pattern: list[Node] = []
                if _is_match(modules, node, pattern):
                    apply_match(
                        pattern,
                        node,
                        (node, pattern, fuse_handler_cls(node)),
                        matched_node_pattern,
                        node_to_subpattern,
                    )
                    break

    return match_map
<|endoftext|># mypy: allow-untyped-defs
import re
from collections import defaultdict, OrderedDict
from collections.abc import Callable
from typing import Any

import torch
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization import QConfig
from torch.ao.quantization.backend_config import BackendConfig, DTypeConfig
from torch.ao.quantization.backend_config.utils import get_module_to_qat_module
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig import (
    _add_module_to_qconfig_obs_ctr,
    qconfig_equals,
    QConfigAny,
)
from torch.ao.quantization.qconfig_mapping import (
    _MODULE_NAME_DICT_KEY,
    _MODULE_NAME_REGEX_DICT_KEY,
    _OBJECT_TYPE_DICT_KEY,
    QConfigMapping,
)
from torch.ao.quantization.utils import _parent_name, get_qconfig_dtypes
from torch.fx import GraphModule
from torch.fx.graph import Graph


__all__: list[str] = []


def _maybe_adjust_qconfig_for_module_name_object_type_order(
    qconfig_mapping: QConfigMapping,
    cur_module_path: str,
    cur_object_type: Callable,
    cur_object_type_idx: int,
    fallback_qconfig: QConfigAny,
) -> QConfigAny:
    for (
        module_name,
        object_type,
        index,
    ), qconfig in qconfig_mapping.module_name_object_type_order_qconfigs.items():
        if (
            (module_name == cur_module_path)
            and (object_type == cur_object_type)
            and (index == cur_object_type_idx)
        ):
            return qconfig
    return fallback_qconfig


def _update_qconfig_for_fusion(model: GraphModule, qconfig_mapping: QConfigMapping):
    """
    Update the QConfigMapping to account for fused modules such as LinearReLU.
    This assumes the QConfigMapping's attributes have already been converted to OrderedDicts.
    """
    object_type_dict = qconfig_mapping.object_type_qconfigs
    if len(object_type_dict) == 0:
        return qconfig_mapping

    modules = dict(model.named_modules())

    for node in model.graph.nodes:
        if node.op == "call_module" and node.target in modules:
            maybe_fused_module = modules[str(node.target)]
            if not isinstance(maybe_fused_module, _FusedModule):
                continue

            ops = list(maybe_fused_module._modules.values())
            fused_qconfig = object_type_dict.get(type(ops[0]), None)

            # Raise an error if the modules in the fused module have
            # different qconfigs specified in the qconfig_dict
            # TODO: currently it only works for modules,
            # need to make this work for torch.nn.functional.relu
            # TODO: currently it only works for object_type configurations,
            # ideally it should work for different types of configurations,
            # maybe we want to redesign this part
            for op in ops[1:]:
                if not qconfig_equals(
                    object_type_dict.get(type(op), None), fused_qconfig
                ):
                    raise LookupError(
                        "During fusion, we need to specify the same "
                        + f"qconfigs for all module types in {type(maybe_fused_module)} "
                        + f"offending type: {type(op)}"
                    )

            if fused_qconfig is not None:
                object_type_dict[type(maybe_fused_module)] = fused_qconfig


def _generate_node_name_to_qconfig(
    root: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    input_graph: Graph,
    qconfig_mapping: QConfigMapping,
    node_name_to_scope: dict[str, tuple[str, type]],
) -> dict[str, QConfigAny]:
    global_qconfig = qconfig_mapping.global_qconfig
    node_name_to_qconfig = {}

    # example:
    #
    #   {'foo.bar': {F.linear: 0, F.conv2d: 1, ...}, ...}
    #
    # meaning in submodule 'foo.bar', we have seen 0 F.linear and
    # 1 F.conv2d invocations so far.
    submodule_to_object_type_to_cur_idx: dict[str, dict[Callable, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for node in input_graph.nodes:
        qconfig = None
        if node.op == "get_attr":
            module_name, _ = _parent_name(node.target)
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, type(modules[module_name]), module_name, global_qconfig
            )
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(
                qconfig, modules.get(node.target)
            )
        elif node.op == "call_function":
            # precedence: module_name_qconfig
            # > function_qconfig > global_qconfig
            # module_name takes precedence over function qconfig
            function_qconfig = _get_object_type_qconfig(
                qconfig_mapping, node.target, global_qconfig
            )
            module_path, module_type = node_name_to_scope[node.name]
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, module_type, module_path, function_qconfig
            )

            cur_object_type_idx = submodule_to_object_type_to_cur_idx[module_path][
                node.target
            ]
            submodule_to_object_type_to_cur_idx[module_path][node.target] += 1
            qconfig = _maybe_adjust_qconfig_for_module_name_object_type_order(
                qconfig_mapping, module_path, node.target, cur_object_type_idx, qconfig
            )
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(
                qconfig, modules.get(node.target)
            )

        elif node.op == "call_method":
            module_path, module_type = node_name_to_scope[node.name]
            # first use node.target (string) to get the qconfig
            # this is to support configs like
            # "object_type": [("reshape", qconfig)]
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, node.target, module_path, global_qconfig
            )
            # if there is no special config for the method, we'll fall back to the
            # config for the module that contains the call_method node
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, module_type, module_path, qconfig
            )
            # currently call_method does not support modifying qconfig
            # by order, we can add this later if it is needed.
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(
                qconfig, modules.get(node.target)
            )

        elif node.op == "call_module":
            # if the node is an observer, just continue - don't add it to the qconfig_map
            if _is_activation_post_process(modules[node.target]):
                continue
            qconfig = _maybe_adjust_qconfig_for_module_type_or_name(
                qconfig_mapping, type(modules[node.target]), node.target, global_qconfig
            )

            module_path, module_type = node_name_to_scope[node.name]
            # Note: for call_module, the module_path is the current module's name.
            # to meaningfully count invocations, we need to count them in the parent
            # module.
            parent_name, _ = _parent_name(module_path)
            cur_object_type_idx = submodule_to_object_type_to_cur_idx[parent_name][
                module_type
            ]
            submodule_to_object_type_to_cur_idx[parent_name][module_type] += 1
            qconfig = _maybe_adjust_qconfig_for_module_name_object_type_order(
                qconfig_mapping, parent_name, module_type, cur_object_type_idx, qconfig
            )
            qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(
                qconfig, modules.get(node.target)
            )

            # regex is not supported eager mode propagate_qconfig_, we'll
            # need to set the qconfig explicitly here in case regex
            # is used
            modules[node.target].qconfig = qconfig_with_device_check
        else:
            qconfig_with_device_check = None

        node_name_to_qconfig[node.name] = qconfig_with_device_check
    return node_name_to_qconfig


def _check_is_valid_config_dict(
    config_dict: Any, allowed_keys: set[str], dict_name: str
) -> None:
    r"""Checks if the given config_dict has the correct keys

    Args:
      `config_dict`: dictionary whose keys we want to check
    """

    for k in config_dict:
        if k not in allowed_keys:
            raise ValueError(
                "Expected "
                + dict_name
                + " to have the following keys: "
                + str(allowed_keys)
                + ". But found '"
                + k
                + "' instead."
            )


def _compare_prepare_convert_qconfig_mappings(
    prepare_qconfig_mapping: QConfigMapping, convert_qconfig_mapping: QConfigMapping
):
    r"""Compare the qconfig_mapping passed in convert to the one from prepare and check the values

    Args:
      `prepare_qconfig_mapping`: configuration for prepare quantization step
      `convert_qconfig_mapping`: configuration for convert quantization step
    """
    if not qconfig_equals(
        prepare_qconfig_mapping.global_qconfig, convert_qconfig_mapping.global_qconfig
    ):
        raise AssertionError(
            "Expected global qconfigs to be the same in the prepare and convert quantization configs"
        )
    prepare_dicts: list[OrderedDict] = [
        prepare_qconfig_mapping.object_type_qconfigs,
        prepare_qconfig_mapping.module_name_qconfigs,
        prepare_qconfig_mapping.module_name_regex_qconfigs,
    ]
    convert_dicts: list[OrderedDict] = [
        convert_qconfig_mapping.object_type_qconfigs,
        convert_qconfig_mapping.module_name_qconfigs,
        convert_qconfig_mapping.module_name_regex_qconfigs,
    ]
    dict_names = [
        _OBJECT_TYPE_DICT_KEY,
        _MODULE_NAME_DICT_KEY,
        _MODULE_NAME_REGEX_DICT_KEY,
    ]
    for i in range(len(prepare_dicts)):
        for name in prepare_dicts[i]:
            if name not in convert_dicts[i]:
                raise AssertionError(
                    f"Missing key {dict_names[i]} {name} in convert QConfigMapping when it was present in prepare"
                )
            if convert_dicts[i][name] is not None and not qconfig_equals(
                prepare_dicts[i][name], convert_dicts[i][name]
            ):
                raise AssertionError(
                    "Expected convert QConfigMapping to have the same qconfig as prepare for key "
                    f"{dict_names[i]} {name}; prepare: {prepare_dicts[i][name]}; convert: {convert_dicts[i][name]}"
                )


def _is_qconfig_supported_by_dtype_configs(
    qconfig: QConfig, dtype_configs: list[DTypeConfig]
):
    for dtype_config in dtype_configs:
        is_dynamic = dtype_config.is_dynamic
        if is_dynamic is None:
            is_dynamic = False
        input_dtype = dtype_config.input_dtype or torch.float
        weight_dtype = dtype_config.weight_dtype or torch.float
        bias_dtype = dtype_config.bias_dtype or torch.float
        output_dtype = dtype_config.output_dtype or torch.float
        (
            qconfig_activation_dtype,
            qconfig_weight_dtype,
            qconfig_input_act_is_dynamic,
        ) = get_qconfig_dtypes(qconfig)
        qconfig_bias_dtype = (
            torch.float16
            if (
                qconfig_activation_dtype == torch.float16
                and qconfig_weight_dtype == torch.float16
                and not is_dynamic
            )
            else torch.float
        )

        if is_dynamic:
            is_match = (
                qconfig_input_act_is_dynamic
                and input_dtype == qconfig_activation_dtype
                and output_dtype == torch.float
                and weight_dtype == qconfig_weight_dtype
            )
        else:
            is_match = (
                input_dtype == qconfig_activation_dtype
                and output_dtype == qconfig_activation_dtype
                and weight_dtype == qconfig_weight_dtype
                and bias_dtype == qconfig_bias_dtype
            )
        if is_match:
            return True
    return False


def _get_object_type_qconfig(
    qconfig_mapping: QConfigMapping,
    object_type: Callable | str,
    fallback_qconfig: QConfigAny,
) -> QConfigAny:
    return qconfig_mapping.object_type_qconfigs.get(object_type, fallback_qconfig)


def _get_module_name_regex_qconfig(qconfig_mapping, module_name, fallback_qconfig):
    for regex_pattern, qconfig in qconfig_mapping.module_name_regex_qconfigs.items():
        if re.match(regex_pattern, module_name):
            # first match wins
            return qconfig
    return fallback_qconfig


def _get_module_name_qconfig(qconfig_mapping, module_name, fallback_qconfig):
    if module_name == "":
        # module name qconfig not found
        return fallback_qconfig
    if module_name in qconfig_mapping.module_name_qconfigs:
        return qconfig_mapping.module_name_qconfigs[module_name]
    else:
        parent, _ = _parent_name(module_name)
        return _get_module_name_qconfig(qconfig_mapping, parent, fallback_qconfig)


def _maybe_adjust_qconfig_for_module_type_or_name(
    qconfig_mapping, module_type, module_name, global_qconfig
):
    # get qconfig for module_name,
    # fallback to module_name_regex_qconfig, module_type_qconfig,
    # global_qconfig if necessary
    module_type_qconfig = _get_object_type_qconfig(
        qconfig_mapping, module_type, global_qconfig
    )
    module_name_regex_qconfig = _get_module_name_regex_qconfig(
        qconfig_mapping, module_name, module_type_qconfig
    )
    module_name_qconfig = _get_module_name_qconfig(
        qconfig_mapping, module_name, module_name_regex_qconfig
    )
    return module_name_qconfig


def _get_flattened_qconfig_dict(
    qconfig_mapping: QConfigMapping,
) -> dict[Callable | str, QConfigAny]:
    """flatten the global, object_type and module_name qconfig
    to the same qconfig_dict so that it can be used by
    propagate_qconfig_ function.
    "module_name_regex" is ignored for now since it's not supported
    in propagate_qconfig_, but it can be fixed later.

    For example:
    Input: {
      "": qconfig,
      "object_type": [
        (torch.add, qconfig)
      ],
      "module_name": [
        ("conv", qconfig)
      ]
    }

    Output: {
      "": qconfig,
      torch.add: qconfig,
      "conv": qconfig
    }
    """
    flattened: dict[Callable | str, QConfigAny] = {"": qconfig_mapping.global_qconfig}
    flattened.update(qconfig_mapping.object_type_qconfigs)
    flattened.update(qconfig_mapping.module_name_qconfigs)  # type: ignore[arg-type]
    return flattened


def _update_qconfig_for_qat(
    qconfig_mapping: QConfigMapping, backend_config: BackendConfig
):
    """
    Update the qconfig_mapping to account for module swaps during QAT.
    During QAT we perform a module swap on the nn.Module types to the corresponding nn.qat.modules types.
    """
    module_to_qat_module_class = get_module_to_qat_module(backend_config)
    object_type_dict = qconfig_mapping.object_type_qconfigs
    new_object_type_dict = object_type_dict.copy()
    for k, v in new_object_type_dict.items():
        if k in module_to_qat_module_class:
            object_type_dict[module_to_qat_module_class[k]] = v
<|endoftext|>from torch.ao.quantization.qconfig import QConfigAny
from torch.fx import GraphModule

from ._lower_to_native_backend import _lower_to_native_backend


__all__ = ["lower_to_fbgemm"]


def lower_to_fbgemm(
    model: GraphModule,
    qconfig_map: dict[str, QConfigAny],
    node_name_to_scope: dict[str, tuple[str, type]],
    keep_original_weights: bool = False,
) -> GraphModule:
    """Lower a quantized reference model (with reference quantized operator patterns)
    to fbgemm
    """
    return _lower_to_native_backend(
        model, qconfig_map, node_name_to_scope, keep_original_weights
    )
<|endoftext|>from torch.ao.quantization.qconfig import QConfigAny
from torch.fx import GraphModule

from ._lower_to_native_backend import _lower_to_native_backend


__all__ = ["lower_to_qnnpack"]


def lower_to_qnnpack(
    model: GraphModule,
    qconfig_map: dict[str, QConfigAny],
    node_name_to_scope: dict[str, tuple[str, type]],
) -> GraphModule:
    """Lower a quantized reference model (with reference quantized operator patterns)
    to qnnpack
    """
    return _lower_to_native_backend(model, qconfig_map, node_name_to_scope)
<|endoftext|># mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.fuser_method_mappings import get_fuser_method_new
from torch.ao.quantization.utils import _parent_name, NodePattern, Pattern
from torch.fx.graph import Graph, Node
from torch.nn.utils.parametrize import type_before_parametrizations

from .custom_config import FuseCustomConfig
from .match_utils import MatchAllNode


__all__ = [
    "DefaultFuseHandler",
    "FuseHandler",
]


# ----------------------------
# Fusion Pattern Registrations
# ----------------------------


# Base Pattern Handler
class FuseHandler(ABC):
    """Base handler class for the fusion patterns"""

    @abstractmethod
    def __init__(self, node: Node):
        pass

    @abstractmethod
    def fuse(
        self,
        load_arg: Callable,
        named_modules: dict[str, torch.nn.Module],
        fused_graph: Graph,
        root_node: Node,
        extra_inputs: list[Any],
        matched_node_pattern: NodePattern,
        fuse_custom_config: FuseCustomConfig,
        fuser_method_mapping: dict[Pattern, torch.nn.Sequential | Callable],
        is_qat: bool,
    ) -> Node:
        pass


class DefaultFuseHandler(FuseHandler):
    def __init__(self, node: Node):  # pylint: disable=useless-parent-delegation
        super().__init__(node)  # type:ignore[safe-super]

    def fuse(
        self,
        load_arg: Callable,
        named_modules: dict[str, torch.nn.Module],
        fused_graph: Graph,
        root_node: Node,
        extra_inputs: list[Any],
        matched_node_pattern: NodePattern,
        fuse_custom_config: FuseCustomConfig,
        fuser_method_mapping: dict[Pattern, torch.nn.Sequential | Callable],
        is_qat: bool,
    ) -> Node:
        if root_node.op != "call_module":
            raise AssertionError("Expecting module node to be a call_module Node")
        root_module = named_modules[str(root_node.target)]

        def get_modules(pattern):
            """Given a node pattern, extract the corresponding modules
            e.g. input: (relu_node, (bn_node, conv_node))
                 output: (relu_module, (bn_module, conv_module))
            """
            if isinstance(pattern, (tuple, list)):
                n, *args = pattern
                modules: list[torch.nn.Module] = []
                modules.append(get_modules(n))
                modules.extend(get_modules(a) for a in args)
                return tuple(modules)
            else:
                n = pattern
                if n.op == "call_module":
                    return named_modules[n.target]
                elif n.op == "call_function" and n.target is torch.nn.functional.relu:
                    relu = torch.nn.ReLU()
                    relu.training = root_module.training
                    return relu
                elif n.op == "call_function" or n.op == "call_method":
                    return n.target
                else:
                    return MatchAllNode

        # since relu can be used multiple times, we'll need to create a relu module for each match
        matched_modules = get_modules(matched_node_pattern)

        def get_matched_types(m):
            if isinstance(m, tuple):
                return tuple(map(get_matched_types, m))
            if isinstance(m, torch.nn.Module):
                return type_before_parametrizations(m)
            return m

        matched_module_types = get_matched_types(matched_modules)
        module_parent_name, module_name = _parent_name(root_node.target)
        fuser_method = get_fuser_method_new(matched_module_types, fuser_method_mapping)
        # TODO: change the signature for fuser_method to take matched module patterns
        # as input
        fused_module = fuser_method(is_qat, *matched_modules)
        setattr(named_modules[module_parent_name], module_name, fused_module)
        extra_args = [load_arg(input) for input in extra_inputs]
        node = fused_graph.node_copy(root_node, load_arg)
        args = list(node.args)
        args.extend(extra_args)
        node.args = tuple(args)
        return node


def _get_fusion_pattern_to_fuse_handler_cls(
    backend_config: BackendConfig,
) -> dict[Pattern, Callable]:
    fusion_pattern_to_fuse_handlers: dict[Pattern, Callable] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.fuser_method is not None:
            # TODO: is this logic right?
            fusion_pattern_to_fuse_handlers[pattern] = DefaultFuseHandler
    return fusion_pattern_to_fuse_handlers
<|endoftext|># mypy: allow-untyped-defs
import operator
import warnings
from collections import namedtuple
from typing import Any
from typing_extensions import TypeIs

import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from torch.ao.quantization.observer import (
    _with_args,
    ObserverBase,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.utils import _parent_name, check_min_max_valid
from torch.fx import GraphModule
from torch.fx.graph import Node

from .utils import (
    get_new_attr_name_with_prefix,
    maybe_get_next_module,
    node_arg_is_weight,
)


CUSTOM_MODULE_SUPP_LIST: list[Any] = []


def reshape_scale(scale: torch.Tensor, axis: int, input: torch.Tensor) -> torch.Tensor:
    """Reshapes the scale so that we can multiply it to the input by the given axis."""
    new_shape = [1] * input.ndim
    new_shape[axis] = input.size(axis)
    return scale.view(new_shape)


qsheme_mapping_per_tensor_to_per_channel = {
    torch.per_tensor_affine: torch.per_channel_affine,
    torch.per_tensor_symmetric: torch.per_channel_symmetric,
}


class _InputEqualizationObserver(nn.Module):
    r"""Observer for tracking the running min/max values of input columns, and
    computing the quantization parameters for the overall min/max input values.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    The running minimum/maximum :math:`x_\text{min/max}` are computed in the
    same way as :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`,
    with the difference that the running min/max values are stored per column.
    This observer is intended to be used along with a WeightEqualizationObserver
    to calculate the equalization scale.
    """

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ) -> None:
        super().__init__()

        if qscheme not in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            raise TypeError("Input qscheme must be per-tensor")

        self.dtype = dtype
        self.qscheme = qscheme

        per_channel_qscheme = qsheme_mapping_per_tensor_to_per_channel[qscheme]
        self.input_obs = PerChannelMinMaxObserver(
            ch_axis=1,
            dtype=dtype,
            qscheme=per_channel_qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
        )

        self.equalization_scale = torch.tensor(1)
        self.equalization_shape: list[int] = []

    def forward(self, x_orig):
        if x_orig.ndim < 2 or x_orig.ndim > 5:
            raise ValueError(
                "InputEqualizationObserver only supports Linear and Conv layers"
            )

        # Calculate the shape needed to reshape the equalization scale later (needed for Conv layers)
        self.equalization_shape = [1] * x_orig.ndim
        self.equalization_shape[1] = x_orig.size(1)

        return self.input_obs(x_orig)

    def get_input_minmax(self):
        return (self.input_obs.min_val, self.input_obs.max_val)

    def set_equalization_scale(self, equalization_scale):
        # Reshape the equalization scale along axis=1 so that it can be
        # multiplied with the input along axis=1
        if equalization_scale.nelement() == 1 and equalization_scale == torch.tensor(1):
            return
        self.equalization_scale = torch.reshape(
            equalization_scale, self.equalization_shape
        )

    def calculate_scaled_minmax(self):
        r"""Returns the scaled min/max inputs"""
        if (
            self.equalization_scale.nelement() == 1
            and self.equalization_scale == torch.tensor(1)
        ):
            warnings.warn(
                "Must call calculate_equalization_scale before calling calculate_scaled_minmax. "
                + "Will not scale the next quantization observer.",
                stacklevel=2,
            )
            return None, None

        # Calculate qparams for the scaled min/max inputs
        # Scale the input by the equalization scale located at the same column
        # index
        (min_inputs, max_inputs) = self.get_input_minmax()
        equalization_scale_reshaped = reshape_scale(
            self.equalization_scale, 0, min_inputs
        )
        min_input_scaled = torch.min(torch.mul(min_inputs, equalization_scale_reshaped))
        max_input_scaled = torch.max(torch.mul(max_inputs, equalization_scale_reshaped))

        return min_input_scaled, max_input_scaled

    with_args = classmethod(_with_args)


class _WeightEqualizationObserver(nn.Module):
    r"""Observer for tracking the running min/max values of weight columns and
    rows, and computing the quantization parameters for the weight rows.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    This observer is made up of 1 PerChannelMinMaxObserver `weight_col_obs` used
    to record the running minimum and maximum of columns of incoming weight
    tensors. This observer is intended to be used along with an
    InputEqualizationObserver to calculate the equalization scale.

    The running minimum/maximum :math:`w_\text{min/max}` are computed in the
    same way as :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`.
    """

    def __init__(
        self,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ) -> None:
        super().__init__()

        self.dtype = dtype
        self.qscheme = qscheme
        self.ch_axis = 1

        per_channel_qscheme = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            per_channel_qscheme = qsheme_mapping_per_tensor_to_per_channel[qscheme]
        self.weight_col_obs = PerChannelMinMaxObserver(
            ch_axis=1,
            dtype=dtype,
            qscheme=per_channel_qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
        )

        self.equalization_scale = torch.tensor(1)

    def forward(self, w_orig):
        if w_orig.ndim < 2 or w_orig.ndim > 5:
            raise ValueError(
                "InputEqualizationObserver only supports Linear and Conv layers"
            )

        return self.weight_col_obs(w_orig)

    def get_weight_col_minmax(self):
        return (self.weight_col_obs.min_val, self.weight_col_obs.max_val)

    def set_equalization_scale(self, equalization_scale):
        self.equalization_scale = equalization_scale

    with_args = classmethod(_with_args)


def calculate_equalization_scale(
    input_obs: _InputEqualizationObserver, weight_obs: _WeightEqualizationObserver
) -> torch.Tensor:
    r"""Calculates the equalization scale and sets the equalization_scale value
    in the observers.

    Args:
        input_obs: Observer that tracks the ranges for the input columns
        weight_obs: Observer that tracks the ranges for the weight columns
    """

    (min_inputs, max_inputs) = input_obs.get_input_minmax()
    (min_weights, max_weights) = weight_obs.get_weight_col_minmax()

    if not (
        check_min_max_valid(min_inputs, max_inputs)
        and check_min_max_valid(min_weights, max_weights)
    ):
        warnings.warn(
            "Must run observer before calling calculate_equalization_scale. "
            + "Returning default equalization scale torch.tensor(1).",
            stacklevel=2,
        )
        return torch.tensor(1)

    if min_inputs.shape != min_weights.shape:
        raise ValueError(
            "Input and Weight must have the same column dimension. "
            + f"Found {min_inputs.shape} and {min_weights.shape} shapes instead."
        )

    equalization_scale = torch.sqrt(
        (max_weights - min_weights) / (max_inputs - min_inputs)
    )
    # Replace all 'inf', 'nan', 0's with 1s to prevent errors
    equalization_scale[equalization_scale == 0.0] = 1
    equalization_scale = torch.nan_to_num(equalization_scale, nan=1, posinf=1, neginf=1)
    return equalization_scale


class EqualizationQConfig(
    # pyrefly: ignore [invalid-inheritance]
    namedtuple("EqualizationQConfig", ["input_activation", "weight"])
):
    """
    Describes how to quantize a layer or a part of the network specifically for
    input-weight equalization by providing settings (observer classes) for
    inputs, outputs, and weights.

    Note that EqualizationQConfig needs to contain observer **classes** (like
    MinMaxObserver) or a callable that returns instances on invocation, not the
    concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of
    the layers.

    Observer classes have usually reasonable default arguments, but they can be
    overwritten with `with_args` method (that behaves like functools.partial):

    my_qconfig = EqualizationQConfig(input_activation=_InputEqualizationObserver.with_args(dtype=torch.qint8),
                                    weight=_WeightEqualizationObserver.with_args(dtype=torch.qint8))
    """

    __slots__ = ()

    def __new__(cls, input_activation=torch.nn.Identity, weight=torch.nn.Identity):
        if isinstance(input_activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError(
                "EqualizationQConfig received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        self = super().__new__(cls, input_activation, weight)
        return self


input_equalization_observer = _InputEqualizationObserver.with_args(
    dtype=torch.quint8, qscheme=torch.per_tensor_symmetric
)
weight_equalization_observer = _WeightEqualizationObserver.with_args(
    dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)
default_equalization_qconfig = EqualizationQConfig(
    input_activation=input_equalization_observer, weight=weight_equalization_observer
)


def fused_module_supports_equalization(module) -> bool:
    """Checks if the fused node supports equalization."""
    return type(module) in [
        nni.LinearReLU,
        nni.ConvReLU1d,
        nni.ConvReLU2d,
        nni.ConvReLU3d,
    ]


def nn_module_supports_equalization(module) -> bool:
    """Checks if the torch.nn node supports equalization."""
    return type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]


def custom_module_supports_equalization(module) -> bool:
    """Checks if the custom node supports equalization."""
    return type(module) in CUSTOM_MODULE_SUPP_LIST


def node_supports_equalization(node: Node, modules) -> bool:
    """Checks if the current node supports equalization
    Currently we only support nn.Linear/F.Linear and nn.Conv/F.conv layers
    """
    if node.op == "call_module":
        return (
            nn_module_supports_equalization(modules[str(node.target)])
            or fused_module_supports_equalization(modules[str(node.target)])
            or custom_module_supports_equalization(modules[str(node.target)])
        )
    elif node.op == "call_function":
        return node.target in [F.linear, F.conv1d, F.conv2d, F.conv3d]
    return False


def is_equalization_observer(
    observer: nn.Module,
) -> TypeIs[_InputEqualizationObserver | _WeightEqualizationObserver]:
    return isinstance(
        observer, (_InputEqualizationObserver, _WeightEqualizationObserver)
    )


###############################################################################
# Functions for equalization during convert                                   #
###############################################################################


def get_op_node_and_weight_eq_obs(
    input_eq_obs_node: Node, model: GraphModule, modules: dict[str, nn.Module]
) -> tuple[Node | None, _WeightEqualizationObserver | None]:
    """Gets the following weight equalization observer. There should always
    exist a weight equalization observer after an input equalization observer.

    Returns the operation node that follows the input equalization observer node
    and the weight equalization observer
    """

    # Find the op node that comes directly after the input equalization observer
    op_node = None
    for user in input_eq_obs_node.users:
        if node_supports_equalization(user, modules):
            op_node = user
            break

    if op_node is None:
        raise AssertionError(
            "Expected an operation node after the input equalization observer"
        )
    if op_node.op == "call_module":
        # If the op_node is a nn.Linear layer, then it must have a
        # WeightEqualizationObserver configuration
        maybe_equalization_node_name_to_config = _get_observed_graph_module_attr(
            model, "equalization_node_name_to_qconfig"
        )
        if maybe_equalization_node_name_to_config is None:
            raise AssertionError(
                "Expected 'equalization_node_name_to_qconfig' attribute in observed graph module"
            )
        equalization_node_name_to_qconfig: dict[str, Any] = (
            maybe_equalization_node_name_to_config  # type: ignore[assignment]
        )
        if equalization_node_name_to_qconfig.get(op_node.name) is None:
            raise AssertionError(
                f"No equalization qconfig found for op node {op_node.name}"
            )
        weight_eq_obs = equalization_node_name_to_qconfig.get(op_node.name).weight()  # type: ignore[union-attr]

        if not isinstance(weight_eq_obs, _WeightEqualizationObserver):
            raise AssertionError(
                "Expected weight equalization observer to be a _WeightEqualizationObserver"
            )
        return op_node, weight_eq_obs

    elif op_node.op == "call_function":
        weight_node = maybe_get_weight_eq_obs_node(op_node, modules)
        if weight_node is not None:
            weight_eq_obs = modules[str(weight_node.target)]
            if not isinstance(weight_eq_obs, _WeightEqualizationObserver):
                raise AssertionError(
                    "Expected weight equalization observer to be a _WeightEqualizationObserver"
                )
            return op_node, weight_eq_obs

    return None, None


def maybe_get_weight_eq_obs_node(
    op_node: Node, modules: dict[str, nn.Module]
) -> Node | None:
    """Gets the weight equalization observer node if it exists."""
    if op_node.op != "call_function":
        raise AssertionError(
            "maybe_get_weight_eq_obs_node expects a call_function op_node"
        )
    for node_arg in op_node.args:
        if node_arg_is_weight(op_node, node_arg):
            if (
                isinstance(node_arg, Node)
                and node_arg.op == "call_module"
                and isinstance(
                    modules[str(node_arg.target)], _WeightEqualizationObserver
                )
            ):
                return node_arg
    return None


def maybe_get_next_input_eq_obs(
    node: Node, modules: dict[str, nn.Module]
) -> _InputEqualizationObserver | None:
    """Gets the following input equalization observer if it exists.

    For example, in the case of connecting linear layers:
        x -> inp_obs1 -> eq_obs1 -> linear1 -> out_obs1 -> eq_obs2 -> linear2 -> out_obs2
    If the node being passed in is the linear1 node, then we want to return eq_obs2,
    the following equalization observer for linear2.

    However, if there are no connecting layers:
        x -> inp_obs1 -> eq_obs1 -> linear1 -> out_obs1 -> add
    Then we want to return None.

    In the case of an unfused linear-relu layer with a connecting linear layer:
        linear1 -> relu -> out_obs1 -> eq_obs2 -> linear2 -> out_obs2
    Since it is unfused, we want to skip over the relu layer and return eq_obs2,
    the following equalization observer for linear2.
    """

    if not node_supports_equalization(node, modules):
        raise AssertionError("Node does not support equalization")

    # Locate the following nn.ReLU or F.relu node if it exists
    maybe_relu_node = maybe_get_next_module(node, modules, nn.ReLU)
    if maybe_relu_node is None:
        maybe_relu_node = maybe_get_next_module(
            node, modules, target_functional_type=F.relu
        )

    # Locate the following output observer if it exists.
    # We will skip the relu node if it exists.
    maybe_obs_node = (
        maybe_get_next_module(node, modules, ObserverBase)
        if maybe_relu_node is None
        else maybe_get_next_module(maybe_relu_node, modules, ObserverBase)
    )
    if maybe_obs_node is None:
        return None

    maybe_eq_obs_node = maybe_get_next_module(
        maybe_obs_node, modules, _InputEqualizationObserver
    )
    if maybe_eq_obs_node is None:
        return None

    maybe_eq_obs = modules[str(maybe_eq_obs_node)]
    if not isinstance(maybe_eq_obs, _InputEqualizationObserver):
        raise AssertionError(
            "Expected the following equalization observer to be an _InputEqualizationObserver"
        )
    return maybe_eq_obs


def maybe_get_next_equalization_scale(
    node: Node, modules: dict[str, nn.Module]
) -> torch.Tensor | None:
    """If the next next node is an InputEqualizationObserver then we want to
    return its equalization scale, else we return 1

    This is used in the case where there are two connecting linear layers:
        linear1 -> LinearOutObs -> InputEqObs -> linear2
    In this case, the node given is linear1 and we want to locate the InputEqObs.
    """
    next_inp_eq_obs = maybe_get_next_input_eq_obs(node, modules)

    if next_inp_eq_obs:
        if (
            next_inp_eq_obs.equalization_scale.nelement() == 1
            and next_inp_eq_obs.equalization_scale == torch.tensor(1)
        ):
            return None
        return next_inp_eq_obs.equalization_scale
    return None


def scale_input_observer(node: Node, modules: dict[str, nn.Module]) -> None:
    """Scales the following input quantization observer's min/max values by
    updating the values with the scaled min/max values calculated by the input
    equalization observer
    """
    input_eq_obs = modules[str(node.target)]
    if not isinstance(input_eq_obs, _InputEqualizationObserver):
        raise AssertionError(
            "Expected the module at node.target to be an _InputEqualizationObserver"
        )

    input_quant_obs_node = node.args[0]
    if not isinstance(input_quant_obs_node, Node):
        raise AssertionError(
            "Expected the input quantization observer node to be a Node"
        )

    input_quant_obs = modules[str(input_quant_obs_node.target)]
    if not isinstance(input_quant_obs, ObserverBase):
        return

    min_input_scaled, max_input_scaled = input_eq_obs.calculate_scaled_minmax()
    if min_input_scaled is None and max_input_scaled is None:
        return
    input_quant_obs.min_val = min_input_scaled
    input_quant_obs.max_val = max_input_scaled


def scale_weight_node(
    node: Node,
    modules: dict[str, nn.Module],
    equalization_scale: torch.Tensor,
    next_equalization_scale: torch.Tensor | None,
) -> None:
    """Scale the weights for input-weight equalization by multiplying the
    weight by 1/equalization_scale and next_equalization_scale

    Args:
        node: Current node whose weights we want to scale
        equalization_scale: Current node's calculated equalization scale
        next_equalization_scale: Next node's calculated equalization scale if
           the following node needs to be equalized, 1 otherwise
    """
    if equalization_scale is None:
        return

    if fused_module_supports_equalization(modules[str(node.target)]):
        op_module = modules[str(node.target)][0]  # type: ignore[index]
    else:
        op_module = modules[str(node.target)]
    if not (
        nn_module_supports_equalization(op_module)
        or custom_module_supports_equalization(op_module)
    ):
        raise AssertionError(
            "Expected operation module to support equalization (nn or custom)"
        )

    # Scale the weights for input-weight equalization
    # If the following layer needs to be equalized then we will multiply its scale
    weight = op_module.weight
    if not isinstance(weight, torch.Tensor):
        raise AssertionError("Expected op_module.weight to be a torch.Tensor")

    # Scale the weights by the reciprocal of the equalization scale
    # Reshape the equalization scale so that we can multiply it to the weight along axis=1
    equalization_scale_reshaped = reshape_scale(equalization_scale, 1, weight)
    scaled_weight = torch.mul(weight, torch.reciprocal(equalization_scale_reshaped))

    if next_equalization_scale is None:
        op_module.weight = nn.Parameter(scaled_weight)
        return

    # Multiply the weights row wise by the next equalization scale
    # Reshape the equalization scale so that we can multiply it to the weight along axis=0
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, weight)
    scaled_weight = torch.mul(scaled_weight, next_equalization_scale_reshaped)

    op_module.weight = nn.Parameter(scaled_weight)

    # Multiply the bias element wise by the next equalization scale
    bias = op_module.bias
    if bias is None:
        return
    if not isinstance(bias, torch.Tensor):
        raise AssertionError("Expected op_module.bias to be a torch.Tensor")

    # Reshape the equalization scale so that we can multiply it element-wise to the bias
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, bias)
    scaled_bias = torch.mul(bias, next_equalization_scale_reshaped)
    op_module.bias = nn.Parameter(scaled_bias)


def scale_weight_functional(
    op_node: Node,
    model: GraphModule,
    modules: dict[str, nn.Module],
    equalization_scale: torch.Tensor,
    next_equalization_scale: torch.Tensor | None,
) -> None:
    """Scales the weight value for functional layers"""
    if equalization_scale is None:
        return

    # From the given op_node, the path looks like:
    #   get_attr(weight) -> weight_quant_obs -> weight_eq_obs -> op_node
    # So we want to trace back from the op_node to get the equalization observer
    # node, then the quantization observer node, and then finally the weight
    # node which contains the weight values.

    # Get the equalization observer node
    weight_eq_obs_node = maybe_get_weight_eq_obs_node(op_node, modules)
    if weight_eq_obs_node is None:
        return

    # Get the quantization observer node
    weight_quant_obs_node = weight_eq_obs_node.args[0]
    if weight_quant_obs_node is None:
        return
    if not (
        isinstance(weight_quant_obs_node, Node)
        and isinstance(modules[str(weight_quant_obs_node.target)], ObserverBase)
    ):
        raise AssertionError(
            "Expected weight_quant_obs_node to be a Node whose module is an ObserverBase"
        )

    # Get the get_attr(weight) node
    weight_node = weight_quant_obs_node.args[0]
    if weight_node is None:
        return
    if not (isinstance(weight_node, Node) and weight_node.op == "get_attr"):
        raise AssertionError("Expected weight node to be a 'get_attr' Node")

    weight_parent_name, weight_name = _parent_name(weight_node.target)
    weight = getattr(modules[weight_parent_name], weight_name)

    # Scale the weights for input-weight equalization
    # If the following layer needs to be equalized then we will multiply its scale
    # Reshape the equalization scale so that we can multiply it to the weight along axis=1
    equalization_scale_reshaped = reshape_scale(equalization_scale, 1, weight)
    scaled_weight = torch.mul(weight, torch.reciprocal(equalization_scale_reshaped))

    if next_equalization_scale is None:
        setattr(modules[weight_parent_name], weight_name, scaled_weight)
        return

    # Multiply the weights row wise by the next equalization scale
    # Reshape the equalization scale so that we can multiply it to the weight along axis=1
    next_equalization_scale_reshaped = reshape_scale(
        next_equalization_scale, 0, scaled_weight
    )
    scaled_weight = torch.mul(scaled_weight, next_equalization_scale_reshaped)

    setattr(modules[weight_parent_name], weight_name, scaled_weight)
    if not torch.allclose(model.get_buffer(str(weight_node.target)), scaled_weight):
        raise AssertionError("Model buffer for weight does not match the scaled weight")

    # Multiply the bias element wise by the next equalization scale
    bias_node = None
    for node in op_node.args:
        # Find the node containing the weight values
        if isinstance(node, Node) and node.op == "get_attr" and "bias" in node.name:
            bias_node = node
            break
    if bias_node is None:
        return

    bias_parent_name, bias_name = _parent_name(bias_node.target)
    bias = getattr(modules[bias_parent_name], bias_name)

    # Reshape the equalization scale so that we can multiply it element-wise to the bias
    next_equalization_scale_reshaped = reshape_scale(next_equalization_scale, 0, bias)
    scaled_bias = torch.mul(bias, next_equalization_scale_reshaped)
    setattr(modules[bias_parent_name], bias_name, scaled_bias)


def clear_weight_quant_obs_node(op_node: Node, modules: dict[str, nn.Module]) -> None:
    """Given the operation node, we want find the corresponding quantization
    observer and reset its min/max values
    """
    weight_eq_obs_node = maybe_get_weight_eq_obs_node(op_node, modules)
    if weight_eq_obs_node is None:
        return

    weight_quant_obs_node = weight_eq_obs_node.args[0]
    if weight_quant_obs_node is None:
        return
    if not isinstance(weight_quant_obs_node, Node):
        raise AssertionError("Expected weight_quant_obs_node to be a Node")

    weight_quant_obs = modules[str(weight_quant_obs_node.target)]
    if not isinstance(modules[str(weight_quant_obs_node.target)], ObserverBase):
        raise AssertionError(
            "Expected the module at weight_quant_obs_node to be an ObserverBase"
        )
    weight_quant_obs.reset_min_max_vals()  # type: ignore[operator]


def remove_node(model: GraphModule, node: Node, prev_node: Node):
    """Removes the given node from the model by replacing all of its users with
    the given previous node
    """
    # For all of the current node's users, replace the current node with
    # the input quantization observer node
    orig_users = list(node.users.keys())
    for user_node in orig_users:
        user_node.replace_input_with(node, prev_node)

    # Erase the InputEqualizationObserver node
    model.graph.erase_node(node)


def update_obs_for_equalization(
    model: GraphModule, modules: dict[str, nn.Module]
) -> dict[str, _WeightEqualizationObserver]:
    """Update all of the observer's equalization scale. For each
    InputEqualizationObserver, we will find the location of the next
    WeightEqualizationObserver, create it, and calculate the equalization scale
    based on the two observers.

    We will then return a dictionary mapping operation node names to
    the corresponding WeightEqualizationObservers for that operation.
    """
    weight_eq_obs_dict = {}
    for node in model.graph.nodes:
        if node.op == "call_module" and isinstance(
            modules[node.target], _InputEqualizationObserver
        ):
            input_eq_obs = modules[node.target]
            if not isinstance(input_eq_obs, _InputEqualizationObserver):
                raise AssertionError(
                    "Expected module at node.target to be an _InputEqualizationObserver"
                )
            op_node, weight_eq_obs = get_op_node_and_weight_eq_obs(node, model, modules)

            if op_node is None or weight_eq_obs is None:
                continue

            if op_node.op == "call_module":
                # Calibrate the weight equalization observer since it has just
                # been created
                if fused_module_supports_equalization(modules[str(op_node.target)]):
                    module = modules[str(op_node.target)][0]  # type: ignore[index]
                    if not nn_module_supports_equalization(module):
                        raise AssertionError(
                            "Expected fused module to support equalization"
                        )
                    weight_eq_obs(module.weight)
                else:
                    weight_eq_obs(modules[str(op_node.target)].weight)

            # Calculate and set the equalization scale values
            equalization_scale = calculate_equalization_scale(
                input_eq_obs, weight_eq_obs
            )
            input_eq_obs.set_equalization_scale(equalization_scale)
            weight_eq_obs.set_equalization_scale(equalization_scale)

            weight_eq_obs_dict[op_node.name] = weight_eq_obs

    return weight_eq_obs_dict


def convert_eq_obs(
    model: GraphModule,
    modules: dict[str, nn.Module],
    weight_eq_obs_dict: dict[str, _WeightEqualizationObserver],
) -> None:
    """Converts the equalization operations and updates the other nodes in the
    following way:
        - Removes the input equalization observers and inserts a mul operator
          along with an equalization scale node wherever applicable (we do not
          want to insert a mul operator between connecting linear layers).
        - Updates the input quantization observers with the scaled input min/max
          values.
        - Scales the weights by the current and next equalization scales.
        - Removes the weight equalization observer node if it exists.

    Before (after prepare):
                                    weight values
                                          |
                                    WeightQuantObs
                                          |
                                      WeightEqObs
                                          |
        x -> InpQuantObs -> InpEqObs -> linear -> OutQuantObs

    After this function:
                                              scaled weight values
                                                      |
       equalization scale                       WeightQuantObs
              |                                       |
        x -> mul -> InpQuantObs (scaled min/max) -> linear -> OutQuantObs

    After convert:
       equalization scale                 scaled weight values
              |                                    |
        x -> mul -> quantize_per_tensor -> quantized::linear

    Note that although the equalization observer appeared after the quantization
    observer after prepare_fx, the mul node appears before the quantization node
    after convert_fx. This is because placing the equalization observer after
    the quantization observer in prepare_fx would allow us to keep the invariant
    that the graph before the current node inserts its observers is not
    modified.

    Having the equalization observer before the quantization observer would also
    cause some inconsistencies between the ordering of the quantization and
    equalization observers.
    For example, a single linear layer would look like:
        x -> InpEqObs1 -> InpQuantObs1 -> linear1 -> OutQuantObs1
    But between two connected linear layers, it would look like:
        linear1 -> OutQuantObs1 -> InpEqObs2 -> linear2 -> OutQuantObs2
    """
    for node in model.graph.nodes:
        if node.op == "call_module" and isinstance(
            modules[node.target], _InputEqualizationObserver
        ):
            inp_quant_obs_node = node.args[0]
            prev_node = inp_quant_obs_node.args[0]

            # If the previous node is a layer that needs to be equalized, then
            # we will remove the current node because we do not need to add any
            # equalization nodes between two layers that need to be equalized

            # Before: linear1/relu (prev_node) -> output_quant_obs1 (inp_quant_obs_node) -> input_eq_obs2 (node) -> linear2
            # After: linear1/relu (prev_node) -> output_quant_obs1 (inp_quant_obs_node) -> linear2
            if (
                node_supports_equalization(prev_node, modules)
                or "relu" in prev_node.name
            ):
                remove_node(model, node, inp_quant_obs_node)
                continue

            # Update the following input quantization observer's min/max values
            scale_input_observer(node, modules)

            # Remove the InputEqualization node and add a mul operator before
            # the quantization observer node that appears before the equalization node
            # Before: x -> input_quant_obs -> input_eq_obs -> linear
            # After: x -> mul -> input_quant_obs -> linear

            # Create a node containing the equalization scale
            with model.graph.inserting_before(inp_quant_obs_node):
                get_new_eq_scale_name = get_new_attr_name_with_prefix(
                    prev_node.name + "_equalization_scale"
                )
                name = get_new_eq_scale_name(modules)
                setattr(model, name, modules[node.target].equalization_scale)
                eq_scale_node = model.graph.create_node("get_attr", name)

            # Create a node multiplying the input with the equalization scale
            with model.graph.inserting_after(eq_scale_node):
                inputs = (prev_node, eq_scale_node)
                mul_node = model.graph.create_node("call_function", torch.mul, inputs)

            # Set the mul nod to be the input_quant_obs_node's input instead of
            # the previous node
            inp_quant_obs_node.replace_input_with(prev_node, mul_node)
            remove_node(model, node, inp_quant_obs_node)

        elif weight_eq_obs_dict.get(node.name) is not None:
            weight_eq_obs = weight_eq_obs_dict.get(node.name)
            if not isinstance(weight_eq_obs, _WeightEqualizationObserver):
                raise AssertionError(
                    "Expected weight equalization observer to be a _WeightEqualizationObserver"
                )
            equalization_scale = weight_eq_obs.equalization_scale

            if (
                equalization_scale.nelement() == 1
                and equalization_scale == torch.tensor(1)
            ):
                equalization_scale = None  # type: ignore[assignment]
            maybe_next_equalization_scale = maybe_get_next_equalization_scale(
                node, modules
            )

            # Scale the weight nodes
            if node.op == "call_module":
                scale_weight_node(
                    node,
                    modules,
                    # pyrefly: ignore [bad-argument-type]
                    equalization_scale,
                    maybe_next_equalization_scale,
                )
            elif node.op == "call_function":
                scale_weight_functional(
                    node,
                    model,
                    modules,
                    # pyrefly: ignore [bad-argument-type]
                    equalization_scale,
                    maybe_next_equalization_scale,
                )

                weight_eq_obs_node = maybe_get_weight_eq_obs_node(node, modules)
                if weight_eq_obs_node is None:
                    return
                if not isinstance(
                    modules[str(weight_eq_obs_node.target)], _WeightEqualizationObserver
                ):
                    raise AssertionError(
                        "Expected weight equalization observer to be a _WeightEqualizationObserver"
                    )

                # Clear the quantization observer's min/max values so that they
                # can get updated later based on the new scale values
                clear_weight_quant_obs_node(node, modules)

                # Erase the weight equalization observer node
                prev_node = weight_eq_obs_node.args[0]
                remove_node(model, weight_eq_obs_node, prev_node)  # type: ignore[arg-type]
            else:
                raise ValueError(
                    "Expected operation node to be 'call_module' or 'call_function"
                    + f"Instead got node {node.name} as '{node.op}'."
                )


def _convert_equalization_ref(model: GraphModule):
    """Reference function which applies changes needed for equalization, but
    does not quantize the nodes
    """
    modules = dict(model.named_modules(remove_duplicate=False))

    # Calculate the equalization scale, update the observers with the scaled
    # inputs, and scale the weight
    weight_eq_obs_dict = update_obs_for_equalization(model, modules)
    convert_eq_obs(model, modules, weight_eq_obs_dict)

    return GraphModule(model, model.graph)


###############################################################################
# Functions for running the equalized model on the Numeric Suite              #
###############################################################################


def get_layer_sqnr_dict(
    model_a: nn.Module, model_b: nn.Module, x: torch.Tensor
) -> dict[str, float]:
    """Runs the Numeric Suite on model_a and model_b and returns a dictionary
    containing the SQNR between layers in model_a and model_b.

    Note: In order to support equalized models, this function has a hacky fix in
    which we do not match any torch.mul operators. This is because equalized
    models contain extra mul operators to scale the input by the equalization
    scale, but this edge case has not been resolved yet within the numeric suite code.

    Args:
        model_a: A float model
        model_b: A quantized model
        x: Inputs to use during calibration
    """
    import torch.ao.ns._numeric_suite_fx as ns
    from torch.ao.ns.fx.mappings import get_unmatchable_types_map

    unmatchable_types_map = get_unmatchable_types_map()
    unmatchable_types_map["funs_unmatchable"].add(torch.mul)

    model_a_ns, model_b_ns = ns.add_loggers(
        "fp32",
        model_a,
        "int8",
        model_b,
        ns.OutputLogger,
        unmatchable_types_map=unmatchable_types_map,
    )

    model_a_ns(x)
    model_b_ns(x)

    activation_comparison_dict = ns.extract_logger_info(
        model_a_ns, model_b_ns, ns.OutputLogger, "int8"
    )
    ns.extend_logger_results_with_comparison(
        activation_comparison_dict,
        "fp32",
        "int8",
        torch.ao.ns.fx.utils.compute_sqnr,
        "sqnr",
    )

    # Construct a dictionary mapping layer names to the SQNR values
    layer_sqnr_dict = {}
    for key in activation_comparison_dict:
        layer = activation_comparison_dict[key]["node_output"]["int8"][0]["fqn"]
        sqnr = activation_comparison_dict[key]["node_output"]["int8"][0]["sqnr"][0]
        layer_sqnr_dict[layer] = sqnr

    return layer_sqnr_dict


def get_equalization_qconfig_dict(
    layer_sqnr_dict: dict[str, float], num_layers_to_equalize: int
) -> Any:
    """Given the layer to SQNR dictionary, find the layers with the highest
    quantization errors, and return an equalization_qconfig_dict
    specifying to only equalize those top layers.

    Args:
        layer_sqnr_dict: Dictionary mapping layer names to SQNR values (found
            when comparing an equalized model against a float model)
        num_layers_to_equalize: Number of layers with the highest quantization
           errors to equalize
    """

    # Sort the layer_sqnr_dictionary values and get the layers with the lowest
    # SQNR values (aka highest quantization errors)
    layer_sqnr_sorted = sorted(layer_sqnr_dict.items(), key=operator.itemgetter(1))
    layers_to_equalize = layer_sqnr_sorted[:num_layers_to_equalize]

    # Constructs an equalization_qconfig_dict that specifies to only equalize
    # the layers with the highest quantization errors
    module_to_qconfig_list = [
        (item[0], default_equalization_qconfig) for item in layers_to_equalize
    ]
    equalization_qconfig_dict = {"module_name": module_to_qconfig_list}
    return equalization_qconfig_dict
<|endoftext|># mypy: allow-untyped-defs
import copy
import functools
import operator
import warnings
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigAny, QuantType
from torch.ao.quantization.backend_config import DTypeWithConstraints
from torch.ao.quantization.fake_quantize import (
    FakeQuantizeBase,
    FixedQParamsFakeQuantize,
)
from torch.ao.quantization.observer import (
    _is_activation_post_process,
    FixedQParamsObserver,
    ObserverBase,
)
from torch.ao.quantization.qconfig import (
    float16_dynamic_qconfig,
    float16_static_qconfig,
    qconfig_equals,
)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
    _assert_and_get_unique_device,
    activation_is_statically_quantized,
)
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node

# importing the lib so that the quantized_decomposed ops are registered
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from .custom_config import PrepareCustomConfig


# TODO: revisit this list. Many helper methods shouldn't be public
__all__ = [
    "all_node_args_except_first",
    "all_node_args_have_no_tensors",
    "assert_and_get_unique_device",
    "collect_producer_nodes",
    "create_getattr_from_value",
    "create_node_from_old_node_preserve_meta",
    "EMPTY_ARG_DICT",
    "get_custom_module_class_keys",
    "get_linear_prepack_op_for_dtype",
    "get_new_attr_name_with_prefix",
    "get_non_observable_arg_indexes_and_types",
    "get_qconv_prepack_op",
    "get_skipped_module_name_and_classes",
    "graph_module_from_producer_nodes",
    "maybe_get_next_module",
    "NodeInfo",
    "node_arg_is_bias",
    "node_arg_is_weight",
