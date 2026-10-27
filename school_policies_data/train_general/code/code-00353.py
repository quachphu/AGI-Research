        # workspace_size should have already been retrieved prior to this call.
        kernel_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("size_t*")

        if node.get_workspace_size() > 0:
            ws = WorkspaceArg(
                count=node.get_workspace_size(),
                device=V.graph.get_current_device_or_throw(),
                zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                outer_name=WorkspaceArg.unique_name(),
            )
            wrapper.generate_workspace_allocation(ws)
            data_ptr = f"{ws.outer_name}.data_ptr()"
            kernel_args.append(
                data_ptr if V.graph.cpp_wrapper else f"c_void_p({data_ptr})"
            )
        else:
            ws = None
            kernel_args.append("nullptr" if V.graph.cpp_wrapper else "None")
        if V.graph.cpp_wrapper:
            arg_types.append("uint8_t*")
        wrapper.generate_kernel_call(
            name,
            kernel_args,
            triton=False,
            arg_types=arg_types,
        )
        if ws:
            wrapper.generate_workspace_deallocation(ws)


class ROCmTemplateCaller(ChoiceCaller):
    """
    ROCmTemplateCaller

    This class represents a caller for ROCm template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (ROCmBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ROCmTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Callable[
            [ROCmTemplateBuffer, Sequence[IRNode] | None], str
        ],
        bmreq: ROCmBenchmarkRequest,
        template: "ROCmTemplate",  # type: ignore[name-defined]
        info_kwargs: dict[str, PrimitiveInfoType | list[PrimitiveInfoType]] | None,  # type: ignore[type-arg]
    ) -> None:
        super().__init__(name, input_nodes, layout, description="")
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        if self.bmreq is None:
            raise AssertionError("expected self.bmreq to be set")
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        if self.bmreq is None:
            raise AssertionError("expected self.bmreq to be set")
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

    def __str__(self) -> str:
        return f"ROCmTemplateCaller(source_file={self.bmreq.source_file}, {self.info_dict()})"

    def call_name(self) -> str:
        return f"rocm_template_kernels.{self.name}"

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "ROCm",
            "name": self.name,
            **dict(self.info_kwargs["op"].dict_items()),  # type: ignore[union-attr, index]
        }

    def output_node(self) -> TensorBox:
        self.bmreq.update_workspace_size()
        buffer = ROCmTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            workspace_size=self.bmreq.workspace_size,
            template=self.template,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return TensorBox.create(buffer)
<|endoftext|># mypy: allow-untyped-defs
import copy
import logging
import random
from typing import Any
from typing_extensions import override

from torch._inductor.virtualized import V

from .rocm_template import ArgInfo


try:
    import ck4inductor  # type: ignore[import]
except ImportError:
    ck4inductor = None

if ck4inductor is not None:
    from ck4inductor.grouped_conv_fwd.gen_instances import (  # type: ignore[import]
        gen_conv_ops_library,
    )
    from ck4inductor.grouped_conv_fwd.op import (  # type: ignore[import]  # noqa: TCH002
        CKGroupedConvFwdOp,
    )
else:

    def gen_conv_ops_library():
        return []


from torch._inductor import config
from torch._inductor.codegen.rocm.ck_template import CKTemplate
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
from torch._inductor.utils import IndentedBuffer


log = logging.getLogger(__name__)


def torch_layout_to_ck_layouts(torch_layout):
    # logically, torch tensors are always NCHW,
    # and channels-last memory layout is visible in the strides
    if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
        # when input or output is NCHW
        # NB: torch.conv2d result is always NCHW
        return ["NGCHW", "GKCYX", "NGKHW"]
    elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-3], 1):
        # when input or output or weight is channels-last
        return ["NHWGC", "GKYXC", "NHWGK"]
    else:
        return None


def torch_layout_to_ck_input_layout(torch_layout):
    if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
        return "NGCHW"
    elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-3], 1):
        return "NHWGC"
    else:
        return None


def torch_layout_to_ck_weight_layout(torch_layout):
    if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
        return "GKCYX"
    elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-3], 1):
        return "GKYXC"
    else:
        return None


def torch_layout_to_ck_output_layout(torch_layout):
    if V.graph.sizevars.statically_known_equals(torch_layout.stride[-1], 1):
        return "NGKHW"
    elif V.graph.sizevars.statically_known_equals(torch_layout.stride[-3], 1):
        return "NHWGK"
    else:
        return None


class CKGroupedConvFwdTemplate(CKTemplate):
    conv_template = r"""
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {
        auto conv = {{instance_type}} {};
        auto invoker = conv.MakeInvoker();

        using ck::index_t;

        constexpr index_t NumDTensor = {{n_d_tensors}};
        constexpr index_t NDimSpatial = {{n_dim_spatial}};
        const std::vector<index_t> FilterSize = { FilterSize_0, FilterSize_1 };
        const std::vector<index_t> InputSize = { InputSize_0, InputSize_1 };
        const std::vector<index_t> ConvolutionStrides = { ConvolutionStrides_0, ConvolutionStrides_1 };
        const std::vector<index_t> Dilations = { Dilations_0, Dilations_1 };
        const std::vector<index_t> LeftPads = { LeftPads_0, LeftPads_1 };
        const std::vector<index_t> RightPads = { RightPads_0, RightPads_1 };


        auto conv_param = ck::utils::conv::ConvParam {
            NDimSpatial,
            GroupCount,
            NBatch,
            NOutChannels,
            NInChannels,
            FilterSize,
            InputSize,
            ConvolutionStrides,
            Dilations,
            LeftPads,
            RightPads,
        };

        using InLayout  = ck::tensor_layout::convolution::{{input_layout}};
        using WeiLayout = ck::tensor_layout::convolution::{{weight_layout}};
        using OutLayout = ck::tensor_layout::convolution::{{output_layout}};

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

        const void* p_a = input;
        const void* p_b = weight;
        const std::array<const void*, NumDTensor> p_ds;
        void* p_e = output;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_lengths;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_k_wos_strides;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides;
        std::array<index_t, NDimSpatial> conv_filter_strides;
        std::array<index_t, NDimSpatial> conv_filter_dilations;
        std::array<index_t, NDimSpatial> input_left_pads;
        std::array<index_t, NDimSpatial> input_right_pads;
        const auto a_element_op = PassThrough {};
        const auto b_element_op = PassThrough {};
        const auto cde_element_op = PassThrough {};

        auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
        copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        auto argument = conv.MakeArgument(
            p_a,
            p_b,
            p_ds,
            p_e,
            a_g_n_c_wis_lengths,
            a_g_n_c_wis_strides,
            b_g_k_c_xs_lengths,
            b_g_k_c_xs_strides,
            ds_g_n_k_wos_lengths,
            ds_g_n_k_wos_strides,
            e_g_n_k_wos_lengths,
            e_g_n_k_wos_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            a_element_op,
            b_element_op,
            cde_element_op
        );
        if (!conv.IsSupportedArgument(argument)) {
            // we do our best to statically avoid this case in `filter_op`
            std::cerr << "invalid argument for conv instance " << conv.GetTypeString() << std::endl;
            argument.Print();
            return -23;
        }
        if (workspace_size) {
            *workspace_size = conv.GetWorkSpaceSize(&argument);
            return 0;
        }

        if (p_a == nullptr) {
            std::cerr << "p_a is nullptr" << std::endl;
            return -1;
        }
        if (p_b == nullptr) {
            std::cerr << "p_b is nullptr" << std::endl;
            return -1;
        }
        if (p_e == nullptr) {
            std::cerr << "p_e is nullptr" << std::endl;
            return -1;
        }

        // when debugging, do time kernel to serialize launches
        auto stream_config = StreamConfig{stream, /* time kernel */ false, /* log level */ 0};

        if (workspace != nullptr) {
            conv.SetWorkSpacePointer(&argument, workspace, stream_config);
        }

        // run the kernel
        float elapsed_time = invoker.Run(argument, stream_config);
        return 0;
    } // kernel definition
    } // extern C

    #ifdef GENERATE_CK_STANDALONE_RUNNER
    int main(int argc, char** argv) {
        (void) argc;
        (void) argv;
        return 0;
    }
    #endif // GENERATE_CK_STANDALONE_RUNNER
"""

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK conv globals

                using NWC   = ck::tensor_layout::convolution::NWC;
                using NHWC  = ck::tensor_layout::convolution::NHWC;
                using NDHWC = ck::tensor_layout::convolution::NDHWC;

                using KXC   = ck::tensor_layout::convolution::KXC;
                using KYXC  = ck::tensor_layout::convolution::KYXC;
                using KZYXC = ck::tensor_layout::convolution::KZYXC;

                using NWK   = ck::tensor_layout::convolution::NWK;
                using NHWK  = ck::tensor_layout::convolution::NHWK;
                using NDHWK = ck::tensor_layout::convolution::NDHWK;

                using GNWC   = ck::tensor_layout::convolution::GNWC;
                using GNHWC  = ck::tensor_layout::convolution::GNHWC;
                using GNDHWC = ck::tensor_layout::convolution::GNDHWC;

                using GKXC   = ck::tensor_layout::convolution::GKXC;
                using GKYXC  = ck::tensor_layout::convolution::GKYXC;
                using GKZYXC = ck::tensor_layout::convolution::GKZYXC;

                using GKCX   = ck::tensor_layout::convolution::GKCX;
                using GKCYX  = ck::tensor_layout::convolution::GKCYX;
                using GKCZYX = ck::tensor_layout::convolution::GKCZYX;

                using GNWK   = ck::tensor_layout::convolution::GNWK;
                using GNHWK  = ck::tensor_layout::convolution::GNHWK;
                using GNDHWK = ck::tensor_layout::convolution::GNDHWK;

                using NGKW   = ck::tensor_layout::convolution::NGKW;
                using NGKHW  = ck::tensor_layout::convolution::NGKHW;
                using NGKDHW = ck::tensor_layout::convolution::NGKDHW;

                using NWGC   = ck::tensor_layout::convolution::NWGC;
                using NHWGC  = ck::tensor_layout::convolution::NHWGC;
                using NDHWGC = ck::tensor_layout::convolution::NDHWGC;

                using KXGC   = ck::tensor_layout::convolution::KXGC;
                using KYXGC  = ck::tensor_layout::convolution::KYXGC;
                using KZYXGC = ck::tensor_layout::convolution::KZYXGC;

                using NWGK   = ck::tensor_layout::convolution::NWGK;
                using NHWGK  = ck::tensor_layout::convolution::NHWGK;
                using NDHWGK = ck::tensor_layout::convolution::NDHWGK;

                using NGCW   = ck::tensor_layout::convolution::NGCW;
                using NGCHW  = ck::tensor_layout::convolution::NGCHW;
                using NGCDHW = ck::tensor_layout::convolution::NGCDHW;

                using G_K    = ck::tensor_layout::convolution::G_K;

                using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
                using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
                using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;

                using ConvolutionForwardSpecialization = ck::tensor_operation::device::ConvolutionForwardSpecialization;

                using OutElementOp = PassThrough;

                namespace ck {
                namespace utils {
                namespace conv {

                ConvParam::ConvParam(ck::index_t n_dim,
                                    ck::index_t group_count,
                                    ck::index_t n_batch,
                                    ck::index_t n_out_channels,
                                    ck::index_t n_in_channels,
                                    const std::vector<ck::index_t>& filters_len,
                                    const std::vector<ck::index_t>& input_len,
                                    const std::vector<ck::index_t>& strides,
                                    const std::vector<ck::index_t>& dilations,
                                    const std::vector<ck::index_t>& left_pads,
                                    const std::vector<ck::index_t>& right_pads)
                    : num_dim_spatial_(static_cast<ck::long_index_t>(n_dim)),
                    G_(static_cast<ck::long_index_t>(group_count)),
                    N_(static_cast<ck::long_index_t>(n_batch)),
                    K_(static_cast<ck::long_index_t>(n_out_channels)),
                    C_(static_cast<ck::long_index_t>(n_in_channels)),
                    filter_spatial_lengths_(num_dim_spatial_),
                    input_spatial_lengths_(num_dim_spatial_),
                    output_spatial_lengths_(num_dim_spatial_),
                    conv_filter_strides_(num_dim_spatial_),
                    conv_filter_dilations_(num_dim_spatial_),
                    input_left_pads_(num_dim_spatial_),
                    input_right_pads_(num_dim_spatial_)
                {
                    if(static_cast<ck::index_t>(filter_spatial_lengths_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(input_spatial_lengths_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(conv_filter_strides_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(conv_filter_dilations_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(input_left_pads_.size()) != num_dim_spatial_ ||
                    static_cast<ck::index_t>(input_right_pads_.size()) != num_dim_spatial_)
                    {
                        throw(
                            std::runtime_error("ConvParam::ConvParam: "
                                            "parameter size is different from number of declared dimensions!"));
                    }

                    for(ck::index_t i = 0; i < num_dim_spatial_; ++i)
                    {
                        filter_spatial_lengths_[i] = static_cast<ck::long_index_t>(filters_len[i]);
                        input_spatial_lengths_[i]  = static_cast<ck::long_index_t>(input_len[i]);
                        conv_filter_strides_[i]    = static_cast<ck::long_index_t>(strides[i]);
                        conv_filter_dilations_[i]  = static_cast<ck::long_index_t>(dilations[i]);
                        input_left_pads_[i]        = static_cast<ck::long_index_t>(left_pads[i]);
                        input_right_pads_[i]       = static_cast<ck::long_index_t>(right_pads[i]);

                        // XEff = (X - 1) * conv_dilation_w + 1;
                        // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
                        const ck::long_index_t x_eff =
                            (filter_spatial_lengths_[i] - 1) * conv_filter_dilations_[i] + 1;

                        output_spatial_lengths_[i] =
                            (input_spatial_lengths_[i] + input_left_pads_[i] + input_right_pads_[i] - x_eff) /
                                conv_filter_strides_[i] +
                            1;
                    }
                }

                } // namespace conv
                } // namespace utils
                } // namespace ck

                // NOTE: HostTensorDescriptor methods are declared in CK headers but defined
                // in CK's utility library. For architectural reasons, generated code doesn't
                // link with this library, so we provide local definitions here.
                // CalculateStrides is omitted as it became a template method in CK 4266f867.
                const std::vector<std::size_t>& ck::HostTensorDescriptor::GetLengths() const { return mLens; }
                const std::vector<std::size_t>& ck::HostTensorDescriptor::GetStrides() const { return mStrides; }
                std::size_t ck::HostTensorDescriptor::GetNumOfDimension() const { return mLens.size(); }
            """
        )
        return res

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK conv headers

                #include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
                #include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
                #include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

                #include "ck/library/utility/convolution_parameter.hpp"
                #include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
            """
        )
        return res

    @staticmethod
    def add_ck_conv_choices(
        choices,
        layout,
        input_nodes,
        *,
        stride,
        padding,
        dilation,
        groups,
        n_spatial_dimensions,
    ):
        template = CKGroupedConvFwdTemplate(
            input_nodes,
            layout,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            n_spatial_dimensions=n_spatial_dimensions,
        )
        ops = template.gen_ops()
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )

    def __init__(
        self,
        input_nodes,
        layout,
        *,
        stride,
        padding,
        dilation,
        groups,
        n_spatial_dimensions,
    ):
        super().__init__(
            "ck_conv_template",
            input_nodes,
            layout,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.n_spatial_dimensions = n_spatial_dimensions

    def filter_op(self, op: "CKGroupedConvFwdOp"):  # type: ignore[name-defined]
        metas = [
            T.get_layout()
            for T in [*self.input_nodes, self.output_node]
            if T is not None
        ]
        X_meta = metas[0]
        W_meta = metas[1]
        Y_meta = metas[-1]
        # disable the instance if dtypes don't match
        if op.a_element_dtype != self._TORCH_DTYPE_TO_CK[X_meta.dtype]:
            return None
        if op.b_element_dtype != self._TORCH_DTYPE_TO_CK[W_meta.dtype]:
            return None
        if op.e_element_dtype != self._TORCH_DTYPE_TO_CK[Y_meta.dtype]:
            return None
        # disable the instance if layouts don't match
        if op.a_layout != torch_layout_to_ck_input_layout(X_meta):
            return None
        if op.b_layout != torch_layout_to_ck_weight_layout(W_meta):
            return None
        if op.e_layout != torch_layout_to_ck_output_layout(Y_meta):
            return None
        # disable the instance if number of spatial dimensions doesn't match
        if op.n_dim_spatial != self.n_spatial_dimensions:
            return None
        # disable 1x1 and odd-channels conv specializations for now
        if "Default" not in op.conv_forward_specialization:
            return None
        if self.is_blocked_by_tf32_setting(op):
            return None
        return op

    def gen_ops(self):
        unfiltered_instances = gen_conv_ops_library()

        filtered_instances = list(
            filter(lambda op: self.filter_op(op), unfiltered_instances)
        )
        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Randomizing the choice seems to solve this.
        random.seed(-11)
        chosen_instances = (
            random.sample(
                filtered_instances,
                min(len(filtered_instances), config.rocm.ck_max_profiling_configs),
            )
            if config.rocm.ck_max_profiling_configs
            else filtered_instances
        )
        log.debug(
            "generated %d ck instances after filter: %s",
            len(chosen_instances),
            chosen_instances,
        )
        return chosen_instances

    def emit_ck_instance(self, op: "CKGroupedConvFwdOp") -> tuple[str, str]:  # type: ignore[name-defined]
        # The Jinja template for generating a C++ type alias *definition* for a Universal GEMM instance
        template_definition = r"""
    // Gemm operator {{operation_name}}
    using Operation_{{operation_name}} =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
            {{template_params}}>;

"""
        # The Jinja template for generating a C++ type alias *usage* for a Universal GEMM instance
        template_type = r"""
    Operation_{{operation_name}}
"""
        template_params = []
        for field_name, field_value in op.dict_items():
            if isinstance(field_value, tuple):
                tuple_elements = ", ".join(map(str, iter(field_value)))
                if "ds" in field_name:  # element type and layout for bias
                    arg = f"/* {field_name} */ Tuple<{tuple_elements}>"
                else:  # tile shape
                    arg = f"/* {field_name} */ S<{tuple_elements}>"
                # pyrefly: ignore [bad-argument-type]
                template_params.append(arg)
            else:
                if field_value is not None:
                    # pyrefly: ignore [bad-argument-type]
                    template_params.append(f"/* {field_name} */ {field_value}")
        return self._template_from_string(template_definition).render(
            operation_name=op.name(),
            template_params=(",\n" + 12 * " ").join(template_params),
        ), self._template_from_string(template_type).render(operation_name=op.name())

    def render(  # type: ignore[override]
        self,
        kernel: ROCmTemplateKernel,
        op: "CKGroupedConvFwdOp",  # type: ignore[name-defined]
        **kwargs,
    ) -> str:
        template_buffer_node = kwargs.get("template_buffer_node")
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = self.input_nodes[2] if 3 == len(self.input_nodes) else None

        op = copy.deepcopy(op)

        instance_definition, instance_type = self.emit_ck_instance(op)

        size_arg_strs = [
            "GroupCount",
            "NBatch",
            "NOutChannels",
            "NInChannels",
            "FilterSize_0",
            "FilterSize_1",
            "InputSize_0",
            "InputSize_1",
            "ConvolutionStrides_0",
            "ConvolutionStrides_1",
            "Dilations_0",
            "Dilations_1",
            "LeftPads_0",
            "LeftPads_1",
            "RightPads_0",
            "RightPads_1",
        ]

        return self._template_from_string(self.conv_template).render(
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            instance_type=instance_type,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W, Bias] if Bias is not None else [X, W],
                outputs=[Y],
                names_str="input, weight, bias, output"
                if Bias is not None
                else "input, weight, output",
                size_args=[f"int32_t {arg}" for arg in size_arg_strs],
            ),
            n_d_tensors=1 if Bias is not None else 0,
            n_dim_spatial=self.n_spatial_dimensions,
            input_layout=op.a_layout,
            weight_layout=op.b_layout,
            output_layout=op.e_layout,
        )

    def size_args(self):
        x, w = self.input_nodes[0], self.input_nodes[1]
        y = self.output_node

        group_count = self.groups
        n_batch = x.shape[0]  # type: ignore[index]
        n_out_channels = y.shape[1]  # type: ignore[index]
        n_in_channels = x.shape[1]  # type: ignore[index]

        filter_size_0, filter_size_1 = w.shape[2:4]  # type: ignore[index]
        input_size_0, input_size_1 = x.shape[2:4]  # type: ignore[index]
        convolution_strides_0, convolution_strides_1 = self.stride
        dilations_0, dilations_1 = self.dilation
        left_pads_0, left_pads_1 = self.padding
        right_pads_0, right_pads_1 = self.padding

        return (
            group_count,
            n_batch,
            n_out_channels,
            n_in_channels,
            filter_size_0,
            filter_size_1,
            input_size_0,
            input_size_1,
            convolution_strides_0,
            convolution_strides_1,
            dilations_0,
            dilations_1,
            left_pads_0,
            left_pads_1,
            right_pads_0,
            right_pads_1,
        )

    @override
    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return []

    @override
    def get_runtime_arg_values(self, **kwargs: Any) -> list[Any]:
        """
        Helper method to retrieve runtime args from generate kwargs
        """
        return []
<|endoftext|># mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from ...autotune_process import TensorMeta
from ...ir import Buffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .rocm_benchmark_request import ROCmBenchmarkRequest
from .rocm_kernel import ROCmTemplateCaller, ROCmTemplateKernel
from .rocm_template_buffer import ROCmTemplateBuffer
from .rocm_utils import DTYPE_TO_ROCM_TYPE


log = logging.getLogger(__name__)


# FIXME: unify with the CUDA version
@dataclass(frozen=True)
class ArgInfo:
    name: str
    ty: str


class ROCmTemplate(KernelTemplate):
    index_counter = itertools.count()
    gfx9_threads_per_warp = 64

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        input_reorder: list[int] | None = None,
    ) -> None:
        """

        Baseclass for ROCm C++ Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the ROCmTemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(  # type: ignore[override]
        self,
        **kwargs,
    ) -> ROCmTemplateCaller:
        """
        Generates the ROCm template caller object for the given GEMM template and operation. This ROCmTemplateCaller
        may be used to call and benchmark the generated ROCm kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A ROCmTemplateCaller object representing the generated ROCm template caller.
        """
        kernel_name = f"rocm_{self.name}"
        kernel_hash_name = f"rocm_{self.name}_{next(self.index_counter)}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)),
            ROCmTemplateKernel(
                kernel_name=kernel_name,
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            ) as kernel,
        ):
            code = self.render(kernel=kernel, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Autotune key: %s, Generated Code:\n%s", kernel_hash_name, code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(DTYPE_TO_ROCM_TYPE),
                kernel.args.python_argdefs(),
            )

        input_reorder = (
            self.input_reorder
            if self.input_reorder is not None
            else list(range(len(self.input_nodes)))
        )
        expected_args = list(
            unique(self.input_nodes[idx].get_name() for idx in input_reorder)
        )
        expected_args.extend([self.output_node.get_name()])
        if list(call_args)[: len(expected_args)] != expected_args:
            raise AssertionError(
                (
                    call_args,
                    expected_args,
                )
            )

        size_args = (
            self.size_args() if hasattr(self, "size_args") else ()
        )  # subclass should define def size_args()
        # Resolve symbolic sizes to concrete ints for benchmarking only.
        size_args_ints = list(V.graph.sizevars.optimization_hints(size_args))
        # The runtime args come right after the size args
        runtime_args = self.get_runtime_arg_values(**kwargs)
        extra_args = size_args_ints + runtime_args
        bmreq = ROCmBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: ROCmTemplateBuffer,
            epilogue_nodes: Sequence[IRNode] | None = None,
        ):
            kernel = ROCmTemplateKernel(
                kernel_name="KERNEL_NAME",
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            )
            return kernel, render

        return ROCmTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
            kwargs,
        )

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                #include <exception>
                #include <iostream>
                #include <memory>
                #include <random>
                #include <vector>
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                // We compile all models with -fvisibility=hidden. Any symbols that need to be
                // exposed in the final shared library must be declared with PT_EXPORT to make
                // them visible.
                #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
                #define PT_EXPORT __attribute__((__visibility__("default")))
                #else
                #ifdef _WIN32
                #define PT_EXPORT __declspec(dllexport)
                #else
                #define PT_EXPORT
                #endif
                #endif
            """
        )
        return res

    def render(self, **kwargs) -> str:
        raise NotImplementedError

    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return []

    def get_runtime_arg_values(self, **kwargs) -> list[Any]:
        return []
<|endoftext|># mypy: allow-untyped-defs, disable-error-code="attr-defined, valid-type"
import copy
import logging
import math
import random
from collections import namedtuple

import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.cpp_utils import DTYPE_TO_CPP
from torch._inductor.codegen.rocm.ck_template import CKTemplate
from torch._inductor.codegen.rocm.compile_command import rocm_compile_command
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
from torch._inductor.ir import Buffer, Layout
from torch._inductor.runtime.runtime_utils import next_power_of_2

from ...utils import IndentedBuffer, is_dynamic, try_import_ck_lib


_, gen_ops_library, gen_ops_preselected, CKGemmOperation = try_import_ck_lib()


log = logging.getLogger(__name__)

# lightweight collection of information about a single op
InductorROCmOp = namedtuple("InductorROCmOp", ["op", "kBatch"])

padding_lookup = {
    "M": {
        "GemmSpecialization::MPadding": True,
        "GemmSpecialization::MNPadding": True,
        "GemmSpecialization::MKPadding": True,
        "GemmSpecialization::MNKPadding": True,
    },
    "N": {
        "GemmSpecialization::NPadding": True,
        "GemmSpecialization::MNPadding": True,
        "GemmSpecialization::NKPadding": True,
        "GemmSpecialization::MNKPadding": True,
    },
    "K": {
        "GemmSpecialization::KPadding": True,
        "GemmSpecialization::MKPadding": True,
        "GemmSpecialization::NKPadding": True,
        "GemmSpecialization::MNKPadding": True,
    },
}


def is_static_int(number):
    return isinstance(number, (int, sympy.Integer))


def torch_layout_to_ck_layout(torch_layout):
    if torch_layout.stride[-1] == 1:
        return "Row"
    elif torch_layout.stride[-2] == 1:
        return "Col"
    else:
        return None


class CKGemmTemplate(CKTemplate):
    # the JINJA template for rendering CK Universal GEMMs
    gemm_template = r"""{{version_comment}}
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {
        auto gemm = {{instance_type}} {};
        auto invoker = gemm.MakeInvoker();
        {% if is_batched %}
        auto argument = gemm.MakeArgument(
            reinterpret_cast<const {{a_element_dtype}}*>(X),
            reinterpret_cast<const {{b_element_dtype}}*>(W),
            std::array<const void*, {{ds_size}}>{ {{ds_names}} },
            reinterpret_cast<{{c_element_dtype}}*>(Y),
            M,
            N,
            K,
            B,
            LDA,
            LDB,
            std::array<ck::index_t, {{ds_size}}>{ {{ds_strides}} },
            LDC,
            M * K, // batch_stride_A
            N * K, // batch_stride_B
            std::array<ck::index_t, {{ds_size}}>{ {{ds_batch_strides}} },
            M * N, // batch_stride_C
            {{a_elementwise_op}},
            {{b_elementwise_op}},
            {{epilogue}} // c_elementwise_op
        );
        {% else %}
        auto argument = gemm.MakeArgument(
            reinterpret_cast<const {{a_element_dtype}}*>(X),
            reinterpret_cast<const {{b_element_dtype}}*>(W),
            std::array<const void*, {{ds_size}}>{ {{ds_names}} },
            reinterpret_cast<{{c_element_dtype}}*>(Y),
            M,
            N,
            K,
            LDA,
            LDB,
            std::array<ck::index_t, {{ds_size}}>{ {{ds_strides}} },
            LDC,
            kBatch, // kBatch
            {{a_elementwise_op}},
            {{b_elementwise_op}},
            {{epilogue}} // c_elementwise_op
        );
        {% endif %}
        if (!gemm.IsSupportedArgument(argument)) {
            // we do our best to statically avoid this case in `filter_op`
            std::cerr << "invalid argument for gemm instance " << gemm.GetTypeString() << std::endl;
            argument.Print();
            return -23;
        }
        if (workspace_size) {
            *workspace_size = gemm.GetWorkSpaceSize(&argument);
            return 0;
        }
        // run the kernel
        #ifdef GENERATE_CK_STANDALONE_RUNNER
        const auto stream_config = StreamConfig{
            stream,
            /* time kernel */ 1,
            /* log level */ 1,
            /* n_cold_iter */ 100,
            /* n_hot_iter */ 100,
            /* flush_l2_cache */ 1,
            /* rotate_count */ 5};
        #else
        const auto stream_config = StreamConfig{stream, /* time kernel */ false, /* log level */ 0};
        #endif

        const float elapsed_time = invoker.Run(argument, stream_config);

        #ifdef GENERATE_CK_STANDALONE_RUNNER
        std::cout << "elapsed time: " << elapsed_time << " ms" << std::endl;
        #else
        (void)elapsed_time;
        #endif
        return 0;
    } // kernel definition
    } // extern C
    """

    standalone_runner_template = r"""
    #ifdef GENERATE_CK_STANDALONE_RUNNER
    // standalone runner for the generated CK GEMM kernel

    {{inline_utils}}

    extern "C" {
    int run_main(int argc, char** argv) {
        {% if is_batched %}
        const int32_t B = {{B}};
        {% endif %}
        const int32_t M = {{M}};
        const int32_t N = {{N}};
        const int32_t K = {{K}};
        const int32_t LDA = {{LDA}};
        const int32_t LDB = {{LDB}};
        const int32_t LDC = {{LDC}};
        const int32_t LDD = {{LDD}};
        const int32_t kBatch = {{kBatch}};

        using AElementType = {{a_ck_dtype}};
        using BElementType = {{b_ck_dtype}};
        using CElementType = {{c_ck_dtype}};
        {% if has_bias %}
        using BiasElementType = {{bias_ck_dtype}};
        {% endif %}
        {% if has_scale %}
        using ScaleAElementType = {{scale_a_ck_dtype}};
        using ScaleBElementType = {{scale_b_ck_dtype}};
        {% endif %}

        using AArgType = {{a_torch_dtype}};
        using BArgType = {{b_torch_dtype}};
        using CArgType = {{c_torch_dtype}};
        {% if has_bias %}
        using BiasArgType = {{bias_torch_dtype}};
        {% endif %}
        {% if has_scale %}
        using ScaleAArgType = {{scale_a_torch_dtype}};
        using ScaleBArgType = {{scale_b_torch_dtype}};
        {% endif %}

        using ALayout = {{a_layout}};
        using BLayout = {{b_layout}};
        using CLayout = {{c_layout}};
        {% if has_bias %}
        using BiasLayout = {{bias_layout}};
        {% endif %}

        {% if is_batched %}
        using strides_t = std::array<int32_t, 3>;
        auto get_strides = [](int32_t batch_stride, int32_t leading_dimension, auto layout) constexpr -> strides_t {
            if constexpr (std::is_same_v<decltype(layout), Row>) {
                return {batch_stride, leading_dimension, 1};
            }
            return {batch_stride, 1, leading_dimension};
        };
        auto a_size = strides_t{B, M, K};
        auto a_stride = get_strides(M * K, LDA, ALayout{});
        auto b_size = strides_t{B, N, K};
        auto b_stride = get_strides(N * K, LDB, BLayout{});
        auto c_size = strides_t{B, M, N};
        auto c_stride = get_strides(M * N, LDC, CLayout{});
        {% else %}
        using strides_t = std::array<int32_t, 2>;
        auto get_strides = [](int32_t leading_dimension, auto layout) constexpr -> strides_t {
            if constexpr (std::is_same_v<decltype(layout), Row>) {
                return {leading_dimension, 1};
            }
            return {1, leading_dimension};
        };
        auto a_size = strides_t{M, K};
        auto a_stride = get_strides(LDA, ALayout{});
        auto b_size = strides_t{N, K};
        auto b_stride = get_strides(LDB, BLayout{});
        auto c_size = strides_t{M, N};
        auto c_stride = get_strides(LDC, CLayout{});
        {% endif %}

        Tensor<AElementType> a_m_k ( HostTensorDescriptor ( a_size, a_stride ) );
        Tensor<BElementType> b_k_n ( HostTensorDescriptor ( b_size, b_stride ) );
        {% if has_bias %}
        Tensor<BiasElementType> d_m_n ( HostTensorDescriptor ( c_size, get_strides(LDD, BiasLayout{}) ) );
        {% endif %}
        {% if has_scale %}
        // NB: these are hardcoded
        Tensor<ScaleAElementType> s_a_m_n ( HostTensorDescriptor ( strides_t{M, N}, get_strides(0, Row{}) ));
        Tensor<ScaleAElementType> s_b_m_n ( HostTensorDescriptor ( strides_t{M, N}, get_strides(0, Col{}) ));
        {% endif %}

        Tensor<CElementType> c_m_n_host ( HostTensorDescriptor ( c_size, c_stride ) );
        Tensor<CElementType> c_m_n_device ( HostTensorDescriptor ( c_size, c_stride ) );

        a_m_k.GenerateTensorValue(GeneratorTensor_2<AElementType>());
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BElementType>());
        {% if has_bias %}
        d_m_n.GenerateTensorValue(GeneratorTensor_2<BiasElementType>());
        {% endif %}
        {% if has_scale %}
        s_a_m_n.GenerateTensorValue(GeneratorTensor_2<ScaleAElementType>());
        s_b_m_n.GenerateTensorValue(GeneratorTensor_2<ScaleBElementType>());
        {% endif %}
        DeviceMem a_m_k_device_buf(sizeof(AElementType) * a_m_k.mDesc.GetElementSpaceSize());
        DeviceMem b_k_n_device_buf(sizeof(BElementType) * b_k_n.mDesc.GetElementSpaceSize());
        {% if has_bias %}
        DeviceMem d_m_n_device_buf(sizeof(BiasElementType) * d_m_n.mDesc.GetElementSpaceSize());
        {% endif %}
        {% if has_scale %}
        DeviceMem s_a_m_n_device_buf(sizeof(ScaleAElementType) * s_a_m_n.mDesc.GetElementSpaceSize());
        DeviceMem s_b_m_n_device_buf(sizeof(ScaleBElementType) * s_b_m_n.mDesc.GetElementSpaceSize());
        {% endif %}
        DeviceMem c_m_n_device_buf(sizeof(CElementType) * c_m_n_device.mDesc.GetElementSpaceSize());

        a_m_k_device_buf.ToDevice(a_m_k.mData.data());
        b_k_n_device_buf.ToDevice(b_k_n.mData.data());
        {% if has_bias %}
        d_m_n_device_buf.ToDevice(d_m_n.mData.data());
        {% endif %}
        {% if has_scale %}
        s_a_m_n_device_buf.ToDevice(s_a_m_n.mData.data());
        s_b_m_n_device_buf.ToDevice(s_b_m_n.mData.data());
        {% endif %}

        {{kernel_name}}(
            static_cast<const AArgType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<const BArgType*>(b_k_n_device_buf.GetDeviceBuffer()),
            {% if has_scale %}
            static_cast<const ScaleAArgType*>(s_a_m_n_device_buf.GetDeviceBuffer()),
            static_cast<const ScaleBArgType*>(s_b_m_n_device_buf.GetDeviceBuffer()),
            {% endif %}
            {% if has_bias %}
            static_cast<const BiasArgType*>(d_m_n_device_buf.GetDeviceBuffer()),
            {% endif %}
            static_cast<CArgType*>(c_m_n_device_buf.GetDeviceBuffer()),
            {% if is_batched %}
            B,
            {% endif %}
            M,
            N,
            K,
            LDA,
            LDB,
            LDC,
            LDD,
            nullptr, // workspace_size
            nullptr, // workspace
            nullptr); // stream

        hip_check_error(hipDeviceSynchronize());

        return 0;
    } // run_main
    } // extern C

    int main(int argc, char** argv) {
        return run_main(argc, argv);
    }
    // compile with: {{compile_cmd}}
    #endif // GENERATE_CK_STANDALONE_RUNNER
    """

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: list[int] | None = None,
    ) -> None:
        is_batched = len(layout.size) == 3
        name = "ck_batched_gemm_template" if is_batched else "ck_gemm_template"
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            input_reorder=input_reorder,
        )
        self.alpha = alpha
        self.beta = beta
        self.is_batched = is_batched

    def header(self) -> IndentedBuffer:
        res = super().header()
        if self.is_batched:
            res.splice(
                """
                    // CK GEMM header(s)

                    #include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp"
                """
            )
        else:
            res.splice(
                """
                    // CK GEMM header(s)

                    #include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
                """
            )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK GEMM globals

                using Row = ck::tensor_layout::gemm::RowMajor;
                using Col = ck::tensor_layout::gemm::ColumnMajor;

                using BlockGemmPipelineScheduler = ck::BlockGemmPipelineScheduler;
                using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;
                using BlockGemmPipelineVersion = ck::BlockGemmPipelineVersion;

                struct MultiplyMultiplyAdd {
                    template <typename E, typename C, typename D0, typename D1, typename D2>
                    __host__ __device__ constexpr void
                    operator()(E& e, const C& c, const D0& d0, const D1& d1, const D2& d2) const {
                        e = ck::type_convert<E>(
                           ck::type_convert<float>(c)
                           * ck::type_convert<float>(d0)
                           * ck::type_convert<float>(d1)
                           + ck::type_convert<float>(d2)
                        );
                    }
                };
            """
        )
        return res

    def inline_utils(self):
        res = IndentedBuffer()
        res.splice(
            """
                #include "host_tensor.cpp"
                #include "device_memory.cpp"
            """
        )
        return res

    def _has_padding(self, dimension, gemm_specialization):
        # Get the relevant padding map for the given dimension
        dimension_padding = padding_lookup.get(dimension, {})

        # Check if the specialization is in the dimension's padding map
        return dimension_padding.get(gemm_specialization, False)

    def filter_op(self, op_info: InductorROCmOp):
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """
        op, kBatch = op_info.op, op_info.kBatch
        metas = [T.get_layout() for T in [*self.input_nodes, self.output_node]]
        X_meta = metas[0]
        W_meta = metas[1]
        Y_meta = metas[-1]
        # disable the instance if dtypes don't match
        if op.a_element_dtype != self._TORCH_DTYPE_TO_CK[X_meta.dtype]:
            return None
        if op.b_element_dtype != self._TORCH_DTYPE_TO_CK[W_meta.dtype]:
            return None
        if op.c_element_dtype != self._TORCH_DTYPE_TO_CK[Y_meta.dtype]:
            return None
        if self.is_blocked_by_tf32_setting(op):
            return None
        # disable the instance if layouts don't match
        if op.a_layout != torch_layout_to_ck_layout(X_meta):
            return None
        if op.b_layout != torch_layout_to_ck_layout(W_meta):
            return None
        if op.c_layout != torch_layout_to_ck_layout(Y_meta):
            return None
        # try to avoid launching the instance with invalid problem size
        # see GridwiseGemm_xdl_cshuffle_v3::CheckValidity

        M = X_meta.size[-2]
        K = X_meta.size[-1]
        N = W_meta.size[-1]

        if is_static_int(M):
            if not self._has_padding("M", op.gemm_specialization):
                if M % op.m_per_block != 0:
                    return None
        if is_static_int(N):
            if not self._has_padding("N", op.gemm_specialization):
                if N % op.n_per_block != 0:
                    return None
        if is_static_int(K):
            if not self._has_padding("K", op.gemm_specialization):
                if K % op.k_per_block != 0:
                    return None
                K_t = kBatch * op.k_per_block
                if K % K_t != 0:
                    return None
            else:
                # need another kBatch check here
                lcm = abs(op.a_k1 * op.b_k1) // math.gcd(op.a_k1, op.b_k1)
                K_t = kBatch * lcm
                k_read_pad_splited = math.ceil(K / K_t) * lcm
                if (k_read_pad_splited * (kBatch - 1)) >= K:
                    return None

        a_contig_size = (
            K if op.a_layout == "Row" else M if op.a_layout == "Col" else None
        )
        if (
            is_static_int(a_contig_size)
            and a_contig_size % op.a_block_transfer_src_scalar_per_vector != 0
        ):
            return None
        b_contig_size = (
            N if op.b_layout == "Row" else K if op.b_layout == "Col" else None
        )
        if (
            is_static_int(b_contig_size)
            and b_contig_size % op.b_block_transfer_src_scalar_per_vector != 0
        ):
            return None
        c_contig_size = (
            N if op.c_layout == "Row" else M if op.c_layout == "Col" else None
        )
        c_shuffle_block_transfer_scalar_per_vector_n_per_block = (
            op.c_shuffle_block_transfer_scalar_per_vector_n_per_block[0]
            if isinstance(
                op.c_shuffle_block_transfer_scalar_per_vector_n_per_block, tuple
            )
            else op.c_shuffle_block_transfer_scalar_per_vector_n_per_block
        )
        if (
            is_static_int(c_contig_size)
            and c_contig_size % c_shuffle_block_transfer_scalar_per_vector_n_per_block
            != 0
        ):
            return None
        if not self._check_num_k_loops(op, kBatch):
            return None
        # TBD disable instances with invalid number of pipeline prefetch stages
        # It will avoid compiling a small percentage of unrunnable instances which fail the gemm argument check

        return op

    def _check_num_k_loops(self, op, kBatch):
        # Additional splitK scenario check
        metas = [T.get_layout() for T in [*self.input_nodes]]
        X_meta = metas[0]
        W_meta = metas[1]
        K = X_meta.size[-1]
        if kBatch > 1:
            if op.block_gemm_pipeline_version != "BlockGemmPipelineVersion::v1":
                try:
                    prefetch_stages = self._prefetch_stages(
                        op,
                        torch.empty((), dtype=X_meta.dtype).element_size(),
                        torch.empty((), dtype=W_meta.dtype).element_size(),
                        torch.cuda.get_device_properties(X_meta.device).warp_size,
                    )
                except Exception as e:
                    log.debug(
                        "Failed to prefetch_stages for %s with exception %s", op.name, e
                    )
                    # be conservative here and disable the op
                    return False

                K_t = op.k_per_block * kBatch
                ak0 = (K + K_t - 1) // K_t * (op.k_per_block // op.a_k1)
                num_k_loop = ak0 // (op.k_per_block // op.a_k1)
                if num_k_loop <= prefetch_stages:
                    log.debug(
                        "Op %s is not compatible due to invalid number of pipeline prefetch stages. "
                        "Parameters: kBatch=%s, block_gemm_pipeline_version=%s, prefetch_stages=%s, num_k_loop=%s",
                        op.name(),
                        kBatch,
                        op.block_gemm_pipeline_version,
                        prefetch_stages,
                        num_k_loop,
                    )
                    return False

        return True

    # small helper to figure out the prefetch stages on AMD
    def _prefetch_stages(self, op, a_dtype_size, b_dtype_size, warp_size: int = 64):
        version_str = op.block_gemm_pipeline_version.split("::")[-1]
        try:
            version = int(version_str[1:])  # Assuming the format is always 'vX'
        except ValueError as e:
            raise ValueError(f"Invalid version string: {version_str}") from e
        if version not in [1, 2, 3, 4, 5]:
            raise ValueError(
                f"unknown prefetch stages for {op.block_gemm_pipeline_version}"
            )
        # Define the mapping of versions to stages
        version_to_stages = {1: 1, 3: 2, 4: 4, 5: 3}
        # Get the stages for the given version
        stages = version_to_stages.get(version)
        if stages is None:
            # This means we're at stage 2, and this requires computation
            # See github.com/ROCm/composable_kernel/blob/d6a4605/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp#L143
            wgp_per_cu = max(4 * warp_size // op.block_size, 1)
            full_mem_band_prefetch_stages = math.ceil(
                32768
                / wgp_per_cu
                / (
                    (op.m_per_block * a_dtype_size + op.n_per_block * b_dtype_size)
                    * op.k_per_block
                )
            )
            stages = min(max(full_mem_band_prefetch_stages, 2), 8)

        return stages

    def emit_ck_instance(self, op: "CKGemmOperation"):
        # The Jinja template for generating a C++ type alias *definition* for a Universal GEMM instance
        struct_name = (
            "DeviceBatchedGemmMultiD_Xdl_CShuffle_V3"
            if self.is_batched
            else "DeviceGemmMultiD_Xdl_CShuffle_V3"
        )
        template_definition = r"""
    // Gemm operator {{operation_name}}
    using Operation_{{operation_name}} =
        ck::tensor_operation::device::{{struct_name}}<
            {{template_params}}>;

"""
        # The Jinja template for generating a C++ type alias *usage* for a Universal GEMM instance
        template_type = r"""
    Operation_{{operation_name}}
"""
        template_params = []
        for field_name, field_value in op.dict_items():
            if isinstance(field_value, tuple):
                tuple_elements = ", ".join(map(str, iter(field_value)))
                if "ds" in field_name:  # element type and layout for bias
                    arg = f"/* {field_name} */ Tuple<{tuple_elements}>"
                else:  # tile shape
                    arg = f"/* {field_name} */ S<{tuple_elements}>"
                # pyrefly: ignore [bad-argument-type]
                template_params.append(arg)
            else:
                if field_value is not None:
                    # pyrefly: ignore [bad-argument-type]
                    template_params.append(f"/* {field_name} */ {field_value}")
        operation_name = op.name().replace("(", "").replace(",", "").replace(")", "")
        return self._template_from_string(template_definition).render(
            operation_name=operation_name,
            template_params=(",\n" + 12 * " ").join(template_params),
            struct_name=struct_name,
        ), self._template_from_string(template_type).render(
            operation_name=operation_name
        )

    def render(  # type: ignore[override]
        self,
        kernel: ROCmTemplateKernel,
        op: "CKGemmOperation",
        **kwargs,
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        """
        epilogue_nodes = kwargs.get("epilogue_nodes")
        if not (epilogue_nodes is None or 0 == len(epilogue_nodes)):
            raise AssertionError("expected no epilogue nodes for CK GEMM template")
        template_buffer_node = kwargs.get("template_buffer_node")
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        # input nodes:
        # * X, W for matmul
        # * X, W, Bias for addmm
        # * X, W, inv_scale_x, inv_scale_w for scaled_mm
        # * X, W, inv_scale_x, inv_scale_w, Bias for scaled_mm with bias
        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = (
            self.input_nodes[2]
            if 3 == len(self.input_nodes)
            else self.input_nodes[4]
            if 5 == len(self.input_nodes)
            else None
        )
        has_bias = Bias is not None
        has_scale = len(self.input_nodes) in (4, 5)
        op = copy.deepcopy(op)

        # This parameter is converted into tuple because of change
        # from DeviceGemm_Xdl_CShuffleV3 to DeviceGemmMultiD_Xdl_CShuffle_V3.
        # The first tuple element corresponds to matmul result...
        if not isinstance(
            op.c_shuffle_block_transfer_scalar_per_vector_n_per_block, tuple
        ):
            op.c_shuffle_block_transfer_scalar_per_vector_n_per_block = (
                op.c_shuffle_block_transfer_scalar_per_vector_n_per_block,
            )

        if has_scale:
            scale_x = self.input_nodes[2]
            scale_w = self.input_nodes[3]
            if 1 == scale_x.get_numel() and 1 == scale_w.get_numel():
                # tensorwise scale for both X, W
                if has_bias:
                    op.c_elementwise_op = "ScaleAdd"
                else:
                    op.c_elementwise_op = "Scale"
            else:
                # rowwise scale for both X, W
                if has_bias:
                    op.c_elementwise_op = "MultiplyMultiplyAdd"
                else:
                    op.c_elementwise_op = "MultiplyMultiply"
                op.c_shuffle_dtype = "F32"
                op.ds_layouts = (
                    torch_layout_to_ck_layout(scale_x.get_layout()),
                    torch_layout_to_ck_layout(scale_w.get_layout()),
                )
                op.ds_element_dtypes = (
                    self._TORCH_DTYPE_TO_CK[scale_x.get_layout().dtype],
                    self._TORCH_DTYPE_TO_CK[scale_w.get_layout().dtype],
                )
                op.c_shuffle_block_transfer_scalar_per_vector_n_per_block += (1, 1)
        else:
            scale_x = None
            scale_w = None

        bias_dtype = ""
        if Bias is not None:
            bias_layout = torch_layout_to_ck_layout(Bias.get_layout())
            bias_dtype = self._TORCH_DTYPE_TO_CK[Bias.get_layout().dtype]
            op.ds_layouts += (bias_layout,)
            op.ds_element_dtypes += (bias_dtype,)
            if not has_scale:
                op.c_elementwise_op = "Bilinear"
            # c_shuffle_dtype is also used for adding bias to matmul result
            # before converting down to the result dtype
            op.c_shuffle_dtype = op.acc_dtype
            # this parameter needs to be set accordingly to bias stride for correct accumulation
            if bias_layout == "Row":
                # bias has (N, ) shape
                bias_shuffle_block_transfer_scalar_per_vector_n_per_block = (
                    op.c_shuffle_block_transfer_scalar_per_vector_n_per_block
                )
            elif bias_layout == "Col":
                # bias has (M, 1) shape
                bias_shuffle_block_transfer_scalar_per_vector_n_per_block = (1,)
            else:
                raise AssertionError(
                    "Bias layout is neither row-major nor column-major"
                )
            # ...and the second tuple element corresponds to the bias
            op.c_shuffle_block_transfer_scalar_per_vector_n_per_block += (
                bias_shuffle_block_transfer_scalar_per_vector_n_per_block
            )

        instance_definition, instance_type = self.emit_ck_instance(op)

        version_comment = rf"""/**
* Generated code for CK inductor backend
* See {type(self).__module__}.{type(self).__qualname__}
*
* Template instance {op}
*
* {torch.__version__=}
* torch.version.git_version={getattr(torch.version, "git_version", "None")}
*/
"""
        epilogue = None

        if op.c_elementwise_op == "Bilinear" and scale_w is None:
            epilogue = f"Bilinear {{ {self.alpha}, {self.beta} }}"

        elif op.c_elementwise_op == "Scale":
            epilogue = "Scale { (inv_scale_w && inv_scale_x) ? (*inv_scale_w * *inv_scale_x) : 1.0f }"

        elif op.c_elementwise_op == "ScaleAdd":
            epilogue = "ScaleAdd { (inv_scale_w && inv_scale_x) ? (*inv_scale_w * *inv_scale_x) : 1.0f }"

        elif op.c_elementwise_op == "MultiplyMultiply":
            epilogue = "MultiplyMultiply {}"

        elif op.c_elementwise_op == "MultiplyMultiplyAdd":
            epilogue = "MultiplyMultiplyAdd {}"

        elif op.c_elementwise_op == "PassThrough":
            epilogue = "PassThrough {}"

        if epilogue is None:
            raise AssertionError("CK GEMM epilogue is not set")

        size_arg_strs = ["M", "N", "K", "LDA", "LDB", "LDC", "LDD"]
        if self.is_batched:
            size_arg_strs.insert(0, "B")

        res = self._template_from_string(self.gemm_template).render(
            inline_utils=self.inline_utils(),
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W, scale_x, scale_w, Bias],  # type: ignore[list-item]
                outputs=[Y],
                names_str="X, W, inv_scale_x, inv_scale_w, Bias, Y",
                input_reorder=self.input_reorder,
                size_args=[f"int32_t {arg}" for arg in size_arg_strs],
            ),
            instance_type=instance_type,
            a_element_dtype=op.a_element_dtype,
            b_element_dtype=op.b_element_dtype,
            c_element_dtype=op.c_element_dtype,
            bias_element_dtype=bias_dtype,
            alpha=self.alpha,
            beta=self.beta,
            a_elementwise_op="PassThrough {}",
            b_elementwise_op="PassThrough {}",
            epilogue=epilogue,
            has_bias=has_bias,
            ds_size=1
            if op.c_elementwise_op in ("Bilinear", "ScaleAdd")
            else 2
            if op.c_elementwise_op == "MultiplyMultiply"
            else 3
            if op.c_elementwise_op == "MultiplyMultiplyAdd"
            else 0,
            ds_names=", ".join(
                ["Bias"]
                if op.c_elementwise_op in ("Bilinear", "ScaleAdd")
                else ["inv_scale_x", "inv_scale_w"]
                if op.c_elementwise_op == "MultiplyMultiply"
                else ["inv_scale_x", "inv_scale_w", "Bias"]
                if op.c_elementwise_op == "MultiplyMultiplyAdd"
                else []
            ),
            ds_strides=", ".join(
                ["LDD"]
                if op.c_elementwise_op in ("Bilinear", "ScaleAdd")
                else ["0", "0"]
                if op.c_elementwise_op == "MultiplyMultiply"
                else ["0", "0", "LDD"]
                if op.c_elementwise_op == "MultiplyMultiplyAdd"
                else []
            ),
            version_comment=version_comment,
            is_batched=self.is_batched,
            ds_batch_strides=", ".join([]),  # FIXME when supporting baddbmm
        )

        if config.rocm.generate_test_runner:
            is_static_problem = all(is_static_int(arg) for arg in self.size_args())
            # NOTE: size_arg_strs is defined above
            size_arg_vals = (
                self.size_args()
                if is_static_problem
                else (
                    f"std::stoi(argv[{k}])" for k, _ in enumerate(self.size_args(), 1)
                )
            )
            size_args = dict(zip(size_arg_strs, size_arg_vals, strict=True))
            runtime_args = dict(
                zip(
                    [a.name for a in self.get_runtime_arg_info()],
                    self.get_runtime_arg_values(),
                )
            )
            runner_code = self._template_from_string(
                self.standalone_runner_template
            ).render(
                inline_utils=self.inline_utils().getvalue(),
                kernel_name=kernel.kernel_name,
                has_bias=has_bias,
                has_scale=has_scale,
                is_batched=self.is_batched,
                a_ck_dtype=op.a_element_dtype,
                b_ck_dtype=op.b_element_dtype,
                c_ck_dtype=op.c_element_dtype,
                bias_ck_dtype=op.ds_element_dtypes[0] if has_bias else "",
                scale_a_ck_dtype=op.ds_element_dtypes[0]
                if has_scale and 2 == len(op.ds_element_dtypes)
                else "BF16",
                scale_b_ck_dtype=op.ds_element_dtypes[1]
                if has_scale and 2 == len(op.ds_element_dtypes)
                else "BF16",
                a_torch_dtype=DTYPE_TO_CPP[X.get_layout().dtype],
                b_torch_dtype=DTYPE_TO_CPP[W.get_layout().dtype],
                c_torch_dtype=DTYPE_TO_CPP[Y.get_layout().dtype],
                bias_torch_dtype=DTYPE_TO_CPP[Bias.get_layout().dtype]
                if Bias is not None
                else "",
                scale_a_torch_dtype=DTYPE_TO_CPP[scale_x.get_layout().dtype]
                if scale_x is not None
                else "",
                scale_b_torch_dtype=DTYPE_TO_CPP[scale_w.get_layout().dtype]
                if scale_w is not None
                else "",
                a_layout=torch_layout_to_ck_layout(X.get_layout()),
                b_layout=torch_layout_to_ck_layout(W.get_layout()),
                c_layout=torch_layout_to_ck_layout(Y.get_layout()),
                bias_layout=torch_layout_to_ck_layout(Bias.get_layout())
                if Bias is not None
                else "",
                compile_cmd=rocm_compile_command(
                    ["<source_file_name>"], "<executable_name>", "exe"
                ),
                **size_args,
                **runtime_args,
            )
            res += runner_code

        return res

    def _is_rcr_f16(self):
        X_meta, W_meta, Y_meta = (
            T.get_layout() for T in [*self.input_nodes, self.output_node]
        )
        X_dtype, W_dtype, Y_dtype = (
            self._TORCH_DTYPE_TO_CK[m.dtype] for m in (X_meta, W_meta, Y_meta)
        )
        X_layout, W_layout, Y_layout = (
            torch_layout_to_ck_layout(m) for m in (X_meta, W_meta, Y_meta)
        )

        return (
            X_dtype == "F16"
            and W_dtype == "F16"
            and Y_dtype == "F16"
            and X_layout == "Row"
            and W_layout == "Col"
            and Y_layout == "Row"
        )

    # helper to calculate a potentially optimal kBatch(es) for a problem
    def _get_kBatch(self, op):
        # we only set a higher kBatch if K > 16 * the larger of M and N
        # this is a hand-tuned heuristic to start
        metas = [T.get_layout() for T in [*self.input_nodes]]
        X_meta = metas[0]
        W_meta = metas[1]
        M = X_meta.size[-2]
        K = X_meta.size[-1]
        N = W_meta.size[-1]
        if is_dynamic(*self.input_nodes):
            return [1]
        if K // max(M, N) < config.rocm.split_k_threshold:
            return [1]
        # if the user is telling us which kBatches to sweep, just use those
        if config.rocm.kBatch_sweep is not None:
            return config.rocm.kBatch_sweep
        # Calculate the number of blocks needed for each dimension
        total_k_blocks = math.ceil(K / op.k_per_block)
        # we want to calculate how many blocks we need to fit per CU
        cus = torch.cuda.get_device_properties(X_meta.device).multi_processor_count
        # again, manual heuristics as much larger kBatch are significantly worse in
        # initial testing
        kBatch = min(max(next_power_of_2(total_k_blocks // cus), 1), 128)
        return [kBatch]

    def gen_ops(self) -> list[InductorROCmOp]:
        """
        Creates a list of `CKGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime.
        Such instances will be assigned +inf runtime by the autotune process.
        """
        try:
            from ck4inductor.batched_universal_gemm.gen_instances import (  # type: ignore[import]
                gen_ops_library as gen_batched_gemm_ops_library,
            )
            from ck4inductor.universal_gemm.gen_instances import (  # type: ignore[import]
                gen_ops_library as gen_gemm_ops_library,
                gen_ops_preselected as gen_gemm_ops_preselected,
            )
        except ImportError:
            return []

        generator = None
        if self.is_batched:
            generator = gen_batched_gemm_ops_library
        else:
            generator = gen_gemm_ops_library
        if config.rocm.use_preselected_instances and self._is_rcr_f16():
            generator = gen_gemm_ops_preselected

        if generator is None:
            raise AssertionError("expected a non-None CK GEMM ops generator")

        rops = generator()
        ops = []
        for o in rops:
            kBatches = self._get_kBatch(o)
            for kBatch in kBatches:
                # pyrefly: ignore [bad-argument-type]
                ops.append(InductorROCmOp(op=o, kBatch=kBatch))

        filtered_instances = list(filter(lambda op: self.filter_op(op), ops))

        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Randomizing the choice seems to solve this.
        random.seed(-11)
        chosen_instances = (
            random.sample(
                filtered_instances,
                min(len(filtered_instances), config.rocm.ck_max_profiling_configs),
            )
            if config.rocm.ck_max_profiling_configs
            else filtered_instances
        )
        log.debug(
            "generated %d ck instances after filter: %s",
            len(chosen_instances),
            chosen_instances,
        )
        return chosen_instances

    @staticmethod
    def add_ck_gemm_choices(
        choices,
        layout,
        input_nodes,
        alpha=1,
        beta=0,
        input_reorder=None,
    ):
        """
        Add Composable Kernel Universal GEMM instance choices to the auto-tuning list.
        """
        template = CKGemmTemplate(
            input_nodes,
            layout,
            alpha=alpha,
            beta=beta,
            input_reorder=input_reorder,
        )
        ops = template.gen_ops()
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op.op,
                kBatch=op.kBatch,
            )

    def size_args(self):
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        Bias = (
            self.input_nodes[2]
            if len(self.input_nodes) == 3
            else self.input_nodes[4]
            if len(self.input_nodes) == 5
            else None
        )
        Y = self.output_node

        M = X.get_size()[-2]
        K = X.get_size()[-1]
        N = W.get_size()[-1]
        LDA = X.get_stride()[-2 if X.get_stride()[-1] == 1 else -1]
        LDB = W.get_stride()[-2 if W.get_stride()[-1] == 1 else -1]
        LDC = Y.get_stride()[-2 if Y.get_stride()[-1] == 1 else -1]
        LDD = (
            0
            if (Bias is None or len(Bias.get_size()) == 1)
            else Bias.get_stride()[-2 if Bias.get_stride()[-1] == 1 else -1]
        )
        if self.is_batched:
            B = X.get_size()[0]
            return B, M, N, K, LDA, LDB, LDC, LDD
        else:
            return M, N, K, LDA, LDB, LDC, LDD
<|endoftext|># mypy: allow-untyped-defs
import logging
import os

from torch._inductor import config
from torch._inductor.utils import is_linux, try_import_ck_lib


log = logging.getLogger(__name__)


def _rocm_include_paths(dst_file_ext: str) -> list[str]:
    from torch.utils import cpp_extension

    rocm_include = (
        os.path.join(config.rocm.rocm_home, "include")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("include")
    )

    if config.is_fbcode():
        from libfb.py import parutil

        ck_path = parutil.get_dir_path("composable-kernel-headers")
    else:
        if not config.rocm.ck_dir:
            ck_dir, _, _, _ = try_import_ck_lib()
            if not ck_dir:
                log.warning("Unspecified Composable Kernel directory")
            config.rocm.ck_dir = ck_dir
        ck_path = config.rocm.ck_dir or cpp_extension._join_rocm_home(
            "composable_kernel"
        )

    log.debug("Using ck path %s", ck_path)

    ck_include = os.path.join(ck_path, "include")
    ck_library_include = os.path.join(ck_path, "library", "include")

    # CK has to take priority over ROCm include paths
    # Since CK is potentially more up-to-date
    paths = [
        os.path.realpath(p) for p in (ck_include, ck_library_include, rocm_include)
    ]
    if dst_file_ext == "exe":
        ck_utility_include = os.path.join(ck_path, "library", "src", "utility")
        paths.append(os.path.realpath(ck_utility_include))
    return paths


def _rocm_lib_options(dst_file_ext: str) -> list[str]:
    from torch.utils import cpp_extension

    rocm_lib_dir = (
        os.path.join(config.rocm.rocm_home, "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("lib")
    )
    hip_lib_dir = (
        os.path.join(config.rocm.rocm_home, "hip", "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("hip", "lib")
    )

    opts = [
        "-include __clang_hip_runtime_wrapper.h",
        f"-L{os.path.realpath(rocm_lib_dir)}",
        f"-L{os.path.realpath(hip_lib_dir)}",
        "-lamdhip64",
    ]
    if dst_file_ext == "exe":
        opts += ["-lpthread", "-lstdc++"]
    return opts


def _rocm_compiler_options() -> list[str]:
    arch_list = config.rocm.arch or ["native"]
    gpu_arch_flags = [f"--offload-arch={arch}" for arch in arch_list]
    opts = [
        config.rocm.compile_opt_level,
        "-x",
        "hip",
        "-std=c++20",
        *gpu_arch_flags,
        "-fno-gpu-rdc",
        "-fPIC",
        "-fvisibility=hidden",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-mllvm",
        "-enable-post-misched=0",
    ]
    if config.rocm.is_debug:
        opts += ["-DDEBUG_LOG=1", "-g"]
    if config.rocm.save_temps:
        opts += ["--save-temps=obj"]
    if config.rocm.print_kernel_resource_usage:
        opts += ["-Rpass-analysis=kernel-resource-usage"]
    if config.rocm.flush_denormals:
        opts += ["-fgpu-flush-denormals-to-zero"]
    if config.rocm.use_fast_math:
        opts += ["-ffast-math"]
    return opts


def rocm_compiler() -> str | None:
    if is_linux():
        if config.rocm.rocm_home:
            return os.path.realpath(
                os.path.join(config.rocm.rocm_home, "llvm", "bin", "clang")
            )
        try:
            from torch.utils import cpp_extension

            return os.path.realpath(
                cpp_extension._join_rocm_home("llvm", "bin", "clang")
            )
        except OSError:
            # neither config.rocm.rocm_home nor env variable ROCM_HOME are set
            return "clang"
    return None


def rocm_compile_command(
    src_files: list[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: list[str] | None = None,
) -> str:
    include_paths = _rocm_include_paths(dst_file_ext)
    lib_options = _rocm_lib_options(dst_file_ext)
    compiler_options = _rocm_compiler_options()
    compiler = rocm_compiler()
    options = (
        compiler_options
        + (extra_args or [])
        + [f"-I{path}" for path in include_paths]
        + lib_options
    )
    src_file = " ".join(src_files)
    # supported extensions: .o, .so, .exe
    if dst_file_ext == "o":
        options.append("-c")
    elif dst_file_ext == "so":
        options.append("-shared")
    elif dst_file_ext == "exe":
        options.append("-DGENERATE_CK_STANDALONE_RUNNER")
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    return f"{compiler} {' '.join(options)} -o {dst_file} {src_file}"
<|endoftext|># mypy: allow-untyped-defs, disable-error-code="attr-defined, valid-type"
import functools
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch._inductor import config
from torch._inductor.codegen.rocm.ck_tile_template import CKTileTemplate
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
from torch._inductor.codegen.rocm.rocm_template import ArgInfo
from torch._inductor.ir import Buffer, Layout
from torch.utils._ordered_set import OrderedSet

from ...utils import IndentedBuffer


log = logging.getLogger(__name__)


def is_static_int(number):
    import sympy

    return isinstance(number, (int, sympy.Integer))


def torch_layout_to_ck_layout(torch_layout):
    if torch_layout.stride[-1] == 1:
        return "Row"
    elif torch_layout.stride[-2] == 1:
        return "Col"
    else:
        return None


@dataclass
class CKTileGemmOperation:
    layout_a: str
    layout_b: str
    layout_c: str

    datatype_a: str
    datatype_b: str
    datatype_c: str

    tile_m: int
    tile_n: int
    tile_k: int

    warp_m: int
    warp_n: int
    warp_k: int

    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    m_is_padded: str
    n_is_padded: str
    k_is_padded: str

    pipeline: str
    scheduler: str
    epilogue: str

    def layout_repr(self):
        return f"{self.layout_a[0]}{self.layout_b[0]}{self.layout_c[0]}"

    def dtype_repr(self):
        return f"{self.datatype_a}{self.datatype_b}{self.datatype_c}"

    def tile_sizes(self):
        return "_".join(
            [
                f"{self.tile_m}{self.tile_n}{self.tile_k}",
                f"{self.warp_m}{self.warp_n}{self.warp_k}",
                f"{self.warp_tile_m}{self.warp_tile_n}{self.warp_tile_k}",
            ]
        )

    def name(self):
        return "ck_tile_gemm_universal_" + "_".join(
            [
                f"{self.layout_repr()}",
                f"{self.dtype_repr()}",
                f"{self.tile_sizes()}",
                f"{self.pipeline}",
                f"{self.scheduler}",
                f"{self.epilogue}",
            ]
        )

    def dict_items(self):
        return asdict(self).items()


@functools.cache
def ops():
    """
    Generate the supported instance dataclasses
    """
    import itertools

    compute_v3_instances = [
        CKTileGemmOperation(
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
            datatype_a=datatype_a,
            datatype_b=datatype_b,
            datatype_c=datatype_c,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=warp_k,
            warp_tile_m=warp_tile_m,
            warp_tile_n=warp_tile_n,
            warp_tile_k=warp_tile_k,
            m_is_padded=m_is_padded,
            n_is_padded=n_is_padded,
            k_is_padded=k_is_padded,
            pipeline="CompV3",
            scheduler="Intrawave",
            epilogue=epilogue,
        )
        for (layout_a, layout_b, layout_c) in [
            ("Row", "Row", "Row"),
            ("Row", "Col", "Row"),
        ]
        for (datatype_a, datatype_b, datatype_c) in [("FP16",) * 3, ("BF16",) * 3]
        for (tile_m, tile_n, tile_k) in [(256, 256, 32), (256, 256, 64)]
        for (warp_m, warp_n, warp_k) in [(2, 2, 1)]
        for (warp_tile_m, warp_tile_n, warp_tile_k) in [(32, 32, 16)]
        for m_is_padded in ["true", "false"]
        for n_is_padded in ["true", "false"]
        for k_is_padded in ["true", "false"]
        for epilogue in ["Default", "CShuffle"]
    ]

    compute_v4_instances = [
        CKTileGemmOperation(
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
            datatype_a=datatype_a,
            datatype_b=datatype_b,
            datatype_c=datatype_c,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=warp_k,
            warp_tile_m=warp_tile_m,
            warp_tile_n=warp_tile_n,
            warp_tile_k=warp_tile_k,
            m_is_padded=m_is_padded,
            n_is_padded=n_is_padded,
            k_is_padded=k_is_padded,
            pipeline="CompV4",
            scheduler="Intrawave",
            epilogue=epilogue,
        )
        for (layout_a, layout_b, layout_c) in [
            ("Row", "Row", "Row"),
            ("Row", "Col", "Row"),
        ]
        for (datatype_a, datatype_b, datatype_c) in [("FP16",) * 3, ("BF16",) * 3]
        for (tile_m, tile_n, tile_k) in [
            (256, 256, 32)
        ]  # half the tile size since it has double buffering
        for (warp_m, warp_n, warp_k) in [(2, 2, 1)]
        for (warp_tile_m, warp_tile_n, warp_tile_k) in [(32, 32, 16)]
        for m_is_padded in ["true", "false"]
        for n_is_padded in ["true", "false"]
        for k_is_padded in ["true", "false"]
        for epilogue in ["Default", "CShuffle"]
    ]

    mem_instances = [
        CKTileGemmOperation(
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
            datatype_a=datatype_a,
            datatype_b=datatype_b,
            datatype_c=datatype_c,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=warp_k,
            warp_tile_m=warp_tile_m,
            warp_tile_n=warp_tile_n,
            warp_tile_k=warp_tile_k,
            m_is_padded=m_is_padded,
            n_is_padded=n_is_padded,
            k_is_padded=k_is_padded,
            pipeline="Mem",
            scheduler=scheduler,
            epilogue=epilogue,
        )
        for (layout_a, layout_b, layout_c) in [
            ("Row", "Row", "Row"),
            ("Row", "Col", "Row"),
        ]
        for (datatype_a, datatype_b, datatype_c) in [("FP16",) * 3, ("BF16",) * 3]
        for (tile_m, tile_n, tile_k) in [(256, 256, 32), (256, 256, 64)]
        for (warp_m, warp_n, warp_k) in [(2, 2, 1)]
        for (warp_tile_m, warp_tile_n, warp_tile_k) in [(32, 32, 16)]
        for m_is_padded in ["true", "false"]
        for n_is_padded in ["true", "false"]
        for k_is_padded in ["true", "false"]
        for scheduler in ["Intrawave", "Interwave"]
        for epilogue in ["Default", "CShuffle"]
    ]

    return list(
        itertools.chain(compute_v3_instances, compute_v4_instances, mem_instances)
    )


class CKTileGemmTemplate(CKTileTemplate):
    """
    This class is used for rendering CK-Tile Universal GEMM kernels
    """

    gemm_template = r"""{{version_comment}}
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {

        using {{instance_namespace}}::BaseGemmPipeline;
        using {{instance_namespace}}::TilePartitioner;

        constexpr auto TileK = {{instance_namespace}}::TileK;
        constexpr auto kPrefetchStages = BaseGemmPipeline::PrefetchStages;

        const auto BiasTerms = std::array<const void*, 0> ();
        const auto BiasStrides = std::array<int32_t, 0> ();

        auto kargs = ck_tile::UniversalGemmKernelArgs<> {
           {X},
           {W},
           BiasTerms,
           Y,
           M,
           N,
           K,
           {LDA},
           {LDB},
           BiasStrides,
           LDC,
           kBatch
        };

        if (workspace_size) {
            *workspace_size = 0;
            return 0;
        }

        const ck_tile::index_t k_grain     = kBatch * TileK;
        const ck_tile::index_t K_split     = (K + k_grain - 1) / k_grain * TileK;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        // run the kernel
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;

            using Kernel = {{instance_namespace}}::Kernel<has_hot_loop_v, tail_number_v>;

            if (!Kernel::IsSupportedArgument(kargs)) {
                // we do our best to statically avoid this case in `filter_op`
                throw std::runtime_error("invalid argument");
            }
            auto stream_config = ck_tile::stream_config{stream};
            auto grid_size = Kernel::GridSize(M, N, kBatch);
            auto block_size = Kernel::BlockSize();
            constexpr auto lds_bytes = 0;
            constexpr auto kBlockPerCU = 1;
            auto gemm = ck_tile::make_kernel<kBlockPerCU>(Kernel{}, grid_size, block_size, lds_bytes, kargs);
            ck_tile::launch_kernel(stream_config, gemm);
        };

        BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);

        return 0;
    } // kernel definition
    } // extern C
    """

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
    ) -> None:
        super().__init__(
            "ck_tile_gemm_template",
            input_nodes=input_nodes,
            layout=layout,
        )

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK GEMM header(s)

                #include "ck_tile/ops/gemm.hpp"
                #include "ck_tile/ops/epilogue.hpp"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK GEMM globals

                using Row = ck_tile::tensor_layout::gemm::RowMajor;
                using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
            """
        )
        return res

    def check_dtypes(self, op: "CKTileGemmOperation"):
        X_dtype, W_dtype, out_dtype = [
            T.get_layout().dtype for T in [*self.input_nodes, self.output_node]
        ]
        if op.datatype_a != self._TORCH_DTYPE_TO_CK[X_dtype]:
            return False
        if op.datatype_b != self._TORCH_DTYPE_TO_CK[W_dtype]:
            return False
        if op.datatype_c != self._TORCH_DTYPE_TO_CK[out_dtype]:
            return False
        return True

    def check_layouts(self, op: "CKTileGemmOperation"):
        X_layout, W_layout, out_layout = [
            torch_layout_to_ck_layout(T.get_layout())
            for T in [*self.input_nodes, self.output_node]
        ]
        if op.layout_a != X_layout:
            return False
        if op.layout_b != W_layout:
            return False
        if op.layout_c != out_layout:
            return False
        return True

    def get_gemm_problem_size(self):
        X_size, W_size = [T.get_layout().size for T in [*self.input_nodes]]

        M, K = X_size
        _, N = W_size

        return M, N, K

    def check_block_tiles(self, op: "CKTileGemmOperation"):
        """
        The contiguous dimension of a tensor must be divisible by the block tile size
        This helper function enforces it for the inputs and the output.
        """
        M, N, K = self.get_gemm_problem_size()

        def check(dim_size, tile_size, is_padded):
            if (
                is_static_int(dim_size)
                and dim_size % tile_size != 0
                and is_padded == "false"
            ):
                return False
            return True

        if op.layout_a == "Row":
            # handle in kBatch check
            return True
        elif op.layout_a == "Col":
            if not check(M, op.tile_m, op.m_is_padded):
                return False
        else:
            raise AssertionError(f"Invalid layout {op.layout_a=}")

        if op.layout_b == "Row":
            if not check(N, op.tile_n, op.n_is_padded):
                return False
        elif op.layout_b == "Col":
            # handle in kBatch check
            return True
        else:
            raise AssertionError(f"Invalid {op.layout_b=}")

        if op.layout_c == "Row":
            if not check(N, op.tile_n, op.n_is_padded):
                return False
        elif op.layout_c == "Col":
            if not check(M, op.tile_m, op.m_is_padded):
                return False
        else:
            raise AssertionError(f"Invalid layout {op.layout_c=}")

        return True

    def check_alignments(self, op: "CKTileGemmOperation"):
        """
        The contiguous dimension of a tensor must be divisible by the vector load size.
        """
        M, N, K = self.get_gemm_problem_size()

        def max_alignment(contiguous_elements_per_tile, elements_per_thread, ck_dtype):
            for vector_load_bytes in (16, 8, 4, 2, 1):
                alignment = vector_load_bytes // self.ck_dtype_to_size[ck_dtype]
                if (
                    alignment > 0
                    and contiguous_elements_per_tile % alignment == 0
                    and elements_per_thread % alignment == 0
                ):
                    return alignment

        threads_per_block = (
            op.warp_m * op.warp_n * op.warp_k * self.gfx9_threads_per_warp
        )
        a_elements_per_thread = op.tile_m * op.tile_k / threads_per_block
        b_elements_per_thread = op.tile_n * op.tile_k / threads_per_block

        if op.layout_a == "Row":
            # K is contiguous tensor dimension
            a_max_vector_size = max_alignment(
                op.tile_k, a_elements_per_thread, op.datatype_a
            )
            if is_static_int(K) and K % a_max_vector_size != 0:
                return False
        elif op.layout_a == "Col":
            # M is contiguous tensor dimension
            a_max_vector_size = max_alignment(
                op.tile_m, a_elements_per_thread, op.datatype_a
            )
            if is_static_int(M) and M % a_max_vector_size != 0:
                return False
        else:
            raise AssertionError(f"Invalid layout {op.layout_a=}")

        if op.layout_b == "Row":
            # N is contiguous tensor dimension
            b_max_vector_size = max_alignment(
                op.tile_n, b_elements_per_thread, op.datatype_b
            )
            if is_static_int(N) and N % b_max_vector_size != 0:
                return False
        elif op.layout_b == "Col":
            # K is contiguous tensor dimension
            b_max_vector_size = max_alignment(
                op.tile_k, b_elements_per_thread, op.datatype_b
            )
            if is_static_int(K) and K % b_max_vector_size != 0:
                return False
        else:
            raise AssertionError(f"Invalid layout {op.layout_b=}")

        # the `default` epilogue writes C to memory by 1 tensor element
        # (divisibility check not necessary)
        # the `cshuffle` epilogue writes C to memory by 16 bytes
        # (so the contiguous C dimension size must be divisible by the number of tensor elements in 16 bytes)
        if op.epilogue == "CShuffle":
            if (
                op.layout_c == "Row"
                and is_static_int(N)
                and N % (16 / self.ck_dtype_to_size[op.datatype_c]) != 0
            ):
                return False

        return True

    def check_warp_tiles(self, op: "CKTileGemmOperation"):
        if op.tile_m % (op.warp_m * op.warp_tile_m) != 0:
            return False
        if op.tile_n % (op.warp_n * op.warp_tile_n) != 0:
            return False
        if op.tile_k % (op.warp_k * op.warp_tile_k) != 0:
            return False
        return True

    def check_block_tile_size(self, op: "CKTileGemmOperation"):
        # assuming LDS size is 64KB
        if op.pipeline == "CompV4":
            max_block_tile_size = 2**15
        else:
            max_block_tile_size = 2**16

        block_tile_size = (
            self.ck_dtype_to_size[op.datatype_a] * op.tile_m * op.tile_k
            + self.ck_dtype_to_size[op.datatype_b] * op.tile_n * op.tile_k
        )
        if block_tile_size > max_block_tile_size:
            return False
        return True

    def filter_op(self, op: "CKTileGemmOperation"):
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """
        if not self.check_dtypes(op):
            return None
        if not self.check_layouts(op):
            return None
        if not self.check_block_tiles(op):
            return None
        if not self.check_alignments(op):
            return None

        return op

    def emit_ck_instance(self, op: "CKTileGemmOperation"):
        """
        This method is used to generate code which defines the type alias for the generated kernel class
        """
        template_definition = r"""
    // Gemm operator {{operation_name}}

    namespace {{operation_name}} {
         // block tile
        constexpr int32_t TileM = {{tile_m}};
        constexpr int32_t TileN = {{tile_n}};
        constexpr int32_t TileK = {{tile_k}};
        // warps per block
        constexpr int32_t WarpM = {{warp_m}};
        constexpr int32_t WarpN = {{warp_n}};
        constexpr int32_t WarpK = {{warp_k}};
        // xdl tile
        constexpr int32_t WarpTileM = {{warp_tile_m}};
        constexpr int32_t WarpTileN = {{warp_tile_n}};
        constexpr int32_t WarpTileK = {{warp_tile_k}};

        constexpr bool kPadM = {{m_is_padded}};
        constexpr bool kPadN = {{n_is_padded}};
        constexpr bool kPadK = {{k_is_padded}};

        using ALayout = {{layout_a}};
        using BLayout = {{layout_b}};
        using CLayout = {{layout_c}};

        using ADataType = {{datatype_a}};
        using BDataType = {{datatype_b}};
        using CDataType = {{datatype_c}};
        using AccDataType = F32;

        constexpr bool permuteA = false;
        constexpr bool permuteB = false;
        constexpr bool DoubleSmemBuffer = {{has_double_smem_buffer}};
        constexpr bool TransposeC = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
        constexpr ck_tile::index_t TilePartitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                   ck_tile::sequence<WarpM, WarpN, WarpK>,
                                   ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
                                   permuteA,
                                   permuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       TilePartitionerGroupNum,
                                                       TilePartitionerM01>;

        using Traits  =
            ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                             ALayout, BLayout, CLayout, TransposeC>;

        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        {{rendered_scheduler}}

        template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  scheduler,
                                                  has_hot_loop_v,
                                                  tail_number_v>;

        {{rendered_pipeline}}

        {{rendered_epilogue}}

        template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline<has_hot_loop_v, tail_number_v>, GemmEpilogue>;
    }

"""

        def render_epilogue(epilogue_type):
            if epilogue_type == "Default":
                return r"""
            using DsDataType = ck_tile::tuple<>;
            using DsLayout = ck_tile::tuple<>;
            using CDElementwise = ck_tile::element_wise::PassThrough;

            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
                                                                          BDataType,
                                                                          DsDataType,
                                                                          AccDataType,
                                                                          CDataType,
                                                                          DsLayout,
                                                                          CLayout,
                                                                          CDElementwise,
                                                                          TileM,
                                                                          TileN,
                                                                          kPadM,
                                                                          kPadN,
                                                                          WarpTileM,
                                                                          WarpTileN,
                                                                          WarpTileK,
                                                                          TransposeC>;
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;
        """
            elif epilogue_type == "CShuffle":
                return r"""
            using DsDataType = ck_tile::tuple<>; // no bias terms for vanilla GEMM
            using DsLayout = ck_tile::tuple<>;
            constexpr auto ELayout = CLayout;
            using CDEElementWise = ck_tile::element_wise::PassThrough; // no-op
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                                     BDataType,
                                                                     DsDataType,
                                                                     AccDataType,
                                                                     CDataType,
                                                                     DsLayout,
                                                                     ELayout,
                                                                     CDEElementWise,
                                                                     TileM,
                                                                     TileN,
                                                                     WarpM,
                                                                     WarpN,
                                                                     WarpTileM,
                                                                     WarpTileN,
                                                                     WarpTileK,
                                                                     TransposeC>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
        """
            else:
                raise AssertionError("Epilogue must be set")

        def render_pipeline(pipeline_type):
            return rf"""
            using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCr{pipeline_type}<GemmPipelineProblem>;

            template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
            using GemmPipeline = ck_tile::GemmPipelineAgBgCr{pipeline_type}<UniversalGemmProblem<has_hot_loop_v, tail_number_v>>;
        """

        def render_scheduler(scheduler_type):
            return rf"""
            constexpr auto scheduler = ck_tile::GemmPipelineScheduler::{scheduler_type};
        """

        rendered_definition = self._template_from_string(template_definition).render(
            operation_name=op.name(),
            **asdict(op),
            rendered_scheduler=render_scheduler(op.scheduler),
            rendered_pipeline=render_pipeline(op.pipeline),
            rendered_epilogue=render_epilogue(op.epilogue),
            has_double_smem_buffer=("true" if op.pipeline == "CompV4" else "false"),
        )
        return rendered_definition

    def render(  # type: ignore[override]
        self, kernel: ROCmTemplateKernel, op: "CKTileGemmOperation", **kwargs
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        """
        epilogue_nodes = kwargs.get("epilogue_nodes")
        if not (epilogue_nodes is None or 0 == len(epilogue_nodes)):
            raise AssertionError("expected no epilogue_nodes for CK tile gemm template")
        template_buffer_node = kwargs.get("template_buffer_node")
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if 2 != len(self.input_nodes):
            raise AssertionError(f"expected 2 input_nodes, got {len(self.input_nodes)}")
        X, W = self.input_nodes
        Y = self.output_node

        instance_definition = self.emit_ck_instance(op)

        version_comment = rf"""/**
* Generated code for CK inductor backend
* See {type(self).__module__}.{type(self).__qualname__}
*
* Template instance {op}
*
* {torch.__version__=}
* torch.version.git_version={getattr(torch.version, "git_version", "None")}
*/
"""

        return self._template_from_string(self.gemm_template).render(
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W],  # type: ignore[list-item]
                outputs=[Y],
                names_str="X, W, Y",
                size_args=[
                    f"int32_t {arg}" for arg in ["M", "N", "K", "LDA", "LDB", "LDC"]
                ],
            ),
            instance_namespace=op.name(),
            version_comment=version_comment,
        )

    def gen_ops(self):
        """
        Creates a list of `CKTileGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime.
        Such instances will be assigned +inf runtime by the autotune process.
        """
        instances = ops()
        if not instances:
            raise AssertionError(
                "No Composable Kernel Universal GEMM instances found. "
                "Please check if the library is installed."
            )
        filtered_instances = list(filter(self.filter_op, instances))
        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Stratified sampling ensures coverage of
        # different pipeline/epilogue combinations while still randomizing within each stratum.
        # Use a local Random instance so we don't mutate the global RNG state shared with
        # other inductor code paths.
        rng = random.Random(-11)
        max_configs = config.rocm.ck_tile_max_profiling_configs
        if max_configs:
            strata: dict[tuple[str, str], list] = defaultdict(list)
            for inst in filtered_instances:
                strata[(inst.pipeline, inst.epilogue)].append(inst)
            # Shuffle stratum keys so that when len(strata) > max_configs, the strata that
            # get represented are not biased by filter/insertion order.
            stratum_keys = list(strata.keys())
            rng.shuffle(stratum_keys)
            # Distribute the budget across strata as evenly as possible. The first
            # `remainder` strata get one extra slot.
            per_stratum, remainder = divmod(max_configs, len(stratum_keys))
            chosen_instances: list = []
            for i, key in enumerate(stratum_keys):
                quota = per_stratum + (1 if i < remainder else 0)
                if quota == 0:
                    break
                group = strata[key]
                chosen_instances.extend(rng.sample(group, min(len(group), quota)))
            # If some strata were smaller than their quota, fill the slack from the global
            # remainder so we still hit max_configs when possible.
            shortfall = max_configs - len(chosen_instances)
            if shortfall > 0:
                chosen_ids = OrderedSet(id(x) for x in chosen_instances)
                remaining = [x for x in filtered_instances if id(x) not in chosen_ids]
                if remaining:
                    chosen_instances.extend(
                        rng.sample(remaining, min(len(remaining), shortfall))
                    )
        else:
            chosen_instances = filtered_instances
        log.debug(
            "generated %d ck instances after sample: %s",
            len(chosen_instances),
            chosen_instances,
        )
        return chosen_instances

    @staticmethod
    def add_choices(
        choices,
        layout,
        input_nodes,
    ):
        """
        Add Composable Kernel Universal GEMM instance choices to the auto-tuning list.
        """
        template = CKTileGemmTemplate(
            input_nodes,
            layout,
        )
        ops = template.gen_ops()
        for op in ops:
            for k_batch in template.k_batch_choices(op):
                template.maybe_append_choice(
                    choices,
                    op=op,
                    kBatch=k_batch,
                )

    def k_batch_choices(self, op: "CKTileGemmOperation") -> tuple[int, ...]:
        """
        Returns a list of k_batch choices for the template.
        """
        default_choices = (1, 2, 4, 8, 16, 32)

        def check(dim_size, tile_size, is_padded):
            if (
                is_static_int(dim_size)
                and dim_size % tile_size != 0
                and is_padded == "false"
            ):
                return False
            return True

        _, _, K, _, _, _ = self.size_args()
        if op.layout_a == "Row" or op.layout_b == "Col":
            choices = tuple(
                filter(
                    lambda k_batch: check(K, op.tile_k * k_batch, op.k_is_padded),
                    default_choices,
                )
            )
        else:
            choices = default_choices

        if op.epilogue == "Default":
            choices = (1,)

        return choices

    def size_args(self):
        """
        Sizes and strides to be used for the kernel call
        """
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        Y = self.output_node

        M = X.get_size()[0]
        K = X.get_size()[1]
        N = W.get_size()[1]
        LDA = X.get_stride()[0 if X.get_stride()[1] == 1 else 1]
        LDB = W.get_stride()[0 if W.get_stride()[1] == 1 else 1]
        LDC = Y.get_stride()[0 if Y.get_stride()[1] == 1 else 1]

        return M, N, K, LDA, LDB, LDC

    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return [ArgInfo("kBatch", "int32_t")]

    def get_runtime_arg_values(self, **kwargs: Any) -> list[Any]:
        # maybe_append_choice kwarg for k_batch must match the name of the argument
        arg_names = OrderedSet([arg.name for arg in self.get_runtime_arg_info()])
        if not arg_names.issubset(kwargs):
            raise ValueError(
                "Missing runtime arguments: " + ", ".join(arg_names - kwargs.keys())
            )
        return [kwargs[k] for k in arg_names]
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
from ctypes import byref, c_int, c_size_t, c_void_p
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor import config
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.codecache import DLLWrapper, ROCmCodeCache


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


log = logging.getLogger(__name__)


class ROCmBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put CUDA Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: torch.Tensor | None = None
        self.DLL: DLLWrapper | None = None
        self._workspace_size_updated = False
        self.hash_key: str = ""
        self.source_file: str = ""
        self.hash_key, self.source_file = ROCmCodeCache.write(self.source_code, "so")

    def precompile(self):
        # Prepopulate code cache
        # may happen in separate Threadpool
        log.debug("Precompiling %s", self)
        ROCmCodeCache.compile(self.source_code, "so")
        if config.rocm.generate_test_runner:
            ROCmCodeCache.compile(self.source_code, "exe")
        log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        self.ensure_dll_loaded()
        self.update_workspace_size()
        args = [c_void_p(tensor.data_ptr()) for tensor in list(input_tensors) + [out]]
        size_args = [c_int(arg) for arg in self.extra_args]
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)
        run_method = getattr(self.DLL, self.kernel_name)
        workspace_ptr = c_void_p(0)
        if self.workspace_size > 0:
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=out.device,
            )
            workspace_ptr = c_void_p(self.workspace.data_ptr())

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *size_args,
            None,  # null workspace size ptr
            workspace_ptr,  # set workspace ptr,
            stream_ptr,
        )

    def update_workspace_size(self) -> None:
        if self._workspace_size_updated:
            return
        self.ensure_dll_loaded()
        unique_input_count = len(
            dict.fromkeys(meta.name for meta in self.input_tensor_meta)
        )
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)

        run_method = getattr(self.DLL, self.kernel_name)
        # Retrieve workspace_size and initialize workspace.
        c_workspace_size = c_size_t()
        size_args = [c_int(arg) for arg in self.extra_args]
        run_method(
            *args,  # input ptrs and output ptrs
            *size_args,
            byref(
                c_workspace_size
            ),  # set workspace size ptr to retrieve workspace size
            None,  # null workspace ptr
            stream_ptr,
        )
        torch.cuda.synchronize()  # shake out any CUDA errors
        self.workspace_size = c_workspace_size.value
        log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        self._workspace_size_updated = True

    def ensure_dll_loaded(self):
        if self.DLL is None:
            self.DLL, self.hash_key, self.source_file = ROCmCodeCache.load(
                self.source_code, "so"
            )

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()
        self.workspace = None

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"
<|endoftext|>import torch
from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate
from torch._inductor.ir import IRNode
from torch._inductor.utils import IndentedBuffer


class CKTileTemplate(ROCmTemplate):
    """
    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic
    """

    _TORCH_DTYPE_TO_CK = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int32: "I32",
        torch.int8: "I8",
        torch.float8_e4m3fnuz: "F8",  # gfx94
        torch.float8_e4m3fn: "F8",  # gfx95
        torch.float8_e5m2fnuz: "BF8",  # gfx94
        torch.float8_e5m2: "BF8",  # gfx95
    }

    ck_dtype_to_size = {
        "FP16": 2,
        "BF16": 2,
    }

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK headers
                #include "ck_tile/core.hpp"

            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                using F8  = ck_tile::fp8_t;
                using BF8 = ck_tile::bf8_t;
                using F16 = ck_tile::half_t;
                using F32 = float;
                using BF16 = ck_tile::bfloat16_t;
            """
        )
        return res

    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._TORCH_DTYPE_TO_CK.get(node.get_dtype())}*)({ptr})"
<|endoftext|>from collections.abc import Callable, Sequence
from typing import TypeVar
from typing_extensions import ParamSpec

from ...ir import Buffer, Layout, TemplateBuffer


_P = ParamSpec("_P")
_T = TypeVar("_T")


class ROCmTemplateBuffer(TemplateBuffer):
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[Buffer],
        make_kernel_render: Callable[_P, _T],
        workspace_size: int,
        template: "ROCmTemplate",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template

    def get_workspace_size(self) -> int:
        return self.workspace_size if self.workspace_size is not None else 0
<|endoftext|>from typing import Any
from typing_extensions import override

import torch
from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate
from torch._inductor.ir import IRNode
from torch._inductor.utils import IndentedBuffer

from .rocm_template import ArgInfo


class CKTemplate(ROCmTemplate):
    """
    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic
    """

    _TORCH_DTYPE_TO_CK = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int32: "I32",
        torch.int8: "I8",
        torch.float8_e4m3fnuz: "F8",  # gfx94
        torch.float8_e4m3fn: "F8",  # gfx95
        torch.float8_e5m2fnuz: "BF8",  # gfx94
        torch.float8_e5m2: "BF8",  # gfx95
    }

    @staticmethod
    def is_blocked_by_tf32_setting(op) -> bool:
        """True if op internally computes in TF32 but the user has disabled TF32.

        CK exposes this via {a,b}_compute_dtype on op (None == strict compute
        matching the element type). Some F32 instances ship with `TF32` here.
        """
        if torch.backends.cuda.matmul.allow_tf32:
            return False
        return (
            getattr(op, "a_compute_dtype", None) == "TF32"
            or getattr(op, "b_compute_dtype", None) == "TF32"
        )

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK headers

                #ifdef DEBUG_LOG
                #define DEBUG_LOG_TMP DEBUG_LOG
                #undef DEBUG_LOG
                #else
                #define DEBUG_LOG_TMP 0
                #endif
                #include "ck/ck.hpp"
                #undef DEBUG_LOG
                #define DEBUG_LOG DEBUG_LOG_TMP

                #include "ck/utility/data_type.hpp"
                #include "ck/library/utility/check_err.hpp"
                #include "ck/library/utility/device_memory.hpp"
                #include "ck/library/utility/fill.hpp"
                #include "ck/library/utility/host_tensor.hpp"
                #include "ck/library/utility/host_tensor_generator.hpp"
                #include "ck/library/utility/literals.hpp"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK globals

                template <ck::index_t... Is>
                using S = ck::Sequence<Is...>;

                template<typename... Ts>
                using Tuple = ck::Tuple<Ts...>;

                using PassThrough = ck::tensor_operation::element_wise::PassThrough;
                using Bilinear = ck::tensor_operation::element_wise::Bilinear;
                using Scale = ck::tensor_operation::element_wise::Scale;
                using ScaleAdd = ck::tensor_operation::element_wise::ScaleAdd;
                using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;

                // see "composable_kernel/include/ck/utility/data_type.hpp"
                using F8  = ck::f8_t;
                using BF8 = ck::bf8_t;
                using F16 = ck::half_t;
                using F32 = float;
                // using F64 = double;
                using BF16 = ck::bhalf_t;
                using TF32 = ck::tf32_t;
                // using I32 = int32_t;
                // using I8 = int8_t;
                // using I4 = ck::int4_t;

                #if DEBUG_LOG
                static constexpr auto kDEBUG_LOG = 1;
                #else
                static constexpr auto kDEBUG_LOG = 0;
                #endif
            """
        )
        return res

    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._TORCH_DTYPE_TO_CK.get(node.get_dtype())}*)({ptr})"

    @override
    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return [ArgInfo("kBatch", "int32_t")]

    @override
    def get_runtime_arg_values(self, **kwargs: Any) -> list[Any]:
        """
        Helper method to retrieve runtime args from generate kwargs
        """
        return [kwargs[arg.name] for arg in self.get_runtime_arg_info()]
<|endoftext|>from __future__ import annotations

from ..common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class MTIADeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch._C import _mtia_getCurrentRawStream as {name}"

    def set_device(self, device_idx: DeviceIdx) -> str:
        return f"torch.mtia.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.mtia.synchronize()"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        return f"torch.mtia.device({device_idx})"

    def current_stream(self) -> str:
        return "torch.mtia.current_stream()"


register_device_op_overrides("mtia", MTIADeviceOpOverrides())
<|endoftext|># mypy: allow-untyped-defs
import hashlib
import logging
from collections.abc import Sequence
from typing import cast

from torch._inductor.utils import Placeholder
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import CuteDSLTemplateBuffer
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...select_algorithm import PartialRender
from ...utils import get_fused_kernel_name, get_kernel_metadata
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer


log = logging.getLogger(__name__)


class CuteDSLScheduling(BaseScheduling):
    """
    Scheduling implementation for CuteDSL (CUTLASS Python DSL) kernels.
    This class is intended to be used in combination with other schedulers,
    and delegated to by CUDACombinedScheduling.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_cutedsl_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is a CuteDSL template."""
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CuteDSLTemplateBuffer
        )

    def is_cutedsl_fused_template(self, node: BaseSchedulerNode) -> bool:
        """Check if a node is a fused CuteDSL template."""
        return isinstance(node, FusedSchedulerNode) and self.is_cutedsl_template(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        TODO CuteDSL doesn't support vertical fusion yet.
        This could be extended in the future for epilogue fusion.
        """
        return False

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """CuteDSL doesn't support horizontal fusion yet."""
        return False

    def define_kernel(
        self, src_code_str: str, node_schedule, precompile_metadata=None
    ) -> str:
        """Produce the kernel string
        Args:
            src_code_str: The finalized kernel code string
            node_schedule: List of nodes in the schedule
            precompile_metadata: Optional dict with shapes/dtypes for subprocess precompilation

        Note:
            This is a little weird since async_compile.cutedsl() has to write the string to
            a file in order to cute compile it. Feels bad to have two...
        """
        wrapper = V.graph.wrapper_code

        # Use the string as the key for caching
        if src_code_str in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code_str]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )

            kernel_hash = hashlib.sha256(src_code_str.encode("utf-8")).hexdigest()[:8]
            if fused_name == "fused":
                kernel_name = f"cutedsl_{kernel_hash}"
            else:
                kernel_name = f"cutedsl_{fused_name}_{kernel_hash}"
            wrapper.src_to_kernel[src_code_str] = kernel_name
            src_code_str = src_code_str.replace(
                str(Placeholder.KERNEL_NAME), kernel_name
            )

            _, _, kernel_path = get_path(code_hash(src_code_str), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.cutedsl({kernel_name!r}, r'''")
            compile_wrapper.splice(src_code_str, strip=True)
            if precompile_metadata is not None:
                compile_wrapper.writeline(
                    f"''', precompile_metadata={precompile_metadata!r})"
                )
            else:
                compile_wrapper.writeline("''')")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a CuteDSL template. Currently doesn't support fusion.
        """
        if not self.is_cutedsl_template(template_node):
            raise AssertionError(
                "Template node passed to CuteDSLScheduling.codegen_template must be a "
                "SchedulerNode that wraps a CuteDSLTemplateBuffer"
            )
        # TODO remove when supported
        if epilogue_nodes:
            raise AssertionError("CuteDSL doesn't support epilogue fusion yet")
        if prologue_nodes:
            raise AssertionError("CuteDSL doesn't support prologue fusion yet")

        template_node = cast(SchedulerNode, template_node)
        ctb: CuteDSLTemplateBuffer = cast(CuteDSLTemplateBuffer, template_node.node)

        kernel, render = ctb.make_kernel_render(ctb)  # type: ignore[misc]
        template_node.mark_run()
        src_code = render()
        # Finalize PartialRender if needed
        if isinstance(src_code, PartialRender):
            src_code_str = src_code.finalize_all()
        else:
            src_code_str = src_code

        precompile_metadata = self._build_precompile_metadata(kernel, ctb)

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(
                src_code_str, node_schedule, precompile_metadata
            )
        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()

    def _build_precompile_metadata(self, kernel, ctb):
        """Extract shapes and dtypes from kernel inputs/output for subprocess precompilation.

        Returns None if shapes are symbolic (dynamic shapes), in which case the
        subprocess will skip precompilation and the kernel compiles lazily on first call.
        """
        if not hasattr(kernel, "_template_input_args"):
            return None

        precompile_shapes = {}
        precompile_strides = {}
        precompile_dtypes = {}

        try:
            for arg_name, input_node in kernel._template_input_args:
                template_name = arg_name.removeprefix("arg_")
                size = input_node.get_size()
                precompile_shapes[template_name] = [int(s) for s in size]
                stride = input_node.get_stride()
                precompile_strides[template_name] = [int(s) for s in stride]
                precompile_dtypes[template_name] = str(
                    input_node.get_dtype()
                ).removeprefix("torch.")

            output_size = ctb.layout.size
            precompile_shapes["output"] = [int(s) for s in output_size]
            output_stride = ctb.layout.stride
            precompile_strides["output"] = [int(s) for s in output_stride]
            precompile_dtypes["output"] = str(ctb.layout.dtype).removeprefix("torch.")
        except (TypeError, RuntimeError, ValueError):
            log.debug(
                "Skipping CuteDSL precompile metadata: symbolic sizes cannot be "
                "resolved to concrete values"
            )
            return None

        device = ctb.layout.device
        device_index = device.index if device.index is not None else 0

        import torch

        device_capability = None
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability(device_index)

        metadata: dict[str, object] = {
            "precompile_shapes": precompile_shapes,
            "precompile_strides": precompile_strides,
            "precompile_dtypes": precompile_dtypes,
            "device_index": device_index,
            "device_capability": device_capability,
        }

        hw_info = self._build_hw_info(kernel)
        if hw_info is not None:
            metadata["hw_info"] = hw_info

        return metadata

    @staticmethod
    def _build_hw_info(kernel) -> tuple[int, int] | None:
        """Compute (sm_count, max_active_clusters) in the main process.

        Reads CLUSTER_M/N from kernel template kwargs and queries
        HardwareInfo so subprocess workers never touch CUDA driver APIs.
        Returns None if CUTLASS is unavailable or the template has no cluster config.
        """
        template_kwargs = getattr(kernel, "_template_kwargs", None)
        if template_kwargs is None:
            return None

        cluster_m = template_kwargs.get("CLUSTER_M")
        cluster_n = template_kwargs.get("CLUSTER_N")
        if cluster_m is None or cluster_n is None:
            return None

        try:
            import cutlass.utils

            cluster_product = int(cluster_m) * int(cluster_n)
            hw = cutlass.utils.HardwareInfo()
            sm_count = hw.get_max_active_clusters(1)
            max_active_clusters = hw.get_max_active_clusters(cluster_product)
            return (sm_count, max_active_clusters)
        except Exception:
            log.debug(
                "Could not compute hw_info for CuTe DSL precompile", exc_info=True
            )
            return None
<|endoftext|>"""Lower elementwise inline PTX over CuteDSL fragments.

Each assembly invocation consumes `pack` adjacent fragment elements. LLVM
constraints select the physical register type for every operand, so inputs are
bitcast into those registers rather than numerically converted. Multiple outputs
are returned as an LLVM struct and extracted before rebuilding the fragment.
Partial packs are zero-padded, and outputs for padded elements are discarded.

Integer register results are narrowed to the requested element width. E8M0 codes
1 through 254 map directly to the matching Float32 exponent bits; codes 0 and 255
map explicitly to `2^-127` and NaN. This lets fused arithmetic use exact Float32
values while preserving E8M0 as the stored result type.
"""

import functools
import hashlib
import math
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cute.tensor import _infer_broadcast_shape
from cutlass.cutlass_dsl import dsl_user_op, T


@functools.cache
def inline_asm_cache_key() -> str:
    """Include this module's source in generated kernel cache keys."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# PTX register classes, as accepted by the NVPTX inline-asm backend.
CONSTRAINT_TYPES = {
    "h": T.i16,
    "r": T.i32,
    "l": T.i64,
    "f": T.f32,
    "d": T.f64,
}


def split_constraints(constraints: str) -> tuple[list[str], list[str]]:
    """Split an LLVM constraint list into output and input register classes."""
    outputs, inputs = [], []
    for entry in constraints.split(","):
        entry = entry.strip()
        target = outputs if entry.startswith("=") else inputs
        letter = entry.lstrip("=&")
        if letter not in CONSTRAINT_TYPES:
            raise NotImplementedError(
                f"CuteDSL inline asm supports constraints {sorted(CONSTRAINT_TYPES)}, "
                f"got {entry!r}"
            )
        target.append(letter)
    return outputs, inputs


def bitcast_to(value: ir.Value, target: ir.Type) -> ir.Value:
    """Reinterpret a value's bits as an equally wide type."""
    if value.type == target:
        return value
    return arith.bitcast(target, value)


def narrow_integer(value: ir.Value, width: int) -> ir.Value:
    """Truncate an integer register to `width` bits."""
    if not ir.IntegerType.isinstance(value.type):
        raise TypeError(f"expected an integer inline-asm result, got {value.type}")
    if ir.IntegerType(value.type).width > width:
        return arith.trunci(ir.IntegerType.get_signless(width), value)
    return value


def decode_e8m0(value: ir.Value) -> ir.Value:
    """Decode one biased E8M0 bit pattern to its exact Float32 value."""
    code = arith.extui(T.i32(), narrow_integer(value, 8))
    bits = arith.shli(code, arith.constant(T.i32(), 23))
    is_zero = arith.cmpi(arith.CmpIPredicate.eq, code, arith.constant(T.i32(), 0))
    bits = arith.select(is_zero, arith.constant(T.i32(), 0x00400000), bits)
    is_nan = arith.cmpi(arith.CmpIPredicate.eq, code, arith.constant(T.i32(), 255))
    bits = arith.select(is_nan, arith.constant(T.i32(), 0x7FC00000), bits)
    return arith.bitcast(T.f32(), bits)


def convert_output(value: ir.Value, result_type) -> ir.Value:
    """Convert a register result to the requested logical element type."""
    if result_type == cutlass.Float8E8M0FNU:
        return decode_e8m0(value)
    target = result_type.mlir_type
    if value.type == target:
        return value
    if ir.IntegerType.isinstance(value.type):
        value = narrow_integer(value, result_type.width)
    return bitcast_to(value, target)


def fragment_element(source, index: int) -> ir.Value:
    """Extract element `index` from a TensorSSA, or unwrap a scalar value."""
    if isinstance(source, cute.TensorSSA):
        return vector.extract(source.ir_value(), [], [index])
    if isinstance(source, ir.Value):
        return source
    if isinstance(source, (int, float)):
        return cutlass.Float32(source).ir_value()
    return source.ir_value()


def zero_for_constraint(letter: str) -> ir.Value:
    """Create a zero in the register type named by an inline-asm constraint."""
    target = CONSTRAINT_TYPES[letter]()
    return arith.constant(target, 0.0 if letter in ("f", "d") else 0)


def packed_operands(
    sources,
    input_letters: list[str],
    scalar_sources: tuple[bool, ...],
    base: int,
    count: int,
    pack: int,
) -> list[ir.Value]:
    """Gather one source-major inline-asm pack, padding a partial tail with zero."""
    operands = []
    for source_index, source in enumerate(sources):
        for lane in range(pack):
            letter = input_letters[source_index * pack + lane]
            index = base + lane
            operands.append(
                bitcast_to(
                    fragment_element(
                        source, 0 if scalar_sources[source_index] else index
                    )
                    if index < count
                    else zero_for_constraint(letter),
                    CONSTRAINT_TYPES[letter](),
                )
            )
    return operands


@dsl_user_op
def inline_asm_elementwise_intrinsic(
    *sources,
    asm: str,
    constraints: str,
    result_type,
    is_pure: bool = True,
    pack: int = 1,
    scalar_sources: tuple[bool, ...] = (),
    loc=None,
    ip=None,
):
    """Apply an inline PTX block elementwise across a TensorSSA fragment.

    Args:
        sources: Broadcastable TensorSSA fragments or repeated scalar values.
        asm: PTX text using `$N` operand syntax, output first.
        constraints: LLVM constraint list, e.g. `"=h,r"`.
        result_type: Logical cutlass numeric type of the produced fragment.
            E8M0 integer results are decoded to Float32 for fused consumers.
        is_pure: Whether the block may be reordered and CSEd.
        pack: Elements consumed per asm invocation. Missing tail elements are
            zero-padded and their corresponding outputs are discarded.
        scalar_sources: Sources whose physical TensorSSA value represents one
            logical scalar and must be repeated for every lane.
    """
    if pack < 1:
        raise ValueError(f"CuteDSL inline asm requires pack >= 1, got pack={pack}")
    output_letters, input_letters = split_constraints(constraints)
    if len(output_letters) != pack:
        raise ValueError(
            f"inline asm pack={pack} requires {pack} output constraints, "
            f"got {len(output_letters)} in {constraints!r}"
        )
    expected_inputs = len(sources) * pack
    if len(input_letters) != expected_inputs:
        raise ValueError(
            f"inline asm pack={pack} with {len(sources)} sources requires "
            f"{expected_inputs} input constraints, got {len(input_letters)} "
            f"in {constraints!r}"
        )
    if not scalar_sources:
        scalar_sources = (False,) * len(sources)
    elif len(scalar_sources) != len(sources):
        raise ValueError(
            f"inline asm received {len(scalar_sources)} scalar-source flags "
            f"for {len(sources)} sources"
        )

    output_types = [CONSTRAINT_TYPES[letter]() for letter in output_letters]
    asm_result_type = output_types[0]
    if len(output_types) > 1:
        fields = ", ".join(str(output_type) for output_type in output_types)
        asm_result_type = ir.Type.parse(f"!llvm.struct<({fields})>")
    compute_type = (
        cutlass.Float32 if result_type == cutlass.Float8E8M0FNU else result_type
    )
    fragments = [
        source
        for source, is_scalar in zip(sources, scalar_sources)
        if isinstance(source, cute.TensorSSA) and not is_scalar
    ]
    if fragments:
        shape = fragments[0].shape
        if any(source.shape != shape for source in fragments[1:]):
            shape = _infer_broadcast_shape(*(source.shape for source in fragments))
            sources = tuple(
                source.broadcast_to(shape)
                if isinstance(source, cute.TensorSSA) and not is_scalar
                else source
                for source, is_scalar in zip(sources, scalar_sources)
            )
        first_fragment = next(
            source
            for source, is_scalar in zip(sources, scalar_sources)
            if isinstance(source, cute.TensorSSA) and not is_scalar
        )
        count = math.prod(ir.VectorType(first_fragment.ir_value().type).shape)
    else:
        count = 1
        shape = None

    converted = []
    for base in range(0, count, pack):
        produced = llvm.inline_asm(
            asm_result_type,
            packed_operands(sources, input_letters, scalar_sources, base, count, pack),
            asm,
            constraints,
            has_side_effects=not is_pure,
            is_align_stack=False,
        )
        outputs = [produced]
        if len(output_types) > 1:
            outputs = [
                llvm.extractvalue(output_type, produced, [index])
                for index, output_type in enumerate(output_types)
            ]
        valid_outputs = min(pack, count - base)
        converted.extend(
            convert_output(output, result_type) for output in outputs[:valid_outputs]
        )

    if shape is None:
        return compute_type(converted[0])
    vector_type = ir.VectorType.get([count], compute_type.mlir_type)
    return cute.TensorSSA(
        vector.from_elements(vector_type, converted), shape, compute_type
    )
<|endoftext|># mypy: disable-error-code=import-not-found
# pyrefly: ignore [import-error, missing-import]
import cutlass
import cutlass.cute as cute


@cute.jit  # type: ignore[misc]
def ssa_to_indexable(ssa_value: cute.TensorSSA, dtype: str) -> cute.Numeric:
    """
    Convert SSA form to indexable non-SSA form.

    Workaround for lack of gather support: SSA values cannot be used directly
    as indices in tensor loads. This converts SSA to a register fragment, then
    extracts a scalar for indexing.
    """
    frag = ssa_to_fragment(ssa_value, dtype)
    return frag[0]


@cute.jit  # type: ignore[misc]
def ssa_to_fragment(ssa_value: cute.TensorSSA, dtype: str) -> cute.Tensor:
    """Materialize an SSA vector into a register fragment.

    Casts to the fragment dtype when the SSA dtype differs (e.g. an int64 index
    expression materialized into an int32 index fragment), since fragment
    stores require matching element types.
    """
    if cutlass.const_expr(ssa_value.dtype != dtype):
        ssa_value = ssa_value.to(dtype)
    frag = cute.make_rmem_tensor(ssa_value.shape, dtype)
    frag.store(ssa_value)
    return frag


@cute.jit  # type: ignore[misc]
def result_to_ssa(value: cute.Numeric, dtype: str) -> cute.TensorSSA:
    """
    Convert non-SSA result back to SSA form.

    After performing operations with non-SSA values (like indexed loads),
    convert the result back to SSA form for further computation.
    """
    frag = cute.make_rmem_tensor(1, dtype)
    frag[0] = value
    return frag.load()
<|endoftext|># mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

from torch._inductor.utils import Placeholder, unique
from torch._inductor.virtualized import V

from ...autotune_process import CuteDSLBenchmarkRequest, TensorMeta
from ...ir import Buffer, ChoiceCaller, CuteDSLTemplateBuffer, IRNode, Layout, TensorBox
from ..common import KernelTemplate
from .cutedsl_kernel import CuteDSLTemplateKernel


log = logging.getLogger(__name__)


class CuteDSLTemplate(KernelTemplate):
    """Template for generating CuteDSL (CUTLASS Python DSL) kernels.

    Subclasses may override ``caller_type`` to attach template-specific
    metadata or precompile behavior while reusing the common render and
    benchmark request construction.
    """

    kernel_type: type[Any] = CuteDSLTemplateKernel
    caller_type: type[Any] | None = None
    index_counter = itertools.count()
    all_templates: dict[str, "CuteDSLTemplate"] = {}

    def __init__(
        self,
        name: str,
        source: str,
        subgraph_fn: Any | None = None,
        mask_fn: Any | None = None,
    ) -> None:
        super().__init__(name)
        self.source = source
        self.subgraph_fn = subgraph_fn
        self.mask_fn = mask_fn
        self.template = CuteDSLTemplate._template_from_string(source)
        # A module that registers templates can be initialized more than once in
        # a single process (e.g. a double-import path). Tolerate re-registration
        # under an existing name as long as the template source matches, but
        # reject a genuine name collision between different templates.
        existing = self.all_templates.get(name)
        if existing is not None and existing.source != self.source:
            raise AssertionError(f"duplicate template name, {name}")
        CuteDSLTemplate.all_templates[name] = self

    @staticmethod
    @functools.lru_cache(None)
    # pyrefly: ignore [bad-override]
    def _template_from_string(source: str) -> Any:
        return KernelTemplate._template_from_string(source)

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> NotImplementedError | None:
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.
        """
        try:
            choices.append(self.generate(**kwargs))
            return None
        except NotImplementedError as e:
            log.debug("CuteDSL template choice generation failed: %s", e)
            return e
        except Exception as e:
            log.debug("CuteDSL template choice generation error: %s", e)
            return NotImplementedError(f"CuteDSL template failed: {e}")

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        """Generate the CuteDSL kernel caller for template autotuning."""
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        mutated_inputs = kwargs.pop("mutated_inputs", None)
        subgraphs = kwargs.pop("subgraphs", None)
        template_kwargs = dict(kwargs)

        kernel_name = f"cutedsl_{self.name}_{next(self.index_counter)}"

        if self.template is None:
            raise RuntimeError("Template compilation failed (Jinja2 required)")

        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        # Patch V.graph.get_dtype to handle the fake buf_out buffer
        with patch.object(
            V.graph, "get_dtype", KernelTemplate._fake_get_dtype(self.output_node)
        ):
            kernel = self.kernel_type(
                kernel_name=kernel_name,
                input_nodes=input_nodes,
                output_node=self.output_node,
                subgraphs=subgraphs,
            )
            code = kernel.render(self.template, **kwargs)

            input_call_args = tuple(kernel.args.input_buffers.keys())
            expected_input_args = tuple(unique(x.get_name() for x in input_nodes))
            if input_call_args[: len(expected_input_args)] != expected_input_args:
                raise RuntimeError(
                    "CuteDSL template input registration order changed while "
                    "collecting captured subgraph buffers. Expected template "
                    "inputs to be registered before captured buffers, got "
                    f"{input_call_args}, expected prefix {expected_input_args}."
                )
            extra_capture_names = input_call_args[len(expected_input_args) :]

            # Resolve captured nodes from the graph-level side table
            # (populated by realize_captures_for_cutedsl) to get view nodes.
            graph_captures = getattr(V.graph, "_cutedsl_capture_nodes", {})
            capture_nodes_by_name: dict[str, Any] = {}
            extra_capture_nodes = []
            for name in extra_capture_names:
                node = graph_captures.get(name)
                if node is None:
                    node = V.graph.get_buffer(name)
                capture_nodes_by_name[name] = node
                extra_capture_nodes.append(node)
            input_nodes = list(input_nodes) + extra_capture_nodes

            kernel.set_capture_input_nodes(capture_nodes_by_name)
            with kernel._patch_get_dtype_for_captures():
                _, call_args, _, _ = kernel.args.python_argdefs()
            expected_args = list(input_call_args)
            expected_args.append(self.output_node.get_name())
            if list(call_args)[: len(expected_args)] != expected_args:
                raise RuntimeError(
                    "CuteDSL template benchmark argument order changed while "
                    "collecting dynamic scalar args. Expected prefix "
                    f"{expected_args}, got {list(call_args)}."
                )
            extra_args = tuple(
                V.graph.sizevars.optimization_hints(call_args[len(expected_args) :])
            )

            bmreq = CuteDSLBenchmarkRequest(
                kernel_name=kernel_name,
                input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
                output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
                extra_args=extra_args,
                source_code=code,
            )

            def make_kernel_render(out_node, hint_override: int | None = None):
                """
                Factory function that creates a kernel renderer for the final output.

                This closure captures the current template and parameters, but allows
                the output node to be specified later. This is used during the final
                kernel selection phase when the actual output buffer is available.
                """
                render_kernel = self.kernel_type(
                    kernel_name=str(Placeholder.KERNEL_NAME),
                    input_nodes=input_nodes,
                    output_node=out_node,
                    subgraphs=subgraphs,
                )
                render_kernel.set_capture_input_nodes(capture_nodes_by_name)

                def render():
                    return render_kernel.render(self.template, **kwargs)

                return render_kernel, render

            caller_type = self.caller_type or CuteDSLTemplateCaller
            return caller_type(
                name=kernel_name,
                input_nodes=input_nodes,
                layout=layout,
                make_kernel_render=make_kernel_render,
                bmreq=bmreq,
                template=self,
                mutated_inputs=mutated_inputs,
                template_kwargs=template_kwargs,
            )


class CuteDSLTemplateCaller(ChoiceCaller):
    """Caller for CuteDSL templates that integrates with the autotuning system."""

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Any,
        bmreq: CuteDSLBenchmarkRequest,
        template: "CuteDSLTemplate",
        mutated_inputs: Iterable[IRNode] | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ):
        description = self._build_description(name, template_kwargs)
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=description,
        )
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.mutated_inputs = mutated_inputs

    def _build_description(
        self, name: str, template_kwargs: dict[str, Any] | None
    ) -> str:
        if not template_kwargs:
            return f"CuteDSL template {name}"
        kwargs_desc = ", ".join(f"{k}={v}" for k, v in template_kwargs.items())
        return f"CuteDSL template {name} ({kwargs_desc})"

    def __str__(self) -> str:
        return f"CuteDSLTemplateCaller({self.name})"

    def benchmark(self, *args, out) -> float:
        """Benchmark the kernel execution."""
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> TensorBox:
        """Create the output node for this template choice."""
        buffer = CuteDSLTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            template=self.template,
            mutated_inputs=self.mutated_inputs,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return TensorBox.create(buffer)

    def call_name(self) -> str:
        """Return the kernel call name."""
        return self.name

    def to_callable(self) -> Any:
        """Return callable that can execute this kernel."""
        return self.make_kernel_render

    def hash_key(self) -> str:
        """Return unique hash key for this choice."""
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def info_dict(self) -> dict[str, Any]:
        """Return information about this kernel."""
        return {
            "name": self.name,
            "backend": "CuteDSL",
            "template": self.template.name,
        }
<|endoftext|># mypy: allow-untyped-defs

import dataclasses

import sympy

from torch.utils._ordered_set import OrderedSet

from ...virtualized import V


@dataclasses.dataclass(frozen=True, slots=True)
class LaneExprInfo:
    """How an index expression changes across vectorized lanes.

    A lane is one element of a vectorized CuteDSL computation. For a 32-lane
    mask/load, lane 0 is the first element in the group and lane 31 is the last.

    Attributes:
        is_uniform: All lanes evaluate to the same value, e.g. for flex-flash``b_idx``.
        is_contiguous: Lanes evaluate to consecutive aligned values for the
            requested width, e.g. ``kv_idx + lane`` when lane 0 is aligned.
        contiguous_width: Largest aligned contiguous power-of-two width proven,
            or None if no vector-width contiguity is proven.
    """

    is_uniform: bool
    is_contiguous: bool
    contiguous_width: int | None


def classify_lane_expr(
    expr: sympy.Expr,
    lane_var: sympy.Symbol,
    *,
    max_width: int,
    uniform_symbols: OrderedSet[sympy.Symbol] | None = None,
) -> LaneExprInfo:
    """Classify how ``expr`` varies across ``lane_var`` vector lanes.

    ``max_width`` is the requested vector width. We start with this as the
    maximum width to check and then decrease in powers of two trying to find a width
    that satisfies contiguity across lanes. We cap since we are trying to get vectorized loads
    which are at 128 and 256(B200) bits so no need to check higher.

    For example a ``lane_var = kv_idx`` and ``max_width = 8``:

    - No dependency on lane var: ``q_idx`` returns
      ``LaneExprInfo(is_uniform=True, is_contiguous=False, contiguous_width=None)``.
    - ``kv_idx`` at an 8-aligned group start returns
      ``LaneExprInfo(is_uniform=False, is_contiguous=True, contiguous_width=8)``.
    - ``kv_idx + 4`` at a 4-aligned but not 8-aligned group start returns
      ``LaneExprInfo(is_uniform=False, is_contiguous=False, contiguous_width=4)``.
    - ``kv_idx * 2`` returns
      ``LaneExprInfo(is_uniform=False, is_contiguous=False, contiguous_width=None)``.
    """
    if max_width <= 1:
        return LaneExprInfo(True, False, None)

    semantic_expr = V.graph.sizevars.simplify(expr)
    if uniform_symbols is not None and semantic_expr.free_symbols <= uniform_symbols:
        return LaneExprInfo(True, False, None)

    lane_contiguity = V.graph.sizevars.analyze_lane_contiguity(semantic_expr, lane_var)
    contiguous_width = aligned_contiguous_width(
        semantic_expr,
        lane_var,
        max_width=max_width,
        lane_contiguity=lane_contiguity,
    )
    return LaneExprInfo(
        is_uniform=lane_contiguity.is_uniform_for(max_width),
        is_contiguous=lane_contiguity.stride == 1 and contiguous_width == max_width,
        contiguous_width=contiguous_width,
    )


def aligned_contiguous_width(
    expr: sympy.Expr,
    lane_var: sympy.Symbol,
    *,
    max_width: int,
    lane_contiguity=None,
) -> int | None:
    """Return the largest aligned power-of-two contiguous lane width."""
    if not (max_width >= 2 and max_width.bit_count() == 1):
        raise AssertionError(f"max_width must be a power of two >= 2, got {max_width}")
    if lane_contiguity is None:
        lane_contiguity = V.graph.sizevars.analyze_lane_contiguity(expr, lane_var)
    start_expr = lane_group_start(expr, lane_var)
    width = max_width
    while width >= 2:
        if (
            lane_contiguity.is_contiguous_for(width)
            and V.graph.sizevars.statically_known_multiple_of(start_expr, width)
            and V.graph.sizevars.statically_known_geq(start_expr, 0)
        ):
            return width
        width //= 2
    return None


def lane_group_start(expr: sympy.Expr, lane_var: sympy.Symbol) -> sympy.Expr:
    """Return the lane-group base value by evaluating ``expr`` at lane 0.

    For example, if ``expr = base + lane_var``, this returns ``base``. This is
    useful when checking vector-load alignment: a contiguous lane expression is only
    safe to vectorize when the first lane's address is sufficiently aligned.
    """
    return V.graph.sizevars.simplify(expr.xreplace({lane_var: sympy.Integer(0)}))


def decompose_affine_lane_expr(
    expr: sympy.Expr, lane_var: sympy.Symbol
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Return ``(stride, base)`` when ``expr`` is affine in ``lane_var``.

    The result satisfies ``expr == stride * lane_var + base``, where neither
    ``stride`` nor ``base`` depends on ``lane_var``.
    For example: ``stride == 0`` means lane-uniform, ``stride == 1`` means
    consecutive lanes access consecutive indices, and other strides are
    non-contiguous gathers.

    Examples with ``lane_var = i``:
        ``b + i`` -> ``(1, b)``
        ``b + 2 * i`` -> ``(2, b)``
        ``b`` -> ``(0, b)``

    Returns ``None`` for non-affine expressions or when the extracted stride or
    base still depends on ``lane_var``.
    """
    expr = V.graph.sizevars.simplify(expr)
    lane_coeff = expr.coeff(lane_var)
    rest = lane_group_start(expr, lane_var)
    if lane_var in lane_coeff.free_symbols or lane_var in rest.free_symbols:
        return None
    residual = V.graph.sizevars.simplify(expr - (lane_coeff * lane_var + rest))
    if residual != 0:
        return None
    return lane_coeff, rest
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import dataclasses
import logging
import textwrap
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import sympy

import torch
from torch._inductor import config
from torch._inductor.codegen.common import (
    CSE,
    CSEVariable,
    IndentedBuffer,
    Kernel,
    PythonPrinter,
    ValueRanges,
)
from torch._inductor.ir import (
    BaseView,
    Buffer,
    ComputedBuffer,
    ExternKernel,
    InputBuffer,
    MutableBox,
    ReinterpretView,
)
from torch._inductor.ops_handler import StoreMode
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V

from ...utils import sympy_index_symbol
from .aux_scalars import CuteDSLAuxScalarBindings
from .cutedsl_op_overrides import CuteDSLCSEVariable, CuteDSLOpOverrides
from .lane_analysis import classify_lane_expr


# TODO setting the 'main' kernel w/ this suffix. We have 3 should probably just auto generate this
MAIN_SUFFIX = "main"


log = logging.getLogger(__name__)
kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


class CuteDSLPrinter(PythonPrinter):
    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("ToFloat expects exactly one argument")
        return f"cutlass.Float64({self.doprint(expr.args[0])})"


cutedsl_pexpr = CuteDSLPrinter().doprint


class CuteDSLKernelWrapper:
    """Wrapper to provide .run() interface for CuteDSL kernels"""

    def __init__(self, kernel_fn: Callable[..., Any], kernel_path: str | None = None):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path
        kernel_code_log.info("CuteDSL kernel path: %s", kernel_path)

    def run(self, *args, stream=None, **kwargs):
        """
        Execute the CuteDSL kernel.

        Args:
            *args: Arguments to pass to the kernel function
            stream: CUDA stream to pass to the kernel function
            **kwargs: Additional keyword arguments for the kernel

        Returns:
            Result of the kernel execution
        """
        return self.kernel_fn(*args, stream=stream, **kwargs)


@dataclasses.dataclass(frozen=True, slots=True)
class CuteDSLVectorLoadConfig:
    vec_size: int
    index: sympy.Symbol
    dim: int

    @staticmethod
    def from_fixed_inputs(
        fixed_inputs: dict[str, Any],
    ) -> "CuteDSLVectorLoadConfig | None":
        vec_size = fixed_inputs.pop("vector_load_vec_size", None)
        index = fixed_inputs.pop("vector_load_index", None)
        dim = fixed_inputs.pop("vector_load_dim", None)
        if vec_size is None or index is None or dim is None:
            return None
        return CuteDSLVectorLoadConfig(
            vec_size=int(vec_size),
            index=sympy_index_symbol(str(index)),
            dim=int(dim),
        )

    def normalized_dim(self, ndim: int) -> int:
        return self.dim + ndim if self.dim < 0 else self.dim


@dataclasses.dataclass(frozen=True, slots=True)
class CuteDSLIndexFragment:
    """Generated index expression metadata for CuteDSL tensor loads.

    Attributes:
        code: Python source expression for the index or index fragment.
        is_static_int: True when code is an inline integer index, not a fragment.
        is_lane_uniform: True when every vector lane has the same index.
        is_contiguous: True when code is aligned and contiguous for the full
            requested vector width.
    """

    code: str
    is_static_int: bool
    is_lane_uniform: bool = False
    is_contiguous: bool = False
    contiguous_width: int | None = None


@dataclasses.dataclass(frozen=True)
class IndexSymbolInfo:
    """What lane analysis may assume about a generated index symbol.

    expr: Recovered semantic index expression, or None when the value's
        semantics are unknown (loaded from a captured tensor, or produced by
        ops outside the traced index algebra). Unknown semantics force a
        per-lane gather: assuming lane-uniform would broadcast lane 0's load
        across the vectorized group (#188878).
    contiguous_ok: False when the value passed through a negative-index wrap
        (idx + size for negative lanes only). Uniform lanes all wrap
        identically, so lane-uniformity survives, but a contiguous window
        straddling 0 jumps from size-1 back to 0 and we can't prove a window
        avoids 0.
    """

    expr: sympy.Expr | None
    contiguous_ok: bool = True


@dataclasses.dataclass
class CuteDSLSubgraphInfo:
    """Minimal subgraph info for CuteDSL kernels."""

    body: IndentedBuffer
    template_mask: str | None = None
    template_out: str | None = None
    cse: CSE[Any, str] | None = None

    def __post_init__(self):
        self.only_copy_if_non_none_fields = ("cse",)

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class CuteDSLTemplateKernel(Kernel):
    """
    Template kernel implementation for CuteDSL (CUTLASS Python DSL).
    Handles code generation and argument management for CuteDSL CUDA kernels.
    Provides CuteDSL-specific functionality for tensor conversion and kernel configuration.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        subgraphs: list[Buffer] | None = None,
    ) -> None:
        # Call parent Kernel constructor
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.subgraphs = subgraphs
        self.subgraph_bodies: dict[str, CuteDSLSubgraphInfo] = {}

        # Template attributes
        self.body: IndentedBuffer = IndentedBuffer()
        self.template_mask: str | None = None
        self.template_out: str | None = None
        self.template_indices: list[Any] | None = None
        self.render_hooks: dict[str, Any] = {}

        # TODO Additional attributes needed by template system
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self.named_input_nodes: dict[str, Buffer] = {}

        # Create named input nodes mapping
        for i, input_node in enumerate(input_nodes):
            node_name = getattr(input_node, "name", f"input_{i}")
            self.named_input_nodes[node_name] = input_node

        self.cse = CSE(name_prefix="tmp")

        # Track all tensor buffers added during modification processing
        self.collected_tensor_buffers: list[str] = []
        self.aux_scalar_bindings = CuteDSLAuxScalarBindings()

        # Captured IR nodes keyed by buffer name. Used by call_kernel to emit
        # reinterpret_tensor for view captures in the python_argdefs loop.
        self._capture_input_nodes: dict[str, Any] = {}

    def set_capture_input_nodes(self, nodes_by_name: dict[str, Any]) -> None:
        """Set captured inputs collected while rendering the CuteDSL template."""
        self._capture_input_nodes = nodes_by_name

    def _get_capture_input_node(self, name: str) -> Any | None:
        """Return the IR node for a captured input name, if known."""
        graph_captures = getattr(V.graph, "_cutedsl_capture_nodes", {})
        return self._capture_input_nodes.get(name, graph_captures.get(name))

    @contextlib.contextmanager
    def _patch_get_dtype_for_captures(self):
        """Teach python_argdefs to resolve synthetic captured input names."""
        original_get_dtype = V.graph.get_dtype

        def get_dtype(name: str) -> torch.dtype:
            capture = self._get_capture_input_node(name)
            if capture is not None:
                return capture.get_dtype()
            return original_get_dtype(name)

        with patch.object(V.graph, "get_dtype", get_dtype):
            yield

    def kexpr(self, expr: sympy.Expr) -> str:
        """Convert sympy expression to CuteDSL string representation."""
        symbol_codes = self.aux_scalar_bindings.symbol_codes_with_renames(
            self.rename_indexing
        )
        if symbol_codes:
            expr = expr.xreplace(
                {symbol: sympy.Symbol(code) for symbol, code in symbol_codes.items()}
            )
        return cutedsl_pexpr(expr)

    def create_cse_var(self, *args, **kwargs):
        return CuteDSLCSEVariable(*args, **kwargs)

    def gen_imports(self) -> str:
        """Generate common imports for CuteDSL templates."""
        imports = IndentedBuffer()
        imports.splice(
            """
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            import cuda.bindings.driver as cuda
            from cutlass._mlir.dialects import math as mlir_math
            import operator
            from torch._inductor.codegen.cutedsl._cutedsl_utils import result_to_ssa, ssa_to_fragment, ssa_to_indexable
            """
        )
        return imports.getvalue()

    def gen_defines(self, **kwargs) -> str:
        """Generate CuteDSL parameter definitions from kwargs, similar to Triton's gen_defines."""
        params = IndentedBuffer()
        for name, val in kwargs.items():
            params.writeline(f"{name}: cutlass.Constexpr = {val}")
        return params.getvalue()

    def render(self, template, **kwargs):
        from torch._inductor.select_algorithm import PartialRender

        """Render the kernel using the template, returning PartialRender object with hooks."""
        self._template_kwargs = dict(kwargs)
        self.aux_scalar_bindings = CuteDSLAuxScalarBindings(
            tuple(kwargs.get("AUX_SCALAR_SYMBOLS", ()))
        )

        # Available {{}} hooks for jinja rendering
        template_env = {
            "def_kernel": self.def_kernel,
            "gen_defines": lambda: self.gen_defines(**kwargs),
            "get_output": self.get_output,
            "get_tensor_buffers": self.get_tensor_buffers,
            "add_tensor_inputs": self.add_tensor_inputs,
            "define_aux_scalars": self.define_aux_scalars,
            "unpack_buffers": self.unpack_buffers,
            "modification": self.modification,
            "set_cute_hash": self.set_cute_hash,
        }

        # Render the template with the environment and provided kwargs
        rendered_code = template.render(
            kernel_name=self.kernel_name,
            input_nodes=self.input_nodes,
            output_node=self.output_node,
            **template_env,
            **kwargs,
        )

        # Always prepend the common imports
        imports = self.gen_imports()
        full_code = imports + rendered_code

        return PartialRender(full_code, self.render_hooks)

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        """Set the active subgraph body for template processing."""
        if not all(
            hasattr(self, field.name)
            for field in dataclasses.fields(CuteDSLSubgraphInfo)
        ):
            raise AssertionError(
                "expected all CuteDSLSubgraphInfo fields to be set on self"
            )
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(CuteDSLSubgraphInfo)
        }

        if body_name not in self.subgraph_bodies:
            self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
                body=IndentedBuffer(),
                template_mask=None,
                template_out=None,
                cse=None,
            )

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key in getattr(
                subgraph, "only_copy_if_non_none_fields", ()
            ):
                continue
            setattr(self, key, value)

        try:
            yield
        finally:
            # Save current state back to subgraph
            self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
                **{
                    key.name: getattr(self, key.name)
                    for key in dataclasses.fields(CuteDSLSubgraphInfo)
                }
            )
            # Restore old state
            for key, value in old_state.items():
                setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str, *, clear_cse: bool = False):
        """Create a new subgraph body for template processing."""
        if body_name in self.subgraph_bodies:
            raise AssertionError(f"Subgraph body '{body_name}' already exists")
        new_cse = self.cse.clone() if clear_cse else None
        self.subgraph_bodies[body_name] = CuteDSLSubgraphInfo(
            body=IndentedBuffer(),
            template_mask=None,
            template_out=None,
            cse=new_cse,
        )
        with self.set_subgraph_body(body_name):
            yield

    def _get_reinterpret_view(self, node) -> ReinterpretView | None:
        """Extract or convert to ReinterpretView from a node, handling all views."""
        while isinstance(node, MutableBox):
            node = node.data
        if isinstance(node, BaseView):
            return ExternKernel.convert_to_reinterpret_view(node)
        return None

    def def_kernel(self, *argnames):
        """Define kernel function signature for CuteDSL templates.

        When inputs are ReinterpretViews of the same underlying buffer (e.g., Q/K/V
        from fused QKV projection), we generate separate arguments for each input
        even though they share the same underlying buffer.
        """
        renames = IndentedBuffer(initial_indent=1)

        # Track template input args - each input gets its own arg even if buffers are shared
        self._template_input_args: list[tuple[str, Buffer]] = []
        self._seen_input_args: OrderedSet[str] = OrderedSet()

        for i, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            # Register with args system (may deduplicate, but we track separately)
            self.args.input(buf_name)

            if i < len(argnames):
                template_name = argnames[i]
                arg_name = f"arg_{template_name}"
                self.args.input_buffers[buf_name] = arg_name
                renames.writeline(f"{template_name} = {arg_name}")
                self._template_input_args.append((arg_name, input_node))
                self._seen_input_args.add(arg_name)

        if self.output_node:
            self.args.output(self.output_node.get_name())

        def hook():
            # Generate signature with template input args plus additional args (output, sizevars)
            code = IndentedBuffer()
            code.writeline(f"# Kernel function signature: {self.kernel_name}")

            # Start with template input args
            params = [arg_name for arg_name, _ in self._template_input_args]

            # Get additional args from python_argdefs (output, sizevars, etc.)
            with self._patch_get_dtype_for_captures():
                arg_defs, _, _, _ = self.args.python_argdefs()
            for arg_def in arg_defs:
                if arg_def.full_name() not in self._seen_input_args:
                    params.append(arg_def.full_name())

            params.append("stream")
            code.writeline(
                f"def {self.kernel_name}_{MAIN_SUFFIX}({', '.join(params)}):"
            )
            with code.indent():
                code.splice(renames.getvalue())
            return code.getvalue()

        if "<DEF_KERNEL>" in self.render_hooks:
            raise AssertionError("<DEF_KERNEL> hook already registered")
        # Placeholder-based rendering: hook will be called when template encounters "<DEF_KERNEL>"
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    def get_output(self):
        """Get the actual argument name for the output buffer."""
        if not self.output_node:
            raise AssertionError("Output node must exist to get output buffer name")
        buf_name = self.output_node.get_name()
        output = self.args.output_buffers.get(buf_name, None)
        if output is None:
            raise ValueError(f"Output buffer '{buf_name}' not found in args")
        return output

    def set_cute_hash(self, func_name: str, suffix: str = ""):
        """Generate code to set __cute_hash__ on a codegen function.

        This allows hash_callable in flash_attn to skip expensive runtime hashing
        for Inductor-generated functions. The hash is based on the kernel name
        which already contains a unique hash suffix.
        """
        hash_value = f"{self.kernel_name}_{suffix}" if suffix else self.kernel_name
        return f'{func_name}.__cute_hash__ = "{hash_value}"'

    def get_tensor_buffers(self):
        """Get list of tensor buffer names that were collected during modifications."""
        return self.collected_tensor_buffers

    def add_tensor_inputs(self, buffers):
        """Register extra tensor buffers and return their rendered input names."""
        buffer_names = []
        for buffer in buffers:
            if isinstance(buffer, sympy.Expr):
                continue
            remapped_name = self.args.input(buffer.get_name())
            if remapped_name not in self.collected_tensor_buffers:
                self.collected_tensor_buffers.append(remapped_name)
            buffer_names.append(remapped_name)
        return buffer_names

    def define_aux_scalars(self, symbols):
        """Return the runtime scalar tuple for captured symbolic scalars."""
        if tuple(symbols) != self.aux_scalar_bindings.symbols:
            raise AssertionError(
                f"Expected aux scalar symbols {self.aux_scalar_bindings.symbols}, "
                f"got {tuple(symbols)}"
            )
        return self.aux_scalar_bindings.tuple_expr(self.rename_indexing, cutedsl_pexpr)

    def unpack_buffers(self, buffer_list_name: str, *, indent_width: int = 4):
        """Generate buffer unpacking code via render hook."""

        def hook():
            tensor_buffers = self.get_tensor_buffers()
            if not tensor_buffers:
                return ""

            # Generate unpacking assignments: in_ptr4 = buffers[0], etc.
            unpacking_lines = []
            for i, buffer_name in enumerate(tensor_buffers):
                # pyrefly: ignore [bad-argument-type]
                unpacking_lines.append(f"{buffer_name} = {buffer_list_name}[{i}]")

            indent = " " * indent_width
            return "\n" + indent + ("\n" + indent).join(unpacking_lines)

        # Register the hook and return placeholder
        placeholder = "<UNPACK_BUFFERS>"
        # TODO: I think double invoking is fine for this specific hook
        # assert placeholder not in self.render_hooks
        self.render_hooks[placeholder] = hook
        return placeholder

    def call_kernel(self, name: str, node=None):
        """Call the kernel function. Simplified version of TritonTemplateKernel.call_kernel.

        For inputs that are ReinterpretViews (e.g., Q/K/V slices from fused QKV),
        we generate reinterpret_tensor() calls to properly handle the views.
        """
        wrapper = V.graph.wrapper_code

        # Build call args matching the signature generated in `def_kernel`
        call_args = []
        arg_types = []

        for _, input_node in self._template_input_args:
            reinterpret_view = self._get_reinterpret_view(input_node)
            if reinterpret_view is not None:
                call_args.append(reinterpret_view.codegen_reference())
            else:
                call_args.append(input_node.get_name())
            arg_types.append(V.graph.get_dtype(input_node.get_name()))

        # Add additional args from python_argdefs (output, sizevars, ..)
        with self._patch_get_dtype_for_captures():
            orig_arg_defs, orig_call_args, _, orig_arg_types = (
                self.args.python_argdefs()
            )
        for arg_def, call_arg, arg_type in zip(
            orig_arg_defs, orig_call_args, orig_arg_types
        ):
            if arg_def.full_name() in self._seen_input_args:
                continue
            capture = self._get_capture_input_node(call_arg)
            if isinstance(capture, ReinterpretView):
                # Pass the original view expression so stride/offset metadata is
                # preserved at the kernel call site.
                call_args.append(capture.codegen_reference())
            else:
                call_args.append(call_arg)
            arg_types.append(arg_type)

        # TODO this karg really should not be called `triton`
        wrapper.generate_kernel_call(name, call_args, triton=True, arg_types=arg_types)

    def _get_subgraph(self, subgraph_number: int):
        """Get subgraph by number for modification processing."""
        if not isinstance(subgraph_number, int):
            raise AssertionError(
                f"expected subgraph_number to be int, got {type(subgraph_number)}"
            )
        if not isinstance(self.subgraphs, list):
            raise AssertionError(
                f"expected self.subgraphs to be list, got {type(self.subgraphs)}"
            )
        if subgraph_number >= len(self.subgraphs):
            raise AssertionError(
                f"Invalid subgraph number provided to create_modification, {subgraph_number} must be < {len(self.subgraphs)}"
            )
        if self.body.getvalue() != "":
            raise AssertionError("Body should be clear before adding a modification")
        return self.subgraphs[subgraph_number]

    def modification(
        self,
        subgraph_number: int,
        output_name: str | None,
        mask: str | None = None,
        **fixed_inputs,
    ) -> str:
        """Generate CuteDSL code for a subgraph modification."""
        # Find unique name to avoid collisions between multiple modifications of same subgraph
        num = 0
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1

        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}", clear_cse=True):
            subgraph = self._get_subgraph(subgraph_number)
            modification_handler = ModificationWrapperCuteDSL(
                self, subgraph_number, fixed_inputs, mask
            )
            with V.set_kernel_handler(self), V.set_ops_handler(modification_handler):
                if not isinstance(subgraph, (ComputedBuffer, list)):
                    raise AssertionError(
                        f"Expected ComputedBuffer or List[ComputedBuffer], got {type(subgraph)}"
                    )

                if isinstance(subgraph, list):
                    raise NotImplementedError(
                        "Scatter graphs are not supported for CuteDSL"
                    )

                if isinstance(subgraph.data, InputBuffer):
                    # grad_score_mod can be InputBuffers
                    out = subgraph.data.make_loader()(())
                else:
                    # Inline a pointwise lowering into the template
                    out = subgraph.data.inner_fn(())

            if output_name is not None:
                if out is None:
                    raise AssertionError(
                        f"Expected computation result for named output {output_name}"
                    )
                self.body.writeline(f"{output_name} = {out.value}")
            else:
                # Side-effect only: no output assignment (currently only for scatter operations)
                raise NotImplementedError(
                    "Side-effect only modifications not yet supported for CuteDSL"
                )

            # Add Buffers that were added during modification
            self.collected_tensor_buffers.extend(modification_handler.tensor_buffers)

            return self.body.getvalue()


class ModificationWrapperCuteDSL(V.WrapperHandler):  # type: ignore[name-defined]
    """
    Wrapper handler that enables CuteDSL code generation during subgraph modifications.

    This class sits between the PyTorch IR and CuteDSL code generation, providing:
    1. Operation substitution: converts PyTorch ops to CuteDSL equivalents via CuteDSLOpOverrides
    2. Placeholder handling: resolves fixed_inputs during template processing
    3. Limited operation support: currently restricted to pointwise operations

    """

    def __init__(
        self,
        kernel,
        subgraph_number: int,
        fixed_inputs: dict[str, Any],
        mask: str | None,
    ):
        cutedsl_ops = CuteDSLOpOverrides()
        super().__init__(cutedsl_ops)
        self.name = f"CuteDSLPlaceholderSubstitution_{subgraph_number}"
        self.kernel = kernel
        self.fixed_inputs = dict(fixed_inputs)
        self.vector_load_config = CuteDSLVectorLoadConfig.from_fixed_inputs(
            self.fixed_inputs
        )
        self.mask = mask
        # Track tensor buffers that get added during modification processing
        self.tensor_buffers: list[str] = []
        # Maps generated CSE names to their semantic provenance so vector-load
        # analysis can reason about the source indexing semantics.
        self._index_symbol_info: dict[sympy.Symbol, IndexSymbolInfo] = {}
        # Captured modifications are read-only, so repeated loads from the same
        # semantic location can share the generated index and value fragments.
        self._captured_load_cache: dict[tuple[str, sympy.Expr], CuteDSLCSEVariable] = {}

    def _get_input_dtype(self, name: str) -> torch.dtype:
        """Get the dtype for an input from the kernel's named_input_nodes."""
        if name in self.kernel.named_input_nodes:
            return self.kernel.named_input_nodes[name].dtype
        # TODO: Fallback for common dimension names - should be replaced with proper dtype tracking
        return torch.float32 if name not in ("b", "h", "m", "n") else torch.int32

    def load(self, name: str, index: sympy.Expr):
        """Handle loading from tensor or fixed(template args) input for CuteDSL."""
        from torch._inductor.kernel.flex.flex_flash_attention import HierarchicalIndex

        if name not in self.fixed_inputs:
            cache_key = (name, index)
            cached = self._captured_load_cache.get(cache_key)
            if cached is not None:
                return self.kernel.cse.generate(self.kernel.body, cached)

            var = self._add_kernel_input(name)
            buffer = self.kernel._get_capture_input_node(name)
            if buffer is None:
                buffer = V.graph.get_buffer(name)
            var_dtype = buffer.dtype

            cute_dtype = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
                var_dtype, "cutlass.Float32"
            )
            dim_indices = (
                index.args if isinstance(index, HierarchicalIndex) else (index,)
            )
            dim_sizes = buffer.get_size()
            if len(dim_sizes) == 0:
                dim_indices = ()
            # Per-dimension indices into captured buffers are bounded by dim
            # sizes (not flattened offsets), so Int32 is safe whatever index
            # dtype the consuming kernel passes in; wider index expressions are
            # narrowed at fragment materialization.
            idx_vars = [
                self._emit_index_fragment(
                    dim_index, dim_size, "cutlass.Int32", torch.int32
                )
                for dim_index, dim_size in zip(dim_indices, dim_sizes, strict=True)
            ]

            val_frag = self._emit_tensor_load_fragment(
                var, idx_vars, cute_dtype, var_dtype, buffer
            )
            final_expr = f"{val_frag}.load()"

            if (
                var_dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                final_expr = f"({final_expr}).to(cutlass.Float32)"
                var_dtype = torch.float32

            out = self.kernel.cse.generate(
                self.kernel.body,
                final_expr,
                dtype=var_dtype,
                bounds=ValueRanges.unknown(),
                shape=(1,),
            )
            self._captured_load_cache[cache_key] = out
            return out

        value = self.fixed_inputs[name]
        dtype = self._get_input_dtype(name)

        result = self.kernel.cse.generate(
            self.kernel.body, value, bounds=ValueRanges.unknown(), dtype=dtype
        )
        if (
            isinstance(result, CuteDSLCSEVariable)
            and isinstance(value, str)
            and dtype in (torch.int32, torch.int64)
        ):
            result.index_expr = sympy_index_symbol(value)
        return result

    def _emit_tensor_load_fragment(
        self,
        var: str,
        idx_vars: list[CuteDSLIndexFragment],
        cute_dtype: str,
        var_dtype: torch.dtype,
        buffer: Any,
    ) -> str:
        """
        Emit a register fragment load from a captured tensor.

        Static integer indices can be emitted directly as scalar tensor indices.
        Dynamic indices are SSA vectors, which must first be materialized into
        register fragments so each lane can be used for scalar gather loads.  If
        all indices are static, emit a single-element fragment; otherwise size
        the value fragment to match the dynamic index lanes.
        """
        dynamic_index_fragment = next(
            (idx for idx in idx_vars if not idx.is_static_int), None
        )
        # Dynamic index fragments all carry the same vector width, so one
        # representative fragment is enough to size the loaded value fragment.
        val_frag = self.kernel.cse.newvar(dtype=var_dtype)
        if len(buffer.get_size()) == 0:
            self.kernel.body.writeline(
                f"{val_frag} = cute.make_rmem_tensor(1, {cute_dtype})"
            )
            self.kernel.body.writeline(f"{val_frag}[0] = ({var}[()])")
            return str(val_frag)
        if dynamic_index_fragment is None:
            self.kernel.body.writeline(
                f"{val_frag} = cute.make_rmem_tensor(1, {cute_dtype})"
            )
            self.kernel.body.writeline(
                f"{val_frag}[0] = ({var}[{', '.join(idx.code for idx in idx_vars)}])"
            )
            return str(val_frag)

        self.kernel.body.writeline(
            f"{val_frag} = cute.make_rmem_tensor("
            f"cute.size({dynamic_index_fragment.code}.shape), {cute_dtype})"
        )
        if self.vector_load_config is None:
            vector_dim = len(idx_vars) - 1
        else:
            vector_dim = self.vector_load_config.normalized_dim(len(idx_vars))
        if self._can_emit_contiguous_dim_load(idx_vars, buffer, var_dtype, vector_dim):
            self._emit_contiguous_dim_load(
                var, idx_vars, val_frag, cute_dtype, var_dtype, vector_dim
            )
        elif self._can_emit_uniform_load(idx_vars):
            self._emit_uniform_load(var, idx_vars, val_frag)
        else:
            self._emit_per_lane_gather_load(var, idx_vars, val_frag)
        return str(val_frag)

    def _can_emit_contiguous_dim_load(
        self,
        idx_vars: list[CuteDSLIndexFragment],
        buffer: Any,
        var_dtype: torch.dtype,
        vector_dim: int,
    ) -> bool:
        """Return true when vector lanes form a safe contiguous tensor tile.

        This is stricter than lane-contiguity analysis: CuTe autovec also needs
        a real byte-addressable copy type, uniform non-vector dimensions, unit
        vector stride, and a statically aligned full-width tile.
        """
        if var_dtype is torch.bool:
            return False
        if not idx_vars or not idx_vars[vector_dim].is_contiguous:
            return False
        if any(
            not (idx.is_static_int or idx.is_lane_uniform)
            for dim, idx in enumerate(idx_vars)
            if dim != vector_dim
        ):
            return False
        sizes = buffer.get_size()
        strides = buffer.get_stride()
        vector_size = (
            1 if self.vector_load_config is None else self.vector_load_config.vec_size
        )
        contiguous_width = idx_vars[vector_dim].contiguous_width
        has_vector_width = vector_size > 1
        has_unit_stride = V.graph.sizevars.statically_known_equals(
            strides[vector_dim], 1
        )
        has_full_width_contiguity = (
            contiguous_width is not None and contiguous_width >= vector_size
        )
        is_size_aligned = V.graph.sizevars.statically_known_multiple_of(
            sizes[vector_dim], vector_size
        )
        is_offset_aligned = V.graph.sizevars.statically_known_multiple_of(
            buffer.get_layout().offset, vector_size
        )
        are_non_vector_strides_aligned = all(
            V.graph.sizevars.statically_known_multiple_of(stride, vector_size)
            for dim, stride in enumerate(strides)
            if dim != vector_dim
        )
        return (
            has_vector_width
            and has_unit_stride
            and has_full_width_contiguity
            and is_size_aligned
            and is_offset_aligned
            and are_non_vector_strides_aligned
        )

    def _emit_contiguous_dim_load(
        self,
        var: str,
        idx_vars: list[CuteDSLIndexFragment],
        val_frag: str,
        cute_dtype: str,
        var_dtype: torch.dtype,
        vector_dim: int,
    ) -> None:
        """Emit an autovec load for lanes contiguous in one tensor dimension."""
        scalar_indices = [
            idx.code if idx.is_static_int else f"{idx.code}[0]" for idx in idx_vars
        ]
        vec_size = f"cute.size({val_frag}.shape)"
        aligned_idx = self.kernel.cse.newvar(dtype=torch.int32)
        source_tile = self.kernel.cse.newvar(dtype=var_dtype)
        aligned_ptr = self.kernel.cse.newvar(dtype=var_dtype)
        aligned_source = self.kernel.cse.newvar(dtype=var_dtype)

        vector_idx = idx_vars[vector_dim].code
        tiler_items = ["1"] * len(idx_vars)
        tiler_items[vector_dim] = vec_size
        coord_items = scalar_indices
        coord_items[vector_dim] = f"{aligned_idx} // {vec_size}"
        source_view_items = ["0"] * len(idx_vars)
        source_view_items[vector_dim] = "None"
        source_view = (
            str(aligned_source)
            if len(idx_vars) == 1
            else f"{aligned_source}[{', '.join(source_view_items)}]"
        )
        tiler = (
            f"({tiler_items[0]},)"
            if len(tiler_items) == 1
            else f"({', '.join(tiler_items)})"
        )
        coord = (
            f"({coord_items[0]},)"
            if len(coord_items) == 1
            else f"({', '.join(coord_items)})"
        )
        self.kernel.body.splice(
            f"""
            {aligned_idx} = cute.assume({vector_idx}[0], divby={vec_size})
            {source_tile} = cute.local_tile({var}, {tiler}, {coord})
            {aligned_ptr} = cute.make_ptr({cute_dtype}, {source_tile}.iterator.toint(), {source_tile}.iterator.memspace, assumed_align=min(16, {vec_size} * {var_dtype.itemsize}))
            {aligned_source} = cute.make_tensor({aligned_ptr}, {source_tile}.layout)
            cute.autovec_copy({source_view}, {val_frag})
            """
        )

    @staticmethod
    def _can_emit_uniform_load(idx_vars: list[CuteDSLIndexFragment]) -> bool:
        """Return true when every vector lane reads the same tensor element."""
        return all(idx.is_static_int or idx.is_lane_uniform for idx in idx_vars)

    def _emit_uniform_load(
        self, var: str, idx_vars: list[CuteDSLIndexFragment], val_frag: str
    ) -> None:
        """Emit one scalar load and broadcast it across the value fragment."""
        scalar_value = self.kernel.cse.newvar(dtype=None)
        scalar_indices = [
            idx.code if idx.is_static_int else f"{idx.code}[0]" for idx in idx_vars
        ]
        self.kernel.body.writeline(
            f"{scalar_value} = ({var}[{', '.join(scalar_indices)}])"
        )
        self.kernel.body.writeline(
            f"for load_idx in cutlass.range(cute.size({val_frag}.shape), unroll_full=True):"
        )
        with self.kernel.body.indent():
            self.kernel.body.writeline(f"{val_frag}[load_idx] = {scalar_value}")

    def _emit_per_lane_gather_load(
        self, var: str, idx_vars: list[CuteDSLIndexFragment], val_frag: str
    ) -> None:
        """Emit scalar per-lane loads for non-contiguous gather indices.

        Lane-uniform index fragments hold a single element, so they are indexed
        at [0] rather than [load_idx] (which would read out of bounds and
        silently corrupt the gathered rows).
        """
        self.kernel.body.writeline(
            f"for load_idx in cutlass.range(cute.size({val_frag}.shape), unroll_full=True):"
        )
        with self.kernel.body.indent():
            load_indices = [
                idx.code
                if idx.is_static_int
                else f"{idx.code}[0]"
                if idx.is_lane_uniform
                else f"{idx.code}[load_idx]"
                for idx in idx_vars
            ]
            self.kernel.body.writeline(
                f"{val_frag}[load_idx] = ({var}[{', '.join(load_indices)}])"
            )

    def _emit_index_fragment(
        self,
        expr: sympy.Expr,
        dim_size: sympy.Expr,
        cute_dtype: str,
        torch_dtype: torch.dtype,
    ) -> CuteDSLIndexFragment:
        """Materialize an index and preserve enough metadata for load selection."""
        static_expr = self._static_integer_expr(expr)
        if static_expr is not None:
            return CuteDSLIndexFragment(
                self.kernel.kexpr(self._wrap_static_index(static_expr, dim_size)),
