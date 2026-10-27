    ) -> Generator[TritonConfig, None, None]:
        """
        Finalizes configs after scaling, applying additional constraints.
        """
        used: OrderedSet[tuple[int | None, ...]] = OrderedSet()

        max_mm_configs = config.test_configs.max_mm_configs

        for conf in configs:
            # Each warp computes a 16x16 tile = 256 elements
            num_warps = min(conf.num_warps, conf.block_m * conf.block_n // 256)

            # Construct key for finding duplicate configs
            key: tuple[int | None, ...] = (
                conf.block_m,
                conf.block_n,
                conf.block_k,
                conf.num_stages,
                conf.hint_override,
                num_warps,
            )

            # Check if gemm specific arg exists - add to key if does
            group_m = getattr(conf, "group_m", None)
            if group_m is not None:
                key += (group_m,)

            # Add BlackwellGPUGemmConfig specific fields to key if present
            if isinstance(conf, BlackwellGPUGemmConfig):
                key += (
                    conf.epilogue_subtile,
                    conf.warp_specialize,
                    conf.flatten,
                    conf.use_meta_ws,
                    conf.data_partition_factor,
                    conf.separate_epilogue_store,
                )

            extra_key, extra_kwargs = self._get_extra_config_key_and_kwargs(conf)
            key += extra_key

            if key not in used and (
                max_mm_configs is None or len(used) < max_mm_configs
            ):
                used.add(key)
                kwargs: dict[str, Any] = {
                    "BLOCK_M": conf.block_m,
                    "BLOCK_N": conf.block_n,
                    "BLOCK_K": conf.block_k,
                    "hint_override": conf.hint_override,
                }
                if group_m is not None:
                    kwargs["GROUP_M"] = group_m

                # Add BlackwellGPUGemmConfig specific fields if present
                if isinstance(conf, BlackwellGPUGemmConfig):
                    kwargs["EPILOGUE_SUBTILE"] = conf.epilogue_subtile
                    kwargs["WARP_SPECIALIZE"] = conf.warp_specialize
                    kwargs["FLATTEN"] = conf.flatten
                    kwargs["USE_META_WS"] = conf.use_meta_ws
                    kwargs["DATA_PARTITION_FACTOR"] = conf.data_partition_factor
                    kwargs["SEPARATE_EPILOGUE_STORE"] = conf.separate_epilogue_store

                kwargs.update(extra_kwargs)

                yield self.triton_config(conf.num_stages, num_warps, **kwargs)

    def _scale_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        scale: float,
        has_int8_tensor: bool,
        exclude: Callable[[sympy.Integer, sympy.Integer, sympy.Integer], bool],
        hint_override: int | None = None,
    ) -> list[BaseConfig]:
        """
        Scales and filters matrix multiplication configs based on input size.
        """
        if not self.should_scale_configs:
            return configs
        from ...runtime.runtime_utils import next_power_of_2

        min_block_size = 16
        min_block_size_k = 32 if (has_int8_tensor or self.has_int8_tensor) else 16

        scaled_configs = []
        for hint_override in [None] + config.multi_kernel_hints:
            m_hint = max(
                next_power_of_2(
                    V.graph.sizevars.optimization_hint_with_override(
                        m,
                        hint_override=hint_override,
                    )
                ),
                min_block_size,
            )
            n_hint = max(
                next_power_of_2(
                    V.graph.sizevars.optimization_hint_with_override(
                        n,
                        hint_override=hint_override,
                    )
                ),
                min_block_size,
            )
            k_hint = max(
                next_power_of_2(
                    V.graph.sizevars.optimization_hint_with_override(
                        k,
                        hint_override=hint_override,
                    )
                ),
                min_block_size_k,
            )

            for c in configs:
                block_m = max(min(int(c.block_m * scale), m_hint), min_block_size)
                block_n = max(min(int(c.block_n * scale), n_hint), min_block_size)
                block_k = max(min(int(c.block_k * scale), k_hint), min_block_size_k)
                if not exclude(block_m, block_n, block_k):
                    # This copy is expensive, so avoid it if we can.
                    if (block_m, block_n, block_k, hint_override) != (
                        c.block_m,
                        c.block_n,
                        c.block_k,
                        c.hint_override,
                    ):
                        c = dataclasses.replace(
                            c,
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            hint_override=hint_override,
                        )

                    scaled_configs.append(c)

        return scaled_configs

    # Estimate theoretical maximum shared memory
    def get_shared_memory_estimation(
        self,
        gemm_config: BaseConfig,
        dtype_size: int,
        has_sm_layout_conversion: bool,
        layout_conversion_byte_size: int,
    ):
        shared_mem_loads = dtype_size * (
            gemm_config.block_m * gemm_config.block_k
            + gemm_config.block_n * gemm_config.block_k
        )

        # Extra bytes to account for barriers in boundary conditions
        extra_bytes = 128

        # In persistent tma case, the layout conversion from mma -> blocked layout
        # is not free and takes additional shared memory, while next loads are prefetched
        # For addmm, the conversion is in the acc dtype, as it is needed before the bias addition
        # For mm, the conversion is in the output dtype, as it happens before the store
        if has_sm_layout_conversion:
            element_bits = layout_conversion_byte_size * 8
            # 8 bytes of padding for fp16/bf16
            max_padding = 128 // element_bits
            block_n = max_padding + gemm_config.block_n
            shared_mem_epilogue = (
                layout_conversion_byte_size * gemm_config.block_m * block_n
            )
        else:
            shared_mem_epilogue = 0

        return (
            shared_mem_loads * gemm_config.num_stages
            + shared_mem_epilogue
            + extra_bytes
        )

    def _get_exceeding_shared_memory_checker(
        self,
        has_sm_layout_conversion: bool,
        layout_conversion_byte_size: int,
    ) -> Callable[[BaseConfig, int], bool] | None:
        """
        Returns a function that checks whether a given configuration exceeds the available shared memory for the device.
        based on the config's theoretical maximum shared memory used.
        If the device does not report available shared memory, returns None.
        """

        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                if hasattr(props, "shared_memory_per_block_optin"):  # for NVidia GPUs
                    sm_available = int(props.shared_memory_per_block_optin)
                elif hasattr(props, "shared_memory_per_block"):  # for ROCm
                    sm_available = int(props.shared_memory_per_block)
                else:
                    return None
            elif torch.xpu.is_available():
                props = torch.xpu.get_device_properties()
                sm_available = int(props.local_mem_size)
            else:
                return None
        except Exception:
            # If device properties cannot be queried, return None
            return None

        # TODO make a BaseDeviceConfigHeuristics to handle different device configuration in its own implementation.
        def exceeds(gemm_config: BaseConfig, dtype_size: int) -> bool:
            estimation = self.get_shared_memory_estimation(
                gemm_config,
                dtype_size,
                has_sm_layout_conversion,
                layout_conversion_byte_size,
            )
            return estimation > sm_available

        return exceeds

    def _prune_exceeding_max_shared_mem_configs(
        self,
        configs: list[BaseConfig],
        dtype_size: int,
        has_sm_layout_conversion: bool = False,
        layout_conversion_byte_size: int = 0,
    ) -> list[BaseConfig]:
        if dtype_size <= 0:
            return configs

        is_exceeding_shared_memory = self._get_exceeding_shared_memory_checker(
            has_sm_layout_conversion, layout_conversion_byte_size
        )
        if is_exceeding_shared_memory is None:
            return configs

        return [c for c in configs if not is_exceeding_shared_memory(c, dtype_size)]

    def _prune_reg_spill_configs(
        self,
        configs: list[BaseConfig],
    ) -> list[BaseConfig]:
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            warp_size = props.warp_size
        except (RuntimeError, AttributeError, AssertionError):
            return configs

        pruned_configs = []
        # NVIDIA-oriented approximation for the per-thread register limit.
        # TODO: generalize this threshold for ROCm devices.
        NUM_REG = 255
        for gemm_config in configs:
            acc_regs = math.ceil(
                gemm_config.block_m
                * gemm_config.block_n
                / (gemm_config.num_warps * warp_size)
            )
            # Lower bound for register spillage, if exceeds the kernel will certainly spill
            if acc_regs > NUM_REG:
                continue

            pruned_configs.append(gemm_config)

        return pruned_configs

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        Filter configs based on specific requirements.
        Subclasses can override this to implement custom filtering logic.
        """
        return configs

    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = False,
        scale: float = 1.0,
        exclude: Callable[
            [sympy.Integer, sympy.Integer, sympy.Integer], bool
        ] = lambda m, n, k: False,
        dtype_size: int = 0,
        op_name: str = "mm",  # For preprocessing overrides e.g. on CPU
        **kwargs,
    ) -> Generator[TritonConfig, None, None]:
        configs = self._filter_configs(configs)
        scaled_configs = self._scale_mm_configs(
            m, n, k, configs, scale, has_int8_tensor, exclude
        )

        # Filter out configs that require more shared memory than is available.
        # Theoretical upper bound, will over-prune configs. Off by default for maximum
        # performance
        if config.max_autotune_prune_choices_based_on_shared_mem:
            scaled_configs = self._prune_exceeding_max_shared_mem_configs(
                scaled_configs,
                dtype_size,
                kwargs.get("has_sm_layout_conversion", False),
                kwargs.get("layout_conversion_byte_size", 0),
            )

        if config.max_autotune_gemm_search_space == "EXHAUSTIVE":
            scaled_configs = self._prune_reg_spill_configs(scaled_configs)
        return self._finalize_mm_configs(scaled_configs)

    def triton_config(
        self, num_stages: int, num_warps: int, **kwargs: Any
    ) -> TritonConfig:
        return TritonConfig(kwargs, num_stages=num_stages, num_warps=num_warps)

    def get_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(self.preprocess_mm_configs, configs=self.mm_configs)

    def get_exhaustive_mm_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        """Get exhaustive MM configs for origami optimization.

        Note: When these configs are used with origami (on ROCm with
        _origami_enabled()=True and max_autotune_gemm_search_space="DEFAULT"),
        origami will filter them to the top-K configurations. Origami currently
        only supports Triton backend, so only Triton-compatible configs will be
        used. ATEN backend is excluded from origami search space.
        """
        return partial(self.preprocess_mm_configs, configs=self.exhaustive_configs)

    def get_conv_configs(self) -> partial[Generator[TritonConfig, None, None]]:
        return partial(
            self.preprocess_mm_configs, configs=self.conv_configs, op_name="conv"
        )

    def get_depthwise_conv_configs(self) -> list[TritonConfig]:
        """Return TritonConfig list for depthwise conv1d autotuning."""
        return [
            TritonConfig(
                {
                    "BLOCK_N": cfg.block_n,
                    "BLOCK_L": cfg.block_l,
                    "BLOCK_C": cfg.block_c,
                },
                num_stages=cfg.num_stages,
                num_warps=cfg.num_warps,
            )
            for cfg in self.depthwise_conv_configs
        ]

    # Flex attn helpers
    def get_flex_attn_fwd_configs(
        self, head_dim: int, seq_len: sympy.Expr, dtype: Any
    ) -> list[FlexConfig]:
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = FlexConfig(64, 64, 3, 4)
            else:
                default_config = FlexConfig(128, 64, 3, 4)
        else:
            if dtype == torch.float32:
                default_config = FlexConfig(32, 16, 3, 4)
            else:
                default_config = FlexConfig(64, 32, 3, 4)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs

    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexBwDConfig]:
        flex_attn_bwd_configs: list[FlexBwDConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        default_config = FlexBwDConfig(16, 16, 16, 16, 1, 4)

        if default_config not in flex_attn_bwd_configs:
            flex_attn_bwd_configs.append(default_config)

        return flex_attn_bwd_configs

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        default_config = FlexDecodeConfig(block_n=64, num_stages=1, num_warps=2)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs


class CPUConfigHeuristic(BaseConfigHeuristic):
    """
    CPU-specific config heuristic with CPU-specific optimizations.
    """

    def _get_cpu_exclude_function(
        self, method: str = "bmm"
    ) -> Callable[[sympy.Integer, sympy.Integer, sympy.Integer], bool]:
        """
        Get CPU-specific exclude function based on method type.
        Returns a function that can be used as exclude condition.
        Moved from mm_common._is_large_block_for_cpu and refactored to return a function.
        """
        if method in ("conv"):

            def exclude_conv(
                m: sympy.Integer, n: sympy.Integer, k: sympy.Integer
            ) -> bool:
                # Thresholds are experimentally determined to reduce Triton CPU compile times
                if m > 256 or n > 256 or k > 256:
                    return True
                return m * n * k > 2**17

            return exclude_conv
        elif method in ("mm", "addmm", "int_mm"):

            def exclude_mm(
                m: sympy.Integer, n: sympy.Integer, k: sympy.Integer
            ) -> bool:
                return m * n > 2**13

            return exclude_mm
        else:  # Default to bmm implementation for unknown methods

            def exclude_bmm(
                m: sympy.Integer, n: sympy.Integer, k: sympy.Integer
            ) -> bool:
                if m > 128 or n > 128 or k > 128:
                    return True
                return m * n > 2**12

            return exclude_bmm

    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = False,
        scale: float = 1.0,
        exclude: Callable[
            [sympy.Integer, sympy.Integer, sympy.Integer], bool
        ] = lambda m, n, k: False,
        dtype_size: int = 0,
        op_name: str = "mm",  # For preprocessing overrides e.g. on CPU
        **kwargs,
    ) -> Generator[TritonConfig, None, None]:
        """
        CPU-specific preprocessing that applies CPU-specific scaling (0.5) and exclusion logic.
        """
        # Get CPU-specific exclude function based on operation type
        cpu_exclude_fn = self._get_cpu_exclude_function(op_name)

        # Apply CPU-specific scaling (0.5) and exclusion logic
        return super().preprocess_mm_configs(
            m,
            n,
            k,
            configs=configs,
            has_int8_tensor=has_int8_tensor,
            scale=0.5,
            exclude=cpu_exclude_fn,
            dtype_size=dtype_size,
            op_name=op_name,
            **kwargs,
        )


class CUDAConfigHeuristic(BaseConfigHeuristic):
    """
    Child class for CUDA device specific gemm/flex attention/conv/ configs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sm_120_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 2, 4),
            (torch.float32, 128): FlexConfig(128, 32, 2, 4),
            (torch.float32, 256): FlexConfig(64, 16, 2, 4),
            (torch.bfloat16, 64): FlexConfig(128, 64, 2, 4),
            (torch.bfloat16, 128): FlexConfig(128, 64, 2, 8),
            (torch.bfloat16, 256): FlexConfig(32, 64, 2, 4),
            (torch.float16, 64): FlexConfig(128, 64, 2, 4),
            (torch.float16, 128): FlexConfig(128, 64, 2, 8),
            (torch.float16, 256): FlexConfig(32, 64, 2, 4),
        }

        self.sm_100_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 3, 4),
            (torch.float32, 128): FlexConfig(32, 64, 3, 4),
            (torch.float32, 192): FlexConfig(32, 64, 2, 4),
            (torch.float32, 256): FlexConfig(32, 32, 3, 4),
            (torch.bfloat16, 64): FlexConfig(128, 64, 3, 4),
            (torch.bfloat16, 128): FlexConfig(128, 128, 2, 8),
            (torch.bfloat16, 192): FlexConfig(128, 128, 1, 8),
            (torch.bfloat16, 256): FlexConfig(64, 32, 3, 4),
            (torch.float16, 64): FlexConfig(128, 64, 3, 4),
            (torch.float16, 128): FlexConfig(128, 128, 2, 8),
            (torch.float16, 192): FlexConfig(128, 128, 1, 8),
            (torch.float16, 256): FlexConfig(64, 32, 3, 4),
        }

        self.h100_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 3, 4),
            (torch.float32, 128): FlexConfig(32, 64, 3, 4),
            (torch.float32, 192): FlexConfig(32, 64, 1, 8),
            (torch.float32, 256): FlexConfig(32, 32, 3, 4),
            (torch.bfloat16, 64): FlexConfig(128, 128, 3, 4),
            (torch.bfloat16, 128): FlexConfig(128, 64, 3, 8),
            (torch.bfloat16, 256): FlexConfig(64, 32, 3, 4),
            (torch.float16, 64): FlexConfig(128, 128, 3, 4),
            (torch.float16, 128): FlexConfig(128, 64, 3, 8),
            (torch.float16, 256): FlexConfig(64, 32, 3, 4),
        }

        self.a100_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 3, 4),
            (torch.float32, 128): FlexConfig(128, 32, 3, 4),
            (torch.float32, 256): FlexConfig(64, 16, 3, 4),
            (torch.bfloat16, 64): FlexConfig(128, 64, 3, 4),
            (torch.bfloat16, 128): FlexConfig(128, 64, 3, 8),
            (torch.bfloat16, 192): FlexConfig(128, 32, 2, 8),
            (torch.bfloat16, 256): FlexConfig(32, 64, 3, 4),
            (torch.float16, 64): FlexConfig(128, 64, 3, 4),
            (torch.float16, 128): FlexConfig(128, 64, 3, 8),
            (torch.float16, 192): FlexConfig(128, 32, 2, 8),
            (torch.float16, 256): FlexConfig(32, 64, 3, 4),
        }

        # Overwriting the configs omitting BLOCK_N of size 128 that cause ULFs
        self.flex_attn_bwd_autotune_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            FlexBwDConfig(BLOCK_M, BLOCK_N, BLOCK_N, BLOCK_M, s, 4)
            for BLOCK_M in [32, 64]
            for BLOCK_N in [32, 64]
            for s in [1, 3, 4, 5]  # num_stages
            if BLOCK_N % BLOCK_M == 0
        ]

    def get_flex_attn_fwd_configs(
        self, head_dim: int, seq_len: sympy.Expr, dtype: Any
    ) -> list[FlexConfig]:
        capability = torch.cuda.get_device_capability()
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = FlexConfig(64, 64, 3, 4)
            else:
                default_config = FlexConfig(64, 64, 3, 4)
            # here we are using sm_120_default_flex_config on THOR as well
            if capability >= (11, 0):
                default_config = self.sm_120_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
            elif capability >= (10, 0):
                default_config = self.sm_100_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
            elif capability == (9, 0):
                default_config = self.h100_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
            elif capability >= (8, 0):
                default_config = self.a100_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
        else:
            if dtype == torch.float32:
                default_config = FlexConfig(32, 16, 3, 4)
            else:
                default_config = FlexConfig(64, 32, 3, 4)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs

    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexBwDConfig]:
        capability = torch.cuda.get_device_capability()
        flex_attn_bwd_configs: list[FlexBwDConfig] = []
        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        major, minor = capability
        if dtype == torch.float32:
            capability_class = "float32"
        elif major == 12:
            capability_class = "sm12x"
        elif major == 11:
            capability_class = "sm11x"
        elif major >= 10:
            capability_class = "sm10x"
        elif capability == (9, 0):
            capability_class = "sm90"
        elif major >= 8:
            capability_class = "sm8x"
        else:
            capability_class = "baseline"

        # fmt: off
        config_map = {
            "float32": lambda h: FlexBwDConfig(16, 16, 16, 16, 1, 4),
            "baseline": lambda h: FlexBwDConfig(16, 16, 16, 16, 1, 4),
            "sm90": lambda h: (
                FlexBwDConfig(64, 64, 64, 64, 3, 4) if h < 64 else
                FlexBwDConfig(64, 128, 128, 64, 3, 8) if h <= 128 else
                FlexBwDConfig(64, 64, 64, 64, 2, 4)
            ),
            "sm10x": lambda h: (
                FlexBwDConfig(32, 64, 64, 32, 3, 4) if h <= 64 else
                FlexBwDConfig(64, 128, 128, 64, 3, 4) if h <= 128 else
                FlexBwDConfig(64, 64, 64, 64, 1, 8) if h <= 192 else
                FlexBwDConfig(64, 64, 64, 64, 1, 4)
            ),
            "sm8x": lambda h: (
                FlexBwDConfig(32, 128, 128, 32, 3, 4)
                if h < 64
                else FlexBwDConfig(
                    64, 64, 64, 64, 3 if minor == 6 and h == 128 else 2, 4
                )
            ),
            "sm11x": lambda h: (
                FlexBwDConfig(32, 128, 128, 32, 3, 4)
                if h < 64
                else FlexBwDConfig(
                    64, 64, 64, 64, 1 if h >= 128 else 2, 4
                )
            ),
            "sm12x": lambda h: (
                FlexBwDConfig(32, 128, 128, 32, 3, 4)
                if h < 64
                else FlexBwDConfig(
                    64, 64, 64, 64, 1 if h >= 128 else 2, 4
                )
            ),
        }
        # fmt: on

        if head_dim <= 256:
            default_config = config_map[capability_class](head_dim)
        else:
            default_config = FlexBwDConfig(16, 16, 16, 16, 1, 4)

        if default_config not in flex_attn_bwd_configs:
            flex_attn_bwd_configs.append(default_config)

        return flex_attn_bwd_configs

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        capability = torch.cuda.get_device_capability()

        default_config = FlexDecodeConfig(64, 1, 2)

        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        if capability in [(9, 0), (10, 0), (10, 3)]:  # sm_90, sm_100, sm_103
            if head_dim > 128 and dtype == torch.float32:
                default_config = FlexDecodeConfig(64, 1, 2)
            else:
                default_config = FlexDecodeConfig(64, 3, 2)
        if capability == (11, 0):
            default_config = FlexDecodeConfig(16, 1, 2)
        else:
            default_config = FlexDecodeConfig(64, 1, 2)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs


class ROCmConfigHeuristic(BaseConfigHeuristic):
    """
    Child class for ROCm specific gemm/flex attention/conv/ configs.
    """

    def __init__(self) -> None:
        super().__init__()

        self.default_num_stages = get_backend_num_stages()

        self.mm_configs: list[BaseConfig] = [
            ROCmGemmConfig(
                16, 16, 256, self.default_num_stages, 4, group_m=4, waves_per_eu=2
            ),
            ROCmGemmConfig(32, 16, 256, self.default_num_stages, 4, group_m=4),
            ROCmGemmConfig(
                32, 32, 16, self.default_num_stages, 4, group_m=8, waves_per_eu=2
            ),
            ROCmGemmConfig(32, 32, 128, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(32, 64, 64, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(
                64, 16, 128, self.default_num_stages, 4, group_m=8, waves_per_eu=2
            ),
            ROCmGemmConfig(64, 32, 32, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(64, 32, 64, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(64, 32, 64, self.default_num_stages, 8, group_m=8),
            ROCmGemmConfig(64, 32, 128, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(64, 64, 16, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(64, 64, 64, self.default_num_stages, 4, group_m=4),
            ROCmGemmConfig(64, 64, 128, self.default_num_stages, 8, group_m=16),
            ROCmGemmConfig(64, 64, 256, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(
                64, 128, 32, self.default_num_stages, 4, group_m=4, waves_per_eu=2
            ),
            ROCmGemmConfig(64, 128, 32, self.default_num_stages, 8, group_m=8),
            ROCmGemmConfig(64, 128, 64, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(64, 128, 128, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(128, 32, 32, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(128, 32, 64, self.default_num_stages, 4, group_m=8),
            ROCmGemmConfig(
                128, 64, 32, self.default_num_stages, 4, group_m=8, waves_per_eu=2
            ),
            ROCmGemmConfig(128, 64, 64, self.default_num_stages, 4, group_m=16),
            ROCmGemmConfig(128, 64, 128, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(
                128, 128, 32, self.default_num_stages, 4, group_m=16, waves_per_eu=2
            ),
            ROCmGemmConfig(128, 128, 32, self.default_num_stages, 8, group_m=16),
            ROCmGemmConfig(
                128, 128, 32, self.default_num_stages, 8, group_m=16, waves_per_eu=2
            ),
            ROCmGemmConfig(128, 128, 64, self.default_num_stages, 4, group_m=16),
            ROCmGemmConfig(128, 128, 64, self.default_num_stages, 8, group_m=8),
            ROCmGemmConfig(128, 128, 128, self.default_num_stages, 8, group_m=16),
            ROCmGemmConfig(
                128, 256, 32, self.default_num_stages, 4, group_m=16, waves_per_eu=2
            ),
            ROCmGemmConfig(128, 256, 64, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(256, 64, 64, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(
                256, 128, 32, self.default_num_stages, 4, group_m=4, waves_per_eu=2
            ),
            ROCmGemmConfig(256, 128, 32, self.default_num_stages, 8, group_m=16),
            ROCmGemmConfig(256, 128, 64, self.default_num_stages, 8, group_m=4),
            ROCmGemmConfig(256, 256, 64, self.default_num_stages, 8, group_m=4),
        ]

        # Exhaustive search for mm configs
        self.exhaustive_configs: list[BaseConfig] = [
            ROCmGemmConfig(
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_stages,
                num_warps,
                group_m=group_m,
                matrix_instr_nonkdim=matrix_instr_nonkdim,
                waves_per_eu=waves_per_eu,
                kpack=kpack,
            )
            for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
                [16, 32, 64, 128, 256], repeat=3
            )
            for num_stages in [1, self.default_num_stages]
            for num_warps in [4, 8]
            for group_m in [4, 8, 16]
            for matrix_instr_nonkdim in [0, 16]
            for waves_per_eu in [0, 2]
            for kpack in [1, 2]
        ]

        # Architecture-aware default kpack for flex configs
        default_kpack = get_default_kpack()

        self.gfx950_default_flex_config = {
            (torch.float32, 64): ROCmFlexConfig(128, 32, 1, 4, kpack=default_kpack),
            (torch.float32, 128): ROCmFlexConfig(128, 32, 1, 4, kpack=default_kpack),
            (torch.float32, 256): ROCmFlexConfig(64, 16, 1, 4, kpack=default_kpack),
            (torch.bfloat16, 64): ROCmFlexConfig(128, 64, 2, 4, kpack=default_kpack),
            (torch.bfloat16, 128): ROCmFlexConfig(128, 64, 2, 4, kpack=default_kpack),
            (torch.bfloat16, 256): ROCmFlexConfig(32, 64, 2, 4, kpack=default_kpack),
            (torch.float16, 64): ROCmFlexConfig(128, 64, 2, 4, kpack=default_kpack),
            (torch.float16, 128): ROCmFlexConfig(128, 64, 2, 4, kpack=default_kpack),
            (torch.float16, 256): ROCmFlexConfig(32, 64, 2, 4, kpack=default_kpack),
        }

        self.gfx942_default_flex_config = {
            (torch.float32, 64): ROCmFlexConfig(128, 32, 1, 4, kpack=default_kpack),
            (torch.float32, 128): ROCmFlexConfig(128, 32, 1, 4, kpack=default_kpack),
            (torch.float32, 256): ROCmFlexConfig(64, 16, 1, 4, kpack=default_kpack),
            (torch.bfloat16, 64): ROCmFlexConfig(64, 64, 2, 8, kpack=default_kpack),
            (torch.bfloat16, 128): ROCmFlexConfig(64, 64, 2, 8, kpack=default_kpack),
            (torch.bfloat16, 256): ROCmFlexConfig(32, 64, 2, 8, kpack=default_kpack),
            (torch.float16, 64): ROCmFlexConfig(64, 32, 1, 8, kpack=default_kpack),
            (torch.float16, 128): ROCmFlexConfig(64, 32, 1, 8, kpack=default_kpack),
            (torch.float16, 256): ROCmFlexConfig(32, 32, 1, 8, kpack=default_kpack),
        }

        # RDNA3 (gfx11xx) optimal configs profiled on gfx1151 (head_dim=256).
        # Three seq_len tiers to match CU occupancy on 16-CU RDNA3:
        #   short  (seq < 128): BLOCK_M=16, tiny tiles keep all CUs busy
        #   medium (128 <= seq < 512): BLOCK_M=32, transitional tile size
        #   long   (seq >= 512): BLOCK_M=64, enough seq rows to fill all CUs
        self.rdna3_short_seq_flex_config = {
            (torch.bfloat16, 256): ROCmFlexConfig(16, 16, 1, 4, waves_per_eu=2),
            (torch.float16, 256): ROCmFlexConfig(16, 16, 1, 4, waves_per_eu=2),
        }
        self.rdna3_medium_seq_flex_config = {
            (torch.bfloat16, 256): ROCmFlexConfig(
                32, 16, 1, 4, matrix_instr_nonkdim=16, waves_per_eu=2
            ),
            (torch.float16, 256): ROCmFlexConfig(
                32, 16, 1, 4, matrix_instr_nonkdim=16, waves_per_eu=2
            ),
        }
        self.rdna3_default_flex_config = {
            (torch.bfloat16, 256): ROCmFlexConfig(64, 16, 1, 4, waves_per_eu=2),
            (torch.float16, 256): ROCmFlexConfig(64, 16, 1, 4, waves_per_eu=2),
        }

        self.flex_attn_fwd_autotune_configs: list[FlexConfig] = [
            ROCmFlexConfig(BLOCK1, BLOCK2, 1, w, kpack=default_kpack)
            for BLOCK1 in [16, 64, 128]
            for BLOCK2 in [16, 32, 64, 128]
            for w in [4, 8]
        ]

        self.flex_attn_bwd_autotune_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            ROCmFlexBwDConfig(
                BLOCK1, BLOCK2, BLOCK2, BLOCK1, 1, w, mfma, kpack=default_kpack
            )
            for BLOCK1 in [16, 32, 64]
            for BLOCK2 in [32, 64, 128]
            for w in ([4, 8] if BLOCK1 >= 128 or BLOCK2 >= 128 else [4])
            for mfma in [0, 16]
            if BLOCK2 % BLOCK1 == 0
        ]

        self.flex_decode_autotune_configs: list[FlexDecodeConfig] = [
            ROCmFlexDecodeConfig(32, 1, 4, kpack=default_kpack),
            ROCmFlexDecodeConfig(64, 1, 4, kpack=default_kpack),
            ROCmFlexDecodeConfig(128, 1, 4, kpack=default_kpack),
            ROCmFlexDecodeConfig(32, 1, 8, kpack=default_kpack),
            ROCmFlexDecodeConfig(64, 1, 8, kpack=default_kpack),
            ROCmFlexDecodeConfig(128, 1, 8, kpack=default_kpack),
        ]

        self.exhaustive_flex_attn_fwd_configs: list[FlexConfig] = [
            ROCmFlexConfig(BLOCK_M, BLOCK_N, num_stages, num_warps, mfma, wpeu, kpack)
            for BLOCK_M in [16, 32, 64, 128]
            for BLOCK_N in [32, 64, 128]
            for num_stages in [1, 2]
            for num_warps in [2, 4, 8]
            for mfma in [0, 16]
            for wpeu in [0, int(8 // num_warps)]
            for kpack in [1, 2]
        ]

        self.exhaustive_flex_attn_bwd_configs: list[FlexBwDConfig] = [
            # See Note: flex bwd configs
            ROCmFlexBwDConfig(
                BLOCK_M1,
                BLOCK_N1,
                BLOCK_M2,
                BLOCK_N2,
                num_stages,
                num_warps,
                mfma,
                wpeu,
                kpack,
            )
            for BLOCK_M1 in [16, 32, 64, 128]
            for BLOCK_N1 in [16, 32, 64, 128]
            for BLOCK_M2 in [16, 32, 64, 128]
            for BLOCK_N2 in [16, 32, 64, 128]
            for num_stages in [1, 2]
            for num_warps in [2, 4, 8]
            for mfma in [0, 16]
            for wpeu in [0, int(8 // num_warps)]
            for kpack in [1, 2]
            if BLOCK_N1 % BLOCK_M1 == 0
            and BLOCK_M2 % BLOCK_N2 == 0  # kernel static assertions
        ]

        self.exhaustive_flex_decode_configs: list[FlexDecodeConfig] = [
            ROCmFlexDecodeConfig(
                block_n, num_stages, num_warps, mfma, wpeu, kpack=kpack
            )
            for block_n in [16, 32, 64, 128]
            for num_stages in [1, 2]
            for num_warps in [2, 4, 8]
            for mfma in [0, 16]
            for wpeu in [0, int(8 // num_warps)]
            for kpack in [1, 2]
        ]

    def _prune_exhaustive_configs(
        self,
        configs: list[BaseConfig],
        dtype_size: int,
    ) -> list[BaseConfig]:
        # these cause AMD compile to crash
        pruned_configs = [
            c
            for c in configs
            if not (
                (
                    getattr(c, "matrix_instr_nonkdim", 0) == 2
                    and getattr(c, "kpack", 0) == 2
                )
                or (c.block_k <= 16 and getattr(c, "kpack", 0) == 2)
            )
        ]
        return pruned_configs

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        ROCm specific filtering
        """
        for c in configs:
            c.num_stages = self.default_num_stages
        return super()._filter_configs(configs)

    def _finalize_mm_configs(
        self,
        configs: list[BaseConfig],
    ) -> Generator[TritonConfig, None, None]:
        """
        Finalizes configs after scaling, applying additional constraints.
        """
        used: OrderedSet[tuple[int, ...]] = OrderedSet()

        max_mm_configs = config.test_configs.max_mm_configs

        for conf in configs:
            # Each warp computes a 16x16 tile = 256 elements
            conf.num_warps = min(conf.num_warps, conf.block_m * conf.block_n // 256)

            # Defaults for AMD triton backend kern args if not set
            matrix_instr_nonkdim: int = getattr(conf, "matrix_instr_nonkdim", 16)
            waves_per_eu: int = getattr(conf, "waves_per_eu", 0)
            # Use explicit kpack if set, otherwise determine optimal value based on
            # architecture and BLOCK_K
            kpack: int = getattr(conf, "kpack", get_default_kpack(conf.block_k))

            if matrix_instr_nonkdim != 0 and (
                conf.block_m % matrix_instr_nonkdim != 0
                or conf.block_n % matrix_instr_nonkdim != 0
            ):
                #  block_m and block_n must be a multiple of matrix_instr_nonkdim
                continue

            # Construct key for finding duplicate configs
            key: tuple[int, ...] = (
                conf.block_m,
                conf.block_n,
                conf.block_k,
                conf.num_stages,
                conf.num_warps,
                waves_per_eu,
                matrix_instr_nonkdim,
                kpack,
            )

            # Check if gemm specific arg exists - add to key if does
            group_m = getattr(conf, "group_m", None)
            # AMD GPU crashes if group_m = 0
            if group_m is not None and group_m <= 0:
                group_m = 8
            if group_m is not None:
                key += (group_m,)

            if waves_per_eu != 0:
                waves_per_eu = int(8 // conf.num_warps)

            if key not in used and (
                max_mm_configs is None or len(used) < max_mm_configs
            ):
                used.add(key)
                kwargs = {
                    "BLOCK_M": conf.block_m,
                    "BLOCK_N": conf.block_n,
                    "BLOCK_K": conf.block_k,
                    "num_stages": conf.num_stages,
                    "num_warps": conf.num_warps,
                    "matrix_instr_nonkdim": matrix_instr_nonkdim,
                    "waves_per_eu": waves_per_eu,
                    "kpack": kpack,
                }
                if group_m is not None:
                    kwargs["GROUP_M"] = group_m
                yield self.triton_config(**kwargs)

    def get_flex_attn_fwd_configs(
        self, head_dim: int, seq_len: sympy.Expr, dtype: Any
    ) -> list[FlexConfig]:
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        capability = torch.cuda.get_device_capability()
        default_kpack = get_default_kpack()

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = ROCmFlexConfig(64, 64, 1, 4, kpack=default_kpack)
            else:
                default_config = ROCmFlexConfig(128, 64, 1, 4, kpack=default_kpack)
            if using_rocm_rdna3():
                sizevars = V.graph.sizevars
                if sizevars.statically_known_lt(seq_len, 128):
                    default_config = self.rdna3_short_seq_flex_config.get(
                        (dtype, head_dim), default_config
                    )
                elif sizevars.statically_known_lt(seq_len, 512):
                    default_config = self.rdna3_medium_seq_flex_config.get(
                        (dtype, head_dim), default_config
                    )
                else:
                    default_config = self.rdna3_default_flex_config.get(
                        (dtype, head_dim), default_config
                    )
            elif capability >= (9, 5):  # gfx950 (MI350X/MI355X)
                default_config = self.gfx950_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
            elif capability >= (9, 4):  # gfx942 (MI300X/MI325X)
                default_config = self.gfx942_default_flex_config.get(
                    (dtype, head_dim), default_config
                )
        else:
            if dtype == torch.float32:
                default_config = ROCmFlexConfig(32, 16, 1, 4, kpack=default_kpack)
            else:
                default_config = ROCmFlexConfig(64, 32, 1, 4, kpack=default_kpack)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs

    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexBwDConfig]:
        flex_attn_bwd_configs: list[FlexBwDConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        default_kpack = get_default_kpack()
        if dtype == torch.float32:
            default_config = ROCmFlexBwDConfig(
                16, 16, 16, 16, 1, 4, kpack=default_kpack
            )
        elif head_dim <= 256:
            if head_dim == 64:
                default_config = ROCmFlexBwDConfig(
                    64, 64, 64, 64, 1, 4, kpack=default_kpack
                )
            elif head_dim == 128:
                default_config = ROCmFlexBwDConfig(
                    64, 128, 128, 64, 1, 4, kpack=default_kpack
                )
            else:
                default_config = ROCmFlexBwDConfig(
                    64, 64, 64, 64, 1, 4, kpack=default_kpack
                )
        else:
            default_config = ROCmFlexBwDConfig(
                16, 16, 16, 16, 1, 4, kpack=default_kpack
            )

        if default_config not in flex_attn_bwd_configs:
            flex_attn_bwd_configs.append(default_config)

        return flex_attn_bwd_configs

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        default_kpack = get_default_kpack()
        default_config = ROCmFlexDecodeConfig(64, 1, 4, kpack=default_kpack)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs


class XPUConfigHeuristic(BaseConfigHeuristic):
    """
    Placeholder child class for Intel GPU specific overrides.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = self.mm_configs + [
            GemmConfig(32, 64, 128, 2, 2),
            GemmConfig(64, 64, 32, 2, 8),
        ]
        self.xpu_default_flex_config = {
            (torch.float32, 64): FlexConfig(128, 32, 1, 16),
            (torch.float32, 128): FlexConfig(128, 32, 1, 16),
            (torch.float32, 256): FlexConfig(64, 16, 1, 8),
            (torch.bfloat16, 64): FlexConfig(128, 64, 1, 16),
            (torch.bfloat16, 128): FlexConfig(128, 64, 1, 16),
            (torch.bfloat16, 256): FlexConfig(32, 64, 1, 4),
            (torch.float16, 64): FlexConfig(128, 64, 1, 16),
            (torch.float16, 128): FlexConfig(128, 64, 1, 16),
            (torch.float16, 256): FlexConfig(32, 64, 1, 4),
        }
        self.flex_attn_fwd_autotune_configs: list[FlexConfig] = [
            FlexConfig(32, 16, 2, 4),
            FlexConfig(128, 64, 2, 16),
            FlexConfig(128, 64, 2, 8),
            FlexConfig(128, 32, 2, 16),
            FlexConfig(128, 32, 2, 8),
        ]
        self.flex_attn_bwd_autotune_configs: list[FlexBwDConfig] = [
            FlexBwDConfig(32, 32, 32, 32, 2, 4),
            FlexBwDConfig(64, 64, 64, 64, 2, 4),
        ]
        self.flex_decode_autotune_configs: list[FlexDecodeConfig] = []

        if not bool(os.getenv("CI")):
            self.flex_attn_bwd_autotune_configs += [
                # See Note: flex bwd configs
                FlexBwDConfig(BLOCK1, BLOCK2, BLOCK2, BLOCK1, s, w)
                for BLOCK1 in [32, 64]
                for BLOCK2 in [32, 64, 128]
                for s in [1, 3, 4, 5]  # num_stages
                for w in ([4, 8] if BLOCK1 >= 128 or BLOCK2 >= 128 else [4])
                if BLOCK2 % BLOCK1 == 0
            ]
            self.flex_decode_autotune_configs += [
                FlexDecodeConfig(32, 1, 2),
                FlexDecodeConfig(32, 1, 1),
                FlexDecodeConfig(32, 2, 2),
                FlexDecodeConfig(32, 2, 1),
                FlexDecodeConfig(64, 1, 2),
                FlexDecodeConfig(64, 1, 1),
                FlexDecodeConfig(64, 2, 2),
                FlexDecodeConfig(64, 2, 1),
            ]

    def get_flex_attn_fwd_configs(
        self, head_dim: int, seq_len: sympy.Expr, dtype: Any
    ) -> list[FlexConfig]:
        flex_attn_fwd_configs: list[FlexConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            flex_attn_fwd_configs += self.flex_attn_fwd_autotune_configs

        if head_dim <= 256:
            if dtype == torch.float32:
                default_config = FlexConfig(64, 64, 1, 8)
            else:
                default_config = FlexConfig(128, 64, 1, 16)
            default_config = self.xpu_default_flex_config.get(
                (dtype, head_dim), default_config
            )
        else:
            if dtype == torch.float32:
                default_config = FlexConfig(32, 16, 1, 4)
            else:
                default_config = FlexConfig(64, 32, 1, 8)

        if default_config not in flex_attn_fwd_configs:
            flex_attn_fwd_configs.append(default_config)

        return flex_attn_fwd_configs

    def get_flex_attn_bwd_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexBwDConfig]:
        flex_attn_bwd_configs: list[FlexBwDConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            flex_attn_bwd_configs += self.flex_attn_bwd_autotune_configs

        if dtype == torch.float32:
            default_config = FlexBwDConfig(16, 16, 16, 16, 1, 4)
        elif head_dim <= 256:
            if head_dim == 64:
                default_config = FlexBwDConfig(64, 64, 64, 64, 1, 8)
            elif head_dim == 128:
                default_config = FlexBwDConfig(64, 64, 64, 64, 1, 8)
            else:
                default_config = FlexBwDConfig(64, 64, 64, 64, 1, 8)
        else:  # modest hardware or extremely large head_dim
            default_config = FlexBwDConfig(16, 16, 16, 16, 1, 4)

        if default_config not in flex_attn_bwd_configs:
            flex_attn_bwd_configs.append(default_config)

        return flex_attn_bwd_configs

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        default_config = FlexDecodeConfig(64, 1, 2)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs

    def _prune_exhaustive_configs(
        self,
        configs: list[BaseConfig],
        dtype_size: int,
    ) -> list[BaseConfig]:
        return configs


class MTIAConfigHeuristic(BaseConfigHeuristic):
    """
    Placeholder child class for MTIA specific overrides.
    """


# Template-specific mixin classes
class MMTemplateConfigMixin(GemmMaxAutotuneTemplateConfigHeuristics):
    """
    Mixin class that converts config lists to template kwargs.
    This handles the logic that was previously in choices.get_mm_configs.

    This mixin expects to be used with BaseConfigHeuristic or its subclasses.
    """

    # Type annotations to ensure the mixin works with BaseConfigHeuristic
    get_mm_configs: Callable[[], partial[Generator[TritonConfig, None, None]]]
    get_exhaustive_mm_configs: Callable[
        [], partial[Generator[TritonConfig, None, None]]
    ]
    _filter_configs: Callable[[list[BaseConfig]], list[BaseConfig]]
    # Provided by ROCmConfigHeuristic / BaseConfigHeuristic when this mixin is
    # combined with a backend heuristic. Annotated here so type-checkers see the
    # attributes used by the origami branch in _get_template_configs_impl.
    default_num_stages: int
    exhaustive_configs: list[BaseConfig]
    _get_exceeding_shared_memory_checker: Callable[
        [bool, int], Callable[[BaseConfig, int], bool] | None
    ]
    preprocess_mm_configs: Callable[..., Generator[TritonConfig, None, None]]

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"Expected MMKernelInputs, got {type(kernel_inputs)}")
        m, n, k = kernel_inputs.mnk_symbolic()
        device_type = kernel_inputs.device_type
        if device_type == "xpu":
            # XPU eager matmul takes TF32 from the oneDNN flag, not the CUDA one.
            allow_tf32 = torch.backends.mkldnn.allow_tf32
        elif device_type == "cuda":
            # allow_tf32 alignment heuristics based on reverse engineering
            # H100 CUDA 12.8 behavior
            size_threshold = V.graph.sizevars.statically_known_true(
                sympy.And(sympy.Ge(m, 16), sympy.Ge(Min(n, k), 512))
            )
            allow_tf32 = (
                torch.backends.cuda.matmul.fp32_precision == "tf32" and size_threshold
            )
        else:
            allow_tf32 = False

        return {
            "ALLOW_TF32": allow_tf32,
        }

    def _valid(self, kernel_inputs: KernelInputs) -> bool:
        return True

    def _get_config_generator(
        self,
    ) -> partial[Generator[TritonConfig, None, None]]:
        """
        Get the appropriate config generator based on search space.
        Can be overridden by subclasses for template-specific behavior.
        """
        # Handle exhaustive search case
        if config.max_autotune_gemm_search_space == "EXHAUSTIVE":
            return self.get_exhaustive_mm_configs()
        else:
            return self.get_mm_configs()

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Convert config lists to template kwargs.
        This replaces the logic from choices.get_mm_configs and inlines mm_options.
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")
        input_nodes = kernel_inputs.nodes()
        if len(input_nodes) < 2:
            raise ValueError(f"Need at least 2 input tensors, got {len(input_nodes)}")
        if not self._valid(kernel_inputs):
            return

        # Extract M, N, K from kernel_inputs
        m, n, k = kernel_inputs.mnk_symbolic()

        # Extract dtype and device_type from kernel_inputs
        dtype = kernel_inputs.dtype()
        # Get the appropriate config generator
        configs = self._get_config_generator()
        # origami is a C++ perf model that requires concrete m, n, k; feeding it
        # sympy symbols (dynamic shapes, e.g. torch.cat with k=s0+s1) yields no
        # usable selection, so restrict origami to fully static problems and let
        # symbolic shapes fall through to the regular config generator below.
        mnk_static = all(not getattr(x, "free_symbols", None) for x in (m, n, k))
        if (
            origami is not None
            and not mnk_static
            and config.max_autotune_gemm_search_space == "DEFAULT"
        ):
            log.debug("Origami skipped: symbolic m/n/k, using regular config generator")
        # `origami is not None` encodes the module-load gate (see top of file);
        # only DEFAULT search space is supported here.
        if (
            origami is not None
            and config.max_autotune_gemm_search_space == "DEFAULT"
            and mnk_static
        ):
            # Extract device and strides for origami GEMM
            device = kernel_inputs.device()
            strides = kernel_inputs.strides_symbolic()
            a_stride = strides[kernel_inputs._mat1_idx]
            b_stride = strides[kernel_inputs._mat2_idx]
            # Pre-filter exhaustive_configs against LDS BEFORE origami sees them.
            # Origami otherwise ranks LDS-busting tiles in its top-K and we'd
            # have to drop its best picks at compile time (Triton OutOfResources
            # is not always caught on the HIP backend). Check against
            # self.default_num_stages because ROCmConfigHeuristic._filter_configs
            # (triton.py:1728) clobbers num_stages to that value downstream.
            lds_check = self._get_exceeding_shared_memory_checker(False, 0)
            exhaustive = self.exhaustive_configs
            if lds_check is not None:
                exhaustive = [
                    c
                    for c in self.exhaustive_configs
                    if not lds_check(
                        dataclasses.replace(c, num_stages=self.default_num_stages),
                        dtype.itemsize,
                    )
                ]
                pruned = len(self.exhaustive_configs) - len(exhaustive)
                if pruned:
                    log.info(
                        "Origami: pruned %d/%d exhaustive configs "
                        "exceeding LDS for dtype %s",
                        pruned,
                        len(self.exhaustive_configs),
                        dtype,
                    )
            allcfgs = self.preprocess_mm_configs(
                m,
                n,
                k,
                configs=exhaustive,
                dtype_size=dtype.itemsize,
                op_name=op_name,
            )
            selector = origami.OrigamiMatmulSelector(
                allcfgs,
                m,
                n,
                k,
                dtype,
                dtype,
                dtype,
                device,
                a_stride,
                b_stride,
            )
            # _origami_{problem,hardware,configs} wrap the private attrs;
            # see TODO at top of file for the upstream-API migration plan.
            topk_results = origami.select_topk_configs(
                _origami_problem(selector),
                _origami_hardware(selector),
                _origami_configs(selector),
                config.rocm.origami_topk,
            )
            seen = OrderedSet()
            # Collect origami-selected configs into a list before filtering
            origami_configs: list[BaseConfig] = []
            # Map config key (immutable tile dimensions) to origami parameters
            origami_triton_configs: dict[tuple[int, int, int], tuple[Any, Any]] = {}

            def _config_key(cfg: BaseConfig) -> tuple[int, int, int]:
                """
                Generate a stable key for origami configs based on immutable tile dimensions.
                Uses only block_m, block_n, block_k since these uniquely identify an origami
                config and remain invariant across _filter_configs modifications (which may
                change num_stages, num_warps, and other attributes).
                """
                return (
                    cfg.block_m,
                    cfg.block_n,
                    cfg.block_k,
                )

            for result in topk_results:
                cfg = result.config
                key = (cfg.mt.m, cfg.mt.n, cfg.mt.k, cfg.occupancy)
                if key in seen:
                    continue
                seen.add(key)
                grid = origami.select_grid_size(
                    _origami_problem(selector),
                    _origami_hardware(selector),
                    cfg,
                    origami.grid_selection_t.data_parallel,
                    _origami_hardware(selector).N_CU,
                )
                wgm_result = origami.select_workgroup_mapping(
                    _origami_problem(selector),
                    _origami_hardware(selector),
                    cfg,
                    grid,
                )
                # One MFMA per warp, capped at 2 * parallel_mi_cu.
                tile_area = cfg.mt.m * cfg.mt.n
                try:
                    warp_size = DeviceProperties.create(device).warp_size or 64
                except (RuntimeError, AttributeError) as e:
                    # Fallback to standard warp size if device properties unavailable
                    log.warning(
                        "Failed to get device warp size: %s. Using default 64.", e
                    )
                    warp_size = 64
                max_warps = 2 * _origami_hardware(selector).parallel_mi_cu
                # mfma_dim from origami hardware object; fall back to 16.
                mfma_dim = getattr(_origami_hardware(selector), "mfma_m", 16)
                num_warps = min(
                    max_warps,
                    max(1, tile_area // (mfma_dim * warp_size)),
                )
                # num_stages: ROCmConfigHeuristic._filter_configs (triton.py:1728)
                # overwrites this to self.default_num_stages, so seed with that
                # value to keep the in-flight GemmConfig consistent with the
                # post-filter state.
                base_config = GemmConfig(
                    block_m=cfg.mt.m,
                    block_n=cfg.mt.n,
                    block_k=cfg.mt.k,
                    num_stages=self.default_num_stages,
                    num_warps=num_warps,
                    group_m=wgm_result.wgm,
                )
                origami_configs.append(base_config)
                # Store triton config and its parameters for later conversion using stable key
                config_key = _config_key(base_config)
                origami_triton_configs[config_key] = (
                    cfg.occupancy,
                    wgm_result.wgm,
                )

            # Apply backend filters (max block size, memory constraints, etc.).
            # LDS-overflow prune already happened upstream against
            # self.exhaustive_configs before origami selection.
            filtered_configs = self._filter_configs(origami_configs)

            # If origami returned configs, use them; otherwise fall back to regular generator
            if filtered_configs:
                # Convert filtered configs to template kwargs and yield
                for filtered_config in filtered_configs:
                    # Retrieve stored origami parameters using stable config key
                    config_key = _config_key(filtered_config)
                    origami_params = origami_triton_configs.get(config_key)
                    if origami_params is None:
                        # If not found, log a warning about the lookup failure
                        log.warning(
                            "Config lookup failed for %s. "
                            "This may indicate a config rebuilding issue in _filter_configs.",
                            config_key,
                        )
                        continue
                    occupancy, group_m = origami_params

                    # Create TritonConfig from filtered config
                    triton_config = self.triton_config(  # pyright: ignore[attr-defined]
                        num_stages=filtered_config.num_stages,
                        num_warps=filtered_config.num_warps,
                        BLOCK_M=filtered_config.block_m,
                        BLOCK_N=filtered_config.block_n,
                        BLOCK_K=filtered_config.block_k,
                        GROUP_M=group_m,
                        waves_per_eu=occupancy,
                    )
                    template_kwargs = self._convert_config_to_template_kwargs(
                        triton_config,
                        m,
                        n,
                        k,
                        kernel_inputs.out_dtype(),
                    )
                    yield template_kwargs
            else:
                # No origami configs returned (e.g., topk=0), fall back to regular generator
                log.warning(
                    "Origami returned no configs, falling back to regular config generator"
                )
                for c in configs(
                    m,
                    n,
                    k,
                    dtype_size=dtype.itemsize,
                    op_name=op_name,
                    **kwargs,
                ):
                    template_kwargs = self._convert_config_to_template_kwargs(
                        c,
                        m,
                        n,
                        k,
                        kernel_inputs.out_dtype(),
                    )
                    yield template_kwargs

        else:
            # Warn if origami was enabled but not used due to EXHAUSTIVE search space.
            # `origami is not None` already covers IS_ROCM + max_autotune + rocm.origami.
            if (
                origami is not None
                and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
            ):
                log.warning(
                    "Origami is enabled but not used: search space is set to EXHAUSTIVE. "
                    "Origami only operates with DEFAULT search space. "
                    "Set max_autotune_gemm_search_space='DEFAULT' to enable origami optimization."
                )
            for c in configs(
                m,
                n,
                k,
                dtype_size=dtype.itemsize,
                op_name=op_name,
                **kwargs,
            ):
                template_kwargs = self._convert_config_to_template_kwargs(
                    c,
                    m,
                    n,
                    k,
                    kernel_inputs.out_dtype(),
                )
                yield template_kwargs

    def _convert_config_to_template_kwargs(
        self,
        triton_config: TritonConfig,
        m: sympy.Integer | sympy.Symbol,
        n: sympy.Integer | sympy.Symbol,
        k: sympy.Integer | sympy.Symbol,
        out_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """
        Convert triton config to template kwargs.
        Moved from mm_common.mm_options.
        """
        # Calculate EVEN_K symbolic. (It isn't worth guarding on this)
        even_k_symbolic = sympy.Eq(Mod(k, triton_config.kwargs["BLOCK_K"]), 0)
        even_k_symbolic = V.graph.sizevars.statically_known_true(even_k_symbolic)

        # Build options dict

        options_dict = dict(
            EVEN_K=even_k_symbolic,
            USE_FAST_ACCUM=False,  # Option for _scaled_mm
            ACC_TYPE=self._get_acc_type(out_dtype),
            num_stages=triton_config.num_stages,
            num_warps=triton_config.num_warps,
            **triton_config.kwargs,
        )

        # If GROUP_M not specified then default to 8
        if "GROUP_M" not in triton_config.kwargs:
            group_m = triton_config.kwargs.get("GROUP_M", 8)
            options_dict["GROUP_M"] = group_m

        return options_dict

    @staticmethod
    def _dtype_to_triton(dtype: torch.dtype) -> str:
        """Convert a torch dtype to a triton type string."""
        return triton_type(dtype)

    def _get_acc_type(self, dtype: torch.dtype) -> str:
        """
        Get accumulator type for the given dtype.
        Moved from mm_common.acc_type.
        """
        if dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        ):
            return "tl.float32"
        return self._dtype_to_triton(dtype)


# INT8 specific mixin to filter correctly
class INT8MMTemplateConfigMixin(MMTemplateConfigMixin):
    """
    Ensure that we feed in has_int8_tensor=True
    """

    def __init__(self) -> None:
        super().__init__()
        self.has_int8_tensor = True


# MMPlusMM specific mixin to avoid running _scale_mm_configs
class MMPlusMMTemplateConfigMixin(MMTemplateConfigMixin):
    """
    Ensure that _should_scale_configs is False
    """

    # TODO(coconutruben): remove this once all tests work
    # with proper scaling on mm_plus_mm
    def __init__(self) -> None:
        super().__init__()
        self.should_scale_configs = False

    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = False,
        scale: float = 1.0,
        exclude: Callable[
            [sympy.Integer, sympy.Integer, sympy.Integer], bool
        ] = lambda m, n, k: False,
        dtype_size: int = 0,
        op_name: str = "mm",
        **kwargs,
    ) -> Generator[TritonConfig, None, None]:
        configs = [
            c for c in configs if V.graph.sizevars.statically_known_lt(c.block_k, k)
        ]
        return super().preprocess_mm_configs(
            m,
            n,
            k,
            configs=configs,
            has_int8_tensor=has_int8_tensor,
            scale=scale,
            exclude=exclude,
            dtype_size=dtype_size,
            op_name=op_name,
            **kwargs,
        )

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError("Expect MMKernelInputs")
        m, n, k = kernel_inputs.mnk_symbolic()
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs, op_name, **kwargs
        ):
            # Apply BLOCK_K constraint specific to mm_plus_mm
            # see https://github.com/triton-lang/triton/issues/1298
            # BLOCK_K = K causes llvm error
            if V.graph.sizevars.statically_known_lt(
                template_kwargs.get("BLOCK_K", k), k
            ):
                yield template_kwargs


class TMAWorkspaceMixin(MMTemplateConfigMixin):
    """
    Small mixin to ensure that the workspace arg is correct for TMA
    and TMA specific filtering can happen.
    """

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        kwargs["workspace_arg"] = get_tma_workspace_arg(
            num_tma_descriptors=2,
            device=kernel_inputs.device(),
        )
        return kwargs

    # pyrefly: ignore [bad-override]
    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        TMA specific filtering, as num_warps=2 not safe for TMA
        """
        configs = [c for c in configs if c.num_warps != 2]
        return super()._filter_configs(configs)


def get_shared_memory_checker_opts(op_name: str, dtype_size: int):
    return {
        "has_sm_layout_conversion": True,
        # addmm requires the acc dtype for layout conversion due to adding bias
        # mm just input dtype
        "layout_conversion_byte_size": 4 if op_name == "addmm" else dtype_size,
    }


# TMA-specific mixin for TMA templates
class TMATemplateConfigMixin(TMAWorkspaceMixin, MMTemplateConfigMixin):
    """
    TMA-specific mixin that uses persistent configs and adds TMA options.
    This inherits from MMTemplateConfigMixin and overrides config generation.
    """

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate TMA template configs by calling super and adding TMA-specific options.
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError("TMATemplateConfigMixin requires MMKernelInputs")
        mat1, mat2 = kernel_inputs.mat1mat2()
        tma_opts = {
            "A_ROW_MAJOR": not mat1.layout.is_transposed(),
            "B_ROW_MAJOR": not mat2.layout.is_transposed(),
            "NUM_SMS": get_num_sms(),
            "TMA_SIZE": TMA_DESCRIPTOR_SIZE,
            "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),
            "tma_store": config.triton.enable_template_tma_store,
            "tma_load_for_template_epilogue": config.triton.enable_tma_load_for_template_epilogue,
            "transpose_discontiguous_tensor_descriptors_override": True,
        }

        # Get base template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
            **get_shared_memory_checker_opts(
                op_name, dtype_size=kernel_inputs.dtype().itemsize
            ),
        ):
            yield {**template_kwargs, **tma_opts}


# TMA mixins for Blackwell templates
class BlackwellTMATemplateConfigMixin(TMATemplateConfigMixin):
    """Config mixin for the Blackwell persistent-TMA GEMM template.

    Generates the fb-triton autoWS (meta-WS) config sweep and prunes the
    combinations the lowering cannot support before they reach codegen.
    """

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate TMA template configs by calling super and adding TMA-specific options.
        """
        # Get base template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
            **kwargs,
        ):
            use_meta_ws = template_kwargs.get("USE_META_WS", False)
            # autoWS configs come from a full sweep; drop combos the lowering
            # does not support so no invalid config reaches codegen.
            if use_meta_ws and not self._autows_constraints_ok(template_kwargs):
                continue
            # Some Triton versions requires num_warps >= 4 for WS
            # to avoid compilation issues. Triton disables WS if num_warps < 4
            # or num_stages < 2. Similar issues have been seen with num_stages=1
            constraints_violated = (
                template_kwargs["num_warps"] < 4 or template_kwargs["num_stages"] < 2
            )
            ws = (
                template_kwargs.get("WARP_SPECIALIZE", True)
                and not constraints_violated
            )
            # Meta WS pass has only validated flatten=False; OSS Triton uses flatten=True
            flatten = (
                template_kwargs.get("FLATTEN", True)
                and not constraints_violated
                and not use_meta_ws
            )
            yield {
                **template_kwargs,
                "NUM_SMS": get_num_sms(),
                "WARP_SPECIALIZE": ws,
                "FLATTEN": flatten,
            }

    @staticmethod
    def _autows_constraints_ok(template_kwargs: dict[str, Any]) -> bool:
        """autoWS lowering constraints; swept configs violating these are pruned."""
        block_m = template_kwargs["BLOCK_M"]
        block_n = template_kwargs["BLOCK_N"]
        subtile = template_kwargs.get("EPILOGUE_SUBTILE", 1)
        # each epilogue subtile is BLOCK_N // EPILOGUE_SUBTILE wide
        if block_n // subtile < 32:
            return False
        # dp=2 splits the row tile into two MMA partitions; BLOCK_M=64 fails in
        # the fb-triton WS pass pipeline, so keep the tile at 128 or 256
        dp = template_kwargs.get("DATA_PARTITION_FACTOR", 1)
        if dp == 2 and block_m not in (128, 256):
            return False
        return True

    def _get_config_generator(
        self,
    ) -> partial[Generator[TritonConfig, None, None]]:
        # No curated autoWS set yet: sweep the full autoWS space for both default
        # and exhaustive search, and let _get_template_configs_impl prune it.
        if _use_template_autows():
            return partial(
                self.preprocess_mm_configs, configs=self._generate_autows_configs()
            )
        return super()._get_config_generator()

    @staticmethod
    def _generate_autows_configs() -> list[BaseConfig]:
        configs: list[BaseConfig] = []
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
            [32, 64, 128, 256], repeat=3
        ):
            for num_stages in [2, 3, 4, 5, 6]:
                # AutoWS doesn't work with num_warps < 4
                for num_warps in [4, 8]:
                    for epilogue_subtile in [1, 2, 4, 8]:
                        for data_partition_factor in [1, 2]:
                            for separate_epilogue_store in [False, True]:
                                configs.append(
                                    BlackwellGPUGemmConfig(
                                        block_m=BLOCK_M,
                                        block_n=BLOCK_N,
                                        block_k=BLOCK_K,
                                        num_stages=num_stages,
                                        num_warps=num_warps,
                                        group_m=8,
                                        epilogue_subtile=epilogue_subtile,
                                        use_meta_ws=True,
                                        data_partition_factor=data_partition_factor,
                                        separate_epilogue_store=separate_epilogue_store,
                                        warp_specialize=True,
                                        flatten=False,
                                    )
                                )
        return configs

    @staticmethod
    def _generate_exhaustive_configs() -> list[BaseConfig]:
        configs: list[BaseConfig] = []
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
            [32, 64, 128, 256],
            repeat=3,
        ):
            for num_stages in [2, 3, 4, 5, 6]:
                # AutoWS doesn't work with num_warps < 4
                for num_warps in [4, 8]:
                    for EPILOGUE_SUBTILE in [1, 2, 4]:
                        configs.append(
                            BlackwellGPUGemmConfig(
                                block_m=BLOCK_M,
                                block_n=BLOCK_N,
                                block_k=BLOCK_K,
                                num_stages=num_stages,
                                num_warps=num_warps,
                                group_m=8,
                                epilogue_subtile=EPILOGUE_SUBTILE,
                                warp_specialize=True,
                                flatten=True,
                            )
                        )
        return configs


# Scaled MM-specific mixin for scaled MM templates
class BaseScaledMMConfigMixin(MMTemplateConfigMixin):
    """
    This is a base that handles the common case for ScaledMM

    The TMA and non-TMA should build on top of this
    """

    def adjust_kernel_inputs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> KernelInputs:
        """
        for scaled_mm, we need to unsqueeze scale tensors, and bias
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError("Expect MMKernelInputs for scaled MM")
        inputs = super().adjust_kernel_inputs(kernel_inputs, op_name)
        nodes = inputs.nodes()
        mat_a, mat_b, scale_a, scale_b, *bias = nodes
        bias = bias[0] if bias else None
        # Prepare triton input nodes and create kernel_inputs at the top
        from ...lowering import lowerings as L

        aten = torch.ops.aten
        if bias and len(mat_b.get_size()) == len(bias.get_size()) + 1:
            # Need to unsqueeze bias from [N] -> [1, N]
            bias = L[aten.unsqueeze](bias, 0)

        def is_tensorwise_scale(scale: Any) -> bool:
            size = scale.get_size()
            return len(size) == 0 or all(
                V.graph.sizevars.statically_known_equals(dim, 1) for dim in size
            )

        def normalize_tensorwise_scale(scale: Any, *, allow_high_rank: bool) -> Any:
            if not is_tensorwise_scale(scale):
                return scale
            if not allow_high_rank and len(scale.get_size()) > 2:
                raise RuntimeError(
                    f"t() expects a tensor with <= 2 dimensions, "
                    f"but self is {len(scale.get_size())}D"
                )
            return L[aten.reshape](scale, [1, 1])

        # The templates load scale suffixes with 2D output indices. Eager accepts
        # high-rank all-ones scale_a, but scale_b is internally transposed and
        # rank > 2 is invalid.
        scale_a = normalize_tensorwise_scale(scale_a, allow_high_rank=True)
        scale_b = normalize_tensorwise_scale(scale_b, allow_high_rank=False)
        nodes = [mat_a, mat_b, scale_a, scale_b]
        if bias:
            nodes.append(bias)
        return MMKernelInputs(
            nodes,
            scalars=kernel_inputs._scalars,
            mat1_idx=kernel_inputs._mat1_idx,
            mat2_idx=kernel_inputs._mat2_idx,
            out_dtype=kernel_inputs._out_dtype,
        )

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate scaled MM template configs with scaled MM-specific options.
        Handles the remaining logic from mm_common, including assertions.
        """
        kernel_inputs = self.adjust_kernel_inputs(kernel_inputs, op_name)
        input_nodes = kernel_inputs.nodes()
        # Initial assertion from mm_common.scaled_mm_options
        if len(input_nodes) < 4:
            raise AssertionError(
                f"scaled_mm requires at least 4 inputs, got {len(input_nodes)}"
            )

        # Extract scale tensors (typically scale_a and scale_b are input_nodes[2] and input_nodes[3])
        scale_a = input_nodes[2]
        scale_b = input_nodes[3]

        # Scale compatibility assertion from mm_common.scaled_mm_options
        def are_compatible_scales(size_a: Any, size_b: Any) -> bool:
            # Same sized scales are compatible
            if len(size_a) == len(size_b):
                return True

            # Both need to be scalars or len(1) tensors
            if len(size_a) <= 1 and len(size_b) <= 1:
                return True

            return False

        size_a, size_b = scale_a.get_size(), scale_b.get_size()
        if not are_compatible_scales(size_a, size_b):
            raise AssertionError(
                "Expect scale_a and scale_b to be either both scalars (including single-element tensors) "
                f"or 1-dimensional tensors with the same size. Got scale_a: {len(size_a)} and scale_b: {len(size_b)}."
            )

        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")

        if not self._valid(kernel_inputs):
            return

        # Get base template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs, op_name, **kwargs
        ):
            # Add scaled MM-specific options (moved from mm_common.scaled_mm_options)
            # Override accumulator type for scaled MM using dynamic type resolution
            template_kwargs["ACC_TYPE"] = self._get_acc_type(kernel_inputs.out_dtype())

            yield template_kwargs


class ScaledMMConfigMixin(BaseScaledMMConfigMixin):
    """Mixin for scaled mm with the regular mm template"""

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        from ...kernel.mm_common import scale_mm_epilogue

        return {
            **kwargs,
            "suffix_args": kernel_inputs.count - 2,
            "epilogue_fn": scale_mm_epilogue(),
            "epilogue_fn_hash": "scale_mm_epilogue",
        }

    def _valid(self, kernel_inputs: KernelInputs) -> bool:
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError("Expect MMKernelInputs for ScaledMMConfigMixin")
        _, _, k = kernel_inputs.mnk_symbolic()
        if V.graph.sizevars.guard_or_false(sympy.Le(k, 16)):
            # Triton crashes however uncommon for real workloads
            return False

        # On NVIDIA B200 GPUs, K dim must be >= 32 for tcgen05.mma.kind::f8f6f4.* PTX instruction to be valid
        # source: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape
        if using_b200() and V.graph.sizevars.guard_or_false(sympy.Lt(k, 32)):
            return False
        return True

    # pyrefly: ignore [bad-override]
    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        Filter out bad configs for specific hardware.
        On AMD MI350X (GFX 9.5+), skip configs with BLOCK_K<=64 due to lack of corresponding MFMA instructions.
        """

        def should_skip_mi350x_config(config: BaseConfig) -> bool:
            """Skip config if BLOCK_K<=64 on MI350X (GFX 9.5+)"""
            try:
                return (
                    config.block_k <= 64
                    and IS_ROCM
                    and torch.cuda.get_device_capability() >= (9, 5)
                )
            except RuntimeError:
                # If no HIP GPUs are available, we can't check device capability
                # so we don't skip any configs
                return False

        filtered_configs = [c for c in configs if not should_skip_mi350x_config(c)]
        return super()._filter_configs(filtered_configs)


# Scaled TMA-specific mixin for scaled MM templates with TMA
class ScaledTMAConfigMixin(TMAWorkspaceMixin, BaseScaledMMConfigMixin):
    """
    Scaled TMA-specific mixin that extends BaseScaledMMConfigMixin with TMA functionality.
    This is for scaled MM templates that use device TMA.
    This inherits from BaseScaledMMConfigMixin and adds TMA-specific options.
    """

    # pyrefly: ignore [bad-override]
    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        TMA specific filtering:
        - num_warps=2 not safe for TMA
        - block_k >= 32 required for TMA (requires inner-most dimension >= 32)
        """
        configs = [c for c in configs if c.num_warps != 2 and c.block_k >= 32]
        return super()._filter_configs(configs)

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate scaled TMA template configs with both scaled MM and TMA-specific options.
        """
        # Get base scaled MM template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
            **kwargs,
        ):
            # Add TMA-specific options for device TMA scaled MM
            template_kwargs["TMA_SIZE"] = TMA_DESCRIPTOR_SIZE
            template_kwargs["NUM_SMS"] = get_num_sms()
            template_kwargs["TMA_EXPERIMENTAL_API"] = not has_triton_stable_tma_api()

            yield template_kwargs


# Scaled Blackwell TMA-specific mixin for scaled MM templates with TMA
class ScaledBlackwellTMAConfigMixin(
    BlackwellTMATemplateConfigMixin, ScaledMMConfigMixin
):
    """
    Scaled Blackwell TMA-specific mixin that extends ScaledMMConfigMixin with TMA functionality.
    This is for scaled MM templates that use device TMA on Blackwell.
    This inherits from ScaledMMConfigMixin, which inherits the scale_mm_epilogue, and adds TMA-specific options.
    """

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        """
        Warp specialization-specific filtering (BlackwellTMATemplateConfigMixin)
        (compilation issues occur in some versions of Triton)
        - num_warps < 4 unsafe for warpspec
        - num_stages < 2 unsafe for warpspec

        TMA-specific filtering:
        - block_k >= 32 required for TMA (requires inner-most dimension >= 32)
        """
        configs = [c for c in configs if c.block_k >= 32]
        return super()._filter_configs(configs)


# Template-specific heuristic classes using multiple inheritance


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
@register_template_heuristic(
    bmm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDAMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for CUDA"""


@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None, op_name="addmm"
)
@register_template_heuristic(
    bmm_template.uid, "cuda", register=torch.version.hip is None, op_name="baddbmm"
)
class CUDAAddMMTemplateConfigHeuristic(AddMMConfigMixin, CUDAMMTemplateConfigHeuristic):
    """Addmm specific mixin for CUDA"""


# TODO(coconutruben): deprecate once autoheuristic is deprecated
@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="mm-ah",
)
class CUDAMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for CUDA using the extra mm configs only (for autoheuristic)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.extra_mm_configs
        self.exhaustive_configs = self.extra_mm_configs


@register_template_heuristic(
    persistent_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDAPersistentTMATemplateConfigHeuristic(
    TMATemplateConfigMixin, CUDAConfigHeuristic
):
    """Persistent TMA template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use persistent_mm_configs
        self.mm_configs = self.persistent_mm_configs


@register_template_heuristic(
    persistent_mm_template.uid,
    "cuda",
    register=IS_ROCM,
)
class PersistentMMTemplateConfigHeuristic(
    MMTemplateConfigMixin,
    ROCmConfigHeuristic,  # type: ignore[misc]
):
    """Persistent MM template heuristic (no TMA, standard pointer loads)"""

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = self.persistent_mm_configs

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs, op_name, **kwargs
        ):
            yield {**template_kwargs, "NUM_SMS": get_num_sms()}


@register_template_heuristic(
    blackwell_ws_persistent_device_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDABlackwellPersistentTMATemplateConfigHeuristic(
    BlackwellTMATemplateConfigMixin, CUDAConfigHeuristic
):
    """Blackwell Persistent TMA template"""

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = self.blackwell_persistent_mm_configs
        self.exhaustive_configs = self._generate_exhaustive_configs()


@register_template_heuristic(
    persistent_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="addmm",
)
class CUDAAddmmPersistentTMATemplateConfigHeuristic(
    AddMMConfigMixin, CUDAPersistentTMATemplateConfigHeuristic
):
    """Addmm specific mixin for CUDA"""


@register_template_heuristic(
    blackwell_ws_persistent_device_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="addmm",
)
class CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic(
    AddMMConfigMixin, CUDABlackwellPersistentTMATemplateConfigHeuristic
):
    """Addmm extension for DataCenter Blackwell Templates"""

    def __init__(self) -> None:
        super().__init__()
        # NOTE: to ensure that we pass tests, addmm needs a small config
        self.mm_configs = (
            self.blackwell_persistent_mm_configs
            + self.blackwell_persistent_addmm_configs
        )
        self.exhaustive_configs = self._generate_exhaustive_configs()


@register_template_heuristic(
    mm_template.uid, "cuda", register=torch.version.hip is None, op_name="scaled_mm"
)
class CUDAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CUDAConfigHeuristic):
    """Scaled MM template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        configs = [c for c in configs if c.block_k >= 32]
        return super()._filter_configs(configs)


@register_template_heuristic(
    scaled_mm_device_tma_epilogue_scaling_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="scaled_mm",
)
class CUDAScaledTMAEpilogueScalingTemplateConfigHeuristic(
    ScaledTMAConfigMixin, CUDAConfigHeuristic
):
    """Scaled TMA template heuristic for CUDA: epilogue scaling variants (TensorWise, RowWise)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_persistent_mm_configs for TMA
        self.mm_configs = self.scaled_persistent_mm_configs


@register_template_heuristic(
    scaled_mm_device_tma_main_loop_scaling_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="scaled_mm",
)
class CUDAScaledTMAMainLoopScalingTemplateConfigHeuristic(
    ScaledTMAConfigMixin, CUDAConfigHeuristic
):
    """
    Scaled TMA template heuristic for CUDA:
        main loop scaling variants (BlockWise1x128, BlockWise1x32, BlockWise1x16, BlockWise128x128)
    """

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_persistent_mm_configs for TMA
        self.mm_configs = self.scaled_persistent_mm_configs

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate main loop scaling kernel inputs.
        """
        mat_a, mat_b, scale_a, scale_b = kernel_inputs._input_nodes
        scale_a_size, scale_b_size = scale_a.get_size(), scale_b.get_size()

        scale_option_a, scale_option_b = get_scaling_options(
            mat_a, mat_b, scale_a_size, scale_b_size
        )
        tile_size_a = get_tile_size(scale_option_a)
        tile_size_b = get_tile_size(scale_option_b)

        # Get base scaled MM template configs from superclass
        for template_kwargs in super()._get_template_configs_impl(
            kernel_inputs,
            op_name,
            **kwargs,
        ):
            # Add scaling-specific options for main loop scaling variants

            # Inductor templates require compile-time constants passed in as tl.constexpr values.
            # In cases in which the block size (BLOCK_*) is smaller than the tile size (128, 32, 16),
            # scales must be broadcasted to BLOCK_* (rather than to a tile_sizextile_size chunk).

            template_kwargs["TILE_SIZE_A"] = tile_size_a
            template_kwargs["TILE_SIZE_B"] = tile_size_b

            template_kwargs["MIN_BLOCK_TILE_AM"] = min(
                template_kwargs["BLOCK_M"], tile_size_a
            )
            template_kwargs["MIN_BLOCK_TILE_AK"] = min(
                template_kwargs["BLOCK_K"], tile_size_a
            )
            template_kwargs["MIN_BLOCK_TILE_BK"] = min(
                template_kwargs["BLOCK_K"], tile_size_b
            )
            template_kwargs["MIN_BLOCK_TILE_BN"] = min(
                template_kwargs["BLOCK_N"], tile_size_b
            )

            yield template_kwargs


@register_template_heuristic(
    # regular Blackwell MM template + scaling epilogue from ScaledMMConfigMixin
    blackwell_ws_persistent_device_tma_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="scaled_mm",
)
class CUDAScaledBlackwellTMATemplateConfigHeuristic(
    ScaledBlackwellTMAConfigMixin, CUDAConfigHeuristic
):
    """Scaled Blackwell TMA template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = self.blackwell_scaled_persistent_mm_configs


@register_template_heuristic(
    mm_plus_mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
)
class CUDAMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, CUDAConfigHeuristic
):
    """MM Plus MM template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="int_mm",
)
class CUDAInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Int8 MM template heuristic for CUDA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


# ROCm template-specific classes


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=IS_ROCM,
)
@register_template_heuristic(
    bmm_template.uid,
    "cuda",
    register=IS_ROCM,
)
class ROCmMMTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Standard MM template heuristic for ROCm"""


# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(mm_template.uid, "cuda", register=IS_ROCM, op_name="addmm")
# TODO(coconutruben): replace with template.name once templates are importable
@register_template_heuristic(
    bmm_template.uid, "cuda", register=IS_ROCM, op_name="baddbmm"
)
class ROCmAddMMTemplateConfigHeuristic(AddMMConfigMixin, ROCmMMTemplateConfigHeuristic):
    """Addmm specific mixin for ROCm"""


@register_template_heuristic(
    persistent_mm_template.uid,
    "cuda",
    register=IS_ROCM,
    op_name="addmm",
)
class ROCmAddMMPersistentTemplateConfigHeuristic(
    AddMMConfigMixin, PersistentMMTemplateConfigHeuristic
):
    """Addmm specific mixin for persistent MM on ROCm"""


# TODO(coconutruben): deprecate once autoheuristic is deprecated
@register_template_heuristic("mm-ah", "cuda", register=IS_ROCM)
class ROCmMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Standard MM template heuristic for ROCm using the extra mm configs only (for autoheuristic)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.extra_mm_configs
        self.exhaustive_configs = self.extra_mm_configs


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=IS_ROCM,
    op_name="scaled_mm",
)
class ROCmScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, ROCmConfigHeuristic):
    """Scaled MM template heuristic for ROCm (non-TMA)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs

    def _filter_configs(self, configs: list[BaseConfig]) -> list[BaseConfig]:
        configs = [c for c in configs if c.block_k >= 32]
        return super()._filter_configs(configs)


@register_template_heuristic(
    mm_template.uid,
    "cuda",
    register=IS_ROCM,
    op_name="int_mm",
)
class ROCmInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Int8 MM template heuristic for ROCm"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


@register_template_heuristic(
    mm_plus_mm_template.uid,
    "cuda",
    register=IS_ROCM,
)
class ROCmMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, ROCmConfigHeuristic
):
    """MM Plus MM template heuristic for ROCm"""

    def __init__(self) -> None:
        super().__init__()
        # self.default_num_stages is used to make sure all configs have that in ROCm land
        # for mm_plus_mm, we actually just want stages = 1, as pipelining brings no benefits
        self.default_num_stages = 1
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


# CPU template-specific classes


@register_template_heuristic(mm_template.uid, "cpu")
@register_template_heuristic(bmm_template.uid, "cpu")
class CPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, CPUConfigHeuristic):
    """Standard MM template heuristic for CPU"""


@register_template_heuristic(mm_template.uid, "cpu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "cpu", op_name="baddbmm")
class CPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, CPUMMTemplateConfigHeuristic):
    """Addmm specific mixin for CPU"""


@register_template_heuristic(mm_template.uid, "cpu", op_name="scaled_mm")
class CPUScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CPUConfigHeuristic):
    """Scaled MM template heuristic for CPU (non-TMA)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_mm_configs


@register_template_heuristic(mm_template.uid, "cpu", op_name="int_mm")
class CPUInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, CPUConfigHeuristic):
    """Int8 MM template heuristic for CPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


@register_template_heuristic(mm_plus_mm_template.uid, "cpu")
class CPUMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, CPUConfigHeuristic
):
    """MM Plus MM template heuristic for CPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


# XPU template-specific classes


@register_template_heuristic(mm_template.uid, "xpu")
@register_template_heuristic(bmm_template.uid, "xpu")
class XPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, XPUConfigHeuristic):
    """Standard MM template heuristic for XPU"""

    def __init__(self) -> None:
        super().__init__()

        # TODO(etaf): Design proper exhaustive search space for XPU.
        self.exhaustive_configs = self.mm_configs


@register_template_heuristic(mm_template.uid, "xpu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "xpu", op_name="baddbmm")
class XPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, XPUMMTemplateConfigHeuristic):
    """Addmm specific mixin for XPU"""


@register_template_heuristic(
    persistent_tma_mm_template.uid,
    "xpu",
)
class XPUPersistentTMATemplateConfigHeuristic(
    TMATemplateConfigMixin, XPUConfigHeuristic
):
    """Persistent TMA template heuristic for XPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use persistent_mm_configs
        self.mm_configs = self.persistent_mm_configs


@register_template_heuristic(persistent_tma_mm_template.uid, "xpu", op_name="addmm")
class XPUAddmmPersistentTMATemplateConfigHeuristic(
    AddMMConfigMixin, XPUPersistentTMATemplateConfigHeuristic
):
    """Addmm specific mixin for XPU"""


@register_template_heuristic(mm_template.uid, "xpu", op_name="scaled_mm")
class XPUScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, XPUConfigHeuristic):
    """Scaled MM template heuristic for XPU (non-TMA)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_mm_configs


@register_template_heuristic(mm_template.uid, "xpu", op_name="int_mm")
class XPUInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, XPUConfigHeuristic):
    """Int8 MM template heuristic for XPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


@register_template_heuristic(mm_plus_mm_template.uid, "xpu")
class XPUMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, XPUConfigHeuristic
):
    """MM Plus MM template heuristic for XPU"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs


# MTIA template-specific classes


@register_template_heuristic(mm_template.uid, "mtia")
@register_template_heuristic(bmm_template.uid, "mtia")
class MTIAMMTemplateConfigHeuristic(MMTemplateConfigMixin, MTIAConfigHeuristic):
    """Standard MM template heuristic for MTIA"""


@register_template_heuristic(mm_template.uid, "mtia", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "mtia", op_name="baddbmm")
class MTIAAddMMTemplateConfigHeuristic(AddMMConfigMixin, MTIAMMTemplateConfigHeuristic):
    """Addmm specific mixin for MTIA"""


@register_template_heuristic(mm_template.uid, "mtia", op_name="scaled_mm")
class MTIAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, MTIAConfigHeuristic):
    """Scaled MM template heuristic for MTIA (non-TMA)"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use scaled_mm_configs
        self.mm_configs = self.scaled_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.scaled_mm_configs


@register_template_heuristic(mm_template.uid, "mtia", op_name="int_mm")
class MTIAInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, MTIAConfigHeuristic):
    """Int8 MM template heuristic for MTIA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use int8_mm_configs
        self.mm_configs = self.int8_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.int8_mm_configs


@register_template_heuristic(mm_plus_mm_template.uid, "mtia")
class MTIAMMPlusMMTemplateConfigHeuristic(
    MMPlusMMTemplateConfigMixin, MTIAConfigHeuristic
):
    """MM Plus MM template heuristic for MTIA"""

    def __init__(self) -> None:
        super().__init__()
        # Override mm_configs to use mm_plus_mm_configs
        self.mm_configs = self.mm_plus_mm_configs
        # NOTE: overriding exhaustive configs here to be the same as mm_configs
        # as we haven't validated exhaustive support here yet
        # TODO(coconutruben): remove this once we have validated exhaustive support
        # for scaled_mm
        self.exhaustive_configs = self.mm_plus_mm_configs
<|endoftext|>from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KernelTemplateParams(ABC):
    """Abstract base class for kernel template parameters."""

    @abstractmethod
    def to_kwargs(self) -> dict[str, Any]:
        """Convert params to kwargs dict for template.choice_or_none()"""

    @abstractmethod
    def to_serializeable_dict(self) -> dict[str, Any]:
        """Convert params to serializable dict for storage/caching"""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelTemplateParams:
        """Create params instance from dict"""


class DictKernelTemplateParams(KernelTemplateParams):
    """Simple implementation that wraps a kwargs dict"""

    # NOTE: this is a compatibility layer, until every template
    # has time to define their own params class, with meaningful
    # defaults etc.

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = kwargs

    def to_kwargs(self) -> dict[str, Any]:
        return self.kwargs.copy()

    def to_serializeable_dict(self) -> dict[str, Any]:
        return self.kwargs.copy()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DictKernelTemplateParams:
        return cls(data)
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sympy

from torch._inductor.heuristics.registry import register_template_heuristic

from ...ir import get_free_symbols
from ...kernel.mm import decompose_k_subgraph_template
from ...kernel_inputs import KernelInputs, MMKernelInputs
from ...utils import get_k_splits
from ...virtualized import V
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator


@register_template_heuristic(decompose_k_subgraph_template.uid, None, op_name="mm")
class EmptyDecomposeKConfigHeuristics(TemplateConfigHeuristics):
    """empty heuristics to skip decompose k on anything not cuda"""


@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "xpu",
    op_name="mm",
)
# Register on CUDA (both NVIDIA and ROCm/HIP)
# Runtime enablement is controlled by config.triton.num_decompose_k_splits (0 disables)
@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "cuda",
    op_name="mm",
)
# TODO(coconutruben): enable decompose k on other devices (xpu, cpu, mps, mtia)
# by either adding specific register_template_heuristic tags, or setting the
# device to None (enabled on all devices)
class DecomposeKConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get all the valid k_splits for the given m, n, k.
        """
        if not isinstance(kernel_inputs, MMKernelInputs):
            raise AssertionError(f"{self.__class__.__name__} requires MMKernelInputs")

        # Check for unbacked symbols - if found, yield nothing
        unbacked_symbols = any(
            len(get_free_symbols(itr, unbacked_only=True)) > 0
            for itr in (
                *kernel_inputs.shapes_symbolic(),
                *kernel_inputs.strides_symbolic(),
            )
        )
        if unbacked_symbols:
            return

        m, n, k = kernel_inputs.mnk_symbolic()
        k_splits = get_k_splits(m, n, k)
        for k_split in k_splits:
            if not V.graph.sizevars.statically_known_true(
                sympy.Eq(sympy.Mod(k, k_split), 0)
            ):
                continue
            yield {"k_split": k_split}
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .params import DictKernelTemplateParams, KernelTemplateParams


if TYPE_CHECKING:
    from collections.abc import Generator

    from ...kernel_inputs import KernelInputs


class TemplateConfigHeuristics:
    """Base class for generating sets of configs for an associated template."""

    def should_run(self, inputs: KernelInputs) -> bool:
        """
        hookup to check whether the configs are right to run at all e.g. you can check
        max-autotune specific to your heuristic here or other things
        If this returns False, get_template_configs will yield no configs

        Args:
            inputs: KernelInputs
        """
        return True

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[KernelTemplateParams, None, None]:
        """
        Get template configs for the given inputs.

        Prefer to override the _get_template_configs_impl method
        to leverage things like should_run
        """
        if not self.should_run(kernel_inputs):
            return

        # Generate configs and fuse with extra_kwargs
        for config_dict in self._get_template_configs_impl(kernel_inputs, op_name):
            # Fuse extra_kwargs into config
            yield DictKernelTemplateParams(config_dict)

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get template configs for the given inputs.
        This is the main entry point for template-specific logic.
        """
        # base implementation yields no entries
        yield from []

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        """
        Get extra kwargs for the given inputs/op for the template.

        Use this to return kwargs that are needed for the template, but
        do not change depending on the config/choice, but are rather
        always the same, for all configs
        """
        return {}

    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        """
        Adjust kernel inputs for the given inputs/op for the template.

        override this to adjust the kernel inputs e.g. (un)squeezing
        """
        return kernel_inputs
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...kernel.mm_common import addmm_epilogue
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from ...kernel_inputs import KernelInputs


class AddMMConfigMixin(TemplateConfigHeuristics):
    """
    Simple mixin to handle scalars for addmm like operators (addmm, baddbmm)
    """

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        if op_name not in ["addmm", "baddbmm"]:
            raise AssertionError(f"op_name={op_name} invalid for AddMMConfigMixin")
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {
            **kwargs,
            "epilogue_fn": addmm_epilogue(kernel_inputs.out_dtype(), alpha, beta),
            "epilogue_fn_hash": str(
                ["addmm_epilogue", kernel_inputs.out_dtype(), alpha, beta]
            ),
            "prefix_args": 1,
        }
<|endoftext|>from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch._inductor import config
from torch._inductor.utils import ensure_nvmatmul_heuristics_available
from torch._logging import getArtifactLogger
from torch.utils._ordered_set import OrderedSet

from .gemm import GemmMaxAutotuneTemplateConfigHeuristics


if TYPE_CHECKING:
    from ...kernel_inputs import KernelInputs, MMKernelInputs


log = logging.getLogger(__name__)
# Use autotuning artifact logger for detailed nvMatmulHeuristics logging
# Enable with TORCH_LOGS="+autotuning"
autotuning_log = getArtifactLogger(__name__, "autotuning")

# Type alias for kernel config key tuple.
# Currently matches on (tile_m, tile_n, cluster_m, cluster_n).
# tile_k excluded because nvMatmulHeuristics and cutlass.operators use it to mean different things.
# TODO(nikhilap): Extend config key for stages/split_k https://github.com/pytorch/pytorch/issues/177578
ConfigKey = tuple[int, int, int, int]


@dataclass
class HeuristicConfig:
    """Configuration recommended by nvMatmulHeuristics."""

    tile_m: int
    tile_n: int
    tile_k: int
    cluster_m: int
    cluster_n: int
    stages: int
    split_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int
    estimated_runtime: float


def _make_config_key_from_heuristic(cfg: HeuristicConfig) -> ConfigKey:
    """Build config key from HeuristicConfig returned by nvMatmulHeuristics."""
    return (cfg.tile_m, cfg.tile_n, cfg.cluster_m, cfg.cluster_n)


def _make_config_key_from_kernel_design(design) -> ConfigKey | None:
    """Build config key from cutlass.operators kernel metadata.design."""
    if (
        hasattr(design, "tile_shape")
        and len(design.tile_shape) >= 2
        and hasattr(design, "cluster_shape")
        and len(design.cluster_shape) >= 2
    ):
        return (
            design.tile_shape[0],
            design.tile_shape[1],
            design.cluster_shape[0],
            design.cluster_shape[1],
        )
    return None


def _make_config_key_from_heuristics_kernel(kernel) -> ConfigKey:
    """Build config key from nvMatmulHeuristics kernel config struct."""
    return (
        kernel.cta[0],
        kernel.cta[1],
        kernel.cluster[0],
        kernel.cluster[1],
    )


class NVUniversalGemmHeuristics(GemmMaxAutotuneTemplateConfigHeuristics):
    """
    Heuristics for NVGEMM kernel selection using nvMatmulHeuristics.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        """Check if heuristics should be used.

        Args:
            inputs: KernelInputs
        """
        return super().should_run(inputs) and ensure_nvmatmul_heuristics_available()

    def filter_kernels(
        self,
        kernels: list,
        inputs: MMKernelInputs,
        count: int,
        accumulator_type: torch.dtype = torch.float32,
    ) -> list:
        """
        Filter and rank kernels using nvMatmulHeuristics.

        Matches on (tile_m, tile_n, tile_k, cluster_m, cluster_n).
        Returns kernels sorted by estimated runtime.

        If nvMatmulHeuristics is not installed or max_autotune is disabled,
        returns the first `count` kernels without heuristic ranking.

        Args:
            kernels: List of cutlass.operators.Operator objects
            inputs: MMKernelInputs with matrix shapes, dtypes, and strides
            count: Maximum number of kernels to return
            accumulator_type: Accumulator dtype

        Returns:
            Filtered list of kernels, sorted by estimated performance
        """
        if not self.should_run(inputs):
            return kernels[:count]

        m, n, k = inputs.mnk_hinted()
        batch_size = inputs.batch_hinted()
        dtype_a = inputs.dtype(inputs._mat1_idx)
        dtype_b = inputs.dtype(inputs._mat2_idx)
        out_dtype = inputs.out_dtype()
        strides = inputs.strides_hinted()
        layout_a = "row" if strides[inputs._mat1_idx][-1] == 1 else "col"
        layout_b = "row" if strides[inputs._mat2_idx][-1] == 1 else "col"

        config_to_kernels = self._extract_config_to_kernels(kernels)

        if not config_to_kernels:
            log.debug("Could not extract kernel configs, using first %d kernels", count)
            return kernels[:count]

        heuristic_configs = self._get_heuristic_configs(
            m,
            n,
            k,
            dtype_a,
            layout_a,
            layout_b,
            count,
            OrderedSet(config_to_kernels.keys()),
            accumulator_type,
            batch_size,
            dtype_b=dtype_b,
            out_dtype=out_dtype,
        )

        if not heuristic_configs:
            log.debug("No heuristic configs found, using first %d kernels", count)
            return kernels[:count]

        # Match kernels to heuristic configs
        matched: list[tuple] = []
        for cfg in heuristic_configs:
            key = _make_config_key_from_heuristic(cfg)
            kernels_for_key = config_to_kernels.get(key)
            if not kernels_for_key:
                continue
            for kernel in kernels_for_key:
                matched.append((kernel, cfg.estimated_runtime))

        if not matched:
            log.debug(
                "No kernels matched heuristic configs, using first %d kernels", count
            )
            return kernels[:count]

        matched.sort(key=lambda x: x[1])
        selected = matched[:count]
        result = [k for k, _ in selected]

        # Supplement with hand-picked configs in the space nvMatmulHeuristics doesn't currently explore.
        if config.nvgemm_supplement_configs:
            _SUPPLEMENT_CONFIGS: OrderedSet[ConfigKey] = OrderedSet(
                [
                    (64, 128, 1, 1),
                    (64, 128, 1, 2),
                    (64, 128, 1, 4),
                    (64, 128, 1, 8),
                    (64, 128, 1, 16),
                    (64, 256, 1, 8),
                    (64, 256, 1, 16),
                    (64, 32, 1, 2),
                    (64, 32, 1, 4),
                    (128, 64, 1, 1),
                    (128, 64, 1, 4),
                    (128, 64, 2, 1),
                    (128, 128, 1, 8),
                    (128, 128, 1, 16),
                    (128, 128, 2, 2),
                    (128, 128, 2, 4),
                    (128, 128, 2, 8),
                    (128, 192, 1, 1),
                    (128, 192, 1, 2),
                    (128, 192, 1, 4),
                    (128, 192, 2, 1),
                    (128, 192, 2, 2),
                    (128, 256, 1, 4),
                    (128, 256, 1, 8),
                    (128, 256, 1, 16),
                    (128, 256, 2, 1),
                    (128, 256, 2, 8),
                    (256, 192, 1, 1),
                    (256, 192, 1, 2),
                    (256, 192, 1, 4),
                    (256, 192, 2, 1),
                    (256, 256, 2, 1),
                    (256, 256, 2, 4),
                    (256, 256, 4, 2),
                    (256, 256, 4, 4),
                    (256, 256, 8, 1),
                    (256, 256, 8, 2),
                    (256, 128, 2, 1),
                    (256, 128, 2, 2),
                    # NVFP4 oracle-best configs (per-shape autotune winners over
                    # an 83-shape LLM sweep) that nvMatmulHeuristics does not
                    # propose; adding them lets autotune reach the oracle-best
                    # config on 73/83 of those shapes.
                    (128, 64, 1, 2),
                    (128, 128, 1, 1),
                    (128, 128, 1, 4),
                    (128, 256, 1, 1),
                    (256, 64, 2, 1),
                    (256, 128, 4, 1),
                    (256, 192, 2, 2),
                    (256, 192, 4, 1),
                    (256, 192, 4, 2),
                    (256, 256, 4, 1),
                ]
            )
            selected_keys = OrderedSet(
                [_make_config_key_from_kernel_design(k.metadata.design) for k in result]
            )
            for key, key_kernels in config_to_kernels.items():
                if key not in selected_keys and key in _SUPPLEMENT_CONFIGS:
                    result.append(key_kernels[0])

        log.debug(
            "Heuristic filtered to %d kernels from %d total", len(result), len(kernels)
        )

        autotuning_log.info(
            "nvMatmulHeuristics kernel filtering: %d heuristic configs matched %d "
            "of %d available kernels, returning top %d",
            len(heuristic_configs),
            len(matched),
            len(kernels),
            len(result),
        )
        for i, (kernel, runtime) in enumerate(selected):
            design = kernel.metadata.design
            autotuning_log.info(
                "  Selected kernel %d: tile=(%d, %d, %d), cluster=(%d, %d), "
                "estimated_runtime=%.2f us",
                i,
                design.tile_shape[0],
                design.tile_shape[1],
                design.tile_shape[2],
                design.cluster_shape[0],
                design.cluster_shape[1],
                runtime * 1e6,
            )

        return result

    def _extract_config_to_kernels(self, kernels: list) -> dict[ConfigKey, list]:
        """Build a map from config key to kernels."""
        config_to_kernels: dict[ConfigKey, list] = defaultdict(list)

        for kernel in kernels:
            key = _make_config_key_from_kernel_design(kernel.metadata.design)
            if key is not None:
                config_to_kernels[key].append(kernel)

        return config_to_kernels

    def _get_layout_enum(self, layout_a: str, layout_b: str):
        """Map layout strings to NvMatmulHeuristicsMatmulLayout enum."""
        import nvMatmulHeuristics

        trans_a = "T" if layout_a == "row" else "N"
        trans_b = "T" if layout_b == "row" else "N"
        layout_str = f"{trans_a}{trans_b}_ROW_MAJOR"
        return nvMatmulHeuristics.NvMatmulHeuristicsMatmulLayout[layout_str]

    def _make_validity_callback(
        self,
        valid_configs: OrderedSet[ConfigKey],
    ):
        """
        Create callback for nvMatmulHeuristics that only accepts configurations
        matching the available cutlass.operators kernel tile/cluster shapes.
        """

        def validity_check(kernel_config_ptr, problem_ptr):
            kernel = kernel_config_ptr.contents
            key = _make_config_key_from_heuristics_kernel(kernel)
            return 1 if key in valid_configs else 0

        return validity_check

    def _get_heuristic_configs(
        self,
        m: int,
        n: int,
        k: int,
        dtype_a: torch.dtype,
        layout_a: str,
        layout_b: str,
        count: int,
        valid_configs: OrderedSet[ConfigKey],
        accumulator_type: torch.dtype = torch.float32,
        batch_size: int = 1,
        dtype_b: torch.dtype | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> list[HeuristicConfig]:
        """
        Get kernel configurations recommended by nvMatmulHeuristics.

        Uses validity callback to filter to cutlass.operators-compatible configs.
        """
        import nvMatmulHeuristics

        dtype_to_cublas = {
            torch.float64: "D",
            torch.float32: "S",
            torch.float16: "H",
            torch.bfloat16: "T",
            torch.float8_e4m3fn: "Q",
            torch.float8_e5m2: "R",
            torch.float4_e2m1fn_x2: "F4",
        }
        a_char = dtype_to_cublas.get(dtype_a, "H")
        b_char = dtype_to_cublas.get(dtype_b or dtype_a, a_char)
        out_char = dtype_to_cublas.get(out_dtype or dtype_a, a_char)
        acc_char = dtype_to_cublas.get(accumulator_type, "S")

        # nvMatmulHeuristics precision string formats:
        # - 3-letter {A}{B}{out}: used for standard GEMM and multi-char tokens (F4, BF)
        # - 5-letter {A}{B}{C}{compute}{D}: used for single-char FP8 types (Q, R, O)
        has_multichar = any(len(c) > 1 for c in (a_char, b_char, out_char))
        if has_multichar:
            precision = f"{a_char}{b_char}{out_char}"
        elif a_char != b_char or a_char in ("Q", "R", "O"):
            precision = f"{a_char}{b_char}{out_char}{acc_char}{out_char}"
        else:
            precision = f"{a_char}{acc_char}{out_char}"

        # NvMatmulHeuristicsInterfaceEx configuration:
        # - backend=CUTLASS3: Use CUTLASS 3.x kernel database for Hopper+ GPUs
        #   TODO(nikhilap): Update when nvMatmulHeuristics supports CUTLASS 4
        # - flags=PERF_MODEL_BASED_AUTO_TUNING: Rank kernels using analytical
        #   performance model (faster than empirical profiling)
        # - load_discovery_implicitly=True: Auto-load kernel discovery sets on demand
        lh = nvMatmulHeuristics.NvMatmulHeuristicsInterfaceEx(
            backend=nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3,
            flags=nvMatmulHeuristics.NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING,
            load_discovery_implicitly=True,
        )

        backend = lh.createBackend(nvMatmulHeuristics.NvMatmulHeuristicsTarget.CUTLASS3)

        validity_callback = self._make_validity_callback(valid_configs)
        lh.setBackendCallbackProperty(
            backend,
            nvMatmulHeuristics.NvMatmulHeuristicsBackendPropertyCallbackKind.KERNEL_ADDITIONAL_VALIDITY_CHECK,
            validity_callback,
        )

        layout = self._get_layout_enum(layout_a, layout_b)

        lh.loadInternalDiscoverySet(layout, precision=precision)

        problem = lh.makeNvMatmulHeuristicsProblem(
            m, n, k, layout, batch_size=batch_size
        )
        raw_configs = lh.getEx(problem, count, backend, precision=precision)
        lh.destroyBackend(backend)

        if not raw_configs:
            autotuning_log.debug(
                "nvMatmulHeuristics returned 0 configs for M=%d, N=%d, K=%d, "
                "dtype=%s, layout=(%s, %s), precision=%s",
                m,
                n,
                k,
                dtype_a,
                layout_a,
                layout_b,
                precision,
            )
            return []

        configs = []
        for cfg in raw_configs:
            kernel = cfg["kernel"]
            configs.append(
                HeuristicConfig(
                    tile_m=kernel.cta_tile_m,
                    tile_n=kernel.cta_tile_n,
                    tile_k=kernel.cta_tile_k,
                    cluster_m=kernel.cluster_m,
                    cluster_n=kernel.cluster_n,
                    stages=kernel.stages,
                    split_k=kernel.split_k,
                    warp_tile_m=kernel.warp_tile_m,
                    warp_tile_n=kernel.warp_tile_n,
                    warp_tile_k=kernel.warp_tile_k,
                    estimated_runtime=cfg["runtime"],
                )
            )

        autotuning_log.info(
            "nvMatmulHeuristics for M=%d, N=%d, K=%d, dtype=%s, layout=(%s, %s), "
            "precision=%s: %d configs returned",
            m,
            n,
            k,
            dtype_a,
            layout_a,
            layout_b,
            precision,
            len(configs),
        )
        for i, cfg in enumerate(configs):
            runtime_us = cfg.estimated_runtime * 1e6
            autotuning_log.info(
                "  Config %d: tile=(%d, %d, %d), cluster=(%d, %d), "
                "stages=%d, split_k=%d, warp_tile=(%d, %d, %d), "
                "estimated_runtime=%.2f us",
                i,
                cfg.tile_m,
                cfg.tile_n,
                cfg.tile_k,
                cfg.cluster_m,
                cfg.cluster_n,
                cfg.stages,
                cfg.split_k,
                cfg.warp_tile_m,
                cfg.warp_tile_n,
                cfg.warp_tile_k,
                runtime_us,
            )

        return configs


# Singleton instance for use in add_nv_universal_gemm_choices
_nvgemm_heuristics: NVUniversalGemmHeuristics | None = None


def get_nvgemm_heuristics() -> NVUniversalGemmHeuristics:
    """Get the singleton NVUniversalGemmHeuristics instance."""
    global _nvgemm_heuristics
    if _nvgemm_heuristics is None:
        _nvgemm_heuristics = NVUniversalGemmHeuristics()
    return _nvgemm_heuristics
<|endoftext|>"""Codegen heuristics for Triton reduction kernel autotuning.

Contains the default (CUDA) heuristic and device-specific overrides for
ROCm/HIP and XPU.  Subclasses override only the methods that differ
from the base.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    register_codegen_heuristic,
)
from torch._inductor.runtime.hints import ReductionHint, TRITON_MAX_BLOCK
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.utils import prefix_is_reduction
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._inductor.runtime.triton_compat import Config


# ------------------------------------------------------------------
# Helpers (moved from triton_heuristics.py, only used by reduction)
# ------------------------------------------------------------------


def _get_tiling_scores(
    inductor_meta: dict[str, Any],
    size_hints: dict[str, int],
) -> dict[str, float]:
    """Retrieve tiling scores, providing suitable defaults if missing."""
    return inductor_meta.get("tiling_scores") or dict.fromkeys(size_hints, 1)


def _match_target_block_product(
    size_hints,
    tiling_scores,
    target_block_product,
    min_block_size=1,
    min_red_block: int | None = 4,
):
    """
    Distribute block sizes across dimensions according to tiling scores,
    aiming to match a target product of block sizes.
    """
    min_red_block = (
        min_block_size if min_red_block is None else max(min_red_block, min_block_size)
    )
    total_score = sum(tiling_scores.values())
    if total_score == 0:
        min_block_size = 1
        tiling_scores = dict.fromkeys(tiling_scores.keys(), target_block_product)
        total_score = target_block_product * len(tiling_scores)

    block_sizes: dict[str, int] = {}
    relative_scores: dict[str, float] = {}
    curr_block_product = 1

    for dim, score in tiling_scores.items():
        if score == 0 and "r" not in dim:
            block_sizes[dim] = 1
            relative_scores[dim] = 0
            continue

        size = min_block_size if "r" not in dim else min_red_block
        block_sizes[dim] = size
        curr_block_product *= size
        relative_scores[dim] = score / total_score

    while curr_block_product < target_block_product and relative_scores:
        dim, score = max(relative_scores.items(), key=lambda item: item[1])

        if (
            block_sizes[dim] >= TRITON_MAX_BLOCK[dim.capitalize()]
            or block_sizes[dim] >= size_hints[dim]
        ):
            del relative_scores[dim]
            continue

        block_sizes[dim] *= 2
        relative_scores[dim] /= 2
        curr_block_product *= 2

    return block_sizes


def _adapt_config_for_tiling(
    size_hints,
    tiling_scores,
    original_x,
    original_r,
    num_warps=None,
    num_stages=1,
    register_intensive=False,
    persistent_reduction=False,
    waves_per_eu=None,
    *,
    warp_size: int = 32,
) -> Config:
    """
    Create an adapted configuration based on tiling scores,
    redistributing the same total block size (x * r) according to tiling scores.
    """
    from torch._inductor.runtime.triton_heuristics import triton_config_tiled_reduction

    if not all(s in tiling_scores for s in size_hints):
        raise AssertionError(
            f"Missing size_hints in tiling_scores: {OrderedSet(size_hints) - OrderedSet(tiling_scores)}"
        )
    target_block_product = original_x * original_r
    block_sizes = _match_target_block_product(
        size_hints, tiling_scores, target_block_product
    )

    return triton_config_tiled_reduction(
        size_hints,
        block_sizes["x"],
        block_sizes["y"],
        block_sizes["r0_"],
        num_stages=num_stages,
        register_intensive=register_intensive,
        waves_per_eu=waves_per_eu,
        warp_size=warp_size,
    )


def _outer_config_opt(
    make_config, size_hints, rnumel, inductor_meta, num_dynamic, register_intensive
):
    """Optimized outer config for CUDA (non-HIP)."""
    max_x_block, x_block = 256, 64
    load_factor = inductor_meta.get("num_load", 0)
    x = size_hints["x"]
    num_warps = None

    if x <= 1024:
        x_block = max(min(x // 128, 8), 2)
        outer_r_block = min(rnumel, 64)
    elif x // 4096 <= 8:
        x_block = 16
        outer_r_block = 512 // x_block
    elif num_dynamic > 1:
        outer_r_block = max(min((rnumel // 64), 64), 8)
    elif num_dynamic == 1:
        outer_r_block = (
            1 if load_factor >= 3 else min(next_power_of_2(max(rnumel, 128) // 128), 8)
        )
    else:
        x_block = max(min(max_x_block, next_power_of_2(x // 4096)), x_block)
        if load_factor < 4 or rnumel <= 128:
            outer_r_block = 512 // x_block
        else:
            if rnumel >= 2048:
                outer_r_block = 64
            else:
                outer_r_block = 32
            x_block = min(x_block, 32)
            num_warps = 4

    return make_config(
        x_block,
        outer_r_block,
        num_warps=num_warps,
        register_intensive=register_intensive,
    )


# ------------------------------------------------------------------
# Default (CUDA) reduction heuristic
# ------------------------------------------------------------------


@register_codegen_heuristic("reduction")
class ReductionHeuristic(CodegenConfigHeuristics):
    """Default reduction autotuning configs (CUDA non-ROCm)."""

    def get_configs(
        self,
        *,
        size_hints: dict[str, int],
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
        num_dynamic: int = 0,
    ) -> list[Config]:
        """Generate non-persistent reduction autotuning configs."""
        from torch._inductor.runtime.triton_heuristics import (
            get_total_reduction_numel,
            make_matmul_triton_config,
            triton_config_reduction,
            triton_native_bmm_configs,
            triton_native_mm_configs,
        )

        reduction_hint = inductor_meta.get("reduction_hint")
        rnumel = get_total_reduction_numel(size_hints)

        max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
            "max_autotune_pointwise"
        )

        register_intensive = False
        loads_and_red = inductor_meta.get("num_load", 0) + inductor_meta.get(
            "num_reduction", 0
        )

        device_major = triton_meta["device"].major
        warp_size = triton_meta["device"].warp_size_or_default
        MAX_R0_BLOCK = 1024 if device_major is not None and device_major >= 10 else 2048
        if size_hints["x"] >= 1024 and loads_and_red >= 10:
            MAX_R0_BLOCK = 1024
            register_intensive = True

        if triton_meta.get("native_matmul"):
            if len(size_hints) == 3:
                return [
                    make_matmul_triton_config(sizes, num_warps, num_stages)
                    for sizes, num_warps, num_stages in triton_native_mm_configs
                ]
            elif len(size_hints) == 4:
                return [
                    make_matmul_triton_config(sizes, num_warps, num_stages)
                    for sizes, num_warps, num_stages in triton_native_bmm_configs
                ]
            else:
                raise NotImplementedError("native matmul only supports mm/bmm pattern")

        def make_config(
            x,
            r,
            num_warps=None,
            num_stages=1,
            register_intensive=False,
            dynamic_scale_rblock=True,
            waves_per_eu=None,
        ):
            if "y" in size_hints:
                tiling_scores = _get_tiling_scores(inductor_meta, size_hints)
                return _adapt_config_for_tiling(
                    size_hints,
                    tiling_scores,
                    x,
                    r,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    register_intensive=register_intensive,
                    waves_per_eu=waves_per_eu,
                    warp_size=warp_size,
                )
            else:
                return triton_config_reduction(
                    size_hints,
                    x,
                    r,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    register_intensive=register_intensive,
                    waves_per_eu=waves_per_eu,
                    dynamic_scale_rblock=dynamic_scale_rblock,
                    reduction_hint=reduction_hint,
                    warp_size=warp_size,
                )

        contiguous_config = make_config(
            # Default XBLOCK=2 launches too few programs to fill
            # the device. Prefer XBLOCK=1 so the autotuner has a candidate
            # that can saturate all CUs.
            1
            if (torch.version.hip and size_hints.get("x", 0) <= 64)
            else (2 if rnumel <= 2048 else 1),
            min(rnumel, MAX_R0_BLOCK),
            register_intensive=register_intensive,
        )
        tiny_config = make_config(
            2 * (256 // rnumel) if rnumel <= 256 else 1,
            min(rnumel, MAX_R0_BLOCK),
            register_intensive=register_intensive,
        )

        outer_config = self._get_outer_config(
            make_config,
            size_hints,
            rnumel,
            inductor_meta,
            num_dynamic,
            register_intensive,
        )

        configs: list[Config] = []

        if inductor_meta.get("add_persistent_rblock") and loads_and_red <= 8:
            xnumel = max(4096 // rnumel, 1)
            c = make_config(
                xnumel,
                min(rnumel, 32768),
                register_intensive=register_intensive,
                dynamic_scale_rblock=False,
            )
            configs.append(c)

        # For 3d tiling, default to more autotuning initially
        if "y" in size_hints:
            pass
        elif max_autotune_enabled:
            pass
        elif reduction_hint == ReductionHint.INNER:
            return configs + [contiguous_config]
        elif reduction_hint == ReductionHint.OUTER:
            return configs + [outer_config]
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return configs + [tiny_config]

        result_configs = configs + [
            contiguous_config,
            outer_config,
            tiny_config,
            make_config(64, 64),
            make_config(8, 512),
            make_config(64, 4, num_warps=8),
        ]

        return self._finalize_configs(
            result_configs, make_config, size_hints, inductor_meta
        )

    def get_persistent_configs(
        self,
        *,
        size_hints: dict[str, int],
        reduction_hint: Any = False,
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
    ) -> list[Config]:
        """Generate persistent reduction autotuning configs."""
        from torch._inductor.runtime.triton_heuristics import (
            get_total_reduction_numel,
            make_matmul_triton_config,
            triton_config_reduction,
            triton_config_tiled_reduction,
            triton_native_persistent_bmm_configs,
            triton_native_persistent_mm_configs,
        )

        inductor_meta = {} if inductor_meta is None else inductor_meta
        # Under deterministic mode, canonicalize the batch-dim hint so the
        # candidate-config branching below (e.g. xnumel // 8 < 128) doesn't pick
        # a different (XBLOCK, num_warps) for bs=N vs bs=N/2. Different picks
        # change the bf16 reduction order and break batch invariance in
        # persistent reductions like LayerNorm.
        if inductor_meta.get("batch_invariant"):
            size_hints = dict(size_hints)
            if "x" in size_hints:
                size_hints["x"] = max(size_hints["x"], 4096)

        xnumel = size_hints["x"]
        rnumel = get_total_reduction_numel(size_hints)

        MAX_PERSISTENT_BLOCK_NUMEL = 4096
        warp_size = triton_meta["device"].warp_size_or_default

        if triton_meta.get("native_matmul"):
            from torch._inductor.runtime.hints import native_matmul_persistent_rblock
            from torch._inductor.runtime.triton_heuristics import (
                _cap_native_matmul_configs,
            )

            native_matmul_rblock = inductor_meta.get("native_matmul_persistent_rblock")
            if native_matmul_rblock is None:
                native_matmul_rblock = native_matmul_persistent_rblock(rnumel)

            if len(size_hints) == 3:
                configs = [
                    make_matmul_triton_config(sizes, num_warps, num_stages)
                    for sizes, num_warps, num_stages in triton_native_persistent_mm_configs
                ]
                return _cap_native_matmul_configs(configs, native_matmul_rblock)
            elif len(size_hints) == 4:
                configs = [
                    make_matmul_triton_config(sizes, num_warps, num_stages)
                    for sizes, num_warps, num_stages in triton_native_persistent_bmm_configs
                ]
                return _cap_native_matmul_configs(configs, native_matmul_rblock)
            else:
                raise NotImplementedError("native matmul only supports mm/bmm pattern")

        max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
            "max_autotune_pointwise"
        )

        xblock_vals = self._persistent_xblock_vals()

        if "y" not in size_hints:
            configs = [
                triton_config_reduction(
                    size_hints,
                    xblock,
                    rnumel,
                    register_intensive=True,
                    reduction_hint=reduction_hint,
                    warp_size=warp_size,
                )
                for xblock in xblock_vals
                if xblock == 1
                or (rnumel * xblock <= MAX_PERSISTENT_BLOCK_NUMEL and xblock <= xnumel)
            ]
        else:
            configs = []
            tiling_scores = _get_tiling_scores(inductor_meta, size_hints)
            x_y_scores = {dim: tiling_scores[dim] for dim in ("x", "y")}
            for target_block_size in xblock_vals:
                if target_block_size * rnumel > MAX_PERSISTENT_BLOCK_NUMEL:
                    continue

                block_sizes = _match_target_block_product(
                    size_hints, x_y_scores, target_block_size
                )
                configs.append(
                    triton_config_tiled_reduction(
                        size_hints,
                        block_sizes["x"],
                        block_sizes["y"],
                        rnumel,
                        warp_size=warp_size,
                    )
                )

        tiny_configs = [
            triton_config_reduction(
                size_hints,
                2 * (256 // rnumel) if rnumel <= 256 else 1,
                rnumel,
                warp_size=warp_size,
            )
        ]

        # Defer to more autotuning for 3d tiling
        if "y" in size_hints:
            pass
        elif not max_autotune_enabled:
            if reduction_hint == ReductionHint.INNER and rnumel >= 256:
                if (
                    rnumel > 1024
                    or xnumel // 8 < 128
                    or inductor_meta.get("RSPLIT_SIZE")
                ):
                    configs = configs[:1]
                else:
                    configs = self._persistent_inner_config(
                        size_hints,
                        rnumel,
                        xnumel,
                        inductor_meta,
                        reduction_hint,
                        warp_size,
                    )
            elif reduction_hint == ReductionHint.OUTER:
                configs = configs[-1:]
            elif reduction_hint == ReductionHint.OUTER_TINY:
                configs = tiny_configs
        else:
            configs = self._persistent_max_autotune_extras(configs, tiny_configs)

        for c in configs:
            for p in size_hints:
                if prefix_is_reduction(p):
                    c.kwargs.pop(f"{p.upper()}BLOCK")  # type: ignore[union-attr]

        return configs

    # ------------------------------------------------------------------
    # Hook methods for device-specific overrides
    # ------------------------------------------------------------------

    def _get_outer_config(
        self,
        make_config,
        size_hints,
        rnumel,
        inductor_meta,
        num_dynamic,
        register_intensive,
    ):
        """Outer config. CUDA uses optimized heuristic."""
        return _outer_config_opt(
            make_config,
            size_hints,
            rnumel,
            inductor_meta,
            num_dynamic,
            register_intensive,
        )

    def _finalize_configs(self, configs, make_config, size_hints, inductor_meta):
        """Post-process non-persistent configs."""
        return configs

    def _persistent_xblock_vals(self) -> list[int]:
        """XBLOCK values for persistent reduction."""
        return [1, 8, 32, 128]

    def _persistent_inner_config(
        self,
        size_hints,
        rnumel,
        xnumel,
        inductor_meta,
        reduction_hint,
        warp_size: int = 32,
    ) -> list[Config]:
        """Config for INNER hint in persistent reduction."""
        from torch._inductor.runtime.triton_heuristics import triton_config_reduction

        x_block = min(1024 // rnumel, 8)
        return [
            triton_config_reduction(
                size_hints,
                x_block,
                rnumel,
                register_intensive=True,
                num_warps=1,
                min_num_warps=1,
                reduction_hint=reduction_hint,
                warp_size=warp_size,
            )
        ]

    def _persistent_max_autotune_extras(self, configs, tiny_configs):
        """Extra processing in max_autotune mode for persistent reduction."""
        return configs

    def get_cooperative_configs(
        self,
        *,
        size_hints: dict[str, int],
        reduction_hint: Any,
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
    ) -> list[Config]:
        """Generate configs for cooperative reduction (RSPLIT)."""
        from torch._inductor.runtime.hints import TRITON_MAX_RSPLIT
        from torch._inductor.runtime.runtime_utils import last_power_of_2

        # Cooperative reductions currently only support a single reduction dimension.
        if len(size_hints) != 2:
            raise AssertionError(
                "Cooperative reductions don't support tiling reduction dims"
            )
        xnumel, rnumel = size_hints["x"], size_hints["r0_"]

        target = last_power_of_2(triton_meta["device"].multi_processor_count)
        split = max(1, min((rnumel, target // xnumel, TRITON_MAX_RSPLIT)))
        if inductor_meta["persistent_reduction"]:
            configs = self.get_persistent_configs(
                size_hints={"x": xnumel, "r0_": rnumel // split},
                reduction_hint=reduction_hint,
                inductor_meta=inductor_meta,
                triton_meta=triton_meta,
            )
        else:
            configs = self.get_configs(
                size_hints={"x": xnumel, "r0_": rnumel // split},
                inductor_meta=inductor_meta,
                triton_meta=triton_meta,
            )
        for config in configs:
            config.kwargs["RSPLIT"] = split  # type: ignore[union-attr]
        return configs

    def apply_rsplit_size(
        self,
        configs: list[Config],
        *,
        size_hints: dict[str, int],
        inductor_meta: dict[str, Any],
    ) -> list[Config]:
        """Apply RSPLIT_SIZE / mix-order post-processing to persistent configs."""
        import copy

        from torch._inductor.runtime.triton_heuristics import unique_configs

        max_autotune_enabled = inductor_meta.get("max_autotune") or inductor_meta.get(
            "max_autotune_pointwise"
        )

        rsplit_size = inductor_meta.get("RSPLIT_SIZE")
        if not rsplit_size:
            return configs

        rnumel_hint = size_hints["r0_"]
        min_x_block = 1
        if rnumel_hint <= 512:
            min_x_block = 4
        required_x_block = 1
        if (
            tma_min_block_sizes := inductor_meta.get("tma_min_block_sizes")
        ) is not None:
            required_x_block = max(
                required_x_block, tma_min_block_sizes.get("XBLOCK", 1)
            )
        x_block = min(max(rsplit_size // 32, min_x_block, required_x_block), 16)

        new_configs: list[Config] = []
        for c in configs:
            c.kwargs["RSPLIT_SIZE"] = rsplit_size  # type: ignore[union-attr]
            c.kwargs["XBLOCK"] = x_block  # type: ignore[union-attr]

            num_iters = rsplit_size // x_block

            if inductor_meta.get("mix_order_reduction_allow_multi_stages", True):
                MAX_NUM_STAGES = 2 if rnumel_hint > 8192 else 3
            else:
                MAX_NUM_STAGES = 1
            # Triton's tl.range pipeliner cannot predicate the ttng.tensormap_create
            # emitted by device-side descriptors.
            if inductor_meta.get("uses_device_tma"):
                MAX_NUM_STAGES = 1
            c.kwargs["NUM_STAGES"] = min(  # type: ignore[union-attr]
                max(num_iters // 4, 1), MAX_NUM_STAGES
            )

            if rnumel_hint <= 1024:
                c.num_warps //= 2  # type: ignore[union-attr]
                c.num_warps = max(c.num_warps, 1)  # type: ignore[union-attr]
                new_configs.append(c)

                if max_autotune_enabled:
                    newc = copy.deepcopy(c)
                    newc.num_warps = 2  # type: ignore[union-attr]
                    new_configs.append(newc)
            else:
                new_configs.append(c)

                max_warps_limit = self._max_warps_limit()
                if max_autotune_enabled and c.num_warps < max_warps_limit:  # type: ignore[union-attr]
                    newc = copy.deepcopy(c)
                    newc.num_warps *= 2  # type: ignore[union-attr]
                    new_configs.append(newc)
        return unique_configs(new_configs)

    def _max_warps_limit(self) -> int:
        """Max warps limit for RSPLIT_SIZE processing."""
        return 32

    def get_split_scan_configs(
        self,
        *,
        size_hints: dict[str, int],
        inductor_meta: dict[str, Any],
        triton_meta: dict[str, Any],
    ) -> list[Config]:
        """Generate configs for split scan kernels."""
        configs = self.get_configs(
            size_hints=size_hints,
            inductor_meta=inductor_meta,
            triton_meta=triton_meta,
        )
        min_rblock = inductor_meta.get("min_split_scan_rblock", 256)
        for cfg in configs:
            for var in list(cfg.kwargs.keys()):  # type: ignore[union-attr]
                if var.startswith("R") and cfg.kwargs[var] < min_rblock:  # type: ignore[union-attr]
                    cfg.kwargs[var] = min_rblock  # type: ignore[union-attr]
        return configs


# ------------------------------------------------------------------
# ROCm/HIP reduction heuristic
# ------------------------------------------------------------------


@register_codegen_heuristic("reduction", "hip", register=torch.version.hip is not None)
class ROCmReductionHeuristic(ReductionHeuristic):
    """Reduction configs for ROCm/HIP devices."""

    def _get_outer_config(
        self,
        make_config,
        size_hints,
        rnumel,
        inductor_meta,
        num_dynamic,
        register_intensive,
    ):
        # HIP uses simple outer config (no outer_config_opt)
        return make_config(64, 8, register_intensive=register_intensive)

    def _finalize_configs(self, configs, make_config, size_hints, inductor_meta):
        hip_configs = [
            make_config(1024, 8, num_warps=4, num_stages=1, waves_per_eu=2),
            make_config(512, 8, num_warps=4, num_stages=1, waves_per_eu=1),
        ]
        configs.extend(hip_configs)

        max_persistent_rblock = inductor_meta.get("max_persistent_rblock", 0)
        if max_persistent_rblock > 0:
            configs = [
                c
                for c in configs
                if c.kwargs.get("XBLOCK", 0) * max_persistent_rblock <= 4096
            ]
        return configs

    def _persistent_xblock_vals(self) -> list[int]:
        return [1, 4, 8, 16, 32, 64, 128, 256]

    def _persistent_max_autotune_extras(self, configs, tiny_configs):
        for conf in tiny_configs:
            if conf not in configs:
                configs.append(conf)
        return configs

    def _max_warps_limit(self) -> int:
        return 16


# ------------------------------------------------------------------
# XPU reduction heuristic
# ------------------------------------------------------------------


@register_codegen_heuristic("reduction", "xpu", register=torch.xpu._is_compiled())
class XPUReductionHeuristic(ReductionHeuristic):
    """Reduction configs for XPU devices."""

    def _persistent_inner_config(
        self,
        size_hints,
        rnumel,
        xnumel,
        inductor_meta,
        reduction_hint,
        warp_size: int = 32,
    ) -> list[Config]:
        from torch._inductor.runtime.triton_heuristics import triton_config_reduction

        # TODO(Intel): CUDA uses num_warps = 1 to disable shared memory.
        # We apply different configurations from #168335.
        # We currently let cost model in Triton to decide whether to use
        # shared memory.
        loads_and_stores = inductor_meta.get("num_load", 0) + inductor_meta.get(
            "num_store", 0
        )
        x_block = 8
        if xnumel // x_block < 128 or loads_and_stores >= 5:
            x_block = 1
        return [
            triton_config_reduction(
                size_hints,
                x_block,
                rnumel,
                register_intensive=True,
                num_warps=None,
                min_num_warps=None,
                reduction_hint=None,
                warp_size=warp_size,
            )
        ]
<|endoftext|>"""
Codegen heuristics for inductor-generated Triton kernels.

These heuristics control autotuning config generation for runtime-compiled
kernels: pointwise, reduction, persistent_reduction, etc.

Device dispatch uses the shared registry in heuristics/registry.py
with name-based keys (e.g., "pointwise").
"""

from __future__ import annotations

from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    get_codegen_heuristic,
    register_codegen_heuristic,
)

# Import submodules to trigger registration
from . import pointwise as pointwise, reduction as reduction


__all__ = [
    "CodegenConfigHeuristics",
    "get_codegen_heuristic",
    "register_codegen_heuristic",
]
<|endoftext|>"""Codegen heuristics for Triton pointwise kernel autotuning.

Contains the default (CUDA) heuristic and device-specific overrides for
ROCm/HIP and XPU.  Subclasses override only the dimension methods that
differ from the base.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    register_codegen_heuristic,
)
from torch._inductor.runtime.hints import TileHint, TRITON_MAX_BLOCK


if TYPE_CHECKING:
    from collections.abc import Callable


# ----------------------------------------------------------------------
# Default (CUDA) pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise")
class PointwiseHeuristic(CodegenConfigHeuristics):
    """Default pointwise autotuning configs (CUDA non-ROCm)."""

    def get_configs(
        self,
        size_hints: dict[str, int],
        bs: int,
        triton_config_fn: Callable[..., Any],
        hinted_configs: list[Any],
        tile_hint: TileHint | None = None,
        inductor_meta: dict[str, Any] | None = None,
    ) -> list[Any]:
        inductor_meta = inductor_meta or {}
        n = len(size_hints)
        if n == 1:
            return self._configs_1d(
                size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
            )
        if n == 2:
            return self._configs_2d(
                size_hints,
                bs,
                hinted_configs,
                triton_config_fn,
                tile_hint,
                inductor_meta,
            )
        if n == 3:
            return self._configs_3d(
                size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
            )
        raise NotImplementedError(f"size_hints: {size_hints}")

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, bs)]
        return [
            triton_config_fn(size_hints, bs, num_elements_per_warp=256),
            triton_config_fn(size_hints, bs // 2, num_elements_per_warp=64),
            *hinted_configs,
        ]

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        if (
            not inductor_meta.get("autotune_pointwise", True)
            or tile_hint == TileHint.SQUARE
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
        ]

    def _configs_3d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        if not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 16, 16, 16)]
        return [
            triton_config_fn(size_hints, 16, 16, 16),
            triton_config_fn(size_hints, 64, 8, 8),
            triton_config_fn(size_hints, 8, 64, 8),
            triton_config_fn(size_hints, 8, 8, 64),
            triton_config_fn(size_hints, bs, 1, 1),
            triton_config_fn(size_hints, 1, bs, 1),
            triton_config_fn(size_hints, 1, 1, bs),
            *hinted_configs,
        ]


# ----------------------------------------------------------------------
# ROCm/HIP pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise", "hip", register=torch.version.hip is not None)
class ROCmPointwiseHeuristic(PointwiseHeuristic):
    """Pointwise configs for ROCm/HIP devices.

    Inherits _configs_3d from PointwiseHeuristic (same configs).
    """

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        configs = super()._configs_1d(
            size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
        )
        if len(configs) > 1:
            configs.extend(
                [
                    triton_config_fn(size_hints, TRITON_MAX_BLOCK["X"], waves_per_eu=2),
                    triton_config_fn(
                        size_hints,
                        4096,  # wrt: better than the max_block for some kernel
                    ),
                    triton_config_fn(
                        size_hints,
                        2048,
                        num_warps=8,
                        num_stages=2,
                        waves_per_eu=1,  # 20% improvement
                    ),
                ]
            )
            if inductor_meta.get("atomic_add_found"):
                configs.append(
                    triton_config_fn(
                        size_hints,
                        64,
                        num_warps=1,
                        num_stages=1,  # 250% improvement
                    )
                )
        return configs

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # ROCm doesn't skip autotune for SQUARE tile hints
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
            triton_config_fn(size_hints, 64, 32),  # better for some kernels
            triton_config_fn(size_hints, 128, 16),  # +10% for some kernels
            triton_config_fn(size_hints, 128, 32),  # additional 10% more
            triton_config_fn(size_hints, 32, 512),  # +30% for some kernels
        ]


# ----------------------------------------------------------------------
# XPU pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise", "xpu", register=torch.xpu._is_compiled())
class XPUPointwiseHeuristic(PointwiseHeuristic):
    """Pointwise configs for XPU devices."""

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        configs = super()._configs_1d(
            size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
        )
        if len(configs) > 1:
            configs.append(
                # intel-xpu-backend-for-triton #5133
                triton_config_fn(size_hints, 32)
            )
        return configs

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # XPU doesn't skip autotune for SQUARE tile hints
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
            # intel-xpu-backend-for-triton #5198
            triton_config_fn(size_hints, 32, 32, num_warps=8),
            # intel-xpu-backend-for-triton #5199
            triton_config_fn(size_hints, 4, 256),
        ]

    def _configs_3d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        # XPU always uses full set of configs (no autotune gate)
        return [
            triton_config_fn(size_hints, 16, 16, 16),
            triton_config_fn(size_hints, 64, 8, 8),
            triton_config_fn(size_hints, 8, 64, 8),
            triton_config_fn(size_hints, 8, 8, 64),
            triton_config_fn(size_hints, bs, 1, 1),
            triton_config_fn(size_hints, 1, bs, 1),
            triton_config_fn(size_hints, 1, 1, bs),
            *hinted_configs,
        ]
<|endoftext|>import operator

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)


class LearnedHeuristic:
    """
    LearnedHeuristic is a base class for all learned heuristics.
    """

    def __init__(self) -> None:
        pass

    def check_precondition(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> bool:
        return True

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        return None

    def get_confidence_threshold(self) -> float:
        return 1.0

    def get_name(self) -> str:
        return ""

    def get_decisions_ranked(self, context: AHContext) -> list[str] | None:
        return None


class LearnedHeuristicRegression(LearnedHeuristic):
    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        return 1.0

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        choice2feedback = {}
        for choice in choices:
            predicted_feedback = self.get_feedback(context, choice)
            choice2feedback[choice] = predicted_feedback
        sorted_choices_feedback = sorted(
            choice2feedback.items(), key=operator.itemgetter(1)
        )
        highest_feedback = sorted_choices_feedback[-1][1]
        second_highest_feedback = sorted_choices_feedback[-2][1]
        if highest_feedback / second_highest_feedback > self.get_confidence_threshold():
            return sorted_choices_feedback[-1][0]
        # We are not sure which choice is the best one
        return None


class LearnedHeuristicDecision(LearnedHeuristic):
    def get_choice(self, idx: int) -> str | None:
        return None

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        best_choices = self.get_best_choices(context)
        if not best_choices:
            return None
        (best_choice_proba, best_choice_idx) = best_choices[0]
        if best_choice_proba <= self.get_confidence_threshold():
            return None
        return self.get_choice(best_choice_idx)

    def get_decisions_ranked(self, context: AHContext) -> list[str] | None:
        feedback_idx_list = self.get_best_choices(context)
        if feedback_idx_list is None:
            return None
        choices = [
            self.get_choice(feedback_idx[1]) for feedback_idx in feedback_idx_list
        ]
        choices = [choice for choice in choices if choice is not None]
        return choices

    def get_best_choices(self, context: AHContext) -> list[tuple[float, int]] | None:
        return []
<|endoftext|>import importlib
import inspect
import pkgutil
from collections import defaultdict
from typing import Any

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import LearnedHeuristic


def find_and_instantiate_subclasses(
    package_name: str, base_class: Any
) -> list[LearnedHeuristic]:
    instances = []

    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            module_basename = module_name.split(".")[-1]
            if not module_basename.startswith("_"):
                # learned heuristics start with an underscore
                continue
            module = importlib.import_module(module_name)

            # look for classes that are subclasses of base_class
            for _name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_class)
                    and obj != base_class
                ):
                    instance = obj()
                    instances.append(instance)
        except Exception as e:
            print(f"Error processing module {module_name}: {e}")

    return instances


class LearnedHeuristicController:
    """
    Class that finds and instantiates all learned heuristics. It also provides
    a way to get the decision of a learned heuristic.
    """

    existing_heuristics: dict[str, list[LearnedHeuristic]] = defaultdict(list)
    """
    A dictionary that stores all the learned heuristics for each optimization.
    The key is the optimization name, and the value is a list of LearnedHeuristic objects.
    """

    heuristics_initialized: bool = False
    """
    A flag that indicates whether the learned heuristics have been initialized.
    Set to true when the get_decision() function is called for the first time.
    """

    def __init__(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> None:
        self.metadata = metadata
        self.context = context

    def get_heuristics(self, name: str) -> list[LearnedHeuristic]:
        """
        Returns a list of learned heuristics for the given optimization name.
        """

        if not LearnedHeuristicController.heuristics_initialized:
            # learned heuristics are generated into the following package
            learned_heuristics_package = "torch._inductor.autoheuristic.artifacts"

            # learned heuristics have to be of type LearnedHeuristic
            base_class = LearnedHeuristic
            found_heuristics = find_and_instantiate_subclasses(
                learned_heuristics_package, base_class
            )

            for learned_heuristic in found_heuristics:
                opt_name = learned_heuristic.get_name()
                LearnedHeuristicController.existing_heuristics[opt_name].append(
                    learned_heuristic
                )
            LearnedHeuristicController.heuristics_initialized = True

        return LearnedHeuristicController.existing_heuristics[name]

    def get_decision(self) -> Choice | None:
        """
        Returns the decision made by the learned heuristic or None if no heuristic was found or the heuristic is unsure
        which choice to make.
        """

        heuristics = self.get_heuristics(self.metadata.name)
        for heuristic in heuristics:
            if heuristic.check_precondition(self.metadata, self.context):
                return heuristic.get_decision(self.context, self.metadata.choices)
        return None

    def get_decisions_ranked(self, top_k: int) -> list[Choice] | None:
        heuristics = self.get_heuristics(self.metadata.name)
        for heuristic in heuristics:
            if heuristic.check_precondition(self.metadata, self.context):
                choices = heuristic.get_decisions_ranked(self.context)
                if choices is None:
                    return None
                avail_choices = [
                    choice for choice in choices if choice in self.metadata.choices
                ]
                return avail_choices[:top_k]
        return None
<|endoftext|>import json
import os
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    AHOperation,
    Choice,
    CHOICE_COL,
    Feedback,
    FEEDBACK_COL,
    get_metadata_str_from_log,
)
from torch._inductor.autoheuristic.learned_heuristic_controller import (
    LearnedHeuristicController,
)
from torch._inductor.ir import ChoiceCaller
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import get_gpu_shared_memory


class LocalFeedback:
    """
    To be able to collect data for a choice, a function providing feedback given a choice has to be provided.
    LocalFeedback can be used when AutoHeuristic should immediately run the function to collect feedback for each choice
    (see pad_mm.py, where the autotuning happens locally, for an example).
    """

    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None:
        self.feedback_fn = feedback_fn

    def __call__(self, choice: Choice) -> Feedback:
        return self.feedback_fn(choice)


class InconsistentMetadata(Exception):
    """
    Exception that is thrown when AutoHeuristic tries to log data to a file where the metadata stored in the file does
    not match the metadata it would store if the file didn't exist.
    """


class AutoHeuristic:
    """
    AutoHeuristic is a framework that allows one to collect data, learn a heuristic (i.e. a regression tree) and
    generate the heuristic to code. This class allows one to collect data. The collected data can then be used to train
    a heuristic (see torchgen/autoheuristic/).
    """

    collected_feedback: dict[Choice, Feedback]

    def __init__(
        self,
        fallback: Callable[[], Choice],
        choices: list[Choice],
        feedback: LocalFeedback | None,
        context: AHContext,
        name: str,
        augment_context: list[AHOperation] | None = None,
        precondition: Callable[[AHMetadata, AHContext], bool] | None = None,
    ) -> None:
        """
        Initializes an instance of the AutoHeuristic class.

        Args:
            fallback: A callable that returns a Choice when the heuristic is unsure which choice to make, or
            AutoHeuristic is in data collection mode.
            choices: A list of possible choices the heuristic can make.
            feedback: An instance of LocalFeedback that provides feedback for a given choice.
            context: Context to store with each choice and feedback.
            name: A string that identifies the heuristic.
            augment_context: An optional list of AHOperation instances that augment the context.
            precondition: A callable that returns a boolean indicating whether AutoHeuristic should run.
        """
        self.fallback = fallback
        self.choices = choices
        self.feedback = feedback
        self.context = context
        self.name = name
        self.collected_feedback = {}
        self.augment_context = augment_context
        self.metadata = AHMetadata(
            get_gpu_shared_memory(),
            torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0),
            self.choices,
            self.name,
        )
        self.precondition = precondition

        if not self.satisfies_precondition():
            return

        if torch._inductor.config.autoheuristic_log_path == "DEFAULT":
            self.log_path = self.get_default_log_path()
        else:
            self.log_path = torch._inductor.config.autoheuristic_log_path

        if torch._inductor.config.collect_autoheuristic(self.name):
            if self.feedback is not None:
                for choice in self.choices:
                    feedback_val = self.feedback(choice)
                    self.save_data(choice, feedback_val)

    def satisfies_precondition(self) -> bool:
        return self.precondition is None or self.precondition(
            self.metadata, self.context
        )

    def get_choice(self) -> Choice:
        """
        Returns the chosen option based on the value of autoheuristic_use.
        If self.name is one of the comma separated strings in autoheuristic_use,
        it queries a learned heuristic to make a decision. Otherwise, it returns the fallback option.
        """

        if not self.satisfies_precondition():
            return self.fallback()

        if torch._inductor.config.use_autoheuristic(self.name):
            if self.augment_context is not None:
                self.context.apply_operations(self.augment_context)
            controller = LearnedHeuristicController(
                self.metadata,
                self.context,
            )
            decision = controller.get_decision()
            if decision not in self.choices:
                # TODO(AlnisM): We might want to allow this in the future
                return self.fallback()
            if decision is not None:
                return decision
        return self.fallback()

    def get_top_k_choices(
        self, top_k: int, always_included: list[str] | None = None
    ) -> list[Choice] | None:
        if not self.satisfies_precondition():
            return None
        if torch._inductor.config.use_autoheuristic(self.name):
            if self.augment_context is not None:
                self.context.apply_operations(self.augment_context)
            controller = LearnedHeuristicController(
                self.metadata,
                self.context,
            )
            choices = controller.get_decisions_ranked(top_k)
            if choices is None:
                return None
            if always_included is not None:
                for choice in always_included:
                    if choice not in choices:
                        choices.append(choice)
            return choices
        return None

    def get_collected_feedback(self, choice: Choice) -> Any:
        return self.collected_feedback.get(choice, None)

    @staticmethod
    def get_device_identifier() -> str:
        # a heuristic might work well for one GPU, but not for another
        # we store the collected data per GPU model and learn a heuristic per GPU model

        # TODO(AlnisM): just using the device name for now, but the same GPU model can have different names
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name().replace(" ", "_")
            return device_name
        return "non_cuda_device"

    def get_default_log_path(self) -> str:
        device_name = self.get_device_identifier()
        path = f"{cache_dir()}/autoheuristic/{device_name}/"
        os.makedirs(path, exist_ok=True)
        path += f"{self.name}.txt"
        return path

    def serialize_metadata(self) -> str:
        metadata_dict = self.metadata.to_dict()
        (
            num_features,
            cat_features,
        ) = self.context.get_numerical_and_categorical_features()
        metadata_dict["numerical_features"] = num_features
        metadata_dict["categorical_features"] = cat_features
        return json.dumps(metadata_dict)

    def save_data(self, choice: Choice, feedback_val: Feedback) -> None:
        self.collected_feedback[choice] = feedback_val
        log_path = self.log_path

        lines = []
        log_exists = os.path.exists(log_path)
        if log_exists:
            # if log already exists, make sure it is consistent
            metadata = self.serialize_metadata()
            existing_metadata = get_metadata_str_from_log(self.log_path)
            if existing_metadata != metadata:
                raise InconsistentMetadata(
                    "Given metadata does not match existing metadata"
                )
        else:
            lines.append(self.serialize_metadata())
            feature_header = self.context.get_feature_names_csv()
            header = feature_header + "," + CHOICE_COL + "," + FEEDBACK_COL
            lines.append(header)

        line = ""
        feature_values = self.context.get_feature_values_csv()
        line += feature_values + "," + choice + "," + str(feedback_val)
        lines.append(line)

        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")


class AutoHeuristicSelectAlgorithm(AutoHeuristic):
    """
    AutoHeuristicSelectAlgorithm is a subclass of AutoHeuristic that allows one to collect data and learn a heuristic
    when one wants to use AutoHeuristic for kernel choice selection.
    """

    def __init__(
        self,
        fallback: Callable[[], ChoiceCaller | None],
        choices: list[ChoiceCaller],
        input_nodes: list[Any],
        context: AHContext,
        name: str,
        augment_context: list[AHOperation] | None = None,
        precondition: Callable[[AHMetadata, AHContext], bool] | None = None,
    ) -> None:
        """
        The arguments choices, input_nodes and name have to match the ones used in the call to
        autotune_select_algorithm(), e.g. if the following call is made
        autotune_select_algorithm(name, choices, input_nodes, layout), the same name, choices and input_nodes
        have to be used here.
        """
        self.input_nodes = input_nodes
        self.choicestr2choice: dict[str, ChoiceCaller] = {}
        for choice in choices:
            self.choicestr2choice[choice.autoheuristic_id()] = choice
        choices_str = list(self.choicestr2choice.keys())

        def fallback_str() -> str:
            fallback_choice = fallback()
            if fallback_choice is None:
                # TODO: Find a nicer way to handle this
                return "unsure"
            return fallback_choice.autoheuristic_id()

        super().__init__(
            fallback_str,
            choices_str,
            None,
            context,
            name,
            augment_context,
            precondition,
        )

        if (
            torch._inductor.config.collect_autoheuristic(self.name)
            and self.satisfies_precondition()
        ):
            self.register_global_feedback(input_nodes, choices)

    def register_global_feedback(
        self, input_nodes: list[Any], choices: list[ChoiceCaller]
    ) -> None:
        """
        Registers a callback in select_algorithm, which is called with the timing of each choice.
        """

        from torch._inductor.select_algorithm import (
            add_feedback_saver,
            create_inputs_key,
            create_precompile_key,
        )

        def store_global_feedback(
            ah_inputs_key: str,
            ah_precompile_key: str,
            timings: dict[ChoiceCaller, float],
            name: str,
            input_nodes: list[Any],
            choices: list[ChoiceCaller],
            profiled_time: Callable[[], dict[ChoiceCaller, float]],
            precompile_times: dict[ChoiceCaller, float],
        ) -> None:
            current_inputs_key = create_inputs_key(input_nodes)
            if current_inputs_key != ah_inputs_key:
                return
            current_precompile_key = create_precompile_key(
                name, current_inputs_key, choices
            )
            if current_precompile_key != ah_precompile_key:
                return
            for choice, time in timings.items():
                self.save_data(choice.autoheuristic_id(), time)

        inputs_key = create_inputs_key(input_nodes)
        precompile_key = create_precompile_key(self.name, inputs_key, choices)
        feedback_saver = partial(store_global_feedback, inputs_key, precompile_key)
        add_feedback_saver(feedback_saver)

    def get_choice_caller(self) -> ChoiceCaller | None:
        choice = self.get_choice()
        return self.choicestr2choice.get(choice, None)

    def get_top_k_choices_caller(
        self, top_k: int, always_included: list[str] | None = None
    ) -> list[ChoiceCaller] | None:
        choices = self.get_top_k_choices(top_k, always_included)
        if choices is None:
            return None
        return [self.choicestr2choice[choice] for choice in choices]
<|endoftext|>import functools
from collections.abc import Callable
from typing import Any

import torch


Feedback = float
Choice = str
Value = Any

CHOICE_COL = "choice"
FEEDBACK_COL = "feedback"


class AHFeature:
    """
    The context, that AutoHeuristic stores, is a list of features. AutoHeuristic needs to know whether a feature is
    categorical (i.e., not a continuous variable) to learn a machine learning model.
    """

    def __init__(self, name: str, value: Value, is_categorical: bool = False) -> None:
        self.name = name
        self.value = value
        self.is_categorical = is_categorical


class AHOperation:
    """
    AHOperation can be used to augment the data collected by AutoHeuristic.
    One might for example store features like m, k, n, but also want to use
    features like m*n, or k*n, to learn a heuristic. Instead of storing features
    that can be created from the collected data, one can use AHOperation to
    create new features from the collected data.
    """

    def __init__(
        self, name: str, func: Callable[[Any], Value], is_categorical: bool = False
    ) -> None:
        self.name = name
        self.func = func
        self.is_categorical = is_categorical

    def apply_operation(self, data: Any) -> None:
        data[self.name] = self.func(data)


class AHContext:
    """
    This class is used to specify which information AutoHeuristic should store. For each choice, AutoHeuristic will
    store the context and the collected feedback. The context could be something like the shape of a tensor, i.e.,
    information that will help to learn a heuristic.
    """

    features: list[AHFeature]
    context_dict: dict[str, Value]

    def __init__(self) -> None:
        self.features = []
        self.context_dict = {}

    def add_feature(
        self, name: str, value: Value, is_categorical: bool = False
    ) -> None:
        self.features.append(AHFeature(name, value, is_categorical=is_categorical))
        self.context_dict[name] = value

    def get_numerical_and_categorical_features(self) -> tuple[list[str], list[str]]:
        numerical_features = []
        categorical_features = []
        for feature in self.features:
            if feature.is_categorical:
                categorical_features.append(feature.name)
            else:
                numerical_features.append(feature.name)

        return numerical_features, categorical_features

    def get_feature_names_csv(self) -> str:
        return ",".join(feature.name for feature in self.features)

    def get_feature_values_csv(self) -> str:
        return ",".join(str(feature.value) for feature in self.features)

    def get_value(self, name: str) -> Value:
        return self.context_dict[name]

    def apply_operations(self, operations: list[AHOperation]) -> None:
        for op in operations:
            op.apply_operation(self.context_dict)


class AHMetadata:
    def __init__(
        self,
        shared_memory: Any,
        device_capa: tuple[int, int],
        choices: list[Choice],
        name: str,
    ) -> None:
        # use amount of shared_memory and device_capability to identify GPU
        # TODO(AlnisM): there might be a better way to do this
        self.shared_memory = shared_memory
        self.device_capa = device_capa
        self.choices = choices
        self.name = name

    def to_dict(self) -> dict[str, Value]:
        return {
            "shared_memory": self.shared_memory,
            "device_capa": self.device_capa,
            "name": self.name,
        }


def get_metadata_str_from_log(log_path: str) -> str:
    with open(log_path, newline="") as file:
        json_string = file.readline().strip()
        return json_string


def check_minsize(context: AHContext, minsize: int) -> bool:
    return (
        context.get_value("m") >= minsize
        and context.get_value("k") >= minsize
        and context.get_value("n") >= minsize
    )


def pad_mm_precondition(metadata: AHMetadata, context: AHContext) -> bool:
    if metadata.shared_memory == 166912 and metadata.device_capa == (8, 0):
        # A100 precondition
        return check_minsize(context, 512)
    elif metadata.shared_memory == 232448 and metadata.device_capa == (9, 0):
        # H100 precondition
        return check_minsize(context, 768)
    return True


def get_mixedmm_precondition(metadata: AHMetadata, context: AHContext) -> bool:
    m = context.get_value("m")
    k = context.get_value("k")
    n = context.get_value("n")
    if m > 128 or k < 1024 or n < 1024:
        return False
    mat1_iscontig = context.get_value("mat1_iscontig")
    mat2_iscontig = context.get_value("mat2_iscontig")
    return mat1_iscontig and not mat2_iscontig


def get_mult_dims_ops() -> list[AHOperation]:
    m_times_k_op = AHOperation("m*k", lambda data: data["m"] * data["k"])
    m_times_n_op = AHOperation("m*n", lambda data: data["m"] * data["n"])
    k_times_n_op = AHOperation("k*n", lambda data: data["k"] * data["n"])
    return [m_times_k_op, m_times_n_op, k_times_n_op]


def get_arith_intensity(data: Any) -> float:
    m = data["m"]
    k = data["k"]
    n = data["n"]
    if m == 0 or k == 0 or n == 0:
        return 0.0
    return m * k * n / (m * k + k * n + m * n)


def pad_mm_operations() -> list[AHOperation]:
    mult_dims_ops = get_mult_dims_ops()
    k_div_m_times_n_op = AHOperation(
        "k/(m*n)", lambda data: data["k"] / (data["m"] * data["n"])
    )

    def bfloat_perf_hit(data: Any) -> bool:
        m = data["m"]
        k = data["k"]
        n = data["n"]
        is_bfloat = str(data["mat1_dtype"]) == "torch.bfloat16"
        return k > (m * 1024) and k > (n * 1024) and is_bfloat

    bfloat_perf_hit_op = AHOperation(
        "bfloat_perf_hit", bfloat_perf_hit, is_categorical=True
    )

    arith_intensity_op = AHOperation("arith_intensity", get_arith_intensity)
    dims_need_padding_ops = get_dims_need_padding_ops()
    dims_multiple_ops = get_dims_multiple_ops()
    is_contig_ops = get_is_contig_ops()

    ah_operations = mult_dims_ops + [
        k_div_m_times_n_op,
        bfloat_perf_hit_op,
        arith_intensity_op,
    ]
    ah_operations.extend(dims_need_padding_ops)
    ah_operations.extend(dims_multiple_ops)
    ah_operations.extend(is_contig_ops)
    return ah_operations


def between_op(data: Any, dim: str, lower: int, upper: int) -> bool:
    return data[dim] >= lower and data[dim] <= upper


def between_ops() -> list[AHOperation]:
    dims = ["m", "k", "n"]
    limits = [(1, 16), (17, 32), (33, 64), (65, 128), (129, 256)]
    ah_operations = []
    for dim in dims:
        for lower, upper in limits:
            between_op_fn = functools.partial(
                between_op, dim=dim, lower=lower, upper=upper
            )
            # using 'LEQ' instead of '<=' because '<=' cannot be exported to dot
            between_op_name = f"{lower}LEQ{dim}LEQ{upper}"
            ah_operations.append(
                AHOperation(between_op_name, between_op_fn, is_categorical=True)
            )
    return ah_operations


def pow2_op(data: Any, dim: str, exponent: int) -> bool:
    return data[dim] == 2**exponent


def mm_operations() -> list[AHOperation]:
    mult_dims_ops = get_mult_dims_ops()
    arith_intensity_op = AHOperation("arith_intensity", get_arith_intensity)
    return mult_dims_ops + [arith_intensity_op]


def mixed_mm_operations() -> list[AHOperation]:
    return mm_operations() + between_ops()


def is_multiple(data: Any, dim: str, mult: int) -> bool:
    return data[dim] % mult == 0


def get_dims_multiple_ops() -> list[AHOperation]:
    multiples = [2, 4, 8, 16, 32]
    dims = ["m", "k", "n"]
    dims_multiple_ops = []
    for dim in dims:
        for mult in multiples:
            is_multiple_fn = functools.partial(is_multiple, dim=dim, mult=mult)
            dims_multiple_op = AHOperation(
                f"{dim}_multiple_{mult}", is_multiple_fn, is_categorical=True
            )
            dims_multiple_ops.append(dims_multiple_op)
    return dims_multiple_ops


def get_dims_need_padding_ops() -> list[AHOperation]:
    def mat1_innermost_needs_padding_fn(data: Any) -> bool:
        mat1_stride_0 = data["mat1_stride_0"]
        mat1_stride_1 = data["mat1_stride_1"]
        m_padded_length = data["m_padded_length"]
        k_padded_length = data["k_padded_length"]
        mat1_innermost_needs_padding = False
        if mat1_stride_0 == 1 and m_padded_length != 0:
            mat1_innermost_needs_padding = True
        if mat1_stride_1 == 1 and k_padded_length != 0:
            mat1_innermost_needs_padding = True
        return mat1_innermost_needs_padding

    mat1_innermost_op = AHOperation(
        "mat1_innermost_needs_padding",
        mat1_innermost_needs_padding_fn,
        is_categorical=True,
    )

    def mat2_innermost_needs_padding_fn(data: Any) -> bool:
        mat2_stride_0 = data["mat2_stride_0"]
        mat2_stride_1 = data["mat2_stride_1"]
        k_padded_length = data["k_padded_length"]
        n_padded_length = data["n_padded_length"]
        mat2_innermost_needs_padding = False
        if mat2_stride_0 == 1 and k_padded_length != 0:
            mat2_innermost_needs_padding = True
        if mat2_stride_1 == 1 and n_padded_length != 0:
            mat2_innermost_needs_padding = True
        return mat2_innermost_needs_padding

    mat2_innermost_op = AHOperation(
        "mat2_innermost_needs_padding",
        mat2_innermost_needs_padding_fn,
        is_categorical=True,
    )

    def num_dims_needs_padding_fn(data: Any) -> int:
        m_padded_length = data["m_padded_length"]
        k_padded_length = data["k_padded_length"]
        n_padded_length = data["n_padded_length"]
        num_dims_needs_padding = 0
        if m_padded_length != 0:
            num_dims_needs_padding += 1
        if k_padded_length != 0:
            num_dims_needs_padding += 1
        if n_padded_length != 0:
            num_dims_needs_padding += 1
        return num_dims_needs_padding

    num_dims_op = AHOperation("num_dims_needs_padding", num_dims_needs_padding_fn)
    return [mat1_innermost_op, mat2_innermost_op, num_dims_op]


def get_is_contig_ops() -> list[AHOperation]:
