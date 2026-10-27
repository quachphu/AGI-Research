    return type_hint is comp_type or get_origin(type_hint) is comp_type


def is_optional_type(type_hint) -> bool:  # type: ignore[no-untyped-def]
    """
    Special case of is_type.
    """
    origin = get_origin(type_hint)

    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args

    return False


def is_callable_type(type_hint) -> bool:  # type: ignore[no-untyped-def]
    """
    Special Case of is_type.
    """
    return type_hint.__name__ == "Callable"


class DummyPass(CustomGraphPass):
    """
    A Dummy pass to be used by ConfigFuzzer
    """

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        return None

    def uuid(self) -> Any | None:
        return None


class DummyPartitionerFn(CustomPartitionerFn):
    """
    A Dummy partitioner function to be used by ConfigFuzzer
    """

    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        return min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)

    def uuid(self) -> Any | None:
        return None


T = TypeVar("T")


class TypeExemplars:
    """
    This class returns examples of a Type, given its class name.
    """

    TYPE_EXEMPLARS: dict[str, Any] = {
        CustomGraphPass.__name__: DummyPass(),
        CustomPartitionerFn.__name__: DummyPartitionerFn(),
        torch.fx.graph.Graph.__name__: torch.fx.graph.Graph(),
        BaseSchedulerNode.__name__: BaseSchedulerNode(None),  # type: ignore[arg-type]
    }

    @staticmethod
    def example(t: type[T]) -> T | None:
        """
        Return an example of a class.
        """

        return TypeExemplars.TYPE_EXEMPLARS.get(t.__name__, None)

    @staticmethod
    def contains(t: type[T]) -> bool:
        return t.__name__ in TypeExemplars.TYPE_EXEMPLARS


def check_halide_import() -> bool:
    """checks if we have halide available"""
    try:
        importlib.import_module("halide")
        return True
    except ModuleNotFoundError:
        return False


if check_halide_import():
    CUDA_BACKEND = ["triton", "halide"]
else:
    CUDA_BACKEND = ["triton"]


class Status(Enum):
    """
    The Status return value enum for Config Fuzzer
    """

    # ConfigFuzzer skipped the test
    SKIPPED = "skipped"
    # ConfigFuzzer compiled and ran the test and function it passed.
    PASSED = "passed"
    # ConfigFuzzer failed to compile the test function
    FAILED_COMPILE = "failed_compile"
    # ConfigFuzzer compiled the test function and running it raised an exception
    FAILED_RUN_COMPILE_EXCEPTION = "failed_run_compile_exception"
    # ConfigFuzzer ran eager and it raised an exception
    FAILED_RUN_EAGER_EXCEPTION = "failed_run_eager_exception"
    # ConfigFuzzer compiled the test function, but the return value indicated that the compiled value didn't match the
    # value from eager (or however else you set up the comparison in the test function)
    FAILED_RUN_RETURN = "failed_run_return"

    def failing(self) -> bool:
        """
        Convenience method to check whether these status represent failure.
        """
        return (
            self == Status.FAILED_COMPILE
            or self == Status.FAILED_RUN_EAGER_EXCEPTION
            or self == Status.FAILED_RUN_COMPILE_EXCEPTION
            or self == Status.FAILED_RUN_RETURN
        )


# Sometime the types of configs aren't expressive enough to be captured by python type system, so the options can be
# manually specified here:
# TODO this needs to be indexed to the module, like inductor or dynamo, for name collisions
TYPE_OVERRIDES: dict[str, list[Any]] = {
    "cuda_backend": CUDA_BACKEND,
    "post_grad_fusion_options": [
        {
            "batch_linear_post_grad": {
                "shape_broadcast_batch_linear": True,
                "fuse_nodes_with_same_users": True,
            },
            "batch_aten_mul": {"fuse_nodes_with_same_parent": False},
            "batch_aten_sigmoid": {"fuse_nodes_with_same_parent": True},
            "batch_aten_add": {"fuse_nodes_with_same_parent": True},
            "normalization_aten_pass": {},
            "unbind_stack_aten_pass": {},
        },
        {
            "batch_aten_add": {},
            "batch_aten_mul": {},
            "batch_aten_sub": {},
            "batch_aten_div": {},
            "group_linear": {"require_fbgemm": True},
        },
    ],
    "autoheuristic_collect": [
        {"pad_mm": True, "mixed_mm": True},
        {"pad_mm": True, "mixed_mm": False},
        {"pad_mm": False, "mixed_mm": True},
        {"pad_mm": False, "mixed_mm": False},
    ],
    "autoheuristic_use": [
        {"pad_mm": True, "mixed_mm": True},
        {"pad_mm": True, "mixed_mm": False},
        {"pad_mm": False, "mixed_mm": True},
        {"pad_mm": False, "mixed_mm": False},
    ],
    "traceable_tensor_subclasses": [OrderedSet()],
    "nontraceable_tensor_subclasses": [OrderedSet()],
}
SamplingType = Callable[[str, type[Any], Any], Any]


class SamplingMethod(Enum):
    """
    This class handles the process of assigning concrete values to type annotations. So a type annotation of
    ```python
    foo: Optional[int] = None
    ```
    Will be assigned an int if the dispatch function gets TOGGLE, or a 50/50 split between an int and None if it gets
    RANDOM.
    """

    TOGGLE = "TOGGLE"  # toggle to the opposite value
    RANDOM = "RANDOM"  # randomly choose an option

    @staticmethod
    def _generate_value_for_type(
        random_sample: bool, field_name: str, type_hint: type[Any], default: Any
    ) -> Any:
        """
        Generates a value of a type based on the setting.
        """
        # look for name in type overrides
        if field_name in TYPE_OVERRIDES:
            return random.choice(TYPE_OVERRIDES[field_name])

        if type_hint is bool:
            return random.choice([True, False]) if random_sample else not default
        elif type_hint is int:
            # NOTE initially tried to use negation of the value, but it doesn't work because most types are ints
            # when they should be natural numbers + zero. Python types to cover these values aren't super convenient.
            return random.randint(0, 1000)
        elif type_hint is float:
            return random.uniform(0, 1000)
        elif type_hint is str:
            characters = string.ascii_letters + string.digits + string.punctuation
            return "".join(
                random.choice(characters) for _ in range(random.randint(1, 20))
            )
        elif is_type(type_hint, list):
            elem_type = (
                get_args(type_hint)
                or ([type(default[0])] if default and len(default) else [type(None)])
            )[0]
            new_default = default[0] if default and len(default) > 0 else None
            return [
                SamplingMethod._generate_value_for_type(
                    random_sample, field_name, elem_type, new_default
                )
                for _ in range(random.randint(1, 3))
            ]
        elif is_type(type_hint, set):  # noqa: set_linter
            indexable = list(default)
            elem_type = (
                get_args(type_hint)
                or ([type(indexable[0])] if default and len(default) else [type(None)])
            )[0]
            new_default = indexable[0] if default and len(default) > 0 else None
            return {  # noqa: set_linter
                SamplingMethod._generate_value_for_type(
                    random_sample, field_name, elem_type, new_default
                )
                for _ in range(random.randint(1, 3))
            }
        elif is_type(type_hint, OrderedSet):
            indexable = list(default)
            elem_type = (
                get_args(type_hint)
                or ([type(indexable[0])] if default and len(default) else [type(None)])
            )[0]
            new_default = indexable[0] if default and len(default) > 0 else None
            return OrderedSet(
                [
                    SamplingMethod._generate_value_for_type(
                        random_sample, field_name, elem_type, new_default
                    )
                    for _ in range(random.randint(1, 3))
                ]
            )
        elif is_type(type_hint, dict):
            key_type, value_type = get_args(type_hint) or (
                tuple(map(type, next(iter(default.items()))))
                if (default is not None and len(default))
                else (type(None), type(None))
            )
            if default is not None and len(default.items()) > 0:
                default_key, default_val = next(iter(default.items()))
            else:
                default_key, default_val = None, None
            return {
                SamplingMethod._generate_value_for_type(
                    random_sample, field_name, key_type, default_key
                ): SamplingMethod._generate_value_for_type(
                    random_sample, field_name, value_type, default_val
                )
                for _ in range(random.randint(0, 3))
            }
        elif is_type(type_hint, Union) or is_type(type_hint, types.UnionType):
            # do whatever is not the type of default
            union_args = get_args(type_hint)
            if len(union_args) <= 1:
                raise ValueError("Union type with no args")
            if random_sample:
                new_type = random.choice(union_args)
            else:
                new_type = random.choice(
                    [t for t in union_args if t is not type(default)]
                )
            try:
                new_default = new_type()
            except Exception:
                # if default constructor doesn't work, try None
                new_default = None

            return SamplingMethod._generate_value_for_type(
                random_sample, field_name, new_type, new_default
            )
        elif is_type(type_hint, tuple):
            args = get_args(type_hint) or tuple(map(type, default))
            zipped = zip(args, default)
            return tuple(
                map(  # noqa: C417
                    lambda x: SamplingMethod._generate_value_for_type(
                        random_sample, field_name, x[0], x[1]
                    ),
                    zipped,
                )
            )
        elif is_type(type_hint, Literal):
            literal_args = get_args(type_hint)
            if not literal_args:
                raise ValueError("Literal type with no args")
            if random_sample:
                return random.choice(literal_args)
            else:
                choices = [t for t in literal_args if t != default]
                if choices:
                    return random.choice(choices)
                else:
                    return default
        elif is_optional_type(type_hint):
            optional_args = get_args(type_hint)
            if not optional_args:
                raise ValueError("Optional type with no args")
            elem_type = optional_args[0]
            if random_sample:
                return random.choice(
                    [
                        None,
                        SamplingMethod._generate_value_for_type(
                            random_sample, field_name, elem_type, default
                        ),
                    ]
                )
            else:
                if default is None:
                    return SamplingMethod._generate_value_for_type(
                        random_sample, field_name, elem_type, None
                    )
                else:
                    return None
        elif type_hint is type(None):
            return None
        elif is_callable_type(type_hint):
            callable_args = get_args(type_hint)
            if not callable_args:
                raise ValueError("Callable type with no args")
            return_type = callable_args[-1]

            @wraps(lambda *args, **kwargs: None)
            def dummy_function(*args, **kwargs):  # type: ignore[no-untyped-def]
                return SamplingMethod._generate_value_for_type(
                    random_sample, field_name, return_type, None
                )

            return dummy_function
        elif type_hint == torch._ops.OpOverload:
            return torch.ops.aten.add.default
        elif TypeExemplars.contains(type_hint):
            return TypeExemplars.example(type_hint)
        elif type_hint == Any:
            return 1 if default != 1 else 2
        else:
            raise ValueError(f"Unable to process type {type_hint}. PRs welcome :)")

    @staticmethod
    def dispatch(sm: "SamplingMethod") -> SamplingType:
        """
        Returns a function that will generate values from a type, based on the SamplingMethod passed in.
        """
        if sm == SamplingMethod.RANDOM:
            return partial(SamplingMethod._generate_value_for_type, True)
        elif sm == SamplingMethod.TOGGLE:
            return partial(SamplingMethod._generate_value_for_type, False)
        else:
            raise ValueError(f"malformed sampling method: {sm}")


class Default:
    """
    Singleton default object that will cause the ConfigFuzzer to always use the default value set in the config.
    """


DEFAULT = Default()

# The combination of config settings being set (based on their strings)
ComboType = tuple[str, ...]


class ResultType:
    """
    The mapping of the combo strings to the result status after running the config fuzzer.
    """

    _vals: dict[ComboType, Status]

    def __repr__(self) -> str:
        return f"ResultType[{self._vals}]"

    def __init__(self) -> None:
        self._vals = {}

    def __len__(self) -> int:
        return len(self._vals)

    def num_ran(self) -> int:
        """
        Returns how many combos actually ran (weren't skipped).
        """
        ret = len(self._vals)
        for status in self._vals.values():
            if status == Status.SKIPPED:
                ret -= 1
        return ret

    def set(self, combo: ComboType, status: Status) -> None:
        combo = tuple(sorted(combo))
        self._vals[combo] = status

    def lookup(self, combo: ComboType) -> Status | None:
        combo = tuple(sorted(combo))
        return self._vals.get(combo, None)

    def keys(self) -> KeysView[ComboType]:
        return self._vals.keys()


# Type that maps config strings to their default value
ConfigType = dict[str, Any]
# Callable that returns a bool
FactoryOutputType = Callable[[], bool]
# input function factory
FactoryType = Callable[[], FactoryOutputType]

# Why are some configs disabled by default? Because if we don't the fuzzer produces uninteresting results.
# It will always hone-in on these failures, even with the most basic model, making it useless for
#   debugging more complex models.
#
# More explicit explanations are below:
# Out of Scope: We can't fuzz, say, the cuda version because that comes from the environment and will
#   produce a failure if not aligned with env.
# Known Failure: Disabled due to known failure. Hopefully re-enable. Known failures are listed in the
#   docstring of this file.
# Required: Required for the fuzzer to operate (removing caching, etc.)
# FSDP: Flag meant for FSDP that fails in non FSDP envs. Re-enable these if you're testing FSDP.
# Typing: disabled because the type annotation of the config isn't constrained enough to produce
#   meaningful fuzz values. These could be improved.
# Timing: These take too long to compile, feel free to enable.
MODULE_DEFAULTS: dict[str, ConfigType] = {
    "torch._inductor.config": {
        "force_disable_caches": True,  # Required
        "cpp.cxx": DEFAULT,  # Out of Scope
        "TYPE_CHECKING": DEFAULT,  # Not a config
        "max_autotune_pointwise": DEFAULT,  # Timing
        "max_autotune_gemm": DEFAULT,  # Timing, re-enable when autotune speed improvements merged.
        "max_autotune_gemm_backends": DEFAULT,  # Timing
        "max_autotune_conv_backends": DEFAULT,  # Timing
        "max_autotune_gemm_search_space": DEFAULT,  # Timing
        "max_autotune_subproc_result_timeout_seconds": DEFAULT,  # Timing
        "max_autotune_subproc_graceful_timeout_seconds": DEFAULT,  # Timing
        "max_autotune_subproc_terminate_timeout_seconds": DEFAULT,  # Timing
        "aot_inductor.presets": DEFAULT,  # Typing
        "cuda.arch": DEFAULT,  # Out of Scope
        "cuda.version": DEFAULT,  # Out of Scope
        "cuda.cutlass_dir": DEFAULT,  # Out of Scope
        "cuda.cuda_cxx": DEFAULT,  # Out of Scope
        "rocm.arch": DEFAULT,  # Out of Scope
        "rocm.ck_supported_arch": DEFAULT,  # Out of Scope
        "rocm.ck_dir": DEFAULT,  # Out of Scope
        "rocm.rocm_home": DEFAULT,  # Out of Scope
        "check_stack_no_cycles_TESTING_ONLY": DEFAULT,  # Testing
        "sleep_sec_TESTING_ONLY": DEFAULT,  # Testing
        "triton.inject_relu_bug_TESTING_ONLY": DEFAULT,  # Testing
        "reorder_for_compute_comm_overlap": DEFAULT,  # FSDP
        "enabled_metric_tables": DEFAULT,  # Typing
        "profile_bandwidth_regex": DEFAULT,  # Known Failure
        "disable_cpp_codegen": DEFAULT,  # Known Failure
        "trace.save_real_tensors": DEFAULT,  # Known Failure
        "pre_grad_fusion_options": DEFAULT,  # Typing
        "external_matmul": DEFAULT,  # Typing, need to add this to type overrides or type exemplars.
        "test_configs.autotune_choice_name_regex": DEFAULT,  # Typing
        "test_configs.autotune_choice_desc_regex": DEFAULT,  # Typing
        "cpp.enable_floating_point_contract_flag": DEFAULT,  # Typing
        "post_grad_custom_pre_pass": DEFAULT,  # Typing
        "post_grad_custom_post_pass": DEFAULT,  # Typing
        "reorder_for_compute_comm_overlap_passes": DEFAULT,  # Typing
        "joint_custom_post_pass": DEFAULT,  # Typing
        "joint_custom_pre_pass": DEFAULT,  # Typing
        "pre_grad_custom_pass": DEFAULT,  # Typing
        "custom_partitioner_fn": DEFAULT,  # Typing
        "inductor_choices_class": DEFAULT,  # Typing
        "cudagraph_policy": DEFAULT,  # Typing
    },
    "torch._dynamo.config": {
        "traceable_tensor_subclasses": DEFAULT,  # Typing
        "nontraceable_tensor_subclasses": DEFAULT,  # Typing
        "compiled_autograd_kwargs_override": DEFAULT,  # Typing
        "fail_on_recompile_limit_hit": DEFAULT,  # fails in combo with suppress_errors
        "suppress_errors": DEFAULT,
        "caching_precompile": False,  # Required
    },
}


class ConfigFuzzer:
    """
    This tool makes it easy to search through config state-space with a minimal reproduction or test, either for
      debugging or just bug hunting.
    It has two entry points:
     - bisect, which randomly flips configs and tries to find the minimal reproduction upon failure.
     - fuzz_n_tuple, which tries every combination of n configs. This grows quickly as a function of n, so beware.
    bisect is recommended, but fuzz_n_tuple can give you peace of mind that a new config will compose with
      every other config.

    The main interface is a function factory that will return Callables to be torch.compiled. This function factory
      should return a test function when it's called. Said test function returns a boolean, which determines whether
      the ConfigFuzzer considers it a successful run or not. Throwing an exception from within the function will be
      considered a failure as well.

    # Example usage:

    ```python
    import torch._inductor.config as cfg


    def create_simple_test_model_gpu() -> FactoryOutputType:
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def test_fn() -> bool:
            inp = torch.randn(batch_size, seq_length, hidden_size, device="cuda")
            weight = torch.randn(hidden_size, hidden_size, device="cuda")
            matmul_output = inp @ weight
            final_output = torch.nn.LayerNorm(hidden_size, device="cuda")(matmul_output)
            return True

        return test_fn


    fuzzer = ConfigFuzzer(cfg, create_simple_test_model_gpu, seed=2)

    # Test every pair of configs:
    results = fuzzer.fuzz_n_tuple(n, max_combinations=10000000)

    visualize_results(n, results)

    # Test random configs with bisection:
    ret = fuzzer.bisect(num_attempts=10)

    # reproduce a failing config
    fuzzer.reproduce(
        [{"triton.autotune_pointwise": ..., "coordinate_descent_tuning": ...}]
    )
    ```

    The list of known failures on inductor config are:
    cpp_wrapper, disable_cpp_codegen
    combo_kernels, benchmark_combo_kernel, profile_bandwidth, profile_bandwidth_regex
    trace.enabled, trace.save_real_tensors
    """

    sample: SamplingType
    default: ConfigType

    def __init__(
        self,
        config_module: ConfigModule,
        test_model_fn_factory: FactoryType,
        seed: int,
        default: ConfigType | None = None,
        sm: SamplingMethod = SamplingMethod.TOGGLE,
        test_timeout: int = 3600,
    ):
        """
        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn_factory: Function that returns a test model, which runs and returns True if successful, or
              the outputs if they should be compared with eager
            seed: Randomness seed.
            default: Default values for the config. Inductor has preset based on known failures.
            sm: How type value samples are generated, default TOGGLE.
            test_timeout: max time a test can take.
        """
        self.seed = seed
        self.test_timeout = test_timeout
        self.detailed_results: dict[ComboType, dict[str, Any]] = {}
        self.config_module = config_module
        self.test_model_fn_factory = test_model_fn_factory
        self.fields: dict[str, _ConfigEntry] = self.config_module._config
        self.sample = SamplingMethod.dispatch(sm)

        if default is None:
            if self.config_module.__name__ in MODULE_DEFAULTS:
                self.default = MODULE_DEFAULTS[self.config_module.__name__]
            else:
                raise ValueError("No default passed to ConfigFuzzer.")
        else:
            self.default = default

    def __repr__(self) -> str:
        return (
            f"ConfigFuzzer(config_module={self.config_module}, "
            f"test_model_fn_factory={self.test_model_fn_factory}, seed={self.seed}, default={self.default})"
        )

    def _set_config(self, field_name: str, value: Any) -> None:
        """Set a config value in the module."""
        setattr(self.config_module, field_name, value)

    def _reset_configs(self) -> None:
        """Reset all configs to their default values."""
        for field_name, field_obj in self.fields.items():
            self._set_config(field_name, field_obj.default)

    def new_config(self) -> ConfigType:
        """creates a new config from the default"""
        ret = {
            name: val if val != DEFAULT else self.fields[name].default
            for name, val in self.default.items()
        }
        return ret

    def reproduce(self, configs: Sequence[ConfigType]) -> ResultType:
        """entrypoint to reproduce any failure"""
        results = ResultType()
        for conf in configs:
            self._reproduce_single_helper(conf, results)
        return results

    def _reproduce_single_helper(self, conf: ConfigType, results: ResultType) -> None:
        print(f"Starting repro of {conf}")
        new_config = self.new_config()
        new_config.update(conf)
        self.test_config(results, new_config)
        print(f"Status of {conf}:\n{results.lookup(tuple(conf.keys()))}")

    def reproduce_single(self, config: ConfigType) -> ResultType:
        results = ResultType()
        self._reproduce_single_helper(config, results)
        return results

    def _fuzz_helper(self, results: ResultType, combo: ComboType) -> Status:
        print(combo)
        if st := results.lookup(combo):
            # we already processed this config
            return st

        config = self.new_config()

        skip = False
        for field_name in combo:
            if field_name in config:
                # don't break here because we need to build the config dict
                skip = True
            if field_name.startswith("_"):
                skip = True
            field = self.fields[field_name]
            value = self.sample(field_name, field.value_type, field.default)
            config[field_name] = value
        if skip:
            results.set(combo, Status.SKIPPED)
            return Status.SKIPPED

        return self.test_config(results, config)

    def fuzz_n_tuple(self, n: int, max_combinations: int = 1000) -> ResultType:
        """
        Test every combination of n configs.

        returns a dict of this shape: {(config-1, config-2... config-n): status}
        """
        results = ResultType()
        print(f"Starting {n}-tuple testing with seed {self.seed}")
        random.seed(self.seed)

        for combo in itertools.combinations(self.fields, n):
            st = self._fuzz_helper(results, combo)
            if st != Status.SKIPPED:
                max_combinations -= 1
                if max_combinations <= 0:
                    print("Reached maximum combinations limit")
                    break

        return results

    def save_state(self, filename: str = "fuzzer_state.pkl") -> None:
        """Save the current fuzzer state to a file"""
        with open(filename, "wb") as f:
            pickle.dump(
                {"results": self.results, "detailed_results": self.detailed_results}, f
            )

    def load_state(self, filename: str = "fuzzer_state.pkl") -> None:
        """Load fuzzer state from a file"""
        with open(filename, "rb") as f:
            state = pickle.load(f)
            self.results = state["results"]
            self.detailed_results = state.get("detailed_results", {})

    def timeout_handler(self, signum: int, frame: FrameType | None) -> None:
        raise TimeoutError("Test execution timed out")

    def test_config(self, results: ResultType, config: ConfigType) -> Status:
        """
        Tests a config by calling the function produced by the factory function.
        """
        original_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.test_timeout)
        print(f"Testing config {config}")
        config_tuple = tuple(config.keys())
        if ret := results.lookup(config_tuple):
            signal.signal(signal.SIGALRM, original_handler)
            return ret

        def print_config() -> None:
            for field, value in config.items():
                print(f"{field} = {value}")

        def get_error_info(exc: Exception) -> dict[str, Any]:
            return {
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "config": config.copy(),
            }

        def handle_return(
            message: str,
            return_status: Status,
            print_traceback: bool,
            exc: Exception | None,
        ) -> Status:
            signal.signal(signal.SIGALRM, original_handler)
            print(f"{message} with config combination:")
            print_config()
            if exc:
                self.detailed_results[config_tuple] = get_error_info(exc)
            if print_traceback:
                traceback.print_exc()
            results.set(config_tuple, return_status)
            return return_status

        # reset config
        torch._dynamo.reset()
        self._reset_configs()
        for name, value in config.items():
            self._set_config(name, value)

        # try running eager
        test_model_fn = self.test_model_fn_factory()
        try:
            test_model_fn()
        except Exception as exc:
            return handle_return(
                "Eager exception", Status.FAILED_RUN_EAGER_EXCEPTION, True, exc
            )

        # try compilation
        try:
            test_model_fn2 = self.test_model_fn_factory()
            comp = torch.compile(test_model_fn2, backend="inductor")
        except Exception as exc:
            return handle_return(
                "Exception compiling", Status.FAILED_COMPILE, True, exc
            )

        # try running compiled
        try:
            compile_result = comp()
        except Exception as exc:
            return handle_return(
                "Exception running compiled",
                Status.FAILED_RUN_COMPILE_EXCEPTION,
                True,
                exc,
            )

        # bool return value means don't compare with eager
        if not compile_result:
            return handle_return(
                "Function returned False", Status.FAILED_RUN_RETURN, False, None
            )
        else:
            return handle_return("Function succeeded", Status.PASSED, False, None)

    def bisect(self, num_attempts: int = 100, p: float = 0.5) -> list[ConfigType]:
        """
        Test configs and bisect to minimal failing configuration.
        """
        print(f"Starting random testing with bisection, seed {self.seed}, and p {p}")
        random.seed(self.seed)
        self._reset_configs()
        results = ResultType()
        ret: list[ConfigType] = []

        for attempt in range(num_attempts):
            print(f"Random attempt {attempt + 1}/{num_attempts}")

            config = self.new_config()

            for field_name, config_entry in self.fields.items():
                if (
                    field_name not in config
                    and not field_name.startswith("_")
                    and "TESTING_ONLY" not in field_name
                    and random.random() < p
                ):
                    value = self.sample(
                        field_name, config_entry.value_type, config_entry.default
                    )
                    config[field_name] = value

            status = self.test_config(results, config)
            if status not in OrderedSet([Status.PASSED, Status.SKIPPED]):
                if minimal_failing_config := self._bisect_failing_config(
                    results, config
                ):
                    print(f"Minimum failing config: {minimal_failing_config}")
                    ret.append(minimal_failing_config)

        return ret

    def _bisect_failing_config(
        self, results: ResultType, failing_config: ConfigType
    ) -> ConfigType | None:
        return self._bisect_failing_config_helper(results, list(failing_config.items()))

    def _bisect_failing_config_helper(
        self, results: ResultType, failing_config: list[tuple[str, Any]]
    ) -> ConfigType | None:
        """
        Bisect a failing configuration to find minimal set of configs that cause failure.

        Splits it into halves, then fourths, then tries dropping configs one-by-one.
        """
        print(f"bisecting config: {failing_config}")

        if not failing_config:
            return None

        def test(x: list[tuple[str, Any]]) -> Status:
            d = dict(x)
            result = self.test_config(results, d)
            return result

        if len(failing_config) <= 1:
            return dict(failing_config) if test(failing_config).failing() else None

        random.shuffle(failing_config)

        mid = len(failing_config) // 2
        first_half = failing_config[:mid]
        second_half = failing_config[mid:]
        if test(first_half).failing():
            return self._bisect_failing_config_helper(results, first_half)
        if test(second_half).failing():
            return self._bisect_failing_config_helper(results, second_half)

        if len(failing_config) >= 8:
            low = len(failing_config) // 4
            high = mid + low
            quart1 = failing_config[low:]
            if test(quart1).failing():
                return self._bisect_failing_config_helper(results, quart1)
            quart2 = failing_config[:low] + second_half
            if test(quart2).failing():
                return self._bisect_failing_config_helper(results, quart2)
            quart3 = first_half + failing_config[:high]
            if test(quart3).failing():
                return self._bisect_failing_config_helper(results, quart3)
            quart4 = failing_config[high:]
            if test(quart4).failing():
                return self._bisect_failing_config_helper(results, quart4)
        # try dropping one value at a time
        for i in range(len(failing_config)):
            new_list = [x for j, x in enumerate(failing_config) if j != i]
            if test(new_list).failing():
                return self._bisect_failing_config_helper(results, new_list)
        # we have the minimal set
        return dict(failing_config)


def visualize_results(
    n: int, results: ResultType, filename: str = "results.html"
) -> None:
    """
    Creates an HTML document representing the results of running the fuzzer with fuzz_n_tuple, with n = 2.
    """
    # TODO support more dimensions
    if n != 2:
        raise AssertionError(f"expected n == 2, got {n}")
    if len(results) <= 0:
        raise AssertionError("expected non-empty results")

    input_set: OrderedSet[str] = OrderedSet({})
    for key in results.keys():  # noqa: SIM118
        input_set.add(key[0])
        input_set.add(key[1])
    input_list = sorted(input_set)

    # Start the HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title> Fuzzer Visualization</title>
        <style>
            table {
                border-collapse: collapse;
                width: 50%;
                margin: 20px auto;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f2f2f2;
            }
            .skipped {
                background-color: yellow;
            }
            .passed {
                background-color: green;
                color: white;
            }
            .failed {
                background-color: red;
                color: white;
            }
        </style>
    </head>
    <body>
        <h2 style="text-align: center;">Fuzzer Visualization</h2>
        <table>
        <thead>
    """

    html_content += "<tr><th>\\</th>"
    for col_name in input_list:
        col = "<br>".join(col_name)
        html_content += f"<th>{col}</th>"
    html_content += "</tr></thead><tbody>"

    # Add table rows
    for row_name in input_list:
        html_content += f"<tr><th>{row_name}</th>"
        for col_name in input_list:
            # Determine the status class for the cell
            status_enum = results.lookup((row_name, col_name))
            status_class = ""
            status_val = ""
            if status_enum == Status.SKIPPED:
                status_class = "skipped"
                status_val = "-"
            elif status_enum == Status.PASSED:
                status_class = "passed"
                status_val = "O"
            elif status_enum == Status.FAILED_RUN_EAGER_EXCEPTION:
                status_class = "failed"
                status_val = "e"
            elif status_enum == Status.FAILED_RUN_COMPILE_EXCEPTION:
                status_class = "failed"
                status_val = "E"
            elif status_enum == Status.FAILED_RUN_RETURN:
                status_class = "failed"
                status_val = "R"
            elif status_enum == Status.FAILED_COMPILE:
                status_class = "failed"
                status_val = "C"
            else:
                status_class = "skipped"
                status_val = "-"

            html_content += f'<td class="{status_class}">{status_val}</td>'
        html_content += "</tr>"

    html_content += """
        </tbody>
        </table>
    </body>
    </html>
    """

    with open(filename, "w") as file:
        file.write(html_content)
<|endoftext|>import logging
from typing import Any

import torch
from torch._inductor.kernel.mm_common import mm_args

from . import config
from .codegen.cpp_gemm_template import CppGemmTemplate, CppWoqInt4GemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .mkldnn_ir import WeightInt4PackMatmul
from .select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template
from .virtualized import V


log = logging.getLogger(__name__)

aten__weight_int8pack_mm = ExternKernelChoice(
    torch._weight_int8pack_mm, "at::_weight_int8pack_mm", has_out_variant=False
)

aten__weight_int4pack_mm_cpu = ExternKernelChoice(
    torch.ops.quantized.int4mm_packed_weight_cpu,
    "at::native::_weight_int4pack_mm_cpu_tensor",
    has_out_variant=False,
    kernel_creator=WeightInt4PackMatmul.create,
)

quantized = torch.ops.quantized
_quantized = torch.ops._quantized
aten = torch.ops.aten


def register_quantized_ops() -> None:
    from . import lowering

    lowering.add_needs_realized_inputs(
        [
            quantized.max_pool2d,
            _quantized.wrapped_fbgemm_pack_gemm_matrix_fp16,
            _quantized.wrapped_fbgemm_linear_fp16_weight,
        ]
    )
    lowering.make_fallback(quantized.max_pool2d)
    lowering.make_fallback(_quantized.wrapped_fbgemm_pack_gemm_matrix_fp16)
    lowering.make_fallback(_quantized.wrapped_fbgemm_linear_fp16_weight)


def register_woq_mm_ops() -> None:
    """Register weight-only quantized matmul lowerings."""
    from . import lowering
    from .lowering import expand, register_lowering

    @register_lowering(aten._weight_int8pack_mm, type_promotion_kind=None)  # type: ignore[misc]
    def int8pack_mm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        layout: Any = None,
    ) -> Any:
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, mat2_transposed=True
        )
        if not (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.int8
        ):
            raise AssertionError(
                f"expected mat1 dtype in [bfloat16, float16, float] and mat2 dtype "
                f"int8, got {mat1.get_dtype()} and {mat2.get_dtype()}"
            )
        aten_layout = layout

        # options to tune from
        choices = (
            [aten__weight_int8pack_mm.bind((mat1, mat2, scale), aten_layout)]
            if use_aten_gemm_kernels()
            else []
        )

        # scale is applied as an epilogue, and the scale tensor is expanded (with a view op)
        # for broadcasting, as it's 1D.
        def _mul_epilogue(buf: torch.Tensor) -> Any:
            return create_epilogue_with_attr(
                buf, "mul", other=realize_inputs(expand(scale, layout.size))
            )

        if use_cpp_gemm_template(aten_layout, mat1, mat2, mat2_transposed=True):
            CppGemmTemplate.add_choices(
                choices,
                aten_layout,
                [mat1, mat2, scale],
                trans_w=True,
                epilogue_creator=_mul_epilogue,  # type: ignore[arg-type]
            )

        node, _ = autotune_select_algorithm(
            "_weight_int8pack_mm", choices, [mat1, mat2, scale], aten_layout
        )
        return node

    @register_lowering(aten._weight_int4pack_mm_for_cpu, type_promotion_kind=None)  # type: ignore[misc]
    def int4pack_mm_cpu(
        input: torch.Tensor,
        weight: torch.Tensor,
        qGroupSize: int,
        qScaleAndZeros: torch.Tensor,
        *,
        layout: Any = None,
    ) -> Any:
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, use_4x2_dim=True, mat2_transposed=True
        )
        if not (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.uint8
        ):
            raise AssertionError(
                f"expected mat1 dtype in [bfloat16, float16, float] and mat2 dtype "
                f"uint8, got {mat1.get_dtype()} and {mat2.get_dtype()}"
            )
        group_size = V.graph.add_tensor_constant(
            torch.tensor(qGroupSize, dtype=torch.int64), name=None
        )
        aten_layout = layout

        # options to tune from
        choices = (
            [
                aten__weight_int4pack_mm_cpu.bind(
                    (mat1, mat2, group_size, qScaleAndZeros), aten_layout
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )
        if (
            (config.max_autotune or config.max_autotune_gemm)
            and use_cpp_gemm_template(
                aten_layout,
                mat1,
                mat2,
                mat2_transposed=True,
                is_woq_int4=True,
                q_group_size=qGroupSize,
            )
            and mat2.get_layout().is_contiguous()
        ):
            # pyrefly: ignore [bad-specialization, missing-attribute, not-a-type]
            CppWoqInt4GemmTemplate[qGroupSize].add_choices(
                choices,
                aten_layout,
                [mat1, mat2, group_size, qScaleAndZeros],
            )

        # define functions to generate example inputs for weight and group size
        # otherwise, autotuner generates example inputs of all zeros for them
        def get_example_weight(x: torch._inductor.ir.IRNode) -> torch.Tensor:
            if not x.get_layout().is_contiguous():
                raise AssertionError("expected x to have a contiguous layout")
            shape = x.get_size()
            device = x.get_device()
            return torch.randint(0, 255, shape, dtype=torch.uint8, device=device)

        input_gen_fns = {
            1: get_example_weight,  # packed weight
            2: lambda x: V.graph.constants[x.get_name()],  # group size
        }

        node, _ = autotune_select_algorithm(
            "_weight_int4pack_mm_for_cpu",
            choices,
            [mat1, mat2, group_size, qScaleAndZeros],
            aten_layout,
            input_gen_fns=input_gen_fns,
        )
        return node

    lowering.make_fallback(aten._dyn_quant_matmul_4bit)
    lowering.make_fallback(aten._dyn_quant_pack_4bit_weight)
<|endoftext|>import logging
from collections import defaultdict

import torch
import torch.fx as fx
from torch._inductor.fx_passes.utils import BitsetAncestors
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


class AugmentedGraphHelper:
    """
    Graph helper that augments the original graph with additional
    dependencies and uses, plus tracks node equivalences for coalescing.

    TODO: if this becomes too large of compile time, consider binding
    graphcycles.cc
    """

    def __init__(
        self,
        graph: fx.Graph,
        node_ancestors: BitsetAncestors | None = None,
    ):
        # Each node starts in its own singleton set
        self.graph = graph
        self.merge_sets = {node: OrderedSet([node]) for node in graph.nodes}

        # Extra dependencies: node depends on dep (dep must come before node)
        self.extra_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        # Extra uses: reverse of extra_deps (node is used by user)
        self.extra_uses: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        # Note: only reflect original ancestors, not maintained through additional deps
        # or merge sets
        self.node_ancestors = node_ancestors

    def add_extra_dep(self, *, n: fx.Node, dep: fx.Node) -> None:
        """Add extra dependency: node depends on dep."""
        self.extra_deps[n].add(dep)
        self.extra_uses[dep].add(n)

    def remove_extra_dep(self, *, n: fx.Node, dep: fx.Node) -> None:
        if dep in self.extra_deps[n]:
            self.extra_deps[n].discard(dep)
            self.extra_uses[dep].discard(n)

    def merge_to_set(self, existing_node: fx.Node, new_node: fx.Node) -> None:
        """
        Merge new_node into existing_node's set. The new node must be a singleton set.
        """
        existing_set = self.merge_sets[existing_node]
        new_set = self.merge_sets[new_node]
        if len(new_set) != 1:
            raise AssertionError(
                f"new_node must be a singleton set, got size {len(new_set)}"
            )

        # Add all nodes from new_set to existing_set
        existing_set.update(new_set)

        # Update all nodes from new_set to point to existing_set
        for node in new_set:
            self.merge_sets[node] = existing_set

    def unmerge_node(self, node: fx.Node) -> None:
        """Remove a node from its merge set, making it singleton."""
        old_set = self.merge_sets[node]

        # If already singleton, nothing to do
        if len(old_set) == 1:
            return

        # Remove from old set
        old_set.remove(node)

        # Make node singleton
        self.merge_sets[node] = OrderedSet([node])

    def get_merged_deps(self, node: fx.Node) -> OrderedSet[fx.Node]:
        """
        Get all dependencies of a node considering merges and extra deps.
        Combines:
        1. Direct deps (all_input_nodes) of node and its merge equivalents
        2. Extra deps of node and its merge equivalents
        """
        deps: OrderedSet[fx.Node] = OrderedSet()

        # For each node in the merge set
        for merged_node in self.merge_sets[node]:
            # Add direct dependencies from all_input_nodes
            deps.update(merged_node.all_input_nodes)
            # Add extra dependencies
            deps.update(self.extra_deps[merged_node])

        return deps

    def has_cycle(self) -> bool:
        return torch._dynamo.graph_deduplication._has_cycle(
            self.graph, self.get_all_extra_deps()
        )

    def _get_all_ancestors(self, node: fx.Node) -> OrderedSet[fx.Node]:
        """Transitive ancestors through both data deps and extra deps."""
        ancestors: OrderedSet[fx.Node] = OrderedSet()
        stack: list[fx.Node] = list(node.all_input_nodes)
        stack.extend(self.extra_deps.get(node, ()))
        while stack:
            n = stack.pop()
            if n not in ancestors:
                ancestors.add(n)
                stack.extend(n.all_input_nodes)
                stack.extend(self.extra_deps.get(n, ()))
        return ancestors

    def has_path(self, source: fx.Node, target: fx.Node) -> bool:
        """Check if there's a path from source to target."""
        # we should not be checking path from node to itself
        if self.merge_sets[source] is self.merge_sets[target]:
            raise AssertionError("source and target must not be in the same merge set")

        # search backwards from target to source
        visited: OrderedSet[fx.Node] = OrderedSet()
        queue = [target]
        visited.add(target)

        while queue:
            current = queue.pop()

            for dep in self.get_merged_deps(current):
                # Check if we reached source or its equivalent
                if dep in self.merge_sets[source]:
                    return True

                if dep in visited:
                    continue

                # We are searching from target, so this node is necessarily an ancestor
                # of target.
                # If dep is an ancestor of source, any path through dep to source would imply a cycle
                if self.node_ancestors:
                    source_set = self.merge_sets[source]
                    is_ancestor_of_source = any(
                        self.node_ancestors.is_ancestor(dep, s) for s in source_set
                    )
                    # Add to visited to avoid recomputing this check if we see dep again
                    if is_ancestor_of_source:
                        visited.add(dep)
                        continue

                visited.add(dep)
                queue.append(dep)

        return False

    def transfer_erased_node_deps(
        self, erased_to_new: dict[fx.Node, fx.Node | None]
    ) -> None:
        """
        Transfer all extra dependencies from erased nodes to their replacements, handling
        cross-dependencies between erased nodes correctly.

        Skips deps where both endpoints resolve to replacement nodes from the
        same erasure batch — these are intra-bucket deps that would create
        cycles (e.g. new_start <-> new_wait within the same bucket).
        """
        erased_merge_sets: dict[fx.Node, fx.Node | None] = {}

        for replaced, new in erased_to_new.items():
            for equiv in self.merge_sets[replaced]:
                erased_merge_sets[equiv] = new

        # Transfer dependencies
        for old_node, new_node in erased_merge_sets.items():
            if new_node is None:
                # Clean up references to removed node
                for extra_use in list(self.extra_uses[old_node]):
                    updated_use = erased_merge_sets.get(extra_use, extra_use)
                    if updated_use is not None:
                        self.extra_deps[updated_use].discard(old_node)
                for extra_dep in list(self.extra_deps[old_node]):
                    updated_dep = erased_merge_sets.get(extra_dep, extra_dep)
                    if updated_dep is not None:
                        self.extra_uses[updated_dep].discard(old_node)
            else:
                # Transfer dependencies FROM old_node (what old_node depended on)
                for extra_dep in self.extra_deps[old_node]:
                    updated_dep = erased_merge_sets.get(extra_dep, extra_dep)
                    if updated_dep is not None and updated_dep != new_node:
                        # Skip if reverse dep already exists (extra or data)
                        if new_node in self.extra_deps.get(
                            updated_dep, ()
                        ) or new_node in OrderedSet(updated_dep.all_input_nodes):
                            continue
                        self.extra_deps[new_node].add(updated_dep)
                        self.extra_uses[updated_dep].discard(old_node)
                        self.extra_uses[updated_dep].add(new_node)

                # Transfer dependencies TO old_node (what depended on old_node)
                for extra_use in self.extra_uses[old_node]:
                    updated_use = erased_merge_sets.get(extra_use, extra_use)
                    if updated_use is not None and updated_use != new_node:
                        # Skip if reverse dep already exists (extra or data)
                        if updated_use in self.extra_deps.get(
                            new_node, ()
                        ) or updated_use in OrderedSet(new_node.all_input_nodes):
                            continue
                        self.extra_deps[updated_use].discard(old_node)
                        self.extra_deps[updated_use].add(new_node)
                        self.extra_uses[new_node].add(updated_use)

        # Clean up erased nodes
        for old_node in erased_merge_sets:
            self.extra_deps[old_node].clear()
            self.extra_uses[old_node].clear()
            del self.merge_sets[old_node]

    def remove_erased_extra_deps(self) -> None:
        """Remove extra deps referencing erased nodes."""
        for node in list(self.extra_deps):
            if node._erased:
                for dep in list(self.extra_deps[node]):
                    self.remove_extra_dep(n=node, dep=dep)
                continue
            for dep in list(self.extra_deps[node]):
                if dep._erased:
                    self.remove_extra_dep(n=node, dep=dep)

    def check_and_maybe_autofix_cyclic_extra_deps(
        self, *, autofix: bool = False
    ) -> None:
        """Check for and optionally remove extra deps that create cycles.

        Args:
            autofix: If True, silently remove cyclic deps.  If False (default),
                raise an error so the root cause gets investigated.
        """
        if not self.has_cycle():
            return
        removed = []
        for node in list(self.extra_deps):
            for dep in list(self.extra_deps[node]):
                ancestors = self._get_all_ancestors(dep)
                if node in ancestors:
                    removed.append((node.name, dep.name))
                    self.remove_extra_dep(n=node, dep=dep)
        if not removed:
            return
        msg = (
            f"Overlap scheduling: detected {len(removed)} cyclic extra "
            f"dep(s): {removed}. Please report this to the overlap "
            f"scheduling developers."
        )
        log.warning(msg)
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_overlap_cyclic_extra_deps",
                "encoding": "string",
            },
            payload_fn=lambda: msg,
        )
        if not autofix:
            raise RuntimeError(
                f"{msg}\nTo unblock, set "
                f"torch._inductor.config.aten_distributed_optimizations"
                f".overlap_scheduling_autofix_cycles = True"
            )

    def get_all_extra_deps(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """
        Get all extra dependencies in a format suitable for topological sort.
        Returns a copy to avoid external modifications.
        """
        return {
            node: OrderedSet(deps)
            for node, deps in self.extra_deps.items()
            if deps  # Only include nodes with non-empty deps
        }
<|endoftext|>import dataclasses
import itertools
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload, TYPE_CHECKING, TypeVar, Union

import sympy

import torch
from torch._inductor.dependencies import index_vars_no_squeeze, ReadWrites
from torch._inductor.utils import sympy_product, sympy_subs
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Identity
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .virtualized import V


T = TypeVar("T")
U = TypeVar("U")


Split = tuple[sympy.Expr, ...]
VarsAndRanges = tuple[list[sympy.Symbol], list[sympy.Expr]]


loop_tiling_log = torch._logging.getArtifactLogger(__name__, "loop_tiling")
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


if TYPE_CHECKING:
    from torch._inductor.scheduler import (
        BaseSchedulerNode,
        FusedSchedulerNode,
        SchedulerNode,
    )


def solve_for_zero(expr: sympy.Expr) -> sympy.Expr | None:
    """
    Given an expr with a single free symbol, solve for a constant relation that would make
    this expression 0.
    """
    if expr.is_constant():
        return None
    elif isinstance(expr, FloorDiv):
        return None

    if len(expr.free_symbols) != 1:
        raise AssertionError(
            f"expected exactly 1 free symbol, got {len(expr.free_symbols)}"
        )
    free_symbol = next(iter(expr.free_symbols))
    if isinstance(expr, ModularIndexing):
        out = try_solve(sympy.Eq(expr.args[0], expr.args[2]), free_symbol)
    else:
        out = try_solve(sympy.Eq(expr, 0), free_symbol)
    if not out or not out[1].is_constant():
        return None
    return out[1]


def solve_for_tiling(expr: sympy.Expr) -> sympy.Expr | None:
    """
    Giving an expr with a single free symbol, try to find a tiling that would
    make the expression coalesced with respect to that symbol.

    Tiling an expression `x` by `y` means that the expression will now be indexed
    by both the original (x) and by (x * y). So we are looking for a
    multiplicative factor that will make ((x + 1) * y) - (x * y) == 1.

    To simplify things for sympy, we'll try just x * y == 1, check x(1) and x(0).
    """

    if len(expr.free_symbols) != 1:
        return None

    free_symbol = next(iter(expr.free_symbols))

    def _solve_simple_expr(expr: sympy.Expr) -> sympy.Expr | None:
        if expr.has(ModularIndexing) or expr.has(FloorDiv):
            # the div approximation could not eliminate all ModularIndexing /
            # FloorDiv nodes; we cannot solve this expression
            return None
        if len(expr.free_symbols) != 1:
            return None

        out = try_solve(sympy.Eq(expr, 1), free_symbol)
        if not out or not out[1].is_constant():
            return None
        return out[1]

    # Sympy solving is very limited with ModularIndexing and FloorDiv,
    # but good otherwise.
    if not expr.has(ModularIndexing) and not expr.has(FloorDiv):
        return _solve_simple_expr(expr)

    required_values = []
    eq_1_expressions = []

    # very piecemeal solution if ModularIndexing or FloorDiv involved.
    # Look for terms we'll try to make 0, and then other terms we'll try to make 1.
    # Expand as needed.
    for arg in sympy.Add.make_args(expr):
        # Try to make mul terms 0
        if isinstance(arg, sympy.Mul):
            seen = False
            # TODO - only need one of these to be solvable to zero
            for mul_arg in arg.args:
                out = solve_for_zero(mul_arg)
                if out is None:
                    continue

                if not out.is_constant():
                    raise AssertionError(f"expected constant, got {out}")
                seen = True
                required_values.append(out)

            if not seen:
                return None
        else:
            eq_1_expressions.append(arg)

    if not eq_1_expressions:
        return None

    eq_1_expr = sum(eq_1_expressions)

    def indexing_div_rep(
        x: sympy.Expr,
        y: sympy.Expr,
        z: sympy.Expr | None = None,
    ) -> sympy.Expr:
        return x / y

    # For the purposes of tiling/coalesced access, approximate ModularIndexing and FloorDiv
    # then check later. simultaneous=False rebuilds bottom up, collapsing nested
    # occurrences in one pass; the second sweep catches nodes that FloorDiv eval
    # reintroduces while rebuilding. Leftovers make _solve_simple_expr bail out.
    eq_1_expr_simplified = eq_1_expr
    for _ in range(2):
        # pyrefly: ignore [missing-attribute]
        eq_1_expr_simplified = eq_1_expr_simplified.replace(
            ModularIndexing, indexing_div_rep, simultaneous=False
        ).replace(FloorDiv, indexing_div_rep, simultaneous=False)

    out = _solve_simple_expr(eq_1_expr_simplified)

    # since we approximated FloorDiv/ModularIndexing, double check here
    if not out or sympy_subs(eq_1_expr, {free_symbol: out}) != 1:
        return None

    required_values.append(out)

    if len(OrderedSet(required_values)) == 1:
        return required_values[0]

    return None


def find_broadcast_var(
    index: sympy.Expr, var_ranges: dict[sympy.Expr, int]
) -> sympy.Expr | None:
    """
    Try to find the variable that this index is broadcast over.
    A broadcast pattern is one where consecutive values of a variable
    access the same memory location (e.g., x // 10).
    """
    # Approximate analysis by evaluating at 1 and 0
    variables: dict[sympy.Symbol, int] = {}
    for v in index.free_symbols:
        if v in var_ranges:
            variables[v] = 0
        else:
            variables[v] = get_hint(v)

    zero_index = sympy_subs(index, variables)
    for v in var_ranges:
        if v not in index.free_symbols:
            continue

        variables[v] = 1
        try:
            new_val = sympy_subs(index, variables)
        except ZeroDivisionError:
            loop_tiling_log.info("zero division error %s %s", index, variables)
            continue
        # Broadcast means the value doesn't change when the variable increments
        if new_val == zero_index:
            return v
        variables[v] = 0

    return None


def find_coalesced_var(
    index: sympy.Expr, var_ranges: dict[sympy.Expr, int]
) -> sympy.Expr | None:
    """
    Try to find the symbol which coalesces this index
    """
    top_level_terms = sympy.Add.make_args(index)
    for v in var_ranges:
        if v in top_level_terms:
            return v

    # Approximate analysis by evaluating at 1 and 0
    variables: dict[sympy.Symbol, int] = {}
    for v in index.free_symbols:
        if v in var_ranges:
            variables[v] = 0
        else:
            variables[v] = get_hint(v)

    zero_index = sympy_subs(index, variables)
    for v in var_ranges:
        variables[v] = 1
        try:
            new_val = sympy_subs(index, variables)
        except ZeroDivisionError:
            loop_tiling_log.info("zero division error %s %s", index, variables)
            continue
        if new_val - zero_index == 1:
            variables[v] = 2
            # in some more complex expressions, 0->1 will be coalesced,
            # but not 1->2
            if (sympy_subs(index, variables) - new_val) == 1:
                return v
        variables[v] = 0

    return None


def has_indirect_access(memory_expr: sympy.Expr) -> bool:
    """
    Check if this memory expression has any indirect indexing.
    """
    return any(symbol_is_type(s, SymT.INDIRECT) for s in memory_expr.free_symbols)


@dataclasses.dataclass(frozen=True)
class FusedNormalizedReadsWrites:
    """
    Normalized reads and writes for nodes in the same FusedSchedulerNode.
    """

    index_vars: OrderedSet[sympy.Symbol]
    reduce_vars: OrderedSet[sympy.Symbol]
    reads: dict[sympy.Expr, OrderedSet[str]]
    writes: dict[sympy.Expr, OrderedSet[str]]
    var_ranges: dict[sympy.Symbol, int]


@dataclasses.dataclass(frozen=True)
class _FusedNodeView:
    nodes: Sequence["BaseSchedulerNode"]
    read_writes: ReadWrites
    group: Any

    def get_nodes(self) -> Sequence["BaseSchedulerNode"]:
        return self.nodes

    def get_buffer_names(self) -> OrderedSet[str]:
        return OrderedSet.union(*(node.get_buffer_names() for node in self.nodes))

    def get_operation_names(self) -> OrderedSet[str]:
        return OrderedSet(node.get_name() for node in self.nodes)


@overload
def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: Literal[True],
) -> tuple[VarsAndRanges, VarsAndRanges] | None: ...


@overload
def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: Literal[False] = False,
) -> tuple[VarsAndRanges, VarsAndRanges]: ...


def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: bool = False,
) -> tuple[VarsAndRanges, VarsAndRanges] | None:
    # nb: use statically_known_equals here to mimic scheduler.
    # TODO : store type of split/broadcast on fused node itself,
    # instead of re-deriving it.
    if n.is_reduction() or V.graph.sizevars.statically_known_equals(
        sympy_product(n._body.sizes[0]), pointwise_numel
    ):
        # pyrefly: ignore [bad-return]
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )  # type: ignore[return-value]

    if get_hint(sympy_product(n._body.sizes[0])) != get_hint(
        pointwise_numel * red_numel  # type: ignore[operator]
    ):
        raise AssertionError(
            "expected pointwise sizes to match pointwise_numel * red_numel"
        )
    i = len(n._body.sizes[0]) - 1
    prod = 1
    while i >= 0:
        prod *= n._body.sizes[0][i]
        if prod == red_numel:
            break
        i -= 1

    if i >= 0:
        pw_splits = n._body.sizes[0][0:i]
        iter_vars = n._body.iter_vars[0:i]

        red_splits = n._body.sizes[0][i:]
        red_vars = n._body.iter_vars[i:]
        return (iter_vars, pw_splits), (red_vars, red_splits)  # type: ignore[return-value]

    if none_if_not_divisible:
        return None
    else:
        # pyrefly: ignore [bad-return]
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )  # type: ignore[return-value]


class NodeSplitGetter:
    """
    Finds a Pointwise, Reduction Split that compatible with all nodes in a SchedulerNode.
    """

    def __init__(
        self,
        node: Union["_FusedNodeView", "FusedSchedulerNode", "SchedulerNode"],
    ):
        self.node = node
        self.pointwise_numel: sympy.Expr = node.group[1][0]
        self.red_numel: sympy.Expr = node.group[1][1]

        self.pw_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)
        self.red_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)

        self.reduction_split: Split = ()
        self.all_node_sizes: OrderedSet[tuple[Split, Split]] = OrderedSet()

        fused_group = node.group[1]
        for n in reversed(node.get_nodes()):
            if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
                continue

            # if we can't split the pw ranges into a (pw, red) split,
            # don't add as a split option, but do make sure we check that this size
            # is splittable
            maybe_splits = get_pw_red_splits(
                n, self.pointwise_numel, self.red_numel, none_if_not_divisible=True
            )
            if maybe_splits is None:
                self.all_node_sizes.add(n._body.sizes)
                continue

            (_, n_pw_splits), (_, n_red_splits) = maybe_splits

            # fill in reduction size
            n_pw_splits, n_red_splits = (
                torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(
                    fused_group, (n_pw_splits, n_red_splits), self.red_numel
                )
            )

            self.pw_split_options[len(n_pw_splits)].add(tuple(n_pw_splits))
            self.red_split_options[len(n_red_splits)].add(tuple(n_red_splits))

            if n_red_splits != ():
                self.reduction_split = (sympy_product(n_red_splits),)

            n_size = (tuple(n_pw_splits), tuple(n_red_splits))
            self.all_node_sizes.add(n_size)

        self.seen_pw_splits: OrderedSet[Split] = OrderedSet()

    def get_node_splits(self) -> tuple[Split, Split]:
        """
        Get a compatible pointwise, reduction split of the node
        """

        if len(self.all_node_sizes) == 1:
            return next(iter(self.all_node_sizes))

        if len(self.pw_split_options) == 0:
            return ((self.pointwise_numel,), (self.red_numel,))

        max_pw_split = max(self.pw_split_options.keys())
        max_red_split = max(self.red_split_options.keys())

        def add_combined_split_options(
            split_options: dict[int, OrderedSet[Split]], curr_length: int
        ) -> None:
            for split in split_options[curr_length]:
                for i in range(len(split) - 1):
                    new_split = tuple(
                        split[0:i] + (sympy_product(split[i : i + 2]),) + split[i + 2 :]
                    )
                    split_options[len(new_split)].add(new_split)

        max_total_splits = max_pw_split + max_red_split
        for curr_iter, total_splits in enumerate(range(max_total_splits, 0, -1)):
            for pw_split_len in range(total_splits, 0, -1):
                for pw_split in self.pw_split_options[pw_split_len]:
                    for red_split in self.red_split_options[
                        total_splits - pw_split_len
                    ]:
                        if out := self.try_split(pw_split, red_split):
                            return out

            add_combined_split_options(self.pw_split_options, max_pw_split - curr_iter)
            add_combined_split_options(
                self.red_split_options, max_red_split - curr_iter
            )

        # if for whatever reason we couldn't split above, return default split
        return ((self.pointwise_numel,), (self.red_numel,))

    def try_split(self, pw: Split, red: Split) -> tuple[Split, Split] | None:
        """
        See if this split is compatible, and potentially returning a longer split
        than the input.
        """

        from torch._inductor.codegen.simd import CantSplit, SIMDKernel

        if pw in self.seen_pw_splits:
            return None
        self.seen_pw_splits.add(pw)

        for n_pw, n_red in self.all_node_sizes:
            try:
                groups = pw + red
                lengths = (n_pw, n_red)
                splits, getters = SIMDKernel._split_iteration_ranges(groups, lengths)
            except CantSplit:
                return None

            if len(getters) != 2:
                raise AssertionError(f"expected 2 getters, got {len(getters)}")
            pw_group_splits = splits[: len(pw)]
            # if we had to divide a variable into two to do this split,
            # then lets try the larger, induced split.
            # e.g. splitting (12, 2) into (2, 12) will split the first var into:
            # (2, 6) and produce an overall split of (2, 6, 2)
            flattened_pw_splits = tuple(itertools.chain.from_iterable(pw_group_splits))
            if flattened_pw_splits != pw:
                if out := self.try_split(flattened_pw_splits, red):
                    return out

        return pw, red


def apply_var_mapping(
    iter_vars: list[sympy.Symbol],
    red_vars: list[sympy.Symbol],
    norm_pw_vars: list[sympy.Symbol],
    norm_red_vars: list[sympy.Symbol],
    new_ranges: list[list[sympy.Expr]],
    return_getters_groups: list[list[Callable[[list[sympy.Expr]], sympy.Expr]]],
) -> dict[sympy.Symbol, sympy.Expr]:
    """Maps original variables to expressions using normalized variables."""

    # the output of split_iteration_range is a new_ranges, return_getters_groups
    # new_ranges is a flattened list of ranges corresponding to the new pw and red vars
    # for example, taking in pw vars of range (6, 6) to normalized range [36],
    # new_ranges would be [[6, 6]]
    # There is a return_getter callable for each input iter_var and red_vars.
    # if you flatten out all of the ranges, and create a variable for each index,
    # then applying the flattening vars to the callables in return_getters_groups
    # gives you the mapping from input vars -> flattened vars.
    # From there, we can compute the output, normalized variables.
    # For instance [6, 6] corresponding to flat vars v0, v1 will be
    # v0 + 6 * v1

    # Create flattened iteration variables
    num_vars = sum(len(s) for s in new_ranges)
    flat_vars = sympy.symbols(f"v_0:{num_vars}")
    count = 0

    if len(iter_vars) == 0 and len(red_vars) == 0:
        return {}

    if len(new_ranges) != len(norm_pw_vars + norm_red_vars):
        raise AssertionError(
            f"expected len(new_ranges) == len(norm_pw_vars + norm_red_vars), "
            f"got {len(new_ranges)} and {len(norm_pw_vars + norm_red_vars)}"
        )
    apply_groups = []
    for group in return_getters_groups:
        apply_groups.append([g(flat_vars) for g in group])

    iter_vars_to_flat_vars = {}
    for i, (group, var_group) in enumerate(
        zip(apply_groups, (iter_vars, red_vars), strict=True)
    ):
        # if the node has sizes (p0, 1) and the fused node is (p0, r0)
        # the reduction var gets filled in for split_iteration_range
        if len(group) != len(var_group):
            if i != 1:
                raise AssertionError(f"expected i == 1, got {i}")
            if len(var_group) != 0:
                raise AssertionError(
                    f"expected empty var_group, got len {len(var_group)}"
                )
            continue

        iter_vars_to_flat_vars.update({v: g for g, v in zip(group, var_group)})

    count = 0
    flat_vars_to_new_vars = {}
    for new_range, new_var in zip(
        new_ranges, norm_pw_vars + norm_red_vars, strict=True
    ):
        range_vars = []
        for _ in range(len(new_range)):
            range_vars.append(flat_vars[count])
            count += 1

        prod = 1
        for i in range(len(new_range) - 1, -1, -1):
            flat_vars_to_new_vars[range_vars[i]] = new_var * prod
            prod = new_range[i] * prod

    return {
        k: sympy_subs(v, flat_vars_to_new_vars)
        for k, v in iter_vars_to_flat_vars.items()
    }


def extract_normalized_read_writes(
    node: Union["_FusedNodeView", "FusedSchedulerNode", "SchedulerNode"],
) -> FusedNormalizedReadsWrites | None:
    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
    reads: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
    writes: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)

    all_output_names = node.get_buffer_names()
    op_names = node.get_operation_names()
    outputs: OrderedSet[str] = OrderedSet()
    removed_buffers: OrderedSet[str] = OrderedSet()
    for buf_name in all_output_names:
        if V.graph.scheduler.can_buffer_be_removed_through_fusion(buf_name, op_names):
            removed_buffers.add(buf_name)
        else:
            outputs.add(buf_name)

    inputs = OrderedSet(
        dep.name for dep in node.read_writes.reads if dep.name not in removed_buffers
    )

    pointwise_numel: sympy.Expr = node.group[1][0]
    red_numel: sympy.Expr = node.group[1][1]

    pw_splits, red_splits = NodeSplitGetter(node).get_node_splits()

    # lets use different prefix (`n`) to distinguish
    (norm_pw_vars, norm_red_vars), ranges = index_vars_no_squeeze(
        pw_splits, red_splits, prefix="n"
    )

    for n in list(node.get_nodes()):
        if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
            continue

        body = n._body

        n_reads: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
        n_writes: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)

        # TODO - will the names for all the inputs/outputs accurately
        # reflect mutation, or do I need to remap with mutation_real_name
        for inp in inputs:
            for expr in body.get_all_read_expr(inp):
                n_reads[expr].add(inp)

        for out in outputs:
            for expr in body.get_all_write_expr(out):
                n_writes[expr].add(out)

        if not n_reads and not n_writes:
            continue

        (iter_vars, n_pw_splits), (red_vars, n_red_splits) = get_pw_red_splits(
            n, pointwise_numel, red_numel
        )

        groups = pw_splits + red_splits
        lengths = (n_pw_splits, (n_red_splits))
        lengths = (
            torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(
                groups, lengths, red_numel
            )
        )
        try:
            new_ranges, return_getters_groups = (
                torch._inductor.codegen.simd.SIMDKernel._split_iteration_ranges(
                    groups, lengths
                )
            )
        except torch._inductor.codegen.simd.CantSplit as e:
            # occasionally with dynamic shapes, we will be unable to prove
            # divisibility
            if not (pointwise_numel.free_symbols or red_numel.free_symbols):
                raise AssertionError(
                    "expected dynamic shapes (free symbols) when split fails"
                ) from e
            return None

        var_map = apply_var_mapping(
            iter_vars,
            red_vars,
            norm_pw_vars,
            norm_red_vars,
            new_ranges,
            return_getters_groups,
        )

        # We create Identity sympy.Functions to prevent expansion to int64,
        # unwrap for tiling analysis.
        def remove_identity(expr: sympy.Expr) -> sympy.Expr:
            return expr.replace(Identity, lambda x: x)

        n_reads_new = {
            sympy_subs(remove_identity(read), var_map): v for read, v in n_reads.items()
        }
        n_writes_new = {
            sympy_subs(remove_identity(write), var_map): v
            for write, v in n_writes.items()
        }

        for expr, buf_names in n_reads_new.items():
            reads[expr] |= buf_names

        for expr, buf_names in n_writes_new.items():
            writes[expr] |= buf_names

    reads = {
        V.graph.sizevars.simplify_with_ranges(r, ranges): v for r, v in reads.items()
    }
    writes = {
        V.graph.sizevars.simplify_with_ranges(w, ranges): v for w, v in writes.items()
    }

    fused_out = FusedNormalizedReadsWrites(
        norm_pw_vars,  # type: ignore[arg-type]
        norm_red_vars,  # type: ignore[arg-type]
        reads,
        writes,
        ranges,
    )
    loop_tiling_log.info("Normalized Fused reads: %s", fused_out)
    return fused_out


def get_score(
    addr: sympy.Expr, var_ranges: dict[sympy.Symbol, int], buf_names: OrderedSet[str]
) -> int:
    """
    Score addr according to its approximate size.
    """
    # TODO - deduplicate with candidate_tilings
    var_sizes = []
    for v in addr.free_symbols:
        v_size = var_ranges.get(v)
        # TODO - reason about indirect vars
        if not symbol_is_type(v, SymT.INDIRECT) and v_size is not None:
            var_sizes.append(v_size)
    from .virtualized import V

    return V.graph.sizevars.optimization_hint(sympy_product(var_sizes))


def try_get_buf_size(buf_name: str) -> int | None:
    buf = V.graph.try_get_buffer(buf_name)
    if not buf:
        return None
    return V.graph.sizevars.optimization_hint(sympy_product(buf.get_size()))


def get_hint(v: sympy.Expr | int) -> int:
    if isinstance(v, int):
        return v
    else:
        return V.graph.sizevars.optimization_hint(v)


@dataclasses.dataclass(frozen=True)
class VarTiling:
    """
    Tiling of a var by `tiling_factor` that yields additional coalesced mem accesses by `benefit_score`
    """

    var: sympy.Symbol
    tiling_factor: int
    score: int


@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    # Var -> Memory Score - not strictly the amount of memory
    # because we multiply writes x2
    # TODO: separate into dataclass that holds mem, dtype, is_write
    coalesced_by_var: dict[sympy.Expr, int]

    uncoalesced_addrs: dict[sympy.Expr, int]

    norm_read_writes: FusedNormalizedReadsWrites

    suggested_split: VarTiling | None = None


def _analyze_memory_coalescing(
    fused_node: Union["_FusedNodeView", "FusedSchedulerNode", "SchedulerNode"],
) -> CoalesceVarAnalysis | None:
    """
    Implementation for BaseSchedulerNode.get_coalesce_analysis().
    Call that node method so loop-transform cache invalidation is honored.

    Find variables that coalesce the reads and writes and score the total size.

    If uncoalesced memory expressions are found, look for additionally tiling of variables
    which will coalesce memory accesses.

    For instance - for the following expression:

    (32*p0) // 2048

    Tiling p0 by 64 will make this expression coalesced.
    """

    norm_read_writes = extract_normalized_read_writes(fused_node)

    if norm_read_writes is None:
        return None

    reads = norm_read_writes.reads
    writes = norm_read_writes.writes
    var_ranges = norm_read_writes.var_ranges

    coalesced_by_var: dict[sympy.Symbol, int] = Counter()
    uncoalesced_addrs: dict[sympy.Expr, int] = Counter()

    # Only check pointwise-only kernels
    index_vars = norm_read_writes.index_vars
    reduce_vars = norm_read_writes.reduce_vars
    innermost_var = (
        next(reversed(index_vars)) if index_vars and not reduce_vars else None
    )

    for is_read, (memory_expr, buf_names) in itertools.chain(
        ((True, item) for item in reads.items()),
        # pyrefly: ignore [bad-argument-type]
        ((False, item) for item in writes.items()),
    ):
        size = get_score(memory_expr, var_ranges, buf_names)
        if size == 0:
            continue

        # accesses with indirect expressions are never coalesced
        indirect_expr = has_indirect_access(memory_expr)

        if indirect_expr:
            maybe_coalesced_var = None
        else:
            maybe_coalesced_var = find_coalesced_var(memory_expr, var_ranges)
            # while broadcasting vars are not technically coalesced,
            # accesses at least stay in cache, so they provide most of the benefit.
            # treat the same for now.
            if maybe_coalesced_var is None:
                maybe_coalesced_var = find_broadcast_var(memory_expr, var_ranges)

        total_score = 0
        for buf_name in buf_names:
            if (buf := V.graph.try_get_buffer(buf_name)) and (
                buf_size := try_get_buf_size(buf_name)
            ):
                # constrain by buf size since we'll read at most that many elements
                # score could be more through either masking or by broadcasting (e.g. x // 16)
                total_score += min(buf_size, size) * buf.dtype.itemsize

        # coalesced writes more important
        total_score *= 1 if is_read else 2

        if maybe_coalesced_var:
            # Check if the coalescing is already achieved in 1D iteration.
            # Skip the innermost variable: it always varies across threads,
            # so its coalescing is always real.
            already_coalesced_1d = False
            if innermost_var is not None and maybe_coalesced_var != innermost_var:
                # Evaluate stride at two points (0->1 and 1->2) to catch
                # non-linear expressions that only look coalesced at the origin.
                subs = dict.fromkeys(var_ranges, 0)
                try:
                    val_0 = sympy_subs(memory_expr, subs)
                    subs[innermost_var] = 1
                    val_1 = sympy_subs(memory_expr, subs)
                    stride_01 = val_1 - val_0
                    if stride_01 in (0, 1):
                        subs[innermost_var] = 2
                        val_2 = sympy_subs(memory_expr, subs)
                        stride_12 = val_2 - val_1
                        if stride_12 in (0, 1):
                            already_coalesced_1d = True
                except (ZeroDivisionError, TypeError):
                    pass

            if not already_coalesced_1d:
                coalesced_by_var[maybe_coalesced_var] += total_score
            else:
                coalesced_by_var[innermost_var] += total_score
        else:
            uncoalesced_addrs[memory_expr] += total_score

    if not uncoalesced_addrs:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
        )

    # map from var -> tiling -> total_score
    tiling_scores: dict[sympy.Expr, dict[int, int]] = defaultdict(Counter)

    for uncoalesced_expr, addr_score in uncoalesced_addrs.items():
        if has_indirect_access(uncoalesced_expr):
            continue

        expr_subs = dict.fromkeys(var_ranges.keys(), 0)
        for v in uncoalesced_expr.free_symbols & var_ranges.keys():
            # skip non iter/reduce var variables
            if v not in var_ranges:
                continue
            # skip small addrs
            if addr_score == 0:
                continue

            del expr_subs[v]
            single_var_expr = sympy_subs(uncoalesced_expr, expr_subs)
            expr_subs[v] = 0

            if len(single_var_expr.free_symbols) != 1:
                continue

            tiling_factor = solve_for_tiling(single_var_expr)

            if (
                tiling_factor is None
                or not tiling_factor.is_constant()
                or not tiling_factor.is_integer
            ):
                continue

            tiling_factor = int(tiling_factor)
            if not V.graph.sizevars.statically_known_lt(tiling_factor, var_ranges[v]):
                continue

            # TODO - if a var is in the middle, such as [n0, n1, n2]
            # n1 can be split beyond range

            MIN_TILING_BLOCK = 8
            if not all(
                V.graph.sizevars.statically_known_lt(MIN_TILING_BLOCK, block)
                for block in (tiling_factor, var_ranges[v] // tiling_factor)
            ):
                continue

            tiling_scores[v][tiling_factor] += addr_score

    if len(tiling_scores) == 0:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
        )

    best_tiling: tuple[sympy.Expr, int] | None = None
    best_tiling_score = 0

    for var, tiling_counter in tiling_scores.items():
        for tile, tile_score in tiling_counter.items():
            if tile_score > best_tiling_score:
                best_tiling = (var, tile)
                best_tiling_score = tile_score

    if best_tiling is None:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
        )

    # TODO - for strictly pointwise fusions,
    # we can consider just swizzling the var if the var we are going to tile
    # does not coalesce a significant portion of global reads
    # TODO - could also prefer index var splits to reduction, better tested
    return CoalesceVarAnalysis(
        coalesced_by_var=coalesced_by_var,
        uncoalesced_addrs=uncoalesced_addrs,
        norm_read_writes=norm_read_writes,
        suggested_split=VarTiling(best_tiling[0], best_tiling[1], best_tiling_score),
    )


def analyze_memory_coalescing_for_nodes(
    nodes: Sequence["BaseSchedulerNode"],
) -> CoalesceVarAnalysis | None:
    if not nodes:
        return None

    from torch._inductor import scheduler

    node_types = (scheduler.FusedSchedulerNode, scheduler.SchedulerNode)
    if not all(isinstance(node, node_types) for node in nodes):
        return None

    if len(nodes) == 1:
        return nodes[0].get_coalesce_analysis()

    graph_scheduler = getattr(V.graph, "scheduler", None)
    if graph_scheduler is not None:
        fused_node = graph_scheduler.name_to_fused_node.get(nodes[0].get_first_name())
        if fused_node is not None:
            fused_nodes = list(fused_node.get_nodes())
            if len(fused_nodes) == len(nodes) and all(
                fused is node for fused, node in zip(fused_nodes, nodes, strict=True)
            ):
                return fused_node.get_coalesce_analysis()

    return _analyze_memory_coalescing(
        _FusedNodeView(
            nodes=nodes,
            read_writes=ReadWrites.merge_list([node.read_writes for node in nodes]),
            group=max(nodes, key=lambda node: int(node.is_reduction())).group,
        )
    )
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import itertools
import logging
import weakref
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code
from torch._functorch.aot_autograd import MutationType
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.constant_folding import constant_fold, replace_node_with_constant
from torch._inductor.freezing_utils import enter_freezing, record_has_frozen_params
from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.fx_passes.post_grad import view_to_reshape

from . import config


aten = torch.ops.aten
prims = torch.ops.prims

log = logging.getLogger(__name__)


def replace_params_with_constants(
    gm: torch.fx.GraphModule,
    flat_params: list[Any],
    fw_metadata: torch._functorch.aot_autograd.ViewAndMutationMeta,
) -> list[int]:
    """
    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.
    Returns a list of indices representing the input parameters that were not converted to constants.
    """
    params = gm.graph.find_nodes(op="placeholder")
    fake_inp_nodes = params[: len(params)]
    preserved_arg_indices = []
    aliased_input_args = [
        out_info.base_idx
        for out_info in fw_metadata.output_info
        if out_info.base_idx is not None
    ]

    # TODO (tmanlaibaatar) figure out why this is different
    # from mutated_inp_runtime_indices
    mutated_inps = [
        i
        for i, m in enumerate(fw_metadata.input_info)
        if m.mutation_type
        in (MutationType.MUTATED_IN_GRAPH, MutationType.MUTATED_OUT_GRAPH)
    ]

    static_indices_new = []
    static_indices_offset = 0
    for i, (real_input, node) in enumerate(zip(flat_params, fake_inp_nodes)):
        if i in mutated_inps or i in aliased_input_args:
            preserved_arg_indices.append(i)
            if i in fw_metadata.static_input_indices:
                new_static_index = i - static_indices_offset
                static_indices_new.append(new_static_index)
        else:
            replace_node_with_constant(gm, node, real_input)
            static_indices_offset += 1
    # add on non param inputs
    preserved_arg_indices.extend(range(len(flat_params), len(params)))
    # is this necessary ?
    fw_metadata.static_input_indices = static_indices_new
    gm.recompile()
    return preserved_arg_indices


def freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: list[torch._subclasses.FakeTensor],
) -> tuple[torch.fx.GraphModule, list[int]]:
    """
    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        dynamo_gm (torch.fx.GraphModule): The Dynamo constructed GraphModule.
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.

    Returns:
        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices
        of the inputs that were preserved (not turned into constants).
    """
    with enter_freezing():
        return _freeze(dynamo_gm, aot_autograd_gm, example_inputs)


def _freeze(
    dynamo_gm: torch.fx.GraphModule,
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: list[torch._subclasses.FakeTensor],
) -> tuple[torch.fx.GraphModule, list[int]]:
    # We have converted conv's weight to channels last which may meet error for .view
    # when doing fake_tensor_prop. So we need to convert view to reshape first.
    # See the details in fx_codegen_and_compile of compile_fx.py.
    view_to_reshape(aot_autograd_gm)

    if tracing_context := torch._guards.TracingContext.try_get():
        fw_metadata = tracing_context.fw_metadata
        if tracing_context.params_flat_unwrap_subclasses is None:
            raise AssertionError(
                "expected tracing_context.params_flat_unwrap_subclasses to be set"
            )
        params_flat = tracing_context.params_flat_unwrap_subclasses
        if not (fw_metadata is not None and params_flat is not None):
            raise AssertionError("expected fw_metadata and params_flat to be set")

        preserved_arg_indices = replace_params_with_constants(
            aot_autograd_gm, params_flat, fw_metadata
        )
    else:
        inputs = aot_autograd_gm.graph.find_nodes(op="placeholder")
        preserved_arg_indices = list(range(len(inputs)))

    # TODO - further restrict cse ? right now needed to dedup aliasing ops
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    aot_autograd_gm.recompile()

    aot_example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
    freezing_passes(aot_autograd_gm, aot_example_inputs)

    constant_fold(aot_autograd_gm)
    # invalidate nn Modules
    if config.freezing_discard_parameters:
        invalidate_eager_modules()
        discard_traced_gm_params(dynamo_gm)

    log.debug(
        "%s", lazy_format_graph_code("FROZEN GRAPH", aot_autograd_gm, colored=True)
    )

    record_has_frozen_params(aot_autograd_gm)
    return aot_autograd_gm, preserved_arg_indices


class ErasedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        return super().__new__(cls, elem.to(device="meta"))

    def __init__(self, elem, name: str | None, mod) -> None:
        self.erased_name = name
        self.owning_mod_ref = weakref.ref(mod)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        erased_tensors = [
            e
            # pyrefly: ignore [bad-unpacking]
            for e in pytree.arg_tree_leaves(*args, **kwargs)
            if isinstance(e, ErasedTensor)
        ]
        if len(erased_tensors) == 0:
            raise AssertionError("expected at least one ErasedTensor argument")
        e = erased_tensors[0]

        raise RuntimeError(
            f"Trying to run Pytorch Eager Module after Dynamo Freezing. "
            "The original parameters have been discarded for memory efficiency. "
            f"Found in op {func} for erased parameter {e.erased_name} of {e.owning_mod_ref()}"
        )


def invalidate_eager_modules():
    with torch.utils._python_dispatch._disable_current_modes():
        for (
            mod
        ) in torch._guards.TracingContext.get().module_context.nn_modules.values():
            if not isinstance(mod, torch.nn.Module):
                continue

            for attr_name, tensor in list(
                itertools.chain(
                    mod.named_parameters(recurse=False),
                    # pyrefly: ignore [bad-argument-type]
                    mod.named_buffers(recurse=False),
                )
            ):
                with torch._dispatch.python.no_python_dispatcher():
                    e_t = ErasedTensor(tensor, attr_name, mod)
                if isinstance(tensor, torch.nn.Parameter):
                    e_t.requires_grad_(True)
                    e_t._is_param = True
                setattr(mod, attr_name, e_t)


def discard_traced_gm_params(mod: torch.fx.GraphModule):
    with torch.utils._python_dispatch._disable_current_modes():
        for attr_name, tensor in list(
            itertools.chain(
                mod.named_parameters(recurse=False),
                # pyrefly: ignore [bad-argument-type]
                mod.named_buffers(recurse=False),
            )
        ):
            with torch._dispatch.python.no_python_dispatcher():
                e_t = ErasedTensor(tensor, attr_name, mod)
            if isinstance(tensor, torch.nn.Parameter):
                e_t.requires_grad_(True)
                e_t._is_param = True
            setattr(mod, attr_name, e_t)


def enforce_output_layout(gm: torch.fx.GraphModule):
    """
    Make sure the output node's layout does not change due to compiler optimizations
    by adding aten.as_strided nodes with the expected strides.

    Only used for inference so we can assume all graph outputs are model outputs.
    """
    *_, output_node = gm.graph.nodes
    out_list = output_node.args[0]
    with gm.graph.inserting_before(output_node):
        for n in out_list:
            if not isinstance(
                n.meta["val"], torch.Tensor
            ) or not torch._prims_common.is_non_overlapping_and_dense_or_false(
                n.meta["val"]
            ):
                continue

            # Use materialize_symints intentionally; see its docstring for why.
            ft = n.meta["val"]
            stride_args = tuple(gm.graph.materialize_symints(ft.stride()))
            new_node = gm.graph.call_function(
                prims.inductor_force_stride_order.default, (n, stride_args)
            )

            # can not call
            # n.replace_all_uses_with(new_node)
            # since it will replace the usage of n in new_node itself.
            output_node.replace_input_with(n, new_node)

    gm.graph.lint()
    gm.recompile()


def enforce_as_strided_input_layout(gm: torch.fx.GraphModule):
    """
    Make sure the as_strided node's input's layout does not change due to compiler
    optimizations, because the as_strided strides info depends on input tensor stride info.
    """

    as_strided_ops = [
        torch.ops.aten.as_strided.default,
        torch.ops.aten.as_strided_.default,
        torch.ops.aten.as_strided_scatter.default,
    ]
    strided_nodes = [n for n in gm.graph.nodes if n.target in as_strided_ops]
    for n in strided_nodes:
        with gm.graph.inserting_before(n):
            # Use materialize_symints intentionally; see its docstring for why.
            ft = n.args[0].meta["val"]
            stride_args = tuple(gm.graph.materialize_symints(ft.stride()))
            new_node = gm.graph.call_function(
                prims.inductor_force_stride_order.default, (n.args[0], stride_args)
            )
        n.replace_input_with(n.args[0], new_node)

    gm.graph.lint()
    gm.recompile()


def convert_conv_weights_to_channels_last(gm: torch.fx.GraphModule):
    """
    Convert 4d convolution weight tensor to channels last format.

    This pass is performed before freezing so the added nodes can be constant
    folded by freezing.
    """
    with dynamo_timed("convert_conv_weights_to_channels_last"):
        convs = [n for n in gm.graph.nodes if n.target is aten.convolution.default]
        for conv in convs:
            weight_node = conv.args[1]
            if len(weight_node.meta["val"].size()) != 4 or weight_node.meta[
                "val"
            ].is_contiguous(memory_format=torch.channels_last):
                # not a 4d tensor or already channels last, skip
                continue

            with gm.graph.inserting_before(conv):
                new_node = gm.graph.call_function(
                    aten.clone.default,
                    (weight_node,),
                    {"memory_format": torch.channels_last},
                )
                conv.replace_input_with(weight_node, new_node)

        enforce_as_strided_input_layout(gm)
        enforce_output_layout(gm)
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

from torch._logging import trace_structured

from .memory import estimate_peak_memory_allocfree


if TYPE_CHECKING:
    from torch.utils._ordered_set import OrderedSet

    from .memory import FreeableInputBuffer, SNodeMemory
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


def _debug_iterative_memory_recompute(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    group_names: str,
    snodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
    peak_memory: int,
    iter_curr_memory: dict[BaseSchedulerNode, tuple[int, int]],
    snodes_allocfree: dict[BaseSchedulerNode, SNodeMemory],
    tlparse_name: str,
    gn_to_bufs_last_use: dict[
        BaseSchedulerNode, list[FreeableInputBuffer | SchedulerBuffer]
    ],
) -> bool:
    iterative_recompute_error = False
    candidate_allocfree = snodes_allocfree[candidate]
    est_peak_memory, snodes_curr_memory, snodes_allocfree, _ = (
        estimate_peak_memory_allocfree(
            snodes, name_to_freeable_input_buf, graph_outputs
        )
    )
    est_curr_memory = dict(zip(snodes, snodes_curr_memory))
    iter_cm = iter_curr_memory[candidate]
    new_cm = est_curr_memory[candidate]
    log = ""
    if est_peak_memory > peak_memory:
        log = "ITERATIVE PEAK DOES NOT MATCH"
        iterative_recompute_error = True
    if iter_cm != new_cm:
        log = "ITERATIVE CURR MEMORY CANDIDATE DOES NOT MATCH"
        iterative_recompute_error = True
    for gn in gns:
        iter_gnm = iter_curr_memory[gn]
        new_gnm = est_curr_memory[gn]
        if iter_gnm != new_gnm:
            log = f"ITERATIVE GN CURR MEMORY DOES NOT MATCH:{gn.get_name()}"
            iterative_recompute_error = True
    if iterative_recompute_error:
        log += (
            f"\nCANDIDATE:{candidate.get_name()}"
            f"\nGROUP:{group_names}"
            f"\nPEAK_MEMORY_BEFORE:{peak_memory}"
            f"\nPEAK_MEMORY_AFTER_SWAP:{est_peak_memory}"
            f"\nCANDIDATE:{candidate.debug_str()}"
            f"\nCANDIDATE_ITER_CURR_MEMORY:{iter_cm}"
            f"\nCANDIDATE_NEW__CURR_MEMORY:{new_cm}"
            f"\nCANDIDATE_ITER_ALLOCFREE:{candidate_allocfree}"
            f"\nCANDIDATE_NEW_ALLOCFREE:{snodes_allocfree[candidate]}"
        )
        peak_log = ""
        for i, (pre, _post) in enumerate(snodes_curr_memory):
            if est_peak_memory == pre:
                n = snodes[i]
                peak_log = (
                    f"\nNEW_PEAK:{est_peak_memory}(BASE:{peak_memory})"
                    f" @ SNODE[{i}/{len(snodes)}]:{n.get_name()} {n.debug_str()}"
                )
                break
        group_log = ""
        for i, gn in enumerate(gns):
            iter_gnm = iter_curr_memory[gn]
            new_gnm = est_curr_memory[gn]
            group_log += (
                f"\nGROUP_NODE[{i}]:{gn.debug_str()}"
                f"\nGROUP_NODE[{i}] ITER_GNM[{gn.get_name()}]:{iter_gnm}"
                f"\nGROUP_NODE[{i}] ESTM_GNM[{gn.get_name()}]:{new_gnm}"
                f"\nGROUP_NODE[{i}] ITER_allocfree:{snodes_allocfree[gn]}"
                f"\nGROUP_NODE[{i}] ESTM_allocfree:{snodes_allocfree[gn]}"
            )
        log += peak_log
        log += group_log
        log += f"\nGN_TO_BUFS_LAST_USE:{gn_to_bufs_last_use}"
        log += "\n\n".join(
            [
                (
                    f"\nSNODE[{i}]\n{n.debug_str()}"
                    f"\nITER_cur_mem:{iter_curr_memory[n]}"
                    f"\nESTM_cur_mem:{est_curr_memory[n]}"
                    f"\nITER_allocfree:{snodes_allocfree[n]}"
                    f"\nESTM_allocfree:{snodes_allocfree[n]}"
                )
                for i, n in enumerate(snodes)
            ]
        )
        tname = f"{tlparse_name}_ITERATIVE_RECOMPUTE_ERROR"
        print(f"{tname}:\n{log}")
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": tname,
                "encoding": "string",
            },
            payload_fn=lambda: log,
        )
    return iterative_recompute_error
<|endoftext|>from __future__ import annotations

import contextlib
import copy
import enum
import functools
import io
import itertools
import json
import logging
import os
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from inspect import currentframe
from itertools import count
from operator import attrgetter
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import Never, override, ParamSpec, Protocol, TypedDict, Unpack
from unittest import mock

import torch._inductor.async_compile
import torch.fx
import torch.utils._pytree as pytree
from functorch.compile import min_cut_rematerialization_partition
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import (
    compiled_autograd,
    config as dynamo_config,
    logging as dynamo_logging,
    utils as dynamo_utils,
)
from torch._dynamo.backends import common as dynamo_common
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.repro.after_aot import wrap_compiler_debug
from torch._dynamo.utils import (
    chromium_event_timed,
    CompileEventLogger,
    counters,
    detect_fake_mode,
    dynamo_timed,
    flatten_graph_inputs,
    get_inputs_devices,
    get_metrics_context,
    GmWrapper,
    lazy_format_graph_code,
    set_feature_use,
)
from torch._functorch import aot_autograd, config as functorch_config
from torch._functorch._aot_autograd.subclass_parametrization import (
    unwrap_tensor_subclass_parameters,
)
from torch._functorch.aot_autograd import (
    aot_export_module,
    GraphOutputName,
    make_boxed_func,
    SerializableAOTDispatchCompiler,
)
from torch._inductor.codecache import code_hash, FxGraphCache, output_code_log
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    cudagraph_trees_clone_live_user_visible_outputs,
    cudagraphs_log,
    format_default_skip_message,
    log_cudagraph_skip_and_bump_counter,
    PlaceholderInfo,
)
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch._inductor.debug import (
    create_kernel_information_json,
    create_mapping_pre_post_grad_nodes,
    get_kernel_information_jsons,
    save_args_for_compile_fx_inner,
)
from torch._inductor.output_code import (
    CompiledAOTI,
    CompiledFxGraph,
    CompiledFxGraphConstantsWithGm,
    copy_strided_storage_,
    get_expanded_dims,
    index_expanded_dims,
    OutputCode,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.utils import (
    BoxedBool,
    count_tangents,
    fresh_cache,
    get_all_devices,
    get_static_bw_input_idxs,
    InputType,
    is_gpu,
    should_assume_input_aligned,
    should_use_remote_fx_graph_cache,
    tensor_is_aligned,
)
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class
from torch._logging import trace_structured
from torch._utils_internal import compile_time_strobelight_meta
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymExprPrinter
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.monitor import _WaitCounter
from torch.utils._ordered_set import OrderedSet

from .._dynamo.exc import ShortenTraceback, SkipFrame
from ..fx._lazy_graph_module import _use_lazy_graph_module
from ..fx.graph import _PyTreeCodeGen
from ..utils._triton import has_triton
from . import config, distributed_autotune, metrics
from .codegen.common import get_wrapper_codegen_for_device, init_backend_registration
from .debug import DebugContext
from .decomposition import select_decomp_table
from .exc import InductorError
from .fx_passes.joint_graph import joint_graph_passes
from .fx_passes.post_grad import post_grad_passes, view_to_reshape
from .fx_passes.pre_grad import pre_grad_passes
from .graph import GraphLowering
from .ir import get_device_type, IRNode
from .triton_bundler import TritonBundler
from .utils import (
    align_inputs_from_check_idxs,
    clone_preserve_strides,
    copy_misaligned_inputs,
    get_cloned_parameter_buffer_name,
    get_first_incompatible_cudagraph_node,
    maybe_get_suppress_shape_guards_ctx,
    output_node,
    remove_unaligned_input_idxs,
    shape_env_from_inputs,
)
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch._inductor.output_code import _StrideExprStr, CompiledFnRunner
    from torch._ops import OpOverload
    from torch.export.pt2_archive._package_weights import Weights

    from .codecache import CacheInfo
    from .ir import ExternKernelNode


_P = ParamSpec("_P")
_T = TypeVar("_T")

if TYPE_CHECKING or not config.is_fbcode():
    # no-op decorator
    def time_and_log(attr: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        return dynamo_utils.identity

    def log_optimus_to_scuba(*args: object, **kwargs: object) -> None:
        pass

else:
    from torch._inductor.fb.utils import log_optimus_to_scuba, time_and_log

if TYPE_CHECKING:
    from torch._functorch._aot_autograd.schemas import (
        FQN,
        GraphInputName,
        GraphSignature,
    )

    CompileFxOutput = (
        Callable[[list[object]], Sequence[torch.Tensor]] | str | list[str] | Weights
    )


class FxCompileMode(enum.Enum):
    NORMAL = 0
    # For testing - use the serde FxCompile scheme to debug serialization and
    # deserialization of GraphModule and CompiledFxGraph.
    SERIALIZE = 1
    # Compile using a subprocess instead of in-process.
    SUBPROCESS = 2


@dataclass
class FxCompileConfig:
    mode: FxCompileMode
    use_async: bool
    use_progressive: bool


def _fx_compile_mode_default() -> FxCompileConfig:
    name = "TORCHINDUCTOR_FX_COMPILE_MODE"
    value = os.environ.get(name)
    if value is None:
        return FxCompileConfig(FxCompileMode.NORMAL, False, False)

    use_async = False
    use_progressive = False

    if value.lower().startswith("progressive+"):
        use_progressive = True
        value = value[12:]
    if value.lower().startswith("async+"):
        use_async = True
        value = value[6:]

    try:
        value = value.upper()
        return FxCompileConfig(FxCompileMode[value], use_async, use_progressive)
    except KeyError:
        import logging

        log = logging.getLogger(__name__)
        log.error(
            "Invalid value of %s for %s. Expected one of %s. Using default.",
            value,
            name,
            ", ".join(sorted(repr(x) for x in FxCompileMode.__members__)),
        )
        # Remove from the environment so subprocesses don't ALSO complain.
        os.environ.pop(name)
        return FxCompileConfig(FxCompileMode.NORMAL, False, False)


def _get_progression_configs() -> list[dict[str, Any]]:
    # TODO make this configurable
    return [
        {"max_autotune": True},
    ]


_fx_compile_config = _fx_compile_mode_default()
fx_compile_mode = _fx_compile_config.mode
fx_compile_async = _fx_compile_config.use_async
fx_compile_progressive = _fx_compile_config.use_progressive

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
pre_grad_graphs_log = torch._logging.getArtifactLogger(__name__, "pre_grad_graphs")
post_grad_graphs_log = torch._logging.getArtifactLogger(__name__, "post_grad_graphs")
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)
inductor_metrics_log = torch._logging.getArtifactLogger(__name__, "inductor_metrics")


def get_static_input_idxs(num_fixed: int) -> list[int]:
    # If we are inlining NNModules, we treat all torch.nn.Parameters as static for the purposes
    # of cudagraphs. Rather than copying these into cudagraph-owned memory
    # like we do for normal inputs on each run, we will re-record a cudagraph if these
    # parameter locations change.
    context = torch._guards.TracingContext.try_get()
    fixed = list(range(num_fixed))
    if not context or not context.fw_metadata:
        return fixed

    return context.fw_metadata.static_input_indices


def record_original_output_strides(gm: GraphModule) -> None:
    output_node = gm.graph.find_nodes(op="output")[0]

    # Don't overwrite strides that were already recorded (e.g., before
    # joint_graph_passes which can introduce padded strides via pad_mm).
    if "original_output_strides" in output_node.meta:
        return

    output_strides = []

    if not isinstance(output_node.args[0], torch.fx.Node):
        output_node_args = output_node.args[0]
    else:
        output_node_args = output_node.args

    for output in output_node_args:
        if (
            isinstance(output, torch.fx.Node)
            and (val := output.meta.get("val")) is not None
            and isinstance(val, torch.Tensor)
        ):
            output_strides.append(val.stride())
        else:
            # pyrefly: ignore [bad-argument-type]
            output_strides.append(None)
    output_node.meta["original_output_strides"] = output_strides


def _recursive_record_original_output_strides(gm: GraphModule) -> None:
    # invoke_subgraph HOP requires output strides to be respected
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        subgraph = getattr(gm, node.args[0].target)
        _recursive_record_original_output_strides(subgraph)

    record_original_output_strides(gm)


def _recursive_record_user_visible_output_idxs(gm: GraphModule) -> None:
    # invoke_subgraph HOP requires output strides to be respected
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        subgraph = getattr(gm, node.args[0].target)

        for node in subgraph.graph.find_nodes(op="output"):
            node.meta["user_visible_output_idxs"] = [
                idx
                for idx in range(len(node.args[0]))
                if isinstance(node.args[0][idx], torch.fx.Node)
            ]
        _recursive_record_user_visible_output_idxs(subgraph)


def _cudagraph_trees_clone_live_user_outputs() -> bool:
    return cudagraph_trees_clone_live_user_visible_outputs()


@functools.lru_cache(None)
def _step_logger() -> Callable[..., None]:
    return dynamo_logging.get_step_logger(log)


@functools.cache
def _warn_tf32_disabled() -> None:
    if (
        torch.cuda.is_available()
        and torch.backends.cuda.matmul.fp32_precision != "tf32"
        and torch.cuda.get_device_capability() >= (8, 0)
    ):
        perf_hint_log.info(
            "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. "
            "Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
        )


def _resolve_name_collision(mod: GraphModule, gm: GraphModule) -> None:
    """
    In aot_export_module (make_fx), we create get_attr nodes with name prefix
    "_tensor_constant" and "_torchbind_obj". See Tracer.create_arg() in
    torch/fx/_symbolic_trace.py

    However, this might result in name collision if the original mod already
    has a different buffer with the same name.

    We resolve this potential name collision here by changing the target name
    with a new number post fix.
    """

    existing_keys = OrderedSet(
        [name for name, val in mod.named_parameters(remove_duplicate=False)]
    )
    existing_keys.update(
        OrderedSet([name for name, val in mod.named_buffers(remove_duplicate=False)])
    )

    def find_smallest_i(graph: fx.Graph, prefix: str) -> int:
        i = 0
        for node in graph.nodes:
            if node.op == "get_attr" and node.target.startswith(prefix):
                if len(node.target) > len(prefix):
                    post_fix = node.target.split(prefix)[-1]
                    if post_fix.isdigit():
                        i = max(i, int(post_fix))
        for key in existing_keys:
            if key.startswith(prefix):
                if len(key) > len(prefix):
                    post_fix = key.split(prefix)[-1]
                    if post_fix.isdigit():
                        i = max(i, int(post_fix))
        return i + 1

    for node in gm.graph.nodes:
        if node.op == "get_attr":
            target_name = node.target
            if not target_name.startswith(
                "_tensor_constant"
            ) and not target_name.startswith("_torchbind_obj"):
                continue

            if not hasattr(mod, target_name):
                continue
            gm_target = attrgetter(target_name)(gm)
            model_target = attrgetter(target_name)(mod)
            if isinstance(gm_target, FakeScriptObject):
                if (
                    isinstance(model_target, FakeScriptObject)
                    and gm_target.real_obj is model_target.real_obj
                ):
                    continue
            elif (
                gm_target.device == model_target.device
                and gm_target.dtype == model_target.dtype
                and torch.equal(gm_target, model_target)
            ):
                # If tensors with same name from gm and model are indeed the same, we don't need to rename
                # Check device first, to avoid torch.equal(wrapper_CUDA__equal) raise when different device
                continue

            prefix = (
                "_tensor_constant"
                if target_name.startswith("_tensor_constant")
                else "_torchbind_obj"
            )
            new_id = find_smallest_i(gm.graph, prefix)
            new_target_name = f"{prefix}{new_id}"
            node.target = new_target_name
            setattr(gm, new_target_name, gm_target)
            existing_keys.add(new_target_name)


def _unlift_graph(
    mod: GraphModule, gm: GraphModule, graph_signature: GraphSignature
) -> GraphModule:
    from torch.export.unflatten import _assign_attr, _AttrKind

    _resolve_name_collision(mod, gm)

    state_dict: dict[str, torch.nn.parameter.Parameter | torch.Tensor] = {}
    for name, param in mod.named_parameters(remove_duplicate=False):
        state_dict[name] = param
        _assign_attr(
            param,
            gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )
    for name, buffer in mod.named_buffers(remove_duplicate=False):
        state_dict[name] = buffer
        _assign_attr(
            buffer,
            gm,
            name,
            attr_kind=_AttrKind.BUFFER,
        )

    placeholder_nodes = gm.graph.find_nodes(op="placeholder")
    lifted_inputs: list[FQN | None] = []

    # In AOTI, module parameters and buffers are not lifted as graph inputs.
    # As a result, mutation to buffers has side effect which makes their initial
    # values different from Eager. So we clone them here as a copy.
    # We are not cloning for parameters, although it will be needed if we want to
    # support training.
    for node in placeholder_nodes:
        node_name = node.name
        if node_name in graph_signature.inputs_to_parameters:
            parameter_name = graph_signature.inputs_to_parameters[node_name]
            lifted_inputs.append(parameter_name)
        elif node_name in graph_signature.inputs_to_buffers:
            buffer_name = graph_signature.inputs_to_buffers[node_name]
            lifted_inputs.append(buffer_name)
            gm.meta[get_cloned_parameter_buffer_name(buffer_name)] = (
                clone_preserve_strides(state_dict[buffer_name])
            )
        else:
            if node_name not in graph_signature.user_inputs:
                raise AssertionError(
                    f"Expected {node_name} to be in graph_signature.user_inputs"
                )
            lifted_inputs.append(None)

    from torch.export._unlift import _unlift

    outputs: tuple[torch.fx.Node, ...] = tuple(gm.graph.output_node().args[0])  # type: ignore[arg-type]
    mutated_outputs = []
    buffer_mutations = graph_signature.buffers_to_mutate
    user_input_mutations = graph_signature.user_inputs_to_mutate
    output_tokens = graph_signature.output_tokens
    for idx, out in enumerate(outputs):
        value: FQN | GraphInputName | None = None

        if idx < len(buffer_mutations) + len(user_input_mutations) + len(output_tokens):
            name = GraphOutputName(out.name)
            if name in buffer_mutations:
                value = buffer_mutations[name]
            elif name in user_input_mutations:
                value = user_input_mutations[name]

        mutated_outputs.append(value)

    unlifted_gm = _unlift(
        gm,
        lifted_inputs,
        mutated_outputs,
        pytree.treespec_leaf(),
        None,
    )
    # After unlifting, the buffer mutation information is lost. Pass the information
    # so that Inductor can do optimizations correctly.
    unlifted_gm.meta["mutated_named_buffers"] = OrderedSet(buffer_mutations.values())
    return unlifted_gm


def _get_subgraph_names(
    gm: GraphModule, skip_invoke_subgraph: bool = False
) -> Generator[str, None, None]:
    all_subgraph_names: OrderedSet[str] = OrderedSet(
        x.target for x in gm.graph.find_nodes(op="get_attr")
    )
    fx_subgraph_names: OrderedSet[str] = OrderedSet()
    for child_name, child_module in gm.named_children():
        # Sometimes an owning_module can have unused children. Skip them
        # by checking them from get_attr node targets.
        if child_name in all_subgraph_names and isinstance(
            child_module, torch.fx.GraphModule
        ):
            fx_subgraph_names.add(child_name)

    if skip_invoke_subgraph:
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        ):
            fx_subgraph_names.discard(node.args[0].target)

    yield from fx_subgraph_names


def _get_nested_region_inductor_config_patches(
    gm: GraphModule,
) -> dict[str, Any] | None:
    nested_config = getattr(gm, "meta", {}).get("nested_region_config")
    patches = getattr(nested_config, "inductor_config_patches", None)
    return patches or None


def _patch_nested_region_inductor_config(
    gm: GraphModule,
) -> AbstractContextManager[None]:
    patches = _get_nested_region_inductor_config_patches(gm)
    if patches is None:
        return contextlib.nullcontext()
    return config.patch(patches)


def _propagate_invoke_subgraph_nested_region_config(gm: GraphModule) -> None:
    # Seed each invoke_subgraph subgraph module's meta from the node meta, which
    # (unlike a GraphModule's meta) survives FX transforms. Re-run at the start of
    # every pass phase because each is an independent entry point that may see a
    # freshly-created module (e.g. the partitioned backward graph); setdefault
    # makes re-entry on an already-seeded module a no-op.
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        nested_config = node.meta.get("custom", {}).get("nested_region_config")
        if nested_config is None:
            continue
        subgraph_node = node.args[0]
        if (
            not isinstance(subgraph_node, torch.fx.Node)
            or subgraph_node.op != "get_attr"
            or not isinstance(subgraph_node.target, str)
        ):
            continue
        subgraph = getattr(gm, subgraph_node.target, None)
        if isinstance(subgraph, GraphModule):
            subgraph.meta.setdefault("nested_region_config", nested_config)


def _recursive_pre_grad_passes(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
) -> GraphModule:
    with dynamo_timed(
        "_recursive_pre_grad_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="pre_grad_pass_time_us",
    ):
        _propagate_invoke_subgraph_nested_region_config(gm)
        with _patch_nested_region_inductor_config(gm):
            if not config.use_pre_grad_passes:
                return gm

            add_passes = config.add_pre_grad_passes
            remove_passes = config.remove_pre_grad_passes
            for subgraph_name in _get_subgraph_names(gm):
                subgraph = getattr(gm, subgraph_name)
                # as we don't have recursive example inputs, passing empty set here
                new_subgraph = _recursive_pre_grad_passes(subgraph, ())
                setattr(gm, subgraph_name, new_subgraph)
            return pre_grad_passes(gm, example_inputs, add_passes, remove_passes)


def _recursive_joint_graph_passes(
    gm: GraphModule,
    skip_invoke_subgraph: bool = False,
    input_device: torch.device | None = None,
) -> GraphModule:
    def _run_on_sub_graph_module(subgraph_name: str) -> None:
        subgraph = getattr(gm, subgraph_name)
        new_subgraph = _recursive_joint_graph_passes(
            subgraph, skip_invoke_subgraph, input_device
        )
        setattr(gm, subgraph_name, new_subgraph)

    with dynamo_timed(
        "_recursive_joint_graph_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="joint_graph_pass_time_us",
    ):
        _propagate_invoke_subgraph_nested_region_config(gm)
        with _patch_nested_region_inductor_config(gm):
            if not config.use_joint_graph_passes:
                return gm

            # invoke_subgraph already runs the _recursive_joint_graph_passes.  In
            # AOTAutograd, `run_joint_graph_passes_on_hops` partitions the
            # invoke_subgraph HOP before calling the partitioner on the outer graph.
            # AOTAutograd has access to partition_fn, which internally calls the
            # `_recursive_joint_graph_passes` for the subgraph. So, skip recursing
            # skip_invoke_subgraph.
            old_subgraph_names = OrderedSet(
                _get_subgraph_names(gm, skip_invoke_subgraph)
            )
            for subgraph_name in old_subgraph_names:
                _run_on_sub_graph_module(subgraph_name)

            out_gm = joint_graph_passes(gm, input_device)

            # Some joint graph passes may create new sub graph module. Run one round
            # for the newly created graph modules.
            # We should not skip graphs for invoke_subgraph HOPs for newly
            # generated subgraphs.
            for subgraph_name in _get_subgraph_names(
                out_gm, skip_invoke_subgraph=False
            ):
                if subgraph_name not in old_subgraph_names:
                    _run_on_sub_graph_module(subgraph_name)
            return out_gm


def _recursive_post_grad_passes(gm: GraphModule, is_inference: bool = False) -> None:
    with dynamo_timed(
        "_recursive_post_grad_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="post_grad_pass_time_us",
    ):
        _propagate_invoke_subgraph_nested_region_config(gm)
        with _patch_nested_region_inductor_config(gm):
            if not config.use_post_grad_passes:
                return

            for subgraph_name in _get_subgraph_names(gm):
                subgraph = getattr(gm, subgraph_name)
                _recursive_post_grad_passes(subgraph, is_inference)
            post_grad_passes(gm, is_inference)


def split_const_gm(
    gm: GraphModule,
    skip_constructor: bool = True,
    lifted_constant_names: list[str] | None = None,
    skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = None,
) -> tuple[GraphModule, dict[str, int]]:
    """
    This function takes an GraphModule input "gm".
    The gm will be split into 2 components,
      1) const_gm, which consists the subgraph of gm that can be constant folded.
      2) gm (being inplace modified,) which returns the graph after constant folding.

    If an additional "lifted_constants" argument is passed in, we will assume the gm has
    been lifted and run the transformation accordingly.

    When a "skip_folding_node_fn" callback is passed, we will skip constant folding on
    the nodes for which the callback returns True.

    const_output_index is a mapping of corresponding node name from gm to the
    output index of const_gm.
    Returns (const_gm, const_output_index)
    """
    from torch._inductor.constant_folding import (
        CONST_MODULE_TAG,
        META_TAG,
        MODULE_TAG,
        replace_node_with_constant,
        run_and_get_constant_graph,
    )

    const_gm = run_and_get_constant_graph(
        gm, skip_constructor, lifted_constant_names, skip_folding_node_fn
    )
    const_result = const_gm() if lifted_constant_names is None else None

    const_outputs = {
        x.name: idx for idx, x in enumerate(tuple(const_gm.graph.nodes)[-1].args[0])
    }

    to_erase_node = []
    to_replace_node = []
    const_output_index = {}
    for node in gm.graph.nodes:
        if node.name in const_outputs:
            to_replace_node.append(node)
        elif node.meta[META_TAG] == CONST_MODULE_TAG and node.op != "placeholder":
            to_erase_node.append(node)

    for node in to_replace_node:
        new_const_name = "_FOLDED_CONST_" + node.name
        replace_node_with_constant(
            gm,
            node,
            (
                const_result[const_outputs[node.name]]  # type:ignore[index]
                if lifted_constant_names is None
                else None
            ),
            new_const_name,
        )
        const_output_index[new_const_name] = const_outputs[node.name]
    for node in to_erase_node[::-1]:
        if node.users:
            for n in node.users:
                if n.meta[META_TAG] != MODULE_TAG:
                    raise AssertionError(f"node: {node} user not empty.")
        else:
            gm.graph.erase_node(node)
    gm.recompile()

    return const_gm, const_output_index


def is_tf32_warning_applicable(gm: GraphModule) -> bool:
    aten = torch.ops.aten
    tf32_ops = OrderedSet(
        [
            aten.mm.default,
            aten.addmm.default,
            aten.bmm.default,
            aten.baddbmm.default,
        ]
    )
    for target in tf32_ops:
        for node in gm.graph.find_nodes(op="call_function", target=target):
            if (
                isinstance(node.meta.get("val", None), torch.Tensor)
                and node.meta["val"].dtype == torch.float32
                and node.meta["val"].device.type == "cuda"
            ):
                return True
    return False


def maybe_disable_comprehensive_padding(
    example_inputs: Sequence[InputType],
) -> AbstractContextManager[None, None]:
    """
    For CPU backend, enable comprehensive padding causes some unit tests
    fail due to changing number of generated kernels. Skip for now.
    """
    has_gpu = any(
        is_gpu(t.device.type) for t in example_inputs if isinstance(t, torch.Tensor)
    )

    if config.disable_padding_cpu and config.comprehensive_padding and not has_gpu:
        perf_hint_log.info("Skip comprehensive padding on CPU")
        return config.patch(comprehensive_padding=False)
    elif config.aot_inductor.use_runtime_constant_folding:
        perf_hint_log.info(
            "Skip comprehensive padding for use_runtime_constant_folding"
        )
        return config.patch(comprehensive_padding=False)
    else:
        return contextlib.nullcontext()


def maybe_disable_graph_partition(
    cpp_wrapper: bool, aot_mode: bool
) -> AbstractContextManager[None, None]:
    """
    graph partition does not support cpp_wrapper and aot_mode yet.
    """
    if cpp_wrapper or aot_mode:
        return config.patch(graph_partition=False)
    else:
        return contextlib.nullcontext()


def fake_tensor_prop(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    force_allow_non_fake_inputs: bool = False,
) -> torch._subclasses.FakeTensorMode:
    """
    If we can not detect fake mode from the context of inputs, create one.

    The created fake mode will be returned.
    """
    # Ensure that decomps that support symbolic shapes are used
    with enable_python_dispatcher():
        fake_mode = detect_fake_mode(example_inputs)
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
            FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
        else:
            ctx = (
                contextlib.nullcontext()
                if not force_allow_non_fake_inputs
                else mock.patch.object(fake_mode, "allow_non_fake_inputs", True)
            )
            with ctx:  # type: ignore[attr-defined]
                FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
                    *example_inputs
                )

    return fake_mode


# pass config dict back to user
def get_patched_config_dict(
    config_patches: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    with config.patch(config_patches):
        return config.get_config_copy()


@contextlib.contextmanager
def with_fresh_cache_if_config() -> Generator[None, None, None]:
    if config.force_disable_caches:
        # Don't delete the cache dir because it has to survive beyond the
        # compile_fx call. Let's put the temp dirs under the default cache
        # dir so they're easier to locate.
        with fresh_cache(dir=cache_dir(), delete=False):
            yield
    else:
        yield


class _CompileFxKwargs(TypedDict, total=False):
    cudagraphs: BoxedBool | None
    static_input_idxs: Sequence[int]
    is_backward: bool
    graph_id: int | None
    cpp_wrapper: bool
    aot_mode: bool
    is_inference: bool
    layout_opt: bool | None
    extern_node_serializer: Callable[[list[ExternKernelNode]], Any] | None
    boxed_forward_device_index: BoxedDeviceIndex | None
    fx_wrapper: bool
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]]


class _CompileFxCallable(Protocol):
    def __call__(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        compile_region_name: str | None = None,
        **kwargs: Unpack[_CompileFxKwargs],
    ) -> OutputCode: ...


def compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    compile_region_name: str | None = None,
    **kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode:
    kwargs.setdefault("cudagraphs", None)
    kwargs.setdefault("static_input_idxs", ())
    kwargs.setdefault("is_backward", False)
    kwargs.setdefault("graph_id", None)
    kwargs.setdefault("cpp_wrapper", False)
    kwargs.setdefault("fx_wrapper", False)
    kwargs.setdefault("is_inference", False)
    kwargs.setdefault("boxed_forward_device_index", None)
    kwargs.setdefault("layout_opt", None)
    kwargs.setdefault("extern_node_serializer", None)

    # Need with_fresh_cache_if_config for compile_fx_inner even if we already have one for
    # compile_fx. The reason is the compilation for backward graph may happen after
    # compile_fx return and we may want to use the _LazyGraphModule for compiling
    # the backward graph as well.
    with contextlib.ExitStack() as stack:
        # When cpp_wrapper is enabled, ensure the required triton config
        # (store_cubin, autotune_at_compile_time, etc.) is applied. This is
        # needed because lazy backward compilation may run after the
        # config.patch context from compile_fx has already exited.
        # Suppress cudagraph skip logging here; compile_fx already logged it.
        if kwargs["cpp_wrapper"]:
            stack.enter_context(
                config.patch(get_cpp_wrapper_config(log_cudagraph_skip=False))
            )
        # Host-side TMA only selects the descriptor flavor; it needs the TMA path
        # itself enabled. Warn (don't silently no-op) if it's set without its
        # prerequisites.
        if config.triton.enable_host_side_tma and not (
            config.triton.use_tensor_descriptor and config.assume_aligned_inputs
        ):
            warnings.warn(
                "config.triton.enable_host_side_tma has no effect unless both "
                "config.triton.use_tensor_descriptor and "
                "config.assume_aligned_inputs are also enabled; host-side TMA "
                "will be skipped.",
                stacklevel=2,
            )
        stack.enter_context(torch.utils._python_dispatch._disable_current_modes())
        stack.enter_context(_use_lazy_graph_module(dynamo_config.use_lazy_graph_module))
        stack.enter_context(
            dynamo_utils.dynamo_timed(
                "compile_fx_inner",
                phase_name="inductor_compile",
                log_pt2_compile_event=True,
                log_waitcounter=True,
                waitcounter_name_override="compile_inductor",
                dynamo_compile_column_us="inductor_cumulative_compile_time_us",
            )
        )
        stack.enter_context(with_fresh_cache_if_config())
        stack.enter_context(DebugContext())
        CompileEventLogger.pt2_compile(
            "inductor_compile",
            is_backward=kwargs["is_backward"],
        )
        compiled_graph = wrap_compiler_debug(
            _compile_fx_inner, compiler_name="inductor"
        )(
            gm,
            example_inputs,
            compile_region_name=compile_region_name,
            **kwargs,
        )
        if config.trace.provenance_tracking_to_timeline:
            compile_id = torch._guards.CompileContext.current_compile_id()
            kernel_information_jsons = get_kernel_information_jsons()
            key = str((f"Torch-Compiled Region: {compile_id}", kwargs["is_backward"]))
            if key not in kernel_information_jsons:
                kernel_information_jsons[key] = create_kernel_information_json()
        return compiled_graph


@time_and_log(attr="compilation time (in seconds)")
def _compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    compile_region_name: str | None = None,
    **graph_kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode:
    """
    Inductor API that compiles a single graph.

    If you change the argument list for this function, make sure you
    also update the call to save_args_for_compile_fx_inner below accordingly.
    """
    aot_mode: bool = V.aot_compilation

    from torch._inductor.autotune_process import use_pipelined_autotuning

    if use_pipelined_autotuning():
        # Warm up max-autotune process pool asap
        from torch._inductor.autotune_process import AutotuneProcessPool

        pool_instance = AutotuneProcessPool.get_instance()
        pool_instance.warm_up()

    # Clean up Compiled Triton Kernels per inductor compile, as the future objects
    # may not be valid for use after they are run/autotuned
    torch._inductor.async_compile.CompiledTritonKernels.cache_clear()

    if (
        dynamo_utils.count_calls(gm.graph) == 0
        and not aot_mode
        and not torch._functorch.config.bundled_autograd_cache
    ):
        # trigger the real recompilation for _LazyGraphModule before returning
        # the forward method.
        from torch._dynamo.utils import CompileEventLogLevel
        from torch.fx._lazy_graph_module import _LazyGraphModule

        _LazyGraphModule.force_recompile(gm)
        compile_id = torch._guards.CompileContext.current_compile_id()
        CompileEventLogger.log_instant_event(
            "backward no-op",
            metadata={"compile_id": compile_id},
            log_level=CompileEventLogLevel.PT2_COMPILE,
        )

        # pyrefly: ignore[bad-return]
        return make_boxed_func(gm.forward)

    static_input_idxs: Sequence[int] = graph_kwargs.setdefault("static_input_idxs", ())
    static_inputs_log.debug("static input idxs compile_fx_inner: %s", static_input_idxs)
    inputs_to_check = get_input_idxs_to_check(example_inputs, static_input_idxs)

    if not isinstance(next(iter(reversed(gm.graph.nodes))).args[0], (tuple, list)):
        raise AssertionError(
            f"inductor can only compile FX graphs which return a tuple/list, but got {gm.graph}"
        )

    _propagate_invoke_subgraph_nested_region_config(gm)

    if graph_kwargs.get("cudagraphs") is None:
        graph_kwargs["cudagraphs"] = BoxedBool(config.triton.cudagraphs)
    if config.save_args:
        save_args_for_compile_fx_inner(
            gm,
            example_inputs,
            compile_region_name=compile_region_name,
            **graph_kwargs,
        )

    start = time.time()

    fx_graph_remote_cache = should_use_remote_fx_graph_cache()

    # Check if the registered backend(s) support caching.
    init_backend_registration()
    backends_support_caching = all(
        backend.supports_caching
        for backend in (
            get_wrapper_codegen_for_device(
                device.type, config.cpp_wrapper, config.fx_wrapper
            )
            for device in get_all_devices(gm)
        )
        if backend is not None
    )

    with dynamo_timed(
        "fx_codegen_and_compile", log_pt2_compile_event=True, log_waitcounter=True
    ):
        use_cache = (
            not config.force_disable_caches
            and (config.fx_graph_cache or fx_graph_remote_cache)
            and not aot_mode
            and backends_support_caching
            and not torch._functorch.config.bundled_autograd_cache
        )
        local = config.fx_graph_cache
        remote = fx_graph_remote_cache
        set_feature_use("fx_cache", use_cache)

        log.debug(
            "FX cache status: use_cache=%s, local=%s, remote=%s, aot_mode=%s, force_disable_caches=%s",
            use_cache,
            local,
            remote,
            aot_mode,
            config.force_disable_caches,
        )

        # TODO: This is a hack purely to get some info to extract_tensor_metadata_for_cache_key,
        # figure out how to not have to modify example inputs
        for i, input in enumerate(example_inputs):
            if (
                isinstance(input, torch.Tensor)
                and is_gpu(input.device.type)
                and i in static_input_idxs
            ):
                input._is_inductor_static = True  # type: ignore[attr-defined]

        mb_compiled_graph: OutputCode | None = None
        key_info = None
        cache_info: CacheInfo | None = None
        remote_cache = None
        constants = CompiledFxGraphConstantsWithGm(gm)
        # TODO: this time will be slightly inconsistent with the one computed
        # in prepare_key/load_with_key, dump those settings of "cache_event_time"
        start_time = time.time_ns()

        if use_cache:
            (key_info, cache_info) = FxGraphCache.prepare_key(
                gm, example_inputs, graph_kwargs, inputs_to_check, remote
            )

            # Attempt a cache lookup
            if key_info is not None:
                key, debug_lines = key_info
                log.debug("FX cache key generated: %s", key)
                if remote:
                    remote_cache = FxGraphCache.get_remote_cache()
                    log.debug("Using remote FX cache")
                mb_compiled_graph, cache_info = FxGraphCache.load_with_key(
                    key,
                    debug_lines,
                    example_inputs,
                    local,
                    remote_cache,
                    is_backward=graph_kwargs.get("is_backward", False),
                    constants=constants,
                )
            else:
                log.debug("Failed to generate FX cache key")

        if torch._functorch.config.bundled_autograd_cache:
            if mb_compiled_graph is not None:
                raise AssertionError(
                    "Expected mb_compiled_graph to be None for bundled_autograd_cache path"
                )
            if cache_info is not None:
                raise AssertionError(
                    "Expected cache_info to be None for bundled_autograd_cache path"
                )
            # When using bundled autograd cache, we still want
            # to use the TritonBundler, but we don't want to save
            # the results here. The results will get saved directly
            # to AOTAutogradCache.
            TritonBundler.begin_compile()
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm,
                    example_inputs,
                    inputs_to_check,
                    compile_region_name=compile_region_name,
                    **graph_kwargs,
                )
                if mb_compiled_graph is None:
                    raise AssertionError(
                        "fx_codegen_and_compile returned None in bundled_autograd_cache path"
                    )
                (
                    triton_bundle,
                    triton_bundler_meta,
                ) = TritonBundler.collect()
                mb_compiled_graph.set_triton_bundle(triton_bundle)
            except (ShortenTraceback, SkipFrame):
                raise
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None
            finally:
                TritonBundler.end_compile()

        # CACHE BYPASS: Compile the graph, don't save it to the cache
        # (this can happen either because cache was disabled, or we
        # determined the input is uncacheable)
        elif cache_info is None or cache_info["cache_state"] == "bypass":
            if mb_compiled_graph is not None:
                raise AssertionError(
                    "Expected mb_compiled_graph to be None in cache bypass path"
                )
            log.debug(
                "FX cache bypass reason: %s",
                (
                    cache_info.get("cache_bypass_reason", "unknown")
                    if cache_info is not None
                    else "FX cache disabled or key generation failed"
                ),
            )
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm,
                    example_inputs,
                    inputs_to_check,
                    compile_region_name=compile_region_name,
                    **graph_kwargs,
                )
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None

        # CACHE MISS: Compile the graph and save to cache
        elif cache_info["cache_state"] == "miss":
            if mb_compiled_graph is not None:
                raise AssertionError(
                    "Expected mb_compiled_graph to be None in cache miss path"
                )
            if key_info is None:
                raise AssertionError("Expected key_info to be set in cache miss path")
            log.debug("FX cache miss, compiling and saving to cache")
            TritonBundler.begin_compile()
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm,
                    example_inputs,
                    inputs_to_check,
                    compile_region_name=compile_region_name,
                    **graph_kwargs,
                )
                if mb_compiled_graph is None:
                    raise AssertionError(
                        "fx_codegen_and_compile returned None in cache miss path"
                    )
                mb_compiled_graph._time_taken_ns = time.time_ns() - start_time
                cache_key, debug_lines = key_info
                mb_compiled_graph._fx_graph_cache_key = cache_key
                mb_compiled_graph._fx_graph_cache_debug_lines = debug_lines
                (
                    triton_bundle,
                    triton_bundler_meta,
                ) = TritonBundler.collect()
                mb_compiled_graph.set_triton_bundle(triton_bundle)
            except (ShortenTraceback, SkipFrame):
                raise
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None
            finally:
                TritonBundler.end_compile()
            if triton_bundler_meta is not None:
                cache_info["triton_bundler_meta"] = str(triton_bundler_meta)
            cache_info["time_taken_ns"] = mb_compiled_graph._time_taken_ns
            log.debug("Saving compiled graph to FX cache with key: %s", cache_key)
            FxGraphCache._save_graph(
                cache_key,
                mb_compiled_graph,
                example_inputs,
                local,
                remote_cache,
            )

        # CACHE HIT: not much to really do, just make sure the cache key
        # is recorded on the graph
        else:
            if cache_info["cache_state"] != "hit":
                raise AssertionError(
                    f"Expected cache_state to be 'hit', got {cache_info['cache_state']!r}"
                )
            if mb_compiled_graph is None:
                raise AssertionError(
                    "Expected mb_compiled_graph to be set in cache hit path"
                )
            if key_info is None:
                raise AssertionError("Expected key_info to be set in cache hit path")
            (cache_key, debug_lines) = key_info
            log.debug("FX cache hit with key: %s", cache_key)
            mb_compiled_graph._fx_graph_cache_key = cache_key
            mb_compiled_graph._fx_graph_cache_debug_lines = debug_lines

        if mb_compiled_graph is None:
            raise AssertionError(
                "Expected mb_compiled_graph to be set after cache handling"
            )
        compiled_graph = mb_compiled_graph
        if isinstance(compiled_graph, CompiledFxGraph):
            compiled_graph.compile_region_name = compile_region_name

        # Logging and observability: we log a single chromium event
        # and a tlparse log for every cache action.
        # In the event of a bypass, we also logged to the remote table earlier
        # with log_cache_bypass.
        cache_state = (
            cache_info["cache_state"] if cache_info is not None else "disabled"
        )
        # Here for grepping:
        # fx_graph_cache_hit
        # fx_graph_cache_miss
        # fx_graph_cache_bypass
        # fx_graph_cache_disabled
        cache_event_metadata = dict(cache_info) if cache_info else {}
        CompileEventLogger.instant(
            f"fx_graph_cache_{cache_state}",
            metadata=cache_event_metadata,
            time_ns=start_time,
        )
        # Add event data about cache hits/miss
        # TODO: add remote cache get/put timings here too
        CompileEventLogger.try_add_pt2_compile(
            "inductor_compile",
            cache_state=cache_state,
            cache_event_time=start_time,
            key=cache_info.get("key") if cache_info else None,
            components=cache_info.get("components") if cache_info else None,
            cache_bypass_reason=(
                cache_info.get("cache_bypass_reason")
                if cache_info
                else "cache not enabled"
            ),
            remote_cache_enabled=remote,
            local_cache_enabled=local,
        )

        # Don't clog up the main tlparse output with disabled cache
        if cache_info is not None:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": f"fx_graph_cache_{cache_state}",
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )
        compiled_graph.post_compile(example_inputs, constants, graph_kwargs)

        policy = config.cudagraph_policy
        if policy is not None:
            compiled_graph = policy.wrap_output(compiled_graph)

    log.debug("FX codegen and compilation took %.3fs", time.time() - start)

    # This message is for printing overview information of inductor mm counts, shapes,etc after lowering
    if log.isEnabledFor(logging.INFO):
        mm_table_data = []
        for key, value in counters["aten_mm_info"].items():
            parts = key.split("_")
            if len(parts) < 3:
                # Unexpected format, show as-is
                mm_table_data.append([key, "-", "?", "?", "?", value])
                continue

            # Determine if this is a batched operation by checking the operation name
            name = "_".join(parts[:-4]) if len(parts) >= 4 else "_".join(parts[:-3])
            is_batched = name.endswith(("bmm", "baddbmm"))

            if is_batched and len(parts) >= 4:
                # Batched operation: last 4 parts are batch, m, n, k
                batch, m, n, k = parts[-4:]
                name = "_".join(parts[:-4])
                mm_table_data.append([name, batch, m, n, k, value])
            else:
                # Non-batched operation: last 3 parts are m, n, k
                m, n, k = parts[-3:]
                name = "_".join(parts[:-3])
                mm_table_data.append([name, "-", m, n, k, value])

        log.info("Overview info of inductor aten mms: ")
        log.info(
            "{:<30} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(  # noqa: G001
                "Name", "B", "M", "N", "K", "Count"
            )
        )
        log.info("-" * 130)
        for row in mm_table_data:
            log.info("{:<30} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(*row))  # noqa: G001
            log.info("-" * 130)

    # Not strictly necessary, but good to clean up straggling futures
    # that are unused to reclaim memory.
    torch._inductor.async_compile.CompiledTritonKernels.cache_clear()

    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if graph_kwargs['is_backward'] else 'FORWARDS'} "
        f"graph {graph_kwargs['graph_id']}",
    )
    return compiled_graph


class _FxCompileStat:
    # Count of successful compiles of this type
    codegen_and_compile: int = 0

    def __repr__(self) -> str:
        return f"codegen_and_compile: {self.codegen_and_compile}"


class FxCompile(ABC):
    """
    An FxCompile represents a mechanism that can turn a GraphModule into an
    OutputCode.
    """

    # Some stats for logging/debugging
    _compile_stats: dict[type[FxCompile], _FxCompileStat] = defaultdict(_FxCompileStat)
    compile_region_name: str | None = None

    # TODO: We should probably eventually add some kind of async version of this
    # so we can kick off a compile and then go do other things - but we'll need
    # to know what kind of API we want for that first.
    @abstractmethod
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...

    @classmethod
    def _reset_stats(cls) -> None:
        cls._compile_stats.clear()


class _InProcessFxCompile(FxCompile):
    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        """
        Generates the OutputCode from the GraphModule and example_inputs.
        """
        # Sorry about the mess, we need graph_kwargs to continue to be able
        # to propagate it further on
        # TODO: _CompileFxKwargs actually has stronger types than in the
        # signature, need to tighten it up

        if "cudagraphs" not in graph_kwargs or graph_kwargs["cudagraphs"] is None:
            raise AssertionError("Expected 'cudagraphs' to be set in graph_kwargs")
        cudagraphs: BoxedBool = graph_kwargs["cudagraphs"]
        static_input_idxs: Sequence[int] = graph_kwargs.get("static_input_idxs", ())
        is_backward: bool = graph_kwargs.get("is_backward", False)
        graph_id: int | None = graph_kwargs.get("graph_id", None)
        cpp_wrapper: bool = graph_kwargs.get("cpp_wrapper", False)
        fx_wrapper: bool = graph_kwargs.get("fx_wrapper", False)
        aot_mode: bool = V.aot_compilation
        is_inference: bool = graph_kwargs.get("is_inference", False)
        extern_node_serializer: Callable[[list[ExternKernelNode]], Any] | None = (
            graph_kwargs.get("extern_node_serializer", None)
        )
        get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = graph_kwargs.get(
            "get_decomp_fn", select_decomp_table
        )
        with (
            _WaitCounter("pytorch.wait_counter.actual_codegen_and_compile").guard(),
            dynamo_utils.preserve_rng_state(),
        ):
            if (sleep_sec := config.sleep_sec_TESTING_ONLY) is not None:
                import time

                log.warning(
                    "Sleeping for %s since sleep_sec_TESTING_ONLY is set", sleep_sec
                )
                time.sleep(sleep_sec)

            if is_tf32_warning_applicable(gm):
                _warn_tf32_disabled()

            inductor_counters = counters["inductor"].copy()

            # lift the maximum depth of the Python interpreter stack
            # to adapt large/deep models
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

            _step_logger()(
                logging.INFO,
                "torchinductor compiling "
                f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
                f"graph {graph_id}",
            )

            # Building this parses the source of every Triton kernel in the
            # graph, and it is only ever used as a structured-trace artifact:
            # logged here, and logged again by the FX graph cache when this
            # graph is loaded from it. Produce it from payload_fn so that
            # trace_structured's own "is anyone listening" check decides
            # whether the work happens at all, and keep what it produced for
            # the cache to replay.
            produced: list[str] = []

            def _fx_graph_runnable_payload() -> str:
                fd = io.StringIO()
                torch._dynamo.repro.after_aot.save_graph_repro(
                    fd, gm, example_inputs, "inductor", save_dir=None
                )
                produced.append(fd.getvalue())
                return produced[0]

            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_runnable",
                    "encoding": "string",
                },
                payload_fn=_fx_graph_runnable_payload,
            )
            runnable_graph_str = produced[0] if produced else ""

            V.debug.fx_graph(gm, example_inputs)
            # TODO: Should we actually dump this?  It should be redundant with the aot
            # structured logs...
            # trace_structured("inductor_input_graph", payload_fn=lambda: gm.print_readable(print_output=False))

            shape_env = gm.shape_env
            if shape_env is None:
                shape_env = shape_env_from_inputs(example_inputs)

            # Convert view to reshape in the graph. This is necessary primarily for
            # layout optimization. Do it unconditionally for uniformity.
            #
            # It's needed because when we do layout optimization, a contiguous tensor
            # in eager mode may becomes a channels last tensor. A view op previously
            # can be applied to the contiguous tensor may not be able to be applied
            # on the channels tensor any more. An error like
            #   RuntimeError: view size is not compatible with input tensor's size and stride
            #   (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            # will be printed.
            #
            # Replace view op to reshape op in this case.
            # As an example, timm_resnest/botnet26t_256/convnext_base etc. will fail if we don't do this.
            #
            # Also this has to be done before FakeTensorProp below to avoid the failed
            # .view() call.
            view_to_reshape(gm)

            with dynamo_timed(
                "additional_fake_tensor_prop", log_pt2_compile_event=True
            ):
                # It is safe to run FakeTensorProp under no_grad because by the time
                # we're in inductor, we assume that AOTAutograd has already "taken care"
                # of autograd, so there should be no more autograd-related API's in the
                # graph.
                with torch.no_grad():
                    fake_mode = fake_tensor_prop(gm, example_inputs)

            _recursive_record_original_output_strides(gm)

            # pattern matcher passes might not preserve striding information
            # on node.meta["val"]. if in the future we rely on these being
            # correct we will need to fix.
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "before_post_grad_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: gm.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            with V.set_fake_mode(fake_mode):
                # has some issues with memory in training
                cuda_context = get_cuda_device_context(gm)
                with cuda_context:
                    _recursive_post_grad_passes(gm, is_inference=is_inference)
                V.debug.fx_graph_transformed(gm, example_inputs)
                post_grad_graphs_log.debug(
                    "%s",
                    lazy_format_graph_code(
                        "AFTER POST GRAD",
                        gm,
                        include_stride=True,
                        include_device=True,
                        colored=True,
                    ),
                )

                # We're printing the graph to be used as a cache key - so a
                # printer which is a little less readable but faster is
                # appropriate.
                inductor_post_grad_graph_str = gm.print_readable(
                    print_output=False,
                    include_stride=True,
                    include_device=True,
                    fast_sympy_print=True,
                )
                # "inductor_post_grad_graph" is used in inductor provenance
                # tracking highlighter front-end.
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "inductor_post_grad_graph",
                        "encoding": "string",
                    },
                    payload_fn=lambda: inductor_post_grad_graph_str,
                )
                if config.effective_provenance_tracking_level() != 0:
                    provenance_tracking_json = (
                        torch.fx.traceback.get_graph_provenance_json(gm.graph)
                    )
                    torch._inductor.debug._inductor_post_to_pre_grad_nodes = (
                        create_mapping_pre_post_grad_nodes(
                            torch._inductor.debug._pre_grad_graph_id,
                            provenance_tracking_json,
                        )
                    )

                metrics_context = get_metrics_context()
                if metrics_context.in_progress():
                    num_graph_breaks = counters["graph_break"].total()
                    CompileEventLogger.compilation_metric(
                        overwrite=True, num_graph_breaks=num_graph_breaks
                    )
                if config.is_fbcode():
                    try:
                        log_optimus_to_scuba(
                            extra_logging={
                                "pt2_configs": str(get_patched_config_dict())
                            }
                        )
                    except Exception:
                        # TODO(T216453900): need to work around for now to support vllm
                        # See details in vllm/compilation/pass_manager.py.
                        log.warning("failed to log pt2_configs")

            with (
                V.set_fake_mode(fake_mode),
                maybe_disable_comprehensive_padding(example_inputs),
                maybe_disable_graph_partition(cpp_wrapper, aot_mode),
            ):
                const_output_index = None
                const_graph = None
                const_wrapper_code = None
                const_kernel_code = None

                if aot_mode and config.aot_inductor.use_runtime_constant_folding:
                    # torchbind objects have name that starts with _torchbind_obj
                    # See caffe2/torch/fx/_symbolic_trace.py?lines=406
                    const_gm, const_output_index = split_const_gm(
                        gm,
                        skip_folding_node_fn=lambda node: node.op == "get_attr"
                        and isinstance(node.target, str)
                        and (
                            node.target.startswith("_torchbind_obj")
                            or isinstance(node.meta.get("val", None), FakeScriptObject)
                        ),
                    )

                    const_graph = GraphLowering(
                        const_gm,
                        example_inputs=[],
                        shape_env=shape_env,
                        graph_id=graph_id,
                        cpp_wrapper=cpp_wrapper,
                        aot_mode=aot_mode,
                        extern_node_serializer=extern_node_serializer,
                        is_inference=is_inference,
                        is_backward=is_backward,
                        is_const_graph=True,
                        fx_wrapper=fx_wrapper,
                        get_decomp_fn=get_decomp_fn,
                    )
                    with (
                        V.set_graph_handler(const_graph),
                        V.set_extern_kernel_nodes([]),
                    ):
                        if not cpp_wrapper:
                            raise AssertionError("AOT mode only supports C++ wrapper")
                        const_graph.run()
                        const_wrapper_code, const_kernel_code = (
                            const_graph.codegen_with_cpp_wrapper()
                        )

                graph = GraphLowering(
                    gm,
                    # example_inputs will be used by AOTInductor to dry-run the generated code for Triton kernel tuning.
                    # For the forward pass, we have the real inputs to be used as example_inputs. For the backward pass,
                    # we currently use fake tensors and defake them later.
                    example_inputs=example_inputs,
                    shape_env=shape_env,
                    graph_id=graph_id,
                    cpp_wrapper=cpp_wrapper,
                    aot_mode=aot_mode,
                    extern_node_serializer=extern_node_serializer,
                    is_inference=is_inference,
                    is_backward=is_backward,
                    const_output_index=const_output_index,
                    const_wrapper_code=(
                        const_wrapper_code.value if const_wrapper_code else None
                    ),
                    const_kernel_code=(
                        const_kernel_code.value if const_kernel_code else None
                    ),
                    const_module=const_graph,
                    inputs_to_check=inputs_to_check,
                    fx_wrapper=fx_wrapper,
                    get_decomp_fn=get_decomp_fn,
                )
                metrics_helper = metrics.CachedMetricsHelper()

                # We are going to start code generating runtime asserts, so make sure
                # you don't start adding new ones in the lowering process
                graph.freeze_runtime_asserts()
                with (
                    V.set_graph_handler(graph),
                    V.set_extern_kernel_nodes([]),
                    distributed_autotune.graph_context(),
                ):
                    graph.run(*example_inputs)
                    output_strides: list[tuple[_StrideExprStr, ...] | None] = []
                    if graph.graph_outputs is not None:
                        # We'll put the output strides in the compiled graph so we
                        # can later return them to the caller via TracingContext
                        p = SymExprPrinter()
                        for out in graph.graph_outputs:
                            if (
                                isinstance(out, IRNode)
                                and out.has_tensor_output()
                                and len(free_unbacked_symbols(out.get_stride())) == 0
                            ):
                                # Convert to string for eval on the load path
                                output_strides.append(
                                    tuple(p.doprint(s) for s in out.get_layout().stride)
                                )
                            else:
                                output_strides.append(None)

                    _check_triton_bf16_support(graph)

                    # TODO: The switching between AOT mode and not here is a bit
                    # messy, but it's localized to the block of code below so I'm
                    # not going to touch it for now

                    compiled_fn: Any
                    compiled_fn_runner: CompiledFnRunner | None = None
                    with dynamo_timed(
                        "GraphLowering.compile_to_fn", log_pt2_compile_event=True
                    ):
                        if graph.aot_mode and graph.fx_wrapper:
                            if graph.cpp_wrapper:
                                raise AssertionError(
                                    "Expected cpp_wrapper to be False for fx_wrapper AOT mode"
                                )
                            compiled_fn = graph.codegen()[0].gm  # type: ignore[attr-defined]
                            output_code_log.debug(
                                "Output graph module: \n%s",
                                compiled_fn.print_readable(print_output=False),
                            )

                        elif graph.aot_mode:
                            from .codecache import AotCodeCompiler

                            if not graph.cpp_wrapper:
                                raise AssertionError(
                                    "AOT mode only supports C++ wrapper"
                                )
                            wrapper_code, kernel_code = graph.codegen_with_cpp_wrapper()
                            output_code_log.debug(
                                "AOT wrapper code: \n%s", wrapper_code.value
                            )
                            if kernel_code.value:
                                output_code_log.debug(
                                    "AOT kernel code:\n%s", kernel_code.value
                                )

                            serialized_extern_kernel_nodes = None
                            if V.extern_kernel_nodes:
                                serialized_extern_kernel_nodes = (
                                    graph.extern_node_serializer(V.extern_kernel_nodes)
                                )
                                output_code_log.debug(
                                    "Serialized Extern Kernel Nodes: \n%s",
                                    serialized_extern_kernel_nodes,
                                )

                            with dynamo_timed(
                                "AotCodeCompiler.compile", log_pt2_compile_event=True
                            ):
                                # Directly return the file path with the compiled code
                                compiled_fn = AotCodeCompiler.compile(
                                    graph,
                                    wrapper_code.value,
                                    kernel_code.value,
                                    serialized_extern_kernel_nodes,
                                    device_type=graph.device_type,
                                    additional_files=[
                                        *dict.fromkeys(
                                            graph.wrapper_code.additional_files
                                            + (
                                                const_graph.wrapper_code.additional_files
                                                if const_graph
                                                else []
                                            )
                                        )
                                    ],
                                )
                        else:
                            compiled_module = graph.compile_to_module()
                            compiled_fn = compiled_module.call
                            compiled_fn_runner = getattr(
                                compiled_module, "runner", None
                            )

                    # Dump provenance artifacts for debugging trace
                    inductor_provenance_tracking_node_mappings = None
                    inductor_kernel_stack_trace_str = None
                    if config.effective_provenance_tracking_level() != 0:
                        inductor_provenance_tracking_node_mappings = json.dumps(
                            torch._inductor.debug.dump_inductor_provenance_info()
                        )
                        inductor_kernel_stack_trace_str = json.dumps(
                            torch._inductor.debug._inductor_kernel_stack_trace
                        )
                        trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "inductor_provenance_tracking_node_mappings",
                                "encoding": "json",
                            },
                            payload_fn=lambda: inductor_provenance_tracking_node_mappings,
                        )
                        trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "inductor_provenance_tracking_kernel_stack_traces",
                                "encoding": "json",
                            },
                            payload_fn=lambda: inductor_kernel_stack_trace_str,
                        )
                        if inductor_kernel_stack_trace_str:
                            metrics_context = get_metrics_context()
                            if metrics_context.in_progress():
                                metrics_context.add_to_set(
                                    "inductor_provenance",
                                    inductor_kernel_stack_trace_str,
                                )

                    node_runtimes = None
                    if inductor_metrics_log.isEnabledFor(logging.INFO):
                        num_bytes, nodes_num_elem, node_runtimes = graph.count_bytes()
                        # pyrefly: ignore [bad-assignment]
                        metrics.num_bytes_accessed += num_bytes
                        metrics.node_runtimes += node_runtimes
                        metrics.nodes_num_elem += nodes_num_elem
                        inductor_metrics_log.info(
                            "Graph Metrics:\n%s",
                            {
                                "num_bytes_accessed": num_bytes,
                                "nodes_num_elem": nodes_num_elem,
                                "node_runtimes": node_runtimes,
                            },
                        )

                    # Collect and dump op runtimes and tensor metadata for TLParse
                    if config.log_tlparse:
                        _, _, node_runtimes = graph.count_bytes()
                        torch._inductor.debug.log_runtime_and_tensor_meta(node_runtimes)

                    # Collect and dump collective-op schedule for external diagnostics
                    torch._inductor.debug.log_collective_schedule(graph.scheduler.nodes)

                    # When graph_partition is enabled, skip this check - partitioning handles dynamic shapes
                    if (
                        cudagraphs
                        and config.triton.cudagraph_skip_dynamic_graphs
                        and not config.graph_partition
                        and not V.graph.disable_cudagraphs_reason
                        and torch._inductor.utils.any_is_symbolic(*example_inputs)
                    ):
                        stack_trace = None
                        for node in gm.graph.nodes:
                            meta_val = node.meta.get("val", None)
                            if (
                                node.op == "placeholder"
                                or not isinstance(meta_val, torch.Tensor)
                                or not torch._inductor.utils.any_is_symbolic(meta_val)
                            ):
                                continue

                            if stack_trace := node.meta.get("stack_trace", None):
                                break
                        disable = "graph with symbolic shapes inputs and config.triton.cudagraph_skip_dynamic_graphs=True."
                        if stack_trace:
                            disable = f"{disable} Found from {stack_trace}\n"
                        else:
                            disable = f"{disable}\n"
                        # pyrefly: ignore [unbound-name]
                        V.graph.disable_cudagraphs_reason = disable

                    # pyrefly: ignore [unbound-name]
                    # When graph_partition is enabled, skip this check - partitioning handles incompatible ops
                    if (
                        cudagraphs
                        # pyrefly: ignore [unbound-name]
                        and not config.graph_partition
                        # pyrefly: ignore [unbound-name]
                        and not V.graph.disable_cudagraphs_reason
                    ):
                        maybe_incompat_node = get_first_incompatible_cudagraph_node(gm)
                        if maybe_incompat_node:
                            disable = f"disabling cudagraphs due to incompatible op {maybe_incompat_node.target}"
                            if stack_trace := maybe_incompat_node.meta.get(
                                "stack_trace", None
                            ):
                                disable = f"{disable} Found from {stack_trace}\n"
                            # pyrefly: ignore [unbound-name]
                            V.graph.disable_cudagraphs_reason = disable

                    # pyrefly: ignore [unbound-name]
                    if V.aot_compilation:
                        if not isinstance(
                            compiled_fn,
                            # pyrefly: ignore [unbound-name]
                            (str, list, torch.fx.GraphModule),
                        ):
                            raise AssertionError(
                                f"Expected compiled_fn to be str, list, or GraphModule, got {type(compiled_fn)}"
                            )
                        return CompiledAOTI(
                            filename=compiled_fn, device_type=graph.device_type
                        )

                    # TODO: Hoist this above V.aot_compilation
                    # pyrefly: ignore [unbound-name]
                    if cudagraphs and not V.graph.disable_cudagraphs_reason:
                        from torch._inductor.cudagraph_utils import (
                            check_lowering_disable_cudagraph,
                        )

                        # pyrefly: ignore [unbound-name]
                        V.graph.disable_cudagraphs_reason = (
                            check_lowering_disable_cudagraph(
                                # pyrefly: ignore [unbound-name]
                                V.graph.device_node_mapping
                            )
                        )

                    if (
                        cudagraphs
                        and not V.graph.disable_cudagraphs_reason
                        and graph.scheduler.count_kernel_nodes(graph.scheduler.nodes)
                        == 0
                    ):
                        V.graph.kernel_free_cudagraph = True

                    self._compile_stats[type(self)].codegen_and_compile += 1

                    if (
                        # pyrefly: ignore [unbound-name]
                        torch._inductor.debug.RECORD_GRAPH_EXECUTION
                        # pyrefly: ignore [unbound-name]
                        and torch._inductor.debug.GRAPH_COMPILE_IDS is not None
                    ):
                        compile_id = str(
                            # pyrefly: ignore [unbound-name]
                            torch._guards.CompileContext.current_compile_id()
                        )
                        graph_id = graph_kwargs.get("graph_id")
                        if graph_id is not None:
                            # pyrefly: ignore [unbound-name]
                            torch._inductor.debug.GRAPH_COMPILE_IDS[graph_id] = (
                                compile_id
                            )

                    return CompiledFxGraph(
                        compiled_fn,
                        graph,
                        gm,
                        output_strides,
                        # pyrefly: ignore [unbound-name]
                        V.graph.disable_cudagraphs_reason,
                        metrics_helper.get_deltas(),
                        counters["inductor"] - inductor_counters,
                        cudagraphs,
                        example_inputs,
                        static_input_idxs,
                        self.compile_region_name,
                        graph_kwargs,
                        inputs_to_check,
                        runnable_graph_str,
                        inductor_post_grad_graph_str,
                        compiled_fn_runner,
                        inductor_provenance_tracking_node_mappings,
                        inductor_kernel_stack_trace_str,
                    )


def fx_codegen_and_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    # This is derivable from the other inputs to this function, but we pass it
    # in explicitly because it's nontrivial to compute
    inputs_to_check: Sequence[int],
    compile_region_name: str | None = None,
    **graph_kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode:
    scheme: FxCompile

    if fx_compile_mode == FxCompileMode.NORMAL:
        scheme = _InProcessFxCompile()
    elif fx_compile_mode == FxCompileMode.SERIALIZE:
        from .compile_fx_ext import _DebugSerdeFxCompile

        scheme = _DebugSerdeFxCompile()
    elif fx_compile_mode == FxCompileMode.SUBPROCESS:
        from .compile_fx_subproc import _SubprocessFxCompile

        scheme = _SubprocessFxCompile()

    if fx_compile_async:
        from .compile_fx_async import _AsyncFxCompile
        from .compile_fx_ext import _OutOfProcessFxCompile

        # pyrefly: ignore [unbound-name]
        if not isinstance(scheme, _OutOfProcessFxCompile):
            raise AssertionError(
                "async is only valid with an out-of-process compile mode"
            )
        # pyrefly: ignore [unbound-name]
        scheme = _AsyncFxCompile(scheme)
        scheme._compile.compile_region_name = (
            compile_region_name  # pyrefly: ignore[attr-defined]
        )

    if fx_compile_progressive:
        from .compile_fx_async import _ProgressiveFxCompile
        from .compile_fx_ext import _OutOfProcessFxCompile

        # pyrefly: ignore [unbound-name]
        if not isinstance(scheme, _OutOfProcessFxCompile):
            raise AssertionError(
                "progressive is only valid with an out-of-process compile mode"
            )

        progression_configs = _get_progression_configs()

        # Use in-process compile for the fast version
        fast_scheme = _InProcessFxCompile()
        fast_scheme.compile_region_name = compile_region_name

        # pyrefly: ignore [unbound-name]
        scheme = _ProgressiveFxCompile(fast_scheme, scheme, progression_configs)
        scheme._optimized_compile.compile_region_name = (
            compile_region_name  # pyrefly: ignore[attr-defined]
        )

    scheme.compile_region_name = compile_region_name  # pyrefly: ignore[unbound-name]

    # pyrefly: ignore [unbound-name]
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)


def get_input_idxs_to_check(
    inputs: Sequence[InputType],
    static_input_idxs: Sequence[int],
) -> Sequence[int]:
    """
    This function runs at compile time, and generates a list of indices for which we
    might need to do a copy to preserve alignment requirements.
    """
    ids_to_check = []

    for i, input in enumerate(inputs):
        if not isinstance(input, torch.Tensor):
            # non-tensors don't need alignment
            continue
        if not is_gpu(input.device.type):
            # right now we only care for gpu tensors
            continue
        with maybe_get_suppress_shape_guards_ctx():
            # suppress guards so that tensor_is_aligned and should_assume_input_aligned
            # do not add guards on input's storage offset
            if i in static_input_idxs and tensor_is_aligned(input):
                continue
            if not should_assume_input_aligned(input):
                continue

        # if we get here, then
        # (a) our triton code assumes that the input is aligned
        # (b) we can't be sure ahead of time that the input will actually be aligned.
        # therefore, at runtime, we'll need to check that the input is aligned
        # (and if not, clone it to make it aligned.)
        ids_to_check.append(i)

    return ids_to_check


def cudagraphify(
    model: Callable[..., Any],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    stack_traces: list[str | None],
    is_backward: bool,
    is_inference: bool,
    constants: tuple[torch.Tensor, ...] = (),
    placeholders: Sequence[PlaceholderInfo] = (),
    mutated_input_idxs: tuple[int, ...] = (),
    kernel_free_cudagraph: bool = False,
    user_visible_output_idxs: tuple[int, ...] = (),
) -> Callable[..., Any]:
    from torch._inductor.cudagraph_trees import (
        cudagraphify_impl as new_cudagraphify_impl,
    )

    cudagraphify_fn: Callable[..., Any]
    if config.triton.cudagraph_trees:
        cudagraphify_fn = functools.partial(
            new_cudagraphify_impl,
            device_index=device_index,
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=constants,
            placeholders=placeholders,
            mutated_input_idxs=mutated_input_idxs,
            kernel_free_cudagraph=kernel_free_cudagraph,
            user_visible_output_idxs=user_visible_output_idxs,
            compile_id=torch._guards.CompileContext.current_compile_id(),
        )
    else:
        cudagraphify_fn = functools.partial(
            cudagraphify_impl,
            kernel_free_cudagraph=kernel_free_cudagraph,
        )

    thread_local = threading.local()

    def run(new_inputs: Sequence[InputType]) -> Any:
        compiled_fn = getattr(thread_local, "compiled_fn", None)
        if compiled_fn is None:
            with dynamo_utils.preserve_rng_state():
                compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)  # type: ignore[arg-type]
            thread_local.compiled_fn = compiled_fn
        return compiled_fn(new_inputs)  # type: ignore[arg-type]

    return run


def static_input(x: torch.Tensor) -> torch.Tensor:
    """
    Copy and input while preserving strides
    """
    return torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)


def index_expanded_dims_and_copy_(
    dst: torch.Tensor,
    src: torch.Tensor,
    expanded_dims: list[int],
) -> None:
    "Index into expanded dimensions of both dst and src then copy_"
    indexed_dst = index_expanded_dims(dst, expanded_dims)
    if torch._debug_has_internal_overlap(indexed_dst) != 0:
        # dst still self-overlaps after dropping expanded dims (rare). A regular
        # copy_ would have ambiguous semantics; bytewise-copy the strided extent
        # instead so dst's overlap pattern is reproduced verbatim.
        copy_strided_storage_(dst, src)
        return
    indexed_src = index_expanded_dims(src, expanded_dims)
    indexed_dst.copy_(indexed_src)


def cudagraphify_impl(
    model: Callable[..., Any],
    inputs: list[torch.Tensor],
    static_input_idxs: Sequence[int] = (),
    *,
    kernel_free_cudagraph: bool = False,
) -> Callable[[list[InputType]], Any]:
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    if kernel_free_cudagraph:
        return model

    check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)  # type: ignore[arg-type]
    # pyrefly: ignore [annotation-mismatch, redefinition]
    static_input_idxs: OrderedSet[int] = OrderedSet(
        remove_unaligned_input_idxs(inputs, static_input_idxs)  # type: ignore[arg-type]
    )
    copy_misaligned_inputs(inputs, check_input_idxs)  # type: ignore[arg-type]

    if not isinstance(inputs, list):
        raise AssertionError(f"Expected inputs to be a list, got {type(inputs)}")

    inps_expanded_dims = [
        get_expanded_dims(x) if idx not in static_input_idxs else []
        for idx, x in enumerate(inputs)
    ]

    # allocate static tensor inputs
    static_inputs = [
        (
            x
            if not isinstance(x, torch.Tensor)
            else static_input(x)
            if idx not in static_input_idxs
            else x.detach()
        )
        for idx, x in enumerate(inputs)
    ]

    # copy over input values for fresh allocations
    for idx, (x, expanded_dims) in enumerate(zip(inputs, inps_expanded_dims)):
        if isinstance(x, torch.Tensor) and idx not in static_input_idxs:
            index_expanded_dims_and_copy_(static_inputs[idx], x, expanded_dims)

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # copy static_inputs because it will be cleared in model
    with torch.cuda.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        static_outputs = model(list(static_inputs))
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    if config.size_asserts:

        def run(new_inputs: list[InputType]) -> Callable[[list[InputType]], Any]:
            if len(static_inputs) != len(new_inputs):
                raise AssertionError(
                    f"Expected static_inputs and new_inputs to have same length, "
                    f"got {len(static_inputs)} vs {len(new_inputs)}"
                )
            for idx, (dst, src, expanded_dims) in enumerate(
                zip(static_inputs, new_inputs, inps_expanded_dims)
            ):
                if not isinstance(dst, torch.Tensor):
                    continue
                if not isinstance(src, torch.Tensor):
                    raise AssertionError(
                        f"Expected src at index {idx} to be a Tensor, got {type(src)}"
                    )
                if idx in static_input_idxs:
                    if dst.data_ptr() != src.data_ptr():
                        raise AssertionError(
                            f"Expected static input at index {idx} to have same data_ptr, "
                            f"got {dst.data_ptr()} vs {src.data_ptr()}"
                        )
                else:
                    # TODO - could make one single op of multiple slices
                    # and avoid dispatch.
                    # Could also pre-index the `dst` tensors
                    index_expanded_dims_and_copy_(dst, src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            # pyrefly: ignore [bad-return]
            return static_outputs

    else:
        copy_indices = [
            idx for idx in range(len(static_inputs)) if idx not in static_input_idxs
        ]

        def run(new_inputs: list[InputType]) -> Callable[[list[InputType]], Any]:
            for idx in copy_indices:
                expanded_dims = inps_expanded_dims[idx]
                src = new_inputs[idx]
                if not isinstance(src, torch.Tensor):
                    raise AssertionError(
                        f"Expected src at index {idx} to be a Tensor, got {type(src)}"
                    )
                index_expanded_dims_and_copy_(static_inputs[idx], src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            # pyrefly: ignore [bad-return]
            return static_outputs

    return align_inputs_from_check_idxs(run, check_input_idxs, OrderedSet())


def compile_fx_aot(
    model_: GraphModule,
    example_inputs_: list[InputType],
    inner_compile: _CompileFxCallable = compile_fx_inner,
    config_patches: dict[str, Any] | None = None,
) -> list[str | Weights] | str | GraphModule:
    if not isinstance(model_, GraphModule):
        raise AssertionError(f"Expected GraphModule, got {type(model_)}: {model_}")

    # [See NOTE] Unwrapping subclasses AOT
    unwrap_tensor_subclass_parameters(model_)

    # pyrefly: ignore [annotation-mismatch, redefinition]
    config_patches: dict[str, Any] = copy.deepcopy(config_patches or {})

    if not (config_patches.get("fx_wrapper", False) or config.fx_wrapper):
        # If fx_wrapper is not set, then set cpp_wrapper
        config_patches["cpp_wrapper"] = True

    output_path = config_patches.get(
        "aot_inductor.output_path", config.aot_inductor.output_path
    )

    if output_path:
        if output_path.endswith(".pt2"):
            raise AssertionError(
                "The output path for aot_compile should not have an extension with .pt2 "
                "this is for specifying the output path for the .so in AOTInductor. "
                "If you would like to package the AOTInductor generated files "
                "into a pt2, please call `torch._inductor.aoti_compile_and_package`."
            )
    else:
        config_patches = {
            **config_patches,
            "aot_inductor.output_path": code_hash(model_.code),
        }

    from .utils import maybe_aoti_standalone_config

    config_patches = maybe_aoti_standalone_config(config_patches)

    extern_node_serializer = config_patches.pop("extern_node_serializer", None)
    saved_compile_id = model_.meta.get("dynamo_compile_id", None)
    saved_compile_context = torch._guards.CompileContext(saved_compile_id)
    with (
        V.set_aot_compilation(True),
        torch._guards.compile_context(saved_compile_context),
        chromium_event_timed(
            "compile_fx_aot",
            log_pt2_compile_event=True,
            reset_event_log_on_exit=True,
        ),
        get_metrics_context(),
    ):
        compiled_artifacts = compile_fx(
            model_,
            example_inputs_,
            inner_compile=functools.partial(
                inner_compile,
                extern_node_serializer=extern_node_serializer,
            ),
            config_patches=config_patches,
        )

        if not isinstance(compiled_artifacts, CompiledAOTI):
            raise AssertionError(
                f"Expected CompiledAOTI, got {type(compiled_artifacts)}"
            )

        return compiled_artifacts.filename


_graph_counter = count(0)


def fw_compiler_freezing(
    aot_autograd_model: GraphModule,
    aot_example_inputs: Sequence[InputType],
    dynamo_model: GraphModule,
    num_example_inputs: int,
    inner_compile: Callable[..., Any],
    # TODO: Take compiler_config_extra instead
    cudagraphs: BoxedBool,
    graph_id: int,
    forward_device: BoxedDeviceIndex,
) -> Callable[[list[object]], Sequence[torch.Tensor]]:
    from torch._inductor.freezing import convert_conv_weights_to_channels_last, freeze

    # partition_fn won't be called
    inputs_devices = get_inputs_devices(aot_example_inputs, aot_autograd_model)
    aot_autograd_model = _recursive_joint_graph_passes(
        aot_autograd_model,
        input_device=next(iter(inputs_devices)),
    )

    layout_opt = GraphLowering.decide_layout_opt(aot_autograd_model, is_inference=True)
    if layout_opt:
        # make sure meta['val'] is properly setup
        fake_tensor_prop(aot_autograd_model, aot_example_inputs, True)
        convert_conv_weights_to_channels_last(aot_autograd_model)

    opt_model, preserved_arg_indices = freeze(
        dynamo_model,
        aot_autograd_model,
        aot_example_inputs,  # type: ignore[arg-type]
    )

    aot_example_inputs = [aot_example_inputs[ind] for ind in preserved_arg_indices]

    fake_mode = detect_fake_mode(aot_example_inputs)

    # for freezing, all graph outputs should be user visible
    *_, model_outputs_node = opt_model.graph.nodes
    model_outputs = model_outputs_node.args[0]
    model_outputs_node.meta["user_visible_output_idxs"] = [
        idx for idx, n in enumerate(model_outputs) if isinstance(n, torch.fx.Node)
    ]

    static_input_idxs: list[Any] = []
    # constant params will be real tensors, not fake
    tracing_context = torch._guards.TracingContext.try_get()
    unwrapped_args_offsets = [0]
    max_offset_idx = 0
    if tracing_context is not None:
        if tracing_context.params_flat_unwrap_subclasses is None:
            raise AssertionError(
                "Expected tracing_context.params_flat_unwrap_subclasses to be set"
            )
        params_flat_unwrap = tracing_context.params_flat_unwrap_subclasses
        max_offset_idx = max(0, len(params_flat_unwrap) - 1)
        preserved_indices_params_flat = OrderedSet[int]()
        unwrapped_idxs = tracing_context.params_unwrapped_to_flat_index
        if unwrapped_idxs is None:
            raise AssertionError(
                "Expected tracing_context.params_unwrapped_to_flat_index to be set"
            )
        current_offset = 0
        if len(params_flat_unwrap) > 0:
            unwrapped_args_offsets = []

        for i in range(len(params_flat_unwrap)):
            if i not in preserved_arg_indices:
                params_flat_unwrap[i] = None
                if i > 0 and unwrapped_idxs[i] == unwrapped_idxs[i - 1]:
                    current_offset += 1
            else:
                preserved_indices_params_flat.add(unwrapped_idxs[i])
            unwrapped_args_offsets.append(current_offset)

        # Deallocate wrapped params, if all subelements were deallocated
        if tracing_context.params_flat is None:
            raise AssertionError("Expected tracing_context.params_flat to be set")
        for i in range(len(tracing_context.params_flat)):
            if i not in preserved_indices_params_flat:
                tracing_context.params_flat[i] = None

        if tracing_context.fw_metadata:
            static_input_idxs = tracing_context.fw_metadata.static_input_indices

    with mock.patch.object(fake_mode, "allow_non_fake_inputs", True):
        optimized_function = inner_compile(
            opt_model,
            aot_example_inputs,
            static_input_idxs=static_input_idxs,
            cudagraphs=cudagraphs,
            graph_id=graph_id,
            is_inference=True,
            boxed_forward_device_index=forward_device,
            layout_opt=layout_opt,
        )

