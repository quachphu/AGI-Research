    log.debug(
        "Autograd graph cache hash details for key %s:\n%s",
        key,
        LazyString(lambda: "\n".join(debug_lines)),
    )
    return debug_lines


def autograd_cache_key(
    mod: torch.fx.GraphModule | torch._dynamo.utils.GmWrapper,
    example_inputs: Sequence[Any],
    config: AOTConfig,
    compiler_config_extra: CompilerConfigExtra | None = None,
    act_input_paths: ActInputPaths = (),
    # TODO: add args and parameters
) -> tuple[str, list[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """

    gm = mod.gm if isinstance(mod, torch._dynamo.utils.GmWrapper) else mod
    try:
        # Capture graph SymInt placeholder names before normalization so the
        # pickler can distinguish graph-native SymInts (deterministic names)
        # from MetaConverter-generated SymInts (non-deterministic names).
        graph_symint_names: set[str] = set()
        symint_name_map: dict[str, str] = {}
        if torch._functorch.config.autograd_cache_normalize_inputs:
            for n in gm.graph.find_nodes(op="placeholder", sort=True):
                if n.type == torch.SymInt:
                    graph_symint_names.add(n.target)
            j = 0
            for inp in example_inputs:
                if isinstance(inp, torch.SymInt):
                    symint_name_map[str(inp)] = f"s_{j}"
                    j += 1
        with sanitize_gm_for_cache(gm):
            check_cacheable(gm)
            _check_triton_cache_version()
            details = AOTAutogradCacheDetails(
                gm,
                example_inputs,
                config,
                create_fx_config(compiler_config_extra),
                act_input_paths,
            )
            pickler = AOTAutogradCachePickler(gm, symint_name_map, graph_symint_names)
            # The prefix distinguishes among the other kinds of objects we cache
            key = AOTAUTOGRAD_CACHE_PREFIX + pickler.get_hash(details)
            debug_lines = _get_debug_lines_for_cache_key(pickler, details, key)
            return key, debug_lines
    except Exception:
        # If enable_aot_compile is set, we're in AOT precompile mode where we always
        # want to use fallback nonce keys. Unlike caching, it's fine if we can't generate
        # a proper key because we are guaranteed in an AOT precompile world users are in
        # complete control of distributing and loading artifacts.
        if torch._functorch.config.bypass_autograd_cache_key:
            log.info(
                "Failed to generate AOTAutograd cache key; falling back to nonce due to enable_aot_compile",
                exc_info=True,
            )
            return str(random.random()), []
        else:
            raise


@contextlib.contextmanager
def sanitize_gm_for_cache(
    gm: torch.fx.GraphModule,
) -> Generator[None, None, None]:
    """
    Clears a few fields in a dynamo supplied Graph Module that are not stable between graph inputs, but don't
    affect inductor or aotdispatch correctness.

    These fields **can** be used by code calling into aotdispatch (namely, dynamo), so we can't null them out completely.

    To ensure that these fields are not accessed by inductor or aotdispatch, we clear them during AOTAutogradCache.load,
    and then put them back before returning. This way, we generate a cache key based off of a canonical graph
    without these fields, and also guarantee they aren't used to affect the cache's output.
    """
    # Mapping from each field to a default value
    IGNORED_FIELDS: dict[str, Any] = {
        # pyrefly: ignore [implicit-any]
        "meta": {},  # metadata used by export
        "compile_subgraph_reason": None,  # Used by dynamo only for logging, no change in inductor/autograd behavior
        "_param_name_to_source": None,  # Encapsulated by aot_config.aot_autograd_arg_pos_to_source
        "_backend_id": None,
    }
    saved_fields = {}
    for field, default_value in IGNORED_FIELDS.items():
        saved_fields[field] = getattr(gm, field, None)
        # Clear the field
        setattr(gm, field, default_value)
    try:
        with normalize_placeholder_names(gm):
            yield
    finally:
        for field, value in saved_fields.items():
            setattr(gm, field, value)


@CacheArtifactFactory.register
class AOTAutogradCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None:
        AOTAutogradCache._write_to_local_cache(self.key, self.content)

    @override
    @staticmethod
    def type() -> str:
        return "aot_autograd"


class AOTAutogradCache(GuardedCache[GenericAOTAutogradResult[Any, Any]]):
    """
    Caches the results of running AOTAutograd. This class mostly handles the save and load logic, whereas
    AOTAutogradResult handles the wrapping/unwrapping logic.

    Cache Inputs (AOTAutogradCacheDetails)
    - AOTAutogradCache takes in the following inputs, which are analogous to inputs given
        to AOTAutograd by dynamo:
        - A fx graph module generated by dynamo
        - A list of args, which consists of:
            - Symint inputs to the graph, generated by dynamo
            - The **real tensor** inputs, which inductor uses for cudagraphs
            - Notably, the real tensor inputs don't have symints in their metadata.
        AOTAutograd then retraces those real tensor arguments into FakeTensors later during execution.
        - A set of global configurations that affect AOTAutograd or Inductor behavior.

    It then generates a cache key given these values. Notably, this means AOTAutogradCache currently
    specializes on the sizes and strides of the real tensor inputs when dynamic shapes are turned on.
    In a later PR, we'll likely generate the cache key based on the FakeTensors AOTAutograd generates
    based on the real tensor inputs, which can contain symints.

    # Cache Outputs (AOTAutogradResult)
    - AOTAutogradCache caches the following values:
        - The compiled forward and backward functions from inductor, via keys to the FXGraphCache
        - Metadata to reconstruct the AOTModule from the compiled inductor artifacts
        - See AOTAutogradResult for more info

    [Note: Caching guards generated by AOTAutograd and Inductor]
    AOTAutograd and inductor both can introduce new guards to the shape environment. FXGraphCache saves guards with each
    compiled graph inductor generates. On a cache hit, AOTAutograd reloads the compiled forward and backward functions
    from FXGraphCache, giving it new symint arguments from the input args.
    FXGraphCache uses those symints and its saved guards to repopulate the ShapeEnv with guards.
    **No new guards are generated into the shape env after inductor finishes compiling**, so the guards
    saved by inductor are sufficient for correctness for both AOTAutograd and Inductor's caches.
    """

    @staticmethod
    def clear() -> None:
        """Clear the cache"""
        shutil.rmtree(AOTAutogradCache._get_tmp_dir(), ignore_errors=True)

    @staticmethod
    def try_load(
        mod: torch.fx.GraphModule | torch._dynamo.utils.GmWrapper,
        args: list[Any],
        aot_config: AOTConfig,
        compiler_config_extra: CompilerConfigExtra | None,
        local: bool,
        remote: bool,
        act_input_paths: ActInputPaths = (),
        compile_region_name: str | None = None,
    ) -> tuple[Callable[..., Any] | None, AOTConfig]:
        """
        Load a result from the cache, and reconstruct a runtime wrapper around
        the object. On a cache miss, return an updated config carrying the
        cache metadata needed to save the eventual compile result.
        """
        compiled_fn = None
        cache_info: dict[str, Any] = {}
        cache_key = None
        debug_lines: list[str] = []
        cache_event_time = time.time_ns()
        cache_state = None
        try:
            cache_key, debug_lines = autograd_cache_key(
                mod, args, aot_config, compiler_config_extra, act_input_paths
            )
            result: tuple[GenericAOTAutogradResult[Any, Any], bytes] | None = (
                AOTAutogradCache._lookup(
                    cache_key, local, remote, args, cache_info, aot_config
                )
            )
            if result is not None:
                (entry, pickled_content) = result
                fx_config = create_fx_config(compiler_config_extra, compile_region_name)
                compiled_fn = entry.wrap_post_compile(args, aot_config, fx_config)
                # Make the compiled_fn serializable, where the serialize function just
                # makes a copy of the original entry before post compile via the pickled content
                compiled_fn = SerializableCompiledFunction(
                    compiled_fn, lambda: pickle.loads(pickled_content)
                )
                log.info("AOTAutograd cache hit for key %s", cache_key)

                counters["aot_autograd"]["autograd_cache_hit"] += 1
                cache_state = "hit"
                cache_event_time = time.time_ns()
                forward_time_saved = entry.forward_time_taken_ns // 1e6
                backward_time_saved = entry.backward_time_taken_ns // 1e6
                cache_info.update(
                    {
                        "forward_time_saved_ms": forward_time_saved,
                        "backward_time_saved_ms": backward_time_saved,
                        "time_saved_ms": forward_time_saved + backward_time_saved,
                    }
                )
                time_saved_ns = (
                    entry.forward_time_taken_ns + entry.backward_time_taken_ns
                )
                # TODO: should we use the same field for remote cache time saved for both
                # FXGraphCache and AOTAutogradCache?
                # get_metrics_context().increment(...)
                if (
                    ephemeral_increase
                    := add_ephemeral_timeout_increase_for_distributed(time_saved_ns)
                ) != 0:
                    cache_info["ephemeral_timeout_increase"] = ephemeral_increase

            if compiled_fn is None:
                log.info("AOTAutograd cache miss for key %s", cache_key)
                counters["aot_autograd"]["autograd_cache_miss"] += 1
                cache_state = "miss"
                cache_event_time = time.time_ns()
        # Count missing the FXGraphCache as a miss not a bypass
        except FXGraphCacheMiss as e:
            counters["aot_autograd"]["autograd_cache_miss"] += 1
            cache_state = "miss"
            if config.strict_autograd_cache or torch._dynamo.config.strict_precompile:
                raise e
        # Most often this is BypassAOTAutogradCache, but
        # if there's ever different reason we can't cache,
        # we still never want to hard throw an exception, since
        # we can always fallback to a cache bypass.
        # As an example, if the user calls autograd via
        # standalone inductor, we will sometimes get a GraphModule
        # that doesn't actually have a `.graph` on it. Instead
        # of checking every single case, we safely catch the exception
        # in those cases.
        except Exception as e:
            cache_key = None
            counters["aot_autograd"]["autograd_cache_bypass"] += 1
            log.info("Bypassing autograd cache due to: %s", e)
            cache_state = "bypass"
            cache_event_time = time.time_ns()
            cache_info["cache_bypass_reason"] = str(e)
            cache_info["cache_bypass_exception_type"] = type(e).__name__
            cache_info["cache_bypass_traceback"] = traceback.format_exc().split("\n")
            # TODO: this gets logged implicitly by cache_bypass_reason,
            # and here we explicitly log it into tlparse.
            # We may want to log this as an extra column in Scuba, though.
            cache_info["cache_bypass_hard_exception"] = not isinstance(
                e, BypassAOTAutogradCache
            )
            if remote:
                log_cache_bypass("bypass_aot_autograd", str(e))
            if config.strict_autograd_cache or torch._dynamo.config.strict_precompile:
                raise e
        if compiled_fn is None:
            # Set the cache key so we can save a cache result later
            symints = AOTAutogradCache._filter_backed_symints(args)
            if cache_key is not None:
                aot_config = dataclasses.replace(
                    aot_config,
                    cache_info=AOTAutogradCacheInfo(
                        cache_key,
                        time.time_ns(),
                        forward_symints=symints,
                    ),
                )

        cache_info.update(
            {
                "key": cache_key,
                "cache_state": cache_state,
                "components": debug_lines,
            }
        )
        if chromium_event_log_active():
            CompileEventLogger.instant(
                f"autograd_cache_{cache_state}",
                metadata=cache_info,
                time_ns=cache_event_time,
            )
            CompileEventLogger.try_add_pt2_compile(
                "backend_compile",
                cache_state=cache_state,
                cache_event_time=cache_event_time,
                key=cache_info.get("key"),
                components=cache_info.get("components"),
                cache_bypass_reason=cache_info.get("cache_bypass_reason"),
                remote_cache_enabled=remote,
                local_cache_enabled=local,
            )

        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"aotautograd_cache_{cache_state}",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(cache_info),
        )

        return compiled_fn, aot_config

    @classmethod
    def generate_guards_expression(
        cls: type[AOTAutogradCache], cache_info: AOTAutogradCacheInfo
    ) -> str | None:
        shape_env = cls._get_shape_env()

        if shape_env is None:
            return None

        symints = cache_info.forward_symints
        guards = shape_env.get_pruned_guards(symints)
        return shape_env.produce_guards_expression(placeholders=symints, guards=guards)

    @classmethod
    def _get_tmp_dir(cls: type[AOTAutogradCache]) -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), "aotautograd")

    @classmethod
    def _get_tmp_dir_for_key(cls: type[AOTAutogradCache], _key: str) -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cls._get_tmp_dir(), _key)

    @classmethod
    def _record_result(
        cls: type[AOTAutogradCache],
        key: str,
        local_hit: bool,
        local_miss: bool,
        remote_hit: bool,
        remote_miss: bool,
    ) -> None:
        """
        Called by GuardedCache to record hit/miss statistics.
        """
        if local_hit:
            cache_stats.hit("LocalAOTAutogradCache")
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_local_cache_hit_count",
            )
        if remote_hit:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_remote_cache_hit_count",
            )
        if local_miss:
            cache_stats.miss("LocalAOTAutogradCache")
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_local_cache_miss_count",
            )
        if remote_miss:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_remote_cache_miss_count",
            )

    @staticmethod
    def evaluate_guards(guard_expr: str, hints: list[int] | list[torch.SymInt]) -> bool:
        if torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards:
            return True
        shape_env = AOTAutogradCache._get_shape_env()
        if shape_env is None:
            raise AssertionError("shape_env must not be None")
        result = shape_env.evaluate_guards_expression(guard_expr, hints)
        return result

    @staticmethod
    def _lookup(
        key: str,
        local: bool,
        remote: bool,
        args: list[Any],
        cache_info: dict[str, Any],
        aot_config: AOTConfig | None,
    ) -> tuple[GenericAOTAutogradResult[Any, Any], bytes] | None:
        """Given a key generated by AOTAutogradCachePickler, look up its location in the cache."""
        remote_cache: RemoteCache[JsonDataTy] | None = None
        if remote:
            remote_cache = AOTAutogradCache.get_remote_cache()

        symints = AOTAutogradCache._filter_backed_symints(args)
        hints = [guarding_hint_or_throw(s) for s in symints]
        entry = None
        pickled_content = None
        try:
            (
                entry,
                pickled_content,
                guard_info,
            ) = AOTAutogradCache.find_guarded_entry(
                key, local, remote_cache, AOTAutogradCache.evaluate_guards, hints
            )

            if entry is None and guard_info["cache_status_detailed"] == "guard_miss":
                counters["aot_autograd"]["autograd_cache_guard_miss"] += 1
            cache_info.update(guard_info)
            CacheArtifactRecorder(
                AOTAutogradCacheArtifact.type(), key
            ).record_if_present(pickled_content)
            if pickled_content is not None:
                if (
                    should_bundle_autograd_cache()
                    and aot_config is not None
                    and aot_config.precompile_backend_id is not None
                ):
                    # NB: We don't want to use the cached aot_config.precompile_backend_id
                    # 1. because we set it to None on save 2. even if we didn't, this new run
                    # that cache hit has a *new* backend id associated with it.
                    PrecompileContext.record_artifact(
                        BundledAOTAutogradCacheArtifact(
                            aot_config.precompile_backend_id, entry
                        ),
                    )
        except Exception as e:
            log.info("AOTAutograd cache unable to load compiled graph: %s", e)
            if config.strict_autograd_cache:
                raise e
        if entry is not None:
            if pickled_content is None:
                raise AssertionError(
                    "pickled_content must not be None when entry is not None"
                )
            return (entry, pickled_content)
        else:
            return None

    @staticmethod
    def _write_to_local_cache(key: str, content: bytes) -> None:
        """Write an entry to the local cache."""
        subdir = AOTAutogradCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)

        # Use a hash of the serialized entry to get a unique file
        # name. The specific name doesn't matter since a lookup involves
        # iterating over all entries in the parent subdir.
        path = os.path.join(subdir, sha256_hash(content))
        log.info("Writing AOTAutograd cache entry to %s", path)
        write_atomic(path, content)

    @staticmethod
    def _find_unpicklable_field(
        entry: GenericAOTAutogradResult[Any, Any],
    ) -> str | None:
        """Find which field of entry is causing pickle to fail."""
        # pyrefly: ignore [implicit-any]
        fields = []
        if hasattr(entry, "__dataclass_fields__"):
            fields = list(entry.__dataclass_fields__.keys())
        elif hasattr(entry, "__dict__"):
            fields = list(entry.__dict__.keys())

        for name in fields:
            try:
                pickle.dumps(getattr(entry, name))
            except Exception:
                return name
        return None

    @staticmethod
    def _pickle_entry(
        entry: GenericAOTAutogradResult[Any, Any], remote: bool
    ) -> bytes | None:
        """Pickle entry, returning None on failure."""
        try:
            with torch.utils._python_dispatch._disable_current_modes():
                return pickle.dumps(entry)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            bad_field = AOTAutogradCache._find_unpicklable_field(entry)
            error_str = str(e)
            log.warning("AOTAutograd cache unable to serialize compiled graph: %s", e)
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aotautograd_cache_pickle_failure",
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps({"error": error_str, "field": bad_field}),
            )
            if remote:
                log_cache_bypass(
                    "bypass_aot_autograd", "Unable to serialize: " + str(e)
                )
            if config.strict_autograd_cache:
                raise
            return None

    @staticmethod
    def _handle_save_error(e: Exception, remote: bool, is_bypass: bool) -> None:
        """Handle exceptions during save, re-raising if strict mode is enabled."""
        if is_bypass:
            counters["aot_autograd"]["autograd_cache_bypass"] += 1
            log.info("Bypassing autograd cache due to: %s", e)
            bypass_reason = str(e)
        else:
            log.warning("AOTAutograd cache unable to serialize compiled graph: %s", e)
            bypass_reason = "Unable to serialize: " + str(e)
        if remote:
            log_cache_bypass("bypass_aot_autograd", bypass_reason)
        if config.strict_autograd_cache:
            raise e

    @staticmethod
    def save(key: str, entry: GenericAOTAutogradResult[Any, Any], remote: bool) -> None:
        """Save a single entry into the cache."""
        content: bytes | None = None
        try:
            entry.pre_save()
            content = AOTAutogradCache._pickle_entry(entry, remote)
            if content is None:
                return None
            CacheArtifactRecorder(AOTAutogradCacheArtifact.type(), key).record(content)
            if (
                should_bundle_autograd_cache()
                and entry.sanitized_aot_config.precompile_backend_id is not None
            ):
                precompile_key = entry.sanitized_aot_config.precompile_backend_id
                artifact = BundledAOTAutogradCacheArtifact(precompile_key, entry)
                entry.sanitized_aot_config.precompile_backend_id = None
                PrecompileContext.record_artifact(artifact)
            AOTAutogradCache._write_to_local_cache(key, content)
            counters["aot_autograd"]["autograd_cache_saved"] += 1
            cache_stats.put("LocalAOTAutogradCache")
        except BypassAOTAutogradCache as e:
            AOTAutogradCache._handle_save_error(e, remote, is_bypass=True)
            return None
        except Exception as e:
            AOTAutogradCache._handle_save_error(e, remote, is_bypass=False)
            return None

        if remote:
            remote_cache = AOTAutogradCache.get_remote_cache()
            if remote_cache is not None:
                time_taken_ms = int(
                    (entry.forward_time_taken_ns + entry.backward_time_taken_ns) // 1e6
                )
                cache_data: JsonDataTy = {
                    "data": base64.b64encode(content).decode("ascii"),
                    "time_taken_ms": time_taken_ms,
                }
                remote_cache.put(key, cache_data)

    @staticmethod
    @functools.cache
    def get_remote_cache() -> RemoteCache[JsonDataTy] | None:
        """
        Attempts to load the remote cache, returns None on error.
        """
        cache_id = "autograd-experimental"
        return create_cache(
            cache_id,
            config.is_fbcode(),
            "FbRemoteAOTAutogradCache",
            "RemoteAOTAutogradCache",
        )

    @staticmethod
    def make_entry(
        compiled_fw_func: OutputCode,
        compiled_bw_func: OutputCode | None,
        aot_joint_graph_str: str | None,
        aot_forward_graph_str: str | None,
        aot_backward_graph_str: str | None,
        runtime_metadata: ViewAndMutationMeta,
        dispatch_wrappers: list[CompilerWrapper],
        maybe_subclass_meta: SubclassMeta | None,
        num_fw_outs_saved_for_bw: int | None,
        indices_of_inps_to_detach: list[int],
        forward_time_taken_ns: int,
        backward_time_taken_ns: int,
        sanitized_aot_config: CacheableAOTConfig,
        guards_expr: str | None,
        backward_state_indices: list[int] | None,
        num_symints_saved_for_bw: int | None,
        serialized_bw_module: SerializedGraphModule | None,
        min_cut_info_str: str | None,
    ) -> GenericAOTAutogradResult[Any, Any]:
        if should_bundle_autograd_cache():
            # Helper function to unwrap all the wrappers we added during aotdispatch
            # They get reapplied on cache load
            def unwrap_output_code(obj: object) -> OutputCode:
                while hasattr(obj, "__wrapped__"):
                    obj = obj.__wrapped__
                if not isinstance(obj, OutputCode):
                    raise AssertionError(f"expected OutputCode, got {type(obj)}")
                return obj

            compiled_fw_graph = unwrap_output_code(compiled_fw_func)
            bundled_compiled_forward = BundledCompiledForward(compiled_fw_graph)
            bundled_compiled_backward = None
            if compiled_bw_func is not None:
                if backward_state_indices is None:
                    raise AssertionError("backward_state_indices must not be None")
                if num_symints_saved_for_bw is None:
                    raise AssertionError("num_symints_saved_for_bw must not be None")
                compiled_bw_graph = unwrap_output_code(compiled_bw_func)
                bundled_compiled_backward = BundledCompiledBackward(
                    compiled_bw_graph, backward_state_indices, num_symints_saved_for_bw
                )

            return BundledAOTAutogradResult(
                compiled_fw=bundled_compiled_forward,
                compiled_bw=bundled_compiled_backward,
                aot_joint_graph_str=aot_joint_graph_str,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=aot_backward_graph_str,
                min_cut_info_str=min_cut_info_str,
                runtime_metadata=runtime_metadata,
                dispatch_wrappers=dispatch_wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
                indices_of_inps_to_detach=indices_of_inps_to_detach,
                forward_time_taken_ns=forward_time_taken_ns,
                backward_time_taken_ns=backward_time_taken_ns,
                sanitized_aot_config=sanitized_aot_config,
                guards_expr=guards_expr,
                serialized_bw_module=serialized_bw_module,
            )

        else:
            fw_key = getattr(compiled_fw_func, "_fx_graph_cache_key", None)
            fw_debug_lines = getattr(
                compiled_fw_func, "_fx_graph_cache_debug_lines", []
            )

            if fw_key is None:
                raise AssertionError("fw_key must not be None")
            compiled_forward = CompiledForward(
                fx_graph_cache_info=(fw_key, fw_debug_lines),
                fx_graph_guard_expr=getattr(compiled_fw_func, "guards_expr", None),
            )
            compiled_backward = None
            if compiled_bw_func is not None:
                bw_key = getattr(compiled_bw_func, "_fx_graph_cache_key", None)
                bw_debug_lines = getattr(
                    compiled_bw_func, "_fx_graph_cache_debug_lines", []
                )
                if bw_key is None:
                    raise AssertionError("bw_key must not be None")
                if backward_state_indices is None:
                    raise AssertionError("backward_state_indices must not be None")
                if num_symints_saved_for_bw is None:
                    raise AssertionError("num_symints_saved_for_bw must not be None")
                compiled_backward = CompiledBackward(
                    fx_graph_cache_info=(bw_key, bw_debug_lines),
                    fx_graph_guard_expr=getattr(compiled_bw_func, "guards_expr", None),
                    backward_state_indices=backward_state_indices,
                    num_symints_saved_for_bw_=num_symints_saved_for_bw,
                )

            return AOTAutogradResult(
                compiled_fw=compiled_forward,
                compiled_bw=compiled_backward,
                aot_joint_graph_str=aot_joint_graph_str,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=aot_backward_graph_str,
                min_cut_info_str=min_cut_info_str,
                runtime_metadata=runtime_metadata,
                dispatch_wrappers=dispatch_wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
                indices_of_inps_to_detach=indices_of_inps_to_detach,
                forward_time_taken_ns=forward_time_taken_ns,
                backward_time_taken_ns=backward_time_taken_ns,
                sanitized_aot_config=sanitized_aot_config,
                guards_expr=guards_expr,
                serialized_bw_module=serialized_bw_module,
            )
<|endoftext|>"""
The various dataclasses, Enums, namedtuples etc used in AOTAutograd. This includes
input/output types, metadata, config, function signatures etc.
"""

from __future__ import annotations

import collections
import functools
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, NewType, Protocol, TYPE_CHECKING, TypeAlias, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._custom_class_base import CustomClassBase
from torch._subclasses.fake_tensor import is_fake, is_fake_tensor
from torch.fx.experimental._backward_state import BackwardState
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config
from .functional_utils import _check_if_mutation_can_be_in_graph, ViewMetaSequence
from .utils import strict_zip


if TYPE_CHECKING:
    import contextlib
    from collections.abc import Callable, Iterable

    from torch._guards import Source
    from torch._inductor.output_code import OutputCode
    from torch._inductor.utils import InputType
    from torch._ops import OpOverload
    from torch._subclasses import FakeTensorMode
    from torch.types import IntLikeType

    from .descriptors import AOTInput, AOTOutput
    from .graph_capture_wrappers import JointFnHandle

_P = ParamSpec("_P")
_R = TypeVar("_R")
zip = strict_zip

ActInputPath: TypeAlias = tuple[int, tuple[str, ...]]
ActInputPaths: TypeAlias = Sequence[ActInputPath]


OutputType = Enum(
    "OutputType",
    (
        # output is not an alias
        "non_alias",
        # output aliases an input
        "alias_of_input",
        # output **is** an input tensor
        "is_input",
        # output has a ._base tensor, which is a graph intermediate.
        # We need to return its ._base as a graph output,
        # so its requires_grad info is populated correctly.
        # Instructs the runtime code to regenerate the current output
        # from a base tensor, graph_intermediates[base_idx]
        "alias_of_intermediate_save_as_output",
        # Same as above; but we don't need to explicitly add its ._base
        # as a graph output, because it already **is** a graph output.
        "alias_of_intermediate",
        # Same as above; but the output's ._base is **already** a user output.
        # Instructs the runtime code to regenerate the current output from
        # a base tensor, user_outputs[base_idx]
        "alias_of_intermediate_base_is_user_output",
        # See Note [Intermediate Bases Optimization]
        "unsafe_view_alias",
        # output is an alias, but has a custom autograd.Function backward.
        # In this case, we don't want to do view-replay, since we won't be able to replay the custom function.
        # Instead, we'll treat this output "normally", and trace its backward into the graph.
        "custom_function_view",
    ),
)


# This class stores info about every user output.
@dataclass(frozen=True)
class OutputAliasInfo:
    # Tells us if this output is:
    # (1) a regular (non-aliased) output
    # (2) an alias of a forward input
    # (3) **is** a forward input (special case of "alias_of_input")
    # (4) an alias of an intermediate (aka an alias of an output of the inner traced forward)
    # (5) an alias of an intermediate, that explicitly requires returning the intermediate
    #     as a graph output
    # (6) an alias of an intermediate, where that intermediate is also a user output
    output_type: OutputType
    # The raw type of the output (torch.Tensor, SymInt, etc)
    raw_type: type
    # If (1) above, then
    # - base_idx is None
    # If (2) or (3) above, then
    # - Tells us that the base of this alias is user_fwd_input[base_idx]
    #   (This is an index into the inputs *before* we make synthetic bases)
    # If (4) or (5) above, then
    # - Tells us that the base of this alias is output_graph_intermediates[base_idx]
    #   here, this refers to the index of the *direct* traced
    # If (6) above, then:
    # - Tells us that the base of this alias is output_user_fwds[base_idx]
    #   here, this refers to the index of the *direct* traced
    base_idx: int | None
    # If it is a Tensor, what the dynamic dims are (otherwise is None)
    dynamic_dims: set[int] | None
    # Whether the output tensor has requires_grad=True. Use for
    # preserving output properties (mark_non_differentiable, alias reconstruction).
    requires_grad: bool
    # Whether this output participates in backward as
    # a tangent. Normally the same as requires_grad, but False for views created
    # under no_grad() that inherit requires_grad from their base without having a
    # grad_fn. Use when constructing tangent lists for torch.autograd.grad().
    requires_grad_for_backward: bool
    # Sequence of ViewMeta objects.
    #
    # Provides us the means to re-run view functions on other tensors.
    #
    # We need to wrap the actual list of ViewMeta with this class so that
    # we compare the ViewMeta elements appropriately, i.e. their type and
    # the elements returned by the `as_tuple()` call.
    view_meta_sequence: ViewMetaSequence | None = None


class MutationType(Enum):
    NOT_MUTATED = 1
    MUTATED_IN_GRAPH = 2
    MUTATED_OUT_GRAPH = 3


# This class tells us info about user inputs.
@dataclass(frozen=True)
class InputAliasInfo:
    is_leaf: bool
    mutates_data: bool
    mutates_metadata: bool
    mutations_hidden_from_autograd: bool
    mutations_under_no_grad_or_inference_mode: bool
    mutation_inductor_storage_resize: bool
    mutates_storage_metadata: bool
    mutation_is_shallow_copy_data: bool
    requires_grad: bool
    keep_input_mutations: bool

    def __post_init__(self) -> None:
        if self.mutates_storage_metadata:
            # For convenience, we guarantee that this is always true.
            # In practice, If we call .set_(), then at runtime there is no need
            # to additionally fix  up the tensor metadata, since our runtime
            # call to inp.set_(updated_inp) will already have the right metadata
            if not self.mutates_metadata:
                raise AssertionError(
                    "mutates_storage_metadata requires mutates_metadata to be True"
                )

    @functools.cached_property
    def mutation_type(self) -> MutationType:
        if (
            (not self.mutates_data)
            and (not self.mutates_metadata)
            and not (self.mutation_inductor_storage_resize)
        ):
            return MutationType.NOT_MUTATED

        if _check_if_mutation_can_be_in_graph(
            self.keep_input_mutations,
            self.mutates_data,
            self.mutates_metadata,
            self.mutations_hidden_from_autograd,
            self.mutations_under_no_grad_or_inference_mode,
            self.mutates_storage_metadata,
            self.mutation_inductor_storage_resize,
            self.requires_grad,
        ):
            return MutationType.MUTATED_IN_GRAPH

        return MutationType.MUTATED_OUT_GRAPH


@dataclass
class MemoryFormatMeta:
    # For static shapes we assume tangents have the same strideness as outputs
    size: Sequence[int] | None = None
    stride: Sequence[int] | None = None

    # For dynamic shapes we assume the same memory format: contiguous, channels_last etc.
    memory_format: torch.memory_format | None = None

    @staticmethod
    def from_tensor(
        t: torch.Tensor, force_use_memory_format: bool = False
    ) -> MemoryFormatMeta | None:
        # We only memorize expected memory format for
        # 1. Traceable wrapper subclasses
        # We can not create restrided subclass tensor, as torch.empty_strided works only with dense tensors.
        # 2. Dynamic shape tensors
        # Support for symbolic shapes is not implemented yet.
        # 3. force_use_memory_format=True (e.g., local_map where shapes change)
        use_memory_format: bool = (
            force_use_memory_format
            or not torch._functorch.config.guess_tangent_strides_as_outputs
            or is_traceable_wrapper_subclass(t)
        )
        if not use_memory_format:
            use_memory_format = t._has_symbolic_sizes_strides

        if (
            not use_memory_format
            and t.layout == torch.strided
            and torch._debug_has_internal_overlap(t) == 1
        ):
            # An internally overlapping output, e.g. from expand(), cannot
            # represent arbitrary incoming gradients with its own strides.
            use_memory_format = True

        if use_memory_format:
            return MemoryFormatMeta(
                # pyrefly: ignore [unbound-name]
                memory_format=torch._prims_common.suggest_memory_format(t),
            )

        return MemoryFormatMeta(
            size=t.size(),
            stride=t.stride(),
        )


@dataclass
class PlainTensorMeta:
    unwrapped_idx: int
    memory_format: MemoryFormatMeta | None = None
    size_symbol_placeholders: tuple[bool, ...] = ()
    stride_symbol_placeholders: tuple[bool, ...] = ()

    @property
    def arg_count(self) -> int:
        return (
            1
            + sum(self.size_symbol_placeholders)
            + sum(self.stride_symbol_placeholders)
        )


@dataclass
class OpaqueMeta:
    pass


@dataclass
class SubclassCreationMeta:
    """
    Used for AOTDispatch.
    This dataclass gives us the information we need to reconstruct a tensor subclass
    from our flat inputs.
    Why is this important? The graph that we'd like to trace out contains flat tensor inputs,
    But the user's original model may have subclass inputs and outputs.
    So we need to wrap/unwrap subclasses as necessary to translate between the user's
    view (subclass inps/outs), and the backend compiler's view (graph with no subclass args).

    Complications arise mostly from the fact that a subclass can hold more than one inner tensor;
    So for a given subclass input/output, we need to carefully track which indices map
    to the subclass tensor in the corresponding "dense-tensor-only" graph.
    """

    # In the inner graph that only takes in dense tensor inputs,
    # this maps to the first index of "tensors that should go in this subclass wrapper"
    flat_tensor_start_idx: int
    # arg_count is inclusive of the arg_counts of any
    # inner tensor subclasses: If I have a TwoTensor and
    # both of its inner elements are TwoTensors, then the
    # arg_count of the outer-most subclass will be 4
    arg_count: int
    # Mark where or not symints were included. This flag is only used in one assertion
    # in "wrap_tensor_subclasses"
    included_subclass_symints: bool
    # meta and attrs are produced by the subclass's __tensor_flatten__.
    # We need to keep them around along with outer_size / outer_stride to plumb them
    # into __tensor_unflatten__
    attrs: dict[str, SubclassCreationMeta | PlainTensorMeta | OpaqueMeta]
    outer_size: Iterable[IntLikeType | None]
    outer_stride: Iterable[IntLikeType | None]
    meta: Any
    # Stores the original subclass itself.
    # This is needed because we need the autograd metadata on the original subclass
    # (this is guaranteed to be a wrapper subclass that holds a fake tensor,
    #  so holding onto this at runtime shouldn't leak memory)
    # This field is nulled out after calling make_runtime_safe()
    original_subclass: torch.Tensor | None

    # Used at runtime to determine the subclass type, so we don't need to save the original subclass
    original_subclass_type: type | None = None
    memory_format: MemoryFormatMeta | None = None
    outer_size_from_attr: str | None = None
    outer_stride_from_attr: str | None = None

    def compute_outer_size_and_stride(
        self,
        all_args: Sequence[torch.Tensor | IntLikeType | CustomClassBase],
        *,
        curr_start_idx: int,
    ) -> tuple[
        Iterable[IntLikeType | None],
        Iterable[IntLikeType | None],
    ]:
        from .subclass_utils import compute_symint_placeholders

        def compute(
            outer: Iterable[IntLikeType | None], start_idx: int
        ) -> tuple[Any, int | Any]:
            placeholders = compute_symint_placeholders(outer)
            has_symbolic = any(placeholders)

            if has_symbolic:
                end = start_idx + sum(placeholders)
                it_args = iter(all_args[start_idx:end])
                it_placeholders = iter(placeholders)
                return pytree.tree_map_only(
                    lambda _: next(it_placeholders), lambda _: next(it_args), outer
                ), end
            else:
                return outer, start_idx

        outer_size, next_idx = compute(self.outer_size, curr_start_idx)
        outer_stride, _ = compute(self.outer_stride, next_idx)
        return outer_size, outer_stride

    def creation_fn(
        self,
        all_args: Sequence[torch.Tensor | IntLikeType | CustomClassBase],
        *,
        is_runtime: bool,
    ) -> torch.Tensor:
        inner_tensors: dict[str, torch.Tensor | CustomClassBase] = {}

        curr_start_idx = self.flat_tensor_start_idx
        for attr, creation_meta in self.attrs.items():
            if isinstance(creation_meta, OpaqueMeta):
                opaque = all_args[curr_start_idx]
                if not isinstance(opaque, CustomClassBase):
                    raise AssertionError(
                        f"CustomClassBase expected, got {type(opaque)}"
                    )
                inner_tensors[attr] = opaque
                curr_start_idx += 1
                continue
            if isinstance(creation_meta, PlainTensorMeta):
                subclass = all_args[curr_start_idx]
                if not isinstance(subclass, Tensor):
                    raise AssertionError("Tensor expected")
                curr_start_idx += creation_meta.arg_count
            else:
                subclass = creation_meta.creation_fn(
                    all_args,
                    is_runtime=is_runtime,
                )
                curr_start_idx += creation_meta.arg_count
            inner_tensors[attr] = subclass

        if is_runtime:
            if self.original_subclass_type is None:
                raise AssertionError(
                    "original_subclass_type must not be None at runtime"
                )
            original_subclass_type = self.original_subclass_type
        else:
            original_subclass_type = type(self.original_subclass)

        if is_runtime:
            outer_size, outer_stride = self.compute_outer_size_and_stride(
                all_args,
                curr_start_idx=curr_start_idx,
            )
            if self.outer_size_from_attr is not None:
                size_attr = inner_tensors[self.outer_size_from_attr]
                if not isinstance(size_attr, Tensor):
                    raise AssertionError("Tensor expected")
                outer_size = size_attr.size()
            if self.outer_stride_from_attr is not None:
                stride_attr = inner_tensors[self.outer_stride_from_attr]
                if not isinstance(stride_attr, Tensor):
                    raise AssertionError("Tensor expected")
                outer_stride = stride_attr.stride()
        else:
            outer_size, outer_stride = self.outer_size, self.outer_stride

        rebuilt = original_subclass_type.__tensor_unflatten__(  # type: ignore[attr-defined]
            inner_tensors, self.meta, outer_size, outer_stride
        )

        if not is_runtime:
            # After wrapping up the inner dense tensors into a subclass, we need to make sure that our new wrapper
            # has correct autograd metadata, since we'll be tracing through the autograd engine with the subclass.
            # We don't trace through the autograd engine at runtime though, so no need
            # to compute this extra metadata then!
            torch._mirror_autograd_meta_to(self.original_subclass, rebuilt)  # type: ignore[attr-defined]

        return rebuilt

    def make_runtime_safe(self) -> None:
        def _make_size_runtime_safe(x: None | int | torch.SymInt) -> int | None:
            dummy = -1
            if isinstance(x, torch.SymInt):
                # Replace nested ints by a dummy value (-1) as NJT ignores
                # the outer_size/outer_stride at runtime.
                return dummy if x.node.is_nested_int() else None
            return x

        if self.original_subclass is None:
            raise AssertionError(
                "original_subclass must not be None before calling make_runtime_safe"
            )
        self.original_subclass_type = type(self.original_subclass)
        self.original_subclass = None

        # Note: NJT outer_size in AOTDispatcher
        # `_make_size_runtime_safe` replaces any nested int with a dummy value (-1)
        # to prevent serializing a SymInt at runtime. Internally, nested tensor __tensor_unflatten__
        # is designed to safely ignore this dummy value.
        # For more details, see: https://github.com/pytorch/pytorch/blob/5141ade8e30c64e873e14dcc8de233da45d15025/torch/nested/_internal/nested_tensor.py#L266-L299
        self.outer_size = tuple(map(_make_size_runtime_safe, self.outer_size))
        self.outer_stride = tuple(map(_make_size_runtime_safe, self.outer_stride))

        # Recurse on nested subclass info
        for creation_meta in self.attrs.values():
            if isinstance(creation_meta, SubclassCreationMeta):
                creation_meta.make_runtime_safe()

    def __post_init__(self) -> None:
        # sanity assert to make sure we don't leak memory
        if not is_fake(self.original_subclass):
            raise AssertionError(
                f"original_subclass must be a fake tensor to avoid memory leaks, got {type(self.original_subclass)}"
            )


# This class encapsulates all aliasing + mutation info we need about the forward graph
# See a more detailed overview of the edge case handling at
# https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit
# NOTE: This class is saved in AOTAutogradCache, If you are adding elements, make sure
# they are covered by warm cache tests.
@dataclass(eq=False)
class ViewAndMutationMeta:
    # length = # user inputs
    # This gives us info about every input, and what sort of mutation happened to it (if any)
    input_info: list[InputAliasInfo]

    # length = # user outputs
    # This gives us info about every output (mostly around whether it aliases other tensors)
    output_info: list[OutputAliasInfo]

    # length = the number of intermediate bases appended as outputs to the end of the forward graph.
    # Note: this is not necessarily the same thing as:
    #   len([x for x in output_info if x.output_type == OutputType.alias_of_intermediate])
    # Because outputs might share a ._base, or an output's ._base might itself be
    # another user output (in both cases, we won't redundantly append bases to the end of the graph)
    num_intermediate_bases: int

    # For inference only: instructs us to keep data-only input mutations directly in the graph
    keep_input_mutations: bool

    # length = (# inputs w data mutations) + (# user outputs that are non_aliasing tensors)
    #        + (# intermediate bases)
    # These are the FakeTensor (or potential SymInt) outputs that we traced from our
    # metadata pass of the user's forward function.
    # Their only use today is to pass them as a best-guess for tangents when tracing the joint.
    # Stashing them as part of our "metadata" makes it simpler if we want to run our analysis
    # pass once, and reuse the output throughout AOTAutograd
    traced_tangents: list[Any]

    # TODO doc
    traced_tangents_descs: list[AOTInput]

    # Each of these is a list telling us about subclasses for the inputs/outputs/grad_outs
    # They are used throughout AOTDispatch to tell us how to generate a list of subclass tensors,
    # Given a (potentially larger) list of plain torch tensors.

    # Taking subclass_inp_meta as an example:
    #   subclass_inp_meta[i] = j (an int) tells us:
    #     "The i'th user input is not a subclass, and corresponds to inputs[j] of the plain-tensor graph."
    #   subclass_inp_meta[i] = SubclassCreationMeta(flat_tensor_start_idx=3, arg_count=2)
    #     "The i'th user input is subclass holding two inner tensors, which are
    #      inputs[3] and inputs[4] of the plain-tensor graph".

    # length = # user inputs
    subclass_inp_meta: list[PlainTensorMeta | SubclassCreationMeta]
    # So, the full set of outputs to the forward graph looks something like:
    # (*mutated_inps, *user_outs, *intermediate_bases, *saved_for_bw_tensors)
    # where the first 3 of those 4 can be subclasses
    # (but not saved_for_bw tensors, since these are internal to the compiler
    # and not user visible, so there's no point in wrapping/unwrapping them at runtime).
    # This list contains subclass information on all of the fw graph outputs
    # except for saved_for_bw_tensors.
    subclass_fw_graph_out_meta: list[PlainTensorMeta | SubclassCreationMeta]
    # length = # backward graph inputs
    subclass_tangent_meta: list[PlainTensorMeta | SubclassCreationMeta]

    # length = (# inputs w data mutations) + (# user outputs that are non_aliasing tensors)
    #        + (# intermediate bases)
    # At runtime, we don't keep the traced_tangents around since they're not serializable.
    # Instead, we keep any necessary subclass metadata necessary about each traced_tangent.
    # This list is generated after calling make_runtime_safe().
    traced_tangent_metas: list[Any] | None = None

    num_symints_saved_for_bw: int | None = None

    # See Note [Activations with no version counter checks in eager]
    # Number of tensors saved for backward that were stashed on ctx (e.g., ctx.x = x)
    # rather than via save_for_backward in an autograd.Function.
    # These tensors are placed at the end of the saved tensors and should skip
    # version counter checks at runtime.
    num_tensors_saved_with_no_vc_check: int | None = None

    # Number of opaque objects saved for backward
    num_opaque_objects_saved_for_bw: int | None = None

    # Whether each saved tensor is also a graph input.
    saved_tensor_is_graph_input: list[bool] = field(default_factory=list)
    # The grad_enabled mutation that will be emitted in the runtime_wrapper epilogue
    # NOTE: AOTAutograd will assume that the ambient `is_grad_enabled` is the grad mode
    # that is intended to be in effect prior to running the graph, in keeping with
    # equivalence to eager mode. It is the responsibility of upstream graph acquisition
    # to reset the grad mode to its pre-graph value prior to calling aot_autograd.
    grad_enabled_mutation: bool | None = None

    # Keeps track of whether `torch.use_deterministic_algorithms` was turned on
    # when the forward was run. If deterministic mode was turned off during the
    # forward, but is turned on during the backward call, then an error is
    # raised
    deterministic: bool | None = None

    # Keeps track of which input indices store parameters (which we will treat as static)
    static_input_indices: list[int] = field(default_factory=list)

    # Input paths that held AsyncCollectiveTensors at compile time. A path is
    # (input_index, attr_path), where an empty attr_path means the top-level
    # input. Used to emit direct trigger_wait() calls at runtime instead of
    # scanning every arg on every graph invocation.
    act_input_paths: list[ActInputPath] = field(default_factory=list)

    # Map of effect type (ex. _EffectType.ORDERED) to token.  If there are
    # side-effectful operators, FunctionalTensorMode will populate this
    # dictionary telling us how many tokens we will need during tracing.
    tokens: dict[Any, torch.Tensor] = field(default_factory=dict)

    # Only filled in if/when we trace the joint function
    # If an input requires grad and is mutated in the backward, it is only safe to keep the mutation
    # in the graph if gradients are disabled while the backward runs
    # (grad mode is disabled by default when users run the backward, but can be turned on with create_graph=True)
    # At runtime during the backward, we use this list of indices to error properly if we find out
    # that it was not safe to include a backward mutation in the graph.
    indices_of_inputs_that_requires_grad_with_mutations_in_bw: list[int] = field(
        default_factory=list
    )

    # Indexes of saved tensors which are donated buffer.
    # Donated buffer means the tensor is not alias of any forward user input, forward user output,
    # and backward output.
    bw_donated_idxs: list[int] | None = None

    # Number of tokens used in backward, appended at the end of backward outputs.
    # Filled after tracing joint function.
    num_backward_tokens: int = 0

    # Number of rng states that will get thread into the forward and backward for
    # cudagraph compatible run_and_save_rng
    num_graphsafe_rng_states: int = 0

    graphsafe_rng_state_index: int | None = None

    # Device for graphsafe RNG states (supports CUDA, TPU, etc.)
    graphsafe_rng_device: torch.device | None = None

    # Stream indices for mutated inputs in the epilogue
    # Maps from index in mutated_inp_runtime_indices to the stream index that last touched
    # the storage of the tensor that will be copied back into the original input
    # None means use the current/default stream
    # This is populated during graph compilation when stream assignments are made
    mutated_inp_stream_indices: list[int | None] | None = None

    # compile ID string (e.g., "1/0") for error messages
    compile_id_str: str | None = None

    # help users identify where to add .detach() in their code
    tangent_source_stack_traces: list[str | None] | None = None

    def __post_init__(self) -> None:
        # pre-compute the indices of the inputs that are mutated.
        # When keep_input_mutations is set, we don't need to worry about our epilogue
        # handling data-only mutations, because we keep them directly in the graph.
        mutated_inp_runtime_indices = [
            i
            for i, m in enumerate(self.input_info)
            if (m.mutation_type == MutationType.MUTATED_OUT_GRAPH)
        ]

        mutated_graph_handled_indices = [
            i
            for i, m in enumerate(self.input_info)
            if m.mutation_type == MutationType.MUTATED_IN_GRAPH
        ]
        self.mutated_graph_handled_indices = mutated_graph_handled_indices
        self.num_mutated_graph_handled_indices = len(self.mutated_graph_handled_indices)

        mutated_graph_handled_indices_seen_by_autograd = [
            i
            for i in mutated_graph_handled_indices
            if not self.input_info[i].mutations_hidden_from_autograd
        ]

        self.mutated_graph_handled_indices_seen_by_autograd = (
            mutated_graph_handled_indices_seen_by_autograd
        )
        self.num_mutated_graph_handled_indices_seen_by_autograd = len(
            self.mutated_graph_handled_indices_seen_by_autograd
        )

        aliased_out_indices = [
            i
            for i, m in enumerate(self.output_info)
            if m.output_type
            not in [
                OutputType.non_alias,
                OutputType.unsafe_view_alias,
                OutputType.custom_function_view,
            ]
        ]
        unsafe_view_out_indices = [
            i
            for i, m in enumerate(self.output_info)
            if m.output_type is OutputType.unsafe_view_alias
        ]

        # This is pre-computed in post_init for perf.
        # It contains the index of every element
        # of input_info that corresponds to a mutation (data or metadata or both)
        self.mutated_inp_runtime_indices = mutated_inp_runtime_indices
        self.num_mutated_inp_runtime_indices = len(self.mutated_inp_runtime_indices)

        # This is pre-computed for perf.
        # It contains the index of every element
        # of output_info that corresponds to an alias (either of an input or intermediate)
        self.aliased_out_indices = aliased_out_indices
        self.unsafe_view_out_indices = unsafe_view_out_indices
        self.num_outputs = len(self.output_info)
        self.num_outputs_non_aliased = len(
            [
                x
                for x in self.output_info
                if x.output_type
                in [
                    OutputType.non_alias,
                    OutputType.unsafe_view_alias,
                    OutputType.custom_function_view,
                ]
            ]
        )
        self.num_outputs_aliased_to_inputs = len(
            [
                x
                for x in self.output_info
                if x.output_type
                in [
                    OutputType.alias_of_input,
                    OutputType.is_input,
                ]
            ]
        )
        self.num_unsafe_view_outputs = len(self.unsafe_view_out_indices)
        self.num_outputs_aliased_to_intermediates = len(
            [
                x
                for x in self.output_info
                if x.output_type
                in [
                    OutputType.alias_of_intermediate,
                    OutputType.alias_of_intermediate_save_as_output,
                    OutputType.alias_of_intermediate_base_is_user_output,
                ]
            ]
        )
        self.num_outputs_aliased = (
            self.num_outputs_aliased_to_inputs
            + self.num_outputs_aliased_to_intermediates
        )

        # Record dynamic outputs of the Dynamo traced forward graph
        # Mark them as dynamic at the end of the runtime wrapper
        self.dynamic_outputs = any(o.dynamic_dims for o in self.output_info)

        # Record the indices of dynamic outputs in the partitioned forward graph
        # Mark them as dynamic in the runtime wrapper
        # activation index -> dynamic dims indices
        self.dynamic_saved_tensors_idxs: dict[int, set[int]] = {}

        # See Note: [AOTAutograd Backward Guards]
        # This is pre-computed for fast asserts on the types of our grad_outputs in the backward.
        # Eventually, we should kill this and replace with real backward guards.
        # (we want to precompute the "runtime" types, so replace FakeTensor with torch.Tensor)
        self.output_types = [
            torch.Tensor if is_fake_tensor(x) else type(x) for x in self.traced_tangents
        ]

        self.is_rng_op_functionalized = config.functionalize_rng_ops
        # All of the above metadata is collected by tracing the fw function.
        # However, extra outputs for rng offsets behave differently. Both fwd
        # and bwd graphs have their own outputs for the total consumed offsets.
        # Unlike mutated inputs, we don't have to worry about sending the right
        # set of tensors between fwd and bwd. Fwd and bwd offsets are
        # independent and simpler to handle. Therefore, we track them
        # separately.
        self.num_outputs_rng_offset = 1 if self.is_rng_op_functionalized else 0

        # Our forward() returns both (tokens, mutated_inputs, outputs, output_intermediate_bases, saved_tensors, saved_symints)
        # Tokens will be split out before mutations/view handling and we do not count them here.
        self.num_forward_returns = (
            self.num_mutated_inp_runtime_indices
            + self.num_outputs
            + self.num_intermediate_bases
        )
        # In case of functionalization of rng ops, the fw_module returns one
        # additional output for rng offset. This rng offset is used right
        # away to advance the rng state, and is not passed on to the raw
        # outputs. However, we need to know the exact boundary to identify
        # which tensors to be saved for the bwd graph.  num_forward captures
        # this information.
        self.num_forward = self.num_forward_returns + self.num_outputs_rng_offset

    def make_runtime_safe(self) -> None:
        """
        There are various fields in ViewAndMutationMeta that aren't serializable. This function is called after all tracing
        is completed to simplify certain fields in the metadata so that they can be safely cached.

        Doing so may lose information (in the case of traced_tangents), but none of the information is needed at runtime.
        """
        # TODO: This function is only a best effort: there are other fields that may not be cache safe
        # (i.e., there's no guarantee that tensor_flatten() returns a serializable result), or that
        # SubclassCreationMeta is cache safe.
        if self.traced_tangent_metas is not None:
            raise AssertionError(
                "traced_tangent_metas should be None before calling make_runtime_safe"
            )

        def extract_metadata(t: object) -> tuple[Sequence[str], object] | None:
            if isinstance(t, torch.Tensor) and is_traceable_wrapper_subclass(t):
                (inner_tensors, flatten_spec) = t.__tensor_flatten__()  # type: ignore[attr-defined]
                # Technically, we only need the flatten_spec, not the inner tensors.
                # However, some Tensor subclasses (like TwoTensor) may have flatten_spec = None.
                # And we want to be able to assert that this metadata is non-None,
                # to distinguish between "this was a tensor subclass with no metadata" vs.
                # "this wasn't a tensor subclass at all".
                return (inner_tensors, flatten_spec)
            else:
                return None

        self.traced_tangent_metas = [extract_metadata(t) for t in self.traced_tangents]
        # Clear traced tangents at runtime
        self.traced_tangents = []
        for inp_meta in self.subclass_inp_meta:
            if isinstance(inp_meta, SubclassCreationMeta):
                inp_meta.make_runtime_safe()
        for inp_meta in self.subclass_fw_graph_out_meta:
            if isinstance(inp_meta, SubclassCreationMeta):
                inp_meta.make_runtime_safe()
        for inp_meta in self.subclass_tangent_meta:
            if isinstance(inp_meta, SubclassCreationMeta):
                inp_meta.make_runtime_safe()

        # Clear view_meta_sequence when it has symbolic inputs, since it won't
        # be used at runtime anyway (gen_alias_from_base skips view replay for
        # symbolic inputs) and the SymInt references make it unpicklable.
        for i, out_info in enumerate(self.output_info):
            if out_info.view_meta_sequence is not None and any(
                vm.has_symbolic_inputs for vm in out_info.view_meta_sequence.sequence
            ):
                self.output_info[i] = replace(out_info, view_meta_sequence=None)

    @property
    def tensors_saved_for_backwards_slice(self) -> slice:
        if self.num_symints_saved_for_bw is None:
            raise AssertionError("num_symints_saved_for_bw must not be None")
        if self.num_tensors_saved_with_no_vc_check is None:
            raise AssertionError("num_tensors_saved_with_no_vc_check must not be None")
        # Fast-path: if no tensors without VC check, just return the VC check slice
        if self.num_tensors_saved_with_no_vc_check == 0:
            return self.tensors_saved_for_backwards_with_vc_check_slice
        # Invariant: total tensors activations = (acts_with_vc_check, acts_no_vc_check)
        vc_slice = self.tensors_saved_for_backwards_with_vc_check_slice
        no_vc_slice = self.tensors_saved_for_backwards_no_vc_check_slice
        # Start should be the same (self.num_forward)
        if vc_slice.start != self.num_forward:
            raise AssertionError(
                f"vc_slice.start ({vc_slice.start}) != self.num_forward ({self.num_forward})"
            )
        # End is the end of the no_vc_check slice
        return slice(vc_slice.start, no_vc_slice.stop)

    @property
    def tensors_saved_for_backwards_with_vc_check_slice(self) -> slice:
        """
        Slice of forward outputs that are tensors saved for backward that
        require version counter checks (i.e., were saved via save_for_backward).
        """
        # See Note [Activations with no version counter checks in eager]
        if self.num_symints_saved_for_bw is None:
            raise AssertionError("num_symints_saved_for_bw must not be None")
        if self.num_tensors_saved_with_no_vc_check is None:
            raise AssertionError("num_tensors_saved_with_no_vc_check must not be None")
        # The tensors with VC check come first, followed by tensors without VC check
        num_no_vc_check = self.num_tensors_saved_with_no_vc_check
        num_opaque = self.num_opaque_objects_saved_for_bw or 0
        num_symints = self.num_symints_saved_for_bw
        num_trailing = num_no_vc_check + num_opaque + num_symints
        if num_trailing > 0:
            return slice(self.num_forward, -num_trailing if num_trailing != 0 else None)
        else:
            return slice(self.num_forward, None)

    @property
    def tensors_saved_for_backwards_no_vc_check_slice(self) -> slice:
        """
        Slice of forward outputs that are tensors saved for backward that
        do NOT require version counter checks (i.e., were stashed on ctx
        rather than via save_for_backward in an autograd.Function).
        """
        if self.num_symints_saved_for_bw is None:
            raise AssertionError("num_symints_saved_for_bw must not be None")
        if self.num_tensors_saved_with_no_vc_check is None:
            raise AssertionError("num_tensors_saved_with_no_vc_check must not be None")
        num_no_vc_check = self.num_tensors_saved_with_no_vc_check
        num_opaque = self.num_opaque_objects_saved_for_bw or 0
        num_symints = self.num_symints_saved_for_bw
        if num_no_vc_check == 0:
            return slice(0, 0)  # empty slice
        num_trailing = num_opaque + num_symints
        if num_trailing > 0:
            return slice(-num_no_vc_check - num_trailing, -num_trailing)
        else:
            return slice(-num_no_vc_check, None)

    @property
    def opaque_objects_saved_for_backwards_slice(self) -> slice:
        num_opaque = self.num_opaque_objects_saved_for_bw or 0
        num_symints = self.num_symints_saved_for_bw or 0
        if num_opaque > 0:
            if num_symints > 0:
                return slice(-num_opaque - num_symints, -num_symints)
            else:
                return slice(-num_opaque, None)
        else:
            return slice(0, 0)  # empty slice

    @property
    def symints_saved_for_backwards_slice(self) -> slice:
        if self.num_symints_saved_for_bw is None:
            raise AssertionError("num_symints_saved_for_bw must not be None")
        if self.num_symints_saved_for_bw > 0:
            return slice(-self.num_symints_saved_for_bw, None)
        else:
            return slice(0, 0)  # empty slice

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ViewAndMutationMeta):
            return NotImplemented
        return (
            self.input_info == other.input_info
            and self.output_info == other.output_info
            and self.num_intermediate_bases == other.num_intermediate_bases
            and self.keep_input_mutations == other.keep_input_mutations
            and self.is_rng_op_functionalized == other.is_rng_op_functionalized
            and self.num_outputs_rng_offset == other.num_outputs_rng_offset
            and len(self.traced_tangents) == len(other.traced_tangents)
            and all(
                x.shape == y.shape and x.dtype == y.dtype
                for x, y in zip(self.traced_tangents, other.traced_tangents)
            )
            and self.num_backward_tokens == other.num_backward_tokens
        )


@dataclass(eq=False)
class SubclassMeta:
    # A copy of all forward metadata, but computed on the *dense* tensor forward (after desugaring subclasses)
    # So for example, if the user had a model containing two `TwoTensor` inputs,
    # Then `SubclassMeta.fw_metadata.input_infos` would have length 4 here.
    fw_metadata: ViewAndMutationMeta

    # Note: [Computing Subclass Metadata about grad_inputs]
    # Given a list of flattened, plain tensor grad_inputs, this tells us how to reconstruct the grad_input subclasses
    #
    # You might think: why not just assume that all grad_inputs will have the same subclass-ness as the original inputs?
    # (AOTAutograd generally assumes other properties, e.g. that grad_outputs are contiguous)
    #
    # This doesn't really work though. take this example:
    #
    # def f(DoubleTensor, DenseTensor):
    #     return DoubleTensor  * DenseTensor
    #
    # In the above example, the .grad field of *both* DoubleTensor and DenseTensor will be a DoubleTensor.
    # When we trace out a joint fw-bw graph, we'll end up returning two subclasses for the two grad_inputs.
    # This means that our backward graph will return 4 outputs (two dense tensors for each DoubleTensor grad_input)
    # and we need to properly store the metadata that tells us how to turn these 4 outputs back into DoubleTensors.
    #
    # Note that this info **cannot** easily be figured out from ViewAndMutationMeta.
    # We can only compute this info by tracing the entire joint and examining the grad_inputs that we computed.
    #
    # See Note: [AOTAutograd Backward Guards]
    # This will also eventually require us to install backward guards,
    # in case we made incorrect assumptions about the subclass-ness of our grad_outputs
    #
    # Optional field because we don't compute for inference graphs
    grad_input_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = None

    def __init__(self) -> None:
        # The fields in this class get set after its construction.
        pass


# This class exists because:
# - the autograd.Function.forward() in aot autograd returns outputs that might alias inputs
# - we only care about the metadata on those aliases, so we can regenerate them.
#   We do not want them to participate in the autograd.Function.
# We do that by wrapping them in an opaque class, so the autograd.Function
# does not know to treat them as tensors.
@dataclass(frozen=True)
class TensorAlias:
    alias: torch.Tensor


@dataclass
class BackwardSignature:
    """
    Provides information about the backward section of an exported
    joint forward-backward graph.
    For a particular fx GraphModule, this class contains information on:
    (1) A mapping from each gradient (backwards output) to the parameter
        it corresponds to (forward input)
    (2) A mapping from each gradient (backwards output) to the user input
        it corresponds to (forward input)
    (3) Which of the forward outputs corresponds to the loss, that we backprop on.

    Each string name is the `node.name` of the corresponding node in the fx graph.
    """

    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str


GraphOutputName = NewType("GraphOutputName", str)
GraphInputName = NewType("GraphInputName", str)
FQN = NewType("FQN", str)


@dataclass
class GraphSignature:
    """
    Provides information about an exported module.
    For a particular fx GraphModule, this class contains information on:
    (1) Which graph inputs are parameters, buffers, or user inputs
    (2) (for params/buffers) a mapping from the name of each graph argument
        to its parameter/buffer FQN in the original nn.Module.
    (3) If there are input mutations, these are represented as extra outputs
        in the fx GraphModule. We provide a mapping from these
        extra output names to the names of the actual inputs.
    (4) The pytree metadata on how to flatten/unflatten inputs and outputs.
        The corresponding FX GraphModule only accepts and returns
        pytree-flattened inputs/outputs.
    (5) (Optionally) if the FX is a joint forward-backward graph, we provide
        a signature on the backward section of the joint graph.
    """

    parameters: list[FQN]
    buffers: list[FQN]

    user_inputs: list[GraphInputName]
    user_outputs: list[GraphOutputName]
    inputs_to_parameters: dict[GraphInputName, FQN]
    inputs_to_buffers: dict[GraphInputName, FQN]

    # If the user's module mutates a buffer,
    # it's represented in the graph as an extra graph output.
    # This dict is a mapping from
    # "graph outputs that correspond to updated buffers"
    # to the FQN names of those mutated buffers.
    buffers_to_mutate: dict[GraphOutputName, FQN]
    parameters_to_mutate: dict[GraphOutputName, FQN]
    user_inputs_to_mutate: dict[GraphOutputName, GraphInputName]

    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec

    backward_signature: BackwardSignature | None

    input_tokens: list[GraphInputName]
    output_tokens: list[GraphOutputName]

    @classmethod
    def from_tracing_metadata(
        cls,
        *,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
        graph_input_names: list[str],
        graph_output_names: list[str],
        view_mutation_metadata: ViewAndMutationMeta,
        named_parameters: list[str],
        named_buffers: list[str],
        num_user_inputs: int,
        num_user_outputs: int,
        trace_joint: bool,
        loss_index: int | None,
        backward_signature: BackwardSignature | None,
    ) -> GraphSignature:
        graph_inputs = graph_input_names
        graph_outputs = graph_output_names
        parameters = list(named_parameters)
        buffers = list(named_buffers)
        num_tokens = len(view_mutation_metadata.tokens)

        # Calling convention assumptions:
        # (1) graph inputs = (input_tokens, params, buffers, user_inputs)
        # (2) graph outputs = (output_tokens, mutated_inputs, user_outs, param_gradients)
        # (If we are capturing an inference graph, this convention is identical
        #  except that param_gradients is empty)
        # See Note [Side-Effectful Tokens in AOTAutograd] for information on tokens

        # Address input calling conventions:
        start, stop = 0, num_tokens
        input_tokens = graph_inputs[start:stop]

        start, stop = stop, stop + len(parameters)
        inputs_to_parameters = dict(zip(graph_inputs[start:stop], parameters))

        start, stop = stop, stop + len(buffers)
        inputs_to_buffers = dict(
            zip(
                graph_inputs[start:stop],
                buffers,
            )
        )

        start, stop = stop, stop + num_user_inputs
        user_inputs = graph_inputs[start:stop]

        # We should've gone through all the inputs now
        if len(graph_inputs) - stop != 0:
            raise AssertionError(
                f"expected all graph_inputs consumed, but {len(graph_inputs) - stop} remain"
            )

        # Address output calling conventions:
        start, stop = 0, num_tokens
        output_tokens = graph_outputs[start:stop]

        names = [*input_tokens, *parameters, *buffers, *user_inputs]
        mutations: list[str] = []
        for idx, input_info in enumerate(view_mutation_metadata.input_info):
            if input_info.mutates_data:
                if trace_joint:
                    # Only buffers can be mutated, not parameters
                    if idx < len(parameters):
                        raise AssertionError(
                            f"expected idx ({idx}) >= len(parameters) ({len(parameters)}) when tracing joint"
                        )
                mutations.append(names[idx + num_tokens])

        if len(mutations) != view_mutation_metadata.num_mutated_inp_runtime_indices:
            raise AssertionError(
                f"len(mutations) ({len(mutations)}) != "
                f"num_mutated_inp_runtime_indices ({view_mutation_metadata.num_mutated_inp_runtime_indices})"
            )

        start, stop = (
            stop,
            stop + view_mutation_metadata.num_mutated_inp_runtime_indices,
        )
        outputs_to_mutations = dict(zip(graph_outputs[start:stop], mutations))

        user_inputs_to_mutate: dict[GraphOutputName, GraphInputName] = {}
        buffers_to_mutate: dict[GraphOutputName, FQN] = {}
        parameters_to_mutate: dict[GraphOutputName, FQN] = {}
        for output_name, mutation_name in outputs_to_mutations.items():
            if mutation_name in user_inputs:
                # pyrefly: ignore [unsupported-operation]
                user_inputs_to_mutate[output_name] = mutation_name
            else:
                if mutation_name not in buffers and mutation_name not in parameters:
                    raise AssertionError(
                        f"mutation_name '{mutation_name}' not found in buffers or parameters"
                    )
                if mutation_name in buffers:
                    # pyrefly: ignore [unsupported-operation]
                    buffers_to_mutate[output_name] = mutation_name
                else:
                    # pyrefly: ignore [unsupported-operation]
                    parameters_to_mutate[output_name] = mutation_name

        start, stop = stop, stop + num_user_outputs
        user_outputs = graph_outputs[start:stop]

        unused_outputs = len(graph_outputs) - stop
        if backward_signature is not None:
            unused_outputs -= len(backward_signature.gradients_to_parameters) + len(
                backward_signature.gradients_to_user_inputs
            )
        if unused_outputs != 0:
            raise AssertionError(f"expected unused_outputs == 0, got {unused_outputs}")

        return GraphSignature(
            parameters=parameters,  # type: ignore[arg-type]
            buffers=buffers,  # type: ignore[arg-type]
            user_inputs=user_inputs,  # type: ignore[arg-type]
            user_outputs=user_outputs,  # type: ignore[arg-type]
            inputs_to_buffers=inputs_to_buffers,  # type: ignore[arg-type]
            inputs_to_parameters=inputs_to_parameters,  # type: ignore[arg-type]
            user_inputs_to_mutate=user_inputs_to_mutate,
            buffers_to_mutate=buffers_to_mutate,  # type: ignore[arg-type]
            parameters_to_mutate=parameters_to_mutate,  # type: ignore[arg-type]
            in_spec=in_spec,
            out_spec=out_spec,
            backward_signature=backward_signature,
            input_tokens=input_tokens,  # type: ignore[arg-type]
            output_tokens=output_tokens,  # type: ignore[arg-type]
        )


@dataclass
class AOTAutogradCacheInfo:
    cache_key: str
    start_time_ns: int
    forward_symints: list[torch.SymInt]


@dataclass
class CacheableAOTConfig:
    """
    Serializable subset of AOTConfig used by cache keys and cached entries.
    """

    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool
    is_export: bool = False
    no_tangents: bool = False
    dynamic_shapes: bool = False
    aot_autograd_arg_pos_to_source: list[Source] | None = None
    static_input_indices: list[int] | None = None
    enable_log: bool = True
    # this is always false outside of export.
    pre_dispatch: bool = False
    precompile_backend_id: str | None = None

    def __post_init__(self) -> None:
        if self.pre_dispatch and not self.is_export:
            raise AssertionError("Can only have pre_dispatch IR for export.")


@dataclass(frozen=True)
class AOTConfig:
    """
    Configuration for AOTDispatcher
    """

    fw_compiler: Callable[..., Any] | None
    bw_compiler: Callable[..., Any] | None
    partition_fn: Callable[..., Any] | None
    decompositions: dict[OpOverload, Callable[..., Any]] | None
    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool
    is_export: bool = False
    no_tangents: bool = False
    dynamic_shapes: bool = False
    aot_autograd_arg_pos_to_source: list[Source] | None = None
    static_input_indices: list[int] | None = None
    inference_compiler: Callable[..., Any] | None = None
    enable_log: bool = True
    # this is always false outside of export.
    pre_dispatch: bool = False
    # Key to use for AOTAutogradCache
    cache_info: AOTAutogradCacheInfo | None = None
    # If we should ignore the shape_env in the ambient tracing_context.
    # The net effect is that if dynamic shapes are on, we end up
    # specializing on example_inputs.
    # Used only by standalone_compile.
    ignore_shape_env: bool = False
    precompile_backend_id: str | None = None
    force_non_lazy_backward_lowering: bool = False
    # This config makes sure to check certain things like
    # mutating input with req_grad in export joint tracing.
    export_trace_joint: bool = False
    disable_functionalization: bool = False
    # If True, disable TorchFunctionMetadataMode during make_fx tracing.
    # This mode is used to track torch_fn metadata but can interfere with
    # certain tracing scenarios.
    _disable_torch_fn_metadata_mode: bool = False

    def to_cacheable(self) -> CacheableAOTConfig:
        return CacheableAOTConfig(
            num_params_buffers=self.num_params_buffers,
            aot_id=self.aot_id,
            keep_inference_input_mutations=self.keep_inference_input_mutations,
            is_export=self.is_export,
            no_tangents=self.no_tangents,
            dynamic_shapes=self.dynamic_shapes,
            aot_autograd_arg_pos_to_source=self.aot_autograd_arg_pos_to_source,
            static_input_indices=self.static_input_indices,
            enable_log=self.enable_log,
            pre_dispatch=self.pre_dispatch,
            precompile_backend_id=self.precompile_backend_id,
        )

    def __post_init__(self) -> None:
        if self.pre_dispatch:
            if not self.is_export:
                raise AssertionError("Can only have pre_dispatch IR for export.")


# TODO: types here
# plain_tensor_trace_fn, when it is joint, has tuple structure on the trace
# info too!
# TODO: this needs to be generic, parameterized on AOTDescriptor
SubclassTracingInfo = collections.namedtuple(
    "SubclassTracingInfo",
    [
        "plain_tensor_trace_fn",
        "plain_tensor_args",
        "plain_tensor_args_descs",
        "maybe_subclass_meta",
    ],
)


@dataclass
class AOTState:
    """
    When we run AOTAutograd, this class encapsulates the state in the compiler which
    must be preserved across stages.  This is state in the traditional sense (not an
    environment) because some values in this structure change as we progress through
    pipelines in AOTAutograd.
    """

    # Whether or not we need to handle autograd when doing graph capture and
    # compilation.  Although the calling convention for non-autograd graph
    # capture in AOTAutograd is simple and can be relied upon, the autograd
    # capture calling convention is quite complicated and in general you are
    # only expected to pass to aot_stage2_compile to process.
    needs_autograd: bool

    # The FAKE flat arguments which we will do tracing with.  Although you
    # might naively expect this to be immutable, it's not: when we perform
    # tracing, we may execute code that modifies the metadata of inputs,
    # causing the args to become "invalid".  It's also nontrivial to have a
    # "golden" set of fake values and deepcopy them just in time when you
    # might destructively mutate them (Voz and I tried very hard to do this).
    # So we just periodically renew this field.  Don't worry too much about
    # this unless you're specifically trying to track down an input metadata
    # mutation bug.
    #
    # (By the way, this is NEVER the joint inputs!  Those only ever go in
    # AOTGraphCapture)
    flat_args: list[FxValue]

    # The descriptor for each argument in flat_args.
    flat_args_descs: list[AOTInput]

    # This contains view and mutation information about the function, which we
    # detected by doing an initial trace when we created this state.
    fw_metadata: ViewAndMutationMeta

    # Top-level configuration. Stage-local compiler choices are threaded
    # explicitly rather than mutating this object in place.
    aot_config: AOTConfig

    # When performing AOTAutograd traces and other passes, we typically
    # require a lot of active context managers; most typically these either
    # (1) ensure we are faithfully replicating the original PyTorch context
    # managers or (2) toggle some behaviors in PyTorch to make it more
    # suitable for tracing.  When you use AOTState, you're expected to have
    # created an ExitStack, entered it; then while we are running AOTAutograd
    # we will add things onto the stack as necessary.  When you're all done
    # with processing AOTAutograd, you can exit this stack.  All functions
    # that take AOTState expect the ExitStack to not have been exited yet.
    #
    # TODO: We potentially could offer a resumable context manager, where you
    # can cancel it and reenable it later when you need it.
    stack: contextlib.ExitStack

    # The fake tensor mode used during tracing.  This is useful for later
    # operations that need to create new fake tensors consistent with the
    # original trace.
    fake_mode: FakeTensorMode


FxValue = Tensor | int | SymInt | BackwardState | CustomClassBase


class CompilerWrapper:
    """
    AOTAutograd needs to do many transformations to the calling convention of the user function
    it is tracing, e.g., deduplicating inputs, unpacking subclasses, etc.  CompilerWrapper lets
    us factor these into compositional stages so we can handle each transformation incrementally
    instead of having to do it all at once.

    Since there is a calling convention change, there are two parts to the wrapper:

    1. The prologue, which is about compile-time behavior: given this original function, what
       is the new function with modified calling convention that we should trace with AOTAutograd
       to get the FX graph we will do joint passes, partitioning and ultimate Inductor compilation on?
       We get (flat_fn, flat_args), the original function under trace and inputs we were
       going to feed it, and produce a new function and new inputs to feed it.

    2. The epilogue, which is about run-time behavior: we have now compiled the modified calling
       convention function, we need to wrap it so that we have a new function that has the
       original calling convention of the original function, so that our users can call it
       at the old signature they expected.  We get (compiled_fn, real arguments), the newly
       compiled function we need to wrap.

    Note about caching: we do NOT directly serialize the runtime wrappers; instead, they
    are reapplied to compiled_fn after we have finished deserializing the compiled_fn.

    Extra metadata that is needed to compute pre or post compile can be passed in via attributes.
    """

    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]:
        """
        Process the inputs to the compiler_fn. You can pass in extra metadata via kwargs.
        Args:
        flat_fn: The function to compile
        flat_args: Metadata from example inputs of the function to compile
        aot_config: AOTConfig passed in at compile time
        fw_metadata: ViewAndMutationMeta generated from flat_fn and flat_args
        """
        return flat_fn, flat_args, flat_args_descs, fw_metadata

    def post_compile(
        self,
        compiled_fn: Callable[_P, _R],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[_P, _R]:
        """
        Given an output of the compiler, wrap it with information received from prologue.
        Args:
        compiled_fn: Callable after calling compiler_fn
        aot_config: AOTConfig after calling prologue
        runtime_metadata: ViewAndMutationMeta after calling all wrappers's pre_compile steps.
        Example:

        def wrapped_compiled_fn(args):
            # do something with args, aot_config, fw_metadata
            return compiled_fn(args)

        return wrapped_compiled_fn
        """
        return compiled_fn


class InductorWrapper:
    """
    This is sort of like CompilerWrapper, but it happens at a different part of the lifecycle:
    it talks about transformations we do to the traced and partitioned FX graph before we
    send it to the Inductor compiler.

    Once again, there are two parts:

    1. The prologue, which "modifies" the FX graph before we send it to
       Inductor.  I say "modifies" because... we don't really actually do
       anything nontrivial in either of our two implementations.
    2. The epilogue, which modifies the compiled function produced by Inductor

    Although hypothetically these wrappers could be used compositionally in a centralized
    wrappers list, in practice they seem to just be invoked manually when needed.

    NB: The flat_args input is sometimes mutated.  This is probably naughty but whatever.
    """

    def pre_compile(
        self,
        fw_module: torch.fx.GraphModule,
        flat_args: list[Tensor],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> None:
        """
        Process the inputs to the compiler_fn. You can pass in extra metadata via kwargs.
        Args:
        flat_fn: The function to compile
        flat_args: Metadata from example inputs of the function to compile
        aot_config: AOTConfig passed in at compile time
        fw_metadata: ViewAndMutationMeta generated from flat_fn and flat_args
        """
        return

    def post_compile(
        self,
        compiled_fn: Callable[_P, _R],
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[_P, _R]:
        """
        Given an output of the compiler, wrap it with information received from prologue.
        Args:
        compiled_fn: Callable after calling compiler_fn
        aot_config: AOTConfig after calling prologue
        runtime_metadata: ViewAndMutationMeta after calling all wrappers's pre_compile steps.
        Example:

        def wrapped_compiled_fn(args):
            # do something with args, aot_config, fw_metadata
            return compiled_fn(args)

        return wrapped_compiled_fn
        """
        return compiled_fn


@dataclass
class AOTGraphCapture:  # Produced by aot_stage1_graph_capture
    # AOTAutograd typically operates by taking complicated graphs and
    # desugaring them into simpler graphs that use PyTorch features.  These
    # wrappers establish invariants so that when we actually do tracing we can
    # assume these invariants hold, leading to a simpler tracing
    # implementation.  However, this means that we have to keep track of how
    # to enter/exit these wrappers when passing inputs into the compiled
    # graph, among other things!
    wrappers: list[CompilerWrapper]

    # The actual captured graph module.  In some circumstances (export) this
    # graph has a specific calling convention that can be relied upon by
    # external callers.  In other situations, the calling convention is
    # unspecified and only aot_stage2_compile knows how to deal with them.
    graph_module: torch.fx.GraphModule

    # When compiling with autograd support, this is the joint_inputs, which is
    # larger than the original flat_args as all tangents get inputs.  The
    # tuple organizes into primals and tangents.  When not autograd it's just
    # a plain list.
    updated_flat_args: list[Any] | tuple[list[Any], list[Any]]

    updated_flat_args_descs: list[AOTInput] | tuple[list[AOTInput], list[AOTInput]]

    # Metadata about subclass inputs/outputs in the graph trace.
    maybe_subclass_meta: Any


FakifiedFlatArgs = NewType("FakifiedFlatArgs", list[Any])


TOutputCode = TypeVar("TOutputCode", bound="OutputCode")


class AOTDispatchCompiler(Protocol):
    """
    Represents a fw or bw_compiler passed to AOTAutograd.
    """

    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[InputType],
    ) -> Any: ...


# TODO: bikeshed on this name
class SerializableAOTDispatchCompiler(AOTDispatchCompiler):
    """
    Represents an AOTDispatchCompiler that returns an OutputCode, and is
    therefore cacheable. SerializableAOTDispatchCompiler always return an OutputCode.
    A _CompileFxCallable usually gets converted into an AOTDispatchCompiler after binding all of
    the kwargs in _CompileFxKwargs.
    """

    def __init__(
        self,
        output_code_ty: type[TOutputCode],
        compiler_fn: Callable[[torch.fx.GraphModule, Sequence[InputType]], TOutputCode],
    ) -> None:
        # pyrefly: ignore [invalid-type-var]
        self.output_code_ty = output_code_ty
        # pyrefly: ignore [invalid-type-var]
        self.compiler_fn = compiler_fn

    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[InputType],
    ) -> OutputCode:
        output_code = self.compiler_fn(gm, example_inputs)
        return output_code


class FlatFn(Protocol):
    def __call__(self, *args: FxValue) -> list[FxValue]: ...


class TraceFn(Protocol):
    def __call__(self, *args: FxValue) -> tuple[list[FxValue], list[AOTOutput]]: ...


class PreppedForAutogradTraceFn(Protocol):
    def __call__(
        self,
        *args: FxValue,
    ) -> tuple[tuple[list[FxValue], list[bool]], list[AOTOutput]]: ...


class JointTraceFn(Protocol):
    handle: JointFnHandle

    def __call__(
        self, primals: list[FxValue], tangents: list[FxValue]
    ) -> tuple[
        tuple[list[FxValue], list[Tensor | None]],
        tuple[list[AOTOutput], list[AOTOutput | None]],
    ]: ...


@dataclass
class JointWithDescriptors:
    _aot_state: AOTState
    _aot_graph_capture: AOTGraphCapture

    # The exact order parameters and buffers are expected to be passed into
    # the final compiled function.  Parameters before buffers.
    params_spec: list[str]
    buffers_spec: list[str]

    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec

    @property
    def graph_module(self) -> torch.fx.GraphModule:
        return self._aot_graph_capture.graph_module

    @graph_module.setter
    def graph_module(self, value: torch.fx.GraphModule) -> None:
        self._aot_graph_capture.graph_module = value

    @property
    def fake_mode(self) -> FakeTensorMode:
        return self._aot_state.fake_mode

    def cache_hash(self) -> str:
        """
        Return a hash string suitable for use as a cache key. This method
        exists to decouple cache key generation from __str__/__repr__, so
        that display-oriented changes don't accidentally affect caching.
        """
        from hashlib import sha256

        return sha256(str(self).encode("utf-8")).hexdigest()
<|endoftext|>"""Reconstruct live Python values as standalone, auditable source.

The composer (to_standalone_python.py) inlines AOTAutograd's runtime wrappers, which
close over baked metadata objects (view-replay recipes, tensor-subclass specs, ...).
This module turns such a live object into a Python expression that reconstructs an
EQUAL value -- emitted as readable source, never a pickle / base64 blob -- or raises
NotImplementedError if it cannot be expressed. ``emit_value`` is the public entry
point; ``_emit_value`` is the internal recursion (it threads the traversal state);
``_emit_via_reduce`` is the pickle-reduce-as-source fallback and ``_REBUILD_HELPER`` is
the ``_rebuild`` function it references in generated code. Kept as a leaf module so it
is safe to import anywhere in the package.
"""

from __future__ import annotations

import inspect
from typing import Any, NewType, TYPE_CHECKING


# id() returns an object identity used here only as a recursion/alias key, not an
# arbitrary int. Shadowing the builtin under TYPE_CHECKING tags every ``id(obj)`` as
# ``_ObjId`` so ``set[_ObjId]`` propagates without wrapping each call site; the real
# builtin runs at runtime (NewType is identity), so behavior is unchanged.
_ObjId = NewType("_ObjId", int)

if TYPE_CHECKING:

    def id(obj: object, /) -> _ObjId: ...


# The runtime helper for reduce-based metadata reconstruction (see _emit_via_reduce).
# Emitted only when some baked value actually needs it -- many graphs (e.g. a plain
# module with no tensor-subclass metadata) reference none, so it would just be dead
# code. Includes its own trailing blank lines so the splice stays clean.
_REBUILD_HELPER: list[str] = [
    "def _rebuild(obj, state):",
    "    # Apply pickle-style reduce state to a freshly __new__'d object, emitted as",
    "    # readable source instead of an opaque blob (no pickle.loads). Used for the",
    "    # opaque value leaves of baked metadata -- e.g. tensor-subclass placement",
    "    # objects -- mirroring pickle's load_build: prefer __setstate__, else update",
    "    # __dict__ (and slot state for the (dict, slots) 2-tuple form).",
    "    if state is None:",
    "        return obj",
    "    if hasattr(obj, '__setstate__'):",
    "        obj.__setstate__(state)",
    "        return obj",
    "    slotstate = None",
    "    if isinstance(state, tuple) and len(state) == 2:",
    "        state, slotstate = state",
    "    if state:",
    "        obj.__dict__.update(state)",
    "    if slotstate:",
    "        for _k, _v in slotstate.items():",
    "            setattr(obj, _k, _v)",
    "    return obj",
    "",
    "",
]


def _emit_importable(obj: Any, imports: set[str]) -> str:
    """Return an expression referencing ``obj`` via its defining module (recording the
    import). Works for classes, functions, and modules. Raises if ``obj`` is not
    reachable as ``module.qualname`` (e.g. a local or lambda), since then it cannot be
    reproduced in the standalone module without embedding it."""
    import importlib

    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module or not qualname or "<" in qualname:
        raise NotImplementedError(
            f"compile_to_python cannot reference {obj!r} by import: it has no "
            "module/qualname or is a local definition."
        )
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part, None)
    if target is not obj:
        raise NotImplementedError(
            f"compile_to_python cannot reference {qualname} from {module}: it does "
            "not round-trip to the same object."
        )
    imports.add(f"import {module}")
    return f"{module}.{qualname}"


def _reject_shared_mutable(obj: Any, _seen: set[_ObjId]) -> None:
    """Fail loud when a mutable object is reached more than once in a single traversal:
    it would be emitted as two independent source literals, silently dropping the
    shared-identity aliasing (mutating one copy would no longer affect the other), which
    is not expressible as a source literal today. Routed here are the kinds reconstructed
    as fresh MUTABLE copies whose aliasing a flat literal cannot preserve: list / dict /
    set, bytearray, non-frozen dataclasses, and the opaque _emit_via_reduce value objects.
    Kinds treated as immutable value objects are exempt -- scalars, tuple / frozenset, frozen
    dataclasses, and the recipe-like value objects emitted by their own branches
    (functools.partial, ViewMeta, ViewMetaSequence), which are never mutated after
    reconstruction -- so a shared one of those still emits fine. ``_seen`` is threaded in
    place across the whole traversal (unlike the per-level in-progress cycle set), so it
    catches shared siblings, not just self-referential ancestors."""
    if id(obj) in _seen:
        raise NotImplementedError(
            f"compile_to_python cannot bake a shared mutable {type(obj).__qualname__}: "
            "it is reached more than once in the value being baked, so emitting it as "
            "independent source literals would silently drop the shared-identity "
            "aliasing."
        )
    _seen.add(id(obj))


def emit_value(obj: Any, imports: set[str]) -> str:
    """Emit a Python expression (valid in the generated module) that rebuilds ``obj``
    from source -- the pickle-free replacement for embedding a base64 blob.

    Public entry point: recurses through containers and metadata so the artifact stays
    auditable and exec'ing it never runs ``pickle.loads``. Bottoms out for opaque value
    objects (e.g. tensor-subclass placement objects) at the pickle reduce protocol, but
    EMITTED AS SOURCE via ``_rebuild`` (see ``_emit_via_reduce``). Raises
    NotImplementedError at any leaf that cannot be expressed as source (a live tensor, a
    lambda, a still-symbolic Sym*), for a self-referential structure, or for a shared
    mutable object whose aliasing a flat source literal cannot preserve.

    Any import needed by the emitted expression is recorded into ``imports`` (as ``import
    module`` statements)."""
    return _emit_value(obj, imports, set(), set())


def _emit_value(
    obj: Any,
    imports: set[str],
    _in_progress: set[_ObjId],
    _seen: set[_ObjId],
) -> str:
    """Internal recursion behind ``emit_value``; requires the traversal state its guards
    thread. Call ``emit_value`` from outside this module.

    ``_in_progress`` is an identity-keyed set of the objects currently being emitted on
    this recursion PATH (copied per level, like a call stack): revisiting one means the
    metadata is self-referential and cannot be a source literal, so we raise rather than
    recurse forever. ``_seen`` is an identity-keyed set of every MUTABLE object already
    emitted anywhere in this traversal (threaded in place, like ``imports``): revisiting
    one means the same mutable object is shared across positions, whose aliasing a flat
    literal cannot preserve, so we raise (see ``_reject_shared_mutable``)."""
    import dataclasses
    import enum
    import functools
    import math
    import types

    import torch
    from torch._C import _functionalization as _F
    from torch.fx.experimental.symbolic_shapes import SymIntEqByExpr

    from .functional_utils import ViewMetaSequence

    # Live tensors / storages must never be baked: their reduce form is a
    # ``torch.storage._load_from_bytes(b'...')`` pickle blob, which both embeds the
    # raw weight bytes and invokes ``pickle.loads`` at exec time -- violating the
    # module's no-weights-baked / never-pickle.loads / fully-auditable guarantees.
    # Reject them explicitly here, before they can fall through to _emit_via_reduce.
    # A non-Tensor object whose reduce IS ``_load_from_bytes`` (a wrapper that
    # delegates pickling to a storage's bytes) is caught at the reduce callable in
    # _emit_via_reduce, where that callable is actually visible.
    if isinstance(
        obj, (torch.Tensor, torch.storage.TypedStorage, torch.UntypedStorage)
    ):
        raise NotImplementedError(
            f"compile_to_python cannot bake a live {type(obj).__qualname__} into "
            "standalone source: that would embed raw tensor bytes and require "
            "pickle.loads at exec time. The standalone artifact bakes no weights."
        )

    # Primitives first, before any cycle / alias bookkeeping: those guards never matter
    # for immutable scalars, and returning here skips the per-level in-progress copy and
    # the mutable-alias check for the (very common) scalar leaves.
    #
    # Non-finite floats: repr() gives bare ``inf`` / ``-inf`` / ``nan``, which are
    # NameErrors in the generated module (it imports no such names). Emit a
    # self-contained constructor instead. Likewise for a complex with a non-finite
    # component (repr would be e.g. ``(inf+0j)``).
    if isinstance(obj, float) and not math.isfinite(obj):
        if math.isnan(obj):
            return "float('nan')"
        return "float('inf')" if obj > 0 else "float('-inf')"
    if isinstance(obj, complex) and not (
        math.isfinite(obj.real) and math.isfinite(obj.imag)
    ):
        # Use the module-level recursion directly (the in-body shadow is not defined until
        # after the guards below); the components are floats, so they bottom out above.
        real = _emit_value_recurse(obj.real, imports, _in_progress, _seen)
        imag = _emit_value_recurse(obj.imag, imports, _in_progress, _seen)
        return f"complex({real}, {imag})"
    # Plain constants reproduce via repr (but not IntEnum/StrEnum members, whose repr
    # is not valid source -- those fall through to the enum handler below). Match EXACT
    # builtin types only: a subclass of a builtin scalar (e.g. a ``str`` subclass, or an
    # ``int`` subclass with a constructor-style ``__repr__``) would mis-bake under repr
    # -- losing its type or emitting a NameError -- so it must fall through to the
    # importable/reduce path, which reconstructs it faithfully or raises.
    if obj is None or (
        not isinstance(obj, enum.Enum)
        and type(obj) in (bool, int, float, complex, str, bytes)
    ):
        return repr(obj)
    # bytearray is the one mutable builtin whose repr round-trips, so it is emitted as a
    # repr literal like the immutable scalars above -- but being mutable it needs the same
    # shared-alias guard the containers get (reached twice it would become two independent
    # literals, silently dropping the aliasing). It cannot be self-referential, so it needs
    # only the alias check and no cycle guard.
    if type(obj) is bytearray:
        _reject_shared_mutable(obj, _seen)
        return repr(obj)

    # Cycle guard: thread an identity-keyed in-progress set down the recursion so a
    # self-referential metadata object raises NotImplementedError naming the offending
    # type instead of recursing until RecursionError. Primitives already returned above,
    # so no immutable-scalar exemption is needed here.
    if id(obj) in _in_progress:
        raise NotImplementedError(
            f"compile_to_python cannot bake {type(obj).__qualname__}: it is "
            "self-referential, which is not expressible as source."
        )
    # Alias guard: a mutable container reached twice anywhere in the traversal would be
    # emitted as two independent literals, silently dropping the shared-identity aliasing.
    # Match EXACT list/dict/set only -- those are the types the container branches below
    # emit; a subclass falls through to _emit_via_reduce, which tracks it there (matching
    # here too would double-count the same object and falsely reject a single subclass).
    if type(obj) in (list, dict, set):
        _reject_shared_mutable(obj, _seen)

    _child = _in_progress | {id(obj)}

    def _emit_value(obj: Any, imports: set[str]) -> str:
        # Shadow the module-level recursion entry point so every recursive call in
        # this function body automatically threads the current in-progress set (with
        # the parent ``obj`` added) plus the traversal-wide seen set, without touching
        # each call site.
        return _emit_value_recurse(obj, imports, _child, _seen)

    # torch scalar singletons whose repr round-trips, plus device/Size/SymInt.
    if isinstance(obj, (torch.dtype, torch.layout, torch.memory_format)):
        imports.add("import torch")
        return repr(obj)
    if isinstance(obj, torch.device):
        imports.add("import torch")
        return f"torch.device({str(obj)!r})"
    if isinstance(obj, torch.Size):
        imports.add("import torch")
        return f"torch.Size([{', '.join(_emit_value(x, imports) for x in obj)}])"
    if isinstance(obj, torch.SymInt):
        concrete = obj.node.maybe_as_int()
        if concrete is None:
            raise NotImplementedError(
                "compile_to_python cannot bake a symbolic SymInt; precompile "
                "specializes to static shapes."
            )
        return repr(concrete)
    # SymBool / SymFloat get the same treatment as SymInt (bake the concrete value, reject
    # a still-symbolic one with the static-shapes message) so all sym scalar types are
    # handled consistently instead of falling into the reduce path with an opaque error.
    if isinstance(obj, torch.SymBool):
        concrete_bool = obj.node.maybe_as_bool()
        if concrete_bool is None:
            raise NotImplementedError(
                "compile_to_python cannot bake a symbolic SymBool; precompile "
                "specializes to static shapes."
            )
        return repr(concrete_bool)
    if isinstance(obj, torch.SymFloat):
        concrete_float = obj.node.maybe_as_float()
        if concrete_float is None:
            raise NotImplementedError(
                "compile_to_python cannot bake a symbolic SymFloat; precompile "
                "specializes to static shapes."
            )
        return repr(concrete_float)

    # Importable definitions: classes, functions, modules.
    if isinstance(obj, type) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return _emit_importable(obj, imports)
    if isinstance(obj, types.ModuleType):
        imports.add(f"import {obj.__name__}")
        return obj.__name__

    # Python enums and pybind11 enums (e.g. InverseReturnMode) both reference a member
    # by name off the importable type. pybind enum members are not singletons (as_tuple
    # hands back fresh copies), so match by value via the type's __members__ map.
    members = getattr(type(obj), "__members__", None)
    name = getattr(obj, "name", None)
    is_enum_like = isinstance(obj, enum.Enum) or (
        members is not None and name in members and members[name] == obj
    )
    if is_enum_like:
        if name is not None and members is not None and name in members:
            return f"{_emit_importable(type(obj), imports)}.{name}"
        # A combined Flag/IntFlag member (A | B) has no single member name, so emit by
        # value -- ``Type(value)`` reproduces the same combined member. A non-Flag enum
        # with no resolvable name is not source-expressible, so reject it rather than
        # emit ``Type.None`` (a SyntaxError that would break the whole module).
        if isinstance(obj, enum.Flag):
            cls = _emit_importable(type(obj), imports)
            return f"{cls}({_emit_value(obj.value, imports)})"
        # Defensive backstop, effectively unreachable for constructible inputs: a
        # pure-Python non-Flag enum.Enum always resolves a name in __members__ (it rejects
        # unregistered values with ValueError) and so returns by-name above, and a pybind
        # enum-like value only reaches here via the ``name in members`` arm that already
        # returned. Kept so any future name-less enum-like type fails loudly, not silently.
        raise NotImplementedError(
            f"compile_to_python cannot bake enum member {obj!r}: it has no resolvable "
            "member name and is not a Flag."
        )

    if isinstance(obj, functools.partial):
        parts = [_emit_value(obj.func, imports)]
        parts += [_emit_value(a, imports) for a in obj.args]
        parts += [
            f"{k}={_emit_value(v, imports)}" for k, v in (obj.keywords or {}).items()
        ]
        imports.add("import functools")
        return f"functools.partial({', '.join(parts)})"

    # AOT view-replay recipes: ViewMeta C++ objects round-trip through as_tuple();
    # ViewMetaSequence has no public constructor so use its _from_parts factory;
    # SymIntEqByExpr wraps a sympy expr that must be a concrete integer here.
    if isinstance(obj, _F.ViewMeta):
        # as_tuple lives on the ViewMeta subclass bindings (create_binding_with_pickle),
        # not on the base ViewMeta stub, so pyrefly cannot see it on the isinstance type.
        tup = obj.as_tuple()  # pyrefly: ignore[missing-attribute]
        return f"{_emit_importable(type(obj), imports)}({_emit_value(tuple(tup), imports)})"
    if isinstance(obj, ViewMetaSequence):
        cls = _emit_importable(ViewMetaSequence, imports)
        seq = _emit_value(list(obj.sequence), imports)
        return f"{cls}._from_parts({seq}, {_emit_value(obj.metadata, imports)})"
    if isinstance(obj, SymIntEqByExpr):
        if not getattr(obj.val, "is_Integer", False):
            raise NotImplementedError(
                "compile_to_python cannot bake symbolic view metadata; precompile "
                "specializes to static shapes."
            )
        return f"{_emit_importable(SymIntEqByExpr, imports)}({int(obj.val)})"

    # namedtuple before plain tuple (it has a richer, by-name constructor). The plain
    # container branches match EXACT builtin types only (like the scalar branch above): a
    # container SUBCLASS (e.g. a ``list`` subclass carrying extra state) would otherwise be
    # silently downcast to its base type, dropping the subclass and its state. Subclasses
    # fall through to _emit_via_reduce, which reconstructs them faithfully or rejects them
    # (its listitems/dictitems guard fires for list/dict subclasses).
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        cls = _emit_importable(type(obj), imports)
        return f"{cls}({', '.join(_emit_value(x, imports) for x in obj)})"
    if type(obj) is tuple:
        items = [_emit_value(x, imports) for x in obj]
        return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"
    if type(obj) is list:
        return f"[{', '.join(_emit_value(x, imports) for x in obj)}]"
    if type(obj) is set or type(obj) is frozenset:
        ctor = "frozenset" if isinstance(obj, frozenset) else "set"
        # Iteration order of a set is not byte-stable across processes, so emit the
        # elements in a canonical order: sort them by their EMITTED SOURCE EXPRESSION --
        # the exact text that lands in the artifact, so it is the only key guaranteed to
        # be byte-stable across processes. A ``repr``-keyed sort would silently admit a
        # custom-but-nondeterministic ``__repr__`` (e.g. one embedding ``id(self)``),
        # making the emitted source differ run to run, and a numeric sort would raise on a
        # ``nan`` element (``sorted`` does not) and does not exist for mixed types. Each
        # element is emitted exactly once (into the real ``imports``, since every element
        # emitted is one that lands in the output) and then the resulting expressions are
        # sorted -- which forces each element through the same source-expressibility gate,
        # so a non-source-expressible element raises here exactly as at final emit.
        exprs = sorted(_emit_value(x, imports) for x in obj)
        return f"{ctor}([{', '.join(exprs)}])"
    if type(obj) is dict:
        items = [
            f"{_emit_value(k, imports)}: {_emit_value(v, imports)}"
            for k, v in obj.items()
        ]
        return f"{{{', '.join(items)}}}"

    # Dataclasses (e.g. MetadataKey, SubclassViewMetaSequence) reproduce field-by-field
    # through their generated constructor, passing only the init fields. That path is
    # faithful ONLY when the object carries no state outside those init fields: a
    # non-init field (init=False) holding state, or a __post_init__ that derives state
    # the constructor call would not reproduce, would be silently dropped. Guard it the
    # same way the rest of this module guards opaque leaves -- reject rather than emit a
    # subtly-wrong artifact -- by round-tripping through the constructor and requiring
    # the rebuilt instance to compare equal to the original. The in-scope metadata
    # types (e.g. MetadataKey) are plain value dataclasses, so they round-trip and keep
    # working; a type with hidden/derived state does not and raises NotImplementedError
    # naming itself. This strategy needs value equality: a dataclass declared eq=False
    # inherits object.__eq__ (identity), so rebuilt == obj is ALWAYS False even for a
    # pure value object -- detect that and fall through to _emit_via_reduce (which has its
    # own source-expressibility rejects) rather than spuriously rejecting it.
    if (
        dataclasses.is_dataclass(obj)
        and not isinstance(obj, type)
        and type(obj).__eq__ is not object.__eq__
    ):
        # A non-frozen dataclass is mutable, so it gets the same shared-alias guard as the
        # containers: reached twice it would be emitted as two independent constructor
        # calls, silently dropping the shared-identity aliasing. A frozen dataclass is
        # immutable and stays exempt, like a tuple.
        if not obj.__dataclass_params__.frozen:
            _reject_shared_mutable(obj, _seen)
        cls = _emit_importable(type(obj), imports)
        fields = dataclasses.fields(obj)
        init_kwargs = {f.name: getattr(obj, f.name) for f in fields if f.init}
        try:
            # type(obj) is the concrete dataclass at runtime, not the protocol pyrefly
            # infers from the is_dataclass narrowing; the call is correct.
            rebuilt = type(obj)(**init_kwargs)  # pyrefly: ignore[bad-instantiation]
            round_trips = rebuilt == obj
        except Exception:
            round_trips = False
        if not round_trips:
            raise NotImplementedError(
                f"compile_to_python cannot bake dataclass {type(obj).__qualname__}: it "
                "does not round-trip through its constructor from its init fields alone "
                "(it likely has a stateful non-init field or a __post_init__ deriving "
                "state the constructor call would not reproduce), so emitting only the "
                "init fields would silently drop that state."
            )
        kw = [f"{k}={_emit_value(v, imports)}" for k, v in init_kwargs.items()]
        return f"{cls}({', '.join(kw)})"

    return _emit_via_reduce(obj, imports, _child, _seen)


# The recursion entry point used by the in-body shadow of ``_emit_value`` and by
# ``_emit_via_reduce`` to thread the identity-keyed in-progress + seen traversal sets.
_emit_value_recurse = _emit_value


def _emit_via_reduce(
    obj: Any,
    imports: set[str],
    _in_progress: set[_ObjId],
    _seen: set[_ObjId],
) -> str:
    """Last resort for opaque value objects (e.g. DTensor placements, which are C++
    objects with no source-friendly constructor): reconstruct from the pickle reduce
    protocol, but EMITTED AS SOURCE -- ``cls.__new__(cls)`` plus ``_rebuild`` applying
    the reduce state -- so there is no ``pickle.loads`` at exec time and the bytes
    are readable. The reduce state recurses back through ``_emit_value``, so any
    non-source-expressible leaf inside it still raises."""
    import copyreg

    import torch

    # These reduce value objects are mutable, so the same guard the containers get in
    # _emit_value applies: an object reached twice would be emitted as two independent
    # reconstructions, silently dropping the shared-identity aliasing.
    _reject_shared_mutable(obj, _seen)

    def _emit_value(obj: Any, imports: set[str]) -> str:
        # Continue threading the cycle guard + seen set through the reduce-state recursion.
        return _emit_value_recurse(obj, imports, _in_progress, _seen)

    try:
        reduced = obj.__reduce_ex__(2)
    except Exception as e:
        raise NotImplementedError(
            f"compile_to_python cannot make {type(obj).__module__}."
            f"{type(obj).__qualname__} self-contained: it is not source-expressible "
            f"and has no usable reduce ({type(e).__name__})."
        ) from e
    if not isinstance(reduced, tuple) or len(reduced) < 2:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: "
            "unsupported __reduce__ form."
        )
    func, args = reduced[0], reduced[1]
    # The reduce protocol requires the second element to be a tuple of constructor args;
    # validate once here so all three func branches below normalize a malformed reduce
    # into the uniform NotImplementedError contract rather than leaking a bare TypeError.
    if not isinstance(args, tuple):
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: reduce "
            "args field is not a tuple."
        )
    state = reduced[2] if len(reduced) > 2 else None
    listitems = reduced[3] if len(reduced) > 3 else None
    dictitems = reduced[4] if len(reduced) > 4 else None
    # A reduce whose callable is ``torch.storage._load_from_bytes`` would embed raw
    # bytes and require a pickle.loads-equivalent at exec time. This is the ONLY place
    # that callable is visible (it is the reduce result's func, not the object's
    # __reduce_ex__ method), so reject it here to uphold the no-bytes / fully-auditable
    # guarantee for any non-Tensor wrapper that delegates pickling to storage bytes.
    if func is getattr(torch.storage, "_load_from_bytes", None):
        raise NotImplementedError(
            f"compile_to_python cannot bake {type(obj).__qualname__} into standalone "
            "source: its reduce is torch.storage._load_from_bytes, which would embed "
            "raw bytes and require pickle.loads at exec time."
        )
    if listitems is not None or dictitems is not None:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: reduce "
            "produced list/dict items (container subclass)."
        )
    # A non-None 6th element is a protocol-5 ``state_setter``: the object opted out of
    # the default __setstate__/__dict__ install that ``_rebuild`` implements, so applying
    # state via _rebuild would silently use the wrong mechanism. Reject rather than emit
    # a subtly-wrong object.
    state_setter = reduced[5] if len(reduced) > 5 else None
    if state_setter is not None:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: reduce "
            "produced a state_setter (protocol-5 form) that _rebuild cannot apply."
        )
    if func is getattr(copyreg, "__newobj__", None):
        if not args:
            raise NotImplementedError(
                f"compile_to_python cannot reconstruct {type(obj).__qualname__}: "
                "__newobj__ reduce produced no class argument."
            )
        cls = _emit_value(args[0], imports)
        extra = ", ".join(_emit_value(a, imports) for a in args[1:])
        base = f"{cls}.__new__({cls}{', ' + extra if extra else ''})"
    elif func is getattr(copyreg, "__newobj_ex__", None):
        if not (
            isinstance(args, tuple)
            and len(args) == 3
            and isinstance(args[1], tuple)
            and isinstance(args[2], dict)
        ):
            raise NotImplementedError(
                f"compile_to_python cannot reconstruct {type(obj).__qualname__}: "
                "__newobj_ex__ reduce did not produce a (cls, args, kwargs) triple."
            )
        cls_obj, new_args, new_kwargs = args
        cls = _emit_value(cls_obj, imports)
        pos = ", ".join(_emit_value(a, imports) for a in new_args)
        kw = ", ".join(f"{k}={_emit_value(v, imports)}" for k, v in new_kwargs.items())
        joined = ", ".join(p for p in (pos, kw) if p)
        base = f"{cls}.__new__({cls}{', ' + joined if joined else ''})"
    else:
        func_expr = _emit_value(func, imports)
        base = f"{func_expr}({', '.join(_emit_value(a, imports) for a in args)})"
    if state is None:
        return base
    return f"_rebuild({base}, {_emit_value(state, imports)})"
<|endoftext|>from __future__ import annotations

import dataclasses
import itertools
from typing import Any, TYPE_CHECKING

import torch
from torch._custom_class_base import CustomClassBase
from torch._library.opaque_object import is_opaque_symbolic_type
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .schemas import OpaqueMeta


if TYPE_CHECKING:
    from collections.abc import Iterable


# This is technically very similar to SubclassCreatingMeta
# in aot_autograd, but we don't need all the stuff in there
# so just recreated a new dataclass.
@dataclasses.dataclass
class SubclassCreationMeta:
    start_idx: int
    num_tensors: int
    class_type: Any
    # None means the attr is a plain tensor (base case of recursion)
    attrs: dict[str, SubclassCreationMeta | OpaqueMeta | None]
    metadata: Any
    outer_size: Iterable[None | int | torch.SymInt]
    outer_stride: Iterable[None | int | torch.SymInt]


class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors) -> torch.Tensor:  # type: ignore[no-untyped-def]
        todo: list[torch.Tensor | CustomClassBase] = list(tensors)

        def _unwrap_tensor_subclasses(subclass_meta, tensors, offset):  # type: ignore[no-untyped-def]
            if subclass_meta is None:
                return tensors[offset], offset + 1
            inner_tensors = {}
            for attr, meta in subclass_meta.attrs.items():
                if isinstance(meta, OpaqueMeta):
                    inner_tensors[attr] = tensors[offset]
                    offset += 1
                else:
                    built_tensor, offset = _unwrap_tensor_subclasses(
                        meta, tensors, offset
                    )
                    inner_tensors[attr] = built_tensor
            rebuilt = subclass_meta.class_type.__tensor_unflatten__(
                inner_tensors,
                subclass_meta.metadata,
                subclass_meta.outer_size,
                subclass_meta.outer_stride,
            )
            return rebuilt, offset

        return _unwrap_tensor_subclasses(self.subclass_meta, todo, 0)[0]

    def right_inverse(
        self, tensor: torch.Tensor
    ) -> list[torch.Tensor | CustomClassBase]:
        if type(tensor) is torch.Tensor:
            raise AssertionError("tensor must be a subclass, not torch.Tensor")
        plain_tensors: list[torch.Tensor | CustomClassBase] = []

        def _create_subclass_meta(tensor, idx, plain_tensor_container):  # type: ignore[no-untyped-def]
            if type(tensor) is torch.Tensor:
                plain_tensor_container.append(tensor)
                return None, idx + 1
            inner_tensors_attrnames, metadata = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
            new_idx = idx
            attr_to_meta: dict[str, SubclassCreationMeta | OpaqueMeta | None] = {}
            for attr in inner_tensors_attrnames:
                val = getattr(tensor, attr)
                match val:
                    case CustomClassBase():
                        if not is_opaque_symbolic_type(type(val)):
                            raise ValueError(
                                f"{type(val).__name__!r} found in tensor attrs of "
                                f"{type(tensor).__name__}.__tensor_flatten__(). "
                                "Only tensors and reference-type opaques are allowed "
                                "in tensor attrs."
                            )
                        attr_to_meta[attr] = OpaqueMeta()
                        plain_tensor_container.append(val)
                        new_idx += 1
                    case torch.Tensor():
                        subclass_meta, new_idx = _create_subclass_meta(
                            val, new_idx, plain_tensor_container
                        )
                        attr_to_meta[attr] = subclass_meta
                    case _:
                        raise AssertionError(
                            f"expected Tensor or CustomClassBase, got {type(val)}"
                        )
            return (
                SubclassCreationMeta(
                    start_idx=idx,
                    num_tensors=new_idx - idx,
                    class_type=type(tensor),
                    attrs=attr_to_meta,
                    metadata=metadata,
                    outer_size=tensor.size(),
                    outer_stride=tensor.stride(),
                ),
                new_idx,
            )

        self.subclass_meta = _create_subclass_meta(tensor, 0, plain_tensors)[0]
        return plain_tensors


def unwrap_tensor_subclass_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Model transformation that replaces all the parameters that are subclasses to plain tensors.
    This reduces runtime overhead of flattening/unflattening the parameters.

    This transformation adds parametrization with `torch.nn.utils.parametrize`.
    The FQNs of the subclass parameters will be changed and state_dict will become incompatible with the original model.
    E.g.
    Original model state_dict: {"p1": torch.testing._internal.TwoTensor}
    becomes: {"parametrizations.p2.original0": torch.Tensor, "parametrizations.p2.original1": torch.Tensor}

    """
    for name, tensor in itertools.chain(
        list(module.named_parameters(recurse=False)),
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        list(module.named_buffers(recurse=False)),
    ):
        if is_traceable_wrapper_subclass(tensor):
            torch.nn.utils.parametrize.register_parametrization(
                module, name, UnwrapTensorSubclass()
            )

    for child in module.children():
        unwrap_tensor_subclass_parameters(child)

    return module
<|endoftext|>"""
Contains various utils for AOTAutograd, including those for handling collections.
"""

import copy
import dataclasses
import logging
import operator
import sys
import warnings
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import partial, wraps
from typing import Any, overload, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar, TypeVarTuple, Unpack

import torch
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class_obj
from torch._logging import getArtifactLogger
from torch._subclasses.fake_tensor import is_fake_tensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import py_sym_types


_T = TypeVar("_T")
if TYPE_CHECKING:
    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    from .schemas import AOTConfig, ViewAndMutationMeta


KNOWN_TYPES = [
    torch.Tensor,
    BackwardState,
    int,
    str,
    float,
    bool,
    type(None),
    *py_sym_types,
    FakeScriptObject,
    torch.ScriptObject,
]

aot_graphs_effects_log = getArtifactLogger(__name__, "aot_graphs_effects")
annotation_log = getArtifactLogger(__name__, "annotation")

strict_zip = partial(zip, strict=True)


def get_loaded_async_collective_tensor_type() -> type["AsyncCollectiveTensor"] | None:
    """Return the ACT type if distributed collectives are already loaded."""
    if "torch.distributed._functional_collectives" not in sys.modules:
        return None
    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    return AsyncCollectiveTensor


def import_async_collective_tensor_type() -> type["AsyncCollectiveTensor"]:
    """Import and return the ACT type."""
    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    return AsyncCollectiveTensor


def is_async_collective_tensor_type(pytype: type) -> bool:
    """Whether pytype is the AsyncCollectiveTensor type (False if not loaded)."""
    act = get_loaded_async_collective_tensor_type()
    return act is not None and pytype is act


def partial_flatten_asdict(obj: object) -> Any:
    if dataclasses.is_dataclass(obj):
        return {
            field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)
        }
    elif isinstance(obj, (list, tuple)):
        return obj.__class__([partial_flatten_asdict(item) for item in obj])
    elif isinstance(obj, dict):
        return {k: partial_flatten_asdict(v) for k, v in obj.items()}
    else:
        return obj


@overload
def normalize_as_list(x: _T) -> list[_T]: ...
@overload
def normalize_as_list(x: tuple[_T, ...]) -> list[_T]: ...
@overload
def normalize_as_list(x: list[_T]) -> list[_T]: ...


def normalize_as_list(x: object) -> list[object]:
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


def _get_autocast_states() -> list[Any]:
    return [
        torch.is_autocast_enabled("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.get_autocast_dtype("cuda"),
        torch.get_autocast_dtype("cpu"),
        torch.is_autocast_cache_enabled(),
    ]


def make_boxed_func(f: Callable[..., Any]) -> Callable[[list[Any]], Any]:
    @simple_wraps(f)
    def g(args: list[Any]) -> Any:
        return f(*args)

    # pyrefly: ignore[missing-attribute]
    g._boxed_call = True
    return g


def make_boxed_compiler(
    compiler: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(compiler)
    def f(fx_g: Any, inps: Any) -> Any:
        out_f = compiler(fx_g, inps)
        fx_g = make_boxed_func(out_f)
        return fx_g

    return f


def call_func_at_runtime_with_args(
    f: Callable[..., Any],
    args: Sequence[Any],
    steal_args: bool = False,
    disable_amp: bool = False,
) -> list[Any]:
    if not steal_args:
        args = list(args)
    if not isinstance(args, list):
        raise AssertionError(f"args must be a list, got {type(args)}")

    context = torch._C._DisableAutocast if disable_amp else nullcontext
    with context():
        if getattr(f, "_boxed_call", False):
            out = normalize_as_list(f(args))
        else:
            # TODO: Please remove soon
            # https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670
            warnings.warn(
                "Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. "
                "Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. "
                "See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.",
                stacklevel=2,
            )
            out = normalize_as_list(f(*args))
    return out


# Inspired by autodidax (thanks!)
class PytreeThunk:
    spec: pytree.TreeSpec | None = None
    # These are some kinda dumb microoptimizations that save about 3-4 us of overhead.
    is_simple: bool | None = (
        None  # if the output spec is a tuple/list, we won't bother unflattening it.
    )
    is_really_simple: bool | None = None  # if the output spec is a LeafSpec

    def set(self, spec: pytree.TreeSpec) -> None:
        if not (self.spec is None or self.spec == spec):
            raise AssertionError(f"spec mismatch: existing={self.spec}, new={spec}")
        if spec is None:
            raise AssertionError("spec must not be None")
        self.spec: pytree.TreeSpec = spec
        if self.spec.type in {tuple, list} and all(
            child.is_leaf() for child in spec.children()
        ):
            self.is_simple = True
        if self.spec.is_leaf():
            self.is_really_simple = True

    def unflatten(self, x: Sequence[Any]) -> Any:
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        if self.spec is None:
            raise AssertionError("spec must be set before calling unflatten")
        return pytree.tree_unflatten(x, self.spec)


# Creates a function that returns flattened inputs and outputs
# Also returns the output tree spec, which is needed to recover the "unflattened"
# output tree structure later.
def create_tree_flattened_fn(
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any] | None = None,
) -> tuple[Callable[..., list[Any]], PytreeThunk]:
    if kwargs is None:
        kwargs = {}
    # Save the args_spec for flat_tensor_args to unflatten while tracing
    _, tensor_args_spec = pytree.tree_flatten((args, kwargs))
    out_spec = PytreeThunk()

    def flat_fn(*flat_args: Any) -> list[Any]:
        # The input are flattened tensor args. Prepare the args in the
        # order that original function expects. Add static args as well.
        # They will appear as tensor constants in the traced graph.
        nonlocal out_spec
        args, kwargs = pytree.tree_unflatten(flat_args, tensor_args_spec)
        tree_out = fn(*args, **kwargs)
        flat_out, spec = pytree.tree_flatten(tree_out)
        for i in flat_out:
            is_known_type = isinstance(i, tuple(KNOWN_TYPES)) or is_custom_class_obj(i)
            if not is_known_type:
                raise RuntimeError(
                    f"Found {type(i)} in output, which is not a known type. "
                    "If this type holds tensors, you need to register a pytree for it. "
                    "See https://github.com/pytorch/functorch/issues/475 for a brief "
                    "explanation why. If you don't need to register a pytree, please "
                    "leave a comment explaining your use case and we'll make this more "
                    "ergonomic to deal with"
                )
        out_spec.set(spec)
        return flat_out

    # Can't use functools.wraps here because the wrapper has different
    # calling convention
    if hasattr(fn, "_orig_mod"):
        # pyrefly: ignore[missing-attribute]
        flat_fn._orig_mod = fn._orig_mod

    return flat_fn, out_spec


# This function takes in a tensor t, and returns one of t, t.view(), or t.clone().
# When tracing the joint forward + backward, for any inputs in the graph that are mutated,
# we need to clone them first (and similarly for metadata-only mutations, we need to view them first).
# The idea is that when we trace the backward, we need to pass in the *original* primals
# to autograd.grad(), before they were mutated.
# Note: when we have synthetic base inputs, we need to clone them *before* creating views off of them.
# This means that "idx" here represents the index of the (potentially) synthetic base.
# What we need to do is:
# (1) map the current (post-synthetic-base calling convention) input argument index
#     to int index pre-synthetic-base-calling-convention.
# (2) There could be multiple, if this index corresponds to a synthetic base
#     that has multiple input aliases.
# (3) If any of those corresponding inputs get metadata mutations, then we clone the base.
def maybe_to_fresh_input(idx: int, t: Any, meta: "ViewAndMutationMeta") -> Any:
    if not isinstance(t, torch.Tensor):
        return t
    if idx in meta.mutated_inp_runtime_indices:
        # We only need to bother cloning mutated inputs that participate in autograd.
        if meta.input_info[idx].requires_grad and meta.input_info[idx].mutates_data:
            # Make sure the primal we pass to autograd.grad()
            # sees the tensor before the mutation
            return t.clone()
        if meta.input_info[idx] and meta.input_info[idx].mutates_metadata:
            # Make sure the primal we pass to autograd.grad()
            # sees the tensor before the metadata mutation
            return t.view(t.shape)
    return t


def is_with_effects(node: torch.fx.Node) -> bool:
    if (
        node.op == "call_function"
        and node.target is torch.ops.higher_order.with_effects
    ):
        return True
    elif (
        node.op == "call_function"
        and node.target is torch.ops.higher_order.invoke_subgraph
    ):
        # Check if subgraph has effects by looking in the cache
        from torch._guards import InvokeSubgraphCache, TracingContext

        tracing_ctx = TracingContext.try_get()
        if tracing_ctx:
            invoke_subgraph_cache = tracing_ctx.hop_dispatch_set_cache.get_cache(
                torch.ops.higher_order.invoke_subgraph
            )
            if invoke_subgraph_cache:
                if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
                    raise AssertionError(
                        f"expected InvokeSubgraphCache, got {type(invoke_subgraph_cache)}"
                    )
                # pyrefly: ignore[bad-argument-type]
                effects = invoke_subgraph_cache.get_effects(node.args[1])
                return effects is not None
    return False


def unlift_tokens(
    fw_module: torch.fx.GraphModule,
    fw_metadata: "ViewAndMutationMeta",
    aot_config: "AOTConfig",
    bw_module: torch.fx.GraphModule | None = None,
) -> None:
    # Remove the tokens from the inputs/outputs of the graph since inductor does
    # not want these extra inputs/outputs, and replace them with
    # _make_token() to create a token, and _sink_tokens() to collect the
    # tokens.  See Note [Side-Effectful Tokens in AOTAutograd]
    # Logic:
    # 1. In the case of with_effects:
    #   Before:
    #   ```
    #   def forward(self, token, arg1_1):
    #       with_effects = torch.ops.higher_order.with_effects(token, ...)
    #       getitem = with_effects[0]
    #       getitem_1 = with_effects[0]
    #       return (getitem, getitem_1)
    #   ```
    #
    #   After:
    #   ```
    #   def forward(self, arg1_1):
    #       _make_token_default = torch.ops.prims._make_token.default()
    #       with_effects = torch.ops.higher_order.with_effects(_make_token_default, ...)
    #       getitem = with_effects[0]
    #       getitem_1 = with_effects[0]
    #       _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem]);
    #       return (getitem_1,)
    #   ```
    #
    # 2. In the case of an invoke_subgraph node, we will use the
    # InvokeSubgraphCache to determine if the subgraph has effects. Then we will
    # turn it into a `with_effects` node. This is so that at the toplevel graph,
    # the nodes will have the correct with_effects threading. We will apply this
    # pass recursively to submodules so the tokens will be removed from the
    # subgraph's inputs.
    #
    #   Before:
    #   ```
    #   def forward(self, token, arg1_1):
    #       repeated_subgraph0 = self.repeated_subgraph0
    #       invoke_subgraph = torch.ops.higher_order.invoke_subgraph(
    #           repeated_subgraph0, 'subgraph_0', token, x, arg1_1)
    #       getitem = invoke_subgraph[0]
    #       getitem_1 = invoke_subgraph[1]
    #       return (getitem, getitem1)
    #   ```
    #
    #   After:
    #   ```
    #   def forward(self, arg1_1):
    #       _make_token_default = torch.ops.prims._make_token.default()
    #       repeated_subgraph0 = self.repeated_subgraph0
    #       with_effects_1 = torch.ops.higher_order.with_effects(
    #           _make_token_default, torch.ops.higher_order.invoke_subgraph,
    #           repeated_subgraph0, 'subgraph_0', arg1_1)
    #       getitem = with_effects_1[0]
    #       getitem_1 = with_effects_1[1];  with_effects_1 = None
    #       _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem])
    #       return (getitem_1,)
    #   ```
    #
    # 3. The toplevel module should have the following invariants:
    #   forward:
    #     expected_num_erased_inputs == len(fw_metadata.tokens)
    #     expected_num_erased_outputs == len(fw_metadata.tokens)
    #   backward:
    #     expected_num_erased_inputs == fw_metadata.num_backward_tokens
    #     expected_num_erased_outputs == fw_metadata.num_backward_tokens
    num_forward_tokens = len(fw_metadata.tokens)
    num_backward_tokens = fw_metadata.num_backward_tokens

    def replace_input_token_with_make_token(
        module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> None:
        with module.graph.inserting_before(node):
            new_token_node = module.graph.call_function(
                torch.ops.prims._make_token.default, ()
            )
            new_token_node.meta["val"] = torch.tensor([])
            new_token_node.meta["tensor_meta"] = torch.tensor([])
            node.replace_all_uses_with(new_token_node)
            module.graph.erase_node(node)

    def get_output_tokens(node: torch.fx.Node) -> set[torch.fx.Node]:
        output_tokens = set()
        for user in list(node.users.keys()):
            # Check if this is a getitem accessing index 0 (the token)
            if (
                user.op == "call_function"
                and user.target is operator.getitem
                and len(user.args) > 1
                and user.args[1] == 0
            ):
                # Check if this getitem is used in an output
                for user_user in list(user.users.keys()):
                    if user_user.op == "output":
                        output_tokens.add(user)
        return output_tokens

    def _unlift_tokens_from_module_helper(
        module: torch.fx.GraphModule,
        subgraph_str: str,
        expected_num_erased: int | None,
    ) -> None:
        input_token_nodes = set()
        output_token_nodes = set()

        for node in module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.higher_order.with_effects
            ):
                if node.args[0].op == "placeholder":
                    input_token_nodes.add(node.args[0])
                    replace_input_token_with_make_token(module, node.args[0])

                tokens_from_with_effects = get_output_tokens(node)
                output_token_nodes = output_token_nodes | tokens_from_with_effects

            elif (
                node.op == "call_function"
                and node.target is torch.ops.higher_order.invoke_subgraph
            ):
                subgraph_node, identifier, *operands = node.args

                # Check if subgraph has effects by looking in the cache
                from torch._guards import InvokeSubgraphCache, TracingContext

                effects = None
                tracing_ctx = TracingContext.try_get()
                if tracing_ctx:
                    invoke_subgraph_cache = (
                        tracing_ctx.hop_dispatch_set_cache.get_cache(
                            torch.ops.higher_order.invoke_subgraph
                        )
                    )
                    if invoke_subgraph_cache:
                        if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
                            raise AssertionError(
                                f"expected InvokeSubgraphCache, got {type(invoke_subgraph_cache)}"
                            )
                        effects = invoke_subgraph_cache.get_effects(identifier)

                if effects is not None:
                    # Wrap invoke_subgraph with with_effects
                    # Before: invoke_subgraph(subgraph, id, token, *args) -> (token_out, result)
                    # After: with_effects(token, invoke_subgraph, subgraph, id, *args) -> (token_out, result)
                    #
                    # Note: The subgraph itself will be unlifted separately when we iterate
                    # through named_modules() below.

                    num_tokens = len(effects)
                    if num_tokens != 1:
                        raise AssertionError(
                            f"Multiple token subgraph NYI, got {num_tokens} tokens"
                        )
                    token_args = operands[:num_tokens]
                    non_token_args = operands[num_tokens:]

                    # Create with_effects wrapper around invoke_subgraph
                    # with_effects(token, op, *args) where op is invoke_subgraph
                    # Pass the subgraph and non-token args to invoke_subgraph
                    with module.graph.inserting_before(node):
                        new_node = module.graph.call_function(
                            torch.ops.higher_order.with_effects,
                            # pyrefly: ignore [bad-argument-type]
                            (
                                token_args[0],  # pyrefly: ignore[bad-argument-type]
                                torch.ops.higher_order.invoke_subgraph,
                                subgraph_node,
                                identifier,
                                *tuple(non_token_args),
                            ),
                        )
                        node.replace_all_uses_with(new_node)
                        new_node.meta = node.meta
                        module.graph.erase_node(node)

                    for token in token_args:
                        if token.op == "placeholder":
                            input_token_nodes.add(token)
                            replace_input_token_with_make_token(module, token)

                    # Get output tokens from the new with_effects node
                    tokens_from_invoke_subgraph = get_output_tokens(new_node)
                    output_token_nodes = (
                        output_token_nodes | tokens_from_invoke_subgraph
                    )

        if not output_token_nodes and not input_token_nodes:
            return

        output_node = next(reversed(module.graph.find_nodes(op="output")))
        if output_node is None:
            raise AssertionError("output node not found in graph")
        with module.graph.inserting_before(output_node):
            module.graph.call_function(
                torch.ops.prims._sink_tokens.default,
                (list(output_token_nodes),),
            )
        new_out_args = tuple(
            [out for out in output_node.args[0] if out not in output_token_nodes]
        )
        output_node.args = (new_out_args,)

        if expected_num_erased:
            if len(input_token_nodes) != expected_num_erased:
                raise AssertionError(
                    f"{subgraph_str} num_erased_inputs:{len(input_token_nodes)} "
                    f"{input_token_nodes} != expected {expected_num_erased} \n"
                    f"{fw_module.print_readable(print_output=False)}"
                )
            if len(output_token_nodes) != expected_num_erased:
                raise AssertionError(
                    f"{subgraph_str} num_erased_outs:{len(output_token_nodes)} "
                    f"{output_token_nodes} != expected {expected_num_erased} \n"
                    f"{fw_module.print_readable(print_output=False)}"
                )

        module.recompile()

    def unlift_tokens_from_module(
        module: torch.fx.GraphModule, subgraph_str: str, expected_num_erased: int
    ) -> None:
        for name, m in module.named_modules():
            if isinstance(m, torch.fx.GraphModule):
                if name == "":
                    _unlift_tokens_from_module_helper(
                        m, subgraph_str, expected_num_erased
                    )
                else:
                    # Subgraph -- we may or may not have effects applied
                    _unlift_tokens_from_module_helper(m, f"{subgraph_str}_{name}", None)

    if num_forward_tokens > 0:
        if aot_config.enable_log:
            from torch._dynamo.utils import lazy_format_graph_code

            aot_graphs_effects_log.debug(
                "%s",
                lazy_format_graph_code(
                    "Forward graph before unlifting tokens",
                    fw_module,
                    aot_config.aot_id,
                    include_stride=True,
                    include_device=True,
                    colored=True,
                ),
            )
        unlift_tokens_from_module(
            fw_module,
            "forward",
            num_forward_tokens,
        )

    if bw_module is not None and num_backward_tokens > 0:
        if aot_config.enable_log:
            from torch._dynamo.utils import lazy_format_graph_code

            aot_graphs_effects_log.debug(
                "%s",
                lazy_format_graph_code(
                    "Backward graph before unlifting tokens",
                    bw_module,
                    aot_config.aot_id,
                    include_stride=True,
                    include_device=True,
                    colored=True,
                ),
            )
        unlift_tokens_from_module(bw_module, "backward", num_backward_tokens)

    # This is sad, but we need to update the metadata to get rid of
    # the tokens.
    fw_metadata.tokens = {}
    fw_metadata.num_backward_tokens = 0


def root_module_when_exporting_non_strict(
    flat_fn: Callable[..., Any],
) -> torch.nn.Module | None:
    # When exporting in non-strict mode, we wrap the root module in a specific pattern.
    # See `_aot_export_non_strict` in torch.export._trace.py.
    # We look for that wrapping pattern here.
    if hasattr(flat_fn, "_orig_mod") and hasattr(flat_fn._orig_mod, "_export_root"):
        return flat_fn._orig_mod._export_root
    else:
        return None


def _is_forward_node_with_seq_nr(node: torch.fx.Node) -> bool:
    # For now, assume that if nn_module_stack_metadata is populated, this
    # node is from the forward. Ignore nodes without `seq_nr`.
    # TODO(future): there is likely a less brittle way to do this by walking
    # the descendants of graph inputs corresponding to fwd inputs, didn't
    # seem obvious at first glance on how to partition graph inputs into
    # fwd vs bwd without relying on string names.
    return node.meta.get("partitioner_tag") != "is_backward" and "seq_nr" in node.meta


def _is_backward_node_with_seq_nr(node: torch.fx.Node) -> bool:
    # For now, assume that if nn_module_stack_metadata is not populated,
    # this node is from the backward. Ignore nodes without `seq_nr`.
    # TODO(future): there is likely a less brittle way to do this, same
    # as with the forward.
    return node.meta.get("partitioner_tag") == "is_backward" and "seq_nr" in node.meta


def _collect_fwd_nodes_from_subgraph(
    fx_g: torch.fx.GraphModule, fwd_seq_nr_to_node: dict[str, torch.fx.Node]
) -> None:
    """Collect forward nodes from a single subgraph into the global mapping."""
    for node in fx_g.graph.nodes:
        if not _is_forward_node_with_seq_nr(node):
            continue
        seq_nr = node.meta["seq_nr"]
        if seq_nr in fwd_seq_nr_to_node:
            # If we already saw an op with the current `seq_nr`, that means
            # that the current op did not create an autograd node, and there
            # is no corresponding backward node, so we skip.
            continue
        fwd_seq_nr_to_node[seq_nr] = node


def _copy_metadata_to_bw_nodes_in_subgraph(
    fx_g: torch.fx.GraphModule, fwd_seq_nr_to_node: dict[str, torch.fx.Node]
) -> None:
    """Copy metadata from forward nodes to backward nodes in a single subgraph."""
    for node in fx_g.graph.nodes:
        annotation_log.debug("node: %s", node.name)
        seq_nr = node.meta.get("seq_nr")
        annotation_log.debug("seq_nr: %s", seq_nr)

        if not _is_backward_node_with_seq_nr(node):
            continue

        # We exclude gradient accumulation nodes from copying tags
        if node.meta.get("is_gradient_acc", False):
            annotation_log.debug("is_gradient_acc")
            continue

        # fwd_node should always exist, but handle non-existence just in case
        fwd_node = fwd_seq_nr_to_node.get(node.meta["seq_nr"])
        if fwd_node is not None:
            node.meta["fwd_nn_module_stack"] = fwd_node.meta.get("nn_module_stack")
            node.meta["fwd_source_fn_stack"] = fwd_node.meta.get("source_fn_stack")
            # TODO: better to change to a specific field of custom?
            custom = fwd_node.meta.get("custom")
            if custom is not None:
                # Merge rather than overwrite so bw-only keys survive
                # fw keys win on conflict.
                node.meta["custom"] = {
                    **node.meta.get("custom", {}),
                    **copy.deepcopy(custom),
                }


def copy_fwd_metadata_to_bw_nodes(fx_g: torch.fx.GraphModule) -> None:
    """
    Input: `fx_g` which contains the joint fwd+bwd FX graph created by
    aot_autograd.

    This function walks the graph and copies over metadata from forward nodes
    to backward nodes, using the `seq_nr` field as a one-to-many mapping
    from forward node to backward node. This metadata is useful for performance
    profiling and debugging.

    This function supports matching forward and backward nodes across different
    subgraphs (e.g., in recursive submodules from HOPs), enabling backward nodes
    in any submodule to match forward nodes in any submodule.
    """

    # Build a global mapping of seq_nr to forward nodes across all subgraphs
    fwd_seq_nr_to_node: dict[str, torch.fx.Node] = {}

    # First pass: collect all forward nodes from all subgraphs
    for submod in fx_g.modules():
        if isinstance(submod, torch.fx.GraphModule):
            _collect_fwd_nodes_from_subgraph(submod, fwd_seq_nr_to_node)

    if annotation_log.isEnabledFor(logging.DEBUG):
        for k, v in fwd_seq_nr_to_node.items():
            annotation_log.debug("forward:: key: %s, value: %s", k, v)

    # Second pass: copy metadata to backward nodes in all subgraphs
    # using the global forward mapping
    for submod in fx_g.modules():
        if isinstance(submod, torch.fx.GraphModule):
            _copy_metadata_to_bw_nodes_in_subgraph(submod, fwd_seq_nr_to_node)


def register_buffer_assignment_hook(
    mod: torch.nn.Module, assigned_buffers: dict[str, str]
) -> Any:
    """
    Register a hook that intercepts buffer assignments.
    This is used to detect when a buffer is assigned to, and then we can
    map that buffer to the corresponding proxy node in the graph.
    """

    def _map_assigned_buffer_to_proxy(
        _mod: torch.nn.Module, name: str, buffer: Any
    ) -> Any:
        # We intercept buffer assignments on the root module through this hook.
        if _mod._buffers is mod._buffers:
            # either buffer is a functional tensor, which wraps a fake tensor
            if isinstance(buffer, FunctionalTensor):
                buffer = buffer.from_functional()
            # or buffer is a fake tensor
            if not is_fake_tensor(buffer):
                raise AssertionError(f"expected FakeTensor, got {type(buffer)}")
            # The fake tensor in turn is associated with a proxy node.
            proxy_mode = torch.fx.experimental.proxy_tensor.get_proxy_mode()
            if proxy_mode is None:
                raise AssertionError("proxy_mode must not be None")
            proxy = torch.fx.experimental.proxy_tensor.get_proxy_slot(
                buffer, proxy_mode.tracer
            ).proxy.node
            # We map the assigned buffer to this proxy node.
            assigned_buffers[name] = proxy.name
        return buffer

    return torch.nn.modules.module.register_module_buffer_registration_hook(
        _map_assigned_buffer_to_proxy
    )


def contain_metadata_mutation_ops(module: torch.fx.GraphModule) -> bool:
    """
    Checks if the module contains any metadata mutation ops.
    """
    for node in module.graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "tags")
            and torch.Tag.inplace_view in node.target.tags
        ):
            return True
    return False


_GRAPHSAFE_RNG_DEVICE_TYPES: set[str] = {"cuda"}


def register_graphsafe_rng_device_type(device_type: str) -> None:
    """Register a device type as supporting graphsafe RNG operations.

    The device backend module (``torch.<device_type>``) must provide:
    - ``_get_generator(device: torch.device) -> Generator``: return the default
      generator for the given device. The generator must implement
      ``graphsafe_get_state()``, ``graphsafe_set_state(state)``, and
      ``clone_state()``.
    - ``get_rng_state(device_index: int) -> Tensor``: return the current RNG
      state tensor.

    Args:
        device_type: The device type string (e.g. "cuda", "xpu").
    """
    from torch._prims.rng_prims import register_graphsafe_rng_dispatch

    key_name = torch._C._dispatch_key_for_device(device_type)
    dispatch_key = getattr(torch._C.DispatchKey, key_name)
    _GRAPHSAFE_RNG_DEVICE_TYPES.add(device_type)
    register_graphsafe_rng_dispatch(dispatch_key)


def supports_graphsafe_rng(device: torch.device) -> bool:
    """Check whether a device supports graphsafe RNG operations."""
    return device.type in _GRAPHSAFE_RNG_DEVICE_TYPES


def get_default_generator(device: torch.device) -> Any:
    """Get the default RNG generator for a device.

    Calls ``torch.<device_type>._get_generator(device)``.
    """
    device_mod = getattr(torch, device.type, None)
    if device_mod is None or not hasattr(device_mod, "_get_generator"):
        raise AssertionError(
            f"Device type '{device.type}' does not implement _get_generator. "
            f"Registered graphsafe RNG types: {sorted(_GRAPHSAFE_RNG_DEVICE_TYPES)}."
        )
    return device_mod._get_generator(device)


def get_device_rng_state(device: torch.device) -> torch.Tensor:
    """Get the RNG state tensor for a device."""
    device_mod = getattr(torch, device.type, None)
    if device_mod is not None and hasattr(device_mod, "get_rng_state"):
        idx = device.index if device.index is not None else 0
        return device_mod.get_rng_state(idx)
    return torch.get_rng_state()


def top_saved_tensors_hooks() -> Any:
    return torch._C._autograd._top_saved_tensors_default_hooks(True)


def saved_tensors_hooks_are_inlineable(hooks: Any) -> bool:
    if not hooks:
        return False
    pack, unpack = hooks
    return isinstance(pack, torch.fx.GraphModule) and isinstance(
        unpack, torch.fx.GraphModule
    )


_P = ParamSpec("_P")
_T = TypeVar("_T")
_S = TypeVar("_S")


def without_output_descs(f: Callable[_P, tuple[_T, _S]]) -> Callable[_P, _T]:
    @wraps(f)
    @simple_wraps(f)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return f(*args, **kwargs)[0]

    return inner


_P2 = ParamSpec("_P2")
_R = TypeVar("_R")
_R2 = TypeVar("_R2")


def simple_wraps(
    f: Callable[_P, _R],
) -> Callable[[Callable[_P2, _R2]], Callable[_P2, _R2]]:
    # NB: omit ('__module__', '__name__', '__qualname__') for ease of
    # debugging
    return wraps(f, assigned=("__doc__", "__annotations__", "__type_params__"))


_Ts = TypeVarTuple("_Ts")


def call_and_expect_output_descs(
    fn: Callable[[*_Ts], tuple[Any, Any]], args: tuple[Unpack[_Ts]]
) -> tuple[Any, Any]:
    from .descriptors import AOTOutput

    outs_pair = fn(*args)
    if not (isinstance(outs_pair, tuple) and len(outs_pair) == 2):
        raise AssertionError(
            f"expected tuple of length 2, got {type(outs_pair)} with value {outs_pair}"
        )
    outs, outs_descs = outs_pair
    # The Tensor tests protects against the test when there are no outputs
    out_vals, out_spec = pytree.tree_flatten(outs)
    out_desc_vals, out_desc_spec = pytree.tree_flatten(outs_descs)
    if out_spec != out_desc_spec:
        raise AssertionError(
            f"output spec mismatch: {fn_wrappers(fn)}, outs={outs}, outs_descs={outs_descs}, "
            f"out_spec={out_spec}, out_desc_spec={out_desc_spec}"
        )
    if any(isinstance(x, AOTOutput) for x in out_vals):
        raise AssertionError(
            f"unexpected AOTOutput in out_vals: {fn_wrappers(fn)}, outs={outs}, "
            f"outs_descs={outs_descs}, out_vals={out_vals}"
        )
    if not all(
        isinstance(d, AOTOutput)
        for (x, d) in zip(out_vals, out_desc_vals)
        if isinstance(x, (torch.Tensor, torch.SymInt)) or type(x) is int
    ):
        raise AssertionError(
            f"expected all descriptors to be AOTOutput: {fn_wrappers(fn)}, outs={outs}, "
            f"outs_descs={outs_descs}, out_vals={out_vals}, out_desc_vals={out_desc_vals}"
        )
    return outs_pair


def fn_wrappers(fn: Callable[..., Any]) -> list[Callable[..., Any]]:
    fns = [fn]
    f = fn
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
        fns.append(f)
    return fns


def _is_primal(node: torch.fx.Node) -> bool:
    return (
        node.op == "placeholder"
        and "tangents" not in str(node.target)
        and not _is_bwd_seed_offset(node)
        and not _is_fwd_seed_offset(node)
    )


def _is_tangent(node: torch.fx.Node) -> bool:
    return node.op == "placeholder" and "tangents" in str(node.target)


def _is_bwd_seed_offset(node: torch.fx.Node) -> bool:
    return node.op == "placeholder" and (
        "bwd_seed" in str(node.target) or "bwd_base_offset" in str(node.target)
    )


def _is_fwd_seed_offset(node: torch.fx.Node) -> bool:
    return node.op == "placeholder" and (
        "fwd_seed" in str(node.target) or "fwd_base_offset" in str(node.target)
    )
<|endoftext|>"""
Codegen for AOTDispatchSubclassWrapper.

Generates a Python function that replaces the data-driven
runtime_unwrap_tensor_subclasses / wrap_tensor_subclasses loop with
a straight-line function where all metadata (indices, attr names,
subclass types, symint positions) is baked in at compile time.
"""

import functools
import keyword
from collections.abc import Callable, Iterable
from typing import cast, TYPE_CHECKING

from torch import SymInt

from .codegen import _compile_and_exec_source, PySourceBuilder
from .schemas import ActInputPaths, OpaqueMeta, PlainTensorMeta, SubclassCreationMeta
from .utils import import_async_collective_tensor_type


if TYPE_CHECKING:
    from torch.distributed._functional_collectives import AsyncCollectiveTensor


def _is_symint_placeholder(x: None | int | SymInt) -> bool:
    """Check whether a size/stride entry is symbolic and needs a runtime value.

    Works both before make_runtime_safe() (entries are SymInt) and after
    (symbolic entries replaced with None, nested ints with -1).
    """
    if x is None:
        return True
    if isinstance(x, SymInt) and not x.node.is_nested_int():
        return True
    return False


def _compute_placeholders(outer: Iterable[None | int | SymInt]) -> list[bool]:
    return [_is_symint_placeholder(s) for s in outer]


def _safe_attr_access(var: str, attr: str) -> str:
    if attr.isidentifier() and not keyword.iskeyword(attr):
        return f"{var}.{attr}"
    return f"getattr({var}, {attr!r})"


class _OutputWrapState:
    def __init__(
        self,
        *,
        remaining_min_count: int,
        remaining_max_count: int,
        optional_symint_check: str,
    ) -> None:
        self.remaining_min_count = remaining_min_count
        self.remaining_max_count = remaining_max_count
        self.optional_symint_check = optional_symint_check

    def mark_required(self) -> None:
        self.remaining_min_count -= 1
        self.remaining_max_count -= 1

    def mark_optional(self) -> None:
        self.remaining_max_count -= 1


def _maybe_wait_async_collective_tensor(
    x: object,
    AsyncCollectiveTensor: type["AsyncCollectiveTensor"],
) -> object:
    """Wait on ACT values and leave all other runtime inputs unchanged."""
    if isinstance(x, AsyncCollectiveTensor):
        return cast("AsyncCollectiveTensor", x).trigger_wait()
    return x


def _codegen_unwrap_subclass(
    state: PySourceBuilder,
    meta: SubclassCreationMeta,
    var: str,
    indent: int = 1,
    include_symints: bool = True,
    act_input_paths: set[tuple[str, ...]] | None = None,
    act_wait_fn: str | None = None,
) -> None:
    """Emit code to recursively unwrap a single subclass input."""
    act_input_paths = act_input_paths or set()

    def emit_plain_tensor_symints(var: str, meta: PlainTensorMeta) -> None:
        if not include_symints:
            return

        if any(meta.size_symbol_placeholders):
            size_var = state.fresh_name("_size")
            state.emit(f"{size_var} = {var}.size()", indent=indent)
            for i, is_sym in enumerate(meta.size_symbol_placeholders):
                if is_sym:
                    state.emit(f"unwrapped_args.append({size_var}[{i}])", indent=indent)

        if any(meta.stride_symbol_placeholders):
            stride_var = state.fresh_name("_stride")
            state.emit(f"{stride_var} = {var}.stride()", indent=indent)
            for i, is_sym in enumerate(meta.stride_symbol_placeholders):
                if is_sym:
                    state.emit(
                        f"unwrapped_args.append({stride_var}[{i}])", indent=indent
                    )

    for attr, attr_meta in meta.attrs.items():
        attr_expr = _safe_attr_access(var, attr)
        attr_act_input_paths = {
            path[1:] for path in act_input_paths if path and path[0] == attr
        }
        match attr_meta:
            case PlainTensorMeta():
                if attr_act_input_paths:
                    if attr_act_input_paths != {()}:
                        raise AssertionError(
                            f"ACT path for {attr} continues past a leaf meta"
                        )
                    if act_wait_fn is None:
                        raise AssertionError("missing ACT wait function")
                    resolved_var = state.fresh_name("_resolved")
                    state.emit(
                        f"{resolved_var} = {act_wait_fn}({attr_expr})",
                        indent=indent,
                    )
                    attr_expr = resolved_var
                state.emit(f"unwrapped_args.append({attr_expr})", indent=indent)
                emit_plain_tensor_symints(attr_expr, attr_meta)
            case OpaqueMeta():
                if attr_act_input_paths:
                    if attr_act_input_paths != {()}:
                        raise AssertionError(
                            f"ACT path for {attr} continues past a leaf meta"
                        )
                    if act_wait_fn is None:
                        raise AssertionError("missing ACT wait function")
                    resolved_var = state.fresh_name("_resolved")
                    state.emit(
                        f"{resolved_var} = {act_wait_fn}({attr_expr})",
                        indent=indent,
                    )
                    state.emit(f"unwrapped_args.append({resolved_var})", indent=indent)
                else:
                    state.emit(
                        f"unwrapped_args.append({attr_expr})",
                        indent=indent,
                    )
            case SubclassCreationMeta():
                if () in attr_act_input_paths:
                    raise AssertionError(f"ACT path for {attr} stops at subclass meta")
                inner_var = state.fresh_name("_inner")
                state.emit(f"{inner_var} = {attr_expr}", indent=indent)
                _codegen_unwrap_subclass(
                    state,
                    attr_meta,
                    inner_var,
                    indent=indent,
                    include_symints=include_symints,
                    act_input_paths=attr_act_input_paths,
                    act_wait_fn=act_wait_fn,
                )

    if include_symints:
        size_placeholders = _compute_placeholders(meta.outer_size)
        stride_placeholders = _compute_placeholders(meta.outer_stride)
        if any(size_placeholders) or any(stride_placeholders):
            size_var = state.fresh_name("_size")
            state.emit(f"{size_var} = {var}.size()", indent=indent)
            for i, is_sym in enumerate(size_placeholders):
                if is_sym:
                    state.emit(f"unwrapped_args.append({size_var}[{i}])", indent=indent)

            stride_var = state.fresh_name("_stride")
            state.emit(f"{stride_var} = {var}.stride()", indent=indent)
            for i, is_sym in enumerate(stride_placeholders):
                if is_sym:
                    state.emit(
                        f"unwrapped_args.append({stride_var}[{i}])", indent=indent
                    )


def _concrete_value(val: None | int | SymInt) -> int:
    """Get the concrete int value for a non-symbolic size/stride entry.

    Used for entries that are NOT symbolic placeholders, meaning they are
    concrete ints or nested ints (represented as -1 after make_runtime_safe).
    """
    if isinstance(val, int):
        return val
    # Before make_runtime_safe: nested ints are SymInts; use -1 as dummy.
    # After make_runtime_safe: they're already -1.
    if isinstance(val, SymInt) and val.node.is_nested_int():
        return -1
    raise AssertionError(f"Expected concrete int, got {type(val)}: {val}")


def _is_optional_symint_output(val: object) -> bool:
    return isinstance(val, (int, SymInt))


def _codegen_wrap_subclass(
    state: PySourceBuilder,
    meta: SubclassCreationMeta,
    output_state: _OutputWrapState,
) -> str:
    """Emit code to reconstruct one subclass output. Returns the variable name."""
    inner_dict_var = state.fresh_name("_out_inner")
    entries: list[str] = []
    attr_exprs: dict[str, str] = {}

    for attr, attr_meta in meta.attrs.items():
        match attr_meta:
            case PlainTensorMeta():
                attr_expr = state.fresh_name("_out_attr")
                state.emit(f"{attr_expr} = unwrapped_outs[_out_idx]")
                state.emit("_out_idx += 1")
                output_state.mark_required()
                _emit_optional_plain_tensor_symint_skips(state, output_state, attr_meta)
            case OpaqueMeta():
                attr_expr = state.fresh_name("_out_attr")
                state.emit(f"{attr_expr} = unwrapped_outs[_out_idx]")
                state.emit("_out_idx += 1")
                output_state.mark_required()
            case SubclassCreationMeta():
                attr_expr = _codegen_wrap_subclass(state, attr_meta, output_state)
        attr_exprs[attr] = attr_expr
        entries.append(f"{attr!r}: {attr_expr}")

    state.emit(f"{inner_dict_var} = {{{', '.join(entries)}}}")

    # Reconstruct outer_size and outer_stride
    size_placeholders = _compute_placeholders(meta.outer_size)
    stride_placeholders = _compute_placeholders(meta.outer_stride)

    def _build_tuple(
        outer: Iterable[None | int | SymInt], placeholders: list[bool]
    ) -> str:
        parts: list[str] = []
        for val, is_sym in zip(outer, placeholders):
            if is_sym:
                sym_expr = state.fresh_name("_out_sym")
                state.emit(f"{sym_expr} = unwrapped_outs[_out_idx]")
                state.emit("_out_idx += 1")
                output_state.mark_required()
                parts.append(sym_expr)
            else:
                parts.append(repr(_concrete_value(val)))
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    def _consume_optional_placeholders(placeholders: list[bool]) -> None:
        for is_sym in placeholders:
            if is_sym:
                _emit_optional_symint_skip(state, output_state)

    outer_size_from_attr = meta.outer_size_from_attr
    outer_stride_from_attr = meta.outer_stride_from_attr
    if outer_size_from_attr is not None:
        size_expr = f"{attr_exprs[outer_size_from_attr]}.size()"
        _consume_optional_placeholders(size_placeholders)
    else:
        size_expr = _build_tuple(meta.outer_size, size_placeholders)

    if outer_stride_from_attr is not None:
        stride_expr = f"{attr_exprs[outer_stride_from_attr]}.stride()"
        _consume_optional_placeholders(stride_placeholders)
    else:
        stride_expr = _build_tuple(meta.outer_stride, stride_placeholders)

    type_name = state.add_global(
        state.fresh_name("_subclass_type"),
        meta.original_subclass_type or type(meta.original_subclass),
    )
    meta_name = state.add_global(state.fresh_name("_meta"), meta.meta)

    result_var = state.fresh_name("_out")
    state.emit(
        f"{result_var} = {type_name}.__tensor_unflatten__("
        f"{inner_dict_var}, {meta_name}, {size_expr}, {stride_expr})"
    )
    return result_var


def _count_output_args(
    meta: PlainTensorMeta | SubclassCreationMeta,
    *,
    include_subclass_symints: bool,
) -> int:
    if isinstance(meta, PlainTensorMeta):
        return meta.arg_count if include_subclass_symints else 1

    total = 0
    for attr_meta in meta.attrs.values():
        if isinstance(attr_meta, OpaqueMeta):
            total += 1
        else:
            total += _count_output_args(
                attr_meta, include_subclass_symints=include_subclass_symints
            )

    if include_subclass_symints:
        total += sum(_compute_placeholders(meta.outer_size))
        total += sum(_compute_placeholders(meta.outer_stride))
    else:
        if meta.outer_size_from_attr is None:
            total += sum(_compute_placeholders(meta.outer_size))
        if meta.outer_stride_from_attr is None:
            total += sum(_compute_placeholders(meta.outer_stride))
    return total


def _emit_optional_symint_skip(
    state: PySourceBuilder,
    output_state: _OutputWrapState,
) -> None:
    remaining_var = state.fresh_name("_remaining_outs")
    include_var = state.fresh_name("_include_symints")
    later_min_count = output_state.remaining_min_count
    later_max_count = output_state.remaining_max_count - 1
    max_fits = (
        f"{remaining_var} - 1 >= {later_min_count} and "
        f"{remaining_var} - 1 <= {later_max_count}"
    )
    min_fits = (
        f"{remaining_var} >= {later_min_count} and {remaining_var} <= {later_max_count}"
    )

    state.emit(f"{remaining_var} = _num_wrapped_outs - _out_idx")
    state.emit(
        f"{include_var} = ({max_fits}) and "
        f"{output_state.optional_symint_check}(unwrapped_outs[_out_idx])"
    )
    state.emit(
        f"assert {include_var} or ({min_fits}), "
        "'could not match output layout with optional SymInt outputs'"
    )
    state.emit(f"if {include_var}:")
    state.emit("_out_idx += 1", indent=2)
    output_state.mark_optional()


def _emit_optional_plain_tensor_symint_skips(
    state: PySourceBuilder,
    output_state: _OutputWrapState,
    meta: PlainTensorMeta,
) -> None:
    for is_sym in meta.size_symbol_placeholders:
        if is_sym:
            _emit_optional_symint_skip(state, output_state)
    for is_sym in meta.stride_symbol_placeholders:
        if is_sym:
            _emit_optional_symint_skip(state, output_state)


def _emit_output_wrapping(
    state: PySourceBuilder,
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
) -> tuple[list[str], int]:
    """Emit wrapping code for output metas.

    Returns (result_exprs, num_args_tallied) where result_exprs are Python
    expression strings referencing each wrapped output.
    """
    result_exprs: list[str] = []
    num_args_tallied = 0
    saved_for_bw = num_fw_outs_saved_for_bw or 0
    min_counts = [
        _count_output_args(meta, include_subclass_symints=False) for meta in out_metas
    ]
    max_counts = [
        _count_output_args(meta, include_subclass_symints=True) for meta in out_metas
    ]
    min_wrapped_outputs = sum(min_counts)
    max_wrapped_outputs = sum(max_counts)

    optional_symint_check = (
        state.add_global(
            state.fresh_name("_is_optional_symint_output"),
            _is_optional_symint_output,
        )
        if min_wrapped_outputs != max_wrapped_outputs
        else ""
    )
    state.emit("_out_idx = 0")
    if saved_for_bw:
        state.emit(f"_num_wrapped_outs = len(unwrapped_outs) - {saved_for_bw}")
    else:
        state.emit("_num_wrapped_outs = len(unwrapped_outs)")
    if min_wrapped_outputs != max_wrapped_outputs:
        state.emit(
            "assert "
            f"{min_wrapped_outputs} <= _num_wrapped_outs <= {max_wrapped_outputs}, "
            f"f'expected between {min_wrapped_outputs} and {max_wrapped_outputs} "
            "wrapped outputs, got {_num_wrapped_outs}'"
        )
    else:
        state.emit(
            f"assert _num_wrapped_outs == {max_wrapped_outputs}, "
            f"f'expected {max_wrapped_outputs} wrapped outputs, "
            "got {_num_wrapped_outs}'"
        )

    output_state = _OutputWrapState(
        remaining_min_count=min_wrapped_outputs,
        remaining_max_count=max_wrapped_outputs,
        optional_symint_check=optional_symint_check,
    )
    for meta in out_metas:
        if isinstance(meta, PlainTensorMeta):
            out_expr = state.fresh_name("_out_plain")
            state.emit(f"{out_expr} = unwrapped_outs[_out_idx]")
            state.emit("_out_idx += 1")
            output_state.mark_required()
            _emit_optional_plain_tensor_symint_skips(state, output_state, meta)
            result_exprs.append(out_expr)
            num_args_tallied += meta.arg_count
        else:
            result_var = _codegen_wrap_subclass(state, meta, output_state)
            result_exprs.append(result_var)
            num_args_tallied += meta.arg_count

    if output_state.remaining_min_count != 0 or output_state.remaining_max_count != 0:
        raise AssertionError(
            "output wrapping counts did not balance: "
            f"{output_state.remaining_min_count}, "
            f"{output_state.remaining_max_count}"
        )
    state.emit(
        "assert _out_idx == _num_wrapped_outs, "
        "f'wrapped {_out_idx} outputs, expected {_num_wrapped_outs}'"
    )
    return result_exprs, num_args_tallied


def _emit_input_unwrapping(
    state: PySourceBuilder,
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    frozen_inp_indices: frozenset[int] = frozenset(),
    include_symints: bool = True,
    act_input_paths_by_input: dict[int, set[tuple[str, ...]]] | None = None,
    act_wait_fn: str | None = None,
) -> None:
    """Emit unwrapping code for input metas into unwrapped_args.

    Caller must have already emitted ``unwrapped_args = []``.
    """
    act_input_paths_by_input = act_input_paths_by_input or {}
    for i, meta in enumerate(inp_metas):
        input_act_paths = act_input_paths_by_input.get(i, set())
        if isinstance(meta, PlainTensorMeta):
            if input_act_paths:
                if input_act_paths != {()}:
                    raise AssertionError(
                        f"ACT path for input {i} continues past a plain meta"
                    )
                if act_wait_fn is None:
                    raise AssertionError("missing ACT wait function")
                state.emit(f"unwrapped_args.append({act_wait_fn}(args[{i}]))")
            else:
                state.emit(f"unwrapped_args.append(args[{i}])")
        elif i in frozen_inp_indices:
            # Frozen by inductor freezing: constant already baked into graph.
            state.emit("unwrapped_args.append(None)")
        else:
            if () in input_act_paths:
                raise AssertionError(f"ACT path for input {i} stops at subclass meta")
            inp_var = state.fresh_name("_inp")
            type_name = state.add_global(
                state.fresh_name("_expected_type"),
                meta.original_subclass_type or type(meta.original_subclass),
            )
            state.emit(f"{inp_var} = args[{i}]")
            state.emit(
                f"assert type({inp_var}) is {type_name}, "
                f"f'expected {{{type_name}}}, got {{type({inp_var})}}'",
            )
            _codegen_unwrap_subclass(
                state,
                meta,
                inp_var,
                indent=1,
                include_symints=include_symints,
                act_input_paths=input_act_paths,
                act_wait_fn=act_wait_fn,
            )


def _codegen_subclass_wrapper_source(
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_paths: ActInputPaths | None = None,
) -> tuple[str, dict[str, object]]:
    """Generate source and globals for a subclass wrapper.

    Returns (source, globals_dict).  The globals_dict will NOT contain
    ``compiled_fn`` — the caller is responsible for adding it before exec.
    """
    state = PySourceBuilder()

    state.emit("def inner_fn(args):", indent=0)

    act_input_paths_by_input: dict[int, set[tuple[str, ...]]] = {}
    act_wait_fn = None
    if act_input_paths:
        AsyncCollectiveTensor = import_async_collective_tensor_type()
        act_wait_fn = state.add_global(
            state.fresh_name("_maybe_wait_act"),
            functools.partial(
                _maybe_wait_async_collective_tensor,
                AsyncCollectiveTensor=AsyncCollectiveTensor,
            ),
        )
        for i, attr_path in act_input_paths:
            act_input_paths_by_input.setdefault(i, set()).add(attr_path)

    # --- Input unwrapping ---
    state.emit("unwrapped_args = []")
    _emit_input_unwrapping(
        state,
        inp_metas,
        frozen_inp_indices=frozen_inp_indices,
        act_input_paths_by_input=act_input_paths_by_input,
        act_wait_fn=act_wait_fn,
    )

    # Pass through any trailing args not covered by inp_metas
    # (e.g. rng seed/offset added by FunctionalizedRngRuntimeWrapper).
    num_inp_metas = len(inp_metas)
    state.emit(f"unwrapped_args.extend(args[{num_inp_metas}:])")
    state.emit("args.clear()")

    # --- Call compiled function ---
    state.emit("unwrapped_outs = compiled_fn(unwrapped_args)")

    # --- Output wrapping ---
    result_exprs, _ = _emit_output_wrapping(state, out_metas, num_fw_outs_saved_for_bw)
    result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
    if num_fw_outs_saved_for_bw is not None:
        state.emit(
            f"_activation_start = len(unwrapped_outs) - {num_fw_outs_saved_for_bw}"
        )
        state.emit(f"return {result_tuple} + tuple(unwrapped_outs[_activation_start:])")
    else:
        state.emit(f"return {result_tuple}")

    source = state.getvalue()
    return source, state.globals


def _codegen_subclass_wrap_source(
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
) -> tuple[str, dict[str, object]]:
    """Generate source for wrapping flat outputs into subclasses.

    Used for the backward epilogue. Shares output-wrapping logic with
    _codegen_subclass_wrapper_source via _emit_output_wrapping.
    """
    state = PySourceBuilder()
    state.emit("def wrap_fn(unwrapped_outs):", indent=0)
    result_exprs, _ = _emit_output_wrapping(
        state, out_metas, num_fw_outs_saved_for_bw=None
    )
    result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
    state.emit(f"return {result_tuple}")
    source = state.getvalue()
    return source, state.globals


def codegen_backward_subclass_fns(
    grad_input_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = None,
) -> tuple[Callable[..., object], Callable[..., object] | None]:
    """Generate codegen'd unwrap and wrap functions for the backward pass.

    Returns (unwrap_fn, wrap_fn). unwrap_fn is used by the backward prologue
    to unwrap non-tangent subclass inputs (always an identity in AOT dispatch
    since the compiled forward operates on unwrapped inner tensors). wrap_fn
    is used by the backward epilogue to wrap flat grad inputs back into
    subclasses; it is None when grad_input_metas is None.
    """
    source = "def unwrap_fn(args):\n    return list(args)"
    globals_dict: dict[str, object] = {}
    unwrap_fn = _compile_and_exec_source(
        source, globals_dict, "unwrap_fn", "backward_subclass_unwrap"
    )

    wrap_fn = None
    if grad_input_metas is not None:
        wrap_source, wrap_globals = _codegen_subclass_wrap_source(grad_input_metas)
        wrap_fn = _compile_and_exec_source(
            wrap_source, wrap_globals, "wrap_fn", "backward_subclass_wrapper"
        )

    return unwrap_fn, wrap_fn


def codegen_subclass_wrapper(
    compiled_fn: Callable[..., object],
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_paths: ActInputPaths | None = None,
) -> Callable[..., object]:
    """Generate a specialized wrapper function for subclass unwrap/wrap."""
    source, globals_dict = _codegen_subclass_wrapper_source(
        inp_metas,
        out_metas,
        num_fw_outs_saved_for_bw,
        frozen_inp_indices,
        act_input_paths=act_input_paths,
    )
    globals_dict["compiled_fn"] = compiled_fn
    return _compile_and_exec_source(
        source, globals_dict, "inner_fn", "subclass_wrapper", wrapped_fn=compiled_fn
    )
<|endoftext|>"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""

import collections
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeGuard, TypeVar

import torch
import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._custom_class_base import CustomClassBase
from torch._library.fake_class_registry import maybe_unwrap_fake_script_object
from torch._library.opaque_object import is_opaque_symbolic_type
from torch._subclasses.fake_tensor import get_plain_tensors
from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_eq
from torch.types import IntLikeType
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TraceableWrapperSubclass,
)

from .descriptors import (
    AOTInput,
    AOTOutput,
    DummyAOTInput,
    SubclassGetAttrAOTInput,
    SubclassGetAttrAOTOutput,
    SubclassSizeAOTInput,
    SubclassSizeAOTOutput,
    SubclassStrideAOTInput,
    SubclassStrideAOTOutput,
)
from .schemas import (
    FakifiedFlatArgs,
    FxValue,
    MutationType,
    OpaqueMeta,
    PlainTensorMeta,
    SubclassCreationMeta,
    ViewAndMutationMeta,
)
from .utils import strict_zip


zip = strict_zip

T = TypeVar("T", bound=torch.Tensor)


def requires_subclass_dispatch(
    args: FakifiedFlatArgs, fw_metadata: ViewAndMutationMeta
) -> bool:
    args_flattened = pytree.arg_tree_leaves(*args)
    any_subclass_args = any(
        is_traceable_wrapper_subclass(x)
        for x in args_flattened
        if isinstance(x, Tensor)
    )
    from torch._functorch._aot_autograd.schemas import SubclassCreationMeta

    any_subclass_outputs = any(
        type(x) is SubclassCreationMeta for x in fw_metadata.subclass_fw_graph_out_meta
    )
    # This tells us whether or not we need to perform any unwrapping/wrapping of tensor subclasses at runtime.
    return bool(any_subclass_args or any_subclass_outputs)


from .schemas import MemoryFormatMeta


def maybe_suggest_memory_format(
    t: Tensor, with_memory_format: bool
) -> MemoryFormatMeta | None:
    if not with_memory_format:
        return None

    return MemoryFormatMeta.from_tensor(t)


def _safe_metadata_eq(
    outer: Iterable[IntLikeType],
    inner: Iterable[IntLikeType],
) -> bool:
    return guard_or_false(sym_eq(tuple(outer), tuple(inner)))


def _find_attr_matching_outer_metadata(
    a: Tensor,
    inner_keys: Sequence[str],
    *,
    metadata_fn: Callable[[Tensor], Iterable[IntLikeType]],
) -> str | None:
    tensor_attrs = [key for key in inner_keys if isinstance(getattr(a, key), Tensor)]
    if not tensor_attrs:
        return None

    outer_metadata = tuple(metadata_fn(a))
    # Only use an attr as the runtime source when every tensor attr agrees
    # with the wrapper metadata; otherwise there is no single attr to trust.
    if all(
        _safe_metadata_eq(outer_metadata, metadata_fn(getattr(a, key)))
        for key in tensor_attrs
    ):
        return tensor_attrs[0]
    return None


def _symint_placeholders(lst: Iterable[IntLikeType]) -> tuple[bool, ...]:
    return tuple(isinstance(s, SymInt) and not s.node.is_nested_int() for s in lst)


def get_subclass_typing_container(
    tensor_subclass: torch.Tensor,
) -> dict[type[torch.Tensor], list[type[torch.Tensor]]]:
    """
    Given a subclass, returns a recursive dictionary mapping each
    inner tensors to its subclass types.
    """

    def _get_types_for_subclass(tensor_subclass: torch.Tensor) -> None:
        if not is_traceable_wrapper_subclass(tensor_subclass):
            return
        tracker[type(tensor_subclass)].append(tensor_subclass)
        inner_keys, _ = tensor_subclass.__tensor_flatten__()
        for key in inner_keys:
            match getattr(tensor_subclass, key):
                case torch.Tensor() as inner_value:
                    _get_types_for_subclass(inner_value)
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )

    tracker: dict[Any, list[Any]] = collections.defaultdict(list)
    _get_types_for_subclass(tensor_subclass)
    return tracker


def create_subclass_metadata(
    a: Any,
    start_idx: int,
    count_symints: bool,
    with_memory_format: bool = False,
) -> tuple[Any, int]:
    if not is_traceable_wrapper_subclass(a):
        size_symbol_placeholders = (
            _symint_placeholders(a.size()) if count_symints else ()
        )
        stride_symbol_placeholders = (
            _symint_placeholders(a.stride()) if count_symints else ()
        )
        meta = PlainTensorMeta(
            start_idx + 1,
            memory_format=maybe_suggest_memory_format(a, with_memory_format),
            size_symbol_placeholders=size_symbol_placeholders,
            stride_symbol_placeholders=stride_symbol_placeholders,
        )
        return (
            meta,
            start_idx + meta.arg_count,
        )

    inner_keys, metadata = a.__tensor_flatten__()
    new_start_idx = start_idx
    attrs: dict[str, SubclassCreationMeta | PlainTensorMeta | OpaqueMeta] = {}

    for key in inner_keys:
        inner_value = getattr(a, key)
        match inner_value:
            case CustomClassBase():
                # During tracing, opaques are wrapped in FakeScriptObject;
                # unwrap to check the real type.
                real_type = type(maybe_unwrap_fake_script_object(inner_value))
                if not is_opaque_symbolic_type(real_type):
                    raise RuntimeError(
                        f"{real_type.__name__!r} found in tensor attrs of "
                        f"{type(a).__name__}.__tensor_flatten__(). "
                        "Only tensors and reference-type opaques are allowed "
                        "in tensor attrs."
                    )
                attrs[key] = OpaqueMeta()
                new_start_idx += 1
            case Tensor():
                new_subclass_meta, new_start_idx = create_subclass_metadata(
                    inner_value,
                    new_start_idx,
                    count_symints=count_symints,
                    with_memory_format=with_memory_format,
                )
                attrs[key] = new_subclass_meta
            case _:
                raise AssertionError(
                    f"expected Tensor or CustomClassBase, got {type(inner_value)}"
                )

    # It *must* be because is_traceable_wrapper_subclass() - but mypy is not smart.
    if not isinstance(a, Tensor):
        raise AssertionError(f"expected Tensor, got {type(a)}")

    new_start_idx = (
        new_start_idx
        + count_symints * len(enumerate_filter_symints(a.size()))
        + count_symints * len(enumerate_filter_symints(a.stride()))
    )

    return (
        SubclassCreationMeta(
            flat_tensor_start_idx=start_idx,
            arg_count=new_start_idx - start_idx,
            included_subclass_symints=count_symints,
            attrs=attrs,
            meta=metadata,
            outer_size=a.size(),  # type: ignore[attr-defined, arg-type]
            outer_stride=a.stride(),  # type: ignore[arg-type]
            original_subclass=a,
            memory_format=maybe_suggest_memory_format(a, with_memory_format),
            outer_size_from_attr=_find_attr_matching_outer_metadata(
                a, inner_keys, metadata_fn=lambda t: t.size()
            ),
            outer_stride_from_attr=_find_attr_matching_outer_metadata(
                a, inner_keys, metadata_fn=lambda t: t.stride()
            ),
        ),
        new_start_idx,
    )


# Given a flat list of arguments, some of which may be tensor subclasses,
# computes metadata about "how to reconstruct the current list of subclasses,
# if we were given their flattened dense tensors instead"
def create_subclass_meta(
    curr_args: list[Any] | tuple[Any, ...],
    *,
    count_symints: bool = True,
    with_memory_format: bool = False,
) -> list[PlainTensorMeta | SubclassCreationMeta]:
    idx = 0
    infos: list[PlainTensorMeta | SubclassCreationMeta] = []
    for a in curr_args:
        if is_traceable_wrapper_subclass(a):
            if not isinstance(a, Tensor):
                raise AssertionError(
                    f"expected Tensor for traceable wrapper subclass, got {type(a)}"
                )
            start_idx = idx
            subclass_meta, _ = create_subclass_metadata(
