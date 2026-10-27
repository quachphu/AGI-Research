        post_processing_timeout_s: float | None = None,
    ) -> None:
        if activities is not None:
            self.activities, self.activity_filters = _parse_activities(activities)
        else:
            self.activities = supported_activities()
            self.activity_filters: dict[ProfilerActivity, set[str]] = {}
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        if (
            experimental_config is not None
            and experimental_config.trace_only
            and with_stack
        ):
            import copy

            warn(
                "trace_only=True is incompatible with with_stack=True "
                "(stack traces require event post-processing). "
                "Disabling trace_only."
            )
            experimental_config_copy: _ExperimentalConfig = copy.copy(
                experimental_config
            )
            experimental_config_copy.trace_only = False
            experimental_config = experimental_config_copy
        self.experimental_config = experimental_config
        self.execution_trace_observer = execution_trace_observer
        self.acc_events = acc_events
        self.custom_trace_id_callback = custom_trace_id_callback
        self.post_processing_timeout_s = post_processing_timeout_s
        self.profiler: prof.profile | None = None
        self.has_cudagraphs = False
        self.mem_tl: MemoryProfileTimeline | None = None
        self.use_device = None
        if ProfilerActivity.CUDA in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = "cuda"
        elif ProfilerActivity.XPU in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = "xpu"
        elif ProfilerActivity.MTIA in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = "mtia"
        elif ProfilerActivity.HPU in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = "hpu"
        elif ProfilerActivity.PrivateUse1 in self.activities:
            # pyrefly: ignore [bad-assignment]
            self.use_device = _get_privateuse1_backend_name()

        # user-defined metadata to be amended to the trace
        self.preset_metadata: dict[str, str] = {}
        self._trace_metadata: dict[str, str] = {}
        self._custom_profiler_config = _parse_custom_profiler_config(
            self.experimental_config
        )
        self._use_cupti_monitor = self._custom_profiler_config.get(
            "backend"
        ) == "cupti_monitor" or bool(self._custom_profiler_config.get("cupti_monitor"))
        # The ProfilerObserver driving the shared CUPTI monitor this session; window opened
        # at start, closed at stop (_monitor_window_id), exported by export_chrome_trace.
        self._cupti_profiler_observer: Any = None
        self._monitor_window_id: int | None = None
        # cupti_monitor exports synchronously by default (like the stock profiler).
        # {"cupti_monitor_async_export": true} hands the merge+write off-thread, joined by
        # wait_for_exports; cupti_monitor-only, rejected elsewhere.
        self._cupti_async_export = False
        if self._use_cupti_monitor:
            if ProfilerActivity.CPU not in self.activities:
                raise ValueError(
                    "cupti_monitor profiler backend currently requires CPU activity"
                )
            self._cupti_async_export = bool(
                self._custom_profiler_config.get("cupti_monitor_async_export", False)
            )
            # Arm graph-dependency recording now, at profiler construction -- before the
            # training loop captures its CUDA graphs. The recording hook must observe each
            # graph's one-time instantiate(); the per-window ProfilerObserver registers it
            # too late (at prepare_trace, after warm-up capture) to catch replay-only graphs.
            if self._custom_profiler_config.get("enable_graph_dependencies"):
                from torch.profiler._cupti._graph_deps import _GraphDependencyRecorder

                _GraphDependencyRecorder().arm()
            # Same early-arm rationale for the CUDA_EVENT -> graph event-record node bridge:
            # the recorder reads each graph's event nodes at its one-time instantiate().
            if self._custom_profiler_config.get("enable_event_node_ids"):
                from torch.profiler._cupti._event_nodes import _EventNodeRecorder

                _EventNodeRecorder().arm()
        elif "cupti_monitor_async_export" in self._custom_profiler_config:
            raise ValueError(
                "cupti_monitor_async_export is only supported with the cupti_monitor "
                "backend"
            )

    def start(self) -> None:
        self.prepare_trace()
        self.start_trace()

    def stop(self) -> None:
        self.stop_trace()

    def prepare_trace(self) -> None:
        if hasattr(torch, "_inductor"):
            import torch._inductor.config as inductor_config

            self.has_cudagraphs = inductor_config.triton.cudagraphs
        if (self.profiler is not None) and (not self.acc_events):
            _warn_once(
                "Warning: Profiler clears events at the end of each cycle. "
                "Only events from the current cycle will be reported. "
                "To keep events across cycles, set acc_events=True."
            )
        if (self.profiler is None) or (not self.acc_events):
            use_device = None if self._use_cupti_monitor else self.use_device
            self.profiler = prof.profile(
                use_cpu=(ProfilerActivity.CPU in self.activities),
                use_device=use_device,
                record_shapes=self.record_shapes,
                with_flops=self.with_flops,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_modules=self.with_modules,
                use_kineto=True,
                experimental_config=self.experimental_config,
                acc_events=self.acc_events,
                custom_trace_id_callback=self.custom_trace_id_callback,
                post_processing_timeout_s=self.post_processing_timeout_s,
                activity_filters=self.activity_filters
                if self.activity_filters
                else None,
            )
        if self._use_cupti_monitor:
            from torch.profiler._cupti.observers.profiler import ProfilerObserver

            self._monitor_trace_window = None
            # Constructing the observer registers it with the shared monitor and starts
            # collection. cuda_sync events are opt-in via the config, matching kineto's flag.
            self._cupti_profiler_observer = ProfilerObserver(
                enable_cuda_sync=bool(
                    self._custom_profiler_config.get("enable_cuda_sync_events")
                ),
                # GPU environment counters (power/clock/thermal/cooling) are periodically
                # sampled; opt-in since the sampling adds overhead.
                enable_environment_counters=bool(
                    self._custom_profiler_config.get("enable_environment_counters")
                ),
                # PM sampling (true SM-active % + DRAM-throughput % counters) is a CUPTI-monitor
                # feature, opt-in like the env counters. The metrics are per-profile
                # (custom_profiler_config["pm_metrics"], a list of CUPTI metric names).
                enable_pm_sampling=bool(
                    self._custom_profiler_config.get("enable_pm_sampling")
                ),
                pm_metrics=self._custom_profiler_config.get("pm_metrics"),
                # Node->node CUDA-graph dependency arrows are opt-in (extra work at graph
                # instantiate + arrow rendering); off unless the config requests them.
                enable_graph_dependencies=bool(
                    self._custom_profiler_config.get("enable_graph_dependencies")
                ),
                # Join CUDA_EVENT records (graph event-record nodes, e.g. NCCL under
                # NCCL_GRAPH_MIXING_SUPPORT) back to their graph_node_id. Opt-in: pulls in the
                # CUDA_EVENT record kind.
                enable_event_node_ids=bool(
                    self._custom_profiler_config.get("enable_event_node_ids")
                ),
                # gzip level for the native .pftrace encoder (0-9; 1 = fast, the default).
                pftrace_compression_level=int(
                    self._custom_profiler_config.get("pftrace_compression_level", 1)
                ),
                # Synchronous export finalizes on the calling thread, so skip the poll thread.
                defer_export=self._cupti_async_export,
            )
            # Publish the observer so record_function routes annotations to it. The reference
            # lives in torch.autograd (not the cupti package), so record_function never
            # imports the cupti chain on a non-cupti run.
            prof._set_active_cupti_profiler_observer(self._cupti_profiler_observer)
        self.profiler._prepare_trace()

    def start_trace(self) -> None:
        if self.execution_trace_observer:
            self.execution_trace_observer.start()
        if self.profiler is None:
            raise AssertionError("Profiler must be initialized before starting trace")
        self.profiler._start_trace()
        if self._use_cupti_monitor and self._cupti_profiler_observer is not None:
            # Open the trace window here (stamps the start boundary, native clock, no
            # device sync); records before this are excluded from the window.
            self._cupti_profiler_observer.open_window()

        if self.profile_memory:
            self.add_metadata_json("profile_memory", "1")
        if self.with_stack:
            self.add_metadata_json("with_stack", "1")
        if self.record_shapes:
            self.add_metadata_json("record_shapes", "1")
        if self.with_modules:
            self.add_metadata_json("with_modules", "1")
        if self.with_flops:
            self.add_metadata_json("with_flops", "1")

        if kineto_available():
            dist_info = self._get_distributed_info()
            if dist_info:
                self.add_metadata_json(
                    "distributedInfo", json.dumps(dist_info, cls=_NumpyEncoder)
                )

            cuda_version = None
            if hasattr(torch, "version"):
                from torch.torch_version import TorchVersion

                cuda_version = TorchVersion(getattr(torch.version, "cuda", "0.0"))

            if self.has_cudagraphs and (
                (cuda_version and cuda_version < "12.6")
                or not profiler_allow_cudagraph_cupti_lazy_reinit_cuda12()
            ):
                os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"
                self.add_metadata_json("DISABLE_CUPTI_LAZY_REINIT", "1")
                # FIXME: CUDA Graph does not work well with CUPTI teardown.
                #   1) crashes on 1st lazy CUPTI re-init after teardown (CUDA 11)
                #   2) crashes on 2nd non-lazy CUPTI re-init after teardown (CUDA 12)
                # Workaround: turn off CUPTI teardown when using CUDA Graphs.
                os.environ["TEARDOWN_CUPTI"] = "0"

            # Insert the preset user metadata to the trace
            for k, v in self.preset_metadata.items():
                self.add_metadata_json(k, v)

    def stop_trace(self) -> None:
        if self.execution_trace_observer:
            self.execution_trace_observer.stop()
        if self.profiler is None:
            raise AssertionError("Profiler must be initialized before stopping trace")
        if self._use_cupti_monitor:
            # Unpublish the observer (record_function stops routing here) and close the trace
            # window (end boundary, native clock, no device sync), queuing it for deferred
            # export; the observer is kept alive past stop for the async write.
            prof._set_active_cupti_profiler_observer(None)
            if self._cupti_profiler_observer is not None:
                self._monitor_window_id = self._cupti_profiler_observer.close_window()
        self.profiler.__exit__(None, None, None)

    def export_chrome_trace(self, path: str, use_python_export: bool = False):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        if self.profiler is None:
            raise AssertionError(
                "Profiler must be initialized before exporting chrome trace"
            )
        if self._use_cupti_monitor:
            obs = self._cupti_profiler_observer
            if obs is None or not obs.available or self._monitor_window_id is None:
                # Nothing to export this cycle: the per-cycle ProfilerObserver didn't register
                # with the CUPTI monitor (available is False -- the intermittent case), or its
                # window wasn't opened/closed (window id None). Skip rather than crash -- a
                # profiler-trace hiccup must not take down a training run -- and clean up below.
                _warn_once(
                    "CUPTI monitor observer unavailable; skipping chrome trace export"
                )
                # join() tears down the poll thread + monitor registration, which exist only
                # when the observer registered (available). An unavailable observer never
                # started either, so just drop the reference and let it be GC'd.
                if obs is not None and obs.available:
                    obs.join()
                self._cupti_profiler_observer = None
                self._monitor_window_id = None
                return
            # Capture the profiler's CPU-side trace (cheap, no device sync) and hand it + the
            # output path to the observer. Async: the poller merges + writes `path` once the
            # GPU records arrive; wait_for_exports() blocks for it.
            fp = tempfile.NamedTemporaryFile(  # noqa: SIM115
                "w+t", suffix=".json", delete=False
            )
            fp.close()
            self.profiler.export_chrome_trace(fp.name, self._trace_metadata)
            self._cupti_profiler_observer.set_export(
                self._monitor_window_id, fp.name, path
            )
            if not self._cupti_async_export:
                # Synchronous: finalize + write now, so `path` exists on return.
                self._cupti_profiler_observer.join()
                self._cupti_profiler_observer = None
                self._monitor_window_id = None
            return
        if use_python_export:
            self.profiler.export_chrome_trace(
                path, self._trace_metadata, use_python_export=True
            )
        elif path.endswith(".gz"):
            with tempfile.NamedTemporaryFile("w+b", suffix=".json") as fp:
                self.profiler.export_chrome_trace(fp.name, self._trace_metadata)
                with open(fp.name, "rb") as fin, gzip.open(path, "wb") as fout:
                    fout.writelines(fin)
        else:
            self.profiler.export_chrome_trace(path, self._trace_metadata)

    def wait_for_exports(self) -> None:
        """Block until every deferred cupti_monitor export is written, then unregister.
        No-op unless the cupti_monitor backend is active. Call on the training thread when
        you need the file(s) on disk; the finalize force-flushes CUPTI (safe here)."""
        if self._use_cupti_monitor and self._cupti_profiler_observer is not None:
            self._cupti_profiler_observer.join()
            self._cupti_profiler_observer = None
            self._monitor_window_id = None

    def take_pending_cupti_export(self) -> Any:
        """Detach this cycle's cupti_monitor ProfilerObserver (with its unwritten window) so
        the deferred export can be finalized OFF the training thread (call
        ``obs.join(force=False)`` on a worker). None for non-cupti backends."""
        if not self._use_cupti_monitor:
            return None
        obs = self._cupti_profiler_observer
        self._cupti_profiler_observer = None
        self._monitor_window_id = None
        return obs

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        """Save stack traces to a file

        Args:
            path (str): save stacks file to this location;
            metric (str): metric to use: "self_cpu_time_total" or "self_cuda_time_total"
        """
        if self.profiler is None:
            raise AssertionError("Profiler must be initialized before exporting stacks")
        return self.profiler.export_stacks(path, metric)

    def toggle_collection_dynamic(
        self, enable: bool, activities: Iterable[ProfilerActivity]
    ) -> None:
        """Toggle collection of activities on/off at any point of collection. Currently supports toggling Torch Ops
        (CPU) and CUDA activity supported in Kineto

        Args:
            activities (iterable): list of activity groups to use in profiling, supported values:
                ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``
        Examples:

        .. code-block:: python

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                code_to_profile_0()
                // turn off collection of all CUDA activity
                p.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                code_to_profile_1()
                // turn on collection of all CUDA activity
                p.toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.CUDA])
                code_to_profile_2()
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
        """
        if self.profiler is None:
            return
        self.profiler.toggle_collection_dynamic(enable, activities)

    def key_averages(
        self,
        group_by_input_shape: bool = False,
        group_by_stack_n: int = 0,
        group_by_overload_name: bool = False,
        include_python_functions: bool = False,
    ):
        """Averages events, grouping them by operator name and (optionally) input shapes, stack
        and overload name.

        Returns an :class:`~torch.autograd.profiler_util.EventList` of the aggregated events.

        .. note::
            To use shape/stack functionality make sure to set record_shapes/with_stack
            when creating profiler context manager.
        """
        if self.profiler is None:
            raise AssertionError(
                "Profiler must be initialized before getting key averages"
            )
        return self.profiler.key_averages(
            group_by_input_shape,
            group_by_stack_n,
            group_by_overload_name,
            include_python_functions,
        )

    def events(self):
        """
        Return the list of unaggregated :class:`~torch.autograd.profiler_util.FunctionEvent`
        objects, for use in the trace callback or after profiling has finished.
        """
        if self.profiler is None:
            raise AssertionError("Profiler must be initialized before accessing events")
        if self.experimental_config is not None and self.experimental_config.trace_only:
            raise RuntimeError(
                "events() is not available when trace_only=True in "
                "ExperimentalConfig. Use export_chrome_trace() instead."
            )
        return self.profiler.function_events

    def add_metadata(self, key: str, value: str) -> None:
        """
        Adds a user defined metadata with a string key and a string value
        into the trace file
        """
        wrapped_value = '"' + value.replace('"', '\\"') + '"'
        self._trace_metadata[key] = wrapped_value
        torch.autograd._add_metadata_json(key, wrapped_value)

    def add_metadata_json(self, key: str, value: str) -> None:
        """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
        self._trace_metadata[key] = value
        torch.autograd._add_metadata_json(key, value)

    def preset_metadata_json(self, key: str, value: str) -> None:
        """
        Preset a user defined metadata when the profiler is not started
        and added into the trace file later.
        Metadata is in the format of a string key and a valid json value
        """
        self.preset_metadata[key] = value

    def _get_distributed_info(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None

        backend = dist.get_backend()
        dist_info = {
            "backend": backend,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "pg_count": dist.get_pg_count(),
            "pg_config": dist.distributed_c10d._get_all_pg_configs(),
        }
        if backend == "nccl":
            nccl_version = torch.cuda.nccl.version()
            # pyrefly: ignore [bad-typed-dict-key, unsupported-operation]
            dist_info["nccl_version"] = ".".join(str(v) for v in nccl_version)
        return dist_info

    def _memory_profile(self) -> MemoryProfile:
        required = ("record_shapes", "profile_memory", "with_stack")
        missing = [f"{i}=True" for i in required if not getattr(self, i)]
        if missing:
            raise ValueError(f"{', '.join(missing)} required for memory profiling.")

        if self.profiler is None or self.profiler.kineto_results is None:
            raise AssertionError(
                "Profiler and kineto_results must be initialized for memory profiling"
            )
        return MemoryProfile(self.profiler.kineto_results)

    @deprecated(
        "`export_memory_timeline` is deprecated and will be removed in a future version. "
        "Please use `torch.cuda.memory._record_memory_history` and `torch.cuda.memory._export_memory_snapshot` instead.",
        category=FutureWarning,
    )
    def export_memory_timeline(self, path: str, device: str | None = None) -> None:
        """Export memory event information from the profiler collected
        tree for a given device, and export a timeline plot. There are 3
        exportable files using ``export_memory_timeline``, each controlled by the
        ``path``'s suffix.

        - For an HTML compatible plot, use the suffix ``.html``, and a memory timeline
          plot will be embedded as a PNG file in the HTML file.

        - For plot points consisting of ``[times, [sizes by category]]``, where
          ``times`` are timestamps and ``sizes`` are memory usage for each category.
          The memory timeline plot will be saved a JSON (``.json``) or gzipped JSON
          (``.json.gz``) depending on the suffix.

        - For raw memory points, use the suffix ``.raw.json.gz``. Each raw memory
          event will consist of ``(timestamp, action, numbytes, category)``, where
          ``action`` is one of ``[PREEXISTING, CREATE, INCREMENT_VERSION, DESTROY]``,
          and ``category`` is one of the enums from
          ``torch.profiler._memory_profiler.Category``.

        Output: Memory timeline written as gzipped JSON, JSON, or HTML.

        .. deprecated::
            ``export_memory_timeline`` is deprecated and will be removed in a future version.
            Please use ``torch.cuda.memory._record_memory_history`` and
            ``torch.cuda.memory._export_memory_snapshot`` instead.
        """
        # Default to device 0, if unset. Fallback on cpu.
        if device is None:
            if self.use_device and self.use_device != "cuda":
                device = self.use_device + ":0"
            else:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Construct the memory timeline plot data
        self.mem_tl = MemoryProfileTimeline(self._memory_profile())

        # Depending on the file suffix, save the data as json.gz or json.
        # For html, we can embed the image into an HTML file.
        if path.endswith(".html"):
            self.mem_tl.export_memory_timeline_html(path, device)
        elif path.endswith(".gz"):
            with tempfile.NamedTemporaryFile("w+t", suffix=".json") as fp:
                if path.endswith("raw.json.gz"):
                    self.mem_tl.export_memory_timeline_raw(fp.name, device)
                else:
                    self.mem_tl.export_memory_timeline(fp.name, device)
                with open(fp.name) as fin, gzip.open(path, "wt") as fout:
                    fout.writelines(fin)
        else:
            self.mem_tl.export_memory_timeline(path, device)


class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals.

    NONE, WARMUP, RECORD, and RECORD_AND_SAVE are user-facing values that may
    be returned from a user-provided schedule. DEVICE_STOPPED is set
    internally by the profiler when device collection stops early due to
    errors; it must not be returned from a user-provided schedule.
    """

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3
    DEVICE_STOPPED = 4


def _unreachable_transition(prev: str, current: str) -> None:
    raise RuntimeError(
        f"Profiler internal error: {prev} -> {current} should be unreachable"
    )


def schedule(
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int = 0,
    skip_first: int = 0,
    skip_first_wait: int = 0,
) -> Callable:
    """
    Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip
    the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`` steps,
    then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
    The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.

    The ``skip_first_wait`` parameter controls whether the first ``wait`` stage should be skipped.
    This can be useful if a user wants to wait longer than ``skip_first`` between cycles, but not
    for the first profile. For example, if ``skip_first`` is 10 and ``wait`` is 20, the first cycle will
    wait 10 + 20 = 30 steps before warmup if ``skip_first_wait`` is zero, but will wait only 10
    steps if ``skip_first_wait`` is non-zero. All subsequent cycles will then wait 20 steps between the
    last active and warmup.
    """

    def schedule_fn(step: int) -> ProfilerAction:
        if step < 0:
            raise AssertionError(f"Step must be non-negative. Got {step}.")
        if step < skip_first:
            return ProfilerAction.NONE
        else:
            step -= skip_first
        # If wait >> skip_first and we want to grab profiling early, shift left by wait if skip_first_wait is True
        if skip_first_wait != 0:
            step += wait
        num_steps = wait + warmup + active
        if repeat > 0 and step / num_steps >= repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps
        if mod_step < wait:
            return ProfilerAction.NONE
        elif mod_step < wait + warmup:
            return ProfilerAction.WARMUP
        else:
            return (
                ProfilerAction.RECORD
                if mod_step < num_steps - 1
                else ProfilerAction.RECORD_AND_SAVE
            )

    if wait < 0 or warmup < 0 or active <= 0 or repeat < 0 or skip_first < 0:
        raise AssertionError(
            f"Invalid profiler schedule arguments. Got wait={wait} (need >= 0), warmup={warmup} (need >= 0), "
            f"active={active} (need > 0), repeat={repeat} (need >= 0), skip_first={skip_first} (need >= 0)."
        )
    if warmup == 0:
        warn(
            "Profiler won't be using warmup, this can skew profiler results",
            stacklevel=2,
        )
    return schedule_fn


def _default_schedule_fn(_: int) -> ProfilerAction:
    """
    Default profiler behavior - immediately starts recording the events,
    keeps doing it on every profiler step.
    """
    return ProfilerAction.RECORD


def tensorboard_trace_handler(
    dir_name: str,
    worker_name: str | None = None,
    use_gzip: bool = False,
    use_python_export: bool = False,
):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import socket
    import time

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        prof.export_chrome_trace(
            os.path.join(dir_name, file_name),
            use_python_export=use_python_export,
        )

    return handler_fn


class profile(_KinetoProfile):
    """Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.

            Each item can be a ``ProfilerActivity`` enum (collects all default
            activity types for that group) or a ``dict`` mapping a ``ProfilerActivity``
            to a list of individual activity type names to collect, e.g.
            ``{ProfilerActivity.CUDA: ["GPU_MEMCPY", "CUDA_RUNTIME"]}``.
            An empty list (e.g. ``{ProfilerActivity.CUDA: []}``) means collect
            nothing for that group.
            The same activity group must not appear more than once.
            See :class:`~torch.profiler.ProfilerActivity` for the valid
            activity type names and device-specific behavior.
        schedule (Callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (Callable): callable invoked at the end of each profiling cycle
            (when ``schedule`` returns ``ProfilerAction.RECORD_AND_SAVE``). Receives the
            :class:`profile` instance as its only argument, typically used to export the
            trace (e.g. via :meth:`export_chrome_trace`) or print a summary.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation.
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B

            .. deprecated::
                ``with_modules`` is deprecated and will be removed in a future version.
                It only collects data for TorchScript models, which are themselves
                deprecated, and does nothing in eager mode. Use ``with_stack=True``,
                which records ``nn.Module`` events for eager models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used for Kineto library features. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler. See the examples section below for a code sample.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles
        post_processing_timeout_s (float): Optional timeout in seconds for post-processing profiler
            results. If specified, event parsing will stop after this duration and return partial
            results. Useful for handling large traces that may take too long to process.
        custom_trace_id_callback (Callable[[], str], optional): User-supplied trace ID generator,
            invoked once per profiling cycle. Defaults to a random UUID; retrieve via
            :meth:`get_trace_id`.

    .. note::
        Use :func:`~torch.profiler.schedule` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.

    .. note::
        Use :func:`~torch.profiler.tensorboard_trace_handler` to generate result files for TensorBoard:

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

        After profiling, result files can be found in the specified directory. Use the command:

        ``tensorboard --logdir dir_name``

        to see the results in TensorBoard.
        For more information, see
        `PyTorch Profiler TensorBoard Plugin <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.


    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    Using the profiler's ``schedule``, ``on_trace_ready`` and ``step`` functions:

    .. code-block:: python

        # Non-default profiler schedule allows user to turn profiler on and off
        # on different iterations of the training loop;
        # trace_handler is called every time a new trace becomes available
        def trace_handler(prof):
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
            )
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")


        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the fourth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=trace_handler,
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
        ) as p:
            for iter in range(N):
                code_iteration_to_profile(iter)
                # send a signal to the profiler that the next iteration has started
                p.step()

    The following sample shows how to setup up an Execution Trace Observer (`execution_trace_observer`)

    .. code-block:: python

        with torch.profiler.profile(
            ...
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback("./execution_trace.json")
            ),
        ) as p:
            for iter in range(N):
                code_iteration_to_profile(iter)
                p.step()

    You can also refer to test_execution_trace_with_kineto() in tests/profiler/test_profiler.py.
    Note: One can also pass any object satisfying the _ITraceObserver interface.
    """

    def __init__(
        self,
        *,
        activities: Iterable[ProfilerActivity | dict[ProfilerActivity, list[str]]]
        | None = None,
        schedule: Callable[[int], ProfilerAction] | None = None,
        on_trace_ready: Callable[..., Any] | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: _ExperimentalConfig | None = None,
        execution_trace_observer: _ITraceObserver | None = None,
        acc_events: bool = False,
        custom_trace_id_callback: Callable[[], str] | None = None,
        post_processing_timeout_s: float | None = None,
    ) -> None:
        if activities is not None:
            activities_set, _ = _parse_activities(activities)
        else:
            activities_set = supported_activities()
        if len(activities_set) == 0:
            raise AssertionError("No valid profiler activities found")

        super().__init__(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
            execution_trace_observer=execution_trace_observer
            or ExecutionTraceObserver.build_execution_trace_obs_from_env(),
            acc_events=acc_events,
            custom_trace_id_callback=custom_trace_id_callback,
            post_processing_timeout_s=post_processing_timeout_s,
        )

        if schedule:
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            self.schedule = _default_schedule_fn
            self.record_steps = False
        self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.step_rec_fn: prof.record_function | None = None

        schedule_action = self.schedule(self.step_num)
        if schedule_action == ProfilerAction.DEVICE_STOPPED:
            raise ValueError(
                "ProfilerAction.DEVICE_STOPPED is set internally by the "
                "profiler and must not be returned by a user-provided schedule"
            )
        self.current_action = schedule_action
        # Raw schedule output of the previous step, separate from
        # current_action which step() may override to DEVICE_STOPPED. Used to
        # detect cycle boundaries (RECORD_AND_SAVE -> next) when warmup=0,
        # so DEVICE_STOPPED can exit and resume profiling on the new cycle.
        self._prev_schedule_action = schedule_action

        self.action_map: dict[
            tuple[ProfilerAction, ProfilerAction | None], list[Any]
        ] = {
            # key is (prev_action, current_action), value is action list corresponding to the state pair.
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.prepare_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: WARMUP followed by NONE"),
                self.start_trace,
                self.stop_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self.start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self.start_trace],
            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: RECORD followed by NONE"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [
                partial(warn, "Incorrect schedule: RECORD followed by WARMUP"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            # DEVICE_STOPPED: entered when device collection stops early (e.g.
            # CUPTI buffer overflow). Absorbs the remainder of the current
            # profiling cycle, then exits at the next cycle boundary.
            #
            # All three entry transitions fire _trace_ready so the user's
            # callback is invoked even when entry happens from WARMUP. This
            # matters for active=1 schedules: the only active step (R&S) gets
            # converted to DEVICE_STOPPED via WARMUP -> R&S override, and
            # without _trace_ready firing here the user would never see a
            # callback for that cycle. The trace will be empty (no recording
            # ran), but the callback firing is the signal that the cycle
            # completed.
            (ProfilerAction.WARMUP, ProfilerAction.DEVICE_STOPPED): [
                self.start_trace,
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.RECORD, ProfilerAction.DEVICE_STOPPED): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.DEVICE_STOPPED): [],
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.WARMUP): [
                self.prepare_trace,
            ],
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.NONE): [],
            # Cycle boundary recovery: when the schedule has no WARMUP phase
            # (warmup=0), DEVICE_STOPPED exits directly into the next cycle's
            # active phase. Mirrors (NONE, RECORD) / (NONE, RECORD_AND_SAVE).
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.RECORD): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.DEVICE_STOPPED, ProfilerAction.RECORD_AND_SAVE): [
                self.prepare_trace,
                self.start_trace,
            ],
            # Unreachable transitions:
            # - prev=NONE: step()'s entry guard excludes it, and start() and
            #   __init__ both reject DEVICE_STOPPED from a user schedule.
            # - prev=RECORD_AND_SAVE: entry guard excludes it because the
            #   natural (R&S, *) transitions already do the right thing to
            #   recover.
            (ProfilerAction.NONE, ProfilerAction.DEVICE_STOPPED): [
                partial(_unreachable_transition, "NONE", "DEVICE_STOPPED"),
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.DEVICE_STOPPED): [
                partial(_unreachable_transition, "RECORD_AND_SAVE", "DEVICE_STOPPED"),
            ],
            # used for exit action
            (ProfilerAction.WARMUP, None): [self.start_trace, self.stop_trace],
            (ProfilerAction.RECORD, None): [self.stop_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.DEVICE_STOPPED, None): [],
        }
        # Start tracking increments to profiler step, this will be used
        # by Kineto
        prof.KinetoStepTracker.init_step_count(PROFILER_STEP_NAME)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        prof.KinetoStepTracker.erase_step_count(PROFILER_STEP_NAME)
        if self.execution_trace_observer:
            self.execution_trace_observer.cleanup()

    def start(self) -> None:
        self._transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def stop(self) -> None:
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
            self.step_rec_fn = None
        self._transit_action(self.current_action, None)
        # Reset current_action to the schedule's view in case step() had
        # overridden it to DEVICE_STOPPED. Without this, a subsequent start()
        # would transit (NONE, DEVICE_STOPPED) — unreachable by contract —
        # and leave the profiler running with no Kineto session.
        if self._prev_schedule_action == ProfilerAction.DEVICE_STOPPED:
            raise AssertionError(
                "_prev_schedule_action must never be DEVICE_STOPPED here"
            )
        self.current_action = self._prev_schedule_action

    def step(self) -> None:
        """
        Signals the profiler that the next profiling step has started.
        """
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
            # Drop our reference so a subsequent stop() / step() — e.g. after
            # this step() raises — doesn't double-exit the same instance.
            self.step_rec_fn = None
        prev_action = self.current_action
        prev_schedule_action = self._prev_schedule_action
        next_step = self.step_num + 1
        schedule_action = self.schedule(next_step)

        # Note that we check schedule validity BEFORE changing the profiler's
        # internal state. This prevents the profiler from being in an invalid
        # state.
        if schedule_action == ProfilerAction.DEVICE_STOPPED:
            raise ValueError(
                "ProfilerAction.DEVICE_STOPPED is set internally by the "
                "profiler and must not be returned by a user-provided schedule"
            )

        self.step_num = next_step
        self.current_action = schedule_action
        self._prev_schedule_action = schedule_action

        # DEVICE_STOPPED handling: when Kineto signals that device collection
        # has stopped early (e.g. CUPTI buffer overflow), we enter
        # DEVICE_STOPPED and stay there until the current profiling cycle ends.
        #
        # Once in DEVICE_STOPPED (prev_action == DEVICE_STOPPED):
        # - We exit at the next cycle boundary, defined by either:
        #   - prev_schedule_action == RECORD_AND_SAVE: since active > 0,
        #     every cycle ends with RECORD_AND_SAVE, so the next step starts
        #     a new cycle regardless of warmup/wait shape.
        #   - current_action == NONE: safety hatch for user-defined schedules
        #     that may emit NONE without a preceding RECORD_AND_SAVE. (The
        #     standard schedule() builder always pairs them.)
        # - Otherwise we stay in DEVICE_STOPPED, absorbing the rest of the
        #   cycle.
        #
        # Entering DEVICE_STOPPED (override current_action to DEVICE_STOPPED):
        # - We enter when all of the following hold:
        #   - _is_kineto_stopped() is True: Kineto reports the stop.
        #   - use_device is set: a CPU-only profiler must not react to a
        #     stale stopped flag from a previous device profiler.
        #   - prev_action is WARMUP or RECORD: these have no built-in cleanup
        #     in their natural transitions.
        #   - current_action is not NONE: if the schedule is already stopping
        #     us (e.g. RECORD_AND_SAVE -> NONE), respect that rather than
        #     overriding.
        # - Otherwise we leave current_action as the schedule's output.
        # - prev_action == RECORD_AND_SAVE is excluded: the natural (R&S, *)
        #   transitions already call prepare_trace, so overriding here would
        #   skip the reset and waste the next cycle.

        if prev_action == ProfilerAction.DEVICE_STOPPED:
            at_cycle_boundary = (
                prev_schedule_action == ProfilerAction.RECORD_AND_SAVE
                or self.current_action == ProfilerAction.NONE
            )
            if not at_cycle_boundary:
                self.current_action = ProfilerAction.DEVICE_STOPPED
        elif (
            torch.autograd._is_kineto_stopped()
            and self.use_device is not None
            and prev_action in (ProfilerAction.WARMUP, ProfilerAction.RECORD)
            and self.current_action != ProfilerAction.NONE
        ):
            warn(
                "Device profiling activity collection was stopped early "
                f"at step {self.step_num}. Profiler schedule is proceeding "
                "until next cycle without actual profiler activity collection."
            )
            self.current_action = ProfilerAction.DEVICE_STOPPED

        self._transit_action(prev_action, self.current_action)
        if os.environ.get("KINETO_USE_DAEMON", "") or (
            is_fbcode() and os.environ.get("KINETO_FORCE_STEP_HOOK", "")
        ):
            prof.KinetoStepTracker.increment_step(PROFILER_STEP_NAME)

        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def set_custom_trace_id_callback(self, callback) -> None:
        """
        Set the trace ID generator. Called at the start of each cycle, so updating
        it between cycles yields distinct IDs per cycle.
        """
        self.custom_trace_id_callback = callback

    def get_trace_id(self):
        """
        Returns the current trace ID.
        """
        if self.profiler is None:
            return None
        return self.profiler.trace_id

    def _trace_ready(self) -> None:
        if self.on_trace_ready:
            self.on_trace_ready(self)

    def _transit_action(self, prev_action, current_action) -> None:
        action_list = self.action_map.get((prev_action, current_action))
        if action_list:
            for action in action_list:
                action()

    def _stats(self) -> prof._ProfilerStats | None:
        if self.profiler is None:
            return None
        return self.profiler._stats


class ExecutionTraceObserver(_ITraceObserver):
    """Execution Trace Observer

    Each process can have a single ExecutionTraceObserver instance. The observer
    can be added to record function callbacks via calling register_callback()
    explicitly. Without calling unregister_callback(), repeated calls to
    register_callback() will not add additional observers to record function
    callbacks. Once an ExecutionTraceObserver is created, the start() and stop()
    methods control when the event data is recorded.

    Deleting or calling unregister_callback() will remove the observer from the
    record function callbacks, finalize the output file, and will stop
    incurring any overheads.
    """

    def __init__(self) -> None:
        """
        Initializes the default states.
        """
        self._registered = False
        self._execution_trace_running = False
        self.extra_resources_collection = False
        self.resources_dir: str = ""
        self.output_file_path: str = ""
        self.output_file_path_observer: str = ""

    def __del__(self) -> None:
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    @staticmethod
    def build_execution_trace_obs_from_env() -> ExecutionTraceObserver | None:
        """
        Returns an ExecutionTraceObserver instance if the environment variable
        ENABLE_PYTORCH_EXECUTION_TRACE is set to 1, otherwise returns None.

        Configures the observer to also collect extra resources if the environment variable
        ``ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS=1``. These are resources such as generated kernels,
        index tensor data etc. that are required to make the Execution Trace replayable.
        """
        if os.environ.get("ENABLE_PYTORCH_EXECUTION_TRACE", "0") == "1":
            try:
                with tempfile.NamedTemporaryFile(
                    "w+t", suffix=".et.json", delete=False
                ) as fp:
                    filename = fp.name
            except Exception as e:
                warn(
                    f"Execution trace will not be recorded. Exception on creating default temporary file: {e}",
                    stacklevel=2,
                )
                return None
            et = ExecutionTraceObserver()
            et.register_callback(filename)
            # additionally, check if the env requires us to collect extra resources
            if os.environ.get("ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS", "0") == "1":
                et.set_extra_resource_collection(True)
            else:
                et.set_extra_resource_collection(False)
            return et
        return None

    def set_extra_resource_collection(self, val) -> None:
        """
        Collects extra resources such as generated kernels, index tensor data, and any other
        metadata that is required to complete the Execution Trace content.

        The caller should call this method with val=True after calling register_callback() if they want
        to collect the extra resources.
        """
        self.extra_resources_collection = val
        if self.extra_resources_collection:
            self.get_resources_dir(can_create=True)
        return

    def register_callback(self, output_file_path: str) -> Self:
        """
        Adds ET observer to record function callbacks. The data will be
        written to output_file_path.
        """

        def get_temp_uncompressed_file() -> str:
            with tempfile.NamedTemporaryFile("w+b", suffix=".json", delete=False) as fp:
                return fp.name

        if not self._registered:
            self.output_file_path = output_file_path
            if output_file_path.endswith(".gz"):
                output_file_path = get_temp_uncompressed_file()
            self.output_file_path_observer = output_file_path
            self._registered = _add_execution_trace_observer(output_file_path)
        return self

    def get_resources_dir(self, can_create=False) -> str | None:
        """
        Generates the resources directory for the generated kernels,
        or index tensor data or any other metadata that is required
        to complete the Execution Trace content.

        The directory is created right where the ET file is being output.

        Only works if the observer has called set_extra_resource_collection(val=True).

        Returns None if the observer is not configured with extra resource collection.
        """
        if not self.extra_resources_collection:
            return None
        if self.resources_dir:
            # already created
            return self.resources_dir
        generated_path = ExecutionTraceObserver.get_resources_dir_for_et_path(
            self.output_file_path, create_dir=can_create
        )
        if not generated_path:
            # could not find or create the resources dir
            return None
        self.resources_dir = generated_path
        return self.resources_dir

    @staticmethod
    def get_resources_dir_for_et_path(
        trace_path, create_dir: bool = False
    ) -> str | None:
        work_dir, file_name = os.path.split(trace_path)
        resource_dir = os.path.join(
            work_dir, os.path.splitext(file_name)[0] + "_resources"
        )
        if not os.path.exists(resource_dir):
            if create_dir:
                try:
                    os.mkdir(resource_dir)
                except Exception:
                    warn(
                        f"Execution trace exception when creating {resource_dir}",
                        stacklevel=2,
                    )
                    return None
            else:
                return None
        return resource_dir

    def unregister_callback(self) -> None:
        """
        Removes ET observer from record function callbacks.
        """

        def _save_triton_kernels() -> None:
            try:
                resource_dir = self.get_resources_dir()
            except Exception as e:
                warn(
                    f"Execution trace exception when generating resource directory: {e}",
                    stacklevel=2,
                )
                return
            if not resource_dir:
                return

            # Save the kernel paths for the generated kernels
            from torch._inductor.codecache import PyCodeCache

            kernel_files = [
                v.__file__
                for v in PyCodeCache.modules
                if getattr(v, "__file__", None) is not None
            ]

            for kernel_file in kernel_files:
                if kernel_file is None:
                    continue
                name = os.path.basename(kernel_file)
                dst = os.path.join(resource_dir, name)
                shutil.copyfile(kernel_file, dst)

        def _save_gz_file(uncompressed_file: str, output_file: str) -> None:
            print(f"Execution Trace: compressing {uncompressed_file} to {output_file}")
            with open(uncompressed_file, "rb") as fin:
                with gzip.open(output_file, "wb") as fout:
                    fout.writelines(fin)
            os.remove(uncompressed_file)

        if self._registered:
            self.stop()

            try:
                _save_triton_kernels()
            except Exception as e:
                warn(f"Execution trace failed to save kernels: {e}", stacklevel=2)

            _remove_execution_trace_observer()
            if self.output_file_path.endswith("gz"):
                _save_gz_file(self.output_file_path_observer, self.output_file_path)

            self._registered = False

    @property
    def is_registered(self):
        """
        Returns True if the execution trace observer is registered, otherwise False.
        """
        return self._registered

    def is_running(self):
        """
        Returns True if the observer is running, otherwise False.
        """
        return self._execution_trace_running

    def start(self) -> None:
        """
        Starts to capture.
        """
        if self._registered and not self._execution_trace_running:
            _enable_execution_trace_observer()
            self._execution_trace_running = True
            self._record_pg_config()

    def stop(self) -> None:
        """
        Stops to capture.
        """
        if self._execution_trace_running:
            _disable_execution_trace_observer()
            self._execution_trace_running = False

    def cleanup(self) -> None:
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    def get_output_file_path(self) -> str | None:
        """
        Returns the output file name or None.
        """
        if self.output_file_path:
            return self.output_file_path
        else:
            return None

    def _record_pg_config(self) -> None:
        # Records the PG config info to the trace as node:
        #  ## process_group:init ##
        if (
            self.is_registered
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            pg_config_info = torch.distributed.distributed_c10d._world.pg_config_info
            torch.autograd._record_function_with_args_enter(
                "## process_group:init ##",
                json.dumps(pg_config_info, cls=_NumpyEncoder),
            )
<|endoftext|># mypy: allow-untyped-defs
from contextlib import contextmanager
from typing import NoReturn


try:
    from torch._C import _itt
except ImportError:

    class _ITTStub:
        @staticmethod
        def _fail(*args, **kwargs) -> NoReturn:
            raise RuntimeError(
                "ITT functions not installed. Are you sure you have a ITT build?"
            )

        @staticmethod
        def is_available() -> bool:
            return False

        rangePush = _fail
        rangePop = _fail
        mark = _fail

    _itt = _ITTStub()  # type: ignore[assignment]


__all__ = ["is_available", "range_push", "range_pop", "mark", "range"]


def is_available():
    """
    Check if ITT feature is available or not
    """
    return _itt.is_available()


def range_push(msg):
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Arguments:
        msg (str): ASCII message to associate with range
    """
    return _itt.rangePush(msg)


def range_pop():
    """
    Pops a range off of a stack of nested range spans. Returns the
    zero-based depth of the range that is ended.
    """
    return _itt.rangePop()


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Arguments:
        msg (str): ASCII message to associate with the event.
    """
    return _itt.mark(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an ITT range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import Any, cast, Literal, TYPE_CHECKING

import torch
from torch._C._profiler import (
    _EventType,
    _ExtraFields_Allocation,
    _ExtraFields_TorchOp,
    _ProfilerEvent,
    _TensorMetadata,
    RecordScope,
)
from torch._utils import _element_size
from torch.profiler import _utils


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch._C import FunctionSchema
    from torch._C._autograd import _ProfilerResult


KeyAndID = tuple["Key", int]
TensorAndID = tuple["TensorKey", int]

log = logging.getLogger(__name__)


class Category(enum.Enum):
    INPUT = enum.auto()
    TEMPORARY = enum.auto()
    ACTIVATION = enum.auto()
    GRADIENT = enum.auto()
    AUTOGRAD_DETAIL = enum.auto()
    PARAMETER = enum.auto()
    OPTIMIZER_STATE = enum.auto()


_CATEGORY_TO_COLORS = {
    Category.PARAMETER: "darkgreen",
    Category.OPTIMIZER_STATE: "goldenrod",
    Category.INPUT: "black",
    Category.TEMPORARY: "mediumpurple",
    Category.ACTIVATION: "red",
    Category.GRADIENT: "mediumblue",
    Category.AUTOGRAD_DETAIL: "royalblue",
    None: "grey",
}

_CATEGORY_TO_INDEX = {c: i for i, c in enumerate(_CATEGORY_TO_COLORS)}


class Action(enum.Enum):
    PREEXISTING = enum.auto()
    CREATE = enum.auto()
    INCREMENT_VERSION = enum.auto()
    DESTROY = enum.auto()


_ACTION_TO_INDEX = {i: i.value for i in Action}


@dataclasses.dataclass(eq=True, unsafe_hash=False, frozen=True)
class Key:
    device: torch.device


@dataclasses.dataclass
class _Storage:
    """Bundle storage pointer and id.

    All profiling logic should use `allocation_id`, however it is useful to
    print storage pointers for debugging and unit tests sometimes look up
    values using the storage data pointer of a live Tensor."""

    ptr: int
    allocation_id: int

    def __repr__(self) -> str:
        return f"{hex(self.ptr):>18} ({self.allocation_id})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Storage) and self.allocation_id == other.allocation_id

    def __hash__(self) -> int:
        return hash(self.allocation_id)


@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class TensorKey(Key):
    """Hashable identifier for a storage which has been assigned an ID.

    A detailed description of Tensor IDs and why they are needed is given in
    `torch/csrc/profiler/collection.h` when `TensorID` is declared. To
    summarize, multiple Storage buffers can map to the same logical Tensor.
    This dataclass is used to refer to a concrete in-memory StorageImpl of
    a Tensor.
    """

    id: int
    storage: _Storage

    def __repr__(self) -> str:
        return f"id={self.id}: {repr(self.storage):<24} ({self.device})"

    def __lt__(self, other: TensorKey) -> bool:
        return self._as_sortable < other._as_sortable

    @staticmethod
    def _make(
        tensor_id: int | None,
        storage_ptr: int | None,
        allocation_id: int | None,
        device: torch.device,
    ) -> TensorKey | None:
        if (
            tensor_id is not None
            and storage_ptr is not None
            and allocation_id is not None
        ):
            return TensorKey(device, tensor_id, _Storage(storage_ptr, allocation_id))
        return None

    @classmethod
    def from_allocation(cls, alloc: _ExtraFields_Allocation) -> TensorKey | None:
        return cls._make(alloc.id, alloc.ptr, alloc.allocation_id, alloc.device)

    @classmethod
    def from_tensor(cls, t: _TensorMetadata | None) -> TensorKey | None:
        if t is not None:
            return cls._make(t.id, t.storage_data_ptr, t.allocation_id, t.device)
        return None

    @property
    def _as_sortable(self) -> tuple[int, int, str, int]:
        return self.id, self.storage.allocation_id, self.device.type, self.device.index


def _extract_parameters_and_gradients(
    node: _ProfilerEvent,
) -> Iterator[tuple[TensorKey | None, TensorKey | None]]:
    children = node.children

    # AccumulateGrad is used in the Autograd engine to handle gradient updates.
    # There are two possible cases:
    # 1) This is a newly created gradient Tensor. In that case there is nothing
    #    to accumulate, so autograd simply detaches the Tensor.
    #
    # 2) There is a preexisting gradient Tensor and we need to add the newly
    #    computed update. This is done with an in-place add (aten::add_) op.
    #    (The underscore suffix denotes "in-place".)
    if (
        node.typed[0] == _EventType.TorchOp
        and node.typed[1].scope == RecordScope.BACKWARD_FUNCTION
        # TODO(robieta): Move away from load bearing names
        and node.name == "torch::autograd::AccumulateGrad"
        and children
        and children[0].typed[0] == _EventType.TorchOp
        and children[0].name in ("aten::detach", "aten::add_")
        and children[0].typed[1].inputs
        and isinstance(children[0].typed[1].inputs[0], _TensorMetadata)
    ):
        yield None, TensorKey.from_tensor(children[0].typed[1].inputs[0])

    # We directly instrument `torch.nn.Module` and `torch.optim.Optimizer`
    # NOTE: The values captured by the python tracer are cached; they can be
    #       used to build up labels but do not imply that a Tensor was live at
    #       a particular time.
    elif node.typed[0] == _EventType.PyCall:
        typed_fields = node.typed[1]
        if typed_fields.module is not None and typed_fields.optimizer is not None:
            raise AssertionError("module and optimizer cannot both be set")
        if typed_fields.module is not None:
            for _, p, p_grad in typed_fields.module.parameters:
                yield TensorKey.from_tensor(p), TensorKey.from_tensor(p_grad)

        if typed_fields.optimizer is not None:
            for p, p_grad, _ in typed_fields.optimizer.parameters:
                yield TensorKey.from_tensor(p), TensorKey.from_tensor(p_grad)


def extract_parameters(node: _ProfilerEvent) -> Iterator[TensorKey]:
    for p, _p_grad in _extract_parameters_and_gradients(node):
        if p is not None:
            yield p


def extract_gradients(
    node: _ProfilerEvent,
) -> Iterator[tuple[TensorKey | None, TensorKey]]:
    for p, p_grad in _extract_parameters_and_gradients(node):
        if p_grad is not None:
            yield p, p_grad


def get_scopes(event: _ProfilerEvent | None) -> tuple[RecordScope, ...]:
    scopes = []
    while event:
        if event.typed[0] == _EventType.TorchOp:
            scopes.append(event.typed[1].scope)
        event = event.parent
    return tuple(scopes)


class SchemaMatcher:
    """Lookup operator schema based on profiled name.

    When profiling we record the operator's name but not the schema. However
    some analysis requires that information. Fortunately we can look up
    registered schema from the recorded name. We do not, however, record the
    overload and so we must compare the profiled arguments with all overloads
    to determine viable matches.

    Note: Once https://github.com/pytorch/pytorch/issues/78871 is completed
    this code will be obsolete.
    """

    @classmethod
    def inputs_are_mutable(cls, t: _ExtraFields_TorchOp) -> tuple[bool | None, ...]:
        """Determine which inputs may have mutated based on function schema.

        Note that we don't need to resolve down to a single schema to perform
        this analysis. An input is mutable if it is mutable in any overload. In
        practice, however, it is overwhelmingly common to match a single
        overload. If we cannot find any valid schema then we must be
        conservative and assume all inputs are mutable.
        """
        mutable: list[bool] | None = None
        for schema in cls.match_schemas(t):
            mutable = mutable or [False for _ in schema.arguments]
            for i, arg in enumerate(schema.arguments):
                # pyrefly: ignore [unsupported-operation]
                mutable[i] |= getattr(arg.alias_info, "is_write", False)

        return tuple(mutable or (None for _ in t.inputs))

    @classmethod
    def match_schemas(cls, t: _ExtraFields_TorchOp) -> tuple[FunctionSchema, ...]:
        signature = tuple(
            # Tensor
            TensorKey.from_tensor(i)
            if isinstance(i, _TensorMetadata)
            #
            # TensorList
            else [TensorKey.from_tensor(j) for j in i]
            if isinstance(i, list)
            #
            # Scalar and uncaptured inputs.
            else i
            for i in t.inputs
        )

        def matches(schema) -> bool:
            return len(schema.arguments) == len(signature) and all(
                cls._types_match(observed, schema_arg.type)
                for observed, schema_arg in zip(
                    signature, schema.arguments, strict=True
                )
            )

        return tuple(s for s in cls.lookup_schemas(t.name) or () if matches(s))

    @classmethod
    def _types_match(cls, observed, schema_type) -> bool:
        if isinstance(schema_type, torch._C.OptionalType):
            schema_type = schema_type.getElementType()
            return observed is None or cls._types_match(observed, schema_type)

        if isinstance(schema_type, torch._C.AnyType):
            return True

        if schema_type.isSubtypeOf(torch._C.ListType.ofTensors()):
            return isinstance(observed, list) and all(
                isinstance(i, TensorKey) for i in observed
            )

        type_map: tuple[tuple[Any, type | tuple[type, ...]], ...] = (
            (torch._C.TensorType, TensorKey),
            (torch._C.NoneType, type(None)),
            (torch._C.BoolType, bool),
            (torch._C.IntType, int),
            (torch._C.FloatType, float),
            (torch._C.ComplexType, complex),
            (torch._C.NumberType, (bool, int, float, complex)),
        )

        for jit_type, py_types in type_map:
            if isinstance(schema_type, jit_type):
                return isinstance(observed, py_types)

        # Profiler only records a subset of possible argument types. If we
        # reach this point then the schema must call for a type that profiler
        # does not record. Thus, the schema can only be a match if `observed`
        # is also None.
        return observed is None

    @staticmethod
    def lookup_schemas(name: str) -> tuple[FunctionSchema, ...] | None:
        # TODO(robieta):
        #   _jit_get_schemas_for_operator is quite expensive. (~100us / call)
        #   Consider adding `functools.lru_cache` if that becomes an issue.

        try:
            # Schema lookup will throw if `name` is malformed. (For example,
            # schemas must be namespaced and schema lookup will fail if name
            # does not include "::".) We simply catch the exception and return
            # `None` to denote that `name` cannot be an operator name.
            #
            # Note that record_function annotations also go through this path,
            # so it is expected that some names will not correspond to PyTorch
            # operators.
            if "::" not in name:
                return None
            return tuple(torch._C._jit_get_schemas_for_operator(name))
        except RuntimeError:
            return None


class OpTree:
    def __init__(self, result: _ProfilerResult) -> None:
        self._root_nodes = result.experimental_event_tree()
        self._sorted_nodes = tuple(sorted(self.dfs(), key=lambda x: x.start_time_ns))

    def dfs(self, *args, **kwargs) -> Iterator[_ProfilerEvent]:
        yield from _utils.traverse_dfs(self._root_nodes, *args, **kwargs)

    @property
    def sorted_nodes(self) -> tuple[_ProfilerEvent, ...]:
        return self._sorted_nodes


class SizeMap:
    def __init__(self, op_tree: OpTree) -> None:
        self._values: dict[TensorKey, int] = {}

        for node in op_tree.sorted_nodes:
            if node.typed[0] == _EventType.TorchOp:
                for t in self._flat_tensor_inputs(node.typed[1]):
                    self._update_values(t)

            elif node.typed[0] == _EventType.PyCall:
                typed_fields = node.typed[1]
                if (
                    typed_fields.module is not None
                    and typed_fields.optimizer is not None
                ):
                    raise AssertionError("module and optimizer cannot both be set")
                if typed_fields.module is not None:
                    for _, p, p_grad in typed_fields.module.parameters:
                        self._update_values(p)
                        self._update_values(p_grad)

                if typed_fields.optimizer is not None:
                    for p, p_grad, state in typed_fields.optimizer.parameters:
                        self._update_values(p)
                        self._update_values(p_grad)
                        for _, t in state:
                            self._update_values(t)

        allocations: dict[TensorKey, int] = {}
        for node in op_tree.sorted_nodes:
            if node.typed[0] == _EventType.Allocation:
                alloc_fields = node.typed[1]
                key = TensorKey.from_allocation(alloc_fields)
                if key:
                    new_size = abs(alloc_fields.alloc_size)
                    prior_size = allocations.setdefault(key, new_size)

                    # It is possible to resize Storage in PyTorch, however we
                    # key on data pointer so most resizes will be treated as a
                    # change in storage. The one corner case that cannot be
                    # handled is `realloc` which successfully resizes the
                    # storage. At time of writing this is not done anywhere in
                    # the core PyTorch codebase.
                    if prior_size != new_size:
                        delta = f"{prior_size} vs. {new_size}"
                        log.warning("Mismatch between allocation and free: %s", delta)

        self._values.update(allocations)

    def _update_values(self, t: _TensorMetadata | None) -> None:
        key = TensorKey.from_tensor(t)
        if key is not None and t is not None and t.layout == torch.strided:
            # Scalars are represented as zero dim Tensors
            n = max(
                i[0] * i[1] for i in zip(t.sizes or [1], t.strides or [1], strict=True)
            )

            num_bytes = n * _element_size(t.dtype)
            if num_bytes < 0:
                raise AssertionError(f"num_bytes must be non-negative, got {num_bytes}")
            self._values[key] = max(self._values.get(key, 0), num_bytes)

    @staticmethod
    def _flat_tensor_inputs(op: _ExtraFields_TorchOp) -> Iterator[_TensorMetadata]:
        for i in op.inputs:
            if isinstance(i, _TensorMetadata):
                yield i
            elif isinstance(i, list):
                yield from i

    def __getitem__(self, key: TensorKey):
        return self._values[key]


@dataclasses.dataclass()
class DataFlowEdge:
    input_version: int | None = None
    mutated: bool | None = False

    @property
    def is_allocation(self) -> bool:
        return self.input_version is None

    @property
    def is_deletion(self) -> bool:
        return self.mutated is None


class DataFlowNode:
    def __init__(self, event: _ProfilerEvent, graph: DataFlowGraph) -> None:
        self._event = event
        self._graph = graph
        self._edges: dict[TensorKey, DataFlowEdge] = self._determine_edges()

        for key, edge in self._edges.items():
            if edge.mutated and not edge.is_allocation:
                self._graph.bump(key)

        # Make sure the version bumping behavior matches what we expect.
        versions = {k: (v, self._graph.lookup(k)) for k, v in self.outputs.items()}
        if not all(i == j for i, j in versions.values()):
            raise AssertionError(f"version mismatch: {versions}, {self._edges}")

    def _determine_edges(self) -> dict[TensorKey, DataFlowEdge]:
        subtree = tuple(_utils.traverse_dfs([self._event]))

        # Start by populating edges from op inputs and outputs.
        mutable_by_key: dict[TensorKey | None, set[bool | None]] = {}
        for op in (i.typed[1] for i in subtree if i.typed[0] == _EventType.TorchOp):
            for op_input, mutable in zip(
                op.inputs, SchemaMatcher.inputs_are_mutable(op), strict=True
            ):
                # Tensor
                if isinstance(op_input, _TensorMetadata):
                    key = TensorKey.from_tensor(op_input)
                    mutable_by_key.setdefault(key, set()).add(mutable)

                # TensorList
                elif isinstance(op_input, list):
                    for op_input_i in op_input:
                        key = TensorKey.from_tensor(op_input_i)
                        mutable_by_key.setdefault(key, set()).add(mutable)

        edges: collections.defaultdict[TensorKey | None, DataFlowEdge]
        edges = collections.defaultdict(DataFlowEdge)
        for key, mutable_set in mutable_by_key.items():
            if key is not None:
                edges[key].input_version = self._graph.lookup(key) if key else -1

                # We consider an op to be mutated if we encounter a schema where it
                # is a mutable argument OR if it is ambiguous. (We never explicitly
                # see it in any schema.)
                mutated = (True in mutable_set) or (tuple(mutable_set) == (None,))
                edges[key].mutated = mutated

        # Then handle deletions. Note that deleting a Tensor implicitly adds
        # it as an input edge.
        for i in subtree:
            if i.typed[0] == _EventType.Allocation and i.typed[1].alloc_size < 0:
                key = TensorKey.from_allocation(i.typed[1])
                edge = edges[key]
                if key is not None and edge.mutated is None:
                    raise AssertionError(f"Double delete: {key}")
                edge.mutated = None
                edge.input_version = self._graph.lookup(key) if key else -1

        # And finally handle allocations. This step must be last, because the
        # previous two steps optimistically add input edges.
        for i in subtree:
            if i.typed[0] == _EventType.Allocation and i.typed[1].alloc_size > 0:
                edges[TensorKey.from_allocation(i.typed[1])].input_version = None

        # We don't need to sort the inputs, but it makes debugging and unit tests nicer.
        return dict(sorted((k, v) for k, v in edges.items() if k is not None))

    @property
    def inputs(self) -> dict[TensorKey, tuple[bool, int]]:
        return {
            # MyPy can't see through `is_allocation` to know that
            # `v.input_version` is not None.
            k: (bool(v.mutated), cast(int, v.input_version))
            for k, v in self._edges.items()
            if not v.is_allocation
        }

    @property
    def outputs(self) -> dict[TensorKey, int]:
        return {
            k: 0 if v.input_version is None else v.input_version + 1
            for k, v in self._edges.items()
            if (v.is_allocation and not v.is_deletion) or v.mutated
        }

    @property
    def intermediates(self) -> tuple[TensorKey, ...]:
        return tuple(
            k for k, v in self._edges.items() if v.is_allocation and v.is_deletion
        )

    @property
    def start_time(self) -> int:
        return self._event.start_time_ns


class DataFlowGraph:
    def __init__(self, op_tree: OpTree) -> None:
        self._op_tree = op_tree
        self._leaf_events = self._extract_leaf_events(op_tree)
        self._active_version: dict[TensorKey, int | None] = {}
        self._flow_nodes = [DataFlowNode(e, self) for e in self.leaf_events]
        self._flow_nodes.sort(key=lambda x: x.start_time)
        self.validate()

    @property
    def flow_nodes(self) -> tuple[DataFlowNode, ...]:
        return tuple(self._flow_nodes)

    def validate(self) -> None:
        # Check that each (Tensor, version) pair has a unique creation node
        outputs: set[tuple[TensorKey, int]] = set()
        for node in self.flow_nodes:
            node_outputs = set(node.outputs.items())
            duplicates = outputs & node_outputs
            if duplicates:
                raise AssertionError(
                    f"duplicate outputs: {node._event.name} {node._edges} {duplicates}"
                )
            outputs |= node_outputs

        # And check that `self._nodes` forms a valid topologically sorted DAG.
        tensor_versions: dict[TensorKey, int] = {}
        for node in self.flow_nodes:
            for key, (_, version) in node.inputs.items():
                expected = tensor_versions.get(key, 0)
                if expected != version:
                    raise AssertionError(
                        f"version mismatch for input: expected {expected}, got {version}"
                    )

            for key, version in node.outputs.items():
                prior_version = tensor_versions.get(key, version)
                if version < prior_version:
                    raise AssertionError(
                        f"version regression: {version} < {prior_version}"
                    )
                tensor_versions[key] = version

    @property
    def leaf_events(self) -> tuple[_ProfilerEvent, ...]:
        return self._leaf_events

    @staticmethod
    def _extract_leaf_events(op_tree: OpTree) -> tuple[_ProfilerEvent, ...]:
        """Partially traverse the op tree and extract top level ops.

        Consider the following code:
        ```
        with record_function("My annotation"):
            x.zero_()
            y.zero_()
        ```

        The op tree (assuming no Autograd) will look like:
          <Python context>
            TorchOp: "My annotation"
              TorchOp: zero_
                TorchOp: fill_
              TorchOp: zero_
                TorchOp: fill_

        The recursive structure of operator calls makes data flow unwieldy.
        In order to simplify analysis we would like to select the highest level
        ops to represent in the graph. In this case those are the `zero_` ops;
        the fact that `fill_` is called is an implementation detail. We also
        do not want to group everything under "My annotation" as this could
        create overly coarse bundles and lose critical semantics.

        To address this issue we walk over the graph and select the topmost
        torch ops ** which match at least one operator schema **. These form
        the leaves of the first pass through the op tree. (As well as any
        allocations or frees which are not part of a kernel.) These events
        form the logical nodes in our data flow graph.
        """

        leaf_events: list[_ProfilerEvent] = []

        def leaf_op(e: _ProfilerEvent) -> bool:
            return e.typed[0] == _EventType.TorchOp and (
                e.typed[1].scope == RecordScope.BACKWARD_FUNCTION
                or bool(SchemaMatcher.match_schemas(e.typed[1]))
            )

        def children_fn(e: _ProfilerEvent):
            if leaf_op(e) or e.tag == _EventType.Allocation:
                leaf_events.append(e)
                return []

            return e.children

        for _ in op_tree.dfs(children_fn=children_fn):
            pass

        return tuple(sorted(leaf_events, key=lambda x: x.start_time_ns))

    def lookup(self, key: TensorKey) -> int:
        version = self._active_version.setdefault(key, 0)
        if version is None:
            raise AssertionError(f"version for key {key} is None")
        return version

    def bump(self, key: TensorKey) -> None:
        prior_version = self._active_version.get(key, None)
        if prior_version is None:
            raise AssertionError(f"prior_version for key {key} is None")
        self._active_version[key] = prior_version + 1

    def delete(self, key: TensorKey) -> None:
        if self._active_version.setdefault(key, 0) is None:
            raise AssertionError(f"cannot delete key {key}, already deleted")
        self._active_version[key] = None


@dataclasses.dataclass
class CategoryElement:
    by_id: Category | None = None
    by_key: dict[TensorKey, Category] = dataclasses.field(default_factory=dict)
    by_version: dict[TensorAndID, Category] = dataclasses.field(default_factory=dict)

    # Used by unit tests to check internals. (And consequently by
    # MemoryProfile.lookup) This should not be used in any other capacity.
    _by_id_keyset: set[TensorKey] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class CategoryDict:
    _values: collections.defaultdict[int, CategoryElement] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(CategoryElement)
    )

    def set_by_id(self, key: TensorKey, category: Category) -> None:
        self._values[key.id].by_id = category
        self._values[key.id]._by_id_keyset.add(key)

    def set_by_key(self, key: TensorKey, category: Category) -> None:
        self._values[key.id].by_key[key] = category

    def set_by_version(self, key: TensorKey, version: int, category: Category) -> None:
        self._values[key.id].by_version[(key, version)] = category

    def setdefault_by_version(
        self, key: TensorKey, version: int, category: Category
    ) -> None:
        self._values[key.id].by_version.setdefault((key, version), category)

    def get(self, key: Key, version: int) -> Category | None:
        if isinstance(key, Key) and not isinstance(key, TensorKey):
            return None
        element = self._values[key.id]
        return (
            element.by_id
            or element.by_key.get(key, None)
            or element.by_version.get((key, version), None)
        )


class MemoryProfile:
    def __init__(self, result: _ProfilerResult) -> None:
        self._op_tree = OpTree(result)
        self._data_flow_graph = DataFlowGraph(self._op_tree)
        self._size_map = SizeMap(self._op_tree)
        self._categories = CategoryDict()

        self._set_gradients_and_temporaries()
        self._set_parameters_using_python_tracer()
        self._set_inputs()
        self._set_parameters_using_data_flow()
        self._set_activations()
        self._set_optimizer_state()
        self._set_autograd_detail()

    @property
    def timeline(self) -> tuple[tuple[int, Action, KeyAndID, int], ...]:
        output: list[tuple[int, Action, KeyAndID, int]] = []
        allocation_times: dict[tuple[TensorKey, bool], int] = {}
        live_unknown: dict[tuple[int, torch.device], Literal[True]] = {}

        for event in self._op_tree.dfs():
            if event.typed[0] == _EventType.Allocation:
                alloc_fields = event.typed[1]
                alloc_size = alloc_fields.alloc_size
                is_allocation = alloc_size > 0
                t = event.start_time_ns

                tkey = TensorKey.from_allocation(alloc_fields)
                if tkey is not None:
                    allocation_times[(tkey, is_allocation)] = t

                else:
                    key = Key(alloc_fields.device)
                    ptr_and_device = (alloc_fields.ptr, key.device)
                    if is_allocation:
                        if ptr_and_device in live_unknown:
                            output.append(
                                (t, Action.INCREMENT_VERSION, (key, 0), alloc_size)
                            )
                        else:
                            live_unknown[ptr_and_device] = True
                            output.append((t, Action.CREATE, (key, 0), alloc_size))
                    else:
                        output.append((t, Action.DESTROY, (key, 0), -alloc_size))
                        if not live_unknown.pop(ptr_and_device, False):
                            output.append(
                                (-1, Action.PREEXISTING, (key, 0), -alloc_size)
                            )

        snapshot = self._category_snapshot()
        last_version = dict(sorted(snapshot.keys()))

        events: list[tuple[int, Action, TensorAndID]] = [
            (-1, Action.PREEXISTING, (key, version))
            for key, version in snapshot
            if (key, True) not in allocation_times and version == 0
        ]

        for node in self._data_flow_graph.flow_nodes:
            for key, edge in node._edges.items():
                if edge.is_allocation:
                    t = allocation_times[(key, True)]
                    events.append((t, Action.CREATE, (key, 0)))

                elif edge.mutated:
                    t = node._event.start_time_ns
                    version = edge.input_version
                    if version is None:
                        raise AssertionError(f"input_version is None for key {key}")
                    events.append((t, Action.INCREMENT_VERSION, (key, version)))

                if edge.is_deletion:
                    t = allocation_times[(key, False)]
                    events.append((t, Action.DESTROY, (key, last_version[key])))

        output.extend(
            (time, action, (key, version), self._size_map[key])
            for time, action, (key, version) in events
        )

        output.sort(key=lambda x: (x[0], x[1].value))
        return tuple(output)

    def _is_gradient(self, *args, **kwargs) -> bool:
        return self._categories.get(*args, **kwargs) == Category.GRADIENT

    def _category_snapshot(self) -> dict[TensorAndID, Category | None]:
        all_tensor_versions: set[TensorAndID] = set()

        for node in self._data_flow_graph.flow_nodes:
            all_tensor_versions.update(((k, v) for k, (_, v) in node.inputs.items()))
            all_tensor_versions.update((key, 0) for key in node.intermediates)
            all_tensor_versions.update(node.outputs.items())

        for i in self._categories._values.values():
            all_tensor_versions.update((key, 0) for key in i._by_id_keyset)

        return {
            (key, version): self._categories.get(key, version)
            for key, version in sorted(all_tensor_versions)
        }

    def _any_version_depends_on_gradient(self) -> set[int]:
        """Extract IDs of Tensors which depend or will depend on a gradient.

        Note that this weakened definition of "depends" requires us to loop
        over the data flow graph multiple times because it allows dependency
        information to flow backward through edges and removes the guarantee
        that nodes are topologically sorted. (Or indeed, even that a valid
        topological order exists.) Put another way, we have converted an
        acyclic data flow graph into a cyclic graph and we are attempting to
        partition cycles involving a gradient from the rest of the graph.
        """
        depends_on_gradient: set[int] = set()
        while True:
            start_size = len(depends_on_gradient)
            for node in self._data_flow_graph.flow_nodes:
                ids = tuple(
                    key.id
                    for key, (_, version) in node.inputs.items()
                    if self._categories.get(key, version)
                    in (Category.GRADIENT, Category.PARAMETER)
                    or key.id in depends_on_gradient
                )

                if ids:
                    depends_on_gradient.update(ids)

                    depends_on_gradient.update(key.id for key in node.outputs)

            # We are guaranteed to exit because there is a finite set of
            # TensorAndID pairs. In practice we do not expect to loop more than
            # three times: once to identify the core parameter update loop,
            # once to fold the first step into that loop, and a third time
            # where no new elements are added.
            if len(depends_on_gradient) == start_size:
                return depends_on_gradient

    def _set_gradients_and_temporaries(self) -> None:
        """Mark Tensors which are unambiguous and simple to reason about."""

        # Gradients are straightforward to detect. We directly check the
        # `.grad` property in the Python tracer, and we can detect any new
        # gradient Tensors from `AccumulateGrad` ops.
        for event in self._op_tree.dfs():
            for _, p_grad in extract_gradients(event):
                self._categories.set_by_id(p_grad, Category.GRADIENT)

        # Similarly, temporary Tensors are easy to identify and are useful to
        # flag since they can make memory use "spikier" than one would
        # otherwise expect.
        for node in self._data_flow_graph.flow_nodes:
            for i in node.intermediates:
                self._categories.set_by_key(i, Category.TEMPORARY)

    def _set_parameters_using_python_tracer(self) -> None:
        for event in self._op_tree.dfs():
            for p in extract_parameters(event):
                if p is not None:
                    self._categories.set_by_id(p, Category.PARAMETER)

    def _set_inputs(self) -> None:
        """Mark inputs based on which Tensors are updated using gradients.

        The process for differentiating between inputs and activations is more
        involved. Most Tensors in a training loop depend on at least one
        gradient: parameters depend on them through updates, and activations
        and optimizer state depend on them transitively through parameters.
        Critically, we do not need to know which Tensors are parameters to
        apply this method; we can simply walk the data flow graph to build the
        set of all values which depend on a gradient and then obtain the set
        of inputs from the conjugate set.

        There is, however, one hiccup. The first time we see a parameter is
        generally on the forward pass of the first step. We know from
        inspection of the data flow graph that v1 of that Tensor depends on
        a gradient (provided we profile an optimizer step), but not v0. To
        address this problem we weaken the definition of "depends on a
        gradient" to "any version of this Tensor depends on a gradient",
        which in turn strengthens the criteria for the input set enough to
        filter the activations in the forward pass of the first step."""

        # All of this analysis is predicated on using at least one training
        # step (or parameters from the python tracer) to partition the graph.
        # Absent that we cannot determine which Tensors are inputs and which
        # ones are part of the model.
        depends_on_gradient = self._any_version_depends_on_gradient()

        # We only want to annotate Tensors which actually contribute to the
        # model calculation.
        produces_gradient: set[TensorAndID] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            tensors = {(key, version) for key, (_, version) in node.inputs.items()}
            tensors |= node.outputs.items()
            if any(
                self._categories.get(*i) in (Category.GRADIENT, Category.PARAMETER)
                or i in produces_gradient
                for i in tensors
            ):
                produces_gradient |= tensors

        # Don't include Tensors created in the backward pass, as these are
        # generally Autograd implementation details rather than proper inputs.
        input_candidates = produces_gradient.copy()
        for node in self._data_flow_graph.flow_nodes:
            if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
                input_candidates -= set(node.outputs.items())

        for key, version in input_candidates:
            if key.id not in depends_on_gradient:
                self._categories.setdefault_by_version(key, version, Category.INPUT)

    def _set_parameters_using_data_flow(self) -> None:
        """Deduce which Tensors are parameters.

        Consider the following code for the step of SGD with momentum
        (nesterov=False), where `d_p` is the gradient of `param` and `buf` is
        the momentum buffer.
        ```
          buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          d_p = buf
          param.add_(d_p, alpha=-lr)
        ```
        Both `param` and `buf` take a gradient and perform an in-place update.

        The python tracer will inspect calls to `nn.Module.forward` and
        `optim.Optimizer.step` to extract parameter and optimizer state
        respectively (including parameters), so this is generally a non-issue.

        However as a fallback we can also exploit several properties of
        parameters to distinguish them from other model state.

        First, they are directly used in the forward pass. (At this point we
        haven't established which parts of the graph correspond to the forward
        pass but we can deduce enough to suffice.) Some mutable state such as
        batch norm moving averages also contribute to the forward pass, but
        optimizer state does not.

        Second, a parameter is by definition used to compute at least one
        gradient and depends on at least one gradient.
        """
        snapshot = self._category_snapshot()

        # Determine which Tensors might be parameters based on forward pass
        # data flow. Note that these are only candidates; we filter nodes that
        # we know are part of the backward pass but that doesn't guarantee that
        # they are part of the forward pass.
        candidate_parameters: set[TensorAndID] = set()
        candidate_fwd_tensors: set[TensorAndID] = {
            i for i, category in snapshot.items() if category == Category.INPUT
        }

        for node in self._data_flow_graph.flow_nodes:
            inputs = {(key, value) for key, (_, value) in node.inputs.items()}
            if (
                # Don't check nodes in the backward pass.
                RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event)
                and not any(self._is_gradient(*i) for i in inputs)
                and not any(self._is_gradient(*i) for i in node.outputs.items())
                #
                # and only check nodes which depend on an input.
                and candidate_fwd_tensors.intersection(inputs)
            ):
                candidate_fwd_tensors |= node.outputs.items()
                candidate_parameters |= inputs.difference(candidate_fwd_tensors)

        # Require that each parameter eventually contributes to the value of a gradient
        used_for_gradient: set[TensorAndID] = set()
        for node in reversed(self._data_flow_graph.flow_nodes):
            if any(
                self._is_gradient(*i) or i in used_for_gradient
                for i in node.outputs.items()
            ):
                used_for_gradient.update(
                    (key, version) for key, (_, version) in node.inputs.items()
                )
        candidate_parameters.intersection_update(used_for_gradient)

        # and depends on a gradient.
        parameter_keys = {key.id for key, _ in candidate_parameters}
        parameter_keys &= self._any_version_depends_on_gradient()

        for key, _ in snapshot:
            if key.id in parameter_keys:
                self._categories.set_by_id(key, Category.PARAMETER)

    def _set_activations(self) -> None:
        """Flood the graph to identify activations."""

        required = {Category.INPUT, Category.ACTIVATION}
        also_allowed = {Category.PARAMETER, Category.TEMPORARY}
        for node in self._data_flow_graph.flow_nodes:
            inputs = {(key, value) for key, (_, value) in node.inputs.items()}
            input_categories = {self._categories.get(*i) for i in inputs}

            if (
                (input_categories & required)
                and not (input_categories - (required | also_allowed))
                #
                # Stop filling when we reach the backward pass.
                and RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event)
            ):
                for i in node.outputs.items():
                    self._categories.setdefault_by_version(*i, Category.ACTIVATION)

    def _set_optimizer_state(self) -> None:
        for event in self._op_tree.dfs():
            if event.typed[0] == _EventType.PyCall and event.typed[1].optimizer:
                parameters = event.typed[1].optimizer.parameters
                for _, t in it.chain.from_iterable(
                    (state for _, _, state in parameters)
                ):
                    key = TensorKey.from_tensor(t)
                    if key is not None:
                        self._categories.set_by_id(key, Category.OPTIMIZER_STATE)

    def _set_autograd_detail(self) -> None:
        prior = {None, Category.AUTOGRAD_DETAIL}
        for node in self._data_flow_graph.flow_nodes:
            if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
                for key, version in node.outputs.items():
                    if version == 0 or self._categories.get(key, version - 1) in prior:
                        self._categories.setdefault_by_version(
                            key, version, Category.AUTOGRAD_DETAIL
                        )


class MemoryProfileTimeline:
    def __init__(self, memory_profile) -> None:
        """The minimum representation of the memory profile timeline
        includes the memory timeline and categories. The timeline
        consists of [timestamp, action, (TensorKey, version), numbytes]
        elements, to denote any actions (pre-existing, create, destroy,
        or increment_version) that occurred to a specific Tensor for a
        chunk of memory. The categories help map each (TensorKey,
        version) pair into a category."""
        self.timeline = memory_profile.timeline
        self.categories = memory_profile._categories

    def _coalesce_timeline(self, device_str):
        """Convert the memory timeline and categories into a memory plot
        consisting of timestamps and their respective sizes by category
        for a given device.

        Input: device
        Output: [timestamps, sizes by category]
        """
        device = torch.device(device_str)
        times: list[int] = []
        sizes: list[list[int]] = []

        def update(key, version, delta) -> None:
            category = (
                self.categories.get(key, version)
                if isinstance(key, TensorKey)
                else None
            )
            index = _CATEGORY_TO_INDEX[category] + 1
            sizes[-1][index] += int(delta)

        t_min = -1
        for t, action, (key, version), numbytes in self.timeline:
            if key.device != device:
                continue

            # Convert timestamps from ns to us, to match trace events.
            if t != -1:
                t = int(t / 1000)

            # Save the smallest timestamp to populate pre-existing allocs.
            if t_min == -1 or (t < t_min and t > 0):
                t_min = t

            # Handle timestep
            if len(times) == 0:
                times.append(t)
                sizes.append([0] + [0 for _ in _CATEGORY_TO_INDEX])

            elif t != times[-1]:
                times.append(t)
                sizes.append(sizes[-1].copy())

            # Handle memory and categories
            if action in (Action.PREEXISTING, Action.CREATE):
                update(key, version, numbytes)

            elif action == Action.INCREMENT_VERSION:
                update(key, version, -numbytes)
                update(key, version + 1, numbytes)

            elif action == Action.DESTROY:
                update(key, version, -numbytes)

            else:
                raise ValueError(f"Unknown action: {action}")

        times = [t_min if t < 0 else t for t in times]
        return times, sizes

    def export_memory_timeline(self, path, device_str) -> None:
        """Saves the memory timeline as [times, sizes by category]
        as a JSON formatted file to the given path for the given
        device."""
        times, sizes = self._coalesce_timeline(device_str)
        # TODO: Write a faster serialize (orjson not available in CI)
        import json

        with open(path, "w") as f:
            json.dump([times, sizes], f)

    def export_memory_timeline_raw(self, path, device_str) -> None:
        """Saves the memory timeline as raw memory event tuples in the
        form of (timestamp, action, numbytes, category)
        as a JSON formatted file to the given path for the given
        device."""
        device = torch.device(device_str)
        raw_events: list[tuple[int, int, int, int]] = []

        def get_category_index(key, version):
            category = (
                self.categories.get(key, version)
                if isinstance(key, TensorKey)
                else None
            )
            return _CATEGORY_TO_INDEX[category]

        for t, action, (key, version), numbytes in self.timeline:
            if key.device != device:
                continue

            if action in (Action.PREEXISTING, Action.CREATE):
                raw_events.append(
                    # pyrefly: ignore [bad-argument-type]
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        numbytes,
                        get_category_index(key, version),
                    )
                )

            elif action == Action.INCREMENT_VERSION:
                raw_events.append(
                    # pyrefly: ignore [bad-argument-type]
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        -numbytes,
                        get_category_index(key, version),
                    )
                )
                raw_events.append(
                    # pyrefly: ignore [bad-argument-type]
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        numbytes,
                        get_category_index(key, version + 1),
                    )
                )

            elif action == Action.DESTROY:
                raw_events.append(
                    # pyrefly: ignore [bad-argument-type]
                    (
                        t,
                        _ACTION_TO_INDEX[action],
                        -numbytes,
                        get_category_index(key, version),
                    )
                )

            else:
                raise ValueError(f"Unknown action: {action}")

        import json

        with open(path, "w") as f:
            json.dump(raw_events, f)

    def export_memory_timeline_html(
        self, path, device_str, figsize=(20, 12), title=None
    ) -> None:
        """Exports the memory timeline as an HTML file which contains
        the memory timeline plot embedded as a PNG file."""
        # Check if user has matplotlib installed, return gracefully if not.
        import importlib.util

        matplotlib_spec = importlib.util.find_spec("matplotlib")
        if matplotlib_spec is None:
            print(
                "export_memory_timeline_html failed because matplotlib was not found."
            )
            return

        from base64 import b64encode
        from tempfile import NamedTemporaryFile

        import matplotlib.pyplot as plt
        import numpy as np

        mt = self._coalesce_timeline(device_str)
        times, sizes = np.array(mt[0]), np.array(mt[1])
        # For this timeline, start at 0 to match Chrome traces.
        t_min = min(times)
        times -= t_min
        stacked = np.cumsum(sizes, axis=1) / 1024**3
        device = torch.device(device_str)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

        # Plot memory timeline as stacked data
        fig = plt.figure(figsize=figsize, dpi=80)
        axes = fig.gca()
        for category, color in _CATEGORY_TO_COLORS.items():
            i = _CATEGORY_TO_INDEX[category]
            axes.fill_between(
                times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
            )
        fig.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS])
        # Usually training steps are in magnitude of ms.
        axes.set_xlabel("Time (ms)")
        axes.set_ylabel("Memory (GB)")
        title = "\n\n".join(
            ([title] if title else [])
            + [
                f"Max memory allocated: {max_memory_allocated / (1024**3):.2f} GiB \n"
                f"Max memory reserved: {max_memory_reserved / (1024**3):.2f} GiB"
            ]
        )
        axes.set_title(title)

        # Embed the memory timeline image into the HTML file
        with NamedTemporaryFile("w+b", suffix=".png") as tmpfile:
            fig.savefig(tmpfile, format="png")

            tmpfile.seek(0, 0)
            encoded = b64encode(tmpfile.read()).decode("utf-8")
            if not encoded:
                raise AssertionError("failed to encode image as base64")
            html = f"""<html>
<head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
<body>
  <img src='data:image/png;base64,{encoded}'>
</body>
</html>"""

            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
<|endoftext|>r"""
PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference.
Profiler's context manager API can be used to better understand what model operators are the most expensive,
examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

.. note::
    An earlier version of the API in :mod:`torch.autograd` module is considered legacy and will be deprecated.

"""

import os
from typing import Any
from typing_extensions import TypeVarTuple, Unpack

from torch._C._autograd import _supported_activities, DeviceType, kineto_available
from torch._C._profiler import _ExperimentalConfig, ProfilerActivity, RecordScope
from torch._environment import is_fbcode
from torch.autograd.profiler import KinetoStepTracker, record_function
from torch.optim.optimizer import Optimizer, register_optimizer_step_post_hook

from .profiler import (
    _KinetoProfile,
    ExecutionTraceObserver,
    profile,
    ProfilerAction,
    schedule,
    supported_activities,
    tensorboard_trace_handler,
)


__all__ = [
    "profile",
    "schedule",
    "supported_activities",
    "tensorboard_trace_handler",
    "ProfilerAction",
    "ProfilerActivity",
    "kineto_available",
    "DeviceType",
    "record_function",
    "ExecutionTraceObserver",
]

from . import itt


_Ts = TypeVarTuple("_Ts")


def _optimizer_post_hook(
    optimizer: Optimizer, args: tuple[Unpack[_Ts]], kwargs: dict[str, Any]
) -> None:
    KinetoStepTracker.increment_step("Optimizer")


if os.environ.get("KINETO_USE_DAEMON", "") or (
    is_fbcode() and os.environ.get("KINETO_FORCE_OPTIMIZER_HOOK", "")
):
    _ = register_optimizer_step_post_hook(_optimizer_post_hook)
<|endoftext|># mypy: allow-untyped-defs
"""
Validates Chrome traces emitted by ``torch.profiler`` against rules derived
from production issues.

Usage::

    from torch.profiler._trace_validator import validate_trace

    passed, violations = validate_trace("trace.pt.trace.json")
    for v in violations:
        print(v)
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from collections import defaultdict
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclasses.dataclass
class Violation:
    """A single rule violation found in a trace."""

    rule_name: str
    message: str

    def __str__(self) -> str:
        return f"{self.rule_name}: {self.message}"


def _load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    return [e for e in events if isinstance(e, dict)]


def _check_gpu_kernel_causality(events: list[dict]) -> list[Violation]:
    """For each (cudaLaunchKernel, GPU kernel) pair matched by External id,
    the GPU kernel must start at or after its cudaLaunchKernel."""
    cpu_launches: dict[int, dict] = {}
    gpu_kernels: dict[int, dict] = {}

    for ev in events:
        if ev.get("ph") != "X":
            continue
        args = ev.get("args", {})
        ext_id = args.get("External id")
        if ext_id is None:
            continue
        ext_id = int(ext_id)
        cat, name, ts = ev.get("cat", ""), ev.get("name", ""), float(ev.get("ts", 0))
        corr = args.get("correlation")

        if cat == "cuda_runtime" and name == "cudaLaunchKernel":
            if ext_id not in cpu_launches or ts < cpu_launches[ext_id]["ts"]:
                cpu_launches[ext_id] = {"ts": ts, "name": name, "corr": corr}
        elif cat == "kernel":
            if ext_id not in gpu_kernels or ts < gpu_kernels[ext_id]["ts"]:
                gpu_kernels[ext_id] = {"ts": ts, "name": name, "corr": corr}

    violations = []
    for ext_id, gpu in gpu_kernels.items():
        launch = cpu_launches.get(ext_id)
        if launch is None:
            continue
        if gpu["ts"] < launch["ts"]:
            skew = launch["ts"] - gpu["ts"]
            violations.append(
                Violation(
                    rule_name="_check_gpu_kernel_causality",
                    message=(
                        f"GPU kernel '{gpu['name']}' (External id={ext_id}, "
                        f"correlation={gpu['corr']}) starts {skew:.1f}us before "
                        f"its cudaLaunchKernel (External id={ext_id}, "
                        f"correlation={launch['corr']}), "
                        f"gpu_ts={gpu['ts']:.1f}, cpu_ts={launch['ts']:.1f}"
                    ),
                )
            )
    return violations


def _check_stream_wait_corr_id_populated(events: list[dict]) -> list[Violation]:
    """Stream Wait Events and Event Synchronize must have
    wait_on_cuda_event_record_corr_id >= 0."""
    TARGET_KINDS = {"Stream Wait Event", "Event Sync"}

    violations = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        args = ev.get("args", {})
        sync_kind = args.get("cuda_sync_kind")
        if sync_kind not in TARGET_KINDS:
            continue
        raw_corr = args.get("wait_on_cuda_event_record_corr_id")
        if raw_corr is None or int(raw_corr) < 0:
            ts = float(ev.get("ts", 0))
            violations.append(
                Violation(
                    rule_name="_check_stream_wait_corr_id_populated",
                    message=(
                        f"'{sync_kind}' event at ts={ts:.1f}us on "
                        f"device={args.get('device')} stream={args.get('stream')} "
                        f"has invalid wait_on_cuda_event_record_corr_id={raw_corr!r}"
                    ),
                )
            )
    return violations


def _check_stream_sync_overlap(events: list[dict]) -> list[Violation]:
    """For each Stream Synchronize on (device, stream), no kernel on that
    stream should still be running when the sync starts."""
    stream_syncs = []
    for ev in events:
        if (
            ev.get("ph") == "X"
            and ev.get("cat") == "cuda_sync"
            and ev.get("args", {}).get("cuda_sync_kind") == "Stream Sync"
        ):
            args = ev.get("args", {})
            stream_syncs.append(
                {
                    "ts": float(ev.get("ts", 0)),
                    "dur": float(ev.get("dur", 0)),
                    "stream": args.get("stream"),
                    "device": args.get("device"),
                }
            )
    if not stream_syncs:
        return []

    kernels_by_stream: dict[tuple, list[dict]] = defaultdict(list)
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") == "kernel":
            args = ev.get("args", {})
            key = (args.get("device"), args.get("stream"))
            ts = float(ev.get("ts", 0))
            kernels_by_stream[key].append(
                {
                    "ts": ts,
                    "end": ts + float(ev.get("dur", 0)),
                    "name": ev.get("name", ""),
                }
            )

    violations = []
    for sync in stream_syncs:
        key = (sync["device"], sync["stream"])
        for k in kernels_by_stream.get(key, []):
            if k["ts"] < sync["ts"] < k["end"]:
                overlap = k["end"] - sync["ts"]
                violations.append(
                    Violation(
                        rule_name="_check_stream_sync_overlap",
                        message=(
                            f"StreamSynchronize on device={sync['device']} "
                            f"stream={sync['stream']} at ts={sync['ts']:.1f}us "
                            f"but kernel '{k['name']}' (ends {k['end']:.1f}us) "
                            f"is still running ({overlap:.1f}us overlap)"
                        ),
                    )
                )
    return violations


_CUDA_EVENT_RECORD_NAMES = {
    "cudaEventRecord",
    "cudaEventRecord_ptsz",
    "cudaEventRecordWithFlags",
    "cudaEventRecordWithFlags_ptsz",
}


def _check_stream_wait_corr_id_in_past(events: list[dict]) -> list[Violation]:
    """wait_on_cuda_event_record_corr_id must point to a cudaEventRecord
    with cudaEventRecord.ts <= stream_wait.ts."""
    event_record_ts: dict[int, float] = {}
    for ev in events:
        if (
            ev.get("ph") == "X"
            and ev.get("cat") in ("cuda_runtime", "cuda_driver")
            and ev.get("name") in _CUDA_EVENT_RECORD_NAMES
        ):
            args = ev.get("args", {})
            ts = float(ev.get("ts", 0))
            for field in ("External id", "correlation"):
                cid = args.get(field)
                if cid is not None:
                    cid = int(cid)
                    if cid not in event_record_ts or ts < event_record_ts[cid]:
                        event_record_ts[cid] = ts

    violations = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        args = ev.get("args", {})
        if args.get("cuda_sync_kind") != "Stream Wait Event":
            continue
        ref = args.get("wait_on_cuda_event_record_corr_id")
        if ref is None or int(ref) < 0:
            continue
        ref = int(ref)
        sw_ts = float(ev.get("ts", 0))
        record_ts = event_record_ts.get(ref)

        if record_ts is None:
            violations.append(
                Violation(
                    rule_name="_check_stream_wait_corr_id_in_past",
                    message=(
                        f"Stream Wait Event at ts={sw_ts:.1f}us references "
                        f"corr_id={ref} but no matching cudaEventRecord in trace"
                    ),
                )
            )
        elif record_ts > sw_ts:
            lag = record_ts - sw_ts
            violations.append(
                Violation(
                    rule_name="_check_stream_wait_corr_id_in_past",
                    message=(
                        f"Stream Wait Event at ts={sw_ts:.1f}us references "
                        f"cudaEventRecord (corr_id={ref}) {lag:.1f}us in the future "
                        f"(event_record_ts={record_ts:.1f})"
                    ),
                )
            )
    return violations


_NCCL_REQUIRED_FIELDS = {
    "Collective name",
    "dtype",
    "In msg nelems",
    "Out msg nelems",
    "Group size",
}


def _check_nccl_metadata(events: list[dict]) -> list[Violation]:
    """record_param_comms events must carry: Collective name, dtype,
    In msg nelems, Out msg nelems, Group size."""
    violations = []
    for ev in events:
        if ev.get("ph") != "X" or ev.get("name") != "record_param_comms":
            continue
        args = ev.get("args", {})
        missing = _NCCL_REQUIRED_FIELDS - set(args.keys())
        if missing:
            violations.append(
                Violation(
                    rule_name="_check_nccl_metadata",
                    message=(
                        f"'record_param_comms' at ts={float(ev.get('ts', 0)):.1f}us "
                        f"missing metadata: {sorted(missing)}"
                    ),
                )
            )
    return violations


def _check_backward_seq_id_uniqueness(events: list[dict]) -> list[Violation]:
    """Per Sequence number, at most one distinct backward op name."""
    seq_to_ops: dict[int, list[str]] = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if "autograd::engine::evaluate_function:" not in name:
            continue
        args = ev.get("args", {})
        seq = args.get("Sequence number") or args.get("seq_num")
        if seq is None:
            continue
        seq = int(seq)
        op = name.split(":", 1)[-1].strip() if ":" in name else name
        if op not in seq_to_ops[seq]:
            seq_to_ops[seq].append(op)

    violations = []
    for seq, ops in seq_to_ops.items():
        if len(ops) > 1:
            violations.append(
                Violation(
                    rule_name="_check_backward_seq_id_uniqueness",
                    message=(
                        f"Sequence number {seq} shared by {len(ops)} backward "
                        f"ops: {ops}"
                    ),
                )
            )
    return violations


_RULES: list[Callable[[list[dict]], list[Violation]]] = [
    _check_gpu_kernel_causality,
    _check_stream_wait_corr_id_populated,
    _check_stream_sync_overlap,
    _check_stream_wait_corr_id_in_past,
    _check_nccl_metadata,
    _check_backward_seq_id_uniqueness,
]


def validate_trace(trace_path: str) -> tuple[bool, list[Violation]]:
    """
    Run all validation rules against a Chrome trace JSON file.

    Args:
        trace_path: Path to ``.pt.trace.json`` or ``.pt.trace.json.gz``.

    Returns:
        A ``(passed, violations)`` tuple.  ``passed`` is ``True`` when no
        violations were found.
    """
    events = _load_events(trace_path)
    all_violations: list[Violation] = []
    for rule in _RULES:
        all_violations.extend(rule(events))
    return len(all_violations) == 0, all_violations
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal


def _traverse(tree, next_fn, children_fn=lambda x: x.children, reverse: bool = False):
    order = reversed if reverse else lambda x: x
    remaining = deque(order(tree))
    while remaining:
        curr_event = next_fn(remaining)
        yield curr_event
        for child_event in order(children_fn(curr_event)):
            remaining.append(child_event)


traverse_dfs = functools.partial(_traverse, next_fn=lambda x: x.pop(), reverse=True)
traverse_bfs = functools.partial(
    _traverse, next_fn=lambda x: x.popleft(), reverse=False
)


def index_of_first_match(seq, predicate, start=0, end=None):
    if end is None or end >= len(seq):
        end = len(seq)
    for i in range(start, end):
        if predicate(seq[i]):
            return i
    return None


# Provide an OSS workaround for cudagraphs + CUPTI issue
# https://github.com/pytorch/pytorch/issues/75504
# TODO(dberard) - deprecate / remove workaround for CUDA >= 12, when
# we stop supporting older CUDA versions.
def _init_for_cuda_graphs() -> None:
    from torch.autograd.profiler import profile

    with profile():
        pass


@dataclass
class TimelineEvent:
    """Represents an event in the profiler timeline."""

    timestamp: int
    event_type: Literal["start", "end", "regular"]
    marker_type: Literal["filename", "node"] | None
    identifier: str | int | None
    event: dict[str, Any]


@dataclass
class ContextStackEntry:
    """Represents a context (filename or node) in the stack."""

    context_type: Literal["filename", "node"]
    identifier: str | int
    metadata: dict | None
    tid: int | None = None  # Thread ID associated with this context


def map_recorded_events_to_aten_ops_with_stack_trace(traced_data):
    """
    Maps recorded profiler events to their corresponding fx nodes and adds stack traces.

    Builds a timeline of all events (regular ops and FX markers for filenames/nodes),
    sorts by timestamp, then processes chronologically while maintaining a context stack of active
    filename/node scopes. Regular events are augmented with stack traces and node names from the
    innermost active context. Runtime is O(n log n) for n events.

    Args:
        traced_data: Json of profiler events from Chrome trace

    Returns:
        Dict mapping recorded event names to their aten operations with added stack traces
    """
    from torch.fx.traceback import _FX_METADATA_REGISTRY

    trace_events = traced_data.get("traceEvents", [])

    # Create event timeline
    event_timeline: list[TimelineEvent] = []

    def is_fx_marker_event(event):
        return (
            event.get("cat") == "cpu_op"
            and event.get("name", "").startswith("## ")
            and event.get("name", "").endswith(" ##")
        )

    def append_fx_marker_event(event_type, identifier, event):
        start_ts = event["ts"]
        end_ts = start_ts + event["dur"]
        event_timeline.append(
            TimelineEvent(start_ts, "start", event_type, identifier, event)
        )
        event_timeline.append(
            TimelineEvent(end_ts, "end", event_type, identifier, event)
        )

    for event in trace_events:
        if "ts" not in event or "dur" not in event:
            continue

        if is_fx_marker_event(event):
            content = event["name"][3:-3]

            if content.endswith(".py"):
                append_fx_marker_event("filename", content, event)
            else:
                try:
                    node_index = int(content)
                except ValueError:
                    pass
                append_fx_marker_event("node", node_index, event)  # type: ignore[possibly-undefined]

        else:
            # Regular event that needs augmentation
            start_ts = event["ts"]
            event_timeline.append(TimelineEvent(start_ts, "regular", None, None, event))

    # Sort by timestamp
    event_timeline.sort(key=lambda x: x.timestamp)

    # Process events in chronological order with a stack
    context_stack: list[ContextStackEntry] = []

    # Invariant: all start event has a corresponding end event
    for timeline_event in event_timeline:
        match timeline_event.event_type:
            case "start":
                if timeline_event.identifier is None:
                    raise AssertionError("identifier must not be None for start event")

                if timeline_event.marker_type == "filename":
                    if not isinstance(timeline_event.identifier, str):
                        raise AssertionError(
                            f"identifier must be str for filename marker, "
                            f"got {type(timeline_event.identifier).__name__}"
                        )
                    # Push filename context - query metadata registry on-demand
                    metadata = _FX_METADATA_REGISTRY.get(timeline_event.identifier)
                    tid = timeline_event.event.get("tid")
                    context_stack.append(
                        ContextStackEntry(
                            "filename", timeline_event.identifier, metadata, tid
                        )
                    )
                elif timeline_event.marker_type == "node":
                    # Find the current filename from stack
                    current_file_metadata = None
                    tid = timeline_event.event.get("tid")
                    for ctx_entry in reversed(context_stack):
                        if (
                            ctx_entry.context_type == "filename"
                            and ctx_entry.tid == tid
                        ):
                            current_file_metadata = ctx_entry.metadata
                            break

                    if current_file_metadata:
                        node_metadata = current_file_metadata.get("node_metadata", {})
                        if timeline_event.identifier in node_metadata:
                            node_meta: dict | None = node_metadata[
                                timeline_event.identifier
                            ]
                            context_stack.append(
                                ContextStackEntry(
                                    "node", timeline_event.identifier, node_meta, tid
                                )
                            )

            case "end":
                # Pop from stack - search backwards to find matching context
                for i in range(len(context_stack) - 1, -1, -1):
                    ctx_entry = context_stack[i]
                    if (
                        timeline_event.marker_type == ctx_entry.context_type
                        and timeline_event.identifier == ctx_entry.identifier
                    ):
                        context_stack.pop(i)
                        break

            case "regular":
                # Apply metadata from current context stack
                # Find the most specific context (node takes precedence over filename)
                # Only augment events with the same tid as the file/node event matched
                current_stack_trace = None
                current_node_name = None
                event_tid = timeline_event.event.get("tid")

                for ctx_entry in reversed(context_stack):
                    # Only apply metadata from contexts with matching tid
                    if ctx_entry.tid == event_tid:
                        if ctx_entry.context_type == "node" and ctx_entry.metadata:
                            current_stack_trace = ctx_entry.metadata.get(
                                "stack_trace", "No model stack trace available"
                            )
                            current_node_name = ctx_entry.metadata.get("name", "")
                            # Do we want to only attach the stack trace of the lowest node or stack trace of all nodes
                            # if nodes are nested, e.g. in nested graph modules
                            break

                # Augment the event
                if current_stack_trace or current_node_name:
                    args = timeline_event.event.setdefault("args", {})
                    if current_stack_trace:
                        args["stack_trace"] = current_stack_trace
                    if current_node_name:
                        args["node_name"] = current_node_name
<|endoftext|>import os
import site
import sys

import torch


def _prefix_regex() -> list[str]:
    raw_paths = (
        site.getsitepackages()
        + sys.path
        + [site.getuserbase()]
        + [site.getusersitepackages()]
        + [os.path.dirname(os.path.dirname(torch.__file__))]
    )

    path_prefixes = sorted({os.path.abspath(i) for i in raw_paths}, reverse=True)
    if not all(isinstance(i, str) for i in path_prefixes):
        raise AssertionError("all path_prefixes must be strings")
    return [i + os.sep for i in path_prefixes]
<|endoftext|># mypy: allow-untyped-defs
"""Stream chrome trace JSON directly from ITraceActivity objects.

Bypasses Kineto's C++ ChromeTraceLogger, writing events one-at-a-time
through a (possibly gzip-compressed) text writer so we never materialize
the full JSON string in memory.
"""

from __future__ import annotations

import gzip
import json
import os
import time as _time
from typing import IO

import torch


_TRIMONTH_SECONDS = 7889238

_FLOW_NAMES = {1: "fwdbwd", 2: "ac2g"}

_PARAM_COMMS_CALL_NAME = "record_param_comms"

_EXCLUDED_EXTERNAL_ID_TYPES = {
    "gpu_memcpy",
    "gpu_memset",
    "kernel",
    "cuda_runtime",
    "cuda_driver",
    "privateuse1_runtime",
    "privateuse1_driver",
}


def _trimester_base_ns() -> int:
    """Epoch nanoseconds at the start of the current trimonth interval.

    Matches libkineto's ChromeTraceBaseTime (floor to 7889238-second
    intervals) to keep JSON timestamps small enough for double precision.
    """
    return (int(_time.time()) // _TRIMONTH_SECONDS) * _TRIMONTH_SECONDS * 1_000_000_000


def _ns_to_us(time_ns: int) -> str:
    return f"{time_ns // 1000}.{time_ns % 1000:03d}"


def _sanitize_tid(tid: int) -> int:
    if tid == -(2**63):
        return 0
    return abs(tid)


def _json_escape(s: str) -> str:
    return json.dumps(s)


def _write_metadata_event(
    f: IO[str], name: str, ts: str, pid, tid, arg_key: str, arg_value: str
):
    f.write(
        f'{{"ph":"M","name":"{name}","ts":{ts},'
        f'"pid":{pid},"tid":{tid},'
        f'"args":{{"{arg_key}":{arg_value}}}}},\n'
    )


def export_chrome_trace(
    kineto_results,
    path: str,
    metadata: dict[str, str] | None = None,
):
    """Export chrome trace from ITraceActivity objects, streaming to disk.

    ``kineto_results`` is a ``_ProfilerResult`` that exposes
    ``trace_activities()`` and ``trace_start_ns()``.

    Writes ``.json`` or ``.json.gz`` depending on the file extension.
    """
    activities = kineto_results.trace_activities()
    base_ns = _trimester_base_ns()

    seen_devices: dict[int, int] = {}
    seen_resources: dict[tuple[int, int], int] = {}
    host_pid = os.getpid()
    min_ts = 2**63
    max_end_ts = 0

    for act in activities:
        did = act.device_id()
        rid = _sanitize_tid(act.resource_id())
        ts = act.timestamp()
        dur = act.duration()
        min_ts = min(min_ts, ts)
        max_end_ts = max(max_end_ts, ts + max(dur, 0))
        if did not in seen_devices:
            seen_devices[did] = ts
        key = (did, rid)
        if key not in seen_resources:
            seen_resources[key] = ts

    def _rel(ns: int) -> str:
        return _ns_to_us(max(ns - base_ns, 0))

    with gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w") as f:
        device_props = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            device_props.append(
                {
                    "id": i,
                    "name": p.name,
                    "totalGlobalMem": p.total_memory,
                    "computeMajor": p.major,
                    "computeMinor": p.minor,
                    "maxThreadsPerBlock": p.max_threads_per_block,  # pyrefly: ignore[missing-attribute]
                    "maxThreadsPerMultiprocessor": p.multi_processor_count,
                    "regsPerMultiprocessor": p.regs_per_multiprocessor,  # pyrefly: ignore[missing-attribute]
                    "warpSize": p.warp_size,
                }
            )

        f.write("{\n")
        f.write('"schemaVersion": 1,\n')
        f.write(f'"deviceProperties": {json.dumps(device_props)},\n')
        if metadata:
            for k, v in metadata.items():
                f.write(f"{_json_escape(k)}: {v},\n")
        f.write('"displayTimeUnit": "ms",\n')
        f.write(f'"baseTimeNanoseconds": {base_ns},\n')
        f.write('"traceEvents": [\n')

        for did, ts in sorted(seen_devices.items()):
            ts_str = _rel(ts)
            if did == host_pid or did < 0:
                label = "CPU" if did == host_pid else "Overhead"
                sort_idx = did if did >= 0 else 0x1000000
            else:
                label = f"GPU {did}"
                sort_idx = 5000000 + did

            _write_metadata_event(f, "process_name", ts_str, did, 0, "name", '"python"')
            _write_metadata_event(
                f, "process_labels", ts_str, did, 0, "labels", f'"{label}"'
            )
            _write_metadata_event(
                f,
                "process_sort_index",
                ts_str,
                did,
                0,
                "sort_index",
                str(sort_idx),
            )

        for (did, rid), ts in sorted(seen_resources.items()):
            ts_str = _rel(ts)
            if did == host_pid or did < 0:
                rname = f"thread {rid}"
            else:
                rname = f"stream {rid} "
            _write_metadata_event(
                f, "thread_name", ts_str, did, rid, "name", f'"{rname}"'
            )
            _write_metadata_event(
                f, "thread_sort_index", ts_str, did, rid, "sort_index", str(rid)
            )

        for act in activities:
            ts = act.timestamp()
            dur = act.duration()

            did = act.device_id()
            rid = _sanitize_tid(act.resource_id())
            cat = act.type()
            name = act.name()
            ts_str = _rel(ts)
            dur_str = _ns_to_us(max(dur, 0))

            args_parts = []
            linked_corr = act.linked_correlation_id()
            if linked_corr:
                args_parts.append(f'"External id": {linked_corr}')
            elif cat not in _EXCLUDED_EXTERNAL_ID_TYPES:
                corr = act.correlation_id()
                if corr:
                    args_parts.append(f'"External id": {corr}')

            md = act.metadata_json()
            if md:
                args_parts.append(md)

            if cat == "kernel":
                linked = act.linked_activity()
                if linked is not None and linked.name() == _PARAM_COMMS_CALL_NAME:
                    linked_md = linked.metadata_json()
                    if linked_md:
                        args_parts.append(linked_md)

            f.write(
                f'{{"ph":"X","cat":{_json_escape(cat)},'
                f'"name":{_json_escape(name)},'
                f'"pid":{did},"tid":{rid},'
                f'"ts":{ts_str},"dur":{dur_str}'
            )
            if args_parts:
                f.write(f',"args":{{{",".join(args_parts)}}}')
            f.write("},\n")

            flow_id = act.flow_id()
            if flow_id > 0:
                flow_cat = _FLOW_NAMES.get(act.flow_type(), "ac2g")
                if act.flow_start():
                    f.write(
                        f'{{"ph":"s","id":{flow_id},'
                        f'"pid":{did},"tid":{rid},'
                        f'"ts":{ts_str},"cat":"{flow_cat}","name":"{flow_cat}"}},\n'
                    )
                else:
                    f.write(
                        f'{{"ph":"f","id":{flow_id},'
                        f'"pid":{did},"tid":{rid},'
                        f'"ts":{ts_str},"cat":"{flow_cat}","name":"{flow_cat}","bp":"e"}},\n'
                    )

        if activities:
            its = _rel(min_ts)
            trace_dur = _ns_to_us(max(max_end_ts - min_ts, 0))
            f.write(
                f'{{"ph":"X","cat":"Trace","name":"PyTorch Profiler (0)",'
                f'"pid":"Spans","tid":"PyTorch Profiler",'
                f'"ts":{its},"dur":{trace_dur},"args":{{"Op count": 0}}}},\n'
            )
            _write_metadata_event(
                f,
                "process_sort_index",
                its,
                '"Spans"',
                0,
                "sort_index",
                str(0x20000000),
            )
            f.write(
                f'{{"ph":"i","s":"g","name":"Iteration Start: PyTorch Profiler",'
                f'"pid":"Traces","tid":"Trace PyTorch Profiler","ts":{its}}},\n'
            )

        end_ts = _rel(max_end_ts + 1000)
        f.write(
            f'{{"ph":"i","s":"g","name":"Record Window End",'
            f'"pid":"","tid":"","ts":{end_ts}}}\n'
        )

        f.write(f'],\n"traceName": {_json_escape(path)}\n}}\n')
<|endoftext|># mypy: allow-untyped-defs
"""CUPTI PM-sampling: continuous GPU performance-monitor sampling (SM-active %, DRAM-throughput
%) that runs concurrently with the activity monitor.

Each CUDA device has at most one :class:`PmSampler` -- a per-device singleton (``PmSampler(device)``
returns the one instance) -- so no device is ever driven by two collectors. Lifecycle follows
NVIDIA's cupti-python PM sampling sample: ``enable`` -> ``configure`` -> ``start`` -> drain ->
``stop`` -> release. ``enable`` initializes the *process-global* CUPTI profiler (idempotent), but
its deinitialize is NOT refcounted by CUPTI -- a second call segfaults -- so PmSampler refcounts its
live collectors and deinitializes exactly once, when the last is released. That makes sampling
several devices at once safe: one device's teardown never deinitializes a profiler another device is
still using."""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, TYPE_CHECKING

import numpy as np
from cupti import (  # pyrefly: ignore[missing-import]
    cupti as _cupti,
    pm_sampling as _pm_sampling,
    profiler_host as _profiler_host,
)
from cupti.pm_sampling import (  # pyrefly: ignore[missing-import]
    metrics_to_c_array as _metrics_to_c_array,
)

import torch


if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing_extensions import Self


logger = logging.getLogger(__name__)

# Interval + look-back defaults; override process-wide (first-come-first-serve) via
# PmSampler.configure() -- there is no env var. Metrics are NOT process-wide -- each consumer
# brings its own via add_consumer(). The HW sampling interval (GPU_TIME_INTERVAL units = ns),
# default 1 ms.
_DEFAULT_SAMPLING_INTERVAL_NS = 1_000_000
# Retained look-back window: the sampler is sized (max_samples + ring) to cover this much wall-clock
# at the interval. Kept modest because the host counter-data image is ~18 KiB/sample (see
# _counter_data_size), so a decode of window/interval samples allocates that many. Default 10 s.
_DEFAULT_LOOKBACK_WINDOW_NS = 10_000 * 1_000_000


def _counter_data_size(
    pm_sampling_object: int, metrics: list[str], max_samples: int
) -> int:
    _, metric_names_ptr = _metrics_to_c_array(metrics)
    p = _cupti.PmSampling_GetCounterDataSize_Params()
    p.struct_size = _cupti.PM_SAMPLING_GET_COUNTER_DATA_SIZE_PARAMS_STRUCT_SIZE
    p.p_pm_sampling_object = pm_sampling_object
    p.p_metric_names = metric_names_ptr
    p.num_metrics = len(metrics)
    p.max_samples = max_samples
    _cupti.pm_sampling_get_counter_data_size(p.ptr)
    return p.counter_data_size


@functools.cache
def _device_chip_name(device: int) -> str:
    p = _cupti.Device_GetChipName_Params()
    p.struct_size = _cupti.DEVICE_GET_CHIP_NAME_PARAMS_STRUCT_SIZE
    p.device_index = device
    _cupti.device_get_chip_name(p.ptr)
    return p.p_chip_name


def _pm_sampling_disable(pm_sampling_object: int) -> None:
    p = _cupti.PmSampling_Disable_Params()
    p.struct_size = _cupti.PM_SAMPLING_DISABLE_PARAMS_STRUCT_SIZE
    p.p_pm_sampling_object = pm_sampling_object
    _cupti.pm_sampling_disable(p.ptr)


def _profiler_deinitialize() -> None:
    p = _cupti.Profiler_DeInitialize_Params()
    p.struct_size = _cupti.PROFILER_DEINITIALIZE_PARAMS_STRUCT_SIZE
    _cupti.profiler_deinitialize(p.ptr)


@functools.cache
def supported_metrics(*, with_sub_metrics: bool = False) -> frozenset[str]:
    try:
        chip_name = _device_chip_name(torch.cuda.current_device())
        host = _profiler_host.ProfilerHost(chip_name, _cupti.ProfilerType.PM_SAMPLING)
        host.initialize()
    except Exception as e:
        logger.warning("PM sampling could not initialize the profiler host: %s", e)
        return frozenset()
    try:
        names: set[str] = set()
        for mt in (
            _cupti.MetricType.COUNTER,
            _cupti.MetricType.RATIO,
            _cupti.MetricType.THROUGHPUT,
        ):
            for base in host.get_base_metrics(mt):
                names.update(
                    host.get_sub_metrics(base, mt) if with_sub_metrics else (base,)
                )
        return frozenset(names)
    except Exception as e:
        logger.warning("PM sampling metric enumeration failed: %s", e)
        return frozenset()
    finally:
        host.deinitialize()


class _Consumer:
    """Internal per-consumer state held by the sampler: the metrics it wants and its cursor (the last
    start_ns returned to it). The caller gets a :class:`_Handle`, not this."""

    __slots__ = ("metrics", "cursor")

    def __init__(self, metrics: list[str]) -> None:
        self.metrics = metrics
        self.cursor = -1  # last start_ns returned; -1 = nothing yet


class _Handle:
    """The caller's handle to a registered consumer: :meth:`poll` for its samples, :meth:`detach` to
    unregister. Dropping the handle also detaches (a GC safety net), but prefer :meth:`detach` for
    deterministic teardown -- detaching the last consumer releases the GPU-clock lock and the CUPTI
    session. The sampler references only the :class:`_Consumer` record, not the handle, so a dropped
    handle is collectable and its :meth:`__del__` fires."""

    __slots__ = ("_sampler", "_consumer")

    def __init__(self, sampler: PmSampler, consumer: _Consumer) -> None:
        self._sampler = sampler
        self._consumer = consumer

    def poll(self) -> dict[str, Any] | None:
        """Drain the HW ring and return this consumer's samples newer than its cursor, sliced to its
        metrics, as a frame (``start_ns`` / ``device_id`` / one column per metric) -- or None if
        none. Poll on your own cadence, but before the KEEP_LATEST ring fills (span = max_samples *
        interval) and drops the oldest; samples another consumer hasn't polled stay retained until it
        does."""
        return self._sampler._poll_consumer(self._consumer)

    def detach(self) -> None:
        """Unregister; the session disables once the last consumer detaches. :meth:`poll` first for a
        final frame -- detaching returns nothing."""
        self._sampler._detach(self._consumer)

    def __del__(self) -> None:
        try:
            self.detach()
        except Exception:
            pass


class PmSampler:
    """The single PM-sampling session on one CUDA device -- a per-device singleton
    (``PmSampler(device)`` returns the one instance, default the current device). Only one PM session
    per device is possible, so all users of a device share this one object. The process-global CUPTI
    profiler deinit is not refcounted by CUPTI, so PmSampler refcounts its live collectors
    (:attr:`_active`) and deinitializes once, when the last is torn down -- keeping concurrent
    sampling on multiple devices safe.

    :meth:`add_consumer` (the metrics wanted) returns a handle; poll the handle for its samples and
    detach it -- or just drop it -- to unregister. The session samples the *union* of all consumers'
    metrics. Delivery is pull-based: ``handle.poll()`` drains the HW ring and returns the samples
    newer than the handle's
    cursor, sliced to its own metrics. decode drains, so a sample another consumer has not yet polled
    is retained (bounded to the look-back window) and GC'd once every consumer has passed it -- a
    lone, caught-up consumer keeps no copy. Frames carry RAW CUPTI-clock-ns timestamps in
    ``start_ns`` -- converting to trace/epoch time is the consumer's job, so the sampler has no clock
    dependency."""

    _instances: dict[int, PmSampler] = {}
    _instances_lock = threading.Lock()
    _active = 0
    # Process-wide sampling config; each device's singleton snapshots these when first created.
    # Set via configure() (first-come-first-serve, no env var); defaults otherwise.
    _sampling_interval_ns: int = _DEFAULT_SAMPLING_INTERVAL_NS
    _lookback_window_ns: int = _DEFAULT_LOOKBACK_WINDOW_NS
    _configured: bool = False

    @classmethod
    def configure(
        cls,
        *,
        sampling_interval_ms: float | None = None,
        lookback_window_ms: float | None = None,
    ) -> None:
        """Set the process-wide PM-sampling interval and look-back window (milliseconds).
        First-come-first-serve: locked once a call lands OR the first sampler is instantiated
        (whichever comes first) -- a live per-device session can't be resized -- so a later
        call is ignored with a warning. Call before the first consumer registers; an unset arg
        keeps its current value."""
        with cls._instances_lock:
            if cls._configured:
                logger.warning(
                    "PmSampler.configure() ignored: already configured (first-come-first-serve)"
                )
                return
            if sampling_interval_ms is not None:
                cls._sampling_interval_ns = int(sampling_interval_ms * 1_000_000)
            if lookback_window_ms is not None:
                cls._lookback_window_ns = int(lookback_window_ms * 1_000_000)
            cls._configured = True

    @classmethod
    def get_config(cls) -> dict[str, int | bool]:
        """The process-wide PM-sampling config a new device session will snapshot: the
        sampling interval and look-back window (nanoseconds), and whether configure() has
        pinned it (first-come-first-serve)."""
        return {
            "sampling_interval_ns": cls._sampling_interval_ns,
            "lookback_window_ns": cls._lookback_window_ns,
            "configured": cls._configured,
        }

    def __new__(cls, device: int | None = None) -> Self:
        if device is None:
            device = torch.cuda.current_device()
        with cls._instances_lock:
            inst = cls._instances.get(device)
            if inst is None:
                inst = super().__new__(cls)
                inst._init(device)
                cls._instances[device] = inst
            return inst

    def _init(self, device: int) -> None:
        self._device = device
        # Snapshot the process-wide config at creation and lock it: a live session can't be
        # resized, so from here configure() is a no-op (first-come-first-serve). Runs under
        # _instances_lock (held by __new__), the same lock configure() takes.
        cls = type(self)
        self._sampling_interval_ns = cls._sampling_interval_ns
        self._lookback_window_ns = cls._lookback_window_ns
        cls._configured = True
        samples = self._lookback_window_ns // self._sampling_interval_ns
        self._max_samples = max(1, samples)
        self._consumers: list[_Consumer] = []
        self._metric_names: list[str] = []
        self._col: Any = None
        # Samples drained from the HW ring but not yet consumed by every consumer (columnar, in the
        # union metric order). Held only while a consumer lags; GC'd by the min-cursor watermark.
        self._retained_ts = np.empty(0, dtype=np.int64)
        self._retained_vals = np.empty((0, 0), dtype=np.float64)
        self._lock = threading.RLock()

    def add_consumer(self, metrics: Iterable[str]) -> _Handle:
        """Register a consumer for ``metrics`` and return its handle (``handle.poll()`` /
        ``handle.detach()``; dropping the handle also detaches). Raises ValueError if ``metrics`` is
        empty or the resulting union can't be collected in one PM pass."""
        metrics = list(metrics)
        if not metrics:
            raise ValueError("PM sampling requires a non-empty metric set")
        consumer = _Consumer(metrics)
        with self._lock:
            self._check_single_pass(self._union_of([*self._consumers, consumer]))
            self._consumers.append(consumer)
            self._reconfigure()
        return _Handle(self, consumer)

    def _detach(self, consumer: _Consumer) -> None:
        with self._lock:
            if consumer not in self._consumers:
                return
            self._consumers.remove(consumer)
            self._reconfigure()
            self._gc()  # the leaver no longer pins the GC watermark

    @staticmethod
    def _union_of(consumers: list[_Consumer]) -> list[str]:
        seen: set[str] = set()
        union: list[str] = []
        for c in consumers:
            for m in c.metrics:
                if m not in seen:
                    seen.add(m)
                    union.append(m)
        return union

    def _reconfigure(self) -> None:
        desired = self._union_of(self._consumers)
        if not desired:
            self._teardown()
        elif desired != self._metric_names:
            # Rebuild, not in-place reconfigure -- a live stop/configure/start segfaults on decode.
            self._teardown()
            self._metric_names = desired
            self._start()

    def _poll_consumer(self, consumer: _Consumer) -> dict[str, Any] | None:
        # Implements _Handle.poll(): drain the HW ring, build `consumer`'s frame from the retained
        # samples newer than its cursor (sliced to its metrics) and advance its cursor, then GC.
        # Returns None if sampling is off, `consumer` isn't registered, or it has nothing new.
        with self._lock:
            if self._col is None or consumer not in self._consumers:
                return None
            self._pull_hw()
            frame: dict[str, Any] | None = None
            ts = self._retained_ts
            mask = ts > consumer.cursor if ts.size else None
            if mask is not None and mask.any():
                dts = ts[mask]
                dvals = self._retained_vals[mask]
                col_index = {m: j for j, m in enumerate(self._metric_names)}
                frame = {
                    "start_ns": dts,
                    "device_id": np.full(len(dts), self._device, dtype=np.int64),
                }
                for m in consumer.metrics:
                    frame[m] = dvals[:, col_index[m]]
                consumer.cursor = int(dts.max())
            self._gc()
            return frame

    def _start(self) -> None:
        if self._col is not None or not self._metric_names:
            return
        self._warn_unsupported()
        try:
            col = _pm_sampling.Collector(device_index=self._device)
            col.enable()
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            return
        self._col = col
        with type(self)._instances_lock:
            type(self)._active += 1
        try:
            col.configure(
                metrics=self._metric_names,
                hardware_buffer_size=_counter_data_size(
                    col._pm_sampling_object, self._metric_names, self._max_samples
                ),
                sampling_interval=self._sampling_interval_ns,
                trigger_mode=_pm_sampling.TriggerMode.GPU_TIME_INTERVAL,
                hw_buffer_append_mode=_pm_sampling.HardwareBuffer_AppendMode.KEEP_LATEST,
            )
            col.start()
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            self._teardown()

    def _teardown(self) -> None:
        col, self._col = self._col, None
        self._metric_names = []
        self._retained_ts = np.empty(0, dtype=np.int64)
        self._retained_vals = np.empty((0, 0), dtype=np.float64)
        if col is None:
            return
        try:
            col.stop()
        except Exception:
            logger.exception("PM sampling stop error")
        try:
            col._finalizer.detach()
            _pm_sampling_disable(col._pm_sampling_object)
        except Exception:
            logger.exception("PM sampling disable error")
        with type(self)._instances_lock:
            type(self)._active -= 1
            last = type(self)._active == 0
        if last:
            try:
                _profiler_deinitialize()
            except Exception:
                logger.exception("PM sampling profiler deinit error")

    def _warn_unsupported(self) -> None:
        try:
            supported = supported_metrics()
        except Exception:
            return
        if not supported:
            return
        known = {s.split(".", 1)[0] for s in supported}
        unknown = [m for m in self._metric_names if m.split(".", 1)[0] not in known]
        if unknown:
            logger.warning(
                "PM sampling: metric(s) not reported by this chip, may fail to enable: %s",
                ", ".join(unknown),
            )

    def _check_single_pass(self, metrics: list[str]) -> None:
        try:
            host = _profiler_host.ProfilerHost(
                _device_chip_name(self._device), _cupti.ProfilerType.PM_SAMPLING
            )
            host.initialize()
            try:
                passes = _profiler_host.get_num_of_passes(
                    host.create_config_image(metrics=metrics)
                )
            finally:
                host.deinitialize()
        except Exception:
            return  # can't determine passes here; configure() still enforces single-pass
        if passes > 1:
            raise ValueError(
                f"PM sampling requires all metrics in a single pass, but the requested set needs "
                f"{passes} passes: {metrics}. Reduce it to a single-pass set (see "
                f"supported_metrics())."
            )

    def _pull_hw(self) -> None:
        col = self._col
        if col is None:
            return
        try:
            cd = col.decode(max_samples=self._max_samples)
        except MemoryError:
            logger.warning(
                "PM sampling HW buffer overflow during decode; samples dropped."
            )
            return
        except Exception:
            logger.exception("PM sampling decode error")
            return
        n = cd.num_completed_samples
        if not n:
            return
        if n >= self._max_samples:
            logger.warning(
                "PM sampling decoded the maximum %d samples; some were dropped.",
                self._max_samples,
            )
        # start_timestamp is a C uint64; read it unsigned so a garbage sample
        # carrying a full-uint64 value can't overflow the int64 assignment.
        raw_ts = np.empty(n, dtype=np.uint64)
        vals = np.empty((n, len(self._metric_names)), dtype=np.float64)
        for i, s in enumerate(cd):
            raw_ts[i] = s.start_timestamp
            vals[i] = s.metric_values
        # Drop an unset interval-start (0), a stale small value from a different
        # clock domain (observed ~7.5e13 vs the real ~1.78e18), or a garbage
        # sample whose timestamp overflows int64. Real samples fall within the
        # look-back of the newest.
        keep = (raw_ts > 0) & (raw_ts <= np.uint64(np.iinfo(np.int64).max))
        if keep.any():
            newest = int(raw_ts[keep].max())
            keep &= raw_ts >= np.uint64(max(0, newest - self._lookback_window_ns))
        if not keep.any():
            return
        ts = raw_ts[keep].astype(np.int64)
        vals = vals[keep]
        if self._retained_ts.size:
            self._retained_ts = np.concatenate([self._retained_ts, ts])
            self._retained_vals = np.concatenate([self._retained_vals, vals])
        else:
            self._retained_ts = ts
            self._retained_vals = vals

    def _gc(self) -> None:
        # Drop retained samples every consumer has passed (<= the min cursor) or that have aged past
        # the look-back window (so a consumer that stops polling can't pin the buffer). A lone,
        # caught-up consumer leaves nothing retained. Caller holds _lock.
        ts = self._retained_ts
        if not ts.size:
            return
        newest = int(ts.max())
        watermark = max(
            min((c.cursor for c in self._consumers), default=newest),
            newest - self._lookback_window_ns,
        )
        keep = ts > watermark
        if keep.all():
            return
        self._retained_ts = ts[keep]
        self._retained_vals = self._retained_vals[keep]
<|endoftext|># mypy: allow-untyped-defs
"""CUPTI user-defined-record (v2) field schema.

In the v2 / user-defined-record path, observers select specific *fields* per activity
kind (rather than whole records). The available fields per kind are the generated
:mod:`_cupti_stubs` catalogs (``Kernel``, ``Memcpy``, ...), one
:class:`~_records_base.Field` per ``CUpti_Activity*FieldIds`` id, each carrying its
:class:`~_records_base.Ctype` for decode. Those catalogs are generated from the CUPTI
ABI (cupti_activity.h) at build time -- cupti-python does not expose the enums.

This module *curates* which of those fields the monitor selects per kind (:data:`FIELDS`)
and derives the lookups the monitor/observers need. The selection is the editorial part
that can't be generated: it bounds record size and per-buffer decode cost.

The monitor does NOT compute record byte layouts: it requires libcupti >= 13.3, which
reports each kind's packed record layout (field offsets/sizes, record size) via
``pBufferCompleteInfo->ppRecordLayouts``. The native layer parses that and attaches it to
each completed buffer; the monitor decodes a buffer against that captured layout, using
:data:`FIELD_CTYPE` only to interpret each field's bytes (signed/unsigned/float/str).
"""

from __future__ import annotations

from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

from torch.profiler._cupti._cupti_stubs import (
    Api,
    CudaEvent,
    Environment,
    ExternalCorrelation,
    GraphHostNode,
    Kernel,
    Memcpy,
    Memcpy2,
    Memset,
    Overhead,
    Synchronization,
)
from torch.profiler._cupti._records_base import Ctype, Field


# Short alias kept for the SYNCHRONIZATION catalog (the generated name is Synchronization).
Sync = Synchronization


__all__ = [
    "Ctype",
    "Field",
    "Api",
    "CudaEvent",
    "Environment",
    "ExternalCorrelation",
    "GraphHostNode",
    "Kernel",
    "Memcpy",
    "Memcpy2",
    "Memset",
    "Overhead",
    "Sync",
    "Synchronization",
    "FIELDS",
    "FIELD_REGISTRY",
    "FIELD_CTYPE",
    "STRING_FIELDS",
    "CORRELATION_FIELD",
    "GRAPH_NODE_FIELD",
    "RecordLayouts",
]


def _all_fields(catalog: type) -> tuple[Field, ...]:
    """Every :class:`Field` on a generated catalog class, in id order. ``sorted`` puts KIND
    (id 0) first, which CUPTI requires (``*_FIELD_KIND`` must lead at enable). Selecting all
    of them and letting the decoder drop fields whose captured size is not 1/2/4/8 yields
    all *decodable* fields, with no hand-maintained per-field list to drift from the header."""
    return tuple(sorted(v for v in vars(catalog).values() if isinstance(v, Field)))


# The only curation: which activity kinds the monitor supports and their generated catalog.
# RUNTIME and DRIVER share the Api catalog. FIELDS / CORRELATION_FIELD / GRAPH_NODE_FIELD all
# derive from this, so adding a kind here is the single edit needed.
_CATALOGS: dict[int, type] = {
    ActivityKind.CONCURRENT_KERNEL: Kernel,
    ActivityKind.MEMCPY: Memcpy,
    ActivityKind.MEMCPY2: Memcpy2,
    ActivityKind.MEMSET: Memset,
    ActivityKind.GRAPH_HOST_NODE: GraphHostNode,
    ActivityKind.RUNTIME: Api,
    ActivityKind.DRIVER: Api,
    ActivityKind.EXTERNAL_CORRELATION: ExternalCorrelation,
    ActivityKind.OVERHEAD: Overhead,
    ActivityKind.CUDA_EVENT: CudaEvent,
    ActivityKind.SYNCHRONIZATION: Synchronization,
    ActivityKind.ENVIRONMENT: Environment,
}

# kind -> the fields the v2 monitor selects: all decodable fields of the generated catalog
# (struct/opaque fields are dropped at decode by their size).
FIELDS: dict[int, tuple[Field, ...]] = {
    kind: _all_fields(cat) for kind, cat in _CATALOGS.items()
}

# kind -> frozenset of supported field ids; source of truth for validating observer
# requests and resolving "all". (Field is an int, so a Field is its id here.)
FIELD_REGISTRY: dict[int, frozenset[int]] = {
    kind: frozenset(fields) for kind, fields in FIELDS.items()
}

# kind -> {field id: Ctype}; how the decoder interprets each selected field's bytes
# (width comes from CUPTI's captured layout, not from here).
FIELD_CTYPE: dict[int, dict[int, Ctype]] = {
    kind: {f.id: f.ctype for f in fields} for kind, fields in FIELDS.items()
}

# kind -> frozenset of field ids that are const char* strings (dereferenced during decode).
STRING_FIELDS: dict[int, frozenset[int]] = {
    kind: frozenset(f.id for f in fields if f.string) for kind, fields in FIELDS.items()
}

# kind -> its CORRELATION_ID field id. The launch correlation id a kernel shares with its
# runtime call; used to join activity to external-correlation (eager annotation). Only kinds
# whose catalog carries the field are included.
CORRELATION_FIELD: dict[int, int] = {
    kind: cat.CORRELATION_ID.id
    for kind, cat in _CATALOGS.items()
    if hasattr(cat, "CORRELATION_ID")
}


# Per-kind graph-node-id field, for the graph annotation resolver: only the GPU-op kinds
# carry a graph_node_id (the field the resolver maps to a region name).
GRAPH_NODE_FIELD: dict[int, int] = {
    kind: cat.GRAPH_NODE_ID.id
    for kind, cat in _CATALOGS.items()
    if hasattr(cat, "GRAPH_NODE_ID")
}


# A record layout as captured by CUPTI (pBufferCompleteInfo->ppRecordLayouts) and attached
# to a completed buffer by the native layer: a list of
# (kind, record_size, [(field_id, offset, size), ...]). This is what the monitor decodes
# against -- no spec/computed layout.
RecordLayouts = list[tuple[int, int, list[tuple[int, int, int]]]]
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import ctypes
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast, NamedTuple, TYPE_CHECKING

import numpy as np
from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

import torch

from . import cupti_python
from .records import (
    Ctype,
    CudaEvent,
    FIELD_CTYPE,
    FIELD_REGISTRY,
    Kernel,
    STRING_FIELDS,
    Sync,
)


# A registration request: either a plain iterable of activity kinds (meaning "all
# fields"), or a field map {kind: iterable of field ids | "all"} selecting specific
# fields per kind. The monitor demuxes the selected fields to columns.
ActivitiesSpec = Mapping[ActivityKind, "Iterable[int] | str"] | Iterable[ActivityKind]


_PY_PROFILER = torch._C._profiler
# The native CUPTI buffer-pool / layout-capture module (C++ side of the monitor).
_cupti_monitor_native = _PY_PROFILER._cupti_monitor

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)

# Buffers are a recycling pool bounded by peak concurrent demand, so the count
# only keeps climbing if the worker can't drain completed buffers as fast as
# CUPTI fills them. Warn once past this many outstanding buffers (1GB at the
# default 4MB size) as a sign of that backpressure.
_OUTSTANDING_WARN_THRESHOLD = 256

# flush(sync=True) fences at a SYNC point: it enables SYNCHRONIZATION, captures
# CUPTI's clock, device-syncs (which produces a SYNCHRONIZATION record at a
# timestamp past that point), waits until the native decoder reports a sync record
# that recent, then disables SYNCHRONIZATION again. CUPTI delivers buffers in fill
# order, so seeing the sync record means everything before it is delivered too. A
# device sync -- unlike a tracer kernel -- adds no kernel, no cudaLaunchKernel, and
# no dispatcher op to the trace; and enabling SYNCHRONIZATION only for the fence
# means the session doesn't record every sync between flushes. KIND + END are the
# fields the fence decodes.
_FENCE_KIND = ActivityKind.SYNCHRONIZATION
_FENCE_END_FIELD = Sync.END.id
_FENCE_FIELDS = frozenset({Sync.KIND.id, _FENCE_END_FIELD})


def _has_active_cuda_context() -> bool:
    try:
        from cuda.bindings import (  # pyrefly: ignore[missing-import]
            driver as cuda_driver,
        )
    except ImportError:
        return False
    rc, ctx = cuda_driver.cuCtxGetCurrent()
    if rc == cuda_driver.CUresult.CUDA_SUCCESS:
        return ctx is not None
    if rc == cuda_driver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        return False
    raise RuntimeError(f"cuCtxGetCurrent failed with rc={rc}")


def _cuda_version_string() -> str:
    return torch.version.cuda or ""


def _deref_cstr(ptr: int) -> str:
    if not ptr:
        return ""
    value = ctypes.cast(ptr, ctypes.c_char_p).value
    return value.decode(errors="replace") if value is not None else ""


# A CUPTI subscriber callback, identified by its (domain, cbid) pair, and the handlers
# registered for one. See CuptiMonitor.register_callback_handler / _dispatch_callback.
_CallbackKey = tuple[int, int]
_CallbackFns = tuple[Callable[..., None], ...]


class _CallbackHandler(NamedTuple):
    """A registered consumer of one CUPTI *subscriber callback*: the ``(domain, cbid)`` it
    wants and the ``fn(domain, cbid, cbdata)`` to invoke.

    Subscriber callbacks are a separate axis from activity records: they fire synchronously
    on the application thread that made the CUDA call, not on the decode worker, and carry a
    raw ``CUpti_CallbackData`` pointer rather than decoded columns. Returned by
    ``CuptiMonitor.register_callback_handler``; hand it back to unregister, mirroring
    ``register`` / ``unregister`` for activity observers."""

    domain: int
    cbid: int
    fn: Callable[..., None]


class _Observer:
    """A registered consumer of the monitor's records: the activity kinds it
    requested, its per-kind field selection (``{kind: frozenset(field_ids)}``), and
    its ``callback(columns)`` -- invoked once per drain (at flush time) with the
    demuxed columns sliced to its selection (see ``CuptiMonitor.register``)."""

    def __init__(
        self,
        activities: Iterable[ActivityKind],
        fields: Mapping[ActivityKind, frozenset[int]],
        callback: Callable[..., None],
    ) -> None:
        self.activities: frozenset[ActivityKind] = frozenset(activities)
        # activity -> the set of field ids wanted for it (the columns to demux).
        self.fields: dict[ActivityKind, frozenset[int]] = dict(fields)
        self.callback = callback


class _SynchronizedClock:
    """Maps CUPTI record timestamps to unix-epoch ns on kineto's axis.

    CUPTI stamps records with cuptiGetTimestamp, which is CLOCK_REALTIME -- the very clock
    kineto's cpu_ops land on (kineto reads the approx clock for cpu_ops and converts it to
    unix == CLOCK_REALTIME at serialization). So a record's timestamp already IS its unix
    time: it is passed through unchanged (identity, offset 0). Records and cpu_ops then share
    the identical realtime clock, so a runtime call's start is its true realtime -- always >=
    the start of the cpu_op that issued it, with no clock estimate in the loop. calibrate()
    requires this and verifies it by bracketing one native read between two
    clock_gettime(CLOCK_REALTIME) reads; a host whose CUPTI clock is something else raises.

    The exception is the timestamp callback: when it engages CUPTI stamps records with the
    profiler's approx clock instead -- kineto's *source* clock -- so records arrive as approx
    ticks and are mapped to unix via the same _ApproximateClockToUnixTimeConverter kineto uses
    (a single-slope line -- median rate over 1001 samples -- so two evaluations recover its
    slope exactly; recovered once so the hot path never calls the converter per record). That
    path lifts the CLOCK_REALTIME requirement and is the ONLY one that touches the converter.

    calibrate() reads the native clock through an injected callable, so the conversion math
    runs (and is tested) without a live CUPTI session.
    """

    def __init__(self) -> None:
        self._callback_active = False
        self._native_is_realtime = False
        self._native_now: Callable[[], int] = lambda: 0
        # Session-start anchor. 0 until calibrated -> conversion is identity.
        self._native_ns = 0
        self._unix_ns = 0
        self._approx_ns = 0
        self._scale = 1.0

    def calibrate(
        self,
        *,
        callback_active: bool,
        native_now: Callable[[], int],
        converter: Any = None,
    ) -> None:
        self._callback_active = callback_active
        self._native_now = native_now
        # cuptiGetTimestamp is CLOCK_REALTIME iff it lands inside a clock_gettime bracket.
        rt_lo = time.clock_gettime_ns(time.CLOCK_REALTIME)
        cupti_now = native_now()
        rt_hi = time.clock_gettime_ns(time.CLOCK_REALTIME)
        self._native_is_realtime = cupti_now != 0 and rt_lo <= cupti_now <= rt_hi
        if callback_active:
            # Records are approx-clock ticks. Build the converter first (it only maps approx
            # reads taken after construction), then bracket the native read -- needed for PM
            # frames, which stay on the native clock -- between two approx reads and pair at
            # the midpoint; recover the converter's slope for vectorized column conversion.
            if converter is None:
                converter = _PY_PROFILER._ApproximateClockToUnixTimeConverter()
            approx_before = _PY_PROFILER._get_approximate_time()
            self._native_ns = (
                time.clock_gettime_ns(time.CLOCK_REALTIME)
                if self._native_is_realtime
                else native_now()
            )
            approx_after = _PY_PROFILER._get_approximate_time()
            self._approx_ns = (approx_before + approx_after) // 2
            self._unix_ns = converter.to_unix_ns(self._approx_ns)
            span = 10**12
            self._scale = (
                converter.to_unix_ns(self._approx_ns + span) - self._unix_ns
            ) / span
        else:
            if not self._native_is_realtime:
                raise RuntimeError(
                    "cupti_monitor requires cuptiGetTimestamp to be CLOCK_REALTIME so "
                    "record timestamps share kineto's unix clock; this host reports a "
                    "different clock source, unsupported until the CUPTI timestamp "
                    "callback is available"
                )
            # cuptiGetTimestamp == CLOCK_REALTIME == unix: records already carry unix time,
            # so pass them through (identity). No approx clock or converter needed.
            self._native_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            self._approx_ns = 0
            self._unix_ns = self._native_ns
            self._scale = 1.0

    # A timestamp is in one of two source domains -- approx-clock ticks or CLOCK_REALTIME/unix --
    # and both map to unix; these are the two conversion primitives (0, and the uncalibrated
    # state, always map to itself). Which domain a record is in depends on the timestamp callback,
    # so the record dispatch lives in CuptiMonitor.convert_time[_array], not here.

    def convert_approx(self, value: int) -> int:
        # Approx-clock tick -> unix ns via the recovered converter slope.
        if value == 0 or self._native_ns == 0:
            return value
        return self._unix_ns + int((value - self._approx_ns) * self._scale)

    def convert_unix(self, value: int) -> int:
        # Native cuptiGetTimestamp clock (CLOCK_REALTIME) -> unix ns: a constant offset.
        if value == 0 or self._native_ns == 0:
            return value
        return value - self._native_ns + self._unix_ns

    def convert_approx_array(self, values: np.ndarray) -> np.ndarray:
        # Vectorized convert_approx. Keep the delta in float (small magnitude) and add the unix
        # anchor as int64 -- adding at the ~1e18 unix magnitude in float64 would lose ~us.
        out = values.astype(np.int64)
        if self._native_ns == 0:
            return out
        ticks = (out - self._approx_ns).astype(np.float64)
        delta = (ticks * self._scale).astype(np.int64)
        return np.where(out == 0, out, self._unix_ns + delta)

    def convert_unix_array(self, values: np.ndarray) -> np.ndarray:
        # Vectorized convert_unix.
        out = values.astype(np.int64)
        if self._native_ns == 0:
            return out
        offset = self._unix_ns - self._native_ns
        return np.where(out == 0, out, out + offset)

    def now_record_ns(self) -> int:
        # Current record-clock value: the approx clock when the callback is active, else the
        # native cuptiGetTimestamp clock (read cheaply via clock_gettime when it is realtime).
        # 0 before calibration.
        if self._native_ns == 0:
            return 0
        if self._callback_active:
            return _PY_PROFILER._get_approximate_time()
        if self._native_is_realtime:
            return time.clock_gettime_ns(time.CLOCK_REALTIME)
        return self._native_now()

    def reset(self) -> None:
        self._native_ns = 0
        self._native_now = lambda: 0

    @property
    def unix_anchor_ns(self) -> int:
        return self._unix_ns

    @property
    def approx_anchor_ns(self) -> int:
        return self._approx_ns


class CuptiMonitor:
    """Process-wide CUPTI monitor / multiplexer singleton. Like PmSampler, ``CuptiMonitor()``
    returns the one instance (constructed on first call); its settings are snapshotted from
    the class config -- set via ``cupti_monitor.configure()`` before first use -- not from
    constructor args."""

    _instance: CuptiMonitor | None = None
    _instance_lock = threading.Lock()
    # Process-wide settings the singleton snapshots when first constructed; set via
    # configure() (first-come-first-serve, no env var), defaults otherwise. Both cadences
    # default to -1 (no background flush / drain -- the caller drives flush()).
    #
    # use_approx_timestamps defaults OFF; when on, the per-subscriber timestamp callback
    # re-times HOST and DEVICE records onto the profiler's approx clock (kineto's timebase).
    # Opt-in via configure(use_approx_timestamps=True).
    _buffer_size: int = 4 * 1024 * 1024
    _background_flush_period_s: float = -1.0
    _background_drain_period_s: float = -1.0
    _use_approx_timestamps: bool = False
    _configured: bool = False

    def __new__(cls) -> Self:
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
                # the live monitor snapshotted the config; lock it
                cls._configured = True
            return cls._instance

    def _init(self) -> None:
        cls = type(self)
        # The monitor is the engine and the multiplexer: it owns the single CUPTI
        # subscription + buffer pool + native decode worker, which demuxes each
        # completed buffer into columns; the monitor drains those columns at flush
        # time and hands every observer the columns it selected. It reaches CUPTI
        # only through the self._cupti.activity_* wrappers -- no ctypes here.
        #
        # It uses CUPTI's v2 user-defined-record API: a subscriber + per-field
        # selection, decoded columnar against a record layout computed from the
        # field-size spec (no captured layout needed). This requires libcupti >= 13.2.
        #
        # Per-buffer pool size (bytes), default 4 MiB. Bigger buffers complete less often
        # (fewer worker wakeups, lower overhead) at the cost of more pinned host memory and
        # coarser delivery. Set process-wide via cupti_monitor.configure().
        self.buffer_size = cls._buffer_size
        # Two independent cadences (seconds); they control different things, so they are
        # separate knobs.
        #
        # background_flush_period_s (default -1) -- how often the native decode thread
        # self-drives cuptiActivityFlushAll, handing CUPTI's completed buffers to the
        # decoder. Sign-encoded:
        #   >= 0 -> the decode thread self-flushes on this cadence (0 = continuously).
        #   <  0 -> NO self-flush; the caller must drive flush() itself (e.g. at end of
        #           step). This is the default, and the escape hatch for a libcupti/libnvperf
        #           HES thread-safety bug: cuptiActivityFlushAll drives CUPTI's HW-trace
        #           processing against live collection state and can wild-write the host
        #           heap when it overlaps concurrent host activity (e.g. NCCL collective
        #           setup). The racy op is the flush, NOT the decode -- the decoder keeps
        #           decoding delivered buffers off-thread regardless. Driving flush() only
        #           from the quiescent foreground avoids the race.
        #
        # background_drain_period_s (default -1) -- how often the Python thread drains the
        # decoded columns and dispatches them to observers (GIL work). Same sign-encoding:
        #   >= 0 -> a background thread drains on this cadence (0 = continuously).
        #   <  0 -> NO background drain thread; the caller drives drain via flush().
        self.background_flush_period_s = cls._background_flush_period_s
        self.background_drain_period_s = cls._background_drain_period_s
        self._cupti = cupti_python.pylibcupti()
        # The CUPTI subscriber handle.
        self._subscriber: int | None = None
        self._latency_enabled = False
        self._device_ts_enabled = False
        # Layout state -- a function of registration, recomputed only when the
        # The fields enabled per kind on the subscriber (a function of the observer
        # field union, recomputed only on register/deregister, never per buffer). The
        # record byte layout is NOT tracked here -- each completed buffer carries
        # CUPTI's own captured layout (ppRecordLayouts) that records.decode reads.
        self._enabled: dict[int, frozenset[int]] = {}

        self._lock = threading.Lock()
        self._started = False
        self._callbacks_registered = False
        # --- subscriber callbacks: a second axis, independent of activity records ---
        # Handlers keyed by (domain, cbid). _dispatch_callback runs on the application
        # thread inside the CUDA call, so it must not take self._lock (register and
        # _apply_selection hold that): it reads the immutable tuple values below, which
        # mutation replaces wholesale under _callback_lock.
        self._callback_handlers: dict[_CallbackKey, _CallbackFns] = {}
        # Per-(domain, cbid) arm refcount, so consumers wanting the same callback over
        # different windows don't disable each other. Enabled on 0->1, disabled on 1->0.
        self._callback_armed: dict[_CallbackKey, int] = {}
        self._callback_lock = threading.Lock()
        # Our single CUpti_CallbackFunc trampoline; must outlive the subscription.
        self._callback_trampoline: Any = None
        self._drain_stop = threading.Event()
        self._drain_and_dispatch_thread: threading.Thread | None = None
        # Serializes _drain_and_dispatch: the native decoder accumulates columns
        # GIL-free; Python drains them here. Only ever one driver at a time (the
        # foreground caller OR the background flush loop, never both), but the lock
        # keeps a stray concurrent drain from interleaving dispatch.
        self._drain_lock = threading.Lock()
        self._observers: list[_Observer] = []
        # {external_id: metadata blob} drained alongside the decoded records (see
        # drain_decoded). Accumulated here until an observer's external-correlation
        # join consumes it via take_external_metadata(); the blob is attached onto
        # the kernel event keyed by the same external id. Guarded by _lock.
        self._external_metadata: dict[int, str] = {}
        self._next_external_id = 1
        # All subsystems push external-correlation ids on ONE CUPTI kind, so CUPTI
        # inserts a single EXTERNAL_CORRELATION record (that kind's stack top) per op
        # -- i.e. it tags a kernel with only the *innermost* active id; the enclosing
        # ids are recovered from our own bookkeeping here. _id_chains[id] is the full
        # active stack (outermost..id) captured when `id` was pushed, so a consumer
        # maps a kernel's innermost id to every context active for that op (see
        # external_id_chain) and picks the one it owns (by membership in its own
        # name/metadata map). The live LIFO is per-thread (CUPTI's external-correlation
        # stacks are per-thread) in _push_tls.stack. A popped id's chain is kept a
        # couple of dispatch cycles -- its activity records arrive after the pop -- then
        # dropped via the _chains_gc_* generations. _id_chains/_chains_gc_* are guarded
        # by _lock; the per-thread stack needs no lock.
        self._push_tls = threading.local()
        self._id_chains: dict[int, tuple[int, ...]] = {}
        self._chains_gc_pending: list[int] = []
        self._chains_gc_ready: list[int] = []
        # Record-timestamp -> unix conversion lives in the clock; the monitor delegates
        # convert_time / now_record_ns to it and calibrates it in start(). Records normally
        # arrive on the native (realtime) clock; configure(use_approx_timestamps=True) puts
        # them directly on the approx clock via CUPTI's per-subscriber timestamp callback
        # (opt-in, sole-subscriber only).
        self._timestamp_callback_enabled = cls._use_approx_timestamps
        self._clock = _SynchronizedClock()
        self._timestamp_callback_active = False

        # Snapshot of the native pool size taken before stop() frees it, so
        # stats() stays meaningful after the monitor has been stopped.
        self._final_allocated_buffers = 0
        self._outstanding_warned = False
        self._dropped_records = 0

        # Opt-in PM sampling (true SM-active % / DRAM-throughput % counters): the monitor registers
        # each requesting observer as a consumer of the current device's per-device PmSampler
        # (only one PM session per device is possible), polls it on the flush cadence, and converts
        # each polled frame's raw CUPTI-clock timestamps into the trace clock before pushing to the
        # observer's sink (the sampler is clock-agnostic; conversion lives here, where the clock base
        # does). Each consumer brings its own metrics; the shared session samples their union, starts
        # on the first consumer, and disables after the last. self._pm_consumers maps an observer's
        # sink to its sampler handle so poll/release can address it.
        self._pm_consumers: dict[Callable[[dict[str, Any]], None], Any] = {}
        self._pm_sampler: Any = None
        # Serializes PM add/poll/remove so a flush-thread poll never decodes the collector while the
        # foreground is tearing it down (concurrent decode on one collector is unsafe).
        self._pm_lock = threading.Lock()

    def _ensure_subscribed(self) -> int:
        """Take the process's one CUPTI subscription if we don't already hold it, and return
        the handle.

        Shared by the activity path (:meth:`register_callbacks`) and the subscriber-callback
        path, which needs the subscription but none of the user-defined-record machinery --
        so a callback-only consumer pays for neither UDR mode nor the decode worker.
        """
        if self._subscriber is not None:
            return self._subscriber
        version = self._cupti.get_version()
        if version < cupti_python.LIBCUPTI_MIN_VERSION:
            raise RuntimeError(
                "CuptiMonitor requires libcupti >= "
                f"{cupti_python.LIBCUPTI_MIN_VERSION}; loaded "
                f"{cupti_python.LIBCUPTI_SONAME} reports {version}"
            )
        # CUPTI allows exactly one callback per subscriber, so we always install our own
        # switchboard and fan out from there. Passing it explicitly -- rather than letting
        # subscribe() fall back to the module-level no-op -- keeps a consumer that swaps that
        # global from clobbering our dispatch.
        self._callback_trampoline = cupti_python._CB_FUNC(self._dispatch_callback)
        # (A prior consumer that left CUPTI attached -- e.g. Kineto -- can make
        # cuptiSubscribe_v2 fail with CUPTI_ERROR_MULTIPLE_SUBSCRIBERS; run such consumers
        # with TEARDOWN_CUPTI=1 so they release CUPTI on teardown rather than us finalizing
        # global state.) Subscribe solo only when the timestamp callback is opted in: CUPTI
        # honors it only while multiple subscribers are NOT allowed. Otherwise allow
        # coexistence (default).
        try:
            self._subscriber = self._cupti.subscribe(
                allow_multiple=not self._timestamp_callback_enabled,
                callback=self._callback_trampoline,
            )
        except cupti_python.CuptiError as e:
            self._callback_trampoline = None
            if self._timestamp_callback_enabled:
                # We requested sole-subscriber mode only because the approx-clock timestamp
                # callback needs it; another CUPTI consumer is likely attached. Point the user
                # at the opt-out so they can coexist (the callback is then off).
                raise RuntimeError(
                    "cupti_monitor could not subscribe as the sole subscriber, which the "
                    "opt-in approx-clock timestamp callback requires (another CUPTI consumer "
                    "is likely attached). Retry with use_approx_timestamps=False to allow "
                    f"coexisting subscribers: {e}"
                ) from e
            raise
        # Arm the per-subscriber approx-clock timestamp callback right after subscribe, before
        # any user-defined record can be produced. It stays armed for the life of the
        # subscription (see _release_subscriber_if_idle), not just the activity session.
        self._timestamp_callback_active = self._try_arm_approx_timestamp_callback(
            self._subscriber
        )
        return self._subscriber

    def register_callbacks(self) -> None:
        if self._callbacks_registered:
            return
        native = _cupti_monitor_native
        request_addr = native.buffer_request_callback_address()
        complete_addr = native.buffer_complete_callback_address()
        # The activity API is subscription-scoped: subscribe, turn on user-defined
        # records, and register the v2 buffer callbacks.
        subscriber = self._ensure_subscribed()
        self._cupti.arm_user_defined_records(subscriber, request_addr, complete_addr)
        self._callbacks_registered = True

    # --- subscriber callbacks ----------------------------------------------

    def _dispatch_callback(
        self, _userdata: int, domain: int, cbid: int, cbdata: int
    ) -> None:
        """The subscription's single ``CUpti_CallbackFunc``: fan out to the handlers
        registered for ``(domain, cbid)``.

        Runs synchronously on the application thread that made the CUDA call, so it takes no
        lock -- registration swaps in a fresh immutable tuple rather than mutating one in
        place -- and lets no exception escape: a raise would propagate into CUPTI's C
        dispatch, and one failing handler must not silence the others.
        """
        for fn in self._callback_handlers.get((domain, cbid), ()):
            try:
                fn(domain, cbid, cbdata)
            except Exception:
                logger.exception(
                    "CUPTI callback handler failed (domain=%d cbid=%d)", domain, cbid
                )

    def register_callback_handler(
        self, domain: int, cbid: int, fn: Callable[..., None]
    ) -> _CallbackHandler:
        """Register ``fn(domain, cbid, cbdata)`` for one CUPTI ``(domain, cbid)`` subscriber
        callback; returns a token for :meth:`unregister_callback_handler`.

        Registering brings the shared subscription up and holds it, exactly as registering an
        activity observer does, so a callback-only consumer can keep CUPTI open with no
        observers present. It does NOT enable the callback -- call :meth:`arm_callback` for
        that, so a consumer can scope delivery to a window (a CUDA-graph capture, say) while
        staying registered across it.

        ``fn`` runs on the application thread inside the CUDA call, so it must be cheap and
        non-blocking; ``cbdata`` is the raw ``CUpti_CallbackData`` pointer as an int.
        Exceptions are logged and swallowed.
        """
        handler = _CallbackHandler(domain, cbid, fn)
        self._ensure_subscribed()
        with self._callback_lock:
            key = (domain, cbid)
            self._callback_handlers[key] = self._callback_handlers.get(key, ()) + (fn,)
        return handler

    def unregister_callback_handler(self, handler: _CallbackHandler) -> None:
        """Remove a handler from :meth:`register_callback_handler`. Idempotent.

        When the last handler for a ``(domain, cbid)`` goes away the callback is disabled
        outright, whatever its arm refcount was -- delivering to nobody would cost the
        application thread for nothing. Releases the shared subscription once no handlers and
        no activity observers remain.
        """
        key = (handler.domain, handler.cbid)
        with self._callback_lock:
            remaining = tuple(
                f for f in self._callback_handlers.get(key, ()) if f is not handler.fn
            )
            if remaining:
                self._callback_handlers[key] = remaining
            elif self._callback_handlers.pop(key, None) is not None:
                self._callback_armed.pop(key, None)
                if self._subscriber is not None:
                    self._cupti.enable_callback(self._subscriber, *key, False)
        self._release_subscriber_if_idle()

    def arm_callback(self, domain: int, cbid: int) -> None:
        """Enable one ``(domain, cbid)`` callback, refcounted: CUPTI is told to enable on the
        first arm and to disable on the last :meth:`disarm_callback`, so consumers scoping the
        same callback to different windows don't disable each other."""
        subscriber = self._ensure_subscribed()
        with self._callback_lock:
            key = (domain, cbid)
            count = self._callback_armed.get(key, 0)
            self._callback_armed[key] = count + 1
            if count == 0:
                self._cupti.enable_callback(subscriber, domain, cbid, True)

    def disarm_callback(self, domain: int, cbid: int) -> None:
        """Drop one arm refcount for ``(domain, cbid)``, disabling the callback at zero.
        No-op when it is not armed."""
        with self._callback_lock:
            key = (domain, cbid)
            count = self._callback_armed.get(key, 0)
            if count == 0:
                return
            if count > 1:
                self._callback_armed[key] = count - 1
                return
            del self._callback_armed[key]
            if self._subscriber is not None:
                self._cupti.enable_callback(self._subscriber, domain, cbid, False)

    def _release_subscriber_if_idle(self) -> None:
        """Unsubscribe once neither activity observers nor callback handlers need CUPTI.

        The subscription is shared, so it outlives the activity session in both directions:
