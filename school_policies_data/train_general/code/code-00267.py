        Any other input raises an exception
        """

        def to_std(v: str) -> Std:  # type: ignore[return]
            s = Std(int(v))
            if s in Std:
                return s
            # return None -> should NEVER reach here since we regex check input

        if re.match(_VALUE_REGEX, vm):  # vm is a number (e.g. 0)
            return to_std(vm)
        elif re.match(_MAPPING_REGEX, vm):  # vm is a mapping (e.g. 0:1,1:2)
            d: dict[int, Std] = {}
            for m in vm.split(","):
                i, v = m.split(":")
                d[int(i)] = to_std(v)
            return d
        else:
            raise ValueError(
                f"{vm} does not match: <{_VALUE_REGEX}> or <{_MAPPING_REGEX}>"
            )


def to_map(val_or_map: Std | dict[int, Std], local_world_size: int) -> dict[int, Std]:
    """
    Certain APIs take redirect settings either as a single value (e.g. apply to all
    local ranks) or as an explicit user-provided mapping. This method is a convenience
    method that converts a value or mapping into a mapping.

    Example:
    ::

     to_map(Std.OUT, local_world_size=2)  # returns: {0: Std.OUT, 1: Std.OUT}
     to_map({1: Std.OUT}, local_world_size=2)  # returns: {0: Std.NONE, 1: Std.OUT}
     to_map(
         {0: Std.OUT, 1: Std.OUT}, local_world_size=2
     )  # returns: {0: Std.OUT, 1: Std.OUT}
    """
    if isinstance(val_or_map, Std):
        return dict.fromkeys(range(local_world_size), val_or_map)
    else:
        map = {}
        for i in range(local_world_size):
            map[i] = val_or_map.get(i, Std.NONE)
        return map


@dataclass
class LogsDest:
    """
    For each log type, holds mapping of local rank ids to file paths.
    """

    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)
    tee_stdouts: dict[int, str] = field(default_factory=dict)
    tee_stderrs: dict[int, str] = field(default_factory=dict)
    error_files: dict[int, str] = field(default_factory=dict)
    filtered_stdout: str = field(default_factory=str)
    filtered_stderr: str = field(default_factory=str)


class LogsSpecs(ABC):
    """
    Defines logs processing and redirection for each worker process.

    Args:
        log_dir:
            Base directory where logs will be written.
        redirects:
            Streams to redirect to files. Pass a single ``Std``
            enum to redirect for all workers, or a mapping keyed
            by local_rank to selectively redirect.
        tee:
            Streams to duplicate to stdout/stderr.
            Pass a single ``Std`` enum to duplicate streams for all workers,
            or a mapping keyed by local_rank to selectively duplicate.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
        local_ranks_filter: set[int] | None = None,
    ) -> None:
        self._root_log_dir = log_dir
        self._redirects = redirects
        self._tee = tee
        self._local_ranks_filter = local_ranks_filter

    @abstractmethod
    def reify(
        self,
        envs: dict[int, dict[str, str]],
    ) -> LogsDest:
        """
        Given the environment variables, builds destination of log files for each of the local ranks.

        Envs parameter contains env variables dict for each of the local ranks, where entries are defined in:
        :func:`~torchelastic.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent._start_workers`.
        """

    @property
    @abstractmethod
    def root_log_dir(self) -> str:
        pass


class DefaultLogsSpecs(LogsSpecs):
    """
    Default LogsSpecs implementation:

    - `log_dir` will be created if it doesn't exist
    - Generates nested folders for each attempt and rank.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        redirects: Std | dict[int, Std] = Std.NONE,
        tee: Std | dict[int, Std] = Std.NONE,
        local_ranks_filter: set[int] | None = None,
    ) -> None:
        if log_dir != os.devnull:
            if not log_dir:
                log_dir = tempfile.mkdtemp(prefix="torchelastic_")
            elif not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            else:
                if os.path.isfile(log_dir):
                    raise NotADirectoryError(f"log_dir: {log_dir} is a file")
        super().__init__(log_dir, redirects, tee, local_ranks_filter)
        # initialized only once
        self._run_log_dir = None

    @property
    def root_log_dir(self) -> str:
        return str(self._root_log_dir)

    def _make_log_dir(self, log_dir: str | None, rdzv_run_id: str):
        base_log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        os.makedirs(base_log_dir, exist_ok=True)
        dir = tempfile.mkdtemp(prefix=f"{rdzv_run_id}_", dir=base_log_dir)
        logger.info("log directory set to: %s", dir)
        return dir

    def reify(
        self,
        envs: dict[int, dict[str, str]],
    ) -> LogsDest:
        """
        Uses following scheme to build log destination paths:

        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stdout.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/stderr.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/<rank>/error.json`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/filtered_stdout.log`
        - `<log_dir>/<rdzv_run_id>/attempt_<attempt>/filtered_stderr.log`
        """
        nprocs = len(envs)
        global_env = {}  # use only to query properties that are not dependent on a rank
        if nprocs > 0:
            global_env = envs[0]
        else:
            logger.warning(
                "Empty envs map provided when defining logging destinations."
            )
        # Keys are always defined, but values can be missing in unit tests
        run_id = global_env.get("TORCHELASTIC_RUN_ID", "test_run_id")
        restart_count = global_env.get("TORCHELASTIC_RESTART_COUNT", "0")

        attempt_log_dir: str = ""
        if self._root_log_dir != os.devnull:
            if not self._run_log_dir:
                self._run_log_dir = self._make_log_dir(self._root_log_dir, run_id)

            attempt_log_dir = os.path.join(
                self._run_log_dir, f"attempt_{restart_count}"
            )  # type: ignore[call-overload]
            shutil.rmtree(attempt_log_dir, ignore_errors=True)
            os.makedirs(attempt_log_dir)

        if self._root_log_dir == os.devnull:
            attempt_log_dir = os.devnull

        # create subdirs for each local rank in the logs_dir
        # logs_dir
        #       |- 0
        #          |- error.json
        #          |- stdout.log
        #          |- stderr.log
        #       |- ...
        #       |- (nprocs-1)
        redirs = to_map(self._redirects, nprocs)
        ts = to_map(self._tee, nprocs)

        # to tee stdout/stderr we first redirect into a file
        # then tail -f stdout.log/stderr.log so add tee settings to redirects
        for local_rank, tee_std in ts.items():
            redirect_std = redirs[local_rank]
            redirs[local_rank] = redirect_std | tee_std

        SYS_STREAM = ""  # special case to indicate to output to console
        stdouts = dict.fromkeys(range(nprocs), SYS_STREAM)
        stderrs = dict.fromkeys(range(nprocs), SYS_STREAM)
        tee_stdouts: dict[int, str] = {}
        tee_stderrs: dict[int, str] = {}
        error_files = {}

        for local_rank in range(nprocs):
            if attempt_log_dir == os.devnull:
                tee_stdouts[local_rank] = os.devnull
                tee_stderrs[local_rank] = os.devnull
                error_files[local_rank] = os.devnull
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = ""
            else:
                clogdir = os.path.join(attempt_log_dir, str(local_rank))
                os.mkdir(clogdir)

                rd = redirs[local_rank]
                if (rd & Std.OUT) == Std.OUT:
                    stdouts[local_rank] = os.path.join(clogdir, "stdout.log")
                if (rd & Std.ERR) == Std.ERR:
                    stderrs[local_rank] = os.path.join(clogdir, "stderr.log")

                t = ts[local_rank]
                if t & Std.OUT == Std.OUT:
                    tee_stdouts[local_rank] = stdouts[local_rank]
                if t & Std.ERR == Std.ERR:
                    tee_stderrs[local_rank] = stderrs[local_rank]

                if (
                    self._local_ranks_filter
                    and local_rank not in self._local_ranks_filter
                ):
                    # If stream is tee'd, only write to file, but don't tail
                    if local_rank in tee_stdouts:
                        tee_stdouts.pop(local_rank, None)
                    if local_rank in tee_stderrs:
                        tee_stderrs.pop(local_rank, None)

                    # If stream is not redirected, don't print
                    if stdouts[local_rank] == SYS_STREAM:
                        stdouts[local_rank] = os.devnull
                    if stderrs[local_rank] == SYS_STREAM:
                        stderrs[local_rank] = os.devnull

                error_file = os.path.join(clogdir, "error.json")
                # pyrefly: ignore [unsupported-operation]
                error_files[local_rank] = error_file
                logger.info(
                    "Setting worker%s reply file to: %s", local_rank, error_file
                )
                envs[local_rank]["TORCHELASTIC_ERROR_FILE"] = error_file

        return LogsDest(
            stdouts,
            stderrs,
            tee_stdouts,
            tee_stderrs,
            # pyrefly: ignore [bad-argument-type]
            error_files,
            os.path.join(attempt_log_dir, "filtered_stdout.log"),
            os.path.join(attempt_log_dir, "filtered_stderr.log"),
        )

    def __repr__(self) -> str:
        return (
            f"DefaultLogsSpecs(root_log_dir={self._root_log_dir}, redirects={self._redirects}, "
            f"tee={self._tee}, local_ranks_filter={self._local_ranks_filter})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DefaultLogsSpecs):
            return False

        return (
            self._root_log_dir == other._root_log_dir
            and self._redirects == other._redirects
            and self._tee == other._tee
            and self._local_ranks_filter == other._local_ranks_filter
        )


@dataclass
class RunProcsResult:
    """
    Results of a completed run of processes started with ``start_processes()``. Returned by ``PContext``.

    Note the following:

    1. All fields are mapped by local rank
    2. ``return_values`` - only populated for functions (not the binaries).
    3. ``stdouts`` - path to stdout.log (empty string if no redirect)
    4. ``stderrs`` - path to stderr.log (empty string if no redirect)

    """

    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)

    def is_failed(self) -> bool:
        return len(self.failures) > 0


class PContext(abc.ABC):
    """
    The base class that standardizes operations over a set of processes that are launched via different mechanisms.

    The name ``PContext`` is intentional to disambiguate with ``torch.multiprocessing.ProcessContext``.

    .. warning:: stdouts and stderrs should ALWAYS be a superset of
                 tee_stdouts and tee_stderrs (respectively) this is b/c
                 tee is implemented as a redirect + tail -f <stdout/stderr.log>

    Args:
        duplicate_stdout_filters:
            If non-empty, duplicates stdouts specified in ``logs_specs``'s ``tee``
            to a file containing only lines that match _any_ of the filter strings.
            The log file is aggregated across all ranks selected by ``tee``.
        duplicate_stderr_filters:
            If non-empty, duplicates stderrs specified in ``logs_specs``'s ``tee``
            to a file containing only lines that match _any_ of the filter strings.
            The log file is aggregated across all ranks selected by ``tee``.
    """

    def __init__(
        self,
        name: str,
        entrypoint: Callable | str,
        args: dict[int, tuple],
        envs: dict[int, dict[str, str]],
        logs_specs: LogsSpecs,
        log_line_prefixes: dict[int, str] | None = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ):
        self.name = name
        # validate that all mappings have the same number of keys and
        # all local ranks are accounted for
        nprocs = len(args)

        # TODO log_line_prefixes can be expanded too
        logs_dest = logs_specs.reify(envs)

        _validate_full_rank(logs_dest.stdouts, nprocs, "stdouts")
        _validate_full_rank(logs_dest.stderrs, nprocs, "stderrs")

        self.entrypoint = entrypoint
        self.args = args
        self.envs = envs
        self.stdouts = logs_dest.stdouts
        self.stderrs = logs_dest.stderrs
        self.error_files = logs_dest.error_files
        self.nprocs = nprocs
        self.filtered_stdout: TextIO | None = None
        self.filtered_stderr: TextIO | None = None

        self._tail_logs = [
            TailLog(name, logs_dest.tee_stdouts, sys.stdout, log_line_prefixes),
            TailLog(name, logs_dest.tee_stderrs, sys.stderr, log_line_prefixes),
        ]

        if duplicate_stdout_filters:
            self.filtered_stdout = open(  # noqa: SIM115
                logs_dest.filtered_stdout, mode="w", errors="replace", buffering=1
            )
            self._tail_logs.append(
                TailLog(
                    name,
                    logs_dest.tee_stdouts,
                    self.filtered_stdout,
                    log_line_prefixes,
                    log_line_filter=lambda line: any(
                        needle in line for needle in duplicate_stdout_filters
                    ),
                )
            )

        if duplicate_stderr_filters:
            self.filtered_stderr = open(  # noqa: SIM115
                logs_dest.filtered_stderr, mode="w", errors="replace", buffering=1
            )
            self._tail_logs.append(
                TailLog(
                    name,
                    logs_dest.tee_stderrs,
                    self.filtered_stderr,
                    log_line_prefixes,
                    log_line_filter=lambda line: any(
                        needle in line for needle in duplicate_stderr_filters
                    ),
                )
            )

    def start(self) -> None:
        """Start processes using parameters defined in the constructor."""
        if threading.current_thread() is threading.main_thread():
            # Register signal handlers for the signals specified in the environment variable
            signals_to_handle = os.environ.get(
                "TORCHELASTIC_SIGNALS_TO_HANDLE", "SIGTERM,SIGINT,SIGHUP,SIGQUIT"
            )
            signal_list = signals_to_handle.split(",")

            for sig_name in signal_list:
                try:
                    sig = getattr(signal, sig_name.strip())
                    signal.signal(sig, _terminate_process_handler)
                    logger.info("Registered signal handler for %s", sig_name)
                except (AttributeError, ValueError):
                    logger.warning(
                        "Failed to register signal handler for %s",
                        sig_name,
                        exc_info=True,
                    )
                except RuntimeError:
                    if IS_WINDOWS and sig_name.strip() in [
                        "SIGHUP",
                        "SIGQUIT",
                        "SIGUSR1",
                        "SIGUSR2",
                    ]:
                        logger.info(
                            "Signal %s is not supported on Windows, skipping", sig_name
                        )
                    else:
                        logger.warning(
                            "Failed to register signal handler for %s",
                            sig_name,
                            exc_info=True,
                        )
        else:
            logger.warning(
                "Failed to register signal handlers since torchelastic is running on a child thread. "
                "This could lead to orphaned worker processes if the torchrun is terminated."
            )
        self._start()
        for tail_log in self._tail_logs:
            tail_log.start()

    @abc.abstractmethod
    def _start(self) -> None:
        """Start processes using strategy defined in a particular context."""
        raise NotImplementedError

    @abc.abstractmethod
    def _poll(self) -> RunProcsResult | None:
        """
        Poll the run status of the processes running under this context.
        This method follows an "all-or-nothing" policy and returns
        a ``RunProcessResults`` object if either all processes complete
        successfully or any process fails. Returns ``None`` if
        all processes are still running.
        """
        raise NotImplementedError

    def wait(self, timeout: float = -1, period: float = 1) -> RunProcsResult | None:
        """
        Wait for the specified ``timeout`` seconds, polling every ``period`` seconds
        for the processes to be done. Returns ``None`` if the processes are still running
        on timeout expiry. Negative timeout values are interpreted as "wait-forever".
        A timeout value of zero simply queries the status of the processes (e.g. equivalent
        to a poll).

        .. note::
            Multiprocessing library registers SIGTERM and SIGINT signal handlers that raise
            ``SignalException`` when the signals received. It is up to the consumer of the code
            to properly handle the exception. It is important not to swallow the exception otherwise
            the process would not terminate. Example of the typical workflow can be:

        .. code-block:: python
            pc = start_processes(...)
            try:
                pc.wait(1)
                .. do some other work
            except SignalException as e:
                pc.shutdown(e.sigval, timeout=30)

        If SIGTERM or SIGINT occurs, the code above will try to shutdown child processes by propagating
        received signal. If child processes will not terminate in the timeout time, the process will send
        the SIGKILL.
        """
        if timeout == 0:
            return self._poll()

        if timeout < 0:
            timeout = sys.maxsize

        expiry = time.time() + timeout
        while time.time() < expiry:
            pr = self._poll()
            if pr:
                return pr
            time.sleep(period)

        return None

    @abc.abstractmethod
    def pids(self) -> dict[int, int]:
        """Return pids of processes mapped by their respective local_ranks."""
        raise NotImplementedError

    @abc.abstractmethod
    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).
        """
        raise NotImplementedError

    def close(self, death_sig: signal.Signals | None = None, timeout: int = 30) -> None:
        r"""
        Terminates all processes managed by this context and cleans up any
        meta resources (e.g. redirect, error_file files).

        Args:
            death_sig: Death signal to terminate processes.
            timeout: Time to wait for processes to finish, if process is
                still alive after this time, it will be terminated via SIGKILL.
        """
        if not death_sig:
            death_sig = _get_default_signal()
        self._close(death_sig=death_sig, timeout=timeout)
        for tail_log in self._tail_logs:
            tail_log.stop()
        if self.filtered_stdout:
            self.filtered_stdout.close()
        if self.filtered_stderr:
            self.filtered_stderr.close()


def get_std_cm(std_rd: str, redirect_fn):
    if IS_WINDOWS or IS_MACOS or not std_rd:
        return nullcontext()
    else:
        return redirect_fn(std_rd)


def _wrap(
    local_rank: int,
    fn: Callable,
    args: dict[int, tuple],
    envs: dict[int, dict[str, str]],
    stdout_redirects: dict[int, str],  # redirect file for stdout (to console if None)
    stderr_redirects: dict[int, str],  # redirect file for stderr (to console if None)
    ret_vals: dict[int, mp.SimpleQueue],
    queue_finished_reading_event: synchronize.Event,
    numa_options: NumaOptions | None,
) -> None:
    # get the per-rank params up front so we fail fast if no mapping is found
    args_ = args[local_rank]
    env_ = envs[local_rank]
    ret_val_ = ret_vals[local_rank]

    stdout_rd = stdout_redirects[local_rank]
    stderr_rd = stderr_redirects[local_rank]

    stdout_cm = get_std_cm(stdout_rd, redirect_stdout)
    stderr_cm = get_std_cm(stderr_rd, redirect_stderr)

    for k, v in env_.items():
        os.environ[k] = v

    with stdout_cm, stderr_cm:
        fn = _maybe_wrap_with_numa_binding(
            fn, device_index=local_rank, numa_options=numa_options
        )
        ret = record(fn)(*args_)
    ret_val_.put(ret)
    queue_finished_reading_event.wait()


class MultiprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a function."""

    def __init__(
        self,
        name: str,
        entrypoint: Callable,
        args: dict[int, tuple],
        envs: dict[int, dict[str, str]],
        start_method: str,
        logs_specs: LogsSpecs,
        log_line_prefixes: dict[int, str] | None = None,
        numa_options: NumaOptions | None = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
            duplicate_stdout_filters,
            duplicate_stderr_filters,
        )

        self.start_method = start_method
        # each ret_val queue will always contain a single element.
        self._ret_vals = {
            local_rank: mp.get_context(self.start_method).SimpleQueue()
            for local_rank in range(self.nprocs)
        }

        # see comments in ``join()`` for what this is
        self._return_values: dict[int, Any] = {}
        self._pc: mp.ProcessContext | None = None
        # Note: set method should ONLY be invoked for the use case when all processes finished
        # successfully. If any process died on event.wait() calling set() method will deadlock.
        self._worker_finished_event = mp.get_context(self.start_method).Event()

        self._numa_options: NumaOptions | None = numa_options

    def _start(self):
        if self._pc:
            raise ValueError(
                "The process context already initialized."
                " Most likely the start method got called twice."
            )
        self._pc = mp.start_processes(
            fn=_wrap,
            args=(
                self.entrypoint,
                self.args,
                self.envs,
                self.stdouts,
                self.stderrs,
                self._ret_vals,
                self._worker_finished_event,
                self._numa_options,
            ),
            nprocs=self.nprocs,
            join=False,
            daemon=False,
            start_method=self.start_method,
        )

    def _is_done(self) -> bool:
        return len(self._return_values) == self.nprocs

    def _poll(self) -> RunProcsResult | None:
        if self._pc is None:
            raise AssertionError  # assertion for mypy type checker

        try:
            # torch.mp.ProcessContext Throws an Exception if some/all of
            # worker processes failed
            # timeout < 0 checks worker status and return immediately
            # Join will never return success since we use synchronize.Event to wait
            # for all processes to finish.
            self._pc.join(-1)

            # IMPORTANT: we use multiprocessing.Queue to carry worker return values
            # back to the parent, the worker process will wait before terminating
            # until all the buffered items are fed by the feeder thread to the underlying
            # pipe. Hence to prevent deadlocks on large return values,
            # we opportunistically try queue.get on each join call
            # See: https://docs.python.org/2/library/multiprocessing.html#all-platforms
            for local_rank in range(self.nprocs):
                return_queue = self._ret_vals[local_rank]
                if not return_queue.empty():
                    # save the return values temporarily into a member var
                    self._return_values[local_rank] = return_queue.get()

            if self._is_done():
                # we should ALWAYS have ALL the return values when all the processes are done
                self._worker_finished_event.set()

                # At this point workers finished running the user function
                # But the child process might still have not exited. Wait for them.
                # pc.join() blocks [forever] until "a" proc exits. Loop until all of them exits.
                while not self._pc.join():
                    logger.debug(
                        "entrypoint fn finished, waiting for all child procs to exit..."
                    )

                _validate_full_rank(
                    self._return_values, self.nprocs, "return_value queue"
                )
                self.close()
                return RunProcsResult(
                    return_values=self._return_values,
                    stdouts=self.stdouts,
                    stderrs=self.stderrs,
                )
            else:
                return None
        except (mp.ProcessRaisedException, mp.ProcessExitedException) as e:
            failed_local_rank = e.error_index

            # entrypoint for MultiprocessContext will always be a Callable
            fn_name = self.entrypoint.__qualname__  # type: ignore[union-attr]
            failed_proc = self._pc.processes[failed_local_rank]
            error_filepath = self.error_files[failed_local_rank]

            logger.exception(
                "failed (exitcode: %s)"
                " local_rank: %s (pid: %s)"
                " of fn: %s (start_method: %s)",
                failed_proc.exitcode,
                failed_local_rank,
                e.error_pid,
                fn_name,
                self.start_method,
            )

            self.close()
            return RunProcsResult(
                failures={
                    failed_local_rank: ProcessFailure(
                        local_rank=failed_local_rank,
                        pid=e.error_pid,
                        exitcode=failed_proc.exitcode,
                        error_file=error_filepath,
                    )
                },
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )

    def pids(self) -> dict[int, int]:
        if self._pc is None:
            raise AssertionError  # assertion for mypy type checking
        return dict(enumerate(self._pc.pids()))

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        if not self._pc:
            return
        for proc in self._pc.processes:
            if proc.is_alive():
                logger.warning(
                    "Closing process %s via signal %s", proc.pid, death_sig.name
                )
                try:
                    os.kill(proc.pid, death_sig)
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be raised, it is safe to ignore it.
                    pass
        end = time.monotonic() + timeout
        for proc in self._pc.processes:
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            proc.join(time_to_wait)
        for proc in self._pc.processes:
            if proc.is_alive():
                logger.warning(
                    "Unable to shutdown process %s via %s, forcefully exiting via %s",
                    proc.pid,
                    death_sig,
                    _get_kill_signal(),
                )
                try:
                    os.kill(proc.pid, _get_kill_signal())
                except ProcessLookupError:
                    # If the process exited because of some reason,
                    # `ProcessLookupError` will be raised, it is safe to ignore it.
                    pass
            # Bounded join: a process stuck in uninterruptible kernel sleep
            # (Linux D-state, common with hung NCCL/GPU collectives) cannot
            # be reaped by SIGKILL. Without a timeout the agent itself would
            # wedge here and the supervising launcher could never exit.
            proc.join(timeout)
            if proc.is_alive():
                logger.error(
                    "Process %s did not exit after %s within %ss;"
                    " abandoning and continuing teardown."
                    " Worker may be in uninterruptible kernel sleep"
                    " (e.g. wedged on GPU/NIC); host may need to be recycled.",
                    proc.pid,
                    _get_kill_signal().name,
                    timeout,
                )


class SubprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a binary."""

    def __init__(
        self,
        name: str,
        entrypoint: str,
        args: dict[int, tuple],
        envs: dict[int, dict[str, str]],
        logs_specs: LogsSpecs,
        log_line_prefixes: dict[int, str] | None = None,
        numa_options: NumaOptions | None = None,
        duplicate_stdout_filters: list[str] | None = None,
        duplicate_stderr_filters: list[str] | None = None,
    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
            duplicate_stdout_filters,
            duplicate_stderr_filters,
        )

        # state vector; _vdone[local_rank] -> is local_rank finished or not
        self._running_local_ranks: set[int] = set(range(self.nprocs))
        self._failures: dict[int, ProcessFailure] = {}
        self.subprocess_handlers: dict[int, SubprocessHandler] = {}
        self._numa_options: NumaOptions | None = numa_options

    def _start(self):
        if self.subprocess_handlers:
            raise ValueError(
                "The subprocess handlers already initialized. Most likely the start method got called twice."
            )
        self.subprocess_handlers = {
            local_rank: get_subprocess_handler(
                entrypoint=self.entrypoint,  # type: ignore[arg-type] # entrypoint is always a str
                args=self.args[local_rank],
                env=self.envs[local_rank],
                stdout=self.stdouts[local_rank],
                stderr=self.stderrs[local_rank],
                local_rank_id=local_rank,
                numa_options=self._numa_options,
            )
            for local_rank in range(self.nprocs)
        }

    def _capture_process_failures(self, done_local_ranks: set[int]):
        for local_rank in self._running_local_ranks:
            handler = self.subprocess_handlers[local_rank]
            exitcode = handler.proc.poll()
            if exitcode is not None:
                done_local_ranks.add(local_rank)
                if exitcode != 0:  # failed or signaled
                    self._failures[local_rank] = ProcessFailure(
                        local_rank=local_rank,
                        pid=handler.proc.pid,
                        exitcode=exitcode,
                        error_file=self.error_files[local_rank],
                    )
                # else: --> succeeded; nothing to do

    def _poll(self) -> RunProcsResult | None:
        done_local_ranks: set[int] = set()
        self._capture_process_failures(done_local_ranks)

        self._running_local_ranks.difference_update(done_local_ranks)

        # if ALL procs are finished or ANY have failed
        if not self._running_local_ranks or self._failures:
            self.close()  # terminate all running procs
            self._capture_process_failures(
                done_local_ranks
            )  # log sigterms and sigkill exit codes in the self._failures for bookkeeping purposes

            result = RunProcsResult(
                failures=self._failures,
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )
            if result.is_failed():
                first_failure = min(result.failures.values(), key=lambda f: f.timestamp)
                logger.error(
                    "failed (exitcode: %s) local_rank: %s (pid: %s) of binary: %s",
                    first_failure.exitcode,
                    first_failure.local_rank,
                    first_failure.pid,
                    self.entrypoint,
                )
            else:
                # Populate return with dummy values. This provides consistency with MultiprocessingHandler
                result.return_values = dict.fromkeys(range(self.nprocs))

            return result
        else:  # there are no failures and procs still running
            return None

    def pids(self) -> dict[int, int]:
        return {
            local_rank: sh.proc.pid
            for local_rank, sh in self.subprocess_handlers.items()
        }

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        if not self.subprocess_handlers:
            return
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                logger.warning(
                    "Sending process %s closing signal %s",
                    handler.proc.pid,
                    death_sig.name,
                )
                handler.close(death_sig=death_sig)
        end = time.monotonic() + timeout
        for handler in self.subprocess_handlers.values():
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            try:
                handler.proc.wait(time_to_wait)
            except subprocess.TimeoutExpired:
                # Ignore the timeout expired exception, since
                # the child process will be forcefully terminated via SIGKILL
                pass
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                logger.warning(
                    "Unable to shutdown process %s via %s, forcefully exiting via %s",
                    handler.proc.pid,
                    death_sig,
                    _get_kill_signal(),
                )
                handler.close(death_sig=_get_kill_signal())
                # Bounded wait: a process stuck in uninterruptible kernel sleep
                # (Linux D-state, common with hung NCCL/GPU collectives) cannot
                # be reaped by SIGKILL. Without a timeout the agent itself would
                # wedge here and the supervising launcher could never exit.
                try:
                    handler.proc.wait(timeout)
                except subprocess.TimeoutExpired:
                    logger.error(
                        "Process %s did not exit after %s within %ss;"
                        " abandoning and continuing teardown."
                        " Worker may be in uninterruptible kernel sleep"
                        " (e.g. wedged on GPU/NIC);"
                        " host may need to be recycled.",
                        handler.proc.pid,
                        _get_kill_signal().name,
                        timeout,
                    )
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Callable
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event
from typing import TextIO, TYPE_CHECKING


if TYPE_CHECKING:
    from concurrent.futures._base import Future

__all__ = ["tail_logfile", "TailLog"]

logger = logging.getLogger(__name__)


def tail_logfile(
    header: str,
    file: str,
    dst: TextIO,
    finished: Event,
    interval_sec: float,
    log_line_filter: Callable[[str], bool] | None = None,
):
    while not os.path.exists(file):
        if finished.is_set():
            return
        time.sleep(interval_sec)

    with open(file, errors="replace") as fp:
        while True:
            line = fp.readline()

            if line:
                if log_line_filter and log_line_filter(line):
                    dst.write(f"{header}{line}")
            else:  # reached EOF
                if finished.is_set():
                    # log line producer is finished
                    break
                else:
                    # log line producer is still going
                    # wait for a bit before looping again
                    time.sleep(interval_sec)


class TailLog:
    """
    Tail the given log files.

    The log files do not have to exist when the ``start()`` method is called. The tail-er will gracefully wait until
    the log files are created by the producer and will tail the contents of the
    log files until the ``stop()`` method is called.

    .. warning:: ``TailLog`` will wait indefinitely for the log file to be created!

    Each log file's line will be suffixed with a header of the form: ``[{name}{idx}]:``,
    where the ``name`` is user-provided and ``idx`` is the index of the log file
    in the ``log_files`` mapping. ``log_line_prefixes`` can be used to override the
    header for each log file.

    Usage:

    ::

     log_files = {0: "/tmp/0_stdout.log", 1: "/tmp/1_stdout.log"}
     tailer = TailLog("trainer", log_files, sys.stdout).start()
     # actually run the trainers to produce 0_stdout.log and 1_stdout.log
     run_trainers()
     tailer.stop()

     # once run_trainers() start writing the ##_stdout.log files
     # the tailer will print to sys.stdout:
     # >>> [trainer0]:log_line1
     # >>> [trainer1]:log_line1
     # >>> [trainer0]:log_line2
     # >>> [trainer0]:log_line3
     # >>> [trainer1]:log_line2

    .. note:: Due to buffering log lines between files may not necessarily
              be printed out in order. You should configure your application's
              logger to suffix each log line with a proper timestamp.

    """

    def __init__(
        self,
        name: str,
        log_files: dict[int, str],
        dst: TextIO,
        log_line_prefixes: dict[int, str] | None = None,
        interval_sec: float = 0.1,
        log_line_filter: Callable[[str], bool] = (lambda _: True),
    ):
        n = len(log_files)
        self._threadpool = None
        if n > 0:
            self._threadpool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix=f"{self.__class__.__qualname__}_{name}",
            )

        self._name = name
        self._dst = dst
        self._log_files = log_files
        self._log_line_prefixes = log_line_prefixes
        self._log_line_filter = log_line_filter
        self._finished_events: dict[int, Event] = {
            local_rank: Event() for local_rank in log_files
        }
        self._futs: list[Future] = []
        self._interval_sec = interval_sec
        self._stopped = False

    def start(self) -> "TailLog":
        if not self._threadpool or not self._dst:
            return self

        for local_rank, file in self._log_files.items():
            header = f"[{self._name}{local_rank}]:"
            if self._log_line_prefixes and local_rank in self._log_line_prefixes:
                header = self._log_line_prefixes[local_rank]
            self._futs.append(
                self._threadpool.submit(
                    tail_logfile,
                    header=header,
                    file=file,
                    dst=self._dst,
                    finished=self._finished_events[local_rank],
                    interval_sec=self._interval_sec,
                    log_line_filter=self._log_line_filter,
                )
            )
        return self

    def stop(self) -> None:
        for finished in self._finished_events.values():
            finished.set()

        for local_rank, f in enumerate(self._futs):
            try:
                f.result()
            except Exception as e:
                logger.exception(
                    "error in log tailer for %s%s. %s",
                    self._name,
                    local_rank,
                    e.__class__.__qualname__,
                )

        if self._threadpool:
            self._threadpool.shutdown(wait=True)

        self._stopped = True

    def stopped(self) -> bool:
        return self._stopped
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any


__all__ = ["ErrorHandler"]

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Write the provided exception object along with some other metadata about
    the error in a structured way in JSON format to an error file specified by the
    environment variable: ``TORCHELASTIC_ERROR_FILE``. If this environment
    variable is not set, then simply logs the contents of what would have been
    written to the error file.

    This handler may be subclassed to customize the handling of the error.
    Subclasses should override ``initialize()`` and ``record_exception()``.
    """

    def _get_error_file_path(self) -> str | None:
        """
        Return the error file path.

        May return ``None`` to have the structured error be logged only.
        """
        return os.environ.get("TORCHELASTIC_ERROR_FILE", None)

    def initialize(self) -> None:
        """
        Call prior to running code that we wish to capture errors/exceptions.

        Typically registers signal/fault handlers. Users can override this
        function to add custom initialization/registrations that aid in
        propagation/information of errors/signals/exceptions/faults.
        """
        try:
            faulthandler.enable(all_threads=True)
        except Exception as e:
            warnings.warn(
                f"Unable to enable fault handler. {type(e).__name__}: {e}", stacklevel=2
            )

    def _write_error_file(self, file_path: str, error_msg: str) -> None:
        """Write error message to the file."""
        try:
            with open(file_path, "w") as fp:
                fp.write(error_msg)
        except Exception as e:
            warnings.warn(
                f"Unable to write error to file. {type(e).__name__}: {e}", stacklevel=2
            )

    def record_exception(self, e: BaseException) -> None:
        """
        Write structured information about the exception into an error file in JSON format.

        If the error file cannot be determined, then logs the content
        that would have been written to the error file.
        """
        file = self._get_error_file_path()
        if file:
            data = {
                "message": {
                    "message": f"{type(e).__name__}: {e}",
                    "extraInfo": {
                        "py_callstack": traceback.format_exc(),
                        "timestamp": str(int(time.time())),
                    },
                }
            }
            with open(file, "w") as fp:
                json.dump(data, fp)

    def maybe_enrich_signal_failure_message(self, message: str, error_file: str) -> str:
        """Hook to enrich a signal (no-traceback) failure message.

        Called from ``ProcessFailure`` when a worker fails by signal (negative
        exitcode). Subclasses may override this to append device-specific fault
        context (e.g. GPU memory faults) scanned from the worker logs near
        ``error_file``, so it surfaces in the propagated failure regardless of
        how the failure is later handled. The base implementation is a no-op
        that returns ``message`` unchanged.
        """
        return message

    def override_error_code_in_rootcause_data(
        self,
        rootcause_error_file: str,
        rootcause_error: dict[str, Any],
        error_code: int = 0,
    ):
        """Modify the rootcause_error read from the file, to correctly set the exit code."""
        if "message" not in rootcause_error:
            logger.warning(
                "child error file (%s) does not have field `message`. \n"
                "cannot override error code: %s",
                rootcause_error_file,
                error_code,
            )
        elif isinstance(rootcause_error["message"], str):
            logger.warning(
                "child error file (%s) has a new message format. \n"
                "skipping error code override",
                rootcause_error_file,
            )
        else:
            rootcause_error["message"]["errorCode"] = error_code

    def dump_error_file(self, rootcause_error_file: str, error_code: int = 0):
        """Dump parent error file from child process's root cause error and error code."""
        with open(rootcause_error_file) as fp:
            rootcause_error = json.load(fp)
            # Override error code since the child process cannot capture the error code if it
            # is terminated by signals like SIGSEGV.
            if error_code:
                self.override_error_code_in_rootcause_data(
                    rootcause_error_file, rootcause_error, error_code
                )
            logger.debug(
                "child error file (%s) contents:\n%s",
                rootcause_error_file,
                json.dumps(rootcause_error, indent=2),
            )

        my_error_file = self._get_error_file_path()
        if my_error_file:
            # Guard against existing error files
            # This can happen when the child is created using multiprocessing
            # and the same env var (TORCHELASTIC_ERROR_FILE) is used on the
            # parent and child to specify the error files (respectively)
            # because the env vars on the child is set in the wrapper function
            # and by default the child inherits the parent's env vars, if the child
            # process receives a signal before the wrapper function kicks in
            # and the signal handler writes to the error file, then the child
            # will write to the parent's error file. In this case just log the
            # original error file contents and overwrite the error file.
            self._rm(my_error_file)
            self._write_error_file(my_error_file, json.dumps(rootcause_error))
            logger.info("dumped error file to parent's %s", my_error_file)
        else:
            logger.error(
                "no error file defined for parent, to copy child error file (%s)",
                rootcause_error_file,
            )

    def _rm(self, my_error_file):
        if os.path.isfile(my_error_file):
            # Log the contents of the original file.
            with open(my_error_file) as fp:
                try:
                    original = json.dumps(json.load(fp), indent=2)
                    logger.warning(
                        "%s already exists"
                        " and will be overwritten."
                        " Original contents:\n%s",
                        my_error_file,
                        original,
                    )
                except json.decoder.JSONDecodeError:
                    logger.warning(
                        "%s already exists"
                        " and will be overwritten."
                        " Unable to load original contents:\n",
                        my_error_file,
                    )
            os.remove(my_error_file)
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Each host in a distributed PyTorch job runs with a single TorchElastic agent,
and multiple workers (as children processes of the TorchElastic agent).
Since the workers are user-provided (your PyTorch script/job), TorchElastic
has a way to propagate errors on the trainers through the agent and up to the
scheduler, which ultimately informs the end-user about the state of the job
and applies any retry policies.

TorchElastic categorizes errors into 3 categories:

+----------------+----------------+--------------------------------------------------------------+
| Category       | Sub-Category   |  Description                                                 |
+================+================+==============================================================+
| User Error     | Input Error    | invalid inputs to TorchElastic APIs (e.g. min > max nodes)   |
|                +----------------+--------------------------------------------------------------+
|                | Worker Failure | any failures on the worker child process                     |
+----------------+----------------+--------------------------------------------------------------+
| Platform Error |      n/a       | failures caused by the agent                                 |
+----------------+----------------+--------------------------------------------------------------+
| Infra Error    |      n/a       | failures outside the domain of the agent and workers         |
|                |                | (e.g. host failures)                                         |
+----------------+----------------+--------------------------------------------------------------+

All errors other than "Worker Failure" are either raised canonically from the
agent process or implicitly or explicitly crash the agent process. So the
standard language (python) provided exception handling strategies apply.

Worker Failures are special because the exception/failure originates on a different
process from the agent so the error needs to be propagated inter-process
(e.g. the agent cannot simply ``try-catch`` an exception raised on the worker process).

TorchElastic agents use :func:`torch.distributed.elastic.multiprocessing.start_processes`
to launch the workers which has a simple file based inter-process error propagation
built-in.

Any function or binary entrypoint decorated with :func:`record`
will write uncaught exceptions (with the trace information) to a file specified by the
environment variable ``TORCHELASTIC_ERROR_FILE``. The parent process (e.g. agent)
sets this env var on each child it launches, then aggregates the error files for all
children, and propagates the one with the **smallest** timestamp (e.g. the **first** error).
"""

import json
import os
import signal
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from string import Template
from typing import Any, Optional, TypeVar, Union
from typing_extensions import ParamSpec

from torch.distributed.elastic.utils.logging import get_logger

from .error_handler import ErrorHandler
from .handlers import get_error_handler


__all__ = [
    "ProcessFailure",
    "ChildFailedError",
    "record",
    "ErrorHandler",
    "get_error_handler",
]

logger = get_logger(__name__)


JSON = dict[str, Any]

_EMPTY_ERROR_DATA: dict[str, Any] = {"message": "<NONE>"}
_NOT_AVAILABLE = "<N/A>"

_R = TypeVar("_R")
_P = ParamSpec("_P")


@dataclass
class ProcessFailure:
    """
    Represent the failed process result. When the worker process fails, it may record failure root cause into the file.

    Tries to read the failure timestamp from the provided ``error_file``,
    if the ``error_file`` does not exist, the timestamp is the current
    timestamp (seconds since epoch).

    The ``message`` field is a concise explanation of the failure. If
    the error file exists then the message is obtained from the error file.
    Otherwise one is generated based on the failure signature.

    .. note:: It is assumed that the ``error_file`` is written by
              ``torch.distributed.elastic.multiprocessing.errors.error_handler.ErrorHandler``.
              Otherwise the behavior is undefined.

    """

    local_rank: int
    pid: int
    exitcode: int
    error_file: str
    error_file_data: JSON = field(init=False)
    message: str = field(init=False)
    timestamp: int = field(init=False)

    def __post_init__(self):
        self.error_file_data = _EMPTY_ERROR_DATA
        if os.path.isfile(self.error_file):
            try:
                with open(self.error_file) as fp:
                    self.error_file_data = json.load(fp)
                    logger.debug(
                        "User process failed with error data: %s",
                        json.dumps(self.error_file_data, indent=2),
                    )
                    self.message, self.timestamp = self._get_error_data(
                        self.error_file_data
                    )
            except Exception:
                logger.exception("Failed to parse reply file: %s", self.error_file)
                raise
        else:
            self._set_no_reply_file()

        # make up an informative message if not already present
        if not self.message:
            # signals typically do not generate an error file message
            if self.exitcode < 0:
                self.message = (
                    f"Signal {-self.exitcode} ({self.signal_name()})"
                    f" received by PID {self.pid}"
                )
            else:
                self.error_file_data["errorTraits"] = {
                    "category": "system_terminated_error",
                    "retryability": "False",
                }
                self.message = "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html"

    def _get_error_data(self, error_file_data: dict[str, Any]) -> tuple[str, int]:
        message = error_file_data["message"]
        if isinstance(message, str):
            timestamp = int(error_file_data.get("timestamp", 0))
        else:
            timestamp = int(message["extraInfo"]["timestamp"])
        return (message, timestamp)

    def _set_no_reply_file(self):
        self.error_file = _NOT_AVAILABLE
        self.error_file_data = _EMPTY_ERROR_DATA
        self.message = ""
        self.timestamp = int(time.time())

    def signal_name(self) -> str:
        if self.exitcode < 0:
            # We don't want to kill the parent process trying to find the signal name.
            # if the signal doesn't map to a known name, use not available.
            try:
                return signal.Signals(-self.exitcode).name
            except Exception:
                return _NOT_AVAILABLE
        else:
            return _NOT_AVAILABLE

    def timestamp_isoformat(self):
        """Return timestamp in ISO format (YYYY-MM-DD_HH:MM:SS)."""
        return datetime.fromtimestamp(self.timestamp).isoformat(sep="_")


GlobalRank = int

_FAILURE_FORMAT_TEMPLATE = """[${idx}]:
  time      : ${time}
  host      : ${hostname}
  rank      : ${rank} (local_rank: ${local_rank})
  exitcode  : ${exitcode} (pid: ${pid}) ${signal_name}
  error_file: ${error_file}
  traceback : ${message}"""

# extra new lines before and after are intentional
_MSG_FORMAT_TEMPLATE = """
${boarder}
${title}
${section}
Failures:
${other_failures}
${section}
Root Cause (first observed failure):
${root_failure}
${boarder}"""


class ChildFailedError(Exception):
    """
    Special exception type that can be raised from a function annotated with the
    ``@record`` decorator to have the child process' (root exception) propagate
    up the stack as-is (e.g. without being wrapped in the parent's traceback).

    Useful in cases where the parent is a simple nanny process
    and the child (worker) processes are actually doing meaningful compute.
    In this case, errors typically occur on the child process as the parent
    is not doing anything non-trivial, and child errors should be propagated
    to the scheduler for accurate root cause diagnostics.

    .. note:: The propagation relies on error files rather than exception handling to
              support both function and binary launches.

    Example:
    ::

     # process tree on a host (container)
     0: scheduler-init-process:
                |- 1: torchelastic_agent:
                         |- 2: trainer_0 (ok)
                         |- 3: trainer_1 (fail) -> error.json
                         |- ...
                         |- n+2: trainer_n (ok)
                |- n+3: other processes
                |- ...

    In the example above, trainer 1's failure (written into error.json) is
    the root cause and should be reported to the scheduler's init process.
    The torchelastic agent raises a ``ChildFailedError("trainer", {1: "trainer_1/error.json"})``
    upon detecting trainer 1's failure which would propagate the contents
    of trainer 1's error file to the scheduler's init process.
    """

    def __init__(self, name: str, failures: dict[GlobalRank, ProcessFailure]):
        self.name = name
        self.failures = failures
        # does not make sense to create a ChildFailedError with no failures
        if not self.failures:
            raise AssertionError
        super().__init__(self.format_msg())

    def get_first_failure(self) -> tuple[GlobalRank, ProcessFailure]:
        rank = min(self.failures.keys(), key=lambda r: self.failures[r].timestamp)
        return rank, self.failures[rank]

    def format_msg(self, boarder_delim="=", section_delim="-"):
        title = f"{self.name} FAILED"
        root_rank, _root_failure = self.get_first_failure()

        root_failure_fmt: str = ""
        other_failures_fmt: list[str] = []
        width = len(title)
        for idx, (rank, failure) in enumerate(self.failures.items()):
            fmt, w = self._format_failure(idx, rank, failure)
            width = max(width, w)
            if rank == root_rank:
                root_failure_fmt = fmt
            else:
                other_failures_fmt.append(fmt)

        # upper boundary on width
        width = min(width, 60)

        return Template(_MSG_FORMAT_TEMPLATE).substitute(
            boarder=boarder_delim * width,
            title=title,
            section=section_delim * width,
            root_failure=root_failure_fmt,
            other_failures="\n".join(other_failures_fmt or ["  <NO_OTHER_FAILURES>"]),
        )

    def _format_failure(
        self, idx: int, rank: int, failure: ProcessFailure
    ) -> tuple[str, int]:
        # failure.message is either a str (when the failure does not generate a traceback - e.g. signals)
        # or a dict (json) of the form
        # {"message": $ERROR_MSG, "extraInfo": {"py_callstack": $TRACEBACK, timestamp: $TS}}
        # so the display logic is:
        # 1. if failure.message is not a dict (it is a str) just show it as is
        # 2. else try to get the traceback (py_callstack)
        # 3.      if the traceback is not there, use the message
        # 4.      if the message  is not there show <N/A>
        msg = failure.message
        if isinstance(failure.message, dict):
            msg = (
                failure.message.get("extraInfo", {})
                .get("py_callstack", failure.message.get("message", "<N/A>"))
                .replace("\n", "\n  ")  # to properly indent the traceback
            )

        # For signal failures, let the (build-swapped) handler append
        # device-fault context (e.g. ROCm GPU faults) to the rendered message.
        # Works on the local string, so error_file_data is never mutated, and it
        # is a no-op in OSS (the base handler returns the message unchanged).
        # Best-effort: never let a handler override break failure rendering.
        if failure.exitcode < 0:
            try:
                msg = get_error_handler().maybe_enrich_signal_failure_message(
                    msg, failure.error_file
                )
            except Exception:
                logger.warning("Failed to enrich signal failure message", exc_info=True)

        signal_name = failure.signal_name()
        signal_name_str = f" ({signal_name})" if signal_name != _NOT_AVAILABLE else ""

        fmt = Template(_FAILURE_FORMAT_TEMPLATE).substitute(
            idx=idx,
            time=failure.timestamp_isoformat(),
            hostname=socket.getfqdn(),
            rank=rank,
            local_rank=failure.local_rank,
            exitcode=failure.exitcode,
            pid=failure.pid,
            signal_name=signal_name_str,
            error_file=failure.error_file,
            message=msg,
        )
        width = 0
        for line in fmt.split("\n"):
            width = max(width, len(line))
        return fmt, width


def record(
    fn: Callable[_P, _R], error_handler: ErrorHandler | None = None
) -> Callable[_P, _R | None]:
    """
    Syntactic sugar to record errors/exceptions that happened in the decorated
    function using the provided ``error_handler``.

    Using this decorator is equivalent to:

    ::

     error_handler = get_error_handler()
     error_handler.initialize()
     try:
         foobar()
     except ChildFailedError as e:
         _, failure = e.get_first_failure()
         error_handler.dump_error_file(failure.error_file, failure.exitcode)
         raise
     except Exception as e:
         error_handler.record_exception(e)
         raise

    .. important:: use this decorator once per process at the top level method,
                   typically this is the main method.

    Example

    ::

     @record
     def main():
         pass


     if __name__ == "__main__":
         main()

    """
    if not error_handler:
        error_handler = get_error_handler()

    def wrap(f: Callable[_P, _R]) -> Callable[_P, _R | None]:
        @wraps(f)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs):
            if error_handler is None:
                raise AssertionError  # assertion for mypy type checker
            error_handler.initialize()
            try:
                return f(*args, **kwargs)
            except SystemExit as se:
                # For run_path based entrypoints, SystemExit with code = 0 will never exit.
                # Handling it here by returning a value:
                if se.code == 0:
                    return None
                else:
                    raise
            except ChildFailedError as e:
                rank, failure = e.get_first_failure()
                if failure.error_file != _NOT_AVAILABLE:
                    error_handler.dump_error_file(failure.error_file, failure.exitcode)
                else:
                    logger.info(
                        (
                            "local_rank %s FAILED with no error file."
                            " Decorate your entrypoint fn with @record for traceback info."
                            " See: https://pytorch.org/docs/stable/elastic/errors.html",
                            rank,
                        )
                    )
                raise
            except Exception as e:
                error_handler.record_exception(e)
                raise

        return wrapper

    return wrap(fn)
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module


from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler


__all__ = ["get_error_handler"]


def get_error_handler() -> ErrorHandler:
    return ErrorHandler()
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import signal
import sys
from subprocess import Popen
from typing import Any

from torch.numa.binding import _maybe_wrap_command_args_with_numa_binding, NumaOptions


__all__ = ["SubprocessHandler"]

IS_WINDOWS = sys.platform == "win32"


def _get_default_signal() -> signal.Signals:
    """Get the default termination signal. SIGTERM for unix, CTRL_C_EVENT for windows."""
    if IS_WINDOWS:
        return signal.CTRL_C_EVENT  # type: ignore[attr-defined]
    else:
        return signal.SIGTERM


class SubprocessHandler:
    """
    Convenience wrapper around python's ``subprocess.Popen``. Keeps track of
    meta-objects associated to the process (e.g. stdout and stderr redirect fds).
    """

    def __init__(
        self,
        entrypoint: str,
        args: tuple,
        env: dict[str, str],
        stdout: str | None,
        stderr: str | None,
        local_rank_id: int,
        numa_options: NumaOptions | None,
    ):
        self._stdout = open(stdout, "w") if stdout else None  # noqa: SIM115
        self._stderr = open(stderr, "w") if stderr else None  # noqa: SIM115
        # inherit parent environment vars
        env_vars = os.environ.copy()
        env_vars.update(env)

        args_str = (entrypoint, *[str(e) for e in args])
        args_str = _maybe_wrap_command_args_with_numa_binding(
            args_str,
            device_index=local_rank_id,
            numa_options=numa_options,
        )

        self.local_rank_id = local_rank_id

        self.proc: Popen = self._popen(args_str, env_vars)

    def _popen(self, args: tuple, env: dict[str, str]) -> Popen:
        kwargs: dict[str, Any] = {}
        if not IS_WINDOWS:
            kwargs["start_new_session"] = True

        return Popen(
            # pyre-fixme[6]: Expected `Union[typing.Sequence[Union[_PathLike[bytes],
            #  _PathLike[str], bytes, str]], bytes, str]` for 1st param but got
            #  `Tuple[str, *Tuple[Any, ...]]`.
            args=args,
            env=env,
            stdout=self._stdout,
            stderr=self._stderr,
            **kwargs,
        )

    def close(self, death_sig: signal.Signals | None = None) -> None:
        if not death_sig:
            death_sig = _get_default_signal()
        if IS_WINDOWS:
            self.proc.send_signal(death_sig)
        else:
            os.killpg(self.proc.pid, death_sig)
        if self._stdout:
            self._stdout.close()
        if self._stderr:
            self._stderr.close()
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler import (
    SubprocessHandler,
)
from torch.numa.binding import NumaOptions


__all__ = ["get_subprocess_handler"]


def get_subprocess_handler(
    entrypoint: str,
    args: tuple,
    env: dict[str, str],
    stdout: str,
    stderr: str,
    local_rank_id: int,
    numa_options: NumaOptions | None = None,
) -> SubprocessHandler:
    return SubprocessHandler(
        entrypoint=entrypoint,
        args=args,
        env=env,
        stdout=stdout,
        stderr=stderr,
        local_rank_id=local_rank_id,
        numa_options=numa_options,
    )
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
import warnings

from torch.distributed.elastic.utils.log_level import get_log_level


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Util function to set up a simple logger that writes
    into stderr. The loglevel is fetched from the LOGLEVEL
    env. variable or WARNING as default. The function will use the
    module name of the caller if no name is provided.

    Args:
        name: Name of the logger. If no name provided, the name will
              be derived from the call stack.
    """

    # Derive the name of the caller, if none provided
    # Use depth=2 since this function takes up one level in the call stack
    return _setup_logger(name or _derive_module_name(depth=2))


def _setup_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOGLEVEL", get_log_level()))
    return logger


def _derive_module_name(depth: int = 1) -> str | None:
    """
    Derives the name of the caller module from the stack frames.

    Args:
        depth: The position of the frame in the stack.
    """
    try:
        stack = inspect.stack()
        if depth >= len(stack):
            raise AssertionError
        # FrameInfo is just a named tuple: (frame, filename, lineno, function, code_context, index)
        frame_info = stack[depth]

        module = inspect.getmodule(frame_info[0])
        if module:
            module_name = module.__name__
        else:
            # inspect.getmodule(frame_info[0]) does NOT work (returns None) in
            # binaries built with @mode/opt
            # return the filename (minus the .py extension) as modulename
            filename = frame_info[1]
            module_name = os.path.splitext(os.path.basename(filename))[0]
        return module_name
    except Exception as e:
        warnings.warn(
            f"Error deriving logger module name, using <None>. Exception: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import os
import socket
from contextlib import closing

import torch.distributed as dist
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.elastic.utils.store import barrier


__all__ = ["create_c10d_store", "get_free_port", "get_socket_with_port"]

logger = get_logger(__name__)

_ADDRESS_IN_USE = "address already in use"
_SOCKET_TIMEOUT = "Socket Timeout"

_TCP_STORE_INIT = "_tcp_store/num_members"


def create_c10d_store(
    is_server: bool,
    server_addr: str,
    server_port: int = -1,
    world_size: int = 1,
    timeout: float = (60 * 10),  # 10 min
    wait_for_workers: bool = True,
    retries=3,
    use_libuv: bool | None = None,
):
    if use_libuv is not None:
        logger.warning(
            "argument use_libuv is deprecated and ignored. Set USE_LIBUV environment "
            'variable to "0" to disable libuv, or "1" to enable it. If the env var '
            "is not set, libuv will be used by default."
        )

    # check os.environ for use_libuv
    use_libuv = os.environ.get("USE_LIBUV", "1") == "1"  # libuv is the default option

    if server_port == -1 and world_size > 1:
        raise ValueError(
            f"server_port must be specified when world_size > 1, got server_port={server_port}, world_size={world_size}"
        )

    if server_port != -1:
        logger.info("sever_port: %s, specified, ignoring retries", server_port)

    # only retry when server_port is NOT static
    attempt = retries if server_port == -1 else 1
    while True:
        if server_port != -1:
            port = server_port
        else:
            port = get_free_port()

        logger.info(
            "Creating c10d store on %s:%s\n"
            "  world_size  : %s\n"
            "  is_server   : %s\n"
            "  timeout(sec): %s\n"
            "  use_libuv   : %s\n",
            server_addr,
            port,
            world_size,
            is_server,
            timeout,
            use_libuv,
        )

        try:
            store = dist.TCPStore(
                host_name=server_addr,
                port=port,
                world_size=world_size,
                is_master=is_server,
                timeout=datetime.timedelta(seconds=timeout),
                wait_for_workers=wait_for_workers,
                use_libuv=use_libuv,
            )
            # skips full rank check when we don't have to wait for all workers
            if wait_for_workers:
                _check_full_rank(store, world_size, timeout=timeout)
            logger.info("Successfully created c10d store")
            return store
        except dist.DistNetworkError as e:
            if _ADDRESS_IN_USE in str(e).lower():  # this will only happen on the server
                if attempt < retries:
                    logger.warning(
                        "port: %s already in use, attempt: [%s/%s]",
                        port,
                        attempt,
                        retries,
                    )
                    attempt += 1
                else:
                    raise dist.DistNetworkError(
                        f"on {server_addr}, port: {port} already in use"
                    ) from e
            else:
                raise


def _check_full_rank(store, world_size, timeout):
    try:
        barrier(store, world_size, key_prefix=_TCP_STORE_INIT, barrier_timeout=timeout)
    except RuntimeError as e:
        if str(e) == _SOCKET_TIMEOUT:
            raise TimeoutError(
                f"timed out waiting for all {world_size} members to join"
            ) from e
        else:
            raise


def get_free_port():
    """
    Returns an unused port on localhost.

    This function finds an unused port on localhost by opening to socket to bind
    to a port and then closing it.

    Returns:
        int: an unused port on localhost

    Example:
        >>> # xdoctest: +SKIP("Nondeterministic")
        >>> get_free_port()
        63976

    .. note::
        The port returned by :func:`get_free_port` is not reserved and may be
        taken by another process after this function returns.
    """
    sock = get_socket_with_port()
    with closing(sock):
        return sock.getsockname()[1]


def get_socket_with_port() -> socket.socket:
    """
    Returns a free port on localhost that is "reserved" by binding a temporary
    socket on it. Close the socket before passing the port to the entity
    that requires it. Usage example

    ::

    sock = _get_socket_with_port()
    with closing(sock):
        port = sock.getsockname()[1]
        sock.close()
        # there is still a race-condition that some other process
        # may grab this port before func() runs
        func(port)
    """

    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            logger.warning("Socket creation attempt failed.", exc_info=e)
    raise RuntimeError("Failed to create a socket")
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import timedelta

import torch


DistStoreError = torch._C._DistStoreError

_NUM_MEMBERS = "/num_members"
_LAST_MEMBER_CHECKIN = "/last_member"
_TRACE = "/TRACE"
_TRACING_GATE = "/TRACING_GATE"
_MAX_TRACE_MISSING_RANKS = 16


__all__ = ["store_timeout", "get_all", "synchronize", "barrier"]


@contextmanager
def store_timeout(store, timeout: float):
    """
    This sets the timeout and then restores the old timeout when the context
    manager exits.

    Args:
        store: the store to set the timeout on
        timeout: the timeout to set
    """

    old_timeout = store.timeout
    store.set_timeout(timedelta(seconds=timeout))
    yield
    store.set_timeout(old_timeout)


def get_all(store, rank: int, prefix: str, world_size: int):
    r"""
    Given a store and a prefix, the method goes through the array of keys
    of the following format: ``{prefix}{idx}``, where idx is in a range
    from 0 to size, and tries to retrieve the data.

    The Rank0 process waits at the end to make sure all other processes
    finished the procedure before exiting.

    Usage

    ::

     values = get_all(store, "torchelastic/data", 3)
     value1 = values[0]  # retrieves the data for key torchelastic/data0
     value2 = values[1]  # retrieves the data for key torchelastic/data1
     value3 = values[2]  # retrieves the data for key torchelastic/data2

    """
    data_arr = store.multi_get([f"{prefix}{idx}" for idx in range(world_size)])

    barrier_key = _barrier_nonblocking(
        store=store,
        world_size=world_size,
        key_prefix=f"{prefix}/finished",
    )
    if rank == 0:
        # Rank0 runs the TCPStore daemon, as a result it needs to exit last.
        # Otherwise, the barrier may timeout if rank0 process finished the work
        # before other processes finished `get_all` method
        store.wait([barrier_key])

    return data_arr


def synchronize(
    store,
    data: bytes,
    rank: int,
    world_size: int,
    key_prefix: str,
    timeout: float = 300,
) -> list[bytes]:
    """
    Synchronizes ``world_size`` agents between each other using the underlying c10d store.
    The ``data`` will be available on each of the agents.

    Note: The data on the path is not deleted, as a result there can be stale data if
        you use the same key_prefix twice.

    Time complexity: O(N) per worker, O(N^2) globally.
    """
    with store_timeout(store, timeout):
        store.set(f"{key_prefix}{rank}", data)
        agent_data = get_all(store, rank, key_prefix, world_size)
        return agent_data


def _try_detecting_missing_ranks(
    store,
    world_size: int,
    key_prefix: str,
    rank: int,
    rank_decoder: Callable[[int], str],
    trace_timeout: float,
) -> Iterable[str] | None:
    store.set(f"{key_prefix}{rank}{_TRACE}", "<val_ignored>")

    def _find_missing_ranks():
        missing_rank_info = set()
        ranks_missing = 0
        for i in range(1, world_size):
            # reduce noise, assuming in general 8 ranks per node
            # It is valuable to know that 1 or >1 nodes have timed-out.
            if ranks_missing >= _MAX_TRACE_MISSING_RANKS:
                break
            try:
                if ranks_missing == 0:
                    store.wait(
                        [f"{key_prefix}{i}{_TRACE}"], timedelta(seconds=trace_timeout)
                    )
                else:
                    # use a shortest timeout, some ranks have failed to check-in
                    store.wait([f"{key_prefix}{i}{_TRACE}"], timedelta(milliseconds=1))
            except DistStoreError:
                ranks_missing += 1
                missing_rank_info.add(rank_decoder(i))
        return missing_rank_info

    def _checkin():
        try:
            store.wait([f"{key_prefix}{_TRACING_GATE}"])
            return [f"[<check rank 0 ({rank_decoder(0)}) for missing rank info>]"]
        except DistStoreError:
            # in case rank0 is the source of the timeout, original exception will be raised
            return None

    if rank == 0:
        missing_rank_info = _find_missing_ranks()
        store.set(f"{key_prefix}{_TRACING_GATE}", "<val_ignored>")
        return missing_rank_info
    else:
        return _checkin()


def _barrier_nonblocking(store, world_size: int, key_prefix: str) -> str:
    """
    Does all the non-blocking operations for a barrier and returns the final key
    that can be waited on.
    """
    num_members_key = key_prefix + _NUM_MEMBERS
    last_member_key = key_prefix + _LAST_MEMBER_CHECKIN

    idx = store.add(num_members_key, 1)
    if idx == world_size:
        store.set(last_member_key, "<val_ignored>")

    return last_member_key


def barrier(
    store,
    world_size: int,
    key_prefix: str,
    barrier_timeout: float = 300,
    rank: int | None = None,
    rank_tracing_decoder: Callable[[int], str] | None = None,
    trace_timeout: float = 10,
) -> None:
    """
    A global lock between agents. This will pause all workers until at least
    ``world_size`` workers respond.

    This uses a fast incrementing index to assign waiting ranks and a success
    flag set by the last worker.

    Time complexity: O(1) per worker, O(N) globally.

    Optionally, passing rank will enable tracing of missing ranks on timeouts.
    `rank_tracing_decoder` lambda arg can be used to convert rank data
    into a more meaningful information at an app level (e.g. hostname).

    Note: Since the data is not removed from the store, the barrier can be used
        once per unique ``key_prefix``.
    """

    if rank is None:
        if rank_tracing_decoder is not None:
            raise AssertionError("Tracing requires rank information")

    with store_timeout(store, barrier_timeout):
        last_member_key = _barrier_nonblocking(
            store=store, world_size=world_size, key_prefix=key_prefix
        )
        try:
            store.wait([last_member_key])
        except DistStoreError as e:
            if rank is None:
                raise e
            else:
                missing_ranks = _try_detecting_missing_ranks(
                    store,
                    world_size,
                    key_prefix,
                    rank,
                    rank_tracing_decoder or (lambda x: str(x)),
                    trace_timeout,
                )
                if missing_ranks is not None:
                    raise DistStoreError(
                        "Timed out waiting on barrier on "
                        "rank {}, for key prefix: {} (world_size={}, missing_ranks={}, timeout={})".format(
                            rank,
                            key_prefix,
                            world_size,
                            f"[{', '.join(missing_ranks)}]",
                            barrier_timeout,
                        )
                    ) from None
                else:
                    raise e
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from string import Template
from typing import Any


def get_env_variable_or_raise(env_name: str) -> str:
    r"""
    Tries to retrieve environment variable. Raises ``ValueError``
    if no environment variable found.

    Args:
        env_name (str): Name of the env variable
    """
    value = os.environ.get(env_name, None)
    if value is None:
        msg = f"Environment variable {env_name} expected, but not set"
        raise ValueError(msg)
    return value


def get_socket_with_port() -> socket.socket:
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError:
            s.close()
    raise RuntimeError("Failed to create a socket")


class macros:
    """
    Defines simple macros for caffe2.distributed.launch cmd args substitution
    """

    local_rank = "${local_rank}"

    @staticmethod
    def substitute(args: list[Any], local_rank: str) -> list[str]:
        args_sub = []
        for arg in args:
            if isinstance(arg, str):
                sub = Template(arg).safe_substitute(local_rank=local_rank)
                args_sub.append(sub)
            else:
                args_sub.append(arg)
        return args_sub
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for inspecting Linux process state from ``/proc``.

These are intentionally agent-agnostic so they can be reused outside of
``torch.distributed.elastic.agent``.
"""

from __future__ import annotations


__all__ = ["read_proc_state", "is_uninterruptible_state"]


def read_proc_state(pid: int) -> str | None:
    """Read the process state char from ``/proc/<pid>/stat``. Linux-only.

    Returns the single-letter state (e.g. ``'R'``, ``'S'``, ``'D'``,
    ``'Z'``) or ``None`` if the file is unreadable (non-Linux, process
    gone, permission denied, or parse error). The ``comm`` field can
    contain spaces and parentheses, so the parser keys off the last
    ``)`` rather than splitting on whitespace.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        rparen = data.rfind(")")
        if rparen == -1:
            return None
        # Fields after comm are space-separated; state is field 3 overall
        # i.e. the first token after the closing ')'.
        rest = data[rparen + 1 :].split()
        if not rest:
            return None
        return rest[0]
    except OSError:
        return None


def is_uninterruptible_state(state: str | None) -> bool:
    """Return True if ``state`` is Linux uninterruptible sleep ('D').

    Includes the ``'D+'`` (foreground) variant; the state field is
    a single char in the kernel format we read, but we accept any
    string that starts with ``'D'`` defensively.
    """
    return state is not None and state.startswith("D")
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Iterator, Sized
from typing import cast, TypeVar

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


T = TypeVar("T")

__all__ = ["ElasticDistributedSampler"]


class ElasticDistributedSampler(DistributedSampler[T]):
    """
    Sampler that restricts data loading to a subset of
    the dataset for elastic training.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        start_index (optional):  Which index of the dataset to start sampling from
    """

    def __init__(
        self,
        dataset: Dataset[T],
        num_replicas: int | None = None,
        rank: int | None = None,
        start_index: int = 0,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        if not isinstance(dataset, Sized):
            raise TypeError("Dataset must be an instance of collections.abc.Sized")

        # Cast to Sized for mypy

        sized_dataset = cast(Sized, dataset)

        if start_index >= len(sized_dataset):
            raise ValueError(
                f"Start index {start_index} should be less than dataset size {len(sized_dataset)}"
            )

        self.start_index = start_index
        sized_dataset = cast(Sized, self.dataset)
        self.num_samples = math.ceil(
            float(len(sized_dataset) - self.start_index) / self.num_replicas
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        sized_dataset = cast(Sized, self.dataset)
        indices = (
            torch.randperm(len(sized_dataset) - self.start_index, generator=g)
            .add(self.start_index)
            .tolist()
        )

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise AssertionError

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
<|endoftext|>#!/usr/bin/env python3

from collections.abc import Callable, Iterator
from typing import TypeVar
from typing_extensions import Self


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_T = TypeVar("_T")

__all__ = ["CyclingIterator"]


class CyclingIterator(Iterator[_T]):
    """
    An iterator decorator that cycles through the
    underlying iterator "n" times. Useful to "unroll"
    the dataset across multiple training epochs.

    The generator function is called as ``generator_fn(epoch)``
    to obtain the underlying iterator, where ``epoch`` is a
    number less than or equal to ``n`` representing the ``k``th cycle

    For example if ``generator_fn`` always returns ``[1,2,3]``
    then ``CyclingIterator(n=2, generator_fn)`` will iterate through
    ``[1,2,3,1,2,3]``
    """

    def __init__(
        self,
        n: int,
        generator_fn: Callable[[int], Iterator[_T]],
        start_epoch: int = 0,
    ):
        self._n = n
        self._epoch = start_epoch
        self._generator_fn = generator_fn
        self._iter = generator_fn(self._epoch)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> _T:
        try:
            return next(self._iter)
        except StopIteration as eod:  # eod == end of data
            if self._epoch < self._n - 1:
                self._epoch += 1
                self._iter = self._generator_fn(self._epoch)
                return self.__next__()
            else:
                raise eod
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Expiration timers are set up on the same process as the agent and
used from your script to deal with stuck workers. When you go into
a code-block that has the potential to get stuck you can acquire
an expiration timer, which instructs the timer server to kill the
process if it does not release the timer by the self-imposed expiration
deadline.

Usage::

    import torchelastic.timer as timer
    import torchelastic.agent.server as agent

    def main():
        start_method = "spawn"
        message_queue = mp.get_context(start_method).Queue()
        server = timer.LocalTimerServer(message, max_interval=0.01)
        server.start() # non-blocking

        spec = WorkerSpec(
                    fn=trainer_func,
                    args=(message_queue,),
                    ...<OTHER_PARAMS...>)
        agent = agent.LocalElasticAgent(spec, start_method)
        agent.run()

    def trainer_func(message_queue):
        timer.configure(timer.LocalTimerClient(message_queue))
        with timer.expires(after=60): # 60 second expiry
            # do some work

In the example above if ``trainer_func`` takes more than 60 seconds to
complete, then the worker process is killed and the agent retries the worker group.
"""

from .api import configure, expires, TimerClient, TimerRequest, TimerServer
from .file_based_local_timer import FileTimerClient, FileTimerRequest, FileTimerServer
from .local_timer import LocalTimerClient, LocalTimerServer
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torch.distributed.elastic.utils.logging import get_logger


logger = get_logger(__name__)

__all__ = ["log_debug_info_for_expired_timers"]


def log_debug_info_for_expired_timers(
    run_id: str,
    expired_timers: dict[int, list[str]],
):
    if expired_timers:
        logger.info("Timers expired for run:[%s] [%s].", run_id, expired_timers)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty
from typing import Any

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer


__all__ = ["LocalTimerClient", "MultiprocessingRequestQueue", "LocalTimerServer"]

logger = logging.getLogger(__name__)


class LocalTimerClient(TimerClient):
    """
    Client side of ``LocalTimerServer``. This client is meant to be used
    on the same host that the ``LocalTimerServer`` is running on and uses
    pid to uniquely identify a worker. This is particularly useful in situations
    where one spawns a subprocess (trainer) per GPU on a host with multiple
    GPU devices.
    """

    def __init__(self, mp_queue):
        super().__init__()
        self._mp_queue = mp_queue

    def acquire(self, scope_id, expiration_time):
        pid = os.getpid()
        acquire_request = TimerRequest(pid, scope_id, expiration_time)
        self._mp_queue.put(acquire_request)

    def release(self, scope_id):
        pid = os.getpid()
        release_request = TimerRequest(pid, scope_id, -1)
        self._mp_queue.put(release_request)


class MultiprocessingRequestQueue(RequestQueue):
    """
    A ``RequestQueue`` backed by python ``multiprocessing.Queue``
    """

    def __init__(self, mp_queue: mp.Queue):
        super().__init__()
        self._mp_queue = mp_queue

    def size(self) -> int:
        return self._mp_queue.qsize()

    def get(self, size, timeout: float) -> list[TimerRequest]:
        requests = []
        wait = timeout
        for _ in range(size):
            start = time.time()

            try:
                r = self._mp_queue.get(block=True, timeout=wait)
            except Empty:
                break

            requests.append(r)
            wait = wait - (time.time() - start)
            if wait <= 0:
                break

        return requests


class LocalTimerServer(TimerServer):
    """
    Server that works with ``LocalTimerClient``. Clients are expected to be
    subprocesses to the parent process that is running this server. Each host
    in the job is expected to start its own timer server locally and each
    server instance manages timers for local workers (running on processes
    on the same host).
    """

    def __init__(
        self, mp_queue: mp.Queue, max_interval: float = 60, daemon: bool = True
    ):
        super().__init__(MultiprocessingRequestQueue(mp_queue), max_interval, daemon)
        self._timers: dict[tuple[Any, str], TimerRequest] = {}

    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        for request in timer_requests:
            pid = request.worker_id
            scope_id = request.scope_id
            expiration_time = request.expiration_time

            # negative expiration is a proxy for a release call
            if expiration_time < 0:
                self._timers.pop((pid, scope_id), None)
            else:
                self._timers[(pid, scope_id)] = request

    def clear_timers(self, worker_ids: set[int]) -> None:
        for pid, scope_id in list(self._timers.keys()):
            if pid in worker_ids:
                self._timers.pop((pid, scope_id))

    def get_expired_timers(self, deadline: float) -> dict[Any, list[TimerRequest]]:
        # pid -> [timer_requests...]
        expired_timers: dict[Any, list[TimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_id, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_id: int) -> bool:
        try:
            os.kill(worker_id, signal.SIGKILL)
            return True
        except ProcessLookupError:
            logger.info("Process with pid=%s does not exist. Skipping", worker_id)
            return True
        except Exception:
            logger.exception("Error terminating pid=%s", worker_id)
        return False
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any


__all__ = [
    "TimerRequest",
    "TimerClient",
    "RequestQueue",
    "TimerServer",
    "configure",
    "expires",
]

logger = logging.getLogger(__name__)


class TimerRequest:
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``TimerClient`` and ``TimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.

    .. note:: the type of ``worker_id`` is implementation specific.
              It is whatever the TimerServer and TimerClient implementations
              have on to uniquely identify a worker.
    """

    __slots__ = ["worker_id", "scope_id", "expiration_time"]

    def __init__(self, worker_id: Any, scope_id: str, expiration_time: float):
        self.worker_id = worker_id
        self.scope_id = scope_id
        self.expiration_time = expiration_time

    def __eq__(self, other):
        if isinstance(other, TimerRequest):
            return (
                self.worker_id == other.worker_id
                and self.scope_id == other.scope_id
                and self.expiration_time == other.expiration_time
            )
        return False


class TimerClient(abc.ABC):
    """
    Client library to acquire and release countdown timers by communicating
    with the TimerServer.
    """

    @abc.abstractmethod
    def acquire(self, scope_id: str, expiration_time: float) -> None:
        """
        Acquires a timer for the worker that holds this client object
        given the scope_id and expiration_time. Typically registers
        the timer with the TimerServer.
        """

    @abc.abstractmethod
    def release(self, scope_id: str):
        """
        Releases the timer for the ``scope_id`` on the worker this
        client represents. After this method is
        called, the countdown timer on the scope is no longer in effect.
        """


class RequestQueue(abc.ABC):
    """
    Consumer queue holding timer acquisition/release requests
    """

    @abc.abstractmethod
    def size(self) -> int:
        """
        Returns the size of the queue at the time this method is called.
        Note that by the time ``get`` is called the size of the queue
        may have increased. The size of the queue should not decrease
        until the ``get`` method is called. That is, the following assertion
        should hold:

        size = q.size()
        res = q.get(size, timeout=0)
        assert size == len(res)

        -- or --

        size = q.size()
        res = q.get(size * 2, timeout=1)
        assert size <= len(res) <= size * 2
        """

    @abc.abstractmethod
    def get(self, size: int, timeout: float) -> list[TimerRequest]:
        """
        Gets up to ``size`` number of timer requests in a blocking fashion
        (no more than ``timeout`` seconds).
        """


class TimerServer(abc.ABC):
    """
    Entity that monitors active timers and expires them
    in a timely fashion. This server is responsible for
    reaping workers that have expired timers.
    """

    def __init__(
        self, request_queue: RequestQueue, max_interval: float, daemon: bool = True
    ):
        """
        :param request_queue: Consumer ``RequestQueue``
        :param max_interval: max time (in seconds) to wait
                             for an item in the request_queue
        :param daemon: whether to run the watchdog thread as a daemon
        """
        super().__init__()
        self._request_queue = request_queue
        self._max_interval = max_interval
        self._daemon = daemon
        self._watchdog_thread: threading.Thread | None = None
        self._stop_signaled = False

    @abc.abstractmethod
    def register_timers(self, timer_requests: list[TimerRequest]) -> None:
        """
        Processes the incoming timer requests and registers them with the server.
        The timer request can either be an acquire-timer or release-timer request.
        Timer requests with a negative expiration_time should be interpreted
        as a release-timer request.
        """

    @abc.abstractmethod
    def clear_timers(self, worker_ids: set[Any]) -> None:
        """
        Clears all timers for the given ``worker_ids``.
        """

    @abc.abstractmethod
    def get_expired_timers(self, deadline: float) -> dict[str, list[TimerRequest]]:
        """
        Returns all expired timers for each worker_id. An expired timer
        is a timer for which the expiration_time is less than or equal to
        the provided deadline.
        """

    @abc.abstractmethod
    def _reap_worker(self, worker_id: Any) -> bool:
        """
        Reaps the given worker. Returns True if the worker has been
        successfully reaped, False otherwise. If any uncaught exception
        is thrown from this method, the worker is considered reaped
        and all associated timers will be removed.
        """

    def _reap_worker_no_throw(self, worker_id: Any) -> bool:
        """
        Wraps ``_reap_worker(worker_id)``, if an uncaught exception is
        thrown, then it considers the worker as reaped.
        """
        try:
            return self._reap_worker(worker_id)
        except Exception:
            logger.exception(
                "Uncaught exception thrown from _reap_worker(), "
                "check that the implementation correctly catches exceptions",
            )
            return True

    def _watchdog_loop(self):
        while not self._stop_signaled:
            try:
                self._run_watchdog()
            except Exception:
                logger.exception("Error running watchdog")

    def _run_watchdog(self):
        batch_size = max(1, self._request_queue.size())
        timer_requests = self._request_queue.get(batch_size, self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_worker_ids = set()
        for worker_id, expired_timers in self.get_expired_timers(now).items():
            logger.info(
                "Reaping worker_id=[%s]. Expired timers: %s",
                worker_id,
                self._get_scopes(expired_timers),
            )
            if self._reap_worker_no_throw(worker_id):
                logger.info("Successfully reaped worker=[%s]", worker_id)
                reaped_worker_ids.add(worker_id)
            else:
                logger.error(
                    "Error reaping worker=[%s]. Will retry on next watchdog.", worker_id
                )
        self.clear_timers(reaped_worker_ids)

    def _get_scopes(self, timer_requests):
        return [r.scope_id for r in timer_requests]

    def start(self) -> None:
        logger.info(
            "Starting %s... max_interval=%s, daemon=%s",
            type(self).__name__,
            self._max_interval,
            self._daemon,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=self._daemon
        )
        logger.info("Starting watchdog thread...")
        self._watchdog_thread.start()

    def stop(self) -> None:
        logger.info("Stopping %s", type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            logger.info("Stopping watchdog thread...")
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            logger.info("No watchdog thread running, doing nothing")


_timer_client: TimerClient | None = None


def configure(timer_client: TimerClient):
    """
    Configures a timer client. Must be called before using ``expires``.
    """
    global _timer_client
    _timer_client = timer_client
    logger.info("Timer client configured to: %s", type(_timer_client).__name__)


@contextmanager
def expires(after: float, scope: str | None = None, client: TimerClient | None = None):
    """
    Acquires a countdown timer that expires in ``after`` seconds from now,
    unless the code-block that it wraps is finished within the timeframe.
    When the timer expires, this worker is eligible to be reaped. The
    exact meaning of "reaped" depends on the client implementation. In
    most cases, reaping means to terminate the worker process.
    Note that the worker is NOT guaranteed to be reaped at exactly
    ``time.now() + after``, but rather the worker is "eligible" for being
    reaped and the ``TimerServer`` that the client talks to will ultimately
    make the decision when and how to reap the workers with expired timers.

    Usage::

        torch.distributed.elastic.timer.configure(LocalTimerClient())
        with expires(after=10):
            torch.distributed.all_reduce(...)
    """
    if client is None:
        if _timer_client is None:
            raise RuntimeError("Configure timer client before using countdown timers.")
        client = _timer_client
    if scope is None:
        # grab the caller file + lineno
        caller = getframeinfo(stack()[1][0])
        scope = f"{caller.filename}#{caller.lineno}"
    expiration = time.time() + after
    client.acquire(scope, expiration)
    try:
        yield
    finally:
        client.release(scope)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import os
import select
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec

from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
from torch.distributed.elastic.timer.debug_info_logging import (
    log_debug_info_for_expired_timers,
)
from torch.distributed.elastic.utils.logging import get_logger


_P = ParamSpec("_P")
_R = TypeVar("_R")

__all__ = ["FileTimerClient", "FileTimerRequest", "FileTimerServer"]

logger = get_logger(__name__)


def _retry(max_retries: int, sleep_time: float) -> Callable:
    """
    A simple retry wrapper.

    Args:
        max_retries: int, the maximum number of retries.
        sleep_time: float, the time to sleep between retries.
    """

    def wrapper(func: Callable[_P, _R]) -> Callable[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    logger.exception("Error running %s. Retrying...", func.__name__)
                    if i < max_retries - 1:
                        time.sleep(sleep_time)
                    else:
                        raise

        return wrapper

    return wrapper


class FileTimerRequest(TimerRequest):
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``FileTimerClient`` and ``FileTimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.
    ``signal`` is the signal to reap the worker process from the server
    process.
    """

    __slots__ = ["version", "signal"]

    def __init__(
        self, worker_pid: int, scope_id: str, expiration_time: float, signal: int = 0
    ) -> None:
        super().__init__(
            worker_id=worker_pid, scope_id=scope_id, expiration_time=expiration_time
        )
        self.version = 1
        self.signal = signal

    @property
    def worker_pid(self) -> int:
        return self.worker_id

    def __eq__(self, other) -> bool:
        if isinstance(other, FileTimerRequest):
            return (
                super().__eq__(other)
                and self.version == other.version
                and self.signal == other.signal
            )
        return False

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "pid": self.worker_pid,
                "scope_id": self.scope_id,
                "expiration_time": self.expiration_time,
                "signal": self.signal,
            },
        )


class FileTimerClient(TimerClient):
    """
    Client side of ``FileTimerServer``. This client is meant to be used
    on the same host that the ``FileTimerServer`` is running on and uses
    pid to uniquely identify a worker.
    This client uses a named_pipe to send timer requests to the
    ``FileTimerServer``. This client is a producer while the
    ``FileTimerServer`` is a consumer. Multiple clients can work with
    the same ``FileTimerServer``.

    Args:

        file_path: str, the path of a FIFO special file. ``FileTimerServer``
                        must have created it by calling os.mkfifo().

        signal: signal, the signal to use to kill the process. Using a
                        negative or zero signal will not kill the process.
    """

    def __init__(
        self,
        file_path: str,
        signal=(signal.SIGKILL if sys.platform != "win32" else signal.CTRL_C_EVENT),  # type: ignore[attr-defined]
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self.signal = signal

    @_retry(max_retries=10, sleep_time=0.1)
    def _open_non_blocking(self) -> io.TextIOWrapper | None:
        # The server may have crashed or may haven't started yet.
        # In such case, calling open() in blocking model blocks the client.
        # To avoid such issue, open it in non-blocking mode, and an OSError will
        # be raised if the server is not there.
        fd = os.open(self._file_path, os.O_WRONLY | os.O_NONBLOCK)
        return os.fdopen(fd, "wt")

    def _send_request(self, request: FileTimerRequest) -> None:
        try:
            file = self._open_non_blocking()
        except Exception as e:
            raise BrokenPipeError(
                "Could not send the FileTimerRequest because FileTimerServer is not available."
            ) from e
        with file:
            json_request = request.to_json()
            # Write request with no greater than select.PIPE_BUF is guarantee to be atomic.
            if len(json_request) > select.PIPE_BUF:
                raise RuntimeError(
                    f"FileTimerRequest larger than {select.PIPE_BUF} bytes "
                    f"is not supported: {json_request}"
                )
            file.write(json_request + "\n")

    def acquire(self, scope_id: str, expiration_time: float) -> None:
        self._send_request(
            request=FileTimerRequest(
                worker_pid=os.getpid(),
                scope_id=scope_id,
                expiration_time=expiration_time,
                signal=self.signal,
            ),
        )

    def release(self, scope_id: str) -> None:
        self._send_request(
            request=FileTimerRequest(
                worker_pid=os.getpid(), scope_id=scope_id, expiration_time=-1, signal=0
            ),
        )


class FileTimerServer:
    """
    Server that works with ``FileTimerClient``. Clients are expected to be
    running on the same host as the process that is running this server.
    Each host in the job is expected to start its own timer server locally
    and each server instance manages timers for local workers (running on
    processes on the same host).

    Args:

        file_path: str, the path of a FIFO special file to be created.

        max_interval: float, max interval in seconds for each watchdog loop.

        daemon: bool, running the watchdog thread in daemon mode or not.
                      A daemon thread will not block a process to stop.
        log_event: Callable[[Dict[str, str]], None], an optional callback for
                logging the events in JSON format.
    """

    def __init__(
        self,
        file_path: str,
        run_id: str,
        max_interval: float = 10,
        daemon: bool = True,
        log_event: Callable[[str, FileTimerRequest | None], None] | None = None,
    ) -> None:
        self._file_path = file_path
        self._run_id = run_id
        self._max_interval = max_interval
        self._daemon = daemon
        self._timers: dict[tuple[int, str], FileTimerRequest] = {}
        self._stop_signaled = False
        self._watchdog_thread: threading.Thread | None = None

        self._is_client_started = False
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        os.mkfifo(self._file_path)
        # For test only. Count the number of requests received.
        self._request_count = 0
        # For test only. Process all requests and stop the server.
        self._run_once = False
        self._log_event = (
            log_event if log_event is not None else lambda name, request: None
        )
        self._last_progress_time = int(time.time())

    def start(self) -> None:
        logger.info(
            "Starting %s... max_interval=%s, daemon=%s, file_path=%s",
            type(self).__name__,
            self._max_interval,
            self._daemon,
            self._file_path,
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=self._daemon
        )
        logger.info("Starting watchdog thread...")
        self._watchdog_thread.start()
        self._log_event("watchdog started", None)

    def stop(self) -> None:
        logger.info("Stopping %s", type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            logger.info("Stopping watchdog thread...")
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            logger.info("No watchdog thread running, doing nothing")
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        self._log_event("watchdog stopped", None)

    def run_once(self) -> None:
        self._run_once = True
        if self._watchdog_thread:
            logger.info("Stopping watchdog thread...")
            self._watchdog_thread.join()
            self._watchdog_thread = None
        else:
            logger.info("No watchdog thread running, doing nothing")
        if os.path.exists(self._file_path):
            os.remove(self._file_path)

    @staticmethod
    def is_process_running(pid: int):
        """
        function to check process is running or not
        """
        try:
            # Check if the process exists and we can send signals to it
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _watchdog_loop(self) -> None:
        # Open the pipe in blocking mode blocks the server thread.
        # This is fine for the following reasons:
        #  1. No client case usually does not happen.
        #  2. We are running the watchdog loop in a separate daemon
        #     thread, which will not block the process to stop.
        try:
            with open(self._file_path) as fd:
                self._is_client_started = True
                while not self._stop_signaled:
                    try:
                        run_once = self._run_once
                        self._run_watchdog(fd)
                        if run_once:
                            break
                        self._last_progress_time = int(time.time())
                    except Exception:
                        logger.exception("Error running watchdog")

        except Exception:
            logger.exception("Could not open the FileTimerServer pipe")
            raise

    def _run_watchdog(self, fd: io.TextIOWrapper) -> None:
        timer_requests = self._get_requests(fd, self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_worker_pids = set()
        kill_process = False
        reap_signal = 0

        all_expired_timers = self.get_expired_timers(now)
        log_debug_info_for_expired_timers(
            self._run_id,
            {
                pid: [expired_timer.to_json() for expired_timer in expired_timers]
                for pid, expired_timers in all_expired_timers.items()
            },
        )

        for worker_pid, expired_timers in all_expired_timers.items():
            logger.info(
                "Reaping worker_pid=[%s]. Expired timers: %s",
                worker_pid,
                self._get_scopes(expired_timers),
            )
            reaped_worker_pids.add(worker_pid)
            # In case we have multiple expired timers, we find the first timer
            # with a valid signal (>0) in the expiration time order.
            expired_timers.sort(key=lambda timer: timer.expiration_time)
            signal = 0
            expired_timer = None
            for timer in expired_timers:
                self._log_event("timer expired", timer)
                if timer.signal > 0:
                    signal = timer.signal
                    expired_timer = timer
                    break
            if signal <= 0:
                logger.info(
                    "No signal specified with worker=[%s]. Do not reap it.", worker_pid
                )
                continue
            if self._reap_worker(worker_pid, signal):
                logger.info(
                    "Successfully reaped worker=[%s] with signal=%s", worker_pid, signal
                )
                self._log_event("kill worker process", expired_timer)
                kill_process = True
                reap_signal = signal
            else:
                logger.error(
                    "Error reaping worker=[%s]. Will retry on next watchdog.",
                    worker_pid,
                )
        if kill_process and reap_signal > 0:
            logger.info(
                "Terminating the server process=[%s] because of expired timers",
                os.getpid(),
            )
            self._reap_worker(os.getpid(), reap_signal)

        self.clear_timers(reaped_worker_pids)

    def _get_scopes(self, timer_requests: list[FileTimerRequest]) -> list[str]:
        return [r.scope_id for r in timer_requests]

    def _get_requests(
        self, fd: io.TextIOWrapper, max_interval: float
    ) -> list[FileTimerRequest]:
        start = time.time()
        requests = []
        while not self._stop_signaled or self._run_once:
            # For named pipe, readline() is blocking when at least one writer opens.
            # It returns only when flush() is called at the writer side.
            # Note that flush() is automatically called inside close().
            # After the last writer closes, readline() is not blocking.
            # It will return an empty string when it's at end-of-file.
            # Since the client side always opens the pipe, writes a message and closes
            # the pipe immediately, the readline() call below is not blocking for long.
            json_request = fd.readline()
            if len(json_request) == 0:
                if self._run_once:
                    break
                time.sleep(min(max_interval, 1))
            else:
                request = json.loads(json_request)
                pid = request["pid"]
                scope_id = request["scope_id"]
                expiration_time = request["expiration_time"]
                signal = request["signal"]
                requests.append(
                    FileTimerRequest(
                        worker_pid=pid,
                        scope_id=scope_id,
                        expiration_time=expiration_time,
                        signal=signal,
                    )
                )
            now = time.time()
            if now - start > max_interval:
                break
        return requests

    def register_timers(self, timer_requests: list[FileTimerRequest]) -> None:
        for request in timer_requests:
            pid = request.worker_pid
            scope_id = request.scope_id
            expiration_time = request.expiration_time
            self._request_count += 1

            key = (pid, scope_id)
            # negative expiration is a proxy for a release call
            if expiration_time < 0:
                if key in self._timers:
                    del self._timers[key]
            else:
                self._timers[key] = request

    def clear_timers(self, worker_pids: set[int]) -> None:
        for pid, scope_id in list(self._timers.keys()):
            if pid in worker_pids or not FileTimerServer.is_process_running(pid):
                del self._timers[(pid, scope_id)]

    def get_expired_timers(self, deadline: float) -> dict[int, list[FileTimerRequest]]:
        # pid -> [timer_requests...]
        expired_timers: dict[int, list[FileTimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_pid, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_pid: int, signal: int) -> bool:
        try:
            os.kill(worker_pid, signal)
            return True
        except ProcessLookupError:
            logger.info("Process with pid=%s does not exist. Skipping", worker_pid)
            return True
        except Exception:
            logger.exception("Error terminating pid=%s", worker_pid)
        return False

    def get_last_progress_time(self) -> int:
        return self._last_progress_time if self._is_client_started else int(time.time())
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import signal
import socket
import tempfile
import time
import uuid
from string import Template
from typing import Any, TYPE_CHECKING

import torch.distributed.elastic.timer as timer
from torch._utils_internal import justknobs_check
from torch.distributed.elastic import events
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.health_check_server import (
    create_healthcheck_server,
    HealthCheckServer,
)
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.multiprocessing import (
    LogsSpecs,
    PContext,
    start_processes,
)
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.elastic.utils.process_state import (
    is_uninterruptible_state,
    read_proc_state,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.elastic.events.api import EventMetadataValue

logger = get_logger(__name__)

__all__ = [
    "LocalElasticAgent",
    "TORCHELASTIC_ENABLE_FILE_TIMER",
    "TORCHELASTIC_TIMER_FILE",
    "TORCHELASTIC_HEALTH_CHECK_PORT",
    "TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT",
]

TORCHELASTIC_ENABLE_FILE_TIMER = "TORCHELASTIC_ENABLE_FILE_TIMER"
TORCHELASTIC_HEALTH_CHECK_PORT = "TORCHELASTIC_HEALTH_CHECK_PORT"
TORCHELASTIC_TIMER_FILE = "TORCHELASTIC_TIMER_FILE"
# Seconds a worker may spend in Linux uninterruptible kernel sleep (D-state)
# before the agent marks the worker group UNHEALTHY. Used as the default when
# ``uninterruptible_state_timeout`` is not passed to ``LocalElasticAgent``.
# Unset / non-positive disables the check. Linux-only.
TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT = (
    "TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT"
)


def _resolve_uninterruptible_state_timeout(explicit: float | None) -> float:
    """Resolve the uninterruptible-state timeout once.

    Precedence: explicit constructor arg > ``TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT``
    env var > 0 (disabled). Non-positive / unparsable values disable the check.
    """
    if explicit is not None:
        return max(0.0, float(explicit))
    raw = os.environ.get(TORCHELASTIC_UNINTERRUPTIBLE_STATE_TIMEOUT, "")
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, value)


class _AliveCallbackProxy:
    """Mutable callback wrapper for the health check server.

    The C++ pybind ``HealthCheckThriftServer`` binds its ``alive_callback``
    at construction time and cannot update it afterward.  This proxy is
    created *before* the health check server so it can be passed as the
    callback.  Initially it returns ``time.time()`` (signalling "alive").
    After the agent is constructed, :meth:`set_delegate` wires it to
    ``agent._get_alive_time`` for real liveness tracking.
    """

    def __init__(self) -> None:
        self._delegate: Callable[[], int] | None = None

    def __call__(self) -> int:
        if self._delegate is not None:
            return self._delegate()
        return int(time.time())

    def set_delegate(self, delegate: Callable[[], int]) -> None:
        self._delegate = delegate


class LocalElasticAgent(SimpleElasticAgent):
    """An implementation of :py:class:`torchelastic.agent.server.ElasticAgent` that handles host-local workers.

    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.


    The worker function and argument passed to the worker function must be
    python multiprocessing compatible. To pass multiprocessing data structures
    to the workers you may create the data structure in the same multiprocessing
    context as the specified ``start_method`` and pass it as a function argument.

    The ``exit_barrier_timeout`` specifies the amount of time (in seconds) to wait
    for other agents to finish. This acts as a safety net to handle cases where
    workers finish at different times, to prevent agents from viewing workers
    that finished early as a scale-down event. It is strongly advised that the
    user code deal with ensuring that workers are terminated in a synchronous
    manner rather than relying on the exit_barrier_timeout.

    A named pipe based watchdog can be enabled in ```LocalElasticAgent``` if an
    environment variable ``TORCHELASTIC_ENABLE_FILE_TIMER`` with value 1 has
    been defined in the ```LocalElasticAgent``` process.
    Optionally, another environment variable ```TORCHELASTIC_TIMER_FILE```
    can be set with a unique file name for the named pipe. If the environment
    variable ```TORCHELASTIC_TIMER_FILE``` is not set, ```LocalElasticAgent```
    will internally create a unique file name and set it to the environment
    variable ```TORCHELASTIC_TIMER_FILE```, and this environment variable will
    be propagated to the worker processes to allow them to connect to the same
    named pipe that ```LocalElasticAgent``` uses.

    Logs are written to the specified log directory. Each log line will be by default
    prefixed by ``[${role_name}${local_rank}]:`` (e.g. ``[trainer0]: foobar``).
    Log prefixes can be customized by passing a `template string
    <https://docs.python.org/3/library/string.html#template-strings>`_ as the
    ``log_line_prefix_template`` argument.
    The following macros (identifiers) are substituted at runtime:
    ``${role_name}, ${local_rank}, ${rank}, ${hostname}``.
    For example, to prefix each log line with
    global rank instead of the local rank, set ``log_line_prefix_template = "[${rank}]:``.
    ``${hostname}`` expands to the name of the node the agent runs on, which
    identifies the offending node in a multi-node job; for example
    ``log_line_prefix_template = "${hostname}:${rank}: "`` renders as
    ``r12i0n8:3: foobar``.


    Example launching function

    ::

        def trainer(args) -> str:
            return "do train"

        def main():
            start_method="spawn"
            shared_queue= multiprocessing.get_context(start_method).Queue()
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint=trainer,
                        args=("foobar",),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec, start_method)
            results = agent.run()

            if results.is_failed():
                print("trainer failed")
            else:
                print(f"rank 0 return value: {results.return_values[0]}")
                # prints -> rank 0 return value: do train

    Example launching binary

    ::

        def main():
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint="/usr/local/bin/trainer",
                        args=("--trainer-args", "foobar"),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec)
            results = agent.run()

            if not results.is_failed():
                print("binary launches do not have return values")

    """

    def __init__(
        self,
        spec: WorkerSpec,
        logs_specs: LogsSpecs,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_line_prefix_template: str | None = None,
        shutdown_timeout: int = 30,
        health_check_server: HealthCheckServer | None = None,
        uninterruptible_state_timeout: float | None = None,
    ):
        super().__init__(spec, exit_barrier_timeout, shutdown_timeout)
        self._start_method = start_method
        self._pcontext: PContext | None = None
        self._rdzv_handler = spec.rdzv_handler
        self._log_line_prefix_template = log_line_prefix_template
        self._worker_watchdog: timer.FileTimerServer | None = None
        self._logs_specs = logs_specs
        self._health_check_server = health_check_server
        # Resolve the uninterruptible-state timeout once at construction time.
        # Explicit constructor arg wins; otherwise fall back to the env var.
        # Non-positive / unparsable values disable the check.
        self._uninterruptible_state_timeout: float = (
            _resolve_uninterruptible_state_timeout(uninterruptible_state_timeout)
        )
        # Maps PID -> monotonic timestamp when the PID was first observed in
        # an uninterruptible (D-state) sleep. Cleared per-PID when the worker
        # leaves that state or exits.
        self._uninterruptible_state_first_seen: dict[int, float] = {}

    def _setup_local_watchdog(self, envs: dict[int, dict[str, str]]) -> None:
        enable_watchdog_env_name = TORCHELASTIC_ENABLE_FILE_TIMER
        watchdog_enabled = os.getenv(enable_watchdog_env_name)
        watchdog_file_env_name = TORCHELASTIC_TIMER_FILE
        watchdog_file_path = os.getenv(watchdog_file_env_name)
        if watchdog_enabled is not None and str(watchdog_enabled) == "1":
            if watchdog_file_path is None:
                watchdog_file_path = os.path.join(
                    tempfile.gettempdir(), "watchdog_timer_" + str(uuid.uuid4())
                )
            logger.info("Starting a FileTimerServer with %s ...", watchdog_file_path)
            if not envs:
                logger.warning(
                    "Empty envs variables, using empty run_id for FileTimerServer"
                )
                run_id = ""
            else:
                run_id = envs[0]["TORCHELASTIC_RUN_ID"]
            self._worker_watchdog = timer.FileTimerServer(
                file_path=watchdog_file_path,
                run_id=run_id,
                max_interval=0.1,
                daemon=True,
                log_event=self._log_watchdog_event,
            )
            self._worker_watchdog.start()
            logger.info("FileTimerServer started")
        else:
            logger.info(
                "Environment variable '%s' not found. Do not start FileTimerServer.",
                enable_watchdog_env_name,
            )
        # Propagate the watchdog file env to worker processes
        if watchdog_file_path is not None:
            for worker_env in envs.values():
                worker_env[watchdog_file_env_name] = watchdog_file_path

    @staticmethod
    def _get_current_time_secs() -> int:
        return int(time.time())

    def _get_alive_time(self) -> int:
        """Return the last progress time from the watchdog, or the current time.

        This callback is passed to the health check server at startup and
        is called on every TW health check poll. During initialization
        (before rendezvous and worker launch), the watchdog does not exist
        yet, so we return the current time to signal the agent is alive.
        Once workers are running and the watchdog is active, we delegate
        to the watchdog's ``get_last_progress_time`` for real liveness
        tracking.

        During the exit barrier wait, workers have finished and the watchdog
        progress time is stale. We return the current time to prevent TW
        from killing the task while agents coordinate shutdown.
        """
        if self._in_exit_barrier:
            return int(time.time())
        if self._worker_watchdog is not None:
            return self._worker_watchdog.get_last_progress_time()
        return int(time.time())

    def _setup_healthcheck(self) -> None:
        healthcheck_port_env_name = TORCHELASTIC_HEALTH_CHECK_PORT
        healthcheck_port = os.getenv(healthcheck_port_env_name)
        if healthcheck_port is not None:
            logger.info(
                "Found healthcheck port %s: %s",
                healthcheck_port_env_name,
                healthcheck_port,
            )

            if justknobs_check(
                "ai_infra/pytorch_distributed:torchelastic_enable_healthcheck_before_rendezvous",
                default=False,
            ):
                # New behavior: idempotent guard + dynamic callback that
                # returns current time before watchdog exists and delegates
                # to watchdog once workers are running.
                if self._health_check_server is not None:
                    return
                alive_callback = self._get_alive_time
            else:
                # Original behavior: pick callback based on watchdog state
                # at call time (only called from _start_workers where
                # watchdog is already set up).
                if self._worker_watchdog is None:
                    logger.info(
                        "FileTimerServer doesn't exist, using current time as dummy callback"
                    )
                    alive_callback = LocalElasticAgent._get_current_time_secs
                else:
                    alive_callback = self._worker_watchdog.get_last_progress_time

            try:
                healthcheck_port_as_int = int(healthcheck_port)
                self._health_check_server = create_healthcheck_server(
                    alive_callback=alive_callback,
                    port=healthcheck_port_as_int,
                    timeout=60,
                )
                self._health_check_server.start()
            except ValueError:
                logger.info(
                    "Invalid healthcheck port value: '%s', expecting integer. Not starting healthcheck server.",
                    healthcheck_port,
                )
        else:
            logger.info(
                "Environment variable '%s' not found. Do not start health check.",
                healthcheck_port_env_name,
            )

    def _get_fq_hostname(self) -> str:
        return socket.getfqdn(socket.gethostname())

    def _log_watchdog_event(
        self,
        name: str,
        request: timer.FileTimerRequest | None,
    ) -> None:
        wg = self._worker_group
        spec = wg.spec
        md = {"watchdog_event": name}
        if request is not None:
            md["worker_pid"] = str(request.worker_pid)
            md["scope_id"] = request.scope_id
            md["expiration_time"] = str(request.expiration_time)
            md["signal"] = str(request.signal)
        md_str = json.dumps(md)
        state = "RUNNING"
        metadata: dict[str, EventMetadataValue] = {
            "run_id": spec.rdzv_handler.get_run_id(),
            "global_rank": None,
            "group_rank": wg.group_rank,
            "worker_id": None,
            "role": spec.role,
            "hostname": self._get_fq_hostname(),
            "state": state,
            "total_run_time": self._total_execution_time,
            "rdzv_backend": spec.rdzv_handler.get_backend(),
            "raw_error": None,
            "metadata": md_str,
            "agent_restarts": spec.max_restarts - self._remaining_restarts,
        }
        # Note: The 'metadata' field of the Event is converted to a TorchelasticStatusLogEntry later.
        #       The 'name' field of the Event is NOT used in the TorchelasticStatusLogEntry.
        event = events.Event(
            name=name, source=events.EventSource.AGENT, metadata=metadata
        )
        events.record(event, self._worker_group.spec.event_log_handler)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        self._shutdown()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        if store is None:
            raise AssertionError
        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store: bool = spec.rdzv_handler.use_agent_store
        logger.info("use_agent_store: %s", use_agent_store)

        args: dict[int, tuple] = {}
        envs: dict[int, dict[str, str]] = {}
        log_line_prefixes: dict[int, str] | None = (
            {} if self._log_line_prefix_template else None
        )
        # Short name (e.g. "r12i0n8") rather than _get_fq_hostname(): the fq name
        # eats horizontal space in every log line, and degrades to an unhelpful
        # reverse-DNS record (e.g. "...ip6.arpa") when the node has no PTR entry.
        hostname = socket.gethostname() if self._log_line_prefix_template else ""
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": worker_group.master_addr,
                "MASTER_PORT": str(worker_group.master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING", str(1)
                ),
            }
            self._set_local_rank_env(worker_env, local_rank, spec)
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            if self._log_line_prefix_template:
                log_line_prefix = Template(
                    self._log_line_prefix_template
                ).safe_substitute(
                    role_name=spec.role,
                    rank=worker.global_rank,
                    local_rank=local_rank,
                    hostname=hostname,
                )
                # pyrefly: ignore [unsupported-operation]
                log_line_prefixes[local_rank] = log_line_prefix

            # pyrefly: ignore [unsupported-operation]
            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        self._setup_local_watchdog(envs=envs)
        self._setup_healthcheck()

        if spec.entrypoint is None:
            raise AssertionError
        if self._logs_specs is None:
            raise AssertionError
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            logs_specs=self._logs_specs,
            log_line_prefixes=log_line_prefixes,
            start_method=self._start_method,
            numa_options=spec.numa_options,
            duplicate_stdout_filters=spec.duplicate_stdout_filters,
            duplicate_stderr_filters=spec.duplicate_stderr_filters,
        )

        return self._pcontext.pids()

    def _set_local_rank_env(
        self, worker_env: dict[str, str | None], local_rank: int, spec: WorkerSpec
    ) -> None:
        # Set CUDA_VISIBLE_DEVICES and LOCAL_RANK based on virtual_local_rank mode.
        # Virtual mode: Each worker sees only its assigned GPU as device 0, LOCAL_RANK=0
        # Traditional mode: Workers see all GPUs, LOCAL_RANK matches actual local rank

        if spec.virtual_local_rank:
            # Set LOCAL_RANK=0 and use CUDA_VISIBLE_DEVICES to control the actual GPU access.

            worker_env["LOCAL_RANK"] = "0"

            # Map local_rank through existing CUDA_VISIBLE_DEVICES
            # HIP uses CUDA_VISIBLE_DEVICES as a compatibility hack:
            # https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html#cuda-visible-devices
            parent_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
            if parent_visible_devices is not None:
                # Parse comma-separated list of GPU IDs
                available_gpus = parent_visible_devices.split(",")
                if local_rank >= len(available_gpus):
                    raise ValueError(
                        f"local_rank {local_rank} exceeds available GPUs in "
                        f"CUDA_VISIBLE_DEVICES={parent_visible_devices}"
                    )

                visible_gpu = available_gpus[local_rank].strip()
            else:
                # No restriction, use local_rank directly
                visible_gpu = str(local_rank)

            worker_env["CUDA_VISIBLE_DEVICES"] = visible_gpu
            return

        # In traditional mode, don't override CUDA_VISIBLE_DEVICES
        # (inherit from parent environment)
        worker_env["LOCAL_RANK"] = str(local_rank)

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            worker_env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

    def _shutdown(
        self, death_sig: signal.Signals = signal.SIGTERM, timeout: int = 30
    ) -> None:
        if self._worker_watchdog is not None:
            self._worker_watchdog.stop()
            self._worker_watchdog = None
        if self._health_check_server is not None:
            self._health_check_server.stop()
            self._health_check_server = None
        if self._pcontext:
            self._pcontext.close(death_sig, timeout)

    def _check_uninterruptible_state_timeout(
        self, worker_group: WorkerGroup, timeout: float
    ) -> RunResult | None:
        """Return UNHEALTHY when any worker has been in Linux uninterruptible
        sleep (D-state) for at least ``timeout`` seconds; otherwise update
        bookkeeping and return None.
        """
        if self._pcontext is None:
            return None
        role = worker_group.spec.role
        live_pids = set(self._pcontext.pids().values())
        # Drop bookkeeping for pids that have exited since the last check.
        for pid in list(self._uninterruptible_state_first_seen):
            if pid not in live_pids:
                self._uninterruptible_state_first_seen.pop(pid, None)

        timed_out: list[tuple[int, float]] = []
        for pid in live_pids:
            elapsed = self._update_uninterruptible_dwell(pid, role, timeout)
            if elapsed is not None and elapsed >= timeout:
                timed_out.append((pid, elapsed))

        if not timed_out:
            return None
        for pid, elapsed in timed_out:
            logger.error(
                "[%s] Worker pid=%s stuck in uninterruptible sleep (D-state)"
                " for %.1fs (>= %.1fs); marking worker group UNHEALTHY and"
                " disabling restarts.",
                role,
                pid,
                elapsed,
                timeout,
            )
        # Disable restarts: a wedged worker is holding a kernel resource
        # (GPU, NIC) that a fresh worker on the same host would conflict
        # with. Force the supervising loop into the _stop_workers branch
        # so the agent can exit promptly.
        self._remaining_restarts = 0
        return RunResult(state=WorkerState.UNHEALTHY)

    def _update_uninterruptible_dwell(
        self, pid: int, role: str, timeout: float
    ) -> float | None:
        """Update bookkeeping for ``pid`` and return how long (seconds) it has
        been continuously in uninterruptible sleep, or ``None`` if it isn't
        (or its state can't be read).
        """
        state = read_proc_state(pid)
        if state is None:
            return None
        if not is_uninterruptible_state(state):
            self._uninterruptible_state_first_seen.pop(pid, None)
            return None
        first = self._uninterruptible_state_first_seen.get(pid)
        if first is None:
            self._uninterruptible_state_first_seen[pid] = time.monotonic()
            logger.warning(
                "[%s] Worker pid=%s entered uninterruptible sleep (D-state);"
                " will mark UNHEALTHY if it remains for %.1fs.",
                role,
                pid,
                timeout,
            )
            return 0.0
        return time.monotonic() - first

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        role = worker_group.spec.role
        worker_pids = {w.id for w in worker_group.workers}
        if self._pcontext is None:
            raise AssertionError
        pc_pids = set(self._pcontext.pids().values())
        if worker_pids != pc_pids:
            logger.error(
                "[%s] worker pids do not match process_context pids."
                " Expected: %s, actual: %s",
                role,
                worker_pids,
                pc_pids,
            )
            return RunResult(state=WorkerState.UNKNOWN)

        result = self._pcontext.wait(0)
        if result:
            if result.is_failed():
                # map local rank failure to global rank
                worker_failures = {}
                for local_rank, failure in result.failures.items():
                    worker = worker_group.workers[local_rank]
                    worker_failures[worker.global_rank] = failure
                return RunResult(
                    state=WorkerState.FAILED,
                    failures=worker_failures,
                )
            else:
                # copy ret_val_queue into a map with global ranks
                workers_ret_vals = {}
                for local_rank, ret_val in result.return_values.items():
                    worker = worker_group.workers[local_rank]
                    workers_ret_vals[worker.global_rank] = ret_val
                return RunResult(
                    state=WorkerState.SUCCEEDED,
                    return_values=workers_ret_vals,
                )
        else:
            timeout = self._uninterruptible_state_timeout
            if timeout > 0:
                ustate_result = self._check_uninterruptible_state_timeout(
                    worker_group, timeout
                )
                if ustate_result is not None:
                    return ustate_result
            return RunResult(state=WorkerState.HEALTHY)
<|endoftext|># mypy: ignore-errors

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import json
import os
import signal
import socket
import time
import traceback
import warnings
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import ProcessFailure, SignalException
from torch.distributed.elastic.rendezvous import RendezvousGracefulExitError
from torch.distributed.elastic.utils.logging import get_logger
from torch.numa.binding import NumaOptions


__all__ = [
    "WorkerSpec",
    "Worker",
    "WorkerState",
    "WorkerGroup",
    "RunResult",
    "ElasticAgent",
    "SimpleElasticAgent",
]
_TERMINAL_STATE_SYNC_ID = "torchelastic/agent/terminal_state"

DEFAULT_ROLE = "default"
logger = get_logger(__name__)


@dataclass
class WorkerSpec:
    """
    Blueprint information about a particular type of worker.

    For a given role, there must only exist a single worker spec.
    Worker spec is expected to be homogeneous across all nodes (machine),
    that is each node runs the same number of workers for a particular spec.

    Args:
        role: user-defined role for the workers with this spec
        local_world_size: number local workers to run
        fn: (deprecated use entrypoint instead)
        entrypoint: worker function or command
        args: arguments to pass to ``entrypoint``
        rdzv_handler: handles rdzv for this set of workers
        max_restarts: number of max retries for the workers
        monitor_interval: monitor status of workers every ``n`` seconds
        master_port: fixed port to run the c10d store on rank 0
                     if not specified then will choose a random free port
        master_addr: fixed master_addr to run the c10d store on rank 0
                     if not specified then will choose hostname on agent rank 0
        redirects: redirect std streams to a file,
                   selectively redirect for a particular
                   local rank by passing a map
        tee: tees the specified std stream(s) to console + file,
             selectively tee for a particular local rank by passing a map,
             takes precedence over ``redirects`` settings.
        event_log_handler: name of the event logging handler as registered in
          `elastic/events/handlers.py <https://docs.pytorch.org/docs/stable/elastic/events.html>`_.
        duplicate_stdout_filters: If non-empty, duplicates stdout to a file containing only lines
                                 that match _any_ of the filter strings.
        duplicate_stderr_filters: If non-empty, duplicates stderr to a file containing only lines
                                 that match _any_ of the filter strings.
        virtual_local_rank: Enable virtual local rank mode for workers (defaults to False).
                            When enabled, LOCAL_RANK is set to 0 for all workers and
                            CUDA_VISIBLE_DEVICES is adjusted so each worker accesses its
                            assigned GPU at device index 0.
    """

    role: str
    local_world_size: int
    rdzv_handler: rdzv.RendezvousHandler
    fn: Callable | None = None
    # TODO @kiuk - make entrypoint a required field
    entrypoint: Callable | str | None = None
    args: tuple = ()
    max_restarts: int = 3
    monitor_interval: float = 0.1
    master_port: int | None = None
    master_addr: str | None = None
    local_addr: str | None = None
    event_log_handler: str = "null"
    numa_options: NumaOptions | None = None
    duplicate_stdout_filters: list[str] | None = None
    duplicate_stderr_filters: list[str] | None = None
    virtual_local_rank: bool = False

    def __post_init__(self):
        if self.local_world_size <= 0:
            raise AssertionError
        if self.monitor_interval <= 0:
            raise AssertionError

        if self.fn:
            warnings.warn(
                "WorkerSpec.fn will be deprecated,"
                " please use WorkerSpec.entrypoint instead",
                stacklevel=2,
                category=DeprecationWarning,
            )
            self.entrypoint = self.fn
        if not self.entrypoint:
            raise AssertionError

    def get_entrypoint_name(self):
        """Get the entry point name.

        If the entrypoint is a function (e.g. ``Callable``) returns its ``__qualname__``
        else if the entrypoint is a binary (e.g. ``str``), returns the binary name.
        """
        if isinstance(self.entrypoint, str):
            return os.path.basename(self.entrypoint)
        else:
            if self.entrypoint is None:
                raise AssertionError
            return self.entrypoint.__qualname__


class Worker:
    """A worker instance.

    Contrast this with ``WorkerSpec`` that represents the specifications of a
    worker. A ``Worker`` is created from a ``WorkerSpec``. A ``Worker`` is to
    a ``WorkerSpec`` as an object is to a class.

    The ``id`` of the worker is interpreted
    by the specific implementation of ``ElasticAgent``. For a local
    agent, it could be the ``pid (int)`` of the worker, for a remote
    agent it could be encoded as ``host:port (string)``.

    Args:
        id (Any): uniquely identifies a worker (interpreted by the agent)
        local_rank (int): local rank of the worker
        global_rank (int): global rank of the worker
        role_rank (int): rank of the worker across all workers that have the same role
        world_size (int): number of workers (globally)
        role_world_size (int): number of workers that have the same role
    """

    __slots__ = [
        "id",
        "local_rank",
        "global_rank",
        "role_rank",
        "world_size",
        "role_world_size",
    ]

    def __init__(
        self,
        local_rank: int,
        global_rank: int = -1,
        role_rank: int = -1,
        world_size: int = -1,
        role_world_size: int = -1,
    ):
        # unique identifier for this worker
        self.id: Any = None

        # rank of the worker among workers with the same role being monitored
        # by the same ``agent`` instance.
        self.local_rank: int = local_rank

        #  rank of the worker among all the workers across all roles
        #  across all ``agent`` instances.
        #  Global rank is not stable between re-rendezvous.
        self.global_rank: int = global_rank

        #  rank of the worker among all the workers with the same role
        #  across all ``agent`` instances.
        #  Role rank is not stable between re-rendezvous.
        self.role_rank: int = role_rank

        # total number of workers (globally). Due to elasticity
        # the world size may change between re-rendezvous.
        self.world_size: int = world_size

        # total number of workers that share the same role. Due to elasticity
        # the role world size may change between re-rendezvous.
        self.role_world_size: int = role_world_size

    def __str__(self):
        return (
            f"local_rank={self.local_rank},global_rank={self.global_rank}"
            f",role_rank={self.role_rank},world_size={self.world_size}"
            f",role_world_size={self.role_world_size}"
        )

    def __repr__(self):
        return str(self)


class WorkerState(str, Enum):
    """A state of the ``WorkerGroup``.

    Workers in a worker group change state as a unit. If a single worker
    in a worker group fails the entire set is considered failed::

      UNKNOWN - agent lost track of worker group state, unrecoverable
      INIT - worker group object created not yet started
      HEALTHY - workers running and healthy
      UNHEALTHY - workers running and unhealthy
      STOPPED - workers stopped (interrupted) by the agent
      SUCCEEDED - workers finished running (exit 0)
      FAILED - workers failed to successfully finish (exit !0)


    A worker group starts from an initial ``INIT`` state,
    then progresses to ``HEALTHY`` or ``UNHEALTHY`` states,
    and finally reaches a terminal ``SUCCEEDED`` or ``FAILED`` state.

    Worker groups can be interrupted and temporarily put into ``STOPPED`` state
    by the agent. Workers in ``STOPPED`` state are scheduled to be restarted
    in the near future by the agent. Some examples of workers being put into
    ``STOPPED`` state are:

    1. Worker group failure|unhealthy observed
    2. Membership change detected

    When actions (start, stop, rdzv, retry, etc) on worker group fail
    and results in the action being partially applied to the worker group
    the state will be ``UNKNOWN``. Typically this happens on uncaught/unhandled
    exceptions during state change events on the agent. The agent is not
    expected to recover worker groups in ``UNKNOWN`` state and is better off
    self terminating and allowing the job manager to retry the node.
    """

    UNKNOWN = "UNKNOWN"
    INIT = "INIT"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @staticmethod
    def is_running(state: "WorkerState") -> bool:
        """Return the state of the Worker.

        Returns:
             True if the worker state represents workers still running
             (e.g. that the process exists but not necessarily healthy).
        """
        return state in {WorkerState.HEALTHY, WorkerState.UNHEALTHY}


class WorkerGroup:
    """A set of ``Worker`` instances.

    The class defines a set of ``Worker`` instances for the given ``WorkerSpec`` managed by ``ElasticAgent``. Whether the worker
    group contains cross instance workers or not depends on the implementation of the agent.
    """

    __slots__ = [
        "spec",
        "workers",
        "store",
        "group_rank",
        "group_world_size",
        "state",
        "master_addr",
        "master_port",
    ]

    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]

        # assigned after rdzv
        self.store = None
        self.group_rank = None
        self.group_world_size = None
        self.master_addr = None
        self.master_port = None

        self.state = WorkerState.INIT


class _RoleInstanceInfo:
    """The class is used by the agent to exchange the information with other agents.

    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """

    __slots__ = ["role", "rank", "local_world_size"]

    def __init__(self, role: str, rank: int, local_world_size: int):
        r"""Initialize the agent class instance.

        Args:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """
        self.role = role
        self.rank = rank
        self.local_world_size = local_world_size

    def serialize(self) -> bytes:
        dict_data = {
            "role": self.role,
            "rank": self.rank,
            "local_world_size": self.local_world_size,
        }
        return json.dumps(dict_data).encode(encoding="UTF-8")

    @staticmethod
    def deserialize(data: bytes):
        dict_data = json.loads(data.decode(encoding="UTF-8"))
        return _RoleInstanceInfo(
            dict_data["role"], dict_data["rank"], dict_data["local_world_size"]
        )

    @staticmethod
    def compare(obj1, obj2) -> int:
        if obj1.role == obj2.role:
            return obj1.rank - obj2.rank
        elif obj1.role > obj2.role:
            return 1
        else:
            return -1

    @staticmethod
    def find_role_boundaries(roles_infos: list, role: str) -> tuple[int, int]:
        start_idx, end_idx = -1, -1
        for idx, role_info in enumerate(roles_infos):
            if role_info.role == role:
                if start_idx == -1:
                    start_idx = idx
                end_idx = idx
        return (start_idx, end_idx)


@dataclass
class RunResult:
    """Return results of the worker executions.

    Run results follow an "all-or-nothing" policy where the run is successful if and
    only if ALL local workers managed by this agent complete successfully.

    If the result is successful (e.g. ``is_failed() = False``) then the ``return_values``
    field contains the outputs (return values) of the workers managed by THIS agent mapped
    by their GLOBAL ranks. That is ``result.return_values[0]`` is the return value of
    global rank 0.

    .. note:: ``return_values`` are only meaningful for when the worker entrypoint
              is a function. Workers specified as a binary entrypoint do not canonically
              have a return value and the ``return_values`` field is meaningless and
              may be empty.

    If ``is_failed()`` returns ``True`` then the ``failures`` field contains the
    failure information, again, mapped by the GLOBAL rank of the worker that failed.

    The keys in ``return_values`` and ``failures`` are mutually exclusive, that is,
    a worker's final state can only be one of: succeeded, failed. Workers intentionally
    terminated by the agent according to the agent's restart policy, are not represented
    in either ``return_values`` nor ``failures``.
    """

    state: WorkerState
    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)

    def is_failed(self) -> bool:
        return self.state == WorkerState.FAILED


def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


class ElasticAgent(abc.ABC):
    """An agent process responsible for managing one or more worker processes.

    The worker processes are assumed to be regular distributed PyTorch scripts.
    When the worker process is created by the agent, the agent provides the
    necessary information for the worker processes to properly initialize
    a torch process group.

    The exact deployment topology and ratio of agent-to-worker is dependent
    on the specific implementation of the agent and the user's job placement
    preferences. For instance, to run a distributed training job on GPU with
    8 trainers (one per GPU) one can:

    1. Use 8 x single GPU instances, place an agent per instance, managing
       1 worker per agent.
    2. Use 4 x double GPU instances, place an agent per instance, managing
       2 workers per agent.
    3. Use 2 x quad GPU instances, place an agent per instance, managing
       4 workers per agent.
    4. Use 1 x 8 GPU instance, place an agent per instance, managing
       8 workers per agent.

    Usage
    ::

     group_result = agent.run()
      if group_result.is_failed():
        # workers failed
        failure = group_result.failures[0]
        logger.exception("worker 0 failed with exit code : %s", failure.exit_code)
      else:
        return group_result.return_values[0] # return rank 0's results

    """

    @abc.abstractmethod
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """Run the agent.

        Supports retrying the worker group on failures up to ``max_restarts``.

        Returns:
            The result of the execution, containing the return values or
            failure details for each worker mapped by the worker's global rank.

        Raises:
            Exception - any other failures NOT related to worker process
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        """Return the ``WorkerGroup`` for the given ``role``.

        Note that the worker group is a mutable object and hence in a
        multi-threaded/process environment it may change state.
        Implementers are encouraged (but not required) to return
        a defensive read-only copy.
        """
        raise NotImplementedError


class SimpleElasticAgent(ElasticAgent):
    """An ``ElasticAgent`` that manages one particular type of worker role.

    An ``ElasticAgent`` that manages workers (``WorkerGroup``) for a single ``WorkerSpec``
    such as one particular type of worker role.
    """

    def __init__(
        self,
        spec: WorkerSpec,
        exit_barrier_timeout: float = 300,
        shutdown_timeout: int = 30,
    ):
        self._worker_group = WorkerGroup(spec)
        self._remaining_restarts = self._worker_group.spec.max_restarts
        self._store = None
        self._exit_barrier_timeout = exit_barrier_timeout
        self._shutdown_timeout = shutdown_timeout
        self._total_execution_time = 0
        self._in_exit_barrier: bool = False

    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        return self._worker_group

    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        r"""Start ``worker_group.spec.local_world_size`` number of workers.

        This is according to worker spec for the worker group .
        Returns a map of ``local_rank`` to worker ``id``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        r"""Stop all workers in the given worker group.

        Implementers must deal with workers in all states defined by
        ``WorkerState``. That is, it must gracefully handle stopping
        non-existent workers, unhealthy (stuck) workers, etc.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        r"""Check on the workers for the ``worker_group``.

        This function also returns the new state of the worker group.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _shutdown(
        self, death_sig: signal.Signals = signal.SIGTERM, timeout: int = 30
    ) -> None:
        """Clean up any resources that were allocated during the agent's work.

        Args:
            death_sig: Signal to send to the child process, SIGTERM is default
            timeout: Time to wait for graceful shutdown before sending SIGKILL
        """
        raise NotImplementedError

    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        r"""Run rendezvous for the workers specified by the worker spec.

        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """
        spec = worker_group.spec

        with self.record_duration("RENDEZVOUS"):
            rdzv_info = spec.rdzv_handler.next_rendezvous()
        store = rdzv_info.store
        group_rank = rdzv_info.rank
        group_world_size = rdzv_info.world_size

        # master_addr/master_port could be explicitly overridden
        # TODO: BC - specific to static rdzv and can be simplified further
        master_addr = spec.master_addr or rdzv_info.bootstrap_store_info.master_addr
        master_port = spec.master_port or rdzv_info.bootstrap_store_info.master_port

        self._store = store

        with self.record_duration("ASSIGN_WORKER_RANKS"):
            workers = self._assign_worker_ranks(
                store, group_rank, group_world_size, spec
            )
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size
        worker_group.master_addr = master_addr
        worker_group.master_port = master_port

        restart_count = spec.max_restarts - self._remaining_restarts

        logger.info(
            "[%(role)s] Rendezvous complete for workers. Result:\n"
            "  restart_count=%(restart_count)s\n"
            "  master_addr=%(master_addr)s\n"
            "  master_port=%(master_port)s\n"
            "  group_rank=%(group_rank)s\n"
            "  group_world_size=%(group_world_size)s\n"
            "  local_ranks=%(local_ranks)s\n"
            "  role_ranks=%(role_ranks)s\n"
            "  global_ranks=%(global_ranks)s\n"
            "  role_world_sizes=%(role_world_sizes)s\n"
            "  global_world_sizes=%(global_world_sizes)s\n"
            "  event_log_handler=%(event_log_handler)s\n",
            {
                "role": spec.role,
                "restart_count": restart_count,
                "master_addr": master_addr,
                "master_port": master_port,
                "group_rank": group_rank,
                "group_world_size": group_world_size,
                "local_ranks": [worker.local_rank for worker in workers],
                "role_ranks": [worker.role_rank for worker in workers],
                "global_ranks": [worker.global_rank for worker in workers],
                "role_world_sizes": [worker.role_world_size for worker in workers],
                "global_world_sizes": [worker.world_size for worker in workers],
                "event_log_handler": spec.event_log_handler,
            },
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _assign_worker_ranks(
        self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
    ) -> list[Worker]:
        """Determine proper ranks for worker processes.

        Fast Path: when all workers have the same role and world size. We calculate
        the global rank to be group_rank * group_world_size + local_rank. And the
        `role_world_size` is the same as `global_world_size`. No TCP store is used in
        this case. This is only enabled when users set the environment variable
        `TORCH_ELASTIC_WORKER_IDENTICAL` to 1.

        Time complexity: each worker O(1), overall O(1)

        Slow Path: when workers have different roles and world sizes. We use the
        following algorithm:

        1. Each agent writes its configuration(group_rank, group_world_size
           , num_workers) to the common store.
        2. The rank 0 agent reads all the role_info from the store and
           determines each agents worker ranks.
        3. Determine the global rank: the global rank of the workers is computed
           by cumulative sum of the local_world_size for all workers in front of it.
           For efficiency reasons each worker is assigned a base global rank
           such that its workers are in the range [base_global_rank,
           base_global_rank + local_world_size).
        4. Determine the role rank: The role rank is determined using the algorithms
           in the point 3 with the exception that the ranks are calculated with
           respect to the role name.
        5. The rank 0 agent writes the assigned ranks to the store.
        6. Each agent reads the assigned ranks from the store.

        Time complexity: each worker O(1), rank0 O(n), overall O(n)
        """

        if os.environ.get("TORCH_ELASTIC_WORKER_IDENTICAL", "0") == "1":
            global_world_size = group_world_size * spec.local_world_size
            base_global_rank = group_rank * spec.local_world_size
            base_role_rank = base_global_rank
            role_world_size = global_world_size
        else:
            ROLE_INFO_PREFIX = "torchelastic/role_info/"
            ASSIGNED_RANKS_PREFIX = "torchelastic/assigned_ranks/"

            agent_role_info = _RoleInstanceInfo(
                spec.role, group_rank, spec.local_world_size
            )
            store.set(f"{ROLE_INFO_PREFIX}{group_rank}", agent_role_info.serialize())

            # tcp store is collocated with rank 0 so we can use it to do extra compute to reduce overall # of operations.
            if group_rank == 0:
                role_infos_bytes = store.multi_get(
                    [f"torchelastic/role_info/{i}" for i in range(group_world_size)]
                )
                role_infos = [
                    _RoleInstanceInfo.deserialize(info_bytes)
                    for info_bytes in role_infos_bytes
                ]

                role_sizes = defaultdict(lambda: 0)
                global_size = 0
                for role_info in role_infos:
                    role_sizes[role_info.role] += role_info.local_world_size
                    global_size += role_info.local_world_size

                base_global_rank = 0
                role_ranks = defaultdict(lambda: 0)

                keys = []
                values = []
                for i, role_info in enumerate(role_infos):
                    keys.append(f"{ASSIGNED_RANKS_PREFIX}{i}")
                    values.append(
                        json.dumps(
                            [
                                base_global_rank,
                                global_size,
                                role_ranks[role_info.role],
                                role_sizes[role_info.role],
                            ]
                        )
                    )

                    base_global_rank += role_info.local_world_size
                    role_ranks[role_info.role] += role_info.local_world_size

                store.multi_set(keys, values)

            # get will block until the data is available in the store.
            (
                base_global_rank,
                global_world_size,
                base_role_rank,
                role_world_size,
            ) = json.loads(store.get(f"{ASSIGNED_RANKS_PREFIX}{group_rank}"))

        workers = []
        for local_rank in range(spec.local_world_size):
            worker = Worker(
                local_rank=local_rank,
                global_rank=base_global_rank + local_rank,
                role_rank=base_role_rank + local_rank,
                world_size=global_world_size,
                role_world_size=role_world_size,
            )
            workers.append(worker)
        return workers

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        r"""Start a fresh set of workers for the worker_group.

        Essentially, a rendezvous followed by a ``start_workers``.
        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
        role = worker_group.spec.role
        logger.info("[%s] Rendezvous'ing worker group", role)

        # TODO after stopping workers, wait at least monitor_interval*2 for
        # workers on different nodes to fail on a collective op before waiting
        # on the rdzv barrier, this way we ensure that nodes enter rdzv
        # at around the same time and reduce false positive rdzv timeout errors
        self._rendezvous(worker_group)

        logger.info("[%s] Starting worker group", role)
        worker_ids = self._start_workers(worker_group)
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id
            record(
                self._construct_event("START", EventSource.WORKER, worker),
                worker_group.spec.event_log_handler,
            )

        worker_group.state = WorkerState.HEALTHY

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        """Restart (stops, rendezvous, starts) all local workers in the group."""
        role = worker_group.spec.role
        logger.info("[%s] Stopping worker group", role)
        self._stop_workers(worker_group)
        worker_group.state = WorkerState.STOPPED
        self._initialize_workers(worker_group)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        start_time = time.monotonic()
        shutdown_called: bool = False
        try:
            result = self._invoke_run(role)
            self._total_execution_time = int(time.monotonic() - start_time)
            self._record_metrics(result)
            self._record_worker_events(result)
            return result
        except RendezvousGracefulExitError as e:
            logger.info("Rendezvous gracefully exited: %s", e)
        except SignalException as e:
            logger.warning("Received %s death signal, shutting down workers", e.sigval)
            self._shutdown(e.sigval, timeout=self._shutdown_timeout)
            shutdown_called = True
            raise
        finally:
            if not shutdown_called:
                self._shutdown(timeout=self._shutdown_timeout)
            # record the execution time in case there were any exceptions during run.
            self._total_execution_time = int(time.monotonic() - start_time)

    def get_event_failed(self) -> Event:
        return self._construct_event(
            state="FAILED",
            source=EventSource.AGENT,
            raw_error=traceback.format_exc(),
        )

    def get_event_succeeded(self) -> Event:
        return self._construct_event(
            state="SUCCEEDED",
            source=EventSource.AGENT,
        )

    def _record_worker_events(self, result: RunResult) -> None:
        for worker in self._worker_group.workers:
            failure = result.failures.get(worker.global_rank)
            state: str = self._get_worker_state(worker, result)
            raw_error = json.dumps(failure.error_file_data) if failure else None
            exit_code = failure.exitcode if failure else None
            worker_pid = failure.pid if failure else None
            record(
                self._construct_event(
                    state=state,
                    source=EventSource.WORKER,
                    worker=worker,
                    raw_error=raw_error,
                    exit_code=exit_code,
                    worker_pid=worker_pid,
                ),
                self._worker_group.spec.event_log_handler,
            )

    def _get_worker_state(self, worker: Worker, result: RunResult) -> str:
        failure = result.failures.get(worker.global_rank)
        if result.state in {WorkerState.UNHEALTHY, WorkerState.FAILED} and not failure:
            # The worker got terminated by the torchelastic agent via SIGTERM signal
            return "TERMINATED"
        elif failure or worker.global_rank in result.return_values:
            return result.state.value
        else:
            raise ValueError(f"Unknown worker: {worker.global_rank}")

    @contextmanager
    def record_duration(self, state: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            record(
                self._construct_event(
                    state=state, source=EventSource.AGENT, duration_ms=duration_ms
                ),
                self._worker_group.spec.event_log_handler,
            )

    def _construct_event(
        self,
        state: str,
        source: EventSource,
        worker: Worker | None = None,
        raw_error: str | None = None,
        duration_ms: float | None = None,
        exit_code: int | None = None,
        worker_pid: int | None = None,
    ) -> Event:
        wg = self._worker_group
        spec = wg.spec
        md = {
            "group_world_size": wg.group_world_size,
            "entry_point": spec.get_entrypoint_name(),
        }
        if worker:
            md["local_rank"] = (worker.local_rank,)
            md["role_rank"] = (worker.role_rank,)
            md["role_world_size"] = (worker.role_world_size,)
            md["exit_code"] = (exit_code,)
            md["worker_pid"] = (worker_pid,)
            global_rank = worker.global_rank
            worker_id = str(worker.id)
        else:
            global_rank = None
            worker_id = None
        md_str = json.dumps(md)
        metadata = {
            "run_id": spec.rdzv_handler.get_run_id(),
            "global_rank": global_rank,
            "group_rank": wg.group_rank,
            "worker_id": worker_id,
            "role": spec.role,
            "hostname": _get_fq_hostname(),
            "state": state,
            "total_run_time": self._total_execution_time,
            "rdzv_backend": spec.rdzv_handler.get_backend(),
            "raw_error": raw_error,
            "metadata": md_str,
            "agent_restarts": spec.max_restarts - self._remaining_restarts,
            "duration_ms": duration_ms,
        }

        return Event(
            f"torchelastic.worker.status.{state}", source=source, metadata=metadata
        )

    def _record_metrics(self, group_results: RunResult):
        is_failed = group_results.is_failed()
        self._record_flakiness_metric(is_failed)
        spec = self._worker_group.spec
        restarts_happened = self._remaining_restarts != spec.max_restarts
        put_metric(f"workers.{spec.role}.run_total", 1)
        self._record_metric_with_condition(
            "run_success_with_retries", not is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_success_no_retries", not is_failed and not restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_with_retries", is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_no_retries", is_failed and not restarts_happened
        )

    def _record_metric_with_condition(self, metric_name, condition):
        spec = self._worker_group.spec
        if condition:
            put_metric(f"workers.{spec.role}.{metric_name}", 1)
        else:
            put_metric(f"workers.{spec.role}.{metric_name}", 0)

    def _record_flakiness_metric(self, is_failed: bool = False):
        if is_failed:
            flakiness = 100.0
        else:
            spec = self._worker_group.spec
            flakiness = 100.0 - 100.0 * (self._remaining_restarts + 1) / (
                spec.max_restarts + 1
            )
        spec = self._worker_group.spec

        put_metric(f"workers.{spec.role}.flakiness", int(flakiness))

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        logger.info(
            "[%s] starting workers for entrypoint: %s", role, spec.get_entrypoint_name()
        )

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        while True:
            if self._worker_group.state == WorkerState.INIT:
                raise AssertionError
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                logger.info(
                    "[%s] worker group successfully finished."
                    " Waiting %s seconds for other agents to finish.",
                    role,
                    self._exit_barrier_timeout,
                )
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    logger.info(
                        "[%s] Worker group %s. "
                        "%s/%s attempts left;"
                        " will restart worker group",
                        role,
                        state.name,
                        self._remaining_restarts,
                        spec.max_restarts,
                    )
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    logger.info(
                        "[%s] Detected %s "
                        "new nodes from group_rank=%s; "
                        "will restart worker group",
                        role,
                        num_nodes_waiting,
                        group_rank,
                    )
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(  # noqa: TRY002
                    f"[{role}] Worker group in {state.name} state"
                )

    def _exit_barrier(self):
        """
        Define a barrier that keeps the agent process alive until all workers finish.

        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times.
        """
        logger.info(
            "Local worker group finished (%s). "
            "Waiting %s seconds for other agents to finish",
            self._worker_group.state,
            self._exit_barrier_timeout,
        )
        start = time.time()
        self._in_exit_barrier = True
        try:
            store_util.barrier(
                store=self._store,
                world_size=self._worker_group.group_world_size,
                key_prefix=_TERMINAL_STATE_SYNC_ID,
                barrier_timeout=self._exit_barrier_timeout,
            )
            logger.info(
                "Done waiting for other agents. Elapsed: %s seconds",
                time.time() - start,
            )
        except SignalException as e:
            logger.warning("Got termination signal: %s", e.sigval)
            raise
        except Exception:
            logger.exception(
                "Error waiting on exit barrier. Elapsed: %s seconds",
                time.time() - start,
            )
        finally:
            self._in_exit_barrier = False
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

from torch.distributed.elastic.utils.logging import get_logger


log = get_logger(__name__)

__all__ = ["HealthCheckServer", "create_healthcheck_server"]


class HealthCheckServer:
    """
    Interface for health check monitoring server, which can be extended
    by starting tcp/http server on the specified port.

    Args:

        alive_callback: Callable[[], int], callback to last progress time of agent

        port: int, port number to start tcp/http server

        timeout: int, timeout seconds to decide agent is alive/dead
    """

    _alive_callback: Callable[[], int]
    _port: int
    _timeout: int

    def __init__(
        self, alive_callback: Callable[[], int], port: int, timeout: int
    ) -> None:
        self._alive_callback = alive_callback
        self._port = port
        self._timeout = timeout

    def start(self) -> None:
        """
        Unsupported functionality for PyTorch, doesn't start any health check server
        """
        log.warning("No health check server started")

    def stop(self) -> None:
        """
        Function to stop health check server
        """
        log.info("Stopping noop health check server.")

    @property
    def alive_callback(self) -> Callable[[], int]:
        return self._alive_callback


def create_healthcheck_server(
    alive_callback: Callable[[], int],
    port: int,
    timeout: int,
) -> HealthCheckServer:
    """
    creates health check server object
    """
    return HealthCheckServer(alive_callback, port, timeout)
<|endoftext|>#!/usr/bin/env/python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metrics API.

**Overview**:

The metrics API in torchelastic is used to publish telemetry metrics.
It is designed to be used by torchelastic's internal modules to
publish metrics for the end user with the goal of increasing visibility
and helping with debugging. However you may use the same API in your
jobs to publish metrics to the same metrics ``sink``.

A ``metric`` can be thought of as timeseries data
and is uniquely identified by the string-valued tuple
``(metric_group, metric_name)``.

torchelastic makes no assumptions about what a ``metric_group`` is
and what relationship it has with ``metric_name``. It is totally up
to the user to use these two fields to uniquely identify a metric.

.. note:: The metric group ``torchelastic`` is reserved by torchelastic for
          platform level metrics that it produces.
          For instance torchelastic may output the latency (in milliseconds)
          of a re-rendezvous operation from the agent as
          ``(torchelastic, agent.rendezvous.duration.ms)``

A sensible way to use metric groups is to map them to a stage or module
in your job. You may also encode certain high level properties
the job such as the region or stage (dev vs prod).

**Publish Metrics**:

Using torchelastic's metrics API is similar to using python's logging
framework. You first have to configure a metrics handler before
trying to add metric data.

The example below measures the latency for the ``calculate()`` function.

::

  import time
  import torch.distributed.elastic.metrics as metrics

  # makes all metrics other than the one from "my_module" to go /dev/null
  metrics.configure(metrics.NullMetricsHandler())
  metrics.configure(metrics.ConsoleMetricsHandler(), "my_module")


  def my_method():
      start = time.time()
      calculate()
      end = time.time()
      metrics.put_metric("calculate_latency", int(end - start), "my_module")

You may also use the torch.distributed.elastic.metrics.prof` decorator
to conveniently and succinctly profile functions

::

  # -- in module examples.foobar --

  import torch.distributed.elastic.metrics as metrics

  metrics.configure(metrics.ConsoleMetricsHandler(), "foobar")
  metrics.configure(metrics.ConsoleMetricsHandler(), "Bar")


  @metrics.prof
  def foo():
      pass


  class Bar:
      @metrics.prof
      def baz():
          pass

``@metrics.prof`` will publish the following metrics
::

  <leaf_module or classname>.success - 1 if the function finished successfully
  <leaf_module or classname>.failure - 1 if the function threw an exception
  <leaf_module or classname>.duration.ms - function duration in milliseconds

**Configuring Metrics Handler**:

`torch.distributed.elastic.metrics.MetricHandler` is responsible for emitting
the added metric values to a particular destination. Metric groups can be
configured with different metric handlers.

By default torchelastic emits all metrics to ``/dev/null``.
By adding the following configuration metrics,
``torchelastic`` and ``my_app`` metric groups will be printed out to
console.

::

  import torch.distributed.elastic.metrics as metrics

  metrics.configure(metrics.ConsoleMetricHandler(), group="torchelastic")
  metrics.configure(metrics.ConsoleMetricHandler(), group="my_app")

**Writing a Custom Metric Handler**:

If you want your metrics to be emitted to a custom location, implement
the `torch.distributed.elastic.metrics.MetricHandler` interface
and configure your job to use your custom metric handler.

Below is a toy example that prints the metrics to ``stdout``

::

  import torch.distributed.elastic.metrics as metrics


  class StdoutMetricHandler(metrics.MetricHandler):
      def emit(self, metric_data):
          ts = metric_data.timestamp
          group = metric_data.group_name
          name = metric_data.name
          value = metric_data.value
          print(f"[{ts}][{group}]: {name}={value}")


  metrics.configure(StdoutMetricHandler(), group="my_app")

Now all metrics in the group ``my_app`` will be printed to stdout as:

::

  [1574213883.4182858][my_app]: my_metric=<value>
  [1574213940.5237644][my_app]: my_metric=<value>

"""

from typing import Optional

from .api import (
    configure,
    ConsoleMetricHandler,
    get_elapsed_time_ms,
    getStream,
    MetricData,
    MetricHandler,
    MetricsConfig,
    NullMetricHandler,
    prof,
    profile,
    publish_metric,
    put_metric,
)


def initialize_metrics(cfg: MetricsConfig | None = None):
    pass


try:
    from torch.distributed.elastic.metrics.static_init import *  # type: ignore[import] # noqa: F403
except ModuleNotFoundError:
    pass
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time
from collections import namedtuple
from functools import wraps
from typing_extensions import deprecated


__all__ = [
    "MetricsConfig",
    "MetricHandler",
    "ConsoleMetricHandler",
    "NullMetricHandler",
    "MetricStream",
    "configure",
    "getStream",
    "prof",
    "profile",
    "put_metric",
    "publish_metric",
    "get_elapsed_time_ms",
    "MetricData",
]

MetricData = namedtuple("MetricData", ["timestamp", "group_name", "name", "value"])


class MetricsConfig:
    __slots__ = ["params"]

    def __init__(self, params: dict[str, str] | None = None):
        self.params = params
        if self.params is None:
            self.params = {}


class MetricHandler(abc.ABC):
    @abc.abstractmethod
    def emit(self, metric_data: MetricData):
        pass


class ConsoleMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData):
        print(
            f"[{metric_data.timestamp}][{metric_data.group_name}]: {metric_data.name}={metric_data.value}"
        )


class NullMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData):
        pass


class MetricStream:
    def __init__(self, group_name: str, handler: MetricHandler):
        self.group_name = group_name
        self.handler = handler

