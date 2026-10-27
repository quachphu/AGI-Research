    if not isinstance(token, str) or not token:
        raise SandboxError(
            "The pool host returned no valid sandbox-scoped token; recycle it with an updated sbx-server."
        )
    return token


class _SandboxServer:
    """HTTP transport to one `sbx-server` instance — a dedicated job or a shared host.

    Owns the `httpx2.Client`, the base URL and the auth headers.
    In dedicated mode a server is paired 1:1 with its [`Sandbox`
    In pool mode one server (one host job) is shared by many sandboxes, and `live`/`capacity` track packing.
    """

    def __init__(
        self,
        *,
        job_id: str,
        owner: str,
        image: str | None,
        base_url: str,
        nonce: str,
        sandbox_token: str,
        api: HfApi,
        max_connections: int = 10,
        capacity: int = 0,
    ) -> None:
        self.job_id = job_id
        self.owner = owner
        self._image = image
        self.base_url = base_url
        # Public nonce the sandbox token is derived from; kept so a host can be persisted
        # to (and rebuilt from) the pool cache without re-reading the job labels.
        self.nonce = nonce
        self._api = api
        self._sandbox_token = sandbox_token
        # Packing bookkeeping (shared mode only).
        self.capacity = capacity
        self.live = 0
        # False only for hosts rebuilt from the (best-effort) pool cache: their job may
        # be gone, so the first failed request drops them instead of failing the create.
        self.verified = True
        # Whether *this* process started this host job, and may therefore cancel
        # it. Ownership belongs to the host, not to the pool handle: a handle used
        # to cancel hosts it had merely discovered via labels, which may have been
        # started by another process -- or another member of the namespace -- and
        # may be serving their sandboxes.
        self.owned = False
        # httpx2.Client is thread-safe, so a single client serves both sequential requests
        # and the concurrent workers used for parallel file transfers / many sandboxes.
        self._client = httpx2.Client(
            headers={"X-Sandbox-Token": sandbox_token},
            auth=_SandboxAuth(api),
            limits=httpx2.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            # Every request on this client carries the HF bearer and a sandbox token, and there
            # is no redirect in this protocol -- the Jobs proxy serves the in-job server
            # directly. Following one would hand both credentials to a destination that the
            # response, rather than the client, chose.
            follow_redirects=False,
        )

    @classmethod
    def from_job(
        cls,
        *,
        job: JobInfo,
        nonce: str,
        sandbox_token: str,
        api: HfApi,
        max_connections: int = 10,
        capacity: int = 0,
    ) -> "_SandboxServer":
        """Build a server from a freshly fetched job (reads its exposed server URL)."""
        return cls(
            job_id=job.id,
            owner=job.owner.name,
            image=job.docker_image or job.space_id,
            base_url=_find_server_url(job),
            nonce=nonce,
            sandbox_token=sandbox_token,
            api=api,
            max_connections=max_connections,
            capacity=capacity,
        )

    @property
    def image(self) -> str | None:
        return self._image

    def _with_token(self, kwargs: dict, sandbox_token: str | None) -> dict:
        """Override `X-Sandbox-Token` for one request.

        The client-level header carries the *host* credential, which manages the
        pool. A pooled sandbox's own operations should present that sandbox's
        capability token instead, so nothing wider than the sandbox is offered on
        a per-sandbox call.
        """
        if sandbox_token is not None:
            kwargs["headers"] = {**kwargs.get("headers", {}), "X-Sandbox-Token": sandbox_token}
        return kwargs

    def request(self, method: str, path: str, *, sandbox_token: str | None = None, **kwargs) -> httpx2.Response:
        """Request to the in-job server. Raises SandboxError on API errors."""
        timeout = kwargs.pop("timeout", httpx2.Timeout(60.0, connect=10.0))
        kwargs = self._with_token(kwargs, sandbox_token)
        response = self._client.request(method, self.base_url + path, timeout=timeout, **kwargs)
        if response.status_code >= 400:
            _raise_for_status(response)
        return response

    @contextmanager
    def stream(
        self, method: str, path: str, *, sandbox_token: str | None = None, **kwargs
    ) -> Iterator[httpx2.Response]:
        """Streaming request to the in-job server. Raises SandboxError on API errors."""
        timeout = kwargs.pop("timeout", httpx2.Timeout(70.0, connect=10.0))  # server pings every 15s
        kwargs = self._with_token(kwargs, sandbox_token)
        with self._client.stream(method, self.base_url + path, timeout=timeout, **kwargs) as response:
            if response.status_code >= 400:
                _raise_for_status(response)
            yield response

    def sandbox_token(self, local_id: str) -> str:
        """Recover a pooled sandbox's capability using the host management credential."""
        data = self.request("GET", f"/v1/sandboxes/{local_id}/token").json()
        return _require_scoped_token(data.get("token"))

    def close(self) -> None:
        self._client.close()

    def cancel_job(self) -> None:
        self._api.cancel_job(job_id=self.job_id, namespace=self.owner)

    def wait_ready(self, start_timeout: float) -> None:
        """Poll /health until the server answers; fail fast (with logs) if the job dies."""
        deadline = time.time() + start_timeout
        last_job_check = 0.0
        while time.time() < deadline:
            try:
                response = self._client.get(self.base_url + "/health", timeout=httpx2.Timeout(5.0))
                if response.status_code == 200:
                    _check_server_protocol(response, self.job_id)
                    return
            except httpx2.RequestError:
                pass
            if time.time() - last_job_check > 2.0:
                last_job_check = time.time()
                job = self._api.inspect_job(job_id=self.job_id, namespace=self.owner)
                if job.status.stage in _TERMINAL_STAGES:
                    logs = _tail_job_logs(self._api, self.job_id, namespace=self.owner)
                    raise SandboxError(
                        f"Sandbox job {self.job_id} terminated during startup "
                        f"(status: {job.status.stage}, message: {job.status.message}).{logs}"
                    )
            time.sleep(0.15)
        raise SandboxError(f"Sandbox job {self.job_id} did not become ready within {start_timeout:.0f}s.")


class _KillMethod:
    """Lets `kill` work both as a classmethod and an instance method."""

    @overload
    def __get__(self, instance: None, owner: type) -> Callable[..., None]: ...
    @overload
    def __get__(self, instance: "Sandbox", owner: type) -> Callable[[], None]: ...
    def __get__(self, instance: "Sandbox | None", owner: type) -> Callable[..., None]:
        if instance is not None:
            return instance._kill

        def kill(sandbox_id: str, *, namespace: str | None = None, token: str | None = None) -> None:
            owner.connect(sandbox_id, namespace=namespace, token=token).kill()  # type: ignore[attr-defined]

        return kill


class Sandbox:
    """An isolated cloud machine running on Hugging Face Jobs.

    > [!NOTE]
    > The Sandbox API is experimental. Its API and behavior may change without notice.

    Create a dedicated one with [`Sandbox.create`] (one job per sandbox), or get many cheap shared ones from a [`SandboxPool`].
    Reattach to a running sandbox from anywhere with [`Sandbox.connect`]. Use as a context manager to terminate it on exit:

    ```python
    >>> from huggingface_hub import Sandbox
    >>> with Sandbox.create(image="python:3.12") as sbx:
    ...     print(sbx.run("python --version").stdout)
    ```
    """

    def __init__(
        self,
        *,
        id: str,
        server: _SandboxServer,
        local_id: str | None,
        owns_sandbox: bool,
        owns_server: bool,
        sandbox_token: str | None = None,
    ) -> None:
        self.id = id
        self._server = server
        # Capability token for this sandbox alone (pool mode). Sent instead of the
        # host credential on every per-sandbox call, so a pooled sandbox's
        # operations -- and the headers handed to a port-proxy client -- confer no
        # authority over its siblings or over the pool. A dedicated job is already
        # one sandbox, so its job credential is the appropriate token.
        self._sandbox_token = server._sandbox_token if local_id is None else _require_scoped_token(sandbox_token)
        # None in dedicated mode; the host-local sandbox id in shared mode.
        self._local_id = local_id
        # Path prefix for all in-server operations: dedicated routes live under
        # /v1/*, shared ones under /v1/sandboxes/<local_id>/*.
        self._base_path = "/v1" if local_id is None else f"/v1/sandboxes/{local_id}"
        # Whether exiting a `with` block terminates the sandbox (True for sandboxes
        # we created, False for ones reattached via connect()).
        self._owns_sandbox = owns_sandbox
        # Whether closing/killing this sandbox also closes the HTTP client. False
        # for pool sandboxes, whose client (the host) is owned by the pool.
        self._owns_server = owns_server
        # Set by SandboxPool to free a packing slot when a shared sandbox is killed.
        self._on_kill: Callable[["Sandbox"], None] | None = None
        self._killed = False
        self.files = SandboxFiles(self)

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def create(
        cls,
        image: str = DEFAULT_IMAGE,
        *,
        flavor: str = "cpu-basic",
        idle_timeout: int | float | str | None = DEFAULT_IDLE_TIMEOUT,
        env: dict[str, Any] | None = None,
        secrets: dict[str, Any] | None = None,
        volumes: List[Volume] | None = None,
        namespace: str | None = None,
        forward_hf_token: bool = False,
        labels: dict[str, str] | None = None,
        start_timeout: float = 120.0,
        token: str | None = None,
    ) -> "Sandbox":
        """Create a dedicated sandbox (one HF Job) and block until it is ready (~7s on cpu-basic).

        Each sandbox is a full isolated VM, so this is the right choice for GPU
        workloads or untrusted code. To fan out many cheap CPU sandboxes instead, use
        [`SandboxPool`].

        The job runs with a fixed 24h maximum lifetime; `idle_timeout` is the real
        keeper — an idle sandbox shuts itself down well before that.

        Args:
            image (`str`, *optional*, defaults to `"python:3.12"`):
                Any Docker image with `/bin/sh` (Docker Hub or `hf.co/spaces/...`).
            flavor (`str`, *optional*, defaults to `"cpu-basic"`):
                Hardware flavor, e.g. `"cpu-basic"`, `"a10g-small"`. See `hf jobs hardware`.
            idle_timeout (`int` or `float` or `str`, *optional*, defaults to `600`):
                Auto-shutdown after this much inactivity (no API calls, no running
                processes). Defaults to 10 minutes; pass `None` to disable. Note that a
                *foreground* command is not currently counted as activity, so a single
                `run()` that takes longer than this without other API traffic can have its
                sandbox shut down under it — raise the timeout (or pass `None`) for long
                single commands.
            env (`dict[str, Any]`, *optional*):
                Environment variables available in the sandbox.
            secrets (`dict[str, Any]`, *optional*):
                Secret environment variables (encrypted server-side).
            volumes (`List[Volume]`, *optional*):
                HF repos/buckets to mount, see [`Volume`].
            namespace (`str`, *optional*):
                User or org namespace to run under (defaults to current user).
            forward_hf_token (`bool`, *optional*, defaults to `False`):
                If True, your HF token is injected as `HF_TOKEN` (opt-in).
            labels (`dict[str, str]`, *optional*):
                Labels to attach to the underlying HF Job.
            start_timeout (`float`, *optional*, defaults to `120.0`):
                Max seconds to wait for the sandbox to become ready.
            token (`str`, *optional*):
                HF token override.

        The image only needs `/bin/sh`. The sandbox server is downloaded at startup with
        `wget`/`curl` if available, otherwise read off an always-mounted server bucket (which
        adds ~2-3s to cold start, so shipping `wget`/`curl` keeps it fast).
        """
        if labels is not None:
            if not isinstance(labels, dict):
                raise ValueError(f"`labels` must be a dict[str, str], got {type(labels).__name__}")
            for key, value in labels.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("`labels` keys and values must be strings")
                if key in RESERVED_SANDBOX_LABELS:
                    raise ValueError(f"Label '{key}' is reserved by huggingface_hub Sandbox and cannot be overridden.")

        api = HfApi(token=token)
        hf_token = _effective_token(api)
        nonce = token_hex(16)
        sandbox_token = _derive_sandbox_token(hf_token, nonce)

        command, job_env, job_secrets, job_volumes = _bootstrap_job_spec(
            api,
            hf_token,
            env=env,
            secrets=secrets,
            volumes=volumes,
            idle_timeout=idle_timeout,
            forward_hf_token=forward_hf_token,
            sandbox_token=sandbox_token,
        )

        job = api.run_job(
            image=image,
            command=command,
            env=job_env,
            secrets=job_secrets,
            flavor=flavor,
            timeout=SANDBOX_MAX_LIFETIME,
            labels={**(labels or {}), SANDBOX_LABEL: "1", MODE_LABEL: MODE_DEDICATED, NONCE_LABEL: nonce},
            volumes=job_volumes or None,
            expose=[SANDBOX_SERVER_PORT],
            namespace=namespace,
        )
        server: "_SandboxServer | None" = None
        try:
            server = _SandboxServer.from_job(
                job=job,
                nonce=nonce,
                sandbox_token=sandbox_token,
                api=api,
                max_connections=SandboxFiles.PARALLEL_MAX_WORKERS + 2,
            )
            server.wait_ready(start_timeout)
        except Exception:
            # run_job already started a billable job; cancel it before re-raising so it
            # doesn't linger as an orphan (e.g. if the server port isn't exposed or startup fails).
            try:
                api.cancel_job(job_id=job.id, namespace=job.owner.name)
            except Exception as e:
                logger.warning(f"Failed to cancel sandbox job {job.id} after startup failure: {e}")
            if server is not None:
                server.close()
            raise
        return cls(id=job.id, server=server, local_id=None, owns_sandbox=True, owns_server=True)

    @classmethod
    def connect(cls, sandbox_id: str, *, namespace: str | None = None, token: str | None = None) -> "Sandbox":
        """Reattach to a running sandbox from anywhere, using only its id."""
        api = HfApi(token=token)
        sandbox_id, namespace = _split_sandbox_id(sandbox_id, namespace)
        if SHARED_ID_SEP in sandbox_id:
            host_job_id, local_id = sandbox_id.split(SHARED_ID_SEP, 1)
            server = _connect_host(api, host_job_id, namespace=namespace)
            try:
                existing = {item["id"] for item in server.request("GET", "/v1/sandboxes").json()}
                if local_id not in existing:
                    raise SandboxError(f"Sandbox {sandbox_id} no longer exists on host {host_job_id}.")
                # Recover the sandbox's capability token with the host credential, so
                # this handle operates with the narrow one from here on.
                sandbox_token = server.sandbox_token(local_id)
            except Exception:
                server.close()  # don't leak the HTTP client when the host is gone/unreachable
                raise
            return cls(
                id=sandbox_id,
                server=server,
                local_id=local_id,
                owns_sandbox=False,
                owns_server=True,
                sandbox_token=sandbox_token,
            )

        job = api.inspect_job(job_id=sandbox_id, namespace=namespace)
        labels = job.labels or {}
        nonce = labels.get(NONCE_LABEL)
        if labels.get(SANDBOX_LABEL) != "1" or nonce is None:
            raise SandboxError(f"Job {sandbox_id} is not a sandbox (missing '{SANDBOX_LABEL}' label).")
        if labels.get(MODE_LABEL) == MODE_POOL:
            raise SandboxError(
                f"Job {sandbox_id} is a sandbox host, not a single sandbox. Connect to one of its "
                f"sandboxes with id '<host_job_id>{SHARED_ID_SEP}<local_id>'."
            )
        if job.status.stage != "RUNNING":
            raise SandboxError(f"Sandbox {sandbox_id} is not running (status: {job.status.stage}).")
        sandbox_token = _derive_sandbox_token(_effective_token(api), nonce)
        server = _SandboxServer.from_job(
            job=job,
            nonce=nonce,
            sandbox_token=sandbox_token,
            api=api,
            max_connections=SandboxFiles.PARALLEL_MAX_WORKERS + 2,
        )
        return cls(id=job.id, server=server, local_id=None, owns_sandbox=False, owns_server=True)

    # `kill` is both a classmethod (`Sandbox.kill(id)`) and an instance method (`sbx.kill()`),
    # dispatched by the `_KillMethod` descriptor; `_kill` holds the instance behaviour.
    kill = _KillMethod()

    def _kill(self) -> None:
        """Terminate the sandbox. Idempotent.

        Dedicated sandboxes cancel their underlying job; shared sandboxes are
        removed from their host (freeing a slot) while the host keeps running.
        """
        if self._killed:
            return
        try:
            if self._local_id is None:
                self._server.cancel_job()
            else:
                self._server.request("DELETE", f"/v1/sandboxes/{self._local_id}", sandbox_token=self._sandbox_token)
        except Exception as e:
            # Don't mark as killed: a later kill() call should retry so nothing leaks.
            logger.warning(f"Failed to kill sandbox {self.id}: {e}")
            return
        self._killed = True
        if self._on_kill is not None:
            self._on_kill(self)
        if self._owns_server:
            self._server.close()

    def close(self) -> None:
        """Release the local HTTP client without terminating the sandbox. Idempotent.

        No-op for pool sandboxes (the client belongs to the pool's host).
        """
        if self._owns_server:
            self._server.close()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc_info) -> None:
        if self._owns_sandbox:
            self.kill()
        else:
            self.close()

    # ------------------------------------------------------------------ exec

    @overload
    def run(
        self,
        cmd: str | List[str],
        *,
        shell: bool | None = ...,
        env: dict[str, Any] | None = ...,
        cwd: str | None = ...,
        timeout: float | None = ...,
        stdin: str | None = ...,
        on_stdout: Callable[[str], None] | None = ...,
        on_stderr: Callable[[str], None] | None = ...,
        check: bool = ...,
        capture_output: bool = ...,
        background: Literal[False] = ...,
    ) -> SandboxCommandResult: ...

    @overload
    def run(
        self,
        cmd: str | List[str],
        *,
        shell: bool | None = ...,
        env: dict[str, Any] | None = ...,
        cwd: str | None = ...,
        background: Literal[True],
    ) -> SandboxProcess: ...

    def run(
        self,
        cmd: str | List[str],
        *,
        shell: bool | None = None,
        env: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        stdin: str | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
        check: bool = True,
        capture_output: bool = True,
        background: bool = False,
    ) -> SandboxCommandResult | SandboxProcess:
        """Run a command in the sandbox and wait for it, streaming output live.

        With `background=True` the command is started detached and `run` returns a
        [`SandboxProcess`] immediately, without waiting for it to finish — handy for
        servers and other long-running processes. List them later with [`Sandbox.processes`]
        and stop one with [`SandboxProcess.kill`]. The streaming/wait-only options
        (`timeout`, `stdin`, `on_stdout`, `on_stderr`, `check`) don't apply in that mode.

        Args:
            cmd (`str` or `List[str]`):
                A shell command string (run with `/bin/sh -c`) or an argv list (exec'd directly).
            shell (`bool`, *optional*):
                Force the execution mode instead of inferring it from the type of `cmd`.
                `True` runs through `/bin/sh -c` and requires `cmd` to be a string; `False`
                exec's `cmd` directly and requires it to be an argv list. `None` (default)
                infers from the type. Set it explicitly to avoid the type-driven footgun (e.g.
                `["echo hi"]` being exec'd as a single program named `"echo hi"`).
            env (`dict[str, Any]`, *optional*):
                Extra environment variables for this command.
            cwd (`str`, *optional*):
                Working directory.
            timeout (`float`, *optional*):
                Kill the command (whole process group) after this many seconds.
            stdin (`str`, *optional*):
                Data to write to the command's stdin.
            on_stdout (`Callable[[str], None]`, *optional*):
                Callback invoked with stdout chunks as they arrive.
            on_stderr (`Callable[[str], None]`, *optional*):
                Callback invoked with stderr chunks as they arrive.
            check (`bool`, *optional*, defaults to `True`):
                If True, raise [`SandboxCommandError`] on non-zero exit.
            capture_output (`bool`, *optional*, defaults to `True`):
                If True, accumulate stdout/stderr into the returned result. Pass `False`
                when you only want `on_stdout`/`on_stderr`: output is then handed to the
                callbacks and dropped, so a command producing gigabytes does not have to fit
                in memory. `result.stdout`/`result.stderr` are empty in that case.
            background (`bool`, *optional*, defaults to `False`):
                If True, start the command detached and return a [`SandboxProcess`] right
                away instead of waiting for it and returning a [`SandboxCommandResult`].

        Returns: a [`SandboxCommandResult`] (with `exit_code`, `stdout`, `stderr`,
        `duration_ms`), or a [`SandboxProcess`] when `background=True`.
        """
        payload = _exec_payload(cmd, shell)
        if env:
            payload["env"] = env
        if cwd:
            payload["cwd"] = cwd
        if background:
            data = self._request("POST", "/processes", json=payload).json()
            return SandboxProcess(
                # Absent on a server that predates opaque ids; `kill()` then says so
                # rather than sending a pid the server will reject.
                id=data.get("id"),
                pid=data["pid"],
                cmd=cmd,
                tag=data.get("tag"),
                _sandbox=self,
            )
        if timeout is not None:
            payload["timeout"] = timeout
        if stdin is not None:
            payload["stdin"] = stdin

        # Output was accumulated unconditionally, even when a callback was
        # consuming it -- so a streaming consumer still paid full memory for the
        # command's entire output. `capture_output=False` opts out.
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        captured = 0
        result: SandboxCommandResult | None = None

        def keep(parts: list[str], data: str) -> None:
            nonlocal captured
            if not capture_output:
                return
            if captured + len(data) > MAX_CAPTURED_OUTPUT_CHARS:
                raise SandboxError(
                    f"command produced more than {MAX_CAPTURED_OUTPUT_CHARS} characters of output. "
                    "Pass `capture_output=False` with `on_stdout`/`on_stderr` to stream it instead, "
                    "or redirect it to a file in the sandbox and download that."
                )
            captured += len(data)
            parts.append(data)

        with self._stream("POST", "/exec", json=payload) as response:
            for event in _iter_events(response):
                if event["event"] == "stdout":
                    keep(stdout_parts, event["data"])
                    if on_stdout is not None:
                        on_stdout(event["data"])
                elif event["event"] == "stderr":
                    keep(stderr_parts, event["data"])
                    if on_stderr is not None:
                        on_stderr(event["data"])
                elif event["event"] == "exit":
                    result = SandboxCommandResult(
                        exit_code=event["exit_code"],
                        stdout="".join(stdout_parts),
                        stderr="".join(stderr_parts),
                        signal=event.get("signal"),
                        timed_out=event.get("timed_out", False),
                        duration_ms=event.get("duration_ms", 0),
                    )
                    # Exit is terminal: a truncated HTTP tail must not mask the command result.
                    break
        if result is None:
            raise SandboxError("connection lost while running command")
        if check and (result.exit_code != 0 or result.timed_out):
            raise SandboxCommandError(cmd=cmd, result=result)
        return result

    def processes(self) -> List[SandboxProcess]:
        """List the background processes of this sandbox.

        Returns the processes started with [`Sandbox.run`]`(..., background=True)`; stop one
        with [`SandboxProcess.kill`]. Recently completed processes stay listed (with
        `running=False` and their `exit_code`); the server keeps a bounded number of them, so a
        sandbox that has run thousands of short commands will not list them all.
        """
        data = self._request("GET", "/processes").json()
        return [
            SandboxProcess(
                id=p.get("id"),
                pid=p["pid"],
                cmd=p["cmd"],
                tag=p.get("tag"),
                started_at_ms=p.get("started_at_ms"),
                running=p["running"],
                exit_code=p.get("exit_code"),
                _sandbox=self,
            )
            for p in data
        ]

    # ------------------------------------------------------------------ misc

    @property
    def image(self) -> str | None:
        return self._server.image

    @property
    def host_id(self) -> str | None:
        """For a shared/pool sandbox, the job id of the host running it (else None)."""
        return self._server.job_id if self._local_id is not None else None

    # ------------------------------------------------------------------ port proxy

    def proxy_url_for(self, port: int | str, path: str = "/", *, scheme: str = "https://") -> str:
        """Public URL that proxies through to a server running *inside* this sandbox.

        Requests to the returned URL are forwarded by the in-job sandbox server to a
        server you started in the sandbox on `port`, including WebSocket (`ws(s)://`)
        upgrades and streamed responses. Pair it with [`proxy_headers`] for auth.

        How the sandbox must listen on `port`:

        - **Pool / shared sandbox**: it cannot bind a TCP port (Landlock), so bind a
          **unix socket** at `$SBX_PROXY_DIR/<port>.sock` (the `SBX_PROXY_DIR` env var
          is set in every sandbox). E.g. `uvicorn app:app --uds $SBX_PROXY_DIR/8000.sock`.
        - **Dedicated sandbox**: bind a normal TCP port on `127.0.0.1:<port>`. (You can
          also expose the port directly via the job proxy without going through here.)

        Args:
            port (`int` or `str`):
                The port (pool: the `<port>` of the unix socket) the inner server listens on.
            path (`str`, *optional*, defaults to `"/"`):
                Path on the inner server to point at, e.g. `"/ws"`.
            scheme (`str`, *optional*, defaults to `"https://"`):
                URL scheme to build the link with. Defaults to `"https://"`; pass
                `"wss://"` for a WebSocket client (the proxy is protocol-agnostic, so only
                the client-side scheme changes).

        Returns:
            `str`: a URL like `https://<job_id>--49983.hf.jobs/v1/.../proxy/8000/ws` (or
            `wss://...` with `scheme="wss://"`).

        Example:
            ```python
            >>> url = sandbox.proxy_url_for(8000, "/ws", scheme="wss://")
            >>> import websockets
            >>> async with websockets.connect(url, additional_headers=sandbox.proxy_headers) as ws:
            ...     await ws.send("hello")
            ```
        """
        # base_url is always https://...; swap in the requested scheme (e.g. wss://) for the client.
        host_and_rest = self._server.base_url.split("://", 1)[-1]
        path = path if path.startswith("/") else "/" + path
        return f"{scheme}{host_and_rest}{self._base_path}/proxy/{port}{path}"

    @property
    def proxy_headers(self) -> dict[str, str]:
        """Auth headers to send with [`proxy_url_for`] requests (HF token + sandbox token).

        Read these at the moment you use them: the HF token is resolved on access,
        so a long-lived handle hands out a current bearer rather than one captured
        when the sandbox was created.

        For a pooled sandbox this is the sandbox's own capability token, not the
        pool host's: these headers typically end up in a browser or WebSocket
        client, so they should confer access to this sandbox and nothing else.
        """
        return {
            "Authorization": f"Bearer {_effective_token(self._server._api)}",
            "X-Sandbox-Token": self._sandbox_token,
        }

    def __repr__(self) -> str:
        return f"Sandbox(id={self.id!r}, image={self.image!r})"

    # ------------------------------------------------------------------ internals

    def _request(self, method: str, resource: str, **kwargs) -> httpx2.Response:
        return self._server.request(method, self._base_path + resource, sandbox_token=self._sandbox_token, **kwargs)

    @contextmanager
    def _stream(self, method: str, resource: str, **kwargs) -> Iterator[httpx2.Response]:
        with self._server.stream(
            method, self._base_path + resource, sandbox_token=self._sandbox_token, **kwargs
        ) as response:
            yield response


class SandboxPool:
    """A fleet of shared "host" jobs, each packing many landlock-isolated sandboxes.

    > [!WARNING]
    > The Sandbox API is experimental. Its API and behavior may change without notice.
    >
    > **Pooled sandboxes are for workloads inside one trust boundary.** A pooled sandbox is a uid plus a Landlock
    > ruleset inside a shared VM — not a VM of its own — and every sandbox on a host shares that host's auth token
    > and its privileged control plane. Use a pool to fan out *your own* code cheaply. For mutually distrusting
    > workloads, use [`Sandbox.create`], which gives each one its own VM. The specific gaps are listed under
    > "Known limitations" in the sandbox conceptual guide.

    One host is one billed HF Job (a VM); it runs the sandbox server and multiplexes
    up to `sandboxes_per_host` lightweight sandboxes, isolated from each other by
    uid + the Landlock LSM. This makes large fan-outs cheap (the VM cost is shared
    across all its sandboxes) and fast (creating a sandbox is ~one proxy round-trip
    once a host is warm). Best for many parallel CPU sandboxes such as RL rollouts;
    for GPU or strong VM-level isolation between mutually-distrusting workloads, use
    [`Sandbox.create`] instead.

    The constructor pre-provisions `warm_up` hosts (default 1) and blocks until they are
    ready; further hosts are then provisioned on demand as sandboxes are requested, and all
    are torn down on `close()` (or when idle, via `idle_timeout`). The user never manages jobs:

    ```python
    >>> from huggingface_hub import SandboxPool
    >>> with SandboxPool(image="python:3.12", flavor="cpu-basic", warm_up=2) as pool:
    ...     boxes = [pool.create() for _ in range(100)]   # packed across the warm hosts
    ...     print(boxes[0].run("echo hi").stdout)
    hi
    ```

    `create()` makes **one** sandbox at a time: it reuses a host that still has free
    capacity before booting a new one, so you grow on demand as work arrives. To avoid
    a cold start on the first few calls, pre-provision hosts with `warm_up` (or
    [`warm`]). Warm hosts are discovered via job labels, so reuse works **across
    processes** too (a fresh pool with the same `image`/`flavor`/`name` attaches to
    hosts an earlier run left behind):

    ```python
    >>> pool = SandboxPool(image="python:3.12")
    >>> sbx = pool.create()    # finds a warm host (here or in another process), else boots one
    ```
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        flavor: str = "cpu-basic",
        sandboxes_per_host: int = DEFAULT_SANDBOXES_PER_HOST,
        warm_up: int = 1,
        max_hosts: int | None = None,
        name: str | None = None,
        idle_timeout: int | float | str | None = DEFAULT_IDLE_TIMEOUT,
        namespace: str | None = None,
        start_timeout: float = 120.0,
        adopt_hosts: str = ADOPT_OWN,
        token: str | None = None,
        _connect_mode: bool = False,
    ) -> None:
        """Configure a pool and pre-provision `warm_up` hosts (blocks until they are ready).

        Env/secrets are *not* set here: they belong to each sandbox and are passed to
        `create(env=...)`, so sandboxes in the same pool can have different environments.

        Args:
            image (`str`, *optional*, defaults to `"python:3.12"`):
                Docker image for the hosts (needs `/bin/sh`). All sandboxes in the
                pool share this image.
            flavor (`str`, *optional*, defaults to `"cpu-basic"`):
                Hardware flavor for the host jobs (e.g. `"cpu-basic"`).
            sandboxes_per_host (`int`, *optional*, defaults to `50`):
                How many sandboxes to pack per host (per VM density).
            warm_up (`int`, *optional*, defaults to `1`):
                How many hosts to pre-provision in the constructor (which blocks
                until they are ready), so an initial burst of `create()` calls doesn't pay
                a host cold start each. Existing warm hosts (from the cache / other processes)
                count towards it, so only the shortfall is booted; capped by `max_hosts`.
                Defaults to 1 (a single host).
            max_hosts (`int`, *optional*):
                Optional cap on the number of host jobs (a cost ceiling). When
                reached and all hosts are full, `create()` raises. The cap is enforced
                per-process against the hosts this handle knows about, so two processes
                using the same pool concurrently can exceed it. (Per-host density,
                `sandboxes_per_host`, *is* enforced by the host itself.)
            name (`str`, *optional*):
                Pool name, used as the `hf-sandbox-pool` job label so the pool is
                discoverable (e.g. `hf sandbox pool ls`, `connect()`). `create()` reuses
                running hosts carrying this label (including from other processes) before
                booting new ones, so distinct names keep separate pools from sharing hosts.
                A random name is generated when omitted.
            idle_timeout (`int` or `float` or `str`, *optional*, defaults to `600`):
                Host idle timeout — a host shuts down once it has had no
                sandboxes for this long (a billing backstop). Each sandbox also has its
                own idle timeout, set at `create()`. Pass `None` to disable.
            namespace (`str`, *optional*):
                User or org namespace to run hosts under.
            start_timeout (`float`, *optional*, defaults to `120.0`):
                Max seconds to wait for a host to become ready.
            adopt_hosts (`str`, *optional*, defaults to `"own"`):
                Which already-running hosts `create()` may pack onto. Hosts are found by
                filtering Jobs on labels, and labels are set by whoever creates the Job — so a
                label match is a claim, not proof. This decides how much more than the claim is
                required:

                - `"own"` (default): only Jobs this principal started, per the Jobs API's
                  `initiator`. Safe in a shared namespace.
                - `"namespace"`: any Job in the namespace carrying the pool's labels, provided
                  its image, flavor, command and exposed URL all match. Restores cross-user
                  host sharing; only use it in a namespace whose members you trust, since any
                  of them can publish a Job your client will then send a token to.
                - `"never"`: no adoption at all — this handle only uses hosts it booted itself.
            token (`str`, *optional*):
                HF token override.
        """
        if adopt_hosts not in (ADOPT_OWN, ADOPT_NAMESPACE, ADOPT_NEVER):
            raise ValueError(
                f"adopt_hosts must be one of {ADOPT_OWN!r}, {ADOPT_NAMESPACE!r}, {ADOPT_NEVER!r}; got {adopt_hosts!r}."
            )
        if sandboxes_per_host < 1:
            raise ValueError("sandboxes_per_host must be >= 1.")
        if warm_up < 1:
            raise ValueError("warm_up must be >= 1.")
        self._api = HfApi(token=token)
        self.image = image
        self.flavor = flavor
        self.sandboxes_per_host = sandboxes_per_host
        self._warm_up = warm_up
        self.max_hosts = max_hosts
        self.name = name if name is not None else f"pool-{token_hex(6)}"
        self._idle_timeout = idle_timeout
        self._namespace = namespace
        # Security context the local cache is keyed by: entries written by another endpoint,
        # credential or namespace are a miss, not a hit (see `CacheContext`).
        self._cache_context = _cache_context(self._api, namespace)
        self._start_timeout = start_timeout
        self._adopt_hosts = adopt_hosts
        # Resolved lazily and only when `adopt_hosts="own"` needs it: whoami() is
        # heavily rate-limited, and a pool that never discovers never needs it.
        self._principal: str | None = None
        self._principal_resolved = False
        self._hosts: List[_SandboxServer] = []
        self._lock = threading.Lock()
        # Held across the whole one-time warm-up so concurrent first create() calls block until
        # it finishes (and see the warm hosts) instead of racing past a half-set flag.
        self._warmup_lock = threading.Lock()
        # Serializes on-demand host creation within the process: a burst of create() calls that
        # all find every host full boots one host at a time (each new host frees
        # `sandboxes_per_host` slots for the threads still queued behind the lock) instead of
        # one host per thread. The CLI is a single process per command, so this is a no-op there.
        self._boot_lock = threading.Lock()
        self._closed = False
        # Whether the one-time warm-up (seed cache + pre-provision `warm_up` hosts) ran.
        self._warmed_up = False
        # Set for connect()'d handles: the pool must already exist, so create() refuses to
        # silently resurrect it by booting a fresh host if discovery confirms every host is gone.
        # Such a handle also never boots during construction — it attaches lazily on create().
        self._require_live_host = _connect_mode
        # Whether close()/`__exit__` cancels the host jobs. True for a pool we created; False
        # for a connect()'d handle, which only releases its HTTP clients on exit — the shared
        # hosts (possibly serving other clients) are left running, like Sandbox.connect().
        # Kept as the default for hosts this handle boots (a connect()'d handle
        # boots nothing during construction), but teardown is decided per host.
        self._owns_hosts = not _connect_mode
        # In-flight create/boot operations. `close()` waits for these to drain, so
        # a host booted by a create() that was already past its closed check
        # cannot land in a list nobody will ever tear down.
        self._in_flight = 0
        self._drained = threading.Condition(self._lock)
        # Job ids of cached hosts we found dead this session; pruned from the cache on save.
        self._dead_host_ids: set[str] = set()

        # Pre-provision `warm_up` hosts now so the constructor returns only once the pool is
        # warm. A connect()'d handle skips this: it must never boot during construction (it can
        # only attach to existing hosts, lazily on the first create()).
        if not _connect_mode:
            self._ensure_warmed_up()

    # ------------------------------------------------------------------ public API

    @classmethod
    def connect(
        cls,
        pool_id: str,
        *,
        namespace: str | None = None,
        adopt_hosts: str = ADOPT_OWN,
        token: str | None = None,
    ) -> "SandboxPool":
        """Reattach to a running pool by id, from any machine — no local state needed.

        Finds a running host labelled with `pool_id` and rebuilds the pool's config
        (image/flavor/density/host-idle) from that host job's spec and env vars, returning
        a [`SandboxPool`] ready to `create()` more sandboxes — packing onto the running
        hosts, or booting a duplicate (same config) when they are full.

        Raises [`SandboxError`] if no running host is found (a pool stops existing once
        all of its hosts are gone — idle-timed-out or killed).

        Args:
            pool_id (`str`):
                The id returned when the pool was first created.
            namespace (`str`, *optional*):
                Namespace to search for the pool's hosts (defaults to yours).
            adopt_hosts (`str`, *optional*, defaults to `"own"`):
                Which hosts may be attached to. See [`SandboxPool`]. Reattaching to a pool whose
                hosts another member of the namespace started needs `"namespace"`.
            token (`str`, *optional*):
                HF token override.
        """
        api = HfApi(token=token)
        # Fast path: rebuild the pool from the local best-effort cache, with no HTTP at all.
        # Only an entry written by this endpoint, credential and namespace is a hit (see
        # `CacheContext`), so `namespace=` is honoured rather than being overridden by whatever
        # the file says -- a `connect(namespace="org-B")` can no longer be served org-A's hosts.
        # The cached hosts are seeded (and validated) lazily on the next create(); if they are
        # all stale, create() falls back to label discovery exactly like a cold connect would.
        cache = read_pool_cache(pool_id, _cache_context(api, namespace))
        if cache is not None:
            return cls(
                image=cache.image,
                flavor=cache.flavor,
                sandboxes_per_host=cache.sandboxes_per_host,
                max_hosts=cache.max_hosts,
                name=pool_id,
                idle_timeout=cache.idle_timeout,
                namespace=namespace,
                adopt_hosts=adopt_hosts,
                token=token,
                _connect_mode=True,  # attach to existing hosts; never boot during construction
            )

        # Cold path: find a running host via labels and rebuild the config from its job spec.
        job = _find_pool_host_job(api, pool_id, namespace=namespace, policy=adopt_hosts)
        env = _host_env(api, job, namespace=namespace)
        idle_raw = env.get("SBX_IDLE_TIMEOUT")
        max_hosts_raw = env.get("SBX_MAX_HOSTS")
        return cls(
            image=job.docker_image or job.space_id or DEFAULT_IMAGE,
            flavor=str(job.flavor) if job.flavor is not None else "cpu-basic",
            sandboxes_per_host=int(env.get("SBX_CAPACITY", DEFAULT_SANDBOXES_PER_HOST)),
            max_hosts=int(max_hosts_raw) if max_hosts_raw is not None else None,
            name=pool_id,
            idle_timeout=int(idle_raw) if idle_raw is not None else None,
            namespace=namespace,
            adopt_hosts=adopt_hosts,
            token=token,
            _connect_mode=True,  # attach to existing hosts; never boot during construction
        )

    def warm(self, num_hosts: int = 1) -> List[str]:
        """Ensure `num_hosts` empty host(s) are running and leave them running. Returns the
        pool's host job ids.

        Used to "create" a pool up front: the hosts carry the pool label and config (in
        their env vars), so a later `SandboxPool.connect(pool_id)` (even from another
        machine) finds them and spawns sandboxes without a cold start. The hosts keep
        billing until killed or idle.

        Adopts hosts already running for this pool (found via job labels) before booting,
        so a `warm()` after `connect()` — or a repeated `warm()` — tops up to `num_hosts`
        instead of duplicating live hosts and blowing past `max_hosts`.
        """
        if self._closed:
            raise SandboxError("This SandboxPool is closed.")
        self._discover_hosts()
        with self._lock:
            shortfall = num_hosts - len(self._hosts)
        if shortfall > 0:
            booted = self._provision_hosts(shortfall, allow_partial=True)
            with self._lock:
                self._hosts.extend(booted)
        with self._warmup_lock:
            self._warmed_up = True  # explicit warm-up satisfies create()'s one-time warm-up
        self._save_cache()
        return self.host_ids

    def create(
        self,
        *,
        env: dict[str, Any] | None = None,
        idle_timeout: int | float | str | None = DEFAULT_IDLE_TIMEOUT,
        forward_hf_token: bool = False,
    ) -> "Sandbox":
        """Create one sandbox, provisioning a host if needed.

        Reuses a host with free capacity (this pool's, or a warm host found via job labels
        / the local cache) before booting a new one, so a `create()` against a warm host
        costs ~one round-trip. Call it repeatedly to fan out; use `warm_up` (or [`warm`])
        to pre-provision hosts and avoid a cold start on the first calls. If a host fills
        up under us (another process packed it) or a cached host is gone, the sandbox is
        re-placed on another host (or a fresh one).

        Args:
            env (`dict[str, Any]`, *optional*):
                Environment variables for this sandbox (each sandbox gets its own).
            idle_timeout (`int` or `float` or `str`, *optional*, defaults to `600`):
                Per-sandbox idle timeout — a sandbox is evicted from its host
                after this much inactivity (no API calls, no running process). Distinct
                from the host idle timeout. Pass `None` to disable.
            forward_hf_token (`bool`, *optional*, defaults to `False`):
                If True, inject your HF token as `HF_TOKEN` in the sandbox
                (opt-in). Unlike a dedicated sandbox's `secrets`, a pooled sandbox's env is
                delivered to the host server at creation (never stored in the host job), so
                it doesn't appear in any job's metadata.
        """
        with self._lock:
            if self._closed:
                raise SandboxError("This SandboxPool is closed.")
            # Registered under the lock together with the closed check, so
            # `close()` cannot snapshot the host list between the two.
            self._in_flight += 1
        try:
            return self._create_locked(env, idle_timeout, forward_hf_token)
        finally:
            with self._lock:
                self._in_flight -= 1
                self._drained.notify_all()

    def _create_locked(
        self,
        env: dict[str, Any] | None,
        idle_timeout: int | float | str | None,
        forward_hf_token: bool,
    ) -> "Sandbox":
        sandbox_env = dict(env or {})
        if forward_hf_token:
            sandbox_env["HF_TOKEN"] = _effective_token(self._api)
        # None disables per-sandbox idle eviction (mirrors Sandbox.create): leave it out of
        # the create body entirely rather than sending 0, which the server reads as "evict now".
        idle_secs = _duration_to_secs(idle_timeout) if idle_timeout is not None else None

        new_hosts: List[_SandboxServer] = []
        discovered = False
        rounds = 0
        try:
            # One-time warm-up: seed from cache (no HTTP) and pre-provision `warm_up` hosts.
            # Warm-up hosts are pool-level (like `warm()`), so they survive a failed create().
            self._ensure_warmed_up()
            while True:
                # 1. Reserve a slot on a host we already track.
                host = self._reserve_one()
                # 2. If none free, attach (once) to warm hosts discovered via labels.
                if host is None and not discovered:
                    discovered = True
                    self._discover_hosts()
                    host = self._reserve_one()
                # 3. Still none? Boot one host (unless we must not resurrect a dead pool).
                if host is None:
                    if self._require_live_host and discovered and not self._hosts:
                        # connect()'d to a pool whose hosts are all gone (stale cache + nothing
                        # found via labels): don't resurrect it under the same id, just report it.
                        delete_pool_cache(self.name, self._cache_context)
                        raise SandboxError(
                            f"No running host found for pool '{self.name}'. The pool has stopped "
                            "(all its hosts were killed or idle-timed-out); create a new one."
                        )
                    host = self._boot_one_host(new_hosts)
                # 4. Create the sandbox on the reserved host. `None` means the host filled up
                #    (another process packed it) or a stale cached host is gone — retry.
                if host is not None:
                    try:
                        sandbox = self._create_one(host, sandbox_env, idle_secs)
                    except Exception:
                        # The create POST failed on a live host: release the slot we optimistically
                        # reserved so the host isn't left permanently one sandbox short.
                        with self._lock:
                            host.live = max(0, host.live - 1)
                        raise
                    if sandbox is not None:
                        self._save_cache()
                        return sandbox
                    # sandbox is None: the host is full (packed by another process) or gone.
                    # Either way it's no longer a host we exclusively own, so stop tracking it
                    # for teardown — cancelling it on a later round's failure would kill another
                    # tenant's sandboxes (full host) or hit an already-dead job (gone host).
                    if host in new_hosts:
                        new_hosts.remove(host)
                discovered = False  # re-scan for room / boot a host next round
                rounds += 1
                if rounds > _MAX_PACK_ROUNDS:
                    raise SandboxError(
                        "Could not place sandbox: hosts kept reporting full. Raise max_hosts or sandboxes_per_host."
                    )
        except Exception:
            # All-or-nothing: tear down hosts we booted in this call.
            with self._lock:
                self._hosts = [h for h in self._hosts if h not in new_hosts]
            for host in new_hosts:
                try:
                    host.cancel_job()
                except Exception:
                    pass
                finally:
                    host.close()
            raise

    @property
    def num_hosts(self) -> int:
        """Number of host jobs currently provisioned."""
        with self._lock:
            return len(self._hosts)

    @property
    def num_sandboxes(self) -> int:
        """Number of sandboxes currently handed out (across all hosts)."""
        with self._lock:
            return sum(host.live for host in self._hosts)

    @property
    def host_ids(self) -> List[str]:
        """Job ids of the provisioned hosts."""
        with self._lock:
            return [host.job_id for host in self._hosts]

    def close(self) -> None:
        """Release the pool. Idempotent.

        For a pool we created, this terminates all host jobs (and therefore all their
        sandboxes). For a `connect()`'d handle it only releases the local HTTP clients: the
        shared hosts may be serving other clients, so — like [`Sandbox.connect`] — leaving a
        `with` block must not tear them down. Terminate a connected pool's hosts explicitly
        with `hf sandbox pool delete <id>`.

        Only hosts *this handle started* are cancelled. A host discovered via labels may
        be serving another process' sandboxes, so it is released rather than terminated.

        Raises [`SandboxError`] if a host job could not be cancelled, naming the jobs that
        are still running — they keep billing, and their cache entries are kept so they stay
        discoverable. When `close()` is reached through `__exit__` with an exception already
        in flight, the failure is logged instead, so it cannot mask the original error.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Wait for in-flight creates so a host booted by one that was already
            # past its closed check lands in `self._hosts` before the snapshot,
            # rather than being orphaned. Bounded, because a wedged create must
            # not hang process exit.
            drain_timeout = self._start_timeout + _CLOSE_DRAIN_TIMEOUT
            if self._in_flight and not self._drained.wait_for(lambda: self._in_flight == 0, timeout=drain_timeout):
                logger.warning(
                    f"Pool '{self.name}': {self._in_flight} operation(s) still in flight after "
                    f"{drain_timeout}s; any host they boot will keep running."
                )
            hosts = self._hosts
            self._hosts = []

        survivors: List[_SandboxServer] = []
        failures: List[str] = []
        for host in hosts:
            try:
                # Only what this handle started. A discovered host may be serving
                # someone else's sandboxes, and cancelling it kills them.
                if host.owned:
                    host.cancel_job()
            except Exception as e:
                # Keep it in the cache: that record is the user's only handle on a
                # job that is still running and still billing.
                survivors.append(host)
                failures.append(f"{host.job_id} ({e})")
                logger.warning(f"Failed to cancel sandbox host {host.job_id}: {e}")
            finally:
                host.close()

        if survivors:
            # Rewrite rather than delete, so the survivors stay discoverable.
            self._save_cache(hosts=survivors)
        elif self._owns_hosts:
            # The pool's hosts are gone; don't leave a stale entry.
            delete_pool_cache(self.name, self._cache_context)
        if failures:
            raise SandboxError(
                f"Pool '{self.name}' could not terminate host job(s): {'; '.join(failures)}. "
                "They are still running and billing; cancel them with `hf jobs cancel <id>`."
            )

    def __enter__(self) -> "SandboxPool":
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            self.close()
        except SandboxError:
            # Never let a teardown failure mask the exception that is already
            # propagating out of the `with` block.
            if exc_info[0] is None:
                raise
            logger.warning(f"Pool '{self.name}' teardown failed while handling another error.", exc_info=True)

    # ------------------------------------------------------------------ internals

    def _ensure_warmed_up(self) -> None:
        """One-time: seed from the cache and pre-provision up to `warm_up` hosts.

        Cheap by default: seeding is local (no HTTP), and if the cache already gives at
        least `warm_up` hosts we trust them and skip discovery/booting (dead ones are
        pruned lazily by `create()`). Only when short do we list_jobs and boot the
        shortfall in parallel, capped by `max_hosts`.

        Runs at most once per pool, under `_warmup_lock`: concurrent first `create()` calls
        block here until the warm-up completes — and see its hosts — rather than racing past
        a half-set flag and each booting their own. `_warmed_up` is only set once the work
        succeeds, so a failed warm-up is retried (seeding/discovery dedupe by job id).

        Warm-up hosts are pool-level (like [`warm`]): they are not torn down if the
        triggering `create()` later fails — `close()` (or the `with` block) reclaims them.
        """
        with self._warmup_lock:
            if self._warmed_up:
                return
            self._seed_hosts_from_cache()
            target = self._warm_up if self.max_hosts is None else min(self._warm_up, self.max_hosts)
            with self._lock:
                shortfall = target - len(self._hosts)
            if shortfall > 0:
                self._discover_hosts()
                # A connect()'d pool must never boot during warm-up: it may only attach to
                # existing hosts. Booting is left to create()'s loop, guarded by
                # `_require_live_host`, so a stopped pool is reported instead of resurrected.
                if not self._require_live_host:
                    with self._lock:
                        shortfall = target - len(self._hosts)
                    if shortfall > 0:
                        booted = self._provision_hosts(shortfall, allow_partial=True)
                        with self._lock:
                            self._hosts.extend(booted)
            self._warmed_up = True
        # Persist the warmed hosts so the next process (e.g. `hf sandbox create --pool` after a
        # bare `hf sandbox pool create`) hits the cache fast path instead of re-discovering.
        if self._hosts:
            self._save_cache()

    def _principal_id(self) -> str | None:
        """Id of the principal this pool authenticates as, for the `"own"` policy."""
        if not self._principal_resolved:
            try:
                self._principal = self._api.whoami(cache=True).get("id")
                self._principal_resolved = self._principal is not None
            except Exception as e:
                # Leave it None: `_adoptable` then refuses rather than guessing.
                logger.warning(f"Could not resolve the current user to validate pool hosts: {e}")
        return self._principal

    def _adoptable(self, job: JobInfo) -> bool:
        """Whether `job` may be adopted as one of this pool's hosts."""
        reason = _host_rejection(
            job,
            policy=self._adopt_hosts,
            principal_id=self._principal_id() if self._adopt_hosts == ADOPT_OWN else None,
            namespace=self._namespace,
            image=self.image,
            flavor=self.flavor,
        )
        if reason is not None:
            logger.debug(f"Not adopting job {job.id} as a host for pool '{self.name}': {reason}.")
            return False
        return True

    def _adoptable_pending(self, job: JobInfo) -> bool:
        """Like `_adoptable`, minus the checks a not-yet-running job cannot pass."""
        reason = _host_rejection(
            job,
            policy=self._adopt_hosts,
            principal_id=self._principal_id() if self._adopt_hosts == ADOPT_OWN else None,
            namespace=self._namespace,
            image=self.image,
            flavor=self.flavor,
            check_url=False,
        )
        if reason is not None:
            logger.debug(f"Not waiting for job {job.id} as a host for pool '{self.name}': {reason}.")
            return False
        return True

    def _reserve_one(self) -> "_SandboxServer | None":
        """Reserve one slot on the first host with free capacity (under lock), else None."""
        with self._lock:
            for host in self._hosts:
                if host.capacity - host.live > 0:
                    host.live += 1
                    return host
        return None

    def _boot_one_host(self, new_hosts: List["_SandboxServer"]) -> "_SandboxServer | None":
        """Provision (or adopt) one host and reserve a slot on it. Returns None to retry.

        Held under `_boot_lock` so that, within a process, only one host is booted at a time:
        a burst of create() calls that all found every host full queue here, and each new host
        frees `sandboxes_per_host` slots for the threads still waiting — so they reuse it instead
        of each booting their own. Before booting, we reuse a slot freed by a concurrent boot and,
        failing that, adopt a host already SCHEDULING for this pool (here or in another process)
        rather than piling on a duplicate.
        """
        with self._boot_lock:
            # A concurrent boot (this process) may have added a host with free slots while we
            # waited for the lock — reuse it instead of booting another.
            host = self._reserve_one()
            if host is not None:
                return host
            # A host already coming up for this pool (ours, or another process'): wait for it
            # to start and adopt it instead of booting a duplicate.
            if self._adopt_pending_host():
                host = self._reserve_one()
                if host is not None:
                    return host
            booted = self._provision_hosts(1)
            new_hosts.extend(booted)
            with self._lock:
                self._hosts.extend(booted)
            return self._reserve_one()

    def _adopt_pending_host(self) -> bool:
        """Wait for and adopt a host already SCHEDULING for this pool, if any.

        Avoids over-provisioning when a host is already on its way up for this pool (started by
        another process, or an earlier create() in this one): rather than booting a duplicate,
        wait for it to reach RUNNING and adopt it via discovery. Returns True if a pending host
        was found (whether or not it eventually came up), False if none is scheduling.
        """
        known = {host.job_id for host in self._hosts}
        pending = next(
            (
                job
                for job in self._api.list_jobs(
                    status="SCHEDULING",
                    labels={MODE_LABEL: MODE_POOL, POOL_LABEL: self.name},
                    namespace=self._namespace,
                )
                # A SCHEDULING job has no exposed URL yet, so only the ownership
                # and spec checks apply here; discovery re-validates it in full
                # once it is RUNNING.
                if job.id not in known and self._adoptable_pending(job)
            ),
            None,
        )
        if pending is None:
            return False
        logger.debug(f"Pool '{self.name}' host {pending.id} is already SCHEDULING; waiting for it instead of booting.")
        deadline = time.time() + self._start_timeout
        while time.time() < deadline:
            stage = self._api.inspect_job(job_id=pending.id, namespace=self._namespace).status.stage
            if stage not in ("SCHEDULING", "RUNNING"):
                return False  # it died before starting up; let the caller boot a fresh one
            if stage == "RUNNING":
                # RUNNING precedes "server ready" (the host still has to fetch + exec sbx-server),
                # so discovery only adopts it once its server answers — keep polling until it does.
                self._discover_hosts()
                if any(host.job_id == pending.id for host in self._hosts):
                    return True
            time.sleep(1.0)
        return False

    def _discover_hosts(self) -> None:
        """Attach to running host jobs carrying this pool's labels.

        Lets `create()` reuse a host warmed by an earlier call or another process
        instead of booting a new one. Hosts are found via job labels; each adopted
        host's free capacity is read from the server, so packing stays accurate.

        Labels select candidates; creator, job specification and proxy URL admission
        checks decide which candidates can receive credentials. Adopted hosts are released
        by close(), not cancelled, because another handle may still be using them.
        """
        known = {host.job_id for host in self._hosts}
        matches = [
            job
            for job in self._api.list_jobs(
                status="RUNNING",
                labels={MODE_LABEL: MODE_POOL, POOL_LABEL: self.name},
                namespace=self._namespace,
            )
            if job.id not in known and self._adoptable(job)
        ]

        for job in matches:
            server = None
            try:
                server = _connect_host(self._api, job.id, namespace=self._namespace)
                # Capacity comes from the host's SBX_CAPACITY env; otherwise an adopted host
                # would silently pack at the pool default and mis-pack.
                env = _host_env(self._api, job, namespace=self._namespace)
                server.capacity = int(env.get("SBX_CAPACITY", self.sandboxes_per_host))
                server.live = len(server.request("GET", "/v1/sandboxes").json())
            except (SandboxError, httpx2.HTTPError, HfHubHTTPError) as e:
                # Host died (e.g. deleted between list_jobs and inspect_job), is still starting
                # up, or is unreachable: skip it, closing the client if one was opened.
                logger.debug(f"Skipping host {job.id} during discovery: {e}")
                if server is not None:
                    server.close()
                continue
            with self._lock:
                if any(host.job_id == job.id for host in self._hosts):
                    server.close()  # adopted concurrently by another thread
                else:
                    self._hosts.append(server)

    def _live_host_count(self) -> int:
        """How many hosts are running for this pool, not just the ones we track.

        `max_hosts` is documented as a cost ceiling, but it was compared against
        `len(self._hosts)` -- an in-process count. Two processes using the same
        pool each counted only their own hosts, so the number of billed host jobs
        could quietly exceed the cap by a factor of however many processes were
        involved.

        Counting via labels narrows that considerably. It does not close it: two
        processes can still list simultaneously and both decide there is room. A
        hard cap has to be enforced where the jobs are created, not by the
        clients racing to count them.
        """
        with self._lock:
            tracked = {host.job_id for host in self._hosts}
        try:
            for job in self._api.list_jobs(
                status="RUNNING",
                labels={MODE_LABEL: MODE_POOL, POOL_LABEL: self.name},
                namespace=self._namespace,
            ):
                tracked.add(job.id)
        except Exception as e:
            # Fall back to the in-process count rather than refusing to boot.
            logger.debug(f"Could not count live hosts for pool '{self.name}': {e}")
        return len(tracked)

    def _provision_hosts(self, num_new: int, *, allow_partial: bool = False) -> List[_SandboxServer]:
        """Boot `num_new` host jobs in parallel, respecting `max_hosts`."""
        if self.max_hosts is not None:
            current = self._live_host_count()
            if allow_partial:
                num_new = min(num_new, max(0, self.max_hosts - current))
            if current + num_new > self.max_hosts:
                allowed = self.max_hosts - current
                raise SandboxError(
                    f"Pool needs {num_new} more host(s) but max_hosts={self.max_hosts} "
                    f"allows only {max(0, allowed)} more ({current} already running for this pool). "
                    "Raise max_hosts or kill some sandboxes."
                )
        if num_new == 0:
            return []
        # Boot in parallel, but collect every result before re-raising: if one boot fails,
        # the others may have already started billable host jobs, so cancel those instead of
        # leaking them (executor.map would surface the first error and drop the rest).
        with ThreadPoolExecutor(max_workers=min(num_new, 32)) as executor:
            futures = [executor.submit(self._boot_host) for _ in range(num_new)]
        booted: List[_SandboxServer] = []
        error: Exception | None = None
        for future in futures:
            try:
                booted.append(future.result())
            except Exception as e:
                error = e
        if error is not None:
            for server in booted:
                try:
                    server.cancel_job()
                except Exception:
                    pass
                finally:
                    server.close()
            raise error
        return booted

    def _boot_host(self) -> _SandboxServer:
        """Start one host job and wait until its server is ready."""
        hf_token = _effective_token(self._api)
        nonce = token_hex(16)
        sandbox_token = _derive_sandbox_token(hf_token, nonce)
        command, job_env, job_secrets, job_volumes = _bootstrap_job_spec(
            self._api,
            hf_token,
            env=None,  # host job carries no user env; sandboxes get env per create request
            secrets=None,
            volumes=None,
            idle_timeout=self._idle_timeout,
            forward_hf_token=False,
            sandbox_token=sandbox_token,
        )
        # Host mode: per-sandbox idle eviction + empty-host shutdown (vs the dedicated
        # whole-job watchdog).
        job_env["SBX_HOST_MODE"] = "1"
        # Server-authoritative packing cap: the host refuses creates past this (the client
        # then packs onto / boots another host). Avoids cross-process over-commit. Read
        # back (with SBX_IDLE_TIMEOUT) by `connect()` to rebuild the pool — labels stay for
        # filtering only.
        job_env["SBX_CAPACITY"] = str(self.sandboxes_per_host)
        # Persist the optional cost ceiling on the host too, so a later connect() (e.g. the CLI
        # `pool create --max-hosts N` then `create --pool <id>` flow) rebuilds the cap instead of
        # defaulting to unlimited and provisioning past it.
        if self.max_hosts is not None:
            job_env["SBX_MAX_HOSTS"] = str(self.max_hosts)
        labels = {SANDBOX_LABEL: "1", MODE_LABEL: MODE_POOL, POOL_LABEL: self.name, NONCE_LABEL: nonce}
        job = self._api.run_job(
            image=self.image,
            command=command,
            env=job_env,
            secrets=job_secrets,
            flavor=self.flavor,
            timeout=SANDBOX_MAX_LIFETIME,
            labels=labels,
            volumes=job_volumes or None,
            expose=[SANDBOX_SERVER_PORT],
            namespace=self._namespace,
        )
        server: "_SandboxServer | None" = None
        try:
            # from_job is inside the try: if it raises (e.g. the server port isn't exposed),
            # the already-started host job must still be cancelled so it doesn't linger billing.
            server = _SandboxServer.from_job(
                job=job,
                nonce=nonce,
                sandbox_token=sandbox_token,
                api=self._api,
                max_connections=min(self.sandboxes_per_host + 8, 256),
                capacity=self.sandboxes_per_host,
            )
            server.wait_ready(self._start_timeout)
        except Exception:
            try:
                self._api.cancel_job(job_id=job.id, namespace=job.owner.name)
            except Exception as e:
                logger.warning(f"Failed to cancel sandbox host {job.id} after startup failure: {e}")
            if server is not None:
                server.close()
            raise
        # We started this job, so this handle is the one that should end it.
        server.owned = self._owns_hosts
        return server

    def _create_one(self, host: "_SandboxServer", env: dict[str, Any], idle_secs: int | None) -> "Sandbox | None":
        """Create one sandbox on a reserved `host`. Returns the sandbox, or None to retry.

        None means either the host filled up between our reservation and the create
        (server-authoritative capacity — another client packed it, so we mark it full and
        place the sandbox elsewhere) or a host rebuilt from the cache is gone/unreachable
        (dropped and re-placed via discovery / a fresh boot).
        """
        body: dict[str, Any] = {"count": 1}
        if idle_secs is not None:
            body["idle_timeout_secs"] = idle_secs
        if env:
            body["env"] = env
        try:
            data = host.request("POST", "/v1/sandboxes", json=body).json()
        except (SandboxError, httpx2.HTTPError) as e:
            if host.verified:
                raise  # a host we booted/discovered this session failing is a real error
            logger.debug(f"Dropping unreachable cached host {host.job_id}: {e}")
            self._drop_host(host)
            return None
        host.verified = True  # it answered, so it's a real live host now
        sandboxes = data.get("sandboxes") or []
        if int(data.get("rejected", 0)) or not sandboxes:
            with self._lock:
                host.live = host.capacity  # the host is full; stop reserving it
            return None
        item = sandboxes[0]
        sandbox = Sandbox(
            id=f"{host.job_id}{SHARED_ID_SEP}{item['id']}",
            server=host,
            local_id=item["id"],
            owns_sandbox=True,
            owns_server=False,
            sandbox_token=item.get("token"),
        )
        sandbox._on_kill = self._on_sandbox_killed
        return sandbox

    def _on_sandbox_killed(self, sandbox: Sandbox) -> None:
        """Free the packing slot of a shared sandbox that was killed."""
        with self._lock:
            sandbox._server.live = max(0, sandbox._server.live - 1)

    def _drop_host(self, host: _SandboxServer) -> None:
        """Forget a host found dead this session (and mark it for cache pruning)."""
        with self._lock:
            self._hosts = [h for h in self._hosts if h is not host]
            self._dead_host_ids.add(host.job_id)
        host.close()

    # ------------------------------------------------------------------ cache (best effort)

    def _seed_hosts_from_cache(self) -> None:
        """Adopt the pool's cached hosts without any HTTP (rebuilt from cached URL + nonce).

        The first thing that happens to a seeded host is an authenticated `POST`, so what the
        file says about *where* it lives is not taken on trust: the URL must be one only this
        job can have (`_cached_host_rejection`), and an entry older than `HOST_TRUST_TTL` is
        re-checked against the Jobs API first (`_confirm_cached_host`). A fresh entry — the
        `pool create` → `create --pool` case the cache exists for — is used as is.

        Liveness stays best-effort: a seeded host is confirmed by its first successful request
        and dropped on the first failure (see `_create_one`). Hosts the cache believes are
        full are skipped — label discovery re-checks them with fresh counts if the seeded ones
        don't satisfy the request, so a stale-full entry never blocks a create.
        """
        cache = read_pool_cache(self.name, self._cache_context)
        if cache is None:
            return
        hf_token = _effective_token(self._api)
        now = time.time()
        for ch in cache.hosts:
            with self._lock:
                known = {host.job_id for host in self._hosts}
            if ch.job_id in known or ch.live >= ch.capacity:
                continue
            reason = _cached_host_rejection(ch)
            if reason is not None:
                logger.debug(f"Dropping cached host {ch.job_id} of pool '{self.name}': it {reason}.")
                with self._lock:
                    self._dead_host_ids.add(ch.job_id)  # an entry we can never use is litter
                continue
            if now - ch.updated_at > HOST_TRUST_TTL and not self._confirm_cached_host(ch):
                continue
            server = _SandboxServer(
                job_id=ch.job_id,
                owner=ch.owner,
                image=self.image,
                base_url=ch.base_url,
                nonce=ch.nonce,
                sandbox_token=_derive_sandbox_token(hf_token, ch.nonce),
                api=self._api,
                max_connections=min(self.sandboxes_per_host + 8, 256),
                capacity=ch.capacity,
            )
            server.live = ch.live
            server.verified = False
            with self._lock:
                if any(host.job_id == ch.job_id for host in self._hosts):
                    server.close()  # seeded concurrently by another thread
                else:
                    self._hosts.append(server)

    def _confirm_cached_host(self, ch: CachedHost) -> bool:
        """Confirm an aged-out cache entry against the Jobs API before crediting it.

        Freshness is what the no-HTTP path rests on: the entry was written by this same
        principal minutes ago, so the job it names is almost certainly still the host it
        described. Past `HOST_TRUST_TTL` that argument weakens, and the entry becomes a hint
        that has to be checked — the job must still be running, still be labelled for *this*
        pool with the nonce the entry derives its token from, still pass the pool's adoption
        policy, and still expose exactly the cached URL. One round-trip, and only for entries
        the fast path would no longer be honest about.

        A definitive "no" prunes the entry; a transient failure aborts this attempt
        without losing the cached host.
        """
        if self._adopt_hosts == ADOPT_OWN and self._principal_id() is None:
            raise SandboxError("Could not validate cached pool hosts: current user is unavailable; retry later.")
        try:
            job = self._api.inspect_job(job_id=ch.job_id, namespace=ch.owner)
            labels = job.labels or {}
            if job.status.stage != "RUNNING":
                reason = f"its job is not running (status: {job.status.stage})"
            elif labels.get(MODE_LABEL) != MODE_POOL or labels.get(POOL_LABEL) != self.name:
                reason = f"its job is not labelled as a host of pool '{self.name}'"
            elif labels.get(NONCE_LABEL) != ch.nonce:
                reason = "its job carries a different nonce"
            elif not self._adoptable(job):
                reason = "its job is not one this pool may adopt"
            elif (exposed := _find_server_url(job)) != ch.base_url:
                reason = f"its job exposes {exposed!r}, not the cached URL"
            else:
                return True
        except httpx2.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise SandboxError(f"Could not confirm cached host {ch.job_id}; retry later.") from e
            reason = "its job no longer exists"
        except Exception as e:
            raise SandboxError(f"Could not confirm cached host {ch.job_id}; retry later.") from e
        logger.debug(f"Dropping stale cached host {ch.job_id} of pool '{self.name}': {reason}.")
        with self._lock:
            self._dead_host_ids.add(ch.job_id)
        return False

    def _save_cache(self, hosts: "List[_SandboxServer] | None" = None) -> None:
        """Persist the pool config + hosts (with their live counts) for next time.

        `hosts` overrides the tracked list, for the one case where they differ:
        `close()` persisting the hosts it could *not* cancel, since that record is
        the caller's only handle on a job that is still billing.
        """
        with self._lock:
            source = self._hosts if hosts is None else hosts
            entries = [
                CachedHost(
                    job_id=host.job_id,
                    owner=host.owner,
                    base_url=host.base_url,
                    nonce=host.nonce,
                    capacity=host.capacity,
                    live=host.live,
                    updated_at=time.time(),
                )
                for host in source
            ]
            dead = set(self._dead_host_ids)
        save_pool_cache(
            self.name,
            context=self._cache_context,
            image=self.image,
            flavor=self.flavor,
            sandboxes_per_host=self.sandboxes_per_host,
            max_hosts=self.max_hosts,
            idle_timeout=_duration_to_secs(self._idle_timeout) if self._idle_timeout is not None else None,
            hosts=entries,
            dead_host_ids=dead,
        )


def _normalize_image(image: str | None) -> str | None:
    """Compare Docker images leniently: registries and tags get normalized
    server-side, and a mis-parse here must not stop a pool from working."""
    if image is None:
        return None
    image = image.strip().lower()
    for prefix in ("docker.io/", "index.docker.io/", "library/"):
        if image.startswith(prefix):
            image = image[len(prefix) :]
    return image


def _server_host_rejection(url: str, *, job_id: str) -> str | None:
    """Why `url` is not a URL we should send a sandbox job's credentials to.

    The hostname must be derived from the job id we already hold, so nothing but the
    client's own knowledge of which job it is talking to can decide where the credentials
    go. The domain is not hard-coded, so staging endpoints keep working.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return f"uses scheme {parsed.scheme!r}, not https"
    # `urlparse` lowercases the hostname, so compare case-insensitively.
    if not (parsed.hostname or "").startswith(f"{job_id}--{SANDBOX_SERVER_PORT}.".lower()):
        return f"is not job {job_id}'s port-{SANDBOX_SERVER_PORT} URL"
    return None


def _server_url_rejection(job: JobInfo) -> str | None:
    """Why this Job's exposed URL is not one we should send credentials to."""
    urls = list(job.status.expose_urls or []) if job.status is not None else []
    if len(urls) != 1:
        return f"exposes {len(urls)} URL(s), expected exactly one for port {SANDBOX_SERVER_PORT}"
    reason = _server_host_rejection(urls[0], job_id=job.id)
    return f"exposes {urls[0]!r}, which {reason}" if reason is not None else None


def _cached_host_rejection(host: CachedHost) -> str | None:
    """Why a cached host entry must not be rebuilt into a credentialed transport.

    Same reasoning as `_server_url_rejection`, for a URL that comes off the local disk
    instead of the Jobs API. Without this, the cache file gets to choose where the HF bearer
    and the derived host token are sent -- `http://127.0.0.1:1234` included -- and it gets to
    choose it *before* any API call could have contradicted it. Deriving the expected
    hostname from the entry's own `job_id` leaves it able to name only a job this client can
    independently identify, which is the same bar a discovered host has to clear.
    """
    reason = _server_host_rejection(host.base_url, job_id=host.job_id)
    return f"names {host.base_url!r}, which {reason}" if reason is not None else None


def _cache_context(api: HfApi, namespace: str | None) -> CacheContext:
    """The local pool cache's security context for this endpoint/credential/namespace."""
    return CacheContext.for_credential(endpoint=api.endpoint, token=_effective_token(api), namespace=namespace)


def _host_rejection(
    job: JobInfo,
    *,
    policy: str,
    principal_id: str | None,
    namespace: str | None,
    image: str | None,
    flavor: str | None,
    check_url: bool = True,
) -> str | None:
    """Why this Job must not be adopted as one of our pool hosts, or `None`.

    Hosts are found by filtering Jobs on labels, and labels are set by whoever
    creates the Job -- so a label match says "this Job claims to be one of our
    hosts", not "this Job is one of our hosts". Anyone who can create a Job in
    the namespace can make that claim, copy the public nonce from a real host's
    labels, and receive the token derived for that host.

    Everything checked here is asserted by the backend rather than by the Job's
    creator: the initiator, the owner, the image, the flavor, the command, and
    the exposed URL. `initiator` is the load-bearing one -- there is no
    client-side way to set it -- and it is what makes `policy="own"` meaningful.
    """
    if policy == ADOPT_NEVER:
        return "host adoption is disabled (adopt_hosts='never')"

    if policy == ADOPT_OWN:
        initiator = job.initiator
        if initiator is None or initiator.id is None:
            return "the Jobs API did not say who started it, so it cannot be confirmed as ours"
        if principal_id is None:
            return "could not resolve the current principal to compare against its initiator"
        if initiator.id != principal_id:
            return f"started by a different principal ({initiator.name or initiator.id})"

    if namespace is not None and job.owner is not None and job.owner.name != namespace:
        return f"runs under namespace {job.owner.name!r}, not {namespace!r}"

    job_image = job.docker_image or job.space_id
    if image is not None and _normalize_image(job_image) != _normalize_image(image):
        return f"runs image {job_image!r}, not {image!r}"

    if flavor is not None and str(job.flavor) != flavor:
        return f"runs flavor {str(job.flavor)!r}, not {flavor!r}"

    # A sandbox host runs the bootstrap script and nothing else. A Job running
    # anything else is not a host, whatever its labels say.
    if job.command is None or list(job.command) != _bootstrap_command():
        return "does not run the sandbox bootstrap command"

    # A SCHEDULING job has no exposed URL yet; the caller re-checks once it runs.
    return _server_url_rejection(job) if check_url else None


def _effective_token(api: HfApi) -> str:
    token = api.token if isinstance(api.token, str) else get_token()
    if not token:
        raise SandboxError("A Hugging Face token is required to use sandboxes. Run `hf auth login` first.")
    return token


def _bootstrap_job_spec(
    api: HfApi,
    hf_token: str,
    *,
    env: dict[str, Any] | None,
    secrets: dict[str, Any] | None,
    volumes: List[Volume] | None,
    idle_timeout: int | float | str | None,
    forward_hf_token: bool,
    sandbox_token: str,
) -> tuple[list[str], dict[str, Any], dict[str, Any], List[Volume]]:
    """Build the (command, env, secrets, volumes) to launch a job running sbx-server.

    Shared by dedicated sandboxes and shared hosts: both fetch and exec the same unified
    `sbx-server` binary at startup (via `/bin/sh`), downloading it with wget/curl, or
    reading it off the always-mounted server bucket when the image ships neither. Either
    way the download is pinned to `SANDBOX_SERVER_SHA256` and verified before it runs.
    """
    # Reserved SBX_* keys go last so user-provided env/secrets can't override them
    # (e.g. clobbering SBX_PORT would break the proxy, SBX_TOKEN would break auth).
    job_env: dict[str, Any] = {**(env or {}), "SBX_PORT": str(SANDBOX_SERVER_PORT)}
    job_secrets: dict[str, Any] = {**(secrets or {}), "SBX_TOKEN": sandbox_token}
    job_volumes = list(volumes or [])
    if idle_timeout is not None:
        job_env["SBX_IDLE_TIMEOUT"] = str(_duration_to_secs(idle_timeout))
    if forward_hf_token:
        job_secrets["HF_TOKEN"] = hf_token

    # The digest-named object, not the mutable `sbx-server` alias: a fetch that names the
    # content it wants cannot be answered with different content, and the bootstrap re-checks
    # the digest anyway so a rewritten object is caught rather than executed.
    server_object = f"sbx-server-{SANDBOX_SERVER_SHA256}"
    job_env["SBX_SERVER_URL"] = f"{api.endpoint}/buckets/{constants.SANDBOX_SERVER_BUCKET}/resolve/{server_object}"
    job_env["SBX_SERVER_SHA256"] = SANDBOX_SERVER_SHA256
    # Always mount the server bucket as a transparent fallback for images without wget/curl.
    # It's only read (paying the ~2-3s FUSE cost) when the bootstrap script can't download.
    job_env["SBX_SERVER_MOUNT"] = _SERVER_MOUNT_PATH
    job_volumes.append(
        Volume(type="bucket", source=constants.SANDBOX_SERVER_BUCKET, mount_path=_SERVER_MOUNT_PATH, read_only=True)
    )
    return _bootstrap_command(), job_env, job_secrets, job_volumes


def _bootstrap_command() -> list[str]:
    """The Job command every sandbox host and dedicated sandbox runs."""
    return ["/bin/sh", "-c", _BOOTSTRAP_DOWNLOAD]


def _host_env(api: HfApi, job: JobInfo, *, namespace: str | None) -> dict[str, Any]:
    """Return a host job's env vars (where pool config lives)"""
    env = job.environment if isinstance(job.environment, dict) else {}
    if "SBX_CAPACITY" not in env:
        env = api.inspect_job(job_id=job.id, namespace=namespace).environment or {}
        env = env if isinstance(env, dict) else {}
    return env


def _find_pool_host_job(api: HfApi, pool_id: str, *, namespace: str | None = None, policy: str = ADOPT_OWN) -> JobInfo:
    """Return a running host job belonging to `pool_id`, found via the pool label.

    This one feeds `SandboxPool.connect`'s cold path, which rebuilds the pool's
    whole configuration (image, flavor, density, caps) from the job it returns --
    so an unvalidated match would let a Job in the namespace choose the image
    that the pool's *next* hosts boot. Image and flavor are unknown here (they
    are what we are about to learn), so this checks ownership and shape.
    """
    principal = None
    if policy == ADOPT_OWN:
        try:
            principal = api.whoami(cache=True).get("id")
        except Exception as e:
            logger.warning(f"Could not resolve the current user to validate pool hosts: {e}")
    rejected = []
    for job in api.list_jobs(
        status="RUNNING", labels={MODE_LABEL: MODE_POOL, POOL_LABEL: pool_id}, namespace=namespace
    ):
        reason = _host_rejection(
            job, policy=policy, principal_id=principal, namespace=namespace, image=None, flavor=None
        )
        if reason is None:
            return job
        rejected.append(f"{job.id} ({reason})")
    if rejected:
        raise SandboxError(
            f"Found host job(s) labelled for pool '{pool_id}' but none usable: {'; '.join(rejected)}. "
            "Pass adopt_hosts='namespace' to SandboxPool if you intend to share hosts with other "
            "members of this namespace."
        )
    raise SandboxError(
        f"No running host found for pool '{pool_id}'. The pool has stopped "
        "(all its hosts were killed or idle-timed-out); create a new one."
    )


def _connect_host(api: HfApi, host_job_id: str, *, namespace: str | None = None) -> _SandboxServer:
    """Reattach to a running host job and return its server transport."""
    job = api.inspect_job(job_id=host_job_id, namespace=namespace)
    labels = job.labels or {}
    nonce = labels.get(NONCE_LABEL)
    if nonce is None or labels.get(MODE_LABEL) != MODE_POOL:
        raise SandboxError(f"Job {host_job_id} is not a sandbox host.")
    if job.status.stage != "RUNNING":
        raise SandboxError(f"Sandbox host {host_job_id} is not running (status: {job.status.stage}).")
    # The caller named this job explicitly, so who started it is their call --
    # but the credentials below must still only reach this job's own URL, over
    # HTTPS, on a job that actually runs the sandbox bootstrap.
    reason = _host_rejection(
        job, policy=ADOPT_NAMESPACE, principal_id=None, namespace=namespace, image=None, flavor=None
    )
    if reason is not None:
        raise SandboxError(f"Job {host_job_id} does not look like a usable sandbox host: {reason}.")
    return _SandboxServer.from_job(
        job=job,
        nonce=nonce,
        sandbox_token=_derive_sandbox_token(_effective_token(api), nonce),
        api=api,
        max_connections=SandboxFiles.PARALLEL_MAX_WORKERS + 2,
    )


def _split_sandbox_id(sandbox_id: str, namespace: str | None) -> tuple[str, str | None]:
    """Accept `namespace/sandbox_id` ids (as shown in the Hub UI), like `hf jobs` does."""
    if "/" not in sandbox_id:
        return sandbox_id, namespace
    extracted_namespace, parsed_id = sandbox_id.split("/", 1)
    if not extracted_namespace or not parsed_id or "/" in parsed_id:
        raise SandboxError(f"Sandbox id must be 'sandbox_id' or 'namespace/sandbox_id', got {sandbox_id!r}.")
    if namespace is not None and namespace != extracted_namespace:
        raise SandboxError(
            f"Conflicting namespace: got namespace={namespace!r} but sandbox id implies namespace={extracted_namespace!r}."
        )
    return parsed_id, extracted_namespace


def _find_server_url(job: JobInfo) -> str:
    for url in job.status.expose_urls or []:
        if f"--{SANDBOX_SERVER_PORT}." in url:
            return url
    raise SandboxError(f"Job {job.id} does not expose the sandbox server port {SANDBOX_SERVER_PORT}.")


def _tail_job_logs(api: HfApi, job_id: str, *, namespace: str | None = None, limit: int = 20) -> str:
    try:
        lines = list(api.fetch_job_logs(job_id=job_id, namespace=namespace))[-limit:]
    except Exception:
        return ""
    return " Last logs:\n" + "\n".join(f"  {line}" for line in lines) if lines else ""
<|endoftext|># Copyright 2024-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains utilities to handle the `../.cache/huggingface` folder in local directories.

First discussed in https://github.com/huggingface/huggingface_hub/issues/1738 to store
download metadata when downloading files from the hub to a local directory (without
using the cache).

./.cache/huggingface folder structure:
[4.0K]  data
├── [4.0K]  .cache
│   └── [4.0K]  huggingface
│       └── [4.0K]  download
│           ├── [  16]  file.parquet.metadata
│           ├── [  16]  file.txt.metadata
│           └── [4.0K]  folder
│               └── [  16]  file.parquet.metadata
│
├── [6.5G]  file.parquet
├── [1.5K]  file.txt
└── [4.0K]  folder
    └── [   16]  file.parquet


Download metadata file structure:
```
# file.txt.metadata
11c5a3d5811f50298f278a704980280950aedb10
a16a55fda99d2f2e7b69cce5cf93ff4ad3049930
1712656091.123

# file.parquet.metadata
11c5a3d5811f50298f278a704980280950aedb10
7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421
1712656091.123
}
```
"""

import base64
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from .utils import WeakFileLock
from .utils._paths import as_extended_path


logger = logging.getLogger(__name__)

CACHEDIR_TAG_CONTENT = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by huggingface_hub.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)


@dataclass
class LocalDownloadFilePaths:
    """
    Paths to the files related to a download process in a local dir.

    Returned by [`get_local_download_paths`].

    Attributes:
        file_path (`Path`):
            Path where the file will be saved.
        lock_path (`Path`):
            Path to the lock file used to ensure atomicity when reading/writing metadata.
        metadata_path (`Path`):
            Path to the metadata file.
    """

    file_path: Path
    lock_path: Path
    metadata_path: Path

    def incomplete_path(self, etag: str) -> Path:
        """Return the path where a file will be temporarily downloaded before being moved to `file_path`."""
        path = self.metadata_path.parent / f"{_short_hash(self.metadata_path.name)}.{etag}.incomplete"
        return Path(as_extended_path(path))


@dataclass(frozen=True)
class LocalUploadFilePaths:
    """
    Paths to the files related to an upload process in a local dir.

    Returned by [`get_local_upload_paths`].

    Attributes:
        path_in_repo (`str`):
            Path of the file in the repo.
        file_path (`Path`):
            Path where the file will be saved.
        lock_path (`Path`):
            Path to the lock file used to ensure atomicity when reading/writing metadata.
        metadata_path (`Path`):
            Path to the metadata file.
    """

    path_in_repo: str
    file_path: Path
    lock_path: Path
    metadata_path: Path


@dataclass
class LocalDownloadFileMetadata:
    """
    Metadata about a file in the local directory related to a download process.

    Attributes:
        filename (`str`):
            Path of the file in the repo.
        commit_hash (`str`):
            Commit hash of the file in the repo.
        etag (`str`):
            ETag of the file in the repo. Used to check if the file has changed.
            For LFS files, this is the sha256 of the file. For regular files, it corresponds to the git hash.
        timestamp (`int`):
            Unix timestamp of when the metadata was saved i.e. when the metadata was accurate.
    """

    filename: str
    commit_hash: str
    etag: str
    timestamp: float


@dataclass
class LocalUploadFileMetadata:
    """
    Metadata about a file in the local directory related to an upload process.
    """

    size: int

    # Default values correspond to "we don't know yet"
    timestamp: float | None = None
    should_ignore: bool | None = None
    sha256: str | None = None
    upload_mode: str | None = None
    remote_oid: str | None = None
    is_uploaded: bool = False
    is_committed: bool = False

    def save(self, paths: LocalUploadFilePaths) -> None:
        """Save the metadata to disk."""
        with WeakFileLock(paths.lock_path):
            with paths.metadata_path.open("w") as f:
                new_timestamp = time.time()
                f.write(str(new_timestamp) + "\n")

                f.write(str(self.size))  # never None
                f.write("\n")

                if self.should_ignore is not None:
                    f.write(str(int(self.should_ignore)))
                f.write("\n")

                if self.sha256 is not None:
                    f.write(self.sha256)
                f.write("\n")

                if self.upload_mode is not None:
                    f.write(self.upload_mode)
                f.write("\n")

                if self.remote_oid is not None:
                    f.write(self.remote_oid)
                f.write("\n")

                f.write(str(int(self.is_uploaded)) + "\n")
                f.write(str(int(self.is_committed)) + "\n")

            self.timestamp = new_timestamp


def _validate_relative_filename(filename: str) -> None:
    """Validate that a repo filename is safe to use as a path relative to a local directory.

    A repo filename is a POSIX-style ('/'-separated) path relative to the repo root. When materialized
    on disk (in `local_dir` or in the cache), it is joined onto a base directory. A malicious repository
    could craft a filename that escapes that base directory, resulting in an arbitrary file write:

    - an absolute path (`/etc/...`),
    - a Windows drive path (`C:\\Windows\\...`) or drive-relative path (`D:foo`),
    - a UNC path (`\\\\attacker-host\\share\\...`) — which on Windows additionally makes the client
      authenticate to the attacker's SMB server during path resolution, leaking a NetNTLMv2 hash,
    - a path using `..` to traverse upward.

    Such filenames would otherwise cause `local_dir / filename` to discard `local_dir` (when the right
    side is anchored) or point outside of it (with `..`). We reject them here, before touching the
    filesystem. The check runs on all platforms and interprets the name under both POSIX and Windows
    rules, so a file materialized on Linux cannot escape when later consumed on Windows.
    """
    # Reject parent-directory traversal ('..' as any segment, whether '/'- or '\'-separated).
    if ".." in filename.replace("\\", "/").split("/"):
        raise ValueError(
            f"Invalid filename '{filename}': cannot contain a '..' path segment. "
            "Please ask the repository owner to rename this file."
        )
    # Reject anchored paths (absolute / drive / root-relative / UNC) under either OS's rules.
    # Check each POSIX segment under Windows rules because os.path.join handles them separately.
    pure_paths = [PurePosixPath(filename), *(PureWindowsPath(part) for part in filename.split("/"))]
    for pure_path in pure_paths:
        if pure_path.drive or pure_path.root:
            raise ValueError(
                f"Invalid filename '{filename}': cannot be an absolute, drive-relative or UNC path. "
                "Please ask the repository owner to rename this file."
            )


def get_local_download_paths(local_dir: Path, filename: str) -> LocalDownloadFilePaths:
    """Compute paths to the files related to a download process.

    Folders containing the paths are all guaranteed to exist.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        [`LocalDownloadFilePaths`]: the paths to the files (file_path, lock_path, metadata_path, incomplete_path).
    """
    _validate_relative_filename(filename)
    # filename is the path in the Hub repository (separated by '/')
    # make sure to have a cross-platform transcription
    sanitized_filename = os.path.join(*filename.split("/"))
    file_path = local_dir / sanitized_filename
    metadata_path = _huggingface_dir(local_dir) / "download" / f"{sanitized_filename}.metadata"
    lock_path = metadata_path.with_suffix(".lock")

    # Parent directories are created below => use the stricter Windows limit for directories.
    file_path = Path(as_extended_path(file_path, max_length=247))
    lock_path = Path(as_extended_path(lock_path, max_length=247))
    metadata_path = Path(as_extended_path(metadata_path, max_length=247))

    file_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    return LocalDownloadFilePaths(file_path=file_path, lock_path=lock_path, metadata_path=metadata_path)


def get_local_upload_paths(local_dir: Path, filename: str) -> LocalUploadFilePaths:
    """Compute paths to the files related to an upload process.

    Folders containing the paths are all guaranteed to exist.

    Args:
        local_dir (`Path`):
            Path to the local directory that is uploaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        [`LocalUploadFilePaths`]: the paths to the files (file_path, lock_path, metadata_path).
    """
    _validate_relative_filename(filename)
    # filename is the path in the Hub repository (separated by '/')
    # make sure to have a cross-platform transcription
    sanitized_filename = os.path.join(*filename.split("/"))
    file_path = local_dir / sanitized_filename
    metadata_path = _huggingface_dir(local_dir) / "upload" / f"{sanitized_filename}.metadata"
    lock_path = metadata_path.with_suffix(".lock")

    # Parent directories are created below => use the stricter Windows limit for directories.
    file_path = Path(as_extended_path(file_path, max_length=247))
    lock_path = Path(as_extended_path(lock_path, max_length=247))
    metadata_path = Path(as_extended_path(metadata_path, max_length=247))

    file_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    return LocalUploadFilePaths(
        path_in_repo=filename, file_path=file_path, lock_path=lock_path, metadata_path=metadata_path
    )


def read_download_metadata(local_dir: Path, filename: str) -> LocalDownloadFileMetadata | None:
    """Read metadata about a file in the local directory related to a download process.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        `[LocalDownloadFileMetadata]` or `None`: the metadata if it exists, `None` otherwise.
    """
    paths = get_local_download_paths(local_dir, filename)
    with WeakFileLock(paths.lock_path):
        if paths.metadata_path.exists():
            try:
                with paths.metadata_path.open() as f:
                    commit_hash = f.readline().strip()
                    etag = f.readline().strip()
                    timestamp = float(f.readline().strip())
                    metadata = LocalDownloadFileMetadata(
                        filename=filename,
                        commit_hash=commit_hash,
                        etag=etag,
                        timestamp=timestamp,
                    )
            except Exception as e:
                # remove the metadata file if it is corrupted / not the right format
                logger.warning(
                    f"Invalid metadata file {paths.metadata_path}: {e}. Removing it from disk and continue."
                )
                try:
                    paths.metadata_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove corrupted metadata file {paths.metadata_path}: {e}")
                return None

            try:
                # check if the file exists and hasn't been modified since the metadata was saved
                stat = paths.file_path.stat()
                if (
                    stat.st_mtime - 1 <= metadata.timestamp
                ):  # allow 1s difference as stat.st_mtime might not be precise
                    return metadata
                logger.info(f"Ignored metadata for '{filename}' (outdated). Will re-compute hash.")
            except FileNotFoundError:
                # file does not exist => metadata is outdated
                return None
    return None


def read_upload_metadata(local_dir: Path, filename: str) -> LocalUploadFileMetadata:
    """Read metadata about a file in the local directory related to an upload process.

    TODO: factorize logic with `read_download_metadata`.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
        filename (`str`):
            Path of the file in the repo.

    Return:
        `[LocalUploadFileMetadata]` or `None`: the metadata if it exists, `None` otherwise.
    """
    paths = get_local_upload_paths(local_dir, filename)
    with WeakFileLock(paths.lock_path):
        if paths.metadata_path.exists():
            try:
                with paths.metadata_path.open() as f:
                    timestamp = float(f.readline().strip())

                    size = int(f.readline().strip())  # never None

                    _should_ignore = f.readline().strip()
                    should_ignore = None if _should_ignore == "" else bool(int(_should_ignore))

                    _sha256 = f.readline().strip()
                    sha256 = None if _sha256 == "" else _sha256

                    _upload_mode = f.readline().strip()
                    upload_mode = None if _upload_mode == "" else _upload_mode
                    if upload_mode not in (None, "regular", "lfs"):
                        raise ValueError(f"Invalid upload mode in metadata {paths.path_in_repo}: {upload_mode}")

                    _remote_oid = f.readline().strip()
                    remote_oid = None if _remote_oid == "" else _remote_oid

                    is_uploaded = bool(int(f.readline().strip()))
                    is_committed = bool(int(f.readline().strip()))

                    metadata = LocalUploadFileMetadata(
                        timestamp=timestamp,
                        size=size,
                        should_ignore=should_ignore,
                        sha256=sha256,
                        upload_mode=upload_mode,
                        remote_oid=remote_oid,
                        is_uploaded=is_uploaded,
                        is_committed=is_committed,
                    )
            except Exception as e:
                # remove the metadata file if it is corrupted / not the right format
                logger.warning(
                    f"Invalid metadata file {paths.metadata_path}: {e}. Removing it from disk and continue."
                )
                try:
                    paths.metadata_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove corrupted metadata file {paths.metadata_path}: {e}")

                # corrupted metadata => we don't know anything expect its size
                return LocalUploadFileMetadata(size=paths.file_path.stat().st_size)

            # TODO: can we do better?
            if (
                metadata.timestamp is not None
                and metadata.is_uploaded  # file was uploaded
                and not metadata.is_committed  # but not committed
                and time.time() - metadata.timestamp > 20 * 3600  # and it's been more than 20 hours
            ):  # => we consider it as garbage-collected by S3
                metadata.is_uploaded = False

            # check if the file exists and hasn't been modified since the metadata was saved
            try:
                if metadata.timestamp is not None and paths.file_path.stat().st_mtime <= metadata.timestamp:
                    return metadata
                logger.info(f"Ignored metadata for '{filename}' (outdated). Will re-compute hash.")
            except FileNotFoundError:
                # file does not exist => metadata is outdated
                pass

    # empty metadata => we don't know anything expect its size
    return LocalUploadFileMetadata(size=paths.file_path.stat().st_size)


def write_download_metadata(local_dir: Path, filename: str, commit_hash: str, etag: str) -> None:
    """Write metadata about a file in the local directory related to a download process.

    Args:
        local_dir (`Path`):
            Path to the local directory in which files are downloaded.
    """
    paths = get_local_download_paths(local_dir, filename)
    with WeakFileLock(paths.lock_path):
        with paths.metadata_path.open("w") as f:
            f.write(f"{commit_hash}\n{etag}\n{time.time()}\n")


def _huggingface_dir(local_dir: Path) -> Path:
    """Return the path to the `.cache/huggingface` directory in a local directory."""
    # Wrap in lru_cache to avoid overwriting the .gitignore file if called multiple times
    path = local_dir / ".cache" / "huggingface"
    # Creating the `.cache/huggingface` directory (and the bookkeeping files below) fails for a deep `local_dir`, so
    # the filesystem operations go through the extended-length form. The un-prefixed `path` is still returned so
    # callers keep re-deriving/prefixing as before.
    target = Path(as_extended_path(path, max_length=247))
    target.mkdir(exist_ok=True, parents=True)

    # Create a CACHEDIR.TAG so backup tools can skip this directory.
    _create_cachedir_tag(target)

    # Create a .gitignore file in the .cache/huggingface directory if it doesn't exist
    # Should be thread-safe enough like this.
    gitignore = target / ".gitignore"
    gitignore_lock = target / ".gitignore.lock"
    if not gitignore.exists():
        try:
            with WeakFileLock(gitignore_lock, timeout=0.1):
                gitignore.write_text("*")
        except IndexError:
            pass
        except OSError:  # TimeoutError, FileNotFoundError, PermissionError, etc.
            pass
        try:
            gitignore_lock.unlink()
        except OSError:
            pass
    return path


def _create_cachedir_tag(cache_dir: Path) -> None:
    """Create a CACHEDIR.TAG file in ``cache_dir`` if one does not already exist.

    The tag follows the `Cache Directory Tagging Standard <http://www.brynosaurus.com/cachedir/>`_
    so that backup tools can recognize and skip cache directories.
    """
    tag_path = cache_dir / "CACHEDIR.TAG"
    if not tag_path.exists():
        try:
            tag_path.write_text(CACHEDIR_TAG_CONTENT)
        except OSError:
            pass


def _short_hash(filename: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha1(filename.encode()).digest()).decode()
<|endoftext|>"""Evaluation results utilities for the `.eval_results/*.yaml` format.

See https://huggingface.co/docs/hub/eval-results for more details.
Specifications are available at https://github.com/huggingface/hub-docs/blob/main/eval_results.yaml.
"""

from dataclasses import dataclass
from typing import Any

from .utils import logging


logger = logging.get_logger(__name__)


@dataclass
class EvalResultEntry:
    """
    Evaluation result entry for the `.eval_results/*.yaml` format.

    Represents evaluation scores stored in model repos that automatically appear on
    the model page and the benchmark dataset's leaderboard.

    For the legacy `model-index` format in `README.md`, use [`EvalResult`] instead.

    See https://huggingface.co/docs/hub/eval-results for more details.

    Args:
        dataset_id (`str`):
            Benchmark dataset ID from the Hub. Example: "cais/hle", "Idavidrein/gpqa".
        task_id (`str`):
            Task identifier within the benchmark. Example: "gpqa_diamond".
        value (`Any`):
            The metric value. Example: 20.90.
        dataset_revision (`str`, *optional*):
            Git SHA of the benchmark dataset.
        verify_token (`str`, *optional*):
            A signature that can be used to prove that evaluation is provably auditable and reproducible.
        date (`str`, *optional*):
            When the evaluation was run (ISO-8601 datetime). Defaults to git commit time.
        source_url (`str`, *optional*):
            Link to the evaluation source (e.g., https://huggingface.co/spaces/SaylorTwift/smollm3-mmlu-pro). Required if `source_name`, `source_user`, or `source_org` is provided.
        source_name (`str`, *optional*):
            Display name for the source. Example: "Eval Logs".
        source_user (`str`, *optional*):
            HF user name for attribution. Example: "celinah".
        source_org (`str`, *optional*):
            HF org name for attribution. Example: "cais".
        notes (`str`, *optional*):
            Details about the evaluation setup. Example: "tools", "no-tools", "chain-of-thought".

    Example:
        ```python
        >>> from huggingface_hub import EvalResultEntry
        >>> # Minimal example with required fields only
        >>> result = EvalResultEntry(
        ...     dataset_id="Idavidrein/gpqa",
        ...     task_id="gpqa_diamond",
        ...     value=0.412,
        ... )
        >>> # Full example with all fields
        >>> result = EvalResultEntry(
        ...     dataset_id="cais/hle",
        ...     task_id="default",
        ...     value=20.90,
        ...     dataset_revision="5503434ddd753f426f4b38109466949a1217c2bb",
        ...     verify_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        ...     date="2025-01-15T10:30:00Z",
        ...     source_url="https://huggingface.co/datasets/cais/hle",
        ...     source_name="CAIS HLE",
        ...     source_org="cais",
        ...     notes="no-tools",
        ... )

        ```
    """

    dataset_id: str
    task_id: str
    value: Any
    dataset_revision: str | None = None
    verify_token: str | None = None
    date: str | None = None
    source_url: str | None = None
    source_name: str | None = None
    source_user: str | None = None
    source_org: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if (
            self.source_name is not None or self.source_user is not None or self.source_org is not None
        ) and self.source_url is None:
            raise ValueError(
                "If `source_name`, `source_user`, or `source_org` is provided, `source_url` must also be provided."
            )


def eval_result_entries_to_yaml(entries: list[EvalResultEntry]) -> list[dict[str, Any]]:
    """Convert a list of [`EvalResultEntry`] objects to a YAML-serializable list of dicts.

    This produces the format expected in `.eval_results/*.yaml` files.

    Args:
        entries (`list[EvalResultEntry]`):
            List of evaluation result entries to serialize.

    Returns:
        `list[dict[str, Any]]`: A list of dictionaries ready to be dumped to YAML.

    Example:
        ```python
        >>> from huggingface_hub import EvalResultEntry, eval_result_entries_to_yaml
        >>> entries = [
        ...     EvalResultEntry(dataset_id="cais/hle", task_id="default", value=20.90),
        ...     EvalResultEntry(dataset_id="Idavidrein/gpqa", task_id="gpqa_diamond", value=0.412),
        ... ]
        >>> yaml_data = eval_result_entries_to_yaml(entries)
        >>> yaml_data[0]
        {'dataset': {'id': 'cais/hle', 'task_id': 'default'}, 'value': 20.9}

        ```

        To upload eval results to the Hub:
        ```python
        >>> import yaml
        >>> from huggingface_hub import upload_file, EvalResultEntry, eval_result_entries_to_yaml
        >>> entries = [
        ...     EvalResultEntry(dataset_id="cais/hle", task_id="default", value=20.90),
        ... ]
        >>> yaml_content = yaml.dump(eval_result_entries_to_yaml(entries))
        >>> upload_file(
        ...     path_or_fileobj=yaml_content.encode(),
        ...     path_in_repo=".eval_results/hle.yaml",
        ...     repo_id="your-username/your-model",
        ... )

        ```
    """
    result = []
    for entry in entries:
        # build the dataset object
        dataset: dict[str, Any] = {"id": entry.dataset_id, "task_id": entry.task_id}
        if entry.dataset_revision is not None:
            dataset["revision"] = entry.dataset_revision

        data: dict[str, Any] = {"dataset": dataset, "value": entry.value}
        if entry.verify_token is not None:
            data["verifyToken"] = entry.verify_token
        if entry.date is not None:
            data["date"] = entry.date
        # build the source object
        if entry.source_url is not None:
            source: dict[str, Any] = {"url": entry.source_url}
            if entry.source_name is not None:
                source["name"] = entry.source_name
            if entry.source_user is not None:
                source["user"] = entry.source_user
            if entry.source_org is not None:
                source["org"] = entry.source_org
            data["source"] = source
        if entry.notes is not None:
            data["notes"] = entry.notes

        result.append(data)
    return result


def parse_eval_result_entries(data: list[dict[str, Any]]) -> list[EvalResultEntry]:
    """Parse a list of dicts into [`EvalResultEntry`] objects.

    This parses the `.eval_results/*.yaml` format. For the legacy `model-index` format,
    use [`model_index_to_eval_results`] instead.

    Args:
        data (`list[dict[str, Any]]`):
            A list of dictionaries (e.g., parsed from YAML or API response).

    Returns:
        `list[EvalResultEntry]`: A list of evaluation result entry objects.

    Example:
        ```python
        >>> from huggingface_hub import parse_eval_result_entries
        >>> data = [
        ...     {"dataset": {"id": "cais/hle", "task_id": "default"}, "value": 20.90},
        ...     {"dataset": {"id": "Idavidrein/gpqa", "task_id": "gpqa_diamond"}, "value": 0.412},
        ... ]
        >>> entries = parse_eval_result_entries(data)
        >>> entries[0].dataset_id
        'cais/hle'
        >>> entries[0].value
        20.9

        ```
    """
    entries = []
    for item in data:
        if "error" in item:  # file failed server-side validation
            logger.warning(f"Skipping invalid eval results file '{item.get('filename')}': {item['error']}")
            continue
        entry_data = item.get("data", item)
        dataset = entry_data.get("dataset", {})
        source = entry_data.get("source", {})
        entry = EvalResultEntry(
            dataset_id=dataset["id"],
            value=entry_data["value"],
            task_id=dataset["task_id"],
            dataset_revision=dataset.get("revision"),
            verify_token=entry_data.get("verifyToken"),
            date=entry_data.get("date"),
            source_url=source.get("url") if source else None,
            source_name=source.get("name") if source else None,
            source_user=source.get("user") if source else None,
            source_org=source.get("org") if source else None,
            notes=entry_data.get("notes"),
        )
        entries.append(entry)
    return entries
<|endoftext|>import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml

from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import upload_file
from huggingface_hub.repocard_data import (
    CardData,
    DatasetCardData,
    EvalResult,
    ModelCardData,
    SpaceCardData,
    eval_results_to_model_index,
    model_index_to_eval_results,
)
from huggingface_hub.utils import HfHubHTTPError, get_session, hf_raise_for_status, is_jinja_available, yaml_dump

from . import constants
from .errors import EntryNotFoundError
from .utils import SoftTemporaryDirectory, logging, validate_hf_hub_args


logger = logging.get_logger(__name__)


TEMPLATE_MODELCARD_PATH = Path(__file__).parent / "templates" / "modelcard_template.md"
TEMPLATE_DATASETCARD_PATH = Path(__file__).parent / "templates" / "datasetcard_template.md"

# exact same regex as in the Hub server. Please keep in sync.
# See https://github.com/huggingface/moon-landing/blob/main/server/lib/ViewMarkdown.ts#L18
REGEX_YAML_BLOCK = re.compile(r"^(\s*---(?:\r\n|\r|\n))([\S\s]*?)((?:\r\n|\r|\n)---[ \t]*(\r\n|\n|$))")


class RepoCard:
    card_data_class = CardData
    default_template_path = TEMPLATE_MODELCARD_PATH
    repo_type = "model"

    def __init__(self, content: str, ignore_metadata_errors: bool = False):
        """Initialize a RepoCard from string content. The content should be a
        Markdown file with a YAML block at the beginning and a Markdown body.

        Args:
            content (`str`): The content of the Markdown file.

        Example:
            ```python
            >>> from huggingface_hub.repocard import RepoCard
            >>> text = '''
            ... ---
            ... language: en
            ... license: mit
            ... ---
            ...
            ... # My repo
            ... '''
            >>> card = RepoCard(text)
            >>> card.data.to_dict()
            {'language': 'en', 'license': 'mit'}
            >>> card.text
            '\\n# My repo\\n'

            ```
        > [!TIP]
        > Raises the following error:
        >
        >     - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
        >       when the content of the repo card metadata is not a dictionary.
        """

        # Set the content of the RepoCard, as well as underlying .data and .text attributes.
        # See the `content` property setter for more details.
        self.ignore_metadata_errors = ignore_metadata_errors
        self.content = content

    @property
    def content(self):
        """The content of the RepoCard, including the YAML block and the Markdown body."""
        line_break = _detect_line_ending(self._content) or "\n"
        return f"---{line_break}{self.data.to_yaml(line_break=line_break, original_order=self._original_order)}{line_break}---{line_break}{self.text}"

    @content.setter
    def content(self, content: str):
        """Set the content of the RepoCard."""
        self._content = content

        match = REGEX_YAML_BLOCK.search(content)
        if match:
            # Metadata found in the YAML block
            yaml_block = match.group(2)
            self.text = content[match.end() :]
            data_dict = yaml.safe_load(yaml_block)

            if data_dict is None:
                data_dict = {}

            # The YAML block's data should be a dictionary
            if not isinstance(data_dict, dict):
                raise ValueError("repo card metadata block should be a dict")
        else:
            # Model card without metadata... create empty metadata
            logger.warning("Repo card metadata block was not found. Setting CardData to empty.")
            data_dict = {}
            self.text = content

        self.data = self.card_data_class(**data_dict, ignore_metadata_errors=self.ignore_metadata_errors)
        self._original_order = list(data_dict.keys())

    def __str__(self):
        return self.content

    def save(self, filepath: Path | str):
        r"""Save a RepoCard to a file.

        Args:
            filepath (`Union[Path, str]`): Filepath to the markdown file to save.

        Example:
            ```python
            >>> from huggingface_hub.repocard import RepoCard
            >>> card = RepoCard("---\nlanguage: en\n---\n# This is a test repo card")
            >>> card.save("/tmp/test.md")

            ```
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # Preserve newlines as in the existing file.
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            f.write(str(self))

    @classmethod
    def load(
        cls,
        repo_id_or_path: str | Path,
        repo_type: str | None = None,
        token: str | None = None,
        ignore_metadata_errors: bool = False,
    ):
        """Initialize a RepoCard from a Hugging Face Hub repo's README.md or a local filepath.

        Args:
            repo_id_or_path (`Union[str, Path]`):
                The repo ID associated with a Hugging Face Hub repo or a local filepath.
            repo_type (`str`, *optional*):
                The type of Hugging Face repo to push to. Defaults to None, which will use "model". Other options
                are "dataset" and "space". Not used when loading from a local filepath. If this is called from a child
                class, the default value will be the child class's `repo_type`.
            token (`str`, *optional*):
                Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
            ignore_metadata_errors (`str`):
                If True, errors while parsing the metadata section will be ignored. Some information might be lost during
                the process. Use it at your own risk.

        Returns:
            [`huggingface_hub.repocard.RepoCard`]: The RepoCard (or subclass) initialized from the repo's
                README.md file or filepath.

        Example:
            ```python
            >>> from huggingface_hub.repocard import RepoCard
            >>> card = RepoCard.load("nateraw/food")
            >>> assert card.data.tags == ["generated_from_trainer", "image-classification", "pytorch"]

            ```
        """

        if Path(repo_id_or_path).is_file():
            card_path = Path(repo_id_or_path)
        elif isinstance(repo_id_or_path, str):
            card_path = Path(
                hf_hub_download(
                    repo_id_or_path,
                    constants.REPOCARD_NAME,
                    repo_type=repo_type or cls.repo_type,
                    token=token,
                )
            )
        else:
            raise ValueError(f"Cannot load RepoCard: path not found on disk ({repo_id_or_path}).")

        # Preserve newlines in the existing file.
        with card_path.open(mode="r", newline="", encoding="utf-8") as f:
            return cls(f.read(), ignore_metadata_errors=ignore_metadata_errors)

    def validate(self, repo_type: str | None = None):
        """Validates card against Hugging Face Hub's card validation logic.
        Using this function requires access to the internet, so it is only called
        internally by [`huggingface_hub.repocard.RepoCard.push_to_hub`].

        Args:
            repo_type (`str`, *optional*, defaults to "model"):
                The type of Hugging Face repo to push to. Options are "model", "dataset", and "space".
                If this function is called from a child class, the default will be the child class's `repo_type`.

        > [!TIP]
        > Raises the following errors:
        >
        >     - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
        >       if the card fails validation checks.
        >     - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
        >       if the request to the Hub API fails for any other reason.
        """

        # If repo type is provided, otherwise, use the repo type of the card.
        repo_type = repo_type or self.repo_type

        body = {
            "repoType": repo_type,
            "content": str(self),
        }
        headers = {"Accept": "text/plain"}

        try:
            response = get_session().post(f"{constants.ENDPOINT}/api/validate-yaml", json=body, headers=headers)
            hf_raise_for_status(response)
        except HfHubHTTPError as exc:
            if response.status_code == 400:
                raise ValueError(response.text)
            else:
                raise exc

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        repo_type: str | None = None,
        commit_message: str | None = None,
        commit_description: str | None = None,
        revision: str | None = None,
        create_pr: bool | None = None,
        parent_commit: str | None = None,
    ):
        """Push a RepoCard to a Hugging Face Hub repo.

        Args:
            repo_id (`str`):
                The repo ID of the Hugging Face Hub repo to push to. Example: "nateraw/food".
            token (`str`, *optional*):
                Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to
                the stored token.
            repo_type (`str`, *optional*, defaults to "model"):
                The type of Hugging Face repo to push to. Options are "model", "dataset", and "space". If this
                function is called by a child class, it will default to the child class's `repo_type`.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit.
            commit_description (`str`, *optional*):
                The description of the generated commit.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            create_pr (`bool`, *optional*):
                Whether or not to create a Pull Request with this commit. Defaults to `False`.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed too concurrently.
        Returns:
            `str`: URL of the commit which updated the card metadata.
        """

        # If repo type is provided, otherwise, use the repo type of the card.
        repo_type = repo_type or self.repo_type

        # Validate card before pushing to hub
        self.validate(repo_type=repo_type)

        with SoftTemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / constants.REPOCARD_NAME
            tmp_path.write_text(str(self), encoding="utf-8")
            url = upload_file(
                path_or_fileobj=str(tmp_path),
                path_in_repo=constants.REPOCARD_NAME,
                repo_id=repo_id,
                token=token,
                repo_type=repo_type,
                commit_message=commit_message,
                commit_description=commit_description,
                create_pr=create_pr,
                revision=revision,
                parent_commit=parent_commit,
            )
        return url

    @classmethod
    def from_template(
        cls,
        card_data: CardData,
        template_path: str | None = None,
        template_str: str | None = None,
        **template_kwargs,
    ):
        """Initialize a RepoCard from a template. By default, it uses the default template.

        Templates are Jinja2 templates that can be customized by passing keyword arguments.

        Args:
            card_data (`huggingface_hub.CardData`):
                A huggingface_hub.CardData instance containing the metadata you want to include in the YAML
                header of the repo card on the Hugging Face Hub.
            template_path (`str`, *optional*):
                A path to a markdown file with optional Jinja template variables that can be filled
                in with `template_kwargs`. Defaults to the default template.
            template_str (`str`, *optional*):
                A raw Jinja template string with optional variables. Used when neither `template_path`
                nor the default template is appropriate. Ignored if `template_path` is also provided.

        Returns:
            [`huggingface_hub.repocard.RepoCard`]: A RepoCard instance with the specified card data and content from the
            template.
        """
        if is_jinja_available():
            import jinja2
        else:
            raise ImportError(
                "Using RepoCard.from_template requires Jinja2 to be installed. Please"
                " install it with `pip install Jinja2`."
            )

        kwargs = card_data.to_dict().copy()
        kwargs.update(template_kwargs)  # Template_kwargs have priority

        if template_path is not None:
            template_str = Path(template_path).read_text()
        if template_str is None:
            template_str = Path(cls.default_template_path).read_text()
        template = jinja2.Template(template_str)
        content = template.render(card_data=card_data.to_yaml(), **kwargs)
        return cls(content)


class ModelCard(RepoCard):
    card_data_class = ModelCardData  # type: ignore[assignment]
    default_template_path = TEMPLATE_MODELCARD_PATH
    repo_type = "model"

    @classmethod
    def from_template(  # type: ignore # violates Liskov property but easier to use
        cls,
        card_data: ModelCardData,
        template_path: str | None = None,
        template_str: str | None = None,
        **template_kwargs,
    ):
        """Initialize a ModelCard from a template. By default, it uses the default template, which can be found here:
        https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md

        Templates are Jinja2 templates that can be customized by passing keyword arguments.

        Args:
            card_data (`huggingface_hub.ModelCardData`):
                A huggingface_hub.ModelCardData instance containing the metadata you want to include in the YAML
                header of the model card on the Hugging Face Hub.
            template_path (`str`, *optional*):
                A path to a markdown file with optional Jinja template variables that can be filled
                in with `template_kwargs`. Defaults to the default template.
            template_str (`str`, *optional*):
                A raw Jinja template string with optional variables. Used when neither `template_path`
                nor the default template is appropriate. Ignored if `template_path` is also provided.

        Returns:
            [`huggingface_hub.ModelCard`]: A ModelCard instance with the specified card data and content from the
            template.

        Example:
            ```python
            >>> from huggingface_hub import ModelCard, ModelCardData, EvalResult

            >>> # Using the Default Template
            >>> card_data = ModelCardData(
            ...     language='en',
            ...     license='mit',
            ...     library_name='timm',
            ...     tags=['image-classification', 'resnet'],
            ...     datasets=['beans'],
            ...     metrics=['accuracy'],
            ... )
            >>> card = ModelCard.from_template(
            ...     card_data,
            ...     model_description='This model does x + y...'
            ... )

            >>> # Including Evaluation Results
            >>> card_data = ModelCardData(
            ...     language='en',
            ...     tags=['image-classification', 'resnet'],
            ...     eval_results=[
            ...         EvalResult(
            ...             task_type='image-classification',
            ...             dataset_type='beans',
            ...             dataset_name='Beans',
            ...             metric_type='accuracy',
            ...             metric_value=0.9,
            ...         ),
            ...     ],
            ...     model_name='my-cool-model',
            ... )
            >>> card = ModelCard.from_template(card_data)

            >>> # Using a Custom Template
            >>> card_data = ModelCardData(
            ...     language='en',
            ...     tags=['image-classification', 'resnet']
            ... )
            >>> card = ModelCard.from_template(
            ...     card_data=card_data,
            ...     template_path='./src/huggingface_hub/templates/modelcard_template.md',
            ...     custom_template_var='custom value',  # will be replaced in template if it exists
            ... )

            ```
        """
        return super().from_template(card_data, template_path, template_str, **template_kwargs)


class DatasetCard(RepoCard):
    card_data_class = DatasetCardData  # type: ignore[assignment]
    default_template_path = TEMPLATE_DATASETCARD_PATH
    repo_type = "dataset"

    @classmethod
    def from_template(  # type: ignore # violates Liskov property but easier to use
        cls,
        card_data: DatasetCardData,
        template_path: str | None = None,
        template_str: str | None = None,
        **template_kwargs,
    ):
        """Initialize a DatasetCard from a template. By default, it uses the default template, which can be found here:
        https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md

        Templates are Jinja2 templates that can be customized by passing keyword arguments.

        Args:
            card_data (`huggingface_hub.DatasetCardData`):
                A huggingface_hub.DatasetCardData instance containing the metadata you want to include in the YAML
                header of the dataset card on the Hugging Face Hub.
            template_path (`str`, *optional*):
                A path to a markdown file with optional Jinja template variables that can be filled
                in with `template_kwargs`. Defaults to the default template.
            template_str (`str`, *optional*):
                A raw Jinja template string with optional variables. Used when neither `template_path`
                nor the default template is appropriate. Ignored if `template_path` is also provided.

        Returns:
            [`huggingface_hub.DatasetCard`]: A DatasetCard instance with the specified card data and content from the
            template.

        Example:
            ```python
            >>> from huggingface_hub import DatasetCard, DatasetCardData

            >>> # Using the Default Template
            >>> card_data = DatasetCardData(
            ...     language='en',
            ...     license='mit',
            ...     annotations_creators='crowdsourced',
            ...     task_categories=['text-classification'],
            ...     task_ids=['sentiment-classification', 'text-scoring'],
            ...     multilinguality='monolingual',
            ...     pretty_name='My Text Classification Dataset',
            ... )
            >>> card = DatasetCard.from_template(
            ...     card_data,
            ...     pretty_name=card_data.pretty_name,
            ... )

            >>> # Using a Custom Template
            >>> card_data = DatasetCardData(
            ...     language='en',
            ...     license='mit',
            ... )
            >>> card = DatasetCard.from_template(
            ...     card_data=card_data,
            ...     template_path='./src/huggingface_hub/templates/datasetcard_template.md',
            ...     custom_template_var='custom value',  # will be replaced in template if it exists
            ... )

            ```
        """
        return super().from_template(card_data, template_path, template_str, **template_kwargs)


class SpaceCard(RepoCard):
    card_data_class = SpaceCardData  # type: ignore[assignment]
    default_template_path = TEMPLATE_MODELCARD_PATH
    repo_type = "space"


def _detect_line_ending(content: str) -> Literal["\r", "\n", "\r\n", None]:  # noqa: F722
    """Detect the line ending of a string. Used by RepoCard to avoid making huge diff on newlines.

    Uses same implementation as in Hub server, keep it in sync.

    Returns:
        str: The detected line ending of the string.
    """
    cr = content.count("\r")
    lf = content.count("\n")
    crlf = content.count("\r\n")
    if cr + lf == 0:
        return None
    if crlf == cr and crlf == lf:
        return "\r\n"
    if cr > lf:
        return "\r"
    else:
        return "\n"


def metadata_load(local_path: str | Path) -> dict | None:
    content = Path(local_path).read_text()
    match = REGEX_YAML_BLOCK.search(content)
    if match:
        yaml_block = match.group(2)
        data = yaml.safe_load(yaml_block)
        if data is None or isinstance(data, dict):
            return data
        raise ValueError("repo card metadata block should be a dict")
    else:
        return None


def metadata_save(local_path: str | Path, data: dict) -> None:
    """
    Save the metadata dict in the upper YAML part Trying to preserve newlines as
    in the existing file. Docs about open() with newline="" parameter:
    https://docs.python.org/3/library/functions.html?highlight=open#open Does
    not work with "^M" linebreaks, which are replaced by \n
    """
    line_break = "\n"
    content = ""
    # try to detect existing newline character
    if os.path.exists(local_path):
        with open(local_path, newline="", encoding="utf8") as readme:
            content = readme.read()
            if isinstance(readme.newlines, tuple):
                line_break = readme.newlines[0]
            elif isinstance(readme.newlines, str):
                line_break = readme.newlines

    # creates a new file if it not
    with open(local_path, "w", newline="", encoding="utf8") as readme:
        data_yaml = yaml_dump(data, sort_keys=False, line_break=line_break)
        # sort_keys: keep dict order
        match = REGEX_YAML_BLOCK.search(content)
        if match:
            output = content[: match.start()] + f"---{line_break}{data_yaml}---{line_break}" + content[match.end() :]
        else:
            output = f"---{line_break}{data_yaml}---{line_break}{content}"

        readme.write(output)
        readme.close()


def metadata_eval_result(
    *,
    model_pretty_name: str,
    task_pretty_name: str,
    task_id: str,
    metrics_pretty_name: str,
    metrics_id: str,
    metrics_value: Any,
    dataset_pretty_name: str,
    dataset_id: str,
    metrics_config: str | None = None,
    metrics_verified: bool = False,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    dataset_revision: str | None = None,
    metrics_verification_token: str | None = None,
) -> dict:
    """
    Creates a metadata dict with the result from a model evaluated on a dataset.

    Args:
        model_pretty_name (`str`):
            The name of the model in natural language.
        task_pretty_name (`str`):
            The name of a task in natural language.
        task_id (`str`):
            Example: automatic-speech-recognition. A task id.
        metrics_pretty_name (`str`):
            A name for the metric in natural language. Example: Test WER.
        metrics_id (`str`):
            Example: wer. A metric id from https://hf.co/metrics.
        metrics_value (`Any`):
            The value from the metric. Example: 20.0 or "20.0 ± 1.2".
        dataset_pretty_name (`str`):
            The name of the dataset in natural language.
        dataset_id (`str`):
            Example: common_voice. A dataset id from https://hf.co/datasets.
        metrics_config (`str`, *optional*):
            The name of the metric configuration used in `load_metric()`.
            Example: bleurt-large-512 in `load_metric("bleurt", "bleurt-large-512")`.
        metrics_verified (`bool`, *optional*, defaults to `False`):
            Indicates whether the metrics originate from Hugging Face's [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator) or not. Automatically computed by Hugging Face, do not set.
        dataset_config (`str`, *optional*):
            Example: fr. The name of the dataset configuration used in `load_dataset()`.
        dataset_split (`str`, *optional*):
            Example: test. The name of the dataset split used in `load_dataset()`.
        dataset_revision (`str`, *optional*):
            Example: 5503434ddd753f426f4b38109466949a1217c2bb. The name of the dataset revision
            used in `load_dataset()`.
        metrics_verification_token (`bool`, *optional*):
            A JSON Web Token that is used to verify whether the metrics originate from Hugging Face's [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator) or not.

    Returns:
        `dict`: a metadata dict with the result from a model evaluated on a dataset.

    Example:
        ```python
        >>> from huggingface_hub import metadata_eval_result
        >>> results = metadata_eval_result(
        ...         model_pretty_name="RoBERTa fine-tuned on ReactionGIF",
        ...         task_pretty_name="Text Classification",
        ...         task_id="text-classification",
        ...         metrics_pretty_name="Accuracy",
        ...         metrics_id="accuracy",
        ...         metrics_value=0.2662102282047272,
        ...         dataset_pretty_name="ReactionJPEG",
        ...         dataset_id="julien-c/reactionjpeg",
        ...         dataset_config="default",
        ...         dataset_split="test",
        ... )
        >>> results == {
        ...     'model-index': [
        ...         {
        ...             'name': 'RoBERTa fine-tuned on ReactionGIF',
        ...             'results': [
        ...                 {
        ...                     'task': {
        ...                         'type': 'text-classification',
        ...                         'name': 'Text Classification'
        ...                     },
        ...                     'dataset': {
        ...                         'name': 'ReactionJPEG',
        ...                         'type': 'julien-c/reactionjpeg',
        ...                         'config': 'default',
        ...                         'split': 'test'
        ...                     },
        ...                     'metrics': [
        ...                         {
        ...                             'type': 'accuracy',
        ...                             'value': 0.2662102282047272,
        ...                             'name': 'Accuracy',
        ...                             'verified': False
        ...                         }
        ...                     ]
        ...                 }
        ...             ]
        ...         }
        ...     ]
        ... }
        True

        ```
    """

    return {
        "model-index": eval_results_to_model_index(
            model_name=model_pretty_name,
            eval_results=[
                EvalResult(
                    task_name=task_pretty_name,
                    task_type=task_id,
                    metric_name=metrics_pretty_name,
                    metric_type=metrics_id,
                    metric_value=metrics_value,
                    dataset_name=dataset_pretty_name,
                    dataset_type=dataset_id,
                    metric_config=metrics_config,
                    verified=metrics_verified,
                    verify_token=metrics_verification_token,
                    dataset_config=dataset_config,
                    dataset_split=dataset_split,
                    dataset_revision=dataset_revision,
                )
            ],
        )
    }


@validate_hf_hub_args
def metadata_update(
    repo_id: str,
    metadata: dict,
    *,
    repo_type: str | None = None,
    overwrite: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
    commit_description: str | None = None,
    revision: str | None = None,
    create_pr: bool = False,
    parent_commit: str | None = None,
) -> str:
    """
    Updates the metadata in the README.md of a repository on the Hugging Face Hub.
    If the README.md file doesn't exist yet, a new one is created with metadata and
    the default ModelCard or DatasetCard template. For `space` repo, an error is thrown
    as a Space cannot exist without a `README.md` file.

    Args:
        repo_id (`str`):
            The name of the repository.
        metadata (`dict`):
            A dictionary containing the metadata to be updated.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if updating to a dataset or space,
            `None` or `"model"` if updating to a model. Default is `None`.
        overwrite (`bool`, *optional*, defaults to `False`):
            If set to `True` an existing field can be overwritten, otherwise
            attempting to overwrite an existing field will cause an error.
        token (`str`, *optional*):
            The Hugging Face authentication token.
        commit_message (`str`, *optional*):
            The summary / title / first line of the generated commit. Defaults to
            `f"Update metadata with huggingface_hub"`
        commit_description (`str`, *optional*):
            The description of the generated commit
        revision (`str`, *optional*):
            The git revision to commit from. Defaults to the head of the
            `"main"` branch.
        create_pr (`boolean`, *optional*):
            Whether or not to create a Pull Request from `revision` with that commit.
            Defaults to `False`.
        parent_commit (`str`, *optional*):
            The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
            If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
            If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
            Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
            especially useful if the repo is updated / committed too concurrently.
    Returns:
        `str`: URL of the commit which updated the card metadata.

    Example:
        ```python
        >>> from huggingface_hub import metadata_update
        >>> metadata = {'model-index': [{'name': 'RoBERTa fine-tuned on ReactionGIF',
        ...             'results': [{'dataset': {'name': 'ReactionGIF',
        ...                                      'type': 'julien-c/reactiongif'},
        ...                           'metrics': [{'name': 'Recall',
        ...                                        'type': 'recall',
        ...                                        'value': 0.7762102282047272}],
        ...                          'task': {'name': 'Text Classification',
        ...                                   'type': 'text-classification'}}]}]}
        >>> url = metadata_update("hf-internal-testing/reactiongif-roberta-card", metadata)

        ```
    """
    commit_message = commit_message if commit_message is not None else "Update metadata with huggingface_hub"

    # Card class given repo_type
    card_class: type[RepoCard]
    if repo_type is None or repo_type == "model":
        card_class = ModelCard
    elif repo_type == "dataset":
        card_class = DatasetCard
    elif repo_type == "space":
        card_class = RepoCard
    else:
        raise ValueError(f"Unknown repo_type: {repo_type}")

    # Either load repo_card from the Hub or create an empty one.
    # NOTE: Will not create the repo if it doesn't exist.
    try:
        card = card_class.load(repo_id, token=token, repo_type=repo_type)
    except EntryNotFoundError:
        if repo_type == "space":
            raise ValueError("Cannot update metadata on a Space that doesn't contain a `README.md` file.")

        # Initialize a ModelCard or DatasetCard from default template and no data.
        # Cast to the concrete expected card type to satisfy type checkers.
        card = card_class.from_template(CardData())  # type: ignore

    for key, value in metadata.items():
        if key == "model-index":
            # if the new metadata doesn't include a name, either use existing one or repo name
            if "name" not in value[0]:
                value[0]["name"] = getattr(card, "model_name", repo_id)
            model_name, new_results = model_index_to_eval_results(value)
            if card.data.eval_results is None:
                card.data.eval_results = new_results
                card.data.model_name = model_name
            else:
                existing_results = card.data.eval_results

                # Iterate over new results
                #   Iterate over existing results
                #       If both results describe the same metric but value is different:
                #           If overwrite=True: overwrite the metric value
                #           Else: raise ValueError
                #       Else: append new result to existing ones.
                for new_result in new_results:
                    result_found = False
                    for existing_result in existing_results:
                        if new_result.is_equal_except_value(existing_result):
                            if new_result != existing_result and not overwrite:
                                raise ValueError(
                                    "You passed a new value for the existing metric"
                                    f" 'name: {new_result.metric_name}, type: "
                                    f"{new_result.metric_type}'. Set `overwrite=True`"
                                    " to overwrite existing metrics."
                                )
                            result_found = True
                            existing_result.metric_value = new_result.metric_value
                            if existing_result.verified is True:
                                existing_result.verify_token = new_result.verify_token
                    if not result_found:
                        card.data.eval_results.append(new_result)
        else:
            # Any metadata that is not a result metric
            if card.data.get(key) is not None and not overwrite and card.data.get(key) != value:
                raise ValueError(
                    f"You passed a new value for the existing meta data field '{key}'."
                    " Set `overwrite=True` to overwrite existing metadata."
                )
            else:
                card.data[key] = value

    return card.push_to_hub(
        repo_id,
        token=token,
        repo_type=repo_type,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
        revision=revision,
        parent_commit=parent_commit,
    )
<|endoftext|># Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains utilities to flag a feature as "experimental" in Huggingface Hub."""

import warnings
from collections.abc import Callable
from functools import wraps

from .. import constants


def experimental(fn: Callable) -> Callable:
    """Decorator to flag a feature as experimental.

    An experimental feature triggers a warning when used as it might be subject to breaking changes without prior notice
    in the future.

    Warnings can be disabled by setting `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` as environment variable.

    Args:
        fn (`Callable`):
            The function to flag as experimental.

    Returns:
        `Callable`: The decorated function.

    Example:

    ```python
    >>> from huggingface_hub.utils import experimental

    >>> @experimental
    ... def my_function():
    ...     print("Hello world!")

    >>> my_function()
    UserWarning: 'my_function' is experimental and might be subject to breaking changes in the future without prior
    notice. You can disable this warning by setting `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` as environment variable.
    Hello world!
    ```
    """
    # For classes, put the "experimental" around the "__new__" method => __new__ will be removed in warning message
    name = fn.__qualname__[: -len(".__new__")] if fn.__qualname__.endswith(".__new__") else fn.__qualname__

    @wraps(fn)
    def _inner_fn(*args, **kwargs):
        if not constants.HF_HUB_DISABLE_EXPERIMENTAL_WARNING:
            warnings.warn(
                f"'{name}' is experimental and might be subject to breaking changes in the future without prior notice."
                " You can disable this warning by setting `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` as environment"
                " variable.",
                UserWarning,
            )
        return fn(*args, **kwargs)

    return _inner_fn
<|endoftext|># Copyright 2025-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parsing helpers shared across modules."""

import re
import time


RE_NUMBER_WITH_UNIT = re.compile(r"(\d+)([a-z]+)", re.IGNORECASE)

BYTE_UNITS: dict[str, int] = {
    "k": 1_000,
    "kb": 1_000,
    "m": 1_000_000,
    "mb": 1_000_000,
    "g": 1_000_000_000,
    "gb": 1_000_000_000,
    "t": 1_000_000_000_000,
    "tb": 1_000_000_000_000,
    "p": 1_000_000_000_000_000,
    "pb": 1_000_000_000_000_000,
}

TIME_UNITS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
    "mo": 30 * 24 * 60 * 60,
    "y": 365 * 24 * 60 * 60,
}


def parse_size(value: str) -> int:
    """Parse a size expressed as a string with digits and unit (like `"10MB"`) to an integer (in bytes)."""
    return _parse_with_unit(value, BYTE_UNITS)


def parse_duration(value: str) -> int:
    """Parse a duration expressed as a string with digits and unit (like `"10s"`) to an integer (in seconds)."""
    return _parse_with_unit(value, TIME_UNITS)


def _parse_with_unit(value: str, units: dict[str, int]) -> int:
    """Parse a numeric value with optional unit."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("Value cannot be empty.")
    try:
        return int(value)
    except ValueError:
        pass

    match = RE_NUMBER_WITH_UNIT.fullmatch(stripped)
    if not match:
        raise ValueError(f"Invalid value '{value}'. Must match pattern '\\d+[a-z]+' or be a plain number.")

    number = int(match.group(1))
    unit = match.group(2).lower()

    if unit not in units:
        raise ValueError(f"Unknown unit '{unit}'. Must be one of {list(units.keys())}.")

    return number * units[unit]


def format_duration(secs: int | None) -> str:
    """Format a duration in seconds as a short human-readable string (e.g. `"1m 32s"`, `"2h 15m"`, `"45s"`).

    Returns `"--"` when `secs` is `None` so it can be used directly as a CLI table cell.
    """
    if secs is None:
        return "--"
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs // 3600}h {(secs % 3600) // 60}m"


def format_timesince(ts: float) -> str:
    """Format timestamp in seconds into a human-readable string, relative to now.

    Vaguely inspired by Django's `timesince` formatter.
    """
    _TIMESINCE_CHUNKS = (
        # Label, divider, max value shown before rolling over to the next unit (e.g. 59 -> "1 minute ago" at 60s)
        ("second", 1, 59),
        ("minute", 60, 59),
        ("hour", 60 * 60, 23),
        ("day", 60 * 60 * 24, 6),
        ("week", 60 * 60 * 24 * 7, 6),
        ("month", 60 * 60 * 24 * 30, 11),
        ("year", 60 * 60 * 24 * 365, None),
    )
    delta = time.time() - ts
    if delta < 20:
        return "a few seconds ago"
    for label, divider, max_value in _TIMESINCE_CHUNKS:  # noqa: B007
        value = round(delta / divider)
        if max_value is not None and value <= max_value:
            break
    return f"{value} {label}{'s' if value > 1 else ''} ago"
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check presence of installed packages at runtime."""

import importlib.metadata
import importlib.util
import os
import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

from .. import __version__, constants


_PY_VERSION: str = sys.version.split()[0].rstrip("+")

_package_versions: dict[str, str] = {}

_CANDIDATES = {
    "aiohttp": {"aiohttp"},
    "fastai": {"fastai"},
    "fastapi": {"fastapi"},
    "fastcore": {"fastcore"},
    "gradio": {"gradio"},
    "graphviz": {"graphviz"},
    "hf_xet": {"hf_xet"},
    "jinja": {"Jinja2"},
    "httpx2": {"httpx2"},
    "keras": {"keras"},
    "numpy": {"numpy"},
    "pillow": {"Pillow"},
    "pydantic": {"pydantic"},
    "pydot": {"pydot"},
    "safetensors": {"safetensors"},
    "tensorboard": {"tensorboardX"},
    "tensorflow": (
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-gpu",
        "tf-nightly",
        "tf-nightly-cpu",
        "tf-nightly-gpu",
        "intel-tensorflow",
        "intel-tensorflow-avx512",
        "tensorflow-rocm",
        "tensorflow-macos",
    ),
    "torch": {"torch"},
}


def _get_version(package_name: str) -> str:
    if package_name not in _package_versions:
        # Write once after the lookup: concurrent callers would read a placeholder as "not installed".
        version = "N/A"
        for distribution_name in _CANDIDATES.get(package_name, (package_name,)):
            try:
                version = importlib.metadata.version(distribution_name)
                break
            except importlib.metadata.PackageNotFoundError:
                pass
        _package_versions[package_name] = version
    return _package_versions[package_name]


def is_package_available(package_name: str) -> bool:
    return _get_version(package_name) != "N/A"


# Python
def get_python_version() -> str:
    return _PY_VERSION


# Huggingface Hub
def get_hf_hub_version() -> str:
    return __version__


# aiohttp
def is_aiohttp_available() -> bool:
    return is_package_available("aiohttp")


def get_aiohttp_version() -> str:
    return _get_version("aiohttp")


# FastAI
def is_fastai_available() -> bool:
    return is_package_available("fastai")


def get_fastai_version() -> str:
    return _get_version("fastai")


# FastAPI
def is_fastapi_available() -> bool:
    return is_package_available("fastapi")


def get_fastapi_version() -> str:
    return _get_version("fastapi")


# Fastcore
def is_fastcore_available() -> bool:
    return is_package_available("fastcore")


def get_fastcore_version() -> str:
    return _get_version("fastcore")


# FastAI
def is_gradio_available() -> bool:
    return is_package_available("gradio")


def get_gradio_version() -> str:
    return _get_version("gradio")


# Graphviz
def is_graphviz_available() -> bool:
    return is_package_available("graphviz")


def get_graphviz_version() -> str:
    return _get_version("graphviz")


# httpx2
def is_httpx2_available() -> bool:
    return is_package_available("httpx2")


def get_httpx2_version() -> str:
    return _get_version("httpx2")


# xet
def is_xet_available() -> bool:
    # since hf_xet is automatically used if available, allow explicit disabling via environment variable
    if constants.HF_HUB_DISABLE_XET:
        return False

    return is_package_available("hf_xet")


def get_xet_version() -> str:
    return _get_version("hf_xet")


# keras
def is_keras_available() -> bool:
    return is_package_available("keras")


def get_keras_version() -> str:
    return _get_version("keras")


# Numpy
def is_numpy_available() -> bool:
    return is_package_available("numpy")


def get_numpy_version() -> str:
    return _get_version("numpy")


# Jinja
def is_jinja_available() -> bool:
    return is_package_available("jinja")


def get_jinja_version() -> str:
    return _get_version("jinja")


# Pillow
def is_pillow_available() -> bool:
    return is_package_available("pillow")


def get_pillow_version() -> str:
    return _get_version("pillow")


# Pydantic
def is_pydantic_available() -> bool:
    if not is_package_available("pydantic"):
        return False
    # For Pydantic, we add an extra check to test whether it is correctly installed or not. If both pydantic 2.x and
    # typing_extensions<=4.5.0 are installed, then pydantic will fail at import time. This should not happen when
    # it is installed with `pip install huggingface_hub[inference]` but it can happen when it is installed manually
    # by the user in an environment that we don't control.
    #
    # Usually we won't need to do this kind of check on optional dependencies. However, pydantic is a special case
    # as it is automatically imported when doing `from huggingface_hub import ...` even if the user doesn't use it.
    #
    # See https://github.com/huggingface/huggingface_hub/pull/1829 for more details.
    try:
        from pydantic import validator  # noqa: F401
    except ImportError:
        # Example: "ImportError: cannot import name 'TypeAliasType' from 'typing_extensions'"
        warnings.warn(
            "Pydantic is installed but cannot be imported. Please check your installation. `huggingface_hub` will "
            "default to not using Pydantic. Error message: '{e}'"
        )
        return False
    return True


def get_pydantic_version() -> str:
    return _get_version("pydantic")


# Pydot
def is_pydot_available() -> bool:
    return is_package_available("pydot")


def get_pydot_version() -> str:
    return _get_version("pydot")


# Tensorboard
def is_tensorboard_available() -> bool:
    return is_package_available("tensorboard")


def get_tensorboard_version() -> str:
    return _get_version("tensorboard")


# Tensorflow
def is_tf_available() -> bool:
    return is_package_available("tensorflow")


def get_tf_version() -> str:
    return _get_version("tensorflow")


# Torch
def is_torch_available() -> bool:
    return is_package_available("torch")


def get_torch_version() -> str:
    return _get_version("torch")


# Safetensors
def is_safetensors_available() -> bool:
    return is_package_available("safetensors")


# Shell-related helpers
try:
    # Set to `True` if script is running in a Google Colab notebook.
    # If running in Google Colab, git credential store is set globally which makes the
    # warning disappear. See https://github.com/huggingface/huggingface_hub/issues/1043
    #
    # Taken from https://stackoverflow.com/a/63519730.
    _is_google_colab = "google.colab" in str(get_ipython())  # type: ignore # noqa: F821
except NameError:
    _is_google_colab = False


def is_notebook() -> bool:
    """Return `True` if code is executed in a notebook (Jupyter, Colab, QTconsole).

    Taken from https://stackoverflow.com/a/39662359.
    Adapted to make it work with Google colab as well.
    """
    try:
        shell_class = get_ipython().__class__  # type: ignore # noqa: F821
        for parent_class in shell_class.__mro__:  # e.g. "is subclass of"
            if parent_class.__name__ == "ZMQInteractiveShell":
                return True  # Jupyter notebook, Google colab or qtconsole
        return False
    except NameError:
        return False  # Probably standard Python interpreter


def is_google_colab() -> bool:
    """Return `True` if code is executed in a Google colab.

    Taken from https://stackoverflow.com/a/63519730.
    """
    return _is_google_colab


def is_colab_enterprise() -> bool:
    """Return `True` if code is executed in a Google Colab Enterprise environment."""
    return os.environ.get("VERTEX_PRODUCT") == "COLAB_ENTERPRISE"


# Check how huggingface_hub has been installed


def installation_method() -> Literal["brew", "hf_installer", "pip", "unknown"]:
    """Return the installation method of the current environment.

    - "hf_installer" if installed via the official installer script
    - "brew" if installed via Homebrew
    - "pip" if pip is available (default fallback for standard Python environments)
    - "unknown" otherwise
    """
    # hf_installer check must come first: the installer creates a venv using the
    # system Python, which may be Homebrew's. Checking brew first would false-positive
    # when the resolved sys.executable points to /opt/homebrew/... inside a venv.
    if _is_hf_installer_installation():
        return "hf_installer"
    if _is_brew_installation():
        return "brew"
    if _is_pip_available():
        return "pip"
    return "unknown"


def _is_brew_installation() -> bool:
    """Check if running from a Homebrew installation.

    Homebrew installs the `hf` formula into a Cellar directory and creates a
    libexec virtualenv at e.g. /opt/homebrew/Cellar/hf/0.30.0/libexec/.
    We check `sys.prefix` (the venv/prefix root) for "/Cellar/hf/" rather
    than checking `sys.executable` — the latter resolves to Homebrew's Python
    (e.g. /opt/homebrew/Cellar/python@3.12/...) even for non-brew installs
    when the system Python happens to come from Homebrew.
    """
    prefix = str(Path(sys.prefix).resolve())
    return "/Cellar/hf/" in prefix


def _is_hf_installer_installation() -> bool:
    """Return `True` if the current environment was set up via the official hf installer script.

    i.e. using one of
        curl -LsSf https://hf.co/cli/install.sh | bash
        powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
    """
    venv = sys.prefix  # points to venv root if active
    marker = Path(venv) / ".hf_installer_marker"
    return marker.exists()


def _is_pip_available() -> bool:
    """Return `True` if pip is importable in the current environment."""
    return importlib.util.find_spec("pip") is not None


def dump_environment_info() -> dict[str, Any]:
    """Dump information about the machine to help debugging issues.

    Similar helper exist in:
    - `datasets` (https://github.com/huggingface/datasets/blob/main/src/datasets/commands/env.py)
    - `diffusers` (https://github.com/huggingface/diffusers/blob/main/src/diffusers/commands/env.py)
    - `transformers` (https://github.com/huggingface/transformers/blob/main/src/transformers/commands/env.py)
    """
    from huggingface_hub import get_token
    from huggingface_hub.utils import is_agent, list_credential_helpers

    token = get_token()

    # Generic machine info
    info: dict[str, Any] = {
        "huggingface_hub version": get_hf_hub_version(),
        "Platform": platform.platform(),
        "Python version": get_python_version(),
    }

    # Interpreter info
    try:
        shell_class = get_ipython().__class__  # type: ignore # noqa: F821
        info["Running in iPython ?"] = "Yes"
        info["iPython shell"] = shell_class.__name__
    except NameError:
        info["Running in iPython ?"] = "No"
    info["Running in notebook ?"] = "Yes" if is_notebook() else "No"
    info["Running in Google Colab ?"] = "Yes" if is_google_colab() else "No"
    info["Running in Google Colab Enterprise ?"] = "Yes" if is_colab_enterprise() else "No"
    # Login info
    info["Token path ?"] = constants.HF_TOKEN_PATH
    info["Has saved token ?"] = token is not None
    if token is not None:
        try:
            from huggingface_hub import whoami

            info["Who am I ?"] = whoami()["name"]
        except Exception:
            pass

    try:
        info["Configured git credential helpers"] = ", ".join(list_credential_helpers())
    except Exception:
        pass

    info["Run by AI agent ?"] = "Yes" if is_agent() else "No"

    # How huggingface_hub has been installed?
    info["Installation method"] = installation_method()

    # Installed dependencies
    info["httpx2"] = get_httpx2_version()
    info["hf_xet"] = get_xet_version()
    info["gradio"] = get_gradio_version()
    info["tensorboard"] = get_tensorboard_version()

    # Environment variables
    info["ENDPOINT"] = constants.ENDPOINT
    info["HF_HUB_CACHE"] = constants.HF_HUB_CACHE
    info["HF_ASSETS_CACHE"] = constants.HF_ASSETS_CACHE
    info["HF_TOKEN_PATH"] = constants.HF_TOKEN_PATH
    info["HF_STORED_TOKENS_PATH"] = constants.HF_STORED_TOKENS_PATH
    info["HF_HUB_OFFLINE"] = constants.HF_HUB_OFFLINE
    info["HF_HUB_DISABLE_TELEMETRY"] = constants.HF_HUB_DISABLE_TELEMETRY
    info["HF_HUB_DISABLE_PROGRESS_BARS"] = constants.HF_HUB_DISABLE_PROGRESS_BARS
    info["HF_HUB_DISABLE_SYMLINKS"] = constants.HF_HUB_DISABLE_SYMLINKS
    info["HF_HUB_DISABLE_SYMLINKS_WARNING"] = constants.HF_HUB_DISABLE_SYMLINKS_WARNING
    info["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = constants.HF_HUB_DISABLE_EXPERIMENTAL_WARNING
    info["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = constants.HF_HUB_DISABLE_IMPLICIT_TOKEN
    info["HF_HUB_DISABLE_XET"] = constants.HF_HUB_DISABLE_XET
    info["HF_HUB_DISABLE_SHARED_BLOBS"] = constants.HF_HUB_DISABLE_SHARED_BLOBS
    info["HF_HUB_ETAG_TIMEOUT"] = constants.HF_HUB_ETAG_TIMEOUT
    info["HF_HUB_DOWNLOAD_TIMEOUT"] = constants.HF_HUB_DOWNLOAD_TIMEOUT
    info["HF_XET_HIGH_PERFORMANCE"] = constants.HF_XET_HIGH_PERFORMANCE

    print("\nCopy-and-paste the text below in your GitHub issue.\n")
    print("\n".join([f"- {prop}: {val}" for prop, val in info.items()]) + "\n")
    return info
<|endoftext|># Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device Code OAuth (RFC 8628) for browser-based login, plus access token refresh.

The flow: the client requests a device code, displays a URL and a short user code, the user
authorizes in a browser, and the client polls ``POST {ENDPOINT}/oauth/token`` until a token is
issued. Access tokens may come with a refresh token, used to renew them transparently (see
``utils/_auth.py::get_token``).

This module is self-contained protocol logic: no printing, no persistence. Interactive flows
live in ``_login.py`` (human/library) and ``cli/auth.py`` (machine-readable event stream).
"""

import time
from collections.abc import Callable
from typing import TypedDict, cast

import httpx2

from .. import constants
from ..errors import DeviceCodeError, OAuthErrorCode
from ._http import get_session, hf_raise_for_status


_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
_REFRESH_TOKEN_GRANT_TYPE = "refresh_token"


class DeviceCodeInfo(TypedDict):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str  # falls back to verification_uri if the server omits it
    interval: int
    expires_in: int


class OAuthTokenResponse(TypedDict, total=False):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str


def request_device_code() -> DeviceCodeInfo:
    """Request a device code from the Hub's OAuth device authorization endpoint.

    The returned dict is normalized: `interval`, `expires_in` and `verification_uri_complete`
    are always set (server values, or sensible defaults).

    Raises:
        [`DeviceCodeError`]: If the request fails.
    """
    try:
        response = get_session().post(
            f"{constants.ENDPOINT}/oauth/device",
            data={"client_id": constants.DEVICE_CODE_OAUTH_CLIENT_ID},
            timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
        )
        hf_raise_for_status(response)
    except httpx2.HTTPError as e:
        raise DeviceCodeError(f"Failed to request device code from {constants.ENDPOINT}/oauth/device: {e}") from e
    info = response.json()
    # `interval` is optional per RFC 8628 (5s is the spec-mandated fallback); `expires_in` is
    # required but defaulted defensively so polling stays bounded if a server omits it.
    info.setdefault("interval", 5)
    info.setdefault("expires_in", 900)
    if not info.get("verification_uri_complete"):
        info["verification_uri_complete"] = info["verification_uri"]
    return cast(DeviceCodeInfo, info)


def poll_device_token(
    device_info: DeviceCodeInfo, *, on_pending: Callable[[], None] | None = None
) -> OAuthTokenResponse:
    """Poll the token endpoint until the user authorizes the device.

    Args:
        device_info (`DeviceCodeInfo`):
            The device authorization response from [`request_device_code`].
        on_pending (`Callable`, *optional*):
            Called after each "authorization pending" response (e.g. to print a progress dot).

    Returns:
        `OAuthTokenResponse`: the full token response: `access_token`, and optionally
        `refresh_token` and `expires_in`.

    Raises:
        [`DeviceCodeError`]: If authorization is denied, the device code expires, or the server
            returns an unexpected OAuth error.
    """
    interval = device_info["interval"]
    deadline = time.monotonic() + device_info["expires_in"]
    while time.monotonic() < deadline:
        # Inconclusive responses (network blip, 5xx, gateway error page, rate limiting) must not
        # abort the login: keep polling until the device code expires (RFC 8628 section 3.5).
        # The deadline bounds the total wait even if the endpoint is genuinely broken.
        data = None
        try:
            response = get_session().post(
                f"{constants.ENDPOINT}/oauth/token",
                data={
                    "grant_type": _DEVICE_CODE_GRANT_TYPE,
                    "device_code": device_info["device_code"],
                    "client_id": constants.DEVICE_CODE_OAUTH_CLIENT_ID,
                },
                timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
            )
            if response.status_code < 500:
                data = response.json()
        except (httpx2.HTTPError, ValueError):
            pass

        if data is not None:
            if "access_token" in data:
                return cast(OAuthTokenResponse, data)

            match data.get("error"):
                case None:
                    pass  # JSON without an OAuth `error` field (proxy error page, ...): transient
                case OAuthErrorCode.AUTHORIZATION_PENDING:
                    if on_pending is not None:
                        on_pending()
                case OAuthErrorCode.SLOW_DOWN:
                    interval += 5
                case OAuthErrorCode.EXPIRED_TOKEN:
                    raise DeviceCodeError(
                        "Device code expired. Please try again.", error_code=OAuthErrorCode.EXPIRED_TOKEN
                    )
                case OAuthErrorCode.ACCESS_DENIED:
                    raise DeviceCodeError(
                        "Authorization was denied. Please try again.", error_code=OAuthErrorCode.ACCESS_DENIED
                    )
                case error:
                    raise DeviceCodeError(
                        f"OAuth error: {error} - {data.get('error_description', '')}", error_code=error
                    )

        time.sleep(interval)

    raise DeviceCodeError("Device code expired (timeout). Please try again.", error_code=OAuthErrorCode.EXPIRED_TOKEN)


def refresh_access_token(refresh_token: str) -> OAuthTokenResponse:
    """Exchange a refresh token for a new access token.

    Returns:
        `OAuthTokenResponse`: the full token response: `access_token`, and optionally a rotated
        `refresh_token` and `expires_in`.

    Raises:
        [`DeviceCodeError`]: If the server rejects the refresh (`error_code="invalid_grant"` when
            the refresh token is expired or revoked) or returns an unexpected response.
    """
    try:
        response = get_session().post(
            f"{constants.ENDPOINT}/oauth/token",
            data={
                "grant_type": _REFRESH_TOKEN_GRANT_TYPE,
                "refresh_token": refresh_token,
                "client_id": constants.DEVICE_CODE_OAUTH_CLIENT_ID,
            },
            # An explicit timeout is critical here: this runs inside `get_token()`, so a hung
            # request would otherwise block every Hub call in the process.
            timeout=constants.HF_HUB_DOWNLOAD_TIMEOUT,
        )
    except httpx2.HTTPError as e:
        raise DeviceCodeError(f"Failed to refresh access token: {e}") from e
    data = _parse_token_response(response)
    if "access_token" in data:
        return cast(OAuthTokenResponse, data)
    error = data.get("error")
    raise DeviceCodeError(
        f"Failed to refresh access token: {error or response.status_code} - {data.get('error_description', '')}",
        error_code=error,
    )


def _parse_token_response(response: httpx2.Response) -> dict:
    try:
        return response.json()
    except ValueError as e:
        raise DeviceCodeError(
            f"Failed to parse response from {constants.ENDPOINT}/oauth/token "
            f"(status {response.status_code}): {response.text[:500]}"
        ) from e
<|endoftext|># Copyright 2022-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains utilities to handle pagination on Huggingface Hub."""

from collections.abc import Iterable
from typing import Any

import httpx2

from . import hf_raise_for_status, http_backoff, logging


logger = logging.get_logger(__name__)


def paginate(path: str, params: dict | list[tuple[str, Any]], headers: dict, timeout: float | None = None) -> Iterable:
    """Fetch a list of models/datasets/spaces and paginate through results.

    This is using the same "Link" header format as GitHub.
    See:
    - https://requests.readthedocs.io/en/latest/api/#requests.Response.links
    - https://docs.github.com/en/rest/guides/traversing-with-pagination#link-header
    """
    r = http_backoff("GET", path, params=params, headers=headers, timeout=timeout)
    hf_raise_for_status(r)
    yield from r.json()

    # Follow pages
    # Next link already contains query params
    next_page = _get_next_page(r)
    while next_page is not None:
        logger.debug(f"Pagination detected. Requesting next page: {next_page}")
        r = http_backoff("GET", next_page, headers=headers, timeout=timeout)
        hf_raise_for_status(r)
        yield from r.json()
        next_page = _get_next_page(r)


def _get_next_page(response: httpx2.Response) -> str | None:
    return response.links.get("next", {}).get("url")
<|endoftext|># Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains utilities to print stuff to the terminal (styling, helpers)."""

import ctypes
import os
import shutil
import sys
from contextlib import contextmanager

from ._detect_agent import is_agent


if sys.platform == "win32":
    import msvcrt
else:
    import select
    import termios
    import tty


class StatusLine:
    """Minimal TTY status line for sync progress (stderr, single-line overwrite)."""

    def __init__(self, enabled: bool = True):
        self._active = enabled and sys.stderr.isatty()

    def update(self, msg: str) -> None:
        if not self._active:
            return
        width = shutil.get_terminal_size().columns
        if len(msg) > width - 1:
            msg = msg[: width - 4] + "..."
        sys.stderr.write(f"\r\033[K\033[90m{msg}\033[0m")
        sys.stderr.flush()

    def done(self, msg: str) -> None:
        if not self._active:
            return
        width = shutil.get_terminal_size().columns
        if len(msg) > width - 1:
            msg = msg[: width - 4] + "..."
        sys.stderr.write(f"\r\033[K\033[90m{msg}\033[0m\n")
        sys.stderr.flush()


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _blue = "\u001b[34m"
    _bold = "\u001b[1m"
    _gray = "\u001b[90m"
    _green = "\u001b[32m"
    _red = "\u001b[31m"
    _reset = "\u001b[0m"
    _underline = "\u001b[4m"
    _yellow = "\u001b[33m"
    _enabled: bool | None = None  # None: fall back to agent detection (callers outside the `hf` CLI)

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        cls._enabled = enabled

    @classmethod
    def blue(cls, s: str) -> str:
        return cls._format(s, cls._blue)

    @classmethod
    def bold(cls, s: str) -> str:
        return cls._format(s, cls._bold)

    @classmethod
    def gray(cls, s: str) -> str:
        return cls._format(s, cls._gray)

    @classmethod
    def green(cls, s: str) -> str:
        return cls._format(s, cls._green)

    @classmethod
    def red(cls, s: str) -> str:
        return cls._format(s, cls._bold + cls._red)

    @classmethod
    def underline(cls, s: str) -> str:
        return cls._format(s, cls._underline)

    @classmethod
    def yellow(cls, s: str) -> str:
        return cls._format(s, cls._yellow)

    @classmethod
    def _format(cls, s: str, code: str) -> str:
        if os.environ.get("NO_COLOR"):
            # See https://no-color.org/
            return s
        enabled = cls._enabled if cls._enabled is not None else not is_agent()
        return f"{code}{s}{cls._reset}" if enabled else s


def select_choice(prompt: str, choices: list[str]) -> int:
    """Single-choice interactive prompt. Returns the index of the selected choice.

    On a TTY, renders an arrow-key menu (Up/Down to move, Enter to confirm, 1-9 to pick
    directly, Ctrl+C to abort). Falls back to a numbered `input()` prompt when raw
    keyboard input is not available. Callers are responsible for not prompting at all in
    non-interactive contexts.
    """
    if not choices:
        raise ValueError("select_choice() requires at least one choice.")
    if _supports_raw_keyboard():
        return _select_with_arrows(prompt, choices)
    return _select_with_numbers(prompt, choices)


def _supports_raw_keyboard() -> bool:
    if not (sys.stdin and sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    if sys.platform == "win32":
        return _enable_windows_vt_processing()
    return True


def _enable_windows_vt_processing() -> bool:
    """Enable VT escape-sequence processing on the Windows console (off by default in legacy
    cmd.exe/PowerShell windows, where the menu would otherwise render as literal `←[K` garbage)."""
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    STD_OUTPUT_HANDLE = -11
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except Exception:
        return False


@contextmanager
def _raw_terminal():
    """Put the terminal in cbreak mode (read keypresses without Enter). No-op on Windows."""
    if sys.platform == "win32":
        yield
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_key() -> str:
    """Read one keypress, normalizing arrow keys to "up"/"down"."""
    if sys.platform == "win32":
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):  # arrow keys come as a two-character sequence
            return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "")
        return char

    # Read the file descriptor directly: `sys.stdin.read(1)` would buffer the whole escape
    # sequence internally, making the fd look empty to `select()` below.
    fd = sys.stdin.fileno()
    char = os.read(fd, 1)
    # Disambiguate a bare Escape from an escape sequence (e.g. "\x1b[A" for Up).
    if char == b"\x1b" and select.select([fd], [], [], 0.05)[0]:
        if os.read(fd, 1) == b"[" and select.select([fd], [], [], 0.05)[0]:
            return {b"A": "up", b"B": "down"}.get(os.read(fd, 1), "")
        return ""
    return char.decode(errors="replace")


def _select_with_arrows(prompt: str, choices: list[str]) -> int:
    selected = 0
    print(ANSI.bold(f"? {prompt}") + ANSI.gray("  [Use arrows, Enter to confirm]"))

    def render() -> None:
        for i, choice in enumerate(choices):
            line = ANSI.green("> ") + ANSI.bold(choice) if i == selected else "  " + choice
            sys.stdout.write(f"\r\x1b[K{line}\n")
        sys.stdout.flush()

    try:
        sys.stdout.write("\x1b[?25l")  # hide cursor
        render()
        with _raw_terminal():
            while True:
                key = _read_key()
                if key == "\x03":
                    # Ctrl+C: POSIX cbreak keeps ISIG so it never reaches here, but on Windows
                    # msvcrt.getwch() returns the raw character instead of raising.
                    raise KeyboardInterrupt
                if key == "up":
                    selected = (selected - 1) % len(choices)
                elif key == "down":
                    selected = (selected + 1) % len(choices)
                elif key.isdecimal() and 1 <= int(key) <= len(choices):
                    selected = int(key) - 1
                    break
                elif key in ("\r", "\n"):
                    break
                else:
                    continue
                sys.stdout.write(f"\x1b[{len(choices)}A")  # move back to the first option line
                render()
    finally:
        sys.stdout.write("\x1b[?25h")  # show cursor
        sys.stdout.flush()

    # Collapse the menu into a single "? prompt answer" summary line, like gh does.
    sys.stdout.write(f"\x1b[{len(choices) + 1}A\r\x1b[J")
    print(ANSI.bold(f"? {prompt} ") + ANSI.green(choices[selected]))
    return selected


def _select_with_numbers(prompt: str, choices: list[str]) -> int:
    print(f"? {prompt}")
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}. {choice}")
    while True:
        raw = input("Choice [1]: ").strip()
        if not raw:
            return 0
        if raw.isdecimal() and 1 <= int(raw) <= len(choices):
            return int(raw) - 1
        print(f"Invalid choice. Enter a number between 1 and {len(choices)}.")


def tabulate(
    rows: list[list[str | int]],
    headers: list[str],
    alignments: dict[str, str] | None = None,
) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    _ALIGN_MAP = {"left": "<", "right": ">"}
    for row in rows:
        if len(row) < len(headers):
            raise IndexError(f"Row has {len(row)} values but expected {len(headers)} (headers: {headers})")
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    col_aligns = [_ALIGN_MAP.get((alignments or {}).get(h, "left"), "<") for h in headers]
    row_format = " ".join(f"{{:{a}{w}}}" for a, w in zip(col_aligns, col_widths))
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return "\n".join(lines)
<|endoftext|># Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""Utility helpers to handle progress bars in `huggingface_hub`.

Example:
    1. Use `huggingface_hub.utils.tqdm` as you would use `tqdm.tqdm` or `tqdm.auto.tqdm`.
    2. To disable progress bars, either use `disable_progress_bars()` helper or set the
       environment variable `HF_HUB_DISABLE_PROGRESS_BARS` to 1.
    3. To re-enable progress bars, use `enable_progress_bars()`.
